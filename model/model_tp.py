from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_minimind import (
    MiniMindConfig,
    RMSNorm,
    apply_rotary_pos_emb,
    precompute_freqs_cis,
    repeat_kv,
)

from .tensor_parallel_mappings import (
    _reduce,
    reduce_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)


################################
# tensor parallel context
################################
@dataclass
class TPContext:
    group: dist.ProcessGroup
    world_size: int
    rank: int
    sequence_parallel: bool = False
    async_communication: bool = False


_COLUMN_PARALLEL_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
)

_ROW_PARALLEL_SUFFIXES = (
    "o_proj.weight",
    "down_proj.weight",
)


def shard_state_dict_for_tp(
    state_dict: Mapping[str, torch.Tensor],
    tp_context: TPContext,
) -> dict[str, torch.Tensor]:
    """Shard a full MiniMind state dict for the current TP rank."""
    tp_state_dict = {}

    for key, value in state_dict.items():
        if key.endswith(_COLUMN_PARALLEL_SUFFIXES):
            value = value.chunk(tp_context.world_size, dim=0)[tp_context.rank]
        elif key.endswith(_ROW_PARALLEL_SUFFIXES):
            value = value.chunk(tp_context.world_size, dim=1)[tp_context.rank]
        # for simplicity, other param are now replicated across all TP ranks
        tp_state_dict[key] = value.contiguous()

    return tp_state_dict


class LinearWithAsyncCommunication(torch.autograd.Function):
    """Overlap the communication and the computation in Linear."""

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        sequence_parallel: bool,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # if use sequence parallel, each rank only save part of the sequence
        # to save activation memory in Attn and MLP
        ctx.save_for_backward(input, weight)
        ctx.group = group
        ctx.use_bias = bias is not None
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            # TODO: change layout in SP to avoid this
            input_first = input.movedim(1, 0).contiguous()
            dim_size = list(input_first.size())
            dim_size[0] = dim_size[0] * dist.get_world_size(group)

            all_gather_input = torch.empty(
                dim_size,
                dtype=input_first.dtype,
                device=input_first.device,
            )
            dist.all_gather_into_tensor(all_gather_input, input_first, group=group)

            total_input = all_gather_input.movedim(0, 1).contiguous()
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
        input, weight = ctx.saved_tensors
        group = ctx.group

        if ctx.sequence_parallel:
            # async all-gather to obatin total input
            input_first = input.movedim(1, 0).contiguous()
            dim_size = list(input_first.size())
            dim_size[0] = dim_size[0] * dist.get_world_size(group)
            all_gather_input = torch.empty(
                dim_size,
                dtype=input_first.dtype,
                device=input_first.device,
            )
            handle_ag = dist.all_gather_into_tensor(
                all_gather_input, input_first, group=group, async_op=True
            )

            # overlap all-gather
            grad_input = grad_output.matmul(weight)

            # async reduce-scatter: collect total grad for sequence shard
            grad_input_first = grad_input.movedim(1, 0).contiguous()
            dim_size = list(grad_input_first.size())
            dim_size[0] = dim_size[0] // dist.get_world_size(group)
            sub_grad_input = torch.empty(
                dim_size,
                dtype=grad_input_first.dtype,
                device=grad_input_first.device,
                requires_grad=False,
            )
            handle_rs = dist.reduce_scatter_tensor(
                sub_grad_input, grad_input_first, group=group, async_op=True
            )

            # wait for all-gather communication
            handle_ag.wait()  # type: ignore
            # TODO: change layout in SP to avoid this
            total_input = all_gather_input.movedim(0, 1).contiguous()  # B, S, D

            # reshape `total_input` and `grad_input` as 2d
            total_input = total_input.reshape(-1, total_input.size(-1))
            grad_output = grad_output.reshape(-1, grad_output.size(-1))

            # overlap reduce-scatter
            grad_weight = grad_output.t().matmul(total_input)
            grad_bias = grad_output.sum(0) if ctx.use_bias else None

            # wait for reduce-scatter communication
            handle_rs.wait()  # type: ignore
            sub_grad_input = sub_grad_input.movedim(0, 1).contiguous()

            return sub_grad_input, grad_weight, grad_bias, None, None

        else:
            grad_input = grad_output.matmul(weight)

            # all-reduce grad_input across TP ranks
            handle_ar = dist.all_reduce(grad_input, group=group, async_op=True)

            # reshape `input` and `grad_output` as 2d
            input = input.reshape(-1, input.size(-1))
            grad_output = grad_output.reshape(-1, grad_output.size(-1))

            # overlap all-reduce
            grad_weight = grad_output.t().matmul(input)
            grad_bias = grad_output.sum(0) if ctx.use_bias else None

            handle_ar.wait()  # type: ignore

        return grad_input, grad_weight, grad_bias, None, None


def linear_with_async_communication(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sequence_parallel: bool,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return LinearWithAsyncCommunication.apply(input, weight, bias, sequence_parallel, group)  # type: ignore


################################
# tensor parallel linear layers
################################
class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    the second dimension as A = [A1, A2, ..., An].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_context: TPContext,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_context = tp_context
        assert output_size % tp_context.world_size == 0
        self.output_size_per_partition = output_size // tp_context.world_size
        # we store weight as [out, in] to use F.linear, here weight is split along dim 0
        # in docstring we use the math notation, weight is [in, out]
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_length, input_size]
        # weight: [output_size_per_partition, input_size]
        # bias: [output_size_per_partition]
        if self.tp_context.async_communication:
            return linear_with_async_communication(
                x, self.weight, self.bias, self.tp_context.sequence_parallel, self.tp_context.group
            )
        else:
            if self.tp_context.sequence_parallel:
                # in sequence parallel, we need to gather the input across TP ranks
                x_parallel = gather_from_sequence_parallel_region(x, self.tp_context.group)
            else:
                x_parallel = copy_to_tensor_model_parallel_region(x, self.tp_context.group)

            return F.linear(x_parallel, self.weight, self.bias)

    def reset_parameters(self) -> None:
        # initialize weight and bias in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    the first dimension and X along the last dimension.
    A = transpose([A1, A2, ..., An]) and X = [X1, X2, ..., Xn].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_context: TPContext,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_context = tp_context
        assert input_size % tp_context.world_size == 0
        self.input_size_per_partition = input_size // tp_context.world_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.empty(self.output_size)) if bias else None
        if tp_context.sequence_parallel and self.bias is not None:
            # in sequence parallel, bias is added on each sequence shard
            # so we need to all-reduce the grad in backward
            group = tp_context.group
            self.bias.register_hook(lambda grad: _reduce(grad, group))

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_length, input_size_per_partition]
        # weight: [output_size, input_size_per_partition]
        # bias: [output_size]
        x_parallel = F.linear(x, self.weight, None)

        if self.tp_context.sequence_parallel:
            # in sequence parallel, we need to reduce-scatter the output across TP ranks
            x = reduce_scatter_to_sequence_parallel_region(x_parallel, self.tp_context.group)
        else:
            x = reduce_from_tensor_model_parallel_region(x_parallel, self.tp_context.group)
        # add bias after all-reduce or reduce-scatter
        x = x + self.bias if self.bias is not None else x

        return x

    def reset_parameters(self) -> None:
        # initialize weight and bias in the same way as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)


################################
# minimind tensor parallel layers
################################
class TPFeedForward(nn.Module):
    def __init__(
        self,
        config: MiniMindConfig,
        tp_context: TPContext,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.tp_context = tp_context
        ######################################################
        # replace proj to corresponding parallel linear layers
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, intermediate_size, tp_context, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, intermediate_size, tp_context, bias=False
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size, tp_context, bias=False
        )
        ######################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TPAttention(nn.Module):
    def __init__(self, config: MiniMindConfig, tp_context: TPContext) -> None:
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        ######################################################
        # now these local dim need to be divided by world size
        assert config.num_attention_heads % tp_context.world_size == 0
        assert self.num_key_value_heads % tp_context.world_size == 0
        self.n_local_heads = config.num_attention_heads // tp_context.world_size
        self.n_local_kv_heads = self.num_key_value_heads // tp_context.world_size
        ######################################################

        assert self.n_local_heads % self.n_local_kv_heads == 0
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.tp_context = tp_context

        ######################################################
        # replace proj to corresponding parallel linear layers
        self.q_proj = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, tp_context, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, tp_context, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, tp_context, bias=False
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, tp_context, bias=False
        )
        #######################################################
        #######################################################
        # RMSNorm only see local heads, and we need to all-reduce the grad in backward
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        group = tp_context.group
        self.q_norm.weight.register_hook(lambda grad: _reduce(grad, group))
        self.k_norm.weight.register_hook(lambda grad: _reduce(grad, group))
        ########################################################
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None
    ):
        if past_key_value is not None or use_cache:
            raise NotImplementedError("TPAttention does not support KV cache.")

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        bsz, seq_len, _ = xq.shape  # get shape after linear projection
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class TPMiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig, tp_context: TPContext) -> None:
        super().__init__()
        self.self_attn = TPAttention(config, tp_context)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TPFeedForward(config, tp_context)
        self.tp_context = tp_context
        if tp_context.sequence_parallel:
            # in sequence_parallel, input_layernorm and post_attention_layernorm
            # only see part of the sequence, so we need to all-reduce the grad in backward
            group = tp_context.group
            self.input_layernorm.weight.register_hook(lambda grad: _reduce(grad, group))
            self.post_attention_layernorm.weight.register_hook(lambda grad: _reduce(grad, group))
        if config.use_moe:
            raise NotImplementedError("MoE is not implemented in TP version yet.")

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class TPMiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig, tp_context: TPContext) -> None:
        super().__init__()
        self.config = config
        self.tp_context = tp_context
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TPMiniMindBlock(l, config, tp_context) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        if tp_context.sequence_parallel:
            # in sequence parallel, the final norm only see part of the sequence,
            # so we need to all-reduce the grad in backward
            group = tp_context.group
            self.norm.weight.register_hook(lambda grad: _reduce(grad, group))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        if past_key_values is not None or use_cache:
            raise NotImplementedError("TP does not support KV cache.")

        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = 0
        hidden_states = self.embed_tokens(input_ids)
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos, self.freqs_sin = (
                freqs_cos.to(hidden_states.device),
                freqs_sin.to(hidden_states.device),
            )
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []

        if self.tp_context.sequence_parallel:
            # in sequence parallel, each layer accept part of the sequence
            # so we scatter the hidden_states at the beginning
            hidden_states = scatter_to_sequence_parallel_region(hidden_states, self.tp_context.group)

        hidden_states = self.dropout(hidden_states)
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # for compatibility
        aux_loss = hidden_states.new_zeros(1).squeeze()
        return hidden_states, presents, aux_loss


class TPMiniMindForCausalLM(PreTrainedModel):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, tp_context: TPContext, config: MiniMindConfig | None = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.tp_context = tp_context
        self.model = TPMiniMindModel(self.config, tp_context)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        # if sequence parallel, each TP rank only has part of the sequence
        # so we need to gather the hidden_states before lm_head
        if self.tp_context.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.tp_context.group, tensor_parallel_output_grad=False
            )

        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
