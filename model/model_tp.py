from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_cp import CPAttention, CPContext, cp_sequence_range
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
from .tensor_parallel_layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    TPContext,
    vocab_parallel_cross_entropy,
)
from .tensor_parallel_mappings import (
    _reduce,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

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

_VOCAB_PARALLEL_SUFFIXES = (
    "model.embed_tokens.weight",
    "lm_head.weight",
)


def shard_state_dict_for_tp(
    state_dict: Mapping[str, torch.Tensor],
    tp_context: TPContext,
) -> dict[str, torch.Tensor]:
    """Shard a full MiniMind state dict for the current TP rank."""
    tp_state_dict = {}

    for key, value in state_dict.items():
        if key.endswith(_VOCAB_PARALLEL_SUFFIXES):
            if tp_context.vocab_parallel:
                value = value.chunk(tp_context.world_size, dim=0)[tp_context.rank]
        elif key.endswith(_COLUMN_PARALLEL_SUFFIXES):
            value = value.chunk(tp_context.world_size, dim=0)[tp_context.rank]
        elif key.endswith(_ROW_PARALLEL_SUFFIXES):
            value = value.chunk(tp_context.world_size, dim=1)[tp_context.rank]
        # for simplicity, other param are now replicated across all TP ranks
        tp_state_dict[key] = value.contiguous()

    return tp_state_dict


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
        seq_len, bsz, _ = xq.shape  # get shape after linear projection
        xq = xq.view(seq_len, bsz, self.n_local_heads, self.head_dim)
        xk = xk.view(seq_len, bsz, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(seq_len, bsz, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings  # seq_len, *
        # unsqueeze to match the shape of xq and xk
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:  # temporary ignore, problematic shape
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.permute(1, 2, 0, 3),
            repeat_kv(xk, self.n_rep).permute(1, 2, 0, 3),
            repeat_kv(xv, self.n_rep).permute(
                1, 2, 0, 3
            ),  # in repeat_kv, assume [bsz, seq_len, ...], but it is compatible with [seq_len, bsz, ...]
        )  # [bsz, n_local_heads, seq_len, head_dim]
        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal
            )  # [bsz, n_local_heads, seq_len, head_dim]
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = output.permute(2, 0, 1, 3).reshape(seq_len, bsz, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class TPMiniMindBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: MiniMindConfig,
        tp_context: TPContext,
        cp_context: CPContext | None,
    ) -> None:
        super().__init__()
        if cp_context is None:
            self.self_attn = TPAttention(config, tp_context)
        else:
            self.self_attn = CPAttention(config, tp_context, cp_context)
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
    def __init__(
        self, config: MiniMindConfig, tp_context: TPContext, cp_context: CPContext | None
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_context = tp_context
        self.cp_context = cp_context
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        if tp_context.vocab_parallel:
            assert config.vocab_size % tp_context.world_size == 0, (
                "vocab size must be divisible by world size for vocab parallel"
            )
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                tp_context,
                reduce_scatter_embeddings=tp_context.sequence_parallel,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                TPMiniMindBlock(layer, config, tp_context, cp_context)
                for layer in range(self.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,  # type: ignore
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

        seq_length, batch_size = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        if self.cp_context is not None:
            start_pos, end_pos = cp_sequence_range(seq_length, self.cp_context)
            input_ids = input_ids[start_pos:end_pos]
        else:
            start_pos, end_pos = 0, seq_length

        hidden_states = self.embed_tokens(input_ids)
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,  # type: ignore
            )
            self.freqs_cos, self.freqs_sin = (
                freqs_cos.to(hidden_states.device),
                freqs_sin.to(hidden_states.device),
            )

        position_embeddings = (
            self.freqs_cos[start_pos:end_pos],
            self.freqs_sin[start_pos:end_pos],
        )
        presents = []

        if self.tp_context.sequence_parallel and not self.tp_context.vocab_parallel:
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

    def __init__(
        self,
        tp_context: TPContext,
        config: MiniMindConfig | None = None,
        cp_context: CPContext | None = None,
    ) -> None:
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.tp_context = tp_context
        self.cp_context = cp_context
        self.model = TPMiniMindModel(self.config, tp_context, cp_context)
        if tp_context.vocab_parallel:
            # lm_head reuse Column Parallel Linear
            self.lm_head = ColumnParallelLinear(
                self.config.hidden_size, self.config.vocab_size, tp_context, bias=False
            )
        else:
            # if no vocab parallel
            # the lm_head computation is duplicated across TP ranks
            # so we use regular Linear
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight  # type: ignore
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
        # change layout to [seq_len, bsz, hidden_size] for TP/SP
        input_ids = input_ids.movedim(1, 0).contiguous()

        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        # if sequence parallel, each TP rank only has part of the sequence
        # so we need to gather the hidden_states before lm_head
        # if vocab parallel is enabled, the inputs will be
        # gathered in lm_head, so we don't need to gather here
        if self.tp_context.sequence_parallel and not self.tp_context.vocab_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.tp_context.group, tensor_parallel_output_grad=False
            )
        logits = self.lm_head(hidden_states[slice_indices, :, :])
        # change layout back to [bsz, seq_len, vocab_size] for compatibility
        logits = logits.movedim(1, 0).contiguous()
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            if self.tp_context.vocab_parallel:
                loss = vocab_parallel_cross_entropy(x, y, self.tp_context.group, ignore_index=-100)
            else:
                loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
