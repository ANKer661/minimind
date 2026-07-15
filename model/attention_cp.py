import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Literal

from .model_minimind import (
    MiniMindConfig,
    RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .tensor_parallel_layers import ColumnParallelLinear, RowParallelLinear, TPContext
from .tensor_parallel_mappings import _reduce
from .context_parallel_mappings import (
    gather_along_sequence_dim,
    all_to_all_seq_to_head,
    all_to_all_head_to_seq,
)


@dataclass
class CPContext:
    world_size: int
    rank: int
    group: torch.distributed.ProcessGroup
    comm_type: Literal["all_gather", "a2a"]


def cp_sequence_range(global_seq_len: int, cp_context: CPContext) -> tuple[int, int]:
    """
    Returns the start and end indices of the sequence range for the current CP rank.
    """
    if global_seq_len % cp_context.world_size != 0:
        raise ValueError(
            f"seq_len {global_seq_len} must be divisible by cp_size {cp_context.world_size}"
        )

    local_seq_len = global_seq_len // cp_context.world_size
    start = cp_context.rank * local_seq_len
    return start, start + local_seq_len


def make_cp_causal_mask(
    q_len: int,
    kv_len: int,
    cp_context: CPContext,
    device: torch.device,
) -> torch.Tensor:
    q_start = cp_context.rank * q_len
    q_positions = torch.arange(q_start, q_start + q_len, device=device)
    kv_positions = torch.arange(kv_len, device=device)
    return kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)


def all_gather_cp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_context: CPContext,
    n_rep: int,
    attn_dropout: nn.Dropout,
    flash: bool,
    is_causal: bool,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Attention with local Q and all-gathered K/V.

    Inputs and output use [S_local, B, heads, head_dim].
    """
    q_len = q.size(0)
    cp_group = cp_context.group
    k = gather_along_sequence_dim(k, cp_group)
    v = gather_along_sequence_dim(v, cp_group)
    kv_len = k.size(0)

    q = q.permute(1, 2, 0, 3)  # [seq_len, bsz, heads, head_dim] -> [bsz, heads, seq_len, head_dim]
    k = repeat_kv(k, n_rep).permute(1, 2, 0, 3)
    v = repeat_kv(v, n_rep).permute(1, 2, 0, 3)

    # For cp_size=1, query/key share the same local sequence coordinates.
    # For real CP, query positions are offset by cp rank, so we need an explicit mask.
    use_local_causal_mask = cp_context.world_size == 1 and attention_mask is None

    causal_mask = None
    if is_causal and not use_local_causal_mask:
        causal_mask = make_cp_causal_mask(q_len, kv_len, cp_context, q.device)

    if attention_mask is not None and not torch.all(attention_mask == 1):
        if attention_mask.size(-1) != kv_len:
            raise NotImplementedError("CP attention_mask must cover the gathered K/V length.")
        key_mask = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
        causal_mask = (
            key_mask if causal_mask is None else causal_mask.view(1, 1, q_len, kv_len) & key_mask
        )

    if flash and q_len > 1:
        if use_local_causal_mask:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=attn_dropout.p if attn_dropout.training else 0.0,
                is_causal=True,
            )
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=attn_dropout.p if attn_dropout.training else 0.0,
                is_causal=False,
            )

        # [bsz, heads, seq_len, head_dim] -> [seq_len, bsz, heads, head_dim]
        return output.permute(2, 0, 1, 3)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if use_local_causal_mask:
        if is_causal:
            scores[:, :, :, -q_len:] += torch.full(
                (q_len, q_len), float("-inf"), device=scores.device
            ).triu(1)
    elif causal_mask is not None:
        if causal_mask.dim() == 2:
            causal_mask = causal_mask.view(1, 1, q_len, kv_len)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)
    output = attn_dropout(attn_weights) @ v
    return output.permute(2, 0, 1, 3)


def a2a_cp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_context: CPContext,
    n_rep: int,
    attn_dropout: nn.Dropout,
    flash: bool,
    is_causal: bool,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    # a2a, q, k, v (S / cp, B, heads, head_dim) -> (S, B, heads / cp, head_dim)

    num_q_heads = q.size(2)
    num_kv_heads = k.size(2)
    assert num_q_heads % cp_context.world_size == 0 and num_kv_heads % cp_context.world_size == 0, (
        f"num_q_heads {num_q_heads} and num_kv_heads {num_kv_heads} must be divisible by cp_size {cp_context.world_size}"
    )

    xq = all_to_all_seq_to_head(q, cp_context.group)  # [S, B, heads / cp, head_dim]
    xk = all_to_all_seq_to_head(k, cp_context.group)
    xv = all_to_all_seq_to_head(v, cp_context.group)
    xk = repeat_kv(xk, n_rep)  # [S, B, heads / cp * n_rep, head_dim]
    xv = repeat_kv(xv, n_rep)  # [S, B, heads / cp * n_rep, head_dim]

    seq_len = xq.size(0)
    head_dim = xq.size(-1)
    xq = xq.permute(1, 2, 0, 3)  # [S, B, heads / cp, head_dim] -> [B, heads / cp, S, head_dim]
    xk = xk.permute(1, 2, 0, 3)
    xv = xv.permute(1, 2, 0, 3)

    # regular mha, result for local heads
    if flash and (seq_len > 1) and (attention_mask is None or torch.all(attention_mask == 1)):
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            dropout_p=attn_dropout.p if attn_dropout.training else 0.0,
            is_causal=is_causal,
        )  # [bsz, n_local_heads, S, head_dim]
    else:
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(head_dim)
        if is_causal:
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device
            ).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = attn_dropout(attn_weights) @ xv  # [bsz, n_local_heads, S, head_dim]

    # a2a, get local sequence results
    output = output.permute(2, 0, 1, 3)  # [S, B, heads / cp, head_dim]
    output = all_to_all_head_to_seq(output, cp_context.group)  # [S / cp, B, heads, head_dim]

    return output


class CPAttention(nn.Module):
    def __init__(self, config: MiniMindConfig, tp_context: TPContext, cp_context: CPContext) -> None:
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
        self.cp_context = cp_context

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
        tp_group = tp_context.group
        self.q_norm.weight.register_hook(lambda grad: _reduce(grad, tp_group))
        self.k_norm.weight.register_hook(lambda grad: _reduce(grad, tp_group))
        ########################################################
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None
    ):
        # assume position_embeddings are local pos embed
        # input shape: [seq_len / cp_size, bsz, hidden_dim]
        if past_key_value is not None or use_cache:
            raise NotImplementedError("CPAttention does not support KV cache.")

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # get shape after linear projection
        # because the input shape may be different if SP is used
        seq_len, bsz, _ = xq.shape
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
        if self.cp_context.comm_type == "all_gather":
            output = all_gather_cp_attention(
                xq,
                xk,
                xv,
                self.cp_context,
                self.n_rep,
                self.attn_dropout,
                self.flash,
                self.is_causal,
                attention_mask,
            )
        elif self.cp_context.comm_type == "a2a":
            output = a2a_cp_attention(
                xq,
                xk,
                xv,
                self.cp_context,
                self.n_rep,
                self.attn_dropout,
                self.flash,
                self.is_causal,
                attention_mask,
            )
        else:
            raise NotImplementedError(f"Unsupported CP comm type: {self.cp_context.comm_type}")

        output = output.reshape(seq_len, bsz, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
