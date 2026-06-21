from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn as nn

from .model_minimind import (
    MiniMindConfig,
    RMSNorm,
    precompute_freqs_cis,
)
from .model_tp import TPContext, TPMiniMindBlock
from .tensor_parallel_layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
)
from .tensor_parallel_mappings import (
    _reduce,
    scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region,
)


@dataclass
class PPContext:
    group: dist.ProcessGroup
    world_size: int
    rank: int
    is_first: bool
    is_last: bool
    pipeline_dtype: torch.dtype


class PipelineStageModel(nn.Module):
    def __init__(self, config: MiniMindConfig, pp_context: PPContext, tp_context: TPContext) -> None:
        super().__init__()
        self.config = config
        self.pp_context = pp_context
        self.tp_context = tp_context
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        num_hidden_layers = self.num_hidden_layers
        pp_rank, pp_size = pp_context.rank, pp_context.world_size
        self.layer_start = num_hidden_layers * pp_rank // pp_size
        self.layer_end = num_hidden_layers * (pp_rank + 1) // pp_size
        # temporarily no tied weights between embedding and lm_head
        # if self.config.tie_word_embeddings:

        # if first stage, include the embedding layer and dropout
        if self.pp_context.is_first:
            if self.tp_context.vocab_parallel:
                assert self.config.vocab_size % self.tp_context.world_size == 0, (
                    "vocab size must be divisible by world size for vocab parallel"
                )
                self.embed_tokens = VocabParallelEmbedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.tp_context,
                    reduce_scatter_embeddings=self.tp_context.sequence_parallel,
                )
            else:
                self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.dropout = nn.Dropout(config.dropout)

        # final norm only belongs to the last stage
        if self.pp_context.is_last:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if tp_context.sequence_parallel:
                # in sequence parallel, the final norm only see part of the sequence,
                # so we need to all-reduce the grad in backward
                group = tp_context.group
                self.norm.weight.register_hook(lambda grad: _reduce(grad, group))

        # module layers in each stage
        self.layers = nn.ModuleDict(
            {
                str(layer): TPMiniMindBlock(layer, config, tp_context)
                for layer in range(self.layer_start, self.layer_end)
            }
        )

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,  # type: ignore
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pp_context.is_first:
            # Change external [batch, sequence] input IDs to internal [sequence, batch] layout.
            input_ids = input_tensor.movedim(1, 0).contiguous()
            hidden_states = self.embed_tokens(input_ids)
            if self.tp_context.sequence_parallel and not self.tp_context.vocab_parallel:
                hidden_states = scatter_to_sequence_parallel_region(
                    hidden_states, self.tp_context.group
                )
            hidden_states = self.dropout(hidden_states)
        else:
            hidden_states = input_tensor

        seq_length = hidden_states.size(0)
        if self.tp_context.sequence_parallel:
            seq_length *= self.tp_context.world_size

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:  # type: ignore
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
            self.freqs_cos[:seq_length],  # type: ignore
            self.freqs_sin[:seq_length],  # type: ignore
        )
        for layer in self.layers.values():
            hidden_states, _ = layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
            )

        if self.pp_context.is_last:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class PipelineStage(nn.Module):
    def __init__(self, config: MiniMindConfig, pp_context: PPContext, tp_context: TPContext) -> None:
        super().__init__()
        self.config = config
        self.pp_context = pp_context
        self.tp_context = tp_context
        self.model = PipelineStageModel(config, pp_context, tp_context)

        # if last stage, include the LM head
        if self.pp_context.is_last:
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

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_tensor, attention_mask)

        if not self.pp_context.is_last:
            return hidden_states

        if self.tp_context.sequence_parallel and not self.tp_context.vocab_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, self.tp_context.group, tensor_parallel_output_grad=False
            )
        logits = self.lm_head(hidden_states)
        # change layout back to [bsz, seq_len, vocab_size] for compatibility
        logits = logits.movedim(1, 0).contiguous()
        return logits
