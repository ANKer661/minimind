import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .tensor_parallel_mappings import (
    _reduce,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
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
    vocab_parallel: bool = False


################################
# async linear: overlap the communication and the computation in linear layer
################################
class LinearWithAsyncCommunication(torch.autograd.Function):
    """Overlap the communication and the computation in Linear."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        sequence_parallel: bool,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # if use sequence parallel, each rank only save part of the sequence
        # to save activation memory in Attn and MLP
        ctx.save_for_backward(input_, weight)
        ctx.group = group
        ctx.use_bias = bias is not None
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            # TODO: change layout in SP to avoid this
            # input_first = input_.movedim(1, 0).contiguous()
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * dist.get_world_size(group)

            all_gather_input = torch.empty(
                dim_size,
                dtype=input_.dtype,
                device=input_.device,
            )
            dist.all_gather_into_tensor(all_gather_input, input_, group=group)

            # total_input = all_gather_input.movedim(0, 1).contiguous()
            total_input = all_gather_input
        else:
            total_input = input_

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
        input_, weight = ctx.saved_tensors
        group = ctx.group

        if ctx.sequence_parallel:
            # async all-gather to obatin total input
            # input_first = input_.movedim(1, 0).contiguous()
            dim_size = list(input_.size())
            dim_size[0] = dim_size[0] * dist.get_world_size(group)
            all_gather_input = torch.empty(
                dim_size,
                dtype=input_.dtype,
                device=input_.device,
            )
            handle_ag = dist.all_gather_into_tensor(
                all_gather_input, input_, group=group, async_op=True
            )

            # overlap all-gather
            grad_input = grad_output.matmul(weight)

            # async reduce-scatter: collect total grad for sequence shard
            # grad_input_first = grad_input.movedim(1, 0).contiguous()
            dim_size = list(grad_input.size())
            dim_size[0] = dim_size[0] // dist.get_world_size(group)
            sub_grad_input = torch.empty(
                dim_size,
                dtype=grad_input.dtype,
                device=grad_input.device,
                requires_grad=False,
            )
            handle_rs = dist.reduce_scatter_tensor(
                sub_grad_input, grad_input, group=group, async_op=True
            )

            # wait for all-gather communication
            handle_ag.wait()  # type: ignore
            # TODO: change layout in SP to avoid this
            # total_input = all_gather_input.movedim(0, 1).contiguous()  # B, S, D
            total_input = all_gather_input  # S, B, D

            # reshape `total_input` and `grad_input` as 2d
            total_input = total_input.reshape(-1, total_input.size(-1))
            grad_output = grad_output.reshape(-1, grad_output.size(-1))

            # overlap reduce-scatter
            grad_weight = grad_output.t().matmul(total_input)
            grad_bias = grad_output.sum(0) if ctx.use_bias else None

            # wait for reduce-scatter communication
            handle_rs.wait()  # type: ignore
            # sub_grad_input = sub_grad_input.movedim(0, 1).contiguous()

            return sub_grad_input, grad_weight, grad_bias, None, None

        else:
            grad_input = grad_output.matmul(weight)

            # all-reduce grad_input across TP ranks
            handle_ar = dist.all_reduce(grad_input, group=group, async_op=True)

            # reshape `input` and `grad_output` as 2d
            input_ = input_.reshape(-1, input_.size(-1))
            grad_output = grad_output.reshape(-1, grad_output.size(-1))

            # overlap all-reduce
            grad_weight = grad_output.t().matmul(input_)
            grad_bias = grad_output.sum(0) if ctx.use_bias else None

            handle_ar.wait()  # type: ignore

        return grad_input, grad_weight, grad_bias, None, None


def linear_with_async_communication(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sequence_parallel: bool,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return LinearWithAsyncCommunication.apply(input_, weight, bias, sequence_parallel, group)  # type: ignore


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


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocab parallelism.

    Args:
        num_embeddings: total number of embeddings (vocab size)
        embedding_dim: dimension of each embedding vector
        tp_context: tensor parallel context
        reduce_scatter_embeddings: whether to reduce-scatter the embedding output across TP ranks
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_context: TPContext,
        reduce_scatter_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_context = tp_context
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        assert num_embeddings % tp_context.world_size == 0
        self.num_embeddings_per_partition = num_embeddings // tp_context.world_size
        rank = tp_context.rank
        self.vocab_start_index = rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_context.world_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input.masked_fill_(input_mask, 0)
        else:
            masked_input = input_

        output_parallel = self.weight[masked_input]

        if self.tp_context.world_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0.0)  # type: ignore

        if self.reduce_scatter_embeddings:
            # output is reduced-scattered across TP ranks, each rank only has part of the sequence
            # this is used in sequence parallel
            output = reduce_scatter_to_sequence_parallel_region(output_parallel, self.tp_context.group)
        elif self.tp_context.world_size > 1:
            # all-reduce across TP ranks to get the final output
            output = reduce_from_tensor_model_parallel_region(output_parallel, self.tp_context.group)
        else:
            output = output_parallel

        return output


class VocabParallelCrossEntropy(torch.autograd.Function):
    """Cross entropy loss with vocab parallelism."""

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # vocab_parallel_logits: [batch_size, seq_length, vocab_size_per_partition]
        # target: [batch_size, seq_length]
        ctx.group = group

        # to fp32 for numerical stability in cross entropy
        vocab_parallel_logits = vocab_parallel_logits.float()
        # max logits value
        logits_max = vocab_parallel_logits.max(dim=-1).values  # [batch_size, seq_length]
        # all-reduce to get the global max logits value across TP ranks
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)

        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = dist.get_rank(group) * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # in-place substraction
        vocab_parallel_logits -= logits_max.unsqueeze(-1)
        # mask to filter out valid target indices for this rank
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target.masked_fill_(target_mask, 0)

        # target logits, logits[target]
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        target_logits_1d = logits_2d[arange_1d, masked_target_1d]
        target_logits_1d = target_logits_1d.clone().contiguous()
        target_logits = target_logits_1d.view_as(target)
        # mask out invalid target logits
        target_logits.masked_fill_(target_mask, 0.0)

        # sum-exp
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        exp_logits_sum = exp_logits.sum(dim=-1)  # [batch_size, seq_length]

        # All-reduce to get the global sum-exp and target logits
        dist.all_reduce(exp_logits_sum, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(exp_logits_sum) - target_logits  # [batch_size, seq_length]
        # normalize to get probability p, used in backward
        exp_logits.div_(exp_logits_sum.unsqueeze(-1))

        ctx.save_for_backward(exp_logits, masked_target_1d, target_mask)

        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore
        softmax, masked_target_1d, target_mask = ctx.saved_tensors

        grad_input = softmax
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(
            -1, partition_vocab_size
        )  # [batch_size*seq_length, vocab_size_per_partition]

        # 1 for valid ids, 0 for invalid ids
        softmax_update = 1.0 - target_mask.float()  # [batch_size, seq_length]

        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)

        # for valid target ids, subtract 1 from the corresponding softmax value
        # for invalid target ids, - 0 <==> no update to softmax
        grad_2d[arange_1d, masked_target_1d] -= softmax_update.view(-1)

        # pointwise multiply with grad_output
        grad_input.mul_(grad_output.unsqueeze(-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup,
    ignore_index: int = -100,
) -> torch.Tensor:
    loss: torch.Tensor = VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, group
    )  # [batch_size, seq_length], # type: ignore

    ignore_mask = target == ignore_index
    loss = loss.masked_fill(ignore_mask, 0.0)

    valid_tokens = (~ignore_mask).sum().clamp_min(1)

    return loss.sum() / valid_tokens
