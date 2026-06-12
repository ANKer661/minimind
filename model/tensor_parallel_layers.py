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

################################
# async linear: overlap the communication and the computation in linear layer
################################
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