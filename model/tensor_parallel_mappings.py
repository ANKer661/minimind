import torch
import torch.distributed as dist
from typing import Any


def _reduce(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-reduce across the model parallel group."""
    assert group is not None, "Model parallel group is not initialized."

    # Skip if only 1 GPU
    if dist.get_world_size(group) == 1:
        return input_

    if not input_.is_contiguous():
        input_ = input_.contiguous()

    dist.all_reduce(input_, group=group)
    return input_


def _split_along_sequence_dim(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Split the input across the sequence parallel group."""
    assert group is not None, "Sequence parallel group is not initialized."

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return input_

    # TODO: change the tensor layout in SP to avoid this
    input_first = input_.movedim(1, 0).contiguous()

    seq_length = input_first.size(0)
    assert seq_length % world_size == 0, "The sequence length must be divisible by the world size."
    local_seq_length = seq_length // world_size
    rank = dist.get_rank(group)
    dim_offset = rank * local_seq_length

    output = input_first[dim_offset : dim_offset + local_seq_length].contiguous()

    return output.movedim(0, 1).contiguous()


def _gather_along_sequence_dim(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather the input and concatenate along the sequence parallel dimension."""
    assert group is not None, "Sequence parallel group is not initialized."

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    # TODO: change the tensor layout in SP to avoid this
    input_first = input_.movedim(1, 0).contiguous()

    output_size = list(input_first.size())
    # change the sequence length to the total sequence length
    output_size[0] = output_size[0] * world_size
    output_first = torch.empty(
        output_size,
        dtype=input_.dtype,
        device=torch.cuda.current_device(),
    )

    torch.distributed.all_gather_into_tensor(output_first, input_first, group=group)

    return output_first.movedim(0, 1).contiguous()


def _reduce_scatter_along_sequence_dim(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Reduce-scatter the input across the sequence parallel group."""
    assert group is not None, "Sequence parallel group is not initialized."

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    # TODO: change the tensor layout in SP to avoid this
    input_first = input_.movedim(1, 0).contiguous()
    seq_length = input_first.size(0)

    assert seq_length % world_size == 0, "The sequence length must be divisible by the world size."
    local_seq_length = seq_length // world_size

    output_size = list(input_first.size())
    output_size[1] = local_seq_length
    output = torch.empty(
        output_size,
        dtype=input_first.dtype,
        device=torch.cuda.current_device(),
    )

    torch.distributed.reduce_scatter_tensor(output, input_first, group=group)

    return output.movedim(0, 1).contiguous()


####################################
# two primitives for tensor parallel
####################################
class CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        return _reduce(grad_output, ctx.group), None


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _reduce(input_, ctx.group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        return grad_output, None


####################################
# primitives for sequence parallel
####################################
class ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Scatter the input to the sequence parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _split_along_sequence_dim(input_, ctx.group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        return _gather_along_sequence_dim(grad_output, ctx.group), None


class ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce-scatter the input to the sequence parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _reduce_scatter_along_sequence_dim(input_, ctx.group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        return _gather_along_sequence_dim(grad_output, ctx.group), None


class GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from the sequence parallel region."""

    @staticmethod
    def forward(
        ctx: Any,
        input_: torch.Tensor,
        group: dist.ProcessGroup,
        tensor_parallel_output_grad=True,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_sequence_dim(input_, ctx.group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        if tensor_parallel_output_grad:
            return _reduce_scatter_along_sequence_dim(grad_output, ctx.group), None, None
        else:
            return _split_along_sequence_dim(grad_output, ctx.group), None, None


####################################
# wrappers for tensor parallel
####################################
def copy_to_tensor_model_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Wrapper for autograd function: forward copy, backward all-reduce."""
    return CopyToModelParallelRegion.apply(input_, group)  # type: ignore


def reduce_from_tensor_model_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Wrapper for autograd function: forward all-reduce, backward copy."""
    return ReduceFromModelParallelRegion.apply(input_, group)  # type: ignore


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Wrapper for autograd function: forward scatter, backward gather."""
    return ScatterToSequenceParallelRegion.apply(input_, group)  # type: ignore


def reduce_scatter_to_sequence_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Wrapper for autograd function: forward reduce-scatter, backward gather."""
    return ReduceScatterToSequenceParallelRegion.apply(input_, group)  # type: ignore


def gather_from_sequence_parallel_region(
    input_: torch.Tensor, group: dist.ProcessGroup, tensor_parallel_output_grad=True
) -> torch.Tensor:
    """Wrapper for autograd function: forward gather, backward reduce-scatter or scatter."""
    return GatherFromSequenceParallelRegion.apply(input_, group, tensor_parallel_output_grad)  # type: ignore
