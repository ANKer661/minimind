from typing import Any

import torch
import torch.distributed as dist
from .tensor_parallel_mappings import gather_from_sequence_parallel_region


def _all_to_all(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-to-all the input across the context parallel group."""
    assert group is not None, "Context parallel group is not initialized."

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    if not input_.is_contiguous():
        input_ = input_.contiguous()

    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=group)
    return output


class A2ASeqToHead(torch.autograd.Function):
    """Convert a sequence-sharded tensor to a head-sharded tensor."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        # [S / cp, B, heads, head_dim] -> [S, B, heads / cp, head_dim]
        cp_size = dist.get_world_size(group)
        _, bsz, num_heads, head_dim = input_.shape
        num_local_heads = num_heads // cp_size
        input_ = (
            input_.view(-1, bsz, cp_size, num_local_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )

        output = _all_to_all(input_, group=group)
        return output.view(-1, bsz, num_local_heads, head_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        cp_size = dist.get_world_size(ctx.group)
        _, bsz, num_local_heads, head_dim = grad_output.shape
        grad_output = grad_output.view(cp_size, -1, bsz, num_local_heads, head_dim)

        grad_input = _all_to_all(grad_output, group=ctx.group)
        grad_input = grad_input.permute(1, 2, 0, 3, 4).reshape(
            -1, bsz, cp_size * num_local_heads, head_dim
        )
        return grad_input, None


class A2AHeadToSeq(torch.autograd.Function):
    """Convert a head-sharded tensor to a sequence-sharded tensor."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        # [S, B, heads / cp, head_dim] -> [S / cp, B, heads, head_dim]
        cp_size = dist.get_world_size(group)
        _, bsz, num_local_heads, head_dim = input_.shape
        input_ = input_.reshape(cp_size, -1, bsz, num_local_heads, head_dim)

        output = _all_to_all(input_, group=group)
        return output.permute(1, 2, 0, 3, 4).reshape(-1, bsz, cp_size * num_local_heads, head_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        cp_size = dist.get_world_size(ctx.group)
        _, bsz, num_heads, head_dim = grad_output.shape
        num_local_heads = num_heads // cp_size
        grad_output = (
            grad_output.view(-1, bsz, cp_size, num_local_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )

        grad_input = _all_to_all(grad_output, group=ctx.group)
        return grad_input.view(-1, bsz, num_local_heads, head_dim), None


def all_to_all_seq_to_head(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return A2ASeqToHead.apply(input_, group)  # type: ignore


def all_to_all_head_to_seq(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return A2AHeadToSeq.apply(input_, group)  # type: ignore


def gather_along_sequence_dim(
    tensor: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Wrapper for context parallel, equivalent to gather_from_sequence_parallel_region but with a more descriptive name."""
    return gather_from_sequence_parallel_region(tensor, group, tensor_parallel_output_grad=True)
