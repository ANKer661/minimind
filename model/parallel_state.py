from collections.abc import Iterable
from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class ProcessGroupCollection:
    tp: dist.ProcessGroup
    cp: dist.ProcessGroup | None
    pp: dist.ProcessGroup
    embd: dist.ProcessGroup | None


def _global_rank(
    pp_rank: int,
    cp_rank: int,
    tp_rank: int,
    cp_size: int,
    tp_size: int,
) -> int:
    return (pp_rank * cp_size + cp_rank) * tp_size + tp_rank


def _create_group_for_rank(
    rank_lists: Iterable[list[int]],
    rank: int,
) -> dist.ProcessGroup | None:
    result = None
    for ranks in rank_lists:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            result = group
    return result


def create_process_groups(
    pp_size: int,
    cp_size: int,
    tp_size: int,
    cp_enabled: bool,
) -> ProcessGroupCollection:
    """Build process groups for the global rank layout [PP, CP, TP]."""
    rank = dist.get_rank()

    tp_group = _create_group_for_rank(
        (
            [
                _global_rank(pp_rank, cp_rank, tp_rank, cp_size, tp_size)
                for tp_rank in range(tp_size)
            ]
            for pp_rank in range(pp_size)
            for cp_rank in range(cp_size)
        ),
        rank,
    )

    cp_group = None
    if cp_enabled:
        cp_group = _create_group_for_rank(
            (
                [
                    _global_rank(pp_rank, cp_rank, tp_rank, cp_size, tp_size)
                    for cp_rank in range(cp_size)
                ]
                for pp_rank in range(pp_size)
                for tp_rank in range(tp_size)
            ),
            rank,
        )

    pp_group = _create_group_for_rank(
        (
            [
                _global_rank(pp_rank, cp_rank, tp_rank, cp_size, tp_size)
                for pp_rank in range(pp_size)
            ]
            for cp_rank in range(cp_size)
            for tp_rank in range(tp_size)
        ),
        rank,
    )

    embd_group = None
    if pp_size > 1:
        embd_group = _create_group_for_rank(
            (
                [
                    _global_rank(0, cp_rank, tp_rank, cp_size, tp_size),
                    _global_rank(pp_size - 1, cp_rank, tp_rank, cp_size, tp_size),
                ]
                for cp_rank in range(cp_size)
                for tp_rank in range(tp_size)
            ),
            rank,
        )

    assert tp_group is not None
    assert pp_group is not None
    return ProcessGroupCollection(
        tp=tp_group,
        cp=cp_group,
        pp=pp_group,
        embd=embd_group,
    )
