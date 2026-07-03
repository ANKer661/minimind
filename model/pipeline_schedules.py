from collections.abc import Callable
from typing import Iterable

import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.model_minimind import MiniMindConfig

from .model_pp import PipelineStage, PPContext
from .pipeline_parallel_p2p_communication import P2PCommunicator
from .tensor_parallel_layers import vocab_parallel_cross_entropy, TPContext


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_context: TPContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    num_tokens = (y != -100).sum()
    if tp_context.vocab_parallel:
        loss_mean = vocab_parallel_cross_entropy(x, y, tp_context.group, ignore_index=-100)
        loss_sum = loss_mean * num_tokens
    else:
        loss_sum = F.cross_entropy(
            x.view(-1, x.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

    return loss_sum, num_tokens


def forward_step(
    stage_model: PipelineStage,
    input_tensor: torch.Tensor | None,
    data_iterator: Iterable | None,
    loss_func: Callable,
    forward_data_store: list[dict[str, torch.Tensor]],
    collect_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_tensor is None:
        # first stage, get input from data iterator
        input_ids, labels = next(data_iterator)  # type: ignore
        input_tensor = input_ids.to(torch.cuda.current_device())

    stage_output = stage_model(input_tensor)
    num_tokens = torch.zeros([], dtype=torch.int, device=stage_output.device)

    if stage_model.pp_context.is_last:
        _, labels = next(data_iterator)  # type: ignore
        assert labels is not None, "labels must be provided on the last pipeline stage"
        labels = labels.to(torch.cuda.current_device())
        logits = stage_output
        stage_output, num_tokens = loss_func(stage_output, labels, stage_model.tp_context)
        forward_data = {
            "loss_sum": stage_output.detach(),
            "num_tokens": num_tokens.detach(),
        }
        if collect_logits:
            forward_data["logits"] = logits.detach()
        forward_data_store.append(forward_data)

    return stage_output, num_tokens


def backward_step(
    input_tensor: torch.Tensor | None,
    output_tensor: torch.Tensor,
    output_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    torch.autograd.backward(
        tensors=output_tensor,
        grad_tensors=output_grad,
    )

    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    return input_tensor_grad


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    config: MiniMindConfig,
    tp_context: TPContext,
) -> torch.Size:
    """Determine tensor shapes for pipeline communication.

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.
    """

    # Fixed sequence lengths - compute shape
    effective_seq_length = seq_length

    if tp_context.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_context.world_size

    return torch.Size([effective_seq_length, micro_batch_size, config.hidden_size])


def run_gpipe(
    stage_model: PipelineStage,
    data_iterator: Iterable | None,
    num_microbatches: int,
    micro_batch_size: int,
    seq_length: int,
    p2p_communicator: P2PCommunicator,
    pp_context: PPContext,
    tp_context: TPContext,
    forward_only: bool = False,
    collect_logits: bool = False,
) -> list[dict[str, torch.Tensor]]:
    input_tensors = []
    output_tensors = []
    forward_data_store: list[dict[str, torch.Tensor]] = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=stage_model.config,
        tp_context=tp_context,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=stage_model.config,
        tp_context=tp_context,
    )

    # all forward passes
    for _ in range(num_microbatches):
        input_tensor = p2p_communicator.recv_forward(
            tensor_shapes=recv_tensor_shapes,  # type: ignore
            is_first_stage=pp_context.is_first,
        )  # None if first stage, else tensor from previous stage

        output_tensor, num_tokens = forward_step(
            stage_model=stage_model,
            input_tensor=input_tensor,  # type: ignore
            data_iterator=data_iterator,
            loss_func=causal_lm_loss,
            forward_data_store=forward_data_store,
            collect_logits=collect_logits,
        )

        p2p_communicator.send_forward(
            output_tensors=output_tensor,  # type: ignore
            is_last_stage=pp_context.is_last,
        )

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        total_num_tokens += num_tokens

    if forward_only:
        return forward_data_store

    # all backward passes
    for _ in range(num_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        output_tensor_grad = p2p_communicator.recv_backward(
            tensor_shapes=send_tensor_shapes,  # type: ignore
            is_last_stage=pp_context.is_last,
        )  # None if last stage, else grad tensor from next stage

        if pp_context.is_last:
            output_tensor = output_tensor / total_num_tokens.clamp_min(1)

        input_tensor_grad = backward_step(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            output_grad=output_tensor_grad,  # type: ignore
        )

        p2p_communicator.send_backward(
            input_tensor_grads=input_tensor_grad,  # type: ignore
            is_first_stage=pp_context.is_first,
        )

    return forward_data_store


def run_pipeline_schedule(
    schedule: str,
    stage_model: PipelineStage,
    data_iterator: Iterable | None,
    num_microbatches: int,
    micro_batch_size: int,
    seq_length: int,
    p2p_communicator: P2PCommunicator,
    pp_context: PPContext,
    tp_context: TPContext,
    forward_only: bool = False,
    collect_logits: bool = False,
) -> list[dict[str, torch.Tensor]]:
    if schedule == "gpipe":
        schedule_func = run_gpipe
    elif schedule == "1f1b":
        schedule_func = run_1f1b
    else:
        raise ValueError(f"unsupported pipeline schedule: {schedule}")

    return schedule_func(
        stage_model=stage_model,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        p2p_communicator=p2p_communicator,
        pp_context=pp_context,
        tp_context=tp_context,
        forward_only=forward_only,
        collect_logits=collect_logits,
    )


def run_1f1b(
    stage_model: PipelineStage,
    data_iterator: Iterable | None,
    num_microbatches: int,
    micro_batch_size: int,
    seq_length: int,
    p2p_communicator: P2PCommunicator,
    pp_context: PPContext,
    tp_context: TPContext,
    forward_only: bool = False,
    collect_logits: bool = False,
) -> list[dict[str, torch.Tensor]]:
    input_tensors = None
    output_tensors = None

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store: list[dict[str, torch.Tensor]] = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=stage_model.config,
        tp_context=tp_context,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        config=stage_model.config,
        tp_context=tp_context,
    )

    num_warmup_microbatches = min(pp_context.world_size - pp_context.rank - 1, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # warmup phase: only forward passes
    for _ in range(num_warmup_microbatches):
        input_tensor = p2p_communicator.recv_forward(
            tensor_shapes=recv_tensor_shapes,  # type: ignore
            is_first_stage=pp_context.is_first,
        )  # None if first stage, else tensor from previous stage

        output_tensor, num_tokens = forward_step(
            stage_model=stage_model,
            input_tensor=input_tensor,  # type: ignore
            data_iterator=data_iterator,
            loss_func=causal_lm_loss,
            forward_data_store=forward_data_store,
            collect_logits=collect_logits,
        )

        p2p_communicator.send_forward(
            output_tensors=output_tensor,  # type: ignore
            is_last_stage=pp_context.is_last,
        )

        if not forward_only:
            input_tensors.append(input_tensor)  # type: ignore
            output_tensors.append(output_tensor)  # type: ignore

        total_num_tokens += num_tokens

    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            tensor_shapes=recv_tensor_shapes,  # type: ignore
            is_first_stage=pp_context.is_first,
        )  # None if first stage, else tensor from previous stage

    # steady phase
    for i in range(num_microbatches_remaining):
        last_iteration = i == num_microbatches_remaining - 1

        output_tensor, num_tokens = forward_step(
            stage_model=stage_model,
            input_tensor=input_tensor,  # type: ignore
            data_iterator=data_iterator,
            loss_func=causal_lm_loss,
            forward_data_store=forward_data_store,
            collect_logits=collect_logits,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(
                output_tensors=output_tensor,  # type: ignore
                is_last_stage=pp_context.is_last,
            )
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    tensor_shapes=recv_tensor_shapes,  # type: ignore
                    is_first_stage=pp_context.is_first,
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensors=output_tensor,  # type: ignore
                tensor_shapes=send_tensor_shapes,  # type: ignore
                is_last_stage=pp_context.is_last,
            )

            input_tensors.append(input_tensor)  # type: ignore
            output_tensors.append(output_tensor)  # type: ignore

            # backward pass
            input_tensor = input_tensors.pop(0)  # type: ignore
            output_tensor = output_tensors.pop(0)  # type: ignore

            input_tensor_grad = backward_step(
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_grad=output_tensor_grad,  # type: ignore
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grads=input_tensor_grad,  # type: ignore
                    is_first_stage=pp_context.is_first,
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grads=input_tensor_grad,  # type: ignore
                    tensor_shapes=recv_tensor_shapes,  # type: ignore
                    is_first_stage=pp_context.is_first,
                )

    # cooldown phase: only backward passes
    if not forward_only:
        for _ in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)  # type: ignore
            output_tensor = output_tensors.pop(0)  # type: ignore

            output_tensor_grad = p2p_communicator.recv_backward(
                tensor_shapes=send_tensor_shapes,  # type: ignore
                is_last_stage=pp_context.is_last,
            )  # None if last stage, else grad tensor from next stage

            input_tensor_grad = backward_step(
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_grad=output_tensor_grad,  # type: ignore
            )

            p2p_communicator.send_backward(
                input_tensor_grads=input_tensor_grad,  # type: ignore
                is_first_stage=pp_context.is_first,
            )

        finalize_model_grads(
            model=[stage_model],
            pp_context=pp_context,
            num_tokens=total_num_tokens,
        )

    return forward_data_store


def finalize_model_grads(
    model: list[PipelineStage],
    pp_context: PPContext,
    num_tokens: torch.Tensor,
) -> None:
    last_rank = dist.get_global_rank(group=pp_context.group, group_rank=pp_context.world_size - 1)
    dist.broadcast(num_tokens, src=last_rank, group=pp_context.group)

    safe_num_tokens = torch.clamp(num_tokens, min=1)
    scaling_factor = 1.0 / safe_num_tokens.float()
    for model_chunk in model:
        model_chunk.scale_grads(scaling_factor)
