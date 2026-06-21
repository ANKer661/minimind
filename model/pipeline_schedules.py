from collections.abc import Callable
from typing import Iterable

import torch
import torch.nn.functional as F

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
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if input_tensor is None:
        # first stage, get input from data iterator
        input_ids, labels = next(data_iterator)  # type: ignore
        input_tensor = input_ids.to(torch.cuda.current_device())

    stage_output = stage_model(input_tensor)
    num_tokens = torch.tensor(0, dtype=torch.int)

    if stage_model.pp_context.is_last:
        _, labels = next(data_iterator)  # type: ignore
        assert labels is not None, "labels must be provided on the last pipeline stage"
        labels = labels.to(torch.cuda.current_device())
        stage_output, num_tokens = loss_func(stage_output, labels, stage_model.tp_context)

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
):
    input_tensors = []
    output_tensors = []
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
        )

        p2p_communicator.send_forward(
            output_tensors=output_tensor,  # type: ignore
            is_last_stage=pp_context.is_last,
        )

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        if pp_context.is_last:
            total_num_tokens += num_tokens

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
