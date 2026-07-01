import torch.distributed as dist
import torch

from model.model_pp import PPContext


def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


def _batched_p2p_ops(
    *,
    tensor_send_prev: torch.Tensor | None,
    tensor_recv_prev: torch.Tensor | None,
    tensor_send_next: torch.Tensor | None,
    tensor_recv_next: torch.Tensor | None,
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
) -> list[dist.Work]:
    """P2P communication operations with batched isend/irecv."""
    ops = []

    if tensor_send_prev is not None:
        send_prev_op = dist.P2POp(
            dist.isend,
            tensor_send_prev,
            prev_pipeline_rank,
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = dist.P2POp(
            dist.irecv,
            tensor_recv_prev,
            prev_pipeline_rank,
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = dist.P2POp(
            dist.isend,
            tensor_send_next,
            next_pipeline_rank,
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = dist.P2POp(
            dist.irecv,
            tensor_recv_next,
            next_pipeline_rank,
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
    else:
        reqs = []

    return reqs


class P2PCommunicator:
    def __init__(self, pp_context: PPContext) -> None:
        self.pp_group = pp_context.group
        self.pp_context = pp_context
        world_size = dist.get_world_size(group=self.pp_group)
        curr_rank_in_pg = dist.get_rank(group=self.pp_group)

        next_rank_pg = (curr_rank_in_pg + 1) % world_size
        prev_rank_pg = (curr_rank_in_pg - 1) % world_size

        self.prev_rank = dist.get_global_rank(group=self.pp_group, group_rank=prev_rank_pg)
        self.next_rank = dist.get_global_rank(group=self.pp_group, group_rank=next_rank_pg)

    def _communicate(
        self,
        *,
        tensor_send_next: torch.Tensor | None,
        tensor_send_prev: torch.Tensor | None,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: torch.Size,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
        pipeline_dtype = self.pp_context.pipeline_dtype

        tensor_recv_prev = None
        tensor_recv_next = None
        if recv_prev:
            tensor_recv_prev = torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=pipeline_dtype,
            )
        if recv_next:
            tensor_recv_next = torch.empty(
                recv_next_shape,
                requires_grad=False,
                device=torch.cuda.current_device(),
                dtype=pipeline_dtype,
            )

        p2p_reqs = _batched_p2p_ops(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=self.pp_group,
            prev_pipeline_rank=self.prev_rank,
            next_pipeline_rank=self.next_rank,
        )

        for req in p2p_reqs:
            req.wait()

        return tensor_recv_prev, tensor_recv_next

    def recv_forward(self, tensor_shapes: list[torch.Size], is_first_stage: bool):
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            tensor_shapes = [tensor_shapes]  # type: ignore
            unwrap_tensor_shapes = True

        input_tensors = []

        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
            else:
                input_tensor, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
            input_tensors.append(input_tensor)

        if unwrap_tensor_shapes:
            return input_tensors[0]
        return input_tensors

    def recv_backward(self, tensor_shapes: list[torch.Size], is_last_stage: bool):
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            tensor_shapes = [tensor_shapes]  # type: ignore
            unwrap_tensor_shapes = True

        output_tensor_grads = []

        for tensor_shape in tensor_shapes:
            if is_last_stage:
                output_tensor_grad = None
            else:
                _, output_tensor_grad = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
            output_tensor_grads.append(output_tensor_grad)

        if unwrap_tensor_shapes:
            return output_tensor_grads[0]
        return output_tensor_grads

    def send_forward(self, output_tensors: list[torch.Tensor], is_last_stage: bool) -> None:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]  # type: ignore

        for output_tensor in output_tensors:
            if not is_last_stage:
                self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=output_tensor.shape,
                )

    def send_backward(self, input_tensor_grads: list[torch.Tensor], is_first_stage: bool) -> None:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]  # type: ignore

        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=input_tensor_grad.shape,
                )

    def send_forawrd_recv_backward(
        self,
        output_tensors: list[torch.Tensor],
        tensor_shapes: list[torch.Size],
        is_last_stage: bool,
    ):
        unwrap_output_tensors = False
        if not isinstance(output_tensors, list):
            unwrap_output_tensors = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]

        output_tensor_grads = []

        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                output_tensor_grad = None
            else:
                _, output_tensor_grad = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )

            output_tensor_grads.append(output_tensor_grad)

        if unwrap_output_tensors:
            return output_tensor_grads[0]
        return output_tensor_grads

    def send_backward_recv_forward(
        self,
        input_tensor_grads: list[torch.Tensor],
        tensor_shapes: list[torch.Size],
        is_first_stage: bool,
    ):
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]

        input_tensors = []

        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
            else:
                input_tensor, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )

            input_tensors.append(input_tensor)

        if unwrap_input_tensor_grads:
            return input_tensors[0]
        return input_tensors
