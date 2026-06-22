import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_pp import PPContext, PipelineStage
from model.model_tp import shard_state_dict_for_tp
from model.pipeline_parallel_p2p_communication import P2PCommunicator
from model.pipeline_schedules import run_gpipe
from model.tensor_parallel_layers import TPContext


COLUMN_PARALLEL_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
)

ROW_PARALLEL_SUFFIXES = (
    "o_proj.weight",
    "down_proj.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MiniMind GPipe pipeline parallelism")
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def create_parallel_groups(
    pp_size: int,
    tp_size: int,
    rank: int,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    tp_group = None
    for pp_rank in range(pp_size):
        ranks = list(range(pp_rank * tp_size, (pp_rank + 1) * tp_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_group = group

    pp_group = None
    for tp_rank in range(tp_size):
        ranks = [pp_rank * tp_size + tp_rank for pp_rank in range(pp_size)]
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            pp_group = group

    assert tp_group is not None
    assert pp_group is not None
    return tp_group, pp_group


def load_dense_weights(
    stage_model: PipelineStage,
    dense_model: MiniMindForCausalLM,
    tp_context: TPContext,
) -> None:
    dense_state_dict = shard_state_dict_for_tp(dense_model.state_dict(), tp_context)
    stage_state_dict = {
        name: dense_state_dict[name]
        for name in stage_model.state_dict()
    }
    stage_model.load_state_dict(stage_state_dict, strict=True)


def make_microbatches(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    batch_size = args.micro_batch_size * args.num_microbatches
    generator = torch.Generator().manual_seed(args.seed + 1)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (batch_size, args.seq_len),
        generator=generator,
    )
    labels = input_ids.clone()
    microbatches = list(
        zip(
            input_ids.split(args.micro_batch_size),
            labels.split(args.micro_batch_size),
        )
    )
    return input_ids, labels, microbatches


def expected_dense_tensor(
    name: str,
    tensor: torch.Tensor,
    tp_context: TPContext,
) -> torch.Tensor:
    if name.endswith(COLUMN_PARALLEL_SUFFIXES):
        tensor = tensor.chunk(tp_context.world_size, dim=0)[tp_context.rank]
    elif name.endswith(ROW_PARALLEL_SUFFIXES):
        tensor = tensor.chunk(tp_context.world_size, dim=1)[tp_context.rank]
    return tensor


def local_max_gradient_diff(
    stage_model: PipelineStage,
    dense_model: MiniMindForCausalLM,
) -> torch.Tensor:
    dense_params = dict(dense_model.named_parameters())
    max_diff = torch.zeros([], dtype=torch.float64, device=torch.cuda.current_device())

    for name, stage_param in stage_model.named_parameters():
        dense_grad = dense_params[name].grad
        stage_grad = stage_param.grad
        assert dense_grad is not None, f"missing Dense gradient: {name}"
        assert stage_grad is not None, f"missing PP gradient: {name}"
        dense_grad = expected_dense_tensor(name, dense_grad, stage_model.tp_context)
        max_diff = torch.maximum(
            max_diff,
            (stage_grad - dense_grad).abs().max().to(torch.float64),
        )

    return max_diff


def local_max_parameter_diff(
    stage_model: PipelineStage,
    dense_model: MiniMindForCausalLM,
) -> torch.Tensor:
    dense_params = dict(dense_model.named_parameters())
    max_diff = torch.zeros([], dtype=torch.float64, device=torch.cuda.current_device())

    for name, stage_param in stage_model.named_parameters():
        dense_param = expected_dense_tensor(
            name,
            dense_params[name],
            stage_model.tp_context,
        )
        max_diff = torch.maximum(
            max_diff,
            (stage_param.detach() - dense_param.detach())
            .abs()
            .max()
            .to(torch.float64),
        )

    return max_diff


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The current P2P communicator requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert args.pp_size >= 2, "GPipe demo requires pp_size >= 2"
    assert args.tp_size >= 1, "tp_size must be positive"
    assert world_size == args.pp_size * args.tp_size, (
        "world_size must equal pp_size * tp_size"
    )
    assert args.num_hidden_layers >= args.pp_size, "each pipeline stage needs at least one layer"
    assert args.micro_batch_size > 0, "micro_batch_size must be positive"
    assert args.num_microbatches > 0, "num_microbatches must be positive"
    assert args.steps > 0, "steps must be positive"

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size
    tp_group, pp_group = create_parallel_groups(args.pp_size, args.tp_size, rank)
    tp_context = TPContext(
        group=tp_group,
        world_size=args.tp_size,
        rank=tp_rank,
    )
    pp_context = PPContext(
        group=pp_group,
        world_size=args.pp_size,
        rank=pp_rank,
        is_first=pp_rank == 0,
        is_last=pp_rank == args.pp_size - 1,
        pipeline_dtype=dtype,
    )

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len, 32),
        tie_word_embeddings=False,
        dropout=0.0,
        flash_attn=False,
        use_moe=False,
    )
    assert config.hidden_size % args.tp_size == 0
    assert config.intermediate_size % args.tp_size == 0
    assert config.num_attention_heads % args.tp_size == 0
    assert config.num_key_value_heads % args.tp_size == 0

    dense_model = MiniMindForCausalLM(config).to(device=device, dtype=dtype)
    stage_model = PipelineStage(config, pp_context, tp_context).to(device=device, dtype=dtype)
    load_dense_weights(stage_model, dense_model, tp_context)

    dense_optimizer = torch.optim.AdamW(dense_model.parameters(), lr=args.learning_rate)
    stage_optimizer = torch.optim.AdamW(stage_model.parameters(), lr=args.learning_rate)
    p2p_communicator = P2PCommunicator(pp_context)

    input_ids, labels, microbatches = make_microbatches(args)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    dense_model.train()
    stage_model.train()

    atol = args.atol
    if atol is None:
        atol = 1e-4 if dtype == torch.float32 else 5e-2

    passed = True
    for step in range(1, args.steps + 1):
        dense_optimizer.zero_grad(set_to_none=True)
        stage_optimizer.zero_grad(set_to_none=True)

        data_iterator = iter(microbatches) if pp_context.is_first or pp_context.is_last else None
        forward_data_store = run_gpipe(
            stage_model=stage_model,
            data_iterator=data_iterator,
            num_microbatches=args.num_microbatches,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_len,
            p2p_communicator=p2p_communicator,
            pp_context=pp_context,
            tp_context=tp_context,
        )

        dense_loss = dense_model(input_ids, labels=labels).loss
        assert dense_loss is not None
        dense_loss.backward()

        pp_loss = torch.zeros([], dtype=torch.float32, device=device)
        if pp_context.is_last:
            loss_sum = torch.stack([item["loss_sum"].float() for item in forward_data_store]).sum()
            num_tokens = torch.stack([item["num_tokens"] for item in forward_data_store]).sum()
            pp_loss = loss_sum / num_tokens.clamp_min(1)
        last_stage_rank = dist.get_global_rank(pp_group, args.pp_size - 1)
        dist.broadcast(pp_loss, src=last_stage_rank, group=pp_group)

        loss_diff = (pp_loss - dense_loss.detach().float()).abs().to(torch.float64)
        grad_diff = local_max_gradient_diff(stage_model, dense_model)
        dist.all_reduce(grad_diff, op=dist.ReduceOp.MAX)

        dense_optimizer.step()
        stage_optimizer.step()

        parameter_diff = local_max_parameter_diff(stage_model, dense_model)
        dist.all_reduce(parameter_diff, op=dist.ReduceOp.MAX)

        step_passed = (
            loss_diff.item() <= atol
            and grad_diff.item() <= atol
            and parameter_diff.item() <= atol
        )
        passed = passed and step_passed

        if rank == 0:
            print(
                f"step {step:>3}: "
                f"dense_loss={dense_loss.item():.6f} "
                f"pp_loss={pp_loss.item():.6f} "
                f"loss_diff={loss_diff.item():.3e} "
                f"grad_max_diff={grad_diff.item():.3e} "
                f"param_max_diff={parameter_diff.item():.3e}"
            )

    passed_tensor = torch.tensor(int(passed), device=device)
    dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
    dist.barrier()
    dist.destroy_process_group()

    if passed_tensor.item() != 1:
        raise RuntimeError(f"GPipe parity check failed with atol={atol}")


if __name__ == "__main__":
    main()
