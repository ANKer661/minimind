import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_tp import TPContext, TPMiniMindForCausalLM


def build_config(args: argparse.Namespace) -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len, 32),
        dropout=0.0,
        flash_attn=args.flash_attn,
        use_moe=False,
    )


def profile_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    trace_dir: Path,
    wait_steps: int,
    warmup_steps: int,
    active_steps: int,
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    total_steps = wait_steps + warmup_steps + active_steps

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=3,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(total_steps):
            with record_function("train_step"):
                optimizer.zero_grad(set_to_none=True)
                with record_function("forward"):
                    output = model(input_ids)
                    loss = output.logits.float().mean()
                with record_function("backward"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
            profiler.step()


def run_dense(args: argparse.Namespace) -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = MiniMindForCausalLM(build_config(args)).to(
        device=device,
        dtype=getattr(torch, args.dtype),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    profile_model(
        model,
        input_ids,
        optimizer,
        Path(args.profile_dir) / "rank_0",
        args.profile_wait,
        args.profile_warmup,
        args.profile_active,
    )


def run_tp(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == args.tp_size
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tp_context = TPContext(
        group=dist.group.WORLD,
        world_size=world_size,
        rank=rank,
        sequence_parallel=args.sequence_parallel,
        async_communication=args.async_communication,
    )
    model = TPMiniMindForCausalLM(tp_context, build_config(args)).to(
        device=device,
        dtype=getattr(torch, args.dtype),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    input_ids = torch.empty(
        args.batch_size,
        args.seq_len,
        device=device,
        dtype=torch.long,
    )
    if rank == 0:
        input_ids.random_(0, args.vocab_size)
    dist.broadcast(input_ids, src=0)
    dist.barrier()

    profile_model(
        model,
        input_ids,
        optimizer,
        Path(args.profile_dir) / f"rank_{rank}",
        args.profile_wait,
        args.profile_warmup,
        args.profile_active,
    )
    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "tp"), required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_hidden_layers", type=int, required=True)
    parser.add_argument("--num_attention_heads", type=int, required=True)
    parser.add_argument("--num_key_value_heads", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--profile_wait", type=int, required=True)
    parser.add_argument("--profile_warmup", type=int, required=True)
    parser.add_argument("--profile_active", type=int, required=True)
    parser.add_argument(
        "--flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sequence_parallel", action="store_true")
    parser.add_argument("--async_communication", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA")
    if args.mode == "dense":
        run_dense(args)
    else:
        run_tp(args)


if __name__ == "__main__":
    main()
