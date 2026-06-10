import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_tp import TPContext, TPMiniMindForCausalLM


RESULT_PREFIX = "TP_MEMORY_RESULT="


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


def benchmark_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[float, float]:
    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids)
        output.logits.float().mean().backward()
        optimizer.step()

    for _ in range(warmup_iters):
        train_step()
    torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        train_step()
    torch.cuda.synchronize(device)

    elapsed_ms = (time.perf_counter() - start) / benchmark_iters * 1000
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    return peak_mib, elapsed_ms


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
    peak_mib, time_ms = benchmark_model(
        model,
        input_ids,
        device,
        optimizer,
        args.warmup_iters,
        args.benchmark_iters,
    )
    print(
        RESULT_PREFIX
        + json.dumps({"mode": "dense", "peak_mib": peak_mib, "time_ms": time_ms})
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
    peak_mib, time_ms = benchmark_model(
        model,
        input_ids,
        device,
        optimizer,
        args.warmup_iters,
        args.benchmark_iters,
    )
    metrics = torch.tensor([peak_mib, time_ms], device=device, dtype=torch.float64)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            RESULT_PREFIX
            + json.dumps(
                {"mode": "tp", "peak_mib": metrics[0].item(), "time_ms": metrics[1].item()}
            )
        )
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
    parser.add_argument("--warmup_iters", type=int, required=True)
    parser.add_argument("--benchmark_iters", type=int, required=True)
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
        raise RuntimeError("This benchmark requires CUDA")
    if args.mode == "dense":
        run_dense(args)
    else:
        run_tp(args)


if __name__ == "__main__":
    main()
