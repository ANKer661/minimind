import argparse
import json
import os
import sys
import time
from collections.abc import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_pp import PPContext, PipelineStage
from model.pipeline_parallel_p2p_communication import P2PCommunicator
from model.pipeline_schedules import run_gpipe
from model.tensor_parallel_layers import TPContext


RESULT_PREFIX = "PP_BENCHMARK_RESULT="


def build_config(args: argparse.Namespace) -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len, 32),
        tie_word_embeddings=False,
        dropout=0.0,
        flash_attn=args.flash_attn,
        use_moe=False,
    )


def estimate_training_flops(args: argparse.Namespace) -> int:
    """Estimate logical model FLOPs for one forward/backward training step.

    Counts GEMMs in attention projections, attention score/value products,
    SwiGLU MLP, and lm_head. Backward is approximated as 2x forward.
    """
    batch_size = args.micro_batch_size * args.num_microbatches
    sequence_length = args.seq_len
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    query_size = args.num_attention_heads * head_dim
    kv_size = args.num_key_value_heads * head_dim
    intermediate_size = build_config(args).intermediate_size

    attention_projections = (
        2 * batch_size * sequence_length * hidden_size * query_size
        + 4 * batch_size * sequence_length * hidden_size * kv_size
        + 2 * batch_size * sequence_length * query_size * hidden_size
    )
    attention_products = 4 * batch_size * sequence_length**2 * query_size
    mlp = 6 * batch_size * sequence_length * hidden_size * intermediate_size
    lm_head = 2 * batch_size * sequence_length * hidden_size * args.vocab_size
    forward_flops = args.num_hidden_layers * (
        attention_projections + attention_products + mlp
    ) + lm_head
    return 3 * forward_flops


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


def benchmark_train_step(
    train_step: Callable[[], None],
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[float, float]:
    for _ in range(warmup_iters):
        train_step()

    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(benchmark_iters):
        train_step()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) / benchmark_iters * 1000
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    metrics = torch.tensor([peak_mib, elapsed_ms], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return metrics[0].item(), metrics[1].item()


def run_ddp(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, float]:
    global_batch_size = args.micro_batch_size * args.num_microbatches
    assert global_batch_size % world_size == 0, (
        "DDP requires global_batch_size divisible by world_size"
    )
    local_batch_size = global_batch_size // world_size

    model = MiniMindForCausalLM(build_config(args)).to(
        device=device,
        dtype=getattr(torch, args.dtype),
    )
    model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank + 1)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (local_batch_size, args.seq_len),
        device=device,
        generator=generator,
    )
    labels = input_ids.clone()

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()

    return benchmark_train_step(
        train_step,
        device,
        args.warmup_iters,
        args.benchmark_iters,
    )


def run_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> tuple[float, float]:
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
        pipeline_dtype=getattr(torch, args.dtype),
    )

    config = build_config(args)
    assert config.hidden_size % args.tp_size == 0
    assert config.intermediate_size % args.tp_size == 0
    assert config.num_attention_heads % args.tp_size == 0
    assert config.num_key_value_heads % args.tp_size == 0

    model = PipelineStage(config, pp_context, tp_context).to(
        device=device,
        dtype=getattr(torch, args.dtype),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    input_ids = torch.empty(
        args.micro_batch_size * args.num_microbatches,
        args.seq_len,
        dtype=torch.long,
        device=device,
    )
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        input_ids.random_(0, args.vocab_size, generator=generator)
    dist.broadcast(input_ids, src=0)
    labels = input_ids.clone()
    microbatches = list(
        zip(
            input_ids.split(args.micro_batch_size),
            labels.split(args.micro_batch_size),
        )
    )
    communicator = P2PCommunicator(pp_context)

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        data_iterator = iter(microbatches) if pp_context.is_first or pp_context.is_last else None
        run_gpipe(
            stage_model=model,
            data_iterator=data_iterator,
            num_microbatches=args.num_microbatches,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_len,
            p2p_communicator=communicator,
            pp_context=pp_context,
            tp_context=tp_context,
        )
        optimizer.step()

    return benchmark_train_step(
        train_step,
        device,
        args.warmup_iters,
        args.benchmark_iters,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ddp", "pp_tp"), required=True)
    parser.add_argument("--pp_size", type=int, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_hidden_layers", type=int, required=True)
    parser.add_argument("--num_attention_heads", type=int, required=True)
    parser.add_argument("--num_key_value_heads", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--micro_batch_size", type=int, required=True)
    parser.add_argument("--num_microbatches", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--warmup_iters", type=int, required=True)
    parser.add_argument("--benchmark_iters", type=int, required=True)
    parser.add_argument(
        "--flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == args.pp_size * args.tp_size
    assert args.num_hidden_layers >= args.pp_size
    assert args.micro_batch_size > 0
    assert args.num_microbatches > 0
    assert args.benchmark_iters > 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == "ddp":
        peak_mib, time_ms = run_ddp(args, device, rank, world_size)
    else:
        peak_mib, time_ms = run_pipeline(args, device, rank)

    global_batch_size = args.micro_batch_size * args.num_microbatches
    training_flops = estimate_training_flops(args)
    flops_per_second = training_flops / (time_ms / 1000)
    if rank == 0:
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "mode": args.mode,
                    "peak_mib": peak_mib,
                    "time_ms": time_ms,
                    "training_flops": training_flops,
                    "flops_per_second": flops_per_second,
                    "global_batch_size": global_batch_size,
                }
            )
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
