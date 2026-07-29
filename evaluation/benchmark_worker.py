import argparse
import json
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model.attention_cp import CPContext
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_pp import PPContext, PipelineStage
from model.model_tp import TPMiniMindForCausalLM
from model.pipeline_parallel_p2p_communication import P2PCommunicator
from model.pipeline_schedules import run_pipeline_schedule
from model.tensor_parallel_layers import TPContext


RESULT_PREFIX = "PARALLEL_BENCHMARK_RESULT="


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


def estimate_training_flops(args: argparse.Namespace, batch_size: int) -> int:
    """Estimate logical model FLOPs for one forward/backward training step.

    Counts GEMMs in attention projections, attention score/value products,
    SwiGLU MLP, and lm_head. Backward is approximated as 2x forward.
    """
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
    cp_size: int,
    tp_size: int,
    rank: int,
    cp_enabled: bool,
) -> tuple[
    dist.ProcessGroup,
    dist.ProcessGroup | None,
    dist.ProcessGroup,
    int,
    int,
    int,
]:
    local_tp_rank = rank % tp_size
    parallel_rank = rank // tp_size
    local_cp_rank = parallel_rank % cp_size
    local_pp_rank = parallel_rank // cp_size

    tp_group = None
    for pp_rank in range(pp_size):
        for cp_rank in range(cp_size):
            ranks = [
                (pp_rank * cp_size + cp_rank) * tp_size + tp_rank
                for tp_rank in range(tp_size)
            ]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group

    cp_group = None
    if cp_enabled:
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                ranks = [
                    (pp_rank * cp_size + cp_rank) * tp_size + tp_rank
                    for cp_rank in range(cp_size)
                ]
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    cp_group = group

    pp_group = None
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            ranks = [
                (pp_rank * cp_size + cp_rank) * tp_size + tp_rank
                for pp_rank in range(pp_size)
            ]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                pp_group = group

    assert tp_group is not None
    assert pp_group is not None
    return (
        tp_group,
        cp_group,
        pp_group,
        local_pp_rank,
        local_cp_rank,
        local_tp_rank,
    )


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
    if args.batch_policy == "fixed_global":
        assert global_batch_size % world_size == 0, (
            "DDP requires global_batch_size divisible by world_size"
        )
        local_batch_size = global_batch_size // world_size
    else:
        local_batch_size = global_batch_size

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
    cp_enabled = args.mode == "parallel" and args.cp
    cp_size = args.cp_size if cp_enabled else 1
    tp_group, cp_group, pp_group, pp_rank, cp_rank, tp_rank = create_parallel_groups(
        args.pp_size,
        cp_size,
        args.tp_size,
        rank,
        cp_enabled,
    )
    configurable = args.mode == "parallel"
    tp_context = TPContext(
        group=tp_group,
        world_size=args.tp_size,
        rank=tp_rank,
        sequence_parallel=args.sequence_parallel if configurable else True,
        async_communication=args.async_communication if configurable else True,
        vocab_parallel=args.vocab_parallel if configurable else True,
    )
    cp_context = None
    if cp_enabled:
        assert cp_group is not None
        cp_context = CPContext(
            world_size=cp_size,
            rank=cp_rank,
            group=cp_group,
            comm_type=args.cp_comm_type,
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
    if tp_context.vocab_parallel:
        assert config.vocab_size % args.tp_size == 0
    if tp_context.sequence_parallel:
        assert args.seq_len % (cp_size * args.tp_size) == 0
    if cp_enabled:
        assert args.seq_len % cp_size == 0
    if cp_enabled and args.cp_comm_type == "a2a":
        assert (config.num_attention_heads // args.tp_size) % cp_size == 0
        assert (config.num_key_value_heads // args.tp_size) % cp_size == 0

    model = PipelineStage(config, pp_context, tp_context, cp_context).to(
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
        run_pipeline_schedule(
            schedule=args.pp_schedule,
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


def run_tp(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, float]:
    config = build_config(args)
    assert config.hidden_size % world_size == 0
    assert config.intermediate_size % world_size == 0
    assert config.num_attention_heads % world_size == 0
    assert config.num_key_value_heads % world_size == 0
    if args.vocab_parallel:
        assert config.vocab_size % world_size == 0
    if args.sequence_parallel:
        assert args.seq_len % world_size == 0

    tp_context = TPContext(
        group=dist.group.WORLD,
        world_size=world_size,
        rank=rank,
        sequence_parallel=args.sequence_parallel,
        async_communication=args.async_communication,
        vocab_parallel=args.vocab_parallel,
    )
    model = TPMiniMindForCausalLM(tp_context, config).to(
        device=device,
        dtype=getattr(torch, args.dtype),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    global_batch_size = args.micro_batch_size * args.num_microbatches
    input_ids = torch.empty(
        global_batch_size,
        args.seq_len,
        dtype=torch.long,
        device=device,
    )
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        input_ids.random_(0, args.vocab_size, generator=generator)
    dist.broadcast(input_ids, src=0)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("ddp", "tp", "pp_tp", "parallel"),
        required=True,
    )
    parser.add_argument("--pp_size", type=int, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument(
        "--cp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument(
        "--cp_comm_type",
        choices=("all_gather", "a2a"),
        default="all_gather",
    )
    parser.add_argument("--sequence_parallel", action="store_true")
    parser.add_argument("--async_communication", action="store_true")
    parser.add_argument("--vocab_parallel", action="store_true")
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_hidden_layers", type=int, required=True)
    parser.add_argument("--num_attention_heads", type=int, required=True)
    parser.add_argument("--num_key_value_heads", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--micro_batch_size", type=int, required=True)
    parser.add_argument("--num_microbatches", type=int, required=True)
    parser.add_argument(
        "--batch_policy",
        choices=("fixed_global", "fixed_per_rank"),
        required=True,
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--warmup_iters", type=int, required=True)
    parser.add_argument("--benchmark_iters", type=int, required=True)
    parser.add_argument(
        "--pp_schedule",
        choices=("gpipe", "1f1b"),
        required=True,
    )
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
    assert args.pp_size >= 1
    assert args.tp_size >= 1
    assert args.cp_size >= 1
    if not args.cp and args.cp_size != 1:
        raise ValueError("--cp_size requires --cp")
    effective_cp_size = args.cp_size if args.mode == "parallel" and args.cp else 1
    if args.mode != "ddp":
        assert world_size == args.pp_size * effective_cp_size * args.tp_size
    if args.mode in ("pp_tp", "parallel"):
        assert args.num_hidden_layers >= args.pp_size
    assert args.micro_batch_size > 0
    assert args.num_microbatches > 0
    assert args.benchmark_iters > 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == "ddp":
        peak_mib, time_ms = run_ddp(args, device, rank, world_size)
    elif args.mode == "tp":
        peak_mib, time_ms = run_tp(args, device, rank, world_size)
    else:
        peak_mib, time_ms = run_pipeline(args, device, rank)

    model_batch_size = args.micro_batch_size * args.num_microbatches
    global_batch_size = (
        model_batch_size * world_size
        if args.mode == "ddp" and args.batch_policy == "fixed_per_rank"
        else model_batch_size
    )
    training_flops = estimate_training_flops(args, global_batch_size)
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
                    "batch_policy": args.batch_policy,
                    "pp_schedule": args.pp_schedule,
                    "model_batch_size": model_batch_size,
                    "global_batch_size": global_batch_size,
                }
            )
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
