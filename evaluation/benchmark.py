import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from evaluation.benchmark_utils import (
    build_result_row,
    format_tokens_per_second,
    save_csv,
    save_plot,
)
from evaluation.presets import (
    DEFAULT_MODES,
    MODE_PRESETS,
    FeatureFlags,
    ResolvedMode,
    resolve_modes,
)


RESULT_PREFIX = "PARALLEL_BENCHMARK_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark configurable TP x CP x PP memory or throughput"
    )
    parser.add_argument(
        "--kind",
        choices=("memory", "throughput"),
        default="memory",
    )
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--tp_size", type=int, default=1)
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
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_PRESETS),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--num_hidden_layers", nargs="+", type=int, default=[16])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--seq_len", nargs="+", type=int, default=[1024])
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--num_microbatches", type=int, default=4)
    parser.add_argument(
        "--batch_policy",
        choices=("fixed_global", "fixed_per_rank"),
        default="fixed_global",
        help=(
            "Interpret micro_batch_size * num_microbatches as the global DDP "
            "batch or as each DDP rank's local batch. Model-parallel modes "
            "always process it once per model-parallel group."
        ),
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--pp_schedule",
        choices=("gpipe", "1f1b"),
        default="gpipe",
    )
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--benchmark_iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_csv")
    parser.add_argument("--output_plot")
    return parser.parse_args()


def count_parameters(args: argparse.Namespace, num_hidden_layers: int) -> int:
    head_dim = args.hidden_size // args.num_attention_heads
    query_size = args.num_attention_heads * head_dim
    kv_size = args.num_key_value_heads * head_dim
    intermediate_size = math.ceil(args.hidden_size * math.pi / 64) * 64
    attention = (
        args.hidden_size * query_size
        + 2 * args.hidden_size * kv_size
        + query_size * args.hidden_size
        + 2 * head_dim
    )
    mlp = 3 * args.hidden_size * intermediate_size
    block_norms = 2 * args.hidden_size
    embedding_and_head = 2 * args.vocab_size * args.hidden_size
    return (
        embedding_and_head
        + num_hidden_layers * (attention + mlp + block_norms)
        + args.hidden_size
    )


def worker_command(
    args: argparse.Namespace,
    mode: ResolvedMode,
    num_hidden_layers: int,
    seq_len: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={mode.world_size}",
        "--module",
        "evaluation.benchmark_worker",
        "--mode",
        mode.runner,
        "--pp_size",
        str(mode.pp_size),
        "--tp_size",
        str(mode.tp_size),
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(num_hidden_layers),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_key_value_heads",
        str(args.num_key_value_heads),
        "--vocab_size",
        str(args.vocab_size),
        "--seq_len",
        str(seq_len),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--num_microbatches",
        str(args.num_microbatches),
        "--batch_policy",
        args.batch_policy,
        "--dtype",
        args.dtype,
        "--learning_rate",
        str(args.learning_rate),
        "--pp_schedule",
        args.pp_schedule,
        "--warmup_iters",
        str(args.warmup_iters),
        "--benchmark_iters",
        str(args.benchmark_iters),
        "--seed",
        str(args.seed),
    ]
    if not args.flash_attn:
        command.append("--no-flash_attn")
    if mode.runner == "general":
        command.extend(["--cp_size", str(mode.cp_size)])
        if mode.cp_enabled:
            command.extend(["--cp", "--cp_comm_type", args.cp_comm_type])
    if mode.runner in ("tp", "general"):
        if mode.sequence_parallel:
            command.append("--sequence_parallel")
        if mode.async_communication:
            command.append("--async_communication")
        if mode.vocab_parallel:
            command.append("--vocab_parallel")
    return command


def run_worker(command: list[str]) -> tuple[str, dict[str, object] | None, str]:
    result = subprocess.run(command, text=True, capture_output=True)
    output = result.stdout + result.stderr
    metrics = None
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            metrics = json.loads(line.removeprefix(RESULT_PREFIX))
            break

    if result.returncode == 0 and metrics is not None:
        return "ok", metrics, output
    lowered = output.lower()
    if "out of memory" in lowered or "outofmemoryerror" in lowered:
        return "oom", None, output
    return "failed", None, output


def main() -> None:
    args = parse_args()
    if args.kind == "throughput":
        args.batch_policy = "fixed_global"
        args.output_csv = args.output_csv or "parallel_throughput.csv"
        args.output_plot = args.output_plot or "parallel_throughput.png"
    else:
        args.output_csv = args.output_csv or "parallel_memory_scaling.csv"
        args.output_plot = args.output_plot or "parallel_memory_scaling.png"
    assert args.tp_size >= 1
    assert args.pp_size >= 1
    assert args.cp_size >= 1
    if not args.cp and args.cp_size != 1:
        raise ValueError("--cp_size requires --cp")
    resolved_modes = resolve_modes(
        args.modes,
        pp_size=args.pp_size,
        cp_size=args.cp_size,
        tp_size=args.tp_size,
        cp_enabled=args.cp,
        cli_features=FeatureFlags(
            sequence_parallel=args.sequence_parallel,
            async_communication=args.async_communication,
            vocab_parallel=args.vocab_parallel,
        ),
    )
    ddp_mode = next(
        (mode for mode in resolved_modes if mode.runner == "ddp"),
        None,
    )
    if len(args.num_hidden_layers) > 1 and len(args.seq_len) > 1:
        raise ValueError(
            "scan either --num_hidden_layers or --seq_len, not both"
        )
    if len(args.num_hidden_layers) > 1:
        scan_dimension = "layers"
    elif len(args.seq_len) > 1:
        scan_dimension = "sequence"
    else:
        scan_dimension = "none"
    plot_dimension = "sequence" if scan_dimension == "sequence" else "layers"
    points = [
        (num_hidden_layers, seq_len, args.num_microbatches)
        for num_hidden_layers in args.num_hidden_layers
        for seq_len in args.seq_len
    ]
    assert all(num_hidden_layers > 0 for num_hidden_layers, _, _ in points)
    assert all(seq_len > 0 for _, seq_len, _ in points)
    assert all(num_microbatches > 0 for _, _, num_microbatches in points)
    if ddp_mode is not None and args.batch_policy == "fixed_global":
        for _, _, num_microbatches in points:
            model_batch_size = args.micro_batch_size * num_microbatches
            if model_batch_size % ddp_mode.world_size != 0:
                raise ValueError(
                    "DDP requires micro_batch_size * num_microbatches "
                    f"({model_batch_size}) divisible by DDP world size "
                    f"({ddp_mode.world_size})"
                )
    if any(mode.pp_size > 1 for mode in resolved_modes):
        assert (
            min(num_hidden_layers for num_hidden_layers, _, _ in points)
            >= args.pp_size
        )
    for _, seq_len, _ in points:
        for mode in resolved_modes:
            if mode.cp_enabled and seq_len % mode.cp_size != 0:
                raise ValueError(
                    f"seq_len {seq_len} must be divisible by CP size "
                    f"{mode.cp_size} for mode {mode.name}"
                )
            if mode.sequence_parallel:
                sequence_parallel_size = mode.cp_size * mode.tp_size
            else:
                sequence_parallel_size = 1
            if seq_len % sequence_parallel_size != 0:
                raise ValueError(
                    f"seq_len {seq_len} must be divisible by sequence-parallel "
                    f"size {sequence_parallel_size} for mode {mode.name}"
                )
    assert args.benchmark_iters > 0

    rows = []
    for num_hidden_layers, seq_len, num_microbatches in points:
        args.num_microbatches = num_microbatches
        model_batch_size = args.micro_batch_size * num_microbatches
        params = count_parameters(args, num_hidden_layers)
        for mode in resolved_modes:
            status, metrics, output = run_worker(
                worker_command(
                    args,
                    mode,
                    num_hidden_layers,
                    seq_len,
                )
            )
            if status == "failed":
                print(output)
                raise RuntimeError(f"{mode.label} worker failed")

            row = build_result_row(
                args,
                mode,
                metrics,
                scan_dimension=scan_dimension,
                num_hidden_layers=num_hidden_layers,
                seq_len=seq_len,
                num_microbatches=num_microbatches,
                model_batch_size=model_batch_size,
                params=params,
                status=status,
            )
            rows.append(row)
            if scan_dimension == "layers":
                scale_label = f"layers={num_hidden_layers:>3}"
            elif scan_dimension == "sequence":
                scale_label = f"seq_len={seq_len:>6}"
            else:
                scale_label = (
                    f"layers={num_hidden_layers:>3}, seq_len={seq_len:>6}"
                )
            if status == "ok":
                if args.kind == "throughput":
                    print(
                        f"{scale_label} {mode.label:<28} "
                        f"{format_tokens_per_second(row['tokens_per_second'])} tokens/s, "
                        f"{row['peak_mib']:.2f} MiB"
                    )
                else:
                    print(
                        f"{scale_label} {mode.label:<28} "
                        f"{row['peak_mib']:.2f} MiB, {row['time_ms']:.2f} ms"
                    )
            else:
                print(f"{scale_label} {mode.label:<7} OOM")

    if args.kind == "memory" and scan_dimension == "sequence":
        print("maximum successful tested sequence length:")
        for mode in resolved_modes:
            successful_lengths = [
                int(row["seq_len"])
                for row in rows
                if row["mode"] == mode.name and row["status"] == "ok"
            ]
            result = f"{max(successful_lengths):,}" if successful_lengths else "none"
            print(f"  {mode.label:<28} {result}")

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)
    save_csv(rows, csv_path)
    mode_names = [mode.name for mode in resolved_modes]
    save_plot(rows, plot_path, mode_names, plot_dimension, args.kind)
    print(f"saved CSV:  {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
