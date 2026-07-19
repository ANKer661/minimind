import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


RESULT_PREFIX = "PARALLEL_BENCHMARK_RESULT="
MODE_LABELS = {
    "ddp": "DDP",
    "tp": "TP (all)",
    "pp_tp": "TP x PP (all)",
    "cp": "CP",
    "pp": "PP",
    "tp_cp": "TP x CP",
    "tp_pp": "TP x PP",
    "pp_cp": "PP x CP",
    "tp_cp_pp": "TP x CP x PP",
    "parallel": "Configured TP x CP x PP",
}
DEFAULT_MODES = ("ddp", "tp", "pp_tp")
COMPOSED_MODES = {
    "cp": (False, True, False),
    "pp": (False, False, True),
    "tp_cp": (True, True, False),
    "tp_pp": (True, False, True),
    "pp_cp": (False, True, True),
    "tp_cp_pp": (True, True, True),
}


def mode_parallel_sizes(
    args: argparse.Namespace,
    mode: str,
) -> tuple[int, int, bool, int]:
    if mode == "parallel":
        return args.pp_size, args.tp_size, args.cp, args.cp_size if args.cp else 1
    if mode in COMPOSED_MODES:
        use_tp, use_cp, use_pp = COMPOSED_MODES[mode]
        cp_enabled = use_cp and args.cp
        return (
            args.pp_size if use_pp else 1,
            args.tp_size if use_tp else 1,
            cp_enabled,
            args.cp_size if cp_enabled else 1,
        )
    return args.pp_size, args.tp_size, False, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark configurable TP x CP x PP memory and step time"
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
        choices=tuple(MODE_LABELS),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--num_microbatches", type=int, default=4)
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
    parser.add_argument("--output_csv", default="parallel_memory_scaling.csv")
    parser.add_argument("--output_plot", default="parallel_memory_scaling.png")
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


def format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    return f"{value / 1_000_000:.0f}M"


def worker_command(
    args: argparse.Namespace,
    mode: str,
    num_hidden_layers: int,
) -> list[str]:
    effective_pp_size, effective_tp_size, cp_enabled, effective_cp_size = (
        mode_parallel_sizes(args, mode)
    )
    worker_mode = "parallel" if mode == "parallel" or mode in COMPOSED_MODES else mode
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={effective_pp_size * effective_cp_size * effective_tp_size}",
        "--module",
        "evaluation.benchmark_worker",
        "--mode",
        worker_mode,
        "--pp_size",
        str(effective_pp_size),
        "--tp_size",
        str(effective_tp_size),
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
        str(args.seq_len),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--num_microbatches",
        str(args.num_microbatches),
        "--batch_policy",
        "fixed_per_rank",
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
    if worker_mode == "parallel":
        command.extend(["--cp_size", str(effective_cp_size)])
        if cp_enabled:
            command.extend(["--cp", "--cp_comm_type", args.cp_comm_type])
        if args.sequence_parallel:
            command.append("--sequence_parallel")
        if args.async_communication:
            command.append("--async_communication")
        if args.vocab_parallel:
            command.append("--vocab_parallel")
    return command


def run_worker(command: list[str]) -> tuple[str, dict[str, float] | None, str]:
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


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    rows: list[dict[str, object]],
    path: Path,
    mode_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    parameter_counts = sorted({int(row["params"]) for row in rows})
    parameter_labels = [format_parameter_count(value) for value in parameter_counts]
    figure, (memory_axis, time_axis) = plt.subplots(1, 2, figsize=(13, 5))

    baseline_mode = "ddp" if "ddp" in mode_names else mode_names[0]
    baseline_memory = {
        int(row["layers"]): float(row["peak_mib"])
        for row in rows
        if row["mode"] == baseline_mode and row["status"] == "ok"
    }
    baseline_times = {
        int(row["layers"]): float(row["time_ms"])
        for row in rows
        if row["mode"] == baseline_mode and row["status"] == "ok"
    }
    positions = list(range(len(parameter_counts)))
    bar_width = 0.8 / len(mode_names)
    layer_by_params = {
        int(row["params"]): int(row["layers"])
        for row in rows
    }
    max_relative_memory = 100.0
    max_relative_time = 100.0
    for mode_index, mode in enumerate(mode_names):
        label = MODE_LABELS[mode]
        mode_memory = {
            int(row["layers"]): float(row["peak_mib"])
            for row in rows
            if row["mode"] == mode and row["status"] == "ok"
        }
        mode_times = {
            int(row["layers"]): float(row["time_ms"])
            for row in rows
            if row["mode"] == mode and row["status"] == "ok"
        }
        relative_memory = []
        relative_times = []
        for params in parameter_counts:
            layer = layer_by_params[params]
            if layer in baseline_memory and layer in mode_memory:
                relative_memory.append(mode_memory[layer] / baseline_memory[layer] * 100)
            else:
                relative_memory.append(math.nan)
            if layer in baseline_times and layer in mode_times:
                relative_times.append(mode_times[layer] / baseline_times[layer] * 100)
            else:
                relative_times.append(math.nan)
        valid_relative_memory = [
            value for value in relative_memory if not math.isnan(value)
        ]
        valid_relative_times = [
            value for value in relative_times if not math.isnan(value)
        ]
        if valid_relative_memory:
            max_relative_memory = max(max_relative_memory, *valid_relative_memory)
        if valid_relative_times:
            max_relative_time = max(max_relative_time, *valid_relative_times)
        offsets = [
            position + (mode_index - (len(mode_names) - 1) / 2) * bar_width
            for position in positions
        ]
        memory_axis.bar(
            offsets,
            relative_memory,
            width=bar_width,
            label=label,
        )
        time_axis.bar(
            offsets,
            relative_times,
            width=bar_width,
            label=label,
        )

    memory_axis.axhline(100, color="black", linewidth=1, linestyle="--")
    memory_axis.set_ylim(0, max_relative_memory * 1.18)
    memory_axis.set_xticks(positions, parameter_labels, rotation=30)
    memory_axis.xaxis.set_minor_formatter(NullFormatter())
    memory_axis.set_xlabel("Model parameters")
    memory_axis.set_ylabel(f"Peak memory relative to {MODE_LABELS[baseline_mode]} (%)")
    memory_axis.set_title("Relative Peak GPU Memory")
    memory_axis.grid(axis="y", alpha=0.3)
    memory_axis.legend()

    time_axis.axhline(100, color="black", linewidth=1, linestyle="--")
    time_axis.set_ylim(0, max_relative_time * 1.18)
    time_axis.set_xticks(positions, parameter_labels, rotation=30)
    time_axis.set_xlabel("Model parameters")
    time_axis.set_ylabel(f"Step time relative to {MODE_LABELS[baseline_mode]} (%)")
    time_axis.set_title("Relative End-to-End Step Time")
    time_axis.grid(axis="y", alpha=0.3)
    time_axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model_batch_size = args.micro_batch_size * args.num_microbatches
    assert args.tp_size >= 1
    assert args.pp_size >= 1
    assert args.cp_size >= 1
    selected_cp_modes = {
        mode
        for mode in args.modes
        if mode in COMPOSED_MODES and COMPOSED_MODES[mode][1]
    }
    if selected_cp_modes and not args.cp:
        names = ", ".join(sorted(selected_cp_modes))
        raise ValueError(f"CP modes require --cp: {names}")
    if "parallel" in args.modes and not args.cp and args.cp_size != 1:
        raise ValueError("--cp_size requires --cp")
    if "pp_tp" in args.modes:
        assert args.pp_size >= 2
    pp_modes = {
        mode
        for mode in args.modes
        if mode in ("pp_tp", "parallel")
        or (mode in COMPOSED_MODES and COMPOSED_MODES[mode][2])
    }
    if pp_modes:
        assert min(args.layers) >= args.pp_size
    assert args.benchmark_iters > 0

    rows = []
    for num_hidden_layers in args.layers:
        params = count_parameters(args, num_hidden_layers)
        for mode in args.modes:
            effective_pp_size, effective_tp_size, cp_enabled, effective_cp_size = (
                mode_parallel_sizes(args, mode)
            )
            world_size = effective_pp_size * effective_cp_size * effective_tp_size
            label = MODE_LABELS[mode]
            status, metrics, output = run_worker(
                worker_command(args, mode, num_hidden_layers)
            )
            if status == "failed":
                print(output)
                raise RuntimeError(f"{label} worker failed")

            row = {
                "mode": mode,
                "label": label,
                "pp_schedule": args.pp_schedule,
                "pp_size": effective_pp_size,
                "tp_size": effective_tp_size,
                "cp_enabled": cp_enabled,
                "cp_size": effective_cp_size,
                "cp_comm_type": args.cp_comm_type if cp_enabled else "",
                "sequence_parallel": (
                    (mode == "parallel" or mode in COMPOSED_MODES)
                    and args.sequence_parallel
                ),
                "async_communication": (
                    (mode == "parallel" or mode in COMPOSED_MODES)
                    and args.async_communication
                ),
                "vocab_parallel": (
                    (mode == "parallel" or mode in COMPOSED_MODES)
                    and args.vocab_parallel
                ),
                "effective_pp_size": (
                    effective_pp_size
                    if mode == "parallel" or mode in COMPOSED_MODES or mode == "pp_tp"
                    else 1
                ),
                "effective_tp_size": (
                    world_size
                    if mode == "tp"
                    else effective_tp_size
                    if mode in ("pp_tp", "parallel") or mode in COMPOSED_MODES
                    else 1
                ),
                "world_size": world_size,
                "batch_policy": "fixed_per_rank",
                "model_batch_size": metrics["model_batch_size"] if metrics else model_batch_size,
                "global_batch_size": metrics["global_batch_size"] if metrics else math.nan,
                "layers": num_hidden_layers,
                "params": params,
                "peak_mib": metrics["peak_mib"] if metrics else math.nan,
                "time_ms": metrics["time_ms"] if metrics else math.nan,
                "training_flops": metrics["training_flops"] if metrics else math.nan,
                "flops_per_second": metrics["flops_per_second"] if metrics else math.nan,
                "worker_pp_schedule": metrics["pp_schedule"] if metrics else args.pp_schedule,
                "status": status,
            }
            rows.append(row)
            if status == "ok":
                print(
                    f"layers={num_hidden_layers:>3} {label:<7} "
                    f"{row['peak_mib']:.2f} MiB, {row['time_ms']:.2f} ms"
                )
            else:
                print(f"layers={num_hidden_layers:>3} {label:<7} OOM")

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)
    save_csv(rows, csv_path)
    save_plot(rows, plot_path, args.modes)
    print(f"saved CSV:  {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
