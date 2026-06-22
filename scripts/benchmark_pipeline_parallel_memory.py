import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


RESULT_PREFIX = "PP_BENCHMARK_RESULT="
MODES = (("ddp", "DDP"), ("pp_tp", "TP x PP"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP and TP x PP memory scaling")
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--tp_size", type=int, default=1)
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
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--benchmark_iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_csv", default="pp_memory_scaling.csv")
    parser.add_argument("--output_plot", default="pp_memory_scaling.png")
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
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun was not found in PATH")
    worker = Path(__file__).with_name("benchmark_pipeline_parallel_worker.py").resolve()
    command = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={args.pp_size * args.tp_size}",
        str(worker),
        "--mode",
        mode,
        "--pp_size",
        str(args.pp_size),
        "--tp_size",
        str(args.tp_size),
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
        "--dtype",
        args.dtype,
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_iters",
        str(args.warmup_iters),
        "--benchmark_iters",
        str(args.benchmark_iters),
        "--seed",
        str(args.seed),
    ]
    if not args.flash_attn:
        command.append("--no-flash_attn")
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


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullFormatter
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    parameter_counts = sorted({int(row["params"]) for row in rows})
    parameter_labels = [format_parameter_count(value) for value in parameter_counts]
    figure, (memory_axis, time_axis) = plt.subplots(1, 2, figsize=(13, 5))

    for mode, label in MODES:
        mode_rows = [
            row for row in rows if row["mode"] == mode and row["status"] == "ok"
        ]
        memory_axis.plot(
            [row["params"] for row in mode_rows],
            [row["peak_mib"] / 1024 for row in mode_rows],
            marker="o",
            label=label,
        )

    memory_axis.set_xscale("log", base=2)
    memory_axis.set_yscale("log", base=2)
    memory_axis.xaxis.set_major_locator(FixedLocator(parameter_counts))
    memory_axis.xaxis.set_major_formatter(FixedFormatter(parameter_labels))
    memory_axis.xaxis.set_minor_formatter(NullFormatter())
    memory_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    memory_axis.set_xlabel("Model parameters")
    memory_axis.set_ylabel("Peak allocated memory per GPU (GB)")
    memory_axis.set_title("Peak GPU Memory")
    memory_axis.grid(which="both", alpha=0.3)
    memory_axis.legend()

    ddp_times = {
        int(row["layers"]): float(row["time_ms"])
        for row in rows
        if row["mode"] == "ddp" and row["status"] == "ok"
    }
    positions = list(range(len(parameter_counts)))
    bar_width = 0.36
    layer_by_params = {
        int(row["params"]): int(row["layers"])
        for row in rows
    }
    for mode_index, (mode, label) in enumerate(MODES):
        mode_times = {
            int(row["layers"]): float(row["time_ms"])
            for row in rows
            if row["mode"] == mode and row["status"] == "ok"
        }
        relative_times = []
        for params in parameter_counts:
            layer = layer_by_params[params]
            if layer in ddp_times and layer in mode_times:
                relative_times.append(mode_times[layer] / ddp_times[layer] * 100)
            else:
                relative_times.append(math.nan)
        time_axis.bar(
            [position + (mode_index - 0.5) * bar_width for position in positions],
            relative_times,
            width=bar_width,
            label=label,
        )

    time_axis.axhline(100, color="black", linewidth=1, linestyle="--")
    time_axis.set_xticks(positions, parameter_labels, rotation=30)
    time_axis.set_xlabel("Model parameters")
    time_axis.set_ylabel("Step time relative to DDP (%)")
    time_axis.set_title("Relative End-to-End Step Time")
    time_axis.grid(axis="y", alpha=0.3)
    time_axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    world_size = args.pp_size * args.tp_size
    global_batch_size = args.micro_batch_size * args.num_microbatches
    assert args.pp_size >= 2
    assert args.tp_size >= 1
    assert min(args.layers) >= args.pp_size
    assert global_batch_size % world_size == 0, (
        f"global batch {global_batch_size} must be divisible by world size {world_size}"
    )
    assert args.benchmark_iters > 0

    rows = []
    for num_hidden_layers in args.layers:
        params = count_parameters(args, num_hidden_layers)
        for mode, label in MODES:
            status, metrics, output = run_worker(
                worker_command(args, mode, num_hidden_layers)
            )
            if status == "failed":
                print(output)
                raise RuntimeError(f"{label} worker failed")

            row = {
                "mode": mode,
                "label": label,
                "pp_size": args.pp_size,
                "tp_size": args.tp_size,
                "world_size": world_size,
                "global_batch_size": global_batch_size,
                "layers": num_hidden_layers,
                "params": params,
                "peak_mib": metrics["peak_mib"] if metrics else math.nan,
                "time_ms": metrics["time_ms"] if metrics else math.nan,
                "training_flops": metrics["training_flops"] if metrics else math.nan,
                "flops_per_second": metrics["flops_per_second"] if metrics else math.nan,
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
    save_plot(rows, plot_path)
    print(f"saved CSV:  {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
