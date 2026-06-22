import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


RESULT_PREFIX = "PP_BENCHMARK_RESULT="
MODE_LABELS = {"ddp": "DDP", "tp": "TP (all)", "pp_tp": "TP x PP (all)"}
DEFAULT_MODES = ("ddp", "tp", "pp_tp")


def format_flops(value: float, suffix: str = "FLOP/s") -> str:
    for scale, prefix in (
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ):
        if value >= scale:
            return f"{value / scale:.2f} {prefix}{suffix}"
    return f"{value:.2f} {suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP, TP, and TP x PP throughput")
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_LABELS),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--microbatches", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--seq_len", type=int, default=1024)
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
    parser.add_argument("--output_csv", default="pp_throughput.csv")
    parser.add_argument("--output_plot", default="pp_throughput.png")
    return parser.parse_args()


def worker_command(
    args: argparse.Namespace,
    mode: str,
    num_microbatches: int,
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
        str(args.num_hidden_layers),
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
        str(num_microbatches),
        "--batch_policy",
        "fixed_global",
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


def save_plot(
    rows: list[dict[str, object]],
    path: Path,
    mode_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    microbatch_counts = sorted({int(row["num_microbatches"]) for row in rows})
    positions = list(range(len(microbatch_counts)))
    figure, (throughput_axis, relative_axis) = plt.subplots(1, 2, figsize=(12, 5))
    bar_width = 0.8 / len(mode_names)

    values_by_mode = {}
    for mode_index, mode in enumerate(mode_names):
        label = MODE_LABELS[mode]
        values = {
            int(row["num_microbatches"]): float(row["flops_per_second"])
            for row in rows
            if row["mode"] == mode and row["status"] == "ok"
        }
        values_by_mode[mode] = values
        throughput_axis.bar(
            [
                position + (mode_index - (len(mode_names) - 1) / 2) * bar_width
                for position in positions
            ],
            [values.get(count, math.nan) for count in microbatch_counts],
            width=bar_width,
            label=label,
        )

    baseline_mode = "ddp" if "ddp" in mode_names else mode_names[0]
    baseline_values = values_by_mode[baseline_mode]
    for mode_index, mode in enumerate(mode_names):
        values = values_by_mode[mode]
        relative_axis.bar(
            [
                position + (mode_index - (len(mode_names) - 1) / 2) * bar_width
                for position in positions
            ],
            [
                values.get(count, math.nan) / baseline_values[count] * 100
                if count in baseline_values and count in values
                else math.nan
                for count in microbatch_counts
            ],
            width=bar_width,
            label=MODE_LABELS[mode],
        )
    relative_axis.axhline(100, color="black", linewidth=1, linestyle="--")

    labels = [f"M={count}" for count in microbatch_counts]
    for axis in (throughput_axis, relative_axis):
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Number of microbatches")
        axis.grid(axis="y", alpha=0.3)
    throughput_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: format_flops(value))
    )
    throughput_axis.set_ylabel("Estimated training throughput")
    throughput_axis.set_title("End-to-End Training Throughput")
    throughput_axis.legend()
    relative_axis.set_ylabel("Relative throughput (%)")
    relative_axis.set_title(f"Relative to {MODE_LABELS[baseline_mode]}")
    relative_axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    world_size = args.pp_size * args.tp_size
    assert args.tp_size >= 1
    if "pp_tp" in args.modes:
        assert args.pp_size >= 2
        assert args.num_hidden_layers >= args.pp_size
    assert args.micro_batch_size > 0
    assert args.benchmark_iters > 0
    for count in args.microbatches:
        global_batch_size = args.micro_batch_size * count
        if "ddp" in args.modes:
            assert global_batch_size % world_size == 0, (
                f"global batch {global_batch_size} must be divisible by world size {world_size}"
            )

    rows = []
    for num_microbatches in args.microbatches:
        global_batch_size = args.micro_batch_size * num_microbatches
        for mode in args.modes:
            label = MODE_LABELS[mode]
            status, metrics, output = run_worker(
                worker_command(args, mode, num_microbatches)
            )
            if status == "failed":
                print(output)
                raise RuntimeError(f"{label} worker failed")

            row = {
                "mode": mode,
                "label": label,
                "pp_size": args.pp_size,
                "tp_size": args.tp_size,
                "effective_pp_size": args.pp_size if mode == "pp_tp" else 1,
                "effective_tp_size": (
                    world_size if mode == "tp" else args.tp_size if mode == "pp_tp" else 1
                ),
                "world_size": world_size,
                "layers": args.num_hidden_layers,
                "num_microbatches": num_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "model_batch_size": metrics["model_batch_size"] if metrics else math.nan,
                "global_batch_size": metrics["global_batch_size"] if metrics else math.nan,
                "training_flops": metrics["training_flops"] if metrics else math.nan,
                "flops_per_second": metrics["flops_per_second"] if metrics else math.nan,
                "time_ms": metrics["time_ms"] if metrics else math.nan,
                "peak_mib": metrics["peak_mib"] if metrics else math.nan,
                "status": status,
            }
            rows.append(row)
            if status == "ok":
                print(
                    f"microbatches={num_microbatches:>3} {label:<7} "
                    f"{format_flops(row['flops_per_second'])}, "
                    f"{row['time_ms']:.2f} ms"
                )
            else:
                print(f"microbatches={num_microbatches:>3} {label:<7} OOM")

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)
    save_csv(rows, csv_path)
    save_plot(rows, plot_path, args.modes)
    print(f"saved CSV:  {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
