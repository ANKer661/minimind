import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RESULT_PREFIX = "TP_MEMORY_RESULT="


@dataclass(frozen=True)
class Variant:
    label: str
    mode: str
    sequence_parallel: bool = False
    async_communication: bool = False


VARIANTS = {
    "dense": Variant("Dense", "dense"),
    "tp": Variant("TP", "tp"),
    "tp_async": Variant("TP + Async", "tp", async_communication=True),
    "tp_sp": Variant("TP + SP", "tp", sequence_parallel=True),
    "tp_sp_async": Variant(
        "TP + SP + Async",
        "tp",
        sequence_parallel=True,
        async_communication=True,
    ),
}


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
    per_layer = attention + mlp + block_norms

    embedding_and_lm_head = args.vocab_size * args.hidden_size
    return (
        embedding_and_lm_head
        + num_hidden_layers * per_layer
        + args.hidden_size
    )


def format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    return f"{value / 1_000_000:.0f}M"


def parse_result(output: str) -> dict[str, float] | None:
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line.removeprefix(RESULT_PREFIX))
            return {
                "peak_mib": float(result["peak_mib"]),
                "time_ms": float(result["time_ms"]),
            }
    return None


def run_subprocess(command: list[str]) -> tuple[str, dict[str, float] | None, str]:
    result = subprocess.run(command, text=True, capture_output=True)
    output = result.stdout + result.stderr
    metrics = parse_result(output)
    if result.returncode == 0 and metrics is not None:
        return "ok", metrics, output

    lowered_output = output.lower()
    if "out of memory" in lowered_output or "outofmemoryerror" in lowered_output:
        return "oom", None, output
    return "failed", None, output


def worker_command(
    args: argparse.Namespace,
    variant: Variant,
    num_hidden_layers: int,
) -> list[str]:
    worker = str(Path(__file__).with_name("benchmark_tensor_parallel_worker.py").resolve())
    common = [
        worker,
        "--mode",
        variant.mode,
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
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_iters",
        str(args.warmup_iters),
        "--benchmark_iters",
        str(args.benchmark_iters),
    ]
    if not args.flash_attn:
        common.append("--no-flash_attn")
    if variant.sequence_parallel:
        common.append("--sequence_parallel")
    if variant.async_communication:
        common.append("--async_communication")

    if variant.mode == "dense":
        return [sys.executable, *common]

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun was not found in PATH")
    return [
        torchrun,
        "--standalone",
        f"--nproc_per_node={args.tp_size}",
        *common,
    ]


def profiler_command(
    args: argparse.Namespace,
    variant: Variant,
    num_hidden_layers: int,
) -> list[str]:
    worker = str(
        Path(__file__).with_name("benchmark_tensor_parallel_profiler.py").resolve()
    )
    profile_dir = Path(args.profile_dir) / (
        f"{args.profile_variant}_layers_{num_hidden_layers}"
    )
    common = [
        worker,
        "--mode",
        variant.mode,
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
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--learning_rate",
        str(args.learning_rate),
        "--profile_dir",
        str(profile_dir),
        "--profile_wait",
        str(args.profile_wait),
        "--profile_warmup",
        str(args.profile_warmup),
        "--profile_active",
        str(args.profile_active),
    ]
    if not args.flash_attn:
        common.append("--no-flash_attn")
    if variant.sequence_parallel:
        common.append("--sequence_parallel")
    if variant.async_communication:
        common.append("--async_communication")

    if variant.mode == "dense":
        return [sys.executable, *common]

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun was not found in PATH")
    return [
        torchrun,
        "--standalone",
        f"--nproc_per_node={args.tp_size}",
        *common,
    ]


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "variant",
                "label",
                "layers",
                "params",
                "peak_mib",
                "time_ms",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    rows: list[dict[str, object]],
    path: Path,
    variant_names: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import (
            FixedFormatter,
            FixedLocator,
            FuncFormatter,
            LogLocator,
            NullFormatter,
        )
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    def format_log_value(value: float, _: int) -> str:
        if value >= 1:
            return f"{value:g}"
        return f"{value:.3g}"

    figure, (memory_axis, time_axis) = plt.subplots(1, 2, figsize=(13, 5))
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        variant_rows = [
            row
            for row in rows
            if row["variant"] == variant_name and row["status"] == "ok"
        ]
        if not variant_rows:
            continue

        memory_axis.plot(
            [row["params"] for row in variant_rows],
            [row["peak_mib"] / 1024 for row in variant_rows],
            marker="o",
            label=variant.label,
        )

    parameter_counts = sorted({int(row["params"]) for row in rows})
    parameter_labels = [format_parameter_count(value) for value in parameter_counts]
    memory_axis.set_xscale("log", base=2)
    memory_axis.xaxis.set_major_locator(FixedLocator(parameter_counts))
    memory_axis.xaxis.set_major_formatter(FixedFormatter(parameter_labels))
    memory_axis.xaxis.set_minor_formatter(NullFormatter())
    memory_axis.tick_params(axis="x", labelrotation=30)
    memory_axis.set_yscale("log", base=2)
    memory_axis.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0,)))
    memory_axis.yaxis.set_major_formatter(FuncFormatter(format_log_value))
    memory_axis.yaxis.set_minor_formatter(NullFormatter())

    memory_axis.set_xlabel("Model parameters")
    memory_axis.set_ylabel("Peak allocated memory per GPU (GB)")
    memory_axis.set_title("Peak GPU Memory")
    memory_axis.grid(which="both", alpha=0.3)
    memory_axis.legend()

    if "dense" in variant_names:
        baseline_name = "dense"
    else:
        baseline_name = variant_names[0]
    baseline_rows = sorted(
        (
            row
            for row in rows
            if row["variant"] == baseline_name and row["status"] == "ok"
        ),
        key=lambda row: int(row["layers"]),
    )
    if baseline_rows:
        baseline_label = VARIANTS[baseline_name].label
        group_positions = list(range(len(baseline_rows)))
        baseline_times = {
            int(row["layers"]): float(row["time_ms"])
            for row in baseline_rows
        }
        bar_width = 0.8 / len(variant_names)
        max_relative_time = 100.0

        for variant_index, variant_name in enumerate(variant_names):
            variant = VARIANTS[variant_name]
            variant_times = {
                int(row["layers"]): float(row["time_ms"])
                for row in rows
                if row["variant"] == variant_name and row["status"] == "ok"
            }
            relative_times = [
                (
                    variant_times[int(row["layers"])]
                    / baseline_times[int(row["layers"])]
                    * 100
                )
                if int(row["layers"]) in variant_times
                else math.nan
                for row in baseline_rows
            ]
            max_relative_time = max(
                max_relative_time,
                *(value for value in relative_times if not math.isnan(value)),
            )
            offsets = [
                position
                + (variant_index - (len(variant_names) - 1) / 2) * bar_width
                for position in group_positions
            ]
            bars = time_axis.bar(
                offsets,
                relative_times,
                width=bar_width,
                label=variant.label,
            )
            for bar, relative_time in zip(bars, relative_times):
                if math.isnan(relative_time):
                    continue
                if variant_name == baseline_name:
                    label = "100%"
                else:
                    label = f"{relative_time - 100:+.0f}%"
                time_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    relative_time,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        time_axis.axhline(100, color="black", linewidth=1, linestyle="--", alpha=0.5)
        time_axis.set_ylim(0, max_relative_time * 1.18)
        time_axis.set_xticks(
            group_positions,
            [format_parameter_count(int(row["params"])) for row in baseline_rows],
            rotation=30,
        )
        time_axis.set_xlabel("Model parameters")
        time_axis.set_ylabel(
            f"Relative training step time ({baseline_label} = 100%)"
        )
        time_axis.set_title(f"Training Step Time vs {baseline_label}")
        time_axis.grid(axis="y", alpha=0.3)
        time_axis.legend()
    else:
        time_axis.text(
            0.5,
            0.5,
            f"No successful {VARIANTS[baseline_name].label} result for relative time.",
            ha="center",
            va="center",
            transform=time_axis.transAxes,
        )
        time_axis.set_axis_off()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def launcher(args: argparse.Namespace) -> None:
    if args.profile_only:
        if args.profile_variant is None:
            raise ValueError("--profile_only requires --profile_variant")
        profile_layers = args.profile_layers or args.layers[0]
        profile_variant = VARIANTS[args.profile_variant]
        print(
            f"profiling only: {profile_variant.label}, layers={profile_layers} "
            f"into {args.profile_dir}"
        )
        result = subprocess.run(
            profiler_command(args, profile_variant, profile_layers),
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("Profiler worker failed")
        print("saved profiler trace")
        return

    rows = []
    active = {variant_name: True for variant_name in args.variants}

    for layers in args.layers:
        params = count_parameters(args, layers)
        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            row: dict[str, object] = {
                "variant": variant_name,
                "label": variant.label,
                "layers": layers,
                "params": params,
                "peak_mib": "",
                "time_ms": "",
                "status": "skipped",
            }

            if active[variant_name]:
                status, metrics, output = run_subprocess(
                    worker_command(args, variant, layers)
                )
                row["status"] = status
                if status == "ok":
                    row["peak_mib"] = metrics["peak_mib"]
                    row["time_ms"] = metrics["time_ms"]
                    summary = (
                        f"{metrics['peak_mib']:.2f} MiB, "
                        f"{metrics['time_ms']:.2f} ms"
                    )
                else:
                    active[variant_name] = False
                    summary = status.upper()
                    if args.show_failures:
                        print(output)

                print(f"layers={layers:>3} {variant.label:<17} {summary}")

            rows.append(row)

        if not any(active.values()):
            break

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)
    save_csv(rows, csv_path)
    save_plot(rows, plot_path, args.variants)
    print(f"saved CSV:  {csv_path}")
    if plot_path.exists():
        print(f"saved plot: {plot_path}")

    if args.profile_variant is not None:
        profile_layers = args.profile_layers or args.layers[0]
        profile_variant = VARIANTS[args.profile_variant]
        print(
            f"profiling {profile_variant.label}, layers={profile_layers} "
            f"into {args.profile_dir}"
        )
        result = subprocess.run(
            profiler_command(args, profile_variant, profile_layers),
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("Profiler worker failed")
        print("saved profiler trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniMind dense and TP memory")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--seq_len", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--benchmark_iters", type=int, default=10)
    parser.add_argument(
        "--flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--show_failures", action="store_true")
    parser.add_argument("--output_csv", default="tp_memory_scaling.csv")
    parser.add_argument("--output_plot", default="tp_memory_scaling.png")
    parser.add_argument(
        "--profile_variant",
        choices=tuple(VARIANTS),
        default=None,
    )
    parser.add_argument("--profile_only", action="store_true")
    parser.add_argument("--profile_layers", type=int, default=None)
    parser.add_argument("--profile_dir", default="tp_profiler_traces")
    parser.add_argument("--profile_wait", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=1)
    parser.add_argument("--profile_active", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher(args)


if __name__ == "__main__":
    main()
