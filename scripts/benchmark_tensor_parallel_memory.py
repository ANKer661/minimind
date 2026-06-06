import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

RESULT_PREFIX = "TP_MEMORY_RESULT="


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


def run_subprocess(command: list[str]) -> tuple[bool, dict[str, float] | None, str]:
    result = subprocess.run(command, text=True, capture_output=True)
    output = result.stdout + result.stderr
    metrics = parse_result(output)
    return result.returncode == 0 and metrics is not None, metrics, output


def worker_command(
    args: argparse.Namespace,
    mode: str,
    num_hidden_layers: int,
) -> list[str]:
    worker = str(Path(__file__).with_name("benchmark_tensor_parallel_worker.py").resolve())
    common = [
        worker,
        "--mode",
        mode,
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
    if mode == "dense":
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
                "layers",
                "params",
                "dense_peak_mib",
                "tp_peak_mib",
                "dense_time_ms",
                "tp_time_ms",
                "dense_oom",
                "tp_oom",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    dense_rows = [row for row in rows if not row["dense_oom"]]
    tp_rows = [row for row in rows if not row["tp_oom"]]
    figure, (memory_axis, time_axis) = plt.subplots(1, 2, figsize=(13, 5))
    if dense_rows:
        memory_axis.plot(
            [row["params"] for row in dense_rows],
            [row["dense_peak_mib"] for row in dense_rows],
            marker="o",
            label="Dense (1 GPU)",
        )
        time_axis.plot(
            [row["params"] for row in dense_rows],
            [row["dense_time_ms"] for row in dense_rows],
            marker="o",
            label="Dense (1 GPU)",
        )
    if tp_rows:
        memory_axis.plot(
            [row["params"] for row in tp_rows],
            [row["tp_peak_mib"] for row in tp_rows],
            marker="o",
            label="TP peak per rank",
        )
        time_axis.plot(
            [row["params"] for row in tp_rows],
            [row["tp_time_ms"] for row in tp_rows],
            marker="o",
            label="TP slowest rank",
        )

    parameter_counts = sorted({int(row["params"]) for row in rows})
    parameter_labels = [format_parameter_count(value) for value in parameter_counts]
    for axis in (memory_axis, time_axis):
        axis.set_xscale("log")
        axis.set_xticks(parameter_counts, parameter_labels, rotation=30)
        axis.set_yscale("log")

    memory_axis.set_xlabel("Model parameters")
    memory_axis.set_ylabel("Peak allocated memory per GPU (MiB)")
    memory_axis.set_title("Peak GPU Memory")
    memory_axis.grid(which="both", alpha=0.3)
    memory_axis.legend()

    time_axis.set_xlabel("Model parameters")
    time_axis.set_ylabel("Average time per iteration (ms)")
    time_axis.set_title("Training Step Time")
    time_axis.grid(which="both", alpha=0.3)
    time_axis.legend()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def launcher(args: argparse.Namespace) -> None:
    rows = []
    dense_active = True
    tp_active = True

    for layers in args.layers:
        params = count_parameters(args, layers)
        row: dict[str, object] = {
            "layers": layers,
            "params": params,
            "dense_peak_mib": "",
            "tp_peak_mib": "",
            "dense_time_ms": "",
            "tp_time_ms": "",
            "dense_oom": not dense_active,
            "tp_oom": not tp_active,
        }

        if dense_active:
            ok, metrics, output = run_subprocess(worker_command(args, "dense", layers))
            dense_active = ok
            row["dense_oom"] = not ok
            row["dense_peak_mib"] = metrics["peak_mib"] if ok else ""
            row["dense_time_ms"] = metrics["time_ms"] if ok else ""
            dense_summary = (
                "OOM/failed"
                if not ok
                else f"{metrics['peak_mib']:.2f} MiB, {metrics['time_ms']:.2f} ms"
            )
            print(f"layers={layers:>3} dense: {dense_summary}")
            if not ok and args.show_failures:
                print(output)

        if tp_active:
            ok, metrics, output = run_subprocess(worker_command(args, "tp", layers))
            tp_active = ok
            row["tp_oom"] = not ok
            row["tp_peak_mib"] = metrics["peak_mib"] if ok else ""
            row["tp_time_ms"] = metrics["time_ms"] if ok else ""
            tp_summary = (
                "OOM/failed"
                if not ok
                else f"{metrics['peak_mib']:.2f} MiB, {metrics['time_ms']:.2f} ms"
            )
            print(f"layers={layers:>3} TP:    {tp_summary}")
            if not ok and args.show_failures:
                print(output)

        rows.append(row)
        if not dense_active and not tp_active:
            break

    csv_path = Path(args.output_csv)
    plot_path = Path(args.output_plot)
    save_csv(rows, csv_path)
    save_plot(rows, plot_path)
    print(f"saved CSV:  {csv_path}")
    if plot_path.exists():
        print(f"saved plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniMind dense and TP memory")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32, 40])
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
    parser.add_argument("--show_failures", action="store_true")
    parser.add_argument("--output_csv", default="tp_memory_scaling.csv")
    parser.add_argument("--output_plot", default="tp_memory_scaling.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher(args)


if __name__ == "__main__":
    main()
