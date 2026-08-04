import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evaluation.presets import MODE_PRESETS, ResolvedMode


MODE_LABELS = {name: preset.label for name, preset in MODE_PRESETS.items()}


def build_result_row(
    args: argparse.Namespace,
    mode: ResolvedMode,
    metrics: dict[str, object] | None,
    *,
    scan_dimension: str,
    num_hidden_layers: int,
    seq_len: int,
    num_microbatches: int,
    model_batch_size: int,
    params: int,
    status: str,
) -> dict[str, object]:
    return {
        "mode": mode.name,
        "label": mode.label,
        "runner": mode.runner,
        "pp_schedule": args.pp_schedule,
        "pp_size": mode.pp_size,
        "tp_size": mode.tp_size,
        "cp_enabled": mode.cp_enabled,
        "cp_size": mode.cp_size,
        "cp_comm_type": args.cp_comm_type if mode.cp_enabled else "",
        "sequence_parallel": mode.sequence_parallel,
        "async_communication": mode.async_communication,
        "vocab_parallel": mode.vocab_parallel,
        "world_size": mode.world_size,
        "batch_policy": args.batch_policy,
        "model_batch_size": (
            metrics["model_batch_size"] if metrics else model_batch_size
        ),
        "global_batch_size": metrics["global_batch_size"] if metrics else math.nan,
        "scaling": scan_dimension,
        "layers": num_hidden_layers,
        "seq_len": seq_len,
        "num_microbatches": num_microbatches,
        "params": params,
        "peak_mib": metrics["peak_mib"] if metrics else math.nan,
        "time_ms": metrics["time_ms"] if metrics else math.nan,
        "tokens_per_second": (
            metrics["tokens_per_second"] if metrics else math.nan
        ),
        "worker_pp_schedule": (
            metrics["pp_schedule"] if metrics else args.pp_schedule
        ),
        "status": status,
    }


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def format_tokens_per_second(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    return f"{value / 1_000_000:.0f}M"


@dataclass(frozen=True)
class PlotPanel:
    metric: str
    title: str
    ylabel: str
    relative: bool
    chart: Literal["bar", "line"]
    format_tokens: bool = False


def _plot_panels(
    kind: str,
    scaling: str,
    baseline_label: str,
) -> tuple[PlotPanel, PlotPanel]:
    if kind == "throughput":
        return (
            PlotPanel(
                "tokens_per_second",
                "End-to-End Training Throughput",
                "Training tokens/s",
                False,
                "bar",
                format_tokens=True,
            ),
            PlotPanel(
                "tokens_per_second",
                "Relative Throughput",
                f"Throughput relative to {baseline_label} (%)",
                True,
                "bar",
            ),
        )
    if scaling == "sequence":
        return (
            PlotPanel(
                "peak_mib",
                "Peak GPU Memory by Sequence Length",
                "Peak GPU memory (MiB)",
                False,
                "line",
            ),
            PlotPanel(
                "time_ms",
                "Step Time by Sequence Length",
                "End-to-end step time (ms)",
                False,
                "line",
            ),
        )
    if scaling == "layers":
        return (
            PlotPanel(
                "peak_mib",
                "Relative Peak GPU Memory",
                f"Peak memory relative to {baseline_label} (%)",
                True,
                "bar",
            ),
            PlotPanel(
                "time_ms",
                "Relative End-to-End Step Time",
                f"Step time relative to {baseline_label} (%)",
                True,
                "bar",
            ),
        )
    raise ValueError(f"Unknown kind: {kind}")


def _metric_values(
    rows: list[dict[str, object]],
    mode_names: list[str],
    scale_key: str,
    metric: str,
) -> dict[str, dict[int, float]]:
    return {
        mode: {
            int(row[scale_key]): float(row[metric])
            for row in rows
            if row["mode"] == mode and row["status"] == "ok"
        }
        for mode in mode_names
    }


def save_plot(
    rows: list[dict[str, object]],
    path: Path,
    mode_names: list[str],
    scaling: str,
    kind: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    scale_key = "layers" if scaling == "layers" else "seq_len"
    scale_values = sorted({int(row[scale_key]) for row in rows})
    if scaling == "layers":
        params_by_layer = {
            int(row["layers"]): int(row["params"])
            for row in rows
        }
        scale_labels = [
            _format_parameter_count(params_by_layer[value])
            for value in scale_values
        ]
        x_label = "Model parameters"
    else:
        scale_labels = [f"{value:,}" for value in scale_values]
        x_label = "Sequence length"

    baseline_mode = "ddp" if "ddp" in mode_names else mode_names[0]
    panels = _plot_panels(kind, scaling, MODE_LABELS[baseline_mode])
    positions = list(range(len(scale_values)))
    bar_width = 0.8 / len(mode_names)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, panel in zip(axes, panels):
        values_by_mode = _metric_values(
            rows,
            mode_names,
            scale_key,
            panel.metric,
        )
        baseline_values = values_by_mode[baseline_mode]
        max_value = 100.0
        for mode_index, mode in enumerate(mode_names):
            mode_values = values_by_mode[mode]
            values = [
                mode_values.get(value, math.nan) / baseline_values[value] * 100
                if panel.relative and value in baseline_values and value in mode_values
                else mode_values.get(value, math.nan)
                if not panel.relative
                else math.nan
                for value in scale_values
            ]
            valid_values = [value for value in values if not math.isnan(value)]
            if valid_values:
                max_value = max(max_value, *valid_values)

            if panel.chart == "bar":
                offsets = [
                    position
                    + (mode_index - (len(mode_names) - 1) / 2) * bar_width
                    for position in positions
                ]
                axis.bar(
                    offsets,
                    values,
                    width=bar_width,
                    label=MODE_LABELS[mode],
                )
            else:
                axis.plot(
                    positions,
                    values,
                    marker="o",
                    label=MODE_LABELS[mode],
                )

        if panel.relative:
            axis.axhline(100, color="black", linewidth=1, linestyle="--")
            axis.set_ylim(0, max_value * 1.18)
        if panel.format_tokens:
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: format_tokens_per_second(value))
            )
        axis.set_xticks(positions, scale_labels, rotation=30)
        axis.set_xlabel(x_label)
        axis.set_ylabel(panel.ylabel)
        axis.set_title(panel.title)
        axis.grid(axis="y" if panel.chart == "bar" else "both", alpha=0.3)
        axis.legend()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
