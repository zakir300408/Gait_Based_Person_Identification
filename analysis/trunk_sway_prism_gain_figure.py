"""Render separate window- and trial-level trunk-sway benchmark figures."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.tifs_final_style import (
    BASELINE_MID,
    HEADER_SIZE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    TEXT_COLOR,
    apply_final_style,
)


ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = Path(__file__).with_name("publication-hybrid.mplstyle")
DATA_DIR = ROOT / "baselines_out" / "treadmill_published_accuracy_tables"
TABLE_PATHS = {
    "Window level": DATA_DIR / "table_window_accuracy_published_vs_prism.csv",
    "Trial level (1 min)": DATA_DIR / "table_trial_accuracy_published_vs_prism.csv",
}
OUTPUT_PATHS = {
    "Window level": DATA_DIR / "fig_trunk_sway_window_accuracy_bars.pdf",
    "Trial level (1 min)": DATA_DIR / "fig_trunk_sway_trial_accuracy_bars.pdf",
}
EVIDENCE_PATH = DATA_DIR / "fig_trunk_sway_accuracy_values.csv"
MM = 1 / 25.4

MODEL_ORDER = (
    "PRISM-Gait",
    "MultiRocket",
    "MiniRocket",
    "FCN-BiLSTM",
    "InceptionTime",
)
CONDITIONS = (("Trunk sway small", "Small trunk sway"), ("Trunk sway large", "Large trunk sway"))
Y_LIMITS = {"Window level": (60, 101), "Trial level (1 min)": (75, 101)}
BASELINE_COLOR = BASELINE_MID


def create_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"trunk_sway_accuracy_{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(DATA_DIR / f"trunk_sway_accuracy_{timestamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_evidence() -> pd.DataFrame:
    frames = []
    required_columns = {"Method", *(column for column, _ in CONDITIONS)}
    for aggregation, path in TABLE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"required accuracy table is missing: {path.name}")
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["Method"].isin(MODEL_ORDER)]
        if set(frame["Method"]) != set(MODEL_ORDER):
            raise ValueError(f"{path.name} does not contain the expected method roster")
        long = frame.melt(
            id_vars="Method",
            value_vars=[column for column, _ in CONDITIONS],
            var_name="Condition",
            value_name="Accuracy (%)",
        )
        long["Aggregation"] = aggregation
        frames.append(long)
    evidence = pd.concat(frames, ignore_index=True)
    if evidence["Accuracy (%)"].isna().any():
        raise ValueError("accuracy values must be complete")
    return evidence


def draw_panel(axis, evidence: pd.DataFrame, condition: str, label: str, limits) -> None:
    rows = (
        evidence.loc[evidence["Condition"].eq(condition)]
        .set_index("Method")
        .reindex(MODEL_ORDER)
    )
    values = rows["Accuracy (%)"].to_numpy(dtype=float)
    if values.min() <= limits[0]:
        raise ValueError(f"lower axis limit hides a value in {condition}")
    x = np.arange(len(MODEL_ORDER))
    colors = [
        PRISM_COLOR if model == "PRISM-Gait" else BASELINE_COLOR
        for model in MODEL_ORDER
    ]
    bars = axis.bar(
        x,
        values - limits[0],
        bottom=limits[0],
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, value, model in zip(bars, values, MODEL_ORDER):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.55,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if model == "PRISM-Gait" else "normal",
            color=TEXT_COLOR,
        )
    axis.set_xticks(x, MODEL_ORDER)
    axis.tick_params(axis="x", labelsize=8.5)
    for tick in axis.get_xticklabels():
        tick.set_rotation(55)
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")
        tick.set_wrap(False)
    axis.set_ylim(*limits)
    axis.set_yticks(np.arange(limits[0], 101, 5))
    axis.minorticks_off()
    axis.text(
        0.5,
        1.035,
        label,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=HEADER_SIZE,
    )
    for tick, model in zip(axis.get_xticklabels(), MODEL_ORDER):
        if model == "PRISM-Gait":
            tick.set_fontweight("bold")
    mark = 0.012
    axis.plot(
        (-mark, mark),
        (-mark, mark),
        transform=axis.transAxes,
        color=TEXT_COLOR,
        linewidth=0.8,
        clip_on=False,
    )


def build_figure(evidence: pd.DataFrame, aggregation: str):
    subset = evidence.loc[evidence["Aggregation"].eq(aggregation)]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 * MM, 80 * MM),
        sharey=True,
    )
    for index, (axis, (condition, label)) in enumerate(zip(axes, CONDITIONS)):
        draw_panel(axis, subset, condition, label, Y_LIMITS[aggregation])
        axis.text(
            0.01,
            1.035,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )
    axes[0].set_ylabel(
        "Window accuracy (%)"
        if aggregation == "Window level"
        else "Trial accuracy (%)"
    )
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.legend(
        handles=[
            Patch(facecolor=BASELINE_COLOR, label="Benchmark models"),
            Patch(facecolor=PRISM_COLOR, label="PRISM-Gait"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.35, top=0.79, wspace=0.10)
    return fig


def render() -> tuple[Path, ...]:
    apply_final_style()
    logger = create_logger()
    logger.info("step_01: loading separated window and trial results")
    evidence = load_evidence()
    evidence["Display order"] = evidence["Method"].map(
        {model: index + 1 for index, model in enumerate(MODEL_ORDER)}
    )
    evidence.to_csv(EVIDENCE_PATH, index=False)
    logger.info("step_02: wrote figure evidence rows=%s", len(evidence))
    for aggregation, output_path in OUTPUT_PATHS.items():
        figure = build_figure(evidence, aggregation)
        figure.savefig(output_path, format="pdf")
        plt.close(figure)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FileNotFoundError(f"figure export failed: {output_path.name}")
        logger.info("step_03: verified %s bytes=%s", output_path.name, output_path.stat().st_size)
    return tuple(OUTPUT_PATHS.values())


if __name__ == "__main__":
    print("\n".join(str(path) for path in render()))
