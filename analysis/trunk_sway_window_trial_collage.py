"""Render the matched four-panel trunk-sway benchmark."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

from analysis.tifs_final_style import (
    BASELINE_LIGHT,
    BASELINE_MID,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    TEXT_COLOR,
    VALUE_SIZE,
    apply_final_style,
)
from analysis.trunk_sway_prism_gain_figure import (
    CONDITIONS,
    DATA_DIR,
    MM,
    MODEL_ORDER,
    ROOT,
    load_evidence,
)


OUTPUT_PATH = ROOT / "final figures" / "sway benchmark.pdf"
STATISTICS_PATH = DATA_DIR / "sway_benchmark_prism_comparisons.csv"
INTERVAL_PATH = DATA_DIR / "sway_benchmark_accuracy_intervals.csv"
CAPTION_PATH = DATA_DIR / "sway_benchmark_caption.txt"
ROWS = (
    ("Window level", "Window level", (60, 104)),
    ("Trial level (1 min)", "Trial level (1 min)", (75, 104)),
)
MODEL_DISPLAY_NAMES = {
    "PRISM-Gait": "PRISM-Gait",
    "MultiRocket": "MultiROCKET",
    "MiniRocket": "MiniROCKET",
    "FCN-BiLSTM": "FCN-BiLSTM",
    "InceptionTime": "InceptionTime",
}


def create_logger() -> logging.Logger:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"trunk_sway_collage_{stamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        DATA_DIR / f"trunk_sway_collage_{stamp}.log",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def load_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not STATISTICS_PATH.is_file() or not INTERVAL_PATH.is_file():
        raise FileNotFoundError(
            "run create_sway_benchmark_statistics.py before rendering"
        )
    statistics = pd.read_csv(STATISTICS_PATH)
    intervals = pd.read_csv(INTERVAL_PATH)
    expected_comparisons = 2 * 2 * (len(MODEL_ORDER) - 1)
    expected_intervals = 2 * 2 * len(MODEL_ORDER)
    if len(statistics) != expected_comparisons:
        raise ValueError("unexpected number of statistical comparisons")
    if len(intervals) != expected_intervals:
        raise ValueError("unexpected number of accuracy intervals")
    return statistics, intervals


def _panel_rows(
    intervals: pd.DataFrame,
    aggregation: str,
    condition: str,
) -> pd.DataFrame:
    rows = (
        intervals.loc[
            intervals["aggregation"].eq(aggregation)
            & intervals["condition"].eq(condition)
        ]
        .set_index("model")
        .reindex(MODEL_ORDER)
    )
    if rows[["accuracy_pct", "ci_lower_pct", "ci_upper_pct"]].isna().any().any():
        raise ValueError(f"incomplete interval evidence for {aggregation}, {condition}")
    return rows


def draw_panel(
    axis,
    intervals: pd.DataFrame,
    statistics: pd.DataFrame,
    aggregation: str,
    condition: str,
    limits: tuple[float, float],
    show_models: bool,
) -> None:
    rows = _panel_rows(intervals, aggregation, condition)
    comparisons = statistics.loc[
        statistics["aggregation"].eq(aggregation)
        & statistics["condition"].eq(condition)
    ].set_index("comparator")
    comparators = MODEL_ORDER[1:]
    y = np.arange(len(MODEL_ORDER))
    prism_value = float(rows.loc["PRISM-Gait", "accuracy_pct"])
    gain_x = prism_value + 0.7
    for row_index in y:
        if row_index == 0:
            band_color = PRISM_COLOR
            band_alpha = 0.055
        elif row_index % 2:
            band_color = BASELINE_LIGHT
            band_alpha = 0.20
        else:
            continue
        axis.axhspan(
            row_index - 0.44,
            row_index + 0.44,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            zorder=0,
        )
    axis.scatter(
        prism_value,
        0,
        s=62,
        marker="D",
        color=PRISM_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axis.text(
        prism_value - 0.55,
        0,
        f"{prism_value:.1f}",
        ha="right",
        va="center",
        fontsize=VALUE_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    for index, model in enumerate(comparators, start=1):
        value = float(rows.loc[model, "accuracy_pct"])
        delta = prism_value - value
        significance = comparisons.loc[model, "significance"]
        axis.plot(
            [value, prism_value],
            [index, index],
            color=BASELINE_LIGHT,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
        axis.scatter(
            value,
            index,
            s=44,
            marker="o",
            color=BASELINE_MID,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.scatter(
            prism_value,
            index,
            s=35,
            marker="D",
            color=PRISM_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.text(
            value - 0.55,
            index,
            f"{value:.1f}",
            ha="right",
            va="center",
            fontsize=VALUE_SIZE,
            color=TEXT_COLOR,
        )
        axis.text(
            gain_x,
            index,
            f"+{delta:.1f} {significance}",
            ha="left",
            va="center",
            fontsize=VALUE_SIZE,
            fontweight="bold",
            color=TEXT_COLOR,
            zorder=4,
        )

    axis.set_xlim(*limits)
    major_ticks = np.arange(limits[0], 101, 5)
    minor_ticks = [
        value
        for value in np.arange(limits[0], 101, 1)
        if value not in set(major_ticks)
    ]
    axis.set_xticks(major_ticks)
    axis.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    axis.spines["bottom"].set_bounds(limits[0], 100)
    axis.set_ylim(len(MODEL_ORDER) - 0.48, -0.68)
    axis.set_yticks(y)
    axis.set_yticklabels([MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER])
    axis.tick_params(
        axis="y",
        left=show_models,
        labelleft=show_models,
        length=3,
    )
    if show_models:
        axis.get_yticklabels()[0].set_fontweight("bold")
    axis.tick_params(which="minor", axis="y", left=False)


def _write_caption() -> None:
    CAPTION_PATH.write_text(
        (
            "Trunk-sway identification benchmark. All panels use the same five model "
            "rows, with PRISM-Gait shown first. Each dumbbell links a benchmark "
            "model's accuracy (gray circle; adjacent value) to the PRISM-Gait "
            "reference (navy diamond). Labels to the right of the PRISM-Gait endpoint "
            "report the improvement in percentage points and the Holm-adjusted "
            "significance level. Pairwise comparisons used two-sided paired Wilcoxon "
            "signed-rank tests with Pratt "
            "handling of zero differences for participant window accuracies and exact "
            "two-sided McNemar tests for paired trial decisions. Holm adjustment was "
            "applied across the four PRISM-Gait-versus-comparator tests within each "
            "panel. Small sway: n = 61 paired trials; large sway: n = 59 paired trials. "
            "* P < .05; ** P < .01; *** P < .001; ns, not significant."
        ),
        encoding="utf-8",
    )


def build_collage(
    intervals: pd.DataFrame,
    statistics: pd.DataFrame,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(180 * MM, 104 * MM),
        sharey=True,
    )
    panel_index = 0
    for row, (aggregation, row_label, limits) in enumerate(ROWS):
        for column, (condition, _condition_label) in enumerate(CONDITIONS):
            axis = axes[row, column]
            draw_panel(
                axis,
                intervals,
                statistics,
                aggregation,
                condition,
                limits,
                show_models=column == 0,
            )
            axis.text(
                0.0,
                1.045,
                f"({chr(97 + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=PANEL_LABEL_SIZE,
                fontweight="bold",
            )
            panel_index += 1
        figure.text(
            0.018,
        0.615 if row == 0 else 0.255,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.5,
            color=TEXT_COLOR,
        )

    figure.text(
        0.305,
        0.925,
        "Small trunk sway",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=TEXT_COLOR,
    )
    figure.text(
        0.745,
        0.925,
        "Large trunk sway",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=TEXT_COLOR,
    )
    figure.supxlabel("Accuracy (%)", x=0.54, y=0.035, fontsize=9.5)
    figure.subplots_adjust(
        left=0.17,
        right=0.985,
        bottom=0.14,
        top=0.85,
        hspace=0.36,
        wspace=0.08,
    )
    return figure


def render() -> Path:
    apply_final_style()
    logger = create_logger()
    logger.info("step_01: loading benchmark evidence and paired comparisons")
    load_evidence()
    statistics, intervals = load_statistics()
    logger.info("step_02: rendering four matched point-and-interval panels")
    figure = build_collage(intervals, statistics)
    figure.savefig(OUTPUT_PATH, format="pdf")
    plt.close(figure)
    _write_caption()
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"figure export failed: {OUTPUT_PATH.name}")
    logger.info("step_03: verified PDF bytes=%s", OUTPUT_PATH.stat().st_size)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render())
