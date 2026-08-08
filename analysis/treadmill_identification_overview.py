"""Render the treadmill identification benchmark and operational timing."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from analysis.trunk_sway_window_trial_collage import draw_panel as draw_sway_panel
from analysis.tifs_final_style import (
    AXIS_LABEL_SIZE,
    LARGE_SWAY_COLOR,
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    SMALL_SWAY_COLOR,
    TICK_SIZE,
    VALUE_SIZE,
    apply_final_style,
)


ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = Path(__file__).with_name("publication-hybrid.mplstyle")
DATA_DIR = ROOT / "baselines_out" / "treadmill_published_accuracy_tables"
TIMING_DIR = ROOT / "baselines_out" / "treadmill_published_plot_data"

SWAY_EVIDENCE_PATH = DATA_DIR / "fig_trunk_sway_accuracy_values.csv"
SWAY_STATISTICS_PATH = DATA_DIR / "sway_benchmark_prism_comparisons.csv"
SWAY_INTERVAL_PATH = DATA_DIR / "sway_benchmark_accuracy_intervals.csv"
NON_SWAY_EVIDENCE_PATH = DATA_DIR / "fig_non_sway_accuracy_values.csv"
NON_SWAY_STATISTICS_PATH = DATA_DIR / "fig_non_sway_prism_pairwise_statistics.csv"
TIMING_PATH = TIMING_DIR / "fig_non_sway_operational_timing_evidence.csv"

OUTPUT_PATH = ROOT / "final figures" / "treadmill identification benchmark.pdf"
TIMING_OUTPUT_PATH = ROOT / "final figures" / "operational timing.pdf"
EVIDENCE_PATH = DATA_DIR / "fig_treadmill_identification_overview_values.csv"
STATISTICS_PATH = DATA_DIR / "fig_treadmill_identification_overview_statistics.csv"
CAPTION_PATH = DATA_DIR / "fig_treadmill_identification_overview_caption.txt"

MM = 1 / 25.4
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 100.0
TIMING_WIDTH_MM = 89.0
TIMING_HEIGHT_MM = 72.0

MODEL_ORDER = (
    "PRISM-Gait",
    "MultiRocket",
    "MiniRocket",
    "FCN-BiLSTM",
    "InceptionTime",
)
MODEL_DISPLAY_NAMES = {
    "PRISM-Gait": "PRISM-Gait (Ours)",
    "MultiRocket": "MultiROCKET",
    "MiniRocket": "MiniROCKET",
    "FCN-BiLSTM": "FCN-BiLSTM",
    "InceptionTime": "InceptionTime",
}
SWAY_CONDITIONS = (
    ("trunk_sway_small", "Sway\nsmall"),
    ("trunk_sway_large", "Sway\nlarge"),
)
CONDITIONS = (
    ("fpa_small", "FPA\nsmall"),
    ("fpa_large", "FPA\nlarge"),
    ("step_width_small", "Step\nwidth\nsmall"),
    ("step_width_large", "Step\nwidth\nlarge"),
    ("walk_speed_small", "Speed\nsmall"),
    ("walk_speed_large", "Speed\nlarge"),
    ("dual_task", "Dual\ntask"),
    ("non_sway_overall", "Non-sway\noverall"),
)
ALL_CONDITIONS = (*SWAY_CONDITIONS, *CONDITIONS)
SWAY_CONDITION_MAP = {
    "Trunk sway small": "trunk_sway_small",
    "Trunk sway large": "trunk_sway_large",
}
AGGREGATIONS = ("Window level", "Trial level (1 min)")

BEST_COLOR = "#2A9D8F"
SECOND_COLOR = "#B9D9D4"
NEUTRAL_COLOR = "#F0F1F1"
TEXT_COLOR = "#1A1A1A"
MID_GRAY = "#7F7F7F"


def create_logger() -> logging.Logger:
    """Create a timestamped audit log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"treadmill_identification_overview_{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        DATA_DIR / f"treadmill_identification_overview_{timestamp}.log",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def significance_label(p_value: float) -> str:
    """Format an adjusted p-value using the fixed publication thresholds."""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_evidence() -> pd.DataFrame:
    """Combine the audited sway and non-sway accuracy evidence."""
    sway = pd.read_csv(SWAY_EVIDENCE_PATH).rename(columns={"Method": "model"})
    sway["condition"] = sway["Condition"].map(SWAY_CONDITION_MAP)
    sway = sway.rename(
        columns={
            "Aggregation": "aggregation",
            "Accuracy (%)": "accuracy_pct",
        }
    )[["model", "aggregation", "condition", "accuracy_pct"]]

    non_sway = pd.read_csv(NON_SWAY_EVIDENCE_PATH).rename(
        columns={
            "Method": "model",
            "Aggregation": "aggregation",
            "Condition": "condition",
            "Accuracy (%)": "accuracy_pct",
        }
    )[["model", "aggregation", "condition", "accuracy_pct"]]

    evidence = pd.concat([sway, non_sway], ignore_index=True)
    evidence = evidence.loc[
        evidence["model"].isin(MODEL_ORDER)
        & evidence["aggregation"].isin(AGGREGATIONS)
        & evidence["condition"].isin([key for key, _ in ALL_CONDITIONS])
    ].copy()
    expected = {
        (model, aggregation, condition)
        for model in MODEL_ORDER
        for aggregation in AGGREGATIONS
        for condition, _ in ALL_CONDITIONS
    }
    observed = set(
        evidence[["model", "aggregation", "condition"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(f"incomplete combined evidence; missing={missing}, extra={extra}")
    evidence["rank"] = (
        evidence.groupby(["aggregation", "condition"])["accuracy_pct"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return evidence


def load_statistics() -> pd.DataFrame:
    """Combine adjusted PRISM-Gait pairwise comparisons for displayed cells."""
    sway = pd.read_csv(SWAY_STATISTICS_PATH)
    sway["condition"] = sway["condition"].map(SWAY_CONDITION_MAP)
    sway = sway.rename(
        columns={
            "aggregation": "aggregation",
            "comparator": "comparison",
            "p_adjusted": "p_adjusted",
            "test": "test",
            "correction": "correction",
        }
    )[["aggregation", "condition", "comparison", "p_adjusted", "test", "correction"]]

    non_sway = pd.read_csv(NON_SWAY_STATISTICS_PATH).rename(
        columns={"p_holm": "p_adjusted"}
    )[["aggregation", "condition", "comparison", "p_adjusted", "test", "correction"]]
    statistics = pd.concat([sway, non_sway], ignore_index=True)
    statistics = statistics.loc[
        statistics["comparison"].isin(MODEL_ORDER[1:])
        & statistics["aggregation"].isin(AGGREGATIONS)
        & statistics["condition"].isin([key for key, _ in ALL_CONDITIONS])
    ].copy()
    expected = 2 * len(ALL_CONDITIONS) * (len(MODEL_ORDER) - 1)
    if len(statistics) != expected:
        raise ValueError(f"expected {expected} pairwise comparisons, found {len(statistics)}")
    statistics["annotation"] = statistics["p_adjusted"].map(significance_label)
    return statistics


def add_axes_mm(
    figure,
    left_mm: float,
    bottom_mm: float,
    width_mm: float,
    height_mm: float,
):
    """Add an axes using physical millimetre coordinates."""
    return figure.add_axes(
        [
            left_mm / FIGURE_WIDTH_MM,
            bottom_mm / FIGURE_HEIGHT_MM,
            width_mm / FIGURE_WIDTH_MM,
            height_mm / FIGURE_HEIGHT_MM,
        ]
    )


def draw_accuracy_panel(
    axis,
    evidence: pd.DataFrame,
    statistics: pd.DataFrame,
    aggregation: str,
    panel_label: str,
    show_condition_labels: bool,
) -> None:
    """Draw one model-by-condition accuracy matrix."""
    condition_order = [key for key, _ in CONDITIONS]
    subset = evidence.loc[evidence["aggregation"].eq(aggregation)]
    values = (
        subset.pivot(index="model", columns="condition", values="accuracy_pct")
        .reindex(MODEL_ORDER)[condition_order]
    )
    ranks = (
        subset.pivot(index="model", columns="condition", values="rank")
        .reindex(MODEL_ORDER)[condition_order]
    )
    stats = statistics.loc[statistics["aggregation"].eq(aggregation)].set_index(
        ["condition", "comparison"]
    )

    for row, model in enumerate(MODEL_ORDER):
        for column, condition in enumerate(condition_order):
            rank = int(ranks.loc[model, condition])
            face = BEST_COLOR if rank == 1 else SECOND_COLOR if rank == 2 else NEUTRAL_COLOR
            axis.add_patch(
                Rectangle(
                    (column - 0.48, row - 0.43),
                    0.96,
                    0.86,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=0.75,
                )
            )
            annotation = ""
            if model != "PRISM-Gait":
                annotation = str(stats.loc[(condition, model), "annotation"])
            value_y = row - 0.10 if annotation else row
            axis.text(
                column,
                value_y,
                f"{values.loc[model, condition]:.1f}",
                ha="center",
                va="center",
                fontsize=VALUE_SIZE,
                fontweight="bold" if rank <= 2 else "normal",
                color="white" if rank == 1 else TEXT_COLOR,
            )
            if annotation:
                axis.text(
                    column,
                    row + 0.20,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=VALUE_SIZE,
                    fontweight="bold",
                    color="white" if rank == 1 else TEXT_COLOR,
                )

    axis.axvline(6.5, color="#A9AFB1", linewidth=0.8, clip_on=False)
    axis.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    axis.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
    axis.set_xticks(np.arange(len(CONDITIONS)))
    axis.set_xticklabels([label for _, label in CONDITIONS])
    axis.set_yticks(
        np.arange(len(MODEL_ORDER)),
        [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER],
    )
    axis.minorticks_off()
    axis.tick_params(axis="x", length=0, pad=4, labelsize=TICK_SIZE)
    axis.tick_params(axis="y", length=0, pad=5, labelsize=TICK_SIZE)
    axis.tick_params(axis="x", labelbottom=show_condition_labels)
    for tick, model in zip(axis.get_yticklabels(), MODEL_ORDER):
        tick.set_fontweight("bold" if model == "PRISM-Gait" else "normal")
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.text(
        -0.105,
        1.04,
        f"({panel_label})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )


def draw_non_sway_profiles(axis, evidence: pd.DataFrame) -> None:
    """Draw restrained model profiles across the secondary perturbations."""
    condition_order = [key for key, _ in CONDITIONS]
    condition_labels = [label for _, label in CONDITIONS]
    values = (
        evidence.loc[evidence["aggregation"].eq("Window level")]
        .pivot(index="model", columns="condition", values="accuracy_pct")
        .reindex(MODEL_ORDER)[condition_order]
    )
    x_conditions = np.arange(len(condition_order) - 1, dtype=float)
    model_styles = {
        "PRISM-Gait": (PRISM_COLOR, "-", "o", 1.45, 5),
        "MultiRocket": ("#6C8EAD", (0, (7, 3)), "D", 1.0, 3),
        "MiniRocket": ("#D59B47", "-.", "^", 1.0, 3),
        "FCN-BiLSTM": ("#C97B63", ":", "s", 1.0, 3),
        "InceptionTime": ("#7FA07B", (0, (1, 2)), "v", 1.0, 3),
    }
    for model in MODEL_ORDER:
        color, linestyle, marker, linewidth, zorder = model_styles[model]
        row = values.loc[model].to_numpy(dtype=float)
        axis.plot(
            x_conditions,
            row[:-1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=3.7,
            markerfacecolor="white" if model != "PRISM-Gait" else color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=MODEL_DISPLAY_NAMES[model],
            zorder=zorder,
        )
    axis.set_xlim(-0.35, 6.35)
    axis.set_ylim(90.5, 100.2)
    axis.set_yticks((92, 94, 96, 98, 100))
    axis.set_ylabel("Window-level accuracy (%)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xticks(x_conditions, condition_labels[:-1])
    axis.tick_params(axis="x", length=0, pad=4, labelsize=TICK_SIZE)
    axis.tick_params(axis="y", labelsize=TICK_SIZE)
    axis.minorticks_on()
    axis.tick_params(which="both", top=False, right=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.8,
        markerscale=1.35,
        handletextpad=0.3,
        columnspacing=0.6,
    )


def draw_sway_vertical_bars(
    axis,
    evidence: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw vertical grouped bars for the two sway conditions."""
    condition_specs = (
        ("trunk_sway_small", -0.18, SMALL_SWAY_COLOR, "#1F786F"),
        ("trunk_sway_large", 0.18, LARGE_SWAY_COLOR, "#6E526B"),
    )
    subset = evidence.loc[evidence["aggregation"].eq("Window level")]
    stats = statistics.loc[statistics["aggregation"].eq("Window level")].set_index(
        ["condition", "comparison"]
    )
    positions = np.arange(len(MODEL_ORDER), dtype=float)
    bar_centers: dict[tuple[str, str], float] = {}
    bar_tops: dict[tuple[str, str], float] = {}
    for condition, offset, face, edge in condition_specs:
        values = np.asarray(
            [
                float(
                    subset.loc[
                        subset["condition"].eq(condition)
                        & subset["model"].eq(model),
                        "accuracy_pct",
                    ].iloc[0]
                )
                for model in MODEL_ORDER
            ]
        )
        bars = axis.bar(
            positions + offset,
            values - 60.0,
            bottom=60.0,
            width=0.32,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.7,
            zorder=2,
        )
        for bar, model, value in zip(bars, MODEL_ORDER, values):
            highlight = model == "PRISM-Gait"
            if highlight:
                bar.set_edgecolor(TEXT_COLOR)
                bar.set_linewidth(1.15)
            center = bar.get_x() + bar.get_width() / 2
            bar_centers[(condition, model)] = center
            bar_tops[(condition, model)] = value
            # Keep small-sway values centered. Nudge only the large-sway value
            # slightly right so it remains close to its bar center while no
            # longer leaning into the neighboring small-sway bar.
            label_offset = 0.055 if condition == "trunk_sway_large" else 0.0
            axis.text(
                center + label_offset,
                value + 0.45,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold" if highlight else "normal",
                color=edge,
            )
    # Bundle each model's two condition bars under a short group cap, then draw
    # one PRISM-to-comparator bracket per model. The single star string is used
    # only when both condition-specific adjusted tests have the same label.
    reference = "PRISM-Gait"
    comparison_order = MODEL_ORDER[1:]
    group_cap_y: dict[str, float] = {}
    group_center_x: dict[str, float] = {}
    # Reserve enough physical clearance for the 9 pt value labels above bars.
    # At the final 58 mm axis height, this leaves about 2.5--3 mm between each
    # label and the nearest cap segment.
    cap_leg = 0.35
    cap_clearance = 4.30
    for model in MODEL_ORDER:
        small_x = bar_centers[("trunk_sway_small", model)]
        large_x = bar_centers[("trunk_sway_large", model)]
        cap_y = max(
            bar_tops[("trunk_sway_small", model)],
            bar_tops[("trunk_sway_large", model)],
        ) + cap_clearance
        group_cap_y[model] = cap_y
        group_center_x[model] = (small_x + large_x) / 2
        axis.plot(
            [small_x, small_x, large_x, large_x],
            [cap_y - cap_leg, cap_y, cap_y, cap_y - cap_leg],
            color=TEXT_COLOR,
            linewidth=0.9,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )

    bracket_base = max(group_cap_y.values()) + 2.15
    comparison_step = 3.15
    for level, comparison in enumerate(comparison_order):
        condition_labels = [
            significance_label(float(stats.loc[(condition, comparison), "p_adjusted"]))
            for condition in ("trunk_sway_small", "trunk_sway_large")
        ]
        if condition_labels[0] != condition_labels[1]:
            raise ValueError(
                f"grouped bracket requires matching condition labels for {comparison}: "
                f"{condition_labels}"
            )
        annotation = condition_labels[0]
        reference_x = group_center_x[reference]
        comparison_x = group_center_x[comparison]
        bracket_y = bracket_base + level * comparison_step
        axis.plot(
            [reference_x, reference_x, comparison_x, comparison_x],
            [group_cap_y[reference], bracket_y, bracket_y, group_cap_y[comparison]],
            color=TEXT_COLOR,
            linewidth=0.9,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )
        axis.text(
            (reference_x + comparison_x) / 2,
            bracket_y + 0.25,
            annotation,
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold",
            color=TEXT_COLOR,
            clip_on=False,
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
                "pad": 0.08,
            },
        )
    axis.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    axis.set_ylim(
        60.0,
        bracket_base
        + (len(comparison_order) - 1) * comparison_step
        + 1.8,
    )
    axis.set_yticks((60, 70, 80, 90, 100))
    axis.set_ylabel("Window-level accuracy (%)", fontsize=AXIS_LABEL_SIZE)
    axis.set_xticks(
        positions,
        [
            {
                "PRISM-Gait": "PRISM-\nGait\n(Ours)",
                "MultiRocket": "Multi-\nROCKET",
                "MiniRocket": "Mini-\nROCKET",
                "FCN-BiLSTM": "FCN-\nBiLSTM",
                "InceptionTime": "Inception-\nTime",
            }[model]
            for model in MODEL_ORDER
        ],
    )
    axis.tick_params(axis="x", length=0, pad=4, labelsize=TICK_SIZE)
    axis.tick_params(axis="y", labelsize=TICK_SIZE)
    axis.minorticks_off()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", top=False, right=False)
    for tick, model in zip(axis.get_xticklabels(), MODEL_ORDER):
        tick.set_fontweight("bold" if model == "PRISM-Gait" else "normal")


def draw_sway_bars(
    axis,
    evidence: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw grouped window-level accuracy bars for both sway conditions."""
    conditions = (
        ("trunk_sway_small", -0.17, SMALL_SWAY_COLOR, "#1F786F"),
        ("trunk_sway_large", 0.17, LARGE_SWAY_COLOR, "#6E526B"),
    )
    stats = statistics.loc[statistics["aggregation"].eq("Window level")].set_index(
        ["condition", "comparison"]
    )
    subset = evidence.loc[evidence["aggregation"].eq("Window level")]
    for row, model in enumerate(MODEL_ORDER):
        for condition, offset, face, edge in conditions:
            y = row + offset
            value = float(
                subset.loc[
                    subset["condition"].eq(condition) & subset["model"].eq(model),
                    "accuracy_pct",
                ].iloc[0]
            )
            highlight = model == "PRISM-Gait"
            annotation = "" if highlight else str(stats.loc[(condition, model), "annotation"])
            axis.barh(
                y,
                value - 60.0,
                left=60.0,
                height=0.29,
                facecolor=face,
                edgecolor=TEXT_COLOR if highlight else edge,
                linewidth=1.15 if highlight else 0.7,
                zorder=2,
            )
            axis.text(
                value + 0.35,
                y,
                f"{value:.1f}{(' ' + annotation) if annotation else ''}",
                ha="left",
                va="center",
                fontsize=7.7,
                fontweight="bold" if highlight else "normal",
                color=TEXT_COLOR,
            )

    axis.set_xlim(60, 101.5)
    axis.set_ylim(len(MODEL_ORDER) - 0.52, -0.52)
    axis.set_xticks((60, 70, 80, 90, 100))
    axis.set_xlabel("Window-level accuracy (%)", fontsize=8.5)
    axis.set_yticks(
        range(len(MODEL_ORDER)),
        [MODEL_DISPLAY_NAMES[model] for model in MODEL_ORDER],
    )
    axis.tick_params(axis="y", length=0, pad=5, labelsize=8.2)
    for tick, model in zip(axis.get_yticklabels(), MODEL_ORDER):
        tick.set_fontweight("bold" if model == "PRISM-Gait" else "normal")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="x", which="both", top=False, right=False)
    axis.text(
        -0.105,
        1.08,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

def format_time(value_seconds: float) -> str:
    """Return a concise time label."""
    if value_seconds < 1.0:
        return f"{1000.0 * value_seconds:.1f} ms"
    if value_seconds < 60.0:
        return f"{value_seconds:.1f} s"
    return f"{value_seconds / 60.0:.1f} min"


def draw_timing_panel(
    axis,
    timing: pd.DataFrame,
    panel_label: str | None = "g",
    title: str | None = "Operational timing",
) -> None:
    """Draw online and registration times on one logarithmic scale."""
    required = {
        "model",
        "registration_seconds",
        "online_median_ms",
        "online_q1_ms",
        "online_q3_ms",
    }
    missing = required.difference(timing.columns)
    if missing:
        raise ValueError(f"timing evidence is missing columns: {sorted(missing)}")
    timing = timing.set_index("model").reindex(MODEL_ORDER)
    if timing[list(required.difference({"model"}))].isna().any().any():
        raise ValueError("timing evidence is incomplete")

    registration = timing["registration_seconds"].to_numpy(dtype=float)
    online = timing["online_median_ms"].to_numpy(dtype=float) / 1000.0
    online_q1 = timing["online_q1_ms"].to_numpy(dtype=float) / 1000.0
    online_q3 = timing["online_q3_ms"].to_numpy(dtype=float) / 1000.0
    y_positions = np.arange(len(MODEL_ORDER), dtype=float)
    model_colors = {
        "PRISM-Gait": PRISM_COLOR,
        "MultiRocket": "#6C8EAD",
        "MiniRocket": "#D59B47",
        "FCN-BiLSTM": "#C97B63",
        "InceptionTime": "#7FA07B",
    }

    for index, model in enumerate(MODEL_ORDER):
        highlight = model == "PRISM-Gait"
        model_color = model_colors[model]
        axis.plot(
            [online[index], registration[index]],
            [index, index],
            color=model_color,
            linewidth=2.15 if highlight else 1.75,
            alpha=0.48,
            zorder=1,
        )
        axis.errorbar(
            online[index],
            index,
            xerr=np.array(
                [
                    [online[index] - online_q1[index]],
                    [online_q3[index] - online[index]],
                ]
            ),
            fmt="D",
            markersize=6.6 if highlight else 5.7,
            color=model_color,
            markeredgecolor=TEXT_COLOR if highlight else "white",
            markeredgewidth=1.0 if highlight else 0.7,
            ecolor=model_color,
            elinewidth=1.5,
            capsize=3.0,
            zorder=3,
        )
        axis.scatter(
            registration[index],
            index,
            s=52 if highlight else 42,
            marker="o",
            color=model_color,
            edgecolor=TEXT_COLOR if highlight else "white",
            linewidth=1.0 if highlight else 0.7,
            zorder=3,
        )
        online_near_left = online[index] < 0.05
        registration_near_right = registration[index] > 1000.0
        axis.annotate(
            format_time(online[index]),
            (online[index], index),
            xytext=(4 if online_near_left else 0, 5),
            textcoords="offset points",
            ha="left" if online_near_left else "center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold" if highlight else "normal",
            color=TEXT_COLOR,
        )
        axis.annotate(
            format_time(registration[index]),
            (registration[index], index),
            xytext=(
                -4 if registration_near_right else 0,
                5 if index >= len(MODEL_ORDER) - 2 else -5,
            ),
            textcoords="offset points",
            ha="right" if registration_near_right else "center",
            va="bottom" if index >= len(MODEL_ORDER) - 2 else "top",
            fontsize=VALUE_SIZE,
            fontweight="bold" if highlight else "normal",
            color=TEXT_COLOR,
        )

    axis.set_xscale("log")
    axis.set_xlim(0.006, 10000.0)
    axis.set_xticks(
        [0.01, 1.0, 100.0, 10000.0],
        [r"$10^{-2}$", "1", r"$10^{2}$", r"$10^{4}$"],
    )
    axis.set_xlabel("Elapsed time (s)")
    axis.set_yticks(
        y_positions,
        [
            "PRISM-Gait\n(Ours)"
            if model == "PRISM-Gait"
            else MODEL_DISPLAY_NAMES[model]
            for model in MODEL_ORDER
        ],
    )
    axis.set_ylim(len(MODEL_ORDER) - 0.65, -0.35)
    axis.tick_params(axis="y", length=0, pad=6, labelsize=TICK_SIZE)
    for tick, model in zip(axis.get_yticklabels(), MODEL_ORDER):
        tick.set_fontweight("bold" if model == "PRISM-Gait" else "normal")
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                color=TEXT_COLOR,
                markerfacecolor=TEXT_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label="Online identification",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=TEXT_COLOR,
                markerfacecolor=TEXT_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label="Registration",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.8,
        markerscale=1.35,
        columnspacing=1.4,
        handletextpad=0.35,
        borderaxespad=0,
    )
    if panel_label is not None:
        axis.text(
            -0.08,
            1.08,
            f"({panel_label})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )
    if title is not None:
        axis.text(
            0.03,
            1.08,
            title,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
        )


def build_figure(
    evidence: pd.DataFrame,
    statistics: pd.DataFrame,
    sway_intervals: pd.DataFrame,
    sway_statistics: pd.DataFrame,
):
    """Build the two-panel window-level treadmill overview."""
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM),
        facecolor="white",
    )

    sway_axis = add_axes_mm(figure, 18.0, 18.0, 72.0, 58.0)
    draw_sway_vertical_bars(sway_axis, evidence, statistics)
    window_axis = add_axes_mm(figure, 103.0, 18.0, 72.0, 58.0)
    draw_non_sway_profiles(window_axis, evidence)
    figure.legend(
        handles=[
            Patch(
                facecolor=SMALL_SWAY_COLOR,
                edgecolor="#1F786F",
                label="Small sway",
            ),
            Patch(
                facecolor=LARGE_SWAY_COLOR,
                edgecolor="#6E526B",
                label="Large sway",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(54 / FIGURE_WIDTH_MM, 0.985),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=0.65,
        handlelength=1.4,
        handleheight=1.15,
        handletextpad=0.35,
    )
    for x_mm, y_mm, label in (
        (4.0, 82.0, "(a)"),
        (92.0, 82.0, "(b)"),
    ):
        figure.text(
            x_mm / FIGURE_WIDTH_MM,
            y_mm / FIGURE_HEIGHT_MM,
            label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )
    return figure


def build_timing_figure(timing: pd.DataFrame) -> plt.Figure:
    """Build operational timing as a standalone single-column figure."""
    figure = plt.figure(
        figsize=(TIMING_WIDTH_MM * MM, TIMING_HEIGHT_MM * MM),
        facecolor="white",
    )
    axis = figure.add_axes(
        [
            22.0 / TIMING_WIDTH_MM,
            15.0 / TIMING_HEIGHT_MM,
            62.0 / TIMING_WIDTH_MM,
            43.0 / TIMING_HEIGHT_MM,
        ]
    )
    draw_timing_panel(axis, timing, panel_label=None, title=None)
    return figure


def write_caption() -> None:
    """Write the manuscript-ready evidence caption."""
    CAPTION_PATH.write_text(
        (
            "Window-level treadmill identification. (a) Teal and plum bars compare small "
            "and large trunk sway. (b) Colored model profiles report window-level accuracy "
            "across seven secondary gait perturbations."
        ),
        encoding="utf-8",
    )


def render() -> Path:
    """Render and export the combined figure and its evidence files."""
    apply_final_style()
    logger = create_logger()
    logger.info("step_01: loading audited sway and non-sway evidence")
    evidence = load_evidence()
    statistics = load_statistics()
    sway_intervals = pd.read_csv(SWAY_INTERVAL_PATH)
    sway_statistics = pd.read_csv(SWAY_STATISTICS_PATH)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(EVIDENCE_PATH, index=False)
    statistics.to_csv(STATISTICS_PATH, index=False)
    write_caption()

    logger.info(
        "step_02: rendering rows=%s comparisons=%s",
        len(evidence),
        len(statistics),
    )
    figure = build_figure(
        evidence,
        statistics,
        sway_intervals,
        sway_statistics,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf")
    plt.close(figure)
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"figure export failed: {OUTPUT_PATH}")
    timing_figure = build_timing_figure(pd.read_csv(TIMING_PATH))
    timing_figure.savefig(TIMING_OUTPUT_PATH, format="pdf")
    plt.close(timing_figure)
    if not TIMING_OUTPUT_PATH.is_file() or TIMING_OUTPUT_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"figure export failed: {TIMING_OUTPUT_PATH}")
    logger.info(
        "step_03: verified output bytes=%s timing_bytes=%s",
        OUTPUT_PATH.stat().st_size,
        TIMING_OUTPUT_PATH.stat().st_size,
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render())
