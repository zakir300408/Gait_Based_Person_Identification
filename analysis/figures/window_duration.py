"""Render PRISM-Gait duration selection, latency, and model timing evidence."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.figures.plot_style import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    PRISM_PALE,
    VALUE_SIZE,
    apply_final_style,
)
from analysis.figures.identification_overview import draw_timing_panel


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "baselines_out" / "prism_duration_trunk_sway"
SUBJECT_PATH = OUTPUT_DIR / "prism_duration_subject_accuracy.csv"
LATENCY_PATH = OUTPUT_DIR / "prism_duration_latency.csv"
SUMMARY_PATH = OUTPUT_DIR / "prism_duration_ablation_summary.csv"
PARTICIPANT_SUMMARY_PATH = OUTPUT_DIR / "prism_duration_participant_summary.csv"
DECISION_PATH = OUTPUT_DIR / "prism_duration_selection_decision.json"
TIMING_PATH = (
    ROOT
    / "baselines_out"
    / "treadmill_published_plot_data"
    / "fig_non_sway_operational_timing_evidence.csv"
)
OUTPUT_PATH = ROOT / "final figures" / "window duration and computational timing.pdf"
MM = 1 / 25.4
GROUPS = {"Small sway": "trunk_sway_small", "Large sway": "trunk_sway_large"}
FILL = {"Small sway": "#2A9D8F", "Large sway": "#8E6C8A"}
EDGE = {"Small sway": "#1F786F", "Large sway": "#6E526B"}
CHOSEN_DURATION = 2.0
TEXT_COLOR = "#1A1A1A"
LIGHT_LINE = "#C7C7C7"
LABEL_FONT_SIZE = AXIS_LABEL_SIZE


def create_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"prism_duration_figure_{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(OUTPUT_DIR / f"prism_duration_figure_{timestamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def summarize_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "win_s", "condition", "subject", "subject_window_accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"subject evidence is missing columns: {sorted(missing)}")
    if set(frame["variant"]) != {"PRISM-Gait"}:
        raise ValueError("duration figure requires fixed PRISM-Gait rows only")
    rows = []
    for group, condition in GROUPS.items():
        values = frame.loc[frame["condition"].eq(condition)]
        summary = values.groupby("win_s")["subject_window_accuracy"].agg(["mean", "std", "count"])
        summary = summary.reset_index().rename(
            columns={"mean": "mean_accuracy", "std": "sd_accuracy", "count": "participants"}
        )
        summary["ci95_accuracy"] = 1.96 * summary["sd_accuracy"] / np.sqrt(summary["participants"])
        summary["group"] = group
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    if result["win_s"].nunique() != 5 or result["sd_accuracy"].isna().any():
        raise ValueError("duration figure requires complete 1–5 s participant evidence")
    return result


def build_summary(subjects: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    measures = ["mean_accuracy", "sd_accuracy", "ci95_accuracy", "participants"]
    accuracy = subjects.pivot(index="win_s", columns="group", values=measures)
    accuracy.columns = [
        f"{group.lower().replace(' ', '_')}_{measure}" for measure, group in accuracy.columns
    ]
    result = latency.set_index("win_s").join(accuracy).reset_index()
    result["batch_windows_per_second"] = 1000 / result["batch_ms_per_window"]
    result["first_id_p95_ms"] = 1000 * result["win_s"] + result["single_window_p95_ms"]
    result["p95_realtime_margin_ms"] = result["update_interval_ms"] - result["single_window_p95_ms"]
    result["p95_within_update_interval"] = result["p95_realtime_margin_ms"].gt(0)
    if set(result["win_s"]) != {1.0, 2.0, 3.0, 4.0, 5.0}:
        raise ValueError("duration summary must contain 1–5 s exactly once")
    return result.sort_values("win_s")


def panel_label(axis, label: str, x: float = -0.11, y: float = 1.06) -> None:
    axis.text(x, y, label, transform=axis.transAxes, ha="left",
              va="bottom", fontsize=PANEL_LABEL_SIZE, fontweight="bold")


def style_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", top=False, right=False)


def draw_accuracy(
    axis,
    subjects: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    durations = np.arange(1, 6, dtype=float)
    column_offsets = {"Small sway": -0.20, "Large sway": 0.20}
    axis.axvspan(1.55, 2.45, facecolor=PRISM_PALE, edgecolor="none", zorder=0)
    for group in GROUPS:
        values = (
            subjects.loc[subjects["group"].eq(group)]
            .sort_values("win_s")
        )
        x_positions = durations + column_offsets[group]
        distributions = [
            100
            * values.loc[values["win_s"].eq(duration), "subject_window_accuracy"].to_numpy(
                dtype=float
            )
            for duration in durations
        ]
        means = np.asarray([float(np.mean(distribution)) for distribution in distributions])
        best_mean = float(np.max(means))
        box = axis.boxplot(
            distributions,
            positions=x_positions,
            widths=0.27,
            vert=True,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": FILL[group], "edgecolor": EDGE[group], "linewidth": 0.75},
            whiskerprops={"color": EDGE[group], "linewidth": 0.7},
            capprops={"color": EDGE[group], "linewidth": 0.7},
            medianprops={"color": "white", "linewidth": 1.2},
            flierprops={
                "marker": ".",
                "markersize": 2.0,
                "markerfacecolor": EDGE[group],
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        for duration, position, distribution, mean, patch in zip(
            durations, x_positions, distributions, means, box["boxes"]
        ):
            selected = np.isclose(duration, CHOSEN_DURATION)
            if selected:
                patch.set_edgecolor(TEXT_COLOR)
                patch.set_linewidth(1.15)
            # Attach the reported participant mean to its own box instead of
            # forcing all values into detached rows at the top of the panel.
            # A small outward nudge separates adjacent condition labels while
            # keeping each value visually centered over its box.
            label_y = float(np.percentile(distribution, 75)) + 0.75
            label_offset = -0.035 if group == "Small sway" else 0.035
            axis.text(
                position + label_offset,
                label_y,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold" if np.isclose(mean, best_mean) else "normal",
                color=TEXT_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.05},
                zorder=4,
            )
    for duration in durations:
        axis.vlines(duration, 0, 106.5, color="#ECECEC", linewidth=0.55, zorder=-1)
    axis.set_xlabel("Window duration", fontsize=LABEL_FONT_SIZE)
    axis.set_xlim(0.55, 5.45)
    axis.set_xticks(durations)
    axis.set_xticklabels([f"{int(value)} s" for value in durations])
    axis.set_ylabel("Window accuracy (%)", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylim(50, 109.5)
    axis.set_yticks((50, 60, 70, 80, 90, 100))
    axis.tick_params(axis="x", length=0)
    for tick, duration in zip(axis.get_xticklabels(), durations):
        if np.isclose(duration, CHOSEN_DURATION):
            tick.set_fontweight("bold")
            tick.set_color(PRISM_COLOR)
    style_axes(axis)
    return None


def build_figure(subjects: pd.DataFrame, summary: pd.DataFrame, timing: pd.DataFrame):
    width_mm, height_mm = 180.0, 120.0
    figure = plt.figure(figsize=(width_mm * MM, height_mm * MM), facecolor="white")
    ax_accuracy = figure.add_axes([24 / width_mm, 67 / height_mm, 151 / width_mm, 31 / height_mm])
    ax_timing = figure.add_axes([24 / width_mm, 13 / height_mm, 151 / width_mm, 31 / height_mm])
    draw_accuracy(ax_accuracy, subjects, summary)
    draw_timing_panel(
        ax_timing,
        timing,
        panel_label="b",
        title="Model registration and online identification time",
    )
    figure.text(
        4 / width_mm,
        99 / height_mm,
        "(a)",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )
    legend_handles = [
        Patch(
            facecolor=FILL["Small sway"],
            edgecolor=EDGE["Small sway"],
            label="Small sway",
        ),
        Patch(
            facecolor=FILL["Large sway"],
            edgecolor=EDGE["Large sway"],
            label="Large sway",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        frameon=False,
        fontsize=LABEL_FONT_SIZE,
        handlelength=1.5,
        handleheight=1.15,
        handletextpad=0.38,
        columnspacing=0.9,
    )
    return figure


def render() -> Path:
    for path in (SUBJECT_PATH, LATENCY_PATH, TIMING_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"run duration benchmark first: {path.name}")
    apply_final_style()
    logger = create_logger()
    logger.info("step_01: loading duration and latency evidence")
    raw_subjects = pd.read_csv(SUBJECT_PATH)
    raw_subjects["group"] = raw_subjects["condition"].map(
        {condition: group for group, condition in GROUPS.items()}
    )
    subjects = summarize_subjects(raw_subjects)
    summary = build_summary(subjects, pd.read_csv(LATENCY_PATH))
    timing = pd.read_csv(TIMING_PATH)
    subjects.to_csv(PARTICIPANT_SUMMARY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    large = subjects.loc[subjects["group"].eq("Large sway")]
    best_row = large.loc[large["mean_accuracy"].idxmax()]
    decision = {
        "selection_metric": "participant-mean large-sway window accuracy",
        "current_duration_s": CHOSEN_DURATION,
        "best_duration_s": float(best_row["win_s"]),
        "current_is_best": bool(np.isclose(best_row["win_s"], CHOSEN_DURATION)),
        "best_large_sway_mean_accuracy": float(best_row["mean_accuracy"]),
    }
    DECISION_PATH.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    logger.info("step_02: selection=%s", decision)
    figure = build_figure(raw_subjects, summary, timing)
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=600)
    plt.close(figure)
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise FileNotFoundError("duration figure export failed")
    logger.info("step_03: verified %s bytes=%s", OUTPUT_PATH.name, OUTPUT_PATH.stat().st_size)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render())
