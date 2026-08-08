"""Render the combined PRISM-Gait design-selection and ablation figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figs import prism_duration_selection_figure as duration
from figs import prism_sway_ablation_figure as ablation
from figs.tifs_final_style import LEGEND_SIZE, PANEL_LABEL_SIZE, VALUE_SIZE, apply_final_style


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "final figures" / "design selection and ablation.pdf"
MM = 1 / 25.4
WIDTH_MM = 180.0
HEIGHT_MM = 100.0


def load_duration_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load duration evidence and refresh its audit outputs."""
    for path in (duration.SUBJECT_PATH, duration.LATENCY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"required duration evidence not found: {path}")
    raw_subjects = pd.read_csv(duration.SUBJECT_PATH)
    raw_subjects["group"] = raw_subjects["condition"].map(
        {condition: group for group, condition in duration.GROUPS.items()}
    )
    participant_summary = duration.summarize_subjects(raw_subjects)
    summary = duration.build_summary(
        participant_summary,
        pd.read_csv(duration.LATENCY_PATH),
    )
    participant_summary.to_csv(duration.PARTICIPANT_SUMMARY_PATH, index=False)
    summary.to_csv(duration.SUMMARY_PATH, index=False)
    large = participant_summary.loc[participant_summary["group"].eq("Large sway")]
    best_row = large.loc[large["mean_accuracy"].idxmax()]
    decision = {
        "selection_metric": "participant-mean large-sway window accuracy",
        "current_duration_s": duration.CHOSEN_DURATION,
        "best_duration_s": float(best_row["win_s"]),
        "current_is_best": bool(
            np.isclose(best_row["win_s"], duration.CHOSEN_DURATION)
        ),
        "best_large_sway_mean_accuracy": float(best_row["mean_accuracy"]),
    }
    duration.DECISION_PATH.write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    return raw_subjects, summary


def load_ablation_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factorial evidence and refresh its source/statistics exports."""
    source = ablation.load_source_data()
    participants = ablation.load_participant_data()
    statistics = ablation.compute_statistics(participants)
    ablation.OUT_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(ablation.SOURCE_PATH, index=False)
    statistics.to_csv(ablation.STATS_PATH, index=False)
    return source, statistics


def add_axes_mm(
    figure: plt.Figure,
    left_mm: float,
    bottom_mm: float,
    width_mm: float,
    height_mm: float,
) -> plt.Axes:
    return figure.add_axes(
        [
            left_mm / WIDTH_MM,
            bottom_mm / HEIGHT_MM,
            width_mm / WIDTH_MM,
            height_mm / HEIGHT_MM,
        ]
    )


def draw_latency(axis: plt.Axes, summary: pd.DataFrame) -> None:
    """Draw the duration-dependent first-decision latency in its own panel."""
    rows = summary.sort_values("win_s")
    durations = rows["win_s"].to_numpy(dtype=float)
    latency_seconds = rows["first_id_median_ms"].to_numpy(dtype=float) / 1000.0
    axis.axvspan(1.55, 2.45, facecolor=duration.PRISM_PALE, edgecolor="none", zorder=0)
    axis.plot(
        durations,
        latency_seconds,
        color="#6F7C80",
        linewidth=1.25,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor="#6F7C80",
        markeredgewidth=0.9,
        zorder=2,
    )
    selected = np.isclose(durations, duration.CHOSEN_DURATION)
    axis.scatter(
        durations[selected],
        latency_seconds[selected],
        s=31,
        color=duration.PRISM_COLOR,
        edgecolor="#1A1A1A",
        linewidth=0.8,
        zorder=3,
    )
    for x_value, y_value in zip(durations, latency_seconds):
        is_selected = np.isclose(x_value, duration.CHOSEN_DURATION)
        axis.text(
            x_value,
            y_value + 0.18,
            f"{y_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_SIZE,
            fontweight="bold" if is_selected else "normal",
            color="#1A1A1A",
        )
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(0.5, 5.55)
    axis.set_xticks(durations, [f"{int(value)} s" for value in durations])
    axis.set_yticks(np.arange(1, 6, dtype=float))
    axis.set_xlabel("Window duration")
    axis.set_ylabel("First-decision latency (s)")
    axis.tick_params(axis="x", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", top=False, right=False)
    for tick, x_value in zip(axis.get_xticklabels(), durations):
        if np.isclose(x_value, duration.CHOSEN_DURATION):
            tick.set_fontweight("bold")
            tick.set_color(duration.PRISM_COLOR)


def build_figure(
    subjects: pd.DataFrame,
    duration_summary: pd.DataFrame,
    ablation_source: pd.DataFrame,
    ablation_statistics: pd.DataFrame,
) -> plt.Figure:
    """Build equal-width duration, latency, and ablation panels."""
    apply_final_style()
    figure = plt.figure(
        figsize=(WIDTH_MM * MM, HEIGHT_MM * MM),
        facecolor="white",
    )
    left_mm, right_mm, panel_width_mm = 15.0, 102.0, 74.0
    top_bottom_mm, top_height_mm = 58.0, 27.0
    lower_bottom_mm, lower_height_mm = 10.0, 30.0
    axes = (
        add_axes_mm(figure, left_mm, top_bottom_mm, panel_width_mm, top_height_mm),
        add_axes_mm(figure, right_mm, top_bottom_mm, panel_width_mm, top_height_mm),
        add_axes_mm(figure, left_mm, lower_bottom_mm, panel_width_mm, lower_height_mm),
        add_axes_mm(figure, right_mm, lower_bottom_mm, panel_width_mm, lower_height_mm),
    )
    duration.draw_accuracy(axes[0], subjects, duration_summary)
    draw_latency(axes[1], duration_summary)
    ablation.draw_ablation_panels(
        axes[2],
        axes[3],
        ablation_source,
        ablation_statistics,
    )

    figure.legend(
        handles=[
            Patch(
                facecolor=duration.FILL["Small sway"],
                edgecolor=duration.EDGE["Small sway"],
                label="Small sway",
            ),
            Patch(
                facecolor=duration.FILL["Large sway"],
                edgecolor=duration.EDGE["Large sway"],
                label="Large sway",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.5,
        handleheight=1.15,
        handletextpad=0.38,
        columnspacing=0.9,
    )
    for x_mm, y_mm, label in (
        (17.0, 87.0, "(a)"),
        (104.0, 87.0, "(b)"),
        (17.0, 42.0, "(c)"),
        (104.0, 42.0, "(d)"),
    ):
        figure.text(
            x_mm / WIDTH_MM,
            y_mm / HEIGHT_MM,
            label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )
    return figure


def render() -> Path:
    subjects, duration_summary = load_duration_evidence()
    ablation_source, ablation_statistics = load_ablation_evidence()
    figure = build_figure(
        subjects,
        duration_summary,
        ablation_source,
        ablation_statistics,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf")
    plt.close(figure)
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"figure export failed: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render())
