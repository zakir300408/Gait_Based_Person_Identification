"""Create the two-panel PRISM-Gait sway ablation figure.

Panel (a) compares participant-mean window accuracy with participant-bootstrap
95% confidence intervals. Panel (b) reports the corresponding pooled small-
and large-sway window error percentages.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.figures.plot_style import (
    PANEL_LABEL_SIZE,
    VALUE_SIZE,
    apply_final_style,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "baselines_out" / "prism_sway_ablation"
FINAL_FIGURE_DIR = ROOT / "final figures"
PDF_PATH = FINAL_FIGURE_DIR / "prism gait ablation.pdf"
SOURCE_PATH = OUT_DIR / "fig_prism_sway_ablation_source_data.csv"
STATS_PATH = OUT_DIR / "fig_prism_sway_ablation_statistics.csv"

STYLE_PATH = Path(__file__).with_name("paper.mplstyle")

CONDITIONS = (
    ("Small sway", "trunk_sway_small", "#2A9D8F", "o"),
    ("Large sway", "trunk_sway_large", "#8E6C8A", "s"),
)
CONFIGURATIONS = (
    {
        "display": "Core",
        "variant": "Lite core",
        "subject_path": ROOT
        / "baselines_out"
        / "prism_gait_ablation_final"
        / "subject_condition_accuracy.csv",
        "summary_path": ROOT
        / "baselines_out"
        / "prism_gait_ablation_final"
        / "component_ablation.csv",
        "feature_dim": 7812,
    },
    {
        "display": "Core + CDKC",
        "variant": "Core+PKD",
        "subject_path": ROOT
        / "baselines_out"
        / "prism_gait_ablation_final"
        / "subject_condition_accuracy.csv",
        "summary_path": ROOT
        / "baselines_out"
        / "prism_gait_ablation_final"
        / "component_ablation.csv",
        "feature_dim": 12420,
    },
    {
        "display": "Core + HCGF",
        "variant": "Core+PHCF",
        "subject_path": ROOT
        / "baselines_out"
        / "prism_architecture_selection"
        / "architecture_candidate_subject_accuracy.csv",
        "summary_path": ROOT
        / "baselines_out"
        / "prism_architecture_selection"
        / "architecture_candidate_summary.csv",
        "feature_dim": 8441,
    },
    {
        "display": "PRISM-Gait",
        "variant": "PRISM-Gait",
        "subject_path": ROOT
        / "baselines_out"
        / "prism_subject_window_accuracy"
        / "prism_subject_condition_accuracy.csv",
        "summary_path": ROOT
        / "baselines_out"
        / "prism_subject_window_accuracy"
        / "prism_condition_summary.csv",
        "feature_dim": 13049,
    },
)
PLOT_ORDER = ("PRISM-Gait", "Core + HCGF", "Core + CDKC", "Core")


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, repetitions: int = 20_000
) -> tuple[float, float, float]:
    """Return participant mean and percentile-bootstrap 95% CI."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap input must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def load_source_data() -> pd.DataFrame:
    """Load and audit the exact factorial configurations used in the figure."""
    rows: list[dict[str, object]] = []
    for config_index, config in enumerate(CONFIGURATIONS):
        subject_path = Path(config["subject_path"])
        summary_path = Path(config["summary_path"])
        for path in (subject_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing required ablation input: {path}")

        subjects = pd.read_csv(subject_path)
        summary = pd.read_csv(summary_path)
        for condition_index, (condition_label, condition, _, _) in enumerate(CONDITIONS):
            subject_rows = subjects.loc[
                subjects["variant"].eq(config["variant"])
                & subjects["condition"].eq(condition)
                & np.isclose(subjects["win_s"].astype(float), 2.0)
            ].copy()
            summary_rows = summary.loc[
                summary["variant"].eq(config["variant"])
                & summary["condition"].eq(condition)
                & np.isclose(summary["win_s"].astype(float), 2.0)
            ].copy()
            if len(summary_rows) != 1:
                raise ValueError(
                    f"expected one summary row for {config['variant']}/{condition}; "
                    f"found {len(summary_rows)}"
                )
            if subject_rows["subject"].duplicated().any():
                raise ValueError(
                    f"duplicate participant rows for {config['variant']}/{condition}"
                )

            summary_row = summary_rows.iloc[0]
            observed_dim = int(summary_row["feature_dim"])
            if observed_dim != int(config["feature_dim"]):
                raise ValueError(
                    f"feature dimension mismatch for {config['variant']}: "
                    f"{observed_dim} != {config['feature_dim']}"
                )
            expected_participants = int(summary_row["n_trials"])
            if len(subject_rows) != expected_participants:
                raise ValueError(
                    f"participant/trial count mismatch for {config['variant']}/{condition}: "
                    f"{len(subject_rows)} != {expected_participants}"
                )

            values = (
                100.0
                * subject_rows["subject_window_accuracy"].to_numpy(dtype=float)
            )
            windows = int(summary_row["n_windows"])
            misidentified_windows = int(
                round(windows * (1.0 - float(summary_row["window_accuracy"])))
            )
            mean, ci_lower, ci_upper = bootstrap_mean_ci(
                values,
                seed=20260727 + 10 * config_index + condition_index,
            )
            rows.append(
                {
                    "configuration": config["display"],
                    "implementation_variant": config["variant"],
                    "condition": condition_label,
                    "condition_key": condition,
                    "participants": len(values),
                    "participant_mean_window_accuracy_pct": mean,
                    "bootstrap_ci_lower_pct": ci_lower,
                    "bootstrap_ci_upper_pct": ci_upper,
                    "pooled_window_accuracy_pct": 100.0
                    * float(summary_row["window_accuracy"]),
                    "misidentified_windows_pct": 100.0
                    * misidentified_windows
                    / windows,
                    "misidentified_windows": misidentified_windows,
                    "windows": windows,
                    "trial_errors": int(summary_row["misid_trials"]),
                    "trials": int(summary_row["n_trials"]),
                    "trial_accuracy_pct": 100.0
                    * float(summary_row["trial_accuracy"]),
                    "feature_dim": observed_dim,
                }
            )
    return pd.DataFrame(rows)


def load_participant_data() -> pd.DataFrame:
    """Load participant-level sway accuracies for paired inference."""
    frames = []
    for config in CONFIGURATIONS:
        subjects = pd.read_csv(Path(config["subject_path"]))
        rows = subjects.loc[
            subjects["variant"].eq(config["variant"])
            & subjects["condition"].isin(
                ["trunk_sway_small", "trunk_sway_large"]
            )
            & np.isclose(subjects["win_s"].astype(float), 2.0),
            ["condition", "subject", "subject_window_accuracy"],
        ].copy()
        rows["configuration"] = config["display"]
        if rows.duplicated(["condition", "subject"]).any():
            raise ValueError(
                f"duplicate participant rows for {config['display']}"
            )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust a one-dimensional array of p-values."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def compute_statistics(participants: pd.DataFrame) -> pd.DataFrame:
    """Compute planned paired comparisons of PRISM-Gait against each ablation."""
    reference = "PRISM-Gait"
    targets = [
        str(config["display"])
        for config in CONFIGURATIONS
        if config["display"] != reference
    ]
    records: list[dict[str, object]] = []
    for condition_index, (condition_label, condition_key, _, _) in enumerate(
        CONDITIONS
    ):
        pivot = participants.loc[
            participants["condition"].eq(condition_key)
        ].pivot(
            index="subject",
            columns="configuration",
            values="subject_window_accuracy",
        )
        for target_index, target in enumerate(targets):
            paired = pivot[[reference, target]].dropna()
            difference = (
                100.0
                * (paired[reference] - paired[target]).to_numpy(dtype=float)
            )
            statistic, p_value = wilcoxon(
                difference,
                zero_method="pratt",
                alternative="two-sided",
                method="approx",
            )
            mean_delta, ci_lower, ci_upper = bootstrap_mean_ci(
                difference,
                seed=20260727 + 100 * condition_index + target_index,
            )
            records.append(
                {
                    "condition": condition_label,
                    "reference": reference,
                    "comparison": target,
                    "participants": len(difference),
                    "mean_delta_pp": mean_delta,
                    "bootstrap_ci_lower_pp": ci_lower,
                    "bootstrap_ci_upper_pp": ci_upper,
                    "median_delta_pp": float(np.median(difference)),
                    "wilcoxon_statistic": float(statistic),
                    "p_raw": float(p_value),
                    "test": "paired Wilcoxon signed-rank, Pratt zeros, two-sided",
                    "correction": "Holm across six planned comparisons",
                }
            )
    result = pd.DataFrame(records)
    result["p_holm"] = holm_adjust(result["p_raw"].to_numpy(dtype=float))
    return result


def format_significance(value: float) -> str:
    """Format Holm-adjusted p-values with a compact three-star maximum."""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def style_axes(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", direction="in", width=0.5)
    ax.tick_params(axis="both", which="minor", direction="in", width=0.5)


def draw_ablation_panels(
    ax_accuracy: plt.Axes,
    ax_errors: plt.Axes,
    source: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw the paired factorial-accuracy and pooled-error panels."""
    labels = list(PLOT_ORDER)
    positions = np.arange(len(labels), dtype=float)
    bar_width = 0.32
    offsets = {"Small sway": -0.18, "Large sway": 0.18}
    colors = {"Small sway": "#2A9D8F", "Large sway": "#8E6C8A"}
    edges = {"Small sway": "#1F786F", "Large sway": "#6E526B"}

    for condition_label, _, _, _ in CONDITIONS:
        condition_rows = (
            source.loc[source["condition"].eq(condition_label)]
            .set_index("configuration")
            .loc[labels]
        )
        means = condition_rows["participant_mean_window_accuracy_pct"].to_numpy(dtype=float)
        ci_lower = condition_rows["bootstrap_ci_lower_pct"].to_numpy(dtype=float)
        ci_upper = condition_rows["bootstrap_ci_upper_pct"].to_numpy(dtype=float)
        x = positions + offsets[condition_label]
        bars = ax_accuracy.bar(
            x,
            means - 72.0,
            bottom=72.0,
            width=bar_width,
            color=colors[condition_label],
            edgecolor=edges[condition_label],
            linewidth=0.7,
            zorder=2,
        )
        ax_accuracy.errorbar(
            x,
            means,
            yerr=np.vstack((means - ci_lower, ci_upper - means)),
            fmt="none",
            ecolor="#1A1A1A",
            elinewidth=0.75,
            capsize=2.5,
            capthick=0.75,
            zorder=4,
        )
        for bar, label, mean, upper in zip(bars, labels, means, ci_upper):
            is_prism = label == "PRISM-Gait"
            if is_prism:
                bar.set_edgecolor("#1A1A1A")
                bar.set_linewidth(1.15)
            ax_accuracy.text(
                bar.get_x() + bar.get_width() / 2,
                upper + 0.65,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold" if is_prism else "normal",
                color="#1A1A1A",
            )

        errors = condition_rows["misidentified_windows_pct"].to_numpy(dtype=float)
        error_bars = ax_errors.bar(
            x,
            errors,
            width=bar_width,
            color=colors[condition_label],
            edgecolor=edges[condition_label],
            linewidth=0.7,
            zorder=2,
        )
        for bar, label, value in zip(error_bars, labels, errors):
            is_prism = label == "PRISM-Gait"
            if is_prism:
                bar.set_edgecolor("#1A1A1A")
                bar.set_linewidth(1.15)
            ax_errors.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.45,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold" if is_prism else "normal",
                color="#1A1A1A",
            )

    wrapped_labels = [
        (
            "PRISM-Gait\n(Ours)"
            if label == "PRISM-Gait"
            else label.replace(" + ", " +\n")
            if label.startswith("Core + ")
            else label
        )
        for label in labels
    ]
    ax_accuracy.set_xlim(-0.55, len(labels) - 0.45)
    ax_accuracy.set_ylim(72.0, 101.5)
    ax_accuracy.set_yticks([75, 80, 85, 90, 95, 100])
    ax_accuracy.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2.5))
    ax_accuracy.set_ylabel("Mean accuracy (%)")
    ax_accuracy.set_xticks(positions, wrapped_labels)
    ax_accuracy.tick_params(axis="x", length=0, pad=4)
    ax_accuracy.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style_axes(ax_accuracy)

    ax_errors.set_xlim(-0.55, len(labels) - 0.45)
    ax_errors.set_ylim(0.0, 22.0)
    ax_errors.set_yticks([0, 5, 10, 15, 20])
    ax_errors.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2.5))
    ax_errors.set_ylabel("Pooled window error (%)")
    ax_errors.set_xticks(positions, wrapped_labels)
    ax_errors.tick_params(axis="x", length=0, pad=4)
    ax_errors.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style_axes(ax_errors)

    for axis in (ax_accuracy, ax_errors):
        for tick, label in zip(axis.get_xticklabels(), labels):
            if label == "PRISM-Gait":
                tick.set_fontweight("bold")


def create_figure(
    source: pd.DataFrame,
    participants: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    apply_final_style()

    width_mm, height_mm = 180.0, 82.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax_accuracy = fig.add_axes([15 / width_mm, 14 / height_mm, 74 / width_mm, 52 / height_mm])
    ax_errors = fig.add_axes([102 / width_mm, 14 / height_mm, 74 / width_mm, 52 / height_mm])
    draw_ablation_panels(ax_accuracy, ax_errors, source, statistics)

    fig.legend(
        handles=[
            Patch(facecolor=colors["Small sway"], edgecolor=edges["Small sway"], label="Small sway"),
            Patch(facecolor=colors["Large sway"], edgecolor=edges["Large sway"], label="Large sway"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.35,
        ncol=2,
        columnspacing=0.9,
    )
    for x, label in ((5, "(a)"), (95, "(b)")):
        fig.text(
            x / width_mm,
            69 / height_mm,
            label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    plt.close(fig)


def main() -> None:
    source = load_source_data()
    participants = load_participant_data()
    statistics = compute_statistics(participants)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE_PATH, index=False)
    statistics.to_csv(STATS_PATH, index=False)
    create_figure(source, participants, statistics)
    print(PDF_PATH)
    print(SOURCE_PATH)
    print(STATS_PATH)


if __name__ == "__main__":
    main()
