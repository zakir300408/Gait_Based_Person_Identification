"""Create the combined PRISM-Gait trunk-sway robustness figure.

The figure uses stored trial-aggregated class scores from the exact enrolled
treadmill protocol. It reports cumulative match characteristic (CMC) curves
for small and large trunk sway without reconstructing ranks from accuracy,
together with the enrolled and subject-held-out verification DET curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from analysis.tifs_final_style import (
    LEGEND_SIZE,
    LARGE_SWAY_COLOR,
    LARGE_SWAY_LIGHT,
    LARGE_SWAY_PALE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    SMALL_SWAY_COLOR,
    SMALL_SWAY_LIGHT,
    SMALL_SWAY_PALE,
    TEXT_COLOR,
    VALUE_SIZE,
    apply_final_style,
)
from analysis.prism_verification_figure import (
    draw_verification_curves,
    load_scores as load_verification_scores,
    verification_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PRISM_SCORES = (
    ROOT
    / "baselines_out"
    / "prism_verification_figure"
    / "enrolled_trial_scores.csv"
)
COMPARATOR_SCORES = (
    ROOT
    / "baselines_out"
    / "prism_verification_figure"
    / "enrolled_comparator_trial_scores.csv"
)
NEURAL_SCORE_PATHS = (
    ROOT / "baselines_out" / "identification_rank_robustness" / "fcn_trial_scores.csv",
    ROOT
    / "baselines_out"
    / "identification_rank_robustness"
    / "inception_trial_scores.csv",
    ROOT / "baselines_out" / "identification_rank_robustness" / "lstm_trial_scores.csv",
)
WINDOW_RANK_PATHS = (
    ROOT / "baselines_out" / "identification_rank_robustness" / "prism_window_ranks.csv",
    ROOT
    / "baselines_out"
    / "identification_rank_robustness"
    / "minirocket_window_ranks.csv",
    ROOT
    / "baselines_out"
    / "identification_rank_robustness"
    / "multirocket_window_ranks.csv",
    ROOT / "baselines_out" / "identification_rank_robustness" / "fcn_window_ranks.csv",
    ROOT
    / "baselines_out"
    / "identification_rank_robustness"
    / "inception_window_ranks.csv",
    ROOT / "baselines_out" / "identification_rank_robustness" / "lstm_window_ranks.csv",
)
OUTPUT_DIR = ROOT / "baselines_out" / "identification_rank_robustness"
SOURCE_PATH = OUTPUT_DIR / "identification_rank_source_data.csv"
STATS_PATH = OUTPUT_DIR / "identification_rank_statistics.csv"
FINAL_DIR = ROOT / "final figures"
PDF_PATH = FINAL_DIR / "identification rank robustness.pdf"
STYLE_PATH = Path(__file__).with_name("publication-hybrid.mplstyle")

CONDITIONS = (
    ("trunk_sway_small", "Small sway"),
    ("trunk_sway_large", "Large sway"),
)
MODELS = (
    ("PRISM-Gait", "PRISM-Gait\n(Ours)", PRISM_COLOR, "-", "o", 5),
    ("MiniROCKET + ridge", "MiniROCKET", "#898989", ":", "D", 4),
    (
        "MultiROCKET + ridge",
        "MultiROCKET",
        "#A2A2A2",
        (0, (7, 3)),
        "v",
        3,
    ),
    ("FCN-BiLSTM", "FCN-BiLSTM", "#555555", "--", "s", 2),
    ("InceptionTime", "InceptionTime", "#707070", "-.", "^", 1),
    (
        "Multi-Model LSTM",
        "Multi-Model LSTM",
        "#BBBBBB",
        (0, (1, 2)),
        "P",
        0,
    ),
)
RANKS = np.arange(1, 11, dtype=int)


def load_and_validate_scores() -> pd.DataFrame:
    """Load matched trial claim scores and validate identity-score integrity."""
    frames = []
    for path in (PRISM_SCORES, COMPARATOR_SCORES, *NEURAL_SCORE_PATHS):
        if not path.is_file():
            raise FileNotFoundError(f"required score artifact not found: {path}")
        frames.append(pd.read_csv(path))
    scores = pd.concat(frames, ignore_index=True)
    required = {
        "model",
        "subject",
        "condition",
        "trial",
        "claim",
        "genuine",
        "score",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"score artifacts are missing required columns: {missing}")

    model_names = {model for model, *_ in MODELS}
    condition_names = {condition for condition, _ in CONDITIONS}
    scores = scores.loc[
        scores["model"].isin(model_names)
        & scores["condition"].isin(condition_names)
    ].copy()
    if scores.empty:
        raise ValueError("no matched trunk-sway trial scores were loaded")
    if scores[["claim", "genuine", "score"]].isna().any().any():
        raise ValueError("claim score rows contain missing values")

    keys = ["model", "condition", "subject", "trial"]
    if scores.duplicated(keys + ["claim"]).any():
        raise ValueError("duplicate identity-claim rows detected")
    grouped = scores.groupby(keys, observed=True)
    claim_counts = grouped["claim"].nunique()
    genuine_counts = grouped["genuine"].sum()
    if claim_counts.nunique() != 1 or int(claim_counts.iloc[0]) != 62:
        raise ValueError("every trial must contain scores for all 62 identities")
    if not genuine_counts.eq(1).all():
        raise ValueError("every trial must contain exactly one genuine identity")

    subject_sets = (
        scores.groupby(["condition", "model"], observed=True)["subject"]
        .apply(lambda values: tuple(sorted(values.unique())))
    )
    for condition, _ in CONDITIONS:
        sets = subject_sets.loc[condition]
        if len(set(sets.tolist())) != 1:
            raise ValueError(f"models are not participant-matched for {condition}")

    scores["identity_rank"] = grouped["score"].rank(
        method="max",
        ascending=False,
    )
    genuine = scores.loc[scores["genuine"].eq(1)].copy()
    if not genuine["identity_rank"].between(1, 62).all():
        raise ValueError("invalid genuine-identity rank detected")
    return genuine


def load_and_validate_window_ranks() -> pd.DataFrame:
    """Load genuine-identity window ranks and validate matched coverage."""
    frames = []
    for path in WINDOW_RANK_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"required window-rank artifact not found: {path}")
        frames.append(pd.read_csv(path))
    ranks = pd.concat(frames, ignore_index=True)
    required = {
        "model",
        "condition",
        "subject",
        "trial",
        "window_start",
        "window_end",
        "identity_rank",
    }
    missing = sorted(required.difference(ranks.columns))
    if missing:
        raise ValueError(f"window-rank artifacts are missing columns: {missing}")
    expected_models = {model for model, *_ in MODELS}
    expected_conditions = {condition for condition, _ in CONDITIONS}
    if set(ranks["model"]) != expected_models:
        raise ValueError("window-rank model set does not match the figure model set")
    if set(ranks["condition"]) != expected_conditions:
        raise ValueError("window-rank condition set is incomplete")
    keys = [
        "model",
        "condition",
        "subject",
        "trial",
        "window_start",
        "window_end",
    ]
    if ranks.duplicated(keys).any():
        raise ValueError("duplicate window-rank rows detected")
    if not ranks["identity_rank"].between(1, 62).all():
        raise ValueError("window identity ranks must lie between 1 and 62")
    counts = ranks.groupby(["model", "condition"], observed=True).size()
    for condition, expected in (
        ("trunk_sway_small", 4036),
        ("trunk_sway_large", 3907),
    ):
        if not counts.xs(condition, level="condition").eq(expected).all():
            raise ValueError(f"window coverage mismatch for {condition}")
    subject_sets = (
        ranks.groupby(["condition", "model"], observed=True)["subject"]
        .apply(lambda values: tuple(sorted(values.unique())))
    )
    for condition, _ in CONDITIONS:
        sets = subject_sets.loc[condition]
        if len(set(sets.tolist())) != 1:
            raise ValueError(f"window participants are not matched for {condition}")
    return ranks


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return 100.0 * (center - half), 100.0 * (center + half)


def build_cmc(genuine: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """Build Rank-1 through Rank-10 cumulative identification rates."""
    records: list[dict[str, object]] = []
    for condition, condition_label in CONDITIONS:
        for model, display, *_ in MODELS:
            ranks = genuine.loc[
                genuine["condition"].eq(condition)
                & genuine["model"].eq(model),
                "identity_rank",
            ].to_numpy(dtype=float)
            if ranks.size == 0:
                raise ValueError(f"no genuine ranks for {model}, {condition}")
            for rank in RANKS:
                successes = int(np.count_nonzero(ranks <= rank))
                if aggregation == "Trial":
                    lower, upper = wilson_interval(successes, len(ranks))
                else:
                    lower, upper = np.nan, np.nan
                records.append(
                    {
                        "aggregation": aggregation,
                        "condition": condition,
                        "condition_label": condition_label,
                        "model": model,
                        "display_model": display,
                        "rank_k": int(rank),
                        "identified_within_rank_k": successes,
                        "trials": int(len(ranks)),
                        "cmc_rate_pct": 100.0 * successes / len(ranks),
                        "wilson_ci_lower_pct": lower,
                        "wilson_ci_upper_pct": upper,
                    }
                )
    return pd.DataFrame(records)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    """Holm-adjust an array of p-values."""
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, (len(values) - position) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def build_trial_rank1_statistics(genuine: pd.DataFrame) -> pd.DataFrame:
    """Compute paired exact Rank-1 comparisons against PRISM-Gait."""
    rows: list[dict[str, object]] = []
    for condition, condition_label in CONDITIONS:
        condition_rows = genuine.loc[genuine["condition"].eq(condition)].copy()
        correctness = condition_rows.assign(
            correct_rank1=condition_rows["identity_rank"].le(1)
        ).pivot(
            index="subject",
            columns="model",
            values="correct_rank1",
        )
        for comparator, display, *_ in MODELS[1:]:
            paired = correctness[["PRISM-Gait", comparator]].dropna()
            prism_only = int(
                (
                    paired["PRISM-Gait"]
                    & ~paired[comparator]
                ).sum()
            )
            comparator_only = int(
                (
                    ~paired["PRISM-Gait"]
                    & paired[comparator]
                ).sum()
            )
            discordant = prism_only + comparator_only
            raw_p = (
                1.0
                if discordant == 0
                else float(
                    binomtest(
                        prism_only,
                        discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "aggregation": "Trial",
                    "condition": condition,
                    "condition_label": condition_label,
                    "reference": "PRISM-Gait",
                    "comparison": display,
                    "participants": int(len(paired)),
                    "prism_rank1_pct": 100.0
                    * float(paired["PRISM-Gait"].mean()),
                    "comparison_rank1_pct": 100.0
                    * float(paired[comparator].mean()),
                    "rank1_delta_pp": 100.0
                    * float(
                        paired["PRISM-Gait"].mean()
                        - paired[comparator].mean()
                    ),
                    "prism_correct_comparison_wrong": prism_only,
                    "prism_wrong_comparison_correct": comparator_only,
                    "p_raw": raw_p,
                    "test": "exact paired McNemar test",
                    "correction": "Holm across ten planned Rank-1 comparisons",
                }
            )
    statistics = pd.DataFrame(rows)
    statistics["p_holm"] = holm_adjust(
        statistics["p_raw"].to_numpy(dtype=float)
    )
    return statistics


def build_window_rank1_statistics(window_ranks: pd.DataFrame) -> pd.DataFrame:
    """Compare participant-level window Rank-1 accuracy against PRISM-Gait."""
    rows: list[dict[str, object]] = []
    correctness = window_ranks.assign(
        correct_rank1=window_ranks["identity_rank"].le(1).astype(float)
    )
    participant_accuracy = (
        correctness.groupby(
            ["condition", "model", "subject"],
            observed=True,
        )["correct_rank1"]
        .mean()
        .reset_index()
    )
    for condition, condition_label in CONDITIONS:
        condition_rows = participant_accuracy.loc[
            participant_accuracy["condition"].eq(condition)
        ]
        paired_accuracy = condition_rows.pivot(
            index="subject",
            columns="model",
            values="correct_rank1",
        )
        for comparator, display, *_ in MODELS[1:]:
            paired = paired_accuracy[["PRISM-Gait", comparator]].dropna()
            differences = (
                paired["PRISM-Gait"] - paired[comparator]
            ).to_numpy(dtype=float)
            raw_p = (
                1.0
                if np.allclose(differences, 0.0)
                else float(
                    wilcoxon(
                        paired["PRISM-Gait"],
                        paired[comparator],
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            )
            rows.append(
                {
                    "aggregation": "Window",
                    "condition": condition,
                    "condition_label": condition_label,
                    "reference": "PRISM-Gait",
                    "comparison": display,
                    "participants": int(len(paired)),
                    "prism_rank1_pct": 100.0
                    * float(paired["PRISM-Gait"].mean()),
                    "comparison_rank1_pct": 100.0
                    * float(paired[comparator].mean()),
                    "rank1_delta_pp": 100.0
                    * float(
                        paired["PRISM-Gait"].mean()
                        - paired[comparator].mean()
                    ),
                    "prism_correct_comparison_wrong": np.nan,
                    "prism_wrong_comparison_correct": np.nan,
                    "p_raw": raw_p,
                    "test": "paired Wilcoxon signed-rank test on participant window accuracy",
                    "correction": "Holm across ten planned Rank-1 comparisons",
                }
            )
    statistics = pd.DataFrame(rows)
    statistics["p_holm"] = holm_adjust(
        statistics["p_raw"].to_numpy(dtype=float)
    )
    return statistics


def style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_linewidth(0.5)
    axis.spines["bottom"].set_linewidth(0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=0.5,
        length=3,
        top=False,
        right=False,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        width=0.5,
        length=1.5,
        top=False,
        right=False,
    )


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def create_figure(
    cmc: pd.DataFrame,
    statistics: pd.DataFrame,
    enrolled_verification: pd.DataFrame,
    open_set_verification: pd.DataFrame,
    verification_metrics: pd.DataFrame,
) -> None:
    """Render grouped rank profiles and verification DET curves."""
    apply_final_style()

    width_mm, height_mm = 180.0, 100.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    rank_axis = fig.add_axes(
        [24 / width_mm, 16 / height_mm, 78 / width_mm, 64 / height_mm]
    )
    verification_axis = fig.add_axes(
        [115 / width_mm, 16 / height_mm, 60 / width_mm, 64 / height_mm]
    )
    display_models = (MODELS[0], MODELS[2], MODELS[1], MODELS[3], MODELS[4])
    model_order = [model for model, *_ in display_models]
    display_lookup = {model: display for model, display, *_ in display_models}
    y_positions = {
        model: 2 * (len(model_order) - 1 - index)
        for index, model in enumerate(model_order)
    }
    small_sway_colors = (
        SMALL_SWAY_COLOR,
        SMALL_SWAY_LIGHT,
        SMALL_SWAY_PALE,
        "#F1F1F1",
    )
    large_sway_colors = (
        LARGE_SWAY_COLOR,
        LARGE_SWAY_LIGHT,
        LARGE_SWAY_PALE,
        "#F1F1F1",
    )
    condition_specs = (
        ("Small sway", 0.32, small_sway_colors),
        ("Large sway", -0.32, large_sway_colors),
    )
    for condition_label, offset, colors in condition_specs:
        condition_rows = cmc.loc[
            cmc["aggregation"].eq("Window")
            & cmc["condition_label"].eq(condition_label)
        ]
        condition_stats = statistics.loc[
            statistics["aggregation"].eq("Window")
            & statistics["condition_label"].eq(condition_label)
        ].set_index("comparison")
        for model, display, *_ in display_models:
            row = condition_rows.loc[
                condition_rows["model"].eq(model)
                & condition_rows["rank_k"].isin((1, 5, 10))
            ].set_index("rank_k")
            rank1 = float(row.loc[1, "cmc_rate_pct"])
            rank5 = float(row.loc[5, "cmc_rate_pct"])
            rank10 = float(row.loc[10, "cmc_rate_pct"])
            segments = (
                rank1,
                rank5 - rank1,
                rank10 - rank5,
                100.0 - rank10,
            )
            y = y_positions[model] + offset
            left = 0.0
            for segment, color in zip(segments, colors):
                rank_axis.barh(
                    y,
                    segment,
                    left=left,
                    height=0.52,
                    color=color,
                    edgecolor="white",
                    linewidth=0.65,
                    zorder=2,
                )
                left += segment
            rank_axis.barh(
                y,
                100.0,
                height=0.52,
                facecolor="none",
                edgecolor=colors[0],
                linewidth=0.7,
                zorder=3,
            )
            suffix = ""
            if model != "PRISM-Gait":
                p_holm = float(condition_stats.loc[display, "p_holm"])
                suffix = f" {significance_label(p_holm)}"
            rank_axis.text(
                rank1 - 1.5,
                y,
                f"{rank1:.1f}{suffix}",
                ha="right",
                va="center",
                fontsize=VALUE_SIZE,
                fontweight="bold",
                color=TEXT_COLOR,
                zorder=4,
            )
    rank_axis.set_xlim(0.0, 101.5)
    rank_axis.set_ylim(-0.75, 8.75)
    rank_axis.set_xticks([0, 20, 40, 60, 80, 100])
    rank_axis.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    rank_axis.set_yticks([y_positions[model] for model in model_order])
    rank_axis.set_yticklabels([display_lookup[model] for model in model_order])
    rank_axis.set_xlabel("Identified within rank (%)")
    for tick, model in zip(rank_axis.get_yticklabels(), model_order):
        if model == "PRISM-Gait":
            tick.set_fontweight("bold")
    rank_axis.tick_params(axis="y", which="both", length=0, pad=4)
    style_axis(rank_axis)

    condition_legend_handles = [
        Patch(facecolor=small_sway_colors[0], edgecolor="none", label="Small sway"),
        Patch(facecolor=large_sway_colors[0], edgecolor="none", label="Large sway"),
    ]
    fig.legend(
        handles=condition_legend_handles,
        loc="upper center",
        bbox_to_anchor=(63 / width_mm, 0.995),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.8,
        handleheight=1.15,
        columnspacing=1.2,
        borderaxespad=0,
    )
    rank_legend_handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.6, label=label)
        for color, label in zip(
            ("#686868", "#A0A0A0", "#CECECE", "#F1F1F1"),
            ("Rank 1", "Ranks 2-5", "Ranks 6-10", "Outside top 10"),
        )
    ]
    fig.legend(
        handles=rank_legend_handles,
        loc="upper center",
        bbox_to_anchor=(63 / width_mm, 0.945),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        handleheight=1.15,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0,
        fontsize=LEGEND_SIZE,
    )
    draw_verification_curves(
        verification_axis,
        enrolled_verification,
        open_set_verification,
        verification_metrics,
        compact_labels=True,
    )
    for x, panel in ((4, "(a)"), (109, "(b)")):
        fig.text(
            x / width_mm,
            84 / height_mm,
            panel,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    plt.close(fig)


def main() -> None:
    trial_ranks = load_and_validate_scores()
    window_ranks = load_and_validate_window_ranks()
    cmc = pd.concat(
        (
            build_cmc(window_ranks, "Window"),
            build_cmc(trial_ranks, "Trial"),
        ),
        ignore_index=True,
    )
    statistics = pd.concat(
        (
            build_window_rank1_statistics(window_ranks),
            build_trial_rank1_statistics(trial_ranks),
        ),
        ignore_index=True,
    )
    enrolled_verification, open_set_verification = load_verification_scores()
    sway_enrolled = enrolled_verification.loc[
        enrolled_verification["group"].eq("trunk_sway_only")
    ].copy()
    sway_open_set = open_set_verification.loc[
        open_set_verification["group"].eq("trunk_sway_only")
    ].copy()
    verification_metrics = verification_summary(sway_enrolled, sway_open_set)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmc.to_csv(SOURCE_PATH, index=False)
    statistics.to_csv(STATS_PATH, index=False)
    create_figure(
        cmc,
        statistics,
        sway_enrolled,
        sway_open_set,
        verification_metrics,
    )
    print(PDF_PATH)
    print(SOURCE_PATH)
    print(STATS_PATH)


if __name__ == "__main__":
    main()
