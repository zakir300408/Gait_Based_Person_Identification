"""Create Figure 7: verification and fixed-threshold trunk-sway transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold

from analysis.tifs_final_style import (
    LEGEND_SIZE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    PRISM_LIGHT,
    apply_final_style,
)


ROOT = Path(__file__).resolve().parents[1]
STYLE = Path(__file__).with_name("publication-hybrid.mplstyle")
OUT_DIR = ROOT / "baselines_out" / "prism_verification_figure"
# Verification protocols use a dedicated blue family so they remain colored
# without being mistaken for the teal/plum sway-magnitude mapping.
ENROLLED_COLOR = PRISM_COLOR
OPEN_SET_COLOR = PRISM_LIGHT


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    enrolled_path = OUT_DIR / "enrolled_trial_scores.csv"
    open_path = (
        ROOT
        / "baselines_out"
        / "prism_operational_verification_treadmill_v2"
        / "open_set_trial_scores.csv"
    )
    for path in (enrolled_path, open_path):
        if not path.exists():
            raise FileNotFoundError(path)
    enrolled = pd.read_csv(enrolled_path)
    open_set = pd.read_csv(open_path)
    if set(enrolled["model"]) != {"PRISM-Gait"} or set(open_set["model"]) != {"PRISM-Gait"}:
        raise ValueError("Figure 7 accepts PRISM-Gait scores only")
    return enrolled, open_set


def det_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    far, tar, _ = roc_curve(frame["genuine"].to_numpy(dtype=int), frame["score"].to_numpy(dtype=float))
    return far * 100.0, (1.0 - tar) * 100.0


def interpolate_det(frame: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    far, frr = det_values(frame)
    ordered = pd.DataFrame({"far": far, "frr": frr}).groupby("far", as_index=False)["frr"].min()
    return np.interp(grid, ordered["far"], ordered["frr"])


def threshold_at_target_far(impostor_scores: np.ndarray, target_far: float) -> float:
    quantile = np.quantile(impostor_scores, 1.0 - target_far, method="higher")
    return float(np.nextafter(quantile, np.inf))


def verification_summary(enrolled: pd.DataFrame, open_set: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for protocol, frame, fold_column in (
        ("enrolled_participant_verification", enrolled, None),
        ("subject_held_out_open_set_verification", open_set, "fold"),
    ):
        groups = [(0, frame)] if fold_column is None else list(frame.groupby(fold_column, sort=True))
        for fold, group in groups:
            far, tar, _ = roc_curve(group["genuine"].to_numpy(dtype=int), group["score"].to_numpy(dtype=float))
            frr = 1.0 - tar
            index = int(np.argmin(np.abs(far - frr)))
            rows.append(
                {
                    "model": "PRISM-Gait",
                    "protocol": protocol,
                    "evaluation_subset": "trunk_sway_trials",
                    "fold": fold,
                    "auc": float(auc(far, tar)),
                    "eer": float((far[index] + frr[index]) / 2.0),
                    "n_genuine": int(group["genuine"].sum()),
                    "n_impostor": int((group["genuine"] == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def fixed_threshold_transfer(enrolled: pd.DataFrame, target_far: float, seed: int) -> pd.DataFrame:
    subjects = np.asarray(sorted(enrolled["subject"].unique()))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for fold, (development_index, evaluation_index) in enumerate(splitter.split(subjects), 1):
        development_subjects = set(subjects[development_index])
        evaluation_subjects = set(subjects[evaluation_index])
        development = enrolled[
            enrolled["subject"].isin(development_subjects) & (enrolled["group"] == "non_sway")
        ]
        evaluation = enrolled[
            enrolled["subject"].isin(evaluation_subjects) & (enrolled["group"] == "trunk_sway_only")
        ]
        development_impostor = development.loc[development["genuine"] == 0, "score"].to_numpy()
        evaluation_impostor = evaluation.loc[evaluation["genuine"] == 0, "score"].to_numpy()
        evaluation_genuine = evaluation.loc[evaluation["genuine"] == 1, "score"].to_numpy()
        threshold = threshold_at_target_far(development_impostor, target_far)
        rows.append(
            {
                "fold": fold,
                "model": "PRISM-Gait",
                "threshold": threshold,
                "target_development_far": target_far,
                "realized_development_far": float(np.mean(development_impostor >= threshold)),
                "sway_evaluation_far": float(np.mean(evaluation_impostor >= threshold)),
                "sway_evaluation_frr": float(np.mean(evaluation_genuine < threshold)),
                "n_development_participants": len(development_subjects),
                "n_sway_evaluation_participants": len(evaluation_subjects),
                "n_development_impostor_claims": len(development_impostor),
                "n_sway_evaluation_impostor_claims": len(evaluation_impostor),
                "n_sway_evaluation_genuine_claims": len(evaluation_genuine),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return the mean and a percentile bootstrap interval across folds."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(values)), float(lower), float(upper)


def draw_verification_curves(
    ax: plt.Axes,
    enrolled: pd.DataFrame,
    open_set: pd.DataFrame,
    summary: pd.DataFrame,
    compact_labels: bool = False,
) -> None:
    grid = np.geomspace(0.01, 30.0, 500)
    enrolled_eer = 100.0 * summary.loc[
        summary["protocol"] == "enrolled_participant_verification", "eer"
    ].iloc[0]
    open_eer = 100.0 * summary.loc[
        summary["protocol"] == "subject_held_out_open_set_verification", "eer"
    ].mean()

    far, frr = det_values(enrolled)
    keep = (far > 0) & (frr > 0)
    ax.plot(
        far[keep],
        frr[keep],
        color=ENROLLED_COLOR,
        lw=1.35,
        ls="-",
        label=(
            f"Enrolled sway (EER {enrolled_eer:.2f}%)"
            if compact_labels
            else f"Enrolled sway verification (EER {enrolled_eer:.2f}%)"
        ),
        zorder=4,
    )

    fold_curves = [
        interpolate_det(fold_rows, grid)
        for _, fold_rows in open_set.groupby("fold", sort=True)
    ]
    fold_matrix = np.vstack(fold_curves)
    mean_curve = fold_matrix.mean(axis=0)
    ci_half_width = 2.776 * fold_matrix.std(axis=0, ddof=1) / np.sqrt(fold_matrix.shape[0])
    ax.fill_between(
        grid,
        np.clip(mean_curve - ci_half_width, 0.005, None),
        np.clip(mean_curve + ci_half_width, 0.005, None),
        color=OPEN_SET_COLOR,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        grid,
        mean_curve,
        color=OPEN_SET_COLOR,
        lw=1.35,
        ls="--",
        label=(
            f"Subject-held-out sway (EER {open_eer:.2f}%)"
            if compact_labels
            else f"Subject-held-out open-set EER (sway; {open_eer:.2f}%)"
        ),
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 30)
    ax.set_ylim(0.01, 30)
    visible_decades = (0.01, 0.1, 1.0, 10.0)
    ax.xaxis.set_major_locator(mticker.FixedLocator(visible_decades))
    ax.yaxis.set_major_locator(mticker.FixedLocator(visible_decades))
    ax.set_xlabel("False-accept rate (%)")
    ax.set_ylabel("False-reject rate (%)")
    ax.minorticks_on()
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        borderaxespad=0,
        fontsize=LEGEND_SIZE,
        handlelength=2.0,
        markerscale=1.35,
    )


def draw_threshold_transfer(
    ax: plt.Axes,
    transfer: pd.DataFrame,
    target_far: float,
) -> None:
    ordered = transfer.sort_values("fold")
    metrics = (
        ordered["sway_evaluation_far"].to_numpy(dtype=float) * 100.0,
        ordered["sway_evaluation_frr"].to_numpy(dtype=float) * 100.0,
    )
    centers = np.asarray([0.0, 1.0])
    jitter = np.asarray([-0.055, -0.028, 0.0, 0.028, 0.055])
    for index, (center, values) in enumerate(zip(centers, metrics)):
        ax.scatter(
            center + jitter,
            values,
            s=17,
            facecolor="white",
            edgecolor=ENROLLED_COLOR,
            linewidth=0.7,
            label="Fold" if index == 0 else None,
            zorder=2,
        )
        mean, lower, upper = bootstrap_mean_ci(values, 420 + 10 * index)
        ax.errorbar(
            center,
            mean,
            yerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="o",
            ms=5.2,
            mfc=ENROLLED_COLOR,
            mec=ENROLLED_COLOR,
            ecolor=ENROLLED_COLOR,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.8,
            label="Mean (95% CI)" if index == 0 else None,
            zorder=4,
        )
        ax.text(
            center - 0.10 if index == 0 else center + 0.10,
            mean + 0.04,
            f"{mean:.2f}%",
            ha="right" if index == 0 else "left",
            va="center",
            fontsize=7.8,
            color="#333333",
        )

    ax.hlines(
        target_far * 100.0,
        -0.42,
        0.42,
        color="#7F7F7F",
        lw=0.8,
        ls=":",
        zorder=0,
    )
    ax.text(
        -0.40,
        -0.02,
        f"Development FAR target: {target_far * 100:g}%",
        ha="left",
        va="top",
        fontsize=7.2,
        color="#555555",
    )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.35, 8.2)
    ax.set_yticks(np.arange(0, 9, 1))
    ax.set_xticks(centers, ["Sway FAR", "Sway FRR"])
    ax.set_ylabel("Error rate (%)")
    ax.minorticks_off()
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.8,
        markerscale=1.35,
        handletextpad=0.5,
    )


def create_figure(
    enrolled: pd.DataFrame,
    open_set: pd.DataFrame,
    summary: pd.DataFrame,
    transfer: pd.DataFrame,
    output: Path,
) -> None:
    apply_final_style()
    mm = 1.0 / 25.4
    figure = plt.figure(figsize=(180 * mm, 82 * mm))
    left = figure.add_axes((15 / 180, 14 / 82, 72 / 180, 59 / 82))
    right = figure.add_axes((103 / 180, 14 / 82, 72 / 180, 59 / 82))
    draw_verification_curves(left, enrolled, open_set, summary)
    draw_threshold_transfer(
        right,
        transfer,
        float(transfer["target_development_far"].iloc[0]),
    )
    figure.text(
        4 / 180, 76 / 82, "(a)", ha="left", va="top",
        fontweight="bold", fontsize=PANEL_LABEL_SIZE
    )
    figure.text(
        92 / 180, 76 / 82, "(b)", ha="left", va="top",
        fontweight="bold", fontsize=PANEL_LABEL_SIZE
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", dpi=600, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "final figures" / "openset_closed set validation.pdf",
    )
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    enrolled, open_set = load_scores()
    sway_enrolled = enrolled[enrolled["group"] == "trunk_sway_only"].copy()
    sway_open_set = open_set[open_set["group"] == "trunk_sway_only"].copy()
    if sway_enrolled.empty or sway_open_set.empty:
        raise ValueError("trunk-sway verification scores are missing")
    summary = verification_summary(sway_enrolled, sway_open_set)
    summary.to_csv(OUT_DIR / "verification_curve_metrics.csv", index=False)
    transfer = fixed_threshold_transfer(enrolled, args.target_far, args.seed)
    transfer.to_csv(OUT_DIR / "fixed_threshold_transfer_fold_metrics.csv", index=False)
    create_figure(sway_enrolled, sway_open_set, summary, transfer, args.output)
    print(transfer[["realized_development_far", "sway_evaluation_far", "sway_evaluation_frr"]].agg(["mean", "std"]))
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
