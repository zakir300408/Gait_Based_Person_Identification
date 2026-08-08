"""Create Figure 8: accuracy on two public gait-identification datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, NullLocator

from figs.tifs_final_style import (
    BASELINE_MID,
    HEADER_SIZE,
    PANEL_LABEL_SIZE,
    PRISM_COLOR,
    TICK_SIZE,
    apply_final_style,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "baselines_out" / "public_external_performance_figure"
PDF_PATH = ROOT / "final figures" / "external dataset validation.pdf"
PREVIEW_PATH = OUT_DIR / "fig8_public_external_performance_preview.png"
STYLE = Path(__file__).with_name("publication-hybrid.mplstyle")

PRISM_MODEL_COLOR = PRISM_COLOR
SLATE = BASELINE_MID
GRAY = "#7F7F7F"
LIGHT = "#D9D9D9"
BLACK = "#1A1A1A"
MODEL_ORDER = [
    "PRISM-Gait",
    "CNN+MHA",
    "FCN-BiLSTM",
    "ResNet1D",
    "MiniROCKET",
    "MultiROCKET",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    whu_summary = read_json(
        ROOT
        / "baselines_out"
        / "whugait_exact_mblite_full_registration"
        / "summary.json"
    )
    whu_local = pd.read_csv(
        ROOT
        / "baselines_out"
        / "public_whugait_dataset3"
        / "whugait_dataset3_public_benchmark_combined.csv"
    )
    ou_summary = read_json(
        ROOT
        / "baselines_out"
        / "ou_isir_exact_prism_registration"
        / "summary.json"
    )
    ou_mini = pd.read_csv(
        ROOT
        / "baselines_out"
        / "public_imu_benchmark_ou_full"
        / "public_imu_benchmark.csv"
    )
    ou_extra = pd.read_csv(
        ROOT
        / "baselines_out"
        / "public_imu_benchmark_ou_matched_additional"
        / "public_imu_benchmark.csv"
    )

    whu = [
        ("PRISM-Gait", 100 * whu_summary["window_accuracy"], "Local"),
        ("CNN+MHA", 95.55, "Published"),
        (
            "ResNet1D",
            100
            * float(
                whu_local.loc[
                    whu_local["model"] == "ResNet1D public", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
        (
            "MiniROCKET",
            100
            * float(
                whu_local.loc[
                    whu_local["model"] == "MiniROCKET + ridge", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
        ("FCN-BiLSTM", 95.29, "Published"),
        (
            "MultiROCKET",
            100
            * float(
                whu_local.loc[
                    whu_local["model"] == "MultiROCKET + ridge", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
    ]

    ou = [
        ("PRISM-Gait", 100 * ou_summary["window_accuracy"], "Local"),
        ("CNN+MHA", 98.58, "Published"),
        ("FCN-BiLSTM", 98.30, "Published"),
        (
            "MiniROCKET",
            100
            * float(
                ou_mini.loc[
                    ou_mini["model"] == "MiniROCKET + ridge", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
        (
            "ResNet1D",
            100
            * float(
                ou_extra.loc[
                    ou_extra["model"] == "ResNet1D public", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
        (
            "MultiROCKET",
            100
            * float(
                ou_extra.loc[
                    ou_extra["model"] == "MultiROCKET + ridge", "accuracy"
                ].iloc[0]
            ),
            "Local",
        ),
    ]
    columns = ["model", "accuracy", "source"]
    return pd.DataFrame(whu, columns=columns), pd.DataFrame(ou, columns=columns)


def accuracy_panel(
    ax,
    data: pd.DataFrame,
    x_limits: tuple[float, float],
    x_ticks: list[float],
) -> None:
    data = data.set_index("model").loc[MODEL_ORDER].reset_index()
    y = np.arange(len(data))[::-1]
    x_min, x_max = x_limits

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", top=False, right=False)
    ax.grid(False)

    for index, row in data.iterrows():
        yi = y[index]
        is_prism = row["model"] == "PRISM-Gait"
        color = PRISM_MODEL_COLOR if is_prism else SLATE
        ax.hlines(yi, x_min, row["accuracy"], color=LIGHT, lw=1.0, zorder=1)
        ax.scatter(
            row["accuracy"],
            yi,
            s=52 if is_prism else 34,
            marker="o",
            facecolor=color,
            edgecolor=color,
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            row["accuracy"] + 0.025 * (x_max - x_min),
            yi,
            f"{row['accuracy']:.2f}%",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold" if is_prism else "normal",
            color=BLACK,
        )

    labels = data["model"].tolist()
    ax.set_yticks(y, labels)
    for tick, model in zip(ax.get_yticklabels(), labels):
        tick.set_fontweight("bold" if model == "PRISM-Gait" else "normal")
    ax.set_xlim(*x_limits)
    ax.set_xticks(x_ticks)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.set_xlabel("Window-level identification accuracy (%)")
    ax.xaxis.set_minor_locator(MultipleLocator((x_ticks[1] - x_ticks[0]) / 2))
    ax.yaxis.set_minor_locator(NullLocator())


def save_evidence(whu: pd.DataFrame, ou: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = pd.concat(
        [
            whu.assign(dataset="WHU-Gait Dataset #3"),
            ou.assign(dataset="OU-ISIR"),
        ],
        ignore_index=True,
    )
    evidence.to_csv(OUT_DIR / "figure8_model_performance_evidence.csv", index=False)
    caption = (
        "**Figure 8 | Identification accuracy on public gait datasets.** "
        "(a) Window-level identification accuracy on the official WHU-Gait Dataset #3 "
        "split (26,283 enrollment and 2,991 test windows; 118 identities; "
        "unconstrained smartphone IMU). (b) Window-level identification accuracy on "
        "the Zou-preprocessed OU-ISIR split (13,212 enrollment and 1,409 test "
        "windows; 745 identities; waist-mounted IMU). The CNN+MHA and "
        "FCN-BiLSTM accuracies are quoted published results under the corresponding "
        "dataset protocols; PRISM-Gait, ResNet1D, MiniROCKET, and MultiROCKET are "
        "local matched-protocol evaluations. PRISM-Gait uses the same 13,049-feature "
        "architecture in both datasets with a separately fitted identity head."
    )
    (OUT_DIR / "figure8_caption.md").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    whu, ou = load_inputs()
    save_evidence(whu, ou)
    apply_final_style()

    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 76 * mm), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.17,
        right=0.975,
        bottom=0.22,
        top=0.79,
        wspace=0.12,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    accuracy_panel(ax_a, whu, (94.8, 96.05), [95.0, 95.5, 96.0])
    accuracy_panel(ax_b, ou, (96.8, 99.35), [97.0, 98.0, 99.0])
    ax_b.tick_params(axis="y", left=False, labelleft=False)

    for ax, panel, dataset, domain in [
        (ax_a, "(a)", "WHU-Gait Dataset #3", "Unconstrained smartphone IMU"),
        (ax_b, "(b)", "OU-ISIR", "Waist-mounted IMU"),
    ]:
        box = ax.get_position()
        fig.text(
            box.x0 - 0.055,
            box.y1 + 0.095,
            panel,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            color=BLACK,
        )
        fig.text(
            box.x0,
            box.y1 + 0.070,
            dataset,
            ha="left",
            va="bottom",
            fontsize=HEADER_SIZE,
            fontweight="bold",
            color=BLACK,
        )
        fig.text(
            box.x0,
            box.y1 + 0.035,
            domain,
            ha="left",
            va="bottom",
            fontsize=TICK_SIZE,
            color=GRAY,
        )

    fig.savefig(PDF_PATH, format="pdf", facecolor="white")
    fig.savefig(PREVIEW_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
