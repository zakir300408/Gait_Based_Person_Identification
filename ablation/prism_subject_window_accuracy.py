"""Compute participant window accuracy for the exact full PRISM-Gait model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ablation.mblite_ablation_study import (
    COMPONENT_VARIANTS,
    evaluate_variant,
    load_original_h5_windowed,
    set_seed,
    write_csv,
)


@dataclass(frozen=True)
class Settings:
    h5_path: Path
    output_dir: Path
    seed: int
    acc_std_min: float
    pkd_groups: int
    pkd_kernels_per_group: int
    pkd_dilations: tuple[int, ...]
    batch_size: int


def create_logger(output_dir: Path) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"prism_subject_accuracy_{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_dir / f"prism_subject_accuracy_{timestamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def run(settings: Settings) -> tuple[Path, Path]:
    if not settings.h5_path.is_file():
        raise FileNotFoundError(f"missing treadmill H5: {settings.h5_path.name}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(settings.output_dir)
    logger.info("step_01: loading two-second treadmill windows")
    set_seed(settings.seed)
    data = load_original_h5_windowed(settings.h5_path, 2.0, settings.acc_std_min)
    full = next(item for item in COMPONENT_VARIANTS if item["variant"] == "MB-Lite")
    logger.info("step_02: fitting full PRISM-Gait participants=%s", len(data["subjects"]))
    rows, subject_rows, dimensions = evaluate_variant(data, full, settings)
    for row in rows:
        row["study"] = "exact_prism"
        row["variant"] = "PRISM-Gait"
    for row in subject_rows:
        row["study"] = "exact_prism"
        row["variant"] = "PRISM-Gait"
    summary_path = settings.output_dir / "prism_condition_summary.csv"
    subject_path = settings.output_dir / "prism_subject_condition_accuracy.csv"
    write_csv(summary_path, rows)
    write_csv(subject_path, subject_rows)
    metadata = {
        "settings": {
            "h5_file": settings.h5_path.name,
            "output_dir": settings.output_dir.name,
            "seed": settings.seed,
            "acc_std_min": settings.acc_std_min,
            "pkd_groups": settings.pkd_groups,
            "pkd_kernels_per_group": settings.pkd_kernels_per_group,
            "pkd_dilations": list(settings.pkd_dilations),
            "batch_size": settings.batch_size,
        },
        "dimensions": dimensions,
    }
    (settings.output_dir / "prism_subject_accuracy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("step_03: saved feature_dimensions=%s subject_rows=%s", sum(dimensions.values()), len(subject_rows))
    return summary_path, subject_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run(Settings(root / "gait_id_pelvis_62.h5", root / "baselines_out/prism_subject_window_accuracy", 42, 0.4, 48, 6, (1, 2, 4, 8), 256))
    print(*result, sep="\n")
