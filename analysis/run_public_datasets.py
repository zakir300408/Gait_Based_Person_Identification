"""Run PRISM-Gait on WHU-Gait Dataset #3 and OU-ISIR."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "whu-gait": (
        "whugait_dataset3",
        ROOT / "data" / "whu_gait" / "Dataset #3" / "Dataset #3",
        ROOT / "baselines_out" / "whugait_exact_prism_registration",
    ),
    "ou-isir": (
        "ou_isir_identification",
        ROOT / "data" / "ou_isir" / "identification" / "data",
        ROOT / "baselines_out" / "ou_isir_exact_prism_registration",
    ),
}


def selected_datasets() -> list[str]:
    if len(sys.argv) != 2 or sys.argv[1] not in {*DATASETS, "all"}:
        choices = ", ".join((*DATASETS, "all"))
        raise SystemExit(f"Usage: python -m analysis.run_public_datasets <{choices}>")
    return list(DATASETS) if sys.argv[1] == "all" else [sys.argv[1]]


def main() -> None:
    for name in selected_datasets():
        dataset_id, data_path, output_path = DATASETS[name]
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"missing {name} data at {data_path.relative_to(ROOT)}; see data/README.md"
            )
        command = [
            sys.executable,
            "-m",
            "analysis.datasets.prism_registration",
            "--dataset",
            dataset_id,
            "--out-dir",
            str(output_path),
        ]
        print(f"Running PRISM-Gait on {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
