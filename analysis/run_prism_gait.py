"""Run the PRISM-Gait analyses reported in the paper."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "analysis.prism_gait.participant_accuracy",
    "analysis.prism_gait.architecture_selection",
    "analysis.prism_gait.window_duration",
    "analysis.prism_gait.nested_selection",
)


def main() -> None:
    dataset = ROOT / "data" / "altergait" / "gait_id_pelvis_62.h5"
    if not dataset.is_file():
        raise FileNotFoundError(f"missing AlterGait file: {dataset.relative_to(ROOT)}")
    if not (ROOT / "baselines_out").is_dir():
        raise FileNotFoundError(
            "supporting results are not extracted; "
            "run Expand-Archive results.zip -DestinationPath ."
        )
    for module in MODULES:
        print(f"Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
