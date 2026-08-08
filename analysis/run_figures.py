"""Rebuild the manuscript result figures from results.zip."""

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "analysis.figures.identification_overview",
    "analysis.figures.design_selection_and_ablation",
    "analysis.figures.identification_and_verification",
)
OUTPUTS = {
    "treadmill identification benchmark.pdf": "treadmill_identification_benchmark.pdf",
    "operational timing.pdf": "operational_timing.pdf",
    "design selection and ablation.pdf": "design_selection_and_ablation.pdf",
    "identification rank robustness.pdf": "identification_rank_robustness.pdf",
}


def main() -> None:
    if not (ROOT / "baselines_out").is_dir():
        raise FileNotFoundError(
            "results are not extracted; run Expand-Archive results.zip -DestinationPath ."
        )
    for module in MODULES:
        print(f"Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
    for source_name, destination_name in OUTPUTS.items():
        source = ROOT / "final figures" / source_name
        if not source.is_file():
            raise FileNotFoundError(f"figure was not created: {source_name}")
        shutil.copy2(source, ROOT / "paper" / destination_name)


if __name__ == "__main__":
    main()
