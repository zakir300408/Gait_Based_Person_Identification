"""Run the public comparator architectures used in the paper."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "rocket": ("prism-gait", "analysis.comparators.rocket", None),
    "fcn-bilstm": ("prism-gait-deep", "analysis.comparators.fcn_bilstm", None),
    "inceptiontime": (
        "gaitid-inceptiontime",
        "analysis.comparators.inceptiontime",
        ROOT / "external" / "InceptionTime",
    ),
    "multi-model-lstm": (
        "prism-gait-deep",
        "analysis.comparators.multi_model_lstm",
        ROOT / "external" / "Gait_Recognition_LSTM",
    ),
}


def selected_models() -> list[str]:
    if len(sys.argv) != 2 or sys.argv[1] not in {*MODELS, "all"}:
        choices = ", ".join((*MODELS, "all"))
        raise SystemExit(f"Usage: python -m analysis.run_benchmark_models <{choices}>")
    return list(MODELS) if sys.argv[1] == "all" else [sys.argv[1]]


def run_model(name: str) -> None:
    environment_name, module, external_source = MODELS[name]
    environment = os.environ.copy()
    if external_source is not None:
        if not external_source.is_dir():
            raise FileNotFoundError(
                f"missing public source at {external_source.relative_to(ROOT)}; "
                "see analysis/comparators/README.md"
            )
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(external_source), existing) if value
        )
    command = ["conda", "run", "-n", environment_name, "python", "-m", module]
    print(f"Running {name}: {module}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    for name in selected_models():
        run_model(name)


if __name__ == "__main__":
    main()
