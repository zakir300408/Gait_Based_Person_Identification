from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "reproduction.json"


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def run_modules(modules: list[str]) -> None:
    for module in modules:
        print(f"Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def synchronize_outputs(outputs: dict[str, str]) -> None:
    generated_root = ROOT / "final figures"
    for generated_name, paper_path in outputs.items():
        source = generated_root / generated_name
        destination = ROOT / paper_path
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(destination.relative_to(ROOT))


def main() -> None:
    config = load_config()
    run_modules(config["figure_modules"])
    synchronize_outputs(config["figure_outputs"])


if __name__ == "__main__":
    main()
