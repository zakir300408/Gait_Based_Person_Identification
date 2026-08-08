from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "release" / "MANIFEST.sha256"
ROOT_FILES = (
    ".gitignore",
    "CITATION.cff",
    "LICENSE.md",
    "README.md",
    "environment.yml",
    "environment-deep.yml",
    "environment-inceptiontime.yml",
    "gait_id_pelvis_62.h5",
)
DIRECTORIES = (
    "ablation",
    "baselines_out",
    "config",
    "figs",
    "paper",
    "scripts",
    "tests",
    "third_party",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    excluded_parts = {"__pycache__", ".pytest_cache", "dist", "compare"}
    return sorted(
        path for path in files if not excluded_parts.intersection(path.relative_to(ROOT).parts)
    )


def main() -> None:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in selected_files()]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {len(lines)} checksums to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
