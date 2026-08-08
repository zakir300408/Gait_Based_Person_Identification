from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "reproduction.json"
OUTPUT_DIR = ROOT / "release" / "dist"
ARCHIVE_PATH = OUTPUT_DIR / "prism_gait_derived_evidence.zip"
HASH_PATH = OUTPUT_DIR / "prism_gait_derived_evidence.sha256"


def selected_files(patterns: list[str]) -> list[Path]:
    files = {path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()}
    forbidden = {"gait_id_pelvis_62.h5", ".pt", ".pth", ".keras", ".hdf5"}
    selected = [path for path in files if path.name not in forbidden and path.suffix not in forbidden]
    if not selected:
        raise ValueError("Figshare package selection is empty")
    return sorted(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    files = selected_files(config["figshare_globs"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    digest = sha256(ARCHIVE_PATH)
    HASH_PATH.write_text(f"{digest}  {ARCHIVE_PATH.name}\n", encoding="ascii")
    print(f"Packaged {len(files)} files: {ARCHIVE_PATH}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
