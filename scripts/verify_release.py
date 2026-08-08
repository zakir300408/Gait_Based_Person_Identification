from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import h5py


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "reproduction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(config: dict) -> None:
    path = ROOT / config["path"]
    if sha256(path) != config["sha256"]:
        raise ValueError(f"dataset checksum mismatch: {path}")
    with h5py.File(path, "r") as handle:
        participants = [key for key in handle.keys() if key != "index"]
        if len(participants) != config["participant_groups"]:
            raise ValueError(f"unexpected participant count: {len(participants)}")
        if int(handle.attrs["sampling_rate_default_hz"]) != config["sampling_rate_hz"]:
            raise ValueError("unexpected sampling rate")


def verify_figures(outputs: dict[str, str]) -> None:
    for destination in outputs.values():
        path = ROOT / destination
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    rank_path = ROOT / outputs["identification rank robustness.pdf"]
    with fitz.open(rank_path) as document:
        text = "\n".join(page.get_text() for page in document)
    if "Subject-held-out sway" not in text or "Open-set sway" in text:
        raise ValueError("Figure 5 verification terminology is stale")


def verify_tables() -> None:
    required = (
        "table_1a_dataset_protocol_metadata.csv",
        "table_2_verification_operating_points.csv",
        "table_2_verification_fold_details.csv",
        "table_3_external_benchmark_identification.csv",
    )
    for name in required:
        path = ROOT / "paper" / "tables" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)


def verify_portability() -> None:
    forbidden = ("E:\\Zakir_phd", "C:\\Users\\admin")
    for folder in (ROOT / "ablation", ROOT / "figs", ROOT / "scripts"):
        for path in folder.glob("*.py"):
            text = path.read_text(encoding="utf-8-sig")
            if any(value in text for value in forbidden):
                raise ValueError(f"machine-specific path in {path.relative_to(ROOT)}")


def verify_manifest() -> None:
    manifest = ROOT / "release" / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_dataset(config["dataset"])
    verify_figures(config["figure_outputs"])
    verify_tables()
    verify_portability()
    verify_manifest()
    print("Release verification passed")


if __name__ == "__main__":
    main()
