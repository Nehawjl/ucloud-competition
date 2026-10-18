#!/usr/bin/env python3
"""Verify the original competition archive and its byte-identical extracted files."""
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "ecc1296a5773c12b1acf5ad137b7bc9ea88f55e7ec82a34e93c7fd00ffad36ad"
EXPECTED_FILES = {"README.md", "run.sh", "runner.py"}


def verify(root: Path = ROOT) -> dict:
    archive = root / "third_party/eval.zip"
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("Competition eval.zip SHA-256 mismatch")
    entries = {}
    with zipfile.ZipFile(archive) as package:
        if set(package.namelist()) != EXPECTED_FILES or len(package.infolist()) != 3:
            raise ValueError("Unexpected competition archive entries")
        for name in sorted(EXPECTED_FILES):
            original = package.read(name)
            extracted = (root / "third_party/official-eval" / name).read_bytes()
            if original != extracted:
                raise ValueError(f"Official evaluation file differs from archive: {name}")
            entries[name] = {"sha256": hashlib.sha256(original).hexdigest(), "bytes": len(original)}
    return {"archive": "third_party/eval.zip", "sha256": archive_hash,
            "status": "byte-identical", "files": entries}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
