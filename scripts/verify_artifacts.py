"""Verify checksummed scientific assets without loading a model."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = json.loads((ROOT / "ARTIFACTS_SHA256.json").read_text())
    for relative, digest in expected.items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch: {relative}")
    print(f"Verified {len(expected)} scientific artifacts.")


if __name__ == "__main__":
    main()
