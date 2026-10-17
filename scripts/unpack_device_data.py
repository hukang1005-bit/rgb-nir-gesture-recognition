"""Restore original device CSV traces and verify their SHA-256 digests."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "device" / "raw_csv"


def main():
    expected = json.loads((ROOT / "SHA256.json").read_text())
    for name, digest in expected.items():
        target = ROOT / name
        temporary = ROOT / (name + ".tmp")
        try:
            with gzip.open(ROOT / (name + ".gz"), "rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError("Decompressed checksum mismatch: " + name)
            temporary.replace(target)
            print("Restored " + name)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
