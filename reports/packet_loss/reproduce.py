"""Reproduce report figures from the bundled raw W&B JSONs, on Windows or Linux.

Run from the repository root: python reports/packet_loss/reproduce.py
Generated files go to the ignored reproduced/ subdirectory, not published assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    manifest = json.loads((base / "PROVENANCE.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = base / record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise SystemExit(f"Missing/changed file: {path}")
    print(f"Verified {len(manifest['files'])} file checksums.", flush=True)
    if args.verify_only:
        return
    common = ["--remaining", str(base / "data/raw/remaining_0_to50"),
              "--battle25", str(base / "data/raw/battle25_0_to50")]
    subprocess.run([sys.executable, str(base / "code/plot_packet_loss_paper_metrics.py"),
                    *common, "--out", str(base / "reproduced/main_0_to50")], check=True)
    subprocess.run([sys.executable, str(base / "code/plot_packet_loss_full_sweep.py"),
                    *common, "--loss100", str(base / "data/raw/loss100"),
                    "--out", str(base / "reproduced/full_0_to100")], check=True)
    print(f"Figures reproduced under {base / 'reproduced'}")


if __name__ == "__main__":
    main()
