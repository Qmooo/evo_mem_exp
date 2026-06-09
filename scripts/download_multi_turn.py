#!/usr/bin/env python3
"""
Download multi-turn benchmark data (AgentBoard) for Evo-Memory.

Usage:
    python scripts/download_multi_turn.py
    python scripts/download_multi_turn.py --output-dir /path/to/agentboard/data
    python scripts/download_multi_turn.py --datasets alfworld babyai

AgentBoard contains 9 task types; we only need 4.  This script downloads the
full tar.gz but extracts only the test.jsonl for the four datasets used here:

    alfworld     - text-based household tasks (also needs game files; see below)
    babyai       - BabyAI grid-world navigation tasks
    scienceworld - science experiment tasks
    pddl         - PDDL planning tasks (gripper/blocks/barman/tyreworld)
    all          - all four (default)

Each file is written to <output-dir>/<dataset>/test.jsonl.
The output directory should match AGENTBOARD_DATA_PATH.

Post-install steps
------------------
AlfWorld game files (required in addition to test.jsonl):
    alfworld-download --data-dir <output-dir>/alfworld

Multi-turn backend packages (install once per env):
    uv pip install "evo-memory[multi_turn]"   # alfworld, minigrid, scienceworld
    # pddlgym requires Python ≤ 3.11:
    uv pip install "pddlgym==0.0.7"
"""

import os
import sys
import argparse
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


_AGENTBOARD_TAR_URL = (
    "https://huggingface.co/datasets/hkust-nlp/agentboard"
    "/resolve/main/data.tar.gz"
)

# The 4 AgentBoard tasks used by Evo-Memory (out of 9 total in the tarball).
_SUPPORTED_DATASETS = {"alfworld", "babyai", "scienceworld", "pddl"}


def _download_with_progress(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Downloading",
        ) as pbar:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def _strip_data_prefix(name: str) -> str:
    """Remove leading 'data/' prefix (AgentBoard tar packs as data/<dataset>/...)."""
    if name.startswith("data/"):
        return name[len("data/"):]
    return name


def download_agentboard_data(output_dir: Path, datasets: set) -> bool:
    """Stream AgentBoard tar.gz and extract only the needed test.jsonl files.

    The tarball contains 9 task types; only members whose top-level directory
    matches one of the requested dataset names are written to disk.
    """
    print(f"\nDownloading AgentBoard data for: {', '.join(sorted(datasets))}")
    print(f"Source: {_AGENTBOARD_TAR_URL}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "agentboard_data.tar.gz"
            _download_with_progress(_AGENTBOARD_TAR_URL, tar_path)

            print("  Extracting test.jsonl files (skipping unused datasets)...")
            extracted = []
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    rel = _strip_data_prefix(member.name)
                    p = Path(rel)
                    # Keep only: <requested-dataset>/test.jsonl
                    if (
                        p.name == "test.jsonl"
                        and len(p.parts) >= 2
                        and p.parts[0] in datasets
                    ):
                        dest = output_dir / p
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        f = tar.extractfile(member)
                        if f:
                            dest.write_bytes(f.read())
                            extracted.append(dest)
                            print(f"    {dest}")

        if not extracted:
            print(
                "  Warning: no test.jsonl files were found in the tarball.\n"
                "  The AgentBoard tar structure may have changed. Check:\n"
                f"    {_AGENTBOARD_TAR_URL}"
            )
            return False

        print(f"  Extracted {len(extracted)} file(s) to {output_dir}")

        return True

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download multi-turn AgentBoard data for Evo-Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/multi_turn",
        help="Output directory for AgentBoard data (default: ./data/multi_turn).",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=["alfworld", "babyai", "scienceworld", "pddl", "all"],
        default=["all"],
        help="Datasets to extract (default: all four)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.datasets)
    if "all" in datasets:
        datasets = set(_SUPPORTED_DATASETS)

    print("=" * 60)
    print("Evo-Memory Multi-Turn Dataset Downloader")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")

    success = download_agentboard_data(output_dir, datasets)

    print("\n" + "=" * 60)
    if success:
        print("Download complete.")
    else:
        print("Download failed. Check the error above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
