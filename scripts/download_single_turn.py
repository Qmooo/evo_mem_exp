#!/usr/bin/env python3
"""
Download single-turn benchmark datasets for Evo-Memory.

Usage:
    python scripts/download_single_turn.py
    python scripts/download_single_turn.py --datasets gpqa aime
    python scripts/download_single_turn.py --output-dir /path/to/data

Datasets:
    mmlu_pro   MMLU-Pro          (TIGER-Lab/MMLU-Pro on HuggingFace)
    gpqa       GPQA Diamond      (Idavidrein/gpqa on HuggingFace)
    aime       AIME 2024/25/26   (MathArena on HuggingFace)
    toolbench  BFCL v4           (ShishirPatil/gorilla on GitHub)
    all        all of the above  (default)
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def download_mmlu_pro(output_dir: Path) -> int:
    from datasets import load_dataset

    print("\n[1/4] Downloading MMLU-Pro...")
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        data = []
        for item in tqdm(dataset, desc="Processing MMLU-Pro"):
            data.append({
                "question": item["question"],
                "options": item.get("options", []),
                "answer": item.get("answer", ""),
                "answer_index": item.get("answer_index"),
                "category": item.get("category", "general"),
            })
        output_file = output_dir / "mmlu_pro.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(data)} samples → {output_file}")
        return len(data)
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def download_gpqa(output_dir: Path) -> int:
    from datasets import load_dataset
    import random

    print("\n[2/4] Downloading GPQA Diamond...")
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        data = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing GPQA")):
            correct_answer = item.get("Correct Answer", "")
            options = [
                item[k]
                for k in ("Correct Answer", "Incorrect Answer 1",
                          "Incorrect Answer 2", "Incorrect Answer 3")
                if k in item and item[k]
            ]
            random.seed(idx)
            random.shuffle(options)
            correct_idx = options.index(correct_answer) if correct_answer in options else 0
            data.append({
                "question": item["Question"],
                "options": options,
                "answer": chr(65 + correct_idx),
                "correct_answer_text": correct_answer,
                "domain": item.get("High-level domain", "science"),
            })
        output_file = output_dir / "gpqa_diamond.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(data)} samples → {output_file}")
        return len(data)
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def download_aime(output_dir: Path, year: int = 2024) -> int:
    """Download AIME for a given year from MathArena on HuggingFace.

    2024 is split into two repos (I and II); 2025/2026 are single repos.
    """
    from datasets import load_dataset

    print(f"\n[3/4] Downloading AIME {year}...")

    SOURCES = {
        2024: ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
        2025: ["MathArena/aime_2025"],
        2026: ["MathArena/aime_2026"],
    }
    try:
        sources = SOURCES.get(year, [f"MathArena/aime_{year}"])
        data = []
        for dataset_name in sources:
            dataset = load_dataset(dataset_name, split="train")
            part_label = dataset_name.split("_")[-1] if len(sources) > 1 else None
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                entry = {
                    "problem": item.get("problem", ""),
                    "answer": str(item.get("answer", "")),
                    "year": year,
                    "problem_idx": item.get("problem_idx"),
                }
                if part_label in ("I", "II"):
                    entry["part"] = part_label
                if "problem_type" in item:
                    entry["problem_type"] = item["problem_type"]
                data.append(entry)
        output_file = output_dir / f"aime_{year}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(data)} samples → {output_file}")
        return len(data)
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def download_toolbench(output_dir: Path) -> int:
    """Download BFCL v4 simple_python + multiple from GitHub.

    Source: ShishirPatil/gorilla main/berkeley-function-call-leaderboard/bfcl_eval/data/
    Output format per item: {id, question (str), function (list), ground_truth, category}
    """
    import urllib.request

    print("\n[4/4] Downloading BFCL v4 (simple_python + multiple)...")

    GITHUB_BASE = (
        "https://raw.githubusercontent.com/ShishirPatil/gorilla"
        "/main/berkeley-function-call-leaderboard/bfcl_eval/data"
    )
    CATEGORIES = ["simple_python", "multiple"]

    def fetch_jsonl(url: str):
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8").strip()
        if raw.startswith("["):
            return json.loads(raw)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def extract_question(q):
        if isinstance(q, str):
            return q
        if isinstance(q, list) and q:
            turn = q[0]
            if isinstance(turn, list) and turn:
                msg = turn[0]
                return msg.get("content", "") if isinstance(msg, dict) else ""
        return ""

    try:
        data = []
        for cat in CATEGORIES:
            items = fetch_jsonl(f"{GITHUB_BASE}/BFCL_v4_{cat}.json")
            gt_items = fetch_jsonl(f"{GITHUB_BASE}/possible_answer/BFCL_v4_{cat}.json")
            gt_map = {item["id"]: item.get("ground_truth", []) for item in gt_items}
            for item in tqdm(items, desc=f"Processing {cat}"):
                question = extract_question(item.get("question", ""))
                if not question:
                    continue
                item_id = item.get("id", "")
                data.append({
                    "id": item_id,
                    "question": question,
                    "function": item.get("function", []),
                    "ground_truth": gt_map.get(item_id, []),
                    "category": cat,
                })
        output_file = output_dir / "toolbench.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(data)} samples → {output_file}")
        return len(data)
    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download single-turn Evo-Memory datasets")
    parser.add_argument(
        "--output-dir", "-o",
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=["mmlu_pro", "gpqa", "aime", "toolbench", "all"],
        default=["all"],
        help="Datasets to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(args.datasets)
    if "all" in datasets:
        datasets = {"mmlu_pro", "gpqa", "aime", "toolbench"}

    print("=" * 60)
    print("Evo-Memory Single-Turn Dataset Downloader")
    print("=" * 60)
    print(f"Output: {output_dir.absolute()}")

    total = 0
    if "mmlu_pro" in datasets:
        total += download_mmlu_pro(output_dir)
    if "gpqa" in datasets:
        total += download_gpqa(output_dir)
    if "aime" in datasets:
        for year in (2024, 2025, 2026):
            total += download_aime(output_dir, year=year)
    if "toolbench" in datasets:
        total += download_toolbench(output_dir)

    print("\n" + "=" * 60)
    print(f"Done. Total samples: {total}")
    print(f"Data saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
