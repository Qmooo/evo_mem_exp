#!/usr/bin/env python3
"""
Download all datasets required for Evo-Memory benchmark.

Usage:
    pip install datasets huggingface_hub
    python scripts/download_datasets.py

This script downloads:
- MMLU-Pro: Multi-disciplinary reasoning
- GPQA: Graduate-level science questions
- AIME: Math competition problems
- ToolBench/Berkeley Function Calling: API calling tasks
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def download_mmlu_pro(output_dir: Path):
    """Download MMLU-Pro dataset from HuggingFace."""
    from datasets import load_dataset

    print("\n[1/4] Downloading MMLU-Pro dataset...")

    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        # Convert to list of dicts
        data = []
        for item in tqdm(dataset, desc="Processing MMLU-Pro"):
            data.append({
                "question": item["question"],
                "options": item.get("options", []),
                "answer": item.get("answer", ""),
                "answer_index": item.get("answer_index"),
                "category": item.get("category", "general"),
            })

        # Save to JSON
        output_file = output_dir / "mmlu_pro.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  ✓ Saved {len(data)} samples to {output_file}")
        return len(data)

    except Exception as e:
        print(f"  ✗ Failed to download MMLU-Pro: {e}")
        return 0


def download_gpqa(output_dir: Path):
    """Download GPQA Diamond dataset from HuggingFace."""
    from datasets import load_dataset

    print("\n[2/4] Downloading GPQA Diamond dataset...")

    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

        data = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing GPQA")):
            # Get options and shuffle
            options = []
            correct_answer = item.get("Correct Answer", "")
            for key in ["Correct Answer", "Incorrect Answer 1",
                       "Incorrect Answer 2", "Incorrect Answer 3"]:
                if key in item and item[key]:
                    options.append(item[key])

            # Shuffle options but track correct index
            import random
            random.seed(idx)
            random.shuffle(options)
            correct_idx = options.index(correct_answer) if correct_answer in options else 0

            data.append({
                "question": item["Question"],
                "options": options,
                "answer": chr(65 + correct_idx),  # A, B, C, D
                "correct_answer_text": correct_answer,
                "domain": item.get("High-level domain", "science"),
            })

        output_file = output_dir / "gpqa_diamond.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  ✓ Saved {len(data)} samples to {output_file}")
        return len(data)

    except Exception as e:
        print(f"  ✗ Failed to download GPQA: {e}")
        return 0


def download_aime(output_dir: Path, year: int = 2024):
    """Download AIME dataset from MathArena on HuggingFace.

    2024 is split into two separate repos (I and II) that are merged.
    2025 and 2026 are available as single combined repos.
    """
    from datasets import load_dataset

    print(f"\n[3/4] Downloading AIME {year} dataset...")

    # MathArena is the authoritative source for all years
    DATASET_SOURCES = {
        2024: ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
        2025: ["MathArena/aime_2025"],
        2026: ["MathArena/aime_2026"],
    }

    try:
        sources = DATASET_SOURCES.get(year, [f"MathArena/aime_{year}"])
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

        print(f"  ✓ Saved {len(data)} samples to {output_file}")
        return len(data)

    except Exception as e:
        print(f"  ✗ Failed to download AIME {year}: {e}")

        # Try alternative source
        try:
            print(f"  Trying alternative source...")
            dataset = load_dataset("lighteval/MATH", split="train")

            # Filter for competition problems
            data = []
            count = 0
            for item in dataset:
                if count >= 50:  # Limit samples
                    break
                if "competition" in item.get("type", "").lower() or "olympiad" in str(item).lower():
                    data.append({
                        "problem": item.get("problem", ""),
                        "answer": item.get("solution", ""),
                        "year": year,
                    })
                    count += 1

            if data:
                output_file = output_dir / f"aime_{year}.json"
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  ✓ Saved {len(data)} samples to {output_file}")
                return len(data)
        except:
            pass

        # Create sample problems as fallback
        print("  Creating sample AIME problems...")
        sample_problems = [
            {
                "problem": "Find the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square.",
                "answer": "38",
                "year": year,
            },
            {
                "problem": "Let S be the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square. Find the remainder when S is divided by 1000.",
                "answer": "463",
                "year": year,
            },
            {
                "problem": "Find the number of positive integers less than 1000 that are divisible by 6 but not by 9.",
                "answer": "111",
                "year": year,
            },
            {
                "problem": "The sequence a_1, a_2, ... is geometric with a_1 = a and common ratio r, where a and r are positive integers. Given that log_8(a_1) + log_8(a_2) + ... + log_8(a_12) = 2006, find the number of possible ordered pairs (a, r).",
                "answer": "46",
                "year": year,
            },
            {
                "problem": "Let N be the number of ordered pairs of nonempty sets A and B that have the following properties: A ∪ B = {1,2,3,4,5,6,7,8,9,10,11,12}, |A ∩ B| = 6. Find the remainder when N is divided by 1000.",
                "answer": "772",
                "year": year,
            },
        ]

        output_file = output_dir / f"aime_{year}.json"
        with open(output_file, "w") as f:
            json.dump(sample_problems, f, indent=2)
        print(f"  ✓ Created {len(sample_problems)} sample problems at {output_file}")
        return len(sample_problems)


def download_toolbench(output_dir: Path):
    """Download BFCL v4 simple_python + multiple from GitHub.

    Data:    github.com/ShishirPatil/gorilla main/berkeley-function-call-leaderboard/bfcl_eval/data/
    GT:      …/possible_answer/  (merged by id)

    Output format per item:
      {id, question (plain str), function (list), ground_truth ([{func: {param: [vals]}}]), category}
    """
    import urllib.request

    print("\n[4/4] Downloading BFCL v4 (simple_python + multiple) from GitHub...")

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

        print(f"  ✓ Saved {len(data)} samples to {output_file}")
        return len(data)

    except Exception as e:
        print(f"  ✗ Failed to download BFCL v4: {e}")

        print("  Creating sample ToolBench tasks...")
        sample_tasks = [
            {
                "id": "simple_python_0",
                "question": "Find the area of a triangle with a base of 10 units and height of 5 units.",
                "function": [{"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "dict", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}],
                "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}],
                "category": "simple_python",
            },
            {
                "id": "simple_python_1",
                "question": "Calculate the factorial of 5 using math functions.",
                "function": [{"name": "math.factorial", "description": "Calculate the factorial of a given number.", "parameters": {"type": "dict", "properties": {"number": {"type": "integer", "description": "The number for which factorial needs to be calculated."}}, "required": ["number"]}}],
                "ground_truth": [{"math.factorial": {"number": [5]}}],
                "category": "simple_python",
            },
        ]

        output_file = output_dir / "toolbench.json"
        with open(output_file, "w") as f:
            json.dump(sample_tasks, f, indent=2)
        print(f"  ✓ Created {len(sample_tasks)} sample tasks at {output_file}")
        return len(sample_tasks)


def create_multi_turn_data(output_dir: Path):
    """Create sample data for multi-turn datasets."""
    print("\n[Bonus] Creating sample multi-turn datasets...")

    # AlfWorld tasks
    alfworld_tasks = [
        {"goal": "Put a hot apple in the fridge.", "type": "cool"},
        {"goal": "Put a clean mug in the cabinet.", "type": "clean"},
        {"goal": "Heat the plate and put it on the countertop.", "type": "heat"},
        {"goal": "Pick up the knife from the drawer.", "type": "pick"},
        {"goal": "Put the cooled tomato in the microwave.", "type": "put"},
        {"goal": "Clean the pan and put it on the stove.", "type": "clean"},
        {"goal": "Put a hot mug on the table.", "type": "heat"},
        {"goal": "Cool the apple and put it in the bowl.", "type": "cool"},
        {"goal": "Pick up the book from the shelf.", "type": "pick"},
        {"goal": "Put the cleaned plate in the cabinet.", "type": "clean"},
    ]

    output_file = output_dir / "alfworld.json"
    with open(output_file, "w") as f:
        json.dump(alfworld_tasks, f, indent=2)
    print(f"  ✓ Created {len(alfworld_tasks)} AlfWorld tasks at {output_file}")

    # BabyAI tasks
    babyai_tasks = [
        {"goal": "go to the red ball", "level": "GoTo"},
        {"goal": "pick up the blue key", "level": "Pickup"},
        {"goal": "open the yellow door", "level": "Open"},
        {"goal": "put the green box next to the red ball", "level": "PutNext"},
        {"goal": "go to the blue key then pick it up", "level": "Seq"},
        {"goal": "pick up the red ball or the blue key", "level": "Or"},
        {"goal": "go to the green box and open the yellow door", "level": "And"},
        {"goal": "pick up a ball", "level": "GoToObj"},
        {"goal": "go to an open door", "level": "GoToDoor"},
        {"goal": "pick up the box after you open the door", "level": "SeqS"},
    ]

    output_file = output_dir / "babyai.json"
    with open(output_file, "w") as f:
        json.dump(babyai_tasks, f, indent=2)
    print(f"  ✓ Created {len(babyai_tasks)} BabyAI tasks at {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download Evo-Memory datasets")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        nargs="+",
        choices=["mmlu_pro", "gpqa", "aime", "toolbench", "all"],
        default=["all"],
        help="Datasets to download"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Evo-Memory Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["mmlu_pro", "gpqa", "aime", "toolbench"]

    total_samples = 0

    if "mmlu_pro" in datasets_to_download:
        total_samples += download_mmlu_pro(output_dir)

    if "gpqa" in datasets_to_download:
        total_samples += download_gpqa(output_dir)

    if "aime" in datasets_to_download:
        total_samples += download_aime(output_dir, year=2024)
        total_samples += download_aime(output_dir, year=2025)
        total_samples += download_aime(output_dir, year=2026)

    if "toolbench" in datasets_to_download:
        total_samples += download_toolbench(output_dir)

    # Always create multi-turn sample data
    create_multi_turn_data(output_dir)

    print("\n" + "=" * 60)
    print(f"Download complete! Total samples: {total_samples}")
    print(f"Data saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
