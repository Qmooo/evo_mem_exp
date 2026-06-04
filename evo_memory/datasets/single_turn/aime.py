"""AIME dataset loader.

AIME (American Invitational Mathematics Examination) contains
Olympiad-style mathematics problems requiring symbolic reasoning.
"""

from typing import List, Dict, Any, Optional
import json
import re

from ..base import SingleTurnDataset, TaskInstance, DatasetSplit


class AIMEDataset(SingleTurnDataset):
    """
    AIME dataset for mathematical problem solving.

    Contains challenging math problems with exact-match evaluation.
    Supports AIME 2024, 2025, and 2026 via MathArena on HuggingFace.
    2024 is split into two separate repos (I and II) that are merged.
    """

    # MathArena is the authoritative source for all years
    _HF_SOURCES: dict = {
        2024: ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
        2025: ["MathArena/aime_2025"],
        2026: ["MathArena/aime_2026"],
    }

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        year: int = 2024,  # 2024, 2025, or 2026
        **kwargs,
    ):
        super().__init__(data_path, split, **kwargs)
        self.year = year

    @property
    def name(self) -> str:
        return f"aime_{self.year}"

    def _load_data(self) -> List[TaskInstance]:
        """Load AIME data from MathArena (HuggingFace), falling back to local JSON."""
        instances = []

        try:
            from datasets import load_dataset

            sources = self._HF_SOURCES.get(self.year, [f"MathArena/aime_{self.year}"])
            idx = 0
            for dataset_name in sources:
                dataset = load_dataset(dataset_name, split="train")
                part = dataset_name.split("_")[-1] if len(sources) > 1 else None
                for item in dataset:
                    problem = item.get("problem", "")
                    answer = str(item.get("answer", ""))
                    metadata = {"year": self.year, "problem_number": idx + 1}
                    if part in ("I", "II"):
                        metadata["part"] = part
                    if "problem_type" in item:
                        metadata["problem_type"] = item["problem_type"]
                    instances.append(TaskInstance(
                        task_id=f"aime_{self.year}_{idx}",
                        input_text=problem,
                        target=answer,
                        metadata=metadata,
                        domain="mathematics",
                        difficulty=self._estimate_difficulty(idx),
                    ))
                    idx += 1

        except Exception as e:
            print(f"Warning: Could not load AIME from HuggingFace: {e}")
            if self.data_path:
                instances = self._load_from_local()
            else:
                instances = self._create_sample_problems()

        return instances

    def _load_from_local(self) -> List[TaskInstance]:
        """Load from local JSON file."""
        instances = []

        try:
            with open(self.data_path) as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                instances.append(TaskInstance(
                    task_id=f"aime_{self.year}_{idx}",
                    input_text=item["problem"],
                    target=str(item["answer"]),
                    metadata={"year": self.year},
                    domain="mathematics",
                    difficulty=self._estimate_difficulty(idx),
                ))

        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")

        return instances

    def _create_sample_problems(self) -> List[TaskInstance]:
        """Create sample AIME-style problems for testing."""
        problems = [
            {
                "problem": "Find the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square.",
                "answer": "38",
            },
            {
                "problem": "Let S be the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square. Find the remainder when S is divided by 1000.",
                "answer": "463",
            },
            {
                "problem": "Find the number of positive integers less than 1000 that are divisible by 6 but not by 9.",
                "answer": "111",
            },
        ]

        return [
            TaskInstance(
                task_id=f"aime_{self.year}_{idx}",
                input_text=p["problem"],
                target=p["answer"],
                metadata={"year": self.year},
                domain="mathematics",
                difficulty="hard",
            )
            for idx, p in enumerate(problems)
        ]

    def _estimate_difficulty(self, problem_number: int) -> str:
        """Estimate difficulty based on problem number."""
        # AIME problems generally increase in difficulty
        if problem_number < 5:
            return "easy"
        elif problem_number < 10:
            return "medium"
        return "hard"

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """
        Evaluate AIME prediction with exact match.

        AIME answers are integers from 000 to 999.
        """
        # Extract numerical answer
        pred_match = re.search(r"(\d+)", prediction.strip())
        pred_num = pred_match.group(1) if pred_match else prediction

        target_match = re.search(r"(\d+)", target.strip())
        target_num = target_match.group(1) if target_match else target

        # Normalize to integer comparison
        try:
            pred_int = int(pred_num)
            target_int = int(target_num)
            is_correct = pred_int == target_int
        except ValueError:
            is_correct = pred_num == target_num

        return {
            "correct": is_correct,
            "exact_match": is_correct,
            "prediction": pred_num,
            "target": target_num,
        }
