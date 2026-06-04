"""ToolBench dataset loader — Berkeley Function Calling Leaderboard v4.

Loads BFCL v4 simple_python and multiple categories from GitHub.
Ground truth lives in possible_answer/ and is merged by id at load time.

GT format: [{"func_name": {"param": [acceptable_val, ...]}}]
  - "" in the acceptable list means the parameter may be omitted.
"""

import ast as pyast
import json
import re
import urllib.request
from typing import Any, Dict, List, Optional

from ..base import DatasetSplit, SingleTurnDataset, TaskInstance

_BFCL_CATEGORIES = ["simple_python", "multiple"]

_GITHUB_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla"
    "/main/berkeley-function-call-leaderboard/bfcl_eval/data"
)


class ToolBenchDataset(SingleTurnDataset):
    """
    Berkeley Function Calling Leaderboard (BFCL) v4 dataset.

    Covers two categories:
    - simple_python: single-function calls with Python-typed parameters
    - multiple: choose the correct function from several candidates

    Ground truth format: [{"func_name": {"param": [val, ...]}}]
    where "" in the value list means the parameter may be omitted.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        category: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(data_path, split, **kwargs)
        self.category = category

    @property
    def name(self) -> str:
        return f"toolbench_{self.category}" if self.category else "toolbench"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> List[TaskInstance]:
        categories = [self.category] if self.category else _BFCL_CATEGORIES

        instances = self._load_from_github(categories)
        if not instances and self.data_path:
            instances = self._load_from_local()
        if not instances:
            instances = self._create_sample_tasks()
        return instances

    def _load_from_github(self, categories: List[str]) -> List[TaskInstance]:
        instances: List[TaskInstance] = []
        for cat in categories:
            try:
                data_url = f"{_GITHUB_BASE}/BFCL_v4_{cat}.json"
                gt_url = f"{_GITHUB_BASE}/possible_answer/BFCL_v4_{cat}.json"

                data_items = self._fetch_jsonl(data_url)
                gt_items = self._fetch_jsonl(gt_url)
                gt_map = {item["id"]: item for item in gt_items}

                instances.extend(self._parse_items(data_items, gt_map, cat))
            except Exception as e:
                print(f"Warning: GitHub download failed for {cat}: {e}")
        return instances

    def _fetch_jsonl(self, url: str) -> List[Dict]:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8").strip()
        if raw.startswith("["):
            return json.loads(raw)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def _parse_items(
        self, items: List[Dict], gt_map: Dict[str, Dict], category: str
    ) -> List[TaskInstance]:
        instances: List[TaskInstance] = []
        for item in items:
            question = self._extract_question(item.get("question", ""))
            if not question:
                continue

            funcs: List[Dict] = item.get("function", [])
            item_id = item.get("id", f"toolbench_{len(instances)}")

            gt_entry = gt_map.get(item_id, {})
            target = json.dumps(gt_entry.get("ground_truth", [])) if gt_entry else ""

            instances.append(TaskInstance(
                task_id=item_id,
                input_text=self._build_input(question, funcs),
                target=target,
                metadata={"functions": funcs, "category": category},
                domain="api",
                difficulty=self._estimate_difficulty(funcs),
            ))
        return instances

    def _load_from_local(self) -> List[TaskInstance]:
        instances: List[TaskInstance] = []
        try:
            with open(self.data_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")
            return instances

        for idx, item in enumerate(data):
            question = self._extract_question(item.get("question", ""))
            if not question:
                continue

            funcs = item.get("function", item.get("functions", []))
            if isinstance(funcs, dict):
                funcs = [funcs]

            gt = item.get("ground_truth", [])
            target = json.dumps(gt) if gt else item.get("answer", "")

            instances.append(TaskInstance(
                task_id=item.get("id", f"toolbench_{idx}"),
                input_text=self._build_input(question, funcs),
                target=target,
                metadata={"functions": funcs, "category": item.get("category", "api")},
                domain=item.get("category", "api"),
                difficulty=self._estimate_difficulty(funcs),
            ))
        return instances

    def _create_sample_tasks(self) -> List[TaskInstance]:
        funcs = [{"name": "calculate_triangle_area", "description": "Calculate the area of a triangle given its base and height.", "parameters": {"type": "dict", "properties": {"base": {"type": "integer", "description": "The base of the triangle."}, "height": {"type": "integer", "description": "The height of the triangle."}, "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"}}, "required": ["base", "height"]}}]
        gt = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
        return [TaskInstance(
            task_id="simple_python_0",
            input_text=self._build_input("Find the area of a triangle with a base of 10 units and height of 5 units.", funcs),
            target=json.dumps(gt),
            metadata={"functions": funcs, "category": "simple_python"},
            domain="api",
            difficulty="easy",
        )]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_question(self, question_field: Any) -> str:
        """Handle v4 nested [[{role, content}]] and plain string formats."""
        if isinstance(question_field, str):
            return question_field
        if isinstance(question_field, list) and question_field:
            turn = question_field[0]
            if isinstance(turn, list) and turn:
                msg = turn[0]
                if isinstance(msg, dict):
                    return msg.get("content", "")
            if isinstance(turn, dict):
                return turn.get("content", "")
        return ""

    def _build_input(self, question: str, funcs: List[Dict]) -> str:
        if not funcs:
            return question
        func_lines = []
        for f in funcs:
            if not isinstance(f, dict):
                continue
            name = f.get("name", "?")
            desc = f.get("description", "")
            props = f.get("parameters", {}).get("properties", {})
            required = f.get("parameters", {}).get("required", [])
            parts = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                parts.append(f"{pname}: {ptype}" + ("" if pname in required else " (optional)"))
            sig = f"{name}({', '.join(parts)})" if parts else name
            func_lines.append(f"- {sig}: {desc}")
        return question + "\n\nAvailable APIs:\n" + "\n".join(func_lines)

    def _estimate_difficulty(self, functions: List[Dict]) -> str:
        n = len(functions)
        if n > 5:
            return "hard"
        if n > 1:
            return "medium"
        return "easy"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """
        Evaluate against BFCL v4 ground truth.

        Target (JSON string): [{"func_name": {"param": [acceptable_val, ...]}}]
        Prediction: "func(param=val, ...)" call string or {"name":..., "arguments":{...}} JSON.
        """
        pred = self._parse_call(prediction)
        gt_list = self._parse_target(target)

        if not gt_list:
            return {"correct": False, "api_accuracy": False, "arg_accuracy": False,
                    "prediction": prediction, "target": target}

        gt_entry = gt_list[0]
        gt_func_name = next(iter(gt_entry))
        gt_params: Dict[str, List] = gt_entry[gt_func_name]

        pred_name = pred.get("name", "").strip()
        pred_args = pred.get("arguments", {})

        api_correct = pred_name.lower() == gt_func_name.lower()
        args_correct = self._check_args(pred_args, gt_params) if api_correct else False

        return {
            "correct": api_correct and args_correct,
            "api_accuracy": api_correct,
            "arg_accuracy": args_correct,
            "prediction": prediction,
            "target": target,
        }

    def _parse_target(self, target: str) -> List[Dict]:
        try:
            obj = json.loads(target)
            return obj if isinstance(obj, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    @staticmethod
    def _standardize_string(s: str) -> str:
        return re.sub(r"[ ,./\-_*^]", "", str(s)).lower().replace("'", '"')

    def _check_args(self, pred_args: Dict, gt_params: Dict[str, List]) -> bool:
        # Reject unexpected parameters not present in GT
        for param in pred_args:
            if param not in gt_params:
                return False

        for param, acceptable in gt_params.items():
            optional = "" in acceptable
            actual_vals = [v for v in acceptable if v != ""]

            if param not in pred_args:
                if not optional:
                    return False
                continue

            pred_val = pred_args[param]

            # Infer expected type from first GT value; allow int→float promotion
            if actual_vals:
                expected_type = type(actual_vals[0])
                if expected_type is float and isinstance(pred_val, int):
                    pred_val = float(pred_val)

            if not any(self._values_match(pred_val, v) for v in actual_vals):
                return False

        return True

    def _values_match(self, pred_val: Any, gt_val: Any) -> bool:
        if isinstance(gt_val, str) and isinstance(pred_val, str):
            return self._standardize_string(pred_val) == self._standardize_string(gt_val)
        return pred_val == gt_val

    def _parse_call(self, call_str: str) -> Dict[str, Any]:
        """Parse a function call string or JSON into {name, arguments}."""
        call_str = call_str.strip()

        # JSON format: {"name": ..., "arguments": {...}}
        try:
            obj = json.loads(call_str)
            if isinstance(obj, dict) and "name" in obj:
                return {"name": obj["name"], "arguments": obj.get("arguments", {})}
        except (json.JSONDecodeError, ValueError):
            pass

        # Python call format: func_name(param=val, ...)
        try:
            tree = pyast.parse(call_str, mode="eval")
            node = tree.body
            if isinstance(node, pyast.Call):
                func = node.func
                if isinstance(func, pyast.Name):
                    name = func.id
                elif isinstance(func, pyast.Attribute):
                    name = pyast.unparse(func)
                else:
                    name = call_str.split("(")[0].strip()

                args: Dict[str, Any] = {}
                for kw in node.keywords:
                    try:
                        args[kw.arg] = pyast.literal_eval(kw.value)
                    except Exception:
                        args[kw.arg] = pyast.unparse(kw.value)
                return {"name": name, "arguments": args}
        except SyntaxError:
            pass

        m = re.match(r"^([\w.]+)\s*\(", call_str)
        return {"name": m.group(1) if m else "", "arguments": {}}
