"""Dataset module for Evo-Memory."""

from .base import BaseDataset, TaskInstance, DatasetSplit
from .single_turn import (
    MMLUProDataset,
    GPQADataset,
    AIMEDataset,
    ToolBenchDataset,
)

# Multi-turn loaders need the multi_turn extra (gymnasium, alfworld, ...), so they
# are imported on first access; single-turn use works without those packages.
_MULTI_TURN = ("AlfWorldDataset", "BabyAIDataset", "PDDLDataset", "ScienceWorldDataset")


def __getattr__(name):
    if name in _MULTI_TURN:
        from . import multi_turn
        return getattr(multi_turn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseDataset",
    "TaskInstance",
    "DatasetSplit",
    # Single-turn
    "MMLUProDataset",
    "GPQADataset",
    "AIMEDataset",
    "ToolBenchDataset",
    # Multi-turn
    "AlfWorldDataset",
    "BabyAIDataset",
    "PDDLDataset",
    "ScienceWorldDataset",
]
