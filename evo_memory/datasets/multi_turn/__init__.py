"""Multi-turn dataset loaders."""

from .alfworld import AlfWorldDataset
from .babyai import BabyAIDataset
from .scienceworld import ScienceWorldDataset

# pddlgym==0.0.7 requires legacy `gym` (not gymnasium) and pillow<10.
# Requires Python ≤ 3.11; incompatible with Python 3.12+.
# Install via: uv sync --extra evo_mem_multi  (on Python 3.11)
try:
    from .pddl import PDDLDataset
except ImportError as _pddl_err:
    import warnings
    warnings.warn(
        f"PDDLDataset unavailable ({_pddl_err}). "
        "Install pddlgym==0.0.7 on Python ≤ 3.11: uv sync --extra evo_mem_multi",
        ImportWarning,
        stacklevel=2,
    )
    PDDLDataset = None  # type: ignore[assignment,misc]

__all__ = [
    "AlfWorldDataset",
    "BabyAIDataset",
    "PDDLDataset",
    "ScienceWorldDataset",
]
