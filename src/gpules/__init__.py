"""gpules — GPU 大涡模拟（统一合并包）。

合并两条研究版本线：高精度数值线（v9）+ 可微/可学习线（diff-v1）。
"""

from .config import GPULESConfig
from .core.boundary import BuildingFields
from .core.grid import Grid
from .core.solver import LESSolver
from .core.state import FlowState
from .diff.params import LearnablePhysicsParams
from .diff.trainer import Trainer
from .version import LINEAGE, __version__

__all__ = [
    "__version__", "LINEAGE", "GPULESConfig", "Grid", "FlowState",
    "LESSolver", "BuildingFields", "LearnablePhysicsParams", "Trainer",
]
