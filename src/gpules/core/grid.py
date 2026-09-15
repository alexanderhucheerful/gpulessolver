"""网格定义。

把原项目散落的 ``DX/DY/DZ/NX/NY/NZ`` 全局常量收进一个不可变网格对象，
并提供高度坐标等派生量。所有算子的格距都从这里取，杜绝"魔法数字漂移"。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float = 5.0
    dy: float = 5.0
    dz: float = 5.0
    device: torch.device = dataclasses.field(default_factory=lambda: torch.device("cpu"))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def domain(self) -> tuple[float, float, float]:
        return (self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)

    def z_coords(self) -> np.ndarray:
        """垂直层中心高度 (m)。"""
        return np.arange(self.nz) * self.dz + self.dz / 2

    def x_coords(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx + self.dx / 2

    def y_coords(self) -> np.ndarray:
        return np.arange(self.ny) * self.dy + self.dy / 2

    def to(self, device: torch.device) -> "Grid":
        return dataclasses.replace(self, device=device)
