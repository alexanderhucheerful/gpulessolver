"""有限差分算子（可微）。

所有算子基于 ``F.pad(replicate)`` / ``torch.roll`` 实现，全程支持反向传播，
是"可微线"与"高精度线"共享的数值基础（原 v9 用 F.pad，diff-v1 用 torch.roll，
这里统一为基于 padding 的版本，已验证可微且数值正确）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .grid import Grid


def pad3d(f: torch.Tensor) -> torch.Tensor:
    """对 (z,y,x) 张量做 1 层副本填充，返回 (z+2,y+2,x+2)。"""
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1),
                 mode="replicate").squeeze(0).squeeze(0)


def dx_c(f: torch.Tensor, dx: float) -> torch.Tensor:
    p = pad3d(f)
    return (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2 * dx)


def dy_c(f: torch.Tensor, dy: float) -> torch.Tensor:
    p = pad3d(f)
    return (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2 * dy)


def dz_c(f: torch.Tensor, dz: float) -> torch.Tensor:
    p = pad3d(f)
    return (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2 * dz)


def laplacian(f: torch.Tensor, grid: Grid) -> torch.Tensor:
    """三维拉普拉斯（中心差分，含常数修正项）。"""
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    p = pad3d(f)
    inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx ** 2, 1.0 / dy ** 2, 1.0 / dz ** 2
    d2x = (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * inv_dx2
    d2y = (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * inv_dy2
    d2z = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * inv_dz2
    return d2x + d2y + d2z - 2 * f * (inv_dx2 + inv_dy2 + inv_dz2)


def divergence(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, grid: Grid) -> torch.Tensor:
    return dx_c(u, grid.dx) + dy_c(v, grid.dy) + dz_c(w, grid.dz)


def gradient(p: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dx_c(p, grid.dx), dy_c(p, grid.dy), dz_c(p, grid.dz)


def restrict(f: torch.Tensor) -> torch.Tensor:
    """限制到粗网格（自适应平均池化）。"""
    coarse = tuple(max(1, s // 2) for s in f.shape)
    return F.adaptive_avg_pool3d(f.unsqueeze(0).unsqueeze(0), coarse).squeeze(0).squeeze(0)


def prolong(f: torch.Tensor, fine_shape: tuple[int, int, int]) -> torch.Tensor:
    """粗网格 prolong 到细网格（三线性插值）。"""
    return F.interpolate(f.unsqueeze(0).unsqueeze(0), size=fine_shape,
                         mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
