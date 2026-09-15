"""压力泊松求解器（可插拔策略）。

原 v9 提供多网格/CG/Jacobi，diff-v1 提供 RFFT。合并后四种皆可选：

  * ``MultigridSolver`` — 几何多网格 V-cycle（inference 默认，收敛快）
  * ``ConjugateGradient`` — 共轭梯度（备选）
  * ``JacobiSolver``      — 简单 Jacobi（教学/调试）
  * ``PoissonFFTSolver``  — 谱求解（train 默认，省 ~50% 显存且全程可微；
    **假设三方向周期边界**，非周期物理域上为 surrogate 近似，与 multigrid 非严格等价）

统一接口 ``solve(rhs, p_prev, grid)``。所有求解器都先减去 rhs 均值以施加
"压力定义到浮点"边界条件（与两版一致）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from .grid import Grid
from .operators import pad3d, prolong, restrict


def _laplacian_xyz(f: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    p = pad3d(f)
    inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx ** 2, 1.0 / dy ** 2, 1.0 / dz ** 2
    d2x = (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) * inv_dx2
    d2y = (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) * inv_dy2
    d2z = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * inv_dz2
    return d2x + d2y + d2z - 2 * f * (inv_dx2 + inv_dy2 + inv_dz2)


class PressureSolver(ABC):
    @abstractmethod
    def solve(self, rhs: torch.Tensor, p_prev: torch.Tensor, grid: Grid) -> torch.Tensor:
        ...


class MultigridSolver(PressureSolver):
    """几何多网格 V-cycle（来自 v9）。"""

    def __init__(self, levels: int = 4, vcycles: int = 3, smooth: int = 3, omega: float = 0.667,
                 tol: float = 1e-6):
        self.levels, self.vcycles, self.smooth, self.omega, self.tol = levels, vcycles, smooth, omega, tol

    def _smooth(self, p, rhs, dx, dy, dz, n):
        inv_d = 1.0 / (2.0 * (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
        for _ in range(n):
            pp = pad3d(p)
            ns = (pp[1:-1, 1:-1, 2:] / dx ** 2 + pp[1:-1, 1:-1, :-2] / dx ** 2
                  + pp[1:-1, 2:, 1:-1] / dy ** 2 + pp[1:-1, :-2, 1:-1] / dy ** 2
                  + pp[2:, 1:-1, 1:-1] / dz ** 2 + pp[:-2, 1:-1, 1:-1] / dz ** 2)
            p_new = (ns - rhs) * inv_d
            p = self.omega * p_new + (1 - self.omega) * p
        return p

    def _vcycle(self, p, rhs, level, max_level, dx, dy, dz):
        if level >= max_level or min(p.shape) <= 4:
            return self._smooth(p, rhs, dx, dy, dz, 50)
        p = self._smooth(p, rhs, dx, dy, dz, self.smooth)
        r = rhs - _laplacian_xyz(p, dx, dy, dz)
        r = r - r.mean()
        r_c = restrict(r)
        p_c = torch.zeros_like(r_c)
        p_c = self._vcycle(p_c, r_c, level + 1, max_level, dx * 2, dy * 2, dz * 2)
        p = p + prolong(p_c, p.shape)
        p = self._smooth(p, rhs, dx, dy, dz, self.smooth)
        return p

    def solve(self, rhs, p_prev, grid):
        rhs = rhs - rhs.mean()
        p = p_prev.clone()
        for _ in range(self.vcycles):
            p = self._vcycle(p, rhs, 0, self.levels, grid.dx, grid.dy, grid.dz)
            if float(((rhs - _laplacian_xyz(p, grid.dx, grid.dy, grid.dz)) ** 2).sum()) < self.tol:
                break
        return p - p.mean()


class ConjugateGradient(PressureSolver):
    """共轭梯度法（来自 v9）。"""

    def __init__(self, max_iters: int = 150, tol: float = 1e-6):
        self.max_iters, self.tol = max_iters, tol

    def solve(self, rhs, p_prev, grid):
        rhs = rhs - rhs.mean()
        p = p_prev.clone()
        r = rhs - _laplacian_xyz(p, grid.dx, grid.dy, grid.dz)
        pk = r.clone()
        rr = float((r * r).sum())
        for i in range(self.max_iters):
            Ap = _laplacian_xyz(pk, grid.dx, grid.dy, grid.dz)
            pAp = float((pk * Ap).sum())
            if abs(pAp) < 1e-20:
                break
            alpha = rr / pAp
            p = p + alpha * pk
            r = r - alpha * Ap
            rr_new = float((r * r).sum())
            if rr_new < self.tol:
                break
            pk = r + (rr_new / (rr + 1e-20)) * pk
            rr = rr_new
        return p - p.mean()


class JacobiSolver(PressureSolver):
    """简单 Jacobi 迭代（固定迭代次数，调试/教学用）。"""

    def __init__(self, n_iters: int = 200, omega: float = 0.667):
        self.n_iters, self.omega = n_iters, omega

    def solve(self, rhs, p_prev, grid):
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        inv_d = 1.0 / (2.0 * (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
        rhs = rhs - rhs.mean()
        p = p_prev.clone()
        for _ in range(self.n_iters):
            pp = pad3d(p)
            ns = (pp[1:-1, 1:-1, 2:] / dx ** 2 + pp[1:-1, 1:-1, :-2] / dx ** 2
                  + pp[1:-1, 2:, 1:-1] / dy ** 2 + pp[1:-1, :-2, 1:-1] / dy ** 2
                  + pp[2:, 1:-1, 1:-1] / dz ** 2 + pp[:-2, 1:-1, 1:-1] / dz ** 2)
            p_new = (ns - rhs) * inv_d
            p = self.omega * p_new + (1 - self.omega) * p
        return p - p.mean()


class PoissonFFTSolver(PressureSolver):
    """谱（RFFT）泊松求解器（来自 diff-v1，可微、省显存）。

    在构造时预计算 ``1/lap_eig``；``solve`` 做 rfftn/irfftn。谱算子假设三方向
    周期边界；对入流/出流/顶盖等非周期域，train 线将其作为可微 surrogate 使用。
    """

    def __init__(self, device: torch.device = None):
        self._poisson_inv = None
        self._device = device

    def prepare(self, grid: Grid) -> None:
        dev = grid.device
        kx = torch.fft.rfftfreq(grid.nx, d=grid.dx, device=dev) * 2 * np.pi
        ky = torch.fft.fftfreq(grid.ny, d=grid.dy, device=dev) * 2 * np.pi
        kz = torch.fft.fftfreq(grid.nz, d=grid.dz, device=dev) * 2 * np.pi
        KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")
        lap_eig = -(KX ** 2 + KY ** 2 + KZ ** 2)
        lap_eig[0, 0, 0] = -1e-10
        self._poisson_inv = 1.0 / lap_eig
        self._device = dev

    def solve(self, rhs, p_prev, grid):
        if self._poisson_inv is None:
            self.prepare(grid)
        rhs_mean = rhs.mean()
        rhs_c = rhs - rhs_mean
        rhs_hat = torch.fft.rfftn(rhs_c)
        p_hat = rhs_hat * self._poisson_inv
        p_hat[0, 0, 0] = 0
        p = torch.fft.irfftn(p_hat, s=rhs.shape)
        return p


def build_pressure_solver(cfg) -> PressureSolver:
    if cfg.pressure_solver == "multigrid":
        return MultigridSolver(cfg.mg_levels, cfg.mg_vcycles, cfg.mg_smooth, cfg.jacobi_omega, cfg.pressure_tol)
    elif cfg.pressure_solver == "cg":
        return ConjugateGradient(cfg.cg_iters, cfg.pressure_tol)
    elif cfg.pressure_solver == "jacobi":
        return JacobiSolver(200, cfg.jacobi_omega)
    elif cfg.pressure_solver == "poisson_fft":
        return PoissonFFTSolver()
    raise ValueError(f"未知 pressure_solver: {cfg.pressure_solver}")
