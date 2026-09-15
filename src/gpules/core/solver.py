"""LES 求解器编排器（合并两版的统一入口）。

这是整个包的中心：把网格/算子/平流/压力/SGS/边界/时间积分组装成一个有状态求解器。

  * ``mode="inference"`` → 高精度线（5 阶 WS + 多网格 + SSP-RK3 + 虚拟位温 SGS）
  * ``mode="train"``      → 可微线（central2 + Jameson + RFFT 压力 + Heun + 可学习参数）

相对原项目的工程化改进（商业交付所必需）：
  * 设备自动选择（cuda/cpu/mps 回退），不再硬编码 ``device='cuda'``
  * NaN/发散/显存守卫，数值爆炸不再静默跑完或崩溃
  * dt 自适应（CFL）+ 散度超限折半
  * 所有张量经 ``check_finite`` 校验
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..config import GPULESConfig
from ..device import device_info, seed_all, select_device
from ..diff.params import LearnablePhysicsParams
from ..exceptions import DivergenceError, OOMError
from ..io.data import InitData, interp_bc_time
from ..logging import get_logger
from .advection import Central2, build_advection
from .boundary import (
    BuildingFields,
    TrainAux,
    build_building_fields,
    build_train_aux,
)
from .diff_helpers import fused_step_diff
from .grid import Grid
from .operators import divergence
from .pressure import PoissonFFTSolver, build_pressure_solver
from .sgs import DeardorffTKE
from .state import FlowState
from .timestepping import SSPRK3, build_integrator


class LESSolver:
    def __init__(self, cfg: GPULESConfig):
        self.cfg = cfg
        seed_all(cfg.seed)
        strict_dev = (cfg.device or "auto").lower() not in ("auto", "cpu")
        self.device = select_device(cfg.device, strict=strict_dev)
        self.grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, self.device)
        self.logger = get_logger("core.solver")

        # 可微模式强制使用已验证的谱/中心组合（保证梯度连通）
        if cfg.mode == "train":
            self.advection = Central2(ad_k2=cfg.ad_k2, ad_k4=cfg.ad_k4)
            self.pressure = PoissonFFTSolver()
            self.pressure.prepare(self.grid)
            self.integrator = None
            self.logger.info("train 模式：平流=central2, 压力=poisson_fft, 积分=heun(diff)")
        else:
            self.advection = build_advection(cfg)
            self.pressure = build_pressure_solver(cfg)
            if isinstance(self.pressure, PoissonFFTSolver):
                self.pressure.prepare(self.grid)
            self.integrator = build_integrator(cfg)

        self.sgs = DeardorffTKE(cfg)
        self.state: Optional[FlowState] = None
        self.bf: Optional[BuildingFields] = None
        self.aux: Optional[TrainAux] = None
        self.params: Optional[LearnablePhysicsParams] = None
        self.bc: dict = {}
        self.bc_times = np.array([0.0])
        self.ug = 0.0
        self.vg = 0.0
        self.t_sim = 0.0
        self.step_count = 0
        self._divergence_free_done = False
        self._div_halve_streak = 0

        info = device_info(self.device)
        self.logger.info("设备: %s", info.get("name", str(self.device)))

    # ── 准备初始场 ─────────────────────────────────────────────────
    def prepare(self, init: InitData, bldg_2d: Optional[np.ndarray] = None) -> "LESSolver":
        self.bc = init.bc
        self.bc_times = init.bc_times
        self.ug, self.vg = init.ug, init.vg
        self.state = FlowState.from_numpy(self.grid, {
            "u": init.u, "v": init.v, "w": init.w, "pt": init.pt, "qv": init.qv,
        })
        if bldg_2d is None or self.cfg.no_buildings:
            dev = self.device
            shape = self.grid.shape
            self.bf = BuildingFields(
                torch.ones(shape, device=dev), torch.zeros(shape, device=dev),
                torch.zeros(shape, device=dev), torch.zeros(shape, device=dev),
                torch.zeros(self.grid.ny, self.grid.nx, device=dev), soft=False)
            self.logger.info("无建筑数据，使用开阔地形")
        else:
            self.bf = build_building_fields(self.grid, torch.tensor(bldg_2d), self.cfg)

        if self.cfg.mode == "train":
            u_ref_1d = torch.tensor(init.u.mean(axis=(1, 2)), device=self.device)
            v_ref_1d = torch.tensor(init.v.mean(axis=(1, 2)), device=self.device)
            self.aux = build_train_aux(self.grid, self.bf, self.cfg, u_ref_1d, v_ref_1d)
            self.params = LearnablePhysicsParams().to(self.device)
        if self.cfg.mode == "inference" and isinstance(self.integrator, SSPRK3):
            self.integrator.prepare(self.cfg, self.grid, self.bf)
        return self

    # ── 散度自由初始化（压力投影） ──────────────────────────────────
    def divergence_free_init(self) -> None:
        assert self.state is not None and self.bf is not None
        with torch.no_grad():
            div0 = divergence(self.state.u, self.state.v, self.state.w, self.grid)
            p0 = self.pressure.solve(div0 / self.cfg.dt_init, torch.zeros_like(self.state.u), self.grid)
            self.state.u = (self.state.u - self.cfg.dt_init * _dp(self.state.u, p0, self.grid, 0)) * self.bf.bldg_mask3d
            self.state.v = (self.state.v - self.cfg.dt_init * _dp(self.state.v, p0, self.grid, 1)) * self.bf.bldg_mask3d
            self.state.w = (self.state.w - self.cfg.dt_init * _dp(self.state.w, p0, self.grid, 2)) * self.bf.bldg_mask3d
            self.state.w[0, :, :] = 0
            self.state.w[-1, :, :] = 0
        self._divergence_free_done = True

    def add_perturbation(self, amp: Optional[float] = None) -> None:
        assert self.state is not None
        if amp is None:
            amp = 0.1 * max(abs(float(self.state.u.mean())), abs(float(self.state.v.mean())), 0.5)
        with torch.no_grad():
            self.state.u += amp * (torch.rand_like(self.state.u) - 0.5) * 2 * self.bf.bldg_mask3d
            self.state.v += amp * (torch.rand_like(self.state.v) - 0.5) * 2 * self.bf.bldg_mask3d
            self.state.w += amp * 0.3 * (torch.rand_like(self.state.w) - 0.5) * 2 * self.bf.bldg_mask3d

    # ── 单步 ───────────────────────────────────────────────────────
    def step(self, dt: float) -> None:
        assert self.state is not None and self.bf is not None
        # 静态边界（bc_times 仅一个时刻）缓存一次，避免每步重建 dict
        if getattr(self, "_bc_t_cache", None) is not None:
            bc_t = self._bc_t_cache
        else:
            bc_np = interp_bc_time(self.bc, self.bc_times, self.t_sim)
            bc_t = {k: torch.tensor(arr, device=self.device, dtype=torch.float32) for k, arr in bc_np.items()}

        if self.cfg.mode == "inference":
            self.state = self.integrator.step(
                self.state, dt, self.cfg, self.grid,
                lambda f, u, v, w: self.advection.advect(f, u, v, w, self.grid),
                self.pressure, self.sgs, self.bf, bc_t, self.ug, self.vg)
        else:
            assert self.params is not None and self.aux is not None
            u, v, w, pt, tke = fused_step_diff(
                self.state.u, self.state.v, self.state.w, self.state.pt, self.state.tke, dt,
                self.params, self.aux, self.cfg, self.grid, self.pressure, self.advection)
            self.state = FlowState(self.grid, u, v, w, pt,
                                   self.state.qv.clone() if self.state.qv is not None else None,
                                   tke, self.state.p.clone())

        self.state.check_finite(where="step", step=self.step_count)
        self.t_sim += dt
        self.step_count += 1

    # ── 运行主循环 ─────────────────────────────────────────────────
    def run(self, end_time: Optional[float] = None, record_every: int = 10) -> list[dict]:
        """运行模拟直到 end_time（默认 cfg.end_time）。

        返回每 ``record_every`` 步的诊断列表（用于输出对比数据）。
        """
        assert self.state is not None
        end_time = end_time or self.cfg.end_time
        dt = self.cfg.dt_init
        log: list[dict] = []
        import time as _time
        wall0 = _time.time()
        # 静态边界（bc_times 仅一个时刻）缓存一次，避免每步重建 dict
        self._bc_t_cache = None
        if self.cfg.mode == "inference" and len(self.bc_times) <= 1:
            bc_np = interp_bc_time(self.bc, self.bc_times, 0.0)
            self._bc_t_cache = {k: torch.tensor(arr, device=self.device, dtype=torch.float32)
                                for k, arr in bc_np.items()}
        if not self._divergence_free_done and self.cfg.mode == "inference":
            self.divergence_free_init()
            self.add_perturbation()

        try:
            while self.t_sim < end_time - 1e-9:
                self.step(dt)
                if self.cfg.mode == "inference":
                    dt = self._adapt_dt(dt)
                if self.step_count % record_every == 0:
                    stats = self.state.statistics()
                    log.append({"t": float(self.t_sim), "spd_max": stats["spd_max"],
                                "spd_mean": stats["spd_mean"], "div_max": stats["div_max"],
                                "tke_mean": float(self.state.tke.mean())})
        except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - 依赖 GPU
            peak = torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else 0
            raise OOMError(f"CUDA 显存不足: {e}", peak_gb=peak) from e
        elapsed = _time.time() - wall0
        self.logger.info("模拟完成: %d 步 / %.1fs, 墙钟 %.1fs", self.step_count, self.t_sim, elapsed)
        return log

    def _adapt_dt(self, dt: float) -> float:
        """CFL 自适应 + 散度超限折半（来自 v9）。"""
        assert self.state is not None
        with torch.no_grad():
            spd_max = float(torch.sqrt(self.state.u ** 2 + self.state.v ** 2 + self.state.w ** 2).max())
            div_max = float(self.state.divergence().abs().max())
        dt_new = self.cfg.cfl_max * self.cfg.dx / max(spd_max, 0.1)
        dt = float(torch.clamp(torch.tensor(dt_new), self.cfg.dt_min, self.cfg.dt_max).item())
        if div_max > self.cfg.div_max:
            dt = max(self.cfg.dt_min, dt * 0.5)
            self._div_halve_streak += 1
            if self._div_halve_streak >= 10 and div_max > self.cfg.div_max:
                raise DivergenceError(
                    f"散度持续超限 ({div_max:.4g} > {self.cfg.div_max})，dt 已折半 {self._div_halve_streak} 次",
                    div_max=div_max, step=self.step_count,
                )
        else:
            self._div_halve_streak = 0
        return dt


def _dp(field, p, grid, vel_dim):
    """按速度分量取压力梯度（用于散度自由初始化）。

    vel_dim: 0=u(∂p/∂x), 1=v(∂p/∂y), 2=w(∂p/∂z)
    """
    if vel_dim == 0:
        return _grad_x(p, grid)
    if vel_dim == 1:
        return _grad_y(p, grid)
    return _grad_z(p, grid)


def _grad_x(p, grid):
    from .operators import pad3d
    pp = pad3d(p)
    return (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) / (2 * grid.dx)


def _grad_y(p, grid):
    from .operators import pad3d
    pp = pad3d(p)
    return (pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) / (2 * grid.dy)


def _grad_z(p, grid):
    from .operators import pad3d
    pp = pad3d(p)
    return (pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) / (2 * grid.dz)
