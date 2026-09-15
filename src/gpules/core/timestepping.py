"""时间积分器（可插拔策略）。

  * ``SSPRK3`` — 3 阶强稳定 Runge-Kutta（inference 默认，来自 v9 ``run_rk3_step``）
  * ``HeunStep`` — 2 阶 Heun（train 默认，来自 diff-v1 ``fused_step_diff``，全程可微）

两者都返回新的 (u,v,w,pt,tke)，由求解器统一处理压力投影与边界。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from ..config import GPULESConfig
from .boundary import BuildingFields
from .grid import Grid
from .operators import divergence, gradient
from .pressure import PressureSolver
from .sgs import DeardorffTKE
from .state import FlowState
from .tendencies import compute_tendencies_hf


class TimeIntegrator(ABC):
    @abstractmethod
    def step(self, state: FlowState, dt: float, cfg: GPULESConfig, grid: Grid,
             advect_fn: Callable, pressure: PressureSolver, sgs: DeardorffTKE,
             bf: BuildingFields, bc: dict, ug: float, vg: float) -> FlowState:
        ...


class SSPRK3(TimeIntegrator):
    """3 阶 SSP-RK3（v9 风格），含每步 TKE 更新 + 压力投影 + 速度上限。"""

    def __init__(self) -> None:
        self._forcing: tuple | None = None

    def prepare(self, cfg: GPULESConfig, grid: Grid, bf: BuildingFields) -> None:
        """预计算静态强迫系数（每步复用，避免重复 GPU 标量写）。"""
        self._forcing = self._compute_forcing_coeffs(cfg, grid, bf)

    def step(self, state, dt, cfg, grid, advect_fn, pressure, sgs, bf, bc, ug, vg):
        u, v, w, pt, qv, tke, p_prev = (state.u, state.v, state.w, state.pt,
                                         state.qv, state.tke, state.p)
        bldg_mask3d = bf.bldg_mask3d
        if self._forcing is None:
            self.prepare(cfg, grid, bf)
        cd_wall, cd_bx, cd_by, rayleigh = self._forcing

        tke, nu_t = sgs.step_hf(tke, u, v, w, pt, qv, dt, bldg_mask3d, grid, advect_fn)

        tu, tv, tw, tpt = compute_tendencies_hf(
            u, v, w, pt, qv, nu_t, ug, vg, cfg, advect_fn, grid, bldg_mask3d,
            cd_wall, cd_bx, cd_by, rayleigh)
        u1 = (u + dt * tu) * bldg_mask3d
        v1 = (v + dt * tv) * bldg_mask3d
        w1 = (w + dt * tw) * bldg_mask3d
        pt1 = pt + dt * tpt

        tu2, tv2, tw2, tpt2 = compute_tendencies_hf(
            u1, v1, w1, pt1, qv, nu_t, ug, vg, cfg, advect_fn, grid, bldg_mask3d,
            cd_wall, cd_bx, cd_by, rayleigh)
        u2 = ((3 * u + u1 + dt * tu2) / 4) * bldg_mask3d
        v2 = ((3 * v + v1 + dt * tv2) / 4) * bldg_mask3d
        w2 = ((3 * w + w1 + dt * tw2) / 4) * bldg_mask3d
        pt2 = (3 * pt + pt1 + dt * tpt2) / 4

        tu3, tv3, tw3, tpt3 = compute_tendencies_hf(
            u2, v2, w2, pt2, qv, nu_t, ug, vg, cfg, advect_fn, grid, bldg_mask3d,
            cd_wall, cd_bx, cd_by, rayleigh)
        u3 = ((u + 2 * u2 + 2 * dt * tu3) / 3) * bldg_mask3d
        v3 = ((v + 2 * v2 + 2 * dt * tv3) / 3) * bldg_mask3d
        w3 = ((w + 2 * w2 + 2 * dt * tw3) / 3) * bldg_mask3d
        pt3 = (pt + 2 * pt2 + 2 * dt * tpt3) / 3

        div = divergence(u3, v3, w3, grid)
        p = pressure.solve(div / dt, p_prev, grid)
        dpdx, dpdy, dpdz = gradient(p, grid)
        dpdz[-1] = 0
        dpdz[0] = 0

        u_new = (u3 - dt * dpdx) * bldg_mask3d
        v_new = (v3 - dt * dpdy) * bldg_mask3d
        w_new = (w3 - dt * dpdz) * bldg_mask3d

        spd = torch.sqrt(u_new ** 2 + v_new ** 2 + w_new ** 2 + 1e-6)
        cap = torch.clamp(cfg.vmax / spd, max=1.0)
        u_new = u_new * cap
        v_new = v_new * cap
        w_new = w_new * cap
        w_new = torch.clamp(w_new, -cfg.wmax, cfg.wmax)

        # 边界应用（高精度复制式）
        from .boundary import apply_bc_hf
        u_new, v_new, w_new, pt_new = apply_bc_hf(
            u_new, v_new, w_new, pt3, bc, cfg, bldg_mask3d, grid)
        return FlowState(grid, u_new, v_new, w_new, pt_new, qv, tke, p)

    @staticmethod
    def _compute_forcing_coeffs(cfg, grid, bf):
        nz, dz = grid.nz, grid.dz
        z_vals = (torch.arange(nz, device=grid.device).float() * dz + dz / 2).view(nz, 1, 1)
        if cfg.use_wall_drag:
            z_k = z_vals.squeeze()
            log_term = torch.log(torch.clamp(z_k / cfg.z0, min=1e-6))
            cd_vals = (cfg.kappa ** 2) / (log_term ** 2)
            mask = (z_k <= cfg.log_law_top) & (z_k > cfg.z0)
            cd_wall = (cd_vals * mask.float()).view(nz, 1, 1)
        else:
            cd_wall = torch.zeros(nz, 1, 1, device=grid.device)
        cd_bldg = 0.55
        cd_bx = cd_bldg * bf.face_x if cfg.use_building_drag else torch.zeros_like(bf.face_x)
        cd_by = cd_bldg * bf.face_y if cfg.use_building_drag else torch.zeros_like(bf.face_y)
        if cfg.use_rayleigh:
            z_frac = (z_vals.squeeze() / (nz * dz))
            mask = torch.clamp((z_frac - cfg.rayleigh_start) / (1 - cfg.rayleigh_start), min=0) ** 2
            rayleigh = (cfg.rayleigh_coef * mask).view(nz, 1, 1)
        else:
            rayleigh = torch.zeros(nz, 1, 1, device=grid.device)
        return cd_wall, cd_bx, cd_by, rayleigh


def build_integrator(cfg) -> TimeIntegrator:
    if cfg.time_integrator == "ssprk3":
        return SSPRK3()
    # heun 由 solver 在 train 模式下调用 diff_helpers.fused_step_diff，这里不单独建类
    raise ValueError("Heun 仅在 train 模式下由 diff_helpers 提供，请用 mode='train'")
