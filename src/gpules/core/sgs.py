"""Deardorff TKE 亚格子（SGS）模型。

两版都用 Deardorff 1.5 阶 TKE 闭合，但实现风格不同：

  * 高精度线（v9）：含虚位温浮力、硬 ``clamp(tke, min)``
  * 可微线（diff-v1）：用 ``soft_clamp_min`` 替代硬 clamp，并引入 van Driest 阻尼

这里统一为一个 :class:`DeardorffTKE`，按 ``mode`` 选择 ``*_hf``（高精度）或
``*_diff``（可微）实现。``nu_t`` 计算遵循两版公式
``nu_t = Ck * delta * sqrt(2) * sqrt(tke)``（delta 为网格特征长度）。
"""

from __future__ import annotations

import math

import torch

from ..config import GPULESConfig
from .operators import dx_c, dy_c, dz_c, laplacian


class DeardorffTKE:
    def __init__(self, cfg: GPULESConfig):
        self.cfg = cfg
        self.delta = (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)

    # ── 粘性系数 ───────────────────────────────────────────────────
    def nu_t_hf(self, tke: torch.Tensor) -> torch.Tensor:
        return (self.cfg.ck * self.delta * math.sqrt(2.0) * torch.sqrt(
            torch.clamp(tke, min=self.cfg.tke_min))
            + self.cfg.cnu * self.delta ** 2)

    # 注：可微线（train 模式）的粘性系数由 ``diff_helpers.compute_tke_viscosity_diff``
    # 统一实现（含 van Driest 阻尼），此处不重复定义以免两版漂移。

    # ── 高精度 TKE 输运（v9 风格） ────────────────────────────────
    def step_hf(self, tke, u, v, w, pt, qv, dt, bldg_mask3d, grid, advect_fn) -> tuple[torch.Tensor, torch.Tensor]:
        dudx, dudy, dudz = dx_c(u, grid.dx), dy_c(u, grid.dy), dz_c(u, grid.dz)
        dvdx, dvdy, dvdz = dx_c(v, grid.dx), dy_c(v, grid.dy), dz_c(v, grid.dz)
        dwdx, dwdy, dwdz = dx_c(w, grid.dx), dy_c(w, grid.dy), dz_c(w, grid.dz)
        shear = (2 * (dudx ** 2 + dvdy ** 2 + dwdz ** 2)
                 + (dudy + dvdx) ** 2 + (dudz + dwdx) ** 2 + (dvdz + dwdy) ** 2)
        nu_t = self.nu_t_hf(tke)
        p_shear = nu_t * shear
        if qv is not None:
            ptv = pt * (1.0 + 0.61 * qv)
            ptv_mean = ptv.mean().detach()
        else:
            ptv = pt
            ptv_mean = pt.mean().detach()
        dptv_dz = dz_c(ptv, grid.dz)
        p_buoy = -(nu_t / self.cfg.pr_t) * (self.cfg.g / ptv_mean) * dptv_dz
        eps = self.cfg.ce * torch.clamp(tke, min=self.cfg.tke_min) ** 1.5 / (self.delta * math.sqrt(2.0))
        eps = torch.clamp(eps, max=50.0)
        adv = advect_fn(tke, u, v, w)
        diff = nu_t * laplacian(tke, grid) / 1.5
        dtke = adv + diff + p_shear + p_buoy - eps
        tke_new = torch.clamp(tke + dt * dtke, min=self.cfg.tke_min) * bldg_mask3d
        nu_t = self.nu_t_hf(tke_new)
        return tke_new, nu_t

    # 注：可微线（train 模式）的 TKE 输运由 ``diff_helpers.compute_tke_tendency_diff``
    # 统一实现，此处不再保留 ``step_diff`` 以免两版定义漂移。
