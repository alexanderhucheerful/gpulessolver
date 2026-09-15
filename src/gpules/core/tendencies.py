"""高精度动量/温度倾向（v9 风格）。

把原 ``compute_tendencies`` 中耦合平流+扩散+科氏力+SGS+建筑阻力+Rayleigh 阻尼
的逻辑收为一个清晰函数。各物理项在此显式列出，便于单独测试与后续替换。
"""

from __future__ import annotations

from typing import Callable

import torch

from .grid import Grid
from .operators import laplacian


def compute_tendencies_hf(u, v, w, pt, qv, nu_t, ug, vg, cfg, advect_fn: Callable,
                          grid: Grid, bldg_mask3d, cd_wall, cd_bx, cd_by, rayleigh):
    adv_u = advect_fn(u, u, v, w)
    adv_v = advect_fn(v, u, v, w)
    adv_w = advect_fn(w, u, v, w)
    adv_pt = advect_fn(pt, u, v, w)

    diff_u = nu_t * laplacian(u, grid)
    diff_v = nu_t * laplacian(v, grid)
    diff_w = nu_t * laplacian(w, grid)
    diff_pt = (nu_t / cfg.pr_t) * laplacian(pt, grid)

    if cfg.use_coriolis:
        cor_u = cfg.f_cor * (v - vg)
        cor_v = -cfg.f_cor * (u - ug)
    else:
        cor_u = torch.zeros_like(u)
        cor_v = torch.zeros_like(v)

    if qv is not None:
        ptv = pt * (1.0 + 0.61 * qv)
        ptv_ref = ptv.mean().detach()
        buoy = cfg.g * (ptv - ptv_ref) / ptv_ref
    else:
        pt_ref = pt.mean().detach()
        buoy = cfg.g * (pt - pt_ref) / pt_ref

    spd_h = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
    drag_u = cd_wall * spd_h * u
    drag_v = cd_wall * spd_h * v
    bldg_drag_u = cd_bx * spd_h * u
    bldg_drag_v = cd_by * spd_h * v
    damp_u = rayleigh * u
    damp_v = rayleigh * v
    damp_w = rayleigh * w

    tu = adv_u + diff_u + cor_u - drag_u - bldg_drag_u - damp_u
    tv = adv_v + diff_v + cor_v - drag_v - bldg_drag_v - damp_v
    tw = adv_w + diff_w + buoy - damp_w
    tpt = adv_pt + diff_pt
    return tu, tv, tw, tpt
