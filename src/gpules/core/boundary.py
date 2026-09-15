"""边界条件与建筑强迫（HF 线 + 可微线）。

包含：
  * :class:`BuildingFields` — 建筑掩模/面密度等预计算场
  * ``build_building_fields`` — 由 2D 建筑高度场构造（支持 sigmoid 软壁面）
  * ``apply_bc_hf``  — 高精度边界（v9 复制式 BC + 松弛 + 硬速度上限）
  * ``apply_bcs_diff`` — 可微边界（replace_slice 非原地 + soft 速度上限 + 壁面函数）
  * :class:`TrainAux` — 可微训练所需的固定辅助场（nudging 廓线/参考廓线/sponge/vd 阻尼等）

Nudging 的高度/宽度几何为物理设计常量（与 diff-v1 一致），集中在此便于审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..config import GPULESConfig
from .grid import Grid

# ── Nudging 几何（物理设计常量，来自 diff-v1） ──────────────────────
NUDGE_U_PEAK_Z = 47.5
NUDGE_U_SIGMA = 60.0
NUDGE_U_BASE = 0.005
NUDGE_U_UPPER_Z = 140.0
NUDGE_U_UPPER_SIGMA = 70.0
NUDGE_V_PEAK = 0.010
NUDGE_V_PEAK_Z = 47.5
NUDGE_V_SIGMA = 60.0
NUDGE_V_BASE = 0.003
NUDGE_V_UPPER_Z = 97.5
NUDGE_V_UPPER_SIGMA = 45.0
EXCESS_RATIO = 1.3
EXCESS_HEIGHT = 60.0
EXCESS_TAPER = 20.0
EXCESS_CD_V = 0.15


@dataclass
class BuildingFields:
    bldg_mask3d: torch.Tensor
    solid: torch.Tensor
    face_x: torch.Tensor
    face_y: torch.Tensor
    bldg_h: torch.Tensor
    soft: bool = False


def build_building_fields(grid: Grid, bldg_2d: torch.Tensor, cfg: GPULESConfig,
                           soft: Optional[bool] = None) -> BuildingFields:
    """由 2D 建筑高度场构造 3D 掩模与面密度。

    Args:
        bldg_2d: (ny, nx) 建筑高度 (m)，0 表示无建筑。
        soft: 是否使用 sigmoid 软壁面；None 时取 ``cfg.use_soft_mask``。
    """
    dev = grid.device
    soft = cfg.use_soft_mask if soft is None else soft
    bldg_h = bldg_2d.to(device=dev, dtype=torch.float32)
    z_vals = (torch.arange(grid.nz, device=dev).float() * grid.dz + grid.dz / 2).view(grid.nz, 1, 1)
    if soft:
        try:
            from scipy.ndimage import gaussian_filter
            bldg_h_smooth = torch.tensor(gaussian_filter(bldg_2d.cpu().numpy(), sigma=1.5),
                                         dtype=torch.float32, device=dev)
            bldg_mask3d = torch.sigmoid((z_vals - bldg_h_smooth.unsqueeze(0)) / max(cfg.smooth_width, 1e-3))
        except Exception:
            bldg_mask3d = (bldg_h.unsqueeze(0) < z_vals).float()
    else:
        bldg_mask3d = (bldg_h.unsqueeze(0) < z_vals).float()
    solid = 1.0 - bldg_mask3d

    face_x_raw = torch.abs(torch.roll(solid, -1, 2) - solid)
    face_y_raw = torch.abs(torch.roll(solid, -1, 1) - solid)
    k = cfg.canopy_smooth
    face_x = F.avg_pool3d(face_x_raw.unsqueeze(0).unsqueeze(0), kernel_size=(k, k, k),
                          stride=1, padding=k // 2).squeeze() * bldg_mask3d
    face_y = F.avg_pool3d(face_y_raw.unsqueeze(0).unsqueeze(0), kernel_size=(k, k, k),
                          stride=1, padding=k // 2).squeeze() * bldg_mask3d
    return BuildingFields(bldg_mask3d, solid, face_x, face_y, bldg_h, soft=soft)


def generate_synthetic_buildings(grid: Grid, cfg: GPULESConfig) -> BuildingFields:
    """生成合成城市街区（用于无建筑数据时的演示/测试）。"""
    ny, nx = grid.ny, grid.nx
    bldg_2d = torch.zeros(ny, nx, dtype=torch.float32)
    blocks = [(50, 90, 100, 120, 50), (150, 180, 200, 230, 30),
              (250, 280, 50, 70, 40), (320, 360, 150, 190, 60)]
    for x0, x1, y0, y1, h in blocks:
        if x1 <= nx and y1 <= ny:
            bldg_2d[y0:y1, x0:x1] = h
    return build_building_fields(grid, bldg_2d, cfg, soft=False)


# ── 高精度边界（v9 复制式） ──────────────────────────────────────────
def apply_bc_hf(u, v, w, pt, bc: dict, cfg: GPULESConfig, bldg_mask3d: torch.Tensor, grid: Grid):
    u = u.clone()
    v = v.clone()
    w = w.clone()
    pt = pt.clone()
    nx, ny = grid.nx, grid.ny
    u[:, :, 0] = bc.get("left_u", u[:, :, 0])
    v[:, :, 0] = bc.get("left_v", v[:, :, 0])
    pt[:, :, 0] = bc.get("left_pt", pt[:, :, 0])
    u[:, 0, :] = bc.get("south_u", u[:, 0, :])
    v[:, 0, :] = bc.get("south_v", v[:, 0, :])
    pt[:, 0, :] = bc.get("south_pt", pt[:, 0, :])
    w[:, :, 0] = 0
    w[:, 0, :] = 0
    u[:, :, -1] = u[:, :, -2]
    v[:, :, -1] = v[:, :, -2]
    w[:, :, -1] = w[:, :, -2]
    u[:, -1, :] = u[:, -2, :]
    v[:, -1, :] = v[:, -2, :]
    w[:, -1, :] = w[:, -2, :]
    for i in range(cfg.relax_width):
        a = cfg.relax_strength * (1 - i / cfg.relax_width)
        ix = nx - 1 - i
        iy = ny - 1 - i
        if "right_u" in bc:
            u[:, :, ix] = (1 - a) * u[:, :, ix] + a * bc["right_u"]
            v[:, :, ix] = (1 - a) * v[:, :, ix] + a * bc["right_v"]
            pt[:, :, ix] = (1 - a) * pt[:, :, ix] + a * bc["right_pt"]
        if "north_u" in bc:
            u[:, iy, :] = (1 - a) * u[:, iy, :] + a * bc["north_u"]
            v[:, iy, :] = (1 - a) * v[:, iy, :] + a * bc["north_v"]
            pt[:, iy, :] = (1 - a) * pt[:, iy, :] + a * bc["north_pt"] if "north_pt" in bc else pt[:, iy, :]
    if cfg.use_rayleigh:
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = 0
        pt[-1, :, :] = pt[-2, :, :]
    w[0, :, :] = 0
    u = u * bldg_mask3d
    v = v * bldg_mask3d
    w = w * bldg_mask3d
    pt = pt  # 温度不乘建筑掩模
    return u, v, w, pt


# ── 可微训练辅助场 ───────────────────────────────────────────────────
@dataclass
class TrainAux:
    ground_drag_base: torch.Tensor
    nudge_u_lower_3d: torch.Tensor
    nudge_u_upper_3d: torch.Tensor
    nudge_v_lower_3d: torch.Tensor
    nudge_v_upper_3d: torch.Tensor
    u_ref: torch.Tensor
    v_ref: torch.Tensor
    u_ref_t: torch.Tensor
    v_ref_t: torch.Tensor
    u_excess_threshold: torch.Tensor
    excess_height_mask: torch.Tensor
    sponge: torch.Tensor
    vd_damping: torch.Tensor
    drag_total_base: torch.Tensor
    bldg_mask3d: torch.Tensor
    face_x: torch.Tensor
    face_y: torch.Tensor
    wall_layer_masks: list = field(default_factory=list)
    wall_m0: torch.Tensor = None
    wall_m1: torch.Tensor = None
    max_bldg_h: float = 0.0


def build_train_aux(grid: Grid, bf: BuildingFields, cfg: GPULESConfig,
                    u_ref_1d: torch.Tensor, v_ref_1d: torch.Tensor) -> TrainAux:
    """构造可微训练所需的固定辅助场（1D 参考廓线 fallback，与 diff-v1 一致）。"""
    dev = grid.device
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    bldg_mask3d = bf.bldg_mask3d
    solid = bf.solid

    # 地面拖曳基础场
    ground_drag_base = torch.zeros(nz, ny, nx, device=dev)
    for k in range(cfg.ground_drag_layers):
        weight = (1.0 - k / cfg.ground_drag_layers) ** 2
        ground_drag_base[k] = weight
    ground_drag_base = ground_drag_base * bldg_mask3d

    # 方向性迎风面积密度（不含 CD）
    face_x_raw = torch.abs(torch.roll(solid, -1, 2) - solid)
    face_y_raw = torch.abs(torch.roll(solid, -1, 1) - solid)
    k = cfg.canopy_smooth
    face_x = F.avg_pool3d(face_x_raw.unsqueeze(0).unsqueeze(0), kernel_size=(k, k, k),
                          stride=1, padding=k // 2).squeeze()
    face_y = F.avg_pool3d(face_y_raw.unsqueeze(0).unsqueeze(0), kernel_size=(k, k, k),
                          stride=1, padding=k // 2).squeeze()
    max_bldg_h = float(bf.bldg_h.max())
    drag_height_mask = (torch.arange(nz, device=dev).float() * grid.dz + grid.dz / 2 <= max_bldg_h + 10.0).float().view(nz, 1, 1)
    face_x = face_x * bldg_mask3d * drag_height_mask
    face_y = face_y * bldg_mask3d * drag_height_mask
    drag_total_base = 0.5 * (face_x + face_y) + ground_drag_base

    # Nudging 高斯廓线
    z_1d = torch.arange(nz, device=dev).float() * grid.dz + grid.dz / 2
    nudge_u_lower = torch.exp(-((z_1d - NUDGE_U_PEAK_Z) / NUDGE_U_SIGMA) ** 2).view(nz, 1, 1).expand(nz, ny, nx)
    nudge_u_upper = torch.exp(-((z_1d - NUDGE_U_UPPER_Z) / NUDGE_U_UPPER_SIGMA) ** 2).view(nz, 1, 1).expand(nz, ny, nx)
    nudge_v_lower = NUDGE_V_PEAK * torch.exp(-((z_1d - NUDGE_V_PEAK_Z) / NUDGE_V_SIGMA) ** 2).view(nz, 1, 1).expand(nz, ny, nx)
    nudge_v_upper = torch.exp(-((z_1d - NUDGE_V_UPPER_Z) / NUDGE_V_UPPER_SIGMA) ** 2).view(nz, 1, 1).expand(nz, ny, nx)

    u_ref = u_ref_1d.view(nz, 1, 1).expand(nz, ny, nx)
    v_ref = v_ref_1d.view(nz, 1, 1).expand(nz, ny, nx)

    # 过速抑制阈值与高度掩模
    u_excess_threshold = (EXCESS_RATIO * torch.sqrt(u_ref_1d ** 2 + v_ref_1d ** 2 + 1e-10)).view(nz, 1, 1).expand(nz, ny, nx)
    excess_height_mask = torch.zeros(nz, 1, 1, device=dev)
    for k in range(nz):
        z = k * grid.dz + grid.dz / 2
        if z < EXCESS_HEIGHT:
            excess_height_mask[k, 0, 0] = 1.0
        elif z < EXCESS_HEIGHT + EXCESS_TAPER:
            excess_height_mask[k, 0, 0] = 1.0 - (z - EXCESS_HEIGHT) / EXCESS_TAPER
    excess_height_mask = excess_height_mask.expand(nz, ny, nx)

    # Sponge 层
    sponge = torch.zeros(nz, ny, nx, device=dev)
    for i in range(cfg.sponge_width):
        ws = cfg.sponge_strength * ((cfg.sponge_width - i) / cfg.sponge_width) ** 2
        sponge[:, i, :] += ws
        sponge[:, -(i + 1), :] += ws
        sponge[:, :, i] += ws
        sponge[:, :, -(i + 1)] += ws
    idx_z = torch.arange(cfg.sponge_top, device=dev)
    frac_z = (cfg.sponge_top - idx_z) / cfg.sponge_top
    sponge[nz - cfg.sponge_top:, :, :] += cfg.sponge_strength * 0.5 * frac_z.view(-1, 1, 1) ** 2

    # van Driest 阻尼
    vd_damping = 1.0 - torch.exp(-z_1d / (25.0 * (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)))
    vd_damping = vd_damping.view(nz, 1, 1)
    vd_damping = torch.clamp(vd_damping, min=0.05)

    # 壁面层掩模
    wall_layer_masks = []
    for _k in range(cfg.ground_drag_layers):
        _m = torch.zeros(nz, 1, 1, device=dev)
        _m[_k] = 1.0
        wall_layer_masks.append(_m)
    wall_m0 = torch.zeros(nz, 1, 1, device=dev)
    wall_m0[0] = 1.0
    wall_m1 = torch.zeros(nz, 1, 1, device=dev)
    wall_m1[1] = 1.0

    return TrainAux(ground_drag_base=ground_drag_base, nudge_u_lower_3d=nudge_u_lower,
                    nudge_u_upper_3d=nudge_u_upper, nudge_v_lower_3d=nudge_v_lower,
                    nudge_v_upper_3d=nudge_v_upper, u_ref=u_ref, v_ref=v_ref, u_ref_t=u_ref_1d,
                    v_ref_t=v_ref_1d, u_excess_threshold=u_excess_threshold,
                    excess_height_mask=excess_height_mask, sponge=sponge, vd_damping=vd_damping,
                    drag_total_base=drag_total_base, bldg_mask3d=bldg_mask3d, face_x=face_x,
                    face_y=face_y, wall_layer_masks=wall_layer_masks, wall_m0=wall_m0,
                    wall_m1=wall_m1, max_bldg_h=max_bldg_h)
