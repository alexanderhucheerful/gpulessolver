"""可微线的核心物理步骤（来自 diff-v1，纯函数式、可插拔）。

把 diff-v1 中依赖大量模块全局变量的函数改写为接受 ``aux``（辅助场）、
``params``（可学习参数）、``cfg``（配置）、``grid`` 的纯函数，便于测试与复用。

覆盖：应变率、TKE 粘性、动量趋势（含可微符号保持限制器/拖曳/nudging/过速抑制）、
TKE 趋势、壁面函数、可微边界、压力投影、Heun 融合步。
"""

from __future__ import annotations

import torch

from ..config import GPULESConfig
from ..diff.stable import (
    get_soft_params,
    replace_slice,
    safe_abs,
    safe_nan_to_num,
    soft_clamp_min,
    soft_gate,
    soft_sign,
    soft_velocity_cap,
)
from .boundary import EXCESS_CD_V, NUDGE_U_BASE, NUDGE_V_BASE, TrainAux
from .grid import Grid
from .operators import divergence, laplacian


def compute_strain_rate(u, v, w, grid: Grid):
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    s11 = (torch.roll(u, -1, 2) - torch.roll(u, 1, 2)) / (2 * dx)
    s22 = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2 * dy)
    s33 = (torch.roll(w, -1, 0) - torch.roll(w, 1, 0)) / (2 * dz)
    s12 = 0.5 * ((torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2 * dy)
                 + (torch.roll(v, -1, 2) - torch.roll(v, 1, 2)) / (2 * dx))
    s13 = 0.5 * ((torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2 * dz)
                 + (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) / (2 * dx))
    s23 = 0.5 * ((torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2 * dz)
                 + (torch.roll(w, -1, 1) - torch.roll(w, 1, 1)) / (2 * dy))
    s_mag = torch.sqrt(2.0 * (s11 * s11 + s22 * s22 + s33 * s33
                              + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)) + 1e-20)
    return s11, s22, s33, s12, s13, s23, s_mag


def compute_tke_viscosity_diff(tke, params, aux: TrainAux, cfg: GPULESConfig) -> torch.Tensor:
    sp = get_soft_params(cfg)
    tke_safe = soft_clamp_min(tke, cfg.tke_min, sharpness=sp["softplus_sharpness"])
    delta = (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)
    nu_t = params.ce * delta * torch.sqrt(tke_safe) * aux.vd_damping
    return nu_t + cfg.nu_min + cfg.cnu * delta ** 2


def compute_tendency_diff(u, v, w, nu_t, tke, dt, params, aux: TrainAux, cfg: GPULESConfig, grid: Grid,
                          advection=None):
    """可微动量趋势（对应 diff-v1 compute_tendency_diff，1D 参考廓线 fallback）。

    ``advection`` 为可插拔平流算子（如 ``Central2``）；传入时复用其 ``advect``
    （含 Jameson 耗散，与 ``cfg.advection`` 一致），否则回退内联 2 阶中心平流。
    """
    if advection is not None:
        adv_u = advection.advect(u, u, v, w, grid)
        adv_v = advection.advect(v, u, v, w, grid)
        adv_w = advection.advect(w, u, v, w, grid)
    else:
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        adv_u = -(u * (torch.roll(u, -1, 2) - torch.roll(u, 1, 2)) / (2 * dx)
                  + v * (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2 * dy)
                  + w * (torch.roll(u, -1, 0) - torch.roll(u, 1, 0)) / (2 * dz))
        adv_v = -(u * (torch.roll(v, -1, 2) - torch.roll(v, 1, 2)) / (2 * dx)
                  + v * (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2 * dy)
                  + w * (torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2 * dz))
        adv_w = -(u * (torch.roll(w, -1, 2) - torch.roll(w, 1, 2)) / (2 * dx)
                  + v * (torch.roll(w, -1, 1) - torch.roll(w, 1, 1)) / (2 * dy)
                  + w * (torch.roll(w, -1, 0) - torch.roll(w, 1, 0)) / (2 * dz))

    diff_u = nu_t * laplacian(u, grid)
    diff_v = nu_t * laplacian(v, grid)
    diff_w = nu_t * laplacian(w, grid)

    cor_u = cfg.f_cor * v if cfg.use_coriolis else torch.zeros_like(u)
    cor_v = -cfg.f_cor * u if cfg.use_coriolis else torch.zeros_like(v)

    # PALM 式交错网格插值
    v_at_u = 0.25 * (v + torch.roll(v, -1, 1) + torch.roll(v, -1, 2) + torch.roll(v, (-1, -1), (1, 2)))
    w_at_u = 0.25 * (w + torch.roll(w, -1, 2) + torch.roll(w, -1, 0) + torch.roll(w, (-1, -1), (0, 2)))
    spd_3d_u = torch.sqrt(u * u + v_at_u * v_at_u + w_at_u * w_at_u + 1e-10)
    u_at_v = 0.25 * (u + torch.roll(u, -1, 1) + torch.roll(u, -1, 2) + torch.roll(u, (-1, -1), (1, 2)))
    w_at_v = 0.25 * (w + torch.roll(w, -1, 1) + torch.roll(w, -1, 0) + torch.roll(w, (-1, -1), (0, 1)))
    spd_3d_v = torch.sqrt(u_at_v * u_at_v + v * v + w_at_v * w_at_v + 1e-10)
    u_at_w = 0.25 * (u + torch.roll(u, -1, 2) + torch.roll(u, -1, 0) + torch.roll(u, (-1, -1), (0, 2)))
    v_at_w = 0.25 * (v + torch.roll(v, -1, 1) + torch.roll(v, -1, 0) + torch.roll(v, (-1, -1), (0, 1)))
    spd_3d_w = torch.sqrt(u_at_w * u_at_w + v_at_w * v_at_w + w * w + 1e-10)

    sp = get_soft_params(cfg)
    # 可微符号保持限制器（建筑拖曳）
    drag_u_raw = -params.cd_face_x * aux.face_x * spd_3d_u * u
    u_new = u + dt * drag_u_raw
    gate_u = soft_gate(-u_new * u, sharpness=sp["gate_sharpness"])
    drag_u = (1.0 - gate_u) * drag_u_raw + gate_u * (-u / dt)
    drag_v_raw = -params.cd_face_y * aux.face_y * spd_3d_v * v
    v_new = v + dt * drag_v_raw
    gate_v = soft_gate(-v_new * v, sharpness=sp["gate_sharpness"])
    drag_v = (1.0 - gate_v) * drag_v_raw + gate_v * (-v / dt)
    drag_w_raw = -0.5 * (params.cd_face_x * aux.face_x + params.cd_face_y * aux.face_y) * spd_3d_w * w * 0.3
    w_new = w + dt * drag_w_raw
    gate_w = soft_gate(-w_new * w, sharpness=sp["gate_sharpness"])
    drag_w = (1.0 - gate_w) * drag_w_raw + gate_w * (-w / dt)

    # 地面粗糙度拖曳
    spd = torch.sqrt(u * u + v * v + w * w + 1e-10)
    gdrag_u = -params.cd_ground * aux.ground_drag_base * spd * u
    gdrag_v = -params.cd_ground * aux.ground_drag_base * spd * v
    gdrag_w = -params.cd_ground * aux.ground_drag_base * spd * w * 0.1

    # 可微 nudging（强度可学习，1D 参考廓线）
    nudge_u_strength = params.nudge_u_peak * aux.nudge_u_lower_3d + params.nudge_u_upper * aux.nudge_u_upper_3d + NUDGE_U_BASE
    nudge_v_strength = aux.nudge_v_lower_3d + params.nudge_v_upper * aux.nudge_v_upper_3d + NUDGE_V_BASE
    nudge_u = -nudge_u_strength * (u - aux.u_ref)
    nudge_v = -nudge_v_strength * (v - aux.v_ref)

    # 可微过速抑制（受 use_excess_suppression 开关控制）
    if cfg.use_excess_suppression:
        abs_u = safe_abs(u)
        abs_v = safe_abs(v)
        u_excess = torch.nn.functional.softplus(abs_u - aux.u_excess_threshold) * soft_sign(u, eps=sp["soft_sign_eps"])
        v_excess = torch.nn.functional.softplus(abs_v - aux.u_excess_threshold) * soft_sign(v, eps=sp["soft_sign_eps"])
        excess_mask = aux.bldg_mask3d * aux.excess_height_mask
        excess_drag_u = -params.excess_cd * u_excess * abs_u * excess_mask
        excess_drag_v = -EXCESS_CD_V * v_excess * abs_v * excess_mask
    else:
        excess_drag_u = torch.zeros_like(u)
        excess_drag_v = torch.zeros_like(v)

    tu = adv_u + diff_u + cor_u + drag_u + gdrag_u + nudge_u + excess_drag_u
    tv = adv_v + diff_v + cor_v + drag_v + gdrag_v + nudge_v + excess_drag_v
    tw = adv_w + diff_w + drag_w + gdrag_w
    return tu, tv, tw


def compute_pt_tendency_diff(pt, u, v, w, nu_t, cfg: GPULESConfig, grid: Grid, advection=None):
    """可微位温趋势（平流 + SGS 扩散，与 inference 线物理一致）。"""
    if advection is not None:
        adv_pt = advection.advect(pt, u, v, w, grid)
    else:
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        adv_pt = -(u * (torch.roll(pt, -1, 2) - torch.roll(pt, 1, 2)) / (2 * dx)
                   + v * (torch.roll(pt, -1, 1) - torch.roll(pt, 1, 1)) / (2 * dy)
                   + w * (torch.roll(pt, -1, 0) - torch.roll(pt, 1, 0)) / (2 * dz))
    diff_pt = (nu_t / cfg.pr_t) * laplacian(pt, grid)
    return adv_pt + diff_pt


def compute_tke_tendency_diff(u, v, w, tke, nu_t, params, aux: TrainAux, cfg: GPULESConfig, grid: Grid):
    sp = get_soft_params(cfg)
    s11, s22, s33, s12, s13, s23, _ = compute_strain_rate(u, v, w, grid)
    p_shear = 2.0 * nu_t * (s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23))
    tke_safe = soft_clamp_min(tke, cfg.tke_min, sharpness=sp["softplus_sharpness"])
    eps = params.ce * torch.sqrt(tke_safe) ** 3 / (cfg.dx * cfg.dy * cfg.dz) ** (1.0 / 3.0)
    diff_tke = (nu_t / cfg.sigma_e) * laplacian(tke, grid)
    adv_tke = -(u * (torch.roll(tke, -1, 2) - torch.roll(tke, 1, 2)) / (2 * grid.dx)
                + v * (torch.roll(tke, -1, 1) - torch.roll(tke, 1, 1)) / (2 * grid.dy)
                + w * (torch.roll(tke, -1, 0) - torch.roll(tke, 1, 0)) / (2 * grid.dz))
    spd_3d = torch.sqrt(u * u + v * v + w * w + 1e-10)
    tke_sink = -2.0 * aux.drag_total_base * spd_3d * tke_safe
    return adv_tke + p_shear - eps + diff_tke + tke_sink


def apply_wall_function_diff(u, v, w, dt, params, aux: TrainAux):
    dt_factor = min(dt / 0.1, 1.0)
    nudge_u = params.wall_nudge * dt_factor
    nudge_v = params.wall_nudge_v * dt_factor
    u_out, v_out = u, v
    for k in range(len(aux.wall_layer_masks)):
        m = aux.wall_layer_masks[k]
        u_out = u_out * (1.0 - m) + (u * (1.0 - nudge_u) + aux.u_ref_t[k] * nudge_u) * m
        v_out = v_out * (1.0 - m) + (v * (1.0 - nudge_v) + aux.v_ref_t[k] * nudge_v) * m
    w_out = w * (1.0 - aux.wall_m0)
    w_out = w_out * (1.0 - aux.wall_m1) + (w * 0.5) * aux.wall_m1
    return u_out, v_out, w_out


def apply_bcs_diff(u, v, w, dt, cfg: GPULESConfig, aux: TrainAux, params, device: torch.device):
    nz, ny, nx = aux.bldg_mask3d.shape
    u = u * aux.bldg_mask3d
    v = v * aux.bldg_mask3d
    w = w * aux.bldg_mask3d
    u, v, w = apply_wall_function_diff(u, v, w, dt, params, aux)

    # 1D 入流（西/南），辐射出流（东/北）—— 与 diff-v1 fallback 一致
    west_u = aux.u_ref_t.view(nz, 1).expand(nz, ny)[:, :ny]
    west_v = aux.v_ref_t.view(nz, 1).expand(nz, ny)[:, :ny]
    south_u = aux.u_ref_t.view(nz, 1).expand(nz, nx)[:, :nx]
    south_v = aux.v_ref_t.view(nz, 1).expand(nz, nx)[:, :nx]
    u = replace_slice(u, 2, 0, west_u)
    v = replace_slice(v, 2, 0, west_v)
    w = replace_slice(w, 2, 0, torch.zeros(nz, ny, device=device))
    u = replace_slice(u, 1, 0, south_u)
    v = replace_slice(v, 1, 0, south_v)
    w = replace_slice(w, 1, 0, torch.zeros(nz, nx, device=device))

    c_factor = 0.8 * dt / cfg.dx
    east_u = u[:, :, nx - 2] - c_factor * (u[:, :, nx - 2] - u[:, :, nx - 3])
    east_v = v[:, :, nx - 2] - c_factor * (v[:, :, nx - 2] - v[:, :, nx - 3])
    east_w = w[:, :, nx - 2] - c_factor * (w[:, :, nx - 2] - w[:, :, nx - 3])
    u = replace_slice(u, 2, nx - 1, east_u)
    v = replace_slice(v, 2, nx - 1, east_v)
    w = replace_slice(w, 2, nx - 1, east_w)
    c_factor_y = 0.8 * dt / cfg.dy
    north_u = u[:, ny - 2, :] - c_factor_y * (u[:, ny - 2, :] - u[:, ny - 3, :])
    north_v = v[:, ny - 2, :] - c_factor_y * (v[:, ny - 2, :] - v[:, ny - 3, :])
    north_w = w[:, ny - 2, :] - c_factor_y * (w[:, ny - 2, :] - w[:, ny - 3, :])
    u = replace_slice(u, 1, ny - 1, north_u)
    v = replace_slice(v, 1, ny - 1, north_v)
    w = replace_slice(w, 1, ny - 1, north_w)

    # 顶部 free-slip + no penetration
    u = replace_slice(u, 0, nz - 1, u[nz - 2])
    v = replace_slice(v, 0, nz - 1, v[nz - 2])
    w = replace_slice(w, 0, nz - 1, torch.zeros(ny, nx, device=device))

    u = safe_nan_to_num(u)
    v = safe_nan_to_num(v)
    w = safe_nan_to_num(w)
    sp = get_soft_params(cfg)
    u, v, w = soft_velocity_cap(u, v, w, v_max=cfg.vmax, k=sp["vel_cap_k"])
    return u, v, w


def project_velocity_diff(u, v, w, dt, pressure_solver, cfg: GPULESConfig, grid: Grid, bldg_mask3d):
    # 中心差分散度，与下方中心差压力梯度（dp_dx 等）自洽，
    # 避免前向差分散度 + 中心差梯度引入的棋盘格/网格尺度误差。
    # 注：PoissonFFTSolver 假设三方向周期，物理域为入流/出流/顶盖非周期，
    # 故 train 线用谱投影属 surrogate 近似（见 README）。
    div = divergence(u, v, w, grid)
    div = safe_nan_to_num(div)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    div = replace_slice(div, 2, nx - 1, div[:, :, nx - 2])
    div = replace_slice(div, 1, ny - 1, div[:, ny - 2, :])
    div = replace_slice(div, 0, nz - 1, div[nz - 2, :, :])
    p = pressure_solver.solve(div / dt, torch.zeros_like(div), grid)
    p = safe_nan_to_num(p)
    dp_dx = (torch.roll(p, -1, 2) - torch.roll(p, 1, 2)) / (2 * grid.dx)
    dp_dy = (torch.roll(p, -1, 1) - torch.roll(p, 1, 1)) / (2 * grid.dy)
    dp_dz = (torch.roll(p, -1, 0) - torch.roll(p, 1, 0)) / (2 * grid.dz)
    u2 = (u - dt * dp_dx) * bldg_mask3d
    v2 = (v - dt * dp_dy) * bldg_mask3d
    w2 = (w - dt * dp_dz) * bldg_mask3d
    return u2, v2, w2


def fused_step_diff(u, v, w, pt, tke, dt, params, aux: TrainAux, cfg: GPULESConfig, grid: Grid,
                    pressure_solver, advection=None) -> tuple:
    """可微 Heun 步（diff-v1 风格），1D 参考廓线 fallback。

    ``advection`` 为可插拔平流算子；传入时平流项复用它，否则回退内联中心差分。
    返回 (u, v, w, pt, tke)。
    """
    sp = get_soft_params(cfg)
    nu_t = compute_tke_viscosity_diff(tke, params, aux, cfg)
    k1_u, k1_v, k1_w = compute_tendency_diff(u, v, w, nu_t, tke, dt, params, aux, cfg, grid, advection)
    k1_pt = compute_pt_tendency_diff(pt, u, v, w, nu_t, cfg, grid, advection)
    k1_tke = compute_tke_tendency_diff(u, v, w, tke, nu_t, params, aux, cfg, grid)
    k1_u = k1_u - aux.sponge * (u - aux.u_ref)
    k1_v = k1_v - aux.sponge * (v - aux.v_ref)
    k1_w = k1_w - aux.sponge * w
    k1_tke = k1_tke - aux.sponge * (tke - cfg.tke_min)

    u1 = u + dt * k1_u
    v1 = v + dt * k1_v
    w1 = w + dt * k1_w
    pt1 = pt + dt * k1_pt
    tke1 = soft_clamp_min(tke + dt * k1_tke, cfg.tke_min, sharpness=sp["softplus_sharpness"])
    u1, v1, w1 = apply_bcs_diff(u1, v1, w1, dt, cfg, aux, params, grid.device)

    nu_t2 = compute_tke_viscosity_diff(tke1, params, aux, cfg)
    k2_u, k2_v, k2_w = compute_tendency_diff(u1, v1, w1, nu_t2, tke1, dt, params, aux, cfg, grid, advection)
    k2_pt = compute_pt_tendency_diff(pt1, u1, v1, w1, nu_t2, cfg, grid, advection)
    k2_tke = compute_tke_tendency_diff(u1, v1, w1, tke1, nu_t2, params, aux, cfg, grid)
    k2_u = k2_u - aux.sponge * (u1 - aux.u_ref)
    k2_v = k2_v - aux.sponge * (v1 - aux.v_ref)
    k2_w = k2_w - aux.sponge * w1
    k2_tke = k2_tke - aux.sponge * (tke1 - cfg.tke_min)

    u2 = u + dt * 0.5 * (k1_u + k2_u)
    v2 = v + dt * 0.5 * (k1_v + k2_v)
    w2 = w + dt * 0.5 * (k1_w + k2_w)
    pt2 = pt + dt * 0.5 * (k1_pt + k2_pt)
    tke2 = soft_clamp_min(tke + dt * 0.5 * (k1_tke + k2_tke), cfg.tke_min, sharpness=sp["softplus_sharpness"])
    u2, v2, w2 = apply_bcs_diff(u2, v2, w2, dt, cfg, aux, params, grid.device)

    if cfg.double_projection:
        u2, v2, w2 = project_velocity_diff(u2, v2, w2, dt, pressure_solver, cfg, grid, aux.bldg_mask3d)
        u2, v2, w2 = apply_bcs_diff(u2, v2, w2, dt, cfg, aux, params, grid.device)
    else:
        u2, v2, w2 = project_velocity_diff(u2, v2, w2, dt, pressure_solver, cfg, grid, aux.bldg_mask3d)
    return u2, v2, w2, pt2, tke2
