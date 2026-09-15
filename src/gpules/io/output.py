"""输出写出（风场 JSON / 对比数据）。

原项目把输出拼装散落在主脚本。这里提供两个函数：
  * ``write_output_json`` — 写出降采样 3D 风场 + 建筑 + 切变/高风区 + 诊断（推理结果）
  * ``write_comparison_data`` — 写出多时刻廓线/快照（可微训练结果，用于与 PALM 对比）
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from ..config import GPULESConfig
from ..core.state import FlowState


def write_output_json(state: FlowState, cfg: GPULESConfig, path: str,
                      buildings: Optional[list] = None, diagnostics: bool = False) -> dict:
    grid = state.grid
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    step_xy = max(1, nx // 100)
    nz_out = min(20, nz)
    z_out = np.linspace(dz, nz * dz, nz_out)

    u = state.u.detach().cpu().numpy()
    v = state.v.detach().cpu().numpy()
    w = state.w.detach().cpu().numpy()

    u_ds = np.zeros((nz_out, ny // step_xy, nx // step_xy), np.float32)
    v_ds = np.zeros((nz_out, ny // step_xy, nx // step_xy), np.float32)
    w_ds = np.zeros((nz_out, ny // step_xy, nx // step_xy), np.float32)
    for ko in range(nz_out):
        zt = z_out[ko]
        ks = max(0, min(nz - 2, int(zt / dz - 0.5)))
        fz = max(0.0, min(1.0, (zt - (ks * dz + dz / 2)) / dz))
        u_ds[ko] = ((1 - fz) * u[ks] + fz * u[ks + 1])[::step_xy, ::step_xy]
        v_ds[ko] = ((1 - fz) * v[ks] + fz * v[ks + 1])[::step_xy, ::step_xy]
        w_ds[ko] = ((1 - fz) * w[ks] + fz * w[ks + 1])[::step_xy, ::step_xy]

    x_coords = (np.arange(nx // step_xy) * dx * step_xy).tolist()
    y_coords = (np.arange(ny // step_xy) * dy * step_xy).tolist()
    max_speed = float(np.sqrt(u_ds ** 2 + v_ds ** 2 + w_ds ** 2).max())

    spd_ds = np.sqrt(u_ds ** 2 + v_ds ** 2 + w_ds ** 2)
    shear_cells, hw_cells = [], []
    for k in range(1, nz_out):
        dz_val = z_out[k] - z_out[k - 1]
        sh = np.sqrt(((u_ds[k] - u_ds[k - 1]) / dz_val) ** 2 + ((v_ds[k] - v_ds[k - 1]) / dz_val) ** 2)
        for j in range(0, ny // step_xy, 2):
            for i in range(0, nx // step_xy, 2):
                if sh[j, i] > 0.05:
                    shear_cells.append({"x": float(i * dx * step_xy), "y": float(j * dy * step_xy),
                                        "z": float(z_out[k] - dz), "w": 10.0, "d": 10.0, "h": 10.0,
                                        "shear": float(sh[j, i])})
                if spd_ds[k, j, i] > 12.0:
                    hw_cells.append({"x": float(i * dx * step_xy), "y": float(j * dy * step_xy),
                                     "z": float(z_out[k] - dz), "w": 10.0, "d": 10.0, "h": 10.0,
                                     "speed": float(spd_ds[k, j, i])})

    out = {
        "meta": {
            "description": "gpules v1.0 GPU LES 输出",
            "model": f"gpules {cfg.mode} (advection={cfg.advection}, pressure={cfg.pressure_solver}, "
                     f"sgs=Deardorff TKE, integrator={cfg.time_integrator})",
            "grid": f"{nx}x{ny}x{nz} -> {nx // step_xy}x{ny // step_xy}x{nz_out}",
            "max_speed": max_speed,
            "device": str(grid.device),
            "mode": cfg.mode,
        },
        "grid": {"nx": nx // step_xy, "ny": ny // step_xy, "nz": nz_out,
                 "dx": dx * step_xy, "dy": dy * step_xy},
        "z_levels": z_out.tolist(),
        "x_coords": x_coords,
        "y_coords": y_coords,
        "wind": {"u": u_ds.flatten().tolist(), "v": v_ds.flatten().tolist(), "w": w_ds.flatten().tolist()},
        "buildings": buildings or [],
        "shear_zones": shear_cells[:500],
        "highwind_zones": hw_cells[:500],
        "domain": {"x": float(nx * dx), "y": float(ny * dy), "z": float(nz * dz)},
    }
    if diagnostics:
        out["diagnostics"] = {"profiles": state.statistics()}

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def write_comparison_data(log: list, cfg: GPULESConfig, path: str, gpu: str = "cpu") -> None:
    """写出可微训练/推理的对比数据（多时刻廓线 + 快照）。``log`` 为每步诊断列表。"""
    z_coords = [k * cfg.dz + cfg.dz / 2 for k in range(cfg.nz)]
    x_coords = [xi * cfg.dx + cfg.dx / 2 for xi in range(cfg.nx)]
    data = {
        "meta": {"model": "gpules differentiable", "domain": f"{cfg.nx}x{cfg.ny}x{cfg.nz}",
                 "dx": cfg.dx, "dz": cfg.dz, "gpu": gpu, "n_steps": len(log)},
        "z": z_coords, "x": x_coords, "timeseries": log,
        "palm_ref": {
            "heights": [k * cfg.dz + cfg.dz / 2 for k in cfg.palm_ref_k],
            "u": cfg.palm_ref_u, "v": cfg.palm_ref_v,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
