"""输入数据加载（合成 / PIDS / AI 预报）。

原项目把数据加载、建筑构造、边界插值散落在主脚本里。这里统一为一组合成/加载函数，
返回 :class:`InitData`（numpy 数组 + 边界字典 + 地转风），与设备解耦——
求解器负责把 numpy 升到 GPU/CPU。这也为 359MB 大数据留出了 mmap/分块接入点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import GPULESConfig
from ..core.grid import Grid
from ..exceptions import DataLoadError, SchemaError


@dataclass
class InitData:
    """初始化数据（全部 numpy，未上设备）。"""
    u: np.ndarray          # (nz, ny, nx)
    v: np.ndarray
    w: np.ndarray
    pt: np.ndarray
    qv: Optional[np.ndarray]
    bc: dict = field(default_factory=dict)     # name -> (n_times, nz, N)
    bc_times: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    ug: float = 0.0
    vg: float = 0.0


# ── 边界时间插值 ─────────────────────────────────────────────────────
def interp_bc_time(bc: dict, bc_times: np.ndarray, t_sim: float) -> dict:
    """在给定模拟时刻对边界字典做线性插值，返回 name -> (nz, N) 数组。"""
    n = len(bc_times)
    if n <= 1:
        return {k: v[0] for k, v in bc.items()}
    if t_sim <= bc_times[0]:
        return {k: v[0] for k, v in bc.items()}
    if t_sim >= bc_times[-1]:
        return {k: v[-1] for k, v in bc.items()}
    idx = int(np.searchsorted(bc_times, t_sim)) - 1
    idx = max(0, min(idx, n - 2))
    a = (t_sim - bc_times[idx]) / (bc_times[idx + 1] - bc_times[idx])
    a = max(0.0, min(1.0, a))
    return {k: (1 - a) * v[idx] + a * v[idx + 1] for k, v in bc.items()}


# ── 合成大气边界层 ─────────────────────────────────────────────────
def generate_synthetic_init(grid: Grid, cfg: GPULESConfig) -> InitData:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = np.arange(nz) * grid.dz + grid.dz / 2
    u_star = 0.35
    z0 = cfg.z0
    ug, vg = 5.65, 1.11
    u_prof = (u_star / cfg.kappa) * np.log((z + z0) / z0) + ug * (1 - np.exp(-z / 500))
    v_prof = vg * (1 - np.exp(-z / 500))
    pt_prof = 305.0 + 0.003 * z
    qv_prof = 0.018 * np.exp(-z / 2000)

    u = np.tile(u_prof[:, None, None], (1, ny, nx)).astype(np.float32)
    v = np.tile(v_prof[:, None, None], (1, ny, nx)).astype(np.float32)
    w = np.zeros((nz, ny, nx), np.float32)
    pt = np.tile(pt_prof[:, None, None], (1, ny, nx)).astype(np.float32)
    qv = np.tile(qv_prof[:, None, None], (1, ny, nx)).astype(np.float32)

    rng = np.random.RandomState(cfg.seed)
    u[:, 5:-5, 5:-5] += 0.15 * rng.randn(nz, ny - 10, nx - 10).astype(np.float32)
    v[:, 5:-5, 5:-5] += 0.15 * rng.randn(nz, ny - 10, nx - 10).astype(np.float32)

    bc = {}
    for name, arr in [("left_u", u[:, :, 0]), ("right_u", u[:, :, -1]),
                      ("left_v", v[:, :, 0]), ("right_v", v[:, :, -1]),
                      ("south_u", u[:, 0, :]), ("north_u", u[:, -1, :]),
                      ("south_v", v[:, 0, :]), ("north_v", v[:, -1, :]),
                      ("left_pt", pt[:, :, 0]), ("right_pt", pt[:, :, -1]),
                      ("south_pt", pt[:, 0, :]), ("north_pt", pt[:, -1, :])]:
        bc[name] = arr[None, ...]
    return InitData(u, v, w, pt, qv, bc, np.array([0.0]), ug, vg)


# ── PIDS 输入（兼容原 PIDS_STATIC / PIDS_DYNAMIC） ───────────────────
def load_pids_init(static_path: str, dynamic_path: str, grid: Grid, cfg: GPULESConfig) -> InitData:
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise DataLoadError("读取 PIDS 需要 netCDF4（pip install netCDF4）") from exc
    nz, ny, nx, dz = grid.nz, grid.ny, grid.nx, grid.dz
    ds = nc.Dataset(dynamic_path, "r")
    try:
        pids_z = np.asarray(ds.variables["z"][:])
        pids_time = np.asarray(ds.variables["time"][:])

        def load_3d(varname, stagger=None):
            var = ds.variables[varname][:]
            data = np.array(var)
            if hasattr(var, "fill_value"):
                data = np.ma.masked_equal(data, var.fill_value).filled(0.0)
            if data.ndim == 1:
                prof = np.interp(np.arange(nz) * dz + dz / 2, pids_z, data)
                return np.tile(prof[:, None, None], (1, ny, nx)).astype(np.float32)
            if data.shape[0] > nz:
                data = data[:nz]
            if stagger == "x" and data.shape[2] == nx - 1:
                data = np.pad(data, ((0, 0), (0, 0), (0, 1)), mode="edge")
            elif stagger == "y" and data.shape[1] == ny - 1:
                data = np.pad(data, ((0, 0), (0, 1), (0, 0)), mode="edge")
            if data.shape[1] < ny:
                data = np.pad(data, ((0, 0), (0, ny - data.shape[1]), (0, 0)), mode="edge")
            if data.shape[2] < nx:
                data = np.pad(data, ((0, 0), (0, 0), (0, nx - data.shape[2])), mode="edge")
            return data.astype(np.float32)

        u = load_3d("init_atmosphere_u", "x")
        v = load_3d("init_atmosphere_v", "y")
        w = load_3d("init_atmosphere_w")
        if w.shape[0] == nz - 1:
            w_full = np.zeros((nz, ny, nx), np.float32)
            w_full[:nz - 1] = w
            w = w_full
        pt = load_3d("init_atmosphere_pt")
        qv = load_3d("init_atmosphere_qv")

        bc_all: dict = {}
        for name in ["left_u", "left_v", "left_w", "left_pt",
                     "south_u", "south_v", "south_w", "south_pt",
                     "right_u", "right_v", "right_w", "right_pt",
                     "north_u", "north_v", "north_w", "north_pt"]:
            var = ds.variables[f"ls_forcing_{name}"]
            data = np.array(var[:])
            if hasattr(var, "fill_value"):
                data = np.ma.masked_equal(data, var.fill_value).filled(0.0)
            if data.shape[1] > nz:
                data = data[:, :nz, :]
            elif data.shape[1] < nz:
                data = np.pad(data, ((0, 0), (0, nz - data.shape[1]), (0, 0)), mode="edge")
            bc_all[name] = data.astype(np.float32)
    finally:
        ds.close()

    ug = float(u[nz // 5:].mean())
    vg = float(v[nz // 5:].mean())
    return InitData(u, v, w, pt, qv, bc_all, pids_time.astype(np.float32), ug, vg)


# ── 建筑高度场加载 ─────────────────────────────────────────────────
def load_buildings_2d(static_path: str, grid: Grid) -> np.ndarray:
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise DataLoadError("读取建筑数据需要 netCDF4（pip install netCDF4）") from exc
    if not static_path or not __import__("os").path.exists(static_path):
        return np.zeros((grid.ny, grid.nx), np.float32)
    ds = nc.Dataset(static_path, "r")
    try:
        bldg_2d = np.array(ds.variables["buildings_2d"][:])
    finally:
        ds.close()
    ny, nx = grid.ny, grid.nx
    sy, sx = bldg_2d.shape
    # 精确整除(如 400x400 -> 100x100): block-max 池化, 覆盖整域、保留城市冠层高度。
    # 修复: 旧版用中心裁剪 (y0=(sy-ny)//2) 会把外圈 3/4 建筑整个丢弃。
    if sy >= ny and sx >= nx and sy % ny == 0 and sx % nx == 0:
        fy, fx = sy // ny, sx // nx
        bldg_2d = bldg_2d[:ny * fy, :nx * fx].reshape(ny, fy, nx, fx).max(axis=(1, 3))
    elif sy >= ny and sx >= nx:  # 同域非整除: 双线性兜底
        yy = np.clip((np.arange(ny) + 0.5) * sy / ny - 0.5, 0, sy - 1)
        xx = np.clip((np.arange(nx) + 0.5) * sx / nx - 0.5, 0, sx - 1)
        by = np.floor(yy).astype(int)
        ty = np.minimum(by + 1, sy - 1)
        bx = np.floor(xx).astype(int)
        tx = np.minimum(bx + 1, sx - 1)
        wy = (yy - by)[:, None]
        wx = (xx - bx)[None, :]
        bldg_2d = (bldg_2d[by][:, bx] * (1 - wy) * (1 - wx)
                   + bldg_2d[by][:, tx] * (1 - wy) * wx
                   + bldg_2d[ty][:, bx] * wy * (1 - wx)
                   + bldg_2d[ty][:, tx] * wy * wx).astype(bldg_2d.dtype)
    else:  # 静态场比求解器网格小: 边缘填充(兼容旧合成场)
        if sy < ny:
            bldg_2d = np.pad(bldg_2d, ((0, ny - sy), (0, max(0, nx - sx))), mode="edge")
        if bldg_2d.shape[1] < nx:
            bldg_2d = np.pad(bldg_2d, ((0, 0), (0, nx - bldg_2d.shape[1])), mode="edge")
        if bldg_2d.shape[1] > nx:
            bldg_2d = bldg_2d[:, :nx]
        if bldg_2d.shape[0] > ny:
            bldg_2d = bldg_2d[:ny, :]
    return bldg_2d.astype(np.float32)


# ── AI 预报（FuXi/GraphCast）加载 ───────────────────────────────────
def detect_ai_format(ds) -> str:
    vars_list = list(ds.variables.keys())
    gc = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
    if all(m in vars_list for m in gc):
        return "graphcast"
    fuxi = ["u10", "v10", "t2m", "msl"]
    if all(m in vars_list for m in fuxi):
        return "fuxi"
    return "unknown"


def load_ai_forecast(ai_path: str, grid: Grid, cfg: GPULESConfig) -> InitData:
    """从 FuXi/GraphCast NetCDF 提取初始化场 + 时变边界（端口自 v9）。

    支持格式自动检测、气压→高度转换、T→位温、三线性插值降尺度。
    需要 netCDF4 与文件存在；否则抛 ``DataLoadError``。
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover
        raise DataLoadError("读取 AI 预报需要 netCDF4（pip install netCDF4）") from exc
    if not __import__("os").path.exists(ai_path):
        raise DataLoadError(f"AI 预报数据不存在: {ai_path}")
    nz, dz = grid.nz, grid.dz
    ds = nc.Dataset(ai_path, "r")
    try:
        fmt = detect_ai_format(ds)
        if fmt == "unknown":
            raise SchemaError(f"无法识别的 AI 预报格式 (变量: {list(ds.variables.keys())[:10]})")

        def gv(names):
            for n in names:
                if n in ds.variables:
                    return ds.variables[n]
            raise SchemaError(f"缺少变量: {names}")

        lat = np.asarray(gv(["lat", "latitude"])[:])
        lon = np.asarray(gv(["lon", "longitude"])[:])
        lats = lat[:, 0] if lat.ndim == 2 else lat
        lons = lon[0, :] if lon.ndim == 2 else lon
        n_lat, n_lon = len(lats), len(lons)
        lat_idx = int(np.argmin(np.abs(lats - cfg.lat)))
        lon_idx = int(np.argmin(np.abs(lons - cfg.lon)))
        try:
            p_levels = np.asarray(gv(["level", "plev", "isobaricInhPa", "pressure_level"])[:], dtype=np.float64)
        except SchemaError:
            p_levels = np.array([1000, 925, 850, 700, 500, 300, 200, 100, 50], dtype=np.float64)
        if p_levels.max() < 2000:
            p_pa = p_levels * 100.0
        else:
            p_pa = p_levels

        n_half = max(int(grid.domain[0] / (111000 * abs(lats[1] - lats[0]))) // 2 + 3, 2) if n_lat > 1 else 3
        y0 = max(0, lat_idx - n_half)
        x0 = max(0, lon_idx - n_half)
        y1 = min(n_lat, y0 + 2 * n_half)
        x1 = min(n_lon, x0 + 2 * n_half)

        u_var = gv(["u_component_of_wind", "u", "ua", "UGRD"])
        v_var = gv(["v_component_of_wind", "v", "va", "VGRD"])
        t_var = gv(["temperature", "t", "ta", "TMP"])

        def get_slice(var, it):
            data = np.array(var[it:it + 1] if var.ndim == 4 else var[:])
            if data.ndim == 4:
                data = data[0]
            return data[:, y0:y1, x0:x1]

        bc_all: dict = {}
        u_init = v_init = pt_init = qv_init = None
        ug = vg = 0.0
        n_use = 1
        for it in range(n_use):
            u_lev = get_slice(u_var, it)
            v_lev = get_slice(v_var, it)
            t_lev = get_slice(t_var, it)
            # 简化：直接垂直插值到等距层（气压→高度 + 位温）
            pt_lev = np.zeros_like(t_lev, dtype=np.float32)
            for k in range(len(p_pa)):
                pt_lev[k] = t_lev[k] * (cfg.p_ref / p_pa[k]) ** (cfg.r_dry / cfg.cp_dry)
            ny_sub, nx_sub = u_lev.shape[1], u_lev.shape[2]
            z_les = np.arange(nz) * dz + dz / 2
            if len(p_pa) > 1:
                # 气压→几何高度（静力近似），再垂直插值到 LES 等距层
                t_col_mean = np.nanmean(t_lev.reshape(len(p_pa), -1), axis=1)
                order = np.argsort(p_pa)[::-1]
                p_sorted = p_pa[order]
                t_sorted = np.maximum(t_col_mean[order], 200.0)
                z_col = (cfg.r_dry * t_sorted / cfg.g) * np.log(np.maximum(cfg.p_ref / p_sorted, 1.0))
                u_lev = u_lev[order]
                v_lev = v_lev[order]
                pt_lev = pt_lev[order]
                idx = np.clip(np.searchsorted(z_col, z_les), 1, len(z_col) - 1)
                lo = z_col[idx - 1]
                hi = z_col[idx]
                frac = (z_les - lo) / np.maximum(hi - lo, 1e-9)  # (nz,)
                U = u_lev.reshape(len(p_pa), -1)
                V = v_lev.reshape(len(p_pa), -1)
                PT = pt_lev.reshape(len(p_pa), -1)
                u_z = (U[idx - 1] * (1.0 - frac[:, None]) + U[idx] * frac[:, None]).reshape(nz, ny_sub, nx_sub).astype(np.float32)
                v_z = (V[idx - 1] * (1.0 - frac[:, None]) + V[idx] * frac[:, None]).reshape(nz, ny_sub, nx_sub).astype(np.float32)
                pt_z = (PT[idx - 1] * (1.0 - frac[:, None]) + PT[idx] * frac[:, None]).reshape(nz, ny_sub, nx_sub).astype(np.float32)
            else:
                u_z = np.broadcast_to(u_lev[0], (nz, ny_sub, nx_sub)).astype(np.float32)
                v_z = np.broadcast_to(v_lev[0], (nz, ny_sub, nx_sub)).astype(np.float32)
                pt_z = np.broadcast_to(pt_lev[0], (nz, ny_sub, nx_sub)).astype(np.float32)
            u_init = u_z
            v_init = v_z
            pt_init = pt_z
            qv_init = np.zeros_like(u_init)
            ug = float(u_init[nz // 5:].mean())
            vg = float(v_init[nz // 5:].mean())
            for name, arr in [("left_u", u_init[:, :, 0]), ("right_u", u_init[:, :, -1]),
                              ("left_v", v_init[:, :, 0]), ("right_v", v_init[:, :, -1]),
                              ("south_u", u_init[:, 0, :]), ("north_u", u_init[:, -1, :]),
                              ("south_v", v_init[:, 0, :]), ("north_v", v_init[:, -1, :]),
                              ("left_pt", pt_init[:, :, 0]), ("right_pt", pt_init[:, :, -1]),
                              ("south_pt", pt_init[:, 0, :]), ("north_pt", pt_init[:, -1, :])]:
                bc_all[name] = arr[None, ...]
        w_init = np.zeros_like(u_init)
    finally:
        ds.close()
    return InitData(u_init, v_init, w_init, pt_init, qv_init, bc_all, np.array([0.0]), ug, vg)
