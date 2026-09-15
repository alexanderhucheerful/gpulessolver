"""gpules 全局配置。

原项目所有物理/数值参数散落在模块顶层全局常量中（README 自述"需改源码调整"），
且副本间极易漂移。这里统一为 :class:`GPULESConfig` 单一事实来源：

  * 可被 YAML 加载/保存（运维友好、可复现）
  * 可被命令行参数覆盖
  * 带有 ``validate()`` 校验，非法配置在入口即报错而非跑到一半崩溃
  * ``mode`` 区分 ``inference``（高精度线）与 ``train``（可微线），
    其余参数描述两条线共享的物理常数与可插拔策略的选择

设计原则：参数即配置，配置即文档。
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Literal

from .exceptions import ConfigError

Mode = Literal["inference", "train"]
AdvectionKind = Literal["ws5", "central2"]
PressureKind = Literal["multigrid", "cg", "jacobi", "poisson_fft"]
IntegratorKind = Literal["ssprk3", "heun"]


@dataclass
class GPULESConfig:
    # ── 运行模式与设备 ──────────────────────────────────────────────
    mode: Mode = "inference"
    device: str = "auto"          # auto | cuda | mps | cpu
    seed: int = 42
    log_level: str = "INFO"

    # ── 网格 ───────────────────────────────────────────────────────
    nx: int = 200
    ny: int = 200
    nz: int = 100
    dx: float = 5.0
    dy: float = 5.0
    dz: float = 5.0

    # ── 物理常数（两线共享） ────────────────────────────────────────
    omega: float = 7.292e-5
    lat: float = 23.1291
    lon: float = 113.2644
    g: float = 9.81
    kappa: float = 0.41
    z0: float = 1.0
    p_ref: float = 101325.0
    pt_ref: float = 300.0
    r_dry: float = 287.04
    cp_dry: float = 1005.0
    l_vap: float = 2.501e6
    epsilon: float = 0.622
    f_cor: float = 0.0          # 由 omega/lat 计算

    # ── Deardorff TKE SGS ──────────────────────────────────────────
    ce: float = 0.845
    ck: float = 0.094
    cnu: float = 0.10
    nu_min: float = 0.5
    tke_min: float = 1e-4
    pr_t: float = 1.0 / 3.0
    sigma_e: float = 1.0

    # ── 平流策略 ───────────────────────────────────────────────────
    advection: AdvectionKind = "ws5"   # inference 默认 5 阶 WS
    ad_k2: float = 0.5                 # central2 的 Jameson 2 阶系数
    ad_k4: float = 1.0 / 32.0          # central2 的 Jameson 4 阶系数

    # ── 压力求解策略 ───────────────────────────────────────────────
    pressure_solver: PressureKind = "multigrid"  # inference 默认多网格
    mg_levels: int = 4
    mg_vcycles: int = 3
    mg_smooth: int = 3
    jacobi_omega: float = 0.667
    pressure_tol: float = 1e-6
    cg_iters: int = 150

    # ── 时间积分策略 ───────────────────────────────────────────────
    time_integrator: IntegratorKind = "ssprk3"  # inference 默认 SSP-RK3
    end_time: float = 120.0
    dt_init: float = 0.05
    dt_min: float = 0.005
    dt_max: float = 0.15
    cfl_max: float = 0.5
    vmax: float = 20.0
    wmax: float = 15.0
    div_max: float = 0.1

    # ── 边界与松弛 ─────────────────────────────────────────────────
    relax_width: int = 5
    relax_strength: float = 0.3
    sponge_width: int = 8        # train 模式的 sponge 层宽度
    sponge_top: int = 10
    sponge_strength: float = 0.5

    # ── 物理过程开关 ───────────────────────────────────────────────
    use_coriolis: bool = True
    use_building_drag: bool = True
    use_wall_drag: bool = True
    use_rayleigh: bool = True
    use_excess_suppression: bool = True   # 过速抑制（train 默认开启）
    ground_drag_layers: int = 2
    canopy_smooth: int = 5
    log_law_top: float = 15.0

    # ── Rayleigh 阻尼 ──────────────────────────────────────────────
    rayleigh_start: float = 0.75
    rayleigh_coef: float = 0.01

    # ── 可微近似参数（train 模式生效） ─────────────────────────────
    use_soft_mask: bool = True
    smooth_width: float = 10.0       # sigmoid 软壁面过渡宽度 (m)
    soft_sign_eps: float = 0.05
    gate_sharpness: float = 50.0
    softplus_sharpness: float = 8.0
    vel_cap_k: float = 8.0
    use_fp16: bool = False
    double_projection: bool = True   # train 模式双压力投影

    # ── 训练配置（train 模式生效） ─────────────────────────────────
    train_window: int = 10
    train_epochs: int = 5
    learning_rate: float = 0.002
    ckpt_interval: int = 5
    l2_reg: float = 0.001
    # PALM 参考层均值（loss 目标），与 palm_ref_k 一一对应。
    # 默认与 configs/default.yaml 一致（PALM 参考层均值）；
    # 真实训练须用 palm_json 中的 PALM 廓线覆盖（见 README）。
    palm_ref_u: list[float] = field(default_factory=lambda: [0.29, 0.75, 1.08, 1.90])
    palm_ref_v: list[float] = field(default_factory=lambda: [1.73, 3.66, 5.34, 8.22])
    palm_ref_k: list[int] = field(default_factory=lambda: [1, 9, 19, 39])

    # ── 输入/输出路径 ──────────────────────────────────────────────
    static_path: str | None = None
    dynamic_path: str | None = None
    palm_json: str | None = None
    ai_forecast: str | None = None
    synthetic: bool = False
    no_buildings: bool = False
    output: str | None = None
    diagnostics: bool = False

    # ── 派生字段（运行时填充，不进 YAML） ──────────────────────────
    data_dir: str | None = None

    # ───────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # Coriolis 参数由纬度推导
        self.f_cor = 2.0 * self.omega * __import__("math").sin(__import__("math").radians(self.lat))
        self.validate()

    # ── 校验 ────────────────────────────────────────────────────────
    def validate(self) -> None:
        if self.mode not in ("inference", "train"):
            raise ConfigError(f"mode 必须为 inference/train，得到 {self.mode!r}")
        for name in ("nx", "ny", "nz"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} 必须为正整数，得到 {getattr(self, name)}")
        for name in ("dx", "dy", "dz"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} 必须为正数，得到 {getattr(self, name)}")
        if self.advection not in ("ws5", "central2"):
            raise ConfigError(f"advection 非法: {self.advection!r}")
        if self.pressure_solver not in ("multigrid", "cg", "jacobi", "poisson_fft"):
            raise ConfigError(f"pressure_solver 非法: {self.pressure_solver!r}")
        if self.time_integrator not in ("ssprk3", "heun"):
            raise ConfigError(f"time_integrator 非法: {self.time_integrator!r}")
        if self.dt_min <= 0 or self.dt_max <= self.dt_min or self.dt_init <= 0:
            raise ConfigError("时间步配置非法：需 0 < dt_min < dt_max 且 dt_init > 0")
        if len(self.palm_ref_u) != len(self.palm_ref_v) or len(self.palm_ref_u) != len(self.palm_ref_k):
            raise ConfigError("palm_ref_u / palm_ref_v / palm_ref_k 长度必须一致")
        if self.mode == "inference" and self.pressure_solver == "poisson_fft":
            import logging
            logging.getLogger("gpules").warning(
                "inference 模式使用 poisson_fft：谱求解假设三方向周期边界，"
                "非周期物理域上与 multigrid 不等价，结果仅供调试",
            )
        if self.mode == "train" and self.nx * self.ny * self.nz > 1_000_000:
            # 训练小子域可省显存；超大域训练会 OOM，给出明确告警
            import logging
            logging.getLogger("gpules").warning(
                "train 模式使用超大网格 (%dx%dx%d)，可训练显存需求高，建议小子域",
                self.nx, self.ny, self.nz,
            )

    # ── 路径解析 ───────────────────────────────────────────────────
    def resolve_paths(self, base_dir: str | None = None) -> None:
        """用工作目录补全相对路径默认值（不覆盖用户显式传入）。"""
        base = base_dir or os.getcwd()
        self.data_dir = os.path.join(base, "data") if self.data_dir is None else self.data_dir
        if self.static_path is None:
            self.static_path = os.path.join(self.data_dir, "PIDS_STATIC")
        if self.dynamic_path is None:
            self.dynamic_path = os.path.join(self.data_dir, "PIDS_DYNAMIC")
        if self.palm_json is None:
            self.palm_json = os.path.join(self.data_dir, "wind_data_palm.json")
        if self.output is None:
            self.output = os.path.join(base, "wind_data_gpu_3d.json")

    # ── CLI 绑定 ───────────────────────────────────────────────────
    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "GPULESConfig":
        """从 argparse 命名空间构造配置；出现的值覆盖默认值。"""
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        # 兼容旧 ``--differentiable`` 开关 → mode="train"
        if getattr(args, "differentiable", False):
            overrides["mode"] = "train"
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in overrides.items() if k in valid}
        cfg = cls(**kwargs)
        cfg.resolve_paths(os.getcwd())
        return cfg

    # ── YAML ───────────────────────────────────────────────────────
    def to_yaml(self, path: str) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("写入 YAML 需要 pyyaml（pip install pyyaml）") from exc
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> "GPULESConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("读取 YAML 需要 pyyaml（pip install pyyaml）") from exc
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - valid
        if unknown:
            import logging
            logging.getLogger("gpules").warning(
                "YAML %s 含未知键（已忽略）: %s", path, sorted(unknown),
            )
        kwargs = {k: v for k, v in data.items() if k in valid}
        cfg = cls(**kwargs)
        return cfg
