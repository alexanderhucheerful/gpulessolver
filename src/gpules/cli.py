"""命令行入口。

用法示例：
  # 推理（合成数据，快速验证）
  python -m gpules inference --synthetic --nx 64 --ny 64 --nz 32 --end-time 30

  # 推理（PIDS 真实数据）
  python -m gpules inference --static-path data/PIDS_STATIC --dynamic-path data/PIDS_DYNAMIC

  # 可微训练
  python -m gpules train --synthetic --nx 32 --ny 32 --nz 24 \
      --train-window 8 --train-epochs 3 --learning-rate 0.002

  # 用 YAML 配置
  python -m gpules inference --config configs/default.yaml --synthetic

所有参数也可用 ``configs/default.yaml`` 集中管理；CLI 覆盖优先级高于 YAML。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

from . import __version__
from .config import GPULESConfig
from .core.solver import LESSolver
from .diff.trainer import Trainer
from .exceptions import GpulesError
from .io.data import (
    generate_synthetic_init,
    load_ai_forecast,
    load_buildings_2d,
    load_pids_init,
)
from .io.output import write_comparison_data, write_output_json
from .logging import get_logger, setup_logging

_BOOL_KEYS = ("synthetic", "no_buildings", "diagnostics", "use_fp16", "double_projection")
_OVERRIDE_KEYS = [
    "mode", "nx", "ny", "nz", "dx", "dy", "dz", "device", "seed", "log_level",
    "end_time", "dt_init", "dt_min", "dt_max", "cfl_max", "vmax", "wmax", "div_max",
    "advection", "pressure_solver", "time_integrator", "relax_width", "relax_strength",
    "rayleigh_start", "rayleigh_coef", "synthetic", "ai_forecast", "no_buildings",
    "static_path", "dynamic_path", "palm_json", "output", "diagnostics",
    "use_fp16", "double_projection", "train_window", "train_epochs",
    "learning_rate", "ckpt_interval", "l2_reg", "lat", "lon", "omega",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpules", description=f"gpules v{__version__} GPU LES 求解器")
    p.add_argument("--version", action="version", version=f"gpules {__version__}")
    sub = p.add_subparsers(dest="command")

    for cmd in ("inference", "train"):
        sp = sub.add_parser(cmd, help=f"{cmd} 模式")
        sp.add_argument("--config", default=None, help="YAML 配置文件路径")
        for k in _OVERRIDE_KEYS:
            if k in _BOOL_KEYS:
                continue
            elif k in ("nx", "ny", "nz", "train_window", "train_epochs", "ckpt_interval"):
                sp.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
            elif k in ("dx", "dy", "dz", "end_time", "dt_init", "dt_min", "dt_max", "cfl_max",
                       "vmax", "wmax", "div_max", "rayleigh_start", "rayleigh_coef",
                       "relax_strength", "learning_rate", "l2_reg"):
                sp.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
            else:
                sp.add_argument(f"--{k.replace('_', '-')}", default=None)
        for k in _BOOL_KEYS:
            flag = k.replace("_", "-")
            sp.add_argument(f"--{flag}", dest=k, action="store_true", default=None)
            if k != "no_buildings":
                sp.add_argument(f"--no-{flag}", dest=k, action="store_false", default=None)
        # 兼容旧 v9 开关
        sp.add_argument("--differentiable", action="store_true", help="等价于 --mode train")
    return p


def _apply_overrides(cfg: GPULESConfig, args: argparse.Namespace) -> None:
    for k in _OVERRIDE_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    if getattr(args, "differentiable", False):
        cfg.mode = "train"
    if args.command == "train" and cfg.mode != "train":
        cfg.mode = "train"
    # 科氏参数随纬度/经度/地球自转角速度变化，覆盖这些字段时必须重算
    # （直接 setattr 绕过了 __post_init__，否则 --lat 30 改了纬度却对科氏力毫无影响）
    if getattr(args, "lat", None) is not None or getattr(args, "lon", None) is not None \
            or getattr(args, "omega", None) is not None:
        cfg.f_cor = 2.0 * cfg.omega * math.sin(math.radians(cfg.lat))


def _load_init(cfg: GPULESConfig):
    """根据配置加载初始化数据与建筑场。"""
    import torch

    from .core.grid import Grid
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    if cfg.ai_forecast:
        init = load_ai_forecast(cfg.ai_forecast, grid, cfg)
        bldg = load_buildings_2d(cfg.static_path, grid) if not cfg.no_buildings else None
    elif cfg.synthetic:
        init = generate_synthetic_init(grid, cfg)
        bldg = None if cfg.no_buildings else _synthetic_bldg(cfg, grid)
    elif cfg.static_path and cfg.dynamic_path:
        init = load_pids_init(cfg.static_path, cfg.dynamic_path, grid, cfg)
        bldg = load_buildings_2d(cfg.static_path, grid) if not cfg.no_buildings else None
    else:
        raise GpulesError("需提供 --synthetic / --ai-forecast / (--static-path + --dynamic-path)")
    return init, bldg


def _synthetic_bldg(cfg, grid):
    from .core.boundary import generate_synthetic_buildings
    bf = generate_synthetic_buildings(grid, cfg)
    return bf.bldg_h.cpu().numpy()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    cfg = GPULESConfig.from_yaml(args.config) if args.config else GPULESConfig()
    _apply_overrides(cfg, args)
    cfg.validate()
    cfg.resolve_paths(os.getcwd())
    setup_logging(cfg.log_level)
    log = get_logger("cli")

    try:
        solver = LESSolver(cfg)
        init, bldg = _load_init(cfg)
        solver.prepare(init, bldg_2d=bldg)

        if args.command == "train":
            trainer = Trainer(solver, cfg)
            train_log = trainer.train()
            out_path = cfg.output or os.path.join(os.getcwd(), "diff_les_train_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trainer.result_dict(), f, indent=2)
            log.info("训练结果已保存: %s", out_path)
            # 写出最终对比数据
            cmp_path = os.path.join(os.path.dirname(out_path), "gpu_les_comparison_data.json")
            write_comparison_data(train_log, cfg, cmp_path,
                                  gpu=solver.device.type)
            return 0

        # inference
        solver.run()
        buildings = []
        if cfg.palm_json and os.path.exists(cfg.palm_json):
            with open(cfg.palm_json, "r", encoding="utf-8") as f:
                buildings = json.load(f).get("buildings", [])
        write_output_json(solver.state, cfg, cfg.output, buildings=buildings,
                          diagnostics=cfg.diagnostics)
        log.info("输出已写入: %s", cfg.output)
        return 0
    except GpulesError as e:
        log.error("配置/数据错误: %s", e)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        log.warning("用户中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
