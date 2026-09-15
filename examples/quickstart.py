"""gpules v1.0 快速入门。

演示两条主线的最小可运行示例（纯 Python API，无需 GPU）：

  1. inference —— 高精度线（5 阶 WS 平流 + 多网格压力 + SSP-RK3 + Deardorff SGS）
  2. train     —— 可微线（central2 + RFFT 压力 + Heun，10 个可学习物理参数在 PALM 监督下被训练）

运行：python examples/quickstart.py
"""

import os
import sys

# 让脚本在「未安装包」时也能 import gpules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from gpules.config import GPULESConfig
from gpules.core.diff_helpers import fused_step_diff
from gpules.core.grid import Grid
from gpules.core.solver import LESSolver
from gpules.diff.loss import compute_loss
from gpules.io.data import generate_synthetic_init
from gpules.io.output import write_output_json


def inference_demo():
    print("[1/2] 推理（高精度线）...")
    cfg = GPULESConfig(
        nx=48, ny=48, nz=24, dx=10.0, dy=10.0, dz=10.0,
        end_time=20.0, dt_init=0.1, device="cpu", mode="inference",
    )
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    log = solver.run(end_time=cfg.end_time)
    out_path = os.path.join(os.path.dirname(__file__), "wind_inference.json")
    write_output_json(solver.state, cfg, out_path, diagnostics=True)
    print(f"     完成 {solver.step_count} 步，末态 spd_max={log[-1]['spd_max']:.3f} m/s")
    print(f"     输出: {out_path}")


def train_demo():
    print("[2/2] 训练（可微线，1 epoch 演示）...")
    cfg = GPULESConfig(
        mode="train", nx=32, ny=32, nz=16, dx=10.0, dy=10.0, dz=10.0,
        dt_init=0.05, device="cpu", train_window=4, train_epochs=1,
        palm_ref_k=[1, 3, 5, 7], palm_ref_u=[0.5, 1.0, 1.5, 2.0],
        palm_ref_v=[0.3, 0.6, 0.9, 1.2],
    )
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    u = solver.state.u.clone().requires_grad_(True)
    v = solver.state.v.clone().requires_grad_(True)
    w = solver.state.w.clone()
    pt = solver.state.pt.clone()
    tke = solver.state.tke.clone().requires_grad_(True)
    for _ in range(cfg.train_window):
        u, v, w, pt, tke = fused_step_diff(
            u, v, w, pt, tke, cfg.dt_init, solver.params, solver.aux, cfg, solver.grid, solver.pressure
        )
    loss, loss_data, loss_reg = compute_loss(u, v, solver.params, cfg)
    loss.backward()
    grads = {n: float(p.grad.abs().mean()) for n, p in solver.params.named_parameters() if p.grad is not None}
    print(f"     训练窗口损失={float(loss):.4f} (data={float(loss_data):.4f}, reg={float(loss_reg):.4f})")
    print(f"     非零梯度参数数={len(grads)}/10")


if __name__ == "__main__":
    inference_demo()
    train_demo()
    print("完成。")
