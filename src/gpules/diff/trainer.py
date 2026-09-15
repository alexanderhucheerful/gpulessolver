"""可微训练器（来自 diff-v1 训练闭环）。

把原项目"import 即训练"的顶层循环收为一个 :class:`Trainer`，支持：

  * Adam 优化可学习物理参数
  * FP16 混合精度（autocast + GradScaler，仅 cuda）
  * 梯度检查点（gradient checkpointing，省显存）
  * 梯度裁剪 + 梯度非零校验
  * 训练日志与结果落盘

``Trainer`` 不触发任何副作用（不 import 即运行），可被 pytest 直接驱动。
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from ..config import GPULESConfig
from ..core.diff_helpers import fused_step_diff
from ..core.solver import LESSolver
from ..device import autocast_ctx
from ..logging import get_logger
from .loss import compute_loss
from .stable import clear_mask_cache


class Trainer:
    def __init__(self, solver: LESSolver, cfg: GPULESConfig):
        if cfg.mode != "train":
            raise ValueError("Trainer 仅用于 mode='train'")
        if solver.params is None:
            raise ValueError("solver 尚未 prepare（缺少 LearnablePhysicsParams）")
        self.solver = solver
        self.cfg = cfg
        self.params = solver.params
        self.grid = solver.grid
        self.pressure = solver.pressure
        self.advection = solver.advection
        self.aux = solver.aux
        self.device = solver.device
        self.logger = get_logger("diff.trainer")
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=cfg.learning_rate)
        self.use_amp = bool(cfg.use_fp16) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.init_state = solver.state.clone()
        self.train_log: list[dict] = []

    def _reset_state(self) -> None:
        self.solver.state = self.init_state.clone()
        self.solver.state.u.requires_grad_(True)
        self.solver.t_sim = 0.0

    def train(self, epochs: Optional[int] = None, window: Optional[int] = None) -> list[dict]:
        epochs = epochs or self.cfg.train_epochs
        window = window or self.cfg.train_window
        dt = self.cfg.dt_init
        amp_ctx = autocast_ctx(self.device, enabled=self.use_amp)
        ckpt_every = max(1, self.cfg.ckpt_interval)
        if window % ckpt_every != 0:
            self.logger.warning(
                "train_window=%d 不是 ckpt_interval=%d 的整数倍，将精确跑满 %d 步",
                window, ckpt_every, window,
            )

        for epoch in range(epochs):
            t0 = time.time()
            self._reset_state()
            self.optimizer.zero_grad()

            u = self.solver.state.u
            v = self.solver.state.v
            w = self.solver.state.w
            pt = self.solver.state.pt
            tke = self.solver.state.tke

            def run_chunk(u_c, v_c, w_c, pt_c, tke_c, n_steps: int):
                for _ in range(n_steps):
                    u_c, v_c, w_c, pt_c, tke_c = fused_step_diff(
                        u_c, v_c, w_c, pt_c, tke_c, dt, self.params, self.aux,
                        self.cfg, self.grid, self.pressure, self.advection)
                return u_c, v_c, w_c, pt_c, tke_c

            with amp_ctx:
                steps_done = 0
                while steps_done < window:
                    chunk = min(ckpt_every, window - steps_done)
                    use_ckpt = chunk == ckpt_every and window - steps_done > chunk
                    if use_ckpt:
                        u, v, w, pt, tke = ckpt(
                            run_chunk, u, v, w, pt, tke, chunk, use_reentrant=False)
                    else:
                        u, v, w, pt, tke = run_chunk(u, v, w, pt, tke, chunk)
                    steps_done += chunk
                    self.solver.t_sim += chunk * dt

                loss, loss_data, loss_reg = compute_loss(u, v, self.params, self.cfg)

            self.scaler.scale(loss).backward()
            # 梯度非零校验（可学习性验收）
            grad_ok = True
            grad_info = []
            for name, p in self.params.named_parameters():
                if p.grad is None:
                    grad_ok = False
                    grad_info.append(f"{name}: NO GRAD")
                elif abs(float(p.grad.item())) < 1e-12:
                    grad_ok = False
                    grad_info.append(f"{name}: grad~0")
                else:
                    grad_info.append(f"{name}: grad={float(p.grad.item()):+.6f}")

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.params.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            clear_mask_cache()

            self.train_log.append({
                "epoch": epoch, "loss": float(loss.item()),
                "loss_data": float(loss_data.item()), "loss_reg": float(loss_reg.item()),
                "time": time.time() - t0, "grad_ok": grad_ok,
            })
            self.logger.info("Epoch %d/%d | loss=%.6f (data=%.6f, reg=%.6f) | %s",
                             epoch + 1, epochs, loss.item(), loss_data.item(), loss_reg.item(),
                             "OK" if grad_ok else "GRAD WARN")
            if not grad_ok:
                self.logger.warning("梯度问题: %s", grad_info)
        return self.train_log

    def result_dict(self) -> dict:
        return {
            "method": "gpules differentiable v1.0",
            "train_config": {
                "window": self.cfg.train_window, "epochs": self.cfg.train_epochs,
                "lr": self.cfg.learning_rate, "ckpt_interval": self.cfg.ckpt_interval,
                "l2_reg": self.cfg.l2_reg,
            },
            "initial_params": self.params.initial_dict(),
            "optimized_params": {n: p.item() for n, p in self.params.named_parameters()},
            "train_log": self.train_log,
        }
