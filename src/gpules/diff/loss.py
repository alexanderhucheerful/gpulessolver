"""可微训练损失（来自 diff-v1）。

``loss = MSE(模拟层均值, PALM 参考) + L2 正则``

  * 数据项：在若干参考高度层比较模拟与 PALM 的 u/v 层均值
  * 正则项：惩罚参数偏离初值，防止训练发散到非物理解

``palm_ref_*`` 与 ``l2_reg`` 来自配置，便于不同城市/场景替换参考目标。
"""

from __future__ import annotations

import torch

from ..config import GPULESConfig
from .params import LearnablePhysicsParams


def compute_loss(u: torch.Tensor, v: torch.Tensor, params: LearnablePhysicsParams,
                 cfg: GPULESConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import logging

    device = u.device
    nz = u.shape[0]
    loss_data = torch.tensor(0.0, device=device)
    used = 0
    for i, k_idx in enumerate(cfg.palm_ref_k):
        if k_idx < 0 or k_idx >= nz:
            # 参考层超出当前网格高度：跳过而非崩溃（小网格训练时常见）
            continue
        u_mean = u[k_idx].mean()
        v_mean = v[k_idx].mean()
        loss_data = loss_data + (u_mean - cfg.palm_ref_u[i]) ** 2
        loss_data = loss_data + (v_mean - cfg.palm_ref_v[i]) ** 2
        used += 1
    if used == 0:
        logging.getLogger("gpules").warning(
            "compute_loss: 所有 PALM 参考层 %s 均超出网格 nz=%d，数据项被置零",
            cfg.palm_ref_k, nz,
        )

    loss_reg = torch.tensor(0.0, device=device)
    init = params.initial_dict()
    for name, p in params.named_parameters():
        loss_reg = loss_reg + (p - init[name]) ** 2

    total = loss_data + cfg.l2_reg * loss_reg
    return total, loss_data, loss_reg
