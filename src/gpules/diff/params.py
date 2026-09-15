"""可学习物理参数（nn.Module）。

diff-v1 的核心创新：把 10 个物理参数提升为 ``nn.Parameter``，使求解器可在
PALM 参考场监督下被训练（Adam 自动调参）。

默认初值沿用 diff-v1 调优结果。实例化后即 ``to(device)`` 由调用方决定。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LearnablePhysicsParams(nn.Module):
    """所有可学习物理参数，初值 = diff-v1 调优结果。

    覆盖：建筑拖曳 (cd_face_x/y)、地面粗糙度 (cd_ground)、Deardorff 耗散
    (ce)、壁面函数 (wall_nudge/v)、U/V nudging 强度、过速抑制 (excess_cd)。
    """

    def __init__(self, init: Optional[dict] = None):
        super().__init__()
        defaults = {
            "cd_face_x": 0.55,
            "cd_face_y": 0.15,
            "cd_ground": 0.15,
            "ce": 0.845,
            "wall_nudge": 0.015,
            "wall_nudge_v": 0.022,
            "nudge_u_peak": 0.040,
            "nudge_u_upper": 0.050,
            "nudge_v_upper": 0.045,
            "excess_cd": 0.4,
        }
        if init:
            defaults.update({k: float(v) for k, v in init.items()})
        self.cd_face_x = nn.Parameter(torch.tensor(defaults["cd_face_x"]))
        self.cd_face_y = nn.Parameter(torch.tensor(defaults["cd_face_y"]))
        self.cd_ground = nn.Parameter(torch.tensor(defaults["cd_ground"]))
        self.ce = nn.Parameter(torch.tensor(defaults["ce"]))
        self.wall_nudge = nn.Parameter(torch.tensor(defaults["wall_nudge"]))
        self.wall_nudge_v = nn.Parameter(torch.tensor(defaults["wall_nudge_v"]))
        self.nudge_u_peak = nn.Parameter(torch.tensor(defaults["nudge_u_peak"]))
        self.nudge_u_upper = nn.Parameter(torch.tensor(defaults["nudge_u_upper"]))
        self.nudge_v_upper = nn.Parameter(torch.tensor(defaults["nudge_v_upper"]))
        self.excess_cd = nn.Parameter(torch.tensor(defaults["excess_cd"]))
        # 冻结构造时的初值，供 L2 正则与训练报告使用（不可随优化漂移）
        self._init_values = {n: float(v) for n, v in defaults.items()}

    def initial_dict(self) -> dict:
        return dict(self._init_values)

    def __repr__(self) -> str:
        return "\n".join(f"  {n}: {p.item():.4f}" for n, p in self.named_parameters())
