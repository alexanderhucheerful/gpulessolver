"""平流算子（可插拔策略）。

原 v9 仅实现 5 阶 WS 平流；diff-v1 仅实现 2 阶中心+Jameson。合并后两者皆可选用：

  * ``WickerSkamarock5`` — 高精度、低耗散（inference 默认）
  * ``Central2``        — 2 阶中心差分 + Jameson 人工耗散（train 默认，全程可微）

两者都暴露统一接口 ``advect(field, u, v, w, grid) -> tendency``，
返回值均为平流**趋势项** ``-u·∇f``（Central2 另含 Jameson 人工耗散），
可直接累加入 du/dt。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from ..config import AdvectionKind
from .grid import Grid


class Advection(ABC):
    kind: AdvectionKind

    @abstractmethod
    def advect(self, f: torch.Tensor, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
               grid: Grid) -> torch.Tensor:
        ...


class WickerSkamarock5(Advection):
    """5 阶 Wicker-Skamarock 迎风偏置平流（可微）。"""

    kind: AdvectionKind = "ws5"

    def advect(self, f, u, v, w, grid):
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        # x 方向
        p = F.pad(f.unsqueeze(0).unsqueeze(0), (3, 3, 0, 0, 0, 0), mode="replicate").squeeze(0).squeeze(0)
        fu_pos = (u >= 0).float()
        dfx_pos = (-2 * p[:, :, 0:-6] + 15 * p[:, :, 1:-5] - 60 * p[:, :, 2:-4]
                   + 20 * p[:, :, 3:-3] + 30 * p[:, :, 4:-2] - 3 * p[:, :, 5:-1]) / (60 * dx)
        dfx_neg = (3 * p[:, :, 1:-5] - 30 * p[:, :, 2:-4] - 20 * p[:, :, 3:-3]
                   + 60 * p[:, :, 4:-2] - 15 * p[:, :, 5:-1] + 2 * p[:, :, 6:]) / (60 * dx)
        dfx = dfx_pos * fu_pos + dfx_neg * (1 - fu_pos)
        # y 方向
        p = F.pad(f.unsqueeze(0).unsqueeze(0), (0, 0, 3, 3, 0, 0), mode="replicate").squeeze(0).squeeze(0)
        fv_pos = (v >= 0).float()
        dfy_pos = (-2 * p[:, 0:-6, :] + 15 * p[:, 1:-5, :] - 60 * p[:, 2:-4, :]
                   + 20 * p[:, 3:-3, :] + 30 * p[:, 4:-2, :] - 3 * p[:, 5:-1, :]) / (60 * dy)
        dfy_neg = (3 * p[:, 1:-5, :] - 30 * p[:, 2:-4, :] - 20 * p[:, 3:-3, :]
                   + 60 * p[:, 4:-2, :] - 15 * p[:, 5:-1, :] + 2 * p[:, 6:, :]) / (60 * dy)
        dfy = dfy_pos * fv_pos + dfy_neg * (1 - fv_pos)
        # z 方向
        p = F.pad(f.unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 3, 3), mode="replicate").squeeze(0).squeeze(0)
        fw_pos = (w >= 0).float()
        dfz_pos = (-2 * p[0:-6, :, :] + 15 * p[1:-5, :, :] - 60 * p[2:-4, :, :]
                   + 20 * p[3:-3, :, :] + 30 * p[4:-2, :, :] - 3 * p[5:-1, :, :]) / (60 * dz)
        dfz_neg = (3 * p[1:-5, :, :] - 30 * p[2:-4, :, :] - 20 * p[3:-3, :, :]
                   + 60 * p[4:-2, :, :] - 15 * p[5:-1, :, :] + 2 * p[6:, :, :]) / (60 * dz)
        dfz = dfz_pos * fw_pos + dfz_neg * (1 - fw_pos)
        return -(u * dfx + v * dfy + w * dfz)


class Central2(Advection):
    """2 阶中心差分平流 + Jameson 人工耗散（可微，来自 diff-v1）。"""

    kind: AdvectionKind = "central2"

    def __init__(self, ad_k2: float = 0.5, ad_k4: float = 1.0 / 32.0):
        self.ad_k2 = ad_k2
        self.ad_k4 = ad_k4

    def advect(self, f, u, v, w, grid):
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        df_dx = (torch.roll(f, -1, 2) - torch.roll(f, 1, 2)) / (2 * dx)
        df_dy = (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2 * dy)
        df_dz = (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2 * dz)
        adv = -(u * df_dx + v * df_dy + w * df_dz)
        return adv + self._jameson(f, u, v, w, grid)

    def _jameson(self, f, u, v, w, grid):
        """Jameson 人工耗散（2 阶 + 4 阶通量差分形式，可微）。"""
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        eps = 1e-20

        def _dir(dim: int, d: float) -> torch.Tensor:
            fp = torch.roll(f, -1, dim)
            fm = torch.roll(f, 1, dim)
            d2 = (fp - 2 * f + fm) / (d * d)
            nu = torch.abs(d2) / (torch.abs(f) + eps + torch.abs(fp) + torch.abs(fm))
            diss2 = self.ad_k2 * (
                nu * (fp - f) - torch.roll(nu, 1, dim) * (f - fm)
            ) / d
            d4 = torch.roll(f, -2, dim) - 3 * fp + 3 * f - fm
            diss4 = self.ad_k4 * (d4 - torch.roll(d4, 1, dim)) / d
            return diss2 + diss4

        return _dir(2, dx) + _dir(1, dy) + _dir(0, dz)


def build_advection(cfg) -> Advection:
    if cfg.advection == "ws5":
        return WickerSkamarock5()
    elif cfg.advection == "central2":
        return Central2(ad_k2=cfg.ad_k2, ad_k4=cfg.ad_k4)
    raise ValueError(f"未知 advection: {cfg.advection}")
