"""流场状态容器。

原项目把 ``u/v/w/pt/qv/tke/p`` 作为一堆散落的全局张量互相传递，函数靠全局副作用通信。
这里用 :class:`FlowState` 把它们和网格绑在一起，作为求解器的唯一状态对象：

  * 字段就地张量（支持可导模式 requires_grad）
  * 提供 ``clone`` / ``to`` / ``requires_grad_`` / ``check_finite`` 等标准操作
  * 提供 ``statistics`` 用于诊断与测试断言
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..exceptions import NaNError
from .grid import Grid


class FlowState:
    def __init__(
        self,
        grid: Grid,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        pt: torch.Tensor,
        qv: Optional[torch.Tensor],
        tke: torch.Tensor,
        p: torch.Tensor,
    ):
        self.grid = grid
        self.u = u
        self.v = v
        self.w = w
        self.pt = pt
        self.qv = qv
        self.tke = tke
        self.p = p

    # ── 构造 ───────────────────────────────────────────────────────
    @classmethod
    def zeros(cls, grid: Grid, *, with_qv: bool = True) -> "FlowState":
        shape = grid.shape
        dev = grid.device
        qv = torch.zeros(shape, device=dev, dtype=torch.float32) if with_qv else None
        return cls(
            grid,
            torch.zeros(shape, device=dev, dtype=torch.float32),
            torch.zeros(shape, device=dev, dtype=torch.float32),
            torch.zeros(shape, device=dev, dtype=torch.float32),
            torch.zeros(shape, device=dev, dtype=torch.float32),
            qv,
            torch.full(shape, 1e-4, device=dev, dtype=torch.float32),
            torch.zeros(shape, device=dev, dtype=torch.float32),
        )

    @classmethod
    def from_numpy(cls, grid: Grid, arrays: dict) -> "FlowState":
        """从 numpy 数组构造（u/v/w/pt/qv/tke/p 均可选，缺省补零）。"""
        shape = grid.shape
        dev = grid.device

        def _to(name, fill=0.0):
            if name in arrays and arrays[name] is not None:
                return torch.as_tensor(np.asarray(arrays[name], dtype=np.float32), device=dev)
            return torch.full(shape, fill, device=dev, dtype=torch.float32)

        qv = _to("qv") if ("qv" in arrays and arrays["qv"] is not None) else None
        return cls(
            grid,
            _to("u"), _to("v"), _to("w"), _to("pt"),
            qv, _to("tke", 1e-4), _to("p"),
        )

    # ── 标准操作 ───────────────────────────────────────────────────
    def clone(self) -> "FlowState":
        return FlowState(
            self.grid,
            self.u.clone(), self.v.clone(), self.w.clone(), self.pt.clone(),
            self.qv.clone() if self.qv is not None else None,
            self.tke.clone(), self.p.clone(),
        )

    def to(self, device: torch.device) -> "FlowState":
        self.grid = self.grid.to(device)
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.w = self.w.to(device)
        self.pt = self.pt.to(device)
        if self.qv is not None:
            self.qv = self.qv.to(device)
        self.tke = self.tke.to(device)
        self.p = self.p.to(device)
        return self

    def requires_grad_(self, flag: bool = True) -> "FlowState":
        self.u.requires_grad_(flag)
        self.v.requires_grad_(flag)
        self.w.requires_grad_(flag)
        return self

    def zero_grad_(self) -> "FlowState":
        for t in (self.u, self.v, self.w):
            if t.grad is not None:
                t.grad = None
        return self

    # ── 健壮性 ──────────────────────────────────────────────────────
    def check_finite(self, where: str = "", step: Optional[int] = None) -> None:
        """检测 NaN/Inf；原项目完全没有这一步。"""
        for name in ("u", "v", "w", "pt", "tke", "p"):
            t = getattr(self, name)
            if not torch.isfinite(t).all():
                raise NaNError(f"场 {name} 出现 NaN/Inf", where=where, step=step)

    # ── 诊断 ───────────────────────────────────────────────────────
    def divergence(self) -> torch.Tensor:
        from .operators import divergence
        return divergence(self.u, self.v, self.w, self.grid)

    def statistics(self) -> dict:
        """层均值廓线与极值统计（用于诊断输出与测试断言）。"""
        with torch.no_grad():
            spd = torch.sqrt(self.u ** 2 + self.v ** 2 + self.w ** 2)
            div = self.divergence()
            diag = {
                "u_profile": self.u.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "v_profile": self.v.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "w_profile": self.w.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "pt_profile": self.pt.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "tke_profile": self.tke.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "spd_profile": spd.mean(dim=(1, 2)).cpu().numpy().tolist(),
                "spd_max": float(spd.max()),
                "spd_mean": float(spd.mean()),
                "div_max": float(div.abs().max()),
                "div_mean": float(div.abs().mean()),
            }
            if self.qv is not None:
                diag["qv_profile"] = self.qv.mean(dim=(1, 2)).cpu().numpy().tolist()
        return diag

    def __repr__(self) -> str:
        return f"FlowState(grid={self.grid.shape}, device={self.grid.device})"
