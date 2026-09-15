"""可微数值稳定工具（来自 diff-v1）。

原 v9 在速度上限/下限截断处使用 ``torch.clamp`` / ``sign`` / 原地 ``where``，
在开启 autograd 时会断梯度。这里用光滑可微替代：

  * ``soft_sign``       tanh 近似 sign
  * ``soft_clamp_min``  softplus 近似下限钳位
  * ``soft_gate``       sigmoid 门控（异号检测）
  * ``soft_velocity_cap`` 可微速度上限（power-law）
  * ``safe_nan_to_num``   isfinite + where（异常处梯度归零）
  * ``safe_abs`` / ``safe_sqrt`` 可微绝对值/平方根
  * ``replace_slice``   非原地切片替换 + 掩模缓存（减少 autograd 重分配）
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..config import GPULESConfig

# ── 掩模缓存（避免每次 replace_slice 重新分配张量） ──────────────────
_SLICE_MASK_CACHE: dict = {}


def clear_mask_cache() -> None:
    _SLICE_MASK_CACHE.clear()


class MaskCache:
    """可实例化的切片掩模缓存（便于测试隔离，替代模块级全局）。"""

    def __init__(self):
        self._cache: dict = {}

    def get(self, dim: int, idx: int, size: int, device: torch.device) -> torch.Tensor:
        key = (dim, idx, size)
        if key not in self._cache:
            m = torch.ones(size, device=device)
            m[idx] = 0.0
            self._cache[key] = m
        return self._cache[key]


def soft_sign(x: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    return torch.tanh(x / eps)


def soft_clamp_min(x: torch.Tensor, min_val: float, sharpness: float = 8.0) -> torch.Tensor:
    return min_val + F.softplus((x - min_val) * sharpness) / sharpness


def soft_gate(score: torch.Tensor, sharpness: float = 50.0) -> torch.Tensor:
    return torch.sigmoid(score * sharpness)


def soft_velocity_cap(u, v, w, v_max: float = 20.0, k: float = 8.0):
    spd = torch.sqrt(u * u + v * v + w * w + 1e-6)
    scale = 1.0 / (1.0 + (spd / v_max) ** k) ** (1.0 / k)
    return u * scale, v * scale, w * scale


def safe_nan_to_num(x: torch.Tensor, v_max: float = 20.0) -> torch.Tensor:
    finite_mask = torch.isfinite(x)
    return torch.where(finite_mask, x, torch.zeros_like(x))


def safe_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0) + eps)


def replace_slice(field: torch.Tensor, dim: int, idx: int, value: torch.Tensor,
                  cache: Optional[dict] = None) -> torch.Tensor:
    """非原地替换 3D 张量的一个切片（可微）。

    dim: 0=z, 1=y, 2=x；idx: 切片索引；value: 与切片同形的替换值。
    """
    c = cache if cache is not None else _SLICE_MASK_CACHE
    size = field.shape[dim]
    key = (dim, idx, size)
    if key not in c:
        m = torch.ones(size, device=field.device)
        m[idx] = 0.0
        c[key] = m
    mask_1d = c[key]
    shape = [1, 1, 1]
    shape[dim] = size
    mask = mask_1d.view(*shape)
    val_expanded = value.unsqueeze(dim)
    return field * mask + val_expanded * (1.0 - mask)


def get_soft_params(cfg: GPULESConfig):
    """把配置中的可微近似参数打包成字典（便于传给各函数）。"""
    return {
        "soft_sign_eps": cfg.soft_sign_eps,
        "gate_sharpness": cfg.gate_sharpness,
        "softplus_sharpness": cfg.softplus_sharpness,
        "vel_cap_k": cfg.vel_cap_k,
    }
