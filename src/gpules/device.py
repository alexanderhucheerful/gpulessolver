"""设备抽象层。

原项目 ``device = torch.device('cuda')`` 在五处硬编码、无回退，导致无 GPU 环境直接崩溃。
这里提供 ``select_device``：优先满足用户偏好，失败时在 cuda → mps → cpu 之间优雅回退，
并给出可读的设备信息。所有张量创建都应通过 ``config.device`` 落到这里返回的设备。
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .exceptions import DeviceError


def seed_all(seed: int) -> None:
    """统一设置 Python / NumPy / PyTorch 随机种子（可复现推理与训练）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto", *, strict: bool = False) -> torch.device:
    """按偏好选择计算设备，并自动回退。

    Args:
        prefer: "auto" | "cuda" | "gpu" | "mps" | "cpu"。
            - "auto": cuda → mps → cpu
            - 显式指定 cuda/mps 但不可用时：``strict=True`` 抛错，否则回退 cpu。
        strict: 为 True 且用户显式指定 cuda/mps 但不可用时抛 ``DeviceError``。

    Returns:
        选定的 ``torch.device``。
    """
    prefer = (prefer or "auto").lower()
    explicit = prefer in ("cuda", "gpu", "mps")

    def cuda_ok() -> bool:
        return torch.cuda.is_available()

    def mps_ok() -> bool:
        # torch >= 1.12 提供 mps；Apple Silicon 专用
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    order: list[str] = []
    if prefer in ("auto", "cuda", "gpu"):
        if cuda_ok():
            order.append("cuda")
        if mps_ok():
            order.append("mps")
        order.append("cpu")
    elif prefer == "mps":
        if mps_ok():
            order.append("mps")
        order.append("cpu")
    elif prefer == "cpu":
        order.append("cpu")
    else:
        raise DeviceError(f"未知设备偏好: {prefer!r}（可选 auto/cuda/mps/cpu）")

    # 去重保序
    seen = set()
    order = [d for d in order if not (d in seen or seen.add(d))]

    if not order:
        raise DeviceError("无任何可用计算设备")
    chosen = torch.device(order[0])
    if strict and explicit and chosen.type == "cpu" and prefer in ("cuda", "gpu", "mps"):
        raise DeviceError(
            f"请求设备 {prefer!r} 不可用（当前仅有 CPU）。"
            "请安装 GPU 版 PyTorch 或使用 --device cpu / auto。"
        )
    return chosen


def device_info(device: torch.device) -> dict:
    """返回可读的设备信息字典（名称 / 总显存 / 架构）。"""
    info: dict = {"device": str(device), "type": device.type}
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(
            name=torch.cuda.get_device_name(idx),
            total_memory_gb=props.total_memory / 1e9,
            major=props.major,
            minor=props.minor,
        )
    elif device.type == "mps":
        info["name"] = "Apple Metal Performance Shaders"
    else:
        import platform

        info.update(name="CPU", arch=platform.processor() or platform.machine())
    return info


def autocast_ctx(device: torch.device, *, enabled: bool, dtype=None):
    """跨设备安全的 autocast 上下文。

    - cuda: 支持 float16 / bfloat16
    - cpu: 仅支持 bfloat16（float16 在 CPU 上多数算子不支持）
    - mps: 支持 float16 / bfloat16

    ``enabled=False`` 时返回空的 ``contextlib.nullcontext()``。
    """
    import contextlib

    if not enabled:
        return contextlib.nullcontext()

    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    return torch.amp.autocast(device_type=device.type, dtype=dtype)
