"""设备抽象层测试：选择、回退、信息、autocast。"""
import contextlib

import pytest
import torch

from gpules.device import DeviceError, autocast_ctx, device_info, select_device


def test_cpu_explicit():
    assert select_device("cpu").type == "cpu"


def test_auto_falls_back_to_known():
    assert select_device("auto").type in ("cuda", "mps", "cpu")


def test_device_info_has_name():
    info = device_info(select_device("cpu"))
    assert "name" in info
    assert info["type"] == "cpu"


def test_autocast_disabled_is_nullcontext():
    ctx = autocast_ctx(select_device("cpu"), enabled=False)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_enabled_cpu_is_bf16():
    # CPU 上 autocast 仅支持 bfloat16；这里仅验证不抛异常
    with autocast_ctx(torch.device("cpu"), enabled=True) as ctx:
        assert ctx is not None


def test_invalid_pref_raises():
    with pytest.raises(DeviceError):
        select_device("nonsense")


def test_strict_cuda_raises_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA 可用，跳过 strict 失败路径")
    with pytest.raises(DeviceError):
        select_device("cuda", strict=True)
