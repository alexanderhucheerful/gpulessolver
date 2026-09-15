"""GPULESConfig 单元测试：默认值、校验、YAML 往返、路径解析。"""
import pytest

from gpules.config import GPULESConfig
from gpules.exceptions import ConfigError


def test_defaults_valid():
    cfg = GPULESConfig()
    assert cfg.mode == "inference"
    # Coriolis 参数应由纬度推导为非零
    assert abs(cfg.f_cor) > 0.0


def test_invalid_mode_raises():
    with pytest.raises(ConfigError):
        GPULESConfig(mode="bogus")


def test_invalid_advection_raises():
    with pytest.raises(ConfigError):
        GPULESConfig(advection="foo")


def test_invalid_pressure_raises():
    with pytest.raises(ConfigError):
        GPULESConfig(pressure_solver="bogus")


def test_invalid_time_integrator_raises():
    with pytest.raises(ConfigError):
        GPULESConfig(time_integrator="bogus")


def test_yaml_roundtrip(tmp_path):
    cfg = GPULESConfig(nx=10, ny=12, nz=8, mode="train", seed=7)
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    cfg2 = GPULESConfig.from_yaml(str(p))
    assert cfg2.nx == 10
    assert cfg2.ny == 12
    assert cfg2.nz == 8
    assert cfg2.mode == "train"
    assert cfg2.seed == 7


def test_resolve_paths(tmp_path):
    cfg = GPULESConfig()
    cfg.resolve_paths(str(tmp_path))
    assert cfg.output.endswith("wind_data_gpu_3d.json")
    assert cfg.data_dir.endswith("data")
