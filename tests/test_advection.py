"""平流算子测试：两种策略的形状/有限性，以及常数场平流趋于零。"""
import torch

from gpules.config import GPULESConfig
from gpules.core.advection import Central2, WickerSkamarock5, build_advection
from gpules.core.grid import Grid


def _grid():
    return Grid(32, 32, 32, 5.0, 5.0, 5.0, torch.device("cpu"))


def test_ws5_zero_on_constant():
    g = _grid()
    f = torch.ones(g.shape)
    u = torch.ones(g.shape)
    v = torch.ones(g.shape)
    w = torch.ones(g.shape)
    out = WickerSkamarock5().advect(f, u, v, w, g)
    assert out.shape == g.shape
    assert torch.isfinite(out).all()
    assert out[3:-3, 3:-3, 3:-3].abs().max() < 1e-3


def test_central2_runs_and_finite():
    g = _grid()
    f = torch.randn(g.shape)
    out = Central2().advect(f, torch.ones(g.shape), torch.zeros(g.shape), torch.zeros(g.shape), g)
    assert out.shape == g.shape
    assert torch.isfinite(out).all()


def test_build_advection_dispatch():
    assert isinstance(build_advection(GPULESConfig(advection="ws5")), WickerSkamarock5)
    assert isinstance(build_advection(GPULESConfig(advection="central2")), Central2)


def test_ws5_and_central2_same_sign_on_ramp():
    """两种平流策略均返回趋势项 -u·∇f，符号应一致。"""
    g = _grid()
    z, y, x = torch.meshgrid(
        torch.arange(g.nz), torch.arange(g.ny), torch.arange(g.nx), indexing="ij"
    )
    f = (x.float() * g.dx + y.float() * g.dy) * 0.01
    u = torch.full(g.shape, 3.0)
    v = torch.zeros(g.shape)
    w = torch.zeros(g.shape)
    ws = WickerSkamarock5().advect(f, u, v, w, g)
    c2 = Central2(ad_k2=0.0, ad_k4=0.0).advect(f, u, v, w, g)
    inner = 3, 3, 3
    assert ws[inner] * c2[inner] > 0
    assert ws[inner].abs() > 1e-6
