"""有限差分算子测试：散度/拉普拉斯/梯度的正确性与散度-梯度恒等式。"""
import torch

from gpules.core.grid import Grid
from gpules.core.operators import divergence, gradient, laplacian


def _grid():
    return Grid(24, 24, 24, 5.0, 5.0, 5.0, torch.device("cpu"))


def test_divergence_constant_is_zero():
    g = _grid()
    u = torch.ones(g.shape)
    v = torch.ones(g.shape)
    w = torch.ones(g.shape)
    div = divergence(u, v, w, g)
    assert torch.isfinite(div).all()
    # 内部应接近 0（复制填充消除边界差异）
    assert div[1:-1, 1:-1, 1:-1].abs().max() < 1e-4


def test_laplacian_quadratic_field():
    g = _grid()
    z, y, x = torch.meshgrid(
        torch.arange(g.nz), torch.arange(g.ny), torch.arange(g.nx), indexing="ij"
    )
    # 物理坐标下的 f = (x·dx)² + (y·dy)²，离散拉普拉斯内部应等于 2 + 2 = 4
    f = (x.float() * g.dx) ** 2 + (y.float() * g.dy) ** 2
    lap = laplacian(f, g)
    c = (g.nz // 2, g.ny // 2, g.nx // 2)
    assert abs(lap[c] - 4.0) < 1e-6


def test_divergence_of_gradient_equals_laplacian():
    g = _grid()
    z, y, x = torch.meshgrid(
        torch.arange(g.nz), torch.arange(g.ny), torch.arange(g.nx), indexing="ij"
    )
    p = x.float() ** 3 + y.float() ** 2 + 2.0 * z.float()
    gx, gy, gz = gradient(p, g)
    div = divergence(gx, gy, gz, g)
    lap = laplacian(p, g)
    assert torch.allclose(div[2:-2, 2:-2, 2:-2], lap[2:-2, 2:-2, 2:-2], atol=1e-4)
