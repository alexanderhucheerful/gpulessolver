"""压力泊松求解器测试：四种策略的形状/有限性，以及谱求解的散度消除能力。"""
import torch

from gpules.config import GPULESConfig
from gpules.core.grid import Grid
from gpules.core.operators import divergence
from gpules.core.pressure import (
    MultigridSolver,
    PoissonFFTSolver,
    build_pressure_solver,
)


def _grid():
    return Grid(32, 32, 32, 5.0, 5.0, 5.0, torch.device("cpu"))


def test_all_solvers_shape_and_finite():
    g = _grid()
    for kind in ("multigrid", "cg", "jacobi", "poisson_fft"):
        cfg = GPULESConfig(pressure_solver=kind)
        s = build_pressure_solver(cfg)
        if isinstance(s, PoissonFFTSolver):
            s.prepare(g)
        rhs = torch.randn(g.shape) * 0.01
        p = s.solve(rhs, torch.zeros_like(rhs), g)
        assert p.shape == g.shape
        assert torch.isfinite(p).all()


def test_poisson_projection_reduces_divergence():
    g = _grid()
    s = PoissonFFTSolver()
    s.prepare(g)
    torch.manual_seed(0)
    # 用平滑（低频）场：谱算子与有限差分算子在此高度一致，投影才会显著消除散度
    z, y, x = torch.meshgrid(
        torch.arange(g.nz), torch.arange(g.ny), torch.arange(g.nx), indexing="ij"
    )
    Lx, Ly, Lz = g.nx * g.dx, g.ny * g.dy, g.nz * g.dz
    u = torch.sin(2 * torch.pi * x * g.dx / Lx) * torch.cos(2 * torch.pi * y * g.dy / Ly)
    v = torch.cos(2 * torch.pi * x * g.dx / Lx) * torch.sin(2 * torch.pi * z * g.dz / Lz)
    w = torch.sin(2 * torch.pi * y * g.dy / Ly) * torch.cos(2 * torch.pi * z * g.dz / Lz)
    dt = 0.05
    div0 = divergence(u, v, w, g).abs().max()

    p = s.solve(divergence(u, v, w, g) / dt, torch.zeros_like(u), g)
    dpx = (torch.roll(p, -1, 2) - torch.roll(p, 1, 2)) / (2 * g.dx)
    dpy = (torch.roll(p, -1, 1) - torch.roll(p, 1, 1)) / (2 * g.dy)
    dpz = (torch.roll(p, -1, 0) - torch.roll(p, 1, 0)) / (2 * g.dz)
    u2 = u - dt * dpx
    v2 = v - dt * dpy
    w2 = w - dt * dpz
    div1 = divergence(u2, v2, w2, g).abs().max()

    # 周期域下谱投影应显著消除散度
    assert div1 < div0 * 0.3 + 1e-4


def test_multigrid_is_instance():
    cfg = GPULESConfig(pressure_solver="multigrid")
    assert isinstance(build_pressure_solver(cfg), MultigridSolver)
