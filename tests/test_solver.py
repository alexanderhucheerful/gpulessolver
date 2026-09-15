"""求解器冒烟测试：小规模推理运行 + 输出 JSON 写出。"""
import torch

from gpules.config import GPULESConfig
from gpules.core.grid import Grid
from gpules.core.operators import divergence
from gpules.core.solver import LESSolver
from gpules.io.data import generate_synthetic_init
from gpules.io.output import write_output_json


def _cfg(**kw):
    base = dict(
        nx=32, ny=32, nz=16, dx=10.0, dy=10.0, dz=10.0,
        end_time=0.5, dt_init=0.05, device="cpu",
    )
    base.update(kw)
    return GPULESConfig(**base)


def test_inference_small_run(tmp_path):
    cfg = _cfg()
    cfg.resolve_paths(str(tmp_path))
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    log = solver.run(end_time=cfg.end_time, record_every=1)
    assert solver.state is not None
    solver.state.check_finite("test")
    stats = solver.state.statistics()
    assert "spd_max" in stats
    assert len(log) >= 1

    out = tmp_path / "wind.json"
    write_output_json(solver.state, cfg, str(out), diagnostics=True)
    assert out.exists()
    assert out.stat().st_size > 0


def test_divergence_free_init_reduces_divergence():
    cfg = _cfg()
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    div0 = float(divergence(solver.state.u, solver.state.v, solver.state.w, grid).abs().max())
    solver.divergence_free_init()
    div1 = float(divergence(solver.state.u, solver.state.v, solver.state.w, grid).abs().max())
    if div0 > 1e-6:
        assert div1 < div0
    else:
        assert div1 < 1e-2


def test_inference_reproducible_with_seed():
    cfg = _cfg(seed=123, end_time=0.3)
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)

    def run_once():
        s = LESSolver(cfg).prepare(init)
        s.run(end_time=cfg.end_time, record_every=1000)
        return float(s.state.u.mean())

    assert run_once() == run_once()


def test_inference_with_poisson_fft(tmp_path):
    cfg = _cfg(pressure_solver="poisson_fft")
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    solver.run(end_time=cfg.end_time)
    solver.state.check_finite("poisson_fft")
    assert torch.isfinite(solver.state.u).all()
