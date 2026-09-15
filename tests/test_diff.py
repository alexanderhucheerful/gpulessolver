"""可微线测试：可学习参数数量、梯度连通性、损失可反向传播。"""
import torch

from gpules.config import GPULESConfig
from gpules.core.diff_helpers import fused_step_diff
from gpules.core.grid import Grid
from gpules.core.solver import LESSolver
from gpules.diff.loss import compute_loss
from gpules.diff.params import LearnablePhysicsParams
from gpules.io.data import generate_synthetic_init


def _train_cfg(**kw):
    base = dict(
        mode="train", nx=32, ny=32, nz=16, dx=10.0, dy=10.0, dz=10.0,
        dt_init=0.05, device="cpu", train_window=4,
        # 参考层索引需落在 nz=16 内
        palm_ref_k=[1, 3, 5, 7],
        palm_ref_u=[0.5, 1.0, 1.5, 2.0],
        palm_ref_v=[0.3, 0.6, 0.9, 1.2],
    )
    base.update(kw)
    return GPULESConfig(**base)


def test_learnable_params_count_and_requires_grad():
    p = LearnablePhysicsParams()
    params = list(p.parameters())
    assert len(params) == 10
    for pr in params:
        assert pr.requires_grad


def test_train_window_finite_and_grad_connected():
    cfg = _train_cfg()
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)

    u = solver.state.u.clone().requires_grad_(True)
    v = solver.state.v.clone().requires_grad_(True)
    w = solver.state.w.clone()
    pt = solver.state.pt.clone()
    tke = solver.state.tke.clone().requires_grad_(True)

    for _ in range(cfg.train_window):
        u, v, w, pt, tke = fused_step_diff(
            u, v, w, pt, tke, cfg.dt_init, solver.params, solver.aux, cfg, solver.grid, solver.pressure
        )

    assert torch.isfinite(u).all()
    assert torch.isfinite(tke).all()

    loss, loss_data, loss_reg = compute_loss(u, v, solver.params, cfg)
    # u/v 在循环中被重赋值成为非叶子张量，retain_grad 才能在 backward 后查看其梯度
    u.retain_grad()
    v.retain_grad()
    loss.backward()

    assert u.grad is not None
    assert v.grad is not None
    # 至少部分可学习参数应获得非零梯度（证明整条可微链连通）
    nz = sum(
        1
        for pr in solver.params.parameters()
        if pr.grad is not None and bool((pr.grad.abs() > 0).any())
    )
    assert nz >= 1
    assert float(loss_data.detach()) >= 0.0
    assert float(loss_reg.detach()) >= 0.0


def test_l2_reg_nonzero_after_param_drift():
    cfg = _train_cfg()
    grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
    init = generate_synthetic_init(grid, cfg)
    solver = LESSolver(cfg).prepare(init)
    with torch.no_grad():
        solver.params.cd_face_x.add_(0.5)
    u = solver.state.u
    v = solver.state.v
    loss, _, loss_reg = compute_loss(u, v, solver.params, cfg)
    assert float(loss_reg) > 0.0
    assert solver.params.initial_dict()["cd_face_x"] != float(solver.params.cd_face_x)
