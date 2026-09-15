"""gpules 版本信息。

gpules = GPU Large-Eddy Simulation. 本项目合并了两条研究版本线:
  * desktop ``gpu_les_v5..v9``  —— 高精度数值线 (5 阶 WS 平流 + 多网格压力 + SSP-RK3)
  * D 盘 ``gpu_les_solver_diff`` —— 可微/可学习求解器线 (diff-v1)

v1.0 将两者收敛为一个可配置、可测试、可交付的统一包。
"""

__version__ = "1.0.0"

# 导出合并来源，便于审计与回溯
LINEAGE = {
    "high_fidelity": "gpu_les_v9 (5th-order Wicker-Skamarock, multigrid pressure, SSP-RK3, Deardorff TKE SGS)",
    "differentiable": "gpu_les_solver_diff (diff-v1, soft-* stable fns, LearnablePhysicsParams, RFFT Poisson)",
}
