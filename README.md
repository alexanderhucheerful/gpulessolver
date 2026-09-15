# gpules v1.0 — GPU 城市风场大涡模拟-适用于AI气象大模型降尺度风场模拟

> 基于 PyTorch 的大涡模拟（LES）求解器，用于预测建筑群周边的三维风场。

> 5 阶 Wicker–Skamarock 平流 + 几何多网格压力求解 + SSP-RK3 时间积分 + Deardorff TKE SGS（含虚位温浮力）。
> 2 阶中心差分 + Jameson 人工耗散 + RFFT 谱压力求解 + Heun 积分 + **10 个可学习物理参数**（在 PALM 参考场监督下用 Adam 自动调参）。

---

## 1. 特性

| 特性 | 说明 |
| --- | --- |
| 双模式 | `inference`（高精度预测）/ `train`（可微参数学习），由单一配置切换 |
| 可插拔数值策略 | 平流 `ws5`/`central2`（统一返回趋势项）；压力 `multigrid`/`cg`/`jacobi`/`poisson_fft`（谱求解为周期 surrogate）；积分 `ssprk3`/`heun` |
| 设备自适应 | `cuda → mps → cpu` 自动回退，不再硬编码 `device='cuda'` |
| 数值健壮性 | NaN/Inf 全程校验、散度超限折半、CFL 自适应步长、CUDA OOM 守卫 |
| 可微物理 | `soft_sign`/`soft_clamp_min`/`soft_gate`/`safe_nan_to_num` 等光滑替代，保证梯度连通 |
| 多源输入 | 合成大气 / PIDS netCDF / FuXi·GraphCast AI 预报（气压→高度、T→位温） |
| 工程化 | 配置即数据（dataclass + YAML + CLI）、结构化日志、自定义异常体系、CLI 入口 |
| 测试与 CI | pytest 单测 + ruff 静态检查 + GitHub Actions 多版本矩阵 |

---

## 2. 安装

```bash
# 方式一：从源码安装（推荐，含 CLI 命令 gpules）
cd gpulesv1.0
pip install -e .

# 方式二：仅安装运行依赖，用 python -m gpules 运行
pip install torch numpy scipy pyyaml netCDF4
```

> CPU 环境（如 CI / 笔记本）安装 CPU 版 torch：
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 3. 快速开始

### 3.1 命令行（CLI）

```bash
# 推理：合成数据快速验证（演示网格 48×48×24）
python -m gpules inference --config configs/default.yaml --synthetic

# 推理：自定义小网格
python -m gpules inference --synthetic --device cpu --nx 64 --ny 64 --nz 32 --end-time 30

# 可微训练
python -m gpules train --synthetic --nx 32 --ny 32 --nz 24 \
    --train-window 8 --train-epochs 3 --learning-rate 0.002

# 真实数据（PIDS）
python -m gpules inference --static-path data/PIDS_STATIC --dynamic-path data/PIDS_DYNAMIC
```

输出：`wind_data_gpu_3d.json`（降采样三维风场 + 切变/高风区 + 诊断）。

### 3.2 Python API

```python
import torch
from gpules.config import GPULESConfig
from gpules.core.grid import Grid
from gpules.core.solver import LESSolver
from gpules.io.data import generate_synthetic_init
from gpules.io.output import write_output_json

cfg = GPULESConfig(nx=48, ny=48, nz=24, dx=10.0, dy=10.0, dz=10.0,
                   end_time=20.0, dt_init=0.1, device="cpu", mode="inference")
grid = Grid(cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, torch.device("cpu"))
init = generate_synthetic_init(grid, cfg)
solver = LESSolver(cfg).prepare(init)
solver.run(end_time=cfg.end_time)
write_output_json(solver.state, cfg, "wind_data_gpu_3d.json", diagnostics=True)
```

更完整的示例见 [`examples/quickstart.py`](examples/quickstart.py)（含推理 + 可微训练两段演示）。

---

## 4. 配置

所有物理/数值参数集中在 `GPULESConfig`（`src/gpules/config.py`），支持：

- **YAML 加载/保存**：`GPULESConfig.from_yaml()` / `to_yaml()`
- **CLI 覆盖**：任意字段均可用 `--<field>` 覆盖（如 `--nx 64 --end-time 30`）
- **校验**：`validate()` 在入口即拦截非法配置（如 `mode`、网格尺寸、参考层索引越界警告）

`configs/default.yaml` 是「演示/快速验证」规模；生产规模把 `nx/ny/nz` 放大到 `200/200/100`、`dx=5.0` 即可。

### 关键参数

| 分组 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| 网格 | `nx,ny,nz,dx,dy,dz` | 200/200/100, 5 | 网格尺寸与分辨率 |
| 模式 | `mode` | `inference` | `inference` / `train` |
| 平流 | `advection` | `ws5` | `ws5`（高精度）/ `central2`（可微） |
| 压力 | `pressure_solver` | `multigrid` | `multigrid`/`cg`/`jacobi`/`poisson_fft` |
| 积分 | `time_integrator` | `ssprk3` | `ssprk3`（推理）/ `heun`（训练内部） |
| SGS | `ce,ck,cnu,nu_min,tke_min` | 0.845… | Deardorff TKE 闭合常数 |
| 训练 | `train_window, train_epochs, learning_rate, l2_reg` | 10/5/0.002/0.001 | 可微训练超参 |
| 参考 | `palm_ref_u/v/k` | 4 层 | PALM 监督层均值目标（越界层自动跳过） |

---

## 5. 架构

```
gpules/
├── __init__.py          公共 API 导出
├── __main__.py          python -m gpules 入口
├── cli.py               命令行解析与编排
├── config.py            GPULESConfig（单一事实来源）
├── version.py           版本与血统记录
├── exceptions.py        自定义异常层级
├── logging.py           结构化日志
├── device.py            设备选择与回退
├── core/
│   ├── grid.py          Grid 网格容器
│   ├── state.py         FlowState 流场状态容器（含统计量/有限性校验）
│   ├── operators.py     可微有限差分算子（散度/拉普拉斯/梯度）
│   ├── advection.py     平流策略（ws5 / central2）
│   ├── pressure.py      压力求解策略（multigrid / cg / jacobi / poisson_fft）
│   ├── sgs.py           Deardorff TKE SGS（hf / diff 两版）
│   ├── boundary.py      建筑强迫与边界条件（HF / 可微）
│   ├── diff_helpers.py  可微核心物理步骤（纯函数）
│   ├── tendencies.py    高精度显式趋势项
│   ├── timestepping.py  时间积分（SSP-RK3）
│   └── solver.py        LESSolver 统一编排器
├── diff/
│   ├── stable.py        可微数值稳定工具（soft_*/safe_*）
│   ├── params.py        LearnablePhysicsParams（10 个可学习参数）
│   ├── loss.py          可微训练损失
│   └── trainer.py       Trainer（Adam + 混合精度 + 检查点 + 梯度裁剪）
└── io/
    ├── data.py          初始化数据加载（合成/PIDS/AI）
    └── output.py        结果写出（风场 JSON / 对比数据）
```

设计原则：**参数即配置、配置即文档**；模块通过显式参数传递而非全局副作用通信；所有张量经有限性校验；`import` 不触发任何模拟或训练副作用。

---

## 6. 测试与 CI

```bash
pytest -v          # 单测：配置/设备/算子/平流/压力/求解器/可微链/回归
ruff check src tests   # 静态检查
```

CI（`.github/workflows/ci.yml`）在 Python 3.10/3.11/3.12 上自动运行 pytest + ruff，并构建 wheel 产物。

---

## 7. 许可证

MIT —— 见 [LICENSE](LICENSE)。
