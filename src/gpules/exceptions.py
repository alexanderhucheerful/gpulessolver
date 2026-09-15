"""gpules 自定义异常层级。

将原来的 ``exit(1)`` / 静默崩溃 统一为可被调用方捕获的语义化异常，
是"可交付"与"研究原型"的分界线之一。
"""

from __future__ import annotations


class GpulesError(Exception):
    """所有 gpules 错误的基类。"""


class ConfigError(GpulesError):
    """配置非法或缺失（例如 dx<=0、mode 非法）。"""


class DataLoadError(GpulesError):
    """输入数据（PIDS / AI 预报 / 建筑）缺失或格式无法解析。"""


class SchemaError(DataLoadError):
    """数据变量/维度与预期 schema 不匹配。"""


class DivergenceError(GpulesError):
    """数值发散：散度/速度超出阈值且无法通过折半 dt 恢复。"""

    def __init__(self, message: str, *, div_max: float | None = None, step: int | None = None):
        super().__init__(message)
        self.div_max = div_max
        self.step = step


class NaNError(GpulesError):
    """场中出现 NaN/Inf。"""

    def __init__(self, message: str, *, where: str = "", step: int | None = None):
        super().__init__(message)
        self.where = where
        self.step = step


class DeviceError(GpulesError):
    """无法获取可用的计算设备（cuda/cpu/mps 均不可用）。"""


class OOMError(GpulesError):
    """显存/内存不足（CUDA OOM 或 CPU 分配失败）。"""

    def __init__(self, message: str, *, peak_gb: float | None = None):
        super().__init__(message)
        self.peak_gb = peak_gb
