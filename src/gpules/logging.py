"""结构化日志工具。

替代原项目散落的 ``print(...)``，提供统一的 logger、可配置级别、
以及 ``[步骤/阶段]`` 风格的进度输出。可通过 ``setup_logging`` 在入口处初始化。
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_CONFIGURED = False


def setup_logging(level: str | int = "INFO", *, stream=None) -> logging.Logger:
    """配置根 logger 并返回 gpules 命名 logger。

    Args:
        level: 日志级别字符串 ("DEBUG"/"INFO"/...) 或 int。
        stream: 输出流，默认 ``sys.stdout``。
    """
    global _CONFIGURED
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("gpules")
    # 避免重复添加 handler
    if not _CONFIGURED:
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(level if isinstance(level, int) else logging.getLevelName(level.upper()))
    root.propagate = False
    return root


def get_logger(name: str = "") -> logging.Logger:
    """获取子 logger（如 ``gpules.core.solver``）。"""
    return logging.getLogger("gpules" if not name else f"gpules.{name}")
