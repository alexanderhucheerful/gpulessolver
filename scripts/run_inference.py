#!/usr/bin/env python3
"""推理便捷脚本。等价于 ``python -m gpules inference ...``。

用法：
  python scripts/run_inference.py --synthetic --nx 64 --ny 64 --nz 32 --end-time 30
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gpules.cli import main

if __name__ == "__main__":
    sys.argv = ["gpules", "inference"] + sys.argv[1:]
    raise SystemExit(main())
