#!/usr/bin/env python3
"""可微训练便捷脚本。等价于 ``python -m gpules train ...``。

用法：
  python scripts/train.py --synthetic --nx 32 --ny 32 --nz 24 \
      --train-window 8 --train-epochs 3
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gpules.cli import main

if __name__ == "__main__":
    sys.argv = ["gpules", "train"] + sys.argv[1:]
    raise SystemExit(main())
