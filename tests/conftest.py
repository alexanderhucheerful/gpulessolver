"""pytest 配置：把 src/ 加入 sys.path，使 `import gpules` 在测试时可用（无需先 pip install）。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
