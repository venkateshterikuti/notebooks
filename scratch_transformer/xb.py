"""Backend selector: NumPy (CPU) or CuPy (CUDA) via env var SCRATCH_DEVICE.
Usage:
  export SCRATCH_DEVICE=cuda  # if CuPy + CUDA available
  export SCRATCH_DEVICE=cpu   # default
"""
import os
_device = os.getenv("SCRATCH_DEVICE", "cpu").lower()
try:
    if _device == "cuda":
        import cupy as np
    else:
        import numpy as np
except Exception:
    import numpy as np  # graceful fallback

# utilities to move to/from CPU when needed
_def_name = getattr(np, "__name__", "numpy")

def to_cpu(x):
    if _def_name.startswith("cupy"):
        import cupy
        return cupy.asnumpy(x)
    return x