"""Make pip-installed NVIDIA runtime libraries visible to CTranslate2 and ONNX Runtime.

The nvidia-*-cu12 wheels drop their .so files under site-packages/nvidia/*/lib, which is not
on the dynamic loader path. Loading them globally once, before the consumers import, is enough.
"""
from __future__ import annotations

import ctypes
import os
import site

_LIBS = [
    "cuda_runtime/lib/libcudart.so.12",
    "cuda_nvrtc/lib/libnvrtc.so.12",
    "cublas/lib/libcublasLt.so.12",
    "cublas/lib/libcublas.so.12",
    "cudnn/lib/libcudnn.so.9",
    "cufft/lib/libcufft.so.11",
    "curand/lib/libcurand.so.10",
]
_done = False


def preload() -> None:
    global _done
    if _done:
        return
    roots = list(site.getsitepackages())
    if site.getusersitepackages():
        roots.append(site.getusersitepackages())
    for root in roots:
        for n in _LIBS:
            p = os.path.join(root, "nvidia", n)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    _done = True
