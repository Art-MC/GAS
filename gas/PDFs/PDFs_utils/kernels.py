import cupy as cp
__all__ = ["kernels"]

kernels = {}

import os

backend="nvrtc"
options=('-gen-opt-lto',)

with open(os.path.join(os.path.dirname(__file__), "./near_neighbors.cu"), "r") as f:
    kernels["find_neighbors"] = cp.RawKernel(
        f.read(),
        "find_neighbors",
        backend=backend,
        options=options,
        )

with open(os.path.join(os.path.dirname(__file__), "./rdf.cu"), "r") as f:
    kernels["rdf"] = cp.RawKernel(
        f.read(),
        "rdf",
        backend=backend,
        options=options,
        )
