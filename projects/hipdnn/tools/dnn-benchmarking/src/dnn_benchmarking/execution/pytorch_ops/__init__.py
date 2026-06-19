# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch operation implementations for graph execution.

These handlers execute on the device of the input tensors (CPU or CUDA).
Used by both PyTorchReferenceProvider (CPU) and PyTorchCudaExecutor (CUDA).
"""

from ._registry import (
    OpHandler,
    execute_graph,
    get_handler,
    get_supported_operations,
    get_unsupported_operations,
    register_handler,
    supports_graph,
)
from ._warnings import get_reference_warnings

# Import handler submodules so their @register_handler decorators run.
from .handlers import (  # noqa: F401
    batchnorm,
    conv,
    matmul,
    norm,
    pointwise,
    reduction,
    resample,
    sdpa,
)

__all__ = [
    "OpHandler",
    "register_handler",
    "get_handler",
    "get_supported_operations",
    "supports_graph",
    "get_unsupported_operations",
    "execute_graph",
    "get_reference_warnings",
]
