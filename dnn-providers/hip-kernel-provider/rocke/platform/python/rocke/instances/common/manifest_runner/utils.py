# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small host helpers shared by manifest-runner problem builders.

The implementations now live in :mod:`rocke.runtime.host_buffers` (they are the
torch-free host-side byte helpers, useful well beyond the manifest runner).
This module re-exports them so existing
``from .utils import as_u8_buffer, nbytes, require_numpy`` call sites keep
working unchanged.
"""

from __future__ import annotations

from rocke.runtime.host_buffers import as_u8_buffer, nbytes, require_numpy

__all__ = ["require_numpy", "nbytes", "as_u8_buffer"]
