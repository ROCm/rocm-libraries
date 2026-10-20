# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared setup for the LSE store tests: harness path and hook factories."""

from __future__ import annotations

import sys
from pathlib import Path

_INSTANCES = Path(__file__).resolve().parents[2] / "platform" / "tests" / "instances"
if str(_INSTANCES) not in sys.path:
    sys.path.insert(0, str(_INSTANCES))

from kernels.common._lse_store import (  # noqa: E402
    make_mfma_lse_epilogue,
    make_wmma_lse_epilogue,
)


def factory_for(arch, layout="bhs"):
    """Hook factory for the probe shell: [B,H,S,1] or [B,S,H,1] element strides."""
    make = (
        make_wmma_lse_epilogue
        if arch.startswith(("gfx11", "gfx12"))
        else make_mfma_lse_epilogue
    )

    def factory(b, lse, *, head, lse_h, band_valid):
        if layout == "bhs":
            strides = dict(row_stride=1, head_stride=lse_h)
        else:
            strides = dict(row_stride=lse_h, head_stride=1)
        return make(b, lse, head_idx=head, extra_valid=band_valid, **strides)

    return factory
