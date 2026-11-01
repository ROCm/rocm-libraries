# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The f64 MMA probe moved to :mod:`rocke.instances.common.f64_mma_probe` (it
runs on every target whose catalog has the f64 atoms); this name stays."""

from ..common.f64_mma_probe import (  # noqa: F401
    F64_MFMA_ARCHS,
    F64MmaProbeSpec,
    build_f64_mma_probe,
    f64_mfma_archs,
)

__all__ = ["F64_MFMA_ARCHS", "F64MmaProbeSpec", "build_f64_mma_probe", "f64_mfma_archs"]
