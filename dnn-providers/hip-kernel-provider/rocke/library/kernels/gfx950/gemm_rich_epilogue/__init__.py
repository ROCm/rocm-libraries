# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rich GEMM epilogues (bias / activation / multi-D / fp8 output / fused
reductions) fused into one gfx950 GEMM kernel."""
from .epilogue import (
    BODIES,
    RICH_EPILOGUE_ARCHS,
    EpiHooks,
    EpiKnobs,
    RichEpilogueSpec,
    body_spec,
    build_rich_epilogue,
    rich_epilogue_gemm_spec,
)

__all__ = [
    "BODIES",
    "RICH_EPILOGUE_ARCHS",
    "EpiHooks",
    "EpiKnobs",
    "RichEpilogueSpec",
    "body_spec",
    "build_rich_epilogue",
    "rich_epilogue_gemm_spec",
]
