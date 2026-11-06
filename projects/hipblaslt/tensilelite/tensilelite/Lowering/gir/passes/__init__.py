# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GIR passes and the ordered lowering pipeline."""

from .pipeline import pipeline, run_pipeline

__all__ = ["pipeline", "run_pipeline"]
