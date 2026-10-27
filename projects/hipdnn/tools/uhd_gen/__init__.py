# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool - Train and export heuristic models for hipDNN."""

import os
import sys

__version__ = "0.1.0"

# `_generated/` holds the flatc bindings for gbdt_model.fbs, which import each other
# absolutely. Prepended so a differently-versioned installed copy cannot win.
# Regenerate with `python projects/hipdnn/scripts/run_flatc.py <schema>.fbs`.
_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")
if _GENERATED_DIR not in sys.path:
    sys.path.insert(0, _GENERATED_DIR)
