# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UHD Generation Tool - Train and export heuristic models for hipDNN."""

import os
import sys

__version__ = "0.1.0"

# `_generated/` holds the flatc-produced Python bindings for uhd.fbs and
# gbdt_model.fbs. They declare the schemas' own namespace
# (`hipdnn_flatbuffers_sdk.data_objects`) and import each other absolutely, so the
# directory holding that namespace has to be importable.
#
# Prepended, not appended: these bindings are generated from the schemas that ship
# beside this tool, and a differently-versioned copy installed elsewhere on the
# path would be silently preferred. Vendoring the bindings only helps if the
# vendored ones win.
#
# Regenerate with `python projects/hipdnn/scripts/run_flatc.py <schema>.fbs`, or by
# building with HIPDNN_GENERATE_SDK_HEADERS=ON.
_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_generated")
if _GENERATED_DIR not in sys.path:
    sys.path.insert(0, _GENERATED_DIR)
