# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared by the torch-free tests that drive a GDN/KDA decode tool with a
stand-in launch: the verdicts (``dispatch.gdn.gate_rules.Gate``) a tool must
never time or rank, one per way the fp32 gate can fail, and one it must."""

from __future__ import annotations

import math

from dispatch.gdn.gate_rules import TOL, Gate

GOOD = Gate(0.0, 0.0, 0.0, False)

BAD_VERDICTS = {
    "nan state behind a finite output": Gate(0.0, math.nan, 0.0, False),
    "nan output": Gate(math.nan, 0.0, 0.0, False),
    "nan conv state": Gate(0.0, 0.0, math.nan, False),
    "infinite error": Gate(0.0, math.inf, 0.0, False),
    "error over the tolerance": Gate(0.0, math.nextafter(TOL, 1.0), 0.0, False),
    "write outside the written slots": Gate(0.0, 0.0, 0.0, True),
}
