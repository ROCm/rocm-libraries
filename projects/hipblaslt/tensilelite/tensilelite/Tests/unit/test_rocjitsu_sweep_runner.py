# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run client-tool tests in the source and installed TensileLite unit lanes."""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
__all__ = ["SweepTests"]

root = Path(__file__).resolve().parents[3]
scripts = root.parent / "clients" / "scripts" / "rocjitsu"
if not scripts.is_dir():
    scripts = root / "rocjitsu"

sys.path.insert(0, str(scripts))
try:
    from test_rocjitsu_race_sweep import SweepTests as SweepTests
finally:
    sys.path.pop(0)
