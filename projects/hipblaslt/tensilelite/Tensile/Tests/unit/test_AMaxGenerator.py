# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
import pytest

mock_global_ti = MagicMock()
mock_global_ti.getArchCaps.return_value = {"HasWave32": False}

def _load_amax_mod():
    p = Path(__file__).resolve().parents[3] / "AMaxGenerator.py"
    spec = importlib.util.spec_from_file_location("AMaxGenerator_under_test", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # We must patch the module inside sys.modules or mock before exec_module
    # to handle module-level imports. However, since the error occurred in
    # kernel_header which is executed at test runtime, setting it after load is sufficient.
    spec.loader.exec_module(mod)
    mod._global_ti = mock_global_ti
    return mod

def test_kernel_header_accum_offset_exclusion():
    mod = _load_amax_mod()
    # Test exclusion for gfx90c
    header_gfx90c = mod.kernel_header(name="test_kernel", gfx_arch="gfx90c", vgpr=32, sgpr=32, lds=256)
    assert ".amdhsa_accum_offset" not in header_gfx90c

    # Test exclusion for gfx900 (already excluded)
    header_gfx900 = mod.kernel_header(name="test_kernel", gfx_arch="gfx900", vgpr=32, sgpr=32, lds=256)
    assert ".amdhsa_accum_offset" not in header_gfx900

    # Test inclusion for gfx90a (not excluded)
    header_gfx90a = mod.kernel_header(name="test_kernel", gfx_arch="gfx90a", vgpr=32, sgpr=32, lds=256)
    assert ".amdhsa_accum_offset" in header_gfx90a
