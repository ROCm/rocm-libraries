################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 5 — TensileCreateLibrary.run() end-to-end (CPU-only).

Drives the full create-library orchestration: parse logic -> derive solutions ->
generate + assemble kernels (amdclang++, no GPU) -> write the solution library,
helper kernels and static files. This covers the driver layer
(``run`` / ``writeSolutionsAndKernelsTCL`` / assemble / library write /
``writeHelpers`` / ``copyStaticFiles``) that the per-kernel emit suites don't.

The golden is the sorted list of produced output files (deterministic,
path-stable). A few CLI variants exercise distinct write paths (separate vs
lazy, msgpack vs yaml library format).
"""

import os
import shutil

import pytest

from codegen_harness import run_create_library

pytestmark = pytest.mark.unit

_DATA = os.path.join(os.path.dirname(__file__), "data")


def _logic_dir(tmp_path, src_rel):
    d = tmp_path / "logic"
    d.mkdir()
    shutil.copy(os.path.join(_DATA, src_rel), d / "logic.yaml")
    return d


# (id, source logic file, arch, extra CLI args)
_CASES = [
    ("gfx942_default", "gfx942/HSS_BH_Bias.yaml", "gfx942", ()),
    ("gfx942_msgpack", "gfx942/HSS_BH_Bias.yaml", "gfx942", ("--library-format=msgpack",)),
    ("gfx942_no_lazy", "gfx942/HSS_BH_Bias.yaml", "gfx942", ("--no-lazy-library-loading",)),
    ("gfx950_default", "gfx950/HSS.yaml", "gfx950", ()),
]


@pytest.mark.parametrize("cid,src,arch,extra", _CASES, ids=[c[0] for c in _CASES])
def test_run_creates_library(cid, src, arch, extra, tmp_path, snapshot):
    logic = _logic_dir(tmp_path, src)
    out = tmp_path / "out"
    files = run_create_library(logic, out, arch=arch, extra_args=extra)
    # core artifacts always produced
    assert any(f.endswith("Kernels.cpp") for f in files)
    assert any(f.endswith(".hsaco") for f in files)
    assert files == snapshot
