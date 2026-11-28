################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 1 — gfx1100 (RDNA3 / WMMA (navi31)) codegen-emit goldens.

Drives the real assembly emitter over curated valid gfx1100 logic files spanning
data-type families, exercising the arch-specific paths in
``KernelWriterAssembly`` / ``KernelWriter`` / ``Components/*``.
Goldens are compact digests; see ``target.md``.
"""

from pathlib import Path

import pytest
import yaml

from codegen_harness import emit_kernels_from_logic
from matrix import data_dir, digests_for_dir

pytestmark = pytest.mark.unit

_ARCH = "gfx1100"


def test_gfx1100_emit_digests(snapshot):
    digests = digests_for_dir(_ARCH)
    assert digests
    assert all(k["err"] == 0 for f in digests for k in f["kernels"])
    assert digests == snapshot


def test_valu_auxiliary_output_codegen(tmp_path):
    logic = yaml.safe_load((Path(data_dir(_ARCH)) / "SB.yaml").read_text())
    logic[4]["UseE"] = True
    path = tmp_path / "valu-aux.yaml"
    path.write_text(yaml.safe_dump(logic))
    results = emit_kernels_from_logic(path)
    assert results
    assert all(err == 0 for _, _, err in results)
