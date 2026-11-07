################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Bias-gradient (BiasSrc B) writes cover the macro tile and every LocalSplitU group.

Drives grad_biassrc_b_tile_lsu.yaml: a 48x640 macro tile with 256 threads,
whose bias write needs a second pass, and a 16x16 macro tile with
LocalSplitU=4, whose partial sums take a K slot per LocalSplitU group.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "grad_biassrc_b_tile_lsu.yaml",
)


def test_bias_gradient_write_covers_tile_and_local_split_u():
    results = emit_kernels_from_config(_CONFIG, limit=4, arch="gfx942")
    assert len(results) == 2, [base for base, _, _ in results]
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}"
    sources = [src for _, src, _ in results]
    tail = [src for src in sources if "pass serial += NumThreads" in src]
    lsu = [src for src in sources if "LocalSplitU: group = tid / group threads" in src]
    assert len(tail) == 1, "The 48x640 kernel writes its bias in two passes"
    assert len(lsu) == 1, "The LocalSplitU kernel stores each group's partial sums"
    assert tail[0] is not lsu[0]
    assert "LocalSplitU K offset" in lsu[0]
