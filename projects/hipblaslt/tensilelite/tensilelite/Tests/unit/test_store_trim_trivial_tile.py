# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The edge-store proof needs a narrowed fact (multiple > 1).

A one-wide tile with the default AssertFree*ElementMultiple=1 (stock gfx950
MT1x2/MT1x4 solutions) is not a narrowing request, so its edge check is kept.
Preservation: dropping the "> 1" condition (proof from 1 % 1 == 0) breaks it.
"""
import re

import pytest

from test_store_trim_codegen import _edge_check


@pytest.mark.parametrize("dim", [0, 1], ids=["m", "n"])
def test_default_multiple_on_one_wide_tile_keeps_edge_branch(dim):
    src = _edge_check(1, 1, mt0=1, mt1=1, dim=dim)
    assert re.search(r"\bs_cbranch_scc1\s+label_edge_target\b", src), "edge check removed without a narrowed fact"
