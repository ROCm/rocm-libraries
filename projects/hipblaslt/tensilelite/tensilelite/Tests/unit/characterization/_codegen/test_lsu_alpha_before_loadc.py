# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest

from codegen_harness import emit_kernels_from_logic


pytestmark = pytest.mark.unit

_LOGIC = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_logic", "gfx942", "BBS_SK_LSU2_MIAV.yaml"
)


@pytest.fixture(scope="module")
def lsu_miav_source():
    results = emit_kernels_from_logic(_LOGIC, canonical=False)
    assert len(results) == 1
    _base, source, error = results[0]
    assert error == 0
    name = source[source.index(".globl ") :].split(None, 2)[1]
    assert "_MIAV1_" in name and "_WG32_4_2_" in name and "_SSO1_" in name
    return source


def test_lsu_batch_alpha_follows_copy_into_valuc(lsu_miav_source):
    """ROCM-32151: with LocalSplitU > 1, each store batch after the first copies its
    reduced sums into ValuC ("load from N to M"). Alpha must be applied after that
    copy, or it scales stale registers and the batch is stored without alpha."""
    source = lsu_miav_source
    assert "// load from " in source, "expected a multi-batch LocalSplitU store"

    pos = source.find("/* rC *= alpha")
    assert pos != -1
    while pos != -1:
        nextStore = source.find("buffer_store", pos)
        assert nextStore != -1
        assert "// load from " not in source[pos:nextStore], (
            "alpha applied before the batch's LocalSplitU copy at offset %d" % pos
        )
        pos = source.find("/* rC *= alpha", pos + 1)
