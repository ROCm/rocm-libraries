# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- sparse family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/sparse`` configs for the emit god-files. Sparse (spmm) configs
exercise gate-residual, TDM gl2-prefetch, mixed-list, DirectToLds, and narrow
metadata arms the ``_designed`` catalog never reaches -- ``f8_gate_r`` alone is
the single highest-yield config in the whole pool. These are explicitly
coverage-oriented smoke cases. Each records the full fork-permutation count
before sampling and the exact emitter-status multiset for the bounded sample.
They do not claim to characterize the instruction sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/sparse/gfx950/f8_gate_r.yaml", "0b788dea88ed", "gfx950", 16, {-2: 2, 0: 2}),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_gl2prefetch.yaml", "dc8961a25b9d", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_fp16_ml1.yaml", "cb5b400c97ca", "gfx1250", 12, {0: 2}),
    ("Tensile/Tests/common/sparse/gfx950/spmm_dtl.yaml", "9a74cd29a31a", "gfx950", 18, {0: 5}),
    ("Tensile/Tests/common/sparse/gfx94x/bf16_activation.yaml", "3d798432e6cc", "gfx942", 4, {0: 2}),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_all.yaml", "79bef2496f1d", "gfx1250", 8, {0: 8}),
    ("Tensile/Tests/common/sparse/gfx950/bf16_gate_r.yaml", "c89d7cc489d6", "gfx950", 16, {0: 4}),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8_mi16.yaml", "5f0fc4210647", "gfx942", 144, {0: 2}),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_vw_lg_one.yaml", "24b66a02a120", "gfx942", 32, {0: 8}),
    ("Tensile/Tests/common/sparse/gfx94x/fp16_gate_r.yaml", "f738eeed11e8", "gfx942", 16, {0: 4}),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8is.yaml", "8c983468a5a4", "gfx942", 8, {0: 3}),
    ("Tensile/Tests/common/sparse/gfx94x/i8_activation.yaml", "d3edd271bd90", "gfx942", 2, {0: 1}),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_bf8n.yaml", "a372e92e6e6e", "gfx942", 8, {0: 3}),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_fp16_mi16.yaml", "3229966bf1dd", "gfx942", 2, {0: 1}),
    ("Tensile/Tests/common/sparse/gfx950/spmm_ldstr.yaml", "1908a2f1ce21", "gfx950", 768, {0: 8}),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize(
    "config,problem_fingerprint,arch,expected_fork_count,expected_statuses",
    _CONFIGS,
    ids=_IDS,
)
def test_setcover_sparse_emits(
    config, problem_fingerprint, arch, expected_fork_count, expected_statuses
):
    """The selected group retains its search size and bounded emit statuses."""
    assert_config_emits(
        config,
        arch,
        limit=8,
        problem_fingerprint=problem_fingerprint,
        expected_fork_count=expected_fork_count,
        expected_statuses=expected_statuses,
    )
