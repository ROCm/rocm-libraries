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
    ("tensilelite/Tests/common/sparse/gfx950/f8_gate_r.yaml", "e1f225d01f5a", "gfx950", 16, {-2: 2, 0: 2}),
    ("tensilelite/Tests/common/sparse/gfx1250/spmm_tdm_gl2prefetch.yaml", "be351d89cb18", "gfx1250", 16, {0: 8}),
    ("tensilelite/Tests/common/sparse/gfx1250/spmm_fp16_ml1.yaml", "9805bc54f436", "gfx1250", 12, {0: 6}),
    ("tensilelite/Tests/common/sparse/gfx950/spmm_dtl.yaml", "9a74cd29a31a", "gfx950", 18, {0: 5}),
    ("tensilelite/Tests/common/sparse/gfx94x/bf16_activation.yaml", "3d798432e6cc", "gfx942", 4, {0: 2}),
    ("tensilelite/Tests/common/sparse/gfx1250/spmm_tdm_all.yaml", "c84f88bf434a", "gfx1250", 8, {0: 4}),
    ("tensilelite/Tests/common/sparse/gfx950/bf16_gate_r.yaml", "c89d7cc489d6", "gfx950", 16, {0: 4}),
    ("tensilelite/Tests/common/sparse/gfx94x/spmm_i8_mi16.yaml", "5f0fc4210647", "gfx942", 144, {0: 2}),
    ("tensilelite/Tests/common/sparse/gfx94x/spmm_vw_lg_one.yaml", "24b66a02a120", "gfx942", 32, {0: 8}),
    ("tensilelite/Tests/common/sparse/gfx94x/fp16_gate_r.yaml", "f738eeed11e8", "gfx942", 16, {0: 4}),
    ("tensilelite/Tests/common/sparse/gfx94x/spmm_i8is.yaml", "d83275f811f5", "gfx942", 32, {0: 3}),
    ("tensilelite/Tests/common/sparse/gfx94x/i8_activation.yaml", "d3edd271bd90", "gfx942", 2, {0: 1}),
    ("tensilelite/Tests/common/sparse/gfx94x/spmm_bf8n.yaml", "ed161ca01864", "gfx942", 32, {0: 3}),
    ("tensilelite/Tests/common/sparse/gfx94x/spmm_fp16_mi16.yaml", "3229966bf1dd", "gfx942", 2, {0: 1}),
    ("tensilelite/Tests/common/sparse/gfx950/spmm_ldstr.yaml", "1908a2f1ce21", "gfx950", 768, {0: 8}),
]

# Emit sample size per config (default 8). The VectorWidthA/B [-1, 1] fork puts
# every valid permutation of these two configs past the first 8, so they need the
# whole search (32) in the sample to emit any kernel.
_DEFAULT_LIMIT = 8
_LIMITS = {
    "tensilelite/Tests/common/sparse/gfx94x/spmm_i8is.yaml": 32,
    "tensilelite/Tests/common/sparse/gfx94x/spmm_bf8n.yaml": 32,
}

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
        limit=_LIMITS.get(config, _DEFAULT_LIMIT),
        problem_fingerprint=problem_fingerprint,
        expected_fork_count=expected_fork_count,
        expected_statuses=expected_statuses,
    )
