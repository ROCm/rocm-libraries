# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- sparse family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/sparse`` configs for the emit god-files. Sparse (spmm) configs
exercise gate-residual, TDM gl2-prefetch, mixed-list, DirectToLds, and narrow
metadata arms the ``_designed`` catalog never reaches -- ``f8_gate_r`` alone is
the single highest-yield config in the whole pool. Each emits CPU-only and its
order-invariant ``{basename, err}`` digest is pinned as a golden. Configs whose
emit returns a non-zero code on some kernels are marked ``all_ok=False``; their
golden pins the actual per-kernel error codes rather than asserting err==0.
"""

import pytest

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/sparse/gfx950/f8_gate_r.yaml", 0, "gfx950", False),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_gl2prefetch.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_fp16_ml1.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx950/spmm_dtl.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/sparse/gfx94x/bf16_activation.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_all.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx950/bf16_gate_r.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8_mi16.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_vw_lg_one.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/fp16_gate_r.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8is.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/i8_activation.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_bf8n.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_fp16_mi16.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx950/spmm_ldstr.yaml", 0, "gfx950", True),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-problem{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize("config,problem_index,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_sparse_emits_golden(config, problem_index, arch, all_ok, snapshot):
    """Config emits >=1 kernel (all err==0 when all_ok); golden pins per-kernel err."""
    assert_config_emits_golden(
        config,
        arch,
        snapshot,
        limit=8,
        all_ok=all_ok,
        problem_index=problem_index,
    )
