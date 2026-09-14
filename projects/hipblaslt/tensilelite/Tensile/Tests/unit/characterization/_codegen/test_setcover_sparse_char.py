# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- sparse family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/sparse`` configs for the emit god-files. Sparse (spmm) configs
exercise gate-residual, TDM gl2-prefetch, mixed-list, DirectToLds, and narrow
metadata arms the ``_designed`` catalog never reaches -- ``f8_gate_r`` alone is
the single highest-yield config in the whole pool. These are explicitly
coverage-oriented smoke cases. Each emits CPU-only and requires successful
generation unless the configuration has known emitter failures. They do not
claim to characterize the instruction sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/sparse/gfx950/f8_gate_r.yaml", "fc56b34290c7", "gfx950", False),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_gl2prefetch.yaml", "af4f75252b6e", "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_fp16_ml1.yaml", "5a324aac1af1", "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx950/spmm_dtl.yaml", "98697a5d2958", "gfx950", True),
    ("Tensile/Tests/common/sparse/gfx94x/bf16_activation.yaml", "c46c6ac1671c", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx1250/spmm_tdm_all.yaml", "511646251deb", "gfx1250", True),
    ("Tensile/Tests/common/sparse/gfx950/bf16_gate_r.yaml", "07350e62fbfc", "gfx950", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8_mi16.yaml", "70e9ae4458bd", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_vw_lg_one.yaml", "6bf5c390a8db", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/fp16_gate_r.yaml", "40ef92273653", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_i8is.yaml", "d2c92bbbb281", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/i8_activation.yaml", "f4a6eeb5ed7b", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_bf8n.yaml", "e9c68c5a2019", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx94x/spmm_fp16_mi16.yaml", "2569298fc218", "gfx942", True),
    ("Tensile/Tests/common/sparse/gfx950/spmm_ldstr.yaml", "a670dc11d62b", "gfx950", True),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize("config,problem_fingerprint,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_sparse_emits(config, problem_fingerprint, arch, all_ok):
    """The selected problem group reaches emission; ordinary cases succeed."""
    assert_config_emits(
        config,
        arch,
        limit=8,
        all_ok=all_ok,
        problem_fingerprint=problem_fingerprint,
    )
