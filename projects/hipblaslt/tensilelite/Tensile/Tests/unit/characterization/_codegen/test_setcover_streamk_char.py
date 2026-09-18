# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- streamk family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/streamk`` configs for the emit god-files. Stream-K configs
combined with MX fp4/fp8, prefetch-across-persistent (PAP), half-PLR, TDM split,
and gl2 prefetch exercise scheduling and global-write arms the ``_designed``
catalog never reaches. These are explicitly coverage-oriented smoke cases.
Each emits CPU-only and requires successful generation unless the configuration
has known emitter failures. They do not claim to characterize the instruction
sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/streamk/sk_mx32f4_quick.yaml", "23ed5c8a0e5c", "gfx942", False),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8_force_dp_only_halfplr_tdm_pap.yaml", "bb4f95f916a4", "gfx1250", True),
    ("Tensile/Tests/common/streamk/gfx950/sk_sgemm_pap.yaml", "793c0936cda1", "gfx950", True),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_bgemm_tdm_split.yaml", "fdc87a84ba11", "gfx1250", True),
    ("Tensile/Tests/common/streamk/gfx950/sk_mxf4gemm_pap.yaml", "c4f8bfd4591c", "gfx950", True),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf4gemm_pap_prefetchgl2.yaml", "31865b42246d", "gfx1250", False),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8gemm_tdm_split.yaml", "d620d4d1320b", "gfx1250", True),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_halfplr_f8gemm_tdm.yaml", "cc297f6c3034", "gfx1250", True),
    ("Tensile/Tests/common/streamk/sk_dynamic.yaml", "acedc4cffbc9", "gfx942", False),
    ("Tensile/Tests/common/streamk/sk_dynamic_work_stealing.yaml", "1d2d14028206", "gfx942", True),
    ("Tensile/Tests/common/streamk/sk_hybrid_work_stealing.yaml", "400c0fc88b6b", "gfx942", True),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize("config,problem_fingerprint,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_streamk_emits(config, problem_fingerprint, arch, all_ok):
    """The selected problem group reaches emission; ordinary cases succeed."""
    assert_config_emits(
        config,
        arch,
        limit=8,
        all_ok=all_ok,
        problem_fingerprint=problem_fingerprint,
    )
