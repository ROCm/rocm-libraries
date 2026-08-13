# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- gemm family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common`` configs for the emit god-files (KernelWriterAssembly,
KernelWriter, GlobalWriteBatch, LocalRead). Each exercises emitter branch arms
the ``_designed`` characterization catalog never reaches (narrow float types,
MX fp6, dot2, swizzle, agent-table, i-cache flush). Each emits CPU-only and its
order-invariant ``{basename, err}`` digest is pinned as a golden. Configs whose
emit returns a non-zero code on some kernels are marked ``all_ok=False``; their
golden pins the actual per-kernel error codes rather than asserting err==0.
"""

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_cls_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/agntab_coverage_gfx950.yaml", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx12/bf6_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/segment_interleave_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/icache_flush.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/mxf6_tdm_gfx1250.yaml", "gfx1250", False),
    ("Tensile/Tests/common/gemm/gfx12/zgemm_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/hh_f8nhs.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/mix_cvt_after_ds_fnuz.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/cgemm_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/dot2_gfx942.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/agntab_coverage_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/swizzleB.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx950/fp8_mxfp4_bf16_tn_act.yaml", "gfx950", False),
    ("Tensile/Tests/common/gemm/fp8nfp16mix_hhs.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/f8b8ss_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/fp32_nt.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/bf16_CLS_gfx1250.yaml", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx11/fp16_HH_BHS_bf16mfma_gfx11.yaml", "gfx1100", True),
    ("Tensile/Tests/common/gemm/lsu_fnuz.yaml", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx11/i8_gsu_gfx11.yaml", "gfx1100", True),
    ("Tensile/Tests/common/gemm/gfx950/f16f8mix_ss_stoch.yaml", "gfx950", False),
    ("Tensile/Tests/common/gemm/gfx950/subtile_bf16.yaml", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/ss_bss.yaml", "gfx950", True),
    ("Tensile/Tests/common/gemm/lsu_i8.yaml", "gfx942", True),
]

_IDS = [c[0].rsplit("/", 1)[-1][:-5] for c in _CONFIGS]


@pytest.mark.parametrize("config,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_gemm_emits_golden(config, arch, all_ok, snapshot):
    """Config emits >=1 kernel (all err==0 when all_ok); golden pins per-kernel err."""
    results = emit_kernels_from_config(config, limit=8, arch=arch)
    assert len(results) >= 1
    if all_ok:
        assert all(err == 0 for (_b, _s, err) in results)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
