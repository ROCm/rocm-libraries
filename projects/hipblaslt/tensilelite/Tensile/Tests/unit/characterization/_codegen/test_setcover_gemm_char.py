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

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_cls_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/agntab_coverage_gfx950.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx12/bf6_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/segment_interleave_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/icache_flush.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/mxf6_tdm_gfx1250.yaml", 0, "gfx1250", False),
    ("Tensile/Tests/common/gemm/gfx12/zgemm_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/hh_f8nhs.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/mix_cvt_after_ds_fnuz.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/cgemm_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/dot2_gfx942.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/agntab_coverage_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/swizzleB.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx950/fp8_mxfp4_bf16_tn_act.yaml", 0, "gfx950", False),
    ("Tensile/Tests/common/gemm/fp8nfp16mix_hhs.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/f8b8ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/fp32_nt.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/bf16_CLS_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx11/fp16_HH_BHS_bf16mfma_gfx11.yaml", 0, "gfx1100", True),
    ("Tensile/Tests/common/gemm/lsu_fnuz.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx11/i8_gsu_gfx11.yaml", 0, "gfx1100", True),
    ("Tensile/Tests/common/gemm/gfx950/f16f8mix_ss_stoch.yaml", 0, "gfx950", False),
    ("Tensile/Tests/common/gemm/gfx950/subtile_bf16.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/ss_bss.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/lsu_i8.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/b8b8s_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/ulsgro1.yaml", 0, "gfx942", False),
    ("Tensile/Tests/common/gradient/gfx1250/bbs_bgradd_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/fp8n_use_e.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx950/f8b8hs.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx12/b6f4ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f4b8ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f6b8ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8b6ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8f4ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/subtile_mxfp8_bias_sav.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/zgemm.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/f4f6ss_tdm_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f6b6ss_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_pk8_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/xfp32_gfx1250.yaml", 0, "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/general_wgm.yaml", 0, "gfx950", True),
    ("Tensile/Tests/common/gemm/swizzleA.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gemm/use_beta_false.yaml", 0, "gfx942", True),
    ("Tensile/Tests/common/gradient/fp8bf8nss_gradient_bias_b.yaml", 0, "gfx942", True),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-problem{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize("config,problem_index,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_gemm_emits_golden(config, problem_index, arch, all_ok, snapshot):
    """Config emits >=1 kernel (all err==0 when all_ok); golden pins per-kernel err."""
    assert_config_emits_golden(
        config,
        arch,
        snapshot,
        limit=8,
        all_ok=all_ok,
        problem_index=problem_index,
    )
