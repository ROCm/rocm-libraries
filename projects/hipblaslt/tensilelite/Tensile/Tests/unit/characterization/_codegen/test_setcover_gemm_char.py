# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- gemm family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common`` configs for the emit god-files (KernelWriterAssembly,
KernelWriter, GlobalWriteBatch, LocalRead). Each exercises emitter branch arms
the ``_designed`` characterization catalog never reaches (narrow float types,
MX fp6, dot2, swizzle, agent-table, i-cache flush). These are explicitly
coverage-oriented smoke cases. Each emits CPU-only and requires successful
generation unless the configuration has known emitter failures. They do not
claim to characterize the instruction sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_cls_gfx1250.yaml", "51b1d0cc5a73", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/agntab_coverage_gfx950.yaml", "abcabf698281", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx12/bf6_gfx1250.yaml", "4daa88f04c04", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/segment_interleave_gfx1250.yaml", "7ea1860de90a", "gfx1250", True),
    ("Tensile/Tests/common/gemm/icache_flush.yaml", "61cc4a8fbdb2", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/mxf6_tdm_gfx1250.yaml", "0b0294fb343f", "gfx1250", False),
    ("Tensile/Tests/common/gemm/gfx12/zgemm_gfx1250.yaml", "01e19b53a329", "gfx1250", True),
    ("Tensile/Tests/common/gemm/hh_f8nhs.yaml", "c89129253f4f", "gfx942", True),
    ("Tensile/Tests/common/gemm/mix_cvt_after_ds_fnuz.yaml", "b629a3029f77", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/cgemm_gfx1250.yaml", "3ea4e31cd366", "gfx1250", True),
    ("Tensile/Tests/common/gemm/dot2_gfx942.yaml", "a62bf58917ba", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/agntab_coverage_gfx1250.yaml", "6284d888337e", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml", "0afdebd2b1d5", "gfx1250", True),
    ("Tensile/Tests/common/gemm/swizzleB.yaml", "9263b87aa328", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx950/fp8_mxfp4_bf16_tn_act.yaml", "933568f08dbb", "gfx950", False),
    ("Tensile/Tests/common/gemm/fp8nfp16mix_hhs.yaml", "728030b6dc6d", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/f8b8ss_gfx1250.yaml", "9fb44e5e7d18", "gfx1250", True),
    ("Tensile/Tests/common/gemm/fp32_nt.yaml", "a12296cecc73", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/bf16_CLS_gfx1250.yaml", "c1726607b2ac", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx11/fp16_HH_BHS_bf16mfma_gfx11.yaml", "87bcbedb31ee", "gfx1100", True),
    ("Tensile/Tests/common/gemm/lsu_fnuz.yaml", "e269ecacf90c", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx11/i8_gsu_gfx11.yaml", "fa8619d65d3b", "gfx1100", True),
    ("Tensile/Tests/common/gemm/gfx950/f16f8mix_ss_stoch.yaml", "c00762ecdd37", "gfx950", False),
    ("Tensile/Tests/common/gemm/gfx950/subtile_bf16.yaml", "7723ca12d93b", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/ss_bss.yaml", "178f4339655d", "gfx950", True),
    ("Tensile/Tests/common/gemm/lsu_i8.yaml", "a97870a8cc9d", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/b8b8s_gfx1250.yaml", "52ce19a768a7", "gfx1250", True),
    ("Tensile/Tests/common/gemm/ulsgro1.yaml", "59da04e10a7f", "gfx942", False),
    ("Tensile/Tests/common/gradient/gfx1250/bbs_bgradd_gfx1250.yaml", "276ac6d407fa", "gfx1250", True),
    ("Tensile/Tests/common/gemm/fp8n_use_e.yaml", "d13c82bf2a8d", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx950/f8b8hs.yaml", "de7b6ca49044", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx12/b6f4ss_gfx1250.yaml", "b1d90192542e", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f4b8ss_gfx1250.yaml", "804b4cbb660d", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f6b8ss_gfx1250.yaml", "d84c62e9579a", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8b6ss_gfx1250.yaml", "8ba05d58543b", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8f4ss_gfx1250.yaml", "42b8267f59a3", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml", "0f5ba3ad03fc", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml", "a59ad8ee5437", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/subtile_mxfp8_bias_sav.yaml", "52407638daa2", "gfx950", True),
    ("Tensile/Tests/common/gemm/zgemm.yaml", "baa14ee84eae", "gfx942", True),
    ("Tensile/Tests/common/gemm/gfx12/f4f6ss_tdm_gfx1250.yaml", "54f294c7f2f3", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f6b6ss_gfx1250.yaml", "838967ae0fb3", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_pk8_gfx1250.yaml", "2f8050b1da9c", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx12/xfp32_gfx1250.yaml", "23ff9e767b94", "gfx1250", True),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml", "384864cf0cc9", "gfx950", True),
    ("Tensile/Tests/common/gemm/gfx950/general_wgm.yaml", "e7a1371235c8", "gfx950", True),
    ("Tensile/Tests/common/gemm/swizzleA.yaml", "826443980ca1", "gfx942", True),
    ("Tensile/Tests/common/gemm/use_beta_false.yaml", "93f986313ea0", "gfx942", True),
    ("Tensile/Tests/common/gradient/fp8bf8nss_gradient_bias_b.yaml", "1f54abc87e38", "gfx942", True),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize("config,problem_fingerprint,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_gemm_emits(config, problem_fingerprint, arch, all_ok):
    """The selected problem group reaches emission; ordinary cases succeed."""
    assert_config_emits(
        config,
        arch,
        limit=8,
        all_ok=all_ok,
        problem_fingerprint=problem_fingerprint,
    )
