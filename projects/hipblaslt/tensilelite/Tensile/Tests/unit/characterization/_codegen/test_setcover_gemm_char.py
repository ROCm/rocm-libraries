# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- gemm family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common`` configs for the emit god-files (KernelWriterAssembly,
KernelWriter, GlobalWriteBatch, LocalRead). Each exercises emitter branch arms
the ``_designed`` characterization catalog never reaches (narrow float types,
MX fp6, dot2, swizzle, agent-table, i-cache flush). These are explicitly
coverage-oriented smoke cases. Each records the full fork-permutation count
before sampling and the exact emitter-status multiset for the bounded sample.
Representative cases also assert a narrow, stable source pattern; the rest do
not claim to characterize the instruction sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_cls_gfx1250.yaml", "2f7e8ed3bd46", "gfx1250", 48, {0: 3}),
    ("Tensile/Tests/common/gemm/gfx950/agntab_coverage_gfx950.yaml", "f794c1c0fa6a", "gfx950", 128, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/bf6_gfx1250.yaml", "7461004022b7", "gfx1250", 64, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx12/segment_interleave_gfx1250.yaml", "f7f947393334", "gfx1250", 72, {0: 8}),
    ("Tensile/Tests/common/gemm/icache_flush.yaml", "04d3661728e3", "gfx942", 1, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx12/mxf6_tdm_gfx1250.yaml", "ae046eba508e", "gfx1250", 4, {-2: 2, 0: 2}),
    ("Tensile/Tests/common/gemm/gfx12/zgemm_gfx1250.yaml", "29c407b9e664", "gfx1250", 18, {0: 8}),
    ("Tensile/Tests/common/gemm/hh_f8nhs.yaml", "2ca1bbea22cc", "gfx942", 4, {0: 4}),
    ("Tensile/Tests/common/gemm/mix_cvt_after_ds_fnuz.yaml", "b1bd4a0a2215", "gfx942", 1024, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/cgemm_gfx1250.yaml", "9640acc49450", "gfx1250", 18, {0: 8}),
    ("Tensile/Tests/common/gemm/dot2_gfx942.yaml", "f79591080413", "gfx942", 2880, {0: 3}),
    ("Tensile/Tests/common/gemm/gfx12/agntab_coverage_gfx1250.yaml", "4b6d8db0315c", "gfx1250", 40, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml", "d1d291359808", "gfx1250", 180, {0: 8}),
    ("Tensile/Tests/common/gemm/swizzleB.yaml", "86c30091d793", "gfx942", 13824, {0: 2}),
    ("Tensile/Tests/common/gemm/gfx950/fp8_mxfp4_bf16_tn_act.yaml", "9158147645bf", "gfx950", 24, {-2: 2, 0: 2}),
    ("Tensile/Tests/common/gemm/fp8nfp16mix_hhs.yaml", "f9f0eccb98d0", "gfx942", 5, {0: 5}),
    ("Tensile/Tests/common/gemm/gfx12/f8b8ss_gfx1250.yaml", "35ae075de6c1", "gfx1250", 8, {0: 4}),
    ("Tensile/Tests/common/gemm/fp32_nt.yaml", "bb7dea403a47", "gfx942", 1, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx12/bf16_CLS_gfx1250.yaml", "fd11213715b9", "gfx1250", 56, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx11/fp16_HH_BHS_bf16mfma_gfx11.yaml", "d87dd1304f3f", "gfx1100", 1536, {0: 8}),
    ("Tensile/Tests/common/gemm/lsu_fnuz.yaml", "55c95a6f640b", "gfx942", 648, {0: 3}),
    ("Tensile/Tests/common/gemm/gfx11/i8_gsu_gfx11.yaml", "f18d461d35eb", "gfx1100", 768, {0: 4}),
    ("Tensile/Tests/common/gemm/gfx950/f16f8mix_ss_stoch.yaml", "56f1cf16835d", "gfx950", 4, {-2: 2, 0: 2}),
    ("Tensile/Tests/common/gemm/gfx950/subtile_bf16.yaml", "aa8c3b5909ee", "gfx950", 372, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx950/ss_bss.yaml", "d25ba0f420c6", "gfx950", 60, {0: 5}),
    ("Tensile/Tests/common/gemm/lsu_i8.yaml", "60602b632251", "gfx942", 1536, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx12/b8b8s_gfx1250.yaml", "e18b653798c4", "gfx1250", 2, {0: 1}),
    ("Tensile/Tests/common/gemm/ulsgro1.yaml", "e330b7af74a1", "gfx942", 1728, {-2: 1, 0: 5}),
    ("Tensile/Tests/common/gradient/gfx1250/bbs_bgradd_gfx1250.yaml", "72c57c435770", "gfx1250", 2, {0: 2}),
    ("Tensile/Tests/common/gemm/fp8n_use_e.yaml", "389ea24cef73", "gfx942", 8, {0: 5}),
    ("Tensile/Tests/common/gemm/gfx950/f8b8hs.yaml", "b02023678f13", "gfx950", 192, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/b6f4ss_gfx1250.yaml", "f5036c12234f", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/f4b8ss_gfx1250.yaml", "90e8957e0743", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/f6b8ss_gfx1250.yaml", "6e91b1125765", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/f8b6ss_gfx1250.yaml", "c514ac5ce49d", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx12/f8f4ss_gfx1250.yaml", "87b96af81366", "gfx1250", 16, {0: 8}),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml", "6b2bcec481bb", "gfx950", 1, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx950/f8f16mix_f8s.yaml", "30f42a0cd29c", "gfx950", 4, {0: 4}),
    ("Tensile/Tests/common/gemm/gfx950/subtile_mxfp8_bias_sav.yaml", "e3fc3e679e73", "gfx950", 5, {0: 5}),
    ("Tensile/Tests/common/gemm/zgemm.yaml", "04e9095dac26", "gfx942", 5376, {0: 4}),
    ("Tensile/Tests/common/gemm/gfx12/f4f6ss_tdm_gfx1250.yaml", "d4ec8d7cc895", "gfx1250", 4, {0: 4}),
    ("Tensile/Tests/common/gemm/gfx12/f6b6ss_gfx1250.yaml", "c108a606b816", "gfx1250", 128, {0: 2}),
    ("Tensile/Tests/common/gemm/gfx12/f8f8s_pk8_gfx1250.yaml", "670377fa0c67", "gfx1250", 2, {0: 1}),
    ("Tensile/Tests/common/gemm/gfx12/xfp32_gfx1250.yaml", "94a59069a0fc", "gfx1250", 3, {0: 3}),
    ("Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml", "c55563e94a51", "gfx950", 2, {0: 2}),
    ("Tensile/Tests/common/gemm/gfx950/general_wgm.yaml", "2415f2a2c2e4", "gfx950", 12, {0: 6}),
    ("Tensile/Tests/common/gemm/swizzleA.yaml", "a3e02daddef4", "gfx942", 13824, {0: 2}),
    ("Tensile/Tests/common/gemm/use_beta_false.yaml", "4a628755da9a", "gfx942", 1, {0: 1}),
    ("Tensile/Tests/common/gradient/fp8bf8nss_gradient_bias_b.yaml", "eb6c17932259", "gfx942", 512, {0: 2}),
]

_SOURCE_PATTERNS = {
    "Tensile/Tests/common/gemm/gfx12/bf6_gfx1250.yaml": (
        ("scaled fp6 matrix instruction", r"^v_wmma_scale_f32_16x16x128_f8f6f4\b"),
    ),
    "Tensile/Tests/common/gemm/gfx12/mxf6_tdm_gfx1250.yaml": (
        ("scaled fp6 matrix instruction", r"^v_wmma_scale_f32_16x16x128_f8f6f4\b"),
        ("TDM global prefetch", r"^global_prefetch_b8\b"),
    ),
    "Tensile/Tests/common/gemm/dot2_gfx942.yaml": (
        ("packed fp16 dot product", r"^v_dot2_f32_f16\b"),
    ),
    "Tensile/Tests/common/gemm/swizzleA.yaml": (
        ("swizzled A wave assignment", r"^v_and_b32 .*// SwizzleTensorA:"),
    ),
    "Tensile/Tests/common/gemm/swizzleB.yaml": (
        ("swizzled B wave assignment", r"^v_lshrrev_b32 .*// SwizzleTensorB:"),
    ),
}

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize(
    "config,problem_fingerprint,arch,expected_fork_count,expected_statuses",
    _CONFIGS,
    ids=_IDS,
)
def test_setcover_gemm_emits(
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
        validate_source=config in _SOURCE_PATTERNS,
        required_source_patterns=_SOURCE_PATTERNS.get(config, ()),
    )
