# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- streamk family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/streamk`` configs for the emit god-files. Stream-K configs
combined with MX fp4/fp8, prefetch-across-persistent (PAP), half-PLR, TDM split,
and gl2 prefetch exercise scheduling and global-write arms the ``_designed``
catalog never reaches. These are explicitly coverage-oriented smoke cases.
Each records the full fork-permutation count before sampling and the exact
emitter-status multiset for the bounded sample. They do not claim to
characterize the instruction sequence.
"""

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/streamk/sk_mx32f4_quick.yaml", "169ad6a42335", "gfx942", 4, {-2: 1, 0: 1}),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8_force_dp_only_halfplr_tdm_pap.yaml", "3e68fae557d9", "gfx1250", 4, {0: 4}),
    ("Tensile/Tests/common/streamk/gfx950/sk_sgemm_pap.yaml", "6d3aa35f8d69", "gfx950", 192, {0: 8}),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_bgemm_tdm_split.yaml", "450ae10a820f", "gfx1250", 12, {0: 6}),
    ("Tensile/Tests/common/streamk/gfx950/sk_mxf4gemm_pap.yaml", "709a3a18f17f", "gfx950", 256, {0: 4}),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf4gemm_pap_prefetchgl2.yaml", "8f87c4a050e1", "gfx1250", 64, {0: 8}),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8gemm_tdm_split.yaml", "a77530d1293d", "gfx1250", 4, {0: 2}),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_halfplr_f8gemm_tdm.yaml", "e2bcdd975b58", "gfx1250", 18, {0: 8}),
    ("Tensile/Tests/common/streamk/sk_dynamic.yaml", "62a98b7e9c7c", "gfx942", 20, {-2: 4, 0: 4}),
    ("Tensile/Tests/common/streamk/sk_dynamic_work_stealing.yaml", "f80dee17348c", "gfx942", 4, {0: 4}),
    ("Tensile/Tests/common/streamk/sk_hybrid_work_stealing.yaml", "04733a1d12cc", "gfx942", 4, {0: 4}),
]

_IDS = [f"{c[0].rsplit('/', 1)[-1][:-5]}-group-{c[1]}" for c in _CONFIGS]


@pytest.mark.parametrize(
    "config,problem_fingerprint,arch,expected_fork_count,expected_statuses",
    _CONFIGS,
    ids=_IDS,
)
def test_setcover_streamk_emits(
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
