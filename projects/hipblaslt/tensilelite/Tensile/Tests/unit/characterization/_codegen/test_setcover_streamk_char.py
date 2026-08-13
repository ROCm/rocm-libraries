# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover emit harvest -- streamk family seeds.

Feature-config seeds selected by the dynamic emit set-cover
(work/mutcov-evidence/feature_setcover.py) as the highest-marginal shipped
``Tests/common/streamk`` configs for the emit god-files. Stream-K configs
combined with MX fp4/fp8, prefetch-across-persistent (PAP), half-PLR, TDM split,
and gl2 prefetch exercise scheduling and global-write arms the ``_designed``
catalog never reaches. Each emits CPU-only and its order-invariant
``{basename, err}`` digest is pinned as a golden. Configs whose emit returns a
non-zero code on some kernels are marked ``all_ok=False``; their golden pins the
actual per-kernel error codes rather than asserting err==0.
"""

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIGS = [
    ("Tensile/Tests/common/streamk/sk_mx32f4_quick.yaml", "gfx942", False),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8_force_dp_only_halfplr_tdm_pap.yaml", "gfx1250", True),
    ("Tensile/Tests/common/streamk/gfx950/sk_sgemm_pap.yaml", "gfx950", True),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_bgemm_tdm_split.yaml", "gfx1250", True),
    ("Tensile/Tests/common/streamk/gfx950/sk_mxf4gemm_pap.yaml", "gfx950", True),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf4gemm_pap_prefetchgl2.yaml", "gfx1250", False),
    ("Tensile/Tests/common/streamk/gfx1250/core/sk_mxf8gemm_tdm_split.yaml", "gfx1250", True),
]

_IDS = [c[0].rsplit("/", 1)[-1][:-5] for c in _CONFIGS]


@pytest.mark.parametrize("config,arch,all_ok", _CONFIGS, ids=_IDS)
def test_setcover_streamk_emits_golden(config, arch, all_ok, snapshot):
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
