################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 3 — capped emit from large in-checkout tuned logic files.

Some emit paths (notably the StreamK kernel body, and the long tail of MI-shape
/ schedule combinations) only appear in the big tuned logic files that are too
large to vendor into the suite (tens of thousands of lines each). Rather than
copy them, we reference them in place under the hipBLASLt tuning tree and emit a
*capped* representative subset (``limit=``) so the StreamK / broad-schedule
branches are covered without emitting thousands of kernels.

If the tuning tree is not present (e.g. a tensilelite-only checkout) the cases
skip — they add coverage where the data exists and never fail where it doesn't.
Order-invariant golden ({basename, err}); see ``target.md``.
"""

import os

import pytest

from codegen_harness import emit_kernels_from_logic

pytestmark = pytest.mark.unit

# tensilelite/Tensile/Tests/unit/characterization/_codegen/ -> tensilelite -> hipblaslt
_LOGIC_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        *([".."] * 6),
        "library",
        "src",
        "amd_detail",
        "rocblaslt",
        "src",
        "Tensile",
        "Logic",
        "asm_full",
    )
)

# (label, relpath-under-asm_full, kernel cap) — capped to keep emit bounded.
_BIG = [
    ("streamk_gfx942_S", "aquavanjaram/gfx942/StreamK/gfx942_Cijk_Alik_Bjlk_S_B_UserArgs.yaml", 6),
    ("streamk_gfx942_Ailk", "aquavanjaram/gfx942/StreamK/gfx942_Cijk_Ailk_Bjlk_S_B_UserArgs.yaml", 4),
    ("freesize_gfx942_F8NH_GSU", "aquavanjaram/gfx942/FreeSize/aquavanjaram_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_HAS_SAB_SAV_Freesize_NTA4_custom_GSU15.yaml", 6),
    ("equality_gfx950_HSS_big", "gfx950/gfx950_id75a3/Equality/gfx950_Cijk_Ailk_Bjlk_HSS_BH_BiasSH_HAS_SAV_UserArgs.yaml", 6),
    ("gfx90a_HSS_big", "aldebaran/110CU/Equality/aldebaran_Cijk_Ailk_Bljk_HSS_BH.yaml", 6),
    ("gfx950_origami_MX", "gfx950/gfx950/Origami/Origami_ntb4/gfx950_Cijk_Alik_Bjlk_S_MX_B_Bias_HAS_SAV_UserArgs.yaml", 6),
    ("gfx1201_I8II", "gfx1201/GridBased/gfx1201_Cijk_Ailk_Bljk_I8II_SHB_HA_S_SCD_SAV_UserArgs.yaml", 6),
    ("gfx1250_GG", "gfx1250/GridBased/gfx1250_Cijk_Alik_Bljk_HSS_BH_Bias_GG_HA_S_SAV_UserArgs.yaml", 6),
    ("navi31_HSS", "navi31/GridBased/navi31_Cijk_Alik_Bjlk_HSS_BH_HAS_SAV_UserArgs.yaml", 6),
    ("aqua_FreeSize_GSU9", "aquavanjaram/gfx942/FreeSize/aquavanjaram_Cijk_Ailk_Bljk_F8NH_HHS_BH_Freesize_custom_GSU9.yaml", 6),
]


@pytest.mark.parametrize("label,rel,cap", _BIG, ids=[b[0] for b in _BIG])
def test_bigfile_capped_emit(label, rel, cap, snapshot):
    path = os.path.join(_LOGIC_ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f"tuning logic tree not present: {rel}")
    results = emit_kernels_from_logic(path, limit=cap)
    assert results
    assert all(e == 0 for _b, _s, e in results)
    digest = [{"basename": b, "err": e} for (b, _s, e) in results]
    assert digest == snapshot
