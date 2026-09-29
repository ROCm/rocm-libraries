################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 3 — capped emit from vendored, trimmed tuned-logic fixtures.

Some emit paths (notably the StreamK kernel body, and the long tail of MI-shape
/ schedule combinations) only appear in the big tuned logic files, which don't
otherwise get exercised by the small curated Phase 1 fixtures. Rather than read
the live production files directly out of ``library/src`` (which are
frequently retuned by unrelated changes and would make this suite's golden
flap on unrelated PRs -- see DECISIONS.md), each entry here is a vendored,
trimmed copy: the original header/problem-type/index-table shape plus *only*
the ``Solutions`` entries needed to reproduce the pinned capped kernel set
(``limit=``), following the same "small self-contained copy of a valid tuning
logic file" convention as Phase 1's ``data/<arch>/*.yaml`` fixtures.

These fixtures are always present in the checkout, so a missing file is a hard
error, not a skip. Order-invariant golden ({basename, err}); see ``target.md``.
"""

import os

import pytest

from codegen_harness import emit_kernels_from_logic

pytestmark = pytest.mark.unit

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "bigfiles")

# (label, filename-under-data/bigfiles, kernel cap) — capped to keep emit bounded.
_BIG = [
    ("streamk_gfx942_S", "streamk_gfx942_S.yaml", 6),
    ("streamk_gfx942_Ailk", "streamk_gfx942_Ailk.yaml", 4),
    ("freesize_gfx942_F8NH_GSU", "freesize_gfx942_F8NH_GSU.yaml", 6),
    ("equality_gfx950_HSS_big", "equality_gfx950_HSS_big.yaml", 6),
    ("gfx90a_HSS_big", "gfx90a_HSS_big.yaml", 6),
    ("gfx950_origami_MX", "gfx950_origami_MX.yaml", 6),
    ("gfx1201_I8II", "gfx1201_I8II.yaml", 6),
    ("gfx1250_GG", "gfx1250_GG.yaml", 6),
    ("navi31_HSS", "navi31_HSS.yaml", 6),
    ("aqua_FreeSize_GSU9", "aqua_FreeSize_GSU9.yaml", 6),
]


@pytest.mark.parametrize("label,rel,cap", _BIG, ids=[b[0] for b in _BIG])
def test_bigfile_capped_emit(label, rel, cap, snapshot):
    path = os.path.join(_DATA_ROOT, rel)
    results = emit_kernels_from_logic(path, limit=cap)
    assert results
    assert all(e == 0 for _b, _s, e in results)
    digest = [{"basename": b, "err": e} for (b, _s, e) in results]
    assert digest == snapshot
