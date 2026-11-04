# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation tests of the unified backward's paged address generator (device).

Each test builds the main kernel with one deliberate fault in
:mod:`kernels.common._attention_bwd_paged` (through ``mock.patch``, in a
private kernel cache) and runs a case that exercises it; the numeric and
integrity checks of the unified harness must fail. A fault the checks cannot
see would mean the suite does not guard that piece of addressing.

* ``no_row_clamp``: K / V rows past ``len_kv`` are read unclamped (they hold
  NaN, which reaches dQ through ``dS * K``);
* ``no_zero_fill``: direct scatter stores only rows ``< len_kv``, so the slots
  of the last live page past the sequence end keep the sentinel;
* ``first_piece_only``: the ``"sub"`` class uses the first page of the tile
  for every row;
* ``no_in_page_offset``: the ``"multi"`` class drops the first in-page slot of
  a tile (wrong rows when a page holds several tiles), in both page modes.

Not mutated here: reading block-table entries past the live blocks, because
no access uses such an entry (an equivalent mutant on device); the IR test
``test_block_table_reads_are_guarded_by_the_live_blocks`` covers it.
"""

from __future__ import annotations

from unittest import mock

import pytest

from kernels.common import _attention_bwd_paged as paged

from ._attention_bwd_harness import DEVICE_ARCHS
from ._attention_bwd_unified_harness import ARCH, run_unified_case
from .sdpa.bwd_unified_cases import get_unified_case

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

_ORIG_SLAB_INIT = paged.PagedSlab.__init__
_ORIG_LOCATE = paged.PagedKvTile.locate
_ORIG_WS = paged.PagedKvTile.workspace_row


def _slab_without_clamp(self, *args, **kw):
    kw["last"] = None
    _ORIG_SLAB_INIT(self, *args, **kw)


def _store_without_zero_fill(self, k):
    live = self.b.cmp_lt(k, self.seq.length)
    return live, live


def _locate_first_piece(self, lr):
    pid, slot = _ORIG_LOCATE(self, lr)
    return self.page_ids[0], slot


def _locate_without_in0(self, lr):
    if self.sub_page is None:
        return self.page_ids[0], lr
    return _ORIG_LOCATE(self, lr)


def _workspace_without_in0(self, ws_ptr, *, page_rows, width, head):
    saved = self.in0
    if self.sub_page is None:
        self.in0 = self.b.const_i32(0)
    try:
        return _ORIG_WS(self, ws_ptr, page_rows=page_rows, width=width, head=head)
    finally:
        self.in0 = saved


MUTATIONS = {
    "no_row_clamp": [(paged.PagedSlab, "__init__", _slab_without_clamp)],
    "no_zero_fill": [(paged.PagedKvTile, "store_rule", _store_without_zero_fill)],
    "first_piece_only": [(paged.PagedKvTile, "locate", _locate_first_piece)],
    "no_in_page_offset": [
        (paged.PagedKvTile, "locate", _locate_without_in0),
        (paged.PagedKvTile, "workspace_row", _workspace_without_in0),
    ],
}

ROWS = [
    ("no_row_clamp", "bwd_unified_page16_br", "direct"),
    ("no_row_clamp", "bwd_unified_page64_br", "atomic"),
    ("no_zero_fill", "bwd_unified_canary_page256_len100", "direct"),
    ("first_piece_only", "bwd_unified_page16_br", "direct"),
    ("first_piece_only", "bwd_unified_page32_br", "atomic"),
    ("no_in_page_offset", "bwd_unified_zero_q_last_br_page256", "direct"),
    ("no_in_page_offset", "bwd_unified_zero_q_last_br_page256", "atomic"),
]


@pytest.mark.parametrize("mutation,case_id,mode", ROWS, ids=lambda x: str(x))
def test_mutation_is_caught(mutation, case_id, mode):
    case = get_unified_case(case_id)
    clean = run_unified_case(case, arch=ARCH, mode=mode, cache={})
    assert not clean.integrity and not clean.problems, (clean.integrity, clean.problems)
    if mutation == "first_piece_only" and clean.plan.specs["main"].sub_page is None:
        # RDNA ships no "sub" tile (its kN0 = 16 divides every page)
        pytest.skip(f"{case_id} runs the 'multi' class on {ARCH}")
    patches = [
        mock.patch.object(obj, attr, fn) for obj, attr, fn in MUTATIONS[mutation]
    ]
    for p in patches:
        p.start()
    try:
        run = run_unified_case(case, arch=ARCH, mode=mode, cache={})
    finally:
        for p in reversed(patches):
            p.stop()
    assert run.integrity or run.problems, f"{mutation} was not caught on {case_id}"
