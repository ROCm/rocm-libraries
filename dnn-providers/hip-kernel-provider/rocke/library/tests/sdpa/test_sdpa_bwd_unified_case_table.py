# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified backward case table: validity, coverage, materialisation.

Coverage the unified numeric gate relies on: page sizes 16, 32, 48, 64, 128
and 256 (and 80); every table kind; zero-length query and kv spans including
the last sequence; every mask; GQA and MQA; shared-prefix tables (atomic
page mode only); the ``page > kN0`` canary; every page class of the CDNA
plans (``"multi"``, ``"sub"``, the one-wave fallback); both page modes for
injective tables. Materialisation: the expected gradient caches equal the
gathered oracle scattered through the table, unreferenced and past-length
slots of K / V hold NaN, poisoned block-table entries name the trap page.
"""

from __future__ import annotations

import numpy as np
import pytest

from .bwd_unified_cases import (
    MASKS,
    TABLES,
    materialize_unified,
    select_unified,
    unified_cases,
    validate_unified_case,
)
from .helpers import TIERS


def test_ids_unique_and_cases_valid():
    cases = unified_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert not validate_unified_case(c), (c.id, validate_unified_case(c))


def test_tiers_are_cumulative():
    sizes = [len(select_unified(t)) for t in TIERS]
    assert sizes == sorted(sizes) and sizes[0] > 0
    assert sizes[-1] == len(unified_cases())


def test_coverage():
    cases = unified_cases()
    assert {16, 32, 48, 64, 128, 256} <= {c.page for c in cases}
    assert set(TABLES) <= {c.table for c in cases}
    assert set(MASKS) <= {c.mask for c in cases}
    assert any(c.shared_prefix and c.modes == ("atomic",) for c in cases)
    assert any(c.h_kv == 1 for c in cases) and any(1 < c.h_q // c.h_kv for c in cases)
    assert any(c.q_lens[-1] == 0 for c in cases)
    assert any(c.kv_lens[-1] == 0 for c in cases)
    assert any(0 in c.q_lens[:-1] for c in cases)
    assert any(0 in c.kv_lens[:-1] for c in cases)
    assert any(c.page > 128 and c.d == 128 and max(c.kv_lens) < c.page for c in cases)
    assert any(c.kv_slack for c in cases)
    assert any("direct" in c.modes and "atomic" in c.modes for c in cases)
    assert {"padded", "narrow"} <= {c.layout for c in cases}
    smoke = select_unified("smoke")
    assert any(c.shared_prefix for c in smoke)
    assert any("canary" in c.tags for c in smoke)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_page_classes_covered(arch):
    from .._attention_bwd_unified_harness import plan_unified_case

    tags = set()
    for c in select_unified("full"):
        for mode in c.modes:
            try:
                tags.add(plan_unified_case(c, arch=arch, mode=mode).route_tag)
            except ValueError:
                # forced knobs the arch does not ship (the numeric test skips
                # the same cases); a case without forced knobs always plans
                assert c.knobs, c.id
    assert tags == {"multi", "sub", "fallback"}


@pytest.mark.parametrize(
    "case_id",
    [
        "bwd_unified_shared_prefix2",
        "bwd_unified_canary_page256_len100",
        "bwd_unified_zero_q_spans",
        "bwd_unified_table_padded_rows",
    ],
)
def test_materialisation(case_id):
    case = next(c for c in unified_cases() if c.id == case_id)
    inp = materialize_unified(case)
    page = case.page
    n_blk = [-(-n // page) for n in case.kv_lens]
    # live entries name referenced pages, everything else the trap page
    for s, n in enumerate(n_blk):
        row = inp.block_table[s]
        assert np.all(row[n:] == inp.trap)
        assert np.all(inp.referenced[row[:n]])
    assert not inp.referenced[inp.trap]
    # the expected caches equal the gathered oracle scattered through the table
    got = np.zeros_like(inp.exp_dk)
    off = 0
    for s, n in enumerate(case.kv_lens):
        for t in range(n):
            got[inp.block_table[s][t // page], t % page] += inp.ref.dk[off + t]
        off += n
    assert np.array_equal(got, inp.exp_dk)
    # never-read K slots hold NaN: unreferenced pages and past-length slots
    k = inp.buffers["k_cache"]
    blk, tok, hs, _ = inp.strides["k_cache"]
    for pid in range(inp.num_blocks):
        for slot in range(page):
            live = inp.referenced[pid] and not inp.past_len[pid, slot]
            v = k[pid * blk + slot * tok]
            assert np.isnan(v) != live, (pid, slot)
    if case.shared_prefix:
        # shared pages are referenced by every sequence
        first = inp.block_table[:, 0]
        assert len(set(first.tolist())) == 1
