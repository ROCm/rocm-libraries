################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for build_idmap, invert_idmap_to_id_to_category, split_for_plr.

These helpers live in ScheduleCapture.py and are the single source of truth
for the CMS category schema. Both customMainLoopSchedule (idMap construction)
and the SIA3 default-side capture path (id-to-category map for tag_by_origin_id)
will route through them.
"""

import pytest

from rocisa.code import Module, TextBlock

from Tensile.Components.ScheduleCapture import (
    build_idmap,
    invert_idmap_to_id_to_category,
    split_for_plr,
)


def _mk_module(name, n_items):
    """Build a Module with n TextBlock children. Returns (module, [children])."""
    m = Module(name)
    children = [TextBlock(f"// {name}_{i}") for i in range(n_items)]
    for c in children:
        m.add(c)
    return m, children


def _basic_kwargs(num_loop_iter, *, per_iter_count=1):
    """Build a complete kwargs dict for build_idmap with synthetic modules.

    Returns (kwargs, leaves_by_category). leaves_by_category gives the
    expected leaf objects per category so tests can assert id-membership.
    """
    LRCodeA, LRCodeA_leaves = [], []
    LRCodeB, LRCodeB_leaves = [], []
    PackCodeA, PackCodeA_leaves = [], []
    PackCodeB, PackCodeB_leaves = [], []
    for u in range(num_loop_iter):
        m, lf = _mk_module(f"LRA{u}", per_iter_count); LRCodeA.append(m); LRCodeA_leaves.append(lf)
        m, lf = _mk_module(f"LRB{u}", per_iter_count); LRCodeB.append(m); LRCodeB_leaves.append(lf)
        m, lf = _mk_module(f"PackA{u}", per_iter_count); PackCodeA.append(m); PackCodeA_leaves.append(lf)
        m, lf = _mk_module(f"PackB{u}", per_iter_count); PackCodeB.append(m); PackCodeB_leaves.append(lf)

    flat_singletons = {}
    for cat in ("globalReadA", "globalReadB",
                "globalReadIncACode", "globalReadIncBCode",
                "localWriteA", "localWriteB",
                "LRSwapA", "LRSwapB", "LWSwapA", "LWSwapB",
                "loopCounterCode", "syncCode", "snopCode"):
        m, lf = _mk_module(cat, 1)
        flat_singletons[cat] = (m, lf)

    kwargs = dict(
        num_loop_iter=num_loop_iter,
        LRCodeA=LRCodeA, PackCodeA=PackCodeA,
        LRCodeB=LRCodeB, PackCodeB=PackCodeB,
        **{k: v[0] for k, v in flat_singletons.items()},
    )

    leaves = {
        f"LRA{u}":   LRCodeA_leaves[u]   for u in range(num_loop_iter)
    }
    for u in range(num_loop_iter):
        leaves[f"LRB{u}"]   = LRCodeB_leaves[u]
        leaves[f"PackA{u}"] = PackCodeA_leaves[u]
        leaves[f"PackB{u}"] = PackCodeB_leaves[u]
    leaves['GRA']    = flat_singletons['globalReadA'][1]
    leaves['GRB']    = flat_singletons['globalReadB'][1]
    leaves['GRIncA'] = flat_singletons['globalReadIncACode'][1]
    leaves['GRIncB'] = flat_singletons['globalReadIncBCode'][1]
    leaves['LWA']    = flat_singletons['localWriteA'][1]
    leaves['LWB']    = flat_singletons['localWriteB'][1]
    leaves['LRSA']   = flat_singletons['LRSwapA'][1]
    leaves['LRSB']   = flat_singletons['LRSwapB'][1]
    leaves['LWSA']   = flat_singletons['LWSwapA'][1]
    leaves['LWSB']   = flat_singletons['LWSwapB'][1]
    leaves['LCC']    = flat_singletons['loopCounterCode'][1]
    leaves['SYNC']   = flat_singletons['syncCode'][1]
    leaves['SNOP']   = flat_singletons['snopCode'][1]
    return kwargs, leaves


def test_categories_present():
    """All expected category keys exist for a typical num_loop_iter."""
    kwargs, _ = _basic_kwargs(num_loop_iter=2)
    idmap = build_idmap(**kwargs)
    expected = {
        'GRIncA', 'GRIncB', 'GRA', 'GRB',
        'LWA', 'LWB',
        'LRSA', 'LRSB', 'LWSA', 'LWSB',
        'LCC', 'SYNC', 'SNOP',
        'LRA0', 'LRA1', 'LRB0', 'LRB1',
        'PackA0', 'PackA1', 'PackB0', 'PackB1',
    }
    assert set(idmap) == expected


def test_per_iter_categories_scale():
    """num_loop_iter=4 produces LRA0..LRA3; num_loop_iter=2 produces only LRA0/LRA1."""
    kwargs2, _ = _basic_kwargs(num_loop_iter=2)
    kwargs4, _ = _basic_kwargs(num_loop_iter=4)
    idmap2 = build_idmap(**kwargs2)
    idmap4 = build_idmap(**kwargs4)
    assert {'LRA0', 'LRA1'} <= set(idmap2)
    assert 'LRA2' not in idmap2 and 'LRA3' not in idmap2
    assert {'LRA0', 'LRA1', 'LRA2', 'LRA3'} <= set(idmap4)


def test_invert_basic():
    """Every input instruction's id maps to its expected category."""
    kwargs, leaves = _basic_kwargs(num_loop_iter=2, per_iter_count=3)
    idmap = build_idmap(**kwargs)
    inv = invert_idmap_to_id_to_category(idmap)
    for cat, leaf_list in leaves.items():
        for leaf in leaf_list:
            assert inv[id(leaf)] == cat, f"leaf in {cat} mis-tagged as {inv.get(id(leaf))}"


def test_invert_handles_module_values():
    """Values that are Modules get walked via .flatitems()."""
    m, leaves = _mk_module("X", 3)
    inv = invert_idmap_to_id_to_category({"X": m})
    for leaf in leaves:
        assert inv[id(leaf)] == "X"


def test_invert_handles_list_values():
    """Values that are plain lists get walked directly."""
    leaves = [TextBlock(f"// l{i}") for i in range(3)]
    inv = invert_idmap_to_id_to_category({"Y": leaves})
    for leaf in leaves:
        assert inv[id(leaf)] == "Y"


def test_invert_handles_none_values():
    """None values are skipped without error."""
    inv = invert_idmap_to_id_to_category({"Z": None})
    assert inv == {}


def test_invert_raises_on_duplicate_ids():
    """Same instruction id under two categories must raise — schema bug."""
    leaf = TextBlock("// shared")
    idmap = {"A": [leaf], "B": [leaf]}
    with pytest.raises(ValueError, match="appears under both"):
        invert_idmap_to_id_to_category(idmap)


def test_invert_same_category_duplicate_is_fine():
    """Same id appearing twice under the SAME category is not a schema bug."""
    leaf = TextBlock("// dup")
    idmap = {"A": [leaf, leaf]}
    inv = invert_idmap_to_id_to_category(idmap)
    assert inv == {id(leaf): "A"}


def test_split_for_plr_preserves_item_identity():
    """split_for_plr returns 2 lists whose items are `is`-identical to the source."""
    m, leaves = _mk_module("plr", 6)
    new0, new1 = split_for_plr(m)
    # Original splitForPLR semantics: first half is iter 1, second half is iter 0
    assert new0 == leaves[3:]
    assert new1 == leaves[:3]
    for got, src in zip(new0, leaves[3:]):
        assert got is src
    for got, src in zip(new1, leaves[:3]):
        assert got is src


def test_split_for_plr_odd_count():
    """Odd-sized module: numInst//2 elements go to iter 1, the rest to iter 0."""
    m, leaves = _mk_module("plr_odd", 5)
    new0, new1 = split_for_plr(m)
    assert new1 == leaves[:2]
    assert new0 == leaves[2:]
