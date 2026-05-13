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
    build_id_to_category_per_iter,
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


def test_per_iter_helper_splits_grinc_from_gr():
    """build_id_to_category_per_iter must distinguish GR-inc items
    (tagged GRIncA/GRIncB) from GR-load items (tagged generic GR or
    GRA/GRB if globalReadA/B is supplied). Without this split, GR-inc
    instructions get lumped under 'GR' on the NLL path and disagree
    with the CMS path's idMap['GRIncA'/'GRIncB']. Items in the inc
    modules are SHARED with items in globalReadCode (perIterGlobalRead)
    via SIA.py:732, so id() lookup against the more-specific tag wins.
    """
    # Build a small synthetic globalReadCode that contains some inc items
    # AND some pure load items. Inc items also live in their own modules.
    m_gri_a, gri_a_leaves = _mk_module("globalReadIncrementA", 2)
    m_gri_b, gri_b_leaves = _mk_module("globalReadIncrementB", 2)
    m_load_a, load_a_leaves = _mk_module("globalReadA", 2)
    m_load_b, load_b_leaves = _mk_module("globalReadB", 2)
    # globalReadCode = the combined per-iter view (inc items appear here too).
    perIterGR = Module("perIterGR")
    for it in load_a_leaves + load_b_leaves + gri_a_leaves + gri_b_leaves:
        perIterGR.add(it)

    out = build_id_to_category_per_iter(
        subiter=0,
        localReadCode=None, localWriteCode=None,
        globalReadCode=perIterGR,
        packCode=None, packPreCode=None,
        globalReadA=m_load_a, globalReadB=m_load_b,
        globalReadIncACode=m_gri_a, globalReadIncBCode=m_gri_b,
    )
    for leaf in gri_a_leaves:
        assert out[id(leaf)] == "GRIncA", f"{leaf} should be GRIncA"
    for leaf in gri_b_leaves:
        assert out[id(leaf)] == "GRIncB"
    for leaf in load_a_leaves:
        assert out[id(leaf)] == "GRA"
    for leaf in load_b_leaves:
        assert out[id(leaf)] == "GRB"


def test_per_iter_helper_falls_back_to_generic_gr():
    """When per-side modules are NOT supplied (caller doesn't have them),
    items in globalReadCode get the generic 'GR' tag — the legacy
    behavior, preserved for back-compat."""
    perIterGR, leaves = _mk_module("perIterGR", 3)
    out = build_id_to_category_per_iter(
        subiter=0,
        localReadCode=None, localWriteCode=None,
        globalReadCode=perIterGR,
        packCode=None, packPreCode=None,
    )
    for leaf in leaves:
        assert out[id(leaf)] == "GR"


def test_per_iter_helper_grinc_overrides_gr_when_shared():
    """If a leaf appears in BOTH globalReadCode AND globalReadIncACode
    (the realistic case), the GRIncA tag must win — it's set first."""
    shared = TextBlock("// shared")
    m_gri_a = Module("globalReadIncrementA"); m_gri_a.add(shared)
    perIterGR = Module("perIterGR"); perIterGR.add(shared)
    out = build_id_to_category_per_iter(
        subiter=0,
        localReadCode=None, localWriteCode=None,
        globalReadCode=perIterGR,
        packCode=None, packPreCode=None,
        globalReadIncACode=m_gri_a,
    )
    assert out[id(shared)] == "GRIncA"


# ---------------------------------------------------------------------------
# Conformance: build_idmap and build_id_to_category_per_iter agree on
# shared categories for shared leaves.
#
# These tests construct inputs in BOTH shapes that wrap the SAME Python
# leaf objects, so id()-based comparison reveals any disagreement on the
# category strings each factory assigns. Categories shared by both:
#   GRIncA, GRIncB, GRA, GRB, LRA{u}, LRB{u}, PackA{u}, PackB{u}.
# ---------------------------------------------------------------------------

def _build_shared_input_for_subiter(subiter):
    """Build inputs in both shapes for one subiter; return (kwargs_for_idmap_subset,
    kwargs_for_per_iter, leaves_by_category).

    The leaves in LRCodeA[subiter] are the SAME Python objects that appear
    inside localReadCode.findNamedItem(f"LocalReadDoA_I{iui}"); ditto for B,
    PackA/B, and the GR-inc / GR-load leaves.
    """
    # Leaves for GR-inc (one per side) — shared between both factories.
    grinc_a_leaves = [TextBlock("// grinc_a_0"), TextBlock("// grinc_a_1")]
    grinc_b_leaves = [TextBlock("// grinc_b_0")]
    # Leaves for GR-load (one per side).
    gra_leaves = [TextBlock("// gra_0"), TextBlock("// gra_1")]
    grb_leaves = [TextBlock("// grb_0")]

    # Per-side LR leaves for this subiter — one inner-unroll worth.
    lra_leaves = [TextBlock(f"// lra_s{subiter}_iui0_0"),
                  TextBlock(f"// lra_s{subiter}_iui0_1")]
    lrb_leaves = [TextBlock(f"// lrb_s{subiter}_iui0_0")]
    packa_leaves = [TextBlock(f"// packa_s{subiter}_iui0_0")]
    packb_leaves = [TextBlock(f"// packb_s{subiter}_iui0_0")]

    # --- Shape 1: per-category source modules for build_idmap ---
    def _wrap(name, leaves):
        m = Module(name)
        for lf in leaves:
            m.add(lf)
        return m

    grinc_a_mod = _wrap("globalReadIncrementA", grinc_a_leaves)
    grinc_b_mod = _wrap("globalReadIncrementB", grinc_b_leaves)
    gra_mod = _wrap("globalReadA", gra_leaves)
    grb_mod = _wrap("globalReadB", grb_leaves)
    lra_per_iter_mod = _wrap(f"LRA{subiter}", lra_leaves)
    lrb_per_iter_mod = _wrap(f"LRB{subiter}", lrb_leaves)
    packa_per_iter_mod = _wrap(f"PackA{subiter}", packa_leaves)
    packb_per_iter_mod = _wrap(f"PackB{subiter}", packb_leaves)

    # --- Shape 2: combined named-submodule view for build_id_to_category_per_iter ---
    # Outer localReadCode contains LocalReadDoA_I0 / LocalReadDoB_I0 sub-modules
    # holding the SAME leaf objects.
    local_read_code = Module("localReadCode")
    lra_sub = Module("LocalReadDoA_I0")
    for lf in lra_leaves:
        lra_sub.add(lf)
    local_read_code.add(lra_sub)
    lrb_sub = Module("LocalReadDoB_I0")
    for lf in lrb_leaves:
        lrb_sub.add(lf)
    local_read_code.add(lrb_sub)

    # Outer packCode with packA_I0 / packB_I0.
    pack_code = Module("packCode")
    packa_sub = Module("packA_I0")
    for lf in packa_leaves:
        packa_sub.add(lf)
    pack_code.add(packa_sub)
    packb_sub = Module("packB_I0")
    for lf in packb_leaves:
        packb_sub.add(lf)
    pack_code.add(packb_sub)

    # Outer globalReadCode (per-iter view): includes both inc and load leaves
    # so that the 'GR' fallback would tag them generically — the per-side
    # tags should win first.
    per_iter_gr = Module("perIterGR")
    for lf in gra_leaves + grb_leaves + grinc_a_leaves + grinc_b_leaves:
        per_iter_gr.add(lf)

    leaves = {
        "GRIncA": grinc_a_leaves,
        "GRIncB": grinc_b_leaves,
        "GRA":    gra_leaves,
        "GRB":    grb_leaves,
        f"LRA{subiter}":   lra_leaves,
        f"LRB{subiter}":   lrb_leaves,
        f"PackA{subiter}": packa_leaves,
        f"PackB{subiter}": packb_leaves,
    }

    # build_idmap kwargs (shape 1) — a single-iter slice for the subiter under test.
    # build_idmap takes per-iter LISTS indexed by u in [0, num_loop_iter); we
    # cover the conformance for a chosen subiter by parameterizing num_loop_iter
    # appropriately and only populating index `subiter` with our wrapper modules.
    # For simplicity we use num_loop_iter=subiter+1 and put empty modules in the
    # other slots.
    LRCodeA = [Module(f"LRA{u}") for u in range(subiter + 1)]
    LRCodeB = [Module(f"LRB{u}") for u in range(subiter + 1)]
    PackCodeA = [Module(f"PackA{u}") for u in range(subiter + 1)]
    PackCodeB = [Module(f"PackB{u}") for u in range(subiter + 1)]
    LRCodeA[subiter] = lra_per_iter_mod
    LRCodeB[subiter] = lrb_per_iter_mod
    PackCodeA[subiter] = packa_per_iter_mod
    PackCodeB[subiter] = packb_per_iter_mod

    idmap_kwargs = dict(
        num_loop_iter=subiter + 1,
        LRCodeA=LRCodeA, PackCodeA=PackCodeA,
        LRCodeB=LRCodeB, PackCodeB=PackCodeB,
        globalReadA=gra_mod, globalReadB=grb_mod,
        globalReadIncACode=grinc_a_mod, globalReadIncBCode=grinc_b_mod,
        localWriteA=Module("localWriteA"), localWriteB=Module("localWriteB"),
        LRSwapA=Module("LRSwapA"), LRSwapB=Module("LRSwapB"),
        LWSwapA=Module("LWSwapA"), LWSwapB=Module("LWSwapB"),
        loopCounterCode=Module("loopCounterCode"),
        syncCode=Module("syncCode"), snopCode=Module("snopCode"),
    )

    per_iter_kwargs = dict(
        subiter=subiter,
        localReadCode=local_read_code, localWriteCode=None,
        globalReadCode=per_iter_gr,
        packCode=pack_code, packPreCode=None,
        globalReadA=gra_mod, globalReadB=grb_mod,
        globalReadIncACode=grinc_a_mod, globalReadIncBCode=grinc_b_mod,
    )
    return idmap_kwargs, per_iter_kwargs, leaves


def test_factories_agree_on_shared_categories_subiter0():
    """Conformance: build_idmap and build_id_to_category_per_iter must
    assign IDENTICAL category strings for every leaf they share, when
    given inputs in their respective shapes that wrap the same leaves.

    Shared categories: GRIncA, GRIncB, GRA, GRB, LRA0, LRB0, PackA0, PackB0.
    """
    idmap_kwargs, per_iter_kwargs, leaves = _build_shared_input_for_subiter(0)
    inv = invert_idmap_to_id_to_category(build_idmap(**idmap_kwargs))
    per_iter = build_id_to_category_per_iter(**per_iter_kwargs)

    shared_cats = {"GRIncA", "GRIncB", "GRA", "GRB",
                   "LRA0", "LRB0", "PackA0", "PackB0"}
    for cat in shared_cats:
        for leaf in leaves[cat]:
            a = inv.get(id(leaf))
            b = per_iter.get(id(leaf))
            assert a == cat, f"build_idmap tagged {leaf!r} as {a!r}, expected {cat!r}"
            assert b == cat, f"build_id_to_category_per_iter tagged {leaf!r} as {b!r}, expected {cat!r}"
            assert a == b, (
                f"factories disagree on leaf {leaf!r}: "
                f"build_idmap={a!r}, build_id_to_category_per_iter={b!r}"
            )


def test_build_idmap_accepts_sparse_mx_kwargs():
    """build_idmap must accept LRCodeMXSA, LRCodeMXSB, LRCodeMetadata kwargs
    (as per-iter lists of modules, defaulting to empty/None) so future
    sparse-MX CMS code paths have a categorization slot ready.

    When provided, leaves should be tagged LRMXSA{u} / LRMXSB{u} /
    LRMetadata{u} — the same naming build_id_to_category_per_iter uses.
    """
    kwargs, _ = _basic_kwargs(num_loop_iter=2)

    # Build per-iter sparse-MX modules with one leaf each.
    LRCodeMXSA, mxsa_leaves = [], []
    LRCodeMXSB, mxsb_leaves = [], []
    LRCodeMetadata, metadata_leaves = [], []
    for u in range(2):
        m, lf = _mk_module(f"LRMXSA{u}", 1); LRCodeMXSA.append(m); mxsa_leaves.append(lf)
        m, lf = _mk_module(f"LRMXSB{u}", 1); LRCodeMXSB.append(m); mxsb_leaves.append(lf)
        m, lf = _mk_module(f"LRMetadata{u}", 1); LRCodeMetadata.append(m); metadata_leaves.append(lf)

    idmap = build_idmap(
        LRCodeMXSA=LRCodeMXSA,
        LRCodeMXSB=LRCodeMXSB,
        LRCodeMetadata=LRCodeMetadata,
        **kwargs,
    )
    inv = invert_idmap_to_id_to_category(idmap)
    for u in range(2):
        for leaf in mxsa_leaves[u]:
            assert inv[id(leaf)] == f"LRMXSA{u}"
        for leaf in mxsb_leaves[u]:
            assert inv[id(leaf)] == f"LRMXSB{u}"
        for leaf in metadata_leaves[u]:
            assert inv[id(leaf)] == f"LRMetadata{u}"


def test_build_idmap_sparse_mx_kwargs_default_empty():
    """If sparse-MX kwargs are NOT supplied, build_idmap behaves exactly
    as before — no LRMXSA / LRMXSB / LRMetadata categories appear."""
    kwargs, _ = _basic_kwargs(num_loop_iter=2)
    idmap = build_idmap(**kwargs)
    for cat in idmap:
        assert not cat.startswith("LRMXSA"), f"unexpected MX category {cat!r}"
        assert not cat.startswith("LRMXSB"), f"unexpected MX category {cat!r}"
        assert not cat.startswith("LRMetadata"), f"unexpected MX category {cat!r}"


def test_factories_agree_on_shared_categories_subiter2():
    """Same conformance check at subiter=2 — shared categories now include
    LRA2, LRB2, PackA2, PackB2 (per-iter-indexed)."""
    idmap_kwargs, per_iter_kwargs, leaves = _build_shared_input_for_subiter(2)
    inv = invert_idmap_to_id_to_category(build_idmap(**idmap_kwargs))
    per_iter = build_id_to_category_per_iter(**per_iter_kwargs)

    shared_cats = {"GRIncA", "GRIncB", "GRA", "GRB",
                   "LRA2", "LRB2", "PackA2", "PackB2"}
    for cat in shared_cats:
        for leaf in leaves[cat]:
            a = inv.get(id(leaf))
            b = per_iter.get(id(leaf))
            assert a == cat, f"build_idmap tagged {leaf!r} as {a!r}, expected {cat!r}"
            assert b == cat, f"build_id_to_category_per_iter tagged {leaf!r} as {b!r}, expected {cat!r}"
            assert a == b, (
                f"factories disagree on leaf {leaf!r}: "
                f"build_idmap={a!r}, build_id_to_category_per_iter={b!r}"
            )


def _build_sparse_mx_input_for_subiter(subiter):
    """Build sparse-MX inputs in BOTH shapes (per-iter list for build_idmap;
    LocalReadDoMXSA/MXSB/Metadata sub-modules for build_id_to_category_per_iter)
    that wrap the SAME leaf objects, so id() comparison reveals any drift in
    category-string naming between the two factories.

    Returns (idmap_kwargs, per_iter_kwargs, leaves_by_category).
    """
    mxsa_leaves = [TextBlock(f"// mxsa_s{subiter}_iui0_0"),
                   TextBlock(f"// mxsa_s{subiter}_iui0_1")]
    mxsb_leaves = [TextBlock(f"// mxsb_s{subiter}_iui0_0")]
    metadata_leaves = [TextBlock(f"// metadata_s{subiter}_iui0_0")]

    def _wrap(name, leaves):
        m = Module(name)
        for lf in leaves:
            m.add(lf)
        return m

    # Shape 1: per-iter list, slot `subiter` populated, others empty.
    LRCodeMXSA = [Module(f"LRMXSA{u}") for u in range(subiter + 1)]
    LRCodeMXSB = [Module(f"LRMXSB{u}") for u in range(subiter + 1)]
    LRCodeMetadata = [Module(f"LRMetadata{u}") for u in range(subiter + 1)]
    LRCodeMXSA[subiter] = _wrap(f"LRMXSA{subiter}", mxsa_leaves)
    LRCodeMXSB[subiter] = _wrap(f"LRMXSB{subiter}", mxsb_leaves)
    LRCodeMetadata[subiter] = _wrap(f"LRMetadata{subiter}", metadata_leaves)

    # Shape 2: outer localReadCode with named MXSA/MXSB/Metadata sub-modules.
    local_read_code = Module("localReadCode")
    sub_mxsa = Module("LocalReadDoMXSA_I0")
    for lf in mxsa_leaves:
        sub_mxsa.add(lf)
    local_read_code.add(sub_mxsa)
    sub_mxsb = Module("LocalReadDoMXSB_I0")
    for lf in mxsb_leaves:
        sub_mxsb.add(lf)
    local_read_code.add(sub_mxsb)
    sub_meta = Module("LocalReadDoMetadata_I0")
    for lf in metadata_leaves:
        sub_meta.add(lf)
    local_read_code.add(sub_meta)

    # Reuse the basic kwargs scaffolding for build_idmap, augmented with
    # sparse-MX kwargs. We only need num_loop_iter=subiter+1 since we're
    # checking one subiter at a time.
    base_kwargs, _ = _basic_kwargs(num_loop_iter=subiter + 1)
    idmap_kwargs = dict(base_kwargs,
                        LRCodeMXSA=LRCodeMXSA,
                        LRCodeMXSB=LRCodeMXSB,
                        LRCodeMetadata=LRCodeMetadata)

    per_iter_kwargs = dict(
        subiter=subiter,
        localReadCode=local_read_code, localWriteCode=None,
        globalReadCode=None,
        packCode=None, packPreCode=None,
    )

    leaves = {
        f"LRMXSA{subiter}": mxsa_leaves,
        f"LRMXSB{subiter}": mxsb_leaves,
        f"LRMetadata{subiter}": metadata_leaves,
    }
    return idmap_kwargs, per_iter_kwargs, leaves


def test_factories_agree_on_sparse_mx_categories_subiter0():
    """Conformance: build_idmap (with new sparse-MX kwargs) and
    build_id_to_category_per_iter must assign IDENTICAL category strings
    for sparse-MX local-read leaves: LRMXSA0, LRMXSB0, LRMetadata0."""
    idmap_kwargs, per_iter_kwargs, leaves = _build_sparse_mx_input_for_subiter(0)
    inv = invert_idmap_to_id_to_category(build_idmap(**idmap_kwargs))
    per_iter = build_id_to_category_per_iter(**per_iter_kwargs)

    for cat in ("LRMXSA0", "LRMXSB0", "LRMetadata0"):
        for leaf in leaves[cat]:
            a = inv.get(id(leaf))
            b = per_iter.get(id(leaf))
            assert a == cat, f"build_idmap tagged {leaf!r} as {a!r}, expected {cat!r}"
            assert b == cat, f"build_id_to_category_per_iter tagged {leaf!r} as {b!r}, expected {cat!r}"
            assert a == b, (
                f"factories disagree on sparse-MX leaf {leaf!r}: "
                f"build_idmap={a!r}, build_id_to_category_per_iter={b!r}"
            )


def test_factories_agree_on_sparse_mx_categories_subiter3():
    """Same sparse-MX conformance at subiter=3 — index propagation check."""
    idmap_kwargs, per_iter_kwargs, leaves = _build_sparse_mx_input_for_subiter(3)
    inv = invert_idmap_to_id_to_category(build_idmap(**idmap_kwargs))
    per_iter = build_id_to_category_per_iter(**per_iter_kwargs)

    for cat in ("LRMXSA3", "LRMXSB3", "LRMetadata3"):
        for leaf in leaves[cat]:
            a = inv.get(id(leaf))
            b = per_iter.get(id(leaf))
            assert a == cat, f"build_idmap tagged {leaf!r} as {a!r}, expected {cat!r}"
            assert b == cat, f"build_id_to_category_per_iter tagged {leaf!r} as {b!r}, expected {cat!r}"
            assert a == b, (
                f"factories disagree on sparse-MX leaf {leaf!r}: "
                f"build_idmap={a!r}, build_id_to_category_per_iter={b!r}"
            )
