# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The collapse repair (lib/collapse_repair.py): which per-cell gate
findings it repairs, the tier keys it adds and the cells it carries, with a
fake audit set whose picks follow the bundle it is given."""

import json

import numpy as np
import pytest

from lib import collapse_repair as cr
from lib import whitelist_tier as wt

LEAF = "Large|Mid|MidK|Bany#M<=1024"
SIBLING = "Large|Mid|MidK|Bany#M>1024"
BASE = "Large|Mid|MidK|Bany"
OTHER = "Mid|Mid|MidK|Bany"
SHAPE = (1000, 200, 100, 64)
SHAPE2 = (900, 300, 120, 32)
KEY = ("cu128_cot", SHAPE)
KEY2 = ("exclusive", SHAPE2)
GOOD, BAD, ALSO = 7, 3, 9
H_GOOD, H_BAD, H_OTHER = 1001, 1003, 1005


def _gate(*tails, regressed=0, floor=0):
    groups = {}
    for group, shape, before, after in tails:
        g = groups.setdefault(group, {"protocol": "v2", "tail": []})
        g["tail"].append(
            {
                "shape": list(shape),
                "cell": BASE,
                "before": before,
                "after": after,
                "confirmed": True,
                "decides": True,
            }
        )
    return {
        "passed": False,
        "n_tail": len(tails),
        "n_regressed": regressed,
        "n_floor": floor,
        "groups": groups,
    }


def _bundle(tier, labels=(LEAF, SIBLING, OTHER), tag="cand"):
    return {
        "models": {
            label: {"view_key_hashes": list(tier.get(label, [H_BAD])), "tag": tag}
            for label in labels
        },
        "whitelist_tier": {"k": 1, "n_keys": sum(len(v) for v in tier.values())},
        "fallback_parents": [],
    }


class FakeAudit:
    """Audit GEMMs served by LEAF, each with its good kernel (`good`, KEY's
    GOOD by default): the bundle's pick is the good one when LEAF's tier
    holds its key, or when it serves without a tier and `ranks_good`; BAD
    otherwise; a carried cell (tag "ship") picks the good one."""

    def __init__(self, ranks_good=True, good=None):
        self.ranks_good = ranks_good
        self.good = dict(good or {KEY: GOOD})
        self.kernel = {GOOD: {"hash": H_GOOD}, BAD: {"hash": H_BAD}}
        self.kernel[ALSO] = {"hash": H_OTHER}
        self.calls = []

    def picks(self, data, keys):
        b = json.loads(data)
        self.calls.append(b)
        e = b["models"].get(LEAF)
        out = {}
        for key in keys:
            good = self.good[key]
            if e is None:
                leaf = sorted(k for k in b["models"] if k.startswith(BASE))[0]
                out[key] = cr.Pick(good, leaf)
            elif e["tag"] == "ship":
                out[key] = cr.Pick(good, LEAF)
            elif b["free"]:
                out[key] = cr.Pick(good if self.ranks_good else BAD, LEAF)
            else:
                held = self.kernel[good]["hash"] in e["view_key_hashes"]
                out[key] = cr.Pick(good if held else BAD, LEAF)
        return out

    def reading(self, key, sid):
        return (0.3 if sid == BAD else 0.9), "v2"

    def label(self, sid):
        return {GOOD: "GOOD", BAD: "BAD", ALSO: "ALSO"}.get(sid)


def pack(bundle):
    free = not bundle.get("serving", {}).get("whitelist_tier", True)
    return json.dumps({"models": bundle["models"], "free": free}).encode()


def _row(keys, costs, scores):
    return (
        0,
        np.array(keys, dtype=np.int64),
        np.array(costs, dtype=np.float64),
        np.array(scores, dtype=np.float64),
    )


INDEX = {H_BAD: 0, H_GOOD: 1, H_OTHER: 2}


def leaf_data(held_rows=()):
    """A leaf whose GEMMs the tier key H_BAD (index 0) wins; `held_rows`
    added to its held-out side."""

    def read(leaves):
        own = [(10.0, _row([0, 2], [10.0, 12.0], [2.0, 1.0])) for _ in range(5)]
        held = [(10.0, _row([0], [10.0], [1.0]))] + list(held_rows)
        return {leaf: (own, held) for leaf in leaves}, INDEX

    return read


def view_key_hash(kernel):
    return kernel["hash"]


def run(cand, ship, gate, audit, data=None, **kw):
    return cr.repair(
        cand,
        ship,
        cr.collapses_of(gate),
        audit,
        pack=pack,
        leaf_data=data or leaf_data(),
        view_key_hash=view_key_hash,
        **kw,
    )


def test_the_collapses_are_the_deciding_tail_picks_of_every_group():
    gate = _gate(("cu128_cot", SHAPE, 0.9, 0.3), ("exclusive", SHAPE, 0.8, 0.4))
    gate["groups"]["exclusive"]["tail"].append(
        {"shape": [1, 2, 3, 4], "before": 0.9, "after": 0.3, "decides": False}
    )
    gate["groups"]["cu64_cot"] = {"protocol": "v2", "reason": "does not pair"}
    got = cr.collapses_of(gate)
    assert [(c.group, c.shape, c.before, c.after) for c in got] == [
        ("cu128_cot", SHAPE, 0.9, 0.3),
        ("exclusive", SHAPE, 0.8, 0.4),
    ]
    assert got[0].text() == "cu128_cot 1000x200x100 batch 64"
    assert got[0].key == KEY
    assert cr.only_the_tail(gate)
    assert not cr.only_the_tail(_gate(("exclusive", SHAPE, 0.9, 0.4), regressed=1))
    assert not cr.only_the_tail(_gate(("exclusive", SHAPE, 0.9, 0.4), floor=1))
    assert not cr.only_the_tail({**gate, "passed": True})
    assert not cr.only_the_tail({**_gate(), "n_tail": 0})


def test_a_tier_that_cut_the_networks_own_pick_gets_its_key_back():
    cand = _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD]})
    ship = _bundle({BASE: [H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    out = run(cand, ship, _gate(("cu128_cot", SHAPE, 0.9, 0.3)), FakeAudit())
    assert out.bundle["models"][LEAF]["view_key_hashes"] == [H_BAD, H_GOOD]
    assert out.bundle["models"][SIBLING] == cand["models"][SIBLING]
    assert out.bundle["whitelist_tier"]["n_keys"] == 3
    rec = out.record
    (c,) = rec["collapses"]
    assert c["found"].startswith("tier-induced")
    assert (c["leaf"], c["pick"], c["unrestricted_pick"]) == (LEAF, "BAD", "GOOD")
    assert c["tier_add"] == H_GOOD and c["lifted_by_tier_add"] is True
    assert (c["repaired_pick"], c["repaired_reading"]) == ("GOOD", 0.9)
    assert rec["carried_cells"] == {} and rec["n_keys_added"] == 1
    assert rec["n_lifted_on_the_audit"] == 1
    (tried,) = rec["tier_adds"][LEAF]["tried"]
    assert tried["kept"] and tried["check"]["held_out"]["n_new_below_floor"] == 0
    # the candidate itself is not changed
    assert cand["models"][LEAF]["view_key_hashes"] == [H_BAD]


def test_a_ranking_error_carries_the_base_cell_from_the_shipped_bundle():
    cand = _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD]})
    ship = _bundle({BASE: [H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    out = run(
        cand,
        ship,
        _gate(("cu128_cot", SHAPE, 0.9, 0.3)),
        FakeAudit(ranks_good=False),
    )
    (c,) = out.record["collapses"]
    assert c["found"].startswith("a ranking error")
    assert c["carried"] == BASE and "tier_add" not in c
    assert sorted(out.bundle["models"]) == [BASE, OTHER]
    assert out.bundle["models"][BASE] is ship["models"][BASE]
    assert out.bundle["models"][OTHER] is cand["models"][OTHER]
    assert out.record["carried_cells"] == {
        BASE: {"removed": [LEAF, SIBLING], "carried": [BASE]}
    }
    assert out.bundle["whitelist_tier"]["n_keys"] == 2
    assert out.bundle["whitelist_tier"]["n_cells"] == 2
    assert c["repaired_pick"] == "GOOD" and out.record["n_lifted_on_the_audit"] == 1


def test_a_key_the_guard_refuses_leaves_the_cell_to_the_shipped_bundle():
    cand = _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD]})
    ship = _bundle({BASE: [H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    # a held-out GEMM of the leaf where the model would serve H_GOOD's key
    # at 2.5x the winner's time
    bad = (10.0, _row([0, 1], [10.0, 25.0], [1.0, 2.0]))
    out = run(
        cand,
        ship,
        _gate(("cu128_cot", SHAPE, 0.9, 0.3)),
        FakeAudit(),
        data=leaf_data([bad]),
    )
    (tried,) = out.record["tier_adds"][LEAF]["tried"]
    assert tried["kept"] is False and "1 on its held-out GEMMs" in tried["why"]
    assert out.record["tier_adds"][LEAF]["added"] == []
    assert out.record["carried_cells"][BASE]["carried"] == [BASE]
    (c,) = out.record["collapses"]
    assert c["carried"] == BASE and "tier_add" not in c
    assert sorted(out.bundle["models"]) == [BASE, OTHER]


def test_no_repair_without_a_collapse_or_a_model_to_carry():
    cand = _bundle({LEAF: [H_BAD]})
    out = run(cand, cand, {"groups": {}}, FakeAudit())
    assert out.bundle is None and out.record["outcome"] == "no collapse to repair"
    ship = _bundle({}, labels=(OTHER,), tag="ship")
    out = run(cand, ship, _gate(("cu128_cot", SHAPE, 0.9, 0.3)), FakeAudit(False))
    assert out.bundle is None
    assert out.record["outcome"].startswith("cannot repair: the bundle that ships")


def test_a_leaf_without_a_tier_is_carried():
    cand = _bundle({LEAF: []})
    ship = _bundle({BASE: [H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    out = run(cand, ship, _gate(("cu128_cot", SHAPE, 0.9, 0.3)), FakeAudit())
    (c,) = out.record["collapses"]
    assert c["found"] == "the leaf has no whitelist tier" and c["carried"] == BASE


def test_tier_keys_follow_the_bundles_own_sign_and_are_not_repeated():
    big = (1 << 63) + 5
    signed = {"models": {LEAF: {"view_key_hashes": [-3, 7]}}}
    out = cr.with_tier_keys(signed, {LEAF: [big, 7]})
    assert out["models"][LEAF]["view_key_hashes"] == [-3, 7, big - (1 << 64)]
    assert cr.tier_hashes(out["models"][LEAF])[-1] == big
    plain = {"models": {LEAF: {"view_key_hashes": [3]}}}
    assert cr.with_tier_keys(plain, {LEAF: [big]})["models"][LEAF][
        "view_key_hashes"
    ] == [3, big]
    assert signed["models"][LEAF]["view_key_hashes"] == [-3, 7]


def test_the_repaired_tier_doc_names_added_keys_and_carried_cells():
    cand_doc = {
        "k": 1,
        "cells": {
            LEAF: {"n_gemms": 5, "keys": [H_BAD]},
            SIBLING: {"n_gemms": 4, "keys": [H_GOOD]},
            OTHER: {"n_gemms": 3, "keys": [H_BAD]},
        },
    }
    ship_doc = {"cells": {BASE: {"n_gemms": 9, "keys": [H_GOOD]}}}
    bundle = cr.with_tier_keys(_bundle({LEAF: [H_BAD], OTHER: [H_BAD]}), {})
    bundle = cr.with_tier_keys(bundle, {OTHER: [H_GOOD]})
    carried, _ = cr.carry_cells(
        bundle, _bundle({BASE: [H_GOOD]}, labels=(BASE,), tag="ship"), [BASE]
    )
    rec = {"carried_cells": {BASE: {}}, "tier_adds": {OTHER: {"added": [H_GOOD]}}}
    doc = cr.tier_doc(cand_doc, ship_doc, rec, carried)
    assert sorted(doc["cells"]) == [BASE, OTHER]
    assert doc["cells"][BASE] == {
        "n_gemms": 9,
        "keys": [H_GOOD],
        cr.CARRIED: True,
    }
    assert doc["cells"][OTHER]["keys"] == [H_BAD, H_GOOD]
    assert doc["cells"][OTHER]["repair_keys"] == [H_GOOD]
    assert cr.tier_doc(None, ship_doc, rec, carried) is None


def test_carrying_needs_every_base_cell_in_the_shipped_bundle():
    with pytest.raises(ValueError, match="no model of Mid"):
        cr.carry_cells(_bundle({}), _bundle({}, labels=(BASE,)), [BASE, OTHER])


def test_the_guard_constants_agree_with_the_gate_and_the_orchestrator():
    from lib import audit as al
    import run_orchestrator as ro

    assert wt.GUARD_FLOOR == al.CELL_GATE_DEFAULTS["tail_floor"]
    assert wt.GUARD_MEAN_FLOOR == pytest.approx(
        1 - al.CELL_GATE_DEFAULTS["cell_tolerance"]
    )
    assert ro.WHITELIST_TIER_GUARD_KEYS == wt.GUARD_KEYS


def test_the_keys_a_leaf_keeps_hold_its_held_out_geomean_together():
    def slower(i):
        return (10.0, _row([0, i], [10.0, 10.2], [1.0, 2.0]))

    # each collapse's key alone costs the leaf's held-out picks x0.9934 of
    # their geomean, the two together x0.9869: the second is refused, its
    # collapse stays, and the base cell is carried
    cand = _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD]})
    ship = _bundle({BASE: [H_GOOD]}, labels=(BASE, OTHER), tag="ship")
    gate = _gate(("cu128_cot", SHAPE, 0.9, 0.3), ("exclusive", SHAPE2, 0.9, 0.3))
    audit = FakeAudit(good={KEY: GOOD, KEY2: ALSO})
    out = run(cand, ship, gate, audit, data=leaf_data([slower(1), slower(2)]))
    first, second = out.record["tier_adds"][LEAF]["tried"]
    assert (first["key"], first["kept"]) == (H_GOOD, True)
    assert (second["key"], second["kept"]) == (H_OTHER, False)
    assert "held-out picks' geomean x0.9869 < x0.99" in second["why"]
    assert second["check"]["held_out"]["geomean_ratio"] == pytest.approx(
        (10.0 / 10.2) ** (2 / 3)
    )
    assert out.record["tier_adds_dropped_by_carry"] == {LEAF: [H_GOOD]}
    assert out.record["carried_cells"][BASE]["carried"] == [BASE]
    assert sorted(out.bundle["models"]) == [BASE, OTHER]
    assert cr.kept_tier_adds(out.record) == {}


def test_a_cell_carried_under_the_candidates_labels_names_no_repair_keys():
    cand_doc = {
        "cells": {
            LEAF: {"keys": [H_BAD]},
            SIBLING: {"keys": [H_GOOD]},
            OTHER: {"keys": [H_BAD]},
        }
    }
    ship_doc = {"cells": {LEAF: {"keys": [H_OTHER]}, SIBLING: {"keys": [H_OTHER]}}}
    added = cr.with_tier_keys(
        _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD], OTHER: [H_BAD]}), {LEAF: [H_GOOD]}
    )
    ship = _bundle({LEAF: [H_OTHER], SIBLING: [H_OTHER]}, labels=(LEAF, SIBLING))
    carried, moved = cr.carry_cells(added, ship, [BASE])
    assert moved == {BASE: {"removed": [LEAF, SIBLING], "carried": [LEAF, SIBLING]}}
    rec = {
        "tier_adds": {LEAF: {"added": [H_GOOD]}},
        "carried_cells": moved,
        "tier_adds_dropped_by_carry": {LEAF: [H_GOOD]},
    }
    doc = cr.tier_doc(cand_doc, ship_doc, rec, carried)
    assert doc["cells"][LEAF] == {"keys": [H_OTHER], cr.CARRIED: True}
    assert doc["cells"][OTHER] == {"keys": [H_BAD]}
    assert cr.kept_tier_adds(rec) == {}


def _cells():
    return {
        "model_labels": [LEAF, OTHER, SIBLING],
        "fallback_parents": [],
        "n_cells_trained": 3,
        "n_subcells_trained": 2,
        "cells": [
            {"label": label, "best_sel_eff": 0.8, "n_whitelist_tier_keys": 1}
            for label in (LEAF, SIBLING, OTHER)
        ],
    }


def test_the_repaired_cells_json_keeps_the_entries_of_the_networks_it_serves():
    # a key added to OTHER, and BASE carried under the candidate's own labels
    bundle = cr.with_tier_keys(
        _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD], OTHER: [H_BAD]}), {OTHER: [H_GOOD]}
    )
    ship = _bundle({LEAF: [H_OTHER], SIBLING: [H_OTHER]}, labels=(LEAF, SIBLING))
    bundle, moved = cr.carry_cells(bundle, ship, [BASE])
    rec = {
        "outcome": "1 tier key(s) added in 1 leaf(s), 1 base cell(s) carried",
        "carried_cells": moved,
        "tier_adds": {OTHER: {"added": [H_GOOD]}},
    }
    cells = _cells()
    doc = cr.cells_doc(cells, bundle, rec)
    assert doc["model_labels"] == sorted([LEAF, SIBLING, OTHER])
    (entry,) = doc["cells"]
    assert entry["label"] == OTHER and entry["best_sel_eff"] == 0.8
    assert (entry["n_whitelist_tier_keys"], entry["n_repair_tier_keys"]) == (2, 1)
    assert (doc["n_cells_trained"], doc["n_subcells_trained"]) == (1, 0)
    assert doc["repair"] == {
        "schema": cr.SCHEMA,
        "outcome": rec["outcome"],
        "tier_keys_added": {OTHER: [H_GOOD]},
        "carried_cells": moved,
    }
    assert cells == _cells()
    # a carried cell under other labels: the candidate's leaves are gone
    other = _bundle({BASE: [H_GOOD]}, labels=(BASE,))
    bundle, moved = cr.carry_cells(
        _bundle({LEAF: [H_BAD], OTHER: [H_BAD]}), other, [BASE]
    )
    doc = cr.cells_doc(cells, bundle, {"carried_cells": moved})
    assert doc["model_labels"] == [BASE, OTHER]
    assert [c["label"] for c in doc["cells"]] == [OTHER]
    assert doc["cells"][0]["n_whitelist_tier_keys"] == 1
    assert "n_repair_tier_keys" not in doc["cells"][0]


def test_the_repaired_metrics_json_keeps_stage05s_numbers_and_the_bundles_tier():
    bundle = cr.with_tier_keys(
        _bundle({LEAF: [H_BAD], OTHER: [H_BAD]}), {LEAF: [H_GOOD]}
    )
    rec = {"outcome": "1 tier key(s) added", "tier_adds": {LEAF: {"added": [H_GOOD]}}}
    met = {
        "global_sel_eff": 0.8,
        "per_cell_sel_eff": {LEAF: 0.8, OTHER: 0.8},
        "whitelist_tier": {"k": 1, "n_keys": 2, "elapsed_s": 5.0},
        "fallback_parents": [],
    }
    doc = cr.metrics_doc(met, bundle, rec)
    assert doc["whitelist_tier"] == {"k": 1, "n_keys": 3, "elapsed_s": 5.0}
    assert (
        doc["global_sel_eff"] == 0.8
        and doc["per_cell_sel_eff"] == met["per_cell_sel_eff"]
    )
    assert doc["repair"]["tier_keys_added"] == {LEAF: [H_GOOD]}
    assert met["whitelist_tier"]["n_keys"] == 2


def test_a_cell_carried_under_the_same_labels_takes_its_origin_from_its_bundle(
    tmp_path,
):
    import run_orchestrator as ro
    from lib import run_bundles as rb

    run_dir = tmp_path / "run"
    shipped = run_dir / "round_5"
    final = run_dir / ro.FINAL_RETRAIN_DIR
    for d, who in ((shipped, "round_5"), (final, "final")):
        (d / "stage05").mkdir(parents=True)
        (d / "stage05" / rb.STAGE_MARKER).write_text(json.dumps({"finished": who}))
        cells = {
            "model_labels": [LEAF, OTHER, SIBLING],
            "cells": [{"label": x, "who": who} for x in (LEAF, SIBLING, OTHER)],
        }
        (d / "stage05" / "cells.json").write_text(json.dumps(cells))
    candidate = json.loads((final / "stage05" / "cells.json").read_text())
    bundle = _bundle({LEAF: [H_BAD], SIBLING: [H_GOOD], OTHER: [H_BAD]})
    ship = _bundle({LEAF: [H_OTHER], SIBLING: [H_OTHER]}, labels=(LEAF, SIBLING))
    bundle, moved = cr.carry_cells(bundle, ship, [BASE])
    doc = cr.cells_doc(candidate, bundle, {"carried_cells": moved})
    (final / "stage05" / "cells.json").write_text(json.dumps(doc))
    ro.record_cell_origins(
        final / "stage05", [shipped], (shipped, moved[BASE]["carried"])
    )
    rec = json.loads((final / "stage05" / rb.CELL_ORIGINS).read_text())
    assert sorted(rec["cells"]) == [LEAF, SIBLING]
    assert {o["bundle"] for o in rec["origins"].values()} == {"round_5"}
    origins = rb.cell_origins([shipped, final])
    assert {x: origins[x]["trained_in"] for x in (LEAF, SIBLING, OTHER)} == {
        LEAF: "round_5",
        SIBLING: "round_5",
        OTHER: ro.FINAL_RETRAIN_DIR,
    }
    assert origins[LEAF]["entry"] == {"label": LEAF, "who": "round_5"}
