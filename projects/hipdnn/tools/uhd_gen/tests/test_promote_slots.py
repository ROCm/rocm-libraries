# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Promotion into a slotted engine (RFC 0021): a ranker lands in its own slot and nowhere else."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("pandas")

from uhd_gen import promote
from uhd_gen.promote import PromoteError, _apply, build_plan
from uhd_gen.provenance import snapshot_provenance

UED = "1f2a3b4c-5d6e-4f70-8192-a3b4c5d6e7f8"
KMD_PREP = "a10c7e1d-5b3f-4c22-8d61-0e9f4a7b2c31"
KMD_DKDV = "b27e90a4-1d6c-4f58-9a03-7c2e5d81f640"
KMD_DQ = "d48a31bc-6e07-4d95-b1f2-3a8c0e5792d6"
MATCH = "8d4f3144-02a7-4ec2-b79c-e849951f8c24"
OLD = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
NEW = "cf37fa30-32dc-4a21-a008-68ef5e0d30a6"
DQ_BOUND = "edc1d5b4-6f12-4a40-a749-403966474bc9"
FEATURES_HASH = "sha256:0123456789abcdef"


@pytest.fixture(autouse=True)
def _feature_evaluator(monkeypatch):
    """Promotion asks the built evaluator for the feature hash and semantics; no build
    is needed to test where a model installs."""
    monkeypatch.setattr(promote, "compute_features_hash", lambda *args: FEATURES_HASH)
    monkeypatch.setattr(promote, "evaluator_feature_semantics_revision", lambda _: 1)


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fields(*names):
    return [{"name": name, "type": "int"} for name in names]


def _tree(root):
    """dkdv exposes block_m and num_warps; dq's KMD has num_warps but exposes only block_m."""
    _write(
        root / "engine.ued.json",
        {
            "version": "1.1",
            "id": UED,
            "name": "test:slotted",
            "slots": [
                {"name": "prep", "metadata": KMD_PREP},
                {
                    "name": "dkdv",
                    "metadata": KMD_DKDV,
                    "knobs": ["block_m", "num_warps"],
                    "sort_kernel_catalog": {"gfx942": OLD},
                },
                {
                    "name": "dq",
                    "metadata": KMD_DQ,
                    "knobs": ["block_m"],
                    "sort_kernel_catalog": {"gfx942": DQ_BOUND},
                },
            ],
        },
    )
    _write(root / "prep.kmd.json", {"version": "1.0", "id": KMD_PREP, "fields": []})
    for kmd, identity in ((KMD_DKDV, "dkdv"), (KMD_DQ, "dq")):
        _write(
            root / f"{identity}.kmd.json",
            {
                "version": "1.0",
                "id": kmd,
                "fields": _fields("block_m", "num_warps"),
            },
        )
    _write(
        root / "match.umd.json",
        {"version": "1.0", "id": MATCH, "scope": "kernel", "match_symbol": "t.match"},
    )
    for number, slot in enumerate(("prep", "dkdv", "dq")):
        _write(
            root / f"{slot}.kdp.json",
            {
                "version": "1.1",
                "id": f"{number:02d}be5fe7-02a7-4ec2-b79c-e849951f8c24",
                "engine": UED,
                "slot": slot,
                "arch": ["gfx942"],
                "matchers": [MATCH],
            },
        )
    for identity in (OLD, DQ_BOUND):
        _write(
            root / f"{identity}.uhd.json",
            {
                "version": "1.1",
                "id": identity,
                "name": "installed",
                "adapter": "static_order",
                "static_order": {},
            },
        )
    return root


def _model(
    root,
    tree,
    *,
    slot="dkdv",
    signature=("$kernel.block_m",),
    provenance=None,
    version="1.1",
    metric=None,
):
    if provenance is None:
        provenance = snapshot_provenance(tree, arch="gfx942", slot=slot)
    document = {
        "version": version,
        "id": NEW,
        "name": "model",
        "adapter": "native",
        "objective": "max",
        "features_signature": list(signature),
        "features_hash": FEATURES_HASH,
        "trained_against": provenance,
        "native": {"symbol": "t.score"},
    }
    if metric:
        document["score"] = {"metric": metric, "calibrated": True}
    _write(root / "heuristic.uhd.json", document)
    _write(
        root / "train_manifest.json",
        {"training_arches": ["gfx942"], "trained_against": provenance},
    )
    return root


def test_a_ranker_installs_into_its_own_slot_only(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    model = _model(tmp_path / "model", tree, slot="dkdv")
    plan = build_plan(model, tree)
    _apply(plan)
    expected = copy.deepcopy(original)
    expected["slots"][1]["sort_kernel_catalog"]["gfx942"] = [NEW]
    assert _read(tree / "engine.ued.json") == expected
    assert plan.slot == "dkdv"
    assert plan.old_heuristic == OLD
    # The model lives under its slot's directory, so another slot's ranker cannot collide.
    assert plan.destination_descriptor.parent == (
        tree / "heuristics" / UED / "dkdv" / "sort_kernel_catalog" / "gfx942"
    )


def test_a_slot_ranker_that_declares_a_metric_is_refused(tmp_path):
    """The loader drops the whole engine for a metric-declaring slot ranker (RFC 0021 §7)."""
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree, slot="dkdv", metric="tflops")
    with pytest.raises(PromoteError, match="a slot ranker must be metric-less"):
        build_plan(model, tree)


def test_the_slot_a_model_records_decides_where_it_installs(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    _apply(build_plan(_model(tmp_path / "model", tree, slot="dq"), tree))
    expected = copy.deepcopy(original)
    expected["slots"][2]["sort_kernel_catalog"]["gfx942"] = [NEW]
    assert _read(tree / "engine.ued.json") == expected


def test_a_slot_with_no_ranker_gains_one(tmp_path):
    tree = _tree(tmp_path / "tree")
    _apply(
        build_plan(
            _model(tmp_path / "model", tree, slot="prep", signature=("$graph.work",)),
            tree,
        )
    )
    prep = _read(tree / "engine.ued.json")["slots"][0]
    assert prep["sort_kernel_catalog"] == {"gfx942": [NEW]}


def test_a_slotted_engine_refuses_a_ranker_that_names_no_slot(tmp_path):
    tree = _tree(tmp_path / "tree")
    provenance = snapshot_provenance(tree, arch="gfx942", slot="dkdv")
    del provenance["slot"]
    with pytest.raises(PromoteError, match="must record trained_against.slot"):
        build_plan(_model(tmp_path / "model", tree, provenance=provenance), tree)


def test_a_ranker_for_an_undeclared_slot_is_refused(tmp_path):
    tree = _tree(tmp_path / "tree")
    provenance = snapshot_provenance(tree, arch="gfx942", slot="dkdv")
    provenance["slot"] = "other"
    with pytest.raises(
        PromoteError, match="slot 'other' is not a slot of test:slotted"
    ):
        build_plan(_model(tmp_path / "model", tree, provenance=provenance), tree)


def test_a_ranker_trained_against_another_slots_kmd_is_refused(tmp_path):
    tree = _tree(tmp_path / "tree")
    provenance = snapshot_provenance(tree, arch="gfx942", slot="dq")
    provenance["slot"] = "dkdv"
    with pytest.raises(PromoteError, match="trained_against.kmd"):
        build_plan(_model(tmp_path / "model", tree, provenance=provenance), tree)


def test_the_slots_have_no_predict_applicable_kernels_role(tmp_path):
    tree = _tree(tmp_path / "tree")
    with pytest.raises(
        PromoteError, match="declares no engine-level predict_applicable_kernels"
    ):
        build_plan(
            _model(tmp_path / "model", tree),
            tree,
            role="predict_applicable_kernels",
        )


@pytest.mark.parametrize(
    "slot,signature,refusal",
    [
        # num_warps is a knob of dkdv but not of dq, though dq's KMD declares the field.
        (
            "dq",
            ("$kernel.num_warps",),
            r"slot 'dq' of the UED does not expose as knobs",
        ),
        ("dq", ("$kernel.tile",), r"KMD of slot 'dq' does not declare as fields"),
        (
            "prep",
            ("$kernel.block_m",),
            r"KMD of slot 'prep' does not declare as fields",
        ),
    ],
)
def test_check_two_runs_against_the_models_own_slot(tmp_path, slot, signature, refusal):
    tree = _tree(tmp_path / "tree")
    with pytest.raises(PromoteError, match=refusal):
        build_plan(
            _model(tmp_path / "model", tree, slot=slot, signature=signature), tree
        )


def test_the_same_axis_is_admitted_by_the_slot_that_exposes_it(tmp_path):
    tree = _tree(tmp_path / "tree")
    plan = build_plan(
        _model(tmp_path / "model", tree, slot="dkdv", signature=("$kernel.num_warps",)),
        tree,
    )
    assert plan.slot == "dkdv"


def test_remove_knob_edits_the_slots_knobs_and_bumps_the_ued_revision(tmp_path):
    tree = _tree(tmp_path / "tree")
    provenance = snapshot_provenance(tree, arch="gfx942", slot="dkdv")
    provenance["ued"]["revision"] = "2.0"
    model = _model(tmp_path / "model", tree, provenance=provenance)
    plan = build_plan(model, tree, remove_knobs=["num_warps"])
    _apply(plan)
    ued = _read(tree / "engine.ued.json")
    assert plan.dropped_knobs == ["num_warps"]
    assert ued["revision"] == "2.0"
    assert ued["slots"][1]["knobs"] == ["block_m"]
    # dq's knobs are a different namespace and stay as authored.
    assert ued["slots"][2]["knobs"] == ["block_m"]


def test_a_knob_only_another_slot_exposes_cannot_be_removed_from_this_one(tmp_path):
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued["slots"][2]["knobs"].append("dq_only")
    _write(tree / "engine.ued.json", ued)
    provenance = snapshot_provenance(tree, arch="gfx942", slot="dkdv")
    provenance["ued"]["revision"] = "2.0"
    with pytest.raises(
        PromoteError, match=r"cannot remove unauthored knobs: \['dq_only'\]"
    ):
        build_plan(
            _model(tmp_path / "model", tree, provenance=provenance),
            tree,
            remove_knobs=["dq_only"],
        )


def test_knob_removal_is_refused_for_a_slot_model_trained_against_the_old_ued(tmp_path):
    tree = _tree(tmp_path / "tree")
    with pytest.raises(PromoteError, match="trained_against.ued: .* incompatible"):
        build_plan(_model(tmp_path / "model", tree), tree, remove_knobs=["num_warps"])


def _engine_level_model(root, tree, signature):
    provenance = {**snapshot_provenance(tree, arch="gfx942"), "selector_revision": "r1"}
    _model(
        root,
        tree,
        signature=signature,
        provenance=provenance,
        metric="tflops",
    )
    return root


def test_an_engine_level_model_installs_at_the_engine_not_in_a_slot(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    model = _engine_level_model(tmp_path / "model", tree, ("$graph.work",))
    plan = build_plan(model, tree, role="predict_engine")
    _apply(plan)
    expected = copy.deepcopy(original)
    expected["predict_engine"] = {"gfx942": [NEW]}
    assert _read(tree / "engine.ued.json") == expected
    assert plan.slot is None


def test_an_engine_level_model_cannot_read_kernel_features(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _engine_level_model(tmp_path / "model", tree, ("$kernel.block_m",))
    with pytest.raises(PromoteError, match="cannot depend on kernel"):
        build_plan(model, tree, role="predict_engine")


def test_an_engine_level_model_must_name_every_slot_kmd(tmp_path):
    tree = _tree(tmp_path / "tree")
    ranker_form = snapshot_provenance(tree, arch="gfx942", slot="dkdv")
    for provenance, refusal in (
        (ranker_form, "names a single slot, but a slotted engine's predict_engine"),
        (
            {k: v for k, v in ranker_form.items() if k != "slot"},
            "names kmd, but a slotted engine's predict_engine",
        ),
    ):
        model = _model(
            tmp_path / "model",
            tree,
            signature=("$graph.work",),
            provenance={**provenance, "selector_revision": "r1"},
            metric="tflops",
        )
        with pytest.raises(PromoteError, match=refusal):
            build_plan(model, tree, role="predict_engine")


def test_a_1_0_uhd_cannot_carry_slot_provenance(tmp_path):
    tree = _tree(tmp_path / "tree")
    with pytest.raises(
        PromoteError, match="slot provenance .* requires UHD version 1.1"
    ):
        build_plan(_model(tmp_path / "model", tree, version="1.0"), tree)


def test_an_unslotted_engine_refuses_a_slot_ranker(tmp_path):
    tree = tmp_path / "tree"
    slotted_provenance = snapshot_provenance(
        _tree(tmp_path / "scratch"), arch="gfx942", slot="dkdv"
    )
    _write(
        tree / "engine.ued.json",
        {"version": "1.0", "id": UED, "name": "test:plain", "metadata": KMD_DKDV},
    )
    _write(
        tree / "dkdv.kmd.json",
        {"version": "1.0", "id": KMD_DKDV, "fields": _fields("block_m")},
    )
    with pytest.raises(PromoteError, match="declares no slots"):
        build_plan(
            _model(
                tmp_path / "model",
                tree,
                provenance=slotted_provenance,
                signature=("$graph.work",),
            ),
            tree,
        )


def _plain_tree(root):
    _write(
        root / "engine.ued.json",
        {
            "version": "1.0",
            "id": UED,
            "name": "test:plain",
            "metadata": KMD_DKDV,
            "knobs": ["block_m", "num_warps"],
            "sort_kernel_catalog": {"gfx942": OLD},
        },
    )
    _write(
        root / "dkdv.kmd.json",
        {"version": "1.0", "id": KMD_DKDV, "fields": _fields("block_m", "num_warps")},
    )
    _write(
        root / "match.umd.json",
        {"version": "1.0", "id": MATCH, "scope": "kernel", "match_symbol": "t.match"},
    )
    _write(
        root / "plain.kdp.json",
        {
            "version": "1.0",
            "id": "00be5fe7-02a7-4ec2-b79c-e849951f8c24",
            "engine": UED,
            "arch": ["gfx942"],
            "matchers": [MATCH],
        },
    )
    _write(
        root / "old.uhd.json",
        {
            "version": "1.0",
            "id": OLD,
            "name": "installed",
            "adapter": "static_order",
            "static_order": {},
        },
    )
    return root


def test_an_unslotted_engine_binds_and_trims_knobs_at_the_engine_level(tmp_path):
    tree = _plain_tree(tmp_path / "tree")
    provenance = snapshot_provenance(tree, arch="gfx942")
    assert "kmd" in provenance and "slot" not in provenance
    provenance["ued"]["revision"] = "2.0"
    model = _model(tmp_path / "model", tree, provenance=provenance, version="1.0")
    plan = build_plan(model, tree, remove_knobs=["num_warps"])
    _apply(plan)
    ued = _read(tree / "engine.ued.json")
    assert ued["sort_kernel_catalog"] == {"gfx942": [NEW]}
    assert ued["knobs"] == ["block_m"] and ued["revision"] == "2.0"
    assert plan.slot is None
    assert plan.destination_descriptor.parent == (
        tree / "heuristics" / UED / "sort_kernel_catalog" / "gfx942"
    )
