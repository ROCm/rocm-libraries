# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor snapshots name real owned dependencies, never inferred versions."""
import copy
import json

import pytest

from uhd_gen.provenance import (
    ProvenanceError,
    compare_provenance,
    record_feature_semantics,
    require_feature_semantics,
    snapshot_provenance,
    validate_provenance,
)

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
MATCH_A = "11be5fe7-02a7-4ec2-b79c-e849951f8c24"
MATCH_B = "22be5fe7-02a7-4ec2-b79c-e849951f8c24"
PACK_A = "33be5fe7-02a7-4ec2-b79c-e849951f8c24"
PACK_B = "44be5fe7-02a7-4ec2-b79c-e849951f8c24"


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "1.0", **doc}), encoding="utf-8")


def _tree(root):
    _write(
        root / "engine.ued.json", {"id": UED, "name": "test:engine", "metadata": KMD}
    )
    _write(root / "metadata.kmd.json", {"id": KMD, "revision": "3.12"})
    _write(root / "a.umd.json", {"id": MATCH_A, "revision": "2.10"})
    _write(root / "b.umd.json", {"id": MATCH_B})
    _write(
        root / "a.kdp.json",
        {
            "id": PACK_A,
            "engine": UED,
            "arch": ["gfx942"],
            "matchers": [MATCH_B, MATCH_A],
        },
    )
    _write(
        root / "b.kdp.json",
        {"id": PACK_B, "engine": UED, "arch": ["gfx950"], "matchers": [MATCH_A]},
    )
    return root


def test_snapshot_resolves_actual_semantic_revisions_and_sorted_matchers(tmp_path):
    tree = _tree(tmp_path)
    assert snapshot_provenance(tree, "test:engine", "gfx942") == {
        "ued": {"id": UED, "revision": "1.0"},
        "kmd": {"id": KMD, "revision": "3.12"},
        "umd": [
            {"id": MATCH_A, "revision": "2.10"},
            {"id": MATCH_B, "revision": "1.0"},
        ],
    }
    assert snapshot_provenance(tree, arch="gfx950")["umd"] == [
        {"id": MATCH_A, "revision": "2.10"}
    ]


@pytest.mark.parametrize(
    "missing,kind", [("metadata.kmd.json", "KMD"), ("a.umd.json", "UMD")]
)
def test_missing_dependency_never_receives_an_invented_version(tmp_path, missing, kind):
    tree = _tree(tmp_path)
    (tree / missing).unlink()
    with pytest.raises(ProvenanceError, match=f"missing {kind} dependency"):
        snapshot_provenance(tree, arch="gfx942")


def test_missing_non_target_matcher_does_not_poison_target_snapshot(tmp_path):
    tree = _tree(tmp_path)
    (tree / "b.umd.json").unlink()
    assert snapshot_provenance(tree, arch="gfx950")["umd"] == [
        {"id": MATCH_A, "revision": "2.10"}
    ]


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_definitions_are_ambiguous_even_if_identical(tmp_path, conflicting):
    tree = _tree(tmp_path)
    _write(
        tree / "duplicate.kmd.json",
        {"id": KMD, "revision": "4.0" if conflicting else "3.12"},
    )
    with pytest.raises(ProvenanceError, match="duplicate/conflicting descriptor"):
        snapshot_provenance(tree)


def test_metadata_cannot_resolve_to_a_matcher_of_the_same_identity(tmp_path):
    tree = _tree(tmp_path)
    (tree / "metadata.kmd.json").unlink()
    _write(tree / "impostor.umd.json", {"id": KMD})
    with pytest.raises(ProvenanceError, match="missing KMD dependency"):
        snapshot_provenance(tree)


def test_a_compatible_minor_revision_preserves_the_model_contract(tmp_path):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx942")
    _write(tree / "a.umd.json", {"id": MATCH_A, "revision": "2.11"})
    compare_provenance(trained, snapshot_provenance(tree, arch="gfx942"))
    assert {"id": MATCH_A, "revision": "2.10"} in trained["umd"]


def test_a_matcher_the_model_never_saw_breaks_the_contract(tmp_path):
    """The runtime disables a model missing one of its arch's matchers
    (DescriptorLoader.hpp: "arch-relevant UMD is missing from training provenance")."""
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950")
    with pytest.raises(
        ProvenanceError, match=f"matcher {MATCH_B} serves this architecture"
    ):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx942"))


@pytest.mark.parametrize("kind", ["ued", "kmd", "umd"])
@pytest.mark.parametrize("change", ["identity", "major", "minor_regression"])
def test_incompatible_dependency_identity_or_revision_is_rejected(
    tmp_path, kind, change
):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree)
    actual = copy.deepcopy(trained)
    entry = actual[kind][0] if kind == "umd" else actual[kind]
    if change == "identity":
        entry["id"] = PACK_A
    elif change == "major":
        entry["revision"] = "99.0"
    else:
        original = trained[kind][0] if kind == "umd" else trained[kind]
        major, minor = map(int, original["revision"].split("."))
        original["revision"] = f"{major}.{minor + 1}"
    refusal = (
        "dependency .* is missing or not owned"
        if change == "identity"
        else ".* revision .* is incompatible with trained revision"
    )
    with pytest.raises(ProvenanceError, match=f"trained_against.{kind}: {refusal}"):
        compare_provenance(trained, actual)


def test_removed_matcher_is_a_contract_break_not_a_coverage_addition(tmp_path):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx942")
    with pytest.raises(
        ProvenanceError,
        match="trained_against.umd: dependency .* is missing or not owned",
    ):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx950"))


@pytest.mark.parametrize("revision", [None, "1", "1.2.3", 1.0, "a.b"])
def test_malformed_explicit_revision_does_not_default(tmp_path, revision):
    tree = _tree(tmp_path)
    _write(tree / "metadata.kmd.json", {"id": KMD, "revision": revision})
    with pytest.raises(
        ProvenanceError, match="expected semantic revision '<major>.<minor>'"
    ):
        snapshot_provenance(tree)


def test_snapshot_rejects_version_only_provenance():
    with pytest.raises(ProvenanceError, match="trained_against.umd must be an array"):
        validate_provenance({"ued": "1.0", "kmd": "1.0", "umd": "1.0"})


def test_an_engine_with_no_descriptors_is_trained_against_its_selector_revision():
    """RFC 0019 §4.1: an engine with no UED has only a provider build to name."""
    recorded = validate_provenance(
        {"selector_revision": "miopen-provider/0.2.0/aiter-fwd-1"}
    )
    assert recorded == {"selector_revision": "miopen-provider/0.2.0/aiter-fwd-1"}


@pytest.mark.parametrize(
    "snapshot,refusal",
    [
        ({}, "must name a descriptor set or a selector_revision"),
        ({"selector_revision": ""}, "selector_revision must be a non-empty string"),
        ({"selector_revision": 3}, "selector_revision must be a non-empty string"),
        (
            {"ued": {"id": UED, "revision": "1.0"}},
            "trained_against requires exactly ued, kmd and umd",
        ),
        (
            {"selector_revision": "rev", "unexpected": "value"},
            r"trained_against has unknown members: \['unexpected'\]",
        ),
    ],
)
def test_provenance_that_names_neither_form_completely_is_refused(snapshot, refusal):
    """A partial descriptor set or empty revision is unverifiable, not weaker."""
    with pytest.raises(ProvenanceError, match=refusal):
        validate_provenance(snapshot)


def test_a_stale_selector_revision_is_refused_rather_than_scored():
    """L1 is compared across engines, so a stale build changes which one wins."""
    trained = {"selector_revision": "aiter-fwd-1"}
    compare_provenance(trained, {"selector_revision": "aiter-fwd-1"})
    with pytest.raises(ProvenanceError, match="selector_revision"):
        compare_provenance(trained, {"selector_revision": "aiter-fwd-2"})


def test_feature_semantics_absent_means_revision_1_and_a_mismatch_names_both():
    shipped = {"selector_revision": "hip-kernel-provider/asm-sdpa-fwd/b162a5ffd743c21d"}
    require_feature_semantics(shipped, 1)
    require_feature_semantics(None, 1)
    with pytest.raises(ProvenanceError, match=r"revision 1\b.*revision 2\b"):
        require_feature_semantics(shipped, 2)
    newer = {**shipped, "feature_semantics_revision": 3}
    require_feature_semantics(newer, 3)
    with pytest.raises(ProvenanceError, match=r"revision 3\b.*revision 2\b"):
        require_feature_semantics(newer, 2)


def test_training_records_the_evaluators_revision_but_never_overwrites_another():
    snapshot = {"selector_revision": "provider-1"}
    assert record_feature_semantics(snapshot, 2) == {
        **snapshot,
        "feature_semantics_revision": 2,
    }
    assert (
        record_feature_semantics({**snapshot, "feature_semantics_revision": 2}, 2)[
            "feature_semantics_revision"
        ]
        == 2
    )
    with pytest.raises(ProvenanceError, match="feature_semantics_revision"):
        record_feature_semantics({**snapshot, "feature_semantics_revision": 1}, 2)


@pytest.mark.parametrize("recorded", [0, -1, True, 1.0, "1", 2**63])
def test_a_feature_semantics_revision_is_a_positive_int64(recorded):
    """Matches UhdParser.hpp: anything it refuses would train but never load."""
    with pytest.raises(ProvenanceError, match="feature_semantics_revision"):
        validate_provenance(
            {"selector_revision": "provider-1", "feature_semantics_revision": recorded}
        )


def test_feature_semantics_is_kept_by_both_provenance_forms():
    descriptor_set = {
        "ued": {"id": UED, "revision": "1.0"},
        "kmd": {"id": KMD, "revision": "1.0"},
        "umd": [],
        "feature_semantics_revision": 4,
    }
    assert validate_provenance(descriptor_set)["feature_semantics_revision"] == 4
    assert validate_provenance(
        {"selector_revision": "r", "feature_semantics_revision": 4}
    ) == {"selector_revision": "r", "feature_semantics_revision": 4}
    # Never on its own: it says what features meant, not what the model binds to.
    with pytest.raises(
        ProvenanceError, match="must name a descriptor set or a selector_revision"
    ):
        validate_provenance({"feature_semantics_revision": 1})


# --- RFC 0021 kernel slots ---------------------------------------------------------------

KMD_PREP = "a10c7e1d-5b3f-4c22-8d61-0e9f4a7b2c31"
KMD_DKDV = "b27e90a4-1d6c-4f58-9a03-7c2e5d81f640"
KMD_DQ = "d48a31bc-6e07-4d95-b1f2-3a8c0e5792d6"
MATCH_PREP = "5a1c0e11-02a7-4ec2-b79c-e849951f8c24"
MATCH_DKDV = "6b2d1f22-02a7-4ec2-b79c-e849951f8c24"
MATCH_DQ = "7c3e2033-02a7-4ec2-b79c-e849951f8c24"
MATCH_SHARED = "8d4f3144-02a7-4ec2-b79c-e849951f8c24"
MATCH_OTHER_ARCH = "9e505255-02a7-4ec2-b79c-e849951f8c24"
UED_SLOTTED = "1f2a3b4c-5d6e-4f70-8192-a3b4c5d6e7f8"
SLOTS = ("prep", "dkdv", "dq")


def _slotted_tree(root):
    """A three-slot engine: UED/KDPs are 1.1, the dependencies they name stay 1.0."""
    _write(
        root / "engine.ued.json",
        {
            "version": "1.1",
            "id": UED_SLOTTED,
            "name": "test:slotted",
            "slots": [
                {"name": "prep", "metadata": KMD_PREP},
                {"name": "dkdv", "metadata": KMD_DKDV, "knobs": ["block_m"]},
                {"name": "dq", "metadata": KMD_DQ, "knobs": ["block_m"]},
            ],
        },
    )
    for kmd, revision_ in ((KMD_PREP, "1.0"), (KMD_DKDV, "2.3"), (KMD_DQ, "1.4")):
        _write(root / f"{kmd}.kmd.json", {"id": kmd, "revision": revision_})
    for matcher in (MATCH_PREP, MATCH_DKDV, MATCH_DQ, MATCH_SHARED, MATCH_OTHER_ARCH):
        _write(root / f"{matcher}.umd.json", {"id": matcher})
    packs = (
        ("prep", [MATCH_SHARED, MATCH_PREP], []),
        ("dkdv", [MATCH_SHARED, MATCH_DKDV], ["gfx950"]),
        ("dkdv", [MATCH_OTHER_ARCH], ["gfx942"]),
        ("dq", [MATCH_SHARED, MATCH_DQ], []),
    )
    for number, (slot, matchers, arch) in enumerate(packs):
        _write(
            root / f"pack{number}.kdp.json",
            {
                "version": "1.1",
                "id": f"{number:02d}be5fe7-02a7-4ec2-b79c-e849951f8c24",
                "engine": UED_SLOTTED,
                "slot": slot,
                "arch": arch,
                "matchers": matchers,
            },
        )
    return root


def _dep(identity, revision_):
    return {"id": identity, "revision": revision_}


def test_a_slot_ranker_snapshot_names_only_its_slots_kmd_and_packs(tmp_path):
    tree = _slotted_tree(tmp_path)
    snapshot = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    assert snapshot == {
        "ued": _dep(UED_SLOTTED, "1.0"),
        "kmd": _dep(KMD_DKDV, "2.3"),
        "slot": "dkdv",
        "umd": [_dep(MATCH_DKDV, "1.0"), _dep(MATCH_SHARED, "1.0")],
    }
    # The other arch's dkdv pack is not this arch's; the other slots' packs never are.
    assert snapshot_provenance(tree, arch="gfx942", slot="dkdv")["umd"] == [
        _dep(MATCH_OTHER_ARCH, "1.0")
    ]
    assert snapshot_provenance(tree, slot="prep")["umd"] == sorted(
        [_dep(MATCH_SHARED, "1.0"), _dep(MATCH_PREP, "1.0")], key=lambda d: d["id"]
    )


def test_an_engine_level_snapshot_of_a_slotted_engine_names_every_slots_kmd(tmp_path):
    tree = _slotted_tree(tmp_path)
    snapshot = snapshot_provenance(tree, arch="gfx950")
    assert "kmd" not in snapshot and "slot" not in snapshot
    assert snapshot["kmds"] == [
        {"slot": "prep", **_dep(KMD_PREP, "1.0")},
        {"slot": "dkdv", **_dep(KMD_DKDV, "2.3")},
        {"slot": "dq", **_dep(KMD_DQ, "1.4")},
    ]
    assert {item["id"] for item in snapshot["umd"]} == {
        MATCH_SHARED,
        MATCH_PREP,
        MATCH_DKDV,
        MATCH_DQ,
    }


@pytest.mark.parametrize("slot", SLOTS)
def test_slot_snapshots_round_trip_through_validation_and_comparison(tmp_path, slot):
    tree = _slotted_tree(tmp_path)
    snapshot = snapshot_provenance(tree, arch="gfx950", slot=slot)
    assert validate_provenance(snapshot) == snapshot
    compare_provenance(snapshot, snapshot_provenance(tree, arch="gfx950", slot=slot))
    engine_level = snapshot_provenance(tree, arch="gfx950")
    assert validate_provenance(engine_level) == engine_level
    compare_provenance(engine_level, snapshot_provenance(tree, arch="gfx950"))


def test_a_slot_ranker_survives_a_compatible_revision_of_its_own_kmd(tmp_path):
    tree = _slotted_tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    _write(tree / f"{KMD_DKDV}.kmd.json", {"id": KMD_DKDV, "revision": "2.4"})
    compare_provenance(trained, snapshot_provenance(tree, arch="gfx950", slot="dkdv"))
    _write(tree / f"{KMD_DKDV}.kmd.json", {"id": KMD_DKDV, "revision": "3.0"})
    with pytest.raises(ProvenanceError, match="trained_against.kmd: .* incompatible"):
        compare_provenance(
            trained, snapshot_provenance(tree, arch="gfx950", slot="dkdv")
        )


def test_another_slots_kmd_revision_does_not_invalidate_a_slot_ranker(tmp_path):
    tree = _slotted_tree(tmp_path)
    ranker = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    engine_level = snapshot_provenance(tree, arch="gfx950")
    _write(tree / f"{KMD_DQ}.kmd.json", {"id": KMD_DQ, "revision": "9.0"})
    compare_provenance(ranker, snapshot_provenance(tree, arch="gfx950", slot="dkdv"))
    # The engine-level model names every slot's KMD, so it does see the dq major bump.
    with pytest.raises(ProvenanceError, match="trained_against.kmds: .* incompatible"):
        compare_provenance(engine_level, snapshot_provenance(tree, arch="gfx950"))


def test_a_slot_ranker_cannot_be_applied_to_another_slot(tmp_path):
    tree = _slotted_tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    with pytest.raises(
        ProvenanceError,
        match="names slot 'dkdv', but the ranker serves slot 'dq'",
    ):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx950", slot="dq"))


def test_a_slot_ranker_must_name_the_slot_it_was_trained_for(tmp_path):
    tree = _slotted_tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    anonymous = {k: v for k, v in trained.items() if k != "slot"}
    with pytest.raises(
        ProvenanceError,
        match="names no slot, but the ranker of slot 'dkdv' must name the slot it was "
        "trained for",
    ):
        compare_provenance(
            anonymous, snapshot_provenance(tree, arch="gfx950", slot="dkdv")
        )


def test_slot_ranker_and_engine_level_provenance_are_not_interchangeable(tmp_path):
    tree = _slotted_tree(tmp_path)
    ranker = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    engine_level = snapshot_provenance(tree, arch="gfx950")
    with pytest.raises(
        ProvenanceError,
        match="names a single slot, but a slotted engine's predict_engine",
    ):
        compare_provenance(ranker, engine_level)
    with pytest.raises(
        ProvenanceError, match="names kmds, but the ranker of slot 'dkdv'"
    ):
        compare_provenance(engine_level, ranker)


def test_an_engine_level_model_must_name_every_slot(tmp_path):
    tree = _slotted_tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950")
    trained["kmds"] = [e for e in trained["kmds"] if e["slot"] != "dq"]
    with pytest.raises(ProvenanceError, match="kmds does not name slot 'dq'"):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx950"))
    trained["kmds"].append({"slot": "extra", **_dep(KMD_DQ, "1.4")})
    with pytest.raises(
        ProvenanceError,
        match="kmds names slot 'extra', which the engine does not declare",
    ):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx950"))


def test_an_unslotted_engine_cannot_satisfy_slot_provenance(tmp_path):
    slotted = _slotted_tree(tmp_path / "slotted")
    plain = _tree(tmp_path / "plain")
    # The same engine identity, so only the missing slots can refuse the model.
    _write(
        plain / "engine.ued.json",
        {"id": UED_SLOTTED, "name": "test:slotted", "metadata": KMD},
    )
    for trained in (
        snapshot_provenance(slotted, arch="gfx950", slot="dkdv"),
        snapshot_provenance(slotted, arch="gfx950"),
    ):
        with pytest.raises(ProvenanceError, match="declares no slots"):
            compare_provenance(trained, snapshot_provenance(plain, arch="gfx942"))
    with pytest.raises(
        ProvenanceError, match="declares no slots, so it has no slot 'dkdv'"
    ):
        snapshot_provenance(plain, arch="gfx942", slot="dkdv")


def test_a_snapshot_of_an_undeclared_slot_is_refused(tmp_path):
    tree = _slotted_tree(tmp_path)
    with pytest.raises(ProvenanceError, match="declares no slot 'nope'"):
        snapshot_provenance(tree, slot="nope")


def test_a_matcher_owned_only_by_another_arch_is_foreign_to_a_slot_ranker(tmp_path):
    from uhd_gen.provenance import (
        foreign_matcher_ids,
        load_descriptor_tree,
        select_engine,
    )

    tree = _slotted_tree(tmp_path)
    trained = snapshot_provenance(tree, arch=None, slot="dkdv")
    actual = snapshot_provenance(tree, arch="gfx950", slot="dkdv")
    with pytest.raises(
        ProvenanceError, match="trained_against.umd: dependency .* missing"
    ):
        compare_provenance(trained, actual)
    index = load_descriptor_tree(tree)
    _, ued = select_engine(index, None)
    foreign = foreign_matcher_ids(index, ued, "gfx950", "dkdv")
    assert foreign == {MATCH_OTHER_ARCH}
    compare_provenance(trained, actual, foreign_matchers=foreign)


def test_version_1_1_descriptors_load_and_unknown_versions_do_not(tmp_path):
    tree = _slotted_tree(tmp_path)
    snapshot_provenance(tree, slot="dq")
    _write(tree / "engine.ued.json", {"version": "1.2", "id": UED_SLOTTED})
    with pytest.raises(ProvenanceError, match="unsupported file-format version '1.2'"):
        snapshot_provenance(tree)


def _ranker(**overrides):
    base = {
        "ued": _dep(UED_SLOTTED, "1.0"),
        "kmd": _dep(KMD_DKDV, "2.3"),
        "slot": "dkdv",
        "umd": [],
    }
    base.update(overrides)
    return base


def _estimate(**overrides):
    base = {
        "ued": _dep(UED_SLOTTED, "1.0"),
        "kmds": [{"slot": "a", **_dep(KMD_DQ, "1.0")}],
        "umd": [],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "snapshot,refusal",
    [
        (
            _ranker(kmds=[{"slot": "dkdv", **_dep(KMD_DKDV, "2.3")}]),
            "names both kmd and kmds",
        ),
        (
            {"ued": _dep(UED_SLOTTED, "1.0"), "umd": [], "slot": "dkdv"},
            "slot requires kmd",
        ),
        (_estimate(slot="a"), "slot requires kmd, and excludes kmds"),
        (_ranker(slot="Bad"), "trained_against.slot 'Bad' is not a valid slot name"),
        (
            _ranker(slot="global"),
            "trained_against.slot 'global' is not a valid slot name",
        ),
        (_ranker(slot=3), "is not a valid slot name"),
        (_estimate(kmds=[]), "trained_against.kmds must be a non-empty array"),
        (_estimate(kmds={}), "trained_against.kmds must be a non-empty array"),
        (
            _estimate(kmds=[{"slot": "a", **_dep(KMD_DQ, "1.0")}] * 2),
            "kmds names slot 'a' twice",
        ),
        (
            _estimate(kmds=[{"slot": "global", **_dep(KMD_DQ, "1.0")}]),
            "kmds slot 'global' is not a valid slot name",
        ),
        (
            _estimate(kmds=[{"id": KMD_DQ, "revision": "1.0"}]),
            "kmds entries require exactly slot, id and revision",
        ),
        (
            _estimate(kmds=[{"slot": "a", **_dep(KMD_DQ, "1")}]),
            "expected semantic revision",
        ),
        (
            {k: v for k, v in _estimate().items() if k != "umd"},
            "requires exactly ued, kmds and umd",
        ),
        ({"selector_revision": "r", "slot": "dkdv"}, "slot requires kmd"),
    ],
)
def test_malformed_slot_provenance_is_refused_like_the_loader_refuses_it(
    snapshot, refusal
):
    with pytest.raises(ProvenanceError, match=refusal):
        validate_provenance(snapshot)


def test_a_slot_ranker_may_omit_the_optional_slot_member():
    anonymous = {k: v for k, v in _ranker().items() if k != "slot"}
    assert validate_provenance(anonymous) == anonymous


@pytest.mark.parametrize(
    "name, edit",
    [
        ("engine.ued.json", {"version": "1.0"}),
        ("pack0.kdp.json", {"version": "1.0"}),
    ],
)
def test_slot_members_need_file_format_1_1(tmp_path, name, edit):
    tree = _slotted_tree(tmp_path)
    path = tree / name
    path.write_text(
        json.dumps({**json.loads(path.read_text(encoding="utf-8")), **edit}),
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="require file-format version 1.1"):
        snapshot_provenance(tree, slot="dkdv")


def test_only_a_ued_or_kdp_may_be_format_1_1(tmp_path):
    tree = _slotted_tree(tmp_path)
    _write(tree / f"{KMD_DKDV}.kmd.json", {"version": "1.1", "id": KMD_DKDV})
    with pytest.raises(ProvenanceError, match="unsupported file-format version '1.1'"):
        snapshot_provenance(tree, slot="dkdv")
