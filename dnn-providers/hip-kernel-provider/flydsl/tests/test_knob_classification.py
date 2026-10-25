# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""An engine exposes as knobs only the schedule its objects were built with.

A UED's ``knobs`` are what a caller may pin and what autotune and the UHD tooling treat
as the variant space. A field the graph fixes (``has_bias``, ``head_dim``) or a tier
bound the matcher gates on (``head_dim_max``) is no choice: offering it would have a
caller pin, and a model rank on, a value the graph already decided.
``gen_descriptors.py`` classifies every metadata field into one of three sets and
exposes only ``TUNING_FIELDS``; these tests hold the shipped UEDs to that and require
the generator to refuse a field it cannot classify. A slotted engine (the SDPA backward
split) is descriptor-only and keeps its knobs per slot, against each slot's own KMD. They
read only the op-level descriptors in git, so they need no ``dvc pull``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
DESCRIPTORS = FLYDSL.parent / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"

sys.path.insert(0, str(FLYDSL))
import gen_descriptors  # noqa: E402

UEDS = sorted(DESCRIPTORS.glob("*/*.ued.json"))


def _entry(op: str, name: str, knobs: dict) -> gen_descriptors.KernelEntry:
    record = {
        "name": name,
        "file": f"{name}.hsaco",
        "symbol": name,
        "dtype": "bf16",
        "priority": 100,
        "knobs": knobs,
    }
    return gen_descriptors.KernelEntry(
        op=op, path=Path("unused"), data=b"", record=record
    )


def test_there_are_engines_to_check() -> None:
    assert UEDS, f"no UEDs under {DESCRIPTORS}"


def _kmd_fields(root: Path, kmd_id: str) -> list[str]:
    """The field names of the KMD @p kmd_id, wherever under @p root it sits."""
    for path in root.rglob("*.kmd.json"):
        kmd = json.loads(path.read_text(encoding="utf-8"))
        if kmd["id"] == kmd_id:
            return [f["name"] for f in kmd["fields"]]
    raise AssertionError(f"no KMD {kmd_id} under {root}")


def _knob_owners(ued: dict, ued_dir: Path, root: Path) -> list[tuple[str, list, list]]:
    """(label, declared knobs, the KMD's fields) for each owner of knobs in a UED.

    An unslotted engine owns its knobs and sits beside its KMD. A slotted engine is
    descriptor-only: each slot owns its knobs and names its own KMD, which sits in the
    slot's content folder.
    """
    if "slots" not in ued:
        kmd = json.loads(next(ued_dir.glob("*.kmd.json")).read_text(encoding="utf-8"))
        return [("engine", ued["knobs"], [f["name"] for f in kmd["fields"]])]
    return [
        (
            f"slot {slot['name']}",
            slot.get("knobs", []),
            _kmd_fields(root, slot["metadata"]),
        )
        for slot in ued["slots"]
    ]


def _assert_exposes_exactly_tuning_fields(ued_path: Path, root: Path) -> None:
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    for label, knobs, fields in _knob_owners(ued, ued_path.parent, root):
        expected = [f for f in fields if f in gen_descriptors.TUNING_FIELDS]
        assert knobs == expected, label
        assert not set(knobs) & gen_descriptors.GRAPH_DETERMINED_FIELDS, label


@pytest.mark.parametrize("ued_path", UEDS, ids=lambda p: p.parent.name)
def test_an_engine_exposes_exactly_its_tuning_fields(ued_path: Path) -> None:
    _assert_exposes_exactly_tuning_fields(ued_path, DESCRIPTORS)


def _split_tree(tmp_path: Path) -> Path:
    """The split engine's descriptors as the generator derives them from synthetic rows."""
    rows = [
        _entry(op, f"{op}_x", {"head_dim": 64, "block_m": 64, "has_bias": 1, **extra})
        for op, extra in (
            ("sdpa_bwd_prep", {}),
            ("sdpa_bwd_dkdv", {"waves_per_eu": 2}),
            ("sdpa_bwd_dq", {"role_order": "auto", "reduce_symbol": ""}),
        )
    ]
    gen_descriptors._write(tmp_path, gen_descriptors.build_documents(rows, "gfx950"))
    return tmp_path


def test_a_slotted_engine_exposes_exactly_each_slots_tuning_fields(
    tmp_path: Path,
) -> None:
    root = _split_tree(tmp_path)
    ued_path = root / "sdpa_bwd_split" / "flydsl_sdpa_bwd_split.ued.json"
    # The engine is descriptor-only: no KMD beside it, so the per-slot KMDs are found
    # by id in the slots' own folders, and each slot's schema differs.
    assert not list(ued_path.parent.glob("*.kmd.json"))
    _assert_exposes_exactly_tuning_fields(ued_path, root)
    knobs = {
        slot["name"]: slot["knobs"]
        for slot in json.loads(ued_path.read_text(encoding="utf-8"))["slots"]
    }
    assert knobs == {
        "prep": ["block_m"],
        "dkdv": ["block_m", "waves_per_eu"],
        "dq": ["block_m", "role_order"],
    }


def test_a_slot_that_exposes_a_graph_determined_field_is_caught(tmp_path: Path) -> None:
    root = _split_tree(tmp_path)
    ued_path = root / "sdpa_bwd_split" / "flydsl_sdpa_bwd_split.ued.json"
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    ued["slots"][2]["knobs"].append("reduce_symbol")
    ued_path.write_text(json.dumps(ued), encoding="utf-8")
    with pytest.raises(AssertionError, match="slot dq"):
        _assert_exposes_exactly_tuning_fields(ued_path, root)


def test_the_three_field_sets_are_disjoint() -> None:
    graph = gen_descriptors.GRAPH_DETERMINED_FIELDS
    selection = gen_descriptors.SELECTION_FIELDS
    tuning = gen_descriptors.TUNING_FIELDS
    assert not graph & selection
    assert not graph & tuning
    assert not selection & tuning


def test_an_unclassified_field_is_refused() -> None:
    with pytest.raises(gen_descriptors.HkpPackError, match="'new_flag'.*found in none"):
        gen_descriptors._fields_for_op(
            [_entry("sdpa_rdna", "sdpa_rdna_x", {"block_m": 64, "new_flag": 1})]
        )


def test_a_field_in_two_sets_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gen_descriptors, "TUNING_FIELDS", gen_descriptors.TUNING_FIELDS | {"has_bias"}
    )
    with pytest.raises(gen_descriptors.HkpPackError, match="'has_bias'.*exactly one"):
        gen_descriptors._fields_for_op(
            [_entry("sdpa_rdna", "sdpa_rdna_x", {"block_m": 64, "has_bias": 0})]
        )
