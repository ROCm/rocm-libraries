# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""An engine exposes as knobs only the schedule its objects were built with.

A UED's ``knobs`` are what a caller may pin and what autotune and the UHD tooling treat
as the variant space. A field the graph fixes (``has_bias``, ``head_dim``) or a tier
bound the matcher gates on (``head_dim_max``) is no choice: offering it would have a
caller pin, and a model rank on, a value the graph already decided.
``gen_descriptors.py`` classifies every metadata field into one of three sets and
exposes only ``TUNING_FIELDS``; these tests hold the shipped UEDs to that and require
the generator to refuse a field it cannot classify. They read only the op-level
descriptors in git, so they need no ``dvc pull``.
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


def _entry(op: str, knobs: dict) -> gen_descriptors.KernelEntry:
    return gen_descriptors.KernelEntry(
        op=op, path=Path("unused"), data=b"", record={"name": f"{op}_x", "knobs": knobs}
    )


def test_there_are_engines_to_check() -> None:
    assert UEDS, f"no UEDs under {DESCRIPTORS}"


@pytest.mark.parametrize("ued_path", UEDS, ids=lambda p: p.parent.name)
def test_an_engine_exposes_exactly_its_tuning_fields(ued_path: Path) -> None:
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    kmd_path = next(ued_path.parent.glob("*.kmd.json"))
    fields = [
        f["name"] for f in json.loads(kmd_path.read_text(encoding="utf-8"))["fields"]
    ]
    expected = [f for f in fields if f in gen_descriptors.TUNING_FIELDS]
    assert ued["knobs"] == expected
    assert not set(ued["knobs"]) & gen_descriptors.GRAPH_DETERMINED_FIELDS


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
            [_entry("sdpa_rdna", {"block_m": 64, "new_flag": 1})]
        )


def test_a_field_in_two_sets_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gen_descriptors, "TUNING_FIELDS", gen_descriptors.TUNING_FIELDS | {"has_bias"}
    )
    with pytest.raises(gen_descriptors.HkpPackError, match="'has_bias'.*exactly one"):
        gen_descriptors._fields_for_op(
            [_entry("sdpa_rdna", {"block_m": 64, "has_bias": 0})]
        )
