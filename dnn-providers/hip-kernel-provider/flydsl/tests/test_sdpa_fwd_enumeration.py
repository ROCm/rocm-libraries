# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA forward engines' alternative knobs are the knobs their shipped UEDs add.

The runtime enumerates an unslotted engine's candidates for a graph and refuses an enumeration
in which two kernels have the same tuple of UED knob values (``GenericPlanBuilder``, "Ambiguous
enrolled knob tuple"). ``gen_descriptors.FORWARD_ALTERNATIVE_KNOBS`` lists, per forward engine, the
graph-determined fields that tell apart the kernels serving one graph; the generator turns them
into knobs. ``TestFlydslSdpaCdnaPacks`` (a census over the shipped kernels and a graph set)
checks that the tuples are distinct; these tests hold the dict to the shipped UEDs and KMDs,
which need no ``dvc pull``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
DESCRIPTORS = FLYDSL.parent / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"

sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402

FORWARD_OPS = sorted(gd.FORWARD_ALTERNATIVE_KNOBS)


def _load(op: str, suffix: str) -> dict:
    path = DESCRIPTORS / op / f"flydsl_{op}.{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_there_are_forward_engines_to_check() -> None:
    assert FORWARD_OPS == ["sdpa_cdna", "sdpa_cdna_dualwave"]


@pytest.mark.parametrize("op", FORWARD_OPS)
def test_every_forward_alternative_knob_is_an_int_kmd_field_of_its_op(op: str) -> None:
    types = {f["name"]: f["type"] for f in _load(op, "kmd")["fields"]}
    for field in gd.FORWARD_ALTERNATIVE_KNOBS[op]:
        assert types.get(field) == "int", (op, field)


@pytest.mark.parametrize("op", FORWARD_OPS)
def test_the_shipped_engine_adds_exactly_the_forward_alternative_knobs(op: str) -> None:
    knobs = set(_load(op, "ued")["knobs"])
    assert knobs - gd.TUNING_FIELDS == gd.FORWARD_ALTERNATIVE_KNOBS[op], op
