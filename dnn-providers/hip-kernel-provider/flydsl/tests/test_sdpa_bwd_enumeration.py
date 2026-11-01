# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The kernels that serve one graph have distinct knob tuples, and every promoted knob is real.

The runtime enumerates a slot's (or an unslotted engine's) candidates for a graph and refuses
an enumeration in which two kernels have the same tuple of UED knob values
(``GenericPlanBuilder::enumerateSlotted``, "Ambiguous enrolled knob tuple"). So the kernels
that match one graph must differ in a knob, and a knob is a performance alternative: some
graph must be served by kernels that differ in it, else it is a feature field that
``gen_descriptors.ALTERNATIVE_KNOBS`` must not list.

``serves()`` below is a **model of the C++ kernel matchers**, not their implementation, and
must move with them:

* split: ``flydslSdpaBwdSplitKernelMatches<SLOT>`` with ``servesHeadDims``, ``floorOf`` and
  ``rungBelow`` in ``FlydslSdpaBwdSplitNative.cpp``;
* merged: ``flydslSdpaBwdMergedKernelMatches`` with ``servesHeadDims`` and ``rungBelow`` in
  ``FlydslSdpaBwdMergedNative.cpp``.

Launch-grid limits are not modelled. Over the shipped manifests (skipped when the tree has
no content) the probe graphs are the classes in ``probe_graphs()``; a synthetic catalogue
that overlaps on purpose always runs.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402
from generators._flydsl_env import CONTENT_DIR  # noqa: E402

ARCH = "gfx950"
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
FP32_LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 512)
SLOTS = ("prep", "dkdv", "dq")


def rung_below(tile: int, dtype: str) -> int:
    ladder = FP32_LADDER if dtype == "fp32" else LADDER
    return max((r for r in ladder if r < tile), default=0)


def floor_of(knobs: dict, dtype: str) -> int:
    floor = knobs["floor"]
    if floor == "g1":
        return 0
    if floor == "tail":
        return rung_below(knobs["head_dim"], dtype)
    return int(floor)


def serves_dims(knobs: dict, dtype: str, qk: int, v: int, prep: bool) -> bool:
    low, high = (v, v) if prep else (min(qk, v), max(qk, v))
    if knobs["padded"] == 0:
        return low == knobs["head_dim"] == high
    return low > floor_of(knobs, dtype) and high <= knobs["head_dim"]


def serves(op: str, row: dict, graph: dict) -> bool:
    """Whether the row of @p op is applicable to @p graph: the C++ kernel_match, modelled."""
    knobs = row["knobs"]
    if row["dtype"] != graph["dtype"]:
        return False
    slot = op.removeprefix("sdpa_bwd_")
    if op == "sdpa_bwd_merged":
        if not serves_dims(knobs, graph["dtype"], graph["qk"], graph["v"], False):
            return False
        if (knobs["dropout"] != 0) != graph["dropout"]:
            return False
    else:
        if not serves_dims(
            knobs, graph["dtype"], graph["qk"], graph["v"], slot == "prep"
        ):
            return False
        if slot == "prep":
            return knobs.get("sink", -1) != 1
        if not (knobs["dform"] == "rt" or (knobs["dropout"] != 0) == graph["dropout"]):
            return False
    if knobs["causal"] != int(graph["cls"] == "causal"):
        return False
    if knobs["has_bias"] != int(graph["cls"] == "bias"):
        return False
    if slot in ("dkdv", "merged") and knobs.get("gqa_pack", -1) == 1:
        if not graph["gqa"] or knobs.get("block_q", -1) > 64:
            return False
    if slot in ("dq", "merged"):
        if (
            knobs["has_bias"]
            and knobs.get("bias_stage", "auto") != "direct"
            and not graph["staging"]
        ):
            return False
        if graph["dbias"] and knobs.get("store_db", -1) == 0:
            return False
        if graph["reduces"] and not knobs.get("reduce_symbol"):
            return False
    return True


def probe_graphs() -> list[dict]:
    """Graph classes: dtype, exact and padded dims (equal and unequal), mask, bias, dropout, GQA."""
    graphs = []
    for dtype in ("bf16", "f16", "fp32"):
        for qk, v in (
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (224, 224),
            (100, 100),
            (200, 200),
            (300, 300),
            (512, 512),
            (100, 64),
            (192, 128),
        ):
            for cls, dbias, reduces in (
                ("plain", False, False),
                ("causal", False, False),
                ("bias", False, False),
                ("bias", True, False),
                ("bias", True, True),
            ):
                for dropout, gqa in itertools.product((False, True), repeat=2):
                    graphs.append(
                        dict(
                            dtype=dtype,
                            qk=qk,
                            v=v,
                            cls=cls,
                            dbias=dbias,
                            reduces=reduces,
                            dropout=dropout,
                            gqa=gqa,
                            staging=True,
                        )
                    )
    return graphs


def knob_tuple(op: str, row: dict) -> tuple:
    names = gd.knob_fields(op, row["knobs"])
    return tuple(row["knobs"][n] for n in names)


def co_applicable(op: str, rows: list[dict]) -> list[list[dict]]:
    """For each probe graph, the rows of @p op that apply to it (none: skipped)."""
    out = []
    for graph in probe_graphs():
        applicable = [r for r in rows if serves(op, r, graph)]
        if applicable:
            out.append(applicable)
    return out


def duplicate_tuples(op: str, groups: list[list[dict]]) -> list[tuple[str, str]]:
    found = []
    for rows in groups:
        seen: dict[tuple, str] = {}
        for row in rows:
            key = knob_tuple(op, row)
            if key in seen:
                found.append((seen[key], row["name"]))
            seen[key] = row["name"]
    return found


def unexercised_alternatives(op: str, groups: list[list[dict]]) -> set[str]:
    """Promoted fields of @p op that no probe graph has two applicable rows differing in."""
    missing = set()
    for field in gd.ALTERNATIVE_KNOBS.get(op, ()):
        if not any(len({r["knobs"].get(field) for r in rows}) > 1 for rows in groups):
            missing.add(field)
    return missing


# ---------------------------------------------------------------------------
# Synthetic catalogue: always runs
# ---------------------------------------------------------------------------


def _row(op: str, name: str, **knobs) -> dict:
    base = dict(
        head_dim=64,
        padded=0,
        floor="tail",
        dform="bk",
        dropout=0,
        causal=0,
        has_bias=0,
        hints=1,
    )
    if op == "sdpa_bwd_prep":
        base.update(sink=-1)
    if op == "sdpa_bwd_dkdv":
        base.update(gqa_pack=-1)
    if op == "sdpa_bwd_dq":
        base.update(store_db=-1, bias_stage="auto", reduce_symbol="")
    if op == "sdpa_bwd_merged":
        base.update(gqa_pack=-1, store_db=-1, reduce_symbol="")
    base.update(knobs)
    return {"name": name, "dtype": "bf16", "knobs": base}


def synthetic() -> dict[str, list[dict]]:
    """Per op, rows that overlap on purpose in exactly the fields the op promotes."""
    catalogue = {}
    for op in ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"):
        rows = [_row(op, "exact64"), _row(op, "padded64", padded=1)]
        if op in ("sdpa_bwd_dkdv", "sdpa_bwd_dq"):
            rows += [
                _row(op, "padded128", head_dim=128, padded=1),
                _row(op, "g1_128", head_dim=128, padded=1, floor="g1"),
                _row(op, "rt64", dform="rt", dropout=1),
                _row(op, "bk64d", dropout=1),
            ]
        if op in ("sdpa_bwd_dkdv", "sdpa_bwd_merged"):
            rows.append(_row(op, "packed64", gqa_pack=1))
        if op in ("sdpa_bwd_dq", "sdpa_bwd_merged"):
            rows.append(_row(op, "nostore", has_bias=1, store_db=0))
            rows.append(_row(op, "store", has_bias=1, reduce_symbol="r"))
        if op == "sdpa_bwd_dq":
            rows.append(
                _row(op, "direct", has_bias=1, bias_stage="direct", reduce_symbol="r")
            )
        catalogue[op] = rows
    return catalogue


@pytest.mark.parametrize("op", sorted(synthetic()))
def test_the_promoted_knobs_keep_overlapping_kernels_distinct(op: str) -> None:
    groups = co_applicable(op, synthetic()[op])
    assert groups
    assert duplicate_tuples(op, groups) == []


@pytest.mark.parametrize("op", sorted(gd.ALTERNATIVE_KNOBS))
def test_every_promoted_knob_is_an_alternative_somewhere(op: str) -> None:
    groups = co_applicable(op, synthetic()[op])
    assert unexercised_alternatives(op, groups) == set()


def test_without_the_promotion_the_synthetic_catalogue_collides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gd, "ALTERNATIVE_KNOBS", {})
    for op in ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"):
        groups = co_applicable(op, synthetic()[op])
        assert duplicate_tuples(op, groups), op


def test_a_feature_field_is_not_an_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promoting `dropout` would list a field no two applicable rows differ in."""
    monkeypatch.setattr(
        gd,
        "ALTERNATIVE_KNOBS",
        {**gd.ALTERNATIVE_KNOBS, "sdpa_bwd_dkdv": frozenset({"has_bias"})},
    )
    groups = co_applicable("sdpa_bwd_dkdv", synthetic()["sdpa_bwd_dkdv"])
    assert unexercised_alternatives("sdpa_bwd_dkdv", groups) == {"has_bias"}


# ---------------------------------------------------------------------------
# The shipped catalogue
# ---------------------------------------------------------------------------


def _manifest(op: str) -> list[dict]:
    path = CONTENT_DIR / op / ARCH / "manifest.json"
    if not path.is_file():
        pytest.skip(f"no {op} content under {CONTENT_DIR}")
    return json.loads(path.read_text(encoding="utf-8"))["instances"]


@pytest.mark.parametrize(
    "op", ["sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq", "sdpa_bwd_merged"]
)
def test_the_shipped_kernels_that_serve_a_graph_have_distinct_knob_tuples(
    op: str,
) -> None:
    groups = co_applicable(op, _manifest(op))
    assert groups, f"no probe graph is served by {op}"
    collisions = duplicate_tuples(op, groups)
    assert not collisions, f"{len(collisions)} collisions, first {collisions[:3]}"


@pytest.mark.parametrize("op", sorted(gd.ALTERNATIVE_KNOBS))
def test_every_promoted_knob_of_the_shipped_kernels_is_a_real_alternative(
    op: str,
) -> None:
    groups = co_applicable(op, _manifest(op))
    assert unexercised_alternatives(op, groups) == set(), op
