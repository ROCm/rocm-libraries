# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Consistency of the SDPA backward instance tables with their Spec grammar and the pin data.

The tables in ``generators/_instances.py`` are generated from the sweep snapshot
(``tools/sdpa_bwd_tables_from_sweep.py``); ``tests/data/sdpa_bwd_reference.csv`` carries the
hash and Spec key the sweep gave each pinned object. These tests hold the two to each
other and to ``generators/_sdpa_bwd_spec.py``, so a table row that no longer names its
pinned object, or a knob outside its domain, fails here. Pure Python, except the engine-constant
test, which resolves the dK/dV tuning and needs FlyDSL.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _instances  # noqa: E402
from generators._sdpa_bwd_spec import (  # noqa: E402
    KERNELS,
    LADDER,
    KNOBS,
    NON_KEY_FIELDS,
    ROW_FIELDS,
    Spec,
    to_spec,
)

DATA = Path(__file__).parent / "data"
TARGET = "gfx950"
OPS = [f"sdpa_bwd_{kernel}" for kernel in KERNELS]
# The values a row may spell for a knob; -1 and "auto" are the policy's.
DOMAIN = {
    "store_db": {-1, 0},
    "tight_registers": {-1, 0, 1},
    "gqa_pack": {-1, 1},
    "sink": {-1, 1},
    "bias_stage": {"auto", "direct"},
    "role_order": {"auto", "concat", "lpt"},
}


def _reference() -> dict[str, dict]:
    with (DATA / "sdpa_bwd_reference.csv").open(newline="", encoding="utf-8") as fh:
        return {row["hash"]: row for row in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def tables() -> dict[str, list]:
    return {op: _instances.instances_for(op, TARGET) for op in OPS}


def test_knob_sets_agree_with_the_spec_grammar() -> None:
    assert _instances.SDPA_BWD_KNOBS == {k: KNOBS[k] for k in KERNELS}


def test_every_row_of_an_op_has_that_ops_knob_set(tables) -> None:
    for kernel in KERNELS:
        expected = (
            set(ROW_FIELDS) | set(KNOBS[kernel]) | set(NON_KEY_FIELDS.get(kernel, ()))
        )
        for instance in tables[f"sdpa_bwd_{kernel}"]:
            assert set(instance.knobs) == expected, instance.name


def test_knobs_stay_in_their_domains(tables) -> None:
    for rows in tables.values():
        for instance in rows:
            for knob, allowed in DOMAIN.items():
                if knob in instance.knobs:
                    assert instance.knobs[knob] in allowed, (instance.name, knob)
            # An unset int knob is -1; a set one is positive (store_db and tight_registers excepted).
            for knob in ("num_waves", "mfma_rows", "block_q", "block_m", "block_n"):
                if knob in instance.knobs:
                    assert instance.knobs[knob] == -1 or instance.knobs[knob] > 0


def test_merged_rows_name_the_reduce_kernel_of_their_class(tables) -> None:
    """Only a dB-storing `bias` merged object holds the dBias reduce kernel, and the field moves no name and no Spec."""
    rows = tables["sdpa_bwd_merged"]
    for instance in rows:
        stores_db = bool(instance.knobs["has_bias"]) and instance.knobs["store_db"] != 0
        assert instance.knobs["reduce_symbol"] == (
            _instances.SDPA_BWD_REDUCE_SYMBOL[instance.dtype] if stores_db else ""
        )
        assert "reduce" not in instance.name
        stripped = replace(
            instance,
            knobs={k: v for k, v in instance.knobs.items() if k != "reduce_symbol"},
        )
        assert to_spec(stripped) == to_spec(instance)
    assert sum(bool(i.knobs["reduce_symbol"]) for i in rows) == 44
    assert sum(bool(i.knobs["has_bias"]) for i in rows) == 120
    assert len(rows) == 309


def test_dq_rows_name_the_reduce_kernel_of_their_dtype(tables) -> None:
    """A dB-storing bias dQ row holds its dtype's dBias reduce kernel; every fp32 bias dQ row stores dB."""
    rows = tables["sdpa_bwd_dq"]
    for instance in rows:
        stores_db = bool(instance.knobs["has_bias"]) and instance.knobs["store_db"] != 0
        assert instance.knobs["reduce_symbol"] == (
            _instances.SDPA_BWD_REDUCE_SYMBOL[instance.dtype] if stores_db else ""
        )
    fp32_bias = [i for i in rows if i.dtype == "fp32" and i.knobs["has_bias"]]
    assert fp32_bias
    assert all(
        i.knobs["reduce_symbol"].startswith("fmha_bwd_dbias_reduce_f32")
        for i in fp32_bias
    )
    assert sum(bool(i.knobs["reduce_symbol"]) for i in rows) == 320


def test_only_the_merged_and_dq_rows_may_carry_a_reduce_symbol(tables) -> None:
    row = tables["sdpa_bwd_prep"][0]
    with pytest.raises(ValueError, match="reduce_symbol"):
        to_spec(replace(row, knobs={**row.knobs, "reduce_symbol": ""}))


def test_rows_round_trip_through_the_spec(tables) -> None:
    """Row -> Spec -> key -> Spec is stable."""
    for rows in tables.values():
        for instance in rows:
            spec = to_spec(instance)  # validates: a malformed row raises here
            assert Spec.parse(spec.key()) == spec


def test_tables_are_the_reference_set(tables) -> None:
    """Each row's Spec hash is sha256(key)[:12], equals the hash the sweep gave that key, and no pin is lost or invented."""
    reference = _reference()
    built = {}
    for rows in tables.values():
        for instance in rows:
            key = to_spec(instance).key()
            digest = hashlib.sha256(key.encode()).hexdigest()[:12]
            assert reference[digest]["spec"] == key
            built[digest] = key
    assert built.keys() == reference.keys()


# Two pins whose objects fault (an LLVM spill miscompile) and the rebuilds that replace them.
SUPERSEDED = {"54973b5b17b1": "dca025070b19", "acf9212f9d2b": "2314101ce9c5"}


def test_the_superseded_pins_are_replaced_in_the_tables_and_the_reference(
    tables,
) -> None:
    """Asserted against literals, not against the generator's own filter: a regression there would feed both sides."""
    in_tables = {to_spec(i).hash() for rows in tables.values() for i in rows}
    assert sum(len(rows) for rows in tables.values()) == 2729
    assert {op: len(rows) for op, rows in tables.items()} == {
        "sdpa_bwd_prep": 86,
        "sdpa_bwd_dkdv": 1319,
        "sdpa_bwd_dq": 1010,
        "sdpa_bwd_merged": 309,
        "sdpa_bwd_dbias_reduce": 3,
        "sdpa_bwd_dsink_reduce": 2,
    }
    for held in (in_tables, _reference().keys()):
        assert len(held) == 2729
        assert not held & SUPERSEDED.keys()
        assert set(SUPERSEDED.values()) <= held


def test_known_flaky_hashes_are_pinned_objects() -> None:
    with (DATA / "sdpa_bwd_known_flaky.csv").open(newline="", encoding="utf-8") as fh:
        flaky = {row["hash"]: row for row in csv.DictReader(fh)}
    reference = _reference()
    assert flaky and flaky.keys() <= reference.keys()
    for digest, row in flaky.items():
        assert row["spec"] == reference[digest]["spec"]


def test_the_engines_restated_constants_match_the_tables(tables, monkeypatch) -> None:
    """The merged engine restates three generator facts (`FlydslSdpaBwdMergedNative.cpp`).

    * `LADDER`: the head-dim tiles objects are built at.
    * The exact-tile range of its graph check: equal head dims of 64, 96, 128, 160, 192, 224 or 256,
      plus one padded tile (32) for smaller dims.
    * `PACK_MAX_BLOCK_Q`: the largest dK/dV block_q of a `gqa_pack == 1` row, which bounds the
      engine's pack_split exactness test.
    """
    pytest.importorskip("flydsl")
    monkeypatch.syspath_prepend(str(FLYDSL / "kernels_src"))
    from generators import _sdpa_bwd_cdna_build as build  # noqa: PLC0415

    assert LADDER == (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)  # LADDER
    rows = tables["sdpa_bwd_merged"]
    assert {to_spec(r).tile for r in rows} == {
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
    }  # exactTile
    assert {to_spec(r).tile for r in rows if to_spec(r).padded} == {32}

    packed = [r for r in rows if r.knobs["gqa_pack"] == 1]
    assert packed
    block_q = {r.name: build._build_merged(to_spec(r)).knobs[0].block_q for r in packed}
    assert max(block_q.values()) <= 64, (  # PACK_MAX_BLOCK_Q
        "a gqa_pack row has a dK/dV block_q above PACK_MAX_BLOCK_Q (64, "
        f"FlydslSdpaBwdMergedNative.cpp): {block_q}"
    )
