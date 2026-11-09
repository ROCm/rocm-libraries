# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Consistency of the SDPA backward instance tables with their Spec grammar and the pin data.

The tables in ``generators/_instances.py`` are generated from the sweep snapshot
(``tools/sdpa_bwd_tables_from_sweep.py``); ``tests/data/sdpa_bwd_reference.csv`` carries the
hash and Spec key the sweep gave each pinned object. These tests hold the two to each
other and to ``generators/_sdpa_bwd_spec.py``, so a table row that no longer names its
pinned object, or a knob outside its domain, fails here. Pure Python, except the pack-bound
test, which resolves the dK/dV tuning and needs FlyDSL.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _instances  # noqa: E402
from generators._sdpa_bwd_cdna_build import read_reference  # noqa: E402
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
    "bias_f32": {-1, 1},
    "u2": {"auto", "dk", "dv", "dq"},
}


def _reference() -> dict[str, dict]:
    return read_reference(DATA / "sdpa_bwd_reference.csv")


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
    """A dB-storing bias dQ row holds the dBias reduce kernel of its bias dtype; every fp32 bias dQ row stores dB."""
    rows = tables["sdpa_bwd_dq"]
    for instance in rows:
        stores_db = bool(instance.knobs["has_bias"]) and instance.knobs["store_db"] != 0
        assert instance.knobs["reduce_symbol"] == (
            _instances.SDPA_BWD_REDUCE_SYMBOL[
                "fp32" if instance.knobs["bias_f32"] == 1 else instance.dtype
            ]
            if stores_db
            else ""
        )
    fp32_bias = [i for i in rows if i.dtype == "fp32" and i.knobs["has_bias"]]
    assert fp32_bias
    assert all(
        i.knobs["reduce_symbol"].startswith("fmha_bwd_dbias_reduce_f32")
        for i in fp32_bias
    )
    assert sum(bool(i.knobs["reduce_symbol"]) for i in rows) == 301


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
    assert sum(len(rows) for rows in tables.values()) == 2691
    assert {op: len(rows) for op, rows in tables.items()} == {
        "sdpa_bwd_prep": 86,
        "sdpa_bwd_dkdv": 1329,
        "sdpa_bwd_dq": 962,
        "sdpa_bwd_merged": 309,
        "sdpa_bwd_dbias_reduce": 3,
        "sdpa_bwd_dsink_reduce": 2,
    }
    for held in (in_tables, _reference().keys()):
        assert len(held) == 2691
        assert not held & SUPERSEDED.keys()
        assert set(SUPERSEDED.values()) <= held


PACKS = FLYDSL.parent / "src/engines/kernel_ingestor_engine/packs"
MERGED_NATIVE = PACKS / "FlydslSdpaBwdMergedNative.cpp"
SPLIT_NATIVE = PACKS / "FlydslSdpaBwdSplitNative.cpp"
NATIVE_SHARED = PACKS / "FlydslSdpaBwdNativeShared.hpp"


def _pack_max_block_q() -> int:
    """`PACK_MAX_BLOCK_Q`, defined once in `FlydslSdpaBwdNativeShared.hpp`; neither engine source defines it."""
    header = re.findall(
        r"\bPACK_MAX_BLOCK_Q\s*=\s*(\d+)\s*;", NATIVE_SHARED.read_text("utf-8")
    )
    assert len(header) == 1, "the bound must have one definition: the shared header's"
    for engine in (MERGED_NATIVE, SPLIT_NATIVE):
        assert not re.search(
            r"\bPACK_MAX_BLOCK_Q\s*=", engine.read_text("utf-8")
        ), f"{engine.name} defines its own PACK_MAX_BLOCK_Q"
    return int(header[0])


def _engine_constants() -> tuple[tuple[int, ...], range, int]:
    """(LADDER, the exact-tile head dims, PACK_MAX_BLOCK_Q) as the engine sources spell them."""
    cpp = MERGED_NATIVE.read_text(encoding="utf-8")
    ladder = re.search(r"\bLADDER\{([\d, ]+)\}", NATIVE_SHARED.read_text("utf-8"))
    exact = re.search(
        r"largest >= (\d+) && largest <= (\d+)\s*&& largest % (\d+) == 0", cpp
    )
    assert ladder and exact, "the engine's constants moved; update this reader"
    low, high, step = (int(g) for g in exact.groups())
    return (
        tuple(int(x) for x in ladder.group(1).split(",")),
        range(low, high + 1, step),
        _pack_max_block_q(),
    )


def test_the_engines_restated_ladder_and_exact_tiles_match_the_tables(tables) -> None:
    """The merged engine restates two generator facts, read here from its source.

    * `LADDER`: the head-dim tiles objects are built at.
    * The exact-tile range of its graph check: equal head dims at the exact tiles, plus one
      padded tile (32) for smaller dims.
    """
    ladder, exact_tiles, _ = _engine_constants()
    assert ladder == LADDER
    rows = tables["sdpa_bwd_merged"]
    assert {to_spec(r).tile for r in rows} == {32, *exact_tiles}
    assert {to_spec(r).tile for r in rows if to_spec(r).padded} == {32}


def test_the_engines_pack_bound_covers_every_packed_row(tables, monkeypatch) -> None:
    """`PACK_MAX_BLOCK_Q`: the largest dK/dV block_q of a `gqa_pack == 1` row, which bounds the
    engine's pack_split exactness test. Resolving the block_q needs FlyDSL."""
    pytest.importorskip("flydsl")
    monkeypatch.syspath_prepend(str(FLYDSL / "kernels_src"))
    from generators import _sdpa_bwd_cdna_build as build  # noqa: PLC0415

    packed = [r for r in tables["sdpa_bwd_merged"] if r.knobs["gqa_pack"] == 1]
    assert packed
    block_q = {r.name: build._build_merged(to_spec(r)).knobs[0].block_q for r in packed}
    bound = _engine_constants()[2]
    assert (
        max(block_q.values()) <= bound
    ), f"a gqa_pack row has a dK/dV block_q above PACK_MAX_BLOCK_Q ({bound}): {block_q}"


GFX11 = "gfx11-generic"
GFX11_SPLIT_OPS = [f"sdpa_bwd_{k}" for k in ("prep", "dkdv", "dq")]


def test_gfx11_table_columns_agree_with_the_spec_grammar() -> None:
    from generators._sdpa_bwd_spec import GENERIC_TABLE_KNOBS  # noqa: PLC0415

    assert _instances.SDPA_BWD_GFX11_KNOBS == GENERIC_TABLE_KNOBS


def test_gfx11_rows_are_the_s1_reference_set() -> None:
    """Every table row's Spec is a pin of ``sdpa_bwd_gfx11_reference.csv`` and none is lost; the dBias reduce rows are the dQ companions."""
    from generators import _sdpa_bwd_rdna_build as build  # noqa: PLC0415

    with (DATA / "sdpa_bwd_gfx11_reference.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        reference = {row["hash"]: row["spec"] for row in csv.DictReader(fh)}
    built = {}
    for kind in ("prep", "dkdv", "dq", "dbias_reduce"):
        for spec in build.specs_for(kind):
            assert spec.hash() not in built
            built[spec.hash()] = spec.key()
    assert built == reference
    assert [
        len(build.specs_for(k)) for k in ("prep", "dkdv", "dq", "dbias_reduce")
    ] == [
        33,
        238,
        230,
        3,
    ]


def test_gfx11_rows_have_the_field_set_of_the_gfx950_rows_of_their_op() -> None:
    """The KMD of an op is one for every arch: same fields, same order."""
    for op in GFX11_SPLIT_OPS:
        gfx950 = list(_instances.instances_for(op, TARGET)[0].knobs)
        for instance in _instances.instances_for(op, GFX11):
            assert list(instance.knobs) == gfx950, instance.name


def test_gfx11_row_values_follow_the_kmd_contract() -> None:
    from generators._sdpa_bwd_spec import GENERIC_LADDER  # noqa: PLC0415

    for instance in _instances.instances_for("sdpa_bwd_prep", GFX11):
        k = instance.knobs
        below = max([r for r in (0, *GENERIC_LADDER) if r < k["head_dim"]])
        assert (k["sink"], k["padded"], k["dform"], k["floor"]) == (
            0,
            1,
            "bk",
            str(below),
        )
    for instance in _instances.instances_for("sdpa_bwd_dkdv", GFX11):
        k = instance.knobs
        assert (
            k["mfma_rows"],
            k["dkv_shards"],
            k["tight_registers"],
            k["num_stream_buffers"],
            k["gqa_pack"],
            k["floor"],
        ) == (-1, -1, -1, -1, 0, "g1")
    rows = _instances.instances_for("sdpa_bwd_dq", GFX11)
    assert sum(bool(i.knobs["has_bias"]) for i in rows) == 82
    for instance in rows:
        k = instance.knobs
        assert k["mfma_rows"] == -1 and k["head_dim_v"] == k["head_dim"]
        if k["has_bias"]:
            assert (k["store_db"] in (0, 1), k["bias_stage"]) == (True, "none")
            assert k["reduce_symbol"] == (
                _instances.SDPA_BWD_GFX11_REDUCE_SYMBOL[
                    "fp32" if k["bias_f32"] == 1 else instance.dtype
                ]
                if k["store_db"] == 1
                else ""
            )
        else:
            assert (k["store_db"], k["bias_stage"], k["reduce_symbol"]) == (
                -1,
                "auto",
                "",
            )


def test_the_split_engines_ladders_are_the_generators() -> None:
    """The 16-bit ladder (shared header) and the fp32 ladder (no 384; split engine) match the ones the padded floors scan."""
    from generators._sdpa_bwd_spec import FP32_LADDER  # noqa: PLC0415

    shared = NATIVE_SHARED.read_text(encoding="utf-8")
    split = SPLIT_NATIVE.read_text(encoding="utf-8")
    sixteen = re.search(r"\bLADDER\{([\d, ]+)\}", shared)
    thirty_two = re.search(r"\bFP32_LADDER\{([\d, ]+)\}", split)
    assert (
        sixteen and thirty_two
    ), "the engines' ladders moved; update this reader"
    assert tuple(int(x) for x in sixteen.group(1).split(",")) == LADDER
    assert tuple(int(x) for x in thirty_two.group(1).split(",")) == FP32_LADDER
    for ladder, cpp in ((sixteen, shared), (thirty_two, split)):
        count = re.search(
            rf"std::array<int64_t, (\d+)> {'FP32_' if ladder is thirty_two else ''}LADDER",
            cpp,
        )
        assert count and int(count.group(1)) == len(ladder.group(1).split(","))


def test_the_engines_header_bound_is_the_one_the_generator_reads() -> None:
    import gen_descriptors  # noqa: PLC0415

    assert gen_descriptors.pack_max_block_q() == _pack_max_block_q()


TOOLS = FLYDSL / "tools"


def _tables_tool():
    sys.path.insert(0, str(TOOLS))
    import sdpa_bwd_tables_from_sweep as tool  # noqa: PLC0415

    return tool


@pytest.mark.parametrize(
    ("arch", "reference"),
    [
        ("gfx950", "sdpa_bwd_reference.csv"),
        ("gfx11-generic", "sdpa_bwd_gfx11_reference.csv"),
    ],
)
def test_the_tables_tool_renders_the_installed_tables_from_their_reference(
    arch: str, reference: str
) -> None:
    """The generated block is what the tool writes for the pins of the reference CSV: rows without a feature knob keep their text."""
    tool = _tables_tool()
    with (DATA / reference).open(newline="", encoding="utf-8") as fh:
        pins = {row["hash"]: row for row in csv.DictReader(fh)}
    begin, end = tool.markers(arch)
    text = (FLYDSL / "generators" / "_instances.py").read_text(encoding="utf-8")
    installed = text[text.index(begin) : text.index(end) + len(end)]
    assert tool.render_tables(arch, pins) == installed


def test_a_feature_row_spells_its_feature_columns_and_the_others_stop_before_them() -> (
    None
):
    tool = _tables_tool()
    columns = KNOBS["dkdv"]
    plain = tool.table_row(Spec.parse("dkdv,f16,64,plain,bk,0,0,tail,0,1"), columns)
    dk = tool.table_row(Spec.parse("dkdv,f16,64,plain,bk,0,0,tail,0,1;u2=dk"), columns)
    f32 = tool.table_row(
        Spec.parse("dkdv,f16,64,bias,bk,0,0,tail,0,1;bias_f32=1"), columns
    )
    assert plain.count(",") == 9 + len(columns) - 2
    assert dk.endswith('-1, "dk"),')
    assert f32.endswith(", 1),")  # u2 is unset and the row stops after bias_f32
    row = tuple(
        eval(dk.strip().rstrip(","), {})
    )  # noqa: S307 - a literal row of the tool's own output
    assert _instances.sdpa_bwd_perf_knobs(columns, row[9:])["u2"] == "dk"


def test_every_gfx950_pin_records_its_rmir_scan() -> None:
    """A pin is `clean` (a matched sweep scan) or `exception` (a pinned exception); the exception rows are exactly the exceptions CSV's; gfx11 records none."""
    with (DATA / "sdpa_bwd_pinned_exceptions.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        exceptions = {row["hash"] for row in csv.DictReader(fh)}
    reference = _reference()
    assert {h for h, r in reference.items() if r["rmir_scan"] == "exception"} == (
        exceptions
    )
    assert {r["rmir_scan"] for r in reference.values()} <= {"clean", "exception"}
    with (DATA / "sdpa_bwd_gfx11_reference.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        assert "rmir_scan" not in (csv.DictReader(fh).fieldnames or ())


def test_the_merged_engine_declines_exactly_the_biased_pinned_exceptions() -> None:
    """The merged engine's `UNCLAMPED_BIAS_OBJECTS` are the objects of the biased dK/dV or merged pinned exceptions.

    A pinned exception ships the pin, which predates the biased dK/dV exponent clamp (a rebuild with the clamp fails the R-MIR
    scan), so every such row is unclamped. The data does not say which pin has the clamp; the exception rows are all the unclamped
    ones, so the check is list equality, and it needs revisiting if an exception row ever ships a clamped pin.
    """
    with (DATA / "sdpa_bwd_pinned_exceptions.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        biased = {
            row["hash"]
            for row in csv.DictReader(fh)
            if row["spec"].split(",")[0] in ("dkdv", "merged")
            and "bias" in row["spec"].split(",")
        }
    descriptors = (
        FLYDSL.parent / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"
    )
    objects = {}
    for op in ("sdpa_bwd_dkdv", "sdpa_bwd_merged"):
        manifest = json.loads(
            (descriptors / op / TARGET / "manifest.json").read_text("utf-8")
        )
        for record in manifest["instances"]:
            if record["spec_hash"] in biased:
                objects[record["spec_hash"]] = Path(record["file"]).stem
    assert set(objects) == biased, "an exception row has no shipped object"
    source = MERGED_NATIVE.read_text("utf-8")
    listed = re.search(r"UNCLAMPED_BIAS_OBJECTS\{(.*?)\};", source, re.DOTALL)
    assert listed, "UNCLAMPED_BIAS_OBJECTS is not defined"
    assert set(re.findall(r'"([^"]+)"', listed.group(1))) == set(objects.values())


def test_the_tables_tool_refuses_a_gfx950_row_without_a_scan_record(
    tmp_path: Path,
) -> None:
    tool = _tables_tool()
    spec = Spec.parse("dkdv,f16,64,plain,bk,0,0,tail,0,1")
    columns = ("hash", "spec", "sha256", "rmir_scan")
    for scan, ok in (("clean", True), ("exception", True), ("", False)):
        (tmp_path / "manifest.csv").write_text(
            ",".join(columns) + f'\n{spec.hash()},"{spec.key()}",x,{scan}\n',
            encoding="utf-8",
        )
        (tmp_path / "known-flaky.csv").write_text("hash,spec,source\n")
        if ok:
            tool.read_pins(tmp_path)
        else:
            with pytest.raises(SystemExit, match="no rmir_scan"):
                tool.read_pins(tmp_path)
