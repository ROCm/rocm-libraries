# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The nondeterministic-build list: its schema, its agreement with the pins, and the merge of a sampling run into it.

The policy that reads the list is tested where it acts: ``test_sdpa_bwd_ship.py`` (what ``generate`` ships and when it stops) and
``test_sdpa_bwd_gate.py`` (the per-kernel gate). Pure Python.
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
from generators import _sdpa_bwd_cdna_build as build  # noqa: E402
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators._sdpa_bwd_spec import Spec  # noqa: E402

DATA = FLYDSL / "tests" / "data"
PLAIN = "dq,f16,64,causal,bk,0,1,tail,32,1"
COMBINED = "dq,fp32,224,bias,bk,0,1,g1,192,1;block_m=32;block_n=16"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def row(spec: str = PLAIN, **fields) -> nondet.Row:
    base = nondet.Row(
        "gfx950", Spec.parse(spec).hash(), spec, "variants", sha("pin"), (), (), "test"
    )
    return replace(base, **fields)


def reference(*rows: nondet.Row) -> dict[str, dict]:
    return {
        r.hash: {"hash": r.hash, "spec": r.spec, "sha256": r.pin_sha256} for r in rows
    }


def combined(spec: Spec) -> bool:
    return bool(build.reduce_symbol_of(spec))


def write_rows(path: Path, *rows: dict[str, str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, nondet.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kind", "flaky", "kind"),
        ("arch", "gfx90a", "arch"),
        ("pin_sha256", "abc", "pin_sha256"),
        ("variant_sha256s", sha("a") + ";short", "not a sha256"),
        ("hash", "0" * 12, "not the spec's"),
        ("evidence", "", "evidence"),
    ],
)
def test_a_row_that_breaks_the_schema_is_refused(tmp_path, field, value, match) -> None:
    csv_row = row().csv_row()
    csv_row[field] = value
    with pytest.raises(nondet.NondeterministicListError, match=match):
        nondet.read(write_rows(tmp_path / "list.csv", csv_row))


def test_a_list_with_a_duplicate_hash_or_other_columns_is_refused(tmp_path) -> None:
    twice = write_rows(tmp_path / "twice.csv", row().csv_row(), row().csv_row())
    with pytest.raises(nondet.NondeterministicListError, match="twice"):
        nondet.read(twice)
    other = tmp_path / "other.csv"
    other.write_text("hash,spec\n", encoding="utf-8")
    with pytest.raises(nondet.NondeterministicListError, match="columns"):
        nondet.read(other)


def test_a_row_the_reference_lacks_or_pins_differently_is_refused() -> None:
    listed = row()
    with pytest.raises(nondet.NondeterministicListError, match="no row"):
        nondet.check([listed], {}, combined)
    other_pin = replace(listed, pin_sha256=sha("another"))
    with pytest.raises(nondet.NondeterministicListError, match="pin_sha256"):
        nondet.check([listed], reference(other_pin), combined)
    nondet.check([listed], reference(listed), combined)


@pytest.mark.parametrize("kind", ["variants", "both"])
def test_a_combined_row_of_variants_needs_function_variants(kind) -> None:
    listed = row(COMBINED, kind=kind)
    with pytest.raises(nondet.NondeterministicListError, match="variant_function"):
        nondet.check([listed], reference(listed), combined)
    with_function = replace(listed, variant_function_sha256s=(sha("fn"),))
    nondet.check([with_function], reference(with_function), combined)


def test_a_combined_dual_object_row_and_a_standalone_variants_row_need_none() -> None:
    rows = [row(COMBINED, kind="dual_object"), row(PLAIN, kind="variants")]
    nondet.check(rows, reference(*rows), combined)


def test_the_in_tree_list_matches_the_pins_and_the_hash_reader_of_the_instance_table() -> (
    None
):
    rows = nondet.read_rows()
    assert rows
    for arch, csv_name in (
        ("gfx950", "sdpa_bwd_reference.csv"),
        ("gfx11-generic", "sdpa_bwd_gfx11_reference.csv"),
    ):
        mine = [r for r in rows if r.arch == arch]
        with (DATA / csv_name).open(newline="", encoding="utf-8") as fh:
            pins = {r["hash"]: r for r in csv.DictReader(fh)}
        nondet.check(mine, pins, lambda s: arch == "gfx950" and combined(s))
    assert _instances._sdpa_bwd_nondeterministic_hashes() == {
        r.hash for r in rows if r.arch == "gfx950"
    }


def test_the_exceptions_list_holds_no_row_pinned_for_nondeterminism() -> None:
    assert {r["reason"] for r in build.read_exceptions().values()} <= {
        "known-bad-fresh-build"
    }


def test_merging_a_sample_unions_variants_kinds_and_evidence_and_adds_new_rows(
    tmp_path,
) -> None:
    base = row(
        COMBINED,
        kind="variants",
        variant_sha256s=(sha("o1"),),
        variant_function_sha256s=(sha("f1"),),
        evidence="migrated",
    )
    standalone = row(PLAIN, kind="dual_object", evidence="old")
    listing = tmp_path / "list.csv"
    nondet.write([base, standalone], listing)
    sampled = tmp_path / "sampled.csv"
    nondet.write(
        [
            replace(
                base,
                kind="dual_object",
                variant_sha256s=(sha("o2"), sha("o1")),
                variant_function_sha256s=(sha("f2"),),
                evidence="sample N=6",
            ),
            row("dkdv,f16,64,plain,bk,0,0,tail,0,1", kind="variants", evidence="new"),
        ],
        sampled,
    )
    assert nondet.merge_file(sampled, listing) == (1, 1)
    merged = nondet.read(listing)
    extended = merged[base.hash]
    assert extended.kind == "both"
    assert extended.variant_sha256s == tuple(sorted({sha("o1"), sha("o2")}))
    assert extended.variant_function_sha256s == tuple(sorted({sha("f1"), sha("f2")}))
    assert extended.evidence == "migrated | sample N=6"
    assert merged[standalone.hash] == standalone
    assert len(merged) == 3
    # The same sample again changes nothing.
    assert nondet.merge_file(sampled, listing) == (0, 0)


def test_a_sample_that_names_another_pin_is_refused(tmp_path) -> None:
    listing, sampled = tmp_path / "list.csv", tmp_path / "sampled.csv"
    nondet.write([row()], listing)
    nondet.write([row(pin_sha256=sha("moved"))], sampled)
    with pytest.raises(nondet.NondeterministicListError, match="pin"):
        nondet.merge_file(sampled, listing)


def test_a_rebuild_is_tolerated_only_as_the_pin_or_a_recorded_variant() -> None:
    listed = row(variant_sha256s=(sha("obj"),), variant_function_sha256s=(sha("fn"),))
    assert nondet.tolerates(listed, sha("pin"))
    assert nondet.tolerates(listed, sha("obj"))
    assert nondet.tolerates(listed, sha("other"), sha("fn"))
    assert nondet.tolerates(listed, sha("other"), sha("pin fn"), sha("pin fn"))
    assert not nondet.tolerates(listed, sha("other"))
    assert not nondet.tolerates(listed, sha("other"), sha("another fn"), sha("pin fn"))
