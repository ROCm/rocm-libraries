# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward pins the tree names without an object: the function digests, the gate that reads them and the reference dir built from the content.

The shipped content holds the object of every pin but a combined object's main kernel and gfx950's standalone reduce kernels
(``tools/sdpa_bwd_pins.py``). Their function digests are ``tests/data/sdpa_bwd_pin_functions.csv``; a regeneration from the tree alone
copies every other pin out of the content (``tools/sdpa_bwd_s1.py --from-content``). Pure Python.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import sdpa_bwd_gate  # noqa: E402
import sdpa_bwd_pins as pins  # noqa: E402
import sdpa_bwd_s1 as s1  # noqa: E402
from generators._sdpa_bwd_spec import Spec, dbias_reduce_spec  # noqa: E402
from generators._manifest import sha256 as _sha  # noqa: E402
from generators._sdpa_bwd_rdna_build import RESOURCE_COLUMNS  # noqa: E402
from test_sdpa_bwd_compare import _pin_metadata  # noqa: E402,F401
from test_sdpa_bwd_gate import MERGED, MERGED_CODE, _elf  # noqa: E402


DATA = FLYDSL / "tests" / "data"
COMBINED = "merged,bf16,64,bias,bk,0,0,tail,0,1"
PLAIN = "dkdv,bf16,64,plain,bk,0,0,tail,0,1"


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_the_digest_file_names_exactly_the_pins_the_content_never_holds() -> None:
    """A catalogue installed without its digests (or with stale ones) leaves a pin the gate cannot judge."""
    wanted = {
        row["hash"]: row["spec"]
        for csv_name in ("sdpa_bwd_reference.csv", "sdpa_bwd_gfx11_reference.csv")
        for row in _rows(DATA / csv_name)
        if pins.holds_no_shipped_object(Spec.parse(row["spec"]))
    }
    have = _rows(pins.PIN_FUNCTIONS_CSV)
    assert {row["hash"]: row["spec"] for row in have} == wanted
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["function_sha256"]) for row in have)


def _gate_fixture(tmp_path: Path, monkeypatch, digest: str | None) -> tuple[Path, Spec]:
    """A reference CSV of one combined row and its reduce pin, an empty reference dir, and a digest file of @p digest (none: no row)."""
    spec = Spec.parse(COMBINED)
    reduce = dbias_reduce_spec(spec.bias_dtype)
    resources = dict.fromkeys(RESOURCE_COLUMNS, "1")
    _write(
        tmp_path / "reference.csv",
        ("hash", "spec", "sha256", *RESOURCE_COLUMNS),
        [
            {"hash": s.hash(), "spec": s.key(), "sha256": "0" * 64, **resources}
            for s in (spec, reduce)
        ],
    )
    functions = tmp_path / "functions.csv"
    _write(
        functions,
        pins.PIN_FUNCTION_COLUMNS,
        [
            {"hash": s.hash(), "spec": s.key(), "function_sha256": digest}
            for s in (spec, reduce)
            if digest
        ],
    )
    monkeypatch.setattr(pins, "PIN_FUNCTIONS_CSV", functions)
    (tmp_path / "ref").mkdir()
    return tmp_path / "reference.csv", spec


def test_a_pin_without_an_object_is_gated_by_its_function_digest(
    tmp_path: Path, monkeypatch
) -> None:
    digest = _sha(MERGED_CODE)
    reference, spec = _gate_fixture(tmp_path, monkeypatch, digest)
    merged, reduce = sdpa_bwd_gate.pin_facts(tmp_path / "ref", reference, spec)
    assert merged["function_sha256"] == digest
    assert reduce["hash"] == dbias_reduce_spec(spec.bias_dtype).hash()


def test_a_pin_with_an_object_is_gated_by_the_objects_function(
    tmp_path: Path, monkeypatch
) -> None:
    """The object is the pin; a digest beside it does not override it."""
    reference, spec = _gate_fixture(tmp_path, monkeypatch, "0" * 64)
    blob = _elf({MERGED: MERGED_CODE})
    rows = _rows(reference)
    rows[0]["sha256"] = _sha(blob)
    _write(reference, ("hash", "spec", "sha256", *RESOURCE_COLUMNS), rows)
    (tmp_path / "ref" / f"{spec.hash()}.hsaco").write_bytes(blob)
    merged, _ = sdpa_bwd_gate.pin_facts(tmp_path / "ref", reference, spec)
    assert merged["function_sha256"] == _sha(MERGED_CODE)


def test_a_pin_with_neither_an_object_nor_a_digest_stops_the_gate(
    tmp_path: Path, monkeypatch
) -> None:
    reference, spec = _gate_fixture(tmp_path, monkeypatch, None)
    with pytest.raises(RuntimeError, match="no pin object"):
        sdpa_bwd_gate.pin_facts(tmp_path / "ref", reference, spec)


def _content_fixture(tmp_path: Path, digest: str | None) -> tuple[Path, Path]:
    """A data dir whose reference CSV holds a shipped dK/dV pin and a combined merged pin, and a content root shipping the first."""
    data, content = tmp_path / "data", tmp_path / "content"
    shipped = b"shipped pin"
    folder = content / "sdpa_bwd_dkdv" / "gfx950"
    folder.mkdir(parents=True)
    (folder / "one.hsaco").write_bytes(shipped)
    (folder / "manifest.json").write_text(
        json.dumps({"instances": [{"file": "one.hsaco", "sha256": _sha(shipped)}]}),
        encoding="utf-8",
    )
    data.mkdir()
    resources = dict.fromkeys(RESOURCE_COLUMNS, "1")
    _write(
        data / "sdpa_bwd_reference.csv",
        ("hash", "spec", "sha256", *RESOURCE_COLUMNS, "rmir_scan"),
        [
            {
                "hash": Spec.parse(PLAIN).hash(),
                "spec": PLAIN,
                "sha256": _sha(shipped),
                **resources,
                "rmir_scan": "clean",
            },
            {
                "hash": Spec.parse(COMBINED).hash(),
                "spec": COMBINED,
                "sha256": "1" * 64,
                **resources,
                "rmir_scan": "clean",
            },
        ],
    )
    spec = Spec.parse(COMBINED)
    _write(
        data / pins.PIN_FUNCTIONS_CSV.name,
        pins.PIN_FUNCTION_COLUMNS,
        (
            [{"hash": spec.hash(), "spec": spec.key(), "function_sha256": digest}]
            if digest
            else []
        ),
    )
    return data, content


def test_the_reference_dir_holds_the_shipped_pins_and_names_the_rest(
    tmp_path: Path,
) -> None:
    data, content = _content_fixture(tmp_path, "2" * 64)
    written, named = s1.reference_dir_from_content(
        content, "gfx950", tmp_path / "ref", data
    )
    assert (written, named) == (1, 1)
    assert [p.name for p in (tmp_path / "ref").iterdir()] == [
        f"{Spec.parse(PLAIN).hash()}.hsaco"
    ]


def test_a_pin_the_content_does_not_hold_and_no_digest_names_stops_the_tool(
    tmp_path: Path,
) -> None:
    data, content = _content_fixture(tmp_path, None)
    with pytest.raises(s1.S1Error, match="names no function digest"):
        s1.reference_dir_from_content(content, "gfx950", tmp_path / "ref", data)
    assert not (tmp_path / "ref").exists()


@pytest.mark.parametrize("scan", ["", "unscanned"])
def test_a_pin_without_a_recorded_scan_is_refused_by_the_tool(
    tmp_path: Path, scan: str
) -> None:
    """Path S takes a gfx950 pin only when its reference row records `clean` or `exception`."""
    data, content = _content_fixture(tmp_path, "2" * 64)
    path = data / "sdpa_bwd_reference.csv"
    rows = _rows(path)
    rows[0]["rmir_scan"] = scan
    _write(path, tuple(rows[0]), rows)
    with pytest.raises(s1.S1Error, match="no clean R-MIR scan") as stop:
        s1.reference_dir_from_content(content, "gfx950", tmp_path / "ref", data)
    assert rows[0]["hash"] in str(stop.value)
    assert not (tmp_path / "ref").exists()


def test_the_tool_refuses_a_reference_dir_that_exists(tmp_path: Path) -> None:
    data, content = _content_fixture(tmp_path, "2" * 64)
    (tmp_path / "ref").mkdir()
    with pytest.raises(s1.S1Error, match="refusing"):
        s1.reference_dir_from_content(content, "gfx950", tmp_path / "ref", data)
