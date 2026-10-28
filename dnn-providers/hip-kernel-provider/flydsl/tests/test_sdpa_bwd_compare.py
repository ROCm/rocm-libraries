# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward compare tool: what it reports, and what turns the exit status red.

A five-row fixture stands in for a generator run: one row identical to its pin, and four
that are not (a changed byte, a changed resource row, an object the generator never
built, a known-flaky ``dual_object``). The compare is report-only for those, and for a
``dual_object`` of the dq causal family. Spill findings, ``error`` rows, any other
``dual_object`` and a reference op with no manifest are red.
Pure Python.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import sdpa_bwd_compare  # noqa: E402
from generators._sdpa_bwd_spec import Spec  # noqa: E402
from test_sdpa_bwd_gate import (
    MERGED,
    MERGED_CODE,
    REDUCE,
    REDUCE_CODE,
    _elf,
)  # noqa: E402

RESOURCES = {
    "vgpr": 100,
    "agpr": 0,
    "vgpr_total": 100,
    "sgpr": 90,
    "sgpr_spill": 4,
    "vgpr_spill": 0,
    "scratch_bytes": 0,
    "lds_bytes": 8192,
    "occupancy": 4,
}
SPECS = {
    "same": "dkdv,bf16,64,plain,bk,0,0,tail,0,1",
    "byte": "dkdv,bf16,64,causal,bk,0,0,tail,0,1",
    "resource": "dq,bf16,64,plain,bk,0,0,tail,0,1",
    "absent": "merged,bf16,64,plain,bk,0,0,tail,0,1",
    "flaky": "dq,f16,64,causal,rt,1,0,tail,0,1",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_fixture(
    tmp_path: Path, objects: dict[str, tuple[bytes, bytes]] | None = None, **edits
) -> list[str]:
    """Reference data and a generator out-dir; returns the compare argv.

    @p objects maps a fixture row name to the (pin, shipped) object bytes that replace the
    default text bodies. @p edits maps a fixture row name to a record patch applied to its
    manifest record.
    """
    objects = objects or {}
    ref_objects = tmp_path / "ref"
    ref_objects.mkdir()
    with (tmp_path / "ref.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("hash", "spec", "sha256", *RESOURCES))
        for name, key in SPECS.items():
            digest = Spec.parse(key).hash()
            body = objects[name][0] if name in objects else f"pin {name}".encode()
            (ref_objects / f"{digest}.hsaco").write_bytes(body)
            writer.writerow((digest, key, _sha(body), *RESOURCES.values()))
    with (tmp_path / "flaky.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("hash", "spec", "source"))
        writer.writerow((Spec.parse(SPECS["flaky"]).hash(), SPECS["flaky"], "bimodal"))

    out = tmp_path / "out"
    by_op: dict[str, list[dict]] = {}
    for name, key in SPECS.items():
        spec = Spec.parse(key)
        digest = spec.hash()
        record = {
            "build_status": "ok",
            "spec": key,
            "spec_hash": digest,
            "file": f"{name}.hsaco",
            "symbol": MERGED,
            "resources": dict(RESOURCES),
            "gates": {"rmir": {"verdict": "ok", "spill_findings": 0}},
        }
        op_dir = out / f"sdpa_bwd_{spec.kernel}" / "gfx950"
        op_dir.mkdir(parents=True, exist_ok=True)
        body = f"pin {name}".encode()
        if name == "byte":
            body = b"pin bytf"
        if name in objects:
            body = objects[name][1]
        record["sha256"] = _sha(body)
        if name == "resource":
            record["resources"]["vgpr"] += 1
        if name == "flaky":
            record.update(build_status="dual_object", file="", detail="dual")
            body = None
        record.update(edits.get(name, {}))
        if body is not None and record["file"]:
            (op_dir / record["file"]).write_bytes(body)
        if name != "absent":
            by_op.setdefault(spec.kernel, []).append(record)
    for kernel in {Spec.parse(k).kernel for k in SPECS.values()}:
        op_dir = out / f"sdpa_bwd_{kernel}" / "gfx950"
        manifest = {
            "toolchain": {"rocm_version": "10.2.0"},
            "op": f"sdpa_bwd_{kernel}",
            "instances": by_op.get(kernel, []),
        }
        (op_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [
        str(tmp_path / "ref.csv"),
        str(ref_objects),
        str(out),
        "--flaky",
        str(tmp_path / "flaky.csv"),
    ]


def _run(argv: list[str], capsys) -> tuple[int, list[str]]:
    status = sdpa_bwd_compare.main(argv)
    return status, capsys.readouterr().out.splitlines()


def test_differences_are_reported_and_the_exit_is_zero(tmp_path, capsys) -> None:
    status, lines = _run(_build_fixture(tmp_path), capsys)
    assert status == 0
    assert "rocm_version: generator 10.2.0; reference unknown" in lines
    assert "2/5 byte-identical" in lines  # the resource-row row has the pin's bytes
    report = "\n".join(lines)
    assert "RED" not in report
    expected = {
        "sha differs": "dkdv tile=64 cls=causal",
        "resource row differs": "dq tile=64 cls=plain",
        "missing": "merged tile=64 cls=plain",
    }
    for kind, where in expected.items():
        assert any(l.startswith(f"{kind}: {where}") and "flaky=no" in l for l in lines)
    assert "vgpr 101 != 100" in report
    flaky = [l for l in lines if "dq tile=64 cls=causal" in l]
    assert (
        len(flaky) == 1 and flaky[0].startswith("missing") and "flaky=yes" in flaky[0]
    )


@pytest.mark.parametrize(
    ("row", "patch", "line"),
    [
        (
            "same",
            {"gates": {"rmir": {"verdict": "red", "spill_findings": 2}}},
            "RED spill finding: dkdv tile=64 cls=plain",
        ),
        (
            "same",
            {"build_status": "error", "file": "", "detail": "boom"},
            "RED missing: dkdv tile=64 cls=plain",
        ),
        (
            "byte",
            {"build_status": "dual_object", "file": "", "detail": "dual"},
            "RED missing: dkdv tile=64 cls=causal",
        ),
    ],
    ids=["spill-finding", "error", "dual-object-outside-the-flaky-set"],
)
def test_red_rows_exit_nonzero(tmp_path, capsys, row, patch, line) -> None:
    status, lines = _run(_build_fixture(tmp_path, **{row: patch}), capsys)
    assert status == 1
    assert any(l.startswith(line) for l in lines)
    assert "RED: 1 row(s)" in lines


DUAL = {"build_status": "dual_object", "file": "", "detail": "dual"}


def test_a_dq_causal_dual_object_outside_the_flaky_list_is_known_nondeterministic(
    tmp_path, capsys
) -> None:
    argv = _build_fixture(tmp_path)
    flaky = tmp_path / "flaky.csv"
    flaky.write_text("hash,spec,source\n", encoding="utf-8")  # nothing listed
    spec = Spec.parse(SPECS["flaky"])
    assert (spec.kernel, spec.cls) == ("dq", "causal")
    status, lines = _run(argv, capsys)
    assert status == 0
    (line,) = [l for l in lines if "dq tile=64 cls=causal" in l]
    assert (
        line.startswith("known nondeterministic (dq-causal family)")
        and "flaky=no" in line
    )
    assert "RED" not in "\n".join(lines)


@pytest.mark.parametrize("row", ["byte", "resource"], ids=["dkdv-causal", "dq-plain"])
def test_a_dual_object_outside_the_dq_causal_family_is_red(
    tmp_path, capsys, row
) -> None:
    status, lines = _run(_build_fixture(tmp_path, **{row: DUAL}), capsys)
    assert status == 1
    assert sum(l.startswith("RED missing") for l in lines) == 1
    assert "RED: 1 row(s)" in lines


def test_an_op_without_a_manifest_is_missing_and_red(tmp_path, capsys) -> None:
    argv = _build_fixture(tmp_path)
    (tmp_path / "out" / "sdpa_bwd_merged" / "gfx950" / "manifest.json").unlink()
    status, lines = _run(argv, capsys)
    assert status == 1
    assert any(
        l.startswith("RED missing: merged tile=64 cls=plain") and "no manifest" in l
        for l in lines
    )
    assert any("no manifest: sdpa_bwd_merged" in l for l in lines)
    assert "2/5 byte-identical" in lines

    status, lines = _run([*argv, "--only-built"], capsys)
    assert status == 0
    assert "2/4 byte-identical" in lines
    assert any("no manifest: sdpa_bwd_merged" in l for l in lines)


def _pin_gate(verdict: str, *problems: str) -> dict:
    return {
        "rmir": {"verdict": "ok", "spill_findings": 0},
        "pin_gate": {"verdict": verdict, "problems": list(problems)},
    }


PIN_ELF = _elf({MERGED: MERGED_CODE})
COMBINED_ELF = _elf({MERGED: MERGED_CODE, REDUCE: REDUCE_CODE})
COMBINED = {"ship": "combined", "gates": _pin_gate("ok")}
CAUSAL = "dkdv tile=64 cls=causal"


@pytest.fixture(autouse=True)
def _pin_metadata(monkeypatch) -> None:
    """The ELF fixtures carry no AMDGPU metadata note; the pin's only kernel is the merged one."""
    monkeypatch.setattr(
        sdpa_bwd_compare.analyze,
        "amdgpu_metadata",
        lambda blob: {"amdhsa.kernels": [{".name": MERGED}]},
    )


def _combined_fixture(
    tmp_path: Path, shipped: bytes = COMBINED_ELF, **patch
) -> list[str]:
    return _build_fixture(
        tmp_path, {"byte": (PIN_ELF, shipped)}, byte={**COMBINED, **patch}
    )


def test_a_combined_row_is_judged_by_its_per_kernel_gate(tmp_path, capsys) -> None:
    status, lines = _run(_combined_fixture(tmp_path), capsys)
    assert status == 0
    assert "1 combined object(s): 1 pass the per-kernel gate" in lines
    assert not any(
        CAUSAL in l for l in lines
    )  # not "sha differs": not the pin by construction


def test_a_combined_row_with_a_red_gate_is_red(tmp_path, capsys) -> None:
    gates = _pin_gate(
        "red", "reduce resource row differs from the pin bb2f7c351824: vgpr 103 != 81"
    )
    status, lines = _run(_combined_fixture(tmp_path, gates=gates), capsys)
    assert status == 1
    assert "1 combined object(s): 0 pass the per-kernel gate" in lines
    assert any(
        l.startswith(f"RED per-kernel gate: {CAUSAL}")
        and "reduce resource row differs" in l
        for l in lines
    )


def test_a_combined_object_with_another_merged_function_is_red(
    tmp_path, capsys
) -> None:
    tampered = _elf({MERGED: MERGED_CODE[:-1] + b"\xff", REDUCE: REDUCE_CODE})
    status, lines = _run(_combined_fixture(tmp_path, tampered), capsys)
    assert status == 1
    assert any(
        l.startswith(f"RED per-kernel gate: {CAUSAL}")
        and "merged function bytes differ from the pin" in l
        for l in lines
    )


def test_a_combined_object_that_is_not_the_manifests_is_red(tmp_path, capsys) -> None:
    argv = _combined_fixture(tmp_path)
    (tmp_path / "out" / "sdpa_bwd_dkdv" / "gfx950" / "byte.hsaco").write_bytes(
        b"garbage"
    )
    status, lines = _run(argv, capsys)
    assert status == 1
    assert any(l.startswith(f"RED shipped: {CAUSAL}") for l in lines)


def test_a_missing_combined_object_is_red(tmp_path, capsys) -> None:
    argv = _combined_fixture(tmp_path)
    (tmp_path / "out" / "sdpa_bwd_dkdv" / "gfx950" / "byte.hsaco").unlink()
    status, lines = _run(argv, capsys)
    assert status == 1
    assert any(l.startswith(f"RED missing: {CAUSAL}") for l in lines)
