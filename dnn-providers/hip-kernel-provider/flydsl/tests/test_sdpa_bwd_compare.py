# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward compare tool: what it reports, and what turns the exit status red.

A five-row fixture stands in for a generator run: one row identical to its pin, and four
that are not (a changed byte, a changed resource row, an object the generator never
built, a known-flaky ``dual_object``). The compare is report-only for those. Spill
findings, ``error`` rows and a ``dual_object`` outside the known-flaky set are red.
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


def _build_fixture(tmp_path: Path, **edits) -> list[str]:
    """Reference data and a generator out-dir; returns the compare argv.

    @p edits maps a fixture row name to a record patch applied to its manifest record.
    """
    ref_objects = tmp_path / "ref"
    ref_objects.mkdir()
    with (tmp_path / "ref.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("hash", "spec", "sha256", *RESOURCES))
        for name, key in SPECS.items():
            digest = Spec.parse(key).hash()
            body = f"pin {name}".encode()
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
            "resources": dict(RESOURCES),
            "gates": {"rmir": {"verdict": "ok", "spill_findings": 0}},
        }
        op_dir = out / f"sdpa_bwd_{spec.kernel}" / "gfx950"
        op_dir.mkdir(parents=True, exist_ok=True)
        body = f"pin {name}".encode()
        if name == "byte":
            body = b"pin bytf"
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
    assert lines[0] == "rocm_version: generator 10.2.0; reference unknown"
    assert lines[1] == "2/5 byte-identical"  # the resource-row row has the pin's bytes
    report = "\n".join(lines[2:])
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
    assert lines[-1] == "RED: 1 row(s)"
