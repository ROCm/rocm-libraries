# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The nondeterministic-build policy on the gated arches (gfx942, gfx90a), through ``_sdpa_bwd_split_build.generate`` and the real RDNA-family row builder.

A 16-bit row or a standalone reduce object of the list ships its pin whatever its rebuild did; the gates of such a row are those of the
shipped pin, and its pin needs a recorded clean R-MIR scan. The compile, the toolchain and the gate tool are replaced by fixtures: nothing is
compiled.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _sdpa_bwd_cdna_build as cdna  # noqa: E402
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators import _sdpa_bwd_rdna_build as rdna  # noqa: E402
from generators import _sdpa_bwd_split_build as split  # noqa: E402
from generators._codeobject import GeneratorError  # noqa: E402
from generators._instances import instances_for  # noqa: E402

PIN, OTHER = b"the pinned object", b"another build of it"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gate_of(blob: bytes, verdict: str = "ok") -> dict:
    return {
        "object_sha256": sha(blob),
        "rmir": {
            "verdict": verdict,
            "spill_findings": 0,
            "other_findings": 0,
            "kernels": [],
        },
        "static_1087": {},
    }


@pytest.fixture(params=["gfx942", "gfx90a"])
def rig(request, tmp_path, monkeypatch):
    arch = request.param
    prep = instances_for("sdpa_bwd_prep", arch)
    instance = next(i for i in prep if rdna.family(arch, i.dtype, "prep") == "rdna")
    spec = rdna.specs_for("prep", arch)[0]
    companion = rdna.specs_for("dbias_reduce", arch)[0]
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = {}
    for row in (spec, companion):
        (reference_dir / f"{row.hash()}.hsaco").write_bytes(PIN)
        reference[row.hash()] = {
            "hash": row.hash(),
            "spec": row.key(),
            "sha256": sha(PIN),
            "rmir_scan": "clean",
            **dict.fromkeys(rdna.RESOURCE_COLUMNS, ""),
        }
    signature, _ = cdna._signature(cdna._PARAMS["prep"])
    reduce_signature, _ = cdna._signature(cdna._PARAMS["dbias_reduce"])

    class Rig:
        op = "sdpa_bwd_prep"
        listed: tuple = ()
        variants: tuple = (sha(OTHER),)
        outcomes: list = []

    rig = Rig()
    rig.arch, rig.instance, rig.spec, rig.companion = arch, instance, spec, companion
    rig.reference, rig.out = reference, tmp_path / "content"

    def described(blob, spec, where):
        wanted = reduce_signature if spec.kernel == "dbias_reduce" else signature
        return {
            "symbol": "k",
            "kernarg_segment_size": wanted[-1][2] + wanted[-1][1],
            "signature": wanted,
            "named_args": 0,
            "group_segment_fixed_size": 0,
            "vgpr_count": 1,
            "block_threads": 64,
        }

    def rows_of(op, arch, only=None):
        if op == "sdpa_bwd_dq":
            return [], [], [companion]
        return [instance], [spec], []

    monkeypatch.setattr(
        cdna,
        "_compile_all",
        lambda f, tasks, a, label: [rig.outcomes.pop(0) for _ in tasks],
    )
    monkeypatch.setattr(cdna, "read_reference", lambda path=None: reference)
    monkeypatch.setattr(cdna, "rows_of", lambda op, arch, only=None: ([], []))
    monkeypatch.setattr(cdna, "build_rows", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(
        cdna,
        "_run_gate",
        lambda obj, greedy, spec, *args, **kwargs: gate_of(obj.read_bytes()),
    )
    monkeypatch.setattr(rdna, "verify", described)
    monkeypatch.setattr(rdna, "rows_of", rows_of)
    monkeypatch.setattr(split, "instances_for", lambda op, arch: [instance])
    monkeypatch.setattr(split.env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(split.env, "prepare", lambda arch: None)
    monkeypatch.setattr(split.env, "provenance", lambda arch: {"flydsl": "test"})
    monkeypatch.setattr(split.env, "rocm_version", lambda: "10.2.0")

    def generate():
        path = tmp_path / "list.csv"
        nondet.write(
            [
                nondet.Row(
                    arch,
                    row.hash(),
                    row.key(),
                    "both",
                    sha(PIN),
                    rig.variants,
                    (),
                    "test",
                )
                for row in rig.listed
            ],
            path,
        )
        return split.generate(
            rig.op,
            arch,
            rig.out,
            None,
            reference_dir,
            nondeterministic_csv=path,
        )

    rig.generate = generate
    rig.manifest = lambda: json.loads(
        (rig.out / rig.op / arch / "manifest.json").read_text()
    )
    return rig


def test_a_listed_row_ships_its_pin_with_the_gates_of_the_pin(rig) -> None:
    rig.listed = (rig.spec,)
    rig.outcomes = [cdna.Outcome("ok", OTHER, gate=gate_of(OTHER))]
    rig.generate()
    (record,) = rig.manifest()["instances"]
    assert record["ship"] == "pin" and record["sha256"] == sha(PIN)
    assert record["rebuilt_sha256"] == sha(OTHER)
    assert record["gates"]["object_matches_delivered"]
    assert record["gates"]["rmir"]["verdict"] == "ok"


def test_a_listed_row_whose_rebuild_produced_no_object_still_ships_its_pin(rig) -> None:
    rig.listed = (rig.spec,)
    rig.outcomes = [cdna.Outcome("dual_object", detail="two objects")]
    rig.generate()
    (record,) = rig.manifest()["instances"]
    assert record["sha256"] == sha(PIN) and record["gates"]["object_matches_delivered"]
    assert record["rebuilt_sha256"] == "" and record["rebuilt_rmir"] == ""


def test_a_listed_row_is_refused_without_a_recorded_clean_scan_of_its_pin(rig) -> None:
    rig.listed = (rig.spec,)
    rig.reference[rig.spec.hash()]["rmir_scan"] = ""
    rig.outcomes = [cdna.Outcome("ok", OTHER, gate=gate_of(OTHER))]
    with pytest.raises(GeneratorError, match="no recorded clean R-MIR scan"):
        rig.generate()
    assert list((rig.out / rig.op / rig.arch).iterdir()) == []


def test_a_listed_row_that_is_its_own_red_rebuild_is_refused(rig) -> None:
    rig.listed = (rig.spec,)
    rig.variants = ()
    rig.outcomes = [cdna.Outcome("ok", PIN, gate=gate_of(PIN, "red"))]
    with pytest.raises(GeneratorError, match="has an R-MIR spill finding"):
        rig.generate()


def test_a_listed_reduce_companion_ships_its_pin(rig) -> None:
    rig.op = "sdpa_bwd_dq"
    rig.listed = (rig.companion,)
    rig.outcomes = [cdna.Outcome("ok", OTHER, gate=gate_of(OTHER))]
    rig.generate()
    (companion,) = rig.manifest()["companions"]
    assert companion["sha256"] == sha(PIN)


def test_a_listed_reduce_companion_is_refused_without_a_recorded_clean_scan(
    rig,
) -> None:
    rig.op = "sdpa_bwd_dq"
    rig.listed = (rig.companion,)
    rig.reference[rig.companion.hash()]["rmir_scan"] = ""
    rig.outcomes = [cdna.Outcome("ok", OTHER, gate=gate_of(OTHER))]
    with pytest.raises(GeneratorError, match="no recorded clean R-MIR scan"):
        rig.generate()


def _dq_specs(arch: str):
    specs = cdna.rows_of("sdpa_bwd_dq", arch)[1]
    combined = next(s for s in specs if cdna.reduce_symbol_of(s))
    standalone = next(s for s in specs if not cdna.reduce_symbol_of(s))
    return combined, standalone


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_the_reference_check_needs_no_pin_object_for_a_combined_row(
    arch, tmp_path
) -> None:
    combined, standalone = _dq_specs(arch)
    reference = {
        s.hash(): {"spec": s.key(), "sha256": sha(PIN)} for s in (combined, standalone)
    }
    (tmp_path / f"{standalone.hash()}.hsaco").write_bytes(PIN)
    split._check_reference(
        (combined, standalone), reference, tmp_path, cdna_family=True
    )


def test_the_reference_check_refuses_a_standalone_row_without_its_object(
    tmp_path,
) -> None:
    _, standalone = _dq_specs("gfx942")
    reference = {standalone.hash(): {"spec": standalone.key(), "sha256": sha(PIN)}}
    with pytest.raises(GeneratorError, match=f"{standalone.hash()}.hsaco"):
        split._check_reference((standalone,), reference, tmp_path, cdna_family=True)


def test_the_reference_check_refuses_an_object_that_is_not_the_pin(tmp_path) -> None:
    _, standalone = _dq_specs("gfx942")
    reference = {standalone.hash(): {"spec": standalone.key(), "sha256": sha(PIN)}}
    (tmp_path / f"{standalone.hash()}.hsaco").write_bytes(OTHER)
    with pytest.raises(GeneratorError, match="is not the pin"):
        split._check_reference((standalone,), reference, tmp_path, cdna_family=True)


def test_the_reference_check_refuses_a_combined_row_the_reference_lacks(
    tmp_path,
) -> None:
    combined, _ = _dq_specs("gfx942")
    with pytest.raises(GeneratorError, match="no row"):
        split._check_reference((combined,), {}, tmp_path, cdna_family=True)
