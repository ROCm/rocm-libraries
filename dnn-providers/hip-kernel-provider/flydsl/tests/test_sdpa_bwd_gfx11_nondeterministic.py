# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The gfx11-generic generator under the nondeterministic-build policy: a listed row ships its pin and its rebuild must be the pin or a recorded variant, an unlisted row must be its pin.

The compile and the toolchain are replaced by fixtures: nothing is compiled.
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
from generators._codeobject import GeneratorError  # noqa: E402
from generators._instances import instances_for  # noqa: E402

OP = "sdpa_bwd_prep"
ARCH = rdna.ARCH
PIN, OTHER = b"the pinned object", b"another build of it"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    instance = instances_for(OP, ARCH)[0]
    spec = rdna.specs_for("prep")[0]
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / f"{spec.hash()}.hsaco").write_bytes(PIN)
    reference = {
        spec.hash(): {
            "hash": spec.hash(),
            "spec": spec.key(),
            "sha256": sha(PIN),
            **dict.fromkeys(rdna.RESOURCE_COLUMNS, ""),
        }
    }
    signature, names = cdna._signature(cdna._PARAMS["prep"])

    class Rig:
        listed = False
        variants: tuple = ()
        outcomes: list = []
        builds = 0

    rig = Rig()
    rig.instance, rig.spec, rig.out = instance, spec, tmp_path / "content"

    def compile_all(compile_one, tasks, arch, label):
        rig.builds += 1
        return [rig.outcomes.pop(0) for _ in tasks]

    described = {
        "symbol": "k",
        "kernarg_segment_size": signature[-1][2] + signature[-1][1],
        "signature": signature,
        "named_args": 0,
        "group_segment_fixed_size": 0,
        "vgpr_count": 1,
        "block_threads": 64,
    }
    monkeypatch.setattr(cdna, "_compile_all", compile_all)
    monkeypatch.setattr(cdna, "read_reference", lambda path=None: reference)
    monkeypatch.setattr(rdna, "verify", lambda blob, spec, where: described)
    monkeypatch.setattr(rdna, "instances_for", lambda op, arch: [instance])
    monkeypatch.setattr(
        rdna, "specs_for", lambda kind: [spec] if kind == "prep" else []
    )
    monkeypatch.setattr(rdna.env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(rdna.env, "prepare", lambda arch: None)
    monkeypatch.setattr(rdna.env, "provenance", lambda arch: {"flydsl": "test"})
    monkeypatch.setattr(rdna.env, "rocm_version", lambda: "10.2.0")

    def listing() -> Path:
        path = tmp_path / "list.csv"
        nondet.write(
            (
                [
                    nondet.Row(
                        ARCH,
                        spec.hash(),
                        spec.key(),
                        "both",
                        sha(PIN),
                        rig.variants,
                        (),
                        "test",
                    )
                ]
                if rig.listed
                else []
            ),
            path,
        )
        return path

    def generate():
        return rdna.generate(
            OP, rig.out, None, reference_dir, nondeterministic_csv=listing()
        )

    rig.generate = generate
    rig.record = lambda: json.loads(
        (rig.out / OP / ARCH / "manifest.json").read_text()
    )["instances"][0]
    return rig


def test_a_listed_row_ships_its_pin_whatever_its_recorded_build_does(rig) -> None:
    rig.listed = True
    rig.variants = (sha(OTHER),)
    rig.outcomes = [cdna.Outcome("ok", OTHER)]
    rig.generate()
    record = rig.record()
    assert record["sha256"] == sha(PIN) and record["ship"] == "pin"
    assert record["rebuilt_sha256"] == sha(OTHER)

    rig.out = rig.out.parent / "second"
    rig.outcomes = [cdna.Outcome("dual_object", detail="two objects")]
    rig.generate()
    assert (
        rig.record()["sha256"] == sha(PIN) and rig.builds == 2
    )  # built once each, never retried


def test_a_listed_row_whose_rebuild_is_neither_its_pin_nor_a_variant_stops_the_run(
    rig,
) -> None:
    rig.listed = True
    rig.outcomes = [cdna.Outcome("ok", OTHER)]
    with pytest.raises(GeneratorError, match="neither its pin"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_an_unlisted_row_that_is_not_its_pin_stops_the_run(rig) -> None:
    rig.outcomes = [cdna.Outcome("ok", OTHER)]
    with pytest.raises(GeneratorError, match="is not the S1 object"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_an_unlisted_dual_object_is_rebuilt_and_stops_after_the_attempts(rig) -> None:
    rig.outcomes = [cdna.Outcome("dual_object", detail="dual"), cdna.Outcome("ok", PIN)]
    rig.generate()
    assert rig.builds == 2 and rig.record()["sha256"] == sha(PIN)
    rig.out = rig.out.parent / "second"
    rig.builds = 0
    rig.outcomes = [cdna.Outcome("dual_object", detail="dual")] * cdna._ATTEMPTS
    with pytest.raises(GeneratorError, match="dual_object"):
        rig.generate()
    assert rig.builds == cdna._ATTEMPTS
