# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shipping the merged objects: which bytes a row ships, when the run stops, and the two-kernel `verify`.

``generate("sdpa_bwd_merged", ...)`` ships the sweep's pins for the 180 rows that never write dB (their
rebuild only verifies) and the combined (merged + dBias reduce) rebuild for the 28 dB-storing
bias rows, whose per-kernel gate must be green; a red gate or a row that does not build ships nothing. The compile
and the toolchain are replaced by fixtures: nothing is compiled.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _sdpa_bwd_cdna_build as build  # noqa: E402
from generators import gen_sdpa_bwd_merged  # noqa: E402
from generators._codeobject import GeneratorError  # noqa: E402
from generators._instances import instances_for  # noqa: E402
from generators._sdpa_bwd_spec import to_spec  # noqa: E402

OP = "sdpa_bwd_merged"
ARCH = "gfx950"
MERGED_SYMBOL = "fmha_bwd_merged_gfx950_kernel_0"
REDUCE_SYMBOL = "fmha_bwd_dbias_reduce_gfx950_kernel_1"
GEOMETRY = {"block_threads": 256, "block_kv": 64, "block_m": 128}
ROW = {
    "vgpr": 100,
    "agpr": 0,
    "vgpr_total": 100,
    "sgpr": 90,
    "sgpr_spill": 0,
    "vgpr_spill": 0,
    "scratch_bytes": 0,
    "lds_bytes": 8192,
    "occupancy": 4,
}

PIN = b"the pinned object"
REBUILT = b"another build of it"
COMBINED = b"the combined object"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows():
    rows = instances_for(OP, ARCH)
    plain = next(i for i in rows if not i.knobs["has_bias"])
    bias = next(i for i in rows if i.knobs["has_bias"] and i.knobs["reduce_symbol"])
    bias_no_db = next(
        i for i in rows if i.knobs["has_bias"] and i.knobs["store_db"] == 0
    )
    return plain, bias, bias_no_db


def _described(symbol: str, params: str, **extra) -> dict:
    signature, _ = build._signature(build._PARAMS[params])
    return {
        "symbol": symbol,
        "kernarg_segment_size": signature[-1][2] + signature[-1][1],
        "targets": [f"amdgcn-amd-amdhsa-unknown-{ARCH}"],
        "signature": signature,
        "named_args": 0,
        "group_segment_fixed_size": 8192,
        "vgpr_count": 100,
        "vgpr_spill_count": 0,
        **extra,
    }


def _gate(blob: bytes, verdict: str = "ok", pin_gate: str | None = None) -> dict:
    gate = {
        "object_sha256": _sha(blob),
        "resources": dict(ROW),
        "static_1087": {},
        "rmir": {
            "verdict": verdict,
            "spill_findings": 0,
            "other_findings": 0,
            "kernels": [MERGED_SYMBOL],
        },
        "per_kernel": {
            MERGED_SYMBOL: {"resources": dict(ROW)},
            REDUCE_SYMBOL: {"resources": {**ROW, "vgpr": 81}},
        },
    }
    if pin_gate:
        gate["pin_gate"] = {
            "verdict": pin_gate,
            "problems": [] if pin_gate == "ok" else ["merged function bytes differ"],
        }
    return gate


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A generate() whose compile returns the outcomes in `rig.outcomes` by row name, over a reference holding `PIN`."""
    plain, bias, bias_no_db = _rows()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / f"{to_spec(plain).hash()}.hsaco").write_bytes(PIN)
    (reference_dir / f"{to_spec(bias_no_db).hash()}.hsaco").write_bytes(PIN)
    reference = {
        to_spec(i).hash(): {
            "hash": to_spec(i).hash(),
            "spec": to_spec(i).key(),
            "sha256": _sha(PIN),
        }
        for i in (plain, bias, bias_no_db)
    }

    class Rig:
        pass

    rig = Rig()
    rig.plain, rig.bias, rig.bias_no_db = plain, bias, bias_no_db
    rig.reference_dir = reference_dir
    rig.out = tmp_path / "content"
    rig.outcomes = {
        plain.name: build.Outcome(
            "ok", REBUILT, gate=_gate(REBUILT), geometry=GEOMETRY
        ),
        bias.name: build.Outcome(
            "ok", COMBINED, gate=_gate(COMBINED, pin_gate="ok"), geometry=GEOMETRY
        ),
        bias_no_db.name: build.Outcome(
            "ok", REBUILT, gate=_gate(REBUILT), geometry=GEOMETRY
        ),
    }
    rig.shipped_gates = []

    def compile_specs(specs, arch, reference_dir=None):
        assert reference_dir == rig.reference_dir
        by_hash = {to_spec(i).hash(): i.name for i in (plain, bias, bias_no_db)}
        return [rig.outcomes[by_hash[s.hash()]] for s in specs]

    def run_gate(obj, greedy, spec, reduce_symbol="", reference_dir=None):
        assert greedy is None
        rig.shipped_gates.append(obj.name)
        return _gate(obj.read_bytes(), verdict="not scanned")

    def verify(blob, spec, arch, where, reduce_symbol=""):
        facts = _described(MERGED_SYMBOL, "merged", block_threads=256)
        if reduce_symbol:
            facts["reduce"] = _described(
                REDUCE_SYMBOL, "dbias_reduce", block_threads=256
            )
        return facts

    monkeypatch.setattr(build, "compile_specs", compile_specs)
    monkeypatch.setattr(build, "_run_gate", run_gate)
    monkeypatch.setattr(build, "verify", verify)
    monkeypatch.setattr(build, "read_reference", lambda: reference)
    monkeypatch.setattr(
        build, "instances_for", lambda op, arch: [plain, bias, bias_no_db]
    )
    monkeypatch.setattr(build.env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(build.env, "prepare", lambda arch: None)
    monkeypatch.setattr(build.env, "provenance", lambda arch: {"flydsl": "test"})

    def generate():
        return build.generate(OP, ARCH, rig.out, None, rig.reference_dir)

    rig.generate = generate
    rig.records = lambda: {
        r["name"]: r
        for r in json.loads((rig.out / OP / ARCH / "manifest.json").read_text())[
            "instances"
        ]
    }
    return rig


def test_a_pinned_row_ships_the_pin_and_a_bias_row_the_combined_rebuild(rig) -> None:
    assert rig.generate() == 0
    records = rig.records()
    op_dir = rig.out / OP / ARCH

    plain = records[rig.plain.name]
    assert (op_dir / plain["file"]).read_bytes() == PIN
    assert plain["sha256"] == plain["pin_sha256"] == _sha(PIN)
    assert plain["rebuilt_sha256"] == _sha(REBUILT)
    assert plain["ship"] == "pin"
    # The shipped bytes are not the ones the rebuild's gates saw: they are analysed again, and R-MIR is not scanned.
    assert plain["gates"]["rmir"]["verdict"] == "not scanned"
    assert plain["gates"]["object_matches_delivered"]
    assert plain["rebuilt_resources"] == ROW
    assert plain["knobs"]["reduce_symbol"] == ""
    assert {k: plain["knobs"][k] for k in GEOMETRY} == GEOMETRY

    bias = records[rig.bias.name]
    assert (op_dir / bias["file"]).read_bytes() == COMBINED
    assert (bias["sha256"], bias["rebuilt_sha256"]) == (_sha(COMBINED), _sha(COMBINED))
    assert bias["pin_sha256"] == _sha(
        PIN
    )  # the standalone merged pin, which the object is gated against
    assert bias["ship"] == "combined"
    assert bias["symbol"] == MERGED_SYMBOL
    assert bias["knobs"]["reduce_symbol"] == REDUCE_SYMBOL
    assert bias["gates"]["pin_gate"]["verdict"] == "ok"
    assert [a["name"] for a in bias["reduce_args"]][:2] == ["DB", "DBias"]
    assert bias["reduce_resources"]["vgpr"] == 81

    # A bias row that never stores dB carries no reduce kernel: it ships its pin like a plain row.
    no_db = records[rig.bias_no_db.name]
    assert (op_dir / no_db["file"]).read_bytes() == PIN
    assert no_db["ship"] == "pin"
    assert no_db["knobs"]["reduce_symbol"] == ""
    assert "pin_gate" not in no_db["gates"] and "reduce_args" not in no_db


def test_a_rebuild_equal_to_the_pin_keeps_its_rmir_verdict(rig) -> None:
    rig.outcomes[rig.plain.name] = build.Outcome(
        "ok", PIN, gate=_gate(PIN, verdict="ok"), geometry=GEOMETRY
    )
    rig.generate()
    plain = rig.records()[rig.plain.name]
    assert plain["gates"]["rmir"]["verdict"] == "ok"
    assert plain["rebuilt_sha256"] == plain["pin_sha256"]
    assert (
        f"{rig.plain.name}.hsaco" not in rig.shipped_gates
    )  # the rebuild's own gates describe the shipped bytes


def test_a_pin_ships_when_its_rebuild_is_a_dual_object(rig) -> None:
    rig.outcomes[rig.plain.name] = build.Outcome(
        "dual_object", detail="embedded objects are not identical", geometry=GEOMETRY
    )
    rig.generate()
    plain = rig.records()[rig.plain.name]
    assert plain["build_status"] == "dual_object"
    assert plain["sha256"] == _sha(PIN) and plain["rebuilt_sha256"] == ""
    assert plain["gates"]["rmir"]["verdict"] == "not scanned"
    assert {k: plain["knobs"][k] for k in GEOMETRY} == GEOMETRY


@pytest.mark.parametrize(
    ("row", "outcome", "reason"),
    [
        (
            "bias",
            build.Outcome(
                "ok", COMBINED, gate=_gate(COMBINED, pin_gate="red"), geometry=GEOMETRY
            ),
            "per-kernel gate red: merged function bytes differ",
        ),
        (
            "bias",
            build.Outcome(
                "ok",
                COMBINED,
                gate=_gate(COMBINED, verdict="red", pin_gate="ok"),
                geometry=GEOMETRY,
            ),
            "R-MIR spill finding",
        ),
        (
            "bias",
            build.Outcome("dual_object", detail="dual", geometry=GEOMETRY),
            "dual_object: dual",
        ),
        (
            "bias",
            build.Outcome("error", detail="boom", geometry=GEOMETRY),
            "error: boom",
        ),
        (
            "plain",
            build.Outcome("error", detail="boom", geometry=GEOMETRY),
            "error: boom",
        ),
    ],
    ids=["red-gate", "red-rmir", "combined-dual-object", "combined-error", "pin-error"],
)
def test_a_red_gate_or_a_row_that_does_not_build_ships_nothing(
    rig, row, outcome, reason
) -> None:
    rig.outcomes[getattr(rig, row).name] = outcome
    with pytest.raises(GeneratorError, match="nothing is shipped") as stop:
        rig.generate()
    assert reason in str(stop.value)
    op_dir = rig.out / OP / ARCH
    assert list(op_dir.iterdir()) == []


def test_a_combined_row_without_a_per_kernel_gate_stops_the_run(rig) -> None:
    rig.outcomes[rig.bias.name] = build.Outcome(
        "ok", COMBINED, gate=_gate(COMBINED), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="no per-kernel gate"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_a_pin_that_is_not_its_reference_object_is_refused_before_compiling(
    rig,
) -> None:
    (rig.reference_dir / f"{to_spec(rig.plain).hash()}.hsaco").write_bytes(b"edited")
    with pytest.raises(GeneratorError, match="not the pin its reference row names"):
        rig.generate()


def test_the_merged_generator_needs_the_reference_dir(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gen_sdpa_bwd_merged.main(["--arch", ARCH, "--out-dir", str(tmp_path)]) == 1
    assert "--reference-dir" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


# --- verify() on a two-kernel object ---------------------------------------------------------


@pytest.fixture
def two_kernels(monkeypatch):
    """`verify` over an object whose kernels are described by `objects`: {symbol: (describe dict, threads)}."""
    _, bias, _ = _rows()
    objects = {
        MERGED_SYMBOL: (_described(MERGED_SYMBOL, "merged"), 256),
        REDUCE_SYMBOL: (_described(REDUCE_SYMBOL, "dbias_reduce"), 256),
    }
    monkeypatch.setattr(build, "kernel_names", lambda blob, where: list(objects))
    monkeypatch.setattr(
        build, "describe", lambda blob, where, symbol=None: objects[symbol][0]
    )
    monkeypatch.setattr(
        build,
        "_max_flat_workgroup_size",
        lambda blob, where, symbol: objects[symbol][1],
    )
    monkeypatch.setattr(build, "verify_arch", lambda described, arch, where: None)

    def verify(reduce_symbol=REDUCE_SYMBOL):
        return build.verify(b"", to_spec(bias), ARCH, "where", reduce_symbol)

    verify.objects = objects
    return verify


def test_verify_checks_both_kernels_of_a_combined_object(two_kernels) -> None:
    facts = two_kernels()
    assert facts["symbol"] == MERGED_SYMBOL and facts["block_threads"] == 256
    assert facts["kernarg_segment_size"] == 452
    assert facts["reduce"]["symbol"] == REDUCE_SYMBOL
    assert facts["reduce"]["kernarg_segment_size"] == 88


def test_verify_refuses_a_reduce_kernel_with_another_layout(two_kernels) -> None:
    described, threads = two_kernels.objects[REDUCE_SYMBOL]
    two_kernels.objects[REDUCE_SYMBOL] = (
        {**described, "signature": described["signature"][:-1]},
        threads,
    )
    with pytest.raises(GeneratorError, match=r"\(%s\).*kernarg layout" % REDUCE_SYMBOL):
        two_kernels()


def test_verify_refuses_a_reduce_kernel_of_another_block_size(two_kernels) -> None:
    described, _ = two_kernels.objects[REDUCE_SYMBOL]
    two_kernels.objects[REDUCE_SYMBOL] = (described, 128)
    with pytest.raises(
        GeneratorError, match="built for 128 threads per workgroup, declared 256"
    ):
        two_kernels()


def test_verify_refuses_an_object_without_the_reduce_kernel(two_kernels) -> None:
    del two_kernels.objects[REDUCE_SYMBOL]
    with pytest.raises(GeneratorError, match="expected a merged kernel"):
        two_kernels()


def test_verify_of_a_merged_kernel_with_the_wrong_layout_is_refused(
    two_kernels,
) -> None:
    described, threads = two_kernels.objects[MERGED_SYMBOL]
    two_kernels.objects[MERGED_SYMBOL] = (
        {**described, "signature": described["signature"][:-1]},
        threads,
    )
    with pytest.raises(GeneratorError, match="kernarg layout"):
        two_kernels()
