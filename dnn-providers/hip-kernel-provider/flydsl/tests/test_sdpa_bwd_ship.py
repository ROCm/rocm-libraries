# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shipping the merged objects: which bytes a row ships, when the run stops, and the two-kernel `verify`.

``generate("sdpa_bwd_merged", ...)`` ships the rebuild of every gfx950 row except the rows of the nondeterministic list and the rows of
the pinned-exceptions list: a standalone row ships the sweep's pin and a combined (merged + dBias reduce) row the combined object stored in
the shipped content, never a rebuild or a variant. A listed row's rebuild must still be its pin or a recorded variant, and a combined rebuild
needs a green per-kernel gate; a rebuild that is not its pin and is unlisted stops the run, and so do a red gate, an R-MIR spill finding, a
row that does not build and a stored object that is missing or altered. The compile and the toolchain are replaced by fixtures:
nothing is compiled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _sdpa_bwd_cdna_build as build  # noqa: E402
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators import gen_sdpa_bwd_merged  # noqa: E402
from generators._manifest import sha256 as _sha  # noqa: E402
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
STORED = b"the stored combined object"
FUNCTION_VARIANT = "e" * 64


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
    (reference_dir / f"{to_spec(bias_no_db).hash()}.hsaco").write_bytes(PIN)
    reference = {
        to_spec(i).hash(): {
            "hash": to_spec(i).hash(),
            "spec": to_spec(i).key(),
            "sha256": _sha(REBUILT if i is plain else PIN),
            "rmir_scan": "clean",
        }
        for i in (plain, bias, bias_no_db)
    }

    class Rig:
        pass

    rig = Rig()
    rig.plain, rig.bias, rig.bias_no_db = plain, bias, bias_no_db
    rig.reference_dir = reference_dir
    rig.reference = reference
    rig.out = tmp_path / "content"
    # The bias row that never stores dB is the nondeterministic list's standalone row, its rebuild a recorded variant; `rig.exceptions` starts empty.
    rig.exceptions = {}
    rig.listed = {
        to_spec(bias_no_db).hash(): nondet.Row(
            ARCH,
            to_spec(bias_no_db).hash(),
            to_spec(bias_no_db).key(),
            "variants",
            _sha(PIN),
            (_sha(REBUILT),),
            (),
            "test",
        )
    }
    rig.expected_shipped = {to_spec(bias_no_db).hash()}
    rig.shipped_gates = []
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

    def compile_specs(
        specs,
        arch,
        reference_dir=None,
        reference_csv=None,
        rebuild_unstable=False,
        shipped_pins=frozenset(),
    ):
        assert reference_dir == rig.reference_dir
        assert rebuild_unstable
        assert shipped_pins == rig.expected_shipped
        by_hash = {to_spec(i).hash(): i.name for i in (plain, bias, bias_no_db)}
        return [rig.outcomes[by_hash[s.hash()]] for s in specs]

    def run_gate(
        obj, greedy, spec, reduce_symbol="", reference_dir=None, reference_csv=None
    ):
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

    # The tail-zero objects have no pin and no reference row (the fixture's reference holds none); the merged build lists them by path.
    rig.companions = [
        {
            "hash": f"tz{dtype}",
            "spec": f"tailzero,{dtype}",
            "file": f"../../sdpa_bwd_dq/{ARCH}/tz_{dtype}.hsaco",
            "symbol": "tz",
            "sha256": _sha(dtype.encode()),
        }
        for dtype in ("f16", "bf16", "fp32")
    ]

    def build_tailzero(arch, op_dir):
        if isinstance(rig.companions, Exception):
            raise rig.companions
        return rig.companions

    monkeypatch.setattr(build, "compile_specs", compile_specs)
    monkeypatch.setattr(build, "verify", verify)
    monkeypatch.setattr(build, "_run_gate", run_gate)
    monkeypatch.setattr(build, "read_exceptions", lambda path=None: rig.exceptions)
    monkeypatch.setattr(
        build,
        "read_nondeterministic",
        lambda arch, reference, specs=None, path=None: rig.listed,
    )
    monkeypatch.setattr(build, "read_reference", lambda path=None: reference)
    monkeypatch.setattr(
        build, "instances_for", lambda op, arch: [plain, bias, bias_no_db]
    )
    monkeypatch.setattr(build.env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(build.env, "prepare", lambda arch: None)
    monkeypatch.setattr(build.env, "provenance", lambda arch: {"flydsl": "test"})
    monkeypatch.setattr(build.env, "rocm_version", lambda: "10.2.0")

    def generate():
        return build.generate(
            OP,
            ARCH,
            rig.out,
            None,
            rig.reference_dir,
            build_tailzero=build_tailzero,
            stored_dir=rig.stored_dir,
        )

    # The shipped content a listed combined row ships its object from: `rig.store(blob)` records `blob` for the bias row.
    rig.stored_dir = None

    def store(blob, manifest_sha=None, in_manifest=True):
        folder = tmp_path / "stored" / OP / ARCH
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bias.hsaco").write_bytes(blob)
        row = {
            "spec_hash": to_spec(bias).hash(),
            "spec": to_spec(bias).key(),
            "file": "bias.hsaco",
            "sha256": manifest_sha or _sha(blob),
        }
        (folder / "manifest.json").write_text(
            json.dumps({"instances": [row] if in_manifest else []})
        )
        rig.stored_dir = tmp_path / "stored"

    def list_combined(variant_functions=(FUNCTION_VARIANT,)):
        rig.listed[to_spec(bias).hash()] = nondet.Row(
            ARCH,
            to_spec(bias).hash(),
            to_spec(bias).key(),
            "variants",
            _sha(PIN),
            (),
            tuple(variant_functions),
            "test",
        )

    rig.store, rig.list_combined = store, list_combined

    rig.generate = generate
    rig.records = lambda: {
        r["name"]: r
        for r in json.loads((rig.out / OP / ARCH / "manifest.json").read_text())[
            "instances"
        ]
    }
    return rig


def test_every_row_ships_its_rebuild_and_a_bias_row_is_gated_against_its_pin(
    rig,
) -> None:
    assert rig.generate() == 0
    records = rig.records()
    op_dir = rig.out / OP / ARCH

    plain = records[rig.plain.name]
    assert (op_dir / plain["file"]).read_bytes() == REBUILT
    assert plain["sha256"] == _sha(REBUILT)
    assert "ship" not in plain and "pin_sha256" not in plain
    assert plain["gates"]["rmir"]["verdict"] == "ok"
    assert plain["gates"]["object_matches_delivered"]
    assert plain["knobs"]["reduce_symbol"] == ""
    assert {k: plain["knobs"][k] for k in GEOMETRY} == GEOMETRY

    bias = records[rig.bias.name]
    assert (op_dir / bias["file"]).read_bytes() == COMBINED
    assert bias["sha256"] == _sha(COMBINED)
    assert bias["pin_sha256"] == _sha(
        PIN
    )  # the standalone merged pin, which the object is gated against
    assert bias["ship"] == "combined"
    assert bias["symbol"] == MERGED_SYMBOL
    assert bias["knobs"]["reduce_symbol"] == REDUCE_SYMBOL
    assert bias["gates"]["pin_gate"]["verdict"] == "ok"
    assert [a["name"] for a in bias["reduce_args"]][:2] == ["DB", "DBias"]
    assert bias["reduce_resources"]["vgpr"] == 81

    # A bias row that never stores dB carries no reduce kernel; this one is a listed standalone row: it ships the pin.
    no_db = records[rig.bias_no_db.name]
    assert (op_dir / no_db["file"]).read_bytes() == PIN
    assert no_db["sha256"] == no_db["pin_sha256"] == _sha(PIN)
    assert (no_db["ship"], no_db["pin_reason"]) == ("pin", "nondeterministic")
    assert no_db["rebuilt_sha256"] == _sha(REBUILT)
    assert no_db["gates"]["rmir"]["verdict"] == "not scanned"
    assert no_db["knobs"]["reduce_symbol"] == ""
    assert "pin_gate" not in no_db["gates"] and "reduce_args" not in no_db
    # The bytes of a listed row are the pin's, whatever its rebuild does.
    assert rig.shipped_gates == [f"{rig.bias_no_db.name}.hsaco"]


def test_the_manifest_lists_the_tailzero_companions_per_dtype_and_role_without_a_pin(
    rig,
) -> None:
    assert rig.generate() == 0
    op_dir = rig.out / OP / ARCH
    listed = json.loads((op_dir / "manifest.json").read_text())["companions"]
    assert {tuple(c["spec"].split(",")[::-1]) for c in listed} == {
        ("f16", "tailzero"),
        ("bf16", "tailzero"),
        ("fp32", "tailzero"),
    }
    assert all(c["file"].startswith(f"../../sdpa_bwd_dq/{ARCH}/") for c in listed)
    assert "pin_sha256" not in listed[0]
    # The merged folder holds its own rows' objects only.
    assert {p.name for p in op_dir.glob("*.hsaco")} == {
        r["file"] for r in rig.records().values()
    }


def test_a_tailzero_object_that_does_not_build_ships_nothing(rig) -> None:
    rig.companions = GeneratorError("tailzero: two builds differ")
    with pytest.raises(GeneratorError, match="two builds differ"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_a_tuned_row_without_a_reference_row_still_stops_the_run(rig) -> None:
    reference = build.read_reference()
    del reference[build.to_spec(rig.plain).hash()]
    with pytest.raises(GeneratorError, match="the reference has no row"):
        rig.generate()


def test_source_md_gives_the_full_regenerate_line(
    rig,
) -> None:
    rig.generate()
    source_md = (rig.out / OP / f"{ARCH}.SOURCE.md").read_text()
    assert (
        f"python -m generators.gen_{OP} --arch {ARCH} --out-dir <content root> "
        "--reference-dir <S1 reference dir>\n"
    ) in source_md
    assert f"| `{rig.bias_no_db.name}` | nondeterministic |" in source_md


def test_a_listed_row_ships_its_pin_when_its_rebuild_is_a_dual_object(rig) -> None:
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "dual_object", detail="two objects", geometry=GEOMETRY
    )
    rig.generate()
    no_db = rig.records()[rig.bias_no_db.name]
    assert no_db["build_status"] == "dual_object"
    assert no_db["sha256"] == _sha(PIN) and no_db["rebuilt_sha256"] == ""


def test_a_listed_row_ships_its_pin_when_its_rebuild_is_a_recorded_variant(
    rig,
) -> None:
    variant = b"a recorded variant"
    rig.listed[to_spec(rig.bias_no_db).hash()] = nondet.Row(
        ARCH,
        to_spec(rig.bias_no_db).hash(),
        to_spec(rig.bias_no_db).key(),
        "variants",
        _sha(PIN),
        (_sha(variant),),
        (),
        "test",
    )
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", variant, gate=_gate(variant), geometry=GEOMETRY
    )
    assert rig.generate() == 0
    no_db = rig.records()[rig.bias_no_db.name]
    assert no_db["sha256"] == _sha(PIN)
    assert no_db["rebuilt_sha256"] == _sha(variant)


def test_a_listed_row_whose_rebuild_has_a_recorded_main_function_ships_its_pin(
    rig,
) -> None:
    function = "f" * 64
    key = to_spec(rig.bias_no_db).hash()
    rig.listed[key] = nondet.Row(
        ARCH,
        key,
        to_spec(rig.bias_no_db).key(),
        "variants",
        _sha(PIN),
        (),
        (function,),
        "test",
    )
    gate = _gate(b"another object")
    gate["per_kernel"] = {
        MERGED_SYMBOL: {"resources": dict(ROW), "function_sha256": function}
    }
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", b"another object", gate=gate, geometry=GEOMETRY
    )
    assert rig.generate() == 0
    assert rig.records()[rig.bias_no_db.name]["sha256"] == _sha(PIN)


def test_a_listed_standalone_row_whose_rebuild_matches_nothing_stops_the_run(
    rig,
) -> None:
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", b"a third object", gate=_gate(b"a third object"), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="neither its pin") as stop:
        rig.generate()
    assert rig.bias_no_db.name in str(stop.value)
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_a_listed_combined_row_ships_its_stored_object_not_its_recorded_variant_rebuild(
    rig,
) -> None:
    rig.list_combined()
    rig.store(STORED)
    assert rig.generate() == 0
    bias = rig.records()[rig.bias.name]
    assert (rig.out / OP / ARCH / bias["file"]).read_bytes() == STORED
    assert bias["sha256"] == _sha(STORED)
    assert bias["rebuilt_sha256"] == _sha(COMBINED) != bias["sha256"]
    assert (bias["ship"], bias["pin_reason"]) == ("combined", "nondeterministic")
    assert bias["pin_sha256"] == _sha(PIN)
    assert bias["gates"]["pin_gate"]["verdict"] == "ok"
    assert bias["knobs"]["reduce_symbol"] == REDUCE_SYMBOL
    assert bias["reduce_resources"]["vgpr"] == 81
    # The bytes the run analysed are the stored ones, not the rebuild.
    assert f"{rig.bias.name}.hsaco" in rig.shipped_gates
    source_md = (rig.out / OP / f"{ARCH}.SOURCE.md").read_text()
    assert (
        f"| `{rig.bias.name}` | nondeterministic, stored combined object |" in source_md
    )


def test_a_listed_combined_row_whose_rebuild_fails_its_gate_stops_though_its_object_is_stored(
    rig,
) -> None:
    rig.list_combined()
    rig.store(STORED)
    rig.outcomes[rig.bias.name] = build.Outcome(
        "ok", COMBINED, gate=_gate(COMBINED, pin_gate="red"), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="per-kernel gate red"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        (lambda rig: None, "needs --stored-dir"),
        (lambda rig: rig.store(STORED, in_manifest=False), "has no object for"),
        (
            lambda rig: rig.store(STORED, manifest_sha=_sha(b"other")),
            "not the object its manifest names",
        ),
    ],
    ids=["no stored dir", "no manifest row", "altered object"],
)
def test_a_listed_combined_row_without_an_intact_stored_object_stops_the_run(
    rig, setup, match
) -> None:
    rig.list_combined()
    setup(rig)
    with pytest.raises(GeneratorError, match=match):
        rig.generate()
    assert not (rig.out / OP / ARCH).exists()


def test_an_exception_row_ships_its_pin_with_the_exceptions_reason(rig) -> None:
    key = to_spec(rig.bias_no_db).hash()
    rig.listed = {}
    rig.exceptions = {
        key: {
            "hash": key,
            "spec": to_spec(rig.bias_no_db).key(),
            "pin_sha256": _sha(PIN),
            "reason": "known-bad-fresh-build",
            "evidence": "test",
        }
    }
    assert rig.generate() == 0
    assert rig.records()[rig.bias_no_db.name]["pin_reason"] == "known-bad-fresh-build"


def _combined_exception(rig, pin_sha256: str) -> None:
    key = to_spec(rig.bias).hash()
    rig.exceptions = {
        key: {
            "hash": key,
            "spec": to_spec(rig.bias).key(),
            "pin_sha256": pin_sha256,
            "reason": "known-bad-fresh-build",
            "evidence": "test",
        }
    }
    rig.expected_shipped = {key, to_spec(rig.bias_no_db).hash()}


def test_a_combined_exception_row_ships_its_stored_object_whatever_its_rebuild_is(
    rig,
) -> None:
    """The rebuild is the known-bad build: red in R-MIR and the per-kernel gate, and never the shipped bytes."""
    _combined_exception(rig, _sha(STORED))
    rig.store(STORED)
    rig.outcomes[rig.bias.name] = build.Outcome(
        "ok",
        COMBINED,
        gate=_gate(COMBINED, verdict="red", pin_gate="red"),
        geometry=GEOMETRY,
    )
    assert rig.generate() == 0
    bias = rig.records()[rig.bias.name]
    assert (rig.out / OP / ARCH / bias["file"]).read_bytes() == STORED
    assert (bias["ship"], bias["pin_reason"]) == ("combined", "known-bad-fresh-build")
    assert bias["sha256"] == _sha(STORED) and bias["pin_sha256"] == _sha(PIN)


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        (lambda rig: None, "needs --stored-dir"),
        (lambda rig: rig.store(STORED, in_manifest=False), "has no object for"),
        (
            lambda rig: rig.store(STORED, manifest_sha=_sha(b"other")),
            "not the object its manifest names",
        ),
        (
            lambda rig: rig.store(b"another stored object"),
            "not the stored combined object",
        ),
    ],
    ids=["no stored dir", "no manifest row", "altered object", "other object"],
)
def test_a_combined_exception_row_without_its_stored_object_stops_the_run(
    rig, setup, match
) -> None:
    _combined_exception(rig, _sha(STORED))
    setup(rig)
    with pytest.raises(GeneratorError, match=match):
        rig.generate()
    assert not (rig.out / OP / ARCH).exists()


def test_a_plain_exception_row_must_name_its_reference_pin(rig) -> None:
    key = to_spec(rig.bias_no_db).hash()
    rig.listed = {}
    rig.exceptions = {
        key: {
            "hash": key,
            "spec": to_spec(rig.bias_no_db).key(),
            "pin_sha256": _sha(STORED),
            "reason": "known-bad-fresh-build",
            "evidence": "test",
        }
    }
    with pytest.raises(GeneratorError, match="not the pin its reference row names"):
        rig.generate()


def test_an_unlisted_row_whose_build_is_not_its_pin_stops_the_run(rig) -> None:
    rig.outcomes[rig.plain.name] = build.Outcome(
        "ok", b"another object", gate=_gate(b"another object"), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="sample and list it") as stop:
        rig.generate()
    assert rig.plain.name in str(stop.value)
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_a_combined_row_is_rebuilt_until_its_dual_object_resolves(monkeypatch) -> None:
    _, bias, _ = _rows()
    results = iter(
        [
            build.Outcome("dual_object", detail="two objects"),
            build.Outcome("dual_object", detail="two objects"),
            build.Outcome("ok", COMBINED, gate=_gate(COMBINED, pin_gate="ok")),
        ]
    )
    monkeypatch.setattr(
        build, "_compile_all", lambda compile_one, tasks, arch, label: [next(results)]
    )
    (outcome,) = build.compile_specs(
        [to_spec(bias)], ARCH, rebuild_unstable=True, shipped_pins=frozenset()
    )
    assert (outcome.status, outcome.attempts) == ("ok", 3)


def test_a_row_that_ships_its_pin_is_built_once_and_the_others_are_rebuilt(
    monkeypatch,
) -> None:
    plain, _, pinned = _rows()
    rounds: list[list[str]] = []

    def compile_all(compile_one, tasks, arch, label):
        rounds.append([task[1] for task in tasks])
        return [build.Outcome("dual_object", detail="two objects") for _ in tasks]

    monkeypatch.setattr(build, "_compile_all", compile_all)
    outcomes = build.compile_specs(
        [to_spec(plain), to_spec(pinned)],
        ARCH,
        rebuild_unstable=True,
        shipped_pins=frozenset({to_spec(pinned).hash()}),
    )
    assert rounds[0] == [to_spec(plain).hash(), to_spec(pinned).hash()]
    assert all(r == [to_spec(plain).hash()] for r in rounds[1:])
    assert len(rounds) == build._ATTEMPTS
    assert [o.attempts for o in outcomes] == [build._ATTEMPTS, 1]


def test_a_listed_pin_that_is_not_its_reference_object_is_refused(rig) -> None:
    (rig.reference_dir / f"{to_spec(rig.bias_no_db).hash()}.hsaco").write_bytes(
        b"edited"
    )
    with pytest.raises(GeneratorError, match="not the pin its reference row names"):
        rig.generate()


def test_a_build_without_a_known_rocm_version_is_refused(rig, monkeypatch) -> None:
    monkeypatch.setattr(build.env, "rocm_version", lambda: "unknown")
    with pytest.raises(GeneratorError, match="ROCM_PATH"):
        rig.generate()
    assert not (rig.out / OP).exists()


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
        (
            "plain",
            build.Outcome("dual_object", detail="dual", geometry=GEOMETRY),
            "dual_object: dual",
        ),
        (
            "plain",
            build.Outcome(
                "ok", REBUILT, gate=_gate(REBUILT, verdict="red"), geometry=GEOMETRY
            ),
            "R-MIR spill finding",
        ),
    ],
    ids=[
        "red-gate",
        "red-rmir",
        "combined-dual-object",
        "combined-error",
        "plain-error",
        "plain-dual-object",
        "plain-red-rmir",
    ],
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


def test_a_build_that_was_not_scanned_ships_nothing(rig) -> None:
    rig.outcomes[rig.plain.name] = build.Outcome(
        "ok", REBUILT, gate=_gate(REBUILT, verdict="not scanned"), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="R-MIR not scanned") as stop:
        rig.generate()
    assert rig.plain.name in str(stop.value)
    assert list((rig.out / OP / ARCH).iterdir()) == []


def test_a_red_build_is_reported_as_red_not_as_a_changed_pin(rig) -> None:
    """The R-MIR finding is the reason even when the red build also differs from its pin."""
    rig.outcomes[rig.plain.name] = build.Outcome(
        "ok",
        b"another object",
        gate=_gate(b"another object", verdict="red"),
        geometry=GEOMETRY,
    )
    with pytest.raises(GeneratorError, match="R-MIR spill finding") as stop:
        rig.generate()
    assert "sample and list it" not in str(stop.value)


def test_a_listed_pin_that_is_its_own_red_rebuild_stops_the_run(rig) -> None:
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", PIN, gate=_gate(PIN, verdict="red"), geometry=GEOMETRY
    )
    with pytest.raises(
        GeneratorError, match="its own rebuild and has an R-MIR"
    ) as stop:
        rig.generate()
    assert rig.bias_no_db.name in str(stop.value)


@pytest.mark.parametrize("scan", ["", "unscanned"])
def test_a_shipped_pin_without_a_recorded_clean_scan_stops_the_run(rig, scan) -> None:
    """A stored pin ships only when its reference row records the pin's own scan (`clean`, or `exception` for a pinned exception)."""
    rig.reference[to_spec(rig.bias_no_db).hash()]["rmir_scan"] = scan
    with pytest.raises(GeneratorError, match="no recorded clean R-MIR scan") as stop:
        rig.generate()
    assert rig.bias_no_db.name in str(stop.value)


def test_a_pinned_exception_ships_with_an_exception_scan_record(rig) -> None:
    key = to_spec(rig.bias_no_db).hash()
    rig.listed = {}
    rig.exceptions = {
        key: {
            "hash": key,
            "spec": to_spec(rig.bias_no_db).key(),
            "pin_sha256": _sha(PIN),
            "reason": "known-bad-fresh-build",
            "evidence": "test",
        }
    }
    rig.reference[key]["rmir_scan"] = "exception"
    assert rig.generate() == 0


def test_a_listed_pin_whose_rebuild_is_a_red_variant_still_ships(rig) -> None:
    """A variant is no scan of the pin: it is a build-gate tolerance only."""
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", REBUILT, gate=_gate(REBUILT, verdict="red"), geometry=GEOMETRY
    )
    assert rig.generate() == 0


def test_an_exception_pin_that_is_its_own_red_rebuild_ships(rig) -> None:
    """A pinned exception is covered by its device screen, not by R-MIR."""
    key = to_spec(rig.bias_no_db).hash()
    rig.listed = {}
    rig.exceptions = {
        key: {
            "hash": key,
            "spec": to_spec(rig.bias_no_db).key(),
            "pin_sha256": _sha(PIN),
            "reason": "known-bad-fresh-build",
            "evidence": "test",
        }
    }
    rig.outcomes[rig.bias_no_db.name] = build.Outcome(
        "ok", PIN, gate=_gate(PIN, verdict="red"), geometry=GEOMETRY
    )
    assert rig.generate() == 0


@pytest.mark.parametrize(("verdict", "rc"), [("ok", 0), ("red", 1), ("not scanned", 1)])
def test_the_specs_mode_fails_on_an_object_that_is_not_scanned_clean(
    tmp_path, monkeypatch, capsys, verdict, rc
) -> None:
    plain, _, _ = _rows()
    spec = to_spec(plain)
    specs = tmp_path / "specs.csv"
    specs.write_text(f'spec\n"{spec.key()}"\n')
    monkeypatch.setattr(
        build,
        "compile_specs",
        lambda entries, arch: [
            build.Outcome("ok", REBUILT, gate=_gate(REBUILT, verdict=verdict))
        ],
    )
    monkeypatch.setattr(build.env, "assert_flydsl_version", lambda: None)
    monkeypatch.setattr(build.env, "prepare", lambda arch: None)
    out = tmp_path / "out"
    assert (
        build.run(["--specs", str(specs), "--arch", ARCH, "--out-dir", str(out)]) == rc
    )
    assert (out / "results.csv").is_file()
    assert ("STOP:" in capsys.readouterr().err) == bool(rc)


def test_a_combined_row_without_a_per_kernel_gate_stops_the_run(rig) -> None:
    rig.outcomes[rig.bias.name] = build.Outcome(
        "ok", COMBINED, gate=_gate(COMBINED), geometry=GEOMETRY
    )
    with pytest.raises(GeneratorError, match="no per-kernel gate"):
        rig.generate()
    assert list((rig.out / OP / ARCH).iterdir()) == []


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
    assert facts["kernarg_segment_size"] == 604
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


def test_every_pinned_exception_names_its_reference_row() -> None:
    """A plain row ships the reference pin; a combined row ships a stored combined object, so its pin differs from the reference's."""
    reference = build.read_reference()
    nondeterministic = nondet.read()
    exceptions = build.read_exceptions()
    assert exceptions
    for h, row in exceptions.items():
        spec = build.Spec.parse(row["spec"])
        assert spec.hash() == h
        assert spec.kernel != "merged" or h not in nondeterministic
        assert reference[h]["spec"] == row["spec"]
        assert (reference[h]["sha256"] == row["pin_sha256"]) != bool(
            build.reduce_symbol_of(spec)
        ), h
