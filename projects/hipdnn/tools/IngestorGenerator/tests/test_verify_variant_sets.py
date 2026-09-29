# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The variant-set gate must fail on each defect it claims to catch: every branch must
be able to reach exit 1, so each class introduces one defect in isolation.

NO PRODUCER IS IMPORTED, here or by the tool. The full-mode fixtures build the
compiler's evidence with `hkp_pack.agreement` itself, so the battery runs on a machine
that has never had rocKE installed. That is the property under test.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_variant_sets.py"

sys.path.insert(0, str(_TOOL.parent))

import verify_variant_sets as gate_module  # noqa: E402

sys.path.insert(0, str(gate_module._agreement_python_root()))

from hkp_pack import agreement  # noqa: E402
from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_pack.kpack_resolver import load_kpack  # noqa: E402

_KMD_ID = "11111111-1111-1111-1111-111111111111"
_UED_ID = "22222222-2222-2222-2222-222222222222"
_KDP_ID = "33333333-3333-3333-3333-333333333333"

_PROFILE = """
bundle: test_engine
vocabulary:
  dtype: [BF16, FP16]
"""

_KMD_FIELDS = [
    {"name": "dtype", "type": "string"},
    {"name": "head_size", "type": "int", "default_value": 128},
    {"name": "seqlen_q", "type": "int", "default_value": 512},
    {"name": "use_exp2_fast", "type": "int", "default_value": -1},
    # Fields the metadata-matches-spec cases perturb. Declared here because a metadata
    # key the KMD does not know is a different defect (it drops the whole pack at load)
    # and would fail those cases for the wrong reason.
    {"name": "ragged", "type": "int", "default_value": 0},
    {"name": "varlen", "type": "int", "default_value": 0},
    {"name": "persist_decode", "type": "string", "default_value": "auto"},
]

#: The declaration a UKD carries when the twin and full-mode cases need the gate to
#: know which field the compiler settles: exactly the six keys
#: `agreement.validate_consumer` requires, partitioning this KMD.
_DECLARATION = {
    "engine_id": _UED_ID,
    "kmd_id": _KMD_ID,
    "metadata_fields": ["use_exp2_fast"],
    "matcher_only_fields": [
        "dtype",
        "head_size",
        "seqlen_q",
        "ragged",
        "varlen",
        "persist_decode",
    ],
    "bindings": {"use_exp2_fast": {"method": "effective_use_exp2_fast"}},
    "vocabulary": {},
}


def _contract(descriptor: dict) -> dict:
    """`descriptor` with the specialization declaration its engine expects."""
    out = copy.deepcopy(descriptor)
    out.setdefault("provenance", {})["specialization_contract"] = {
        "schema_version": 1,
        "consumers": [copy.deepcopy(_DECLARATION)],
    }
    return out


def _descriptor(name: str, seqlen_q: int, use_exp2_fast: int | None = None) -> dict:
    """A descriptor whose metadata agrees with its spec. `use_exp2_fast` absent from the
    spec is the authoring form for "the kernel settles this at build time"; the metadata
    still states which binary resulted, since a field absent there takes the KMD
    default."""
    spec = {"dtype": "bf16", "head_size": 128, "seqlen_q": seqlen_q}
    if use_exp2_fast is not None:
        spec["use_exp2_fast"] = use_exp2_fast
    return {
        "version": "1.0",
        "id": f"id-{name}",
        "name": name,
        "kernel_source": {"kind": "rocke", "builder": "build_test", "spec": spec},
        "metadata": {
            "dtype": "BF16",
            "head_size": 128,
            "seqlen_q": seqlen_q,
            "use_exp2_fast": 1 if seqlen_q >= 4096 else 0,
        },
        "priority": 0,
    }


def _pinned_descriptor(name: str, seqlen_q: int, use_exp2_fast: int) -> dict:
    """PINS `use_exp2_fast` in its spec rather than leaving it to the kernel: the shape
    of an override, as opposed to `_descriptor()`."""
    out = _descriptor(name, seqlen_q, use_exp2_fast)
    out["metadata"]["use_exp2_fast"] = use_exp2_fast
    return out


def _kmd(fields=None, ident=_KMD_ID) -> dict:
    return {"version": "1.0", "id": ident, "fields": fields or _KMD_FIELDS}


def _ued(metadata=_KMD_ID, ident=_UED_ID) -> dict:
    return {
        "version": "1.0",
        "id": ident,
        "name": "test:Engine",
        "metadata": metadata,
    }


def _kdp(descriptors, ident=_KDP_ID, engine=_UED_ID, arch=None) -> dict:
    doc = {
        "version": "1.0",
        "id": ident,
        "engine": engine,
        "kernelDescriptors": descriptors,
    }
    if arch is not None:
        doc["arch"] = arch
    return doc


@pytest.fixture
def gate(tmp_path):
    """A working gate environment: an id-wired bundle and a nesting pair."""
    profile = tmp_path / "profile.yaml"
    profile.write_text(_PROFILE)

    def write(tag: str, descriptors: list[dict], fields=None, arch=None) -> Path:
        root = tmp_path / tag
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kdp.json").write_text(
            json.dumps(_kdp(descriptors, arch=arch))
        )
        (root / "test_engine.ued.json").write_text(json.dumps(_ued()))
        (root / "test_engine.kmd.json").write_text(json.dumps(_kmd(fields)))
        return root

    def run(*args, profiled: bool = True, mode: str = "structural"):
        argv = [sys.executable, str(_TOOL), *args, "--mode", mode]
        if profiled:
            argv += ["--profile", str(profile)]
        return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)

    small = [_descriptor("k_sq512", 512), _descriptor("k_sq4096", 4096)]
    big = small + [_descriptor("k_sq8192", 8192)]
    write("small", small)
    write("big", big)

    return type(
        "Gate",
        (),
        {
            "write": staticmethod(write),
            "run": staticmethod(run),
            "small": small,
            "big": big,
            "tmp": tmp_path,
            "profile": profile,
        },
    )


class TestModeIsAlwaysStated:
    """Neither claim may be made by default."""

    def test_omitting_the_mode_is_a_usage_error(self, gate):
        result = subprocess.run(
            [sys.executable, str(_TOOL), "small", "small"],
            cwd=gate.tmp,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--mode" in result.stderr


class TestGatePasses:
    """The control: every failure assertion below is worthless without this."""

    def test_a_clean_nesting_pair_passes(self, gate):
        result = gate.run("small", "small", "big", "big")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "GATE PASSED" in result.stdout


class TestTheSchemaIsReachedByReference:
    """Reaching the KMD by filename surgery -- swapping `.kdp.json` for `.kmd.json` on
    the same stem -- answers "which schema governs this bundle" with a coincidence of
    naming, so every defaulted field, completed tuple and type is decided by the wrong
    document."""

    def test_a_correctly_wired_bundle_resolves(self, gate):
        result = gate.run("small", "small")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_kdp_whose_engine_matches_nothing_fails_naming_the_hop(self, gate):
        root = gate.write("dangling_engine", gate.small)
        (root / "test_engine.kdp.json").write_text(
            json.dumps(_kdp(gate.small, engine="no-such-ued"))
        )
        result = gate.run("bad", "dangling_engine")
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "engine" in combined
        assert "no-such-ued" in combined

    def test_a_ued_whose_metadata_matches_nothing_fails_naming_the_hop(self, gate):
        root = gate.write("dangling_metadata", gate.small)
        (root / "test_engine.ued.json").write_text(
            json.dumps(_ued(metadata="no-such-kmd"))
        )
        result = gate.run("bad", "dangling_metadata")
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "metadata" in combined
        assert "no-such-kmd" in combined

    def test_two_documents_claiming_one_id_fail_naming_both(self, gate):
        root = gate.write("ambiguous", gate.small)
        (root / "other_engine.kmd.json").write_text(json.dumps(_kmd()))
        result = gate.run("bad", "ambiguous")
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "test_engine.kmd.json" in combined
        assert "other_engine.kmd.json" in combined

    def test_a_same_stem_pair_that_is_not_wired_by_id_fails(self, gate):
        root = gate.write("stem_only", gate.small)
        doc = _kdp(gate.small)
        doc.pop("engine")
        (root / "test_engine.kdp.json").write_text(json.dumps(doc))
        result = gate.run("bad", "stem_only")
        assert result.returncode == 1
        assert "engine" in (result.stdout + result.stderr)


class TestGateCatchesEachDefect:

    def test_catches_a_shipped_sentinel(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = -1
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "unset sentinel" in result.stdout

    def test_catches_the_builders_vocabulary_in_metadata(self, gate):
        # Loads cleanly, reconciles on every count, matches nothing.
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["dtype"] = "bf16"
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "wrong vocabulary" in result.stdout

    def test_catches_a_duplicate_loader_tuple(self, gate):
        # A duplicate drops the WHOLE ENGINE at load, not the offending entry.
        bad = copy.deepcopy(gate.small)
        bad[1]["metadata"] = copy.deepcopy(bad[0]["metadata"])
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "loader-tuple collisions" in result.stdout

    def test_catches_a_superset_that_lost_a_binary(self, gate):
        # The property the whole comparison rests on: the larger set must still be
        # able to choose everything the smaller one could.
        short = [_descriptor("k_sq4096", 4096), _descriptor("k_sq8192", 8192)]
        gate.write("short", short)
        result = gate.run("small", "small", "big", "short")
        assert result.returncode == 1
        assert "MISSING" in result.stdout

    def test_tuple_check_substitutes_kmd_defaults_like_the_loader(self, gate):
        """Absent key and explicit default are ONE catalog entry: the two descriptors
        differ on disk and collide only after the loader applies default_value."""
        pinned = _descriptor("k_pinned", 512)
        unset = _descriptor("k_unset", 512)
        unset["metadata"].pop("seqlen_q")
        gate.write("bad", [pinned, unset])
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "loader-tuple collisions" in result.stdout


class TestTheDeskCheckIdentityIsEngineWideAndArchAware:
    """The loader assembles ONE catalog per engine per device, so the identity is
    engine-wide and carries the effective architecture: an identical tuple in a gfx942
    and a gfx950 pack is legal, while an overlap -- including a wildcard over a concrete
    arch -- drops the engine."""

    def _two_packs(self, gate, tag, left_arch, right_arch):
        root = gate.tmp / tag
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.ued.json").write_text(json.dumps(_ued()))
        (root / "test_engine.kmd.json").write_text(json.dumps(_kmd()))
        (root / "test_engine.kdp.json").write_text(
            json.dumps(
                _kdp([_descriptor("k_left", 512)], ident="kdp-left", arch=left_arch)
            )
        )
        (root / "second_pack.kdp.json").write_text(
            json.dumps(
                _kdp([_descriptor("k_right", 512)], ident="kdp-right", arch=right_arch)
            )
        )
        return root

    def test_equal_tuples_on_disjoint_arches_are_both_accepted(self, gate):
        self._two_packs(gate, "disjoint", ["gfx942"], ["gfx950"])
        result = gate.run("d", "disjoint", profiled=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "loader-tuple" not in result.stdout

    def test_equal_tuples_overlapping_on_one_arch_collide(self, gate):
        self._two_packs(gate, "overlap", ["gfx942", "gfx950"], ["gfx950"])
        result = gate.run("o", "overlap", profiled=False)
        assert result.returncode == 1, result.stdout
        assert "loader-tuple collisions" in result.stdout
        assert "gfx950" in result.stdout

    def test_a_wildcard_arch_overlaps_a_concrete_one(self, gate):
        """An absent arch list is "every device", so it meets the concrete pack on that
        pack's own device."""
        self._two_packs(gate, "wildcard", None, ["gfx950"])
        result = gate.run("w", "wildcard", profiled=False)
        assert result.returncode == 1, result.stdout
        assert "loader-tuple collisions" in result.stdout

    def test_one_and_one_point_zero_are_one_tuple_on_a_float_field(self, gate):
        """The catalog holds a value of the field's declared type, so 1 and 1.0 on a
        FLOAT field are one entry; comparing JSON spellings raw lets the duplicate
        ship."""
        fields = [
            {"name": "dtype", "type": "string"},
            {"name": "scale", "type": "float", "default_value": 1.0},
        ]
        left = {
            "version": "1.0",
            "id": "id-left",
            "name": "k_int",
            "kernel_source": {"kind": "rocke", "builder": "b", "spec": {}},
            "metadata": {"dtype": "BF16", "scale": 1},
        }
        right = copy.deepcopy(left)
        right["id"], right["name"] = "id-right", "k_float"
        right["metadata"]["scale"] = 1.0
        gate.write("floaty", [left, right], fields=fields)
        result = gate.run("f", "floaty", profiled=False)
        assert result.returncode == 1, result.stdout
        assert "loader-tuple collisions" in result.stdout


class TestStructuralModeDegradesLoudly:
    """Structural mode must narrow, and must say which claim it did not make."""

    def test_structural_checks_still_run(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = -1
        gate.write("bad", bad)
        result = gate.run("bad", "bad", profiled=False)
        assert result.returncode == 1, "a sentinel needs no evidence to spot"
        assert "unset sentinel" in result.stdout

    def test_compiled_agreement_is_named_not_silently_skipped(self, gate):
        result = gate.run("small", "small")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "NOT CHECKED" in result.stdout
        assert "COMPILED SPECIALIZATION AGREEMENT" in result.stdout
        assert "GATE PASSED on what it checked" in result.stdout
        assert "GATE PASSED:" not in result.stdout, (
            "the unqualified pass line asserts compiled agreement, which this mode "
            "never checked"
        )


class TestGateRefusesAmbiguity:
    def test_two_engines_without_a_pin_is_an_error(self, gate):
        root = gate.write("multi", gate.small)
        (root / "second_engine.ued.json").write_text(
            json.dumps(_ued(metadata="kmd-second", ident="ued-second"))
        )
        (root / "second_engine.kmd.json").write_text(
            json.dumps(_kmd(ident="kmd-second"))
        )
        (root / "second_engine.kdp.json").write_text(
            json.dumps(
                _kdp(
                    [_descriptor("k_other", 1024)],
                    ident="kdp-second",
                    engine="ued-second",
                )
            )
        )
        result = gate.run("multi", "multi", profiled=False)
        assert result.returncode == 1
        assert "Set 'bundle' in the profile" in (result.stdout + result.stderr), (
            "guessing which engine to gate could pass while the one under test "
            "is broken"
        )


class TestGateCatchesSpecializationTwins:
    """A bigger set may only OVERRIDE a compiler-settled knob if it keeps the settled
    variant beside it. The knob is read off the UKDs' own declarations, since the fields
    the compiler specializes on are the only ones a descriptor may leave out of its
    spec."""

    def test_override_without_the_twin_fails_naming_the_knob(self, gate):
        small = [_contract(_descriptor("k_sq512", 512))]
        big = [_contract(_pinned_descriptor("k_sq512_pinned", 512, 1))]
        gate.write("twin_small", small)
        gate.write("twin_big", big)
        result = gate.run("twin_small", "twin_small", "twin_big", "twin_big")
        assert result.returncode == 1, result.stdout
        assert "specialization twin" in result.stdout
        assert "use_exp2_fast" in result.stdout
        assert "carry BOTH" in result.stdout

    def test_carrying_both_variants_passes(self, gate):
        small = [_contract(_descriptor("k_sq512", 512))]
        big = [
            _contract(_descriptor("k_sq512", 512)),
            _contract(_pinned_descriptor("k_sq512_pinned", 512, 1)),
        ]
        gate.write("twin_small", small)
        gate.write("twin_big_both", big)
        result = gate.run("twin_small", "twin_small", "twin_big_both", "twin_big_both")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "specialization twin" not in result.stdout

    def test_a_set_with_no_declared_specialization_reports_no_twins(self, gate):
        """Without a declaration nothing states that a field is compiler-settled, so the
        twin check must not invent one from a field that merely looks tri-state."""
        result = gate.run("small", "small", "big", "big")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "specialization twin" not in result.stdout


class TestMetadataMustAgreeWithTheSpecItIsBuiltFrom:
    """A metadata key that is ALSO a spec key must match it. The descriptor is checked
    against ITSELF, so this runs in both modes; a tree labelled aligned whose binary is
    ragged reaches STATIC clean and still builds and packs."""

    def test_catches_a_flag_whose_metadata_contradicts_its_spec(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["kernel_source"]["spec"]["ragged"] = True
        bad[0]["metadata"]["ragged"] = 0
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1, result.stdout
        assert "metadata contradicts the spec" in result.stdout
        assert "ragged" in result.stdout

    def test_catches_a_shape_field_whose_metadata_contradicts_its_spec(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["head_size"] = 64  # spec still says 128
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1, result.stdout
        assert "head_size" in result.stdout

    def test_a_true_spec_flag_with_a_zero_metadata_default_is_caught(self, gate):
        # The direction the ABI guard cares about: the descriptor's own build spec
        # says it was compiled WITH a feature that adds kernarg slots, while its
        # metadata -- what the matcher compares -- claims it was not.
        bad = copy.deepcopy(gate.small)
        bad[0]["kernel_source"]["spec"]["varlen"] = True
        bad[0]["metadata"]["varlen"] = 0
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1, result.stdout
        assert "varlen" in result.stdout

    def test_runs_without_a_profile_at_all(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["head_size"] = 64
        gate.write("bad", bad)
        result = gate.run("bad", "bad", profiled=False)
        assert result.returncode == 1, result.stdout
        assert "metadata contradicts the spec" in result.stdout

    def test_bool_and_int_spellings_of_the_same_value_agree(self, gate):
        # Control. A spec carries Python True where metadata carries 1; that is a
        # spelling difference, not a mislabelling, and reporting it would make the
        # check unusable on every real descriptor set.
        ok = copy.deepcopy(gate.small)
        ok[0]["kernel_source"]["spec"]["ragged"] = False
        ok[0]["metadata"]["ragged"] = 0
        ok[1]["kernel_source"]["spec"]["ragged"] = True
        ok[1]["metadata"]["ragged"] = 1
        gate.write("ok", ok)
        result = gate.run("ok", "ok")
        assert result.returncode == 0, result.stdout

    def test_a_declared_vocabulary_translation_is_not_a_mismatch(self, gate):
        # Control. dtype is spelled "bf16" in the spec and "BF16" in metadata BY
        # DESIGN -- that is what the vocabulary declaration means.
        result = gate.run("small", "small")
        assert result.returncode == 0, result.stdout
        assert "metadata contradicts the spec" not in result.stdout

    def test_an_undeclared_string_field_is_named_not_guessed_at(self, gate):
        # Without a vocabulary declaration there is no way to know whether two spellings
        # of a string are a translation or a defect, so the check declines to guess and
        # says so.
        gate.write("ok", copy.deepcopy(gate.small))
        result = gate.run("ok", "ok", profiled=False)
        assert result.returncode == 0, result.stdout
        assert "UNDECLARED STRING" in result.stdout
        assert "dtype" in result.stdout

    def test_a_string_field_absent_from_a_declared_vocabulary_is_still_compared(
        self, gate
    ):
        # The profile's vocabulary declares dtype and says nothing about persist_decode,
        # a second string field both layers carry, so an unmentioned string field is
        # compared raw.
        bad = copy.deepcopy(gate.small)
        bad[0]["kernel_source"]["spec"]["persist_decode"] = "auto"
        bad[0]["metadata"]["persist_decode"] = "manual"
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1, result.stdout
        assert "metadata contradicts the spec" in result.stdout
        assert "persist_decode" in result.stdout


# --- full mode --------------------------------------------------------------

_PAYLOAD = b"\x7fELF fake code object bytes"
_PAYLOAD_SHA = hashlib.sha256(_PAYLOAD).hexdigest()
_ARCH = "gfx942"
_SYMBOL = "test_kernel_symbol"


class _Payloads:
    """The bytes a packed descriptor names, supplied directly: the gate's own reader
    needs rocm_kpack, and everything full mode is about happens after the bytes are in
    hand."""

    def __init__(self, payload: bytes = _PAYLOAD):
        self.payload = payload

    def read(self, entry, arch):
        return self.payload


def _packed_ukd(metadata=None, payload_sha=_PAYLOAD_SHA) -> dict:
    return {
        "version": "1.0",
        "id": "ukd-packed",
        "name": "k_packed",
        "arch": [_ARCH],
        "kernel_source": {
            "kind": "kpack",
            "library": "kpack/test.kpack",
            "toc_key": "v0",
            "symbol": _SYMBOL,
            "sha256": payload_sha,
        },
        "metadata": metadata
        or {
            "dtype": "BF16",
            "head_size": 128,
            "seqlen_q": 512,
            "use_exp2_fast": 1,
            "ragged": 0,
            "varlen": 0,
            "persist_decode": "auto",
        },
        "provenance": {
            "source": "kernels/test.py",
            "builder": "build_test",
            "spec": {"dtype": "bf16", "head_size": 128, "seqlen_q": 512},
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [copy.deepcopy(_DECLARATION)],
            },
        },
    }


def _identity(name: str) -> dict:
    return {
        "module": "kernels.test",
        "qualname": name,
        "sha256": "0" * 64,
    }


def _publish(ukd: dict, kmd: dict, kdp_doc: dict, ued: dict, observed_value=1) -> None:
    """Write the evidence a producing compile would have written onto `ukd`, through
    `agreement` itself, so the fixture cannot guess the record's shape or key order."""
    declaration = agreement.select_declaration(ukd, ued, kmd, {kmd["id"]: kmd}, kdp_doc)
    request = agreement.observation_request(declaration, kmd)
    header = {k: v for k, v in kdp_doc.items() if k != "kernelDescriptors"}
    header["arch"] = [_ARCH]
    records = agreement.canonical_records(
        [agreement.consumer_record(ukd, ued, kmd, header, _ARCH, declaration)]
    )
    observations = {
        "producer": {"builder": _identity("build_test"), "spec": _identity("TestSpec")},
        "arch": _ARCH,
        "symbol": _SYMBOL,
        "code_object_sha256": ukd["kernel_source"]["sha256"],
        "requests": {
            agreement.digest(request): {
                "values": {"use_exp2_fast": observed_value},
                "accessors": {"use_exp2_fast": _identity("effective_use_exp2_fast")},
            }
        },
    }
    agreement.publish(ukd, observations, records)


@pytest.fixture
def packed(tmp_path):
    """A packed bundle carrying real compiler-written evidence, plus a mutator.

    `build(mutate=...)` applies `mutate(docs)` AFTER the evidence is published, which is
    what a tamper is. The documents are round-tripped through JSON first, as writing
    them to disk does; otherwise the record still holds live references to the engine
    and schema objects and altering the KMD would alter the evidence with it.
    """

    def build(mutate=None, tag="packed"):
        kmd = _kmd()
        ued = _ued()
        ukd = _packed_ukd()
        kdp = _kdp([ukd], arch=[_ARCH])
        _publish(ukd, kmd, kdp, ued)
        frozen = json.loads(json.dumps({"kmd": kmd, "ued": ued, "kdp": kdp}))
        docs = {
            "kmd": frozen["kmd"],
            "ued": frozen["ued"],
            "kdp": frozen["kdp"],
            "ukd": frozen["kdp"]["kernelDescriptors"][0],
        }
        if mutate is not None:
            mutate(docs)
        root = tmp_path / tag
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kmd.json").write_text(json.dumps(docs["kmd"]))
        (root / "test_engine.ued.json").write_text(json.dumps(docs["ued"]))
        (root / "test_engine.kdp.json").write_text(json.dumps(docs["kdp"]))
        return root

    return build


def _run_full(root, payloads=None, arch=_ARCH):
    """`check` in full mode over `root`, returning its failures and narrowings."""
    _binaries, _descriptors, failures, unchecked, _unverified, _knobs = (
        gate_module.check(
            "set",
            str(root),
            gate_module.Profile.empty(),
            "full",
            arch,
            payloads or _Payloads(),
        )
    )
    return failures, unchecked


class TestFullModeChecksTheProducingBuildRecord:
    """The effective values come from the compiler's own evidence: nothing imports a
    producer, redirects an import root, or re-derives a policy. Every case below is a
    way that correspondence can break, and each must be a FAILURE."""

    def test_a_valid_packed_fixture_passes(self, packed):
        failures, unchecked = _run_full(packed())
        assert failures == []
        assert not any("COMPILED SPECIALIZATION" in u for u in unchecked)

    def test_an_absent_record_fails(self, packed):
        def mutate(docs):
            docs["ukd"]["provenance"].pop("effective_spec")

        failures, _ = _run_full(packed(mutate))
        assert any("effective_spec" in f for f in failures), failures

    def test_an_unsupported_schema_version_fails(self, packed):
        def mutate(docs):
            docs["ukd"]["provenance"]["effective_spec"]["schema_version"] = 2

        failures, _ = _run_full(packed(mutate))
        assert any("effective_spec" in f for f in failures), failures

    def test_altered_metadata_fails(self, packed):
        def mutate(docs):
            docs["ukd"]["metadata"]["head_size"] = 64

        failures, _ = _run_full(packed(mutate))
        assert failures

    def test_an_altered_kmd_schema_fails(self, packed):
        def mutate(docs):
            docs["kmd"]["fields"][1]["default_value"] = 256

        failures, _ = _run_full(packed(mutate))
        assert failures

    def test_an_altered_declaration_fails(self, packed):
        def mutate(docs):
            contract = docs["ukd"]["provenance"]["specialization_contract"]
            consumer = contract["consumers"][0]
            # The waiver that would make a full-mode check pass while proving
            # nothing: relabel the field the compiler actually specializes on as
            # the matcher's alone.
            consumer["metadata_fields"] = []
            consumer["matcher_only_fields"] = sorted(
                _DECLARATION["matcher_only_fields"] + ["use_exp2_fast"]
            )
            consumer["bindings"] = {}

        failures, _ = _run_full(packed(mutate))
        assert failures

    def test_an_altered_effective_arch_fails(self, packed):
        """Evidence written for one architecture, checked as another, describes a
        compile that did not produce these bytes."""

        def mutate(docs):
            docs["ukd"]["arch"] = ["gfx950"]
            docs["kdp"]["arch"] = ["gfx950"]

        failures, _ = _run_full(packed(mutate), arch="gfx950")
        assert any("mismatch" in f for f in failures), failures

    def test_altered_payload_bytes_fail(self, packed):
        failures, _ = _run_full(packed(), _Payloads(b"different bytes entirely"))
        assert any("payload" in f for f in failures), failures

    def test_a_forged_authored_record_fails(self, packed, tmp_path):
        """`provenance.effective_spec` is the producing compiler's alone, so an authored
        descriptor -- `kind: rocke`, no bytes yet -- supplying one claims a compile that
        has not happened."""
        kmd, ued = _kmd(), _ued()
        ukd = _packed_ukd()
        kdp = _kdp([ukd], arch=[_ARCH])
        _publish(ukd, kmd, kdp, ued)
        ukd["kernel_source"] = {
            "kind": "rocke",
            "builder": "build_test",
            "spec": {"dtype": "bf16", "head_size": 128, "seqlen_q": 512},
        }
        root = tmp_path / "forged"
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kmd.json").write_text(json.dumps(kmd))
        (root / "test_engine.ued.json").write_text(json.dumps(ued))
        (root / "test_engine.kdp.json").write_text(json.dumps(kdp))
        failures, _ = _run_full(root, gate_module.Payloads())
        assert any("packed dialect" in f for f in failures), failures

    def test_a_ukd_with_no_declaration_fails_rather_than_passing_unchecked(
        self, packed
    ):
        def mutate(docs):
            docs["ukd"]["provenance"].pop("specialization_contract")

        failures, _ = _run_full(packed(mutate))
        assert any("specialization declaration" in f for f in failures), failures

    def test_an_unreadable_payload_fails(self, packed, tmp_path):
        """Reached before any archive library is imported, so it runs anywhere."""
        root = packed(tag="unreadable")
        failures, _ = gate_module.check(
            "set",
            str(root),
            gate_module.Profile.empty(),
            "full",
            _ARCH,
            gate_module.Payloads(),
        )[2:4]
        assert any("does not exist" in f for f in failures), failures

    def test_a_kdp_level_declaration_covers_every_kernel_under_it(self, tmp_path):
        """Shared carriage: every kernel still resolves to a declaration and its
        evidence still binds the bytes, so declaring once passes as repeating per kernel
        does."""
        kmd, ued = _kmd(), _ued()
        ukd = _packed_ukd()
        contract = ukd["provenance"].pop("specialization_contract")
        kdp = _kdp([ukd], arch=[_ARCH])
        kdp["provenance"] = {"specialization_contract": contract}
        _publish(ukd, kmd, kdp, ued)
        root = tmp_path / "shared"
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kmd.json").write_text(json.dumps(kmd))
        (root / "test_engine.ued.json").write_text(json.dumps(ued))
        (root / "test_engine.kdp.json").write_text(json.dumps(kdp))
        failures, _ = _run_full(root)
        assert not failures, failures


class TestFullModeReportsAKernelWithNothingToBind:
    """`metadata_fields: []` is the MANDATORY declaration for a non-compiled source: an
    AOT hip bundle has no builder object to bind and no producing-build record to read.
    Failing it would make full mode unpassable for every hip bundle; passing it silently
    would claim a binding never made, so it is reported."""

    @staticmethod
    def tree(tmp_path, metadata_fields, tag):
        """A packed bundle with no producer evidence, as an AOT hip pack ships."""
        kmd, ued = _kmd(), _ued()
        ukd = _packed_ukd()
        consumer = ukd["provenance"]["specialization_contract"]["consumers"][0]
        consumer["metadata_fields"] = list(metadata_fields)
        consumer["matcher_only_fields"] = sorted(
            set(f["name"] for f in _KMD_FIELDS) - set(metadata_fields)
        )
        consumer["bindings"] = {f: _DECLARATION["bindings"][f] for f in metadata_fields}
        kdp = _kdp([ukd], arch=[_ARCH])
        root = tmp_path / tag
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kmd.json").write_text(json.dumps(kmd))
        (root / "test_engine.ued.json").write_text(json.dumps(ued))
        (root / "test_engine.kdp.json").write_text(json.dumps(kdp))
        return root

    def test_it_is_reported_rather_than_failed(self, tmp_path):
        root = self.tree(tmp_path, [], "unbound")
        _b, _d, failures, _unchecked, unverified, _k = gate_module.check(
            "set", str(root), gate_module.Profile.empty(), "full", _ARCH, _Payloads()
        )
        assert failures == []
        assert len(unverified) == 1

    def test_it_does_not_fail_the_gate(self, tmp_path, monkeypatch, capsys):
        """On the exit code, which is what an integrator reads."""
        monkeypatch.setattr(gate_module, "Payloads", lambda *_a, **_k: _Payloads())
        root = self.tree(tmp_path, [], "unbound_exit")
        profile = tmp_path / "profile.yaml"
        profile.write_text(_PROFILE)
        code = gate_module.main(
            ["set", str(root), "--mode", "full", "--profile", str(profile)]
        )
        out = capsys.readouterr().out
        assert code == 0, out
        assert "GATE FAILED" not in out

    def test_a_kernel_that_claims_a_field_and_has_no_evidence_still_fails(
        self, tmp_path
    ):
        """Nothing above may become a way to skip a real check."""
        root = self.tree(tmp_path, ["use_exp2_fast"], "claimed")
        failures, _ = _run_full(root)
        assert failures

    @staticmethod
    def verdict(root):
        """`(failures, unverified)` from one full-mode run. Both lists are read, since a
        kernel landing in neither, or in both, is a different verdict."""
        _b, _d, failures, _unchecked, unverified, _k = gate_module.check(
            "set", str(root), gate_module.Profile.empty(), "full", _ARCH, _Payloads()
        )
        return failures, unverified

    def with_origin(self, tmp_path, tag, origin_kind):
        """`tree` writes no `provenance.origin_kind` and publishes no `effective_spec`,
        so the stamp is the only thing separating the origin cases."""
        root = self.tree(tmp_path, [], tag)
        path = root / "test_engine.kdp.json"
        doc = json.loads(path.read_text())
        provenance = doc["kernelDescriptors"][0]["provenance"]
        assert "effective_spec" not in provenance
        provenance["origin_kind"] = origin_kind
        path.write_text(json.dumps(doc))
        return root

    def test_a_rocke_origin_cannot_waive_its_own_evidence(
        self, tmp_path, monkeypatch, capsys
    ):
        """A waiver keyed on the claim alone is a self-service exemption: the packer
        publishes `effective_spec` onto every rocKE UKD it ships, so deleting the record
        and relabelling the specialized fields as matcher-only would carry a shipped
        rocKE shard to a clean exit with the archive bytes never read."""
        root = self.with_origin(tmp_path, "rocke_origin", "rocke")
        failures, unverified = self.verdict(root)
        assert unverified == []
        assert len(failures) == 1, failures
        # The report has to name which kernel; an integrator reading `GATE FAILED`
        # off a multi-kernel bundle has nothing else to go on.
        assert failures[0].startswith("k_packed: "), failures[0]
        assert (
            "A rocKE-produced kernel is required to carry its compiler evidence"
            in failures[0]
        )

        monkeypatch.setattr(gate_module, "Payloads", lambda *_a, **_k: _Payloads())
        profile = tmp_path / "rocke-profile.yaml"
        profile.write_text(_PROFILE)
        code = gate_module.main(
            ["set", str(root), "--mode", "full", "--profile", str(profile)]
        )
        out = capsys.readouterr().out
        assert code == 1, out
        assert "GATE FAILED" in out
        assert "k_packed" in out

    def test_a_non_rocke_origin_in_the_same_state_still_waives(self, tmp_path):
        """An AOT hip pack has no builder object to bind, so the waiver is legitimate;
        without this pair the rocKE case passes against a check that fails every
        recordless kernel."""
        root = self.with_origin(tmp_path, "hip_origin", "hip")
        failures, unverified = self.verdict(root)
        assert failures == []
        assert len(unverified) == 1

    def test_an_absent_origin_kind_is_not_read_as_rocke(self, tmp_path):
        """Hand-authored trees and descriptors packed without `origin_kind` carry no
        origin at all, so reading rocKE out of the absence would fail them over evidence
        they were never asked to produce."""
        root = self.tree(tmp_path, [], "absent_origin")
        doc = json.loads((root / "test_engine.kdp.json").read_text())
        assert "origin_kind" not in doc["kernelDescriptors"][0]["provenance"]
        failures, unverified = self.verdict(root)
        assert failures == []
        assert len(unverified) == 1


class TestFullModeCannotPassOnANarrowedRun:
    """Asserted on this tool's EXIT CODE, at `main`: a caller reading the claim off the
    exit status must not need a second tool to scrape the caveat out of stdout."""

    def test_a_narrowed_full_run_exits_nonzero(self, packed, monkeypatch, capsys):
        monkeypatch.setattr(gate_module, "Payloads", lambda *_a, **_k: _Payloads())
        root = packed(tag="narrowed_full")
        code = gate_module.main(["set", str(root), "--mode", "full"])
        out = capsys.readouterr().out
        # The narrowing is real: this tree declares no vocabulary anywhere, so the
        # vocabulary check has nothing to judge the string fields against.
        assert "NOT RUN" in out
        assert code == 1, out
        assert "GATE PASSED" not in out


class TestStructuralModeNeverClaimsCompiledAgreement:
    """Structural mode cannot see any full-mode failure -- that is what the mode is --
    so the property is that it says so."""

    def test_it_passes_its_own_properties_and_names_what_it_did_not_check(
        self, packed, capsys
    ):
        def mutate(docs):
            docs["ukd"]["metadata"]["head_size"] = 64

        root = packed(mutate, tag="tampered_structural")
        _b, _d, failures, unchecked, _u, _k = gate_module.check(
            "set", str(root), gate_module.Profile.empty(), "structural"
        )
        assert failures == []
        assert any("COMPILED SPECIALIZATION AGREEMENT" in u for u in unchecked)
        assert "COMPILED SPECIALIZATION AGREEMENT" in capsys.readouterr().out


# TestRealArchiveSelectedConsumer is the one class that builds a real kpack archive,
# carried as a skip rather than a hard failure so a checkout without rocm_kpack keeps
# the no-producer property.
_NO_ROCM_KPACK = (
    "rocm_kpack is not installed and HIPKERNELPROVIDER_ROCM_KPACK_DIR is unset; set it "
    "to the rocm-kpack 'python' directory to run this class"
)


def _kpack_python_dir() -> str | None:
    """The rocm-kpack `python` directory an operator named, or None to import
    `rocm_kpack` from the environment. Skips when neither is available."""
    # An exported-but-empty variable names no directory and counts as unset. `""` is not
    # `None`, so left alone it resolves to the working directory and the class dies on a
    # raw import error instead of naming the dependency it wants.
    python_dir = os.environ.get("HIPKERNELPROVIDER_ROCM_KPACK_DIR") or None

    # Only a genuinely absent dependency skips: find_spec answers that without executing
    # the package, so a broken rocm_kpack still reaches load_kpack and fails. An
    # operator who set the directory gets their value passed to load_kpack unexamined.
    if python_dir is None and importlib.util.find_spec("rocm_kpack") is None:
        pytest.skip(_NO_ROCM_KPACK)

    return python_dir


@pytest.fixture
def real_archive():
    """Real rocm-kpack serialization; the payload is deliberately non-executable."""
    python_dir = _kpack_python_dir()

    kpack, compression = load_kpack(python_dir)

    def write(root):
        path = root / "kpack" / "test.kpack"
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = kpack.PackedKernelArchive(
            group_name="test",
            gfx_arch_family=_ARCH,
            gfx_arches=[_ARCH],
            compressor=compression.ZstdCompressor(compression_level=3),
        )
        archive.add_kernel(
            archive.prepare_kernel(
                relative_path="v0", gfx_arch=_ARCH, hsaco_data=_PAYLOAD, metadata={}
            )
        )
        archive.finalize_archive()
        archive.write(path)

    return write, python_dir


class TestTheArchiveDependencyIsResolvedOrSkipped:
    """An exported-but-empty directory is a variable, not a path:
    `HIPKERNELPROVIDER_ROCM_KPACK_DIR=` survives a shell export and a CMake `-D` that
    resolved to nothing, while a stale path an operator named must still fail."""

    @staticmethod
    def _outcome():
        try:
            return ("directory", _kpack_python_dir())
        except pytest.skip.Exception as skipped:
            return ("skip", str(skipped))

    def test_an_empty_value_is_unset_while_a_named_one_still_runs(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("HIPKERNELPROVIDER_ROCM_KPACK_DIR", raising=False)
        unset = self._outcome()

        monkeypatch.setenv("HIPKERNELPROVIDER_ROCM_KPACK_DIR", "")
        assert self._outcome() == unset, (
            "an empty value names no directory, so it has to reach the same outcome "
            "as an absent one on this machine"
        )

        stale = tmp_path / "no-such-kpack-checkout"
        monkeypatch.setenv("HIPKERNELPROVIDER_ROCM_KPACK_DIR", str(stale))
        assert self._outcome() == ("directory", str(stale)), (
            "a named directory is a request to run this class; a nonexistent one is "
            "a loud failure in load_kpack, never a skip"
        )

    def test_a_stale_named_directory_raises_out_of_load_kpack(
        self, monkeypatch, tmp_path
    ):
        """Where "fails loudly" happens; the case above only proves the stale path is
        HANDED ON. NOT marked `needs_rocm_kpack`: the directory check precedes any
        import, so proving a named path fails must not need the dependency it is proving
        absent."""
        stale = tmp_path / "no-such-kpack-checkout"
        monkeypatch.setenv("HIPKERNELPROVIDER_ROCM_KPACK_DIR", str(stale))

        with pytest.raises(HkpPackError) as excinfo:
            load_kpack(_kpack_python_dir())

        # The path is the discriminator: `load_kpack`'s other HkpPackError -- the one
        # a machine without rocm_kpack installed raises from the import -- names no
        # directory, so a message carrying this one can only be the stale-path branch.
        assert str(stale.resolve()) in str(excinfo.value)


@pytest.mark.needs_rocm_kpack
class TestRealArchiveSelectedConsumer:
    """Selected authority and packed-input gates cannot borrow sibling evidence.

    The skip keeps a default run green without rocm_kpack; the marker is the handle
    `-m needs_rocm_kpack` selects on, which a fixture-internal skip cannot offer.
    """

    @staticmethod
    def run(root, python_dir, tmp_path, mode="full", arch=None):
        profile = tmp_path / "real-profile.json"
        profile.write_text(
            json.dumps(
                {"bundle": "test_engine", "vocabulary": {"dtype": ["BF16", "FP16"]}}
            )
        )
        args = [
            sys.executable,
            str(_TOOL),
            "set",
            str(root),
            "--mode",
            mode,
            "--profile",
            str(profile),
        ]
        if arch is not None:
            args += ["--arch", arch]
        if python_dir:
            args += ["--kpack-python-dir", python_dir]
        return subprocess.run(args, capture_output=True, text=True)

    def test_requested_arch_without_consumer_records_is_classified_failure(
        self, tmp_path, packed, real_archive
    ):
        write_archive, python_dir = real_archive
        root = packed()
        write_archive(root)
        control = self.run(root, python_dir, tmp_path, arch=_ARCH)
        assert control.returncode == 0, control.stdout + control.stderr
        assert "GATE PASSED" in control.stdout
        assert "NOT VERIFIED HERE" not in control.stdout

        result = self.run(root, python_dir, tmp_path, arch="gfx950")
        output = result.stdout + result.stderr
        assert result.returncode == 1, output
        assert "GATE FAILED" in result.stdout
        assert "k_packed" in result.stdout
        assert "gfx950" in result.stdout
        assert "Traceback" not in output
        assert "NOT VERIFIED HERE" not in output
        assert "GATE PASSED" not in output

    def test_selected_inline_consumer_cannot_borrow_sibling_declaration(
        self, tmp_path, real_archive
    ):
        write_archive, python_dir = real_archive
        root = tmp_path / "shared"
        root.mkdir()
        schema, engine = _kmd(), _ued()
        sibling_engine = _ued(ident="ued-sibling")
        ukd = _packed_ukd()
        contract = ukd["provenance"].pop("specialization_contract")
        selected = _kdp([ukd], arch=[_ARCH])
        selected["provenance"] = {"specialization_contract": contract}
        sibling = _kdp([ukd], ident="kdp-sibling", engine="ued-sibling", arch=[_ARCH])
        sibling_contract = copy.deepcopy(contract)
        sibling_contract["consumers"][0]["engine_id"] = "ued-sibling"
        sibling["provenance"] = {"specialization_contract": sibling_contract}
        _publish(ukd, schema, selected, engine)
        observations = ukd["provenance"]["effective_spec"]["observations"]
        records = []
        for doc, ued in ((selected, engine), (sibling, sibling_engine)):
            declaration = agreement.select_declaration(
                ukd, ued, schema, {schema["id"]: schema}, doc
            )
            header = {k: v for k, v in doc.items() if k != "kernelDescriptors"}
            records.append(
                agreement.consumer_record(ukd, ued, schema, header, _ARCH, declaration)
            )
        agreement.publish(ukd, observations, agreement.canonical_records(records))
        (root / "test_engine.kmd.json").write_text(json.dumps(schema))
        (root / "test_engine.ued.json").write_text(json.dumps(engine))
        (root / "sibling.ued.json").write_text(json.dumps(sibling_engine))
        selected_path = root / "test_engine.kdp.json"
        sibling_path = root / "sibling.kdp.json"
        selected_path.write_text(json.dumps(selected))
        sibling_path.write_text(json.dumps(sibling))
        write_archive(root)
        control = self.run(root, python_dir, tmp_path)
        assert control.returncode == 0, control.stdout + control.stderr
        assert "NOT VERIFIED HERE" not in control.stdout

        # A valid sibling record copied with its inline UKD must not authorize a
        # second KDP that declares nothing. The bytes and UKD binding still agree.
        selected.pop("provenance")
        agreement.publish(ukd, observations, agreement.canonical_records(records[1:]))
        selected_path.write_text(json.dumps(selected))
        sibling_path.write_text(json.dumps(sibling))
        result = self.run(root, python_dir, tmp_path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "no specialization declaration" in result.stdout

    def test_unclaimed_hip_must_still_be_packed(self, tmp_path, real_archive):
        write_archive, python_dir = real_archive
        root = TestFullModeReportsAKernelWithNothingToBind.tree(tmp_path, [], "hip")
        path = root / "test_engine.kdp.json"
        doc = json.loads(path.read_text())
        ukd = doc["kernelDescriptors"][0]
        ukd["provenance"]["origin_kind"] = "hip"
        path.write_text(json.dumps(doc))
        write_archive(root)
        control = self.run(root, python_dir, tmp_path)
        assert control.returncode == 0, control.stdout + control.stderr
        assert "NOT VERIFIED HERE" in control.stdout

        ukd["kernel_source"]["kind"] = "hip"
        path.write_text(json.dumps(doc))
        result = self.run(root, python_dir, tmp_path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "packed dialect" in result.stdout
        structural = self.run(root, python_dir, tmp_path, mode="structural")
        assert structural.returncode == 0, structural.stdout + structural.stderr
