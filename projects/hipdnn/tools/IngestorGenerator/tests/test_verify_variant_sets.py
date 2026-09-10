# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The variant-set gate must fail on each defect it claims to catch.

A gate is only worth its exit code if every branch can reach 1. The adversarial
review of the gfx942 attention_dense work made exactly this point twice: it built
five deliberately-corrupted trees to prove the original gate was failable, and it
separately flagged a "negative control" fixture that did not actually exercise the
defect its name advertised. This file makes that battery permanent, and adds the
checks the generalised gate needs that the hardcoded one did not:

  * that a clean pair PASSES (a gate that refuses everything is not a gate);
  * that the schema is reached BY REFERENCE -- a dangling `engine`, a dangling
    `metadata`, two documents claiming one id, and a same-stem pair nothing wires
    together all FAIL, because binding a bundle to a filename gates whatever schema
    happens to sit beside it;
  * that `--mode structural` degrades LOUDLY -- compiled specialization agreement is
    named as NOT CHECKED rather than quietly skipped, because "the gate passed" must
    never mean "the gate stopped looking";
  * that `--mode full` fails on every way the producing compiler's evidence can stop
    describing the artifact in hand, and does so with no rocKE and no producer import
    anywhere;
  * that an ambiguous tree is refused rather than guessed at, since silently gating
    the wrong engine would pass while the one under test is broken.

NO PRODUCER IS IMPORTED, here or by the tool. The full-mode fixtures build the
compiler's evidence with `hkp_pack.agreement` itself -- the same pure-stdlib module
the packer publishes through -- so the battery runs on a machine that has never had
rocKE installed. That is not a convenience: it is the property under test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_variant_sets.py"

sys.path.insert(0, str(_TOOL.parent))

import verify_variant_sets as gate_module  # noqa: E402

sys.path.insert(0, str(gate_module._agreement_python_root()))

from hkp_pack import agreement  # noqa: E402

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
    # Fields the metadata-matches-spec cases below perturb. Declared here because a
    # metadata key the KMD does not know is a different defect (an undeclared field
    # drops the whole pack at load) and would fail those cases for the wrong reason.
    {"name": "ragged", "type": "int", "default_value": 0},
    {"name": "varlen", "type": "int", "default_value": 0},
    {"name": "persist_decode", "type": "string", "default_value": "auto"},
]

#: The declaration a UKD carries when the twin and full-mode cases need the gate to
#: know which field the compiler settles. Exactly the six keys
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
    """A descriptor whose metadata agrees with the spec it is built from.

    `use_exp2_fast` absent from the spec is the authoring form that means "the
    kernel settles this at build time"; the metadata still states which binary
    resulted, because the matcher compares metadata and a field absent there
    resolves to the KMD default, which is a different kernel.
    """
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
    """A descriptor that PINS `use_exp2_fast` in its spec rather than leaving it to
    the kernel -- the shape of an override, as opposed to `_descriptor()`."""
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
    """The control. Every failure assertion below is worthless without this."""

    def test_a_clean_nesting_pair_passes(self, gate):
        result = gate.run("small", "small", "big", "big")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "GATE PASSED" in result.stdout


class TestTheSchemaIsReachedByReference:
    """F09. A bundle's KMD is found by walking the ids the documents declare.

    Reaching it by filename surgery -- swapping `.kdp.json` for `.kmd.json` on the
    same stem -- answers "which schema governs this bundle" with a coincidence of
    naming. A tree where the two are not wired together then gates the descriptors
    against a schema nothing connects them to, and every defaulted field, every
    completed tuple and every type below is decided by the wrong document.
    """

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
        """The defect this replaces. The KDP and the KMD share a filename stem and
        sit in one directory, which is exactly what the old resolution accepted --
        and nothing in either document references the other."""
        root = gate.write("stem_only", gate.small)
        doc = _kdp(gate.small)
        doc.pop("engine")
        (root / "test_engine.kdp.json").write_text(json.dumps(doc))
        result = gate.run("bad", "stem_only")
        assert result.returncode == 1
        assert "engine" in (result.stdout + result.stderr)


class TestGateCatchesEachDefect:
    """One case per property, each defect introduced in isolation."""

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
        """Absent key and explicit default are ONE catalog entry, not two.

        This is the collision the JSON does not show: the two descriptors differ on
        disk and collide only after the loader applies default_value.
        """
        pinned = _descriptor("k_pinned", 512)
        unset = _descriptor("k_unset", 512)
        unset["metadata"].pop("seqlen_q")
        gate.write("bad", [pinned, unset])
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "loader-tuple collisions" in result.stdout


class TestTheDeskCheckIdentityIsEngineWideAndArchAware:
    """F20. The loader assembles ONE catalog per engine per device.

    Two KDPs of one engine that each look unique alone still collide there, so the
    identity has to be engine-wide. It also has to carry the effective architecture:
    a gfx942 pack and a gfx950 pack never meet on one device, so an identical tuple
    in both is legal, while an overlap -- including a wildcard sitting over a
    concrete arch -- is the collision that drops the engine.
    """

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
        """An absent arch list is "every device", so it meets the concrete pack on
        the concrete pack's own device. Treating absence as its own bucket is how a
        wildcard duplicate ships."""
        self._two_packs(gate, "wildcard", None, ["gfx950"])
        result = gate.run("w", "wildcard", profiled=False)
        assert result.returncode == 1, result.stdout
        assert "loader-tuple collisions" in result.stdout

    def test_one_and_one_point_zero_are_one_tuple_on_a_float_field(self, gate):
        """The catalog holds a value of the field's declared type, so 1 and 1.0 on a
        FLOAT field are the same entry. Comparing the JSON spellings raw reports two
        distinct tuples and lets the duplicate that drops the engine ship."""
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
    """A bigger set may only OVERRIDE a compiler-settled knob if it also keeps the
    settled variant beside it. Overriding alone silently drops the smaller set's
    kernel from the candidate list.

    The knob is read off the UKDs' own declarations -- the fields the compiler
    specializes on are the only ones a descriptor may legitimately leave out of its
    spec, so they are the only ones that can have a twin.
    """

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
        """The control for the knob source. Without a declaration nothing states
        that any field is compiler-settled, so nothing can be a twin -- and the
        twin check must not invent one from a field that merely looks tri-state."""
        result = gate.run("small", "small", "big", "big")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "specialization twin" not in result.stdout


class TestMetadataMustAgreeWithTheSpecItIsBuiltFrom:
    """Property (4a): a metadata key that is ALSO a spec key must match it.

    This needs no evidence, no profile and no kernel knowledge -- it is the
    descriptor checked against ITSELF, so it runs in both modes. It did not exist
    until a review demonstrated the hole by mutation: property (4) iterated a
    profile's policy block and nothing else, so a kernel with no policy-owned knob
    had property (4) checking NOTHING while the gate printed a clean pass.

    The case that matters most is the one the shipping commit names as "the dangerous
    direction": a descriptor labelled aligned whose binary is actually ragged. The C++
    matcher tests catch that at the matcher rung; this is the STATIC rung, which
    coverage_gate.py's docstring insists is separate precisely because each catches
    what the other cannot. A mislabelled tree that reaches STATIC clean still builds
    and still packs.
    """

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
        # Without a vocabulary declaration there is no way to know whether two
        # spellings of a string are a translation or a defect, so the check declines
        # to guess -- and says so, because a field nobody can judge is a liability
        # the author should see.
        gate.write("ok", copy.deepcopy(gate.small))
        result = gate.run("ok", "ok", profiled=False)
        assert result.returncode == 0, result.stdout
        assert "UNDECLARED STRING" in result.stdout
        assert "dtype" in result.stdout

    def test_a_string_field_absent_from_a_declared_vocabulary_is_still_compared(
        self, gate
    ):
        # Regression for the escape a review found on the real gfx950 tree: a
        # profile's vocabulary declares dtype and says nothing about persist_decode,
        # a second string field both layers carry. Once a vocabulary section exists
        # the author had the exact place to declare it translated and did not, so an
        # unmentioned string field is compared raw.
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
    """The bytes a packed descriptor names, supplied directly.

    The gate's own reader pulls them out of the arch's `.kpack` archive, which needs
    rocm_kpack -- a build artifact. Everything the full-mode battery is about happens
    AFTER the bytes are in hand: whether the compiler's evidence still describes this
    descriptor, this schema, this architecture and these bytes. Substituting the
    reader keeps the battery runnable on any machine while leaving the property under
    test untouched.
    """

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
        "file": f"/src/kernels/{name}.py",
        "sha256": "0" * 64,
    }


def _publish(ukd: dict, kmd: dict, kdp_doc: dict, ued: dict, observed_value=1) -> None:
    """Write the evidence a producing compile would have written onto `ukd`.

    Built through `agreement` itself rather than by hand: the record's shape and
    order are that module's business, and a fixture that reconstructed them would
    pass or fail on its own guess about key order rather than on the property.
    """
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

    `build(mutate=...)` writes the tree, applying `mutate(docs)` AFTER the evidence
    is published -- which is what a tamper is: the record was true of the artifact
    that left the compiler, and something changed underneath it.

    The documents are round-tripped through JSON before the mutation, exactly as
    writing them to disk does. Without that the record the compiler stored still
    holds live references to the engine and schema objects, so altering the KMD
    would alter the stored evidence at the same time and the tamper would be
    invisible for the wrong reason.
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
    """F10/F21. The effective values come from the compiler's own evidence.

    Nothing here imports a producer, redirects an import root, or re-derives a
    policy: the descriptor carries the declaration and the record, and the gate
    checks them against the descriptors, the schema, the architecture and the bytes
    in hand. Every case below is a way that correspondence can break, and each one
    must be a FAILURE rather than a property left unchecked -- an artifact that
    cannot say what it was built from has not established agreement.
    """

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
        """The evidence was written for one architecture. Checked as another, it
        describes a compile that did not produce these bytes -- even though every
        byte of the descriptor is otherwise untouched."""

        def mutate(docs):
            docs["ukd"]["arch"] = ["gfx950"]
            docs["kdp"]["arch"] = ["gfx950"]

        failures, _ = _run_full(packed(mutate), arch="gfx950")
        assert any("mismatch" in f for f in failures), failures

    def test_altered_payload_bytes_fail(self, packed):
        failures, _ = _run_full(packed(), _Payloads(b"different bytes entirely"))
        assert any("payload" in f for f in failures), failures

    def test_a_forged_authored_record_fails(self, packed, tmp_path):
        """Evidence authored into the input rather than written by the compiler.

        `provenance.effective_spec` is the producing compiler's alone. An authored
        descriptor -- `kind: rocke`, no bytes yet -- that supplies one is claiming a
        compile that has not happened, and no amount of internal consistency makes
        it evidence about an artifact that does not exist.
        """
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
        """The gate's own reader, on a descriptor naming bytes that are not there.
        Reached before any archive library is imported, so it runs anywhere."""
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
        """Shared carriage: the declaration is written once and inherited.

        The property full mode checks is unchanged -- every kernel resolves to a
        declaration and its evidence binds the bytes -- so a bundle that declares
        once passes exactly as one that repeats itself per kernel does.
        """
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
    """A packed kernel that declares no specialized field, in full mode.

    `metadata_fields: []` is the MANDATORY declaration for a non-compiled source --
    an AOT hip bundle has no builder object, so there is nothing to bind and no
    producing-build record to read. Failing it would make full mode unpassable for
    every hip bundle; passing it silently would claim a binding that was never made.
    It is reported instead, exactly as `hkp_pack.desk_check` reports the same
    artifact -- two readers of one tree must not disagree about it.
    """

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
        """The other half. Nothing above may become a way to skip a real check."""
        root = self.tree(tmp_path, ["use_exp2_fast"], "claimed")
        failures, _ = _run_full(root)
        assert failures


class TestFullModeCannotPassOnANarrowedRun:
    """A full run claims every property, so a check it could not RUN is a gap.

    Asserted on this tool's own EXIT CODE, at `main`, rather than on the output: a
    caller that reads the claim off the exit status must not need a second tool to
    scrape the caveat back out of stdout, and a narrowed run that exits 0 is a green
    gate for a property nobody checked.
    """

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
    """The same tampered trees, checked structurally.

    Structural mode cannot see any of the full-mode failures above -- that is what
    the mode is -- so the property under test is that it says so. Reporting a clean
    structural pass on a tree whose evidence no longer matches its bytes is only
    dangerous if the output reads as though it checked.
    """

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
