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
  * that removing the profile degrades LOUDLY -- the policy-dependent checks are
    named as NOT CHECKED rather than quietly skipped, because "the gate passed"
    must never mean "the gate stopped looking";
  * that an ambiguous tree is refused rather than guessed at, since silently
    gating the wrong engine would pass while the one under test is broken.

The fixtures use a stub policy module rather than importing rocKE. The gate's
contract is "ask the kernel's own resolver"; which resolver is a profile fact, and
binding the tests to a real kernel would make them a rocKE integration test.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_variant_sets.py"

_STUB_POLICY = """
def resolve(head_size, dtype, seqlen_q):
    # Shaped like the real rule: on above a sequence-length threshold, off below.
    return int(seqlen_q) >= 4096
"""

_PROFILE = """
bundle: test_engine
provider_root: {root}
vocabulary:
  dtype: [BF16, FP16]
policies:
  use_exp2_fast:
    module: stubpol
    function: resolve
    args: [head_size, dtype, seqlen_q]
"""

_KMD_FIELDS = [
    {"name": "dtype", "type": "string"},
    {"name": "head_size", "type": "int", "default_value": 128},
    {"name": "seqlen_q", "type": "int", "default_value": 512},
    {"name": "use_exp2_fast", "type": "int", "default_value": -1},
]


def _descriptor(name: str, seqlen_q: int) -> dict:
    """A descriptor whose metadata agrees with what the stub policy resolves."""
    return {
        "version": "1.0",
        "id": f"id-{name}",
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "builder": "build_test",
            "spec": {"dtype": "bf16", "head_size": 128, "seqlen_q": seqlen_q},
        },
        "metadata": {
            "dtype": "BF16",
            "head_size": 128,
            "seqlen_q": seqlen_q,
            "use_exp2_fast": 1 if seqlen_q >= 4096 else 0,
        },
        "priority": 0,
    }


def _pinned_descriptor(name: str, seqlen_q: int, use_exp2_fast: int) -> dict:
    """A descriptor that PINS `use_exp2_fast` explicitly rather than leaving it
    to the stub policy -- the shape of an override, as opposed to `_descriptor()`
    which always leaves the knob for the policy to decide.
    """
    return {
        "version": "1.0",
        "id": f"id-{name}",
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "builder": "build_test",
            "spec": {
                "dtype": "bf16",
                "head_size": 128,
                "seqlen_q": seqlen_q,
                "use_exp2_fast": use_exp2_fast,
            },
        },
        "metadata": {
            "dtype": "BF16",
            "head_size": 128,
            "seqlen_q": seqlen_q,
            "use_exp2_fast": use_exp2_fast,
        },
        "priority": 0,
    }


@pytest.fixture
def gate(tmp_path):
    """A working gate environment: stub policy, profile, and a nesting pair."""
    lib = tmp_path / "rocke" / "library"
    lib.mkdir(parents=True)
    (lib / "stubpol.py").write_text(_STUB_POLICY)

    profile = tmp_path / "profile.yaml"
    profile.write_text(_PROFILE.format(root=tmp_path))

    def write(tag: str, descriptors: list[dict], fields=None) -> Path:
        root = tmp_path / tag
        root.mkdir(parents=True, exist_ok=True)
        (root / "test_engine.kdp.json").write_text(
            json.dumps({"version": "1.0", "kernelDescriptors": descriptors})
        )
        (root / "test_engine.kmd.json").write_text(
            json.dumps({"version": "1.0", "fields": fields or _KMD_FIELDS})
        )
        return root

    def run(*args, profiled: bool = True):
        argv = [sys.executable, str(_TOOL), *args]
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
        },
    )


class TestGatePasses:
    """The control. Every failure assertion below is worthless without this."""

    def test_a_clean_nesting_pair_passes(self, gate):
        result = gate.run("small", "small", "big", "big")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "GATE PASSED" in result.stdout
        assert (
            "NOT CHECKED" not in result.stdout
        ), "with a full profile every check must actually run"


class TestGateCatchesEachDefect:
    """One case per property, each defect introduced in isolation."""

    def test_catches_a_shipped_sentinel(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = -1
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "unset sentinel" in result.stdout

    def test_catches_metadata_that_mislabels_its_binary(self, gate):
        # Policy resolves seqlen 512 to OFF; the metadata claims ON. The matcher
        # would select this kernel believing it is something it is not.
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = 1
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "mislabel their binary on 'use_exp2_fast'" in result.stdout

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
        unset["kernel_source"]["spec"]["seqlen_q"] = 512
        gate.write("bad", [pinned, unset])
        result = gate.run("bad", "bad")
        assert result.returncode == 1
        assert "loader-tuple collisions" in result.stdout


class TestGateDegradesLoudly:
    """Without a profile the gate must narrow, and must say which checks it dropped."""

    def test_structural_checks_still_run_without_a_profile(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = -1
        gate.write("bad", bad)
        result = gate.run("bad", "bad", profiled=False)
        assert result.returncode == 1, "a sentinel needs no kernel knowledge to spot"
        assert "unset sentinel" in result.stdout

    def test_policy_checks_are_named_not_silently_skipped(self, gate):
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = 1  # mislabels its binary
        gate.write("bad", bad)
        result = gate.run("bad", "bad", profiled=False)
        # It genuinely cannot catch this without the policy...
        assert result.returncode == 0
        # ...but it must not report an unqualified pass.
        assert "NOT CHECKED" in result.stdout
        assert "metadata-matches-binary" in result.stdout
        assert "GATE PASSED on what it checked" in result.stdout


class TestGateRefusesAmbiguity:
    def test_two_bundles_without_a_pin_is_an_error(self, gate):
        root = gate.write("multi", gate.small)
        for name in ("second_engine.kdp.json", "second_engine.kmd.json"):
            source = "test_engine" + name[len("second_engine") :]
            (root / name).write_text((root / source).read_text())
        result = gate.run("multi", "multi", profiled=False)
        assert result.returncode == 1
        assert "Set 'bundle' in the profile" in (result.stdout + result.stderr), (
            "guessing which engine to gate could pass while the one under test "
            "is broken"
        )

    def test_a_policy_that_will_not_import_fails_loudly(self, gate, tmp_path):
        (tmp_path / "rocke" / "library" / "stubpol.py").unlink()
        result = gate.run("small", "small")
        assert result.returncode == 1
        assert "will not import" in (result.stdout + result.stderr)


class TestGateCatchesPolicyTwins:
    """A bigger set may only OVERRIDE a policy-decided knob if it also keeps the
    policy-decided variant beside it. Overriding alone silently drops the
    smaller set's kernel from the candidate list -- see module docstring
    property 1 and the "policy twins" note in verify_variant_sets.py.
    """

    def test_override_without_the_policy_twin_fails_naming_the_knob(self, gate):
        # small leaves use_exp2_fast to the policy (resolves to 0 at seqlen 512).
        # big carries ONLY a pinned override at a DIFFERENT value: the
        # policy-decided kernel has nowhere to be chosen from any more.
        small = [_descriptor("k_sq512", 512)]
        big = [_pinned_descriptor("k_sq512_pinned", 512, 1)]
        gate.write("twin_small", small)
        gate.write("twin_big", big)
        result = gate.run("twin_small", "twin_small", "twin_big", "twin_big")
        assert result.returncode == 1
        assert "policy twin" in result.stdout
        assert "use_exp2_fast" in result.stdout
        assert "carry BOTH" in result.stdout

    def test_carrying_both_variants_passes(self, gate):
        # big keeps the policy-decided descriptor AND adds the pinned override:
        # exactly the fix the diagnostic above asks for.
        small = [_descriptor("k_sq512", 512)]
        big = [
            _descriptor("k_sq512", 512),
            _pinned_descriptor("k_sq512_pinned", 512, 1),
        ]
        gate.write("twin_small", small)
        gate.write("twin_big_both", big)
        result = gate.run("twin_small", "twin_small", "twin_big_both", "twin_big_both")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "policy twin" not in result.stdout

    def test_pinning_the_policy_resolved_value_is_not_a_false_positive(self, gate):
        # The override pins EXACTLY what the policy would have resolved (0, since
        # seqlen_q=512 is below the stub's threshold). Same binary, different
        # spelling -- _binary_key() already normalises this, so it must not be
        # reported as a missing twin.
        small = [_descriptor("k_sq512", 512)]
        big = [_pinned_descriptor("k_sq512_pinned0", 512, 0)]
        gate.write("twin_small", small)
        gate.write("twin_big_same", big)
        result = gate.run("twin_small", "twin_small", "twin_big_same", "twin_big_same")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "policy twin" not in result.stdout


class TestAnEmptyPolicyBlockIsAnAnswerNotSilence:
    """`policies:` absent and `policies: {}` are DIFFERENT claims.

    Absent means nobody asked, and the gate must narrow and say so. `{}` asserts the
    kernel has no policy-owned knob -- a real, checkable fact about a kernel whose
    builder takes a spec with no tri-state fields (gfx950 attention_dense: every
    tri-state on its gfx942 sibling lives on a private subclass that does not exist
    there, and `_use_exp2_fast` is not defined in the module at all).

    Collapsing them makes an honest profile look negligent: the gate reported
    'NOT CHECKED ... supply a profile so the policy checks actually execute' at an
    author who HAD supplied one, and coverage_gate.py turned that into a hard FAIL.
    The pressure that creates is to invent a policy entry to quiet the tool, which
    would then be checked against a function that does not exist.
    """

    def _empty_policy_profile(self, gate) -> Path:
        profile = gate.tmp / "empty_policies.yaml"
        profile.write_text(
            "bundle: test_engine\n"
            f"provider_root: {gate.tmp}\n"
            "vocabulary:\n  dtype: [BF16, FP16]\n"
            "policies: {}\n"
        )
        return profile

    def _run(self, gate, profile, tag="small"):
        return subprocess.run(
            [sys.executable, str(_TOOL), tag, tag, "--profile", str(profile)],
            cwd=gate.tmp,
            capture_output=True,
            text=True,
        )

    def test_an_explicit_empty_block_is_not_reported_as_narrowed(self, gate):
        result = self._run(gate, self._empty_policy_profile(gate))
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "metadata-matches-binary" not in result.stdout
        ), "an answered question must not be reported as an unasked one"
        assert "GATE PASSED:" in result.stdout, (
            "the unqualified pass line is the point: a vacuous check is satisfied, "
            "not skipped"
        )

    def test_omitting_the_key_entirely_still_narrows(self, gate):
        """The control. If this also passed unqualified, the distinction would be
        gone in the other direction and a genuinely unchecked set would report
        clean."""
        profile = gate.tmp / "no_policies.yaml"
        profile.write_text(
            "bundle: test_engine\n"
            f"provider_root: {gate.tmp}\n"
            "vocabulary:\n  dtype: [BF16, FP16]\n"
        )
        result = self._run(gate, profile)
        assert result.returncode == 0
        assert "NOT CHECKED" in result.stdout
        assert "metadata-matches-binary" in result.stdout
        assert "GATE PASSED on what it checked" in result.stdout

    def test_the_structural_checks_still_bite_under_an_empty_block(self, gate):
        """`policies: {}` must not become a way to quiet the WHOLE gate. A sentinel
        needs no kernel knowledge and must still fail."""
        bad = copy.deepcopy(gate.small)
        bad[0]["metadata"]["use_exp2_fast"] = -1
        gate.write("sentinel_set", bad)
        result = self._run(gate, self._empty_policy_profile(gate), tag="sentinel_set")
        assert result.returncode == 1
        assert "unset sentinel" in result.stdout


class TestMetadataMustAgreeWithTheSpecItIsBuiltFrom:
    """Property (4a): a metadata key that is ALSO a spec key must match it.

    This needs no profile, no policy and no kernel knowledge -- it is the descriptor
    checked against ITSELF. It did not exist until a review demonstrated the hole by
    mutation: property (4) iterated `profile.policies` and nothing else, so a kernel
    correctly declaring `policies: {}` had property (4) checking NOTHING while the
    gate printed a clean pass.

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
        # The point of (4a) being profile-free: it is the half of property (4) that
        # needs no kernel knowledge, so a narrowed run must still perform it.
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
        # DESIGN -- that is what the vocabulary declaration means. The shipped
        # fixtures are exactly this shape, so a check that flagged it would fail
        # every set in the tree.
        result = gate.run("small", "small")
        assert result.returncode == 0, result.stdout
        assert "metadata contradicts the spec" not in result.stdout

    def test_an_undeclared_string_field_is_not_guessed_at(self, gate):
        # Without a vocabulary declaration there is no way to know whether two
        # spellings of a string are a translation or a defect, so the check declines
        # to guess rather than inventing a failure. Numeric fields carry no such
        # ambiguity and ARE compared with no profile -- proven by
        # test_runs_without_a_profile_at_all above.
        ok = copy.deepcopy(gate.small)
        gate.write("ok", ok)
        result = gate.run("ok", "ok", profiled=False)
        assert result.returncode == 0, result.stdout

    def test_an_undeclared_string_field_is_named_not_silently_skipped(self, gate):
        # The decline above must not be invisible: a field nobody can judge is a
        # liability the author should see, exactly like the policy-knob narrowing
        # TestGateDegradesLoudly pins.
        ok = copy.deepcopy(gate.small)
        gate.write("ok", ok)
        result = gate.run("ok", "ok", profiled=False)
        assert result.returncode == 0, result.stdout
        assert "NOT CHECKED" in result.stdout
        assert "UNDECLARED STRING" in result.stdout
        assert "dtype" in result.stdout

    def test_a_string_field_absent_from_a_declared_vocabulary_is_still_compared(
        self, gate
    ):
        # Regression for the escape a review found on the real gfx950 tree: a
        # profile's `vocabulary:` block declares dtype but says nothing about
        # persist_decode, a second string field the two layers also both carry.
        # Once a vocabulary section exists at all, the author had the exact place
        # to declare persist_decode translated and did not -- so an unmentioned
        # string field is compared raw, same as any plain field, instead of being
        # waved through as "undeclared and therefore ambiguous".
        bad = copy.deepcopy(gate.small)
        bad[0]["kernel_source"]["spec"]["persist_decode"] = "auto"
        bad[0]["metadata"]["persist_decode"] = "manual"
        gate.write("bad", bad)
        result = gate.run("bad", "bad")
        assert result.returncode == 1, result.stdout
        assert "metadata contradicts the spec" in result.stdout
        assert "persist_decode" in result.stdout
        assert "spec='auto'" in result.stdout
        assert "metadata='manual'" in result.stdout
