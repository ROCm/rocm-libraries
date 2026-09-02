# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The reference library is the applicability oracle, and a decline it does not
share is a defect.

An engine's own bundles cannot establish this. They test the graphs the author
thought of against a reference the author chose, and they go green exactly when the
author's model of "what we support" is self-consistent -- which is also the state an
integration is in when it is silently under-covering.

The rule these tests defend:

    If the reference serves an equivalent request and its result validates, this
    integration must serve it too. A decline the reference does not share is missing
    coverage or a matcher bug -- never a scope decision.

The fourth case is real and deliberately out of scope here: a reference that ACCEPTS
a request it then computes wrongly. That is a reference defect and a finding to
report; it is not licence to decline quietly. This tool asks only about
applicability, never numerics, and its docstring says so.

Fixtures use a stub library rather than importing the real one: what is under test is
the RECONCILIATION LOGIC -- which combinations pass, which fail, and what the
diagnostic says -- not any particular kernel's support matrix. Binding these to a
real library would make them an integration test that breaks whenever that library's
coverage changes, which is the opposite of what they are for.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_TOOL = _TOOLS / "reconcile_applicability.py"

sys.path.insert(0, str(_TOOLS))

#: A stub standing in for both the kernel factory and the library entry point. Which
#: shapes each serves is controlled per-test by the thresholds baked into the module.
_STUB = '''
import dataclasses


@dataclasses.dataclass(frozen=True)
class Spec:
    batch: int
    seqlen_q: int
    head_size: int


class Request:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        # Structural rejection at CONSTRUCTION, before any predicate -- the case a
        # predicate-only support check silently reports as supported.
        if int(kw.get("head_size", 128)) not in (64, 128):
            raise ValueError("head_size must be 64 or 128")


class Candidate:
    def __init__(self, spec_id, algorithm, min_seqlen, opt_in=False, arches=()):
        self.spec_id, self.algorithm = spec_id, algorithm
        self.arches = arches
        self.family = "shared_family"   # deliberately the same for ALL candidates
        self._min, self._opt_in = min_seqlen, opt_in

    def _supports(self, req):
        """Residual checks ONLY. Deliberately does not re-check arch: real libraries
        move that into `capability`, so a predicate can happily accept a target its
        capability block forbids."""
        if self._opt_in and getattr(req, "algorithm", None) != self.algorithm:
            return False, f"{self.algorithm} is opt-in"
        if int(req.seqlen_q) < self._min:
            return False, f"seqlen_q must be at least {self._min}"
        return True, ""

    def admits(self, req):
        """The complete question: capability prefilter, then the predicate."""
        if self.arches and getattr(req, "arch", None) not in self.arches:
            return False, f"capability: arch {getattr(req, 'arch', None)!r} not in {self.arches}"
        return self._supports(req)


def candidates():
    return (
        Candidate("the_kernel", "dense", 256, opt_in=True, arches=("gfxstub",)),
        Candidate("other_arch", "dense", 1, opt_in=True, arches=("gfxother",)),
        Candidate("sibling", "tiled", 1),      # serves what dense refuses
    )


def kernel_spec(req):
    """THIS kernel: refuses short sequences."""
    if int(req.seqlen_q) < 256:
        raise ValueError("seqlen_q must be at least 256")
    return Spec(batch=int(req.batch), seqlen_q=int(req.seqlen_q),
                head_size=int(req.head_size))
'''


@pytest.fixture
def env(tmp_path):
    lib = tmp_path / "rocke" / "library"
    lib.mkdir(parents=True)
    (tmp_path / "rocke" / "platform" / "python").mkdir(parents=True)
    (lib / "stublib.py").write_text(_STUB)

    def profile(family="dense", match="algorithm", opt_in=True) -> Path:
        defaults = "{algorithm: dense}" if opt_in else "{}"
        path = tmp_path / f"profile_{family}_{match}_{opt_in}.yaml"
        path.write_text(
            textwrap.dedent(
                f"""
                provider_root: {tmp_path}
                slug: stub
                arch: gfxstub
                source: stub.py
                builder: build_stub
                engine: {{name: "hipkernel:Stub"}}
                kmd_fields: []
                metadata_fields: []
                dispatch: {{module: stublib, function: kernel_spec}}
                request:
                  module: stublib
                  class: Request
                  defaults: {defaults}
                reference_candidates:
                  module: stublib
                  function: candidates
                  match: {match}
                  family: {family}
                """
            )
        )
        return path

    def shapes(*specs) -> Path:
        path = tmp_path / "shapes.json"
        path.write_text(json.dumps(list(specs)))
        return path

    def run(profile_path, shapes_path, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(_TOOL),
                "--profile",
                str(profile_path),
                "--shapes",
                str(shapes_path),
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    return type(
        "Env",
        (),
        {
            "profile": staticmethod(profile),
            "shapes": staticmethod(shapes),
            "run": staticmethod(run),
        },
    )


_LONG = {"batch": 1, "seqlen_q": 4096, "head_size": 128}
_SHORT = {"batch": 1, "seqlen_q": 64, "head_size": 128}
_UNBUILDABLE = {"batch": 1, "seqlen_q": 4096, "head_size": 256}


class TestBothServe:
    def test_a_shape_both_serve_reconciles(self, env):
        """The control. Every failure assertion below is worthless without it."""
        result = env.run(env.profile(), env.shapes(_LONG))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RECONCILED" in result.stdout
        assert "both serve              1" in result.stdout


class TestScopingIsTheWholeDesign:
    """A sibling kernel's coverage is not this integration's gap.

    Comparing library-wide was tried against a real corpus and reported 51 shapes as
    gaps in a dense integration -- decode and large head sizes that the dense kernel
    declines for exactly the reasons hipDNN does, being served by sibling candidates.
    A false alarm with a plausible story attached is the expensive kind.
    """

    def test_a_shape_only_a_SIBLING_family_serves_is_not_a_gap(self, env):
        """The headline. The sibling serves the short sequence; we are integrating
        dense, which does not -- so this reconciles as a shared decline."""
        result = env.run(env.profile(family="dense"), env.shapes(_SHORT))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "both decline            1" in result.stdout
        assert "ONLY THE REFERENCE      0" in result.stdout

    def test_the_same_shape_IS_a_gap_when_integrating_that_sibling(self, env):
        """The converse, so the test above is not just 'always reconcile'. Point the
        profile at the tiled family and the identical shape becomes a real gap."""
        result = env.run(env.profile(family="tiled", opt_in=False), env.shapes(_SHORT))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "ONLY THE REFERENCE      1" in result.stdout

    def test_a_family_that_matches_nothing_is_refused(self, env):
        """A profile naming a family that does not exist would report EVERY shape as
        unreconciled -- a maximally alarming, entirely wrong result."""
        result = env.run(env.profile(family="no_such_kernel"), env.shapes(_LONG))
        assert result.returncode == 2
        assert "no registered candidate" in (result.stdout + result.stderr)

    def test_matching_on_the_wrong_attribute_is_refused(self, env):
        """Every stub candidate shares `family`, as the real library does -- so
        matching on it cannot discriminate and must not silently pass."""
        result = env.run(env.profile(family="dense", match="family"), env.shapes(_LONG))
        assert result.returncode == 2
        assert "no registered candidate" in (result.stdout + result.stderr)


class TestOptInSelectorIsInherited:
    def test_without_the_selector_the_oracle_asks_nothing(self, env):
        """An opt-in candidate declines everything unless the request names it, so a
        run that drops the selector reconciles trivially -- a gate that passes by
        asking nothing at all. With the selector, the long shape must SERVE."""
        with_selector = env.run(env.profile(opt_in=True), env.shapes(_LONG))
        assert "both serve              1" in with_selector.stdout

        without = env.run(env.profile(opt_in=False), env.shapes(_LONG))
        assert "both serve              0" in without.stdout, (
            "dropping the opt-in selector should make the reference decline "
            "everything -- if it does not, the fixture is not exercising opt-in"
        )


class TestOnlyTheReferenceServes:
    """The finding this tool exists for, within one family."""

    def test_the_diagnostic_names_both_sides(self, env):
        result = env.run(env.profile(family="tiled", opt_in=False), env.shapes(_SHORT))
        assert "sibling" in result.stdout
        assert "seqlen_q must be at least 256" in result.stdout

    def test_it_names_the_three_legitimate_responses(self, env):
        """'We chose not to' must not read as one of them."""
        result = env.run(env.profile(family="tiled", opt_in=False), env.shapes(_SHORT))
        assert "add the variant" in result.stdout
        assert "fix the matcher" in result.stdout
        assert "INCORRECT result" in result.stdout, (
            "the third option -- show the reference is wrong -- must be stated, or "
            "an author facing a genuine reference defect has no legitimate move"
        )

    def test_the_escape_hatch_is_explicit_and_still_reports(self, env):
        result = env.run(
            env.profile(family="tiled", opt_in=False),
            env.shapes(_SHORT),
            "--allow-unreconciled",
        )
        assert result.returncode == 0
        assert "UNDER PROTEST" in result.stdout
        assert "ONLY THE REFERENCE      1" in result.stdout


class TestBothDecline:
    def test_a_construction_rejection_counts_as_a_shared_decline(self, env):
        """Structural rejections raise at request CONSTRUCTION, before any predicate.
        Both sides hit it, so it reconciles -- but only because the tool builds the
        request inside the try on both sides."""
        result = env.run(env.profile(), env.shapes(_UNBUILDABLE))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "both decline            1" in result.stdout


class TestOracleDeclaration:
    def test_a_missing_oracle_is_a_named_error_not_a_pass(self, env, tmp_path):
        path = tmp_path / "no_oracle.yaml"
        path.write_text(
            textwrap.dedent(
                f"""
                provider_root: {tmp_path}
                slug: stub
                arch: gfxstub
                source: stub.py
                builder: build_stub
                engine: {{name: "hipkernel:Stub"}}
                kmd_fields: []
                metadata_fields: []
                dispatch: {{module: stublib, function: kernel_spec}}
                request: {{module: stublib, class: Request}}
                """
            )
        )
        result = env.run(path, env.shapes(_LONG))
        assert result.returncode == 2
        assert "no oracle" in (result.stdout + result.stderr)


class TestRuntimeDeclinesOverrideTheOfflineAnswer:
    def test_a_runtime_decline_beats_what_the_dispatcher_says(self, env, tmp_path):
        """What the engine ACTUALLY did outranks what the dispatcher says it could do."""
        declines = tmp_path / "declines.json"
        declines.write_text(json.dumps({"0": "no engine configurations available"}))
        result = env.run(env.profile(), env.shapes(_LONG), "--declines", str(declines))
        assert result.returncode == 1, result.stdout + result.stderr
        assert "ONLY THE REFERENCE      1" in result.stdout


class TestTheCompleteEligibilityQuestion:
    """Ask `admits`, not the raw predicate.

    Registered candidates keep their arch and dtype gates in `capability`, so the
    underscore predicate carries only the RESIDUAL checks -- the library's own
    docstring calls admits "the only eligibility question a caller should ask" and
    gives the worked example of a predicate that happily accepts a target its
    capability block forbids.

    Calling the predicate alone therefore reports the reference as serving a shape it
    cannot, which turns a CORRECT hipDNN decline into a phantom coverage gap -- the
    same class of false alarm as comparing library-wide, arriving by a different route.
    """

    def test_a_candidate_gated_out_by_capability_does_not_count_as_serving(self, env):
        """`other_arch` shares the dense family and its predicate accepts a short
        sequence, but its capability block excludes this arch. If the tool asked the
        predicate directly it would report the short shape as served by the reference,
        and flag our correct decline as a gap."""
        result = env.run(env.profile(family="dense"), env.shapes(_SHORT))
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "both decline            1" in result.stdout
        ), "an arch-excluded candidate must not count as the reference serving it"
        assert "ONLY THE REFERENCE      0" in result.stdout

    def test_the_arch_gated_candidate_would_otherwise_have_accepted(self, env):
        """Guards the test above from passing for the wrong reason: prove the
        predicate really does accept, so capability is what excluded it."""
        import subprocess as sp
        import sys as s

        probe = (
            "import sys; sys.path.insert(0, %r); import stublib;"
            "c = [x for x in stublib.candidates() if x.spec_id == 'other_arch'][0];"
            "r = stublib.Request(batch=1, seqlen_q=64, head_size=128,"
            "                    arch='gfxstub', algorithm='dense');"
            "print('predicate:', c._supports(r)[0], '| admits:', c.admits(r)[0])"
        )
        lib = str(Path(env.profile()).parent / "rocke" / "library")
        out = sp.run(
            [s.executable, "-c", probe % lib], capture_output=True, text=True
        ).stdout.strip()
        assert out == "predicate: True | admits: False", out


class TestDeclineReasonsAreNotMaskedBySiblings:
    """The verdict can be right while every recorded reason is wrong.

    Scoping on a shared attribute matches several candidates. If one rejects on
    CAPABILITY (wrong arch) that says only "this sibling is not the one for this
    target" -- true, and useless as evidence. Keeping whichever decline came last
    made every recorded reason the capability one and masked the real,
    kernel-specific reason on every shape.

    Counts were unaffected, which is what makes it dangerous: the gate passes, the
    write-up records a uniformly wrong root cause, and nothing looks broken. On the
    real corpus this contaminated 51 of 51 decline reasons.
    """

    def test_the_substantive_reason_wins_over_a_capability_rejection(self, env):
        """`other_arch` shares the family and rejects this arch on capability;
        `the_kernel` rejects on the real predicate. The report must show the latter."""
        result = env.run(env.profile(family="dense"), env.shapes(_SHORT))
        assert result.returncode == 0, result.stdout + result.stderr
        assert (
            "seqlen_q must be at least 256" in result.stdout
        ), "the kernel-specific reason was masked by a sibling's capability gate"
        assert (
            "capability: arch" not in result.stdout
        ), "a sibling's capability rejection is not evidence about this shape"

    def test_the_reason_names_which_candidate_gave_it(self, env):
        """With several candidates in a family, an unattributed reason cannot be
        acted on -- a reader cannot tell which sibling spoke."""
        result = env.run(env.profile(family="dense"), env.shapes(_SHORT))
        assert "[the_kernel]" in result.stdout


class TestAGateThatCannotPassByAskingNothing:
    """Three ways this tool reported success without comparing anything."""

    def test_a_declines_key_matching_no_shape_is_a_hard_failure(self, env, tmp_path):
        """A typo'd graph name was silently ignored: the shape it marked stayed
        counted as served and the run exited 0 having asked nothing about it.

        Index keys make this worse, not better -- they are the only option for a
        corpus without graph names, and they shift when the corpus is re-mined with
        different flags, so the same file quietly marks a DIFFERENT shape."""
        declines = tmp_path / "typo.json"
        declines.write_text(json.dumps({"no_such_graph_name": "engine declined"}))
        result = env.run(env.profile(), env.shapes(_LONG), "--declines", str(declines))
        assert result.returncode == 2, result.stdout
        assert "matched no shape" in result.stderr

    def test_a_key_that_does_match_is_accepted(self, env, tmp_path):
        """So the failure above is about the key matching nothing, not about
        --declines being rejected wholesale."""
        declines = tmp_path / "real.json"
        declines.write_text(json.dumps({"0": "engine declined at runtime"}))
        result = env.run(env.profile(), env.shapes(_LONG), "--declines", str(declines))
        assert "matched no shape" not in result.stderr
        assert "ONLY THE REFERENCE      1" in result.stdout

    def test_neither_side_serving_anything_is_not_agreement(self, env):
        """A profile whose request.defaults omits the opt-in selector makes the
        reference decline every shape -- and this integration decline them for the
        same reason. The tool used to print RECONCILED over 0-of-N served.

        The gate needs BOTH conditions: NOTHING served by either side, and the
        scoping key absent from request.defaults. A shape either side serves proves
        the comparison is live; a corpus nothing serves is otherwise legitimate."""
        result = env.run(env.profile(family="dense", opt_in=False), env.shapes(_SHORT))
        assert result.returncode == 2, result.stdout
        assert "agreement about nothing" in result.stderr
        assert "algorithm" in result.stderr, "the diagnostic must name the missing key"

    def test_an_empty_comparison_can_be_opted_into(self, env):
        """The check guards a misconfiguration; it does not claim an empty corpus is
        never legitimate."""
        result = env.run(
            env.profile(family="dense", opt_in=False),
            env.shapes(_SHORT),
            "--allow-empty",
        )
        assert result.returncode == 0

    def test_an_unreadable_declines_path_names_the_problem(self, env, tmp_path):
        """It used to raise IsADirectoryError through the traceback."""
        result = env.run(env.profile(), env.shapes(_LONG), "--declines", str(tmp_path))
        assert result.returncode == 2
        assert "FAIL: --declines" in result.stderr
        assert "Traceback" not in result.stderr

    def test_a_declines_file_that_is_not_a_mapping_is_refused(self, env, tmp_path):
        declines = tmp_path / "list.json"
        declines.write_text(json.dumps(["0"]))
        result = env.run(env.profile(), env.shapes(_LONG), "--declines", str(declines))
        assert result.returncode == 2
        assert "must be a JSON mapping" in result.stderr


class TestServingWhatTheReferenceDeclines:
    """The opposite direction, which used to be filed as agreement.

    The bucket branch keyed only on the reference's answer, so a shape WE serve and
    the reference declines landed in "both decline" -- counted as reconciled, with
    the reference's decline reason printed as though we shared it.
    """

    def test_a_shape_only_we_serve_is_reported_separately(self, env, tmp_path):
        """Our dispatch serves the long shape. Point the reference at a family whose
        only candidate rejects this arch on capability, and the reference declines
        what we serve -- which is neither a gap nor agreement."""
        lib = tmp_path / "rocke" / "library"
        (lib / "narrow.py").write_text(
            "import stublib\n"
            "class Narrow(stublib.Candidate):\n"
            "    pass\n"
            "def candidates():\n"
            "    return (Narrow('elsewhere', 'dense', 1, arches=('gfxother',)),)\n"
        )
        path = tmp_path / "narrow.yaml"
        path.write_text(
            textwrap.dedent(
                f"""
                provider_root: {tmp_path}
                slug: stub
                arch: gfxstub
                source: stub.py
                builder: build_stub
                engine: {{name: "hipkernel:Stub"}}
                kmd_fields: []
                metadata_fields: []
                dispatch: {{module: stublib, function: kernel_spec}}
                request:
                  module: stublib
                  class: Request
                  defaults: {{algorithm: dense}}
                reference_candidates:
                  module: narrow
                  function: candidates
                  match: algorithm
                  family: dense
                """
            )
        )
        result = env.run(path, env.shapes(_LONG))
        assert "only this integration   1" in result.stdout, result.stdout
        assert (
            "both decline            0" in result.stdout
        ), "a shape we serve is not a shared decline"

    def test_a_live_comparison_is_never_called_vacuous(self, env):
        """The control for the two above. With the scoping key still absent, a shape
        OUR side serves makes the comparison live -- the gate must stay quiet, or it
        would fail every run whose profile happens to omit a default."""
        result = env.run(env.profile(family="dense", opt_in=False), env.shapes(_LONG))
        assert "asked nothing" not in result.stderr
        assert "agreement about nothing" not in result.stderr
