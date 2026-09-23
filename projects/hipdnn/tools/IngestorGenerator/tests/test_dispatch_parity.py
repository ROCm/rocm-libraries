# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What `dispatch_parity.py` reports, and what it refuses to report.

A shape that is not served has exactly ONE per-shape explanation: the eligibility
predicate ran, returned false, and gave a reason. Spec construction failing aborts the
command instead, because a corpus the request class cannot hydrate makes every
remaining count untrustworthy, and a `rejected` bucket that can only print 0 claims a
failure was checked for. The dispatcher, request class and predicate are stubs.

Also what the tool BINDS before it can report anything: where a profile's
``provider_root`` resolves from, and which dispatch arm the shipped profile pins.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_parity  # noqa: E402
import launch_surface  # noqa: E402

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_SHIPPED_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx950_attention_dense.profile.yaml"
)

#: The decline carries a real reason rather than a blanket refusal so the served
#: control survives alongside it.
_STUB_PROVIDER = '''
import dataclasses


@dataclasses.dataclass
class Request:
    seqlen_q: int
    head_size: int = 128


@dataclasses.dataclass
class Spec:
    seqlen_q: int
    head_size: int
    block_n: int


def resolve(request):
    """Derive a field rather than defaulting it, as a real dispatcher would."""
    return Spec(
        seqlen_q=request.seqlen_q,
        head_size=request.head_size,
        block_n=64 if request.seqlen_q >= 1024 else 32,
    )


def resolve_but_raise(request):
    """A dispatcher that fails operationally once the per-shape loop calls it.

    The message names the request it was handed, so a test can tell "the factory
    ran and threw" apart from "the factory was never reached".
    """
    raise ValueError(f"dispatcher exploded on seqlen_q {request.seqlen_q}")


def supports(spec, arch=None):
    if spec.seqlen_q == 777:
        return False, "seqlen_q 777 is not a supported prefill length"
    return True, ""
'''


@pytest.fixture
def parity(tmp_path, monkeypatch):
    """A profile, a corpus and a provider root the tool can bind. Returns a callable
    over the shape list, so each test states its own corpus."""
    library = tmp_path / "provider" / "rocke" / "library"
    library.mkdir(parents=True)
    (tmp_path / "provider" / "rocke" / "platform" / "python").mkdir(parents=True)
    (library / "stub_provider.py").write_text(_STUB_PROVIDER)
    # The tool inserts the provider dirs itself; popping the module keeps one test's
    # import from satisfying the next one's from a stale sys.modules entry.
    monkeypatch.delitem(sys.modules, "stub_provider", raising=False)

    profile = {
        "slug": "stub_attention",
        "source": "kernels/stub.py",
        "builder": "build_stub",
        "engine": {"name": "stub:Engine"},
        "kmd_fields": [{"name": "seqlen_q", "type": "int", "default_value": 256}],
        "provider_root": str(tmp_path / "provider"),
        "dispatch": {"module": "stub_provider", "function": "resolve"},
        "request": {"module": "stub_provider", "class": "Request"},
        "predicate": {"module": "stub_provider", "function": "supports"},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))

    def argv(shapes: list, *extra: str) -> list:
        shapes_path = tmp_path / "shapes.json"
        shapes_path.write_text(json.dumps(shapes))
        return [
            "--profile",
            str(profile_path),
            "--shapes",
            str(shapes_path),
            *extra,
        ]

    return argv


_SERVED_AND_DECLINED = [{"seqlen_q": 256}, {"seqlen_q": 2048}, {"seqlen_q": 777}]


class TestTheReportCarriesNoUnpopulatableBucket:
    def test_the_summary_names_no_rejected_bucket(self, parity, capsys):
        """Only two `kind` values can exist: the dataclass default "constructed" and the
        "declined" the predicate path sets, since a construction failure returns 2 long
        before the summary prints."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED)) == 0
        out = capsys.readouterr().out
        assert "rejected" not in out, (
            "the summary still prints a bucket nothing can populate; a count that "
            "is structurally always 0 reads as a check that passed"
        )
        assert "spec construction raised" not in out, (
            "the summary still offers spec construction as a per-shape outcome, but "
            "that path aborts the command instead of bucketing the shape"
        )

    def test_the_counts_that_remain_are_still_right(self, parity, capsys):
        """A control: the live counts are right, so an absent `rejected` line is a
        report with a bucket missing rather than a harness that printed nothing."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED)) == 0
        out = capsys.readouterr().out
        assert "shapes in         3" in out
        assert "servable          2" in out
        assert "declined          1" in out

    def test_report_gaps_lists_the_decline_with_its_reason(self, parity, capsys):
        """`--report-gaps` is the tool's whole answer to an uncovered shape, and it
        prints from the same loop the dead bucket would join."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED, "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" in out
        assert "seqlen_q 777 is not a supported prefill length" in out

    def test_report_gaps_prints_nothing_when_every_shape_is_served(
        self, parity, capsys
    ):
        """No gaps means no gap lines, not a bucket header with 0 under it."""
        assert dispatch_parity.main(parity([{"seqlen_q": 256}], "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" not in out
        assert "rejected" not in out


class TestConstructionFailureAbortsRatherThanBuckets:
    def test_an_unhydratable_shape_exits_2_naming_the_failure(self, parity, capsys):
        """A corpus key the request class does not accept is not a shape-level verdict:
        the tool cannot say whether the kernel would serve it."""
        shapes = [{"seqlen_q": 256}, {"seqlen_q": 512, "nonexistent_field": 1}]
        assert dispatch_parity.main(parity(shapes)) == 2

        captured = capsys.readouterr()
        assert "request/spec construction failed" in captured.err
        assert captured.err.startswith("FAIL:")
        assert "dispatcher parity" not in captured.out, (
            "a summary was printed for a corpus that failed to hydrate; the counts "
            "would describe only the shapes processed before the failure"
        )

    def test_a_dispatcher_that_raises_also_exits_2(self, parity, capsys, monkeypatch):
        """The factory is inside the same try as the request constructor, so a
        dispatcher that raises is operational, never a decline. The dispatcher's own
        message is asserted so an exit 2 raised while resolving the symbol does not
        pass."""
        shapes = [{"seqlen_q": 256}]
        argv = parity(shapes)
        real_resolve_shapes = dispatch_parity.resolve_shapes

        def with_raising_dispatcher(shapes_arg, profile):
            profile = dict(profile)
            profile["dispatch"] = {
                "module": "stub_provider",
                "function": "resolve_but_raise",
            }
            return real_resolve_shapes(shapes_arg, profile)

        monkeypatch.setattr(dispatch_parity, "resolve_shapes", with_raising_dispatcher)
        assert dispatch_parity.main(argv) == 2

        captured = capsys.readouterr()
        assert "FAIL:" in captured.err
        assert "dispatcher exploded on seqlen_q 256" in captured.err
        assert "request/spec construction failed" in captured.err
        assert "dispatcher parity" not in captured.out, (
            "a summary was printed for a corpus whose dispatcher raised; the counts "
            "would describe only the shapes resolved before the failure"
        )

    def test_a_predicate_decline_is_not_promoted_to_an_abort(self, parity, capsys):
        """The abort policy must not swallow the one outcome that IS a per-shape
        verdict."""
        assert dispatch_parity.main(parity([{"seqlen_q": 777}])) == 1
        assert "no shape resolved" in capsys.readouterr().err


#: The relative root the shipped profile names. Spelled out so the decoy below can
#: reproduce it exactly: under a cwd-relative resolution the decoy is what binds.
_REPO_RELATIVE_ROOT = "dnn-providers/hip-kernel-provider"


def _make_provider(root: Path) -> Path:
    """The two directories ``_bind_provider`` requires of a provider root."""
    (root / "rocke" / "library").mkdir(parents=True)
    (root / "rocke" / "platform" / "python").mkdir(parents=True)
    return root


class TestProviderBindingIsIndependentOfTheInvocationDirectory:
    """``provider_root`` is repository-relative, anchored on the tool's own location.

    A root resolved against the current directory makes one profile correct from one
    directory and silently wrong from every other: the import fails where the tree is
    absent, and -- worse -- binds a same-shaped tree that happens to sit under the
    caller's cwd.
    """

    @pytest.fixture
    def bind(self, monkeypatch):
        """Bind a root and return the entries it ADDED to ``sys.path``.

        The difference, not a substring scan of the whole path: earlier tests in this
        module bind stub providers of their own, and a scan would report those too. A
        copy of ``sys.path`` is swapped in for the duration, so one test's provider
        cannot satisfy the next one's import and no real rocKE library outlives it.
        """
        monkeypatch.setattr(sys, "path", list(sys.path))
        baseline = list(sys.path)

        def _bind(root):
            dispatch_parity._bind_provider(root)
            return [entry for entry in sys.path if entry not in baseline]

        return _bind

    def test_a_relative_root_binds_the_checkout_not_a_look_alike_under_the_cwd(
        self, bind, tmp_path, monkeypatch
    ):
        """The decoy has the SAME relative layout and sits at the cwd, so a pass here
        cannot be explained by the tool simply failing to find anything."""
        decoy = _make_provider(tmp_path / "decoy" / _REPO_RELATIVE_ROOT)
        monkeypatch.chdir(tmp_path / "decoy")

        added = bind(_REPO_RELATIVE_ROOT)

        assert added, "nothing was bound at all"
        assert not [entry for entry in added if str(decoy) in entry], (
            f"the look-alike tree under the current directory was bound: {added}. "
            "The root was resolved against the cwd, which is exactly F7"
        )
        expected = launch_surface.find_repo_root(_TOOLS) / _REPO_RELATIVE_ROOT
        for entry in added:
            assert str(expected) in entry, (
                f"{entry!r} is not under the checkout's {expected} -- the anchor is "
                "neither the cwd nor the repository root"
            )

    def test_the_shipped_profile_binds_from_an_unrelated_directory(
        self, bind, tmp_path, monkeypatch
    ):
        """End to end on the value that actually ships, so a correct resolver paired
        with a stale profile string still fails."""
        profile = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))
        root = profile["provider_root"]
        assert not os.path.isabs(root), (
            f"the shipped profile names an absolute provider root ({root!r}); it "
            "would only resolve on the machine that wrote it"
        )
        monkeypatch.chdir(tmp_path)

        added = bind(root)

        assert added, (
            f"the shipped provider_root {root!r} did not bind from an unrelated "
            "directory"
        )

    def test_a_relative_root_that_is_absent_names_the_checkout_it_looked_under(
        self, bind, tmp_path, monkeypatch
    ):
        """The failure has to say where it looked, or a mis-anchored root reads as a
        missing checkout."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(dispatch_parity.ParityError) as excinfo:
            bind("no/such/provider")
        message = str(excinfo.value)
        assert str(launch_surface.find_repo_root(_TOOLS)) in message, message
        assert (
            str(tmp_path) not in message
        ), f"the message names the current directory: {message}"

    def test_an_absolute_root_is_still_taken_verbatim(
        self, bind, tmp_path, monkeypatch
    ):
        """Repository-relative resolution is for relative values only; an absolute
        root may legitimately point outside the checkout."""
        provider = _make_provider(tmp_path / "elsewhere")
        monkeypatch.chdir(tmp_path)
        added = bind(str(provider))
        assert [entry for entry in added if str(provider) in entry], added

    def test_a_user_relative_root_is_still_expanded(self, bind, tmp_path, monkeypatch):
        """``~`` expands to an absolute path, so it must not fall into the
        repository-relative branch and be looked for inside the checkout."""
        provider = _make_provider(tmp_path / "home" / "provider")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        added = bind(os.path.join("~", "provider"))
        assert [entry for entry in added if str(provider) in entry], added

    def test_the_environment_is_the_fallback_when_the_profile_names_none(
        self, bind, tmp_path, monkeypatch
    ):
        provider = _make_provider(tmp_path / "from_env")
        monkeypatch.setenv("ROCKE_PROVIDER_ROOT", str(provider))
        added = bind(None)
        assert [entry for entry in added if str(provider) in entry], added

    def test_a_nonempty_profile_value_still_outranks_the_environment(
        self, bind, tmp_path, monkeypatch
    ):
        chosen = _make_provider(tmp_path / "from_profile")
        ignored = _make_provider(tmp_path / "from_env")
        monkeypatch.setenv("ROCKE_PROVIDER_ROOT", str(ignored))
        added = bind(str(chosen))
        assert [entry for entry in added if str(chosen) in entry], added
        assert not [entry for entry in added if str(ignored) in entry], added

    def test_the_sibling_tools_share_this_one_binding(self):
        """``knob_sweep`` and ``reconcile_applicability`` read the same ``provider_root``
        out of the same profiles. They must reach it through THIS function rather than
        resolving a root of their own, or the fix above holds for one tool only."""
        import knob_sweep
        import reconcile_applicability

        assert knob_sweep._bind_provider is dispatch_parity._bind_provider
        assert reconcile_applicability._bind_provider is dispatch_parity._bind_provider


class TestTheShippedProfilePinsTheDispatchArmItsCatalogWasBuiltFrom:
    """F6: the request defaults decide which kernel the dispatcher resolves, and this
    catalog contains one arm of that choice only."""

    def test_dense_persistent_is_the_string_off_not_a_yaml_boolean(self):
        defaults = dispatch_parity._load_profile(str(_SHIPPED_PROFILE))["request"][
            "defaults"
        ]
        assert "dense_persistent" in defaults, (
            "the profile leaves dense_persistent unset, so AttentionRequest defaults "
            "it to 'auto' and the dispatcher resolves the persistent arm once "
            "work >= dense_num_persistent -- a kernel this catalog does not ship"
        )
        value = defaults["dense_persistent"]
        assert isinstance(value, str), (
            f"dense_persistent parsed as {type(value).__name__} ({value!r}): the key "
            "was written unquoted and PyYAML read `off` as a boolean. The dispatcher "
            "calls .strip().lower() on it, so the tool aborts rather than pinning "
            "the non-persistent arm"
        )
        assert value == "off", value
