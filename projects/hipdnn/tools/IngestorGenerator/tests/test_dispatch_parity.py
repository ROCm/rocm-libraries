# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What `dispatch_parity.py` reports, and what it refuses to report.

A shape that does not end up served has exactly ONE per-shape explanation here: the
eligibility predicate ran, returned false, and gave a reason. Spec construction
failing is not a second explanation -- it aborts the command, because a corpus the
request class cannot even hydrate makes every remaining count untrustworthy, and a
tool that reported "n rejected" beside a servable count would be inviting the reader
to trust the rest of the line.

These tests exist because the counted-bucket spelling is the one that looks right.
A `rejected` bucket reads as thorough, but it is unpopulatable, and a count that can
only ever print 0 tells a reader the failure it names was checked for and did not
happen -- a stronger claim than the tool can make.

Nothing here imports rocKE. The dispatcher, the request class and the predicate are
supplied as stubs written into a fake provider root, because what is under test is
the tool's own reporting and error policy, not any particular kernel's answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_parity  # noqa: E402

#: The decline carries a real reason rather than a blanket refusal so the served
#: control survives alongside it -- a stub that refused everything would satisfy a
#: "declined is reported" check while proving nothing about the split.
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
    """A profile, a corpus and a provider root the tool can actually bind.

    Returns a callable over the shape list, so each test states its own corpus and
    nothing is shared but the stubs.
    """
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
        """The regression pin -- the tool is already right here.

        Only two `kind` values can exist: the dataclass default "constructed" and the
        "declined" the predicate path sets. Reaching this line at all means a
        construction failure could not have occurred -- it returns 2 long before the
        summary prints.
        """
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
        """The other control: `--report-gaps` is the tool's whole answer to an
        uncovered shape, and it prints from the same loop the dead bucket would be
        concatenated into."""
        assert dispatch_parity.main(parity(_SERVED_AND_DECLINED, "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" in out
        assert "seqlen_q 777 is not a supported prefill length" in out

    def test_report_gaps_prints_nothing_when_every_shape_is_served(
        self, parity, capsys
    ):
        """No gaps means no gap lines -- not a bucket header with 0 under it."""
        assert dispatch_parity.main(parity([{"seqlen_q": 256}], "--report-gaps")) == 0
        out = capsys.readouterr().out
        assert "[declined]" not in out
        assert "rejected" not in out


class TestConstructionFailureAbortsRatherThanBuckets:
    def test_an_unhydratable_shape_exits_2_naming_the_failure(self, parity, capsys):
        """A corpus key the request class does not accept is not a shape-level
        verdict: the tool cannot say whether the kernel would serve it."""
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
        dispatcher that raises is an operational failure too -- never a decline.

        The shape hydrates and the per-shape loop is entered before the failure, so
        the tool has reached the point where it COULD bucket the shape and aborts
        instead. The dispatcher's own message is asserted for that reason -- without
        it, an exit 2 raised while resolving the symbol would pass.
        """
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
        """The other direction: the abort policy must not swallow the one outcome
        that IS a per-shape verdict."""
        assert dispatch_parity.main(parity([{"seqlen_q": 777}])) == 1
        assert "no shape resolved" in capsys.readouterr().err


class TestTheDocstringStatesTheActualErrorPolicy:
    """The docstring is the only place the error policy is written down.

    A docstring promising a two-bucket report is worse than silence: a reader
    reconciling counts against it looks for a second bucket that does not exist and
    concludes the tool is broken.
    """

    def test_it_does_not_promise_a_second_rejection_bucket(self):
        doc = dispatch_parity.__doc__
        assert "Both rejection kinds are reported here" not in doc
        assert "they differ by 59" not in doc, (
            "the docstring cites a two-denominator delta this tool has no code path "
            "to compute"
        )

    def test_it_states_the_abort_and_the_exit_status(self):
        doc = dispatch_parity.__doc__.lower()
        assert "exit 2" in doc, "the error policy's observable outcome is unstated"
