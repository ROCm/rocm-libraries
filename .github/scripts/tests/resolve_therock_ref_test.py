import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import resolve_therock_ref as rtr
from resolve_therock_ref import Commit


NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def dt(days_ago: float) -> datetime:
    return NOW - timedelta(days=days_ago)


class FakeClient:
    """In-memory GitHub client for exercising resolve_ref without HTTP."""

    def __init__(
        self,
        *,
        merge_base: Optional[Commit] = None,
        at_or_before: Optional[Commit] = None,
        live_tip: Optional[Commit] = None,
        named_commit: Optional[Commit] = None,
    ):
        self._merge_base = merge_base
        self._at_or_before = at_or_before
        self._live_tip = live_tip
        self._named_commit = named_commit
        self.calls: list[str] = []

    def get_merge_base(self, repo: str, base_sha: str, head_sha: str) -> Commit:
        self.calls.append("get_merge_base")
        assert self._merge_base is not None
        return self._merge_base

    def get_commit_at_or_before(
        self, repo: str, branch: str, until: datetime
    ) -> Optional[Commit]:
        self.calls.append("get_commit_at_or_before")
        return self._at_or_before

    def get_live_tip(self, repo: str, branch: str) -> Commit:
        self.calls.append("get_live_tip")
        assert self._live_tip is not None
        return self._live_tip

    def get_commit(self, repo: str, ref: str) -> Commit:
        self.calls.append("get_commit")
        assert self._named_commit is not None
        return self._named_commit


class ResolveRefTest(unittest.TestCase):
    def test_pull_request_maps_merge_base_time_to_therock_sha(self):
        merge_base = Commit(sha="a" * 40, committed_at=dt(2))
        chosen = Commit(sha="b" * 40, committed_at=dt(2.1))
        client = FakeClient(merge_base=merge_base, at_or_before=chosen)

        result = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            now=NOW,
        )

        self.assertEqual(result.therock_ref, "b" * 40)
        self.assertEqual(result.mode, rtr.MODE_MERGE_BASE)
        self.assertEqual(result.merge_base, merge_base)
        self.assertEqual(result.warnings, [])
        self.assertIn("get_merge_base", client.calls)
        self.assertIn("get_commit_at_or_before", client.calls)
        self.assertNotIn("get_live_tip", client.calls)

    def test_override_short_circuits(self):
        override_commit = Commit(sha="c" * 40, committed_at=dt(1))
        client = FakeClient(named_commit=override_commit)

        result = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="v1.2.3",
            now=NOW,
        )

        self.assertEqual(result.therock_ref, "c" * 40)
        self.assertEqual(result.mode, rtr.MODE_OVERRIDE)
        self.assertEqual(client.calls, ["get_commit"])

    def test_push_uses_live_tip(self):
        tip = Commit(sha="d" * 40, committed_at=dt(0.5))
        client = FakeClient(live_tip=tip)

        result = rtr.resolve_ref(
            client,
            event_name="push",
            source_repo="ROCm/rocm-libraries",
            base_sha="",
            head_sha="",
            override="",
            now=NOW,
        )

        self.assertEqual(result.therock_ref, "d" * 40)
        self.assertEqual(result.mode, rtr.MODE_LIVE_TIP)
        self.assertEqual(client.calls, ["get_live_tip"])

    def test_pull_request_falls_back_to_live_tip_when_nothing_before_T(self):
        merge_base = Commit(sha="a" * 40, committed_at=dt(400))
        tip = Commit(sha="e" * 40, committed_at=dt(0.5))
        client = FakeClient(merge_base=merge_base, at_or_before=None, live_tip=tip)

        result = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            now=NOW,
        )

        self.assertEqual(result.therock_ref, "e" * 40)
        self.assertEqual(result.mode, rtr.MODE_MERGE_BASE_FALLBACK)
        self.assertIn("get_live_tip", client.calls)
        self.assertTrue(any("fell back to the live tip" in w for w in result.warnings))

    def test_staleness_warns_but_still_returns_sha(self):
        merge_base = Commit(sha="a" * 40, committed_at=dt(30))
        chosen = Commit(sha="b" * 40, committed_at=dt(30))
        client = FakeClient(merge_base=merge_base, at_or_before=chosen)

        result = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            staleness_days=14,
            now=NOW,
        )

        self.assertEqual(result.therock_ref, "b" * 40)
        self.assertTrue(any("days old" in w for w in result.warnings))

    def test_staleness_quiet_when_fresh(self):
        merge_base = Commit(sha="a" * 40, committed_at=dt(1))
        chosen = Commit(sha="b" * 40, committed_at=dt(1))
        client = FakeClient(merge_base=merge_base, at_or_before=chosen)

        result = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            staleness_days=14,
            now=NOW,
        )

        self.assertEqual(result.warnings, [])


class BuildSummaryTest(unittest.TestCase):
    def _pr_resolution(self) -> rtr.Resolution:
        merge_base = Commit(sha="a" * 40, committed_at=dt(2))
        chosen = Commit(sha="b" * 40, committed_at=dt(2))
        client = FakeClient(merge_base=merge_base, at_or_before=chosen)
        return rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            now=NOW,
        )

    def test_summary_includes_both_times_and_explanation(self):
        summary = rtr.build_summary(self._pr_resolution(), now=NOW)

        # Plain-language explanation of the mechanism is present.
        self.assertIn("merge-base", summary)
        self.assertIn("stays frozen", summary)
        # Both timestamps rendered (merge-base time T and TheRock commit time).
        self.assertEqual(summary.count(rtr.iso_utc(dt(2))), 2)
        # Resolved TheRock commit is linked, merge-base labeled, mode shown.
        self.assertIn("Resolved TheRock commit", summary)
        self.assertIn("Merge-base commit", summary)
        self.assertIn("How to change this", summary)
        self.assertIn(("b" * 40)[:12], summary)

    def test_summary_live_tip_is_not_pr_worded(self):
        tip = Commit(sha="d" * 40, committed_at=dt(0.5))
        client = FakeClient(live_tip=tip)
        resolution = rtr.resolve_ref(
            client,
            event_name="push",
            source_repo="ROCm/rocm-libraries",
            base_sha="",
            head_sha="",
            override="",
            now=NOW,
        )

        summary = rtr.build_summary(resolution, now=NOW)
        self.assertIn("live tip", summary)
        self.assertNotIn("This PR is built against", summary)
        self.assertNotIn("stays frozen", summary)
        # No "merge or rebase your base branch" guidance in non-PR modes.
        self.assertNotIn("merge or rebase your base branch", summary)

    def test_summary_override_is_not_pr_worded(self):
        override_commit = Commit(sha="c" * 40, committed_at=dt(0.5))
        client = FakeClient(named_commit=override_commit)
        resolution = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="v1.2.3",
            now=NOW,
        )

        summary = rtr.build_summary(resolution, now=NOW)
        self.assertIn("explicitly pinned", summary)
        self.assertNotIn("This PR is built against", summary)
        self.assertNotIn("merge or rebase your base branch", summary)

    def test_staleness_hint_is_mode_agnostic_for_live_tip(self):
        tip = Commit(sha="d" * 40, committed_at=dt(30))
        client = FakeClient(live_tip=tip)
        resolution = rtr.resolve_ref(
            client,
            event_name="push",
            source_repo="ROCm/rocm-libraries",
            base_sha="",
            head_sha="",
            override="",
            staleness_days=14,
            now=NOW,
        )

        warning = " ".join(resolution.warnings)
        self.assertIn("days old", warning)
        self.assertNotIn("your base branch", warning)
        self.assertIn("therock_ref_override", warning)

    def test_summary_surfaces_warning(self):
        merge_base = Commit(sha="a" * 40, committed_at=dt(30))
        chosen = Commit(sha="b" * 40, committed_at=dt(30))
        client = FakeClient(merge_base=merge_base, at_or_before=chosen)
        resolution = rtr.resolve_ref(
            client,
            event_name="pull_request",
            source_repo="ROCm/rocm-libraries",
            base_sha="base",
            head_sha="head",
            override="",
            staleness_days=14,
            now=NOW,
        )

        summary = rtr.build_summary(resolution, now=NOW)
        self.assertIn("[!WARNING]", summary)
        self.assertIn("days old", summary)


if __name__ == "__main__":
    unittest.main()
