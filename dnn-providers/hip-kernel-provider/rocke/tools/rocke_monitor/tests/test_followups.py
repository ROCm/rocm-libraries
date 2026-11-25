# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk followup-scan N` and the tick's scan on merge: every reviewer's open items,
resolved threads included, as candidates for the follow-up tickets."""

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import followups, gh, registry  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

CFG = {
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "rocke_path": "rk",
    "deployment": {"phase": "review"},
    "merge_authority": "propose",
}
URL = "https://github.com/ROCm/r/pull/7"
# Line 1 of our review, giving this deployment's automated reviewer name (the container's: CFG sets none), by which
# the scan knows the review for ours.
LINE1 = "Automated review by rocke-monitor: Model - x, effort: max"
BOTS = ("copilot-pull-request-reviewer", "github-advanced-security")
TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket


def c(cid, who, at, body):
    return {
        "databaseId": cid,
        "url": f"{URL}#discussion_r{cid}",
        "createdAt": f"2026-11-0{at}Z",
        "body": body,
        "author": {"login": who, "__typename": "Bot" if who in BOTS else "User"},
    }


def thread(path, line, comments, resolved=True, outdated=False, by="dev"):
    return {
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": f"rk/{path}",
        "line": line,
        "originalLine": line,
        "resolvedBy": {"login": by} if resolved else None,
        "comments": {"nodes": comments},
    }


THREADS = [
    thread(
        "platform/a.py",
        12,
        [
            c(201, "rev1", "1T10:00:00", "Please add a bounds check here."),
            c(
                202,
                "dev",
                "1T11:00:00",
                f"Good catch. Out of time here; I will do it in a follow-up PR ({TK}-1234).",
            ),
        ],
    ),
    thread(
        "platform/b.py",
        40,
        [
            c(
                203,
                BOTS[0],
                "1T10:05:00",
                "This loop never ends when n == 0, and the result is used later. Consider checking "
                "n first.",
            )
        ],
    ),
    thread(
        "platform/c.py",
        7,
        [
            c(
                204, "rev1", "1T10:10:00", "Should this be int64?"
            ),  # a co-author answered
            c(205, "dev2", "1T12:00:00", "Fixed in bbbbbbb."),
        ],
    ),
    thread(
        "platform/d.py",
        9,
        [c(206, "rev2", "1T10:20:00", "Please rename this.")],
        outdated=True,
    ),  # the code moved
    thread(
        "platform/e.py",
        3,
        [c(207, BOTS[1], "1T10:30:00", "Unused import? Please remove it.")],
        resolved=False,
    ),
]
COMMENTS = [
    {
        "id": 11,
        "login": "github-actions[bot]",
        "type": "Bot",
        "url": f"{URL}#issuecomment-11",
        "at": "2026-11-01T09:00:00Z",
        "body": "TODO: CI summary. Please look?",
    },
    {
        "id": 99,
        "login": "me",
        "type": "User",
        "url": f"{URL}#issuecomment-99",
        "at": "2026-11-01T13:00:00Z",
        "body": f"{LINE1}\n## PR #7 t — \U0001f7e0 a follow-up? see the table",
    },
    {
        "id": 13,
        "login": "rev2",
        "type": "User",
        "url": f"{URL}#issuecomment-13",
        "at": "2026-11-02T10:00:00Z",
        "body": "Can you split the docs change out?",
    },
    {
        "id": 14,
        "login": "dev",
        "type": "User",
        "url": f"{URL}#issuecomment-14",
        "at": "2026-11-02T11:00:00Z",
        "body": "Done, in a separate commit.",
    },
    {
        "id": 15,
        "login": "rev3",
        "type": "User",
        "url": f"{URL}#issuecomment-15",
        "at": "2026-11-03T10:00:00Z",
        "body": "> Done, in a separate commit.\n\nCould you add a test for the empty case?",
    },
    {
        "id": 16,
        "login": "rev3",
        "type": "User",
        "url": f"{URL}#issuecomment-16",
        "at": "2026-11-03T11:00:00Z",
        "body": "LGTM, thanks",
    },
]
REVIEWS = [
    {
        "id": 21,
        "login": "Copilot",
        "type": "Bot",
        "state": "COMMENTED",
        "url": f"{URL}#pullrequestreview-21",
        "at": "2026-11-01T10:06:00Z",
        "body": "## Pull request overview\nLooks fine; one thing for later?",
    },
    {
        "id": 22,
        "login": "rev1",
        "type": "User",
        "state": "APPROVED",
        "url": f"{URL}#pullrequestreview-22",
        "at": "2026-11-04T10:00:00Z",
        "body": "Approving; the TODO on the cache can go in a later PR.",
    },
    {
        "id": 23,
        "login": "rev2",
        "type": "User",
        "state": "COMMENTED",
        "url": f"{URL}#pullrequestreview-23",
        "at": "2026-11-04T11:00:00Z",
        "body": "",
    },
]
COMMITS = [
    {"sha": "a" * 40, "login": "dev", "at": "2026-11-01T09:00:00Z", "msg": "first"},
    {
        "sha": "b" * 40,
        "login": "dev2",
        "at": "2026-11-01T12:30:00Z",
        "msg": "Use int64",
    },
]
POST = (
    f"{LINE1}\n## PR #7 t — \U0001f7e0 one should-fix\n\n| Prior item | Status |\n|---|---|\n"
    "| Our 11-01 N1: old thing | \U0001f7e2 FIXED `abc` |\n"
    "| Our 11-01 N2: label order | \U0001f7e0 OPEN<br>• untouched (read) |\n"
    "| rev9 (11-01) ×2 threads | \U0001f7e0 OPEN<br>• one unanswered |\n"
    "| Bots / CI | \U0001f7e0 Running |\n| Code quality | \U0001f7e0 Should-fix<br>• N3 |\n\n"
    "| # | Sev | file:line | Finding |\n|---|---|---|---|\n"
    f"| N3 | \U0001f7e0 | `platform/b.py:5` | **Empty input crashes** see {TK}-2000 |\n"
    "| N4 | \U0001f7e1 | `platform/c.py:1` | **Nit** |\n"
)


class Fake:
    """GitHub for PR 7: review threads in two GraphQL pages, then the REST lists and our post."""

    def __init__(self, threads=None, fail=False, comments=None, reviews=None):
        self.threads, self.fail, self.calls = (
            THREADS if threads is None else threads,
            fail,
            [],
        )
        self.comments, self.reviews = COMMENTS if comments is None else comments, (
            REVIEWS if reviews is None else reviews
        )

    def graphql(self, query, timeout=120, attempts=3, **v):
        self.calls.append(v)
        if self.fail:
            raise gh.GhError("gh api graphql: gh: HTTP 502")
        first = v.get("after") is None
        return {
            "repository": {
                "pullRequest": {
                    "title": "t7",
                    "mergedAt": "2026-11-05T10:00:00Z",
                    "author": {"login": "dev"},
                    "mergedBy": {"login": "boss"},
                    "reviewThreads": {
                        "pageInfo": {
                            "hasNextPage": first,
                            "endCursor": "c1" if first else None,
                        },
                        "nodes": self.threads[:3] if first else self.threads[3:],
                    },
                }
            }
        }

    def rest(self, path, jq=None, paginate=False, timeout=120):
        return {
            "repos/ROCm/r/issues/7/comments?per_page=100": self.comments,
            "repos/ROCm/r/pulls/7/reviews?per_page=100": self.reviews,
            "repos/ROCm/r/pulls/7/commits?per_page=100": COMMITS,
            "repos/ROCm/r/issues/comments/99": POST,
        }[path]

    def patch(self):
        return mock.patch.object(gh, "graphql", self.graphql), mock.patch.object(
            gh, "rest", self.rest
        )


class TestFollowupScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.tmp.name) / "state")
        self.st = empty_state()
        self.st["prs"] = {
            "7": {"n": 7, "status": "merged", "ours": {"url": f"{URL}#issuecomment-99"}}
        }

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, cfg=CFG, fake=None):
        fake = fake or Fake()
        a, b = fake.patch()
        with a, b:
            data = followups.read(cfg, 7)
            rows = followups.post.open_rows(
                followups.post.our_post_body(cfg, self.st["prs"]["7"]["ours"]),
                prior=True,
            )
        cands, counts = followups.candidates(cfg, data, rows, f"{URL}#issuecomment-99")
        return (
            {c["url"] if c["kind"] != "ours" else c["quotes"][0][0]: c for c in cands},
            counts,
            fake,
        )

    def test_a_resolved_thread_that_defers_is_found_with_its_ticket(self):
        found, _, fake = self.scan()
        self.assertEqual(
            [v.get("after") for v in fake.calls], [None, "c1"]
        )  # every page of threads
        d = found[f"{URL}#discussion_r202"]
        self.assertEqual(
            (
                d["kind"],
                d["reviewer"],
                d["by"],
                d["resolved"],
                d["tickets"],
                d["where"],
            ),
            ("deferred", "rev1", "dev", True, [f"{TK}-1234"], "platform/a.py:12"),
        )
        self.assertEqual(d["quotes"][0], ("ask", "Please add a bounds check here."))
        r = found[
            f"{URL}#pullrequestreview-22"
        ]  # an approval that defers a TODO to a later PR
        self.assertEqual((r["kind"], r["reviewer"]), ("deferred", "rev1"))

    def test_an_unanswered_resolved_thread_is_found(self):
        found, _, _ = self.scan()
        u = found[f"{URL}#discussion_r203"]  # Copilot's "used later" is no deferral
        self.assertEqual(
            (u["kind"], u["reviewer"], u["resolved"], u["resolved_by"], u["where"]),
            ("unanswered", BOTS[0], True, "dev", "platform/b.py:40"),
        )
        x = found[
            f"{URL}#issuecomment-15"
        ]  # no reply after it; quoted in its own words, not the line it quotes
        self.assertEqual(
            (x["kind"], x["quotes"]),
            ("unanswered", [("ask", "Could you add a test for the empty case?")]),
        )

    def test_answered_or_fixed_feedback_is_not_found(self):
        found, _, _ = self.scan()
        for url in (
            f"{URL}#discussion_r204",  # a co-author (a commit's author) answered it
            f"{URL}#discussion_r206",  # a later commit changed its code (outdated)
            f"{URL}#issuecomment-13",  # the author answered later
            f"{URL}#issuecomment-14",
            f"{URL}#issuecomment-16",
        ):  # the author's own; no request
            self.assertNotIn(url, found)

    def test_bots_are_skipped_but_copilot_threads_count(self):
        found, counts, _ = self.scan()
        for url in (
            f"{URL}#discussion_r207",
            f"{URL}#issuecomment-11",
            f"{URL}#pullrequestreview-21",
            f"{URL}#issuecomment-99",
        ):  # a bot thread, a bot comment, Copilot's overview, our own post
            self.assertNotIn(url, found)
        self.assertIn(f"{URL}#discussion_r203", found)
        self.assertEqual(counts["bots"], 3)

    def test_our_open_rows_come_from_our_latest_post(self):
        found, _, _ = self.scan()
        ours = {k: v for k, v in found.items() if v["kind"] == "ours"}
        self.assertEqual(
            sorted(ours), ["N3", "Our 11-01 N2: label order", "rev9 (11-01) ×2 threads"]
        )
        self.assertEqual(
            (ours["N3"]["where"], ours["N3"]["tickets"]),
            ("platform/b.py:5", [f"{TK}-2000"]),
        )
        self.assertEqual(
            ours["Our 11-01 N2: label order"]["quotes"][0][1],
            "\U0001f7e0 OPEN • untouched (read)",
        )

    def test_the_jira_projects_tickets_defer_from_config(self):
        # The package: the JIRA project's key comes from config (jira.project, tracker_keys), never from the code; a
        # blank followup_scan key keeps the built-in list.
        with_project = dict(
            CFG,
            jira={"project": TK},
            followup_scan={"defer_patterns": [], "ticket_pattern": ""},
        )
        defer, _requests, ticket = followups._patterns(with_project)
        self.assertTrue(any(p.search(f"see {TK}-77") for p in defer))
        self.assertTrue(any(p.search("a follow-up PR") for p in defer))
        self.assertEqual(
            ticket.pattern, registry.default("followup_scan.ticket_pattern")
        )
        defer, _requests, _ticket = followups._patterns(CFG)
        self.assertFalse(any(p.search(f"see {TK}-77") for p in defer))

    def test_the_patterns_are_the_package_s_while_empty(self):
        """The deferring, request and ticket patterns are config followup_scan's, each the package's
        (config.defaults.json) while it is empty; a ticket of the JIRA project defers beside any list.
        """
        defer, requests, ticket = followups._patterns({})
        self.assertEqual(
            [p.pattern for p in defer],
            registry.default("followup_scan.defer_patterns"),
        )
        self.assertEqual(
            [p.pattern for p in requests],
            registry.default("followup_scan.request_patterns"),
        )
        self.assertTrue(any(p.search("a follow-up PR") for p in defer))
        self.assertTrue(any(p.search("Could you rename it?") for p in requests))
        self.assertEqual(
            ticket.pattern, registry.default("followup_scan.ticket_pattern")
        )
        own = dict(
            CFG,
            jira={"project": TK},
            followup_scan={"defer_patterns": [r"\bpunt\b"]},
        )
        defer, _requests, _ticket = followups._patterns(own)
        self.assertEqual(len(defer), 2)
        self.assertTrue(any(p.search(f"see {TK}-77") for p in defer))
        self.assertFalse(any(p.search("a follow-up PR") for p in defer))

    def test_a_reviewers_own_words_are_no_deferral(self):
        """A thread's ask naming a ticket and long reviews naming a separate PR are no deferrals.
        Only a reply, a PR owner's comment or an approval defers; a reviewer's own words are requests.
        """
        threads = THREADS + [
            thread(
                "platform/f.py",
                2,
                [
                    c(
                        210,
                        "rev4",
                        "1T14:00:00",
                        "Should this TODO name its ticket? It breaks later.",
                    )
                ],
            )
        ]
        reviews = REVIEWS + [
            {
                "id": 24,
                "login": "rev4",
                "type": "User",
                "state": "COMMENTED",
                "url": f"{URL}#pullrequestreview-24",
                "at": "2026-11-04T12:00:00Z",
                "body": "Split the staging changes into a separate PR; the follow-up commit only renames.",
            }
        ]
        comments = COMMENTS + [
            {
                "id": 17,
                "login": "dev",
                "type": "User",
                "url": f"{URL}#issuecomment-17",
                "at": "2026-11-04T09:00:00Z",
                "body": "@reviewer-4 the narrow follow-ups are pushed in abc1234.",
            },
            {
                "id": 18,
                "login": "dev",
                "type": "User",
                "url": f"{URL}#issuecomment-18",
                "at": "2026-11-04T09:30:00Z",
                "body": "The cache cleanup goes in a follow-up PR (@reviewer-3).",
            },
            {
                "id": 19,
                "login": "me",
                "type": "User",
                "url": f"{URL}#issuecomment-19",
                "at": "2026-11-05T12:00:00Z",
                "body": f"## PR #7 — t — \U0001f7e0 should N1 go later? ({TK}-2000)",
            },
        ]  # our early post, no line 1
        found, _, _ = self.scan(fake=Fake(threads, comments=comments, reviews=reviews))
        self.assertNotIn(f"{URL}#issuecomment-19", found)
        self.assertEqual(
            found[f"{URL}#discussion_r210"]["kind"], "unanswered"
        )  # resolved by dev without a word
        self.assertEqual(found[f"{URL}#pullrequestreview-24"]["kind"], "unanswered")
        self.assertNotIn(
            f"{URL}#issuecomment-17", found
        )  # fixes pushed, nothing deferred
        d = found[f"{URL}#issuecomment-18"]
        self.assertEqual(
            (d["kind"], d["reviewer"], d["by"]), ("deferred", "reviewer-3", "dev")
        )

    def test_the_patterns_come_from_config(self):
        cfg = dict(CFG, followup_scan={"defer_patterns": [r"\bpunt\b"]})
        threads = THREADS + [
            thread(
                "platform/f.py",
                2,
                [
                    c(208, "rev4", "1T14:00:00", "Is this needed?"),
                    c(209, "dev", "1T15:00:00", "Let's punt it."),
                ],
                resolved=False,
            )
        ]
        found, _, _ = self.scan(cfg, Fake(threads))
        self.assertEqual(found[f"{URL}#discussion_r209"]["kind"], "deferred")
        self.assertNotIn(
            f"{URL}#discussion_r202", found
        )  # "follow-up" is no longer a deferral: the author answered
        self.assertNotIn(f"{URL}#pullrequestreview-22", found)

    def test_the_tick_scans_each_merge_once(self):
        fake, ev = Fake(), [{"kind": "MERGED", "pr": 7, "by": "boss"}]
        scans = pathlib.Path(self.tmp.name) / "scans"
        cfg = dict(CFG, followup_scan={"dir": str(scans)})
        a, b = fake.patch()
        with a, b:
            lines = followups.tick(cfg, self.st, self.store, ev)
            path = scans / "followups_7.md"
            self.assertEqual(
                lines,
                [
                    f"FOLLOWUP_SCAN #7: 7 candidates (2 deferred, 2 unanswered, 3 ours): {path}"
                ],
            )
            self.assertEqual(
                followups.tick(cfg, self.st, self.store, ev), []
            )  # the same merge: no second scan
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(self.st["followup_scans"]["7"]["file"], str(path))
        text = path.read_text(encoding="utf-8")
        for want in (
            "## Deferred (2)",
            "## Unanswered (2)",
            "## Ours (3)",
            f"ticket {TK}-1234",
            "resolved yes by dev",
            "`platform/a.py:12`",
            "> reply: Good catch.",
        ):
            self.assertIn(want, text)

    def test_a_scan_writes_only_in_the_scans_folder_the_configuration_names(self):
        a, b = Fake().patch()
        with a, b:
            lines = followups.tick(
                CFG, self.st, self.store, [{"kind": "MERGED", "pr": 7}]
            )
        self.assertEqual(
            lines,
            [
                "(followup scan #7 failed, try 1 of 3: followup_scan.dir is blank: the configuration has no "
                "folder for the scans' files)"
            ],
        )
        self.assertEqual(list(self.store.dir.rglob("followups_*.md")), [])

    def test_a_failed_scan_is_retried_then_reported(self):
        a, b = Fake(fail=True).patch()
        with a, b:
            first = followups.tick(
                CFG, self.st, self.store, [{"kind": "MERGED", "pr": 8}]
            )
            later = [followups.tick(CFG, self.st, self.store, []) for _ in range(3)]
        self.assertTrue(
            first[0].startswith("(followup scan #8 failed, try 1 of 3: gh api graphql"),
            first,
        )
        self.assertTrue(
            later[0][0].startswith("(followup scan #8 failed, try 2 of 3"), later
        )
        self.assertTrue(
            later[1][0].startswith("!! FOLLOWUP_SCAN #8 failed 3 times"), later
        )
        self.assertEqual(later[2], [])

    def test_the_command_writes_only_its_file(self):
        out, printed = pathlib.Path(self.tmp.name) / "f.md", []
        a, b = Fake().patch()
        with a, b, mock.patch.object(
            rkmon, "load_config", lambda: CFG
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *x, **kw: printed.append(" ".join(map(str, x)))
        ):
            self.assertEqual(rkmon.main(["followup-scan", "7", "--out", str(out)]), 0)
        self.assertEqual(
            printed,
            [
                f"FOLLOWUP_SCAN #7: 7 candidates (2 deferred, 2 unanswered, 3 ours): {out}"
            ],
        )
        self.assertFalse(
            self.store.state_path.exists()
        )  # an untracked PR: ours read from GitHub, no state written


class TestDeletedPost(unittest.TestCase):
    def test_a_deleted_review_comment_reads_as_no_rows(self):
        def gone(*a, **k):
            raise gh.GhError("gh: Not Found (HTTP 404)")

        with mock.patch.object(gh, "rest", gone):
            self.assertEqual(
                followups.post.our_post_body(CFG, {"url": f"{URL}#issuecomment-99"}), ""
            )

    def test_other_errors_still_raise(self):
        def down(*a, **k):
            raise gh.GhError("gh: Bad Gateway (HTTP 502)")

        with mock.patch.object(gh, "rest", down), self.assertRaises(gh.GhError):
            followups.post.our_post_body(CFG, {"url": f"{URL}#issuecomment-99"})


class TestDeferWording(unittest.TestCase):
    def test_thanks_for_the_follow_up_is_no_deferral(self):
        """The author's "Thanks for the follow-up. N1: fixed in ..." replies are no deferrals."""
        pats = followups._patterns({"jira": {"project": "PROJ"}})[
            0
        ]  # the JIRA project's tickets defer too
        for text in (
            "Thanks for the follow-up. N1: fixed in the PR body.",
            "thank you for your follow up review",
            "Answered in the follow-up review below.",
        ):
            self.assertFalse(followups.defers_text(text, pats), text)
        for text in (
            "Thanks for the follow-up; the rest goes in a follow-up PR.",
            "I'll fix this later.",
            "Tracked in PROJ-" + "1234.",
        ):
            self.assertTrue(followups.defers_text(text, pats), text)

    def test_a_fix_reply_named_follow_up_is_no_deferral(self):
        """The author's "Follow-up on the review of 11-03: ..." replies and a review's "author follow-up:" link are
        no deferrals."""
        pats = followups._patterns({})[0]
        for text in (
            "Follow-up on the open review items, pushed as 6be5c4b9180 and 9c63579645d:",
            "Follow-up on the review of 11-03: N4 fixed.",
            "Follow-up to the Copilot review of 11-04 and our automated review of 11-04, in eight commits:",
            "(the current test includes gfx942; author follow-up: the reply above)",
            "@dev-a the narrow follow-ups are pushed in 9ebbdab25ed.",
            "@dev-a — follow-up cleanup is pushed in fce52078fa8.",
            "Follow-up to [the November 5 review](https://example.com/7#issuecomment-1): N2 fixed.",
            "The layout pin now requires ABI 11 or later.",
        ):
            self.assertFalse(followups.defers_text(text, pats), text)
        for text in (
            "Follow-up on the review of 11-03: N5 goes to a separate PR.",
            "Follow-up on the review: the rest is a follow-up.",
        ):
            self.assertTrue(followups.defers_text(text, pats), text)


if __name__ == "__main__":
    unittest.main()
