# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The HTML dashboard renders from real state shapes, escapes untrusted titles, and draws both charts."""

import datetime as dt
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import guard_labels  # noqa: E402

from rkmon_lib import daylog, guard, jobs, post, report, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402


def ago(minutes: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class TestWebdash(unittest.TestCase):
    def test_render(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {
                "features": {
                    "merge_queue": {"mode": "live", "auto_merge": "enabled"},
                    "ticket_creation": {
                        "enabled": True,
                        "triggers": {
                            "review_follow_up": True,
                            "out_of_scope_issue": True,
                        },
                    },
                },
                "operator": {"github_login": "me"},
                "repo": "ROCm/r",
                "codeowner_teams": {"platform": ["plat"]},
                "paths": {
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            }
            st = empty_state()
            st["enabled"] = True
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {
                "7": {
                    "n": 7,
                    "head": "7" * 40,
                    "status": "open",
                    "draft": False,
                    "head_seen_at": ago(90),
                    "title": "<script>alert(1)</script>",
                    "author": "eve",
                }
            }
            store.append_jsonl(
                "metrics.jsonl",
                [
                    {"ts": ago(30), "review_backlog": 5},
                    {"ts": now(), "review_backlog": 4},
                ],
            )
            store.append_jsonl(
                "events.jsonl", [{"kind": "NEW_PR", "pr": 7, "ts": now()}]
            )
            store.ledger_add("post", 7, "7" * 40, "https://github.com/ROCm/r/pull/7#c")
            plan = [
                {
                    "prio": 3,
                    "kind": "FIRST_ROUND",
                    "pr": 7,
                    "text": "",
                    "critical": False,
                }
            ]
            board = report.board(cfg, st, store, plan, [])
            page = pathlib.Path(
                webdash.write(cfg, st, store, plan, [], board)
            ).read_text(encoding="utf-8")
            published = (store.dir / webdash.PUBLISHED_FILE).read_text(encoding="utf-8")
        self.assertTrue(page.startswith("<!doctype html>"))
        # The models' tokens in the insights, on the local page only; tokens, never a cost
        self.assertIn("aria-label='Token usage'", page)
        self.assertIn("Claude tokens a review (7 d)", page)
        self.assertIn("Reviews posted a day (7 d)</div><div class='value'>0.1<", page)
        self.assertNotIn("Token usage", published)
        self.assertNotIn(
            "cost",
            page.split("aria-label='Token usage'", 1)[1].split("</section>", 1)[0],
        )
        # The same sections, but for the admin dashboard's Tickets card (found-issue-tickets is on), its "Not sent" card
        # (review automation is in prototype) and its own cards (the scoreboard, review spend, today's log).
        self.assertEqual(
            page.count("<section"),
            published.count("<section") + 2 + len(webdash.ADMIN_CARDS),
        )
        self.assertIn("<section id='tickets'", page)
        self.assertIn("<section id='not-sent'", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("aria-label='Review backlog'", page)
        self.assertIn("aria-label='Cleared vs arrived per day'", page)
        self.assertIn("Nothing needs you right now", page)
        # rocKE merges per day (tile and chart), and "Cleared today" collapsed by default
        self.assertIn("aria-label='rocKE PRs merged per day'", page)
        self.assertIn("Merged today (rocKE)", page)
        self.assertIn("<details id='d-cleared'><summary>", page)
        self.assertNotIn("d-cleared' open", page)
        # One approval box per code-owner team (none ready here), and no "Close to merge"
        self.assertIn("<h2>Needs plat approval <span class='c'>", page)
        self.assertIn("<p class='sub'>None right now.</p>", page)
        self.assertNotIn("Close to merge", page)

    def test_the_reviewer_model_is_named_as_configured(self):
        """The board and both dashboards name the configured reviewer model (its display name), or "the
        reviewer model" while none is set, and never a model the package assumes."""
        gateway = {
            "backend": "openai",
            "url": "https://gw.example.com/v1",
            "model": "example-flash-deploy-1",
            "model_name": "Example-5.3-Flash",
        }
        for worker, name in (
            (gateway, "Example-5.3-Flash"),
            (None, "the reviewer model"),
            (
                {
                    "backend": "ollama",
                    "url": "http://127.0.0.1:9",
                    "model": "example-coder:30b-q4",
                },
                "example-coder",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                store = Store(pathlib.Path(d))
                cfg = {
                    "operator": {"github_login": "me"},
                    "repo": "ROCm/r",
                    "paths": {
                        "working_dir": d,
                        "html": str(pathlib.Path(d) / "x.html"),
                        "daily_dir": d,
                    },
                }
                if worker:
                    cfg["worker"] = worker
                st = empty_state()
                st["develop"]["sha"] = "d" * 40
                st["prs"] = {
                    "7": {
                        "n": 7,
                        "head": "7" * 40,
                        "status": "open",
                        "draft": False,
                        "head_seen_at": ago(30),
                        "title": "a change",
                        "author": "dev-a",
                    }
                }
                daylog.save(store, cfg, 7, "verify", "fu: 3 claims corrected")
                plan = [
                    {
                        "prio": 3,
                        "kind": "FIRST_ROUND",
                        "pr": 7,
                        "text": "",
                        "critical": False,
                    }
                ]
                board = report.board(cfg, st, store, plan, [])
                page = pathlib.Path(
                    webdash.write(cfg, st, store, plan, [], board)
                ).read_text(encoding="utf-8")
                published = (store.dir / webdash.PUBLISHED_FILE).read_text(
                    encoding="utf-8"
                )
                for text in (board["md"], page, published):
                    self.assertNotIn("qwen", text.lower())
                # The In review box names the parts the pipeline runs, not a model (report.in_review_hint).
                self.assertIn(webdash.E(report.in_review_hint(cfg)), page)
                self.assertNotIn(f"{name} drafting", page)
                self.assertIn(f"{name}'s claims corrected 1", board["md"])
                self.assertIn(
                    webdash.E(f"{name}'s claims corrected 1"), page
                )  # the admin page's Issues avoided tile
                self.assertNotIn("Issues avoided", published)

    def test_merged_today_agrees_everywhere_and_counts_develop_only(self):
        # The "Merged today" stage, the tile and the chart agree (PRs merged into a stack branch are not develop's)
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = {
                "features": {"merge_queue": {"mode": "live", "auto_merge": "enabled"}},
                "operator": {"github_login": "me"},
                "repo": "ROCm/r",
                "paths": {
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            }
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {
                "5": {"n": 5, "status": "merged", "base": "develop", "title": "lands"},
                "6": {
                    "n": 6,
                    "status": "merged",
                    "base": "users/x/stack",
                    "title": "rides the stack",
                },
            }
            daylog.from_events(
                store,
                cfg,
                st,
                [{"kind": "MERGED", "pr": 5}, {"kind": "MERGED", "pr": 6}],
            )
            daylog.add(
                store,
                cfg,
                "merged",
                6,
                "an entry logged before the stacked kind existed",
            )
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn("id='st-merged'><div class='n'>1</div>", page)
        self.assertIn("Merged today (rocKE)</div><div class='value'>1<", page)


class TestChipBox(unittest.TestCase):
    def test_priority_chips_keep_the_size_of_their_neighbours(self):
        """Priority chips keep the size of their neighbours: the chip's class is `chip-prio`, never "prio" (the
        priority form's `.prio{margin;padding;border-radius}` rule), its ring an inset shadow (no size change),
        and chip rows centre their items."""
        css = pathlib.Path(webdash.__file__).read_text()
        self.assertIn(
            ".chip.chip-prio{border-color:var(--good);box-shadow:inset 0 0 0 1px var(--good)}",
            css,
        )
        self.assertNotRegex(css, r"\.chip\.chip-prio\{[^}]*(border:|padding|margin)")
        self.assertRegex(css, r"\.chips\{[^}]*align-items:center")
        st = empty_state()
        st["user_priority"] = {"7": {"why": "x"}}
        chip = webdash._chip({"repo": "o/r"}, st, 7)
        self.assertIn("class='chip chip-prio'", chip)
        self.assertNotIn(
            "'chip prio'", chip
        )  # a bare "prio" class picks up the form's box rule


def _jobs(states: dict):
    """jobs.status by job id: a fixture's jobs, no processes."""
    return mock.patch.object(
        jobs, "status", lambda store, job: states.get((job or {}).get("id"), "none")
    )


def _cfg(d: str, **kw) -> dict:
    return dict(
        {
            "features": {"merge_queue": {"mode": "live", "auto_merge": "enabled"}},
            "operator": {"github_login": "me"},
            "repo": "o/r",
            "settle_min": 30,
            "reviewer": "default",
            "codeowner_teams": {"platform": ["plat"]},
            "paths": {
                "html": str(pathlib.Path(d) / "x.html"),
                "daily_dir": d,
                "working_dir": d,
                "root": d,
            },
        },
        **kw,
    )


def _open(n: int, **kw) -> dict:
    return dict(
        {
            "n": n,
            "head": str(n % 10) * 40,
            "status": "open",
            "draft": False,
            "head_seen_at": ago(600),
            "title": f"t{n}",
            "author": "a",
            "ci_rollup": "SUCCESS",
            "reviews": [],
            "labels": [],
        },
        **kw,
    )


def _section(page: str, sid: str) -> str:
    at = page.index(f"<section id='{sid}'")
    return page[at : page.index("</section>", at)]


def _stage(page: str, sid: str) -> str:
    at = page.index(f"<div class='stage' id='st-{sid}'>")
    nxt = page.find("<div class='stage'", at + 1)
    return page[at : nxt if nxt >= 0 else None]


class TestInReview(unittest.TestCase):
    """The In review box counts the work in flight, each review by its part, the same rows as `rk review-status`
    (report.in_review), and names no model: with no first-pass drafter the verifier reviews alone.
    """

    def page(self, st: dict, store, cfg: dict, plan=(), q=()):
        board = report.board(cfg, st, store, list(plan), list(q))
        return webdash.render(
            cfg, st, store, list(plan), list(q), board, webdash.PUBLISHED
        )

    def test_no_draft_reviews_in_flight_are_counted_by_part(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = _cfg(d)
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {str(n): _open(n) for n in (11, 12, 13, 14)}
            st["review_pipes"] = [
                {
                    "pr": 11,
                    "kind": "follow-up",
                    "head": st["prs"]["11"]["head"],
                    "since": ago(60),
                    "started": ago(30),
                    "stage": "worker",
                    "stage_name": "none",
                    "worker": {"id": "w11", "started": ago(30)},
                    "checks": {"id": "c11"},
                }
            ]
            st["review_ready"] = [
                {
                    "pr": 12,
                    "kind": "follow-up",
                    "head": st["prs"]["12"]["head"],
                    "at": ago(20),
                },
                {
                    "pr": 13,
                    "kind": "first-round",
                    "head": st["prs"]["13"]["head"],
                    "at": ago(20),
                },
            ]
            # `rk review-queue 14 --again`: it may start now
            st["review_first"] = [{"pr": 14, "again": True}]
            # rk dispatch-pack ran: #13 is being verified
            facts = pathlib.Path(d) / "tmp" / "v13fr" / "facts.txt"
            facts.parent.mkdir(parents=True)
            facts.write_text("facts\n")
            with _jobs({"w11": "done", "c11": "running"}):
                rows = report.in_review(cfg, st, store)
                page = self.page(st, store, cfg)
        self.assertEqual(
            [(r["pr"], r["part"]) for r in rows],
            [
                (14, "queued to start"),
                (11, "checks running"),
                (12, "awaiting verification"),
                (13, "verifying"),
            ],
        )
        box = _stage(page, "inreview")
        self.assertIn("<div class='n'>4</div><div class='lbl'>In review</div>", box)
        self.assertIn("staging, then checks, then my verification", box)
        self.assertIn(
            "1 queued to start · 1 checks running · 1 awaiting verification · 1 verifying",
            box,
        )
        self.assertIn("#11 checks running", box)
        flow = _section(page, "flow")
        for gone in ("qwen", "drafting", "waiting for a draft"):
            self.assertNotIn(gone, flow)
        reviews = page[page.index("<section id='reviews'") :]
        self.assertIn("follow-up: checks running", reviews[:3000])

    def test_a_review_kept_unposted_or_skipped_waits_on_no_verifier(self):
        """Neither a verified review kept unposted (posting off: post.save_draft finishes it) nor one of a skipped PR
        waits on a verifier (review.waiting_on_verifier): the In review box, the board and the Reviews card show
        neither as verifying or waiting, though each one's review directory got its facts after the hand-off.
        """
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = _cfg(d)
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["prs"] = {str(n): _open(n) for n in (12, 13, 16)}
            st["review_ready"] = [
                {
                    "pr": n,
                    "kind": "first-round",
                    "head": st["prs"][str(n)]["head"],
                    "at": ago(20),
                }
                for n in (12, 13, 16)
            ]
            # Posting off: #12's verified review is kept unposted.
            verified = pathlib.Path(d) / "pr12_verified.md"
            verified.write_text("the verified review\n", encoding="utf-8")
            post.save_draft(cfg, st, store, 12, str(verified))
            st["review_skip"] = [13]  # rk review-skip 13
            for n in (12, 13, 16):  # rk dispatch-pack ran for each
                facts = pathlib.Path(d) / "tmp" / f"v{n}fr" / "facts.txt"
                facts.parent.mkdir(parents=True)
                facts.write_text("facts\n")
            rows = report.in_review(cfg, st, store)
            board = report.board(cfg, st, store, [], [])
            page = self.page(st, store, cfg)
        self.assertEqual(rows, [{"pr": 16, "part": "verifying"}])
        (line,) = [x for x in board["md"].splitlines() if "my verification" in x]
        self.assertIn("handed to my verification: #16 (verifying, ", line)
        reviews = _section(page, "reviews")
        self.assertIn(" gap after our last review: ", reviews)
        handed = reviews[reviews.index("Handed to my verification:") :]
        handed = handed[: handed.index("<br>")]
        self.assertIn("#16", handed)
        for text in (line, handed):
            self.assertNotIn("#12", text)
            self.assertNotIn("#13", text)

    def test_an_interrupted_review_counts_as_queued_to_start(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {"15": _open(15)}
            st["review_interrupted"] = {
                "15": {
                    "pr": 15,
                    "kind": "first-round",
                    "head": "5" * 40,
                    "again": False,
                    "at": ago(5),
                }
            }
            rows = report.in_review(_cfg(d), st, store)
        self.assertEqual(rows, [{"pr": 15, "part": "queued to start"}])

    def test_the_parts_of_the_review_in_flight(self):
        pipe = {
            "pr": 1,
            "stage": "worker",
            "stage_name": "none",
            "worker": {"id": "w"},
            "checks": None,
        }
        for states, checks, part in (
            ({"w": "running"}, None, "staging"),
            ({"w": "done"}, None, "checks queued"),
            ({"w": "done", "c": "running"}, {"id": "c"}, "checks running"),
            ({"w": "done", "c": "done"}, {"id": "c"}, "checks done"),
        ):
            with _jobs(states):
                self.assertEqual(
                    report.pipe_part(dict(pipe, checks=checks), None), part
                )
        # A drafter stage shows only when a drafter drafts: the tool-loop or the local worker.
        with _jobs({"w": "running", "c": "running"}):
            self.assertEqual(
                report.pipe_part(
                    dict(pipe, stage_name="tool-loop", checks={"id": "c"}), None
                ),
                "drafting",
            )
            self.assertEqual(
                report.pipe_part(dict(pipe, checks={"id": "c"}), None),
                "checks running",
            )
        self.assertEqual(
            report.in_review_hint({"reviewer": "default"}),
            "staging, then checks, then my verification",
        )
        self.assertEqual(
            report.in_review_hint({"reviewer": "worker"}),
            "drafting and checks, then my verification",
        )
        self.assertEqual(report.in_review_counts([]), "none")


class TestMergeableCandidates(unittest.TestCase):
    """A Mergeable Candidate has passed CI and can merge now (the ready tier). An approved, clean candidate whose CI
    still runs is only counted ("+N waiting on CI"); one whose CI fails on its own goes to Needs Work; known
    infrastructure or develop failing too counts as waiting on CI."""

    def render(self, prs: dict, q: list, develop_ci: dict | None = None):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = _cfg(d)
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            if develop_ci:
                st["develop"]["ci"] = {"d" * 40: develop_ci}
            st["prs"] = prs
            board = report.board(cfg, st, store, [], q)
            return st, webdash.render(cfg, st, store, [], q, board, webdash.PUBLISHED)

    @staticmethod
    def ours(n: int) -> dict:
        return {
            "verdict": "\U0001f7e2",
            "head": str(n % 10) * 40,
            "at": now(),
            "url": "u",
        }

    def test_ci_green_mergeable_ci_running_counted_own_failure_needs_work(self):
        gs = {k: "pass" for k in report.GATES}
        prs = {
            "21": _open(21, ours=self.ours(21)),
            "22": _open(22, ours=self.ours(22), ci_rollup="PENDING"),
            "23": _open(23, ours=self.ours(23), ci_rollup="FAILURE"),
        }
        q = [
            {"n": 21, "position": 1, "stage": "ready", "blockers": [], "gs": gs},
            {
                "n": 22,
                "stage": "candidate",
                "blockers": [],
                "gs": dict(gs, G3="pending"),
            },
            {
                "n": 23,
                "stage": "candidate",
                "blockers": ["G3: CI FAILURE"],
                "gs": dict(gs, G3="fail"),
            },
        ]
        st, page = self.render(prs, q)
        box = _stage(page, "candidate")
        self.assertIn(
            "<div class='n'>1</div><div class='lbl'>Mergeable Candidates</div>", box
        )
        self.assertIn("pull/21", box)
        self.assertNotIn("pull/22", box)  # CI still runs: not mergeable
        self.assertNotIn("pull/23", box)
        self.assertIn("+1 waiting on CI", box)
        self.assertIn("#22 CI running", box)
        # a CI failure of its own: Needs Work
        self.assertIn("pull/23", _stage(page, "author"))
        self.assertIn("<div class='lbl'>Needs Work</div>", page)
        card = _section(page, "merge")
        self.assertIn("<h2>Mergeable Candidates <span class='c'>", card)
        self.assertIn("pull/21", card)
        self.assertNotIn("pull/22", card)
        self.assertIn("Not mergeable yet: +1 waiting on CI (#22 CI running)", card)
        m = report.mergeable(st, q)
        self.assertEqual(
            ([r["n"] for r in m["rows"]], m["ci"], m["work"]),
            ([21], [(22, "CI running")], [23]),
        )

    def test_known_infrastructure_or_develop_failing_too_waits_on_ci(self):
        prs = {
            "31": _open(
                31,
                ours=self.ours(31),
                ci_rollup="FAILURE",
                ci_infra={
                    "head": "1" * 40,
                    "jobs": {"HipDNN Linux Superbuild": "signature"},
                },
            ),
            "32": _open(32, ours=self.ours(32), ci_rollup="FAILURE"),
        }
        st = {"prs": prs, "develop": {"sha": "d" * 40, "ci": {}}}
        self.assertEqual(
            report.ci_cause(st, prs["31"]),
            ("waiting", "known CI infrastructure: HipDNN Linux Superbuild"),
        )
        self.assertEqual(report.ci_cause(st, prs["32"])[0], "pr")
        red = {
            "runs": {
                "Multi-Arch CI": {
                    "status": "completed",
                    "conclusion": "failure",
                    "failed_jobs": ["x"],
                }
            }
        }
        st["develop"]["ci"] = {"d" * 40: red}
        self.assertEqual(
            report.ci_cause(st, prs["32"]),
            ("waiting", "CI FAILURE, and develop's CI fails too (red-other)"),
        )
        # the infrastructure record is for another head
        stale = dict(prs["31"], head="9" * 40)
        self.assertEqual(report.ci_cause({"develop": {}}, stale)[0], "pr")


class TestCardOrder(unittest.TestCase):
    def test_reviews_follow_the_pipeline_and_mergeable_candidates_precede_the_plan(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = _cfg(
                d,
                codeowner_teams={
                    "platform": ["rocke-platform"],
                    "library": ["rocke-lib"],
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            board = report.board(cfg, st, store, [], [])
            pages = [
                webdash.render(
                    cfg,
                    st,
                    store,
                    [],
                    [],
                    board,
                    v,
                )
                for v in (webdash.LOCAL, webdash.PUBLISHED)
            ]
        want = [
            "needs",
            "flow",
            "reviews",
            "approvals",
            "approvals-rocke-lib",
            "approvals-any",
            "merge",
            "plan",
        ]
        for page in pages:
            sections = re.findall(r"<section id='([^']+)'", page)
            # directly after the pipeline
            self.assertEqual(sections[sections.index("flow") + 1], "reviews")
            self.assertEqual([s for s in sections if s in want], want)
            nav = re.findall(
                r"<a href='#([^']+)'>([^<]+)",
                page[page.index("<nav") : page.index("</nav>")],
            )
            # the admin page may show its Tickets card between Needs you and the Pipeline
            self.assertEqual([i for i, _ in nav if i in want], want)
            self.assertEqual(dict(nav)["merge"].strip(), "Mergeable Candidates")
            self.assertEqual(
                re.findall(r"<div class='lbl'>([^<]+)</div>", page),
                [
                    "Review queue",
                    "In review",
                    "Needs Work",
                    "Needs Approval",
                    "Mergeable Candidates",
                    "Merged today",
                ],
            )
            for gone in (
                "Merge pipeline",
                "With the author",
                "Awaiting approval",
                "Ready for ",
            ):
                self.assertNotIn(gone, page)
            # The merge queue is a tile (ready / candidates), never a stage of the Pipeline card.
            self.assertIn("<div class='label'>Merge queue</div>", page)


GREEN = {
    "runs": {"TheRock Multi-Arch CI": {"status": "completed", "conclusion": "success"}},
    "complete": True,
}
RUNNING = {
    "runs": {"TheRock Multi-Arch CI": {"status": "queued", "conclusion": None}},
    "complete": False,
}
ROCKE_RED = {
    "runs": {
        "TheRock Multi-Arch CI": {
            "status": "completed",
            "conclusion": "failure",
            "failed_jobs": ["Test hipkernelprovider"],
            "rocke_failed": ["Test hipkernelprovider"],
        }
    },
    "complete": True,
}
OTHER_RED = {
    "runs": {
        "TheRock Multi-Arch CI": {
            "status": "completed",
            "conclusion": "failure",
            "failed_jobs": ["miopen-dbsync"],
        }
    },
    "complete": True,
}


class TestDevelopStatus(unittest.TestCase):
    """One word from the latest COMPLETED develop state: develop's newest commit's CI nearly always still runs
    (develop moves several times an hour; Multi-Arch CI queues for hours), so the head's own CI is no status.
    """

    NEW, OLD = "b" * 40, "a" * 40

    def status(self, window: dict, incidents=(), issues=(), last_green=None, **kw):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            if incidents:
                guard.save_incidents(
                    store,
                    [
                        dict(
                            {
                                "start": ago(30),
                                "signals": [],
                                "summary": "",
                                "candidates": [],
                            },
                            **x,
                        )
                        for x in incidents
                    ],
                    keep_prev=False,
                )
            st = empty_state()
            st["develop"].update(
                {
                    "sha": next(iter(window)) if window else self.NEW,
                    "ci": window,
                    "last_green": last_green,
                }
            )
            st["post_merge"] = {"at": now(), "open": list(issues)}
            cfg = _cfg(d, **kw)
            s = report.develop_status(cfg, st, store)
            board = report.board(cfg, st, store, [], [])
            page = webdash.render(cfg, st, store, [], [], board, webdash.PUBLISHED)
            md = report.dashboard(cfg, st, [], [], [], board["md"], store=store)
        return s, board["md"], page, md

    def test_healthy_from_the_last_completed_run_with_a_newer_commit_running(self):
        s, board, page, md = self.status({self.NEW: RUNNING, self.OLD: GREEN})
        self.assertEqual(
            (s["word"], s["light"], s["sha"], s["newer"]),
            ("Healthy", "🟢", self.OLD, True),
        )
        self.assertEqual(
            report.develop_status_line(s), "aaaaaaaaaa · newer commit running"
        )
        self.assertIn(
            "<div class='label'>Develop</div><div class='value'>🟢 Healthy</div>", page
        )
        self.assertIn("aaaaaaaaaa · newer commit running", page)
        self.assertIn(
            "| Develop guard | 🟢 Healthy: aaaaaaaaaa · newer commit running", board
        )
        self.assertIn("**Develop: 🟢 Healthy** (aaaaaaaaaa · newer commit running)", md)
        tile = page[page.index("<div class='label'>Develop</div>") :][:300]
        for gone in ("Pending", "pending</div>"):
            self.assertNotIn(gone, tile)
        s, *_ = self.status({self.OLD: GREEN})
        self.assertFalse(s["newer"])  # the verdict is develop's head itself
        self.assertEqual(report.develop_status_line(s), "aaaaaaaaaa")

    def test_ci_failures_outside_rocke(self):
        s, *_ = self.status({self.NEW: RUNNING, self.OLD: OTHER_RED})
        self.assertEqual(
            (s["word"], s["light"], s["level"]), ("CI-failures", "🟡", "warning")
        )

    def test_broken_rocke_jobs_an_incident_or_a_post_merge_issue(self):
        s, *_ = self.status({self.NEW: RUNNING, self.OLD: ROCKE_RED})
        self.assertEqual((s["word"], s["light"]), ("Broken", "🔴"))
        inc = {"id": "INC-1", "status": "open", "culprit": 11111, "issue": None}
        s, *_ = self.status({self.OLD: GREEN}, incidents=[inc])
        self.assertEqual(s["word"], "Broken")
        # a red sanity run
        sanity = {
            "id": "INC-2",
            "status": "open",
            "culprit": None,
            "signals": ["conv: failed"],
        }
        self.assertEqual(
            self.status({self.OLD: GREEN}, incidents=[sanity])[0]["word"], "Broken"
        )
        issue = {
            "n": 22222,
            "kind": "ci_red",
            "label": guard_labels.RED,
            "rocke": True,
            "labels": ["red-develop", "needs-revert"],
            "title": "t",
            "author": "b",
            "assignees": [],
            "created": now(),
        }
        self.assertEqual(
            self.status({self.OLD: GREEN}, issues=[issue])[0]["word"], "Broken"
        )

    def test_a_ci_time_limit_with_no_culprit_is_a_ci_failure_not_a_break(self):
        """An incident with no culprit whose issue is labelled test-timeout (a CI Test step's time limit)."""
        issue = {
            "n": 33333,
            "kind": "ci_red",
            "label": guard_labels.RED,
            "rocke": True,
            "labels": ["red-develop", "test-timeout", "needs-revert"],
            "title": "t",
            "author": "b",
            "assignees": [],
            "created": now(),
        }
        inc = {
            "id": "INC-2026-01-01-1",
            "status": "open",
            "culprit": None,
            "issue": 33333,
        }
        stable = dict(
            issue,
            n=44444,
            kind="stability",
            label=guard_labels.STABILITY,
            labels=["silent-break"],
        )
        limit = {"time_limit_labels": ["test-timeout"]}  # the settings name the label
        s, board, page, md = self.status(
            {self.NEW: RUNNING, self.OLD: GREEN},
            incidents=[inc],
            issues=[issue, stable],
            incident_issue=dict(guard_labels.INCIDENT_ISSUE, **limit),
        )
        self.assertEqual((s["word"], s["light"]), ("CI-failures", "🟡"))
        self.assertIn(
            "INC-2026-01-01-1: no culprit, its issue #33333 is a CI time limit",
            s["why"],
        )
        self.assertIn("1 silent-break issue open (CI does not see them)", s["why"])
        self.assertIn("🟡 CI-failures", page)
        # the same incident with a culprit is a break
        s, *_ = self.status(
            {self.OLD: GREEN},
            incidents=[dict(inc, culprit=55555)],
            issues=[issue],
            incident_issue=dict(guard_labels.INCIDENT_ISSUE, **limit),
        )
        self.assertEqual(s["word"], "Broken")
        # the labels are a setting: with none, the same issue is a break
        s, *_ = self.status(
            {self.OLD: GREEN},
            incidents=[inc],
            issues=[issue],
            incident_issue=dict(guard_labels.INCIDENT_ISSUE, time_limit_labels=[]),
        )
        self.assertEqual(s["word"], "Broken")

    def test_the_time_limit_labels_are_a_setting(self):
        self.assertEqual(
            report.time_limit_labels({}), set()
        )  # none configured: no label says a time limit
        self.assertEqual(
            report.time_limit_labels(
                {
                    "incident_issue": dict(
                        guard_labels.INCIDENT_ISSUE, time_limit_labels=["ci-time-limit"]
                    )
                }
            ),
            {"ci-time-limit"},
        )

    def test_unknown_only_with_no_completed_run(self):
        s, board, page, md = self.status({self.NEW: RUNNING})
        self.assertEqual((s["word"], s["light"], s["sha"]), ("Unknown", "⚪", None))
        self.assertEqual(
            report.develop_status_line(s), "no completed develop CI run yet"
        )
        # the last green stands in when no run in the window has completed
        s, *_ = self.status(
            {self.NEW: RUNNING},
            last_green={"sha": self.OLD, "at": now(), "how": "ci"},
        )
        self.assertEqual((s["word"], s["sha"], s["newer"]), ("Healthy", self.OLD, True))


if __name__ == "__main__":
    unittest.main()
