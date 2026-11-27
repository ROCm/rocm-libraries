# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Several deployments of one repository under one GitHub account, against a fake GitHub.

Line 1 of each review gives its deployment's automated reviewer name, and each deployment takes only its own reviews
for its own: the latest review its follow-ups answer, and the reviews its weekly report counts. The moved deployment
also takes the reviews an older monitor posted with no name before its time (reviews.unnamed_are_ours_until), and a
mirror takes each of them, whenever it was posted; after a change of its name, its earlier reviews stay its own
(reviews.automated_reviewer_names_used). The readers that leave
out the account's posts by the account alone take another deployment's review for no request among the follow-up
candidates and for no activity since their own review. `rk post` refuses another line 1. G1, the review gate, reads a
deployment's own review where it reviews, and people's approvals alone where it does not. A review of the account at
the head of a deployment's review that stands for its own (its line 1 gives no name: an older monitor's posted after
the move, or one posted by hand) closes that review without a post (`rk review-close`), and is its last review from
then on; another deployment's review never does, and a mirror closes none. `rk facts` lists the account's reviews it
did not post: the head of each that stands for its own read from GitHub alone, another deployment's marked not ours to
take, and in a mirror none posted after the feed line. The deploy guide, the verifier's team file, the review procedure
and the skill say the same.
"""

import contextlib
import datetime as dt
import fcntl
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import (
    bindings,
    config,
    facts,
    features,
    followups,
    gh,
    mirror,
    plan,
    post,
    prep,
    queue,
    reviewer_name,
    snapshot,
    stats,
)  # noqa: E402

# The review module under another name: review() below writes a review's body.
from rkmon_lib import review as review_mod  # noqa: E402
from rkmon_lib.review import activity_since_ours  # noqa: E402
from rkmon_lib.store import Store, empty_state, parse_ts  # noqa: E402

GREEN, ORANGE, RED, YELLOW = "\U0001f7e2", "\U0001f7e0", "\U0001f534", "\U0001f7e1"
REPO = "o/r"
ACCOUNT = "review-account"  # the one account every deployment here posts under
# An older monitor's line 1, in the form its reviews give it today (reviewer_name.given): it gives no name.
OLDER_LINE = "Automated review: Model - Claude Opus 5.5, effort: high"
# The moved deployment's time: the older monitor's reviews with no name posted before it are its own.
MOVED_AT = "2026-11-01T12:00:00Z"


def deployment(name: str = "", project: str = "", until: str = "", used=()) -> dict:
    """A deployment's configuration: its automated reviewer name (blank: its container's name, PROJECT, else the
    default), the time before which an older monitor's reviews with no name are its own (blank: none are, as it moved
    from none), and the names it has posted under.
    """
    cfg = {
        "repo": REPO,
        "operator": {"github_login": ACCOUNT},
        "worker": {"model": "qwen"},  # the local worker drafts: line 1 gives its model
        "features": {
            "review_automation": {
                "enabled": True,
                "mode": "live",
                "automated_reviewer_name": name,
            }
        },
        "reviews": {
            "unnamed_are_ours_until": until,
            "automated_reviewer_names_used": list(used),
        },
    }
    if project:
        cfg["deployment"] = {"project": project}
    return cfg


ALPHA = deployment(name="Platform team", until=MOVED_AT)  # moved from the older monitor
BETA = deployment(project="beta-monitor")  # a new one, under its container's name
FRESH = deployment()  # a new one at the default name, which has posted nothing


def line(name: str) -> str:
    return f"Automated review by {name}: Model - qwen, effort: {config.MODEL_DEFAULT}"


def review(line1: str, verdict: str, rnd: str) -> str:
    return (
        f"{line1}\n\n## PR #7 — t — {verdict} {rnd}: x — y\n\n| Prior item | Status |\n"
    )


def comment(cid: int, day: int, body: str, login: str = ACCOUNT) -> dict:
    return {
        "id": cid,
        "login": login,
        "at": f"2026-11-0{day}T10:00:00Z",
        "url": f"https://github.com/{REPO}/pull/7#issuecomment-{cid}",
        "body": body,
    }


COMMENTS = [
    comment(1, 1, review(OLDER_LINE, ORANGE, "First-round")),  # the older monitor's
    comment(2, 2, review(line("Platform team"), YELLOW, "Follow-up")),  # alpha's
    comment(3, 3, review(line("beta-monitor"), GREEN, "First-round")),  # beta's
    comment(4, 4, f"{post.notice_line(ALPHA, 'queue')}\n\nqueued"),  # alpha's notice
    comment(5, 5, "@dev-a please rebase onto develop"),  # the account's, by hand
    # Another account's comment that copies alpha's line 1: not the account's, so no one's review.
    comment(6, 6, review(line("Platform team"), RED, "Follow-up"), login="dev-b"),
]
# An early formal review of the older monitor's, from before line 1: its header alone.
REVIEWS = [
    dict(
        comment(7, 0, "## PR #7 — t — 🟠 First-round: early — z\n"),
        at="2026-10-31T10:00:00Z",
        state="COMMENTED",
        commit="c" * 40,
    )
]


class FakeGitHub:
    """GitHub's REST lists of pull request 7's comments, formal reviews, review comments, commits (none) and timeline
    events (none unless given: rows as head_at's jq leaves them), and its review threads by GraphQL (none), with dev-a
    its author: each row by its author, as gh returns them through the
    callers' jq filters. A `select(.user.login=="...")` keeps that login's rows alone, a `select(.user.login !=
    "...")` every other login's, and a `select(.user.type != "Bot")` leaves out the bots'; a filter that ends
    `| .<time>]` gives the rows' times alone. One comment or formal review by its id gives its body (`.body`), and one
    it does not hold is GitHub's 404."""

    LOGIN = re.compile(r'\.user\.login\s*(==|!=)\s*"([^"]*)"')
    TIMES = re.compile(r"\|\s*\.(?:submitted_at|updated_at|created_at)\]$")
    ONE = re.compile(rf"repos/{REPO}/(issues/comments|pulls/7/reviews)/(\d+)")

    def __init__(self, comments=None, reviews=None, review_comments=(), timeline=()):
        self.comments = COMMENTS if comments is None else comments
        self.reviews = REVIEWS if reviews is None else reviews
        self.review_comments = list(review_comments)
        self.timeline = list(timeline)

    def rest(self, path, jq=None, paginate=False, timeout=120):
        one = self.ONE.fullmatch(path)
        if one:
            rows = self.comments if one.group(1) == "issues/comments" else self.reviews
            row = next((r for r in rows if str(r["id"]) == one.group(2)), None)
            if row is None:
                raise gh.GhError(f"gh: HTTP 404: Not Found ({path})")
            return row["body"] if jq == ".body" else dict(row)
        rows = {
            f"repos/{REPO}/issues/7/comments?per_page=100": self.comments,
            f"repos/{REPO}/pulls/7/reviews?per_page=100": self.reviews,
            f"repos/{REPO}/pulls/7/comments?per_page=100": self.review_comments,
            f"repos/{REPO}/pulls/7/commits?per_page=100": [],
            f"repos/{REPO}/issues/7/timeline?per_page=100": self.timeline,
        }[path]
        jq = jq or ""
        m = self.LOGIN.search(jq)
        if m:
            op, login = m.groups()
            rows = [r for r in rows if (r["login"] == login) is (op == "==")]
        if '.user.type != "Bot"' in jq:
            rows = [r for r in rows if r.get("type") != "Bot"]
        if self.TIMES.search(jq):
            return [r["at"] for r in rows]
        return [dict(r) for r in rows]

    def graphql(self, query, **variables):
        threads = {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}
        pr = {
            "title": "t",
            "mergedAt": None,
            "author": {"login": "dev-a"},
            "mergedBy": None,
            "reviewThreads": threads,
        }
        return {"repository": {"pullRequest": pr}}

    def patch(self):
        return mock.patch.multiple("rkmon_lib.gh", rest=self.rest, graphql=self.graphql)


def latest(cfg: dict, **kw) -> dict | None:
    with FakeGitHub(**kw).patch():
        return snapshot.our_latest_review(cfg, 7)


def follow_up_candidates(cfg: dict, comments: list) -> list[dict]:
    """followups' candidates on pull request 7 for CFG, read from the fake GitHub; the ours kind left out."""
    with FakeGitHub(comments=comments).patch():
        data = followups.read(cfg, 7)
    return followups.candidates(cfg, data, [], None)[0]


def activity(cfg: dict, **kw) -> list[str]:
    """review.activity_since_ours on pull request 7 for CFG: what others left since its latest review of its own."""
    ours = latest(cfg, **kw)
    with FakeGitHub(**kw).patch():
        return activity_since_ours(cfg, 7, {"ours": ours})


class TestLineOne(unittest.TestCase):
    def test_line_1_gives_each_deployment_s_name(self):
        self.assertEqual(bindings.first_line(ALPHA), line("Platform team"))
        self.assertEqual(bindings.first_line(BETA), line("beta-monitor"))
        self.assertEqual(bindings.first_line(FRESH), line("rocke-monitor"))
        self.assertEqual(reviewer_name.given(line("Platform team")), "Platform team")
        self.assertEqual(reviewer_name.given(OLDER_LINE), "")

    def test_two_lines_1_compare_apart_from_the_names_they_give(self):
        """reviewer_name.name_marked, the round trip's: the name a line 1 gives, its mark in its place; a line that
        gives no name, as it is."""
        marked = reviewer_name.name_marked(line("Platform team"))
        self.assertEqual(marked, line(reviewer_name.NAME_MARK))
        self.assertEqual(reviewer_name.name_marked(line("beta-monitor")), marked)
        other = line("beta-monitor").replace("qwen", "llama")
        self.assertNotEqual(reviewer_name.name_marked(other), marked)
        self.assertEqual(reviewer_name.name_marked(OLDER_LINE), OLDER_LINE)
        self.assertEqual(reviewer_name.name_marked(""), "")


class TestOwnReviews(unittest.TestCase):
    def test_each_deployment_takes_only_its_own_latest_review(self):
        alpha, beta = latest(ALPHA), latest(BETA)
        # Alpha: its own review, then the older monitor's two (it moved from it); never beta's, the notice, the
        # account's hand-written comment, or another account's copy of its line 1.
        self.assertEqual(
            (alpha["id"], alpha["verdict"], alpha["count"]), (2, YELLOW, 3)
        )
        self.assertEqual(alpha["line1"], line("Platform team"))
        self.assertEqual((beta["id"], beta["verdict"], beta["count"]), (3, GREEN, 1))
        self.assertIsNone(latest(FRESH))

    def test_the_moved_deployment_also_takes_the_reviews_with_no_name(self):
        older = [COMMENTS[0], *COMMENTS[3:]]  # no named review on the PR
        got = latest(ALPHA, comments=older)
        self.assertEqual((got["id"], got["verdict"], got["count"]), (1, ORANGE, 2))
        got = latest(ALPHA, comments=older, reviews=[])
        self.assertEqual((got["kind"], got["count"]), ("comment", 1))
        self.assertIsNone(latest(BETA, comments=older))
        got = latest(ALPHA, comments=[], reviews=REVIEWS)  # the early review alone
        self.assertEqual(
            (got["kind"], got["head"], got["count"]), ("review", "c" * 40, 1)
        )
        self.assertIsNone(latest(deployment(name="Platform team"), comments=older))

    def test_a_review_with_no_name_is_the_moved_deployment_s_own_only_before_its_time(
        self,
    ):
        """The older monitor's reviews posted before reviews.unnamed_are_ours_until are the moved deployment's own;
        one posted at that time or after it, as by an older monitor still running beside it, is not. Blank, or no
        time: none of them is."""
        before = COMMENTS[0]  # day 1
        after = comment(8, 5, review(OLDER_LINE, RED, "Follow-up"))
        got = latest(ALPHA, comments=[before, after], reviews=[])
        self.assertEqual((got["id"], got["verdict"], got["count"]), (1, ORANGE, 1))
        at = dict(after, at=MOVED_AT)
        self.assertEqual(latest(ALPHA, comments=[before, at], reviews=[])["count"], 1)
        # A formal review by its submitted time: the early review is before it.
        late = dict(REVIEWS[0], id=9, at="2026-11-05T10:00:00Z")
        self.assertIsNone(latest(ALPHA, comments=[], reviews=[late]))
        self.assertEqual(latest(ALPHA, comments=[], reviews=[*REVIEWS, late])["id"], 7)
        # The weekly report counts the earlier one alone.
        end = dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc)
        with FakeGitHub(comments=[before, after], reviews=[]).patch():
            counted = stats.our_reviews(ALPHA, [7], end)[7]
        self.assertEqual([r["at"] for r in counted], [before["at"]])
        # A later time takes both.
        later = deployment(name="Platform team", until="2026-11-06T00:00:00Z")
        got = latest(later, comments=[before, after], reviews=[])
        self.assertEqual((got["id"], got["count"]), (8, 2))
        for until in ("", "yes", "2026-11-06"):
            with self.subTest(until=until):
                cfg = deployment(name="Platform team", until=until)
                self.assertIsNone(latest(cfg, comments=[before, after], reviews=[]))

    def test_in_mirror_mode_every_review_with_no_name_is_its_own(self):
        """A mirror follows an older live monitor under the account: each review whose line 1 gives no name is the
        mirror's own whenever it was posted, with reviews.unnamed_are_ours_until blank. One that gives another name is
        not; a named live monitor's name is the mirror's own, as any deployment's is."""
        mirror = dict(FRESH, mirror={"enabled": True})
        before = COMMENTS[0]  # day 1
        after = comment(8, 5, review(OLDER_LINE, RED, "Follow-up"))
        got = latest(mirror, comments=[before, after], reviews=[])
        self.assertEqual((got["id"], got["verdict"], got["count"]), (8, RED, 2))
        self.assertIsNone(latest(FRESH, comments=[before, after], reviews=[]))
        own = reviewer_name.own_review
        for at in (before["at"], after["at"], None):
            with self.subTest(at=at):
                self.assertTrue(own(mirror, after["body"], at))
                self.assertFalse(own(FRESH, after["body"], at))
        named = review(line("beta-monitor"), GREEN, "First-round")
        self.assertFalse(own(mirror, named, None))
        self.assertTrue(own(dict(BETA, mirror={"enabled": True}), named, None))

    def test_after_a_change_of_name_its_earlier_reviews_stay_its_own(self):
        renamed = deployment(name="Platform", until=MOVED_AT, used=["Platform team"])
        self.assertEqual(bindings.first_line(renamed), line("Platform"))
        self.assertEqual(reviewer_name.names(renamed), ["Platform", "Platform team"])
        self.assertEqual(latest(renamed)["id"], 2)
        # Without the record of the name it posted under, the review is no longer its own.
        self.assertEqual(latest(deployment(name="Platform", until=MOVED_AT))["id"], 1)
        # A container's name it posted under counts the same way.
        moved_on = deployment(project="beta-monitor-2", used=["beta-monitor"])
        self.assertEqual(latest(moved_on)["id"], 3)

    def test_the_weekly_report_counts_each_deployment_s_own_reviews(self):
        end = dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc)
        with FakeGitHub().patch():
            alpha = stats.our_reviews(ALPHA, [7], end)[7]
            beta = stats.our_reviews(BETA, [7], end)[7]
            fresh = stats.our_reviews(FRESH, [7], end)[7]
        # Alpha's: the early review, the older monitor's, its own.
        self.assertEqual([r["verdict"] for r in alpha], [ORANGE, ORANGE, YELLOW])
        self.assertEqual([r["verdict"] for r in beta], [GREEN])
        self.assertEqual(fresh, [])

    def test_the_weekly_report_takes_its_own_among_the_account_s_reviews(self):
        """stats.our_reviews reads our account's reviews where the other readers do (snapshot.account_reviews) and
        keeps this deployment's own among them (reviewer_name.own_review); it reads GitHub no other way.
        """
        end = dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc)
        account = [dict(c, kind="comment", commit=None) for c in COMMENTS[:3]]
        unread = AssertionError("GitHub read past snapshot.account_reviews")
        with mock.patch.object(
            snapshot, "account_reviews", lambda cfg, n: account
        ), mock.patch.object(gh, "rest", side_effect=unread):
            got = stats.our_reviews(ALPHA, [7], end)[7]
        # The older monitor's review, from before the move, and alpha's own; never beta's.
        self.assertEqual(
            [(r["at"], r["verdict"]) for r in got],
            [(COMMENTS[0]["at"], ORANGE), (COMMENTS[1]["at"], YELLOW)],
        )

    def test_own_review_reads_line_1_alone(self):
        own, at = reviewer_name.own_review, COMMENTS[0]["at"]
        self.assertFalse(own(ALPHA, post.notice_line(ALPHA, "queue"), at))
        self.assertFalse(own(ALPHA, "", at))
        self.assertFalse(own(ALPHA, None, at))
        # A name later in the post is no line 1 of it.
        self.assertFalse(own(BETA, f"{OLDER_LINE}\n\n{line('beta-monitor')}\n", at))
        self.assertTrue(own(ALPHA, f"{OLDER_LINE}\n\n{line('beta-monitor')}\n", at))
        # A line 1 that gives a name is that name's whenever it was posted; one that gives none, posted at a time
        # unknown, is not the moved deployment's own.
        self.assertTrue(own(BETA, line("beta-monitor"), None))
        self.assertFalse(own(ALPHA, OLDER_LINE, None))
        self.assertFalse(own(ALPHA, OLDER_LINE, "no time"))

    def test_a_quoted_or_bold_line_1_gives_its_name(self):
        """Line 1 is the first line that is not blank, read after its quote marks and bold or italic marks: a quoted or
        bold named line 1 is that name's review, not one with no name, which the moved deployment would take.
        """
        own, at = reviewer_name.own_review, COMMENTS[0]["at"]
        for first in (
            f"> {line('beta-monitor')}",
            f"**{line('beta-monitor')}**",
            f"> > _{line('beta-monitor')}_",
            f"\n\n*{line('beta-monitor')}* ",
        ):
            body = review(first, GREEN, "First-round")
            with self.subTest(first=first):
                self.assertEqual(reviewer_name.first_line(body), line("beta-monitor"))
                self.assertTrue(own(BETA, body, at))
                self.assertFalse(own(ALPHA, body, at))
        # A quoted line 1 that gives no name still gives none.
        body = review(f"> **{OLDER_LINE}**", ORANGE, "First-round")
        self.assertEqual(reviewer_name.first_line(body), OLDER_LINE)
        self.assertTrue(own(ALPHA, body, at))
        self.assertFalse(own(BETA, body, at))


class TestTheAccountAlone(unittest.TestCase):
    """The readers that leave out the account's posts by the account alone: another deployment's review under it is no
    request among followups' candidates and no activity to review.activity_since_ours.
    """

    ASK = "Should the empty case have a test?"

    def test_another_deployment_s_review_is_no_follow_up_candidate(self):
        beta_asks = comment(
            3, 3, review(line("beta-monitor"), ORANGE, "First-round") + self.ASK
        )
        dev_b_asks = comment(8, 6, self.ASK, login="dev-b")
        got = follow_up_candidates(ALPHA, [COMMENTS[1], beta_asks, dev_b_asks])
        # dev-b's question is a request no one answered; beta's review asks the same, and is beta's to follow up.
        self.assertEqual(
            [(c["kind"], c["reviewer"], c["url"]) for c in got],
            [("unanswered", "dev-b", dev_b_asks["url"])],
        )

    def test_a_quoted_or_bold_line_1_is_no_follow_up_candidate(self):
        """The account's post whose line 1, quoted or bold, is a review's or a notice's (reviewer_name.first_line) is
        no request, though it asks."""
        dev_b_asks = comment(8, 6, self.ASK, login="dev-b")
        for first in (
            f"> {line('beta-monitor')}",
            f"**{line('beta-monitor')}**",
            f"> **{post.notice_line(ALPHA, 'queue')}**",
            f"\n\n_{post.notice_line(BETA, 'hold')}_",
        ):
            asks = comment(3, 3, f"{first}\n\n{self.ASK}")
            with self.subTest(first=first):
                got = follow_up_candidates(ALPHA, [COMMENTS[1], asks, dev_b_asks])
                self.assertEqual(
                    [(c["kind"], c["reviewer"], c["url"]) for c in got],
                    [("unanswered", "dev-b", dev_b_asks["url"])],
                )

    def test_another_deployment_s_review_is_no_activity_since_ours(self):
        # After alpha's review (day 2), beta's review, the merge-queue status and the account's own comment.
        self.assertEqual(activity(ALPHA, comments=COMMENTS[:5]), [])
        beta = dict(
            comment(9, 3, review(line("beta-monitor"), GREEN, "Follow-up")),
            state="COMMENTED",
            commit="d" * 40,
        )
        self.assertEqual(
            activity(ALPHA, comments=COMMENTS[:5], reviews=[*REVIEWS, beta]), []
        )
        # Another account's comment and review comment are activity.
        nit = comment(10, 6, "nit: a clearer name", login="dev-b")
        self.assertEqual(
            activity(ALPHA, review_comments=[nit]),
            [
                "since our last review (2026-11-02T10:00:00Z): 0 reviews, 1 review comment and 1 comment from "
                "others: answer them, and restate each row whose status they change"
            ],
        )


class TestReviewGate(unittest.TestCase):
    """G1 reads a deployment's own review where it reviews, and people's approvals where it does not."""

    HEAD = "a" * 40
    APPROVAL = {
        "login": "lead",
        "state": "APPROVED",
        "at": "2026-11-07T10:00:00Z",
        "commit": HEAD,
    }

    def gate(self, cfg, comments=None, **kw):
        ours = latest(cfg, comments=comments)
        if ours:
            # The reviewed head, as resolve_reviewed_head finds it.
            ours["head"] = self.HEAD
        rec = {"n": 7, "status": "open", "head": self.HEAD, "author": "dev-a"}
        return queue.gate_review(cfg, dict(rec, ours=ours, **kw))

    def test_a_merge_queue_only_deployment_gates_on_people_s_approvals_alone(self):
        gamma = deployment(project="queue-monitor")
        gamma["features"] = {
            "review_automation": {"enabled": False},
            "merge_queue": {"enabled": True, "mode": "live"},
        }
        moved = dict(gamma, reviews={"unnamed_are_ours_until": MOVED_AT})
        self.assertIsNone(latest(gamma))  # neither deployment's review is its own
        # The older monitor's review is the moved one's own.
        self.assertEqual(latest(moved)["verdict"], ORANGE)
        for cfg in (gamma, moved):
            with self.subTest(own=latest(cfg)):
                self.assertEqual(
                    self.gate(cfg, reviews=[], review_decision="REVIEW_REQUIRED"),
                    (queue.FAIL, "no approval by a person yet"),
                )
                self.assertEqual(
                    self.gate(cfg, reviews=[self.APPROVAL], review_decision="APPROVED"),
                    (queue.PASS, "approved at head by lead"),
                )

    def test_a_reviewing_deployment_gates_on_its_own_review_alone(self):
        self.assertEqual(self.gate(BETA), (queue.PASS, f"{GREEN} at head"))
        self.assertEqual(self.gate(ALPHA)[0], queue.PASS)  # its 🟡
        # Beta's 🟢 is no review of alpha's: without its own, alpha's gate fails, whatever the approvals.
        others = [x for x in COMMENTS if x["id"] != 2]
        self.assertEqual(
            self.gate(
                deployment(name="Platform team"),
                comments=others,
                reviews=[self.APPROVAL],
                review_decision="APPROVED",
            ),
            (queue.FAIL, "not reviewed by us"),
        )


class TestPostLineOne(unittest.TestCase):
    def test_rk_post_refuses_another_deployment_s_line_1(self):
        st = {"prs": {}}
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "p.md"
            for cfg, other in (
                (ALPHA, line("beta-monitor")),
                (BETA, line("Platform team")),
                (BETA, OLDER_LINE),
            ):
                with self.subTest(own=bindings.first_line(cfg), other=other):
                    f.write_text(review(other, GREEN, "First-round"), encoding="utf-8")
                    with self.assertRaises(post.PostRefused) as cm:
                        post.preflight(cfg, st, None, 7, str(f))
                    self.assertIn(
                        f"line 1 must be exactly {bindings.first_line(cfg)!r}",
                        str(cm.exception),
                    )
                    # Its own line 1 passes: the next guard refuses, for a pull request it does not track.
                    f.write_text(
                        review(bindings.first_line(cfg), GREEN, "First-round"),
                        encoding="utf-8",
                    )
                    with self.assertRaises(post.PostRefused) as cm:
                        post.preflight(cfg, st, None, 7, str(f))
                    self.assertEqual(str(cm.exception), "#7 is not a tracked open PR")


HEAD, OLD, NEW = "a" * 40, "b" * 40, "e" * 40
# Pull request 7's heads: OLD from day 1, HEAD pushed on day 3 at 09:00, before beta's review (COMMENTS[2], 10:00).
TIMELINE = [
    {"event": "committed", "sha": OLD, "t": "2026-11-01T09:00:00Z"},
    {"event": "head_ref_force_pushed", "sha": HEAD, "t": "2026-11-03T09:00:00Z"},
]
STATE_FILES = ("state.json", "journal.md", "actions.jsonl")
MY_POST_AT = "2026-11-02T10:00:02Z"  # alpha's post of COMMENTS[1], in its ledger
# The older monitor's review of HEAD, its line 1 giving no name, posted after alpha moved from it (MOVED_AT): an older
# monitor still running beside alpha, or its last review before it stopped. It is no review of alpha's own
# (reviewer_name.own_review), but it stands for one (reviewer_name.stands_for_ours), so `rk review-close` takes it.
OLDER_AT_HEAD = dict(
    comment(11, 3, review(OLDER_LINE, GREEN, "Follow-up")), at="2026-11-03T10:30:00Z"
)
# A formal review posted by hand under the account, its header alone (no line 1): the commit it names is its head.
BY_HAND = dict(
    comment(9, 3, f"## PR #7 — t — {GREEN} Follow-up: fine — z\n"),
    at="2026-11-03T12:00:00Z",
    url=f"https://github.com/{REPO}/pull/7#pullrequestreview-9",
    state="COMMENTED",
    commit=HEAD,
)


def ledger_post(store, at: str, head: str, url: str, note: str) -> None:
    """A post of alpha's in its ledger, made at `at`, as `rk post` records one."""
    row = {"ts": at, "kind": "post", "pr": 7, "head": head, "ref": url, "note": note}
    store.append_jsonl("actions.jsonl", [row])


def hour_after(t: str) -> str:
    return (parse_ts(t) + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def url(cid: int) -> str:
    """The link of the account's comment CID on pull request 7."""
    return f"https://github.com/{REPO}/pull/7#issuecomment-{cid}"


class AccountCase(unittest.TestCase):
    """This deployment, alpha ("Platform team", moved from the older monitor at MOVED_AT), reviewed pull request 7 at
    OLD (its post, COMMENTS[1], in its ledger); its follow-up of HEAD awaits verification. Under the same account, beta
    posted its review of HEAD (COMMENTS[2]), and so did the older monitor, after the move (OLDER_AT_HEAD).
    """

    def setUp(self):
        self.fresh()

    def fresh(self):
        """The case above, in a folder of its own."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = self.base = pathlib.Path(tmp.name)
        self.store = Store(d / "state")
        self.cfg = dict(
            deployment(name="Platform team", until=MOVED_AT),
            paths={
                "working_dir": str(d),
                "repo": "/nonexistent",
                "workers": str(d),
                "posted_dir": str(d / "outputs"),
            },
        )
        self.comments, self.reviews = [*COMMENTS, OLDER_AT_HEAD], []
        self.timeline, self.unread = list(TIMELINE), None
        mine = COMMENTS[1]
        ledger_post(self.store, MY_POST_AT, OLD, mine["url"], "first_round")
        self.st = empty_state()
        self.st["prs"]["7"] = {
            "n": 7,
            "status": "open",
            "head": HEAD,
            "ours": dict(
                snapshot.as_ours(dict(mine, kind="comment", commit=None), 1),
                head=OLD,
                head_src="ledger",
            ),
        }
        self.st["review_ready"] = [
            {
                "pr": 7,
                "kind": "follow-up",
                "head": HEAD,
                "at": "2026-11-03T11:00:00Z",
                "jobs": {},
            }
        ]
        self.store.save(self.st)

    @contextlib.contextmanager
    def github(self):
        """The fake GitHub, each of its REST reads failing with self.unread when that is set. A gh started outside it,
        a write among them, fails the test."""
        fake = FakeGitHub(
            comments=self.comments, reviews=self.reviews, timeline=self.timeline
        )
        if self.unread:
            fake.rest = mock.Mock(side_effect=self.unread)
        outside = AssertionError("a gh call outside the fake GitHub")
        with fake.patch(), mock.patch.object(gh, "_exec", side_effect=outside):
            yield

    def files(self) -> dict:
        """The state, the journal and the ledger, as they stand."""
        return {
            name: (self.store.dir / name).read_bytes()
            for name in STATE_FILES
            if (self.store.dir / name).exists()
        }

    def worker_fetched(self, head: str, at: str) -> None:
        """Our worker's fetch of pull request 7 at `head`, at the time `at`: its meta.md, as snapshot.worker_meta_head
        reads it."""
        meta = self.base / "pr7" / "pr" / "meta.md"
        meta.parent.mkdir(parents=True)
        meta.write_text(f"- Head: pr7 @ {head};\n", encoding="utf-8")
        t = parse_ts(at).timestamp()
        os.utime(meta, (t, t))


class TestReviewClose(AccountCase):
    """`rk review-close N`: our account's review at the head of this deployment's review closes it without a post when
    it stands for ours (its line 1 gives no name, or this deployment's: an older monitor's posted after the move, or one
    posted by hand), and is its last review of the pull request from then on; another deployment's review never does.
    Refused, nothing changed, when none covers the head, no review waits, or in a mirror. It reads GitHub before it
    takes the state lock, and writes nothing there: any gh call outside the fake GitHub fails.
    """

    COVERED = f"(a review our account posted covers it: {OLDER_AT_HEAD['url']})"
    LAST = f"#7: that review is our last review of the PR, at {HEAD[:10]}: a new head gets a follow-up"

    def rk(self, *argv) -> tuple[int, list[str]]:
        import rkmon

        out = []
        with self.github(), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            code = rkmon.main(list(argv))
        return code, out

    def test_an_older_monitor_s_review_after_the_move_closes_the_review_and_is_our_last(
        self,
    ):
        """The case the command is for: the older monitor, running on after alpha's move from it, reviewed HEAD under
        the same account. Posted after the move, its review is not alpha's own, but it stands for alpha's review of
        HEAD, which closes without a post; beta's review of HEAD, another deployment's, is left alone.
        """
        body, at = OLDER_AT_HEAD["body"], OLDER_AT_HEAD["at"]
        self.assertEqual(reviewer_name.first_line(body), OLDER_LINE)
        self.assertFalse(reviewer_name.own_review(self.cfg, body, at))
        self.assertTrue(reviewer_name.stands_for_ours(self.cfg, body))
        ledger = self.store.ledger()
        self.assertEqual(
            self.rk("review-close", "7"),
            (
                0,
                [
                    f"#7: the draft awaiting verification at {HEAD[:10]} is closed without a post {self.COVERED}",
                    self.LAST,
                ],
            ),
        )
        st = self.store.load()
        self.assertEqual(st["review_ready"], [])
        ours = st["prs"]["7"]["ours"]
        self.assertTrue(ours.pop(post.TAKEN))
        # The older monitor's review, its head the PR's when it was posted (the timeline), counted as a post of ours
        # would be.
        self.assertEqual(
            ours,
            dict(
                snapshot.as_ours(dict(OLDER_AT_HEAD, kind="comment", commit=None), 2),
                head=HEAD,
                head_src="timeline",
            ),
        )
        journal = self.store.journal_path.read_text()
        self.assertIn(
            f"#7 draft closed without a post at {HEAD[:10]}: a review our account posted covers it: "
            f"{OLDER_AT_HEAD['url']}",
            journal,
        )
        self.assertIn(
            f"#7 review-close: our last review is our account's review at {HEAD[:10]}, posted {at}: "
            f"{OLDER_AT_HEAD['url']}",
            journal,
        )
        self.assertEqual(self.store.ledger(), ledger)  # nothing posted
        # Its line of the review costs: a review of its own, closed (review_cost.record_close), which `rk spend` prices.
        (line,) = self.store.read_jsonl("review_cost.jsonl")
        self.assertEqual(
            {
                k: line.get(k)
                for k in ("pr", "round", "head", "post", "closed", "unposted")
            },
            {
                "pr": 7,
                "round": "fu1",
                "head": HEAD,
                "post": OLDER_AT_HEAD["url"],
                "closed": True,
                "unposted": None,
            },
        )

    def test_another_deployment_s_review_is_not_ours_to_take(self):
        """Beta's review of HEAD under the same account, a comment or a formal review: it gives another name, so it
        never stands for alpha's review. Each deployment posts a review of its own."""
        before = self.files()
        refused = (
            4,
            [
                f"REFUSED: our account's review of #7 at {HEAD[:10]} on GitHub is another deployment's "
                "(beta-monitor), not ours to take: nothing closed"
            ],
        )
        self.comments = list(COMMENTS)
        self.assertEqual(self.rk("review-close", "7"), refused)
        self.assertEqual(self.files(), before)
        self.comments = [COMMENTS[1]]
        self.reviews = [
            dict(BY_HAND, body=review(line("beta-monitor"), GREEN, "Follow-up"))
        ]
        self.assertEqual(self.rk("review-close", "7"), refused)
        self.assertEqual(self.files(), before)

    def test_the_pr_is_not_reviewed_again_at_that_head_and_a_new_head_gets_a_follow_up(
        self,
    ):
        import rkmon

        self.rk("review-close", "7")
        st = self.store.load()
        rec = st["prs"]["7"]
        self.assertTrue(snapshot.covers(rec["ours"], HEAD))
        # The author pushes NEW. The poll reads alpha's own latest review again: from day 2, before the take.
        rec["head"] = NEW
        with self.github():
            self.assertEqual(rkmon.refresh_ours(self.cfg, self.store, st, {"7"}), 0)
        self.assertEqual(rec["ours"]["id"], OLDER_AT_HEAD["id"])
        self.assertTrue(snapshot.covers(rec["ours"], HEAD))
        self.assertFalse(snapshot.covers(rec["ours"], NEW))
        self.assertEqual(
            (review_mod.round_kind(rec), prep.default_round(st, 7)),
            ("follow-up", "fu2"),
        )
        # A review of alpha's own of NEW, posted an hour after the take: our last review from then on.
        at = hour_after(rec["ours"][post.TAKEN])
        own = dict(
            comment(8, 4, review(line("Platform team"), GREEN, "Follow-up")), at=at
        )
        self.comments.append(own)
        ledger_post(self.store, at, NEW, own["url"], "followup")
        with self.github():
            self.assertEqual(rkmon.refresh_ours(self.cfg, self.store, st, {"7"}), 1)
        self.assertEqual(
            (rec["ours"]["id"], rec["ours"]["head"], post.taken(rec["ours"])),
            (8, NEW, False),
        )

    def test_a_review_older_than_our_own_last_one_stands_once_taken(self):
        """The PR went back to HEAD after alpha's own review of NEW (day 4): the older monitor's review of HEAD, from
        before it, covers HEAD, and stays our last review."""
        import rkmon

        own = comment(8, 4, review(line("Platform team"), GREEN, "Follow-up"))
        self.comments.append(own)
        ledger_post(self.store, own["at"], NEW, own["url"], "followup")
        self.st["prs"]["7"]["ours"] = dict(
            snapshot.as_ours(dict(own, kind="comment", commit=None), 2),
            head=NEW,
            head_src="ledger",
        )
        self.store.save(self.st)
        self.assertEqual(self.rk("review-close", "7")[0], 0)
        st = self.store.load()
        rec = st["prs"]["7"]
        self.assertEqual((rec["ours"]["id"], rec["ours"]["count"]), (11, 3))
        with self.github():
            self.assertEqual(rkmon.refresh_ours(self.cfg, self.store, st, {"7"}), 0)
        self.assertEqual(rec["ours"]["id"], 11)
        self.assertTrue(snapshot.covers(rec["ours"], HEAD))

    def test_the_follow_up_gap_and_the_facts_date_our_last_review_by_the_taken_one(
        self,
    ):
        """One rule for both (plan.last_review_time): the later of alpha's last post and the review it took."""
        self.rk("review-close", "7")
        ours = self.store.load()["prs"]["7"]["ours"]
        last_posts = plan.last_review_posts(self.store.ledger())
        self.assertEqual(last_posts[7], MY_POST_AT)
        self.assertEqual(
            plan.last_review_time(7, ours, last_posts), OLDER_AT_HEAD["at"]
        )
        later = hour_after(OLDER_AT_HEAD["at"])
        self.assertEqual(plan.last_review_time(7, ours, {7: later}), later)

    def test_a_review_in_flight_is_stopped_and_frees_its_slot(self):
        killed = []
        self.st["review_ready"] = []
        self.st["review_pipes"] = [
            {
                "pr": 7,
                "head": HEAD,
                "kind": "follow-up",
                "stage": "worker",
                "worker": {"id": "w"},
                "checks": {"id": "c"},
            }
        ]
        self.st["jobs"] = {"heavy": {"id": "c"}}
        self.store.save(self.st)
        with mock.patch.object(
            review_mod.jobs, "status", lambda store, job: "running"
        ), mock.patch.object(
            review_mod.jobs, "kill", lambda store, job: killed.append(job["id"])
        ):
            got = self.rk("review-close", "7")
        self.assertEqual(
            got,
            (
                0,
                [
                    f"#7: the review in flight at {HEAD[:10]} is stopped and closed without a post {self.COVERED}",
                    self.LAST,
                ],
            ),
        )
        st = self.store.load()
        self.assertEqual(killed, ["w", "c"])
        self.assertEqual((st["review_pipes"], st["jobs"]["heavy"]), ([], None))
        self.assertEqual(st["prs"]["7"]["ours"]["id"], OLDER_AT_HEAD["id"])
        # A review in flight had no verifier run: no line of the review costs.
        self.assertEqual(self.store.read_jsonl("review_cost.jsonl"), [])

    def test_a_formal_review_posted_by_hand_covers_the_head_it_names(self):
        self.comments = [COMMENTS[1]]
        self.reviews = [BY_HAND]
        code, out = self.rk("review-close", "7")
        self.assertEqual(code, 0, out)
        ours = self.store.load()["prs"]["7"]["ours"]
        self.assertEqual(
            (ours["kind"], ours["id"], ours["head"], ours["head_src"], ours["count"]),
            ("review", 9, HEAD, "review-commit", 2),
        )

    def test_the_head_of_a_review_this_deployment_did_not_post_comes_from_github_alone(
        self,
    ):
        """Never from this deployment's ledger or worker files (snapshot.account_review_head): the older monitor's
        review, posted after HEAD's push, is at HEAD by the PR's timeline, though a post of ours at OLD is in the
        ledger minutes before it, or our worker fetched OLD before it (snapshot.resolve_reviewed_head, for this
        deployment's own posts, would date it to OLD by either)."""
        for where in ("ledger", "worker"):
            with self.subTest(where=where):
                self.fresh()
                self.comments = [COMMENTS[1], OLDER_AT_HEAD]
                if where == "ledger":
                    ledger_post(
                        self.store, "2026-11-03T10:20:00Z", OLD, url(99), "followup"
                    )
                else:
                    self.worker_fetched(OLD, "2026-11-03T09:50:00Z")
                code, out = self.rk("review-close", "7")
                self.assertEqual(code, 0, out)
                ours = self.store.load()["prs"]["7"]["ours"]
                self.assertEqual(
                    (ours["id"], ours["head"], ours["head_src"]),
                    (OLDER_AT_HEAD["id"], HEAD, "timeline"),
                )

    def test_a_taken_review_s_text_is_read_from_github_by_its_id(self):
        """Our last review's text, against which the next follow-up checks its prior items (post.our_last_review), is
        the taken review's as GitHub holds it, a formal review's or a comment's, read by its id; never a copy of this
        deployment's own posts, which is no copy of that review, even when GitHub no longer has it.
        """
        for kind, taken, comments, reviews in (
            ("review", BY_HAND, [COMMENTS[1]], [BY_HAND]),
            ("comment", OLDER_AT_HEAD, [COMMENTS[1], OLDER_AT_HEAD], []),
        ):
            with self.subTest(kind=kind):
                self.fresh()
                copy = (
                    pathlib.Path(self.cfg["paths"]["posted_dir"])
                    / "PR_FIRST_ROUND_7.md"
                )
                copy.parent.mkdir()
                copy.write_text("ALPHA'S OWN EARLIER REVIEW\n", encoding="utf-8")
                self.comments, self.reviews = comments, reviews
                code, out = self.rk("review-close", "7")
                self.assertEqual(code, 0, out)
                ours = self.store.load()["prs"]["7"]["ours"]
                self.assertEqual((ours["kind"], ours["id"]), (kind, taken["id"]))
                with self.github():
                    self.assertEqual(
                        post.our_last_review(self.cfg, 7, ours),
                        (taken["body"], "GitHub"),
                    )
                self.comments, self.reviews = [COMMENTS[1]], []  # deleted since
                with self.github():
                    self.assertEqual(post.our_last_review(self.cfg, 7, ours), ("", ""))

    def test_nothing_is_written_to_github_in_live_prototype_or_standby(self):
        """Whatever review posting's mode, and in standby: the close reads GitHub alone (any gh call outside the fake
        GitHub, a write among them, fails the test), and the ledger of our posts is unchanged.
        """
        for mode, standby in (("live", False), ("prototype", False), ("live", True)):
            with self.subTest(mode=mode, standby=standby):
                self.fresh()
                self.cfg["features"]["review_automation"]["mode"] = mode
                ledger = self.store.ledger()
                with mock.patch.object(
                    features, "standby", lambda home=None, on=standby: on
                ):
                    code, out = self.rk("review-close", "7")
                self.assertEqual((code, out[-1:]), (0, [self.LAST]))
                self.assertEqual(self.store.ledger(), ledger)

    def test_a_mirror_closes_none(self):
        """In a mirror every review of our account that gives no name is the live monitor's, the other side of the
        comparison, which never stands for one of the mirror's: rk review-close refuses in one line, before it reads
        GitHub, and changes nothing."""
        self.cfg["mirror"] = {"enabled": True}
        before = self.files()
        self.unread = AssertionError("GitHub read in a mirror")
        self.assertEqual(
            self.rk("review-close", "7"), (4, [f"REFUSED: {mirror.NO_CLOSE}"])
        )
        self.assertEqual(self.files(), before)

    def test_github_is_read_before_the_state_lock_is_taken(self):
        """A tick never waits on the close's GitHub reads: none is made while the state lock is held. rk review-close
        takes the lock after them, to read the review's heads again and write."""
        held, reads = [], []
        rest = FakeGitHub.rest

        def probe(fake, path, *args, **kwargs):
            fd = os.open(self.store.dir / "lock", os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                held.append(path)
            finally:
                os.close(fd)
            reads.append(path)
            return rest(fake, path, *args, **kwargs)

        with mock.patch.object(FakeGitHub, "rest", probe):
            code, out = self.rk("review-close", "7")
        self.assertEqual((code, out[-1:]), (0, [self.LAST]))
        self.assertTrue(reads)
        self.assertEqual(held, [])

    def test_a_review_that_moved_on_while_github_was_read_is_not_closed(self):
        """The heads are read again under the lock: a tick that, while GitHub was read, handed the review of NEW to
        verification in place of HEAD's leaves nothing to close at HEAD, and NEW's stays.
        """
        rest = FakeGitHub.rest

        def tick_meanwhile(fake, path, *args, **kwargs):
            st = self.store.load()
            st["review_ready"][0]["head"] = NEW
            self.store.save(st)
            return rest(fake, path, *args, **kwargs)

        with mock.patch.object(FakeGitHub, "rest", tick_meanwhile):
            got = self.rk("review-close", "7")
        self.assertEqual(
            got,
            (
                4,
                [
                    f"REFUSED: #7's review moved on to {NEW[:10]} while GitHub was read: nothing closed; run "
                    "rk review-close 7 again"
                ],
            ),
        )
        st = self.store.load()
        self.assertEqual([r["head"] for r in st["review_ready"]], [NEW])
        self.assertFalse(post.taken(st["prs"]["7"]["ours"]))

    def test_refused_and_nothing_changed_when_nothing_covers_the_head(self):
        before = self.files()
        # Our account's reviews are at OLD: the older monitor's from before the move and alpha's own. Its notice and
        # its hand-written comment are no reviews, and dev-b's copy of alpha's line 1 is another account's.
        self.comments = [c for c in COMMENTS if c["id"] != 3]
        self.assertEqual(
            self.rk("review-close", "7"),
            (
                4,
                [
                    f"REFUSED: our account has no review of #7 at {HEAD[:10]} on GitHub (`rk facts 7` lists the ones "
                    "this deployment did not post): nothing closed"
                ],
            ),
        )
        self.assertEqual(self.files(), before)
        self.unread = gh.GhError("gh: HTTP 502")
        self.assertEqual(
            self.rk("review-close", "7"),
            (
                4,
                [
                    "REFUSED: our account's reviews on #7 were not read (gh: HTTP 502): nothing closed"
                ],
            ),
        )
        self.assertEqual(self.files(), before)

    def test_refused_and_nothing_changed_when_no_review_waits(self):
        before = self.files()
        self.assertEqual(
            self.rk("review-close", "9"),
            (4, ["REFUSED: #9 is not tracked: nothing to close"]),
        )
        self.st["review_ready"] = []
        self.store.save(self.st)
        before = self.files()
        self.assertEqual(
            self.rk("review-close", "7"),
            (
                4,
                [
                    "REFUSED: #7 has no review awaiting verification or in flight: nothing to close"
                ],
            ),
        )
        self.assertEqual(self.files(), before)


class TestFactsAccount(AccountCase):
    """`rk facts N` lists our account's reviews that this deployment's ledger does not hold, each with its time, its
    light and its link: one that stands for ours with its head, read from GitHub alone and said in plain words, saying
    this deployment did not post it; another deployment's marked not ours to take; in a mirror, none posted after the
    feed line."""

    ESTIMATE = "(the PR's head when it was posted, an estimate)"

    def test_the_account_s_reviews_this_deployment_did_not_post(self):
        self.reviews = REVIEWS
        posts = [r for r in self.store.ledger() if r["pr"] == 7]
        before = self.files()
        with self.github():
            lines = facts._account(self.cfg, self.st, self.store, 7, {"posts": posts})
        self.assertEqual(
            lines,
            [
                f"2026-10-31T10:00:00Z review at cccccccccc (the commit the review names) {ORANGE} {url(7)}: not "
                "posted by this deployment",
                f"2026-11-01T10:00:00Z comment at {OLD[:10]} {self.ESTIMATE} {ORANGE} {url(1)}: not posted by this "
                "deployment",
                f"2026-11-03T10:00:00Z comment {GREEN} {url(3)}: another deployment's (beta-monitor), not ours to "
                "take",
                f"2026-11-03T10:30:00Z comment at {HEAD[:10]} {self.ESTIMATE} {GREEN} {url(11)}: not posted by this "
                "deployment",
            ],
        )
        self.assertEqual(self.files(), before)  # read-only
        self.comments, self.reviews = [COMMENTS[1]], []
        with self.github():
            self.assertEqual(
                facts._account(self.cfg, self.st, self.store, 7, {"posts": posts}),
                ["none"],
            )

    def test_a_mirror_lists_none_posted_after_the_feed_line(self):
        """In a mirror, a review of our account posted after the feed line of the review mirrored is the live
        monitor's review of that round, which the mirror's verifier reviews blind (mirror.posts_until); with no review
        started from the feed, the section lists them up to now."""
        cfg = dict(self.cfg, mirror={"enabled": True})
        self.comments = [COMMENTS[0], OLDER_AT_HEAD]
        title = next(t for t, section in facts.SECTIONS if section is facts._account)
        older = (
            f"2026-11-01T10:00:00Z comment at {OLD[:10]} {self.ESTIMATE} {ORANGE} {url(1)}: not posted by this "
            "deployment"
        )
        live = (
            f"2026-11-03T10:30:00Z comment at {HEAD[:10]} {self.ESTIMATE} {GREEN} {url(11)}: not posted by this "
            "deployment"
        )
        st = self.store.load()
        started = mirror.state(st)["started"]
        for feed_at, rows in (("2026-11-03T10:15:00Z", [older]), ("", [older, live])):
            with self.subTest(feed_at=feed_at):
                started.clear()
                if feed_at:
                    started[mirror.key_of(7, HEAD)] = {
                        "pr": 7,
                        "head": HEAD,
                        "feed_at": feed_at,
                    }
                with self.github(), mock.patch.object(
                    facts, "SECTIONS", ((title, facts._account),)
                ):
                    out, failed = facts.render(cfg, st, self.store, 7)
                self.assertEqual((failed, out[2:]), (0, rows))

    def test_the_list_follows_our_reviews(self):
        sections = [section for _, section in facts.SECTIONS]
        i = sections.index(facts._ours)
        self.assertIs(sections[i + 1], facts._account)
        self.assertIn("this deployment did not post", facts.SECTIONS[i + 1][0])


class TestTheDocs(unittest.TestCase):
    """What the docs say of a review of the account this deployment did not post. The deploy guide gives a pull request
    in two deployments' scopes a review from each, whatever the account: no deployment takes another's review. The
    verifier's team file, the review procedure and the skill close a review without a post only on a review that
    stands for ours, never on one the facts mark "not ours to take", and the verifier confirms its head from its
    content first.
    """

    def flat(self, rel: str) -> str:
        """A doc of the package, its lines joined by single spaces."""
        root = pathlib.Path(__file__).resolve().parent.parent
        return " ".join((root / rel).read_text(encoding="utf-8").split())

    def test_each_deployment_posts_its_own_review_whatever_the_account(self):
        guide = self.flat("QUICK_DEPLOY_GUIDE.md")
        self.assertIn(
            "A pull request in two deployments' scopes gets a review from each, as intended.",
            guide,
        )
        self.assertNotIn("review-close", guide)

    def test_close_no_post_only_on_a_review_that_stands_for_ours(self):
        for rel in (
            "agentic/team_members/rocke_review_verifier.md",
            "agentic/skills/rocke-monitor/review_procedure.md",
            "agentic/skills/rocke-monitor/SKILL.md",
        ):
            with self.subTest(rel=rel):
                self.assertIn('"not ours to take"', self.flat(rel))
        verifier = self.flat("agentic/team_members/rocke_review_verifier.md")
        self.assertIn(
            "confirm that head from the review itself: read it once by its link with your read `gh`.",
            verifier,
        )
        self.assertIn(
            "When its content does not fit this head, or you cannot tell, verify as usual.",
            verifier,
        )


if __name__ == "__main__":
    unittest.main()
