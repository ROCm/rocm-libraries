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
deployment's own review where it reviews, and people's approvals alone where it does not.
"""

import datetime as dt
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
    followups,
    post,
    queue,
    reviewer_name,
    snapshot,
    stats,
)  # noqa: E402
from rkmon_lib.review import activity_since_ours  # noqa: E402

GREEN, ORANGE, RED, YELLOW = "\U0001f7e2", "\U0001f7e0", "\U0001f534", "\U0001f7e1"
REPO = "o/r"
ACCOUNT = "review-account"  # the one account every deployment here posts under
# An older monitor's line 1: it gives no name.
OLDER_LINE = "Automated review: Model - qwen, effort: max"
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
    """GitHub's REST lists of pull request 7's comments, formal reviews, review comments and commits (none), and its
    review threads by GraphQL (none), with dev-a its author: each row by its author, as gh returns them through the
    callers' jq filters. A `select(.user.login=="...")` keeps that login's rows alone, a `select(.user.login !=
    "...")` every other login's, and a `select(.user.type != "Bot")` leaves out the bots'; a filter that ends
    `| .<time>]` gives the rows' times alone."""

    LOGIN = re.compile(r'\.user\.login\s*(==|!=)\s*"([^"]*)"')
    TIMES = re.compile(r"\|\s*\.(?:submitted_at|updated_at|created_at)\]$")

    def __init__(self, comments=None, reviews=None, review_comments=()):
        self.comments = COMMENTS if comments is None else comments
        self.reviews = REVIEWS if reviews is None else reviews
        self.review_comments = list(review_comments)

    def rest(self, path, jq=None, paginate=False, timeout=120):
        rows = {
            f"repos/{REPO}/issues/7/comments?per_page=100": self.comments,
            f"repos/{REPO}/pulls/7/reviews?per_page=100": self.reviews,
            f"repos/{REPO}/pulls/7/comments?per_page=100": self.review_comments,
            f"repos/{REPO}/pulls/7/commits?per_page=100": [],
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


if __name__ == "__main__":
    unittest.main()
