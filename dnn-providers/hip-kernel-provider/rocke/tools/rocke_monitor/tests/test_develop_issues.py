# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The develop-issue rules: red-develop only for a develop break that makes CI red; otherwise silent-break,
assigned to the author of the breaking PR. Open PRs are swept for fixes to the open issues, their authors are told,
and the fixes go to the top of the review and merge queues."""

import pathlib
import sys
import tempfile
import unittest

TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import guard_labels  # noqa: E402
import rkmon  # noqa: E402
from rkmon_lib import (
    breakage,
    gh,
    mergeplan,
    plan,
    post,
    queue,
    report,
    webdash,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

G = "\U0001f7e2"
# daily_dir: the daily-log file of a guarded write goes to a folder of this module's own.
_DAILY = tempfile.TemporaryDirectory()


def tearDownModule():
    _DAILY.cleanup()


CFG = {
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "incident_issue": guard_labels.INCIDENT_ISSUE,
    "features": {
        "merge_queue": {"mode": "live", "auto_merge": "enabled", "develop_guard": True},
        "alerts": {
            "enabled": True,
            "mode": "live",
            "develop_breaks": True,
            "needs_you": True,
        },
    },
    "max_posts_per_hour": 6,
    "review_first_line": "x",
    "settle_min": 30,
    "codeowner_teams": {"platform": ["plat"]},
    "scope": {"labels": ["rocke: platform"], "path_labels": []},
    "paths": {
        "working_dir": "/nonexistent",
        "repo": "/nonexistent",
        "daily_dir": _DAILY.name,
    },
}
# Line 1 of a develop issue's body and of our comments on one: this deployment's report line.
REPORT = post.notice_line(CFG, "report")
TITLE = (
    "[rocke] test_selector_parity.py::TestKvAddrLimit::test_every_tiled_flag_is_classified_for_kv_addressing "
    "fails on develop since #1111 (use_additive_bias is not classified)"
)
BODY = (
    REPORT + "\n\n## Context\n\n"
    "- **Failing test:** `library/tests/test_selector_parity.py::TestKvAddrLimit::"
    "test_every_tiled_flag_is_classified_for_kv_addressing` (host only)\n"
    "- **Component:** the KV-addressing classes (`library/tests/test_selector_parity.py`, "
    "`library/kernels/gfx950/attention_tiled_2d.py`)\n"
    "- **Also reported by:** #4567 lists `TestKvAddrLimit` among its known failures.\n\n"
    "## Why it slipped through CI\n\nNo CTest lane in `platform/CMakeLists.txt` names it; sibling files such as "
    "`test_conv_abi.py` run there.\n"
)


def head(n: int) -> str:
    return (str(n) * 40)[:40]


def pr(n: int, **kw) -> dict:
    r = {
        "n": n,
        "head": head(n),
        "size": 100,
        "status": "open",
        "draft": False,
        "author": f"dev{n}",
        "title": f"t{n}",
        "labels": [],
        "base": "develop",
        "head_ref": f"b{n}",
        "mergeable": "MERGEABLE",
        "ci_rollup": "SUCCESS",
        "review_decision": "APPROVED",
        "reviews": [],
        "closes": [],
        "head_seen_at": "2026-10-29T00:00:00Z",
        "files_cache": {"head": head(n), "hot": [], "areas": ["platform"]},
        "ours": {
            "verdict": G,
            "head": head(n),
            "at": now(),
            "url": "https://github.com/ROCm/r/pull/1#c",
        },
        "threads": {"updated_at": None, "open": []},
    }
    r.update(kw)
    return r


def issue(n: int = 11111, kind: str = breakage.STABILITY, **kw) -> dict:
    i = {
        "n": n,
        "label": breakage.kind_label(CFG, kind),
        "title": TITLE,
        "body": BODY,
        "author": "me",
        "assignees": [],
        "created": now(),
        "url": "u",
        "labels": [kind, "rocke: platform"],
        "kind": kind,
    }
    i.update(kw)
    return i


def write(d: str, text: str) -> str:
    p = pathlib.Path(d) / "f.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestTwoLabels(unittest.TestCase):
    def test_parse_file_wants_exactly_one_label_and_the_authors_login_for_stability(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:

            def parse(meta: str, cfg: dict = CFG) -> dict:
                return breakage.parse_file(
                    write(
                        d,
                        f"---\ntitle: [develop] x\n{meta}\n---\n{REPORT}\n",
                    ),
                    cfg,
                )

            with self.assertRaisesRegex(
                ValueError,
                "exactly one of red-develop .* or silent-break .*; got both",
            ):
                parse("labels: red-develop, silent-break, rocke: platform")
            with self.assertRaisesRegex(ValueError, "got neither"):
                parse("labels: rocke: platform")  # never inserted silently any more
            with self.assertRaisesRegex(
                ValueError,
                "assigned to the author of the breaking PR: add `assign: LOGIN`",
            ):
                parse("labels: silent-break, rocke: platform")
            with self.assertRaisesRegex(ValueError, "not GitHub logins: no way"):
                parse("labels: silent-break\nassign: no way")
            f = parse("labels: silent-break, rocke: platform\nassign: @dev-k, dev2")
            self.assertEqual(
                (f["kind"], f["assign"], f["labels"]),
                (
                    breakage.STABILITY,
                    ["dev-k", "dev2"],
                    ["silent-break", "rocke: platform"],
                ),
            )
            renamed = dict(
                CFG,
                incident_issue={"labels": ["develop-break"], "stability_label": "stab"},
            )
            self.assertEqual(
                parse("labels: develop-break", renamed)["kind"], breakage.RED
            )

    def test_fetch_open_reads_both_labels_and_merges_by_number(self):
        def rest(path, jq=None, paginate=False):
            if "labels=red-develop" in path:
                return [
                    {"n": 1, "labels": ["red-develop"]},
                    {"n": 3, "labels": ["red-develop", "silent-break"]},
                ]
            self.assertIn("labels=silent-break", path)
            return [
                {"n": 2, "labels": ["silent-break"]},
                {"n": 3, "labels": ["red-develop", "silent-break"]},
            ]

        with mock.patch.object(gh, "rest", rest):
            got = breakage.fetch_open(CFG)
        self.assertEqual(
            [(i["n"], i["kind"]) for i in got],
            [(3, "ci_red"), (2, "stability"), (1, "ci_red")],
        )  # both labels: CI is red

    def test_poll_names_the_kind_and_keeps_what_the_sweep_needs(self):
        st = empty_state()
        old = issue(7777, breakage.RED, title="[rocke] gdn pytest times out")
        with mock.patch.object(breakage, "fetch_open", lambda cfg: [dict(old)]):
            self.assertEqual(breakage.poll(CFG, st), [])
        st["post_merge"]["open"][0].pop(
            "kind"
        )  # an older record, stored without a kind
        with mock.patch.object(breakage, "fetch_open", lambda cfg: [issue()]):
            lines = breakage.poll(CFG, st)
        self.assertTrue(
            lines[0].startswith(
                "BREAKAGE_OPEN #11111 silent-break (me): [rocke] test_selector"
            ),
            lines,
        )
        self.assertEqual(
            lines[1], "BREAKAGE_CLOSED #7777 red-develop: [rocke] gdn pytest times out"
        )
        stored = st["post_merge"]["open"][0]
        self.assertNotIn("body", stored)
        self.assertEqual(stored["kind"], breakage.STABILITY)
        self.assertEqual(
            stored["scan"],
            {
                "files": [
                    "library/kernels/gfx950/attention_tiled_2d.py",
                    "library/tests/test_selector_parity.py",
                    "platform/CMakeLists.txt",
                    "test_selector_parity.py",
                ],  # not the sibling test_conv_abi.py
                "tests": [
                    "library/tests/test_selector_parity.py",
                    "test_selector_parity.py",
                ],
                "names": [
                    "TestKvAddrLimit",
                    "test_every_tiled_flag_is_classified_for_kv_addressing",
                    "test_selector_parity",
                ],
                "refs": [1111, 4567],
            },
        )

    def test_issue_file_assigns_the_author_and_names_the_kind(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            path = write(
                d,
                "---\ntitle: [rocke] test_x fails on develop since #1111\nlabels: silent-break, rocke: "
                f"platform\nassign: dev-k\n---\n{REPORT}\n\nculprit #1111\n",
            )
            missing = mock.Mock(side_effect=gh.GhError("gh api: Not Found (HTTP 404)"))
            with mock.patch.object(gh, "rest", missing):
                with self.assertRaisesRegex(
                    post.PostRefused,
                    "no silent-break label: create the label first "
                    r"\(`rk label-create silent-break",
                ):
                    post.issue_file(CFG, empty_state(), store, path, new=True)
            done = mock.Mock(
                returncode=0, stdout="https://github.com/ROCm/r/issues/99\n", stderr=""
            )
            dup = [(11111, "selector parity", ["#1111"], breakage.STABILITY)]
            with mock.patch.object(
                gh, "rest", return_value="silent-break"
            ), mock.patch.object(
                breakage, "duplicates", lambda cfg, t, b: dup
            ), mock.patch.object(
                post.subprocess, "run", return_value=done
            ) as run:
                with self.assertRaisesRegex(
                    post.PostRefused,
                    "possible duplicate of open silent-break #11111",
                ):
                    post.issue_file(CFG, empty_state(), store, path)
                post.issue_file(CFG, empty_state(), store, path, new=True)
                with self.assertRaisesRegex(post.PostRefused, "already filed"):
                    post.issue_file(CFG, empty_state(), store, path, new=True)
            args = run.call_args[0][0]
            self.assertEqual(
                args[args.index("--label") :],
                [
                    "--label",
                    "silent-break",
                    "--label",
                    "rocke: platform",
                    "--assignee",
                    "dev-k",
                ],
            )
            self.assertEqual(
                store.ledger()[-1]["note"],
                "silent-break: [rocke] test_x fails on develop since #1111",
            )
            self.assertIn(
                "silent-break issue #99: [rocke] test_x fails on develop since #1111, assigned to "
                "dev-k",
                store.journal_tail(1)[0],
            )
            store.ledger_add(
                "issue", 98, "-", "u", note="[rocke] y fails"
            )  # an older note: the title alone
            again = write(
                d,
                f"---\ntitle: [rocke] y fails\nlabels: red-develop\n---\n{REPORT}\n",
            )
            with mock.patch.object(gh, "rest", return_value="red-develop"):
                with self.assertRaisesRegex(post.PostRefused, "already filed"):
                    post.issue_file(CFG, empty_state(), store, again, new=True)

    def test_issue_relabel_swaps_the_two_labels_once(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["post_merge"] = {
                "at": now(),
                "open": [
                    {
                        "n": 12345,
                        "kind": "ci_red",
                        "label": guard_labels.RED,
                        "labels": ["red-develop"],
                    }
                ],
            }
            live = {
                "state": "open",
                "labels": ["red-develop", "rocke: platform"],
                "assignees": [],
            }

            def rest(path, jq=None, paginate=False):
                if "/labels/" in path:
                    if path.endswith("/labels/silent-break"):
                        return "silent-break"
                    raise gh.GhError("gh api: Not Found (HTTP 404)")
                return dict(live)

            ran = []
            with mock.patch.object(gh, "rest", rest), mock.patch.object(
                post.subprocess,
                "run",
                lambda args, **kw: ran.append(args)
                or mock.Mock(returncode=0, stdout="", stderr=""),
            ):
                with self.assertRaisesRegex(post.PostRefused, "not a GitHub login"):
                    post.issue_relabel(CFG, st, store, 12345, "stability", "a b")
                out = post.issue_relabel(CFG, st, store, 12345, "stability", "@dev-k")
                self.assertEqual(
                    ran,
                    [
                        [
                            "gh",
                            "issue",
                            "edit",
                            "12345",
                            "--repo",
                            "ROCm/r",
                            "--add-label",
                            "silent-break",
                            "--remove-label",
                            "red-develop",
                            "--add-assignee",
                            "dev-k",
                        ]
                    ],
                )
                self.assertEqual(
                    out,
                    "#12345 relabelled silent-break (was red-develop), assigned to dev-k",
                )
                self.assertEqual(
                    st["post_merge"]["open"][0]["kind"], breakage.STABILITY
                )
                self.assertEqual(  # the reports name it by its label until the next poll
                    report.develop_issues(CFG, st["post_merge"]["open"]),
                    f"{guard_labels.STABILITY} #12345",
                )
                self.assertTrue(
                    store.ledger_has("issue-relabel", 12345, breakage.STABILITY)
                )
                with self.assertRaisesRegex(
                    post.PostRefused, "already relabelled silent-break"
                ):
                    post.issue_relabel(CFG, st, store, 12345, "stability")
                live["labels"] = ["silent-break"]
                with self.assertRaisesRegex(
                    post.PostRefused, "already has the silent-break label"
                ):
                    post.issue_relabel(CFG, st, store, 22222, "stability")
                live["state"] = "closed"
                with self.assertRaisesRegex(post.PostRefused, "#23456 is closed"):
                    post.issue_relabel(CFG, st, store, 23456, "stability")
                with self.assertRaisesRegex(
                    post.PostRefused, "no red-develop label: create the label first"
                ):
                    post.issue_relabel(CFG, st, store, 14655, "ci_red")
                for _ in range(10):
                    store.ledger_add("post", 1, "h", "u")
                live.update(state="open", labels=["red-develop"])
                with self.assertRaisesRegex(post.PostRefused, "rate limit"):
                    post.issue_relabel(CFG, st, store, 14656, "stability")
            self.assertEqual(len(ran), 1)

    def test_ci_red_goes_with_post_merge(self):
        """#7777 (red-develop, needs-revert) moves to silent-break once CI stops running its test. needs-revert goes with
        red-develop, and comes back with it; an issue file cannot pair it with silent-break.
        """
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            live = {
                "state": "open",
                "labels": ["red-develop", "needs-revert", "rocke: platform"],
                "assignees": ["dev-s"],
            }

            def rest(path, jq=None, paginate=False):
                return path.rsplit("/", 1)[-1] if "/labels/" in path else dict(live)

            ran = []
            with mock.patch.object(gh, "rest", rest), mock.patch.object(
                post.subprocess,
                "run",
                lambda args, **kw: ran.append(args)
                or mock.Mock(returncode=0, stdout="", stderr=""),
            ):
                out = post.issue_relabel(CFG, st, store, 7777, "stability")
                live["labels"] = ["silent-break", "rocke: platform"]
                post.issue_relabel(CFG, st, store, 7777, "ci_red")
            self.assertEqual(
                out, "#7777 relabelled silent-break (was red-develop, needs-revert)"
            )
            self.assertEqual(
                ran,
                [
                    [
                        "gh",
                        "issue",
                        "edit",
                        "7777",
                        "--repo",
                        "ROCm/r",
                        "--add-label",
                        "silent-break",
                        "--remove-label",
                        "red-develop",
                        "--remove-label",
                        "needs-revert",
                    ],
                    [
                        "gh",
                        "issue",
                        "edit",
                        "7777",
                        "--repo",
                        "ROCm/r",
                        "--add-label",
                        "red-develop",
                        "--add-label",
                        "needs-revert",
                        "--remove-label",
                        "silent-break",
                    ],
                ],
            )
            f = pathlib.Path(d) / "issue.md"
            f.write_text(
                f"---\ntitle: [rocke] x fails\nlabels: silent-break, needs-revert\nassign: dev1\n---\n"
                f"{REPORT}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "needs-revert goes with red-develop only"
            ):
                breakage.parse_file(str(f), CFG)

    def test_the_teams_alert_is_for_ci_red_issues_only(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                CFG,
                teams={
                    "channel": "Team channel",
                    "webhook_file": str(pathlib.Path(d) / "hook"),
                },
            )
            (pathlib.Path(d) / "hook").write_text("https://example.invalid/hook\n")
            st = empty_state()
            got = {
                "number": 12345,
                "title": "t",
                "state": "open",
                "html_url": "u",
                "body": "",
                "labels": ["silent-break"],
            }
            with mock.patch.object(gh, "rest", lambda path, jq=None: dict(got)):
                with self.assertRaisesRegex(
                    post.PostRefused,
                    "lacks the red-develop label: a silent-break issue goes "
                    "to the breaking PR's author, not the channel",
                ):
                    post.teams_breakage(cfg, st, store, 12345)
                got["labels"] = [
                    "red-develop"
                ]  # relabelled since the fix was recorded under its old kind
                st["develop_fixes"] = {"7": ["silent-break #12345"]}
                with self.assertRaisesRegex(
                    post.PostRefused, r"a fix is planned \(#7\)"
                ):
                    post.teams_breakage(cfg, st, store, 12345)

    def test_develop_fixes_name_the_kind(self):
        st = empty_state()
        st["post_merge"] = {
            "at": now(),
            "fixes": {"12345": [3]},
            "open": [
                {"n": 7777, "kind": "ci_red", "label": guard_labels.RED},
                {"n": 12345, "kind": "stability", "label": guard_labels.STABILITY},
                {"n": 6666, "label": guard_labels.RED},
            ],
        }  # no kind: a CI-red one
        st["fix_links"] = {"12345": [4], "999": [5]}  # #999 is not open
        st["prs"] = {str(n): pr(n) for n in range(1, 7)}
        for n, i in ((1, 7777), (2, 12345), (6, 6666)):
            st["prs"][str(n)]["closes"] = [i]
        self.assertEqual(
            queue.refresh_develop_fixes(st, []),
            {
                "1": ["red-develop #7777"],
                "2": ["silent-break #12345"],
                "3": ["silent-break #12345"],
                "4": ["silent-break #12345"],
                "6": ["red-develop #6666"],
            },
        )

    def test_the_board_and_the_page_list_both_kinds(self):
        self.assertEqual(
            report.develop_issues(
                CFG,
                [
                    {"n": 8888, "kind": "stability", "label": guard_labels.STABILITY},
                    {"n": 7777, "label": guard_labels.RED},
                    {"n": 9999, "kind": "stability", "label": guard_labels.STABILITY},
                ],
            ),
            "red-develop (CI red) #7777; silent-break #8888, #9999",
        )
        self.assertEqual(report.develop_issues(CFG, []), "none")
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                CFG,
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            st["post_merge"] = {
                "at": now(),
                "open": [
                    {
                        "n": 12345,
                        "title": "a",
                        "rocke": True,
                        "kind": "stability",
                        "label": guard_labels.STABILITY,
                        "author": "b",
                        "assignees": [],
                        "created": now(),
                        "labels": [],
                    },
                    {
                        "n": 7777,
                        "title": "b",
                        "rocke": True,
                        "kind": "ci_red",
                        "label": guard_labels.RED,
                        "author": "b",
                        "assignees": [],
                        "created": now(),
                        "labels": [],
                    },
                ],
            }
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn(
            "rocKE develop issues open: red-develop (CI red) #7777; silent-break #12345",
            board["md"],
        )
        self.assertIn("rocKE issues: 1 red-develop (CI red), 1 silent-break", page)
        self.assertIn("<th>Kind</th>", page)
        self.assertIn(">red-develop (CI red)</td>", page)


class TestStoredKinds(unittest.TestCase):
    """An entry of st["post_merge"] stored with its label's name as its kind, and no label of its own, is read by
    the labels stored with it until the next poll stores it again."""

    def stored(self) -> list[dict]:
        common = {
            "rocke": True,
            "title": "t",
            "author": "a",
            "assignees": [],
            "created": now(),
        }
        return [
            dict(
                common,
                n=7777,
                kind=guard_labels.RED,
                labels=[guard_labels.RED, guard_labels.EXTRA],
            ),
            dict(
                common,
                n=8888,
                kind=guard_labels.STABILITY,
                labels=[guard_labels.STABILITY],
            ),
        ]

    def test_each_reader_keeps_its_kind(self):
        old = self.stored()
        self.assertEqual(
            [breakage.entry_kind(CFG, i) for i in old],
            [breakage.RED, breakage.STABILITY],
        )
        self.assertEqual(
            report.develop_issues(CFG, old),
            f"{guard_labels.RED} (CI red) #7777; {guard_labels.STABILITY} #8888",
        )
        st = empty_state()
        st["post_merge"] = {"at": now(), "open": old}
        self.assertEqual(breakage.kind_in_state(CFG, st, 8888), breakage.STABILITY)
        with tempfile.TemporaryDirectory() as d:
            status = report.develop_status(CFG, st, Store(pathlib.Path(d)))
        self.assertEqual(status["word"], report.BROKEN)
        self.assertIn(f"#7777 ({guard_labels.RED}, CI red)", status["why"])
        st["develop_fixes"] = {
            "1": [f"{guard_labels.STABILITY} #8888"],
            "2": [queue.USER_PRIORITY + " x"],
            "3": [f"{guard_labels.RED} #7777"],
        }  # the fix of the CI-red issue lands first
        self.assertEqual(mergeplan._priority_order(CFG, st, [1, 2, 3], {}), [3, 2, 1])


def blank(red: bool = True, stability: bool = True) -> dict:
    """CFG with the develop guard's CI-red label, its stability label, or both blank in the settings."""
    return dict(
        CFG,
        incident_issue={
            "labels": ["" if red else guard_labels.RED, guard_labels.EXTRA],
            "stability_label": "" if stability else guard_labels.STABILITY,
        },
    )


class TestBlankLabels(unittest.TestCase):
    """A develop guard's label the settings leave blank (the guard off, or on and not live): no issue is filed or
    read by it."""

    def test_an_issue_file_is_refused_while_a_label_is_blank(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(
                d,
                f"---\ntitle: [develop] x\nlabels: {guard_labels.RED}\n---\n{REPORT}\n",
            )
            guard = "features.merge_queue.develop_guard"
            for cfg, said in (
                (blank(red=False), f"{guard}.stability_label is blank: set it"),
                (blank(stability=False), f"{guard}.ci_red_label is blank: set it"),
                (
                    blank(),
                    f"{guard}.ci_red_label and {guard}.stability_label are blank: set them",
                ),
            ):
                with self.subTest(labels=breakage.labels(cfg)):
                    with self.assertRaises(ValueError) as refused:
                        breakage.parse_file(path, cfg)
                    self.assertEqual(
                        str(refused.exception),
                        f"{said} in your deploy settings file, then rkmon confirm and rkmon apply",
                    )
        # the host's command as the operator types it, which the container's wrapper passes in
        with mock.patch.dict("os.environ", {"RKMON_WRAPPER": "./rkmon-team"}):
            self.assertTrue(
                breakage.blank_labels([breakage.RED]).endswith(
                    "then ./rkmon-team confirm and ./rkmon-team apply"
                )
            )

    def test_the_team_alert_says_a_blank_ci_red_label_is_not_set(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            (pathlib.Path(d) / "hook").write_text("https://example.invalid/hook\n")
            cfg = dict(
                blank(stability=False),
                teams={"webhook_file": str(pathlib.Path(d) / "hook")},
            )
            got = {
                "number": 12345,
                "title": "t",
                "state": "open",
                "html_url": "u",
                "body": "",
                "labels": [guard_labels.STABILITY],
            }
            with mock.patch.object(gh, "rest", lambda path, jq=None: dict(got)):
                with self.assertRaisesRegex(
                    post.PostRefused,
                    "^no issue counts as CI-red, so none goes to the channel: "
                    "features.merge_queue.develop_guard.ci_red_label is blank: set it",
                ):
                    post.teams_breakage(cfg, empty_state(), store, 12345)

    def test_a_blank_label_is_never_read(self):
        asked = []

        def rest(path, jq=None, paginate=False):
            asked.append(path)
            return []

        with mock.patch.object(gh, "rest", rest):
            breakage.fetch_open(blank(stability=False))
            self.assertEqual(len(asked), 1, asked)
            self.assertIn(f"?labels={guard_labels.STABILITY}&", asked[0])
            self.assertEqual(breakage.fetch_open(blank()), [])
        self.assertEqual(len(asked), 1, asked)
        with mock.patch.object(breakage.snapshot, "search_pages") as pages:
            self.assertEqual(breakage._issues(blank(), empty_state(), None), [])
        pages.assert_not_called()

    def test_a_kind_with_no_label_is_named_by_its_word(self):
        cfg = blank()
        self.assertEqual(
            [breakage.kind_label(cfg, k) for k in (breakage.RED, breakage.STABILITY)],
            ["CI-red", "stability"],
        )
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                cfg,
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn(
            "breakage tracker: CI-red when CI is red, stability when CI does not see it",
            page,
        )
        self.assertIn("rocKE issues: 0 CI-red (CI red), 0 stability", page)

    def test_an_extra_label_never_counts_as_ci_red(self):
        cfg = blank()
        self.assertEqual(
            breakage.labels(cfg), {breakage.RED: "", breakage.STABILITY: ""}
        )
        self.assertEqual(breakage.extra_labels(cfg), [guard_labels.EXTRA])


class TestRenamedLabels(unittest.TestCase):
    def test_the_reports_and_the_dashboards_print_the_labels_the_settings_name(self):
        """The settings name labels of their own: the poll reads the issues by them, and the board, the admin
        dashboard and the team dashboard print them."""
        renamed = dict(
            CFG,
            incident_issue={
                "labels": ["broke-ci", "revert-me"],
                "stability_label": "unseen-break",
            },
        )

        def found(n: int, labels: list) -> dict:
            """Issue N as the query for open develop issues gives it: no kind, no label of the monitor's."""
            return {
                k: v
                for k, v in issue(n, labels=labels).items()
                if k not in ("kind", "label")
            }

        def rest(path, jq=None, paginate=False):
            if "?labels=broke-ci&" in path:
                return [found(7777, ["broke-ci", "revert-me", "rocke: platform"])]
            self.assertIn("?labels=unseen-break&", path)
            return [found(12345, ["unseen-break", "rocke: platform"])]

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            cfg = dict(
                renamed,
                paths={
                    "working_dir": d,
                    "html": str(pathlib.Path(d) / "x.html"),
                    "daily_dir": d,
                },
            )
            st = empty_state()
            st["develop"]["sha"] = "d" * 40
            with mock.patch.object(gh, "rest", rest):
                breakage.poll(cfg, st)
            board = report.board(cfg, st, store, [], [])
            page = pathlib.Path(webdash.write(cfg, st, store, [], [], board)).read_text(
                encoding="utf-8"
            )
        self.assertIn(
            "rocKE develop issues open: broke-ci (CI red) #7777; unseen-break #12345",
            board["md"],
        )
        self.assertIn("rocKE issues: 1 broke-ci (CI red), 1 unseen-break", page)
        self.assertIn(">broke-ci (CI red)</td>", page)
        for label in guard_labels.INCIDENT_ISSUE["labels"] + [guard_labels.STABILITY]:
            self.assertNotIn(label, board["md"] + page)


SEL, CMAKE = "rk/library/tests/test_selector_parity.py", "rk/platform/CMakeLists.txt"


class TestRepositoryNames(unittest.TestCase):
    """Which develop issues concern the watched folder, and the test names that make two issues one breakage, come
    from the settings (incident_issue.title_pattern, test_name_pattern, umbrella_test_names), which the repository
    defaults set; the code names none of its own."""

    def cfg(self, **names) -> dict:
        return dict(
            CFG,
            scope={"labels": ["area: widgets"]},
            incident_issue=dict(guard_labels.INCIDENT_ISSUE, **names),
        )

    def test_an_issue_concerns_the_folder_by_its_labels_or_a_title_the_settings_name(
        self,
    ):
        cfg = self.cfg(title_pattern="widgets|gizmo-kit")
        for title, labels, concerns in (
            ("[develop] Widgets fail to build", [], True),
            ("[develop] the GIZMO-KIT shard times out", [], True),
            ("[develop] something else fails", ["area: widgets"], True),
            ("[develop] something else fails", [], False),
            ("[develop] rocke fails", [], False),
        ):
            with self.subTest(title=title, labels=labels):
                self.assertEqual(
                    breakage.relevant(cfg, {"title": title, "labels": labels}),
                    concerns,
                )
        # blank: the scope's labels alone
        for cfg in (self.cfg(title_pattern=""), self.cfg()):
            self.assertFalse(
                breakage.relevant(cfg, {"title": "[develop] Widgets", "labels": []})
            )
            self.assertTrue(
                breakage.relevant(cfg, {"title": "x", "labels": ["area: widgets"]})
            )

    def test_two_issues_are_one_breakage_by_the_test_names_the_settings_name(self):
        cfg = self.cfg(
            test_name_pattern=r"\bwidget_\w{4,}", umbrella_test_names=["widget_all"]
        )
        self.assertEqual(
            breakage.overlaps(
                cfg,
                "widget_render fails, so widget_all is red since #1111",
                "widget_render and widget_all fail after #1111",
            ),
            ["#1111", "widget_render"],
        )
        self.assertEqual(
            breakage.overlaps(cfg, "widget_all fails", "widget_all times out"), []
        )
        # blank, as the package's generic value is: test functions and test classes alone, no repository's own
        self.assertEqual(
            breakage.overlaps(
                self.cfg(test_name_pattern=""),
                "test_widget_render and widget_render",
                "test_widget_render and widget_render",
            ),
            ["test_widget_render"],
        )
        generic = self.cfg()
        self.assertEqual(
            breakage.overlaps(
                generic,
                "test_attention_parity and rocke_differential_golden fail; TestWidgetRender too",
                "test_attention_parity, rocke_differential_golden, TestWidgetRender",
            ),
            ["TestWidgetRender", "test_attention_parity"],
        )


class TestFixSweep(unittest.TestCase):
    FILES = {
        1: [
            SEL,
            "rk/library/kernels/x.py",
        ],  # b: a PR titled as a fix changes the failing test's file
        2: [
            "rk/library/kernels/gfx950/attention_tiled_2d.py"
        ],  # a: its body mentions the issue
        3: [CMAKE],  # c: names TestKvAddrLimit and changes a build file the issue names
        4: [CMAKE],  # changes a file the issue names, names nothing
        5: [SEL],  # a draft
        6: [SEL],  # already closes the issue
        8: [SEL],  # dismissed
        9: [
            "rk/library/kernels/other.py"
        ],  # names a failing test in its title, changes nothing the issue names
        10: [
            "rk/library/kernels/other.py"
        ],  # its body mentions the issue, but nothing else matches: never read
        11: [SEL],  # a feature PR that shares the test file
        12: [SEL],  # names only the test's module, and changes only that test file
        1111: [SEL],  # the culprit
        4567: [SEL],
    }  # the issue names it: it reported the failure
    BODIES = {
        1: "Classifies the flag.",
        2: "Follow-up to #11111: classify use_additive_bias.",
        3: "Adds a CPU lane; TestKvAddrLimit passes.",
        4: "Bumps a lane.",
        9: "Nothing related.",
        10: "See #11111.",
        11: "New tiles.",
        12: "Updates test_selector_parity.py for the new flags.",
        4567: "Known failures on develop: TestKvAddrLimit.",
    }

    def setUp(self):
        i = issue()
        i.update(rocke=True, scan=breakage.scan(i))
        self.st = empty_state()
        self.st["post_merge"] = {
            "at": now(),
            "open": [{k: v for k, v in i.items() if k != "body"}],
        }
        self.st["prs"] = {str(n): pr(n) for n in self.FILES}
        self.st["prs"]["1"][
            "title"
        ] = "fix(rocke): classify use_additive_bias for KV addressing"
        self.st["prs"]["5"]["draft"] = True
        self.st["prs"]["6"]["closes"] = [11111]
        self.st["prs"]["9"]["title"] = "Make TestKvAddrLimit faster"
        self.st["prs"]["11"]["title"] = "feat(rocke): new tiles"
        self.st["fix_dismissed"] = {"11111": [8]}
        self.git, self.reads = [], []

    def sweep(self) -> list[str]:
        def changed(repo, base, ref):
            self.git.append(int(ref.rsplit("pr", 1)[1]))
            return self.FILES[self.git[-1]]

        def rest(path, jq=None, paginate=False):
            self.reads.append(int(path.rsplit("/", 1)[1]))
            return {"body": self.BODIES.get(self.reads[-1], "")}

        with mock.patch.object(
            breakage.gitops, "changed_files", changed
        ), mock.patch.object(gh, "rest", rest), mock.patch.object(
            queue, "ensure_pr_ref", lambda cfg, r: f"refs/remotes/origin/pr{r['n']}"
        ):
            return breakage.fix_sweep(CFG, self.st)

    def test_signals_and_what_is_left_out(self):
        lines = self.sweep()
        self.assertEqual(
            [ln.split(":")[0] for ln in lines],
            [
                "FIX_CANDIDATE #1 -> #11111 (silent-break)",
                "FIX_CANDIDATE #2 -> #11111 (silent-break)",
                "FIX_CANDIDATE #3 -> #11111 (silent-break)",
            ],
        )
        self.assertEqual(
            lines[0],
            "FIX_CANDIDATE #1 -> #11111 (silent-break): its title reads as a fix and it "
            "changes library/tests/test_selector_parity.py. Verify it, then `rk fix-link 11111 1 "
            "[--evidence FILE]`; or `rk fix-dismiss 11111 1`",
        )
        self.assertIn("): its body mentions #11111. Verify", lines[1])
        self.assertIn(
            "): it names TestKvAddrLimit and changes platform/CMakeLists.txt. ",
            lines[2],
        )
        self.assertEqual(
            self.reads, [1, 2, 3, 4, 9, 11, 12, 1111, 4567]
        )  # only where a title or file matches
        self.assertEqual(
            sorted(self.git), [1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 1111, 4567]
        )  # all open but the draft
        i, named = self.st["post_merge"]["open"][0], self.st["prs"]["4567"]
        sc = {
            "hits": {"11111": [SEL]},
            "body": {"refs": [], "names": ["TestKvAddrLimit"]},
        }
        self.assertEqual(
            breakage.signals(CFG, i, named, sc), []
        )  # the issue names it: only a mention counts
        sc["body"]["refs"] = [11111]
        self.assertEqual(
            breakage.signals(CFG, i, named, sc), ["its body mentions #11111"]
        )

    def test_known_failures_named_in_a_body_are_no_fix_signal(self):
        """A test plan that lists other PRs' files as failing identically on develop names no fix."""
        body = (
            "- **Host tests:** results match develop. The 26 remaining failures in `test_strided_kv_lds.py`, "
            "`test_strided_kv_decode.py` and `test_selector_parity.py` fail identically on develop.\n"
            "- [x] Library host tests - Status: Passed (3 files fail identically on develop)\n"
            "This fixes test_every_tiled_flag_is_classified_for_kv_addressing, which fails on develop since #1111."
        )
        self.assertEqual(
            breakage.fix_names(body),
            ["test_every_tiled_flag_is_classified_for_kv_addressing"],
        )
        self.assertEqual(
            breakage.fix_names("Known failures: test_a_b_c and TestKvAddrLimit."), []
        )
        self.assertEqual(breakage.fix_names(None), [])

    def test_files_and_bodies_are_read_once_per_head(self):
        first = self.sweep()
        self.git, self.reads = [], []
        self.assertEqual(self.sweep(), first)
        self.assertEqual((self.git, self.reads), ([], []))
        self.st["prs"]["1"]["head"] = "e" * 40  # a push
        self.assertEqual(self.sweep(), first)
        self.assertEqual((self.git, self.reads), ([1], [1]))
        new = issue(
            14700,
            title="[rocke] other fails",
            body="`library/tests/test_other.py` fails",
        )
        new.update(rocke=True, scan=breakage.scan(new))
        self.st["post_merge"]["open"].append(
            new
        )  # a new issue: files are matched again, bodies are kept
        self.git, self.reads = [], []
        self.sweep()
        self.assertEqual((len(self.git), self.reads), (12, []))
        self.st["prs"]["2"]["status"] = "merged"
        self.st["fix_links"] = {"11111": [1]}
        self.assertEqual(
            [ln.split(":")[0][14:] for ln in self.sweep()],
            ["#3 -> #11111 (silent-break)"],
        )
        self.assertNotIn("fix_scan", self.st["prs"]["2"])

    def test_dismiss_stops_the_line_and_undoes_a_link(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            self.assertIn("dismissed", breakage.fix_dismiss(self.st, store, 11111, 1))
            self.assertNotIn("#1 ->", " ".join(self.sweep()))
            self.st["fix_links"] = {"11111": [2]}
            self.assertIn("2", queue.refresh_develop_fixes(self.st, []))
            self.assertIn("unlinked", breakage.fix_dismiss(self.st, store, 11111, 2))
            self.assertEqual(
                (self.st["fix_links"], sorted(self.st["develop_fixes"])), ({}, ["6"])
            )  # #6 closes it

    def test_scan_matches_paths_from_another_root_and_bare_node_ids(self):
        s = breakage.scan(
            {
                "title": "[rocke] rocke_library_gdn_pytest times out",
                "body": (
                    "60: tests\\library\\tests\\test_gdn_decode_spec.py::TestEmission::test_registered_tiles_compile +Timeout+"
                ),
            }
        )
        self.assertEqual(s["tests"], ["tests/library/tests/test_gdn_decode_spec.py"])
        self.assertEqual(
            s["names"],
            ["TestEmission", "test_gdn_decode_spec", "test_registered_tiles_compile"],
        )
        self.assertTrue(
            breakage.same_file(
                s["tests"][0], "rk/library/tests/test_gdn_decode_spec.py"
            )
        )
        self.assertFalse(
            breakage.same_file(
                "library/tests/test_registry.py", "rk/platform/tests/test_registry.py"
            )
        )
        self.assertTrue(
            breakage.same_file("platform/CMakeLists.txt", "rk/platform/CMakeLists.txt")
        )
        self.assertFalse(
            breakage.same_file("platform/CMakeLists.txt", "rk/library/CMakeLists.txt")
        )
        self.assertTrue(
            breakage.same_file("test_conv_abi.py", "rk/library/tests/test_conv_abi.py")
        )
        self.assertEqual(
            breakage.mentions(
                "see #11111, ROCm/example#12345, "
                "https://github.com/ROCm/example/issues/22222; not x#23456"
            ),
            {11111, 12345, 22222},
        )


class TestFixLink(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["post_merge"] = {"at": now(), "open": [dict(issue(), rocke=True)]}
        self.st["prs"] = {
            "5": pr(5, author="dev-l", title="Classify use_additive_bias")
        }
        self.live = {"state": "open", "draft": False}
        self.iss = {
            "title": "selector parity fails",
            "state": "open",
            "labels": ["silent-break"],
            "pr": False,
        }
        self.ran, self.bodies, self.fail = [], [], False

    def tearDown(self):
        self.d.cleanup()

    def link(self, evidence: str | None = None) -> str:
        def rest(path, jq=None, paginate=False):
            return dict(self.live) if "/pulls/" in path else dict(self.iss)

        def run(args, **kw):
            self.ran.append(args[:3])
            self.bodies.append(
                pathlib.Path(args[args.index("--body-file") + 1]).read_text(
                    encoding="utf-8"
                )
            )
            if args[1] == "pr" and self.fail:
                return mock.Mock(returncode=1, stdout="", stderr="boom")
            url = "https://github.com/ROCm/r/" + (
                "issues/11111#issuecomment-1"
                if args[1] == "issue"
                else "pull/5#issuecomment-2"
            )
            return mock.Mock(returncode=0, stdout=url + "\n", stderr="")

        with mock.patch.object(gh, "rest", rest), mock.patch.object(
            post.subprocess, "run", run
        ):
            return post.fix_link(CFG, self.st, self.store, 11111, 5, evidence)

    def test_guards(self):
        for change, why in (
            (lambda: self.st["prs"].pop("5"), "#5 is not a tracked open PR"),
            (
                lambda: self.st["prs"]["5"].update(closes=[11111]),
                "already closes #11111",
            ),
            (lambda: self.live.update(draft=True), "#5 is a draft"),
            (
                lambda: self.iss.update(state="closed"),
                "#11111 is not an open develop issue",
            ),
            (lambda: self.iss.update(labels=["bug"]), "is not an open develop issue"),
        ):
            self.setUp()
            change()
            with self.assertRaisesRegex(post.PostRefused, why):
                self.link()
            self.assertEqual((self.ran, self.st.get("fix_links") or {}), ([], {}))
        ev = pathlib.Path(self.d.name) / "ev.md"
        ev.write_text(
            "Verified on the internal dashboard https://wiki.internal.example/x\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(post.PostRefused, "non-public host"):
            self.link(str(ev))
        self.assertEqual(self.ran, [])

    def test_comment_notice_and_a_develop_fix_at_once(self):
        ev = pathlib.Path(self.d.name) / "ev.md"
        ev.write_text(
            "Ran the failing test on the PR head: it passes.\n", encoding="utf-8"
        )
        out = self.link(str(ev))
        self.assertEqual(
            self.ran, [["gh", "issue", "comment"], ["gh", "pr", "comment"]]
        )
        comment, notice = self.bodies
        self.assertEqual(comment.splitlines()[0], REPORT)
        self.assertIn(
            "**Fix candidate:** #5\n\nRan the failing test on the PR head: it passes.",
            comment,
        )
        self.assertTrue(
            comment.endswith("\n\nThis monitor reviews and merges it first.\n")
        )
        self.assertEqual(
            notice.splitlines()[0],
            post.notice_line(CFG, "fix"),
        )
        self.assertIn(
            "Hi @dev-l, this PR looks like the fix for #11111 (selector parity fails).",
            notice,
        )
        self.assertIn('Please add "Fixes #11111" to the description', notice)
        self.assertIn("moved this PR to the top of its review and merge queues", notice)
        self.assertTrue(
            out.startswith(
                "#5 is a develop fix (silent-break #11111); issue comment https://"
            ),
            out,
        )
        self.assertEqual(self.st["fix_links"], {"11111": [5]})
        self.assertEqual(self.st["develop_fixes"], {"5": ["silent-break #11111"]})
        self.assertTrue(self.store.ledger_has("issue-comment", 11111, "fix-5"))
        self.assertTrue(self.store.ledger_has("notice-fix", 5, "issue-11111"))
        self.assertTrue(
            any(
                "comment on silent-break issue #11111 (fix-5)" in ln
                for ln in self.store.journal_tail()
            )
        )
        with self.assertRaisesRegex(post.PostRefused, "already linked"):
            self.link()

    def test_the_users_own_pr_gets_the_issue_comment_and_no_notice(self):
        """The operator's own PR gets the issue comment and no PR notice, which would ask its author (the same
        account) to edit it."""
        self.st["prs"]["5"]["author"] = CFG["operator"]["github_login"]
        out = self.link()
        self.assertEqual(self.ran, [["gh", "issue", "comment"]])
        self.assertIn(
            'no PR notice: #5 is your own PR, so add "Fixes #11111" to its description yourself',
            out,
        )
        self.assertEqual(self.st["fix_links"], {"11111": [5]})
        with self.assertRaisesRegex(post.PostRefused, "already linked"):
            self.link()

    def test_a_failed_write_keeps_the_link_and_a_rerun_finishes(self):
        self.fail = True
        with self.assertRaisesRegex(post.PostRefused, "gh pr comment failed: boom"):
            self.link()
        self.assertEqual(
            self.st["fix_links"], {"11111": [5]}
        )  # fast-tracked although the notice failed
        self.fail, self.ran = False, []
        self.assertIn("PR notice https://", self.link())
        self.assertEqual(
            self.ran, [["gh", "pr", "comment"]]
        )  # the issue comment is not repeated

    def test_a_linked_fix_gets_review_p0_and_merge_tier_0(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["post_merge"] = {"at": now(), "open": [dict(issue(), rocke=True)]}
        st["fix_links"] = {
            "11111": [5],
            "999": [6],
        }  # #999 is not an open issue: no develop fix
        st["prs"] = {
            "5": pr(5, size=5000, ours=None),
            "6": pr(6, size=10, ours=None),
            "7": pr(7, size=1),
        }
        self.assertEqual(
            queue.refresh_develop_fixes(st, []), {"5": ["silent-break #11111"]}
        )
        items = [
            i
            for i in plan.make_plan(CFG, st, [], [])
            if i["kind"] in ("FIRST_ROUND", "FOLLOW_UP")
        ]
        self.assertEqual([(i["pr"], i["prio"]) for i in items], [(5, 0), (6, 3)])
        self.assertTrue(
            items[0]["text"].startswith(
                "develop fix (silent-break #11111): first-round review of #5"
            )
        )
        st["prs"]["5"]["ours"] = {"verdict": G, "head": head(5), "at": now()}
        with mock.patch.object(
            queue, "pr_hotspots", lambda cfg, r: r["files_cache"]
        ), mock.patch.object(
            queue, "gate_fresh_merge", lambda cfg, r, local: (queue.PASS, "clean")
        ), mock.patch.object(
            queue, "rocke_tree", lambda cfg, sha: None
        ), mock.patch.object(
            queue, "disruption", lambda cfg, st, cands: {r["n"]: [] for r in cands}
        ), mock.patch.object(
            queue, "abi_lanes", lambda *a: {}
        ), mock.patch.object(
            queue.abi, "gate_abi", lambda r, lanes, waived: (queue.PASS, "")
        ), mock.patch.object(
            queue.batch, "build_batches", lambda *a: []
        ):
            q = queue.build_queue(CFG, st, "d" * 40)
        self.assertEqual(
            [(r["n"], r["tier"], r["position"]) for r in q if r.get("position")],
            [(5, 0, 1), (7, 2, 2)],
        )


# The develop guard's issue repository, another than the watched one.
ELSEWHERE = dict(
    CFG, incident_issue=dict(guard_labels.INCIDENT_ISSUE, repo="ROCm/tracker")
)


class TestIssueRepository(unittest.TestCase):
    """Develop issues live in the develop issues' repository (the develop guard's issue repository, else the watched
    one): they are read, filed, labelled, assigned and commented on there, their labels are checked and created there,
    and a fix's PR and its issue name each other in full across the two repositories."""

    def test_the_rule_and_the_references(self):
        self.assertEqual(breakage.issue_repository(CFG), "ROCm/r")
        blank = dict(CFG, incident_issue=dict(guard_labels.INCIDENT_ISSUE, repo=" "))
        self.assertEqual(breakage.issue_repository(blank), "ROCm/r")
        self.assertEqual(breakage.issue_repository(ELSEWHERE), "ROCm/tracker")
        self.assertEqual(
            (breakage.issue_ref(CFG, 7), breakage.pr_ref(CFG, 5)), ("#7", "#5")
        )
        self.assertEqual(
            (breakage.issue_ref(ELSEWHERE, 7), breakage.pr_ref(ELSEWHERE, 5)),
            ("ROCm/tracker#7", "ROCm/r#5"),
        )
        same = dict(
            CFG, incident_issue=dict(guard_labels.INCIDENT_ISSUE, repo="rocm/R")
        )  # GitHub's names ignore case
        self.assertEqual(breakage.issue_ref(same, 7), "#7")

    def test_every_develop_issue_call_goes_there(self):
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            reads, ran = [], []

            def rest(path, jq=None, paginate=False):
                reads.append(path)
                if "/labels/" in path:
                    if path.endswith("/labels/new-label"):
                        raise gh.GhError("gh api: Not Found (HTTP 404)")
                    return path.rsplit("/", 1)[-1]
                if "?" in path or (jq or "").startswith("[.assignees"):
                    return []
                return {"state": "open", "labels": ["red-develop"], "assignees": []}

            def run(args, **kw):
                ran.append(args)
                url = "https://github.com/ROCm/tracker/issues/9"
                return mock.Mock(returncode=0, stdout=url + "\n", stderr="")

            filed = write(
                d,
                f"---\ntitle: [rocke] test_x fails\nlabels: red-develop\n---\n{REPORT}\n",
            )
            evidence = pathlib.Path(d) / "evidence.md"
            evidence.write_text(f"{REPORT}\n\nbisected\n", encoding="utf-8")
            with mock.patch.object(gh, "rest", rest), mock.patch.object(
                post.subprocess, "run", run
            ):
                breakage.fetch_open(ELSEWHERE)
                post.issue_file(ELSEWHERE, st, store, filed)
                post.issue_comment(ELSEWHERE, st, store, 9, str(evidence), "bisect")
                post.issue_relabel(ELSEWHERE, st, store, 9, "stability", "dev-k")
                post.issue_assign(ELSEWHERE, store, 9, "dev-j")
                out = post.label_create(ELSEWHERE, store, "new-label", "d", "FBCA04")
            self.assertTrue(reads, reads)
            self.assertEqual(
                [p for p in reads if not p.startswith("repos/ROCm/tracker/")], []
            )
            self.assertEqual(
                [(a[1], a[2], a[a.index("--repo") + 1]) for a in ran],
                [
                    ("issue", "create", "ROCm/tracker"),
                    ("issue", "comment", "ROCm/tracker"),
                    ("issue", "edit", "ROCm/tracker"),
                    ("issue", "edit", "ROCm/tracker"),
                    ("label", "create", "ROCm/tracker"),
                ],
            )
            self.assertEqual(out, "label new-label created in ROCm/tracker")

    def test_a_missing_label_is_missing_there(self):
        missing = mock.Mock(side_effect=gh.GhError("gh api: Not Found (HTTP 404)"))
        with mock.patch.object(gh, "rest", missing):
            got = post.label_missing(ELSEWHERE, "red-develop")
        self.assertEqual(
            missing.call_args[0][0], "repos/ROCm/tracker/labels/red-develop"
        )
        self.assertTrue(got[0].startswith("ROCm/tracker has no red-develop label"), got)

    def test_a_fix_link_names_the_pr_and_the_issue_in_full(self):
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            st["post_merge"] = {"at": now(), "open": [dict(issue(), rocke=True)]}
            st["prs"] = {"5": pr(5, author="dev-l"), "6": pr(6, author="me")}
            reads, writes, bodies = [], [], []

            def rest(path, jq=None, paginate=False):
                reads.append(path)
                if "/pulls/" in path:
                    return {"state": "open", "draft": False}
                return {
                    "title": "t",
                    "state": "open",
                    "labels": ["silent-break"],
                    "pr": False,
                }

            def run(args, **kw):
                writes.append((args[1], args[args.index("--repo") + 1]))
                body = pathlib.Path(args[args.index("--body-file") + 1])
                bodies.append(body.read_text(encoding="utf-8"))
                url = "https://github.com/ROCm/tracker/issues/11111#issuecomment-1"
                return mock.Mock(returncode=0, stdout=url + "\n", stderr="")

            with mock.patch.object(gh, "rest", rest), mock.patch.object(
                post.subprocess, "run", run
            ):
                post.fix_link(ELSEWHERE, st, store, 11111, 5)
                own = post.fix_link(ELSEWHERE, st, store, 11111, 6)
        self.assertEqual(
            reads,
            [
                "repos/ROCm/r/pulls/5",
                "repos/ROCm/tracker/issues/11111",
                "repos/ROCm/r/pulls/6",
                "repos/ROCm/tracker/issues/11111",
            ],
        )
        self.assertEqual(
            writes,
            [("issue", "ROCm/tracker"), ("pr", "ROCm/r"), ("issue", "ROCm/tracker")],
        )
        self.assertIn("**Fix candidate:** ROCm/r#5\n", bodies[0])
        self.assertIn(
            "this PR looks like the fix for ROCm/tracker#11111 (t).", bodies[1]
        )
        self.assertIn(
            'Please add "Fixes ROCm/tracker#11111" to the description', bodies[1]
        )
        self.assertIn("**Fix candidate:** ROCm/r#6\n", bodies[2])
        self.assertIn('add "Fixes ROCm/tracker#11111" to its description yourself', own)

    def test_the_teams_alert_reads_the_issue_there(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(
                ELSEWHERE,
                teams={"channel": "c", "webhook_file": str(pathlib.Path(d) / "hook")},
            )
            (pathlib.Path(d) / "hook").write_text("https://example.invalid/hook\n")
            got = {"state": "closed", "labels": ["red-develop"], "body": ""}
            read = mock.Mock(return_value=got)
            with mock.patch.object(gh, "rest", read):
                with self.assertRaisesRegex(post.PostRefused, "#12345 is closed"):
                    post.teams_breakage(
                        cfg, empty_state(), Store(pathlib.Path(d)), 12345
                    )
        self.assertEqual(read.call_args[0][0], "repos/ROCm/tracker/issues/12345")

    def test_the_readers_name_the_issue_repository(self):
        from rkmon_lib import agentic, scoreboard

        self.assertEqual(
            agentic.PLACEHOLDERS["<ROCKE_GITHUB_ISSUES>"][1](ELSEWHERE),
            "https://github.com/ROCm/tracker/issues",
        )
        self.assertEqual(
            agentic.PLACEHOLDERS["<ROCKE_GITHUB_ISSUES>"][1](CFG),
            "https://github.com/ROCm/r/issues",
        )
        self.assertIn(
            "href='https://github.com/ROCm/tracker/issues/7'",
            webdash._issue(ELSEWHERE, 7),
        )
        read = mock.Mock(return_value=[])
        with mock.patch.object(gh, "rest", read):
            scoreboard.github_breaks(
                ELSEWHERE, scoreboard.dt.datetime.now(scoreboard.dt.timezone.utc)
            )
        self.assertEqual(
            [c[0][0].split("?")[0] for c in read.call_args_list],
            ["repos/ROCm/tracker/issues"] * 2,
        )
        q = mock.Mock(return_value=[])
        with mock.patch.object(breakage.snapshot, "search_pages", q):
            breakage._issues(ELSEWHERE, empty_state(), None)
        self.assertTrue(
            q.call_args[0][0].startswith("repo:ROCm/tracker is:issue"), q.call_args
        )


class TestCli(unittest.TestCase):
    def test_fix_link_keeps_a_recorded_link_when_a_write_fails_and_dismiss_undoes_it(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))

            def half(cfg, st, store_, i, p, evidence):
                st.setdefault("fix_links", {})[str(i)] = [p]
                raise post.PostRefused("gh pr comment failed: boom")

            with mock.patch.object(
                rkmon, "load_config", lambda: CFG
            ), mock.patch.object(rkmon, "Store", lambda: store), mock.patch.object(
                post, "fix_link", half
            ), mock.patch(
                "builtins.print"
            ):
                self.assertEqual(rkmon.main(["fix-link", "11111", "5"]), 4)
                self.assertEqual(store.load()["fix_links"], {"11111": [5]})
                self.assertEqual(rkmon.main(["fix-dismiss", "11111", "5"]), 0)
            saved = store.load()
            self.assertEqual(
                (saved["fix_links"], saved["fix_dismissed"]), ({}, {"11111": [5]})
            )


class TestNamedFixes(unittest.TestCase):
    """Only a PR the issue names as its fix counts (a mere mention makes no PR a develop fix)."""

    def test_named_fixes(self):
        from rkmon_lib.breakage import named_fixes

        cases = {
            "- **Fix candidate:** #6789 (it adds `use_additive_bias`, and #4444 adds the flag too)": {
                6789
            },
            "**Fix candidates:** #5555, #5678": {5555, 5678},
            "**Fix candidate:** #5555. Also see #6666": {5555},
            "Fixed by #1234 (" + TK + "-2670).": {1234},
            "fix PR #1234": {1234},
            "Proposed fix: #2345": {2345},
            "fixed in #3333": {3333},
            # #33333: a candidate line that says none, and a PR mentioned near "fix", name no fix
            "- **Fix candidate:** none yet. Our review of #3456 asked for the fix there, because its lane runs it.": set(),
            "- Land the fix with or before #3456, so its new lane starts green.": set(),
            "**Fix candidate:** not yet (#3456 will carry it)": set(),
            "The fix there broke #2222.": set(),
        }
        for body, want in cases.items():
            self.assertEqual(named_fixes(body), want, body)


class TestFixLinksOff(unittest.TestCase):
    def test_the_candidates_are_listed_and_nothing_acts(self):
        from rkmon_lib import breakage, tickfilter, triage

        line = (
            "FIX_CANDIDATE #6 -> #5 (red-develop): it names test_x. Verify it, then `rk fix-link 5 6 [--evidence "
            "FILE]`; or `rk fix-dismiss 5 6`"
        )
        listed, other = breakage.listed_only([line, "(fix sweep failed: x)"])
        self.assertEqual(
            listed,
            "FIX_LISTED #6 -> #5 (red-develop): it names test_x (fix links are off: listed for you; "
            "nothing is linked)",
        )
        self.assertEqual(other, "(fix sweep failed: x)")
        self.assertEqual(
            tickfilter.actionable(listed), []
        )  # the orchestrator is not woken
        self.assertEqual(
            triage.classify(listed, "guard"), triage.USER
        )  # the user is told


if __name__ == "__main__":
    unittest.main()
