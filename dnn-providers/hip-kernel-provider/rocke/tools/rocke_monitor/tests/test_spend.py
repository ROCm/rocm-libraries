# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Review spend (rkmon_lib/spend.py, rkmon_lib/spend_scan.py): Claude Code transcripts parsed and cached, runs and turns
booked on the reviews, the day and week figures, the outliers, the files, the dashboards' card and the tick step. The
transcripts, the ledger and the journal are written here; no git, no GitHub."""

import copy
import datetime as dt
import json
import os
import pathlib
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import checks, doctor, report, spend, spend_scan, webdash  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

PRICES = {
    "_source": "test",
    "claude-opus-5-5": {
        "in": 4.0,
        "out": 20.0,
        "cache_read": 0.2,
        "cache_write_5m": 5.0,
        "cache_write_1h": 8.0,
    },
    "claude-sonnet-5-5": {
        "in": 2.0,
        "out": 10.0,
        "cache_read": 0.2,
        "cache_write_5m": 2.5,
        "cache_write_1h": 4.0,
    },
}
MON = "<command-name>/rocke-monitor</command-name>"


def usage(inp=0, w5=0, w1h=0, read=0, out=5):
    return {
        "input_tokens": inp,
        "cache_creation_input_tokens": w5 + w1h,
        "cache_read_input_tokens": read,
        "cache_creation": {
            "ephemeral_5m_input_tokens": w5,
            "ephemeral_1h_input_tokens": w1h,
        },
        "output_tokens": out,
    }


def asst(t, mid, u, content, model="claude-opus-5-5"):
    return {
        "type": "assistant",
        "timestamp": t,
        "message": {"id": mid, "model": model, "usage": u, "content": content},
    }


def user(t, text):
    return {
        "type": "user",
        "timestamp": t,
        "message": {"role": "user", "content": text},
    }


def tool_result(t):
    return {
        "type": "user",
        "timestamp": t,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "x",
                    "content": "REVIEW_READY #" "10199 and #" "10101",
                }
            ],
        },
    }


def bash(cmd):
    return {"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": cmd}}


def write_jsonl(path: pathlib.Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_files(
    root: pathlib.Path,
    session: str,
    aid: str,
    agent_type: str,
    desc: str,
    t: str,
    u: dict,
    text: str = "x" * 100,
    brief: str = "",
    model: str = "claude-opus-5-5",
) -> None:
    sub = root / session / "subagents"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"agent-{aid}.meta.json").write_text(
        json.dumps({"agentType": agent_type, "description": desc, "effort": "high"})
    )
    write_jsonl(
        sub / f"agent-{aid}.jsonl",
        [
            user(t, brief or desc),
            asst(t, f"m-{aid}", u, [{"type": "text", "text": text}], model),
        ],
    )


class Case(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.base = pathlib.Path(self.d.name)
        self.root = self.base / "projects"
        self.root.mkdir()
        self.store = Store(self.base / "state")
        self.cfg = {
            "prices": PRICES,
            "paths": {
                "root": str(self.base),
                "claude_projects": str(self.root),
                "dashboard": str(self.base / "out" / "queue.md"),
            },
            "repo": "o/r",
            "operator": {"github_login": "me"},
        }


class TestScan(Case):
    def build(self):
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user("2026-11-09T10:00:00Z", f"{MON} tick"),
                asst(
                    "2026-11-09T10:00:05Z",
                    "o1",
                    usage(inp=1000, read=10000),
                    [bash("rk review-prep 10101")],
                ),
                tool_result("2026-11-09T10:00:06Z"),
                asst(
                    "2026-11-09T10:00:07Z",
                    "o1",
                    usage(inp=1000, read=10000),
                    [{"type": "text", "text": "y" * 200}],
                ),
                user(
                    "2026-11-09T10:30:00Z",
                    '<task-notification><summary>Agent "Verify PR '
                    '10101 follow-up 2" '
                    "completed</summary></task-notification>",
                ),
                asst(
                    "2026-11-09T10:30:05Z",
                    "o2",
                    usage(w5=2000),
                    [{"type": "thinking", "thinking": "", "signature": "s" * 1000}],
                ),
            ],
        )
        write_jsonl(
            self.root / "s2.jsonl",
            [
                user("2026-11-09T09:00:00Z", "an unrelated session"),
                asst("2026-11-09T09:00:01Z", "z", usage(inp=99), []),
            ],
        )
        run_files(
            self.root,
            "s1",
            "a1",
            "rocke-review-verifier",
            "Verify PR " "10101 follow-up 2",
            "2026-11-09T10:01:00Z",
            usage(inp=100, w1h=1000, read=50000),
        )
        run_files(
            self.root,
            "s1",
            "a2",
            "rocke-code-quality-reviewer",
            "Code quality lane PR " "10101 fu2",
            "2026-11-09T10:01:00Z",
            usage(read=20000),
        )
        run_files(
            self.root,
            "s1",
            "a3",
            "rocke-review-verifier",
            "Replay3 verifier PR " "10101 fu2",
            "2026-11-09T10:02:00Z",
            usage(read=1000),
        )
        run_files(
            self.root,
            "s1",
            "a4",
            "general-purpose",
            "Draft overlap notice for 10102",
            "2026-11-09T10:03:00Z",
            usage(read=1000),
        )
        run_files(
            self.root,
            "s2",
            "b1",
            "rocke-review-verifier",
            "Verify PR " "10101 follow-up 2",
            "2026-11-09T10:01:00Z",
            usage(read=1000),
        )

    def test_runs_turns_roles_and_the_output_estimate(self):
        self.build()
        sc = spend_scan.scan(self.root, self.store)
        runs = {r["id"]: r for r in sc["runs"]}
        self.assertEqual(
            set(runs), {"a1", "a2", "a3", "a4"}, "only the monitor session's subagents"
        )
        self.assertEqual(
            [runs[k]["role"] for k in ("a1", "a2", "a3", "a4")],
            ["verifier", "quality", "experiment", "notice"],
        )
        self.assertEqual((runs["a1"]["pr"], runs["a1"]["round"]), (10101, "fu2"))
        self.assertEqual((runs["a2"]["pr"], runs["a2"]["round"]), (10101, "fu2"))
        use = runs["a1"]["use"]["opus"]
        self.assertEqual(use[:5], [100, 0, 1000, 50000, 5])
        self.assertAlmostEqual(use[5], 0.45 * 100)
        self.assertEqual(len(sc["turns"]), 2)
        t1, t2 = sc["turns"]
        self.assertEqual(t1["cat"], "monitor command")
        self.assertIn("10101", t1["nums"], "named in a tool call")
        self.assertNotIn("10199", t1["nums"], "a tool result names nothing")
        self.assertEqual(t1["use"]["opus"][6], 1, "one request, counted once")
        self.assertEqual(t1["use"]["opus"][0], 1000)
        self.assertAlmostEqual(
            t1["use"]["opus"][5],
            0.45 * (len(json.dumps({"command": "rk review-prep 10101"})) + 200),
        )
        self.assertEqual(
            (t2["cat"], t2["nums"], t2["roles"]),
            ("agent or job done", ["10101"], ["verifier"]),
        )
        self.assertAlmostEqual(t2["use"]["opus"][5], 0.266 * 1000)

    def test_cache_reuses_unchanged_files_and_reads_on_where_a_session_grew(self):
        self.build()
        spend_scan.scan(self.root, self.store)
        with mock.patch.object(
            spend_scan, "parse_run", side_effect=AssertionError("re-read")
        ), mock.patch.object(
            spend_scan, "parse_main", side_effect=AssertionError("re-read")
        ):
            sc = spend_scan.scan(self.root, self.store)
        self.assertEqual(len(sc["runs"]), 4)
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user("2026-11-09T11:00:00Z", "next"),
                asst("2026-11-09T11:00:01Z", "o3", usage(inp=7), []),
            ],
            mode="a",
        )
        seen = []
        real = spend_scan.parse_main

        def spy(path, ent, budget, asked=None):
            seen.append(ent.get("offset"))
            return real(path, ent, budget, asked)

        with mock.patch.object(spend_scan, "parse_main", spy):
            sc = spend_scan.scan(self.root, self.store)
        self.assertEqual(len(sc["turns"]), 3)
        self.assertTrue(seen[0] > 0, "read on from the last offset")
        self.assertEqual(sc["turns"][-1]["use"]["opus"][0], 7)

    def test_a_projects_folder_reads_each_project_folder(self):
        # $CLAUDE_CONFIG_DIR/projects (the container's default) holds project folders, not sessions
        project = self.root / "-work-claude-workspace"
        self.root = project
        self.build()
        sc = spend_scan.scan(project.parent, self.store)
        self.assertEqual(
            ({r["id"] for r in sc["runs"]}, len(sc["turns"])),
            ({"a1", "a2", "a3", "a4"}, 2),
        )
        write_jsonl(project / "s1" / "stray.jsonl", [user("2026-11-09T10:00:00Z", MON)])
        mains = [project / "s1.jsonl", project / "s2.jsonl"]
        self.assertEqual(spend_scan.main_sessions(project.parent), mains)
        self.assertEqual(
            spend_scan.main_sessions(project),
            mains,
            "a session's own folder is no project folder",
        )

    def test_a_byte_budget_leaves_the_rest_for_the_next_pass(self):
        self.build()
        sc = spend_scan.scan(self.root, self.store, max_bytes=300)
        self.assertGreater(sc["left"], 0)
        sc = spend_scan.scan(self.root, self.store)
        self.assertEqual((sc["left"], len(sc["turns"]), len(sc["runs"])), (0, 2, 4))

    def test_a_line_still_being_written_waits(self):
        write_jsonl(self.root / "s1.jsonl", [user("2026-11-09T10:00:00Z", MON)])
        with open(self.root / "s1.jsonl", "a", encoding="utf-8") as f:
            f.write(
                '{"type": "user", "timestamp": "2026-11-09T10:01:00Z", "message": {"content": "par'
            )
        sc = spend_scan.scan(self.root, self.store)
        self.assertEqual(len(sc["turns"]), 1)

    def test_a_time_leaves_out_the_sessions_last_written_before_it(self):
        """A pass given a time (`since`) leaves out each main session last written before it, and its subagents,
        whenever they were written: none of theirs is read, returned or cached. A pass without it reads them, and
        only them, the others being cached."""
        self.build()
        write_jsonl(
            self.root / "s3.jsonl",
            [
                user("2026-11-01T10:00:00Z", f"{MON} tick"),
                asst("2026-11-01T10:00:05Z", "o9", usage(inp=300), []),
            ],
        )
        run_files(
            self.root,
            "s3",
            "c1",
            "rocke-review-verifier",
            "Verify PR " "10103",
            "2026-11-01T10:01:00Z",
            usage(read=10),
        )
        for name, at in (
            ("s3.jsonl", "2026-11-01T10:00:05+00:00"),
            ("s1.jsonl", "2026-11-09T10:30:05+00:00"),
        ):
            written = dt.datetime.fromisoformat(at).timestamp()
            os.utime(self.root / name, (written, written))
        since = dt.datetime.fromisoformat("2026-11-05T00:00:00+00:00").timestamp()
        sc = spend_scan.scan(self.root, self.store, since=since)
        self.assertEqual(
            ({r["id"] for r in sc["runs"]}, len(sc["turns"])),
            ({"a1", "a2", "a3", "a4"}, 2),
        )
        cached = spend_scan.load_cache(self.store)["files"]
        self.assertFalse(
            [
                k
                for k in cached
                if "s3" in pathlib.Path(k).parts or k.endswith("s3.jsonl")
            ]
        )
        read = []
        real = spend_scan.parse_main

        def spy(path, ent, budget, asked=None):
            read.append(path.name)
            return real(path, ent, budget, asked)

        with mock.patch.object(spend_scan, "parse_main", spy):
            sc = spend_scan.scan(self.root, self.store)
        self.assertEqual(read, ["s3.jsonl"])
        self.assertEqual(
            ({r["id"] for r in sc["runs"]}, len(sc["turns"])),
            ({"a1", "a2", "a3", "a4", "c1"}, 3),
        )

    def test_a_pass_read_by_family_as_a_pass_given_the_ids_records_it(self):
        """as_asked: a pass read with no ids asked for, as a pass given them records it: each request of a pass that
        asks for an id under that id, whatever family it reported, but for the model guard's `off`; the runs of a
        pass that asks for none, and a pass given no ids, as they are. The pass read is not changed.
        """
        self.build()
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user("2026-11-09T11:00:00Z", "tick"),
                asst(
                    "2026-11-09T11:00:01Z",
                    "o4",
                    usage(inp=10),
                    [{"type": "text", "text": "z" * 20}],
                    "claude-sonnet-5-5",
                ),
                asst("2026-11-09T11:00:02Z", "o5", usage(inp=20), []),
            ],
            mode="a",
        )
        asked = {
            spend_scan.MAIN: "claude-opus-5-5",
            "verifier": "claude-opus-5-5",
            "quality": "claude-sonnet-5-5",
        }
        by_family = spend_scan.scan(self.root, self.store)
        before = copy.deepcopy(by_family)
        by_id = spend_scan.scan(self.root, Store(self.base / "by_id"), asked=asked)
        self.assertTrue(
            [x for x in by_id["runs"] + by_id["turns"] if x.get("off")],
            "the model guard sees a pass on another family",
        )

        def without_off(sc: dict) -> dict:
            return {
                k: [{n: v for n, v in x.items() if n != "off"} for x in sc[k]]
                for k in ("runs", "turns")
            }

        self.assertEqual(
            without_off(spend_scan.as_asked(by_family, asked)), without_off(by_id)
        )
        self.assertEqual(by_family, before)
        self.assertEqual(spend_scan.as_asked(by_family, None), by_family)
        last = spend_scan.as_asked(by_family, asked)["turns"][-1]["use"]
        self.assertEqual(
            (list(last), last["claude-opus-5-5"][0]), (["claude-opus-5-5"], 30)
        )

    def test_review_actions_name_prs_and_mentions_do_not(self):
        refs = spend_scan.review_refs
        self.assertEqual(refs("cat /w/tmp/v11011fu4/pr11011_fu4.md"), {"11011"})
        self.assertEqual(refs(r"C:\x\w\tmp\post14444fu3.sh"), {"14444"})
        self.assertEqual(
            refs("~/w/monitor/rk review-prep 12121 2>&1; rk post 12122 /w/p.md"),
            {"12121", "12122"},
        )
        self.assertEqual(
            refs(
                "rk dispatch-pack 10101; rk spec-post 10102 --round fu2; rk wait-quality "
                "/h/w/tmp/v10103fu1"
            ),
            {"10101", "10102", "10103"},
        )
        self.assertEqual(
            refs("gh api repos/o/r/pulls/11111 --jq .head; git -C ~/x/wt11111m log"),
            set(),
        )
        self.assertEqual(
            refs("rk heads 10101 10102 10103; grep -n 10104 dashboard.md"), set()
        )
        self.assertEqual(
            spend_scan.agent_refs("Verify PR " "10101 follow-up 2", "verifier"),
            {"10101"},
        )

    def test_a_pr_of_any_number_of_digits_is_named(self):
        """A repository's PRs may have fewer or more than five digits: a review's directory, post, scripts and commands
        name them as they name one of five, and a template's or a read's PR is still no review work.
        """
        refs = spend_scan.review_refs
        self.assertEqual(refs("cat /w/tmp/v512fu2/pr512_fu2.md"), {"512"})
        self.assertEqual(
            refs("rk post 987654 /w/p.md; bash facts77.sh"), {"987654", "77"}
        )
        self.assertEqual(
            refs("sed 's/512/613/g' facts512.sh > facts613.sh && bash facts613.sh"),
            {"613"},
        )
        self.assertEqual(refs("cat facts512.sh; rk heads 512 613"), set())

    def test_a_run_and_its_notification_name_a_pr_of_any_number_of_digits(self):
        """A verifier's or lane's description names its PR as "#N" or "PR N", with as many digits as the PR has: the
        run's role, PR and round, and the turn its notification wakes, are the review's. A bare number names no PR.
        """
        role, refs, pr_round = (
            spend_scan.role_of,
            spend_scan.agent_refs,
            spend_scan.pr_round,
        )
        three, seven = "Verify PR #" "512 fu2", "Quality lane PR " "1234567 fr"
        self.assertEqual(role("general-purpose", three), "verifier")
        self.assertEqual(refs(three, "verifier"), {"512"})
        self.assertEqual(pr_round(three), (512, "fu2"))
        self.assertEqual(role("general-purpose", seven), "quality")
        self.assertEqual(refs(seven, "quality"), {"1234567"})
        self.assertEqual(pr_round(seven), (1234567, "fr"))
        self.assertEqual(
            role("general-purpose", "Re-review #" "1234567 fu1, deep review"),
            "deep_verifier",
        )
        self.assertEqual(pr_round("Count the 12000 lines of the log"), (None, ""))
        done = '<task-notification><summary>Agent "{}" completed</summary></task-notification>'
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user("2026-11-09T10:00:00Z", f"{MON} tick"),
                asst("2026-11-09T10:00:05Z", "o1", usage(inp=10), []),
                user("2026-11-09T10:30:00Z", done.format(three)),
                asst("2026-11-09T10:30:05Z", "o2", usage(inp=10), []),
                user("2026-11-09T10:40:00Z", done.format(seven)),
                asst("2026-11-09T10:40:05Z", "o3", usage(inp=10), []),
            ],
        )
        turns = spend_scan.scan(self.root, self.store)["turns"]
        self.assertEqual(
            [(t["cat"], t["nums"], t["roles"]) for t in turns[1:]],
            [
                ("agent or job done", ["512"], ["verifier"]),
                ("agent or job done", ["1234567"], ["quality"]),
            ],
        )

    def test_a_template_renumbered_copied_or_only_read_is_no_review_work(self):
        """A PR's facts script used as the template of other PRs' facts scripts (renumbered, copied, or only read) is
        no review work for the template's PR; running, backing up or reading its own files is.
        """
        refs = spend_scan.review_refs
        self.assertEqual(
            refs(
                "sed 's/13131/11212/g' facts13131.sh > facts11212.sh && bash facts11212.sh 2>&1 | head "
                "-150; ls -d v11212* 2>/dev/null; ls /home/op/w/lessons/notes/ | grep "
                "11212"
            ),
            {"11212"},
        )
        self.assertEqual(
            refs(
                "sed 's/13131/10011/g' facts13131.sh > facts10011.sh && MSYS_NO_PATHCONV=1 wsl.exe -d "
                "Ubuntu -u op -- bash /home/op/w/tmp/facts10011.sh 2>&1 | "
                "cut -c1-600 | head -230"
            ),
            {"10011"},
        )
        self.assertEqual(
            refs(
                "sed 's/13131/14141/g' ~/w/tmp/facts13131.sh > ~/w/tmp/"
                "facts14141.sh && bash ~/w/tmp/facts14141.sh"
            ),
            {"14141"},
        )
        self.assertEqual(
            refs(
                "cat //wsl.localhost/Ubuntu/home/op/w/tmp/facts13131.sh; ls "
                "//wsl.localhost/Ubuntu/home/op/w/tmp/ | grep -E "
                "'^v13232|^facts13232'"
            ),
            {"13232"},
        )
        self.assertEqual(
            refs("cp ~/w/tmp/prep13333.sh ~/w/tmp/prep13434.sh"), {"13434"}
        )
        read = {
            "type": "tool_use",
            "name": "Read",
            "input": {"file_path": "/x/w/tmp/facts13131.sh"},
        }
        self.assertEqual(spend_scan._tool_refs(read), (set(), ""))
        # Review work stays: the PR's own script run, its own copy (a backup), its review files read.
        self.assertEqual(refs("bash ~/w/tmp/facts13131.sh 2>&1 | head -80"), {"13131"})
        self.assertEqual(
            refs("cp /w/x/spec16161fr.json " "/w/x/spec16161fr.json.bak 2>/dev/null"),
            {"16161"},
        )
        self.assertEqual(
            refs("cat /w/tmp/v17171fu2/notes.md /w/tmp/v17171fu2/lv.py"), {"17171"}
        )
        self.assertEqual(refs("sed -i 's/\\r$//' ~/w/tmp/spec16161fr.json"), {"16161"})
        read = {
            "type": "tool_use",
            "name": "Read",
            "input": {"file_path": "/x/w/tmp/v17171fu2/notes.md"},
        }
        self.assertEqual(spend_scan._tool_refs(read), ({"17171"}, ""))
        self.assertEqual(
            spend_scan.agent_refs("Replay3 verifier PR " "10101 fu2", "experiment"),
            set(),
        )

    def test_a_review_directory_in_the_brief_gives_the_pr(self):
        """By prep's rule for the directory's name (prep.REVIEW_DIR), whatever the PR's length."""
        for brief, pr in (
            ("Inputs: /w/tmp/v100234fu2/facts.txt", 100234),
            ("Inputs: /w/tmp/v12345fr2/facts.txt", 12345),
        ):
            self.assertEqual(spend_scan.pr_round("Verify the draft", brief)[0], pr)

    def test_roles_and_rounds(self):
        role = spend_scan.role_of
        self.assertEqual(
            role("general-purpose", "Verify #" "10101 follow-up draft (fu3)"),
            "verifier",
        )
        self.assertEqual(
            role("general-purpose", "Verify 4 fix-sweep candidates"), "other"
        )
        self.assertEqual(
            role("rocke-review-verifier-s55", "E7 s1 verifier 10101 fu2"), "experiment"
        )
        self.assertEqual(role("fanout-haiku-55", "r1 B2 lens3 haiku"), "experiment")
        self.assertEqual(
            role("rocke-code-quality-reviewer", "Quality lane #" "10101 fu2"), "quality"
        )
        self.assertEqual(
            role("rocke-review-verifier", "Verify #" "10101 fu2, deep review"),
            "deep_verifier",
        )
        self.assertEqual(
            role("general-purpose", "Re-review #" "10101 fr, Deep review"),
            "deep_verifier",
        )
        self.assertEqual(
            role("general-purpose", "Build review spend statistics"), "other"
        )
        self.assertEqual(
            spend_scan.pr_round("Verify #" "11313 first-round draft"), (11313, "fr")
        )
        self.assertEqual(
            spend_scan.pr_round("Verify #" "10101 follow-up draft (fu3)"),
            (10101, "fu3"),
        )
        self.assertEqual(
            spend_scan.pr_round("Triage merged PR " "10011's follow-ups"), (10011, "fu")
        )

    def test_a_run_naming_the_code_quality_pass_is_the_quality_role(self):
        """Whatever its type, a run whose description names a PR and the code quality pass, or the quality lane,
        is the quality role's; one that names no PR is not."""
        role = spend_scan.role_of
        for desc in (
            "Code quality pass PR " "10101 fu2",
            "Code quality pass #" "10101 fr",
            "Code quality lane PR " "10101 fu2",
        ):
            self.assertEqual(role("general-purpose", desc), "quality", desc)
        self.assertEqual(
            role("general-purpose", "Code quality pass on the toolkit"), "other"
        )

    def test_offload_md_s_deep_review_description_gives_its_role(self):
        """The description offload.md gives a deep review's verifier, filled in, makes its run a deep review's
        verifier, with its PR and round."""
        words = "Verify #N <round>, deep review"
        offload = (
            pathlib.Path(__file__).resolve().parent.parent
            / "agentic"
            / "skills"
            / "rocke-monitor"
            / "offload.md"
        )
        self.assertIn(f"`{words}`", offload.read_text(encoding="utf-8"))
        desc = words.replace("#N", "#" "10101").replace("<round>", "fu2")
        self.assertEqual(
            spend_scan.role_of("rocke-review-verifier", desc), "deep_verifier"
        )
        self.assertEqual(spend_scan.pr_round(desc), (10101, "fu2"))


class TestCacheFile(Case):
    def test_the_cache_is_written_whole_at_the_shared_mode(self):
        """The cache is written whole (whole_file.write): a new file of its own beside it, made new and never through a
        link, at the mode of the files the monitor and the writer share whatever the umask.
        """
        elsewhere = self.base / "elsewhere.json"
        elsewhere.write_text("{}\n", encoding="utf-8")
        (self.store.dir / (spend_scan.CACHE + ".tmp")).symlink_to(elsewhere)
        cache = {"v": spend_scan.VERSION, "files": {"s1.jsonl": {"stat": [1, 2]}}}
        umask = os.umask(0o077)
        try:
            spend_scan.save_cache(self.store, cache)
        finally:
            os.umask(umask)
        written = self.store.dir / spend_scan.CACHE
        self.assertFalse(written.is_symlink())
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(spend_scan.load_cache(self.store), cache)
        self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o660)

    def test_the_sizes_and_the_card_are_written_whole_at_the_shared_mode(self):
        """The reviewed sizes' cache and the dashboards' card are written whole too, at the same mode."""
        elsewhere = self.base / "elsewhere.json"
        elsewhere.write_text("{}\n", encoding="utf-8")
        for name in (spend.SIZES, spend.SUMMARY):
            (self.store.dir / (name + ".tmp")).symlink_to(elsewhere)
        cfg = dict(self.cfg, paths=dict(self.cfg["paths"], repo=str(self.base)))
        review = {"head": "a" * 40, "old": ""}
        umask = os.umask(0o077)
        try:
            with mock.patch(
                "rkmon_lib.hybrid.size.measure",
                return_value=[{"eff": 12, "tier": "S", "src": "diff"}],
            ), mock.patch.object(spend, "_develop", return_value="b" * 40):
                self.assertEqual(spend.measure(cfg, self.store, [review]), 0)
            spend.save_card(self.store, {"at": f"{DAY}T12:00:00Z"})
        finally:
            os.umask(umask)
        for name in (spend.SIZES, spend.SUMMARY):
            written = self.store.dir / name
            self.assertFalse(written.is_symlink())
            self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o660)
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "{}\n")
        self.assertEqual(
            spend.load_sizes(self.store), {"a" * 40 + ":": {"eff": 12, "tier": "S"}}
        )
        self.assertEqual(spend.load_card(self.store), {"at": f"{DAY}T12:00:00Z"})


class TestPrices(unittest.TestCase):
    def test_dollars_by_kind_with_estimated_or_recorded_output(self):
        p = spend.price_table({"prices": PRICES})
        self.assertNotIn("_source", p)
        use = {"opus": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 10, 1_000_000, 3]}
        self.assertAlmostEqual(spend.usd(p, use), 4 + 5 + 8 + 0.2 + 20)
        self.assertAlmostEqual(
            spend.usd(p, use, estimated=False), 4 + 5 + 8 + 0.2 + 20 * 10 / 1e6
        )
        self.assertAlmostEqual(
            spend.usd(p, {"sonnet": [1_000_000, 0, 0, 0, 0, 0, 1]}), 2
        )

    def test_no_table_is_refused_and_a_model_with_no_price_costs_unknown(self):
        with self.assertRaises(ValueError):
            spend.price_table({})
        p, unpriced = spend.price_table({"prices": PRICES}), set()
        use = {
            "haiku": [1, 0, 0, 0, 0, 0, 1],
            "claude-haiku-5-5": [1, 0, 0, 0, 0, 0, 1],
            "claude-opus-5-5": [1_000_000, 0, 0, 0, 0, 0, 1],
        }
        self.assertAlmostEqual(spend.usd(p, use, unpriced=unpriced), 4)
        self.assertEqual(unpriced, {"haiku", "claude-haiku-5-5"})
        self.assertAlmostEqual(spend.usd(p, use), 4, "no set given: still no guess")

    def test_an_id_by_its_own_entry_a_family_by_its_newest_id(self):
        p = spend.price_table({"prices": dict(PRICES, **{"claude-opus-4-7": OLDER})})
        self.assertIs(spend.price_of(p, "claude-opus-4-7"), p["claude-opus-4-7"])
        self.assertIs(spend.price_of(p, "claude-opus-5-5"), p["claude-opus-5-5"])
        self.assertIs(spend.price_of(p, "opus"), p["claude-opus-5-5"], "newest")
        self.assertIsNone(
            spend.price_of(p, "claude-opus-6-0"), "an id with no entry: no guess"
        )
        self.assertIsNone(spend.price_of(p, "haiku"))
        fable = spend.price_table({"prices": dict(PRICES, fable=OLDER)})
        self.assertIs(
            spend.price_of(fable, "fable"),
            fable["fable"],
            "the id fable, a family's name too, by its own entry",
        )

    def test_an_id_with_a_context_suffix_by_its_base_id_s_entry(self):
        """An id with Claude Code's context suffix, as the session runs with (opus[1m]), is priced by its base id's
        entry when the table has none of its own; one of its own wins, and a base id with no entry is no guess.
        """
        p = spend.price_table({"prices": PRICES})
        self.assertIs(spend.price_of(p, "claude-opus-5-5[1m]"), p["claude-opus-5-5"])
        own = spend.price_table(
            {"prices": dict(PRICES, **{"claude-opus-5-5[1m]": OLDER})}
        )
        self.assertIs(
            spend.price_of(own, "claude-opus-5-5[1m]"), own["claude-opus-5-5[1m]"]
        )
        self.assertIsNone(spend.price_of(p, "claude-opus-6-0[1m]"))


OLDER = {  # an older id's prices: ten times claude-opus-5-5's
    "in": 40.0,
    "out": 200.0,
    "cache_read": 2.0,
    "cache_write_5m": 50.0,
    "cache_write_1h": 80.0,
}
WEEK = [f"2026-11-{d:02d}" for d in range(9, 16)]  # 2026-W46, Monday to Sunday


def monitor_day(root: pathlib.Path, d: str) -> None:
    """One day D of the monitor's session, written under the transcripts folder ROOT: a turn whose cache reads cost
    $0.20 at claude-opus-5-5's prices, its verifier's run $1 and its code quality pass's $0.50; every request reports
    claude-opus-5-5. The week recorded in WEEK's days is the round trip's too (tests/test_round_trip.py).
    """
    first = not (root / "s1.jsonl").exists()
    write_jsonl(
        root / "s1.jsonl",
        [
            user(f"{d}T09:00:00Z", MON if first else "tick"),
            asst(f"{d}T09:00:01Z", f"o-{d}", usage(read=1_000_000, out=0), []),
        ],
        mode="w" if first else "a",
    )
    for aid, kind, desc, reads in (
        ("v", "rocke-review-verifier", "Verify PR " "10101 follow-up 1", 5),
        ("q", "rocke-code-quality-reviewer", "Code quality lane PR " "10101", 2.5),
    ):
        run_files(
            root,
            "s1",
            f"{aid}{d[-2:]}",
            kind,
            desc,
            f"{d}T10:00:00Z",
            usage(read=int(reads * 1_000_000), out=0),
            text="",
        )


class TestPricedByModelId(Case):
    """Usage recorded by the model id its pass asked for and priced by that id's entry; usage recorded before that, by
    family (as the scan's cache keeps it), priced by the family's newest id in the table.
    """

    def setUp(self):
        super().setUp()
        self.cfg["prices"] = dict(PRICES, **{"claude-opus-4-7": OLDER})
        self.prices = spend.price_table(self.cfg)

    def day(self, d: str) -> None:
        monitor_day(self.root, d)

    def week(self, store, cfg=None) -> list[dict]:
        """The runs and turns one scan holds: read by family with no configuration (as before usage was recorded by
        id), else by the ids cfg's passes ask for."""
        sc = spend_scan.scan(
            self.root, store, asked=spend.asked_ids(cfg) if cfg else None
        )
        return sc["runs"] + sc["turns"]

    def priced(self, items: list[dict], unpriced: set | None = None) -> float:
        return sum(spend.usd(self.prices, x["use"], unpriced=unpriced) for x in items)

    def test_each_pass_records_under_the_id_it_asked_for(self):
        cfg = dict(
            self.cfg,
            orchestrator={"model": "opus"},
            models={"audit": "sonnet", "deep": "haiku"},
        )
        self.assertEqual(
            spend.asked_ids(cfg),
            {
                spend_scan.MAIN: "claude-opus-5-5",
                "verifier": "claude-sonnet-5-5",
                "deep_verifier": "claude-haiku-5-5",
                "quality": "claude-haiku-5-5",
            },
        )
        self.day(WEEK[0])
        items = self.week(self.store, cfg)
        self.assertEqual(
            sorted((x.get("role", "turn"), list(x["use"])) for x in items),
            [
                ("quality", ["claude-haiku-5-5"]),
                ("turn", ["claude-opus-5-5"]),
                ("verifier", ["claude-sonnet-5-5"]),
            ],
            "by the id each pass asked for, whatever model the request reports",
        )

    def test_a_week_of_recorded_usage_priced_both_ways(self):
        for d in WEEK[:4]:
            self.day(d)
        self.week(
            self.store
        )  # the first four days, read before usage was recorded by id
        for d in WEEK[4:]:
            self.day(d)
        week = self.week(self.store, self.cfg)
        self.assertEqual(
            sorted({k for x in week for k in x["use"]}),
            ["claude-opus-5-5", "opus"],
            "the cache keeps the first days by family; the rest is read by the id each pass asked for",
        )
        by_family = self.week(Store(self.base / "by_family"))
        by_id = self.week(Store(self.base / "by_id"), self.cfg)
        self.assertEqual({k for x in by_family for k in x["use"]}, {"opus"})
        self.assertEqual({k for x in by_id for k in x["use"]}, {"claude-opus-5-5"})
        for items in (week, by_family, by_id):
            self.assertAlmostEqual(
                self.priced(items),
                7 * 1.7,
                msg="by family at the family's newest id, claude-opus-5-5, not the older id at ten times its prices",
            )
        older = dict(
            self.cfg,
            orchestrator={"model": "claude-opus-4-7"},
            models={"audit": "claude-opus-4-7"},
        )
        self.assertAlmostEqual(
            self.priced(self.week(Store(self.base / "older"), older)),
            7 * (2.0 + 10.0 + 0.5),
            msg="by the id each pass asked for: the older id's prices for the session and the verifier",
        )

    def test_a_model_with_no_price_costs_unknown(self):
        self.ledger_post()
        for d in WEEK:
            self.day(d)
        cfg = dict(self.cfg, orchestrator={"model": "claude-opus-6-0"})
        unpriced: set = set()
        self.assertAlmostEqual(
            self.priced(self.week(Store(self.base / "newer"), cfg), unpriced),
            7 * 1.5,
            msg="the runs only: no guess at the session's model from the table's ids",
        )
        self.assertEqual(unpriced, {"claude-opus-6-0"})
        data = spend.collect(cfg, self.store)
        self.assertEqual(data["unpriced"], ["claude-opus-6-0"])
        words = "Cost unknown for claude-opus-6-0: the prices have no entry for it"
        self.assertIn(words, spend.report_md(cfg, data))
        card = spend.card_data(data, WEEK[-1])
        self.assertIn(words, spend.card_note(card))
        self.assertIn(words, webdash.spend_card(card))
        self.assertIn(words, "\n".join(spend.card_md(card)))
        self.assertNotIn(
            "Cost unknown",
            spend.report_md(self.cfg, spend.collect(self.cfg, Store(self.base / "s2"))),
        )

    def test_a_session_with_a_context_suffix_costs_what_it_spent(self):
        """A week of a session on opus[1m], as production's runs: its turns are recorded under claude-opus-5-5[1m] and
        priced at claude-opus-5-5's entry, so no cost is unknown."""
        self.ledger_post()
        for d in WEEK:
            self.day(d)
        cfg = dict(self.cfg, orchestrator={"model": "opus[1m]"})
        items = self.week(self.store, cfg)
        self.assertIn("claude-opus-5-5[1m]", {k for x in items for k in x["use"]})
        unpriced: set = set()
        self.assertAlmostEqual(self.priced(items, unpriced), 7 * 1.7)
        self.assertEqual(unpriced, set())
        data = spend.collect(cfg, self.store)
        self.assertEqual(data["unpriced"], [])
        self.assertNotIn("Cost unknown", spend.report_md(cfg, data))

    def ledger_post(self) -> None:
        """One post, the week's first day."""
        write_jsonl(
            self.store.ledger_path,
            [
                {
                    "ts": f"{WEEK[0]}T08:00:00Z",
                    "kind": "post",
                    "pr": 10100,
                    "head": "h",
                    "ref": "",
                    "note": "first_round",
                }
            ],
        )


DAY = "2026-11-15"
OFF = ("claude-opus-5-5", "claude-sonnet-5-5")  # asked for, reported: another family


class TestModelGuard(Case):
    """The model guard: a request whose reported model is of another family than the id its pass asked
    for is a WARN in rk spend, on the admin dashboard's card and in rk doctor; a gateway's own spelling of the same
    family is not."""

    def session(self, verifier_reports: str) -> None:
        """A day of the monitor: a turn and a code quality pass run that report the model asked for in a gateway's own
        spelling, and a verifier run that reports VERIFIER_REPORTS in two entries of one request.
        """
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user(f"{DAY}T09:00:00Z", MON),
                asst(f"{DAY}T09:00:01Z", "o1", usage(read=1000), [], "Claude-Opus-5.5"),
            ],
        )
        lane, verifier = ("q1", "rocke-code-quality-reviewer"), (
            "v1",
            "rocke-review-verifier",
        )
        run_files(
            self.root,
            "s1",
            *lane,
            "Code quality lane PR " "10101",
            f"{DAY}T10:00:00Z",
            usage(read=1000),
            model="CLAUDE-OPUS-5-5",
        )
        run_files(
            self.root,
            "s1",
            *verifier,
            "Verify PR " "10101 follow-up 1",
            f"{DAY}T10:00:00Z",
            usage(read=1000),
            model=verifier_reports,
        )
        write_jsonl(
            self.root / "s1" / "subagents" / "agent-v1.jsonl",
            [asst(f"{DAY}T10:05:00Z", "m-v1", usage(read=1000), [], verifier_reports)],
            mode="a",
        )

    def refresh(self) -> tuple[dict, dict]:
        """The data and the card a refresh of the spend card makes on DAY (each pass asks for claude-opus-5-5)."""
        with mock.patch.object(spend, "now", return_value=f"{DAY}T12:00:00Z"):
            data = spend.collect(self.cfg, self.store)
            card = spend.card_data(data)
        spend.save_card(self.store, card)
        return data, card

    def test_another_family_is_a_warn(self):
        self.session("claude-sonnet-5-5")
        data, card = self.refresh()
        self.assertEqual(
            [(o["asked"], o["reported"], o["calls"]) for o in data["off_family"]],
            [OFF + (1,)],
            "the verifier's request, counted once; the turn and the lane in a gateway's spelling are not",
        )
        words = (
            "1 call of the last 7 days ran on another model family than the one asked for: claude-sonnet-5-5 "
            "answered for claude-opus-5-5 (1 call, the last at 2026-11-15T10:05:00Z)"
        )
        self.assertEqual(
            card["off_family"],
            [
                {
                    "asked": OFF[0],
                    "reported": OFF[1],
                    "calls": 1,
                    "last": "2026-11-15T10:05:00",
                }
            ],
        )
        row = doctor.model_guard_check(self.cfg, self.store.dir)
        self.assertEqual((row.status, row.name), ("WARN", "model guard"))
        self.assertIn(words, row.found)
        self.assertIn("rk spend", checks.render([row]))
        page = webdash.spend_card(card)
        self.assertIn("! WARN</span> <b>Model guard:</b> " + words, page)
        with mock.patch.object(spend, "now", return_value=f"{DAY}T12:00:00Z"):
            self.assertIn(f"**WARN:** {words}.", spend.report_md(self.cfg, data))
        self.assertNotIn("Model guard", "\n".join(spend.card_md(card)), "admin only")

    def test_a_gateway_s_own_spelling_of_the_family_is_not(self):
        self.session("Claude-Opus-5.5")
        data, card = self.refresh()
        self.assertEqual((data["off_family"], card["off_family"]), ([], []))
        row = doctor.model_guard_check(self.cfg, self.store.dir)
        self.assertEqual(row.status, "PASS")
        self.assertIn(
            "every call of the last 7 days ran on the model family asked", row.found
        )
        self.assertNotIn("WARN", webdash.spend_card(card))
        with mock.patch.object(spend, "now", return_value=f"{DAY}T12:00:00Z"):
            self.assertNotIn("Model guard", spend.report_md(self.cfg, data))

    def test_a_deep_review_s_verifier_is_asked_for_on_the_deep_model(self):
        """A deep review's verifier runs the deep model (offload.md): with the audit model opus and the deep model
        sonnet, its run is recorded under the deep model's id and is no WARN, beside a review's verifier recorded under
        the audit model's."""
        self.cfg = dict(self.cfg, models={"audit": "opus", "deep": "sonnet"})
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user(f"{DAY}T09:00:00Z", MON),
                asst(f"{DAY}T09:00:01Z", "o1", usage(read=1000), []),
            ],
        )
        opus, sonnet = "claude-opus-5-5", "claude-sonnet-5-5"
        for aid, agent_type, desc, model in (
            ("v1", "rocke-review-verifier", "Verify #" "10101 fu1", opus),
            (
                "v2",
                "rocke-review-verifier",
                "Verify #" "10101 fu2, deep review",
                sonnet,
            ),
            ("q2", "rocke-code-quality-reviewer", "Quality lane #" "10101 fu2", sonnet),
        ):
            run_files(
                self.root,
                "s1",
                aid,
                agent_type,
                desc,
                f"{DAY}T10:00:00Z",
                usage(read=1000),
                model=model,
            )
        sc = spend_scan.scan(self.root, self.store, asked=spend.asked_ids(self.cfg))
        self.assertEqual(
            {r["id"]: (r["role"], list(r["use"]), r.get("off")) for r in sc["runs"]},
            {
                "v1": ("verifier", [opus], None),
                "v2": ("deep_verifier", [sonnet], None),
                "q2": ("quality", [sonnet], None),
            },
        )
        data, card = self.refresh()
        self.assertEqual((data["off_family"], card["off_family"]), ([], []))
        row = doctor.model_guard_check(self.cfg, self.store.dir)
        self.assertEqual(row.status, "PASS")

    def test_the_last_seven_days_and_an_id_of_no_family(self):
        old = {"asked": OFF[0], "reported": OFF[1]}
        data = {
            "off_family": [
                dict(old, t="2026-11-08T23:59:59", calls=5),
                dict(old, t="2026-11-09T00:00:00", calls=1),
                dict(old, t="2026-11-15T10:00:00", calls=2),
                dict(old, t="2026-11-16T00:00:00", calls=7),
                {
                    "t": "2026-11-12T10:00:00",
                    "asked": "claude-sonnet-5-5",
                    "reported": "example-model-1",
                    "calls": 1,
                },
            ]
        }
        self.assertEqual(
            spend.guard(data, DAY),
            [
                dict(old, calls=3, last="2026-11-15T10:00:00"),
                {
                    "asked": "claude-sonnet-5-5",
                    "reported": "example-model-1",
                    "calls": 1,
                    "last": "2026-11-12T10:00:00",
                },
            ],
        )
        self.session("claude-sonnet-5-5")
        sc = spend_scan.scan(
            self.root, self.store, asked={spend_scan.MAIN: "example-model-1"}
        )
        self.assertFalse(
            [x for x in sc["runs"] + sc["turns"] if x.get("off")],
            "an id of no known family is not compared",
        )

    def test_the_doctor_row_is_off_without_spend_and_waits_for_a_refresh(self):
        self.assertEqual(doctor.model_guard_check({}, self.store.dir).status, "OFF")
        row = doctor.model_guard_check(self.cfg, self.store.dir)
        self.assertEqual(row.status, "SKIP", "no card yet")
        spend.save_card(self.store, {"at": f"{DAY}T12:00:00Z"})
        self.assertEqual(
            doctor.model_guard_check(self.cfg, self.store.dir).status,
            "SKIP",
            "a card refreshed before the guard",
        )


def opus(dollars_read: float) -> dict:
    """A use worth `dollars_read` in cache reads only."""
    return {"opus": [0, 0, 0, round(dollars_read / 0.2 * 1e6), 0, 0, 1]}


def run(rid, role, pr, rnd, start, dollars):
    return {
        "id": rid,
        "role": role,
        "pr": pr,
        "round": rnd,
        "start": start,
        "use": opus(dollars),
    }


def turn(t, dollars, nums=(), roles=()):
    return {
        "t": t,
        "cat": "x",
        "nums": [str(n) for n in nums],
        "roles": list(roles),
        "use": opus(dollars),
    }


class Booking(Case):
    def ledger(self, rows):
        write_jsonl(
            self.store.ledger_path,
            [
                {
                    "ts": ts,
                    "kind": "post",
                    "pr": pr,
                    "head": head,
                    "ref": "",
                    "note": note,
                }
                for ts, pr, head, note in rows
            ],
        )

    def costs(self, rows):
        """state/review_cost.jsonl: a line per review through `rk post --premises`, `unposted` for one kept unposted
        while review posting is off (it has no ledger entry)."""
        write_jsonl(
            self.store.dir / "review_cost.jsonl",
            [
                {
                    "pr": pr,
                    "round": rnd,
                    "head": head,
                    "at": at,
                    "post": f"pr{pr}_{rnd}.md",
                    **({"unposted": True} if kept else {}),
                }
                for at, pr, head, rnd, kept in rows
            ],
        )

    def data(self, runs, turns, today="2026-11-10T12:00:00Z"):
        reviews = spend.paid_reviews(self.store)
        booked = spend.attribute(spend.price_table(self.cfg), reviews, runs, turns)
        for p in reviews:
            p["size"] = {"eff": 10, "tier": "<1.5k"}
        return {
            "at": today,
            "reviews": reviews,
            "overhead": booked["overhead"],
            "other": booked["other"],
            "unfollowed": booked["unfollowed"],
            "covered": spend.covered(self.store),
            "ledgers": {},
            "unread_bytes": 0,
            "unmeasured": 0,
            "first": reviews[0]["day"] if reviews else None,
        }


class TestAttribution(Booking):
    def test_runs_and_turns_are_booked_on_their_posts(self):
        self.ledger(
            [
                ("2026-11-09T09:00:00Z", 10101, "h0", "first_round"),
                ("2026-11-09T12:00:00Z", 10101, "h1", "followup"),
                ("2026-11-09T13:00:00Z", 10102, "h2", "first_round"),
            ]
            + [
                ("2026-11-09T23:00:00Z", n, f"h{n}", "first_round")
                for n in (10104, 10105, 10106)
            ]
        )
        runs = [
            run("v1", "verifier", 10101, "fu1", "2026-11-09T10:00:00", 3.0),
            run("v2", "verifier", 10101, "fu1", "2026-11-09T11:00:00", 1.0),
            run("q1", "quality", 10101, "fu1", "2026-11-09T10:00:00", 2.0),
            run("v3", "verifier", 10102, "fr", "2026-11-09T12:30:00", 4.0),
            run(
                "v4", "verifier", 10103, "fr", "2026-11-09T12:30:00", 5.0
            ),  # never posted
            run("e1", "experiment", 10101, "fu1", "2026-11-09T10:00:00", 9.0),
            run("n1", "notice", 10102, "", "2026-11-09T10:00:00", 0.5),
            run("o1", "other", None, "", "2026-11-09T10:00:00", 7.0),
            run("v0", "verifier", 10101, "fr", "2026-11-09T08:00:00", 6.0),
        ]  # the first post's own
        turns = [
            turn("2026-11-09T11:30:00", 2.0, [10101, 10102]),  # split: one each
            turn("2026-11-09T12:10:00", 1.0, [10101]),  # after the post: last one, 6 h
            turn(
                "2026-11-09T11:00:00", 0.4, [10101], ["experiment"]
            ),  # a replay: not review
            turn("2026-11-09T11:00:00", 0.3, [], ["other"]),  # another task
            turn(
                "2026-11-09T11:00:00", 0.25, [10101, 10102, 10104, 10105, 10106]
            ),  # too many: overhead
            turn("2026-11-09T11:00:00", 0.6),  # nothing named: overhead
            turn("2026-11-09T08:30:00", 50.0),
        ]  # before the first post
        d = self.data(runs, turns)
        p0, p1, p2 = d["reviews"][:3]
        self.assertEqual((p0["total"], p0["parts"]["verifier"]), (6.0, 6.0))
        self.assertAlmostEqual(p1["parts"]["verifier"], 3.0)
        self.assertAlmostEqual(p1["parts"]["verifier2"], 1.0)
        self.assertAlmostEqual(p1["parts"]["lane"], 2.0)
        self.assertAlmostEqual(p1["parts"]["orchestration"], 1.0 + 1.0)
        self.assertEqual(p1["round"], "fu1")
        self.assertAlmostEqual(p2["parts"]["verifier"], 4.0)
        self.assertAlmostEqual(p2["parts"]["orchestration"], 1.0)
        self.assertEqual((p2["round"], p0["round"]), ("fr", "fr"))
        ov = d["overhead"]["2026-11-09"]
        self.assertAlmostEqual(ov["unfollowed"], 5.0)
        self.assertAlmostEqual(ov["notice"], 0.5)
        self.assertAlmostEqual(ov["orchestration"], 0.85)
        self.assertAlmostEqual(d["other"]["2026-11-09"]["experiment"], 9.4)
        self.assertAlmostEqual(d["other"]["2026-11-09"]["other"], 7.3)
        self.assertEqual([r["id"] for r in d["unfollowed"]], ["v4"])

    def test_orchestration_over_its_runs_from_shared_turns_only_is_suspect(self):
        """The optional guard: a post whose orchestration comes only from turns that name it beside the PR each one
        reviews, and outweighs its verifier and lane, is suspect."""
        self.ledger(
            [
                ("2026-11-04T11:55:00Z", 13131, "h1", "followup"),
                ("2026-11-04T12:00:00Z", 14141, "h2", "first_round"),
                ("2026-11-04T12:05:00Z", 14142, "h3", "first_round"),
            ]
        )
        runs = [
            run("v1", "verifier", 13131, "fu6", "2026-11-04T10:30:00", 2.0),
            run("v2", "verifier", 14141, "fr", "2026-11-04T10:30:00", 2.0),
            run("v3", "verifier", 14142, "fr", "2026-11-04T10:30:00", 1.0),
        ]
        turns = [
            turn("2026-11-04T11:00:00", 6.0, [13131, 14141]),
            turn("2026-11-04T11:10:00", 6.0, [13131, 14141]),
            turn("2026-11-04T11:20:00", 1.0, [14141]),
            turn("2026-11-04T11:30:00", 4.0, [14142]),
        ]
        d = self.data(runs, turns)
        by = {p["pr"]: p for p in d["reviews"]}
        self.assertTrue(by[13131]["suspect"])  # 6.0 orchestration over 2.0, never alone
        self.assertFalse(by[14141]["suspect"])  # alone in one turn
        self.assertFalse(by[14142]["suspect"])  # alone, though over its verifier
        md = spend.report_md(self.cfg, d, day="2026-11-04")
        self.assertIn(
            f"Suspect attribution ({spend.SUSPECT_NOTE}; check its turns): #"
            f"13131 fu6 orchestration 6.00 "
            "vs verifier + lane 2.00",
            md,
        )
        self.assertIn("| 6.00 (suspect) |", md)

    def test_a_deep_review_s_verifier_is_its_post_s_verifier(self):
        """A deep review's verifier run is a verifier run: its post's verifier, a later one its second verifier, and
        a turn that starts or hears back from one is review work."""
        self.ledger([("2026-11-09T12:00:00Z", 10101, "h1", "followup")])
        d = self.data(
            [
                run("v1", "deep_verifier", 10101, "fu1", "2026-11-09T10:00:00", 3.0),
                run("q1", "quality", 10101, "fu1", "2026-11-09T10:30:00", 2.0),
                run("v2", "verifier", 10101, "fu1", "2026-11-09T11:00:00", 1.0),
            ],
            [
                turn(
                    "2026-11-09T11:00:00",
                    0.5,
                    [10101],
                    ["deep_verifier", "experiment"],
                )
            ],
        )
        parts = d["reviews"][0]["parts"]
        for part, dollars in (
            ("verifier", 3.0),
            ("verifier2", 1.0),
            ("lane", 2.0),
            ("orchestration", 0.5),
        ):
            self.assertAlmostEqual(parts[part], dollars, msg=part)
        self.assertEqual((d["other"], d["unfollowed"]), ({}, []))

    def test_a_run_too_long_before_a_review_is_overhead(self):
        self.ledger([("2026-11-09T12:00:00Z", 10101, "h1", "followup")])
        d = self.data(
            [
                run("v1", "verifier", 10101, "fu1", "2026-11-09T12:00:00", 1.0),
                run("v0", "verifier", 10101, "fu1", "2026-11-12T13:00:00", 1.0),
            ],
            [],
        )
        self.assertEqual([r["id"] for r in d["unfollowed"]], ["v0"])

    def test_a_review_kept_unposted_is_a_review_with_its_runs(self):
        """While review posting is off a review is kept unposted: a line of review_cost.jsonl and no ledger entry. It
        is a paid review: the runs and turns before it are its verifier, lane and orchestration, never
        overhead, its round gives its kind, and the reviews table marks it. The overhead's runs are the runs no review
        followed: the reports never call them unposted."""
        self.ledger([("2026-11-09T09:00:00Z", 10101, "h0", "first_round")])
        self.costs(
            [
                ("2026-11-09T09:00:00Z", 10101, "h0", "fr", False),
                ("2026-11-09T12:00:00Z", 10101, "h1", "fu1", True),
                ("2026-11-09T13:00:00Z", 12345, "h5", "fr", True),
            ]
        )
        runs = [
            run("v0", "verifier", 10101, "fr", "2026-11-09T08:00:00", 6.0),
            run("v1", "verifier", 10101, "fu1", "2026-11-09T10:00:00", 3.0),
            run("q1", "quality", 10101, "fu1", "2026-11-09T10:30:00", 2.0),
            run("v2", "verifier", 12345, "fr", "2026-11-09T12:30:00", 4.0),
        ]
        d = self.data(runs, [turn("2026-11-09T12:40:00", 1.0, [12345])])
        self.assertEqual(
            [
                (p["pr"], p["round"], p["kind"], p["unposted"], p["total"])
                for p in d["reviews"]
            ],
            [
                (10101, "fr", "first_round", False, 6.0),
                (10101, "fu1", "followup", True, 5.0),
                (12345, "fr", "first_round", True, 5.0),
            ],
        )
        parts = d["reviews"][2]["parts"]
        self.assertEqual((parts["verifier"], parts["orchestration"]), (4.0, 1.0))
        self.assertEqual((d["unfollowed"], dict(d["overhead"])), ([], {}))
        md = spend.report_md(self.cfg, d, day="2026-11-09")
        self.assertIn("| 13:00Z (kept unposted) | #12345 | fr | first round |", md)
        self.assertIn("runs no review followed", md)
        self.assertEqual(md.count("unposted"), md.count("(kept unposted)"))
        self.assertNotIn("unposted", spend.card_note({"at": "2026-11-10T12:00:00Z"}))
        # Its own lines say what a review is: posted, or kept.
        self.assertIn(
            "_Reviews: each review posted, or kept while review posting is off, on the day it was posted or "
            "kept.",
            md,
        )
        self.assertTrue(
            spend.card_note({"at": "2026-11-10T12:00:00Z"}).startswith(
                "Reviews: posted, or kept while review posting is off. Total $: "
            )
        )
        self.assertIn(
            "\n| Posted or kept | PR | Round | Kind | Size tier | Verifier | Second verifier | Code quality pass "
            "| Orchestration | Total $ |\n|---|---|---|---|---|---:|---:|---:|---:|---:|\n",
            md,
        )

    def test_the_review_a_follow_up_follows_up_on(self):
        """A post follows up on our previous post, whatever was kept unposted between them: our last review is the
        one GitHub shows once review posting is on. A kept review follows up on our previous review of either kind.
        A post's round is its own line's, never that of a review kept later at the same head.
        """
        self.ledger(
            [
                ("2026-11-09T09:00:00Z", 10101, "h0", "first_round"),
                ("2026-11-09T15:00:00Z", 10101, "h3", "followup"),
            ]
        )
        self.costs(
            [
                ("2026-11-09T09:00:00Z", 10101, "h0", "fr", False),
                ("2026-11-09T10:00:00Z", 10101, "h0", "fu1", True),
                ("2026-11-09T11:00:00Z", 10101, "h1", "fu2", True),
                ("2026-11-09T12:00:00Z", 10101, "h2", "fu3", True),
                ("2026-11-09T15:00:00Z", 10101, "h3", "fu4", False),
            ]
        )
        d = self.data([], [])
        self.assertEqual(
            [(p["round"], p["head"], p["old"]) for p in d["reviews"]],
            [
                ("fr", "h0", ""),
                ("fu1", "h0", "h0"),
                ("fu2", "h1", "h0"),
                ("fu3", "h2", "h1"),
                ("fu4", "h3", "h0"),
            ],
        )


class TestFigures(Booking):
    def setUp(self):
        super().setUp()
        rows, runs = [], []
        costs = {"2026-11-09": [5, 6, 7, 8, 40], "2026-11-10": [10, 12]}
        for day, cs in costs.items():
            for i, c in enumerate(cs):
                pr = 10100 + len(rows)
                rows.append(
                    (
                        f"{day}T{10 + i:02d}:00:00Z",
                        pr,
                        f"h{pr}",
                        "followup" if i else "first_round",
                    )
                )
                runs.append(
                    run(f"v{pr}", "verifier", pr, "fu1", f"{day}T{9 + i:02d}:30:00", c)
                )
        self.ledger(rows)
        self.store.journal_path.write_text(
            "2026-11-09T11:00:00Z review   #"
            "10101 follow-up skipped: rebase-only (aaaaaaaa -> bbbbbbbb); our review "
            "stands\n2026-11-10T11:00:00Z review   #"
            "10102 follow-up skipped: merge-only (cccccccc -> dddddddd); our "
            "review stands\n"
        )
        self.d = self.data(runs, [turn("2026-11-10T09:00:00", 3.0)])

    def test_day_week_cumulative_and_outliers(self):
        f = spend.period(self.d, ["2026-11-09"])
        s = f["stats"]
        self.assertEqual(
            (s["n"], s["total"], s["mean"], s["median"], s["min"], s["max"]),
            (5, 66, 13.2, 7, 5, 40),
        )
        self.assertAlmostEqual(s["p90"], 27.2)  # 8 + 0.6 x (40 - 8)
        self.assertEqual([p["pr"] for p in f["outliers"]], [10104])
        self.assertEqual(f["covered"], {"merge_only": 0, "rebase": 1})
        self.assertEqual(
            spend.period(self.d, ["2026-11-10"])["covered"],
            {"merge_only": 1, "rebase": 0},
        )
        cum = spend.cumulative(self.d, "2026-11-10")
        self.assertEqual(cum, {"2026-11-09": 66.0, "2026-11-10": 66.0 + 22 + 3})
        w = spend.period(self.d, spend.week_days("2026-W46"))
        self.assertEqual((w["stats"]["n"], w["overhead"]), (7, 3.0))
        self.assertEqual(spend.week_of("2026-11-09"), "2026-W46")
        self.assertEqual(spend.driver(f["outliers"][0]), "verifier")

    def test_report_files_and_breakdowns(self):
        text = spend.report_md(self.cfg, self.d)
        self.assertIn(
            "| 2026-11-09 | 5 | 0 / 1 | 66.00 | 0.00 | 66.00 | 13.20 | 7.00 |", text
        )
        self.assertIn("| 2026-W46 | 7 |", text)
        self.assertIn(
            "| #" "10104 | fu1 | follow-up | 40.00 | <1.5k | ? | verifier |", text
        )
        self.assertNotIn("Experiments", text, "experiments only on request")
        self.assertIn(
            "Experiments and other tasks",
            spend.report_md(self.cfg, self.d, show_experiments=True),
        )
        kinds = spend.report_md(self.cfg, self.d, day="2026-11-09", by="kind")
        self.assertIn("| first round | 1 | 5.00 |", kinds)
        self.assertIn("### Reviews", kinds)
        self.assertIn(
            "| <1.5k | 7 |",
            spend.report_md(self.cfg, self.d, week="2026-W46", by="size"),
        )
        self.assertIn(
            "| #" "10104 | 1 | 40.00 |", spend.report_md(self.cfg, self.d, by="pr")
        )
        paths = spend.write_files(self.cfg, self.d, ["2026-11-09"])
        self.assertEqual(
            [p.name for p in paths], ["spend_2026-11-09.md", "spend_2026-W46.md"]
        )
        self.assertTrue(
            str(paths[0]).startswith(str(self.base / "out" / "rocke_spend"))
        )
        self.assertIn("## By size", paths[0].read_text(encoding="utf-8"))
        self.assertIn("## PRs by spend", paths[1].read_text(encoding="utf-8"))

    def test_card_md_and_html(self):
        card = spend.card_data(self.d, "2026-11-10T12:00:00Z")
        self.assertEqual((card["today"]["reviews"], card["today"]["total"]), (2, 25.0))
        self.assertEqual(
            (card["week"]["week"], card["week"]["reviews"]), ("2026-W46", 7)
        )
        self.assertEqual([r["day"] for r in card["days"]], ["2026-11-09", "2026-11-10"])
        self.assertEqual(
            [(o["pr"], o["driver"]) for o in card["outliers"]], [(10104, "verifier")]
        )
        md = "\n".join(spend.card_md(card))
        self.assertIn("| Today (2026-11-10) | 2 | 25.00 | 11.00 | 11.00 |", md)
        self.assertIn("**Outliers this week:** #" "10104 fu1 $40.00 (verifier)", md)
        page = webdash.spend_card(card)
        self.assertIn("<section id='spend' class='card'>", page)
        self.assertIn("#" "10104 fu1 $40.00 (verifier)", page)
        for text in (md, page):
            self.assertNotIn("xperiment", text, "review workload only")
        self.assertIn("Not computed yet", webdash.spend_card(None))
        self.assertIn("Not computed yet", "\n".join(spend.card_md(None)))

    def test_the_md_dashboard_ends_with_the_card(self):
        st = empty_state()
        cfg = {"deployment": {"phase": "review"}, "merge_authority": "propose"}
        out = report.dashboard(
            cfg, st, [], [], ["j"], None, spend.card_md(spend.card_data(self.d))
        )
        self.assertLess(out.index("## Journal (latest)"), out.index("## Review spend"))
        self.assertTrue(out.rstrip().endswith("GitHub reads only._"))
        self.assertNotIn("## Review spend", report.dashboard(cfg, st, [], [], ["j"]))


class TestTick(Booking):
    def test_card_hourly_and_yesterdays_files_once(self):
        self.ledger([("2026-11-09T10:00:00Z", 10101, "h", "first_round")])
        d = self.data(
            [run("v", "verifier", 10101, "fr", "2026-11-09T09:00:00", 2.0)], []
        )
        st = {}
        with mock.patch.object(
            spend, "collect", return_value=d
        ) as col, mock.patch.object(
            spend, "now", return_value="2026-11-10T01:00:00Z"
        ), mock.patch.object(
            spend,
            "minutes_since",
            side_effect=lambda ts: (
                dt.datetime(2026, 11, 10, 1, tzinfo=dt.timezone.utc)
                - dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=dt.timezone.utc
                )
            ).total_seconds()
            / 60,
        ):
            lines = spend.tick(self.cfg, st, self.store)
            self.assertEqual(len(lines), 1)
            self.assertIn("spend files for 2026-11-09 written", lines[0])
            self.assertEqual(
                st["spend"], {"at": "2026-11-10T01:00:00Z", "files": "2026-11-09"}
            )
            self.assertTrue(
                (self.base / "out" / "rocke_spend" / "spend_2026-11-09.md").exists()
            )
            self.assertEqual(spend.load_card(self.store)["today"]["day"], "2026-11-10")
            self.assertEqual(
                spend.tick(self.cfg, st, self.store),
                [],
                "refreshed less than an hour ago",
            )
            self.assertEqual(col.call_count, 1)

    def test_files_wait_until_the_day_has_ended_and_the_transcripts_are_read(self):
        self.ledger([("2026-11-09T10:00:00Z", 10101, "h", "first_round")])
        d = dict(self.data([], []), unread_bytes=10)
        st = {}
        with mock.patch.object(spend, "collect", return_value=d):
            self.assertEqual(spend.tick(self.cfg, st, self.store), [])
        self.assertNotIn("files", st["spend"])

    def test_no_prices_or_folder_does_nothing_and_a_failure_is_a_line(self):
        self.assertEqual(spend.tick({"paths": {}}, {}, self.store), [])
        with mock.patch.object(spend, "collect", side_effect=OSError("boom")):
            self.assertEqual(
                spend.tick(self.cfg, {}, self.store), ["(spend not refreshed: boom)"]
            )

    def test_files_due_timer(self):
        st = {"spend": {"files": "2026-11-08"}}
        with mock.patch.object(spend, "now", return_value="2026-11-10T01:00:00Z"):
            self.assertEqual(
                list(spend._files_due(self.cfg, st, self.store)),
                [("spend files", "2026-11-10T00:15:00Z")],
            )
            st["spend"]["files"] = "2026-11-09"
            self.assertEqual(list(spend._files_due(self.cfg, st, self.store)), [])


class TestCollectAndWeekly(Booking):
    def test_collect_from_transcripts_and_the_weekly_section(self):
        self.ledger([("2026-11-09T12:00:00Z", 10101, "h1", "followup")])
        write_jsonl(
            self.root / "s1.jsonl",
            [
                user("2026-11-09T10:00:00Z", MON),
                asst(
                    "2026-11-09T10:00:01Z",
                    "o",
                    usage(read=1_000_000, out=0),
                    [bash("rk post 10101 x")],
                ),
            ],
        )
        run_files(
            self.root,
            "s1",
            "a1",
            "rocke-review-verifier",
            "Verify PR " "10101 follow-up 1",
            "2026-11-09T11:00:00Z",
            usage(read=5_000_000, out=0),
            text="",
        )
        d = spend.collect(self.cfg, self.store)
        p = d["reviews"][0]
        self.assertAlmostEqual(p["parts"]["verifier"], 1.0)
        self.assertAlmostEqual(
            p["parts"]["orchestration"],
            0.2 + 20 * 0.45 * len(json.dumps({"command": "rk post 10101 x"})) / 1e6,
        )
        self.assertEqual((p["round"], p["size"]["tier"]), ("fu1", "?"))
        lines = spend.weekly_section(self.cfg, self.store, "2026-W46")
        self.assertEqual(lines[0], "## Review spend")
        self.assertTrue(any(x.startswith("| 2026-11-09 | 1 |") for x in lines))
        self.assertIn(
            "Not available", "\n".join(spend.weekly_section({}, self.store, "2026-W46"))
        )

    def test_a_history_counts_before_its_until_and_the_transcripts_after(self):
        until = "2026-11-09T12:00:00Z"
        self.ledger(
            [
                ("2026-11-09T11:00:00Z", 10101, "h1", "followup"),
                ("2026-11-09T14:00:00Z", 10102, "h2", "followup"),
            ]
        )
        old = {
            "role": "verifier",
            "pr": 10101,
            "round": "fu3",
            "type": "rocke-review-verifier",
            "desc": "",
            "use": {"opus": [0, 0, 0, 5_000_000, 0, 0, 1]},
        }
        (self.store.dir / spend.HISTORY).write_text(
            json.dumps(
                {
                    "until": until,
                    "runs": [
                        dict(old, id="h1", start="2026-11-09T10:00:00Z"),
                        dict(old, id="late", start="2026-11-09T12:30:00Z", pr=10102),
                    ],
                    "turns": [
                        {
                            "t": "2026-11-09T10:59:00Z",
                            "nums": ["10101"],
                            "roles": [],
                            "use": {"opus": [0, 0, 0, 1_000_000, 0, 0, 1]},
                        }
                    ],
                }
            )
        )
        write_jsonl(self.root / "s1.jsonl", [user("2026-11-09T09:00:00Z", MON)])
        for aid, desc, t in (
            ("shadow", "Verify PR " "10101 follow-up 3", "2026-11-09T10:30:00Z"),
            ("new", "Verify PR " "10102 follow-up 1", "2026-11-09T13:00:00Z"),
        ):
            run_files(
                self.root,
                "s1",
                aid,
                "rocke-review-verifier",
                desc,
                t,
                usage(read=10_000_000, out=0),
                text="",
            )
        p1, p2 = spend.collect(self.cfg, self.store)["reviews"]
        self.assertEqual(
            (p1["parts"]["verifier"], p1["parts"]["verifier2"], p1["round"]),
            (1.0, 0.0, "fu3"),
            "the history's run, priced; this machine's run before `until` is not counted",
        )
        self.assertAlmostEqual(p1["parts"]["orchestration"], 0.2)
        self.assertEqual(
            (p2["parts"]["verifier"], p2["parts"]["verifier2"]),
            (2.0, 0.0),
            "this machine's run from `until` on; the history's past `until` is not counted",
        )
        (self.store.dir / spend.HISTORY).write_text("not json")
        p1, p2 = spend.collect(self.cfg, self.store)["reviews"]
        self.assertEqual(
            (p1["parts"]["verifier"], p2["parts"]["verifier"]),
            (2.0, 2.0),
            "no history: every transcript counts",
        )

    def test_cli_writes_the_card_and_the_files(self):
        self.ledger([("2026-11-09T12:00:00Z", 10101, "h1", "followup")])
        write_jsonl(self.root / "s1.jsonl", [user("2026-11-09T10:00:00Z", MON)])
        args = types.SimpleNamespace(
            day=None, week=None, since=None, by="part", experiments=False, write=True
        )
        out = spend.cli(self.cfg, self.store, args)
        self.assertIn("## Review spend (UTC)", out)
        self.assertIn("spend_2026-11-09.md", out)
        self.assertIsNotNone(spend.load_card(self.store))


if __name__ == "__main__":
    unittest.main()
