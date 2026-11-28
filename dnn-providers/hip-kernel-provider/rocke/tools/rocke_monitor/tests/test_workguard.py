# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The session's Bash guard keeps the agents off the state they share (rkmon_lib/workguard.py, through
orchestrator.guard and its PreToolUse hook): git config writes, PR refs fetched without `+`, worktrees outside a review
directory's wt_v*/wt_q* paths or whose folder `rk pr-tree` did not make first in the same command, and writes into the
checks' shared build or its tree (a build in a slot's folder only while the slot is kept for it). Each refusal says
what to run instead; reads and an agent's own work pass. The script a shell reads from a here-document (the agents'
`bash -s <<'EOF'` steps) is read as commands; any other here-document is data. `rk review-prep` prints the two
worktree paths and how each is added.
"""

import asyncio
import contextlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import orchestrator, prep, slots, workguard  # noqa: E402

B = "/work/build/wt7"  # a shared build
T = "/ramdisk/workers/pr7/wt7"  # its tree
SHARED = (B, T)
RD = "/work/scratch/tmp/v7fu2"  # a review directory


class TestGitConfig(unittest.TestCase):
    def test_a_write_is_refused_and_says_git_c(self):
        for cmd in (
            "git config user.email a@b",
            "git -C /work/repo config core.sparseCheckout true",
            "cd /work/wt && git config --add remote.origin.fetch x",
            "git config --unset core.hooksPath",
            "git config --global safe.directory '*'",
            "bash -lc 'git config gc.auto 0'",
            "git log -1 && git config pull.rebase true",
        ):
            why = orchestrator.guard(cmd)
            self.assertIsNotNone(why, cmd)
            self.assertIn("git -c KEY=VALUE COMMAND", why)

    def test_reads_and_a_setting_for_one_command_pass(self):
        for cmd in (
            "git config --get remote.origin.url",
            "git config -l",
            "git config --list --show-origin",
            "git config get core.bare",
            "git config core.bare",
            "git -c core.hooksPath=/dev/null -C /work/wt commit -m x",
            "git -c gc.auto=0 fetch origin develop",
            "echo 'never git config x y'",
            "python3 - <<'EOF'\nprint('git config a b')\nEOF",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)


class TestPrFetch(unittest.TestCase):
    def test_a_pr_ref_without_plus_or_into_fetch_head_is_refused(self):
        for cmd in (
            "git -C /work/repo fetch origin pull/7/head:refs/x/pr7",
            "git fetch origin refs/pull/7/head:refs/remotes/origin/pr7",
            "git fetch -q --filter=blob:none origin pull/7/head",
            "git fetch origin +pull/7/head",
        ):
            why = orchestrator.guard(cmd)
            self.assertIsNotNone(why, cmd)
            self.assertIn("+pull/<N>/head:refs/remotes/origin/pr<N>", why)

    def test_a_forced_fetch_into_a_named_ref_passes(self):
        for cmd in (
            "git fetch origin +pull/7/head:refs/remotes/origin/pr7",
            "git -C /work/repo fetch -q origin +refs/pull/7/head:refs/rk/pr7 +develop:refs/remotes/origin/develop",
            "git fetch origin develop",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)


class TestWorktree(unittest.TestCase):
    def test_a_worktree_outside_wt_v_and_wt_q_is_refused(self):
        for cmd in (
            f"git -C /work/repo worktree add {RD}/wt abc123",
            "git worktree add /tmp/mytree HEAD",
            f"git worktree add -b x {RD}/tree",
            "git worktree add --detach ../wt_v1",  # a parent that is no review directory
            f"git worktree add /work/scratch/tmp/elsewhere/wt_q HEAD",
            # git reads the path in its -C folder: the mirror's wt_v
            "cd /work/scratch/tmp/v7fr && rk pr-tree wt_v && git -C /work/repo worktree add wt_v HEAD",
        ):
            why = orchestrator.guard(cmd)
            self.assertIsNotNone(why, cmd)
            self.assertIn("rk review-prep N", why)

    def test_the_verifiers_and_the_lanes_paths_made_trees_first_pass(self):
        for cmd in (
            f"rk pr-tree {RD}/wt_v && git -C /work/repo worktree add --detach {RD}/wt_v abc123",
            f"rk pr-tree {RD}/wt_vdev/ && git -C /work/repo worktree add -b mut {RD}/wt_vdev HEAD",
            f"/work/toolkit/rk pr-tree {RD}/wt_q2 && git worktree add {RD}/wt_q2 abc",
            f'rk pr-tree "$D/wt_q/" && git worktree add "$D/wt_q" abc',
            "cd /work/scratch/tmp/v7fr && rk pr-tree wt_v && git worktree add wt_v HEAD",
            f"rk pr-tree {RD}/wt_v1 {RD}/wt_q1\ngit -C /work/repo worktree add {RD}/wt_q1 abc",
            "git worktree list",
            f"git worktree remove --force {RD}/wt",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)

    def test_a_worktree_rk_pr_tree_did_not_make_first_is_refused(self):
        for cmd in (
            f"git -C /work/repo worktree add --detach {RD}/wt_v abc123",
            f"rk pr-tree {RD}/wt_v1 && git worktree add {RD}/wt_v2 abc",  # another folder
            f"git worktree add {RD}/wt_q abc && rk pr-tree {RD}/wt_q",  # after it
            # another program's words
            f"echo pr-tree {RD}/wt_v && git worktree add {RD}/wt_v abc",
            # the folder rk pr-tree made is ./wt_v; git's -C reads the path in the review directory
            "rk pr-tree wt_v && git -C /work/scratch/tmp/v7fr worktree add wt_v HEAD",
        ):
            why = orchestrator.guard(cmd)
            self.assertIsNotNone(why, cmd)
            self.assertIn("rk pr-tree PATH && git -C MIRROR worktree add", why)


class TestHereDocument(unittest.TestCase):
    def test_the_script_a_shell_reads_from_a_here_document_is_read(self):
        for cmd, why in (
            ("bash -s <<'EOF'\ngit config user.email a@b\nEOF", "git -c KEY=VALUE"),
            (
                "bash <<EOF\ncd /work/wt\ngit config --add x y\nEOF",
                "git -c KEY=VALUE",
            ),
            (
                f"nice -n 10 bash -euo pipefail -s <<'EOF' > {RD}/log 2>&1\n"
                f"git -C /work/repo worktree add --detach {RD}/wt_v HEAD\nEOF",
                "rk pr-tree PATH",
            ),
            (
                "sh -s -- 7 <<-EOF\n\tgit fetch origin pull/7/head\n\tEOF",
                "+pull/<N>/head",
            ),
            ("bash - <<EOF\ngit config a b\nEOF", "git -c KEY=VALUE"),
            ("bash -- <<EOF\ngit config a b\nEOF", "git -c KEY=VALUE"),
            # its redirections name no script file
            ("bash <<'EOF' 2>/dev/null\ngit config a b\nEOF", "git -c KEY=VALUE"),
            (
                "echo start\nbash -s <<'A'\necho one\nA\nbash -s <<'B'\ngit config a b\nB",
                "git -c KEY=VALUE",
            ),
            # two on one line: each its own body
            (
                "cat <<'A'; bash -s <<'B'\necho data\nA\ngit config a b\nB",
                "git -c KEY=VALUE",
            ),
            # a body ends at its end line, tab-indented after <<-
            ("cat <<-EOF\n\tdata\n\tEOF\ngit config a b", "git -c KEY=VALUE"),
            ("bash -s <<'EOF'\ngit config a b", "git -c KEY=VALUE"),  # no end line
            ("cat <<< word\ngit config a b", "git -c KEY=VALUE"),  # a here-string
        ):
            self.assertIn(why, orchestrator.guard(cmd) or "", cmd)

    def test_its_shared_build_and_tree_rules_hold_there(self):
        for cmd in (
            f"bash -s <<'EOF'\ncd {B}/core && ninja\nEOF",
            f"cd {B}/core && bash -s <<'EOF'\nninja\nEOF",
        ):
            self.assertIn(
                "read-only", orchestrator.guard(cmd, shared=SHARED) or "", cmd
            )
        for cmd in (
            f"bash -s <<'EOF'\nrk pr-tree {RD}/wt_v && git -C /work/repo worktree add --detach {RD}/wt_v abc\nEOF",
            "bash -s <<'EOF'\nrk pr-tree $RD/wt_q1 && git -C $REPO worktree add --detach $RD/wt_q1 $SHA\nEOF",
            # a tree made before the script, in the same command
            f"rk pr-tree {RD}/wt_v\nbash -s <<'EOF'\ngit -C /work/repo worktree add --detach {RD}/wt_v abc\nEOF",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)

    def test_any_other_here_document_is_data(self):
        for cmd in (
            "cat > /tmp/x.sh <<'EOF'\ngit config a b\nEOF",
            "bash /tmp/x.sh <<'EOF'\ngit config a b\nEOF",  # a script file's input
            "bash -c 'cat' <<'EOF'\ngit config a b\nEOF",  # -c's input
            "bash -o pipefail x.sh <<EOF\ngit config a b\nEOF",
            "python3 - <<'EOF'\nimport os\nos.system('git config a b')\nEOF",
        ):
            self.assertIsNone(orchestrator.guard(cmd), cmd)


class TestSharedBuild(unittest.TestCase):
    def test_writes_into_the_shared_build_or_its_tree_are_refused(self):
        for cmd in (
            f"cd {B}/core && /usr/bin/ar qc librocke_core.a x.o",
            f"ar rcs {B}/core/librocke_core.a a.o",
            f"ninja -C {B}/core rocke_core",
            f"cd {B}/core && ninja",
            f"cmake --build {B}/core -j4",
            f"cmake -S {T}/rocke -B {B}/core",
            f"rm -rf {B}/pybind",
            f"cp mine.so {B}/pybind/",
            f"touch {T}/rocke/x.py",
            f"sed -i s/a/b/ {T}/rocke/platform/x.py",
            f"git -C {T} checkout -- .",
            f"git -C {T} apply fix.patch",
            f"echo x > {B}/core/marker",
            f"tar -xzf a.tgz -C {B}",
            f"tar czf {B}/out.tgz src",
            f"unzip -d {T} a.zip",
            f"nice -n 10 timeout 600 ninja -C {B}/core",
            f"bash -lc 'cd {B}/core && ninja rocke_core'",
        ):
            why = orchestrator.guard(cmd, shared=SHARED)
            self.assertIsNotNone(why, cmd)
            self.assertIn("read-only", why)

    def test_reading_it_and_building_elsewhere_pass(self):
        for cmd in (
            f"ctest --test-dir {B}/core -R abi --output-on-failure",
            f"ninja -C {B}/core -t commands rocke_core | grep x.cpp",
            f"PYTHONPATH={B}/pybind python -B -c 'import rocke_engine'",
            f"cp -a {B}/install {RD}/wt_v/install",
            f"git -C {T} rev-parse HEAD",
            f"git -C {T} diff --stat",
            f"cat {B}/core/compile_commands.json > {RD}/cc.json",
            f"tar -xzf {B}/x.tgz -C {RD}/wt_v",
            f"ninja -C {RD}/wt_v/build",
            f"cmake --build {RD}/wt_vdev/build",
            f"ls {B}; grep -r x {T}/rocke",
        ):
            self.assertIsNone(orchestrator.guard(cmd, shared=SHARED), cmd)

    def test_without_a_shared_build_nothing_is_one(self):
        self.assertIsNone(orchestrator.guard(f"ninja -C {B}/core"))

    def test_the_shared_builds_are_the_checks_records(self):
        with tempfile.TemporaryDirectory() as d:
            w = pathlib.Path(d)
            for n, line in (
                (7, f"{B}\n{T}\n{'a' * 40}\n"),
                (8, "garbage\n"),
                (10, f"/work/build dir/wt10\n/work/tree dir/wt10\n{'b' * 40}\n"),
            ):
                (w / f"pr{n}" / "checks").mkdir(parents=True)
                (w / f"pr{n}" / "checks" / "shared_build").write_text(line)
            (w / "pr9" / "checks").mkdir(parents=True)  # no record
            self.assertEqual(
                workguard.shared_paths(str(w)),
                ["/work/build dir/wt10", "/work/tree dir/wt10", B, T],  # a space: whole
            )
            self.assertEqual(workguard.shared_paths(str(w / "none")), [])

    def test_a_build_in_a_slot_is_shared_only_while_the_slot_is_kept_for_it(self):
        with tempfile.TemporaryDirectory() as d:
            top, locks = pathlib.Path(d) / "slots", pathlib.Path(d) / "slots" / "locks"
            locks.mkdir(parents=True)
            sb = f"{top}/rkpr2/build/wt7m"
            w = pathlib.Path(d) / "workers"
            (w / "pr7" / "checks").mkdir(parents=True)
            (w / "pr7" / "checks" / "shared_build").write_text(
                f"{sb}\n{T}\n{'a' * 40}\n"
            )
            with mock.patch.multiple(L, SLOTS_DIR=str(top), SLOT_LOCKS=str(locks)):
                self.assertEqual(workguard.shared_paths(str(w)), [])
                slots.Records().write("rkpr2", {"state": "kept", "build": sb})
                self.assertEqual(workguard.shared_paths(str(w)), [sb, T])


class TestOutwardFirst(unittest.TestCase):
    def test_the_outward_guard_still_answers_first_and_unsplittable_text_passes_here(
        self,
    ):
        self.assertIn(
            "goes through an rk command",
            orchestrator.guard("git config a b; gh pr comment 7 -b x"),
        )
        self.assertIsNone(workguard.check("echo 'unbalanced"))


class TestHook(unittest.TestCase):
    """The PreToolUse hook on Bash reads the checks' shared build records and denies with the reason."""

    def test_the_hook_denies_with_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            w = pathlib.Path(d) / "workers"
            (w / "pr7" / "checks").mkdir(parents=True)
            (w / "pr7" / "checks" / "shared_build").write_text(
                f"{B}\n{T}\n{'a' * 40}\n"
            )
            cfg = {"paths": {"workers": str(w), "working_dir": d}, "features": {}}
            hooks = {}

            class Matcher:
                def __init__(self, matcher, hooks, timeout):
                    self.matcher, self.hooks = matcher, hooks

            class Options:
                def __init__(self, **kw):
                    hooks.update(kw["hooks"])

            sdk = types.SimpleNamespace(HookMatcher=Matcher, ClaudeAgentOptions=Options)
            records = []
            patches = (
                mock.patch.object(
                    orchestrator, "record", lambda cfg, rec: records.append(rec)
                ),
                mock.patch.object(
                    orchestrator, "settings", lambda cfg: {"workspace": d}
                ),
                mock.patch.object(orchestrator, "allowed_tools", lambda cfg: []),
                mock.patch.object(orchestrator, "session_env", lambda cfg: {}),
                mock.patch.object(orchestrator, "tickets_on", lambda cfg: False),
                mock.patch.object(
                    orchestrator.config, "orchestrator_model", lambda cfg: "m"
                ),
                mock.patch.object(
                    orchestrator.paths, "get", lambda cfg, key: cfg["paths"][key]
                ),
            )
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                orchestrator.options(cfg, sdk, None)
                bash = hooks["PreToolUse"][0].hooks[0]

                def ask(cmd):
                    return asyncio.run(
                        bash(
                            {
                                "tool_input": {"command": cmd},
                                "agent_type": "rocke-review-verifier",
                            },
                            "t1",
                            None,
                        )
                    )

                out = ask(f"ninja -C {B}/core")["hookSpecificOutput"]
                self.assertEqual(out["permissionDecision"], "deny")
                self.assertIn("shared build", out["permissionDecisionReason"])
                self.assertEqual(ask(f"ctest --test-dir {B}/core -R x"), {})
                self.assertEqual(records[0]["agent"], "rocke-review-verifier")


class TestReviewPrepPaths(unittest.TestCase):
    def test_the_worktree_line_names_both_paths_and_the_guard_takes_them(self):
        out = pathlib.Path(RD)
        line = prep.worktree_line(out)
        self.assertEqual(
            line,
            f"worktrees: verifier {RD}/wt_v<name> · code quality reviewer {RD}/wt_q<name>, each added as "
            "`rk pr-tree PATH && git -C MIRROR worktree add --detach PATH COMMIT` "
            "(a worktree added anywhere else, or not made so, is refused)",
        )
        for p in (f"{RD}/wt_v", f"{RD}/wt_vmut", f"{RD}/wt_q", f"{RD}/wt_q2"):
            self.assertTrue(workguard.worktree_ok(p), p)
        self.assertFalse(workguard.worktree_ok(f"{RD}/wt"))


if __name__ == "__main__":
    unittest.main()
