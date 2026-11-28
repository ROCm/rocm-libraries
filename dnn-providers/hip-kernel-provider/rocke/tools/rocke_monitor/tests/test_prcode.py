# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PR code in the PR runners' slots (rkmon_lib/prcode.py, docker/rk_prcode.py), and the start's root steps
(docker/rk_root.py): outside a container, against stand-ins for sudo, the launcher (tests/fakes/fake_slots.py), chgrp
and /proc.

- Every job script that builds or runs a PR's code takes a slot first (`prslot`) and runs each such command through
  `prc` (the launcher in the image, the command itself elsewhere), in the slot's home or a tree (`prc -C`), never in
  a folder of the monitor's; PR code writes only its slot's out, and the monitor copies what it keeps (`prr`) and
  writes develop's caches itself.
- `prc` passes the job's environment to the launcher on stdin, temporary assignments included; without the launcher
  it runs the command in a folder of the session user's that holds the slot's folders. `prc`, `prr` and `prin` need a
  slot taken first (in a subshell, `prslot` takes one of the subshell's own); `prtree` makes a folder for a tree PR
  code reads.
- A job's kill frees the slots its processes took (a kept one stays); rk stop and rk kill ask the launcher to stop
  every runner's processes.
- A review's end (its post, its PR's close, a new head) frees the slots kept and taken for it, at the next advance of
  the reviews.
- `rk pr-slot status` says where PR code runs: each slot free, taken, kept or broken; on a host, as the session user.
  rk doctor names each broken slot with its why: a WARN while other slots are left, a FAIL when none is, its fix the
  container's next start; nothing on a host.
- A slot taken in Python is begun, used and freed; a SIGTERM to its holder frees it too. Named work keeps its slot and
  files until it ends; a host's folder for it that is not this user's folder is refused, and never followed.
- Every Python caller that runs PR code runs it in a slot of its own: the first-pass tools' run_pytest, run_probe and
  run_mutant, the drafter's saved probe and worker/run_suite.py through `rk pr-run`, in a tree or the slot's home, never
  in a folder of the monitor's; rk probe-check, rk mutants, rk import-probe, rk ctest-labels, rk werror and rk pr-run
  themselves.
  The tools that run no PR code (grep, git, the rule scan) and the git-only head checks do not. The re-run commands the
  checks print are rk pr-run's, with the run's environment file in the folder the job copies the results to, which
  each test run is told.
- `rk pr-run` takes a slot, runs the command in it and frees it; named work keeps its slot until `--end`; what the
  result reader refuses fails the run. `rk pr-slot take` prints the slot's folders as shell assignments, `keep` keeps
  it, `free` frees it.
- The start moves an older deployment's write login to the write user's volume, once, and never follows a link.
"""

import contextlib
import grp
import io
import json
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "fakes"))
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import drift, guard, jobs, prcode, review, slots  # noqa: E402
import fake_slots  # noqa: E402
import pr_env  # noqa: E402
import pr_user  # noqa: E402

CFG = {
    "rocke_path": "rk",
    "repo": "o/r",
    "paths": {"working_dir": "/w"},
}
PLAN = {
    "families": ["gemm"],
    "tests": ["rk/platform/tests/test_x.py"],
    "changed_tests": ["rk/platform/tests/test_x.py"],
    "conftests": [],
    "modules": ["rk/platform/python/rocke/x.py"],
    "build": True,
    "golden": True,
    "engine_wide": True,
    "outside_rocke": [],
    "libparity": True,
    "_dirs": {"gemm": "rk/platform/x"},
}
# The commands that build or run a PR's code, as the job scripts start them; the monitor's steps that read what such
# code left as data (the golden report, the installed cache's update, the sweep's history and its carry-only round, git
# and the history only) are not among them.
PR_CODE = re.compile(
    r"build_native\.py|check_byte_identity\.py|-m pytest|golden_vs_base\.py(?! report)|run_diff\.py"
    r"|pytest_vs_base\.py|installed_tests\.py(?! cache-add)|p_emit_compile\.py"
    r"|mutation_sweep\.py(?! append)(?![^;]*--carry-only)|import_probe\.py|run_probes\.py"
    r"|ci_condition\.py|tidy_check\.py|winsim\.py run"
    r"|(?<![\w/.-])ctest\s"
)
# Where a shell command starts, within a line.
COMMAND_START = re.compile(r";|&&|\|\||\||\(|\{\s|\bthen\b|\bdo\b|\belse\b|`")
# A worker tool's path in a script (paths.worker_tool): its test is no run of it.
WORKER_TOOL = re.compile(
    r"\$\(\[ -f \$T/([\w.]+) \] && echo \$T \|\| echo \$M/worker\)/"
)
PREFIX = re.compile(r"^\s*(?:(?:if|!|net)\s+|[A-Za-z_]\w*=\S*\s+)*")
# The checks that read blobs of the clone in their slot: each runs its `--fetch` first, as the monitor (prf).
FETCHING = ("mutation_sweep.py", "ctest_labels.py", "import_probe.py")


def scripts() -> dict[str, str]:
    """Every job script that builds or runs a PR's code (jobs.PR_CODE_KINDS), each section on."""
    out = {
        "checks": review.checks_script(CFG, 7, "a" * 40, "first-round", PLAN, None),
        "checks follow-up": review.checks_script(
            CFG, 7, "a" * 40, "follow-up", PLAN, None, reviewed="b" * 40
        ),
        "checks merge-only": review.checks_script(
            CFG,
            7,
            "a" * 40,
            "follow-up",
            PLAN,
            None,
            reviewed="b" * 40,
            merge_only=True,
        ),
        "gate": review.checks_script(CFG, 7, "a" * 40, "gate", PLAN, None, mode="gate"),
        "sanity": guard.sanity_script(
            CFG, "a" * 40, "/x/wtsanity", "/x/out", ["gemm"], True
        ),
        "bisect": guard.bisect_script(
            CFG,
            "INC-1",
            [("b" * 40, 12)],
            ["build", "golden", "dg", "byteid"],
            ["gemm"],
        ),
        "drift": drift.script(
            CFG,
            "/w/workers/drift/x",
            "d" * 40,
            [{"n": 3, "head": "e" * 40, "tests": ["rk/platform/tests/test_x.py"]}],
        ),
    }
    return out


def unwrapped(script: str) -> list[str]:
    """The PR-code commands of `script` that do not start with `prc`."""
    bad = []
    for line in script.splitlines():
        line = WORKER_TOOL.sub(r"$T/", line)
        for m in PR_CODE.finditer(line):
            starts = [s.end() for s in COMMAND_START.finditer(line, 0, m.start())]
            cmd = line[(starts[-1] if starts else 0) : m.end()]
            cmd_ = PREFIX.sub("", cmd)
            if cmd_.startswith(
                ("echo ", "st ")
            ):  # it prints a tool's name, it runs nothing
                continue
            if cmd_.startswith("prf "):  # the tool's --fetch, as the monitor: git only
                continue
            if not cmd_.startswith("prc "):
                bad.append(cmd.strip()[-160:])
    return bad


def given_out(script: str) -> list[str]:
    """The prc commands of `script` given the monitor's out ($OUT) as an argument. A redirect is the monitor's shell
    writing it, and a command substitution ($(...)) the monitor's shell reading it before the command starts.
    """
    prev = None
    while prev != script:
        prev, script = script, re.sub(r"\$\([^()]*\)", "$(...)", script)
    return re.findall(r"\bprc [^;|&<>]*\$OUT\b[^;|&<>]*", script)


class TestWiring(unittest.TestCase):
    def test_every_pr_code_command_runs_through_prc(self):
        for name, s in scripts().items():
            with self.subTest(script=name):
                self.assertIn(prcode.SHELL, s)
                self.assertEqual(unwrapped(s), [])
                self.assertEqual(
                    subprocess.run(
                        ["bash", "-n"], input=s, capture_output=True, text=True
                    ).returncode,
                    0,
                )
        checks = scripts()["checks"]
        for marker in (
            "build_native.py",
            "ctest -j",
            "check_byte_identity.py",
            "-m pytest",
            "golden_vs_base.py",
            "run_diff.py",
            "pytest_vs_base.py",
            "installed_tests.py",
            "p_emit_compile.py",
            "mutation_sweep.py",
            "import_probe.py",
            "run_probes.py",
        ):
            self.assertIn(marker, checks)  # the check above saw each of them

    def test_the_check_catches_a_command_without_prc(self):
        self.assertEqual(
            unwrapped("x; python3 -I -B $M/tools/pytest_vs_base.py $OUT"),
            ["python3 -I -B $M/tools/pytest_vs_base.py"],
        )
        self.assertEqual(
            unwrapped("(cd $BD/x/core && timeout 1800 ctest -j4 > $OUT/ctest.log)"),
            ["timeout 1800 ctest"],
        )
        self.assertEqual(
            unwrapped("(cd $P && ROCKE_LLVM_FLAVOR=$FL prc timeout 9 ctest -j1)"), []
        )
        self.assertEqual(unwrapped("grep x $OUT/ctest.log; ctest_labels.py"), [])
        for monitors in (
            "python3 -I -B $M/tools/golden_vs_base.py report $OUT/golden $P/tests self",
            'python3 -I -B $M/tools/installed_tests.py cache-add $DO "$IC"',
            "timeout 60 python -I -B $M/tools/mutation_sweep.py append --out $OUT/mutants",
            "timeout 120 python -I -B $M/tools/mutation_sweep.py --repo $R0 --out $OUT/mutants --carry-only",
        ):
            self.assertEqual(unwrapped(monitors), [], monitors)
        self.assertEqual(
            unwrapped(
                "python3 -I -B $M/tools/golden_vs_base.py gate $SO/golden $P/tests"
            ),
            ["python3 -I -B $M/tools/golden_vs_base.py"],
        )
        self.assertEqual(
            unwrapped(
                "timeout 60 python -I -B $M/tools/mutation_sweep.py --repo $R0 --out $SO/mutants"
            ),
            ["timeout 60 python -I -B $M/tools/mutation_sweep.py"],
        )
        self.assertEqual(
            unwrapped(
                "net prf python -I -B $M/tools/import_probe.py --repo $R0 > $OUT/r 2>&1; "
                "net python -I -B $M/tools/import_probe.py --repo $R0 >> $OUT/r 2>&1"
            ),
            ["net python -I -B $M/tools/import_probe.py"],
        )

    def test_each_check_that_reads_the_clone_fetches_first(self):
        """The sweep, the labels and the import probe read blobs of the clone in their slot, where git fetches
        nothing: the same command's `--fetch` runs first, as the monitor (prf), into the same report.
        """
        line = re.compile(
            r'^net prf (?P<fetch>.+?) > "\$OUT/(?P<report>[^"\s]+)" 2>&1; '
            r'net prc nice -n 10 timeout \d+ (?P<run>.+) >> "\$OUT/(?P=report)" 2>&1$'
        )
        seen = set()
        for name, text in scripts().items():
            for ln in text.replace(prcode.SHELL, "").splitlines():
                for tool in FETCHING:
                    if f"$M/tools/{tool}" in ln and "prc " in ln:
                        with self.subTest(script=name, tool=tool):
                            m = line.match(ln)
                            self.assertIsNotNone(m, ln)
                            self.assertEqual(m["fetch"], m["run"])
                            seen.add((name, tool))
        self.assertLessEqual(
            {(n, t) for n in ("checks", "checks follow-up") for t in FETCHING}, seen
        )
        # the tools prf runs, each with a --fetch
        named = set()
        for text in scripts().values():
            named |= set(re.findall(r"\bprf python -I -B \$M/tools/(\w+\.py)", text))
        self.assertEqual(named, set(FETCHING))
        for tool in sorted(named):
            p = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "tools" / tool), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertIn("--fetch", p.stdout, tool)

    def test_prf_runs_the_command_with_fetch_lazy_fetches_on_and_no_input(self):
        """Through `rk pr-fetch`, prcode.fetch_for: --fetch before the command's --, what it prints on stderr."""
        script = (
            prcode.SHELL
            + "\nexport GIT_NO_LAZY_FETCH=1\n"
            + 'echo hi | prf sh -c \'cat; echo "${GIT_NO_LAZY_FETCH-unset}" "$@"; exit 3\' sh a -- t\n'
        )
        p = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(
            (p.stdout, p.stderr, p.returncode), ("", "unset a --fetch -- t\n", 3)
        )

    def test_rk_pr_fetch_needs_a_command(self):
        p = subprocess.run(
            [str(ROOT / "rk"), "pr-fetch", "--"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("give the command after --", p.stderr)

    def test_rk_pr_tree_makes_trees_pr_code_reads_with_no_configuration(self):
        """Each folder the PR runners' group's (here a group this user is in), 2750; one holding something else exits 1
        and says why; no folder is a usage error."""
        import rkmon

        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "full").mkdir()
            (d / "full" / "a.py").write_text("x")
            err = io.StringIO()
            with mock.patch.object(L, "PR_GROUP", grp.getgrgid(os.getgid()).gr_name):
                with mock.patch.dict(os.environ, {"RKMON_HOME": str(d / "none")}):
                    with contextlib.redirect_stderr(err):
                        made = rkmon.main(
                            ["pr-tree", str(d / "wt_v1"), str(d / "wt_q1")]
                        )
                        refused = rkmon.main(
                            ["pr-tree", str(d / "wt_v2"), str(d / "full")]
                        )
                        with self.assertRaises(SystemExit) as usage:
                            rkmon.main(["pr-tree"])
            self.assertEqual((made, refused, usage.exception.code), (0, 1, 2))
            for t in ("wt_v1", "wt_q1", "wt_v2"):
                s = (d / t).stat()
                self.assertEqual(stat.S_IMODE(s.st_mode), 0o2750, t)
                self.assertEqual(s.st_gid, os.getgid(), t)
            self.assertIn(f"rk pr-tree: {d / 'full'}: not empty", err.getvalue())
            self.assertFalse((d / "none").exists())  # no configuration read, nor made

    def test_fetch_for_runs_the_checks_fetch_before_its_targets(self):
        code = (
            "import os, sys; print(sys.argv[1:]); print(os.environ.get('GIT_NO_LAZY_FETCH', 'unset')); "
            "print(sys.stdin.read() or 'no input'); sys.exit(3)"
        )
        err = io.StringIO()
        r, w = os.pipe()  # the caller's input, which the fetch never reads
        os.write(w, b"the caller's input\n")
        os.close(w)
        kept = os.dup(0)
        os.dup2(r, 0)
        try:
            with contextlib.redirect_stderr(err):
                rc = prcode.fetch_for(
                    [sys.executable, "-c", code, "a", "--", "t"],
                    {"GIT_NO_LAZY_FETCH": "1", "PATH": os.environ.get("PATH", "")},
                )
        finally:
            os.dup2(kept, 0)
            os.close(kept)
            os.close(r)
        self.assertEqual(rc, 3)
        self.assertEqual(
            err.getvalue().splitlines(),
            ["['a', '--fetch', '--', 't']", "unset", "no input"],
        )
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(prcode.fetch_for([sys.executable, "-c", "pass"]), 0)
            self.assertEqual(
                prcode.fetch_for([sys.executable, "-c", "import sys; sys.exit(2)"]), 2
            )
            self.assertEqual(
                prcode.fetch_for(
                    [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5
                ),
                124,
            )
            with tempfile.TemporaryDirectory() as d:
                none = str(pathlib.Path(d) / "none")
                self.assertEqual(prcode.fetch_for([none]), 127)
        said = err.getvalue().splitlines()
        self.assertEqual(said[0], "fetch: exit 2")
        self.assertIn("were not fetched in", said[1])
        self.assertTrue(said[2].startswith(f"fetch: {none}: "), said)

    def test_pr_code_writes_only_its_slot_and_the_monitor_its_own_folders(self):
        """Each script takes its slot before its first PR-code command, runs none of it from a folder of its own (a
        `cd` of the monitor's into a tree), and hands PR code no folder of the monitor's to write: PR code writes its
        slot's out ($SO), the monitor makes $OUT and copies the results into it (prr), and develop's caches are the
        monitor's alone."""
        s = scripts()
        for name, text in s.items():
            with self.subTest(script=name):
                # the functions' definitions run nothing
                body = text.replace(prcode.SHELL, "").replace(review.FLAVORS_FN, "")
                self.assertNotIn("prw ", body)
                self.assertLess(body.index("prslot"), body.index("prc "))
                self.assertNotRegex(body, r"\bcd [^;)]*&& *(\w+=\S* +)*prc ")
                self.assertEqual(
                    given_out(body), []
                )  # nothing PR code runs is given the monitor's out
        self.assertIn('rm -rf "$OUT"; mkdir -p "$OUT"; R="$OUT/checks.md"', s["checks"])
        self.assertIn(
            '[ -n "$DS" ] && mkdir -p $CP && GC=$CP/golden_develop_', s["checks"]
        )
        self.assertIn(
            '[ -n "$DS" ] && mkdir -p $CP && IC=$CP/installed_develop_', s["checks"]
        )
        self.assertIn(f"CP={review.DEVELOP_CACHE}", s["checks"])
        # develop's cached lines and outcomes are written by the monitor's steps, never by PR code
        for step in ("golden_vs_base.py report", "installed_tests.py cache-add"):
            self.assertIn(step, s["checks"])
            self.assertNotRegex(s["checks"], r"\bprc [^;]*" + re.escape(step))
        self.assertIn('rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/status"', s["sanity"])
        self.assertIn('mkdir -p "$J" "$TP"; rm -rf "$TD"; prtree "$TD"', s["drift"])

    def test_the_monitor_makes_and_removes_its_out_folder_whole(self):
        """The checks' and the sanity run's opening lines remove and make their out folder ($OUT) whole, a path with a
        space too: OUT is quoted where it is set, removed, made and written, and so is each folder made in it. Nothing
        beside it is touched."""
        for name, text in scripts().items():
            with self.subTest(script=name):
                self.assertEqual(re.findall(r"(?:rm -rf|mkdir -p) \$OUT\b", text), [])
        with tempfile.TemporaryDirectory() as d:
            near = pathlib.Path(
                d, "work"
            )  # what a removal of the path's first word would remove
            near.mkdir()
            (near / "kept").write_text("x")
            out = f"{d}/work dir/pr7/checks"
            for name, script in (
                (
                    "checks",
                    review.checks_script(
                        CFG, 7, "a" * 40, "first-round", PLAN, None, out=out
                    ),
                ),
                (
                    "sanity",
                    guard.sanity_script(CFG, "a" * 40, f"{d}/wt", out, ["gemm"], True),
                ),
            ):
                with self.subTest(script=name):
                    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
                    pathlib.Path(out, "old").write_text("x")
                    lines = script.split("\n")
                    opening = next(i for i, x in enumerate(lines) if "OUT/status" in x)
                    r = subprocess.run(
                        ["bash", "-c", "\n".join(lines[: opening + 1])],
                        cwd=d,  # a path split at its space would leave its folders here
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertEqual(os.listdir(out), ["status"])  # made afresh
                    self.assertEqual(sorted(os.listdir(d)), ["work", "work dir"])
                    self.assertTrue((near / "kept").exists())

    def test_each_test_run_is_told_where_its_results_are_copied(self):
        """The re-run commands a test run prints name its environment file in the folder the job copies its results
        to: each prc of pytest_vs_base.py is told that folder (RESULTS_VAR), the one its own `prr` copies to, and
        installed_tests.py's (inst) the one its `prr` copies the last run to. It is a name the commands print: PR code
        writes only its slot."""

        def results(pre: str):
            return dict(a.split("=", 1) for a in pre.split()).get(prcode.RESULTS_VAR)

        s = scripts()
        for name in ("checks", "checks follow-up", "drift"):
            with self.subTest(script=name):
                runs = re.findall(
                    r"((?:[A-Za-z_]\w*=\S*\s+)*)prc [^;]*?pytest_vs_base\.py \$SO/(\w+) ",
                    s[name],
                )
                self.assertTrue(runs)
                for pre, o in runs:
                    self.assertIn(f"prr {o} {results(pre)};", s[name])
        pre = re.search(
            r"inst\(\) \{ o=\$1; shift; ((?:[A-Za-z_]\w*=\S*\s+)*)prc ", s["checks"]
        ).group(1)
        self.assertEqual(
            set(re.findall(r"prr (?:installed|again) (\S+);", s["checks"])),
            {results(pre)},
        )

    def test_each_script_names_the_trees_its_git_reads(self):
        """Git in a slot takes as safe only the trees the job names (docker/rk_prcode.py): the checks name the merge,
        the head, develop's tree and the clone, a sanity run and a bisect their tree, before their first PR-code
        command."""
        for name, text in scripts().items():
            trees = {"sanity": "$WT", "bisect": "$WT", "drift": None}.get(
                name, '"$W:$D/wt$N:$BW:$R0"'
            )
            with self.subTest(script=name):
                # the functions' definitions run nothing
                body = text.replace(prcode.SHELL, "").replace(review.FLAVORS_FN, "")
                if trees is None:  # the PRs' trees are archives: no git there
                    self.assertNotIn(prcode.TREES_VAR, body)
                    continue
                line = f"export {prcode.TREES_VAR}={trees}"
                self.assertIn(line, body)
                self.assertLess(body.index(line), body.index("prc "))

    def test_with_trees_names_the_trees_after_the_ones_named(self):
        env = prcode.with_trees(
            {"A": "1", prcode.TREES_VAR: "/w/a"}, "/w/b", pathlib.Path("/w/a"), "", None
        )
        self.assertEqual(env, {"A": "1", prcode.TREES_VAR: "/w/a:/w/b"})
        self.assertEqual(prcode.with_trees({}, "/w/x"), {prcode.TREES_VAR: "/w/x"})

    def test_the_folder_check_catches_pr_code_given_the_monitors_out(self):
        self.assertEqual(
            given_out("prc python3 x.py --out $OUT/golden > $OUT/r.txt"),
            ["prc python3 x.py --out $OUT/golden "],
        )
        self.assertEqual(
            given_out('prc python3 x.py --out "$OUT/golden" > "$OUT/r.txt"'),
            ['prc python3 x.py --out "$OUT/golden" '],
        )
        self.assertEqual(
            given_out("prc python3 x.py --out $SO/golden > $OUT/r.txt 2>&1"), []
        )
        self.assertEqual(
            given_out(
                'prc x.py --base "$(cat $OUT/develop_sha 2>/dev/null)" --out $SO/t'
            ),
            [],
        )


class TestShell(unittest.TestCase):
    """The slot commands (prslot, prc, prr, prin, prfree, prtree), run by bash: on a host, where a slot is a folder of
    the session user's, and with stand-ins for rk's slot commands, sudo and the launcher.
    """

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        b = self.d / "bin"
        b.mkdir()
        # sudo: the slot's user, then the launcher as it is
        (b / "sudo").write_text(
            '#!/bin/sh\necho "sudo $*" >> "$LOG"\n[ "$1 $2 $3" = "-n -u rkpr1" ] || exit 9\nshift 3\nexec "$@"\n'
        )
        # the launcher: the folder it is given, the environment from stdin, then the command with it
        (self.d / "runner").write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "a = sys.argv[1:]\n"
            "assert a[0] == 'run', a\n"
            "a = a[1:]\n"
            "if a[0] == '-C':\n"
            "    os.chdir(a[1])\n"
            "    a = a[2:]\n"
            "assert a[0] == '--', a\n"
            "env = dict(e.split('=', 1) for e in sys.stdin.read().split('\\0') if '=' in e)\n"
            "open(os.environ['LOG'], 'a').write('env ROCKE_X=' + env.get('ROCKE_X', '-') + '\\n')\n"
            "os.execvpe(a[1], a[1:], env)\n"
        )
        # rk's slot commands: take prints the slot of rkpr1 (prcode.ASSIGNED), a folder of the test's
        self.pkg = self.d / "pkg"
        self.pkg.mkdir()
        (self.pkg / "worker").symlink_to(ROOT / "worker")
        slot = self.d / "slot"
        (self.pkg / "rk").write_text(
            '#!/bin/sh\necho "rk $*" >> "$LOG"\n'
            f'[ "$2" = take ] && mkdir -p {slot}/out {slot}/build {slot}/inputs && '
            f'echo "SLOT=rkpr1 SD={slot} SO={slot}/out SB={slot}/build SI={slot}/inputs"\nexit 0\n'
        )
        for f in (b / "sudo", self.d / "runner", self.pkg / "rk"):
            f.chmod(0o755)
        self.log = self.d / "log"

    def bash(self, body: str, runner: bool) -> subprocess.CompletedProcess:
        env = dict(
            os.environ,
            PATH=f"{self.d / 'bin'}:{os.environ['PATH']}",
            LOG=str(self.log),
            TMPDIR=str(self.d),
        )
        env.pop(prcode.RUNNER_VAR, None)
        if runner:
            env[prcode.RUNNER_VAR] = str(self.d / "runner")
        return subprocess.run(
            ["bash", "-c", prcode.shell(self.pkg) + "\n" + body],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    def test_on_a_host_a_slot_is_a_fresh_folder_of_the_session_users(self):
        """Without the launcher the command runs in the slot's home with the job's environment and RKMON_OUT naming
        the slot's out; what it leaves there is copied back, and freeing the slot removes its folder.
        """
        p = self.bash(
            'prslot || exit 9; echo "$SD"; ROCKE_X=1 FOO=2 prc sh -c \'echo "$ROCKE_X ${FOO:-none}"; pwd; '
            'echo r > "$RKMON_OUT/res"\'; '
            f'prr res {self.d}/got; prfree; [ -e "$SD" ] || echo freed',
            False,
        )
        sd, foo, cwd, freed = p.stdout.splitlines()
        self.assertTrue(sd.startswith(f"{self.d}/rkslot-"), sd)
        self.assertEqual(
            (foo, cwd, freed), ("1 none", f"{sd}/{L.SLOT_HOME}", "freed"), p.stderr
        )
        self.assertEqual((self.d / "got" / "res").read_text(), "r\n")
        self.assertFalse(self.log.exists())  # neither rk, sudo nor the launcher ran

    def test_pr_code_runs_only_in_a_slot_taken_here_and_a_subshell_takes_its_own(self):
        p = self.bash(
            "prc true; echo rc=$?; prr x /y; echo rc=$?; prin /etc/hostname; echo rc=$?",
            False,
        )
        self.assertEqual(p.stdout, "rc=3\nrc=3\nrc=3\n")
        self.assertIn("prc: no slot of a PR runner is taken here (prslot)", p.stderr)
        p = self.bash(
            'prslot; A=$SD; (prslot; [ "$SD" != "$A" ] && echo own; prfree); [ -d "$A" ] && echo kept',
            False,
        )
        self.assertEqual(p.stdout, "own\nkept\n", p.stderr)

    def test_with_the_launcher_it_runs_in_the_slot_rk_took_with_the_jobs_environment(
        self,
    ):
        p = self.bash(
            "prslot 7 || exit 9; export ROCKE_B=2 BAR=3; ROCKE_X=1 FOO=4 prc sh -c 'echo $ROCKE_X $ROCKE_B "
            "${BAR:-none} ${FOO:-none}; cat'; echo rc=$?; "
            f"prc -C {self.d} pwd; prc sh -c 'exit 3'; echo rc=$?",
            True,
        )
        # its stdin was the environment, read; the folder it was given is where it ran
        self.assertEqual(p.stdout, f"1 2 none none\nrc=0\n{self.d}\nrc=3\n", p.stderr)
        log = self.log.read_text().splitlines()
        took = re.fullmatch(r"rk pr-slot take --pid (\d+) --pr 7", log[0])
        self.assertTrue(took, log)
        self.assertIn(
            f"sudo -n -u rkpr1 {self.d / 'runner'} run -- sh -c echo $ROCKE_X $ROCKE_B ${{BAR:-none}} "
            "${FOO:-none}; cat",
            log,
        )
        self.assertIn(
            f"sudo -n -u rkpr1 {self.d / 'runner'} run -C {self.d} -- pwd", log
        )
        self.assertIn("env ROCKE_X=1", log)
        self.assertEqual(
            log[-1], f"rk pr-slot free --pid {took[1]}"
        )  # freed when the script ends

    def test_prtree_makes_folders_for_trees_pr_code_reads(self):
        """prtree is worker/perms.py tree: here, with no PR runners' group, plain folders no one else writes."""
        p = self.bash(f"prtree {self.d}/t1 {self.d}/t2 && echo ok", False)
        self.assertEqual(p.stdout, "ok\n", p.stderr)
        for t in ("t1", "t2"):
            mode = (self.d / t).stat().st_mode
            self.assertTrue(stat.S_ISDIR(mode))
            self.assertEqual(stat.S_IMODE(mode) & 0o022, 0)


class TestStops(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def test_a_jobs_kill_frees_its_slots_and_a_stop_asks_every_runner(self):
        with fake_slots.Slots(self.d).here() as fs:
            recs = slots.Records()
            me = recs.holder(os.getpid())
            for runner, job in (
                ("rkpr2", "/s/jobs/checks-1"),
                ("rkpr3", "/s/jobs/checks-2"),
            ):
                recs.write(
                    runner, {"state": "taken", **me, "job": job, "pr": "7", "since": 1}
                )
            kept = {
                "state": "kept",
                "pr": "7",
                "head": "a" * 40,
                "build": f"{fs.slot('rkpr4')}/{L.SLOT_BUILD}",
                "record": str(self.d / "shared_build"),
                "since": 1,
            }
            recs.write("rkpr4", kept)
            prcode.kill_job("/s/jobs/checks-1")
            self.assertEqual(fs.calls, [("rkpr2", "end")])
            self.assertEqual(sorted(fs.records()), ["rkpr3", "rkpr4"])
            self.assertEqual(
                fs.records()["rkpr4"], kept
            )  # kept for its review: it stays
            prcode.kill_job("/s/jobs/checks-1")
            self.assertEqual(len(fs.calls), 1)  # nothing of that job is left
            fs.stopped = {"rkpr3": 2}
            self.assertEqual(
                prcode.stop_all(now=True), ["PR_SLOT rkpr3: stopped 2 process(es)"]
            )
            prcode.stop_all()
            runners = sorted(u.name for u in L.PR_RUNNERS)
            self.assertEqual(
                sorted(r for r, a in fs.asked if a == ["kill", "--now"]), runners
            )
            self.assertEqual(sorted(r for r, a in fs.asked if a == ["kill"]), runners)
            self.assertEqual(
                sorted(fs.records()), ["rkpr3", "rkpr4"]
            )  # a stop leaves the records
        asked = len(fs.asked)
        self.assertEqual(
            prcode.kill_job("/x", environ={}), []
        )  # a host: nothing to ask
        self.assertEqual(prcode.stop_all(environ={}), [])
        self.assertEqual(len(fs.asked), asked)

    def test_a_pr_code_jobs_kill_asks_the_launcher(self):
        store = mock.Mock(dir=pathlib.Path("/s"))
        with mock.patch.object(jobs, "members", lambda st, j: []), mock.patch.object(
            prcode, "kill_job", return_value=["PR_SLOT rkpr2: stopped 2 process(es)"]
        ) as k:
            jobs.kill(store, {"id": "checks-pr7-1", "kind": "checks"})
            jobs.kill(store, {"id": "worker-pr7-1", "kind": "worker"})
        k.assert_called_once_with("/s/jobs/checks-pr7-1")
        self.assertIn(
            mock.call("job", "checks-pr7-1: PR_SLOT rkpr2: stopped 2 process(es)"),
            store.journal.call_args_list,
        )

    def test_rk_stop_and_rk_kill_ask_it_for_every_process(self):
        """rk stop asks the launcher to stop the PR runners' processes, rk kill to kill them at once, and each prints
        what the launcher said."""
        from rkmon_lib import config, control
        from rkmon_lib.store import Store, empty_state

        asked = []

        def stop_all(now=False, environ=None, run=None):
            asked.append(now)
            return [f"PR_SLOT rkpr1: stopped (now={now})"]

        with tempfile.TemporaryDirectory() as d:
            c = config.load_defaults()
            c.pop("_doc", None)
            c.update(
                repo="acme/widgets",
                rocke_path="a/b/rocke",
                operator={"github_login": "op-login"},
                scope={"labels": ["x"]},
            )
            c["paths"] = dict(c["paths"], working_dir=f"{d}/work")
            cfg = config.normalize(c)
            store = Store(pathlib.Path(d) / "state")
            store.save(empty_state())
            out = []
            with mock.patch.object(prcode, "stop_all", stop_all):
                self.assertEqual(
                    control.stop(cfg, store, write_wait=1, out=out.append), 0
                )
                self.assertEqual(asked, [False])
                self.assertEqual(
                    control.kill_all(cfg, store, wait=1, out=out.append), 0
                )
        self.assertEqual(asked, [False, True])
        self.assertIn("PR_SLOT rkpr1: stopped (now=False)", out)
        self.assertIn("PR_SLOT rkpr1: stopped (now=True)", out)


class TestReviewEnd(unittest.TestCase):
    """The slots kept and taken for a review are freed at the next advance after it ends (review.end_slots)."""

    def setUp(self):
        from rkmon_lib.store import empty_state

        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.st = empty_state()

    def keep(self, fs, runner: str, pr: int, head: str) -> pathlib.Path:
        """`runner` kept for PR `pr`'s review of `head`, with the record that names its build as the shared one."""
        build = f"{fs.slot(runner)}/{L.SLOT_BUILD}"
        record = self.d / f"shared_build_{runner}"
        record.write_text(f"{build}\n/work/wt{pr}m\n{head}\n")
        slots.Records().write(
            runner,
            {
                "state": "kept",
                "pr": str(pr),
                "head": head,
                "build": build,
                "record": str(record),
                "since": 1,
            },
        )
        return record

    def test_the_slots_of_a_review_that_ended_are_freed(self):
        a, b, c = "a" * 40, "b" * 40, "c" * 40
        self.st["review_pipes"] = [
            {"pr": 7, "kind": "follow-up", "stage": "worker", "head": a}
        ]
        self.st["review_ready"] = [{"pr": 9, "kind": "first-round", "head": c}]
        with fake_slots.Slots(self.d).here() as fs:
            r7, r9 = self.keep(fs, "rkpr1", 7, a), self.keep(fs, "rkpr2", 9, c)
            slots.Records().write(
                "rkpr3",
                {
                    "state": "taken",
                    "name": "pr9-build",
                    "pr": "9",
                    "head": c,
                    "since": 1,
                },
            )
            self.assertEqual(review.end_slots(self.st), [])
            self.assertEqual(fs.calls, [])  # both reviews are in flight
            self.st["review_ready"] = []  # PR 9's review is posted
            self.assertEqual(review.end_slots(self.st), [])
            self.assertEqual(fs.calls, [("rkpr2", "end"), ("rkpr3", "end")])
            self.assertEqual(sorted(fs.records()), ["rkpr1"])
            self.assertFalse(r9.exists())  # the shared build's record goes with it
            self.st["review_pipes"][0]["head"] = b  # a new head supersedes PR 7's
            review.end_slots(self.st)
            self.assertEqual(fs.calls[2:], [("rkpr1", "end")])
            self.assertEqual(fs.records(), {})
            self.assertFalse(r7.exists())
        self.assertEqual(review.end_slots(self.st, environ={}), [])  # a host

    def test_each_advance_from_outside_frees_them(self):
        from rkmon_lib.store import Store

        cfg = dict(CFG, paths={"working_dir": str(self.d / "work")})
        said = ["PR_SLOT rkpr2: stopped 1 process(es)"]
        with mock.patch.object(review, "end_slots", return_value=said) as end:
            lines = review.advance(cfg, self.st, Store(self.d / "state"), [])
        end.assert_called_once_with(self.st)
        self.assertIn(said[0], lines)


class TestStatus(unittest.TestCase):
    def test_where_pr_code_runs_here(self):
        self.assertIn("session user", prcode.status({})[0])
        with tempfile.TemporaryDirectory() as d, fake_slots.Slots(d).here() as fs:
            recs = slots.Records()
            build = f"{fs.slot('rkpr1')}/{L.SLOT_BUILD}"
            recs.write(
                "rkpr1",
                {
                    "state": "kept",
                    "pr": "7",
                    "head": "a" * 40,
                    "build": build,
                    "record": "/r",
                    "since": 0,
                },
            )
            recs.write(
                "rkpr2",
                {
                    "state": "taken",
                    "name": "pr8-x",
                    "pr": "8",
                    "head": "b" * 40,
                    "since": 0,
                },
            )
            init, boot = slots._start(1, "/proc"), slots._boot("/proc")
            recs.write(
                "rkpr3",
                {
                    "state": "broken",
                    "init": init,
                    "boot": boot,
                    "why": "its mode changed",
                    "since": 0,
                },
            )
            me = recs.holder(os.getpid())
            recs.write(
                "rkpr4",
                {
                    "state": "taken",
                    **me,
                    "job": "/s/jobs/checks-1",
                    "pr": "9",
                    "since": 0,
                },
            )
            recs.write("rkpr5", {"state": "freeing", **me, "since": 0})
            # a process that is gone: its slot is free
            recs.write(
                "rkpr6",
                {
                    "state": "taken",
                    "pid": 1,
                    "start": "-",
                    "boot": boot,
                    "job": "/j",
                    "pr": "",
                    "since": 0,
                },
            )
            lines = prcode.status()
        since = "since 1970-01-01T00:00:00Z"
        self.assertEqual(len(lines), 1 + len(L.PR_RUNNERS))
        self.assertIn(f"the launcher {fake_slots.LAUNCHER}", lines[0])
        self.assertEqual(
            lines[1:8],
            [
                f"  rkpr1: kept for #7 at aaaaaaaaaa (its build {build}, read by the other runners) {since}",
                f"  rkpr2: named work pr8-x for the review at bbbbbbbbbb {since}",
                f"  rkpr3: broken until the container's next start (its mode changed) {since}",
                f"  rkpr4: taken by process {os.getpid()} #9 (job /s/jobs/checks-1) {since}",
                f"  rkpr5: being freed by process {os.getpid()} {since}",
                "  rkpr6: free",
                "  rkpr7: free",
            ],
        )

    def test_the_doctor_names_each_broken_slot(self):
        from rkmon_lib import doctor

        self.assertEqual(prcode.doctor_checks({}), [])  # a host: no launcher
        n = len(L.PR_RUNNERS)
        with tempfile.TemporaryDirectory() as d, fake_slots.Slots(d).here():
            recs = slots.Records()
            (row,) = prcode.doctor_checks()
            self.assertEqual(
                (row.status, row.found), ("PASS", f"{n} slots, none broken")
            )
            why = slots.stuck("rkpr2", "end", -9, 3)
            recs.write("rkpr2", slots.broken(recs, why))
            recs.write(
                "rkpr3", dict(slots.broken(recs, "x"), init="-")
            )  # before this start
            env = {slots.RUNNER_VAR: fake_slots.LAUNCHER, "RKMON_WRAPPER": "rkmon"}
            (row,) = prcode.doctor_checks(env)
            self.assertEqual(row.status, "WARN")
            self.assertTrue(
                row.found.startswith(
                    f"1 of {n} broken: rkpr2: broken until the container's next start ({why}) since "
                ),
                row.found,
            )
            self.assertIn(slots.RESTART, row.found)
            self.assertEqual(
                str(row.fix),
                "rkmon stop, then rkmon start: the container's next start makes them again",
            )
            with mock.patch.object(doctor, "deployment_image", return_value=False):
                self.assertIn(row, doctor.container_checks({}))
            for r in recs.runners:
                recs.write(r, slots.broken(recs, "x"))
            (row,) = prcode.doctor_checks()
            self.assertEqual(row.status, "FAIL")
            self.assertIn("it blocks every job that runs PR code", row.found)


class TestSlotsInPython(unittest.TestCase):
    """prcode.Slot, run and call: in the container (the stand-in) a runner's slot, its record and the launcher's begin,
    run and end; on a host a folder of the session user's."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def test_a_slot_is_begun_used_and_freed(self):
        with fake_slots.Slots(self.d).here() as fs:
            env = dict(os.environ, LANG="C.UTF-8", GH_TOKEN="not-for-pr-code")
            env[jobs.JOB_ENV] = "/s/jobs/checks-1"
            with prcode.Slot(pr=7, environ=env) as s:
                self.assertEqual((s.runner, s.dir), ("rkpr1", str(fs.slot("rkpr1"))))
                rec = fs.records()["rkpr1"]
                self.assertEqual(
                    (rec["state"], rec["pid"], rec["job"], rec["pr"]),
                    ("taken", os.getpid(), "/s/jobs/checks-1", "7"),
                )
                self.assertEqual(sorted(os.listdir(s.dir)), sorted(L.SLOT_FOLDERS))
                p = s.run(["echo", "hi"], env=env)
                self.assertEqual((p.returncode, p.stdout), (0, "AS-PR-RUNNER\n"))
                ran = fs.ran[-1]
                self.assertEqual(
                    (ran["runner"], ran["cmd"], ran["cwd"]),
                    ("rkpr1", ["echo", "hi"], s.home),
                )
                self.assertEqual(ran["env"].get("LANG"), "C.UTF-8")
                self.assertNotIn(
                    "GH_TOKEN", ran["env"]
                )  # only what worker/pr_env.py lists
                s.run(["make"], cwd=L.SLOT_BUILD)
                self.assertEqual(fs.ran[-1]["cwd"], os.path.join(s.dir, L.SLOT_BUILD))
                # what the monitor hands the job, and what it reads back of the job's
                given = self.d / "list.txt"
                given.write_text("a\n")
                self.assertEqual(s.give(given), f"{fs.inputs}/rkpr1/list.txt")
                pathlib.Path(s.out, "r.json").write_text('{"ok": 1}\n')
                self.assertEqual(s.read_json("r.json"), {"ok": 1})
                os.symlink(given, os.path.join(s.out, "l.json"))
                with self.assertRaises(OSError):
                    s.read_json("l.json")  # a link is never followed
            self.assertEqual(fs.records(), {})
            self.assertEqual(
                fs.calls,
                [
                    ("rkpr1", "begin"),
                    ("rkpr1", "run"),
                    ("rkpr1", "run"),
                    ("rkpr1", "end"),
                ],
            )
            self.assertEqual(os.listdir(fs.slot("rkpr1")), [])
            self.assertEqual(
                os.listdir(f"{fs.inputs}/rkpr1"), []
            )  # what it was handed goes with it

    def test_call_copies_the_jobs_results_and_one_refused_fails_it(self):
        with fake_slots.Slots(self.d).here() as fs:
            fs.on_run = lambda slot, cmd: (slot / L.SLOT_OUT / "report.txt").write_text(
                "r\n"
            )
            dest = self.d / "kept"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(
                    prcode.call(lambda s: ["tool", "--out", s.out], out=dest), 0
                )
            self.assertEqual(
                out.getvalue(), "AS-PR-RUNNER\n"
            )  # the command's output is the caller's
            self.assertEqual(
                fs.ran[-1]["cmd"], ["tool", "--out", f"{fs.slot('rkpr1')}/{L.SLOT_OUT}"]
            )
            self.assertEqual((dest / "report.txt").read_text(), "r\n")
            fs.on_run = lambda slot, cmd: os.symlink(
                dest / "report.txt", slot / L.SLOT_OUT / "report.txt"
            )
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                err
            ):
                self.assertEqual(prcode.call(["tool"], out=self.d / "kept2"), 1)
            self.assertIn("pr_results: refused", err.getvalue())
            self.assertFalse((self.d / "kept2" / "report.txt").exists())
            fs.rc = 5
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(
                    prcode.call(["tool"], out=self.d / "kept3"), 5
                )  # its own exit stands
            self.assertEqual(fs.records(), {})

    def test_with_no_slot_to_take_run_and_call_say_why(self):
        with fake_slots.Slots(self.d).here() as fs:
            recs = slots.Records()
            broken = {
                "state": "broken",
                "init": slots._start(1, "/proc"),
                "boot": slots._boot("/proc"),
                "why": "x",
                "since": 0,
            }
            for u in L.PR_RUNNERS:
                recs.write(u.name, broken)
            p = prcode.run(["true"])
            self.assertEqual(p.returncode, slots.REFUSED)
            self.assertIn("no slot can be taken", p.stderr)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(prcode.call(["true"]), slots.REFUSED)
            self.assertIn("no slot can be taken", err.getvalue())
            self.assertEqual(fs.calls, [])

    def test_a_sigterm_to_its_holder_frees_it(self):
        code = (
            "import sys, time\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from rkmon_lib import prcode\n"
            f"with prcode.Slot(environ={{}}, tmp={str(self.d)!r}) as s:\n"
            "    print(s.dir, flush=True)\n"
            "    time.sleep(60)\n"
        )
        p = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", code], stdout=subprocess.PIPE, text=True
        )
        try:
            d = p.stdout.readline().strip()
            self.assertTrue(os.path.isdir(os.path.join(d, L.SLOT_HOME)), d)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(timeout=60), 128 + signal.SIGTERM)
        finally:
            p.kill()
            p.wait()
            p.stdout.close()
        self.assertFalse(os.path.exists(d))

    def test_named_work_on_a_host_keeps_its_folder_until_it_ends(self):
        host = {"environ": {}, "tmp": str(self.d)}
        with prcode.Slot(name="pr7-a", end=False, **host) as s:
            d = s.dir
            pathlib.Path(s.home, "f").write_text("x")
        self.assertEqual(d, str(self.d / "rkslot-pr7-a"))
        with prcode.Slot(name="pr7-a", end=False, **host) as s:
            self.assertEqual(s.dir, d)
            self.assertTrue(
                os.path.isfile(os.path.join(s.home, "f"))
            )  # its files stay for its next command
        self.assertEqual(prcode.end_work("pr7-a", **host), [])
        self.assertFalse(os.path.exists(d))
        with prcode.Slot(name="pr7-b", **host) as s:  # its last command ends it
            d = s.dir
        self.assertFalse(os.path.exists(d))
        with prcode.Slot(**host) as s:  # a slot of its own
            d = s.dir
            self.assertTrue(os.path.basename(d).startswith("rkslot-"), d)
        self.assertFalse(os.path.exists(d))

    def test_a_hosts_folder_for_named_work_that_is_not_its_own_is_refused_and_never_followed(
        self,
    ):
        target = self.d / "elsewhere"
        target.mkdir()
        (self.d / "rkslot-pr7-a").symlink_to(target)
        (self.d / "rkslot-pr7-b").write_text("a file")
        for name in ("pr7-a", "pr7-b"):
            with self.subTest(name=name):
                with self.assertRaises(prcode.NoSlot) as cm:
                    with prcode.Slot(name=name, environ={}, tmp=str(self.d)):
                        self.fail("taken")
                self.assertIn("is not this user's folder", cm.exception.lines[0])
                prcode.end_work(name, environ={}, tmp=str(self.d))
        self.assertEqual(
            list(target.iterdir()), []
        )  # nothing was made through the link
        self.assertTrue((self.d / "rkslot-pr7-a").is_symlink())  # nor removed
        self.assertEqual((self.d / "rkslot-pr7-b").read_text(), "a file")

    def test_named_work_in_a_slot_gets_it_back_until_it_ends(self):
        with fake_slots.Slots(self.d).here() as fs:
            owner = slots.Records().holder(os.getpid())
            work = {"name": "pr7-a", "pr": 7, "end": False, "owner": owner}
            with prcode.Slot(**work) as s:
                runner = s.runner
                pathlib.Path(s.home, "f").write_text("x")
            rec = fs.records()[runner]
            self.assertEqual(
                (rec["state"], rec["name"], rec["pr"], rec["owner"]),
                ("taken", "pr7-a", "7", owner),
            )
            with prcode.Slot() as other:  # another job takes another slot
                self.assertNotEqual(other.runner, runner)
            with prcode.Slot(**work) as s:
                self.assertEqual(s.runner, runner)
                self.assertTrue(
                    os.path.isfile(os.path.join(s.home, "f"))
                )  # its files stay for its next command
            self.assertEqual(
                [c for c in fs.calls if c[0] == runner], [(runner, "begin")]
            )
            prcode.end_work("pr7-a")
            self.assertNotIn(runner, fs.records())
            self.assertEqual(fs.calls[-1], (runner, "end"))
            with prcode.Slot(
                name="pr7-b", pr=7, owner=owner
            ) as s:  # its last command ends it
                second = s.runner
            self.assertNotIn(second, fs.records())
            self.assertEqual(fs.calls[-1], (second, "end"))

    def test_a_job_scripts_slot_as_shell_assignments(self):
        with fake_slots.Slots(self.d).here() as fs:
            lines = prcode.assignments("rkpr3")
            self.assertEqual(
                [ln.split("=", 1)[0] for ln in lines], list(prcode.ASSIGNED)
            )
            top = fs.slot("rkpr3")
            got = subprocess.run(
                ["bash", "-c", "\n".join(lines) + '\necho "$SLOT|$SD|$SO|$SB|$SI"'],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        self.assertEqual(
            got,
            f"rkpr3|{top}|{top}/{L.SLOT_OUT}|{top}/{L.SLOT_BUILD}|{fs.inputs}/rkpr3\n",
        )


def config_home(d: pathlib.Path, w: pathlib.Path) -> pathlib.Path:
    """A monitor's home in `d` (d/home) for the package's rk on a host: the default configuration, its working folder
    `w`. The tools' tests run the re-run commands through it too (tests/test_pytest_vs_base.py,
    tests/test_installed_tests.py)."""
    from rkmon_lib import config

    cfg = config.load_defaults()
    cfg.pop("_doc", None)
    cfg.update(
        deployment=dict(cfg["deployment"], phase="review"),
        repo="o/r",
        rocke_path="rk",
        operator={"github_login": "someone"},
        scope={"labels": ["x"], "path_labels": []},
    )
    cfg["paths"]["working_dir"] = str(w)
    home = d / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps(cfg))
    return home


class TestEveryCaller(unittest.TestCase):
    """Each Python caller that runs PR code. The worker tools' command line runs a stand-in for rk, which logs what it
    was asked (its arguments, FOO of the environment handed on its stdin, its folder) and runs nothing; the package's
    own callers take a slot of the stand-in launcher's (tests/fakes/fake_slots.py)."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.log = self.d / "log"
        self.rk = self.d / "rk"
        self.rk.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "env = dict(e.split('=', 1) for e in sys.stdin.read().split('\\0') if '=' in e)\n"
            f"with open({str(self.log)!r}, 'a') as f:\n"
            "    f.write(json.dumps({'argv': sys.argv[1:], 'FOO': env.get('FOO', '-'), 'cwd': os.getcwd()}) + '\\n')\n"
            "print('AS-PR-RUNNER')\n"
        )
        self.rk.chmod(0o755)
        values = {"group": L.PR_GROUP, "home": L.SLOT_HOME, "rk": str(self.rk)}
        for p in (
            mock.patch.dict(
                os.environ, {prcode.RUNNER_VAR: fake_slots.LAUNCHER, "FOO": "1"}
            ),
            mock.patch.object(pr_user, "values", lambda: dict(values)),
        ):
            p.start()
            self.addCleanup(p.stop)

    def launched(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(ln) for ln in self.log.read_text().splitlines()]

    def tree(self) -> pathlib.Path:
        w = self.d / "w"
        (w / "wt7" / "rk" / "tests").mkdir(parents=True)
        (w / "wt7" / "rk" / "tests" / "test_x.py").write_text("def test_x(): pass\n")
        (w / "probes" / "pr7").mkdir(parents=True)
        (w / "probes" / "pr7" / "p1.sh").write_text("exit 0\n")
        return w

    def reviewer_tools(self):
        import review_sandbox
        import reviewer_tools

        self.addCleanup(setattr, review_sandbox, "_KIND", review_sandbox._KIND)
        return reviewer_tools, review_sandbox

    def rk_in(self, home: pathlib.Path):
        """rkmon.main in this process, with the configuration in `home`: (exit, stdout, stderr)."""
        import rkmon

        def rk(*argv) -> tuple[int, str, str]:
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = rkmon.main(list(argv))
            return code, out.getvalue(), err.getvalue()

        return rk

    def test_the_first_pass_tools_run_pr_code_in_a_slot_of_its_own(self):
        rt, sandbox = self.reviewer_tools()
        w = self.tree()
        env = {
            "RKMON_WORKING_DIR": str(w),
            "RKMON_ROCKE_PATH": "rk",
            "RKMON_VENV": str(w / "venv"),
            "RKMON_TOOLKIT": str(ROOT),
            "RKMON_BUILD": str(w / "build"),
            "RKMON_TMP": str(w / "tmp"),
            "RKMON_PROBES": str(w / "probes"),
        }
        t = rt.Tree(w / "ws", 7, get=lambda k, required=True: env.get(k, ""))
        text = rt.run_tool(t, "run_pytest", {"paths": "rk/tests/test_x.py"})[0]
        self.assertEqual(text, "AS-PR-RUNNER")
        run = self.launched()[-1]
        self.assertEqual(
            run["argv"][:6],
            ["pr-run", "--env", "/dev/stdin", "-C", f"{w}/wt7/rk", "--"],
        )
        self.assertIn("-B -m pytest", " ".join(run["argv"]))
        self.assertIn(f"{w}/wt7/rk/tests/test_x.py", run["argv"])
        self.assertEqual(run["FOO"], "1")  # the tool's environment, on rk's stdin
        sandbox._KIND = ""
        self.assertEqual(
            rt.run_tool(t, "run_probe", {"name": "p1", "tree": "head"})[0],
            "AS-PR-RUNNER",
        )
        run = self.launched()[-1]
        # rk runs in the review's folder; the probe in the slot's home, which its runner writes
        self.assertEqual(run["argv"][3:6], ["-C", L.SLOT_HOME, "--"])
        self.assertIn(f"bash {w}/probes/pr7/p1.sh {w}/wt7", " ".join(run["argv"]))
        self.assertEqual(run["cwd"], str(w / "ws" / "runs"))
        sandbox._KIND = "bwrap"
        (w / "ws" / "mutants").mkdir(parents=True)
        (w / "ws" / "mutants" / "rkmut_m1.py").write_text("")
        argv = rt.build(t, "run_mutant", {"name": "m1", "tests": "rk/tests/test_x.py"})[
            0
        ]
        self.assertEqual(
            argv[:7],
            [str(self.rk), "pr-run", "--env", "/dev/stdin", "-C", f"{w}/wt7/rk", "--"],
        )
        self.assertIn("bwrap", argv[7:])  # the sandbox, in the slot
        before = len(self.launched())
        rt.run_tool(t, "grep_code", {"pattern": "def"})
        self.assertEqual(
            len(self.launched()), before
        )  # grep runs no PR code: the session's
        with mock.patch.dict(os.environ, {prcode.RUNNER_VAR: ""}):
            argv = rt.build(t, "run_pytest", {"paths": "rk/tests/test_x.py"})[0]
        # a host: through worker/pr_env.py, as the session user
        self.assertEqual(argv[1:3], ["-I", str(ROOT / "worker" / "pr_env.py")])

    def test_the_drafters_saved_probe_runs_in_a_slot_of_its_own(self):
        import review_tools

        w = self.tree()
        t = review_tools.Tree(w / "ws", 7, w, "rk")
        out = review_tools.run_tool(t, "probe", {"name": "p1", "tree": "head"})[0]
        self.assertEqual(out, "AS-PR-RUNNER")
        run = self.launched()[-1]
        self.assertEqual(run["argv"][3:5], ["-C", f"{w}/wt7"])
        self.assertIn(f"run_probes.py {w}/probes/pr7 {w}/wt7 p1", " ".join(run["argv"]))
        self.assertEqual(run["FOO"], "1")
        before = len(self.launched())
        review_tools.run_tool(t, "grep_code", {"pattern": "def"})
        self.assertEqual(len(self.launched()), before)

    def test_rk_probe_check_runs_the_probes_in_a_slot(self):
        from rkmon_lib import probes

        w = self.tree()
        with fake_slots.Slots(self.d).here() as fs:
            probes.run_on(w / "probes" / "pr7", w / "wt7", "", dict(os.environ))
            self.assertEqual(fs.records(), {})
        self.assertEqual(
            fs.calls, [("rkpr1", "begin"), ("rkpr1", "run"), ("rkpr1", "end")]
        )
        ran = fs.ran[0]
        self.assertIn(f"run_probes.py {w}/probes/pr7 {w}/wt7", " ".join(ran["cmd"]))
        self.assertEqual(ran["env"].get("PROBE_DETAIL"), "1")
        self.assertEqual(ran["env"].get(prcode.TREES_VAR), str(w / "wt7"))
        self.assertNotIn("FOO", ran["env"])  # only what worker/pr_env.py lists

    def test_run_suite_runs_the_tests_in_a_slot_of_its_own(self):
        import run_suite

        w = self.tree()
        out = self.d / "out"
        self.addCleanup(os.chdir, os.getcwd())
        env = {
            "RKMON_WORKING_DIR": str(w),
            "RKMON_REPO_DIR": str(w / "repo"),
            "RKMON_REPO": "o/r",
            "RKMON_ROCKE_PATH": "rk",
            "RKMON_VENV": str(w / "venv"),
        }
        # The script prints the output's last lines on stdout's buffer.
        printed = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(printed):
            self.assertEqual(
                run_suite.main(["run_suite.py", "wt7", "plat", str(out)]), 0
            )
        self.assertIn("AS-PR-RUNNER", (out / "out.txt").read_text())
        run = self.launched()[-1]
        self.assertEqual(
            run["argv"][:6], ["pr-run", "--env", "/dev/stdin", "-C", L.SLOT_HOME, "--"]
        )
        self.assertIn("-I -B -m pytest", " ".join(run["argv"]))
        # rk runs in the output folder, the script's own
        self.assertEqual((run["FOO"], run["cwd"]), ("1", str(out)))

    def test_rk_runs_pr_code_in_a_slot_of_its_own(self):
        w = self.tree()
        rk = self.rk_in(config_home(self.d, w))
        here = str(w / "wt7" / "rk")
        with fake_slots.Slots(self.d).here() as fs:
            with mock.patch.dict(os.environ, {"RKMON_CALLER_DIR": here}):
                self.assertEqual(
                    rk("pr-run", "--", "echo", "hi")[:2], (0, "AS-PR-RUNNER\n")
                )
            self.assertEqual(
                (fs.ran[-1]["cmd"], fs.ran[-1]["cwd"]), (["echo", "hi"], here)
            )
            self.assertEqual(fs.records(), {})
            fetched = []

            def fetch_for(argv, env=None, **kw):
                fetched.append((list(argv), len(fs.calls)))
                return 0

            for cmd, tool in (
                ("mutants", "mutation_sweep.py"),
                ("import-probe", "import_probe.py"),
                ("ctest-labels", "ctest_labels.py"),
            ):
                with self.subTest(cmd=cmd):
                    before = len(fs.calls)
                    with mock.patch.object(prcode, "fetch_for", fetch_for):
                        self.assertEqual(rk(cmd, "7", "--head", "a" * 40)[0], 0)
                    ran = fs.ran[-1]["cmd"]
                    self.assertIn(f"tools/{tool} --repo", " ".join(ran))
                    # its --fetch first, as the monitor: the same command, before the slot is taken
                    self.assertEqual(fetched[-1], (ran[:-2], before))
                    # the trees its git reads: the clone, and the sweep's head tree
                    trees = fs.ran[-1]["env"][prcode.TREES_VAR].split(":")
                    self.assertEqual(trees[0], ran[ran.index("--repo") + 1])
                    self.assertEqual(
                        trees[1:],
                        [ran[ran.index("--tree") + 1]] if "--tree" in ran else [],
                    )
                    self.assertEqual(cmd == "mutants", "--tree" in ran)
                    # its results: the slot's out, copied into the monitor's folder as data
                    self.assertEqual(
                        ran[-2:], ["--out", f"{fs.slot('rkpr1')}/{L.SLOT_OUT}"]
                    )
            asked = len(fs.calls)
            with mock.patch.object(prcode, "fetch_for", fetch_for):
                rk("moved-refs", "7", "--head", "a" * 40)  # git only: no PR code
            self.assertEqual(len(fs.calls), asked)
            self.assertEqual(len(fetched), 3)
            self.assertEqual(fs.records(), {})

    def test_rk_werror_runs_the_tidy_check_in_a_slot_on_the_reviews_shared_build(self):
        w = self.tree()
        rk = self.rk_in(config_home(self.d, w))
        fetched = []

        def fetch_for(argv, env=None, **kw):
            fetched.append(list(argv))
            return 0

        said = "TIDY pass 0 errors in 1 TU (no feature macro)\n"
        with fake_slots.Slots(self.d, printed=said.encode()).here() as fs:
            record = w / "workers" / "pr7" / "checks" / review.SHARED_BUILD
            record.parent.mkdir(parents=True)
            kept = fs.slot("rkpr3") / L.SLOT_BUILD / "wt7"
            kept.mkdir(parents=True)
            record.write_text(f"{kept}\n{w / 'wt7'}\n{'c' * 40}\n")
            slots.Records().write(
                "rkpr3",
                {"state": "kept", "pr": "7", "head": "a" * 40, "build": str(kept)},
            )
            with mock.patch.object(prcode, "fetch_for", fetch_for):
                self.assertEqual(rk("werror", "7", "--base", "b" * 40)[:2], (0, said))
            ran = fs.ran[-1]
            cmd = ran["cmd"]
            self.assertIn("tools/tidy_check.py", " ".join(cmd))
            self.assertEqual(
                ran["runner"], "rkpr1"
            )  # a slot of its own: no job runs in the kept one
            self.assertEqual(cmd[cmd.index("--build") + 1], f"{kept}/core")
            # its --fetch first, as the monitor; the trees its git reads: the clone and the tree
            self.assertEqual(fetched, [cmd])
            self.assertEqual(
                ran["env"][prcode.TREES_VAR],
                f"{cmd[cmd.index('--repo') + 1]}:{w / 'wt7'}",
            )
            # the slot taken back for another job: the build folder's, and a line that says why
            slots.Records().write("rkpr3", {"state": "taken", "pid": 41, "pr": "9"})
            with mock.patch.object(prcode, "fetch_for", fetch_for):
                code, out, _ = rk("werror", "7", "--base", "b" * 40)
            self.assertIn(
                f"the review's shared build {kept} is gone (its slot was freed): give --build",
                out,
            )
            cmd = fs.ran[-1]["cmd"]
            self.assertEqual(cmd[cmd.index("--build") + 1], f"{w}/build/wt7/core")

    def test_rk_pr_run_named_work_its_environment_and_its_results(self):
        w = self.tree()
        rk = self.rk_in(config_home(self.d, w))
        with fake_slots.Slots(self.d).here() as fs:
            self.assertEqual(rk("pr-run")[0], 2)  # no command
            self.assertEqual(
                rk("pr-run", "--end", "--", "true")[0], 2
            )  # --end ends named work
            self.assertEqual(
                rk("pr-run", "--slot", "x7", "--", "true")[0], 2
            )  # no work name
            self.assertEqual(
                rk("pr-run", "--env", str(self.d / "none"), "--", "true")[0], 2
            )
            self.assertEqual(fs.calls, [])
            self.assertEqual(rk("pr-run", "--slot", "pr7-a", "--", "make")[0], 0)
            self.assertEqual(
                rk("pr-run", "--slot", "pr7-a", "--", "make", "test")[0], 0
            )
            self.assertEqual(
                fs.calls, [("rkpr1", "begin"), ("rkpr1", "run"), ("rkpr1", "run")]
            )
            self.assertEqual(fs.records()["rkpr1"]["name"], "pr7-a")
            self.assertEqual(rk("pr-run", "--slot", "pr7-a", "--end")[0], 0)
            self.assertEqual((fs.calls[-1], fs.records()), (("rkpr1", "end"), {}))
            env = self.d / "env"
            env.write_bytes(b"LANG=C.UTF-8\0GH_TOKEN=not-for-pr-code\0")
            self.assertEqual(rk("pr-run", "--env", str(env), "--", "true")[0], 0)
            self.assertEqual(fs.ran[-1]["env"], {"LANG": "C.UTF-8", **pr_env.FIXED})
            fs.on_run = lambda slot, cmd: os.symlink(env, slot / L.SLOT_OUT / "x")
            code, _out, err = rk("pr-run", "--out", str(self.d / "res"), "--", "true")
            self.assertEqual(code, 1)
            self.assertIn("pr_results: refused", err)
            self.assertEqual(fs.records(), {})

    def test_rk_pr_slot_takes_keeps_and_frees_a_job_scripts_slot(self):
        w = self.tree()
        rk = self.rk_in(config_home(self.d, w))
        pid = str(os.getpid())
        with fake_slots.Slots(self.d).here() as fs:
            code, out, _err = rk("pr-slot", "take", "--pid", pid, "--pr", "7")
            self.assertEqual(code, 0)
            got = subprocess.run(
                ["bash", "-c", out + 'echo "$SLOT $SO"'],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            self.assertEqual(got, f"rkpr1 {fs.slot('rkpr1')}/{L.SLOT_OUT}\n")
            rec = fs.records()["rkpr1"]
            self.assertEqual(
                (rec["state"], rec["pid"], rec["pr"]), ("taken", os.getpid(), "7")
            )
            self.assertEqual(rk("pr-slot", "free", "--pid", pid)[0], 0)
            self.assertEqual(
                (fs.calls, fs.records()), ([("rkpr1", "begin"), ("rkpr1", "end")], {})
            )
            rk("pr-slot", "take", "--pid", pid, "--pr", "7")
            build = f"{fs.slot('rkpr1')}/{L.SLOT_BUILD}"
            keep = ("--pid", pid, "--pr", "7", "--head", "a" * 40, "--build", build)
            code, out, _err = rk(
                "pr-slot", "keep", *keep, "--record", str(self.d / "shared_build")
            )
            self.assertEqual(code, 0, out)
            self.assertEqual(fs.records()["rkpr1"]["state"], "kept")
            self.assertTrue(slots.kept_build(build))
            self.assertIn("rkpr1: kept for #7", rk("pr-slot", "status")[1])
            self.assertEqual(rk("pr-slot", "take")[0], 2)  # no --pid
            self.assertEqual(rk("pr-slot", "keep", *keep)[0], 2)  # no --record

    def test_on_a_host_rk_pr_run_runs_the_command_in_the_folder_rk_was_called_from(
        self,
    ):
        w = self.tree()
        env = dict(os.environ, RKMON_HOME=str(config_home(self.d, w)))
        env.pop(prcode.RUNNER_VAR)
        p = subprocess.run(
            ["sh", str(ROOT / "rk"), "pr-run", "--", "pwd"],
            cwd=w / "wt7" / "rk",
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            (p.returncode, p.stdout), (0, f"{w / 'wt7' / 'rk'}\n"), p.stderr
        )

    def test_the_rerun_commands_start_with_rk_pr_run(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import installed_tests
        import pytest_vs_base

        self.assertIn(f"export {prcode.RERUN_VAR}=", prcode.SHELL)
        via = subprocess.run(
            ["bash", "-c", prcode.SHELL + f'\necho "${prcode.RERUN_VAR}"'],
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(via, prcode.RERUN_VIA)
        self.assertTrue(via.endswith("/rk pr-run"), via)
        pytest = b" ".join(pytest_vs_base.PYTEST).decode()
        # the run's environment file where the job copies the results (/r/), the tree, then the command
        self.assertEqual(
            pytest_vs_base.rerun_line(b"/r/", b"/t", b"a.py", via.encode()).decode(),
            f"   {via} --env /r/tests_env.0 -C /t -- {pytest} --rootdir=/t a.py",
        )
        self.assertEqual(
            pytest_vs_base.rerun_line(b"/o/", b"/t", b"a.py").decode(),
            f"   (. /o/tests_env.sh && cd /t && {pytest} --rootdir=/t a.py)",
        )
        entry = {"name": "e", "cwd": "/t", "python": "python"}
        plan = {
            "mapped": ["a.py"],
            "lanes": {"a.py": ["e"]},
            "whole": False,
            "runs": [],
        }
        with mock.patch.dict(
            os.environ, {prcode.RERUN_VAR: via, prcode.RESULTS_VAR: "/r"}
        ), mock.patch.object(
            installed_tests, "env_data", return_value={"COLUMNS": "400"}
        ) as data, mock.patch.object(
            installed_tests,
            "pytest_cmd",
            return_value=["python", "-m", "pytest", "a.py"],
        ):
            lines = installed_tests.rerun_lines(
                plan, [entry], {"a.py": "a.py"}, "", "", "/o"
            )
        # written in the run's out, named where the job copies it; the names its file leaves out on the line
        self.assertEqual(data.call_args.args[3], "/o/env_e.0")
        self.assertEqual(
            lines,
            [
                f"   a.py [e]: {via} --env /r/env_e.0 -C /t -- env COLUMNS=400 python -m pytest a.py"
            ],
        )


class TestRootSteps(unittest.TestCase):
    """docker/rk_root.py: the write login's move, as the start runs it."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.old, self.new = self.d / "secrets" / "gh", self.d / "secrets-write" / "gh"
        self.old.mkdir(parents=True)
        (self.d / "secrets-write").mkdir()
        self.chowned = []

    def chown(self, path, uid, gid, follow_symlinks=True):
        self.assertFalse(follow_symlinks)
        self.chowned.append((str(path), uid, gid))

    def move(self):
        import rk_root

        return rk_root.move_write_login(
            str(self.old), str(self.new), owner=(1999, 1000), chown=self.chown
        )

    def test_an_older_login_moves_once_and_nothing_of_it_is_printed(self):
        (self.old / "hosts.yml").write_text(
            "github.com:\n    oauth_token: FAKE-WRITE-TOKEN\n"
        )
        (self.old / "config.yml").write_text("version: 1\n")
        lines = self.move()
        self.assertEqual(len(lines), 1)
        self.assertIn("moved the GitHub write login", lines[0])
        self.assertNotIn("FAKE-WRITE-TOKEN", "\n".join(lines))
        self.assertFalse(self.old.exists())
        self.assertIn("FAKE-WRITE-TOKEN", (self.new / "hosts.yml").read_text())
        self.assertEqual(stat.S_IMODE(self.new.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.new / "hosts.yml").stat().st_mode), 0o600)
        self.assertEqual(
            sorted(p for p, _u, _g in self.chowned),
            sorted(
                [
                    str(self.new),
                    str(self.new / "hosts.yml"),
                    str(self.new / "config.yml"),
                ]
            ),
        )
        self.assertEqual(self.move(), [])  # again: nothing to do

    def test_nothing_to_move_and_a_login_already_there(self):
        self.assertEqual(self.move(), [])
        self.new.mkdir()
        (self.new / "hosts.yml").write_text("x")
        (self.old / "hosts.yml").write_text("y")
        self.assertEqual(self.move(), [])
        self.assertTrue(
            (self.old / "hosts.yml").exists()
        )  # the write volume's login stands

    def test_a_link_is_never_followed(self):
        secret = self.d / "elsewhere"
        secret.write_text("not the login")
        (self.old / "hosts.yml").symlink_to(secret)
        lines = self.move()
        self.assertIn("was not moved", lines[0])
        self.assertFalse(self.new.exists())
        self.assertTrue(self.old.exists())
        shutil.rmtree(self.old)
        target = self.d / "dir"
        target.mkdir()
        (target / "hosts.yml").write_text("x")
        self.old.symlink_to(target)
        self.assertIn("was not moved", self.move()[0])
        self.assertEqual(self.chowned, [])

    def test_the_write_volume_is_the_write_users(self):
        import rk_root

        vol = self.d / "secrets-write"
        vol.chmod(0o755)
        lines = rk_root.write_volume(str(vol), owner=(1999, 1000), chown=self.chown)
        self.assertIn(f"{vol} is now mode 700", lines)
        self.assertEqual(stat.S_IMODE(vol.stat().st_mode), 0o700)
        self.assertEqual(rk_root.write_volume(str(self.d / "none")), [])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(rk_root.main(["x"]), 2)  # no argument, ever

    def test_each_start_says_how_to_add_the_read_login(self):
        """While there is no read login, the start of `run` names the one step that adds it (the root step itself runs
        in the root phase: tests/test_entrypoint.py)."""
        from test_entrypoint import function

        body = function("read_login_note")
        say = 'say() { echo "SAY $*"; }\n'
        secrets = self.d / "secrets"

        def run():
            return subprocess.run(
                ["sh", "-c", say + body + "read_login_note\n"],
                capture_output=True,
                text=True,
                timeout=30,
                env=dict(os.environ, RKMON_SECRETS_DIR=str(secrets)),
            ).stdout

        self.assertIn("SAY no GitHub read login yet", run())
        self.assertIn("rkmon login github_read", run())
        (secrets / "gh-read").mkdir()
        (secrets / "gh-read" / "hosts.yml").write_text("x")
        self.assertEqual(run(), "")

    def test_the_image_allows_these_two_and_nothing_else(self):
        text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        rules = re.findall(r"'(rkmon ALL=[^']*)'", text)
        self.assertEqual(
            rules,
            [
                "rkmon ALL=(rkwrite) NOPASSWD: /opt/rocke_monitor/docker/rk_write.py",
                "rkmon ALL=(%s) NOPASSWD: /opt/rocke_monitor/docker/rk_prcode.py\\n",
            ],
        )
        # The launcher's rule names each PR runner (tests/test_container_layout.py checks the list).
        self.assertIn("\"$(seq -s, -f 'rkpr%g' 1 16)\"", text)
        self.assertIn("-G rkpr -s /bin/bash rkmon", text)
        self.assertIn("RKMON_PR_RUNNER=/opt/rocke_monitor/docker/rk_prcode.py", text)
        # The root step is no sudo rule's: the entry point's root phase runs it (tests/test_entrypoint.py).
        self.assertNotIn("sudoers.d/rkmon-root", text)
        for f in ("rk_prcode.py", "rk_root.py", "rk_write.py"):
            self.assertTrue(os.access(ROOT / "docker" / f, os.X_OK), f)
            self.assertTrue(
                (ROOT / "docker" / f).read_text().startswith("#!/usr/bin/python3 -I\n")
            )


if __name__ == "__main__":
    unittest.main()
