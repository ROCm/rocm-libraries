# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/probes.py: `rk probe-check` (a saved probe reads 2 on DEVBASE, 1 on the tree, 0 on a simulated fix, and
carries no flag) and `rk probe-port` (probes written for a live layout rewritten to the probe environment contract).
"""

import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import sandbox  # noqa: E402
from rkmon_lib import probes  # noqa: E402

LIVE = "/home/user/rocke_review"  # a live layout (a placeholder user)

# A probe of a made-up finding: the PR adds `feature.txt` holding "bug"; our fix line says the value must not be "bug".
PROBE = """#!/bin/bash
# exit 1 while the feature file holds the bug, 0 once fixed, 2 where the PR's change is not there
f="${1%/}/src/feature.txt"
[ -f "$f" ] || { echo "not applicable: no feature file"; exit 2; }
grep -qx bug "$f" && { echo "the feature holds the bug"; exit 1; }
echo "fixed"; exit 0
"""


def git(cwd, *args):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


class Repo:
    """A clone with develop (no feature) and the PR's head (the feature, with the bug), checked out at the head."""

    def __init__(self, base: pathlib.Path):
        self.tree = base / "tree"
        (self.tree / "src").mkdir(parents=True)
        git(self.tree, "init", "-q", "-b", "develop")
        (self.tree / "src" / "core.txt").write_text("core\n")
        git(self.tree, "add", "-A")
        git(self.tree, "commit", "-q", "-m", "develop")
        git(self.tree, "checkout", "-q", "-b", "pr")
        (self.tree / "src" / "feature.txt").write_text("bug\n")
        git(self.tree, "add", "-A")
        git(self.tree, "commit", "-q", "-m", "the PR")
        self.fix = base / "fix.patch"
        self.fix.write_text(
            "--- a/src/feature.txt\n+++ b/src/feature.txt\n@@ -1 +1 @@\n-bug\n+fine\n"
        )


class TestReading(unittest.TestCase):
    def test_a_crash_pytests_rc_5_and_other_codes_read_3(self):
        self.assertEqual(
            [probes.reading(rc, False) for rc in (0, 1, 2, 3, 5, 134)],
            [0, 1, 2, 3, 3, 3],
        )
        self.assertEqual(probes.reading(1, True), 3)

    def test_the_run_probes_line_is_read(self):
        m = probes.LINE.match(
            "N1_x: ERROR rc=1 (crashed) (tree abc, 2 s): ImportError: y"
        )
        self.assertEqual(
            (m.group("name"), m.group("rc"), bool(m.group("crash"))),
            ("N1_x", "1", True),
        )
        m = probes.LINE.match("N2_y: REPRODUCES rc=1 (tree abc, 0 s): still")
        self.assertEqual((m.group("rc"), m.group("crash")), ("1", None))


class TestFlags(unittest.TestCase):
    def test_host_paths_failed_greps_the_gpu_mask_and_no_tree_root(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "N1_bad.sh").write_text(
                "#!/bin/bash\n# a comment may say /home/user/x\n"
                f"{LIVE}/venv/bin/python -B N1_bad_check.py\n"
                "export HIP_VISIBLE_DEVICES=-1\n"
                "out | grep -c '^FAILED'\n"
            )
            (p / "N1_bad_check.py").write_text(
                'LLC = "/opt/rocm/core-9.9/lib/llvm/bin/llc"\n'
            )
            (p / "N2_good.sh").write_text(
                PROBE + '"$PROBE_PYTHON" -m pytest -q | grep -E "^(SUB)?FAILED"\n'
            )
            bad = probes.flags(p / "N1_bad.sh")
            good = probes.flags(p / "N2_good.sh")
        text = "\n".join(bad)
        self.assertIn(
            "N1_bad.sh:3: hardcoded host path /home/<user>/rocke_review/venv/bin/python",
            text,
        )
        self.assertNotIn("N1_bad.sh:2", text)  # a comment is no code
        self.assertIn(
            "N1_bad_check.py:1: hardcoded host path /opt/rocm/core-9.9/lib/llvm/bin/llc",
            text,
        )
        self.assertIn("greps ^FAILED without SUBFAILED", text)
        self.assertIn("HIP_VISIBLE_DEVICES=-1", text)
        self.assertIn("never reads its tree root ($1)", text)
        self.assertEqual(good, [])

    def test_static_reads_a_probes_root_but_its_retired_folders(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            for sub in ("pr1", "pr2", "retired"):
                (p / sub).mkdir()
            (p / "pr1" / "N1.sh").write_text(PROBE)
            (p / "pr2" / "N1.sh").write_text(f"{LIVE}/venv/bin/python x.py $1\n")
            (p / "retired" / "N1.sh").write_text(f"{LIVE}/venv/bin/python x.py\n")
            lines, code = probes.static(p)
        self.assertEqual(code, 1)
        self.assertEqual(lines[-1], "2 probes, 1 flagged, 1 clean")
        self.assertIn("pr2/N1.sh: 1 flag(s)", lines)


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.d.name)
        self.repo = Repo(self.base)
        self.folder = self.base / "probes"
        self.folder.mkdir()
        self.env = dict(os.environ)

    def tearDown(self):
        self.d.cleanup()

    def check(self, **kw):
        return probes.check(
            self.folder, self.repo.tree, env=self.env, scratch=self.base, **kw
        )

    def test_a_good_probe_reads_2_1_0(self):
        (self.folder / "N1_bug.sh").write_text(PROBE)
        lines, code = self.check(simfix=self.repo.fix)
        self.assertEqual(lines[0], "N1_bug: DEVBASE 2 · tree 1 · simfix 0 · OK", lines)
        self.assertEqual(code, 0)
        self.assertTrue(any(x.startswith("(DEVBASE ") for x in lines), lines)
        # the clone is untouched: no worktree was added, its files are as they were
        out = subprocess.run(
            ["git", "-C", str(self.repo.tree), "worktree", "list"],
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertEqual((self.repo.tree / "src" / "feature.txt").read_text(), "bug\n")
        self.assertEqual(
            [p.name for p in self.base.iterdir() if p.name.startswith("probe_check_")],
            [],
        )

    def test_a_probe_that_keys_on_nothing_fails_the_devbase_and_simfix_reads(self):
        (self.folder / "N2_always.sh").write_text(
            '#!/bin/bash\necho "always $1"\nexit 1\n'
        )
        lines, code = self.check(simfix=self.repo.fix)
        self.assertEqual(
            lines[0],
            "N2_always: DEVBASE 1 · tree 1 · simfix 1 · NOT OK (DEVBASE must read 2; simfix must read 0)",
        )
        self.assertIn("   DEVBASE: N2_always: REPRODUCES rc=1", lines[1])
        self.assertEqual(code, 1)

    def test_a_crash_reads_3(self):
        (self.folder / "N3_crash.sh").write_text(
            '#!/bin/bash\n[ -d "$1" ] || exit 2\n'
            f'{sys.executable} -c "import no_such_module_here"\n'
        )
        lines, code = self.check(simfix=self.repo.fix, only="N3_crash")
        self.assertTrue(
            lines[0].startswith("N3_crash: DEVBASE 3 · tree 3 · simfix 3 · NOT OK"),
            lines,
        )

    def test_without_a_simulated_fix_the_check_is_not_ok(self):
        (self.folder / "N1_bug.sh").write_text(PROBE)
        lines, code = self.check()
        self.assertEqual(
            lines[0], "N1_bug: DEVBASE 2 · tree 1 · simfix - · NOT OK (simfix not run)"
        )
        self.assertIn("--simfix PATCH", "\n".join(lines))
        self.assertEqual(code, 1)

    def test_given_trees_are_used_as_they_are(self):
        (self.folder / "N1_bug.sh").write_text(PROBE)
        dev, fixed = self.base / "dev", self.base / "fixed"
        (dev / "src").mkdir(parents=True)
        (fixed / "src").mkdir(parents=True)
        (fixed / "src" / "feature.txt").write_text("fine\n")
        lines, code = self.check(devbase_tree=dev, simfix_tree=fixed)
        self.assertEqual(lines[0], "N1_bug: DEVBASE 2 · tree 1 · simfix 0 · OK")

    def test_a_merged_tree_takes_its_first_parent_as_devbase(self):
        git(self.repo.tree, "checkout", "-q", "develop")
        (self.repo.tree / "src" / "other.txt").write_text("develop moved\n")
        git(self.repo.tree, "add", "-A")
        git(self.repo.tree, "commit", "-q", "-m", "develop moves")
        dev = subprocess.run(
            ["git", "-C", str(self.repo.tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        git(self.repo.tree, "merge", "-q", "--no-ff", "pr", "-m", "merge")
        self.assertEqual(probes.devbase_rev(self.repo.tree, "develop"), dev)

    def test_a_patch_that_does_not_apply_is_said(self):
        (self.folder / "N1_bug.sh").write_text(PROBE)
        bad = self.base / "bad.patch"
        bad.write_text("--- a/src/none.txt\n+++ b/src/none.txt\n@@ -1 +1 @@\n-x\n+y\n")
        lines, code = self.check(simfix=bad)
        self.assertIn("simfix -", lines[0])
        self.assertTrue(
            any("the simulated fix does not apply" in x for x in lines), lines
        )


class TestPortShell(unittest.TestCase):
    def test_the_common_forms_become_the_contracts_names(self):
        src = (
            "#!/bin/bash\n"
            f"# needs {LIVE}/build/<tree>/core\n"
            f"PY={LIVE}/venv/bin/python3.12\n"
            "PY2=${PY2:-$HOME/rocke_review/venv/bin/python}\n"
            'A=~/rocke_review/build/$(basename "$1")/core/librocke_core.a\n'
            'B="$HOME"/rocke_review/repo\n'
            "export PATH=$HOME/rocke_review/venv/bin:/opt/rocm/core-9.9/lib/llvm/bin:$PATH\n"
            "W=$(mktemp -d ${HOME}/rocke_review/tmp/x.XXXX)\n"
            "INC=/opt/rocm/core-9.9/include\n"
            "C=/opt/rocm/llvm/bin/llvm-objdump\n"
        )
        out, left, n = probes.port_sh(src)
        self.assertEqual(left, [])
        self.assertEqual(n, 10)
        self.assertEqual(
            out,
            "#!/bin/bash\n"
            "# needs $RKMON_BUILD_DIR/core\n"
            "PY=$PROBE_PYTHON\n"
            "PY2=${PY2:-$PROBE_PYTHON}\n"
            "A=$RKMON_BUILD_DIR/core/librocke_core.a\n"
            "B=$RKMON_REPO_DIR\n"
            "export PATH=${PROBE_PYTHON%/*}:$RKMON_ROCM_LLVM_BIN:$PATH\n"
            "W=$(mktemp -d ${TMPDIR:-/tmp}/x.XXXX)\n"
            "INC=${RKMON_ROCM_LLVM_BIN%/lib/llvm/bin}/include\n"
            "C=$RKMON_ROCM_LLVM_BIN/llvm-objdump\n",
        )
        self.assertEqual(
            subprocess.run(["bash", "-n"], input=out, text=True).returncode, 0
        )

    def test_what_has_no_contract_name_or_would_not_expand_is_a_hand_edit(self):
        src = (
            f"T={LIVE}/venv-torch/bin/python\n"
            f"G={LIVE}/venv-gen/bin/python\n"
            f"P={LIVE}/monitor/tools/pytest_plugins\n"
            f"W={LIVE}/wt12/x\n"
            f"echo '{LIVE}/venv/bin/python'\n"
            "cat <<'EOF'\n"
            f"{LIVE}/venv/bin/python\n"
            "EOF\n"
            '[ "$x" = /opt/rocm ]\n'
        )
        out, left, n = probes.port_sh(src)
        self.assertEqual(n, 0)
        self.assertEqual(out, src)
        reasons = [x.split(": ", 2)[2] for x in left]
        self.assertEqual(
            reasons,
            [
                "a torch venv: the contract has none (CI runs without torch)",
                "a second venv (venv-gen): the contract has none",
                "the live toolkit's folder: the contract has none",
                "a review's worktree: use the tree root ($1)",
                "inside single quotes",
                "inside a quoted heredoc",
                "names /opt/rocm itself (a host's ROCm, not the job's)",
            ],
        )
        self.assertTrue(all("/home/<user>/" in x for x in left[:5]), left)

    def test_another_root_name(self):
        out, left, n = probes.port_sh(
            "/home/user/ck/venv/bin/python x\n", names=("ck",)
        )
        self.assertEqual((out, left, n), ("$PROBE_PYTHON x\n", [], 1))


class TestPortPython(unittest.TestCase):
    def test_literals_expanduser_comments_and_docstrings(self):
        src = (
            f"#!{LIVE}/venv/bin/python3.12\n"
            f'"""Runs {LIVE}/venv/bin/python on the tree."""\n'
            "from __future__ import annotations\n"
            "import sys\n"
            f'PY = "{LIVE}/venv/bin/python"  # was {LIVE}/venv/bin/python\n'
            'SCR = os.path.expanduser("~/rocke_review/scratch/x")\n'
            'CLANG = "/opt/rocm/core-9.9/lib/llvm/bin/clang"\n'
            f'B = Path("{LIVE}/build/wt7m/core")\n'
        )
        out, left, n = probes.port_py(src)
        self.assertEqual(left, [])
        self.assertEqual(
            out,
            "#!/usr/bin/env python3\n"
            '"""Runs $PROBE_PYTHON on the tree."""\n'
            "from __future__ import annotations\n"
            "import os  # rk probe-port: the probe environment contract\n"
            "import sys\n"
            'PY = os.environ.get("PROBE_PYTHON", "python3")  # was $PROBE_PYTHON\n'
            'SCR = os.environ.get("TMPDIR", "/tmp") + "/x"\n'
            'CLANG = os.environ["RKMON_ROCM_LLVM_BIN"] + "/clang"\n'
            'B = Path(os.environ["RKMON_BUILD_DIR"] + "/core")\n',
        )
        compile(out, "x.py", "exec")

    def test_paths_inside_longer_strings_f_strings_and_home_expressions_are_hand_edits(
        self,
    ):
        src = (
            "import os\n"
            f'MSG = "run it with {LIVE}/venv-torch"\n'
            'env = {"PATH": "/opt/rocm/core-9.9/bin:" + os.environ["PATH"]}\n'
            'B = f"~/rocke_review/build/{name}"\n'
            'C = Path.home() / "rocke_review/build"\n'
        )
        out, left, n = probes.port_py(src)
        self.assertEqual(
            n, 1
        )  # line 3's literal starts with the path: ROCm's root from the contract, + "/bin:"
        self.assertIn(
            'env = {"PATH": os.environ["RKMON_ROCM_LLVM_BIN"].rsplit("/lib/llvm/bin", 1)[0] + "/bin:" + os.environ',
            out,
        )
        self.assertEqual(
            [x.split(":")[0] for x in left], ["line 2", "line 4", "line 5"]
        )
        self.assertIn("a torch venv", left[0])
        self.assertIn("a home path built in Python", left[-1])


class TestPort(unittest.TestCase):
    def test_dry_run_counts_and_write_rewrites_only_the_clean_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            pr = p / "pr7"
            pr.mkdir()
            (pr / "N1_clean.sh").write_text(
                f'{LIVE}/venv/bin/python N1_clean_check.py "$1"\n'
            )
            (pr / "N1_clean_check.py").write_text(f'PY = "{LIVE}/venv/bin/python"\n')
            (pr / "N2_torch.sh").write_text(f'{LIVE}/venv-torch/bin/python x.py "$1"\n')
            (pr / "N3_none.sh").write_text(PROBE)
            os.chmod(pr / "N1_clean.sh", 0o755)
            lines, code = probes.port(p)
            self.assertEqual(code, 1)
            self.assertIn(
                "probes: 3: 1 convert cleanly, 1 need a hand edit, 1 need nothing",
                lines,
            )
            self.assertIn(
                "files: 4: 2 convert cleanly, 1 need a hand edit, 1 need nothing "
                "(dry run: --write rewrites the clean ones)",
                lines,
            )
            self.assertIn(
                f"{LIVE}/venv/bin", (pr / "N1_clean.sh").read_text()
            )  # a dry run writes nothing
            self.assertFalse(
                any("/home/user/" in x for x in lines), lines
            )  # the user's name is never printed
            lines, code = probes.port(p, write=True)
            self.assertEqual(
                (pr / "N1_clean.sh").read_text(),
                '$PROBE_PYTHON N1_clean_check.py "$1"\n',
            )
            self.assertTrue((pr / "N1_clean.sh").stat().st_mode & stat.S_IXUSR)
            self.assertIn("venv-torch", (pr / "N2_torch.sh").read_text())
            self.assertEqual(
                probes.port(p)[0][0],
                "pr7/N2_torch.sh: needs a hand edit (1; 0 path(s) would convert)",
            )

    def test_nothing_to_read(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(probes.port(pathlib.Path(d))[1], 2)


class TestCommands(unittest.TestCase):
    def setUp(self):
        # The commands read the configuration in the monitor's home: one of the test's own, which has none.
        sandbox.own_home(self)

    def rk(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "rkmon.py"), *args],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )

    def test_rk_probe_port_and_probe_check_run_with_any_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "N1.sh").write_text(f"{LIVE}/venv/bin/python x.py $1\n")
            r = self.rk("probe-port", str(p))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("N1.sh: converts (1 path(s))", r.stdout)
            r = self.rk("probe-check", str(p))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("1 probes, 1 flagged, 0 clean", r.stdout)
            r = self.rk("probe-check", str(p), str(p / "no-tree"))
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
