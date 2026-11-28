# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/golden_vs_base.py: the representative golden gate, its failing lines sorted into load errors and drift by family
and flavor, and compared with develop's (lessons L-523: a load error never reads like develop's llvm24 drift). The gate
is a job's step and the report the monitor's: the report runs no gate, alone writes develop's cached lines, and follows
no link in the PR's tree.
"""

import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
TOOL = PACKAGE / "tools" / "golden_vs_base.py"
FLAVORS = "llvm20, llvm22, llvm23, llvm24"
DEV_DRIFT = [f"[llvm24] attention/case{i}: aaa -> bbb" for i in range(8)]
# The checks' configuration and plan, for their golden section.
CFG = {"rocke_path": "rk", "repo": "o/r", "paths": {"working_dir": "/w"}}
PLAN = {
    "families": [],
    "tests": [],
    "changed_tests": [],
    "conftests": [],
    "modules": [],
    "build": True,
    "golden": True,
    "engine_wide": False,
    "outside_rocke": [],
    "libparity": False,
    "_dirs": {},
}


def fake_tests(
    root: pathlib.Path,
    name: str,
    lines: list[str],
    rc: int = 1,
    shards: dict | None = None,
    marker: pathlib.Path | None = None,
) -> pathlib.Path:
    """A tree's platform/tests with a stub gate printing what the real one prints, and golden shards whose first lines
    carry the given alias maps."""
    t = root / name
    g = t / "golden" / "rocke_representative_ir_sha256"
    g.mkdir(parents=True)
    if lines:
        text = (
            f"rocKE installed golden gate: FAIL ({FLAVORS})\n  "
            + "\n  ".join(lines)
            + "\n"
        )
    else:
        text = f"rocKE installed golden gate: PASS ({FLAVORS}, golden=rocke_representative_ir_sha256)\n"
        rc = 0
    touch = f"open({str(marker)!r}, 'w').close()\n" if marker else ""
    (t / "rocke_installed_golden_test.py").write_text(
        f"import sys\n{touch}sys.stdout.write({text!r})\n" f"sys.exit({rc})\n"
    )
    for shard, aliases in (shards or {"attention": "{}", "gemm": "{}"}).items():
        (g / f"{shard}.json").write_text(
            '{\n  "aliases": ' + aliases + ',\n  "flavors": {}\n}\n'
        )
    return t


def tool(*args, **extra) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHON=sys.executable, **extra)
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def cached(cache) -> bool:
    return (
        bool(cache)
        and pathlib.Path(cache).is_file()
        and pathlib.Path(cache).stat().st_size > 0
    )


def run(out, tests, base, cache="", **extra):
    """What the checks print for the PR's gate (rkmon_lib/review.py): the gate; on failing lines, develop's own gate
    (its OUT out/golden_develop) when BASE names develop's tree and the cache lacks its lines, then the report.
    """
    g = tool("gate", out, tests, **extra)
    assert g.returncode in (0, 1, 2), (g.returncode, g.stderr)
    lines = g.stdout.splitlines()
    if g.returncode != 1:
        return lines
    dev = ""
    if str(base) not in ("-", "self") and not cached(cache):
        dev = pathlib.Path(out) / "golden_develop"
        tool("gate", dev, base, **extra)
    r = tool("report", out, tests, base, cache, dev, **extra)
    assert r.returncode == 0, (
        r.returncode,
        r.stderr,
    )  # the report's last line says the result; the exit is 0
    return lines + r.stdout.splitlines()


class GoldenVsBase(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._d.name)
        self.out = self.root / "out"
        self.out.mkdir()

    def tearDown(self):
        self._d.cleanup()

    def test_a_passing_gate_passes(self):
        t = fake_tests(self.root, "merge", [])
        out = run(self.out, t, "-")
        self.assertEqual(
            out[-1],
            f"GOLDEN pass PASS ({FLAVORS}, golden=rocke_representative_ir_sha256)",
        )

    def test_drift_develop_prints_too_is_not_the_prs(self):
        t = fake_tests(self.root, "merge", DEV_DRIFT)
        b = fake_tests(self.root, "develop", DEV_DRIFT)
        out = run(self.out, t, b)
        self.assertIn(
            "fail on develop too, not this PR's doing (8): attention@llvm24 ×8", out
        )
        self.assertEqual(
            out[-1],
            "GOLDEN pass every failing line fails on develop too (8: attention@llvm24 ×8)",
        )

    def test_the_prs_drift_is_counted_by_family_and_flavor_with_the_re_record_command(
        self,
    ):
        mine = [
            "[llvm20] gemm/a: x -> y",
            "[llvm22] gemm/a: x -> z",
            "[llvm24] cases: new current conv/b",
        ]
        t = fake_tests(self.root, "merge", DEV_DRIFT + mine)
        b = fake_tests(self.root, "develop", DEV_DRIFT)
        out = run(self.out, t, b)
        self.assertIn(
            "drift (3, on the merge only): conv@llvm24 ×1, gemm@llvm20 ×1, gemm@llvm22 ×1",
            out,
        )
        self.assertIn(
            "re-record (from platform/, on a merge with develop): PYTHONPATH=python:../library python "
            "tests/instances/rocke_ir_parity_harness.py --write tests/golden/rocke_representative_ir_sha256 "
            "--only conv,gemm",
            out,
        )
        self.assertEqual(
            out[-1],
            "GOLDEN fail 3 drifting (on the merge only): conv@llvm24 ×1, gemm@llvm20 ×1, "
            "gemm@llvm22 ×1",
        )

    def test_a_shard_that_does_not_load_is_named_with_its_odd_alias_map(self):
        err = (
            "golden shard conv_winograd_fwd_rt.json has aliases {'llvm24': 'llvm23'}, attention.json has {}; "
            "re-bless every shard"
        )
        shards = {
            "attention": "{}",
            "gemm": "{}",
            "conv_winograd_fwd_rt": '{\n    "llvm24": "llvm23"\n  }',
        }
        t = fake_tests(self.root, "merge", [err], shards=shards)
        b = fake_tests(self.root, "develop", DEV_DRIFT)
        out = run(self.out, t, b)
        self.assertIn("the golden does not load (1, on the merge only):", out)
        self.assertIn('   1 "aliases":{"llvm24":"llvm23"} conv_winograd_fwd_rt', out)
        self.assertIn('   2 "aliases":{} attention,gemm', out)
        self.assertTrue(
            any(x.endswith("--only conv_winograd_fwd_rt") for x in out), out
        )
        self.assertEqual(
            out[-1],
            "GOLDEN fail the golden does not load (on the merge only): " + err[:150],
        )

    def test_a_cached_develop_run_is_used_and_a_complete_one_is_kept(self):
        t = fake_tests(self.root, "merge", DEV_DRIFT)
        ran = self.root / "develop_ran"
        b = fake_tests(self.root, "develop", DEV_DRIFT, marker=ran)
        cache = self.root / "golden_develop_37e8aeb87cb1.lines"
        self.assertTrue(run(self.out, t, b, cache)[-1].startswith("GOLDEN pass"))
        self.assertTrue(ran.exists())
        self.assertEqual(
            cache.read_text().splitlines()[0],
            f"#gate rocKE installed golden gate: FAIL ({FLAVORS})",
        )
        ran.unlink()
        out = run(self.out, t, b, cache)
        self.assertFalse(ran.exists(), "develop's gate ran again despite the cache")
        self.assertTrue(
            out[-1].startswith(
                "GOLDEN pass every failing line fails on develop too (8"
            ),
            out,
        )

    def test_the_gate_says_how_it_ended_in_its_exit_code(self):
        """0 pass, 1 failing lines (the report's to sort), 2 no result: the checks run the report on 1 only."""
        t = fake_tests(self.root, "pass", [])
        g = tool("gate", self.out / "p", t)
        self.assertEqual(g.returncode, 0, g.stderr)
        self.assertEqual(
            g.stdout.splitlines()[-1],
            f"GOLDEN pass PASS ({FLAVORS}, golden=rocke_representative_ir_sha256)",
        )
        t = fake_tests(self.root, "merge", DEV_DRIFT)
        g = tool("gate", self.out / "f", t)
        self.assertEqual(g.returncode, 1, g.stderr)
        self.assertEqual(
            g.stdout.splitlines(), [f"rocKE installed golden gate: FAIL ({FLAVORS})"]
        )  # no verdict: the report gives it
        self.assertEqual(
            (self.out / "f" / "golden.log.lines").read_text().splitlines(),
            sorted(DEV_DRIFT),
        )
        g = tool("gate", self.out / "n", self.root / "nowhere")
        self.assertEqual(g.returncode, 2, g.stderr)
        self.assertTrue(
            g.stdout.splitlines()[-1].startswith("GOLDEN fail no gate result (rc=1)"),
            g.stdout,
        )
        bad = tool("report", self.out)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("usage: golden_vs_base.py gate OUT TESTS", bad.stderr)

    def test_the_report_runs_no_gate_and_caches_only_develops_own_complete_run(self):
        t = fake_tests(self.root, "merge", DEV_DRIFT)
        ran = self.root / "develop_ran"
        b = fake_tests(self.root, "develop", DEV_DRIFT, marker=ran)
        cache = self.root / "golden_develop_37e8aeb87cb1.lines"
        self.assertEqual(tool("gate", self.out, t).returncode, 1)
        # No develop run handed to it: the report runs none, and keeps nothing.
        r = tool("report", self.out, t, b, cache)
        self.assertFalse(ran.exists(), "the report ran develop's gate")
        self.assertFalse(cache.exists())
        self.assertIn("develop: no gate line", r.stdout.splitlines())
        self.assertTrue(
            r.stdout.splitlines()[-1].startswith(
                "GOLDEN fail 8 drifting (on the merge only)"
            ),
            r.stdout,
        )
        # A develop run that printed no gate line (a timeout) is used, never kept.
        cut = self.root / "cut"
        cut.mkdir()
        (cut / "golden.log").write_text("started\n")
        (cut / "golden.log.lines").write_text("")
        tool("report", self.out, t, b, cache, cut)
        self.assertFalse(cache.exists())
        # Develop's own gate run (a job of its own): the report keeps its lines for the develop SHA.
        dev = self.root / "dev_out"
        self.assertEqual(tool("gate", dev, b).returncode, 1)
        ran.unlink()
        r = tool("report", self.out, t, b, cache, dev)
        self.assertFalse(ran.exists(), "the report ran develop's gate")
        self.assertEqual(
            cache.read_text().splitlines(),
            [f"#gate rocKE installed golden gate: FAIL ({FLAVORS})"]
            + sorted(DEV_DRIFT),
        )
        self.assertEqual(
            r.stdout.splitlines()[-1],
            "GOLDEN pass every failing line fails on develop too (8: attention@llvm24 ×8)",
        )
        # The gate's failing lines missing (a job that left none to read): never a pass.
        (self.out / "golden.log.lines").unlink()
        r = tool("report", self.out, t, b, cache)
        self.assertEqual(
            r.stdout.splitlines(),
            ["GOLDEN fail no gate result: the gate's failing lines are not there"],
        )

    def test_the_report_follows_no_link_in_the_prs_golden_folder(self):
        """The report runs as the monitor: a link the PR's tree holds would show it a file of its own (here, an
        alias map it would print)."""
        err = "golden shard gemm.json has aliases {}, attention.json has {}; re-bless every shard"
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "state.json").write_text(
            '{\n  "aliases": {"llvm24": "llvm23"},\n  "flavors": {}\n}\n'
        )
        (outside / "attention.json").write_text(
            '{\n  "aliases": {},\n  "flavors": {}\n}\n'
        )
        t = fake_tests(self.root, "merge", [err])
        g = t / "golden" / "rocke_representative_ir_sha256"
        (g / "secret.json").symlink_to(outside / "state.json")
        odd = '"llvm24":"llvm23"'
        out = run(self.out, t, "-")
        self.assertIn("the golden does not load (1, not compared with develop):", out)
        self.assertFalse(any(odd in x or "secret" in x for x in out), out)
        # A golden folder that is a link: nothing in it is read.
        t2 = fake_tests(self.root, "merge2", [err])
        g2 = t2 / "golden" / "rocke_representative_ir_sha256"
        for f in g2.iterdir():
            f.unlink()
        g2.rmdir()
        g2.symlink_to(outside)
        out = run(self.root / "out2", t2, "-")
        self.assertIn("the golden does not load (1, not compared with develop):", out)
        self.assertNotIn("alias maps (shards per map, fewest first):", out)
        self.assertFalse(any(odd in x or "state" in x for x in out), out)

    def test_no_gate_line_is_no_result(self):
        t = self.root / "merge"
        t.mkdir()
        (t / "rocke_installed_golden_test.py").write_text(
            "import sys\nprint('Traceback: ImportError: rocke')\n" "sys.exit(1)\n"
        )
        out = run(self.out, t, "-")
        self.assertEqual(
            out[-1], "GOLDEN fail no gate result (rc=1): Traceback: ImportError: rocke"
        )

    def test_without_a_develop_tree_every_line_is_reported_as_not_compared(self):
        t = fake_tests(self.root, "merge", ["[llvm24] attention/case0: aaa -> bbb"])
        self.assertEqual(
            run(self.out, t, "-")[-1],
            "GOLDEN fail 1 drifting (not compared with develop): attention@llvm24 ×1",
        )
        self.assertEqual(
            run(self.out, t, "self")[-1], "GOLDEN fail 1 drifting: attention@llvm24 ×1"
        )

    def test_a_gate_past_its_time_box_is_no_result(self):
        t = self.root / "merge"
        t.mkdir()
        (t / "rocke_installed_golden_test.py").write_text(
            "import time\nprint('started', flush=True)\ntime.sleep(5)\n"
        )
        out = run(self.out, t, "-", GOLDEN_TIMEOUT="1")
        self.assertEqual(out[-1], "GOLDEN fail no gate result (rc=124): started")

    def test_the_drift_report_lists_six_lines_and_the_files_it_writes(self):
        mine = [f"[llvm20] gemm/k{i}: a -> b" for i in range(7)] + [
            "noflavor family/x",
            "colon:only",
        ]
        t = fake_tests(self.root, "merge", DEV_DRIFT + mine)
        b = fake_tests(self.root, "develop", DEV_DRIFT)
        out = run(self.out, t, b)
        # A family with a space counts under its first word, as `uniq -c | awk '{... $2 ...}'` printed it.
        i = out.index(
            "drift (9, on the merge only): colon@all ×1, gemm@llvm20 ×7, noflavor ×1"
        )
        self.assertEqual(out[i + 1 : i + 7], [f"   {x}" for x in sorted(mine)[:6]])
        self.assertEqual(
            (self.out / "golden_old.txt").read_text().splitlines(), sorted(DEV_DRIFT)
        )
        self.assertEqual(
            (self.out / "golden_pr_drift.txt").read_text().splitlines(), sorted(mine)
        )
        self.assertEqual((self.out / "golden_pr_load.txt").read_text(), "")


class TheScriptsRunTheSteps(unittest.TestCase):
    """The checks' golden section (rkmon_lib/review.py) and the develop sanity's (guard.py) run the steps in turn:
    develop's gate only when its lines are not cached, the report only after failing lines.
    """

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._d.name)
        self.out = self.root / "out"
        self.out.mkdir()

    def tearDown(self):
        self._d.cleanup()

    @staticmethod
    def section(script: str, start: str) -> str:
        lines = script.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith(start))
        j = next(k for k, ln in enumerate(lines[i:], i) if "st golden " in ln)
        return "\n".join(lines[i : j + 1])

    def sh(self, body: str, **var) -> list[str]:
        from rkmon_lib import prcode, review

        env = {k: v for k, v in os.environ.items() if k != prcode.RUNNER_VAR}
        env["PYTHON"] = sys.executable
        env["TMPDIR"] = str(
            self.root
        )  # the slot: a folder of the session user's (a host's)
        pre = [
            prcode.SHELL,
            review.U8_FN,
            'sec() { echo "== $*"; }; st() { echo "ST $*"; }; base_tree() { :; }',
        ] + [f"{k}={shlex.quote(str(v))}" for k, v in var.items()]
        pre.append("prslot || exit 9")
        r = subprocess.run(
            ["bash", "-c", "set -u\n" + "\n".join(pre) + "\n" + body],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.splitlines()

    def test_the_checks_run_develops_gate_once_per_develop_sha(self):
        from rkmon_lib import review

        body = self.section(
            review.checks_script(CFG, 1, "a" * 40, "follow-up", PLAN, None),
            "sec 'Representative golden gate",
        )
        fake_tests(
            self.root, "w/rk/platform/tests", DEV_DRIFT + ["[llvm20] gemm/a: x -> y"]
        )
        ran = self.root / "develop_ran"
        fake_tests(self.root, "bw/rk/platform/tests", DEV_DRIFT, marker=ran)
        (self.out / "develop_sha").write_text("37e8aeb87cb1" + "0" * 28 + "\n")
        var = {
            "M": PACKAGE,
            "P": self.root / "w/rk/platform",
            "BR": "develop",
            "BW": self.root / "bw",
            "K": "rk",
            "OUT": self.out,
            "CP": self.root / "cache" / "develop",
        }
        verdict = "ST golden fail 1 drifting (on the merge only): gemm@llvm20 ×1"
        self.assertIn(verdict, self.sh(body, **var))
        self.assertTrue(ran.exists())
        cache = self.root / "cache" / "develop" / "golden_develop_37e8aeb87cb1.lines"
        self.assertEqual(
            cache.read_text().splitlines(),
            [f"#gate rocKE installed golden gate: FAIL ({FLAVORS})"]
            + sorted(DEV_DRIFT),
        )
        ran.unlink()
        out = self.sh(body, **var)
        self.assertIn(verdict, out)
        self.assertIn(
            f"develop (an earlier run at the same SHA): rocKE installed golden gate: FAIL ({FLAVORS})",
            out,
        )
        self.assertFalse(ran.exists(), "develop's gate ran again despite the cache")
        # A passing gate: no develop gate, no report.
        fake_tests(self.root, "ok/rk/platform/tests", [])
        out = self.sh(body, **dict(var, P=self.root / "ok/rk/platform"))
        self.assertIn(
            f"ST golden pass PASS ({FLAVORS}, golden=rocke_representative_ir_sha256)",
            out,
        )
        self.assertFalse(ran.exists())

    def test_the_develop_sanity_reports_failing_lines_against_none(self):
        from rkmon_lib import guard

        body = self.section(
            guard.sanity_script(CFG, "b" * 40, "/w/wt", "/w/out", ["gemm"], False),
            "echo; echo '## Representative golden'",
        )
        fake_tests(
            self.root, "wt/rk/platform/tests", ["[llvm24] attention/case0: aaa -> bbb"]
        )
        var = {"M": PACKAGE, "P": self.root / "wt/rk/platform", "OUT": self.out}
        self.assertIn(
            "ST golden fail 1 drifting: attention@llvm24 ×1", self.sh(body, **var)
        )
        fake_tests(self.root, "ok/rk/platform/tests", [])
        self.assertIn(
            f"ST golden pass PASS ({FLAVORS}, golden=rocke_representative_ir_sha256)",
            self.sh(body, **dict(var, P=self.root / "ok/rk/platform")),
        )


if __name__ == "__main__":
    unittest.main()
