# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the gfx1250 hardware job's gate: which suites run for which event and files."""

import fnmatch
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from configure_gfx1250_hw_ci import PATHS, decide  # noqa: E402

RULES = tomllib.loads(PATHS.read_text())
ORG = '["organization: ROCm"]'


def pr_env(**extra):
    env = {"EVENT": "pull_request", "ACTION": "synchronize", "PR": "1", "LABELS": ORG}
    env.update(extra)
    return env


def suites(decision):
    return decision.run_hipblaslt, decision.run_tensilelite


class EventTest(unittest.TestCase):
    def test_schedule_runs_both_suites(self):
        self.assertEqual(suites(decide({"EVENT": "schedule"}, [], RULES)), (True, True))

    def test_dispatch_follows_its_inputs(self):
        env = {
            "EVENT": "workflow_dispatch",
            "RUN_HIPBLASLT": "true",
            "RUN_TENSILELITE": "false",
        }
        self.assertEqual(suites(decide(env, [], RULES)), (True, False))

    def test_dispatch_with_both_suites_off_skips_the_job(self):
        env = {
            "EVENT": "workflow_dispatch",
            "RUN_HIPBLASLT": "false",
            "RUN_TENSILELITE": "false",
        }
        self.assertFalse(decide(env, [], RULES).run)

    def test_dispatch_without_inputs_runs_both_suites(self):
        self.assertEqual(
            suites(decide({"EVENT": "workflow_dispatch"}, [], RULES)), (True, True)
        )


class PullRequestGateTest(unittest.TestCase):
    CHANGE = ["projects/hipblaslt/tensilelite/tensilelite/KernelWriter.py"]

    def test_draft_is_skipped(self):
        self.assertFalse(decide(pr_env(DRAFT="true"), self.CHANGE, RULES).run)

    def test_pr_without_an_org_label_is_skipped(self):
        self.assertFalse(decide(pr_env(LABELS="[]"), self.CHANGE, RULES).run)

    def test_streamhpc_label_alone_is_skipped(self):
        env = pr_env(LABELS='["organization: streamhpc"]')
        self.assertFalse(decide(env, self.CHANGE, RULES).run)

    def test_unrelated_label_event_is_skipped(self):
        self.assertFalse(
            decide(pr_env(ACTION="labeled", LABEL="ci:skip"), self.CHANGE, RULES).run
        )

    def test_org_label_arriving_runs(self):
        env = pr_env(ACTION="labeled", LABEL="organization: ROCm")
        self.assertTrue(decide(env, self.CHANGE, RULES).run)

    def test_unlisted_files_run_both_suites(self):
        self.assertEqual(suites(decide(pr_env(), None, RULES)), (True, True))

    def test_truncated_file_list_runs_both_suites(self):
        env = pr_env(CHANGED_FILES="3500")
        self.assertEqual(suites(decide(env, ["docs/a.md"] * 3000, RULES)), (True, True))


class PathRulesTest(unittest.TestCase):
    def run_for(self, files):
        return suites(decide(pr_env(), files, RULES))

    def test_docs_only_change_skips_the_job(self):
        files = ["projects/hipblaslt/docs/index.rst", "projects/hipblaslt/README.md"]
        self.assertEqual(self.run_for(files), (False, False))

    def test_unrelated_project_skips_the_job(self):
        self.assertEqual(
            self.run_for(["projects/rocblas/library/src/a.cpp"]), (False, False)
        )

    def test_hipblaslt_library_change_skips_tensilelite(self):
        files = ["projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp"]
        self.assertEqual(self.run_for(files), (True, False))

    def test_tensilelite_change_runs_both_suites(self):
        files = ["projects/hipblaslt/tensilelite/tensilelite/KernelWriter.py"]
        self.assertEqual(self.run_for(files), (True, True))

    def test_mixed_change_runs_both_suites(self):
        files = [
            "projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp",
            "shared/origami/src/a.cpp",
        ]
        self.assertEqual(self.run_for(files), (True, True))

    def test_therock_pin_runs_both_suites(self):
        self.assertEqual(
            self.run_for([".github/actions/ci-env/action.yml"]), (True, True)
        )

    def test_every_run_pattern_matches_a_tracked_file(self):
        # A pattern for a moved or deleted path would silently stop triggering the job.
        # git lists the whole tree even in the gate job's sparse checkout.
        repo_root = Path(__file__).resolve().parents[3]
        tracked = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        for pattern in RULES["run"]:
            self.assertTrue(
                any(fnmatch.fnmatch(path, pattern) for path in tracked),
                f"{pattern} matches no tracked file",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
