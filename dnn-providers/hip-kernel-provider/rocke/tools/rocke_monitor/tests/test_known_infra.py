# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""known_infra: a merge-only head's new CI failure is known infrastructure by a configured log signature, by the same
first error line at develop's latest SHA, or by the same on at least two other open PRs today; anything unread or
unmatched is not known."""

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rkmon_lib import gh, known_infra  # noqa: E402
import home_repository  # noqa: E402

CFG = home_repository.config({"repo": "o/r"})  # with the home repository's signatures
TODAY = "2026-11-10"
JOB = "Windows::release / Test hipkernelprovider (shard 1/2)"


def run(
    job: int, conclusion: str = "failure", completed: str = TODAY + "T10:00:00Z"
) -> dict:
    return {
        "name": JOB,
        "status": "completed",
        "conclusion": conclusion,
        "started": completed,
        "completed": completed,
        "url": f"https://github.com/ROCm/r/actions/runs/1/job/{job}",
    }


def log(first: str) -> str:
    return (
        "\ufeff2026-11-10T10:00:00.1Z ##[group]Run tests\n2026-11-10T10:00:01.2Z setting up\n"
        f"2026-11-10T10:00:02.3Z {first}\n2026-11-10T10:00:03.4Z ##[error]Process completed with exit code 1.\n"
    )


class TestKnownInfra(unittest.TestCase):
    def setUp(self):
        self.runs, self.logs = {}, {}
        self.st = {
            "develop": {"sha": "d" * 40},
            "prs": {
                str(k): {
                    "n": k,
                    "status": "open",
                    "head": str(k % 10) * 40,
                    "head_seen_at": f"2026-11-10T0{k % 10}:00:00Z",
                }
                for k in (7, 11, 12, 13)
            },
        }
        for p in (
            mock.patch.object(
                known_infra.ci_runs,
                "check_runs",
                lambda cfg, sha: self.runs.get(sha, []),
            ),
            mock.patch.object(known_infra.gh, "raw", self.raw),
        ):
            p.start()
            self.addCleanup(p.stop)

    def raw(self, path):
        job = int(path.split("/jobs/")[1].split("/")[0])
        if job not in self.logs:
            raise gh.GhError("HTTP 404")
        return self.logs[job]

    def classify(self, cfg=CFG):
        return known_infra.classify(cfg, self.st, 7, "h" * 40, [JOB], today=TODAY)

    def test_first_error_skips_the_runners_exit_line_and_the_timestamp(self):
        self.assertEqual(
            known_infra.first_error(log("##[error]pwsh: command not found")),
            "pwsh: command not found",
        )
        self.assertEqual(
            known_infra.first_error(log("FAILED test_x.py::test_a - assert 0")),
            "FAILED test_x.py::test_a - assert 0",
        )
        self.assertEqual(
            known_infra.first_error(
                "all good\n##[error]Process completed with exit code 2.\n"
            ),
            "",
        )

    def test_a_configured_signature(self):
        self.runs["h" * 40] = [run(101)]
        self.logs[101] = log("/bin/bash: line 1: pwsh: command not found")
        self.assertEqual(self.classify(), {JOB: "signature `pwsh: command not found`"})
        self.assertIsNone(
            self.classify(dict(CFG, ci_infra_signatures=["No space left on device"]))
        )

    def test_the_same_first_error_at_develop(self):
        self.runs["h" * 40], self.runs["d" * 40] = [run(101)], [run(201)]
        self.logs[101] = self.logs[201] = log(
            "##[error]The runner could not reach its cache."
        )
        self.assertEqual(
            self.classify(),
            {
                JOB: f"develop {'d' * 10} fails the same: "
                "`The runner could not reach its cache.`"
            },
        )
        self.logs[201] = log("##[error]Another error")
        self.assertIsNone(
            self.classify()
        )  # develop fails another way, and no other PR fails the same

    def test_the_signatures_are_data_in_the_repository_defaults(self):
        """The home repository's defaults hold its public signatures, each a pattern with its scope and note; every
        regular expression compiles. The package's generic defaults hold none: without the key, nothing is known by
        signature."""
        from rkmon_lib import config

        raw = home_repository.VALUES["ci_infra_signatures"]
        self.assertEqual(len(raw), 14)
        sigs = known_infra.signatures(CFG)
        self.assertEqual(len(sigs), 14)  # none left out: every expression compiles
        self.assertTrue(all(s["note"] for s in sigs))
        self.assertEqual(config.load_defaults()["ci_infra_signatures"], [])
        self.assertEqual(config.load_defaults()["ci_infra_site_signatures"], [])
        self.assertEqual(known_infra.signatures({"repo": "o/r"}), [])
        self.runs["h" * 40] = [run(101)]
        self.logs[101] = log("##[error]The runner has received a shutdown signal.")
        self.assertEqual(
            self.classify(),
            {JOB: "signature `The runner has received a shutdown signal`"},
        )

    def test_a_signature_takes_only_the_jobs_of_its_scope_and_not_beside_a_failure(
        self,
    ):
        timeout = "##[error]The action 'Test' has timed out after 30 minutes."
        self.runs["h" * 40] = [run(101)]
        self.logs[101] = log(timeout)
        self.assertEqual(
            self.classify(),
            {JOB: "signature `The action 'Test' has timed out after 30 minutes`"},
        )
        sigs = known_infra.signatures(CFG)
        self.assertIsNone(  # a job outside the scope
            known_infra.match(sigs, "Linux::release / Build", log(timeout))
        )
        self.logs[101] = (
            "Test #41: rocke_conv_pytest ..........***Failed  3.2 sec\n" + log(timeout)
        )
        self.assertIsNone(self.classify())  # the PR's own failure beside it

    def test_plain_strings_still_work_and_a_bad_expression_is_left_out(self):
        sigs = known_infra.signatures(
            {
                "ci_infra_signatures": [
                    "No space left on device",
                    {"pattern": "([", "regex": True},
                    {"pattern": ""},
                ]
            }
        )
        self.assertEqual([s["pattern"] for s in sigs], ["No space left on device"])
        self.assertTrue(known_infra.match(sigs, "any job", "x No space left on device"))

    def test_the_site_packs_signatures_follow_and_replace_by_pattern(self):
        cfg = {
            "ci_infra_signatures": [
                {"pattern": "pwsh: command not found", "jobs": "windows"},
                "getIDToken call failed",
            ],
            "ci_infra_site_signatures": [
                {
                    "pattern": "pwsh: command not found",
                    "jobs": "windows",
                    "ends_at": "e" * 40,
                },
                {"pattern": "Site runner gone", "note": "a site's pool"},
            ],
        }
        sigs = known_infra.signatures(cfg)
        self.assertEqual(
            [(s["pattern"], s["ends_at"]) for s in sigs],
            [
                ("pwsh: command not found", "e" * 40),
                ("getIDToken call failed", ""),
                ("Site runner gone", ""),
            ],
        )

    def test_a_signature_ends_for_a_head_that_contains_its_fix(self):
        """ends_at: GitHub's compare of the fix commit with the head; "ahead" (the head contains it) ends the
        signature, "diverged" keeps it; a compare that cannot be read ends it (the job goes to verification).
        """
        cfg = dict(
            CFG,
            ci_infra_signatures=[
                {"pattern": "pwsh: command not found", "ends_at": "e" * 40}
            ],
        )
        self.runs["h" * 40] = [run(101)]
        self.logs[101] = log("/bin/bash: line 1: pwsh: command not found")
        asked = []

        def rest(path, jq=None, **_):
            asked.append(path)
            if self.compare is None:
                raise gh.GhError("HTTP 502")
            return self.compare

        with mock.patch.object(known_infra.gh, "rest", rest):
            self.compare = "diverged"
            self.assertEqual(
                self.classify(cfg), {JOB: "signature `pwsh: command not found`"}
            )
            self.assertEqual(asked, [f"repos/o/r/compare/{'e' * 40}...{'h' * 40}"])
            self.compare = "ahead"
            self.assertIsNone(self.classify(cfg))
            self.compare = None
            self.assertIsNone(self.classify(cfg))

    def test_the_same_on_two_other_open_prs_today(self):
        first = "##[error]Timed out waiting for a runner"
        self.runs["h" * 40] = [run(101)]
        self.logs[101] = log(first)
        self.runs["1" * 40], self.logs[111] = [run(111)], log(first)
        self.assertIsNone(self.classify())  # one other PR is not enough
        self.runs["2" * 40], self.logs[112] = [
            run(112, completed="2026-11-09T23:00:00Z")
        ], log(first)
        self.assertIsNone(self.classify())  # yesterday's does not count
        self.runs["3" * 40], self.logs[113] = [run(113)], log(first)
        self.assertEqual(
            self.classify(),
            {JOB: "the same on #13, #11 today: `Timed out waiting for a runner`"},
        )

    def test_anything_unread_is_not_known(self):
        self.runs["h" * 40] = [run(101)]
        self.assertIsNone(self.classify())  # its log cannot be read
        with mock.patch.object(
            known_infra.ci_runs, "check_runs", side_effect=gh.GhError("HTTP 502")
        ):
            self.assertIsNone(self.classify())
        self.runs["h" * 40] = []
        self.assertIsNone(
            self.classify()
        )  # the job is not failing at the head any more

    def test_the_line_names_the_pr_each_job_and_why(self):
        self.assertEqual(
            known_infra.line(7, "h" * 40, {JOB: "signature `pwsh: command not found`"}),
            f"CI_INFRA #7 at hhhhhhhhhh: merge-only, and its only flag is known CI infrastructure: {JOB} "
            "(signature `pwsh: command not found`)",
        )


if __name__ == "__main__":
    unittest.main()
