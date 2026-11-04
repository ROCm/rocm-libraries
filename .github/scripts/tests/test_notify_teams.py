import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import notify_teams

MIOPEN_URL = "https://tenant.webhook.office.com/webhookb2/miopen"
HIPDNN_URL = "https://tenant.webhook.office.com/webhookb2/hipdnn"

WINDOWS_BUILD = "Windows::release / Build Windows packages"
LINUX_TEST_HIPDNN = (
    "Linux::release / Test gfx94X-dcgpu / Test hipdnn (shard 1/1) (gfx94X-dcgpu)"
)
LINUX_TEST_MIOPEN = (
    "Linux::release / Test gfx94X-dcgpu / Test miopen (shard 1/1) (gfx94X-dcgpu)"
)
LINUX_TEST_MIOPEN_XFAIL = (
    "Linux::release / Test gfx94X-dcgpu / Test miopen (xfail) (gfx94X-dcgpu)"
)


def job(job_id, name, conclusion="failure"):
    return {"id": job_id, "name": name, "conclusion": conclusion}


class FakeGitHub:
    """Serves `gh api` endpoints from in-memory jobs pages and logs."""

    def __init__(self, job_pages, logs, failing_endpoints=()):
        self.job_pages = job_pages
        self.logs = logs
        self.failing_endpoints = failing_endpoints

    def __call__(self, endpoint):
        if any(part in endpoint for part in self.failing_endpoints):
            raise subprocess.CalledProcessError(
                1, ["gh", "api", endpoint], stderr="HTTP 404: Not Found\n"
            )
        if "/jobs?" in endpoint:
            page = int(endpoint.rsplit("page=", 1)[1])
            total = sum(len(p) for p in self.job_pages)
            jobs = self.job_pages[page - 1] if page <= len(self.job_pages) else []
            return json.dumps({"total_count": total, "jobs": jobs})
        job_id = int(endpoint.split("/jobs/")[1].split("/")[0])
        return self.logs[job_id]


class SelectFailedJobsTest(unittest.TestCase):
    def test_keeps_only_failed_therock_jobs_without_xfail(self):
        jobs = [
            job(1, WINDOWS_BUILD),
            job(2, LINUX_TEST_MIOPEN),
            job(3, LINUX_TEST_MIOPEN_XFAIL),
            job(4, LINUX_TEST_HIPDNN, conclusion="success"),
            job(5, LINUX_TEST_HIPDNN, conclusion="cancelled"),
            job(6, "Multi-Arch CI Summary"),
            job(7, "rocjitsu race check (gfx94X-dcgpu)"),
        ]

        selected = notify_teams.select_failed_jobs(jobs)

        self.assertEqual([j["id"] for j in selected], [1, 2])


class ClassifyJobTest(unittest.TestCase):
    def test_test_job_routes_by_job_name(self):
        self.assertEqual(
            notify_teams.classify_job(LINUX_TEST_HIPDNN, "projects/miopen"),
            ("test", LINUX_TEST_HIPDNN, "linux"),
        )

    def test_build_job_routes_by_component_name(self):
        self.assertEqual(
            notify_teams.classify_job(WINDOWS_BUILD, "projects/miopen"),
            ("build", "projects/miopen", "windows"),
        )


class TruncateAtFirstErrorTest(unittest.TestCase):
    def test_keeps_up_to_and_including_first_error_line(self):
        log = (
            "step output\n"
            "##[error]Process completed with exit code 1.\n"
            "Post job cleanup.\n"
            "##[error]second error\n"
        )

        self.assertEqual(
            notify_teams.truncate_at_first_error(log),
            "step output\n##[error]Process completed with exit code 1.\n",
        )

    def test_error_on_last_line_without_newline(self):
        log = "step output\n##[error]boom"

        self.assertEqual(notify_teams.truncate_at_first_error(log), log)

    def test_log_without_error_is_unchanged(self):
        log = "step output\nPost job cleanup.\n"

        self.assertEqual(notify_teams.truncate_at_first_error(log), log)


class NotifyFailedJobsFromRunTest(unittest.TestCase):
    def run_notifier(self, fake_github, send_results=None, pr_title="Fix bug"):
        sent = []
        results = iter(send_results or [])

        def send_notification(self_notifier, **kwargs):
            sent.append({"webhook_url": self_notifier.webhook_url, **kwargs})
            return next(results, True)

        with mock.patch.object(notify_teams, "gh_api", fake_github), mock.patch.object(
            notify_teams.TeamsNotifier, "send_notification", send_notification
        ):
            success = notify_teams.notify_failed_jobs_from_run(
                repository="ROCm/rocm-libraries",
                run_id="100",
                run_attempt="2",
                component_name="projects/miopen",
                webhook_urls={"miopen": MIOPEN_URL, "hipdnn": HIPDNN_URL},
                webhook_url="",
                project="",
                pr_number="13678",
                pr_title=pr_title,
                run_url="https://github.com/ROCm/rocm-libraries/actions/runs/100",
                dry_run=False,
            )
        return success, sent

    def test_notifies_each_failed_job_across_pages_with_its_own_routing(self):
        fake_github = FakeGitHub(
            job_pages=[
                [job(1, WINDOWS_BUILD), job(2, "Multi-Arch CI Summary")],
                [job(3, LINUX_TEST_HIPDNN), job(4, LINUX_TEST_MIOPEN_XFAIL)],
            ],
            logs={
                1: "compiling\nfatal error: undefined reference to foo\n"
                "##[error]Process completed with exit code 1.\n",
                3: "[  FAILED  ] Suite.Case\n##[error]Process completed with exit code 1.\n"
                "Post job cleanup.\nerror: unrelated cleanup failure\n",
            },
        )

        success, sent = self.run_notifier(fake_github)

        self.assertTrue(success)
        self.assertEqual(
            [
                (s["job_name"], s["project"], s["webhook_url"], s["failure_stage"])
                for s in sent
            ],
            [
                (WINDOWS_BUILD, "miopen", MIOPEN_URL, "build"),
                (LINUX_TEST_HIPDNN, "hipdnn", HIPDNN_URL, "test"),
            ],
        )
        self.assertEqual([s["platform"] for s in sent], ["windows", "linux"])
        self.assertIn("undefined reference to foo", sent[0]["error_context"])
        self.assertIn("[  FAILED  ] Suite.Case", sent[1]["error_context"])
        self.assertNotIn("unrelated cleanup failure", sent[1]["error_context"])

    def test_pr_title_tag_overrides_job_component(self):
        fake_github = FakeGitHub(
            job_pages=[[job(3, LINUX_TEST_HIPDNN)]],
            logs={3: "##[error]boom\n"},
        )

        _, sent = self.run_notifier(fake_github, pr_title="[miopen] Fix bug")

        self.assertEqual([s["project"] for s in sent], ["miopen"])

    def test_one_failed_send_fails_the_run_but_later_jobs_are_still_sent(self):
        fake_github = FakeGitHub(
            job_pages=[[job(1, WINDOWS_BUILD), job(3, LINUX_TEST_HIPDNN)]],
            logs={1: "##[error]a\n", 3: "##[error]b\n"},
        )

        success, sent = self.run_notifier(fake_github, send_results=[False, True])

        self.assertFalse(success)
        self.assertEqual(len(sent), 2)

    def test_unavailable_job_log_still_notifies(self):
        fake_github = FakeGitHub(
            job_pages=[[job(3, LINUX_TEST_HIPDNN)]],
            logs={},
            failing_endpoints=("/jobs/3/logs",),
        )

        success, sent = self.run_notifier(fake_github)

        self.assertTrue(success)
        self.assertEqual([s["job_name"] for s in sent], [LINUX_TEST_HIPDNN])

    def test_job_listing_failure_fails_without_notifying(self):
        fake_github = FakeGitHub(
            job_pages=[], logs={}, failing_endpoints=("/attempts/2/jobs",)
        )

        success, sent = self.run_notifier(fake_github)

        self.assertFalse(success)
        self.assertEqual(sent, [])

    def test_no_failed_therock_jobs_succeeds_without_notifying(self):
        fake_github = FakeGitHub(
            job_pages=[
                [job(2, "Multi-Arch CI Summary"), job(4, LINUX_TEST_MIOPEN_XFAIL)]
            ],
            logs={},
        )

        success, sent = self.run_notifier(fake_github)

        self.assertTrue(success)
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
