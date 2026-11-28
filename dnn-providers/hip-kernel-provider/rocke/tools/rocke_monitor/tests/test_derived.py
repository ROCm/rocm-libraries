# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The derived keys (rkmon_lib/derived.py; one test per function): each key's value from the keys before it
and the layout, the given value of a configuration file kept where the key has one.
"""

import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    bindings,
    config,
    container_layout,
    deploy,
    derived,
    paths,
    registry,
    secrets_registry,
)
import home_repository  # noqa: E402


def run(
    given=None, container=False, image_venv="", image_llvm_bin="", **values
) -> dict:
    """Every derived key's value, in FUNCTIONS' order, over VALUES (dotted keys: the keys before them)."""
    layout = derived.Layout(
        container=container,
        given=dict(given or {}),
        image_venv=image_venv,
        image_llvm_bin=image_llvm_bin,
    )
    out = {k.replace("__", "."): v for k, v in values.items()}
    for key, f in derived.FUNCTIONS.items():
        out[key] = f(out, layout)
    return out


class TestPaths(unittest.TestCase):
    def test_the_ramdisk_path_is_the_given_one_else_the_container_s(self):
        self.assertEqual(
            run({"ramdisk.path": "/mnt/fast"})["ramdisk.path"], "/mnt/fast"
        )
        self.assertEqual(run()["ramdisk.path"], container_layout.RAMDISK)

    def test_the_working_folder_is_the_given_one_its_older_name_s_or_the_container_s(
        self,
    ):
        self.assertEqual(run({"paths.working_dir": "/w"})["paths.working_dir"], "/w")
        self.assertEqual(run({"paths.root": "/old"})["paths.working_dir"], "/old")
        self.assertEqual(
            run(container=True)["paths.working_dir"], container_layout.WORK_DIR
        )
        self.assertEqual(run()["paths.working_dir"], "")

    def test_the_claude_workspace_is_the_given_one_else_the_container_s(self):
        self.assertEqual(run({"claude_workspace": "/cw"})["claude_workspace"], "/cw")
        self.assertEqual(
            run(container=True)["claude_workspace"],
            f"{container_layout.WORK_DIR}/claude-workspace",
        )
        self.assertEqual(run({"paths.working_dir": "/w"})["claude_workspace"], "")

    def test_every_path_is_paths_resolve_s_over_the_given_paths(self):
        given = {
            "paths.working_dir": "/w",
            "paths.repo": "/src",
            "weekly_report.dir": "/weekly",
        }
        got = run(given, **{"ramdisk.enabled": True})
        want = paths.resolve(
            {
                "paths": {"working_dir": "/w", "repo": "/src"},
                "ramdisk": {"path": container_layout.RAMDISK, "enabled": True},
                "weekly_report": {"dir": "/weekly"},
                "claude_workspace": "",
            }
        )
        for name in derived.PATHS:
            with self.subTest(path=name):
                self.assertEqual(got[f"paths.{name}"], want[name])
        self.assertEqual(got["paths.repo"], "/src")
        self.assertEqual(got["paths.ramdisk"], container_layout.RAMDISK)

    def test_in_the_container_the_secrets_and_the_venv_are_the_container_s(self):
        got = run(container=True, image_venv="/opt/venv")
        self.assertEqual(got["paths.secrets_dir"], container_layout.SECRETS_DIR)
        self.assertEqual(got["paths.venv"], "/opt/venv")
        self.assertEqual(got["paths.toolkit"], container_layout.MONITOR_HOME)
        given = run(
            {"paths.secrets_dir": "/s", "paths.venv": "/v"},
            container=True,
            image_venv="/opt/venv",
        )
        self.assertEqual(
            (given["paths.secrets_dir"], given["paths.venv"]), ("/s", "/v")
        )

    def test_in_the_container_rocm_s_llvm_tools_are_the_image_s(self):
        """paths.rocm_llvm_bin: in the container, the folder the image's ROCm record names (Layout.image_llvm_bin), over
        a given one, as the image's ROCm is the only one there (a given venv is kept); the given one where the image
        records none, else blank. On a host, the given one, else blank, whatever the image records.
        """
        llvm = "/opt/rocm-9.9.0/lib/llvm/bin"
        key = "paths.rocm_llvm_bin"
        self.assertEqual(run(container=True, image_llvm_bin=llvm)[key], llvm)
        self.assertEqual(
            run({key: "/l"}, container=True, image_llvm_bin=llvm)[key], llvm
        )
        self.assertEqual(run({key: "/l"}, container=True)[key], "/l")
        self.assertEqual(run(container=True)[key], "")
        host = {"paths.working_dir": "/w"}
        self.assertEqual(run(host, image_llvm_bin=llvm)[key], "")
        self.assertEqual(run({**host, key: "/l"}, image_llvm_bin=llvm)[key], "/l")

    def test_the_scans_folder_is_the_given_one_else_the_follow_ups_folder_in_the_monitor_s_state(
        self,
    ):
        got = run({"paths.working_dir": "/w"})
        self.assertEqual(
            got["followup_scan.dir"],
            str(pathlib.Path(got["paths.toolkit"]) / "state" / config.FOLLOWUPS),
        )
        self.assertEqual(
            run(container=True)["followup_scan.dir"],
            f"{container_layout.MONITOR_HOME}/state/followups",
        )
        self.assertEqual(
            run({"paths.working_dir": "/w", "followup_scan.dir": "~/s"})[
                "followup_scan.dir"
            ],
            "~/s",
        )
        self.assertEqual(run()["followup_scan.dir"], "")

    def test_the_site_lessons_are_a_file_in_the_monitor_s_state(self):
        """The lessons file is found through the registry: paths.site_lessons, the site lessons in the monitor's
        state folder (config.state_dir of paths.toolkit, the monitor's home), which an update keeps.
        """
        got = run({"paths.working_dir": "/w"})
        self.assertEqual(
            got["paths.site_lessons"],
            str(config.state_dir(got["paths.toolkit"]) / paths.SITE_LESSONS),
        )
        self.assertEqual(got["paths.site_lessons"], "/w/monitor/state/site/lessons.md")
        self.assertEqual(
            run(container=True)["paths.site_lessons"],
            f"{container_layout.MONITOR_HOME}/state/site/lessons.md",
        )
        self.assertEqual(run()["paths.site_lessons"], "")

    def test_the_weekly_reports_folder_is_the_resolved_one(self):
        self.assertEqual(
            run({"paths.working_dir": "/w"})["weekly_report.dir"], "/w/rocke_weekly"
        )
        self.assertEqual(
            run({"paths.working_dir": "/w", "weekly_report.dir": "/r"})[
                "weekly_report.dir"
            ],
            "/r",
        )


class TestNamesAndPorts(unittest.TestCase):
    def test_the_code_owners_organization_is_the_given_one_else_the_repository_s_owner(
        self,
    ):
        self.assertEqual(run(repo="org/repo")["codeowner_org"], "org")
        self.assertEqual(
            run({"codeowner_org": "other"}, repo="org/repo")["codeowner_org"], "other"
        )
        self.assertEqual(run(repo="")["codeowner_org"], "")

    def test_the_clone_address_is_the_given_one_else_github_s(self):
        self.assertEqual(
            run(repo="org/repo")["clone_url"], f"{derived.GITHUB}/org/repo"
        )
        self.assertEqual(
            run({"clone_url": " file:///mirror "}, repo="org/repo")["clone_url"],
            "file:///mirror",
        )
        self.assertEqual(run(repo="")["clone_url"], "")
        # the monitor's clone takes the rule from here
        cfg = {"repo": "org/repo", "clone_url": ""}
        self.assertEqual(deploy.clone_url(cfg), f"{derived.GITHUB}/org/repo")
        self.assertEqual(deploy.clone_url(dict(cfg, clone_url="/m")), "/m")

    def test_the_drafters_profile_is_the_given_folder_else_the_repository_defaults_folder(
        self,
    ):
        home = {
            "repo": home_repository.REPOSITORY,
            "rocke_path": home_repository.FOLDER,
        }
        folder = home_repository.FOUND.path.parent
        self.assertEqual(run(**home)["profile_dir"], str(folder))
        self.assertTrue((folder / "drafter.json").is_file())
        self.assertEqual(run({"profile_dir": "~/p"}, **home)["profile_dir"], "~/p")
        # a repository, or a folder, without repository defaults: the generic profile
        self.assertEqual(
            run(repo="o/r", rocke_path=home_repository.FOLDER)["profile_dir"], ""
        )
        self.assertEqual(
            run(repo=home_repository.REPOSITORY, rocke_path="f")["profile_dir"], ""
        )

    def test_the_local_worker_s_tag_is_the_given_one_else_its_model_s_leading_letters(
        self,
    ):
        self.assertEqual(
            run(**{"worker__model": "Qwen3-coder:30b"})["worker.tag"], "qwen"
        )
        self.assertEqual(
            run({"worker.tag": "q"}, **{"worker__model": "qwen3"})["worker.tag"], "q"
        )
        self.assertEqual(run()["worker.tag"], "worker")

    def test_the_draft_stage_is_none_in_no_draft_mode_else_the_given_one_else_by_the_reviewer(
        self,
    ):
        bound = {
            "reviewer": "cand",
            "model_bindings": {
                "cand": {
                    "provider": "openai",
                    "url": "https://llm.example.com",
                    "model": "m",
                }
            },
        }
        for given, values, want in (
            (None, {"reviewer": "default"}, "none"),
            ({"draft_stage": "tool-loop"}, {"reviewer": "default"}, "none"),
            (None, {"reviewer": "worker"}, "local-worker"),
            (None, bound, "tool-loop"),
            ({"draft_stage": "auto"}, bound, "tool-loop"),
            ({"draft_stage": "local-worker"}, bound, "local-worker"),
            ({"draft_stage": "none"}, bound, "none"),
        ):
            with self.subTest(given=given, reviewer=values["reviewer"]):
                self.assertEqual(run(given, **values)["draft_stage"], want)

    def test_the_verifier_s_name_is_the_given_one_else_the_audit_model_s_and_its_line_1_follows_it(
        self,
    ):
        models = {"models__audit": "claude-opus-5-5", "effort__verify": "high"}
        got = run(**models)
        self.assertEqual(got["verifier_name"], "Claude Opus 5.5")
        self.assertEqual(
            got["review_first_line_no_draft"],
            "Automated review by rocke-monitor: Model - Claude Opus 5.5, effort: high",
        )
        got = run({"verifier_name": "V"}, **models)
        self.assertEqual(
            (got["verifier_name"], got["review_first_line_no_draft"]),
            ("V", "Automated review by rocke-monitor: Model - V, effort: high"),
        )
        # A given line 1 is not kept: it would give no automated reviewer name, or another model's
        line = "Automated review: Model - W, effort: max"
        self.assertEqual(
            run({"review_first_line_no_draft": line}, **models)[
                "review_first_line_no_draft"
            ],
            "Automated review by rocke-monitor: Model - Claude Opus 5.5, effort: high",
        )
        named = {
            "features__review_automation__automated_reviewer_name": "Platform team",
            "deployment__project": "rkmon-two",
        }
        self.assertEqual(
            run(**models, **named)["review_first_line_no_draft"],
            "Automated review by Platform team: Model - Claude Opus 5.5, effort: high",
        )
        named["features__review_automation__automated_reviewer_name"] = ""
        self.assertEqual(
            run(**models, **named)["review_first_line_no_draft"],
            "Automated review by rkmon-two: Model - Claude Opus 5.5, effort: high",
        )

    def test_the_drafting_model_is_the_given_choice_a_blank_one_too(self):
        self.assertEqual(run()["models.draft"], bindings.WORKER)
        self.assertEqual(run({"models.draft": ""})["models.draft"], "")
        self.assertEqual(run({"models.draft": "claude"})["models.draft"], "claude")

    def test_the_dashboards_ports_are_the_given_ones_else_the_container_s(self):
        got = run()
        self.assertEqual(
            (got["admin_dashboard.port"], got["team_dashboard.port"]),
            (container_layout.ADMIN_PORT, container_layout.TEAM_PORT),
        )
        got = run({"admin_dashboard.port": 9000, "team_dashboard.port": 9001})
        self.assertEqual(
            (got["admin_dashboard.port"], got["team_dashboard.port"]), (9000, 9001)
        )

    def test_the_dashboards_listen_on_the_given_address_else_the_container_s_or_the_loopback(
        self,
    ):
        for key in ("admin_dashboard.bind", "team_dashboard.bind"):
            with self.subTest(key=key):
                self.assertEqual(run({key: "10.0.0.1"})[key], "10.0.0.1")
                self.assertEqual(
                    run(container=True)[key], container_layout.LISTEN_ADDRESS
                )
                self.assertEqual(run()[key], "")

    def test_a_secret_s_place_is_its_entry_s_own_and_a_given_place_is_passed_over(
        self,
    ):
        """Each key that names a secret's or a login's place is its entry's own place in its keeper's secrets folder
        (in the container, a writer's entry is in the writer's volume), by the layout alone, "" while no secrets folder
        is known; the first review pass's key file is the file of the secret worker.api_key_secret names.
        """
        host = {"paths.working_dir": "/w"}
        mine = run(host)["paths.secrets_dir"]
        self.assertTrue(mine)
        monitor, writer = (
            container_layout.SECRETS_DIR,
            container_layout.WRITE_SECRETS_DIR,
        )
        places = (
            ("gpu.ssh_key", "gpu_ssh_key", monitor),
            ("publish.html.key", "web_host_ssh_key", writer),
            ("admin_dashboard.access_token_file", "admin_access_token", monitor),
            ("teams.webhook_file", "teams_webhook", writer),
            ("gh.read_config_dir", "gh-read", monitor),
            ("gh.write_config_dir", "gh", writer),
        )
        self.assertEqual(
            {key for key, _name, _keeper in places},
            {e.key for e in secrets_registry.entries({}) if e.key},
        )
        for key, name, keeper in places:
            with self.subTest(key=key):
                given = {key: "/elsewhere"}
                self.assertEqual(run(host)[key], f"{mine}/{name}")
                self.assertEqual(run({**host, **given})[key], f"{mine}/{name}")
                self.assertEqual(run(given, container=True)[key], f"{keeper}/{name}")
                self.assertEqual(run(given)[key], "")
                with mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"}):
                    self.assertEqual(run(host)[key], f"{mine}/{name}")
        self.assertEqual(
            run(container=True)["gh.write_config_dir"], container_layout.WRITE_LOGIN_DIR
        )
        key, named = "worker.api_key_file", {"worker__api_key_secret": "glm_api_key"}
        given = {key: "/k"}
        self.assertEqual(run(host, **named)[key], f"{mine}/glm_api_key")
        self.assertEqual(run({**host, **given}, **named)[key], f"{mine}/glm_api_key")
        self.assertEqual(
            run(given, container=True, **named)[key], f"{monitor}/glm_api_key"
        )
        self.assertEqual(run({**host, **given})[key], "")

    def test_the_first_review_pass_s_key_file_is_derived_after_the_secrets_places(
        self,
    ):
        """worker.api_key_file comes after each key of SECRET_PLACES, in FUNCTIONS' order and in the load's
        (registry.keys()), so the first review pass's key file, of a fixed secret with a place, is that place.
        """
        loads = [k.name for k in registry.keys()]
        for order in (list(derived.FUNCTIONS), loads):
            self.assertGreater(
                order.index("worker.api_key_file"),
                max(order.index(k) for k in derived.SECRET_PLACES),
            )
        got = run({"paths.working_dir": "/w"}, worker__api_key_secret="gpu_ssh_key")
        self.assertTrue(got["gpu.ssh_key"])
        self.assertEqual(got["worker.api_key_file"], got["gpu.ssh_key"])

    def test_the_team_page_is_the_given_one_else_the_site_folder_s(self):
        site = {"publish.html.site": "public_html/my-team/"}
        self.assertEqual(
            run(**site)["publish.html.dest"], "public_html/my-team/index.html"
        )
        self.assertEqual(
            run({"publish.html.dest": "x/page.html"}, **site)["publish.html.dest"],
            "x/page.html",
        )
        self.assertEqual(run()["publish.html.dest"], "")

    def test_the_requests_file_lies_at_the_top_of_the_home_never_under_the_site_folder(
        self,
    ):
        """The requests file holds the requesters' names and reasons: at the top of the login's home, never under the
        site folder's first folder, which a web server may publish whole (a per-user public_html), and named after the
        site folder, so two sites never share one."""
        for site, want in (
            ("public_html/my-team", "public_html+my-team-requests.jsonl"),
            ("public_html/my-team/", "public_html+my-team-requests.jsonl"),
            ("public_html", "public_html-requests.jsonl"),
            ("site", "site-requests.jsonl"),
            (" ", ""),
            (None, ""),
        ):
            with self.subTest(site=site):
                self.assertEqual(derived.site_requests(site), want)
        # sites a join with "-", "_" or "." would give one name: each is in a plain name, as "/" would be
        sites = (
            "public_html/my-team",
            "public_html/my/team",
            "public_html/a/b-c",
            "public_html/a-b/c",
            "public_html/a/b_c",
            "public_html/a_b/c",
            "public_html/a/b.c",
            "public_html/a.b/c",
            "public_html",
            "www/my-team",
            "my-team",
        )
        files = [derived.site_requests(site) for site in sites]
        for site, file in zip(sites, files):
            with self.subTest(site=site):
                place = pathlib.PurePosixPath(file)
                self.assertEqual(place.parent, pathlib.PurePosixPath("."))
                self.assertFalse(place.is_relative_to(site.split("/")[0]))
        self.assertEqual(len(set(files)), len(sites))

    def test_line_1_is_the_drafter_s_whatever_is_given(self):
        worker = {"reviewer": bindings.WORKER, "worker.model": "m-1"}
        want = f"Automated review by rocke-monitor: Model - m-1, effort: {config.MODEL_DEFAULT}"
        self.assertEqual(run(**worker)["review_first_line"], want)
        self.assertEqual(
            run({"review_first_line": "Line"}, **worker)["review_first_line"], want
        )
        self.assertEqual(run(reviewer="default")["review_first_line"], "")


if __name__ == "__main__":
    unittest.main()
