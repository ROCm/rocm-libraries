# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/translate_production.py: production's config.json and claude_env's shape as a deploy
settings file of this package, and the report of what is not one value moved to one key.

- The synthetic fixture, tests/data/production_config.json and production_claude_env_shape.json, has production's
  shape and switch pattern and invented values. It translates with no stop to a file that passes its own
  check, and each rule is tested on a change to it.
- --inside reads config.json and claude_env in the monitor's home, as the one-off container does, and prints the
  translation: claude_env's key is never printed, and nothing in the home changes.
- --container runs the one-off container through docker: a fake docker on PATH records each command and answers it,
  so nothing calls the real docker.

`production_fixture`, `changed`, `sections` and `read_back` load the fixture, change it, split a report into its
sections and read a translated file back, for any test of the translation.
"""

import contextlib
import copy
import io
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import home_repository  # noqa: E402
import sandbox  # noqa: E402
import translate_production as tp  # noqa: E402
from test_settings_schemas import CLOSING_WORDS, REJECTED, UNWANTED  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    claude_env_shape,
    coherence,
    config,
    container_layout,
    deploy_tree,
    deployment,
    features,
    file_types,
    from_layout1,
    image,
    orchestrator,
    private_files,
    registry,
    renames,
    repository_defaults,
    secrets_registry,
    settings_file,
    steps,
)
from rkmon_lib.claude_env_shape import Header, Shape  # noqa: E402

TOOL = ROOT / "tools" / "translate_production.py"
DATA = HERE / "data"
RA, MQ, TEAM, TICKETS = tp.RA, tp.MQ, tp.TEAM, tp.TICKETS
DELETE = object()  # a change that removes its key
# config.json's nested feature map, which has no older switches: each feature on, the merge queue live.
NESTED = {
    "review_automation": {"enabled": True, "mode": "live"},
    "merge_queue": {
        "enabled": True,
        "mode": "live",
        "auto_merge": "enabled",
        "auto_hold": True,
        "develop_guard": True,
    },
    "dashboards": {"admin": {"enabled": True}, "team": {"enabled": True}},
    "alerts": {"enabled": True},
    "ticket_creation": {
        "enabled": True,
        "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
    },
}
# claude_env's key, invented: it is never printed.
KEY = "-".join(("invented", "gateway", "key", "0123456789abcdef"))


# Production's reference clang, a removed key: the gate compiler's commit is a package build default.
CLANG = (
    'paths.rocm_reference_clang "AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb...: '
    "the gate compiler's commit is a package build default, which rk doctor compares this machine's clang with"
)
# Production's test time limit of an update, a removed key: an update runs no tests.
TEST_TIMEOUT = (
    "update.test_timeout_min 45: an update runs no tests: the package build runs them"
)


def production_fixture() -> tuple:
    """The synthetic fixture: (config.json's data, claude_env's shape)."""
    cfg = json.loads((DATA / "production_config.json").read_text(encoding="utf-8"))
    raw = json.loads(
        (DATA / "production_claude_env_shape.json").read_text(encoding="utf-8")
    )
    return cfg, claude_env_shape.from_json(raw)


def setUpModule():
    # Production's repository, invented in the fixture, stands for the home repository: its repository defaults apply.
    cfg, _shape = production_fixture()
    home_repository.stand_in(cfg["repo"], cfg["rocke_path"])


def changed(cfg: dict, changes: dict) -> dict:
    """A copy of CFG with each dotted key of CHANGES set to its value; DELETE removes the key."""
    out = copy.deepcopy(cfg)
    for key, value in changes.items():
        *parents, last = key.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        if value is DELETE:
            node.pop(last, None)
        else:
            node[last] = copy.deepcopy(value)
    return out


def nested(**features) -> dict:
    """The changes that give the fixture the nested feature map, each of FEATURES in place of its feature's map."""
    return {
        "features": {**NESTED, **features},
        "deployment.phase": DELETE,
        "deployment.profile": DELETE,
        "merge_authority": DELETE,
        "gpu.enabled": False,
        "weekly_report.enabled": True,
    }


def sections(report: list) -> dict:
    """The report's items under each heading, each item's wrapped lines joined into one."""
    out, title = {}, None
    for line in report:
        if not line.startswith(" "):
            title = line
            out[title] = []
        elif line.startswith("  - "):
            out[title].append(line[4:])
        else:
            out[title][-1] += " " + line.strip()
    return out


def own_check(text: str) -> list:
    """The rows of the own check of a deploy settings file's TEXT, as the report shows them: settings_file.validate's,
    then those of the rules on the container's share of the host, with no figure of this host (coherence.resources).
    """
    doc = settings_file.loads(text, file_types.DEPLOY)
    rows = settings_file.validate(doc) + coherence.resources(doc)
    return [f"{r.status} {r.name}: {r.found}" for r in rows]


def refused(values: dict) -> list:
    """What each row that refuses the container's share of the host in the deploy settings VALUES found, with no
    figure of this host: the file's own rules on it (coherence.resources), which `rkmon check` and `rkmon confirm`
    run too."""
    doc = settings_file.Document(file_types.DEPLOY, file_types.DEPLOY.version, values)
    return [
        r.found for r in coherence.resources(doc) if r.status is not checks.Status.PASS
    ]


def read_back(text: str) -> dict:
    """The values of a deploy settings file's TEXT, as rkmon reads the file."""
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "deploy.yaml"
        p.write_text(text, encoding="utf-8")
        p.chmod(0o600)
        return settings_file.read(p, file_types.DEPLOY).values


def claude_env() -> dict:
    """A claude_env whose shape is the fixture's: the key in the bearer variable and in one custom header."""
    return {
        "ANTHROPIC_BASE_URL": "https://llm-gateway.example.com/anthropic",
        "ANTHROPIC_AUTH_TOKEN": KEY,
        "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}",
        "ANTHROPIC_MODEL": "model-one",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "model-one",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "model-two",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model-three",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


# A secret, invented, planted in production's values and keys; PLANTED_PART, a part of it that is no secret's shape on
# its own, so a secret shown in part is found too.
PLANTED_PART = "PLANTEDzz"
PLANTED = "gh" + "p_" + PLANTED_PART + "0123456789abcdefghijklmnop"


def texts(node, path=()):
    """The path of each text in NODE, config.json's data: each value and each list's item that is text, as its keys
    and list indexes."""
    if isinstance(node, str):
        yield path
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from texts(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from texts(v, (*path, i))


def planted(cfg: dict, path: tuple, value) -> dict:
    """A copy of CFG with VALUE at PATH, its keys and list indexes."""
    out = copy.deepcopy(cfg)
    node = out
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    return out


def keys(node, path=()):
    """The path of each key in NODE, config.json's data: each mapping's key, in lists too, as its keys and list
    indexes."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield (*path, k)
            yield from keys(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from keys(v, (*path, i))


def renamed(cfg: dict, path: tuple, name: str) -> dict:
    """A copy of CFG whose key at PATH, its keys and list indexes, is NAME, in the same place."""
    out = copy.deepcopy(cfg)
    node = out
    for p in path[:-1]:
        node = node[p]
    items = list(node.items())
    node.clear()
    node.update((name if k == path[-1] else k, v) for k, v in items)
    return out


class _Translation(unittest.TestCase):
    """The fixture's translation with changes."""

    def ok(self, changes=None, shape=None) -> tuple:
        """(the file's values, the report's sections) for the fixture with CHANGES, and SHAPE as claude_env's."""
        cfg, fixture_shape = production_fixture()
        done = tp.translate(changed(cfg, changes or {}), shape or fixture_shape)
        return read_back(done.deploy_file), sections(done.report)

    def stopped(self, changes=None, shape=None) -> list:
        """The stops of the fixture with CHANGES, and SHAPE as claude_env's, each (what was found, its fix as the report
        shows it): the report names each one under Stops, its fix under it."""
        cfg, fixture_shape = production_fixture()
        with self.assertRaises(tp.Untranslatable) as cm:
            tp.translate(changed(cfg, changes or {}), shape or fixture_shape)
        pairs = [(s.found, str(s.fix)) for s in cm.exception.stops]
        self.assertEqual(
            sections(cm.exception.report)[tp.STOPS],
            [f"{found} {checks.FIX_LABEL}{fix}" for found, fix in pairs],
        )
        return pairs

    def stops(self, changes=None, shape=None) -> list:
        """What each stop of the fixture with CHANGES found (`stopped`)."""
        return [found for found, _fix in self.stopped(changes, shape)]

    def fixes(self, changes=None, shape=None) -> list:
        """Each stop's fix, as the report shows it (`stopped`)."""
        return [fix for _found, fix in self.stopped(changes, shape)]

    def assertValues(self, values: dict, want: dict) -> None:
        for path, value in want.items():
            with self.subTest(path=path):
                self.assertEqual(deploy_tree.get(values, path), value)


class TestTheFixture(_Translation):
    """The fixture's translation: each section in order, no stop, each key known, a file that passes its own check
    with production's values, and production's configuration left as it was."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.shape = production_fixture()
        cls.before = copy.deepcopy(cls.cfg)
        cls.done = tp.translate(cls.cfg, cls.shape)
        cls.values = read_back(cls.done.deploy_file)
        cls.s = sections(cls.done.report)

    def test_each_section_in_order_and_each_line_in_the_width(self):
        self.assertEqual(
            [x for x in self.done.report if not x.startswith(" ")], list(tp.SECTIONS)
        )
        for line in self.done.report:
            self.assertLessEqual(len(line), checks.WIDTH, line)

    def test_no_stop_each_key_known_and_the_files_own_check_passes(self):
        self.assertEqual(self.s[tp.STOPS], ["none"])
        self.assertEqual(self.s[tp.UNKNOWN], ["none"])
        self.assertEqual(
            [x.split(":", 1)[0] for x in self.s[tp.CHECK]],
            ["PASS settings", "PASS coherence"],
        )

    def test_the_file_read_back_passes_its_check(self):
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, self.values
        )
        failed = [
            r for r in settings_file.validate(doc) if r.status == checks.Status.FAIL
        ]
        self.assertEqual(failed, [])

    def test_the_check_is_the_files_as_rkmon_check_reads_it(self):
        self.assertEqual(self.s[tp.CHECK], own_check(self.done.deploy_file))

    def test_the_configuration_is_left_as_it_was(self):
        self.assertEqual(self.cfg, self.before)

    def test_no_line_looks_like_a_secret(self):
        for line in [*self.done.report, *self.done.deploy_file.splitlines()]:
            self.assertFalse(settings_file.secret_shape(line), line)

    def test_the_files_values(self):
        opus_price = {
            "input": 5,
            "output": 25,
            "cache_read": 0.5,
            "cache_write_5_minutes": 6.25,
            "cache_write_1_hour": 10,
        }
        passes = {
            "verification_pass": {"model": "opus", "effort": "high"},
            "code_quality_pass": {"model": "opus", "effort": "medium"},
            "first_review_pass": {"model": "default", "effort": "default"},
            "ci_triage_pass": {"model": "default", "effort": "default"},
            "first_code_quality_pass": {"model": "default", "effort": "default"},
            "notice_draft_pass": {"model": "default", "effort": "default"},
        }
        self.assertValues(
            self.values,
            {
                "container.name": "rkmon",
                "image": {"registry": "", "name": "rocke-pr-monitor", "tag": "rkmon"},
                "repository.name": "example-org/example-repo",
                "repository.folder": "projects/sample-tool",
                "repository.scope_labels": ["area: tools"],
                "repository.shared_label_folders": {
                    "area: shared": "projects/sample-tool"
                },
                "host.resources": {"cpus": 8, "memory_gib": 32, "ramdisk_gib": 12},
                "host.users": {"monitor_uid": None, "monitor_gid": None},
                "monitor": {
                    "tick_interval_minutes": 10,
                    "full_tick_max_interval_minutes": 60,
                    "parallel_builds": 2,
                },
                f"{RA}.enabled": True,
                f"{RA}.mode": "live",
                f"{RA}.orchestrator": {"model": "opus", "effort": "high"},
                f"{RA}.passes": passes,
                f"{RA}.remote_gpu_tests.enabled": False,
                f"{RA}.spend.enabled": True,
                f"{RA}.spend.usd_per_million_tokens.claude-opus-5-5": opus_price,
                f"{MQ}.enabled": True,
                f"{MQ}.mode": "live",
                f"{MQ}.auto_merge.enabled": True,
                f"{MQ}.auto_merge_guard.on_your_pull_requests": "alert_only",
                f"{MQ}.auto_hold.enabled": True,
                f"{MQ}.queue_notices.enabled": True,
                f"{MQ}.drift_notices.enabled": True,
                f"{MQ}.develop_guard.enabled": True,
                f"{MQ}.develop_guard.ci_red_label": "post-merge",
                f"{MQ}.develop_guard.ci_red_extra_labels": ["ci-red"],
                f"{MQ}.develop_guard.stability_label": "rocke-stability",
                "features.dashboards.admin": {"enabled": True, "port": 18614},
                f"{TEAM}.enabled": True,
                f"{TEAM}.mode": "live",
                f"{TEAM}.public_url": "http://web.example.com:8080",
                f"{TEAM}.priority_requests.enabled": True,
                f"{TEAM}.delivery": "publish",
                f"{TEAM}.publish": {
                    "web_host": "web.example.com",
                    "username": "publisher",
                    "site_folder": "public_html/dash",
                },
                "features.alerts": {
                    "mode": "live",
                    "message_label": "Monitor alerts",
                    "message_cooldown_minutes": 30,
                    "develop_breaks": {"enabled": True},
                    "needs_you": {"enabled": False},
                },
                f"{TICKETS}.mode": "live",
                f"{TICKETS}.project_key": "SAMPLE",
                f"{TICKETS}.components": ["Component A", "Component B"],
                f"{TICKETS}.labels": ["label-a", "label-b"],
                f"{TICKETS}.follow_up_tickets.enabled": True,
                f"{TICKETS}.found_issue_tickets.enabled": True,
                "features.weekly_report.enabled": True,
                "integrations.github_cli.username": "op-login",
                "integrations.atlassian_cli.jira_url": "https://jira.example.com",
                "integrations.claude_code": {
                    "base_url": "https://llm-gateway.example.com/anthropic",
                    "api_key_sent_as": "header_and_bearer",
                    "api_key_header": "X-Gateway-Key",
                    "api_key_scheme": "",
                },
                "model_bindings": {},
            },
        )

    def test_what_was_read_the_switches_and_the_forms(self):
        self.assertEqual(
            self.s[tp.READ],
            [
                "config.json: 319 keys; claude_env's shape: 8 variables and 1 custom header (names and settings only)",
                "teams.needs_you_every_min is now teams.message_cooldown_min: moved",
            ],
        )
        older = '(`deployment.phase "queue"`, `deployment.profile ""`, `merge_authority "none"`'
        self.assertEqual(
            self.s[tp.SWITCHES],
            [
                f'{RA}: enabled true, mode "live" {older}, `review-posting true`)',
                "`headless-claude true`: no key; the Claude session runs while review automation is on, as today",
                "`overlap-management true`: no key; overlap notices go out while review automation is live, as today",
                f'{MQ}: enabled true, mode "live"; auto_merge.enabled true, auto_hold.enabled true {older}, '
                "`auto-merge true`, `holds true`)",
                '`auto-merge-guard "act"`: no key; the auto-merge guard runs with the merge queue: it alerts in '
                "prototype and does its GitHub work in live, as today",
                f"{MQ}.queue_notices.enabled true (`queue-notices false`): the merge queue's switch, as today; "
                "today's monitor does not read `queue-notices` either",
                f"{MQ}.drift_notices.enabled true (`drift-notices true`)",
                f"{MQ}.develop_guard.enabled true (`incident-issues true`, `stability-issues true`, `fix-links true`)",
                "features.dashboards.admin.enabled true",
                f'{TEAM}: enabled true, mode "live", delivery "publish" (`team-dashboard true`); it is pushed today '
                "whenever it is on",
                'features.alerts: develop_breaks.enabled true, needs_you.enabled false, mode "live" '
                "(`chat-alerts true`, `needs-you-alerts false`); today's monitor sends both while either is on; this "
                "file sends only the develop-break alerts",
                f'{TICKETS}: follow_up_tickets.enabled true, found_issue_tickets.enabled true, mode "live" '
                "(`follow-up-tickets true`, `found-issue-tickets true`); tickets are filed today whenever a type is "
                "on",
                f"{RA}.remote_gpu_tests.enabled false (`remote-gpu-tests false`)",
                "features.weekly_report.enabled true (`weekly-report true`)",
                "`services false`: no key; an install without a container runs no services",
                "`jira-mcp false`: no key; no feature uses the JIRA MCP server",
            ],
        )
        self.assertEqual(
            self.s[tp.FORMS],
            [
                'image.name "rocke-pr-monitor" and image.tag "rkmon": the image production runs, as '
                'deployment.image "" names it',
                "updates: left out (update.repo is blank and update.branch is blank)",
                'repository.shared_label_folders: scope.path_labels "area: shared", each with rocke_path\'s folder '
                '"projects/sample-tool", the one folder it counts for',
                "host.resources.ramdisk_gib 12: ramdisk.enabled true, ramdisk.size_gb 12",
                f'{MQ}.auto_merge_guard.on_your_pull_requests "alert_only": automerge_guard_own false',
                f'{MQ}.develop_guard.ci_red_label "post-merge" and ci_red_extra_labels ["ci-red"]: '
                'incident_issue.labels ["post-merge"], with "ci-red", the label the code adds to each CI-red issue',
                'publish.html.dest "public_html/dash/index.html" and publish.html.requests '
                f'"public_html+dash-requests.jsonl": no keys; each is derived from {TEAM}.publish.site_folder, and '
                f"{TEAM}.priority_requests.enabled is true",
                'team_dashboard.publish_host "127.0.0.1" and team_dashboard.public_port 0: not carried; delivery '
                '"publish" takes no serve block, and today\'s monitor reads them only while it serves the page',
                f'{TICKETS}.components ["Component A", "Component B"]: jira.component "Component A, Component B", '
                "split at commas",
                "integrations.atlassian_cli.jira_url: atlassian.site and jira.url, which agree",
                f"{RA}.spend.enabled true: production has prices for 3 models; each price is the same, its fields "
                "renamed (in to input, out to output, cache_write_5m to cache_write_5_minutes, cache_write_1h to "
                "cache_write_1_hour)",
                f"{RA}.remote_gpu_tests.node.environment: ROCKE_BACKEND left out, the repository's default, which "
                "each job sets first",
                "tick_interval_min 10: no key; one value with monitor.tick_interval_minutes, which is 10",
                "loop_log_max_mb 20: no key; one value with the package's retention.log_max_mb, which is 20",
            ],
        )

    def test_the_passes(self):
        self.assertEqual(
            self.s[tp.EFFORTS],
            [
                f'{RA}.orchestrator.effort "high" (claude.effort)',
                f'{RA}.passes.verification_pass.effort "high" (claude.effort)',
                f'{RA}.passes.code_quality_pass.effort "medium" (effort.quality)',
                f'{RA}.passes.first_review_pass.effort "default"',
                f'{RA}.passes.ci_triage_pass.effort "default" (effort.ci_triage "default")',
                f'{RA}.passes.first_code_quality_pass.effort "default" (effort.quality_draft "default")',
                f'{RA}.passes.notice_draft_pass.effort "default" (effort.notice_draft "default")',
            ],
        )
        self.assertEqual(
            self.s[tp.FIRST],
            [
                f'{RA}.passes.first_review_pass.model "default": no draft; the verification pass reviews alone, as '
                "today",
                'the local worker\'s settings, not carried: no pass runs on it. worker.backend "openai", worker.url '
                '"https://llm-gateway.example.com/drafter/v1", worker.model "drafter-model-a", worker.stall_s 900, '
                'worker.max_parallel 4, worker.rpm 60, worker.tpm 400000, worker.think "max"; which provider '
                "worker.backend stands for is unverified",
                'the headers its requests send: "X-Api-Key": "<key>", "Authorization": "Bearer <key>" '
                '(worker.auth_header "X-Api-Key", worker.auth_scheme "", worker.auth_mode "both")',
                "to bind a drafting model later: add a binding to model_bindings, name it in "
                f"{RA}.passes.first_review_pass.model, and install its key with rkmon secret",
            ],
        )
        self.assertEqual(
            self.s[tp.MODELS],
            [
                f'{RA}.orchestrator.model "opus"',
                f'{RA}.passes.verification_pass.model "opus"',
                f'{RA}.passes.code_quality_pass.model "opus"',
            ],
        )

    def test_claude_code_and_the_logins(self):
        self.assertEqual(
            self.s[tp.CLAUDE],
            [
                "claude_env sets ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_CUSTOM_HEADERS, ANTHROPIC_MODEL, "
                "ANTHROPIC_DEFAULT_OPUS_MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL, ANTHROPIC_DEFAULT_HAIKU_MODEL, "
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                "custom header X-Gateway-Key: its value is the key",
                "integrations.claude_code.base_url: ANTHROPIC_BASE_URL",
                'integrations.claude_code.api_key_sent_as "header_and_bearer", api_key_header "X-Gateway-Key", '
                'api_key_scheme ""',
                "ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_OPUS_MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL and "
                "ANTHROPIC_DEFAULT_HAIKU_MODEL: not carried; each call names its model by its built-in id, which the "
                "gateway takes",
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC "1": not carried; the container sets it to 1 for every '
                "session",
            ],
        )
        self.assertEqual(
            self.s[tp.LOGINS],
            [
                "github_read: a GitHub read token of this deployment's own: rkmon login github_read",
                "github_write: the GitHub login that writes: rkmon login github_write",
                "claude: the key in claude_env: copied once by the one-time move",
                "atlassian: the atlassian CLI's login: rkmon login atlassian, never copied",
                "admin_access_token: made on this host: rkmon secret admin_access_token --generate",
                "teams_webhook: copied once by the one-time move, to the GitHub writer's volume",
                "web_host_ssh_key: copied once by the one-time move, from the secrets folder's webhost_ed25519; the "
                "web host's own key is then recorded: rkmon login shows it, and records it once you accept it",
                "worker_api_key: not copied; no binding uses it",
            ],
        )

    def test_the_other_keys(self):
        s = self.s
        self.assertEqual(s[tp.PACKAGE], ["70 package keys at the package's own value"])
        self.assertEqual(
            s[tp.REMOVED],
            [
                CLANG,
                TEST_TIMEOUT,
                "58 other removed keys blank, off or at the package's value",
            ],
        )
        self.assertEqual(
            s[tp.STATE],
            [
                "codeowner_teams, clone_since, base_branch, worker.model_name, worker.chat_path, worker.models_url, "
                "worker.probe, ci_infra_site_signatures, publish.html.host_keys, jira.accounts"
            ],
        )
        [derived] = s[tp.DERIVED]
        self.assertTrue(derived.startswith("56: codeowner_org, clone_url, "), derived)
        self.assertEqual(
            s[tp.INTENDED],
            [
                f"container.{name}: the package's {uid} (no file outside the container's volumes is this user's, so "
                "no deployment changes it)"
                for name, uid in (("write_uid", 1999), ("pr_uid", 1998))
            ],
        )
        for line in (
            f"{RA}.large_prs.split_part_lines (config.json has no split_part_lines)",
            f"{TICKETS}.epic_key (config.json has no jira.default_epic)",
            "model_bindings (no rule sets it from production's values)",
        ):
            self.assertIn(line, s[tp.DEFAULTED])
        self.assertFalse([x for x in s[tp.DEFAULTED] if x.startswith(f"{TEAM}.")])
        self.assertEqual(
            s[tp.NEXT],
            [
                "for a second deployment on this host, beside production: set container.name and "
                "features.dashboards.admin.port in the file to values of its own, then run the next step"
            ],
        )


class TestTheSchema(unittest.TestCase):
    """The translation's lists against the deploy settings schema: each key SPECIAL names is the schema's, each key
    made of several of today's keys is in SPECIAL, the passes' keys are today's, and each new key takes its default.
    """

    @classmethod
    def setUpClass(cls):
        cls.keys = {
            path: node
            for path, node, _required in settings_file.schema_keys(
                file_types.DEPLOY.load()
            )
        }

    def test_special_names_keys_of_the_schema(self):
        self.assertEqual([p for p in tp.SPECIAL if p not in self.keys], [])

    def test_each_key_made_of_several_is_special(self):
        several = [
            p
            for p, node in self.keys.items()
            if isinstance(node.get("x-internal"), list)
        ]
        self.assertTrue(several)
        self.assertEqual([p for p in several if p not in tp.SPECIAL], [])

    def at(self, path: str, value) -> dict:
        """A configuration with VALUE at the key the x-internal of the deploy key PATH names."""
        cfg = {}
        deploy_tree.put(cfg, self.keys[path]["x-internal"], value)
        return cfg

    def test_the_passes_roles_read_the_schemas_keys(self):
        """Each pass's role to layout 1's rules (tp.ROLE) names the key its effort's x-internal names, which today's
        config.role_effort reads for that role; each always-run pass's model role names, to config.claude_model, the
        key its model's x-internal names."""
        passes = [key for key, _model in tp.CLAUDE_PASSES] + list(tp.OPTIONAL_PASSES)
        self.assertEqual(sorted(tp.ROLE), sorted(passes))
        for key, role in tp.ROLE.items():
            with self.subTest(key=key):
                self.assertEqual(
                    self.keys[f"{key}.effort"]["x-internal"], f"effort.{role}"
                )
                for effort in ("low", "max"):
                    cfg = self.at(f"{key}.effort", effort)
                    self.assertEqual(config.role_effort(cfg, role)[0], effort)
        for key, model_role in tp.CLAUDE_PASSES:
            with self.subTest(key=key):
                cfg = self.at(f"{key}.model", "model-x")
                self.assertEqual(config.claude_model(cfg, model_role), "model-x")

    def test_the_price_fields_are_the_schemas(self):
        """Each field of a price the file has comes from the field of production's its x-internal names
        (deploy_tree.entries), as layout 1's rule renames it (from_layout1.PRICE_FIELDS).
        """
        fields = deploy_tree.entries(f"{RA}.spend.usd_per_million_tokens")
        self.assertEqual(
            {internal: field for field, internal in fields.items()},
            from_layout1.PRICE_FIELDS,
        )

    def test_one_switch_for_both_alerts_and_one_section_for_both_ticket_types(self):
        """Both kinds of alert name the alerts' one switch, then each its own, and both ticket types name one switch
        of ticket creation, then each its own trigger, as the rules read them (`alerts`, `tickets`).
        """
        for section, kinds in (
            (tp.ALERTS, ("develop_breaks", "needs_you")),
            (TICKETS, ("follow_up_tickets", "found_issue_tickets")),
        ):
            with self.subTest(section=section):
                keys = [
                    self.keys[f"{section}.{k}.enabled"]["x-internal"] for k in kinds
                ]
                self.assertEqual([len(k) for k in keys], [2, 2])
                self.assertEqual(keys[0][0], keys[1][0])
                self.assertNotEqual(keys[0][1], keys[1][1])

    def test_todays_functions_read_the_schemas_keys(self):
        """The other functions of today's monitor a rule calls read the keys the x-internals of the deploy keys it
        fills name."""
        self.assertEqual(
            deployment.project(self.at("container.name", "probe")), "probe"
        )
        cfg = self.at("image.name", "registry.example.com/probe:tag-x")
        self.assertEqual(
            (deployment.image_name(cfg), deployment.image_tag(cfg)),
            ("registry.example.com/probe", "tag-x"),
        )
        cfg = {}
        switch, size = self.keys["host.resources.ramdisk_gib"]["x-internal"]
        deploy_tree.put(cfg, switch, True)
        deploy_tree.put(cfg, size, 3)
        self.assertEqual(deployment.resources(cfg)["ramdisk_gb"], 3)
        address = "https://gateway.example.com"
        cfg = self.at(f"{tp.CLAUDE_CODE}.base_url", address)
        self.assertEqual(orchestrator.gateway(cfg), address)
        cfg = self.at(f"{tp.FIRST_REVIEW_PASS}.model", "worker")
        self.assertEqual(config.first_pass_choice(cfg), "worker")

    def test_each_new_key_takes_the_packages_default(self):
        new = [
            p
            for p, node in self.keys.items()
            if "x-internal" not in node
            and p not in settings_file.HEADER_KEYS
            and p not in tp.SPECIAL
        ]
        self.assertTrue(new)
        cfg, shape = production_fixture()
        defaulted = sections(tp.translate(cfg, shape).report)[tp.DEFAULTED]
        self.assertEqual([p for p in new if f"{p} (a new key)" not in defaulted], [])


class TestTheRulesKeys(unittest.TestCase):
    """Each rule reads production's keys through the x-internals of the deploy keys it fills (_Translator.internal),
    never by a name of its own: a configuration key renamed in the schema's x-internals and in production's
    configuration, as a later layout of the internal keys renames one, is read at its new name.
    """

    def test_a_key_renamed_is_read_at_its_new_name(self):
        cfg, shape = production_fixture()
        schema = list(settings_file.schema_keys(file_types.DEPLOY.load()))
        paths = {path for path, _node, _required in schema}
        internals = {path: node.get("x-internal") for path, node, _required in schema}
        names = []
        for path in tp.SPECIAL:
            v = internals[path] or []
            names += [n for n in (v if isinstance(v, list) else [v]) if n not in names]
        self.assertEqual(len(names), 61)
        real_internal, real_moves = tp._Translator.internal, tp.renames.moves
        today, renamed_lines = real_moves(copy.deepcopy(cfg))
        moved_lines = {line for _old, _new, line in renamed_lines}
        # the keys production's fixture does not have, each planted at its new name with a value it means
        planted = {
            "effort.first_pass": "default",
            "features.merge_queue.queue_notices": True,
            "features.merge_queue.drift_notices": True,
            "claude.api_key_sent_as": "header",
            "claude.api_key_header": "",
            "claude.api_key_scheme": "",
        }
        absent = [n for n in names if deploy_tree.get(today, n) is deploy_tree.MISSING]
        self.assertEqual(absent, list(planted))
        for name in names:
            new = f"renamed.{name}"

            def internal(translator, path, name=name, new=new):
                v = real_internal(translator, path)
                if isinstance(v, list):
                    return [new if n == name else n for n in v]
                return new if v == name else v

            def moves(raw, name=name, new=new):
                out, lines = real_moves(raw)
                *parents, last = name.split(".")
                node = out
                for p in parents:
                    node = node[p]
                deploy_tree.put(out, new, node.pop(last, planted.get(name)))
                return out, lines

            with (
                self.subTest(name=name),
                mock.patch.object(tp._Translator, "internal", internal),
                mock.patch.object(tp.renames, "moves", moves),
            ):
                try:
                    report = tp.translate(cfg, shape).report
                except tp.Untranslatable as exc:
                    report = exc.report
                got = sections(report)
                unread = [x for x in got[tp.UNKNOWN] if x.split(".")[0] == "renamed"]
                self.assertEqual(unread, [])
                # a key's name in a line of the report, but in the lines of the renames: a name that is a word, or a
                # deploy key too, is no sign
                if ("." in name or "_" in name) and name not in paths:
                    old = re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-]|\.\w)")
                    lines = [x for items in got.values() for x in items]
                    named = [x for x in lines if old.search(x) and x not in moved_lines]
                    self.assertEqual(named, [])

    def test_the_keys_come_from_deploy_trees_feeds(self):
        """The translation finds the keys a rule reads (`internal`) in deploy_tree.feeds, the one walk of the schema's
        x-internals: a deploy key with one pair, its configuration key; one with several, their keys in order.
        """
        feeds = (("a.one", "x.one"), ("b.two", "x.second"), ("b.two", "x.first"))
        with mock.patch.object(tp.deploy_tree, "feeds", lambda: feeds):
            internals = tp._internals()
        self.assertEqual(
            internals, {"a.one": "x.one", "b.two": ["x.second", "x.first"]}
        )
        with mock.patch.object(tp, "_internals", lambda: internals):
            t = tp._Translator(*production_fixture())
        self.assertEqual(
            (t.internal("a.one"), t.internal("b.two")),
            ("x.one", ["x.second", "x.first"]),
        )


class TestTheTables(unittest.TestCase):
    """The translation's own tables against the package: each entry of the tables of keys by where they go
    is used, and each login and secret of the logins table is the registry's."""

    @classmethod
    def setUpClass(cls):
        cls.t = tp._Translator(*production_fixture())
        cls.t.run()

    def test_each_entry_of_the_key_tables_is_a_key_no_rule_reads(self):
        """Each entry names a key of the package's defaults (config.defaults.json), or one that the defaults have no
        more and production has, or an older setup's file, whose removed keys the one-time move drops by the same
        table (rkmon_lib/removed_keys.py), or a path that is in no configuration file (registry.DERIVED_NOT_IN_FILES);
        no rule reads it, and it is no other entry nor under one: else it is never used.
        """
        defaults = config.load_defaults()
        older = json.loads(
            (ROOT / "tests" / "data" / "older_setup_config.json").read_text(
                encoding="utf-8"
            )
        )
        entries = [
            (table, key)
            for table in (
                "STATE_KEYS",
                "DERIVED_KEYS",
                "PACKAGE_KEYS",
                "REMOVED_KEYS",
                "WHY_DROPPED",
                "INTENDED_KEYS",
            )
            for key in getattr(tp, table)
        ]
        self.assertGreater(len(entries), 150)
        self.assertEqual(
            [
                e
                for e in entries
                if deploy_tree.get(defaults, e[1]) is deploy_tree.MISSING
                and deploy_tree.get(self.t.cfg, e[1]) is deploy_tree.MISSING
                and (
                    e[0] != "REMOVED_KEYS"
                    or deploy_tree.get(older, e[1]) is deploy_tree.MISSING
                )
                and (
                    e[0] != "DERIVED_KEYS" or e[1] not in registry.DERIVED_NOT_IN_FILES
                )
            ],
            [],
        )
        self.assertEqual([e for e in entries if tp._listed(e[1], self.t.read)], [])
        self.assertEqual(
            [
                (a, b)
                for a in entries
                for b in entries
                if a != b and tp._listed(a[1], [b[1]])
            ],
            [],
        )

    def test_each_key_layout_1_drops_has_its_rules_why(self):
        """Each entry of WHY_DROPPED names a key of layout 1's files that layout 1's rule says why it drops."""
        self.assertEqual(
            [
                k
                for k, old in tp.WHY_DROPPED.items()
                if not from_layout1.why_dropped(old)
            ],
            [],
        )

    def test_each_login_is_the_registrys(self):
        """Each name of the logins table is an entry of the registry of secrets; the local worker's key is the first
        review pass's key's older name while a binding names that key, and an older package's file otherwise. That
        older name has its one home in the registry (secrets_registry.OLDER_KEY_NAMES): the translation never spells
        it.
        """
        names = [name for name, _needed, _how in self.t.login_rows()]
        self.assertEqual(len(names), 8)
        cfg = config.load_defaults()
        self.assertEqual(
            [n for n in names if secrets_registry.entry(cfg, n) is None], []
        )
        bound = dict(cfg, worker={"api_key_secret": secrets_registry.FIRST_PASS_KEY})
        self.assertEqual(
            secrets_registry.renamed(bound, tp.WORKER_KEY).name,
            secrets_registry.FIRST_PASS_KEY,
        )
        self.assertIn(tp.WORKER_KEY, secrets_registry.OLDER_SECRET_FILES)
        self.assertEqual(
            secrets_registry.OLDER_KEY_NAMES[tp.WORKER_KEY],
            secrets_registry.FIRST_PASS_KEY,
        )
        self.assertNotIn(json.dumps(tp.WORKER_KEY), TOOL.read_text(encoding="utf-8"))


class TestSwitches(_Translation):
    """The older switches, and the nested feature map before them."""

    def test_the_guard_on_your_own_pull_requests(self):
        for own, word in ((False, "alert_only"), (True, "same_as_others")):
            with self.subTest(own=own):
                values, s = self.ok({"automerge_guard_own": own})
                self.assertEqual(
                    deploy_tree.get(
                        values, f"{MQ}.auto_merge_guard.on_your_pull_requests"
                    ),
                    word,
                )
                self.assertIn(
                    f'{MQ}.auto_merge_guard.on_your_pull_requests "{word}": automerge_guard_own '
                    + ("true" if own else "false"),
                    s[tp.FORMS],
                )
        # a blank one, as today's monitor reads it (from_layout1.guard_own)
        for own, word in ((None, "null"), ("", '""')):
            with self.subTest(own=own):
                values, s = self.ok({"automerge_guard_own": own})
                self.assertEqual(
                    deploy_tree.get(
                        values, f"{MQ}.auto_merge_guard.on_your_pull_requests"
                    ),
                    "alert_only",
                )
                self.assertIn(
                    f'{MQ}.auto_merge_guard.on_your_pull_requests "alert_only": automerge_guard_own {word}, which '
                    "today's monitor reads as false",
                    s[tp.FORMS],
                )
        self.assertEqual(
            self.stopped({"automerge_guard_own": "x"}),
            [
                (
                    'automerge_guard_own is "x": true or false carries',
                    "set automerge_guard_own in config.json to true or false",
                )
            ],
        )
        _values, s = self.ok({"automerge_guard_own": DELETE})
        self.assertIn(
            f"{MQ}.auto_merge_guard.on_your_pull_requests (no rule sets it from production's values)",
            s[tp.DEFAULTED],
        )

    def test_auto_merge(self):
        values, _s = self.ok({"features.auto-merge": False})
        self.assertIs(deploy_tree.get(values, f"{MQ}.auto_merge.enabled"), False)

    def test_the_auto_merge_guard_with_the_merge_queues_mode(self):
        """The auto-merge guard runs with the merge queue: on while the queue is on, and acting while it is live. A
        guard production has that says otherwise is one today's monitor does not read either.
        """
        line = (
            "`auto-merge-guard {}`: no key; the auto-merge guard runs with the merge queue: it alerts in prototype "
            "and does its GitHub work in live, as today"
        )
        prototype = {"deployment.phase": "guard"}
        for changes, guard, same in (
            ({}, "act", True),
            ({"features.auto-merge-guard": "alert"}, "alert", False),
            (prototype, "alert", True),
            (prototype, "act", False),
            (prototype, False, False),
        ):
            with self.subTest(changes=changes, guard=guard):
                _values, s = self.ok({**changes, "features.auto-merge-guard": guard})
                want = line.format(json.dumps(guard))
                if not same:
                    want += "; today's monitor does not read `auto-merge-guard` either"
                self.assertIn(want, s[tp.SWITCHES])

    def test_the_write_login_only_while_a_feature_is_live(self):
        """The GitHub login that writes is needed while review automation or the merge queue is live."""
        for changes, modes, needed in (
            ({}, ("live", "live"), True),
            ({"deployment.phase": "review"}, ("live", "prototype"), True),
            ({"features.review-posting": False}, ("prototype", "live"), True),
            ({"deployment.phase": "observe"}, ("prototype", "prototype"), False),
        ):
            with self.subTest(changes=changes):
                values, s = self.ok(changes)
                self.assertEqual(
                    (
                        deploy_tree.get(values, f"{RA}.mode"),
                        deploy_tree.get(values, f"{MQ}.mode"),
                    ),
                    modes,
                )
                self.assertEqual(
                    "github_write: the GitHub login that writes: rkmon login github_write"
                    in s[tp.LOGINS],
                    needed,
                )

    def test_a_switch_production_has_and_its_feature_does_not_read(self):
        _values, s = self.ok({"features.headless-claude": False})
        self.assertIn(
            "`headless-claude false`: no key; the Claude session runs while review automation is on, as today; "
            "today's monitor does not read `headless-claude` either",
            s[tp.SWITCHES],
        )

    def test_no_word_of_the_word_lists_but_in_production_s_names(self):
        """The report's own words pass the word lists of the package's texts (tests/doc_words.yaml); production's older
        switches, whose names and values hold words the lists reject (holds, act, observe), are quoted as code.
        """
        ollama = {"provider": "ollama", "url": "http://127.0.0.1:11434", "model": "m"}
        for changes in (
            {},
            {"deployment.phase": "observe"},
            {"deployment.phase": "guard", "features.auto-merge-guard": False},
            {"features.stability-issues": DELETE, "features.queue-notices": DELETE},
            {"features.drift-notices": DELETE, "features.queue-notices": False},
            {"features.headless-claude": DELETE, "services.claude_timer": False},
            nested(),
            nested(ticket_creation={"enabled": True, "triggers": {}}),
            {"update.repo": "origin", "update.path": "monitor"},
            {"huge_pr_lines": "lines" + HALF},
            {"incident_issue.labels": [], "incident_issue.stability_label": " "},
            {"claude.effort": "", "effort.orchestrator": "", "automerge_guard_own": ""},
            {"models.notice_draft": "ol", "model_bindings": {"ol": ollama}},
            {"jira.url": "https://jira.example.com/", "jira.component": ["A"]},
        ):
            with self.subTest(changes=changes):
                cfg, shape = production_fixture()
                try:
                    report = tp.translate(changed(cfg, changes), shape).report
                except tp.Untranslatable as exc:
                    report = exc.report
                for title, items in sections(report).items():
                    for text in (title, *items):
                        for words in (REJECTED, UNWANTED, CLOSING_WORDS):
                            self.assertIsNone(words.search(text), text)

    def test_the_keys_of_the_services_of_an_install_without_a_container_are_removed_keys(
        self,
    ):
        """The keys of the services of an install without a container go (rkmon_lib/removed_keys.py): each one
        production has is a removed key, listed when it is set and counted when it is blank or off, and never said in a
        switch's place.
        """
        why = "an install without a container runs no services"
        _values, s = self.ok(
            {
                "features.services": DELETE,
                "features.headless-claude": DELETE,
                "services.enabled": True,
                "services.claude_timer": False,
                "wsl_distro": "Ubuntu",
            }
        )
        self.assertEqual(
            [x for x in s[tp.REMOVED] if x.startswith(("services", "wsl_distro"))],
            [f'wsl_distro "Ubuntu": {why}', f"services.enabled true: {why}"],
        )  # services.claude_timer is off: counted
        self.assertFalse(
            [x for x in s[tp.SWITCHES] if "services" in x or "headless-claude" in x]
        )

    def test_the_jira_mcp_server_s_keys_are_removed_keys(self):
        """The JIRA MCP server is an unused feature, gone with its keys (renames.DROPPED): each key production has is
        a removed key, listed when it is set and counted when it is blank or off, and production's older switch is
        said in the switches' place, as a feature gone."""
        _values, s = self.ok({"jira.mcp_enabled": True, "jira.mcp_server": "srv"})
        why = "an unused feature, the JIRA MCP server"
        self.assertEqual(
            [x for x in s[tp.REMOVED] if x.startswith("jira.")],
            [f"jira.mcp_enabled dropped: {why}", f"jira.mcp_server dropped: {why}"],
        )  # jira.mcp_url is blank: counted
        self.assertIn(
            "`jira-mcp false`: no key; no feature uses the JIRA MCP server",
            s[tp.SWITCHES],
        )

    def test_the_nested_feature_map(self):
        values, s = self.ok(nested())
        self.assertValues(
            values,
            {
                f"{RA}.enabled": True,
                f"{RA}.mode": "live",
                f"{MQ}.enabled": True,
                f"{MQ}.mode": "live",
                f"{MQ}.auto_merge.enabled": True,
                f"{MQ}.auto_hold.enabled": True,
                f"{MQ}.queue_notices.enabled": True,
                f"{MQ}.drift_notices.enabled": True,
                f"{MQ}.develop_guard.enabled": True,
                "features.dashboards.admin.enabled": True,
                f"{TEAM}.enabled": True,
                f"{TEAM}.mode": "live",
                "features.alerts.mode": "live",
                "features.alerts.develop_breaks.enabled": True,
                "features.alerts.needs_you.enabled": True,
                f"{TICKETS}.mode": "live",
                f"{TICKETS}.follow_up_tickets.enabled": True,
                f"{TICKETS}.found_issue_tickets.enabled": True,
                "features.weekly_report.enabled": True,
                f"{RA}.remote_gpu_tests.enabled": False,
            },
        )
        self.assertEqual(
            s[tp.SWITCHES],
            [
                f'{RA}: enabled true, mode "live"',
                f'{MQ}: enabled true, mode "live"; auto_merge.enabled true, auto_hold.enabled true',
                f"{MQ}.queue_notices.enabled true: the merge queue's switch",
                f"{MQ}.drift_notices.enabled true: the merge queue's switch and review automation's, as Claude "
                "writes them",
                "features.dashboards.admin.enabled true",
                f'{TEAM}: enabled true, mode "live", delivery "publish"; it is pushed today whenever it is on',
                'features.alerts: develop_breaks.enabled true, needs_you.enabled true, mode "live" '
                "(features.alerts.enabled true); they are sent today whenever they are on",
                f'{TICKETS}: follow_up_tickets.enabled true, found_issue_tickets.enabled true, mode "live"; tickets '
                "are filed today whenever a type is on",
                f"{RA}.remote_gpu_tests.enabled false",
                "features.weekly_report.enabled true",
            ],
        )
        self.assertIn(
            "config.json: 311 keys; claude_env's shape: 8 variables and 1 custom header (names and settings only)",
            s[tp.READ],
        )

    def test_the_nested_merge_queue(self):
        """With no notice switches, the notices follow the merge queue's switch, not its mode: in prototype the queue
        posts nothing in either layout, and a later move to live posts them, as today; the drift notices need review
        automation too. The auto-merge map, and a value it does not have."""
        prototype = dict(NESTED["merge_queue"], mode="prototype")
        values, s = self.ok(nested(merge_queue=prototype))
        self.assertEqual(deploy_tree.get(values, f"{MQ}.mode"), "prototype")
        self.assertIs(deploy_tree.get(values, f"{MQ}.queue_notices.enabled"), True)
        self.assertIs(deploy_tree.get(values, f"{MQ}.drift_notices.enabled"), True)
        self.assertIn(
            f"{MQ}.queue_notices.enabled true: the merge queue's switch",
            s[tp.SWITCHES],
        )
        # review automation off: the follow-up tickets, which need it, off too
        triggers = {"review_follow_up": False, "out_of_scope_issue": True}
        values, s = self.ok(
            nested(
                review_automation={"enabled": False, "mode": "live"},
                ticket_creation={"enabled": True, "triggers": triggers},
            )
        )
        self.assertIs(deploy_tree.get(values, f"{MQ}.queue_notices.enabled"), True)
        self.assertIs(deploy_tree.get(values, f"{MQ}.drift_notices.enabled"), False)
        self.assertIn(
            f"{MQ}.drift_notices.enabled false: the merge queue's switch and review automation's, as Claude writes "
            "them",
            s[tp.SWITCHES],
        )
        hints = dict(NESTED["merge_queue"], auto_merge="hints_only")
        values, _s = self.ok(nested(merge_queue=hints))
        self.assertIs(deploy_tree.get(values, f"{MQ}.auto_merge.enabled"), False)
        # today's monitor arms auto-merge on "enabled" alone (from_layout1.auto_merge_enabled): a switch is no map
        for odd in ("sometimes", True):
            with self.subTest(auto_merge=odd):
                self.assertEqual(
                    self.stopped(
                        nested(merge_queue=dict(NESTED["merge_queue"], auto_merge=odd))
                    ),
                    [
                        (
                            f"features.merge_queue.auto_merge is {json.dumps(odd)}: "
                            '"enabled" or "hints_only" carries',
                            'set features.merge_queue.auto_merge in config.json to "enabled" or "hints_only"',
                        )
                    ],
                )

    def test_the_queue_notices_own_switch_decides_alone(self):
        """A configuration with the queue notices' own switch (features.merge_queue.queue_notices, which the code
        reads) gives the file its value, as it is."""
        for switch in (False, True):
            with self.subTest(queue_notices=switch):
                queue = dict(NESTED["merge_queue"], queue_notices=switch)
                values, s = self.ok(nested(merge_queue=queue))
                self.assertIs(
                    deploy_tree.get(values, f"{MQ}.queue_notices.enabled"), switch
                )
                word = str(switch).lower()
                self.assertIn(
                    f"{MQ}.queue_notices.enabled {word} ({MQ}.queue_notices {word})",
                    s[tp.SWITCHES],
                )

    def test_the_nested_alerts_off(self):
        values, s = self.ok(nested(alerts={"enabled": False}))
        self.assertIs(
            deploy_tree.get(values, "features.alerts.develop_breaks.enabled"), False
        )
        self.assertIs(
            deploy_tree.get(values, "features.alerts.needs_you.enabled"), False
        )
        self.assertIn(
            "features.alerts: develop_breaks.enabled false, needs_you.enabled false (features.alerts.enabled false)",
            s[tp.SWITCHES],
        )
        # a feature that is off is written with no mode of its own, as layout 1's rule has it
        self.assertFalse(
            [x for x in s[tp.DEFAULTED] if x.startswith("features.alerts")]
        )

    def test_the_nested_alerts_on_only_while_true(self):
        """features.alerts.enabled turns the alerts on only when it is true, as today's monitor reads it."""
        for given in ("yes", "true", 1):
            with self.subTest(given=given):
                changes = nested(alerts={"enabled": given})
                today = renames.moves(changed(production_fixture()[0], changes))[0]
                self.assertIs(features.enabled(today, "alerts"), False)
                values, s = self.ok(changes)
                self.assertValues(
                    values,
                    {
                        "features.alerts.develop_breaks.enabled": False,
                        "features.alerts.needs_you.enabled": False,
                    },
                )
                self.assertIn(
                    "features.alerts: develop_breaks.enabled false, needs_you.enabled false "
                    f"(features.alerts.enabled {json.dumps(given)})",
                    s[tp.SWITCHES],
                )

    def test_the_nested_alerts_each_kind_by_its_own_switch(self):
        """In the shape after the older switches, a kind of alert is on while the one switch and its own are, as the
        code reads them: the file keeps each kind's, and a fix that turns a kind off edits its own switch alone.
        """
        alerts = {"enabled": True, "develop_breaks": True, "needs_you": False}
        changes = nested(alerts=alerts)
        moved = renames.moves(changed(production_fixture()[0], changes))[0]
        self.assertEqual(
            [features.on(moved, n) for n in ("chat-alerts", "needs-you-alerts")],
            [True, False],
        )
        values, s = self.ok(changes)
        self.assertValues(
            values,
            {
                "features.alerts.mode": "live",
                "features.alerts.develop_breaks.enabled": True,
                "features.alerts.needs_you.enabled": False,
            },
        )
        self.assertIn(
            'features.alerts: develop_breaks.enabled true, needs_you.enabled false, mode "live" '
            "(features.alerts.enabled true, features.alerts.develop_breaks true, features.alerts.needs_you false); "
            "they are sent today whenever they are on",
            s[tp.SWITCHES],
        )
        unguarded = dict(NESTED["merge_queue"], develop_guard=False)
        ((_found, fix),) = self.stopped(nested(alerts=alerts, merge_queue=unguarded))
        self.assertTrue(
            fix.startswith(
                "set features.alerts.develop_breaks in config.json to false, or turn on what it needs: "
            ),
            fix,
        )

    def test_the_alerts_two_switches_that_differ(self):
        """Production's monitor, of the earlier layout, sends both kinds of alert while either switch is on; the file
        keeps production's two, the package sends each kind while its own is on (renames.moves makes them from
        production's), and the report says what changes.
        """
        for breaks, needs_you, sent in (
            ("true", "false", "the develop-break alerts"),
            ("false", "true", "the needs-you alerts"),
        ):
            with self.subTest(sent=sent):
                flags = {
                    "features.chat-alerts": breaks == "true",
                    "features.needs-you-alerts": needs_you == "true",
                }
                moved = renames.moves(changed(production_fixture()[0], flags))[0]
                self.assertIs(features.on(moved, "chat-alerts"), breaks == "true")
                self.assertIs(
                    features.on(moved, "needs-you-alerts"), needs_you == "true"
                )
                values, s = self.ok(flags)
                self.assertValues(
                    values,
                    {
                        "features.alerts.mode": "live",
                        "features.alerts.develop_breaks.enabled": breaks == "true",
                        "features.alerts.needs_you.enabled": needs_you == "true",
                    },
                )
                self.assertIn(
                    f"features.alerts: develop_breaks.enabled {breaks}, needs_you.enabled {needs_you}, mode "
                    f'"live" (`chat-alerts {breaks}`, `needs-you-alerts {needs_you}`); today\'s monitor sends both '
                    f"while either is on; this file sends only {sent}",
                    s[tp.SWITCHES],
                )

    def test_the_queue_notices_stay_on_whatever_the_feature_map_says(self):
        """Production's queue-notices decides nothing, as today's monitor does not read it: the queue notices go with
        the merge queue's switch, so production's stay on although its queue-notices is false.
        """
        for flag, tail in (
            (False, "; today's monitor does not read `queue-notices` either"),
            (True, ""),
        ):
            with self.subTest(flag=flag):
                values, s = self.ok({"features.queue-notices": flag})
                self.assertIs(
                    deploy_tree.get(values, f"{MQ}.queue_notices.enabled"), True
                )
                self.assertIn(
                    f"{MQ}.queue_notices.enabled true (`queue-notices {str(flag).lower()}`): the merge queue's "
                    f"switch, as today{tail}",
                    s[tp.SWITCHES],
                )

    def test_a_notice_switch_the_feature_map_lacks(self):
        """A notice switch production's feature map does not have is as today's monitor posts them: the queue notices
        with the merge queue's switch, the drift notices with review automation's too.
        """
        for name, path, rule in (
            ("queue-notices", "queue_notices", "the merge queue's switch"),
            (
                "drift-notices",
                "drift_notices",
                "the merge queue's switch and review automation's, as Claude writes them",
            ),
        ):
            with self.subTest(name=name):
                values, s = self.ok({f"features.{name}": DELETE})
                self.assertIs(deploy_tree.get(values, f"{MQ}.{path}.enabled"), True)
                self.assertIn(
                    f"{MQ}.{path}.enabled true (`{name}` absent): {rule}",
                    s[tp.SWITCHES],
                )

    def test_the_develop_guards_three_switches(self):
        """The three switches off turn the develop guard off, which the develop breaks' alert needs (check 2):
        with chat-alerts on, the file's own check stops the translation, and its fix names both ways out in
        production's keys: the break alerts off, or the guard on. Three that differ stop it too, as they move together.
        """
        trio = {
            f"features.{n}": False
            for n in ("incident-issues", "stability-issues", "fix-links")
        }
        values, s = self.ok({**trio, "features.chat-alerts": False})
        self.assertIs(deploy_tree.get(values, f"{MQ}.develop_guard.enabled"), False)
        self.assertIn(
            f"{MQ}.develop_guard.enabled false (`incident-issues false`, `stability-issues false`, "
            "`fix-links false`); today's monitor does not read them: its develop guard is true, the merge queue's "
            "switch",
            s[tp.SWITCHES],
        )
        guard_on = (
            "set features.incident-issues in config.json to true, set features.stability-issues in config.json to "
            "true, and set features.fix-links in config.json to true (the develop guard: it files the develop issues "
            "and the fix links)"
        )
        self.assertEqual(
            self.stopped(trio),
            [
                (
                    "develop breaks: features.alerts.develop_breaks is on, and it needs "
                    f"{MQ}.develop_guard, which is off",
                    f"set features.chat-alerts in config.json to false, or turn on what it needs: {guard_on}",
                )
            ],
        )
        self.assertEqual(
            self.stopped({"features.stability-issues": False}),
            [
                (
                    "`incident-issues true`, `stability-issues false`, `fix-links true`: the three move together, as "
                    "the develop guard's one switch, and they differ",
                    f"{guard_on}, or set each of them to false",
                )
            ],
        )
        self.assertEqual(
            self.stops({"features.stability-issues": DELETE}),
            [
                "`incident-issues true`, `stability-issues` absent (off), `fix-links true`: the three move "
                "together, as the develop guard's one switch, and they differ"
            ],
        )

    def test_a_nested_switch_fixes_its_own_key(self):
        """With the nested feature map, a row of the file's own check names production's own switch in its fix."""
        self.assertEqual(
            self.fixes(
                nested(merge_queue=dict(NESTED["merge_queue"], develop_guard=False))
            ),
            [
                "set features.alerts.enabled in config.json to false, or turn on what it needs: set "
                "features.merge_queue.develop_guard in config.json to true (the develop guard: it files the develop "
                "issues and the fix links)"
            ],
        )

    def test_each_ticket_type(self):
        values, _s = self.ok({"features.follow-up-tickets": False})
        self.assertValues(
            values,
            {
                f"{TICKETS}.mode": "live",
                f"{TICKETS}.follow_up_tickets.enabled": False,
                f"{TICKETS}.found_issue_tickets.enabled": True,
            },
        )
        values, s = self.ok(
            {"features.follow-up-tickets": False, "features.found-issue-tickets": False}
        )
        self.assertValues(
            values,
            {
                f"{TICKETS}.follow_up_tickets.enabled": False,
                f"{TICKETS}.found_issue_tickets.enabled": False,
            },
        )
        self.assertIn(
            f"{TICKETS}: follow_up_tickets.enabled false, found_issue_tickets.enabled false "
            "(`follow-up-tickets false`, `found-issue-tickets false`)",
            s[tp.SWITCHES],
        )
        self.assertFalse([x for x in s[tp.LOGINS] if x.startswith("atlassian:")])
        # a feature that is off is written with no mode of its own, as layout 1's rule has it
        self.assertFalse(
            [x for x in s[tp.DEFAULTED] if x.startswith(f"{TICKETS}.mode")]
        )

    def test_a_ticket_trigger_the_feature_map_lacks(self):
        """A trigger the nested feature map does not have is on while ticket creation is, as today's monitor reads it
        (layout 1's package read it in features.switch, from its deploy schema's default, which
        from_layout1.TRIGGER_DEFAULT keeps; this package's features take the deploy settings file's, where each type of
        ticket is off until set); the report names it.
        """
        trigger = f"{TICKETS}.triggers.out_of_scope_issue"
        changes = nested(
            ticket_creation={"enabled": True, "triggers": {"review_follow_up": True}}
        )
        today = renames.moves(changed(production_fixture()[0], changes))[0]
        self.assertIs(deploy_tree.get(today, trigger), deploy_tree.MISSING)
        self.assertIs(from_layout1.TRIGGER_DEFAULT, True)
        values, s = self.ok(changes)
        self.assertIs(
            deploy_tree.get(values, f"{TICKETS}.found_issue_tickets.enabled"), True
        )
        self.assertIn(
            f'{TICKETS}: follow_up_tickets.enabled true, found_issue_tickets.enabled true, mode "live"; {trigger} '
            "is not set, which today's monitor reads as on; tickets are filed today whenever a type is on",
            s[tp.SWITCHES],
        )
        # ticket creation off: no type is on, and nothing to say of the trigger
        values, s = self.ok(nested(ticket_creation={"enabled": False, "triggers": {}}))
        self.assertIn(
            f"{TICKETS}: follow_up_tickets.enabled false, found_issue_tickets.enabled false",
            s[tp.SWITCHES],
        )

    def test_a_switch_the_profile_made(self):
        """The earlier layout's profile made the features' switches, which no one key of production's gives: a fix
        that turns one on keeps the file's key."""
        file = tp.THE_FILE
        self.assertEqual(
            self.stopped({"deployment.profile": "merge-queue"}),
            [
                (
                    f"follow-up tickets: {TICKETS}.follow_up_tickets is on, and it needs {RA}, which is off",
                    "set features.follow-up-tickets in config.json to false, or turn on what it needs: set "
                    f"{RA}.enabled in {file} to true",
                ),
                (
                    f"drift notices: {MQ}.drift_notices is on, and it needs {RA}, which is off",
                    "set features.drift-notices in config.json to false, or turn on what it needs: set "
                    f"{RA}.enabled in {file} to true",
                ),
            ],
        )

    def test_a_ticket_type_turned_off_by_its_trigger_alone(self):
        """In the nested feature map a ticket type is ticket creation's switch and its trigger; a fix that turns the
        type off sets the trigger alone, and leaves the other type on."""
        off = {"admin": {"enabled": False}, "team": {"enabled": False}}
        self.assertEqual(
            self.stopped(nested(dashboards=off)),
            [
                (
                    f"found-issue tickets: {TICKETS}.found_issue_tickets is on, and it needs "
                    "features.dashboards.admin, which is off",
                    f"set {TICKETS}.triggers.out_of_scope_issue in config.json to false, or turn on what it needs: "
                    "set features.dashboards.admin.enabled in config.json to true",
                )
            ],
        )
        triggers = {"review_follow_up": True, "out_of_scope_issue": False}
        values, _s = self.ok(
            nested(
                dashboards=off, ticket_creation={"enabled": True, "triggers": triggers}
            )
        )
        self.assertValues(
            values,
            {
                f"{TICKETS}.follow_up_tickets.enabled": True,
                f"{TICKETS}.found_issue_tickets.enabled": False,
            },
        )

    def test_the_alerts_the_team_dashboard_and_the_gpu_tests_need_their_secrets(self):
        values, s = self.ok({"features.chat-alerts": False})
        self.assertIs(
            deploy_tree.get(values, "features.alerts.develop_breaks.enabled"), False
        )
        self.assertFalse([x for x in s[tp.LOGINS] if x.startswith("teams_webhook:")])
        values, s = self.ok({"features.team-dashboard": False})
        self.assertEqual(deploy_tree.get(values, f"{TEAM}.mode"), "prototype")
        self.assertFalse([x for x in s[tp.LOGINS] if x.startswith("web_host_ssh_key:")])
        # the tests on need their cluster, which production's pattern leaves blank
        on = {"features.remote-gpu-tests": True}
        stopped = self.stopped(on)
        self.assertEqual(
            [found.split(":", 1)[0] for found, _fix in stopped],
            [
                f"{RA}.remote_gpu_tests.{key}"
                for key in (
                    "cluster.login_host",
                    "cluster.ssh_user",
                    "cluster.partition",
                    "cluster.accounts",
                    "cluster.gpu_types",
                    "node.python",
                )
            ],
        )
        # each fix sets the key of production's the setting comes from (its x-internal)
        self.assertEqual(
            [fix.split(" to ", 1)[0] for _found, fix in stopped],
            [
                f"set gpu.{key} in config.json"
                for key in (
                    "login_host",
                    "ssh_user",
                    "partition",
                    "accounts",
                    "gres",
                    "remote_python",
                )
            ],
        )
        self.assertEqual(
            stopped[0][1],
            "set gpu.login_host in config.json to the host name of the cluster's login node, such as "
            '"login.cluster.example.com"',
        )
        cluster = {
            "gpu.login_host": "login.cluster.example.com",
            "gpu.ssh_user": "gpu-runner",
            "gpu.partition": "batch",
            "gpu.accounts": ["team-a"],
            "gpu.gres": {"gfx942": "mi300x"},
            "gpu.remote_python": "/opt/tests/bin/python",
        }
        values, s = self.ok({**on, **cluster})
        self.assertIs(deploy_tree.get(values, f"{RA}.remote_gpu_tests.enabled"), True)
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.remote_gpu_tests.cluster.gpu_types"),
            {"gfx942": "mi300x"},
        )
        self.assertIn("gpu_ssh_key: rkmon secret gpu_ssh_key FILE", s[tp.LOGINS])


class TestValues(_Translation):
    """The values that change form: each rule on a change to the fixture, and what stops it."""

    def test_the_container_and_the_image(self):
        values, s = self.ok({"deployment.project": ""})
        self.assertEqual(deploy_tree.get(values, "container.name"), "rkmon")
        self.assertIn(
            'container.name "rkmon": what a blank deployment.project means, so the deployment keeps its name, its '
            "volumes and its state",
            s[tp.FORMS],
        )
        values, s = self.ok({"deployment.image": "rocke-monitor:v7"})
        self.assertValues(values, {"image.name": "rocke-monitor", "image.tag": "v7"})
        self.assertFalse([x for x in s[tp.FORMS] if x.startswith("image.name ")])
        values, s = self.ok({"deployment.image": DELETE})
        self.assertValues(
            values, {"image.name": "rocke-pr-monitor", "image.tag": "rkmon"}
        )
        self.assertIn(
            'image.name "rocke-pr-monitor" and image.tag "rkmon": the image production runs, as deployment.image '
            "(not set) names it",
            s[tp.FORMS],
        )
        image = "registry.example.com/team/rkmon-image:v7"
        stopped = [
            (found, fix)
            for found, fix in self.stopped({"deployment.image": image})
            if found.startswith("image.name: ")
        ]
        self.assertEqual(len(stopped), 1, stopped)
        # the key of production's that makes both image.name and image.tag: its value is no longer the file's
        self.assertTrue(
            stopped[0][1].startswith(
                "set deployment.image in config.json to a value that makes image.name a value of this form: "
            ),
            stopped,
        )

    def test_the_image_as_todays_monitor_splits_it(self):
        """image.name, image.tag and container.name name the image and the container today's monitor runs
        (deployment.image_ref, deployment.project): a blank name or tag is the image name of its package's default
        (from_layout1.IMAGE_NAME) or the container's name, and a reference with no tag names neither.
        """
        for ref in ("probe", ":tag-x", "name-x:", "", " ", "rocke-monitor:v7"):
            for project in ("project-x", ""):
                with self.subTest(ref=ref, project=project):
                    changes = {"deployment.image": ref, "deployment.project": project}
                    # a blank container name, as today's monitor reads it (from_layout1.CONTAINER_NAME)
                    named = {
                        **changes,
                        "deployment.project": project or from_layout1.CONTAINER_NAME,
                    }
                    today = renames.moves(changed(production_fixture()[0], named))[0]
                    values, _s = self.ok(changes)
                    # a blank image name, as today's monitor reads it: its package's default
                    with mock.patch.object(
                        image, "DEFAULT_NAME", from_layout1.IMAGE_NAME
                    ):
                        runs = (
                            deployment.image_name(today),
                            deployment.image_tag(today),
                            deployment.project(today),
                        )
                    self.assertEqual(
                        tuple(
                            deploy_tree.get(values, k)
                            for k in ("image.name", "image.tag", "container.name")
                        ),
                        runs,
                    )

    def test_the_users_ids(self):
        values, _s = self.ok({"container.uid": "1001", "container.gid": 1002})
        self.assertEqual(
            deploy_tree.get(values, "host.users"),
            {"monitor_uid": 1001, "monitor_gid": 1002},
        )
        # a blank id is null, digits a number (from_layout1.user_id); any other value is carried for the file's own
        # check to judge, and its row's fix sets the key of production's the id comes from
        values, _s = self.ok({"container.uid": " ", "container.gid": DELETE})
        self.assertEqual(
            deploy_tree.get(values, "host.users"),
            {"monitor_uid": None, "monitor_gid": None},
        )
        for given, found, fix in (
            (
                "abc",
                'host.users.monitor_uid: "abc" is not a whole number or null',
                "set container.uid in config.json to a number from 1000 to 2147483647",
            ),
            (
                "-1000",
                'host.users.monitor_uid: "-1000" is text, in quotes, and this key takes a whole number or null',
                "write it without quotes: set container.uid in config.json to -1000",
            ),
        ):
            with self.subTest(given=given):
                self.assertEqual(self.stopped({"container.uid": given}), [(found, fix)])
        self.assertEqual(
            self.stopped({"container.uid": 65534}),
            [
                (
                    "host.users.monitor_uid: 65534 is a system account's id (below 1000, or nobody's 65534), which "
                    "the container refuses at each start",
                    "set container.uid in config.json to an id of 1000 or more, or to null, which means your own "
                    "user id on this host",
                )
            ],
        )

    def test_the_ramdisk(self):
        values, s = self.ok({"ramdisk.enabled": False})
        self.assertEqual(deploy_tree.get(values, "host.resources.ramdisk_gib"), 0)
        self.assertIn(
            "host.resources.ramdisk_gib 0: ramdisk.enabled false, ramdisk.size_gb 12",
            s[tp.FORMS],
        )

    def test_a_memory_or_cpus_production_has_no_value_for_is_one_the_files_own_rules_take(
        self,
    ):
        """With no value of production's for the container's memory or its CPUs, the file takes the package's default
        where the file's own rules on them take it (coherence.resources, which `rkmon check` and `rkmon confirm` run
        too); else the least value they take (coherence.least), which the key's line among the keys that take the
        package's default says, with why. The CPUs count the reviews while review automation is on, as those rules
        do.
        """
        none = {"resources": DELETE}
        values, s = self.ok(
            {**none, "ramdisk.size_gb": 24, "review.pipes": 5, "checks.limit": 3}
        )
        self.assertValues(
            values, {"host.resources.memory_gib": 32, "host.resources.cpus": 8}
        )
        self.assertEqual(
            [x for x in s[tp.DEFAULTED] if x.startswith("host.resources.")],
            [
                "host.resources.cpus 8: config.json has no resources.cpus, and the package's 4 is fewer than "
                f"{RA}.parallel_reviews (5) plus monitor.parallel_builds (3), the reviews and builds that may run "
                "at once",
                "host.resources.memory_gib 32: config.json has no resources.memory_gb, and the package's 8 is less "
                "than host.resources.ramdisk_gib (24) plus the 8 GiB the work needs",
            ],
        )
        self.assertEqual(s[tp.STOPS], ["none"])
        self.assertEqual(refused(values), [])
        # the least they take: one less, and the rules refuse it
        for key, less in (
            ("host.resources.memory_gib", 31),
            ("host.resources.cpus", 7),
        ):
            with self.subTest(key=key):
                fewer = copy.deepcopy(values)
                deploy_tree.put(fewer, key, less)
                self.assertEqual([f.split(" ", 1)[0] for f in refused(fewer)], [key])
        # defaults they take stay the defaults
        values, s = self.ok(
            {**none, "ramdisk.enabled": False, "review.pipes": 2, "checks.limit": 2}
        )
        self.assertValues(
            values, {"host.resources.memory_gib": 8, "host.resources.cpus": 4}
        )
        self.assertEqual(
            [x for x in s[tp.DEFAULTED] if x.startswith("host.resources.")],
            [
                "host.resources.cpus (config.json has no resources.cpus)",
                "host.resources.memory_gib (config.json has no resources.memory_gb)",
            ],
        )
        # review automation off (and the follow-up tickets, which need it): a CPU for each build alone
        triggers = {"review_follow_up": False, "out_of_scope_issue": True}
        off = nested(
            review_automation={"enabled": False, "mode": "live"},
            ticket_creation={"enabled": True, "triggers": triggers},
        )
        values, s = self.ok({**none, **off, "review.pipes": 5, "checks.limit": 6})
        self.assertEqual(deploy_tree.get(values, "host.resources.cpus"), 6)
        self.assertIn(
            "host.resources.cpus 6: config.json has no resources.cpus, and the package's 4 is fewer than "
            "monitor.parallel_builds (6), the builds that may run at once while review automation is off",
            s[tp.DEFAULTED],
        )

    def test_the_files_own_check_runs_its_rules_on_the_containers_share_of_the_host(
        self,
    ):
        """The file's own rules on the container's share of the host, with no figure of this host
        (coherence.resources), are in its own check: production's memory with no room for its RAM disk and the work
        stops the translation, its fix in production's keys; production's CPUs, fewer than the reviews and builds that
        may run at once, are a WARN row of the report."""
        ((found, fix),) = self.stopped(
            {"resources.memory_gb": 16, "ramdisk.size_gb": 24}
        )
        self.assertTrue(
            found.startswith("memory: host.resources.memory_gib 16 is less than "),
            found,
        )
        self.assertTrue(
            fix.startswith("set resources.memory_gb in config.json to 32 or more"), fix
        )
        values, s = self.ok({"resources.cpus": 2})
        self.assertEqual(deploy_tree.get(values, "host.resources.cpus"), 2)
        self.assertEqual(
            [x.partition(":")[0] for x in s[tp.CHECK] if "host.resources.cpus 2 " in x],
            ["WARN CPUs"],
        )

    def test_the_shared_label_folders(self):
        values, _s = self.ok({"scope.path_labels": ["area: one", "area: two"]})
        self.assertEqual(
            deploy_tree.get(values, "repository.shared_label_folders"),
            {"area: one": "projects/sample-tool", "area: two": "projects/sample-tool"},
        )
        self.assertEqual(
            self.stopped({"scope.path_labels": "area: shared"}),
            [
                (
                    "scope.path_labels is no list of labels",
                    "set scope.path_labels in config.json to a list of labels",
                )
            ],
        )
        # no folder: the labels wait for it, and the file's own check stops at the folder it requires, which the file
        # written holds as the template has it
        ((found, fix),) = self.stopped({"rocke_path": DELETE})
        self.assertEqual(found, "repository.folder: still the template's <folder>")
        self.assertTrue(fix.startswith("set rocke_path in config.json to "), fix)
        # a folder not in its form: its fix is production's key; the labels' folders are made of two keys of
        # production's, neither of which gives a label's folder alone, so their fix keeps the file's key
        (folder, label) = self.stopped({"rocke_path": "/projects/sample-tool/"})
        self.assertEqual(
            [folder[0], label[0]],
            [
                'repository.folder: "/projects/sample-tool/" is not in the form this key takes',
                'repository.shared_label_folders.area: shared: "/projects/sample-tool/" is not in the form this key '
                "takes",
            ],
        )
        self.assertTrue(
            folder[1].startswith(
                "set rocke_path in config.json to a value of this form: "
            ),
            folder[1],
        )
        self.assertTrue(
            label[1].startswith(
                f"set repository.shared_label_folders.area: shared in {tp.THE_FILE} to a value of this form: "
            ),
            label[1],
        )

    def test_a_rule_never_puts_a_missing_value(self):
        """A key a rule fills holds a value: MISSING, alone or in a mapping, is the rule's fault, never a line of the
        file."""
        t = tp._Translator(*production_fixture())
        for value in (tp.MISSING, {"area: shared": tp.MISSING}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, r"\Ano value for repository\.shared_label_folders\Z"
                ):
                    t.put("repository.shared_label_folders", value)
        self.assertEqual(t.values, {})

    def test_the_develop_guards_labels(self):
        for labels, extra in (
            (["post-merge", "extra-label"], ["extra-label", "ci-red"]),
            (["post-merge", "ci-red"], ["ci-red"]),
        ):
            with self.subTest(labels=labels):
                values, _s = self.ok({"incident_issue.labels": labels})
                self.assertValues(
                    values,
                    {
                        f"{MQ}.develop_guard.ci_red_label": "post-merge",
                        f"{MQ}.develop_guard.ci_red_extra_labels": extra,
                    },
                )
        self.assertEqual(
            self.stopped({"incident_issue.labels": "post-merge"}),
            [
                (
                    "incident_issue.labels is no list of labels",
                    "set incident_issue.labels in config.json to a list of labels",
                )
            ],
        )
        # no labels: the ones today's guard puts on a develop issue (from_layout1.ci_red_labels)
        for labels, said in ((DELETE, "not set"), ([], "[]"), ([" "], '[" "]')):
            with self.subTest(labels=labels):
                values, s = self.ok({"incident_issue.labels": labels})
                self.assertValues(
                    values,
                    {
                        f"{MQ}.develop_guard.ci_red_label": "post-merge",
                        f"{MQ}.develop_guard.ci_red_extra_labels": ["ci-red"],
                    },
                )
                self.assertIn(
                    f'{MQ}.develop_guard.ci_red_label "post-merge" and ci_red_extra_labels ["ci-red"]: the labels '
                    f'today\'s guard puts on a develop issue, as incident_issue.labels is {said}, with "ci-red", the '
                    "label the code adds to each CI-red issue",
                    s[tp.FORMS],
                )
        # the stability label (from_layout1.stability_label): a blank one is the one today's guard puts
        stability = f"{MQ}.develop_guard.stability_label"
        for label, said in ((DELETE, "not set"), (None, "not set"), (" ", "blank")):
            with self.subTest(stability_label=label):
                values, s = self.ok({"incident_issue.stability_label": label})
                self.assertEqual(deploy_tree.get(values, stability), "rocke-stability")
                self.assertIn(
                    f'{stability} "rocke-stability": the label today\'s guard puts on any other develop issue, as '
                    f"incident_issue.stability_label is {said}",
                    s[tp.FORMS],
                )
        values, s = self.ok({"incident_issue.stability_label": "flaky-develop"})
        self.assertEqual(deploy_tree.get(values, stability), "flaky-develop")
        self.assertFalse([x for x in s[tp.FORMS] if x.startswith(stability)])

    def test_a_guard_that_is_off_and_keeps_its_default_labels_is_its_switch_alone(self):
        """As migrate writes it (from_layout1.guard_labels_written): the labels are written while the guard
        is in use, or when they are production's own; a guard that is off and keeps its defaults is its switch alone.
        """
        guard = f"{MQ}.develop_guard"
        off = {
            **{
                f"features.{n}": False
                for n in ("incident-issues", "stability-issues", "fix-links")
            },
            "features.chat-alerts": False,
        }
        alone = "a feature that is off and keeps its defaults is written as its switch alone"
        for labels, stability, said in (
            (
                ["post-merge"],
                "rocke-stability",
                ('["post-merge"]', '"rocke-stability"'),
            ),
            (DELETE, DELETE, ("not set", "not set")),
        ):
            with self.subTest(labels=labels, stability=stability):
                values, s = self.ok(
                    {
                        **off,
                        "incident_issue.labels": labels,
                        "incident_issue.stability_label": stability,
                    }
                )
                self.assertEqual(deploy_tree.get(values, guard), {"enabled": False})
                for line in (
                    f"{guard}.ci_red_label and ci_red_extra_labels: left out, as {guard}.enabled is false and "
                    f"incident_issue.labels is {said[0]}, which gives the labels today's guard puts by default; "
                    f"{alone}",
                    f"{guard}.stability_label: left out, as {guard}.enabled is false and "
                    f"incident_issue.stability_label is {said[1]}, which gives the label today's guard puts by "
                    f"default; {alone}",
                ):
                    self.assertIn(line, s[tp.FORMS])
                self.assertFalse([x for x in s[tp.DEFAULTED] if x.startswith(guard)])
        # labels of production's own are written, the guard's section in full with them
        values, s = self.ok(
            {
                **off,
                "incident_issue.labels": ["broken"],
                "incident_issue.stability_label": "flaky-develop",
            }
        )
        self.assertValues(
            values,
            {
                f"{guard}.enabled": False,
                f"{guard}.ci_red_label": "broken",
                f"{guard}.ci_red_extra_labels": ["ci-red"],
                f"{guard}.stability_label": "flaky-develop",
            },
        )
        self.assertFalse([x for x in s[tp.FORMS] if "left out, as" in x and guard in x])

    def test_the_admin_dashboard(self):
        values, s = self.ok(
            {
                "admin_dashboard.public_port": 0,
                "admin_dashboard.publish_host": "0.0.0.0",
            }
        )
        self.assertEqual(
            deploy_tree.get(values, "features.dashboards.admin.port"), 18614
        )
        self.assertIn(
            "features.dashboards.admin.port 18614: what admin_dashboard.public_port 0 means",
            s[tp.FORMS],
        )
        self.assertIn(
            'admin_dashboard.publish_host "0.0.0.0": no key; the admin dashboard listens on this host\'s 127.0.0.1 '
            "only, and another machine reaches it through a tunnel",
            s[tp.FORMS],
        )
        values, _s = self.ok({"admin_dashboard.public_port": 18700})
        self.assertEqual(
            deploy_tree.get(values, "features.dashboards.admin.port"), 18700
        )

    def test_the_team_dashboards_address(self):
        """The team dashboard's address is one of two keys, a trailing "/" ignored (from_layout1.same_address), and
        written without it."""
        for both in (False, True):
            changes = {"team_dashboard.url": "http://web.example.com:8080/"}
            if both:
                changes["publish.html.url"] = "http://web.example.com:8080/"
            with self.subTest(changes=changes):
                values, _s = self.ok(changes)
                self.assertEqual(
                    deploy_tree.get(values, f"{TEAM}.public_url"),
                    "http://web.example.com:8080",
                )
        self.assertEqual(
            self.stopped({"team_dashboard.url": "https://other.example.com/x"}),
            [
                (
                    "publish.html.url and team_dashboard.url name two addresses, and the team dashboard has one",
                    "set team_dashboard.url in config.json to the address publish.html.url names, or set "
                    "publish.html.url in config.json to the address team_dashboard.url names",
                )
            ],
        )

    def test_an_address_the_request_service_does_not_serve_stops(self):
        """With priority requests on and "publish", as in the fixture, the web host's request service serves the page
        over plain http, at the root of its own port: production's shape, an http address
        with a port and a folder, stops the translation, its fix the root address; an https one, its fix what to give.
        """
        for url, found, fix in (
            (
                "http://web.example.com:8080/dash/",
                'has the path "/dash", and the web host\'s request service serves the page at the root of its own '
                "port, http://web.example.com:8080/",
                '"http://web.example.com:8080/"',
            ),
            (
                "https://web.example.com/dash/",
                "is not an http:// address with a port from 1024 to 65535, such as "
                "http://dashboards.example.com:8080/: the web host's request service serves the page itself, over "
                "plain http, on a port its login may open",
                "an http address with a port of 1024 or more and no path, such as "
                '"http://dashboards.example.com:8080/"',
            ),
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    self.stopped({"publish.html.url": url}),
                    [
                        (
                            f"team dashboard: {TEAM}.public_url {found}",
                            f"set publish.html.url in config.json to {fix}",
                        )
                    ],
                )

    def test_the_pages_place_and_the_requests_files(self):
        """The page and the requests file are derived from the site folder (derived.site_page, derived.site_requests):
        production's at another place, inside the site folder, beside it or elsewhere, stops the translation, whose fix
        names the place the site folder gives; so does a blank or missing page, which today's monitor copies to the
        home folder of the web host's user, and the stop says so."""
        for dest in ("public_html/other/index.html", "public_html/dash/page.html"):
            with self.subTest(dest=dest):
                self.assertEqual(
                    self.stopped({"publish.html.dest": dest}),
                    [
                        (
                            f'publish.html.dest "{dest}" is not "public_html/dash/index.html", the page '
                            'publish.html.site "public_html/dash" gives, and the page\'s place is derived from the '
                            "site folder",
                            'set publish.html.dest in config.json to "public_html/dash/index.html"',
                        )
                    ],
                )
        for name, dest, shown in (
            ("blank", "", '""'),
            ("null", None, "(not set)"),
            ("missing", DELETE, "(not set)"),
        ):
            with self.subTest(page=name):
                self.assertEqual(
                    self.stopped({"publish.html.dest": dest}),
                    [
                        (
                            f"publish.html.dest {shown}: today's monitor copies the page to the home folder of "
                            'publish.html.user on the web host, not to "public_html/dash/index.html", the page '
                            'publish.html.site "public_html/dash" gives, and the page\'s place is derived from the '
                            "site folder",
                            'set publish.html.dest in config.json to "public_html/dash/index.html"',
                        )
                    ],
                )
        for requests in (
            "public_html/dash/requests.jsonl",
            "public_html/requests.jsonl",
            "public_html/dash-requests.jsonl",
        ):
            with self.subTest(requests=requests):
                self.assertEqual(
                    self.stopped({"publish.html.requests": requests}),
                    [
                        (
                            f'publish.html.requests "{requests}" is not "public_html+dash-requests.jsonl", the '
                            'requests file publish.html.site "public_html/dash" gives, and the requests file\'s '
                            "place is derived from the site folder",
                            'set publish.html.requests in config.json to "public_html+dash-requests.jsonl"',
                        )
                    ],
                )
        # layout 1 read a false or blank requests file as none (from_layout1.priority_requests), and a key production
        # does not have is a blank one
        for requests, shown in (("", '""'), (False, "false"), (DELETE, "(not set)")):
            with self.subTest(requests=shown):
                values, s = self.ok({"publish.html.requests": requests})
                self.assertIs(
                    deploy_tree.get(values, f"{TEAM}.priority_requests.enabled"),
                    False,
                )
                self.assertIn(
                    f'publish.html.dest "public_html/dash/index.html" and publish.html.requests {shown}: no keys; '
                    f"each is derived from {TEAM}.publish.site_folder, and {TEAM}.priority_requests.enabled is false",
                    s[tp.FORMS],
                )
        # no site folder: the file's own check, which names it
        for site in ("", DELETE):
            with self.subTest(site=site):
                self.assertEqual(
                    self.fixes({"publish.html.site": site}),
                    [
                        "set publish.html.site in config.json to the folder the web host serves the page from, "
                        'relative to that user\'s home folder, such as "public_html/my-team"'
                    ],
                )

    def test_the_team_dashboard_served(self):
        """Served here: the serve block, port 0 being the one it serves on inside, and priority requests on while the
        dashboard is on, as today; the publish block and the page's paths are not carried.
        """
        values, s = self.ok(
            {
                "team_dashboard.mode": "serve",
                "team_dashboard.public_port": 0,
                "publish.html.dest": "elsewhere/index.html",
            }
        )
        self.assertValues(
            values,
            {
                f"{TEAM}.delivery": "serve",
                f"{TEAM}.serve": {"address": "127.0.0.1", "port": 18615},
                f"{TEAM}.priority_requests.enabled": True,
            },
        )
        for line in (
            f"{TEAM}.serve.port 18615: what team_dashboard.public_port 0 means",
            f"{TEAM}.priority_requests.enabled true: the page served here takes requests whenever it is on, as "
            "today",
            'publish.html.host "web.example.com", publish.html.user "publisher", publish.html.site '
            '"public_html/dash", publish.html.dest "elsewhere/index.html" and publish.html.requests '
            '"public_html+dash-requests.jsonl": not carried; delivery "serve" takes no publish block, and today\'s '
            "monitor reads them only while it publishes the page",
        ):
            self.assertIn(line, s[tp.FORMS])
        self.assertFalse(
            [x for x in s[tp.DEFAULTED] if x.startswith(f"{TEAM}.publish.")]
        )
        self.assertFalse([x for x in s[tp.LOGINS] if x.startswith("web_host_ssh_key:")])
        values, _s = self.ok(
            {"team_dashboard.mode": "serve", "team_dashboard.public_port": 18700}
        )
        self.assertEqual(deploy_tree.get(values, f"{TEAM}.serve.port"), 18700)
        values, _s = self.ok(
            {"team_dashboard.mode": "serve", "features.team-dashboard": False}
        )
        self.assertIs(
            deploy_tree.get(values, f"{TEAM}.priority_requests.enabled"), False
        )
        self.assertEqual(
            self.stopped({"team_dashboard.mode": "push"}),
            [
                (
                    'team_dashboard.mode is "push": "publish" or "serve" carries',
                    'set team_dashboard.mode in config.json to "publish" or "serve"',
                )
            ],
        )

    def test_serves_address_and_port_only_while_it_serves(self):
        """Production's address and port go to the serve block while it serves the page. While it publishes, they are
        not carried, and the file has no serve block of production's: its defaults. A blank address, or
        none, and no port take the defaults."""
        every = {
            "team_dashboard.publish_host": container_layout.EVERY_ADDRESS,
            "team_dashboard.public_port": 18700,
        }
        values, s = self.ok({**every, "team_dashboard.mode": "serve"})
        self.assertEqual(
            deploy_tree.get(values, f"{TEAM}.serve"),
            {"address": container_layout.EVERY_ADDRESS, "port": 18700},
        )
        self.assertFalse([x for x in s[tp.FORMS] if x.startswith(f"{TEAM}.serve.")])
        values, s = self.ok(every)
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, values
        )
        self.assertEqual(
            deploy_tree.get(settings_file.effective(doc), f"{TEAM}.serve"),
            {"address": "127.0.0.1", "port": 18615},
        )
        self.assertIn(
            f'team_dashboard.publish_host "{container_layout.EVERY_ADDRESS}" and team_dashboard.public_port 18700: '
            'not carried; delivery "publish" takes no serve block, and today\'s monitor reads them only while it '
            "serves the page",
            s[tp.FORMS],
        )
        self.assertFalse([x for x in s[tp.DEFAULTED] if x.startswith(f"{TEAM}.serve.")])
        for address in ("", DELETE):
            with self.subTest(address=address):
                values, s = self.ok(
                    {
                        "team_dashboard.mode": "serve",
                        "team_dashboard.publish_host": address,
                        "team_dashboard.public_port": DELETE,
                    }
                )
                self.assertEqual(
                    deploy_tree.get(values, f"{TEAM}.serve"),
                    {"address": "127.0.0.1", "port": 18615},
                )
                for key in ("address", "port"):
                    self.assertIn(
                        f"{TEAM}.serve.{key} (no rule sets it from production's values)",
                        s[tp.DEFAULTED],
                    )

    def test_no_delivery_is_publish(self):
        for mode in ("", DELETE):
            with self.subTest(mode=mode):
                values, s = self.ok({"team_dashboard.mode": mode})
                self.assertEqual(deploy_tree.get(values, f"{TEAM}.delivery"), "publish")
                self.assertIn(
                    f'{TEAM}.delivery "publish": what team_dashboard.mode '
                    + ('""' if mode == "" else "(not set)")
                    + " means",
                    s[tp.FORMS],
                )

    def test_the_web_hosts_key(self):
        """The web host's ssh key is copied once by the one-time move, which records no key of the web host itself:
        rkmon login records it next, once you accept it."""
        then = "; the web host's own key is then recorded: rkmon login shows it, and records it once you accept it"
        _values, s = self.ok({"publish.html.key": "/secrets/web_key"})
        self.assertIn(
            "web_host_ssh_key: copied once by the one-time move, from the file publish.html.key names"
            + then,
            s[tp.LOGINS],
        )
        for key in ("", DELETE):
            with self.subTest(key=key):
                _values, s = self.ok({"publish.html.key": key})
                self.assertIn(
                    "web_host_ssh_key: copied once by the one-time move, from the secrets folder's webhost_ed25519"
                    + then,
                    s[tp.LOGINS],
                )

    def test_the_components_and_the_jira_site(self):
        """The components are a text split at commas (from_layout1.jira_components), as today's monitor reads them;
        the JIRA site is one of two keys, a trailing "/" ignored (from_layout1.same_address).
        """
        values, s = self.ok({"jira.component": " A one ,, B two "})
        self.assertEqual(
            deploy_tree.get(values, f"{TICKETS}.components"), ["A one", "B two"]
        )
        values, s = self.ok({"jira.component": ""})
        self.assertEqual(deploy_tree.get(values, f"{TICKETS}.components"), [])
        self.assertFalse([x for x in s[tp.FORMS] if "split at commas" in x])
        for given in (3, ["A one", "B two"]):
            with self.subTest(given=given):
                self.assertEqual(
                    self.stopped({"jira.component": given}),
                    [
                        (
                            f"jira.component is {json.dumps(given)}: the components are a text, with commas "
                            "between them",
                            "set jira.component in config.json to the components: a text with commas between them",
                        )
                    ],
                )
        values, s = self.ok({"jira.url": "https://jira.example.com/"})
        self.assertEqual(
            deploy_tree.get(values, "integrations.atlassian_cli.jira_url"),
            "https://jira.example.com",
        )
        self.assertIn(
            "integrations.atlassian_cli.jira_url: atlassian.site and jira.url, which agree",
            s[tp.FORMS],
        )
        for url in ("https://jira.example.com", "https://jira.example.com/"):
            with self.subTest(url=url):
                values, s = self.ok({"atlassian.site": DELETE, "jira.url": url})
                self.assertEqual(
                    deploy_tree.get(values, "integrations.atlassian_cli.jira_url"),
                    "https://jira.example.com",
                )
                self.assertFalse([x for x in s[tp.FORMS] if "which agree" in x])
        self.assertEqual(
            self.stopped({"jira.url": "https://other.example.com"}),
            [
                (
                    "atlassian.site and jira.url name two JIRA sites, and the file has one",
                    "set jira.url in config.json to the address atlassian.site names, or set atlassian.site in "
                    "config.json to the address jira.url names",
                )
            ],
        )
        _values, s = self.ok({"atlassian.enabled": False})
        self.assertIn(
            "atlassian.enabled false: no key; the atlassian CLI is used while ticket creation is on",
            s[tp.FORMS],
        )

    def test_a_bare_jira_site_name_is_its_cloud_address(self):
        """The atlassian CLI names a cloud site by its one label (from_layout1.jira_address): beside the site's address
        it agrees, alone it is carried as that address, and the file holds one address; two sites still stop it.
        """
        site, other = "example", "other"
        cloud = f"https://{site}.atlassian.net"
        path = "integrations.atlassian_cli.jira_url"
        named = f'{path} "{cloud}": atlassian.site "{site}", the atlassian CLI\'s name for that site'
        for url in (cloud, cloud + "/"):
            with self.subTest(url=url):
                values, s = self.ok({"atlassian.site": site, "jira.url": url})
                self.assertEqual(deploy_tree.get(values, path), cloud)
                self.assertIn(
                    f"{path}: atlassian.site and jira.url, which agree", s[tp.FORMS]
                )
                self.assertIn(named, s[tp.FORMS])
                self.assertFalse([x for x in s[tp.DEFAULTED] if x.startswith(path)])
        values, s = self.ok({"atlassian.site": site, "jira.url": DELETE})
        self.assertEqual(deploy_tree.get(values, path), cloud)
        self.assertIn(named, s[tp.FORMS])
        self.assertFalse([x for x in s[tp.FORMS] if "which agree" in x])
        # another name, in either key, or an address that is not the name's, beside it: two sites
        for changes in (
            {"atlassian.site": site, "jira.url": f"https://{other}.atlassian.net"},
            {"atlassian.site": site, "jira.url": other},
            {"atlassian.site": site, "jira.url": "https://jira.example.com"},
            {"atlassian.site": other, "jira.url": cloud},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    self.stops(changes),
                    [
                        "atlassian.site and jira.url name two JIRA sites, and the file has one"
                    ],
                )

    def test_the_prices(self):
        _values, s = self.ok(
            {"prices.claude-opus-5-5._source": "invented", "prices._note": "invented"}
        )
        self.assertIn(
            "notes in the prices, dropped: prices.claude-opus-5-5._source, prices._note",
            s[tp.FORMS],
        )
        # fields that are no price, as leftovers nothing reads: removed keys, by migrate's rule, and the model's prices
        # carried
        values, s = self.ok(
            {
                "prices.claude-opus-5-5.in_long_context": 10,
                "prices.claude-opus-5-5.out_long_context": 37.5,
            }
        )
        self.assertEqual(
            deploy_tree.get(
                values, f"{RA}.spend.usd_per_million_tokens.claude-opus-5-5"
            ),
            {
                "input": 5,
                "output": 25,
                "cache_read": 0.5,
                "cache_write_5_minutes": 6.25,
                "cache_write_1_hour": 10,
            },
        )
        self.assertIs(deploy_tree.get(values, f"{RA}.spend.enabled"), True)
        why = "a price is one of in, out, cache_read, cache_write_5m and cache_write_1h"
        self.assertEqual(why, from_layout1.NOT_A_PRICE)
        self.assertEqual(
            s[tp.REMOVED],
            [
                f"prices.claude-opus-5-5.in_long_context 10: {why}",
                f"prices.claude-opus-5-5.out_long_context 37.5: {why}",
                CLANG,
                TEST_TIMEOUT,
                "58 other removed keys blank, off or at the package's value",
            ],
        )
        self.assertEqual(s[tp.UNKNOWN], ["none"])
        self.assertEqual(
            self.stopped({"prices.claude-opus-5-5": 3}),
            [
                (
                    "prices.claude-opus-5-5 is no mapping of prices",
                    "set prices.claude-opus-5-5 in config.json to a mapping of prices",
                )
            ],
        )
        self.assertEqual(
            self.stopped({"prices": "none"}),
            [
                (
                    "prices is no mapping of model ids to prices",
                    "set prices in config.json to a mapping of model ids to prices",
                )
            ],
        )
        # the file's own check: a field given and one missing, each fixed at the field of production's it comes from
        stopped = self.stopped(
            {"prices.claude-opus-5-5.in": -1, "prices.claude-opus-5-5.out": DELETE}
        )
        price = f"{RA}.spend.usd_per_million_tokens.claude-opus-5-5"
        self.assertEqual(
            [found for found, _fix in stopped],
            [f"{price}.input: -1 is less than 0", f"{price}.output: missing"],
        )
        self.assertEqual(
            stopped[0][1], "set prices.claude-opus-5-5.in in config.json to 0 or more"
        )
        self.assertTrue(
            stopped[1][1].startswith(
                "set prices.claude-opus-5-5.out in config.json to "
            ),
            stopped[1][1],
        )
        values, s = self.ok({"prices": DELETE})
        self.assertIs(deploy_tree.get(values, f"{RA}.spend.enabled"), False)
        self.assertIn(
            f"{RA}.spend.enabled false: production has no prices, so each cost is unknown today",
            s[tp.FORMS],
        )

    def test_the_gpu_tests_environment(self):
        values, s = self.ok({"gpu.env": {"ROCKE_BACKEND": "python", "EXTRA_FLAG": "1"}})
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.remote_gpu_tests.node.environment"),
            {"EXTRA_FLAG": "1"},
        )
        values, s = self.ok({"gpu.env": {"ROCKE_BACKEND": "hip"}})
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.remote_gpu_tests.node.environment"),
            {"ROCKE_BACKEND": "hip"},
        )
        self.assertFalse([x for x in s[tp.FORMS] if "ROCKE_BACKEND left out" in x])
        self.assertEqual(
            self.stopped({"gpu.env": "ROCKE_BACKEND=python"}),
            [
                (
                    "gpu.env is no mapping of variables",
                    "set gpu.env in config.json to a mapping of variable names to their values",
                )
            ],
        )

    def test_no_secret_is_carried_or_shown(self):
        """A value that looks like a secret stops the translation, never shown: its fix sets the keys of production's
        it comes from to a value with no secret in it."""
        address = "https://u:" + "p" * 8 + "@" + "jira.example.com"
        token = "gh" + "p_" + "A" * 36
        no_secret = "in config.json to a value with no secret in it"
        for changes, stop, secret in (
            (
                {"jira.url": address, "atlassian.site": address},
                (
                    "integrations.atlassian_cli.jira_url's value looks like an address with a user or password in "
                    "it, and a settings file has no secret",
                    f"set atlassian.site {no_secret}, and set jira.url {no_secret}",
                ),
                "p" * 8,
            ),
            (
                {"teams.channel": token},
                (
                    "features.alerts.message_label's value looks like a token or key, and a settings file has no "
                    "secret",
                    f"set teams.channel {no_secret}",
                ),
                token,
            ),
        ):
            with self.subTest(stop=stop):
                self.assertEqual(self.stopped(changes), [stop])
                self.assertNotIn(secret, json.dumps(self.stopped(changes)))
        # a model id that looks like a secret names the binding made for it: the file's reader's check
        # (settings_file.secret_texts) leaves that key out of the paths under it, and each stop's fix is the model's
        stopped = self.stopped({"models.audit": token})
        self.assertEqual(
            stopped,
            [
                (
                    f"{RA}.passes.verification_pass.model's value looks like a token or key, and a settings file has "
                    "no secret",
                    f"set models.audit {no_secret}",
                ),
                (
                    "a key in model_bindings looks like a token or key, and a settings file has no secret",
                    f"set models.audit {no_secret}",
                ),
                (
                    "model_bindings.model_id's value looks like a token or key, and a settings file has no secret",
                    f"set models.audit {no_secret}",
                ),
            ],
        )
        self.assertNotIn(token, json.dumps(stopped))
        # a key of production's that looks like a secret: deleted, said, never shown
        binding = {
            "provider": "openai",
            "url": "https://gw.example.com/v1",
            "model": "m-one",
        }
        stopped = self.stopped({"model_bindings": {token: binding}})
        self.assertIn(
            (
                "a key in model_bindings looks like a token or key, and a settings file has no secret",
                "delete model_bindings.(the key that looks like a token or key) from config.json",
            ),
            stopped,
        )
        self.assertNotIn(token, json.dumps(stopped))


class TestTheReportsValues(unittest.TestCase):
    """A value of production's as the report shows it (_shown): its JSON, cut to 80 characters; "(not set)" for
    none."""

    def test_a_value_is_its_json_cut_to_80_characters(self):
        whole = "x" * 78  # its JSON, with the quotes, is 80 characters
        self.assertEqual(tp._shown(whole), json.dumps(whole))
        self.assertEqual(tp._shown(whole + "y"), json.dumps(whole)[:77] + "...")
        self.assertEqual(len(tp._shown(list(range(100)))), 80)
        self.assertEqual(tp._shown(deploy_tree.MISSING), "(not set)")


class TestNoSecretIsShown(unittest.TestCase):
    """A secret in any of production's values or keys (PLANTED, invented) is never shown, whole or in part: not in the
    report, its stops or the file. Each is planted on its own, as a stop may leave out lines that follow it.
    """

    def assertNotShown(self, cfg: dict, shape: Shape) -> None:
        """Neither the report nor the file shows PLANTED, nor any stop, what it found or its fix (which the report
        shows under it, and the command's closing line again)."""
        try:
            done = tp.translate(cfg, shape)
            shown = [*done.report, done.deploy_file]
        except tp.Untranslatable as exc:
            shown = [*exc.report, *(f"{s.found}\n{s.fix}" for s in exc.stops)]
        self.assertNotIn(PLANTED_PART, "\n".join(shown))

    def test_a_secret_in_any_value(self):
        cfg, shape = production_fixture()
        paths = list(texts(cfg))
        self.assertGreater(len(paths), 200)
        for path in paths:
            with self.subTest(path=path):
                self.assertNotShown(planted(cfg, path, PLANTED), shape)

    def test_a_secret_in_any_key(self):
        cfg, shape = production_fixture()
        paths = list(keys(cfg))
        self.assertGreater(len(paths), 400)
        for path in paths:
            with self.subTest(path=path):
                self.assertNotShown(renamed(cfg, path, PLANTED), shape)


# Half a character: a \u escape from \ud800 to \udfff without its other half, which JSON's text may hold, as a str does.
HALF = "\ud800"


class TestTheCheck(_Translation):
    """The file's own check is what `rkmon check` says of the file: the text the writer wrote, read back. Half a
    character, which only the file's reader sees, stops the translation with the key of production's that gives it; a
    value the writer cannot write stops it with the check of the values; the writer refusing what that check passes is
    a fault of the translation."""

    def test_half_a_character_stops_as_the_files_reader_refuses_it(self):
        cfg, shape = production_fixture()
        with self.assertRaises(tp.Untranslatable) as cm:
            tp.translate(
                changed(cfg, {"operator.github_login": "op-login" + HALF}), shape
            )
        self.assertEqual(
            [(s.found, str(s.fix)) for s in cm.exception.stops],
            [
                (
                    "integrations.github_cli.username: the value in your deploy settings has a \\u escape for half "
                    "a character (\\ud800 to \\udfff) without its other half",
                    "set operator.github_login in config.json to a value whose characters are whole",
                )
            ],
        )
        self.assertEqual(
            sections(cm.exception.report)[tp.CHECK],
            ["not checked: its reader refuses the text, as the stops above say"],
        )

    def test_a_value_the_writer_cannot_write_stops_with_the_check_of_the_values(
        self,
    ):
        # a list of lists, which the writer refuses: a list's item is its list's key, at the same place
        self.assertEqual(
            self.stopped({"scope.labels": [["area: tools"]]}),
            [
                (
                    "repository.scope_labels[0]: a list is not text",
                    "set scope.labels[0] in config.json to text",
                )
            ],
        )

    def test_the_writer_refusing_what_the_check_passes_is_a_fault(self):
        cfg, shape = production_fixture()
        with mock.patch.object(
            tp.settings_file, "dump", side_effect=ValueError("refused")
        ):
            with self.assertRaises(ValueError) as cm:
                tp.translate(cfg, shape)
        self.assertNotIsInstance(cm.exception, tp.Untranslatable)

    def test_the_team_dashboard_served_in_prototype_has_its_row_beside_the_others(self):
        """While the team dashboard is served in prototype and serve.address names another address than the one its
        page is served on, the check has a PASS row of its own, "team dashboard", which says so; the report shows it
        with the other rows, as `rkmon check` shows them. Served live, the page is served where serve.address says,
        and the check has no such row."""
        row = (
            f"PASS team dashboard: in prototype, its page is served on {container_layout.LOOPBACK} only, not on "
            f'"{container_layout.EVERY_ADDRESS}" ({TEAM}.serve.address), until it is live'
        )
        served = {
            "team_dashboard.mode": "serve",
            "team_dashboard.publish_host": container_layout.EVERY_ADDRESS,
        }
        for mode, rows in (("prototype", [row]), ("live", [])):
            with self.subTest(mode=mode):
                team = {"enabled": True, "mode": mode}
                cfg, shape = production_fixture()
                changes = {
                    **nested(dashboards={**NESTED["dashboards"], "team": team}),
                    **served,
                }
                done = tp.translate(changed(cfg, changes), shape)
                self.assertValues(
                    read_back(done.deploy_file),
                    {
                        f"{TEAM}.mode": mode,
                        f"{TEAM}.delivery": "serve",
                        f"{TEAM}.serve.address": container_layout.EVERY_ADDRESS,
                    },
                )
                s = sections(done.report)
                self.assertEqual(s[tp.STOPS], ["none"])
                self.assertEqual(s[tp.CHECK], own_check(done.deploy_file))
                self.assertEqual(
                    [x for x in s[tp.CHECK] if x.startswith("PASS team dashboard:")],
                    rows,
                )


class TestHalfACharacterIsNeverShown(unittest.TestCase):
    """Half a character (HALF) in any of production's values or keys: the report, its stops and the file hold it only as
    a \\u escape, never as it is, which no UTF-8 text holds; and the file, when one is written, is one its reader takes.
    Each is planted on its own."""

    def assertWhole(self, cfg: dict, shape: Shape) -> None:
        try:
            done = tp.translate(cfg, shape)
            shown = [*done.report, done.deploy_file]
            settings_file.loads(done.deploy_file, file_types.DEPLOY)
        except tp.Untranslatable as exc:
            shown = [*exc.report, *(f"{s.found}\n{s.fix}" for s in exc.stops)]
        "\n".join(shown).encode("utf-8")

    def test_in_any_value(self):
        cfg, shape = production_fixture()
        paths = list(texts(cfg))
        self.assertGreater(len(paths), 200)
        for path in paths:
            value = cfg
            for p in path:
                value = value[p]
            with self.subTest(path=path):
                self.assertWhole(planted(cfg, path, value + HALF), shape)

    def test_in_any_key(self):
        cfg, shape = production_fixture()
        paths = list(keys(cfg))
        self.assertGreater(len(paths), 400)
        for path in paths:
            with self.subTest(path=path):
                self.assertWhole(renamed(cfg, path, f"{path[-1]}{HALF}"), shape)


class TestAnyKeyMissing(unittest.TestCase):
    """Each of production's keys taken out on its own: the translation stops or writes, never fails, and no line of
    the report, nor any stop, shows a missing value as Python names an object, as it would one taken for text.
    """

    def test_each_key_taken_out(self):
        cfg, shape = production_fixture()
        paths = list(keys(cfg))
        self.assertGreater(len(paths), 400)
        for path in paths:
            out = copy.deepcopy(cfg)
            node = out
            for p in path[:-1]:
                node = node[p]
            del node[path[-1]]
            with self.subTest(path=path):
                try:
                    shown = tp.translate(out, shape).report
                except tp.Untranslatable as exc:
                    shown = [*exc.report, *(f"{s.found}\n{s.fix}" for s in exc.stops)]
                self.assertNotRegex("\n".join(shown), r"object at 0x[0-9a-f]+")


class TestUpdates(_Translation):
    """update.repo and update.branch as the updates section, by layout 1's rule, which `rkmon migrate`
    shares (from_layout1.updates_section, tested in its own module): the <org>/<repo> of a GitHub address, and the
    branch; any other value leaves the section out, and the report names what is left out and why. Each address is in
    the package's one GitHub organization, and an @ stands alone, as the privacy check wants.
    """

    REPO = "ROCm/sample-tool"
    URL = "https://github.com/ROCm/sample-tool.git"

    def test_the_translation_carries_a_github_repository(self):
        values, s = self.ok({"update.repo": self.URL, "update.branch": " develop "})
        self.assertEqual(
            values["updates"], {"repository": self.REPO, "branch": "develop"}
        )
        self.assertIn(
            f'updates.repository "{self.REPO}" and updates.branch "develop": update.repo "{self.URL}", '
            'update.branch " develop "',
            s[tp.FORMS],
        )
        self.assertFalse([x for x in s[tp.DEFAULTED] if x.startswith("updates.")])
        # a branch that is a number is its name, as text
        values, s = self.ok({"update.repo": self.REPO, "update.branch": 7})
        self.assertEqual(values["updates"], {"repository": self.REPO, "branch": "7"})
        # a branch the file does not take: the file's own check stops, its fix production's key
        ((found, fix),) = self.stopped(
            {"update.repo": self.URL, "update.branch": "two words"}
        )
        self.assertEqual(
            found, 'updates.branch: "two words" is not in the form this key takes'
        )
        self.assertTrue(
            fix.startswith(
                "set update.branch in config.json to a value of this form: "
            ),
            fix,
        )

    def test_the_translation_leaves_out_what_it_cannot_carry(self):
        url = self.URL
        for changes, line in (
            (
                {"update.repo": "", "update.branch": ""},
                "updates: left out (update.repo is blank and update.branch is blank)",
            ),
            (
                {"update": DELETE},
                "updates: left out (update.repo is blank and update.branch is blank)",
            ),
            (
                {"update.repo": DELETE, "update.branch": "develop"},
                'updates: left out (update.repo is blank); update.branch "develop" is not carried',
            ),
            (
                {"update.repo": "origin", "update.branch": "develop"},
                'updates: left out (update.repo "origin" names no GitHub repository); update.branch "develop" is '
                "not carried",
            ),
            (
                {"update.repo": 7, "update.branch": None},
                "updates: left out (update.repo 7 names no GitHub repository and update.branch is blank)",
            ),
            (
                {"update.repo": url, "update.branch": "  "},
                f'updates: left out (update.branch is blank); update.repo "{url}" is not carried',
            ),
            (
                {"update.repo": url, "update.branch": ["develop"]},
                f'updates: left out (update.branch ["develop"] names no branch); update.repo "{url}" is not carried',
            ),
        ):
            with self.subTest(changes=changes):
                values, s = self.ok(changes)
                self.assertNotIn("updates", values)
                self.assertIn(line, s[tp.FORMS])
                self.assertFalse(
                    [x for x in s[tp.DEFAULTED] if x.startswith("updates.")]
                )
        values, s = self.ok({"update.repo": "origin", "update.path": "monitor"})
        self.assertIn(
            'update.path "monitor": the package knows its own folder in the repository it comes from',
            s[tp.REMOVED],
        )

    def test_an_address_with_a_password_gives_its_repository_alone(self):
        password = "p" * 8
        url = (
            "https://example-user:" + password + "@" + "github.com/ROCm/sample-tool.git"
        )
        values, s = self.ok({"update.repo": url, "update.branch": "develop"})
        self.assertEqual(
            values["updates"], {"repository": self.REPO, "branch": "develop"}
        )
        (line,) = [x for x in s[tp.FORMS] if x.startswith("updates.")]
        self.assertNotIn(password, line)
        self.assertIn("update.repo (a value that looks like a secret: not shown)", line)


class TestPasses(_Translation):
    """The passes' models and efforts: the Claude session's, the first review pass's and the other
    optional passes', and the bindings they name."""

    WORKER = {
        "provider": "openai_compatible",
        "model_id": "drafter-model-a",
        "base_url": "https://llm-gateway.example.com/drafter/v1",
        "api_key_secret_name": "first_review_pass_api_key",
        "api_key_sent_as": "header_and_bearer",
        "api_key_header": "X-Api-Key",
        "limits": {
            "requests_per_minute": 60,
            "tokens_per_minute": 400000,
            "max_parallel_requests": 4,
        },
        "stall_timeout_seconds": 900,
    }
    GATEWAY = {
        "provider": "openai",
        "url": "https://gw.example.com/v1",
        "model": "m-one",
        "api_key_secret": "gw_key",
        "auth_mode": "bearer",
    }

    def test_the_effort_chain_of_a_pass_that_always_runs(self):
        """Layout 1's chain (from_layout1.pass_effort), as today's config.role_effort resolves it: the pass's own
        effort, a missing one being its role's default; else the base effort, a missing one being today's; where the
        chain ends in the model's own effort, which the file cannot name for a pass that always runs, its role's
        default, and the report says so."""
        orchestrator = f"{RA}.orchestrator.effort"
        for changes, effort, line in (
            (
                {"effort.orchestrator": DELETE},
                "high",
                f'{orchestrator} "high" (its role\'s default, as config.json has no effort.orchestrator)',
            ),
            (
                {"effort.orchestrator": "", "claude.effort": "max"},
                "max",
                f'{orchestrator} "max" (claude.effort)',
            ),
            (
                {"effort.orchestrator": "", "claude.effort": DELETE},
                "high",
                f'{orchestrator} "high" (today\'s base effort, as config.json has no claude.effort)',
            ),
            (
                {"effort.orchestrator": " ", "claude.effort": ""},
                "high",
                f"{orchestrator} \"high\": its role's default; today the pass runs on the model's own effort, as "
                "neither effort.orchestrator nor claude.effort names one, and the file names one for a pass that "
                "always runs",
            ),
        ):
            with self.subTest(changes=changes):
                today = renames.moves(changed(production_fixture()[0], changes))[0]
                values, s = self.ok(changes)
                self.assertEqual(deploy_tree.get(values, orchestrator), effort)
                self.assertIn(line, s[tp.EFFORTS])
                if not line.endswith("always runs"):
                    self.assertEqual(
                        config.role_effort(today, "orchestrator")[0], effort
                    )
                else:
                    self.assertEqual(config.role_effort(today, "orchestrator")[0], "")

    def test_each_passs_own_effort(self):
        values, s = self.ok({"effort.orchestrator": "max", "effort.ci_triage": "low"})
        self.assertValues(
            values,
            {
                f"{RA}.orchestrator.effort": "max",
                f"{RA}.passes.ci_triage_pass.effort": "low",
            },
        )
        self.assertIn(
            f'{RA}.orchestrator.effort "max" (effort.orchestrator)', s[tp.EFFORTS]
        )
        self.assertIn(
            f'{RA}.passes.ci_triage_pass.effort "low" (effort.ci_triage "low")',
            s[tp.EFFORTS],
        )

    def test_the_local_worker_as_the_first_review_pass(self):
        values, s = self.ok({"reviewer": "worker"})
        self.assertValues(
            values,
            {
                "model_bindings": {"worker": self.WORKER},
                f"{RA}.passes.first_review_pass": {
                    "model": "worker",
                    "effort": "default",
                },
            },
        )
        self.assertEqual(
            s[tp.FIRST],
            [
                f'{RA}.passes.first_review_pass.model "worker": the local worker leaves the package, and the pass '
                "keeps its model as a binding named worker"
            ],
        )
        self.assertIn(
            f'{RA}.passes.first_review_pass.effort "default" (worker.think "max", the local worker\'s)',
            s[tp.EFFORTS],
        )
        self.assertIn(
            'model_bindings.worker.api_key_secret_name "first_review_pass_api_key": the one-time move copies '
            "worker_api_key to it",
            s[tp.LOGINS],
        )
        self.assertFalse([x for x in s[tp.LOGINS] if x.startswith("worker_api_key:")])

    def test_the_local_workers_think_and_blank_stall(self):
        values, s = self.ok(
            {"reviewer": "worker", "worker.stall_s": "", "worker.think": "low"}
        )
        self.assertValues(
            values,
            {
                "model_bindings.worker.stall_timeout_seconds": 900,
                f"{RA}.passes.first_review_pass.effort": "low",
            },
        )
        self.assertIn(
            "model_bindings.worker.stall_timeout_seconds 900: what a blank worker.stall_s ran on",
            s[tp.FIRST],
        )

    def test_a_local_ollama(self):
        values, s = self.ok(
            {
                "reviewer": "worker",
                "worker.backend": "ollama",
                "worker.url": "http://127.0.0.1:11434",
                "worker.model": "local-model",
                "worker.max_parallel": 0,
                "worker.rpm": 0,
                "worker.tpm": 0,
                "worker.auth_mode": "",
                "worker.auth_header": "Authorization",
                "worker.auth_scheme": "Bearer",
            }
        )
        self.assertEqual(
            deploy_tree.get(values, "model_bindings"),
            {
                "worker": {
                    "provider": "ollama",
                    "model_id": "local-model",
                    "server_url": "http://127.0.0.1:11434",
                    "limits": {"max_parallel_requests": 1},
                    "stall_timeout_seconds": 900,
                }
            },
        )
        self.assertIn(
            "model_bindings.worker.limits.max_parallel_requests 1: worker.max_parallel 0 on a local Ollama ran one "
            "request at a time",
            s[tp.FIRST],
        )

    def test_another_pass_on_the_local_worker(self):
        values, s = self.ok({"models.ci_triage": "worker"})
        self.assertValues(
            values,
            {
                "model_bindings": {"worker": self.WORKER},
                f"{RA}.passes.ci_triage_pass.model": "worker",
                f"{RA}.passes.first_review_pass.model": "default",
            },
        )
        self.assertIn(
            f'{RA}.passes.ci_triage_pass.model "worker": the local worker\'s binding (models.ci_triage "worker")',
            s[tp.MODELS],
        )
        self.assertEqual(
            s[tp.FIRST],
            [
                f'{RA}.passes.first_review_pass.model "default": no draft; the verification pass reviews alone, as '
                "today"
            ],
        )
        values, s = self.ok({"models.ci_triage": "worker", "worker": DELETE})
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.passes.ci_triage_pass.model"), "default"
        )
        self.assertIn(
            f'{RA}.passes.ci_triage_pass.model "default": models.ci_triage "worker" names no binding production '
            "has, and no local worker: not carried",
            s[tp.MODELS],
        )

    def test_productions_own_bindings(self):
        spare = dict(self.GATEWAY, temperature=0.7)
        values, s = self.ok(
            {"reviewer": "gw", "model_bindings": {"gw": self.GATEWAY, "spare": spare}}
        )
        carried = {
            "provider": "openai_compatible",
            "model_id": "m-one",
            "base_url": "https://gw.example.com/v1",
            "api_key_secret_name": "gw_api_key",
            "api_key_sent_as": "bearer",
        }
        self.assertValues(
            values,
            {
                "model_bindings": {"gw": carried, "spare": carried},
                f"{RA}.passes.first_review_pass.model": "gw",
            },
        )
        # Every binding's key's name ends in _api_key: production's gets that end, and the one-time move copies it.
        self.assertIn(
            'model_bindings.gw.api_key_secret_name "gw_api_key": the one-time move copies gw_key to it',
            s[tp.LOGINS],
        )
        self.assertEqual(
            s[tp.FIRST][0],
            f'{RA}.passes.first_review_pass.model "gw": the binding production names',
        )
        self.assertTrue(
            s[tp.FIRST][1].startswith(
                "the local worker's settings, not carried: no pass runs on it."
            )
        )
        self.assertEqual(len(s[tp.FIRST]), 3)
        self.assertIn(
            "model_bindings.spare.temperature 0.7: not carried; the file has no place for it",
            s[tp.MODELS],
        )
        self.assertIn("model_bindings.spare: carried; no pass names it", s[tp.MODELS])
        values, s = self.ok(
            {"models.notice_draft": "gw", "model_bindings": {"gw": self.GATEWAY}}
        )
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.passes.notice_draft_pass.model"), "gw"
        )
        self.assertIn(
            f'{RA}.passes.notice_draft_pass.model "gw": the binding production names (models.notice_draft)',
            s[tp.MODELS],
        )

    def test_an_ollama_bindings_parallel_requests(self):
        """The limit on parallel requests of an ollama binding production has, by layout 1's rule
        (from_layout1.ollama_parallel_requests): a blank one had no cap, which the file writes as 0; any other value as
        it is."""
        ollama = {
            "provider": "ollama",
            "url": "http://127.0.0.1:11434",
            "model": "local-model",
        }
        path = "model_bindings.ol.limits.max_parallel_requests"
        blank = f"{path} 0: no cap, what a blank model_bindings.ol.concurrency ran on"
        for concurrency, want, lines in (
            (DELETE, 0, [blank]),
            ("", 0, [blank]),
            (0, 0, []),
            (3, 3, []),
        ):
            with self.subTest(concurrency=concurrency):
                b = dict(ollama)
                if concurrency is not DELETE:
                    b["concurrency"] = concurrency
                values, s = self.ok(
                    {"models.notice_draft": "ol", "model_bindings": {"ol": b}}
                )
                self.assertEqual(deploy_tree.get(values, path), want)
                self.assertEqual([x for x in s[tp.MODELS] if x.startswith(path)], lines)

    def test_how_a_bindings_key_is_sent(self):
        """How a binding's key is sent is what today's client sends (worker/llm_client.py, auth_headers), in
        layout 2's words (from_layout1.key_sending): "both" adds Authorization beside another header only; a blank
        scheme is left out, the file's default."""
        bound = {k: v for k, v in self.GATEWAY.items() if k != "auth_mode"}
        fields = ("api_key_sent_as", "api_key_header", "api_key_scheme")
        for how, sent in (
            ({}, ("header", "Authorization", "Bearer")),
            ({"auth_mode": "both"}, ("header", "Authorization", "Bearer")),
            (
                {"auth_mode": "both", "auth_header": ""},
                ("header", "Authorization", "Bearer"),
            ),
            (
                {
                    "auth_mode": "both",
                    "auth_header": "authorization",
                    "auth_scheme": "Token",
                },
                ("header", "authorization", "Token"),
            ),
            (
                {"auth_mode": "both", "auth_header": "X-Api-Key", "auth_scheme": ""},
                ("header_and_bearer", "X-Api-Key"),
            ),
            (
                {"auth_mode": "header", "auth_header": "X-Api-Key"},
                ("header", "X-Api-Key", "Bearer"),
            ),
            ({"auth_mode": "bearer", "auth_header": "X-Api-Key"}, ("bearer",)),
        ):
            with self.subTest(how=how):
                values, _ = self.ok(
                    {"reviewer": "gw", "model_bindings": {"gw": {**bound, **how}}}
                )
                got = deploy_tree.get(values, "model_bindings.gw")
                self.assertEqual(tuple(got[k] for k in fields if k in got), sent)

    def test_a_name_production_has_no_binding_for(self):
        values, s = self.ok({"reviewer": "nobody", "models.quality_draft": "nobody"})
        self.assertValues(
            values,
            {
                f"{RA}.passes.first_review_pass.model": "default",
                f"{RA}.passes.first_code_quality_pass.model": "default",
            },
        )
        self.assertIn(
            f'{RA}.passes.first_review_pass.model "default": reviewer "nobody" names no binding production has: '
            "not carried",
            s[tp.FIRST],
        )
        self.assertIn(
            f'{RA}.passes.first_code_quality_pass.model "default": models.quality_draft "nobody" names no binding '
            "production has, and no local worker: not carried",
            s[tp.MODELS],
        )

    def test_bindings_the_file_cannot_carry(self):
        stopped = self.stopped(
            {
                "models.notice_draft": "odd",
                "model_bindings": {"odd": {"provider": "carrier-pigeon", "model": "m"}},
            }
        )
        self.assertEqual(
            [found for found, _fix in stopped],
            [
                'model_bindings.odd.provider "carrier-pigeon" is no provider the file has'
            ],
        )
        self.assertRegex(
            stopped[0][1],
            r'^set model_bindings\.odd\.provider in config\.json to one of "[a-z-]+"(, "[a-z-]+")* or "[a-z-]+"$',
        )
        self.assertIn('"openai-compatible"', stopped[0][1])
        self.assertEqual(
            self.stopped(
                {"reviewer": "worker", "model_bindings": {"worker": self.GATEWAY}}
            ),
            [
                (
                    "model_bindings.worker: production has a binding of that name, and a pass's model needs it",
                    "delete model_bindings.worker from config.json, or give it another name, and the keys that name "
                    "it too",
                )
            ],
        )
        self.assertEqual(
            self.stopped({"model_bindings": ["gw"]}),
            [
                (
                    "model_bindings is no mapping of bindings",
                    "set model_bindings in config.json to a mapping of binding names to bindings",
                )
            ],
        )
        self.assertEqual(
            self.stopped({"model_bindings": {"gw": "m-one"}}),
            [
                (
                    "model_bindings.gw is no binding",
                    "set model_bindings.gw in config.json to a mapping of the binding's settings",
                )
            ],
        )

    def test_the_local_workers_backend(self):
        """The local worker's binding takes its provider from worker.backend, whose fix names the backends."""
        stopped = self.stopped(
            {"reviewer": "worker", "worker.backend": "carrier-pigeon"}
        )
        self.assertEqual(
            stopped,
            [
                (
                    'worker.backend "carrier-pigeon" is no provider the file has',
                    "set worker.backend in config.json to one of "
                    + " or ".join(
                        json.dumps(v) for v in deploy_tree.WORKER_BACKEND.values()
                    ),
                )
            ],
        )

    def test_the_claude_sessions_models(self):
        values, s = self.ok(
            {"models.audit": "sonnet", "models.deep": "claude-opus-5-5[1m]"}
        )
        self.assertValues(
            values,
            {
                f"{RA}.passes.verification_pass.model": "sonnet",
                f"{RA}.passes.code_quality_pass.model": "claude-opus-5-5-1m",
                "model_bindings": {
                    "claude-opus-5-5-1m": {
                        "provider": "claude",
                        "model_id": "claude-opus-5-5[1m]",
                    }
                },
            },
        )
        self.assertIn(
            f'{RA}.passes.code_quality_pass.model "claude-opus-5-5-1m": a claude binding of the model id '
            '"claude-opus-5-5[1m]", named without the characters a binding\'s name does not take',
            s[tp.MODELS],
        )
        same = {"provider": "claude", "model": "claude-opus-5-5[1m]"}
        values, _s = self.ok(
            {
                "orchestrator.model": "claude-opus-5-5[1m]",
                "model_bindings": {"claude-opus-5-5-1m": same},
            }
        )
        self.assertEqual(
            deploy_tree.get(values, f"{RA}.orchestrator.model"), "claude-opus-5-5-1m"
        )
        self.assertEqual(
            list(deploy_tree.get(values, "model_bindings")), ["claude-opus-5-5-1m"]
        )


class TestClaudeCode(_Translation):
    """integrations.claude_code from claude_env's shape, and claude.gateway, which wins over it."""

    def test_the_gateway_wins(self):
        values, s = self.ok({"claude.gateway": "https://other-gateway.example.com/a"})
        self.assertEqual(
            deploy_tree.get(values, "integrations.claude_code.base_url"),
            "https://other-gateway.example.com/a",
        )
        self.assertIn(
            "integrations.claude_code.base_url: claude.gateway, which wins over ANTHROPIC_BASE_URL (which differs)",
            s[tp.CLAUDE],
        )

    def test_the_gateway_drops_claude_envs_address_with_its_credential(self):
        """claude.gateway wins over ANTHROPIC_BASE_URL: an address with a credential in it stops the translation only
        when there is no gateway."""
        _cfg, shape = production_fixture()
        shape = shape._replace(
            base_url="", base_url_credential=claude_env_shape.KEY_IN_ADDRESS
        )
        self.assertEqual(
            self.stopped(shape=shape),
            [
                (
                    "claude_env: ANTHROPIC_BASE_URL looks like an address with the key in it, and a settings file "
                    "has no credential",
                    "set ANTHROPIC_BASE_URL in claude_env to the gateway's address, with no credential in it",
                )
            ],
        )
        address = "https://other-gateway.example.com/a"
        values, _s = self.ok({"claude.gateway": address}, shape=shape)
        self.assertEqual(
            deploy_tree.get(values, "integrations.claude_code.base_url"), address
        )

    def test_headers_no_setting_carries(self):
        _cfg, shape = production_fixture()
        for headers, stop in (
            (
                (Header("X-A", True, ""), Header("X-B", True, "")),
                (
                    "claude_env: ANTHROPIC_CUSTOM_HEADERS has 2 headers, and the settings send the key in one header "
                    "at most",
                    "set ANTHROPIC_CUSTOM_HEADERS in claude_env to the one header that sends the key, when the "
                    "gateway takes requests without the others",
                ),
            ),
            (
                (Header("X-A", False, ""),),
                (
                    "claude_env: the custom header X-A sends a value that is not the key, and no setting sends a "
                    "header of fixed text",
                    "set ANTHROPIC_CUSTOM_HEADERS in claude_env to its lines without the header X-A, when the "
                    "gateway takes requests without it",
                ),
            ),
        ):
            with self.subTest(stop=stop):
                self.assertEqual(
                    self.stopped(shape=shape._replace(headers=headers)),
                    [stop],
                )

    def test_each_fix_edits_claude_env(self):
        """Each fix of what the settings cannot carry (claude_env_shape.settings), as a stop shows it: an edit of
        claude_env."""
        for shape, stop in (
            (
                Shape(
                    variables=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                    keys_differ=True,
                ),
                (
                    "claude_env: ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are two different keys, and the claude "
                    "login is one key",
                    "delete ANTHROPIC_API_KEY from claude_env, or delete ANTHROPIC_AUTH_TOKEN from claude_env, "
                    "whichever the gateway does not take",
                ),
            ),
            (
                Shape(
                    variables=("ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS"),
                    headers=(Header("", False, ""),),
                ),
                (
                    "claude_env: ANTHROPIC_CUSTOM_HEADERS has 1 line that does not read as a header: a name, a colon "
                    "and a value",
                    "set ANTHROPIC_CUSTOM_HEADERS in claude_env to one header a line: a name, a colon and a value",
                ),
            ),
        ):
            with self.subTest(stop=stop):
                self.assertEqual(self.stopped(shape=shape), [stop])

    def test_a_key_alone_and_no_key(self):
        values, s = self.ok(shape=Shape(variables=("ANTHROPIC_API_KEY",)))
        self.assertEqual(
            deploy_tree.get(values, "integrations.claude_code"),
            {
                "base_url": "",
                "api_key_sent_as": "header",
                "api_key_header": "",
                "api_key_scheme": "",
            },
        )
        self.assertNotIn(
            "claude_env sets no key: the claude login is the key, which rkmon login claude installs",
            s[tp.CLAUDE],
        )
        shape = Shape(
            variables=("ANTHROPIC_BASE_URL",), base_url="https://gw.example.com"
        )
        values, s = self.ok(shape=shape)
        self.assertEqual(
            deploy_tree.get(values, "integrations.claude_code.base_url"),
            "https://gw.example.com",
        )
        self.assertIn(
            "claude_env sets no key: the claude login is the key, which rkmon login claude installs",
            s[tp.CLAUDE],
        )

    def test_the_nonessential_traffic(self):
        _cfg, shape = production_fixture()
        _values, s = self.ok(shape=shape._replace(nonessential_traffic="0"))
        self.assertIn(
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC "0": not carried; the container sets it to 1 for every '
            "session, so the nonessential traffic production sends stops",
            s[tp.CLAUDE],
        )


class TestOtherKeys(_Translation):
    """Every key of config.json no rule reads, by where it goes: the package's own values, removed keys,
    intended differences, the two log and tick values merged into one, and keys this translation does not know.
    """

    def test_each_key_by_where_it_goes(self):
        _values, s = self.ok(
            {
                "huge_pr_lines": 9000,
                "mirror.enabled": True,
                "invented_key": 1,
                "container.write_uid": 1500,
                "tick_interval_min": 15,
            }
        )
        self.assertEqual(
            s[tp.PACKAGE],
            [
                "huge_pr_lines: production's 9000, the package's 8000",
                "69 other package keys at the package's own value",
            ],
        )
        self.assertEqual(
            s[tp.REMOVED],
            [
                "mirror.enabled true: an unused feature",
                CLANG,
                TEST_TIMEOUT,
                "57 other removed keys blank, off or at the package's value",
            ],
        )
        self.assertEqual(s[tp.UNKNOWN], ["invented_key"])
        self.assertIn(
            "container.write_uid: the package's 1999, not production's 1500 (no file outside the container's "
            "volumes is this user's, so no deployment changes it)",
            s[tp.INTENDED],
        )
        self.assertIn(
            "tick_interval_min 15: no key; one value with monitor.tick_interval_minutes, which is 10, so "
            "production's own is not carried",
            s[tp.FORMS],
        )

    def test_a_path_in_no_configuration_file_added_by_hand_is_a_derived_key(self):
        """A path that is in no configuration file (registry.DERIVED_NOT_IN_FILES), such as the site lessons, is
        derived: added to production's file by hand, it is listed with the derived keys, not with the keys the
        translation does not know, and it is not carried."""
        changes = {
            key: f"/srv/invented/{key.split('.')[-1]}"
            for key in registry.DERIVED_NOT_IN_FILES
        }
        self.assertIn("paths.site_lessons", changes)
        values, s = self.ok(changes)
        self.assertEqual(s[tp.UNKNOWN], ["none"])
        [before] = self.ok()[1][tp.DERIVED]
        [derived] = s[tp.DERIVED]
        count, _, keys = derived.partition(": ")
        self.assertEqual(int(count), int(before.partition(": ")[0]) + len(changes))
        self.assertEqual([k for k in changes if k not in keys.split(", ")], [])
        self.assertNotIn("/srv/invented/", json.dumps(values))

    def test_a_removed_key_names_the_program(self):
        _values, s = self.ok({"container.mounts": ["/srv/key:/srv/key:ro"]})
        self.assertIn(
            'container.mounts ["/srv/key:/srv/key:ro"]: the remote GPU tests\' key is the secret gpu_ssh_key, '
            "installed with: rkmon secret gpu_ssh_key FILE",
            s[tp.REMOVED],
        )

    def test_the_images_keys_are_build_settings(self):
        # The image's keys, as production's config.json has them: each with its key in a build settings file by the
        # build's rules, a key with an older name through the key in its place; no build file is written.
        folder = "https://installers.example.com/rocm"
        values, s = self.ok(
            {
                "image.base": "ubuntu:24.04",
                "rocm.installer_base": folder,
                "rocm.version": "10.1.0",
            }
        )
        self.assertEqual(
            s[tp.BUILD],
            [
                'image.base "ubuntu:24.04": operating_system.base_image',
                f'rocm.installer_base "{folder}": rocm.package_url, which takes one installer\'s full URL, not a '
                "folder",
                'rocm.version "10.1.0": rocm.version',
            ],
        )
        self.assertEqual(s[tp.UNKNOWN], ["none"])
        self.assertEqual([k for k in ("operating_system", "rocm") if k in values], [])
        self.assertEqual(self.ok()[1][tp.BUILD], ["none"])
        # Each key of the image's goes as layout 1's build settings move it.
        for key, new in from_layout1.BUILD_MOVES.items():
            with self.subTest(key=key):
                self.assertEqual(tp._build_key(key), new)
        self.assertEqual(tp._build_key("rocm.clang_commit"), "")
        self.assertEqual(tp._build_key("invented_key"), "")


class TestInside(unittest.TestCase):
    """--inside, as the one-off container runs it: config.json in the monitor's home and claude_env in its secrets
    folder, read and never changed; the translation printed as one JSON object; claude_env's key never printed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self.tmp.name)
        self.secrets = self.home / "secrets"
        self.secrets.mkdir(mode=0o700)
        self.cfg = changed(
            production_fixture()[0], {"paths.secrets_dir": str(self.secrets)}
        )
        self.write(self.home / "config.json", self.cfg)
        self.write(self.secrets / "claude_env", claude_env())

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path: pathlib.Path, data) -> None:
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        path.chmod(0o600)

    def inside(self) -> tuple:
        p = subprocess.run(
            [sys.executable, "-I", "-B", str(TOOL), "--inside"],
            cwd=self.home,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
                "RKMON_HOME": str(self.home),
            },
            capture_output=True,
            text=True,
            timeout=120,
        )
        return p.returncode, p.stdout, p.stderr

    def snapshot(self) -> dict:
        """Each file and folder in the home: its bytes and its mode."""
        return {
            str(p.relative_to(self.home)): (
                p.read_bytes() if p.is_file() else None,
                stat.S_IMODE(p.stat().st_mode),
            )
            for p in self.home.rglob("*")
        }

    def failed(self) -> str:
        """The one line on stderr of a run that cannot read what it needs."""
        rc, out, err = self.inside()
        self.assertEqual((rc, out), (1, ""), err)
        self.assertEqual(len(err.splitlines()), 1, err)
        self.assertNotIn(KEY, err)
        return err.strip()

    def test_the_translation_of_the_homes_files(self):
        before = self.snapshot()
        rc, out, err = self.inside()
        self.assertEqual((rc, err), (0, ""))
        # The tool runs in a process of its own, which reads the package's own repository defaults.
        with mock.patch.object(
            repository_defaults, "FOLDER", home_repository.PACKAGE_FOLDER
        ):
            want = tp._answer(self.cfg, production_fixture()[1])
        self.assertEqual(json.loads(out), want)
        self.assertNotIn(KEY, out)
        self.assertEqual(self.snapshot(), before)

    def test_claude_envs_shape_is_the_fixtures(self):
        self.assertEqual(
            claude_env_shape.read(
                self.secrets / "claude_env", settings_file.secret_shape
            ),
            production_fixture()[1],
        )

    def test_a_setting_that_cannot_be_carried(self):
        self.write(
            self.home / "config.json",
            changed(self.cfg, {"features.stability-issues": False}),
        )
        rc, out, err = self.inside()
        self.assertEqual((rc, err), (0, ""))
        data = json.loads(out)
        self.assertEqual(data["deploy_file"], "")
        stops = [(s["found"], checks.fix_from_json(s["fix"])) for s in data["stops"]]
        self.assertEqual(len(stops), 1)
        self.assertEqual(
            sections(data["report"])[tp.STOPS],
            [f"{found} {checks.FIX_LABEL}{fix}" for found, fix in stops],
        )

    def test_no_config_json(self):
        (self.home / "config.json").unlink()
        self.assertEqual(
            self.failed(),
            f"{self.home / 'config.json'}: cannot be read (No such file or directory)",
        )

    def test_a_config_json_that_is_no_mapping_of_settings(self):
        (self.home / "config.json").write_text("{", encoding="utf-8")
        self.assertTrue(self.failed().startswith("config.json is not JSON: "))
        (self.home / "config.json").write_text("[]", encoding="utf-8")
        self.assertEqual(self.failed(), "config.json is not a mapping of settings")

    def test_a_number_json_has_not(self):
        """Python's reader takes NaN, Infinity and a number too large for a float, which no JSON number is and no
        file carries: config.json is not read, and the number is not shown."""
        for text, said in (
            ('{"a": NaN}', "NaN is no JSON number"),
            ('{"a": -Infinity}', "-Infinity is no JSON number"),
            ('{"a": 1e999}', "it has a number too large for one"),
        ):
            with self.subTest(text=text):
                (self.home / "config.json").write_text(text, encoding="utf-8")
                self.assertEqual(self.failed(), f"config.json is not JSON: {said}")

    def test_a_claude_env_it_refuses(self):
        self.write(self.secrets / "claude_env", dict(claude_env(), OTHER_VARIABLE="x"))
        self.assertTrue(
            self.failed().endswith(
                "names variables the session does not take: OTHER_VARIABLE"
            )
        )

    def test_a_fault_prints_where_it_is_never_its_message(self):
        """A fault of the translation's code prints its kind and its place: its message may hold a value."""

        def fault(config, claude_env):
            raise RuntimeError(f"a value: {KEY}")

        err = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)}),
            mock.patch.object(tp, "translate", fault),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(err),
        ):
            rc = tp._inside()
        self.assertEqual((rc, out.getvalue()), (1, ""))
        self.assertRegex(
            err.getvalue(),
            r"\Athe translation failed: RuntimeError at test_translate_production\.py:\d+\n\Z",
        )


class FakeDockerCase(unittest.TestCase):
    """The tool run with --container in a process of its own, with a fake docker on PATH that records each command and
    answers it (`answer`): nothing calls the real docker. The tests of the round trip's command use it too
    (tests/test_round_trip.py).
    """

    NAME = "l2test-production"
    IMAGE = "sha256:" + "0" * 64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.bin = sandbox.safe_path(self.d)
        shutil.copy(HERE / "fakes" / "fake_docker", pathlib.Path(self.bin) / "docker")
        (pathlib.Path(self.bin) / "docker").chmod(0o755)
        self.log = self.d / "docker.log"
        self.answers = self.d / "answers.json"
        self.env = {
            "PATH": self.bin,
            "HOME": str(self.d),
            "LANG": "C.UTF-8",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_DOCKER_ANSWERS": str(self.answers),
        }
        self.cfg, self.shape = production_fixture()
        self.translation = tp._answer(self.cfg, self.shape)
        self.mounts = [
            {
                "Type": "volume",
                "Name": "l2test-work",
                "Destination": container_layout.WORK_DIR,
            },
            {
                "Type": "volume",
                "Name": "l2test-secrets",
                "Destination": container_layout.SECRETS_DIR,
            },
            {
                "Type": "volume",
                "Name": "l2test-secrets-write",
                "Destination": "/secrets-write",
            },
        ]
        self.answer(self.translation)

    def tearDown(self):
        self.tmp.cleanup()

    def answer(
        self, translation, mounts=None, run_code: int | str = 0, ids=None
    ) -> None:
        """The fake docker's answers: inspect gives the image, MOUNTS and IDS (by default the compose file's blank
        RKMON_UID and RKMON_GID); run prints TRANSLATION (as JSON unless it is text) and exits RUN_CODE: a number,
        or "SIGINT", Ctrl-C's (tests/fakes/fake_docker).
        """
        inspect = {
            "image": self.IMAGE,
            "mounts": self.mounts if mounts is None else mounts,
            "ids": ["RKMON_UID=", "RKMON_GID=", None] if ids is None else ids,
        }
        run = translation if isinstance(translation, str) else json.dumps(translation)
        rows = [
            ["^container inspect ", json.dumps(inspect), 0],
            ["^run --rm ", run, run_code],
        ]
        self.answers.write_text(json.dumps(rows), encoding="utf-8")

    def run_(self, *argv, env=None) -> tuple:
        p = subprocess.run(
            [sys.executable, "-B", str(TOOL), *map(str, argv)],
            cwd=self.d,
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return p.returncode, p.stdout, p.stderr

    def calls(self) -> list:
        """Each docker command, its arguments joined."""
        if not self.log.exists():
            return []
        return [
            x.split("\t", 1)[1]
            for x in self.log.read_text(encoding="utf-8").splitlines()
        ]

    def shown(self, path) -> str:
        """PATH as the tool shows it, its home folder the test's (settings_file.shown, the way the rows show it)."""
        with mock.patch.dict(os.environ, {"HOME": self.env["HOME"]}):
            return settings_file.shown(path)

    def assertCloses(self, out: str, closing: str) -> None:
        """OUT ends with CLOSING word for word: the outcome's lines, however outcome.finish wraps them."""
        words = closing.split()
        self.assertEqual(out.split()[-len(words) :], words)


class TestContainer(FakeDockerCase):
    """--container: docker inspect for the container's image and volumes, then the one-off container,
    with a fake docker on PATH that records each command and answers it. Nothing calls the real docker.
    """

    def test_the_one_off_container_reads_the_volumes_read_only(self):
        """Each of docker's two commands, whole: inspect reads the image, the mounts and the monitor's ids alone; the
        one-off container, hardened, runs as the monitor's user with no capability, one CPU's time and a memory limit
        at most, and mounts the work and secrets volumes and this package's folder read-only. Docker writes its id to
        a file in a private temporary folder (--cidfile), which is gone after."""
        rc, out, err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(rc, 0, out + err)
        inspect, run = self.calls()
        cidfile = pathlib.Path(run.split(" ")[3])
        self.assertEqual(cidfile.name, tp.CIDFILE)
        self.assertTrue(cidfile.parent.name.startswith("translate_production-"))
        self.assertFalse(cidfile.parent.exists())
        self.assertEqual(
            inspect,
            "container inspect --format "
            '{"image": {{json .Image}}, "mounts": {{json .Mounts}}, "ids": [{{range .Config.Env}}{{if or '
            '(eq (index (split . "=") 0) "RKMON_UID") (eq (index (split . "=") 0) "RKMON_GID")}}{{json .}}, '
            "{{end}}{{end}}null]} " + self.NAME,
        )
        whole = [
            "run",
            "--rm",
            "--cidfile",
            str(cidfile),
            "--pull",
            "never",
            "--log-driver",
            "none",
            "--network",
            "none",
            "--cpus",
            tp.ONE_OFF_CPUS,
            "--memory",
            tp.ONE_OFF_MEMORY,
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--user",
            "rkmon:rkmon",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            "type=volume,src=l2test-work,dst=/work,readonly,volume-nocopy",
            "--mount",
            "type=volume,src=l2test-secrets,dst=/secrets,readonly,volume-nocopy",
            "--mount",
            f"type=bind,src={ROOT},dst=/translation,readonly",
            "--entrypoint",
            "python3",
            self.IMAGE,
            "-I",
            "-B",
            "/translation/tools/translate_production.py",
            "--inside",
        ]
        self.assertEqual(run, " ".join(whole))

    def test_the_one_off_container_runs_the_tool_with_its_callers_arguments(self):
        """The tool in the one-off container gets the arguments one_off is given, as they are: main's --inside, and
        with it --round-trip for the round trip; others, or none."""
        work, secrets = ("volume", "l2test-work"), ("volume", "l2test-secrets")
        tool = [self.IMAGE, "-I", "-B", "/translation/tools/translate_production.py"]
        for args in (
            ("--inside",),
            ("--inside", "--round-trip"),
            ("--another", "value"),
            (),
        ):
            with self.subTest(args=args):
                command = tp.one_off(
                    "docker", self.IMAGE, work, secrets, "rkmon:rkmon", args
                )
                self.assertEqual(command[command.index(self.IMAGE) :], [*tool, *args])

    def test_the_monitors_user(self):
        """The one-off container runs as the monitor's user, with the ids the container sets (docker/rk_ids.py), each
        one blank or unset the image's own; an id that is no user id is an error, and no container runs.
        """
        for ids, user in (
            (["RKMON_UID=1234", "RKMON_GID=1235", None], "1234:1235"),
            (["RKMON_UID= 1234 ", "RKMON_GID=", None], "1234:rkmon"),
            (["RKMON_GID=1235", None], "rkmon:1235"),
            ([None], "rkmon:rkmon"),
            (None, "rkmon:rkmon"),
        ):
            with self.subTest(ids=ids):
                self.assertEqual(tp.monitor_user(ids), user)
        self.answer(self.translation, ids=["RKMON_UID=1234", "RKMON_GID=1235", None])
        rc, out, err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(" --user 1234:1235 --cap-drop ALL ", self.calls()[1])
        for given in ("0", "-1", "abc", "2147483648", "1e3"):
            with self.subTest(given=given):
                self.log.unlink()
                self.answer(self.translation, ids=[f"RKMON_UID={given}", None])
                rc, out, _err = self.run_("--container", self.NAME, "--report-only")
                self.assertEqual(
                    (rc, out.splitlines()),
                    (
                        1,
                        [
                            f'ERROR: the container {self.NAME}\'s RKMON_UID "{given}" is no user id (a number from 1 '
                            "to 2147483647)"
                        ],
                    ),
                )
                self.assertEqual(len(self.calls()), 1)

    def test_report_only_prints_the_report_and_writes_nothing(self):
        """The report, then PASSED and the next line: the round trip."""
        before = {p.name for p in self.d.iterdir()}
        rc, out, err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(
            out.splitlines(),
            [
                *self.translation["report"],
                f"PASSED: the translation carries {self.NAME}'s settings; nothing written",
                f"next: python3 tools/translate_production.py --container {self.NAME} --round-trip",
            ],
        )
        self.assertEqual({p.name for p in self.d.iterdir()}, before | {"docker.log"})

    def test_out_writes_the_file_yours_only(self):
        """The file, yours only; the outcome and the next line show it `~/…` under the home folder."""
        folder = self.d / "settings"
        folder.mkdir()
        path = folder / "deploy.yaml"
        rc, out, err = self.run_("--container", self.NAME, "--out", path)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(
            path.read_text(encoding="utf-8"), self.translation["deploy_file"]
        )
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        shown = self.shown(path)
        self.assertEqual(shown, "~/settings/deploy.yaml")
        self.assertCloses(
            out,
            f"WROTE {shown}\n{steps.next_line(steps.DEPLOY_STEPS[1], FILE=shown)}",
        )

    def test_out_that_cannot_be_written_is_an_error(self):
        """A file whose folder is a file: an error, which shows the file as the other lines do."""
        (self.d / "settings").write_text("", encoding="utf-8")
        path = self.d / "settings" / "deploy.yaml"
        rc, out, _err = self.run_("--container", self.NAME, "--out", path)
        self.assertEqual(rc, 1)
        self.assertCloses(
            out, f"ERROR: {self.shown(path)} cannot be written: File exists"
        )
        self.assertEqual((self.d / "settings").read_text(encoding="utf-8"), "")

    def test_a_setting_that_cannot_be_carried_writes_nothing(self):
        """The closing line counts the stops; its fix is each stop's in turn, and its next line runs the tool
        again."""
        for changes, count in (
            ({"features.stability-issues": False}, "1 setting"),
            (
                {
                    "features.stability-issues": False,
                    "scope.path_labels": "area: shared",
                },
                "2 settings",
            ),
        ):
            translation = tp._answer(changed(self.cfg, changes), self.shape)
            self.answer(translation)
            fixes = [checks.fix_from_json(s["fix"]) for s in translation["stops"]]
            path = self.d / "deploy.yaml"
            for argv, word in (
                (("--out", path), "REFUSED"),
                (("--report-only",), "FAILED"),
            ):
                with self.subTest(count=count, word=word):
                    rc, out, _err = self.run_("--container", self.NAME, *argv)
                    self.assertEqual(rc, 4)
                    lines = out.splitlines()
                    self.assertEqual(
                        lines[: len(translation["report"])], translation["report"]
                    )
                    closing = lines[len(translation["report"]) :]
                    self.assertEqual(
                        closing[0],
                        f"{word}: {count} cannot be carried (each under Stops above, with its fix); nothing "
                        "written",
                    )
                    # the fix's lines (outcome.fix_lines): the label, then further lines under its first word; the
                    # next line after them, wrapped as the outcome's lines are (outcome.finish)
                    n = [x.startswith("next: ") for x in closing].index(True)
                    self.assertTrue(closing[1].startswith(checks.FIX_LABEL))
                    indent = " " * len(checks.FIX_LABEL)
                    self.assertTrue(all(x.startswith(indent) for x in closing[2:n]))
                    self.assertEqual(
                        " ".join(x.strip() for x in closing[1:n]),
                        checks.FIX_LABEL + "; ".join(map(str, fixes)),
                    )
                    typed = (
                        "python3",
                        tp.TOOL,
                        "--container",
                        self.NAME,
                        *map(str, argv),
                    )
                    self.assertEqual(
                        " ".join(closing[n:]), f"next: {checks.Command(typed)}"
                    )
                    self.assertFalse(path.exists())

    def test_a_stop_with_no_fix_the_tool_reads(self):
        """A one-off container's stop whose fix is missing or unreadable is an error: no report, no file."""
        translation = tp._answer(
            changed(self.cfg, {"features.stability-issues": False}), self.shape
        )
        for fix in (DELETE, None, "words alone", {"kind": "edit"}):
            stops = [
                {"found": s["found"]} if fix is DELETE else dict(s, fix=fix)
                for s in translation["stops"]
            ]
            self.answer(dict(translation, stops=stops))
            with self.subTest(fix=fix):
                rc, out, _err = self.run_("--container", self.NAME, "--report-only")
                self.assertEqual(
                    (rc, out.splitlines()),
                    (
                        1,
                        [
                            f"ERROR: the one-off container of {self.NAME}'s image gave a stop with no fix this tool "
                            "reads"
                        ],
                    ),
                )

    def test_out_into_a_git_work_tree_is_refused(self):
        """The refusal's row and the outcome show the file one way, `~/…` under the home folder."""
        repo = self.d / "repo"
        (repo / ".git").mkdir(parents=True)
        path = repo / "deploy.yaml"
        rc, out, _err = self.run_("--container", self.NAME, "--out", path)
        self.assertEqual(rc, 4)
        shown = self.shown(path)
        self.assertIn(f"{shown} is inside the git work tree", " ".join(out.split()))
        self.assertCloses(
            out, f"REFUSED: {shown} is not written (the row above says why)"
        )
        # the refusal as it shows itself: its row, then the line that names the rules
        self.assertIn(f"\n{private_files.RULES}\nREFUSED: ", out)
        self.assertFalse(path.exists())

    def test_a_refusals_fix_names_another_file_after_out(self):
        """The fix of a file --out names that is refused runs the tool again, another file after --out as you typed
        it."""
        repo = self.d / "repo"
        (repo / ".git").mkdir(parents=True)
        path = repo / "deploy.yaml"
        for argv, again in (
            (("--out", str(path)), ("--out", "~/deploy.yaml")),
            ((f"--out={path}",), ("--out=~/deploy.yaml",)),
        ):
            with self.subTest(argv=argv):
                rc, out, _err = self.run_("--container", self.NAME, *argv)
                self.assertEqual(rc, 4)
                command = checks.Command(
                    ("python3", tp.TOOL, "--container", self.NAME, *again)
                )
                self.assertIn(
                    f"{checks.FIX_LABEL}{command}",
                    " ".join(x.strip() for x in out.splitlines()),
                )
                self.assertFalse(path.exists())

    def test_no_docker_on_the_path(self):
        env = dict(self.env, PATH=sandbox.safe_path(self.d / "plain"))
        rc, out, _err = self.run_("--container", self.NAME, "--report-only", env=env)
        self.assertEqual(
            (rc, out.splitlines()),
            (
                3,
                [
                    "FAILED: docker is not on this host's PATH, and the translation runs in a one-off container"
                ],
            ),
        )
        self.assertEqual(self.calls(), [])

    def test_a_container_docker_cannot_inspect(self):
        self.answers.write_text(
            json.dumps([["^container inspect ", "", 1]]), encoding="utf-8"
        )
        rc, out, _err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(
            (rc, out.splitlines()),
            (
                1,
                [f"ERROR: docker cannot inspect the container {self.NAME}: no message"],
            ),
        )
        self.assertEqual(len(self.calls()), 1)

    def test_the_containers_mounts(self):
        self.answer(self.translation, mounts=self.mounts[:1])
        rc, out, _err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(
            (rc, out.splitlines()),
            (
                1,
                [
                    f"ERROR: the container {self.NAME} has no volume or folder at {container_layout.SECRETS_DIR}"
                ],
            ),
        )
        self.assertEqual(len(self.calls()), 1)
        comma = [
            dict(self.mounts[0], Name="l2test-work,dst=/elsewhere"),
            *self.mounts[1:],
        ]
        self.answer(self.translation, mounts=comma)
        rc, out, _err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(rc, 1)
        # the outcome line, wrapped (outcome.finish)
        self.assertEqual(
            " ".join(out.splitlines()),
            f"ERROR: the container {self.NAME}'s mount at {container_layout.WORK_DIR} has a comma in its name, "
            "which docker run's --mount does not take",
        )
        folder = [
            {
                "Type": "bind",
                "Source": "/srv/monitor-work",
                "Destination": container_layout.WORK_DIR,
            },
            *self.mounts[1:],
        ]
        self.answer(self.translation, mounts=folder)
        self.log.unlink()
        rc, out, err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(rc, 0, out + err)
        self.assertIn(
            f" --mount type=bind,src=/srv/monitor-work,dst={container_layout.WORK_DIR},readonly --mount ",
            self.calls()[1],
        )

    def test_a_one_off_container_that_fails(self):
        self.answer("", run_code=125)
        rc, out, _err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(
            (rc, out.splitlines()),
            (1, [f"ERROR: the one-off container of {self.NAME}'s image: no message"]),
        )

    def test_a_one_off_container_killed_names_its_memory_limit(self):
        """A one-off container killed (exit tp.KILLED), as the kernel does at its memory limit: the ERROR names the
        limit and where it is set."""
        self.answer("", run_code=tp.KILLED)
        rc, out, _err = self.run_("--container", self.NAME, "--report-only")
        self.assertEqual(rc, 1)
        self.assertEqual(
            " ".join(out.split()),
            f"ERROR: the one-off container of {self.NAME}'s image: no message; killed (exit 137), as it is when it "
            f"reaches its memory limit, {tp.ONE_OFF_MEMORY} (ONE_OFF_MEMORY in tools/translate_production.py)",
        )

    def test_docker_that_does_not_answer_in_time(self):
        """In process, docker's answers from a stand-in for subprocess.run: inspect, then the one-off container,
        each hanging past its time. The one-off container's error says that nothing changed, as docker made no
        container to remove (tests/test_round_trip.py has the one it made), and the next line runs the command again.
        """
        inspect = json.dumps({"image": self.IMAGE, "mounts": self.mounts})
        again = f"next: python3 tools/translate_production.py --container {self.NAME} --report-only"
        for hangs, seconds, text, after in (
            (
                0,
                tp.INSPECT_SECONDS,
                f"docker cannot inspect the container {self.NAME}",
                [],
            ),
            (
                1,
                tp.RUN_SECONDS,
                f"the one-off container of {self.NAME}'s image",
                [again],
            ),
        ):
            calls = []

            def run(argv, **kwargs):
                calls.append(kwargs["timeout"])
                if len(calls) > hangs:
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                return subprocess.CompletedProcess(argv, 0, inspect, "")

            out = io.StringIO()
            with (
                self.subTest(text=text),
                mock.patch.object(tp.shutil, "which", return_value="docker"),
                mock.patch.object(tp.subprocess, "run", run),
                contextlib.redirect_stdout(out),
            ):
                rc = tp.main(["--container", self.NAME, "--report-only"])
            unchanged = "; nothing changed" if after else ""
            self.assertEqual(
                (rc, out.getvalue().splitlines()),
                (
                    1,
                    [
                        f"ERROR: {text}: no answer in {seconds} seconds{unchanged}",
                        *after,
                    ],
                ),
            )
            self.assertEqual(calls, [tp.INSPECT_SECONDS, tp.RUN_SECONDS][: hangs + 1])

    def test_usage(self):
        for argv in (
            (),
            ("--container", self.NAME),
            ("--report-only",),
            ("--container", self.NAME, "--out", "deploy.yaml", "--report-only"),
        ):
            with self.subTest(argv=argv):
                rc, out, err = self.run_(*argv)
                self.assertEqual((rc, out), (2, ""))
                self.assertIn("usage:", err)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
