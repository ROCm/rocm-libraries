# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The checks across keys (rkmon_lib/coherence.py), on the real schemas:

- registered as the deploy and the build file types' coherence, which validate runs once every key fits the schema
  and no <...> is left; the container's share of this host is not, and validate gives no row of it;
- each check's PASS, WARN and FAIL, with the row it prints, and some rows' whole text as they print;
- the keys Required when a value of another key decides it, the schemas' own, and the keys Required when their feature
  is on that the template leaves blank;
- check 13's rows, validate's, as they print;
- one row for each key, from any mix of checks: a value with a placeholder has the placeholder's row alone, and is
  given, never missing or blank;
- the efforts a model that isn't Claude takes, as the container reads them, and no rejected or unwanted word in a row.
"""

import collections
import copy
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import (  # noqa: E402
    checks,
    coherence,
    container_layout,
    deploy_tree,
    deployment,
    file_types,
    program,
    repository_defaults,
    settings_file,
    verifier_file,
    wrapping,
)
from rkmon_lib.checks import Status  # noqa: E402
from test_settings_schemas import REJECTED, UNWANTED  # noqa: E402

DEPLOY, BUILD = file_types.DEPLOY, file_types.BUILD
# The repository defaults these checks see: an invented repository's, for BASE's repository and folder (check 12).
INVENTED_DEFAULTS = ROOT / "tests" / "data" / "repository"


def setUpModule():
    patch = mock.patch.object(repository_defaults, "FOLDER", INVENTED_DEFAULTS)
    patch.start()
    unittest.addModuleCleanup(patch.stop)


# Each file as the rows show it, in your home folder; no test reads or writes a file there.
DEPLOY_FILE = pathlib.Path.home() / "rkmon-deploy.yaml"
BUILD_FILE = pathlib.Path.home() / "rkmon-build.yaml"
W, WB = settings_file.shown(DEPLOY_FILE), settings_file.shown(BUILD_FILE)
AGAIN = f"rkmon check --settings {W}"
GONE = object()  # a key taken out
SHA = "0123456789abcdef" * 4
RA_KEY = "features.review_automation"
PASSES = RA_KEY + ".passes"
GPU_KEY = RA_KEY + ".remote_gpu_tests"
PRICES_KEY = RA_KEY + ".spend.usd_per_million_tokens"
TEAM_KEY = "features.dashboards.team"
GUARD_KEY = "features.merge_queue.develop_guard"

# A deploy file that fits, with every feature off but the admin dashboard.
BASE = {
    "image": {"tag": "1.0.0"},
    "repository": {
        "name": "my-org/my-repo",
        "folder": "projects/mine",
        "scope_labels": ["mine"],
    },
    "features": {
        "review_automation": {"enabled": False},
        "merge_queue": {"enabled": False},
        "dashboards": {"admin": {"enabled": True}, "team": {"enabled": False}},
        "alerts": {
            "develop_breaks": {"enabled": False},
            "needs_you": {"enabled": False},
        },
        "ticket_creation": {
            "follow_up_tickets": {"enabled": False},
            "found_issue_tickets": {"enabled": False},
        },
        "weekly_report": {"enabled": False},
    },
    "integrations": {"github_cli": {"username": "example-user"}},
}
# Each feature on, with the keys it requires.
RA = {
    "enabled": True,
    "mode": "prototype",
    "remote_gpu_tests": {"enabled": False},
    "spend": {"enabled": False},
}
QUEUE = {
    "enabled": True,
    "mode": "prototype",
    "auto_merge": {"enabled": False},
    "auto_hold": {"enabled": True},
    "queue_notices": {"enabled": True},
    "drift_notices": {"enabled": False},
    "develop_guard": {"enabled": True},
}
TEAM = {
    "enabled": True,
    "mode": "prototype",
    "priority_requests": {"enabled": False},
    "delivery": "serve",
    "serve": {"address": "127.0.0.1"},
}
TEAM_PUBLISH = {
    "enabled": True,
    "mode": "prototype",
    "priority_requests": {"enabled": False},
    "delivery": "publish",
    "public_url": "https://dashboards.example.com/my-team/",
    "publish": {
        "web_host": "web.example.com",
        "username": "web-user",
        "site_folder": "public_html/my-team",
    },
}
GPU = {
    "enabled": True,
    "cluster": {
        "login_host": "login.cluster.example.com",
        "ssh_user": "rkmon-tests",
        "partition": "gpu",
        "accounts": ["team-a"],
        "gpu_types": {"gfx942": "mi300x"},
    },
    "node": {"python": "~/venvs/tests/bin/python"},
}
TICKETS = {
    "features.ticket_creation.mode": "prototype",
    "features.ticket_creation.project_key": "PROJ",
    "integrations.atlassian_cli.jira_url": "https://jira.example.com",
}
OLLAMA = {
    "provider": "ollama",
    "model_id": "llama3",
    "server_url": "http://ollama.example.com:11434",
}
GATEWAY = {
    "provider": "openai_compatible",
    "model_id": "big-model",
    "base_url": "https://llm.example.com/v1",
}
MINE = {"provider": "claude", "model_id": "claude-sonnet-5-5"}
PRICES = {
    "input": 3,
    "output": 15,
    "cache_read": 0.3,
    "cache_write_5_minutes": 3.75,
    "cache_write_1_hour": 6,
}
# A build file that fits, with each download checked.
BUILD_BASE = {
    "rocm": {
        "version": "7.1.0",
        "package_url": "https://example.com/rocm-installer.run",
        "package_sha256": SHA,
    },
    "atlassian_cli": {
        "package_url": "https://example.com/atlassian",
        "package_sha256": SHA,
    },
    "claude_agent_sdk": {
        "package_url": "https://example.com/claude_agent_sdk.whl",
        "package_sha256": SHA,
    },
}
PASS = ("coherence", Status.PASS, "the features' settings fit together", None)
BUILD_PASS = ("coherence", Status.PASS, "the build settings fit together", None)
SKIP = (
    "coherence",
    Status.SKIP,
    "checked once every key fits the schema and no <...> is left",
    None,
)


def changed(changes=None, values=None):
    """BASE (or VALUES) with each dotted key set to its value (GONE: taken out)."""
    values = copy.deepcopy(BASE if values is None else values)
    for dotted, value in (changes or {}).items():
        *parents, last = dotted.split(".")
        cur = values
        for part in parents:
            cur = cur.setdefault(part, {})
        if value is GONE:
            cur.pop(last, None)
        else:
            cur[last] = copy.deepcopy(value)
    return values


def document(values, file_type=DEPLOY):
    path = DEPLOY_FILE if file_type is DEPLOY else BUILD_FILE
    return settings_file.Document(file_type, file_type.version, values, path=path)


def as_shown(rows):
    """Each row as (name, status, found, fix), the fix as it is shown (None for none)."""
    return [
        (r.name, r.status, r.found, None if r.fix is None else str(r.fix)) for r in rows
    ]


def schema_node(path, file_type=DEPLOY):
    """The schema of the key at PATH (a map's entry as "*")."""
    for key, node, _ in settings_file.schema_keys(file_type.load()):
        if key == path:
            return node
    raise KeyError(path)


def to(path, file_type=DEPLOY):
    """What a fix sets the key at PATH to: where its value comes from, its schema's x-source
    (settings_file.source_of; each key these checks fill has one, tested below)."""
    return str(settings_file.source_of(schema_node(path, file_type)))


class Checks(unittest.TestCase):
    maxDiff = None

    def checked(self, changes=None, values=None, file_type=DEPLOY):
        """The rows validate gives the file, which must fit, after its own PASS row: the checks across keys, after the
        row of any value set for a feature that is off."""
        rows = settings_file.validate(document(changed(changes, values), file_type))
        self.assertEqual(
            (rows[0].name, rows[0].status),
            ("settings", Status.PASS),
            checks.render(rows),
        )
        return as_shown(rows[1:])

    def direct(self, changes=None, values=None):
        """coherence.deploy's rows, called on the file itself."""
        return as_shown(coherence.deploy(document(changed(changes, values))))


class TestRegistration(Checks):
    def test_each_file_type_has_its_checks(self):
        self.assertIs(DEPLOY.coherence, coherence.deploy)
        self.assertIs(BUILD.coherence, coherence.build)
        self.assertIsNone(file_types.INTERNAL_STATE.coherence)

    def test_files_that_fit_together_have_one_pass_row(self):
        self.assertEqual(self.checked(), [PASS])
        self.assertEqual(self.direct(), [PASS])
        self.assertEqual(self.checked(values=BUILD_BASE, file_type=BUILD), [BUILD_PASS])

    def test_validate_runs_them_once_every_key_fits(self):
        """A key that does not fit: the checks across keys wait, so the team dashboard's need of the admin dashboard
        has no row yet."""
        team_alone = {
            TEAM_KEY: TEAM,
            "features.dashboards.admin.enabled": False,
        }
        rows = as_shown(
            settings_file.validate(document(changed({**team_alone, "image.tag": 5})))
        )
        self.assertEqual([r[0] for r in rows], ["image.tag", "coherence"])
        self.assertEqual(rows[-1], SKIP)
        self.assertEqual([r[0] for r in self.checked(team_alone)], ["team dashboard"])

    def test_this_hosts_share_is_no_rows_of_validate(self):
        values = changed({"host.resources": {"memory_gib": 512, "cpus": 256}})
        self.assertEqual(self.checked(values=values), [PASS])
        self.assertEqual(
            [r.name for r in coherence.resources(document(values), 64.0, 16)],
            ["memory", "CPUs"],
        )


class TestRequiredWhen(Checks):
    def test_the_keys_a_value_of_another_key_decides_are_the_schemas_own(self):
        """A key's x-required-when names no feature that decides it when a default or a blank (x-blank) stands for
        the key left out, or when no feature reads it (settings_file.required_by): the check across keys does.
        """
        for file_type, table in (
            (DEPLOY, coherence.DEPLOY_REQUIRED),
            (BUILD, coherence.BUILD_REQUIRED),
        ):
            with self.subTest(file=file_type.kind):
                root = file_type.load()
                decided = sorted(
                    path
                    for path, node, _ in settings_file.schema_keys(root)
                    if node.get("x-required-when")
                    and (
                        "default" in node
                        or "x-blank" in node
                        or settings_file._readers(root, path) == ([], [])
                    )
                )
                self.assertEqual(decided, sorted(table))

    def test_the_keys_their_feature_requires_that_the_template_leaves_blank(self):
        """The keys you fill in once you turn their feature on: the template writes them blank or empty."""
        blank = sorted(
            path
            for path, node, _ in settings_file.schema_keys(DEPLOY.load())
            if node.get("x-required-when")
            and "default" not in node
            and "x-blank" not in node
            and node.get("x-template", "?") in ("", [], {})
        )
        cluster, node = GPU_KEY + ".cluster.", GPU_KEY + ".node."
        self.assertEqual(
            blank,
            sorted(
                [
                    cluster + "login_host",
                    cluster + "ssh_user",
                    cluster + "partition",
                    cluster + "accounts",
                    cluster + "gpu_types",
                    node + "python",
                    "features.ticket_creation.project_key",
                    "integrations.atlassian_cli.jira_url",
                ]
            ),
        )

    def test_the_develop_guard_needs_its_labels_while_the_merge_queue_is_live(self):
        live = dict(QUEUE, mode="live")
        why = f'{GUARD_KEY} is on, with features.merge_queue.mode "live"'
        red, stability = GUARD_KEY + ".ci_red_label", GUARD_KEY + ".stability_label"
        self.assertEqual(
            self.checked({"features.merge_queue": live}),
            [
                (
                    red,
                    Status.FAIL,
                    f"missing, and {why}",
                    f"set {red} in {W} to {to(red)}",
                ),
                (
                    stability,
                    Status.FAIL,
                    f"missing, and {why}",
                    f"set {stability} in {W} to {to(stability)}",
                ),
            ],
        )
        blank = dict(live, develop_guard={"enabled": True, "ci_red_label": ""})
        self.assertEqual(
            [r[:3] for r in self.checked({"features.merge_queue": blank})],
            [
                (red, Status.FAIL, f"blank, and {why}"),
                (stability, Status.FAIL, f"missing, and {why}"),
            ],
        )
        labels = {
            "enabled": True,
            "ci_red_label": "ci-red",
            "stability_label": "stable",
        }
        for queue in (dict(live, develop_guard=labels), QUEUE):
            self.assertEqual(self.checked({"features.merge_queue": queue}), [PASS])

    def test_publish_needs_its_keys_and_an_address_for_every_host_needs_public_url(
        self,
    ):
        url = TEAM_KEY + ".public_url"
        publishing = f'{TEAM_KEY}.delivery is "publish"'
        rows = self.checked({TEAM_KEY: dict(TEAM_PUBLISH, public_url="", publish={})})
        self.assertEqual(
            rows,
            [
                (
                    url,
                    Status.FAIL,
                    f"blank, and {publishing}",
                    f"set {url} in {W} to {to(url)}",
                ),
            ]
            + [
                (
                    f"{TEAM_KEY}.publish.{key}",
                    Status.FAIL,
                    f"missing, and {publishing}",
                    f"set {TEAM_KEY}.publish.{key} in {W} to {to(f'{TEAM_KEY}.publish.{key}')}",
                )
                for key in ("web_host", "username", "site_folder")
            ],
        )
        for address in ("0.0.0.0", "::"):
            # in prototype, check 7's own row after it says the page is this host's only
            # (test_served_in_prototype_the_row_says...)
            self.assertEqual(
                self.checked({TEAM_KEY: dict(TEAM, serve={"address": address})}),
                [
                    (
                        url,
                        Status.FAIL,
                        f'missing, and {TEAM_KEY}.serve.address is "{address}"',
                        f"set {url} in {W} to {to(url)}",
                    ),
                    (
                        coherence.TEAM_ROW,
                        Status.PASS,
                        f'in prototype, its page is served on 127.0.0.1 only, not on "{address}" '
                        f"({TEAM_KEY}.serve.address), until it is live",
                        None,
                    ),
                ],
            )
        # live: no such row
        everyone = dict(
            TEAM,
            mode="live",
            serve={"address": "::"},
            public_url="https://dashboards.example.com/my-team/",
        )
        for team in (TEAM, TEAM_PUBLISH, everyone):
            self.assertEqual(self.checked({TEAM_KEY: team}), [PASS])

    def test_a_binding_that_sends_its_key_in_a_header_names_the_header(self):
        header = "model_bindings.gateway.api_key_header"
        for sent_as in ("header", "header_and_bearer"):
            for given, state in ((GONE, "missing"), ("", "blank")):
                with self.subTest(sent_as=sent_as, header=state):
                    gateway = dict(GATEWAY, api_key_sent_as=sent_as)
                    if given is not GONE:
                        gateway["api_key_header"] = given
                    bound = {RA_KEY: RA, "model_bindings": {"gateway": gateway}}
                    self.assertEqual(
                        self.checked(bound),
                        [
                            (
                                header,
                                Status.FAIL,
                                f'{state}, and model_bindings.gateway.api_key_sent_as is "{sent_as}"',
                                f"set {header} in {W} to {to('model_bindings.*.api_key_header')}",
                            )
                        ],
                    )
        for gateway in (
            GATEWAY,  # "bearer", its default
            dict(GATEWAY, api_key_sent_as="header", api_key_header="x-api-key"),
        ):
            self.assertEqual(
                self.checked({RA_KEY: RA, "model_bindings": {"gateway": gateway}}),
                [PASS],
            )

    def test_a_key_its_feature_requires_left_blank_while_the_feature_is_on(self):
        blanks = {
            "cluster": {
                "login_host": "",
                "ssh_user": "",
                "partition": "",
                "accounts": [],
                "gpu_types": {},
            },
            "node": {"python": ""},
        }
        on = f"{GPU_KEY} is on"
        rows = self.checked(
            {RA_KEY: dict(RA, remote_gpu_tests=dict(blanks, enabled=True))}
        )
        self.assertEqual(
            rows,
            [
                (
                    key,
                    Status.FAIL,
                    f"{state}, and {on}",
                    f"set {key} in {W} to {to(key)}",
                )
                for key, state in (
                    (GPU_KEY + ".cluster.login_host", "blank"),
                    (GPU_KEY + ".cluster.ssh_user", "blank"),
                    (GPU_KEY + ".cluster.partition", "blank"),
                    (GPU_KEY + ".cluster.accounts", "empty"),
                    (GPU_KEY + ".cluster.gpu_types", "empty"),
                    (GPU_KEY + ".node.python", "blank"),
                )
            ],
        )
        tickets = {
            **TICKETS,
            RA_KEY: RA,
            "features.ticket_creation.follow_up_tickets.enabled": True,
            "features.ticket_creation.project_key": "",
            "integrations.atlassian_cli.jira_url": "",
        }
        on = "features.ticket_creation.follow_up_tickets is on"
        self.assertEqual(
            self.checked(tickets),
            [
                (key, Status.FAIL, f"blank, and {on}", f"set {key} in {W} to {to(key)}")
                for key in (
                    "features.ticket_creation.project_key",
                    "integrations.atlassian_cli.jira_url",
                )
            ],
        )
        # The template's blanks while the feature is off: nothing reads them.
        self.assertEqual(
            self.checked(
                {RA_KEY: dict(RA, remote_gpu_tests=dict(blanks, enabled=False))}
            ),
            [PASS],
        )

    def test_a_package_url_needs_its_sha256(self):
        sha = "github_cli.package_sha256"
        self.assertEqual(
            self.checked(
                {"github_cli.package_url": "https://example.com/gh.tar.gz"},
                BUILD_BASE,
                BUILD,
            ),
            [
                (
                    sha,
                    Status.FAIL,
                    "missing, and github_cli.package_url is set",
                    f"set {sha} in {WB} to {to(sha, BUILD)}",
                )
            ],
        )
        given = {
            "github_cli": {
                "package_url": "https://example.com/gh.tar.gz",
                "package_sha256": SHA,
            }
        }
        self.assertEqual(self.checked(given, BUILD_BASE, BUILD), [BUILD_PASS])

    def test_each_key_these_checks_fill_says_where_its_value_comes_from(self):
        """One rule for what a fix sets a key to, settings_file's: the key's x-source. Each key a check here fills,
        while a value of another key or its feature requires it, has one; but a key that takes one of a few values
        (a choice, or a switch), which the fix lists."""
        for file_type, table, also in (
            (DEPLOY, coherence.DEPLOY_REQUIRED, ()),
            (BUILD, coherence.BUILD_REQUIRED, ("claude_agent_sdk.package_url",)),
        ):
            filled = set(table) | set(also)
            filled |= {
                path
                for path, node, _ in settings_file.schema_keys(file_type.load())
                if node.get("x-required-when")
                and "default" not in node
                and "x-blank" not in node
            }
            for path in sorted(filled):
                node = schema_node(path, file_type)
                with self.subTest(key=path):
                    if "enum" in node or node.get("type") == "boolean":
                        self.assertNotIn("x-source", node)
                        continue
                    # It reads after "set KEY in FILE to": no capital first, and no full stop.
                    self.assertRegex(node.get("x-source", ""), r"^[a-z].*[^.]$")


class TestDependencies(Checks):
    def test_a_feature_on_while_the_feature_it_needs_is_off(self):
        def row(title, feature, needed, *switches):
            turn_on = ", and ".join(f"set {s}.enabled in {W} to true" for s in switches)
            return (
                title,
                Status.FAIL,
                f"{feature} is on, and it needs {needed}, which is off",
                f"set {feature}.enabled in {W} to false, or turn on what it needs: {turn_on}",
            )

        no_admin = {"features.dashboards.admin.enabled": False}
        for changes, rows in (
            (
                {**TICKETS, "features.ticket_creation.follow_up_tickets.enabled": True},
                [
                    row(
                        "follow-up tickets",
                        "features.ticket_creation.follow_up_tickets",
                        RA_KEY,
                        RA_KEY,
                    )
                ],
            ),
            (
                {"features.merge_queue": dict(QUEUE, drift_notices={"enabled": True})},
                [
                    row(
                        "drift notices",
                        "features.merge_queue.drift_notices",
                        RA_KEY,
                        RA_KEY,
                    )
                ],
            ),
            (
                {
                    "features.alerts.develop_breaks.enabled": True,
                    "features.alerts.mode": "prototype",
                },
                [
                    row(
                        "develop breaks",
                        "features.alerts.develop_breaks",
                        GUARD_KEY,
                        "features.merge_queue",
                        GUARD_KEY,
                    )
                ],
            ),
            (
                {
                    "features.alerts.develop_breaks.enabled": True,
                    "features.alerts.mode": "prototype",
                    "features.merge_queue": dict(
                        QUEUE, develop_guard={"enabled": False}
                    ),
                },
                [
                    row(
                        "develop breaks",
                        "features.alerts.develop_breaks",
                        GUARD_KEY,
                        GUARD_KEY,
                    )
                ],
            ),
            (
                {
                    **TICKETS,
                    **no_admin,
                    "features.ticket_creation.found_issue_tickets.enabled": True,
                },
                [
                    row(
                        "found-issue tickets",
                        "features.ticket_creation.found_issue_tickets",
                        "features.dashboards.admin",
                        "features.dashboards.admin",
                    )
                ],
            ),
            (
                {**no_admin, TEAM_KEY: TEAM},
                [
                    row(
                        "team dashboard",
                        TEAM_KEY,
                        "features.dashboards.admin",
                        "features.dashboards.admin",
                    )
                ],
            ),
        ):
            with self.subTest(rows=[r[0] for r in rows]):
                self.assertEqual(self.checked(changes), rows)

    def test_each_feature_with_what_it_needs(self):
        self.assertEqual(
            self.checked(
                {
                    **TICKETS,
                    RA_KEY: RA,
                    "features.merge_queue": dict(
                        QUEUE, drift_notices={"enabled": True}
                    ),
                    TEAM_KEY: TEAM,
                    "features.alerts.develop_breaks.enabled": True,
                    "features.alerts.mode": "prototype",
                    "features.ticket_creation.follow_up_tickets.enabled": True,
                    "features.ticket_creation.found_issue_tickets.enabled": True,
                }
            ),
            [PASS],
        )


class TestPasses(Checks):
    def test_models_and_efforts_that_fit(self):
        passes = {
            "code_quality_pass": {"model": "mine", "effort": "max"},
            "first_review_pass": {"model": "local", "effort": "high"},
            "ci_triage_pass": {"model": "local", "effort": "default"},
            "notice_draft_pass": {"model": "sonnet"},
        }
        for changes in (
            {RA_KEY: RA},
            {
                RA_KEY: dict(RA, orchestrator={"model": "mine"}, passes=passes),
                "model_bindings": {"mine": MINE, "local": OLLAMA},
            },
            # A Claude binding may take a built-in model's name.
            {
                RA_KEY: RA,
                "model_bindings": {"opus": dict(MINE, model_id="claude-opus-5-5")},
            },
        ):
            self.assertEqual(self.checked(changes), [PASS])

    def test_a_model_that_does_not_exist(self):
        bound = {"mine": MINE, "local": OLLAMA}
        verification, first = (
            PASSES + ".verification_pass.model",
            PASSES + ".first_review_pass.model",
        )
        found = '"mystery" is neither a built-in model nor an entry of model_bindings'
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: dict(
                        RA,
                        passes={
                            "verification_pass": {"model": "mystery"},
                            "first_review_pass": {"model": "mystery"},
                        },
                    ),
                    "model_bindings": bound,
                }
            ),
            [
                (
                    verification,
                    Status.FAIL,
                    found,
                    f'set {verification} in {W} to "opus", "sonnet", "haiku" or "mine"',
                ),
                (
                    first,
                    Status.FAIL,
                    found,
                    f'set {first} in {W} to "default", "opus", "sonnet", "haiku", "mine" or "local"',
                ),
            ],
        )

    def test_a_pass_in_charge_takes_a_claude_model(self):
        key = RA_KEY + ".orchestrator.model"
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: dict(RA, orchestrator={"model": "local"}),
                    "model_bindings": {"local": OLLAMA},
                }
            ),
            [
                (
                    key,
                    Status.FAIL,
                    '"local" is an entry of model_bindings with provider "ollama", and this pass takes a Claude model',
                    f'set {key} in {W} to "opus", "sonnet" or "haiku"',
                )
            ],
        )

    def test_an_effort_a_model_that_isnt_claude_does_not_take(self):
        key = PASSES + ".ci_triage_pass.effort"
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: dict(
                        RA,
                        passes={
                            "ci_triage_pass": {"model": "local", "effort": "xhigh"}
                        },
                    ),
                    "model_bindings": {"local": OLLAMA},
                }
            ),
            [
                (
                    key,
                    Status.FAIL,
                    '"xhigh", and the pass\'s model, "local", has provider "ollama", which takes only "low", "medium" '
                    'or "high"',
                    f'set {key} in {W} to "default", "low", "medium" or "high"',
                )
            ],
        )

    def test_a_built_in_models_name_takes_only_a_claude_binding(self):
        """The binding has the row; a pass that names it still has the built-in model, and no row of its own."""
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: dict(RA, passes={"code_quality_pass": {"model": "sonnet"}}),
                    "model_bindings": {"sonnet": GATEWAY},
                }
            ),
            [
                (
                    "model_bindings.sonnet",
                    Status.FAIL,
                    '"sonnet" is a built-in model\'s name, which takes only a binding with provider "claude", and '
                    'this entry\'s provider is "openai_compatible"',
                    f"rename model_bindings.sonnet in {W}, and each pass that names it, to a name of your own, then "
                    f"run {AGAIN}",
                )
            ],
        )


class TestHeaders(Checks):
    def test_header_and_bearer_with_the_authorization_header(self):
        for prefix, changes in (
            (
                "model_bindings.gateway",
                {
                    "model_bindings": {
                        "gateway": dict(
                            GATEWAY,
                            api_key_sent_as="header_and_bearer",
                            api_key_header="Authorization",
                        )
                    }
                },
            ),
            (
                "integrations.claude_code",
                {
                    "integrations.claude_code": {
                        "api_key_sent_as": "header_and_bearer",
                        "api_key_header": "authorization",
                    }
                },
            ),
        ):
            header, sent_as = prefix + ".api_key_header", prefix + ".api_key_sent_as"
            shown = changed(changes)
            value = (
                shown["model_bindings"]["gateway"]["api_key_header"]
                if "model_bindings" in shown
                else "authorization"
            )
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    self.checked({RA_KEY: RA, **changes}),
                    [
                        (
                            header,
                            Status.FAIL,
                            f'"{value}", and {sent_as} is "header_and_bearer", which sends a bearer token in that '
                            "header too: the key would go twice in one header",
                            f'set {sent_as} in {W} to "bearer", or set {header} in {W} to another header',
                        )
                    ],
                )

    def test_header_and_bearer_with_a_header_of_its_own(self):
        both = {"api_key_sent_as": "header_and_bearer", "api_key_header": "x-api-key"}
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: RA,
                    "integrations.claude_code": both,
                    "model_bindings": {"gateway": dict(GATEWAY, **both)},
                }
            ),
            [PASS],
        )


class TestSpend(Checks):
    def test_a_model_in_use_with_no_price(self):
        spend = {"enabled": True}
        opus = PRICES_KEY + ".claude-opus-5-5"
        words = (
            "one model's prices, in US dollars per million tokens: input, output, cache_read, cache_write_5_minutes "
            "and cache_write_1_hour"
        )
        self.assertEqual(
            self.checked({RA_KEY: dict(RA, spend=spend)}),
            [
                (
                    "spend",
                    Status.WARN,
                    f'{PRICES_KEY} has no entry for claude-opus-5-5 ("opus"), a model the passes use: its cost shows '
                    "as unknown",
                    f"set {opus} in {W} to {words}",
                )
            ],
        )
        # Two names of one model, and a model of a binding.
        passes = {
            "code_quality_pass": {"model": "mine"},
            "first_review_pass": {"model": "local"},
        }
        bound = {"mine": dict(MINE, model_id="claude-opus-5-5"), "local": OLLAMA}
        self.assertEqual(
            self.checked(
                {RA_KEY: dict(RA, spend=spend, passes=passes), "model_bindings": bound}
            ),
            [
                (
                    "spend",
                    Status.WARN,
                    f'{PRICES_KEY} has no entry for claude-opus-5-5 ("opus" and "mine") and llama3 ("local"), '
                    "models the passes use: their cost shows as unknown",
                    f"set {opus} in {W} to {words}, and set {PRICES_KEY}.llama3 in {W} to that model's prices",
                )
            ],
        )

    def test_every_model_in_use_priced_or_spend_off(self):
        priced = {
            "enabled": True,
            "usd_per_million_tokens": {"claude-opus-5-5": PRICES},
        }
        self.assertEqual(self.checked({RA_KEY: dict(RA, spend=priced)}), [PASS])
        self.assertEqual(self.checked({RA_KEY: RA}), [PASS])


class TestTeamDashboard(Checks):
    def test_served_in_prototype_the_row_says_the_page_is_this_host_s_only(self):
        """Check 7: served in prototype at another address, a PASS row of check 7's own says the page
        is on 127.0.0.1 only until it is live, after the other rows, whatever they are: the PASS row, or another
        check's WARN (check 12's, for a repository the package has no defaults for). At 127.0.0.1, or live, there is
        no such row."""
        everywhere = dict(
            TEAM, serve={"address": "0.0.0.0"}, public_url="https://example.com/team"
        )
        row = (
            coherence.TEAM_ROW,
            Status.PASS,
            'in prototype, its page is served on 127.0.0.1 only, not on "0.0.0.0" '
            f"({TEAM_KEY}.serve.address), until it is live",
            None,
        )
        self.assertEqual(self.direct({TEAM_KEY: everywhere}), [PASS, row])
        self.assertEqual(self.checked({TEAM_KEY: everywhere}), [PASS, row])
        no_defaults = {TEAM_KEY: everywhere, "repository.name": "other-org/other-repo"}
        self.assertEqual(
            [r[:2] for r in self.checked(no_defaults)],
            [("repository defaults", Status.WARN), row[:2]],
        )
        self.assertEqual(self.checked(no_defaults)[-1], row)
        self.assertEqual(self.direct({TEAM_KEY: TEAM}), [PASS])
        self.assertEqual(self.direct({TEAM_KEY: dict(everywhere, mode="live")}), [PASS])

    def test_each_delivery_and_its_block(self):
        delivery = TEAM_KEY + ".delivery"
        for team, row in (
            (
                dict(TEAM_PUBLISH, serve={"address": "127.0.0.1"}),
                (
                    f'{TEAM_KEY}.serve is set, and {delivery} "publish" takes no serve block',
                    f'delete {TEAM_KEY}.serve from {W}, or set {delivery} in {W} to "serve"',
                ),
            ),
            (
                dict(TEAM, publish={"web_host": ""}),
                (
                    f'{TEAM_KEY}.publish is set, and {delivery} "serve" takes no publish block',
                    f'delete {TEAM_KEY}.publish from {W}, or set {delivery} in {W} to "publish"',
                ),
            ),
        ):
            with self.subTest(found=row[0]):
                self.assertEqual(
                    self.checked({TEAM_KEY: team}),
                    [("team dashboard", Status.FAIL, *row)],
                )
        # A serve block left out means its defaults: 127.0.0.1, port 18615.
        no_serve = {k: v for k, v in TEAM.items() if k != "serve"}
        for team in (TEAM, TEAM_PUBLISH, no_serve):
            self.assertEqual(self.checked({TEAM_KEY: team}), [PASS])

    def test_with_the_box_on_publish_takes_an_address_the_request_service_serves(
        self,
    ):
        """Check 7: with priority requests on and "publish", the web host's request service serves
        the page over plain http, at the root of its own port: public_url gives http, a port of 1024 or more, and no
        path, else the confirm fails, the fix naming the address to give (for an address with a path, that address at
        the root). A blank one has the Required when row alone. With the box off, or "serve", any address does.
        """
        url = TEAM_KEY + ".public_url"
        box = {"enabled": True}
        for given, found, give in (
            (
                "http://dashboards.example.com:8080/my-team/",
                f'{url} has the path "/my-team/", and the web host\'s request service serves the page at the root of '
                "its own port, http://dashboards.example.com:8080/",
                '"http://dashboards.example.com:8080/"',
            ),
            (
                "https://dashboards.example.com/my-team/",
                f"{url} is not an http:// address with a port from 1024 to 65535, such as "
                "http://dashboards.example.com:8080/: the web host's request service serves the page itself, over "
                "plain http, on a port its login may open",
                "an http address with a port of 1024 or more and no path, such as "
                '"http://dashboards.example.com:8080/"',
            ),
            (
                "http://web_1.example.com:8080/",
                f'the host of {url} is neither an IP address nor a name of letters, digits, "." and "-"',
                "an http address with a port of 1024 or more and no path, such as "
                '"http://dashboards.example.com:8080/"',
            ),
        ):
            with self.subTest(given=given):
                team = dict(TEAM_PUBLISH, priority_requests=box, public_url=given)
                self.assertEqual(
                    self.checked({TEAM_KEY: team}),
                    [
                        (
                            "team dashboard",
                            Status.FAIL,
                            found,
                            f"set {url} in {W} to {give}",
                        )
                    ],
                )
        blank = dict(TEAM_PUBLISH, priority_requests=box, public_url="")
        self.assertEqual(
            self.checked({TEAM_KEY: blank}),
            [
                (
                    url,
                    Status.FAIL,
                    f'blank, and {TEAM_KEY}.delivery is "publish"',
                    f"set {url} in {W} to {to(url)}",
                )
            ],
        )
        for team in (
            dict(
                TEAM_PUBLISH,
                priority_requests=box,
                public_url="http://dashboards.example.com:8080/",
            ),
            dict(
                TEAM_PUBLISH, priority_requests=box, public_url="http://192.0.2.10:8080"
            ),
            dict(
                TEAM_PUBLISH, public_url="http://dashboards.example.com:8080/my-team/"
            ),
            dict(TEAM, priority_requests=box, public_url="https://example.com/team/"),
        ):
            with self.subTest(team=team):
                self.assertEqual(self.checked({TEAM_KEY: team}), [PASS])


class TestUsers(Checks):
    UID, GID = "host.users.monitor_uid", "host.users.monitor_gid"

    def test_the_system_nobody_users_id(self):
        self.assertEqual(
            self.checked({self.UID: 65534}),
            [
                (
                    self.UID,
                    Status.FAIL,
                    "65534 is a system account's id (below 1000, or nobody's 65534), which the container refuses at "
                    "each start",
                    f"set {self.UID} in {W} to an id of 1000 or more, or to null, which means your own user id "
                    "on this host",
                )
            ],
        )
        for uid in (1000, 65533, 65535, 2147483647, None):
            with self.subTest(uid=uid):
                self.assertEqual(self.checked({self.UID: uid}), [PASS])

    def test_the_containers_own_users_ids_and_the_pr_runners_groups(self):
        """The PR runners' ids and the GitHub writer's are the package's (container_layout), and the entry point gives
        neither to the monitor; nor the PR runners' group, nor a runner's own group, whose members would read the
        monitor's files. The rows word the entry point's reasons (docker/rk_ids.py clashes).
        """
        runners = sorted(u.uid for u in container_layout.PR_RUNNERS)
        # The rows name the runners' ids as a range.
        self.assertEqual(runners, list(range(runners[0], runners[-1] + 1)))
        span = f"{runners[0]} to {runners[-1]}, the package's"
        for uid, why in (
            (runners[0], f"a PR runner's id ({span})"),
            (container_layout.PR_RUNNER.uid, f"a PR runner's id ({span})"),
            (container_layout.WRITER.uid, "the GitHub writer's id"),
        ):
            with self.subTest(uid=uid):
                self.assertEqual(
                    self.checked({self.UID: uid}),
                    [
                        (
                            self.UID,
                            Status.FAIL,
                            f"{uid} is {why}, which the container refuses at each start",
                            f"set {self.UID} in {W} to another id, or to null, which means your own user id on "
                            "this host",
                        )
                    ],
                )
        gids = sorted(u.gid for u in container_layout.PR_RUNNERS)
        self.assertEqual(gids, list(range(gids[0], gids[-1] + 1)))
        own = f"a PR runner's own group's id ({gids[0]} to {gids[-1]}, the package's)"
        for gid, why in (
            (container_layout.PR_GID, "the PR runners' group's id"),
            (gids[0], own),
            (gids[-1], own),
        ):
            with self.subTest(gid=gid):
                self.assertEqual(
                    self.checked({self.UID: 1000, self.GID: gid}),
                    [
                        (
                            self.GID,
                            Status.FAIL,
                            f"{gid} is {why}, which the container refuses at each start",
                            f"set {self.GID} in {W} to another group's id, or to null, which means your own "
                            "group id on this host",
                        )
                    ],
                )

    def test_the_settings_refuse_the_ids_the_entry_point_refuses(self):
        """Every id docker/rk_ids.py refuses at each start, the settings refuse (the schema's bounds, or check 8), and
        no other. These settings set the monitor's ids alone: the writer's and the PR runners' are the package's, as
        the entry point has them when nothing sets them."""
        ids = deployment.rk_ids()
        have = {
            "uid": 1000,
            "gid": 1000,
            "write_uid": container_layout.WRITER.uid,
        }
        own = {u.uid for u in container_layout.PR_RUNNERS} | {
            container_layout.WRITER.uid
        }
        gids = {u.gid for u in container_layout.PR_RUNNERS}
        around = sorted(
            {
                *range(min(own) - 2, max(own) + 3),
                *range(min(gids) - 2, max(gids) + 3),
                1,
                999,
                1000,
                65533,
                65534,
                65535,
                2**31 - 1,
            }
        )
        for key, var, values in (
            (self.UID, "RKMON_UID", {}),
            (self.GID, "RKMON_GID", {self.UID: 1000}),
        ):
            for value in around:
                with self.subTest(key=key, value=value):
                    try:
                        ids.wanted({var: str(value)}, have)
                        entry = False
                    except ids.Refused:
                        entry = True
                    rows = settings_file.validate(
                        document(changed({**values, key: value}))
                    )
                    refused = any(r.status == Status.FAIL for r in rows)
                    self.assertEqual(refused, entry, as_shown(rows))

    def test_any_group_a_system_groups_included(self):
        """The monitor's group may be any group's id, as the entry point takes it: a system group's too."""
        for gid in (1, 100, 999, 65534, 2147483647, None):
            with self.subTest(gid=gid):
                self.assertEqual(self.checked({self.UID: 1000, self.GID: gid}), [PASS])

    def test_an_id_the_entry_point_refuses_is_validates_row(self):
        for key, value, side, low in (
            (self.UID, 999, "less than 1000", 1000),
            (self.UID, 2**31, "more than 2147483647", 1000),
            (self.GID, 0, "less than 1", 1),
            (self.GID, 2**31, "more than 2147483647", 1),
        ):
            with self.subTest(key=key, value=value):
                rows = settings_file.validate(document(changed({key: value})))
                self.assertEqual(
                    as_shown(rows),
                    [
                        (
                            key,
                            Status.FAIL,
                            f"{value} is {side}",
                            f"set {key} in {W} to a number from {low} to 2147483647",
                        ),
                        SKIP,
                    ],
                )

    def test_the_schemas_ids_are_the_entry_points(self):
        """The ids the schema takes are the ones docker/rk_ids.py maps at each start, the one home of their rules
        (the image's own copy runs as root, and imports nothing of the package): a user id from FIRST_ID to MAX_ID,
        NOBODY refused (check 8, which reads it there), and any group's id from 1 to MAX_ID.
        """
        ids = deployment.rk_ids()
        uid, gid = schema_node(self.UID), schema_node(self.GID)
        self.assertEqual((uid["minimum"], uid["maximum"]), (ids.FIRST_ID, ids.MAX_ID))
        self.assertEqual((gid["minimum"], gid["maximum"]), (1, ids.MAX_ID))
        self.assertIn(f" {ids.NOBODY}, ", uid["description"])
        self.assertTrue(ids.system_id(ids.NOBODY))
        self.assertFalse(ids.system_id(ids.FIRST_ID))


class TestLoopback(Checks):
    def test_an_address_on_the_loopback(self):
        def row(key, url):
            return (
                key,
                Status.WARN,
                f'"{url}" is on the loopback: inside the container, localhost is the container itself',
                f"set {key} in {W} to an address the container can reach, such as this host's own address",
            )

        bound = {
            "local": dict(OLLAMA, server_url="http://127.0.0.1:11434"),
            "v6": dict(GATEWAY, base_url="http://[::1]:8000/v1"),
            "named": dict(GATEWAY, base_url="http://models.localhost:8000/v1"),
            "away": OLLAMA,
            "lan": dict(GATEWAY, base_url="http://192.0.2.10:8000/v1"),
        }
        self.assertEqual(
            self.checked(
                {
                    RA_KEY: RA,
                    "integrations.claude_code.base_url": "http://localhost:4000",
                    "model_bindings": bound,
                }
            ),
            [
                row("integrations.claude_code.base_url", "http://localhost:4000"),
                row("model_bindings.local.server_url", "http://127.0.0.1:11434"),
                row("model_bindings.v6.base_url", "http://[::1]:8000/v1"),
                row("model_bindings.named.base_url", "http://models.localhost:8000/v1"),
            ],
        )


class TestRepositoryDefaults(Checks):
    """Check 12: repository defaults for repository.name and repository.folder, else one WARN naming the keys that
    take the package's generic values, and where to see each value the monitor runs with.
    """

    SHOW = f"{program.name()} rk config show --source shows each value the monitor runs with and where it comes from"

    def test_another_folder_or_repository_takes_the_generic_values(self):
        for name, folder in (
            ("my-org/my-repo", "projects/other"),
            ("my-org/other-repo", "projects/mine"),
        ):
            with self.subTest(name=name, folder=folder):
                self.assertEqual(
                    self.checked(
                        {"repository.name": name, "repository.folder": folder}
                    ),
                    [
                        (
                            "repository defaults",
                            Status.WARN,
                            f"none for {name} and its folder {folder}: the package's generic values apply to "
                            "ci_absent_modules and develop_rocke_job_pattern",
                            self.SHOW,
                        )
                    ],
                )

    def test_the_repository_whatever_its_case_and_the_folder_whatever_its_slashes(self):
        self.assertEqual(self.checked({"repository.name": "My-Org/My-Repo"}), [PASS])
        self.assertEqual(self.direct({"repository.folder": "/projects/mine/"}), [PASS])

    def test_a_placeholder_has_its_own_row_alone(self):
        rows = self.direct({"repository.name": "<org>/<repo>"})
        self.assertNotIn("repository defaults", [r[0] for r in rows])


class TestRemoteGpuTests(Checks):
    C, N = GPU_KEY + ".cluster.", GPU_KEY + ".node."
    SLURM = "letters, digits, '.', ',', '&', '|', '!', '-' and '_'"

    def gpu(self, cluster=None, node=None, enabled=True):
        """The changes that turn review automation and the remote GPU tests on (or the tests off), with CLUSTER's
        and NODE's values over GPU's own."""
        tests = dict(
            GPU,
            enabled=enabled,
            cluster=dict(GPU["cluster"], **(cluster or {})),
            node=dict(GPU["node"], **(node or {})),
        )
        return {RA_KEY: dict(RA, remote_gpu_tests=tests)}

    def validated(self, changes):
        return as_shown(settings_file.validate(document(changed(changes))))

    def test_the_default_target_is_one_of_the_gpu_types(self):
        """Check 17: what crosses keys, while the tests are on."""
        key = self.C + "default_target"
        types = {"gfx942": "mi300x", "gfx90a": "mi250"}
        self.assertEqual(
            self.checked(self.gpu({"gpu_types": types, "default_target": "gfx1100"})),
            [
                (
                    key,
                    Status.FAIL,
                    f'"gfx1100" is not in {self.C}gpu_types',
                    f'set {key} in {W} to "gfx942" or "gfx90a", or to "" to use the first entry in gpu_types',
                )
            ],
        )
        for target in ("gfx90a", ""):
            with self.subTest(target=target):
                cluster = {"gpu_types": types, "default_target": target}
                self.assertEqual(self.checked(self.gpu(cluster)), [PASS])
        off = self.gpu({"default_target": "gfx1100"}, enabled=False)
        self.assertEqual(self.direct(off), [PASS])

    def test_validate_refuses_an_ssh_option_as_the_login_host(self):
        key = self.C + "login_host"
        self.assertEqual(
            self.validated(self.gpu({"login_host": "-oProxyCommand=x"})),
            [
                (
                    key,
                    Status.FAIL,
                    '"-oProxyCommand=x" is not in the form this key takes',
                    f"set {key} in {W} to a value of this form: The host name of the cluster's login node, where the "
                    "monitor submits jobs: letters, digits, '.', '-' and '_', starting with a letter or a digit, with "
                    'no user@ and no port. Example: "login.cluster.example.com".',
                ),
                SKIP,
            ],
        )

    def test_a_value_out_of_its_form_is_validates_row_which_states_the_form(self):
        """Each value's own form is its key's schema pattern, the one rkmon_lib/gpu.py checks
        (tests/test_settings_schemas.py holds the two the same): validate's row, whose fix quotes the description,
        which states the form; a list's item and a map's entry, its list's or map's."""
        for section, name, value, at, shown, form in (
            (
                "cluster",
                "login_host",
                "user@login",
                "login_host",
                '"user@login"',
                "no user@ and no port",
            ),
            (
                "cluster",
                "ssh_user",
                "9user",
                "ssh_user",
                '"9user"',
                "starting with a letter or '_'",
            ),
            (
                "cluster",
                "partition",
                "gpu queue",
                "partition",
                '"gpu queue"',
                self.SLURM,
            ),
            ("cluster", "qos", "high priority", "qos", '"high priority"', self.SLURM),
            ("cluster", "constraint", "mi300 x", "constraint", '"mi300 x"', self.SLURM),
            (
                "cluster",
                "time_limit",
                "90:00",
                "time_limit",
                '"90:00"',
                '"HH:MM:SS", "D-HH:MM:SS" or minutes',
            ),
            (
                "cluster",
                "accounts",
                ["team-a", "team b"],
                "accounts[1]",
                '"team b"',
                "Each is letters, digits, '.', '-' and '_'",
            ),
            (
                "cluster",
                "excluded_nodes",
                ["node 1"],
                "excluded_nodes[0]",
                '"node 1"',
                "letters, digits, '.', ',', '[', ']', '-' and '_'",
            ),
            (
                "cluster",
                "gpu_types",
                {"gfx942": "mi/300"},
                "gpu_types.gfx942",
                '"mi/300"',
                "letters, digits, '.', ':', '-' and '_'",
            ),
            (
                "cluster",
                "default_target",
                "GFX942",
                "default_target",
                '"GFX942"',
                "one of gpu_types' targets",
            ),
            (
                "node",
                "work_folder",
                "rkmon\ngpu",
                "work_folder",
                '"rkmon\\ngpu"',
                "on one line",
            ),
            (
                "node",
                "python",
                "bin/python\n",
                "python",
                '"bin/python\\n"',
                "on one line",
            ),
            (
                "node",
                "setup_command",
                "module load rocm\nsource env",
                "setup_command",
                '"module load rocm\\nsource env"',
                "on one line",
            ),
            (
                "node",
                "environment",
                {"EXTRA": "a\nb"},
                "environment.EXTRA",
                '"a\\nb"',
                "each value on one line",
            ),
            (
                "node",
                "clone_url",
                "https://git.example.com/my repo.git",
                "clone_url",
                '"https://git.example.com/my repo.git"',
                "an address with no spaces",
            ),
        ):
            with self.subTest(key=name):
                key = f"{GPU_KEY}.{section}.{at}"
                own = schema_node(f"{GPU_KEY}.{section}.{name}")
                described = settings_file.closing_of(own["description"])[0]
                self.assertIn(form, described)
                fix = f"set {key} in {W} to a value of this form: {described}"
                changes = self.gpu(**{section: {name: value}})
                self.assertEqual(
                    self.validated(changes),
                    [
                        (
                            key,
                            Status.FAIL,
                            f"{shown} is not in the form this key takes",
                            fix,
                        ),
                        SKIP,
                    ],
                )

    def test_a_name_out_of_its_maps_form_is_validates_row_which_states_the_form(self):
        for section, name, entry, form in (
            (
                "cluster",
                "gpu_types",
                {"GFX942": "mi300x"},
                '"gfx", then lowercase letters and digits',
            ),
            (
                "node",
                "environment",
                {"9LIVES": "1"},
                "each NAME letters, digits and '_', starting with a letter or '_'",
            ),
        ):
            with self.subTest(key=name):
                path = f"{GPU_KEY}.{section}.{name}"
                (bad,) = entry
                description = schema_node(path)["description"]
                described = settings_file.closing_of(description)[0]
                self.assertIn(form, described)
                takes = f"a name {path} takes ({described.rstrip('.')})"
                self.assertEqual(
                    self.validated(self.gpu(**{section: {name: entry}})),
                    [
                        (
                            f"{path}.{bad}",
                            Status.FAIL,
                            f'"{bad}" is not a name {path} takes',
                            f"rename {path}.{bad} in {W} to {takes}, then run {AGAIN}",
                        ),
                        SKIP,
                    ],
                )

    def test_a_carriage_return_or_a_nul_is_not_one_line(self):
        """The one-line values refuse a carriage return and a NUL, as they refuse a line break."""
        for section, name, written, at in (
            ("node", "work_folder", str, "work_folder"),
            ("node", "python", str, "python"),
            ("node", "setup_command", str, "setup_command"),
            ("node", "environment", lambda v: {"EXTRA": v}, "environment.EXTRA"),
            ("node", "clone_url", str, "clone_url"),
        ):
            for value, shown in (
                ("module\rload", '"module\\rload"'),
                ("module\x00load", '"module\\u0000load"'),
            ):
                with self.subTest(key=name, value=shown):
                    changes = self.gpu(**{section: {name: written(value)}})
                    self.assertEqual(
                        [row[:3] for row in self.validated(changes)],
                        [
                            (
                                f"{GPU_KEY}.{section}.{at}",
                                Status.FAIL,
                                f"{shown} is not in the form this key takes",
                            ),
                            SKIP[:3],
                        ],
                    )

    def test_values_in_their_forms(self):
        cluster = {
            "login_host": "9login.example.com",
            "ssh_user": "_rkmon.tests-1",
            "partition": "gpu",
            "qos": "normal",
            "constraint": "mi300x&ib|!old,new",
            "time_limit": "1-02:00:00",
            "accounts": ["team-a", "team_b.1"],
            "excluded_nodes": ["node[01-04],node7"],
            "gpu_types": {"gfx942": "mi300x:8", "gfx90a": "mi250.x-1"},
            "default_target": "gfx90a",
        }
        node = {
            "work_folder": "/shared/rkmon gpu",
            "setup_command": "module load rocm; export X=1",
            "environment": {"HIP_VISIBLE_DEVICES": "0", "_X1": "a b"},
            "clone_url": "https://git.example.com/my-org/my-repo.git",
        }
        for time_limit in ("1-02:00:00", "01:00:00", "90"):
            with self.subTest(time_limit=time_limit):
                changes = self.gpu(dict(cluster, time_limit=time_limit), node)
                self.assertEqual(self.checked(changes), [PASS])


class TestSharedLabels(Checks):
    def test_a_label_in_both(self):
        self.assertEqual(
            self.checked(
                {
                    "repository.scope_labels": ["mine", "needs-ci"],
                    "repository.shared_label_folders": {"needs-ci": "projects/shared"},
                }
            ),
            [
                (
                    "shared labels",
                    Status.FAIL,
                    '"needs-ci" is in both repository.scope_labels and repository.shared_label_folders: the first '
                    "puts every pull request with it in scope, the second only some",
                    f'set repository.scope_labels in {W} to the list without "needs-ci", or delete '
                    f"repository.shared_label_folders.needs-ci from {W}",
                )
            ],
        )

    def test_every_scope_label_in_both(self):
        """The scope keeps a label of its own: the fix deletes the folders' entries."""
        self.assertEqual(
            self.checked(
                {
                    "repository.scope_labels": ["a", "b"],
                    "repository.shared_label_folders": {
                        "a": "projects/a",
                        "b": "projects/b",
                    },
                }
            ),
            [
                (
                    "shared labels",
                    Status.FAIL,
                    '"a" and "b" are in both repository.scope_labels and repository.shared_label_folders: the first '
                    "puts every pull request with one of them in scope, the second only some",
                    f"delete repository.shared_label_folders.a from {W}, and delete repository.shared_label_folders.b "
                    f"from {W}",
                )
            ],
        )

    def test_a_label_in_the_folders_only(self):
        changes = {"repository.shared_label_folders": {"needs-ci": "projects/shared"}}
        self.assertEqual(self.checked(changes), [PASS])


class TestThisHostsShare(Checks):
    def rows(self, resources, host_memory_gib=0.0, host_cpus=0, ra=True):
        values = changed({"host.resources": resources, **({RA_KEY: RA} if ra else {})})
        return as_shown(
            coherence.resources(document(values), host_memory_gib, host_cpus)
        )

    def test_the_least_the_container_takes(self):
        """coherence.least, the one home of checks 8's and 9's numbers: three reviews and one build need 6.5 GiB of
        RAM disk at their first estimates, about 7; a RAM disk of 4 GiB takes a memory limit of 12, beside the work's 8;
        and each review and build takes a CPU."""
        self.assertEqual(coherence.least(4, 3, 1), coherence.Least(6.5, 7, 12, 4))
        self.assertEqual(coherence.least(), coherence.Least(0, 0, 8, 0))

    def test_a_ram_disk_smaller_than_the_starting_reviews_and_builds_need(self):
        """The whole row as it prints, while the memory holds a RAM disk that size beside the work: three reviews and
        one build, at first."""
        values = changed(
            {RA_KEY: RA, "host.resources": {"ramdisk_gib": 4, "memory_gib": 16}}
        )
        self.assertEqual(
            checks.render(coherence.resources(document(values))),
            "  WARN  RAM disk: host.resources.ramdisk_gib 4 is less than the starting reviews and builds need (about 7 "
            "GiB)\n"
            "        fix: set host.resources.ramdisk_gib in ~/rkmon-deploy.yaml to 7 or more, or to 0 for no RAM disk",
        )
        # Without review automation, the one build fits.
        self.assertEqual(self.rows({"ramdisk_gib": 4, "memory_gib": 12}, ra=False), [])
        self.assertEqual(self.rows({"ramdisk_gib": 7, "memory_gib": 15}), [])

    def test_a_ram_disk_a_gib_short_of_a_whole_need(self):
        """Two reviews and two builds need 7 GiB, just: a RAM disk of 6 is short of it, and one of 7 is not."""
        for ramdisk, rows in (
            (
                6,
                [
                    (
                        "RAM disk",
                        Status.WARN,
                        "host.resources.ramdisk_gib 6 is less than the starting reviews and builds need (about 7 GiB)",
                        f"set host.resources.ramdisk_gib in {W} to 7 or more, or to 0 for no RAM disk",
                    )
                ],
            ),
            (7, []),
        ):
            with self.subTest(ramdisk=ramdisk):
                values = changed(
                    {
                        RA_KEY: dict(RA, parallel_reviews=2),
                        "monitor.parallel_builds": 2,
                        "host.resources": {"ramdisk_gib": ramdisk, "memory_gib": 16},
                    }
                )
                self.assertEqual(as_shown(coherence.resources(document(values))), rows)

    def test_a_larger_ram_disk_comes_with_the_memory_to_hold_it(self):
        """Where a RAM disk that size would not fit memory_gib, its fix raises memory_gib too, and the memory's own fix
        offers no RAM disk the RAM disk's row warns of: following either fix passes, on this 64 GiB host.
        """
        memory, ramdisk = "host.resources.memory_gib", "host.resources.ramdisk_gib"
        warn = (
            "RAM disk",
            Status.WARN,
            f"{ramdisk} 4 is less than the starting reviews and builds need (about 7 GiB)",
            f"set {ramdisk} in {W} to 7 or more, and set {memory} in {W} to 15 or more; or set {ramdisk} in {W} to "
            "0 for no RAM disk",
        )
        self.assertEqual(
            self.rows({"ramdisk_gib": 4, "memory_gib": 12}, 64.0),
            [
                warn,
                (
                    "memory",
                    Status.PASS,
                    "the container may use 12 of this host's 64 GiB",
                    None,
                ),
            ],
        )
        self.assertEqual(
            self.rows({"ramdisk_gib": 4, "memory_gib": 10}, 64.0),
            [
                warn,
                (
                    "memory",
                    Status.FAIL,
                    f"{memory} 10 is less than {ramdisk} (4) plus the 8 GiB the work needs",
                    f"set {memory} in {W} to 12 or more, or set {ramdisk} in {W} to 0 for no RAM disk",
                ),
            ],
        )
        self.assertEqual(self.rows({"ramdisk_gib": 4, "memory_gib": 12}), [warn])
        for resources in (
            {"ramdisk_gib": 7, "memory_gib": 15},
            {"ramdisk_gib": 0, "memory_gib": 10},
        ):
            with self.subTest(resources=resources):
                rows = self.rows(resources, 64.0)
                self.assertEqual([r[:2] for r in rows], [("memory", Status.PASS)])
        # A host with no room for the RAM disk the work needs: both fixes offer none.
        for resources, host, fix in (
            (
                {"ramdisk_gib": 6, "memory_gib": 32},
                12.0,
                f"set {memory} in {W} to 12 or less, and set {ramdisk} in {W} to 0 for no RAM disk",
            ),
            (
                {"ramdisk_gib": 4, "memory_gib": 6},
                10.0,
                f"set {memory} in {W} to 10, and set {ramdisk} in {W} to 0 for no RAM disk",
            ),
        ):
            with self.subTest(resources=resources, host=host):
                self.assertEqual(
                    [(r[0], r[3]) for r in self.rows(resources, host)],
                    [
                        ("RAM disk", f"set {ramdisk} in {W} to 0 for no RAM disk"),
                        ("memory", fix),
                    ],
                )

    def test_a_ram_disk_on_a_host_with_no_room_for_one_that_size(self):
        """A RAM disk of 7 GiB and the work's 8 need 15 GiB: on a host with less, the fix is no RAM disk."""
        memory, ramdisk = "host.resources.memory_gib", "host.resources.ramdisk_gib"
        found = f"{ramdisk} 4 is less than the starting reviews and builds need (about 7 GiB)"
        self.assertEqual(
            self.rows({"ramdisk_gib": 4, "memory_gib": 12}, 12.0),
            [
                (
                    "RAM disk",
                    Status.WARN,
                    found,
                    f"set {ramdisk} in {W} to 0 for no RAM disk",
                ),
                (
                    "memory",
                    Status.PASS,
                    "the container may use 12 of this host's 12 GiB",
                    None,
                ),
            ],
        )
        for resources, fix in (
            (
                {"ramdisk_gib": 4, "memory_gib": 15},
                f"set {ramdisk} in {W} to 7 or more, or to 0 for no RAM disk",
            ),
            (
                {"ramdisk_gib": 4, "memory_gib": 12},
                f"set {ramdisk} in {W} to 7 or more, and set {memory} in {W} to 15 or more; or set {ramdisk} in {W} "
                "to 0 for no RAM disk",
            ),
        ):
            with self.subTest(resources=resources):
                self.assertEqual(
                    self.rows(resources, 15.0)[0],
                    ("RAM disk", Status.WARN, found, fix),
                )

    def test_the_memory_on_this_host(self):
        self.assertEqual(
            self.rows({}, 76.0),
            [
                (
                    "memory",
                    Status.PASS,
                    "the container may use 8 of this host's 76 GiB",
                    None,
                )
            ],
        )
        self.assertEqual(self.rows({}), [])  # this host's memory not known
        memory, ramdisk = "host.resources.memory_gib", "host.resources.ramdisk_gib"
        for resources, host, found, fix in (
            (
                {"memory_gib": 16},
                12.0,
                f"{memory} 16 is more than this host has (12 GiB)",
                f"set {memory} in {W} to 12 or less",
            ),
            (
                {"memory_gib": 16, "ramdisk_gib": 6},
                12.0,
                f"{memory} 16 is more than this host has (12 GiB)",
                f"set {memory} in {W} to 12 or less, and set {ramdisk} in {W} to 4 or less",
            ),
            (
                {"memory_gib": 10, "ramdisk_gib": 4},
                64.0,
                f"{memory} 10 is less than {ramdisk} (4) plus the 8 GiB the work needs",
                f"set {memory} in {W} to 12 or more, or set {ramdisk} in {W} to 2 or less",
            ),
            (
                {"memory_gib": 8, "ramdisk_gib": 4},
                0.0,
                f"{memory} 8 is less than {ramdisk} (4) plus the 8 GiB the work needs",
                f"set {memory} in {W} to 12 or more, or set {ramdisk} in {W} to 0 for no RAM disk",
            ),
            (
                {"memory_gib": 6, "ramdisk_gib": 4},
                10.0,
                f"{memory} 6 is less than {ramdisk} (4) plus the 8 GiB the work needs",
                f"set {memory} in {W} to 10, and set {ramdisk} in {W} to 2 or less",
            ),
            (
                {"memory_gib": 6},
                0.0,
                f"{memory} 6 is less than the 8 GiB the work needs",
                f"set {memory} in {W} to 8 or more",
            ),
            (
                {},
                6.0,
                f"{memory} 8 is more than this host has (6 GiB), and the work needs 8 GiB at least",
                f"run {AGAIN} on a host with 8 GiB or more",
            ),
            (
                {"memory_gib": 6},
                6.0,
                f"{memory} 6 is less than the 8 GiB the work needs, and this host has 6 GiB",
                f"run {AGAIN} on a host with 8 GiB or more",
            ),
        ):
            with self.subTest(resources=resources, host=host):
                self.assertEqual(
                    self.rows(dict(resources, cpus=1), host, ra=False),
                    [("memory", Status.FAIL, found, fix)],
                )

    def test_the_cpus_on_this_host(self):
        cpus = "host.resources.cpus"
        self.assertEqual(
            self.rows({"cpus": 8}, 0.0, 4),
            [
                (
                    "CPUs",
                    Status.FAIL,
                    f"{cpus} 8 is more than this host has (4)",
                    f"set {cpus} in {W} to 4 or fewer",
                )
            ],
        )
        fewer = (
            f"{cpus} 2 is fewer than the reviews and builds that may run at once (4)"
        )
        self.assertEqual(
            self.rows({"cpus": 2}, 0.0, 64),
            [("CPUs", Status.WARN, fewer, f"set {cpus} in {W} to 4")],
        )
        # A host with fewer CPUs than they need: fewer reviews at first.
        self.assertEqual(
            self.rows({"cpus": 2}, 0.0, 2),
            [
                (
                    "CPUs",
                    Status.WARN,
                    fewer,
                    f"set {RA_KEY}.parallel_reviews in {W} to 1",
                )
            ],
        )
        self.assertEqual(
            self.rows({"cpus": 1}, 0.0, 2),
            [
                (
                    "CPUs",
                    Status.WARN,
                    f"{cpus} 1 is fewer than the reviews and builds that may run at once (4)",
                    f"set {cpus} in {W} to 2, and set {RA_KEY}.parallel_reviews in {W} to 1",
                )
            ],
        )
        self.assertEqual(self.rows({"cpus": 4}, 0.0, 4), [])
        self.assertEqual(self.rows({"cpus": 1}, 0.0, 4, ra=False), [])
        # One review and one build on one CPU: the fewest there can be.
        values = changed(
            {RA_KEY: dict(RA, parallel_reviews=1), "host.resources": {"cpus": 1}}
        )
        self.assertEqual(coherence.resources(document(values), 0.0, 1), [])

    def test_a_value_that_does_not_fit_its_key_has_validates_row_alone(self):
        values = changed(
            {
                RA_KEY: dict(RA, parallel_reviews=9),
                "host.resources": {"memory_gib": 0.5, "cpus": 0, "ramdisk_gib": -1},
            }
        )
        self.assertEqual(coherence.resources(document(values), 4.0, 2), [])


class TestBuild(Checks):
    def test_a_download_nothing_checks(self):
        rocm, sdk = "rocm.package_sha256", "claude_agent_sdk.package_sha256"
        unchecked = {"rocm": {"version": "7.1.0"}, "claude_agent_sdk": {}}
        self.assertEqual(
            self.checked(unchecked, BUILD_BASE, BUILD),
            [
                (
                    rocm,
                    Status.WARN,
                    "not checked; the build installs AMD's public installer for this version without checking it",
                    f"set {rocm} in {WB} to {to(rocm, BUILD)}",
                ),
                (
                    sdk,
                    Status.WARN,
                    "not checked; the build installs the SDK from the package index without checking it",
                    f"set claude_agent_sdk.package_url in {WB} to the address of the SDK's wheel (the .whl file), "
                    f"and set {sdk} in {WB} to the wheel's SHA-256 (download it once, then run sha256sum FILE)",
                ),
            ],
        )
        # The public installer, checked against its SHA-256.
        checked = {"rocm": {"version": "7.1.0", "package_sha256": SHA}}
        self.assertEqual(self.checked(checked, BUILD_BASE, BUILD), [BUILD_PASS])

    def test_a_package_mirror_on_a_base_that_is_not_ubuntu(self):
        mirror, base = "operating_system.package_mirror", "operating_system.base_image"
        system = {"package_mirror": "https://mirror.example.com/ubuntu"}
        self.assertEqual(
            self.checked(
                {"operating_system": dict(system, base_image="rockylinux:9")},
                BUILD_BASE,
                BUILD,
            ),
            [
                (
                    mirror,
                    Status.FAIL,
                    f'set, and {base} "rockylinux:9" is not an Ubuntu image: a package mirror is for Ubuntu base '
                    "images only",
                    f'delete {mirror} from {WB}, or set {base} in {WB} to "ubuntu:26.04" or "ubuntu:24.04"',
                )
            ],
        )
        for given in (
            dict(system, base_image="ubuntu:24.04"),
            {"base_image": "almalinux:9"},
        ):
            self.assertEqual(
                self.checked({"operating_system": given}, BUILD_BASE, BUILD),
                [BUILD_PASS],
            )


class TestCheck13(Checks):
    """Check 13 is validate's (rkmon_lib/settings_file.py): its rows on the real schema, and no copy of them here."""

    def validated(self, changes):
        return settings_file.validate(document(changed(changes)))

    def test_a_key_missing_while_its_feature_is_on(self):
        """The whole row as it prints: one for the section, with the section to print."""
        rows = self.validated(
            {
                "features.merge_queue": QUEUE,
                "features.alerts.develop_breaks.enabled": True,
            }
        )
        self.assertEqual(
            checks.render(rows[:1]),
            "  FAIL  alerts: features.alerts.develop_breaks is on, and features.alerts.mode is missing\n"
            "        fix: run rkmon template section features.alerts, paste its text over features.alerts in"
            " ~/rkmon-deploy.yaml,\n"
            "             then set develop_breaks.enabled back to true and fill in each key whose comment says"
            ' "Required when"',
        )
        self.assertEqual(as_shown(rows[1:]), [SKIP])
        # A key missing while its feature is on is no check across keys' row too.
        gpu = dict(
            GPU, cluster={k: v for k, v in GPU["cluster"].items() if k != "login_host"}
        )
        self.assertEqual(self.direct({RA_KEY: dict(RA, remote_gpu_tests=gpu)}), [PASS])

    def test_an_unknown_key_has_the_closest_name(self):
        rows = as_shown(self.validated({"features.alerts.mod": "live"}))
        self.assertEqual(
            rows,
            [
                (
                    "features.alerts.mod",
                    Status.FAIL,
                    "no such key in deploy settings; did you mean features.alerts.mode?",
                    f"rename features.alerts.mod in {W} to features.alerts.mode, or delete features.alerts.mod "
                    f"from {W}",
                ),
                SKIP,
            ],
        )

    def test_a_value_set_for_a_feature_that_is_off(self):
        key = GPU_KEY + ".cluster.partition"
        off = {"enabled": False, "cluster": {"partition": "gpu"}}
        rows = self.checked({RA_KEY: dict(RA, remote_gpu_tests=off)})
        self.assertEqual(
            rows,
            [
                (
                    key,
                    Status.WARN,
                    f"set, and {GPU_KEY} is off: nothing reads it",
                    f"delete {key} from {W}, or turn it on: set {GPU_KEY}.enabled in {W} to true",
                ),
                PASS,
            ],
        )


# --- one row for each key ------------------------------------------------------------------------------------------

# A file with a placeholder in each value a check across keys reads.
HOLES = changed(
    {
        RA_KEY: dict(
            RA,
            parallel_reviews="<reviews>",
            remote_gpu_tests={
                "enabled": True,
                "cluster": {
                    "login_host": "<host>",
                    "ssh_user": "<user>",
                    "partition": "<partition>",
                    "qos": "<qos>",
                    "accounts": ["<account>"],
                    "gpu_types": {"<target>": "<type>"},
                    "default_target": "<target>",
                    "time_limit": "<time>",
                },
                "node": {
                    "python": "<python>",
                    "setup_command": "<command>",
                    "environment": {"<NAME>": "<value>"},
                },
            },
            spend={
                "enabled": True,
                "usd_per_million_tokens": {"<model>": dict.fromkeys(PRICES, "<price>")},
            },
            orchestrator={"model": "<model>"},
            passes={
                "verification_pass": {"model": "<model>"},
                "first_review_pass": {"model": "<model>", "effort": "<effort>"},
            },
        ),
        "model_bindings": {
            "<name>": dict(OLLAMA, server_url="<url>"),
            "gateway": dict(
                GATEWAY,
                model_id="<model>",
                base_url="<url>",
                api_key_sent_as="header_and_bearer",
                api_key_header="<header>",
            ),
        },
        "integrations.claude_code": {
            "base_url": "<url>",
            "api_key_sent_as": "header_and_bearer",
            "api_key_header": "<header>",
        },
        "features.merge_queue": dict(
            QUEUE,
            mode="live",
            develop_guard={
                "enabled": True,
                "ci_red_label": "<label>",
                "stability_label": "<label>",
            },
        ),
        TEAM_KEY: dict(
            TEAM_PUBLISH,
            public_url="<url>",
            publish={
                "web_host": "<host>",
                "username": "<user>",
                "site_folder": "<folder>",
            },
        ),
        "features.ticket_creation.mode": "prototype",
        "features.ticket_creation.project_key": "<project>",
        "features.ticket_creation.follow_up_tickets.enabled": True,
        "integrations.atlassian_cli.jira_url": "<url>",
        "host.users": {"monitor_uid": "<uid>", "monitor_gid": "<gid>"},
        "host.resources": {
            "memory_gib": "<GiB>",
            "cpus": "<cpus>",
            "ramdisk_gib": "<GiB>",
        },
        "repository.scope_labels": ["<label>"],
        "repository.shared_label_folders": {"<label>": "<folder>"},
    }
)

# A file that fits, with a problem for nearly every check.
TROUBLE = changed(
    {
        RA_KEY: dict(
            RA,
            remote_gpu_tests=dict(
                GPU,
                cluster=dict(GPU["cluster"], login_host="", default_target="gfx90a"),
                node={"python": ""},
            ),
            spend={"enabled": True},
            passes={
                "verification_pass": {"model": "local"},
                "first_review_pass": {"model": "local", "effort": "max"},
                "ci_triage_pass": {"model": "mystery"},
            },
        ),
        "model_bindings": {
            "local": dict(OLLAMA, server_url="http://localhost:11434"),
            "gateway": dict(GATEWAY, api_key_sent_as="header"),
            "opus": OLLAMA,
            "relay": dict(
                GATEWAY,
                api_key_sent_as="header_and_bearer",
                api_key_header="Authorization",
            ),
        },
        "features.merge_queue": dict(
            QUEUE, mode="live", drift_notices={"enabled": True}
        ),
        "features.dashboards.admin.enabled": False,
        TEAM_KEY: dict(TEAM, delivery="publish"),
        "features.alerts": {
            "mode": "prototype",
            "develop_breaks": {"enabled": True},
            "needs_you": {"enabled": False},
        },
        "features.ticket_creation.mode": "prototype",
        "features.ticket_creation.project_key": "",
        "features.ticket_creation.found_issue_tickets.enabled": True,
        "integrations.atlassian_cli.jira_url": "",
        "host.users.monitor_uid": 65534,
        "host.resources": {"memory_gib": 16, "cpus": 8, "ramdisk_gib": 4},
        "repository.scope_labels": ["mine", "needs-ci"],
        "repository.shared_label_folders": {"needs-ci": "projects/shared"},
    }
)

# A file whose values do not all fit: the checks across keys wait, and this host's share has no row for them.
MISFIT = changed(
    {
        RA_KEY: dict(
            RA,
            parallel_reviews=9,
            remote_gpu_tests=dict(
                GPU,
                cluster={k: v for k, v in GPU["cluster"].items() if k != "login_host"},
            ),
        ),
        "host.resources": {"memory_gib": 0.5, "cpus": 0, "ramdisk_gib": -1},
        "host.users.monitor_uid": 65534,
    }
)

TOP_KEYS = "|".join(
    sorted(
        {*DEPLOY.load()["properties"], *BUILD.load()["properties"]},
        key=len,
        reverse=True,
    )
)
# A key as a row names it: from a top-level key, each part after a dot, with a list's [i].
KEY = re.compile(rf"(?<![\w.])(?:{TOP_KEYS})(?:\.[\w:\[\]<>-]+)+")


def about(row):
    """The keys a problem row is about: its name, when that is a key; the keys missing, in validate's row of a
    section; else the first key its text names."""
    name, _status, found, _fix = row
    if KEY.fullmatch(name):
        return [name]
    if found.endswith(" missing") and " on, and " in found:
        return KEY.findall(found.split(" on, and ", 1)[1])
    first = KEY.search(found)
    return [first.group(0) if first else name]


def every_row(values, host_memory_gib=12.0, host_cpus=4):
    """Each row a file gets from rkmon check: validate's (with the checks across keys, once every key fits), then
    this host's share."""
    doc = document(values)
    return as_shown(settings_file.validate(doc)) + as_shown(
        coherence.resources(doc, host_memory_gib, host_cpus)
    )


class TestOneRowPerKey(Checks):
    def assertOneRowEach(self, rows):
        keys = collections.Counter(
            key for row in rows if row[1] in checks.PROBLEMS for key in about(row)
        )
        twice = {
            key: [r for r in rows if key in about(r)]
            for key, n in keys.items()
            if n > 1
        }
        self.assertEqual(twice, {})
        return keys

    def test_a_placeholder_has_its_row_alone(self):
        """No check across keys adds a row for a key whose value has one: the value is given, not missing or
        blank, and fits whatever the key takes."""
        doc = document(HOLES)
        self.assertEqual(as_shown(coherence.deploy(doc)), [PASS])
        self.assertEqual(coherence.resources(doc, 4.0, 1), [])
        rows = every_row(HOLES, 4.0, 1)
        keys = self.assertOneRowEach(rows)
        self.assertGreater(len(keys), 30)
        for row in rows[:-1]:
            self.assertEqual(row[1], Status.FAIL, row)
            self.assertIn("<", row[2] + str(row[0]), row)
        self.assertEqual(rows[-1], SKIP)

    def test_no_key_has_two_rows(self):
        for name, values in (("trouble", TROUBLE), ("misfit", MISFIT)):
            with self.subTest(file=name):
                self.assertOneRowEach(every_row(values))
        rows = every_row(TROUBLE)
        self.assertEqual(rows[0][:2], ("settings", Status.PASS))
        # Each check has a row: those named by a title, beside 19 named by their keys.
        titled = [r[0] for r in rows[1:] if not KEY.fullmatch(r[0])]
        self.assertEqual(
            titled,
            [
                "found-issue tickets",
                "team dashboard",
                "spend",
                "team dashboard",
                "shared labels",
                "RAM disk",
                "memory",
                "CPUs",
            ],
        )
        self.assertEqual(len(rows) - 1 - len(titled), 19)
        self.assertEqual(
            [r[0] for r in every_row(MISFIT) if r[0] in ("memory", "CPUs", "RAM disk")],
            [],
        )

    def test_no_rejected_or_unwanted_word(self):
        rows = every_row(TROUBLE) + every_row(HOLES) + every_row(MISFIT)
        rows += as_shown(
            coherence.build(
                document(changed({"rocm": {"version": "7.1.0"}}, {}), BUILD)
            )
        )
        for row in rows:
            text = " ".join(str(part) for part in row)
            for words in (REJECTED, UNWANTED):
                self.assertIsNone(words.search(text), row)


# --- check 18: the team's skills folder ------------------------------------------------------------------------------

SKILLS_KEY = RA_KEY + ".skills_folder"
VERIFIER = verifier_file.FILE
PACKAGE_VERIFIER = (ROOT / "agentic" / VERIFIER).read_text(encoding="utf-8")
# The places the key's description names, as check 18 reads them.
PLACES = (
    "team_members/",
    "agents/",
    "review_book.md",
    "skills/rocke-monitor/review_procedure.md",
)


def lacking(*leads: str) -> str:
    """The package's verifier with each of LEADS' bullets left in, its lead no longer in bold: the file lacks them."""
    text = PACKAGE_VERIFIER
    for lead in leads:
        if f"- **{lead}" not in text:
            raise AssertionError(f"no bullet starts with **{lead}")
        text = text.replace(f"- **{lead}", f"- {lead}")
    return text


class TestSkillsFolder(Checks):
    """Check 18: while skills_folder is set, the folder exists, you can read it and it is outside the package
    folder; each file in it is a regular Markdown file, no link, at a place the key's description allows; a verifier
    file there keeps each checklist lead a review in parts quotes. Each FAIL names the key; the one PASS row names the
    package's files the team's replace and those it adds. No row of validate's: rkmon check and rkmon confirm add
    them (tests/test_deploy_path_confirm.py)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = pathlib.Path(tmp.name)
        self.team = self.base / "team"
        self.team.mkdir()
        self.shown = settings_file.shown(self.team)

    def write(self, rel: str, body="") -> pathlib.Path:
        """The file REL of the team's folder, with BODY: text, or bytes as they are."""
        f = self.team / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            f.write_bytes(body)
        else:
            f.write_text(body, encoding="utf-8")
        return f

    def rows(self, folder=None, package_dir=ROOT) -> list:
        """Check 18's rows of a file whose skills_folder is FOLDER (the team's folder by default), review automation
        on, each with no rejected or unwanted word; validate's rows of the file are its own PASS row and coherence's
        alone."""
        folder = str(self.team if folder is None else folder)
        values = changed({RA_KEY: RA, SKILLS_KEY: folder})
        self.assertEqual(self.checked(values=values), [PASS])
        rows = as_shown(coherence.skills(document(values), package_dir))
        for row in rows:
            text = " ".join(str(part) for part in row)
            for words in (REJECTED, UNWANTED):
                self.assertIsNone(words.search(text), row)
        return rows

    def refused(self, *named) -> list:
        """Check 18's FAIL row of the things NAMED in the team's folder, each (its path, why)."""
        has = wrapping.joined(f"{rel} ({why})" for rel, why in named)
        return [
            (
                "skills folder",
                Status.FAIL,
                f"{SKILLS_KEY} {self.shown} may contain only Markdown files, each at "
                f"{wrapping.joined(PLACES, 'or')}, and it has {has}",
                f"take each out of {self.shown} (a link: put the file itself in its place), then run {AGAIN}",
            )
        ]

    def test_the_places_are_the_key_s_description_s(self):
        """The places a file may be come from the key's description, their one home, and each is a place in the
        package's agentic folder; a description that names none is the package's own fault, and raises.
        """
        self.assertEqual(coherence.skills_places(), PLACES)
        self.assertIn(
            f"it may contain only {wrapping.joined(PLACES)}.",
            deploy_tree.description(SKILLS_KEY),
        )
        for place in PLACES:
            with self.subTest(place=place):
                there = ROOT / "agentic" / place
                self.assertTrue(
                    there.is_dir() if place.endswith("/") else there.is_file()
                )
        with mock.patch.object(
            deploy_tree, "description", return_value="A folder with your skills."
        ):
            with self.assertRaisesRegex(ValueError, re.escape(SKILLS_KEY)):
                coherence.skills_places()

    def test_a_verifier_file_that_keeps_the_leads_passes(self):
        """One PASS row: the package's files the team's replace, then the files it adds. The session renders the
        team's verifier: tests/test_agentic.py."""
        self.write(
            VERIFIER,
            PACKAGE_VERIFIER + "\n## The team's rule\n\nAsk about tile sizes.\n",
        )
        self.write("agents/team-perf-reviewer.md", "# The team's perf reviewer\n")
        self.write("review_book.md", "# The team's review book\n")
        self.assertEqual(
            self.rows(),
            [
                (
                    "skills folder",
                    Status.PASS,
                    f"the team's skills in {self.shown} replace the package's review_book.md and {VERIFIER}, and "
                    "add agents/team-perf-reviewer.md",
                    None,
                )
            ],
        )
        (self.team / "review_book.md").unlink()
        (self.team / "agents/team-perf-reviewer.md").unlink()
        self.assertEqual(
            self.rows()[0][2],
            f"the team's skills in {self.shown} replace the package's {VERIFIER}",
        )

    def test_an_empty_folder_passes_and_a_blank_key_has_no_row(self):
        self.assertEqual(
            self.rows(),
            [
                (
                    "skills folder",
                    Status.PASS,
                    f"{self.shown} has no file: the package's skills run as they are",
                    None,
                )
            ],
        )
        self.assertEqual(self.rows(""), [])
        # A placeholder has validate's row alone.
        doc = document(changed({SKILLS_KEY: "<your team's skills folder>"}))
        self.assertEqual(coherence.skills(doc, ROOT), [])

    def test_a_link_fails(self):
        """A link is named and never followed, to a file or to a folder."""
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "verifier.md").write_text(PACKAGE_VERIFIER, encoding="utf-8")
        (self.team / "team_members").mkdir()
        (self.team / VERIFIER).symlink_to(outside / "verifier.md")
        (self.team / "agents").symlink_to(outside, target_is_directory=True)
        self.assertEqual(
            self.rows(), self.refused(("agents", "a link"), (VERIFIER, "a link"))
        )

    def test_a_file_at_a_place_the_key_does_not_allow_fails(self):
        """A file at none of the places, and a folder at none of them, named once with its / and not looked into;
        the files at the places pass."""
        self.write("agents/team-perf-reviewer.md", "# The team's perf reviewer\n")
        self.write("notes.md", "# Notes\n")
        self.write("skills/rocke-monitor/SKILL.md", "# The team's skill\n")
        self.write("skills/rocke-monitor/review_procedure.md", "# The procedure\n")
        self.write("skills/other/one.md", "# One\n")
        self.write("skills/other/two.md", "# Two\n")
        not_there = coherence.NOT_A_PLACE
        self.assertEqual(
            self.rows(),
            self.refused(
                ("notes.md", not_there),
                ("skills/other/", not_there),
                ("skills/rocke-monitor/SKILL.md", not_there),
            ),
        )

    def test_a_missing_lead_fails(self):
        """A verifier file there that lacks a checklist lead: a part of each kind that quotes it would say it is
        missing (tests/test_verifier_file.py)."""
        self.write(VERIFIER, lacking("New IR op handlers"))
        fix = (
            "start a bullet of its What to Check with {} in bold, as the package's agentic/"
            f"{VERIFIER} does, then run {AGAIN}"
        )
        self.assertEqual(
            self.rows(),
            [
                (
                    "skills folder",
                    Status.FAIL,
                    f"{SKILLS_KEY} {self.shown}: {VERIFIER} lacks the checklist lead **New IR op handlers**, which a "
                    "review in parts quotes from it: a part would say it is missing",
                    fix.format("it"),
                )
            ],
        )
        self.write(VERIFIER, lacking("New IR op handlers", "Launch paths"))
        self.assertEqual(
            self.rows(),
            [
                (
                    "skills folder",
                    Status.FAIL,
                    f"{SKILLS_KEY} {self.shown}: {VERIFIER} lacks the checklist leads **Launch paths** and **New IR "
                    "op handlers**, which a review in parts quotes from it: a part would say they are missing",
                    fix.format("each"),
                )
            ],
        )
        # A file of the same name elsewhere is no verifier file.
        (self.team / VERIFIER).unlink()
        self.write("agents/rocke_review_verifier.md", lacking("Launch paths"))
        self.assertEqual([r[1] for r in self.rows()], [Status.PASS])

    def test_any_other_file_fails_with_why(self):
        """A file of another kind (a named pipe, which is not waited on), one whose name does not end in .md, and one
        that is not UTF-8 text, in one row with the others, beside the lead's own row.
        """
        self.write(VERIFIER, lacking("Launch paths"))
        self.write("agents/run.sh", "echo\n")
        self.write("agents/latin.md", b"caf\xe9\n")
        os.mkfifo(self.team / "agents" / "fifo.md")
        rows = self.rows()
        self.assertEqual(
            rows[:1],
            self.refused(
                ("agents/fifo.md", "not a regular file"),
                ("agents/latin.md", "not UTF-8 text"),
                ("agents/run.sh", "not a Markdown file"),
            ),
        )
        self.assertEqual(
            [(r[0], r[1]) for r in rows[1:]], [("skills folder", Status.FAIL)]
        )
        self.assertIn("**Launch paths**", rows[1][2])

    @unittest.skipIf(os.geteuid() == 0, "root reads every file")
    def test_what_you_cannot_read_fails(self):
        """A file or a folder in it that you cannot read is named; the folder itself, alone."""
        self.write("agents/locked.md", "# Locked\n").chmod(0)
        self.write("team_members/other.md", "# Other\n")
        (self.team / "team_members").chmod(0)
        self.addCleanup((self.team / "team_members").chmod, 0o700)
        self.assertEqual(
            self.rows(),
            self.refused(
                ("agents/locked.md", "cannot be read"),
                ("team_members/", "cannot be read: Permission denied"),
            ),
        )
        self.team.chmod(0)
        self.addCleanup(self.team.chmod, 0o700)
        self.assertEqual(
            self.rows(),
            [
                (
                    "skills folder",
                    Status.FAIL,
                    f"{SKILLS_KEY} {self.shown}: you cannot read it (Permission denied)",
                    f"make it and its files readable by you, such as chmod -R u+rX {self.shown}, then run {AGAIN}",
                )
            ],
        )

    def test_the_folder_itself_has_its_row_alone(self):
        """No such folder, a file in its place, or a folder inside the package folder, which a package update
        replaces."""
        edits = (
            f"set {SKILLS_KEY} in {W} to the folder your team's skills are in, or set {SKILLS_KEY} in {W} to "
            '"" for the package\'s skills only'
        )
        gone = self.base / "gone"
        file = self.base / "skills.md"
        file.write_text("# Skills\n", encoding="utf-8")
        for folder, found in ((gone, "no such folder"), (file, "not a folder")):
            with self.subTest(found=found):
                shown = settings_file.shown(folder)
                self.assertEqual(
                    self.rows(folder),
                    [
                        (
                            "skills folder",
                            Status.FAIL,
                            f"{SKILLS_KEY} {shown}: {found}",
                            edits,
                        )
                    ],
                )
        package = self.base / "package"
        inside = package / "team"
        inside.mkdir(parents=True)
        self.assertEqual(
            self.rows(inside, package),
            [
                (
                    "skills folder",
                    Status.FAIL,
                    f"{SKILLS_KEY} {settings_file.shown(inside)}: inside the package folder, which a package update "
                    "replaces",
                    f"move it out of {settings_file.shown(package)}, then set {SKILLS_KEY} in {W} to its new place",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
