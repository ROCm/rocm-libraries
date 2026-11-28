# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The step from layout 1 to layout 2 of the deploy and the build settings (rkmon_lib/from_layout1.py), on invented
files of the earlier layout: a golden test for each of its rules beyond one value moved to one key, the build file's
included; its moves of one value to one key, each with its note; what a step does with a value no rule takes; the
refusals the steps raise, as upgrade words them; and whole files of the earlier layout, read, upgraded, written by the
one writer and read again, with no value lost.
"""

import copy
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import home_repository  # noqa: E402
from rkmon_lib import (
    answers,
    bindings,
    build_plan,
    config,
    file_types,
    from_layout1,
    image,
    registry,
    settings_file,
)  # noqa: E402
from rkmon_lib.checks import Status  # noqa: E402
from rkmon_lib.settings_file import Document, Refused  # noqa: E402

DEPLOY, BUILD = file_types.DEPLOY, file_types.BUILD
RA = "features.review_automation"
PASSES = f"{RA}.passes"
RGT = f"{RA}.remote_gpu_tests"
MQ = "features.merge_queue"
GUARD = f"{MQ}.develop_guard"
ADMIN = "features.dashboards.admin"
TEAM = "features.dashboards.team"
ALERTS = "features.alerts"
TC = "features.ticket_creation"
GONE = object()  # a key the values do not have
EPIC = "PROJ-" + "7"  # a ticket key, in pieces, so that no file holds one
AT_GITHUB = (
    "@" + "github.com"
)  # in pieces, as the privacy check reads any name@host as an email address
SHA = "0123456789abcdef" * 4  # an invented SHA-256
IMAGE_NAME = "rocke-pr-monitor"  # what a blank or missing image name meant in layout 1, in either file
NAMED = {"image": {"name": IMAGE_NAME}}  # what a build file with no image name gets
PYPI = "https://pypi.org/simple"  # what a blank Python package index meant in layout 1
# The develop guard's labels, as layout 1's develop guard used them when the file names none.
LABEL_NOTES = [
    f'{GUARD}.ci_red_label "post-merge" and {GUARD}.ci_red_extra_labels ["ci-red"]: the labels layout 1\'s develop '
    'guard put on a develop issue while CI was red; "ci-red" is the label layout 1\'s develop guard added beside the '
    "first, when the repository had it",
    f'{GUARD}.stability_label "rocke-stability": the label layout 1\'s develop guard put on any other develop issue',
]


def loaded(text: str) -> dict:
    """TEXT, the keys of a layout 1 file after its header, as settings_file.read gives them to a step: each plain
    value as written (answers.Plain)."""
    return answers.loads(text, plain=True) if text.strip() else {}


def upgraded(text: str, file_type=DEPLOY) -> tuple:
    """(values, notes) for TEXT, through settings_file.upgrade: the step, then each value typed by layout 2's
    schema."""
    doc, notes = settings_file.upgrade(Document(file_type, 1, loaded(text)))
    return doc.values, notes


def refusal(text: str, file_type=DEPLOY) -> list:
    """The rows of upgrade's refusal of TEXT, as (name, status, found, the fix as shown)."""
    try:
        upgraded(text, file_type)
    except Refused as exc:
        return [(r.name, r.status, r.found, str(r.fix)) for r in exc.rows]
    raise AssertionError("upgrade took the file")


def nested(dotted: str, value: str) -> str:
    """YAML for the key DOTTED holding VALUE, in block style; a VALUE that starts on a new line is a section, its
    lines under the key."""
    parts = dotted.split(".")
    text = "".join(f"{'  ' * i}{part}:\n" for i, part in enumerate(parts[:-1]))
    indent = "  " * (len(parts) - 1)
    first, *more = value.split("\n")
    text += f"{indent}{parts[-1]}:" + (f" {first}" if first else "") + "\n"
    return text + "".join(f"{indent}  {line}\n" for line in more)


def cloud(name: str) -> str:
    """The address of the atlassian CLI's cloud site NAME, built here, so that no file holds a site's address."""
    return f"https://{name}.atlassian.net"


def at(values, dotted: str):
    """The value at DOTTED, or GONE."""
    cur = values
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return GONE
        cur = cur[part]
    return cur


def on(notes, *keys) -> list:
    """The notes that start with one of KEYS."""
    return [n for n in notes if n.startswith(keys)]


class TestRules(unittest.TestCase):
    """A golden test for each of the step's rules beyond one value moved to one key, on the deploy file."""

    def test_a_secret_values_section_is_left_for_upgrade_to_refuse(self):
        values = loaded("secret_values:\n  a: x\n  b: y\n  c: z\n")
        out, notes = from_layout1.deploy(values, DEPLOY)
        self.assertEqual(out["secret_values"], {"a": "x", "b": "y", "c": "z"})
        self.assertEqual(on(notes, "secret_values"), [])
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(Document(DEPLOY, 1, values))
        self.assertEqual(cm.exception.reason, "secret")
        self.assertEqual(
            [(r.status, r.found, str(r.fix)) for r in cm.exception.rows],
            [
                (
                    Status.FAIL,
                    "your deploy settings has a secret_values section (3 entries), and no secret goes in a settings "
                    "file",
                    "move each value into a file only you can read, delete secret_values from your deploy settings, "
                    "and run migrate again; after rkmon confirm, rkmon login lists each secret and installs it",
                )
            ],
        )

    def test_secrets_setup_and_the_exports_record_are_dropped_with_a_note(self):
        values, notes = upgraded(
            "secrets:\n  claude: claude_env\n  github: github_token\nsetup:\n  answered: true\n"
            'exported_at: "2026-01-01T00:00:00Z"\npackage_version: "1.0.0"\nsource: rkmon settings export\n'
        )
        for key in ("secrets", "setup", "exported_at", "package_version", "source"):
            self.assertIs(at(values, key), GONE)
        self.assertEqual(
            on(notes, "secrets", "setup", "exported_at"),
            [
                "secrets (2 entries): dropped, as no secret goes in a settings file; after rkmon confirm, rkmon "
                "login lists each secret and installs it",
                "setup: dropped, as a settings file keeps no record of the interview that wrote it",
                "exported_at, package_version and source: dropped, as a settings file keeps no record of the export "
                "that wrote it",
            ],
        )
        self.assertEqual(
            on(upgraded("source: rkmon settings export\n")[1], "source"),
            [
                "source: dropped, as a settings file keeps no record of the export that wrote it"
            ],
        )

    def test_a_blank_container_name_is_rkmon(self):
        note = (
            'container.name: "rkmon", what a blank name meant: the deployment keeps its name, so its volumes and '
            "its state"
        )
        for text in ('container:\n  name: ""\n', "container:\n  name: null\n", ""):
            with self.subTest(text=text):
                values, notes = upgraded(text)
                self.assertEqual(values["container"], {"name": "rkmon"})
                self.assertEqual(on(notes, "container.name"), [note])
        values, notes = upgraded("container:\n  name: my-monitor\n")
        self.assertEqual(values["container"], {"name": "my-monitor"})
        self.assertEqual(on(notes, "container"), [])

    def test_a_blank_image_tag_is_the_containers_name(self):
        values, notes = upgraded(
            'container:\n  name: my-monitor\n  image:\n    tag: ""\n'
        )
        self.assertEqual(values["image"], {"name": IMAGE_NAME, "tag": "my-monitor"})
        self.assertEqual(
            on(notes, "image.tag"),
            [
                'image.tag: "my-monitor", the container\'s name, what a blank tag meant; rkmon confirm says whether '
                "that image suits this package"
            ],
        )
        self.assertEqual(upgraded("")[0]["image"], {"name": IMAGE_NAME, "tag": "rkmon"})
        values, notes = upgraded(
            'container:\n  name: my-monitor\n  image:\n    tag: "2.0"\n'
        )
        self.assertEqual(values["image"], {"name": IMAGE_NAME, "tag": "2.0"})
        self.assertEqual(
            on(notes, "image.tag", "container.image"),
            ["container.image.tag is now image.tag"],
        )

    def test_a_blank_image_name_is_what_it_meant(self):
        # Written, so the step means the same in every later package, with no note: layout 2's default is
        # the same name.
        for file_type, old in (
            (DEPLOY, "container.image.name"),
            (BUILD, "image.name"),
        ):
            for text in (nested(old, '""'), nested(old, "null"), ""):
                with self.subTest(old=old, text=text):
                    values, notes = upgraded(text, file_type)
                    self.assertEqual(values["image"]["name"], IMAGE_NAME)
                    self.assertEqual(on(notes, "image.name", old), [])

    def test_the_admin_dashboards_address(self):
        # Blank: dropped, with no note; 127.0.0.1: dropped, with one; any other: refused.
        values, notes = upgraded(
            'features:\n  dashboards:\n    admin:\n      enabled: true\n      bind: ""\n'
        )
        self.assertEqual(at(values, ADMIN), {"enabled": True})
        self.assertEqual(on(notes, ADMIN), [])
        values, notes = upgraded(
            "features:\n  dashboards:\n    admin:\n      port: 18620\n      bind: 127.0.0.1\n"
        )
        self.assertEqual(at(values, ADMIN), {"port": 18620})
        self.assertEqual(
            on(notes, ADMIN),
            [
                f'{ADMIN}.bind "127.0.0.1": dropped, as the admin dashboard listens on this host\'s 127.0.0.1 only'
            ],
        )
        self.assertEqual(
            refusal("features:\n  dashboards:\n    admin:\n      bind: 0.0.0.0\n"),
            [
                (
                    f"{ADMIN}.bind",
                    Status.FAIL,
                    f'{ADMIN}.bind is "0.0.0.0", and the admin dashboard listens on this host\'s 127.0.0.1 only',
                    f'delete {ADMIN}.bind from your deploy settings; README.md, "The admin dashboard", says how to '
                    "reach it from another machine",
                )
            ],
        )

    def test_path_labels_are_shared_label_folders(self):
        values, notes = upgraded(
            "repository:\n  path: projects/my-project\n  path_labels: [shared-a, shared-b]\n"
        )
        self.assertEqual(
            values["repository"],
            {
                "folder": "projects/my-project",
                "shared_label_folders": {
                    "shared-a": "projects/my-project",
                    "shared-b": "projects/my-project",
                },
            },
        )
        self.assertEqual(
            on(notes, "repository.path_labels"),
            [
                "repository.path_labels is now repository.shared_label_folders: each label with repository.folder "
                '("projects/my-project") as its folder, the one folder it counted for'
            ],
        )
        # No folder: the template's placeholder, which rkmon check asks for.
        values = upgraded("repository:\n  path_labels: [shared-a]\n")[0]
        self.assertEqual(
            values["repository"], {"shared_label_folders": {"shared-a": "<folder>"}}
        )
        values, notes = upgraded("repository:\n  path_labels: []\n")
        self.assertIs(at(values, "repository"), GONE)
        self.assertEqual(on(notes, "repository"), [])

    def test_the_alerts_switch_is_both_alerts_switches(self):
        for switch in (True, False):
            word = "true" if switch else "false"
            with self.subTest(switch=word):
                values, notes = upgraded(f"features:\n  alerts:\n    enabled: {word}\n")
                self.assertEqual(
                    (
                        at(values, f"{ALERTS}.develop_breaks.enabled"),
                        at(values, f"{ALERTS}.needs_you.enabled"),
                    ),
                    (switch, switch),
                )
                self.assertIn(
                    f"{ALERTS}.enabled is now {ALERTS}.develop_breaks.enabled and {ALERTS}.needs_you.enabled, "
                    f"each {word}",
                    notes,
                )

    def test_each_ticket_type_is_the_sections_switch_and_its_trigger(self):
        # (the section's switch, review_follow_up, out_of_scope_issue; None: not in the file) -> the two types.
        for section, follow_up, found, expected in (
            ("true", "true", "true", (True, True)),
            ("true", "true", "false", (True, False)),
            ("true", "false", "true", (False, True)),
            ("true", None, None, (True, True)),  # each trigger's default was true
            ("false", "true", "true", (False, False)),
            (None, "true", "true", (False, False)),  # the section's default was false
        ):
            with self.subTest(section=section, follow_up=follow_up, found=found):
                text = "features:\n  ticket_creation:\n"
                if section:
                    text += f"    enabled: {section}\n"
                if follow_up:
                    text += f"    triggers:\n      review_follow_up: {follow_up}\n      out_of_scope_issue: {found}\n"
                values = upgraded(text)[0]
                self.assertEqual(
                    (
                        at(values, f"{TC}.follow_up_tickets.enabled"),
                        at(values, f"{TC}.found_issue_tickets.enabled"),
                    ),
                    expected,
                )
        notes = upgraded(
            "features:\n  ticket_creation:\n    enabled: true\n    triggers:\n      out_of_scope_issue: false\n"
        )[1]
        self.assertIn(
            f"{TC}.enabled and its triggers are now {TC}.follow_up_tickets.enabled true and "
            f"{TC}.found_issue_tickets.enabled false: each the section's switch and the type's trigger",
            notes,
        )

    def test_alerts_ticket_creation_and_the_team_dashboard_get_a_mode(self):
        on_text = (
            "features:\n  alerts:\n    enabled: true\n  ticket_creation:\n    enabled: true\n  dashboards:\n"
            "    team:\n      enabled: true\n"
        )
        modes = (f"{TEAM}.mode", f"{ALERTS}.mode", f"{TC}.mode")
        values, notes = upgraded(on_text)
        self.assertEqual([at(values, k) for k in modes], ["live"] * 3)
        self.assertEqual(
            on(notes, *modes),
            [
                f'{TEAM}.mode: "live", as the team dashboard is on in the file, and layout 1 sent whenever it was on',
                f'{ALERTS}.mode: "live", as alerts are on in the file, and layout 1 sent whenever it was on',
                f'{TC}.mode: "live", as ticket creation is on in the file, and layout 1 sent whenever it was on',
            ],
        )
        # One that is off is written as any feature that is off: no mode, and no note names one.
        for text in (on_text.replace("true", "false"), ""):
            with self.subTest(text=text):
                values, notes = upgraded(text)
                self.assertEqual([at(values, k) for k in modes], [GONE] * 3)
                self.assertEqual([n for n in notes if ".mode" in n], [])
        # Ticket creation is on while a type of ticket is: layout 2 has no switch for the section, so with both
        # triggers off it is a feature that is off.
        values, notes = upgraded(
            "features:\n  ticket_creation:\n    enabled: true\n    triggers:\n      review_follow_up: false\n"
            "      out_of_scope_issue: false\n"
        )
        self.assertIs(at(values, f"{TC}.mode"), GONE)
        self.assertEqual(on(notes, f"{TC}.mode"), [])
        values = upgraded(
            "features:\n  ticket_creation:\n    enabled: true\n    triggers:\n      review_follow_up: false\n"
        )[0]
        self.assertEqual(at(values, f"{TC}.mode"), "live")

    def test_a_pass_model_default_or_none_is_default(self):
        values, notes = upgraded(
            "features:\n  review_automation:\n    passes:\n"
            "      first_pass:\n        model: default\n"
            "      ci_triage:\n        model: none\n"
            "      quality_draft:\n        model: claude-opus\n"
            "      notice_draft:\n        model: notices\n"
        )
        self.assertEqual(
            [
                at(values, f"{PASSES}.{p}.model")
                for p in (
                    "first_review_pass",
                    "ci_triage_pass",
                    "first_code_quality_pass",
                    "notice_draft_pass",
                )
            ],
            ["default", "default", "default", "notices"],
        )
        self.assertEqual(
            [n for n in notes if ".model " in n],
            [
                f'{PASSES}.ci_triage.model "none" is now "default": the pass in charge does the work',
                f'{PASSES}.quality_draft.model "claude-opus" is now "default": the pass in charge does the work',
            ],
        )
        # A blank model on an optional pass is "default" too, with no note: the file named no model of its own.
        values, notes = upgraded(
            'features:\n  review_automation:\n    passes:\n      ci_triage:\n        model: ""\n'
        )
        self.assertEqual(at(values, f"{PASSES}.ci_triage_pass.model"), "default")
        self.assertEqual([n for n in notes if ".model" in n], [])

    def test_default_or_none_on_a_pass_that_always_runs(self):
        # The orchestrator, the verification pass and the code quality pass take no "default": either word is the
        # model layout 1 ran the pass on when the file named none, reported.
        values, notes = upgraded(
            "monitor:\n  orchestrator:\n    model: default\n"
            "features:\n  review_automation:\n    passes:\n"
            "      verify:\n        model: none\n"
            "      quality:\n        model: default\n"
        )
        self.assertEqual(
            [
                at(values, f"{RA}.orchestrator.model"),
                at(values, f"{PASSES}.verification_pass.model"),
                at(values, f"{PASSES}.code_quality_pass.model"),
            ],
            ["opus", "opus", "opus"],
        )
        why = "the model layout 1 ran it on when the file named none: it always runs, so it takes a model of its own"
        self.assertEqual(
            [n for n in notes if ".model " in n],
            [
                f'monitor.orchestrator.model "default" is now "opus", {why}',
                f'{PASSES}.verify.model "none" is now "opus", {why}',
                f'{PASSES}.quality.model "default" is now "opus", {why}',
            ],
        )
        # A blank one is left out, with no note; any other model is carried as it is, for rkmon check to judge.
        values, notes = upgraded(
            "features:\n  review_automation:\n    passes:\n"
            '      verify:\n        model: ""\n'
            "      quality:\n        model: sonnet\n"
        )
        self.assertIs(at(values, f"{PASSES}.verification_pass.model"), GONE)
        self.assertEqual(at(values, f"{PASSES}.code_quality_pass.model"), "sonnet")
        self.assertEqual([n for n in notes if ".model " in n], [])

    def test_a_first_pass_that_names_worker(self):
        first = "features:\n  review_automation:\n    passes:\n      first_pass:\n        model: worker\n"
        binding = (
            "model_bindings:\n  worker:\n    provider: openai-compatible\n    endpoint: https://llm.example.com/v1\n"
            "    model_id: example-model\n"
        )
        # With a binding named worker: the pass keeps it.
        values, notes = upgraded(first + binding)
        self.assertEqual(at(values, f"{PASSES}.first_review_pass.model"), "worker")
        self.assertEqual(
            at(values, "model_bindings.worker.provider"), "openai_compatible"
        )
        self.assertIn(
            f'{PASSES}.first_review_pass.model: "worker", the file\'s binding of that name, an ordinary binding now, '
            "through which the pass drafts in steps; the local worker that the name chose leaves the package",
            notes,
        )
        self.assertEqual(on(notes, "model_bindings.worker:"), [])
        # Without one: "default", and how to bind a model.
        values, notes = upgraded(first)
        self.assertEqual(at(values, f"{PASSES}.first_review_pass.model"), "default")
        self.assertIn(
            f'{PASSES}.first_pass.model "worker" is now "default", as the file binds no model named worker: the local '
            "worker that the name chose leaves the package. To have the pass draft, bind a model (rkmon template "
            f"section model_bindings prints the forms) and name it in {PASSES}.first_review_pass.model",
            notes,
        )
        # A binding named worker that no pass names says so too.
        self.assertIn(
            "model_bindings.worker: an ordinary binding now; the local worker that the name chose leaves the "
            "package",
            upgraded(binding)[1],
        )

    def test_a_worker_binding_with_a_blank_timeout_ran_on_900_seconds(self):
        values, notes = upgraded(
            "model_bindings:\n  worker:\n    provider: ollama\n    model_id: example-model\n    limits:\n"
            '      timeout_min: ""\n  other:\n    provider: ollama\n    model_id: example-model\n'
        )
        self.assertEqual(at(values, "model_bindings.worker.stall_timeout_seconds"), 900)
        self.assertIs(at(values, "model_bindings.other.stall_timeout_seconds"), GONE)
        self.assertEqual(
            on(notes, "model_bindings.worker.stall"),
            [
                "model_bindings.worker.stall_timeout_seconds: 900, what the binding named worker ran on with no "
                "timeout"
            ],
        )

    def test_an_ollama_bindings_requests_at_a_time(self):
        # A blank or missing limits.max_parallel ran with no cap; the binding named worker at this host's Ollama
        # ran one request at a time with a blank, missing or 0 one; any other number is carried as it is.
        here, there = "http://127.0.0.1:11434", "http://gpu-box.example.com:11434"
        no_cap = "{}.limits.max_parallel_requests: 0, what the binding ran on with no limits.max_parallel: no cap"
        one = (
            "{}.limits.max_parallel_requests: 1, what the binding named worker at this host's Ollama ran on with {}: "
            "one request at a time"
        )
        moved = "{0}.limits.max_parallel is now {0}.limits.max_parallel_requests"
        for name, url, given, written, note in (
            ("other", here, None, 0, no_cap),
            ("other", here, '""', 0, no_cap),
            ("other", here, "0", 0, moved),
            ("other", here, "3", 3, moved),
            ("worker", here, None, 1, one.format("{}", "no limits.max_parallel")),
            ("worker", here, '""', 1, one.format("{}", "no limits.max_parallel")),
            (
                "worker",
                "http://localhost:11434",
                "0",
                1,
                one.format("{}", "limits.max_parallel 0"),
            ),
            ("worker", here, "2", 2, moved),
            ("worker", there, None, 0, no_cap),
            ("worker", there, "0", 0, moved),
        ):
            with self.subTest(name=name, url=url, given=given):
                text = f"model_bindings:\n  {name}:\n    provider: ollama\n    model_id: m\n    endpoint: {url}\n"
                if given is not None:
                    text += f"    limits:\n      max_parallel: {given}\n"
                values, notes = upgraded(text)
                key = f"model_bindings.{name}"
                self.assertEqual(
                    at(values, f"{key}.limits.max_parallel_requests"), written
                )
                self.assertEqual(on(notes, f"{key}.limits"), [note.format(key)])

    def test_a_bindings_timeout_in_minutes_is_stall_timeout_seconds(self):
        values, notes = upgraded(
            "model_bindings:\n  a:\n    provider: ollama\n    model_id: m\n    limits:\n      timeout_min: 10\n"
            "  b:\n    provider: ollama\n    model_id: m\n    limits:\n      timeout_min: 1.5\n"
            "  c:\n    provider: ollama\n    model_id: m\n    limits:\n      timeout_min: 0\n"
            "  worker:\n    provider: ollama\n    model_id: m\n    limits:\n      timeout_min: 0\n"
        )
        self.assertEqual(at(values, "model_bindings.a.stall_timeout_seconds"), 600)
        self.assertIs(at(values, "model_bindings.a.limits.timeout_min"), GONE)
        self.assertIs(at(values, "model_bindings.b.stall_timeout_seconds"), GONE)
        self.assertIs(at(values, "model_bindings.c.stall_timeout_seconds"), GONE)
        # The binding named worker ran on 900 s with no timeout of its own.
        self.assertEqual(at(values, "model_bindings.worker.stall_timeout_seconds"), 900)
        self.assertEqual(
            on(notes, *(f"model_bindings.{n}.limits.timeout_min" for n in "abc")),
            [
                "model_bindings.a.limits.timeout_min 10 is now model_bindings.a.stall_timeout_seconds 600",
                "model_bindings.b.limits.timeout_min is 1.5: not carried, as it is a whole number of minutes from 1",
                "model_bindings.c.limits.timeout_min is 0: not carried, as it is a whole number of minutes from 1",
            ],
        )
        self.assertEqual(
            on(
                notes,
                "model_bindings.worker.limits.timeout_min",
                "model_bindings.worker.stall",
            ),
            [
                "model_bindings.worker.limits.timeout_min is 0: not carried, as it is a whole number of minutes "
                "from 1",
                "model_bindings.worker.stall_timeout_seconds: 900, what the binding named worker ran on with no "
                "timeout",
            ],
        )

    def test_a_bindings_api_key_secret_is_api_key_secret_name(self):
        values, notes = upgraded(
            "model_bindings:\n"
            "  worker:\n    provider: openai-compatible\n    model_id: m\n    api_key_secret: worker_api_key\n"
            "  other:\n    provider: openai-compatible\n    model_id: m\n    api_key_secret: other_api_key\n"
        )
        self.assertEqual(
            at(values, "model_bindings.worker.api_key_secret_name"),
            "first_review_pass_api_key",
        )
        self.assertEqual(
            at(values, "model_bindings.other.api_key_secret_name"), "other_api_key"
        )
        self.assertEqual(
            on(notes, "model_bindings.worker.api_key", "model_bindings.other.api_key"),
            [
                'model_bindings.worker.api_key_secret "worker_api_key" is now '
                'model_bindings.worker.api_key_secret_name "first_review_pass_api_key", the name the one-time move '
                "copies that secret to",
                "model_bindings.other.api_key_secret is now model_bindings.other.api_key_secret_name",
            ],
        )

    def test_the_merge_queues_notices(self):
        # With the queue on, its queue notices are on, and its drift notices while review automation is: Claude
        # writes them.
        notices = (f"{MQ}.queue_notices.enabled", f"{MQ}.drift_notices.enabled")
        drift = f"{MQ}.drift_notices.enabled: false, as Claude writes them and review automation is off"
        values, notes = upgraded("")
        self.assertEqual([at(values, k) for k in notices], [True, True])
        self.assertNotIn(drift, notes)
        values, notes = upgraded(
            "features:\n  review_automation:\n    enabled: false\n"
        )
        self.assertEqual([at(values, k) for k in notices], [True, False])
        self.assertIn(drift, notes)

    def test_auto_merge_is_a_switch(self):
        for word, switch, shown in (
            ("enabled", True, "true"),
            ("hints_only", False, "false"),
        ):
            with self.subTest(word=word):
                values, notes = upgraded(
                    f"features:\n  merge_queue:\n    auto_merge: {word}\n"
                )
                self.assertEqual(at(values, f"{MQ}.auto_merge"), {"enabled": switch})
                self.assertIn(
                    f'{MQ}.auto_merge "{word}" is now {MQ}.auto_merge.enabled {shown}',
                    notes,
                )
        values, notes = upgraded(
            "features:\n  merge_queue:\n    auto_merge: sometimes\n"
        )
        self.assertIs(at(values, f"{MQ}.auto_merge"), GONE)
        self.assertIn(
            f'{MQ}.auto_merge is "sometimes": not carried, as layout 1 took "enabled" or "hints_only"',
            notes,
        )

    def test_the_team_dashboards_port_and_address_go_to_serve_only(self):
        team = "features:\n  dashboards:\n    team:\n      enabled: true\n{}      port: 18700\n      bind: 0.0.0.0\n"
        values, notes = upgraded(team.format("      delivery: serve\n"))
        self.assertEqual(
            at(values, f"{TEAM}.serve"), {"port": 18700, "address": "0.0.0.0"}
        )
        self.assertEqual(
            on(notes, f"{TEAM}.port", f"{TEAM}.bind"),
            [
                f"{TEAM}.port is now {TEAM}.serve.port",
                f"{TEAM}.bind is now {TEAM}.serve.address",
            ],
        )
        for delivery in ("      delivery: publish\n", ""):
            with self.subTest(delivery=delivery):
                values, notes = upgraded(team.format(delivery))
                self.assertIs(at(values, f"{TEAM}.serve"), GONE)
                self.assertEqual(
                    on(notes, f"{TEAM}.port", f"{TEAM}.bind"),
                    [
                        f'{TEAM}.port 18700: dropped, as the team dashboard\'s delivery is "publish", and only "serve" '
                        "listens",
                        f'{TEAM}.bind "0.0.0.0": dropped, as the team dashboard\'s delivery is "publish", and only '
                        '"serve" listens',
                    ],
                )

    def test_the_web_hosts_keys_go_to_publish_only(self):
        # The mirror of the rule above: each delivery takes its own block alone. The page's address and
        # its priority requests go whatever the delivery, and each key layout 2 has none for says where it goes.
        team = (
            "features:\n  dashboards:\n    team:\n      enabled: true\n{}advanced:\n  publish:\n    html:\n"
            "      host: web.example.com\n      user: op-login\n      site: public_html/team\n      requests: on\n"
            "      url: https://dash.example.com/team\n      key: /run/secrets/web_host_key\n"
            "      known_hosts: /run/secrets/known_hosts\n"
            '      host_keys: ["web.example.com ssh-ed25519 <key>"]\n      dest: /var/www/team/index.html\n'
        )
        own_key = (
            ": dropped, as the monitor keeps the web host's own key in its state: rkmon login shows it, and records "
            "it once you accept it"
        )
        elsewhere = [
            'advanced.publish.html.key "/run/secrets/web_host_key": dropped, as the web host\'s SSH key is the '
            "secret web_host_ssh_key, installed with: rkmon secret web_host_ssh_key FILE",
            f'advanced.publish.html.known_hosts "/run/secrets/known_hosts"{own_key}',
            f'advanced.publish.html.host_keys ["web.example.com ssh-ed25519 <key>"]{own_key}',
            'advanced.publish.html.dest "/var/www/team/index.html": dropped, as the page\'s place is worked out '
            f"from {TEAM}.publish.site_folder",
            f"advanced.publish.html.url is now {TEAM}.public_url",
        ]
        for delivery in ("      delivery: publish\n", ""):
            with self.subTest(delivery=delivery):
                values, notes = upgraded(team.format(delivery))
                self.assertEqual(
                    at(values, f"{TEAM}.publish"),
                    {
                        "web_host": "web.example.com",
                        "username": "op-login",
                        "site_folder": "public_html/team",
                    },
                )
                self.assertEqual(
                    at(values, f"{TEAM}.public_url"), "https://dash.example.com/team"
                )
                self.assertEqual(
                    on(notes, "advanced.publish"),
                    [
                        f"advanced.publish.html.host is now {TEAM}.publish.web_host",
                        f"advanced.publish.html.user is now {TEAM}.publish.username",
                        f"advanced.publish.html.site is now {TEAM}.publish.site_folder",
                        *elsewhere,
                        f"advanced.publish.html.requests is now {TEAM}.priority_requests.enabled true; the requests "
                        f"file's place comes from {TEAM}.publish.site_folder",
                    ],
                )
        values, notes = upgraded(team.format("      delivery: serve\n"))
        self.assertIs(at(values, f"{TEAM}.publish"), GONE)
        self.assertEqual(
            at(values, f"{TEAM}.public_url"), "https://dash.example.com/team"
        )
        self.assertIs(at(values, f"{TEAM}.priority_requests.enabled"), True)
        dropped = (
            ': dropped, as the team dashboard\'s delivery is "serve", and only "publish" copies the page to a '
            "web host; the value stays in the earlier file, which migrate keeps"
        )
        self.assertEqual(
            on(notes, "advanced.publish"),
            [
                f'advanced.publish.html.host "web.example.com"{dropped}',
                f'advanced.publish.html.user "op-login"{dropped}',
                f'advanced.publish.html.site "public_html/team"{dropped}',
                *elsewhere,
                f"advanced.publish.html.requests is now {TEAM}.priority_requests.enabled true",
            ],
        )

    def test_the_update_source_is_the_updates_section(self):
        # A GitHub address or <org>/<repo>, with a branch, is the updates section; with anything else the section
        # is left out, and the note says why. update.path goes, as the package knows its own folder.
        moved = [
            'advanced.update.repo is now updates.repository "ROCm/my-repo"',
            "advanced.update.branch is now updates.branch",
        ]
        repo, branch = "advanced.update.repo", "advanced.update.branch"
        no_github = f"{repo} is no GitHub address or <org>/<repo>"

        def left_out(why: str) -> list:
            return [
                f"updates: left out, as {why}; rkmon template section updates prints the section"
            ]

        for update, expected, notes in (
            (
                "repo: https://github.com/ROCm/my-repo.git\n    branch: develop\n",
                {"repository": "ROCm/my-repo", "branch": "develop"},
                moved,
            ),
            (
                "repo: ROCm/my-repo\n    branch: develop\n",
                {"repository": "ROCm/my-repo", "branch": "develop"},
                moved,
            ),
            (
                'repo: ""\n    branch: develop\n',
                GONE,
                left_out(f"{repo} is blank, so {branch} is not carried either"),
            ),
            (
                "branch: develop\n",
                GONE,
                left_out(f"{repo} is missing, so {branch} is not carried either"),
            ),
            (
                "repo: /srv/git/my-repo\n    branch: develop\n",
                GONE,
                left_out(f"{no_github}, so {branch} is not carried either"),
            ),
            (
                'repo: origin\n    branch: ""\n',
                GONE,
                left_out(f"{no_github} and {branch} is blank"),
            ),
            (
                "repo: ROCm/my-repo\n",
                GONE,
                left_out(f"{branch} is missing, so {repo} is not carried either"),
            ),
            (
                "repo: ROCm/my-repo\n    branch: [develop]\n",
                GONE,
                left_out(
                    f"{branch} is no branch's name, so {repo} is not carried either"
                ),
            ),
            (
                'repo: ""\n    branch: ""\n',
                GONE,
                left_out(f"{repo} is blank and {branch} is blank"),
            ),
            (
                "path: tools/monitor\n",
                GONE,
                [
                    'advanced.update.path "tools/monitor": dropped, as the package knows its own folder in the '
                    "repository it comes from"
                ],
            ),
            ("drain_timeout_min: 60\n", GONE, []),
        ):
            with self.subTest(update=update):
                values, got = upgraded(f"advanced:\n  update:\n    {update}")
                self.assertEqual(at(values, "updates"), expected)
                self.assertEqual(on(got, "advanced.update", "updates"), notes)
        self.assertEqual(
            on(upgraded('advanced:\n  update:\n    path: ""\n')[1], "advanced"),
            ["advanced: 1 value the same as the package's defaults: dropped"],
        )

    def test_a_blank_user_or_group_id_is_null(self):
        values = upgraded(
            'host:\n  users:\n    monitor_uid: ""\n    monitor_gid: "1001"\n'
        )[0]
        self.assertEqual(
            values["host"], {"users": {"monitor_uid": None, "monitor_gid": 1001}}
        )
        values = upgraded(
            "host:\n  users:\n    monitor_uid: 1000\n    monitor_gid: null\n"
        )[0]
        self.assertEqual(
            values["host"], {"users": {"monitor_uid": 1000, "monitor_gid": None}}
        )

    def test_advanced_values_like_the_packages_are_counted_and_others_reported(self):
        values, notes = upgraded(
            "advanced:\n  mirror:\n    enabled: false\n  roles: {}\n  max_posts_per_hour: 6\n  nope: 1\n"
            "  offload:\n    mode: shadow\n"
        )
        self.assertIs(at(values, "advanced"), GONE)
        self.assertEqual(
            on(notes, "advanced"),
            [
                "advanced.nope is 1: not carried, as the package has no such setting",
                'advanced.offload.mode is "shadow" (the package\'s default is "off"): not carried, as layout 2 has '
                "no setting for it",
                "advanced: 3 values the same as the package's defaults: dropped",
            ],
        )
        self.assertEqual(
            on(upgraded("advanced:\n  max_posts_per_hour: 6\n")[1], "advanced"),
            ["advanced: 1 value the same as the package's defaults: dropped"],
        )
        self.assertEqual(
            on(upgraded("advanced: hello\n")[1], "advanced"),
            [
                'advanced is "hello": not carried, as it maps the package\'s keys to values'
            ],
        )

    def test_advanced_auth_keys_say_how_the_drafting_binding_sends_its_key(self):
        binding = "model_bindings:\n  worker:\n    provider: openai-compatible\n    model_id: m\n"
        sent_keys = ("api_key_sent_as", "api_key_header", "api_key_scheme")
        # (auth_mode, auth_header, auth_scheme; None: not in the file) -> what the binding sends.
        for mode, header, scheme, sent in (
            (
                "both",
                "X-Api-Key",
                None,
                {
                    "api_key_sent_as": "header_and_bearer",
                    "api_key_header": "X-Api-Key",
                    "api_key_scheme": "Bearer",
                },
            ),
            (
                "both",
                None,
                "Token",
                {
                    "api_key_sent_as": "header",
                    "api_key_header": "Authorization",
                    "api_key_scheme": "Token",
                },
            ),
            ("bearer", None, None, {"api_key_sent_as": "bearer"}),
            (
                "header",
                "X-Api-Key",
                '""',
                {"api_key_sent_as": "header", "api_key_header": "X-Api-Key"},
            ),
        ):
            with self.subTest(mode=mode, header=header, scheme=scheme):
                text = binding + f"advanced:\n  worker:\n    auth_mode: {mode}\n"
                if header:
                    text += f"    auth_header: {header}\n"
                if scheme:
                    text += f"    auth_scheme: {scheme}\n"
                worker = upgraded(text)[0]["model_bindings"]["worker"]
                self.assertEqual(
                    {k: v for k, v in worker.items() if k in sent_keys}, sent
                )
        notes = upgraded(
            binding
            + "advanced:\n  worker:\n    auth_header: X-Api-Key\n    auth_mode: both\n"
        )[1]
        self.assertIn(
            "advanced.worker.auth_header and advanced.worker.auth_mode are now model_bindings.worker.api_key_sent_as "
            '"header_and_bearer", model_bindings.worker.api_key_header "X-Api-Key" and '
            'model_bindings.worker.api_key_scheme "Bearer"',
            notes,
        )
        notes = upgraded(binding + "advanced:\n  worker:\n    auth_mode: both\n")[1]
        self.assertIn(
            'advanced.worker.auth_mode is now model_bindings.worker.api_key_sent_as "header", '
            'model_bindings.worker.api_key_header "Authorization" and model_bindings.worker.api_key_scheme "Bearer" '
            '(with the Authorization header, "both" sent the key once)',
            notes,
        )
        # A mode layout 1 did not name sent the key in the header, as "header" did, and the note says so.
        values, notes = upgraded(
            binding + "advanced:\n  worker:\n    auth_mode: token\n"
        )
        worker = values["model_bindings"]["worker"]
        self.assertEqual(
            {k: v for k, v in worker.items() if k in sent_keys},
            {
                "api_key_sent_as": "header",
                "api_key_header": "Authorization",
                "api_key_scheme": "Bearer",
            },
        )
        self.assertEqual(
            on(notes, "advanced.worker"),
            [
                'advanced.worker.auth_mode is now model_bindings.worker.api_key_sent_as "header", '
                'model_bindings.worker.api_key_header "Authorization" and model_bindings.worker.api_key_scheme '
                '"Bearer" (advanced.worker.auth_mode "token" read as "header", as layout 1 read it)'
            ],
        )
        # The package's own values are counted; with no such binding, a value of the file's own is reported.
        notes = upgraded(
            "advanced:\n  worker:\n    auth_header: Authorization\n    auth_mode: both\n"
        )[1]
        self.assertEqual(
            on(notes, "advanced"),
            [
                'advanced.worker.auth_mode is "both": not carried, as the file has no openai_compatible binding '
                "named worker to send a key",
                "advanced: 1 value the same as the package's defaults: dropped",
            ],
        )

    def test_advanced_automerge_guard_own(self):
        key = f"{MQ}.auto_merge_guard.on_your_pull_requests"
        for word, value in (("true", "same_as_others"), ("false", "alert_only")):
            with self.subTest(word=word):
                values, notes = upgraded(f"advanced:\n  automerge_guard_own: {word}\n")
                self.assertEqual(at(values, key), value)
                self.assertEqual(
                    on(notes, "advanced"),
                    [f'advanced.automerge_guard_own {word} is now {key} "{value}"'],
                )
        # A blank one is false, as layout 1 read it, and moves as false does.
        for word in ("null", '""'):
            with self.subTest(word=word):
                values, notes = upgraded(f"advanced:\n  automerge_guard_own: {word}\n")
                self.assertEqual(at(values, key), "alert_only")
                self.assertEqual(
                    on(notes, "advanced"),
                    [
                        f'advanced.automerge_guard_own {word} is now {key} "alert_only", as layout 1 read a blank '
                        "one as false"
                    ],
                )
        values, notes = upgraded("advanced:\n  automerge_guard_own: maybe\n")
        self.assertIs(at(values, key), GONE)
        self.assertEqual(
            on(notes, "advanced"),
            [
                'advanced.automerge_guard_own is "maybe": not carried, as a switch is true or false'
            ],
        )

    def test_jira_issue_type_and_extra_fields(self):
        values, notes = upgraded(
            "advanced:\n  jira:\n    issue_type: Bug\n    extra_fields:\n      customfield_1:\n        value: x\n"
        )
        self.assertEqual(
            (at(values, f"{TC}.issue_type"), at(values, f"{TC}.required_fields")),
            ("Bug", {"customfield_1": {"value": "x"}}),
        )
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.jira.issue_type is now {TC}.issue_type",
                f"advanced.jira.extra_fields is now {TC}.required_fields",
            ],
        )

    def test_host_mounts_are_dropped_with_a_note(self):
        values, notes = upgraded("host:\n  mounts: [/srv/keys/gpu_key, /srv/other]\n")
        self.assertIs(at(values, "host"), GONE)
        self.assertEqual(
            on(notes, "host"),
            [
                "host.mounts (2 entries): dropped, as the remote GPU tests' key is the secret gpu_ssh_key, installed "
                "with: rkmon secret gpu_ssh_key FILE"
            ],
        )
        self.assertEqual(on(upgraded("host:\n  mounts: []\n")[1], "host"), [])

    def test_the_confluence_site_is_dropped_with_a_note(self):
        values, notes = upgraded(
            "integrations:\n  atlassian_cli:\n    confluence_site: https://jira.example.com/wiki\n"
        )
        self.assertIs(at(values, "integrations"), GONE)
        self.assertEqual(
            on(notes, "integrations"),
            [
                'integrations.atlassian_cli.confluence_site "https://jira.example.com/wiki": dropped, as no '
                "feature reads it"
            ],
        )

    def test_each_effort_through_layout_1s_chain(self):
        keys = (
            f"{RA}.orchestrator.effort",
            f"{PASSES}.verification_pass.effort",
            f"{PASSES}.code_quality_pass.effort",
            f"{PASSES}.first_review_pass.effort",
            f"{PASSES}.ci_triage_pass.effort",
        )

        def efforts(
            orchestrator=None,
            verify=None,
            quality=None,
            first=None,
            triage=None,
            base=None,
        ):
            """(each role's effort, the notes) for a file with these efforts; None: not in the file."""
            text = ""
            if orchestrator is not None:
                text += f"monitor:\n  orchestrator:\n    effort: {orchestrator}\n"
            passes = "".join(
                f"      {name}:\n        effort: {effort}\n"
                for name, effort in (
                    ("verify", verify),
                    ("quality", quality),
                    ("first_pass", first),
                    ("ci_triage", triage),
                )
                if effort is not None
            )
            if passes:
                text += f"features:\n  review_automation:\n    passes:\n{passes}"
            if base is not None:
                text += f"advanced:\n  claude:\n    effort: {base}\n"
            values, notes = upgraded(text)
            return [at(values, k) for k in keys], notes

        # Its own; missing, the role's default; an optional pass's "default" stays.
        self.assertEqual(
            efforts("low", "xhigh", "max", "medium", "default")[0],
            ["low", "xhigh", "max", "medium", "default"],
        )
        self.assertEqual(efforts()[0], ["high", "high", "medium", GONE, GONE])
        # Blank: advanced.claude.effort, else the package's base effort; on an optional pass, "default".
        self.assertEqual(
            efforts('""', '""', '""', '""', '""')[0],
            ["high", "high", "high", "default", "default"],
        )
        values, notes = efforts('""', '""', '""', base="low")
        self.assertEqual(values, ["low", "low", "low", GONE, GONE])
        self.assertEqual(
            on(notes, "advanced"),
            [
                'advanced.claude.effort is "low" (the package\'s default is "high"): not carried, as each Claude '
                "role's effort is written as it resolved, and layout 2 has no base effort"
            ],
        )
        # Not an effort: as a blank one, with a note.
        values, notes = efforts("huge", quality="tiny", first="huge", base="medium")
        self.assertEqual(values, ["medium", "high", "medium", "default", GONE])
        self.assertEqual(
            [n for n in notes if ".effort " in n and not n.startswith("advanced")],
            [
                f'monitor.orchestrator.effort "huge" is now {RA}.orchestrator.effort "medium", from '
                "advanced.claude.effort, as layout 1 took no such effort",
                f'{PASSES}.quality.effort "tiny" is now {PASSES}.code_quality_pass.effort "medium", from '
                "advanced.claude.effort, as layout 1 took no such effort",
                f'{PASSES}.first_pass.effort "huge" is now "default", as layout 1 read it: the effort of the pass in '
                "charge",
            ],
        )
        # A chain that ended in the model's own effort: the role's default, which layout 2 names.
        values, notes = efforts('""', base='""')
        self.assertEqual(values, ["high", "high", "medium", GONE, GONE])
        self.assertIn(
            f'{RA}.orchestrator.effort: "high", its default: the pass ran on the model\'s own effort, as neither '
            "monitor.orchestrator.effort nor advanced.claude.effort named one, and layout 2 names one for a pass "
            "that always runs",
            notes,
        )

    def test_the_price_fields(self):
        values, notes = upgraded(
            "advanced:\n  prices:\n    claude-opus-5-5:\n      in: 15\n      out: 75\n      cache_read: 1.5\n"
            "      cache_write_5m: 18.75\n      cache_write_1h: 30\n      batch: 7\n"
        )
        self.assertEqual(
            at(values, f"{RA}.spend"),
            {
                "enabled": True,
                "usd_per_million_tokens": {
                    "claude-opus-5-5": {
                        "input": 15,
                        "output": 75,
                        "cache_read": 1.5,
                        "cache_write_5_minutes": 18.75,
                        "cache_write_1_hour": 30,
                    }
                },
            },
        )
        self.assertEqual(
            on(notes, "advanced"),
            [
                "advanced.prices.claude-opus-5-5.batch 7: not carried, as a price is one of in, out, cache_read, "
                "cache_write_5m and cache_write_1h",
                f"advanced.prices is now {RA}.spend.usd_per_million_tokens, by model id, with in, out, cache_write_5m "
                "and cache_write_1h named input, output, cache_write_5_minutes and cache_write_1_hour; "
                f"{RA}.spend.enabled true",
            ],
        )
        # The package's own (none): counted, and spend stays as layout 2 starts it.
        values, notes = upgraded("advanced:\n  prices: {}\n")
        self.assertIs(at(values, f"{RA}.spend"), GONE)
        self.assertEqual(
            on(notes, "advanced"),
            ["advanced: 1 value the same as the package's defaults: dropped"],
        )


class TestBuildRules(unittest.TestCase):
    """The step's rules beyond one value moved to one key, on the build file."""

    def test_the_claude_code_version_is_the_agent_sdks(self):
        values, notes = upgraded('claude_code:\n  version: "0.9.0"\n', BUILD)
        self.assertEqual(values, {**NAMED, "claude_agent_sdk": {"version": "0.9.0"}})
        self.assertEqual(
            notes,
            [
                "claude_code.version is now claude_agent_sdk.version",
                "claude_agent_sdk.package_url and .package_sha256 start blank: the package index serves it",
            ],
        )

    def test_the_two_package_indexes_are_one(self):
        index = "https://pypi.example.com/simple"
        # The same index (a / at the end is no difference): one.
        values, notes = upgraded(
            f"claude_code:\n  package_index: {index}/\npython_packages:\n  package_index: {index}\n",
            BUILD,
        )
        self.assertEqual(values, {**NAMED, "python_packages": {"package_index": index}})
        self.assertEqual(
            notes,
            [
                "claude_code.package_index: the same index as python_packages.package_index, the one for every "
                "Python package"
            ],
        )
        # Claude Code's alone, the index every other package came from already: it moves.
        values, notes = upgraded(f"claude_code:\n  package_index: {PYPI}\n", BUILD)
        self.assertEqual(values, {**NAMED, "python_packages": {"package_index": PYPI}})
        self.assertEqual(
            notes,
            [
                "claude_code.package_index is now python_packages.package_index, the one index for every Python "
                "package"
            ],
        )
        # Two different indexes: refused, naming both keys.
        for python, shown in (
            ("https://other.example.com/simple", '"https://other.example.com/simple"'),
            ('""', f"blank, which means {PYPI}"),
        ):
            with self.subTest(python=python):
                self.assertEqual(
                    refusal(
                        f"claude_code:\n  package_index: {index}\npython_packages:\n  package_index: {python}\n",
                        BUILD,
                    ),
                    [
                        (
                            "claude_code.package_index",
                            Status.FAIL,
                            f'claude_code.package_index is "{index}" and python_packages.package_index is {shown}, '
                            "and layout 2 has one package index for every Python package",
                            "set python_packages.package_index in your build settings to the one index every Python "
                            "package comes from, then delete claude_code.package_index from your build settings",
                        )
                    ],
                )

    def test_the_rocm_pins_are_the_packages(self):
        # This package's pins, its build defaults (rkmon_lib/build_plan.py).
        pinned, targets = build_plan.ROCM_CLANG_COMMIT, ",".join(
            build_plan.ROCM_LIBRARY_TARGETS
        )
        values, notes = upgraded(
            f"rocm:\n  clang_commit: {pinned}\n  gfx_targets: [{targets}]\n", BUILD
        )
        self.assertEqual(values, NAMED)
        self.assertEqual(
            notes,
            [
                f'rocm.clang_commit "{pinned}": dropped, the package\'s own',
                f'rocm.gfx_targets ["{targets}"]: dropped, the package\'s own',
            ],
        )
        values, notes = upgraded(
            "rocm:\n  clang_commit: any\n  gfx_targets: [gfx90a, gfx942]\n", BUILD
        )
        self.assertEqual(values, NAMED)
        self.assertEqual(
            notes,
            [
                f'rocm.clang_commit "any": dropped, as the package pins the clang commit itself ("{pinned}" for '
                f"ROCm {build_plan.ROCM_RELEASE})",
                f'rocm.gfx_targets ["gfx90a", "gfx942"]: dropped, as the package sets its GPU targets itself '
                f"({targets})",
            ],
        )
        # Another ROCm release: the package pins no commit for it, so "any", no pin, was the package's own too.
        values, notes = upgraded(
            'rocm:\n  version: "9.0.0"\n  clang_commit: any\n', BUILD
        )
        self.assertEqual(values, {**NAMED, "rocm": {"version": "9.0.0"}})
        self.assertEqual(
            notes, ['rocm.clang_commit "any": dropped, the package\'s own']
        )
        notes = upgraded('rocm:\n  version: "9.0.0"\n  clang_commit: abc123\n', BUILD)[
            1
        ]
        self.assertEqual(
            notes,
            [
                'rocm.clang_commit "abc123": dropped, as the package pins the clang commit itself (none for ROCm '
                "9.0.0)"
            ],
        )
        self.assertEqual(
            upgraded('rocm:\n  clang_commit: ""\n  gfx_targets: null\n', BUILD),
            (NAMED, []),
        )

    def test_the_pins_compared_are_the_package_build_defaults(self):
        """The step compares the file's pins with rkmon_lib/build_plan.py's, their one home: other build defaults,
        other notes."""
        other = "f" * 40
        with mock.patch.object(
            build_plan, "ROCM_CLANG_COMMIT", other
        ), mock.patch.object(build_plan, "ROCM_LIBRARY_TARGETS", ("gfx90a", "gfx942")):
            values, notes = upgraded(
                f"rocm:\n  clang_commit: {other}\n  gfx_targets: [gfx90a, gfx942]\n",
                BUILD,
            )
            self.assertEqual(values, NAMED)
            self.assertEqual(
                notes,
                [
                    f'rocm.clang_commit "{other}": dropped, the package\'s own',
                    'rocm.gfx_targets ["gfx90a", "gfx942"]: dropped, the package\'s own',
                ],
            )
        # A blank pin is compared with nothing.
        for key in ("rocm.clang_commit", "rocm.gfx_targets"):
            with self.subTest(key=key):
                self.assertEqual(upgraded(nested(key, '""'), BUILD), (NAMED, []))


class TestMoves(unittest.TestCase):
    """The step's moves: one value to one key, each with its note."""

    def test_the_deploy_files_keys(self):
        # The earlier key, a value as written, its key in layout 2, the value there.
        for old, written, new, value in (
            ("container.image.name", "my-image", "image.name", "my-image"),
            (
                "container.image.pull_registry",
                "registry.example.com/team",
                "image.registry",
                "registry.example.com/team",
            ),
            (
                "repository.path",
                "projects/my-project",
                "repository.folder",
                "projects/my-project",
            ),
            ("host.resources.memory_gb", "12", "host.resources.memory_gib", 12),
            ("host.resources.ramdisk_gb", "2", "host.resources.ramdisk_gib", 2),
            ("monitor.tick_min", "5", "monitor.tick_interval_minutes", 5),
            (
                "monitor.full_tick_min",
                "30",
                "monitor.full_tick_max_interval_minutes",
                30,
            ),
            (
                f"{RA}.schedule.push_cooldown_min",
                "20",
                f"{RA}.schedule.push_cooldown_minutes",
                20,
            ),
            (
                f"{RA}.schedule.follow_up_cooldown_min",
                "45",
                f"{RA}.schedule.follow_up_cooldown_minutes",
                45,
            ),
            (f"{RA}.capacity.parallel_reviews", "4", f"{RA}.parallel_reviews", 4),
            (
                f"{RA}.capacity.parallel_heavy_checks",
                "2",
                "monitor.parallel_builds",
                2,
            ),
            (
                f"{RA}.large_prs.manual_review_over_lines",
                "20000",
                f"{RA}.large_prs.in_depth_review_over_lines",
                20000,
            ),
            (f"{MQ}.auto_hold", "false", f"{MQ}.auto_hold.enabled", False),
            (f"{MQ}.develop_guard", "false", f"{GUARD}.enabled", False),
            (
                f"{ALERTS}.teams_channel",
                "reviews",
                f"{ALERTS}.message_label",
                "reviews",
            ),
            (
                f"{ALERTS}.message_cooldown_min",
                "15",
                f"{ALERTS}.message_cooldown_minutes",
                15,
            ),
            (f"{TC}.jira_project", "PROJ", f"{TC}.project_key", "PROJ"),
            (f"{TC}.default_epic", EPIC, f"{TC}.epic_key", EPIC),
            (
                "integrations.github_cli.account",
                "op-login",
                "integrations.github_cli.username",
                "op-login",
            ),
            (
                "integrations.atlassian_cli.jira_site",
                "https://jira.example.com",
                "integrations.atlassian_cli.jira_url",
                "https://jira.example.com",
            ),
            (
                "integrations.claude_code.gateway",
                "https://llm.example.com/claude",
                "integrations.claude_code.base_url",
                "https://llm.example.com/claude",
            ),
        ):
            with self.subTest(old=old):
                values, notes = upgraded(nested(old, written))
                self.assertEqual(at(values, new), value)
                self.assertIn(f"{old} is now {new}", notes)

    def test_the_keys_that_keep_their_names(self):
        values, notes = upgraded(
            "repository:\n  name: my-org/my-repo\n  scope_labels: [my-project]\nhost:\n  resources:\n    cpus: 6\n"
            "features:\n  review_automation:\n    enabled: true\n    mode: live\n    large_prs:\n"
            "      split_part_lines: 6000\n  merge_queue:\n    enabled: true\n    mode: live\n"
            "  dashboards:\n    admin:\n      enabled: false\n      port: 18620\n    team:\n      enabled: true\n"
            "      delivery: serve\n"
        )
        for key, value in (
            ("repository.name", "my-org/my-repo"),
            ("repository.scope_labels", ["my-project"]),
            ("host.resources.cpus", 6),
            (f"{RA}.enabled", True),
            (f"{RA}.mode", "live"),
            (f"{RA}.large_prs.split_part_lines", 6000),
            (f"{MQ}.enabled", True),
            (f"{MQ}.mode", "live"),
            (f"{ADMIN}.enabled", False),
            (f"{ADMIN}.port", 18620),
            (f"{TEAM}.enabled", True),
            (f"{TEAM}.delivery", "serve"),
        ):
            with self.subTest(key=key):
                self.assertEqual(at(values, key), value)
                self.assertEqual(on(notes, f"{key} is now"), [])

    def test_the_passes_new_names(self):
        values, notes = upgraded(
            "monitor:\n  orchestrator:\n    model: opus\n    effort: xhigh\nfeatures:\n  review_automation:\n"
            "    passes:\n"
            + "".join(
                f"      {p}:\n        model: {m}\n        effort: {e}\n"
                for p, m, e in (
                    ("verify", "opus", "high"),
                    ("quality", "sonnet", "medium"),
                    ("first_pass", "default", "default"),
                    ("ci_triage", "notices", "low"),
                    ("quality_draft", "default", "max"),
                    ("notice_draft", "notices", "default"),
                )
            )
        )
        self.assertEqual(
            at(values, f"{RA}.orchestrator"), {"model": "opus", "effort": "xhigh"}
        )
        self.assertEqual(
            at(values, PASSES),
            {
                "verification_pass": {"model": "opus", "effort": "high"},
                "code_quality_pass": {"model": "sonnet", "effort": "medium"},
                "first_review_pass": {"model": "default", "effort": "default"},
                "ci_triage_pass": {"model": "notices", "effort": "low"},
                "first_code_quality_pass": {"model": "default", "effort": "max"},
                "notice_draft_pass": {"model": "notices", "effort": "default"},
            },
        )
        self.assertEqual(
            [n for n in notes if " is now " in n],
            [f"monitor.orchestrator is now {RA}.orchestrator"]
            + [
                f"{PASSES}.{old} is now {PASSES}.{new}"
                for old, new in (
                    ("verify", "verification_pass"),
                    ("quality", "code_quality_pass"),
                    ("first_pass", "first_review_pass"),
                    ("ci_triage", "ci_triage_pass"),
                    ("quality_draft", "first_code_quality_pass"),
                    ("notice_draft", "notice_draft_pass"),
                )
            ],
        )

    def test_the_packages_user_ids(self):
        values, notes = upgraded(
            "host:\n  users:\n    github_writer_uid: 1999\n    pr_runner_uid: 1500\n"
        )
        self.assertIs(at(values, "host"), GONE)
        self.assertEqual(
            on(notes, "host"),
            [
                "host.users.github_writer_uid 1999: dropped, as that user's id is the package's (1999)",
                "host.users.pr_runner_uid 1500: dropped, as that user's id is the package's (1998)",
            ],
        )

    def test_the_human_keys_of_advanced(self):
        # Each key of advanced that moves as it is: a value other than the package's, as written, and as typed.
        written = {
            "gpu.enabled": ("true", True),
            "gpu.login_host": ("gpu-login.example.com", "gpu-login.example.com"),
            "gpu.ssh_port": ("2222", 2222),
            "gpu.ssh_user": ("op-login", "op-login"),
            "gpu.partition": ("gpu", "gpu"),
            "gpu.accounts": ("[acct-a]", ["acct-a"]),
            "gpu.constraint": ("big", "big"),
            "gpu.time_limit": ('"02:00:00"', "02:00:00"),
            "gpu.gres": ("\ngfx942: gpu-a", {"gfx942": "gpu-a"}),
            "gpu.default_arch": ("gfx942", "gfx942"),
            "gpu.exclude": ("[node-1]", ["node-1"]),
            "gpu.remote_dir": ("work", "work"),
            "gpu.remote_python": ("/usr/bin/python3", "/usr/bin/python3"),
            "gpu.setup": ('"source env.sh"', "source env.sh"),
            "gpu.clone_url": (
                "https://example.com/my-org/my-repo.git",
                "https://example.com/my-org/my-repo.git",
            ),
            "gpu.max_jobs_per_pr": ("3", 3),
            "gpu.pending_report_min": ("120", 120),
            "gpu.resubmits": ("2", 2),
            "weekly_report.enabled": ("true", True),
            "publish.html.host": ("web.example.com", "web.example.com"),
            "publish.html.user": ("op-login", "op-login"),
            "publish.html.site": ("/var/www/team", "/var/www/team"),
            "incident_issue.repo": ("my-org/my-issues", "my-org/my-issues"),
            "jira.labels": ("[monitor]", ["monitor"]),
            "jira.issue_type": ("Bug", "Bug"),
            "jira.extra_fields": ("\ncustomfield_1: x", {"customfield_1": "x"}),
            "publish.html.url": (
                "https://dash.example.com/team",
                "https://dash.example.com/team",
            ),
        }
        moves = {
            **dict(from_layout1.HUMAN),
            **{old: f"{TEAM}.publish.{new}" for old, new in from_layout1.WEB_HOST},
            "publish.html.url": f"{TEAM}.public_url",
        }
        self.assertEqual(set(written), set(moves))
        for old, (text, value) in written.items():
            with self.subTest(old=old):
                values, notes = upgraded(nested(f"advanced.{old}", text))
                self.assertEqual(at(values, moves[old]), value)
                self.assertIn(f"advanced.{old} is now {moves[old]}", notes)

    def test_the_human_keys_at_the_packages_values_are_counted(self):
        # Layout 2 writes the same values, so nothing is lost.
        values, notes = upgraded(
            "advanced:\n  gpu:\n    enabled: false\n    ssh_port: 22\n    remote_dir: rkmon-gpu\n"
            '  weekly_report:\n    enabled: false\n  jira:\n    issue_type: ""\n'
        )
        self.assertIs(at(values, RGT), GONE)
        self.assertEqual(
            on(notes, "advanced"),
            ["advanced: 5 values the same as the package's defaults: dropped"],
        )

    def test_the_gpu_jobs_environment(self):
        values, notes = upgraded(
            'advanced:\n  gpu:\n    env:\n      ROCKE_BACKEND: python\n      EXTRA: "1"\n'
        )
        self.assertEqual(at(values, f"{RGT}.node.environment"), {"EXTRA": "1"})
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.gpu.env is now {RGT}.node.environment, without ROCKE_BACKEND=python, which the "
                "repository's defaults set first"
            ],
        )
        values, notes = upgraded(
            "advanced:\n  gpu:\n    env:\n      ROCKE_BACKEND: cpp\n"
        )
        self.assertEqual(
            at(values, f"{RGT}.node.environment"), {"ROCKE_BACKEND": "cpp"}
        )
        self.assertEqual(
            on(notes, "advanced"), [f"advanced.gpu.env is now {RGT}.node.environment"]
        )

    def test_the_develop_guards_labels(self):
        # Layout 1's develop guard used the package's labels, and added "ci-red" beside the first.
        values, notes = upgraded("")
        self.assertEqual(
            at(values, GUARD),
            {
                "ci_red_label": "post-merge",
                "ci_red_extra_labels": ["ci-red"],
                "stability_label": "rocke-stability",
            },
        )
        self.assertEqual(on(notes, GUARD), LABEL_NOTES)
        values, notes = upgraded(
            "advanced:\n  incident_issue:\n    labels: [broken, red]\n    stability_label: flaky\n"
        )
        self.assertEqual(
            at(values, GUARD),
            {
                "ci_red_label": "broken",
                "ci_red_extra_labels": ["red", "ci-red"],
                "stability_label": "flaky",
            },
        )
        self.assertEqual(
            on(notes, "advanced"),
            [
                f'advanced.incident_issue.labels is now {GUARD}.ci_red_label "broken" and {GUARD}.ci_red_extra_labels '
                '["red", "ci-red"]; "ci-red" is the label layout 1\'s develop guard added beside the first, when the '
                "repository had it",
                f"advanced.incident_issue.stability_label is now {GUARD}.stability_label",
            ],
        )
        # With the guard off, layout 1's labels, given or not, are not written: the guard is its switch alone, and
        # the labels given are counted.
        values, notes = upgraded(
            "features:\n  merge_queue:\n    develop_guard: false\n"
            "advanced:\n  incident_issue:\n    labels: [post-merge]\n    stability_label: rocke-stability\n"
        )
        self.assertEqual(at(values, GUARD), {"enabled": False})
        self.assertEqual(
            on(notes, "advanced", f"{GUARD}."),
            ["advanced: 2 values the same as the package's defaults: dropped"],
        )
        # With the queue off, the package's labels are not written; labels of the file's own are.
        values = upgraded("features:\n  merge_queue:\n    enabled: false\n")[0]
        self.assertEqual(at(values, MQ), {"enabled": False})
        values = upgraded(
            "features:\n  merge_queue:\n    enabled: false\n"
            "advanced:\n  incident_issue:\n    labels: [broken, ci-red]\n"
        )[0]
        self.assertEqual(
            at(values, MQ),
            {
                "enabled": False,
                "develop_guard": {
                    "ci_red_label": "broken",
                    "ci_red_extra_labels": ["ci-red"],
                },
            },
        )

    def test_the_merged_keys(self):
        values, notes = upgraded(
            "advanced:\n  claude:\n    effort: low\n  worker:\n    think: low\n  tick_interval_min: 5\n"
            "  loop_log_max_mb: 50\n  jira:\n    url: https://jira.example.com/\n  team_dashboard:\n"
            "    url: https://dash.example.com/a\n  publish:\n    html:\n      url: https://dash.example.com/b\n"
            "integrations:\n  atlassian_cli:\n    jira_site: https://jira.example.com\n"
        )
        self.assertEqual(at(values, f"{TEAM}.public_url"), "https://dash.example.com/b")
        self.assertEqual(
            at(values, "integrations.atlassian_cli.jira_url"),
            "https://jira.example.com",
        )
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.publish.html.url is now {TEAM}.public_url",
                'advanced.team_dashboard.url is "https://dash.example.com/a": not carried, as '
                f'{TEAM}.public_url is "https://dash.example.com/b", from advanced.publish.html.url',
                "advanced.jira.url: the same as integrations.atlassian_cli.jira_site, the one "
                "integrations.atlassian_cli.jira_url",
                'advanced.claude.effort is "low" (the package\'s default is "high"): not carried, as each Claude '
                "role's effort is written as it resolved, and layout 2 has no base effort",
                'advanced.worker.think is "low" (the package\'s default is "max"): not carried, as a pass\'s effort '
                "is its own key in layout 2",
                "advanced.tick_interval_min is 5 (the package's default is 10): not carried, as "
                "monitor.tick_interval_minutes is the one tick interval in layout 2",
                "advanced.loop_log_max_mb is 50 (the package's default is 20): not carried, as the log's size limit "
                "is the package's in layout 2",
            ],
        )
        # Alone, each merged key with a home moves there.
        values, notes = upgraded(
            "advanced:\n  jira:\n    url: https://jira.example.com\n  team_dashboard:\n"
            "    url: https://dash.example.com/a\n"
        )
        self.assertEqual(
            (
                at(values, f"{TEAM}.public_url"),
                at(values, "integrations.atlassian_cli.jira_url"),
            ),
            ("https://dash.example.com/a", "https://jira.example.com"),
        )
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.team_dashboard.url is now {TEAM}.public_url",
                "advanced.jira.url is now integrations.atlassian_cli.jira_url",
            ],
        )
        # The team dashboard's address in both keys is one, a trailing "/" aside, as the JIRA address is.
        values, notes = upgraded(
            "advanced:\n  team_dashboard:\n    url: https://dash.example.com/a/\n  publish:\n    html:\n"
            "      url: https://dash.example.com/a\n"
        )
        self.assertEqual(at(values, f"{TEAM}.public_url"), "https://dash.example.com/a")
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.publish.html.url is now {TEAM}.public_url",
                "advanced.team_dashboard.url: the same as advanced.publish.html.url, the one "
                f"{TEAM}.public_url",
            ],
        )

    def test_two_jira_sites_are_refused(self):
        # Layout 2 has one home for the JIRA site: two different addresses are refused, naming both keys, and
        # a trailing "/" makes no difference.
        site = (
            "integrations:\n  atlassian_cli:\n    jira_site: https://jira.example.com\n"
        )
        self.assertEqual(
            refusal(f"advanced:\n  jira:\n    url: https://other.example.com/\n{site}"),
            [
                (
                    "advanced.jira.url",
                    Status.FAIL,
                    'advanced.jira.url is "https://other.example.com/" and integrations.atlassian_cli.jira_site is '
                    '"https://jira.example.com", and layout 2 has one address for your JIRA site',
                    "set integrations.atlassian_cli.jira_site in your deploy settings to your JIRA site's address, "
                    "then delete advanced.jira.url from your deploy settings",
                )
            ],
        )
        values = upgraded(
            f"advanced:\n  jira:\n    url: https://jira.example.com/\n{site}"
        )[0]
        self.assertEqual(
            at(values, "integrations.atlassian_cli.jira_url"),
            "https://jira.example.com",
        )
        self.assertIs(at(values, "advanced"), GONE)

    def test_a_bare_jira_site_name_is_its_cloud_address(self):
        # The atlassian CLI's name for a cloud site stands for its address (jira_address), compared and carried.
        url = "integrations.atlassian_cli.jira_url"

        def text(site: str = "", jira_url: str = "") -> str:
            return (
                f"advanced:\n  jira:\n    url: {jira_url}\n" if jira_url else ""
            ) + (
                f"integrations:\n  atlassian_cli:\n    jira_site: {site}\n"
                if site
                else ""
            )

        carried = (
            'integrations.atlassian_cli.jira_site "example" is now '
            f'{url} "{cloud("example")}", the address of the site it names'
        )
        # beside its address, one site: no refusal, and one address carried
        values, notes = upgraded(text("example", cloud("example") + "/"))
        self.assertEqual(at(values, url), cloud("example"))
        self.assertEqual(
            on(notes, "advanced", "integrations"),
            [
                carried,
                f"advanced.jira.url: the same as integrations.atlassian_cli.jira_site, the one {url}",
            ],
        )
        # alone, in either key, carried as its address
        values, notes = upgraded(text("example"))
        self.assertEqual(
            (at(values, url), on(notes, "integrations")), (cloud("example"), [carried])
        )
        values, notes = upgraded(text(jira_url="example"))
        self.assertEqual(at(values, url), cloud("example"))
        self.assertEqual(
            on(notes, "advanced"),
            [
                f'advanced.jira.url "example" is now {url} "{cloud("example")}", the address of the site it names'
            ],
        )
        # another name, or an address that is not the name's, is another site: refused
        for jira_url in (cloud("other"), "other", "https://jira.example.com"):
            with self.subTest(jira_url=jira_url):
                self.assertEqual(
                    [
                        (name, found)
                        for name, _status, found, _fix in refusal(
                            text("example", jira_url)
                        )
                    ],
                    [
                        (
                            "advanced.jira.url",
                            f'advanced.jira.url is "{jira_url}" and integrations.atlassian_cli.jira_site is "example", '
                            "and layout 2 has one address for your JIRA site",
                        )
                    ],
                )

    def test_the_requests_file_and_the_components(self):
        values, notes = upgraded(
            "advanced:\n  publish:\n    html:\n      requests: /var/www/team/requests.json\n  jira:\n"
            '    component: "Tools, Platform"\n'
        )
        self.assertEqual(at(values, f"{TEAM}.priority_requests"), {"enabled": True})
        self.assertEqual(at(values, f"{TC}.components"), ["Tools", "Platform"])
        self.assertEqual(
            on(notes, "advanced"),
            [
                f"advanced.publish.html.requests is now {TEAM}.priority_requests.enabled true; the requests file's "
                f"place comes from {TEAM}.publish.site_folder",
                f'advanced.jira.component is now {TC}.components, split at commas: ["Tools", "Platform"]',
            ],
        )

    def test_the_build_files_keys(self):
        # The earlier key, a value as written, its key in layout 2, the value there.
        for old, written, new, value in (
            ("image.name", "my-image", "image.name", "my-image"),
            ("image.tag", '"1.2.3"', "image.tag", "1.2.3"),
            (
                "image.push_registry",
                "registry.example.com/team",
                "image.registry",
                "registry.example.com/team",
            ),
            (
                "image.base",
                '"ubuntu:24.04"',
                "operating_system.base_image",
                "ubuntu:24.04",
            ),
            (
                "image.ca_certificates",
                "[/etc/ssl/extra.pem]",
                "operating_system.ca_certificates",
                ["/etc/ssl/extra.pem"],
            ),
            (
                "os_packages.package_mirror",
                "http://mirror.example.com/ubuntu",
                "operating_system.package_mirror",
                "http://mirror.example.com/ubuntu",
            ),
            ("rocm.version", '"10.1.0"', "rocm.version", "10.1.0"),
            (
                "rocm.package_url",
                "https://example.com/rocm.deb",
                "rocm.package_url",
                "https://example.com/rocm.deb",
            ),
            ("rocm.package_sha256", SHA, "rocm.package_sha256", SHA),
            ("github_cli.version", '"2.60.0"', "github_cli.version", "2.60.0"),
            (
                "github_cli.package_url",
                "https://example.com/gh.deb",
                "github_cli.package_url",
                "https://example.com/gh.deb",
            ),
            ("github_cli.package_sha256", SHA, "github_cli.package_sha256", SHA),
            ("atlassian_cli.version", '"1.0.0"', "atlassian_cli.version", "1.0.0"),
            (
                "atlassian_cli.package_url",
                "https://example.com/acli",
                "atlassian_cli.package_url",
                "https://example.com/acli",
            ),
            ("atlassian_cli.package_sha256", SHA, "atlassian_cli.package_sha256", SHA),
            (
                "python_packages.package_index",
                "https://pypi.example.com/simple",
                "python_packages.package_index",
                "https://pypi.example.com/simple",
            ),
        ):
            with self.subTest(old=old):
                values, notes = upgraded(nested(old, written), BUILD)
                section, key = new.split(".")
                expected = copy.deepcopy(NAMED)
                expected.setdefault(section, {})[key] = value
                self.assertEqual(values, expected)
                self.assertEqual(notes, [f"{old} is now {new}"] if old != new else [])

    def test_each_build_move_is_its_new_keys_x_internal(self):
        # The keys that move as they are, frozen beside the step: layout 2's build schema names each in its x-internal.
        internal = {
            path: node.get("x-internal")
            for path, node, _required in settings_file.schema_keys(BUILD.load())
        }
        for old, new in from_layout1.BUILD_MOVES.items():
            with self.subTest(old=old):
                self.assertEqual(internal[new], old)

    def test_a_model_binding_reshaped(self):
        # Each provider's keys.
        values, notes = upgraded(
            "model_bindings:\n"
            "  big:\n    provider: claude\n    model_id: claude-opus-5-5\n"
            "  local:\n    provider: ollama\n    endpoint: http://localhost:11434\n    model_id: example-model\n"
            "    limits:\n      max_parallel: 2\n"
            "  remote:\n    provider: openai-compatible\n    endpoint: https://llm.example.com/v1\n"
            "    model_id: example-model\n    api_key_secret: remote_api_key\n    limits:\n"
            "      requests_per_min: 30\n      tokens_per_min: 1000\n      max_parallel: 4\n      timeout_min: 5\n"
        )
        self.assertEqual(
            values["model_bindings"],
            {
                "big": {"provider": "claude", "model_id": "claude-opus-5-5"},
                "local": {
                    "provider": "ollama",
                    "model_id": "example-model",
                    "server_url": "http://localhost:11434",
                    "limits": {"max_parallel_requests": 2},
                },
                "remote": {
                    "provider": "openai_compatible",
                    "model_id": "example-model",
                    "base_url": "https://llm.example.com/v1",
                    "limits": {
                        "requests_per_minute": 30,
                        "tokens_per_minute": 1000,
                        "max_parallel_requests": 4,
                    },
                    "api_key_secret_name": "remote_api_key",
                    "stall_timeout_seconds": 300,
                },
            },
        )
        self.assertEqual(
            on(notes, "model_bindings"),
            [
                "model_bindings.local.endpoint is now model_bindings.local.server_url",
                "model_bindings.local.limits.max_parallel is now model_bindings.local.limits.max_parallel_requests",
                'model_bindings.remote.provider "openai-compatible" is now "openai_compatible"',
                "model_bindings.remote.endpoint is now model_bindings.remote.base_url",
                "model_bindings.remote.limits.requests_per_min is now model_bindings.remote.limits.requests_per_minute",
                "model_bindings.remote.limits.tokens_per_min is now model_bindings.remote.limits.tokens_per_minute",
                "model_bindings.remote.limits.max_parallel is now model_bindings.remote.limits.max_parallel_requests",
                "model_bindings.remote.api_key_secret is now model_bindings.remote.api_key_secret_name",
                "model_bindings.remote.limits.timeout_min 5 is now model_bindings.remote.stall_timeout_seconds 300",
            ],
        )

    def test_a_binding_layout_1_did_not_take(self):
        values, notes = upgraded(
            "model_bindings:\n  a: text\n  b:\n    provider: gemini\n    model_id: m\n  c:\n    model_id: m\n"
            "  d:\n    provider: ollama\n    model_id: m\n    colour: blue\n"
        )
        self.assertEqual(
            values["model_bindings"],
            {
                "d": {
                    "provider": "ollama",
                    "model_id": "m",
                    "limits": {"max_parallel_requests": 0},
                }
            },
        )
        self.assertEqual(
            on(notes, "model_bindings"),
            [
                'model_bindings.a is "text": not carried, as a binding is a section',
                'model_bindings.b {"model_id": "m"}: not carried, as its provider is "gemini", and layout 1\'s '
                "providers are claude, ollama and openai-compatible",
                'model_bindings.c {"model_id": "m"}: not carried, as its provider is missing, and layout 1\'s '
                "providers are claude, ollama and openai-compatible",
                "model_bindings.d.limits.max_parallel_requests: 0, what the binding ran on with no "
                "limits.max_parallel: no cap",
                'model_bindings.d.colour "blue": not carried, as a binding of provider "ollama" has no such key',
            ],
        )


class TestStep(unittest.TestCase):
    """What a step does beyond its rules."""

    def test_the_values_given_are_not_changed(self):
        for text, step, file_type in (
            (FULL_DEPLOY, from_layout1.deploy, DEPLOY),
            (FULL_BUILD, from_layout1.build, BUILD),
        ):
            values = loaded(text.split("\n", 3)[3])
            before = copy.deepcopy(values)
            first = step(values, file_type)
            self.assertEqual(values, before)
            self.assertEqual(step(values, file_type), first)

    def test_a_value_carried_to_a_new_key_is_a_move_upgrade_records(self):
        # Its note is a settings_file.Moved, so migrate can name the file's own key for a value it cannot write: one
        # case for each rule that carries a value as it is to a new key.
        doc = settings_file.upgrade(
            Document(
                DEPLOY,
                1,
                loaded(
                    "monitor:\n  tick_min: 5\n"
                    "features:\n  review_automation:\n    passes:\n      verify:\n        model: opus\n"
                    "  dashboards:\n    team:\n      enabled: true\n      delivery: serve\n      port: 18700\n"
                    "model_bindings:\n  remote:\n    provider: openai-compatible\n    model_id: m\n"
                    "    endpoint: https://llm.example.com/v1\n    api_key_secret: remote_key\n"
                    "advanced:\n  gpu:\n    login_host: gpu-login.example.com\n    env:\n      A: b\n"
                    "  prices:\n    m:\n      in: 1\n"
                    "  incident_issue:\n    stability_label: own-label\n"
                    "  team_dashboard:\n    url: https://team.example.com\n"
                ),
            )
        )[0]
        self.assertEqual(
            doc.moves,
            (
                {
                    "monitor.tick_interval_minutes": "monitor.tick_min",
                    f"{PASSES}.verification_pass": f"{PASSES}.verify",
                    f"{TEAM}.serve.port": f"{TEAM}.port",
                    "model_bindings.remote.base_url": "model_bindings.remote.endpoint",
                    "model_bindings.remote.api_key_secret_name": "model_bindings.remote.api_key_secret",
                    f"{GUARD}.stability_label": "advanced.incident_issue.stability_label",
                    f"{RGT}.cluster.login_host": "advanced.gpu.login_host",
                    f"{RGT}.node.environment": "advanced.gpu.env",
                    f"{RA}.spend.usd_per_million_tokens": "advanced.prices",
                    f"{TEAM}.public_url": "advanced.team_dashboard.url",
                },
            ),
        )
        build = settings_file.upgrade(
            Document(BUILD, 1, loaded(f"claude_code:\n  package_index: {PYPI}/\n"))
        )[0]
        self.assertEqual(
            build.moves,
            ({"python_packages.package_index": "claude_code.package_index"},),
        )

    def test_the_step_is_given_its_file_type_and_imports_none(self):
        # file_types builds each file type with its step; the step reaches back into it at no time, not even for a
        # placeholder of layout 2 (a label's folder) or its section command (a pass's model "worker").
        values = {
            "repository": {"path_labels": ["shared-a"]},
            "features": {
                "review_automation": {"passes": {"first_pass": {"model": "worker"}}}
            },
        }
        code = (
            "import sys\n"
            "from rkmon_lib import file_types, from_layout1\n"
            "del sys.modules['rkmon_lib.file_types']\n"
            f"notes = from_layout1.deploy({values!r}, file_types.DEPLOY)[1]\n"
            "from_layout1.build({}, file_types.BUILD)\n"
            "print(len(notes) > 2, 'rkmon_lib.file_types' in sys.modules)\n"
        )
        done = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(done.stdout, "True False\n")

    def test_a_blank_value_is_not_carried_and_makes_no_note(self):
        values, notes = upgraded(
            'repository:\n  name: ""\n  scope_labels: null\nfeatures:\n  alerts:\n    teams_channel: ""\n'
            "  review_automation:\n    passes:\nnope: null\n"
        )
        self.assertIs(at(values, "repository"), GONE)
        self.assertIs(at(values, f"{ALERTS}.message_label"), GONE)
        self.assertEqual(on(notes, "repository", f"{ALERTS}.teams", "nope", RA), [])

    def test_a_value_no_rule_takes_is_reported(self):
        notes = upgraded(
            "nope: 1\nfeatures:\n  review_automation:\n    colour: blue\n    enabled: maybe\n"
        )[1]
        self.assertEqual(
            on(notes, "nope", RA),
            [
                f'{RA}.enabled is "maybe": not carried, as a switch is true or false; it meant true, its default',
                "nope 1: not carried, as it is no setting of layout 1",
                f'{RA}.colour "blue": not carried, as it is no setting of layout 1',
            ],
        )
        self.assertEqual(
            upgraded("nope: 1\nimage:\n  colour: blue\n", BUILD)[1],
            [
                "nope 1: not carried, as it is no setting of layout 1",
                'image.colour "blue": not carried, as it is no setting of layout 1',
            ],
        )

    def test_a_long_value_is_cut_in_its_note(self):
        notes = upgraded("nope: " + "x" * 80 + "\n")[1]
        self.assertEqual(
            notes[-1],
            'nope "' + "x" * 58 + "…: not carried, as it is no setting of layout 1",
        )

    def test_what_layout_1_meant_is_frozen_beside_the_step(self):
        """The step means the same in every later package: its frozen values agree with layout 1's frozen
        defaults, and the package's own defaults, binding defaults, image name and package index may change while
        the step writes the same."""
        with open(from_layout1.DEFAULTS_FILE, encoding="utf-8") as f:
            frozen = json.load(f)
        self.assertEqual(from_layout1.BASE_EFFORT, frozen["claude"]["effort"])
        self.assertEqual(
            list(from_layout1.GUARD_LABELS), frozen["incident_issue"]["labels"]
        )
        self.assertEqual(
            from_layout1.STABILITY_LABEL, frozen["incident_issue"]["stability_label"]
        )
        self.assertEqual(
            {k: from_layout1.BINDING_DEFAULTS[k] for k in from_layout1.AUTH_KEYS},
            {k: frozen["worker"][k] for k in from_layout1.AUTH_KEYS},
        )
        deploy = loaded(
            'container:\n  image:\n    name: ""\nfeatures:\n  merge_queue:\n    develop_guard: true\n'
            "  review_automation:\n    passes:\n      verify:\n        model: none\n        effort: none\n"
            "model_bindings:\n  drafter:\n    provider: openai-compatible\n    model_id: m\n"
            "  big:\n    provider: claude\n    model_id: m\n"
            "advanced:\n  model_bindings:\n    drafter:\n      auth_header: Authorization\n      temperature: 0.3\n"
            "    big:\n      auth_header: x-api-key\n      chat_path: /v1/messages\n"
        )
        build = loaded(f"claude_code:\n  package_index: {PYPI}\n")

        def stepped():
            return (
                from_layout1.deploy(deploy, DEPLOY),
                from_layout1.build(build, BUILD),
            )

        before = stepped()
        self.assertEqual(
            on(before[0][1], "advanced"),
            ["advanced: 4 values the same as the package's defaults: dropped"],
        )
        self.assertEqual(
            at(before[0][0], f"{PASSES}.verification_pass.model"),
            from_layout1.ALWAYS_RUNS_MODEL,
        )
        real_key = registry.key

        def other_defaults(name):
            # the package's base effort and orchestrator model at other defaults, in their one home
            k = real_key(name)
            other = {"claude.effort": "low", "orchestrator.model": "sonnet"}
            return k._replace(default=other.get(name, k.default))

        with mock.patch.dict(
            bindings.binding_defaults(),
            {"auth_header": "X-Other", "temperature": 0.9},
        ), mock.patch.dict(
            bindings.PROVIDER_DEFAULTS,
            {"anthropic": {"auth_header": "x-other", "chat_path": "/other"}},
        ), mock.patch.object(
            image, "DEFAULT_NAME", "other-image"
        ), mock.patch.object(
            build_plan, "PUBLIC_PYPI", "https://other.example.com/simple"
        ), mock.patch.object(
            registry, "key", other_defaults
        ):
            self.assertEqual(stepped(), before)

    def test_a_step_uses_no_network(self):
        def refused(*args, **kwargs):
            raise AssertionError("a step reached for the network")

        with mock.patch.object(socket, "socket", refused), mock.patch.object(
            socket, "create_connection", refused
        ), mock.patch.object(socket, "getaddrinfo", refused):
            from_layout1.deploy(loaded(FULL_DEPLOY.split("\n", 3)[3]), DEPLOY)
            from_layout1.build(loaded(FULL_BUILD.split("\n", 3)[3]), BUILD)


class TestLayout1Rules(unittest.TestCase):
    """Layout 1's rules, each a public function on plain values that a translation of layout 1's configuration
    calls too: a golden test each, on a value as a file has it (answers.Plain) and as a configuration has it (typed),
    MISSING for a key that is not there."""

    P = answers.Plain
    MISSING = from_layout1.MISSING

    def test_leaves(self):
        values = {
            "a": {"b": 1, "c": {}},
            "gpu": {"env": {"X": "1"}},
            "roles": {"verify": "opus"},
        }
        self.assertEqual(
            list(from_layout1.leaves(values)),
            [("a.b", 1), ("gpu.env", {"X": "1"}), ("roles", {"verify": "opus"})],
        )

    def test_the_containers_and_the_images_names(self):
        for value in ("", None, self.P(""), self.MISSING):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.container_name(value), "rkmon")
                self.assertEqual(from_layout1.image_name(value), IMAGE_NAME)
                self.assertEqual(from_layout1.image_tag(value, self.P("box")), "box")
        self.assertEqual(from_layout1.container_name(self.P("box")), "box")
        self.assertEqual(from_layout1.image_name(self.P("mine")), "mine")
        self.assertEqual(from_layout1.image_tag(self.P("v1"), "box"), "v1")

    def test_labels_and_shared_label_folders(self):
        for value, expected in (
            ("", []),
            (self.MISSING, []),
            ([self.P("a"), self.P(""), "b"], ["a", "b"]),
            ([], []),
            (self.P("a"), None),
            ([["a"]], None),
            ([{"a": 1}], None),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.labels(value), expected)
        self.assertEqual(
            from_layout1.shared_label_folders(["a", "b"], "src/tools"),
            {"a": "src/tools", "b": "src/tools"},
        )

    def test_a_user_or_group_id(self):
        for value, expected in (
            ("", None),
            (None, None),
            (self.P("1000"), 1000),
            (" 1000 ", 1000),
            (1000, 1000),
            (True, True),
            (self.P("abc"), "abc"),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.user_id(value), expected)

    def test_auto_merge_and_the_guard_on_your_pull_requests(self):
        for value, expected in (
            (self.P("enabled"), True),
            ("hints_only", False),
            ("", None),
            ("always", None),
        ):
            with self.subTest(value=value):
                self.assertIs(from_layout1.auto_merge_enabled(value), expected)
        for value, expected in (
            (self.P("true"), "same_as_others"),
            (False, "alert_only"),
            (self.P(""), "alert_only"),  # a blank one is false, as layout 1 read it
            (None, "alert_only"),
            (self.P("maybe"), None),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.guard_own(value), expected)

    def test_the_develop_guards_labels(self):
        for names, expected in (
            ([], ("post-merge", ["ci-red"])),
            (["incident", "triage"], ("incident", ["triage", "ci-red"])),
            (["incident", "ci-red"], ("incident", ["ci-red"])),
            (["ci-red"], ("ci-red", [])),
        ):
            with self.subTest(names=names):
                self.assertEqual(from_layout1.ci_red_labels(names), expected)
        self.assertEqual(from_layout1.stability_label(""), "rocke-stability")
        self.assertEqual(from_layout1.stability_label(self.MISSING), "rocke-stability")
        self.assertEqual(from_layout1.stability_label(self.P("calm")), "calm")

    def test_which_of_the_develop_guards_labels_are_written(self):
        # Each group while the guard is in use (it and the merge queue on), or when it is the file's own; a guard
        # that is off and keeps its defaults is written as its switch alone.
        own, in_use = from_layout1.LABELS_OWN, from_layout1.GUARD_IN_USE
        defaults = ([self.P("post-merge")], self.P("rocke-stability"))
        for queue_on, guard_on, ci_red, stability, expected in (
            (True, True, self.MISSING, self.MISSING, (in_use, in_use)),
            (True, True, *defaults, (in_use, in_use)),
            (True, False, *defaults, (None, None)),
            (False, True, *defaults, (None, None)),
            (True, False, self.MISSING, "", (None, None)),
            # a list that names no label is the guard's own labels; a value that is no list of labels is no label
            (True, False, [self.P(""), " "], None, (None, None)),
            (True, False, self.P("post-merge"), None, (None, None)),
            # the file's own, whatever the switches
            (False, False, [self.P("broken")], self.P("flaky"), (own, own)),
            (True, False, ["post-merge", "ci-red"], self.MISSING, (own, None)),
            (True, True, self.MISSING, "flaky", (in_use, own)),
        ):
            with self.subTest(
                queue_on=queue_on, guard_on=guard_on, ci_red=ci_red, stability=stability
            ):
                self.assertEqual(
                    from_layout1.guard_labels_written(
                        queue_on, guard_on, ci_red, stability
                    ),
                    expected,
                )

    def test_the_merge_queues_notices(self):
        # Without the source's own switches: queue notices with the queue, drift notices with review automation too.
        for queue_on, ra_on, expected in (
            (True, True, (True, True)),
            (True, False, (True, False)),
            (False, True, (False, False)),
        ):
            with self.subTest(queue_on=queue_on, ra_on=ra_on):
                self.assertEqual(from_layout1.notices(queue_on, ra_on), expected)
        # With them, the source's own.
        self.assertEqual(
            from_layout1.notices(True, False, queue=False, drift=True), (False, True)
        )

    def test_a_features_mode_and_the_ticket_types(self):
        self.assertEqual(from_layout1.feature_mode(True), "live")
        self.assertIsNone(from_layout1.feature_mode(False))
        for args, expected in (
            ((True,), (True, True)),  # a missing trigger is on
            ((True, None, False), (True, False)),
            ((True, False, False), (False, False)),
            ((False, True, True), (False, False)),
        ):
            with self.subTest(args=args):
                self.assertEqual(from_layout1.ticket_types(*args), expected)

    def test_the_team_dashboards_delivery_and_its_block(self):
        for value, delivery, block in (
            ("", "publish", "publish"),
            (self.MISSING, "publish", "publish"),
            (self.P("serve"), "serve", "serve"),
            ("email", "email", None),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.team_delivery(value), delivery)
                self.assertEqual(from_layout1.team_block(delivery), block)

    def test_the_alerts_switches(self):
        for on in (True, False):
            with self.subTest(on=on):
                self.assertEqual(
                    from_layout1.alert_switches(on),
                    {"develop_breaks": on, "needs_you": on},
                )

    def test_the_jira_components(self):
        for value, expected in (
            (self.P("Tools, Platform"), ["Tools", "Platform"]),
            (" , Tools ,, Platform ", ["Tools", "Platform"]),
            (self.P("5"), None),
            (["Tools"], None),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.jira_components(value), expected)

    def test_two_addresses_name_one_place_but_for_a_trailing_slash(self):
        for a, b, expected in (
            ("https://jira.example.com/", self.P("https://jira.example.com"), True),
            ("https://dash.example.com/a", "https://dash.example.com/a/", True),
            ("https://dash.example.com/a", "https://dash.example.com/b", False),
        ):
            with self.subTest(a=a, b=b):
                self.assertIs(from_layout1.same_address(a, b), expected)

    def test_a_bare_jira_site_name_is_its_cloud_address(self):
        # The atlassian CLI's name for a cloud site, one label, stands for its address; any other value is as it is.
        self.assertEqual(
            from_layout1.CLOUD_SITE.format(name="example"), cloud("example")
        )
        for value, expected in (
            (self.P("example"), cloud("example")),
            (" my-site2 ", cloud("my-site2")),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.jira_address(value), expected)
        for value in (
            self.P("https://jira.example.com"),
            cloud("example"),
            "jira.example.com",
            "-site",
            "two words",
            "",
            self.MISSING,
        ):
            with self.subTest(value=value):
                self.assertIs(from_layout1.jira_address(value), value)
        # compared as the address it stands for
        for a, b, expected in (
            ("example", cloud("example") + "/", True),
            (self.P("example"), "example", True),
            ("example", cloud("other"), False),
            ("example", "other", False),
            ("example", "https://jira.example.com", False),
        ):
            with self.subTest(a=a, b=b):
                self.assertIs(
                    from_layout1.same_address(
                        from_layout1.jira_address(a), from_layout1.jira_address(b)
                    ),
                    expected,
                )

    def test_priority_requests(self):
        # on while the value names a requests file; false, 0, null and blank text name none, as layout 1 read them
        for value in (self.P("/var/www/requests.json"), "requests.jsonl", True):
            with self.subTest(value=value):
                self.assertIs(from_layout1.priority_requests(value), True)
        for value in (
            "",
            " ",
            None,
            self.MISSING,
            False,
            0,
            self.P("false"),
            self.P("0"),
            self.P("null"),
        ):
            with self.subTest(value=value):
                self.assertIs(from_layout1.priority_requests(value), False)

    def test_the_prices_and_the_spend_switch(self):
        self.assertEqual(
            from_layout1.model_prices(
                {
                    "in": 3,
                    "out": 15,
                    "cache_read": 0.3,
                    "cache_write_5m": 3.75,
                    "cache_write_1h": 6,
                    "batch": 1,
                }
            ),
            (
                {
                    "input": 3,
                    "output": 15,
                    "cache_read": 0.3,
                    "cache_write_5_minutes": 3.75,
                    "cache_write_1_hour": 6,
                },
                {"batch": 1},
            ),
        )
        self.assertIs(from_layout1.spend_enabled({}), False)
        self.assertIs(from_layout1.spend_enabled({"m": {"input": 3}}), True)

    def test_the_gpu_jobs_environment(self):
        self.assertEqual(
            from_layout1.gpu_environment({"ROCKE_BACKEND": self.P("python"), "A": "1"}),
            {"A": "1"},
        )
        self.assertEqual(
            from_layout1.gpu_environment({"ROCKE_BACKEND": "cpp"}),
            {"ROCKE_BACKEND": "cpp"},
        )

    def test_the_effort_chain(self):
        own, owner, base, role = (
            from_layout1.FROM_OWN,
            from_layout1.FROM_OWNER,
            from_layout1.FROM_BASE,
            from_layout1.FROM_ROLE,
        )
        self.assertEqual(from_layout1.base_effort(), "high")
        self.assertEqual(from_layout1.base_effort(self.P("LOW")), "low")
        self.assertEqual(from_layout1.base_effort(""), "")  # the model's own effort
        for args, expected in (
            (("verify", self.P("low")), ("low", own)),
            (("verify",), ("high", own)),  # a missing one: the role's default
            (("quality",), ("medium", own)),
            (("first_pass", "bogus"), ("default", owner)),
            (("first_pass",), ("default", owner)),
            (("verify", "bogus", "medium"), ("medium", base)),
            (("verify", "bogus", ""), ("high", role)),
            (("quality", "", ""), ("medium", role)),
        ):
            with self.subTest(args=args):
                self.assertEqual(from_layout1.pass_effort(*args), expected)

    def test_a_passs_model(self):
        for args, expected in (
            ((self.P("none"), True), "default"),
            (("", True), "default"),
            (("claude-opus", True), "default"),
            (("worker", True, True), "worker"),
            (("worker", True, False), "default"),
            ((self.P("sonnet"), True), "sonnet"),
            (("", False), None),
            ((self.P("opus"), False), "opus"),
            ((self.P("default"), False), "opus"),
            ((self.P("none"), False), "opus"),
            ((self.P("off"), False), "off"),
        ):
            with self.subTest(args=args):
                self.assertEqual(from_layout1.pass_model(*args), expected)

    def test_a_bindings_key(self):
        self.assertEqual(
            from_layout1.api_key_secret_name(self.P("worker_api_key")),
            "first_review_pass_api_key",
        )
        # Every binding's key's name ends in _api_key (secrets_registry.BINDING_KEY_SUFFIX): a name without that end
        # gets it, in the place of a last _key, and the one-time move copies the secret to it.
        for given, name in (
            ("other_key", "other_api_key"),
            ("glm", "glm_api_key"),
            ("gw_token", "gw_token_api_key"),
            ("other_api_key", "other_api_key"),
        ):
            with self.subTest(given=given):
                self.assertEqual(from_layout1.api_key_secret_name(self.P(given)), name)
        self.assertEqual(from_layout1.api_key_secret_name("Not-A-Name"), "Not-A-Name")
        claude = from_layout1.binding_defaults("claude")
        self.assertEqual(
            (claude["auth_header"], claude["chat_path"]), ("x-api-key", "/v1/messages")
        )
        self.assertEqual(
            from_layout1.binding_defaults("ollama")["auth_header"], "Authorization"
        )
        for args, expected in (
            (
                ("", "", ""),
                {"api_key_sent_as": "header", "api_key_header": "Authorization"},
            ),
            (("bearer", "X-Key", "Token"), {"api_key_sent_as": "bearer"}),
            (
                (self.P("both"), "X-Key", ""),
                {"api_key_sent_as": "header_and_bearer", "api_key_header": "X-Key"},
            ),
            (
                ("both", "authorization", "Bearer"),
                {
                    "api_key_sent_as": "header",
                    "api_key_header": "authorization",
                    "api_key_scheme": "Bearer",
                },
            ),
            (
                ("token", "X-Key", ""),
                {"api_key_sent_as": "header", "api_key_header": "X-Key"},
            ),
        ):
            with self.subTest(args=args):
                self.assertEqual(from_layout1.key_sending(*args), expected)

    def test_a_bindings_stall_timeout(self):
        for name, minutes, expected in (
            ("other", self.P("2"), 120),
            ("other", 2.0, 120),
            ("worker", 15, 900),
            ("worker", "", 900),
            ("worker", self.MISSING, 900),
            ("other", "", None),
            ("other", 0, None),
            ("other", 1.5, None),
            ("other", True, None),
        ):
            with self.subTest(name=name, minutes=minutes):
                self.assertEqual(
                    from_layout1.stall_timeout_seconds(name, minutes), expected
                )

    def test_an_address_on_this_host(self):
        for url, expected in (
            ("http://127.0.0.1:11434", True),
            (self.P("http://localhost:11434/"), True),
            ("http://LOCALHOST:11434", True),
            ("http://[::1]:11434", True),
            ("http://gpu-box.example.com:11434", False),
            ("http://localhost.example.com:11434", False),
            ("", False),
            (self.MISSING, False),
            ("http://[::1", False),
        ):
            with self.subTest(url=url):
                self.assertIs(from_layout1.on_this_host(url), expected)

    def test_an_ollama_bindings_requests_at_a_time(self):
        here, there = "http://127.0.0.1:11434", "http://gpu-box.example.com:11434"
        for name, url, value, expected in (
            ("worker", here, self.MISSING, 1),
            ("worker", here, "", 1),
            ("worker", here, None, 1),
            ("worker", "http://[::1]:11434", 0, 1),
            ("worker", here, self.P("0"), 1),
            ("worker", here, 4, 4),
            ("worker", here, False, False),
            ("worker", there, self.MISSING, 0),
            ("worker", there, 0, 0),
            ("worker", "", self.MISSING, 0),
            ("other", here, self.MISSING, 0),
            ("other", here, 0, 0),
            ("other", here, self.P("3"), self.P("3")),
        ):
            with self.subTest(name=name, url=url, value=value):
                self.assertEqual(
                    from_layout1.ollama_parallel_requests(name, url, value), expected
                )

    def test_why_a_key_is_dropped(self):
        for key, expected in (
            (
                "host.mounts",
                "the remote GPU tests' key is the secret gpu_ssh_key, installed with: rkmon secret gpu_ssh_key FILE",
            ),
            (
                "secrets",
                "no secret goes in a settings file; after rkmon confirm, rkmon login lists each secret and installs "
                "it",
            ),
            ("integrations.atlassian_cli.confluence_site", "no feature reads it"),
            ("setup", "a settings file keeps no record of the interview that wrote it"),
            ("source", "a settings file keeps no record of the export that wrote it"),
            (
                "advanced.publish.html.key",
                "the web host's SSH key is the secret web_host_ssh_key, installed with: rkmon secret web_host_ssh_key "
                "FILE",
            ),
            (
                "advanced.publish.html.known_hosts",
                "the monitor keeps the web host's own key in its state: rkmon login shows it, and records it once "
                "you accept it",
            ),
            (
                "advanced.publish.html.host_keys",
                "the monitor keeps the web host's own key in its state: rkmon login shows it, and records it once "
                "you accept it",
            ),
            (
                "advanced.publish.html.dest",
                "the page's place is worked out from features.dashboards.team.publish.site_folder",
            ),
            (
                "advanced.update.path",
                "the package knows its own folder in the repository it comes from",
            ),
            ("container.name", ""),
        ):
            with self.subTest(key=key):
                self.assertEqual(from_layout1.why_dropped(key), expected)

    def test_the_admin_dashboards_address(self):
        for value, expected in (
            ("", True),
            (self.MISSING, True),
            (self.P("127.0.0.1"), True),
            ("0.0.0.0", False),
            ("localhost", False),
        ):
            with self.subTest(value=value):
                self.assertIs(from_layout1.admin_address_allowed(value), expected)

    def test_a_github_repository_and_a_branch_are_the_updates_section(self):
        for value, expected in (
            ("https://github.com/ROCm/my-repo", "ROCm/my-repo"),
            (self.P("https://github.com/ROCm/my-repo.git"), "ROCm/my-repo"),
            ("http://GitHub.com/ROCm/my-repo/", "ROCm/my-repo"),
            (f"https://user{AT_GITHUB}/ROCm/my.repo", "ROCm/my.repo"),
            (f"git{AT_GITHUB}:ROCm/my-repo.git", "ROCm/my-repo"),
            (f"ssh://git{AT_GITHUB}/ROCm/my-repo", "ROCm/my-repo"),
            (self.P("ROCm/my-repo"), "ROCm/my-repo"),
            (" ROCm/my_repo ", "ROCm/my_repo"),
            ("", ""),
            (None, ""),
            (self.MISSING, ""),
            ("origin", ""),
            ("/srv/git/my-repo", ""),
            ("./ROCm/my-repo", ""),
            ("~/ROCm/my-repo", ""),
            ("ROCm/..", ""),
            ("https://gitlab.example.com/ROCm/my-repo", ""),
            ("https://github.com/ROCm/my-repo/tree/develop", ""),
            (["ROCm/my-repo"], ""),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.github_repository(value), expected)
        for value, expected in (
            (self.P("develop"), "develop"),
            (" release/7.0 ", "release/7.0"),
            (self.P("7"), "7"),
            ("", ""),
            (self.MISSING, ""),
            (["develop"], ""),
            ({"name": "develop"}, ""),
        ):
            with self.subTest(value=value):
                self.assertEqual(from_layout1.branch_name(value), expected)
        github = f"git{AT_GITHUB}:ROCm/my-repo.git"
        self.assertEqual(
            from_layout1.updates_section(github, self.P("develop")),
            {"repository": "ROCm/my-repo", "branch": "develop"},
        )
        for repo, branch in (
            (github, ""),
            (github, self.MISSING),
            ("", "develop"),
            (self.MISSING, "develop"),
            ("origin", "develop"),
            (self.MISSING, self.MISSING),
        ):
            with self.subTest(repo=repo, branch=branch):
                self.assertEqual(from_layout1.updates_section(repo, branch), {})


# A whole file of each kind in layout 1, invented: every rule in use, the deploy file's features on.
FULL_DEPLOY = """\
# rocKE monitor: deploy settings.
kind: deploy_settings
schema_version: 1
container:
  name: my-monitor
  image:
    name: my-image
    tag: ""
    pull_registry: registry.example.com/team
repository:
  name: my-org/my-repo
  path: projects/my-project
  scope_labels: [my-project]
  path_labels: [shared-a, shared-b]
host:
  resources:
    cpus: 6
    memory_gb: 12
    ramdisk_gb: 2
  users:
    monitor_uid: ""
    monitor_gid: "1001"
    github_writer_uid: 1999
    pr_runner_uid: 1998
  mounts: [/srv/keys/gpu_key]
monitor:
  tick_min: 5
  full_tick_min: 30
  orchestrator:
    model: opus
    effort: xhigh
features:
  review_automation:
    enabled: true
    mode: live
    schedule:
      push_cooldown_min: 20
      follow_up_cooldown_min: 45
    capacity:
      parallel_reviews: 4
      parallel_heavy_checks: 2
    large_prs:
      manual_review_over_lines: 20000
      split_part_lines: 6000
    passes:
      verify:
        model: opus
        effort: high
      quality:
        model: sonnet
        effort: ""
      first_pass:
        model: worker
        effort: default
      ci_triage:
        model: none
        effort: default
      quality_draft:
        model: default
        effort: default
      notice_draft:
        model: notices
        effort: low
  merge_queue:
    enabled: true
    mode: prototype
    auto_merge: enabled
    auto_hold: true
    develop_guard: true
  dashboards:
    admin:
      enabled: true
      port: 18614
      bind: 127.0.0.1
    team:
      enabled: true
      delivery: serve
      port: 18700
      bind: 0.0.0.0
  alerts:
    enabled: true
    teams_channel: reviews
    message_cooldown_min: 15
  ticket_creation:
    enabled: true
    jira_project: PROJ
    default_epic: ""
    triggers:
      review_follow_up: true
      out_of_scope_issue: false
integrations:
  github_cli:
    account: op-login
  atlassian_cli:
    jira_site: https://jira.example.com
    confluence_site: https://jira.example.com/wiki
  claude_code:
    gateway: https://llm.example.com/claude
model_bindings:
  worker:
    provider: openai-compatible
    endpoint: https://llm.example.com/v1
    model_id: example-model
    api_key_secret: worker_api_key
    limits:
      requests_per_min: 30
      tokens_per_min: 200000
      max_parallel: 4
  notices:
    provider: openai-compatible
    endpoint: https://llm.example.com/v1
    model_id: example-flash
    api_key_secret: notices_api_key
    limits:
      timeout_min: 10
  local:
    provider: ollama
    endpoint: http://localhost:11434
    model_id: "example:7b"
advanced:
  worker:
    auth_header: X-Api-Key
    auth_mode: both
    think: max
    context: 262144
  claude:
    effort: high
  automerge_guard_own: true
  gpu:
    enabled: true
    login_host: gpu-login.example.com
    ssh_user: op-login
    partition: gpu
    accounts: [acct-a]
    gres:
      gfx942: "gpu:gfx942:1"
    remote_python: /usr/bin/python3
    env:
      ROCKE_BACKEND: python
      EXTRA: "1"
  prices:
    claude-opus-5-5:
      in: 15
      out: 75
      cache_read: 1.5
      cache_write_5m: 18.75
      cache_write_1h: 30
  incident_issue:
    labels: [post-merge]
    stability_label: rocke-stability
  jira:
    component: "Tools, Platform"
    issue_type: Task
    extra_fields:
      customfield_1:
        value: x
  weekly_report:
    enabled: true
  publish:
    html:
      host: web.example.com
      user: op-login
      site: public_html/team
      url: https://dash.example.com/team
  team_dashboard:
    url: https://dash.example.com/team
  tick_interval_min: 10
  mirror:
    enabled: false
  offload:
    mode: shadow
secrets:
  claude: claude_env
setup:
  answered: true
exported_at: "2026-01-01T00:00:00Z"
package_version: "1.0.0"
source: rkmon settings export
"""

FULL_DEPLOY_VALUES = {
    "container": {"name": "my-monitor"},
    "image": {
        "name": "my-image",
        "registry": "registry.example.com/team",
        "tag": "my-monitor",
    },
    "repository": {
        "name": "my-org/my-repo",
        "folder": "projects/my-project",
        "scope_labels": ["my-project"],
        "shared_label_folders": {
            "shared-a": "projects/my-project",
            "shared-b": "projects/my-project",
        },
    },
    "host": {
        "resources": {"cpus": 6, "memory_gib": 12, "ramdisk_gib": 2},
        "users": {"monitor_uid": None, "monitor_gid": 1001},
    },
    "monitor": {
        "tick_interval_minutes": 5,
        "full_tick_max_interval_minutes": 30,
        "parallel_builds": 2,
    },
    "features": {
        "review_automation": {
            "enabled": True,
            "mode": "live",
            "schedule": {
                "push_cooldown_minutes": 20,
                "follow_up_cooldown_minutes": 45,
            },
            "parallel_reviews": 4,
            "large_prs": {
                "in_depth_review_over_lines": 20000,
                "split_part_lines": 6000,
            },
            "orchestrator": {"model": "opus", "effort": "xhigh"},
            "passes": {
                "verification_pass": {"model": "opus", "effort": "high"},
                "code_quality_pass": {"model": "sonnet", "effort": "high"},
                "first_review_pass": {"model": "worker", "effort": "default"},
                "ci_triage_pass": {"model": "default", "effort": "default"},
                "first_code_quality_pass": {"model": "default", "effort": "default"},
                "notice_draft_pass": {"model": "notices", "effort": "low"},
            },
            "remote_gpu_tests": {
                "enabled": True,
                "cluster": {
                    "login_host": "gpu-login.example.com",
                    "ssh_user": "op-login",
                    "partition": "gpu",
                    "accounts": ["acct-a"],
                    "gpu_types": {"gfx942": "gpu:gfx942:1"},
                },
                "node": {"python": "/usr/bin/python3", "environment": {"EXTRA": "1"}},
            },
            "spend": {
                "enabled": True,
                "usd_per_million_tokens": {
                    "claude-opus-5-5": {
                        "input": 15,
                        "output": 75,
                        "cache_read": 1.5,
                        "cache_write_5_minutes": 18.75,
                        "cache_write_1_hour": 30,
                    }
                },
            },
        },
        "merge_queue": {
            "enabled": True,
            "mode": "prototype",
            "auto_merge": {"enabled": True},
            "auto_hold": {"enabled": True},
            "develop_guard": {
                "enabled": True,
                "ci_red_label": "post-merge",
                "ci_red_extra_labels": ["ci-red"],
                "stability_label": "rocke-stability",
            },
            "queue_notices": {"enabled": True},
            "drift_notices": {"enabled": True},
            "auto_merge_guard": {"on_your_pull_requests": "same_as_others"},
        },
        "dashboards": {
            "admin": {"enabled": True, "port": 18614},
            "team": {
                "enabled": True,
                "delivery": "serve",
                "serve": {"port": 18700, "address": "0.0.0.0"},
                "mode": "live",
                "public_url": "https://dash.example.com/team",
            },
        },
        "alerts": {
            "develop_breaks": {"enabled": True},
            "needs_you": {"enabled": True},
            "message_label": "reviews",
            "message_cooldown_minutes": 15,
            "mode": "live",
        },
        "ticket_creation": {
            "follow_up_tickets": {"enabled": True},
            "found_issue_tickets": {"enabled": False},
            "project_key": "PROJ",
            "mode": "live",
            "issue_type": "Task",
            "required_fields": {"customfield_1": {"value": "x"}},
            "components": ["Tools", "Platform"],
        },
        "weekly_report": {"enabled": True},
    },
    "integrations": {
        "github_cli": {"username": "op-login"},
        "atlassian_cli": {"jira_url": "https://jira.example.com"},
        "claude_code": {"base_url": "https://llm.example.com/claude"},
    },
    "model_bindings": {
        "worker": {
            "provider": "openai_compatible",
            "model_id": "example-model",
            "base_url": "https://llm.example.com/v1",
            "limits": {
                "requests_per_minute": 30,
                "tokens_per_minute": 200000,
                "max_parallel_requests": 4,
            },
            "api_key_secret_name": "first_review_pass_api_key",
            "stall_timeout_seconds": 900,
            "api_key_sent_as": "header_and_bearer",
            "api_key_header": "X-Api-Key",
            "api_key_scheme": "Bearer",
        },
        "notices": {
            "provider": "openai_compatible",
            "model_id": "example-flash",
            "base_url": "https://llm.example.com/v1",
            "api_key_secret_name": "notices_api_key",
            "stall_timeout_seconds": 600,
        },
        "local": {
            "provider": "ollama",
            "model_id": "example:7b",
            "server_url": "http://localhost:11434",
            "limits": {"max_parallel_requests": 0},
        },
    },
}


def setUpModule():
    # The full deploy file's repository stands for the home repository: its repository defaults apply.
    repository = FULL_DEPLOY_VALUES["repository"]
    home_repository.stand_in(repository["name"], repository["folder"])


FULL_DEPLOY_NOTES = [
    "container.image.name is now image.name",
    "container.image.pull_registry is now image.registry",
    'image.tag: "my-monitor", the container\'s name, what a blank tag meant; rkmon confirm says whether that image '
    "suits this package",
    "repository.path is now repository.folder",
    "repository.path_labels is now repository.shared_label_folders: each label with repository.folder "
    '("projects/my-project") as its folder, the one folder it counted for',
    "host.resources.memory_gb is now host.resources.memory_gib",
    "host.resources.ramdisk_gb is now host.resources.ramdisk_gib",
    "host.users.github_writer_uid 1999: dropped, as that user's id is the package's (1999)",
    "host.users.pr_runner_uid 1998: dropped, as that user's id is the package's (1998)",
    "host.mounts (1 entry): dropped, as the remote GPU tests' key is the secret gpu_ssh_key, installed with: rkmon "
    "secret gpu_ssh_key FILE",
    "monitor.tick_min is now monitor.tick_interval_minutes",
    "monitor.full_tick_min is now monitor.full_tick_max_interval_minutes",
    f"{RA}.schedule.push_cooldown_min is now {RA}.schedule.push_cooldown_minutes",
    f"{RA}.schedule.follow_up_cooldown_min is now {RA}.schedule.follow_up_cooldown_minutes",
    f"{RA}.capacity.parallel_reviews is now {RA}.parallel_reviews",
    f"{RA}.capacity.parallel_heavy_checks is now monitor.parallel_builds",
    f"{RA}.large_prs.manual_review_over_lines is now {RA}.large_prs.in_depth_review_over_lines",
    f"monitor.orchestrator is now {RA}.orchestrator",
    f"{PASSES}.verify is now {PASSES}.verification_pass",
    f"{PASSES}.quality is now {PASSES}.code_quality_pass",
    f"{PASSES}.first_pass is now {PASSES}.first_review_pass",
    f'{PASSES}.first_review_pass.model: "worker", the file\'s binding of that name, an ordinary binding now, through '
    "which the pass drafts in steps; the local worker that the name chose leaves the package",
    f"{PASSES}.ci_triage is now {PASSES}.ci_triage_pass",
    f'{PASSES}.ci_triage.model "none" is now "default": the pass in charge does the work',
    f"{PASSES}.quality_draft is now {PASSES}.first_code_quality_pass",
    f"{PASSES}.notice_draft is now {PASSES}.notice_draft_pass",
    f'{MQ}.auto_merge "enabled" is now {MQ}.auto_merge.enabled true',
    f"{MQ}.auto_hold is now {MQ}.auto_hold.enabled",
    f"{MQ}.develop_guard is now {GUARD}.enabled",
    *LABEL_NOTES,
    f'advanced.automerge_guard_own true is now {MQ}.auto_merge_guard.on_your_pull_requests "same_as_others"',
    f'{ADMIN}.bind "127.0.0.1": dropped, as the admin dashboard listens on this host\'s 127.0.0.1 only',
    f"{TEAM}.port is now {TEAM}.serve.port",
    f"{TEAM}.bind is now {TEAM}.serve.address",
    f'{TEAM}.mode: "live", as the team dashboard is on in the file, and layout 1 sent whenever it was on',
    f"{ALERTS}.enabled is now {ALERTS}.develop_breaks.enabled and {ALERTS}.needs_you.enabled, each true",
    f"{ALERTS}.teams_channel is now {ALERTS}.message_label",
    f"{ALERTS}.message_cooldown_min is now {ALERTS}.message_cooldown_minutes",
    f'{ALERTS}.mode: "live", as alerts are on in the file, and layout 1 sent whenever it was on',
    f"{TC}.enabled and its triggers are now {TC}.follow_up_tickets.enabled true and {TC}.found_issue_tickets.enabled "
    "false: each the section's switch and the type's trigger",
    f"{TC}.jira_project is now {TC}.project_key",
    f'{TC}.mode: "live", as ticket creation is on in the file, and layout 1 sent whenever it was on',
    "integrations.github_cli.account is now integrations.github_cli.username",
    "integrations.atlassian_cli.jira_site is now integrations.atlassian_cli.jira_url",
    'integrations.atlassian_cli.confluence_site "https://jira.example.com/wiki": dropped, as no feature reads it',
    "integrations.claude_code.gateway is now integrations.claude_code.base_url",
    "integrations.claude_code.api_key_sent_as, .api_key_header and .api_key_scheme take their defaults, the key in "
    "the x-api-key header: set them as your gateway takes the key",
    'model_bindings.worker.provider "openai-compatible" is now "openai_compatible"',
    "model_bindings.worker.endpoint is now model_bindings.worker.base_url",
    "model_bindings.worker.limits.requests_per_min is now model_bindings.worker.limits.requests_per_minute",
    "model_bindings.worker.limits.tokens_per_min is now model_bindings.worker.limits.tokens_per_minute",
    "model_bindings.worker.limits.max_parallel is now model_bindings.worker.limits.max_parallel_requests",
    'model_bindings.worker.api_key_secret "worker_api_key" is now model_bindings.worker.api_key_secret_name '
    '"first_review_pass_api_key", the name the one-time move copies that secret to',
    "model_bindings.worker.stall_timeout_seconds: 900, what the binding named worker ran on with no timeout",
    'model_bindings.notices.provider "openai-compatible" is now "openai_compatible"',
    "model_bindings.notices.endpoint is now model_bindings.notices.base_url",
    "model_bindings.notices.api_key_secret is now model_bindings.notices.api_key_secret_name",
    "model_bindings.notices.limits.timeout_min 10 is now model_bindings.notices.stall_timeout_seconds 600",
    "model_bindings.local.endpoint is now model_bindings.local.server_url",
    "model_bindings.local.limits.max_parallel_requests: 0, what the binding ran on with no limits.max_parallel: no "
    "cap",
    "advanced.worker.auth_header and advanced.worker.auth_mode are now model_bindings.worker.api_key_sent_as "
    '"header_and_bearer", model_bindings.worker.api_key_header "X-Api-Key" and model_bindings.worker.api_key_scheme '
    '"Bearer"',
    f"advanced.gpu.enabled is now {RGT}.enabled",
    f"advanced.gpu.login_host is now {RGT}.cluster.login_host",
    f"advanced.gpu.ssh_user is now {RGT}.cluster.ssh_user",
    f"advanced.gpu.partition is now {RGT}.cluster.partition",
    f"advanced.gpu.accounts is now {RGT}.cluster.accounts",
    f"advanced.gpu.gres is now {RGT}.cluster.gpu_types",
    f"advanced.gpu.remote_python is now {RGT}.node.python",
    "advanced.weekly_report.enabled is now features.weekly_report.enabled",
    f"advanced.jira.issue_type is now {TC}.issue_type",
    f"advanced.jira.extra_fields is now {TC}.required_fields",
    'advanced.publish.html.host "web.example.com": dropped, as the team dashboard\'s delivery is "serve", and only '
    '"publish" copies the page to a web host; the value stays in the earlier file, which migrate keeps',
    'advanced.publish.html.user "op-login": dropped, as the team dashboard\'s delivery is "serve", and only "publish" '
    "copies the page to a web host; the value stays in the earlier file, which migrate keeps",
    'advanced.publish.html.site "public_html/team": dropped, as the team dashboard\'s delivery is "serve", and only '
    '"publish" copies the page to a web host; the value stays in the earlier file, which migrate keeps',
    f"advanced.gpu.env is now {RGT}.node.environment, without ROCKE_BACKEND=python, which the repository's defaults "
    "set first",
    f"advanced.prices is now {RA}.spend.usd_per_million_tokens, by model id, with in, out, cache_write_5m and "
    "cache_write_1h named input, output, cache_write_5_minutes and cache_write_1_hour; "
    f"{RA}.spend.enabled true",
    f"advanced.publish.html.url is now {TEAM}.public_url",
    f"advanced.team_dashboard.url: the same as advanced.publish.html.url, the one {TEAM}.public_url",
    f'advanced.jira.component is now {TC}.components, split at commas: ["Tools", "Platform"]',
    'advanced.offload.mode is "shadow" (the package\'s default is "off"): not carried, as layout 2 has no setting for '
    "it",
    "secrets (1 entry): dropped, as no secret goes in a settings file; after rkmon confirm, rkmon login lists each "
    "secret and installs it",
    "setup: dropped, as a settings file keeps no record of the interview that wrote it",
    "exported_at, package_version and source: dropped, as a settings file keeps no record of the export that "
    "wrote it",
    "advanced: 7 values the same as the package's defaults: dropped",
]

FULL_BUILD = f"""\
# rocKE monitor: build settings.
kind: build_settings
schema_version: 1
image:
  name: my-image
  tag: "1.2.3"
  push_registry: registry.example.com/team
  base: "ubuntu:24.04"
  ca_certificates: [/etc/ssl/extra.pem]
os_packages:
  package_mirror: http://mirror.example.com/ubuntu
rocm:
  version: "10.1.0"
  clang_commit: any
  gfx_targets: [gfx90a, gfx942]
github_cli:
  version: "2.60.0"
atlassian_cli:
  version: "1.0.0"
  package_url: https://example.com/acli
  package_sha256: {SHA}
claude_code:
  version: "0.9.0"
  package_index: https://pypi.example.com/simple
python_packages:
  package_index: https://pypi.example.com/simple/
exported_at: "2026-01-01T00:00:00Z"
"""


class TestFiles(unittest.TestCase):
    """Whole files of layout 1, read as migrate reads them, upgraded, written by the one writer and read again."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = pathlib.Path(tmp.name)

    def migrated(self, text: str, file_type=DEPLOY) -> tuple:
        """(the upgraded document, its notes, the text written, the written file read again)."""
        path = self.folder / "settings.yaml"
        path.write_text(text, encoding="utf-8")
        new, notes = settings_file.upgrade(
            settings_file.read(path, file_type, older=True)
        )
        written = settings_file.dump(new)
        path.write_text(written, encoding="utf-8")
        return new, notes, written, settings_file.read(path, file_type)

    def assertNothingLost(self, values: dict, back, file_type):
        """Each value of VALUES is in BACK, the file read again, or, left out of it, is the value the template
        holds for its key, which the key means when it is left out (as an off feature's keys are).
        """
        base = settings_file._template_values(file_type.load())

        def leaves(v, prefix=""):
            for k, x in v.items():
                if isinstance(x, dict) and x:
                    yield from leaves(x, f"{prefix}{k}.")
                else:
                    yield f"{prefix}{k}", x

        lost = [
            (k, v)
            for k, v in leaves(values)
            if at(back.values, k) != v
            and not (at(back.values, k) is GONE and at(base, k) == v)
        ]
        self.assertEqual(lost, [])

    def test_a_deploy_file_with_its_features_on(self):
        new, notes, written, back = self.migrated(FULL_DEPLOY)
        self.assertEqual(new.values, FULL_DEPLOY_VALUES)
        self.assertEqual(notes, FULL_DEPLOY_NOTES)
        # Check's rows on the file written: every key fits the schema, then the sample's own two warnings, the models
        # the passes use with no price, and a model binding on the loopback.
        self.assertEqual(
            [(r.name, r.status) for r in settings_file.validate(back)],
            [
                ("settings", Status.PASS),
                ("spend", Status.WARN),
                ("model_bindings.local.server_url", Status.WARN),
            ],
        )
        self.assertNothingLost(new.values, back, DEPLOY)
        self.assertEqual(back.version, 2)

    def test_a_feature_that_is_off_is_its_switch_alone_unless_it_holds_a_value_of_its_own(
        self,
    ):
        new, notes, written, back = self.migrated(
            "kind: deploy_settings\nschema_version: 1\ncontainer:\n  name: my-monitor\n  image:\n"
            '    tag: "1.0.0"\nrepository:\n  name: my-org/my-repo\n  path: projects/my-project\n'
            "  scope_labels: [my-project]\nfeatures:\n  review_automation:\n    enabled: false\n    mode: live\n"
            "  merge_queue:\n    enabled: false\nintegrations:\n  github_cli:\n    account: op-login\n"
        )
        lines = [
            line for line in written.splitlines() if not line.lstrip().startswith("#")
        ]

        def section(key: str) -> list:
            """The lines under the feature KEY, without their indent."""
            start = lines.index(f"  {key}:")
            under = []
            for line in lines[start + 1 :]:
                if len(line) - len(line.lstrip()) <= 2:
                    break
                under.append(line.strip())
            return under

        self.assertEqual(section("merge_queue"), ["enabled: false"])
        automation = section("review_automation")
        self.assertEqual(automation[:2], ["enabled: false", 'mode: "live"'])
        self.assertIn("orchestrator:", automation)
        self.assertNothingLost(new.values, back, DEPLOY)

    def test_an_empty_deploy_file(self):
        new, notes, written, back = self.migrated(
            "kind: deploy_settings\nschema_version: 1\n"
        )
        self.assertEqual(
            notes,
            [
                'container.name: "rkmon", what a blank name meant: the deployment keeps its name, so its volumes and '
                "its state",
                'image.tag: "rkmon", the container\'s name, what a blank tag meant; rkmon confirm says whether that '
                "image suits this package",
                *LABEL_NOTES,
            ],
        )
        self.assertEqual(
            (at(back.values, "container.name"), at(back.values, "image.tag")),
            ("rkmon", "rkmon"),
        )
        self.assertNothingLost(new.values, back, DEPLOY)

    def test_a_build_file(self):
        new, notes, written, back = self.migrated(FULL_BUILD, BUILD)
        self.assertEqual(
            new.values,
            {
                "image": {
                    "name": "my-image",
                    "tag": "1.2.3",
                    "registry": "registry.example.com/team",
                },
                "operating_system": {
                    "base_image": "ubuntu:24.04",
                    "ca_certificates": ["/etc/ssl/extra.pem"],
                    "package_mirror": "http://mirror.example.com/ubuntu",
                },
                "rocm": {"version": "10.1.0"},
                "github_cli": {"version": "2.60.0"},
                "atlassian_cli": {
                    "version": "1.0.0",
                    "package_url": "https://example.com/acli",
                    "package_sha256": SHA,
                },
                "claude_agent_sdk": {"version": "0.9.0"},
                "python_packages": {
                    "package_index": "https://pypi.example.com/simple/"
                },
            },
        )
        self.assertEqual(
            notes,
            [
                "image.push_registry is now image.registry",
                "image.base is now operating_system.base_image",
                "image.ca_certificates is now operating_system.ca_certificates",
                "os_packages.package_mirror is now operating_system.package_mirror",
                'rocm.clang_commit "any": dropped, as the package pins the clang commit itself '
                f'("{build_plan.ROCM_CLANG_COMMIT}" for ROCm {build_plan.ROCM_RELEASE})',
                'rocm.gfx_targets ["gfx90a", "gfx942"]: dropped, as the package sets its GPU targets itself '
                f'({",".join(build_plan.ROCM_LIBRARY_TARGETS)})',
                "claude_code.version is now claude_agent_sdk.version",
                "claude_agent_sdk.package_url and .package_sha256 start blank: the package index serves it",
                "claude_code.package_index: the same index as python_packages.package_index, the one for every "
                "Python package",
                "exported_at: dropped, as a settings file keeps no record of the export that wrote it",
            ],
        )
        # The SDK's package_sha256 starts blank, and the sample sets none for ROCm's installer: check warns
        # that neither download is checked.
        self.assertEqual(
            [(r.name, r.status) for r in settings_file.validate(back)],
            [
                ("settings", Status.PASS),
                ("rocm.package_sha256", Status.WARN),
                ("claude_agent_sdk.package_sha256", Status.WARN),
            ],
        )
        self.assertNothingLost(new.values, back, BUILD)

    def test_the_refusals_as_migrate_meets_them(self):
        # A secret_values section: read refuses it first.
        path = self.folder / "secrets.yaml"
        path.write_text(
            "kind: deploy_settings\nschema_version: 1\nsecret_values:\n  a: x\n  b: y\n  c: z\n",
            encoding="utf-8",
        )
        where = settings_file.shown(path)
        with self.assertRaises(Refused) as cm:
            settings_file.read(path, DEPLOY, older=True)
        self.assertEqual(
            [(r.found, str(r.fix)) for r in cm.exception.rows],
            [
                (
                    f"{where} has a secret_values section (3 entries), and no secret goes in a settings file",
                    f"move each value into a file only you can read, delete secret_values from {where}, and run "
                    "migrate again; after rkmon confirm, rkmon login lists each secret and installs it",
                )
            ],
        )
        # An admin dashboard's address: read takes the file, and upgrade refuses it, naming the file.
        path = self.folder / "admin.yaml"
        path.write_text(
            "kind: deploy_settings\nschema_version: 1\nfeatures:\n  dashboards:\n    admin:\n"
            "      bind: 0.0.0.0\n",
            encoding="utf-8",
        )
        where = settings_file.shown(path)
        with self.assertRaises(Refused) as cm:
            settings_file.upgrade(settings_file.read(path, DEPLOY, older=True))
        self.assertEqual(cm.exception.reason, "step")
        self.assertEqual(
            [(r.found, str(r.fix)) for r in cm.exception.rows],
            [
                (
                    f'{ADMIN}.bind is "0.0.0.0", and the admin dashboard listens on this host\'s 127.0.0.1 only',
                    f'delete {ADMIN}.bind from {where}; README.md, "The admin dashboard", says how to reach it from '
                    "another machine",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
