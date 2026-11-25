# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deploy path's confirm step (rkmon_lib/confirm.py, rkmon_lib/deployment.py): the summary of the configuration
the container runs with the settings, section by section in the file's order; each pass's model and effort resolved (a
default pass follows its owner); each coherence rule failing and passing; each model in use tried; check 16, an older
package's Claude settings against integrations.claude_code; and the configuration the container runs, the package's
defaults, the file and the container's layout. docker/rkmon's own
part (the question, the record in this folder's records, the start's and the apply's refusal) is
tests/test_deploy_path_confirm.py's; the fingerprint the confirm records is tests/test_deploy_plan.py's.
"""

import copy
import dataclasses
import http.server
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import NamedTuple
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import home_repository  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import (
    change_plan,
    claude_env_shape,
    code_volume,
    coherence,
    config,
    confirm,
    container_layout,
    deliver,
    deploy_plan,
)  # noqa: E402
from rkmon_lib import deploy_tree, file_types, image  # noqa: E402
from rkmon_lib import deployment as d  # noqa: E402
from rkmon_lib import (
    program,
    registry,
    releases,
    runtime_config,
    secrets_registry,
    settings_file,
    steps,
)  # noqa: E402
from rkmon_lib.checks import WIDTH, Check, Status, render  # noqa: E402
from rkmon_lib.container_layout import (  # noqa: E402
    LOOPBACK,
    SECRETS_DIR,
    WRITE_SECRETS_DIR,
)
from rkmon_lib.from_layout1 import UPDATES  # noqa: E402

GOOD = {
    "org.rocke.package.version": image.package_version(),
    "org.rocke.build.date": "2026-11-13T00:00:00Z",
    "org.rocke.base": "ubuntu:26.04",
    "org.rocke.rocm.version": "10.1.0-5",
}
# A deploy settings file with the keys it requires, each feature as it starts, and the monitor's ids.
BASE = {
    "kind": file_types.DEPLOY.kind,
    "schema_version": file_types.DEPLOY.version,
    "image": {"tag": image.package_version()},
    "repository": {
        "name": "acme/widgets",
        "folder": "widgets/rocke",
        "scope_labels": ["widgets"],
    },
    "host": {"users": {"monitor_uid": 4242, "monitor_gid": 4343}},
    "features": {
        "review_automation": {
            "enabled": True,
            "mode": "prototype",
            "remote_gpu_tests": {"enabled": False},
            "spend": {"enabled": False},
        },
        "merge_queue": {
            "enabled": True,
            "mode": "prototype",
            "auto_merge": {"enabled": False},
            "auto_hold": {"enabled": True},
            "queue_notices": {"enabled": True},
            "drift_notices": {"enabled": True},
            "develop_guard": {"enabled": True},
        },
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
    "integrations": {"github_cli": {"username": "op-login"}},
}
GLM = {
    "provider": "openai_compatible",
    "base_url": "https://models.example.com/v1",
    "model_id": "GLM-5.3-Flash",
    "api_key_secret_name": "glm_api_key",
}


def setUpModule():
    # BASE's repository and folder stand for the package's home repository, whose defaults the container runs with
    home_repository.stand_in("acme/widgets", "widgets/rocke")


def merged(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = (
            merged(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else copy.deepcopy(v)
        )
    return out


class Read(NamedTuple):
    """A deploy settings file as the confirm reads it: its document, the configuration it makes, and its rows and its
    settings' (deployment.read's FAIL and WARN rows, deployment.settings_checks', then coherence.resources', here with
    no figure of this host), as docker/rkmon's confirm_checks gives them; then what it makes on this host
    (deploy_plan.plan, for a user and a group 1000) and the configuration the container runs with the settings the
    plan delivers (runtime_config.from_settings).
    """

    doc: settings_file.Document
    cfg: dict
    rows: list
    plan: deploy_plan.DeployPlan
    run: runtime_config.Resolved


def read(**parts) -> Read:
    """BASE with `parts` over it, as the confirm reads it."""
    with tempfile.TemporaryDirectory() as t:
        path = pathlib.Path(t) / "deploy.yaml"
        path.write_text(json.dumps(merged(BASE, parts)), encoding="utf-8")
        cfg, rows = d.read(path)
        doc = settings_file.read(path, file_types.DEPLOY)
    host = deploy_plan.Host(1000, 1000, "")
    made = deploy_plan.plan(doc, image.package_version(), image.code_sha256(), host)
    run = runtime_config.from_settings(made.layer["settings"])
    rows += d.settings_checks(cfg, str(path)) + coherence.resources(doc)
    return Read(doc, cfg, rows, made, run)


def summary(labels=None, logins=None, **parts) -> str:
    """The confirm's summary of BASE with `parts` over it, as docker/rkmon's show_summary renders it: the updates
    section from the file itself."""
    r = read(**parts)
    return confirm.render(
        r.plan, r.run.values, labels, logins, r.doc.values.get(UPDATES)
    )


def section(text: str, title: str) -> list:
    """The lines of the summary TEXT's section TITLE: its heading, or its one line, then its own indented lines."""
    lines = text.splitlines()
    at = next(
        i for i, ln in enumerate(lines) if ln == title or ln.startswith(f"{title}: ")
    )
    out = [lines[at]]
    for ln in lines[at + 1 :]:
        if not ln.startswith(" "):
            break
        out.append(ln)
    return out


def checks(image=(), models=(), **parts) -> list:
    r = read(**parts)
    return confirm.checks(r.run, r.doc, r.rows, image, models)


def found(got: list, name: str) -> list:
    return [(c.status, c.found) for c in got if c.name == name]


def passes(**over) -> dict:
    return {"features": {"review_automation": {"passes": over}}}


class TestSummary(unittest.TestCase):
    def test_the_headline_and_the_sections_in_the_file_s_order(self):
        text = summary()
        self.assertEqual(
            text.split("\n\n", 1)[0],
            "container rocke-monitor · review automation prototype · merge queue prototype"
            " · auto_merge hints_only\n"
            "· admin dashboard on · team dashboard off · alerts off · ticket creation off",
        )
        sch = file_types.DEPLOY.load()["properties"]
        want = []
        for part in deploy_tree.parts():
            if part == "features":
                want += [f["title"] for f in sch["features"]["properties"].values()]
            else:
                want.append(sch[part]["title"])
        # Each part of the file has a section of its own, and so has each feature; the updates are one line while
        # the file has none.
        r = read()
        heads = [
            ln.split(": ", 1)[0]
            for ln in confirm.sections(r.plan, r.run.values)
            if ln and not ln.startswith(" ")
        ]
        self.assertEqual(heads, want + ["Logins"])

    def test_the_headline_names_the_container_then_each_feature(self):
        """The headline: the container's name, then each feature's state, joined by " · "; a line that would pass
        WIDTH goes on at its last "·", which starts the next line."""
        values = read(
            container={"name": "rocke-monitor-2"},
            features={"merge_queue": {"enabled": False}},
        ).run.values
        self.assertEqual(
            confirm.headline(values).splitlines(),
            [
                "container rocke-monitor-2 · review automation prototype · merge queue off · admin dashboard on · "
                "team dashboard off",
                "· alerts off · ticket creation off",
            ],
        )

    def test_each_key_under_its_own_section(self):
        """The registry, the image's name and its tag are the Image section's, a blank registry in plain words and
        the tag with whether its image is on this host; the parallel builds are the Monitor's and the orchestrator
        Review automation's, as the file has them; the updates show the file's section, or one line while it has
        none; the weekly report's switch says what it writes.
        """
        text = summary()
        self.assertRegex(
            "\n".join(section(text, "Image")),
            r"\AImage\n  registry: +none \(the image comes from a file\)\n  name: +rocke-monitor\n"
            r"  tag: +\S+ \(not on this host yet: step 4 brings it\)\Z",
        )
        monitor = "\n".join(section(text, "Monitor"))
        automation = "\n".join(section(text, "Review automation"))
        self.assertRegex(monitor, r"(?m)^  parallel_builds: +1 \(at the start")
        self.assertRegex(
            automation,
            r"(?m)^  orchestrator\.model: +opus\n  orchestrator\.effort: +high$",
        )
        self.assertNotRegex(monitor, r"(?m)^  orchestrator")
        self.assertNotRegex(automation, r"(?m)^  parallel_builds:")
        self.assertEqual(
            section(text, "Updates"),
            [
                "Updates: none (package updates only from a folder or an archive: "
                f"{steps.command_line(settings_file.PACKAGE_UPDATE)})"
            ],
        )
        self.assertEqual(
            section(text, "Weekly report"), ["Weekly report", "  enabled: false"]
        )
        text = summary(
            image={"registry": "registry.example.com/team"},
            updates={"repository": "acme/rocke", "branch": "develop"},
            features={"weekly_report": {"enabled": True}},
        )
        self.assertRegex(
            "\n".join(section(text, "Image")),
            r"(?m)^  registry: +registry\.example\.com/team$",
        )
        self.assertRegex(
            "\n".join(section(text, "Updates")),
            r"\AUpdates\n  repository: +acme/rocke\n  branch: +develop\Z",
        )
        self.assertEqual(
            section(text, "Weekly report"),
            [
                "Weekly report",
                "  enabled: true (every week, a statistics report and notes for a lessons-learned review)",
            ],
        )

    def test_every_setting_under_its_own_section_in_the_file_s_order(self):
        """The first time, confirm shows every setting, as its help says: each key of the file
        under its own section, both in the file's order, each named from its section, at the value the confirm delivers,
        private values included. Those are the delivered layer's keys: every key of the file but its header (kind,
        schema_version) and the updates, which only the host reads (Updates shows the file's own section:
        test_each_key_under_its_own_section); a map is one value. Each value shows as rk config show shows it
        (config.safe_value: test_each_value_shows_as_config_show_shows_it), a text as it is, any other value as JSON,
        whole; plain words follow a value, or stand for a blank one; a long value goes on under itself.
        """
        r = read(
            model_bindings={"glm-flash": GLM},
            repository={"shared_label_folders": {"rocke:platform": "projects/rocke"}},
            **passes(ci_triage_pass={"model": "glm-flash"}),
        )
        settings = r.plan.layer["settings"]
        sch = file_types.DEPLOY.load()["properties"]
        want = []
        for key in registry.leaves(
            settings, settings_file.map_keys(file_types.DEPLOY.load())
        ):
            part = (
                key.split(".")[:2]
                if key.startswith("features.")
                else key.split(".")[:1]
            )
            node = (
                sch[part[0]]
                if len(part) == 1
                else sch["features"]["properties"][part[1]]
            )
            name = key[len(".".join(part)) + 1 :] or key
            value = config.safe_value(registry.get(settings, key))
            shown = (
                value
                if isinstance(value, str) and value
                else json.dumps(value, ensure_ascii=False)
            )
            want.append((node["title"], name, shown))
        got, heading = [], None
        for line in (
            confirm.render(r.plan, r.run.values).split("\n\n", 1)[1].splitlines()
        ):
            if not line.startswith(" "):
                heading = line
            elif heading not in ("Updates", "Logins") and re.match(r"  [^ :]+: ", line):
                name, value = line[2:].split(": ", 1)
                got.append([heading, name, value.strip()])
            elif line.startswith("   ") and got and got[-1][0] == heading:
                got[-1][2] += " " + line.strip()  # a value that goes on under itself
        self.assertEqual([(h, n) for h, n, _v in got], [(t, n) for t, n, _s in want])
        for (_h, name, value), (_t, _n, shown) in zip(got, want):
            blank = shown == '""' and value.startswith("none (")
            self.assertTrue(
                value == shown or value.startswith(f"{shown} (") or blank,
                (name, value),
            )
        self.assertIn(["Integrations", "github_cli.username", "op-login"], got)
        self.assertIn(["Repository", "name", "acme/widgets"], got)
        self.assertIn(["Host", "users.monitor_uid", "4242"], got)
        self.assertIn(
            [
                "Repository",
                "shared_label_folders",
                '{"rocke:platform": "projects/rocke"}',
            ],
            got,
        )

    def test_each_value_shows_as_config_show_shows_it(self):
        """Each value shows as rk config show shows it (config.safe_value): an address without its query and
        fragment, in a map's value too (the file's read refuses a user, a password or a credential's name in
        one)."""
        r = read(
            model_bindings={
                "glm-flash": dict(
                    GLM, base_url="https://models.example.com/v1?deployment=d1"
                )
            },
            integrations={
                "claude_code": {
                    "base_url": "https://gateway.example.com/claude?team=reviews#top"
                }
            },
            **passes(ci_triage_pass={"model": "glm-flash"}),
        )
        text = confirm.render(r.plan, r.run.values)
        self.assertRegex(
            text,
            r"(?m)^  claude_code\.base_url: +https://gateway\.example\.com/claude$",
        )
        self.assertIn(
            '"base_url": "https://models.example.com/v1"', " ".join(text.split())
        )
        for query in ("deployment=d1", "team=reviews", "#top"):
            self.assertNotIn(query, text)

    def test_a_feature_that_is_off_shows_its_keys_and_why(self):
        """A feature that is off shows each of its keys too, its switch with why it does
        nothing."""
        text = summary(
            features={
                "merge_queue": {"enabled": False},
                "dashboards": {"admin": {"enabled": False}},
            }
        )
        merge = section(text, "Merge queue")
        self.assertRegex(
            merge[1], r"^  enabled: +false \(no queue, gates, plan or develop guard\)$"
        )
        self.assertRegex("\n".join(merge), r"(?m)^  develop_guard\.enabled: +true$")
        self.assertRegex(
            "\n".join(section(text, "Dashboards")),
            r"(?m)^  admin\.enabled: +false \(no web server\)$",
        )
        self.assertIn("merge queue off", text.splitlines()[0])

    def test_the_passes_resolved(self):
        """Each pass's model, provider and effort; a default pass says whose it takes, and follows its owner."""
        text = summary()
        for row in (
            r"(?m)^  pass +model +provider +effort$",
            r"(?m)^  verification_pass +opus +claude +high$",
            r"(?m)^  code_quality_pass +opus +claude +medium$",
            r"(?m)^  ci_triage_pass +default → opus +claude +high \(from verification_pass\)$",
            r"(?m)^  first_code_quality_pass +default → opus +claude +medium \(from code_quality_pass\)$",
            r"(?m)^  notice_draft_pass +default → opus +claude +high \(from orchestrator\)$",
        ):
            self.assertRegex(text, row)
        moved = summary(
            **passes(verification_pass={"model": "sonnet", "effort": "xhigh"})
        )
        self.assertRegex(
            moved,
            r"(?m)^  ci_triage_pass +default → sonnet +claude +xhigh \(from verification_pass\)$",
        )
        self.assertRegex(
            moved,
            r"(?m)^  first_review_pass +default → sonnet +claude +xhigh \(from verification_pass\)$",
        )
        bound = summary(
            model_bindings={"glm-flash": GLM},
            **passes(ci_triage_pass={"model": "glm-flash", "effort": "default"}),
        )
        self.assertRegex(
            bound,
            r"(?m)^  ci_triage_pass +glm-flash → GLM-5.3-Flash +openai-compatible +-$",
        )
        # the binding itself is the Model bindings section's, as the file has it
        bindings = " ".join(" ".join(section(bound, "Model bindings")).split())
        self.assertIn('"base_url": "https://models.example.com/v1"', bindings)

    def test_the_logins_each_feature_needs(self):
        text = summary()
        self.assertIn("  claude:             needed by Claude session", text)
        self.assertRegex(
            text, r"(?m)^  claude_code\.base_url: +none \(Anthropic's own API\)$"
        )  # the claude login is the key alone: no address
        self.assertIn("  github_write:       NOT NEEDED", text)  # nothing is live
        self.assertIn("  admin_access_token: needed by admin dashboard", text)
        live = summary(features={"review_automation": {"mode": "live"}})
        self.assertIn("  github_write:       needed by review posting", live)
        rules = summary(features={"review_automation": {"enabled": False}})
        self.assertIn("  claude:             NOT NEEDED", rules)

    def test_the_image_and_the_logins_once_they_can_be_read(self):
        text = summary()
        self.assertIn("(not on this host yet: step 4 brings it)", text)
        rows = [
            {"name": "github_read", "state": "SET"},
            {"name": "github_write", "state": "NOT NEEDED"},
        ]
        text = summary(GOOD, rows)
        self.assertIn(f"package {image.package_version()}", text)
        self.assertIn("ubuntu:26.04, ROCm 10.1.0-5", text)
        self.assertIn("  github_read:        set", text)
        self.assertIn("  github_write:       NOT NEEDED", text)

    def test_the_host_in_gib_and_the_parallel_builds(self):
        """The summary's units and names are the deploy file's: GiB, the parallel builds; no row for a key the file
        does not have (the mounts, a Confluence site of its own)."""
        text = summary(host={"resources": {"memory_gib": 12, "ramdisk_gib": 3}})
        self.assertRegex(text, r"(?m)^  resources\.memory_gib: +12$")
        self.assertRegex(text, r"(?m)^  resources\.ramdisk_gib: +3 \(at /")
        self.assertRegex(text, r"(?m)^  parallel_builds: +1 \(at the start, within 1-")
        self.assertNotIn("GB", text.replace("GiB", ""))
        self.assertNotRegex(text, r"(?m)^  (mounts|Confluence site):")
        self.assertRegex(text, r"(?m)^  atlassian_cli\.jira_url: +null$")

    def test_the_merge_queue_s_notes_by_its_mode(self):
        """In prototype the merge queue arms and files nothing (what it would send is kept as an unsent item); live,
        it arms, and its develop guard files issues and incidents."""
        on = {
            "merge_queue": {
                "auto_merge": {"enabled": True},
                "develop_guard": {"enabled": True},
            }
        }
        proto = summary(features=on)
        self.assertRegex(
            proto,
            r"auto_merge\.enabled: +true \(arms nothing until the merge queue is live\)",
        )
        self.assertNotIn("files issues and incidents", proto)
        live = summary(features=merged(on, {"merge_queue": {"mode": "live"}}))
        self.assertRegex(
            live, r"develop_guard\.enabled: +true \(files issues and incidents\)"
        )
        self.assertNotIn("arms nothing", live)

    def test_the_notes_of_the_other_features_that_send_by_their_modes(self):
        """In prototype the alerts, the team dashboard and ticket creation do their work and send nothing, and their
        modes' notes say so; live, they send."""
        on = {
            "alerts": {"mode": "prototype", "develop_breaks": {"enabled": True}},
            "dashboards": {
                "team": {
                    "enabled": True,
                    "mode": "prototype",
                    "delivery": "publish",
                    "priority_requests": {"enabled": False},
                    "publish": {"web_host": "web.example.com"},
                }
            },
            "ticket_creation": {
                "mode": "prototype",
                "follow_up_tickets": {"enabled": True},
            },
        }
        proto = summary(features=on)
        for line in (
            r"(?m)^  mode: +prototype \(messages are written and kept here, nothing is sent\)$",
            r"(?m)^  team\.enabled: +true \(published to web\.example\.com once live\)$",
            r"(?m)^  team\.mode: +prototype \(the page is made every tick, nothing is pushed to a web host\)$",
            r"(?m)^  mode: +prototype \(tickets are written and kept here, nothing is filed\)$",
        ):
            self.assertRegex(proto, line)
        live = summary(
            features=merged(
                on,
                {
                    "alerts": {"mode": "live"},
                    "dashboards": {"team": {"mode": "live"}},
                    "ticket_creation": {"mode": "live"},
                },
            )
        )
        for line in (
            r"(?m)^  mode: +live \(messages are sent\)$",
            r"(?m)^  team\.enabled: +true \(published to web\.example\.com\)$",
            r"(?m)^  team\.mode: +live \(the page is delivered every tick\)$",
            r"(?m)^  mode: +live \(tickets are filed\)$",
        ):
            self.assertRegex(live, line)

    def test_a_served_team_dashboard_shows_where(self):
        text = summary(
            features={
                "dashboards": {
                    "team": {
                        "enabled": True,
                        "mode": "live",
                        "delivery": "serve",
                        "priority_requests": {"enabled": False},
                        "serve": {"address": "0.0.0.0", "port": 18700},
                    }
                }
            }
        )
        self.assertRegex(
            text, r"(?m)^  team\.enabled: +true \(served here, 0\.0\.0\.0:18700\)$"
        )

    def test_a_served_team_dashboard_is_open_to_anyone_who_reaches_it(self):
        """While this host serves the team dashboard, the exposure sentence names its address and port: anyone
        who reaches them can read it and send a priority request, with priority_requests off too (its server takes
        them whenever it serves the page); no sentence while it is published to a web host or off.
        """
        served = {
            "enabled": True,
            "mode": "live",
            "delivery": "serve",
            "priority_requests": {"enabled": False},
            "serve": {"address": "0.0.0.0", "port": 18700},
        }
        r = read(features={"dashboards": {"team": served}})
        self.assertEqual(
            confirm.exposure(r.plan.variables),
            "anyone who reaches 0.0.0.0:18700 can read the team dashboard and send a priority request",
        )
        published = {
            "enabled": True,
            "mode": "live",
            "delivery": "publish",
            "priority_requests": {"enabled": True},
            "publish": {"web_host": "web.example.com"},
        }
        for team in (published, {"enabled": False}):
            with self.subTest(team=team):
                r = read(features={"dashboards": {"team": team}})
                self.assertEqual(confirm.exposure(r.plan.variables), "")

    def test_who_can_reach_the_team_dashboard_a_change_serves_or_moves(self):
        """After the first start: the exposure sentence when a change serves the team dashboard, or moves the
        address or port it is served on, by docker/.env's variables as they are on this host; none while it stays where
        it is served, or stops being served."""

        def served(address="0.0.0.0", port=18700) -> dict:
            team = {"enabled": True, "mode": "live", "delivery": "serve"}
            team["serve"] = {"address": address, "port": port}
            return read(features={"dashboards": {"team": team}}).plan.variables

        here, off = served(), read().plan.variables
        sentence = "anyone who reaches {} can read the team dashboard and send a priority request"
        self.assertEqual(
            confirm.exposure_changed(off, here), sentence.format("0.0.0.0:18700")
        )
        self.assertEqual(confirm.exposure_changed(here, here), "")
        self.assertEqual(confirm.exposure_changed(here, off), "")
        self.assertEqual(
            confirm.exposure_changed(here, served(port=18701)),
            sentence.format("0.0.0.0:18701"),
        )
        self.assertEqual(
            confirm.exposure_changed(here, served(address="127.0.0.1")),
            sentence.format("127.0.0.1:18700"),
        )
        # docker/.env edited by hand: its port without its address is not where the plan serves it
        hand = {k: v for k, v in here.items() if k != "RKMON_TEAM_HOST"}
        self.assertEqual(
            confirm.exposure_changed(hand, here), sentence.format("0.0.0.0:18700")
        )

    def test_the_alerts_message_label(self):
        """The alerts' section names the text each message starts with, the deploy file's message label; a blank one
        shows blank."""
        on = {"alerts": {"mode": "live", "develop_breaks": {"enabled": True}}}
        labelled = merged(on, {"alerts": {"message_label": "the team"}})
        text = summary(features=labelled)
        self.assertRegex(text, r"(?m)^  message_label: +the team$")
        self.assertNotIn("channel", text)
        text = summary(features=on)
        self.assertRegex(text, r'(?m)^  message_label: +""$')

    def test_never_a_secret(self):
        r = read(
            model_bindings={"b1": dict(GLM, api_key_secret_name="b1_api_key")},
            features={
                "dashboards": {
                    "team": {
                        "enabled": True,
                        "mode": "live",
                        "delivery": "publish",
                        "priority_requests": {"enabled": False},
                        "publish": {"web_host": "web.example.com"},
                    }
                },
                "review_automation": {"passes": {"ci_triage_pass": {"model": "b1"}}},
            },
        )
        # A deploy settings file refuses an address with a user or a password in it, and holds no secret's path: the
        # configuration takes these from elsewhere, and the summary still shows none of them.
        values = r.run.values
        for key, value in (
            ("model_bindings.b1.url", "https://u:secret-x3@models.example.com/v1"),
            ("atlassian.proxy_url", "http://u:secret-x1@proxy.invalid"),
            ("publish.html.key", "/keys/secret-x2"),
            ("teams.webhook_file", "/k/secret-x7"),
            ("admin_dashboard.token_file", "/k/secret-x8"),
        ):
            deploy_tree.put(values, key, value)
        text = confirm.render(
            r.plan,
            values,
            GOOD,
            [{"name": "github_read", "state": "SET", "found": "secret-x9"}],
        )
        self.assertNotRegex(text.lower(), r"secret[-_]x")
        image = confirm.image_rows("rocke-monitor:t", (["rk"], ""))
        said = {
            u: f"refused: {u}" for _n, _p, u, _used in confirm.models_in_use(values)
        }
        models = confirm.model_rows(values, said)
        got = confirm.checks(r.run, r.doc, r.rows, image, models)
        self.assertNotRegex(render(got).lower(), r"secret[-_]x")


class TestRows(unittest.TestCase):
    def test_what_the_container_refuses_to_run_is_a_fail_with_its_edit(self):
        """The configuration's problems are the ones the container's load finds in the settings the confirm delivers
        (runtime_config.Resolved's), each a FAIL whose fix is an edit of the file at the key as the file names it.
        """
        r = read()
        line = "claude.tick_min 0 is less than 1"
        got = confirm.checks(r.run._replace(problems=[line]), r.doc, r.rows)
        self.assertEqual(
            [(c.status, c.found) for c in got if c.name == "configuration"],
            [(Status.FAIL, line)],
        )
        self.assertIn("monitor.tick_interval_minutes", render(got))

    def test_the_coherence_row_then_the_problems_each_with_its_fix(self):
        """The checks are rows as checks.render prints them: the features' coherence first, as the check step shows
        it, then each problem, a WARN with its fix."""
        got = checks()
        self.assertTrue(all(isinstance(c, Check) for c in got), got)
        self.assertEqual(
            (got[0].name, got[0].status, got[0].found),
            ("coherence", Status.PASS, "the features' settings fit together"),
        )
        self.assertIn(
            "  PASS  coherence: the features' settings fit together", render(got)
        )
        both_off = checks(
            features={
                "review_automation": {"enabled": False},
                "merge_queue": {"enabled": False},
            }
        )
        warned = [c for c in both_off if c.status is Status.WARN]
        self.assertTrue(warned, render(both_off))
        self.assertTrue(all(c.fix is not None for c in warned))
        self.assertRegex(render(both_off), r"(?m)^ +fix: ")


class TestWalkList(unittest.TestCase):
    """The login walk's list (confirm.walk_list), which deploy step 5 shows and state 7 reads."""

    @staticmethod
    def rows(values, **states) -> list:
        """`rk logins --json`'s rows for VALUES, as far as the list reads them: each entry the walk lists, in the
        state STATES names, else MISSING."""
        return [
            {"name": e.name, "state": states.get(e.name, "MISSING"), "found": ""}
            for e in secrets_registry.walk_entries(values)
        ]

    @staticmethod
    def listed(values, rows) -> dict:
        return {r["name"]: r for r in confirm.walk_list(values, rows)}

    def test_what_needs_each_one_comes_from_the_settings(self):
        """Each entry the walk lists, in its order, needed by what the settings turn on: one they need is MISSING
        though the rows read it as NOT NEEDED; one nothing needs is NOT NEEDED unless it is set. MISSING and WRONG are
        UNSET. Each has the command that sets it."""
        values = read(features={"review_automation": {"mode": "live"}}).run.values
        rows = self.rows(
            values,
            github_read="SET",
            github_write="NOT NEEDED",
            claude="WRONG",
            teams_webhook="SET",
        )
        listed = self.listed(values, rows)
        self.assertEqual(
            list(listed), [e.name for e in secrets_registry.walk_entries(values)]
        )
        self.assertEqual(
            {name: r["state"] for name, r in listed.items()},
            {
                "github_read": "SET",
                "github_write": "MISSING",
                "atlassian": "NOT NEEDED",
                "claude": "WRONG",
                "admin_access_token": "MISSING",
                "teams_webhook": "SET",
                "web_host_ssh_key": "NOT NEEDED",
                "gpu_ssh_key": "NOT NEEDED",
            },
        )
        self.assertEqual(
            [name for name, r in listed.items() if r["state"] in confirm.UNSET],
            ["github_write", "claude", "admin_access_token"],
        )
        self.assertEqual(
            (listed["github_write"]["needed_by"], listed["github_write"]["waiting"]),
            (["review posting"], ""),
        )
        self.assertEqual(
            [listed[n]["command"] for n in ("github_write", "admin_access_token")],
            ["login github_write", "secret admin_access_token --generate"],
        )

    def test_one_a_feature_in_prototype_waits_for_names_the_mode(self):
        """Nothing that is on needs one a feature in prototype waits for: the modes it waits for, in
        words; the atlassian login's use meanwhile, with the program's name; none for one whose feature is off.
        """
        values = read(
            features={
                "alerts": {"mode": "prototype", "develop_breaks": {"enabled": True}},
                "ticket_creation": {
                    "mode": "prototype",
                    "follow_up_tickets": {"enabled": True},
                },
            }
        ).run.values
        mode = "features.{}.mode is prototype: nothing goes out until it is live"
        with mock.patch.object(program, "name", return_value="./rkmon"):
            listed = self.listed(values, self.rows(values))
        self.assertEqual(
            {
                n: (r["state"], r["waiting"])
                for n, r in listed.items()
                if not r["needed_by"]
            },
            {
                "github_write": (
                    "NOT NEEDED",
                    f"{mode.format('review_automation')}; {mode.format('merge_queue')}",
                ),
                "atlassian": ("NOT NEEDED", mode.format("ticket_creation")),
                "teams_webhook": ("NOT NEEDED", mode.format("alerts")),
                "web_host_ssh_key": ("NOT NEEDED", ""),
                "gpu_ssh_key": ("NOT NEEDED", ""),
            },
        )
        self.assertEqual(
            {n: r["use"] for n, r in listed.items() if r["use"]},
            {
                "atlassian": "tickets are prepared without it, and with it ./rkmon check reads your JIRA project's "
                "required fields early"
            },
        )

    def test_one_the_rows_do_not_list(self):
        """An entry rk logins does not list (an older image's) is UNKNOWN, and says so, while something needs it, and
        NOT NEEDED while nothing does; with no settings (VALUES None), the rows as they are.
        """
        values = read().run.values
        listed = self.listed(values, [])
        self.assertEqual(
            (listed["github_read"]["state"], listed["github_read"]["found"]),
            ("UNKNOWN", "the image's rk logins does not list it"),
        )
        self.assertEqual(listed["gpu_ssh_key"]["state"], "NOT NEEDED")
        row = {"name": "github_read", "state": "MISSING", "found": ""}
        self.assertEqual(
            confirm.walk_list(None, [row, "not a row"]),
            [{"waiting": "", "use": "", **row}],
        )

    def test_the_ssh_key_of_a_host_whose_own_key_is_not_recorded(self):
        """The ssh key of a host something needs, set while the host's own key is not recorded (rk logins: WRONG, and
        its host_key): `login`, which shows that key and records it; while nothing needs it, NOT NEEDED, with the
        command that sets it."""
        team = {"enabled": True, "mode": "live", "delivery": "publish"}
        wrong = {
            "name": "web_host_ssh_key",
            "state": "WRONG",
            "found": "the key is set; the web host's own key is not recorded",
            "host_key": True,
        }
        for parts, want in (
            ({"dashboards": {"team": team}}, ("WRONG", "login", True)),
            ({}, ("NOT NEEDED", "secret web_host_ssh_key", False)),
        ):
            with self.subTest(parts=parts):
                values = read(features=parts).run.values
                row = self.listed(values, [wrong])["web_host_ssh_key"]
                self.assertEqual((row["state"], row["command"], row["host_key"]), want)


class TestRules(unittest.TestCase):
    """Each coherence rule, failing then passing."""

    def check(self, bad: dict, good: dict, name: str, status: str, words: str):
        got = found(checks(**bad), name)
        self.assertTrue(any(s == status and words in f for s, f in got), got)
        got = found(checks(**good), name)
        self.assertFalse(any(words in f for s, f in got), got)

    def test_ticket_creation_needs_its_project_and_the_atlassian_site(self):
        on = {
            "features": {
                "ticket_creation": {
                    "mode": "live",
                    "follow_up_tickets": {"enabled": True},
                    "found_issue_tickets": {"enabled": True},
                }
            }
        }
        site = {
            "integrations": {"atlassian_cli": {"jira_url": "https://jira.example.com"}}
        }
        self.check(
            on,
            merged(
                merged(on, site),
                {"features": {"ticket_creation": {"project_key": "TK"}}},
            ),
            "features",
            "FAIL",
            "features.ticket_creation is enabled but features.ticket_creation.project_key is blank",
        )
        self.check(
            on,
            merged(on, site),
            "features",
            "FAIL",
            "features.ticket_creation is enabled but integrations.atlassian_cli.jira_url is blank",
        )

    def test_a_follow_up_ticket_needs_review_automation(self):
        tickets = {
            "ticket_creation": {
                "mode": "live",
                "follow_up_tickets": {"enabled": True},
                "project_key": "TK",
            }
        }
        site = {"atlassian_cli": {"jira_url": "https://jira.example.com"}}
        self.check(
            {
                "features": dict(tickets, review_automation={"enabled": False}),
                "integrations": site,
            },
            {"features": tickets, "integrations": site},
            "features",
            "FAIL",
            "features.ticket_creation.follow_up_tickets.enabled is true but features.review_automation is off",
        )

    def test_the_team_dashboard(self):
        def team(delivery: str) -> dict:
            return {
                "features": {
                    "dashboards": {
                        "team": {
                            "enabled": True,
                            "mode": "live",
                            "delivery": delivery,
                            "priority_requests": {"enabled": False},
                        }
                    }
                }
            }

        serve = team("serve")
        self.check(
            team("publish"),
            serve,
            "features",
            "FAIL",
            "features.dashboards.team is enabled (delivery publish) but these are blank: "
            "features.dashboards.team.publish.web_host",
        )
        self.check(
            merged(serve, {"features": {"dashboards": {"admin": {"enabled": False}}}}),
            serve,
            "features",
            "FAIL",
            "features.dashboards.team is enabled but features.dashboards.admin is not",
        )

    def test_alerts_need_no_message_label(self):
        """The message label only labels each message (the webhook decides the channel): alerts on fail nothing
        in the features, with a blank label or with one."""
        on = {
            "features": {
                "alerts": {"mode": "live", "develop_breaks": {"enabled": True}}
            }
        }
        label = {"features": {"alerts": {"message_label": "the team"}}}
        for given in (on, merged(on, label)):
            got = found(checks(**given), "features")
            self.assertFalse([f for s, f in got if s == "FAIL" and "alerts" in f], got)

    def test_auto_merge_in_prototype_arms_nothing(self):
        on = {"features": {"merge_queue": {"auto_merge": {"enabled": True}}}}
        self.check(
            on,
            merged(on, {"features": {"merge_queue": {"mode": "live"}}}),
            "features",
            "WARN",
            "features.merge_queue.auto_merge is enabled, but the merge queue is in prototype",
        )

    def test_both_features_off(self):
        self.check(
            {
                "features": {
                    "review_automation": {"enabled": False},
                    "merge_queue": {"enabled": False},
                }
            },
            {"features": {"review_automation": {"enabled": False}}},
            "features",
            "WARN",
            "features.review_automation and features.merge_queue are both off",
        )

    def test_more_parallel_work_than_cpus(self):
        reviews = {"review_automation": {"parallel_reviews": 3}}
        self.check(
            {"host": {"resources": {"cpus": 2}}, "features": reviews},
            {"host": {"resources": {"cpus": 4}}, "features": reviews},
            "CPUs",
            "WARN",
            "host.resources.cpus 2 is fewer than the reviews and builds that may run at once (4)",
        )

    def test_empty_scope_labels(self):
        self.check(
            {"repository": {"scope_labels": []}},
            {"repository": {"scope_labels": ["x"]}},
            "settings file",
            "FAIL",
            "repository.scope_labels is blank",
        )

    def test_the_file_s_refusals_once_each_by_its_keys(self):
        """Each value the file's reader refuses is one FAIL, named by its key; the configuration's own checks of what
        the key feeds add nothing, and no internal name shows."""
        for bad, key, words in (
            (
                {"monitor": {"tick_interval_minutes": 0}},
                "monitor.tick_interval_minutes",
                "0 is less than 1",
            ),
            (
                {
                    "model_bindings": {"b1": {"provider": "ollama", "model_id": "m"}},
                    **passes(ci_triage_pass={"model": "b1"}),
                },
                "model_bindings.b1.server_url",
                "missing",
            ),
            (
                passes(ci_triage_pass={"model": "nowhere"}),
                "features.review_automation.passes.ci_triage_pass.model",
                '"nowhere" is neither a built-in model nor an entry of model_bindings',
            ),
            (
                {
                    "model_bindings": {"glm-flash": GLM},
                    **passes(verification_pass={"model": "glm-flash"}),
                },
                "features.review_automation.passes.verification_pass.model",
                "this pass takes a Claude model",
            ),
        ):
            with self.subTest(key=key):
                got = [c for c in checks(**bad) if c.status == "FAIL"]
                self.assertEqual([c.name for c in got], [key], [c.found for c in got])
                self.assertIn(words, got[0].found)
                for internal in ("claude.tick_min", "models.", ".url ", "reviewer"):
                    self.assertNotIn(internal, got[0].found)

    def test_the_configuration_s_checks_repeat_no_row_of_the_file(self):
        got = render(checks())
        self.assertNotIn("paths.working_dir", got)
        got = checks(integrations={"github_cli": {"username": ""}})
        self.assertIn("integrations.github_cli.username", [c.name for c in got])
        self.assertEqual([c.found for c in got if c.name == "configuration"], [])
        self.assertEqual(
            sum("integrations.github_cli.username is blank" in c.found for c in got),
            1,
            [c.found for c in got],
        )


class TestImage(unittest.TestCase):
    """The image row: the image image.tag names was built from this package's image recipe when its own copies of the
    files an image keeps (code_volume.IMAGE_BOUND) are this package's, by the fingerprints a one-off container of it
    prints: code_volume.fingerprints' own, a few KB where the copies themselves were megabytes.
    """

    def printed(self, folder=ROOT) -> str:
        """What CODE_PROBE prints in a container whose own code is FOLDER."""
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                confirm.CODE_PROBE,
                str(folder),
                *code_volume.IMAGE_BOUND,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def test_the_image_s_own_copies_are_read_by_their_fingerprints(self):
        printed = self.printed()
        held = json.loads(printed)
        self.assertEqual(held, code_volume.fingerprints(ROOT, code_volume.IMAGE_BOUND))
        self.assertEqual(confirm.changed_copies(printed, ROOT), [])
        # a fingerprint for each file, not the file
        self.assertLess(len(printed), 200 * len(held))
        with tempfile.TemporaryDirectory() as t:
            other = pathlib.Path(t)
            (other / "rk").write_bytes((ROOT / "rk").read_bytes())
            (other / "rk").chmod(0o644)  # the same bytes, another execute bit
            held["rk"] = code_volume.fingerprints(other, ("rk",))["rk"]
        held["docker/Dockerfile"] = held["docker/pkgs.sh"]
        del held["docker/entrypoint.sh"]
        self.assertEqual(
            confirm.changed_copies(json.dumps(held), ROOT),
            ["docker/Dockerfile", "docker/entrypoint.sh", "rk"],
        )

    def test_an_answer_that_is_no_fingerprint_of_a_file_it_keeps_is_refused(self):
        held = "0123456789abcdef" * 4
        for printed, words in (
            ("TLS ok", "its answer is no JSON"),
            ("[]", "its answer is no JSON object"),
            (json.dumps({"../outside": held}), "names ../outside,"),
            (json.dumps({"/etc/hosts": held}), "names /etc/hosts,"),
            (json.dumps({"requirements/../../x": held}), "names requirements/../../x,"),
            (json.dumps({"README.md": held}), "names README.md,"),
            (json.dumps({"rk": "x"}), "its answer has no fingerprint of rk"),
            (json.dumps({"rk": [held, False]}), "its answer has no fingerprint of rk"),
            (json.dumps({"rk": held.upper()}), "its answer has no fingerprint of rk"),
        ):
            with self.subTest(printed=printed):
                with self.assertRaisesRegex(ValueError, re.escape(words)):
                    confirm.changed_copies(printed, ROOT)

    def test_the_image_row_passes_fails_or_waits_for_the_image(self):
        ref = "rocke-monitor:2026.11.20"

        def rows(code, **kw):
            return [(c.status, c.found) for c in confirm.image_rows(ref, code, **kw)]

        self.assertEqual(
            rows(([], "")),
            [(Status.PASS, f"{ref} was built from this package's image recipe")],
        )
        # Not on this host: step 4 brings it, before the first start; then the command that brings it.
        self.assertEqual(
            rows(None),
            [(Status.SKIP, "checked at step 4, once it is on this host")],
        )
        for pull, command in ((False, "image load IMAGE_FILE"), (True, "image pull")):
            self.assertEqual(
                rows(None, first=False, pull=pull),
                [
                    (
                        Status.SKIP,
                        f"{ref} is not on this host; {steps.command_line(command)} checks it",
                    )
                ],
            )
        for code, found in (
            (
                (["docker/Dockerfile", "rk"], ""),
                f"{ref} was not built from this package's image recipe: its copies of docker/Dockerfile, rk are "
                "not this package's",
            ),
            (
                (None, "exec: python3: not found"),
                f"{ref}: its own code could not be read (exec: python3: not found)",
            ),
        ):
            with self.subTest(found=found):
                [row] = confirm.image_rows(ref, code)
                self.assertEqual((row.status, row.found), (Status.FAIL, found))
                fix = render([row])
                self.assertIn("QUICK_BUILD_GUIDE.md", fix)
                self.assertIn("image.tag", fix)


class TestHeld(unittest.TestCase):
    """Before the first start the logins and secrets are read in the volumes that keep them: where the
    container keeps each one the settings need (login_places), and what a one-off container of the image prints of
    them (HELD_PROBE: secrets_registry.held's own code), as unheld reads it."""

    def test_each_place_is_the_container_s(self):
        values = read(features={"review_automation": {"mode": "live"}}).run.values
        rows = confirm.login_places(values)
        places: dict = {}
        for n, place, files in rows:
            places.setdefault(
                n, (place, files)
            )  # its own place first, then an older name's
        self.assertEqual(list(places), confirm.logins_needed(values))
        self.assertEqual(
            places["github_read"], (f"{SECRETS_DIR}/gh-read", ["hosts.yml"])
        )
        self.assertEqual(
            places["github_write"], (f"{WRITE_SECRETS_DIR}/gh", ["hosts.yml"])
        )
        self.assertEqual(places["claude"], (f"{SECRETS_DIR}/claude_key", None))
        self.assertIn(
            ("github_write", f"{SECRETS_DIR}/gh", ["hosts.yml"]), rows
        )  # a container from before the read/write split kept the write login in the monitor's volume
        self.assertIn(
            ("claude", f"{SECRETS_DIR}/claude_env", None), rows
        )  # an older package kept the Claude session's key in the secret claude_env

    def test_the_claude_login_under_its_older_name_is_held(self):
        """Until the one-time move copies the Claude session's key from the older secret claude_env to the claude
        login's file, a claude_env in the monitor's volume holds the login (state 7).
        """
        rows = [r for r in confirm.login_places(read().run.values) if r[0] == "claude"]
        self.assertEqual(
            rows,
            [
                ("claude", f"{SECRETS_DIR}/claude_key", None),
                ("claude", f"{SECRETS_DIR}/claude_env", None),
            ],
        )
        with tempfile.TemporaryDirectory() as t:
            places = [(n, p.replace(SECRETS_DIR, t, 1), f) for n, p, f in rows]

            def unheld() -> tuple:
                printed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        confirm.HELD_PROBE,
                        json.dumps(places),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                return confirm.unheld(printed, places)

            self.assertEqual(unheld(), ("claude",))
            older = pathlib.Path(t) / "claude_env"
            older.write_text('{"ANTHROPIC_API_KEY": "invented"}\n', encoding="utf-8")
            self.assertEqual(unheld(), ())
            older.unlink()
            (pathlib.Path(t) / "claude_key").write_text("invented\n", encoding="utf-8")
            self.assertEqual(unheld(), ())

    def test_an_older_copy_the_move_copies_is_one_of_its_places(self):
        """Until the one-time move ends, an entry under an older name the move copies counts as set (state 7): a row for
        that place follows the entries' own, and none for an older name the move only removes.
        """
        from rkmon_lib import secrets_registry as sr

        values = read().run.values
        entry = sr.Secret(
            "new_secret",
            "a secret",
            sr.ONE_LINE,
            "you",
            sr.MONITOR,
            (sr.ALWAYS,),
            older=(sr.Older("old_secret"), sr.Older("left_secret", copied=False)),
        )
        login = sr.Secret(
            "a-login",
            "a login",
            sr.LOGIN,
            "you",
            sr.WRITER,
            (sr.ALWAYS,),
            files=("hosts.yml",),
            older=(sr.Older("a-login"), sr.Older("a-login-token", file=True)),
        )
        with mock.patch.object(sr, "_fixed", return_value=(entry, login)):
            self.assertEqual(
                confirm.login_places(values),
                [
                    ("new_secret", f"{SECRETS_DIR}/new_secret", None),
                    ("a-login", f"{WRITE_SECRETS_DIR}/a-login", ["hosts.yml"]),
                    ("new_secret", f"{SECRETS_DIR}/old_secret", None),
                    ("a-login", f"{SECRETS_DIR}/a-login", ["hosts.yml"]),
                    ("a-login", f"{SECRETS_DIR}/a-login-token", None),
                ],
            )

    def test_the_probe_names_the_places_that_hold_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            base = pathlib.Path(t)
            (base / "login").mkdir()
            (base / "login" / "hosts.yml").write_text("x", encoding="utf-8")
            (base / "token").write_text("t\n", encoding="utf-8")
            (base / "blank").write_text("  \n", encoding="utf-8")
            places = [
                ("a-login", str(base / "login"), ["hosts.yml"]),
                ("half-a-login", str(base / "login"), ["hosts.yml", "config.yml"]),
                ("a-secret", str(base / "token"), None),
                ("a-blank-secret", str(base / "blank"), None),
                ("a-folder-for-a-secret", str(base / "login"), None),
                ("none-there", str(base / "none"), None),
                ("no-place", "", None),
                # An entry with an older copy the one-time move copies: held when one of its places holds it.
                ("two-places", str(base / "none"), None),
                ("two-places", str(base / "token"), None),
                ("two-empty-places", str(base / "none"), None),
                ("two-empty-places", "", None),
            ]
            printed = subprocess.run(
                [sys.executable, "-I", "-c", confirm.HELD_PROBE, json.dumps(places)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        self.assertEqual(
            confirm.unheld(printed, places),
            (
                "half-a-login",
                "a-blank-secret",
                "a-folder-for-a-secret",
                "none-there",
                "no-place",
                "two-empty-places",
            ),
        )
        for answer in ("", "{}", '["another"]', "[1]"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                confirm.unheld(answer, places)


# An invented key, and the model variables an older package's Claude settings set.
KEY = "test-" + "key-not-real"
MODELS = {
    "ANTHROPIC_MODEL": "model-a",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "model-b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "model-c",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model-d",
}
GATEWAY = "https://gateway.example.com/claude"


class TestClaudeEnv(unittest.TestCase):
    """Check 16, while this deployment's secrets keep claude_env, an older package's Claude settings:
    where the container keeps it and the claude login (claude_env_places, the registry's places); what a one-off
    container of the image prints of it (SHAPE_PROBE: names and settings, never the key, and why root's own reading
    copies no key), as claude_env_answer reads it; and its rows against integrations.claude_code (claude_env_rows):
    while the claude login is not set, a WARN in root's words when the move copies no key; a WARN for each thing the
    session will not send; a FAIL for each setting that sends the session or the key otherwise, whose fix sets the
    file's; a PASS when none does; and a PASS that names the model variables the move leaves behind. No row and no
    answer holds the key.
    """

    NAME = "claude_env"
    SAME = "integrations.claude_code sends the session as claude_env does"
    NO_COPY = "the one-time move copies no key from claude_env to the claude login: "
    NOT_SENT = "the session will not send all that claude_env sends: "
    # The fix of what the session will not send, after the file's own while the claude login is not set: check 16 cannot
    # tell whether the move has run. With the login set, the file's own is the fix alone.
    EITHER = ", before the first start; or set the key with rkmon login claude"

    def not_sent_fix(self, own: str, login: bool) -> str:
        """The fix of what the session will not send: the file's own OWN, and the login's way while LOGIN is not set."""
        return own if login else own + self.EITHER

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        login = secrets_registry.entry({}, secrets_registry.CLAUDE_LOGIN)
        self.set_login = str(secrets_registry.set_fix(login))

    def shape(self, env: dict):
        """The shape of a claude_env of ENV, as the image's own reader gives it to check 16."""
        path = self.dir / "shaped"
        path.write_text(json.dumps(env), encoding="utf-8")
        return claude_env_shape.read(path, settings_file.secret_shape)

    def found(self, env: dict, login=False):
        """What check 16's one-off container finds of a claude_env of ENV, with the claude login set or not (LOGIN):
        its shape, and why root's own reading of it, with no judgement, copies no key.
        """
        shape = self.shape(env)
        move = claude_env_shape.move(claude_env_shape.read(self.dir / "shaped"))
        return confirm.Found(self.NAME, shape, move, login)

    def rows(self, env, login=False, **claude_code) -> tuple:
        """(Check 16's rows, as (status, found, fix), of a claude_env of ENV, with the claude login set or not
        (LOGIN), against BASE with integrations.claude_code CLAUDE_CODE; the file a fix edits, as messages show it.)
        """
        r = read(integrations={"claude_code": claude_code})
        found = self.found(env, login)
        got = confirm.claude_env_rows(r.run.values, r.doc, found)
        self.assertNotIn(KEY, render(got))
        rows = [(c.status, c.found, c.fix and str(c.fix)) for c in got]
        return rows, settings_file.shown(r.doc.path)

    def test_a_bearer_token_against_a_header_and_the_model_variables(self):
        """A claude_env that sends the key as a bearer token, against settings that send it in a
        header, and its four model variables."""
        r = read()
        doc = dataclasses.replace(r.doc, path=pathlib.Path.home() / "rkmon-deploy.yaml")
        found = self.found({"ANTHROPIC_AUTH_TOKEN": KEY, **MODELS})
        got = confirm.claude_env_rows(r.run.values, doc, found)
        want = """
  FAIL  Claude Code: claude_env sends the key as a bearer token, and integrations.claude_code.api_key_sent_as
        says "header"
        fix: set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml to "bearer"
  PASS  Claude Code: claude_env sets ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_OPUS_MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL
        and ANTHROPIC_DEFAULT_HAIKU_MODEL; the move leaves them behind, since each pass asks for its model
        (features.review_automation) by its built-in id
"""
        # the same words, wrapped at the terminal's width
        self.assertEqual(" ".join(render(got).split()), " ".join(want.split()))
        self.assertEqual(
            [(c.status, c.fix and str(c.fix)) for c in got],
            [
                (
                    Status.FAIL,
                    'set integrations.claude_code.api_key_sent_as in ~/rkmon-deploy.yaml to "bearer"',
                ),
                (Status.PASS, None),
            ],
        )

    def test_each_setting_that_sends_otherwise_fails_and_its_fix_sets_the_file_s(self):
        """Where the session sends, only where the file sets an address; how the key is sent; then, but as a bearer
        token alone, in which header; then, in a header of its own, after which word."""
        api = {"ANTHROPIC_API_KEY": KEY}
        header = "api_key_header"
        for env, claude_code, want in (
            (
                {"ANTHROPIC_BASE_URL": GATEWAY, **api},
                {},
                [("base_url", f"sends the session to {GATEWAY}", "", GATEWAY)],
            ),
            (
                api,
                {"api_key_sent_as": "bearer"},
                [("api_key_sent_as", "sends the key in a header", "bearer", "header")],
            ),
            (
                {"ANTHROPIC_AUTH_TOKEN": KEY, **api},
                {},
                [
                    (
                        "api_key_sent_as",
                        "sends the key in a header and as a bearer token",
                        "header",
                        "header_and_bearer",
                    )
                ],
            ),
            (
                {"ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}", **api},
                {},
                [
                    (
                        header,
                        "sends the key in the custom header X-Gateway-Key",
                        "",
                        "X-Gateway-Key",
                    ),
                    (
                        "api_key_scheme",
                        "sends the word Token before the key",
                        "",
                        "Token",
                    ),
                ],
            ),
            (
                api,
                {header: "X-Gateway-Key", "api_key_scheme": "Token"},
                [(header, "sends the key in no custom header", "X-Gateway-Key", "")],
            ),
            (
                {"ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}", **api},
                {header: "X-Gateway-Key", "api_key_scheme": "Token"},
                [("api_key_scheme", "sends no word before the key", "Token", "")],
            ),
        ):
            with self.subTest(env=sorted(env), claude_code=claude_code):
                got, file = self.rows(env, **claude_code)
                self.assertEqual(
                    got,
                    [
                        (
                            Status.FAIL,
                            f"claude_env {does}, and integrations.claude_code.{k} says {json.dumps(says)}",
                            f"set integrations.claude_code.{k} in {file} to {json.dumps(value)}",
                        )
                        for k, does, says, value in want
                    ],
                )

    def test_settings_that_send_as_the_file_does_pass(self):
        """Each claude_env against the settings claude_env_shape.settings makes from it: one PASS. The address is not
        compared where the file sets none, nor the header and the word before the key where it sends the key as a bearer
        token alone."""
        api, bearer = {"ANTHROPIC_API_KEY": KEY}, {"ANTHROPIC_AUTH_TOKEN": KEY}
        for env, claude_code in (
            (api, {}),
            (
                api,
                {"api_key_sent_as": "header", "base_url": "https://other.example.com"},
            ),
            (
                bearer,
                {
                    "api_key_sent_as": "bearer",
                    "api_key_header": "X-Other",
                    "api_key_scheme": "Token",
                },
            ),
            ({**api, **bearer}, {"api_key_sent_as": "header_and_bearer"}),
            (
                {"ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: {KEY}", **api},
                {"api_key_header": "X-Gateway-Key"},
            ),
            (
                {"ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}\n", **api},
                {"api_key_header": "X-Gateway-Key", "api_key_scheme": "Token"},
            ),
            (
                {"ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Bearer {KEY}", **bearer},
                {
                    "api_key_sent_as": "header_and_bearer",
                    "api_key_header": "X-Gateway-Key",
                    "api_key_scheme": "Bearer",
                },
            ),
            ({"ANTHROPIC_BASE_URL": GATEWAY, **api}, {"base_url": GATEWAY}),
        ):
            with self.subTest(env=sorted(env), claude_code=claude_code):
                got, _file = self.rows(env, **claude_code)
                self.assertEqual(got, [(Status.PASS, self.SAME, None)])

    def test_the_model_variables_are_left_behind(self):
        got, _file = self.rows({"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_MODEL": "model-a"})
        self.assertEqual(
            got,
            [
                (Status.PASS, self.SAME, None),
                (
                    Status.PASS,
                    "claude_env sets ANTHROPIC_MODEL; the move leaves it behind, since each pass asks for its model "
                    "(features.review_automation) by its built-in id",
                    None,
                ),
            ],
        )

    def test_while_the_move_copies_no_key_a_warn_names_the_login_until_it_is_set(self):
        """A shape the settings cannot carry (then nothing is compared), a file with no key, and a file its reader
        refuses: the one-time move copies no key, so a WARN in root's words, whose fix sets the claude login, while it
        is not set. What the session will not send has its own WARN, the login set or not, its fix the file's edit, and
        while the login is not set the login's way too.
        """
        other = "test-" + "other-key-not-real"
        two = "ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are two different keys, and the claude login is one key"
        delete = (
            "delete ANTHROPIC_API_KEY from claude_env, or delete ANTHROPIC_AUTH_TOKEN from claude_env, whichever the "
            "gateway does not take"
        )
        for env, why, not_sent in (
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": other},
                f"the deploy settings cannot carry everything in it: {two}",
                True,
            ),
            ({"ANTHROPIC_BASE_URL": GATEWAY}, claude_env_shape.NO_KEY, False),
            (
                {"ANTHROPIC_BASE_URL": GATEWAY, "ANTHROPIC_API_KEY": " \n"},
                claude_env_shape.NO_KEY,
                False,
            ),
        ):
            for login in (False, True):
                with self.subTest(why=why, login=login, env=sorted(env)):
                    got, file = self.rows(env, login=login)
                    self.assertNotIn(other, repr(got))
                    warn = (
                        []
                        if login
                        else [(Status.WARN, self.NO_COPY + why, self.set_login)]
                    )
                    two_keys = (
                        Status.WARN,
                        self.NOT_SENT + two,
                        self.not_sent_fix(delete, login),
                    )
                    fail = (
                        Status.FAIL,
                        f'claude_env sends the session to {GATEWAY}, and integrations.claude_code.base_url says ""',
                        f'set integrations.claude_code.base_url in {file} to "{GATEWAY}"',
                    )
                    self.assertEqual(got, warn + [two_keys if not_sent else fail])
        r = read()
        refused = f"{SECRETS_DIR}/claude_env is a link, not a file"
        for login in (False, True):
            found = confirm.Found(self.NAME, None, refused, login)
            got = confirm.claude_env_rows(r.run.values, r.doc, found)
            self.assertEqual(
                [(c.status, c.found, str(c.fix)) for c in got],
                (
                    []
                    if login
                    else [(Status.WARN, self.NO_COPY + refused, self.set_login)]
                ),
            )
        self.assertEqual(confirm.claude_env_rows(r.run.values, r.doc, None), [])

    def test_what_the_session_will_not_send_warns_with_the_login_set_too(self):
        """A custom header of fixed text: the session will not send it, the login set or not. The row's fix is the
        file's own, which clears it, while the login is set (secrets_registry.claude_env_fix); while it is not, the
        file's own before the first start, or the claude login's command (secrets_registry.claude_env_or_login_fix),
        and the move copies no key either, in root's words, which name no header.
        """
        env = {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": "X-Team: blue"}
        own = (
            "set ANTHROPIC_CUSTOM_HEADERS in claude_env to its lines without the header X-Team, when the gateway takes "
            "requests without it"
        )

        def header(login):
            return (
                Status.WARN,
                self.NOT_SENT
                + "the custom header X-Team sends a value that is not the key, and no setting sends a header of fixed "
                "text",
                self.not_sent_fix(own, login),
            )

        move = (
            Status.WARN,
            self.NO_COPY
            + "the deploy settings cannot carry everything in it: a custom header sends a value that is not the key, "
            "and no setting sends a header of fixed text",
            self.set_login,
        )
        # While the login is set, the row's fix is the file's edit alone.
        self.assertEqual(self.rows(env, login=True)[0], [header(True)])
        self.assertEqual(self.rows(env)[0], [move, header(False)])

    def test_an_address_only_the_judgement_flags_has_a_row_of_its_own(self):
        """Root reads the file with no judgement and copies the key while its address holds none: check 16 says so (no
        WARN about the key), gives the address a row of its own, and compares how the key is sent.
        """
        password = "pw-" + "not-real"
        env = {
            "ANTHROPIC_API_KEY": KEY,
            "ANTHROPIC_BASE_URL": f"https://user:{password}@gateway.example.com",
        }
        own = "set ANTHROPIC_BASE_URL in claude_env to the gateway's address, with no credential in it"

        def address(login):
            return (
                Status.WARN,
                self.NOT_SENT
                + "ANTHROPIC_BASE_URL looks like an address with a user or password in it, and a settings file has "
                "no credential",
                self.not_sent_fix(own, login),
            )

        self.assertEqual(self.found(env).move, "")
        for login in (False, True):
            with self.subTest(login=login):
                got, file = self.rows(env, login=login)
                self.assertNotIn(password, repr(got))
                self.assertEqual(
                    got,
                    [
                        address(login),
                        (
                            Status.PASS,
                            "integrations.claude_code sends the key as claude_env does",
                            None,
                        ),
                    ],
                )
        got, file = self.rows(env, api_key_sent_as="bearer")
        self.assertEqual(
            got,
            [
                address(False),
                (
                    Status.FAIL,
                    'claude_env sends the key in a header, and integrations.claude_code.api_key_sent_as says "bearer"',
                    f'set integrations.claude_code.api_key_sent_as in {file} to "header"',
                ),
            ],
        )

    def test_the_skip_says_why_the_secrets_are_not_read(self):
        """A SKIP shows no fix, so its words name what to do before confirm runs again: load the image (step 4)."""
        places = confirm.Places(self.NAME, f"{SECRETS_DIR}/claude_env", "")
        [row] = confirm.claude_env_skip(places, "the image is not on this host")
        self.assertEqual(
            (row.name, row.status, row.found),
            (
                "Claude Code",
                Status.SKIP,
                "this deployment's secrets are not read for claude_env, an older package's Claude settings: the "
                f"image is not on this host; load the image (step 4), then run {steps.command_line('confirm')} again",
            ),
        )
        self.assertEqual([s.n for s in steps.DEPLOY_STEPS if s.name == "image"], [4])

    def test_a_one_off_that_read_nothing_says_what_to_fix(self):
        """No answer in time names docker's wait; docker's own refusal asks for docker's fix; the image's code that
        could not read the secrets asks for this package's image; each then asks for confirm again.
        """
        places = confirm.Places(self.NAME, f"{SECRETS_DIR}/claude_env", "")
        ref, start = (
            "rocke-monitor:<tag>",
            "this deployment's secrets are not read for claude_env, an older package's",
        )
        again = f"then run {steps.command_line('confirm')} again"
        cases = {
            (124, "timed out after 120 seconds"): (
                f"a one-off container of {ref} timed out after 120 seconds; check that docker runs containers"
            ),
            (125, "docker: Error response from daemon: no space left on device"): (
                f"docker could not run a one-off container of {ref} (docker: Error response from daemon: no space "
                "left on device); fix what docker says"
            ),
            (137, "the container stopped"): (
                f"a one-off container of {ref} could not read them (the container stopped); load this package's image "
                "(step 4)"
            ),
            (0, "the answer is not one line"): (
                f"a one-off container of {ref} could not read them (the answer is not one line); load this package's "
                "image (step 4)"
            ),
        }
        for (code, why), words in cases.items():
            with self.subTest(code=code):
                [row] = confirm.claude_env_unread(places, ref, code, why)
                self.assertEqual(
                    (row.name, row.status, row.found),
                    (
                        "Claude Code",
                        Status.SKIP,
                        f"{start} Claude settings: {words}, {again}",
                    ),
                )

    def test_check_16_s_rows_come_last(self):
        r = read()
        row = Check(confirm.CLAUDE_ROW, Status.PASS, self.SAME)
        self.assertEqual(confirm.checks(r.run, r.doc, r.rows, claude=[row])[-1], row)

    def test_its_places_are_the_registry_s(self):
        """claude_env in the monitor's secrets folder, the older name the one-time move copies the claude login's key
        from (secrets_registry.older_places), and the login's own place, as the logins step reads it; none while the
        registry names no older place the move copies, or none can be known."""
        values = read().run.values
        places = confirm.claude_env_places(values)
        self.assertEqual(
            (places.name, places.older), (self.NAME, f"{SECRETS_DIR}/claude_env")
        )
        own = next(
            p
            for n, p, _f in confirm.login_places(values)
            if n == secrets_registry.CLAUDE_LOGIN
        )
        self.assertEqual(places.login, own)
        self.assertEqual(os.path.dirname(places.login), SECRETS_DIR)
        with mock.patch.object(secrets_registry, "older_places", return_value=[]):
            self.assertIsNone(confirm.claude_env_places(values))
        with mock.patch.object(secrets_registry, "path", side_effect=ValueError):
            self.assertIsNone(confirm.claude_env_places(values))

    def probe(self, older, login) -> str:
        """What SHAPE_PROBE prints in a container whose own code is this package's, of the places OLDER and LOGIN: never
        the key, nor on its errors."""
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                confirm.SHAPE_PROBE,
                str(ROOT),
                str(older),
                str(login),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertNotIn(KEY, p.stdout + p.stderr)
        self.assertEqual(len(p.stdout.splitlines()), 1, p.stdout)
        return p.stdout

    def test_the_probe_prints_names_and_settings_never_the_key(self):
        older, login = self.dir / "claude_env", self.dir / "login"
        places = confirm.Places(self.NAME, str(older), str(login))
        # No such file: nothing to compare.
        self.assertIsNone(
            confirm.claude_env_answer(0, self.probe(older, login), places)
        )
        env = {
            "ANTHROPIC_BASE_URL": GATEWAY,
            "ANTHROPIC_API_KEY": KEY,
            "ANTHROPIC_CUSTOM_HEADERS": f"X-Gateway-Key: Token {KEY}",
            **MODELS,
        }
        older.write_text(json.dumps(env), encoding="utf-8")
        want = claude_env_shape.read(older, settings_file.secret_shape)
        self.assertEqual(
            confirm.claude_env_answer(0, self.probe(older, login), places),
            confirm.Found(self.NAME, want, "", False),
        )
        login.write_text("invented\n", encoding="utf-8")
        self.assertEqual(
            confirm.claude_env_answer(0, self.probe(older, login), places),
            confirm.Found(self.NAME, want, "", True),
        )
        # Why root's own reading, with no judgement, copies no key: the words root's move prints.
        password = "pw-" + "not-real"
        for env, move in (
            ({"ANTHROPIC_BASE_URL": GATEWAY}, claude_env_shape.NO_KEY),
            (
                {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_CUSTOM_HEADERS": "X-Team: blue"},
                "the deploy settings cannot carry everything in it: a custom header sends a value that is not the key, "
                "and no setting sends a header of fixed text",
            ),
            (
                {
                    "ANTHROPIC_API_KEY": KEY,
                    "ANTHROPIC_BASE_URL": f"https://user:{password}@gateway.example.com",
                },
                "",
            ),
        ):
            with self.subTest(env=sorted(env)):
                older.write_text(json.dumps(env), encoding="utf-8")
                printed = self.probe(older, login)
                self.assertNotIn(password, printed)
                want = claude_env_shape.read(older, settings_file.secret_shape)
                self.assertEqual(
                    confirm.claude_env_answer(0, printed, places),
                    confirm.Found(self.NAME, want, move, True),
                )
        # Its reader refuses a link, and a variable the session does not take is named only when its name looks like
        # no secret.
        elsewhere = self.dir / "elsewhere"
        older.rename(elsewhere)
        older.symlink_to(elsewhere)
        self.assertEqual(
            confirm.claude_env_answer(0, self.probe(older, login), places),
            confirm.Found(self.NAME, None, f"{older} is a link, not a file", True),
        )
        older.unlink()
        odd = {
            "ANTHROPIC_API_KEY": KEY,
            "OTHER_SETTING": "x",
            "AKIA" + "ABCDEFGHIJKLMNOP": "y",
        }
        older.write_text(json.dumps(odd), encoding="utf-8")
        self.assertEqual(
            confirm.claude_env_answer(0, self.probe(older, login), places),
            confirm.Found(
                self.NAME,
                None,
                f"{older} names variables the session does not take: OTHER_SETTING and 1 more",
                True,
            ),
        )

    def test_an_answer_that_cannot_be_read_is_refused(self):
        places = confirm.Places(self.NAME, f"{SECRETS_DIR}/claude_env", "")
        shape = claude_env_shape.to_json(self.shape({"ANTHROPIC_API_KEY": KEY}))
        token = "sk-" + "ant-" + "notreal0123456789abcdef"
        what = "its answer is no JSON object of what it found"
        unshown = "its answer says why it refused the file in words it cannot show"
        no_move = "its answer says why the move copies no key in words it cannot show"
        for code, printed, words in (
            (125, "docker: Error response from daemon: no such image", "docker: Error"),
            (137, "", "docker exited 137"),
            (0, "TLS ok", "its answer is no JSON"),
            (0, "", "its answer is no JSON"),
            (0, "[]", what),
            (0, json.dumps({"shape": shape}), what),
            (0, json.dumps({"shape": shape, "login": False}), what),
            (0, json.dumps({"shape": shape, "move": "", "login": "yes"}), what),
            (
                0,
                json.dumps(
                    {"shape": shape, "move": "", "refused": "x", "login": False}
                ),
                what,
            ),
            (
                0,
                json.dumps(
                    {
                        "shape": dict(shape, variables=["OTHER"]),
                        "move": "",
                        "login": False,
                    }
                ),
                "a shape's variables are the session's",
            ),
            (
                0,
                json.dumps(
                    {
                        "shape": dict(shape, base_url=f"https://{token}@example.com"),
                        "move": "",
                        "login": False,
                    }
                ),
                "a shape's address carries no credential",
            ),
            (0, json.dumps({"refused": "one\ntwo", "login": False}), unshown),
            (0, json.dumps({"refused": f"names {token}", "login": False}), unshown),
            (0, json.dumps({"refused": 3, "login": False}), unshown),
            (0, json.dumps({"refused": "", "login": False}), unshown),
            (
                0,
                json.dumps({"shape": shape, "move": "one\ntwo", "login": False}),
                no_move,
            ),
            (
                0,
                json.dumps({"shape": shape, "move": f"names {token}", "login": False}),
                no_move,
            ),
            (0, json.dumps({"shape": shape, "move": None, "login": False}), no_move),
        ):
            with self.subTest(code=code, printed=printed[:60]):
                with self.assertRaisesRegex(ValueError, re.escape(words)) as cm:
                    confirm.claude_env_answer(code, printed, places)
                self.assertNotIn(token, str(cm.exception))
        # docker's warnings before its answer are passed over
        self.assertIsNone(
            confirm.claude_env_answer(0, "WARNING: a platform note\nnull\n", places)
        )


class TestModels(unittest.TestCase):
    """Each model a pass uses, never one only bound and unused, is called as the container will reach it: from a
    one-off container of the image on this deployment's network; until that can run, and for a model
    whose address only the container knows, a SKIP that names where the container calls it.
    """

    PARTS = dict(
        model_bindings={
            "glm-flash": GLM,
            "idle": dict(GLM, base_url="https://idle.example.com/v1"),
        },
        **passes(ci_triage_pass={"model": "glm-flash"}),
    )

    def test_each_model_a_pass_uses_answers_or_warns(self):
        values = read(**self.PARTS).run.values
        urls = [u for _n, _p, u, _used in confirm.models_in_use(values) if u]
        self.assertEqual(urls, ["https://models.example.com/v1"])  # not idle's
        ok = confirm.model_rows(values, {urls[0]: ""})
        self.assertEqual(
            found(ok, "model glm-flash"),
            [
                (
                    Status.PASS,
                    "answers, called from a one-off container on this deployment's network",
                )
            ],
        )
        self.assertEqual(found(ok, "model idle"), [])
        # A claude model with no gateway: the session reaches Anthropic's own API, from the container.
        self.assertEqual(
            found(ok, "model opus"),
            [
                (
                    Status.SKIP,
                    "checked at step 7, from the container (it needs the claude login, step 5)",
                )
            ],
        )
        [row] = [
            c
            for c in confirm.model_rows(values, {urls[0]: "timed out"})
            if c.name == "model glm-flash"
        ]
        self.assertEqual(
            (row.status, row.found),
            (
                Status.WARN,
                "no answer at https://models.example.com/v1, called from a one-off container on this deployment's "
                "network (timed out)",
            ),
        )
        self.assertIn(
            f"{steps.command_line('doctor')} calls it again from the container",
            " ".join(render([row]).split()),
        )

    def test_a_model_waits_for_the_container_until_it_can_be_called(self):
        values = read(**self.PARTS).run.values
        doctor = steps.command_line("doctor")
        for first, there, login in (
            (
                True,
                "checked at step 7, from the container",
                " (it needs the claude login, step 5)",
            ),
            (
                False,
                f"{doctor} calls it from the container",
                " (it needs the claude login)",
            ),
        ):
            with self.subTest(first=first):
                got = confirm.model_rows(values, None, first)
                self.assertEqual(found(got, "model glm-flash"), [(Status.SKIP, there)])
                self.assertEqual(
                    found(got, "model opus"), [(Status.SKIP, there + login)]
                )

    def test_a_claude_gateway_in_the_file_is_called(self):
        """The gateway is called with no key: its PASS, on any answer (a 401 too, before the key is set), says so, and
        names what checks the key."""
        gateway = "https://llm-gateway.example.com"
        values = read(
            integrations={
                "github_cli": {"username": "op-login"},
                "claude_code": {"base_url": gateway},
            }
        ).run.values
        login = steps.command_line("login NAME", NAME="claude")
        doctor = steps.command_line("doctor")
        self.assertEqual(
            found(confirm.model_rows(values, {gateway: ""}), "model opus"),
            [
                (
                    Status.PASS,
                    f"the gateway answers at {gateway}, called from a one-off container on this deployment's "
                    f"network (no key is sent: {login} and {doctor} check it)",
                )
            ],
        )

    def test_what_the_one_off_container_printed_is_read(self):
        a, b = "https://a.example.com/v1", "https://b.example.com/v1"
        for code, printed, want in (
            (0, json.dumps({a: "", b: "timed out"}), {a: "", b: "timed out"}),
            (0, json.dumps({a: ""}), {a: "", b: "no answer"}),
            (
                0,
                "TLS ok",
                dict.fromkeys((a, b), "its answer could not be read (TLS ok)"),
            ),
            (
                125,
                "docker: Error response from daemon: network x_default not found",
                dict.fromkeys(
                    (a, b),
                    "the one-off container did not run (docker: Error response from daemon: network x_default not "
                    "found)",
                ),
            ),
            (
                125,
                "",
                dict.fromkeys(
                    (a, b), "the one-off container did not run (docker exited 125)"
                ),
            ),
        ):
            with self.subTest(code=code, printed=printed):
                self.assertEqual(confirm.model_answers(code, printed, [a, b]), want)

    def test_the_probe_takes_any_http_answer_as_an_answer(self):
        """MODEL_PROBE, run here as the one-off container runs it: an address that gives any HTTP answer (a 501 here)
        answers; one where nothing listens does not."""
        server = http.server.HTTPServer(
            (LOOPBACK, 0), http.server.BaseHTTPRequestHandler
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        up = f"http://{LOOPBACK}:{server.server_port}/v1"
        down = f"http://{LOOPBACK}:{sandbox.closed_port(self)}/v1"
        env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
        printed = subprocess.run(
            [sys.executable, "-I", "-c", confirm.MODEL_PROBE, "5", up, down],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout
        got = confirm.model_answers(0, printed, [up, down])
        self.assertEqual(got[up], "")
        self.assertTrue(got[down])


def env_text(**variables) -> str:
    """A docker/.env's text: its comment line, then NAME=value for each of VARIABLES."""
    return "# made by rkmon\n" + "".join(f"{k}={v}\n" for k, v in variables.items())


def plan(
    package=None, yours=(), outside=(), blank=(), defaults=()
) -> change_plan.ChangePlan:
    return change_plan.ChangePlan(package, yours, outside, blank, defaults)


Change = change_plan.Change
NO_RESTART, NEW_CONTAINER = change_plan.NO_RESTART, change_plan.NEW_CONTAINER
MISSING = deploy_tree.MISSING
ENV = deliver.ENV_FILE


class TestChanges(unittest.TestCase):
    """Confirm after the first start: only what changes, each change as "what runs now -> what apply makes
    it" with how it applies; docker/.env variable by variable, named by the deploy key it is made from; a package update
    first; the count, and what apply will do for it."""

    RAN = env_text(COMPOSE_PROJECT_NAME="rkmon", RKMON_PORT="18614", RKMON_CPUS="4")

    def test_the_sample(self):
        sample = plan(
            yours=(
                Change(
                    "features.review_automation.passes.first_review_pass.model",
                    "default",
                    "fast",
                    NO_RESTART,
                ),
                Change(
                    "model_bindings.fast", MISSING, {"provider": "ollama"}, NO_RESTART
                ),
                Change("host.resources.cpus", 4, 6, NEW_CONTAINER),
            ),
            outside=(
                Change(
                    ENV,
                    self.RAN.replace("RKMON_PORT=18614", "RKMON_PORT=18620"),
                    self.RAN,
                    NEW_CONTAINER,
                    ENV,
                ),
            ),
        )
        lines, count = confirm.change_lines(sample)
        self.assertEqual(
            lines,
            [
                "Your changes:",
                '  ~ features.review_automation.passes.first_review_pass.model  "default" -> "fast"',
                "        no restart: read at its next use",
                "  + model_bindings.fast",
                "        no restart: read at its next use",
                "  ~ host.resources.cpus  4 -> 6",
                "        restart: the work in flight finishes first",
                "Changed outside your file (apply puts your file's value back):",
                "  ~ features.dashboards.admin.port, as docker/.env has it  18620 -> 18614",
                "        restart",
                "Deploy keys you left blank whose default changed: none",
                "Package defaults that changed: none",
            ],
        )
        self.assertEqual(count, 4)
        self.assertEqual(
            " ".join(confirm.apply_lines(count, True, 60)),
            "4 changes. apply will: let the work in flight finish (up to 60 minutes), write the new settings (the "
            "previous ones kept), make the container again, run the doctor, and put the previous settings back on a "
            "new failure.",
        )
        for line in confirm.apply_lines(count, True, 60):
            self.assertLessEqual(len(line), WIDTH)
        self.assertEqual(
            " ".join(confirm.apply_lines(1, False, 60)),
            "1 change. apply will: write the new settings (the previous ones kept), run the doctor, and put the "
            "previous settings back on a new failure.",
        )
        # apply --no-drain (apply's step 2): the work in flight stops, and starts again later.
        self.assertEqual(
            " ".join(confirm.apply_lines(count, True, None)),
            "4 changes. apply will: stop the work in flight (it starts again later), write the new settings (the "
            "previous ones kept), make the container again, run the doctor, and put the previous settings back on a "
            "new failure.",
        )
        # On a stopped container: no drain, restart or doctor, whatever the changes.
        # The first apply over an older package makes the container again and stops it on a failure,
        # after the drain while one runs, with none over a stopped container.
        self.assertEqual(
            " ".join(confirm.apply_lines(count, True, 60, first=True)),
            "4 changes. apply will: let the work in flight finish (up to 60 minutes), write the new settings (the "
            "previous ones kept), make the container again, run the doctor, and stop the new container on a failure.",
        )
        self.assertEqual(
            " ".join(confirm.apply_lines(count, True, 60, stopped=True, first=True)),
            "4 changes. apply will: write the new settings (the previous ones kept), make the container again, run the "
            "doctor, and stop the new container on a failure.",
        )
        for restarts in (True, False):
            self.assertEqual(
                " ".join(confirm.apply_lines(count, restarts, 60, stopped=True)),
                "4 changes. apply will: write the new settings (the previous ones kept), and leave the container "
                "stopped.",
            )

    def test_the_web_host_s_install_under_a_change_its_service_is_made_from(self):
        """While the confirmed settings need the request service on the web host (web_host_install.nothing_to_install),
        a change of a key that service is made from (web_host_install.deploy_keys), in your file or a default that
        changed, says under how it applies that the service takes it when you run rkmon web-host install after apply.
        Another change says nothing more, and neither does any change while the web host needs no
        service (the monitor serves the page, or the priority box is off) or the configuration is not known.
        """
        site, cpus = (
            "features.dashboards.team.publish.site_folder",
            "host.resources.cpus",
        )
        url = "features.dashboards.team.public_url"
        changes = plan(
            yours=(
                Change(
                    site, "rocke-monitor/site", "rocke-monitor/other", NEW_CONTAINER
                ),
                Change(cpus, 4, 6, NEW_CONTAINER),
            ),
            blank=(Change(url, "http://a.example.com:8080/", MISSING, NEW_CONTAINER),),
        )

        def values(delivery: str, box: bool) -> dict:
            return registry.nest(
                {
                    "features.dashboards.team.enabled": True,
                    "team_dashboard.mode": delivery,
                    "publish.html.requests": (
                        "rocke-monitor+site-requests.jsonl" if box else ""
                    ),
                }
            )

        restart = "        restart: the work in flight finishes first"
        install = "        web host: the request service there takes it when you run ./rkmon web-host install after apply"
        with mock.patch.object(program, "name", return_value="./rkmon"):
            lines, count = confirm.change_lines(changes, values("publish", True))
            self.assertEqual(
                lines,
                [
                    "Your changes:",
                    f'  ~ {site}  "rocke-monitor/site" -> "rocke-monitor/other"',
                    restart,
                    install,
                    f"  ~ {cpus}  4 -> 6",
                    restart,
                    "Changed outside your file: none",
                    "Deploy keys you left blank whose default changed:",
                    f'  - {url}  "http://a.example.com:8080/"',
                    restart,
                    install,
                    "Package defaults that changed: none",
                ],
            )
            self.assertEqual(count, 3)
            for given in (values("serve", True), values("publish", False), None):
                with self.subTest(values=given):
                    self.assertEqual(
                        confirm.change_lines(changes, given),
                        ([x for x in lines if x != install], 3),
                    )

    def test_what_rollback_will_do(self):
        """rollback --dry-run's lines: apply's drain, the settings put back, and the code while it
        goes back too, the restart and the doctor; on a stopped container, the put-back alone.
        """
        self.assertEqual(
            " ".join(confirm.rollback_lines(True, 60)),
            "rollback will: let the work in flight finish (up to 60 minutes), put back the code and settings from "
            "before your last apply, make the container again, and run the doctor.",
        )
        for line in confirm.rollback_lines(True, 60):
            self.assertLessEqual(len(line), WIDTH)
        self.assertEqual(
            " ".join(confirm.rollback_lines(False, 60, stopped=True)),
            "rollback will: put back the settings from before your last apply, and leave the container stopped.",
        )

    def test_values_added_removed_and_whole(self):
        lines, count = confirm.change_lines(
            plan(
                yours=(
                    Change("repository.scope_labels", ["a"], ["a", "b"], NEW_CONTAINER),
                    Change(
                        "features.alerts.message_label",
                        MISSING,
                        "[rkmon]",
                        NEW_CONTAINER,
                    ),
                    Change("monitor.tick_interval_minutes", 5, MISSING, NEW_CONTAINER),
                    Change("container.name", "a", "b", ""),
                )
            )
        )
        self.assertEqual(
            lines[1:8:2],
            [
                '  ~ repository.scope_labels  ["a"] -> ["a", "b"]',
                '  + features.alerts.message_label  "[rkmon]"',
                "  - monitor.tick_interval_minutes  5",
                '  ~ container.name  "a" -> "b"',
            ],
        )
        # a key of no class restarts
        self.assertEqual(lines[8], "        restart: the work in flight finishes first")
        self.assertEqual(count, 4)

    def test_docker_env_variable_by_variable_any_other_file_whole(self):
        found = (
            self.RAN.replace("RKMON_PORT=18614", "RKMON_PORT=18620").replace(
                "RKMON_CPUS=4\n", ""
            )
            + "MY_NOTE=5\n"
        )
        layer = deliver.LAYER_NAME
        override = deliver.OVERRIDE_FILE
        lines, count = confirm.change_lines(
            plan(
                outside=(
                    Change(ENV, found, self.RAN, NEW_CONTAINER, ENV),
                    Change(
                        override, "services: {}\n", MISSING, NEW_CONTAINER, override
                    ),
                    Change(layer, "{}", "{ }", NEW_CONTAINER, layer),
                )
            )
        )
        self.assertEqual(
            lines[1:],
            [
                "Changed outside your file (apply puts your file's value back):",
                "  ~ features.dashboards.admin.port, as docker/.env has it  18620 -> 18614",
                "        restart",
                "  + host.resources.cpus, removed from docker/.env by hand  4",
                "        restart",
                "  - MY_NOTE, added to docker/.env by hand  5",
                "        restart",
                "  - docker/compose.override.yaml, added by hand",
                "        restart",
                "  ~ the settings volume's deploy.json, edited by hand",
                "        restart",
                "Deploy keys you left blank whose default changed: none",
                "Package defaults that changed: none",
            ],
        )
        self.assertEqual(count, 5)
        # A comment edited, no variable: the file whole; a file removed by hand, which apply writes again.
        commented = Change(ENV, self.RAN + "# mine\n", self.RAN, NEW_CONTAINER, ENV)
        removed = Change(ENV, MISSING, self.RAN, NEW_CONTAINER, ENV)
        for change, line in (
            (commented, "  ~ docker/.env, edited by hand"),
            (removed, "  + docker/.env, removed by hand"),
        ):
            lines, count = confirm.change_lines(plan(outside=(change,)))
            self.assertEqual((lines[2], count), (line, 1))

    def test_a_package_update_comes_first_its_defaults_for_your_information(self):
        lines, count = confirm.change_lines(
            plan(
                package=("1.2.2", "1.3.0"),
                blank=(Change("monitor.tick_interval_minutes", 5, 4, NEW_CONTAINER),),
                defaults=(Change("update.drain_timeout_min", 60, 90, NEW_CONTAINER),),
            )
        )
        self.assertEqual(
            lines,
            [
                "This package: 1.2.2 -> 1.3.0",
                "        restart: the work in flight finishes first",
                "Your changes: none",
                "Deploy keys you left blank whose default changed:",
                "  ~ monitor.tick_interval_minutes  5 -> 4",
                "        restart: the work in flight finishes first",
                "Package defaults that changed, for your information (no key of your file sets them):",
                "  ~ update.drain_timeout_min  60 -> 90",
            ],
        )
        self.assertEqual(
            count, 2
        )  # the package line and the default: not the package default

    def test_other_code_under_the_same_version_is_named(self):
        """A package update to a later commit of updates.branch: the package line says that the code is other."""
        lines, count = confirm.change_lines(plan(package=("1.3.0", "1.3.0")))
        self.assertEqual(
            (lines[:2], count),
            (
                [
                    "This package: 1.3.0 -> 1.3.0 (other code under the same version)",
                    "        restart: the work in flight finishes first",
                ],
                1,
            ),
        )

    def test_no_change(self):
        self.assertEqual(
            confirm.change_lines(plan()),
            (["No change: the deployment has these settings already."], 0),
        )

    def test_only_the_models_a_change_brings_are_called(self):
        ran = read(**TestModels.PARTS).run.values
        values = read(
            model_bindings={
                **TestModels.PARTS["model_bindings"],
                "fast": dict(GLM, base_url="https://fast.example.com/v1"),
            },
            **passes(
                ci_triage_pass={"model": "glm-flash"},
                first_review_pass={"model": "fast"},
            ),
        ).run.values
        added = confirm.models_added(values, ran)
        self.assertEqual(added, {"fast"})
        got = confirm.model_rows(
            values, {"https://fast.example.com/v1": ""}, False, added
        )
        self.assertEqual(
            [(c.name, c.status) for c in got], [("model fast", Status.PASS)]
        )
        self.assertEqual(confirm.models_added(ran, ran), set())

    def test_a_login_a_change_needs_that_is_not_set_yet(self):
        values = read(features={"review_automation": {"mode": "live"}}).run.values
        sets = f"{steps.command_line('login')} sets it before apply"
        rows = [
            {"name": "github_write", "state": "NOT NEEDED"},
            {"name": "claude", "state": "SET"},
        ]
        self.assertEqual(
            [(c.name, c.status, c.found) for c in confirm.login_skips(values, rows)],
            [("github_write", Status.SKIP, f"not set yet; {sets}")],
        )
        self.assertEqual(confirm.login_skips(values, None), [])
        self.assertEqual(
            confirm.login_skips(values, [{"name": "github_write", "state": "SET"}]), []
        )
        # not needed by the new settings either: no row
        prototype = read().run.values
        self.assertEqual(confirm.login_skips(prototype, rows), [])
        self.assertEqual(
            confirm.checks(read().run, read().doc, [], logins=["mine"])[-1], "mine"
        )


class TestConfiguration(unittest.TestCase):
    def test_the_configuration_the_container_runs_is_the_defaults_the_file_and_the_layout(
        self,
    ):
        """deployment.configuration: the package's defaults, then the file, then what the container's layout derives
        (its working folder), as the container's load makes it of the settings the confirm delivers.
        """
        r = read(features={"review_automation": {"parallel_reviews": 2}})
        e = d.configuration(r.cfg)
        self.assertEqual(e["review"]["pipes"], 2)
        self.assertEqual(e["checks"]["limit"], 1)  # the package's default
        self.assertEqual(e["features"]["merge_queue"]["mode"], "prototype")
        self.assertEqual(e["paths"]["working_dir"], container_layout.WORK_DIR)
        self.assertEqual(e, r.run.values)

    def test_the_dashboards_are_where_docker_publishes_them(self):
        """The summary's dashboards are at the host addresses and ports docker/.env's variables give, the plan's."""
        r = read()
        moved = r.plan._replace(variables=dict(r.plan.variables, RKMON_PORT="18999"))
        text = confirm.render(moved, r.run.values)
        host = moved.variables["RKMON_ADMIN_HOST"]
        self.assertRegex(
            text, rf"(?m)^  admin\.port: +18614 \({host}:18999 on this host\)$"
        )


class TestSkillsFolder(unittest.TestCase):
    def test_check_18_s_pass_row_shows_with_the_problems(self):
        """Check 18: its PASS row names the team's skills the confirm delivers, so the confirm's checks
        show it among the problems of the rows docker/rkmon's confirm_checks gives; another PASS row there does not
        show."""
        r = read()
        row = Check(
            coherence.SKILLS,
            Status.PASS,
            "the team's skills in ~/team replace the package's review_book.md",
        )
        other = Check("the package folder", Status.PASS, "nothing stray in it")
        got = confirm.checks(r.run, r.doc, [*r.rows, other, row])
        self.assertIn(row, got)
        self.assertNotIn(other, got)
        self.assertIn(
            "  PASS  skills folder: the team's skills in ~/team replace the package's review_book.md",
            render(got),
        )


if __name__ == "__main__":
    unittest.main()
