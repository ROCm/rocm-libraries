# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The carry from the deploy settings file (file_types.DEPLOY) into today's config.json (rkmon_lib/deploy_tree.py): a
file in layout 1, written in layout 2 by rkmon migrate, gives the configuration the layout 1 file gave, key by key, but
for the listed differences, each in the comparisons its row names and with the values it records
(tests/data/deploy_tree_carry.json, tests/data/deploy_tree_differences.json), and today's deploy checks take it; every
key of the schema carried, or named with the reason it is not; the features' modes, the passes, the bindings, the keys
layout 2 merges into the file's and the rules the schema cannot say; an image's reference split into its name and its
tag; the file read, or refused (a file in the layout before layout 1 in migrate's words); the names messages give
configuration keys; a section of secrets' values refused; the gateway a file names, over the Claude login's.
"""

import copy
import functools
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "worker"))
sys.dont_write_bytecode = True

import home_repository  # noqa: E402
import llm_client  # noqa: E402
from rkmon_lib import (  # noqa: E402
    bindings,
    breakage,
    checks,
    coherence,
    config,
    deploy_tree,
    deployment,
    features,
    file_types,
    from_layout1,
    image,
    migrate,
    orchestrator,
    registry,
    settings_file,
)

DATA = ROOT / "tests" / "data"
CARRY = json.loads((DATA / "deploy_tree_carry.json").read_text(encoding="utf-8"))
DIFFERENCES = json.loads(
    (DATA / "deploy_tree_differences.json").read_text(encoding="utf-8")
)["differences"]
SAMPLES = CARRY["samples"]
COMPARISONS = ("defaults", *SAMPLES)
HEADER = {"kind": file_types.DEPLOY.kind, "schema_version": file_types.DEPLOY.version}
VERSION = "{version}"  # in a value a row records: this package's version (image.package_version)


def setUpModule():
    # The samples' repository stands for the home repository: its repository defaults apply.
    (repository,) = {
        (s["layout_1"]["repository"]["name"], s["layout_1"]["repository"]["path"])
        for s in SAMPLES.values()
    }
    home_repository.stand_in(*repository)


def leaves(d: dict, prefix: str = "") -> dict:
    """Each leaf of nested mappings by its dotted key; an empty mapping is a leaf."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(leaves(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def merged(*layers: dict) -> dict:
    """The layers laid one over the other, mappings merged key by key."""
    out = {}
    for layer in layers:
        for k, v in layer.items():
            both = isinstance(v, dict) and isinstance(out.get(k), dict)
            out[k] = merged(out[k], v) if both else copy.deepcopy(v)
    return out


def migrated(layout_1: dict) -> dict:
    """The values of a deploy settings file in layout 1, LAYOUT_1, once rkmon migrate has written the file in layout 2
    (migrate.run, whose step from layout 1 is from_layout1.deploy), read back as the deploy reads them.
    """
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "rkmon-deploy.yaml"
        text = json.dumps({**HEADER, "schema_version": 1, **layout_1})
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
        out = io.StringIO()
        argv = ("rkmon", "migrate", "--settings", str(path))
        if migrate.run(argv, str(path), str(path), out=out) != 0:
            raise AssertionError(
                f"migrate does not write the sample:\n{out.getvalue()}"
            )
        return settings_file.read(path, file_types.DEPLOY).values


@functools.lru_cache(maxsize=None)
def _layout_2(name: str) -> str:
    sample = SAMPLES[name]
    after = sample.get("set_after_migrate", {})
    return json.dumps(merged(migrated(sample["layout_1"]), after))


def layout_2(name: str) -> dict:
    """The sample NAME in layout 2: its layout 1 file, migrated, with the values migrate cannot write set on it
    (set_after_migrate); a copy of its own for each caller."""
    return json.loads(_layout_2(name))


def layout_1_defaults() -> dict:
    """The defaults the configuration had in layout 1: today's (config.load_defaults), each key the deploy settings
    file feeds at layout 1's package default (from_layout1.DEFAULTS_FILE), or left out where layout 1 had none. Both
    sides of a comparison start from them, so only what the file carries differs."""
    base = config.load_defaults()
    layout_1 = json.loads(from_layout1.DEFAULTS_FILE.read_text(encoding="utf-8"))
    for name in deploy_tree.rules():
        value = registry.get(layout_1, name)
        if value is not registry.MISSING:
            registry.put(base, name, value)
            continue
        *path, last = name.split(".")
        holder = registry.get(base, ".".join(path)) if path else base
        if isinstance(holder, dict):
            holder.pop(last, None)
    return base


def comparisons() -> dict:
    """Name -> (the configuration layout 1 gave, the one layout 2 gives): at the defaults, then for each sample, each
    over layout_1_defaults; layout 2's defaults lay the file's own over them (deploy_tree.internal_defaults).
    """
    base = layout_1_defaults()
    defaults = merged(base, deploy_tree.internal_defaults())
    old_defaults = merged(base, CARRY["layout_1_config"]["defaults"])
    out = {"defaults": (old_defaults, defaults)}
    for name in SAMPLES:
        out[name] = (
            merged(old_defaults, CARRY["layout_1_config"][name]),
            merged(defaults, deploy_tree.to_config(layout_2(name))),
        )
    return out


def differences() -> dict:
    """(comparison, configuration key) -> (its value in the configuration layout 1 gave, in the one layout 2 gives),
    MISSING where it has none, for each key whose two values differ."""
    out = {}
    for name, (old, new) in comparisons().items():
        a, b = leaves(old), leaves(new)
        for key in sorted(set(a) | set(b)):
            pair = (a.get(key, deploy_tree.MISSING), b.get(key, deploy_tree.MISSING))
            if pair[0] != pair[1]:
                out[name, key] = pair
    return out


def listed() -> list:
    """The differences the rows record, in differences()'s form, one entry for each comparison a row names and each key
    it records there: a key the row's old or new values leave out has no value there."""
    version = image.package_version()

    def value(values: dict, key: str):
        v = values.get(key, deploy_tree.MISSING)
        return v.replace(VERSION, version) if isinstance(v, str) else v

    return [
        ((name, key), (value(pair["old"], key), value(pair["new"], key)))
        for row in DIFFERENCES
        for name, pair in row["comparisons"].items()
        for key in sorted(set(pair["old"]) | set(pair["new"]))
    ]


class TestTheCarry(unittest.TestCase):
    """Each sample's layout 1 file, migrated, gives today's config.json the same values the layout 1 file gave, key by
    key, but for the listed differences: the rows of tests/data/deploy_tree_differences.json, each in the comparisons
    it names, with the old and the new values it records there, and why. Today's deploy checks take each sample. An
    address the web host's request service does not serve migrates, and the file's check fails it.
    """

    def test_every_difference_is_listed_and_every_row_is_one(self):
        found, rows = differences(), listed()
        where = [entry for entry, _values in rows]
        self.assertEqual(
            sorted({e for e in where if where.count(e) > 1}),
            [],
            "a key two rows record in one comparison",
        )
        rows = dict(rows)
        self.assertEqual(
            {e: v for e, v in found.items() if rows.get(e) != v},
            {},
            "a difference no row records, or records with other values",
        )
        self.assertEqual(
            {e: v for e, v in rows.items() if found.get(e) != v},
            {},
            "a difference a row records that the comparison does not have",
        )

    def test_each_row_says_where_what_changes_and_why(self):
        for row in DIFFERENCES:
            with self.subTest(why=row.get("why")):
                self.assertEqual(sorted(row), ["comparisons", "files", "why"])
                self.assertTrue(row["why"].strip() and row["files"].strip())
                self.assertTrue(row["comparisons"])
                for name, pair in row["comparisons"].items():
                    self.assertIn(name, COMPARISONS)
                    self.assertEqual(sorted(pair), ["new", "old"])
                    self.assertTrue(set(pair["old"]) | set(pair["new"]))

    def test_each_sample_migrates_to_a_file_the_checks_take(self):
        """The step migrate runs from layout 1 is from_layout1.deploy; each file it writes passes the file's checks,
        with the values migrate cannot write set on it."""
        self.assertIs(file_types.DEPLOY.upgrades[1], from_layout1.deploy)
        for name in SAMPLES:
            with self.subTest(sample=name):
                doc = settings_file.Document(
                    file_types.DEPLOY, file_types.DEPLOY.version, layout_2(name)
                )
                fails = [
                    r
                    for r in settings_file.validate(doc)
                    if r.status == checks.Status.FAIL
                ]
                self.assertEqual(fails, [])

    def test_an_address_the_request_service_does_not_serve_migrates_and_fails_its_check(
        self,
    ):
        """The full sample, priority requests on and "publish", with an address the web host's request service does
        not serve the page at: migrate writes the file, and its check, as rkmon check reads
        it, fails the team dashboard: an http address with a port and a folder, its fix the root address; an https
        one, its fix what to give."""
        sample = SAMPLES["full"]
        for url, found, fix in (
            (
                "http://web.example.com:8080/dash/",
                'has the path "/dash/", and the web host\'s request service serves the page at the root of its own '
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
                layout_1 = copy.deepcopy(sample["layout_1"])
                layout_1["advanced"]["publish"]["html"]["url"] = url
                values = merged(migrated(layout_1), sample.get("set_after_migrate", {}))
                doc = settings_file.Document(
                    file_types.DEPLOY, file_types.DEPLOY.version, values
                )
                key = "features.dashboards.team.public_url"
                self.assertEqual(
                    [
                        (r.name, r.found, str(r.fix))
                        for r in settings_file.validate(doc)
                        if r.status == checks.Status.FAIL
                    ],
                    [
                        (
                            "team dashboard",
                            f"{key} {found}",
                            f"set {key} in your deploy settings to {fix}",
                        )
                    ],
                )

    def test_today_s_deploy_checks_take_each_carried_sample(self):
        """deployment.settings_checks on each sample's configuration: no FAIL. The full sample's team page, published
        to the web host, has its place there from the site folder the file sets."""
        for name in SAMPLES:
            with self.subTest(sample=name):
                cfg = deploy_tree.to_config(layout_2(name))
                self.assertEqual(
                    [
                        (c.name, c.found, str(c.fix))
                        for c in deployment.settings_checks(cfg)
                        if c.status == "FAIL"
                    ],
                    [],
                )


class TestEveryKeyIsCarried(unittest.TestCase):
    def test_every_key_feeds_the_configuration_or_says_why_not(self):
        """Each key of the schema feeds a configuration key (its x-internal), is read by a rule, or is NOT_CARRIED with
        its reason; a section answers by its keys."""
        keys = [
            path
            for path, _node, _required in settings_file.schema_keys(
                file_types.DEPLOY.load()
            )
        ]
        fed = {path for path, _internal in deploy_tree.feeds()}
        read = {path for rule in deploy_tree.rules().values() for path in rule.reads}
        for path in keys:
            if path in settings_file.HEADER_KEYS or any(
                k.startswith(path + ".") for k in keys
            ):
                continue
            with self.subTest(key=path):
                self.assertTrue(
                    path in fed or path in read or path in deploy_tree.NOT_CARRIED
                )
        for path, why in deploy_tree.NOT_CARRIED.items():
            with self.subTest(not_carried=path):
                self.assertIn(path, keys)
                self.assertNotIn(path, read)
                self.assertTrue(why.strip())

    def test_spend_carries_its_prices_while_it_is_on(self):
        full = layout_2("full")
        self.assertEqual(
            deploy_tree.to_config(full)["prices"],
            {
                "claude-sonnet-5-5": {
                    "in": 3,
                    "out": 15,
                    "cache_read": 0.3,
                    "cache_write_5m": 3.75,
                    "cache_write_1h": 6,
                }
            },
        )
        off = merged(
            full, {"features": {"review_automation": {"spend": {"enabled": False}}}}
        )
        self.assertEqual(deploy_tree.to_config(off)["prices"], {})

    def test_the_drift_notices_switch_reaches_their_gate(self):
        """With the merge queue live, the drift notices go out while their own switch is on."""
        for on in (True, False):
            with self.subTest(drift_notices=on):
                values = layout_2("minimal")
                deploy_tree.put(values, "features.merge_queue.enabled", True)
                deploy_tree.put(values, "features.merge_queue.mode", "live")
                deploy_tree.put(
                    values, "features.merge_queue.drift_notices.enabled", on
                )
                cfg = deploy_tree.to_config(values)
                self.assertIs(
                    config.lookup(cfg, "features.merge_queue.drift_notices"), on
                )
                self.assertIs(features.on(cfg, "drift-notices"), on)

    def test_the_github_request_cap_reaches_the_counter(self):
        """The cap on GitHub requests is the configuration's gh.max_requests_per_hour, which each tick gives the counter
        (rkmon_lib/gh_cap.py); 0 by default, no cap."""
        values = layout_2("minimal")
        self.assertEqual(
            registry.number(deploy_tree.to_config(values), "gh.max_requests_per_hour"),
            0,
        )
        deploy_tree.put(values, "integrations.github_cli.max_requests_per_hour", 120)
        cfg = deploy_tree.to_config(values)
        self.assertEqual(config.lookup(cfg, "gh.max_requests_per_hour"), 120)
        self.assertEqual(registry.number(cfg, "gh.max_requests_per_hour"), 120)


class TestTheRulesReadTheSchema(unittest.TestCase):
    """A rule reads the keys that feed its configuration key, as the schema's x-internals say (deploy_tree._fed), or
    those that feed the configuration keys it names; rules() refuses a rule that does not read every key that feeds
    its configuration key, and a joined rule that reads no key or does not take the keys it reads.
    """

    FIRST = "features.review_automation.passes.first_review_pass"

    def setUp(self):
        deploy_tree.rules.cache_clear()
        self.addCleanup(deploy_tree.rules.cache_clear)

    def refusal(self, fed: dict) -> str:
        """rules()'s refusal while the schema's x-internals say FED."""
        with mock.patch.object(deploy_tree, "_fed", lambda: fed):
            with self.assertRaises(ValueError) as caught:
                deploy_tree.rules()
        return str(caught.exception)

    def also(self, internal: str, path: str) -> dict:
        """_fed(), with PATH feeding INTERNAL too."""
        fed = dict(deploy_tree._fed())
        fed[internal] = (*fed.get(internal, ()), path)
        return fed

    def test_each_rule_reads_the_keys_that_feed_it(self):
        fed, rules = deploy_tree._fed(), deploy_tree.rules()
        for internal, paths in fed.items():
            if "*" not in internal:
                with self.subTest(internal=internal):
                    self.assertLessEqual(set(paths), set(rules[internal].reads))
        self.assertEqual(
            rules["atlassian.enabled"].reads, fed["features.ticket_creation.enabled"]
        )
        self.assertEqual(
            rules["worker.think"].reads, (f"{self.FIRST}.model", f"{self.FIRST}.effort")
        )

    def test_a_rule_that_does_not_read_a_key_that_feeds_it_is_refused(self):
        other = "features.review_automation.passes.code_quality_pass.effort"
        self.assertEqual(
            self.refusal(self.also("effort.verify", other)),
            f"effort.verify is fed by {other}, which its rule does not read",
        )
        # a joined rule takes each key that feeds it: a third label key is refused, not dropped
        guard = "features.merge_queue.develop_guard"
        self.assertEqual(
            self.refusal(
                self.also("incident_issue.labels", f"{guard}.stability_label")
            ),
            "the rule for incident_issue.labels does not take the keys it reads: "
            f"{guard}.ci_red_label, {guard}.ci_red_extra_labels, {guard}.stability_label",
        )

    def test_a_joined_rule_that_reads_no_key_is_refused(self):
        fed = {k: v for k, v in deploy_tree._fed().items() if k != "update.repo"}
        self.assertEqual(
            self.refusal(fed),
            "the rule for update.repo reads no key: nothing feeds update.repo",
        )


class TestTheModes(unittest.TestCase):
    """Alerts, the team dashboard and ticket creation each carry their mode as it is, as review automation and the
    merge queue do, and each is on while a switch that feeds it is on, whatever its mode: in prototype it
    does its work, and what it would send stops at the one outward gate. Alerts are on while either kind is, and each
    kind's own switch is carried as it is: its sender reads both.
    """

    FEATURES = {
        "features.alerts.enabled": (
            "features.alerts.mode",
            (
                "features.alerts.develop_breaks.enabled",
                "features.alerts.needs_you.enabled",
            ),
        ),
        "features.dashboards.team.enabled": (
            "features.dashboards.team.mode",
            ("features.dashboards.team.enabled",),
        ),
        "features.ticket_creation.enabled": (
            "features.ticket_creation.mode",
            (
                "features.ticket_creation.follow_up_tickets.enabled",
                "features.ticket_creation.found_issue_tickets.enabled",
            ),
        ),
    }

    def test_on_while_a_switch_is_on_in_either_mode_which_is_carried(self):
        for internal, (mode, switches) in self.FEATURES.items():
            for switch in switches:
                for on, mode_value, want in (
                    (True, "live", True),
                    (True, "prototype", True),
                    (False, "live", False),
                ):
                    with self.subTest(switch=switch, on=on, mode=mode_value):
                        values = layout_2("minimal")
                        deploy_tree.put(values, switch, on)
                        deploy_tree.put(values, mode, mode_value)
                        cfg = deploy_tree.to_config(values)
                        self.assertIs(config.lookup(cfg, internal), want)
                        self.assertEqual(config.lookup(cfg, mode), mode_value)

    def test_each_kind_of_alert_goes_out_while_its_own_switch_is_on(self):
        """Production's file: the develop-break alerts on and the Needs-you alerts off, in mode live."""
        values = layout_2("minimal")
        deploy_tree.put(values, "features.alerts.mode", "live")
        deploy_tree.put(values, "features.alerts.develop_breaks.enabled", True)
        deploy_tree.put(values, "features.alerts.needs_you.enabled", False)
        cfg = deploy_tree.to_config(values)
        self.assertEqual(
            [
                config.lookup(cfg, f"features.alerts.{k}")
                for k in ("enabled", "develop_breaks", "needs_you")
            ],
            [True, True, False],
        )
        self.assertEqual(
            [features.on(cfg, n) for n in ("chat-alerts", "needs-you-alerts")],
            [True, False],
        )


class TestThePasses(unittest.TestCase):
    def test_a_pass_s_model_and_effort(self):
        values = merged(
            layout_2("minimal"),
            {
                "features": {
                    "review_automation": {
                        "orchestrator": {"model": "big", "effort": "max"},
                        "passes": {
                            "verification_pass": {"model": "sonnet", "effort": "xhigh"},
                            "ci_triage_pass": {"model": "haiku", "effort": "default"},
                        },
                    }
                },
                "model_bindings": {
                    "big": {"provider": "claude", "model_id": "claude-opus-5-5[1m]"}
                },
            },
        )
        cfg = deploy_tree.to_config(values)
        # A pass Claude Code runs: a claude binding by its model_id, a built-in model by its name.
        self.assertEqual(cfg["orchestrator"]["model"], "claude-opus-5-5[1m]")
        self.assertEqual(
            (
                cfg["models"]["audit"],
                cfg["effort"]["verify"],
                cfg["effort"]["orchestrator"],
            ),
            ("sonnet", "xhigh", "max"),
        )
        # An optional pass: its model as it is, "default" when left out; the effort "default" is "".
        self.assertEqual(
            (cfg["models"]["ci_triage"], cfg["effort"]["ci_triage"]), ("haiku", "")
        )
        self.assertEqual(cfg["models"]["notice_draft"], "default")
        # A built-in model an optional pass names has a claude binding of its own; one only Claude Code's passes
        # name has none.
        self.assertEqual(
            cfg["model_bindings"]["haiku"], {"provider": "claude", "model": "haiku"}
        )
        self.assertNotIn("sonnet", cfg["model_bindings"])


class TestTheBindings(unittest.TestCase):
    def test_how_a_binding_s_key_is_sent(self):
        values = merged(
            layout_2("minimal"),
            {
                "model_bindings": {
                    "gw": {
                        "provider": "openai_compatible",
                        "model_id": "m-1",
                        "base_url": "https://gw.example.invalid/v1",
                        "api_key_secret_name": "gw_api_key",
                        "api_key_sent_as": "header_and_bearer",
                        "api_key_header": "X-Api-Key",
                    },
                    "plain": {
                        "provider": "openai_compatible",
                        "model_id": "m-2",
                        "base_url": "https://llm.example.invalid/v1",
                    },
                }
            },
        )
        b = deploy_tree.to_config(values)["model_bindings"]
        self.assertEqual(
            {
                k: b["gw"].get(k)
                for k in ("provider", "auth_mode", "auth_header", "auth_scheme")
            },
            {
                "provider": "openai-compatible",
                "auth_mode": "both",
                "auth_header": "X-Api-Key",
                "auth_scheme": "",
            },
        )
        self.assertEqual(b["plain"]["auth_mode"], "bearer")
        self.assertNotIn(
            "auth_header", b["plain"]
        )  # config.json's header: Authorization

    def test_the_binding_named_worker_is_today_s_local_worker(self):
        values = merged(
            layout_2("minimal"),
            {
                "model_bindings": {
                    "worker": {
                        "provider": "openai_compatible",
                        "model_id": "m-3",
                        "base_url": "https://llm.example.invalid/v1",
                        "api_key_secret_name": "worker_api_key",
                        "limits": {"max_parallel_requests": 3},
                    }
                }
            },
        )
        cfg = deploy_tree.to_config(values)
        self.assertEqual(
            {
                k: cfg["worker"].get(k)
                for k in (
                    "backend",
                    "model",
                    "url",
                    "api_key_secret",
                    "max_parallel",
                    "auth_mode",
                )
            },
            {
                "backend": "openai",
                "model": "m-3",
                "url": "https://llm.example.invalid/v1",
                "api_key_secret": "worker_api_key",
                "max_parallel": 3,
                "auth_mode": "bearer",
            },
        )
        self.assertNotIn("worker", cfg["model_bindings"])
        self.assertNotIn("worker", deploy_tree.to_config(layout_2("minimal")))

    def test_the_worker_s_keys_by_their_binding_names(self):
        """worker_fields: each key the binding named worker carries, by its name in config.json's bindings, with its
        name under config.json's worker (tools/translate_production.py reads production's worker back so).
        """
        self.assertEqual(
            deploy_tree.worker_fields(),
            {
                "provider": "backend",
                "model": "model",
                "url": "url",
                "api_key_secret": "api_key_secret",
                "auth_mode": "auth_mode",
                "auth_header": "auth_header",
                "auth_scheme": "auth_scheme",
                "rpm": "rpm",
                "tpm": "tpm",
                "concurrency": "max_parallel",
                "stall_s": "stall_s",
            },
        )


class TestTheMergedKeys(unittest.TestCase):
    """The configuration keys that layout 2 merges into a key of the file and that today's code still reads: each
    takes that key's value (the key's x-internal names it after the key's own), and the local worker's effort is the
    first review pass's while the pass runs on the worker."""

    def carried(self, values: dict) -> dict:
        return deploy_tree.to_config(merged(layout_2("minimal"), values))

    def test_the_jira_site_is_the_atlassian_cli_s(self):
        cfg = self.carried(
            {
                "integrations": {
                    "atlassian_cli": {"jira_url": "https://jira.example.com"}
                }
            }
        )
        self.assertEqual(
            (config.lookup(cfg, "jira.url"), config.lookup(cfg, "atlassian.site")),
            ("https://jira.example.com", "https://jira.example.com"),
        )

    def test_the_team_page_s_address_is_its_public_url(self):
        url = "https://web.example.com/team/"
        cfg = self.carried({"features": {"dashboards": {"team": {"public_url": url}}}})
        self.assertEqual(
            (
                config.lookup(cfg, "team_dashboard.url"),
                config.lookup(cfg, "publish.html.url"),
            ),
            (url, url),
        )

    def test_the_tick_loop_s_interval_is_the_monitor_s(self):
        cfg = self.carried({"monitor": {"tick_interval_minutes": 7}})
        self.assertEqual(
            (
                config.lookup(cfg, "tick_interval_min"),
                config.lookup(cfg, "claude.tick_min"),
            ),
            (7, 7),
        )

    def test_the_worker_s_effort_is_the_first_review_pass_s_on_the_worker(self):
        """worker.think: the first review pass's effort while the pass runs on the binding named worker and the
        effort is one the worker sends; else none, and the package's own stands."""
        for model, effort, want in (
            ("worker", "low", "low"),
            ("worker", "medium", "medium"),
            ("worker", "high", "high"),
            ("worker", "default", deploy_tree.MISSING),
            ("worker", "xhigh", deploy_tree.MISSING),
            ("worker", "max", deploy_tree.MISSING),
            ("sonnet", "high", deploy_tree.MISSING),
        ):
            with self.subTest(model=model, effort=effort):
                first = {"model": model, "effort": effort}
                cfg = self.carried(
                    {
                        "features": {
                            "review_automation": {
                                "passes": {"first_review_pass": first}
                            }
                        }
                    }
                )
                self.assertEqual(deploy_tree.get(cfg, "worker.think"), want)
        # the package's own, under any configuration (config.worker_defaults): unset, so no effort is sent
        self.assertIsNone(config.worker_settings(config.load_defaults())["think"])

    def test_claude_s_base_effort_and_the_tick_log_s_size_stay_the_package_s(self):
        """claude.effort is the base effort of the Claude roles the file names no pass for; loop_log_max_mb, the tick
        log's size, merges into a key of the package, not of the file."""
        for key in ("claude.effort", "loop_log_max_mb"):
            with self.subTest(key=key):
                self.assertNotIn(key, deploy_tree.held())
                self.assertIsNone(
                    config.lookup(deploy_tree.to_config(layout_2("full")), key)
                )


class TestTheJoinedRules(unittest.TestCase):
    """The rules the schema cannot say (deploy_tree._JOINED) whose configuration keys a difference row names in the
    carry: the image, the labels of an issue about a break that turns develop's CI red, how the auto-merge guard treats
    your own pull requests, whether Atlassian is on, and the priority requests file."""

    def carried(self, values: dict) -> dict:
        return deploy_tree.to_config(merged(layout_2("minimal"), values))

    def test_the_image_is_its_name_and_its_tag(self):
        """NAME:TAG, the name rocke-monitor by default; a blank tag is container.name's (deployment.image_ref), so
        it gives NAME:, and no value at all with the default name."""
        for section, want in (
            ({"name": "acme-rkmon", "tag": "2.0.1"}, "acme-rkmon:2.0.1"),
            ({"tag": "2.0.1"}, "rocke-monitor:2.0.1"),
            ({"name": "acme-rkmon", "tag": ""}, "acme-rkmon:"),
            ({"tag": ""}, ""),
        ):
            with self.subTest(image=section):
                values = layout_2("minimal")
                values["image"] = section
                cfg = deploy_tree.to_config(values)
                self.assertEqual(config.lookup(cfg, "deployment.image"), want)

    def test_a_ci_red_break_s_labels_are_its_label_then_its_extra_labels(self):
        """incident_issue.labels: ci_red_label, then ci_red_extra_labels; a blank ci_red_label keeps its place while
        there are extra labels, so the develop guard reads no extra label as the CI-red one. The stability label stays
        a key of its own."""
        for label, extra, want in (
            ("ci-red", ["develop-break", "p1"], ["ci-red", "develop-break", "p1"]),
            ("", ["develop-break"], ["", "develop-break"]),
            ("ci-red", [], ["ci-red"]),
            ("", [], []),
        ):
            with self.subTest(label=label, extra=extra):
                guard = {
                    "ci_red_label": label,
                    "ci_red_extra_labels": extra,
                    "stability_label": "stability",
                }
                cfg = self.carried(
                    {"features": {"merge_queue": {"develop_guard": guard}}}
                )
                self.assertEqual(
                    (
                        config.lookup(cfg, "incident_issue.labels"),
                        config.lookup(cfg, "incident_issue.stability_label"),
                    ),
                    (want, "stability"),
                )
                self.assertEqual(
                    (breakage.labels(cfg)[breakage.RED], breakage.extra_labels(cfg)),
                    (label, extra),
                )

    def test_the_auto_merge_guard_on_your_own_pull_requests(self):
        """automerge_guard_own: true for same_as_others; alert_only, the default, gives false."""
        for own, want in (
            ({"on_your_pull_requests": "same_as_others"}, True),
            ({"on_your_pull_requests": "alert_only"}, False),
            ({}, False),
        ):
            with self.subTest(own=own):
                cfg = self.carried(
                    {"features": {"merge_queue": {"auto_merge_guard": own}}}
                )
                self.assertIs(config.lookup(cfg, "automerge_guard_own"), want)

    def test_atlassian_is_on_while_a_type_of_ticket_is(self):
        """atlassian.enabled: on while either type of ticket is on, with or without JIRA's address; the address alone
        leaves it off."""
        site = {
            "integrations": {"atlassian_cli": {"jira_url": "https://jira.example.com"}}
        }
        for address, follow_up, found_issue, want in (
            (site, False, False, False),
            ({}, True, False, True),
            ({}, False, True, True),
            (site, True, True, True),
        ):
            with self.subTest(
                address=bool(address), follow_up=follow_up, found_issue=found_issue
            ):
                tickets = {
                    "follow_up_tickets": {"enabled": follow_up},
                    "found_issue_tickets": {"enabled": found_issue},
                }
                cfg = self.carried(
                    merged(address, {"features": {"ticket_creation": tickets}})
                )
                self.assertIs(config.lookup(cfg, "atlassian.enabled"), want)

    def test_the_requests_file_lies_at_the_top_of_the_home_while_priority_requests_are_on(
        self,
    ):
        """publish.html.requests: the requests file the site folder gives, at the top of the login's home
        (derived.site_requests), while priority requests are on; none, "", while they are off or no site folder is
        set."""
        for on, site, want in (
            (True, "public_html/my-team", "public_html+my-team-requests.jsonl"),
            (True, "public_html/my-team/", "public_html+my-team-requests.jsonl"),
            (True, "public_html", "public_html-requests.jsonl"),
            (False, "public_html/my-team", ""),
            (True, "", ""),
        ):
            with self.subTest(on=on, site=site):
                team = {
                    "priority_requests": {"enabled": on},
                    "publish": {"site_folder": site},
                }
                cfg = self.carried({"features": {"dashboards": {"team": team}}})
                self.assertEqual(config.lookup(cfg, "publish.html.requests"), want)


class TestTheImagesParts(unittest.TestCase):
    """deploy_tree.image_parts: an image's reference, <name>:<tag>, split into its name and its tag."""

    def test_the_tag_follows_the_last_colon_that_no_slash_follows(self):
        """The last ":" splits, unless a "/" follows it: a ":" before the last "/" is a registry's port, so
        host:5000/name has no tag; ("", "") with no ":" at all."""
        for ref, want in (
            ("rocke-monitor:v1", ("rocke-monitor", "v1")),
            (" team/rkmon: ", ("team/rkmon", "")),
            (":tag-x", ("", "tag-x")),
            ("host:5000/name", ("host:5000/name", "")),
            ("host:5000/name:tag", ("host:5000/name", "tag")),
            ("rkmon", ("", "")),
            ("", ("", "")),
            (None, ("", "")),
        ):
            with self.subTest(ref=ref):
                self.assertEqual(deploy_tree.image_parts(ref), want)

    def test_the_image_section_s_reference_splits_back_into_its_name_and_its_tag(self):
        """deployment.image, joined from image.name and image.tag, splits into them again, a port in the name or not
        and a blank tag or not."""
        for name in ("acme-rkmon", "host:5000/team/rkmon"):
            for tag in ("2.0.1", ""):
                with self.subTest(name=name, tag=tag):
                    values = layout_2("minimal")
                    values["image"] = {"name": name, "tag": tag}
                    ref = config.lookup(
                        deploy_tree.to_config(values), "deployment.image"
                    )
                    self.assertEqual(deploy_tree.image_parts(ref), (name, tag))


class TestRead(unittest.TestCase):
    """deploy_tree.read: a file's configuration and its FAIL and WARN rows; a file the reader refuses raises
    settings_file.Refused, with its reason, a file in the layout before layout 1 with migrate's row; one that cannot
    be opened, its OSError."""

    def read(self, text: str):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "deploy.yaml"
            path.write_text(text, encoding="utf-8")
            return deploy_tree.read(path)

    def refusal(self, text: str) -> str:
        with self.assertRaises(settings_file.Refused) as caught:
            self.read(text)
        return caught.exception.reason

    def test_a_file_gives_its_configuration_and_its_rows(self):
        cfg, rows = self.read(json.dumps({**HEADER, **layout_2("minimal")}))
        self.assertEqual(rows, [])
        self.assertEqual(cfg, deploy_tree.to_config(layout_2("minimal")))
        bad = merged(layout_2("minimal"), {"host": {"resources": {"cpus": 0}}})
        _cfg, rows = self.read(json.dumps({**HEADER, **bad}))
        self.assertEqual(
            [(r.name, r.status) for r in rows],
            [("host.resources.cpus", checks.Status.FAIL)],
        )

    def test_a_file_the_reader_refuses(self):
        older = {**HEADER, "schema_version": 1, "repository": {"name": "a/b"}}
        self.assertEqual(self.refusal(json.dumps(older)), "older")
        self.assertEqual(
            self.refusal("rkmon_settings: 1\nconfig:\n  repo: a/b\n"), "header"
        )
        held = {
            **HEADER,
            **layout_2("minimal"),
            "secret_values": {"teams_webhook": "x"},
        }
        self.assertEqual(self.refusal(json.dumps(held)), "secret")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):
                deploy_tree.read(pathlib.Path(d) / "none.yaml")

    def test_a_file_in_the_layout_before_layout_1_is_refused_in_migrate_s_words(self):
        """migrate's refusal, from its one home (migrate.before_layout_1), named as the reader's rows are; a file with
        no kind line that starts with another key keeps the reader's own row."""
        name = f"{file_types.DEPLOY.noun} file"
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "deploy.yaml"
            where = settings_file.shown(path)
            for text, first in (
                ("rkmon_settings: 1\nconfig:\n  repo: a/b\n", "rkmon_settings"),
                ('{"rkmon_build_settings": 1, "image": {}}\n', "rkmon_build_settings"),
            ):
                with self.subTest(first=first):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(settings_file.Refused) as caught:
                        deploy_tree.read(path)
                    exc = caught.exception
                    self.assertEqual((exc.reason, exc.first_key), ("header", first))
                    self.assertEqual(
                        exc.rows, [migrate.before_layout_1(first, where, name)]
                    )
                    [row] = exc.rows
                    self.assertEqual(
                        (row.name, row.found),
                        (
                            name,
                            f"{where} is in the layout before layout 1 (it starts with {first}), which this package "
                            "does not read",
                        ),
                    )
            path.write_text("repository:\n  name: a/b\n", encoding="utf-8")
            with self.assertRaises(settings_file.Refused) as caught:
                deploy_tree.read(path)
            [row] = caught.exception.rows
            self.assertEqual(
                (row.name, row.found),
                (name, f"{where} has no kind line, so it is not deploy settings"),
            )


class TestTheDefaults(unittest.TestCase):
    def test_the_file_answers_whether_atlassian_is_on(self):
        """At the defaults no type of ticket is on, and whether JIRA is used has no default: the
        deploy settings file answers it."""
        self.assertNotIn(
            "enabled", deploy_tree.internal_defaults().get("atlassian", {})
        )
        prototype = layout_2("prototype")
        self.assertIs(deploy_tree.to_config(prototype)["atlassian"]["enabled"], True)
        self.assertIs(
            deploy_tree.to_config(layout_2("minimal"))["atlassian"]["enabled"], False
        )


class TestNames(unittest.TestCase):
    def test_a_configuration_key_is_named_by_its_file_key(self):
        for internal, key in (
            ("repo", "repository.name"),
            ("operator.github_login", "integrations.github_cli.username"),
            ("worker.model", "model_bindings.worker.model_id"),
            ("model_bindings.gw.url", "model_bindings.gw"),
            ("model_bindings.gw.model", "model_bindings.gw.model_id"),
            ("features.ticket_creation.enabled", "features.ticket_creation"),
            (
                "effort.verify",
                "features.review_automation.passes.verification_pass.effort",
            ),
            ("tick_interval_min", "monitor.tick_interval_minutes"),
            ("worker.think", "features.review_automation.passes.first_review_pass"),
        ):
            with self.subTest(internal=internal):
                self.assertEqual(deploy_tree.file_key(internal), key)
                self.assertEqual(deploy_tree.name(internal), key)
        self.assertIsNone(deploy_tree.file_key("hold_label_patterns"))
        self.assertEqual(deploy_tree.name("hold_label_patterns"), "hold_label_patterns")
        held = deploy_tree.held()
        self.assertEqual(held["repo"], "repository.name")
        self.assertFalse([k for k in held if "*" in k])

    def test_a_description_has_its_key_s_own_values(self):
        """{minimum}, {maximum} and {default} in a description are its key's own, and {package:NAME} a package
        default (settings_file.description): no field is left in the text."""
        for key in deploy_tree.held().values():
            with self.subTest(key=key):
                self.assertNotRegex(
                    deploy_tree.description(key),
                    r"\{(minimum|maximum|default|package:[^}]*)\}",
                )

    def test_the_package_s_numbers_in_a_description_are_the_registry_s(self):
        """The memory the work needs beside the RAM disk, the RAM disk's first estimates and the settle's limits are
        the package's defaults when a description is made, never typed into the schema.
        """
        package = registry.package_defaults()
        package["resources"]["work_memory_gib"] = 12
        package["ramdisk"].update(review_estimate_gib=2.5, build_estimate_gib=3)
        package.update(settle_burst_min=50, settle_develop_fix_min=7)
        with mock.patch.object(registry, "package_defaults", return_value=package):
            for key, want in (
                ("host.resources.memory_gib", "At least ramdisk_gib + 12."),
                (
                    "host.resources.ramdisk_gib",
                    "Allow about 2.5 GiB for each review and 3 GiB for each build",
                ),
                (
                    "features.review_automation.schedule.push_cooldown_minutes",
                    "the review starts within 50 minutes, or within 7 for your priorities",
                ),
            ):
                with self.subTest(key=key):
                    self.assertIn(want, deploy_tree.description(key))


class TestTheModels(unittest.TestCase):
    def test_the_built_in_models_are_the_latest_claude(self):
        self.assertEqual(
            deploy_tree.builtin(),
            {
                "opus": "claude-opus-5-5",
                "sonnet": "claude-sonnet-5-5",
                "haiku": "claude-haiku-5-5",
            },
        )

    def test_a_built_in_model_is_sent_by_its_id(self):
        for name, sent in (
            ("opus", "claude-opus-5-5"),
            ("sonnet", "claude-sonnet-5-5"),
            ("haiku", "claude-haiku-5-5"),
            ("opus[1m]", "claude-opus-5-5[1m]"),
            ("claude-sonnet-5-5", "claude-sonnet-5-5"),
            ("fable", "fable"),
        ):
            self.assertEqual(deploy_tree.model_id(name), sent, name)
        cfg = {
            "orchestrator": {"model": "opus"},
            "models": {"audit": "sonnet", "deep": "haiku"},
        }
        self.assertEqual(
            [config.claude_model_id(cfg, r) for r in ("orchestrator", "audit", "deep")],
            ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
        )

    def test_a_model_s_base_id_is_it_without_its_suffix_in_brackets(self):
        for model, base in (
            ("claude-opus-5-5[1m]", "claude-opus-5-5"),
            ("opus[1m]", "opus"),
            ("claude-sonnet-5-5", "claude-sonnet-5-5"),
            ("", ""),
        ):
            self.assertEqual(deploy_tree.base_id(model), base, model)

    def test_a_model_s_family_is_the_name_the_agent_tool_takes(self):
        self.assertEqual(deploy_tree.FAMILIES, ("opus", "sonnet", "haiku", "fable"))
        for model, fam in (
            ("claude-opus-5-5[1m]", "opus"),
            ("opus", "opus"),
            ("claude-sonnet-5-5", "sonnet"),
            ("Claude-Haiku-5.5", "haiku"),  # a gateway's own spelling
            ("fable", "fable"),
            ("example-model-1", ""),
            ("", ""),
        ):
            self.assertEqual(deploy_tree.family(model), fam, model)

    def test_a_claude_binding_of_a_built_in_model_sends_its_id_not_a_gateway_s_name(
        self,
    ):
        # The binding a pass that names "sonnet" gets (_bindings); a gateway's name for the model is read nowhere.
        cfg = {"model_bindings": {"sonnet": {"provider": "claude", "model": "sonnet"}}}
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_DEFAULT_SONNET_MODEL": "example-sonnet-1"}
        ):
            got = bindings.binding_by_name(cfg, "sonnet")
        self.assertEqual(got["model"], "claude-sonnet-5-5")

    def test_the_efforts_of_a_model_that_isnt_claude_have_one_home(self):
        # The checks of a pass's effort and of a binding's read deploy_tree's list; the worker's requests send it.
        self.assertFalse(hasattr(coherence, "OTHER_EFFORTS"))
        self.assertFalse(hasattr(bindings, "EFFORTS"))
        self.assertEqual(llm_client.EFFORTS, deploy_tree.OTHER_EFFORTS)


class TestTheGateway(unittest.TestCase):
    def test_the_sessions_address_is_the_file_s_and_its_key_the_claude_login_s(self):
        """The Claude session's address is the deploy settings file's (integrations.claude_code.base_url; none:
        Anthropic's own API), and its key the claude login's: an older claude_env's address and key are never read.
        """
        with tempfile.TemporaryDirectory() as d:
            cfg = {"paths": {"working_dir": d, "secrets_dir": d}}
            (pathlib.Path(d) / "claude_env").write_text(
                json.dumps(
                    {
                        "ANTHROPIC_BASE_URL": "https://login.example.invalid",
                        "ANTHROPIC_API_KEY": "older",
                    }
                )
            )
            (pathlib.Path(d) / "claude_key").write_text("k\n")
            self.assertEqual(
                orchestrator.session_variables(cfg, orchestrator.read_key(cfg)),
                {"ANTHROPIC_API_KEY": "k"},
            )  # none in the file: Anthropic's own API
            values = merged(
                layout_2("minimal"),
                {
                    "integrations": {
                        "claude_code": {
                            "base_url": "https://gw.example.invalid/anthropic"
                        }
                    }
                },
            )
            cfg = dict(cfg, **deploy_tree.to_config(values))
            self.assertEqual(
                orchestrator.session_variables(cfg, orchestrator.read_key(cfg)),
                {
                    "ANTHROPIC_BASE_URL": "https://gw.example.invalid/anthropic",
                    "ANTHROPIC_API_KEY": "k",
                },
            )


if __name__ == "__main__":
    unittest.main()
