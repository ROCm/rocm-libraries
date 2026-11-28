# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/round_trip.py: the deploy settings file the translation makes of production's configuration,
taken back to today's names, and compared with production's: key by key, and as the monitor runs each.

The synthetic fixture is the translation's (tests/data/production_config.json and production_claude_env_shape.json:
production's shape and switch pattern, invented values); `translated` gives its file's values as rkmon reads them. The
second fixture binds a drafting model, as production does not: the fixture with the local worker drafting. The week
of recorded usage is the spend tests' (tests/test_spend.py, `monitor_day`). The comparisons run as in the one-off
container of production's image, whose environment says so (RKMON_CONTAINER, docker/Dockerfile). The list of intended
differences is the package's (round_trip.DIFFERENCES); the rows the owner agrees are a deployment's own values, so
their file here is invented (AGREED), as the real one stays outside the package. The command (translate_production's
--round-trip) runs only on the fixtures: its half in the one-off container in process (--inside), and the host's half
in a process of its own with a fake docker that answers with the fixture's round trip (FakeDockerCase); nothing calls
the real docker.
"""

import contextlib
import copy
import datetime
import fnmatch
import io
import json
import os
import pathlib
import signal
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

import doc_lints  # noqa: E402
import home_repository  # noqa: E402
import round_trip as rt  # noqa: E402
import translate_production as tp  # noqa: E402
from test_spend import WEEK, monitor_day  # noqa: E402  (the spend tests' week of usage)
from test_translate_production import (  # noqa: E402  (the fixture's loaders; its own tests run in their module)
    DATA,
    DELETE,
    KEY,
    FakeDockerCase,
    changed,
    claude_env,
    production_fixture,
    read_back,
)
from rkmon_lib import (  # noqa: E402
    bindings,
    claude_env_shape,
    config,
    container_layout,
    custom_headers,
    deploy_plan,
    deploy_tree,
    deployment,
    features,
    first_pass,
    from_layout1,
    orchestrator,
    privacy,
    private_files,
    renames,
    reviewer_name,
    secrets_registry,
    spend,
    spend_scan,
)
from rkmon_lib.claude_env_shape import Header, Shape  # noqa: E402
from rkmon_lib.registry import MISSING  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402

CLAUDE_CODE = "integrations.claude_code"
API_KEY, AUTH_TOKEN = claude_env_shape.KEY_VARS
# The file's key of the spend prices (its x-internal: prices).
PRICES = "features.review_automation.spend.usd_per_million_tokens"
# Invented rows in the form of those the owner agrees.
AGREED = DATA / "round_trip_agreed.json"
# The file's key of the verification pass, whose model and effort the verifier's line 1 gives.
VERIFIER = "features.review_automation.passes.verification_pass"


def setUpModule():
    # Production's repository, invented in the fixture, stands for the home repository: its repository defaults apply.
    cfg, _shape = production_fixture()
    home_repository.stand_in(cfg["repo"], cfg["rocke_path"])
    patch = mock.patch.dict(os.environ, {"RKMON_CONTAINER": "1"})
    patch.start()
    unittest.addModuleCleanup(patch.stop)


def translated(cfg=None, shape=None) -> dict:
    """The values of the deploy settings file the translation makes of CFG and SHAPE (the fixture's by default), read
    back as rkmon reads the file."""
    fixture_cfg, fixture_shape = production_fixture()
    done = tp.translate(
        fixture_cfg if cfg is None else cfg, fixture_shape if shape is None else shape
    )
    return read_back(done.deploy_file)


def second_fixture() -> dict:
    """The second fixture's config.json: the fixture with the local worker drafting the first review pass, which the
    file binds as a drafting model."""
    cfg, _shape = production_fixture()
    return changed(cfg, {"reviewer": "worker"})


def edited(values: dict, changes: dict) -> dict:
    """A copy of the file's VALUES with each file key of CHANGES set to its value."""
    out = copy.deepcopy(values)
    for key, value in changes.items():
        deploy_tree.put(out, key, value)
    return out


def by_key(found) -> dict:
    """The differences FOUND by (key, kind)."""
    return {(d.key, d.kind): d for d in found}


def marked(model: str, effort: str) -> str:
    """Line 1 that gives MODEL and EFFORT, as the round trip compares it: apart from the name it gives
    (reviewer_name.name_marked)."""
    return reviewer_name.REVIEW_LINE.format(
        name=reviewer_name.NAME_MARK, model=model, effort=effort
    )


# The one-off container's id, as docker writes it to the file --cidfile names (invented).
ONE_OFF_ID = "0123456789abcdef" * 4


def made(argv) -> pathlib.Path:
    """The one-off container made, as docker run makes it: ONE_OFF_ID written to the file the command ARGV's --cidfile
    names, which is the file given back."""
    path = pathlib.Path(argv[argv.index("--cidfile") + 1])
    path.write_text(ONE_OFF_ID, encoding="utf-8")
    return path


@contextlib.contextmanager
def as_a_terminal_starts_a_command():
    """While the block runs, a command this process starts begins with SIGINT at its default, as a terminal starts one:
    here SIGINT is caught (KeyboardInterrupt), and a command's start (exec) puts a caught signal back to its default,
    where an ignored one stays ignored. A process started with SIGINT ignored, as a background job of a shell that is
    not interactive is (a script's or CI's run of the suite), would pass that on, and the command would never take a
    Ctrl-C.
    """
    before = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, before)


def week_written(root: pathlib.Path) -> pathlib.Path:
    """The spend tests' week (WEEK, `monitor_day`) as transcripts under ROOT, a projects folder it makes; ROOT."""
    root.mkdir(parents=True)
    for d in WEEK:
        monitor_day(root, d)
    return root


def recorded_week(folder: pathlib.Path, run: dict) -> rt.Usage:
    """The spend tests' week, its transcripts written under FOLDER (`week_written`) and recorded both ways
    (rt.usage_of): by family, as today's records keep it, and by the id each pass of the running configuration RUN
    asks for.
    """
    return rt.usage_of(week_written(folder / "projects"), folder, run)


def claude_settings(settings: dict) -> dict:
    """The configuration integrations.claude_code's SETTINGS make, each under its key's x-internal."""
    out = {}
    for key, value in settings.items():
        deploy_tree.put(out, rt.internal(f"{CLAUDE_CODE}.{key}"), value)
    return out


def shape(*variables, headers=(), base_url="", nonessential="") -> Shape:
    """A claude_env shape that sets VARIABLES (in claude_env_shape.CLAUDE_VARS' order), with HEADERS."""
    return Shape(
        variables=tuple(v for v in claude_env_shape.CLAUDE_VARS if v in variables),
        headers=tuple(headers),
        base_url=base_url,
        nonessential_traffic=nonessential,
    )


def each_kind_changed(cfg: dict, values: dict) -> list:
    """A change of each kind to production's configuration CFG and its translated file's VALUES, each as (production's
    configuration, the file's values, the difference it makes): a value the file feeds; a feature's switch; how the
    Claude session sends its key; a pass's model, then its effort; the verifier's model by its id, the same model as
    production's alias, which line 1 gives in other words; a package key's own value in production's file, which the
    file does not carry. The price total's needs a recorded week (`recorded_week`).
    """
    out = [
        (cfg, edited(values, changes), want)
        for changes, want in (
            (
                {"features.ticket_creation.project_key": "OTHER"},
                rt.Difference("jira.project", rt.VALUE, "SAMPLE", "OTHER"),
            ),
            (
                {"features.weekly_report.enabled": False},
                rt.Difference("weekly-report", rt.FEATURE, True, False),
            ),
            (
                {f"{CLAUDE_CODE}.api_key_sent_as": "bearer"},
                rt.Difference(
                    rt.SESSION_HEADERS,
                    rt.HEADER,
                    ["Authorization: Bearer <key>", "X-Gateway-Key: <key>"],
                    ["Authorization: Bearer <key>"],
                ),
            ),
            (
                {"features.review_automation.passes.code_quality_pass.model": "sonnet"},
                rt.Difference(
                    "deep", rt.MODEL_ID, "claude-opus-5-5", "claude-sonnet-5-5"
                ),
            ),
            (
                {"features.review_automation.passes.code_quality_pass.effort": "low"},
                rt.Difference("quality", rt.EFFORT, "medium", "low"),
            ),
            (
                {f"{VERIFIER}.model": "claude-opus-5-5"},
                rt.Difference(
                    "review_first_line_no_draft",
                    rt.LINE_1,
                    marked("Claude Opus", "high"),
                    marked("Claude Opus 5.5", "high"),
                ),
            ),
        )
    ]
    larger = changed(cfg, {"huge_pr_lines": 12000})
    out.append(
        (
            larger,
            translated(larger),
            rt.Difference("huge_pr_lines", rt.INTERNAL, 12000, 8000),
        )
    )
    return out


# Each claude_env shape the settings carry (claude_env_shape.settings): the key in one variable or both, and a custom
# header with the key, after a word or alone, beside one of them.
GATEWAY_KEY = Header("X-Gateway-Key", True, "")
GATEWAY_WORD = Header("X-Gateway-Key", True, "Token")
CARRIED_SHAPES = {
    "the API key, no custom header": shape(API_KEY),
    "the bearer token, no custom header": shape(AUTH_TOKEN),
    "both, no custom header": shape(API_KEY, AUTH_TOKEN),
    "the API key and a header with the key": shape(
        API_KEY, rt.CUSTOM_HEADERS, headers=[GATEWAY_KEY]
    ),
    "the API key and a header with a word and the key": shape(
        API_KEY, rt.CUSTOM_HEADERS, headers=[GATEWAY_WORD]
    ),
    "the bearer token and a header with the key": shape(
        AUTH_TOKEN, rt.CUSTOM_HEADERS, headers=[GATEWAY_KEY]
    ),
    "the bearer token and a header with a word and the key": shape(
        AUTH_TOKEN, rt.CUSTOM_HEADERS, headers=[GATEWAY_WORD]
    ),
}


class TestInverse(unittest.TestCase):
    def test_each_claude_env_shape_comes_back_from_its_settings(self):
        """The inverse of claude_env_shape.settings: each shape the settings carry comes back
        from them, its base address too; its model variables and its nonessential traffic switch, which no setting
        carries, do not."""
        base = "https://llm-gateway.example.com/anthropic"
        for name, want in CARRIED_SHAPES.items():
            with self.subTest(shape=name):
                given = want._replace(
                    variables=(
                        *want.variables,
                        rt.BASE_URL,
                        *claude_env_shape.MODEL_VARS,
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                    ),
                    base_url=base,
                    nonessential_traffic="1",
                )
                settings, cannot = claude_env_shape.settings(given)
                self.assertEqual(cannot, [])
                back = rt.secret_shape(claude_settings(settings))
                self.assertEqual(
                    back,
                    want._replace(
                        variables=shape(rt.BASE_URL, *want.variables).variables,
                        base_url=base,
                    ),
                )

    def test_the_shared_labels_go_back_to_today_s_list_where_it_can_say_them(self):
        """Each label of the map that counts for the repository's own folder: today's list, in the map's order; a
        label for another folder keeps the map; the configuration given is not changed.
        """
        name = rt.internal(rt.SHARED_LABELS)
        folder = rt.internal(rt.FOLDER)
        for shared, want in (
            ({"b: two": "x/y", "a: one": "x/y/"}, ["b: two", "a: one"]),
            ({}, []),
            ({"b: two": "x/y", "a: one": "x/z"}, None),
        ):
            with self.subTest(shared=shared):
                cfg = {}
                deploy_tree.put(cfg, folder, "x/y")
                deploy_tree.put(cfg, name, shared)
                given = copy.deepcopy(cfg)
                back = rt.labels_back(cfg)
                self.assertEqual(cfg, given)
                self.assertEqual(
                    deploy_tree.get(back, name), shared if want is None else want
                )

    def test_the_fixture_s_file_comes_back_in_today_s_names(self):
        """The fixture's file, taken back: its values under their x-internals, the shared labels as today's list, the
        RAM disk and the image through deploy_tree's own rules (production's blank image as the image it stands for,
        which the file writes), and claude_env's shape but for what no setting carries.
        """
        cfg, production_shape = production_fixture()
        today = renames.migrate(cfg)[0]
        back = rt.inverse(translated())
        for key in (
            "repo",
            rt.internal(rt.FOLDER),
            "scope.labels",
            rt.internal(rt.SHARED_LABELS),
            "prices",
            "jira.component",
            "deployment.project",
            "ramdisk.enabled",
            "ramdisk.size_gb",
            "automerge_guard_own",
            "orchestrator.model",
            "models.audit",
            "models.deep",
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    deploy_tree.get(back.config, key), deploy_tree.get(today, key)
                )
        self.assertEqual(deploy_tree.get(today, "deployment.image"), "")
        # what production ran for it: the name layout 1's package used (from_layout1.IMAGE_NAME), tagged with the
        # container's name, as deployment.image_ref tags a blank one
        self.assertEqual(
            deploy_tree.get(back.config, "deployment.image"),
            f"{from_layout1.IMAGE_NAME}:{deployment.project(today)}",
        )
        self.assertEqual(
            back.shape,
            production_shape._replace(
                variables=tuple(
                    v
                    for v in production_shape.variables
                    if v in (rt.BASE_URL, AUTH_TOKEN, rt.CUSTOM_HEADERS)
                ),
                nonessential_traffic="",
            ),
        )

    def test_the_image_and_the_ram_disk_come_back_by_deploy_tree_s_rules(self):
        """The image and the RAM disk go back through the rules deploy_tree carries them by: an image's name and tag
        joined, the default name with a blank tag no value; a RAM disk of 0 GiB off."""
        cfg, _shape = production_fixture()
        for changes, want in (
            (
                {"deployment.image": "team/monitor:1.2", "ramdisk.enabled": False},
                {"deployment.image": "team/monitor:1.2", "ramdisk.enabled": False},
            ),
            (
                {"ramdisk.size_gb": 4},
                {"ramdisk.enabled": True, "ramdisk.size_gb": 4},
            ),
        ):
            with self.subTest(changes=changes):
                back = rt.inverse(translated(changed(cfg, changes))).config
                for key, value in want.items():
                    self.assertEqual(deploy_tree.get(back, key), value, key)


class TestComparisons(unittest.TestCase):
    """rt.compare: production's configuration and claude_env's shape against the translated file's, key
    by key and as the monitor runs them, each difference by its key and its kind."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.shape = production_fixture()
        cls.values = translated()
        cls.found = by_key(rt.compare(cls.cfg, cls.shape, cls.values))

    def compared(self, cfg=None, values=None, usage=None) -> dict:
        return by_key(
            rt.compare(
                self.cfg if cfg is None else cfg,
                self.shape,
                self.values if values is None else values,
                usage,
            )
        )

    def test_the_fixture_differs_in_values_and_one_feature_each_named_with_both_sides(
        self,
    ):
        """The fixture's differences are values, of the keys the file feeds and of the others, and one feature: the
        needs-you alerts, which production sends while its chat alerts are on, as one switch sends both alerts, and the
        file, which keeps production's two switches, does not. No header, model id or effort differs. Of line 1, the
        verifier's, posted with no first-pass drafter, is the same, and the drafter's differs: the file binds no
        drafting model, so its side has none. Each is named by its key with both sides; a mapping one side leaves empty
        is told by the entries the other has; a local worker's setting production sets and the file, which binds no
        drafting model, leaves out, by rt.NO_DRAFTING_MODEL; a key the file feeds is never an internal value, nor a key
        the Claude session's settings feed a value of its own (they are compared as the session gets them). A package
        key production's file does not hold is compared at the package's value for its repository, as the package reads
        it: the repository defaults' variables for each GPU job are no difference. The running configuration is the
        container's, whose image has the checks venv production's names: no difference either.
        """
        self.assertEqual(
            {kind for _key, kind in self.found},
            {rt.VALUE, rt.FEATURE, rt.LINE_1, rt.INTERNAL},
        )
        self.assertEqual(
            [k for k, kind in self.found if kind == rt.FEATURE], ["needs-you-alerts"]
        )
        self.assertEqual(
            [k for k, kind in self.found if kind == rt.LINE_1], ["review_first_line"]
        )
        for key, kind, production, after in (
            (
                "publish.html.url",
                rt.VALUE,
                "http://web.example.com:8080/",
                "http://web.example.com:8080",
            ),
            ("needs-you-alerts", rt.FEATURE, True, False),
            ("ANTHROPIC_MODEL", rt.VALUE, rt.SET, MISSING),
            ("worker.model", rt.VALUE, "drafter-model-a", rt.NO_DRAFTING_MODEL),
            ("gpu.env.ROCKE_BACKEND", rt.VALUE, "python", MISSING),
            ("container.pr_uid", rt.INTERNAL, 1998, MISSING),
            (
                "review_first_line",
                rt.LINE_1,
                marked("drafter-model-a", config.MODEL_DEFAULT),
                "",
            ),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    self.found[key, kind], rt.Difference(key, kind, production, after)
                )
        for key, kind in (
            ("gpu.env", rt.VALUE),
            ("gpu.base_env", rt.INTERNAL),
            ("gpu.base_env.ROCKE_BACKEND", rt.INTERNAL),
            ("paths.venv", rt.INTERNAL),
        ):
            self.assertNotIn((key, kind), self.found)
        held = deploy_tree.held()
        claude = {i for p, i in deploy_tree.feeds() if p.startswith(CLAUDE_CODE + ".")}
        for key, kind in self.found:
            parts = key.split(".")
            names = {".".join(parts[:n]) for n in range(1, len(parts) + 1)}
            if kind == rt.INTERNAL:
                self.assertFalse(names & set(held), key)
            self.assertFalse(names & claude, key)

    def test_a_change_of_each_kind_is_a_difference_of_that_kind_by_its_key(self):
        """A change the fixture's comparison does not have (`each_kind_changed`): each is a difference of its kind,
        named by its key, with both sides.
        """
        for production, values, want in each_kind_changed(self.cfg, self.values):
            with self.subTest(kind=want.kind):
                found = self.compared(production, values)
                self.assertNotIn((want.key, want.kind), self.found)
                self.assertEqual(found.get((want.key, want.kind)), want)

    def test_a_difference_is_shown_by_its_key_a_session_variable_under_the_older_secret(
        self,
    ):
        """rt.named, the parts the report shows a difference's key by: a variable of the Claude session under the
        name of the older Claude secret (secrets_registry.CLAUDE_ENV), which sets it today; any other key split at
        its dots.
        """
        for key, want in (
            ("ANTHROPIC_MODEL", (secrets_registry.CLAUDE_ENV, "ANTHROPIC_MODEL")),
            (rt.NONESSENTIAL, (secrets_registry.CLAUDE_ENV, rt.NONESSENTIAL)),
            ("publish.html.url", ("publish", "html", "url")),
            ("gpu.env.ROCKE_BACKEND", ("gpu", "env", "ROCKE_BACKEND")),
            ("needs-you-alerts", ("needs-you-alerts",)),
        ):
            with self.subTest(key=key):
                self.assertEqual(rt.named(key), want)

    def test_the_switches_production_s_code_joined_or_did_not_have_are_read_as_it_ran_them(
        self,
    ):
        """rt.switches_ran, today's configuration as FEATURE reads it: each alert by the alerts' one switch, which
        sent both (from_layout1.alert_switches); the queue notices by the merge queue's switch, and the
        drift notices by the merge queue's and review automation's (from_layout1.notices), whatever switches of
        their own it has, which production's code did not read. Its other values stay, and the configuration given
        is not changed."""
        for alerts, queue, review, sent, notices in (
            (True, True, True, True, (True, True)),
            (False, True, True, False, (True, True)),
            (True, True, False, True, (True, False)),
            (True, False, True, True, (False, False)),
        ):
            cfg = {
                "features": {
                    "alerts": {
                        "enabled": alerts,
                        "develop_breaks": not alerts,
                        "needs_you": False,
                    },
                    "merge_queue": {
                        "enabled": queue,
                        "mode": "live",
                        "queue_notices": False,
                        "drift_notices": not queue,
                    },
                    "review_automation": {"enabled": review, "mode": "live"},
                },
                "huge_pr_lines": 9000,
            }
            given = copy.deepcopy(cfg)
            with self.subTest(alerts=alerts, queue=queue, review=review):
                switched = rt.switches_ran(cfg)
                self.assertEqual(cfg, given)
                self.assertEqual(
                    [deploy_tree.get(switched, k) for k in renames.ALERT_KINDS],
                    [sent, sent],
                )
                self.assertEqual(
                    tuple(
                        deploy_tree.get(switched, rt.internal(p)) for p in rt.NOTICES
                    ),
                    notices,
                )
                self.assertEqual(switched["huge_pr_lines"], 9000)
                self.assertEqual(
                    [
                        features.value(switched, name)
                        for name in (
                            "chat-alerts",
                            "needs-you-alerts",
                            "queue-notices",
                            "drift-notices",
                        )
                    ],
                    [alerts, alerts, *notices],
                )

    def test_the_second_fixture_s_first_review_pass_sends_the_same_headers(self):
        """The first review pass's headers on the second fixture, which binds a drafting model: today's from the
        local worker's settings, the file's from its binding, the key as the marker; production's fixture binds none,
        so its first review pass sends none. A change in how the binding sends its key is a header difference.
        """
        second = second_fixture()
        values = translated(second)
        want = ("X-Api-Key: <key>", "Authorization: Bearer <key>")
        self.assertEqual(rt.first_pass_headers(rt.today(second)), want)
        self.assertEqual(rt.first_pass_headers(rt.running(values, second)), want)
        found = self.compared(second, values)
        self.assertEqual(
            {kind for _key, kind in found}, {rt.VALUE, rt.FEATURE, rt.INTERNAL}
        )
        self.assertEqual(
            [k for k, kind in found if kind == rt.FEATURE], ["needs-you-alerts"]
        )
        self.assertEqual(rt.first_pass_headers(rt.today(self.cfg)), ())
        self.assertEqual(rt.first_pass_headers(rt.running(self.values, self.cfg)), ())
        found = self.compared(
            second, edited(values, {"model_bindings.worker.api_key_sent_as": "bearer"})
        )
        self.assertEqual(
            found[rt.FIRST_PASS_HEADERS, rt.HEADER],
            rt.Difference(
                rt.FIRST_PASS_HEADERS,
                rt.HEADER,
                list(want),
                ["Authorization: Bearer <key>"],
            ),
        )

    def test_each_claude_env_shape_gives_back_its_headers(self):
        """Each claude_env shape in CARRIED_SHAPES: the headers and the base address the session gets from it today
        are those it gets from the settings made from it (orchestrator.session_variables), in the order sent
        (orchestrator.request_headers), the key as the marker (rkmon_lib/custom_headers.py's KEY_MARK).
        """
        base = "https://llm-gateway.example.com/anthropic"
        mark = custom_headers.KEY_MARK
        for name, want in CARRIED_SHAPES.items():
            with self.subTest(shape=name):
                given = want._replace(
                    variables=(rt.BASE_URL, *want.variables), base_url=base
                )
                settings, cannot = claude_env_shape.settings(given)
                self.assertEqual(cannot, [])
                today = orchestrator.request_headers(rt.shape_variables(given))
                after = orchestrator.session_variables(claude_settings(settings), mark)
                self.assertEqual(
                    list(orchestrator.request_headers(after).items()),
                    list(today.items()),
                )
                self.assertEqual(after[rt.BASE_URL], base)
                self.assertTrue(today)
                for value in today.values():
                    self.assertTrue(value.endswith(mark), value)

    def test_the_key_stands_as_the_packages_one_marker(self):
        """Wherever the round trip shows the key, rkmon_lib/custom_headers.py's KEY_MARK stands for it, the marker the
        bindings' key headers carry too: in the claude_env shape the settings stand for, in the session's headers,
        today's and the running configuration's, and in the first review pass's headers. Here it is another marker.
        """
        mark = "[key]"
        word = CARRIED_SHAPES["the API key and a header with a word and the key"]
        second = second_fixture()
        with mock.patch.object(custom_headers, "KEY_MARK", mark):
            settings, _cannot = claude_env_shape.settings(word)
            self.assertEqual(rt.secret_shape(claude_settings(settings)), word)
            self.assertEqual(
                rt.first_pass_headers(rt.today(second)),
                (f"X-Api-Key: {mark}", f"Authorization: Bearer {mark}"),
            )
            found = self.compared(
                values=edited(self.values, {f"{CLAUDE_CODE}.api_key_sent_as": "bearer"})
            )
        self.assertEqual(
            found[rt.SESSION_HEADERS, rt.HEADER],
            rt.Difference(
                rt.SESSION_HEADERS,
                rt.HEADER,
                [f"Authorization: Bearer {mark}", f"X-Gateway-Key: {mark}"],
                [f"Authorization: Bearer {mark}"],
            ),
        )

    def test_each_pass_runs_on_the_same_model_id_and_effort(self):
        """The model id each pass and the session run on, and each effort as sent: the same on both sides, on the
        fixture and on the second fixture, whose first review pass runs on its drafting model with no effort sent
        (production's "max" is never sent to a model that is not Claude)."""
        claude = dict.fromkeys(("orchestrator", "audit", "deep"), "claude-opus-5-5")
        optional = dict.fromkeys(rt.OPTIONAL_PASSES)
        sent = {
            "orchestrator": "high",
            "verify": "high",
            "quality": "medium",
            "audit": "high",
            "deep": "medium",
            **dict.fromkeys(("tick", "first_pass", *config.FIRST_PASS_ROLES)),
        }
        for cfg, first in ((self.cfg, None), (second_fixture(), "drafter-model-a")):
            with self.subTest(first_pass=first):
                run = rt.running(translated(cfg), cfg)
                for side in (rt.today(cfg), run):
                    self.assertEqual(
                        rt.model_ids(side), dict(claude, first_pass=first, **optional)
                    )
                    self.assertEqual(rt.efforts(side), sent)

    def test_the_first_review_pass_s_effort_is_the_one_its_drafter_s_requests_carry(
        self,
    ):
        """The first review pass's effort as its drafter's requests carry it, by bindings.drafter_effort, the rule's
        one home: a local worker on an Ollama server that does not think sends think false (bindings.THINKING_OFF),
        which one that thinks does not, so the two differ; a Claude binding carries the pass's own effort.
        """
        side = rt.today(second_fixture())
        for think, sent in (
            (False, bindings.THINKING_OFF),
            (True, None),
            ("high", "high"),
        ):
            with self.subTest(think=think):
                worker = dict(side.get("worker") or {}, backend="ollama", think=think)
                self.assertEqual(
                    rt.efforts(dict(side, worker=worker))["first_pass"], sent
                )
        claude = dict(
            side,
            reviewer="c",
            model_bindings={"c": {"provider": "claude", "model": "model-x"}},
            effort=dict(side.get("effort") or {}, first_pass="low"),
        )
        self.assertEqual(rt.efforts(claude)["first_pass"], "low")

    def test_one_recorded_week_costs_the_same_priced_both_ways(self):
        """The spend tests' week, recorded by family as today and by the id each pass asks for as this package
        does, priced at production's prices and at the file's: the same total each day, for each pass and for the
        week. A price the file changes changes each total; a pass on another model is priced by its own id.
        """
        run = rt.running(self.values, self.cfg)
        with tempfile.TemporaryDirectory() as d:
            usage = recorded_week(pathlib.Path(d), run)
        days = rt.week_of(usage.today)
        self.assertEqual(days, tuple(WEEK))
        # a day: the session's turn $0.50, the verifier's run $2.50 and the code quality lane's $1.25 at the
        # fixture's prices (cache reads at $0.50 a million tokens)
        self.assertEqual(
            rt.totals(usage.after, spend.price_table(run), days),
            {
                **{f"{rt.USAGE}.{d}": 4.25 for d in WEEK},
                f"{rt.USAGE}.orchestrator": 3.5,
                f"{rt.USAGE}.verifier": 17.5,
                f"{rt.USAGE}.quality": 8.75,
                f"{rt.USAGE}.{rt.WEEK}": 29.75,
            },
        )
        found = self.compared(usage=usage)
        self.assertFalse([k for k, kind in found if kind == rt.PRICE])
        dearer = edited(self.values, {f"{PRICES}.claude-opus-5-5.cache_read": 0.6})
        found = self.compared(values=dearer, usage=usage)
        self.assertEqual(
            {k for k, kind in found if kind == rt.PRICE},
            {
                *(f"{rt.USAGE}.{d}" for d in WEEK),
                *(f"{rt.USAGE}.{p}" for p in ("orchestrator", "verifier", "quality")),
                f"{rt.USAGE}.{rt.WEEK}",
            },
        )
        self.assertEqual(
            found[f"{rt.USAGE}.{rt.WEEK}", rt.PRICE],
            rt.Difference(f"{rt.USAGE}.{rt.WEEK}", rt.PRICE, 29.75, 35.7),
        )
        sonnet = edited(
            self.values,
            {"features.review_automation.passes.verification_pass.model": "sonnet"},
        )
        with tempfile.TemporaryDirectory() as d:
            usage = recorded_week(pathlib.Path(d), rt.running(sonnet, self.cfg))
        found = self.compared(values=sonnet, usage=usage)
        self.assertEqual(
            found[f"{rt.USAGE}.verifier", rt.PRICE],
            rt.Difference(f"{rt.USAGE}.verifier", rt.PRICE, 17.5, 10.5),
        )

    def test_the_week_is_the_seven_days_that_end_on_the_last_recorded(self):
        items = [{"start": "2026-11-20T10:00:00Z"}, {"t": "2026-11-02T09:00:00Z"}, {}]
        self.assertEqual(
            rt.week_of(items),
            tuple(f"2026-11-{d}" for d in range(14, 21)),
        )
        self.assertEqual(rt.week_of([{}]), ())

    def test_the_differences_go_through_json_as_they_are(self):
        """rt.to_json and rt.from_json, as the one-off container prints the differences and the host reads them: each
        comes back, a side with no value left out of the JSON; data of another form is a ValueError.
        """
        found = rt.compare(self.cfg, self.shape, self.values)
        data = json.loads(json.dumps(rt.to_json(found)))
        self.assertEqual(rt.from_json(data), found)
        at = [n for n, d in enumerate(found) if d.after is MISSING]
        self.assertTrue(at)
        self.assertEqual(sorted(data[at[0]]), ["key", "kind", "production"])
        for bad in (
            {},
            [["k"]],
            [{"key": "k"}],
            [{"key": "k", "kind": "price"}],
            [{"key": 1, "kind": rt.VALUE}],
            [{"key": "k", "kind": rt.VALUE, "note": 1}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rt.from_json(bad)


class TestTheRead(unittest.TestCase):
    """The monitor's transcripts read once for the week priced (rt.usage_of): the main sessions last written in the
    READ_DAYS days up to the newest one's last write, with their subagents, and all of them when the week those
    record reaches back past them (rt.holds). Either way the week and its prices are a whole read's, and both ways are
    priced from the one read. Each session is the spend tests' monitor's (`monitor_day`), in a project folder of
    its own, its file last written when the test says."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        cfg = production_fixture()[0]
        self.today, self.run = rt.today(cfg), rt.running(translated(cfg), cfg)

    def session(
        self, root: pathlib.Path, project: str, days, written: str
    ) -> pathlib.Path:
        """The monitor's session in PROJECT, a folder of its own in the projects folder ROOT: a day of `monitor_day`
        for each of DAYS, its file last written at WRITTEN, a UTC time. Its file."""
        folder = root / project
        folder.mkdir(parents=True)
        for d in days:
            monitor_day(folder, d)
        session = folder / "s1.jsonl"
        at = datetime.datetime.fromisoformat(written).timestamp()
        os.utime(session, (at, at))
        return session

    def read(self, root: pathlib.Path, name: str, whole: bool = False) -> rt.Usage:
        """The transcripts under ROOT read by rt.usage_of, the scan's cache in a folder NAME; WHOLE: all of them
        first, as with no time to read from (rt.since_of None)."""
        with (
            mock.patch.object(rt, "since_of", return_value=None)
            if whole
            else contextlib.nullcontext()
        ):
            return rt.usage_of(root, self.dir / name, self.run)

    def priced(self, usage: rt.Usage) -> tuple:
        """USAGE's week, and its totals priced both ways: at production's prices by family, at the file's by id."""
        days = rt.week_of(usage.today)
        return (
            days,
            rt.totals(usage.today, spend.price_table(self.today), days),
            rt.totals(usage.after, spend.price_table(self.run), days),
        )

    def test_the_time_read_from_and_whether_the_week_holds(self):
        root = self.dir / "projects"
        self.assertIsNone(rt.since_of(root))
        self.session(root, "current", WEEK, "2026-11-15T10:00:00+00:00")
        newest = datetime.datetime.fromisoformat("2026-11-15T10:00:00+00:00")
        self.assertEqual(
            rt.since_of(root),
            (newest - datetime.timedelta(days=rt.READ_DAYS)).timestamp(),
        )
        since = datetime.datetime.fromisoformat("2026-11-07T10:00:00+00:00").timestamp()

        def week(first: str) -> tuple:
            start = datetime.date.fromisoformat(first)
            return tuple(
                str(start + datetime.timedelta(days=n)) for n in range(rt.WEEK_DAYS)
            )

        self.assertTrue(rt.holds(week("2026-11-09"), since))
        self.assertFalse(rt.holds(week("2026-11-08"), since), "the day after since's")
        self.assertFalse(rt.holds((), since))

    def test_the_sessions_written_last_are_read_and_the_week_is_a_whole_reads(self):
        """A session last written a month before the week is never read: the week of the sessions read is the
        week of all of them, priced the same both ways."""
        root = self.dir / "projects"
        older = self.session(root, "older", ["2026-10-01"], "2026-10-01T10:00:00+00:00")
        self.session(root, "current", WEEK, "2026-11-15T10:00:00+00:00")
        read = []
        real = spend_scan.parse_main

        def spy(path, ent, budget, asked=None):
            read.append(path)
            return real(path, ent, budget, asked)

        with mock.patch.object(spend_scan, "parse_main", spy):
            usage = self.read(root, "newest")
        self.assertNotIn(older, read)
        whole = self.read(root, "whole", whole=True)
        self.assertEqual(rt.week_of(whole.today), tuple(WEEK))
        self.assertEqual(self.priced(usage), self.priced(whole))
        self.assertLess(len(usage.today), len(whole.today))

    def test_a_week_that_reaches_back_past_the_sessions_read_first_reads_them_all(self):
        """The newest session records a day after the week and was last written days later, so the sessions read
        first leave out one that records most of their week: all are read then, and the week and its prices are a
        whole read's; those of the sessions read first would not be."""
        root = self.dir / "projects"
        self.session(root, "current", WEEK, "2026-11-15T10:00:00+00:00")
        self.session(root, "late", ["2026-11-16"], "2026-11-25T10:00:00+00:00")
        usage, whole = self.read(root, "newest"), self.read(root, "whole", whole=True)
        days = tuple(f"2026-11-{d}" for d in range(10, 17))
        self.assertEqual(self.priced(usage)[0], days)
        self.assertEqual(self.priced(usage), self.priced(whole))
        since = rt.since_of(root)
        first = spend_scan.scan(root, Store(self.dir / "first"), since=since)
        items = first["runs"] + first["turns"]
        self.assertFalse(rt.holds(rt.week_of(items), since))
        prices = spend.price_table(self.today)
        self.assertLess(
            rt.totals(items, prices, days)[f"{rt.USAGE}.{rt.WEEK}"],
            self.priced(whole)[1][f"{rt.USAGE}.{rt.WEEK}"],
        )

    def test_a_transcript_written_while_it_is_read_changes_no_price(self):
        """The monitor writes on while the round trip reads its transcripts: here each scan is followed by one tick
        more on the week's last day (`monitor_day`). Both ways are priced from one read, so no price differs, when
        the sessions read first hold and when all are read after them; a second read, as each way once had, holds
        the tick more."""
        for name, holds in (
            ("the sessions read first", True),
            ("all read after them", False),
        ):
            with self.subTest(read=name):
                root = self.dir / name / "projects"
                session = self.session(
                    root, "current", WEEK, "2026-11-15T10:00:00+00:00"
                )
                real = spend_scan.scan

                def then_a_tick(*args, **kwargs):
                    out = real(*args, **kwargs)
                    monitor_day(session.parent, WEEK[-1])
                    return out

                with (
                    mock.patch.object(spend_scan, "scan", then_a_tick),
                    mock.patch.object(rt, "holds", return_value=holds),
                ):
                    usage = rt.usage_of(root, self.dir / name / "scan", self.run)
                self.assertEqual(rt.prices_compared(usage, self.today, self.run), [])
                again = spend_scan.scan(
                    root,
                    Store(self.dir / name / "again"),
                    asked=spend.asked_ids(self.run),
                )
                week = f"{rt.USAGE}.{rt.WEEK}"
                days = rt.week_of(usage.today)
                prices = spend.price_table(self.run)
                self.assertGreater(
                    rt.totals(again["runs"] + again["turns"], prices, days)[week],
                    rt.totals(usage.after, prices, days)[week],
                )


class TestTheList(unittest.TestCase):
    """The list of intended differences: the rows that say why production's configuration and the
    translated file's may differ, read from round_trip.DIFFERENCES; the round trip passes only when each difference is
    on it."""

    URL = "https://github.com/ROCm/sample-tool.git"
    # Each variant of the fixture: its name, its config.json's changes and its claude_env shape's, and differences it
    # makes, each (key, kind), that a row says why of.
    VARIANTS = (
        (
            "the fixture",
            {},
            {},
            (
                ("container.uid", rt.VALUE),
                ("effort.verify", rt.VALUE),
                ("gpu.env.ROCKE_BACKEND", rt.VALUE),
                ("publish.html.url", rt.VALUE),
                ("team_dashboard.url", rt.VALUE),
                ("worker.model", rt.VALUE),
                ("worker.think", rt.VALUE),
                ("ANTHROPIC_MODEL", rt.VALUE),
                ("ANTHROPIC_DEFAULT_OPUS_MODEL", rt.VALUE),
                ("needs-you-alerts", rt.FEATURE),
                ("admin_dashboard.bind", rt.INTERNAL),
                ("admin_dashboard.token_file", rt.INTERNAL),
                ("clone_url", rt.INTERNAL),
                ("container.pr_uid", rt.INTERNAL),
                ("draft_stage", rt.INTERNAL),
                ("followup_scan.dir", rt.INTERNAL),
                ("gpu.known_hosts", rt.INTERNAL),
                ("paths.rocm_reference_clang", rt.INTERNAL),
                ("profile_dir", rt.INTERNAL),
                ("review_first_line", rt.LINE_1),
            ),
        ),
        (
            "the second fixture",
            {"reviewer": "worker"},
            {},
            (("worker.think", rt.VALUE),),
        ),
        (
            "the alerts' switches the other way round",
            {"features.chat-alerts": False, "features.needs-you-alerts": True},
            {},
            (("chat-alerts", rt.FEATURE),),
        ),
        (
            "the team dashboard served",
            {"team_dashboard.mode": "serve", "team_dashboard.public_port": 0},
            {},
            (("team_dashboard.public_port", rt.VALUE),),
        ),
        (
            "the team dashboard served on a blank port",
            {"team_dashboard.mode": "serve", "team_dashboard.public_port": ""},
            {},
            (("team_dashboard.public_port", rt.VALUE),),
        ),
        (
            "blanks that stand for values",
            {
                "admin_dashboard.public_port": 0,
                "jira.url": "",
                "atlassian.site": "https://jira.example.com/",
            },
            {},
            (
                ("admin_dashboard.public_port", rt.VALUE),
                ("jira.url", rt.VALUE),
                ("atlassian.site", rt.VALUE),
            ),
        ),
        (
            "a blank public port",
            {"admin_dashboard.public_port": ""},
            {},
            (("admin_dashboard.public_port", rt.VALUE),),
        ),
        (
            "addresses with a closing slash",
            {
                "jira.url": "https://jira.example.com/",
                "atlassian.site": "https://jira.example.com/",
                "team_dashboard.url": "http://web.example.com:8080/",
            },
            {},
            (
                ("jira.url", rt.VALUE),
                ("atlassian.site", rt.VALUE),
                ("team_dashboard.url", rt.VALUE),
            ),
        ),
        (
            "the toolkit in another folder",
            {"paths.toolkit": "/srv/monitor"},
            {},
            (("paths.toolkit", rt.INTERNAL),),
        ),
        (
            "keys merged into another",
            {"tick_interval_min": 15, "loop_log_max_mb": 50},
            {},
            (("tick_interval_min", rt.VALUE), ("loop_log_max_mb", rt.INTERNAL)),
        ),
        (
            "Claude's effort",
            {"claude.effort": "max"},
            {},
            (("audit", rt.EFFORT), ("claude.effort", rt.INTERNAL)),
        ),
        (
            "nonessential traffic on",
            {},
            {"nonessential_traffic": "0"},
            ((rt.NONESSENTIAL, rt.VALUE),),
        ),
        (
            "a blank draft stage",
            {"draft_stage": ""},
            {},
            (("draft_stage", rt.INTERNAL),),
        ),
        (
            "the admin dashboard published on every address",
            {"admin_dashboard.publish_host": "0.0.0.0"},
            {},
            (("admin_dashboard.publish_host", rt.INTERNAL),),
        ),
        (
            "no address for the admin dashboard",
            {"admin_dashboard.publish_host": DELETE},
            {},
            (("admin_dashboard.publish_host", rt.INTERNAL),),
        ),
    )

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.shape = production_fixture()
        cls.values = translated()
        cls.rows = rt.rows(rt.DIFFERENCES)

    def test_the_fixtures_differ_only_where_the_list_says_why(self):
        """The fixture, the second fixture and each variant: each difference a row lists, the differences the
        variant makes among them. Each row says why."""
        for name, changes, shape_changes, made in self.VARIANTS:
            with self.subTest(variant=name):
                cfg = changed(self.cfg, changes)
                shape = self.shape._replace(**shape_changes)
                found = rt.compare(cfg, shape, translated(cfg, shape))
                self.assertEqual(rt.unlisted(found, self.rows), [])
                self.assertLessEqual(set(made), {(d.key, d.kind) for d in found})
        for row in self.rows:
            self.assertTrue(row.why.strip(), row.keys)

    def test_the_admin_dashboards_address_is_compared_with_the_deploy_plans(self):
        """The admin dashboard's address, which the file does not hold, is compared with the one the deploy plan
        publishes it on (rt.planned): 127.0.0.1 (container_layout.LOOPBACK). Production's 127.0.0.1, the fixture's,
        makes no difference; another address is listed, the plan's on the file's side. A plan that publishes the
        dashboard on another address is not listed."""
        key = "admin_dashboard.publish_host"
        loopback, every = container_layout.LOOPBACK, container_layout.EVERY_ADDRESS
        self.assertEqual(rt.planned(self.values), {key: loopback})
        self.assertEqual(rt.planned({}), {key: loopback})
        found = rt.compare(self.cfg, self.shape, self.values)
        self.assertNotIn(key, [d.key for d in found])
        cfg = changed(self.cfg, {key: every})
        found = rt.compare(cfg, self.shape, translated(cfg))
        self.assertIn(rt.Difference(key, rt.INTERNAL, every, loopback), found)
        self.assertEqual(rt.unlisted(found, self.rows), [])
        real = deploy_plan.env_values

        def elsewhere(*args):
            return dict(real(*args), RKMON_ADMIN_HOST=every)

        with mock.patch.object(deploy_plan, "env_values", elsewhere):
            found = rt.compare(self.cfg, self.shape, self.values)
        self.assertEqual(
            rt.unlisted(found, self.rows),
            [rt.Difference(key, rt.INTERNAL, loopback, every)],
        )

    def test_a_blank_or_missing_container_name_is_the_one_layout_1_ran(self):
        """Production's config.json with no deployment.project, or a blank one: production ran the container's name
        layout 1's rule gives it (from_layout1.CONTAINER_NAME), which the file writes, so no difference names it, and
        no row lists one."""
        for name, project in (("none", DELETE), ("blank", ""), ("spaces", "  ")):
            with self.subTest(project=name):
                cfg = changed(self.cfg, {"deployment.project": project})
                values = translated(cfg)
                self.assertEqual(
                    deploy_tree.get(values, "container.name"),
                    from_layout1.CONTAINER_NAME,
                )
                found = rt.compare(cfg, self.shape, values)
                self.assertEqual(rt.unlisted(found, self.rows), [])
                self.assertNotIn("deployment.project", [d.key for d in found])
        self.assertEqual(
            [row for row in self.rows if "deployment.project" in row.keys], []
        )

    def test_what_production_ran_by_a_rule_of_layout_1_is_no_difference(self):
        """Production's config.json as production ran it (rt.ran), which the file writes: a blank image, or its blank
        name or tag, is the image production ran; the develop guard's labels on a red CI are those it put, the
        companion label with them, and a blank stability label is its own; an update source on GitHub, in any form,
        is the repository the file names; a blank delivery is "publish", and a requests file of false names none;
        production did not read the block of the delivery it did not run; and the drafting model's key is in the
        secret it names, or the local worker's own, by the name the one-time move copies it to. So no difference names
        the key, and no row lists one."""
        updates = {"update.repo": self.URL, "update.branch": "main"}
        served = {"team_dashboard.mode": "serve"}
        drafts = {"reviewer": "worker"}
        for name, changes, key in (
            ("the drafting model's key", drafts, "worker.api_key_secret"),
            (
                "a key's secret of its own",
                dict(drafts, **{"worker.api_key_secret": "gw_key"}),
                "worker.api_key_secret",
            ),
            (
                "a blank key's secret",
                dict(drafts, **{"worker.api_key_secret": " "}),
                "worker.api_key_secret",
            ),
            ("a blank image", {}, "deployment.image"),
            ("no tag", {"deployment.image": "rocke-monitor"}, "deployment.image"),
            (
                "a blank tag",
                {"deployment.image": "rocke-monitor:"},
                "deployment.image",
            ),
            ("a tag alone", {"deployment.image": ":v1"}, "deployment.image"),
            ("the guard's label", {}, "incident_issue.labels"),
            ("no labels", {"incident_issue.labels": []}, "incident_issue.labels"),
            (
                "labels of its own",
                {"incident_issue.labels": ["breaks", "x"]},
                "incident_issue.labels",
            ),
            (
                "the companion too",
                {"incident_issue.labels": ["breaks", "ci-red"]},
                "incident_issue.labels",
            ),
            (
                "a blank stability label",
                {"incident_issue.stability_label": ""},
                "incident_issue.stability_label",
            ),
            ("a GitHub address", updates, "update.repo"),
            (
                "<org>/<repo>",
                dict(updates, **{"update.repo": "ROCm/sample-tool"}),
                "update.repo",
            ),
            ("a blank delivery", {"team_dashboard.mode": ""}, "team_dashboard.mode"),
            (
                "a requests file of false",
                {"publish.html.requests": False},
                "publish.html.requests",
            ),
            (
                "the team dashboard's public port",
                {"team_dashboard.public_port": 8080},
                "team_dashboard.public_port",
            ),
            (
                "the team dashboard's publish address",
                {"team_dashboard.publish_host": "0.0.0.0"},
                "team_dashboard.publish_host",
            ),
            ("a served page's web host", served, "publish.html.host"),
            ("a served page's place", served, "publish.html.dest"),
            ("a served page's requests file", served, "publish.html.requests"),
        ):
            with self.subTest(production=name):
                cfg = changed(self.cfg, changes)
                found = rt.compare(cfg, self.shape, translated(cfg))
                self.assertEqual(rt.unlisted(found, self.rows), [])
                self.assertNotIn(key, [d.key for d in found])
        self.assertEqual(
            [row for row in self.rows if set(row.keys) & set(rt.AS_LAYOUT1_RAN)], []
        )

    def test_a_key_renames_drops_is_no_difference(self):
        """A key of production's config.json that renames.DROPPED drops (the JIRA MCP server's, the host's ROCm):
        production is read in today's names (rt.ran, renames.migrate), which drop it, blank as in the fixture or set,
        so no difference names it, and no row lists one; the translation's report says why each set one goes.
        """
        dropped = tuple(renames.DROPPED)
        set_ = {
            "jira.mcp_enabled": True,
            "jira.mcp_server": "jira-mcp",
            "jira.mcp_url": "https://jira.example.com/mcp",
            "rocm.mode": "host",
            "rocm.host_dir": "/opt/rocm",
        }
        self.assertEqual(set(set_), set(dropped))
        for name, cfg in (("blank", self.cfg), ("set", changed(self.cfg, set_))):
            with self.subTest(production=name):
                found = rt.compare(cfg, self.shape, translated(cfg))
                self.assertEqual(rt.unlisted(found, self.rows), [])
                self.assertEqual(
                    [
                        d.key
                        for d in found
                        if d.key in dropped
                        or d.key.startswith(tuple(f"{k}." for k in dropped))
                    ],
                    [],
                )
        self.assertEqual(
            [row.keys for row in self.rows if set(row.keys) & set(dropped)], []
        )

    def test_a_change_of_what_production_ran_is_not_listed(self):
        """The file changed where production's side is read as production ran it (rt.ran): the difference is not
        listed, by its key. Another tag, over a blank image or over a tag that begins the file's; another develop
        label; another branch or repository to update from, or a repository with no branch, which the file cannot
        carry; the web host, the login, the site folder or the publish block gone, or priority requests off, while the
        page is published; another delivery or stability label over production's blank one.
        """
        updates = {"update.repo": self.URL, "update.branch": "main"}
        team, guard = tp.TEAM, tp.GUARD
        for name, changes, edits, key in (
            ("another tag", {}, {"image.tag": "other"}, "deployment.image"),
            (
                "a longer tag",
                {"deployment.image": "team/monitor:1"},
                {"image.tag": "10"},
                "deployment.image",
            ),
            (
                "another label",
                {},
                {f"{guard}.ci_red_extra_labels": ["ci-red", "x"]},
                "incident_issue.labels",
            ),
            ("another branch", updates, {"updates.branch": "develop"}, "update.branch"),
            (
                "another repository",
                updates,
                {"updates.repository": "other-org/other"},
                "update.repo",
            ),
            ("no branch", dict(updates, **{"update.branch": ""}), {}, "update.repo"),
            ("no web host", {}, {f"{team}.publish.web_host": ""}, "publish.html.host"),
            ("no login", {}, {f"{team}.publish.username": ""}, "publish.html.user"),
            (
                "no site folder",
                {},
                {f"{team}.publish.site_folder": ""},
                "publish.html.site",
            ),
            ("no publish block", {}, {f"{team}.publish": {}}, "publish.html.host"),
            (
                "no priority requests",
                {},
                {f"{team}.priority_requests.enabled": False},
                "publish.html.requests",
            ),
            (
                "another delivery",
                {"team_dashboard.mode": ""},
                {f"{team}.delivery": "serve"},
                "team_dashboard.mode",
            ),
            (
                "another stability label",
                {"incident_issue.stability_label": ""},
                {f"{guard}.stability_label": "other"},
                "incident_issue.stability_label",
            ),
        ):
            with self.subTest(change=name):
                cfg = changed(self.cfg, changes)
                values = edited(translated(cfg), edits)
                left = rt.unlisted(rt.compare(cfg, self.shape, values), self.rows)
                self.assertIn((key, rt.VALUE), [(d.key, d.kind) for d in left])

    def test_an_address_or_a_served_port_the_file_changes_is_not_listed(self):
        """The rows of an address the file writes without its closing slash, and of a served page's port that 0 or a
        blank stands for, hold for that alone: the JIRA address or the public address left blank or cut short in the
        file, and a served page's own port changed to the default, are not listed."""
        team = tp.TEAM
        served = {"team_dashboard.mode": "serve", "team_dashboard.public_port": 8080}
        short = "https://dashboards.example.com"
        for name, changes, edits, key in (
            (
                "no JIRA address",
                {},
                {"integrations.atlassian_cli.jira_url": ""},
                "jira.url",
            ),
            ("no public address", {}, {f"{team}.public_url": ""}, "publish.html.url"),
            (
                "a public address cut short",
                {},
                {f"{team}.public_url": short},
                "publish.html.url",
            ),
            (
                "the default port",
                served,
                {f"{team}.serve.port": 18615},
                "team_dashboard.public_port",
            ),
        ):
            with self.subTest(change=name):
                cfg = changed(self.cfg, changes)
                values = edited(translated(cfg), edits)
                left = rt.unlisted(rt.compare(cfg, self.shape, values), self.rows)
                self.assertIn((key, rt.VALUE), [(d.key, d.kind) for d in left])

    def test_the_local_worker_s_settings_are_listed_only_while_no_model_drafts(self):
        """The row of the local worker's settings holds while the file binds no drafting model, for a setting
        the file leaves out: on the fixture, which binds none, each one production sets is at rt.NO_DRAFTING_MODEL
        on the file's side, and listed. On the second fixture, which binds the local worker, its address left blank,
        or its key's secret renamed or left blank, is not listed, by its key; nor, on the fixture, is a binding no
        pass uses, at another address."""
        found = rt.compare(self.cfg, self.shape, self.values)
        worker = [
            d for d in found if d.kind == rt.VALUE and d.key.startswith("worker.")
        ]
        self.assertTrue(worker)
        self.assertEqual({d.after for d in worker}, {rt.NO_DRAFTING_MODEL})
        self.assertEqual(rt.unlisted(worker, self.rows), [])
        second = second_fixture()
        bound = f"model_bindings.{deploy_tree.WORKER}"
        binding = deploy_tree.get(translated(second), bound)
        unused = dict(binding, base_url="https://other.example.com/v1")
        for name, cfg, edits, key in (
            ("no address", second, {f"{bound}.base_url": ""}, "worker.url"),
            (
                "another secret",
                second,
                {f"{bound}.api_key_secret_name": "other_api_key"},
                "worker.api_key_secret",
            ),
            (
                "no secret",
                second,
                {f"{bound}.api_key_secret_name": ""},
                "worker.api_key_secret",
            ),
            ("a binding no pass uses", self.cfg, {bound: unused}, "worker.url"),
        ):
            with self.subTest(change=name):
                values = edited(translated(cfg), edits)
                left = rt.unlisted(rt.compare(cfg, self.shape, values), self.rows)
                self.assertIn((key, rt.VALUE), [(d.key, d.kind) for d in left])

    def test_line_1_differs_by_its_model_and_effort_and_the_name_by_its_own_key(self):
        """Line 1 of a review (rt.FIRST_LINES) is compared apart from the automated reviewer name it gives. A line that
        gives another model or effort than today's is a difference no row lists, even where the model ids and the
        efforts sent are the same: the verifier's, posted with no first-pass drafter, on the fixture, and the drafter's
        on the second fixture, which binds a drafting model. The drafter's line is listed only where the file binds
        none, by a row whose why gives the name; the verifier's never is. The name is a value of its own: a name the
        file sets, or another container name, is a value difference no row lists, and line 1 differs as on the fixture
        alone.
        """
        drafter = rt.Difference(
            "review_first_line",
            rt.LINE_1,
            marked("drafter-model-a", config.MODEL_DEFAULT),
            "",
        )
        row = rt.listed(drafter, self.rows)
        self.assertEqual(row.keys, ("review_first_line",))
        self.assertIn(reviewer_name.NAME, row.why)
        opus = marked("Claude Opus", "high")
        self.assertIsNone(
            rt.listed(
                rt.Difference("review_first_line_no_draft", rt.LINE_1, opus, ""),
                self.rows,
            )
        )
        verifier_named = changed(self.cfg, {"verifier_name": "Claude Opus 5.5"})
        second = second_fixture()
        bound = f"model_bindings.{deploy_tree.WORKER}"
        for name, cfg, values, want, besides in (
            (
                "the verifier on another model",
                self.cfg,
                edited(self.values, {f"{VERIFIER}.model": "sonnet"}),
                rt.Difference(
                    "review_first_line_no_draft",
                    rt.LINE_1,
                    opus,
                    marked("Claude Sonnet", "high"),
                ),
                {("models.audit", rt.VALUE), ("audit", rt.MODEL_ID)},
            ),
            (
                "the verifier's model by its id",
                self.cfg,
                edited(self.values, {f"{VERIFIER}.model": "claude-opus-5-5"}),
                rt.Difference(
                    "review_first_line_no_draft",
                    rt.LINE_1,
                    opus,
                    marked("Claude Opus 5.5", "high"),
                ),
                {("models.audit", rt.VALUE)},
            ),
            (
                "the verifier at another effort",
                self.cfg,
                edited(self.values, {f"{VERIFIER}.effort": "low"}),
                rt.Difference(
                    "review_first_line_no_draft",
                    rt.LINE_1,
                    opus,
                    marked("Claude Opus", "low"),
                ),
                {("verify", rt.EFFORT)},
            ),
            (
                "production's own verifier name",
                verifier_named,
                translated(verifier_named),
                rt.Difference(
                    "review_first_line_no_draft",
                    rt.LINE_1,
                    marked("Claude Opus 5.5", "high"),
                    opus,
                ),
                set(),
            ),
            (
                "the drafter on another model",
                second,
                edited(translated(second), {f"{bound}.model_id": "drafter-model-b"}),
                rt.Difference(
                    "review_first_line",
                    rt.LINE_1,
                    marked("drafter-model-a", config.MODEL_DEFAULT),
                    marked("drafter-model-b", config.MODEL_DEFAULT),
                ),
                {("worker.model", rt.VALUE), ("first_pass", rt.MODEL_ID)},
            ),
        ):
            with self.subTest(change=name):
                found = rt.compare(cfg, self.shape, values)
                self.assertIn(want, found)
                self.assertEqual(
                    {(d.key, d.kind) for d in rt.unlisted(found, self.rows)},
                    {(want.key, want.kind), *besides},
                )
        for name, values, want in (
            (
                "a name the file sets",
                edited(self.values, {reviewer_name.NAME: "Platform team"}),
                rt.Difference(reviewer_name.NAME, rt.VALUE, "", "Platform team"),
            ),
            (
                "another container name",
                edited(self.values, {"container.name": "other-monitor"}),
                rt.Difference(
                    reviewer_name.CONTAINER_NAME, rt.VALUE, "rkmon", "other-monitor"
                ),
            ),
        ):
            with self.subTest(change=name):
                found = rt.compare(self.cfg, self.shape, values)
                self.assertIn(want, rt.unlisted(found, self.rows))
                self.assertEqual([d for d in found if d.kind == rt.LINE_1], [drafter])

    def test_the_alerts_production_sends_and_the_file_does_not_are_listed(self):
        """The alerts whose switch is off while the other is on: the needs-you alerts in production's pattern,
        chat-alerts on and needs-you-alerts off (the fixture's), and the develop-break alerts the other way round.
        Production sends them, as one switch sends both alerts, and the file, which keeps production's two switches,
        does not. The round trip shows the difference, a row of the intended differences lists it, and the
        translation's report says what stops. The row holds for that alone: with both of production's switches on, a
        file that turns either alert off is not listed, by its switch; nor are the drift notices production posts
        whatever its feature map says, which a file with the map's switch off stops."""
        for changes, stop, sent in (
            ({}, "needs-you-alerts", "the develop-break alerts"),
            (
                {"features.chat-alerts": False, "features.needs-you-alerts": True},
                "chat-alerts",
                "the needs-you alerts",
            ),
        ):
            with self.subTest(stop=stop):
                cfg = changed(self.cfg, changes)
                done = tp.translate(cfg, self.shape)
                found = rt.compare(cfg, self.shape, read_back(done.deploy_file))
                difference = rt.Difference(stop, rt.FEATURE, True, False)
                self.assertEqual(
                    [d for d in found if d.kind == rt.FEATURE], [difference]
                )
                self.assertEqual(
                    rt.listed(difference, self.rows).keys,
                    ("needs-you-alerts", "chat-alerts"),
                )
                self.assertIn(
                    f"today's monitor sends both while either is on; this file sends only {sent}",
                    flat("\n".join(done.report)),
                )
        both = changed(self.cfg, {"features.needs-you-alerts": True})
        for alert in ("needs_you", "develop_breaks"):
            with self.subTest(off=alert):
                values = edited(
                    translated(both), {f"{from_layout1.ALERTS}.{alert}.enabled": False}
                )
                left = rt.unlisted(rt.compare(both, self.shape, values), self.rows)
                self.assertIn(
                    (f"features.alerts.{alert}", rt.VALUE),
                    [(d.key, d.kind) for d in left],
                )
        drift = changed(self.cfg, {"features.drift-notices": False})
        left = rt.unlisted(rt.compare(drift, self.shape, translated(drift)), self.rows)
        self.assertIn(rt.Difference("drift-notices", rt.FEATURE, True, False), left)

    def test_a_blank_page_stops_the_translation_so_no_row_lists_it(self):
        """A blank or missing page (publish.html.dest), which today's monitor copies to the home folder of the web
        host's user, never to the page the site folder gives: the translation stops on it, so the round trip has not
        run, and the derived keys' row never lists the page's difference."""
        for name, dest in (("blank", ""), ("missing", DELETE)):
            with self.subTest(page=name):
                stopped = tp._round_trip(
                    changed(self.cfg, {"publish.html.dest": dest}), self.shape
                )
                self.assertEqual(
                    [s["found"].startswith(f"{tp.PAGE} ") for s in stopped["stops"]],
                    [True],
                )
                self.assertNotIn("differences", stopped)

    def test_a_difference_of_each_kind_no_row_says_why_of_is_not_listed(self):
        """A change of each kind (`each_kind_changed`, and a dearer price for the week): the difference it makes is
        not listed, by its key and its kind; nor is the value the change sets in the file, where it sets one.
        """
        for production, values, want in each_kind_changed(self.cfg, self.values):
            with self.subTest(kind=want.kind):
                left = rt.unlisted(
                    rt.compare(production, self.shape, values), self.rows
                )
                self.assertIn((want.key, want.kind), [(d.key, d.kind) for d in left])
                self.assertLessEqual({d.kind for d in left}, {want.kind, rt.VALUE})
        dearer = edited(self.values, {f"{PRICES}.claude-opus-5-5.cache_read": 0.6})
        with tempfile.TemporaryDirectory() as d:
            usage = recorded_week(pathlib.Path(d), rt.running(self.values, self.cfg))
        left = rt.unlisted(rt.compare(self.cfg, self.shape, dearer, usage), self.rows)
        self.assertIn(
            (f"{rt.USAGE}.{rt.WEEK}", rt.PRICE), {(d.key, d.kind) for d in left}
        )

    def test_translate_then_the_inverse_map_gives_back_each_value_no_row_says_why_of(
        self,
    ):
        """On both fixtures, each value of a key the file feeds comes back from the translated file through the
        inverse map, but where a row says why not."""
        for cfg in (self.cfg, second_fixture()):
            with self.subTest(reviewer=cfg.get("reviewer")):
                back = rt.inverse(translated(cfg)).config
                found = rt.values_compared(cfg, back)
                self.assertTrue(found)
                self.assertEqual(rt.unlisted(found, self.rows), [])
                self.assertEqual(
                    deploy_tree.get(back, "jira.project"),
                    deploy_tree.get(rt.today(cfg), "jira.project"),
                )

    def test_the_evidence_check_s_budget_stays_600_seconds_and_the_offload_s_900_goes(
        self,
    ):
        """The first review pass's evidence check's time budget (bindings.drafter_settings, which
        rkmon_lib/evidence.py reads): the fixture sets none, so 600 seconds on both sides, and no difference. The
        offload's 900 seconds (offload.check_total_s, config.defaults.json's and the fixture's) goes with the
        offload: a listed difference. The first review pass's at 900 is not listed."""
        for side in (rt.today(self.cfg), rt.running(self.values, self.cfg)):
            self.assertEqual(bindings.drafter_settings(side)["check_total_s"], 600)
        found = by_key(rt.compare(self.cfg, self.shape, self.values))
        self.assertFalse([k for k, _kind in found if k.startswith("drafter.")])
        self.assertEqual(deploy_tree.get(self.cfg, "offload.check_total_s"), 900)
        self.assertEqual(config.load_defaults()["offload"]["check_total_s"], 900)
        row = rt.listed(
            rt.Difference("offload.check_total_s", rt.INTERNAL, 900, MISSING),
            self.rows,
        )
        self.assertEqual(row.keys, ("offload.check_total_s",))
        self.assertIn("600 seconds", row.why)
        self.assertIsNone(
            rt.listed(
                rt.Difference("drafter.check_total_s", rt.INTERNAL, 600, 900),
                self.rows,
            )
        )

    def test_the_rows_the_owner_agrees_come_from_a_file_of_the_same_form(self):
        """A package key production sets to another value is on no row of the package's list; on a row the owner
        agreed (an invented file here, as the real one is a deployment's own and stays outside the package), it is
        listed at the value agreed alone."""
        agreed = rt.rows(AGREED)
        for value, on_it in ((9000, True), (12000, False)):
            with self.subTest(value=value):
                cfg = changed(self.cfg, {"huge_pr_lines": value})
                found = rt.compare(cfg, self.shape, translated(cfg))
                own = rt.Difference("huge_pr_lines", rt.INTERNAL, value, 8000)
                self.assertEqual(rt.unlisted(found, self.rows), [own])
                self.assertEqual(
                    rt.unlisted(found, [*self.rows, *agreed]), [] if on_it else [own]
                )

    def test_each_key_the_translation_leaves_out_is_listed(self):
        """Each key the translation's report says goes, and each it says differs by intent
        (tools/translate_production.py REMOVED_KEYS, WHY_DROPPED and INTENDED_KEYS): its difference is listed, whatever
        production's value."""
        for key in (*tp.REMOVED_KEYS, *tp.WHY_DROPPED, *tp.INTENDED_KEYS):
            for value in ("x", 7, ["x"], {"x": 1}):
                with self.subTest(key=key, value=value):
                    gone = rt.Difference(key, rt.kind_of(key), value, MISSING)
                    self.assertIsNotNone(rt.listed(gone, self.rows))

    def test_each_row_names_keys_of_its_kind(self):
        """A row's keys can differ in its kind: a value's, keys the file feeds (or the sections that hold them) and
        the Claude session's values; an internal value's, keys it does not feed, never line 1's; an effort's, the
        roles' efforts; a feature's, the capabilities; line 1's, line 1's keys (rt.FIRST_LINES).
        """
        held = deploy_tree.held()
        for row in self.rows:
            for key in row.keys:
                with self.subTest(key=key, kind=row.kind):
                    if row.kind == rt.EFFORT:
                        self.assertIn(key, config.EFFORT_ROLES)
                    elif row.kind == rt.FEATURE:
                        self.assertIn(key, features.CAPABILITIES)
                    elif row.kind == rt.LINE_1:
                        self.assertIn(key, rt.FIRST_LINES)
                    elif key.isupper():
                        self.assertEqual(row.kind, rt.VALUE)
                        self.assertTrue(
                            [
                                v
                                for v in rt.SESSION_VALUES
                                if fnmatch.fnmatchcase(v, key)
                            ]
                        )
                    elif "*" not in key and row.kind == rt.VALUE:
                        self.assertTrue(
                            rt.kind_of(key) == rt.VALUE
                            or [h for h in held if h.startswith(key + ".")]
                        )
                    elif "*" not in key:
                        self.assertEqual(row.kind, rt.INTERNAL)
                        self.assertEqual(rt.kind_of(key), rt.INTERNAL)
                        self.assertNotIn(key, rt.FIRST_LINES)

    def test_the_list_and_the_fixtures_pass_the_privacy_check(self):
        files = [
            rt.DIFFERENCES,
            AGREED,
            DATA / "production_config.json",
            DATA / "production_claude_env_shape.json",
        ]
        with tempfile.TemporaryDirectory() as d:
            deny = pathlib.Path(d) / "deny.txt"
            deny.write_text("invented" + "-deny-value-r\n", encoding="utf-8")
            text, ok, found = privacy.run(files, deny, environ={})
        self.assertEqual(found, [])
        self.assertTrue(ok, text)

    def test_the_list_s_texts_pass_the_docs_lints(self):
        self.assertEqual(
            doc_lints.run(ROOT, [rt.DIFFERENCES.relative_to(ROOT).as_posix()]), []
        )


class TestTheListsForm(unittest.TestCase):
    """rt.rows, rt.listed and the conditions a row may have: on rows made here."""

    ROW = {"keys": ["k"], "kind": rt.VALUE, "why": "w"}

    def test_a_list_file_that_breaks_the_form_is_refused_naming_the_file_and_the_row(
        self,
    ):
        row = self.ROW
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "list.json"
            for text, says in (
                ("{", "not valid JSON"),
                (json.dumps({"rows": [row]}), "no list of differences"),
                (json.dumps({"differences": row}), "no list of differences"),
            ):
                with self.subTest(says=says, text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "^" + str(path) + ": "):
                        rt.rows(path)
                    with self.assertRaisesRegex(ValueError, says):
                        rt.rows(path)
            for bad, says in (
                (["k"], "a row is a JSON object"),
                (dict(row, note="n"), "no field note"),
                ({"keys": ["k"], "kind": rt.VALUE}, "no why"),
                (dict(row, why=""), "no why"),
                (dict(row, keys="k"), "keys is a list"),
                (dict(row, keys=[""]), "keys is a list"),
                (dict(row, kind="price"), "kind 'price' is none of"),
                (dict(row, why=7), "why is text"),
                (dict(row, owner="someone"), "owner 'someone' is none of"),
                (dict(row, blank="after"), "blank is a list of sides"),
                (dict(row, blank=["before"]), "blank is a list of sides"),
                (dict(row, default=["both"]), "default is a list of sides"),
                (dict(row, extends="both"), "extends is one side"),
                (dict(row, extends="production", by=["/"]), "by is text"),
                (dict(row, by="/"), "by goes with extends"),
            ):
                with self.subTest(says=says):
                    path.write_text(
                        json.dumps({"differences": [row, bad]}), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError) as caught:
                        rt.rows(path)
                    self.assertTrue(
                        str(caught.exception).startswith(f"{path}, row 2: "),
                        caught.exception,
                    )
                    self.assertIn(says, str(caught.exception))
            # a problem names the file as the caller shows it, when it gives a way
            for text, says in (
                ("{", "~/list.json: not valid JSON"),
                (json.dumps({"differences": [row, ["k"]]}), "~/list.json, row 2: "),
            ):
                with self.subTest(shown=says):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError) as caught:
                        rt.rows(path, "~/list.json")
                    self.assertTrue(str(caught.exception).startswith(says))
            path.write_text(
                json.dumps(
                    {
                        "differences": [
                            dict(
                                row,
                                owner=rt.NO_OWNER,
                                production=0,
                                after=None,
                                blank=["after"],
                                default=["production"],
                                extends="production",
                                by="/",
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                rt.rows(path),
                [
                    rt.Row(
                        ("k",),
                        rt.VALUE,
                        "w",
                        rt.NO_OWNER,
                        (("production", 0), ("after", None)),
                        ("after",),
                        ("production",),
                        "production",
                        "/",
                    )
                ],
            )

    def test_the_row_that_names_the_key_lists_it_before_one_it_is_under_or_a_pattern(
        self,
    ):
        exact = rt.Row(("a.b",), rt.VALUE, "the key")
        parent = rt.Row(("a",), rt.VALUE, "a key it is under")
        pattern = rt.Row(("a.*",), rt.VALUE, "a pattern")
        d = rt.Difference("a.b", rt.VALUE, 1, 2)
        self.assertEqual(rt.listed(d, [pattern, parent, exact]), exact)
        self.assertEqual(rt.listed(d, [pattern, parent]), parent)
        self.assertEqual(rt.listed(d, [pattern]), pattern)
        twin = rt.Row(("x", "a.b"), rt.VALUE, "the key too")
        self.assertEqual(rt.listed(d, [exact, twin]), exact)
        self.assertEqual(rt.listed(d, [twin, exact]), twin)
        # of its kind; a key under a name, not one that begins with its text
        self.assertIsNone(rt.listed(d._replace(kind=rt.INTERNAL), [exact, parent]))
        self.assertIsNone(rt.listed(d._replace(key="a.bc"), [exact]))
        self.assertIsNone(rt.listed(d._replace(key="ab"), [parent, pattern]))
        self.assertEqual(
            rt.unlisted([d, d._replace(key="c")], [exact]), [d._replace(key="c")]
        )

    def test_each_condition_must_hold(self):
        def holds(row, *differences):
            return [rt.listed(d, [row]) is row for d in differences]

        d = rt.Difference("k", rt.VALUE)
        # blank: no value, null, "", [] or {}; not 0 or false
        self.assertEqual(
            holds(
                rt.Row(("k",), rt.VALUE, "w", blank=("production",)),
                *(
                    d._replace(production=v)
                    for v in (MISSING, None, "", [], {}, 0, False)
                ),
            ),
            [True] * 5 + [False] * 2,
        )
        # a value, as JSON writes it: true is not 1
        self.assertEqual(
            holds(
                rt.Row(("k",), rt.VALUE, "w", values=(("after", 1),)),
                *(d._replace(after=v) for v in (1, True, "1", MISSING)),
            ),
            [True, False, False, False],
        )
        # extends: a text that begins with the other's text, a list that begins with the other's items; never a blank
        # one's, which every text and list begins with
        self.assertEqual(
            holds(
                rt.Row(("k",), rt.VALUE, "w", extends="after"),
                d._replace(production="ab", after="abc"),
                d._replace(production=["a"], after=["a", "b"]),
                d._replace(production="abc", after="ab"),
                d._replace(production=["b"], after=["a", "b"]),
                d._replace(production=1, after=12),
                d._replace(production="", after="abc"),
                d._replace(production=[], after=["a"]),
                d._replace(production=MISSING, after="abc"),
            ),
            [True, True, False, False, False, False, False, False],
        )
        # by: the text that more is, exactly
        self.assertEqual(
            holds(
                rt.Row(("k",), rt.VALUE, "w", extends="production", by="/"),
                d._replace(
                    production="https://x.example.com/", after="https://x.example.com"
                ),
                d._replace(
                    production="https://x.example.com/a", after="https://x.example.com"
                ),
                d._replace(
                    production="https://x.example.com//", after="https://x.example.com"
                ),
                d._replace(
                    production="https://x.example.com", after="https://x.example.com/"
                ),
                d._replace(production="/", after=""),
                d._replace(production=["a", "/"], after=["a"]),
            ),
            [True, False, False, False, False, False],
        )
        # default: the key's default, an entry's in its key's; the registry's owner of the key
        self.assertEqual(
            holds(
                rt.Row(("huge_pr_lines",), rt.INTERNAL, "w", default=("after",)),
                rt.Difference("huge_pr_lines", rt.INTERNAL, 1, 8000),
                rt.Difference("huge_pr_lines", rt.INTERNAL, 8000, 1),
            ),
            [True, False],
        )
        self.assertEqual(rt.default_of("huge_pr_lines"), 8000)
        self.assertEqual(rt.default_of("drafter.check_total_s"), 600)
        self.assertIs(rt.default_of("invented_key"), MISSING)
        derived = rt.Row(("*",), rt.INTERNAL, "w", owner="derived")
        none = rt.Row(("*",), rt.INTERNAL, "w", owner=rt.NO_OWNER)
        for key, owner in (
            ("draft_stage", "derived"),
            ("huge_pr_lines", "package"),
            ("invented_key", rt.NO_OWNER),
        ):
            with self.subTest(key=key):
                self.assertEqual(rt.owner_of(key), owner)
                at = rt.Difference(key, rt.INTERNAL, 1, 2)
                self.assertEqual(
                    holds(derived, at) + holds(none, at),
                    [owner == "derived", owner == rt.NO_OWNER],
                )


def flat(text: str) -> str:
    """TEXT with each run of spaces and line breaks one space: a report's rows and an outcome, however they wrap."""
    return " ".join(text.split())


class TestTheInside(unittest.TestCase):
    """--inside --round-trip, in process as the one-off container runs it: config.json in the monitor's home,
    claude_env in its secrets folder and the monitor's transcripts in Claude's config folder (the spend tests'
    week), each read and never changed; the round trip printed as one JSON object; claude_env's key never printed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.tmp.name)
        self.home, self.claude = d / "home", d / "claude"
        secrets = self.home / "secrets"
        secrets.mkdir(parents=True, mode=0o700)
        self.cfg = changed(production_fixture()[0], {"paths.secrets_dir": str(secrets)})
        for path, data in (
            (self.home / "config.json", self.cfg),
            (secrets / "claude_env", claude_env()),
        ):
            path.write_text(json.dumps(data), encoding="utf-8")
            path.chmod(0o600)
        self.root = week_written(self.claude / "projects")
        self.scans = d / "scans"
        environ = {"RKMON_HOME": str(self.home), "CLAUDE_CONFIG_DIR": str(self.claude)}
        patch = mock.patch.dict(os.environ, environ)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def snapshot(self) -> dict:
        """Each file and folder of the home and of Claude's config folder: its bytes and its mode."""
        return {
            str(p): (
                p.read_bytes() if p.is_file() else None,
                stat.S_IMODE(p.stat().st_mode),
            )
            for top in (self.home, self.claude)
            for p in top.rglob("*")
        }

    def test_the_round_trip_of_the_homes_files(self):
        """The differences compare finds between the home's files and the file their translation makes, with the week
        priced both ways; the file itself is not printed, and each difference is on the list: the fixture passes.
        Each step is announced on stderr as it begins, in order, by its name alone.
        """
        before = self.snapshot()
        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            rc = tp.main(["--inside", "--round-trip"])
        self.assertEqual(rc, 0)
        self.assertEqual(
            err.getvalue().splitlines(), [f"{tp.STEP}{name}" for name in tp.TRIP_STEPS]
        )
        self.assertEqual(self.snapshot(), before)
        self.assertNotIn(KEY, out.getvalue())
        values = translated(self.cfg)
        usage = rt.usage_of(self.root, self.scans, rt.running(values, self.cfg))
        want = rt.compare(self.cfg, production_fixture()[1], values, usage)
        answer = json.loads(out.getvalue())
        self.assertEqual(
            answer["differences"], json.loads(json.dumps(rt.to_json(want)))
        )
        self.assertEqual(
            answer["usage"],
            {
                "root": str(self.root),
                "records": len(rt.on_days(usage.today, WEEK)),
                "days": list(WEEK),
            },
        )
        self.assertEqual((answer["deploy_file"], answer["stops"]), ("", []))
        self.assertEqual(rt.unlisted(want, rt.rows(rt.DIFFERENCES)), [])

    def test_a_fault_prints_where_it_is_never_its_message(self):
        """A fault of the round trip's code prints its kind and its place, the last line after the steps begun: its
        message may hold a value."""

        def fault(*_args):
            raise RuntimeError(f"a value: {KEY}")

        with (
            mock.patch.object(rt, "compare", fault),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            rc = tp.main(["--inside", "--round-trip"])
        self.assertEqual((rc, out.getvalue()), (1, ""))
        *begun, last = err.getvalue().splitlines()
        self.assertEqual(begun, [f"{tp.STEP}{name}" for name in tp.TRIP_STEPS])
        self.assertRegex(
            last, r"\Athe round trip failed: RuntimeError at test_round_trip\.py:\d+\Z"
        )
        self.assertNotIn(KEY, err.getvalue())


class TestTheCommand(FakeDockerCase):
    """--round-trip: the round trip in the one-off container, its report and its outcome.
    The one-off container's answer is the fixture's round trip on the spend tests' week, made in
    process as the container makes it (`trip`); a fake docker gives it back, so nothing calls the real docker.
    """

    @classmethod
    def setUpClass(cls):
        cfg, shape = production_fixture()
        with tempfile.TemporaryDirectory() as d:
            claude = pathlib.Path(d) / "claude"
            week_written(claude / "projects")
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(claude)}):
                cls.trip = tp._round_trip(cfg, shape)
        cls.found = rt.from_json(cls.trip["differences"])
        cls.write = f"next: python3 tools/translate_production.py --container {cls.NAME} --out FILE"
        cls.again = (
            f"python3 tools/translate_production.py --container {cls.NAME} --round-trip"
        )

    def setUp(self):
        super().setUp()
        self.answer(self.trip)

    def trip_with(self, *differences) -> dict:
        """The fixture's round trip with DIFFERENCES besides its own."""
        return dict(
            self.trip, differences=self.trip["differences"] + rt.to_json(differences)
        )

    def agreed_file(self, folder=None, mode: int = 0o600) -> pathlib.Path:
        """The invented rows the owner agreed (AGREED) as a file in FOLDER (the test's home), at MODE."""
        path = (folder or self.d) / "agreed.json"
        path.write_text(AGREED.read_text(encoding="utf-8"), encoding="utf-8")
        path.chmod(mode)
        return path

    def test_the_fixture_passes(self):
        """Each difference of the fixture's round trip is on the list: a PASS row for each, by its key and its kind
        (a variable of the Claude session under the name of the older Claude secret, which sets it today), with both
        sides and the reason of the row that lists it, after what it compared and the week it priced; it passes, and
        the next line writes the file. The one-off container runs this tool with --inside --round-trip.
        """
        rc, out, err = self.run_("--container", self.NAME, "--round-trip")
        self.assertEqual((rc, err), (0, ""), out)
        self.assertEqual(
            out.splitlines()[0],
            f"The round trip of {self.NAME}'s settings: each difference, by its key and its kind",
        )
        usage = self.trip["usage"]
        self.assertIn(
            f"Usage: {usage['records']} records under {usage['root']}, priced both ways over the week {WEEK[0]} to "
            f"{WEEK[-1]} PASS ",
            flat(out),
        )
        self.assertEqual(out.count("\n  PASS  "), len(self.found))
        self.assertNotIn("  FAIL  ", out)
        d = by_key(self.found)["publish.html.url", rt.VALUE]
        row = rt.listed(d, rt.rows(rt.DIFFERENCES))
        self.assertIn(
            'PASS publish.html.url (value): production "http://web.example.com:8080/", after '
            f'"http://web.example.com:8080"; listed: {row.why}',
            flat(out),
        )
        d = by_key(self.found)["ANTHROPIC_MODEL", rt.VALUE]
        row = rt.listed(d, rt.rows(rt.DIFFERENCES))
        self.assertIn(
            f'PASS {secrets_registry.CLAUDE_ENV}.ANTHROPIC_MODEL (value): production "set", after (not set); listed: '
            f"{row.why}",
            flat(out),
        )
        self.assertCloses(out, f"PASSED: {len(self.found)} checks {self.write}")
        _inspect, run = self.calls()
        self.assertTrue(
            run.endswith(
                f" {self.IMAGE} -I -B /translation/tools/translate_production.py --inside --round-trip"
            ),
            run,
        )

    def test_no_usage_recorded_fails_naming_where_the_transcripts_are_looked_for(
        self,
    ):
        """With no usage recorded under the transcripts' root, no price is compared: a FAIL row, "Usage: none recorded
        under ROOT", whose fix names where the monitor's transcripts are looked for, then runs it again; no line of a
        week priced. It fails, 1 of the checks, as a difference no row lists does.
        """
        cfg, shape = production_fixture()
        with tempfile.TemporaryDirectory() as d:
            claude = pathlib.Path(d) / "claude"
            (claude / "projects").mkdir(parents=True)
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(claude)}):
                unrecorded = tp._round_trip(cfg, shape)
        root = str(claude / "projects")
        self.assertEqual(unrecorded["usage"], {"root": root, "records": 0, "days": []})
        found = rt.from_json(unrecorded["differences"])
        self.answer(unrecorded)
        rc, out, err = self.run_("--container", self.NAME, "--round-trip")
        self.assertEqual((rc, err), (4, ""), out)
        self.assertEqual(out.count("\n  FAIL  "), 1)
        self.assertNotIn("priced both ways", out)
        self.assertIn(
            f"FAIL Usage: none recorded under {root} fix: the monitor's transcripts are looked for under {root}, the "
            "folder paths.claude_projects names in production's config.json, else $CLAUDE_CONFIG_DIR/projects in its "
            f"container: set paths.claude_projects to their folder, then run {self.again} PASS ",
            flat(out),
        )
        self.assertCloses(
            out,
            f"FAILED: 1 of {len(found) + 1} checks failed; nothing changed (fix each FAIL above) next: {self.again} "
            "--agreed FILE",
        )

    def test_a_difference_of_each_kind_no_row_lists_fails_naming_its_key_and_kind(self):
        """A difference of each kind that no row lists (`each_kind_changed`'s, and a price total of a dearer week): a
        FAIL row names it by its key and its kind, with both sides (a header's key only as the marker) and its fix;
        it fails, 1 of the checks, and the next line runs it again with the file of the rows the owner agrees.
        """
        cfg, _shape = production_fixture()
        wants = [want for _p, _v, want in each_kind_changed(cfg, translated())]
        wants.append(rt.Difference(f"{rt.USAGE}.{rt.WEEK}", rt.PRICE, 29.75, 35.7))
        self.assertEqual({w.kind for w in wants}, set(rt.KINDS))
        again = f"{self.again} --agreed FILE"
        for want in wants:
            with self.subTest(kind=want.kind):
                self.answer(self.trip_with(want))
                rc, out, err = self.run_("--container", self.NAME, "--round-trip")
                self.assertEqual((rc, err), (4, ""), out)
                self.assertEqual(out.count("\n  FAIL  "), 1)
                self.assertIn(
                    f"FAIL {want.key} ({want.kind}): production {tp._shown(want.production, None)}, after "
                    f"{tp._shown(want.after, None)}; not on the list fix: explain it, then agree it, ",
                    flat(out),
                )
                self.assertCloses(
                    out,
                    f"then run {again} FAILED: 1 of {len(self.found) + 1} checks failed; nothing changed (fix each "
                    f"FAIL above) next: {again}",
                )
                if want.kind == rt.HEADER:
                    self.assertIn('after ["Authorization: Bearer <key>"]', flat(out))

    def test_the_rows_the_owner_agreed_list_the_differences_they_name(self):
        """A package key production sets to another value fails the round trip; with --agreed naming the file of the
        rows the owner agreed (invented here), its row lists it, and the round trip passes.
        """
        own = rt.Difference("huge_pr_lines", rt.INTERNAL, 9000, 8000)
        self.answer(self.trip_with(own))
        rc, out, _err = self.run_("--container", self.NAME, "--round-trip")
        self.assertEqual(rc, 4, out)
        self.assertIn(
            "FAIL huge_pr_lines (internal value): production 9000, after 8000; not on the list",
            flat(out),
        )
        path = self.agreed_file()
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--agreed", path
        )
        self.assertEqual((rc, err), (0, ""), out)
        (row,) = rt.rows(AGREED)
        self.assertIn(
            "PASS huge_pr_lines (internal value): production 9000, after 8000; "
            f"listed: {row.why}",
            flat(out),
        )
        self.assertCloses(out, f"PASSED: {len(self.found) + 1} checks {self.write}")

    def test_a_file_of_agreed_rows_that_is_not_private_is_not_read(self):
        """The rows the owner agreed hold a deployment's own values: a file others can read, or one in a git work
        tree, is refused with the row that says why and its fix, before docker runs."""
        path = self.agreed_file(mode=0o644)
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--agreed", path
        )
        self.assertEqual((rc, err), (4, ""))
        self.assertIn(f"  FAIL  {tp.AGREED}: ~/agreed.json has permissions 644: ", out)
        self.assertIn("fix: chmod 600 ~/agreed.json", out)
        self.assertCloses(
            out, "REFUSED: ~/agreed.json is not read (the row above says why)"
        )
        # the refusal as it shows itself: its row, then the line that names the rules
        self.assertIn(f"\n{private_files.RULES}\nREFUSED: ", out)
        repo = self.d / "repo"
        (repo / ".git").mkdir(parents=True)
        path = self.agreed_file(repo)
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--agreed", path
        )
        self.assertEqual((rc, err), (4, ""))
        self.assertIn(
            f"FAIL {tp.AGREED}: ~/repo/agreed.json is inside the git work tree {repo}: ",
            flat(out),
        )
        self.assertCloses(
            out, "REFUSED: ~/repo/agreed.json is not read (the row above says why)"
        )
        self.assertEqual(self.calls(), [])

    def test_a_file_of_agreed_rows_that_breaks_the_list_s_form_is_not_read(self):
        path = self.d / "agreed.json"
        path.write_text(
            json.dumps(
                {"differences": [{"keys": ["huge_pr_lines"], "kind": rt.INTERNAL}]}
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--agreed", path
        )
        self.assertEqual((rc, err), (4, ""))
        self.assertEqual(
            flat(out),
            "REFUSED: ~/agreed.json, row 1: no why fix: give the file the list's form (tools/round_trip.py), then "
            f"run {self.again} --agreed {path}",
        )
        self.assertEqual(self.calls(), [])

    def test_out_writes_the_report_yours_only_and_the_terminal_gets_the_outcome_alone(
        self,
    ):
        """With --out the host writes the report as FILE, at 600: what the terminal gets without --out, but for the
        outcome. The terminal gets the outcome and the next line alone, which name no value of production's.
        """
        _rc, printed, _err = self.run_("--container", self.NAME, "--round-trip")
        report = self.d / "round_trip.txt"
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--out", report
        )
        self.assertEqual((rc, err), (0, ""), out)
        self.assertEqual(stat.S_IMODE(report.stat().st_mode), 0o600)
        text = report.read_text(encoding="utf-8")
        self.assertEqual(
            printed, text + f"PASSED: {len(self.found)} checks\n{self.write}\n"
        )
        self.assertEqual(
            flat(out),
            f"PASSED: {len(self.found)} checks; the report is in ~/round_trip.txt {self.write}",
        )
        self.answer(
            self.trip_with(rt.Difference("jira.project", rt.VALUE, "SAMPLE", "OTHER"))
        )
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--out", report
        )
        self.assertEqual((rc, err), (4, ""), out)
        self.assertEqual(
            flat(out),
            f"FAILED: 1 of {len(self.found) + 1} checks failed; nothing changed (fix each FAIL in "
            f"~/round_trip.txt) next: {self.again} --out {report} --agreed FILE",
        )
        self.assertIn(
            'FAIL jira.project (value): production "SAMPLE", after "OTHER"; not on the list',
            flat(report.read_text(encoding="utf-8")),
        )

    def test_out_where_the_report_may_not_go_is_refused_before_the_round_trip_runs(
        self,
    ):
        repo = self.d / "repo"
        (repo / ".git").mkdir(parents=True)
        report = repo / "round_trip.txt"
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--out", report
        )
        self.assertEqual((rc, err), (4, ""))
        self.assertIn(
            f"FAIL private file: ~/repo/round_trip.txt is inside the git work tree {repo}: ",
            flat(out),
        )
        self.assertCloses(
            out,
            "REFUSED: ~/repo/round_trip.txt is not written, and the round trip has not run (the row above says why)",
        )
        self.assertIn(f"\n{private_files.RULES}\nREFUSED: ", out)
        self.assertFalse(report.exists())
        self.assertEqual(self.calls(), [])

    def test_a_report_place_refused_after_the_round_trip_shows_the_refusal_as_it_is(
        self,
    ):
        """--out passes its check before the round trip runs, and its folder becomes a git work tree while the round
        trip runs (the fake docker makes the .git entry as it answers): the write after it is refused, the refusal
        shown as it shows itself (its row, then the line that names the rules), and nothing is written.
        """
        folder = self.d / "reports"
        folder.mkdir()
        report = folder / "round_trip.txt"
        rows = json.loads(self.answers.read_text(encoding="utf-8"))
        rows[-1] = [*rows[-1], {}, {"makes": str(folder / ".git")}]
        self.answers.write_text(json.dumps(rows), encoding="utf-8")
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--out", report
        )
        self.assertEqual((rc, err), (4, ""), out)
        self.assertIn(
            f"FAIL private file: ~/reports/round_trip.txt is inside the git work tree {folder}: ",
            flat(out),
        )
        self.assertIn(f"\n{private_files.RULES}\nREFUSED: ", out)
        self.assertCloses(
            out,
            "REFUSED: ~/reports/round_trip.txt is not written (the row above says why)",
        )
        self.assertFalse(report.exists())

    def test_a_setting_that_stops_the_translation_fails_the_round_trip(self):
        """A setting the file cannot carry stops the translation, so the round trip cannot run: the translation's
        report with its Stops, then FAILED with the fixes and the command again. With --out the report goes to FILE,
        and the terminal gets the outcome alone."""
        cfg, shape = production_fixture()
        stopped = tp._round_trip(
            changed(cfg, {"features.stability-issues": False}), shape
        )
        self.assertEqual(len(stopped["stops"]), 1)
        self.assertNotIn("differences", stopped)
        self.answer(stopped)
        rc, out, err = self.run_("--container", self.NAME, "--round-trip")
        self.assertEqual((rc, err), (4, ""), out)
        self.assertTrue(out.startswith("\n".join(stopped["report"]) + "\n"))
        self.assertIn(
            "FAILED: 1 setting cannot be carried (each under Stops above, with its fix), so the round trip has not "
            "run; nothing changed fix: ",
            flat(out),
        )
        self.assertCloses(out, f"next: {self.again}")
        report = self.d / "round_trip.txt"
        rc, out, err = self.run_(
            "--container", self.NAME, "--round-trip", "--out", report
        )
        self.assertEqual((rc, err), (4, ""), out)
        self.assertEqual(
            flat(out),
            "FAILED: 1 setting cannot be carried (each under Stops in ~/round_trip.txt, with its fix), so the round "
            f"trip has not run; nothing changed next: {self.again} --out {report}",
        )
        self.assertEqual(
            report.read_text(encoding="utf-8").splitlines(), stopped["report"]
        )

    def test_an_answer_whose_differences_this_tool_does_not_read(self):
        for broken in (
            dict(self.trip, differences={}),
            dict(self.trip, differences=[{"key": "k", "kind": "price"}]),
            dict(self.trip, usage=None),
        ):
            with self.subTest(broken=sorted(broken)):
                self.answer(broken)
                rc, out, _err = self.run_("--container", self.NAME, "--round-trip")
                self.assertEqual(
                    (rc, out.splitlines()),
                    (
                        1,
                        [
                            f"ERROR: the one-off container of {self.NAME}'s image gave differences this tool does "
                            "not read"
                        ],
                    ),
                )

    def in_process(self, one_off, *options, removed=None) -> tuple:
        """The tool with --container run in process, with OPTIONS (--round-trip by default), and a stand-in for
        subprocess.run: docker's inspect of the container answers as the fake docker does (`answer`); ONE_OFF, called
        as subprocess.run is, answers for the one-off container, which it makes by writing ONE_OFF_ID to its --cidfile
        (`made`); REMOVED, called with the arguments after docker, answers each call that removes it or asks of it
        (docker rm -f exits 0 by default). The exit code, what it printed, and each docker call: its arguments after
        docker, and its time limit."""
        inspect = json.dumps(
            {
                "image": self.IMAGE,
                "mounts": self.mounts,
                "ids": ["RKMON_UID=", "RKMON_GID=", None],
            }
        )
        calls = []

        def run(argv, **kwargs):
            calls.append((argv[1:], kwargs["timeout"]))
            if argv[1] == "run":
                return one_off(argv, **kwargs)
            if argv[-1] == self.NAME:
                return subprocess.CompletedProcess(argv, 0, inspect, "")
            code, stdout, stderr = (removed or (lambda _args: (0, "", "")))(argv[1:])
            return subprocess.CompletedProcess(argv, code, stdout, stderr)

        with (
            mock.patch.object(tp.shutil, "which", return_value="docker"),
            mock.patch.object(tp.subprocess, "run", run),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            rc = tp.main(["--container", self.NAME, *(options or ("--round-trip",))])
        return rc, out.getvalue(), calls

    def test_the_round_trip_has_its_own_time_and_says_how_far_it_got(self):
        """The one-off container of the round trip, which reads the monitor's transcripts, may take
        ROUND_TRIP_SECONDS (`in_process`). Past it, an error that says the limit was reached, and how far it got: the
        last step it announced on stderr by then (TRIP_STEPS), and no other line of it; then that the container is
        removed, by the id docker wrote (`made`), and the next line, the command again.
        """
        reached = (
            f"ERROR: the round trip in the one-off container of {self.NAME}'s image reached its time limit, "
            f"{tp.ROUND_TRIP_SECONDS} seconds (ROUND_TRIP_SECONDS in tools/translate_production.py),"
        )
        steps = [f"{tp.STEP}{name}" for name in tp.TRIP_STEPS]
        for name, printed, where in (
            ("nothing printed", None, "before it began"),
            ("an empty stderr", b"", "before it began"),
            (
                "the read of the transcripts",
                "\n".join(steps[:2]).encode(),
                f"while {tp.TRIP_STEPS['usage']}",
            ),
            (
                "the read, among lines that are not steps",
                "\n".join([*steps[:2], f"a value: {KEY}", f"{tp.STEP}{KEY}"]).encode(),
                f"while {tp.TRIP_STEPS['usage']}",
            ),
            ("text", "\n".join(steps) + "\n", f"while {tp.TRIP_STEPS['compare']}"),
        ):
            with self.subTest(stderr=name):

                def hangs(argv, **kwargs):
                    made(argv)
                    raise subprocess.TimeoutExpired(
                        argv, kwargs["timeout"], stderr=printed
                    )

                rc, out, calls = self.in_process(hangs)
                self.assertEqual(
                    (rc, flat(out)),
                    (
                        1,
                        f"{reached} {where}; the container is removed, and nothing changed next: {self.again}",
                    ),
                )
                self.assertEqual(
                    [(args[0], timeout) for args, timeout in calls],
                    [
                        ("container", tp.INSPECT_SECONDS),
                        ("run", tp.ROUND_TRIP_SECONDS),
                        ("rm", tp.INSPECT_SECONDS),
                    ],
                )
                self.assertEqual(calls[-1][0], ["rm", "-f", ONE_OFF_ID])

    def test_a_round_trip_that_stops_shows_its_last_line_never_a_step(self):
        """A one-off container of the round trip that stops (`in_process`) shows the last line it printed on stderr,
        never a step it announced: its error, or, when a step is the last, no message and how far it got; one killed,
        as the kernel does at its memory limit, names the limit too.
        """
        steps = [f"{tp.STEP}{name}" for name in tp.TRIP_STEPS]
        fault = "the round trip failed: RuntimeError at round_trip.py:1"
        for name, code, printed, why in (
            ("a fault", 1, [*steps, fault], fault),
            (
                "killed",
                137,
                steps[:2],
                f"no message, while {tp.TRIP_STEPS['usage']}; killed (exit 137), as it is when it reaches its "
                f"memory limit, {tp.ONE_OFF_MEMORY} (ONE_OFF_MEMORY in tools/translate_production.py)",
            ),
            ("before it began", 125, [], "no message"),
        ):
            with self.subTest(stopped=name):

                def stops(argv, **kwargs):
                    return subprocess.CompletedProcess(
                        argv, code, "", "".join(f"{line}\n" for line in printed)
                    )

                rc, out, _calls = self.in_process(stops)
                self.assertEqual(
                    (rc, flat(out)),
                    (1, f"ERROR: the one-off container of {self.NAME}'s image: {why}"),
                )

    def test_ctrl_c_removes_the_one_off_container_by_the_id_docker_wrote(self):
        """Ctrl-C while the one-off container runs (`in_process`), of the round trip or of the translation
        (--report-only): docker wrote the container's id to a file of its own in a private temporary folder
        (--cidfile, `made`), and the tool removes the container by that id, never by the name of the container it
        read, with a second Ctrl-C ignored meanwhile. INTERRUPTED, exit 130, says so, and the next line runs the
        command again; the folder is gone after."""
        seen = []
        handler = signal.getsignal(signal.SIGINT)

        def interrupted(argv, **kwargs):
            cidfile = pathlib.Path(argv[argv.index("--cidfile") + 1])
            private = cidfile.parent
            seen.append(
                (cidfile, stat.S_IMODE(private.stat().st_mode), list(private.iterdir()))
            )
            made(argv)
            raise KeyboardInterrupt

        def removed(args):
            seen.append(signal.getsignal(signal.SIGINT))
            return 0, f"{ONE_OFF_ID}\n", ""

        for option in ("--round-trip", "--report-only"):
            with self.subTest(option=option):
                seen.clear()
                rc, out, calls = self.in_process(interrupted, option, removed=removed)
                again = f"python3 tools/translate_production.py --container {self.NAME} {option}"
                self.assertEqual(
                    (rc, flat(out)),
                    (
                        130,
                        f"INTERRUPTED: the one-off container of {self.NAME}'s image is removed, and nothing changed "
                        f"next: {again}",
                    ),
                )
                (cidfile, mode, before), ignored = seen
                self.assertEqual((cidfile.name, mode, before), (tp.CIDFILE, 0o700, []))
                self.assertFalse(cidfile.parent.exists())
                self.assertEqual(
                    (ignored, signal.getsignal(signal.SIGINT)),
                    (signal.SIG_IGN, handler),
                )
                self.assertEqual(
                    [args for args, _timeout in calls[2:]], [["rm", "-f", ONE_OFF_ID]]
                )
                self.assertEqual(
                    [args[0] for args, _timeout in calls if self.NAME in args],
                    ["container"],
                )

    def test_ctrl_c_as_a_terminal_sends_it_removes_the_one_off_container(self):
        """Ctrl-C as a terminal sends it: SIGINT to the tool, run in a process of its own that starts with SIGINT at its
        default, as a terminal starts a command (`as_a_terminal_starts_a_command`), while the one-off container runs,
        of the round trip or of the translation (--report-only). The fake docker sends it (its exit code
        "SIGINT", tests/fakes/fake_docker) once it wrote the container's id to the file --cidfile names
        (FAKE_DOCKER_CID), as docker does. The tool removes the container by that id: INTERRUPTED, exit 130, and the
        next line runs the command again; the private folder of that file, in the tool's temporary folder, is gone
        after.
        """
        self.answer("", run_code="SIGINT")
        env = dict(self.env, FAKE_DOCKER_CID=ONE_OFF_ID, TMPDIR=str(self.d))
        for option in ("--round-trip", "--report-only"):
            with self.subTest(option=option):
                self.log.unlink(missing_ok=True)
                with as_a_terminal_starts_a_command():
                    rc, out, err = self.run_("--container", self.NAME, option, env=env)
                again = f"python3 tools/translate_production.py --container {self.NAME} {option}"
                self.assertEqual(
                    (rc, err, flat(out)),
                    (
                        130,
                        "",
                        f"INTERRUPTED: the one-off container of {self.NAME}'s image is removed, and nothing changed "
                        f"next: {again}",
                    ),
                )
                _inspect, run, *after = self.calls()
                self.assertIn(f" --cidfile {self.d}{os.sep}translate_production-", run)
                self.assertEqual(after, [f"rm -f {ONE_OFF_ID}"])
                self.assertEqual(list(self.d.glob("translate_production-*")), [])

    def test_ctrl_c_before_the_one_off_container_is_made_leaves_nothing_to_undo(self):
        """Ctrl-C while docker inspects the container, or before docker made the one-off container (no id written):
        INTERRUPTED, exit 130, nothing changed, and the next line; nothing is removed.
        """

        def before_made(argv, **kwargs):
            raise KeyboardInterrupt

        rc, out, calls = self.in_process(before_made)
        closing = f"INTERRUPTED: nothing changed next: {self.again}"
        self.assertEqual((rc, flat(out)), (130, closing))
        self.assertEqual([args[0] for args, _timeout in calls], ["container", "run"])
        with (
            mock.patch.object(tp.shutil, "which", return_value="docker"),
            mock.patch.object(
                tp.subprocess, "run", side_effect=KeyboardInterrupt
            ) as run,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            rc = tp.main(["--container", self.NAME, "--round-trip"])
        self.assertEqual((rc, flat(out.getvalue())), (130, closing))
        self.assertEqual(run.call_count, 1)

    def test_a_one_off_container_left_is_named_in_the_fix(self):
        """Ctrl-C, then a docker rm -f that fails: docker ps, filtered by the id, tells whether the container is there.
        One docker's --rm removed first, or one docker is removing, is gone. One still there, or a docker that cannot
        say or gives no answer, is left: the outcome says why, and the fix removes it by its id.
        """
        daemon = "Error response from daemon: "
        busy = f"{daemon}cannot remove container: device or resource busy"
        progress = f"{daemon}removal of container {ONE_OFF_ID} is already in progress"
        down = "Cannot connect to the Docker daemon. Is the docker daemon running?"

        def interrupted(argv, **kwargs):
            made(argv)
            raise KeyboardInterrupt

        def no_answer(args):
            raise subprocess.TimeoutExpired(args, tp.INSPECT_SECONDS)

        for name, answers, left in (
            (
                "there",
                {"rm": (1, "", f"{busy}\n"), "ps": (0, "running\n", "")},
                f"docker rm: {busy}",
            ),
            (
                "cannot say",
                {"rm": (1, "", f"{down}\n"), "ps": (1, "", f"{down}\n")},
                f"docker rm: {down}",
            ),
            (
                "no answer",
                no_answer,
                f"docker gave no answer in {tp.INSPECT_SECONDS} seconds",
            ),
            (
                "removed first",
                {
                    "rm": (1, "", f"{daemon}No such container: {ONE_OFF_ID}\n"),
                    "ps": (0, "", ""),
                },
                None,
            ),
            (
                "being removed",
                {"rm": (1, "", f"{progress}\n"), "ps": (0, "removing\n", "")},
                None,
            ),
        ):
            with self.subTest(container=name):
                removed = (
                    answers
                    if callable(answers)
                    else (lambda args, a=answers: a[args[0]])
                )
                rc, out, calls = self.in_process(interrupted, removed=removed)
                container = f"the one-off container of {self.NAME}'s image"
                said = (
                    f"{container} is removed, and nothing changed"
                    if left is None
                    else f"{container} could not be removed ({left}), and nothing else changed fix: docker rm -f "
                    f"{ONE_OFF_ID}"
                )
                self.assertEqual(
                    (rc, flat(out)), (130, f"INTERRUPTED: {said} next: {self.again}")
                )
                asked = [
                    ["rm", "-f", ONE_OFF_ID],
                    [
                        "ps",
                        "--all",
                        "--no-trunc",
                        "--filter",
                        f"id={ONE_OFF_ID}",
                        "--format",
                        tp.CONTAINER_STATE,
                    ],
                ]
                self.assertEqual(
                    [args for args, _timeout in calls[2:]],
                    asked[: 1 if callable(answers) else 2],
                )

    def test_the_translation_s_time_limit_removes_its_one_off_container_too(self):
        """The translation's one-off container (--report-only) past RUN_SECONDS: the error, that the container is
        removed by its id, and the next line; with no id written, nothing to remove."""
        for name, make in (("made", True), ("not made", False)):
            with self.subTest(container=name):

                def hangs(argv, **kwargs):
                    if make:
                        made(argv)
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

                rc, out, calls = self.in_process(hangs, "--report-only")
                words = (
                    "the container is removed, and nothing changed"
                    if make
                    else "nothing changed"
                )
                self.assertEqual(
                    (rc, flat(out)),
                    (
                        1,
                        f"ERROR: the one-off container of {self.NAME}'s image: no answer in {tp.RUN_SECONDS} seconds; "
                        f"{words} next: python3 tools/translate_production.py --container {self.NAME} --report-only",
                    ),
                )
                self.assertEqual(
                    [(args[0], timeout) for args, timeout in calls],
                    [
                        ("container", tp.INSPECT_SECONDS),
                        ("run", tp.RUN_SECONDS),
                        *((("rm", tp.INSPECT_SECONDS),) if make else ()),
                    ],
                )

    def test_usage(self):
        for argv in (
            ("--round-trip",),
            ("--container", self.NAME, "--round-trip", "--report-only"),
            ("--container", self.NAME, "--report-only", "--agreed", "agreed.json"),
            (
                "--container",
                self.NAME,
                "--out",
                "deploy.yaml",
                "--agreed",
                "agreed.json",
            ),
        ):
            with self.subTest(argv=argv):
                rc, out, err = self.run_(*argv)
                self.assertEqual((rc, out), (2, ""))
                self.assertIn("usage:", err)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
