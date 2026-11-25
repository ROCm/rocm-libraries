# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The work volume's one-time move (rkmon_lib/one_time_move.py): the start that
finds your settings delivered beside the config.json an older package left splits it once by owner: each state key
carried into the internal state where the schema takes it, every other key named in the report with what runs in its
place or why it is dropped, the time before which the older monitor's unnamed reviews are this deployment's own set to
the move's, and the monitor in standby. The report shows no private key's value; the record keeps each. rk start makes
the move after its load; rk provision, which each start runs before it, records no clone start date of its own while the
move is due, so the move carries config.json's. Whether the work volume is an older install's is decided once, at the
first read of this package's code on it, and kept in the internal state: a config.json there then moves; one written
later on a work volume this package set up, or after the end, never does, and neither a second start nor a config.json
that changes while the move is open moves anything. Only rkmon adopt's call makes the move due again: the start after
adopt splits the config.json it copied, standby keeping adopt's reason, and rkmon standby off ends it; the mark of the
adoption keeps the monitor in standby on every road until then, and standby off refuses while the mark says that the
adopt stopped part-way. A load, `rk applied --load`, a home with no config.json and a volume with no layer never move. A
pending move is standby whatever the internal state says. Until the end the move only adds; the end, once, renames
config.json read-only, removes each older copy of a secret its rule lets go, keeping, with the fix, each one whose entry
is not set in its own place, which a later start removes once it is, and sets that time again, to its own; while the
move is due and not made, it refuses, with its fix. Once the move has ended, an older copy no longer counts as its entry
set.
"""

import contextlib
import errno
import hashlib
import io
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    adopt,
    container_layout,
    control,
    deploy,
    features,
    file_types,
    image,
    logins,
    one_time_move,
    paths,
    provision,
    registry,
    removed_keys,
    renames,
    runtime_config,
    standby,
    state_store,
    steps,
    store,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
from rkmon_lib.checks import FIX_LABEL  # noqa: E402

# A layer's settings whose configuration runs, and the keys a layer lists as set in your file.
from test_runtime_config import SETTINGS, set_in  # noqa: E402

RC, MOVE = runtime_config, one_time_move
VERSION = image.package_version(paths.PACKAGE)
FINGERPRINT = "sha256:" + "1" * 64
MOVED_AT, LATER = "2026-01-02T03:00:00Z", "2026-01-03T03:00:00Z"
AGAIN = "2026-01-04T03:00:00Z"
# config.json as an older setup wrote it, with values of an older install's own over it.
OLDER_SETUP = ROOT / "tests" / "data" / "older_setup_config.json"
OLDER = {
    "repo": "acme/gadgets",
    "clone_since": "2026-01-05",
    "drift_batch": 97,
    "no_such_key": 1,
}
# The clone's start date rk provision takes when the internal state holds none (deploy.default_since).
SINCE = "2026-03-04"
STANDBY_OFF = steps.command_line("standby off")
# The internal state's time before which the older monitor's reviews whose line 1 names no automated reviewer are this
# deployment's own.
UNTIL = "reviews.unnamed_are_ours_until"


def until_line(now: str, then: str = "") -> str:
    """The line that says UNTIL is NOW."""
    return (
        f"{UNTIL} is {now}: the reviews the older monitor posted before then, whose line 1 names no automated "
        f"reviewer, are this deployment's own{then}"
    )


# The move's last line.
UNNAMED = until_line(MOVED_AT, f"; {STANDBY_OFF} sets it again, to its own time")


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = self.dir / "work" / "monitor"
        self.home.mkdir(parents=True)
        self.settings_dir = self.dir / "settings"
        self.secrets = self.dir / "secrets"
        self.write_secrets = self.dir / "secrets-write"
        for d in (self.settings_dir, self.secrets, self.write_secrets):
            d.mkdir()
        self.config_file = self.home / RC.CONFIG_JSON
        self.clock = MOVED_AT
        for patch in (
            # The container's layout, its folders the test's: where the layer is delivered, the work volume, with the
            # monitor's home, and the two secrets volumes.
            mock.patch.object(RC, "SETTINGS_DIR", str(self.settings_dir)),
            mock.patch.object(container_layout, "WORK_DIR", str(self.dir / "work")),
            mock.patch.object(container_layout, "MONITOR_HOME", str(self.home)),
            mock.patch.object(container_layout, "SECRETS_DIR", str(self.secrets)),
            mock.patch.object(
                container_layout, "WRITE_SECRETS_DIR", str(self.write_secrets)
            ),
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.object(store, "now", side_effect=lambda: self.clock),
            mock.patch.dict(
                os.environ,
                {
                    "RKMON_CONTAINER": "1",
                    "RKMON_HOME": str(self.home),
                    "RKMON_WRAPPER": "./rkmon",
                    "XDG_CONFIG_HOME": str(self.dir / "xdg"),
                },
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def layer(self, settings: dict | None = None) -> None:
        """Your confirmed settings, delivered: SETTINGS, or the test's own."""
        settings = SETTINGS if settings is None else settings
        layer = {
            "kind": RC.LAYER_KIND,
            "schema_version": RC.LAYER_SCHEMA_VERSION,
            "deploy_schema_version": file_types.DEPLOY.version,
            "package_version": VERSION,
            "confirmed_at": "2026-01-02T01:00:00Z",
            "fingerprint": FINGERPRINT,
            "settings": settings,
            "set_in_file": set_in(settings),
            "container": {},
        }
        (self.settings_dir / RC.LAYER_FILE).write_text(
            json.dumps(layer), encoding="utf-8"
        )

    def older_config(self, **over) -> bytes:
        """config.json as an older package left it in the monitor's home: its content."""
        raw = json.loads(OLDER_SETUP.read_text(encoding="utf-8"))
        raw.update(OLDER, **over)
        data = json.dumps(raw, indent=1).encode()
        self.config_file.write_bytes(data)
        return data

    def start(self) -> list:
        """What rk start does of the move, with the configuration it loads (one_time_move.at_start)."""
        return MOVE.at_start(self.home, RC.load(self.home))

    def file(self, path: pathlib.Path, text: str = "a value\n") -> pathlib.Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def state_file(self) -> pathlib.Path:
        return state_store.path(self.home)

    def nothing_moved(self) -> None:
        self.assertFalse(self.state_file().exists())
        self.assertIsNone(MOVE.record(self.home))
        self.assertFalse((self.home / MOVE.MIGRATED).exists())

    def until(self) -> str:
        """The time before which the older monitor's unnamed reviews are this deployment's own, as the configuration
        the monitor runs reads it."""
        return registry.get(RC.load(self.home).values, UNTIL)

    def standby_off(self) -> str:
        """rkmon standby off (rk standby-off, rkmon_lib/standby.py): the end of the move, then standby's own switch
        off; its output."""
        out = io.StringIO()
        rc = standby.off(RC.load(self.home).values, self.home, out)
        self.assertEqual(rc, 0, out.getvalue())
        return out.getvalue()


class SplitTest(Base):
    def test_a_start_splits_config_json_by_owner_and_says_so(self):
        self.layer()
        data = self.older_config()
        lines = self.start()
        self.assertEqual(
            lines[0],
            f"moved: {self.config_file}, an older install's configuration, split by owner, once; the monitor is in "
            f"standby: it sends nothing until you run {STANDBY_OFF}",
        )
        notes = renames.migrate(json.loads(data), RC.CONFIG_JSON)[1]
        self.assertTrue(notes)
        self.assertEqual(lines[1 : 1 + len(notes)], notes)
        self.assertEqual(
            lines[1 + len(notes)],
            "carried into the internal state: clone_since, codeowner_teams",
        )
        # A deploy key: what your deploy settings give runs. The line of a private key (registry.Key.klass), as repo
        # is, shows neither value; the record keeps both, for the terminal.
        self.assertEqual(registry.key("repo").klass, "private")
        shown = "repo: config.json's value is not carried; your deploy settings' repository.name runs"
        full = (
            "repo: config.json's \"acme/gadgets\" is not carried; your deploy settings' repository.name "
            '"acme/widgets" runs'
        )
        self.assertIn(shown, lines)
        for value in ("acme/gadgets", "acme/widgets"):
            self.assertEqual([x for x in lines if value in x], [], value)
        # A package key off its default: the package's runs.
        package = registry.get(registry.package_defaults(), "drift_batch")
        self.assertIn(
            f"drift_batch: config.json's 97 is not carried; the package's {package} runs",
            lines,
        )
        # Dropped, each named: a removed key with why, a derived key, a deploy key your settings set none of, and a
        # name no key has. The keys of the services of an install without a container go in one line.
        self.assertIn(
            "dropped (an install without a container runs no services): services, wsl_distro",
            lines,
        )
        self.assertIn("clone_url", self.dropped(lines, "each load computes them"))
        self.assertIn(
            "worker.backend",
            self.dropped(lines, "your deploy settings set none of them"),
        )
        self.assertEqual(
            self.dropped(lines, "no key of this package has the name"), ["no_such_key"]
        )
        self.assertEqual(lines[-1], UNNAMED)
        # A key config.json sets to nothing is never said.
        for line in lines:
            self.assertNotRegex(line, r"config\.json's (\[\]|\{\}|\"\"|null) ")
        state = state_store.read(self.home)
        self.assertEqual(state["clone_since"], "2026-01-05")
        self.assertEqual(state["reviews"], {"unnamed_are_ours_until": MOVED_AT})
        self.assertEqual(
            state["standby"], {"on": True, "reason": MOVE.REASON, "since": MOVED_AT}
        )
        self.assertNotIn("repo", state)
        rec = MOVE.record(self.home)
        self.assertEqual(
            {
                k: rec[k]
                for k in (
                    "moved_at",
                    "fingerprint",
                    "package_version",
                    "config_json_sha256",
                    "ended_at",
                )
            },
            {
                "moved_at": MOVED_AT,
                "fingerprint": FINGERPRINT,
                "package_version": VERSION,
                "config_json_sha256": hashlib.sha256(data).hexdigest(),
                "ended_at": "",
            },
        )
        # The record's report is the report with each value shown: only a private key's line differs.
        self.assertEqual(len(rec["report"]), len(lines))
        differ = [(a, b) for a, b in zip(lines, rec["report"]) if a != b]
        self.assertIn((shown, full), differ)
        self.assertEqual(
            [a for a, _ in differ if registry.key(a.split(":")[0]).klass != "private"],
            [],
        )
        self.assertEqual(
            MOVE.record_file(self.home), self.home / "state" / MOVE.RECORD_FILE
        )

    @staticmethod
    def dropped(lines: list, why: str) -> list:
        lead = f"dropped ({why}): "
        found = [
            line[len(lead) :].split(", ") for line in lines if line.startswith(lead)
        ]
        return found[0] if len(found) == 1 else found

    def test_a_state_key_the_internal_state_holds_is_kept(self):
        self.layer()
        self.older_config()
        state_store.set(self.home, "clone_since", "2026-02-01")
        lines = self.start()
        self.assertIn("carried into the internal state: codeowner_teams", lines)
        self.assertIn(
            'clone_since: config.json\'s "2026-01-05" is not carried; the internal state\'s "2026-02-01" is kept',
            lines,
        )
        self.assertEqual(state_store.read(self.home)["clone_since"], "2026-02-01")

    def test_a_key_config_json_sets_to_nothing_is_never_said_or_carried(self):
        self.layer()
        state_store.set(self.home, "base_branch", "develop")
        self.older_config(repo="", base_branch="", codeowner_teams={})
        lines = self.start()
        self.assertIn("carried into the internal state: clone_since", lines)
        for name in ("repo", "base_branch", "codeowner_teams"):
            self.assertEqual([x for x in lines if x.startswith(f"{name}: ")], [])
        state = state_store.read(self.home)
        self.assertEqual(state["base_branch"], "develop")
        self.assertNotIn("codeowner_teams", state)

    def test_a_value_the_internal_states_schema_refuses_is_said_and_not_carried(self):
        self.layer()
        self.older_config(clone_since=12)
        lines = self.start()
        self.assertIn("carried into the internal state: codeowner_teams", lines)
        refused = [line for line in lines if line.startswith("clone_since: ")]
        self.assertEqual(len(refused), 1, lines)
        self.assertTrue(
            refused[0].startswith(
                "clone_since: config.json's 12 is not carried; the internal state's schema refuses it ("
            ),
            refused,
        )
        state = state_store.read(self.home)
        self.assertNotIn("clone_since", state)
        self.assertIn("codeowner_teams", state)
        self.assertTrue(state["standby"]["on"])

    def test_a_private_state_keys_lines_show_no_value_and_the_record_keeps_each(self):
        """codeowner_teams is a private state key (registry.Key.klass): the line that says the internal state's value
        is kept, and the one that says the schema refuses config.json's, whose why quotes it, show no value.
        """
        self.assertEqual(registry.key("codeowner_teams").klass, "private")
        self.layer()
        held = {"widgets": ["team-x"]}
        for over, state, shown, full in (
            (
                {},
                held,
                "codeowner_teams: config.json's value is not carried; the internal state's value is kept",
                'codeowner_teams: config.json\'s {"library": [], "platform": [], "root": [], "style": []} is not '
                'carried; the internal state\'s {"widgets": ["team-x"]} is kept',
            ),
            (
                {"codeowner_teams": 12},
                None,
                "codeowner_teams: config.json's value is not carried; the internal state's schema refuses it",
                "codeowner_teams: config.json's 12 is not carried; the internal state's schema refuses it (",
            ),
        ):
            with self.subTest(shown=shown):
                state_store.path(self.home).unlink(missing_ok=True)
                MOVE.record_file(self.home).unlink(missing_ok=True)
                if state is not None:
                    state_store.set(self.home, "codeowner_teams", state)
                self.older_config(**over)
                lines = self.start()
                said = [x for x in lines if x.startswith("codeowner_teams: ")]
                self.assertEqual(said, [shown])
                report = MOVE.record(self.home)["report"]
                kept = [x for x in report if x.startswith("codeowner_teams: ")]
                self.assertEqual(len(kept), 1, kept)
                self.assertTrue(kept[0].startswith(full), kept)
                got = state_store.read(self.home).get("codeowner_teams")
                self.assertEqual(got, state)

    def test_the_keys_whose_value_the_package_sets_are_dropped_with_why(self):
        """No settings file holds them (removed_keys.PACKAGE_SET): the admin dashboard's address and the users' ids."""
        self.layer()
        raw = json.loads(OLDER_SETUP.read_text(encoding="utf-8"))
        self.older_config(
            admin_dashboard=dict(raw["admin_dashboard"], publish_host="0.0.0.0"),
            container={"write_uid": 1999, "pr_uid": 1998},
        )
        lines = self.start()
        for names in (
            ["admin_dashboard.publish_host"],
            ["container.pr_uid", "container.write_uid"],
        ):
            why = removed_keys.why(names[0])
            self.assertEqual(why, removed_keys.PACKAGE_SET[names[-1]])
            self.assertEqual(sorted(self.dropped(lines, why)), names)
        self.assertEqual(
            self.dropped(lines, "no key of this package has the name"), ["no_such_key"]
        )

    def test_a_config_json_that_cannot_be_read_is_moved_with_nothing_carried(self):
        self.layer()
        self.config_file.write_text("{", encoding="utf-8")
        lines = self.start()
        self.assertTrue(
            lines[0].startswith("moved: nothing of config.json is carried, as "), lines
        )
        self.assertTrue(
            lines[0].endswith(
                f"; the monitor is in standby: it sends nothing until you run {STANDBY_OFF}"
            )
        )
        self.assertEqual(lines[1:], [UNNAMED])
        state = state_store.read(self.home)
        self.assertEqual(state["standby"]["on"], True)
        self.assertEqual(state["reviews"], {"unnamed_are_ours_until": MOVED_AT})

    def test_standby_on_already_keeps_its_reason_and_time(self):
        self.layer()
        self.older_config()
        for name, value in (
            ("on", True),
            ("reason", "adopted"),
            ("since", "2026-01-01T00:00:00Z"),
        ):
            state_store.set(self.home, f"standby.{name}", value)
        self.start()
        state = state_store.read(self.home)
        self.assertEqual(
            state["standby"],
            {"on": True, "reason": "adopted", "since": "2026-01-01T00:00:00Z"},
        )
        self.assertEqual(state["reviews"], {"unnamed_are_ours_until": MOVED_AT})


class OnceTest(Base):
    def test_a_second_start_moves_nothing(self):
        self.layer()
        self.older_config()
        self.start()
        state, rec = (
            self.state_file().read_bytes(),
            MOVE.record_file(self.home).read_bytes(),
        )
        self.clock = LATER
        self.assertEqual(self.start(), [])
        self.assertEqual(self.state_file().read_bytes(), state)
        self.assertEqual(MOVE.record_file(self.home).read_bytes(), rec)

    def test_migrate_moves_only_when_a_move_is_due_under_the_lock(self):
        """Two starts at once: the second finds, under the lock, that the first has moved config.json."""
        self.layer()
        self.older_config()
        running = RC.load(self.home)
        self.assertTrue(state_store.migrate(self.home, running))
        rec = MOVE.record_file(self.home).read_bytes()
        self.clock = LATER
        self.assertEqual(state_store.migrate(self.home, running), [])
        self.assertEqual(MOVE.record_file(self.home).read_bytes(), rec)

    def test_a_config_json_that_changes_while_the_move_is_open_is_not_moved_again(
        self,
    ):
        """Only rkmon adopt makes the move due again (DueAgainTest)."""
        self.layer()
        data = self.older_config()
        self.start()
        state, rec = (
            self.state_file().read_bytes(),
            MOVE.record_file(self.home).read_bytes(),
        )
        self.older_config(drift_batch=98)
        self.clock = LATER
        self.assertTrue(MOVE.pending(self.home))
        self.assertFalse(MOVE.due(self.home))
        self.assertEqual(self.start(), [])
        self.assertEqual(self.state_file().read_bytes(), state)
        self.assertEqual(MOVE.record_file(self.home).read_bytes(), rec)
        self.assertEqual(
            MOVE.record(self.home)["config_json_sha256"],
            hashlib.sha256(data).hexdigest(),
        )

    def test_until_the_end_the_move_only_adds(self):
        self.layer()
        data = self.older_config()
        older = [
            self.file(self.secrets / "teams_webhook", "https://example.com/hook\n"),
            self.file(self.secrets / "admin_token"),
            self.file(self.secrets / "atlassian" / "credentials"),
            self.file(self.secrets / "worker_headers"),
            self.file(self.secrets / "jira_mcp_headers"),
        ]
        self.start()
        self.clock = LATER
        self.start()
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertFalse((self.home / MOVE.MIGRATED).exists())
        for p in older:
            self.assertTrue(p.is_file(), p)

    def test_a_move_whose_record_cannot_be_written_is_said_and_made_at_the_next_start(
        self,
    ):
        self.layer()
        self.older_config()
        real = state_store.write_file

        def write_file(file, data):
            if pathlib.Path(file) == MOVE.record_file(self.home):
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            lines = self.start()
        folder = MOVE.record_file(self.home).parent
        self.assertEqual(
            lines,
            [
                f"not moved: the record of the one-time move cannot be written in {folder}: Permission denied; the "
                "monitor is in standby until the move is made, which the next start tries again",
                f"{FIX_LABEL}make {folder} writable for the monitor, then restart the monitor: "
                f"{steps.command_line('stop')}, then {steps.command_line('start')}",
            ],
        )
        self.assertIsNone(MOVE.record(self.home))
        self.assertTrue(MOVE.due(self.home))
        self.assertTrue(features.standby(self.home))
        self.clock = LATER
        lines = self.start()
        self.assertTrue(lines[0].startswith(f"moved: {self.config_file}, "), lines)
        # what the first try carried is in the internal state already
        self.assertNotIn(
            "carried into the internal state: clone_since, codeowner_teams", lines
        )
        self.assertEqual(state_store.read(self.home)["clone_since"], "2026-01-05")
        self.assertEqual(MOVE.record(self.home)["moved_at"], LATER)


class UntilTest(Base):
    """The time before which the older monitor's unnamed reviews are this deployment's own, as the configuration the
    monitor runs reads it."""

    def test_blank_before_the_move_the_moves_time_after_it_then_the_ends(self):
        self.layer()
        self.older_config()
        self.assertEqual(self.until(), "")
        self.start()
        self.assertEqual(self.until(), MOVED_AT)
        # a second start of the same config.json leaves it alone
        self.clock = LATER
        self.assertEqual(self.start(), [])
        self.assertEqual(self.until(), MOVED_AT)
        lines = MOVE.end(self.home, RC.load(self.home).values)
        self.assertEqual(self.until(), LATER)
        self.assertEqual(lines[-1], until_line(LATER))
        self.assertEqual(MOVE.record(self.home)["ended_at"], LATER)


class NeverTest(Base):
    def test_a_new_deployment_never_moves(self):
        self.layer()
        self.assertFalse(MOVE.pending(self.home))
        self.assertEqual(self.start(), [])
        self.nothing_moved()
        self.assertFalse(features.standby(self.home))
        self.assertEqual(self.until(), "")
        # nor is there a move to end
        self.assertEqual(MOVE.end(self.home, RC.load(self.home).values), [])
        self.nothing_moved()

    def test_a_volume_with_no_layer_never_moves(self):
        data = self.older_config()
        running = RC.load(self.home)
        self.assertIsNone(running.layer)
        self.assertFalse(MOVE.pending(self.home))
        self.assertEqual(MOVE.at_start(self.home, running), [])
        self.nothing_moved()
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertFalse(features.standby(self.home))

    def test_a_load_never_moves(self):
        self.layer()
        data = self.older_config()
        RC.load(self.home)
        RC.folders(self.home)
        self.nothing_moved()
        self.assertEqual(self.config_file.read_bytes(), data)

    def test_a_home_other_than_the_work_volumes_never_moves_or_is_in_standby(self):
        """A test's own home, with a layer delivered beside a config.json, as the configuration's tests set it up."""
        self.layer()
        data = self.older_config()
        elsewhere = self.dir / "work" / "elsewhere"
        with mock.patch.object(container_layout, "MONITOR_HOME", str(elsewhere)):
            self.assertFalse(MOVE.pending(self.home))
            self.assertFalse(features.standby(self.home))
            self.assertEqual(self.start(), [])
        self.nothing_moved()
        self.assertEqual(self.config_file.read_bytes(), data)

    def test_outside_the_container_nothing_moves(self):
        self.layer()
        self.older_config()
        with mock.patch.dict(os.environ, {"RKMON_CONTAINER": ""}):
            self.assertFalse(MOVE.pending(self.home))
            self.assertEqual(
                MOVE.at_start(self.home, RC.load(self.home, self.settings_dir)), []
            )
        self.nothing_moved()

    def test_rk_start_moves_after_the_load_and_rk_applied_load_never_moves(self):
        self.layer()
        self.older_config()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rkmon.main(["applied", "--load"]), 0)
        self.nothing_moved()
        out = []
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        control.start({}, store.Store(), run=run, out=out.append)
        loaded = [
            i
            for i, line in enumerate(out)
            if line.startswith("loaded: the settings you confirmed at ")
        ]
        self.assertEqual(len(loaded), 1, out)
        self.assertTrue(
            out[loaded[0] + 1].startswith(f"moved: {self.config_file}, "), out
        )
        self.assertIn(UNNAMED, out)
        self.assertTrue(state_store.read(self.home)["standby"]["on"])


class OlderInstallTest(Base):
    """Whether the work volume is an older install's: decided once, at the first read of this package's code on it,
    which is rk provision's at the first start (provision.older_install), and kept in the internal state.
    """

    def first_start(self) -> list:
        """This package's first start on the work volume: rk provision (no steps here), then rk start's move."""
        with mock.patch.object(provision, "STEPS", ()):
            self.assertEqual(provision.main(out=lambda line: None), 0)
        return self.start()

    def clone_step(self) -> None:
        """rk provision with its clone step alone, on the older install's clone in place (no git command runs), SINCE
        the start date it takes when the internal state holds none."""
        with contextlib.ExitStack() as stack:
            for patch in (
                mock.patch.object(provision, "STEPS", (provision._clone,)),
                mock.patch.object(deploy, "default_since", return_value=SINCE),
                mock.patch.object(deploy, "clone_check", return_value=None),
                mock.patch.object(provision, "_checked_out", return_value=True),
                mock.patch.object(deploy, "run"),
            ):
                stack.enter_context(patch)
            self.assertEqual(provision.main(out=lambda line: None), 0)

    def test_the_clone_step_before_the_move_leaves_the_clone_s_start_date_to_it(
        self,
    ):
        """Each start runs rk provision before rk start: while the move is due, its clone step records no start date
        of its own, so the move carries config.json's, the date the older install's clone goes back to.
        """
        self.layer()
        self.older_config()
        self.clone_step()
        self.assertNotIn("clone_since", state_store.read(self.home))
        lines = self.start()
        self.assertIn(
            "carried into the internal state: clone_since, codeowner_teams", lines
        )
        self.assertEqual(
            state_store.read(self.home)["clone_since"], OLDER["clone_since"]
        )
        self.clone_step()
        self.assertEqual(
            state_store.read(self.home)["clone_since"], OLDER["clone_since"]
        )

    def test_once_a_move_carried_no_start_date_the_clone_step_records_its_own(self):
        self.layer()
        self.older_config(clone_since=None)
        self.clone_step()
        self.assertNotIn("clone_since", state_store.read(self.home))
        self.assertTrue(self.start()[0].startswith(f"moved: {self.config_file}, "))
        self.assertNotIn("clone_since", state_store.read(self.home))
        self.clone_step()
        self.assertEqual(state_store.read(self.home)["clone_since"], SINCE)

    def test_the_first_read_of_a_volume_with_a_config_json_moves_it(self):
        self.layer()
        self.older_config()
        lines = self.first_start()
        self.assertIs(state_store.read(self.home)["provisioned"]["older_install"], True)
        self.assertTrue(provision.older_install(self.home))
        self.assertTrue(lines[0].startswith(f"moved: {self.config_file}, "), lines)
        self.assertTrue(features.standby(self.home))
        self.assertEqual(self.until(), MOVED_AT)

    def test_a_config_json_written_later_on_a_volume_this_package_set_up_never_moves(
        self,
    ):
        self.layer()
        self.assertEqual(self.first_start(), [])
        self.assertNotIn("older_install", state_store.read(self.home)["provisioned"])
        state = self.state_file().read_bytes()
        data = self.older_config()
        self.clock = LATER
        self.assertFalse(provision.older_install(self.home))
        self.assertFalse(MOVE.pending(self.home))
        self.assertFalse(MOVE.due(self.home))
        self.assertFalse(features.standby(self.home))
        self.assertEqual(self.start(), [])
        self.assertEqual(MOVE.end(self.home, RC.load(self.home).values), [])
        self.assertIsNone(MOVE.record(self.home))
        self.assertEqual(self.state_file().read_bytes(), state)
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertEqual(self.until(), "")

    def test_a_config_json_written_after_the_end_never_moves(self):
        self.layer()
        self.older_config()
        self.first_start()
        self.clock = LATER
        self.standby_off()
        self.assertFalse(features.standby(self.home))
        state, rec = (
            self.state_file().read_bytes(),
            MOVE.record_file(self.home).read_bytes(),
        )
        data = self.older_config(drift_batch=98)
        self.clock = AGAIN
        self.assertFalse(MOVE.pending(self.home))
        self.assertFalse(MOVE.due(self.home))
        self.assertFalse(features.standby(self.home))
        self.assertEqual(self.start(), [])
        self.assertEqual(MOVE.end(self.home, RC.load(self.home).values), [])
        self.assertEqual(self.state_file().read_bytes(), state)
        self.assertEqual(MOVE.record_file(self.home).read_bytes(), rec)
        self.assertEqual(self.config_file.read_bytes(), data)
        self.assertEqual(self.until(), LATER)


class DueAgainTest(Base):
    """rkmon adopt, once it has copied an earlier monitor's config.json into the monitor's home, makes the move due
    again (one_time_move.due_again, with the steps that run adopt again after a fix): the next start splits that
    config.json, once."""

    def again(self) -> tuple:
        """The words and the command rkmon adopt gives due_again, which run it again after a fix."""
        return ("run", steps.command_line("adopt --from DIR"), "again")

    def due_again(self) -> None:
        self.assertEqual(
            MOVE.due_again(self.home, self.again()),
            [
                f"the one-time move is due again: the next start splits {self.config_file} by owner, once, and the "
                f"monitor stands by until you run {STANDBY_OFF}"
            ],
        )
        self.assertIsNone(MOVE.record(self.home))
        self.assertTrue(provision.older_install(self.home))
        self.assertTrue(MOVE.due(self.home))
        self.assertTrue(MOVE.pending(self.home))
        self.assertTrue(features.standby(self.home))

    def moves_once(self, data: bytes) -> None:
        """The next start moves DATA, config.json's content, and the one after moves nothing."""
        self.clock = AGAIN
        lines = self.start()
        self.assertTrue(lines[0].startswith(f"moved: {self.config_file}, "), lines)
        rec = MOVE.record(self.home)
        self.assertEqual(rec["moved_at"], AGAIN)
        self.assertEqual(rec["config_json_sha256"], hashlib.sha256(data).hexdigest())
        self.assertTrue(state_store.read(self.home)["standby"]["on"])
        self.assertEqual(self.until(), AGAIN)
        self.assertEqual(self.start(), [])

    def test_after_the_end(self):
        self.layer()
        self.older_config()
        self.start()
        self.clock = LATER
        self.standby_off()
        data = self.older_config(drift_batch=98)
        self.assertFalse(MOVE.due(self.home))
        self.due_again()
        self.moves_once(data)

    def test_while_the_move_is_open(self):
        """Adopt run a second time: the next start splits the config.json it copied."""
        self.layer()
        self.older_config()
        self.start()
        data = self.older_config(drift_batch=98)
        self.due_again()
        self.moves_once(data)

    def test_on_a_volume_this_package_set_up(self):
        self.layer()
        with mock.patch.object(provision, "STEPS", ()):
            self.assertEqual(provision.main(out=lambda line: None), 0)
        data = self.older_config()
        self.assertFalse(MOVE.due(self.home))
        self.due_again()
        self.moves_once(data)

    def test_with_no_config_json_it_does_nothing(self):
        self.layer()
        self.assertEqual(MOVE.due_again(self.home, self.again()), [])
        self.nothing_moved()

    def test_a_state_folder_it_cannot_write_is_said_with_its_fix(self):
        self.layer()
        self.older_config()
        real = state_store.write_file

        def write_file(file, data):
            if pathlib.Path(file) == self.state_file():
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            with self.assertRaises(MOVE.MoveError) as caught:
                MOVE.due_again(self.home, self.again())
        folder = self.state_file().parent
        self.assertEqual(
            str(caught.exception),
            f"the one-time move cannot be made due again in {folder}: Permission denied",
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {folder} writable for the monitor, then run {again} again",
        )


class AdoptTest(Base):
    """rkmon adopt's state, then the start: the earlier monitor's config.json comes with its
    state, in standby, and the start after it splits that config.json, once, standby keeping adopt's reason and time;
    rkmon standby off then ends the move, and standby. The mark of the adoption (adopt.marked) keeps the monitor in
    standby on every road until standby off, which refuses while the mark says that the adopt stopped part-way.
    """

    def earlier(self, config_json: bool = True, prs=("101",)) -> pathlib.Path:
        """A copy of an earlier monitor's /work with PRS in flight, staged as rkmon adopt stages it: the stage. With
        CONFIG_JSON, its config.json, as an older setup wrote it, comes too."""
        earlier = self.dir / "earlier"
        state = {"prs": {n: {"status": "open"} for n in prs}}
        self.file(earlier / "monitor" / "state" / "state.json", json.dumps(state))
        if config_json:
            raw = json.loads(OLDER_SETUP.read_text(encoding="utf-8"))
            raw.update(OLDER)
            self.file(earlier / "monitor" / RC.CONFIG_JSON, json.dumps(raw, indent=1))
        stage = self.dir / "stage"
        adopt.stage(adopt.old_folders(earlier, {}), stage)
        return stage

    def adopt(self, stage: pathlib.Path, at: str) -> list:
        """rk adopt apply from STAGE, its backup stamped AT: the lines it returns."""
        targets = {"state": str(self.state_file().parent), "home": str(self.home)}
        return adopt.apply(stage, targets, self.dir / "work", now=at)

    def start_after_adopt(self) -> list:
        """The start after adopt: the work volume set up around the adopted state (rk provision), then the move."""
        with mock.patch.object(provision, "STEPS", ()):
            self.assertEqual(provision.main(out=lambda line: None), 0)
        return self.start()

    def refused_off(self) -> str:
        """rkmon standby off that refuses, nothing changed (exit 4): its output, its words one space apart."""
        out = io.StringIO()
        rc = standby.off(RC.load(self.home).values, self.home, out)
        self.assertEqual(rc, 4, out.getvalue())
        return " ".join(out.getvalue().split())

    def unfinished(self) -> str:
        """What standby off says while the mark of the adoption says that the adopt stopped part-way."""
        adopt_cmd, again = steps.command_line("adopt"), steps.command_line(
            "adopt --from DIR"
        )
        return (
            f"REFUSED: {adopt_cmd} stopped part-way; standby stays on; nothing changed {FIX_LABEL}run {again} "
            "again"
        )

    def test_two_adopts_each_split_by_the_start_after_it_then_standby_off(self):
        """Adopt, then the start, which splits the config.json adopt copied, once (a second start moves
        nothing), standby keeping adopt's reason and time; with the container stopped, adopt again, from a later copy,
        which makes the move due again, and the start after it splits it again; then rkmon standby off ends the move
        and standby, and removes the mark of the adoption. Each apply finishes (adopt.unfinished).
        """
        lines = self.adopt(self.earlier(), "20260101T000000Z")
        self.assertTrue(lines[-2].startswith("the one-time move is due again: "), lines)
        self.assertTrue(lines[-1].startswith("APPLIED: "), lines)
        self.assertIsNone(adopt.unfinished(self.home))
        standby = {"on": True, "reason": adopt.REASON, "since": MOVED_AT}
        self.assertEqual(state_store.read(self.home)["standby"], standby)
        data = self.config_file.read_bytes()
        self.layer()
        self.clock = LATER
        moved = self.start_after_adopt()
        self.assertTrue(moved[0].startswith(f"moved: {self.config_file}, "), moved)
        rec = MOVE.record(self.home)
        self.assertEqual(rec["config_json_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(state_store.read(self.home)["standby"], standby)
        self.assertEqual(self.start(), [])
        # the container stopped, the earlier monitor ran on meanwhile: adopt again, from a later copy
        self.clock = AGAIN
        lines = self.adopt(self.earlier(prs=("101", "202")), "20260102T000000Z")
        self.assertTrue(lines[-2].startswith("the one-time move is due again: "), lines)
        self.assertIsNone(adopt.unfinished(self.home))
        self.assertTrue(MOVE.due(self.home))
        again = {"on": True, "reason": adopt.REASON, "since": AGAIN}
        self.assertEqual(state_store.read(self.home)["standby"], again)
        prs = json.loads((self.state_file().parent / "state.json").read_text())["prs"]
        self.assertEqual(sorted(prs), ["101", "202"])
        self.clock = later = "2026-01-05T03:00:00Z"
        moved = self.start_after_adopt()
        self.assertTrue(moved[0].startswith(f"moved: {self.config_file}, "), moved)
        self.assertEqual(MOVE.record(self.home)["moved_at"], later)
        self.assertEqual(state_store.read(self.home)["standby"], again)
        self.clock = "2026-01-06T03:00:00Z"
        self.standby_off()
        self.assertFalse(adopt.marked(self.home))
        self.assertFalse(features.standby(self.home))
        self.assertTrue((self.home / MOVE.MIGRATED).exists())
        self.assertEqual(self.until(), self.clock)

    def test_an_adopt_stopped_part_way_holds_standby_until_adopt_again_and_standby_off(
        self,
    ):
        """An apply that stops once this deployment's own state went aside, here a full volume as its state.json goes,
        after its internal state, which said that the monitor sends: no internal state, and the mark of the adoption
        keeps the monitor in standby; standby off refuses, nothing changed, its fix adopt again. adopt again, from the
        same copy, finishes the adoption, the start makes the move, and standby off ends standby and removes the mark.
        """
        folder = self.state_file().parent
        state_store.set(
            self.home, "standby.on", False
        )  # this deployment's own: it sends
        self.file(folder / "state.json", json.dumps({"prs": {}}))
        self.assertFalse(features.standby(self.home))
        stage = self.earlier()
        move = shutil.move

        def full(src, dst, *args, **kwargs):
            # this deployment's own entries go aside in their names' order: internal_state.json before state.json
            if pathlib.Path(src) == folder / "state.json":
                no_space = errno.ENOSPC
                raise OSError(no_space, os.strerror(no_space), str(src), None, str(dst))
            return move(src, dst, *args, **kwargs)

        with mock.patch.object(shutil, "move", side_effect=full):
            with self.assertRaises(adopt.Stopped) as caught:
                self.adopt(stage, "20260101T000000Z")
        self.assertEqual(caught.exception.where, adopt.PART_WAY)
        self.assertFalse(self.state_file().exists())
        self.assertTrue(features.standby(self.home))
        self.layer()
        self.assertEqual(self.refused_off(), self.unfinished())
        self.assertFalse(self.state_file().exists())
        self.assertTrue(features.standby(self.home))
        self.adopt(stage, "20260101T000100Z")
        self.assertIsNone(adopt.unfinished(self.home))
        moved = self.start_after_adopt()
        self.assertTrue(moved[0].startswith(f"moved: {self.config_file}, "), moved)
        self.clock = LATER
        self.standby_off()
        self.assertFalse(adopt.marked(self.home))
        self.assertFalse(features.standby(self.home))

    def test_an_internal_state_moved_aside_after_an_adoption_keeps_standby(self):
        """An earlier monitor with no config.json brings no move to hold standby: the mark of the adoption does. An
        internal state that cannot be read counts as standby, and its fix, moving it aside, leaves none: the mark
        keeps the monitor in standby until standby off, which ends standby and removes the mark.
        """
        self.adopt(self.earlier(config_json=False), "20260101T000000Z")
        self.assertIsNone(adopt.unfinished(self.home))
        self.layer()
        self.assertEqual(self.start_after_adopt(), [])
        self.assertFalse(MOVE.pending(self.home))
        p = self.state_file()
        p.write_text("{ not JSON", encoding="utf-8")
        self.assertTrue(features.standby(self.home))
        p.rename(p.with_name(p.name + ".aside"))  # the internal state's own fix
        self.assertTrue(features.standby(self.home))
        self.clock = LATER
        self.standby_off()
        self.assertFalse(adopt.marked(self.home))
        self.assertFalse(features.standby(self.home))

    def test_an_adopt_stopped_before_the_move_was_due_again_holds_standby(self):
        """An apply that stops once the earlier monitor's state and config.json are in place, but before the move is
        made due again: no start would split that config.json, so standby off refuses, nothing changed, its fix adopt
        again: config.json stays unsplit, and the time before which the older monitor's reviews are this deployment's
        own unset. adopt again makes the move due, the start after it makes it, and standby off ends it.
        """
        stage = self.earlier()
        error = MOVE.MoveError(
            "the one-time move cannot be made due again: a made-up reason"
        )
        with mock.patch.object(MOVE, "due_again", side_effect=error):
            with self.assertRaises(adopt.Stopped) as caught:
                self.adopt(stage, "20260101T000000Z")
        self.assertEqual(caught.exception.where, adopt.PLACED)
        self.assertTrue(self.config_file.is_file())
        self.assertFalse(MOVE.due(self.home))
        self.layer()
        self.assertEqual(self.refused_off(), self.unfinished())
        self.assertTrue(self.config_file.is_file())
        self.assertFalse((self.home / MOVE.MIGRATED).exists())
        self.assertFalse(self.until())
        self.assertTrue(features.standby(self.home))
        self.adopt(stage, "20260101T000100Z")
        self.assertTrue(MOVE.due(self.home))
        moved = self.start_after_adopt()
        self.assertTrue(moved[0].startswith(f"moved: {self.config_file}, "), moved)
        self.clock = LATER
        self.standby_off()
        self.assertFalse(adopt.marked(self.home))
        self.assertTrue((self.home / MOVE.MIGRATED).exists())
        self.assertEqual(self.until(), LATER)


class StandbyTest(Base):
    def test_a_pending_move_is_standby_whatever_the_internal_state_says(self):
        self.layer()
        self.older_config()
        # before the move is made
        self.assertTrue(features.standby(self.home))
        self.start()
        self.assertTrue(features.standby(self.home))
        # a moved deployment whose internal state is missing, until the move ends
        self.state_file().unlink()
        self.assertTrue(features.standby(self.home))
        MOVE.end(self.home, RC.load(self.home).values)
        self.assertFalse(MOVE.pending(self.home))
        self.assertFalse(features.standby(self.home))


class EndTest(Base):
    def setUp(self):
        super().setUp()
        self.layer()
        self.data = self.older_config()
        by = {e.name: e for e in sr.entries(RC.load(self.home).values)}
        self.entries = by
        # Set in their own places: the team chat's webhook, the claude login, the write login.
        self.own = [
            self.file(
                self.write_secrets / "teams_webhook", "https://example.com/hook\n"
            ),
            self.file(self.secrets / "claude_key"),
            *(
                self.file(self.write_secrets / "gh" / f)
                for f in by["github_write"].files
            ),
        ]
        # Older copies whose entry is set: they go at the end.
        self.goes = {
            self.file(
                self.secrets / "teams_webhook", "https://example.com/hook\n"
            ): "teams_webhook",
            self.file(self.secrets / "claude_env"): "claude",
            self.file(self.secrets / "gh" / "hosts.yml").parent: "github_write",
        }
        # Older copies of a login the move never copies, neither login set in its own place, and files nothing reads
        # now, the local worker's key among them while no binding names the first review pass's key: they go at the
        # end.
        self.not_copied = {
            self.file(self.secrets / "atlassian" / "credentials").parent: "atlassian",
            self.file(self.secrets / "gh_read_token"): "github_read",
        }
        self.unread = [
            self.file(self.secrets / n)
            for n in ("worker_headers", "jira_mcp_headers", "worker_api_key")
        ]
        # Older copies whose entry is not set in its own place: kept, with the fix.
        self.kept = {
            self.file(self.secrets / "webhost_ed25519"): "web_host_ssh_key",
            self.file(self.secrets / "admin_token"): "admin_access_token",
        }
        self.start()

    def end(self) -> list:
        return MOVE.end(self.home, RC.load(self.home).values)

    def test_the_end_renames_config_json_read_only_and_removes_the_older_copies(self):
        lines = self.end()
        migrated = self.home / MOVE.MIGRATED
        self.assertFalse(self.config_file.exists())
        self.assertEqual(migrated.read_bytes(), self.data)
        self.assertEqual(stat.S_IMODE(migrated.stat().st_mode) & 0o222, 0)
        self.assertEqual(
            lines[0], f"renamed {self.config_file} {MOVE.MIGRATED}, read-only"
        )
        for p, name in self.goes.items():
            self.assertFalse(os.path.lexists(p), p)
            self.assertIn(
                f"removed {p}: an older copy of {name}, which is set in its own place",
                lines,
            )
        for p, name in self.not_copied.items():
            self.assertFalse(os.path.lexists(p), p)
            said = (
                f"removed {p}: an older copy of the login {name}, which the move never copies; the login is not set "
                "in its own place"
            )
            self.assertIn(said, lines)
            # the fix that sets it: for the atlassian login, rkmon login atlassian
            self.assertEqual(
                lines[lines.index(said) + 1],
                f"{FIX_LABEL}{sr.set_fix(self.entries[name])}",
            )
        self.assertIn(
            str(steps.command_line("login atlassian")),
            str(sr.set_fix(self.entries["atlassian"])),
        )
        for p in self.unread:
            self.assertFalse(os.path.lexists(p), p)
            self.assertIn(
                f"removed {p}: an older package kept a secret there, which nothing reads now",
                lines,
            )
        for p in self.own:
            self.assertTrue(p.is_file(), p)
        for p, name in self.kept.items():
            self.assertTrue(p.is_file(), p)
            said = (
                f"kept {p}: an older copy of {name}, as it is not set in its own place; once it is set, the next "
                "start removes this copy"
            )
            self.assertIn(said, lines)
            self.assertEqual(
                lines[lines.index(said) + 1],
                f"{FIX_LABEL}{sr.set_fix(self.entries[name])}",
            )
        rec = MOVE.record(self.home)
        self.assertEqual(rec["ended_at"], MOVED_AT)
        self.assertEqual(rec["end"], lines)
        self.assertEqual(
            sorted((k["entry"], k["place"]) for k in rec["kept"]),
            sorted((name, str(p)) for p, name in self.kept.items()),
        )
        self.assertFalse(MOVE.pending(self.home))
        self.assertFalse(MOVE.due(self.home))

    def test_the_end_is_made_once(self):
        self.end()
        self.clock = LATER
        self.assertEqual(self.end(), [])
        self.assertEqual(MOVE.record(self.home)["ended_at"], MOVED_AT)
        self.assertEqual(
            state_store.read(self.home)["reviews"], {"unnamed_are_ours_until": MOVED_AT}
        )

    def test_an_end_that_cannot_set_the_time_says_so_and_ends_when_run_again(self):
        real = state_store.write_file

        def write_file(file, data):
            if pathlib.Path(file) == self.state_file():
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        self.clock = LATER
        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            with self.assertRaises(MOVE.MoveError) as caught:
                self.end()
        folder = self.state_file().parent
        self.assertEqual(
            str(caught.exception),
            f"the one-time move cannot end: {UNTIL} cannot be set in {folder}: Permission denied",
        )
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {folder} writable for the monitor, then run {STANDBY_OFF}",
        )
        # what the end did before it stopped: the rename, then each older copy it removed
        done = caught.exception.done
        self.assertEqual(
            done[0], f"renamed {self.config_file} {MOVE.MIGRATED}, read-only"
        )
        for p, name in self.goes.items():
            self.assertIn(
                f"removed {p}: an older copy of {name}, which is set in its own place",
                done,
            )
        self.assertNotIn(until_line(LATER), done)
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        self.assertEqual(
            state_store.read(self.home)["reviews"], {"unnamed_are_ours_until": MOVED_AT}
        )
        # the end, run again, ends it
        self.assertEqual(self.end()[-1], until_line(LATER))
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))
        self.assertEqual(
            state_store.read(self.home)["reviews"], {"unnamed_are_ours_until": LATER}
        )

    def test_an_end_whose_internal_state_breaks_its_schema_says_so_with_its_fix(self):
        cfg = RC.load(self.home).values
        body = json.loads(self.state_file().read_text(encoding="utf-8"))
        body["standby"]["on"] = "yes"
        self.state_file().write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(MOVE.MoveError) as caught:
            MOVE.end(self.home, cfg)
        lead = (
            f"the one-time move cannot end: {UNTIL} cannot be set: the internal state {self.state_file()} breaks "
            "its schema: "
        )
        self.assertTrue(str(caught.exception).startswith(lead), caught.exception)
        with self.assertRaises(state_store.StateError) as read:
            state_store.read(self.home)
        self.assertEqual(
            str(caught.exception.fix("./rkmon")), str(read.exception.fix("./rkmon"))
        )
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))

    def test_a_kept_copy_goes_at_the_start_after_its_entry_is_set(self):
        self.end()
        self.file(self.secrets / "admin_access_token")
        admin_token = self.secrets / "admin_token"
        self.assertEqual(
            self.start(),
            [
                f"removed {admin_token}: an older copy of admin_access_token, which is set in its own place"
            ],
        )
        self.assertFalse(admin_token.exists())
        rec = MOVE.record(self.home)
        self.assertEqual(
            rec["kept"],
            [
                {
                    "entry": "web_host_ssh_key",
                    "place": str(self.secrets / "webhost_ed25519"),
                }
            ],
        )
        self.assertEqual(self.start(), [])

    def test_an_end_that_cannot_rename_config_json_changes_nothing(self):
        real = os.replace

        def replace(src, dst, *args, **kwargs):
            if pathlib.Path(src) == self.config_file:
                raise PermissionError(errno.EACCES, "Permission denied", str(src))
            return real(src, dst, *args, **kwargs)

        with mock.patch.object(os, "replace", side_effect=replace):
            with self.assertRaises(MOVE.MoveError) as caught:
                self.end()
        self.assertEqual(
            str(caught.exception),
            f"the one-time move cannot end: {self.config_file} cannot be renamed {MOVE.MIGRATED}: Permission denied",
        )
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {self.home} writable for the monitor, then run {STANDBY_OFF}",
        )
        self.assertEqual(caught.exception.done, ())
        self.nothing_changed()
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))

    def test_an_end_whose_record_cannot_be_written_says_what_it_did(self):
        real = state_store.write_file
        record = MOVE.record_file(self.home)

        def write_file(file, data):
            if pathlib.Path(file) == record:
                raise PermissionError(errno.EACCES, "Permission denied", str(file))
            return real(file, data)

        self.clock = LATER
        with mock.patch.object(state_store, "write_file", side_effect=write_file):
            with self.assertRaises(MOVE.MoveError) as caught:
                self.end()
        self.assertEqual(
            str(caught.exception),
            f"the record of the one-time move cannot be written in {record.parent}: Permission denied",
        )
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {record.parent} writable for the monitor, then run {STANDBY_OFF}",
        )
        done = caught.exception.done
        self.assertEqual(
            (done[0], done[-1]),
            (
                f"renamed {self.config_file} {MOVE.MIGRATED}, read-only",
                until_line(LATER),
            ),
        )
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))
        # the end, run again, ends it
        self.assertEqual(
            self.end()[0], f"{self.config_file} is not there: nothing is renamed"
        )
        self.assertFalse(MOVE.is_open(MOVE.record(self.home)))

    def test_an_end_that_cannot_take_the_internal_states_lock_changes_nothing(self):
        folder = self.state_file().parent
        lock = folder / state_store.LOCK_FILE
        denied = PermissionError(errno.EACCES, "Permission denied", str(lock))
        with mock.patch.object(state_store, "locked_in", side_effect=denied):
            with self.assertRaises(MOVE.MoveError) as caught:
                self.end()
        self.assertEqual(
            str(caught.exception),
            f"the one-time move cannot end: the internal state's lock cannot be taken in {folder}: "
            "Permission denied",
        )
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {folder} writable for the monitor, then run {STANDBY_OFF}",
        )
        self.assertEqual(caught.exception.done, ())
        self.nothing_changed()
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))

    def test_an_older_copy_the_end_kept_no_longer_sets_its_entry(self):
        cfg = RC.load(self.home).values
        entry = self.entries["web_host_ssh_key"]
        self.assertEqual(logins.state(cfg, entry)[0], "SET")
        self.end()
        self.assertTrue((self.secrets / "webhost_ed25519").is_file())
        self.assertEqual(logins.older_copy(cfg, entry), "")
        self.assertEqual(logins.state(cfg, entry)[0], "MISSING")

    def nothing_changed(self) -> None:
        """config.json and every older copy are as they were."""
        self.assertEqual(self.config_file.read_bytes(), self.data)
        for p in [*self.goes, *self.not_copied, *self.unread, *self.kept]:
            self.assertTrue(os.path.lexists(p), p)

    def test_an_end_while_the_move_is_due_and_not_made_says_so_with_its_fix(self):
        """The start could not make the move, so there is no record: the monitor stays in standby while the move is
        pending, and the end refuses rather than let standby off end a standby it cannot end.
        """
        MOVE.record_file(self.home).unlink()
        self.assertTrue(MOVE.due(self.home))
        with self.assertRaises(MOVE.MoveError) as caught:
            self.end()
        self.assertEqual(
            str(caught.exception),
            "the one-time move cannot end: it is not made yet; the monitor's start makes it",
        )
        stop, start = steps.command_line("stop"), steps.command_line("start")
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"restart the monitor, whose start makes the move: {stop}, then {start}; then run {STANDBY_OFF}",
        )
        self.assertEqual(caught.exception.done, ())
        self.nothing_changed()
        self.assertIsNone(MOVE.record(self.home))
        self.assertTrue(features.standby(self.home))
        # the restart makes the move, and the end then ends it
        self.assertTrue(self.start()[0].startswith(f"moved: {self.config_file}, "))
        self.assertEqual(
            self.end()[0], f"renamed {self.config_file} {MOVE.MIGRATED}, read-only"
        )

    def test_an_end_whose_configuration_names_no_secrets_folder_says_so_with_its_fix(
        self,
    ):
        cfg = dict(RC.load(self.home).values, paths={})
        with self.assertRaises(paths.PathsError):
            paths.get(cfg, "secrets_dir")
        with self.assertRaises(MOVE.MoveError) as caught:
            MOVE.end(self.home, cfg)
        self.assertEqual(
            str(caught.exception),
            "the one-time move cannot end: the configuration names no secrets folder, so the older copies of the "
            "secrets cannot be found",
        )
        doctor = steps.command_line("doctor")
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"run {doctor} and put right what it says, then run {STANDBY_OFF}",
        )
        self.assertEqual(caught.exception.done, ())
        self.nothing_changed()
        self.assertTrue(MOVE.is_open(MOVE.record(self.home)))

    def test_the_end_gives_config_json_the_name_provision_reads_as_an_older_setup(self):
        self.assertIn(MOVE.MIGRATED, provision.OLDER_SETUP)


class BoundKeyTest(Base):
    """A binding names the first review pass's key: the local worker's key (worker_api_key) is its older copy, which
    the end keeps while the key is not set in its own place, and a later start removes once it is.
    """

    # Your settings with a binding for the first review pass, whose key is first_review_pass_api_key.
    BOUND = dict(
        SETTINGS,
        model_bindings={
            "worker": {
                "provider": "openai_compatible",
                "model_id": "drafter-model-a",
                "base_url": "https://models.example.com/v1",
                "api_key_secret_name": sr.FIRST_PASS_KEY,
            }
        },
        features={
            "review_automation": {"passes": {"first_review_pass": {"model": "worker"}}}
        },
    )

    def test_the_end_keeps_the_older_key_until_the_first_review_pass_key_is_set(self):
        self.layer(self.BOUND)
        self.older_config()
        older = self.file(self.secrets / "worker_api_key")
        self.start()
        cfg = RC.load(self.home).values
        entry = {e.name: e for e in sr.entries(cfg)}[sr.FIRST_PASS_KEY]
        lines = MOVE.end(self.home, cfg)
        said = (
            f"kept {older}: an older copy of {sr.FIRST_PASS_KEY}, as it is not set in its own place; once it is set, "
            "the next start removes this copy"
        )
        self.assertIn(said, lines)
        self.assertEqual(
            lines[lines.index(said) + 1], f"{FIX_LABEL}{sr.set_fix(entry)}"
        )
        self.assertTrue(older.is_file())
        self.assertEqual(
            MOVE.record(self.home)["kept"],
            [{"entry": sr.FIRST_PASS_KEY, "place": str(older)}],
        )
        # once the key is set in its own place, the next start removes the older copy
        self.file(pathlib.Path(sr.path(cfg, entry)))
        self.assertEqual(
            self.start(),
            [
                f"removed {older}: an older copy of {sr.FIRST_PASS_KEY}, which is set in its own place"
            ],
        )
        self.assertFalse(older.exists())
        self.assertEqual(MOVE.record(self.home)["kept"], [])


if __name__ == "__main__":
    unittest.main()
