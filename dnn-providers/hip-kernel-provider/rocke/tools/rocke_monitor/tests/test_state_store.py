# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The internal state's one writer (rkmon_lib/state_store.py): the file is read and written
by the configuration's names, each state key in its place in the file; a key that is not state is refused, naming who
sets it; a value the schema refuses is refused, with the fix its caller gives, and nothing is written; the file is
written whole, at 0660 whatever the umask, under a lock that keeps two writers from losing each other's change; a
file that breaks its schema cannot be read, and each refusal to read the file names its fix; so does a write that its
lock or its folder keeps out (unwritable). A staged copy, which no monitor reads yet, is written as set writes it, with
no lock and no worker module, its maker's alone.
"""

import errno
import json
import os
import pathlib
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    breakage,
    checks,
    config,
    deploy_tree,
    file_types,
    paths,
    registry,
    repository_defaults,
    settings_file,
    state_store,
)

perms = paths.worker_module("perms")


class StateStoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = pathlib.Path(tmp.name)

    def file(self) -> dict:
        return json.loads(state_store.path(self.home).read_text(encoding="utf-8"))

    def put_file(self, body) -> None:
        p = state_store.path(self.home)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body), encoding="utf-8")

    def test_the_file_is_in_the_homes_state_folder(self):
        self.assertEqual(
            state_store.path(self.home),
            config.state_dir(self.home) / "internal_state.json",
        )
        self.assertEqual(config.state_dir(self.home), self.home / "state")

    def test_a_home_with_no_state_reads_as_none(self):
        self.assertEqual(state_store.read(self.home), {})
        self.assertFalse(state_store.path(self.home).exists())

    def test_a_key_set_is_read_back_by_its_name(self):
        state_store.set(self.home, "standby.on", True)
        state_store.set(self.home, "standby.reason", "a reason")
        self.assertEqual(
            state_store.read(self.home), {"standby": {"on": True, "reason": "a reason"}}
        )
        self.assertEqual(
            self.file(),
            {
                "kind": "internal_state",
                "schema_version": 1,
                "standby": {"on": True, "reason": "a reason"},
            },
        )

    def test_each_key_is_in_its_place_in_the_file(self):
        state_store.set(self.home, "clone_since", "2026-01-02")
        state_store.set(
            self.home, "publish.html.host_keys", ["<host> ssh-ed25519 AAAA"]
        )
        body = self.file()
        self.assertEqual(body["repository"], {"clone_since": "2026-01-02"})
        self.assertEqual(
            body["team_dashboard"], {"web_host_keys": ["<host> ssh-ed25519 AAAA"]}
        )
        self.assertEqual(
            state_store.read(self.home),
            {
                "clone_since": "2026-01-02",
                "publish": {"html": {"host_keys": ["<host> ssh-ed25519 AAAA"]}},
            },
        )

    def test_the_local_workers_binding_is_the_workers_and_no_other_binding_is_read(
        self,
    ):
        self.put_file(
            {
                "kind": "internal_state",
                "schema_version": 1,
                "model_bindings": {"drafter": {"chat_path": "/v1/other"}},
            }
        )
        state_store.set(self.home, "worker.chat_path", "/v1/chat/completions")
        self.assertEqual(
            self.file()["model_bindings"],
            {
                deploy_tree.WORKER: {"chat_path": "/v1/chat/completions"},
                "drafter": {"chat_path": "/v1/other"},
            },
        )
        self.assertEqual(
            state_store.read(self.home),
            {"worker": {"chat_path": "/v1/chat/completions"}},
        )

    def test_an_entry_of_a_state_keys_mapping_is_set_alone(self):
        state_store.set(self.home, "jira.accounts.<user_login>", "account-1")
        state_store.set(self.home, "jira.accounts.<other_login>", "account-2")
        self.assertEqual(
            state_store.read(self.home),
            {
                "jira": {
                    "accounts": {
                        "<user_login>": "account-1",
                        "<other_login>": "account-2",
                    }
                }
            },
        )

    def test_every_state_key_takes_its_default(self):
        for key in registry.keys():
            if key.owner == registry.STATE:
                state_store.set(self.home, key.name, key.default)
                self.assertEqual(
                    registry.get(state_store.read(self.home), key.name),
                    key.default,
                    key.name,
                )

    def test_a_key_that_is_not_state_is_refused_naming_who_sets_it(self):
        # A package key the package's repository defaults set too, and one they do not.
        shared = repository_defaults.keys()[0]
        self.assertNotIn("loop_heartbeat_min", repository_defaults.keys())
        cases = {
            "repo": (
                registry.DEPLOY,
                "your deploy settings file sets it (repository.name)",
            ),
            "loop_heartbeat_min": (
                registry.PACKAGE,
                "the package sets it (config.defaults.json)",
            ),
            shared: (
                registry.PACKAGE,
                "the package sets it (config.defaults.json, or the repository defaults that apply to your "
                "repository and folder)",
            ),
            "paths.repo": (
                registry.DERIVED,
                "it is computed from other keys at each load",
            ),
        }
        for name, (owner, why) in cases.items():
            self.assertEqual(registry.key(name).owner, owner, name)
            with self.assertRaises(state_store.StateError) as e:
                state_store.set(self.home, name, "x")
            self.assertEqual(str(e.exception), f"{name} is not internal state: {why}")
        for name, why in {
            "standby": "standby is a section, not a key: set its keys one by one",
            "no_such.key": "no_such.key is no key of the configuration",
            "standby.on.inside": "standby.on.inside is no key of the configuration",
            "repo.inside": "repo.inside is no key of the configuration",
        }.items():
            with self.assertRaises(state_store.StateError) as e:
                state_store.set(self.home, name, "x")
            self.assertEqual(str(e.exception), why)
        self.assertFalse(state_store.path(self.home).exists())

    def test_a_jira_account_s_login_is_one_breakage_s_rule_takes(self):
        """The state schema's pattern for a JIRA account's GitHub login, which rk jira-account add applies when it
        records one, is breakage.LOGIN, the rule a develop issue's assignee is checked with (breakage, post), written
        as JSON Schema writes a whole match."""
        schema = file_types.INTERNAL_STATE.load()
        names = settings_file.schema_at(schema, "jira.accounts")["propertyNames"]
        self.assertEqual(names["pattern"], f"^{breakage.LOGIN.pattern}$")

    def test_a_value_the_schema_refuses_is_refused_and_nothing_is_written(self):
        with self.assertRaises(state_store.StateError) as e:
            state_store.set(self.home, "standby.on", "yes")
        self.assertEqual(
            str(e.exception),
            'standby.on is not set: standby.on: "yes" is not true or false',
        )
        self.assertFalse(state_store.path(self.home).exists())
        state_store.set(self.home, "standby.on", True)
        was = state_store.path(self.home).read_bytes()
        with self.assertRaises(state_store.StateError):
            state_store.set(self.home, "jira.accounts.<user_login>", 7)
        self.assertEqual(state_store.path(self.home).read_bytes(), was)

    def test_a_staged_copy_is_written_as_set_writes_it_with_no_lock_its_makers_alone(
        self,
    ):
        """staged: values set in a copy that no monitor reads yet (rkmon adopt's stage, which the host makes with no
        worker/ beside it), each in its place, the file written whole with no lock beside it and at its maker's mode
        alone; refused as set refuses, with nothing written."""
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        no_worker = ImportError("no worker/ beside the package's code")
        with mock.patch.object(paths, "worker_module", side_effect=no_worker):
            state_store.staged(
                self.home, {"standby.on": True, "standby.reason": "a reason"}
            )
            was = state_store.path(self.home).read_bytes()
            with self.assertRaises(state_store.StateError) as e:
                state_store.staged(
                    self.home, {"standby.on": "yes", "clone_since": "2026-01-02"}
                )
        self.assertEqual(
            str(e.exception),
            'standby.on, clone_since are not set: standby.on: "yes" is not true or false',
        )
        self.assertEqual(state_store.path(self.home).read_bytes(), was)
        self.assertEqual(
            self.file(),
            {
                "kind": "internal_state",
                "schema_version": 1,
                "standby": {"on": True, "reason": "a reason"},
            },
        )
        self.assertEqual(
            sorted(p.name for p in config.state_dir(self.home).iterdir()),
            [state_store.STATE_FILE],
        )
        self.assertEqual(
            stat.S_IMODE(state_store.path(self.home).stat().st_mode),
            state_store.STAGED_MODE,
        )

    def test_a_state_error_is_a_configuration_error(self):
        self.assertTrue(issubclass(state_store.StateError, config.ConfigError))

    def test_a_file_that_breaks_its_schema_cannot_be_read_or_changed(self):
        p = state_store.path(self.home)
        for body, found in (
            ("{", "is no JSON"),
            ("[]", "is no JSON object"),
            (
                '{"kind": "applied_settings", "schema_version": 1}',
                "breaks its schema: it is of the kind",
            ),
            (
                '{"kind": "internal_state", "schema_version": 2}',
                "at layout 2, not 'internal_state' at layout 1",
            ),
            (
                '{"kind": "internal_state", "schema_version": 1, "standby": {"on": 1}}',
                "standby.on: 1 is not",
            ),
            (
                '{"kind": "internal_state", "schema_version": 1, "nothing": 1}',
                "nothing: no such key",
            ),
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
            for attempt in (
                lambda: state_store.read(self.home),
                lambda: state_store.set(self.home, "standby.on", True),
            ):
                with self.assertRaises(state_store.StateError) as e:
                    attempt()
                self.assertIn(found, str(e.exception))
                self.assertIn(str(p), str(e.exception))
            self.assertEqual(p.read_text(encoding="utf-8"), body)

    def test_each_refusal_to_read_the_file_names_its_fix(self):
        """config.ConfigError.fix: the file mended where its JSON breaks, or moved aside; each key whose value the
        schema refuses deleted, by name."""
        p = state_store.path(self.home)
        where = settings_file.shown(p)
        aside = f"move it aside: mv {where} {where}.old"
        broken = {
            "kind": "internal_state",
            "schema_version": 1,
            "jira": {"accounts": {"bad login.x": 5}},
            "standby": {"on": 1},
            "nothing": 1,
        }
        for body, fix in (
            ("{", f"mend it at the line and column above, or {aside}"),
            ("[]", aside),
            ('{"kind": "applied_settings", "schema_version": 1}', aside),
            ('{"kind": "internal_state", "schema_version": 2}', aside),
            (
                '{"kind": "internal_state", "schema_version": 1, "nothing": 1}',
                f"delete nothing from {where}",
            ),
            (
                json.dumps(broken),
                f"delete jira.accounts.bad login.x from {where}, delete standby.on from {where} and delete "
                f"nothing from {where}",
            ),
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
            for attempt in (
                lambda: state_store.read(self.home),
                lambda: state_store.set(self.home, "standby.on", True),
            ):
                with self.assertRaises(state_store.StateError) as e:
                    attempt()
                self.assertEqual(str(e.exception.fix("rkmon")), fix, body)
        p.unlink()
        p.mkdir()  # a folder where the file goes
        with self.assertRaises(state_store.StateError) as e:
            state_store.read(self.home)
        p.rmdir()
        self.assertEqual(str(e.exception.fix("rkmon")), aside)

    @unittest.skipIf(os.geteuid() == 0, "root reads a file whatever its mode")
    def test_a_file_whose_mode_keeps_the_monitor_out_names_the_mode_it_takes(self):
        self.put_file({"kind": "internal_state", "schema_version": 1})
        p = state_store.path(self.home)
        p.chmod(0)
        self.addCleanup(p.chmod, 0o600)
        with self.assertRaises(state_store.StateError) as e:
            state_store.read(self.home)
        mode = format(perms.SHARED_FILE_MODE, "o")
        self.assertEqual(
            str(e.exception.fix("rkmon")),
            f"chmod {mode} {settings_file.shown(p)} as its owner: the monitor and the writer share it",
        )

    @unittest.skipIf(os.geteuid() == 0, "root writes in a folder that is read-only")
    def test_a_write_the_lock_or_the_folder_keeps_out_names_the_mode_it_takes(self):
        """set lets the OSError of its lock or of its write through, nothing written; unwritable says it, with the mode
        of the lock file or of the state folder as the fix, and with no fix for an error that no mode makes.
        """
        state_store.set(self.home, "standby.on", True)
        folder = config.state_dir(self.home)
        lock = folder / state_store.LOCK_FILE
        self.addCleanup(os.chmod, folder, 0o770)
        self.addCleanup(os.chmod, lock, 0o660)
        said = f"the internal state {state_store.path(self.home)} cannot be written: "
        for place, mode, shared, back in (
            (folder, 0o555, perms.SHARED_DIR_MODE, 0o770),
            (lock, 0o444, perms.SHARED_FILE_MODE, 0o660),
        ):
            with self.subTest(place=place.name):
                os.chmod(place, mode)
                try:
                    with self.assertRaises(OSError) as e:
                        state_store.set(self.home, "standby.on", False)
                finally:
                    os.chmod(place, back)
                exc = state_store.unwritable(self.home, e.exception)
                self.assertEqual(str(exc), said + os.strerror(errno.EACCES))
                self.assertEqual(
                    str(exc.fix("rkmon")),
                    f"chmod {format(shared, 'o')} {settings_file.shown(place)} as its owner: the monitor and the "
                    "writer share it",
                )
                self.assertIs(state_store.read(self.home)["standby"]["on"], True)
        full = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(folder / "x.new"))
        exc = state_store.unwritable(self.home, full)
        self.assertEqual(str(exc), said + os.strerror(errno.ENOSPC))
        self.assertIsNone(exc.fix("rkmon"))

    def test_a_value_the_schema_refuses_names_the_fix_its_caller_gives(self):
        fix = checks.Command(("rk", "jira-account", "add", "LOGIN", "ACCOUNT"))
        with self.assertRaises(state_store.StateError) as e:
            state_store.set(self.home, "standby.on", "yes", fix=fix)
        self.assertIs(e.exception.fix("rkmon"), fix)
        with self.assertRaises(state_store.StateError) as e:
            state_store.set(self.home, "standby.on", "yes")
        self.assertIsNone(e.exception.fix("rkmon"))

    def test_the_file_and_its_lock_are_0660_whatever_the_umask(self):
        old = os.umask(0o077)
        self.addCleanup(os.umask, old)
        state_store.set(self.home, "standby.on", True)
        for p in (
            state_store.path(self.home),
            state_store.path(self.home).with_name(state_store.LOCK_FILE),
        ):
            self.assertEqual(
                stat.S_IMODE(p.stat().st_mode), perms.SHARED_FILE_MODE, p.name
            )
        self.assertEqual(
            stat.S_IMODE(config.state_dir(self.home).stat().st_mode),
            perms.SHARED_DIR_MODE,
        )

    def test_a_write_leaves_the_old_file_or_the_new_one_and_nothing_beside_it(self):
        state_store.set(self.home, "standby.on", True)
        was = state_store.path(self.home).read_bytes()
        with mock.patch.object(
            os, "fsync", side_effect=OSError(5, "Input/output error")
        ):
            with self.assertRaises(OSError):
                state_store.set(self.home, "standby.on", False)
        self.assertEqual(state_store.path(self.home).read_bytes(), was)
        state_store.write_file(self.home / "other.json", {"b": 1, "a": "é"})
        self.assertEqual(
            (self.home / "other.json").read_text(encoding="utf-8"),
            '{\n "a": "é",\n "b": 1\n}\n',
        )
        self.assertEqual(
            sorted(p.name for p in config.state_dir(self.home).iterdir()),
            sorted([state_store.LOCK_FILE, state_store.STATE_FILE]),
        )

    def test_two_writers_never_lose_each_others_change(self):
        write = state_store.write_file

        def slow(*args):
            time.sleep(
                0.05
            )  # every writer has read the file before the first one writes, unless the lock waits
            write(*args)

        logins = [f"<user_login_{n}>" for n in range(5)]
        with mock.patch.object(state_store, "write_file", side_effect=slow):
            threads = [
                threading.Thread(
                    target=state_store.set,
                    args=(self.home, f"jira.accounts.{login}", "account"),
                )
                for login in logins
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(
            sorted(state_store.read(self.home)["jira"]["accounts"]), sorted(logins)
        )


class UnsentItemsTest(unittest.TestCase):
    """The unsent items: what prototype or standby kept instead of sending, one JSON line each in
    state/unsent.jsonl, kept once, for the package's retention."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = pathlib.Path(tmp.name)
        self.folder = config.state_dir(self.home)

    def item(self, **fields) -> dict:
        return {
            "type": "comment",
            "for": "pull request 7",
            "feature": "merge_queue",
            "why": "prototype",
            "content": "a queue notice",
            **fields,
        }

    def test_an_item_is_one_line_beside_the_internal_state(self):
        self.assertTrue(state_store.keep_unsent(self.folder, self.item()))
        p = state_store.unsent_path(self.folder)
        self.assertEqual(p, config.state_dir(self.home) / "unsent.jsonl")
        (line,) = p.read_text(encoding="utf-8").splitlines()
        kept = json.loads(line)
        self.assertEqual({k: v for k, v in kept.items() if k != "at"}, self.item())
        self.assertRegex(kept["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(state_store.unsent_items(self.folder), [kept])

    def test_the_same_item_is_kept_once(self):
        self.assertTrue(state_store.keep_unsent(self.folder, self.item()))
        self.assertFalse(state_store.keep_unsent(self.folder, self.item()))
        self.assertTrue(
            state_store.keep_unsent(self.folder, self.item(content="another notice"))
        )
        self.assertTrue(state_store.keep_unsent(self.folder, self.item(why="standby")))
        self.assertEqual(
            [i["content"] for i in state_store.unsent_items(self.folder)],
            ["a queue notice", "another notice", "a queue notice"],
        )

    def test_the_team_dashboards_page_is_one_item_replaced_at_each_keep(self):
        """The page is one item, replaced at each tick: one line in the file, the newest page's; a keep is
        new (the caller journals it) only when the page it replaces was another."""
        page = {
            "type": state_store.PAGE,
            "for": "the team dashboard's web host",
            "feature": "dashboards.team",
            "file": "/w/state/published.html",
        }
        self.assertTrue(
            state_store.keep_unsent(self.folder, dict(page, why="prototype"))
        )
        state_store.keep_unsent(self.folder, self.item())
        self.assertTrue(state_store.keep_unsent(self.folder, dict(page, why="standby")))
        self.assertFalse(
            state_store.keep_unsent(self.folder, dict(page, why="standby"))
        )  # the same page: replaced, and not new
        self.assertTrue(
            state_store.keep_unsent(self.folder, dict(page, why="prototype"))
        )
        items = state_store.unsent_items(self.folder)
        self.assertEqual(
            [(i["type"], i["why"]) for i in items],
            [("comment", "prototype"), (state_store.PAGE, "prototype")],
        )
        p = state_store.unsent_path(self.folder)
        self.assertEqual(len(p.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), perms.SHARED_FILE_MODE)

    def test_a_page_replaced_keeps_the_other_items_and_a_line_cut_short_apart(self):
        page = {
            "type": state_store.PAGE,
            "for": "the team dashboard's web host",
            "feature": "dashboards.team",
            "file": "/w/state/published.html",
            "why": "prototype",
        }
        state_store.keep_unsent(self.folder, self.item())
        p = state_store.unsent_path(self.folder)
        with p.open("a", encoding="utf-8") as f:
            f.write('{"type": "comm')  # a write cut short
        state_store.keep_unsent(self.folder, page)
        state_store.keep_unsent(self.folder, self.item(content="the next"))
        self.assertEqual(
            [
                (i["type"], i.get("content", ""))
                for i in state_store.unsent_items(self.folder)
            ],
            [
                ("comment", "a queue notice"),
                (state_store.PAGE, ""),
                ("comment", "the next"),
            ],
        )

    def test_an_item_without_its_fields_or_with_another_why_or_type_is_refused(self):
        for bad in (
            self.item(type=""),
            {k: v for k, v in self.item().items() if k != "for"},
            self.item(why="off"),
            self.item(type="a letter"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                state_store.keep_unsent(self.folder, bad)
        self.assertFalse(state_store.unsent_path(self.folder).exists())

    def test_a_line_cut_short_is_skipped_and_the_next_item_starts_its_own(self):
        state_store.keep_unsent(self.folder, self.item())
        p = state_store.unsent_path(self.folder)
        with p.open("a", encoding="utf-8") as f:
            f.write('{"type": "comm')  # a write cut short
        state_store.keep_unsent(self.folder, self.item(content="the next"))
        self.assertEqual(
            [i["content"] for i in state_store.unsent_items(self.folder)],
            ["a queue notice", "the next"],
        )

    def test_the_file_is_0660_whatever_the_umask(self):
        old = os.umask(0o077)
        try:
            state_store.keep_unsent(self.folder, self.item())
        finally:
            os.umask(old)
        mode = stat.S_IMODE(state_store.unsent_path(self.folder).stat().st_mode)
        self.assertEqual(mode, perms.SHARED_FILE_MODE)

    def test_retention_keeps_the_items_one_rotation_then_drops_them(self):
        from rkmon_lib import retention
        from rkmon_lib.store import Store

        self.assertIn("unsent.jsonl", retention.LOGS)
        store = Store(self.folder)
        small = {"retention": {"log_max_mb": 0.0001}}  # about 100 bytes
        state_store.keep_unsent(self.folder, self.item(content="old " * 40))
        retention.rotate(small, store)
        state_store.keep_unsent(self.folder, self.item(content="newer " * 40))
        self.assertEqual(len(state_store.unsent_items(self.folder)), 2)
        retention.rotate(small, store)
        state_store.keep_unsent(self.folder, self.item(content="newest"))
        self.assertEqual(
            [i["content"].split()[0] for i in state_store.unsent_items(self.folder)],
            ["newer", "newest"],
        )

    def test_a_rotation_waits_for_a_keep_that_writes_the_file_whole(self):
        """retention moves the unsent items under the lock a keep takes: a rotation that starts while the team
        dashboard's page is kept (the file written whole) waits for it, so the lines it moves are never written back,
        and each item is kept once."""
        from rkmon_lib import retention, whole_file
        from rkmon_lib.store import Store

        store = Store(self.folder)
        small = {"retention": {"log_max_mb": 0.0001}}  # about 100 bytes
        state_store.keep_unsent(self.folder, self.item(content="old " * 40))
        page = {
            "type": state_store.PAGE,
            "for": "the team dashboard's web host",
            "feature": "dashboards.team",
            "file": "/w/state/published.html",
            "why": "prototype",
        }
        real, rotations = whole_file.write, []

        def write(*a, **kw):
            """The page's keep writes the file whole, as a rotation starts beside it; with no lock to wait for, the
            rotation is over before the write."""
            t = threading.Thread(target=retention.rotate, args=(small, store))
            rotations.append(t)
            t.start()
            t.join(0.5)
            real(*a, **kw)

        with mock.patch.object(whole_file, "write", write):
            self.assertTrue(state_store.keep_unsent(self.folder, page))
        rotations[0].join(10)
        self.assertFalse(rotations[0].is_alive())
        self.assertEqual(
            [
                (i["type"], i.get("content", "")[:4])
                for i in state_store.unsent_items(self.folder)
            ],
            [("comment", "old "), (state_store.PAGE, "")],
        )
        self.assertFalse(
            state_store.unsent_path(self.folder).exists()
        )  # all of it moved, after the keep


if __name__ == "__main__":
    unittest.main()
