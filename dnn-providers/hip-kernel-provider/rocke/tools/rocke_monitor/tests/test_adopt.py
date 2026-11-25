# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/adopt.py: moving an earlier monitor's state into a deployment, over a made-up earlier monitor (invented
paths and PR numbers): what moves and what is left behind (secrets, by each name the secrets registry gives them, the
older ones too; locks; the job folders), the work in flight cleared, standby set in the staged internal state with the
adoption recorded beside it, an earlier internal state that cannot be read refused, its config.json staged, the plan,
and the apply that sets this deployment's copies aside, rewrites the old paths from both configurations and --map, puts
config.json in the monitor's home and makes the one-time move due again, each file it writes at the mode its folder
gives the monitor's files; an OSError, or a move it cannot make due again, stops it with what it did, where that leaves
the state, and the fix. The mark of the adoption, written in the monitor's home before apply moves anything and
finished once apply has, keeps the monitor in standby however apply ends, until standby off removes it.
"""

import errno
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import (  # noqa: E402
    adopt,
    config,
    features,
    one_time_move,
    orchestrator,
    paths,
    provision,
    secrets_registry,
    state_store,
    states,
    steps,
    store,
)

perms = paths.worker_module("perms")

AT = "2026-01-02T03:00:00Z"  # store.now: the staging's time


def write(p: pathlib.Path, text: str) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


class TestAdopt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = pathlib.Path(self.tmp.name)
        self.old = t / "old-monitor"
        self.outputs = t / "old-outputs"
        self.work = t / "work"
        old = self.old
        cfg = {
            "paths": {
                "working_dir": str(old),
                "outputs_dir": str(self.outputs),
                "daily_dir": str(self.outputs / "daily"),
            }
        }
        write(old / "monitor" / "config.json", json.dumps(cfg))
        state = {
            "prs": {
                "101": {"status": "open", "ours": {"head": "aaa"}},
                "202": {"status": "merged", "ours": {}},
            },
            "review_pipes": [{"pr": 101}],
            "review_ready": {"202": {}},
            "review_pipeline": {"pr": 101},
            "jobs": {"gate": {"id": "j1"}, "history": [{"id": "j0"}]},
            "stopped": {"how": "stop"},
            "holds": {"303": {}},
            "last_tick": "2026-01-01T00:00:00Z",
            "worktree": f"{old}/wt101",
        }
        write(old / "monitor" / "state" / "state.json", json.dumps(state))
        write(
            old / "monitor" / "state" / "actions.jsonl",
            json.dumps({"log": f"{old}/workers/pr101/log"}) + "\n",
        )
        write(old / "monitor" / "state" / "journal.md", "a journal\n")
        write(
            old / "monitor" / "state" / "spend_cache.json", json.dumps({f"{old}/x": 1})
        )
        for name in (
            "lock",
            "review_driver.lock",
            "web.pid",
            "tick_loop.log",
            "state.json.bak-1",
            "gh_logins.json",
        ):
            write(old / "monitor" / "state" / name, "x")
        write(old / "monitor" / "state" / "jobs" / "j1" / "log", "job log")
        write(
            old / "monitor" / "state" / "teams_webhook",
            "https://hooks.example.invalid/secret",
        )
        write(old / "probes" / "p.json", json.dumps({"tree": f"{old}/probes/p"}))
        write(old / "lessons" / "notes" / "n1.md", "note\n")
        write(old / "cache" / "ci.json", "{}")
        write(old / "cache" / "mutants" / "m.json", "{}")
        write(old / "cache" / "clash" / "big.bin", "rebuilt by the scans")
        write(old / "scratch" / "post" / "pr101_fr.md", "a post\n")
        write(old / ".config" / "rkmon" / "admin_token", "secret")
        write(self.outputs / "PR_101_review.md", "posted\n")
        write(self.outputs / "rocke_merge_queue.md", "the dashboard, rebuilt\n")
        write(self.outputs / "daily" / "2026-01-01.md", "a day\n")
        self.targets = {
            "state": str(self.work / "monitor" / "state"),
            "home": str(self.work / "monitor"),
            "probes": str(self.work / "probes"),
            "lessons": str(self.work / "lessons"),
            "cache": str(self.work / "cache"),
            "posts": str(self.work / "scratch" / "post"),
            "posted": str(self.work),
            "daily": str(self.work / "rocke_daily"),
            "spend": str(self.work / "rocke_spend"),
            "workers": str(self.work / "workers"),
            "root": str(self.work),
            "outputs": str(self.work),
        }
        # This deployment's own state before the move.
        write(self.work / "monitor" / "state" / "state.json", json.dumps({"prs": {}}))
        write(self.work / "monitor" / "state" / "scaler_state.json", '{"mine": true}')
        write(self.work / "lessons" / "lessons.md", "the skill's lessons\n")
        write(self.work / "lessons" / "notes" / "n1.md", "this deployment's note\n")

    def stage(self, maps=()):
        config_json = self.old / "monitor" / "config.json"
        cfg = json.loads(config_json.read_text()) if config_json.is_file() else {}
        folders = adopt.old_folders(self.old, cfg)
        self.stage_dir = pathlib.Path(self.tmp.name) / "stage"
        return adopt.stage(folders, self.stage_dir, maps)

    def test_the_old_folders_come_from_its_configuration(self):
        f = adopt.old_folders(
            self.old, json.loads((self.old / "monitor" / "config.json").read_text())
        )
        self.assertEqual(f["state"], str(self.old / "monitor" / "state"))
        self.assertEqual(f["daily"], str(self.outputs / "daily"))
        self.assertEqual(f["posted"], str(self.outputs))
        self.assertEqual(f["posts"], str(self.old / "scratch" / "post"))
        self.assertEqual(adopt.old_folders(self.old, {})["posted"], str(self.old))

    def test_the_state_folder_is_the_one_paths_names(self):
        """The old monitor's state folder is the folder of its home that paths.STATE names, the name's one home."""
        with mock.patch.object(paths, "STATE", "kept"):
            self.assertEqual(
                adopt.old_folders(self.old, {})["state"],
                str(self.old / "monitor" / "kept"),
            )

    def test_stage_leaves_secrets_locks_and_jobs_behind_and_clears_the_work_in_flight(
        self,
    ):
        m = self.stage()
        s = self.stage_dir
        self.assertTrue((s / "state" / "actions.jsonl").is_file())
        for gone in (
            "lock",
            "review_driver.lock",
            "web.pid",
            "tick_loop.log",
            "state.json.bak-1",
            "gh_logins.json",
            "teams_webhook",
        ):
            self.assertFalse((s / "state" / gone).exists(), gone)
        self.assertFalse((s / "state" / "jobs").exists())
        self.assertFalse((s / "cache" / "clash").exists())
        self.assertTrue((s / "cache" / "mutants" / "m.json").is_file())
        self.assertEqual(
            sorted(p.name for p in (s / "posted").iterdir()), ["PR_101_review.md"]
        )
        self.assertFalse(any(".config" in str(p) for p in s.rglob("*")))
        st = json.loads((s / "state" / "state.json").read_text())
        self.assertEqual(
            (st["review_pipes"], st["review_ready"], st["review_pipeline"]),
            ([], {}, None),
        )
        self.assertEqual(st["jobs"], {"gate": None, "history": [{"id": "j0"}]})
        self.assertNotIn("stopped", st)
        self.assertEqual(st["holds"], {"303": {}})
        self.assertEqual(m["cleared"]["review_pipes"], [101])
        self.assertEqual(m["records"]["with_our_review"], 1)
        report = "\n".join(adopt.stage_report(m))
        self.assertIn(" open, 1 with our review), ", report)
        self.assertIn("Left behind: every secret and login", report)
        self.assertNotIn("hooks.example.invalid", json.dumps(m))

    def test_the_claude_session_on_record_stays_on_the_host_that_made_it(self):
        """The orchestrator's session on record (rkmon_lib/orchestrator.py STATE) names a Claude conversation, which
        stays on the host that made it, so this deployment's orchestrator could never resume it: it is left behind,
        and the report's line of what is left behind says so."""
        state = self.old / "monitor" / "state"
        write(
            state / orchestrator.STATE,
            json.dumps({"session_id": "0a1b2c3d-0000-4000-8000-000000000000"}),
        )
        m = self.stage()
        self.assertTrue((self.stage_dir / "state" / "journal.md").is_file())
        self.assertFalse((self.stage_dir / "state" / orchestrator.STATE).exists())
        self.assertIn(str(state / orchestrator.STATE), m["left"])
        left = next(x for x in adopt.stage_report(m) if x.startswith("Left behind: "))
        self.assertIn(
            "the Claude session on record (a Claude conversation stays on the host that made it: the orchestrator "
            "here starts a new one)",
            left,
        )

    def test_what_the_secrets_registry_names_is_left_behind_wherever_it_is(self):
        """The secrets registry says which files and folders may hold a secret (secrets_registry.holds_secret): each
        secret's file and its copies, a login's files and folders, a model binding's key, a key file. The rest of a
        folder moves."""
        state = self.old / "monitor" / "state"
        left = (
            "claude_env",
            "teams_webhook.old",
            "gh-read-token",
            "glm_api_key",
            "hosts.yml",
            "credentials",
            "id_ed25519",
            "site.pem",
            "gh-read/config.yml",
            "atlassian/settings.json",
            "secrets/notes.md",
            "notes/hosts.yml",
            "notes/webhost_ed25519.pub",
        )
        for name in left:
            write(state / name, "made-up secret")
        write(state / "notes" / "kept.md", "a note\n")
        self.stage()
        s = self.stage_dir / "state"
        for name in left:
            self.assertFalse((s / name).exists(), name)
        for folder in ("gh-read", "atlassian", "secrets"):
            self.assertFalse((s / folder).exists(), folder)
        self.assertEqual(sorted(p.name for p in (s / "notes").iterdir()), ["kept.md"])

    def test_plan_then_apply(self):
        self.stage(maps=[("/srv/old-temp", "/ramdisk/tmp")])
        lines = "\n".join(adopt.plan(self.stage_dir, self.targets))
        self.assertIn("this deployment's 0 PR records (0 with our review) -> ", lines)
        self.assertIn(f"{self.old} -> {self.work}", lines)
        self.assertIn("/srv/old-temp -> /ramdisk/tmp", lines)
        home = self.work / "monitor"
        self.assertFalse((self.work / "probes").exists())  # the plan writes nothing
        self.assertFalse(adopt.marked(home))
        out = adopt.apply(
            self.stage_dir, self.targets, self.work, now="20260101T000000Z"
        )
        backup = self.work / "adopt-backup-20260101T000000Z"
        self.assertIn(f"APPLIED: this deployment's own copies are in {backup}", out)
        st = json.loads((self.work / "monitor" / "state" / "state.json").read_text())
        self.assertEqual(sorted(st["prs"]), ["101", "202"])
        self.assertEqual(
            st["worktree"], f"{self.work}/wt101"
        )  # the old paths, rewritten
        self.assertIn(
            f"{self.work}/workers/pr101/log",
            (self.work / "monitor" / "state" / "actions.jsonl").read_text(),
        )
        self.assertIn(
            str(self.old),
            (self.work / "monitor" / "state" / "spend_cache.json").read_text(),
        )  # history
        self.assertEqual(
            (self.work / "monitor" / "state" / "scaler_state.json").read_text(),
            '{"mine": true}',
        )  # kept
        self.assertEqual(
            json.loads((backup / "state" / "state.json").read_text()), {"prs": {}}
        )
        self.assertEqual(
            (self.work / "lessons" / "lessons.md").read_text(), "the skill's lessons\n"
        )  # merged in
        self.assertEqual(
            (self.work / "lessons" / "notes" / "n1.md").read_text(), "note\n"
        )
        self.assertEqual(
            (backup / "lessons" / "notes" / "n1.md").read_text(),
            "this deployment's note\n",
        )
        self.assertTrue((self.work / "rocke_daily" / "2026-01-01.md").is_file())
        self.assertTrue((self.work / "PR_101_review.md").is_file())
        self.assertTrue((self.work / "monitor" / "state" / "jobs").is_dir())
        self.assertEqual(
            json.loads((self.work / "probes" / "p.json").read_text()),
            {"tree": f"{self.work}/probes/p"},
        )
        # the mark of the adoption says that apply finished, and keeps the monitor in standby until standby off
        mark = json.loads(adopt.mark_file(home).read_text())
        self.assertEqual(mark["kind"], adopt.MARK_KIND)
        self.assertTrue(mark["applied_at"])
        self.assertIsNone(adopt.unfinished(home))
        self.assertTrue(features.standby(home))

    def test_bad_input(self):
        with self.assertRaises(adopt.AdoptError):
            adopt.parse_maps(["relative=/x"])
        self.assertEqual(adopt.parse_maps(["/a/=/b"]), [("/a", "/b")])
        with self.assertRaises(adopt.AdoptError):
            adopt.stage(
                {"state": str(self.old / "nowhere")}, pathlib.Path(self.tmp.name) / "s2"
            )
        with self.assertRaises(adopt.AdoptError):
            adopt.plan(pathlib.Path(self.tmp.name) / "nothing", self.targets)

    def test_each_name_the_secrets_registry_gives_a_secret_is_left_behind(self):
        """Each secret's own file or folder and each of its older names (secrets_registry.entries, older_names), and
        the files older packages kept secrets in (OLDER_SECRET_FILES): none moves, in any folder; the rest does.
        """
        state = self.old / "monitor" / "state"
        folders = secrets_registry.secret_folders()
        names = [
            name
            for e in secrets_registry.entries({})
            for name in (e.place or e.name, *secrets_registry.older_names(e))
        ]
        names += secrets_registry.OLDER_SECRET_FILES
        for where in (state, state / "notes"):
            for name in names:
                write(
                    where / name / "hosts.yml" if name in folders else where / name,
                    "made-up secret",
                )
        write(state / "notes" / "kept.md", "a note\n")
        self.stage()
        staged = self.stage_dir / "state"
        for where in (staged, staged / "notes"):
            for name in names:
                self.assertFalse(os.path.lexists(where / name), where / name)
        self.assertEqual(
            sorted(p.name for p in (staged / "notes").iterdir()), ["kept.md"]
        )

    def test_stage_sets_standby_and_records_the_adoption_in_the_staged_internal_state(
        self,
    ):
        """The staged internal state is in standby, with adopt's reason and the staging's time, and
        records the adoption beside it (provision.ADOPTED_AT); the earlier work volume's setup record stays behind, and
        what the earlier monitor learned comes along. Its own internal state is read, never written, and no lock is
        staged."""
        old_home = self.old / "monitor"
        names = ["acme-review-bot"]
        for name, value in (
            ("reviews.automated_reviewer_names_used", names),
            (provision.PACKAGE_VERSION, "1999.01.01"),
            (provision.AT, "2025-12-01T00:00:00Z"),
        ):
            state_store.set(old_home, name, value)
        before = state_store.path(old_home).read_bytes()
        with mock.patch.object(store, "now", return_value=AT):
            self.stage()
        staged = state_store.read(self.stage_dir)
        self.assertEqual(
            staged["standby"], {"on": True, "reason": adopt.REASON, "since": AT}
        )
        self.assertEqual(staged["provisioned"], {"adopted_at": AT})
        self.assertEqual(staged["reviews"]["automated_reviewer_names_used"], names)
        self.assertEqual(state_store.path(old_home).read_bytes(), before)
        self.assertEqual([p.name for p in self.stage_dir.rglob("*.lock")], [])

    def test_an_earlier_internal_state_that_cannot_be_read_is_refused(self):
        """Standby cannot be set in an internal state that cannot be read: the stage refuses, and names the earlier
        monitor's file, with the internal state's own fix, then adopt again."""
        bad = write(state_store.path(self.old / "monitor"), "{ not JSON")
        with self.assertRaises(adopt.AdoptError) as caught:
            self.stage()
        text = str(caught.exception)
        self.assertTrue(text.startswith(f"the internal state {bad} is no JSON: "), text)
        self.assertTrue(
            text.endswith(", so standby cannot be set in the adopted state"), text
        )
        with self.assertRaises(state_store.StateError) as read:
            state_store.read(self.old / "monitor")
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"{read.exception.fix('./rkmon')}, then run {again} again",
        )

    def test_config_json_comes_along_and_the_move_is_made_due_again(self):
        """The earlier monitor's config.json is staged with its state; the plan says where it goes and that the
        deployment starts in standby; apply puts it in the monitor's home, its old paths rewritten and this
        deployment's own set aside, then makes the one-time move due again (one_time_move.due_again), so that the next
        start splits it, the work volume not set up until that start's run.
        """
        source = self.old / "monitor" / "config.json"
        m = self.stage()
        self.assertEqual(m["config_json"], str(source))
        self.assertEqual(
            adopt.staged_config(self.stage_dir).read_bytes(), source.read_bytes()
        )
        self.assertIn(f"  config.json: from {source}", adopt.stage_report(m))
        home = self.work / "monitor"
        here = write(home / "config.json", '{"mine": true}')
        lines = adopt.plan(self.stage_dir, self.targets)
        self.assertIn(
            f"{here}: the earlier monitor's comes in, this deployment's going aside; the next start splits it by "
            "owner, once (the one-time move)",
            lines,
        )
        self.assertIn(f"the deployment starts {states.in_standby()}", lines)
        out = adopt.apply(
            self.stage_dir, self.targets, self.work, now="20260101T000000Z"
        )
        backup = self.work / "adopt-backup-20260101T000000Z"
        self.assertEqual(
            json.loads(here.read_text())["paths"],
            {
                "working_dir": str(self.work),
                "outputs_dir": str(self.work),
                "daily_dir": str(self.work / "rocke_daily"),
            },
        )
        self.assertEqual(
            (backup / "home" / "config.json").read_text(), '{"mine": true}'
        )
        put, due, applied = out[-3:]
        self.assertTrue(put.startswith(f"{here}: the earlier monitor's, "), put)
        self.assertTrue(
            put.endswith(
                " old path(s) rewritten in it; this deployment's is set aside"
            ),
            put,
        )
        self.assertTrue(
            due.startswith(
                f"the one-time move is due again: the next start splits {here} by owner, once"
            ),
            due,
        )
        self.assertEqual(
            applied, f"APPLIED: this deployment's own copies are in {backup}"
        )
        self.assertTrue(provision.older_install(home))
        self.assertIsNone(provision.provisioned(home))
        self.assertIsNone(one_time_move.record(home))
        self.assertTrue(state_store.read(home)["standby"]["on"])

    def test_with_no_config_json_no_move_is_made_due(self):
        """An earlier monitor with no config.json brings none: apply leaves this deployment's own where it is, and
        the adopted work volume is no older install's for it, now or once the next start sets it up.
        """
        (self.old / "monitor" / "config.json").unlink()
        home = self.work / "monitor"
        own = write(home / "config.json", '{"mine": true}')
        m = self.stage()
        self.assertEqual(m["config_json"], "")
        self.assertIn("  config.json: none in the copy", adopt.stage_report(m))
        self.assertIn(
            "config.json: none in the earlier monitor's copy, so no one-time move",
            adopt.plan(self.stage_dir, self.targets),
        )
        out = adopt.apply(
            self.stage_dir, self.targets, self.work, now="20260101T000000Z"
        )
        self.assertEqual(own.read_text(), '{"mine": true}')
        self.assertEqual([line for line in out if "one-time move" in line], [])
        self.assertFalse(provision.older_install(home))
        self.assertIsNone(provision.provisioned(home))
        self.assertTrue(state_store.read(home)["standby"]["on"])

    def test_a_move_that_cannot_be_made_due_again_stops_apply_with_the_state_in_place(
        self,
    ):
        """When the move cannot be made due again, apply stops once the earlier monitor's state is in place
        (Stopped, PLACED): the error, with its fix, and the lines of what it did."""
        self.stage()
        error = one_time_move.MoveError(
            "the one-time move cannot be made due again: a made-up reason"
        )
        with mock.patch.object(one_time_move, "due_again", side_effect=error):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        backup = self.work / "adopt-backup-20260101T000000Z"
        here = self.work / "monitor" / "config.json"
        done = caught.exception.done
        self.assertIs(caught.exception.error, error)
        self.assertEqual(str(caught.exception), str(error))
        self.assertEqual(caught.exception.where, adopt.PLACED)
        self.assertTrue(done[-2].startswith(f"{here}: the earlier monitor's, "), done)
        self.assertEqual(done[-1], f"this deployment's own copies are in {backup}")
        self.assertTrue(here.is_file())
        # the mark says that the adopt stopped part-way: standby off refuses until adopt again finishes it
        self.assertEqual(
            str(adopt.unfinished(self.work / "monitor")),
            f"{steps.command_line('adopt')} stopped part-way",
        )

    def test_a_full_volume_stops_apply_with_what_it_did_where_the_state_is_and_the_fix(
        self,
    ):
        """An OSError as apply copies stops it (Stopped), never a traceback: here the volume is full as this
        deployment's own copy of a lesson goes aside, once the earlier monitor's state is in place. It says what it
        did, that the earlier monitor's state is in place, in standby, and the fix: space on that volume, then adopt
        again."""
        self.stage()
        lessons = self.work / "lessons"
        copy2 = shutil.copy2

        def full(src, dst, **kwargs):
            if pathlib.Path(src).is_relative_to(lessons):
                # as shutil's copy says it: the file it read, then the one it could not write
                no_space = errno.ENOSPC
                raise OSError(no_space, os.strerror(no_space), str(src), None, str(dst))
            return copy2(src, dst, **kwargs)

        with mock.patch.object(shutil, "copy2", side_effect=full):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        stopped = caught.exception
        backup = self.work / "adopt-backup-20260101T000000Z"
        state = self.work / "monitor" / "state"
        aside = backup / "lessons" / "notes" / "n1.md"
        files = sum(1 for p in (self.stage_dir / "state").rglob("*") if p.is_file())
        self.assertEqual(stopped.where, adopt.PLACED)
        self.assertEqual(
            stopped.done,
            (
                f"{state}: replaced ({files} files)",
                f"{self.work / 'probes'}: replaced (1 files)",
                f"this deployment's own copies are in {backup}",
            ),
        )
        self.assertEqual(
            str(stopped), f"adopt stopped at {aside}: {os.strerror(errno.ENOSPC)}"
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(stopped.error.fix("./rkmon")),
            f"free space on the volume of {aside.parent}, then run {again} again",
        )
        self.assertTrue(state_store.read(self.work / "monitor")["standby"]["on"])

    def test_a_full_volume_as_the_new_state_is_built_stops_apply_with_nothing_changed(
        self,
    ):
        """The new state is built inside the state folder first (adopt.BUILD): a build the volume cannot hold, here
        once two of its files are written, stops apply (Stopped) with nothing changed: no line of what it did, no
        backup folder, and this deployment's state as it was, the build removed. The error names the file where it
        was going; the fix: space on that volume, then adopt again."""
        self.stage()
        state = self.work / "monitor" / "state"
        before = sorted((p.name, p.read_bytes()) for p in state.iterdir())
        copyfile = shutil.copyfile

        def full(src, dst, **kwargs):
            if pathlib.Path(dst) == state / adopt.BUILD / "journal.md":
                no_space = errno.ENOSPC
                raise OSError(no_space, os.strerror(no_space), str(src), None, str(dst))
            return copyfile(src, dst, **kwargs)

        with mock.patch.object(shutil, "copyfile", side_effect=full):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        stopped = caught.exception
        self.assertEqual((stopped.where, stopped.done), (adopt.NOTHING, ()))
        self.assertEqual(
            str(stopped),
            f"adopt stopped at {state / 'journal.md'}: {os.strerror(errno.ENOSPC)}",
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(stopped.error.fix("./rkmon")),
            f"free space on the volume of {state}, then run {again} again",
        )
        self.assertEqual(
            sorted((p.name, p.read_bytes()) for p in state.iterdir()), before
        )
        self.assertEqual(list(self.work.glob("adopt-backup-*")), [])
        self.assertFalse(adopt.marked(self.work / "monitor"))

    def test_a_swap_cut_short_leaves_the_state_folder_part_way_and_in_standby(self):
        """The built state goes in once this deployment's own has gone aside, the internal state first: a swap cut
        short after it stops apply (Stopped, PART_WAY), the folder in standby already, with the line that says where
        this deployment's own copies are; the fix makes the folder writable, then adopt again.
        """
        self.stage()
        state = self.work / "monitor" / "state"
        rename = pathlib.Path.rename

        def cut(src, dst):
            if pathlib.Path(dst) == state / "actions.jsonl":
                denied = errno.EACCES
                raise OSError(denied, os.strerror(denied), str(src), None, str(dst))
            return rename(src, dst)

        with mock.patch.object(pathlib.Path, "rename", autospec=True, side_effect=cut):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        stopped = caught.exception
        backup = self.work / "adopt-backup-20260101T000000Z"
        self.assertEqual(
            (stopped.where, stopped.done),
            (adopt.PART_WAY, (f"this deployment's own copies are in {backup}",)),
        )
        self.assertEqual(
            str(stopped),
            f"adopt stopped at {state / 'actions.jsonl'}: {os.strerror(errno.EACCES)}",
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(stopped.error.fix("./rkmon")),
            f"make {state} writable for the monitor, then run {again} again",
        )
        self.assertEqual(
            json.loads((backup / "state" / "state.json").read_text()), {"prs": {}}
        )
        self.assertTrue(state_store.read(self.work / "monitor")["standby"]["on"])
        self.assertEqual(
            str(adopt.unfinished(self.work / "monitor")),
            f"{steps.command_line('adopt')} stopped part-way",
        )

    def test_an_apply_cut_short_as_the_state_goes_aside_stays_in_standby_by_the_mark(
        self,
    ):
        """Ctrl-C, or a kill, as this deployment's own state goes aside, here once its internal state has and before
        the earlier monitor's is in: the state folder holds no internal state, and the mark of the adoption, written
        before the swap, keeps the monitor in standby (features.standby), as it does whatever the internal state says;
        it says that the adopt stopped part-way, its fix adopt again (adopt.unfinished).
        """
        home = self.work / "monitor"
        state_store.set(home, "standby.on", False)  # this deployment's own: it sends
        self.assertFalse(features.standby(home))
        self.stage()
        move = shutil.move

        def cut(src, dst, *args, **kwargs):
            # this deployment's own entries go aside in their names' order: internal_state.json before state.json
            if pathlib.Path(src).name == "state.json":
                raise KeyboardInterrupt
            return move(src, dst, *args, **kwargs)

        with mock.patch.object(shutil, "move", side_effect=cut):
            with self.assertRaises(KeyboardInterrupt):
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        self.assertFalse(state_store.path(home).exists())
        self.assertTrue(features.standby(home))
        unfinished = adopt.unfinished(home)
        self.assertEqual(
            str(unfinished), f"{steps.command_line('adopt')} stopped part-way"
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(str(unfinished.fix("./rkmon")), f"run {again} again")
        # adopt again, from the same copy, finishes it: the mark says so, and standby holds until standby off
        adopt.apply(self.stage_dir, self.targets, self.work, now="20260101T000100Z")
        self.assertIsNone(adopt.unfinished(home))
        self.assertTrue(features.standby(home))

    def test_a_mark_that_cannot_be_written_stops_apply_with_nothing_changed(self):
        """The mark of the adoption is written before the state folder's swap: a home that cannot take it stops
        apply (Stopped) with nothing changed: no line of what it did, no backup folder, no mark, and this deployment's
        state as it was, the build removed. The error names the mark's file; the fix: the home made writable for the
        monitor, then adopt again."""
        self.stage()
        home = self.work / "monitor"
        state = home / "state"
        before = sorted((p.name, p.read_bytes()) for p in state.iterdir())
        write_file = state_store.write_file

        def denied(file, data):
            if pathlib.Path(file) == adopt.mark_file(home):
                raise PermissionError(
                    errno.EACCES, os.strerror(errno.EACCES), str(file)
                )
            return write_file(file, data)

        with mock.patch.object(state_store, "write_file", side_effect=denied):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        stopped = caught.exception
        self.assertEqual((stopped.where, stopped.done), (adopt.NOTHING, ()))
        self.assertEqual(
            str(stopped),
            f"adopt stopped at {adopt.mark_file(home)}: {os.strerror(errno.EACCES)}",
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(stopped.error.fix("./rkmon")),
            f"make {home} writable for the monitor, then run {again} again",
        )
        self.assertEqual(
            sorted((p.name, p.read_bytes()) for p in state.iterdir()), before
        )
        self.assertEqual(list(self.work.glob("adopt-backup-*")), [])
        self.assertFalse(adopt.marked(home))

    def test_a_mark_that_cannot_say_apply_finished_stops_it_with_the_state_in_place(
        self,
    ):
        """The mark says that apply finished last: when it cannot, apply stops (Stopped, PLACED) with what it did,
        and the mark it wrote before the swap still says that the adopt stopped part-way, so standby holds and standby
        off refuses until adopt again finishes it."""
        self.stage()
        home = self.work / "monitor"
        write_file = state_store.write_file

        def denied(file, data):
            if pathlib.Path(file) == adopt.mark_file(home) and data["applied_at"]:
                raise PermissionError(
                    errno.EACCES, os.strerror(errno.EACCES), str(file)
                )
            return write_file(file, data)

        with mock.patch.object(state_store, "write_file", side_effect=denied):
            with self.assertRaises(adopt.Stopped) as caught:
                adopt.apply(
                    self.stage_dir, self.targets, self.work, now="20260101T000000Z"
                )
        stopped = caught.exception
        backup = self.work / "adopt-backup-20260101T000000Z"
        self.assertEqual(stopped.where, adopt.PLACED)
        self.assertEqual(
            stopped.done[-1], f"this deployment's own copies are in {backup}"
        )
        self.assertEqual(
            str(stopped),
            f"adopt stopped at {adopt.mark_file(home)}: {os.strerror(errno.EACCES)}",
        )
        again = steps.command_line("adopt --from DIR")
        self.assertEqual(
            str(stopped.error.fix("./rkmon")),
            f"make {home} writable for the monitor, then run {again} again",
        )
        self.assertEqual(
            str(adopt.unfinished(home)),
            f"{steps.command_line('adopt')} stopped part-way",
        )
        self.assertTrue(features.standby(home))

    def test_the_mark_of_an_adoption(self):
        """The mark in the monitor's home (adopt.MARK_FILE): one that says apply has not finished, one that cannot be
        read and one that is no mark each say why the adoption has not finished, the fix adopt again (unfinished); one
        that says apply finished, or none, says nothing. unmark removes it, and says so with its fix when it cannot.
        """
        home = self.work / "monitor"
        p = adopt.mark_file(home)
        self.assertEqual(p, home / adopt.MARK_FILE)
        self.assertEqual((adopt.marked(home), adopt.unfinished(home)), (False, None))
        again = f"run {steps.command_line('adopt --from DIR')} again"
        for text, said in (
            (
                json.dumps({"kind": adopt.MARK_KIND, "applied_at": ""}),
                f"{steps.command_line('adopt')} stopped part-way",
            ),
            ("{ not JSON", f"{p} cannot be read as the mark of an adoption: "),
            (json.dumps({"kind": "other", "applied_at": AT}), f"{p} is no mark"),
            (json.dumps([AT]), f"{p} is no mark of an adoption"),
        ):
            write(p, text)
            with self.subTest(text=text):
                self.assertTrue(adopt.marked(home))
                why = adopt.unfinished(home)
                self.assertTrue(str(why).startswith(said), why)
                self.assertEqual(str(why.fix("./rkmon")), again)
        write(p, json.dumps({"kind": adopt.MARK_KIND, "applied_at": AT}))
        self.assertIsNone(adopt.unfinished(home))
        denied = PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(p))
        with mock.patch.object(pathlib.Path, "unlink", side_effect=denied):
            with self.assertRaises(config.ConfigError) as caught:
                adopt.unmark(home)
        off = steps.command_line("standby off")
        self.assertEqual(
            str(caught.exception),
            f"the mark of the adoption, {p}, cannot be removed: {os.strerror(errno.EACCES)}",
        )
        self.assertEqual(
            str(caught.exception.fix("./rkmon")),
            f"make {home} writable for the monitor, then run {off}",
        )
        self.assertTrue(adopt.marked(home))
        adopt.unmark(home)
        self.assertFalse(adopt.marked(home))
        adopt.unmark(home)  # none: nothing to remove

    def test_what_apply_writes_takes_the_modes_the_monitor_gives_its_files(self):
        """A file apply writes takes the mode its folder gives a file the monitor makes (perms.UMASK, and the
        folder's default ACL where the file system keeps one), never the stage's 0644; a file it writes over keeps its
        own mode; the state folder it replaces stays itself, with its mode; and each file keeps its times.
        """
        umask = os.umask(perms.UMASK)  # as the monitor's processes run
        self.addCleanup(os.umask, umask)
        when = 1_700_000_000
        for name in (
            "monitor/state/actions.jsonl",
            "monitor/state/journal.md",
            "lessons/notes/n1.md",
        ):
            os.utime(self.old / name, (when, when))
        state = self.work / "monitor" / "state"
        state.chmod(perms.SHARED_DIR_MODE)
        try:
            perms.set_default_acl(state, perms.SHARED)
        except OSError as exc:  # a file system that keeps no ACL: the umask alone
            if exc.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
        own = self.work / "lessons" / "notes" / "n1.md"
        own.chmod(perms.SHARED_FILE_MODE)
        folder = state.stat()
        self.stage()
        adopt.apply(self.stage_dir, self.targets, self.work, now="20260101T000000Z")
        made = state / "made-by-the-monitor"
        made.write_text("x", encoding="utf-8")
        mode = stat.S_IMODE(made.stat().st_mode)
        self.assertNotEqual(mode, 0o644)
        for name in (
            "state.json",
            "actions.jsonl",
            "journal.md",
            state_store.STATE_FILE,
        ):
            self.assertEqual(stat.S_IMODE((state / name).stat().st_mode), mode, name)
        self.assertEqual(
            (own.read_text(), stat.S_IMODE(own.stat().st_mode)),
            ("note\n", perms.SHARED_FILE_MODE),
        )
        after = state.stat()
        self.assertEqual(
            (after.st_ino, stat.S_IMODE(after.st_mode)),
            (folder.st_ino, perms.SHARED_DIR_MODE),
        )
        for p in (state / "actions.jsonl", state / "journal.md", own):
            self.assertEqual(int(p.stat().st_mtime), when, p)


if __name__ == "__main__":
    unittest.main()
