# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rk jira-account (rkmon.py's cmd_jira_account): add records a GitHub login's JIRA account in the internal
state, in place of one recorded already, and keeps the pairs the configuration runs with; list prints them; a login
GitHub does not take is refused with the fix, this command again, and nothing is written; an internal state that
cannot be read, or cannot be written, is refused with its own fix; add takes a pair, and list none.
"""

import contextlib
import errno
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
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    checks,
    container_layout,
    doctor,
    settings_file,
    state_store,
)

# A first setup's answers, with a pair config.json holds: a configuration that runs.
CONFIG = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "operator": {"github_login": "<user_login>"},
    "scope": {"labels": ["team: widgets"]},
    "jira": {"accounts": {"dev-a": "dev-a@example.com"}},
}
# The fix of a refused pair: the command again, with a pair the internal state's schema takes.
AGAIN = (
    "rkmon rk jira-account add LOGIN ACCOUNT again, with LOGIN a GitHub login (letters, digits and hyphens) and "
    "ACCOUNT the JIRA account (no spaces)"
)


class TestJiraAccount(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.home = self.dir / "home"
        self.home.mkdir()
        (self.home / "config.json").write_text(
            json.dumps(dict(CONFIG, paths={"working_dir": str(self.dir / "work")})),
            encoding="utf-8",
        )
        for patch in (
            mock.patch.object(doctor, "deployment_image", return_value=False),
            mock.patch.object(
                container_layout, "IMAGE_VENV", str(self.dir / "no-venv")
            ),
            mock.patch.dict(os.environ, {"RKMON_HOME": str(self.home)}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def rk(self, *args: str) -> tuple:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(["jira-account", *args])
        return code, out.getvalue()

    def test_add_records_the_pair_in_the_internal_state_and_list_prints_them(self):
        self.assertEqual(self.rk("list"), (0, "dev-a dev-a@example.com\n"))
        self.assertEqual(
            self.rk("add", "dev-b", "dev-b@example.com"),
            (0, "dev-b: JIRA account dev-b@example.com recorded\n"),
        )
        both = {"dev-a": "dev-a@example.com", "dev-b": "dev-b@example.com"}
        self.assertEqual(state_store.read(self.home)["jira"]["accounts"], both)
        self.assertEqual(
            self.rk("list"),
            (0, "dev-a dev-a@example.com\ndev-b dev-b@example.com\n"),
        )
        self.assertEqual(
            self.rk("add", "dev-a", "invented-account-id"),
            (
                0,
                "dev-a: JIRA account invented-account-id recorded, in place of dev-a@example.com\n",
            ),
        )
        self.assertEqual(
            state_store.read(self.home)["jira"]["accounts"],
            dict(both, **{"dev-a": "invented-account-id"}),
        )

    def test_a_login_github_does_not_take_is_refused_and_nothing_is_written(self):
        for login, account in (("dev.a", "dev-a@example.com"), ("dev-a", "two words")):
            code, out = self.rk("add", login, account)
            self.assertEqual(code, 4, out)
            self.assertTrue(out.startswith("REFUSED: jira.accounts is not set: "), out)
            self.assertEqual(out.splitlines()[1:], [f"{checks.FIX_LABEL}{AGAIN}"], out)
        self.assertFalse(state_store.path(self.home).exists())

    def test_an_internal_state_that_cannot_be_read_is_refused_with_its_own_fix(self):
        p = state_store.path(self.home)
        p.parent.mkdir(parents=True)
        broken = {"kind": "internal_state", "schema_version": 1, "standby": {"on": 1}}
        p.write_text(json.dumps(broken), encoding="utf-8")
        code, out = self.rk("add", "dev-b", "dev-b@example.com")
        self.assertEqual(code, 4, out)
        self.assertTrue(
            out.startswith(f"REFUSED: the internal state {p} breaks its schema: "), out
        )
        fix = f"delete standby.on from {settings_file.shown(p)}"
        self.assertEqual(out.splitlines()[1:], [f"{checks.FIX_LABEL}{fix}"], out)

    @unittest.skipIf(os.geteuid() == 0, "root writes in a folder that is read-only")
    def test_an_internal_state_that_cannot_be_written_is_refused_with_its_fix(self):
        """An internal state that reads but cannot be written, its folder or its lock file kept from the monitor by its
        mode: add is refused (no error escapes), the reason, then the fix that makes that place writable at the mode
        the monitor and the writer share, a line each, and nothing is written."""
        self.rk("add", "dev-b", "dev-b@example.com")
        p = state_store.path(self.home)
        folder, lock = p.parent, p.parent / state_store.LOCK_FILE
        self.assertTrue(lock.is_file())
        self.addCleanup(os.chmod, folder, 0o770)
        self.addCleanup(os.chmod, lock, 0o660)
        why = f"REFUSED: the internal state {p} cannot be written: {os.strerror(errno.EACCES)}"
        for place, mode, shared, back in (
            (folder, 0o555, "2770", 0o770),
            (lock, 0o444, "660", 0o660),
        ):
            with self.subTest(place=place.name):
                os.chmod(place, mode)
                try:
                    code, out = self.rk("add", "dev-c", "dev-c@example.com")
                finally:
                    os.chmod(place, back)
                self.assertEqual(code, 4, out)
                fix = (
                    f"chmod {shared} {settings_file.shown(place)} as its owner: the monitor and the writer "
                    "share it"
                )
                self.assertEqual(
                    out.splitlines(), [why, f"{checks.FIX_LABEL}{fix}"], out
                )
        self.assertEqual(
            sorted(state_store.read(self.home)["jira"]["accounts"]), ["dev-a", "dev-b"]
        )

    def test_add_takes_a_pair_and_list_none(self):
        for args in (["add", "dev-a"], ["add"], ["list", "dev-a"]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as e:
                rkmon.main(["jira-account", *args])
            self.assertEqual(e.exception.code, 2, args)
            self.assertIn(
                "add takes LOGIN and ACCOUNT; list takes neither", err.getvalue()
            )


if __name__ == "__main__":
    unittest.main()
