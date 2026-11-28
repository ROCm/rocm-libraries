# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import atexit
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from rkmon_lib import derived, host_keys, publish

_fd, KEY = tempfile.mkstemp(
    prefix="rkmon_test_key_"
)  # the key must be set (a file that is not blank) for the team dashboard to be set up
os.write(_fd, b"not a real key\n")
os.close(_fd)
atexit.register(os.unlink, KEY)
LINE = "web.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExamplePublicKeyNotReal0000000000000000"
CFG = {
    "features": {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
    "publish": {
        "html": {
            "host": "web.example",
            "user": "op",
            "key": KEY,
            "dest": "site/dash/index.html",
            "host_keys": [LINE],  # the web host's own key, recorded: ssh reaches it
        }
    },
}


class Done:
    def __init__(self, rc, err=""):
        self.returncode, self.stderr, self.stdout = rc, err, ""


class PublishTest(unittest.TestCase):
    def test_nothing_configured_pushes_nothing(self):
        calls = []
        self.assertIsNone(
            publish.html({}, "/w/x.html", run=lambda *a, **k: calls.append(a))
        )
        self.assertEqual(calls, [])

    def test_the_page_goes_to_the_configured_host_by_rsync_over_the_named_key(self):
        seen = {}

        def run(cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw
            return Done(0)

        self.assertIsNone(publish.html(CFG, "/w/x.html", run=run))
        cmd = seen["cmd"]
        self.assertEqual(cmd[0], "rsync")
        self.assertIn("--chmod=F644", cmd)
        self.assertEqual(cmd[-2:], ["/w/x.html", "op@web.example:site/dash/index.html"])
        ssh = cmd[cmd.index("-e") + 1]
        self.assertIn("BatchMode=yes", ssh)
        self.assertIn(f"-i {shlex.quote(KEY)} ", ssh)
        self.assertTrue(seen["kw"]["timeout"] <= 60)

    def test_a_push_or_a_read_waits_for_the_web_host_s_own_key_to_be_recorded(self):
        """While the state records no key of the web host, the only keys ssh trusts it by, a push of the page and a
        read of the requests file are refused before rsync runs, in plain words with the command that records one,
        through the check the remote GPU tests' ssh asks too (host_keys.unrecorded).
        """
        from rkmon_lib import host_keys

        ran = []
        html = dict(
            CFG["publish"]["html"], host_keys=[], requests="site-requests.jsonl"
        )
        cfg = dict(CFG, publish={"html": html})
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            why = f"the web host's own key is not recorded: {host_keys.record_fix()}"
            self.assertEqual(
                publish.html(cfg, "/w/x.html", run=lambda *a, **k: ran.append(a)), why
            )
            self.assertEqual(
                publish.read_requests(cfg, run=lambda *a, **k: ran.append(a)),
                ("", why),
            )
            self.assertEqual(host_keys.unrecorded(cfg, host_keys.WEB), why)
            self.assertEqual(host_keys.unrecorded(CFG, host_keys.WEB), "")
        self.assertEqual(ran, [])  # rsync did not run

    def test_a_failed_push_returns_its_last_error_line_and_never_raises(self):
        self.assertEqual(
            publish.html(
                CFG,
                "/w/x.html",
                run=lambda *a, **k: Done(255, "warn\nssh: timed out\n"),
            ),
            "ssh: timed out",
        )

        def boom(*a, **k):
            raise subprocess.TimeoutExpired("rsync", 45)

        self.assertTrue(
            publish.html(CFG, "/w/x.html", run=boom).startswith("TimeoutExpired")
        )

    def test_nothing_is_published_in_mirror_mode_or_while_stopped(self):
        """A mirror would overwrite the live monitor's team page, and a stopped monitor sends nothing out: the page is
        not pushed, the reason is said, and each change of the outcome is one journal line.
        """
        import pathlib
        from unittest import mock

        from rkmon_lib.store import Store, empty_state

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            pushed = []
            cfg = dict(
                CFG,
                features={"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            )
            with mock.patch.object(
                publish, "html", lambda c, path: pushed.append(path)
            ):
                why = publish.team(
                    dict(cfg, mirror={"enabled": True}), st, store, "/w/x.html"
                )
                self.assertIn("this monitor is a mirror (config mirror.enabled)", why)
                st["stopped"] = {"at": "2026-01-01T00:00:00Z", "how": "stop"}
                self.assertIn("stopped", publish.team(cfg, st, store, "/w/x.html"))
                self.assertIn("stopped", publish.team(cfg, st, store, "/w/x.html"))
                self.assertEqual(pushed, [])
                st.pop("stopped")
                self.assertIsNone(publish.team(cfg, st, store, "/w/x.html"))
                self.assertIsNone(publish.team(cfg, st, store, "/w/x.html"))
                self.assertEqual(pushed, ["/w/x.html"] * 2)
                self.assertIsNone(
                    publish.team(dict(cfg, features={}), st, store, "/w/x.html")
                )  # off: nothing to say
            lines = [
                ln
                for ln in store.journal_path.read_text().splitlines()
                if " publish " in ln
            ]
            self.assertEqual(
                len(lines), 3, lines
            )  # mirror, stopped, published: once each
            self.assertTrue(lines[-1].endswith("team dashboard published"))

    def test_in_prototype_and_in_standby_the_page_is_kept_and_not_pushed(self):
        """While the team dashboard does not send, the page made after every tick is kept as the one unsent
        page item, replaced at each tick, and nothing is written to the web host; the journal says so once per change,
        in the gate's line when the page item is new, else in publish's. Once standby ends, the page goes, and no unsent
        item is sent."""
        import pathlib
        from unittest import mock

        from rkmon_lib import state_store
        from rkmon_lib.store import Store, empty_state

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            home = pathlib.Path(d) / "home"
            st = empty_state()
            pushed = []
            proto = dict(
                CFG,
                features={
                    "dashboards": {"team": {"enabled": True, "mode": "prototype"}}
                },
            )
            with mock.patch.object(
                publish, "html", lambda c, path: pushed.append(path)
            ), mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
                for _tick in range(2):
                    self.assertIsNone(publish.team(proto, st, store, "/w/x.html"))
                (page,) = state_store.unsent_items(store.dir)
                self.assertEqual(
                    (
                        page["type"],
                        page["for"],
                        page["file"],
                        page["content"],
                        page["why"],
                    ),
                    (
                        state_store.PAGE,
                        "the team dashboard's web host",
                        "/w/x.html",
                        "",
                        state_store.PROTOTYPE,
                    ),
                )
                state_store.set(home, "standby.on", True)
                self.assertIsNone(publish.team(CFG, st, store, "/w/x.html"))
                self.assertEqual(
                    [
                        (i["type"], i["why"])
                        for i in state_store.unsent_items(store.dir)
                    ],
                    [(state_store.PAGE, state_store.STANDBY)],
                )
                self.assertEqual(
                    len(state_store.unsent_path(store.dir).read_text().splitlines()), 1
                )  # one page item, replaced
                self.assertEqual(pushed, [])
                state_store.set(home, "standby.on", False)
                self.assertIsNone(publish.team(CFG, st, store, "/w/x.html"))
                self.assertEqual(pushed, ["/w/x.html"])
                # Back to prototype, live, and prototype again: the second time, the page item kept is the same one.
                for cfg in (proto, CFG, proto):
                    self.assertIsNone(publish.team(cfg, st, store, "/w/x.html"))
                self.assertEqual(pushed, ["/w/x.html"] * 2)
            journal = [  # (kind, text)
                ln.split(None, 2)[1:]
                for ln in store.journal_path.read_text().splitlines()
            ]
            self.assertEqual(
                [text for kind, text in journal if kind == "unsent"],
                [
                    "page push for the team dashboard's web host: not sent (the team dashboard in prototype)",
                    "page push for the team dashboard's web host: not sent (the monitor in standby)",
                    "page push for the team dashboard's web host: not sent (the team dashboard in prototype)",
                ],
            )
            published = [text for kind, text in journal if kind == "publish"]
            # Kept as a new page item, the page has the gate's line alone; kept as the same item again, publish's.
            self.assertEqual(published[:2], ["team dashboard published"] * 2)
            self.assertEqual(len(published), 3, published)
            self.assertTrue(
                published[2].startswith(
                    "team dashboard not published: UNSENT: the team dashboard's page for the team dashboard's web "
                    "host, kept as an unsent item: features.dashboards.team.mode is prototype"
                ),
                published,
            )

    def test_served_here_the_page_is_served_in_prototype(self):
        """Served by this monitor, the page leaves no machine: it is made and served in prototype as in live, and kept
        as no unsent item."""
        import pathlib

        from rkmon_lib import state_store
        from rkmon_lib.store import Store, empty_state

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            page = store.dir / "published.html"
            page.write_text("<html>the page</html>", encoding="utf-8")
            cfg = {
                "features": {
                    "dashboards": {"team": {"enabled": True, "mode": "prototype"}}
                },
                "team_dashboard": {"mode": "serve"},
            }
            self.assertIsNone(publish.team(cfg, empty_state(), store, str(page)))
            self.assertEqual(
                (publish.site_dir(store.dir) / "index.html").read_text(
                    encoding="utf-8"
                ),
                "<html>the page</html>",
            )
            self.assertEqual(state_store.unsent_items(store.dir), [])

    def test_served_here_the_page_and_the_outcome_are_written_whole(self):
        """Served by this monitor, the page is written whole at 644 and the publish outcome at the state's shared mode,
        whatever the umask, with nothing left beside either."""
        from rkmon_lib import paths
        from rkmon_lib.store import Store, empty_state

        umask = os.umask(0o077)
        self.addCleanup(os.umask, umask)
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            page = store.dir / "published.html"
            page.write_text("<html>the page</html>", encoding="utf-8")
            cfg = {
                "features": {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
                "team_dashboard": {"mode": "serve"},
            }
            self.assertIsNone(publish.team(cfg, empty_state(), store, str(page)))
            site = publish.site_dir(store.dir)
            self.assertEqual([p.name for p in site.iterdir()], ["index.html"])
            self.assertEqual(stat.S_IMODE((site / "index.html").stat().st_mode), 0o644)
            outcome = store.dir / publish.STATUS
            self.assertEqual(outcome.read_text(encoding="utf-8"), "published\n")
            self.assertEqual(
                stat.S_IMODE(outcome.stat().st_mode),
                paths.worker_module("perms").SHARED_FILE_MODE,
            )
            self.assertEqual(
                sorted(p.name for p in store.dir.iterdir() if p.is_file()),
                sorted([page.name, outcome.name, store.journal_path.name]),
            )  # files only: the store makes folders of its own

    def test_priority_requests_are_read_back_over_the_same_key(self):
        """Other teams' requests from the published page wait for the operator's OK."""
        cfg = {
            "features": {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            "publish": {
                "html": dict(
                    CFG["publish"]["html"],
                    requests="rocke-monitor/requests/priority_requests.jsonl",
                )
            },
        }
        seen = {}

        def run(cmd, **kw):
            seen["cmd"] = cmd
            return Done(0)

        self.assertIsNone(publish.pull_requests(cfg, "/w/r.jsonl", run=run))
        self.assertEqual(
            seen["cmd"][-2:],
            [
                "op@web.example:rocke-monitor/requests/priority_requests.jsonl",
                "/w/r.jsonl",
            ],
        )
        self.assertIn("BatchMode=yes", seen["cmd"][seen["cmd"].index("-e") + 1])
        # Nobody has asked yet: no file there, and no error.
        missing = (
            'rsync: [sender] link_stat "/home/op/rocke-monitor/requests/priority_requests.jsonl" failed: '
            "No such file or directory (2)\nrsync error: some files/attrs were not transferred (see previous errors) "
            "(code 23)"
        )
        self.assertIsNone(
            publish.pull_requests(
                cfg, "/w/r.jsonl", run=lambda *a, **k: Done(23, missing)
            )
        )
        # A missing key is an error, though its message says "No such file or directory" too.
        nokey = "Warning: Identity file /k not accessible: No such file or directory.\nPermission denied (publickey)."
        self.assertEqual(
            publish.pull_requests(
                cfg, "/w/r.jsonl", run=lambda *a, **k: Done(255, nokey)
            ),
            "Permission denied (publickey).",
        )
        calls = []  # not configured: nothing read
        self.assertIsNone(
            publish.pull_requests(
                CFG, "/w/r.jsonl", run=lambda *a, **k: calls.append(a)
            )
        )
        # In prototype the page is not on the web host: its requests file is not read (they are taken in where the
        # page is served).
        proto = dict(
            cfg,
            features={"dashboards": {"team": {"enabled": True, "mode": "prototype"}}},
        )
        self.assertIsNone(
            publish.pull_requests(
                proto, "/w/r.jsonl", run=lambda *a, **k: calls.append(a)
            )
        )
        self.assertEqual(calls, [])

    def test_a_broken_pipe_names_the_cause_printed_before_it(self):
        """A broken pipe names the cause rsync printed before it, never only "rsync: [Receiver] write error: Broken
        pipe (32)"."""
        cfg = {
            "features": {"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            "publish": {
                "html": dict(
                    CFG["publish"]["html"],
                    requests="rocke-monitor/requests/priority_requests.jsonl",
                )
            },
        }
        pipe = "rsync: [Receiver] write error: Broken pipe (32)"
        err = f"rrsync error: sending files is disabled\n{pipe}\n"
        self.assertEqual(
            publish.pull_requests(cfg, "/w/r.jsonl", run=lambda *a, **k: Done(12, err)),
            f"rrsync error: sending files is disabled ({pipe})",
        )
        self.assertEqual(
            publish.pull_requests(
                cfg, "/w/r.jsonl", run=lambda *a, **k: Done(12, pipe)
            ),
            pipe,
        )


def fake_ssh_tools(scanned=(LINE,), readable=True):
    """ssh-keyscan (the host's keys: `scanned`) and ssh-keygen -lf (a fingerprint; with READABLE false, none: exit
    255), with no network and no ssh on the machine; any other command fails the test.
    """
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "ssh-keyscan":
            return subprocess.CompletedProcess(
                argv,
                0,
                "".join(x + "\n" for x in scanned),
                "# web.example:22 SSH-2.0\n",
            )
        if argv[:2] == ["ssh-keygen", "-lf"] and not readable:
            return subprocess.CompletedProcess(
                argv, 255, "", "(stdin) is not a public key file.\n"
            )
        if argv[:2] == ["ssh-keygen", "-lf"]:
            return subprocess.CompletedProcess(
                argv, 0, "256 SHA256:exampleFingerprint web.example (ED25519)\n", ""
            )
        raise AssertionError(argv)

    return run, calls


# while a test watches a folder: the folder, then each path in it opened (_audit)
_WATCHED = []
_AUDITING = []  # [True] once _audit is this process's audit hook


def _audit(event, args):
    """An audit hook (sys.addaudithook), which stays for the process: each file opened in the folder a test watches
    goes to _WATCHED; nothing while no test watches one."""
    if _WATCHED and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(args[0]))
        if path.startswith(_WATCHED[0]):
            _WATCHED.append(path)


class HostKeysTest(unittest.TestCase):
    """ssh never asks (BatchMode), and trusts the web host by its own keys the state records alone
    (publish.html.host_keys; rkmon_lib/host_keys.py): never by a known_hosts file the settings name, nor
    by ssh's own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        # a space in the working folder's name: ssh splits an option's value at white space, so the option that
        # names the known hosts file in it quotes its path (tests/test_host_keys.py holds the rule)
        self.work = self.d / "a work folder"
        self.known = self.work / derived.WEB_HOST_KNOWN_HOSTS

    def cfg(self, *keys, **html):
        """CFG in the working folder, with KEYS the web host's own keys the state records."""
        html = dict(CFG["publish"]["html"], host_keys=list(keys), **html)
        return dict(CFG, paths={"working_dir": str(self.work)}, publish={"html": html})

    def test_ssh_trusts_the_web_host_by_the_recorded_keys_alone(self):
        cfg = self.cfg(LINE)
        argv = publish.ssh_argv(cfg)
        self.assertEqual(
            argv,
            [
                "ssh",
                "-i",
                KEY,
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f'UserKnownHostsFile="{self.known}"',
                "-o",
                f"GlobalKnownHostsFile={os.devnull}",
                "-o",
                "KnownHostsCommand=none",
                "-o",
                "VerifyHostKeyDNS=no",
                "-o",
                "ControlPath=none",
            ],
        )
        self.assertEqual(self.known.read_text(), LINE + "\n")
        self.assertEqual(stat.S_IMODE(self.known.stat().st_mode), 0o644)
        seen = {}

        def run(cmd, **kw):
            seen["cmd"] = cmd
            return Done(0)

        self.assertIsNone(publish.html(cfg, "/w/x.html", run=run))
        # rsync splits -e as the shell would: each option stays one
        self.assertEqual(shlex.split(seen["cmd"][seen["cmd"].index("-e") + 1]), argv)
        # The file holds the recorded keys and nothing else: one that holds another key is written whole again.
        self.known.write_text(LINE + "\nother.example ssh-ed25519 AAAAOtherKey\n")
        publish.ssh_argv(cfg)
        self.assertEqual(self.known.read_text(), LINE + "\n")
        # A known_hosts file the settings name is not read.
        given = str(self.d / "given_known_hosts")
        self.assertEqual(publish.ssh_argv(self.cfg(LINE, known_hosts=given)), argv)
        # No key recorded, or no working folder known: ssh trusts no host.
        for c in (
            self.cfg(),
            self.cfg(known_hosts=given),
            dict(CFG, publish=self.cfg(LINE)["publish"]),
        ):
            self.assertIn(f"UserKnownHostsFile={os.devnull}", publish.ssh_argv(c))

    def test_the_doctor_fails_while_the_web_host_s_own_key_is_not_recorded(self):
        """The doctor's row reads the state: FAIL while no key of the web host is recorded, its fix the command that
        shows the host's key and records it once you accept it; PASS once one is; no row while nothing is pushed.
        """
        from rkmon_lib import doctor

        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                doctor.webhost_key_checks(
                    dict(
                        self.cfg(),
                        features={"dashboards": {"team": {"enabled": True}}},
                    )
                ),
                [],
            )  # in prototype nothing is pushed, so the web host's key is not needed
            (c,) = doctor.webhost_key_checks(self.cfg())
        self.assertEqual((c.name, c.status), ("team dashboard's web host key", "FAIL"))
        self.assertEqual(
            c.found,
            "web.example: its own key is not recorded; it blocks the team dashboard: ssh trusts the host by no other "
            "key, so every push fails host key verification",
        )
        self.assertEqual(
            str(c.fix), "./rkmon login shows it, and records it once you accept it"
        )
        (c,) = doctor.webhost_key_checks(self.cfg(LINE))
        self.assertEqual(
            (c.status, c.found),
            (
                "PASS",
                "web.example: its own key is recorded (1 line(s)), the only key ssh trusts it by",
            ),
        )
        # A key recorded for the host the settings named before is none of this host's.
        (c,) = doctor.webhost_key_checks(self.cfg(LINE, host="web2.example"))
        self.assertEqual(c.status, "FAIL")
        self.assertTrue(
            c.found.startswith("web2.example: its own key is not recorded"), c.found
        )
        self.assertEqual(doctor.webhost_key_checks(dict(self.cfg(), features={})), [])
        self.assertFalse(self.known.exists())  # the row reads the state alone

    def test_nothing_opens_a_file_in_ssh_s_own_folder(self):
        """ssh's own known_hosts (~/.ssh, which names the web host here) is never read: neither ssh's options, nor
        the doctor's row, nor the read of the host's own keys `rk host-key` shows, nor a push opens a file there.
        """
        from rkmon_lib import doctor

        user = self.d / "user"
        (user / ".ssh").mkdir(parents=True)
        (user / ".ssh" / "known_hosts").write_text(LINE + "\n")
        run, _ = fake_ssh_tools()
        if not _AUDITING:
            sys.addaudithook(_audit)
            _AUDITING.append(True)
        seen = []
        with mock.patch.dict(os.environ, {"HOME": str(user)}):
            _WATCHED[:] = [str(user / ".ssh")]
            try:
                for keys in ((), (LINE,)):
                    cfg = self.cfg(*keys)
                    seen += publish.ssh_argv(cfg)
                    doctor.webhost_key_checks(cfg)
                    host_keys.read(cfg, host_keys.WEB, run=run)
                    publish.html(cfg, "/w/x.html", run=lambda *a, **k: Done(0))
            finally:
                opened = _WATCHED[1:]
                _WATCHED.clear()
        self.assertEqual(opened, [])
        self.assertNotIn(".ssh", " ".join(seen))


if __name__ == "__main__":
    unittest.main()
