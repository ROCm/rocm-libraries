# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The own keys of the hosts the monitor reaches over ssh (rkmon_lib/host_keys.py): read with ssh-keyscan,
shown by fingerprint, recorded in the internal state once accepted, and the only keys ssh trusts each host by; and rk
host-key, which reads them and records them for `rkmon secret NAME FILE`. With fake ssh tools, no network and no ssh
on the machine."""

import contextlib
import io
import json
import os
import pathlib
import stat
import subprocess
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
    deploy_tree,
    derived,
    doctor,
    host_keys,
    state_store,
)

WEB_LINE = "web.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleWebHostKeyNotReal00000000000000"
GPU_LINE = "[login.example.com]:2222 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYExampleNotReal="
# A line of a key's shape that ssh-keygen reads no key in (fake_tools).
UNREADABLE = "web.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5Unreadable"
TYPES = {"ssh-ed25519": "ED25519", "ecdsa-sha2-nistp256": "ECDSA", "ssh-rsa": "RSA"}


def fake_tools(
    scanned=(WEB_LINE,), code=0, err="# web.example.com:22 SSH-2.0-OpenSSH\n"
):
    """ssh-keyscan printing SCANNED and ERR, exit CODE, and ssh-keygen -lf naming each key's fingerprint by the key
    (SHA256:example1 for the first key it reads, ...), as ssh-keygen does: a line that holds "Unreadable" it skips,
    and none read is exit 255; each call's argv and stdin in `calls`. Any other command fails the test.
    """
    calls, keys = [], {}

    def run(argv, **kw):
        calls.append((argv, kw.get("input")))
        if argv[0] == "ssh-keyscan":
            return subprocess.CompletedProcess(
                argv, code, "".join(x + "\n" for x in scanned), err
            )
        if argv == ["ssh-keygen", "-lf", "-"]:
            read = [
                x.split() for x in kw["input"].splitlines() if "Unreadable" not in x
            ]
            out = "".join(
                f"256 SHA256:example{keys.setdefault(k[2], len(keys) + 1)} {k[0]} ({TYPES[k[1]]})\n"
                for k in read
            )
            if not out:
                return subprocess.CompletedProcess(
                    argv, 255, "", "(stdin) is not a public key file.\n"
                )
            return subprocess.CompletedProcess(argv, 0, out, "")
        raise AssertionError(argv)

    return run, calls


class HostKeysTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.work = self.d / "work"

    def cfg(self, web=(), gpu=(), **more):
        """A configuration in the working folder, naming both hosts (the GPU login host on port 2222), with WEB and
        GPU the hosts' own keys the state records."""
        return {
            "paths": {"working_dir": str(self.work)},
            "publish": {"html": {"host": "web.example.com", "host_keys": list(web)}},
            "gpu": {"login_host": "login.example.com", "ssh_port": 2222},
            "remote_gpu_tests": {"login_host_keys": list(gpu)},
            **more,
        }

    def test_the_hosts_by_their_secrets(self):
        self.assertEqual(host_keys.host("web_host_ssh_key"), host_keys.WEB)
        self.assertEqual(host_keys.host("gpu_ssh_key"), host_keys.GPU)
        self.assertIsNone(host_keys.host("teams_webhook"))
        self.assertEqual(
            host_keys.target(self.cfg(), host_keys.WEB), ("web.example.com", 22)
        )
        self.assertEqual(
            host_keys.target(self.cfg(), host_keys.GPU), ("login.example.com", 2222)
        )
        self.assertEqual(host_keys.target({}, host_keys.GPU), ("", 22))

    def test_scan_reads_the_host_s_keys_on_its_port(self):
        run, calls = fake_tools(
            scanned=("# web.example.com:22 SSH-2.0-OpenSSH", WEB_LINE, "", "too short")
        )
        self.assertEqual(host_keys.scan("web.example.com", run=run), ([WEB_LINE], ""))
        self.assertEqual(
            calls[-1][0],
            [
                "ssh-keyscan",
                "-T",
                "10",
                "-t",
                "ed25519,ecdsa,rsa",
                "--",
                "web.example.com",
            ],
        )
        run, calls = fake_tools(scanned=(GPU_LINE,))
        self.assertEqual(
            host_keys.scan("login.example.com", 2222, run=run), ([GPU_LINE], "")
        )
        self.assertEqual(
            calls[-1][0],
            [
                "ssh-keyscan",
                "-T",
                "10",
                "-t",
                "ed25519,ecdsa,rsa",
                "-p",
                "2222",
                "--",
                "login.example.com",
            ],
        )
        # None read: why, as ssh-keyscan says it, else its exit code.
        refused = "connect to host login.example.com port 2222: Connection refused"
        run, _ = fake_tools(
            scanned=(), code=1, err=f"# login.example.com:2222 SSH-2.0\n{refused}\n"
        )
        self.assertEqual(
            host_keys.scan("login.example.com", 2222, run=run), ([], refused)
        )
        run, _ = fake_tools(scanned=(), code=1, err="")
        self.assertEqual(
            host_keys.scan("web.example.com", run=run),
            ([], "no key came (ssh-keyscan exited 1)"),
        )

        def missing(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        lines, why = host_keys.scan("web.example.com", run=missing)
        self.assertEqual(lines, [])
        self.assertTrue(why.startswith("FileNotFoundError: "), why)

    def test_each_key_s_fingerprint(self):
        run, calls = fake_tools()
        self.assertEqual(
            host_keys.fingerprints([WEB_LINE, UNREADABLE, GPU_LINE], run=run),
            ["ED25519 SHA256:example1", "", "ECDSA SHA256:example2"],
        )
        # One ssh-keygen per line: given several, it skips a line it reads no key in, and what it prints no longer
        # says which line each fingerprint is of.
        self.assertEqual(
            calls,
            [
                (["ssh-keygen", "-lf", "-"], f"{x}\n")
                for x in (WEB_LINE, UNREADABLE, GPU_LINE)
            ],
        )

        def broken(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 10)

        self.assertEqual(host_keys.fingerprints([WEB_LINE], run=broken), [])

    def test_lines_that_are_not_the_host_s_keys_are_refused(self):
        """A line is a key of the host alone, as ssh-keyscan prints one: beside the host's name, no other name, pattern
        or negation, for which ssh would trust the key too, or not trust it for the host. And ssh-keygen reads its
        key: no line is recorded that you were not shown by its fingerprint."""
        run, _ = fake_tools()
        self.assertEqual(host_keys.refusal([WEB_LINE], "web.example.com", run=run), "")
        self.assertEqual(
            host_keys.refusal([GPU_LINE], "login.example.com", 2222, run=run), ""
        )
        # ssh matches a host's name whatever its case.
        upper = WEB_LINE.replace("web.example.com", "Web.Example.COM")
        self.assertEqual(host_keys.refusal([upper], "web.example.com", run=run), "")
        others = (
            "web.example.com,192.0.2.10",
            "*,web.example.com",
            "!web.example.com,web.example.com",
            "*.example.com",
        )
        for lines, name, port, why in (
            ([WEB_LINE], "", 22, "no host is named"),
            ([], "web.example.com", 22, "no key is given"),
            (
                [WEB_LINE, "web.example.com ssh-ed25519"],
                "web.example.com",
                22,
                "line 2 is no host's key (a host, a key type and a key)",
            ),
            (
                ["web.example.com not-a-type AAAA"],
                "web.example.com",
                22,
                "line 1 is no host's key (a host, a key type and a key)",
            ),
            (
                ["web.example.com ssh-ed25519 not;a;key"],
                "web.example.com",
                22,
                "line 1 is no host's key (a host, a key type and a key)",
            ),
            (
                [WEB_LINE.replace("web.example.com", "other.example.com")],
                "web.example.com",
                22,
                "line 1 is a key of other.example.com, not of web.example.com alone",
            ),
            (
                [GPU_LINE],
                "login.example.com",
                22,
                "line 1 is a key of [login.example.com]:2222, not of login.example.com alone",
            ),
            *(
                (
                    [WEB_LINE, WEB_LINE.replace("web.example.com", other)],
                    "web.example.com",
                    22,
                    f"line 2 is a key of {other}, not of web.example.com alone",
                )
                for other in others
            ),
            (
                [WEB_LINE, UNREADABLE],
                "web.example.com",
                22,
                "line 2 has no key ssh-keygen reads",
            ),
        ):
            with self.subTest(why=why):
                self.assertEqual(host_keys.refusal(lines, name, port, run=run), why)

        def missing(argv, **kw):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        self.assertEqual(
            host_keys.refusal([WEB_LINE], "web.example.com", run=missing),
            "ssh-keygen did not run, so no key was read",
        )

    def test_the_state_records_the_keys_and_the_fix_names_the_login_walk(self):
        home = self.d / "home"
        host_keys.record(home, host_keys.WEB, [WEB_LINE])
        host_keys.record(home, host_keys.GPU, [GPU_LINE])
        got = state_store.read(home)
        self.assertEqual(got["publish"]["html"]["host_keys"], [WEB_LINE])
        self.assertEqual(got["remote_gpu_tests"]["login_host_keys"], [GPU_LINE])
        cfg = self.cfg(
            web=got["publish"]["html"]["host_keys"],
            gpu=got["remote_gpu_tests"]["login_host_keys"],
        )
        self.assertEqual(host_keys.recorded(cfg, host_keys.WEB), [WEB_LINE])
        self.assertEqual(host_keys.recorded(cfg, host_keys.GPU), [GPU_LINE])
        # Recorded again: in the place of the keys recorded before.
        other = WEB_LINE.replace("ssh-ed25519 AAAAC3", "ssh-ed25519 AAAAC4")
        host_keys.record(home, host_keys.WEB, [other])
        self.assertEqual(
            state_store.read(home)["publish"]["html"]["host_keys"], [other]
        )
        # A state that cannot be read: nothing recorded.
        state_store.path(home).write_text("[]", encoding="utf-8")
        with self.assertRaises(state_store.StateError):
            host_keys.record(home, host_keys.WEB, [WEB_LINE])
        self.assertEqual(state_store.path(home).read_text(encoding="utf-8"), "[]")
        # Lines too short to be a key are no key; none recorded is none.
        self.assertEqual(
            host_keys.recorded(
                self.cfg(web=["web.example.com", WEB_LINE]), host_keys.WEB
            ),
            [WEB_LINE],
        )
        self.assertEqual(host_keys.recorded({}, host_keys.GPU), [])
        # Either host's key is recorded by deploy step 5, rkmon login, which sets the ssh key first when it is not set.
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            self.assertEqual(
                str(host_keys.record_fix()),
                "./rkmon login shows it, and records it once you accept it",
            )
        self.assertEqual(
            str(host_keys.record_fix("rkmon")),
            "rkmon login shows it, and records it once you accept it",
        )

    def test_a_key_recorded_for_another_host_or_port_is_none_of_the_host_s(self):
        """The keys the state records count for the host and port the settings name now alone (refusal): once they
        name another web host, or another port of the GPU login host, the file ssh reads trusts no host, ssh does not
        run (unrecorded), and rk host-key counts no key of another host as one the host's keys replace.
        """
        cfg = self.cfg(web=[WEB_LINE], gpu=[GPU_LINE])
        for h, line in ((host_keys.WEB, WEB_LINE), (host_keys.GPU, GPU_LINE)):
            self.assertEqual(host_keys.recorded(cfg, h), [line])
        moved = dict(
            cfg,
            publish={"html": {"host": "web2.example.com", "host_keys": [WEB_LINE]}},
            gpu={"login_host": "login.example.com", "ssh_port": 22},
        )
        for h in host_keys.HOSTS:
            with self.subTest(host=h.words):
                self.assertEqual(host_keys.recorded(moved, h), [])
                self.assertEqual(host_keys.known_hosts_file(moved, h), os.devnull)
                self.assertTrue(
                    host_keys.unrecorded(moved, h).startswith(
                        f"{h.words}'s own key is not recorded: "
                    )
                )
        # Each line by itself: those of the host named now are kept; with no host named, none is.
        web2 = WEB_LINE.replace("web.example.com", "web2.example.com")
        both = self.cfg(web=[web2, WEB_LINE])
        self.assertEqual(host_keys.recorded(both, host_keys.WEB), [WEB_LINE])
        unnamed = dict(cfg, publish={"html": {"host": "", "host_keys": [WEB_LINE]}})
        self.assertEqual(host_keys.recorded(unnamed, host_keys.WEB), [])
        run, _ = fake_tools(scanned=(web2,))
        got = host_keys.read(moved, host_keys.WEB, run=run)
        self.assertEqual(
            (got["lines"], got["recorded"], got["others"]), ([web2], False, 0)
        )

    def test_ssh_reads_the_recorded_keys_alone_strictly(self):
        cfg = self.cfg(web=[WEB_LINE], gpu=[GPU_LINE])
        for h, line, name in (
            (host_keys.WEB, WEB_LINE, derived.WEB_HOST_KNOWN_HOSTS),
            (host_keys.GPU, GPU_LINE, derived.GPU_KNOWN_HOSTS),
        ):
            with self.subTest(host=h.words):
                known = self.work / name
                self.assertEqual(
                    host_keys.ssh_options(cfg, h),
                    [
                        "-o",
                        "StrictHostKeyChecking=yes",
                        "-o",
                        f"UserKnownHostsFile={known}",
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
                self.assertEqual(known.read_text(encoding="utf-8"), line + "\n")
                self.assertEqual(stat.S_IMODE(known.stat().st_mode), 0o644)
                # Holding the recorded keys already: not written again.
                before = known.stat()
                host_keys.known_hosts_file(cfg, h)
                self.assertEqual(
                    (known.stat().st_ino, known.stat().st_mtime_ns),
                    (before.st_ino, before.st_mtime_ns),
                )
                # Holding anything else: written whole again.
                known.write_text(
                    f"{line}\nother.example.com ssh-ed25519 AAAAOther\n",
                    encoding="utf-8",
                )
                host_keys.known_hosts_file(cfg, h)
                self.assertEqual(known.read_text(encoding="utf-8"), line + "\n")
        # No key recorded, no working folder known, or a file that cannot be written: ssh trusts no host.
        blocked = self.d / "a_file"
        blocked.write_text("", encoding="utf-8")
        for c in (
            self.cfg(),
            {k: v for k, v in cfg.items() if k != "paths"},
            dict(cfg, paths={"working_dir": str(blocked / "work")}),
        ):
            for h in host_keys.HOSTS:
                self.assertEqual(host_keys.known_hosts_file(c, h), os.devnull)
        # ssh splits an option's value at white space: a folder that holds some is quoted.
        spaced = self.d / "a folder"
        c = dict(cfg, paths={"working_dir": str(spaced)})
        self.assertIn(
            f'UserKnownHostsFile="{spaced / derived.WEB_HOST_KNOWN_HOSTS}"',
            host_keys.ssh_options(c, host_keys.WEB),
        )

    def test_read_says_what_the_host_shows_and_what_the_state_records(self):
        run, calls = fake_tools()
        want = {
            "host": "web.example.com",
            "port": 22,
            "lines": [WEB_LINE],
            "fingerprints": ["ED25519 SHA256:example1"],
            "recorded": False,
            "others": 0,
            "why": "",
        }
        self.assertEqual(host_keys.read(self.cfg(), host_keys.WEB, run=run), want)
        self.assertEqual(
            host_keys.read(self.cfg(web=[WEB_LINE]), host_keys.WEB, run=run),
            dict(want, recorded=True),
        )
        older = WEB_LINE.replace("AAAAC3", "AAAAC4")
        self.assertEqual(
            host_keys.read(self.cfg(web=[older]), host_keys.WEB, run=run),
            dict(want, others=1),
        )
        # A line ssh-keygen reads no key in is left out: the keys shown are those recorded.
        run, calls = fake_tools(scanned=(UNREADABLE, WEB_LINE))
        self.assertEqual(host_keys.read(self.cfg(), host_keys.WEB, run=run), want)
        run, calls = fake_tools(scanned=(UNREADABLE,))
        self.assertEqual(
            host_keys.read(self.cfg(), host_keys.WEB, run=run),
            dict(
                want,
                lines=[],
                fingerprints=[],
                why="ssh-keygen read none of the keys ssh-keyscan printed",
            ),
        )
        run, calls = fake_tools(scanned=(GPU_LINE,))
        got = host_keys.read(self.cfg(), host_keys.GPU, run=run)
        self.assertEqual(
            (got["host"], got["port"], got["lines"]),
            ("login.example.com", 2222, [GPU_LINE]),
        )
        self.assertIn("-p", calls[0][0])
        # Nothing read: why, and no fingerprint asked for.
        run, calls = fake_tools(scanned=(), code=1, err="")
        got = host_keys.read(self.cfg(), host_keys.WEB, run=run)
        self.assertEqual(
            (got["lines"], got["fingerprints"], got["why"]),
            ([], [], "no key came (ssh-keyscan exited 1)"),
        )
        self.assertEqual([c[0][0] for c in calls], ["ssh-keyscan"])
        # No host named: nothing is read.
        run, calls = fake_tools()
        got = host_keys.read(
            {"publish": {"html": {"host": ""}}}, host_keys.WEB, run=run
        )
        self.assertEqual(
            got["why"],
            f"your settings name no host for it ({deploy_tree.name('publish.html.host')})",
        )
        self.assertEqual(calls, [])


# A configuration that runs, as rk jira-account's tests have it, naming both hosts: the GPU login host on port 2222.
CONFIG = {
    "repo": "acme/widgets",
    "rocke_path": "a/b/rocke",
    "operator": {"github_login": "<user_login>"},
    "scope": {"labels": ["team: widgets"]},
    "publish": {"html": {"host": "web.example.com"}},
    "gpu": {"login_host": "login.example.com", "ssh_port": 2222},
}


class RkHostKeyTest(unittest.TestCase):
    """`rk host-key NAME [--json | --record]` (rkmon.py cmd_host_key), which `rkmon secret NAME FILE` runs in the
    walk's one-off container once the ssh key is set: the host's own keys as read there, by their fingerprints, and
    whether the state records them, exit 4 when none is read; --record records the lines on stdin once each one is a
    key of the host the settings name, which ssh-keygen reads, and refuses any other, recording nothing.
    """

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

    def rk(self, *args: str, stdin: str = "", scanned=(WEB_LINE,), code=0, err=""):
        """rk host-key ARGS, STDIN its input, with the fake ssh tools (fake_tools: SCANNED, CODE, ERR), each call in
        self.calls: (exit code, output)."""
        run, self.calls = fake_tools(scanned, code, err)
        read, refusal = host_keys.read, host_keys.refusal
        out = io.StringIO()
        with mock.patch.object(
            host_keys, "read", lambda cfg, h: read(cfg, h, run=run)
        ), mock.patch.object(
            host_keys,
            "refusal",
            lambda lines, name, port: refusal(lines, name, port, run=run),
        ), mock.patch.object(
            sys, "stdin", io.StringIO(stdin)
        ), contextlib.redirect_stdout(
            out
        ):
            rc = rkmon.main(["host-key", *args])
        return rc, out.getvalue()

    def test_it_shows_the_host_s_own_keys_as_read_here(self):
        self.assertEqual(
            self.rk("web_host_ssh_key"),
            (
                0,
                "the web host's own keys (web.example.com), as ssh-keyscan reads them here: the state does not "
                "record them\n  ED25519 SHA256:example1\n",
            ),
        )
        code, out = self.rk("web_host_ssh_key", "--json")
        self.assertEqual(code, 0, out)
        self.assertEqual(
            json.loads(out),
            {
                "host": "web.example.com",
                "port": 22,
                "lines": [WEB_LINE],
                "fingerprints": ["ED25519 SHA256:example1"],
                "recorded": False,
                "others": 0,
                "why": "",
            },
        )
        # The GPU login host, on its port.
        code, out = self.rk("gpu_ssh_key", "--json", scanned=(GPU_LINE,))
        got = json.loads(out)
        self.assertEqual(
            (code, got["host"], got["port"], got["lines"]),
            (0, "login.example.com", 2222, [GPU_LINE]),
        )
        self.assertIn("2222", self.calls[0][0])
        # None read: why, exit 4.
        refused = "connect to host web.example.com port 22: Connection refused"
        self.assertEqual(
            self.rk("web_host_ssh_key", scanned=(), code=1, err=refused + "\n"),
            (4, f"no key of the web host was read: {refused}\n"),
        )
        code, out = self.rk("web_host_ssh_key", "--json", scanned=(), code=1, err="")
        self.assertEqual(
            (code, json.loads(out)["why"]), (4, "no key came (ssh-keyscan exited 1)")
        )

    def test_record_records_the_lines_on_stdin_as_the_host_s_own_keys(self):
        self.assertEqual(
            self.rk("web_host_ssh_key", "--record", stdin=WEB_LINE + "\n"),
            (
                0,
                "the web host's own keys recorded (1 key, team_dashboard.web_host_keys): ssh trusts web.example.com "
                "by them alone\n",
            ),
        )
        # Nothing is read from the host: the lines are those you accepted; ssh-keygen reads the key in each.
        self.assertEqual(self.calls, [(["ssh-keygen", "-lf", "-"], WEB_LINE + "\n")])
        self.assertEqual(
            state_store.read(self.home)["publish"]["html"]["host_keys"], [WEB_LINE]
        )
        code, out = self.rk("web_host_ssh_key")
        self.assertEqual(
            out.splitlines()[0],
            "the web host's own keys (web.example.com), as ssh-keyscan reads them here: the state records them",
        )
        # The host's keys changed: what is read says how many others the state records.
        newer = WEB_LINE.replace("AAAAC3", "AAAAC4")
        code, out = self.rk("web_host_ssh_key", scanned=(newer,))
        self.assertIn(
            "the state does not record them; it records 1 other key of it\n", out
        )
        code, out = self.rk("gpu_ssh_key", "--record", stdin=f"\n{GPU_LINE}\n\n")
        self.assertEqual(code, 0, out)
        self.assertIn("(1 key, remote_gpu_tests.login_host_keys)", out)
        self.assertEqual(
            state_store.read(self.home)["remote_gpu_tests"]["login_host_keys"],
            [GPU_LINE],
        )

    def test_lines_that_are_not_the_host_s_keys_are_refused_and_nothing_is_recorded(
        self,
    ):
        other = WEB_LINE.replace("web.example.com", "other.example.com")
        both = WEB_LINE.replace("web.example.com", "web.example.com,other.example.com")
        for name, stdin, why in (
            (
                "web_host_ssh_key",
                other,
                "line 1 is a key of other.example.com, not of web.example.com alone",
            ),
            (
                "web_host_ssh_key",
                both,
                "line 1 is a key of web.example.com,other.example.com, not of web.example.com alone",
            ),
            (
                "gpu_ssh_key",
                GPU_LINE.replace("[login.example.com]:2222", "login.example.com"),
                "line 1 is a key of login.example.com, not of [login.example.com]:2222 alone",
            ),
            (
                "web_host_ssh_key",
                f"{WEB_LINE}\n{UNREADABLE}",
                "line 2 has no key ssh-keygen reads",
            ),
            (
                "web_host_ssh_key",
                "web.example.com not-a-key-type AAAAExample",
                "line 1 is no host's key (a host, a key type and a key)",
            ),
            ("web_host_ssh_key", "", "no key is given"),
        ):
            with self.subTest(name=name, why=why):
                words = host_keys.host(name).words
                self.assertEqual(
                    self.rk(name, "--record", stdin=stdin),
                    (4, f"REFUSED: {words}'s own keys are not recorded: {why}\n"),
                )
        self.assertFalse(state_store.path(self.home).exists())

    def test_a_state_that_refuses_them_is_refused_with_its_fix(self):
        fix = checks.Command(("chmod", "660", "internal_state.json"))
        with mock.patch.object(
            host_keys,
            "record",
            side_effect=state_store.StateError(
                "the internal state cannot be written", fix
            ),
        ):
            code, out = self.rk("web_host_ssh_key", "--record", stdin=WEB_LINE)
        self.assertEqual(code, 4, out)
        self.assertEqual(
            out.splitlines(),
            [
                "REFUSED: the internal state cannot be written",
                f"{checks.FIX_LABEL}chmod 660 internal_state.json",
            ],
        )

    def test_it_takes_the_name_of_a_host_s_ssh_key_and_one_of_its_options(self):
        for args in (
            ["teams_webhook"],
            ["web_host_ssh_key", "--json", "--record"],
            [],
        ):
            with self.subTest(args=args):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ) as e:
                    rkmon.main(["host-key", *args])
                self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
