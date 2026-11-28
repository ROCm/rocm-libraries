# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The writer (rkmon_lib/writer.py, docker/rk_write.py): in the container, the user that alone
reads the secrets the outward writes need, kept in its own volume, and runs the listed rk commands that use them.

- Its secrets: each entry of the secrets registry whose keeper is the writer, in the writer's volume in the container
  (whatever a setting names) and in the one secrets folder on a host.
- Its commands (`refused`, `routes`): the probe of its secrets in its two forms and `rk secret-set` for a secret it
  keeps, besides the GitHub writes and their login's probe; a found issue's ticket, a team chat message and its test
  card, the team dashboard's page and requests file; the atlassian CLI's listed commands, each as listed
  (`atlassian_call`); nothing else, and no `rk secret-set` for the monitor's own.
- The monitor sends them through sudo (`routed`, `through`: a writer that could not be started is exit 126 and why,
  one with no answer in time exit 124, as it may still be running; the card waits as long as a filing may take) and keeps
  the writer's answer for KEPT_S seconds (`kept`, `forget`). A secret the writer keeps is set from the host only, as
  the writer itself: the monitor's rk refuses one and sends nothing, and the writer refuses one sudo brings.
- `rk writer-secrets`: whether each is set, where, and when it last changed (its file's time and inode); never a
  value, nor its size.
- The writer's halves, where the secret is read: `rk teams-send` (a message as teams_message makes it), `rk web-push`
  and `rk web-requests`, and `rk atlassian` (the CLI on PATH: tests/fakes/fake_atlassian). The configuration, which
  the session may edit, steers none of them: the web host's login, host and paths are refused unless plain, and no
  proxy it names reaches the CLI.
- As the monitor in the container, each of those sends goes through the writer (`the_writer`: sudo, then
  docker/rk_write.py's checks, then the writer's rk, all in this process): a found issue's ticket filed, a team chat
  message, the page pushed and the requests read, and each atlassian CLI call.
- The outward gate: before each send it makes for another process (rkmon_lib/outward.py: gate GATED), the
  writer's rk asks it, whoever asked for the send: a mirror, the send's feature off, or a stop, refuses it, and
  nothing leaves; a read passes. A GitHub write meets its command's own gate in the writer: in prototype, nothing goes
  out. A team chat message meets the mirror's refusal once in each process, in one wording.
- The Claude session, as the monitor's user with its own environment (orchestrator.session_env), each command past its
  guard (orchestrator.guard): it has no atlassian login; a follow-up ticket is filed and read through the writer, as
  the skill's steps say; no secret of the writer's reaches it; an atlassian command off the list is refused; a write
  while its feature is off, or in prototype, is refused. The text it and its agents follow (agentic/) gives each JIRA
  call as `<RK> atlassian`, with a command and options the writer lists.
- docker/rk_write.py: a command off the list is refused before anything runs; no file argument names a secret of the
  writer's; `rk secret-set` takes none; through sudo, no secret of the writer's is set, and no command makes a login.
  Every refusal of the writer's commands is the one line writer.refusal makes, wherever it is refused. In a one-off
  container of the logins' walk the writer's rk, through sudo or run by the writer's own, reads the confirmed
  settings the monitor's rk reads.
"""

import contextlib
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rk_write  # noqa: E402
from rkmon_lib import writer_run  # noqa: E402
import rkmon  # noqa: E402
from rkmon_lib import (  # noqa: E402
    adf,
    atlassian,
    config,
    container_layout,
    control,
    features,
    gh,
    logins,
    mirror,
    needs_teams,
    orchestrator,
    outward,
    post,
    publish,
    reach,
    report,
    state_store,
    summary,
    tickets,
    writer,
)
from rkmon_lib import secrets_registry as sr  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402
from test_secrets_registry import a_value  # noqa: E402  (an invented value of a shape)

BROKER = "/b/rk_write.py"  # the image's writer entry point, as its environment names it (writer.BROKER_VAR)
FAKES = ROOT / "tests" / "fakes"
# An invented project key, built at run time (the privacy check looks for <KEY>-<n>).
PROJ = "TK" + "T"
ELSEWHERE = "EL" + "S"  # another project the login may reach
HOOK = "https://hooks.example.invalid/workflows/Hook-Path-Q"  # an invented webhook
# Ticket creation on and live (it files), with both types of ticket: a trigger the configuration leaves out reads as
# off, as each type of ticket in the deploy settings file is off until set.
TICKETS_ON = {
    "enabled": True,
    "mode": "live",
    "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
}
# What the writer's one refusal line (writer.refusal) puts before and after its reason.
REFUSAL_HEAD, REFUSAL_TAIL = writer.refusal("WHY").split("WHY")
DRAFT = """---
title: The nightly lane skips a case on every runner
type: Bug
priority: P3
summary: one case reports SKIPPED on every runner
---
What happens: it reports SKIPPED on develop.
"""


class Exec(Exception):
    """rkmon_lib/writer_run.py's execve in the tests: (ARGV, ENV), the rk that would run in its place."""


def exec_(exe, argv, env):
    raise Exec(argv, env)


class Answer:
    """urllib's answer to a POST, as a context manager: its HTTP status."""

    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Base(unittest.TestCase):
    """A configuration of invented folders, in the container; the writer's volume a temporary folder; no answer of the
    writer kept from one test to the next."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.write_secrets = self.d / "secrets-write"
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        c["paths"] = dict(
            c["paths"],
            working_dir=str(self.d / "work"),
            secrets_dir=str(self.d / "secrets"),
        )
        self.cfg = config.normalize(c)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (writer.BROKER_VAR, gh.BROKERED)
        }
        env["RKMON_CONTAINER"] = "1"
        for p in (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                container_layout, "WRITE_SECRETS_DIR", str(self.write_secrets)
            ),
        ):
            p.start()
            self.addCleanup(p.stop)
        writer.forget()
        self.addCleanup(writer.forget)
        self.store = Store(self.d / "state")
        self.store.save(empty_state())
        self.sent = []  # each command the monitor sent the writer (the_writer)
        self.marked = (
            []
        )  # whether each carried the mark of the monitor's own writes (outward.OWN_WORD)

    def kept_by(self, keeper: str, login: bool = False) -> list[str]:
        """The registry's secrets (LOGIN: logins) that KEEPER keeps."""
        return [
            e.name
            for e in sr.entries(self.cfg)
            if e.keeper == keeper and (e.shape == sr.LOGIN) == login
        ]

    def as_the_monitor(self):
        """This process as the monitor in the container: the image names the writer's entry point."""
        return mock.patch.dict(os.environ, {writer.BROKER_VAR: BROKER})

    def as_the_writer(self):
        """This process as the writer's rk: docker/rk_write.py's environment says so (gh.BROKERED)."""
        return mock.patch.dict(os.environ, {gh.BROKERED: "1"})

    def rk(self, argv: list[str], stdin: str = "") -> tuple[int, str]:
        """rk ARGV in this process, with this configuration and STDIN: its exit code and what it printed."""
        out = io.StringIO()
        with mock.patch.object(
            sys, "stdin", io.StringIO(stdin)
        ), contextlib.redirect_stdout(out):
            code = self.rk_main(argv)
        return code, out.getvalue()

    def rk_main(self, argv: list[str]) -> int:
        """rkmon.main(ARGV) with this configuration and state."""
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store):
            try:
                return rkmon.main(argv)
            except SystemExit as exc:  # argparse's refusal
                return exc.code if isinstance(exc.code, int) else 1

    def entry_point(self):
        """docker/rk_write.py as the writer: its user check passed by a stand-in for the password database, its log and
        its copies in the temporary folder, the code it runs this tree's, whose check of the whole tree passes (the
        tests' files are not root's)."""
        me = mock.Mock(pw_name=rk_write.WRITE_USER, pw_dir=str(self.d / "writer-home"))
        stack = contextlib.ExitStack()
        for p in (
            mock.patch.object(rk_write, "LOG", str(self.d / "rk-write.log")),
            mock.patch.object(rk_write.pwd, "getpwuid", lambda uid: me),
            mock.patch.object(rk_write, "picked", lambda: str(ROOT)),
            mock.patch.object(rk_write, "problems", lambda *a, **kw: []),
            mock.patch.object(rk_write, "links", lambda *a, **kw: []),
            mock.patch.object(writer_run, "LOG", str(self.d / "rk-write.log")),
            mock.patch.object(
                writer_run, "config_folders_outside", lambda protected: ""
            ),
            mock.patch.object(rk_write.os, "umask", lambda m: 0),
        ):
            stack.enter_context(p)
        return stack

    @staticmethod
    def sudo_caller() -> dict:
        """What sudo puts in the environment of the command it runs for the monitor's user: who called it."""
        return {
            writer.SUDO_VAR: container_layout.MONITOR.name,
            "SUDO_UID": str(os.getuid()),
            "SUDO_GID": str(os.getgid()),
        }

    def the_writer(self):
        """sudo as it meets the monitor's calls to the writer (writer.routed): docker/rk_write.py's checks, then the
        writer's rk in this process, in the writer's environment (gh.BROKERED: it runs each command itself, and
        outward.OWN_VAR as writer_run set it), with the caller's input on its stdin; its output captured, or the
        caller's. Every other command runs as it would. Each command sent is a list in self.sent, without the word that
        marks a write of the monitor's own; whether it carried that word is in self.marked.
        """
        real = subprocess.run
        sudo = ["sudo", "-n", "-u", container_layout.WRITER.name, BROKER]
        caller = self.sudo_caller()

        def run(argv, *a, input=None, capture_output=False, stdin=None, **kw):
            if list(argv[:5]) != sudo:
                return real(
                    argv,
                    *a,
                    input=input,
                    capture_output=capture_output,
                    stdin=stdin,
                    **kw,
                )
            sent = list(argv[5:])
            own = sent[:1] == [outward.OWN_WORD]
            self.sent.append(sent[1:] if own else sent)
            self.marked.append(own)
            out, err = io.StringIO(), io.StringIO()
            users, umask = rk_write.pwd.getpwuid, rk_write.os.umask

            def writers_rk(args, env, cwd):
                """The broker's child (python3 -I -B -c BOOT TREE ARGV): rkmon_lib/writer_run.py's checks in the
                broker's copies folder, then in its place the writer's rk in this process, with the password database
                and the umask its own process has, while the broker's copies are there.
                """
                try:
                    with contextlib.chdir(cwd):
                        return writer_run.main(
                            list(args[6:]), environ=env, execve=exec_
                        )
                except Exec as done:
                    args, env = done.args
                given = (
                    io.StringIO(input or "")
                    if input is not None or stdin == subprocess.DEVNULL
                    else sys.stdin
                )
                mark = {k: env[k] for k in (gh.BROKERED, outward.OWN_VAR) if k in env}
                with mock.patch.object(
                    rk_write.pwd, "getpwuid", users
                ), mock.patch.object(rk_write.os, "umask", umask), mock.patch.dict(
                    os.environ, mark
                ), mock.patch.object(
                    sys, "stdin", given
                ), contextlib.redirect_stderr(
                    err
                ):
                    if (
                        outward.OWN_VAR not in env
                    ):  # the writer's rk has the caller's mark alone
                        os.environ.pop(outward.OWN_VAR, None)
                    return self.rk_main(args[4:])

            with self.entry_point(), contextlib.redirect_stdout(out):
                code = rk_write.main(list(argv[5:]), run=writers_rk, environ=caller)
            if capture_output:
                return subprocess.CompletedProcess(
                    argv, code, out.getvalue(), err.getvalue()
                )
            sys.stdout.write(out.getvalue())
            sys.stderr.write(err.getvalue())
            return subprocess.CompletedProcess(argv, code)

        return mock.patch.object(subprocess, "run", run)

    def cli_here(self, **env: str) -> pathlib.Path:
        """The atlassian CLI on PATH (tests/fakes/fake_atlassian, with ENV), its JIRA's state in the temporary folder;
        the file of its calls, one JSON list a line."""
        bindir = self.d / "bin"
        bindir.mkdir(exist_ok=True)
        shutil.copy(FAKES / "fake_atlassian", bindir / "atlassian")
        (bindir / "atlassian").chmod(0o755)
        log = self.d / "atlassian.log"
        p = mock.patch.dict(
            os.environ,
            {
                "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
                "FAKE_ATLASSIAN_LOG": str(log),
                "FAKE_ATLASSIAN_STATE": str(self.d / "jira"),
                **env,
            },
        )
        p.start()
        self.addCleanup(p.stop)
        return log

    @staticmethod
    def calls(log: pathlib.Path) -> list[list[str]]:
        """The CLI's calls, as its log has them."""
        return (
            [json.loads(x) for x in log.read_text().splitlines()]
            if log.exists()
            else []
        )


class TestItsSecrets(Base):
    def test_the_writer_keeps_the_outward_writes_secrets(self):
        """The writer keeps the Teams webhook and the web host's key, and the GitHub write login and
        the atlassian login; the monitor keeps the rest, the Claude session's key (the claude login) among them.
        """
        self.assertEqual(
            sorted(self.kept_by(sr.WRITER)), ["teams_webhook", "web_host_ssh_key"]
        )
        self.assertEqual(
            sorted(self.kept_by(sr.WRITER, login=True)), ["atlassian", "github_write"]
        )
        self.assertIn("claude", self.kept_by(sr.MONITOR, login=True))

    def test_in_the_container_they_are_in_the_writer_s_volume_whatever_a_setting_names(
        self,
    ):
        cfg = dict(self.cfg, teams={"webhook_file": str(self.d / "elsewhere")})
        for name in self.kept_by(sr.WRITER) + self.kept_by(sr.WRITER, login=True):
            with self.subTest(name=name):
                entry = sr.entry(cfg, name)
                self.assertEqual(
                    sr.path(cfg, entry), str(self.write_secrets / (entry.place or name))
                )
        self.assertEqual(
            sr.path(cfg, sr.entry(cfg, "claude")),
            str(self.d / "secrets" / "claude_key"),
        )  # the monitor's own
        with mock.patch.dict(
            os.environ, {"RKMON_CONTAINER": ""}
        ):  # a host: one user, one folder
            self.assertEqual(
                sr.path(self.cfg, sr.entry(self.cfg, "teams_webhook")),
                str(self.d / "secrets" / "teams_webhook"),
            )


class TestItsCommands(Base):
    def test_it_runs_the_listed_commands_only(self):
        for argv in (
            [writer.WRITER_SECRETS],
            writer.SECRETS_PROBE,
            writer.WRITE_PROBE,
            ["post", "7", "f.md"],
            [writer.TICKET_FILE, "t1"],
            [writer.TEAMS_SEND, "--type", post.KIND_NEEDS_YOU],
            [writer.TEAMS_TEST],
            [writer.WEB_PUSH, "page.html"],
            [writer.WEB_REQUESTS],
            [
                writer.ATLASSIAN,
                "--site",
                "example",
                "jira",
                "issue",
                "get",
                f"{PROJ}-3",
            ],
        ):
            self.assertEqual(writer.refused(argv), "", argv)
            self.assertTrue(writer.routes(argv), argv)
        for name in self.kept_by(sr.WRITER):
            for argv in (
                [writer.SECRET_SET, name],
                [writer.SECRET_SET, name, "--remove"],
            ):
                self.assertEqual(writer.refused(argv), "", argv)
                self.assertFalse(writer.routes(argv), argv)  # from the host only
                self.assertEqual(writer.sudo_refusal(argv, {}), "", argv)
                self.assertEqual(
                    writer.sudo_refusal(argv, {writer.SUDO_VAR: "rkmon"}),
                    f"{name} is set from the host only: rkmon secret {name}",
                    argv,
                )
        for name in self.kept_by(sr.MONITOR) + self.kept_by(sr.WRITER, login=True):
            argv = [writer.SECRET_SET, name]
            self.assertEqual(
                writer.refused(argv), f"{name} is not a secret the writer keeps", argv
            )
            self.assertFalse(writer.routes(argv), argv)
        # The request service's install, over the web host's key, which the writer keeps: from the host only.
        argv = [writer.WEB_HOST_INSTALL]
        self.assertEqual(writer.refused(argv), "")
        self.assertFalse(writer.routes(argv))
        self.assertEqual(writer.sudo_refusal(argv, {}), "")
        self.assertEqual(
            writer.sudo_refusal(argv, {writer.SUDO_VAR: "rkmon"}),
            "the request service is installed on the web host from the host only: rkmon web-host install",
        )
        for argv, why in (
            (
                [writer.WRITER_SECRETS, "--all"],
                "it runs `rk writer-secrets [--json]` only",
            ),
            ([writer.SECRET_SET], "it runs `rk secret-set NAME [--remove]` only"),
            (
                [writer.SECRET_SET, "teams_webhook", "--force"],
                "it runs `rk secret-set NAME [--remove]` only",
            ),
            (
                [writer.SECRET_SET, "teams_webhook", "web_host_ssh_key"],
                "it runs `rk secret-set NAME [--remove]` only",
            ),
            (["gh-logins"], "it runs `rk gh-logins --side write --json` only"),
            (
                # the web host, its login and its paths come from the settings the writer runs on, never its caller
                [writer.WEB_HOST_INSTALL, "web2.example.com"],
                "it runs `rk web-host-install` only: it reads the web host, its login and its paths from the settings "
                "it runs on",
            ),
            (["status"], "rk status is not one of the writer's commands"),
            (["config", "--show"], "rk config is not one of the writer's commands"),
            ([], "no command"),
        ):
            self.assertEqual(writer.refused(argv), why, argv)
        self.assertNotIn(
            writer.SECRET_SET, writer.FILE_ARGS
        )  # its value comes on stdin
        self.assertNotIn(writer.WRITER_SECRETS, writer.FILE_ARGS)
        self.assertIn(writer.WEB_PUSH, writer.FILE_ARGS)  # the page: a checked copy
        for cmd in (
            writer.TEAMS_SEND,
            writer.ATLASSIAN,
        ):  # a message on stdin; the CLI's words are never files
            self.assertNotIn(cmd, writer.FILE_ARGS)

    def test_the_monitor_sends_them_through_sudo(self):
        sudo = ["sudo", "-n", "-u", container_layout.WRITER.name, BROKER]
        env = {writer.BROKER_VAR: BROKER}
        self.assertEqual(
            writer.routed([writer.TEAMS_TEST], env), sudo + [writer.TEAMS_TEST]
        )
        self.assertEqual(
            writer.routed(writer.SECRETS_PROBE, env), sudo + writer.SECRETS_PROBE
        )
        for name in ("claude_env", "teams_webhook"):
            # the monitor's own is set here; the writer's from the host only
            self.assertIsNone(writer.routed([writer.SECRET_SET, name], env), name)
        self.assertIsNone(writer.routed([writer.TEAMS_TEST], {}))  # a host
        self.assertIsNone(
            writer.routed([writer.TEAMS_TEST], dict(env, **{gh.BROKERED: "1"}))
        )  # the writer's own rk runs it
        seen = []
        with self.as_the_monitor(), mock.patch.object(
            rkmon.subprocess,
            "run",
            lambda argv, **kw: seen.append((argv, kw))
            or subprocess.CompletedProcess(argv, 0),
        ):
            self.assertEqual(rkmon.main([writer.TEAMS_TEST]), 0)
        self.assertEqual(
            seen, [(sudo + [writer.TEAMS_TEST], {})]
        )  # stdin: the caller's

    def test_the_monitor_s_rk_refuses_a_secret_the_writer_keeps_and_sends_nothing(self):
        """`rk secret-set NAME` as the monitor, the session's user too: refused, with the host's command, whatever it
        is given; nothing reaches the writer, and nothing is written."""
        sent = []
        with self.as_the_monitor(), mock.patch.object(
            subprocess, "run", lambda argv, **kw: sent.append(argv)
        ):
            for name in self.kept_by(sr.WRITER):
                value = a_value(sr.entry(self.cfg, name).shape, 1)
                for argv in (
                    [writer.SECRET_SET, name],
                    [writer.SECRET_SET, name, "--remove"],
                ):
                    with self.subTest(argv=argv):
                        self.assertEqual(
                            self.rk(argv, stdin=value),
                            (
                                4,
                                f"REFUSED: {name} is set from the host only: rkmon secret {name}; nothing "
                                "changed\n",
                            ),
                        )
        self.assertEqual(sent, [])
        self.assertFalse(self.write_secrets.exists())

    def test_the_monitor_s_rk_refuses_the_web_host_install_and_sends_nothing(self):
        """`rk web-host-install` as the monitor, the session's user too, which does not read the web host's key:
        refused, with the host's command; nothing reaches the writer or the web host."""
        sent = []
        with self.as_the_monitor(), mock.patch.object(
            subprocess, "run", lambda argv, **kw: sent.append(argv)
        ):
            self.assertEqual(
                self.rk([writer.WEB_HOST_INSTALL]),
                (
                    4,
                    "REFUSED: the request service is installed on the web host from the host only: rkmon web-host "
                    "install; nothing changed\n",
                ),
            )
        self.assertEqual(sent, [])

    def test_through_gives_the_writer_s_answer_or_why_it_did_not_run(self):
        calls = []

        def run(argv, **kw):
            calls.append((argv, kw))
            return subprocess.CompletedProcess(argv, 0, "sent\n", "")

        with self.as_the_monitor(), mock.patch.object(writer.subprocess, "run", run):
            p = writer.through(
                [writer.TEAMS_SEND, "--type", post.KIND_NEEDS_YOU],
                input="v\n",
                timeout_s=5,
            )
            self.assertEqual((p.returncode, p.stdout), (0, "sent\n"))
            writer.through(writer.SECRETS_PROBE)
        (argv, kw), (_, kw2) = calls
        self.assertEqual(
            argv[:5], ["sudo", "-n", "-u", container_layout.WRITER.name, BROKER]
        )
        self.assertEqual(
            (kw["input"], kw["stdin"], kw["timeout"], kw["capture_output"]),
            ("v\n", None, 5, True),
        )
        self.assertEqual(
            kw2["stdin"], subprocess.DEVNULL
        )  # nothing of the caller's stdin
        with self.as_the_monitor(), mock.patch.object(
            writer.subprocess,
            "run",
            side_effect=FileNotFoundError(2, "No such file", "sudo"),
        ):
            p = writer.through(writer.SECRETS_PROBE)
        self.assertEqual(p.returncode, 126)
        self.assertTrue(
            p.stderr.startswith("the writer did not run (FileNotFoundError"), p.stderr
        )
        with self.assertRaises(
            ValueError
        ):  # not the monitor in the container: nothing to send
            writer.through(writer.SECRETS_PROBE)

    def test_through_says_a_writer_with_no_answer_in_time_may_still_be_running(self):
        """sudo's end at the time limit does not end the writer's rk: the command may have run, or still be running,
        so a timeout is never "did not run"."""
        no_answer = "no answer in 7.5 s; it may still be running"
        with self.as_the_monitor(), mock.patch.object(
            writer.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["sudo"], 7.5),
        ):
            p = writer.through([writer.TICKET_FILE, "0a1b2c3d"], timeout_s=7.5)
        self.assertEqual((p.returncode, p.stdout, p.stderr), (124, "", no_answer))
        self.assertEqual(writer.said(p), no_answer)

    def test_rk_refuses_in_the_writer_s_words_when_sudo_cannot_start_it(self):
        """As the monitor: a command for the writer that sudo cannot start is refused in the writer's one line
        (writer.refusal), exit 4."""
        with self.as_the_monitor(), mock.patch.object(
            rkmon.subprocess,
            "run",
            side_effect=FileNotFoundError(2, "No such file", "sudo"),
        ):
            code, out = self.rk(writer.SECRETS_PROBE)
        why = "it did not start (FileNotFoundError: [Errno 2] No such file: 'sudo')"
        self.assertEqual((code, out), (4, writer.refusal(why) + "\n"))

    def test_the_monitor_keeps_the_writer_s_answer_a_while(self):
        answers = [
            json.dumps({"teams_webhook": {"set": True, "where": "w", "mark": "1:2"}}),
            "not JSON",
        ]
        asked = []

        def through(argv, input=None, timeout_s=None):
            asked.append((argv, timeout_s))
            return subprocess.CompletedProcess(argv, 0, answers[len(asked) - 1], "")

        now = [1000.0]
        with mock.patch.object(writer, "through", through), mock.patch.object(
            writer.time, "monotonic", lambda: now[0]
        ):
            self.assertTrue(writer.kept()["teams_webhook"]["set"])
            now[0] += writer.KEPT_S - 1
            self.assertTrue(
                writer.kept()["teams_webhook"]["set"]
            )  # kept: not asked again
            self.assertEqual(asked, [(writer.SECRETS_PROBE, writer.TIMEOUT_S)])
            now[0] += 2
            self.assertEqual(
                writer.kept(), {}
            )  # asked again; an answer that is not JSON is none
            self.assertEqual(len(asked), 2)
            writer.forget()
            with mock.patch.object(
                writer,
                "through",
                lambda argv, **kw: subprocess.CompletedProcess(
                    argv, 4, "{}", "REFUSED"
                ),
            ):
                self.assertEqual(writer.kept(), {})  # a refusal is no answer


class TestWriterSecretsCommand(Base):
    def test_whether_each_is_set_and_where_never_a_value(self):
        value = a_value(sr.HTTPS_URL, 7)
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(value + "\n")
        code, out = self.rk([writer.WRITER_SECRETS])
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertIn(
            f"teams_webhook: set ({self.write_secrets / 'teams_webhook'})", lines
        )
        self.assertIn(
            f"web_host_ssh_key: not set ({self.write_secrets / 'web_host_ssh_key'})",
            lines,
        )
        self.assertIn(f"atlassian: not set ({self.write_secrets / 'atlassian'})", lines)
        code, out = self.rk(writer.SECRETS_PROBE)
        got = json.loads(out)
        self.assertEqual(
            sorted(got),
            sorted(self.kept_by(sr.WRITER) + self.kept_by(sr.WRITER, login=True)),
        )
        self.assertEqual(
            (got["teams_webhook"]["set"], got["web_host_ssh_key"]["set"]), (True, False)
        )
        s = (self.write_secrets / "teams_webhook").stat()
        # When it last changed: its file's time and inode, never its size (a secret's length is a part of it).
        self.assertEqual(got["teams_webhook"]["mark"], f"{s.st_mtime_ns}:{s.st_ino}")
        self.assertNotIn(value.rsplit("/", 1)[1], out)
        self.assertEqual(got, logins.writer_secrets(self.cfg))

    def test_the_monitor_asks_the_writer(self):
        seen = []
        with self.as_the_monitor(), mock.patch.object(
            rkmon.subprocess,
            "run",
            lambda argv, **kw: seen.append(argv)
            or subprocess.CompletedProcess(argv, 0),
        ):
            self.assertEqual(rkmon.main(writer.SECRETS_PROBE), 0)
        self.assertEqual(seen[0][4:], [BROKER, *writer.SECRETS_PROBE])


class TestTheMonitorAsksTheWriter(Base):
    """As the monitor in the container, what needs to know whether one of the writer's secrets is set takes the
    writer's answer (logins.found) and never reads the writer's volume: the doctor's team chat row, a feature's needs
    and how it is configured, the setup summary, and the webhook's state. No answer is "not known", never "missing".
    """

    def setUp(self):
        super().setUp()
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(
            "https://hooks.example.invalid/Hook-Path-Q\n"
        )  # there, but the monitor never reads it
        self.cfg = dict(
            self.cfg,
            teams={"channel": "ops"},
            features={
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                },
                "team_dashboard": {"enabled": True},
            },
            publish={"html": {"key": str(self.d / "named_key")}},
        )
        p = self.as_the_monitor()
        p.start()
        self.addCleanup(p.stop)

    def the_writer_says(self, kept: dict | None):
        """The writer's answer to its secrets probe (None: it did not run)."""
        writer.forget()
        return mock.patch.object(
            writer,
            "through",
            lambda argv, **kw: subprocess.CompletedProcess(
                argv, 126 if kept is None else 0, json.dumps(kept or {}), ""
            ),
        )

    def test_set_as_the_writer_says(self):
        set_ = {
            n: {"set": True, "where": f"/w/{n}", "mark": "1:2"}
            for n in ("teams_webhook", "web_host_ssh_key")
        }
        reached = []
        with self.the_writer_says(set_):
            (row,) = reach.chat(self.cfg, reach=lambda host: reached.append(host) or "")
            self.assertEqual(row.status, "PASS", row.found)
            self.assertIn("the writer has the webhook and alone reads it", row.found)
            self.assertEqual(
                reached, []
            )  # its host is the writer's to reach (rk teams-test)
            self.assertTrue(post.teams_hook_set(self.cfg))
            self.assertEqual(features.missing(self.cfg, "chat-alerts"), [])
            self.assertIn(
                "webhook set", features.configured_with(self.cfg, "chat-alerts")
            )
            self.assertIn(
                "key set", features.configured_with(self.cfg, "team-dashboard")
            )
            rows = dict(summary.secrets(self.cfg))
            self.assertEqual(rows["team chat webhook"], "set (/w/teams_webhook)")
        self.assertEqual(
            publish.key_file(self.cfg), str(self.write_secrets / "web_host_ssh_key")
        )  # the key the writer's push reads, whatever publish.html.key names

    def test_missing_or_not_known(self):
        missing = {
            "teams_webhook": {"set": False, "where": "/w/teams_webhook", "mark": ""}
        }
        with self.the_writer_says(missing):
            (row,) = reach.chat(self.cfg)
            self.assertEqual(row.status, "WARN")
            self.assertIn("the webhook is missing (/w/teams_webhook)", row.found)
            self.assertFalse(post.teams_hook_set(self.cfg))
            self.assertEqual(
                features.missing(self.cfg, "chat-alerts"),
                [
                    "the channel's incoming-webhook URL (the secret teams_webhook: rkmon secret teams_webhook)"
                ],
            )
        with self.the_writer_says(None):
            (row,) = reach.chat(self.cfg)
            self.assertEqual(
                (row.status, row.found),
                (
                    "WARN",
                    f"whether the webhook is set is not known ({logins.NOT_SAID})",
                ),
            )
            self.assertFalse(post.teams_hook_set(self.cfg))
            self.assertEqual(
                features.missing(self.cfg, "chat-alerts"),
                [
                    f"the channel's incoming-webhook URL (not known whether it is set: {logins.NOT_SAID})"
                ],
            )
            self.assertIn(
                f"webhook not known ({logins.NOT_SAID})",
                features.configured_with(self.cfg, "chat-alerts"),
            )
            rows = dict(summary.secrets(self.cfg))
            self.assertEqual(
                rows["team chat webhook"], f"not known ({logins.NOT_SAID})"
            )


class TestTheAtlassianCommands(Base):
    """`rk atlassian WORDS` (writer.atlassian_call): the atlassian CLI's listed commands, each as listed
    (the global options first, each option as one `--option=value` word), and nothing else.
    """

    def test_the_listed_commands_as_the_writer_runs_them(self):
        key = f"{PROJ}-3"
        meta = f"/rest/api/3/issue/createmeta/{PROJ}/issuetypes"
        jql = f'project = {PROJ} AND summary ~ "lane"'
        for words, argv, writes in (
            (["version"], ["version"], False),
            (["--json", "site", "list"], ["--json", "site", "list"], False),
            (
                ["--site", "example", "--json", "jira", "issue", "get", key],
                ["--site=example", "--json", "jira", "issue", "get", key],
                False,
            ),
            (  # the global options anywhere, as the CLI takes them
                ["jira", "issue", "get", key, "--json", "--site=example"],
                ["--site=example", "--json", "jira", "issue", "get", key],
                False,
            ),
            (
                ["jira", "issue", "search", "--jql", jql, "--max-results", "50"],
                ["jira", "issue", "search", f"--jql={jql}", "--max-results=50"],
                False,
            ),
            (  # the project's create metadata (check 6): its issue types, then one type's fields
                ["rest", "--product", "jira", "--method", "GET", "--path", meta],
                ["rest", "--product=jira", "--method=GET", f"--path={meta}"],
                False,
            ),
            (
                [
                    "rest",
                    "--product=jira",
                    "--method=GET",
                    f"--path={meta}/10001?startAt=50&maxResults=50",
                ],
                [
                    "rest",
                    "--product=jira",
                    "--method=GET",
                    f"--path={meta}/10001?startAt=50&maxResults=50",
                ],
                False,
            ),
            (
                [
                    "jira",
                    "issue",
                    "create",
                    "--project",
                    PROJ,
                    "--type",
                    "Task",
                    "--summary",
                    "A follow-up",
                    "--field-json",
                    'description={"type": "doc"}',
                ],
                [
                    "jira",
                    "issue",
                    "create",
                    f"--project={PROJ}",
                    "--type=Task",
                    "--summary=A follow-up",
                    '--field-json=description={"type": "doc"}',
                ],
                True,
            ),
            (
                ["jira", "issue", "link-epic", key, "--epic", f"{PROJ}-1"],
                ["jira", "issue", "link-epic", key, f"--epic={PROJ}-1"],
                True,
            ),
            (
                ["jira", "issue", "update", key, "--assignee", "0123456789abcdef"],
                ["jira", "issue", "update", key, "--assignee=0123456789abcdef"],
                True,
            ),
            (
                [
                    "jira",
                    "issue",
                    "comment",
                    "add",
                    key,
                    "--body",
                    "Seen again\non develop.",
                ],
                [
                    "jira",
                    "issue",
                    "comment",
                    "add",
                    key,
                    "--body=Seen again\non develop.",
                ],
                True,
            ),
            (
                [
                    "jira",
                    "issue",
                    "remote-link",
                    key,
                    "--url",
                    "https://example.com/pull/7",
                    "--title",
                    "the fix",
                ],
                [
                    "jira",
                    "issue",
                    "remote-link",
                    key,
                    "--url=https://example.com/pull/7",
                    "--title=the fix",
                ],
                True,
            ),
        ):
            with self.subTest(words=words):
                call = writer.atlassian_call(words)
                self.assertEqual(
                    (call.argv, call.writes, call.timeout_s),
                    (argv, writes, writer.ATLASSIAN_TIMEOUT_S),
                )
                self.assertEqual(writer.refused([writer.ATLASSIAN, *words]), "")
        call = writer.atlassian_call(["--timeout-s", "20", "--", "site", "active"])
        self.assertEqual(
            (call.argv, call.command, call.timeout_s),
            (["site", "active"], ("site", "active"), 20),
        )

    def test_anything_else_is_refused_before_it_runs(self):
        """A command, an argument or an option off the list, a value out of its form, and a value that starts with @
        (a CLI may read the file named after an @, and the writer reads its own secrets).
        """
        key = f"{PROJ}-3"
        off = "is not one of the atlassian commands the writer runs"
        most = f"1 to {writer.ATLASSIAN_TIMEOUT_MAX_S}"
        for words, why in (
            ([], "no atlassian command"),
            (["jira"], "`atlassian jira` is not a whole command"),
            (["auth", "login"], f"`atlassian auth login` {off}"),
            (["auth", "logout"], f"`atlassian auth logout` {off}"),
            (["jira", "issue", "delete", key], f"`atlassian jira issue delete` {off}"),
            (
                ["jira", "issue", "transition", key],
                f"`atlassian jira issue transition` {off}",
            ),
            (["confluence", "page", "create"], f"`atlassian confluence` {off}"),
            (
                [
                    "confluence",
                    "search",
                    "--cql",
                    'space = "OPS"',
                    "--max-results",
                    "1",
                ],
                f"`atlassian confluence` {off}",
            ),
            (
                [
                    "rest",
                    "--product",
                    "jira",
                    "--method",
                    "POST",
                    "--path",
                    "/rest/api/3/issue",
                ],
                "--method's value is not one the writer passes to `atlassian rest`",
            ),
            (
                [
                    "rest",
                    "--product",
                    "jira",
                    "--method",
                    "GET",
                    "--path",
                    "/rest/api/3/myself",
                ],
                "--path's value is not one the writer passes to `atlassian rest`",
            ),
            (
                [
                    "rest",
                    "--product",
                    "jira",
                    "--method",
                    "GET",
                    "--path",
                    f"/rest/api/3/issue/createmeta/{PROJ}/issuetypes?expand=all",
                ],
                "--path's value is not one the writer passes to `atlassian rest`",
            ),
            (
                ["rest", "--product", "jira", "--method", "GET"],
                "`atlassian rest` needs --path",
            ),
            (["--verbose", "version"], "--verbose is not an option the writer passes"),
            (
                ["version", "--verbose"],
                "--verbose is not an option the writer passes to `atlassian version`",
            ),
            (
                ["--profile", "other", "version"],
                "--profile is not an option the writer passes",
            ),
            (
                ["jira", "issue", "get", key, "--expand", "changelog"],
                "--expand is not an option the writer passes to `atlassian jira issue get`",
            ),
            (
                ["jira", "issue", "get", key, key],
                f"{key!r} is not an argument the writer passes to `atlassian jira issue get`",
            ),
            (
                ["jira", "issue", "get"],
                "`atlassian jira issue get` needs 1 argument(s)",
            ),
            (
                ["jira", "issue", "create", "--summary", "x"],
                "`atlassian jira issue create` needs --project",
            ),
            (
                ["jira", "issue", "create", "--project", f"{PROJ} OR x"],
                "--project's value is not one the writer passes to `atlassian jira issue create`",
            ),
            (
                ["jira", "user", "get", "@f"],
                "'@f' is not an argument the writer passes to `atlassian jira user get`",
            ),
            (
                ["jira", "issue", "comment", "add", key, "--body", "@/w/teams_webhook"],
                "--body's value is not one the writer passes to `atlassian jira issue comment add` (a value "
                "never starts with @)",
            ),
            (
                ["jira", "issue", "comment", "add", key, "--body=@./notes.md"],
                "--body's value is not one the writer passes to `atlassian jira issue comment add` (a value "
                "never starts with @)",
            ),
            (
                [
                    "jira",
                    "issue",
                    "create",
                    "--project",
                    PROJ,
                    "--field-json",
                    "description=@/w/x",
                ],
                "--field-json's value is not one the writer passes to `atlassian jira issue create`",
            ),
            (
                ["jira", "issue", "remote-link", key, "--url", "file:///etc/hostname"],
                "--url's value is not one the writer passes to `atlassian jira issue remote-link`",
            ),
            (
                ["jira", "issue", "update", key, "--assignee", "a b"],
                "--assignee's value is not one the writer passes to `atlassian jira issue update`",
            ),
            (["--site", "a site", "version"], "--site 'a site' is not a site's name"),
            (
                ["--timeout-s", "0", "version"],
                f"--timeout-s takes a number of seconds, {most}",
            ),
            (
                ["--timeout-s=601", "version"],
                f"--timeout-s takes a number of seconds, {most}",
            ),
            (["--timeout-s"], "--timeout-s needs a value"),
            (["jira", "issue", "get", key, "--site"], "--site needs a value"),
        ):
            with self.subTest(words=words):
                with self.assertRaises(ValueError) as cm:
                    writer.atlassian_call(words)
                self.assertEqual(str(cm.exception), why)
                self.assertEqual(writer.refused([writer.ATLASSIAN, *words]), why)

    def test_an_option_given_twice_is_refused_but_a_field(self):
        """The CLI takes an option's last value: each option comes once, so the value checked is the one passed. A
        field may repeat, one field each."""
        key, twice = (
            f"{PROJ}-3",
            "is given twice: only --field and --field-json may repeat",
        )
        for words, name in (
            (["--site", "a", "--site", "b", "version"], "--site"),
            (["--site=a", "version", "--site", "a"], "--site"),
            (["--json", "version", "--json"], "--json"),
            (["--timeout-s", "5", "--timeout-s=6", "version"], "--timeout-s"),
            (
                ["jira", "issue", "create", "--project", PROJ, "--project", ELSEWHERE],
                "--project",
            ),
            (
                ["jira", "issue", "update", key, "--summary", "a", "--summary=b"],
                "--summary",
            ),
            (["jira", "issue", "list", "--jql", "a", "--jql", "b"], "--jql"),
        ):
            with self.subTest(words=words):
                with self.assertRaises(ValueError) as cm:
                    writer.atlassian_call(words)
                self.assertEqual(str(cm.exception), f"{name} {twice}")
                self.assertEqual(
                    writer.refused([writer.ATLASSIAN, *words]), f"{name} {twice}"
                )
        fields = ["--field", "a=1", "--field=b=2", "--field-json", "c=[]"]
        call = writer.atlassian_call(
            ["jira", "issue", "update", key, *fields, "--field-json", "d={}"]
        )
        self.assertEqual(
            call.argv[-4:],
            ["--field=a=1", "--field=b=2", "--field-json=c=[]", "--field-json=d={}"],
        )

    def test_a_write_goes_to_the_configured_site_and_project_alone(self):
        """writer.held, with the configured site and project (atlassian.listed passes them): a write names the site,
        and the project it files in, the ticket it changes and the epic it links that to are the project's; the
        configured default epic may be another project's. A read goes anywhere the login reaches.
        """
        site, key, other = "example-site", f"{PROJ}-3", f"{ELSEWHERE}-4"
        on_site = f"a JIRA write goes to the configured site alone: --site {site}"
        in_project = (
            f"a JIRA write goes to the configured project's tickets alone ({PROJ}), not"
        )
        for words, why in (
            (
                [
                    "--site",
                    "elsewhere",
                    "jira",
                    "issue",
                    "update",
                    key,
                    "--summary",
                    "x",
                ],
                on_site,
            ),
            (["jira", "issue", "comment", "add", key, "--body", "x"], on_site),
            (
                ["--site", site, "jira", "issue", "create", "--project", ELSEWHERE],
                f"{in_project} {ELSEWHERE}",
            ),
            (
                ["--site", site, "jira", "issue", "update", other, "--assignee", "a-1"],
                f"{in_project} {other}",
            ),
            (
                [
                    "--site",
                    site,
                    "jira",
                    "issue",
                    "comment",
                    "add",
                    other,
                    "--body",
                    "x",
                ],
                f"{in_project} {other}",
            ),
            (
                [
                    "--site",
                    site,
                    "jira",
                    "issue",
                    "remote-link",
                    other,
                    "--url",
                    "https://example.com/x",
                ],
                f"{in_project} {other}",
            ),
            (
                ["--site", site, "jira", "issue", "link-epic", key, "--epic", other],
                f"{in_project} {other}",
            ),
            (["--site", site, "jira", "issue", "create", "--project", PROJ], ""),
            (
                [
                    "--site",
                    site,
                    "jira",
                    "issue",
                    "link-epic",
                    key,
                    "--epic",
                    f"{PROJ}-1",
                ],
                "",
            ),
            (["--site", "elsewhere", "--json", "jira", "issue", "get", other], ""),
            (["--json", "jira", "project", "versions", ELSEWHERE], ""),
        ):
            with self.subTest(words=words):
                call = writer.atlassian_call(words)
                self.assertEqual(writer.held(call, site, PROJ), why)
        link = ["--site", site, "jira", "issue", "link-epic", key, "--epic", other]
        self.assertEqual(
            writer.held(writer.atlassian_call(link), site, PROJ, other), ""
        )
        create = writer.atlassian_call(
            ["--site", site, "jira", "issue", "create", "--project", PROJ]
        )
        self.assertEqual(
            writer.held(create, "", PROJ),
            "a JIRA write goes to the configured site alone, and integrations.atlassian_cli.jira_url is not set",
        )
        self.assertEqual(
            writer.held(create, site, ""),
            "a JIRA write goes to the configured project alone, and features.ticket_creation.project_key is not "
            "set",
        )

    def test_the_site_a_write_names_is_the_one_the_writer_keeps_writes_to(self):
        """atlassian.jira_site, the site the CLI's JIRA calls name: the site's name, as the CLI names it, whether the
        settings give JIRA's address (the deploy settings file's jira_url: the name is its host's first label) or the
        name itself, as a host's setup does. The filer's write names it (tickets), and the writer makes that write
        (atlassian.outside), not one to another site."""
        site, key, account = "example-site", f"{PROJ}-3", "0" * 24
        for value in (
            f"https://{site}.atlassian.net",
            f"https://{site}.atlassian.net/",
            site,
        ):
            with self.subTest(value=value):
                cfg = dict(self.cfg, atlassian={"site": value}, jira={"project": PROJ})
                self.assertEqual(atlassian.jira_site(cfg), site)
                sent = []
                tickets.assign(cfg, key, account, lambda c: sent.append(c) or (0, ""))
                self.assertEqual(sent[0][1:3], ["--site", site])
                write = writer.atlassian_call(sent[0][1:])
                self.assertEqual(atlassian.outside(cfg, write), "")
                elsewhere = writer.atlassian_call(["--site", "elsewhere", *sent[0][3:]])
                self.assertEqual(
                    atlassian.outside(cfg, elsewhere),
                    f"a JIRA write goes to the configured site alone: --site {site}",
                )
        self.assertEqual(atlassian.jira_site(dict(self.cfg, atlassian={})), "")


class TestTheWritersHalves(Base):
    """As the writer's rk (on a host, the one user's), where the secret is read: `rk teams-send`, `rk web-push`, `rk
    web-requests` and `rk atlassian` send or read with it, after their own checks."""

    def setUp(self):
        super().setUp()
        p = self.as_the_writer()
        p.start()
        self.addCleanup(p.stop)

    def test_a_team_chat_message_from_stdin(self):
        self.cfg = dict(
            self.cfg,
            features={
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                }
            },
        )
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(HOOK + "\n")
        posted = []
        message = post.teams_message(
            "rkmon-b", "1 decision", [{"type": "TextBlock", "text": "#7 waits"}]
        )
        with mock.patch(
            "urllib.request.urlopen",
            lambda req, timeout=None: posted.append(
                (req.full_url, json.loads(req.data), timeout)
            )
            or Answer(202),
        ):
            code, out = self.rk(
                [writer.TEAMS_SEND, "--type", post.KIND_NEEDS_YOU],
                stdin=json.dumps(message),
            )
            self.assertEqual((code, out), (0, "HTTP 202\n"))
            self.assertEqual(posted, [(HOOK, message, post.TEAMS_TIMEOUT_S)])
            for stdin, why in (
                ("not JSON", "the message is not JSON"),
                (json.dumps({"text": "any text"}), "not a team chat message"),
                (
                    json.dumps(dict(message, attachments=message["attachments"] * 2)),
                    "not a team chat message",
                ),
                (
                    "x" * (post.TEAMS_MESSAGE_MAX_BYTES + 1),
                    f"larger than {post.TEAMS_MESSAGE_MAX_BYTES} bytes",
                ),
            ):
                with self.subTest(why=why):
                    code, out = self.rk(
                        [writer.TEAMS_SEND, "--type", post.KIND_BREAK], stdin=stdin
                    )
                    self.assertEqual(code, 4, out)
                    self.assertTrue(out.startswith("REFUSED: "), out)
                    self.assertIn(why, out)
                    self.assertIn("nothing was sent", out)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    self.rk(
                        [writer.TEAMS_SEND, "--type", "other"],
                        stdin=json.dumps(message),
                    )[0],
                    2,
                )  # not a type of message
        self.assertEqual(len(posted), 1)

    def test_the_page_and_the_requests_file_over_the_key(self):
        self.cfg = dict(
            self.cfg,
            features={"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            publish={
                "html": {
                    "host": "web.example",
                    "user": "op",
                    "dest": "site/index.html",
                    "requests": "site-requests.jsonl",
                    "host_keys": [  # the web host's own key, recorded: ssh reaches it
                        "web.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExamplePublicKeyNotReal0000000000000000"
                    ],
                }
            },
        )
        page = self.d / "page.html"
        page.write_text("<p>the page</p>")
        ran = []

        def rsync(cmd, **kw):
            ran.append(cmd)
            if cmd[-1].endswith("requests.jsonl") and not cmd[-1].startswith("op@"):
                pathlib.Path(cmd[-1]).write_text('{"pr": 7}\n')
            return subprocess.CompletedProcess(cmd, 0, "", "")

        self.assertIsNone(publish.push_page(self.cfg, str(page), run=rsync))
        self.assertEqual(
            publish.read_requests(self.cfg, run=rsync), ('{"pr": 7}\n', None)
        )
        push, read = ran
        self.assertEqual(push[-2:], [str(page), "op@web.example:site/index.html"])
        self.assertIn("--chmod=F644", push)
        self.assertEqual(read[-2], "op@web.example:site-requests.jsonl")
        for cmd in ran:  # the key the writer reads: its volume's
            self.assertIn(
                f"-i {self.write_secrets / 'web_host_ssh_key'} ",
                cmd[cmd.index("-e") + 1],
            )
        missing = 'rsync: [sender] link_stat "/home/op/site-requests.jsonl" failed: No such file or directory (2)'
        self.assertEqual(
            publish.read_requests(
                self.cfg,
                run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 23, "", missing),
            ),
            ("", None),
        )  # nobody has asked yet
        with mock.patch.object(
            publish,
            "_rsync",
            lambda *a, **k: "Permission denied (publickey).",
        ):
            self.assertEqual(
                self.rk([writer.WEB_PUSH, str(page)]),
                (1, f"{publish.NOT_PUSHED}Permission denied (publickey).\n"),
            )
            self.assertEqual(
                self.rk([writer.WEB_REQUESTS]),
                (1, f"{publish.NOT_READ}Permission denied (publickey).\n"),
            )
        with mock.patch.object(publish, "_rsync", lambda *a, **k: None):
            self.assertEqual(self.rk([writer.WEB_PUSH, str(page)]), (0, "pushed\n"))
        self.cfg = dict(self.cfg, publish={})
        self.assertEqual(
            self.rk([writer.WEB_PUSH, str(page)])[0], 1
        )  # nowhere to push it

    def test_the_web_host_s_login_host_and_paths_as_the_configuration_names_them(self):
        """The configuration is the monitor's, which the session may edit: the writer refuses a login or host that is
        not a plain name, a path that is absolute, starts with ~ or -, or has a .. or dot part, and a requests file
        other than the one the page's folder gives, at the top of the login's home (derived.site_requests), the place
        beside that folder included; nothing is pushed or read. A site folder one level deep has its requests file
        there too."""
        good = {
            "host": "web.example",
            "user": "op",
            "dest": "rocke-monitor/site/index.html",
            "requests": "rocke-monitor+site-requests.jsonl",
        }
        page = self.d / "page.html"
        page.write_text("<p>the page</p>")
        push, read = [writer.WEB_PUSH, str(page)], [writer.WEB_REQUESTS]
        plain = "is not a plain name"
        home = "the writer takes a path inside the web host login's home"
        dots = "has a .., dot-folder or dot-file part"
        outside = (
            "publish.html.requests is not the requests file named after the folder of publish.html.dest, at the top "
            "of the login's home"
        )
        cases = (
            ({"user": "-oProxyCommand=x"}, push, f"publish.html.user {plain}"),
            ({"user": "op@web.example"}, read, f"publish.html.user {plain}"),
            ({"host": "-oProxyCommand=x"}, push, f"publish.html.host {plain}"),
            ({"host": "web.example:22"}, read, f"publish.html.host {plain}"),
            (
                {"dest": "/var/www/index.html"},
                push,
                f"publish.html.dest starts with /: {home}",
            ),
            (
                {"dest": "~/site/page"},
                push,
                f"publish.html.dest starts with ~: {home}",
            ),
            ({"dest": "-e.html"}, push, f"publish.html.dest starts with -: {home}"),
            (
                {"dest": "rocke-monitor/../.ssh/authorized_keys"},
                push,
                f"publish.html.dest {dots}",
            ),
            ({"dest": ".ssh/authorized_keys"}, push, f"publish.html.dest {dots}"),
            ({"dest": "rocke-monitor/.profile"}, push, f"publish.html.dest {dots}"),
            (
                {"requests": "/etc/shadow"},
                read,
                f"publish.html.requests starts with /: {home}",
            ),
            (
                {"requests": "~/rocke-monitor/r.jsonl"},
                read,
                f"publish.html.requests starts with ~: {home}",
            ),
            (
                {"requests": "-r.jsonl"},
                read,
                f"publish.html.requests starts with -: {home}",
            ),
            (
                {"requests": "rocke-monitor/../../etc/shadow"},
                read,
                f"publish.html.requests {dots}",
            ),
            (
                {"requests": "rocke-monitor/.ssh/id_ed25519"},
                read,
                f"publish.html.requests {dots}",
            ),
            ({"requests": "other/r.jsonl"}, read, outside),
            ({"requests": "r.jsonl"}, read, outside),
            ({"requests": "rocke-monitor/requests/r.jsonl"}, read, outside),
            ({"requests": "rocke-monitor/site/requests.jsonl"}, read, outside),
            ({"requests": "rocke-monitor/site-requests.jsonl"}, read, outside),
            ({"dest": "index.html", "requests": "r.jsonl"}, read, outside),
        )
        ran = []
        with mock.patch.object(
            publish, "_rsync", lambda cfg, src, dest, *a, **k: ran.append(src) or None
        ):
            for change, argv, why in cases:
                with self.subTest(change=change):
                    self.cfg = dict(
                        self.cfg,
                        features={
                            "dashboards": {"team": {"enabled": True, "mode": "live"}}
                        },
                        publish={"html": dict(good, **change)},
                    )
                    self.assertEqual(self.rk(argv), (4, writer.refusal(why) + "\n"))
            self.assertEqual(ran, [])  # nothing was pushed or read
            self.cfg = dict(self.cfg, publish={"html": good})
            self.assertEqual(self.rk(push), (0, "pushed\n"))
            self.assertEqual(self.rk(read), (0, ""))
            one_level = {"dest": "site/index.html", "requests": "site-requests.jsonl"}
            self.cfg = dict(self.cfg, publish={"html": dict(good, **one_level)})
            self.assertEqual(self.rk(read), (0, ""))
        self.assertEqual(
            ran,
            [
                str(page),
                "op@web.example:rocke-monitor+site-requests.jsonl",
                "op@web.example:site-requests.jsonl",
            ],
        )

    def test_a_proxy_the_configuration_names_never_reaches_the_cli(self):
        """The configuration is the monitor's, which the session may edit: the CLI the writer runs with its login is
        never given the proxy it names (no feature reads atlassian.proxy_url)."""
        self.cfg = dict(
            self.cfg,
            atlassian=dict(
                self.cfg["atlassian"], proxy_url="https://proxy.example.invalid"
            ),
        )
        given = self.d / "atlassian-env.log"
        self.cli_here(FAKE_ATLASSIAN_ENV_LOG=str(given))
        os.environ.pop("ATLASSIAN_PROXY_URL", None)
        self.assertEqual(
            self.rk([writer.ATLASSIAN, "auth", "status"]),
            (0, "Logged in as op (fake)\n"),
        )
        self.assertEqual(
            [json.loads(x) for x in given.read_text().splitlines()],
            [{"ATLASSIAN_CONFIG_DIR": str(self.write_secrets / "atlassian")}],
        )

    def test_the_atlassian_cli_runs_the_listed_command_with_its_login(self):
        self.cfg = dict(
            self.cfg,
            features={"ticket_creation": TICKETS_ON},
            atlassian={"site": "example"},  # where its writes go (writer.held)
            jira={"project": PROJ},
        )
        log = self.cli_here(FAKE_ATLASSIAN_LOGGED_OUT="1")
        login = self.write_secrets / "atlassian"
        login.mkdir(parents=True)
        (login / "login").write_text(
            "read-write\n"
        )  # the fake's own login, in the writer's volume
        key = f"{PROJ}-1"
        code, out = self.rk([writer.ATLASSIAN, "auth", "status"])
        self.assertEqual((code, out), (0, "Logged in as op (fake)\n"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, out = self.rk(
                [
                    writer.ATLASSIAN,
                    "--site",
                    "example",
                    "--json",
                    "jira",
                    "issue",
                    "create",
                    "--project",
                    PROJ,
                    "--type",
                    "Task",
                    "--summary",
                    "A follow-up",
                ]
            )
        self.assertEqual((code, json.loads(out)["key"]), (0, key))
        self.assertIn("is available", err.getvalue())  # the CLI's stderr, as it gave it
        code, out = self.rk(
            [writer.ATLASSIAN, "--site=example", "jira", "issue", "get", key]
        )
        self.assertEqual(
            (code, json.loads(out)["fields"]["summary"]), (0, "A follow-up")
        )
        self.assertEqual(
            self.calls(log),
            [
                ["auth", "status"],
                [
                    "--site=example",
                    "--json",
                    "jira",
                    "issue",
                    "create",
                    f"--project={PROJ}",
                    "--type=Task",
                    "--summary=A follow-up",
                ],
                ["--site=example", "jira", "issue", "get", key],
            ],
        )
        for argv, why in (
            (
                [writer.ATLASSIAN, "jira", "issue", "delete", key],
                "`atlassian jira issue delete` is not one of the atlassian commands the writer runs",
            ),
            (
                [writer.ATLASSIAN, "--timeout-s", "0", "version"],
                f"--timeout-s takes a number of seconds, 1 to {writer.ATLASSIAN_TIMEOUT_MAX_S}",
            ),
        ):
            self.assertEqual(self.rk(argv), (4, writer.refusal(why) + "\n"))
        self.assertEqual(len(self.calls(log)), 3)  # nothing ran

    def test_the_cli_s_answer_its_time_limit_and_no_token(self):
        """The login's own commands lose each line that looks like a token; the rest is the CLI's answer as it gave
        it. The CLI runs with its login folder, nothing on its stdin, within its time limit.
        """
        ran = []

        def run(argv, **kw):
            ran.append((argv, kw))
            return subprocess.CompletedProcess(
                argv, 0, "Logged in as op\naccess token: " + "A" * 40 + "\n", "B" * 41
            )

        with mock.patch.object(
            atlassian.shutil, "which", lambda name: "/x/atlassian"
        ), mock.patch.object(atlassian.subprocess, "run", run):
            p = atlassian.listed(self.cfg, ["auth", "status"])
            self.assertEqual((p.stdout, p.stderr), ("Logged in as op\n", ""))
            p = atlassian.listed(
                self.cfg, ["--timeout-s=5", "jira", "issue", "get", f"{PROJ}-1"]
            )
            self.assertIn("A" * 40, p.stdout)
        (argv, kw), (_argv, kw2) = ran
        self.assertEqual(argv, ["/x/atlassian", "auth", "status"])
        self.assertEqual(
            (kw["timeout"], kw2["timeout"], kw["stdin"]),
            (writer.ATLASSIAN_TIMEOUT_S, 5, subprocess.DEVNULL),
        )
        self.assertEqual(
            kw["env"]["ATLASSIAN_CONFIG_DIR"], str(self.write_secrets / "atlassian")
        )
        with mock.patch.object(
            atlassian.shutil, "which", lambda name: "/x/atlassian"
        ), mock.patch.object(
            atlassian.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("atlassian", 5),
        ):
            self.assertEqual(atlassian.listed(self.cfg, ["version"]).returncode, 124)
        with mock.patch.object(atlassian.shutil, "which", lambda name: None):
            self.assertEqual(atlassian.listed(self.cfg, ["version"]).returncode, 127)


class TestTheMonitorSendsThroughTheWriter(Base):
    """As the monitor in the container, which reads none of the writer's secrets, each send or read that needs one
    goes to the writer: sudo, docker/rk_write.py's checks, then the writer's rk (`the_writer`).
    """

    def setUp(self):
        super().setUp()
        self.write_secrets.mkdir()
        for p in (self.as_the_monitor(), self.the_writer()):
            p.start()
            self.addCleanup(p.stop)

    def test_a_team_chat_message(self):
        (self.write_secrets / "teams_webhook").write_text(HOOK + "\n")
        self.cfg = dict(
            self.cfg,
            features={
                "alerts": {
                    "enabled": True,
                    "mode": "live",
                    "develop_breaks": True,
                    "needs_you": True,
                }
            },
            teams={"channel": "ops"},
        )
        self.assertFalse(logins.here(sr.entry(self.cfg, "teams_webhook")))
        posted = []
        st = self.store.load()
        asks = [
            report.need_ask(
                "HOLD_STALE",
                7,
                "h7",
                "#7 has been held 30 h without a push: keep its place?",
            )
        ]
        with mock.patch(
            "urllib.request.urlopen",
            lambda req, timeout=None: posted.append(
                (req.full_url, json.loads(req.data))
            )
            or Answer(202),
        ):
            lines = needs_teams.send(self.cfg, st, self.store, asks)
            self.assertEqual(len(posted), 1, lines)
            self.assertEqual(posted[0][0], HOOK)  # read by the writer
            self.assertIn([writer.TEAMS_SEND, "--type", post.KIND_NEEDS_YOU], self.sent)
            self.sent.clear()
            with self.assertRaises(post.PostRefused) as cm:
                post.teams_post(
                    dict(self.cfg, mirror={"enabled": True}),
                    posted[0][1],
                    post.KIND_BREAK,
                )
            self.assertEqual(
                str(cm.exception), mirror.REFUSAL.format(cmd=writer.TEAMS_SEND)
            )
            self.assertEqual(self.sent, [])  # refused before the writer is asked
            with self.assertRaises(post.PostRefused) as cm:
                post.teams_post(self.cfg, {"text": "any text"}, post.KIND_BREAK)
            self.assertIn("not a team chat message", str(cm.exception))  # the writer's
        self.assertEqual(len(posted), 1)

    def test_the_team_dashboard_s_page_and_requests(self):
        self.cfg = dict(
            self.cfg,
            features={"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            publish={
                "html": {
                    "host": "web.example",
                    "user": "op",
                    "dest": "site/index.html",
                    "requests": "site-requests.jsonl",
                }
            },
        )
        (self.write_secrets / "web_host_ssh_key").write_text(
            a_value(sr.entry(self.cfg, "web_host_ssh_key").shape, 5)
        )
        self.assertFalse(publish.key_here(self.cfg))
        self.assertTrue(
            features.active(self.cfg, "team-dashboard")
        )  # the key set, as the writer says
        seen = []

        def rsync(cfg, src, dest, run, *opts, missing_ok=False):
            sent = pathlib.Path(src).read_text() if os.path.isfile(src) else None
            seen.append((src, dest, sent))
            if not dest.startswith("op@"):
                pathlib.Path(dest).write_text('{"pr": 7}\n')
            return None

        page = self.d / "published.html"
        page.write_text("<p>the page</p>")
        pulled = self.d / "pulled.jsonl"
        with mock.patch.object(publish, "_rsync", rsync):
            self.assertIsNone(publish.html(self.cfg, str(page)))
            self.assertIsNone(publish.pull_requests(self.cfg, str(pulled)))
        self.assertEqual(pulled.read_text(), '{"pr": 7}\n')
        (src, to, sent), (frm, _dest, _) = seen
        self.assertEqual(
            (to, frm),
            ("op@web.example:site/index.html", "op@web.example:site-requests.jsonl"),
        )
        self.assertNotEqual(src, str(page))  # the writer pushed its checked copy
        self.assertEqual(sent, "<p>the page</p>")
        self.assertFalse(os.path.exists(src))  # gone when the writer's run ended
        self.assertIn([writer.WEB_PUSH, str(page)], self.sent)
        self.assertIn([writer.WEB_REQUESTS], self.sent)

        def emptied(cfg, src, dest, run, *opts, missing_ok=False):
            pathlib.Path(dest).write_text("")  # the web host's file, emptied
            return None

        with mock.patch.object(publish, "_rsync", emptied):
            self.assertIsNone(publish.pull_requests(self.cfg, str(pulled)))
        self.assertEqual(
            pulled.read_text(), ""
        )  # emptied here too, as rsync empties it
        with mock.patch.object(
            publish, "_rsync", lambda *a, **k: "Permission denied (publickey)."
        ):
            self.assertEqual(
                publish.html(self.cfg, str(page)), "Permission denied (publickey)."
            )
            self.assertEqual(
                publish.pull_requests(self.cfg, str(pulled)),
                "Permission denied (publickey).",
            )

    def test_in_prototype_the_monitor_asks_the_writer_for_nothing(self):
        """While the alerts and the team dashboard do not send, the monitor keeps its team chat message and
        its page as unsent items where it makes them, and asks the writer for nothing: none of the writer's secrets is
        needed (none is set here)."""
        self.cfg = dict(
            self.cfg,
            features={
                "alerts": {
                    "enabled": True,
                    "mode": "prototype",
                    "develop_breaks": True,
                    "needs_you": True,
                },
                "dashboards": {"team": {"enabled": True, "mode": "prototype"}},
            },
            teams={"channel": "ops"},
            publish={
                "html": {
                    "host": "web.example",
                    "user": "op",
                    "dest": "site/index.html",
                    "requests": "site-requests.jsonl",
                }
            },
        )
        asks = [
            report.need_ask(
                "HOLD_STALE",
                7,
                "h7",
                "#7 has been held 30 h without a push: keep its place?",
            )
        ]
        page = self.d / "published.html"
        page.write_text("<p>the page</p>")
        st = self.store.load()
        with mock.patch(
            "urllib.request.urlopen", side_effect=AssertionError("a request")
        ), mock.patch.object(publish, "_rsync", side_effect=AssertionError("rsync")):
            needs_teams.send(self.cfg, st, self.store, asks, own=True)
            self.assertIsNone(publish.team(self.cfg, st, self.store, str(page)))
            self.assertIsNone(
                publish.pull_requests(self.cfg, str(self.d / "pulled.jsonl"))
            )
        self.assertEqual(self.sent, [])
        self.assertEqual(
            sorted(i["type"] for i in state_store.unsent_items(self.store.dir)),
            sorted([state_store.ALERT, state_store.PAGE]),
        )

    def test_the_atlassian_cli_s_calls(self):
        log = self.cli_here(
            FAKE_ATLASSIAN_LOGGED_OUT="1", FAKE_ATLASSIAN_PROJECT_SITES="example-site"
        )
        (self.write_secrets / "atlassian").mkdir()
        (self.write_secrets / "atlassian" / "login").write_text(
            "read-write\n"
        )  # the fake's own login, in the writer's volume
        self.cfg = dict(
            self.cfg, atlassian={"site": "example-site"}, jira={"project": PROJ}
        )
        self.assertFalse(atlassian.login_here(self.cfg))
        s = atlassian.status(self.cfg)
        self.assertEqual((s["logged_in"], s["site_ok"]), (True, True), s)
        (row,) = reach.jira(self.cfg, get=lambda url, h=None: (200, ""))
        self.assertEqual((row.name, row.status), ("JIRA project", "PASS"), row.found)
        self.assertEqual(
            self.sent[:3],
            [
                [writer.ATLASSIAN, "--timeout-s=30", "version"],
                [writer.ATLASSIAN, "--timeout-s=30", "auth", "status"],
                [
                    writer.ATLASSIAN,
                    "--timeout-s=30",
                    "--site",
                    "example-site",
                    "site",
                    "active",
                ],
            ],
        )
        self.assertIn(
            [
                writer.ATLASSIAN,
                "--timeout-s=30",
                "--site",
                "example-site",
                "--json",
                "jira",
                "project",
                "versions",
                PROJ,
            ],
            self.sent,
        )
        ran = len(self.calls(log))
        rc, text = atlassian._run(
            ["atlassian", "jira", "issue", "delete", f"{PROJ}-1"], self.cfg
        )
        self.assertEqual(rc, 4)
        self.assertIn(
            "`atlassian jira issue delete` is not one of the atlassian commands the writer runs",
            text,
        )
        self.assertEqual(len(self.calls(log)), ran)  # nothing ran

    def a_draft_to_file(self) -> tuple[pathlib.Path, str]:
        """The CLI here, its login in the writer's volume, ticket creation on, and one draft waiting: the CLI's log
        and the draft's id."""
        log = self.cli_here()
        (self.write_secrets / "atlassian").mkdir()
        (self.write_secrets / "atlassian" / "credentials").write_text(
            "{}"
        )  # the CLI's login, in the writer's volume
        self.cfg = dict(
            self.cfg,
            features={"ticket_creation": TICKETS_ON},
            jira={
                "project": PROJ,
                "url": "https://jira.example.com",
                "labels": ["monitor"],
            },
            atlassian={"site": "example-site"},
        )
        with self.store.locked(wait=True):
            st = self.store.load()
            _ok, _msg, tid = tickets.add(self.cfg, st, self.store, tickets.parse(DRAFT))
            self.store.save(st)
        return log, tid

    def test_a_found_issue_s_ticket_is_filed_by_the_writer(self):
        log, tid = self.a_draft_to_file()
        ok, msg, extra = tickets.approve(self.cfg, self.store, tid, by="op")
        self.assertTrue(ok, msg)
        self.assertEqual(extra["key"], f"{PROJ}-1")
        self.assertEqual(self.sent, [[writer.TICKET_FILE, tid]])
        self.assertEqual(
            [c[3:6] for c in self.calls(log)], [["jira", "issue", "create"]]
        )
        self.assertEqual(tickets.find(self.store.load(), tid)["status"], "filed")
        ok, msg, _extra = tickets.approve(self.cfg, self.store, tid, by="op")
        self.assertFalse(ok)
        self.assertIn("one action per proposed ticket", msg)  # the writer's answer

    def test_the_card_waits_as_long_as_the_writer_s_filing_may_take(self):
        """The writer's rk ticket-file waits for the write lock as each outward rk command does, then makes six CLI
        calls at most (tickets.FILING_MAX_S): the card's Approve waits that long for its answer, and the writer's
        start."""
        _log, tid = self.a_draft_to_file()
        fake, lock, timeouts, waits = subprocess.run, control.write_lock, [], []

        def run(argv, *a, **kw):
            if argv[:1] == ["sudo"]:
                timeouts.append(kw.get("timeout"))
            return fake(argv, *a, **kw)

        def write_lock(folder, timeout=None):
            waits.append(timeout)
            return lock(folder, timeout)

        with mock.patch.object(subprocess, "run", run), mock.patch.object(
            control, "write_lock", write_lock
        ):
            ok, msg, _extra = tickets.approve(self.cfg, self.store, tid, by="op")
        self.assertTrue(ok, msg)
        self.assertEqual(waits, [control.WRITE_WAIT_S])  # the writer's rk ticket-file
        self.assertEqual(len(timeouts), 1)
        self.assertGreaterEqual(
            timeouts[0], waits[0] + 6 * tickets.CLI_TIMEOUT_S + writer.START_S
        )


class TestTheWriterAsksTheOutwardGate(Base):
    """Before each send it makes for another process (rkmon_lib/outward.py: gate GATED), the writer's rk
    asks the one outward gate (rkmon.py write_gate), whoever asked for the send: a mirror, the send's feature off,
    or a stop, refuses it, and nothing leaves. While the send's feature does not send (prototype, or standby), a send
    you type is refused, naming which, and one of the monitor's own is kept as an unsent item, or refused when its
    sender keeps its own item first. The other commands' own code asks before theirs (gate OWN).
    """

    ON = {
        "alerts": {
            "enabled": True,
            "mode": "live",
            "develop_breaks": True,
            "needs_you": True,
        },
        "dashboards": {"team": {"enabled": True, "mode": "live"}},
        "ticket_creation": TICKETS_ON,
    }
    PROTO = {  # each feature on, in prototype
        "alerts": dict(ON["alerts"], mode="prototype"),
        "dashboards": {"team": {"enabled": True, "mode": "prototype"}},
        "ticket_creation": dict(TICKETS_ON, mode="prototype"),
    }
    OFF = {
        "alerts": {"enabled": False},
        "dashboards": {"team": {"enabled": False}},
        "ticket_creation": {"enabled": False},
    }

    def setUp(self):
        super().setUp()
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(HOOK + "\n")
        self.log = self.cli_here()
        page = self.d / "page.html"
        page.write_text("<p>the page</p>")
        message = json.dumps(
            post.teams_message(
                "rkmon-b", "1 decision", [{"type": "TextBlock", "text": "#7 waits"}]
            )
        )
        # Each send of the table's GATED commands, one per type of team chat message: rk's words, its stdin, and the
        # capability it sends for.
        argv = {
            writer.WEB_PUSH: [writer.WEB_PUSH, str(page)],
            writer.ATLASSIAN: [
                writer.ATLASSIAN,
                "--site",
                "example-site",
                "jira",
                "issue",
                "create",
                "--project",
                PROJ,
                "--type",
                "Task",
                "--summary",
                "A follow-up",
            ],
        }
        self.sends = []
        for r in outward.TABLE:
            if r.gate != outward.GATED:
                continue
            if r.name == writer.TEAMS_SEND:
                self.sends += [
                    ([r.name, "--type", kind], message, needs[0])
                    for kind, needs in post.TEAMS_FEATURES.items()
                ]
            else:
                self.sends.append((argv[r.name], "", r.features[0]))
        self.cfg = dict(
            self.cfg,
            publish={
                "html": {"host": "web.example", "user": "op", "dest": "site/index.html"}
            },
            atlassian={"site": "example-site"},  # where its writes go (writer.held)
            jira={"project": PROJ},
        )
        self.out = []  # each message posted and each page pushed
        for p in (
            mock.patch(
                "urllib.request.urlopen",
                lambda req, timeout=None: self.out.append(req.full_url) or Answer(202),
            ),
            mock.patch.object(
                publish,
                "_rsync",
                lambda cfg, src, dest, *a, **k: self.out.append(dest),
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def refused(self, argv: list[str], stdin: str, name: str):
        """rk ARGV is refused because capability NAME is off, in the gate's words."""
        code, out = self.rk(argv, stdin=stdin)
        title = features.lowered(features.CAPABILITIES[name].title)
        self.assertEqual(code, 4, out)
        self.assertTrue(out.startswith(f"REFUSED: {title} is off ("), out)
        self.assertTrue(out.endswith(" is not sent\n"), out)

    def test_each_command_meets_one_gate(self):
        """Each of the writer's commands that sends meets one gate, its own (OWN) or rk's before the send (GATED: each
        one's sends here); one that sends nothing meets none."""
        self.assertEqual(len(set(writer.COMMANDS)), len(writer.COMMANDS))
        for name in writer.COMMANDS:
            with self.subTest(cmd=name):
                r = outward.row(name)
                self.assertIn(r.gate, (outward.OWN, outward.GATED, outward.NONE))
                self.assertEqual(r.gate == outward.NONE, r.dest == outward.NONE)
        self.assertEqual(
            {argv[0] for argv, _in, _name in self.sends},
            set(outward.names(gate=outward.GATED)),
        )

    def test_a_stop_refuses_each_send_and_a_read_passes(self):
        """As the monitor in the container, while the monitor is stopped: each send the writer makes for another
        process is refused there, as the monitor's own sends are (control.refusal), and nothing leaves; an atlassian
        read still runs."""
        st = self.store.load()
        st["stopped"] = {"at": "2026-01-01T00:00:00Z", "how": "stop", "boot": ""}
        self.store.save(st)
        self.cfg = dict(self.cfg, features=self.ON)
        with self.as_the_monitor(), self.the_writer():
            for argv, stdin, _name in self.sends:
                with self.subTest(argv=argv[:3]):
                    code, out = self.rk(argv, stdin=stdin)
                    self.assertEqual(code, 4, out)
                    self.assertTrue(out.startswith("REFUSED: "), out)
                    self.assertIn(f"rk {argv[0]} is refused until then", out)
            code, out = self.rk([writer.ATLASSIAN, "auth", "status"])
            self.assertEqual(code, 0, out)
        self.assertEqual(
            [argv[0] for argv in self.sent],
            [argv[0] for argv, _in, _name in self.sends] + [writer.ATLASSIAN],
        )  # each went to the writer
        self.assertEqual(self.out, [])
        self.assertEqual([c[0] for c in self.calls(self.log)], ["auth"])

    def test_a_send_whose_feature_is_off_is_refused_and_nothing_leaves(self):
        with self.as_the_writer():
            self.cfg = dict(self.cfg, features=self.OFF)
            for argv, stdin, name in self.sends:
                with self.subTest(argv=argv[:3]):
                    self.refused(argv, stdin, name)
            self.cfg = dict(
                self.cfg,
                features=dict(
                    self.ON,
                    ticket_creation={
                        "enabled": True,
                        "triggers": {
                            "review_follow_up": False,
                            "out_of_scope_issue": True,
                        },
                    },
                ),
            )
            self.refused(*self.sends[-1])  # a found issue's ticket is rk ticket-file's
            self.assertEqual((self.out, self.calls(self.log)), ([], []))
            self.cfg = dict(self.cfg, features=self.ON)
            for argv, stdin, _name in self.sends:
                with self.subTest(argv=argv[:3]):
                    with contextlib.redirect_stderr(io.StringIO()):  # the CLI's notices
                        code, out = self.rk(argv, stdin=stdin)
                    self.assertEqual(code, 0, out)
        self.assertEqual(self.out, [HOOK, HOOK, "op@web.example:site/index.html"])
        self.assertEqual(
            [c[1:4] for c in self.calls(self.log)], [["jira", "issue", "create"]]
        )

    def test_a_send_you_type_is_refused_naming_prototype_then_standby(self):
        """A send typed as an rk command (no mark of the monitor's own writes: outward.own) while its feature does not
        send: the writer's gate refuses it, naming the feature's mode, then standby; nothing leaves, nothing is kept.
        """
        home = self.d / "home"
        with self.as_the_writer(), mock.patch.dict(
            os.environ, {"RKMON_HOME": str(home)}
        ):
            for standby in (False, True):
                self.cfg = dict(self.cfg, features=self.ON if standby else self.PROTO)
                state_store.set(home, "standby.on", standby)
                for argv, stdin, name in self.sends:
                    with self.subTest(argv=argv[:3], standby=standby):
                        held = features.held_words(
                            self.cfg, features.CAPABILITIES[name].feature
                        )
                        self.assertIn("standby" if standby else "prototype", held)
                        what = outward.row(argv[0]).what
                        self.assertEqual(
                            self.rk(argv, stdin=stdin),
                            (4, f"REFUSED: {what} is not sent: {held}\n"),
                        )
        self.assertEqual((self.out, self.calls(self.log)), ([], []))
        self.assertEqual(state_store.unsent_items(self.store.dir), [])

    def test_a_held_send_of_the_monitors_own_is_kept_or_refused(self):
        """A send of the monitor's own (outward.OWN_VAR) that the writer's gate holds: a JIRA write (the session's
        follow-up ticket) is kept as an unsent item, a ticket for the JIRA project, its content the CLI's words as the
        writer would run them, and its held line alone says so (exit 0, as for every item kept); a team chat message
        and the page, whose senders keep their own items before they ask the writer, are refused. Nothing leaves; the
        journal names the item's type and what it was for, never its content."""
        self.cfg = dict(self.cfg, features=self.PROTO)
        with self.as_the_writer(), mock.patch.dict(os.environ, {outward.OWN_VAR: "1"}):
            for argv, stdin, name in self.sends:
                with self.subTest(argv=argv[:3]):
                    held = features.held_words(
                        self.cfg, features.CAPABILITIES[name].feature
                    )
                    what = outward.row(argv[0]).what
                    said = (
                        (
                            0,
                            f"{post.UNSENT}: {what} for the JIRA project {PROJ}, kept as an unsent item: {held}\n",
                        )
                        if argv[0] == writer.ATLASSIAN
                        else (4, f"REFUSED: {what} is not sent: {held}\n")
                    )
                    self.assertEqual(self.rk(argv, stdin=stdin), said)
        jira = next(a for a, _in, _name in self.sends if a[0] == writer.ATLASSIAN)
        (item,) = state_store.unsent_items(self.store.dir)
        self.assertEqual(
            (item["type"], item["for"], item["feature"], item["why"], item["content"]),
            (
                state_store.TICKET,
                f"the JIRA project {PROJ}",
                "ticket_creation",
                state_store.PROTOTYPE,
                shlex.join(writer.atlassian_call(jira[1:]).argv),
            ),
        )
        journal = self.store.journal_path.read_text()
        self.assertIn(
            f"ticket for the JIRA project {PROJ}: not sent (ticket creation in prototype)",
            journal,
        )
        self.assertNotIn("A follow-up", journal)
        self.assertEqual((self.out, self.calls(self.log)), ([], []))

    def test_a_read_passes_and_a_mirror_refuses_each_send(self):
        """Checks are not sends: they run whatever the features say (check 6's read of the ticket fields
        runs in prototype). In a mirror nothing is sent, whatever the features say.
        """
        createmeta = [
            writer.ATLASSIAN,
            "rest",
            "--product",
            "jira",
            "--method",
            "GET",
            "--path",
            f"/rest/api/3/issue/createmeta/{PROJ}/issuetypes",
        ]
        search = [writer.ATLASSIAN, "jira", "issue", "search", "--jql", "project = X"]
        with self.as_the_writer():
            self.cfg = dict(self.cfg, features=self.OFF)
            for argv in ([writer.ATLASSIAN, "auth", "status"], createmeta, search):
                with self.subTest(argv=argv[1:4]):
                    code, out = self.rk(argv)
                    self.assertEqual(code, 0, out)
            self.cfg = dict(self.cfg, features=self.ON, mirror={"enabled": True})
            for argv, stdin, _name in self.sends:
                with self.subTest(argv=argv[:3]):
                    self.assertEqual(
                        self.rk(argv, stdin=stdin),
                        (4, f"REFUSED: {mirror.REFUSAL.format(cmd=argv[0])}\n"),
                    )
            self.assertEqual(self.rk(createmeta)[0], 0)
        self.assertEqual(self.out, [])
        self.assertEqual(
            [c[0] for c in self.calls(self.log)], ["auth", "rest", "jira", "rest"]
        )  # no create among them

    def test_a_team_chat_message_meets_the_mirror_s_refusal_once_in_each_process(self):
        """As the monitor in the container: a team chat message meets the mirror's refusal once in the monitor's process
        (post.teams_post, before it asks the writer) and once in the writer's (its gate), refused in one wording
        (gh's own check, of gh's writes at each command's start, aside: gh.configure).
        """
        checks = []
        real = mirror.on

        def on(cfg):
            checks.append("writer" if os.environ.get(gh.BROKERED) == "1" else "monitor")
            return real(cfg)

        argv, message, _name = next(
            s for s in self.sends if s[0][0] == writer.TEAMS_SEND
        )
        self.cfg = dict(self.cfg, features=self.ON)
        with self.as_the_monitor(), self.the_writer(), mock.patch.object(
            mirror, "on", on
        ), mock.patch.object(gh, "configure", lambda cfg: None):
            self.assertEqual(
                post.teams_post(self.cfg, json.loads(message), argv[2]), 202
            )
            self.assertEqual(checks, ["monitor", "writer"])
            self.cfg = dict(self.cfg, mirror={"enabled": True})
            with self.assertRaises(post.PostRefused) as cm:
                post.teams_post(self.cfg, json.loads(message), argv[2])
            self.assertEqual(
                self.rk(argv, stdin=message), (4, f"REFUSED: {cm.exception}\n")
            )
        self.assertEqual(self.out, [HOOK])
        self.assertEqual([a[0] for a in self.sent], [writer.TEAMS_SEND] * 2)

    def test_a_send_that_reaches_the_writer_any_other_way_meets_the_gate(self):
        """As the monitor in the container: a send typed as an rk command, not made by the monitor's code (which asks
        the gate first), is refused by the writer while its feature is off; a GitHub write while its feature is in
        prototype meets its command's own gate in the writer."""
        self.cfg = dict(self.cfg, features=self.OFF)
        with self.as_the_monitor(), self.the_writer():
            for argv, stdin, name in self.sends:
                with self.subTest(argv=argv[:3]):
                    self.refused(argv, stdin, name)
            self.assertEqual(
                [argv[0] for argv in self.sent],
                [argv[0] for argv, _in, _name in self.sends],
            )  # each went to the writer
            self.cfg = dict(
                self.cfg,
                features={"merge_queue": {"enabled": True, "mode": "prototype"}},
            )
            with mock.patch.object(
                post, "_gh_write", side_effect=AssertionError("a GitHub write")
            ):
                code, out = self.rk(["arm", "7"])
            self.assertEqual(self.sent[-1], ["arm", "7"])
            self.assertTrue(
                out.startswith("OFF: auto-merge is off, so nothing was done"), out
            )
        self.assertEqual((self.out, self.calls(self.log)), ([], []))


class TestTheSessionsJiraGoesThroughTheWriter(Base):
    """The Claude session as the monitor's user in the container, with its own environment (orchestrator.session_env):
    it has no atlassian login, and its JIRA calls are `rk atlassian`, which the writer runs with the login in the
    writer's volume. Each call is first the session's Bash command, which the session's guard checks as
    its hook does (orchestrator.guard).
    """

    SITE = ["--site", "example-site"]

    def setUp(self):
        super().setUp()
        self.write_secrets.mkdir()
        (self.write_secrets / "teams_webhook").write_text(HOOK + "\n")
        login = self.write_secrets / "atlassian"
        login.mkdir()
        # The CLI's login, as the writer's probe finds it, and the fake CLI's own mark of it.
        (login / "credentials").write_text("{}")
        (login / "login").write_text("read-write\n")
        self.log = self.cli_here(FAKE_ATLASSIAN_LOGGED_OUT="1")
        self.cfg = dict(
            self.cfg,
            features={"ticket_creation": TICKETS_ON},
            atlassian={"site": "example-site"},
            jira={"project": PROJ, "url": "https://jira.example.com"},
        )
        self.session = orchestrator.session_env(self.cfg)
        inherited = {
            k: v for k, v in os.environ.items() if not k.startswith("ATLASSIAN_")
        }
        for p in (
            mock.patch.dict(os.environ, dict(inherited, **self.session), clear=True),
            mock.patch.object(writer_run, "WRITE_DIR", str(self.write_secrets / "gh")),
            self.as_the_monitor(),
            self.the_writer(),
        ):
            p.start()
            self.addCleanup(p.stop)

    def bash(self, argv: list[str]) -> str | None:
        """Why the session's guard refuses its Bash command `rk ARGV`; None when it runs."""
        return orchestrator.guard(
            shlex.join(["/work/monitor/rk", *argv]), orchestrator.tickets_on(self.cfg)
        )

    def session_rk(self, argv: list[str]) -> tuple[int, str, str]:
        """rk ARGV as the session runs it: its exit code, what it printed, and what it printed on stderr."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code, out = self.rk(argv)
        return code, out, err.getvalue()

    def jira(self, *words: str):
        """The session's `rk atlassian --site SITE WORDS`: its guard lets it through and the writer runs it; the CLI's
        answer (JSON)."""
        argv = [writer.ATLASSIAN, *self.SITE, *words]
        self.assertIsNone(self.bash(argv), argv)
        code, out, err = self.session_rk(argv)
        self.assertEqual(code, 0, out + err)
        return json.loads(out)

    def jira_sent(self) -> list[list[str]]:
        """The `rk atlassian` commands the writer was sent."""
        return [argv for argv in self.sent if argv[0] == writer.ATLASSIAN]

    def test_its_environment_names_no_atlassian_login(self):
        self.assertEqual([k for k in self.session if k.startswith("ATLASSIAN_")], [])
        # Its writes are the monitor's own (outward.OWN_VAR): the gate keeps one it stops, never refuses it.
        self.assertEqual(self.session[outward.OWN_VAR], "1")
        self.assertEqual([k for k in os.environ if k.startswith("ATLASSIAN_")], [])
        # The CLI the session runs itself has no login; `rk atlassian`, which the writer runs, has the writer's.
        p = subprocess.run(
            ["atlassian", "auth", "status"], capture_output=True, text=True
        )
        self.assertEqual((p.returncode, p.stdout), (1, "Not logged in\n"))
        argv = [writer.ATLASSIAN, "auth", "status"]
        self.assertIsNone(self.bash(argv))
        self.assertEqual(self.session_rk(argv)[:2], (0, "Logged in as op (fake)\n"))
        self.assertEqual(self.jira_sent(), [argv])

    def test_a_follow_up_ticket_is_filed_and_read_through_the_writer(self):
        """The skill's steps (agentic/skills/rocke-monitor/commands.md, JIRA through the atlassian CLI)."""
        md = self.d / "ticket.md"
        md.write_text("## What happens\n\nThe nightly lane skips a case.\n")
        code, adf = self.rk(["md2adf", str(md)])
        self.assertEqual(code, 0, adf)
        key, summary = f"{PROJ}-1", "Follow up: the nightly lane skips a case"
        body = "Follows up the merged PR.\nIts review's item is still open on develop."
        jql = f'project = {PROJ} AND text ~ "nightly lane" ORDER BY updated DESC'
        steps = [
            ["--json", "jira", "issue", "list", "--jql", jql, "--max-results", "20"],
            ["--json", "jira", "user", "get", "author@example.com"],
            [
                "--json",
                "jira",
                "issue",
                "create",
                "--project",
                PROJ,
                "--type",
                "Task",
                "--summary",
                summary,
                "--priority",
                "P3: Low",
                "--labels",
                "follow-up",
                "--field-json",
                'components=[{"name":"Widgets"}]',
            ],
            ["jira", "issue", "link-epic", key, "--epic", f"{PROJ}-9"],
            [
                "jira",
                "issue",
                "update",
                key,
                "--field-json",
                f"description={adf.strip()}",
            ],
        ]
        answers = [self.jira(*words) for words in steps]
        self.assertEqual(answers[0], {"issues": []})
        self.assertEqual(answers[2]["key"], key)
        account = answers[1]["accountId"]
        steps += [
            ["jira", "issue", "update", key, "--assignee", account],
            [
                "jira",
                "issue",
                "remote-link",
                key,
                "--url",
                "https://example.com/acme/widgets/pull/7",
                "--title",
                "PR #7: Add a widget",
            ],
            ["jira", "issue", "comment", "add", key, "--body", body],
            ["--json", "jira", "issue", "get", key],
        ]
        answers += [self.jira(*words) for words in steps[len(answers) :]]
        self.assertEqual(answers[-1]["fields"]["summary"], summary)
        sent = [[writer.ATLASSIAN, *self.SITE, *words] for words in steps]
        self.assertEqual(self.jira_sent(), sent)
        self.assertEqual(
            self.marked, [True] * len(self.sent)
        )  # each command of the session's reached the writer as the monitor's own (outward.OWN_WORD)
        self.assertEqual(
            self.calls(self.log), [writer.atlassian_call(a[1:]).argv for a in sent]
        )  # each as the writer lists it, with the login
        comments = (self.d / "jira.comments").read_text().splitlines()
        self.assertEqual(
            [json.loads(c) for c in comments], [{"key": key, "body": body}]
        )

    def test_no_secret_of_the_writer_s_reaches_it(self):
        hook = str(self.write_secrets / "teams_webhook")
        key = f"{PROJ}-1"
        for argv in (
            [writer.WEB_PUSH, hook],  # a file argument the writer would read
            [
                writer.ATLASSIAN,
                *self.SITE,
                "jira",
                "issue",
                "comment",
                "add",
                key,
                "--body",
                f"@{hook}",
            ],  # the CLI reads the file an @ names
            [
                writer.ATLASSIAN,
                *self.SITE,
                "jira",
                "issue",
                "update",
                key,
                "--field-json",
                f"description=@{hook}",
            ],
        ):
            with self.subTest(argv=argv[:1] + argv[-1:]):
                code, out, err = self.session_rk(argv)
                self.assertEqual(code, 4, out + err)
                self.assertTrue(out.startswith(REFUSAL_HEAD), out)
                self.assertTrue(out.endswith(REFUSAL_TAIL + "\n"), out)
                self.assertNotIn(HOOK, out + err)
        code, out, _err = self.session_rk([writer.WRITER_SECRETS])
        self.assertEqual(code, 0, out)
        self.assertIn("teams_webhook: set", out)
        self.assertNotIn(HOOK, out)
        self.assertEqual(self.calls(self.log), [])  # the CLI never ran

    def test_a_command_off_the_list_is_refused_and_nothing_runs(self):
        """The guard lets these through (follow-up tickets are on); the writer refuses each, in its own words."""
        key = f"{PROJ}-1"
        for words in (
            ["jira", "issue", "delete", key],
            ["jira", "issue", "transition", key, "--status", "Done"],
            ["jira", "issue", "get", key, "--verbose"],
            ["auth", "login"],
        ):
            argv = [writer.ATLASSIAN, *self.SITE, *words]
            with self.subTest(words=words):
                with self.assertRaises(ValueError) as cm:
                    writer.atlassian_call(argv[1:])
                self.assertIsNone(self.bash(argv))
                self.assertEqual(
                    self.session_rk(argv)[:2],
                    (4, writer.refusal(str(cm.exception)) + "\n"),
                )
        self.assertEqual(self.calls(self.log), [])

    def test_a_write_elsewhere_is_refused_and_nothing_runs(self):
        """The guard lets these through (follow-up tickets are on); the writer's rk keeps each write to the configured
        site and project (writer.held), and reads another project's tickets."""
        key, other = f"{PROJ}-1", f"{ELSEWHERE}-2"
        for words in (
            [
                "--site",
                "another-site",
                "jira",
                "issue",
                "comment",
                "add",
                key,
                "--body",
                "x",
            ],
            [
                *self.SITE,
                "jira",
                "issue",
                "create",
                "--project",
                ELSEWHERE,
                "--summary",
                "x",
            ],
            [*self.SITE, "jira", "issue", "comment", "add", other, "--body", "x"],
            [*self.SITE, "jira", "issue", "link-epic", key, "--epic", other],
        ):
            argv = [writer.ATLASSIAN, *words]
            with self.subTest(words=words):
                self.assertIsNone(self.bash(argv))
                why = writer.held(writer.atlassian_call(words), "example-site", PROJ)
                self.assertTrue(why)
                self.assertEqual(
                    self.session_rk(argv)[:2], (4, writer.refusal(why) + "\n")
                )
        self.assertEqual(self.calls(self.log), [])
        read = ["--json", "jira", "issue", "list", "--jql", f"project = {ELSEWHERE}"]
        self.assertEqual(self.jira(*read), {"issues": []})

    def test_a_write_for_a_feature_that_is_off_or_in_prototype_is_refused(self):
        create = [
            writer.ATLASSIAN,
            *self.SITE,
            "jira",
            "issue",
            "create",
            "--project",
            PROJ,
            "--type",
            "Task",
            "--summary",
            "A follow-up",
        ]
        read = ["--json", "jira", "issue", "list", "--jql", f"project = {PROJ}"]
        self.cfg = dict(self.cfg, features={"ticket_creation": {"enabled": False}})
        # The guard leaves it to rk, which weighs the feature on each call: the writer refuses it.
        self.assertIsNone(self.bash(create))
        code, out, _err = self.session_rk(create)
        self.assertEqual(code, 4, out)
        self.assertTrue(out.startswith("REFUSED: follow-up tickets is off ("), out)
        self.assertEqual(self.jira(*read), {"issues": []})  # a read runs
        self.cfg = dict(
            self.cfg, features={"merge_queue": {"enabled": True, "mode": "prototype"}}
        )
        self.assertIsNone(self.bash(["arm", "7"]))
        with mock.patch.object(
            post, "_gh_write", side_effect=AssertionError("a GitHub write")
        ):
            code, out, _err = self.session_rk(["arm", "7"])
        self.assertEqual(self.sent[-1], ["arm", "7"])  # the writer's to run
        self.assertTrue(
            out.startswith("OFF: auto-merge is off, so nothing was done"), out
        )
        self.assertEqual(
            self.calls(self.log), [writer.atlassian_call([*self.SITE, *read]).argv]
        )  # the read alone: no create

    def test_in_prototype_and_in_standby_its_jira_writes_are_kept_until_they_go(self):
        """The session's writes are the monitor's own (outward.OWN_VAR). While ticket creation does not
        send, in prototype and then in standby, the writer files nothing: the follow-up ticket's write is kept as an
        unsent item each time (the journal names its type and what it was for, never its content), a write the writer
        would refuse in live too is refused and kept as nothing, and a read runs. Once standby ends, with the feature
        live, the next write goes, and no unsent item is sent."""
        create = [
            "jira",
            "issue",
            "create",
            "--project",
            PROJ,
            "--type",
            "Task",
            "--summary",
            "A follow-up",
        ]
        elsewhere = ["jira", "issue", "comment", "add", f"{ELSEWHERE}-2", "--body", "x"]
        read = ["--json", "jira", "issue", "list", "--jql", f"project = {PROJ}"]
        home = self.d / "home"
        with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
            for why, mode in (
                (state_store.PROTOTYPE, "prototype"),
                (state_store.STANDBY, "live"),
            ):
                self.cfg = dict(
                    self.cfg, features={"ticket_creation": dict(TICKETS_ON, mode=mode)}
                )
                state_store.set(home, "standby.on", why == state_store.STANDBY)
                held = features.held_words(self.cfg, "ticket_creation")
                self.assertIn(why, held)
                for words, said in (
                    (
                        create,
                        (
                            0,
                            f"{post.UNSENT}: a JIRA write for the JIRA project {PROJ}, kept as an unsent item: {held}\n",
                        ),
                    ),
                    (elsewhere, (4, f"REFUSED: a JIRA write is not sent: {held}\n")),
                ):
                    argv = [writer.ATLASSIAN, *self.SITE, *words]
                    with self.subTest(why=why, words=words[:4]):
                        self.assertIsNone(self.bash(argv))
                        self.assertEqual(self.session_rk(argv)[:2], said)
                self.assertEqual(self.jira(*read), {"issues": []})
            journal = self.store.journal_path.read_text()
            state_store.set(home, "standby.on", False)
            self.assertEqual(self.jira(*create)["key"], f"{PROJ}-1")
        content = shlex.join(writer.atlassian_call([*self.SITE, *create]).argv)
        self.assertEqual(
            [
                (i["type"], i["why"], i["content"])
                for i in state_store.unsent_items(self.store.dir)
            ],
            [
                (state_store.TICKET, state_store.PROTOTYPE, content),
                (state_store.TICKET, state_store.STANDBY, content),
            ],
        )  # kept, and not sent once standby ended
        for why in ("ticket creation in prototype", "the monitor in standby"):
            self.assertIn(
                f"ticket for the JIRA project {PROJ}: not sent ({why})", journal
            )
        self.assertNotIn("A follow-up", journal)
        self.assertEqual(
            self.calls(self.log),
            [writer.atlassian_call([*self.SITE, *read]).argv] * 2
            + [writer.atlassian_call([*self.SITE, *create]).argv],
        )
        self.assertEqual(self.marked, [True] * len(self.sent))


class TestTheBundleNamesTheWritersCommands(unittest.TestCase):
    """The text the session and its agents follow (agentic/: the skill's JIRA steps, the agents' and team members'
    ticket reads) gives each JIRA call as `<RK> atlassian`, never the bare CLI, which has no login where they run:
    with its placeholders filled in, a call the writer runs as written (writer.atlassian_call).
    """

    # md2adf's answer for a short description, as it stands inside the span's double quotes, where the shell takes a
    # command's output whole: its backslashes and quotes escaped
    ADF = (
        json.dumps(adf.convert("## A heading\n\nA line with `code`."))
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )
    # Each placeholder of the bundle's atlassian calls, in the order they are filled in, with an invented value
    FILLED = {
        "$(<RK> md2adf <the file>)": ADF,
        "<RK>": "rk",
        "<ATLASSIAN_SITE>": "example-site",
        "<JIRA_PROJECT>-NNNN": f"{PROJ}-1",
        "<JIRA_PROJECT>": PROJ,
        "KEY": f"{PROJ}-1",
        "EPIC": f"{PROJ}-2",
        "<the epic>": f"{PROJ}-2",
        "<words>": "flaky test",
        "<email or name>": "A Person",
        "<email or display name>": "A Person",
        "<title>": "A title",
        "<its title>": "a title",
        "<JIRA_ISSUE_TYPE>": "Task",
        "<its priority>": "Medium",
        "<JIRA_LABELS>": "a-label",
        "<JIRA_COMPONENT>": "A component",
        "<accountId>": "account-1",
        "<the PR's URL>": "https://example.com/pull/1",
        "<text>": "A comment",
    }

    def test_each_atlassian_command_of_the_bundle(self):
        found = [
            (f.relative_to(ROOT).as_posix(), span)
            for f in sorted((ROOT / "agentic").rglob("*.md"))
            for span in re.findall(
                r"`([^`]*\batlassian [-a-z][^`]*)`", f.read_text(encoding="utf-8")
            )
        ]
        self.assertGreaterEqual(len(found), 10, found)
        for where, span in found:
            with self.subTest(where=where, command=span[:70]):
                self.assertTrue(span.startswith("<RK> atlassian --site "), span)
                command = span
                for placeholder, value in self.FILLED.items():
                    command = re.sub(
                        rf"(?<!\w){re.escape(placeholder)}(?!\w)",
                        lambda _m: value,
                        command,
                    )
                self.assertNotRegex(command, r"<[^<>]+>")  # each placeholder filled in
                words = shlex.split(command)
                self.assertEqual(words[:2], ["rk", writer.ATLASSIAN])
                if words[-1] == "..." and not words[-2].startswith("-"):
                    # the family of commands that start with these words
                    cmd = words[2:-1]
                    while cmd[:1] in (["--site"], ["--json"]):
                        cmd = cmd[2:] if cmd[0] == "--site" else cmd[1:]
                    self.assertTrue(
                        any(
                            k[: len(cmd)] == tuple(cmd)
                            for k in writer.ATLASSIAN_COMMANDS
                        ),
                        cmd,
                    )
                    continue
                try:
                    writer.atlassian_call(words[2:])
                except ValueError as exc:
                    self.fail(f"the writer refuses {command!r}: {exc}")


class TestTheEntryPoint(Base):
    """docker/rk_write.py, as the writer (its user check passed by a stand-in for the password database): run by the
    writer's own rk (ENVIRON empty: the host's `rkmon secret`), or by sudo for the monitor's user (sudo_caller).
    """

    def run_(self, argv: list[str], environ=None) -> tuple[int, list, str]:
        """(its exit code, the rk it runs as [(argv, env, the broker's copies folder)], what it said)."""
        ran, out = [], io.StringIO()

        def child(args, env, cwd):
            """The broker's child: rkmon_lib/writer_run.py's checks in the broker's copies folder CWD, then rk."""
            try:
                with contextlib.chdir(cwd):
                    return writer_run.main(list(args[6:]), environ=env, execve=exec_)
            except Exec as done:
                ran.append((*done.args, cwd))
                return 0

        with self.entry_point(), contextlib.redirect_stdout(out):
            code = rk_write.main(
                argv, run=child, environ={} if environ is None else environ
            )
        return code, ran, out.getvalue()

    def test_run_as_another_user_it_refuses_before_anything_runs(self):
        """It runs as the writer only: as the monitor's user, the PR runners' or root, it exits 4 and never runs rk,
        whatever the command."""
        mine = rk_write.WRITE_USER
        for name in (
            container_layout.MONITOR.name,
            container_layout.PR_RUNNER.name,
            "root",
        ):
            other = mock.Mock(pw_name=name, pw_dir=str(self.d / "other-home"))
            ran, out = [], io.StringIO()
            with self.subTest(user=name):
                with self.entry_point(), mock.patch.object(
                    rk_write.pwd, "getpwuid", lambda uid, o=other: o
                ), contextlib.redirect_stdout(out):
                    code = rk_write.main(
                        writer.WRITE_PROBE,
                        run=lambda *a: ran.append(a),
                        environ=self.sudo_caller(),
                    )
                self.assertEqual(
                    (code, ran, out.getvalue()),
                    (
                        4,
                        [],
                        writer.refusal(
                            f"it runs as {mine} only (sudo -u {mine}), not {name}"
                        )
                        + "\n",
                    ),
                )

    def test_sudo_brings_no_secret_of_the_writer_s_and_no_command_makes_a_login(self):
        """sudo runs it for the monitor's user, which is the session's too: a secret the writer keeps is refused from
        it, with the host's command; and no command of the writer's makes a login, whoever asks.
        """
        for name in self.kept_by(sr.WRITER):
            for argv in (
                [writer.SECRET_SET, name],
                [writer.SECRET_SET, name, "--remove"],
            ):
                with self.subTest(argv=argv):
                    self.assertEqual(
                        self.run_(argv, self.sudo_caller()),
                        (
                            4,
                            [],
                            writer.refusal(
                                f"{name} is set from the host only: rkmon secret {name}"
                            )
                            + "\n",
                        ),
                    )
        logins = [[writer.SECRET_SET, n] for n in self.kept_by(sr.WRITER, login=True)]
        for environ in ({}, self.sudo_caller()):
            for argv in ([writer.ATLASSIAN, "auth", "login"], *logins):
                with self.subTest(argv=argv, sudo=bool(environ)):
                    code, ran, out = self.run_(argv, environ)
                    self.assertEqual((code, ran), (4, []))
                    self.assertTrue(out.startswith(REFUSAL_HEAD), out)
                    self.assertTrue(out.endswith(REFUSAL_TAIL + "\n"), out)

    def test_the_host_sets_one_as_the_writer_without_sudo(self):
        """`rkmon secret NAME` on the host: the writer's own rk, which the container's entry point gives to the broker
        without sudo, puts the value from stdin in the writer's volume."""
        code, ran, out = self.run_([writer.SECRET_SET, "teams_webhook"])
        self.assertEqual(code, 0, out)
        argv, env, _cwd = ran[0]
        with mock.patch.dict(os.environ, {gh.BROKERED: env[gh.BROKERED]}):
            code, out = self.rk(argv[4:], stdin=HOOK + "\n")
        self.assertEqual(code, 0, out)
        self.assertEqual(
            (self.write_secrets / "teams_webhook").read_text(encoding="utf-8"),
            HOOK + "\n",
        )
        self.assertNotIn(HOOK, out)

    def test_the_host_installs_the_request_service_as_the_writer_without_sudo(self):
        """`rkmon web-host install` on the host: the writer's own rk, which the container's entry point gives to the
        broker without sudo, runs `rk web-host-install` as given. A word after it is refused, whoever asks: the web
        host, its login and its paths come from the settings; and through sudo, for the monitor's user, it is refused
        with the host's command. Nothing runs then."""
        code, ran, out = self.run_([writer.WEB_HOST_INSTALL])
        self.assertEqual(code, 0, out)
        ((argv, env, _cwd),) = ran
        self.assertEqual(argv[4:], [writer.WEB_HOST_INSTALL])
        self.assertNotIn(writer.BROKER_VAR, env)
        words = (
            "it runs `rk web-host-install` only: it reads the web host, its login and its paths from the settings it "
            "runs on"
        )
        for environ in ({}, self.sudo_caller()):
            with self.subTest(sudo=bool(environ)):
                self.assertEqual(
                    self.run_([writer.WEB_HOST_INSTALL, "web2.example.com"], environ),
                    (4, [], writer.refusal(words) + "\n"),
                )
        self.assertEqual(
            self.run_([writer.WEB_HOST_INSTALL], self.sudo_caller()),
            (
                4,
                [],
                writer.refusal(
                    "the request service is installed on the web host from the host only: rkmon web-host install"
                )
                + "\n",
            ),
        )

    def test_a_secret_the_monitor_keeps_is_refused_and_one_the_writer_keeps_runs(self):
        code, ran, out = self.run_([writer.SECRET_SET, "claude_env"])
        self.assertEqual((code, ran), (4, []))
        self.assertEqual(
            out, writer.refusal("claude_env is not a secret the writer keeps") + "\n"
        )
        (self.d / "teams_webhook").write_text("a file of that name, never read\n")
        os.chdir(self.d)
        self.addCleanup(os.chdir, ROOT)
        code, ran, out = self.run_([writer.SECRET_SET, "teams_webhook"])
        self.assertEqual(code, 0, out)
        argv, env, _cwd = ran[0]
        self.assertEqual(
            argv[-2:], [writer.SECRET_SET, "teams_webhook"]
        )  # as given: no file argument
        self.assertEqual(env["RKMON_GH_BROKER"], "1")
        self.assertNotIn(writer.BROKER_VAR, env)

    def test_the_atlassian_cli_s_words_are_checked_before_anything_runs(self):
        key = f"{PROJ}-1"
        code, ran, out = self.run_([writer.ATLASSIAN, "jira", "issue", "delete", key])
        self.assertEqual((code, ran), (4, []))
        self.assertEqual(
            out,
            writer.refusal(
                "`atlassian jira issue delete` is not one of the atlassian commands the writer runs"
            )
            + "\n",
        )
        code, ran, out = self.run_([writer.ATLASSIAN, "jira", "issue", "get", key])
        self.assertEqual(code, 0, out)
        self.assertEqual(
            ran[0][0][-5:], [writer.ATLASSIAN, "jira", "issue", "get", key]
        )  # as given: its words are never files

    def test_no_file_argument_names_a_secret_of_the_writer_s(self):
        """Reading any of the writer's secrets is refused: a file argument naming one (or a file in a login of the
        writer's) is never passed to rk, whichever command takes file arguments."""
        self.write_secrets.mkdir()
        files = []
        for name in self.kept_by(sr.WRITER):
            f = self.write_secrets / name
            f.write_text(a_value(sr.entry(self.cfg, name).shape, 3))
            files.append(f)
        for name in self.kept_by(sr.WRITER, login=True):
            folder = self.write_secrets / name
            folder.mkdir()
            for part in sr.entry(self.cfg, name).files:
                (folder / part).write_text("a login's file\n")
                files.append(folder / part)
        caller = {"SUDO_UID": str(os.getuid()), "SUDO_GID": str(os.getgid())}
        copies = self.d / "copies"
        copies.mkdir()
        for cmd in writer.FILE_ARGS:
            for f in files:
                with self.subTest(cmd=cmd, file=f.name), self.assertRaises(ValueError):
                    writer_run.guarded_args(
                        [cmd, "7", str(f)],
                        [str(self.write_secrets)],
                        caller,
                        str(copies),
                    )
        self.assertEqual(list(copies.iterdir()), [])

    def test_the_writer_s_rk_reads_the_confirmed_settings_the_monitor_s_rk_reads(self):
        """In a one-off container of the logins' walk (deploy step 5), or of `rkmon web-host install`, the root phase
        writes your confirmed settings where rk reads them (rkmon_lib/runtime_config.py handed_layer). The writer's rk,
        run by its own rk (the host's `rkmon secret NAME` and `rkmon web-host install`) or through sudo for the
        monitor's user (the monitor's rk logins asking it), gets no variable that holds them, whatever its caller's
        environment holds, and in the environment the broker gives it reads the same settings as the monitor's rk.
        """
        import rk_root
        from rkmon_lib import doctor, file_types, image, paths, runtime_config as RC
        from test_runtime_config import set_in

        handed, home = self.d / "handed", self.d / "monitor-home"
        home.mkdir()
        settings = {"repository": {"name": "acme/gadgets"}}
        layer = {
            "kind": RC.LAYER_KIND,
            "schema_version": RC.LAYER_SCHEMA_VERSION,
            "deploy_schema_version": file_types.DEPLOY.version,
            "package_version": image.package_version(paths.PACKAGE),
            "confirmed_at": "2026-01-02T03:04:05Z",
            "fingerprint": "sha256:" + "0" * 64,
            "settings": settings,
            "set_in_file": set_in(settings),
            "container": {},
        }
        rk_root.hand_layer(
            json.dumps(layer).encode(),
            str(handed / RC.LAYER_FILE),
            (os.getuid(), os.getgid()),
        )
        variable = container_layout.HANDED_LAYER_VARIABLE
        with contextlib.ExitStack() as stack:
            for p in (
                mock.patch.object(RC, "HANDED_DIR", str(handed)),
                mock.patch.object(
                    RC, "ROOT_UID", os.getuid()
                ),  # the test's user stands for root
                mock.patch.object(
                    RC, "SETTINGS_DIR", str(self.d / "settings")
                ),  # none delivered
                mock.patch.object(container_layout, "IMAGE_VENV", str(self.d / "venv")),
            ):
                stack.enter_context(p)
            with mock.patch.dict(os.environ, {"RKMON_HOME": str(home)}):
                monitor = RC.load()
            self.assertEqual(monitor.layer, layer)
            for argv, caller in (
                ([writer.SECRET_SET, "teams_webhook"], {}),
                # the host's `rkmon web-host install`: the web host, its login and its paths as the settings name them
                ([writer.WEB_HOST_INSTALL], {}),
                (list(writer.SECRETS_PROBE), self.sudo_caller()),
            ):
                with self.subTest(argv=argv, sudo=bool(caller)):
                    code, ran, out = self.run_(argv, dict(caller, **{variable: "{}"}))
                    self.assertEqual(code, 0, out)
                    ((_, env, _copies),) = ran
                    self.assertNotIn(variable, env)
                    # the monitor's home, which this machine has not: the test's
                    self.assertEqual(env["RKMON_HOME"], container_layout.MONITOR_HOME)
                    with mock.patch.dict(
                        os.environ, dict(env, RKMON_HOME=str(home)), clear=True
                    ):
                        self.assertTrue(doctor.deployment_image())
                        theirs = RC.load()
                    self.assertEqual(theirs.layer, monitor.layer)
                    self.assertEqual(theirs.values, monitor.values)


if __name__ == "__main__":
    unittest.main()
