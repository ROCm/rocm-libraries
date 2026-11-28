# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The writer: the container's user that makes the outward writes and alone reads the secrets they need, and the rk
commands it runs.

In the container the writer's secrets (rkmon_lib/secrets_registry.py: each entry whose keeper is the writer) live in
its own volume, container_layout.WRITE_SECRETS_DIR, which no other user reads. The monitor, the Claude session, its
agents, the jobs and pull requests' code reach them only through the commands below. One sudo rule lets the monitor's
user run docker/rk_write.py as the writer, and nothing else; it takes one of these commands and runs it with code no
user can change, which it checks first, and that code's writer (rkmon_lib/writer_run.py) checks its file arguments and
the configuration's folders before it runs it. On a host the one user is both: nothing is routed.

- The writer's commands are the rows of rkmon_lib/outward.py's table whose `writer` column is set, with what gates
  each one's send (its `gate` column), so that a send that reaches the writer any other way meets the same
  gate:
  - rk's GitHub write commands. None approves a review, merges at once or dismisses one. `rk spec-post` and
    `rk post --premises` are not among them: their premises read git and their AFTER lines run commands, as the
    monitor; their post itself is a plain `rk post` (premises.post_checked).
  - TICKET_FILE: `rk ticket-file ID [--again]`, a found issue's proposed ticket filed in JIRA, whole (tickets.approve
    sends the admin dashboard's Approve here).
  - TEAMS_SEND: `rk teams-send --type TYPE`, one message to the team chat channel, the message on stdin
    (post.teams_post sends each here). TEAMS_TEST: `rk teams-test`, the test card, whole.
  - WEB_PUSH: `rk web-push FILE`, the team dashboard's page pushed to the web host (publish.html). WEB_REQUESTS:
    `rk web-requests`, the web host's priority requests file read and printed (publish.pull_requests).
  - ATLASSIAN: `rk atlassian [--site SITE] [--json] [--timeout-s N] WORDS...`, one of the atlassian CLI's listed
    commands (ATLASSIAN_COMMANDS) with its login: the checks' reads, the project's create metadata, and the Claude
    session's ticket reads and follow-up tickets (atlassian._run sends each call here).
  - WRITE_PROBE: the GitHub write login's probe (`rk gh-logins --side write --json`), which makes nothing.
  - `rk writer-secrets [--json]`: whether each of the writer's secrets is set, where, and when it last changed
    (logins.writer_secrets), never a value. SECRETS_PROBE is its JSON form, which `kept` reads.
  - `rk secret-set NAME [--remove]` for a secret the writer keeps (a login aside: `rkmon login NAME`): the value on
    stdin into its file in the writer's volume, or that file removed.
  - WEB_HOST_INSTALL: `rk web-host-install`, the request service installed on the team dashboard's web host
    (rkmon_lib/web_host_install.py install), over the web host's key. Its words name no host, login or path: the
    writer reads them from the settings it runs on.
- COMMANDS: each of them, by its name. FROM_HOST: those only the host asks for, as the writer itself (`rkmon secret
  NAME` and `rkmon web-host install` run them through the container's entry point), never through sudo: the monitor's
  user is the Claude session's too. FILE_ARGS: those whose arguments rkmon_lib/writer_run.py reads as file names. No
  command here makes a login: `rkmon login NAME` makes each, from the host, as the writer.
- `atlassian_call(words)`: `rk atlassian WORDS` as the writer runs it (a Call): the CLI's words (the global options
  first, then the command's words, its argument and each option as one `--option=value` word), the listed command,
  whether it writes to JIRA, the seconds it may run (`--timeout-s`, rk's own option before the CLI's words, 1 to
  ATLASSIAN_TIMEOUT_MAX_S; ATLASSIAN_TIMEOUT_S when not given), and the site, arguments and options it names.
  ValueError, in words, for anything off the list: a command, an argument or an option it does not list, an option
  given twice (REPEATED aside: one field each), a value out of its form, a value that starts with @ (a CLI may read the
  file named after an @, and the writer reads its own secrets).
- `held(call, site, project, epic="")`: why the writer does not make CALL's write to JIRA, in words: a write goes to
  the configured site and project alone (`atlassian.site`, `jira.project`; atlassian.listed passes them). It names
  that site with --site, and the project it files in, the ticket it changes and the epic it links that ticket to are
  the project's (the epic may also be EPIC, the configured default one). "" when it makes it, and for a read.
- `with_premises(argv)`: whether rk ARGV is `rk post ... --premises SPEC` (argparse takes an abbreviation too).
- `refused(argv)`: why the writer does not run rk ARGV, in words; "" when it is one of the commands above, as listed
  (its words are checked here where they decide what runs; rk's own parser checks the rest). `refusal(why)`: the one
  line the writer's commands are refused with (exit 4), by the broker (docker/rk_write.py, its copy REFUSAL), by the
  writer's run (rkmon_lib/writer_run.py), by rk's half of a command
  (rkmon.py cmd_writer_half), and by the monitor's rk when sudo cannot start the writer.
- `sudo_refusal(argv, environ)`: why the writer does not run rk ARGV for a caller sudo let in (SUDO_VAR is set: the
  monitor's user, and so the session): a FROM_HOST command (`from_host`); "" otherwise.
- `from_host(argv)`: the words that say rk ARGV, a FROM_HOST command, is asked for from the host only, with the
  host's command: `set_from_host` for `rk secret-set NAME`. `set_from_host(name)`: the words that say NAME, a secret
  the writer keeps, is set from the host only, with the command that sets it.
- `routes(argv)`: whether the monitor sends rk ARGV to the writer: one of the commands above, by its name and, for
  `gh-logins`, its words; never a FROM_HOST command. The writer refuses one that is not as listed.
- `routing(environ=None)`: whether this process sends the writer's commands to the writer: in the container, whose
  image names the writer's entry point (BROKER_VAR), but not in the writer's own rk (gh.BROKERED).
- `routed(argv, environ=None, own=None)`: the command that runs rk ARGV as the writer, through sudo; None when this
  process does not send it there (`routing`, `routes`). A write of the monitor's own (OWN: outward.own by default)
  carries the word outward.OWN_WORD before ARGV, which the writer's run takes off (rkmon_lib/writer_run.py).
- `through(argv, input=None, timeout_s=None, own=None)`: runs rk ARGV as the writer: its CompletedProcess (text, stdout and
  stderr captured, INPUT on stdin). A writer that could not be started gives NOT_RUN (126) and the reason as stderr;
  one that gave no answer within timeout_s seconds gives NO_ANSWER (124) and says that it may still be running (the
  end of sudo does not end the writer's rk). ValueError when this process does not send ARGV to the writer. START_S:
  what sudo and the writer's start add to a command's own time limit.
- `said(p)`: the writer's answer in one line: the last line it printed (on stderr when it printed nothing else),
  without the REFUSED: before it.
- `kept()`: the writer's answer to SECRETS_PROBE: {name: {"set", "where", "mark"}}; {} when it gave none. A process
  keeps an answer for KEPT_S seconds; `forget()` drops it (after a secret is set).

It loads with the standard library, rkmon_lib/gh.py and rkmon_lib/outward.py only: the writer's run
(rkmon_lib/writer_run.py) imports it before it checks anything. Each function imports the other modules it reads.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import NamedTuple

from . import gh, outward
from .outward import (  # noqa: F401  (the writer's commands, by the names rk's code uses)
    ATLASSIAN,
    GH_LOGINS,
    SECRET_SET,
    TEAMS_SEND,
    TEAMS_TEST,
    TICKET_FILE,
    WEB_HOST_INSTALL,
    WEB_PUSH,
    WEB_REQUESTS,
    WRITER_SECRETS,
)

# The image's writer entry point (docker/Dockerfile sets it): the monitor sends the writer's commands through it.
BROKER_VAR = "RKMON_WRITE_BROKER"
WRITE_PROBE = [GH_LOGINS, "--side", "write", "--json"]
SECRETS_PROBE = [WRITER_SECRETS, "--json"]
COMMANDS = tuple(r.name for r in outward.TABLE if r.writer)
FROM_HOST = outward.names(writer=outward.HOST)
FILE_ARGS = outward.names(file_args=True)
# sudo sets it for the command it runs, and nothing else unsets it
SUDO_VAR = "SUDO_USER"
KEPT_S = 30  # how long a process keeps the writer's answer to SECRETS_PROBE
TIMEOUT_S = 60  # the most the monitor waits for the writer's probes and `rk secret-set`
START_S = 15  # what sudo and the writer's start add to a command's own time limit
# `through`'s exit codes: the writer could not be started; it gave no answer in time
NOT_RUN, NO_ANSWER = 126, 124
_KEPT: dict = {}

# --- the atlassian CLI's listed commands ------------------------------------------------------------------------------

# rk atlassian's own option: the most seconds the CLI runs.
TIMEOUT_OPTION = "--timeout-s"
ATLASSIAN_TIMEOUT_S, ATLASSIAN_TIMEOUT_MAX_S = 90, 600
SITE = r"[A-Za-z0-9][A-Za-z0-9.-]*"
_KEY = r"[A-Z][A-Z0-9_]*-[0-9]+"
_PROJECT = r"[A-Z][A-Z0-9_]*"
_NUMBER = r"[0-9]{1,4}"
_ACCOUNT = r"[A-Za-z0-9:_-]{1,128}"
_TEXT = r"(?!@).*"  # any text that does not start with @
_WHO = r"(?![-@]).+"  # an address or a name
_FIELD = r"[A-Za-z0-9_.-]+=(?!@).*"  # NAME=VALUE
_URL = r"https://\S+"
_CREATEMETA = (
    r"/rest/api/3/issue/createmeta/[A-Za-z0-9_]+/issuetypes(?:/[0-9]+)?"
    r"(?:\?(?:startAt|maxResults)=[0-9]{1,4}(?:&(?:startAt|maxResults)=[0-9]{1,4})?)?"
)
_TICKET_FIELDS = {
    "--summary": _TEXT,
    "--priority": _TEXT,
    "--labels": _TEXT,
    "--field": _FIELD,
    "--field-json": _FIELD,
}
# The options a call may give more than once: one field each.
REPEATED = ("--field", "--field-json")


class Listed(NamedTuple):
    """One of the atlassian CLI's commands the writer runs: the form of each argument it takes, in order, the form of
    each option's value (a regular expression the whole value matches), whether it writes to JIRA, and the options it
    must be given."""

    args: tuple = ()
    options: dict = {}
    writes: bool = False
    required: tuple = ()


# By the command's words. Two global options may come anywhere: --site SITE and --json.
ATLASSIAN_COMMANDS = {
    # The checks' reads: the CLI, its login, its sites and the project.
    ("version",): Listed(),
    ("auth", "status"): Listed(),
    ("site", "list"): Listed(),
    ("site", "active"): Listed(),
    ("jira", "project", "versions"): Listed((_PROJECT,)),
    # The project's create metadata (check 6): its issue types, then the fields of one.
    ("rest",): Listed(
        options={"--product": "jira", "--method": "GET", "--path": _CREATEMETA},
        required=("--product", "--method", "--path"),
    ),
    # The tickets: read one, search, a person's account; file one, link it to its epic, edit it, comment on it.
    ("jira", "issue", "get"): Listed((_KEY,)),
    ("jira", "issue", "list"): Listed(
        options={"--jql": _TEXT, "--max-results": _NUMBER}
    ),
    ("jira", "issue", "search"): Listed(
        options={"--jql": _TEXT, "--max-results": _NUMBER}
    ),
    ("jira", "user", "get"): Listed((_WHO,)),
    ("jira", "issue", "create"): Listed(
        options={"--project": _PROJECT, "--type": _TEXT, **_TICKET_FIELDS},
        writes=True,
        required=("--project",),
    ),
    ("jira", "issue", "link-epic"): Listed(
        (_KEY,), {"--epic": _KEY}, writes=True, required=("--epic",)
    ),
    ("jira", "issue", "update"): Listed(
        (_KEY,), {"--assignee": _ACCOUNT, **_TICKET_FIELDS}, writes=True
    ),
    ("jira", "issue", "comment", "add"): Listed(
        (_KEY,), {"--body": _TEXT}, writes=True, required=("--body",)
    ),
    ("jira", "issue", "remote-link"): Listed(
        (_KEY,), {"--url": _URL, "--title": _TEXT}, writes=True, required=("--url",)
    ),
}


class Call(NamedTuple):
    """`rk atlassian WORDS` as the writer runs it (atlassian_call)."""

    argv: list  # the CLI's words, without the CLI itself
    command: tuple  # the listed command's words
    writes: bool
    timeout_s: int
    site: str = ""  # --site's value; "" when not given (the CLI's active site)
    args: tuple = ()  # the command's arguments
    options: tuple = ()  # each option as (its name, its value), in order


def _form(value: str, pattern: str) -> bool:
    return re.fullmatch(pattern, value, re.S) is not None


def _twice(name: str) -> ValueError:
    return ValueError(
        f"{name} is given twice: only {' and '.join(REPEATED)} may repeat"
    )


def _option(words: list[str], i: int) -> tuple[str, str, int]:
    """The option at words[i] and its value, `--name=value` or `--name value` (the next word, whatever it is, as the
    CLI takes it), and the index after them."""
    name, eq, value = words[i].partition("=")
    if eq:
        return name, value, i + 1
    if i + 1 >= len(words):
        raise ValueError(f"{name} needs a value")
    return name, words[i + 1], i + 2


def _timeout(value: str) -> int:
    if not value.isdigit() or not 0 < int(value) <= ATLASSIAN_TIMEOUT_MAX_S:
        raise ValueError(
            f"{TIMEOUT_OPTION} takes a number of seconds, 1 to {ATLASSIAN_TIMEOUT_MAX_S}"
        )
    return int(value)


def atlassian_call(words: list[str]) -> Call:
    """`rk atlassian WORDS` as the writer runs it (the module's docstring)."""
    timeout_s, rest, i, timed = ATLASSIAN_TIMEOUT_S, [], 0, False
    # rk's own options, before the CLI's words
    while i < len(words) and words[i].startswith("-"):
        if words[i] == "--":
            i += 1
            break
        name = words[i].partition("=")[0]
        if name == TIMEOUT_OPTION:
            if timed:
                raise _twice(name)
            _n, value, i = _option(words, i)
            timeout_s, timed = _timeout(value), True
            continue
        if name not in ("--site", "--json"):
            raise ValueError(f"{name} is not an option the writer passes")
        width = 2 if name == "--site" and "=" not in words[i] else 1
        rest += words[i : i + width]
        i += width
    rest += words[i:]
    site, as_json, cmd, args, opts, listed, i = None, False, [], [], [], None, 0
    while i < len(rest):
        w = rest[i]
        if w == "--json":
            if as_json:
                raise _twice(w)
            as_json, i = True, i + 1
            continue
        if w.partition("=")[0] == "--site":
            if site is not None:
                raise _twice("--site")
            _n, site, i = _option(rest, i)
            if not _form(site, SITE):
                raise ValueError(f"--site {site!r} is not a site's name")
            continue
        said = f"`atlassian {' '.join(cmd)}`"
        if w.startswith("-"):
            name = w.partition("=")[0]
            if listed is None or name not in listed.options:
                raise ValueError(
                    f"{name} is not an option the writer passes"
                    + (f" to {said}" if listed else "")
                )
            if name not in REPEATED and any(n == name for n, _v in opts):
                raise _twice(name)
            name, value, i = _option(rest, i)
            if not _form(value, listed.options[name]):
                raise ValueError(
                    f"{name}'s value is not one the writer passes to {said}"
                    + (
                        " (a value never starts with @)"
                        if value.startswith("@")
                        else ""
                    )
                )
            opts.append((name, value))
            continue
        if listed is None:
            cmd.append(w)
            listed = ATLASSIAN_COMMANDS.get(tuple(cmd))
            if listed is None and not any(
                k[: len(cmd)] == tuple(cmd) for k in ATLASSIAN_COMMANDS
            ):
                raise ValueError(
                    f"`atlassian {' '.join(cmd)}` is not one of the atlassian commands the writer runs"
                )
        elif len(args) < len(listed.args) and _form(w, listed.args[len(args)]):
            args.append(w)
        else:
            raise ValueError(f"{w!r} is not an argument the writer passes to {said}")
        i += 1
    if listed is None:
        raise ValueError(
            f"`atlassian {' '.join(cmd)}` is not a whole command"
            if cmd
            else "no atlassian command"
        )
    said = f"`atlassian {' '.join(cmd)}`"
    if len(args) < len(listed.args):
        raise ValueError(f"{said} needs {len(listed.args)} argument(s)")
    given = {n for n, _v in opts}
    lacking = [o for o in listed.required if o not in given]
    if lacking:
        raise ValueError(f"{said} needs {', '.join(lacking)}")
    front = ([f"--site={site}"] if site else []) + (["--json"] if as_json else [])
    return Call(
        front + cmd + args + [f"{n}={v}" for n, v in opts],
        tuple(cmd),
        listed.writes,
        timeout_s,
        site or "",
        tuple(args),
        tuple(opts),
    )


def held(call: Call, site: str, project: str, epic: str = "") -> str:
    """Why the writer does not make CALL's write to JIRA, in words (the module's docstring); "" when it makes it, and
    for a read."""
    if not call.writes:
        return ""
    from . import deploy_tree  # only the refusals read it

    if not site or call.site != site:
        return (
            f"a JIRA write goes to the configured site alone: --site {site}"
            if site
            else "a JIRA write goes to the configured site alone, and "
            f"{deploy_tree.name('atlassian.site')} is not set"
        )
    if not project:
        return (
            "a JIRA write goes to the configured project alone, and "
            f"{deploy_tree.name('jira.project')} is not set"
        )
    keys = [a for a in call.args if _form(a, _KEY)] + [
        v for n, v in call.options if n == "--epic" and v != epic
    ]
    outside = [v for n, v in call.options if n == "--project" and v != project] + [
        k for k in keys if k.rpartition("-")[0] != project
    ]
    if outside:
        return f"a JIRA write goes to the configured project's tickets alone ({project}), not {outside[0]}"
    return ""


# --- the commands -----------------------------------------------------------------------------------------------------


def with_premises(argv: list[str]) -> bool:
    """Whether rk `argv` is `rk post ... --premises SPEC` (argparse takes an abbreviation of the option too)."""
    return argv[:1] == ["post"] and any(
        len(a.split("=", 1)[0]) > 3 and "--premises".startswith(a.split("=", 1)[0])
        for a in argv[1:]
    )


def _writers_secret(name: str) -> bool:
    """Whether NAME is a secret the writer keeps (a login aside)."""
    from . import secrets_registry as sr

    entry = sr.entry({}, name)
    return bool(entry and entry.keeper == sr.WRITER and entry.shape != sr.LOGIN)


def _secret_set(words: list[str]) -> str:
    """Why the writer does not run `rk secret-set WORDS` ("" when it does)."""
    names = [w for w in words if not w.startswith("-")]
    flags = [w for w in words if w.startswith("-")]
    if len(names) != 1 or flags not in ([], ["--remove"]):
        return "it runs `rk secret-set NAME [--remove]` only"
    if not _writers_secret(names[0]):
        return f"{names[0]} is not a secret the writer keeps"
    return ""


def refusal(why: str) -> str:
    """The one line the writer's commands are refused with (exit 4), wherever they are refused: WHY, and that nothing
    ran (a read is refused in the same words as a write)."""
    return f"REFUSED: the writer: {why}; nothing was run"


def refused(argv: list[str]) -> str:
    """Why the writer does not run rk ARGV, in words; "" when it does (the module's docstring)."""
    if not argv:
        return "no command"
    cmd, words = argv[0], argv[1:]
    if with_premises(argv):
        return "rk post --premises runs as the monitor: its premises read git and run commands"
    if cmd == ATLASSIAN:
        try:
            atlassian_call(words)
        except ValueError as exc:
            return str(exc)
        return ""
    if cmd == GH_LOGINS:
        return (
            "" if argv == WRITE_PROBE else f"it runs `rk {' '.join(WRITE_PROBE)}` only"
        )
    if cmd == WRITER_SECRETS:
        return (
            ""
            if words in ([], SECRETS_PROBE[1:])
            else "it runs `rk writer-secrets [--json]` only"
        )
    if cmd == SECRET_SET:
        return _secret_set(words)
    if cmd == WEB_HOST_INSTALL and words:
        return (
            f"it runs `rk {WEB_HOST_INSTALL}` only: it reads the web host, its login and its paths from the settings "
            "it runs on"
        )
    if cmd in COMMANDS:
        return ""
    return f"rk {cmd} is not one of the writer's commands"


def from_host(argv: list[str]) -> str:
    """rk ARGV, a FROM_HOST command, is asked for from the host only: the words, with the host's command."""
    if argv and argv[0] == WEB_HOST_INSTALL:
        from .web_host_install import wrapper_command

        return f"the request service is installed on the web host from the host only: {wrapper_command()}"
    names = [w for w in argv[1:] if not w.startswith("-")]
    return set_from_host(names[0] if names else "NAME")


def set_from_host(name: str) -> str:
    """NAME, a secret the writer keeps, is set from the host only: the words, with the command that sets it."""
    from . import checks, secrets_registry as sr
    from .doctor import wrapper_name

    entry = sr.entry({}, name)
    words = sr.set_command(entry) if entry else ("secret", name)
    return (
        f"{name} is set from the host only: {checks.Command((wrapper_name(), *words))}"
    )


def sudo_refusal(argv: list[str], environ) -> str:
    """Why the writer does not run rk ARGV for a caller sudo let in (the module's docstring); "" when it does."""
    if not argv or argv[0] not in FROM_HOST or not environ.get(SUDO_VAR):
        return ""
    return from_host(argv)


def routes(argv: list[str]) -> bool:
    """Whether the monitor sends rk ARGV to the writer (the module's docstring)."""
    if not argv or argv[0] not in COMMANDS or argv[0] in FROM_HOST:
        return False
    if argv[0] == GH_LOGINS:
        return argv == WRITE_PROBE
    return not with_premises(argv)


def routing(environ=None) -> bool:
    """Whether this process sends the writer's commands to the writer (the module's docstring)."""
    e = os.environ if environ is None else environ
    return bool(e.get(BROKER_VAR, "")) and e.get(gh.BROKERED) != "1"


def routed(argv: list[str], environ=None, own=None) -> list[str] | None:
    """The command that runs rk ARGV as the writer, through sudo; None when this process does not send it there. With
    the word that marks a write of the monitor's own when it is one (OWN, else this process's mark: outward.own).
    """
    from . import container_layout

    e = os.environ if environ is None else environ
    if not routing(e) or not routes(argv):
        return None
    mark = [outward.OWN_WORD] if (outward.own(e) if own is None else own) else []
    return [
        "sudo",
        "-n",
        "-u",
        container_layout.WRITER.name,
        e[BROKER_VAR],
        *mark,
        *argv,
    ]


def through(
    argv: list[str],
    input: str | None = None,
    timeout_s: float | None = None,
    own: bool | None = None,
) -> subprocess.CompletedProcess:
    """Run rk ARGV as the writer (the module's docstring)."""
    cmd = routed(argv, own=own)
    if not cmd:
        raise ValueError(f"rk {' '.join(argv[:1])} is not sent to the writer here")
    try:
        return subprocess.run(
            cmd,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=None if input is not None else subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, NO_ANSWER, "", f"no answer in {timeout_s:g} s; it may still be running"
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            cmd, NOT_RUN, "", f"the writer did not run ({type(exc).__name__}: {exc})"
        )


def said(p: subprocess.CompletedProcess) -> str:
    """The writer's answer in one line (the module's docstring)."""
    for text in (p.stdout, p.stderr):
        lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
        if lines:
            return lines[-1].removeprefix("REFUSED: ")
    return f"the writer answered nothing (exit {p.returncode})"


def kept() -> dict:
    """The writer's answer to SECRETS_PROBE (the module's docstring)."""
    now = time.monotonic()
    if _KEPT and now - _KEPT["at"] < KEPT_S:
        return _KEPT["answer"]
    p = through(SECRETS_PROBE, timeout_s=TIMEOUT_S)
    try:
        answer = json.loads(p.stdout) if p.returncode == 0 else {}
    except ValueError:
        answer = {}
    answer = answer if isinstance(answer, dict) else {}
    _KEPT.update(at=now, answer=answer)
    return answer


def forget() -> None:
    """Drop the kept answer to SECRETS_PROBE: the next `kept` asks the writer again."""
    _KEPT.clear()
