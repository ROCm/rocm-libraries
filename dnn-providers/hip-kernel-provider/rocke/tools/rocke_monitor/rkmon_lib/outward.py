# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rk commands that send outward, the writer's commands that send nothing, and the commands the Claude session never
runs: one table, the one home of where each command sends, what gates it, how it runs and who may run it. Every layer
that decides about a send reads it: the stop (rkmon_lib/control.py), a mirror's refusal
(rkmon_lib/mirror.py), the feature gates, the write lock and the dispatch (rkmon.py), the team chat's message types
(rkmon_lib/post.py), the writer's commands, routing and file checks (rkmon_lib/writer.py, docker/rk_write.py), the
session's guard (rkmon_lib/orchestrator.py) and a review's AFTER lines (rkmon_lib/premises.py).

- `Command`: one row. Its columns:
  - name: the rk command.
  - dest: where it sends: GITHUB, TEAMS (the team chat channel), WEB (the web host), JIRA, or NONE (it reads, probes,
    puts a secret in, or readies the web host for live: setup you run yourself, which counts as no send).
  - gate: OWN, it runs whole in one process, whose own code asks the gates before each send (rkmon.py feature_gate,
    post.outward_gate, tickets.approve_refusal); GATED, a send the writer makes for another process, which rk gates
    before the send, whoever asked for it (rkmon.py write_gate); NONE.
  - features: the features it sends for (rkmon_lib/features.py). With two, it runs while either is on (`rk teams-test`:
    its card checks the channel both alerts send to), or for the one an issue command's arguments pick (ISSUES: the
    issue's type). Empty where another table picks them: `rk notice` by its --kind (post.NOTICE_FEATURE),
    `rk teams-send` by its --type (the rows of the message column).
  - protective: it only takes auto-merge off.
  - writer: how it reaches the writer: SUDO, the monitor sends it there through sudo (writer.routes); HOST, the host
    alone asks for it, as the writer itself, never through sudo; "" when the monitor runs it itself.
  - file_args: docker/rk_write.py reads each argument that names a file as the caller may, and gives rk a checked copy.
  - handler: the rkmon.py handler it shares: HALF, cmd_writer_half (the writer's half of a send or read whose secret
    the writer alone reads; on a host, the send itself); POST_LOCKED, one of post.py's writes under the state lock;
    "" when it has a handler of its own.
  - dry_run: with --dry-run it sends nothing and is no outward write: no write lock, and neither a mirror nor a
    stop refuses it. (`rk post --dry-run` runs rk post's guards under the write lock, as a post does.)
  - drafts: it keeps a review rather than being refused: a mirror diverts it to its folder, and with review
    posting off the review is kept unposted.
  - message: the type of team chat message it sends (`rk teams-send --type`).
  - what: what a GATED command sends, in the outward gate's words.
  - session: whether the Claude session may run it. The session's guard refuses the rest, the commands only the operator
    or rk itself runs: a proposed ticket filed or declined, the Teams test card, a writer's secret set, the read login
    set, a host's own keys read or recorded, the request service installed on the web host, and the end of standby once
    the monitor moved from has stopped (the operator's; standby ends by rkmon standby off); a raw team chat
    message, the team dashboard's push and the web host's requests read (rk's own: the session's message to the team
    chat is `rk teams-breakage N`, with its cooldown and its ledger line). A review's AFTER lines, which the session
    writes, never run them either, live or kept. The session's guard is one against mistakes: it reads the shell line
    (not_the_sessions), and inline Python gets around that. What keeps a write out is the writer's gate and the write
    login, which only the writer has. An in-depth review's approval (rk review-approve) is the operator's, never the
    session's.
  - `stops`: whether a stop refuses it: it sends, and is not protective. A stop refuses an `rk atlassian` only when it
    writes to JIRA (rkmon.py write_gate).
- TABLE: every row, the GitHub writes first. The commands that rk's code names have their names here (TICKET_FILE,
  TEAMS_SEND, ...).
- `row(name)`: the row of rk NAME, or None. `names(**columns)`: the names of the rows whose columns have these values,
  in the table's order.
- `not_the_sessions(line) -> str`: the command of a row the session may not run that the shell line LINE runs
  through rk, or "": rk, rkmon.py or the writer's entry point (docker/rk_write.py, as sudo runs it), by a path or not,
  then anything up to the command's name (rk's own options: --home DIR). RK_LINE and NAME_END are its pattern's two
  parts. The session's guard reads it, and `finish` for a review's AFTER lines.
- `not_the_sessions_why(name, login="") -> str`: why the session may not run rk NAME, in the words the session's guard
  and a review's AFTER lines give: it is for LOGIN to run (the operator's GitHub login, config.operator_login, which the
  skill files name; "a person" when it is blank) or for rk itself.
- `own(environ=None)`: whether this process's writes are the monitor's own, not ones a person typed: its
  environment carries OWN_VAR at "1". The monitor's processes that run rk's write commands carry it: the Claude session
  (rkmon_lib/orchestrator.py session_env) and the commands of the tick run by fixed rules (rkmon_lib/rules_tick.py); the
  monitor's sends inside one process say so to the gate themselves. It reaches the writer as the word OWN_WORD before
  the command's (rkmon_lib/writer.py routed), which the writer's run takes off and turns back into OWN_VAR
  (rkmon_lib/writer_run.py). It never lets a write out: a write the gate stops is kept as an unsent item when it is the
  monitor's own, and refused when a person typed it.

It loads with the standard library only: docker/rk_write.py loads it, through rkmon_lib/writer.py, before it checks
anything.
"""

from __future__ import annotations

import os
import re
from typing import NamedTuple

GITHUB, TEAMS, WEB, JIRA, NONE = "github", "teams", "web", "jira", "none"
OWN_VAR, OWN_WORD = "RKMON_OWN_WRITE", "--own-write"  # the monitor's own writes (`own`)
OWN, GATED = "own", "gated"
SUDO, HOST = "sudo", "host"
HALF, POST_LOCKED = "half", "post-locked"
ISSUES = ("incident-issues", "stability-issues")
TICKET_FILE, TEAMS_SEND, TEAMS_TEST, WEB_PUSH, WEB_REQUESTS, ATLASSIAN = (
    "ticket-file",
    "teams-send",
    "teams-test",
    "web-push",
    "web-requests",
    "atlassian",
)
GH_LOGINS, WRITER_SECRETS, SECRET_SET = "gh-logins", "writer-secrets", "secret-set"
TICKET_DECLINE = "ticket-decline"
REVIEW_APPROVE = "review-approve"
HOST_KEY = "host-key"
WEB_HOST_INSTALL = "web-host-install"
READ_LOGIN = "read-login"
STANDBY_OFF = "standby-off"


class Command(NamedTuple):
    """One row of TABLE (the module's docstring)."""

    name: str
    dest: str
    gate: str
    features: tuple = ()
    protective: bool = False
    writer: str = ""
    file_args: bool = False
    handler: str = ""
    dry_run: bool = False
    drafts: bool = False
    message: str = ""
    what: str = ""
    session: bool = True

    @property
    def stops(self) -> bool:
        """Whether a stop refuses it: it sends, and is not protective."""
        return self.dest != NONE and not self.protective


def _github(name: str, features: tuple = (), **columns) -> Command:
    """A GitHub write the monitor sends to the writer, its arguments checked as file names."""
    return Command(name, GITHUB, OWN, features, writer=SUDO, file_args=True, **columns)


TABLE = (
    _github("post", ("review-posting",), drafts=True),
    Command("spec-post", GITHUB, OWN, ("review-posting",), dry_run=True, drafts=True),
    _github("notice"),
    _github("arm", ("auto-merge",)),
    _github("disarm", ("auto-merge",), protective=True),
    _github("queue-notice", ("queue-notices",)),
    _github(
        "automerge-off",
        ("auto-merge-guard",),
        protective=True,
        handler=POST_LOCKED,
    ),
    _github("issue-file", ISSUES, handler=POST_LOCKED),
    _github("issue-comment", ISSUES, handler=POST_LOCKED),
    _github("issue-relabel", ISSUES, handler=POST_LOCKED),
    _github("issue-assign", ISSUES, handler=POST_LOCKED),
    _github("label-create", ISSUES, handler=POST_LOCKED),
    _github("request-changes", ("holds",), handler=POST_LOCKED),
    _github("fix-link", ("fix-links",)),
    _github("gpu-row", ("remote-gpu-tests",)),
    Command(
        "teams-breakage",
        TEAMS,
        OWN,
        ("chat-alerts",),
        handler=POST_LOCKED,
        message="develop-break",
    ),
    Command(
        TEAMS_TEST,
        TEAMS,
        OWN,
        ("chat-alerts", "needs-you-alerts"),
        writer=SUDO,
        handler=POST_LOCKED,
        session=False,
    ),
    Command(
        "needs-you-teams",
        TEAMS,
        OWN,
        ("needs-you-alerts",),
        dry_run=True,
        message="needs-you",
    ),
    Command(
        TICKET_FILE, JIRA, OWN, ("found-issue-tickets",), writer=SUDO, session=False
    ),
    Command(
        TEAMS_SEND,
        TEAMS,
        GATED,
        writer=SUDO,
        handler=HALF,
        what="a team chat message",
        session=False,
    ),
    Command(
        WEB_PUSH,
        WEB,
        GATED,
        ("team-dashboard",),
        writer=SUDO,
        file_args=True,
        handler=HALF,
        what="the team dashboard's page",
        session=False,
    ),
    Command(
        ATLASSIAN,
        JIRA,
        GATED,
        ("follow-up-tickets",),
        writer=SUDO,
        handler=HALF,
        what="a JIRA write",
    ),
    Command(WEB_REQUESTS, NONE, NONE, writer=SUDO, handler=HALF, session=False),
    Command(GH_LOGINS, NONE, NONE, writer=SUDO),
    Command(WRITER_SECRETS, NONE, NONE, writer=SUDO),
    Command(SECRET_SET, NONE, NONE, writer=HOST, session=False),
    Command(WEB_HOST_INSTALL, NONE, NONE, writer=HOST, session=False),
    Command(TICKET_DECLINE, NONE, NONE, session=False),
    Command(REVIEW_APPROVE, NONE, NONE, session=False),
    Command(HOST_KEY, NONE, NONE, session=False),
    Command(READ_LOGIN, NONE, NONE, session=False),
    Command(STANDBY_OFF, NONE, NONE, session=False),
)
_BY_NAME = {r.name: r for r in TABLE}


def row(name: str) -> Command | None:
    """The row of rk NAME, or None."""
    return _BY_NAME.get(name)


def names(**columns) -> tuple[str, ...]:
    """The names of the rows whose columns have these values, in the table's order."""
    return tuple(
        r.name
        for r in TABLE
        if all(getattr(r, key) == value for key, value in columns.items())
    )


RK_LINE = r"\b(?:rk|rkmon\.py|rk_write\.py)\b[^|;&\n]*?\s['\"]?"
NAME_END = r"['\"]?(?=$|[\s;&|)])"
_NOT_THE_SESSIONS = re.compile(
    RK_LINE + "(" + "|".join(map(re.escape, names(session=False))) + ")" + NAME_END
)


def not_the_sessions(line: str) -> str:
    """The command the session may not run that the shell line LINE runs through rk, or "" (the module's docstring)."""
    m = _NOT_THE_SESSIONS.search(line or "")
    return m.group(1) if m else ""


def not_the_sessions_why(name: str, login: str = "") -> str:
    """Why the session may not run rk NAME, a command of a row the session may not run (the module's docstring)."""
    return f"rk {name} is for {login or 'a person'} to run, or for rk itself, never for the session"


def own(environ=None) -> bool:
    """Whether this process's writes are the monitor's own (the module's docstring)."""
    e = os.environ if environ is None else environ
    return e.get(OWN_VAR) == "1"
