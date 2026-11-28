# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk rules-tick`: a tick run by fixed rules, for a deployment with no Claude session (the merge queue on and review
automation off, rkmon_lib/features.py rules_only).

It runs `rk tick` and reads its lines as the orchestrator would (rkmon_lib/tickfilter.py: the ACT lines, and each
STANDING line once until it goes away and comes back; a line no row knows counts as one to act on). A line whose action
is fixed (RULES: arm a cleared batch, disarm a PR the queue let go of, turn off a risky auto-merge the auto-merge guard
acts on, start a bisect, post or update a PR's queue notice) runs its `rk` command, whose own guards still decide. Every
other line, and a rule whose command refused, needs a person: it is kept in the state (NEEDS_KEY) and shown on the admin
dashboard's Needs-you list (report.board) and in `rk status`, never dropped. A standing line leaves the list once a full
tick does not print it; any other leaves after EXPIRE_H hours, or with `rk rules-tick --clear`.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import re
import subprocess

from . import config, outward, paths, tickfilter
from .store import now

NEEDS_KEY = "needs_person"
EXPIRE_H = 24
SEEN = "rules_seen.txt"  # in the state folder: the standing lines already acted on or listed
# (the line, after an optional "!! " and "CRITICAL ": the rk command its action is). Each command keeps its own guards:
# `rk arm` refuses a PR that is not cleared, `rk automerge-off` one the auto-merge guard must not touch, `rk
# queue-notice` sends nothing while queue notices are off, and so on.
RULES = (
    (re.compile(r"(?:!! )?ARM_READY #(\d+)\b"), lambda m: ["arm", m.group(1)]),
    (re.compile(r"(?:!! )?DISARM_NEEDED #(\d+)\b"), lambda m: ["disarm", m.group(1)]),
    (
        re.compile(r"(?:!! )?AUTOMERGE_RISK #(\d+)\b.*run `rk automerge-off \1`"),
        lambda m: ["automerge-off", m.group(1)],
    ),
    (re.compile(r"(?:!! )?(?:CRITICAL )?BISECT_NEEDED "), lambda m: ["bisect"]),
    (
        re.compile(r"(?:!! )?QUEUE_NOTICE #(\d+)\b"),
        lambda m: ["queue-notice", m.group(1)],
    ),
)
_PR = re.compile(r"#(\d+)\b")


def rule(line: str) -> list[str] | None:
    """The rk command a line's fixed action is, or None when the line needs judgement."""
    for rx, cmd in RULES:
        m = rx.match(line)
        if m:
            return cmd(m)
    return None


def key(line: str) -> str:
    """The same ask while it is the same: the line without a develop commit and numbers that change between ticks."""
    return tickfilter._standing_key(line)


def _rk() -> list[str]:
    return [str(paths.PACKAGE / "rk")]


def _env() -> dict:
    """The environment of the tick and of each rule's command: this monitor's home, and the mark of the monitor's own
    writes (outward.own: the outward gate keeps one it stops as an unsent item)."""
    return dict(os.environ, RKMON_HOME=str(config.home()), **{outward.OWN_VAR: "1"})


def run(cfg: dict, store, runner=subprocess.run, say=print, timeout: float = 0) -> int:
    """One rules tick. Its exit code is the tick's."""
    from . import idle

    limit = timeout or idle.settings(cfg)["tick_timeout_min"] * 60
    try:
        p = runner(
            [*_rk(), "tick"],
            capture_output=True,
            text=True,
            timeout=limit,
            env=_env(),
        )
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "", 2
    say(out.rstrip())
    seen = pathlib.Path(store.dir) / SEEN
    lines = tickfilter.actionable(out, rc, seen=seen)
    classes = dict(tickfilter.needed(out, rc))
    full = any(ln.startswith(tickfilter.TIME) for ln in out.splitlines())
    person = []
    for line in lines:
        cmd = rule(line)
        if cmd is None:
            person.append((line, ""))
            continue
        try:
            r = runner(
                [*_rk(), *cmd],
                capture_output=True,
                text=True,
                timeout=600,
                env=_env(),
            )
            got, code = ((r.stdout or "") + (r.stderr or "")).strip(), r.returncode
        except subprocess.TimeoutExpired:
            got, code = "timed out", 2
        first = got.splitlines()[0] if got else f"exit {code}"
        say(f"RULE {' '.join(cmd)}: {first[:200]}")
        if code != 0:
            person.append(
                (
                    line,
                    f"the rule's `rk {' '.join(cmd)}` did not go through: {first[:200]}",
                )
            )
    with store.locked(wait=True, timeout=120):
        st = store.load()
        record(st, person, classes, full)
        store.save(st)
    if person:
        say(
            f"NEEDS_PERSON: {len(person)} line(s) need a decision (no Claude session: review automation is off): "
            "the admin dashboard's Needs-you list and rk status show them"
        )
    return rc


def record(
    st: dict, person: list, classes: dict, full: bool, at: str | None = None
) -> None:
    """Keep each line that needs a person, by key; drop a standing one a full tick no longer prints, and any other
    past EXPIRE_H hours."""
    at = at or now()
    book = st.setdefault(NEEDS_KEY, {})
    standing_now = {
        key(ln) for ln, cls in classes.items() if cls == tickfilter.STANDING
    }
    for line, why in person:
        k = key(line)
        e = book.get(k) or {"first": at}
        e.update(
            line=line,
            why=why,
            last=at,
            standing=classes.get(line) == tickfilter.STANDING,
        )
        m = _PR.search(line)
        e["pr"] = int(m.group(1)) if m else None
        book[k] = e
    cutoff = _ago(at, EXPIRE_H)
    for k, e in list(book.items()):
        if e.get("standing"):
            if full and k not in standing_now:
                del book[k]
        elif (e.get("last") or "") < cutoff:
            del book[k]


def _ago(at: str, hours: float) -> str:
    t = dt.datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ") - dt.timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def entries(st: dict) -> list[dict]:
    """The lines waiting for a person, oldest first."""
    return sorted(
        (st.get(NEEDS_KEY) or {}).values(), key=lambda e: e.get("first") or ""
    )


def clear(store) -> int:
    """`rk rules-tick --clear`: empty the list (a person dealt with them); how many there were."""
    with store.locked(wait=True, timeout=120):
        st = store.load()
        n = len(st.get(NEEDS_KEY) or {})
        st[NEEDS_KEY] = {}
        store.save(st)
    return n


def status_lines(st: dict) -> list[str]:
    """`rk status`'s section: the lines waiting for a person."""
    es = entries(st)
    if not es:
        return []
    out = [f"Needs a person (no Claude session here): {len(es)}"]
    for e in es[:20]:
        out.append(
            f"  since {e.get('first')}: {e['line'][:200]}"
            + (f" ({e['why']})" if e.get("why") else "")
        )
    return out
