# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The user's priorities: PRs floated to the top of the review and merge queues, as a develop fix is.
queue.refresh_develop_fixes lists them among the develop fixes as "your priority: WHY": review P0 with the short
settle (`settle_develop_fix_min`), merge tier 0, analysis first, and a priority in the merge plan's waves (after a
develop fix CI sees, before other develop fixes: mergeplan._priority_order).

One home for setting them, used by `rk prioritize N [WHY]` / `rk unprioritize N` and by the HTML dashboard's "Request PR
priority" box (a PR number and a button: the PR joins the priority list, for its reviews and its merge, unless it is
there already), which the local service (webserve) answers with `request`.

Requests from other teams (their clicks are requests that wait for the user's OK): the published page posts them to the
web host (tools/webhost_server.py), which appends them to a requests file at the top of the login's home
(derived.site_requests); each tick copies that file (`pull`) and keeps the new requests in st["priority_requests"]: one
pending request per PR (another requester is folded in; the same requester again is a duplicate), a tick line for each
new one, and a request for a PR that is already a top priority, or not an open PR the monitor tracks, answered at once
instead of queued. The user approves (`approve`, the same prioritize with "requested by WHO from the dashboard, approved
by ME"; `rk prioritize N` too) or declines (`decline`: recorded, dropped from the list). All of it is local: state and
journal, no GitHub write, no ledger entry.
The callers hold the store lock and save the state.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import tempfile

from . import guard, idle, queue
from . import config
from .store import minutes_since, now

DASHBOARD_WHY = "requested from the dashboard"
REQUESTS = "priority_requests"  # st key: {"items": [...], "seen": {id: at}, "pulled_at", "error"}
WHO_MAX, WHY_MAX = 40, 300
KEEP_DAYS = 14  # decided requests, and the ids already read, are kept this long
SHOWN_H = 24  # an answered or approved request stays in the list this long; a declined one leaves it at once
REQUEST_ID = re.compile(r"[A-Za-z0-9]{6,32}")
TS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


def _at(ts: str | None) -> str:
    return f"{ts[:10]} {ts[11:16]}Z" if ts else "an unknown time"


def text(v, most: int) -> str:
    """Untrusted text from a request: one line of printable characters, cut to `most`."""
    s = " ".join(v.split()) if isinstance(v, str) else ""
    return "".join(ch for ch in s if ch.isprintable())[:most]


def requesters(item: dict) -> str:
    return ", ".join([item["who"]] + [a["who"] for a in item.get("also") or []])


def requested_why(item: dict, me: str) -> str:
    return f"requested by {requesters(item)} from the dashboard, approved by {me}"


def _book(st: dict) -> dict:
    book = st.setdefault(REQUESTS, {})
    book.setdefault("items", [])
    book.setdefault("seen", {})
    return book


def pending(st: dict, n: int) -> dict | None:
    """PR n's pending request, if any."""
    return next(
        (
            it
            for it in (st.get(REQUESTS) or {}).get("items") or []
            if it["pr"] == n and it["status"] == "pending"
        ),
        None,
    )


def _decide(item: dict, status: str, by: str, note: str = "") -> None:
    item.update(
        status=status,
        decided_at=now(),
        decided_by=by,
        note=note or item.get("note", ""),
    )


def _not_open(st: dict, n: int) -> str | None:
    rec = st["prs"].get(str(n))
    if not rec:
        return f"#{n} is not an open rocKE PR the monitor tracks (as of the last tick, {_at(st.get('last_tick'))})"
    if rec.get("status") != "open":
        return f"#{n} is {str(rec.get('status')).replace('-', ' ')}, not open"
    return None


def _already(st: dict, n: int) -> str | None:
    had = (st.get("user_priority") or {}).get(str(n))
    if had:
        return f"#{n} is already in the priority queue since {_at(had.get('at'))}" + (
            f" ({had['why']})" if had.get("why") else ""
        )
    return None


def set_priority(
    st: dict, store, n: int, why: str = "", on: bool = True, by: str | None = None
) -> list[int]:
    """`rk prioritize N WHY` (on) or `rk unprioritize N` (off): st["user_priority"], the develop fixes it feeds, and the
    journal line. Prioritizing approves PR n's pending request, if any (with no WHY, the reason names its requesters
    and `by`). Returns the user's priorities, sorted."""
    pri = st.setdefault("user_priority", {})
    item = pending(st, n) if on else None
    if on:
        why = why or (requested_why(item, by) if item and by else "")
        pri[str(n)] = {"why": why, "at": now()}
    else:
        pri.pop(str(n), None)
    queue.refresh_develop_fixes(st, guard.incidents(store))
    mine = sorted(int(k) for k in pri)
    store.journal(
        "mode",
        f"{'prioritize' if on else 'unprioritize'} #{n}"
        + (f" ({why})" if on and why else "")
        + f": your priorities {mine}",
    )
    if item:
        _decide(
            item,
            "approved",
            by or "rk prioritize",
            f"prioritized: {why}" if why else "prioritized",
        )
    return mine


def request(
    st: dict, store, n: int, why: str = DASHBOARD_WHY, by: str | None = None
) -> tuple[bool, str]:
    """The user's own priority request for PR n from the dashboard (on the monitor's machine): what `rk prioritize N
    requested from the dashboard` does, once per PR. Returns (applied, the message for the page). Refused, with nothing
    changed: a PR the monitor does not track as open (unknown to it, merged, closed, out of scope), or one already in
    the priority queue. A develop fix, already a top priority, is prioritized as `rk prioritize` allows: the user's
    priorities rank before the other develop fixes among the plan's priorities, after a fix that CI sees.
    """
    k = str(n)
    refused = _not_open(st, n) or _already(st, n)
    if refused:
        return False, f"{refused}: nothing changed."
    rec = st["prs"][k]
    fix = list((st.get("develop_fixes") or {}).get(k) or [])
    set_priority(st, store, n, why, by=by)
    return True, (
        f"#{n} added to the priority queue: reviewed first and merged first. The next tick (within 10 min) "
        "makes it a priority in the merge plan."
        + (
            f" It was a top priority already ({', '.join(fix)}); now it is yours too."
            if fix
            else ""
        )
        + (
            " It is a draft: it takes its place once it is ready for review."
            if rec.get("draft")
            else ""
        )
    )


# --- requests from other teams -----------------------------------------------------------------------------------------


def clean(raw) -> dict | None:
    """One line of the web host's requests file, checked again here (the file is outside our control): None when it is
    not a well-formed request."""
    if not isinstance(raw, dict):
        return None
    pr, rid, at = raw.get("pr"), raw.get("id"), raw.get("at")
    if not (isinstance(pr, int) and not isinstance(pr, bool) and 0 < pr < 10**7):
        return None
    if not (
        isinstance(rid, str)
        and REQUEST_ID.fullmatch(rid)
        and isinstance(at, str)
        and TS.fullmatch(at)
    ):
        return None
    who, why = text(raw.get("who"), WHO_MAX), text(raw.get("why"), WHY_MAX)
    return (
        {"id": rid, "pr": pr, "who": who, "why": why, "at": at} if who and why else None
    )


def intake(st: dict, raw) -> str | None:
    """One request read from the web host: queued once per PR, answered when the PR is already a top priority or not
    an open PR the monitor tracks. Returns the tick line for a new request, None for a duplicate or a bad line.
    """
    req = clean(raw)
    book = _book(st)
    if not req or req["id"] in book["seen"]:
        return None
    book["seen"][req["id"]] = req["at"]
    n, item = req["pr"], pending(st, req["pr"])
    line = f"PRIORITY_REQUEST #{n} by {req['who']}: {req['why']}"
    if item:
        if req["who"].casefold() in {
            w.casefold()
            for w in [item["who"]] + [a["who"] for a in item.get("also") or []]
        }:
            return None  # the same person again (a second click): one request
        item.setdefault("also", []).append(
            {k: req[k] for k in ("id", "who", "why", "at")}
        )
        return line
    fix = [
        w
        for w in (st.get("develop_fixes") or {}).get(str(n)) or []
        if not w.startswith(queue.USER_PRIORITY)
    ]
    answer = (
        _not_open(st, n)
        or _already(st, n)
        or (
            f"#{n} is a top priority already: it fixes develop " f"({', '.join(fix)})"
            if fix
            else None
        )
    )
    item = dict(req, got=now(), status="pending", note="")
    book["items"].append(item)
    if answer:
        _decide(item, "answered", "monitor", answer)
        return f"(priority request #{n} by {req['who']} answered: {answer})"
    return line


def refresh(st: dict) -> None:
    """Pending requests the state has overtaken: a PR prioritized meanwhile is approved, one no longer open answered.
    Decided requests and read ids older than KEEP_DAYS go."""
    book = _book(st)
    for item in book["items"]:
        if item["status"] != "pending":
            continue
        gone = _not_open(st, item["pr"])
        if gone:
            _decide(item, "answered", "monitor", f"{gone} any more")
        elif str(item["pr"]) in (st.get("user_priority") or {}):
            _decide(item, "approved", "rk prioritize", "prioritized")
    old = KEEP_DAYS * 24 * 60
    book["items"] = [
        it
        for it in book["items"]
        if it["status"] == "pending"
        or minutes_since(it.get("decided_at") or it["at"]) < old
    ]
    book["seen"] = {
        k: at for k, at in book["seen"].items() if minutes_since(at) < old + 24 * 60
    }


ENDED = (
    "merged",
    "closed",
)  # a PR in either state is done: its priority and its requests leave every list


def ended(st: dict, n: int) -> str | None:
    """PR n's status when it merged or closed, else None (open, out of scope, or not tracked: kept, since scope can come
    back and a missing record is no evidence)."""
    status = (st["prs"].get(str(n)) or {}).get("status")
    return status if status in ENDED else None


def prune(st: dict, store) -> list[str]:
    """Each tick: a priority whose PR merged or closed leaves st["user_priority"], with a journal line and a tick line,
    so its rows leave the dashboard. shown() drops the PR's requests at the same time.
    """
    pri = st.get("user_priority") or {}
    lines = []
    for k in sorted(pri, key=int):
        status = ended(st, int(k))
        if status:
            why = pri.pop(k).get("why") or ""
            store.journal(
                "mode",
                f"priority #{k} cleared: {status}"
                + (f" (it was: {why})" if why else ""),
            )
            lines.append(
                f"PRIORITY_CLEARED #{k}: {status}; it leaves the board and the dashboard"
            )
    return lines


def pull(cfg: dict, st: dict, store, fetch) -> list[str]:
    """Each tick: copy the web host's requests file, or those the team dashboard served here took (`fetch(cfg,
    dest)`: publish.pull_requests), and take in the requests not read yet. Advisory: a failure is one line, and the
    pending requests stay as they are."""
    from . import publish

    if not publish.takes_requests(cfg):
        return []
    dest = store.dir / "priority_requests.pulled.jsonl"
    book = _book(st)
    book["pulled_at"] = now()
    why = fetch(cfg, str(dest))
    book["error"] = why
    lines = [f"(priority requests not read: {why})"] if why else []
    if not why and dest.exists():
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=KEEP_DAYS)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        for raw_line in dest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                raw = json.loads(raw_line)
            except ValueError:
                continue
            if (
                isinstance(raw, dict)
                and isinstance(raw.get("at"), str)
                and raw["at"] >= cutoff
            ):
                line = intake(st, raw)
                if line:
                    lines.append(line)
    refresh(st)
    return lines


def approve(cfg: dict, st: dict, store, n: int) -> tuple[bool, str]:
    """The user approves PR n's pending request: the same prioritize as `request`, its reason naming the requesters."""
    item = pending(st, n)
    if not item:
        return False, f"#{n} has no pending priority request: nothing changed."
    gone = _not_open(st, n) or _already(st, n)
    if gone:
        _decide(item, "answered", "monitor", gone)
        return False, f"{gone}: the request is answered, nothing else changed."
    me = config.operator_login(cfg) or "the monitor owner"
    why = requested_why(item, me)
    set_priority(st, store, n, why, by=me)
    return True, (
        f"#{n} approved ({why}): reviewed first and merged first. The next tick (within 10 min) makes it a "
        "priority in the merge plan."
    )


def decline(cfg: dict, st: dict, store, n: int) -> tuple[bool, str]:
    """The user declines PR n's pending request: recorded in the state and the journal, dropped from the list."""
    item = pending(st, n)
    if not item:
        return False, f"#{n} has no pending priority request: nothing changed."
    me = config.operator_login(cfg) or "the monitor owner"
    _decide(item, "declined", me)
    store.journal(
        "mode",
        f"priority request #{n} by {requesters(item)} ({item['why']}) declined by {me}",
    )
    return True, f"Declined the priority request for #{n} (by {requesters(item)})."


def shown(st: dict) -> list[dict]:
    """The requests to list: the pending ones (oldest first), then those answered or approved in the last SHOWN_H
    hours (newest first). Declined ones are not listed, and neither is a request whose PR merged or closed (prune).
    """
    items = [
        it
        for it in (st.get(REQUESTS) or {}).get("items") or []
        if not ended(st, it["pr"])
    ]
    done = [
        it
        for it in items
        if it["status"] in ("answered", "approved")
        and minutes_since(it.get("decided_at") or it["at"]) < SHOWN_H * 60
    ]
    return [it for it in items if it["status"] == "pending"] + sorted(
        done, key=lambda it: it.get("decided_at") or it["at"], reverse=True
    )


def status_text(item: dict) -> str:
    if item["status"] == "pending":
        return "pending"
    return f"{item['status']}: {item.get('note') or ''}".rstrip(": ")


def lines(cfg: dict, st: dict) -> list[str]:
    """`rk priority-requests`: every listed request, one line each."""
    out = []
    for it in shown(st):
        out.append(
            f"#{it['pr']} {status_text(it)} · requested by {requesters(it)} at {_at(it['at'])}: {it['why']}"
            + (
                f" (also: {'; '.join(a['why'] for a in it.get('also') or [])})"
                if it.get("also")
                else ""
            )
        )
    book = st.get(REQUESTS) or {}
    out.append(
        f"({sum(it['status'] == 'pending' for it in book.get('items') or [])} pending; read from the web host "
        f"{_at(book.get('pulled_at')) if book.get('pulled_at') else 'never'}"
        + (f"; last read failed: {book['error']}" if book.get("error") else "")
        + ")"
    )
    return out


@idle.part("priority requests", remote=True)
def _requests(cfg: dict, st: dict, store) -> str | None:
    """The web host's requests file as `pull` copies it (its digest): a new request makes the next tick full. A read
    that fails reads as "unread"; the full tick's pull reports why."""
    if not ((cfg.get("publish") or {}).get("html") or {}).get("requests"):
        return None
    from . import publish

    with tempfile.TemporaryDirectory() as d:
        dest = pathlib.Path(d) / "requests.jsonl"
        if publish.pull_requests(cfg, str(dest)):
            return "unread"
        return (
            idle.digest(dest.read_text(encoding="utf-8", errors="replace"))
            if dest.exists()
            else None
        )
