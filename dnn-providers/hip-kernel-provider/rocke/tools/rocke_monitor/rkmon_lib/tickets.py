# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tickets waiting for you (the admin dashboard's card, built like the priority-request card): proposed JIRA tickets
for the other problems found along the way (ticket creation's out_of_scope_issue trigger: not a develop breakage, not
a PR's review finding), each of which waits for the operator's OK. Follow-up tickets for merged PRs are not proposed
here: they are filed without asking, as the skill's rules say.

- `rk ticket-draft FILE|-` is how the monitor's Claude session proposes one: a `---` block (title, type, priority,
  assignee, summary, and optionally labels, component, pr, source), then the full description; or a JSON object with
  the same keys and `description`. The title, the priority, the summary and the description are required (`check`:
  check 6 counts the priority among the fields every new ticket gets, ticket_fields.FILLED). It refuses a proposed
  ticket whose title a waiting one has, or one filed or declined in the last KEEP_DAYS days; but one with the title
  of a waiting proposed ticket that cannot be filed (`unfit`) takes its place, under its id: that is how such a
  proposed ticket is edited.
- Approve (the card's button; `rk ticket-file ID` is the operator's command-line fallback, the same code) files it
  through the atlassian CLI (`atlassian --site S --json jira issue create ...`; with a default epic
  (features.ticket_creation.default_epic) the issue is made without its description, linked with `jira issue
  link-epic KEY --epic EPIC`, and only then given its description with `jira issue update KEY`, since linking can
  garble a description set before it; then `jira issue update KEY --assignee` when the proposed ticket names
  someone), records the key in the ledger (kind `ticket`) and shows it in the row. In the container the writer, which
  alone has the CLI's login, files it: `approve` sends the card's Approve to it (`rk ticket-file ID`, the same code).
  The project, the default type, the priority names and any fields the project requires come from config (`jira.*`),
  never from the package. Nothing is filed while ticket creation is off or the monitor is stopped: the proposed
  ticket waits. One that `check` refuses (`unfit`) is neither filed nor kept as an unsent item: it waits, with
  check's words as its note, and the refusal names the two fixes, a proposed ticket of its title (which takes its
  place) or Decline.
- While ticket creation does not send (prototype, or the monitor in standby: features.held), the card's Approve files
  nothing: the one outward gate (post.outward_gate) keeps the ticket as an unsent item (for the JIRA project,
  `unsent_for`; its content the fields it would be filed with and its description, `unsent_text`), and the proposed
  ticket is done (status unsent); no CLI login is needed. `rk ticket-file ID` typed then is refused, naming the mode
  or standby.
- Decline (the card; `rk ticket-decline ID [WHY]`) drops the proposed ticket and journals it.
- One action per proposed ticket: once approved, being filed, kept as an unsent item or declined, every other action
  on it is refused, naming the first.

State: st["tickets"]["items"], each {id, title, type, priority, assignee, summary, description, labels, component, pr,
source, by, at, status (waiting, filing, filed, unsent, declined), key, url, note, decided_by, decided_at}. The callers
of `add` and `decline` hold the store lock and save the state; `approve` takes the locks itself (the store's, around
its two state changes, and, unless its caller holds it, the write lock around the JIRA write and its ledger entry).
"""

from __future__ import annotations

import html
import json
import re
import secrets
import shutil

from . import config, control, derived, ticket_fields
from .store import minutes_since, now

BOOK = "tickets"
ID = re.compile(r"[0-9a-f]{8}")
FIELDS = (
    "title",
    "type",
    "priority",
    "assignee",
    "summary",
    "labels",
    "component",
    "pr",
    "source",
)
# The fields `add` fills from the configuration where a proposed ticket leaves them blank (jira.issue_type,
# jira.labels, jira.component): the operator's settings, not the proposed ticket's, so `unfit` leaves them out.
FROM_SETTINGS = ("type", "labels", "component")
LIMITS = {
    "title": 200,
    "summary": 300,
    "type": 40,
    "priority": 40,
    "assignee": 100,
    "component": 80,
    "source": 300,
}
DESCRIPTION_MAX = 30000
LABEL = re.compile(r"[A-Za-z0-9_.-]{1,50}")
# Filed, unsent and declined proposed tickets are kept this long: one with the same title is refused meanwhile.
KEEP_DAYS = 14
SHOWN_H = 24  # a filed proposed ticket stays on the card this long, with its key
NOBODY = ("", "-", "none", "unassigned", "nobody")
CARD_WAIT_S = 60  # how long the card's Approve waits for the write lock where it files the ticket itself (a host)
CLI_TIMEOUT_S = 90
# The longest the writer's filing takes (the card's Approve in the container waits for its answer): rk ticket-file's
# wait for the write lock, as every outward rk command waits (control.WRITE_WAIT_S), then six CLI calls at most (the
# issue, the search after a failed one, the epic link, the description, the account and the assignee).
FILING_MAX_S = control.WRITE_WAIT_S + 6 * CLI_TIMEOUT_S
FEATURE = "found-issue-tickets"


class DraftError(ValueError):
    """What is wrong with a proposed ticket, in words."""


class TicketError(RuntimeError):
    """Why a filing failed, in words."""


class TicketInDoubt(TicketError):
    """A filing that failed or timed out after JIRA may have taken it, and JIRA could not be searched to settle it: the
    proposed ticket stays `filing` (in doubt), and the next try (`rk ticket-file ID --again`) searches JIRA first.
    """


class TicketUnfit(TicketError):
    """`check` refuses the proposed ticket (`unfit`): nothing was sent, and it waits with check's words as its note."""


DOUBT_SLACK_MIN = 2  # the search looks this much earlier than the try (clock skew)
# JIRA's own refusal of a request (a 4xx answer): the issue was not made, so no doubt.
REFUSED_BY_JIRA = re.compile(
    r"\b4\d\d\b|Bad Request|Forbidden|Unauthorized|not valid", re.I
)


# --- proposed tickets -------------------------------------------------------------------------------------------------


def _line(v, most: int) -> str:
    """One line of printable text (white space collapsed), or "" when `v` is no text."""
    s = (
        " ".join(str(v).split())
        if isinstance(v, (str, int)) and not isinstance(v, bool)
        else ""
    )
    return "".join(ch for ch in s if ch.isprintable())[: most + 1]


def _text(v) -> str:
    """The description: its lines kept, other control characters dropped."""
    s = v if isinstance(v, str) else ""
    s = "".join(
        ch for ch in s.replace("\r\n", "\n") if ch.isprintable() or ch in "\n\t"
    )
    return s.strip("\n")


def parse(text: str) -> dict:
    """A proposed ticket's file: a `---` block of `key: value` lines, then the full description; or a JSON object.
    Returns the checked proposed ticket (`check`); DraftError names everything wrong with it.
    """
    t = text.lstrip("\ufeff")
    if t.lstrip().startswith("{"):
        try:
            data = json.loads(t)
        except ValueError as exc:
            raise DraftError(f"not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise DraftError("a proposed ticket in JSON is an object")
        return check(data)
    m = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", t, re.S)
    if not m:
        raise DraftError(
            "a proposed ticket starts with a --- block (title:, type:, priority:, assignee:, summary:), then the full "
            "description; or it is a JSON object with those keys and description"
        )
    meta = {}
    for ln in m.group(1).splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if ":" not in ln:
            raise DraftError(
                f"not a `key: value` line in the --- block: {ln.strip()[:60]}"
            )
        k, v = ln.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return check(dict(meta, description=t[m.end() :]))


def check(data: dict) -> dict:
    """The proposed ticket, normalized: one-line fields within their limits, the title, the priority and the summary
    required, labels as a list, pr as a number. DraftError lists every problem at once.
    """
    bad, out = [], {}
    unknown = sorted(set(data) - set(FIELDS) - {"description"})
    if unknown:
        bad.append(
            f"unknown keys {', '.join(unknown)} (the keys: {', '.join(FIELDS)}, description)"
        )
    for k, most in LIMITS.items():
        v = _line(data.get(k, ""), most)
        if len(v) > most:
            bad.append(f"{k}: at most {most} characters on one line")
        out[k] = v[:most]
    # The priority too: check 6 counts it among the fields every new ticket gets (ticket_fields.FILLED), so the filer
    # sends one with each ticket.
    for k in ("title", "priority", "summary"):
        if not out[k]:
            bad.append(f"{k}: required (one line)")
    if out["assignee"].lower() in NOBODY:
        out["assignee"] = ""
    desc = _text(data.get("description", ""))
    if not desc.strip():
        bad.append("description: required (the full text, after the --- block)")
    elif len(desc) > DESCRIPTION_MAX:
        bad.append(f"description: at most {DESCRIPTION_MAX} characters")
    out["description"] = desc[:DESCRIPTION_MAX]
    labels = data.get("labels") or []
    labels = [
        x.strip()
        for x in (labels.split(",") if isinstance(labels, str) else labels)
        if str(x).strip()
    ]
    if len(labels) > 10 or not all(
        isinstance(x, str) and LABEL.fullmatch(x) for x in labels
    ):
        bad.append(
            "labels: at most 10, each letters, digits, _ . or - (comma-separated)"
        )
    out["labels"] = [x for x in labels if isinstance(x, str)][:10]
    pr = data.get("pr")
    if pr in (None, ""):
        out["pr"] = None
    elif (
        re.fullmatch(r"#?\d{1,7}", str(pr).strip())
        and int(str(pr).strip().lstrip("#")) > 0
    ):
        out["pr"] = int(str(pr).strip().lstrip("#"))
    else:
        bad.append("pr: a PR number")
        out["pr"] = None
    if bad:
        raise DraftError("; ".join(bad))
    return out


def unfit(item: dict) -> str:
    """What `check` refuses in recorded proposed ticket ITEM, in check's words ("" when nothing): one another
    deployment recorded (adopt brings its state over) can lack what check asks for. The fields `add` fills from the
    configuration (FROM_SETTINGS) are the operator's settings, not the proposed ticket's: they are left out.
    """
    try:
        check(
            {k: item.get(k) for k in (*FIELDS, "description") if k not in FROM_SETTINGS}
        )
    except DraftError as exc:
        return str(exc)
    return ""


def draft_id(data) -> str | None:
    """A request body's proposed ticket id: {"id": "0a1b2c3d"}."""
    v = data.get("id") if isinstance(data, dict) else None
    return v if isinstance(v, str) and ID.fullmatch(v) else None


def clean_id(v) -> str | None:
    return v if isinstance(v, str) and ID.fullmatch(v) else None


def _book(st: dict) -> dict:
    book = st.setdefault(BOOK, {})
    book.setdefault("items", [])
    return book


def _norm(title: str) -> str:
    return " ".join(title.split()).casefold()


def _at(ts: str | None) -> str:
    return f"{ts[:10]} {ts[11:16]}Z" if ts else "an unknown time"


def find(st: dict, tid: str | None) -> dict | None:
    return next(
        (it for it in (st.get(BOOK) or {}).get("items") or [] if it["id"] == tid), None
    )


def prune(st: dict) -> None:
    """Filed, unsent and declined proposed tickets older than KEEP_DAYS go."""
    book = _book(st)
    old = KEEP_DAYS * 24 * 60
    book["items"] = [
        it
        for it in book["items"]
        if it["status"] in ("waiting", "filing")
        or minutes_since(it.get("decided_at") or it["at"]) < old
    ]


def waiting(st: dict) -> list[dict]:
    """The proposed tickets that wait for the operator (and one being filed), oldest first."""
    return [
        it
        for it in (st.get(BOOK) or {}).get("items") or []
        if it["status"] in ("waiting", "filing")
    ]


def shown(st: dict) -> list[dict]:
    """The card's rows: the waiting proposed tickets (oldest first), then those filed or kept as unsent items in the
    last SHOWN_H hours (newest first). A declined one is dropped from the card at once.
    """
    items = (st.get(BOOK) or {}).get("items") or []
    done = [
        it
        for it in items
        if it["status"] in ("filed", "unsent")
        and minutes_since(it.get("decided_at") or it["at"]) < SHOWN_H * 60
    ]
    return waiting(st) + sorted(
        done, key=lambda it: it.get("decided_at") or it["at"], reverse=True
    )


def _jira(cfg: dict) -> dict:
    return cfg.get("jira") or {}


def add(
    cfg: dict, st: dict, store, draft: dict, by: str = "the monitor's Claude session"
) -> tuple[bool, str, str]:
    """A checked proposed ticket (`parse`) into the waiting list: (added, the message, its id or the id it collides
    with). One with the title of a waiting proposed ticket that cannot be filed (`unfit`) takes its place and its id.
    """
    prune(st)
    book = _book(st)
    key = _norm(draft["title"])
    replaced, why = None, ""
    for it in book["items"]:
        if _norm(it["title"]) != key:
            continue
        why = unfit(it) if it["status"] == "waiting" else ""
        if why:  # it cannot be filed as it is: this proposed ticket is its edit
            replaced = it
            break
        if it["status"] in ("waiting", "filing"):
            return (
                False,
                f"proposed ticket {it['id']} has this title and waits already: nothing added",
                it["id"],
            )
        if it["status"] == "filed":
            return (
                False,
                (
                    f"filed already as {it.get('key')} on {_at(it.get('decided_at'))} (proposed ticket {it['id']}): "
                    "nothing added"
                ),
                it["id"],
            )
        if it["status"] == "unsent":
            return (
                False,
                (
                    f"kept as an unsent item on {_at(it.get('decided_at'))} (proposed ticket {it['id']}): nothing added"
                ),
                it["id"],
            )
        return (
            False,
            (
                f"declined by {it.get('decided_by') or 'the operator'} on {_at(it.get('decided_at'))} (proposed "
                f"ticket {it['id']}): nothing added"
            ),
            it["id"],
        )
    if replaced:
        tid = replaced["id"]
    else:
        ids = {it["id"] for it in book["items"]}
        tid = secrets.token_hex(4)
        while tid in ids:
            tid = secrets.token_hex(4)
    j = _jira(cfg)
    item = dict(
        draft,
        id=tid,
        type=draft.get("type") or ticket_fields.issue_type(j.get("issue_type")),
        labels=draft.get("labels")
        or [x for x in (j.get("labels") or []) if isinstance(x, str)],
        component=draft.get("component") or (j.get("component") or "").strip(),
        by=by,
        at=now(),
        status="waiting",
        key="",
        url="",
        note="",
    )
    if replaced:
        book["items"][book["items"].index(replaced)] = item
    else:
        book["items"].append(item)
    store.journal(
        "ticket",
        f"{'proposed again' if replaced else 'proposed'} {tid}: {item['title']} ({item['type']}, "
        f"{item['priority'] or 'no priority'}; by {by})",
    )
    me = config.operator_login(cfg) or "the operator"
    return (
        True,
        (
            f"TICKET_DRAFTED {tid}: {item['title']}."
            + (
                f" It takes the place of the waiting proposed ticket of this title, which could not be filed: {why}."
                if replaced
                else ""
            )
            + f" It waits for {me}'s OK on the admin dashboard's \"Tickets waiting for you\" card, which files it "
            "or drops it."
        ),
        tid,
    )


# --- deciding -----------------------------------------------------------------------------------------------------------


def _first_action(item: dict) -> str | None:
    """The action already taken on a proposed ticket, in words (None while it waits)."""
    who, when = item.get("decided_by") or "the operator", _at(
        item.get("decided_at") or item.get("filing_at")
    )
    return {
        "filed": f"proposed ticket {item['id']} is filed already as {item.get('key')} (approved by {who}, {when})",
        "unsent": f"proposed ticket {item['id']} was kept as an unsent item (approved by {who}, {when})",
        "declined": f"proposed ticket {item['id']} was declined by {who} ({when})",
        "filing": f"proposed ticket {item['id']} is being filed since {when} (approved by {who})",
    }.get(item["status"])


def status_text(cfg: dict, item: dict) -> str:
    if item["status"] == "filed":
        return f"filed as {item.get('key')}" + (
            f"; {item['note']}" if item.get("note") else ""
        )
    if item["status"] == "filing":
        return f"being filed (since {_at(item.get('filing_at'))})"
    if item["status"] == "declined":
        return "declined"
    if item["status"] == "unsent":
        return "not filed: kept as an unsent item" + (
            f" ({item['note']})" if item.get("note") else ""
        )
    return "waiting" + (f"; {item['note']}" if item.get("note") else "")


def sends_for() -> str:
    """The feature a found issue's ticket sends for (features.Capability.feature): ticket creation."""
    from . import features

    return features.CAPABILITIES[FEATURE].feature


def held_words(cfg: dict) -> str:
    """Why Approve files nothing now and keeps the ticket as an unsent item (features.held_words), "" when it files."""
    from . import features

    return features.held_words(cfg, sends_for())


def unsent_for(cfg: dict) -> str:
    """What an unsent ticket was for, in words: the JIRA project it would be filed in (jira_project)."""
    project = jira_project(cfg)
    return f"the JIRA project {project}" if project else "the JIRA project"


def unsent_text(cfg: dict, item: dict, by: str) -> str:
    """Proposed ticket ITEM as its unsent item keeps it: the fields it would be filed with, one a line, then its
    description as it would be filed (`description`; approved by BY)."""
    fields = (
        ("summary", item["title"]),
        ("type", item["type"]),
        ("priority", item.get("priority") or ""),
        ("assignee", item.get("assignee") or ""),
        ("labels", ", ".join(item.get("labels") or ())),
        ("component", item.get("component") or ""),
        ("epic", epic(cfg)),
    )
    return (
        "\n".join(f"{k}: {v}" for k, v in fields if v)
        + "\n\n"
        + description(cfg, item, by)
    )


def approve_refusal(
    cfg: dict, st: dict, tid: str, again: bool = False, own: bool | None = None
) -> str | None:
    """Why proposed ticket `tid` may not be filed now (None when it may): one you typed (OWN false, outward.own)
    while ticket creation does not send among them, naming the mode or standby (post.outward_check).
    """
    from . import features, post
    from .store import halted, stopped

    item = find(st, tid)
    if not item:
        return f"no proposed ticket {tid}: nothing changed"
    from . import mirror

    if mirror.on(cfg):
        return "this monitor is a mirror (config mirror.enabled): nothing is filed; the proposed ticket waits"
    first = _first_action(item)
    if first and not (again and item["status"] == "filing"):
        return f"{first}: one action per proposed ticket, nothing changed"
    gate = features.refusal(cfg, FEATURE)
    if gate:
        return gate[1]
    if stopped(st):
        return (
            f"{halted(st)}: nothing is filed until rk start; the proposed ticket waits"
        )
    try:
        post.outward_check(cfg, st, f"proposed ticket {tid}", (FEATURE,), own=own)
    except post.PostRefused as exc:
        return f"{exc}; it waits"
    return None


def unfit_refusal(tid: str, why: str) -> str:
    """The refusal of a filing that `unfit` stops (WHY: check's words), naming the two fixes."""
    return (
        f"proposed ticket {tid} was not filed: {why}. It waits: propose it again under its title with what is "
        "missing (rk ticket-draft FILE; it takes this one's place), or decline it (Decline on the card, or rk "
        f"ticket-decline {tid})."
    )


def approve(
    cfg: dict,
    store,
    tid: str,
    by: str,
    lock_writes: bool = True,
    again: bool = False,
    run=None,
    own: bool | None = None,
) -> tuple[bool, str, dict]:
    """The operator approves proposed ticket `tid` (the card, or `rk ticket-file`): (done, the message, {"key", "url",
    "row": the row's state}). `lock_writes`: take the write lock here (the card); `rk ticket-file` runs under it
    already. `again`: file one whose filing was cut off (the operator checked JIRA first). OWN: the monitor's own
    Approve (the card's), which, while ticket creation does not send, keeps the ticket as an unsent item (`_keep`); one
    you typed is refused then (approve_refusal). Where this process does not read the CLI's login (the container), the
    writer files it (`_approve_by_writer`). One `check` refuses (`unfit`) is neither filed nor kept as an unsent item:
    it waits, with check's words as its note (`unfit_refusal`); one whose filing is in doubt is searched for in JIRA
    first, and is not sent again (`_file_and_record`)."""
    from . import atlassian, features

    if features.sends(cfg, sends_for()) and not atlassian.login_here(cfg):
        return _approve_by_writer(cfg, store, tid, again, own)
    with store.locked(wait=True):
        st = store.load()
        why = approve_refusal(cfg, st, tid, again, own)
        if why:
            return False, why, {}
        item = find(st, tid)
        why = unfit(item) if item["status"] == "waiting" else ""
        if why:
            item["note"] = why[:300]
            store.save(st)
            store.journal("ticket", f"filing {tid} refused: {why}")
            return False, unfit_refusal(tid, why), {"row": status_text(cfg, item)}
        kept = _keep(cfg, st, store, item, by, own)
        if kept:
            return True, kept, {"row": status_text(cfg, item)}
        item.update(status="filing", filing_at=now(), decided_by=by)
        store.save(st)
    snapshot = dict(item)
    key, url, why, note, doubt, refused = "", "", "", "", False, False
    try:
        if lock_writes:
            with control.write_lock(store.dir, CARD_WAIT_S) as held:
                if not held:
                    why = "an outward write is in flight (another write, or rk stop or rk kill): try again"
                else:
                    key, url, note = _file_and_record(cfg, store, snapshot, by, run)
        else:
            key, url, note = _file_and_record(cfg, store, snapshot, by, run)
    except TicketInDoubt as exc:
        why, doubt = str(exc), True
    except TicketUnfit as exc:
        why, refused = str(exc), True
    except TicketError as exc:
        why = str(exc)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a filing that cannot run leaves the proposed ticket waiting, with the reason
        why = f"{type(exc).__name__}: {exc}"
    with store.locked(wait=True):
        st = store.load()
        item = find(st, tid)
        if key:
            item.update(
                status="filed",
                key=key,
                url=url,
                decided_at=now(),
                decided_by=by,
                note=note,
            )
        elif (
            doubt
        ):  # it may be in JIRA: it stays being filed, and the next try searches JIRA first
            item.update(
                status="filing",
                doubt_since=item.get("doubt_since") or snapshot.get("filing_at"),
                note=f"in doubt: {why}"[:300],
            )
        else:
            item.update(
                status="waiting",
                note=(why if refused else f"the last filing failed: {why}")[:300],
            )
            item.pop("filing_at", None)
            item.pop("doubt_since", None)
        store.save(st)
    if not key:
        store.journal(
            "ticket", f"filing {tid} {'refused' if refused else 'failed'}: {why}"
        )
        return (
            False,
            (
                f"proposed ticket {tid} may have been filed: {why}. It stays being filed: `rk ticket-file {tid} "
                "--again` searches JIRA for it first, and files it only when it is not there."
                if doubt
                else (
                    unfit_refusal(tid, why)
                    if refused
                    else f"proposed ticket {tid} was not filed: {why}. It waits."
                )
            ),
            {"row": status_text(cfg, item)},
        )
    store.journal(
        "ticket",
        f"filed {tid} as {key}: {snapshot['title']} (approved by {by})"
        + (f"; {note}" if note else ""),
    )
    return (
        True,
        (
            f"Filed proposed ticket {tid} as {key}: {snapshot['title']}"
            + (f" ({url})" if url else "")
            + (f". {note[0].upper()}{note[1:]}." if note else ".")
        ),
        {"key": key, "url": url, "row": status_text(cfg, item)},
    )


def _keep(cfg: dict, st: dict, store, item: dict, by: str, own) -> str:
    """The one outward gate (post.outward_gate) for proposed ticket ITEM, right before it is filed: "" when it goes
    out; else, ticket creation not sending, the line that says so, with the ticket kept as an unsent item (its one
    journal line the gate's, post.keep) and the proposed ticket done (status unsent, saved; the caller locks the store).
    """
    from . import post, state_store

    try:
        post.outward_gate(
            cfg,
            st,
            f"proposed ticket {item['id']}",
            (FEATURE,),
            post.unsent_item(
                state_store.TICKET, unsent_for(cfg), unsent_text(cfg, item, by)
            ),
            own=own,
            store=store,
        )
    except post.Held as kept:
        item.update(
            status="unsent", decided_at=now(), decided_by=by, note=held_words(cfg)[:300]
        )
        store.save(st)
        return str(kept)
    return ""


def _approve_by_writer(
    cfg: dict, store, tid: str, again: bool, own: bool | None = None
) -> tuple[bool, str, dict]:
    """`approve` through the writer, which files the proposed ticket with the same code, its locks and its checks
    (`rk ticket-file ID [--again]`; OWN, the Approve's, with it: writer.through): its message, and the row read back
    from the state it wrote. Kept as an unsent item there (the monitor went into standby meanwhile), it is done too.
    """
    from . import writer

    p = writer.through(
        [writer.TICKET_FILE, tid, *(["--again"] if again else [])],
        timeout_s=FILING_MAX_S + writer.START_S,
        own=own,
    )
    item = find(store.load(), tid)
    row = {"row": status_text(cfg, item)} if item else {}
    if p.returncode == 0 and item and item.get("status") == "unsent":
        return True, writer.said(p), row
    if p.returncode == 0 and item and item.get("status") == "filed":
        return (
            True,
            writer.said(p),
            {"key": item.get("key", ""), "url": item.get("url", ""), **row},
        )
    return False, writer.said(p), row


def _file_and_record(
    cfg: dict, store, item: dict, by: str, run
) -> tuple[str, str, str]:
    """The JIRA write and its ledger entry (under the write lock): (key, url, a note on the assignee). A proposed
    ticket the ledger already records as filed (a filing cut off after its write) is not filed twice, and one `check`
    refuses is not sent (TicketUnfit).
    """
    done = next(
        (
            r
            for r in store.ledger()
            if r["kind"] == "ticket" and r["head"] == item["id"]
        ),
        None,
    )
    if done:
        return (
            done["ref"],
            _browse(cfg, done["ref"]),
            "it was filed before (the ledger has it)",
        )
    found = ""
    if item.get(
        "doubt_since"
    ):  # an earlier try may have gone in: JIRA is searched first
        try:
            found = find_ticket(cfg, item["title"], item["doubt_since"], run)
        except TicketError as exc:
            raise TicketInDoubt(
                f"an earlier filing is in doubt and JIRA could not be searched to settle it ({exc}); "
                "nothing was filed"
            ) from None
    if found:
        key, url = found, _browse(cfg, found)
        store.journal(
            "ticket",
            f"{item['id']}: an earlier filing that was in doubt did go in: {key}",
        )
    else:
        # A filing in doubt that JIRA does not have: check's refusal stops it here, as `approve` stops a waiting
        # proposed ticket that check refuses.
        why = unfit(item)
        if why:
            raise TicketUnfit(why)
        key, url = file_ticket(cfg, item, by, run)
        epic_note = link_epic(cfg, key, item, by, run)
    store.ledger_add(
        "ticket", item.get("pr") or 0, item["id"], key, note=item["title"][:120]
    )
    notes = [] if found else [epic_note]
    notes.append(assign(cfg, key, item.get("assignee") or "", run))
    return key, url, "; ".join(n for n in notes if n)


def decline(
    cfg: dict, st: dict, store, tid: str, by: str, why: str = ""
) -> tuple[bool, str]:
    """The operator declines proposed ticket `tid`: dropped from the card, kept in the state for KEEP_DAYS (so the
    same title is not proposed again meanwhile) and journaled."""
    item = find(st, tid)
    if not item:
        return False, f"no proposed ticket {tid}: nothing changed"
    first = _first_action(item)
    if first:
        return False, f"{first}: one action per proposed ticket, nothing changed"
    why = _line(why, 300)[:300]
    item.update(status="declined", decided_at=now(), decided_by=by, note=why)
    store.journal(
        "ticket",
        f"declined {tid}: {item['title']} (by {by})" + (f": {why}" if why else ""),
    )
    return (
        True,
        f"Declined proposed ticket {tid} ({item['title']}): dropped from the card, and journaled.",
    )


# --- the atlassian CLI --------------------------------------------------------------------------------------------------


def _cli(cfg: dict, run):
    """(the command's start, a runner returning (exit code, output))."""
    from . import atlassian

    binary = shutil.which(atlassian.BINARY) or atlassian.BINARY
    site = atlassian.jira_site(cfg)
    return [binary, "--site", site, "--json"], run or (
        lambda cmd: atlassian._run(cmd, cfg, timeout=CLI_TIMEOUT_S)
    )


def _first_line(text: str) -> str:
    return next(
        (ln.strip() for ln in (text or "").splitlines() if ln.strip()), "no output"
    )[:200]


def _json_objects(text: str):
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                v = json.loads(ln)
            except ValueError:
                continue
            if isinstance(v, dict):
                yield v
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            yield v
    except (ValueError, TypeError):
        pass


def _browse(cfg: dict, key: str) -> str:
    base = (_jira(cfg).get("url") or "").rstrip("/")
    return f"{base}/browse/{key}" if base and key else ""


def description(cfg: dict, item: dict, by: str) -> str:
    """The ticket's description: the proposed ticket's, then where it came from."""
    address = derived.github_address(cfg.get("repo") or "")
    tail = [
        f"Proposed by the rocKE monitor ({item.get('by') or 'the monitor'}, {_at(item.get('at'))}); "
        f"approved by {by} on {now()[:10]}."
    ]
    if item.get("pr"):
        tail.append(
            f"Found on PR #{item['pr']}"
            + (f": {address}/pull/{item['pr']}" if address else "")
            + "."
        )
    if item.get("source"):
        tail.append(f"Source: {item['source']}")
    return item["description"].rstrip() + "\n\n----\n" + "\n".join(tail)


def epic(cfg: dict) -> str:
    """The epic new tickets are linked to (features.ticket_creation.default_epic, config jira.default_epic); "" for
    none."""
    return str(_jira(cfg).get("default_epic") or "").strip()


def jira_project(cfg: dict) -> str:
    """The JIRA project tickets are filed in (config jira.project); "" when not set."""
    return str(_jira(cfg).get("project") or "").strip()


def _description_field(cfg: dict, item: dict, by: str) -> list[str]:
    """The description as the CLI's ADF field: its --description stores the text as it is (Markdown marks shown as
    typed)."""
    from . import adf

    return [
        "--field-json",
        "description=" + json.dumps(adf.convert(description(cfg, item, by))),
    ]


def file_ticket(cfg: dict, item: dict, by: str, run=None) -> tuple[str, str]:
    """`atlassian --site S --json jira issue create ...` for one proposed ticket: (the new issue's key, its address).
    With a default epic the issue is made without its description, which link_epic sets after the link. TicketError
    when the CLI fails or answers without a key."""
    j = _jira(cfg)
    project = jira_project(cfg)
    base, run = _cli(cfg, run)
    cmd = (
        base
        + [
            "jira",
            "issue",
            "create",
            "--project",
            project,
            "--summary",
            item["title"],
            "--type",
            item["type"],
        ]
        + ([] if epic(cfg) else _description_field(cfg, item, by))
    )
    pri = item.get("priority") or ""
    pm = j.get("priority_map") or {}
    pri = (
        (
            pm.get(pri)
            or next((v for k, v in pm.items() if k.lower() == pri.lower()), pri)
        )
        if pri
        else ""
    )
    if pri:
        cmd += ["--priority", str(pri)]
    if item.get("labels"):
        cmd += ["--labels", ",".join(item["labels"])]
    names = [c.strip() for c in (item.get("component") or "").split(",") if c.strip()]
    if (
        names
    ):  # "A, B" is two components (jira.component lists them so), never one named "A, B"
        cmd += [
            "--field-json",
            "components=" + json.dumps([{"name": c} for c in names]),
        ]
    for k, v in (j.get("extra_fields") or {}).items():
        if ticket_fields.blank(v):  # sets nothing, as check 6 reads it
            continue
        cmd += (
            ["--field-json", f"{k}={json.dumps(v)}"]
            if isinstance(v, (dict, list))
            else ["--field", f"{k}={v}"]
        )
    since = now()
    rc, text = run(cmd)
    if rc != 0:
        # A failure or a timeout after JIRA took the issue (the answer was lost): JIRA is searched for it before the
        # filing is called failed, so a retry never files it twice.
        why = f"atlassian jira issue create failed (exit {rc}): {_first_line(text)}"
        try:
            found = find_ticket(cfg, item["title"], since, run)
        except TicketError as exc:
            if REFUSED_BY_JIRA.search(text or ""):  # JIRA answered no: nothing was made
                raise TicketError(why) from None
            raise TicketInDoubt(
                f"{why}; JIRA could not be searched to check whether it was filed ({exc})"
            ) from None
        if found:
            return found, _browse(cfg, found)
        raise TicketError(why)
    key = next(
        (
            str(o["key"])
            for o in _json_objects(text)
            if re.fullmatch(r"[A-Z][A-Z0-9_]*-\d+", str(o.get("key")))
        ),
        "",
    )
    if not key:
        m = re.search(rf"\b({re.escape(project)}-\d+)\b", text or "")
        key = m.group(1) if m else ""
    if not key:
        raise TicketError(
            f"atlassian answered without an issue key: {_first_line(text)}"
        )
    return key, _browse(cfg, key)


def link_epic(cfg: dict, key: str, item: dict, by: str, run=None) -> str:
    """With a default epic: link the new issue to it (`atlassian jira issue link-epic KEY --epic EPIC`), then set its
    description (`jira issue update KEY`): a description set before the link can come out garbled, so it goes last,
    and it goes whether or not the link took. "" when done (or no epic is set); else a note for the row (the ticket
    stays filed)."""
    want = epic(cfg)
    if not want:
        return ""
    base, run = _cli(cfg, run)
    notes = []
    rc, text = run(base + ["jira", "issue", "link-epic", key, "--epic", want])
    if rc != 0:
        notes.append(f"not linked to the epic {want}: {_first_line(text)}")
    rc, text = run(
        base + ["jira", "issue", "update", key] + _description_field(cfg, item, by)
    )
    if rc != 0:
        notes.append(
            f"its description was not set ({_first_line(text)}): add it in JIRA by hand"
        )
    return "; ".join(notes)


def find_ticket(cfg: dict, title: str, since: str, run=None) -> str:
    """The key of an issue in the project whose summary is `title`, created at or after `since` (less a slack), or "":
    `atlassian --site S --json jira issue search --jql ...` (a read). The JQL names only the project and the age; the
    summary is compared here, whole, so no text search escaping is needed. TicketError when the search fails.
    """
    import datetime as dt

    from .store import parse_ts

    project = jira_project(cfg)
    base, run = _cli(cfg, run)
    try:
        age = (dt.datetime.now(dt.timezone.utc) - parse_ts(since)).total_seconds()
    except (TypeError, ValueError):
        age = 0
    minutes = max(5, int(age // 60) + DOUBT_SLACK_MIN + 1)
    jql = f'project = "{project}" AND created >= -{minutes}m ORDER BY created DESC'
    rc, text = run(
        base + ["jira", "issue", "search", "--jql", jql, "--max-results", "50"]
    )
    if rc != 0:
        raise TicketError(
            f"atlassian jira issue search failed (exit {rc}): {_first_line(text)}"
        )
    rows: list = []
    for line in [text] + (text or "").splitlines():
        try:
            v = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(v, dict):
            v = v.get("issues") or [v]
        rows += [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []
    want = (title or "").strip()
    for r in rows:
        summary = r.get("summary") or (r.get("fields") or {}).get("summary") or ""
        if str(summary).strip() == want and re.fullmatch(
            r"[A-Z][A-Z0-9_]*-\d+", str(r.get("key"))
        ):
            return str(r["key"])
    return ""


def account(cfg: dict, assignee: str, run=None) -> tuple[str, str]:
    """(the JIRA account id for a proposed ticket's assignee, why not): a GitHub login through jira.accounts, an
    email, or an account id as it is."""
    who = config.jira_accounts(cfg).get(assignee) or assignee
    if "@" not in who:
        if re.fullmatch(r"[0-9a-f]{24}|\d+:[0-9a-f-]{20,}|[0-9a-f-]{32,}", who):
            return who, ""
        return "", f"no JIRA account is known for {assignee} (jira.accounts)"
    base, run = _cli(cfg, run)
    rc, text = run(base + ["jira", "user", "get", who])
    if rc == 0:
        for o in _json_objects(text):
            acct = o.get("accountId") or o.get("account_id")
            if acct:
                return str(acct), ""
    return "", (
        f"JIRA did not find {who}: {_first_line(text)}"
        if rc
        else f"JIRA named no account for {who}"
    )


def assign(cfg: dict, key: str, assignee: str, run=None) -> str:
    """Assign the new issue (after its creation: the CLI's create takes no assignee). "" when done or nobody is named;
    else a note for the row (the ticket stays filed, unassigned)."""
    if not assignee:
        return ""
    acct, why = account(cfg, assignee, run)
    if not acct:
        return f"not assigned: {why}"
    base, run = _cli(cfg, run)
    rc, text = run(base + ["jira", "issue", "update", key, "--assignee", acct])
    return "" if rc == 0 else f"not assigned to {assignee}: {_first_line(text)}"


# --- what the operator sees -------------------------------------------------------------------------------------------


def lines(cfg: dict, st: dict) -> list[str]:
    """`rk tickets`: every proposed ticket on the card, one line each, and the count."""
    out = []
    for it in shown(st):
        out.append(
            f"{it['id']} {status_text(cfg, it)} · {it['type']} · {it['priority'] or 'no priority'} · "
            f"{it['assignee'] or 'unassigned'} · {it['title']}"
        )
        out.append(f"    {it['summary']}")
    n = len(waiting(st))
    out.append(
        f"({n} waiting for your OK: approve on the admin dashboard's Tickets card or with rk ticket-file ID; "
        "decline there or with rk ticket-decline ID)"
        if n
        else "(no proposed ticket waits)"
    )
    return out


def page(cfg: dict, item: dict) -> str:
    """The proposed ticket's full text, for the card's link (every value escaped)."""
    E = html.escape
    meta = " · ".join(
        E(x)
        for x in (
            item["type"],
            item["priority"] or "no priority",
            f"assignee {item['assignee'] or 'none'}",
            status_text(cfg, item),
        )
    )
    where = [
        f"Proposed {E(_at(item.get('at')))} by {E(item.get('by') or 'the monitor')}"
    ]
    if item.get("pr"):
        where.append(f"found on PR #{item['pr']}")
    if item.get("source"):
        where.append(f"source: {E(item['source'])}")
    if item.get("labels"):
        where.append(f"labels: {E(', '.join(item['labels']))}")
    if item.get("component"):
        where.append(f"component: {E(item['component'])}")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'>"
        f"<title>Proposed ticket {E(item['id'])}</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:"
        "60rem;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap;background:rgba(127,127,127,.08);"
        "padding:1rem;border-radius:6px}.m{color:#666}</style></head><body>"
        f"<h1>{E(item['title'])}</h1><p class='m'>{meta}</p><p><b>{E(item['summary'])}</b></p>"
        f"<pre>{E(item['description'])}</pre><p class='m'>{'; '.join(where)}.</p></body></html>"
    )
