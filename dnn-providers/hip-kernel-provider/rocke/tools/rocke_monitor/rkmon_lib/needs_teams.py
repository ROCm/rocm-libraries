# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Needs-you list's decisions, copied to the team's Teams channel (features.alerts, off by default), so the user
hears of them when nobody reads the orchestrator's chat. The dashboard's Needs-you list stays the
main channel; the Teams message is a second copy of it.

- What: the board's `asks` (report.board), the Needs-you items that are a decision for the user, each with a key that
  stays the same while it is the same decision (report.need_ask): PROPOSE, AUTOMERGE_RISK (when the auto-merge guard
  only alerts, or auto-merge is on again after that guard turned it off), AUTOMERGE_JUMP, PRIORITY_READY_EARLY,
  HOLD_STALE, HOLD_DISMISSED, HELD_MERGED, MERGED_OVER_RED, REVERT_OFFER, INCIDENT_AGED, BACKLOG_GROWING,
  READY_FOR_APPROVAL (one per PR), PRIORITY_REQUEST, IN_DEPTH_REVIEW, TICKET_DRAFTS (the proposed tickets waiting for
  the user's OK), NEEDS_PERSON (with no Claude session, each tick line that needs a person) and VERIFY_STALLED
  (reviews that have waited on verification for hours: one key for the whole stall). The other lines are
  news, not decisions, and are not sent: a critical alert the orchestrator acts on at once, an auto-merge that the
  auto-merge guard turns off itself.
- Once: each sent key stays in the state (`needs_teams.sent`) while its item is on the list. A key that leaves the list
  is forgotten, so an item that comes back is sent again; a new head, wave, request or incident state is a new key.
- Together: the items new at one full tick go out as one message, one line each (kind, PR link, the ask), at most
  MAX_LINES lines, then "and K more on the dashboard"; every item the message counts is sent.
- Not too often: at most one message per features.alerts.message_cooldown_minutes (config
  teams.message_cooldown_min: post.message_cooldown), on this kind's own clock (post.last_sent: a develop break
  message has its own). Items new in between go in the next message; a timer makes the first tick after the gap full.
- Never in the way: a failed post is a tick line, never a tick failure. Its items stay unsent and the next full tick
  tries again; a timer makes one due within the gap.
- In prototype and in standby (features.held): the message is made as live makes it, at the same times, and the one
  outward gate (post.outward_gate) keeps it as an unsent item instead of sending it (its content: post.card_text); its
  items count as sent, so each one is kept once, and no webhook is needed. `rk needs-you-teams` typed then is
  refused, naming the mode or standby.
- The text: its first line names the deployment that sends it (post.automated_line: its automated reviewer name, else
  its container name), then it is addressed to config `operator.github_login` by name, with no mention. The
  public-repo rules apply to every line (post.public; a line that breaks them reads "details on the dashboard"). The
  only links are the PRs on the repository and the published dashboard (`publish.html.url`); without that URL the
  message says "see the dashboard".

Each message is one ledger line (kind `teams`, ref `needs-you`) and one journal line. The webhook URL is read only by
post._teams_post and never printed. `rk needs-you-teams --dry-run` prints what the next message would say and writes
nothing; `rk needs-you-teams` sends it now (the gap does not apply; the feature must be on).
"""

from . import idle, post, reviewer_name, state_store
from . import config, derived
from .report import clip
from .store import now

FEATURE = "needs-you-alerts"
KEY = "needs_teams"  # st[KEY]: {"sent": {key: time}, "last_try", "error", "held"}; the last message: post.last_sent
KIND = post.KIND_NEEDS_YOU
MAX_LINES = 15
ASK_CHARS = 220
ON_DASHBOARD = "details on the dashboard"


def enabled(cfg: dict) -> bool:
    """The switch (features.py): off unless config `features` turns it on."""
    from . import features

    return features.on(cfg, FEATURE)


def unsent(st: dict, asks: list[dict]) -> list[dict]:
    """The decisions on the list not sent yet, in the list's order, each key once."""
    sent = (st.get(KEY) or {}).get("sent") or {}
    out, seen = [], set()
    for a in asks:
        if a["key"] not in sent and a["key"] not in seen:
            seen.add(a["key"])
            out.append(a)
    return out


def dashboard_url(cfg: dict) -> str | None:
    return ((cfg.get("publish") or {}).get("html") or {}).get("url") or None


def rows(cfg: dict, items: list[dict]) -> list[dict]:
    """The message's lines, at most MAX_LINES: kind, PR, PR link and the ask (the item's Needs-you line as the team sees
    it, report.need_ask's `team`, cut to ASK_CHARS; one that breaks the public-repo rules reads ON_DASHBOARD).
    """
    out = []
    for a in items[:MAX_LINES]:
        ask = clip(" ".join(a["team"].split()), ASK_CHARS)
        n = a.get("pr")
        out.append(
            {
                "kind": a["kind"],
                "pr": n,
                "url": f"{derived.github_address(cfg['repo'])}/pull/{n}" if n else None,
                "ask": ON_DASHBOARD if post.public(ask) else ask,
            }
        )
    return out


def headline(cfg: dict, items: list[dict]) -> str:
    who = config.operator_login(cfg) or "Operator"
    return f"{who}: {len(items)} decision{'s' if len(items) != 1 else ''} for you on the Needs-you list"


def footer(cfg: dict, items: list[dict]) -> list[str]:
    more = len(items) - MAX_LINES
    url = dashboard_url(cfg)
    return ([f"…and {more} more on the dashboard."] if more > 0 else []) + [
        (
            f"The dashboard has the full list: {url}"
            if url
            else "See the dashboard for the full list."
        )
    ]


def message(cfg: dict, items: list[dict]) -> dict:
    """The Teams message for `items`: one card, a line per item."""
    lines = [
        {
            "type": "TextBlock",
            "wrap": True,
            "text": f"**{r['kind']}** · "
            + (f"[#{r['pr']}]({r['url']}) · " if r["url"] else "")
            + r["ask"],
        }
        for r in rows(cfg, items)
    ]
    body = (
        [
            {
                "type": "TextBlock",
                "text": headline(cfg, items),
                "weight": "Bolder",
                "size": "Medium",
                "wrap": True,
            }
        ]
        + lines
        + [
            {"type": "TextBlock", "wrap": True, "isSubtle": True, "text": t}
            for t in footer(cfg, items)
        ]
    )
    url = dashboard_url(cfg)
    return post.teams_message(
        reviewer_name.name(cfg),
        post.labelled(post.message_label(cfg), headline(cfg, items)),
        body,
        (
            [{"type": "Action.OpenUrl", "title": "Open the dashboard", "url": url}]
            if url
            else None
        ),
    )


def text(cfg: dict, items: list[dict]) -> list[str]:
    """The message for `items` as plain lines (`rk needs-you-teams --dry-run`)."""
    return (
        [headline(cfg, items)]
        + [
            f"- {r['kind']} "
            + (f"#{r['pr']} ({r['url']}) " if r["url"] else "")
            + f"· {r['ask']}"
            for r in rows(cfg, items)
        ]
        + footer(cfg, items)
    )


def send(
    cfg: dict,
    st: dict,
    store,
    asks: list[dict],
    force: bool = False,
    own: bool | None = None,
) -> list[str]:
    """Forget the sent keys no longer on the list, then send the new decisions as one message, unless the last
    message is younger than the gap (`force`: send now), or, while alerts do not send, keep it as an unsent item
    (OWN: the monitor's own message, post.outward_gate). Returns the tick's lines."""
    b = st.setdefault(KEY, {})
    sent = b.setdefault("sent", {})
    live = {a["key"] for a in asks}
    for k in [k for k in sent if k not in live]:
        del sent[k]
    new = unsent(st, asks)
    b["held"] = len(new)
    if not new:
        return []
    wait = post.cooldown_until(cfg, st, KIND)
    if not force and wait:
        return [
            f"NEEDS_YOU_TEAMS waiting: {len(new)} new item{'s' if len(new) != 1 else ''} go in the next message, "
            f"at {wait[11:16]}Z"
        ]
    b["last_try"] = now()
    what = f"the Needs-you message ({len(new)} item{'s' if len(new) != 1 else ''})"
    try:
        held = post.outward_check(cfg, st, what, (FEATURE,), own=own)
        if not held and not post.teams_hook_set(cfg):
            raise post.PostRefused(
                f"no Teams webhook configured in {post._teams_hook(cfg)}"
            )
        post._rate(cfg, store)
        card = message(cfg, new)
        try:
            post.outward_gate(
                cfg,
                st,
                what,
                (FEATURE,),
                post.unsent_item(
                    state_store.ALERT, post.teams_channel(cfg), post.card_text(card)
                ),
                own=own,
                store=store,
            )
        except post.Held as kept:
            _sent(st, store, new)
            return [f"NEEDS_YOU_TEAMS {kept}"]
        code = post.teams_post(cfg, card, KIND)
    except (post.PostRefused, OSError) as exc:
        b["error"] = str(exc)[:200]
        return [
            f"(needs-you Teams message not sent: {b['error']}; its {len(new)} item"
            f"{'s wait' if len(new) != 1 else ' waits'} for the next full tick)"
        ]
    t = _sent(st, store, new)
    store.ledger_add(
        "teams",
        0,
        f"needs-you-{t}",
        "needs-you",
        note=f"HTTP {code}; {len(new)} item(s): "
        + " ".join(a["key"] for a in new)[:600],
    )
    store.journal(
        "post",
        f"Teams Needs-you message: {len(new)} item(s) (HTTP {code})",
    )
    return [
        f"NEEDS_YOU_TEAMS sent {len(new)} item{'s' if len(new) != 1 else ''} to the team chat (HTTP {code})"
    ]


def _sent(st: dict, store, new: list[dict]) -> str:
    """The items NEW marked sent, and the message's clock, now (a message sent, or kept as an unsent item); saved at
    once (the caller holds the state lock): a tick killed after the send but before its own save never sends the same
    items again. The time."""
    b = st.setdefault(KEY, {})
    t = now()
    for a in new:
        b.setdefault("sent", {})[a["key"]] = t
    post.mark_sent(st, KIND, t)
    b.update(error=None, held=0)
    store.save(st)
    return t


def tick(cfg: dict, st: dict, store, board: dict) -> list[str]:
    """The full tick's step, after the board: nothing while the feature is off or not set up (features.tick_reports
    names what it lacks). The message is the monitor's own (send's OWN)."""
    from . import features

    if not features.active(cfg, FEATURE):
        return []
    return send(cfg, st, store, board.get("asks") or [], own=True)


def command(cfg: dict, st: dict, store, board: dict, dry_run: bool) -> list[str]:
    """`rk needs-you-teams [--dry-run]`. The dry run prints the message the new decisions would make and writes
    nothing; without it the message goes now (the feature must be on). While alerts do not send, one you typed is
    refused, naming the mode or standby, and the monitor's own is kept as an unsent item (send).
    """
    if dry_run:
        items = unsent(st, board.get("asks") or [])
        b = st.get(KEY) or {}
        state = (
            f"needs-you alerts are {'on' if enabled(cfg) else 'off'}; message label {post.message_label(cfg) or 'none'}; "
            f"webhook {'set' if post.teams_hook_set(cfg) else 'NOT set'}; last message "
            f"{post.last_sent(st, KIND) or 'none yet'}; one message per {post.message_cooldown(cfg):g} min"
        )
        return ["DRY RUN (nothing sent, nothing written): " + state] + (
            text(cfg, items) if items else ["nothing new to send"]
        )
    post.outward_check(cfg, st, "the Needs-you message", (FEATURE,))
    return send(cfg, st, store, board.get("asks") or [], force=True) or [
        "nothing new to send"
    ]


@idle.timer("needs-you teams")
def _due(cfg: dict, st: dict, store):
    """When the held decisions may go: the gap after the last message (or after a failed try)."""
    if not enabled(cfg):
        return
    b = st.get(KEY) or {}
    since = b.get("last_try") if b.get("error") else post.last_sent(st, KIND)
    if b.get("held") and since:
        yield "needs-you Teams message", idle.plus(since, post.message_cooldown(cfg))
