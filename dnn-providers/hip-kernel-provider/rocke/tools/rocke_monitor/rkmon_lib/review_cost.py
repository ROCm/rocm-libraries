# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Each review's start and cost, for comparing the review pipeline with a mirror of it (a monitor beside the live one
that sends nothing and reviews the same PRs at the same heads: rkmon_lib/mirror.py).

- state/mirror_feed.jsonl: one line per review start (review.start_review), {pr, head, round, kind, again, merge_only,
  at}. The mirror mounts it read-only and starts its own review of that PR at that head.
- state/review_cost.jsonl: one line per post through `rk post N FILE --premises SPEC` (premises.finish), and per
  review kept unposted while review posting is off (`unposted`: true, its post the kept file), there or by a plain
  `rk post N FILE` (post.keep_as_draft: no spec, so no tokens of the verifier or the code quality pass): the PR,
  round, head and post; the live draft's tokens, requests and minutes (a drafter binding's draft record, else the local
  worker's minutes from its stats_agent.json: it counts no tokens); the shadow draft's, when one drafted that head; the
  verifier's and the code quality pass's tokens, from the spec's optional COST ({"verifier": N, "quality": N}: the
  totals the subagents' task notifications report); and the minutes from the review's start (its feed line) to the
  post. A review awaiting verification that `rk review-close` closed without a post has a line too (`closed`: true,
  its post the URL of our account's review that stands for it; record_close), so that `rk spend` prices it as a
  review of its own.
Neither ever holds a review or a post: a line that cannot be written is said in the journal or the post's output.
"""

from __future__ import annotations

import datetime as dt
import json

from . import bindings, draft_stage
from .snapshot import same_commit
from .store import now

FEED = "mirror_feed.jsonl"
COSTS = "review_cost.jsonl"
COST_KEYS = ("verifier", "quality")


def feed(
    st: dict, store, n: int, kind: str, again: bool = False, merge_only: bool = False
) -> None:
    """PR n's review started at its state head: its feed line."""
    from . import prep  # a call-time import: prep imports review, which calls this

    try:
        line = {
            "pr": n,
            "head": st["prs"][str(n)]["head"],
            "round": prep.default_round(st, n),
            "kind": kind,
            "again": bool(again),
            "merge_only": bool(merge_only),
            "at": now(),
        }
        store.append_jsonl(FEED, [line])
    except (OSError, KeyError, ValueError) as exc:
        store.journal(
            "review",
            f"#{n} mirror feed line not written: {type(exc).__name__}: {str(exc)[:120]}",
        )


def check_cost(cost) -> dict:
    """The spec's COST, checked: {"verifier": N, "quality": N}, each a count of tokens (or null when unknown), no other
    key. ValueError names what is wrong."""
    if not isinstance(cost, dict):
        raise ValueError('COST: {"verifier": tokens, "quality": tokens}')
    unknown = sorted(set(cost) - set(COST_KEYS))
    if unknown:
        raise ValueError(
            f"COST: unknown {', '.join(unknown)} (the keys are {', '.join(COST_KEYS)})"
        )
    for k, v in cost.items():
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise ValueError(f"COST {k}: a count of tokens, or null")
    return cost


def _minutes(a: str, b: str) -> float | None:
    try:
        t = [dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") for x in (a, b)]
    except (TypeError, ValueError):
        return None
    return round((t[1] - t[0]).total_seconds() / 60.0, 1)


def _draft(d: dict | None) -> dict | None:
    if not d:
        return None
    return {
        "by": d.get("binding") or "",
        "role": d.get("role") or "drafter",
        "method": d.get("method") or "",
        "status": d.get("status") or "",
        "tokens": d.get("tokens"),
        "requests": d.get("requests"),
        "wall_min": round(float(d.get("wall_s") or 0) / 60.0, 1),
    }


def drafts(cfg: dict, st: dict, n: int, head: str) -> tuple[dict | None, dict | None]:
    """The live draft and the shadow draft of PR n at `head` that came to verification (ready or missing: never a
    dropped one), from a binding's draft records (state drafts: draft_stage.draft_for). The live one is the draft the
    review was staged with: none, with no draft to count, when its record says it was staged with no drafter; with no
    record (the local worker's stage keeps none), the local worker's minutes from its stats_agent.json.
    """
    live = draft_stage.draft_for(
        st, n, head, statuses=draft_stage.STAGED, role="drafter"
    )
    shadow = draft_stage.draft_for(st, n, head, role="shadow_drafter")
    out = _draft(live)
    if live and live["status"] == "none":  # no drafter: no draft to count
        out = {
            "by": bindings.NONE,
            "role": "drafter",
            "method": draft_stage.NONE.name,
            "status": "none",
            "tokens": None,
            "requests": None,
            "wall_min": None,
        }
    if out is None:
        try:
            stats = json.loads(
                (draft_stage.folder(cfg, n) / "stats_agent.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            stats = {}
        out = {
            "by": bindings.worker_tag(cfg),
            "role": "drafter",
            "method": "local-worker",
            "status": "ready" if stats else "",
            "tokens": None,
            "requests": None,
            "wall_min": (
                round(float(stats.get("wall_s") or 0) / 60.0, 1) if stats else None
            ),
        }
    return out, _draft(shadow)


def started(store, n: int, head: str) -> str:
    """When PR n's review at `head` started: its last feed line's time ("" without one)."""
    at = ""
    for r in store.read_jsonl(FEED, lenient=True):
        if r.get("pr") == n and same_commit(r.get("head"), head):
            at = r.get("at") or at
    return at


def _write(
    cfg: dict,
    st: dict,
    store,
    n: int,
    rnd: str | None,
    head: str,
    post: str,
    kept: bool,
    cost: dict,
    closed: bool = False,
) -> dict:
    """PR n's review at `head`, round RND, posted (its copy POST), `kept` unposted (POST the kept file) or `closed`
    without a post (POST the URL of the review that stands for it): its line, added to state/review_cost.jsonl, with
    COST's tokens of the verifier and the code quality pass.
    """
    draft, shadow = drafts(cfg, st, n, head)
    t0, t1 = started(store, n, head), now()
    line = {
        "pr": n,
        "round": rnd,
        "head": head,
        "at": t1,
        "post": post,
        **({"unposted": True} if kept else {}),
        **({"closed": True} if closed else {}),
        "draft": draft,
        "shadow": shadow,
        "verifier": cost.get("verifier"),
        "quality": cost.get("quality"),
        "started": t0 or None,
        "review_min": _minutes(t0, t1) if t0 else None,
    }
    store.append_jsonl(COSTS, [line])
    return line


def record_keep(
    cfg: dict, st: dict, store, n: int, rnd: str, head: str, post: str
) -> None:
    """A review of PR n kept unposted by `rk post` (post.keep_as_draft) at `head`, round RND, its kept file POST: its
    line, with no tokens of the verifier or the code quality pass (no spec gives them). A line that cannot be written
    is said in the journal."""
    try:
        _write(cfg, st, store, n, rnd, head, post, True, {})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        store.journal(
            "post",
            f"#{n} review cost line not written: {type(exc).__name__}: {str(exc)[:120]}",
        )


def record_close(
    cfg: dict, st: dict, store, n: int, rnd: str, head: str, url: str
) -> None:
    """A review of PR n awaiting verification at `head`, round RND, that `rk review-close` closed without a post
    (review.close_posted), our account's review at URL standing for it: its line (`closed`), with no tokens of the
    verifier or the code quality pass (no spec gives them). A line that cannot be written is said in the journal.
    """
    try:
        _write(cfg, st, store, n, rnd, head, url, False, {}, closed=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        store.journal(
            "review",
            f"#{n} review cost line not written: {type(exc).__name__}: {str(exc)[:120]}",
        )


def record(
    cfg: dict, store, n: int, spec: dict, ready: dict, kept: bool = False
) -> list[str]:
    """After a post through `rk post --premises`, or a review `kept` unposted there (review posting off): its line of
    state/review_cost.jsonl; the line to print. A review the writer's `rk post` kept has its line from that keep (the
    same PR, kept file and head: post.keep_as_draft), and gets no second one."""
    head, post = ready.get("sha") or "", str(ready.get("copy") or "")
    if kept and any(
        r.get("pr") == n and r.get("post") == post and same_commit(r.get("head"), head)
        for r in store.read_jsonl(COSTS, lenient=True)
    ):
        return [f"cost recorded when the review was kept: {post}"]
    line = _write(
        cfg,
        store.load(),
        store,
        n,
        spec.get("ROUND"),
        head,
        post,
        kept,
        spec.get("COST") or {},
    )
    draft = line["draft"]
    tokens = draft.get("tokens")
    said = [
        f"draft {draft.get('by') or '-'} {tokens if tokens is not None else 'uncounted'}"
    ]
    said += [
        f"{k} {line[k] if line[k] is not None else 'not given'}" for k in COST_KEYS
    ]
    return [
        "cost recorded: "
        + ", ".join(said)
        + (
            f"; {line['review_min']} min from the start"
            if line["review_min"] is not None
            else ""
        )
    ]
