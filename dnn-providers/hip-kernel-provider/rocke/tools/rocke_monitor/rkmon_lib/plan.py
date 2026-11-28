# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What the next tick should do, in priority order. While a feature is in prototype, or the monitor in standby, what
would go outward is planned as in live, and the one outward gate keeps it as an unsent item (post.outward_gate).

Priorities: 0 develop health, and reviews of PRs that fix a develop break or that the
user floated to the top (`rk prioritize`); 1 the
merge-queue head, and final reviews: follow-ups on PRs close to merge (approved, or waiting on the user's approval,
with no CHANGES_REQUESTED), and on a held PR's fix; 2 follow-ups on other
near-ready PRs; 3 first rounds on new PRs; 4 other follow-ups. Critical items are reported to the user immediately.

A follow-up waits while our last review on its PR (posted, or kept unposted) is under `followup_min_gap_min` minutes
old: its item carries `gap_until`, and its line says "(gap: after HH:MMZ)". A push of merges only never waits
(`merge_only` on its item): its checks decide whether anything is posted (review.merge_only_flags).
"""

from __future__ import annotations

import datetime as dt

from . import idle
from . import config, registry
from .post import kept_unposted, taken
from .queue import PASS, USER_PRIORITY, approvals, fixes_develop
from .snapshot import covers
from .store import parse_ts, settle_left

# One follow-up review per PR every `followup_min_gap_min` minutes (0: no gap). The PRs config
# `followup_gap_exempt` names never wait, such as the develop fixes, the P0 reviews of st["develop_fixes"], which keep
# their short settle (`settle_develop_fix_min`): unblocking develop comes first.
CI_UPDATE = "ci-update"  # the ledger note of a post that is no review (post.post)


def last_review_posts(ledger: list[dict]) -> dict[int, str | None]:
    """PR -> the time of our last review post on it (a first round or a follow-up), from the ledger; None when every post
    of ours on it is a CI update. A CI update is no review: `rk post` notes one (a post at a head our review already
    covers, L-209), and a second post at a head an earlier post had is one too. Notices have ledger kinds of their
    own."""
    out: dict[int, str | None] = {}
    heads: dict[int, set] = {}
    for r in ledger:
        if r.get("kind") != "post":
            continue
        n = r.get("pr")
        seen = heads.setdefault(n, set())
        again = r.get("head") in seen
        seen.add(r.get("head"))
        out.setdefault(n, None)
        if r.get("note") != CI_UPDATE and not again:
            out[n] = r["ts"]
    return out


def last_review_time(n: int, ours: dict | None, last_posts: dict) -> str | None:
    """When our last review of PR n, posted or kept unposted, was made: one rule for the follow-up gap (gap_until) and
    `rk facts`. It is the time of our last review post on it in `last_posts` (last_review_posts), or our review's own
    (`ours`, its record rec["ours"]) for a PR the ledger never saw and for a review kept unposted (post.save_draft),
    which no ledger entry dates; the later of the two for a review of our account that `rk review-close` took
    (post.taken), which no ledger entry dates either. None when neither gives one."""
    ours = ours or {}
    if n not in last_posts or kept_unposted(ours):
        return ours.get("at")
    if taken(ours):
        return max(last_posts[n] or "", ours.get("at") or "") or None
    return last_posts[n]


def gap_exempt(cfg: dict, st: dict, n: int) -> bool:
    """PR n's follow-ups never wait for the gap. Config `followup_gap_exempt` lists "develop_fix" (a PR that fixes a
    develop break), "priority" (a PR the user floated to the top) and PR numbers."""
    why = (st.get("develop_fixes") or {}).get(str(n)) or []
    for e in registry.value(cfg, "followup_gap_exempt") or ():
        if (
            (e == "develop_fix" and fixes_develop(why))
            or str(e).lstrip("#") == str(n)
            or (e == "priority" and any(w.startswith(USER_PRIORITY) for w in why))
        ):
            return True
    return False


def merge_only_push(rec: dict) -> dict | None:
    """The tick's merge-only finding (review.mark_merges) when it is for PR rec's current head against our reviewed
    head, else None."""
    mc = rec.get("merge_check") or {}
    ours = rec.get("ours") or {}
    if (
        mc.get("merge_only")
        and mc.get("old") == ours.get("head")
        and mc.get("new") == rec.get("head")
    ):
        return mc
    return None


def gap_until(cfg: dict, st: dict, rec: dict, last_posts: dict) -> str | None:
    """When a follow-up on PR rec may start (UTC, as the state writes times) while our last review on it is under
    `followup_min_gap_min` minutes old, else None. One rule for the plan, the review driver and `rk review-start`; the
    review's time is last_review_time's, from `last_posts` (last_review_posts). A merge-only push never waits: its
    checks run at once, and one that flags something may be landing (an armed auto-merge).
    """
    gap = registry.value(cfg, "followup_min_gap_min")
    n = rec.get("n")
    if not gap or gap_exempt(cfg, st, n) or merge_only_push(rec):
        return None
    last = last_review_time(n, rec.get("ours"), last_posts)
    if not last:
        return None
    try:
        end = parse_ts(last) + dt.timedelta(minutes=gap)
    except ValueError:
        return None
    if end <= dt.datetime.now(dt.timezone.utc):
        return None
    return end.strftime("%Y-%m-%dT%H:%M:%SZ")


def gap_label(cfg: dict) -> str:
    """The gap as the board and the page say it: "2-h", or "90-min"."""
    m = registry.value(cfg, "followup_min_gap_min")
    return f"{m // 60}-h" if m and m % 60 == 0 else f"{m}-min"


def _item(
    prio: int, kind: str, text: str, pr: int | None = None, critical: bool = False
) -> dict:
    return {"prio": prio, "kind": kind, "pr": pr, "text": text, "critical": critical}


def review_kind(rec: dict) -> str | None:
    """ "FIRST_ROUND" or "FOLLOW_UP" when PR rec waits for our review of its head, else None: open, not a draft, and
    never reviewed, or reviewed at a head that does not cover the current one."""
    if rec.get("status") != "open" or rec.get("draft"):
        return None
    ours = rec.get("ours")
    if not ours:
        return "FIRST_ROUND"
    return "FOLLOW_UP" if ours.get("head") and not covers(ours, rec["head"]) else None


def make_plan(
    cfg: dict,
    st: dict,
    events: list[dict],
    queue: list[dict],
    last_posts: dict | None = None,
) -> list[dict]:
    """`last_posts` (last_review_posts of the ledger) dates the follow-up gap; without it no item waits for the gap."""
    items = []
    for e in events:
        if e["kind"] == "DEVELOP_RED":
            items.append(
                _item(
                    0,
                    "INCIDENT",
                    f"develop red at {e['sha']} in rocKE jobs {e['rocke_jobs']}: freeze, "
                    f"bisect, JIRA P1 to the culprit's author, swarm, tell the user",
                    critical=True,
                )
            )
        elif e["kind"] == "DEVELOP_RED_OTHER":
            items.append(
                _item(
                    0,
                    "ALERT",
                    f"develop red outside rocKE at {e['sha']}: {e['workflows']}",
                    critical=True,
                )
            )
    pend = st["develop"].get("pending_sanity") or []
    if pend:
        items.append(
            _item(
                0,
                "SANITY",
                f"develop sanity on {pend[-1][:10]} (covers {len(pend)} landing(s))",
            )
        )
    rows = {r["n"]: r for r in queue}
    fixes = st.get("develop_fixes") or {}
    for n, rec in sorted(st["prs"].items(), key=lambda kv: kv[1]["size"]):
        kind = review_kind(rec)
        if not kind:
            continue
        fix = fixes.get(n)
        # A develop fix is reviewed first and waits less for its head to settle; a burst
        # of pushes waits longer (lessons L-534). One rule for the plan, the driver and `rk review-start`.
        left, waiting = settle_left(cfg, st, rec)
        wait = f" (wait: {waiting})" if left > 0 else ""
        why = (
            f"{'develop fix' if fixes_develop(fix) else 'top priority'} ({', '.join(fix)}): "
            if fix
            else ""
        )
        ours = rec.get("ours")
        row = rows.get(rec["n"])
        near = bool(row) and (
            row["gates"]["G2"][0] == PASS or row["gates"]["G3"][0] == PASS
        )
        if kind == "FIRST_ROUND":
            items.append(
                _item(
                    0 if fix else 3,
                    "FIRST_ROUND",
                    why + f"first-round review of #{n} ({rec['author']}, "
                    f"{rec['size']} lines)" + wait,
                    pr=rec["n"],
                )
            )
        else:
            a = approvals(cfg, st, rec)
            # A held PR's fix is a final review too: the user re-approves it once we have checked it.
            hr = (st.get("hold_reviews") or {}).get(n)
            held_fix = bool(hr) and (
                a["changes"] == [config.operator_login(cfg)]
                or bool(hr.get("dismissed"))
            )
            final = held_fix or (
                not a["changes"]
                and (a["approved"] or bool(a["mine"]) or a["requested"])
            )
            until = (
                gap_until(cfg, st, rec, last_posts) if last_posts is not None else None
            )
            mo = merge_only_push(rec)
            item = _item(
                0 if fix else 1 if final else 2 if near else 4,
                "FOLLOW_UP",
                why
                + (
                    "final review (held PR's fix): "
                    if held_fix and not fix
                    else "final review (close to merge): " if final and not fix else ""
                )
                + f"follow-up on #{n}: head {ours['head'][:8]} -> {rec['head'][:8]}, "
                + (
                    "merge-only: its checks decide, nothing is posted when they are clean"
                    if mo
                    else "classify by range-diff"
                )
                + wait
                + (f" (gap: after {until[11:16]}Z)" if until else ""),
                pr=rec["n"],
            )
            if until:
                item["gap_until"] = until
            if mo:
                item["merge_only"] = True
            items.append(item)
    dev = (st["develop"]["sha"] or "")[:10]
    batches = st.get("batches") or []
    if batches:
        b = batches[0]
        names = " → ".join(f"#{n}" for n in b["members"])
        if b.get("status") != "validated":
            kind = "BATCH_GATE" if len(b["members"]) > 1 else "GATE"
            items.append(
                _item(
                    1,
                    kind,
                    f"{'batch ' if len(b['members']) > 1 else ''}gate for {names} on develop {dev}"
                    + (
                        " (each prefix too: goldens/ABI involved)"
                        if b["check_prefixes"] and len(b["members"]) > 1
                        else ""
                    ),
                    pr=b["members"][0],
                )
            )
        else:
            from . import features

            cleared = set(st.get("cleared", []))
            # Armed only with auto-merge on (features.merge_queue enabled, auto_merge enabled) and every member cleared
            # (in prototype the arming is kept as an unsent item: post.arm); otherwise (auto_merge hints_only) the order
            # is proposed to the user.
            arm = features.arms(cfg) and all(n in cleared for n in b["members"])
            items.append(
                _item(
                    1,
                    "ARM" if arm else "PROPOSE",
                    f"{'arm, one at a time in order:' if arm else 'propose to the user, merge in order:'} "
                    f"{names} (validated together on develop {dev})",
                    pr=b["members"][0],
                )
            )
    for lane, L in (st.get("abi_lanes") or {}).items():
        if L["head"] is None or not L["groups"]:
            continue
        head = L["groups"][L["head"]]
        for n in head["members"]:
            g9 = (next((r for r in queue if r["n"] == n), {}).get("gates") or {}).get(
                "G9"
            )
            if g9 and g9[0] == "fail" and "lane head" in g9[1]:
                items.append(_item(1, "ABI_RENUMBER", f"#{n}: {g9[1]}", pr=n))
    return sorted(items, key=lambda i: (i["prio"], i["pr"] or 0))


# --- timers of the change fingerprint (idle.py) -----------------------------------------------------------------------


@idle.timer("settle")
def _settle_due(cfg: dict, st: dict, store):
    """When each head that waits for our review settles (settle_left): the tick that may start its review."""
    t = dt.datetime.now(dt.timezone.utc)
    for k, rec in (st.get("prs") or {}).items():
        if review_kind(rec):
            left, _ = settle_left(cfg, st, rec)
            if left > 0:
                yield f"settle #{k}", (t + dt.timedelta(minutes=left)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )


@idle.timer("follow-up gap")
def _gap_due(cfg: dict, st: dict, store):
    """When each follow-up that waits for the gap after our last review of its PR may start (gap_until)."""
    last = last_review_posts(store.ledger())
    for k, rec in (st.get("prs") or {}).items():
        if review_kind(rec) == "FOLLOW_UP":
            until = gap_until(cfg, st, rec, last)
            if until:
                yield f"gap #{k}", until
