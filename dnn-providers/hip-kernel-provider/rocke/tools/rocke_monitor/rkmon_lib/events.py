# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Diff the stored state against a fresh snapshot, emit events, and fold the snapshot into the state.

Pure apart from the injected `resolve_gone` callback (which asks GitHub what happened to a PR that left the
open in-scope list), so the whole diff is unit-testable with fixtures.
"""

from __future__ import annotations

import re

from .store import now

PR_FIELDS = (
    "title",
    "author",
    "draft",
    "labels",
    "base",
    "head_ref",
    "head",
    "created_at",
    "updated_at",
    "size",
    "files",
    "mergeable",
    "merge_state",
    "review_decision",
    "ci_rollup",
    "reviews",
    "auto_merge",
    "review_requests",
    "closes",
    "head_committed_at",
)
MERGED_PR = re.compile(r"\(#(\d+)\)\s*$")
PUSH_TIMES_KEPT = 8  # per PR: enough for the burst settle (store.settle_left)


def _ev(kind: str, pr: int | None = None, **detail) -> dict:
    return {"kind": kind, "pr": pr, **detail}


def apply_push(rec: dict, head: str, ts: str) -> dict:
    """PR record `rec` moved to `head`, seen at `ts`: what a push resets, and its PUSH event. One home for the poll and
    for a review start that finds the state's head out of date (review.start_review)."""
    ev = _ev("PUSH", rec["n"], old=(rec.get("head") or "")[:10], new=head[:10])
    rec["head"], rec["head_seen_at"] = head, ts
    # When each push was seen: a burst of them stretches the settle (store.settle_left, lessons L-534). A PR's first
    # sighting is no push: its head may be days old when a label brings it into scope.
    rec["push_times"] = (rec.get("push_times") or [])[-(PUSH_TIMES_KEPT - 1) :] + [ts]
    rec["gates"] = None  # gate results are per head
    return ev


def take_pending(st: dict) -> list[dict]:
    """Events found between ticks, for the next tick to report with its own. A review start that takes a PR's live head
    records the push there, and the poll then sees no change; the merge queue still acts on it (an armed head moved).
    """
    return st.pop("pending_events", None) or []


def diff_and_apply(
    st: dict, snap: dict, resolve_gone, ts: str | None = None
) -> list[dict]:
    ts = ts or now()
    events: list[dict] = []
    seen = set()
    for p in snap["prs"]:
        key = str(p["n"])
        seen.add(key)
        old = st["prs"].get(key)
        if old is None or old.get("status") in ("merged", "closed", "out-of-scope"):
            events.append(
                _ev(
                    "NEW_PR" if not p["draft"] else "NEW_DRAFT",
                    p["n"],
                    author=p["author"],
                )
            )
            rec = {f: p.get(f) for f in PR_FIELDS}
            # A PR back after it was merged, closed or out of scope keeps our last review of it: a post is read again
            # from GitHub, but a review kept unposted is in this record alone.
            rec.update(
                n=p["n"],
                first_seen=ts,
                head_seen_at=ts,
                status="open",
                ours=(old or {}).get("ours"),
                gates=None,
            )
            st["prs"][key] = rec
            continue
        if old["draft"] and not p["draft"]:
            events.append(_ev("READY", p["n"]))
        if not old["draft"] and p["draft"]:
            events.append(_ev("TO_DRAFT", p["n"]))
        if old["head"] != p["head"]:
            events.append(apply_push(old, p["head"], ts))
        elif old["ci_rollup"] != p["ci_rollup"]:
            events.append(_ev("CI", p["n"], state=p["ci_rollup"], was=old["ci_rollup"]))
        if old["review_decision"] != p["review_decision"]:
            events.append(
                _ev(
                    "DECISION",
                    p["n"],
                    state=p["review_decision"],
                    was=old["review_decision"],
                )
            )
        old_rv = {
            (r["login"], r["state"], r["commit"]) for r in old.get("reviews") or []
        }
        for r in p["reviews"]:
            if (r["login"], r["state"], r["commit"]) not in old_rv:
                events.append(
                    _ev(
                        "HUMAN_REVIEW",
                        p["n"],
                        login=r["login"],
                        state=r["state"],
                        commit=(r["commit"] or "")[:8],
                    )
                )
        # Against the last definite answer: GitHub reports UNKNOWN while it recomputes after develop moves, and
        # CONFLICTING -> UNKNOWN -> CONFLICTING is no new conflict.
        known = (
            old["mergeable"]
            if old["mergeable"] != "UNKNOWN"
            else old.get("mergeable_known")
        )
        if known != "CONFLICTING" and p["mergeable"] == "CONFLICTING":
            events.append(_ev("CONFLICT", p["n"]))
        old["mergeable_known"] = (
            p["mergeable"] if p["mergeable"] != "UNKNOWN" else known
        )
        if bool(old.get("auto_merge")) != bool(p.get("auto_merge")):
            events.append(
                _ev(
                    "AUTOMERGE_ON" if p.get("auto_merge") else "AUTOMERGE_OFF",
                    p["n"],
                    by=(p.get("auto_merge") or {}).get("by"),
                )
            )
        for f in PR_FIELDS:
            old[f] = p.get(f)
        old["status"] = "open"
    for key, old in st["prs"].items():
        if key in seen or old.get("status") in ("merged", "closed", "out-of-scope"):
            continue
        fate = resolve_gone(
            int(key)
        )  # {"state": "merged"|"closed"|"open", "merged_at", "merge_sha", "by"}
        kind = {"merged": "MERGED", "closed": "CLOSED"}.get(
            fate["state"], "OUT_OF_SCOPE"
        )
        events.append(
            _ev(kind, int(key), **{k: v for k, v in fate.items() if k != "state" and v})
        )
        old["status"] = {"MERGED": "merged", "CLOSED": "closed"}.get(
            kind, "out-of-scope"
        )
        old["gone_at"] = ts
        if fate.get(
            "merge_sha"
        ):  # the develop guard tells reds from before a merged fix (guard.stale_red)
            old["merge_sha"] = fate["merge_sha"]
    dev = snap.get("develop")
    if dev:
        d = st["develop"]
        if d.get("sha") and d["sha"] != dev["sha"]:
            prs = [
                int(m.group(1))
                for _, s in dev["rocke_commits"]
                if (m := MERGED_PR.search(s))
            ]
            events.append(
                _ev(
                    "DEVELOP_MOVED",
                    None,
                    old=d["sha"][:10],
                    new=dev["sha"][:10],
                    commits=len(dev["new_commits"]),
                    rocke=len(dev["rocke_commits"]),
                    rocke_prs=prs,
                )
            )
            if dev["rocke_commits"]:
                d.setdefault("pending_sanity", []).append(dev["sha"])
        from .snapshot import ci_verdict, red_cleared_by

        for sha, ci in dev["ci"].items():
            new_v, old_v = ci_verdict(ci), ci_verdict(d["ci"].get(sha) or {})
            if new_v == old_v:
                continue
            runs = ci.get("runs", {})
            if new_v in ("red", "red-other"):
                kw = dict(
                    sha=sha[:10],
                    rocke_jobs=sorted(
                        {j for r in runs.values() for j in r.get("rocke_failed", [])}
                    )[:8],
                    workflows=sorted(
                        n for n, r in runs.items() if r["conclusion"] == "failure"
                    ),
                )
                # A red that newer SHAs already show gone (a late re-run attempt, a flake) is history, not a break.
                green = red_cleared_by(dev["ci"], sha)
                if green:
                    events.append(
                        _ev("DEVELOP_RED_CLEARED", None, green=green[:10], **kw)
                    )
                else:
                    events.append(
                        _ev(
                            "DEVELOP_RED" if new_v == "red" else "DEVELOP_RED_OTHER",
                            None,
                            **kw,
                        )
                    )
            elif new_v == "infra":
                events.append(
                    _ev(
                        "DEVELOP_CI_INFRA",
                        None,
                        sha=sha[:10],
                        jobs=sorted(
                            {
                                j
                                for r in runs.values()
                                for j in r.get("infra_failed", [])
                            }
                        )[:8],
                    )
                )
        # The window lists SHAs newest first; the newest green one becomes last_green unless it is older than ours.
        window = list(dev["ci"])
        greens = [s for s in window if ci_verdict(dev["ci"][s]) == "green"]
        lg = d.get("last_green")
        if greens and (
            not lg
            or (
                lg["sha"] != greens[0]
                and (
                    lg["sha"] not in window
                    or window.index(greens[0]) < window.index(lg["sha"])
                )
            )
        ):
            d["last_green"] = {"sha": greens[0], "at": ts, "how": "ci"}
            events.append(_ev("DEVELOP_GREEN", None, sha=greens[0][:10]))
        d["ci"] = dev["ci"]
        if d.get("sha") != dev["sha"]:
            d["sha"], d["seen_at"] = dev["sha"], ts
    return events
