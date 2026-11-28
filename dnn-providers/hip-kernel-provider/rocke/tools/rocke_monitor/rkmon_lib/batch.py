# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge batches: ready PRs that can land back to back under one combined gate and one develop sanity run.

The goal: fewer re-gates, re-blesses and sanity runs, without letting develop go red.

**Forming a batch.** A batch is built greedily in queue order from PRs whose static gates (G1–G4, G8, G9) pass; with a
merge plan, only from its head wave (mergeplan: the PRs that may land now, conflict-free by construction). A PR joins
only if it is compatible with every member already in the batch:
- no textual conflict between the two (the pair cache: merged on today's develop by the merge plan, or head to head
  without one);
- not in the same ABI lane as any member (at most one claim per lane);
- the batch stays within the size limits.

Stacked PRs are not batched; a stack lands bottom first.

**Validating a batch.** A batch becomes safe only after a batch gate job passes for exactly its member heads and the
develop SHA. That job:
- merges the members in landing order onto develop;
- runs the full gate suite (build, ctest, goldens ×3, the differential golden, byte identity for the union of touched
  families, the touched tests, ABI coherence) on the combined tree;
- also runs it on each prefix when a member touches a golden or an ABI counter, so every intermediate develop state is
  validated too.

On failure the job stops at the first member that conflicts or fails (`conflict #N`, `fail at step k (#N)`). That
member is held for triage (merge.advance), and the others re-batch without it on the next tick. There is no bisection:
without prefix checks, a failure after the last merge names the last member.

**After landing.** develop CI still runs on every push, and the develop sanity check runs once on the final SHA. A red
result is bisected over the batch's merges and handled as an incident.
"""

MAX_SIZE = 4
MAX_LINES = 12000


def _conflict(st: dict, a: dict, b: dict) -> bool:
    pairs = st.get("pair_cache", {})
    for key in (
        f"{a['head'][:12]}:{b['head'][:12]}",
        f"{b['head'][:12]}:{a['head'][:12]}",
    ):
        if key in pairs:
            p = pairs[key]
            return bool(p["conflicts"]) or p.get("undecided", False)
    return True  # unknown pair: not batchable until the pair has been checked


def _lanes_of(n: int, lanes: dict) -> set:
    return {lane for lane, L in lanes.items() for g in L["groups"] if n in g["members"]}


def _batch(
    st: dict, members: list[int], lanes: dict, origin: str, notes: list[str]
) -> dict:
    prs = st["prs"]
    heads = {str(n): prs[str(n)]["head"] for n in members}
    prefix = any(
        set((prs[str(n)].get("files_cache") or {}).get("hot", [])) & {"RG", "DG", "ABI"}
        or _lanes_of(n, lanes)
        for n in members
    )
    prev = next(
        (
            b
            for b in st.get("batches", [])
            if b.get("heads") == heads and b.get("develop") == st["develop"]["sha"]
        ),
        None,
    )
    status = (prev or {}).get("status", "candidate")
    if notes and status == "candidate":
        status = "waiting"
    return {
        "members": members,
        "heads": heads,
        "develop": st["develop"]["sha"],
        "origin": origin,
        "lines": sum(prs[str(n)]["size"] for n in members),
        "check_prefixes": prefix,
        "notes": notes,
        "status": status,
        "gate": (prev or {}).get("gate"),
    }


def user_groups(st: dict, ready: list[dict], lanes: dict) -> tuple[list[dict], set]:
    """Groups the user pinned (`rk group A B C`), in the user's order. A group stays one batch even when a member is
    not ready yet; its notes say what it waits for. Members that conflict are noted, not silently dropped.
    """
    prs, ready_ns = st["prs"], {r["n"] for r in ready}
    out, used = [], set()
    for members in st.get("groups", []):
        live = [n for n in members if prs.get(str(n), {}).get("status") == "open"]
        if len(live) < 1:
            continue
        notes = [f"#{n} not ready" for n in live if n not in ready_ns]
        notes += [
            f"#{a} and #{b} conflict"
            for i, a in enumerate(live)
            for b in live[i + 1 :]
            if _conflict(st, prs[str(a)], prs[str(b)])
        ]
        shared = [
            lane
            for lane in lanes
            if sum(1 for n in live if lane in _lanes_of(n, lanes)) > 1
        ]
        notes += [f"two claims in the {lane} ABI lane" for lane in shared]
        out.append(_batch(st, live, lanes, "user", notes))
        used |= set(live)
    return out, used


def build_batches(cfg: dict, st: dict, ready: list[dict], lanes: dict) -> list[dict]:
    prs = st["prs"]
    batches, used = user_groups(st, ready, lanes)
    for r in ready:
        if r["n"] in used:
            continue
        members, lines, lanes_used = [r], r["size"], _lanes_of(r["n"], lanes)
        for s in ready:
            if s["n"] in used or any(s["n"] == m["n"] for m in members):
                continue
            if len(members) >= MAX_SIZE:
                break
            if lines + s["size"] > MAX_LINES:
                continue
            if any(_conflict(st, prs[str(s["n"])], prs[str(m["n"])]) for m in members):
                continue
            ls = _lanes_of(s["n"], lanes)
            if ls & lanes_used:
                continue
            members.append(s)
            lines += s["size"]
            lanes_used |= ls
        used |= {m["n"] for m in members}
        batches.append(_batch(st, [m["n"] for m in members], lanes, "auto", []))
    return batches
