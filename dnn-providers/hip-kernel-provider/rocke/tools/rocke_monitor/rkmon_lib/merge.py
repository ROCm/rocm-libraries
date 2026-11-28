# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The merge queue. The tick validates the head of the queue and tells the orchestrator what to do; every
outward action (propose, arm, disarm, coordination notice) is a separate guarded `rk` command the orchestrator runs.

**Gate job** (single PR): refresh `wt<N>m` (PR head merged with today's develop) and run the checks suite in gate mode.
Its status becomes G5–G7 for exactly that head and develop. Review checks record the same gates, so a PR reviewed at
today's develop needs no extra gate job.

**Batch gate job** (two or more PRs): in `wtbatch`, merge the members one by one, in landing order, onto the develop
SHA, and run the suite after every merge when `check_prefixes` is set (only after the last merge otherwise). The result
is `validated`, `conflict #N`, or `fail at step k (#N): sections`.

**Carry-over.** A validated batch stays valid while develop moves only by its own members, landing in order: each
landed prefix was itself validated. Any other develop move invalidates it.

**One armed PR at a time.** The tick signals DISARM_NEEDED when the queue freezes, the armed head moves, or develop
moves under it without carry-over.

**Holds.** A gate or batch failure holds the PR ("triage") instead of re-testing it every
tick. The orchestrator checks the failure on develop, then confirms the hold (`rk hold`: the PR's fault) or releases it
(`rk release`: develop or tool). A held PR keeps its rank; the queue moves on to PRs that do not force it to rebase or
re-bless (queue.apply_holds). A push ends the hold and the PR is re-gated. After `hold_stale_h` hours the user decides.

**Two tiers.** Candidates (approvals plus a clean review at head) get their gate job
early, as the heavy slot allows; only ready PRs (CI green too) are batched, proposed or armed. Gate results stay fresh
while develop's rocKE tree is unchanged. Auto-merge someone else enabled on a PR we know is unsafe is an alert, and with
the auto-merge-guard feature at act, `rk automerge-off`.
"""

from __future__ import annotations

import hashlib
import re
import shlex

from . import (
    config,
    features,
    gh,
    gitops,
    guard,
    idle,
    jobs,
    mergeplan,
    paths,
    prcode,
    queue,
    registry,
    review,
)
from .store import halted, minutes_since, now

BATCH_CULPRIT = re.compile(r"(?:fail at step \d+ \(#|conflict #)(\d+)")


# --- gate and batch jobs ---------------------------------------------------------------------------------------------


def _plan(cfg: dict, rec: dict, develop: str) -> dict:
    files = review._changed_files(cfg, rec)
    fams = guard.emitter_families_at(cfg, develop)
    pc = review.plan_checks(
        cfg, rec, files, fams, source=("git", cfg["paths"]["repo"], develop)
    )
    pc["_dirs"] = {f: fams[f] for f in pc["families"]}
    return pc


def gate_script(cfg: dict, rec: dict, develop: str) -> str:
    n = rec["n"]
    pre = (
        f"{paths.shell_vars(cfg)}; N={n}\n{config.export_line(cfg)}\n"
        "for W in wt$N wt${N}m; do [ -d $D/$W ] && { git -C $D/$W merge --abort 2>/dev/null; "
        "git -C $D/$W reset -q --hard 2>/dev/null; }; done\n"
        "mkdir -p $WK/pr$N\n"
        f"python3 -B {paths.worker_tool('mkwt.py')} $N > $WK/pr$N/gate_mkwt.log 2>&1 || {{ echo 'mkwt failed'; "
        "tail -5 $WK/pr$N/gate_mkwt.log; }\n"
    )
    return pre + review.checks_script(
        cfg, n, rec["head"], "gate", _plan(cfg, rec, develop), None, mode="gate"
    )


def cancel_gate(st: dict, store, reason: str) -> str:
    """Stop the running gate job (e.g. one started on a PR that cannot be analysed), or else the running batch gate
    (e.g. one whose batch is still validated on the same rocKE tree), and free the heavy slot.
    """
    qs = st.setdefault("queue_state", {})
    g = qs.get("gate")
    if not g:
        bg = qs.get("batch_gate")
        if not bg:
            return "no gate job running"
        jobs.kill(store, bg["job"])
        qs["batch_gate"] = None
        jobs.release_heavy(st, bg["job"])
        order = " → ".join(f"#{n}" for n in bg["members"])
        store.journal("queue", f"batch gate {order} cancelled: {reason}")
        return f"cancelled the batch gate {order} ({bg['job']['id']})"
    jobs.kill(store, g["job"])
    qs["gate"] = None
    jobs.release_heavy(st, g["job"])
    store.journal("queue", f"gate job on #{g['pr']} cancelled: {reason}")
    return f"cancelled the gate job on #{g['pr']} ({g['job']['id']})"


def start_gate(cfg: dict, st: dict, store, n: int) -> list[str]:
    rec, dev = st["prs"][str(n)], st["develop"]["sha"]
    # The PR list can lag a merge by a tick (a PR merged a minute ago may still read open), so the
    # PR must still be open on GitHub before it takes the heavy slot. No answer: go ahead, as before.
    try:
        live = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq=".state")
    except Exception:  # noqa: BLE001
        live = "open"
    if live != "open":
        return [
            f"GATE_SKIP #{n}: it is {live} on GitHub; the PR list catches up next tick"
        ]
    job = jobs.start(
        store, "gate", f"pr{n}", gate_script(cfg, rec, dev), timeout_min=180
    )
    st.setdefault("queue_state", {})["gate"] = {
        "pr": n,
        "head": rec["head"],
        "develop": dev,
        "job": job,
    }
    jobs.take_heavy(st, job, store)
    return [f"GATE_START #{n} against develop {dev[:10]}"]


def batch_id(members: list[int], develop: str) -> str:
    return f"b{develop[:8]}-" + "-".join(str(m) for m in members)


def batch_script(cfg: dict, st: dict, batch: dict) -> str:
    dev, members = batch["develop"], batch["members"]
    bid = batch_id(members, dev)
    out = f"{paths.get(cfg, 'workers')}/{bid}"
    recs = [st["prs"][str(n)] for n in members]
    union = {
        "families": [],
        "tests": [],
        "conftests": [],
        "changed_tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "_dirs": {},
    }
    for rec in recs:
        pc = _plan(cfg, rec, dev)
        for k in (
            "families",
            "tests",
            "conftests",
            "changed_tests",
        ):  # changed_tests: the installed-tree run (L-526)
            union[k] = sorted(set(union[k]) | set(pc.get(k) or []))
        for k in ("build", "golden", "engine_wide"):
            union[k] = union[k] or pc[k]
        union["_dirs"].update(pc["_dirs"])
    union["tests"] = union["tests"][:40]
    L = [
        paths.shell_vars(cfg),
        prcode.SHELL,  # prtree: the batch's tree is one PR code reads (each step's checks run it in a slot)
        f'WT="$D/wtbatch"; OUT={shlex.quote(out)}; mkdir -p "$OUT"; : > "$OUT/result"',
        'if [ ! -d "$WT" ]; then prtree "$WT" && git -C "$R0" worktree add -q --no-checkout --detach "$WT" '
        + dev
        + ' && git -C "$WT" '
        f"sparse-checkout set --no-cone {' '.join(paths.sparse_dirs(cfg))}; fi",
        'git -C "$WT" merge --abort 2>/dev/null; git -C "$WT" checkout -q --detach '
        + dev
        + ' && git -C "$WT" reset -q --hard '
        '&& git -C "$WT" clean -fdxq',
    ]
    for k, rec in enumerate(recs, 1):
        n, head = rec["n"], rec["head"]
        L += [
            f'git -C "$R0" fetch -q --filter=blob:none origin +refs/pull/{n}/head:refs/remotes/origin/pr{n}',
            f'git -C "$WT" -c user.name=rkmon -c user.email={review.BATCH_EMAIL} merge -q --no-ff --no-edit {head} || '
            f'{{ git -C "$WT" merge --abort; echo \'conflict #{n}\' > "$OUT/result"; exit 0; }}',
        ]
        if batch.get("check_prefixes") or k == len(recs):
            step = f"{out}/step{k}"
            L += [
                "cat > \"$OUT/step%d.sh\" <<'RKMON_EOF'" % k,
                review.checks_script(
                    cfg,
                    n,
                    head,
                    f"batch step {k}",
                    union,
                    None,
                    mode="gate",
                    wt="wtbatch",
                    out=step,
                    base_sha=dev,
                ).rstrip("\n"),
                "RKMON_EOF",
                f'bash "$OUT/step{k}.sh"; if grep -q \' fail \' "$OUT/step{k}/status"; then echo "fail at step {k} '
                f"(#{n}): $(grep ' fail ' \"$OUT/step{k}/status\" | cut -d' ' -f1 | tr '\\n' ' ')\" > \"$OUT/result\"; "
                "exit 0; fi",
            ]
    L.append('echo validated > "$OUT/result"')
    return "\n".join(L) + "\n"


def start_batch(cfg: dict, st: dict, store, batch: dict) -> list[str]:
    bid = batch_id(batch["members"], batch["develop"])
    trees = [paths.worktree(cfg, "wtbatch")]
    try:
        trees.append(f"{paths.get(cfg, 'build')}/wtbatch")
    except paths.PathsError:
        pass
    freed = jobs.free_tree(trees, st, store, "batch gate")
    job = jobs.start(store, "batch", bid, batch_script(cfg, st, batch), timeout_min=360)
    st.setdefault("queue_state", {})["batch_gate"] = {
        "id": bid,
        "members": batch["members"],
        "heads": batch["heads"],
        "develop": batch["develop"],
        "job": job,
    }
    jobs.take_heavy(st, job, store)
    return freed + [
        f"BATCH_GATE_START {' → '.join('#' + str(n) for n in batch['members'])} on develop {batch['develop'][:10]}"
        + (" (each prefix checked)" if batch.get("check_prefixes") else "")
    ]


def _hold(st: dict, store, n: int, head: str, reason: str, source: str) -> None:
    st.setdefault("holds", {})[str(n)] = {
        "head": head,
        "reason": reason[:200],
        "status": "triage",
        "since": now(),
        "source": source,
    }
    store.journal("queue", f"#{n} held pending triage ({source}): {reason[:160]}")


def _read(path: str) -> str | None:
    import pathlib

    p = pathlib.Path(path)
    return p.read_text().strip() if p.exists() and p.read_text().strip() else None


# --- coordination notices ----------------------------------------------------------------------------------------------


def notice_text(cfg: dict, st: dict, n: int, state: str, detail: str) -> str:
    """Our merge-queue status comment on PR n in STATE (desired_notices), its first line this deployment's
    (post.notice_line)."""
    from . import post

    body = {
        "next": f"**Next in line.** Every merge gate is green against develop `{detail}`. Merging order: {{order}}.",
        "batch": f"**In a batch validated together** on develop `{detail}`. Please merge in this order: {{order}}. "
        "Merging out of order, or merging other rocKE work in between, invalidates the batch.",
        "hold": f"**Please hold.** {detail} The queue will tell you here when this PR is next.",
        "abi": f"**ABI lane: you are next.** {detail}",
        "blocked": f"**Blocked at the merge gate:** {detail} Push a fix and the queue re-tests this PR; it keeps its "
        "place in the queue meanwhile.",
    }[state]
    order = " → ".join(f"#{m}" for m in st.get("_notice_order", {}).get(str(n), [n]))
    return (
        f"{post.notice_line(cfg, 'queue')}\n\n{body.format(order=order)}\n\n"
        "_This comment is updated in place as the queue changes. This monitor validates merges against today's "
        "develop and coordinates the order; it does not approve, reject or merge on its own._\n"
    )


def desired_notices(cfg: dict, st: dict, q: list[dict]) -> dict[int, tuple[str, str]]:
    """PR -> (state, detail) for PRs that need a coordination notice now. Only actionable states."""
    out: dict[int, tuple[str, str]] = {}
    dev = (st["develop"]["sha"] or "")[:10]
    order_map = {}
    for b in (st.get("batches") or [])[:1]:
        if b.get("status") == "validated":
            for n in b["members"]:
                order_map[str(n)] = b["members"]
                out[n] = ("batch" if len(b["members"]) > 1 else "next", dev)
    st["_notice_order"] = order_map
    for lane, L in (st.get("abi_lanes") or {}).items():
        if L.get("head") is not None and L["groups"]:
            g = L["groups"][L["head"]]
            for n in g["members"]:
                row = next((r for r in q if r["n"] == n), None)
                if (
                    row
                    and row["gates"]["G9"][0] == "fail"
                    and "lane head" in row["gates"]["G9"][1]
                ):
                    out.setdefault(n, ("abi", row["gates"]["G9"][1]))
    for k, h in queue.active_holds(st).items():
        if (
            h["status"] == "held"
        ):  # confirmed by triage only: authors never hear about an untriaged failure
            out.setdefault(int(k), ("blocked", h["reason"][:160]))
    return out


# --- the two tiers: candidate analysis and auto-merge risks ------------------------------------------------------------


def analyse_candidates(cfg: dict, st: dict, store, q: list[dict]) -> list[str]:
    """Candidate analysis (the two-tier model). A PR with its approvals and a clean review at head
    (stage candidate, or ready) but no fresh G5-G7 gets its gate job early, while CI may still be running, so golden and
    merge problems surface before it is ready. The heavy slot goes to develop sanity, the head batch and review checks
    first (reviews are the bottleneck); a candidate that has waited `analysis_max_wait_min` (default 90) goes next
    anyway. Never in a reviewer-model worker's first 5 minutes."""
    qs = st.setdefault("queue_state", {})
    if st.get("frozen") or qs.get("gate") or qs.get("batch_gate"):
        return []
    holds = queue.active_holds(st)
    dev, tree = st["develop"]["sha"], st["develop"].get("rocke_tree")
    # Members of a batch validated at their heads on this develop (or carried over since) are covered by it: a gate job
    # of their own would only repeat it, and hold up a review.
    val = qs.get("validated")
    covered = set()
    if val and (val["develop"] == dev or _carry(cfg, st, val)):
        covered = {
            n
            for n in val["members"]
            if (st["prs"].get(str(n)) or {}).get("head") == val["heads"].get(str(n))
        }
    # A PR that conflicts with develop cannot be merged for analysis: it needs the author's rebase first (G4). Fresh
    # G5-G7 come from the PR's recorded gates, not `q`: a GATE_DONE earlier in this tick may have refreshed them after
    # `q` was built (else an analysis that just finished starts again).
    todo = [
        r
        for r in q
        if r.get("stage") in ("candidate", "ready")
        and str(r["n"]) not in holds
        and r["gates"]["G4"][0] != queue.FAIL
        and r["n"] not in covered
        and any(
            v[0] == "pending"
            for v in queue.gates_dynamic(st["prs"][str(r["n"])], dev, tree).values()
        )
    ]
    waits = st.setdefault("analysis_wait", {})
    for k in [k for k in waits if k not in {str(r["n"]) for r in todo}]:
        del waits[k]
    for r in todo:
        waits.setdefault(str(r["n"]), now())
    if not todo or jobs.heavy_busy(cfg, st, store) or halted(st):
        return []
    todo.sort(
        key=lambda r: (
            r.get("tier") != 0,
            r["stage"] != "ready",
            waits[str(r["n"])],
            r["size"],
        )
    )
    top = todo[0]
    waited = minutes_since(waits[str(top["n"])])
    if review.worker_prefill(cfg, st):
        return []
    patient = (
        waited < registry.value(cfg, "analysis_max_wait_min") and top.get("tier") != 0
    )
    for pipe in review.pipes(st):
        # A worker on a huge PR has a long prompt, and it times out when the machine is loaded. Analysis waits for it,
        # unless the candidate fixes develop or has waited too long.
        worker = pipe.get("worker") or {}
        if (
            worker.get("id")
            and jobs.status(store, worker) == "running"
            and (st["prs"].get(str(pipe["pr"])) or {}).get("size", 0)
            >= registry.value(cfg, "huge_pr_lines")
            and patient
        ):
            return []
        # A review in flight needs a slot for its checks, unless the candidate waited too long or fixes develop.
        if pipe.get("checks") is None and patient:
            return []
    del waits[str(top["n"])]
    return [
        f"ANALYSIS_START #{top['n']} ({top['stage']}, waited {waited:.0f} min): "
        + " ".join(start_gate(cfg, st, store, top["n"]))
    ]


PLAIN = {
    "G5": (
        "the outputs differ from the stored expected outputs (golden files)",
        "re-record (re-bless) the affected golden files on top of the current develop branch",
    ),
    "G6": (
        "the build or the tests fail",
        "merge the current develop branch into this PR and fix the failing build or tests",
    ),
    "G7": (
        "some kernel configurations do not compile for their GPU target",
        "make those configurations compile again, or say in the PR why the rejection is intended",
    ),
}


def automerge_guard_acts(cfg: dict, rec: dict) -> bool:
    """Whether the auto-merge guard turns auto-merge off itself (the auto-merge-guard feature: act), or only alerts the
    user.
    The user's own PRs only with `features.merge_queue.auto_merge_guard.on_your_pull_requests` at `same_as_others`.
    Auto-merge the user turned on themselves is never turned off: it is their call, even when a later review finds a
    🟠."""
    return (
        features.guard_mode(cfg) == "act"
        and (rec.get("auto_merge") or {}).get("by") != config.operator_login(cfg)
        and (
            rec["author"] != config.operator_login(cfg)
            or cfg.get("automerge_guard_own")
        )
    )


def automerge_risks(cfg: dict, st: dict, store=None) -> list[str]:
    """Auto-merge someone enabled on a PR whose gates fail (we post a message on the PR that explains it in plain words,
    and turn auto-merge off). The gates GitHub does not enforce itself:
    - G1: our 🔴/🟠 review stands at its head;
    - G5–G7: a gate failure on the merge with today's develop that triage confirmed as the PR's (a hold; untriaged
      failures never count);
    - G8: a hold label (DO NOT MERGE, WIP, on hold, ...).
    CI, approvals and conflicts GitHub enforces itself. Reasons and fixes are plain words for the PR comment. With
    `automerge_guard` "act" the line says to run `rk automerge-off N`; otherwise it is an alert for the user. The
    auto-merge guard acts once per head (post.automerge_off): once the ledger has our turn-off at this head, a re-enable
    is the user's call, and the line says so, as the board does. Our own armed PR is DISARM_NEEDED's business.
    """
    armed = ((st.get("queue_state") or {}).get("armed") or {}).get("pr")
    holds = queue.active_holds(st)
    old = st.get("automerge_risks") or {}
    new, lines = {}, []
    mp = st.get("merge_plan") or {}
    rows = mergeplan.wave_rows(mp)
    wave1 = (
        set(rows[0][1]) | set(rows[0][2]) if rows else set()
    )  # its end after a rebase included
    kept = {}
    for k, rec in st["prs"].items():
        am = rec.get("auto_merge")
        if rec.get("status") != "open" or not am or rec["n"] == armed:
            continue
        reasons, fixes = [], []
        ours = rec.get("ours") or {}
        g1 = queue.gate_review(cfg, rec)
        if (
            g1[0] == queue.FAIL
            and g1[1].startswith("our verdict")
            and ours.get("verdict") in ("🔴", "🟠")
        ):
            reasons.append(
                "our latest review of this exact version found "
                + (
                    "blocking problems"
                    if ours["verdict"] == "🔴"
                    else "problems to fix first"
                )
                + ", and they are still open"
            )
            fixes.append(
                "address the open items in our latest review"
                + (f" ({ours['url']})" if ours.get("url") else "")
            )
        h = holds.get(k)
        if h and h["status"] == "held":
            parts = [p.strip() for p in h["reason"].split("; ") if p.strip()]
            for p in parts:
                plain = PLAIN.get(p[:2])
                if plain:
                    # "G6 tests: …" or "G6: …"; the PR comment gets whole words, never a cut "(".
                    detail = p[2:].lstrip(" :")
                    detail = (
                        detail if len(detail) <= 160 else detail[:159].rstrip() + "…"
                    )
                    reasons.append(
                        f"merged with today's develop branch, {plain[0]} ({detail})"
                    )
                    fixes.append(plain[1])
            if not any(PLAIN.get(p[:2]) for p in parts):
                reasons.append(
                    f"a pre-merge check failed on the merge with today's develop branch: {h['reason'][:140]}"
                )
                fixes.append("push a fix")
        for lab in queue.hold_labels(cfg, rec):
            reasons.append(
                f"it carries the label '{lab}', which says it should not merge yet"
            )
            fixes.append(f"remove the label '{lab}' once it is ready")
        if not reasons:
            continue
        if (
            am.get("by") == config.operator_login(cfg)
            and rec["n"] in wave1
            and len(reasons) == 1
            and ours.get("verdict") == "\U0001f7e0"
            and reasons[0].startswith("our latest review")
        ):
            # The user armed a PR in the plan's wave 1 (what lands first) over our 🟠: the auto-merge guard lets it be
            # (its open items get follow-up tickets anyway). Merged, its open items become follow-up tickets
            # (MERGED_OVER_RED). A 🔴, a hold or a hold label still alerts: a ticket does not undo a develop break.
            kept[k] = {
                "head": rec["head"],
                "since": (st.get("automerge_kept") or {}).get(k, {}).get("since")
                or now(),
            }
            continue
        prev = old.get(k) or {}
        new[k] = {
            "head": rec["head"],
            "by": am.get("by"),
            "reasons": reasons,
            "fixes": list(dict.fromkeys(fixes)),
            "since": prev.get("since") or now(),
        }
        if prev.get("reasons") != reasons or prev.get("head") != rec["head"]:
            off = (
                [
                    r
                    for r in store.ledger()
                    if r["kind"] == "automerge-off"
                    and r["pr"] == rec["n"]
                    and r["head"] == rec["head"]
                ]
                if store
                else []
            )
            if off:
                act = (
                    f"I turned it off at this head at {off[-1]['ts'][11:16]}Z and it was turned back on: alert the "
                    "user (the auto-merge guard turns auto-merge off once per head)"
                )
            else:
                act = (
                    f"run `rk automerge-off {k}`"
                    if automerge_guard_acts(cfg, rec)
                    else "alert the user"
                )
            lines.append(
                f"AUTOMERGE_RISK #{k} (auto-merge enabled by {am.get('by')}): {'; '.join(reasons)}. {act}"
            )
    st["automerge_risks"] = new
    st["automerge_kept"] = kept
    return lines


def _reapproved(rec: dict, me: str, hold: dict) -> bool:
    """The user's own approval after the hold: of a later commit than the held head, or submitted after the hold."""
    r = next((x for x in rec.get("reviews") or [] if x["login"] == me), None)
    if not r or r["state"] != "APPROVED":
        return False
    c = r.get("commit") or ""
    return bool(c and c[:10] != hold["head"][:10]) or (r.get("at") or "") > (
        hold.get("at") or "~"
    )


def hold_reviews(cfg: dict, st: dict) -> list[str]:
    """Keep held PRs blocked on GitHub until the user re-approves them. A hold confirmed
    at a PR's head needs a change request from our account (`rk request-changes`); one the user submitted counts. It
    stands after the hold ends with a push: the author fixes, re-requests the user's review, and the user re-approves
    explicitly. We never approve or dismiss. st["hold_reviews"] follows each one. Lines:
    - !! HOLD_REVIEW_NEEDED #N: a confirmed hold that GitHub would still let merge (critical);
    - HOLD_FIXED #N: the author pushed since the hold; the fix gets our follow-up review and a re-gate;
    - HOLD_REAPPROVED #N: the user approved after the hold (the fix, or a later review), so the record ends;
    - !! HOLD_DISMISSED #N: the change request is gone without that approval (e.g. the author dismissed it); the
      record stays, and G2 waits for the user's explicit re-approval."""
    me = config.operator_login(cfg)
    if not me:
        return []  # without the user's login there is no change request to follow
    live = queue.active_holds(st)
    hr = st.setdefault("hold_reviews", {})
    lines = []
    for k, rec in st["prs"].items():
        mine = next(
            (r["state"] for r in rec.get("reviews") or [] if r["login"] == me), None
        )
        if k in hr:
            r = hr[k]
            if rec.get("status") != "open":
                del hr[k]
            elif mine != "CHANGES_REQUESTED":
                if _reapproved(rec, me, r):
                    lines.append(f"HOLD_REAPPROVED #{k}: you re-approved it")
                    del hr[k]
                elif not r.get("dismissed"):
                    r["dismissed"] = now()
                    lines.append(
                        f"!! HOLD_DISMISSED #{k}: the change request from the hold was removed without your "
                        f"re-approval (your review: {(mine or 'none').lower()}); GitHub now lets it merge. "
                        "Your explicit re-approval is still owed: tell the user"
                    )
            elif rec["head"] != r["head"] and r.get("fix_seen") != rec["head"]:
                r["fix_seen"] = rec["head"]
                lines.append(
                    f"HOLD_FIXED #{k}: {rec['author']} pushed {rec['head'][:10]} since the hold "
                    f"({r['reason'][:80]}). We review and re-gate the fix; then {rec['author']} re-requests "
                    "your review and you re-approve"
                )
            continue
        h = live.get(k)
        if not h or h.get("status") != "held" or rec.get("author") == me:
            continue
        if mine == "CHANGES_REQUESTED":  # the user requested changes themselves
            hr[k] = {
                "head": rec["head"],
                "reason": h["reason"][:200],
                "by": "you",
                "at": now(),
                "url": "",
            }
            continue
        lines.append(
            f"!! HOLD_REVIEW_NEEDED #{k}: held ({h['reason'][:100]}), but GitHub would still let it merge. Run "
            f"`rk request-changes {k} FILE`: what fails, how to fix it, and to re-request a review once fixed"
        )
    return lines


# --- the tick's queue step ---------------------------------------------------------------------------------------------


def _carry(cfg: dict, st: dict, val: dict) -> bool:
    """Has develop moved since val['develop'] only by val's members, landing in order, and by commits outside rocKE?
    A commit outside rocKE cannot change a rocKE gate result (the two-tier model): it never re-runs a 30-minute batch
    gate that passed on the same rocKE tree."""
    repo = cfg["paths"]["repo"]
    try:
        rows = gitops.first_parent_since(repo, val["develop"], st["develop"]["sha"])
        inside = {
            sha
            for sha, _ in gitops.first_parent_since(
                repo, val["develop"], st["develop"]["sha"], paths.repository_folder(cfg)
            )
        }
    except gitops.GitError:
        return False
    landed = []
    for sha, subj in reversed(rows):
        m = guard.MERGED_PR.search(subj)
        n = int(m.group(1)) if m else None
        if n is not None and n in val["members"]:
            landed.append(n)
        elif sha in inside:
            return False  # other rocKE work landed in between
    return landed == val["members"][: len(landed)]


def merged_over_red(
    cfg: dict, st: dict, store, events: list[dict], told: set
) -> list[str]:
    """A merge over our open 🔴/🟠 at the merged head, or over a hold whose change request was dismissed without the
    user's re-approval. A PR that already left the holds gets no HELD_MERGED, so this line is what says it merged over
    our findings, before develop breaks as the review said."""
    out = []
    for ev in (
        e for e in events if e["kind"] == "MERGED" and str(e.get("pr")) not in told
    ):
        k = str(ev["pr"])
        rec = st["prs"].get(k) or {}
        ours, head = rec.get("ours") or {}, (rec.get("head") or "")[:10]
        red = (
            ours.get("verdict") not in (None, *queue.GOOD_VERDICTS)
            and bool(head)
            and any(
                (h or "")[:10] == head
                for h in [ours.get("head"), *ours.get("equivalent_heads", [])]
            )
        )
        hcr = (st.get("hold_reviews") or {}).get(k) or {}
        unapproved = bool(hcr.get("dismissed")) and not any(
            r.get("login") == config.operator_login(cfg)
            and r.get("state") == "APPROVED"
            for r in rec.get("reviews") or []
        )
        if not (red or unapproved):
            continue
        why = ([f"our {ours['verdict']} review at {head} is open"] if red else []) + (
            ["held earlier: its change request was dismissed without your re-approval"]
            if unapproved
            else []
        )
        by = ev.get("by") or "someone"
        out.append(
            f"!! MERGED_OVER_RED #{k}: merged by {by} ({'; '.join(why)}). Reproduce our open findings on "
            "develop, file the develop issue if they fail, and tell the user"
        )
        store.journal("queue", f"#{k} merged by {by} over {'; '.join(why)}")
        merged_over(
            st,
            k,
            "MERGED_OVER_RED",
            by,
            rec.get("head") or "",
            "while " + "; ".join(why),
        )
    return out


def merged_over(st: dict, k: str, kind: str, by: str, head: str, why: str) -> None:
    """Keep a merge while held or over our open findings for the board's Needs-you list (report.MERGED_SHOWN_H)."""
    st.setdefault("merged_over", {})[k] = {
        "kind": kind,
        "by": by,
        "head": head,
        "why": why,
        "at": now(),
    }


def prune_merged_over(st: dict) -> None:
    """Drop the merges the Needs-you list no longer shows."""
    from .report import MERGED_SHOWN_H

    mo = st.get("merged_over") or {}
    for k in [
        k for k, m in mo.items() if minutes_since(m["at"]) >= MERGED_SHOWN_H * 60
    ]:
        del mo[k]


def advance(cfg: dict, st: dict, store, q: list[dict], events: list[dict]) -> list[str]:
    qs = st.setdefault("queue_state", {})
    lines, dev = [], st["develop"]["sha"]
    holds = st.setdefault("holds", {})
    live = queue.active_holds(st)
    told = set()
    prune_merged_over(st)
    for k in [k for k in holds if k not in live]:
        rec = st["prs"].get(k) or {}
        if rec.get("status") == "merged":
            # A held head merged (our change request dismissed): more than the quiet HOLD_CLEARED below. The hold's
            # failure may now be on develop.
            h, ev = holds[k], next(
                (e for e in events if e["kind"] == "MERGED" and str(e.get("pr")) == k),
                {},
            )
            same = (h.get("head") or "")[:10] == (rec.get("head") or "")[:10]
            where = (
                f"at the held head {h.get('head', '')[:10]}"
                if same
                else "after a push since the hold"
            )
            lines.append(
                f"!! HELD_MERGED #{k}: merged while {'held' if h.get('status') == 'held' else 'in triage'} "
                f"(by {ev.get('by') or 'someone'}, {where}). The hold: {(h.get('reason') or '')[:160]}. "
                "Tell the user; reproduce the failure on develop and file the develop issue if it fails"
            )
            store.journal(
                "queue",
                f"#{k} merged while held, by {ev.get('by') or 'someone'}, {where}",
            )
            merged_over(
                st,
                k,
                "HELD_MERGED",
                ev.get("by") or "someone",
                rec.get("head") or "",
                f"while {'held' if h.get('status') == 'held' else 'in triage'} {where} "
                f"({(h.get('reason') or '')[:120]})",
            )
            told.add(k)
        why = (
            "closed or merged"
            if rec.get("status") != "open"
            else "pushed; it is re-gated"
        )
        lines.append(f"HOLD_CLEARED #{k}: {why}")
        store.journal("queue", f"hold on #{k} cleared: {why}")
        del holds[k]
    lines += merged_over_red(cfg, st, store, events, told)
    armed = qs.get("armed")
    merged = {e["pr"] for e in events if e["kind"] == "MERGED"}
    pushed = {e["pr"] for e in events if e["kind"] == "PUSH"}
    if armed:
        if armed["pr"] in merged:
            lines.append(f"LANDED #{armed['pr']} (armed by us); develop sanity follows")
            store.journal("queue", f"#{armed['pr']} landed while armed")
            qs["armed"] = None
        elif st.get("frozen"):
            lines.append(
                f"DISARM_NEEDED #{armed['pr']}: queue frozen ({st['frozen'].get('reason')}). Run `rk disarm "
                f"{armed['pr']}`"
            )
        elif armed["pr"] in pushed or (st["prs"].get(str(armed["pr"])) or {}).get(
            "head"
        ) not in (None, armed["head"]):
            # Said on every tick until the disarm goes through (a failed `rk disarm` keeps the PR armed).
            lines.append(
                f"DISARM_NEEDED #{armed['pr']}: its head moved. Run `rk disarm {armed['pr']}`"
            )
        elif armed["develop"] != dev and not _carry(
            cfg, st, qs.get("validated") or {"develop": armed["develop"], "members": []}
        ):
            lines.append(
                f"DISARM_NEEDED #{armed['pr']}: develop moved by other work since its gate. Run "
                f"`rk disarm {armed['pr']}`"
            )
    g = qs.get("gate")
    gs = jobs.status(store, g["job"]) if g else None
    if g and gs != "running":
        # past its time, or lost: stop what is left before letting go
        jobs.stop_unfinished(store, g["job"], gs)
        status, dsha, hsha = review.read_checks_status(
            f"{paths.get(cfg, 'workers')}/pr{g['pr']}/gate"
        )
        rec = st["prs"].get(str(g["pr"]))
        if (
            gs != "done"
        ):  # cut short: what it wrote is part of a run, never the gate's result
            lines.append(
                f"GATE_INCOMPLETE #{g['pr']}: job {gs}"
                + (f" after {', '.join(sorted(status))}" if status else "")
                + ": nothing recorded; it is gated again"
            )
        elif rec and status and dsha:
            rec["gates"] = {
                "head": hsha or g["head"],
                "develop": dsha,
                "develop_tree": queue.rocke_tree(cfg, dsha),
                "results": review.gates_from_status(status),
                "at": now(),
                "source": "gate-job",
            }
            res = rec["gates"]["results"]
            lines.append(
                f"GATE_DONE #{g['pr']}: "
                + "; ".join(f"{k} {v[0]}" for k, v in res.items())
            )
            bad = [f"{k} {v[1]}" for k, v in res.items() if v[0] == "fail"]
            if bad:
                _hold(st, store, g["pr"], hsha or g["head"], "; ".join(bad), "gate")
                lines.append(
                    f"TRIAGE #{g['pr']}: gate failed, held. Run the failing check on develop: the PR's fault "
                    f"-> `rk hold {g['pr']} REASON`; develop or our tool -> `rk release {g['pr']}`"
                )
        else:
            lines.append(
                f"GATE_DONE #{g['pr']}: no result (job {jobs.status(store, g['job'])})"
            )
        qs["gate"] = None
        jobs.release_heavy(st, g["job"])
    bg = qs.get("batch_gate")
    bgs = jobs.status(store, bg["job"]) if bg else None
    if bg and bgs != "running":
        jobs.stop_unfinished(store, bg["job"], bgs)
        res = _read(f"{paths.get(cfg, 'workers')}/{bg['id']}/result") or "no result"
        if (
            res == "validated" and bgs != "done"
        ):  # its script ends there: a job that did not finish validates nothing
            res = f"no result (job {bgs})"
        if res == "validated":
            qs["validated"] = {
                "members": bg["members"],
                "heads": bg["heads"],
                "develop": bg["develop"],
                "at": now(),
            }
            lines.append(
                f"BATCH_VALIDATED {' → '.join('#' + str(n) for n in bg['members'])} on develop "
                f"{bg['develop'][:10]}"
            )
        else:
            lines.append(
                f"BATCH_FAILED {' → '.join('#' + str(n) for n in bg['members'])}: {res}"
            )
            qs.setdefault("failed_batches", []).append(
                {"id": bg["id"], "result": res, "at": now()}
            )
            m = BATCH_CULPRIT.search(res)
            if m and m.group(1) in st["prs"]:
                bad_n = int(m.group(1))
                _hold(
                    st,
                    store,
                    bad_n,
                    bg["heads"].get(str(bad_n)) or st["prs"][str(bad_n)]["head"],
                    f"batch {bg['id']}: {res}",
                    "batch",
                )
                lines.append(
                    f"TRIAGE #{bad_n}: {res}; held, and the others re-batch without it. Triage it like a "
                    "gate failure"
                )
        qs["batch_gate"] = None
        jobs.release_heavy(st, bg["job"])
    # Mark the head batch validated when a validated batch (or the PR's own fresh gates) covers it.
    batches = st.get("batches") or []
    if batches and any(str(m) in queue.active_holds(st) for m in batches[0]["members"]):
        batches = (
            []
        )  # a head-batch member was just held: the next tick's queue re-forms the batches without it
    val = qs.get("validated")
    if batches:
        b = batches[0]
        # Only a prefix of the validated order carries: the batch gate checked each member on top of the ones before
        # it: when a member drops out, a later member was never gated on develop without it, and is not validated.
        if (
            val
            and all(
                b["heads"].get(str(n)) == val["heads"].get(str(n)) for n in b["members"]
            )
            and list(val["members"][: len(b["members"])]) == list(b["members"])
            and (val["develop"] == dev or _carry(cfg, st, val))
        ):
            b["status"] = "validated"
        elif len(b["members"]) == 1:
            # The PR's recorded gates, not `q`: a GATE_DONE above may have refreshed them after `q` was built.
            rec1 = st["prs"].get(str(b["members"][0]))
            if rec1 and all(
                v[0] == "pass"
                for v in queue.gates_dynamic(
                    rec1, dev, st["develop"].get("rocke_tree")
                ).values()
            ):
                b["status"] = "validated"
    if batches and batches[0].get("status") != "validated" and not st.get("frozen"):
        b = batches[0]
        if (
            not jobs.heavy_busy(cfg, st, store)
            and not qs.get("gate")
            and not qs.get("batch_gate")
            and not review.worker_prefill(cfg, st)
            and not halted(st)
        ):
            failed = {f["id"] for f in qs.get("failed_batches", [])}
            if batch_id(b["members"], dev) in failed:
                lines.append(
                    f"BATCH_BLOCKED {' → '.join('#' + str(n) for n in b['members'])}: failed on this develop; "
                    "waiting for a push or a develop move"
                )
            elif len(b["members"]) == 1:
                lines += start_gate(cfg, st, store, b["members"][0])
            else:
                lines += start_batch(cfg, st, store, dict(b, develop=dev))
    lines += analyse_candidates(cfg, st, store, q)
    lines += automerge_risks(cfg, st, store)
    lines += mergeplan.automerge_lines(
        st
    )  # auto-merge that would land a PR ahead of the merge plan
    lines += hold_reviews(cfg, st)
    if (
        batches
        and batches[0].get("status") == "validated"
        and not qs.get("armed")
        and not st.get("frozen")
    ):
        b = batches[0]
        cleared = set(st.get("cleared", []))
        nxt = b["members"][0]
        order = " → ".join(f"#{n}" for n in b["members"])
        if features.arms(cfg) and all(n in cleared for n in b["members"]):
            lines.append(
                f"ARM_READY #{nxt}: batch {order} validated on develop {dev[:10]}; run `rk arm {nxt}`"
            )
        else:  # auto_merge hints_only, or a member not cleared
            lines.append(
                f"PROPOSE {order}: validated on develop {dev[:10]}; tell the user to merge in this order"
            )
    for k, h in queue.active_holds(st).items():
        age_h = minutes_since(h["since"]) / 60
        if age_h >= registry.value(cfg, "hold_stale_h") and not h.get("stale_told"):
            h["stale_told"] = True
            lines.append(
                f"HOLD_STALE #{k}: held {age_h:.0f}h without a push ({h['reason'][:80]}). Ask the user: keep "
                f"its place (`rk hold {k}`) or release it (`rk release {k}`)"
            )
    if features.active(cfg, "queue-notices"):
        want = desired_notices(cfg, st, q)
        posted = st.setdefault("notices", {})
        for n, (state, detail) in want.items():
            key = hashlib.sha1(
                f"{state}:{detail}:{st.get('_notice_order', {}).get(str(n))}".encode()
            ).hexdigest()[:10]
            if posted.get(str(n), {}).get("key") != key:
                lines.append(
                    f"QUEUE_NOTICE #{n}: {state} ({detail[:60]}); run `rk queue-notice {n}`"
                )
    return lines


# --- timers of the change fingerprint (idle.py) -----------------------------------------------------------------------


@idle.timer("analysis wait")
def _analysis_due(cfg: dict, st: dict, store):
    """When each candidate waiting for its gate job may take the heavy slot ahead of a review's checks
    (`analysis_max_wait_min`)."""
    wait = registry.value(cfg, "analysis_max_wait_min")
    for k, since in (st.get("analysis_wait") or {}).items():
        yield f"analysis wait #{k}", idle.plus(since, wait)


@idle.timer("hold stale")
def _hold_due(cfg: dict, st: dict, store):
    """When each hold has gone `hold_stale_h` hours without a push (HOLD_STALE)."""
    for k, h in queue.active_holds(st).items():
        if h.get("since") and not h.get("stale_told"):
            yield f"hold stale #{k}", idle.plus(
                h["since"], registry.value(cfg, "hold_stale_h") * 60
            )
