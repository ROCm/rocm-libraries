# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge gates and the least-disruptive merge order.

This module computes the gates that need only GitHub data and git plumbing: G1 review, G2 humans, G3 CI,
G4 fresh merge and G8 dependencies. G5 goldens, G6 build and engines and G7 arch coverage come from gate jobs
that run against a specific (PR head, develop SHA). Until such a job has run for the current pair, those gates
are "pending".

Local git work (fetching PR refs, merge simulation, pairwise conflicts) is done only for merge candidates:
PRs whose review is not failing and that have no hard dependency. That keeps huge or stale PRs from slowing ticks.
"""

from __future__ import annotations

import re

from . import abi, batch, gitops
from .breakage import entry_label
from . import config, paths, registry
from .snapshot import covers, same_commit

PASS, FAIL, PENDING = "pass", "fail", "pending"
GOOD_VERDICTS = ("\U0001f7e2", "\U0001f7e1")  # 🟢, 🟡
DEVELOP = "refs/remotes/origin/develop"


def ensure_pr_ref(cfg: dict, rec: dict) -> str:
    repo = cfg["paths"]["repo"]
    ref = f"refs/remotes/origin/pr{rec['n']}"
    have = gitops.git(repo, "rev-parse", "--verify", "-q", ref, check=False).strip()
    if have != rec["head"]:
        have = gitops.fetch_pr(repo, rec["n"])
    return ref


def base_of(cfg: dict, rec: dict | None, fetch: bool = False) -> str:
    """The ref PR rec's own change is read against: its base branch's (gitops.base_ref; `fetch`: fetched now), else,
    when that branch is not in the mirror, develop's."""
    base = (rec or {}).get("base")
    if gitops.is_develop(base):
        return DEVELOP
    repo = cfg["paths"]["repo"]
    if fetch:
        try:
            return gitops.fetch_base(repo, base)
        except gitops.GitError:
            pass
    ref = gitops.base_ref(base)
    return (
        ref
        if gitops.git(repo, "rev-parse", "--verify", "-q", ref, check=False).strip()
        else DEVELOP
    )


def areas_of(files: list[str], rocke_path: str) -> list[str]:
    """CODEOWNERS areas of the rocKE tree a change touches: platform, library, style, or root."""
    out = set()
    for f in files:
        if f.startswith(rocke_path + "/"):
            top = f[len(rocke_path) + 1 :].split("/", 1)[0]
            out.add(top if top in ("platform", "library", "style") else "root")
    return sorted(out)


def hot_paths(cfg: dict, key: str) -> list[str]:
    """A hot spot's paths: files, or directories ending in "/"; none for a kind config hot_spots does not name. A list
    while a file becomes a directory (the representative golden split into per-family shards under
    rocke_representative_ir_sha256/). `<ROCKE_PATH>` in a path stands for `rocke_path`, so the repository defaults
    follow wherever rocKE sits in the repository.
    """
    p = (cfg.get("hot_spots") or {}).get(key)
    if not p:
        return []
    return [
        x.replace("<ROCKE_PATH>", paths.repository_folder(cfg))
        for x in (p if isinstance(p, list) else [p])
    ]


def path_hits(paths: list[str], f: str) -> bool:
    """f is one of paths, or lies under one that is a directory (ends in "/")."""
    return any(f == p or (p.endswith("/") and f.startswith(p)) for p in paths)


def pr_hotspots(cfg: dict, rec: dict) -> dict:
    """Changed files (vs the merge base with develop), hot spots and code-owner areas for rec's head; cached. The files
    themselves feed the merge plan: two PRs that share none cannot conflict."""
    cache = rec.get("files_cache")
    if (
        cache
        and cache.get("head") == rec["head"]
        and "areas" in cache
        and "files" in cache
    ):
        return cache
    ref = ensure_pr_ref(cfg, rec)
    files = gitops.changed_files(cfg["paths"]["repo"], DEVELOP, ref)
    hot = sorted(
        k
        for k in cfg.get("hot_spots") or {}
        if any(path_hits(hot_paths(cfg, k), f) for f in files)
    )
    folder = paths.repository_folder(cfg)
    cache = {
        "head": rec["head"],
        "n_files": len(files),
        "rocke_files": sum(1 for f in files if f.startswith(folder)),
        "hot": hot,
        "areas": areas_of(files, folder),
        "files": files,
    }
    rec["files_cache"] = cache
    return cache


def reviews_here(cfg: dict) -> bool:
    """This deployment makes reviews (review automation enabled): G1 reads our review. Without them (the merge queue alone) G1
    reads the repository's own human review (gate_approvals)."""
    from . import features

    return features.on(cfg, "review-drafts")


def gate_review(cfg: dict, rec: dict) -> tuple:
    """G1, the review gate: our clean review at the head when this deployment reviews; else gate_approvals."""
    if not reviews_here(cfg):
        return gate_approvals(cfg, rec)
    return gate_ours(cfg, rec)


def gate_ours(cfg: dict, rec: dict) -> tuple:
    """G1 from our own review: a 🟢/🟡 verdict at the head, or at a head the head is a rebase-only or merge-only push
    of (snapshot.covers)."""
    ours = rec.get("ours")
    if not ours:
        return FAIL, "not reviewed by us"
    if ours.get("head") and not covers(ours, rec["head"]):
        return (
            FAIL,
            f"changed since our review ({ours['verdict']} at {ours['head'][:8]})",
        )
    if ours["verdict"] not in GOOD_VERDICTS:
        return FAIL, f"our verdict {ours['verdict']}"
    if not ours.get("head"):
        return PENDING, f"our verdict {ours['verdict']}; reviewed head unknown"
    return PASS, f"{ours['verdict']} at head"


def human_approvals(rec: dict) -> list[dict]:
    """The PR's standing approvals by people: GitHub's latest review per reviewer that is APPROVED, from neither the
    author nor a bot."""
    return [
        r
        for r in rec.get("reviews") or []
        if r["state"] == "APPROVED"
        and r["login"] != rec.get("author")
        and not r["login"].endswith("[bot]")
    ]


def approval_covers(rec: dict, commit: str | None) -> bool:
    """An approval given at `commit` stands for the PR's head: the head itself, or a head this deployment found to be a
    rebase-only or merge-only push of it (rec["carried"], carry_heads)."""
    if not commit:
        return False
    if same_commit(commit, rec["head"]):
        return True
    c = rec.get("carried") or {}
    return same_commit(c.get("head"), rec["head"]) and any(
        same_commit(commit, x) for x in c.get("of") or []
    )


def gate_approvals(cfg: dict, rec: dict) -> tuple:
    """G1 in a deployment that makes no reviews (the merge queue alone): the repository's own human review stands for
    ours. It passes when no change request stands, GitHub's review decision is APPROVED (its required reviews, code
    owners' included, are met; a repository that requires none needs one approval), and a person's approval is at the
    head or at a commit the head is a rebase-only or merge-only push of. Under the same account, a review comment
    (another deployment's, or one by hand) neither passes nor fails G1 (a merge queue gates on its own
    deployment's reviews alone, and this one makes none); the account's change request or approval counts like
    anyone's."""
    blocking = sorted(
        {
            r["login"]
            for r in rec.get("reviews") or []
            if r["state"] == "CHANGES_REQUESTED"
        }
    )
    if blocking:
        return FAIL, f"changes requested by {', '.join(blocking)}"
    approvals = human_approvals(rec)
    if not approvals:
        return FAIL, "no approval by a person yet"
    at_head = sorted(
        {r["login"] for r in approvals if approval_covers(rec, r.get("commit"))}
    )
    if not at_head:
        last = max(approvals, key=lambda r: r.get("at") or "")
        return FAIL, (
            f"approved at {str(last.get('commit') or '?')[:8]} ({last['login']}), and the PR changed since: a new "
            "approval at its head is needed"
        )
    if rec.get("review_decision") not in ("APPROVED", None):
        return PENDING, (
            f"approved at head by {', '.join(at_head)}, but GitHub's required reviews are not met yet "
            f"(review decision {rec.get('review_decision')})"
        )
    return PASS, f"approved at head by {', '.join(at_head)}"


# The most rebase checks one tick runs for carry_heads (each fetches the PR's head and its base branch).
CARRY_PER_TICK = 5


def carry_heads(cfg: dict, st: dict) -> list[str]:
    """In a deployment that makes no reviews, a push the reviewing side would carry is carried here too: for each open
    PR whose head moved since a person's approval, the push is classified once per head (review.classify: rebase-only,
    merge-only, ...). A rebase-only or merge-only push keeps the approval (rec["carried"]: the head and the commits it
    stands for), so G1 reads it as at head. At most CARRY_PER_TICK checks a tick; the rest wait for the next. One line
    per carried head.
    """
    if reviews_here(cfg):
        return []
    from . import review

    out, budget = [], CARRY_PER_TICK
    for rec in st["prs"].values():
        if rec.get("status") != "open" or rec.get("draft") or not rec.get("head"):
            continue
        c = rec.get("carried") or {}
        if not same_commit(c.get("head"), rec["head"]):
            c = {"head": rec["head"], "of": [], "checked": []}
        olds = [r.get("commit") for r in human_approvals(rec)]
        todo = [
            o
            for o in dict.fromkeys(x for x in olds if x)
            if not same_commit(o, rec["head"])
            and not any(same_commit(o, x) for x in c["checked"])
        ]
        for old in todo:
            if budget <= 0:
                break
            budget -= 1
            c["checked"].append(old)
            try:
                kind = review.classify(cfg, rec, old)["kind"]
            except (
                Exception
            ):  # noqa: BLE001 - a carry check never fails the tick: the approval just stays not at head
                continue
            if kind in ("rebase-only", "merge-only"):
                c["of"].append(old)
                out.append(
                    f"CARRIED #{rec['n']}: {kind} from {old[:8]} to {rec['head'][:8]}: the approval there stands"
                )
        rec["carried"] = c
    return out


# Merge states in which GitHub reports that no ruleset rule blocks the merge.
NOT_BLOCKED = ("CLEAN", "UNSTABLE", "HAS_HOOKS")


def github_blocks(rec: dict) -> tuple | None:
    """GitHub's own verdict. With CI green and the approvals we count present, BLOCKED means a develop ruleset rule we
    do not model: unresolved review threads (`required_review_thread_resolution`), or another approval the ruleset asks
    for (`require_extra_approval_for_unattributed_changes`). A required check that never reported also blocks; G3
    reports that one (required_missing, e.g. "Math CI Summary"), so it is not G2's. Develop does not ask
    for a fresh approval after a push (`require_last_push_approval` is false), so never suggest one.

    A known unresolved thread blocks in every other state too, unless GitHub reports the merge unblocked: right after
    develop moves, the state reads UNKNOWN while GitHub recomputes, and a PR with open threads must not head the
    queue."""
    known = rec.get("threads")
    if known and known["open"] and rec.get("merge_state") not in NOT_BLOCKED:
        t = known["open"]
        return PENDING, (
            f"GitHub still blocks the merge: {len(t)} unresolved review thread{'s' if len(t) > 1 else ''}"
            f" ({', '.join(sorted({x['login'] for x in t}))})"
        )
    if (
        rec.get("merge_state") == "BLOCKED"
        and rec.get("ci_rollup") == "SUCCESS"
        and not required_missing(rec)
    ):
        if known:
            return PENDING, (
                "GitHub still blocks the merge: a ruleset rule we do not model (an extra approval for "
                "changes not attributed to the author?)"
            )
        return PENDING, (
            "GitHub still blocks the merge: unresolved review threads, or a ruleset rule we do not model "
            "(an extra approval for changes not attributed to the author?)"
        )
    return None


def gate_humans(
    cfg: dict,
    rec: dict,
    owners: dict | None = None,
    areas: list[str] | None = None,
    hold_cr: dict | None = None,
) -> tuple:
    """CHANGES_REQUESTED fails. GitHub's APPROVED decision passes. When GitHub reports no decision, check that
    every touched CODEOWNERS area has an approval from one of its teams (owners: team -> logins). Either way, a merge
    GitHub still blocks with CI green stays pending (github_blocks).

    `hold_cr` is the change request a hold put on the PR from our account (st["hold_reviews"]). Once the author has pushed since the hold, it is pending, not failed: the author re-requests the
    user's review once the fix is in, and only the user's explicit re-approval passes G2.
    """
    blocking = [r for r in rec["reviews"] if r["state"] == "CHANGES_REQUESTED"]
    if (
        hold_cr
        and hold_cr.get("dismissed")
        and not any(
            r["login"] == config.operator_login(cfg) and r["state"] == "APPROVED"
            for r in rec["reviews"]
        )
    ):
        return PENDING, (
            f"held earlier ({hold_cr['reason'][:60]}): its change request was dismissed; waiting for "
            "your explicit re-approval"
        )
    if (
        hold_cr
        and [r["login"] for r in blocking] == [config.operator_login(cfg)]
        and rec["head"] != hold_cr["head"]
    ):
        why = f"held earlier ({hold_cr['reason'][:60]})"
        if config.operator_login(cfg) in (rec.get("review_requests") or []):
            return PENDING, f"{why}: waiting for your explicit re-approval of the fix"
        return (
            PENDING,
            f"{why}: {rec['author']} re-requests your review once the fix is in",
        )
    if blocking:
        who = ", ".join(
            r["login"]
            + (
                " (our account: only you can clear it)"
                if r["login"] == config.operator_login(cfg)
                else ""
            )
            for r in blocking
        )
        return FAIL, f"CHANGES_REQUESTED by {who}"
    if rec["review_decision"] == "APPROVED":
        return github_blocks(rec) or (PASS, "approved")
    approvers = {r["login"] for r in rec["reviews"] if r["state"] == "APPROVED"}
    if rec["review_decision"] is None and owners and areas and approvers:
        missing = [
            a
            for a in areas
            if not approvers
            & {m for t in cfg["codeowner_teams"].get(a, []) for m in owners.get(t, [])}
        ]
        if not missing:
            return github_blocks(rec) or (
                PASS,
                f"code-owner approval for {', '.join(areas)} "
                f"({', '.join(sorted(approvers))})",
            )
        return PENDING, "needs a code-owner approval for " + ", ".join(
            f"{a} ({'/'.join(cfg['codeowner_teams'].get(a, []))})" for a in missing
        )
    return PENDING, f"review decision {rec['review_decision'] or 'none'}"


def approvals(cfg: dict, st: dict, rec: dict) -> dict:
    """A PR's approvals against CODEOWNERS: the rocKE areas it touches, who approved, the areas still missing an owner's
    approval, which of those the user can give, whether the user's review is requested, and who requests changes.
    Approvals at older heads count: the develop ruleset does not dismiss stale reviews. `other` means GitHub still
    requires a review although every rocKE area is covered (owners outside the rocKE tree, or a count).
    """
    me = config.operator_login(cfg)
    owners = (st.get("codeowners") or {}).get("teams") or {}
    teams = {t for t, logins in owners.items() if me in logins}
    reviews = rec.get("reviews") or []
    approvers = sorted({r["login"] for r in reviews if r["state"] == "APPROVED"})
    areas = (rec.get("files_cache") or {}).get("areas") or []
    approved = rec.get("review_decision") == "APPROVED"
    missing = (
        []
        if approved
        else [
            a
            for a in areas
            if not set(approvers)
            & {
                m
                for t in cfg.get("codeowner_teams", {}).get(a, [])
                for m in owners.get(t, [])
            }
        ]
    )
    return {
        "areas": areas,
        "approvers": approvers,
        "missing": missing,
        "approved": approved,
        "mine": [
            a for a in missing if teams & set(cfg.get("codeowner_teams", {}).get(a, []))
        ],
        "requested": bool(({me} | teams) & set(rec.get("review_requests") or [])),
        "my_state": next((r["state"] for r in reviews if r["login"] == me), None),
        "changes": sorted(
            r["login"] for r in reviews if r["state"] == "CHANGES_REQUESTED"
        ),
        "other": not approved
        and not missing
        and rec.get("review_decision") == "REVIEW_REQUIRED",
    }


def required_missing(rec: dict) -> dict:
    """Required checks (develop's ruleset) not yet passed at the PR's current head, when known: name -> state."""
    req = rec.get("required") or {}
    if req.get("head") != rec.get("head"):
        return {}
    return {k: v for k, v in (req.get("checks") or {}).items() if v != "success"}


def gate_ci(rec: dict) -> tuple:
    g = {
        "SUCCESS": (PASS, "CI green"),
        "PENDING": (PENDING, "CI running"),
        "EXPECTED": (PENDING, "CI expected"),
        "NONE": (PENDING, "no CI yet"),
    }.get(rec["ci_rollup"], (FAIL, f"CI {rec['ci_rollup']}"))
    left = required_missing(rec)
    if (
        g[0] == PASS and left
    ):  # every reported check passed, but a required one never reported or still runs
        bad = {k: v for k, v in left.items() if v == "failure"}
        if bad:
            return FAIL, "required check failed: " + ", ".join(bad)
        return PENDING, "required check not passed yet: " + ", ".join(
            f"{k} ({v})" for k, v in left.items()
        )
    return g


def hold_labels(cfg: dict, rec: dict) -> list[str]:
    """Labels that put a PR's merge candidacy on hold until removed, e.g. "DO NOT MERGE: documentation", "on hold",
    "WIP", "Not ready to Review"."""
    pats = [re.compile(p, re.I) for p in registry.value(cfg, "hold_label_patterns")]
    return [lab for lab in rec.get("labels") or [] if any(p.search(lab) for p in pats)]


def gate_deps(cfg: dict, st: dict, rec: dict, open_by_ref: dict) -> tuple:
    deps = []
    if rec["base"] != "develop":
        base_pr = open_by_ref.get(rec["base"])
        deps.append(
            f"stacked on #{base_pr}" if base_pr else f"base branch {rec['base']}"
        )
    held = hold_labels(cfg, rec)
    if held:
        deps.append(
            "label "
            + ", ".join(f"'{lab}'" for lab in held)
            + " (out of the queue until removed)"
        )
    if st.get("frozen"):
        deps.append("queue frozen: " + (st["frozen"].get("reason") or ""))
    return (FAIL, "; ".join(deps)) if deps else (PASS, "no dependencies")


def github_conflicting(rec: dict) -> bool:
    """GitHub reports CONFLICTING, or UNKNOWN with CONFLICTING as its last definite answer: it recomputes for a while
    after develop moves, and a conflicting PR must not read as ready for approval meanwhile.
    """
    m = rec.get("mergeable")
    return m == "CONFLICTING" or (
        m == "UNKNOWN" and rec.get("mergeable_known") == "CONFLICTING"
    )


def gate_fresh_merge(cfg: dict, rec: dict, local: bool) -> tuple:
    if github_conflicting(rec):
        return FAIL, "conflicts with develop (GitHub)"
    if not local:
        return (
            (PASS, "mergeable (GitHub)")
            if rec["mergeable"] == "MERGEABLE"
            else (PENDING, "mergeability unknown")
        )
    try:
        conflicts = gitops.merge_conflicts(
            cfg["paths"]["repo"], DEVELOP, ensure_pr_ref(cfg, rec)
        )
    except (gitops.GitError, OSError) as exc:
        return PENDING, f"merge check failed: {str(exc)[:80]}"
    if conflicts is None:
        return PENDING, "merge check undecided"
    if conflicts:
        return FAIL, "conflicts with today's develop: " + ", ".join(
            c.rsplit("/", 1)[-1] for c in conflicts[:4]
        )
    return PASS, "merges cleanly with today's develop"


_TREES: dict[str, str] = {}


def rocke_tree(cfg: dict, sha: str | None) -> str | None:
    """Git tree id of the rocKE subtree at a develop commit (cached); None when unknown."""
    if not sha:
        return None
    if sha not in _TREES:
        t = gitops.git(
            cfg["paths"]["repo"],
            "rev-parse",
            f"{sha}:{paths.repository_folder(cfg)}",
            check=False,
        ).strip()
        if len(t) != 40 or any(c not in "0123456789abcdef" for c in t):
            return None
        _TREES[sha] = t
    return _TREES[sha]


def gates_dynamic(rec: dict, develop_sha: str, develop_tree: str | None = None) -> dict:
    """G5-G7 from the latest gate job for this head. Fresh while develop's rocKE tree is unchanged: develop moves
    outside rocKE cannot change a rocKE gate result (the two-tier model). Pending otherwise.
    """
    res = rec.get("gates") or {}
    same = res.get("develop") == develop_sha or bool(
        develop_tree and res.get("develop_tree") == develop_tree
    )
    fresh = res.get("head") == rec["head"] and same
    out = {
        g: (
            tuple(res["results"][g])
            if fresh and g in res.get("results", {})
            else (PENDING, "gate job not run for this head + develop")
        )
        for g in ("G5", "G6", "G7")
    }
    # A drift check that failed for this head on this develop fails G6 too (drift.py: tests that fail on today's develop
    # never leave the gates pending). A dismissed run does not count.
    from .drift import (
        counts as drift_counts,
    )  # drift imports review, which imports this module

    d = rec.get("drift") or {}
    if drift_counts(d, rec["head"], develop_sha, develop_tree):
        out["G6"] = (
            FAIL,
            f"tests fail on today's develop (drift check): {d.get('detail') or ''}"[
                :200
            ],
        )
    return out


def disruption(cfg: dict, st: dict, cands: list[dict]) -> dict:
    """For each candidate A: the other candidates that would need a rebase or a re-bless if A lands first. The fallback
    when the merge plan has no develop merge to work from (mergeplan.prepare): heads merged on their own merge base,
    plus shared hot spots."""
    pairs = st.setdefault("pair_cache", {})
    hot = {r["n"]: set(pr_hotspots(cfg, r)["hot"]) for r in cands}
    score = {r["n"]: [] for r in cands}
    live = set()
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            key = f"{a['head'][:12]}:{b['head'][:12]}"
            live.add(key)
            if key not in pairs:
                c = gitops.merge_conflicts(
                    cfg["paths"]["repo"],
                    f"refs/remotes/origin/pr{a['n']}",
                    f"refs/remotes/origin/pr{b['n']}",
                )
                pairs[key] = {"conflicts": (c or [])[:6], "undecided": c is None}
            why = sorted(hot[a["n"]] & hot[b["n"]]) + (
                ["files"] if pairs[key]["conflicts"] else []
            )
            if why:
                score[a["n"]].append((b["n"], why))
                score[b["n"]].append((a["n"], why))
    for k in [k for k in pairs if k not in live]:
        del pairs[k]
    return score


USER_PRIORITY = "your priority"


def refresh_develop_fixes(st: dict, incidents: list[dict]) -> dict[str, list[str]]:
    """PRs that fix a develop break, which climb to the top of the review and merge queues:
    - the fix PR recorded on an open incident (`rk incident-note <id> fixed_by N`);
    - an open PR that closes an open develop issue of either kind ("Fixes #N");
    - a fix candidate an open develop issue names ("Fix candidate: #N", "Fixed by #N");
    - a PR linked to an open develop issue once verified (`rk fix-link`).
    Stored as st["develop_fixes"]: open PR -> the breaks it fixes ("incident <id>", "<label> #N", the issue's
    label, breakage.entry_label). Review P0, merge tier 0, analysis first.

    PRs the user floated to the top (`rk prioritize N`) get the same treatment, listed as "your priority: WHY".
    """
    fixes: dict[str, set[str]] = {}
    for inc in incidents:
        if inc.get("status") == "open" and inc.get("fixed_by"):
            fixes.setdefault(str(inc["fixed_by"]), set()).add(
                f"incident {inc.get('id') or '?'}"
            )
    pm = st.get("post_merge") or {}
    kinds = {i["n"]: entry_label(i) for i in pm.get("open") or []}  # each by its label
    for issue, prs in list((pm.get("fixes") or {}).items()) + list(
        (st.get("fix_links") or {}).items()
    ):
        if int(issue) in kinds:
            for n in prs:
                fixes.setdefault(str(n), set()).add(f"{kinds[int(issue)]} #{issue}")
    for k, rec in st["prs"].items():
        for i in rec.get("closes") or []:
            if i in kinds:
                fixes.setdefault(k, set()).add(f"{kinds[i]} #{i}")
    # The user's priorities (`rk prioritize N`) climb as develop fixes do.
    for k, p in (st.get("user_priority") or {}).items():
        fixes.setdefault(k, set()).add(
            f"{USER_PRIORITY}: {p.get('why') or 'floated to the top'}"
        )
    st["develop_fixes"] = {
        k: sorted(v)
        for k, v in fixes.items()
        if (st["prs"].get(k) or {}).get("status") == "open"
    }
    return st["develop_fixes"]


def fixes_develop(why: list[str]) -> bool:
    """True when a top-priority PR fixes a develop break; False when only the user floated it to the top."""
    return any(not w.startswith(USER_PRIORITY) for w in why)


def active_holds(st: dict) -> dict:
    """Holds still in force: the PR is open and still at the held head. A push ends a hold (the author acted)."""
    out = {}
    for k, h in (st.get("holds") or {}).items():
        rec = st["prs"].get(k)
        if rec and rec.get("status") == "open" and rec["head"] == h["head"]:
            out[k] = h
    return out


def abi_only(cfg: dict, files: list[str] | None) -> bool:
    """A conflict only on the ABI counters (abi.h, and its mirror in portable_ir/src/abi.py): a lane matter. Every lane
    member conflicts with every other there (a lane's PRs form a textual clique on abi.h), and the
    lane serializes them whatever the order."""
    return bool(files) and all(
        f.endswith((*hot_paths(cfg, "ABI"), "portable_ir/src/abi.py")) for f in files
    )


def _lane_only(cfg: dict, st: dict, a: dict, b: dict, why: list[str]) -> bool:
    """True when a and b overlap only on the ABI counter: a lane matter, which readiness order decides."""
    if set(why) - {"ABI", "files"}:
        return False
    if "files" not in why:
        return True
    pc = st.get("pair_cache") or {}
    p = (
        pc.get(f"{a['head'][:12]}:{b['head'][:12]}")
        or pc.get(f"{b['head'][:12]}:{a['head'][:12]}")
        or {}
    )
    return abi_only(cfg, p.get("conflicts") or [])


def apply_holds(cfg: dict, st: dict, rows: list[dict], score: dict) -> None:
    """A held PR keeps its rank but is neither batched nor armed. A PR whose landing would force a held PR to rebase
    or re-bless waits too (the "no impact" rule). An overlap only on the ABI counter is exempt: ABI lanes follow
    readiness order, so the next ready lane member may take the number.

    With the merge plan, `score` holds only textual conflicts on today's develop (only conflicts batch and hold; a
    shared golden shard or CMake list is care, shown but never acted on)."""
    holds = active_holds(st)
    for r in rows:
        h = holds.get(str(r["n"]))
        if h:
            r["held"] = h
            r["blockers"].insert(0, f"HOLD ({h['status']}): {h['reason']}")
            r["static_ok"] = False
            continue
        rec = st["prs"][str(r["n"])]
        hits = [
            f"#{m}"
            for m, why in score.get(r["n"], [])
            if str(m) in holds and not _lane_only(cfg, st, rec, st["prs"][str(m)], why)
        ]
        if hits:
            r["blockers"].append(
                f"waits: landing it first would force held {', '.join(hits)} to rebase or re-bless"
            )
            r["static_ok"] = False


def hold_bound(r: dict) -> bool:
    """A queue row the hold rule keeps from landing: held, or waiting for a held PR it conflicts with (apply_holds). It
    never blocks its ABI lane, as the rule exempts ABI overlaps: it renumbers after the lane's other claims, and its
    ABI conflicts in the merge plan are shown, not ordered."""
    return bool(r.get("held")) or any(
        b.startswith("waits:") for b in r.get("blockers") or []
    )


def build_queue(cfg: dict, st: dict, develop_sha: str) -> list[dict]:
    from . import (
        mergeplan,
    )  # here, not at the top: the plan builds on this module's gates

    open_recs = [
        r for r in st["prs"].values() if r.get("status") == "open" and not r["draft"]
    ]
    open_by_ref = {r["head_ref"]: r["n"] for r in open_recs}
    stack_bases = {
        open_by_ref[r["base"]] for r in open_recs if r["base"] in open_by_ref
    }
    fix_prs = {
        int(k) for k in st.get("develop_fixes") or {}
    }  # tier 0: develop fixes land first
    owners = (st.get("codeowners") or {}).get("teams")
    dtree = rocke_tree(cfg, develop_sha)
    st["develop"]["rocke_tree"] = dtree
    rows = []
    for rec in open_recs:
        g = {
            "G1": gate_review(cfg, rec),
            "G3": gate_ci(rec),
            "G8": gate_deps(cfg, st, rec, open_by_ref),
        }
        candidate = (
            g["G1"][0] != FAIL and g["G8"][0] == PASS and not github_conflicting(rec)
        )
        # Code-owner areas for candidates, develop fixes and the user's priorities, and every PR heading for a merge
        # (our latest verdict 🟢/🟡, maybe before a later push), so "what's left" can say whose approval is missing;
        # cached per head.
        heading = (rec.get("ours") or {}).get("verdict") in GOOD_VERDICTS and g["G8"][
            0
        ] == PASS
        areas = (
            pr_hotspots(cfg, rec)["areas"]
            if candidate or heading or rec["n"] in fix_prs
            else None
        )
        g["G2"] = gate_humans(
            cfg, rec, owners, areas, (st.get("hold_reviews") or {}).get(str(rec["n"]))
        )
        g["G4"] = gate_fresh_merge(cfg, rec, local=candidate)
        g.update(gates_dynamic(rec, develop_sha, dtree))
        rows.append(
            {
                "n": rec["n"],
                "author": rec["author"],
                "title": rec["title"],
                "size": rec["size"],
                "gates": {k: list(v) for k, v in sorted(g.items())},
                "blockers": [
                    f"{k}: {v[1]}" for k, v in sorted(g.items()) if v[0] == FAIL
                ],
                "candidate": candidate and g["G4"][0] != FAIL,
                "reapproval": g["G2"][0] == PENDING
                and g["G2"][1].startswith("held earlier"),
                "static_ok": all(
                    g[k][0] == PASS for k in ("G1", "G2", "G3", "G4", "G8")
                ),
                "tier": (
                    0 if rec["n"] in fix_prs else 1 if rec["n"] in stack_bases else 2
                ),
                "cleared": rec["n"] in st.get("cleared", []),
            }
        )
    cands = [st["prs"][str(r["n"])] for r in rows if r["candidate"]]
    # The ABI claims first: two claims in one lane conflict in the plan, and the lanes
    # then follow the plan's order (abi_lanes below).
    refresh_claims(cfg, st, open_recs, develop_sha)
    # The merge plan: textual conflicts on today's develop among the near-merge PRs
    # order, batch and hold; without a develop merge to work from, the head-to-head check and `d` as before.
    mx = mergeplan.prepare(cfg, st, rows, develop_sha)
    if mx is not None:
        score = mergeplan.conflict_score(mx)
    else:
        score = (
            disruption(cfg, st, cands)
            if len(cands) > 1
            else {r["n"]: [] for r in cands}
        )
    for r in rows:
        rec = st["prs"][str(r["n"])]
        r["hot"] = (rec.get("files_cache") or {}).get("hot", [])
        r["disrupts"] = [f"#{n}({'+'.join(w)})" for n, w in score.get(r["n"], [])]
        r["disruption"] = len(score[r["n"]]) if r["n"] in score else None
    apply_holds(cfg, st, rows, score)
    mp = mergeplan.make(cfg, st, rows, mx) if mx is not None else None
    # G9: ABI lanes, ordered by readiness (all other static gates green first), then by the queue's sort key.
    lanes = abi_lanes(cfg, st, rows, open_recs, develop_sha)
    waived = {tuple(x) for x in st.get("abi_waived", [])}
    for r in rows:
        g9 = abi.gate_abi(st["prs"][str(r["n"])], lanes, waived)
        r["gates"]["G9"] = list(g9)
        if g9[0] == FAIL:
            r["blockers"].append(f"G9: {g9[1]}")
        r["static_ok"] = r["static_ok"] and g9[0] == PASS
    # Two tiers. A candidate has its approvals and a clean review at head (G1, G2, G8);
    # its merge analysis (gate jobs) runs early even while CI is pending or red. Ready adds CI green, a clean merge
    # with develop and its ABI lane turn; only ready PRs are batched, proposed or armed. A held PR's fix (pushed since
    # the hold, with our change request still standing) is a candidate too: its analysis re-gates the fix before the
    # user re-approves.
    for r in rows:
        g = r["gates"]
        r["stage"] = (
            "held"
            if r.get("held")
            else (
                "ready"
                if r["static_ok"]
                else (
                    "candidate"
                    if g["G1"][0] == PASS
                    and g["G8"][0] == PASS
                    and (g["G2"][0] == PASS or r["reapproval"])
                    else None
                )
            )
        )
    # The plan's landing order (wave by wave); without a plan, (tier, d, size) as before.
    ready = sorted(
        (r for r in rows if r["static_ok"]),
        key=lambda r: (
            r.get("plan_rank") is None,
            r.get("plan_rank") or 0,
            r["tier"],
            r["disruption"] or 0,
            r["size"],
            r["n"],
        ),
    )
    for i, r in enumerate(ready, 1):
        r["position"] = i
    # The head wave feeds the batches: a ready PR in a later wave waits for the PRs it conflicts with.
    st["batches"] = batch.build_batches(
        cfg, st, [r for r in ready if r.get("plan_go")] if mp else ready, lanes
    )
    if (
        mp
    ):  # the Ready stage: a top priority ready in a later wave is the user's call (mergeplan.ready_early)
        mergeplan.finish(cfg, st, mp, {r["n"] for r in ready})
    return ready + sorted(
        (r for r in rows if not r["static_ok"]),
        key=lambda r: (
            r["stage"] not in ("candidate", "held"),
            not r["candidate"],
            len(r["blockers"]),
            r["n"],
        ),
    )


def refresh_claims(
    cfg: dict, st: dict, open_recs: list[dict], develop_sha: str
) -> None:
    """Develop's ABI counters and every open PR's ABI claims at its head (abi.pr_claims: cached per head and develop's
    abi.h), before the merge plan: two claims in one lane are a conflict there, and who
    is in a lane does not depend on the order the plan then gives the lanes (abi_lanes).
    """
    dev = st["develop"].get("abi")
    if not dev or dev.get("sha") != develop_sha:
        c = abi.develop_counters(cfg)
        st["develop"]["abi"] = {
            "sha": develop_sha,
            "binary": c["binary"],
            "recipe": c["recipe"],
        }
    for rec in open_recs:
        try:
            abi.pr_claims(cfg, rec, ensure_pr_ref(cfg, rec))
        except (gitops.GitError, OSError) as exc:
            rec["abi_cache"] = {
                "head": None,
                "lanes": {},
                "families": [],
                "notes": [f"ABI scan failed: {exc}"[:150]],
            }


def abi_lanes(
    cfg: dict, st: dict, rows: list[dict], open_recs: list[dict], develop_sha: str
) -> dict:
    """The ABI lanes from the claims refresh_claims read, in the order below; develop's counters only when read on this
    develop."""
    dev = st["develop"].get("abi") or {}
    dev = dev if dev.get("sha") == develop_sha else {}
    # Readiness order: ready PRs first (queue order); then, among the rest, the top priorities (tier 0) before the
    # closest to ready (fewest failing gates other than G9), so a top priority never sinks in its lane when an infra
    # failure turns its CI red. The lane head is still the first member whose other gates are green. With a merge plan,
    # its landing order comes first, and only its head wave counts as ready: the lane head is the lane member in the
    # earliest wave. The
    # plan never puts two claims of one lane in one wave (its ABI conflicts), so the lanes follow its order. A PR the
    # hold rule keeps waiting goes behind the rest of its lane (hold_bound): it never blocks the lane.
    planned = any(r.get("plan_rank") is not None for r in rows)
    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("plan_rank") is None,
            hold_bound(r),
            r.get("plan_rank") or 0,
            not r["static_ok"],
            r["tier"],
            len([b for b in r["blockers"] if not b.startswith("G9")]),
            r["disruption"] or 0,
            r["size"],
            r["n"],
        ),
    )
    lanes = abi.build_lanes(
        dev,
        open_recs,
        [r["n"] for r in ordered],
        {r["n"] for r in rows if r["static_ok"] and (r.get("plan_go") or not planned)},
    )
    st["abi_lanes"] = lanes
    return lanes
