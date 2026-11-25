# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Review automation. The tick (or the review driver) keeps up to the parallel reviews' count of drafts running at once:
each pipe starts the reviewer model's worker on one PR, then the generated check script in a heavy slot, and
reports REVIEW_READY. The orchestrator (Claude) verifies each with a subagent, in parallel, and posts through
`rk post`.

Follow-ups are classified first by comparing the PR's own change before and after the push, using only the added and
removed lines, not context. When only rebase context changed, the review still stands: no follow-up is needed, and the
new head is recorded as equivalent to the reviewed one.

A push of merges only (develop brought in, conflict resolutions included) runs its review's worker and its free checks
at once (no gap), since a merge can break a PR (a conflict resolution can leave abi.py and abi.h at different ABI
versions): when the checks come back clean, nothing is posted and the head is recorded as covered by our review; when
they flag anything (a failing or unrun check, an ABI note, a probe that changed state, a new CI failure), the review goes
to verification as usual, saying why. Other follow-ups wait until our last review of the PR is
`followup_min_gap_min` old (plan.gap_until).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess

from . import (
    bindings,
    ci_runs,
    config,
    draft_lint,
    draft_stage,
    events,
    gh,
    gitops,
    idle,
    jobs,
    known_infra,
    lanes,
    nogpu,
    paths,
    peer_carry,
    prcode,
    registry,
    scaler,
    slots,
)
from .clashes import SECTION as CLASH_SECTION
from .overlaps import SECTION as OVERLAP_SECTION
from .peer_checks import CHECKS as PEER_CHECKS
from .peer_checks import tracker_allow
from .plan import gap_until, last_review_posts
from .queue import DEVELOP, base_of, ensure_pr_ref, hot_paths, path_hits
from .snapshot import covers, same_commit
from .store import minutes_since, now, settle_left

ENGINE_CORE = (
    "platform/python/rocke/core/",
    "platform/cpp/core/",
    "platform/python/rocke/portable_ir/",
    "platform/cpp/include/rocke/ir",
    "platform/cpp/include/rocke/lower_",
    "platform/python/rocke/ir",
)
# The differential golden's LLVM flavors exactly as CI runs them, read from the tree's CTest entry ($P is the platform
# dir), so a new lane is checked as soon as the tree has it (a fixed list misses a lane CI adds). Older trees fall
# back to llvm20/22/23.
DG_FLAVORS = (
    "DGF=$(grep -oE 'rocke_differential_golden_ll .*--flavor [a-z0-9,]+' $P/CMakeLists.txt | head -1 | "
    "grep -oE '[a-z0-9,]+$'); DGF=${DGF:-llvm20,llvm22,llvm23}"
)
# The flags the differential golden's CI commands pass to run_diff.py: a tree whose `run_diff.py --help` lacks one reads
# NOT RUN, never a verdict (a PR on an older base branch).
DG_FLAGS = ("--mode", "--python-only", "--check-golden", "--flavor", "--canonical")
# The per-flavor loops (byte identity, library parity) run at the tree's distinct LLVM emission columns: an aliased lane
# emits as its source, so it is checked once. A tree that lists llvm24 has it as a column of its own, which fixed
# llvm20/22/23 loops would miss (L-325). Older trees fall back to those three. The tree's own Python tells them, so it
# runs as PR code (`prc`, rkmon_lib/prcode.py).
FLAVORS_FN = (
    "flavors() { local f; f=$(prc -C $P/python python -B -c 'from rocke.core.llvm_profile import "
    'LLVM_EMISSION_FLAVORS as F; print(" ".join(F))\' 2>/dev/null | tail -1); case "$f" in llvm*) '
    "echo \"$f\" ;; *) echo 'llvm20 llvm22 llvm23' ;; esac; }"
)
# Who makes our merge commits (their author e-mail): worker/mkwt.py merges develop into the PR head, so the head is the
# merge's first parent; a batch step (merge.batch_script) merges the head into develop and the earlier members, so the
# head is its second.
MKWT_EMAIL, BATCH_EMAIL = "r@x", "rkmon@localhost"


# --- follow-up classification ---------------------------------------------------------------------------------------


def _patch_fingerprint(diff: str) -> str:
    """A file's patch as the multiset of its added and removed lines. Order is ignored: once develop changes the base,
    git may align a repeated line (a `# ----` separator) differently in the same patch, which would read as a
    substantive change."""
    body = sorted(
        line
        for line in diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )
    return hashlib.sha1("\n".join(body).encode()).hexdigest()


def _fingerprint(repo: str, mb: str, ref: str, path: str) -> str:
    return _patch_fingerprint(
        gitops.git(repo, "diff", "-U0", mb, ref, "--", path, check=False)
    )


# The pushes our review already covers: no follow-up starts, and the new head is recorded as equivalent to the reviewed
# one (`carry`). A rebase plus a golden re-record is covered only after a 🟢/🟡 review (start_review); a merge-only push
# once its checks come back clean (merge_only_flags).
COVERED = ("same-head", "rebase-only")


def _pr_files(
    repo: str, old: str, new: str, lazy: bool = True, base: str = DEVELOP
) -> set[str]:
    """The files the PR changes at either head, from its merge base with its base branch (`base`, a ref: develop's by
    default). No rename pairing: a rename lists both paths, and only trees are read (a partial clone fetches no
    blob)."""
    out: set[str] = set()
    for head in (old, new):
        mb = gitops.git(repo, "merge-base", base, head, lazy=lazy).strip()
        out |= set(
            gitops.git(
                repo, "diff", "--no-renames", "--name-only", mb, head, lazy=lazy
            ).splitlines()
        )
    return out


def merge_only(
    repo: str, old: str, new: str, lazy: bool = True, base: str = DEVELOP
) -> dict | None:
    """Whether the push from our reviewed head `old` to `new` is merges only: `new` descends from `old`, at least one new
    commit is a merge, and no new non-merge commit outside develop changes a file the PR changes at either head. A
    merge's conflict resolution may change the PR's files: it still counts. {"merges": k, "own": m} (m new non-merge
    commits, none on the PR's files, e.g. an empty commit), else None.

    Develop (or `base`, the PR's base branch's ref) is the mirror's, as last fetched: a commit newer than that reads as
    one of the PR's own, so the push gets a full follow-up, never a wrong carry. Commits and trees only, so a partial
    clone fetches no blob; `lazy=False` fetches nothing at all (a pruned old head is then no ancestor).
    """
    if not gitops.is_ancestor(repo, old, new, lazy=lazy):
        return None
    revs = [
        ln.split()
        for ln in gitops.git(
            repo, "rev-list", "--parents", new, "--not", old, base, lazy=lazy
        ).splitlines()
        if ln.strip()
    ]
    merges = sum(1 for r in revs if len(r) > 2)  # the commit, then its parents
    own = len(revs) - merges
    if not merges:
        return None
    if own:
        touched = {
            f
            for f in gitops.git(
                repo,
                "log",
                "--no-merges",
                "--no-renames",
                "--name-only",
                "--format=",
                new,
                "--not",
                old,
                base,
                lazy=lazy,
            ).splitlines()
            if f
        }
        if touched & _pr_files(repo, old, new, lazy=lazy, base=base):
            return None
    return {"merges": merges, "own": own}


def classify(cfg: dict, rec: dict, old_head: str) -> dict:
    """The push from our reviewed head `old_head` to PR rec's head, against its base branch (develop, or the feature
    branch it targets)."""
    repo, K = cfg["paths"]["repo"], paths.repository_folder(cfg)
    new = ensure_pr_ref(cfg, rec)
    base = base_of(cfg, rec, fetch=True)
    new_sha = gitops.git(repo, "rev-parse", new).strip()
    if same_commit(old_head, new_sha):
        return {
            "kind": "same-head",
            "old": old_head,
            "new": new_sha,
            "changed_files": [],
            "new_commits": [],
        }
    if not gitops.has_commit(repo, old_head):
        gitops.git(
            repo, "fetch", "-q", "--filter=blob:none", "origin", old_head, check=False
        )
    if not gitops.has_commit(repo, old_head):
        return {
            "kind": "unknown-old-head",
            "old": old_head,
            "new": new_sha,
            "changed_files": [],
            "new_commits": [],
        }
    mb_old = gitops.git(repo, "merge-base", base, old_head).strip()
    mb_new = gitops.git(repo, "merge-base", base, new).strip()
    files = set(
        gitops.git(repo, "diff", "--name-only", mb_old, old_head).splitlines()
    ) | set(gitops.git(repo, "diff", "--name-only", mb_new, new).splitlines())
    changed = sorted(
        f
        for f in files
        if _fingerprint(repo, mb_old, old_head, f) != _fingerprint(repo, mb_new, new, f)
    )
    noise = list(
        cfg.get("rebase_noise_paths") or hot_paths(cfg, "RG") + hot_paths(cfg, "DG")
    )
    commits = gitops.git(
        repo,
        "log",
        "--no-merges",
        "--format=%h %s",
        new,
        "--not",
        old_head,
        base,
        check=False,
    ).splitlines()
    # Merges only, whatever they did to the PR's lines (a conflict resolution), comes first: the checks decide whether
    # anything is posted (start_review, merge_only_flags).
    mo = merge_only(repo, old_head, new_sha, base=base)
    if mo:
        kind = "merge-only"
    elif not changed:
        kind = "rebase-only"
    elif all(path_hits(noise, f) for f in changed):
        kind = "rebase+re-record"
    else:
        kind = "substantive"
    short = [f[len(K) + 1 :] if f.startswith(K + "/") else f for f in changed]
    return {
        "kind": kind,
        "old": old_head,
        "new": new_sha,
        "changed_files": short[:40],
        "n_changed": len(changed),
        "new_commits": commits[:20],
        **({"merges": mo["merges"]} if mo else {}),
    }


def classify_pr(cfg: dict, rec: dict) -> dict:
    """`rk classify N` (and `rk review-prep N`'s classify.txt): PR rec's own change at our reviewed head and now, or
    "never reviewed"."""
    old = (rec.get("ours") or {}).get("head")
    return classify(cfg, rec, old) if old else {"kind": "never reviewed"}


def carry(store, rec: dict, c: dict, detail: str = "") -> str:
    """Record PR rec's new head `c["new"]` as covered by our review (a push of kind `c["kind"]`): journal it, the line
    report.equivalent_clears counts, and return the REVIEW_SKIP line."""
    if not covers(rec["ours"], c["new"]):
        rec["ours"].setdefault("equivalent_heads", []).append(c["new"])
    store.journal(
        "review",
        f"#{rec['n']} follow-up skipped: {c['kind']} ({c['old'][:8]} -> {c['new'][:8]}); "
        "our review stands",
    )
    return f"REVIEW_SKIP #{rec['n']}: {c['kind']} since our review{detail}; recorded {c['new'][:10]} as equivalent"


def mark_merges(cfg: dict, st: dict) -> list[str]:
    """The tick's step before the plan: every open PR whose head moved since our review is checked once per head for a
    merge-only push (`merge_only`; commits and trees, no fetch beyond the PR's head), into rec["merge_check"]. Such a
    follow-up is not held back by the gap (plan.gap_until), and its checks decide whether anything is posted
    (merge_only_flags). A git failure is a line, and the next tick tries again.

    A merge-only head recorded as covered while its merge_check has no merge count has had no checks: it is re-opened
    (MERGE_ONLY_REOPENED), so they run."""
    repo = cfg["paths"]["repo"]
    out = []
    for k, rec in sorted(st["prs"].items(), key=lambda kv: int(kv[0])):
        ours = rec.get("ours") or {}
        old, new = ours.get("head"), rec.get("head")
        if rec.get("status") != "open" or rec.get("draft") or not old or not new:
            continue
        seen = rec.get("merge_check") or {}
        if (
            seen.get("merge_only")
            and "merges" not in seen
            and seen.get("old") == old
            and seen.get("new") == new
            and new in (ours.get("equivalent_heads") or [])
        ):
            ours["equivalent_heads"].remove(new)
            rec.pop("merge_check")
            seen = {}
            out.append(
                f"MERGE_ONLY_REOPENED #{k}: {new[:10]} was recorded as covered before its checks ran; they run now"
            )
        if covers(ours, new) or (seen.get("old") == old and seen.get("new") == new):
            continue  # covered, or checked at this head already
        try:
            ensure_pr_ref(cfg, rec)
            mo = merge_only(repo, old, new, lazy=False, base=base_of(cfg, rec))
        except (gitops.GitError, OSError, subprocess.SubprocessError) as exc:
            out.append(f"(merge-only check of #{k} failed: {str(exc)[:160]})")
            continue
        rec["merge_check"] = {
            "old": old,
            "new": new,
            "merge_only": bool(mo),
            **(mo or {}),
        }
    return out


# --- a merge-only head's checks: clean, or flagged ------------------------------------------------------------------

# A check that skipped because there was nothing to check is clean; one that could not run (low memory, a time box) is
# not.
NOTHING_TO_CHECK = re.compile(
    r"no test to run|none is in an installed CTest entry|the PR deletes them all"
)
# Sections of leads for the reviewers, which flag no merge-only head (and feed no gate): the clash scan's findings are
# the other PR's change as much as this one's (its review shows them too), and a merge-only push changes none of the
# PR's own checks, so the mutation sweep's survivors stand as reviewed. The CTest tiers, moved-file references and
# engine import probe compare the head with its merge base: leads for the verifier to confirm, not verdicts.
LEAD_SECTIONS = (
    "clash",
    "mutants",
    "labels",
    "movedrefs",
    "imports",
    "since",
    "docrefs",
    "importers",
    "devtests",
    "cicond",
    "tidy",
    "winsim",
) + tuple(key for _tag, key, _title, lead in PEER_CHECKS.values() if lead)
PROBE_LINE = re.compile(r"^(\S+): (REPRODUCES|FIXED|ERROR)\b", re.M)


def _section(text: str, title: str) -> str | None:
    """The body of the checks.md section whose heading starts `## <title>`; None when the report has none."""
    m = re.search(rf"^## {re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1) if m else None


def probe_results(text: str) -> dict[str, str] | None:
    """Our saved probes' results in a checks.md (probe name -> REPRODUCES, FIXED or ERROR); None without the section."""
    sec = _section(text, "Our open findings, re-probed")
    return None if sec is None else dict(PROBE_LINE.findall(sec))


def abi_notes(text: str) -> list[str]:
    """The ABI scan's notes in a checks.md (what G9 waits for `rk abi-ok` on), without their prefix."""
    return [
        re.sub(r"^NOTE(?: \([^)]*\))?: ?", "", ln)
        for ln in (_section(text, "ABI") or "").splitlines()
        if ln.startswith("NOTE")
    ]


def _checks_text(cfg: dict, n: int) -> str:
    p = pathlib.Path(f"{paths.get(cfg, 'workers')}/pr{n}/checks/checks.md")
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def probe_baseline(cfg: dict, rec: dict, n: int) -> dict[str, str] | None:
    """Our probes' results at a head our review covers: the last recorded (rec["probes_seen"], kept by
    _record_checks_gates), else the previous review's checks when they ran at such a head (read before a new worker wipes
    workers/prN). None when neither is known."""
    seen = rec.get("probes_seen") or {}
    if seen.get("head") and covers(rec.get("ours"), seen["head"]):
        return dict(seen.get("results") or {})
    _, _, head = read_checks_status(f"{paths.get(cfg, 'workers')}/pr{n}/checks")
    if head and covers(rec.get("ours"), head):
        return probe_results(_checks_text(cfg, n)) or {}
    return None


def live_head(cfg: dict, n: int) -> str:
    """PR n's head on GitHub now (GhError when it cannot be read)."""
    return str(gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq=".head.sha")).strip()


def ci_failing(cfg: dict, sha: str) -> set[str]:
    """The check runs whose latest attempt failed at `sha` (as `rk ci-fails` counts them)."""
    return {
        name
        for name, r in ci_runs.latest(ci_runs.check_runs(cfg, sha)).items()
        if ci_runs.failed(r)
    }


CI_NEW_FLAG = "CI fails at the head and not at our reviewed head"
MICRO = "micro"  # REVIEW_READY's word for a merge-only head whose only new items are threads or a title problem


def our_last_review_body(cfg: dict, rec: dict, n: int) -> str:
    """Our last review's body on PR n, posted or kept unposted: its staged copy (pr/ours.md) when staged at the head of
    our last review, else read where post.our_last_review reads it; "" when neither is read.
    """
    from . import post as post_mod

    ours = rec.get("ours") or {}
    try:
        text = pathlib.Path(f"{paths.get(cfg, 'workers')}/pr{n}/pr/ours.md").read_text(
            encoding="utf-8"
        )
        m = re.search(r"^- Head: (\w+)$", text, re.M)
        if m and ours.get("head") and same_commit(m.group(1), ours["head"]):
            return text
    except OSError:
        pass
    try:
        return post_mod.our_last_review(cfg, n, ours)[0]
    except gh.GhError:
        return ""


def peer_split(
    cfg: dict, n: int, pipe: dict, flags: list[str], ours=None
) -> tuple[list[str], list[str], list[str]]:
    """A merge-only head's flags without the item flags its items account for (peer_carry.judge: their items, against
    those at a head our review covers and, for the failing tests and pull-request references, our last review's open
    rows; `ours`: () -> its body), the threads and title items new since, and the checks accounted for.
    """
    out_dir = f"{paths.get(cfg, 'workers')}/pr{n}/checks"
    status, _, _ = read_checks_status(out_dir)
    settled, new = peer_carry.judge(
        status,
        peer_carry.read(out_dir),
        (pipe.get("merge_only") or {}).get("peer"),
        ours,
    )
    rest = [
        f for f in flags if not any(f.startswith(f"{key} fails ") for key in settled)
    ]
    return rest, new, settled


def activity_since_ours(cfg: dict, n: int, rec: dict) -> list[str]:
    """What others left on PR n since our last review (rec["ours"]["at"]: posted, or kept unposted while posting is
    off): reviews, review comments and comments, by anyone but our login and bots, as one item for a merge-only head's
    micro round (a re-review to answer is never carried silently). [] when nothing came, with no review of ours on
    record, or when GitHub cannot be read (the checks' flags decide alone then). Our login's posts are left out by the
    account alone: a review another deployment posted under it is no activity this deployment answers, as no deployment
    answers another's reviews."""
    since = (rec.get("ours") or {}).get("at")
    if not since:
        return []
    me = str(config.operator_login(cfg) or "")
    who = ' | select(.user.type != "Bot")' + (
        f' | select(.user.login != "{me}")' if me.replace("-", "").isalnum() else ""
    )
    counts = []
    try:
        for path, field in (
            (f"pulls/{n}/reviews", "submitted_at"),
            (f"pulls/{n}/comments", "updated_at"),
            (f"issues/{n}/comments", "updated_at"),
        ):
            times = gh.rest(
                f"repos/{cfg['repo']}/{path}?per_page=100",
                paginate=True,
                jq=f"[.[]{who} | .{field}]",
            )
            counts.append(sum(1 for at in times or [] if (at or "") > since))
    except (gh.GhError, KeyError, TypeError):
        return []
    reviews, review_comments, comments = counts
    if not any(counts):
        return []
    return [
        f"since our last review ({since}): {reviews} review{'s' if reviews != 1 else ''}, {review_comments} review "
        f"comment{'s' if review_comments != 1 else ''} and {comments} comment{'s' if comments != 1 else ''} from "
        "others: answer them, and restate each row whose status they change"
    ]


def merge_only_flags(cfg: dict, n: int, pipe: dict) -> list[str]:
    """Why a merge-only head's follow-up still goes to verification, from its finished checks; [] when they are clean:
    every check passes (a skip only when there was nothing to check), the ABI scan
    has no note, no saved probe changed state since our reviewed head (a probe with no earlier result counts as changed
    unless it reproduces: a probe saved for an open finding reproduces where it was written), and no check run fails at
    the head that did not fail at our reviewed head. Those new failures' names go to pipe["ci_new"], for the
    known-infrastructure test (known_infra.classify)."""
    mo = pipe.get("merge_only") or {}
    status, _, head = read_checks_status(f"{paths.get(cfg, 'workers')}/pr{n}/checks")
    if not status:
        return ["the checks wrote no status"]
    flags = (
        []
        if same_commit(head, pipe["head"])
        else [f"the checks ran at {(head or '?')[:10]}, not {pipe['head'][:10]}"]
    )
    if "merge" not in status:
        flags.append("no merge with develop was checked")
    text = _checks_text(cfg, n)
    notes = abi_notes(text)
    for sec, (verdict, detail) in status.items():
        if sec in LEAD_SECTIONS or (sec == "abi" and verdict == "fail" and notes):
            continue  # a lead; or the ABI notes, each listed below
        if verdict == "fail":
            flags.append(f"{sec} fails ({detail[:120]})")
        elif verdict != "pass" and not NOTHING_TO_CHECK.search(detail):
            flags.append(f"{sec} did not run ({detail[:120]})")
    flags += [f"ABI: {note[:200]}" for note in notes]
    before = mo.get("probes")
    for name, state in sorted((probe_results(text) or {}).items()):
        was = (before or {}).get(name)
        if was != state and (was is not None or state != "REPRODUCES"):
            flags.append(f"probe {name}: {was or 'no earlier result'} -> {state}")
    try:
        new = sorted(ci_failing(cfg, pipe["head"]) - ci_failing(cfg, mo["old"]))
    except (gh.GhError, KeyError) as exc:
        return flags + [f"CI not read ({str(exc)[:100]})"]
    pipe["ci_new"] = new
    if new:
        flags.append(
            f"{CI_NEW_FLAG}: {', '.join(new[:4])}"
            + (f" (+{len(new) - 4} more)" if len(new) > 4 else "")
        )
    return flags


# --- what the checks should cover -------------------------------------------------------------------------------


def emitter_families(tree: str, K: str) -> dict[str, str]:
    """family -> parity dir (relative to the rocKE tree) for every <family>_emit.py in the tree."""
    import pathlib

    out = {}
    for rel in ("platform/tests/instances/parity", "library/tests/parity"):
        for p in pathlib.Path(tree, K, rel).glob("*_emit.py"):
            out[p.name[: -len("_emit.py")]] = rel
    return out


def _family_match(fam: str, anchored: bool = False):
    """path -> whether the path is one of family `fam`'s own files: its name, with `_` or `/` between the parts, and
    its arch when the family names one. `anchored`: the name starts a path component, so `gemm` is not the own family
    of `conv_implicit_gemm.py`."""
    m = re.match(r"(gfx\d+[a-z]?)_(.+)", fam)
    arch, rest = (m.group(1), m.group(2)) if m else (None, fam)
    pat = re.compile(
        ("(?:^|/)" if anchored else "") + re.escape(rest).replace("_", "[_/]")
    )
    return lambda f: bool(pat.search(f)) and (arch is None or arch in f)


def touched_families(files: list[str], families: dict[str, str]) -> list[str]:
    """Parity families whose name matches a changed code path. Docs and markdown never select a family."""
    files = [
        f
        for f in files
        if f.endswith((".py", ".c", ".cpp", ".h", ".hpp", ".json"))
        and not re.search(r"(^|/)(docs|dsl_docs)/", f)
    ]
    out = []
    for fam in families:
        match = _family_match(fam)
        if any(match(f) for f in files):
            out.append(fam)
    return sorted(out)


# Kernel-side code that families share. A changed helper there, which no family name matches, selects every family
# whose own files name it by module or header (a gate helper such as `conv_fwd_wmma_gate` is called first by the
# families that use it, and no family name matches the helper).
HELPER_DIRS = (
    "library/kernels/",
    "platform/cpp/instances/",
    "platform/cpp/include/rocke/instance_",
    "platform/python/rocke/instances/",
)
FAMILY_CODE = (
    "library/kernels",
    "library/tests/parity",
    "platform/cpp/instances",
    "platform/tests/instances/parity",
    "platform/python/rocke/instances",
)
GENERIC_STEMS = {
    "__init__",
    "_common",
    "common",
    "utils",
    "helpers",
    "builder",
    "emitter",
    "validate",
}
MAX_HELPER_FAMILIES = 8  # more than this: check every family, as for an engine change
CODE = (".py", ".c", ".cpp", ".h", ".hpp")


def _family_code(source, K: str):
    """The families' kernel-side code (paths relative to the rocKE tree) and a reader for it (paths -> texts), from a
    worktree path or from git at a commit, `("git", repo, sha)`: gate jobs and develop sanity plan before any
    worktree exists."""
    import pathlib
    import subprocess

    if isinstance(source, tuple):
        _, repo, sha = source
        listing = gitops.git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            sha,
            "--",
            *(f"{K}/{d}" for d in FAMILY_CODE),
            check=False,
        )
        code = [p[len(K) + 1 :] for p in listing.splitlines() if p.endswith(CODE)]

        def read(paths: list[str]) -> list[str]:
            p = subprocess.run(
                ["git", "-C", repo, "cat-file", "--batch"],
                capture_output=True,
                timeout=300,
                input="".join(f"{sha}:{K}/{f}\n" for f in paths).encode(),
            )
            out, data, i = [], p.stdout, 0
            while i < len(data) and len(out) < len(paths):
                nl = data.index(b"\n", i)
                hdr = data[i:nl].split()
                if hdr[-1] == b"missing":
                    out.append("")
                    i = nl + 1
                    continue
                size = int(hdr[2])
                out.append(data[nl + 1 : nl + 1 + size].decode("utf-8", "replace"))
                i = nl + 1 + size + 1
            return out + [""] * (len(paths) - len(out))

        return code, read
    root = pathlib.Path(source, K)
    code = [
        str(p.relative_to(root))
        for d in FAMILY_CODE
        for p in (root / d).rglob("*")
        if p.suffix in CODE
    ]
    return code, lambda paths: [
        (root / f).read_text(encoding="utf-8", errors="replace") for f in paths
    ]


def helper_families(
    source, K: str, files: list[str], families: dict[str, str], touched: list[str]
) -> dict[str, list[str]]:
    """helper -> the families (beyond `touched`) one of whose own files names the changed helper `files` hold. The
    family files come from `source`: a worktree path, or `("git", repo, sha)`."""
    import pathlib

    stems = {}
    for f in files:
        p = pathlib.PurePosixPath(f)
        if (
            f.startswith(HELPER_DIRS)
            and p.suffix in CODE
            and not touched_families([f], families)
            and len(p.stem) >= 8
            and "_" in p.stem.strip("_")
            and p.stem not in GENERIC_STEMS
            and not p.stem.startswith("test_")
        ):
            stems[p.stem] = re.compile(rf"\b{re.escape(p.stem)}\b")
    if not stems:
        return {}
    code, read = _family_code(source, K)
    own = {
        fam: [f for f in code if _family_match(fam, anchored=True)(f)]
        for fam in sorted(families)
        if fam not in touched
    }
    paths = sorted({f for fs in own.values() for f in fs})
    texts = dict(zip(paths, read(paths)))
    out: dict[str, list[str]] = {}
    for fam, fs in own.items():
        for stem, rx in stems.items():
            if any(rx.search(texts[f]) for f in fs):
                out.setdefault(stem, []).append(fam)
    return out


# The changed test files a review runs, at most (the sections say when the cap cut the list, and how many ran).
TESTS_CAP = 40

# The tests the PR does not change that import a Python module it changes (tools/importer_tests.py: a failure in an
# unchanged test otherwise shows only in CI's logs): at most IMPORTERS_CAP files, within IMPORTERS_BUDGET_S seconds;
# the section names the files over the cap, and a run cut by the time box reads skip.
IMPORTERS_SECTION, IMPORTERS_CAP, IMPORTERS_BUDGET_S = (
    "Tests that import the modules the PR changes",
    12,
    600,
)
# Develop's tests changed since the develop the PR's CI merged (tools/devtests.py, lessons L-596): CI never ran them
# with the PR. At most DEVTESTS_CAP files, within DEVTESTS_BUDGET_S seconds, on the merge, compared with develop.
DEVTESTS_SECTION, DEVTESTS_CAP, DEVTESTS_BUDGET_S = (
    "Develop's tests changed since CI's merge",
    12,
    600,
)

# The installed-tree section's time box (lessons L-526): the install, the changed tests and, when a failure needs it,
# develop's build and run. Over it, what did not run reads NOT RUN.
INSTALLED_BUDGET_S = 1200

# The touched families the compile check compiles, at most: it holds the heavy slot. The report and the status say
# when the cap cut the list, so a family past the cap is never left uncompiled in silence.
COMPILE_FAMILIES_CAP = 6
# What the compile section says on a machine without ROCm's LLVM tools (paths.rocm_llvm_bin blank).
COMPILE_NO_LLVM = "not run: no ROCm LLVM tools here (paths.rocm_llvm_bin is blank); the compile gate stays pending"

# The clash scan's hard limit (its own budget is config clashes.budget_s; a cold facts cache costs up to 4 minutes),
# and the mutation sweep's time box (its tests' runs; over it, the mutants left read NOT RUN; the hard limit adds the
# copy and the clean-up). The sweep holds the heavy slot, at about 40 reviews a day.
CLASH_TIMEOUT_S = 600
MUTANTS_BUDGET_S = 600
# The sweep's survivors re-run against every importer, under CI's python, torch's (paths.torch_python) and with the
# merged tree's engine build: seconds of their own.
MUTANTS_RERUN_S = 300
MUTANTS_SECTION = "Mutation sweep of the PR's checks"
SINCE_SECTION = "Since our last review"  # a follow-up's range-diff and own-file delta (tools/since_review.py)
# Each PR's sweeps, round by round (pr<N>.jsonl), under paths.cache.
MUTANTS_HISTORY = f"$CA/{paths.CACHE_MUTANTS}"
# Develop's results per develop commit, under paths.cache: its golden gate's failing lines and its installed tests'
# outcomes. Only the monitor writes them, from develop's own runs; a PR's job reads them.
DEVELOP_CACHE = f"$CA/{paths.CACHE_DEVELOP}"

# The PR's head against its merge base, merge or no merge: its CTest entries' labels and tiers
# in the build and installed trees (a configure of each, no build), references to the files it moves or deletes, and
# its Python imported with only the platform engine built. Plain scripts: each a time box in seconds; the hard limit
# adds the clean-up.
LABELS_SECTION, LABELS_BUDGET_S = "CTest labels and tiers", 120
# The changed tests run as CI's test environment runs them (tools/ci_condition.py): a lead section, time-boxed.
CICOND_SECTION, CICOND_BUDGET_S = (
    "CI-condition run (no torch, no built engines, ROCm tools hidden)",
    600,
)
# The Windows lane simulated on Linux (tools/winsim.py): the changed tests run plainly and under the Windows
# simulation, and the C/C++ the PR adds scanned for windows.h's macro names. A lead section, time-boxed.
WINSIM_SECTION = "Windows simulation (cp1252 text, CRLF writes, temp-file reopen; windows.h macro names)"
MOVED_REFS_SECTION, MOVED_REFS_BUDGET_S = "References to moved or deleted files", 120
IMPORTS_SECTION, IMPORTS_BUDGET_S = "Engine import probe", 240
# Doc lines that name what the PR removes or renames, and new env vars, ops and dtypes the reference docs lack
# (tools/doc_refs.py; the agents grepped changed docs against the code by hand in every review).
DOCREFS_SECTION, DOCREFS_BUDGET_S = "Docs against code", 120
# The ABI scan (tools/abi_scan.py): abi.h's counters, struct layouts and exported functions in include/rocke/, and
# abi.h against portable_ir/src/abi.py. Its notes are a check flag (status "abi") and what G9 waits on.
ABI_SECTION = "ABI (abi.h counters and abi.py mirrors, C struct layouts and exported functions in include/rocke/)"
# The checks from peer reviewers' catches (rkmon_lib/peer_checks.py, PEER_CHECKS): each its own section and status line,
# a flag but for the leads (LEAD_SECTIONS); each time-boxed.
PEER_BUDGET_S = 180
# A review's native build of the merged tree (build_native.py: core with the native tests, and the rocke_engine pybind
# module) is the one head build of the review: the verifier and the code quality reviewer reuse it read-only and build
# only a mutant or another tree they need (both agents built in many rounds). The checks name it in their native build
# section and in SHARED_BUILD (the build folder, its tree and the tree's commit, one to a line, so a path with a space
# stays whole: slots.shared_build_lines reads it), which `rk dispatch-pack` checks is still that tree's. It is in the
# checks' slot's build folder ($SB), and the checks keep that slot for the review at their end (`prkeep`).
SHARED_BUILD = "shared_build"
SHARED_BUILD_LINES = (
    f'printf \'%s\\n\' "$SB/${{WTN}}" "$W" "$(git -C "$W" rev-parse HEAD)" > "$OUT/{SHARED_BUILD}"; '
    'echo "shared build: $SB/${WTN} (core: librocke_core and the native tests; pybind: rocke_engine) of the tree '
    '$W at $(git -C "$W" rev-parse --short=10 HEAD): reuse it read-only"; '
)
# The native comparisons' switches for the shared build's binaries (tools/native_env.py), exported before a section's
# pytest: the build is the checks' slot's runner's ($SLOT; on a host, with no slots, it is blank: the monitor's own).
NATIVE_ENV = 'eval "$(python3 -I -B $M/tools/native_env.py $P/tests/run_all.py $SB/${WTN}/core ${SLOT:-} 2>/dev/null)"'


def shared_build_record(cfg: dict, n: int) -> list[str]:
    """PR n's SHARED_BUILD record in its checks' folder, as its lines (slots.shared_build_lines): the build folder, its
    tree and the tree's commit; [] when there is none or it cannot be read. Whether the build may still be reused is
    slots.kept_build's to say.
    """
    try:
        workers = paths.get(cfg, "workers")
    except paths.PathsError:
        return []
    return slots.shared_build_lines(
        pathlib.Path(workers, f"pr{n}", "checks", SHARED_BUILD)
    )


def peer_args(cfg: dict, check: str) -> str:
    """A peer check's own arguments in the checks script: the staged title, the PR's threads (GitHub reads), and the
    tracker-like prefixes that never flag (peer_checks.tracker_allow). The threads, the title and the
    pull-request references also write their items (peer_carry.ITEMS), which a later merge-only head's flags are
    compared with."""
    if check == "title":
        return f" --meta $WK/pr$N/pr/meta.md --items-json \"$OUT/{peer_carry.ITEMS['title']}\""
    if check == "threads":
        return f" --slug {cfg['repo']} --pr $N --items-json \"$OUT/{peer_carry.ITEMS['threads']}\""
    if check == "prnumbers":
        return f" --items-json \"$OUT/{peer_carry.ITEMS['prnumbers']}\""
    if check == "trackers":
        return " --allow " + shlex.quote(",".join(tracker_allow(cfg)))
    return ""


# Contract tests that guard a file, run whenever a PR changes that file (a CMake include added to
# platform/tests/CMakeLists.txt can break the registration contract test on every CI shard, and a run of the PR's
# changed tests alone never runs it).
COMPANION_TESTS = [
    (
        re.compile(r"^platform/tests/(CMakeLists\.txt$|cmake/)"),
        "platform/tests/tools/test_cmake_test_registration.py",
    ),
    # The gate-verdict census pins the gfx1250_lowering parity config count and order; new configs
    # break it, and no CI lane collects that directory.
    (
        re.compile(
            r"^platform/tests/instances/(parity/gfx1250_lowering_emit\.(py|c)"
            r"|differential/(run_diff|ir_artifact_diff)\.py)$"
        ),
        "platform/tests/instances/differential/test_gate_verdict.py",
    ),
    # platform/CMakeLists.txt writes the installed CTest entries and their scoped ENVIRONMENT; its
    # contract test configures a build and reads them (about 15 s).
    (
        re.compile(r"^platform/CMakeLists\.txt$"),
        "platform/tests/test_engine_linkage.py",
    ),
    # Two develop tests read platform/CMakeLists.txt's text (the staged arch-domain tools'
    # install(FILES …) block, the run-checks install guard); a restyle can break both, and no CI lane
    # runs either.
    (
        re.compile(r"^platform/CMakeLists\.txt$"),
        "platform/tests/tools/test_arch_domain_artifact.py",
    ),
    (
        re.compile(r"^platform/CMakeLists\.txt$"),
        "platform/tests/tools/test_run_checks.py",
    ),
    # A decl key with no arch-domain cell fails the domain test, which no CI lane runs, at the
    # checks' own flavor; a run of the PR's changed tests alone never runs it.
    (
        re.compile(
            r"^platform/(python/rocke/core/(arch/|lower_llvm\.py$)|cpp/core/lower_llvm/data\.cpp$)"
        ),
        "platform/tests/tools/test_arch_domain_artifact.py",
    ),
    (
        re.compile(r"^platform/python/rocke/core/arch/data/arch_specs\.json$"),
        "platform/tests/core/test_arch_target_identity.py",
    ),
    # Lessons L-538: a change to the conv dispatch policy or its instance lists can move a case off
    # its path; CI's rocke_conv_reference_unit_pytest runs the reference contract test, so the checks
    # run it too.
    (
        re.compile(
            r"^library/dispatch/(grouped_convolution\.py$|grouped_convolution_instances/"
            r"|conv_winograd_plan\.py$)"
        ),
        "library/tests/test_conv_reference_contract.py",
    ),
    # L-538: GDN decode is Python-only, so byte identity runs none of it; its C++/Python IR parity
    # test is in no CI lane and needs the engine, which the checks build (<slot build>/<wt>/pybind). The
    # cases come from the golden test's list, the golden, the GDN dispatch and the decode kernel.
    (
        re.compile(
            r"^library/(dispatch/gdn/|kernels/gfx950/gdn_decode\.py$"
            r"|tests/(test_gdn_decode_golden\.py|golden/gdn_decode_gfx950_ir_sha256\.json)$)"
        ),
        "library/tests/test_gdn_decode_ir_cpp_parity.py",
    ),
]
# Not companions (L-538): the HIP source identity gate for lower_hip.py, lower_hip/*.cpp and the parity emitters
# (platform/tests/instances/differential/test_hip_source_identity.py) is not on develop; the IngestorGenerator suite
# lives outside the rocKE tree and needs venv-gen (L-327), which this test section lacks.


# The test runs get the tools CI's test lanes have (a test that needs cmake or PyYAML would skip and still read as a
# pass): `paths.cmake`, a
# folder holding cmake and ctest, goes first on PATH; `paths.pyyaml`, PyYAML's `yaml` package folder (or the
# site-packages folder that holds it), is linked into a folder of its own, exported as RKMON_PYEXTRA, which
# tools/pytest_vs_base.py and tools/installed_tests.py append to the tests' PYTHONPATH (only that package: a whole
# site-packages there would shadow the test venv's own). A blank key leaves the tool to PATH and the venv; a tool still
# missing makes its tests read NOT RUN: <tool> missing (tools/pytest_plugins/rkmon_skips.py).


def sparse_extra(cfg: dict, files: list[str]) -> list[str]:
    """The paths outside rocKE's provider (the top folder of config rocke_path) a review's worktrees check out: the
    paths outside it the tests read (config test_sparse, so the tests run as CI's checkout of the whole repository runs
    them), then the top two levels of the PR's files elsewhere (6 at most).
    """
    top = (
        paths.repository_folder(cfg).split("/", 1)[0] + "/"
    )  # the worktrees check out rocKE's provider; other projects are extra
    extra = list(cfg.get("test_sparse") or ())
    own = sorted({"/".join(f.split("/")[:2]) for f in files if not f.startswith(top)})
    return extra + [x for x in own if x not in extra][:6]


def tool_env_lines(cfg: dict, shim: str) -> list[str]:
    """Bash lines that give a test run cmake and PyYAML from the configured paths (the comment above sparse_extra);
    `shim` is the folder that holds the PyYAML link, a shell word the lines put in double quotes ("$SI/pyshim").
    """
    p = cfg.get("paths") or {}
    cm, yp = (
        os.path.expanduser(str(p.get(k) or "").strip()) for k in ("cmake", "pyyaml")
    )
    lines = [f"export PATH={shlex.quote(cm)}:$PATH"] if cm else []
    if yp:
        lines.append(
            f'Y={shlex.quote(yp)}; [ -d "$Y/yaml" ] && Y=$Y/yaml; mkdir -p "{shim}" && ln -sfn "$Y" '
            f'"{shim}/yaml" && export RKMON_PYEXTRA="{shim}"'
        )
    return lines


# A section's result line from a tool's last line "<TAG> pass|fail|skip <detail>": "result: PASS <detail>" (a pass with
# skips says "(k skipped: ...)"), "result: FAIL <detail>", "result: NOT RUN: <why>" or "result: SKIP <detail>".
RESULT_FN = (
    'res() { echo "$1" | awk \'{v = $2; d = $0; sub(/^[^ ]+ [^ ]+ ?/, "", d); print "result: " '
    '(v == "pass" ? "PASS " : v == "fail" ? "FAIL " : d ~ /^NOT RUN/ ? "" : "SKIP ") d}\'; }'
)
# Whole UTF-8 characters only: GNU `cut -c` counts bytes, so a cut can end inside a character ("…", "—"), while
# uutils' `cut -c` (the coreutils of Ubuntu 26.04, the image's base) counts characters. u8 drops a cut character;
# `cutc N` keeps at most N bytes with either cut (`cut -b`) and never leaves a cut character; st writes its detail
# through u8, so a status line is always valid UTF-8 (read_checks_status reads the file).
U8_FN = (
    "u8() { if command -v iconv >/dev/null; then iconv -f UTF-8 -t UTF-8 -c 2>/dev/null; else cat; fi; }; "
    "cutc() { cut -b1-$1 | u8; }"
)
ST_FN = 'st() { echo "$1 $2 $(printf \'%s\' "$3" | u8)" >> "$OUT/status"; }'
# A section that reads GitHub and fails on a network failure (DNS, a refused or reset connection, a timeout) runs once
# more after RKMON_NET_RETRY_S seconds (NET_RETRY_S), instead of reading NOT RUN: `net CMD ...` runs CMD, and when it
# fails (its exit status) and its output or errors name such a failure, again. A run that succeeds is never run again,
# whatever its output or errors say (sudo's warning that it cannot resolve the host's own name, for one). Its output
# is the last run's; its errors go to stderr as before.
NET_RETRY_S = 60
NET_ERRORS = (
    "Could not resolve host|Temporary failure in name resolution|Name or service not known|Network is unreachable|"
    "error connecting to|Connection refused|Connection reset|connection timed out|TLS handshake timeout|i/o timeout"
)
NET_FN = (
    'net() { local e o r i; e=$(mktemp); for i in 1 2; do o=$("$@" 2>"$e"); r=$?; [ "$r" != 0 ] && '
    f"{{ printf '%s\\n' \"$o\"; cat \"$e\"; }} | grep -qiE '{NET_ERRORS}' || break; "
    '[ "$i" = 1 ] && { echo "(a network error: run once more in ${RKMON_NET_RETRY_S:-'
    f"{NET_RETRY_S}"
    '} s)" >&2; sleep "${RKMON_NET_RETRY_S:-'
    f"{NET_RETRY_S}"
    '}"; }; done; [ -n "$o" ] && printf \'%s\\n\' "$o"; cat "$e" >&2; rm -f "$e"; return $r; }'
)


def plan_checks(
    cfg: dict, rec: dict, files: list[str], families: dict[str, str], source=None
) -> dict:
    """What the checks cover. With `source` (the merge worktree, or `("git", repo, sha)`), a changed shared helper
    also selects the families that use it (`helper_families`)."""
    K = paths.repository_folder(cfg)
    rel = [f[len(K) + 1 :] for f in files if f.startswith(K + "/")]
    fams = touched_families(rel, families)
    engine = any(f.startswith(ENGINE_CORE) for f in rel)
    helpers = helper_families(source, K, rel, families, fams) if source else {}
    via = sorted({f for fs in helpers.values() for f in fs})
    if len(via) > MAX_HELPER_FAMILIES:
        engine = True  # a helper many families share: every family at every flavor
    else:
        fams = sorted(set(fams) | set(via))
    cpp = any(
        f.startswith("platform/cpp/") or f.endswith(("CMakeLists.txt", ".cmake"))
        for f in rel
    )  # CMake builds too
    # Companion contract tests go first, so the cap never cuts them.
    companions = sorted(
        {t for pat, t in COMPANION_TESTS if any(pat.match(f) for f in rel)}
    )
    tests = companions + sorted(
        f for f in rel if re.search(r"(^|/)test_[^/]*\.py$", f) and f not in companions
    )
    # The test files the PR itself adds, changes or deletes, uncapped: the installed-tree section maps them to CI's
    # installed CTest entries and applies its own cap (lessons L-526).
    changed_tests = sorted(f for f in rel if re.search(r"(^|/)test_[^/]*\.py$", f))
    provider = (
        K.rsplit("/", 1)[0] if "/" in K else K
    )  # the hip-kernel-provider tree the worktrees check out
    # Library kernels have no emitter-family mapping, so byte identity never covers them: a library kernel change can
    # break the C++ engine unseen. Their parity test runs with this tree's engine while `libparity_check` is on (turn it
    # on only while the test passes on develop: else it fails for every PR).
    libparity = bool(cfg.get("libparity_check")) and any(
        f.startswith("library/kernels/") for f in rel
    )
    # A conftest.py changes how the tests beside it run, so the checks run some of them (a PR that changes only a
    # conftest.py has no test file of its own to run).
    conftests = sorted(f for f in rel if re.search(r"(^|/)conftest\.py$", f))[:3]
    # The Python modules it changes (not tests or conftests): the tests that import them run too (IMPORTERS_SECTION).
    modules = sorted(
        f
        for f in rel
        if f.endswith(".py")
        and f.startswith(("platform/python/", "library/"))
        and not re.search(r"(^|/)(test_[^/]*|conftest)\.py$", f)
    )
    return {
        "families": fams,
        "engine_wide": engine,
        "build": bool(cpp or fams or engine or libparity),
        "golden": bool(fams or engine or any("golden" in f for f in rel)),
        "tests": tests[:TESTS_CAP],
        "tests_total": len(tests),
        "changed_tests": changed_tests,
        "libparity": libparity,
        "conftests": conftests,
        "helpers": helpers,
        "modules": modules,
        "outside_rocke": sorted(
            {
                "/".join(f.split("/")[:2])
                for f in files
                if not f.startswith(provider + "/")
            }
        )[:10],
    }


# Clang's diagnostics the GCC build misses (tools/tidy_check.py): leads, since the local clang-tidy may be newer than
# CI's.
TIDY_SECTION = "Superbuild tidy (clang-tidy with the provider .clang-tidy, clang -Werror; the touched TUs)"
TIDY_BUDGET_S = 900


def status_lines(
    run: str, report: str, tag: str, key: str, shown: int, what: str, fetch: str = ""
) -> list[str]:
    """A section body for a tool that prints its report, then one last line "<TAG> pass|fail|skip <detail>": the run
    (its output in $OUT/<report>), the report without that line, then the line and its status entry. A tool that wrote
    no such line (it timed out or failed) reads skip, never clean. A run that fails on a network failure (the blobs the
    clone fetches from GitHub) runs once more (NET_FN). `fetch`: the tool's command when the run is PR code in a slot,
    where git fetches nothing: its `--fetch` runs first, as the monitor (prcode's `prf`), and what it prints opens the
    report."""
    first = f'net {run} > "$OUT/{report}" 2>&1'
    if fetch:
        first = (
            f'net prf {fetch} > "$OUT/{report}" 2>&1; net {run} >> "$OUT/{report}" 2>&1'
        )
    return [
        first,
        f"grep -v '^{tag} ' \"$OUT/{report}\" | head -{shown}",
        f'x=$(grep \'^{tag} \' "$OUT/{report}" | tail -1); [ -n "$x" ] || x=\'{tag} skip no result (the {what} '
        f'timed out or failed)\'; echo "$x"; st {key} $(echo "$x" | cut -d\' \' -f2) "$(echo "$x" | cut -d\' \' '
        '-f3- | cutc 200)"',
    ]


def checks_script(
    cfg: dict,
    n: int,
    head: str,
    kind: str,
    plan: dict,
    since: str | None,
    mode: str = "review",
    wt: str | None = None,
    out: str | None = None,
    base_sha: str | None = None,
    reviewed: str | None = None,
    merge_only: bool = False,
    base: str | None = None,
) -> str:
    """Bash for a checks job on the merge worktree wt<N>m.

    mode "review": the full report for the verifier, in workers/prN/checks/.
    mode "gate": a merge gate for the queue, without CI, threads or rule_scan, in workers/prN/gate/.
    `reviewed`: a follow-up's last reviewed head (the mutation sweep takes only the checks the push since then adds or
    changes); `merge_only`: a merge-only round (the sweep does not run); `base`: the PR's base branch (`base.ref`;
    develop when none): mkwt.py merges it (RKMON_BASE), and the merge base, the tests at head and the scripted checks
    compare with it. The section names keep "develop", which the readers of checks.md look for.

    Every section is bounded and failure-tolerant, and writes one machine-read line to `status` (`<section>
    pass|fail|skip <detail>`). `develop_sha` and `head_sha` record the exact merge that was checked. queue.py turns the
    status into gates G5 (goldens), G6 (build, engines, tests) and G7 (compile per target).
    """
    fams = ",".join(plan["families"])
    hs = ";".join(
        f"{s}->{'+'.join(fs)}" for s, fs in sorted((plan.get("helpers") or {}).items())
    )
    tests = " ".join(plan["tests"])
    total = plan.get("tests_total") or len(plan["tests"])
    capped = (
        f"echo 'ran {len(plan['tests'])} of {total} changed test files (cap {TESTS_CAP}); the rest are not run here'"
        if total > len(plan["tests"])
        else ":"
    )
    since_s = since or "1970-01-01T00:00:00Z"
    sub = "checks" if mode == "review" else "gate"
    # The mkwt log of the merge this script checks: the review's, or the single-PR gate's own (a gate that read the
    # review's log would print a stale conflict list). A batch step merges in its own script and has none.
    mklog = (
        None if wt else f"$WK/pr$N/{'mkwt' if mode == 'review' else 'gate_mkwt'}.log"
    )
    # The develop this merge worktree was built against: the batch base, or the one mkwt merged (its log).
    mk_dev = base_sha or (
        f"$(grep -oE 'develop=[0-9a-f]+' {mklog} 2>/dev/null | head -1 | cut -d= -f2)"
        if mklog
        else ""
    )
    bref = gitops.base_ref(
        base
    )  # the PR's base branch's ref: develop's, or the feature branch it targets
    dev_ref = (
        base_sha or "$(git -C $W rev-parse $BREF 2>/dev/null)"
    )  # without a merged tree
    wt = wt or f"wt{n}m"
    out = out or f"{paths.get(cfg, 'workers')}/pr{n}/{sub}"
    sparse = " ".join(paths.sparse_dirs(cfg))
    lines = [
        "set -u",
        paths.shell_vars(cfg),
        prcode.SHELL,  # prslot, prc, prr, prin, prkeep, prtree: PR code in a PR runner's slot (rkmon_lib/prcode.py)
        f'K={paths.repository_folder(cfg)}; N={n}; WTN={wt}; W="$D/$WTN"; P=$W/$K/platform; L=$W/$K/library',
        f'OUT={shlex.quote(out)}; rm -rf "$OUT"; mkdir -p "$OUT"; R="$OUT/checks.md"; : > "$OUT/status"',
        f"CP={DEVELOP_CACHE}",  # develop's golden lines (the report writes them) and installed outcomes (read only)
        f"BREF={shlex.quote(bref)}",
        'export PATH=$VB${LB:+:$LB}:$PATH TMPDIR="$TP" ROCKE_LLVM_FLAVOR=llvm22',
        config.export_line(cfg),
        nogpu.SHELL,
        # The slot every PR-code command of this job runs in, its results copied back: one for the whole job, kept
        # for the review at the end when its build is the review's shared build.
        f'prslot $N || {{ echo "# Checks for #$N: not run: no slot of a PR runner could be taken" > "$R"; exit 1; }}',
        *tool_env_lines(cfg, "$SI/pyshim"),
        'sec() { echo; echo "## $1"; echo; }',
        U8_FN,
        ST_FN,
        NET_FN,
        RESULT_FN,
        "memok() { [ $(free -g | awk 'NR==2{print $7}') -ge %d ]; }"
        % config.min_free_memory_gb(cfg),
        FLAVORS_FN,
        # The develop tree (wtbase, at the exact develop SHA of this merge), set up on first use: the tests tell the
        # PR's failures from develop's, and the compile checks new rejections from old ones.
        f"BS={base_sha or ''}; BW=\"$D/wtbase\"; BR=''",
        # The trees the job's git reads, which git in the slot takes as safe: the merge, the head, develop's, the clone.
        f'export {prcode.TREES_VAR}="$W:$D/wt$N:$BW:$R0"',
        'base_tree() { [ -n "$BR" ] && return 0; BS=${BS:-$(cat "$OUT/develop_sha" 2>/dev/null)}; [ -n "$BS" ] || '
        "return 1; if [ ! -e $BW/.git ]; then prtree $BW && git -C $R0 worktree add -q --no-checkout --detach $BW $BS "
        f">/dev/null 2>&1 && git -C $BW sparse-checkout set --no-cone {sparse} >/dev/null 2>&1; fi; "
        "git -C $BW reset -q --hard 2>/dev/null; git -C $BW clean -fdxq 2>/dev/null; "
        'git -C $BW checkout -q --detach $BS 2>/dev/null && BR="--base-root $BW/$K"; }',
        # The head the worktree holds, read from the tree itself, never assumed (checks generated for the state's
        # older head while mkwt merged the live head would otherwise call our own merge commit the head). Our merge
        # commit holds it as a parent: the first for mkwt's, the second for a batch step's (MKWT_EMAIL, BATCH_EMAIL).
        # Without one, HEAD is the head: mkwt's merge stopped, or it was a no-op. That head may itself be the author's
        # merge of develop, so only our merge commit counts as the merge.
        "H1=$(git -C $W rev-parse -q --verify HEAD^1 2>/dev/null); H2=$(git -C $W rev-parse -q --verify HEAD^2 2>/dev/null)",
        "TH=$(git -C $W rev-parse -q --verify HEAD 2>/dev/null || echo none); TD=''; [ -n \"$H2\" ] && case "
        f'"$(git -C $W log -1 --format=%ae HEAD 2>/dev/null)" in {MKWT_EMAIL}) TH=$H1; TD=$H2 ;; {BATCH_EMAIL}) '
        f"TH=$H2; TD={base_sha or '$H1'} ;; esac",
        # Checks generated for another head say so and check nothing: the merged-tree sections, the tests at head and
        # the probes would read a tree that is not the head under review. The status and head_sha name the head the
        # checks were for, so the failure never sticks to the head the tree holds.
        f'MM=\'\'; if [ "$TH" != "{head}" ]; then MERGED=0; MM="head mismatch: worktree has ${{TH:0:10}}, checks '
        f'expected {head[:10]}"; echo {head} > "$OUT/head_sha"; echo {dev_ref} > "$OUT/develop_sha"; '
        'elif [ -n "$TD" ]; then MERGED=1; echo "$TD" > "$OUT/develop_sha"; echo $TH > "$OUT/head_sha"; '
        # The head already contains develop (the author merged today's develop; 0 behind), so our merge was a no-op and
        # the head is the merge, never "no clean merge". The develop is the one mkwt merged (its log), not today's ref,
        # which may have moved on since.
        f'elif DV=$(git -C $W rev-parse -q --verify {mk_dev} 2>/dev/null); [ -n "$DV" ] && git -C $W merge-base '
        '--is-ancestor $DV HEAD 2>/dev/null; then MERGED=1; echo $DV > "$OUT/develop_sha"; echo $TH > "$OUT/head_sha"; '
        f'else MERGED=0; echo $TH > "$OUT/head_sha"; echo {dev_ref} > "$OUT/develop_sha"; fi',
        f'echo "# Checks for #$N at {head[:10]} ({kind}, {mode}), generated by rkmon $(date -u +%FT%TZ)" > "$R"',
        # The base every comparison below is with: the PR's base branch (`base.ref`) and the commit merged.
        "echo "
        + shlex.quote(f"base: {'develop' if gitops.is_develop(base) else base} @")
        + ' "$(cut -c1-10 "$OUT/develop_sha" 2>/dev/null)" >> "$R"',
        "{",
        "sec 'Merge with develop'",
        *(
            []
            if gitops.is_develop(base)
            else [
                "echo "
                + shlex.quote(
                    f"base branch: {base} (not develop): the merge, the merge base and the comparisons "
                    "below are with it"
                ),
                # A mkwt that does not know RKMON_BASE merged develop: say so, never pass it off as the base's merge.
                *(
                    [
                        f"grep -qF ' base={base}' {mklog} 2>/dev/null || echo '!! the mkwt log names no base={base}: "
                        "worker/mkwt.py merged develop, not the base branch (it needs RKMON_BASE)'"
                    ]
                    if mklog
                    else []
                ),
            ]
        ),
        # "no clean merge" only with a conflicting file to show for it.
        (
            "NC=$(awk '/^conflicting files vs develop:/ {s=1; next} s && !NF {s=0} s {n++} END {print n+0}' "
            f"{mklog} 2>/dev/null)"
            if mklog
            else "NC=''"
        ),
        'if [ $MERGED = 1 ]; then echo "merge $(git -C $W log -1 --format=%h) = head $(cut -c1-10 "$OUT/head_sha") + '
        'develop $(cut -c1-10 "$OUT/develop_sha")"; st merge pass clean; elif [ -n "$MM" ]; then echo "$MM: the '
        'worktree is not the head under review, so nothing below checks it"; st merge fail "$MM"; '
        'elif [ "${NC:-0}" -gt 0 ]; then echo "no clean merge: head $(cut -c1-10 "$OUT/head_sha") does not merge into '
        'develop $(cut -c1-10 "$OUT/develop_sha")"; st merge fail \'no clean merge with develop\'; else echo "no '
        'merged tree: head $(cut -c1-10 "$OUT/head_sha") has no merge with develop '
        '$(cut -c1-10 "$OUT/develop_sha") in $WTN, and no conflicting file is listed"; st merge fail '
        "'no merged tree, and no conflicting file listed'; fi",
        # The whole conflict list, with its count (never cut short).
        (
            (
                'awk \'/^!!/ || $1 == "behind/ahead:" {print} /^conflicting files vs develop:/ {s=1; next} s && !NF {s=0} '
                's {f[++n]=$0} END {print "conflicting files vs develop: " n+0; for (i=1; i<=n; i++) print "  " f[i]}\' '
                f"{mklog} 2>/dev/null"
            )
            if mklog
            else ":"
        ),
        f"echo 'plan: families=[{fams}] engine_wide={plan['engine_wide']} build={plan['build']} "
        f"golden={plan['golden']} tests={len(plan['tests'])} outside_rocke={plan['outside_rocke']}"
        f"{' helpers=' + hs if hs else ''}'",
    ]
    split = len(lines)  # every section from here on needs the merged tree
    # Isolation: every section that reads the native build (byte identity, the native comparisons in the tests) gets a
    # fresh build of this exact tree, in the slot's build folder; build_native.py wipes <build>/<worktree> first, so
    # nothing from an earlier run leaks.
    if plan["build"] or plan["families"] or plan["engine_wide"] or plan["tests"]:
        lines += [
            "sec 'Native build + ctest'",
            # native_ok: this run built this tree (the installed-tree section must never install an earlier build);
            # the job reads it in its inputs.
            f"if memok; then if prc python3 -B {paths.worker_tool('build_native.py')} ${{WTN}} "
            '> "$OUT/build.log" 2>&1; then '
            "echo 'native build OK'; "
            ': > "$OUT/native_ok"; prin "$OUT/native_ok"; '
            + (SHARED_BUILD_LINES if mode == "review" else "")
            + f'prc -C $SB/${{WTN}}/core timeout 1800 ctest -j{paths.build_jobs()} > "$OUT/ctest.log" 2>&1; '
            'r=$(grep -E \'tests passed|tests failed\' "$OUT/ctest.log"); echo "$r"; '
            "grep -E 'Failed|\\*\\*\\*|Skipped' \"$OUT/ctest.log\" | head -8; "
            'if echo "$r" | grep -q \'100% tests passed\'; then st build pass "$r"; else st build fail "$r"; fi; '
            "else echo 'NATIVE BUILD FAILED'; tail -15 \"$OUT/build.log\"; st build fail 'native build failed'; fi; "
            "else echo 'SKIPPED: low memory'; st build skip 'low memory'; fi",
        ]
    if plan["build"]:
        # Over the native build's compile_commands.json: only once this run built this tree. As PR code (the compile
        # commands are what its build wrote), after its --fetch (the provider's .clang-tidy the sparse tree links to).
        tidy = (
            "python3 -I -B $M/tools/tidy_check.py --tree $W --build $SB/${WTN}/core --rocke-path $K --repo $R0 "
            f'--base "$(cat "$OUT/develop_sha" 2>/dev/null)" --dev-tree "$TDV" --llvm-bin "$LB" '
            f"--budget {TIDY_BUDGET_S}"
        )
        lines += (
            [
                f"sec '{TIDY_SECTION}'",
                'if [ -f "$OUT/native_ok" ]; then',
                # develop's copy of each TU (the develop checkout, wtbase): an error develop has too is not counted
                "base_tree; TDV=''; [ -n \"$BR\" ] && TDV=$BW",
            ]
            + status_lines(
                f"prc nice -n 10 timeout {TIDY_BUDGET_S + 120} {tidy}",
                "tidy_report.txt",
                "TIDY",
                "tidy",
                50,
                "tidy check",
                fetch=tidy,
            )
            + [
                "else echo 'not run: no native build of this tree'; st tidy skip 'no native build of this tree'; fi"
            ]
        )
    if plan["families"] or plan["engine_wide"]:
        lines += ["sec 'Byte identity (C++ vs Python)'"]
        if plan["families"]:
            lines += [
                f"bad=''; FLS=$(flavors); for FL in $FLS; do ROCKE_LLVM_FLAVOR=$FL prc -C $P timeout 2400 python -B "
                f"tools/check_byte_identity.py --only {fams} --build-root $SB/${{WTN}}/core > "
                '"$OUT/bi_$FL.log" 2>&1; r=$(grep -E \'^RESULT\' "$OUT/bi_$FL.log" | cutc 70); echo "$FL: $r"; '
                # Anything not GREEN, then the touched families by name, then a count: `--only gemm` also
                # matches 20+ gemm families, and a cut list of those would hide the touched ones.
                "grep -E '^ *(RED|YELLOW) ' \"$OUT/bi_$FL.log\" | head -12 | sed 's/^/   /'; "
                f"for f in {' '.join(plan['families'])}; do grep -E \"^ *GREEN +$f +configs=\" \"$OUT/bi_$FL.log\"; "
                "done | sed 's/^/   /'; "
                'echo "   ($(grep -cE \'^ *GREEN +[A-Za-z0-9_]+ +configs=\' "$OUT/bi_$FL.log") families GREEN '
                'in all)"; echo "$r" | grep -q GREEN || bad="$bad $FL"; done; '
                '[ -z "$bad" ] && st byteid pass "touched families at $FLS" || st byteid fail "not GREEN at$bad"'
            ]
        if plan["engine_wide"]:
            # Every family at every emission column (one run at the host's flavor would leave the other flavors to be
            # run by hand, about a minute each).
            lines += [
                "bad=''; FLS=$(flavors); for FL in $FLS; do ROCKE_LLVM_FLAVOR=$FL prc -C $P timeout 3600 python -B "
                'tools/check_byte_identity.py --build-root $SB/${WTN}/core > "$OUT/bi_all_$FL.log" 2>&1; '
                'r=$(grep -E \'^RESULT\' "$OUT/bi_all_$FL.log" | cutc 70); echo "$FL, every family: $r"; '
                "grep -E '^ *(RED|YELLOW)' \"$OUT/bi_all_$FL.log\" | head -12 | sed 's/^/   /'; "
                'echo "$r" | grep -q GREEN || bad="$bad $FL"; done; '
                '[ -z "$bad" ] && st byteid_all pass "every family at $FLS" || st byteid_all fail "not GREEN at$bad"'
            ]
    if plan.get("libparity"):
        # A skip counts as a failure: without this tree's engine the test proves nothing (CI skips it the same way).
        lines += [
            "sec 'C++/Python parity of the library kernels (test_attention_ir_cpp_parity.py, this tree engine)'",
            "bad=''; FLS=$(flavors); for FL in $FLS; do ROCKE_LLVM_FLAVOR=$FL ROCKE_CPP_STRICT=1 "
            "PYTHONPATH=$P/python:$L:$SB/${WTN}/pybind prc -C $W/$K timeout 1500 python -B -m pytest "
            "library/tests/test_attention_ir_cpp_parity.py -q -rs -p no:cacheprovider --no-header -o addopts= "
            "> \"$OUT/libparity_$FL.log\" 2>&1; r=$(grep -E '[0-9]+ (passed|failed|error|skipped)' "
            '"$OUT/libparity_$FL.log" | tail -1); echo "$FL: $r"; if echo "$r" | grep -qE \'failed|error|skipped\' '
            '|| ! echo "$r" | grep -q passed; then bad="$bad $FL"; grep -E \'^(FAILED|ERROR|SKIPPED)\' '
            "\"$OUT/libparity_$FL.log\" | head -4 | sed 's/^/   /'; fi; done; "
            '[ -z "$bad" ] && st libparity pass "at $FLS" || st libparity fail "not passing at$bad"',
        ]
    if plan["golden"]:
        # One run, as CI's rocke_golden_static: the gate checks every flavor the golden holds. Its failing lines are
        # sorted into the golden's own load errors and drift by family and flavor, and compared with develop's, whose
        # lines are cached per develop SHA: develop's gate runs only when the cache lacks them, in a slot of its own.
        # The report runs as the monitor, reads both gates' results as data and alone writes the cache
        # (tools/golden_vs_base.py; the first "gate:" line alone can hide a load error, L-523).
        lines += [
            "sec 'Representative golden gate (one run, every flavor, as CI; compared with develop)'",
            'base_tree; GB=-; [ -n "$BR" ] && GB=$BW/$K/platform/tests; DS=$(cut -c1-12 "$OUT/develop_sha" '
            "2>/dev/null); GC=''; [ -n \"$DS\" ] && mkdir -p $CP && GC=$CP/golden_develop_$DS.lines",
            'prc python3 -I -B $M/tools/golden_vs_base.py gate $SO/golden $P/tests > "$OUT/golden_report.txt" 2>&1; '
            'g=$?; prr golden "$OUT/golden"; if [ $g = 1 ]; then GDO=\'\'; if [ "$GB" != - ] && [ ! -s "$GC" ]; then '
            'GDO="$OUT/golden_develop"; (prslot || exit 3; prc python3 -I -B $M/tools/golden_vs_base.py gate '
            "$SO/golden $GB > /dev/null 2>&1; prr golden $GDO; prfree); fi; "
            'python3 -I -B $M/tools/golden_vs_base.py report "$OUT/golden" $P/tests $GB "$GC" $GDO '
            '>> "$OUT/golden_report.txt" 2>&1; fi; '
            "grep -v '^GOLDEN ' \"$OUT/golden_report.txt\" | head -30; "
            "g=$(grep '^GOLDEN ' \"$OUT/golden_report.txt\" | tail -1); [ -n \"$g\" ] || g='GOLDEN fail no result'; "
            'st golden $(echo "$g" | cut -d\' \' -f2) "$(echo "$g" | cut -d\' \' -f3- | cutc 200)"',
        ]
    branch = "develop" if gitops.is_develop(base) else base
    lines += [
        "sec 'Differential golden (the CI CTest commands)'",
        # The flags the CI commands pass, probed on this tree's run_diff.py first: a PR on an older base (a feature
        # branch) may have a run_diff.py without them, and a run refused for a flag is no golden verdict.
        "DGH=$(prc -C $P/tests/instances/differential timeout 120 python -B run_diff.py --help 2>&1); "
        f"DGM=''; for f in {' '.join(DG_FLAGS)}; do printf '%s' \"$DGH\" | grep -q -- \"$f\" || DGM=\"$DGM $f\"; done",
        'if [ -n "$DGM" ]; then echo "NOT RUN: this tree\'s run_diff.py --help names no$DGM (base '
        + shlex.quote(branch)
        + '): the differential golden is not checked here"; st dg skip "run_diff.py has no$DGM"; else',
        f"({DG_FLAVORS}; DGD=$P/tests/instances/differential; prc -C $DGD timeout 900 python -B run_diff.py --mode ll "
        '--python-only --check-golden --flavor $DGF > "$OUT/dg_ll.log" 2>&1; a=$?; echo "ll ($DGF): rc=$a $(python3 '
        '-I -B $M/tools/dg_summary.py "$OUT/dg_ll.log" | cutc 400)"; prc -C $DGD timeout 900 '
        'python -B run_diff.py --mode ir --canonical --python-only --check-golden > "$OUT/dg_can.log" 2>&1; b=$?; '
        'echo "canonical: rc=$b $(python3 -I -B $M/tools/dg_summary.py "$OUT/dg_can.log" | cutc 400)"; '
        "if [ $a -eq 0 ] && [ $b -eq 0 ]; then st dg pass 'GOLDEN OK'; else st dg fail "
        '"re-bless needed (ll rc=$a, canonical rc=$b)"; fi)',
        "fi",
    ]
    conftests = " ".join(plan.get("conftests") or [])
    if plan["tests"] or conftests:
        lines += [
            "sec 'Tests the PR adds or changes (CPU; GPU-only cases skip)'",
            NATIVE_ENV,
            "env | grep -oE '^ROCKE_[A-Z_]*(EMIT|TEST|CLI|EXE)=' | tr -d = | tr '\\n' ' ' | "
            "sed 's/^/native comparisons enabled: /'; echo",
            # The tests a changed conftest.py governs: the files beside it that it names by a test_ prefix,
            # else the 4 smallest there; 8 at most. A sample by name: the files a conftest names may not be the
            # ones its change acts on, so their passes may prove nothing about it, and the report says so.
            f"CT=''; for c in {conftests}; do d=$(dirname $c); sel=''; for p in $(grep -oE 'test_[A-Za-z0-9_]+' "
            '$W/$K/$c | sort -u); do sel="$sel $(cd $W/$K && ls $d/$p*.py 2>/dev/null | head -4)"; done; '
            '[ -n "$(echo $sel)" ] || sel=$(cd $W/$K && ls -Sr $d/test_*.py 2>/dev/null | head -4); '
            "CT=\"$CT $sel\"; done; CT=$(echo $CT | tr ' ' '\\n' | grep . | sort -u | head -8 | tr '\\n' ' ')",
            '[ -z "$CT" ] || echo "tests the changed conftest.py governs, a sample by name (the first files of '
            "the test_ prefixes it names, else the smallest); they may not exercise its change, so check which "
            'files it applies to: $CT"',
            # Every file runs (no -x), and each failure is retried on develop (tools/pytest_vs_base.py).
            'base_tree; TB=-; [ -n "$BR" ] && TB=$BW/$K',
            capped,
            # A test file the PR deletes is in its changed files too; pytest stops on a missing path and runs
            # nothing, so only files that exist run.
            f"TS=''; for t in {tests}; do if [ -e $W/$K/$t ]; then TS=\"$TS $t\"; "
            'else echo "not run (the PR deletes it): $t"; fi; done',
            "if [ -z \"$(echo $TS $CT)\" ]; then echo 'no test to run'; st tests skip 'no test to run'; else "
            f'{prcode.RESULTS_VAR}="$OUT/tests" '
            "prc python3 -I -B $M/tools/pytest_vs_base.py $SO/tests $W/$K $TB $P/python:$L:$SB/${WTN}/pybind "
            '$BW/$K/platform/python:$BW/$K/library $TS $CT > "$OUT/tests_report.txt" 2>&1; prr tests "$OUT/tests"; '
            # Up to 150 lines: the skipped tests with their reasons and a re-run command per file come last.
            "grep -v '^TESTS ' \"$OUT/tests_report.txt\" | head -150; "
            "t=$(grep '^TESTS ' \"$OUT/tests_report.txt\" | tail -1); [ -n \"$t\" ] || t='TESTS fail no result'; "
            'res "$t"; '
            # Only the conftest sample ran: the status says so, since its pass count may prove nothing.
            "CS=''; [ -z \"$(echo $TS)\" ] && CS='conftest sample by name, may not exercise the change: '; "
            'st tests $(echo "$t" | cut -d\' \' -f2) "$CS$(echo "$t" | cut -d\' \' -f3- | cutc 160)"; fi',
        ]
    if mode == "review" and plan["tests"]:
        # The same files once more as CI's test environment runs them (tools/ci_condition.py): no torch or built
        # engines (ci_absent_modules), ROCm's tools hidden. "0 of n run" is a lead: CI runs nothing of the file.
        absent = shlex.quote(",".join(lanes.absent_modules(cfg)))
        lines += [f"sec '{CICOND_SECTION}'"] + status_lines(
            f"prc nice -n 10 timeout {CICOND_BUDGET_S + 120} python3 -I -B $M/tools/ci_condition.py --tree $W/$K "
            f"--python python --absent {absent} --budget {CICOND_BUDGET_S} {tests}",
            "cicond_report.txt",
            "CICOND",
            "cicond",
            60,
            "run",
        )
    if mode == "review" and (plan["tests"] or plan["build"]):
        # The Windows lane on Linux (tools/winsim.py): the macro scan reads git only (this user), and the job reads its
        # report from the slot's inputs; the test runs are PR code (prc). A test that passes plainly and fails
        # simulated, and a windows.h macro name the PR adds, are leads.
        budget = lanes.WINSIM_BUDGET_S
        lines += [
            f"sec '{WINSIM_SECTION}'",
            f"timeout 600 python3 -I -B $M/tools/winsim.py scan --repo $R0 --head {head} --base $BREF --rocke-path $K "
            '--tree $W/$K > "$OUT/winsim_scan.txt" 2>&1; prin "$OUT/winsim_scan.txt"',
        ] + status_lines(
            f"prc nice -n 10 timeout {budget + 120} python3 -I -B $M/tools/winsim.py run --tree $W/$K --python python "
            f"--scan-report $SI/winsim_scan.txt --budget {budget} -- {tests}",
            "winsim_report.txt",
            "WINSIM",
            "winsim",
            60,
            "run",
        )
    modules = plan.get("modules") or []
    if modules:
        # The tests the PR does not change that import a module it changes, on the merged tree, each failure retried
        # on the base (tools/pytest_vs_base.py), within a time box: a lead, as the other scripted sections.
        own = " ".join(
            sorted(set(plan.get("changed_tests") or []) | set(plan["tests"]))
        )
        lines += [
            f"sec '{IMPORTERS_SECTION} (CPU; tests the PR does not change)'",
            NATIVE_ENV,
            'base_tree; TB=-; [ -n "$BR" ] && TB=$BW/$K; mkdir -p "$OUT/importers"',
            f'python3 -I -B $M/tools/importer_tests.py --tree $W/$K --list "$OUT/importers/files.txt" '
            f"--map \"$OUT/importers/map.json\" --cap {IMPORTERS_CAP} --exclude {own} -- {' '.join(modules)} 2>&1 | "
            "head -40",
            'IS=$(cat "$OUT/importers/files.txt" 2>/dev/null)',
            "if [ -z \"$(echo $IS)\" ]; then echo 'no test to run'; st importers skip 'no test to run'; else "
            f'{prcode.RESULTS_VAR}="$OUT/importers" prc timeout {IMPORTERS_BUDGET_S} python3 -I -B '
            "$M/tools/pytest_vs_base.py $SO/importers $W/$K $TB "
            "$P/python:$L:$SB/${WTN}/pybind $BW/$K/platform/python:$BW/$K/library $IS > "
            '"$OUT/importers/report.txt" 2>&1; prr importers "$OUT/importers"; '
            "grep -v '^TESTS ' \"$OUT/importers/report.txt\" | head -100; "
            # One IMPORTERS_FAIL line per importer failing on the merge only, with the changed modules it imports.
            'python3 -I -B $M/tools/importer_tests.py --fails "$OUT/importers/tests_pr_failed.txt" '
            '--map "$OUT/importers/map.json"; '
            't=$(grep \'^TESTS \' "$OUT/importers/report.txt" | tail -1); [ -n "$t" ] || '
            f"t='TESTS skip NOT RUN: the run passed its {IMPORTERS_BUDGET_S} s time box'; res \"$t\"; "
            'st importers $(echo "$t" | cut -d\' \' -f2) "$(echo "$t" | cut -d\' \' -f3- | cutc 200)"; fi',
        ]
    # Develop's tests CI never ran with the PR (lessons L-596): the test files develop added or changed after the
    # develop CI merged (the PR's base.sha in pr/meta.md, else the merge base), run on the merge, each failure retried
    # on develop; DEVTESTS_FAIL names a file that fails on the merge only, with the develop commit that changed it.
    own_all = " ".join(
        sorted(set(plan.get("changed_tests") or []) | set(plan["tests"]))
    )
    lines += [
        f"sec '{DEVTESTS_SECTION} (CPU; on the merge, compared with develop)'",
        NATIVE_ENV,
        'base_tree; TB=-; [ -n "$BR" ] && TB=$BW/$K; mkdir -p "$OUT/devtests"',
        "CIB=$(grep -oE 'base: [^ ]+ @ [0-9a-f]+' $WK/pr$N/pr/meta.md 2>/dev/null | head -1 | awk '{print $4}'); "
        'CIW="the PR\'s base commit (base.sha)"; if [ -z "$CIB" ]; then CIB=$(git -C $W merge-base '
        '$(cat "$OUT/head_sha") $(cat "$OUT/develop_sha") 2>/dev/null); '
        "CIW='the merge base (no base.sha staged)'; fi; "
        'echo "CI\'s develop: $CIW ${CIB:0:10}"',
        f'python3 -I -B $M/tools/devtests.py --git $W --rocke $K --tree $W/$K --since "$CIB" '
        '--develop $(cat "$OUT/develop_sha") --list "$OUT/devtests/files.txt" --map "$OUT/devtests/map.json" '
        f'--cap {DEVTESTS_CAP} --exclude-list "$OUT/importers/files.txt"'
        + (f" --exclude {own_all}" if own_all else "")
        + " 2>&1 | head -30",
        'DT=$(cat "$OUT/devtests/files.txt" 2>/dev/null)',
        "if [ -z \"$(echo $DT)\" ]; then echo 'no test to run'; st devtests skip 'no test to run'; else "
        f'{prcode.RESULTS_VAR}="$OUT/devtests" prc timeout {DEVTESTS_BUDGET_S} python3 -I -B '
        "$M/tools/pytest_vs_base.py $SO/devtests $W/$K $TB "
        "$P/python:$L:$SB/${WTN}/pybind $BW/$K/platform/python:$BW/$K/library $DT > "
        '"$OUT/devtests/report.txt" 2>&1; prr devtests "$OUT/devtests"; '
        "grep -v '^TESTS ' \"$OUT/devtests/report.txt\" | head -100; "
        'python3 -I -B $M/tools/devtests.py --fails "$OUT/devtests/tests_pr_failed.txt" '
        '--map "$OUT/devtests/map.json"; '
        't=$(grep \'^TESTS \' "$OUT/devtests/report.txt" | tail -1); [ -n "$t" ] || '
        f"t='TESTS skip NOT RUN: the run passed its {DEVTESTS_BUDGET_S} s time box'; res \"$t\"; "
        'st devtests $(echo "$t" | cut -d\' \' -f2) "$(echo "$t" | cut -d\' \' -f3- | cutc 200)"; fi',
    ]
    changed = plan.get("changed_tests") or []
    if changed:
        # CI's rocKE shards run the INSTALLED tree (lessons L-526, L-142: a test that reads files through
        # Path(__file__).parents[k] can pass in the source layout and fail installed). The slot's build is installed
        # into its build folder (<wt>/inst); each changed test file runs under the installed CTest entries that collect
        # it, with their commands and ENVIRONMENT. Failures are compared with develop's outcomes at the same develop
        # SHA in the cache, which PR code reads and never writes. The ids it lacks run in a develop job of their own
        # (a slot of its own: develop built and installed in its build folder); the monitor adds that job's finished
        # outcomes to the cache, and the run is made again with them (tools/installed_tests.py, time-boxed; the time
        # is on the status line).
        # The hard limit leaves room for worker_grace.py's wait (at most 16 min) before the time box starts.
        tool = f"nice -n 10 timeout {INSTALLED_BUDGET_S + 1200} python3 -I -B $M/tools/installed_tests.py"
        grace = "--grace $M/tools/worker_grace.py"
        dev_build = f"python3 -B {paths.worker_tool('build_native.py')} wtbase"
        lines += [
            'sec "Changed tests in the installed tree (as CI runs them)"',
            'base_tree; IT=-; [ -n "$BR" ] && IT=$BW/$K; DS=$(cat "$OUT/develop_sha" 2>/dev/null); IC=\'\'; '
            '[ -n "$DS" ] && mkdir -p $CP && IC=$CP/installed_develop_${DS:0:12}.tsv',
            f'inst() {{ o=$1; shift; {prcode.RESULTS_VAR}="$OUT/installed" prc {tool} --out $SO/$o --tree $W/$K '
            f"--build $SB/${{WTN}}/core "
            f'--prefix $SB/${{WTN}}/inst --build-ok $SI/native_ok {grace} --dev-tree $IT --cache "$IC" "$@" '
            f"--budget {INSTALLED_BUDGET_S} {' '.join(changed)} > \"$OUT/installed_report.txt\" 2>&1; }}",
            'inst installed; prr installed/develop_todo.tsv "$OUT/installed_todo"',
            # The develop job: the ids the cache lacks, in a slot of its own.
            'if [ -n "$IC" ] && [ "$IT" != - ] && [ -s "$OUT/installed_todo/develop_todo.tsv" ]; then '
            f'(prslot || exit 3; prin "$OUT/installed_todo/develop_todo.tsv"; prc {tool} develop --out $SO/develop '
            f'--todo $SI/develop_todo.tsv --dev-tree $IT --dev-sha "$DS" --dev-build "{dev_build}" '
            f"--dev-core $SB/wtbase/core --dev-prefix $SB/wtbase/inst {grace} "
            f"--min-free-gb {config.min_free_memory_gb(cfg):g} --budget {INSTALLED_BUDGET_S} "
            '> "$OUT/installed_develop.txt" 2>&1; prr develop "$OUT/installed_develop"; prfree) || '
            "echo 'develop job: not run (no slot of a PR runner could be taken)'; "
            "tail -1 \"$OUT/installed_develop.txt\" 2>/dev/null | sed 's/^/develop job: /'; "
            'DO="$OUT/installed_develop/develop_outcomes.tsv"; '
            'python3 -I -B $M/tools/installed_tests.py cache-add $DO "$IC" > /dev/null 2>&1; fi',
            'if [ -s "${DO:-}" ]; then prin $DO; inst again --outcomes $SI/develop_outcomes.tsv; '
            'prr again "$OUT/installed"; else prr installed "$OUT/installed"; fi',
            "grep -v '^INSTALLED ' \"$OUT/installed_report.txt\" | head -150",
            'i=$(grep \'^INSTALLED \' "$OUT/installed_report.txt" | tail -1); [ -n "$i" ] || '
            "i='INSTALLED fail no result'; res \"$i\"; "
            'st installed $(echo "$i" | cut -d\' \' -f2) "$(echo "$i" | cut -d\' \' -f3- | cutc 200)"',
        ]
    if plan["families"]:
        todo, rest = (
            plan["families"][:COMPILE_FAMILIES_CAP],
            plan["families"][COMPILE_FAMILIES_CAP:],
        )
        capnote = (
            (
                f"compiled {len(todo)} of {len(plan['families'])} touched families (cap {COMPILE_FAMILIES_CAP}); "
                f"not compiled here: {' '.join(rest)}"
            )
            if rest
            else ""
        )
        lines += [
            "sec 'Compile checks (ROCm clang 24, -O3, each config for its target; rejections vs develop)'",
            # Without ROCm's LLVM tools nothing compiles here: say so, and leave G7 pending, never passed or failed
            # for the PR (gates_from_status): a machine without ROCm runs every other check.
            f"if [ -z \"$LB\" ]; then echo '   {COMPILE_NO_LLVM}'; st compile skip '{COMPILE_NO_LLVM}'; else",
            ': > "$OUT/compile.log"',
            "base_tree || echo 'no develop tree: rejections are not compared with develop'",
            f"CAPNOTE='{capnote}'",
            '[ -z "$CAPNOTE" ] || echo "   $CAPNOTE"',
        ]
        for fam in todo:
            lines += [
                f"prc timeout 1800 python -B $M/tools/p_emit_compile.py --root $W/$K $BR "
                f"{plan['_dirs'][fam]}:{fam} 2>&1 | head -6 | tee -a \"$OUT/compile.log\" | sed 's/^/   /'"
            ]
        # A family with nothing compiled proves nothing: it fails instead of passing (configs of a flavor the host
        # lacks all count as rejected).
        lines += [
            "if grep -qE ' [1-9][0-9]* (compile failures|newly rejected)' \"$OUT/compile.log\"; then st compile fail "
            '"$(grep -E \' [1-9][0-9]* (compile failures|newly rejected)\' "$OUT/compile.log" | head -2 | '
            "tr '\\n' ' ')${CAPNOTE:+; $CAPNOTE}\"; elif grep -qE '^[A-Za-z0-9_]+: 0 compiled' \"$OUT/compile.log\"; "
            'then st compile fail "nothing compiled: $(grep -oE \'^[A-Za-z0-9_]+: 0 compiled\' "$OUT/compile.log" | '
            "head -3 | tr '\\n' ' ')${CAPNOTE:+; $CAPNOTE}\"; else st compile pass \"$(grep -oE 'spills [0-9]+' "
            "\"$OUT/compile.log\" | sort | uniq -c | tr '\\n' ' ')${CAPNOTE:+; $CAPNOTE}\"; fi",
            "fi",
        ]
    # A PR that conflicts with develop has no merged tree: skip the heavy sections instead of building a conflicted
    # tree (which "rejects" every config). Skipped sections leave G5-G7 pending.
    heavy = lines[split:]
    del lines[split:]
    skipped = [
        "else",
        "sec 'Build, goldens, byte identity, tests and compile'",
        'if [ -n "$MM" ]; then echo "SKIPPED: $MM."; elif [ "${NC:-0}" -gt 0 ]; then echo \'SKIPPED: the '
        "PR does not merge cleanly with develop, so there is no merged tree to check. The author rebases; the "
        "next review checks the merge.'; else echo 'SKIPPED: there is no merged tree to check (see Merge with "
        "develop).'; fi",
    ]
    if mode == "review" and plan["tests"]:
        # Still run the PR's own tests at head (the clean wt<N>), against the merge base: the PR's effect without
        # develop's later changes (a head can fail tests that pass at its merge base). `head_tests` feeds no gate: G5-G7
        # stay about the merge. Not on a head
        # mismatch: wt<N> holds the worktree's head, not the one under review.
        skipped += [
            "sec 'Tests the PR adds or changes, at head (no merged tree: compared with the merge base; Python only)'",
            'if [ -n "$MM" ]; then echo "not run: $MM"; else',
            f'WH="$D/wt$N"; MBS=$(git -C $R0 merge-base $BREF {head} 2>/dev/null)',
            capped,
            "if [ ! -d $WH/$K ] || [ -z \"$MBS\" ]; then echo 'no head worktree or merge base: not run'; else "
            "BS=$MBS; BR=''; base_tree; TB=-; [ -n \"$BR\" ] && TB=$BW/$K; "
            f"TS=''; for t in {tests}; do [ -e $WH/$K/$t ] && TS=\"$TS $t\"; done; "
            'if [ -z "$(echo $TS)" ]; then echo \'no test to run\'; else mkdir -p "$OUT/head_tests"; '
            f'{prcode.RESULTS_VAR}="$OUT/head_tests" '
            "prc python3 -I -B $M/tools/pytest_vs_base.py $SO/head_tests $WH/$K $TB "
            "$WH/$K/platform/python:$WH/$K/library $BW/$K/platform/python:$BW/$K/library $TS "
            '> "$OUT/head_tests/report.txt" 2>&1; prr head_tests "$OUT/head_tests"; '
            "sed -e 's/on the merge only/at head only/' -e 's/fail on develop too/fail at the merge base too/' "
            "-e 's/develop skips them/the merge base skips them/' "
            '"$OUT/head_tests/report.txt" > "$OUT/head_tests/r2.txt"; '
            "grep -v '^TESTS ' \"$OUT/head_tests/r2.txt\" | head -150; "
            "t=$(grep '^TESTS ' \"$OUT/head_tests/r2.txt\" | tail -1); [ -n \"$t\" ] || t='TESTS fail no result'; "
            'res "$t (at head, vs the merge base ${MBS:0:10})"; '
            'st head_tests $(echo "$t" | cut -d\' \' -f2) "$(echo "$t" | cut -d\' \' -f3- | cutc 200)"; fi; fi',
            "fi",
        ]
    lines += ['if [ "$MERGED" = 1 ]; then'] + (heavy or [":"]) + skipped + ["fi"]
    if mode == "review":
        # Each check the PR adds or changes, disabled once, against the PR's tests (the sweep the code quality reviewer
        # would otherwise make by hand). At the head, in a scratch copy (never wt<N>), time-boxed; a survivor is a check
        # no test catches (tools/mutation_sweep.py). A follow-up takes only the checks its push adds or changes; a
        # merge-only round changes none of the PR's lines. Every round's mutants go to the PR's sweep history
        # (MUTANTS_HISTORY), and a later round lists the earlier survivors and NOT RUN items still at its head (the code
        # quality reviewer mutated every check by hand on follow-ups whose section read only "no new checks"). The sweep
        # reads the history and writes none: the append step after it adds the mutants, read from the sweep's
        # mutants.json as data, and what the step refuses opens the section.
        if reviewed:
            # What the push since our review really changed, base-aware: each commit the same patch, changed, new or
            # dropped (range-diff), and the PR's own files whose patch differs (the agents ran both by hand).
            lines += [f"sec '{SINCE_SECTION}'"] + status_lines(
                f"timeout 300 python -I -B $M/tools/since_review.py --repo $R0 --old {reviewed} --new {head} "
                "--rocke-path $K --base $BREF",
                "since_report.txt",
                "SINCE",
                "since",
                60,
                "comparison",
            )
        lines += [f'sec "{MUTANTS_SECTION}"']
        hist = f" --history {MUTANTS_HISTORY} --pr $N"
        if merge_only and reviewed:
            lines += status_lines(
                f"timeout 120 python -I -B $M/tools/mutation_sweep.py --repo $R0 --head {head} "
                f'--since {reviewed} --out "$OUT/mutants" --rocke-path $K --base $BREF{hist} '
                "--carry-only",
                "mutants_report.txt",
                "MUTANTS",
                "mutants",
                60,
                "sweep",
            )
        else:
            # PR code in the slot, after its `--fetch` (prf: the blobs of the clone it reads; git in the slot fetches
            # nothing). The labels and the import probe below run so too.
            sweep = (
                f"python -I -B $M/tools/mutation_sweep.py --repo $R0 --head {head}"
                f"{f' --since {reviewed}' if reviewed else ''} --out "
                f'$SO/mutants --tree "$D/wt$N" --rocke-path $K --base $BREF --python '
                f'"$(command -v python)" --budget {MUTANTS_BUDGET_S}{hist} --rerun-budget {MUTANTS_RERUN_S} '
                "--engine $SB/${WTN}/pybind"
                + (
                    f" --torch-python {shlex.quote(tp)}"
                    if (tp := paths.resolve(cfg).get("torch_python"))
                    else ""
                )
            )
            sweep_run = status_lines(
                f"prc nice -n 10 timeout {MUTANTS_BUDGET_S + MUTANTS_RERUN_S + 300} {sweep}",
                "mutants_report.txt",
                "MUTANTS",
                "mutants",
                60,
                "sweep",
                fetch=sweep,
            )
            append_step = (
                f'prr mutants "$OUT/mutants"; timeout 60 python -I -B $M/tools/mutation_sweep.py append --out '
                f'"$OUT/mutants"{hist} --head {head} 2>&1 | head -5'
            )
            lines += sweep_run[:1] + [append_step] + sweep_run[1:]
        # The head against its merge base, merge or no merge, in scratch copies (never wt<N>), git and parsers only:
        # CTest entries that lose a YAML tier in one tree (moved host tests), references left to the files the PR moves
        # or deletes (a README's link to the old path), and backend="python" entry points that need an engine module
        # only the library builds. Each runs on a merge-only round too: a conflict resolution can drop an entry or a
        # file.
        # The test map's PR-CI lanes: config ci_test_tiers (none: the map stops at the tiers).
        ci_tiers = (
            f" --ci-tiers {shlex.quote(json.dumps(cfg['ci_test_tiers']))}"
            if cfg.get("ci_test_tiers")
            else ""
        )
        labels = (
            f"python -I -B $M/tools/ctest_labels.py --repo $R0 --head {head} --out $SO/labels --rocke-path $K "
            f"--base $BREF --budget {LABELS_BUDGET_S}{ci_tiers}"
        )
        labels_run = status_lines(
            f"prc nice -n 10 timeout {LABELS_BUDGET_S + 120} {labels}",
            "labels_report.txt",
            "LABELS",
            "labels",
            70,
            "check",
            fetch=labels,
        )
        lines += (
            [f"sec '{LABELS_SECTION}'"]
            + labels_run[:1]
            + ['prr labels "$OUT/labels"']
            + labels_run[1:]
        )
        lines += [f"sec '{MOVED_REFS_SECTION}'"] + status_lines(
            f"nice -n 10 timeout {MOVED_REFS_BUDGET_S + 60} python -I -B $M/tools/moved_refs.py --repo $R0 --head "
            f'{head} --rocke-path $K --base $BREF --json "$OUT/moved_refs.json" --budget {MOVED_REFS_BUDGET_S}',
            "moved_refs_report.txt",
            "MOVEDREFS",
            "movedrefs",
            50,
            "check",
        )
        lines += [f"sec '{DOCREFS_SECTION}'"] + status_lines(
            f"nice -n 10 timeout {DOCREFS_BUDGET_S + 60} python -I -B $M/tools/doc_refs.py --repo $R0 --head "
            f"{head} --rocke-path $K --base $BREF --budget {DOCREFS_BUDGET_S}",
            "doc_refs_report.txt",
            "DOCREFS",
            "docrefs",
            50,
            "check",
        )
        imports = (
            f"python -I -B $M/tools/import_probe.py --repo $R0 --head {head} --out $SO/imports --rocke-path $K "
            f'--base $BREF --python "$(command -v python)" --budget {IMPORTS_BUDGET_S}'
        )
        imports_run = status_lines(
            f"prc nice -n 10 timeout {IMPORTS_BUDGET_S + 120} {imports}",
            "imports_report.txt",
            "IMPORTS",
            "imports",
            40,
            "probe",
            fetch=imports,
        )
        lines += (
            [f"sec '{IMPORTS_SECTION}'"]
            + imports_run[:1]
            + ['prr imports "$OUT/imports"']
            + imports_run[1:]
        )
        # Our open findings' saved probes, re-run on this tree: the merge, or the head without one (lessons L-525: a
        # finding can sit in a file the push does not touch). Feeds no gate. Not on a head mismatch (wt<N> would hold
        # another head).
        # A draft that already ran every saved probe on this very tree (its probes.txt names the tree's commit and the
        # probes) gives its lines instead: the probes run once a review.
        drafts = ["$WK/pr$N/probes.txt"]
        if draft_stage.shadow_binding(cfg):
            drafts.append(
                shlex.quote(str(draft_stage.shadow_folder(cfg, n) / "probes.txt"))
            )
        # Each line gives the exit code, the tree's commit and the time (PROBE_DETAIL); with a merged tree the probes
        # also run on the head alone, indented, so a reader tells the PR's state from develop's without a re-run (the
        # agents re-ran saved probes by hand in 24 of 39 reviews, agent_script_overlap.md fix 3). The unindented lines
        # (the merge, else the head) are the ones merge_only_flags compares.
        lines += [
            "sec 'Our open findings, re-probed (probes/prN; REPRODUCES = still open)'",
            'if [ -n "$MM" ]; then echo "not run: $MM"; else PT=$W; [ "$MERGED" = 1 ] || PT="$D/wt$N"; '
            'if [ "$MERGED" = 1 ]; then echo "on the merge $(git -C $W rev-parse --short=10 HEAD 2>/dev/null) = '
            'head $(cut -c1-10 "$OUT/head_sha") + develop $(cut -c1-10 "$OUT/develop_sha"):"; else echo "on the head '
            '$(git -C $PT rev-parse --short=10 HEAD 2>/dev/null) (no merged tree):"; fi; '
            'PH="# probes of pr$N at $(git -C $PT rev-parse HEAD 2>/dev/null): $(cd $PB/pr$N 2>/dev/null '
            "&& ls *.sh 2>/dev/null | sed 's/[.]sh$//' | LC_ALL=C sort | paste -sd, -)\"; PF=''; "
            f"for f in {' '.join(drafts)}; do [ -s $f ] && [ \"$(head -1 $f)\" = \"$PH\" ] && PF=$f; done; "
            'if [ -n "$PF" ]; then sed 1d $PF; echo "(a draft ran these probes on this tree: $PF)"; '
            "else PROBE_DETAIL=1 prc python3 -I -B $M/tools/run_probes.py $PB/pr$N $PT; fi; "
            'if [ "$MERGED" = 1 ] && ls $PB/pr$N/*.sh >/dev/null 2>&1; then '
            'HS=$(git -C "$D/wt$N" rev-parse HEAD 2>/dev/null); if [ "$HS" = "$(cat "$OUT/head_sha")" ]; then '
            'echo "on the head alone ${HS:0:10}:"; PROBE_DETAIL=1 prc python3 -I -B $M/tools/run_probes.py $PB/pr$N '
            '"$D/wt$N" | sed \'s/^/   /\'; else echo "on the head alone: not run (wt$N is at ${HS:0:10})"; fi; fi; fi',
        ]
        # Every page: a head with more than 100 check runs (re-runs, duplicate runs) counted from the first page only
        # would miss failures.
        lines += [
            "sec 'CI at head'",
            f"net python3 -I -B $M/tools/ghr.py api --paginate \"repos/{cfg['repo']}/commits/{head}/check-runs?per_page=100\" --jq '.check_runs[] "
            "| .conclusion // .status' 2>/dev/null | sort | uniq -c "
            '| awk \'{printf "%s%s %s", (n++ ? ", " : ""), $2, $1} END {print ""}\'',
            # Per failing job: its first error line, failing tests, develop's latest run of the same job and a
            # known-infrastructure label (tools/ci_head.py; the agents read job logs by hand in 34 of 39
            # reviews, agent_script_overlap.md fix 1). Re-run attempts are marked. "Develop" is the PR's base branch.
            "DVS=$(git -C $R0 rev-list -n 6 $BREF 2>/dev/null | tr '\\n' ' '); "
            f"net timeout 600 python -I -B $M/tools/ci_head.py {head} --develop $DVS 2>&1 | head -60",
            # A re-run hides its earlier attempts from the view above (lessons L-108: a failure in attempt 1
            # behind a hand re-run is still the head's failure).
            f"net python3 -I -B $M/tools/ci_attempts.py {head} {cfg['repo']} 2>/dev/null | head -12",
            # ***Timeout too: a test that times out on every shard is a failure.
            f"net python3 -I -B $M/tools/ci_sha.py {head} 'FAILED |\\*\\*\\*(Failed|Timeout|Exception|Not Run)|Error:' "
            "2>/dev/null | grep -vE '^[0-9]+\\s' | head -20",
            "sec 'Threads and reviews since our last review'",
            # Reviewers' bodies are quoted, so their headings never read as our sections (a reviewer's
            # "## 🔴 Blockers" heading would otherwise read as a checks blocker in a brief).
            f"net python3 -I -B $M/tools/prthreads.py $N {since_s} 500 2>/dev/null "
            "| sed -E '/^(== |20[0-9][0-9]-)/!s/^/    > /' | head -80",
            # Work another open PR also adds, for the code quality pass to give one owner (the operator wants it in
            # every review's checks); read-only, from the tick's scan.
            f"sec '{OVERLAP_SECTION}'",
            "$M/rk overlaps $N --section 2>&1 | head -60",
            # Both land orders with every other open PR, and the silent clashes that merge clean: a name one
            # removes that the other's new lines use, a file one moves that the other includes
            # (the gold audit's land-order traps). Git only; facts cached per head.
            f"sec '{CLASH_SECTION}'",
            *status_lines(
                f"nice -n 10 timeout {CLASH_TIMEOUT_S} python -I -B $M/tools/clash_scan.py $N --head "
                f'{head} --cache $CA/clash --json "$OUT/clash.json"',
                "clash_report.txt",
                "CLASH",
                "clash",
                70,
                "scan",
            ),
            # Struct layouts too: an inserted field has no removed line (lessons L-527); generated
            # headers and exported functions as well. Any NOTE fails the section: a check flag. On the merge, the
            # counters' homes across all files (abi.h define and history, abi.py, the layout canaries) must agree: a
            # half renumber fails it too (lessons L-579).
            f"sec '{ABI_SECTION}'",
            'AM=-; [ "$MERGED" = 1 ] && AM=$(git -C $W rev-parse HEAD 2>/dev/null || echo -)',
            *status_lines(
                f"timeout 300 python -I -B $M/tools/abi_scan.py $R0 $K {head} $BREF $AM",
                "abi_report.txt",
                "ABI",
                "abi",
                14,
                "scan",
            ),
            "sec 'rule_scan (tracker IDs, perf numbers, env reads in added lines)'",
            "python3 -I $T/rule_scan.py $WK/pr$N/pr/diff.patch 2>&1",
            # The repo squash-merges with the PR body as the commit message, so perf numbers in the body land in
            # develop's history too, where the diff scan never sees them.
            "sec 'PR body (a squash merge commits it to develop): perf numbers'",
            "(echo '+++ b/PR_BODY.md'; sed -n '/^## Description/,$p' $WK/pr$N/pr/meta.md | sed 's/^/+/') "
            '> "$OUT/body.patch"; python3 -I $T/rule_scan.py "$OUT/body.patch" 2>&1 | sed -n \'/^== perf/,/^== env/p\' '
            "| grep -v '^== env'",
            # What peer reviewers catch that a script can (peer_reviewer_gap.md): each its own section and status
            # line, a flag but for the leads (LEAD_SECTIONS).
            *[
                x
                for check, (tag, key, title, _lead) in PEER_CHECKS.items()
                for x in [f"sec '{title}'"]
                + status_lines(
                    f"timeout {PEER_BUDGET_S} python -I -B $M/tools/peer_checks.py {check} --repo $R0 "
                    f"--head {head} --base $BREF --rocke-path $K{peer_args(cfg, check)}",
                    f"peer_{check}.txt",
                    tag,
                    key,
                    30,
                    "check",
                )
            ],
            # An armed auto-merge commits the title and message stored when it was armed, not the live ones
            # (lessons L-136: a PR renamed or re-described after it was armed lands its old title).
            "sec 'Auto-merge (an armed squash commits the title and message stored when it was armed)'",
            f"python3 -I -B $M/tools/automerge_msg.py {cfg['repo']} $N 2>&1 | head -6",
        ]
    lines += ['} >> "$R" 2>&1']
    if mode == "review":
        # The slot is kept for the review when its build is the review's shared build (SHARED_BUILD_LINES): the
        # verifier and the code quality reviewer run their tests on it. Its review's end frees it (end_slots).
        lines.append(
            f'[ -s "$OUT/{SHARED_BUILD}" ] && prkeep $N {head} "$SB/${{WTN}}" "$OUT/{SHARED_BUILD}"'
        )
    lines += ["prfree", "echo done"]
    return "\n".join(lines) + "\n"


def gates_from_status(status: dict) -> dict:
    """G5-G7 from a checks/gate status. Missing sections count as not run (pending), never as pass."""

    def verdict(names: list[str], label: str):
        got = {k: status[k] for k in names if k in status}
        if not got:
            return ["pending", f"{label}: not run"]
        bad = [f"{k}: {v[1]}" for k, v in got.items() if v[0] == "fail"]
        if bad:
            return ["fail", "; ".join(bad)[:200]]
        return ["pass", ", ".join(f"{k} {v[0]}" for k, v in got.items())]

    g5 = verdict(["golden", "dg"], "goldens")
    if (
        g5[0] == "pass" and (status.get("dg") or ("",))[0] == "skip"
    ):  # the differential was not run: never a pass
        g5 = ["pending", f"goldens: differential not run ({status['dg'][1]})"]
    g6 = verdict(
        ["merge", "build", "byteid", "byteid_all", "libparity", "tests", "installed"],
        "build and engines",
    )
    g7 = verdict(["compile"], "compile per target")
    if (status.get("compile") or ("",))[
        0
    ] == "skip":  # nothing compiled here (no LLVM tools): not a pass
        g7 = ["pending", f"compile per target: not run ({status['compile'][1]})"]
    elif g7[0] == "pending":
        g7 = ["pass", "no parity family touched: nothing to compile per target"]
    return {"G5": g5, "G6": g6, "G7": g7}


def read_checks_status(out_dir: str) -> tuple[dict, str | None, str | None]:
    import pathlib

    d = pathlib.Path(out_dir)
    status = {}
    if (d / "status").exists():
        for line in (
            (d / "status").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            parts = line.split(" ", 2)
            if len(parts) >= 2:
                status[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    rd = lambda f: (d / f).read_text().strip() if (d / f).exists() else None
    return status, rd("develop_sha"), rd("head_sha")


# --- the pipeline ---------------------------------------------------------------------------------------------------


def _changed_files(cfg: dict, rec: dict) -> list[str]:
    """The files PR rec changes from its merge base with its base branch (develop, or the feature branch it targets)."""
    repo = cfg["paths"]["repo"]
    ref = ensure_pr_ref(cfg, rec)
    mb = gitops.git(repo, "merge-base", base_of(cfg, rec, fetch=True), ref).strip()
    return gitops.git(repo, "diff", "--name-only", mb, ref).splitlines()


# The round and line 1 the worker's draft must carry (lessons L-513, L-001: a draft left to guess its round names the
# wrong one). The launcher's fetch_pr.py stages them in pr/meta.md as `- Round:` and `- Line 1:`; one that predates them
# ignores the two variables.
ROUNDS = {"first-round": "First-round", "follow-up": "Follow-up"}


def mkwt_env(base: str | None) -> str:
    """The environment that has mkwt.py merge the PR's base branch (RKMON_BASE; develop when none) into wt<N>m."""
    return "" if gitops.is_develop(base) else f"RKMON_BASE={shlex.quote(base)} "


def remerge_script(cfg: dict, n: int, base: str | None) -> str:
    """mkwt.py again (the merge with today's base), before a checks job."""
    return (
        f"{paths.shell_vars(cfg)}; N={n}\n{config.export_line(cfg)}\nmkdir -p $WK/pr$N\n"
        f"{mkwt_env(base)}python3 -B {paths.worker_tool('mkwt.py')} $N > $WK/pr$N/mkwt.log 2>&1 "
        "|| echo 'mkwt failed' >> $WK/pr$N/mkwt.log\n"
    )


MKWT_TIMEOUT_S = 900


CLONE_RACE = "shallow file has changed since we read it"  # git's words when another git changed the clone's history


def clone_race(cfg: dict, n: int) -> bool:
    """PR n's staging failed on the shared clone's shallow-file race (its mkwt log says so)."""
    try:
        text = (
            pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "mkwt.log"
        ).read_text(errors="replace")
    except (OSError, paths.PathsError):
        return False
    return CLONE_RACE in text


def trees_gone(cfg: dict, n: int) -> list[str]:
    """PR n's worktrees (wt<N>, wt<N>m) that are not there: a restart empties the RAM disk they live on (ramdisk.py),
    and a link to a tree that is gone counts as gone."""
    return [
        w
        for w in (f"wt{n}", f"wt{n}m")
        if not pathlib.Path(paths.worktree(cfg, w)).is_dir()
    ]


def ensure_merge_tree(
    cfg: dict, rec: dict | None, n: int, run=subprocess.run
) -> list[str]:
    """Make PR n's worktrees again with mkwt.py (the head merged with today's base) when they are gone, before
    something reads them: a draft awaiting verification keeps its pipe's state across a restart, but not its trees on
    the RAM disk. [] when both are there; else one line, what was done or why it failed.
    """
    gone = trees_gone(cfg, n)
    if not gone:
        return []
    try:
        p = run(
            ["bash", "-c", remerge_script(cfg, n, (rec or {}).get("base"))],
            capture_output=True,
            text=True,
            timeout=MKWT_TIMEOUT_S,
        )
        why = (p.stderr or p.stdout or "").strip()[-200:]
    except (OSError, subprocess.SubprocessError, paths.PathsError) as exc:
        why = str(exc)[:200]
    still = trees_gone(cfg, n)
    if still:
        return [
            f"WORKTREES #{n}: {', '.join(gone)} gone, and mkwt.py did not make {', '.join(still)} again "
            f"({why or 'see workers/pr' + str(n) + '/mkwt.log'})"
        ]
    return [
        f"WORKTREES #{n}: {', '.join(gone)} made again with mkwt.py (gone, e.g. the RAM disk emptied by a restart): "
        "the head merged with today's base"
    ]


def worker_script(
    cfg: dict,
    n: int,
    extra_sparse: list[str],
    kind: str,
    unload: bool = True,
    rnd: str = "",
    base: str | None = None,
) -> str:
    """The worker job: a fresh workspace, then review_agent_run.py. `unload`: first unload the models a local Ollama
    holds, so the draft starts on a clean GPU; only when no other pipe's worker is running, since they share it. `rnd`
    (fr, fu<k>) names the review's round for the offload passes (rkmon_lib/offload.py). `base`: the PR's base branch,
    which mkwt.py merges (mkwt_env)."""
    sparse = " ".join(extra_sparse)
    stage = (
        f"REVIEW_ROUND={ROUNDS[kind]} REVIEW_LINE1={shlex.quote(bindings.first_line(cfg))} RKMON_PIPE=pr{n} "
        + (f"RKMON_OFFLOAD_ROUND={shlex.quote(rnd)} " if rnd else "")
        + mkwt_env(base)
    )
    return (
        f"{paths.shell_vars(cfg)}; N={n}\n{config.export_line(cfg)}\n"
        "chmod -R u+w $WK/pr$N 2>/dev/null; rm -rf $WK/pr$N\n"
        + (
            "for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop $m 2>/dev/null; done\n"
            if unload
            else ""
        )
        + f"{stage}SPARSE_EXTRA='{sparse}' NUM_BATCH=256 python3 -B {paths.worker_tool('review_agent_run.py')} $N\n"
    )


def has_draft(cfg: dict, n: int, m: str | None = None) -> bool:
    """The reviewer model's draft: review_agent.md, or with the offload on (`m`, the review's offload mode; the
    configuration's when it is not given), the draft pass's draft.md."""
    import pathlib
    from . import offload

    name = "draft.md" if (m or offload.mode(cfg)) == "on" else "review_agent.md"
    p = pathlib.Path(f"{paths.get(cfg, 'workers')}/pr{n}/{name}")
    return p.exists() and p.stat().st_size > 0


def ready_done(st: dict, n: int) -> bool:
    """PR n left the awaiting-verification list (its review finished: review_finished; its head moved; or skipped:
    `rk review-skip`); whether it was on it."""
    before = st.get("review_ready", [])
    st["review_ready"] = [r for r in before if r["pr"] != n]
    return len(st["review_ready"]) < len(before)


def review_finished(st: dict, n: int, ours: dict | None = None) -> None:
    """PR n's review is finished: posted (post.post), kept unposted while posting is off (post.save_draft), or written
    to the mirror's folder (mirror.divert). Its pipe and its entry awaiting verification end, and with them what they
    hold: its trees and its share of the RAM disk (cleanup.in_use), and its place among the work in flight, which
    the scaler, the shadow drafter, the driver's picks and the dashboards count. `ours`, when given, is our last review
    on the PR from now on (rec["ours"]), as the round logic reads it: the PR is not reviewed again at that head, and a
    new head gets a follow-up."""
    drop_pipe(st, n)
    ready_done(st, n)
    rec = st["prs"].get(str(n))
    if ours is not None and rec is not None:
        rec["ours"] = ours


def waiting_on_verifier(st: dict) -> list[dict]:
    """The reviews that wait on a verifier, in the state's order: review_ready's entries (each handed to verification
    by the driver's REVIEW_READY), except one of a PR the operator skipped (`rk review-skip`). A finished review has
    left the list (review_finished). Whatever asks which reviews wait on a verifier reads them here.
    """
    skip = set(st.get("review_skip") or [])
    return [
        r
        for r in st.get("review_ready") or []
        if isinstance(r, dict) and r.get("pr") not in skip
    ]


def close_review(st: dict, store, n: int, head: str, why: str) -> list[str]:
    """Close PR n's review of `head` without a post, once our last review covers that head (`rk classify N --record` on
    a merge- or rebase-only head): a draft awaiting verification leaves the list, and a review in flight (its pipe)
    stops its jobs and frees its heavy slot. Journalled; the lines say what was closed.
    """
    out = []
    if any(
        r["pr"] == n and same_commit(r.get("head"), head)
        for r in st.get("review_ready") or []
    ):
        ready_done(st, n)
        store.journal(
            "review", f"#{n} draft closed without a post at {head[:10]}: {why}"
        )
        out.append(
            f"#{n}: the draft awaiting verification at {head[:10]} is closed without a post ({why})"
        )
    pipe = pipe_of(st, n) or {}
    if same_commit(pipe.get("head"), head):
        stop_pipe(st, store, n)
        store.journal(
            "review",
            f"#{n} review in flight closed without a post at {head[:10]}: {why}",
        )
        out.append(
            f"#{n}: the review in flight at {head[:10]} is stopped and closed without a post ({why})"
        )
    return out


def settled(cfg: dict, st: dict, rec: dict) -> bool:
    """The plan's settle rule (store.settle_left), at the time of asking: a head is reviewed once it is `settle_min`
    old (`settle_develop_fix_min` at most for a develop fix), or `settle_burst_min` after the last of a burst of pushes.
    """
    return settle_left(cfg, st, rec)[0] <= 0


# --- the parallel reviews -----------------------------------------------------------------------------------------------
# Up to the scaler's current pipes run at once (rkmon_lib/scaler.py: from 1 to config.pipes_max(), the deploy schema's
# maximum, starting at `review.pipes`), fewer when the reviewer endpoint allows fewer (config.pipes): one worker job,
# one pair of worktrees (wt<N>, wt<N>m) and one workspace (workers/pr<N>) per PR. Each pipe's checks need a heavy slot
# (jobs.heavy_busy: the scaler's current heavy setting, starting at `checks.limit`, which the pipes never change); a
# draft that finishes while the slots are taken waits for one. With the RAM disk on, a pipe or a heavy job also waits
# for room there (scaler.room).


def pipes(st: dict) -> list[dict]:
    """The parallel reviews running (state `review_pipes`), each one PR's draft: {"pr", "kind", "head", "since",
    "started", "stage", "worker", "checks", "files", "plan"?}. A state from before the pipes holds one
    `review_pipeline`: it becomes the first pipe."""
    old = st.pop("review_pipeline", None)
    ps = st.setdefault("review_pipes", [])
    if (
        old
        and old.get("pr") is not None
        and not any(p.get("pr") == old.get("pr") for p in ps)
    ):
        ps.insert(0, old)
    return ps


def pipe_of(st: dict, n: int) -> dict | None:
    return next((p for p in pipes(st) if p.get("pr") == n), None)


def drop_pipe(st: dict, n: int) -> dict | None:
    """Take PR n's pipe out of the pipes (its jobs are the caller's to stop) and return it."""
    p = pipe_of(st, n)
    if p is not None:
        st["review_pipes"] = [x for x in pipes(st) if x is not p]
    return p


# --- reviews a stop or a restart interrupted ----------------------------------------------------------------------------
# `rk stop` and `rk kill` (a container stop, a restart, an update's restart) end every pipe's jobs. The review is not
# given up: its pipe is recorded in state `review_interrupted` (by PR: {"pr", "kind", "head", "again", "at",
# "queue_entry"?}) and the first advances after `rk start` start it again at that head, before any new review (the
# worker and the checks run again; a draft that already awaits verification stays in review_ready). A PR closed, or
# whose head moved meanwhile, leaves the record instead (one the review queue started goes back to the queue: requeue);
# `rk review-skip N` drops it, and `rk review-start N` replaces it (first_started). In a mirror the feed line's review
# stays started meanwhile.
INTERRUPTED = "review_interrupted"


def interrupt_pipe(st: dict, store, p: dict) -> str:
    """Record pipe `p` as interrupted (its jobs are the caller's to stop) and take it out of the pipes; the line."""
    drop_pipe(st, p["pr"])
    st.setdefault(INTERRUPTED, {})[str(p["pr"])] = {
        "pr": p["pr"],
        "kind": p.get("kind") or "first-round",
        "head": p.get("head") or "",
        "again": bool(p.get("again")),
        "at": now(),
        **carried_entry(p),
    }
    store.journal(
        "review",
        f"#{p['pr']}'s review stopped by rk stop/kill; it starts again at {str(p.get('head'))[:10]} after rk start",
    )
    return (
        f"#{p['pr']}'s review ({p.get('kind')}) stopped: it starts again after rk start"
    )


def interrupted(st: dict, n: int | None = None):
    """The interrupted reviews (oldest first), or PR n's record (None)."""
    book = st.get(INTERRUPTED) or {}
    if n is not None:
        return book.get(str(n))
    return sorted(book.values(), key=lambda e: e.get("at") or "")


def forget_interrupted(st: dict, n: int) -> bool:
    return (st.get(INTERRUPTED) or {}).pop(str(n), None) is not None


def resume_interrupted(
    cfg: dict, st: dict, store, room: int
) -> tuple[list[str], set[int]]:
    """Start the interrupted reviews again, oldest first, up to `room` pipes: (the lines, the PRs started). A PR closed,
    untracked or with another head now leaves the record (its review ended: a pipe would drop it too, and one the review
    queue started goes back to the queue); one in a pipe or awaiting verification already leaves it as well. A start
    that waits (GitHub not read) keeps the record.
    """
    from . import mirror

    lines, started = [], set()
    for e in interrupted(st):
        if len(started) >= room:
            break
        n = e["pr"]
        rec = st["prs"].get(str(n))
        if (
            pipe_of(st, n)
            or any(r["pr"] == n for r in st.get("review_ready") or [])
            or n in (st.get("review_skip") or [])
        ):
            forget_interrupted(st, n)
            continue
        if (
            not rec
            or rec.get("status") != "open"
            or not same_commit(rec["head"], e["head"])
        ):
            forget_interrupted(st, n)
            store.journal(
                "review",
                f"#{n} interrupted review not started again: closed or head moved",
            )
            lines.append(
                f"REVIEW_RESUME #{n}: not started again: its head moved since the stop; {REQUEUED}"
                if requeue(st, store, e, "its head moved while its review was stopped")
                else f"REVIEW_RESUME #{n}: not started again: closed or its head moved since the stop"
            )
            continue
        ok, why = scaler.room(cfg, st, store, "pipe", n)
        if not ok:
            lines.append(f"REVIEW_WAIT #{n}: it waits for room on this machine: {why}")
            break
        lines.append(
            f"REVIEW_RESUME #{n}: the {e['kind']} a stop interrupted starts again at {e['head'][:10]}"
        )
        if mirror.on(cfg):
            s = mirror.state(st)["started"].get(mirror.key_of(n, e["head"])) or {}
            got = mirror.start(
                cfg,
                st,
                store,
                {
                    "pr": n,
                    "head": e["head"],
                    "round": s.get("round") or "",
                    "rkind": e["kind"],
                    "again": bool(s.get("again", e.get("again"))),
                    "at": s.get("feed_at") or e["at"],
                },
            )
        else:
            got = start_review(
                cfg, st, store, n, e["kind"], force=True, again=e.get("again", False)
            )
        lines += got
        last = got[-1] if got else ""
        if last.startswith("REVIEW_START"):
            started.add(n)
            p = pipe_of(st, n)
            if p is not None:
                p.update(carried_entry(e))
        if not last.startswith(("REVIEW_WAIT", "REVIEW_SETTLING")):
            forget_interrupted(st, n)
    return lines, started


def stop_pipe(st: dict, store, n: int) -> dict | None:
    """Drop PR n's pipe and stop its running jobs, freeing its heavy slot."""
    p = drop_pipe(st, n)
    for which in ("worker", "checks", "finish", QUALITY, FIRST_PASS):
        job = (p or {}).get(which)
        if job and jobs.status(store, job) == "running":
            jobs.kill(store, job)
        jobs.release_heavy(st, job)
    return p


# A local worker's first prompt is its biggest. Heavy CPU (a native build) alongside it wedges a local runner: it logs
# `new prompt` and then nothing until the request times out (under a gate job's build, for example). So
# while the reviewer runs on this machine, no heavy job starts in a worker's first PREFILL_MIN minutes (the standing
# machine limit), and no worker starts in a running heavy job's first HEAVY_BUILD_MIN minutes, while it builds. A remote
# reviewer's prompts load another machine: neither rule applies.
PREFILL_MIN = 5
HEAVY_BUILD_MIN = 10


def worker_prefill(cfg: dict, st: dict) -> bool:
    """A local reviewer's worker started less than PREFILL_MIN minutes ago: no heavy job may start now. A draft stage
    that runs its model elsewhere (draft_stage.py: prefill_min 0) holds nothing back."""
    if not config.reviewer_is_local(cfg):
        return False
    return any(
        draft_stage.of_pipe(p).prefill_min > 0
        and (p.get("worker") or {}).get("started")
        and minutes_since(p["worker"]["started"]) < PREFILL_MIN
        for p in pipes(st)
    )


def heavy_building(cfg: dict, st: dict, store) -> dict | None:
    """A heavy job in its first HEAVY_BUILD_MIN minutes, while the reviewer runs on this machine: no worker may start
    now."""
    if not config.reviewer_is_local(cfg):
        return None
    for h in jobs.heavy_holders(st):
        if (
            h.get("started")
            and minutes_since(h["started"]) < HEAVY_BUILD_MIN
            and jobs.status(store, h) == "running"
        ):
            return h
    return None


# --- the review queue -------------------------------------------------------------------------------------------------
# State review_first: the reviews the driver starts before the plan's, in order, each {"pr": N, "again": bool} (`rk
# review-queue`), with "kind" when it carries a round forced on the PR. An entry leaves the queue once a start of its
# review starts it, finds the PR closed, or finds the head covered by our review and records it so (REVIEW_SKIP); the
# advance drops the entries of PRs closed or merged. The review an entry starts keeps the entry (QUEUE_ENTRY: on its
# pipe, then on its review_ready entry or its interrupted record): dropped with no post while its PR is open, the review
# puts the entry back at the queue's front as it was, carrying the review's kind when a start forced it (`rk
# review-start N --kind K`), so the next start is the same round (requeue).
# `again` reviews once more the head of our last review (posted, or kept unposted while posting is off) or a head we
# posted on whose post of ours is gone from the PR: neither our review there nor the follow-up gap holds the entry back
# (the settle does), and start_review does not carry the head of our last review; a rebase- or merge-only push since
# our review is handled as without it.
FIRST = "review_first"
QUEUED = "QUEUED"  # the kind of a review-queue entry among the items pick_next reads (first_items)
LEAVES_QUEUE = (
    "REVIEW_START #",
    "REVIEW_SKIP #",
    "REVIEW_CLOSED #",
)  # a start's last line that ends the PR's entry
QUEUE_ENTRY = "queue_entry"  # a review's field: the review-queue entry that started it
REQUEUED = "back in the review queue"  # what a drop line and the journal say of a review that requeue puts back


def first_items(st: dict) -> list[dict]:
    """The review queue's entries as items for pick_next. The driver passes them ahead of the plan, so a queued review
    goes first; the pool's lookahead (draft_stage.candidates) reads the plan alone."""
    return [
        {"kind": QUEUED, "pr": e["pr"], "entry": e, "text": "review queue"}
        for e in st.get(FIRST) or []
    ]


def round_kind(rec: dict) -> str:
    """The review a start of PR rec makes: a follow-up once our review has a head, else a first round."""
    return "follow-up" if (rec.get("ours") or {}).get("head") else "first-round"


def entry_kind(rec: dict, e: dict) -> str:
    """The review a start of review-queue entry `e` of PR rec makes: the round it carries (requeue), else the PR's
    next one (round_kind)."""
    return e.get("kind") or round_kind(rec)


def first_wait(cfg: dict, st: dict, store, e: dict, last: dict | None = None) -> str:
    """What review-queue entry `e` waits for, "" when its review may start now. Every entry waits while its PR is not
    open, its review is in a pipe or awaits verification, it is in review_skip, its in-depth review waits for the
    operator's approval (needs_approval), or its head has not settled. Without `again` it also waits while we posted at
    the head or our review covers it, and, as a follow-up, for the gap after our last review of the PR (plan.gap_until).
    Its review is the round the entry carries, else the PR's next one (entry_kind). `last`: last_review_posts.
    """
    n = e["pr"]
    rec = st["prs"].get(str(n))
    if not rec or rec.get("status") != "open":
        return f"its PR is {(rec or {}).get('status') or 'not tracked'}"
    if pipe_of(st, n):
        return "its review is in flight"
    if any(r["pr"] == n for r in st.get("review_ready") or []):
        return "its review awaits verification"
    if n in (st.get("review_skip") or []):
        return "it is in review_skip (rk review-unskip)"
    kind = entry_kind(rec, e)
    if needs_approval(cfg, st, n, kind):
        return "your approval"
    left, why = settle_left(cfg, st, rec)
    if left > 0:
        return f"its head settles in {left:.0f} min ({why})"
    if e.get("again"):
        return ""
    posted = store.ledger_has("post", n, rec["head"])
    if posted or covers(rec.get("ours"), rec["head"]):
        return (
            f"{'we posted at its head' if posted else 'our review covers its head'} (a re-review there: rk "
            f"review-queue {n} --again)"
        )
    until = (
        gap_until(
            cfg, st, rec, last_review_posts(store.ledger()) if last is None else last
        )
        if kind == "follow-up"
        else None
    )
    return f"the gap after our last review, until {until[11:16]}Z" if until else ""


def unqueue(st: dict, store, n: int, why: str) -> dict | None:
    """PR n leaves the review queue, journalled: its entry, None when it was not queued."""
    q = st.get(FIRST) or []
    e = next((x for x in q if x["pr"] == n), None)
    if e is None:
        return None
    st[FIRST] = [x for x in q if x["pr"] != n]
    store.journal("review", f"#{n} left the review queue: {why}")
    return e


def first_started(st: dict, store, n: int, lines: list[str]) -> None:
    """After a start of PR n's review (the driver's pick, or `rk review-start`): its queue entry ends once the start's
    last line says the review started, that our review covers the head (REVIEW_SKIP), or that the PR is closed; after
    REVIEW_SETTLING or REVIEW_WAIT it stays. The review it started keeps the entry (QUEUE_ENTRY), for requeue. A review
    a stop interrupted gives way to the one started: its record ends, and the entry it carries passes to the new review
    when the PR was not queued since.
    """
    if not lines or not lines[-1].startswith(LEAVES_QUEUE):
        return
    e = unqueue(st, store, n, lines[-1].split(":", 1)[0])
    p = pipe_of(st, n) if lines[-1].startswith("REVIEW_START #") else None
    if p is None:
        return
    stopped = interrupted(st, n) or {}
    p.update({QUEUE_ENTRY: dict(e)} if e else carried_entry(stopped))
    if stopped:
        forget_interrupted(st, n)


def carried_entry(r: dict) -> dict:
    """The review queue's entry that review record `r` (a pipe, a review_ready entry, an interrupted record) took, as
    the field to copy into the record that follows it; {} for a review the plan started.
    """
    e = r.get(QUEUE_ENTRY)
    return {QUEUE_ENTRY: dict(e)} if e else {}


def requeue(st: dict, store, r: dict, why: str) -> bool:
    """Review record `r` ended with no post: one the review queue started goes back to the queue's front, as its entry
    was, while its PR is open; journalled with `why`. Whether it went back. An entry the PR took since (`rk
    review-queue` during the review) goes to the front instead, with its own options. The entry carries the review's
    kind when a start forced it (`rk review-start N --kind K`, other than the PR's next round: round_kind), so the next
    start is the same round: a first round forced on a reviewed giant PR stays the in-depth review its approval is
    for."""
    e, n = r.get(QUEUE_ENTRY), r["pr"]
    rec = st["prs"].get(str(n)) or {}
    if not e or rec.get("status") != "open":
        return False
    q = st.get(FIRST) or []
    back = dict(next((x for x in q if x["pr"] == n), e))
    back.pop("kind", None)
    if r.get("kind") and r["kind"] != round_kind(rec):
        back["kind"] = r["kind"]
    st[FIRST] = [back] + [x for x in q if x["pr"] != n]
    store.journal("review", f"#{n} {REQUEUED}: {why}")
    return True


def prune_first(st: dict, store) -> None:
    """The review-queue entries whose PR is closed, merged or no longer tracked leave the queue."""
    for e in list(st.get(FIRST) or []):
        status = (st["prs"].get(str(e["pr"])) or {}).get("status") or "not tracked"
        if status in ("closed", "merged", "not tracked"):
            unqueue(st, store, e["pr"], f"the PR is {status}")


def queue_add(st: dict, store, ns: list[int], again: bool) -> str:
    """`rk review-queue N [N ...] [--again]`: each PR joins the end of the review queue, in the order given; a PR queued
    already keeps its place and takes `again`. ValueError, before any change, for a PR that is not open.
    """
    for n in ns:
        rec = st["prs"].get(str(n))
        if not rec or rec.get("status") != "open":
            raise ValueError(
                f"#{n} is {(rec or {}).get('status') or 'not tracked'}: only an open PR is queued"
            )
    q = st.setdefault(FIRST, [])
    for n in dict.fromkeys(ns):
        e = next((x for x in q if x["pr"] == n), None)
        if e:
            e["again"] = again
        else:
            q.append({"pr": n, "again": again})
    line = f"queued {', '.join('#%d' % n for n in dict.fromkeys(ns))}{' with --again' if again else ''}"
    store.journal(
        "review", f"review queue: {line}; now {', '.join('#%d' % e['pr'] for e in q)}"
    )
    return line


def queue_lines(cfg: dict, st: dict, store) -> list[str]:
    """The review queue, for `rk review-queue` and `rk review-status`: each entry in order, and what it waits for."""
    q = st.get(FIRST) or []
    if not q:
        return ["review queue: empty"]
    last, ahead = last_review_posts(store.ledger()), False
    out = [
        f"review queue ({len(q)}; the driver starts these before the plan, in order):"
    ]
    for i, e in enumerate(q, 1):
        why = first_wait(cfg, st, store, e, last)
        state = f"waits: {why}" if why else "may start" if ahead else "next"
        ahead = ahead or not why
        kind = f" ({e['kind']})" if e.get("kind") else ""
        out.append(
            f"  {i}. #{e['pr']}{kind}{' --again' if e.get('again') else ''}: {state}"
        )
    return out


def unpickable(st: dict) -> set[int]:
    """The PRs no start picks: in review_skip, in a pipe, or awaiting verification."""
    return (
        set(st.get("review_skip", []))
        | {r["pr"] for r in st.get("review_ready", [])}
        | {p["pr"] for p in pipes(st)}
    )


def pick_next(cfg: dict, st: dict, store, plan: list[dict], exclude=()) -> dict | None:
    """The next review to start, {"pr", "kind", "again"}: the first of `plan`'s items that may start now: settled, not
    skipped, not in a pipe (or `exclude`) or awaiting verification, not too large for the autopilot, and not already
    covered by our review. The driver passes the review queue's entries first (first_items): an entry may start once
    first_wait finds nothing to wait for, and none of the plan's checks applies to it.
    """
    skip = unpickable(st) | set(exclude)
    # A review too large for the autopilot, once it could start by hand, holds back every lower-priority one, so the
    # advance names it (REVIEW_MANUAL) rather than starting them first (were REVIEW_MANUAL to fire only once nothing
    # else could start, a P1 follow-up would wait behind P4s).
    big = {b["pr"] for b in oversized(cfg, st, plan, store)}
    last = last_review_posts(store.ledger())
    held_below = None
    for item in plan:
        if item["kind"] == QUEUED:
            e = item["entry"]
            if e["pr"] not in exclude and not first_wait(cfg, st, store, e, last):
                return {
                    "pr": e["pr"],
                    "kind": entry_kind(st["prs"][str(e["pr"])], e),
                    "again": bool(e.get("again")),
                }
            continue
        if item["kind"] not in ("FOLLOW_UP", "FIRST_ROUND"):
            continue
        n = item["pr"]
        if (
            held_below is not None
            and item.get("prio") is not None
            and item["prio"] > held_below
        ):
            return None
        rec = st["prs"].get(str(n))
        if (
            not rec
            or n in skip
            or rec["size"] > registry.value(cfg, "autopilot_max_lines")
        ):
            if n in big and held_below is None and item.get("prio") is not None:
                held_below = item["prio"]
            continue
        # The driver uses the last tick's plan, so re-check its "(wait: ...)" notes now (else a P1 follow-up waits for
        # the next tick while the driver starts P3 first rounds), and any head that plan never saw: a
        # review start since may have taken a newer live head (start_review).
        unseen = (rec.get("head_seen_at") or "") > (st.get("last_tick") or "")
        if ("(wait:" in item["text"] or unseen) and not settled(cfg, st, rec):
            continue
        if store.ledger_has("post", n, rec["head"]):
            continue
        if covers(rec.get("ours"), rec["head"]):
            # Our review covers this head since the plan was made: kept unposted, or recorded as rebase-only.
            continue
        # One follow-up post per PR per gap, checked now as the settle is: it waits, and the reviews behind it go on.
        if item["kind"] == "FOLLOW_UP" and gap_until(cfg, st, rec, last):
            continue
        return {
            "pr": n,
            "kind": "follow-up" if item["kind"] == "FOLLOW_UP" else "first-round",
            "again": False,
        }
    return None


def oversized(cfg: dict, st: dict, plan: list[dict], store=None) -> list[dict]:
    """The waiting reviews too large for the autopilot (`autopilot_max_lines`), each {"pr", "kind", "size", "cap"}, the
    cap being the one it was judged by. The orchestrator starts them by hand, one at a time (`rk review-start N`): the
    driver alone never starts them and says "nothing to start now". An in-depth review the operator has not approved
    is not among them: it waits on the Needs you list (awaiting_approval).
    """
    cap = registry.value(cfg, "autopilot_max_lines")
    busy = unpickable(st)
    last = last_review_posts(store.ledger()) if store is not None else {}
    out = []
    for item in plan:
        rec = st["prs"].get(str(item["pr"])) or {}
        if (
            item["kind"] in ("FOLLOW_UP", "FIRST_ROUND")
            and item["pr"] not in busy
            and (rec.get("size") or 0) > cap
            and not needs_approval(cfg, st, item["pr"])
        ):
            # Only a settled head, as on the automatic path (never a minute after a push).
            if rec.get("head_seen_at") and not settled(cfg, st, rec):
                continue
            # Reviewed since the plan was made (posted, or kept unposted): nothing to start.
            if covers(rec.get("ours"), rec.get("head")):
                continue
            if store is not None and store.ledger_has(
                "post", item["pr"], rec.get("head")
            ):
                continue
            # Nor a follow-up inside the gap after our last review.
            if item["kind"] == "FOLLOW_UP" and gap_until(cfg, st, rec, last):
                continue
            out.append(
                {
                    "pr": item["pr"],
                    "kind": (
                        "follow-up" if item["kind"] == "FOLLOW_UP" else "first-round"
                    ),
                    "size": rec["size"],
                    "cap": cap,
                }
            )
    return out


def manual_lines(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    """REVIEW_MANUAL for the first oversized review, once nothing else can start; the driver exits on it. One verifier
    cannot read such a diff, so a first round's verification is split (`rk review-split N`: parts, then a cross-cutting
    pass); a follow-up is a light check of our open findings and the delta.
    """
    big = oversized(cfg, st, plan, store)
    if not big:
        return []
    b, h = big[0], heavy_building(cfg, st, store)
    when = f"once {h['id']} is past its first {HEAVY_BUILD_MIN} minutes" if h else "now"
    more = f" (then {', '.join('#%d' % x['pr'] for x in big[1:])})" if big[1:] else ""
    return [
        f"REVIEW_MANUAL #{b['pr']} ({b['kind']}, {b['size']} lines > {b['cap']}): too "
        f"large for the autopilot; start it by hand {when} with `rk review-start {b['pr']}`, and at its "
        f"REVIEW_READY {in_depth_step(b['pr'], b['kind'])}{more}"
    ]


def in_depth_step(n: int, kind: str) -> str:
    """What the orchestrator does at the REVIEW_READY of PR n's review too large for the autopilot, a review of `kind`:
    REVIEW_MANUAL says it ahead, and REVIEW_READY again (a review the queue started had no REVIEW_MANUAL). A giant PR
    gets one in-depth review, split; its later rounds are light: one verifier checks our open findings and the delta,
    with no split and no code quality pass.
    """
    return (
        f"split the verification with `rk review-split {n}`"
        if kind == "first-round"
        else "brief one light verifier on our open findings and the delta (no split, no code quality pass)"
    )


# --- the operator's approval of an in-depth review --------------------------------------------------------------------
# An in-depth review is a first round of a PR over autopilot_max_lines: its verification is split into many agents
# (`rk review-split N`), so it costs too much to start unasked. It is the PR's first review, or a first round forced on
# a reviewed PR (`rk review-start N --kind first-round`). None starts until the operator approves it (`rk
# review-approve N`, or Approve on the admin dashboard), whatever starts it: start_review refuses it, the driver and a
# mirror's feed leave it, REVIEW_MANUAL does not name it, a review queue entry waits for it (first_wait) and `rk
# review-start` refuses it, while the Needs you list asks for the approval of a PR's first review (awaiting_approval,
# approval_need). Decline is `rk review-skip N` (skip), or Decline on the admin dashboard (decline). An approval stands,
# whatever the PR's head does, until a review of ours on the PR is recorded after it, the review it was for, or the PR
# closes (prune_approved). The later rounds are light, one verifier each, and need no approval. State
# review_approved: the approvals, each {"pr": N, "at": when the operator gave it}.
APPROVED = "review_approved"


def approval(n: int) -> dict:
    """The operator's approval of an in-depth review of PR n, given now: its entry in review_approved."""
    return {"pr": n, "at": now()}


def approved(st: dict, n: int) -> bool:
    """Whether an approval of an in-depth review of PR n stands."""
    return any(a["pr"] == n for a in st.get(APPROVED) or [])


def needs_approval(cfg: dict, st: dict, n: int, kind: str | None = None) -> bool:
    """Whether a start of PR n's review as `kind` (by default its next round: round_kind) would be an in-depth one the
    operator has not approved: the PR is over autopilot_max_lines, and the start is a first round or the PR's next
    round is (a forced follow-up of its first review does not get around it)."""
    rec = st["prs"].get(str(n)) or {}
    return (
        (rec.get("size") or 0) > registry.value(cfg, "autopilot_max_lines")
        and "first-round" in (kind, round_kind(rec))
        and not approved(st, n)
    )


def awaiting_approval(cfg: dict, st: dict, plan: list[dict]) -> list[int]:
    """The open PRs whose in-depth review waits for the operator's approval, in the order the driver reaches them: the
    review queue's entries, then the plan's first rounds; none in review_skip, in a pipe or awaiting verification.
    """
    ns = [e["pr"] for e in st.get(FIRST) or []] + [
        i["pr"] for i in plan if i.get("kind") == "FIRST_ROUND"
    ]
    taken = unpickable(st)
    return [
        n
        for n in dict.fromkeys(ns)
        if n not in taken
        and (st["prs"].get(str(n)) or {}).get("status") == "open"
        and needs_approval(cfg, st, n)
    ]


def approve_how(n: int, page: bool = True) -> str:
    """Where the operator approves PR n's in-depth review: the admin dashboard's Approve, or the command; with `page`
    False the command alone, for a first round forced on a reviewed PR, which the Needs you list does not ask for.
    """
    from .doctor import rk_command

    cmd = f"`{rk_command('review-approve', str(n))}`"
    return f"on the admin dashboard or with {cmd}" if page else f"with {cmd}"


def approval_wait(cfg: dict, st: dict, store, n: int) -> str:
    """What `rk review-queue N`, `rk review-start N` and a refused start say of a PR whose in-depth review waits for the
    operator's approval: its size, the agents and cost the review runs (split.estimate), and how to approve it.
    """
    from . import split

    rec = st["prs"][str(n)]
    return (
        f"#{n} is {rec['size']:,} changed lines: its in-depth review runs {split.estimate(cfg, store, rec['size'])}. "
        f"It starts once you approve it {approve_how(n, round_kind(rec) == 'first-round')}."
    )


def approval_need(cfg: dict, st: dict, store, n: int, team: bool = False) -> str:
    """PR n's line on the Needs you list while its in-depth review waits for the operator's approval
    (awaiting_approval): its size, the agents and cost the review runs (split.estimate), and how to approve or decline
    it. `team`: the line as the team sees it (the team dashboard, the Teams copy), short enough to fit whole: the
    size and the agents, with no cost, which stays on the admin side, and no command, as the commands stay on the
    board and the admin dashboard."""
    from . import split
    from .doctor import rk_command

    rec = st["prs"][str(n)]
    ask = (
        f"an in-depth review of {rec['size']:,} changed lines needs your OK: "
        f"{split.estimate(cfg, store, rec['size'], cost=not team)}"
    )
    if team:
        return f"#{n}: {ask}. Approve or decline it on the admin dashboard."
    who = f" ({rec['author']})" if rec.get("author") else ""
    return (
        f"#{n}{who}: {ask}. Approve it {approve_how(n)}; decline it there or with "
        f"`{rk_command('review-skip', str(n))}`."
    )


def not_in_depth(cfg: dict, st: dict, n: int) -> str:
    """Why PR n has no in-depth review for the operator to approve: it is not open, or within the autopilot's limit;
    "" when it has one, its first review or a first round forced on it once reviewed."""
    rec = st["prs"].get(str(n))
    if not rec or rec.get("status") != "open":
        return f"#{n} is {(rec or {}).get('status') or 'not tracked'}: nothing changed."
    size, cap = rec.get("size") or 0, registry.value(cfg, "autopilot_max_lines")
    if size <= cap:
        return (
            f"#{n} is {size:,} changed lines, within the {cap:,} the autopilot reviews: its review needs no "
            "approval. Nothing changed."
        )
    return ""


def approve(cfg: dict, st: dict, store, n: int) -> tuple[bool, str]:
    """The operator approves PR n's in-depth review (`rk review-approve N`, the admin dashboard's Approve): (done, what
    it says). Its first review starts at its turn, from the review queue or the plan (REVIEW_MANUAL); once the PR is
    reviewed, a first round forced on it may start (`rk review-start N --kind first-round`). The PR leaves
    review_skip; journalled. Refused, nothing changed, when PR n has no in-depth review (not_in_depth).
    """
    from . import split
    from .doctor import rk_command

    why = not_in_depth(cfg, st, n)
    if why:
        return False, why
    rec = st["prs"][str(n)]
    skipped = n in (st.get("review_skip") or [])
    st[APPROVED] = [a for a in st.get(APPROVED) or [] if a["pr"] != n] + [approval(n)]
    st["review_skip"] = [x for x in st.get("review_skip") or [] if x != n]
    store.journal(
        "review",
        f"review-approve #{n}: its in-depth review ({rec['size']} changed lines) may start"
        + ("; out of review_skip" if skipped else ""),
    )
    est = f"{rec['size']:,} changed lines: {split.estimate(cfg, store, rec['size'])}"
    said = (
        f"#{n} approved: its in-depth review ({est}) starts at its turn"
        if round_kind(rec) == "first-round"
        else f"#{n} approved: an in-depth review of it ({est}) may start with "
        f"`{rk_command('review-start', str(n), '--kind', 'first-round')}`; its next review is otherwise a light one"
    )
    return True, said + (", and the PR is out of review_skip." if skipped else ".")


def skip(st: dict, store, n: int) -> list[str]:
    """`rk review-skip N`, and the admin dashboard's Decline of an in-depth review: PR n joins review_skip, out of the
    autopilot's picks; its review in flight stops, an interrupted one is not started again, one awaiting verification
    is withdrawn, and an approval of its in-depth review ends; journalled. What it did, one line each.
    """
    st["review_skip"] = sorted(set(st.get("review_skip") or []) | {n})
    lines = []
    if stop_pipe(st, store, n):
        lines.append(f"#{n}'s review stopped")
    if forget_interrupted(st, n):
        lines.append(f"#{n}'s interrupted review is not started again")
    if ready_done(st, n):
        lines.append(f"#{n}'s review awaiting verification is withdrawn")
    if approved(st, n):
        st[APPROVED] = [a for a in st[APPROVED] if a["pr"] != n]
        lines.append(f"#{n}'s approval of its in-depth review is withdrawn")
    store.journal("review", f"review-skip #{n}")
    return lines


def decline(cfg: dict, st: dict, store, n: int) -> tuple[bool, str]:
    """The admin dashboard's Decline of PR n's in-depth review, its first review: `rk review-skip N` (skip). (done,
    what it says); refused, nothing changed, as approve is, and once the PR is reviewed: its next review is a light
    one, which a Decline would skip with every round after it."""
    from .doctor import rk_command

    why = not_in_depth(cfg, st, n)
    if not why and round_kind(st["prs"][str(n)]) != "first-round":
        why = (
            f"#{n}'s next review is a later round, a light one (one verifier): it needs no approval. Nothing "
            "changed."
        )
    if why:
        return False, why
    skip(st, store, n)
    return True, (
        f"#{n} declined: its in-depth review does not start, and the PR is in review_skip "
        f"(`{rk_command('review-unskip', str(n))}` puts it back)."
    )


def prune_approved(st: dict) -> None:
    """An approval ends with the review it was for: once a review of ours on the PR is recorded after it (rec["ours"],
    posted or kept unposted, is later than the approval), or once the PR is closed, merged or no longer tracked.
    """
    if st.get(APPROVED):
        st[APPROVED] = [
            a
            for a in st[APPROVED]
            if (rec := st["prs"].get(str(a["pr"])) or {}).get("status") == "open"
            and ((rec.get("ours") or {}).get("at") or "") <= a["at"]
        ]


# What a review start says when it starts nothing. After REVIEW_CLOSED or REVIEW_SETTLING the advance tries the next PR;
# after REVIEW_WAIT (GitHub did not answer) it stops.
NOT_STARTED = ("REVIEW_WAIT #", "REVIEW_CLOSED #", "REVIEW_SETTLING #")
TRY_NEXT = ("REVIEW_SKIP", "REVIEW_CLOSED #", "REVIEW_SETTLING #")


def take_live_head(cfg: dict, st: dict, store, rec: dict, head: str) -> str:
    """The state's head of PR `rec` is out of date: take the live `head` as the next tick would (events.apply_push),
    with its commit date for the settle, and leave the PUSH event for that tick to report.
    """
    old = rec["head"]
    st.setdefault("pending_events", []).append(events.apply_push(rec, head, now()))
    try:  # the settle counts from the commit date too (store.head_age_min); when unknown, from now
        rec["head_committed_at"] = (
            gh.rest(f"repos/{cfg['repo']}/commits/{head}", jq=".commit.committer.date")
            or None
        )
    except gh.GhError:
        rec["head_committed_at"] = None
    note = f"#{rec['n']}: state head {old[:10]} was out of date; using live head {head[:10]}"
    store.journal("review", note)
    return f"review {note}"


def start_review(
    cfg: dict,
    st: dict,
    store,
    n: int,
    kind: str,
    force: bool = False,
    again: bool = False,
) -> list[str]:
    """Start PR n's review at its live head, in a pipe of its own: the follow-up classification, then the
    reviewer model's worker, its worktrees and workspace (a follow-up after a rebase only needs none: the new head
    is recorded as equivalent, REVIEW_SKIP). `force` (`rk review-start --now`) starts a live head that has not
    settled. `again` (`rk review-start --again`, a review-queue entry's) asks for a re-review: the head of our last
    review (posted, or kept unposted while posting is off) is reviewed once more, as a follow-up at that head, instead
    of carried; a rebase- or merge-only push is handled as without it. A re-review at that head, or at a head we posted
    on whose post of ours is gone from the PR, is journalled as one. An in-depth review the operator has not approved
    starts nothing, whatever starts it (REVIEW_WAIT: needs_approval)."""
    rec = st["prs"][str(n)]
    if needs_approval(cfg, st, n, kind):
        return [f"REVIEW_WAIT #{n}: {approval_wait(cfg, st, store, n)}"]
    # One head per review, read from GitHub now: the worker and mkwt fetch the PR's head as they start, so the review
    # record and the checks take the same one (with the tick loop down, the state's head may be older than the one the
    # worker reviews).
    try:
        live = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq="{state, head: .head.sha}")
        if not isinstance(live, dict) or not live.get("head"):
            raise gh.GhError(f"unexpected answer {str(live)[:80]!r}")
    except gh.GhError as exc:
        return [
            f"REVIEW_WAIT #{n}: its live head was not read ({str(exc)[:160]}); nothing started"
        ]
    if live.get("state") != "open":
        return [
            f"REVIEW_CLOSED #{n}: it is {live.get('state')} on GitHub; nothing started (the next tick catches up)"
        ]
    out = []
    if not same_commit(live["head"], rec["head"]):
        out.append(take_live_head(cfg, st, store, rec, live["head"]))
        left, why = settle_left(cfg, st, rec)
        if left > 0 and not force:
            return out + [
                f"REVIEW_SETTLING #{n}: its live head {live['head'][:10]} has not settled ({why}); it "
                f"settles in {left:.0f} min"
            ]
    since, mo, rereview = None, None, again and store.ledger_has("post", n, rec["head"])
    if kind == "follow-up" and (rec.get("ours") or {}).get("head"):
        c = classify(cfg, rec, rec["ours"]["head"])
        rec["classify"] = c
        since = rec["ours"]["at"]
        # A re-record (golden re-bless) after a 🔴/🟠 review may be the very fix it asked for (a re-blessed golden our
        # 🟠 asked for is no equivalent head), so it gets a follow-up.
        from .queue import GOOD_VERDICTS

        rerecord_ok = rec["ours"].get("verdict") in GOOD_VERDICTS
        if again and c["kind"] == "same-head":
            # The head of our last review, reviewed once more: not carried.
            rereview = True
        elif c["kind"] in COVERED or (c["kind"] == "rebase+re-record" and rerecord_ok):
            return out + [carry(store, rec, c)]
        elif c["kind"] == "merge-only":
            # Its worker and free checks run (the worker beside the checks, at no cost to the pipeline, so a flagged
            # head has its draft); _advance posts nothing when the checks come back clean. The probes' results and the
            # checks' items (peer_carry) at our reviewed head are read now, before the worker wipes workers/prN.
            m = c.get("merges") or 0
            mo = {
                "old": rec["ours"]["head"],
                "merges": m,
                "probes": probe_baseline(cfg, rec, n),
                "peer": peer_carry.baseline(
                    rec, f"{paths.get(cfg, 'workers')}/pr{n}/checks"
                ),
            }
            rec["merge_check"] = {
                "old": rec["ours"]["head"],
                "new": c["new"],
                "merge_only": True,
                "merges": m,
            }
            out.append(
                f"#{n} is merge-only since our review ({m} merge{'s' if m != 1 else ''}): nothing is posted if its "
                "checks come back clean"
            )
        else:
            out.append(
                f"#{n} changed substantively: {c['n_changed']} files, e.g. {', '.join(c['changed_files'][:4])}"
                if "n_changed" in c
                else f"#{n} changed substantively: our reviewed head {c['old'][:10]} is not "
                f"in the mirror ({c['kind']})"
            )
    files = _changed_files(cfg, rec)
    extra = sparse_extra(cfg, files)
    stage = draft_stage.for_config(cfg)
    m, rnd = "off", ""
    if stage is draft_stage.LOCAL:
        # The local worker wipes workers/prN: a pool draft still writing there stops first (no pool feeds this stage).
        draft_stage.drop(cfg, st, store, n, "the live stage is the local review loop")
        # A local Ollama's models are unloaded before a draft only when no other pipe's worker uses them.
        sharing = [
            p
            for p in pipes(st)
            if p.get("pr") != n
            and (p.get("worker") or {}).get("id")
            and jobs.status(store, p["worker"]) == "running"
        ]
        unload = (
            config.reviewer_is_local(cfg)
            and config.worker(cfg)["backend"] == "ollama"
            and not sharing
        )
        # The review offload (offload.mode shadow or on): the worker job also runs the offload passes, for this round.
        from . import offload, prep

        m = offload.mode(cfg)
        rnd = prep.default_round(st, n) if m != "off" else ""
        job, how = (
            jobs.start(
                store,
                "worker",
                f"pr{n}",
                worker_script(
                    cfg, n, extra, kind, unload=unload, rnd=rnd, base=rec.get("base")
                ),
                timeout_min=offload.WORKER_TIMEOUT_MIN if m != "off" else 60,
            ),
            "worker started",
        )
    else:
        # The draft stage's contract (draft_stage.py): the pool's draft of this head, else a new job once the scaler
        # allows another draft.
        pooled = draft_stage.adopt(st, n, rec["head"])
        if pooled:
            job, how = pooled["job"], "pool draft adopted"
        else:
            draft_stage.drop(
                cfg, st, store, n, "the review starts at another head"
            )  # it writes the same folder
            if stage.parallel:
                ok, reason = scaler.may_start(
                    cfg,
                    st,
                    store,
                    st.get("last_plan") or [],
                    n,
                    draft_stage.running(st, store),
                )
                if not ok:
                    return out + [f"REVIEW_WAIT #{n}: its draft waits: {reason}"]
            job, how = (
                draft_stage.start_job(cfg, st, store, stage, n, ROUNDS[kind], extra),
                f"{stage.label} started",
            )
    drop_pipe(st, n)
    pipes(st).append(
        {
            "pr": n,
            "kind": kind,
            "head": rec["head"],
            "since": since,
            "started": now(),
            "stage": "worker",
            "worker": job,
            "checks": None,
            "files": len(files),
            "merge_only": mo,
            "again": bool(again),
            "reviewed": (
                (rec.get("ours") or {}).get("head") if kind == "follow-up" else None
            ),
            **({} if stage is draft_stage.LOCAL else {"stage_name": stage.name}),
            **({"offload": m, "round": rnd, "finish": None} if m != "off" else {}),
        }
    )
    what = f"{kind}{', merge-only' if mo else ''}{', a re-review at the same head' if rereview else ''}"
    store.journal(
        "review",
        f"#{n} {what}: {how} ({job['id']}) at {rec['head'][:10]}; "
        f"{len(pipes(st))} parallel review(s) running",
    )
    shown = "reviewer model worker started" if stage is draft_stage.LOCAL else how
    from . import (
        review_cost,
    )  # a call-time import: review_cost reads prep, which imports this module

    review_cost.feed(st, store, n, kind, again=again, merge_only=bool(mo))
    return out + [
        f"REVIEW_START #{n} ({what}): {shown} at {rec['head'][:10]} (parallel review "
        f"{len(pipes(st))} of {pipes_in_effect(cfg, store)})"
    ]


# The review driver advances from the state the last tick saved, and the state is as old as that tick. Past this age no
# review starts: the tick loop may be down, and a review would start from heads hours old.
TICK_STALE_MIN = 25


def tick_stale(st: dict) -> str | None:
    """TICK_STALE once the last tick is over TICK_STALE_MIN minutes old: `rk review-advance` then starts nothing, and
    the review driver exits on the line, so the orchestrator hears of it. Jobs already running keep running; a tick
    clears it."""
    last = st.get("last_tick")
    if not last:
        return "TICK_STALE: no tick on record: the tick loop may be down; run rk tick"
    age = minutes_since(last)
    if age <= TICK_STALE_MIN:
        return None
    return f"TICK_STALE: the last tick was at {last} ({age:.0f} min ago): the tick loop may be down; run rk tick"


_DEPTH = [
    0
]  # advance's own nesting: the draft pool and the shadow drafts advance once a call from outside


def advance(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    _DEPTH[0] += 1
    try:
        top = _DEPTH[0] == 1
        out = []
        if top:
            # The scaler's sample (the current pipes and heavy checks, the RAM disk's footprints), then the RAM disk's
            # prompt cleanup: a review that is over leaves it at once (rkmon_lib/scaler.py, rkmon_lib/ramdisk.py).
            from . import cleanup, ramdisk, review_keep

            out += scaler.adjust(cfg, st, store) + ramdisk.sweep(
                cfg, st, store, cleanup.in_use(st)
            )
            try:  # the review directories on the RAM disk, kept on the disk too (a restart empties the RAM disk)
                out += review_keep.sync(cfg, st, store)
            except OSError as exc:
                out.append(f"(the review directories were not kept: {str(exc)[:160]})")
            out += end_slots(st)
        # The shadow drafts that ended are recorded first, so REVIEW_READY names one that is done.
        out += finish_rechecks(cfg, st, store) + (
            draft_stage.shadow_collect(cfg, st, store) if top else []
        )
        out += _advance(cfg, st, store, plan)
        if top:
            from . import mirror

            pool = (
                [] if mirror.on(cfg) else draft_stage.pool_advance(cfg, st, store, plan)
            )  # mirror: feed lines only
            out += pool + draft_stage.shadow_start(cfg, st, store, plan)
        return out
    finally:
        _DEPTH[0] -= 1


def end_slots(st: dict, environ=None, run=None, records=None) -> list[str]:
    """Free the PR runners' slots of the reviews that ended. A review ends at its post, when its PR closes, or when a new
    head supersedes its own: it is no longer in flight then (review_keep.heads_in_flight). The slot its checks kept for
    its shared build and the named work taken for it are freed, and the named work whose owner is gone with them
    (prcode.free_ended). The lines; nothing on a host."""
    from . import review_keep

    heads = review_keep.heads_in_flight(st)
    return prcode.free_ended(
        lambda pr, head: (str(pr), str(head)) in heads, environ, run, records
    )


def pipes_in_effect(cfg: dict, store=None) -> int:
    """How many reviews run at once: the scaler's current pipes (review.pipes before its first sample), never past
    what the reviewer endpoint allows the local worker (config.pipes: a local Ollama allows fewer). A draft stage that
    drafts elsewhere, or none, has no such cap: its endpoint's limits are the scaler's and the limiter's.
    """
    return scaler.pipes_in_effect(cfg, store)


def _advance(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    """One step of every pipe in flight (oldest first, so the oldest draft gets a free heavy slot first), then new
    pipes up to the number in effect. Prints REVIEW_READY for each finished draft."""
    out = []
    for r in list(
        st.get("review_ready", [])
    ):  # a draft awaiting verification goes stale when its PR moves on
        rec = st["prs"].get(str(r["pr"]))
        if (
            not rec
            or rec.get("status") != "open"
            or not same_commit(rec["head"], r["head"])
        ):
            ready_done(st, r["pr"])
            store.journal(
                "review",
                f"#{r['pr']} draft dropped before verification: PR closed or head moved",
            )
            out.append(
                f"REVIEW_DROP #{r['pr']}: its head moved before verification; {REQUEUED}"
                if requeue(st, store, r, "its head moved before its review was posted")
                else f"REVIEW_DROP #{r['pr']}: closed or head moved before verification; will re-pick"
            )
    for pipe in sorted(
        pipes(st),
        key=lambda p: (p.get("worker") or {}).get("started") or p.get("started") or "",
    ):
        out += (
            quality_start(cfg, st, store, pipe)
            + first_pass_start(cfg, st, store, pipe)
            + _advance_pipe(cfg, st, store, pipe)
        )
    return out + _fill(cfg, st, store, plan)


def _advance_pipe(cfg: dict, st: dict, store, pipe: dict) -> list[str]:
    n, rec = pipe["pr"], st["prs"].get(str(pipe["pr"]))
    if (
        not rec
        or rec.get("status") != "open"
        or not same_commit(rec["head"], pipe["head"])
    ):
        # Stop its jobs first: a dropped worker would keep running, and the next advance would start another beside it
        # (two workers on one GPU).
        stop_pipe(st, store, n)
        if draft_stage.of_pipe(pipe) is not draft_stage.LOCAL:
            draft_stage.record(
                cfg,
                st,
                store,
                n,
                pipe,
                "dropped",
                pipe.get("checks"),
                "closed or head moved",
            )
        store.journal("review", f"#{n}'s review dropped: PR closed or head moved")
        if requeue(st, store, pipe, "its head moved during the review"):
            return [f"REVIEW_DROP #{n}: its head moved; {REQUEUED}"]
        return [f"REVIEW_DROP #{n}: closed or head moved; will re-pick"]
    stage = draft_stage.of_pipe(pipe)
    ws, cs = jobs.status(store, pipe["worker"]), jobs.status(store, pipe["checks"])
    # The checks start once no worker of a local reviewer is in its first minutes (PREFILL_MIN, its own included), in
    # a free heavy slot; a remote reviewer's worker loads this machine lightly, so they start with it. A stage that
    # drafts elsewhere starts them once it has staged the PR (or has ended, so a staging that failed is seen below).
    if stage is draft_stage.LOCAL:
        checks_go = not (
            config.reviewer_is_local(cfg)
            and minutes_since(pipe["worker"]["started"]) < PREFILL_MIN
        )
    else:
        checks_go = draft_stage.staged(cfg, n) or ws != "running"
    if pipe["checks"] is None and checks_go and not worker_prefill(cfg, st):
        from . import mirror

        if mirror.live_checks(
            cfg
        ):  # a mirror: the live review's checks for this head, copied; no build here
            wait = int(float(mirror.settings(cfg)["checks_wait_min"]))
            job = jobs.start(
                store,
                "checks",
                f"pr{n}",
                mirror.checks_script(cfg, n, pipe["head"]),
                timeout_min=wait + 5,
            )
            pipe["checks"] = job
            return [
                f"REVIEW #{n}: worker {ws}; checks: the live review's for {pipe['head'][:10]}, copied once whole "
                f"({job['id']})"
            ]
        busy = jobs.heavy_busy(cfg, st, store)
        if busy:
            return [
                f"REVIEW #{n}: worker {ws}; checks wait: all parallel builds are busy ({busy['id']} among them)"
            ]
        ok, why = scaler.room(
            cfg, st, store, "heavy", n
        )  # its build goes on the RAM disk, when it is on
        if not ok:
            return [f"REVIEW #{n}: worker {ws}; checks wait: {why}"]
        # Develop sanity after a landing goes first (P0): this step freed a finished check's slot and took it again in
        # the same advance, so a pending sanity would never find it free. The tick's develop guard step starts the
        # sanity once the slot is free.
        pend, g = st["develop"].get("pending_sanity") or [], st.get("guard") or {}
        if (
            pend
            and not g.get("running")
            and (g.get("last_sanity_green") or "")[:12] != pend[-1][:12]
        ):
            return [
                f"REVIEW #{n}: worker {ws}; checks wait for develop sanity on {pend[-1][:10]}"
            ]
        import pathlib

        tree = paths.worktree(cfg, f"wt{n}m")
        if not pathlib.Path(tree).exists():
            if ws not in ("running",) and clone_race(cfg, n):
                # Its staging met another git changing the shared clone (worker/mkwt.py): not the PR's fault, so no
                # skip; it starts again (in a mirror its feed line is taken again, not dropped; one the review queue
                # started goes back to the queue).
                from . import mirror

                drop_pipe(st, n)
                mirror.state(st)["started"].pop(mirror.key_of(n, pipe["head"]), None)
                store.journal(
                    "review",
                    f"#{n}: its staging met a change of the shared clone; it starts again",
                )
                back = requeue(
                    st, store, pipe, "its staging met a change of the shared clone"
                )
                return [
                    f"REVIEW_FAIL #{n}: its staging met another fetch into the shared clone (shallow file changed); "
                    + (REQUEUED if back else "it starts again, not skipped")
                ]
            if ws not in (
                "running",
            ):  # the worker ended without making its worktrees: give up on this PR for now
                drop_pipe(st, n)
                st.setdefault("review_skip", []).append(n)
                store.journal(
                    "review",
                    f"#{n} failed: worker {ws} and no worktrees; skipped until `rk review-unskip`",
                )
                return [
                    f"REVIEW_FAIL #{n}: worker {ws}, no worktrees (see {jobs.tail(store, pipe['worker'], 5)!r})"
                ]
            return [f"REVIEW #{n}: worker {ws}; waiting for its worktrees"]
        fams = emitter_families(tree, paths.repository_folder(cfg))
        pc = plan_checks(cfg, rec, _changed_files(cfg, rec), fams, source=tree)
        pc["_dirs"] = {f: fams[f] for f in pc["families"]}
        # The review's own head, the one its worker fetched (start_review); the script checks the tree holds it. A pool
        # draft staged over drafter.restage_min ago is merged with today's develop first, as a re-check is.
        remerge = ""
        if (
            stage is not draft_stage.LOCAL
            and ws != "running"
            and draft_stage.staged_min(cfg, n)
            > bindings.drafter_settings(cfg)["restage_min"]
        ):
            remerge = remerge_script(cfg, n, rec.get("base"))
        reviewed = pipe.get("reviewed") or (
            (rec.get("ours") or {}).get("head") if pipe["kind"] == "follow-up" else None
        )
        job = jobs.start(
            store,
            "checks",
            f"pr{n}",
            remerge
            + checks_script(
                cfg,
                n,
                pipe["head"],
                pipe["kind"],
                pc,
                pipe["since"],
                reviewed=reviewed,
                merge_only=bool(pipe.get("merge_only")),
                base=rec.get("base"),
            ),
            timeout_min=240,
        )
        pipe["checks"], pipe["plan"] = job, {
            k: v for k, v in pc.items() if k != "_dirs"
        }
        jobs.take_heavy(st, job, store)
        return [
            f"REVIEW #{n}: worker {ws}; checks started ({len(pc['families'])} families, build={pc['build']})"
        ]
    for which, s in (("worker", ws), ("checks", cs)):
        if s == "timeout":
            jobs.kill(store, pipe[which])
    if (
        stage.on_missing == "ready"
        and stage.retries
        and ws != "running"
        and pipe["checks"] is not None
        and not has_draft(cfg, n, "off")
        and not pipe.get("draft_retry")
    ):
        # The draft stage's one retry, on the staging already there, as soon as the job ended without a draft.
        ok, reason = scaler.may_start(
            cfg, st, store, st.get("last_plan") or [], n, draft_stage.running(st, store)
        )
        if not ok:
            return [
                f"REVIEW #{n}: the draft ended with no draft ({ws}); its retry waits: {reason}"
            ]
        job = draft_stage.start_job(
            cfg, st, store, stage, n, ROUNDS[pipe["kind"]], [], retry=True
        )
        pipe["draft_retry"] = {"failed": ws, "at": now(), "job": pipe["worker"]}
        pipe["worker"] = job
        store.journal(
            "review",
            f"#{n} the draft ended with no draft ({ws}); retrying once ({job['id']})",
        )
        return [
            f"REVIEW #{n}: the draft ended with no draft ({ws}); retrying once ({job['id']})"
        ]
    if ws in ("running",) or cs in ("running", "none"):
        return [f"REVIEW #{n}: worker {ws}, checks {cs}"]
    if quality_wait(cfg, store, pipe):
        return [
            f"REVIEW #{n}: worker {ws}, checks {cs}; the code quality pass runs ({pipe[QUALITY]['id']})"
        ]
    # A merge-only head: clean checks, and nothing is posted; the head is recorded as covered by our review and its
    # pipe takes the next review. So is a head whose only flag is CI failures that are all known infrastructure (a
    # configured log signature, or the same first error on develop or on other open PRs today: known_infra), with a
    # CI_INFRA line naming each job and why. The item flags count only for what our review did not see (peer_carry):
    # every flagged item seen at a head it covers carries the head too (a failing test or a pull-request reference also
    # when our last review's open rows name it); new threads or a new title problem make a micro round, the verifier
    # alone on them (REVIEW_READY's `micro:`). So do reviews and comments others left since our last review
    # (activity_since_ours): a head with something to answer is never carried silently. Anything else flagged, a new
    # failing test or reference among them, goes to verification as usual, saying why.
    flags, micro, pre = [], [], []
    if pipe.get("merge_only"):
        _record_checks_gates(cfg, rec, n)
        flags = merge_only_flags(cfg, n, pipe)
        rest, new_items, settled = peer_split(
            cfg, n, pipe, flags, lambda: our_last_review_body(cfg, rec, n)
        )
        infra = None
        if len(rest) == 1 and rest[0].startswith(CI_NEW_FLAG) and pipe.get("ci_new"):
            infra = known_infra.classify(cfg, st, n, pipe["head"], pipe["ci_new"])
            if (
                infra
            ):  # the dashboards count such a PR as waiting on CI, not as a failure of its own (report.ci_cause)
                rec["ci_infra"] = {"head": pipe["head"], "jobs": infra}
        if not rest or infra:
            new_items = new_items + activity_since_ours(cfg, n, rec)
        if (
            not rest or infra
        ) and new_items:  # a micro round: the verifier alone, on the new items
            micro, flags = new_items, []
            if infra:
                pre = [known_infra.line(n, pipe["head"], infra)]
                store.journal("review", pre[-1])
        elif not rest or infra:
            k = pipe["merge_only"].get("merges") or 0
            why = (
                "its checks are clean"
                if not rest
                else "its only flag is CI failures that are known infrastructure"
            )
            if settled:
                names = (
                    ", ".join(settled[:-1])
                    + (" and " if len(settled) > 1 else "")
                    + settled[-1]
                )
                why += f"; its {names} flag{'s repeat' if len(settled) > 1 else ' repeats'} what our review saw"
            line = carry(
                store,
                rec,
                {
                    "kind": "merge-only",
                    "old": pipe["merge_only"]["old"],
                    "new": pipe["head"],
                },
                f" ({k} merge{'s' if k != 1 else ''}; {why}, so nothing is posted)",
            )
            lines = [line]
            if infra:
                lines.append(known_infra.line(n, pipe["head"], infra))
                store.journal("review", lines[-1])
            if stage is not draft_stage.LOCAL:
                draft_stage.record(
                    cfg,
                    st,
                    store,
                    n,
                    pipe,
                    "dropped",
                    pipe.get("checks"),
                    (
                        "merge-only, clean checks"
                        if not rest
                        else "merge-only, known CI infrastructure"
                    ),
                )
            stop_pipe(st, store, n)
            return lines
    why = (
        f" Merge-only since our review, but the checks flag: {'; '.join(flags)[:600]}."
        if flags
        else ""
    )
    if micro:
        why = (
            f" Merge-only since our review; {MICRO}: {'; '.join(micro)[:600]}. The verifier alone (no code quality "
            "pass), briefed on these items only; the other rows carry from our last review."
        )
    elif (rec.get("size") or 0) > registry.value(cfg, "autopilot_max_lines"):
        why += (
            f" {rec['size']:,} changed lines, too large for the autopilot: "
            f"{in_depth_step(n, pipe['kind'])}."
        )
    from . import post as post_mod

    if store.ledger_has(
        "post", n, pipe["head"]
    ):  # rk post refuses a second post at a head unless told --again
        why += " We posted at this head already: post it with --again."
    elif post_mod.keeps(cfg) and (was := post_mod.last_review_at(rec, pipe["head"])):
        # While posting is off, a keep at the head of our last review is refused unless told --again too.
        why += (
            f" Our last review is at this head already ({was}): keep it with --again."
        )
    m = pipe.get("offload") or "off"
    drafted = ws == "done" and (
        has_draft(cfg, n, m) if m == "on" else has_draft(cfg, n)
    )
    if m != "off" and drafted:
        # The review offload's finish job, once the checks are done: the assembly pass (it reads the checks' report)
        # and `rk evidence-check`. A light job: the checks' heavy slot is free while it runs.
        from . import offload

        jobs.release_heavy(st, pipe.get("checks"))
        if pipe.get("finish") is None:
            job = jobs.start(
                store,
                "offload",
                f"pr{n}",
                offload.finish_script(cfg, n, pipe["round"], m),
                timeout_min=offload.FINISH_TIMEOUT_MIN,
            )
            pipe["finish"], pipe["stage"] = job, "finish"
            return [
                f"REVIEW #{n}: worker done, checks {cs}; the offload's assembly and evidence check started "
                f"({job['id']})"
            ]
        fs = jobs.status(store, pipe["finish"])
        if fs == "running":
            return [
                f"REVIEW #{n}: the offload's assembly and evidence check run ({pipe['finish']['id']})"
            ]
        if fs == "timeout":
            jobs.kill(store, pipe["finish"])
    drop_pipe(st, n)
    jobs.release_heavy(st, pipe.get("checks"))
    if stage is draft_stage.LOCAL and not drafted:
        # No draft from the reviewer model: never hand this to verification (line 1 credits it). Retry once, then skip;
        # one the review queue started is retried from the queue.
        tries = st.setdefault("review_retries", {})
        tries[str(n)] = tries.get(str(n), 0) + 1
        if tries[str(n)] >= 2:
            st.setdefault("review_skip", []).append(n)
            store.journal(
                "review",
                f"#{n} worker {ws} twice with no draft; skipped until `rk review-unskip {n}`",
            )
            return [
                f"REVIEW_FAIL #{n}: worker {ws} again, no draft; skipped until `rk review-unskip {n}`"
            ]
        store.journal("review", f"#{n} worker {ws}, no draft; it will be retried")
        if requeue(st, store, pipe, f"its worker ended {ws} with no draft"):
            return [f"REVIEW_FAIL #{n}: worker {ws}, no draft; {REQUEUED}"]
        return [f"REVIEW_FAIL #{n}: worker {ws}, no draft; will retry"]
    pipe["stage"] = "verify"
    _record_checks_gates(cfg, rec, n)
    # The GPU-only cases the checks skipped go to a remote GPU node, queued (remote-gpu-tests), never holding the
    # review: the post carries their row, pending, and `rk gpu-row` updates it later.
    from . import gpu

    try:
        gpu_lines = gpu.after_checks(cfg, st, store, n, pipe["head"])
    except Exception as exc:  # noqa: BLE001 - a GPU runner problem never holds a review
        gpu_lines = [f"(gpu: #{n}'s GPU-only cases were not queued: {str(exc)[:200]})"]
    w = f"{paths.get(cfg, 'workers')}/pr{n}"
    # Pipelining (the operator's rule: keep the backlog moving): the draft and checks are done, so its pipe
    # takes the next review now while the orchestrator verifies this one; verifications run in parallel.
    ready = {
        "pr": n,
        "kind": pipe["kind"],
        "head": pipe["head"],
        "at": now(),
        "started": pipe.get("started"),
        "jobs": {
            k: pipe[k] for k in ("worker", "checks", QUALITY, FIRST_PASS) if pipe.get(k)
        },
        QUALITY_ON_BINDING: quality_runs(cfg, st, pipe),
        **({"micro": micro} if micro else {}),
        **carried_entry(pipe),
    }
    st.setdefault("review_ready", []).append(ready)
    if (
        m == "on"
    ):  # the reviewer model's review: audited, or reviewed in depth (rkmon_lib/offload.py)
        from . import offload

        folder = offload.review_dir(cfg, n, m)
        reasons = offload.deep_reasons(
            cfg, rec, pipe["kind"], _changed_files(cfg, rec), folder
        )
        ready.update(
            offload="deep" if reasons else "audit", why=reasons, round=pipe.get("round")
        )
        store.journal(
            "review",
            f"#{n} offload review ready for {'a deep review' if reasons else 'its audit'}"
            + (f": {'; '.join(reasons)}" if reasons else "")
            + "; the next review starts in its place",
        )
        return (
            pre
            + gpu_lines
            + [
                _ready_line(
                    ready,
                    offload.ready_line(cfg, n, pipe["kind"], w, folder, reasons)
                    + why
                    + quality_note(cfg, st, store, pipe)
                    + mirror_note(cfg, st, pipe),
                )
            ]
        )
    store.journal(
        "review",
        f"#{n} ready for verification"
        + (
            f" ({MICRO}: {len(micro)} new item" f"{'s' if len(micro) != 1 else ''})"
            if micro
            else ""
        )
        + "; the next review starts in its place",
    )
    if stage is not draft_stage.LOCAL:
        # The draft stage's contract: a draft, or REVIEW_READY with the draft marked missing (its retry spent).
        status = (
            "ready"
            if has_draft(cfg, n, "off")
            else "missing" if stage.retries else "none"
        )
        rec_d = draft_stage.record(cfg, st, store, n, pipe, status, pipe.get("checks"))
        return (
            pre
            + gpu_lines
            + [
                _ready_line(
                    ready,
                    draft_stage.ready_line(cfg, n, pipe["kind"], cs, why, rec_d)
                    + quality_note(cfg, st, store, pipe)
                    + mirror_note(cfg, st, pipe),
                )
            ]
        )
    shadow = (
        f". Shadow offload (posts nothing): {w}/offload; after our post, `rk shadow-report {n}`"
        if m == "shadow"
        else ""
    )
    return (
        pre
        + gpu_lines
        + [
            _ready_line(
                ready,
                f"REVIEW_READY #{n} ({pipe['kind']}): worker {ws}, checks {cs}.{why} Inputs: "
                f"{w}/review_agent.md, {w}/checks/checks.md, {w}/pr/meta.md. Verify with the rocKE Review "
                f"Verifier, then `rk post {n} FILE`{shadow}"
                + draft_lint.ready_note(n, w, config.tracker_keys(cfg))
                + draft_stage.shadow_note(cfg, st, n, pipe["head"])
                + quality_note(cfg, st, store, pipe)
                + mirror_note(cfg, st, pipe),
            )
        ]
    )


def _ready_line(ready: dict, line: str) -> str:
    """The REVIEW_READY line, kept on its review_ready entry too: the orchestrator hands it again, verbatim, when no
    verifier holds the review (its first hand-over was lost: rkmon_lib/orchestrator.py unheld_ready).
    """
    ready["line"] = line
    return line


def ready_line_of(cfg: dict, r: dict) -> str:
    """A review_ready entry's REVIEW_READY line: the one its driver printed, or, for an entry from before the entries
    kept it, one naming the draft's inputs."""
    if r.get("line"):
        return r["line"]
    n, w = r["pr"], f"{paths.get(cfg, 'workers')}/pr{r['pr']}"
    return (
        f"REVIEW_READY #{n} ({r.get('kind') or 'review'}): awaiting verification at {str(r.get('head'))[:10]} since "
        f"{r.get('at')}. Inputs: {w}/review_agent.md, {w}/checks/checks.md, {w}/pr/meta.md. Verify with the rocKE "
        f"Review Verifier, then `rk post {n} FILE`"
    )


# --- the code quality pass on a binding (roles.quality) ---------------------------------------------------------------
# With config roles.quality naming a binding, the driver runs the code quality pass itself (tools/quality_run.py,
# worker/quality_lane.py) beside the draft and the checks, once the PR is staged, and REVIEW_READY waits for it and
# names its quality.md, which `rk review-prep` puts in the review directory for the verifier. Without one the
# orchestrator's code quality reviewer runs it, as the skill says. A merge-only head gets none, nor does a review too
# large for the autopilot (quality_unused) or one in mirror arm A (quality_runs). REVIEW_READY keeps the review's
# answer on its review_ready entry (QUALITY_ON_BINDING), for what asks once the pipe is gone (quality_runs_for).
QUALITY = "quality"
QUALITY_ON_BINDING = "quality_on_binding"


def quality_binding(cfg: dict) -> dict | None:
    """The quality role's binding (config roles.quality), or None."""
    return bindings.binding(cfg, QUALITY)


def quality_unused(cfg: dict, rec: dict | None) -> bool:
    """Whether nothing uses a code quality pass on the binding in PR rec's review: a review too large for the autopilot
    (over autopilot_max_lines), whose REVIEW_READY says what takes the pass's place (in_depth_step: the split review's
    part verifiers run its checks, and a later round is light, with none)."""
    return ((rec or {}).get("size") or 0) > registry.value(cfg, "autopilot_max_lines")


def quality_runs(cfg: dict, st: dict, pipe: dict) -> bool:
    """Whether the review in `pipe` gets the code quality pass on the quality binding: the role has a binding, and the
    review is neither at a merge-only head, nor too large for the autopilot (quality_unused), nor in mirror arm A
    (Claude's code quality pass, as live). The pass's start, REVIEW_READY (its note, and its entry for
    quality_runs_for) and a mirror's note all ask here.
    """
    from . import mirror

    return bool(
        quality_binding(cfg)
        and not pipe.get("merge_only")
        and not quality_unused(cfg, st["prs"].get(str(pipe["pr"])))
        and mirror.arm_of(cfg, st, pipe["pr"], pipe["head"]) != "A"
    )


def quality_runs_for(cfg: dict, st: dict, n: int) -> bool:
    """quality_runs for PR n's review, whose pipe may be gone (`rk review-prep`, a mirror's `rk post`): the answer its
    REVIEW_READY kept on its review_ready entry; with none kept (no entry, or one from before the entries kept it), the
    rule for its pipe, else for a review at the head it is for."""
    r = next((x for x in st.get("review_ready") or [] if x.get("pr") == n), None) or {}
    if QUALITY_ON_BINDING in r:
        return bool(r[QUALITY_ON_BINDING])
    head = r.get("head") or (st["prs"].get(str(n)) or {}).get("head") or ""
    return quality_runs(cfg, st, pipe_of(st, n) or {"pr": n, "head": head})


def quality_folder(cfg: dict, n: int) -> pathlib.Path:
    """The folder of PR n's code quality pass: <workers>/pr<N>/quality."""
    return pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / QUALITY


def quality_script(cfg: dict, n: int) -> str:
    """The bash of the code quality pass's job: tools/quality_run.py with the worker tools' settings."""
    root = (cfg.get("paths") or {}).get("root") or paths.get(cfg, "working_dir")
    return (
        f"{config.export_line(cfg)}\nRKMON_REVIEW_ROOT={shlex.quote(str(root))} "
        f"RKMON_WORKERS={shlex.quote(paths.get(cfg, 'workers'))} "
        f"python3 -B {shlex.quote(str(paths.PACKAGE / 'tools' / 'quality_run.py'))} {n}\n"
    )


def quality_start(cfg: dict, st: dict, store, pipe: dict) -> list[str]:
    """Start the review's code quality pass on the quality binding once its PR is staged (the draft stage's staging;
    the local worker's data and merged tree), when the review gets one (quality_runs), unless one started or the review
    is out of date; [] when nothing starts.
    """
    n = pipe["pr"]
    rec = st["prs"].get(str(n))
    if (
        pipe.get(QUALITY) is not None
        or not quality_runs(cfg, st, pipe)
        or not rec
        or rec.get("status") != "open"
        or not same_commit(rec["head"], pipe["head"])
    ):
        return []
    ws = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}"
    # The merged tree, once the worker that stages it is done, may be gone (a restart empties the RAM disk): the pass's
    # job makes it again first (mkwt.py), never while the worker may still be making it.
    worker_done = (
        jobs.status(store, pipe.get("worker")) != "running"
        if pipe.get("worker")
        else True
    )
    remake = worker_done and bool(trees_gone(cfg, n))
    if draft_stage.of_pipe(pipe) is draft_stage.LOCAL:
        ready = (ws / "pr" / "meta.md").is_file() and (
            pathlib.Path(paths.worktree(cfg, f"wt{n}m")).exists() or remake
        )
    else:
        ready = draft_stage.staged(cfg, n)
    if not ready:
        return []
    wall = float(bindings.drafter_settings(cfg)["quality_wall_min"])
    script = (
        remerge_script(cfg, n, rec.get("base")) if remake else ""
    ) + quality_script(cfg, n)
    job = jobs.start(store, QUALITY, f"pr{n}", script, timeout_min=int(wall) + 15)
    pipe[QUALITY] = job
    store.journal(
        "review",
        f"#{n} code quality pass started on {quality_binding(cfg)['binding']} ({job['id']})",
    )
    return [
        f"REVIEW #{n}: code quality pass started on {quality_binding(cfg)['binding']} ({job['id']})"
    ]


def quality_wait(cfg: dict, store, pipe: dict) -> str:
    """ "running" while the review's code quality pass runs (REVIEW_READY waits for it), else ""; a pass past its
    time is stopped."""
    job = pipe.get(QUALITY)
    if not job:
        return ""
    s = jobs.status(store, job)
    if s == "timeout":
        jobs.kill(store, job)
    return "running" if s == "running" else ""


def quality_note(cfg: dict, st: dict, store, pipe: dict) -> str:
    """What REVIEW_READY says of the review's code quality pass: its quality.md for the verifier, or that the
    orchestrator's code quality reviewer runs it; "" for a review that gets no pass on the binding (quality_runs). A
    review too large for the autopilot has its REVIEW_READY say what becomes of the pass (in_depth_step).
    """
    if not quality_runs(cfg, st, pipe):
        return ""
    b = quality_binding(cfg)
    folder = quality_folder(cfg, pipe["pr"])
    try:
        run = json.loads((folder / "quality_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {}
    if (folder / "quality.md").is_file() and same_commit(
        str(run.get("head") or ""), pipe["head"]
    ):
        return (
            f" Code quality pass ({b['binding']}): {folder}/quality.md, done: `rk review-prep {pipe['pr']}` puts "
            "it in the review directory; don't dispatch the code quality reviewer: the verifier folds it in."
        )
    state = jobs.status(store, pipe[QUALITY]) if pipe.get(QUALITY) else "not started"
    return (
        f" Code quality pass ({b['binding']}): {state}, no quality.md at this head: dispatch the code quality "
        "reviewer as usual."
    )


# --- the review's first passes (models.ci_triage, models.quality_draft: rkmon_lib/first_pass.py) ------------------------
# With either role bound to a model, the driver runs `rk first-pass N --head SHA` once the PR is staged, beside the
# draft and the checks; `rk review-prep` copies its leads for the head into the review directory. REVIEW_READY never
# waits for it: a first pass not done by then is a miss, and the owners work without it. A merge-only head gets none.
FIRST_PASS = "first_pass"


def first_pass_start(cfg: dict, st: dict, store, pipe: dict) -> list[str]:
    """Start the pipe's first-pass job once its PR is staged, unless one started, its head is merge-only, neither role
    is bound, or the pipe is out of date; [] when nothing starts."""
    from . import first_pass

    n = pipe["pr"]
    rec = st["prs"].get(str(n))
    roles = first_pass.driver_roles(cfg)
    if (
        pipe.get(FIRST_PASS) is not None
        or pipe.get("merge_only")
        or not roles
        or not rec
        or rec.get("status") != "open"
        or not same_commit(rec["head"], pipe["head"])
    ):
        return []
    ws = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}"
    if not ((ws / "pr" / "meta.md").is_file() or draft_stage.staged(cfg, n)):
        return []
    wall = first_pass.caps(cfg)["wall_s"] / 60
    job = jobs.start(
        store,
        FIRST_PASS,
        f"pr{n}",
        first_pass.script(cfg, n, pipe["head"]),
        timeout_min=int(wall * len(roles)) + 5,
    )
    pipe[FIRST_PASS] = job
    store.journal(
        "review", f"#{n} first pass started: {', '.join(roles)} ({job['id']})"
    )
    return [f"REVIEW #{n}: first pass started: {', '.join(roles)} ({job['id']})"]


def first_pass_running(st: dict, store, n: int) -> bool:
    """PR n's first-pass job still runs (a pipe's; a job past its time is stopped)."""
    for p in pipes(st):
        job = p.get(FIRST_PASS) if p["pr"] == n else None
        if job:
            s = jobs.status(store, job)
            if s == "timeout":
                jobs.kill(store, job)
            if s == "running":
                return True
    return False


def mirror_note(cfg: dict, st: dict, pipe: dict) -> str:
    """REVIEW_READY's note in a mirror (rkmon_lib/mirror.py): "" outside one."""
    from . import mirror

    return mirror.ready_note(
        cfg, st, pipe["pr"], pipe["head"], quality_runs(cfg, st, pipe)
    )


def _fill(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    """Start reviews while a pipe is free: up to the pipes in effect (config.pipes). Nothing starts while the monitor
    is stopped or draining, or while the reviewer model is refused (it failed the start's test chat or tool-call probe:
    config.probe_failure; review-drafts says so).
    A follow-up that is a rebase only is recorded at once (REVIEW_SKIP), and a PR closed on GitHub or with a newer live
    head that has not settled starts nothing (TRY_NEXT); then the next item is tried, each PR once, a bounded number of
    times (a loop, not recursion: a stale plan can re-pick the same PR without end). The review queue's entries come
    before the plan's items (first_items); the entries of PRs closed or merged leave it first, and the approvals whose
    round is over end (prune_approved). In a mirror the items are the live monitor's feed lines alone, each started at
    its head (rkmon_lib/mirror.py), and MIRROR_DONE says once that the reviews to write are written.
    """
    from . import mirror
    from .store import halted

    local = (
        draft_stage.for_config(cfg) is draft_stage.LOCAL
    )  # the reviewer model drafts (else a binding's stage)
    if halted(st) or (local and config.probe_failure(config.worker(cfg))):
        return []
    mirroring = mirror.on(cfg)
    if mirroring:
        lines = mirror.done_line(cfg, st)
    else:
        prune_first(st, store)
        prune_approved(st)
        lines = []

    def pick(exclude: set) -> dict | None:
        if mirroring:
            return next(
                (i for i in mirror.items(cfg, st, store) if i["pr"] not in exclude),
                None,
            )
        return pick_next(cfg, st, store, first_items(st) + plan, exclude=exclude)

    effective = pipes_in_effect(
        cfg, store
    )  # a lower setting stops no pipe: none starts until fewer run
    taken, tried = set(), set()
    if interrupted(st):  # the reviews a stop or restart interrupted go first
        got, taken = resume_interrupted(
            cfg, st, store, max(0, effective - len({p["pr"] for p in pipes(st)}))
        )
        lines += got
        tried |= taken
    for _ in range(12):
        if len({p["pr"] for p in pipes(st)} | taken) >= effective:
            break
        h = heavy_building(cfg, st, store) if local else None
        if h:
            if pick(taken | tried):
                lines.append(
                    f"REVIEW_WAIT: the next worker starts once {h['id']} is past its first {HEAVY_BUILD_MIN} "
                    "minutes"
                )
            break
        nxt = pick(taken | tried)
        if not nxt:
            lines += [] if mirroring else manual_lines(cfg, st, store, plan)
            break
        tried.add(nxt["pr"])  # each PR once per advance
        ok, why = scaler.room(
            cfg, st, store, "pipe", nxt["pr"]
        )  # the RAM disk's room and the memory, when it is on
        if not ok:
            lines.append(
                f"REVIEW_WAIT #{nxt['pr']}: it waits for room on this machine: {why}"
            )
            break
        if mirroring:
            got = mirror.start(cfg, st, store, nxt)
        else:
            got = start_review(
                cfg, st, store, nxt["pr"], nxt["kind"], again=nxt.get("again", False)
            )
            first_started(st, store, nxt["pr"], got)
        lines += got
        if got and got[-1].startswith("REVIEW_START"):
            taken.add(nxt["pr"])
        elif not (
            got and got[-1].startswith(TRY_NEXT + mirror.TRY_NEXT)
        ):  # REVIEW_WAIT: no answer from GitHub
            break
    return lines


def _record_checks_gates(cfg: dict, rec: dict, n: int) -> None:
    """A review's finished checks: G5-G7 for their head, our probes' results there (rec["probes_seen"]: what the next
    merge-only head's probes are compared with, merge_only_flags), and the items of their threads, title, tests,
    installed and prnumbers checks (rec["peer_seen"]: peer_carry)."""
    status, dev_sha, head_sha = read_checks_status(
        f"{paths.get(cfg, 'workers')}/pr{n}/checks"
    )
    if status and dev_sha:
        from .queue import rocke_tree

        rec["gates"] = {
            "head": head_sha or rec["head"],
            "develop": dev_sha,
            "develop_tree": rocke_tree(cfg, dev_sha),
            "results": gates_from_status(status),
            "at": now(),
            "source": "review-checks",
        }
    probes = probe_results(_checks_text(cfg, n))
    if head_sha and probes is not None:
        rec["probes_seen"] = {"head": head_sha, "results": probes}
    peer_carry.record(rec, f"{paths.get(cfg, 'workers')}/pr{n}/checks", head_sha)


def recheck(cfg: dict, st: dict, store, n: int) -> list[str]:
    """Re-run the checks of a draft awaiting verification, at the same head: after a fix to the checks themselves.
    `finish_rechecks` records
    the result when the job ends."""
    r = next((x for x in st.get("review_ready", []) if x["pr"] == n), None)
    rec = st["prs"].get(str(n))
    if not r or not rec or not same_commit(rec["head"], r["head"]):
        return [
            f"RECHECK #{n}: refused: no draft awaiting verification at its current head"
        ]
    heavy = jobs.heavy_busy(cfg, st, store)
    if heavy:
        return [
            f"RECHECK #{n}: all parallel builds are busy ({heavy['id']} among them); try again when one is free"
        ]
    ok, why = scaler.room(cfg, st, store, "heavy", n)
    if not ok:
        return [f"RECHECK #{n}: not now: {why}; try again later"]
    made = ensure_merge_tree(cfg, rec, n)  # the plan below reads the merged tree
    if trees_gone(cfg, n):
        return made + [f"RECHECK #{n}: refused: its merged tree is not there"]
    tree = paths.worktree(cfg, f"wt{n}m")
    fams = emitter_families(tree, paths.repository_folder(cfg))
    pc = plan_checks(cfg, rec, _changed_files(cfg, rec), fams, source=tree)
    pc["_dirs"] = {f: fams[f] for f in pc["families"]}
    since = (rec.get("ours") or {}).get("at") if r["kind"] == "follow-up" else None
    # Merge with today's develop first, as a gate job does: develop may have moved since the draft's merge, and the old
    # merge worktree checked against a newer develop skips everything.
    remerge = remerge_script(cfg, n, rec.get("base"))
    reviewed = (rec.get("ours") or {}).get("head") if r["kind"] == "follow-up" else None
    job = jobs.start(
        store,
        "checks",
        f"pr{n}",
        remerge
        + checks_script(
            cfg,
            n,
            rec["head"],
            r["kind"],
            pc,
            since,
            reviewed=reviewed,
            base=rec.get("base"),
        ),
        timeout_min=240,
    )
    jobs.take_heavy(st, job, store)
    st.setdefault("rechecks", {})[str(n)] = job
    store.journal("review", f"#{n} checks re-run at {rec['head'][:10]} ({job['id']})")
    return made + [
        f"RECHECK #{n}: checks started ({len(pc['families'])} families, build={pc['build']}); "
        f"output in {paths.get(cfg, 'workers')}/pr{n}/checks"
    ]


def finish_rechecks(cfg: dict, st: dict, store) -> list[str]:
    out = []
    for k, job in list((st.get("rechecks") or {}).items()):
        if jobs.status(store, job) == "running":
            continue
        del st["rechecks"][k]
        rec = st["prs"].get(k)
        if rec:
            _record_checks_gates(cfg, rec, int(k))
        store.journal("review", f"#{k} checks re-run done ({jobs.status(store, job)})")
        out.append(
            f"RECHECK_DONE #{k}: checks {jobs.status(store, job)}; see "
            f"{paths.get(cfg, 'workers')}/pr{k}/checks/checks.md"
        )
    return out


# --- the change fingerprint (idle.py) ---------------------------------------------------------------------------------


@idle.timer("reviews")
def _pipeline_due(cfg: dict, st: dict, store):
    """When each parallel review's local worker leaves its first PREFILL_MIN minutes (its checks and the other heavy
    jobs may start then), and when the heavy jobs have all left their first HEAVY_BUILD_MIN minutes (the next worker may
    start then)."""
    for pipe in draft_stage.in_flight(st):
        started = (pipe.get("worker") or {}).get("started")
        if started and draft_stage.of_pipe(pipe).prefill_min > 0:
            yield f"worker prefill #{pipe.get('pr')}", idle.plus(started, PREFILL_MIN)
    starts = [h["started"] for h in jobs.heavy_holders(st) if h.get("started")]
    if starts:
        yield "build's first minutes", idle.plus(max(starts), HEAVY_BUILD_MIN)


@idle.part("review files")
def _review_files(cfg: dict, st: dict, store):
    """What the parallel reviews' checks wait on besides time: each one's merged tree, and its draft stage's
    `staged` file."""
    out = {}
    for pipe in draft_stage.in_flight(st):
        n = pipe.get("pr")
        if n and pipe.get("checks") is None:
            out[str(n)] = [
                pathlib.Path(paths.worktree(cfg, f"wt{n}m")).exists(),
                draft_stage.staged(cfg, n),
            ]
    return out or None
