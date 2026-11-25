# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Develop drift: a PR's own tests on its head merged with today's develop, for PRs heading for a merge.

A PR whose head never moves is checked once; a later develop change can still break its tests on today's develop,
while the board calls it closest to ready and its CI stays green on the old merge ref (lessons L-323).

Gate jobs (G5-G7) run only for candidates, PRs with their approvals in. A PR waiting for approval kept the test results
of its last review however far develop moved, and approvers read the board. The drift check closes that: whenever
develop's rocKE tree moves past the tree a PR's tests last ran on, the PR's test files run on its head merged with
develop and on develop alone (git archive of the two rocKE trees: no worktree, no build; tools/pytest_vs_base.py tells
the PR's failures from develop's), in a PR runner's slot of their own. A failure on the merge only fails G6 for that
head and develop tree (queue.gates_dynamic) and shows as a merge blocker before any approval (report.merge_blockers),
which also keeps the PR off the approval lists. The tick prints DRIFT_FAIL with the rocKE develop commits since the PR's
last test run that touch its areas: the likely cause.

One drift job at a time, a few PRs per job in merge-plan order, never in a reviewer-model worker's first minutes.

Like with like. Every tree a job runs (develop, the merge, the head) is the same extraction: the rocKE tree and what
its tests read around it (config `drift_extract`, `extract_paths`: such as the repository's top-level files, the
provider's own files and the folders that mark the repository's root). A test that
fails on the merge and not on develop runs again on the PR's head alone under that extraction; one that fails there
too is the head's own (a test new to develop that the extraction cannot serve, or the PR's own failure, which its
review checks report), never drift. A failure that is still a harness false positive is dismissed by hand (`rk
drift-dismiss N D --reason TEXT`, `dismiss`): G6 and the merge blockers stop counting that run, and
`drift_dismissed.jsonl` in the state folder keeps the record."""

from __future__ import annotations

import pathlib
import re
import shlex

from . import first_pass, gitops, jobs, nogpu, paths, prcode, registry, review
from .store import halted, now

PER_PR_MIN = 15  # pytest time box per PR
PR_REF = re.compile(r"\(#(\d+)\)")
DISMISSED_FILE = "drift_dismissed.jsonl"


def extract_paths(cfg: dict) -> tuple[list[str], list[str]]:
    """(whole paths, folders whose files only) a drift run extracts: the rocKE tree first, then config drift_extract's
    paths, each a path in the repository: "{provider}" in one is the rocKE tree's parent folder, and one ending in "/*"
    (or "*" alone) takes that folder's files only."""
    K = paths.repository_folder(cfg)
    provider = K.rsplit("/", 1)[0] if "/" in K else ""
    whole, files_of = [K], []
    for p in cfg.get("drift_extract") or ():
        p = p.replace("{provider}", provider).strip("/")
        if p == "*" or p.endswith("/*"):
            files_of.append(p[:-1].rstrip("/"))
        elif p and p not in whole:
            whole.append(p)
    return whole, files_of


def counts(d: dict, head: str, develop: str, tree: str | None) -> bool:
    """Drift record `d` fails G6 for `head` on this develop (or its rocKE tree): it failed there and no one dismissed
    it as a harness false positive."""
    return (
        d.get("status") == "fail"
        and not d.get("dismissed")
        and d.get("head") == head
        and (d.get("develop") == develop or bool(tree and d.get("tree") == tree))
    )


def _fresh(rec: dict, key: str, develop: str, tree: str | None) -> bool:
    """rec[key] ("gates" or "drift") was recorded for this head on this develop, or on the same rocKE tree."""
    r = rec.get(key) or {}
    return r.get("head") == rec.get("head") and (
        r.get("develop") == develop
        or bool(tree and (r.get("develop_tree") or r.get("tree")) == tree)
    )


def fresh_fail(st: dict, rec: dict) -> str | None:
    """The drift failure recorded for this PR's head on today's develop (or its rocKE tree), else None."""
    dev = st.get("develop") or {}
    d = rec.get("drift") or {}
    if counts(d, rec.get("head") or "", dev.get("sha") or "", dev.get("rocke_tree")):
        return d.get("detail") or "tests fail on the merge with develop"
    return None


def targets(cfg: dict, st: dict, rows: list[dict]) -> list[dict]:
    """PRs whose tests should run on today's develop: our review is clean at the head (G1), its stack is in (G8), it
    merges cleanly (G4), and neither a gate job nor a drift check has run its tests on this develop's rocKE tree. Merge-
    plan order first (what lands next), then develop fixes, then the smallest."""
    dev = st["develop"]["sha"]
    tree = st["develop"].get("rocke_tree")
    out = []
    for r in rows:
        rec = st["prs"].get(str(r["n"])) or {}
        g = r.get("gates") or {}
        if rec.get("status") != "open" or rec.get("draft"):
            continue
        # G8 also fails every PR while the queue is frozen; the drift check still runs then (it compares with develop
        # alone, and develop's own failures do not count), when develop moves the most.
        g8 = g.get("G8") or ["?", ""]
        g8_ok = g8[0] == "pass" or str(g8[1]).startswith("queue frozen")
        if (
            (g.get("G1") or ["?"])[0] != "pass"
            or not g8_ok
            or (g.get("G4") or ["?"])[0] == "fail"
        ):
            continue
        if _fresh(rec, "gates", dev, tree) or _fresh(rec, "drift", dev, tree):
            continue
        out.append(r)
    out.sort(
        key=lambda r: (
            r.get("plan_rank") is None,
            r.get("plan_rank") or 0,
            r.get("tier", 2),
            r.get("size", 0),
            r["n"],
        )
    )
    return out


def _tests(cfg: dict, rec: dict, files: list[str]) -> list[str]:
    """The PR's test files to run (the checks' selection: companion contract tests, then its own test files), as paths
    under the rocKE tree."""
    return review.plan_checks(cfg, rec, files, {})["tests"]


def script(cfg: dict, job_dir: str, develop: str, items: list[dict]) -> str:
    """One drift job: develop's tree once, then per PR its merge with develop and its head alone, its tests on the merge
    and on develop, the failures develop does not share again on the head alone (pytest_vs_base.py's RKMON_HEAD_ROOT),
    a result line. Every tree is the same extraction (extract_paths), in a tree PR code reads (`$D/wtdrift`); each PR's
    tests run in a PR runner's slot of their own, and their results are copied into the job's folder, in `pr<N>/tests`.
    """
    K = paths.repository_folder(cfg)
    whole, files_of = extract_paths(cfg)
    # xt REV DIR: REV's extraction under DIR (the paths it has: a missing one would stop git archive); true when the
    # rocKE tree is there.
    xt = (
        "xt() { { git -C $R0 ls-tree --name-only $1 -- "
        + " ".join(shlex.quote(p) for p in whole)
        + "; "
        + (
            f"for f in {' '.join(shlex.quote(p) for p in files_of)}; do git -C $R0 ls-tree $1 ${{f:+-- $f/}} "
            "| awk -F'\\t' '$1 ~ / blob / {print $2}'; done; "
            if files_of
            else ""
        )
        + '} | sort -u | xargs -r -d \'\\n\' git -C $R0 archive $1 -- | tar -x -C "$2" && [ -d "$2/$K" ]; }'
    )
    # cmake and PyYAML, as the review checks' tests have them, in each PR's slot
    tools = "".join(f"{ln}; " for ln in review.tool_env_lines(cfg, "$SI/pyshim"))
    L = [
        paths.shell_vars(cfg),
        prcode.SHELL,  # prslot, prc, prr, prtree: each PR's tests in a PR runner's slot (rkmon_lib/prcode.py)
        f'K={K}; J={shlex.quote(job_dir)}; DEV={develop}; TD="$D/wtdrift"',
        'export PATH=$VB:$PATH TMPDIR="$TP" ROCKE_LLVM_FLAVOR=llvm22',
        nogpu.SHELL,
        xt,
        'mkdir -p "$J" "$TP"; rm -rf "$TD"; prtree "$TD" && mkdir "$TD/dev" || '
        "{ echo 'no folder for the trees' > \"$J/error\"; exit 1; }",
        'xt $DEV "$TD/dev" || { echo \'develop tree not extracted\' > "$J/error"; exit 1; }',
    ]
    for it in items:
        n, head, tests = it["n"], it["head"], it["tests"]
        o = f"$J/pr{n}"
        L += [
            f'O="{o}"; rm -rf "$O"; mkdir -p "$O"; echo {head} > "$O/head"',
            f"git -C $R0 fetch -q --filter=blob:none origin +refs/pull/{n}/head:refs/remotes/origin/pr{n} 2>/dev/null",
        ]
        if not tests:
            L.append("echo 'SKIP no test file to run' > \"$O/result\"")
            continue
        files = " ".join(shlex.quote(t) for t in tests)
        # A test file the PR deletes or renames is in its changed files too; pytest stops the whole batch on a missing
        # path (rc 4, which would read as a drift failure), so only files the merge has run, as the review checks do.
        # The head alone (HR): a head that does not extract runs without it, every failure develop lacks then counts.
        L += [
            "if T=$(git -C $R0 merge-tree --write-tree $DEV "
            + head
            + " 2>/dev/null); then "
            'rm -rf "$TD/mrg" "$TD/htree"; mkdir "$TD/mrg" && xt $T "$TD/mrg" && { '
            f'TS=\'\'; for t in {files}; do [ -e "$TD/mrg/$K/$t" ] && TS="$TS $t"; done; '
            'if [ -z "$(echo $TS)" ]; then echo \'SKIP no test file in the merge\' > "$O/result"; else '
            f'HR="$TD/htree/$K"; mkdir "$TD/htree" && xt {head} "$TD/htree" || HR=\'\'; '
            f"(prslot {n} || exit 3; {tools}RKMON_HEAD_ROOT=$HR RKMON_HEAD_PYTHONPATH=$HR/platform/python:$HR/library "
            f'{prcode.RESULTS_VAR}="$O/tests" prc nice -n 10 timeout {PER_PR_MIN * 60} python3 -I -B '
            '$M/tools/pytest_vs_base.py $SO/tests "$TD/mrg/$K" "$TD/dev/$K" '
            '"$TD/mrg/$K/platform/python:$TD/mrg/$K/library" "$TD/dev/$K/platform/python:$TD/dev/$K/library" '
            '$TS > "$O/report.txt" 2>&1; prr tests "$O/tests"; prfree) || '
            "echo 'ERROR no slot of a PR runner could be taken' > \"$O/report.txt\"; "
            'grep -E \'^(TESTS|ERROR) \' "$O/report.txt" | tail -1 > "$O/result"; '
            '[ -s "$O/result" ] || echo \'ERROR no result (time box?)\' > "$O/result"; fi; }; '
            'rm -rf "$TD/mrg" "$TD/htree"; '
            "else echo 'CONFLICT merge with develop' > \"$O/result\"; fi"
        ]
    L.append('rm -rf "$TD"; echo done > "$J/done"')
    return "\n".join(L) + "\n"


def start(cfg: dict, st: dict, store, rows: list[dict]) -> list[str]:
    """Start a drift job for the next PRs that need one, when none runs and no reviewer-model worker is in its
    first minutes."""
    js = st.setdefault("jobs", {})
    cur = js.get("drift")
    if cur and jobs.status(store, cur) == "running":
        return []
    if review.worker_prefill(cfg, st) or halted(st):
        return []
    todo = targets(cfg, st, rows)[: registry.value(cfg, "drift_batch")]
    if not todo:
        return []
    dev = st["develop"]["sha"]
    items = []
    for r in todo:
        rec = st["prs"][str(r["n"])]
        try:
            files = review._changed_files(cfg, rec)
        except gitops.GitError as exc:
            items.append(
                {"n": r["n"], "head": rec["head"], "tests": [], "error": str(exc)[:120]}
            )
            continue
        items.append(
            {"n": r["n"], "head": rec["head"], "tests": _tests(cfg, rec, files)}
        )
    name = f"d{dev[:8]}"
    job_dir = f"{paths.get(cfg, 'workers')}/drift/{name}"
    job = jobs.start(
        store,
        "drift",
        name,
        script(cfg, job_dir, dev, items),
        timeout_min=PER_PR_MIN * len(items) + 15,
    )
    js["drift"] = job
    st["drift_job"] = {
        "job": job,
        "dir": job_dir,
        "develop": dev,
        "tree": st["develop"].get("rocke_tree"),
        "items": [
            {"n": i["n"], "head": i["head"], "tests": len(i["tests"])} for i in items
        ],
        "started": now(),
    }
    return [
        f"DRIFT_START on develop {dev[:10]}: " + " ".join(f"#{i['n']}" for i in items)
    ]


def _parse(line: str) -> tuple[str, str]:
    """A result line: 'TESTS pass|fail <detail>', 'SKIP …', 'CONFLICT …', 'ERROR …'."""
    line = line.strip()
    if line.startswith("TESTS "):
        parts = line.split(" ", 2)
        return (parts[1] if len(parts) > 1 else "error"), (
            parts[2] if len(parts) > 2 else ""
        )
    for word in ("SKIP", "CONFLICT", "ERROR"):
        if line.startswith(word):
            return word.lower(), line[len(word) :].strip()
    return "error", line[:160] or "no result"


def suspects(
    cfg: dict, since: str | None, develop: str, rec: dict, failed: list[str]
) -> list[int]:
    """Develop's commits between the PR's last test run (`since`) and `develop` that touch what its tests depend on,
    closest first: the PR's own files and its failing test files; else their directories; else rocKE's top two levels
    there. Their PR numbers, newest first. The PR's own files name the cause far more often than rocKE's top levels,
    which name every later PR."""
    if not since or since == develop:
        return []
    K = paths.repository_folder(cfg)
    own = {
        f
        for f in (rec.get("files_cache") or {}).get("files") or []
        if f.startswith(K + "/")
    }
    own |= {
        f"{K}/{t.split('::')[0]}" for t in failed if t.split("::")[0].endswith(".py")
    }
    tiers = [
        sorted(own)[:40],
        sorted({f.rsplit("/", 1)[0] for f in own if "/" in f})[:20],
        sorted({"/".join(f.split("/")[: len(K.split("/")) + 2]) for f in own})[:12],
    ]
    for tier in tiers:
        if not tier:
            continue
        out = gitops.git(
            cfg["paths"]["repo"],
            "log",
            "--format=%s",
            f"{since}..{develop}",
            "--",
            *tier,
            check=False,
        )
        nums = []
        for line in out.splitlines():
            m = PR_REF.search(line)
            if m and int(m.group(1)) not in nums:
                nums.append(int(m.group(1)))
        if nums:
            return nums[:6]
    return []


def collect(cfg: dict, st: dict, store) -> list[str]:
    """Read a finished drift job: record each PR's result (rec["drift"]) and print DRIFT_FAIL for the failures."""
    dj = st.get("drift_job")
    if not dj:
        return []
    status = jobs.status(store, dj["job"])
    if status == "running":
        return []
    st["drift_job"] = None
    jdir = pathlib.Path(dj["dir"])
    if (jdir / "error").exists():
        return [
            f"(drift check on develop {dj['develop'][:10]} failed: {(jdir / 'error').read_text().strip()[:120]})"
        ]
    lines = []
    for it in dj["items"]:
        n = it["n"]
        rec = st["prs"].get(str(n))
        if not rec:
            continue
        rf = jdir / f"pr{n}" / "result"
        kind, detail = _parse(rf.read_text() if rf.exists() else "")
        failed = []
        ff = jdir / f"pr{n}" / "tests" / "tests_failed.txt"
        if kind == "fail" and ff.exists():
            failed = [x for x in ff.read_text().splitlines() if x.strip()][:20]
        prev = (
            (rec.get("drift") or {})
            if (rec.get("drift") or {}).get("head") == it["head"]
            else {}
        )
        last = (
            (rec.get("gates") or {}).get("develop")
            if (rec.get("gates") or {}).get("head") == it["head"]
            else None
        ) or prev.get("develop")
        rec["drift"] = {
            "head": it["head"],
            "develop": dj["develop"],
            "tree": dj.get("tree"),
            "status": kind,
            "detail": detail[:300],
            "failed": failed,
            "since": last,
            "at": now(),
        }
        if kind == "fail":
            sus = suspects(cfg, last, dj["develop"], rec, failed)
            rec["drift"]["suspects"] = sus
            store.journal(
                "drift",
                f"#{n} at {it['head'][:10]}: tests fail on develop {dj['develop'][:10]}: "
                f"{detail[:160]}",
            )
            lines.append(
                f"DRIFT_FAIL #{n} at develop {dj['develop'][:10]}: {detail[:200]}"
                + (
                    f" (tests last ran on develop {last[:10]}"
                    if last
                    else " (no earlier run recorded"
                )
                + (
                    f"; rocKE changes since in its areas: {', '.join('#%d' % s for s in sus)})"
                    if sus
                    else ")"
                )
                + f": tell the author (`rk notice {n} FILE --kind drift --key {dj['develop'][:10]}`), or, a "
                f"harness false positive, `rk drift-dismiss {n} {dj['develop'][:10]} --reason TEXT`"
                + first_pass.notice_hint(cfg, n, "drift")
            )
        elif kind == "error":
            lines.append(f"(drift check #{n}: {detail[:160]})")
    ok = [
        str(it["n"])
        for it in dj["items"]
        if (st["prs"].get(str(it["n"])) or {}).get("drift", {}).get("status") == "pass"
    ]
    if ok:
        lines.append(
            f"DRIFT ok on develop {dj['develop'][:10]}: "
            + " ".join(f"#{n}" for n in ok)
        )
    return lines


def dismiss(st: dict, store, n: int, develop: str, reason: str, by: str) -> str:
    """`rk drift-dismiss N D --reason TEXT`: PR n's drift failure on develop D (a commit prefix of 7 or more) is a
    harness false positive. Local only: the run's record is marked dismissed (G6 and the merge blockers stop counting
    it; a later run on another develop counts again), and the dismissal is appended to DISMISSED_FILE in the state
    folder. ValueError when PR n has no failed drift run on develop D or no reason is given.
    """
    rec = (st.get("prs") or {}).get(str(n))
    if not rec:
        raise ValueError(f"#{n} is not in the state")
    reason = " ".join((reason or "").split())
    if not reason:
        raise ValueError("--reason TEXT: say why the failure is the harness's")
    d = rec.get("drift") or {}
    want = (develop or "").strip().lower()
    if (
        len(want) < 7
        or not re.fullmatch(r"[0-9a-f]+", want)
        or not (d.get("develop") or "").startswith(want)
    ):
        raise ValueError(
            f"#{n} has no drift run on develop {want[:10] or '-'} (its last: "
            f"{d.get('status') or 'none'} on develop {(d.get('develop') or '-')[:10]})"
        )
    if d.get("status") != "fail":
        raise ValueError(
            f"#{n}'s drift run on develop {d['develop'][:10]} did not fail ({d.get('status')})"
        )
    if d.get("dismissed"):
        return (
            f"#{n}: the drift failure at {d['head'][:10]} on develop {d['develop'][:10]} is already dismissed "
            f"({d['dismissed'].get('reason')})"
        )
    entry = {
        "at": now(),
        "pr": n,
        "head": d["head"],
        "develop": d["develop"],
        "tree": d.get("tree"),
        "detail": d.get("detail") or "",
        "failed": d.get("failed") or [],
        "reason": reason[:400],
        "by": by,
    }
    store.append_jsonl(DISMISSED_FILE, [entry])
    d["dismissed"] = {"at": entry["at"], "reason": entry["reason"], "by": by}
    store.journal(
        "drift",
        f"#{n} at {d['head'][:10]}: the drift failure on develop {d['develop'][:10]} dismissed as "
        f"a harness false positive: {entry['reason'][:160]}",
    )
    return (
        f"#{n}: the drift failure at {d['head'][:10]} on develop {d['develop'][:10]} dismissed as a harness false "
        "positive; G6 does not count it"
    )
