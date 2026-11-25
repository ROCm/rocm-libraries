# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The develop guard ("no forgiveness" while the monitor is ON), while features.merge_queue.develop_guard is on, in
prototype and live alike.

Every develop SHA that brings rocKE changes gets a sanity run. Runs are coalesced, so a burst of landings is tested
once at its newest SHA. Each run uses a dedicated worktree (`wtsanity`) and covers:
- the native build and ctest;
- the representative golden at three flavors;
- the two differential-golden CTest commands;
- byte identity for the families the landed commits touched;
- the ABI history invariant.

Each section writes a status line, so the verdict is machine-read.

**When a run is red** (or develop CI turns red in a rocKE job):
1. An incident opens and the queue freezes.
2. The culprit is found among the first-parent merges since the last green SHA: the only candidate, or the first
   failing one in a bisect job that re-runs only the failing sections.
3. The tick tells the orchestrator. While the merge queue is live, the orchestrator files or updates the develop
   issue on GitHub (`rk issue-file`, `rk issue-comment`) and comments on the culprit (incident_procedure.md); in
   prototype the break is reported to the operator only.
4. After `revert_offer_after_min` (2 h), the tick asks for a revert offer.
5. A later green run closes the incident and unfreezes the queue.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import sys

from . import config, gitops, idle, jobs, nogpu, paths, prcode, registry
from . import state_store, whole_file
from .review import (
    DG_FLAVORS,
    FLAVORS_FN,
    MAX_HELPER_FAMILIES,
    U8_FN,
    helper_families,
    touched_families,
    worker_prefill,
)
from .store import halted
from .store import minutes_since, now

MERGED_PR = re.compile(r"\(#(\d+)\)\s*$")
INCIDENTS_FILE = "incidents.json"  # in state/


# --- incidents (state/incidents.json is shared with the weekly statistics) -------------------------------------------


def incidents(store) -> list[dict]:
    """The incidents. A file left empty or unreadable (a write cut short) is recovered, with a warning: from the copy
    the last save kept (incidents.json.prev), else as no incidents; the bad file is kept beside it, and the recovered
    list is saved, so the tick goes on."""
    p = store.dir / INCIDENTS_FILE
    if not p.exists():
        return []
    raw = p.read_text()
    try:
        inc = json.loads(raw)
        if isinstance(inc, list):
            return inc
    except ValueError:
        pass
    prev = p.with_name(INCIDENTS_FILE + ".prev")
    try:
        got = json.loads(prev.read_text())
        got = got if isinstance(got, list) else []
        how = f"recovered from {prev.name} ({len(got)} incidents)"
    except (OSError, ValueError):
        got, how = [], "no good copy: starting from no incidents"
    if raw.strip():
        bad = p.with_name(f"{INCIDENTS_FILE}.bad-{now().replace(':', '')}")
        bad.write_text(raw)
        how += f"; the bad file is kept as {bad.name}"
    save_incidents(store, got, keep_prev=False)
    msg = f"{INCIDENTS_FILE} was {'unreadable' if raw.strip() else 'empty'}: {how}"
    store.journal("INCIDENT", msg)
    print(f"warning: {msg}", file=sys.stderr)
    return got


def save_incidents(store, inc: list[dict], keep_prev: bool = True) -> None:
    """The incidents INC written whole and durable (state_store.write_file); the file they replace, when it reads as
    JSON, stays as incidents.json.prev, written whole too (whole_file), at the mode of the files the monitor and the
    writer share (worker/perms.py)."""
    p = store.dir / INCIDENTS_FILE
    if keep_prev and p.exists():
        try:
            data = p.read_bytes()
            json.loads(data)
            whole_file.write(
                p.with_name(INCIDENTS_FILE + ".prev"),
                data,
                paths.worker_module("perms").SHARED_FILE_MODE,
            )
        except (OSError, ValueError):
            pass
    state_store.write_file(p, inc)


def _trees(cfg: dict, name: str) -> list[str]:
    """A job's worktree and its build folder (build_native.py builds `<build dir>/<worktree name>`)."""
    out = [paths.worktree(cfg, name)]
    try:
        out.append(f"{paths.get(cfg, 'build')}/{name}")
    except paths.PathsError:
        pass
    return out


# An incident note or close by hand makes the next tick full (idle.py).
idle.file_part("incidents", lambda store: store.dir / INCIDENTS_FILE)


def revert_offer_min(cfg: dict) -> int:
    """`revert_offer_after_min`: the minutes an open incident waits for the revert offer."""
    return registry.value(cfg, "revert_offer_after_min")


@idle.timer("incident age")
def _incident_due(cfg: dict, st: dict, store):
    """When each open incident reaches `revert_offer_after_min`: the revert offer, or the note it has no culprit."""
    after = revert_offer_min(cfg)
    for inc in incidents(store):
        if inc.get("status") == "open" and inc.get("start"):
            yield f"incident {inc.get('id')}", idle.plus(inc["start"], after)


def _kinds(signals: list[str]) -> set[str]:
    """An incident's kinds of signal: develop CI (a red rocKE job) or our sanity run. Each clears only on its own green."""
    return {"ci" if s.startswith("develop CI:") else "sanity" for s in signals}


def open_incident(
    store,
    st: dict,
    sha: str,
    signals: list[str],
    last_green: str | None,
    candidates: list,
    ci_jobs: list[str] | None = None,
) -> dict:
    """Open a develop-break incident, or fold new signals into an open one. `ci_jobs` are the failed rocKE CI jobs of
    a develop CI signal: the incident's CI side clears only when those jobs pass again (ci_cleared).

    A red with landed candidates is its own break while the open incidents carry none (folded into an incident with no
    candidates, the bisect would have nothing to run): it opens a second incident. Otherwise new signals fold into the newest open incident that fits.
    """
    inc = incidents(store)
    opens = [i for i in inc if i.get("status") == "open"]
    if candidates:
        cur = next(
            (i for i in reversed(opens) if i.get("candidates") or i.get("culprit")),
            None,
        )
    else:
        cur = opens[-1] if opens else None
    if cur:
        cur["signals"] = sorted(set(cur.get("signals", [])) | set(signals))
        cur["red_shas"] = sorted(set(cur.get("red_shas", [])) | {sha[:12]})
        cur["ci_jobs"] = sorted(set(cur.get("ci_jobs") or []) | set(ci_jobs or []))
        known = {n for _c, n in cur.get("candidates") or []}
        cur["candidates"] = list(cur.get("candidates") or []) + [
            list(c) for c in candidates if c[1] not in known
        ]
        for kind in _kinds(
            signals
        ):  # a new red voids an earlier green of the same kind
            (cur.get("green") or {}).pop(kind, None)
        save_incidents(store, inc)
        return cur
    day = now()[:10]
    rec = {
        "id": f"INC-{day}-{len(inc) + 1}",
        "status": "open",
        "start": now(),
        "end": None,
        "red_shas": [sha[:12]],
        "last_green": (last_green or "")[:12],
        "signals": signals,
        "candidates": candidates,
        "culprit": None,
        "contributing": [],
        "fixed_by": None,
        "jira": None,
        "summary": "; ".join(signals)[:300],
        "recorded": now(),
        "actions": [],
        "ci_jobs": sorted(ci_jobs or []),
    }
    inc.append(rec)
    save_incidents(store, inc)
    st["frozen"] = {
        "since": now(),
        "reason": f"develop red ({rec['id']})",
        "incident": rec["id"],
    }
    store.journal("INCIDENT", f"open {rec['id']} at {sha[:10]}: {rec['summary']}")
    return rec


def stale_red(cfg: dict, st: dict, store, sha: str, jobs_red: list[str]) -> str | None:
    """Why a develop red is old news: its commit predates the merged fix of an incident whose failing jobs it repeats
    (develop runs that finish after the fix landed must neither keep the incident open nor offer a revert). None when
    it is news."""
    repo = cfg["paths"]["repo"]
    for inc in reversed(incidents(store)):
        fix = inc.get("fixed_by")
        msha = (st["prs"].get(str(fix)) or {}).get("merge_sha") if fix else None
        if not msha:
            continue
        known = set(inc.get("ci_jobs") or [])
        if known and not set(jobs_red) <= known:
            continue  # a job that incident never saw fail: news
        if sha[:10] != msha[:10] and gitops.is_ancestor(repo, sha, msha):
            return f"its commit predates #{fix}, the fix of {inc['id']} (merged as {msha[:10]})"
    return None


def ci_cleared(st: dict, inc: dict) -> str | None:
    """The develop SHA whose CI shows the incident's failed rocKE jobs passing again: the red SHA itself after a re-run
    (a flake), or a later SHA that ran those jobs. A later SHA that is green only because path filters skipped the rocKE
    jobs proves nothing (a commit outside rocKE never closes a rocKE incident). A newer red stops
    the search. Incidents recorded without job names accept any green SHA."""
    from .snapshot import ci_verdict

    # Compared by a 10-char prefix: the DEVELOP_RED event carries 10 chars, the CI keys more (a full key never matches
    # its 10-char prefix, and an incident would close at an older green while its re-run still runs).
    jobs, reds = set(inc.get("ci_jobs") or []), {
        r[:10] for r in inc.get("red_shas") or []
    }
    for sha, ci in (st["develop"].get("ci") or {}).items():  # newest first
        verdict = ci_verdict(ci)
        if verdict == "green":
            passed = {
                j
                for r in (ci.get("runs") or {}).values()
                for j in r.get("rocke_passed") or []
            }
            if jobs <= passed:
                return sha
        elif verdict == "red":
            return None  # a rocKE job is red at or after this point
        if sha[:10] in reds:
            return None
    return None


def close_incident(
    store,
    st: dict,
    green_sha: str,
    fixed_by: int | None = None,
    kind: str | None = None,
    inc_id: str | None = None,
) -> dict | None:
    """Record a green. `kind` ("ci" or "sanity") names the green this is, and it counts for the open incidents with
    that kind of signal (or only `inc_id`). An incident closes only once every kind of signal it has is green again (a
    green sanity run never closes a CI-test incident while develop CI still runs). Without `kind` the
    first open incident (or `inc_id`) closes at once. Returns the first incident it closed.
    """
    inc = incidents(store)
    opens = [
        i
        for i in inc
        if i.get("status") == "open" and (inc_id is None or i.get("id") == inc_id)
    ]
    if kind:
        targets = [i for i in opens if kind in _kinds(i.get("signals") or [])]
    else:
        targets = opens[:1]
    closed = []
    for cur in targets:
        if kind:
            green = cur.setdefault("green", {})
            green[kind] = green_sha[:12]
            left = _kinds(cur.get("signals") or []) - set(green)
            if left:
                store.journal(
                    "INCIDENT",
                    f"{cur['id']}: {kind} green at {green_sha[:10]}; it stays open until "
                    f"{' and '.join(sorted(left))} is green too",
                )
                continue
        cur.update(status="closed", end=now(), green_sha=green_sha[:12])
        if fixed_by and not cur.get("fixed_by"):
            cur["fixed_by"] = fixed_by
        store.journal("INCIDENT", f"closed {cur['id']} at green {green_sha[:10]}")
        closed.append(cur)
    save_incidents(store, inc)
    left_open = [i for i in inc if i.get("status") == "open"]
    if (st.get("frozen") or {}).get("incident") in {c["id"] for c in closed}:
        # The freeze stays while any develop break is open (two incidents may be open at once).
        st["frozen"] = (
            {
                "since": now(),
                "reason": f"develop red ({left_open[-1]['id']})",
                "incident": left_open[-1]["id"],
            }
            if left_open
            else None
        )
    return closed[0] if closed else None


# --- the sanity script -----------------------------------------------------------------------------------------------


def emitter_families_at(cfg: dict, sha: str) -> dict[str, str]:
    """family -> parity dir for every <family>_emit.py at `sha`, read from git (no worktree needed)."""
    K = paths.repository_folder(cfg)
    out = {}
    for rel in ("platform/tests/instances/parity", "library/tests/parity"):
        names = gitops.git(
            cfg["paths"]["repo"],
            "ls-tree",
            "--name-only",
            f"{sha}:{K}/{rel}",
            check=False,
        )
        for name in names.splitlines():
            if name.endswith("_emit.py"):
                out[name[: -len("_emit.py")]] = rel
    return out


def landed(cfg: dict, old: str, new: str) -> list[tuple[str, str, int | None]]:
    """(sha, subject, PR number) of first-parent rocKE commits in old..new, oldest first."""
    rows = gitops.first_parent_since(
        cfg["paths"]["repo"], old, new, paths.repository_folder(cfg)
    )
    out = []
    for sha, subj in reversed(rows):
        m = MERGED_PR.search(subj)
        out.append((sha, subj, int(m.group(1)) if m else None))
    return out


def sanity_script(
    cfg: dict,
    sha: str,
    wt: str,
    out: str,
    families: list[str],
    full: bool,
    sections: list[str] | None = None,
) -> str:
    """Bash for one sanity run at `sha` in worktree `wt`; writes out/sanity.md and out/status (one line per section:
    `<section> pass|fail|skip <detail>`). `sections` limits the run, e.g. to a bisect's failing sections.
    """
    want = lambda s: sections is None or s in sections
    fams = ",".join(families)
    name = pathlib.Path(wt).name
    L = [
        "set -u",
        paths.shell_vars(cfg),
        prcode.SHELL,  # prslot, prc, prr, prtree: the tree's code in a PR runner's slot (rkmon_lib/prcode.py)
        f"K={paths.repository_folder(cfg)}; SHA={sha}; WT={wt}; OUT={shlex.quote(out)}; P=$WT/$K/platform; "
        "L=$WT/$K/library",
        f"export {prcode.TREES_VAR}=$WT",  # the tree the slot's git reads
        'rm -rf "$OUT"; mkdir -p "$OUT"; : > "$OUT/status"; R=$OUT/sanity.md',
        "export PATH=$VB${LB:+:$LB}:$PATH TMPDIR=$TP ROCKE_LLVM_FLAVOR=llvm22",
        config.export_line(cfg),
        nogpu.SHELL,
        U8_FN,
        'st() { local d; d=$(printf \'%s\' "$3" | u8); echo "$1 $2 $d" >> $OUT/status; echo "**$1: $2** $d"; }',
        FLAVORS_FN,
        # The slot the tree's code runs in, its results copied back. Without one the run is cut short, never red.
        'prslot || { echo "# Develop sanity at ${SHA:0:12}: not run: no slot of a PR runner could be taken" > $R; '
        "exit 1; }",
        "if [ ! -d $WT ]; then prtree $WT && git -C $R0 worktree add -q --no-checkout --detach $WT $SHA && "
        f"git -C $WT sparse-checkout set --no-cone {paths.sparse_dirs(cfg)[0]}; fi",
        # Isolation: a pristine tree every run; build_native.py also builds from scratch.
        "git -C $WT reset -q --hard 2>/dev/null; git -C $WT clean -fdxq 2>/dev/null",
        'git -C $WT checkout -q --detach $SHA || { st checkout fail "cannot check out $SHA"; exit 1; }',
        'echo "# Develop sanity at ${SHA:0:12} ($(date -u +%FT%TZ))" > $R',
        "{",
    ]
    if want("build"):
        L += [
            "echo; echo '## Native build + ctest'",
            f"if prc python3 -B {paths.worker_tool('build_native.py')} {name} > $OUT/build.log 2>&1; then "
            f"prc -C $SB/{name}/core timeout 1800 ctest -j{paths.build_jobs()} > $OUT/ctest.log 2>&1; "
            "r=$(grep -E 'tests passed|tests failed' $OUT/ctest.log); echo \"$r\"; "
            'if echo "$r" | grep -q \'100% tests passed\'; then st build pass "$r"; else st build fail "$r; '
            "$(grep -E '\\*\\*\\*(Failed|Timeout)|Exception' $OUT/ctest.log | head -3 | tr -s ' ' | tr '\\n' ';')\"; "
            "fi; else st build fail \"native build failed: $(tail -3 $OUT/build.log | tr '\\n' ' ')\"; fi",
        ]
    if want("golden"):
        # One run, as CI's rocke_golden_static (it checks every flavor the golden holds), with the failing lines sorted
        # into load errors and drift by family and flavor (tools/golden_vs_base.py, L-523). Develop is its own base.
        # The gate is the tree's code; the report runs as the monitor, on the gate's results copied back, as data.
        L += [
            "echo; echo '## Representative golden'",
            "prc python3 -I -B $M/tools/golden_vs_base.py gate $SO/golden $P/tests > $OUT/golden_report.txt 2>&1; "
            "gr=$?; prr golden $OUT/golden; [ $gr != 1 ] || python3 -I -B $M/tools/golden_vs_base.py report "
            "$OUT/golden $P/tests self >> $OUT/golden_report.txt 2>&1; "
            "grep -v '^GOLDEN ' $OUT/golden_report.txt | head -20; "
            "g=$(grep '^GOLDEN ' $OUT/golden_report.txt | tail -1); [ -n \"$g\" ] || g='GOLDEN fail no result'; "
            'st golden $(echo "$g" | cut -d\' \' -f2) "$(echo "$g" | cut -d\' \' -f3- | cutc 200)"',
        ]
    if want("dg"):
        L += [
            "echo; echo '## Differential golden (CI CTest commands)'",
            f"({DG_FLAVORS}; DGD=$P/tests/instances/differential; prc -C $DGD timeout 900 python -B run_diff.py "
            "--mode ll --python-only --check-golden --flavor $DGF > $OUT/dg_ll.log 2>&1; a=$?; prc -C $DGD timeout 900 "
            "python -B run_diff.py --mode ir --canonical --python-only --check-golden > $OUT/dg_can.log 2>&1; b=$?; "
            "echo \"ll ($DGF) rc=$a canonical rc=$b\"; if [ $a -eq 0 ] && [ $b -eq 0 ]; then st dg pass 'GOLDEN OK'; else "
            "st dg fail \"$(grep -hE 'UNBLESSED|MISMATCH' $OUT/dg_ll.log $OUT/dg_can.log | head -2 | tr '\\n' ' ' | "
            'cut -c1-200)"; fi)',
        ]
    if want("byteid") and (families or full):
        L += ["echo; echo '## Byte identity'"]
        if families:
            L += [
                f"bad=''; FLS=$(flavors); for FL in $FLS; do ROCKE_LLVM_FLAVOR=$FL prc -C $P timeout 2400 python -B "
                f"tools/check_byte_identity.py --only {fams} --build-root $SB/{name}/core > "
                "$OUT/bi_$FL.log 2>&1; r=$(grep -E '^RESULT' $OUT/bi_$FL.log | cutc 60); echo \"$FL: $r\"; "
                'echo "$r" | grep -q GREEN || bad="$bad $FL"; done; '
                '[ -z "$bad" ] && st byteid pass "touched families at $FLS" || st byteid fail "not GREEN at$bad: $(grep '
                "-hE '^ *(RED|YELLOW)' $OUT/bi_*.log | head -3 | tr -s ' ' | tr '\\n' ';')\""
            ]
        if full:
            # Every family at every emission column, as the review checks (never only the host's flavor).
            L += [
                "bad=''; FLS=$(flavors); for FL in $FLS; do ROCKE_LLVM_FLAVOR=$FL prc -C $P timeout 3600 python -B "
                f"tools/check_byte_identity.py --build-root $SB/{name}/core > "
                "$OUT/bi_all_$FL.log 2>&1; r=$(grep -E '^RESULT' $OUT/bi_all_$FL.log | cutc 60); "
                'echo "$FL every family: $r"; echo "$r" | grep -q GREEN || bad="$bad $FL"; done; '
                '[ -z "$bad" ] && st byteid_all pass "every family at $FLS" || st byteid_all fail "not GREEN at$bad: '
                "$(grep -hE '^ *(RED|YELLOW)' $OUT/bi_all_*.log | head -3 | tr -s ' ' | tr '\\n' ';')\""
            ]
    if want("abi"):
        L += [
            "echo; echo '## ABI history invariant'",
            'o=$(python3 -I $M/tools/abi_check.py $P/cpp/include/rocke/abi.h); echo "$o"; '
            'echo "$o" | grep -q \'^OK\' && st abi pass "$o" || st abi fail "$o"',
        ]
    L += ["} >> $R 2>&1", "prfree", "echo done"]
    return "\n".join(L) + "\n"


def read_status(out: str) -> dict:
    p = pathlib.Path(out) / "status"
    res = {}
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split(" ", 2)
            if len(parts) >= 2:
                res[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return res


def bisect_script(
    cfg: dict,
    inc_id: str,
    cands: list[tuple[str, int | None]],
    sections: list[str],
    families: list[str],
) -> str:
    """Re-run only the failing sections on each candidate merge, oldest first; the first failing one is the culprit.
    Writes scratch/sanity/bisect-<id>/result (`culprit <sha> <pr>` or `none`)."""
    base = f"{paths.get(cfg, 'scratch')}/sanity/bisect-{inc_id}"
    parts = [f"mkdir -p {base}; : > {base}/result"]
    for sha, pr in cands:
        out = f"{base}/{sha[:12]}"
        body = sanity_script(
            cfg, sha, paths.worktree(cfg, "wtbisect"), out, families, False, sections
        )
        parts += [
            f"cat > {out}.sh <<'RKMON_EOF'",
            body.rstrip("\n"),
            "RKMON_EOF",
            f"bash {out}.sh; if grep -q ' fail ' {out}/status; then echo 'culprit {sha} {pr or '-'}' > "
            f"{base}/result; exit 0; fi",
        ]
    parts.append(f"echo none > {base}/result")
    return "\n".join(parts) + "\n"


def start_bisect(cfg: dict, st: dict, store) -> list[str]:
    opens = [i for i in incidents(store) if i.get("status") == "open"]
    if not opens:
        return ["no open incident"]
    inc = next(
        (i for i in reversed(opens) if i.get("candidates") and not i.get("culprit")),
        opens[0],
    )
    cands = [(c, n) for c, n in inc.get("candidates", [])]
    if not cands:
        return [
            f"{inc['id']}: no candidate merges recorded (CI-only signal); run `rk sanity` first"
        ]
    failing = sorted(
        {s.split(":", 1)[0] for s in inc.get("signals", [])}
        & {"build", "golden", "dg", "byteid", "abi"}
    )
    fams = st.get("guard", {}).get("last_run_families") or []
    full = []
    for c, _n in cands:  # full SHAs for the worktree checkouts
        full.append((gitops.git(cfg["paths"]["repo"], "rev-parse", c).strip(), _n))
    freed = jobs.free_tree(_trees(cfg, "wtbisect"), st, store, "bisect")
    job = jobs.start(
        store,
        "bisect",
        inc["id"],
        bisect_script(cfg, inc["id"], full, failing or None, fams),
        timeout_min=480,
    )
    st.setdefault("guard", {})["bisect"] = {"incident": inc["id"], "job": job}
    jobs.take_heavy(st, job, store)
    return freed + [
        f"BISECT_START {inc['id']}: {len(full)} candidates, sections {failing or 'all'}"
    ]


def bisect_result(cfg: dict, inc_id: str) -> str | None:
    p = pathlib.Path(f"{paths.get(cfg, 'scratch')}/sanity/bisect-{inc_id}/result")
    return p.read_text().strip() if p.exists() and p.read_text().strip() else None


# --- the tick's guard step -------------------------------------------------------------------------------------------


def _out(cfg: dict, sha: str) -> str:
    return f"{paths.get(cfg, 'scratch')}/sanity/{sha[:12]}"


def start_sanity(
    cfg: dict,
    st: dict,
    store,
    sha: str,
    base: str | None = None,
    rehearsal: bool = False,
) -> list[str]:
    g = st.setdefault("guard", {})
    lg = (st["develop"].get("last_green") or {}).get("sha")
    base = base or g.get("last_sanity_green") or lg or f"{sha}~1"
    try:
        rows = landed(cfg, base, sha)
    except gitops.GitError:
        rows = []
    files = []
    for c, _s, _n in rows:
        files += gitops.git(
            cfg["paths"]["repo"], "diff", "--name-only", f"{c}^1", c, check=False
        ).splitlines()
    K = paths.repository_folder(cfg)
    rel = [f[len(K) + 1 :] for f in files if f.startswith(K + "/")]
    tree = paths.worktree(cfg, "wtsanity")
    every = emitter_families_at(cfg, sha)
    fams = touched_families(rel, every)
    # A landed shared helper also selects the families that use it; one many families share means a full run.
    via = sorted(
        {
            f
            for fs in helper_families(
                ("git", cfg["paths"]["repo"], sha), K, rel, every, fams
            ).values()
            for f in fs
        }
    )
    wide = len(via) > MAX_HELPER_FAMILIES
    if not wide:
        fams = sorted(set(fams) | set(via))
    full = (
        wide
        or any(
            f.startswith(("platform/python/rocke/core/", "platform/cpp/core/"))
            for f in rel
        )
        or minutes_since(g.get("last_full") or "2000-01-01T00:00:00Z") > 24 * 60
    )
    script = sanity_script(cfg, sha, tree, _out(cfg, sha), fams, full)
    # never two runs in one tree
    freed = jobs.free_tree(_trees(cfg, "wtsanity"), st, store, "sanity")
    job = jobs.start(store, "sanity", sha[:12], script, timeout_min=240)
    g.update(
        running={
            "sha": sha,
            "job": job,
            "families": fams,
            "full": full,
            "landed": [(c[:12], n) for c, _s, n in rows],
            "base": base,
            "rehearsal": rehearsal,
        },
        last_run_families=fams,
    )
    jobs.take_heavy(st, job, store)
    if full:
        g["last_full"] = now()
    return freed + [
        f"SANITY_START {sha[:10]}:{len(rows)} rocKE landing(s) since {base[:10]} "
        f"({', '.join('#' + str(n) for _c, _s, n in rows if n)}); families {fams or '-'}; full={full}"
    ]


def advance(cfg: dict, st: dict, store, events: list[dict]) -> list[str]:
    g = st.setdefault("guard", {})
    lines = []
    # Develop CI red in a rocKE job is a signal of its own.
    for e in events:
        if e["kind"] == "DEVELOP_RED":
            why = stale_red(cfg, st, store, e["sha"], e.get("rocke_jobs") or [])
            if why:
                lines.append(f"DEVELOP_RED_STALE {e['sha']}: {why}; not an incident")
                continue
            inc = open_incident(
                store,
                st,
                e["sha"],
                [f"develop CI: {', '.join(e.get('rocke_jobs', [])) or 'rocKE job'}"],
                (st["develop"].get("last_green") or {}).get("sha"),
                [],
                ci_jobs=e.get("rocke_jobs"),
            )
            lines.append(
                f"INCIDENT_OPEN {inc['id']}: develop CI red at {e['sha']} ({inc['signals'][-1]})"
            )
    # The CI side of an open incident clears when its failed rocKE jobs pass again (a re-run of the red SHA counts),
    # checked every tick: a re-run turning green raises no event of its own.
    for cur in [i for i in incidents(store) if i.get("status") == "open"]:
        if "ci" not in _kinds(cur.get("signals") or []) or "ci" in (
            cur.get("green") or {}
        ):
            continue
        sha = ci_cleared(st, cur)
        if sha:
            closed = close_incident(store, st, sha, kind="ci", inc_id=cur["id"])
            if closed:
                lines.append(
                    f"INCIDENT_CLOSED {closed['id']}: develop CI green again at {sha[:10]} (the failed rocKE "
                    "jobs passed)"
                )
    b = g.get("bisect")
    bs = jobs.status(store, b["job"]) if b else None
    if b and bs != "running":
        # a bisect past its time still builds: stop it before letting go
        jobs.stop_unfinished(store, b["job"], bs)
        res = bisect_result(cfg, b["incident"])
        g["bisect"] = None
        jobs.release_heavy(st, b["job"])
        if res and res.startswith("culprit"):
            _, sha, pr = res.split()
            ii = incidents(store)
            for x in ii:
                if x.get("id") == b["incident"]:
                    x["culprit"] = None if pr == "-" else int(pr)
                    x["culprit_sha"] = sha[:12]
            save_incidents(store, ii)
            lines.append(f"CULPRIT {b['incident']}: #{pr} ({sha[:10]}) by bisect")
        else:
            lines.append(
                f"BISECT_DONE {b['incident']}: {res or 'no result'} (the failure may not reproduce per "
                "merge: check flakiness or an interaction)"
            )
    run = g.get("running")
    if run:
        s = jobs.status(store, run["job"])
        if s in ("running",):
            return lines + [f"SANITY {run['sha'][:10]}: running"]
        # past its time, or lost: stop what is left before letting go
        jobs.stop_unfinished(store, run["job"], s)
        res = read_status(_out(cfg, run["sha"]))
        failed = {k: v for k, v in res.items() if v[0] == "fail"}
        g["running"] = None
        jobs.release_heavy(st, run["job"])
        if s != "done" and not failed:
            # Cut short (past its time, lost, or its script stopped): the sections it wrote are not the whole run, so
            # it is never green and closes nothing. Run again once; a second cut-short run of the same SHA is said.
            got = f"{', '.join(sorted(res))} only" if res else "no result"
            if g.get("incomplete") != run["sha"] and not run.get("rehearsal"):
                g["incomplete"] = run["sha"]
                pend = st["develop"].setdefault("pending_sanity", [])
                if run["sha"] not in pend:
                    pend.insert(0, run["sha"])
                return lines + [
                    f"SANITY_INCOMPLETE {run['sha'][:10]}: job {s} ({got}): not green; it runs again"
                ]
            return lines + [
                f"!! SANITY_INCOMPLETE {run['sha'][:10]}: job {s} again ({got}): not green, not run a "
                "third time; check the job's log (`rk jobs`) and run `rk sanity` when it is fixed"
            ]
        if not failed and not run.get("rehearsal"):
            g["last_sanity_green"] = run["sha"]
            closed = close_incident(store, st, run["sha"], kind="sanity")
            lines.append(f"SANITY_GREEN {run['sha'][:10]}: {', '.join(sorted(res))}")
            if closed:
                lines.append(
                    f"INCIDENT_CLOSED {closed['id']}: develop green again at {run['sha'][:10]}"
                )
        else:
            signals = [f"{k}: {v[1][:140]}" for k, v in failed.items()]
            cands = [(c, n) for c, n in run["landed"]]
            inc = open_incident(store, st, run["sha"], signals, run["base"], cands)
            if run.get("rehearsal"):
                ii = incidents(store)
                for x in ii:
                    if x.get("id") == inc["id"]:
                        x["rehearsal"] = True
                save_incidents(store, ii)
            if len(cands) == 1 and not inc.get("culprit"):
                inc["culprit"] = cands[0][1]
                ii = incidents(store)
                for x in ii:
                    if x.get("id") == inc["id"]:
                        x["culprit"] = cands[0][1]
                save_incidents(store, ii)
            lines.append(
                f"INCIDENT_OPEN {inc['id']}: sanity red at {run['sha'][:10]}: {'; '.join(signals)[:300]}"
            )
            lines.append(
                f"CULPRIT {inc['id']}: #{inc['culprit']}"
                if inc.get("culprit")
                else f"BISECT_NEEDED {inc['id']}: {len(cands)} candidates {[n for _c, n in cands]} "
                f"(run `rk bisect`)"
            )
    pend = st["develop"].get("pending_sanity") or []
    if (
        pend
        and not g.get("running")
        and not jobs.heavy_busy(cfg, st, store)
        and not worker_prefill(cfg, st)
        and not halted(st)
    ):
        sha = pend[-1]
        st["develop"]["pending_sanity"] = []
        if (g.get("last_sanity_green") or "")[:12] != sha[
            :12
        ]:  # already tested green: nothing new to check
            lines += start_sanity(cfg, st, store, sha)
    incs, told = incidents(store), False
    for inc in incs:
        if inc.get("status") == "open":
            age = minutes_since(inc["start"])
            acts = inc.setdefault("actions", [])
            if age < revert_offer_min(cfg) or "revert_offered" in acts:
                continue
            fixed = (
                (st["prs"].get(str(inc.get("fixed_by"))) or {}).get("merge_sha")
                if inc.get("fixed_by")
                else None
            )
            if fixed:  # nothing to revert once the fix landed
                if "fix_merged_told" not in acts:
                    acts.append("fix_merged_told")
                    told = True
                    lines.append(
                        f"INCIDENT_FIX_MERGED {inc['id']}: #{inc['fixed_by']} merged as {fixed[:10]}: no "
                        "revert; it closes when develop's run after it is green (or `rk incident-close "
                        f"{inc['id']} {inc['fixed_by']}`)"
                    )
                continue
            if inc.get("culprit"):
                lines.append(
                    f"REVERT_OFFER {inc['id']}: open {age:.0f} min; prepare a local revert of "
                    f"#{inc['culprit']} and ask the user"
                )
            elif "aged_told" not in acts:
                # Nothing to revert without a culprit (e.g. a test past its timeout): the way out is the develop issue's
                # fix. Said once, not critical.
                acts.append("aged_told")
                told = True
                lines.append(
                    f"INCIDENT_AGED {inc['id']}: open {age:.0f} min with no culprit to revert; the way out is "
                    f"the fix for {'#' + str(inc['issue']) if inc.get('issue') else 'its develop issue'}"
                )
    if told:
        save_incidents(store, incs)
    return lines
