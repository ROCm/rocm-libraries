# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk evidence-check N [--round R]`: the offload passes' evidence checked again, with no model (rkmon_lib/offload.py;
the records are worker/offload_agent.py's evidence.jsonl and runs.jsonl). The auditor reads its report first.

For each claim:
- read: the quote must stand verbatim at file:line in the tree it names, read from the monitor's clone at the commit
  the run recorded (`git show <sha>:<file>`, never a lazy fetch): the head, the head merged with develop, or the merge
  base; a quote of the checks' report is read from the copy kept with the run;
- ran: the run must be one of this review's, and the command the tool's own builder makes now (worker/reviewer_tools.py)
  must be the recorded one. A run of an allow-listed tool that took at most offload.check_rerun_max_s is re-run under
  a time cap (offload.check_total_s in all, 🔴/🟠 claims first) and must give the same output as the record, reduced
  to what must match (reviewer_tools.normalize: pytest's counts and failing tests, grep's matches, a probe's verdict).
  CI reads (ci_fails, job_log) are read again: a difference is CHANGED (CI moved on), not a failure. A re-run needs
  its tree still at the recorded commit;
- none: a claim the passes left without evidence is MISSING.
Each claim is PASS, FAIL, CHANGED, NOT RE-RUN (and why) or MISSING. The result is FAIL when a claim fails, a 🔴/🟠
claim is missing, or the run recorded no claim at all, else PASS. It also draws the auditor's spot-check sample of the 🟡/🟢 claims (offload.spot_check, at
least offload.spot_check_min), the same for the same run.

Without --round it checks the live run (workers/pr<N>, or its offload/ in shadow mode); with --round, that round's
run, the live one when it is that round's, else the kept one. The report (evidence_check.md, evidence_check.json)
goes in the run's folder, and the live run is kept with it in <lessons>/offload/pr<N>_<round>/. Exit 0 on PASS, 1 on
FAIL, 2 when there is no run to check.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from . import config, offload, paths
from .snapshot import same_commit

ALLOWED = (
    "grep_code",
    "git_diff",
    "git_log",
    "run_pytest",
    "rule_scan",
    "merge_conflicts",
    "run_probe",
    "run_mutant",
    "ci_fails",
    "job_log",
)
CI_TOOLS = ("ci_fails", "job_log")
MERGED_TOOLS = (
    "merge_conflicts",
)  # the rest run on the head tree (a probe names its tree)
SEVERITY_ORDER = {"🔴": 0, "🟠": 1, "🟡": 2, "🟢": 3, "⚪": 4, "": 5}
KEEP = (
    "offload_run.json",
    "draft.md",
    "quality.md",
    "regrade.md",
    "post_draft.md",
    "notes_draft.md",
    "evidence.jsonl",
    "runs.jsonl",
    "tokens.jsonl",
    "tokens_total.json",
    "evidence_check.md",
    "evidence_check.json",
)
KEEP_DIRS = ("probes", "mutants", "prep")


def tools():
    sys.path.insert(0, str(paths.PACKAGE / "worker"))
    import reviewer_tools

    return reviewer_tools


def _jsonl(path: pathlib.Path) -> list[dict]:
    try:
        return [
            json.loads(x)
            for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    except (OSError, ValueError):
        return []


def run_header(folder: pathlib.Path) -> dict:
    try:
        return json.loads((folder / "offload_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find(cfg: dict, n: int, rnd: str | None = None) -> pathlib.Path | None:
    """The run to check: the live one (any round, or the round asked for), else the kept one of that round."""
    for m in ("on", "shadow"):
        live = offload.review_dir(cfg, n, m)
        h = run_header(live)
        if h and (not rnd or h.get("round") == rnd):
            return live
    if rnd:
        kept = offload.archive_dir(cfg, n, rnd)
        if run_header(kept):
            return kept
    return None


def tree_for(cfg: dict, folder: pathlib.Path, n: int):
    """The review tools' Tree on PR n's worktrees, with the job's paths from this configuration; the run's staged PR
    data (pr/) where it was."""
    env = config.env(cfg)
    rt = tools()
    pr_dir = folder / "pr" if (folder / "pr").is_dir() else folder.parent / "pr"
    return rt.Tree(
        folder, n, pr_dir=pr_dir, get=lambda name, required=True: env.get(name, "")
    )


def git_lines(repo: str, sha: str, path: str) -> list[str] | None:
    """A file's lines at a commit in the monitor's clone, None when it is not there (or its blob is not local)."""
    if not sha:
        return None
    p = subprocess.run(
        ["git", "-C", repo, "show", f"{sha}:{path}"],
        capture_output=True,
        timeout=60,
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"),
    )
    return (
        p.stdout.decode("utf-8", "replace").splitlines() if p.returncode == 0 else None
    )


def check_read(cfg: dict, folder: pathlib.Path, e: dict) -> tuple[str, str]:
    rt = tools()
    tree = e.get("tree") or "head"
    if tree == "checks":
        copy = folder / "checks.md"
        live = (
            folder.parent / "checks" / "checks.md"
            if folder.name == "offload"
            else folder / "checks" / "checks.md"
        )
        src = live if live.is_file() else copy
        try:
            lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "FAIL", "the checks' report is gone"
    else:
        lines = git_lines(
            paths.get(cfg, "repo"), e.get("sha") or "", e.get("file") or ""
        )
        if lines is None:
            return (
                "FAIL",
                f"no {e.get('file')} at {str(e.get('sha'))[:10]} in the clone",
            )
    ok, where = rt.quote_at(lines, int(e.get("line") or 0), e.get("quote") or "")
    if ok:
        return "PASS", f"quote at {e.get('file')}:{e.get('line')}"
    return "FAIL", (
        f"the quote is not at {e.get('file')}:{e.get('line')}"
        + (f" (it is at line {where})" if where else " (nowhere in the file)")
    )


class Reruns:
    """The re-runs of one check, each run once whatever the number of claims that cite it, within the time budget."""

    def __init__(self, cfg: dict, folder: pathlib.Path, n: int, header: dict):
        self.s = offload.settings(cfg)
        self.folder, self.n, self.header = folder, n, header
        self.live = any(
            folder == offload.review_dir(cfg, n, m) for m in ("on", "shadow")
        )
        self.t = tree_for(cfg, folder, n)
        self.runs = {r["run"]: r for r in _jsonl(folder / "runs.jsonl")}
        self.left = float(self.s["check_total_s"])
        self.done: dict[str, tuple[str, str]] = {}

    def check(self, rid: str) -> tuple[str, str]:
        if rid not in self.done:
            self.done[rid] = self._check(rid)
        return self.done[rid]

    def _tree_moved(self, tool: str, r: dict) -> str:
        merged = tool in MERGED_TOOLS or (
            tool == "run_probe"
            and (r.get("args") or {}).get("tree", "merged") == "merged"
        )
        want = r.get("merged") if merged else r.get("head")
        have = self.t.sha("merged" if merged else "head")
        if want and not (have and same_commit(want, have)):
            where = f"moved to {have[:10]}" if have else "is gone"
            return f"the {'merged' if merged else 'head'} worktree {where} (recorded {want[:10]})"
        return ""

    def _check(self, rid: str) -> tuple[str, str]:
        rt = tools()
        r = self.runs.get(rid)
        if not r:
            return "FAIL", f"no run {rid} in runs.jsonl"
        tool, args = r.get("tool"), dict(r.get("args") or {})
        if tool not in ALLOWED:
            return "FAIL", f"{tool} is no tool a claim may cite"
        if not self.live:
            return "NOT RE-RUN", "a kept run: only its quotes are read again"
        if tool == "run_mutant":
            args["control"] = False
        if tool == "run_probe":
            args.setdefault("timeout", self.s["probe_timeout_s"])
        moved = "" if tool in CI_TOOLS else self._tree_moved(tool, r)
        if moved:
            return "NOT RE-RUN", moved
        built = rt.build(self.t, tool, args)
        if isinstance(built, str):
            return "FAIL", f"the tool refuses the recorded arguments now: {built[:160]}"
        argv, cwd, timeout, env = built
        if [str(a) for a in argv] != list(r.get("argv") or []):
            return "FAIL", "the recorded command is not the one the tool builds"
        secs = float(r.get("secs") or 0)
        if secs > float(self.s["check_rerun_max_s"]):
            return (
                "NOT RE-RUN",
                f"not cheap: the recorded run took {secs:.0f} s (offload.check_rerun_max_s)",
            )
        cap = min(max(30.0, 3 * secs + 10), self.left)
        if cap < 5:
            return "NOT RE-RUN", "the time budget is spent (offload.check_total_s)"
        t0 = time.monotonic()
        res = rt.execute(argv, cwd, int(cap), env)
        self.left -= time.monotonic() - t0
        if res.rc is None:
            return "NOT RE-RUN", f"the re-run passed its {cap:.0f} s cap"
        if tool == "run_mutant":
            return self._mutant(r, res)
        norm = rt.normalize(tool, res.out, self.t)
        if rt.digest(norm) == r.get("digest"):
            return "PASS", f"re-run {tool}: the same output"
        why = f"re-run {tool}: recorded {_short(r.get('norm'))}; now {_short(norm)}"
        return ("CHANGED", why) if tool in CI_TOOLS else ("FAIL", why)

    def _mutant(self, r: dict, res) -> tuple[str, str]:
        rt = tools()
        now = json.loads(rt.normalize("run_mutant", res.out))
        try:
            was = json.loads(r.get("norm") or "{}")
        except ValueError:
            was = {}
        if now["counts"] == was.get("mutant") and now["failing"] == was.get("failing"):
            return "PASS", f"re-run mutant: {was.get('verdict', '?')} again"
        return (
            "FAIL",
            f"re-run mutant: recorded {was.get('mutant')}; now {now['counts']}",
        )


def _short(text) -> str:
    t = " ".join(str(text or "").split())
    return (t[:140] + "…") if len(t) > 140 else (t or "(nothing)")


def sample(claims: list[dict], s: dict, seed: str) -> list[str]:
    """The auditor's spot-check of the 🟡/🟢 claims: offload.spot_check of them, at least spot_check_min (all when
    there are fewer), drawn by a hash of the run and the claim, so the same run always gives the same sample.
    """
    pool = sorted(
        {c["id"] for c in claims if c["severity"] in ("🟡", "🟢", "")},
        key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(),
    )
    k = max(int(s["spot_check_min"]), -(-len(pool) * float(s["spot_check"]) // 1))
    return sorted(pool[: int(min(len(pool), k))], key=_id_key)


def _id_key(cid: str) -> tuple:
    return ("NDQPC".find(cid[:1]), int(cid[1:]) if cid[1:].isdigit() else 0)


def check(cfg: dict, n: int, folder: pathlib.Path) -> dict:
    """Check one run; its report (also written to evidence_check.md and .json in the folder)."""
    header = run_header(folder)
    records = _jsonl(folder / "evidence.jsonl")
    reruns = Reruns(cfg, folder, n, header)
    by_claim: dict[str, list[dict]] = {}
    for e in records:
        by_claim.setdefault(e["id"], []).append(e)
    rows = []
    for cid, es in by_claim.items():
        sev = next((e["severity"] for e in es if e.get("severity")), "")
        rows.append(
            {
                "id": cid,
                "pass": es[0].get("pass"),
                "severity": sev,
                "claim": next(
                    (e.get("claim") for e in es if e.get("kind") != "none"),
                    es[0].get("claim"),
                ),
                "evidence": es,
            }
        )
    rows.sort(key=lambda c: (SEVERITY_ORDER.get(c["severity"], 5), _id_key(c["id"])))
    for c in rows:
        found = [e for e in c["evidence"] if e.get("kind") in ("ran", "read")]
        if not found:
            c["result"], c["why"] = "MISSING", "no evidence"
            continue
        results = []
        for e in found:
            res = (
                check_read(cfg, folder, e)
                if e["kind"] == "read"
                else reruns.check(e.get("run") or "")
            )
            results.append((res, e))
        worst = min(
            results,
            key=lambda x: ("FAIL", "MISSING", "CHANGED", "NOT RE-RUN", "PASS").index(
                x[0][0]
            ),
        )
        (c["result"], c["why"]), e = worst
        c["shown"] = (
            f"ran {e.get('run')} {e.get('tool')}"
            if e["kind"] == "ran"
            else f"read {e.get('file')}:{e.get('line')} ({e.get('tree')})"
        )
    counts = {
        k: sum(1 for c in rows if c["result"] == k)
        for k in ("PASS", "FAIL", "CHANGED", "NOT RE-RUN", "MISSING")
    }
    serious_missing = [
        c["id"]
        for c in rows
        if c["result"] == "MISSING" and c["severity"] in ("🔴", "🟠")
    ]
    failed = [c["id"] for c in rows if c["result"] == "FAIL"]
    result = "FAIL" if failed or serious_missing or not rows else "PASS"
    summary = (
        f"{result}: {len(rows)} claims, {counts['PASS']} pass, {counts['FAIL']} fail, {counts['CHANGED']} "
        f"changed, {counts['NOT RE-RUN']} not re-run, {counts['MISSING']} missing"
        + (f" ({len(serious_missing)} 🔴/🟠)" if serious_missing else "")
        if rows
        else "FAIL: no claims recorded (the passes left nothing to check)"
    )
    s = offload.settings(cfg)
    report = {
        "pr": n,
        "round": header.get("round"),
        "head": header.get("head"),
        "mode": header.get("mode"),
        "result": result,
        "summary": summary,
        "counts": counts,
        "failed": failed,
        "missing_serious": serious_missing,
        "sample": sample(rows, s, f"{header.get('head')}:{header.get('round')}"),
        "claims": [
            {
                k: c.get(k)
                for k in ("id", "pass", "severity", "claim", "result", "why", "shown")
            }
            for c in rows
        ],
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (folder / "evidence_check.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (folder / "evidence_check.md").write_text(render(report, s), encoding="utf-8")
    return report


def render(report: dict, s: dict) -> str:
    head = str(report.get("head") or "")[:10]
    lines = [
        f"# Evidence check: #{report['pr']} {report.get('round') or ''} at {head} — {report['result']}",
        "",
        report["summary"],
        "",
    ]
    if report["sample"]:
        lines += [
            f"Spot-check these 🟡/🟢 claims (offload.spot_check {float(s['spot_check']) * 100:.0f}%, at least "
            f"{s['spot_check_min']}): {', '.join(report['sample'])}",
            "",
        ]
    lines += ["| Claim | Sev | Evidence | Result |", "|---|---|---|---|"]
    for c in report["claims"]:
        why = str(c.get("why") or "").replace("|", "/")
        lines.append(
            f"| {c['id']} | {c.get('severity') or '—'} | {(c.get('shown') or '—').replace('|', '/')} | "
            f"{c['result']}{': ' + why if why and c['result'] != 'PASS' else ''} |"
        )
    lines += [
        "",
        "PASS held again · FAIL contradicted (a quote not at its line, a re-run that differs, a command that is "
        "not the tool's) · CHANGED CI moved on since · NOT RE-RUN check it yourself · MISSING no evidence",
        "",
    ]
    return "\n".join(lines)


def keep(cfg: dict, folder: pathlib.Path, n: int, header: dict) -> pathlib.Path | None:
    """Copy the live run into <lessons>/offload/pr<N>_<round>/, with the checks' report it was assembled from."""
    rnd = header.get("round")
    if not rnd:
        return None
    dest = offload.archive_dir(cfg, n, rnd)
    dest.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (folder / name).is_file():
            shutil.copyfile(folder / name, dest / name)
    for name in KEEP_DIRS:
        if (folder / name).is_dir():
            shutil.copytree(folder / name, dest / name, dirs_exist_ok=True)
    live_checks = (
        folder.parent / "checks" / "checks.md"
        if folder.name == "offload"
        else folder / "checks" / "checks.md"
    )
    if live_checks.is_file():
        shutil.copyfile(live_checks, dest / "checks.md")
    return dest


def cmd(cfg: dict, n: int, rnd: str | None = None) -> tuple[list[str], int]:
    """`rk evidence-check N [--round R]`: the report's path, its summary and the spot-check sample; exit 0 PASS, 1
    FAIL, 2 nothing to check."""
    folder = find(cfg, n, rnd)
    if folder is None:
        return [
            f"no offload run of #{n}"
            + (f" for round {rnd}" if rnd else "")
            + " to check (offload.mode "
            f"{offload.mode(cfg)}; kept runs: {offload.archive_root(cfg)})"
        ], 2
    report = check(cfg, n, folder)
    lines = [f"{folder}/evidence_check.md", report["summary"]]
    if report["sample"]:
        lines.append(f"spot-check: {', '.join(report['sample'])}")
    live = any(folder == offload.review_dir(cfg, n, m) for m in ("on", "shadow"))
    if live:
        kept = keep(cfg, folder, n, run_header(folder))
        if kept:
            lines.append(f"kept in {kept}")
    return lines, 0 if report["result"] == "PASS" else 1
