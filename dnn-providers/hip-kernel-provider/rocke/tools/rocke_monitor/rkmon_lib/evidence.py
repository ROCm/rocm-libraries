# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk evidence-check N [--dir DIR]`: a draft's evidence checked again, with no model (the records are the tool-loop
drafter's evidence.jsonl and runs.jsonl, worker/drafter.py). The draft and shadow stages run it before their jobs end.

For each claim:
- read: the quote must stand at file:line in the tree it names, read from the monitor's clone at the commit the run
  recorded (`git show <sha>:<file>`, never a lazy fetch): the head, the head merged with develop, or the merge base; a
  quote of the staged PR data is read from the run's pr/ folder;
- ran: the run must be one of this draft's, and the command the tool's own builder makes now (worker/review_tools.py)
  must be the recorded one. A run that took at most drafter.check_rerun_max_s is re-run under a time cap
  (drafter.check_total_s in all, 🔴/🟠 claims first) and must give the same output, reduced to what must match
  (review_tools.normalize). A re-run needs its tree still at the recorded commit. A saved probe's run is not re-run: its
  probe file must still be the one that ran (its digest) and its tree at the recorded commit;
- unchanged: each cited file's blobs at the four recorded commits (our last review's head, the head, the develop then
  and today's), read again from the clone's trees (no fetch), must be the recorded ones;
- data: the fact read from the staged PR data (the Threads row's counts) must read the same again;
- none: a claim the draft left without evidence is MISSING.
Each claim is PASS, FAIL, NOT RE-RUN (and why) or MISSING. The result is FAIL when a claim fails, a 🔴/🟠 claim is
missing, or the draft recorded no claim at all, else PASS.

The report goes in the run's folder (evidence_check.md, evidence_check.json), and with it prior_settled.md and .json
(rkmon_lib/prior_settled.py: the prior rows settled with passing evidence, and the ones to look at). The draft keeps
every claim: each FAIL or MISSING claim's row in review_agent.md gets a mark saying so (MARK), and a second check
replaces the marks. The run is kept with its report in <paths.root>/lessons/drafts/pr<N>_<head10>/ (a shadow draft's
in pr<N>_<head10>_<binding>/). Exit 0 on PASS, 1 on FAIL, 2 when there is no draft run to check.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from . import bindings, paths, prior_settled
from .snapshot import same_commit

WORKER = pathlib.Path(__file__).resolve().parent.parent / "worker"
ALLOWED = ("grep_code", "git_log", "rule_scan", "merge_conflicts", "probe")
MERGED_TOOLS = ("merge_conflicts",)
SEVERITY_ORDER = {"🔴": 0, "🟠": 1, "🟡": 2, "🟢": 3, "⚪": 4, "": 5}
RUN_FILE = "draft_run.json"
KEEP = (
    RUN_FILE,
    "review_agent.md",
    "draft_findings.json",
    "evidence.jsonl",
    "runs.jsonl",
    "tokens_total.json",
    "evidence_check.md",
    "evidence_check.json",
    "agent_log.txt",
    "prior_leads.md",
    prior_settled.FILE_MD,
    prior_settled.FILE_JSON,
    "probes.txt",
)
MARK = re.compile(r"<br>• \*\*evidence check (?:FAIL|MISSING)\*\*[^|]*")


def tools():
    sys.path.insert(0, str(WORKER))
    import review_tools

    return review_tools


def settle_rules():
    sys.path.insert(0, str(WORKER))
    import prior_settle

    return prior_settle


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
        return json.loads((folder / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find(cfg: dict, n: int, d: str | None = None) -> pathlib.Path | None:
    """The run to check: DIR, else <paths.workers>/pr<N>; None when it holds no draft run."""
    folder = (
        pathlib.Path(d)
        if d
        else pathlib.Path((cfg.get("paths") or {}).get("workers") or "") / f"pr{n}"
    )
    return folder if run_header(folder) else None


def tree_for(cfg: dict, folder: pathlib.Path, n: int, header: dict):
    rt = tools()
    return rt.Tree(
        folder,
        n,
        (cfg.get("paths") or {}).get("root") or str(folder.parent.parent),
        paths.repository_folder(cfg),
        trees=header.get("trees") or None,
    )


def git_lines(repo: str, sha: str, path: str) -> list[str] | None:
    """A file's lines at a commit in the monitor's clone; None when it is not there (or its blob is not local)."""
    if not sha or not repo:
        return None
    try:
        p = subprocess.run(
            ["git", "-C", repo, "show", f"{sha}:{path}"],
            capture_output=True,
            timeout=60,
            env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (
        p.stdout.decode("utf-8", "replace").splitlines() if p.returncode == 0 else None
    )


def check_read(cfg: dict, folder: pathlib.Path, e: dict) -> tuple[str, str]:
    rt = tools()
    if (e.get("tree") or "head") == "staged":
        name = pathlib.PurePosixPath(str(e.get("file") or "")).name
        try:
            lines = (
                (folder / "pr" / name)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            return "FAIL", f"the staged {name} is gone"
    else:
        lines = git_lines(
            (cfg.get("paths") or {}).get("repo") or "",
            e.get("sha") or "",
            e.get("file") or "",
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


def check_unchanged(cfg: dict, e: dict) -> tuple[str, str]:
    """The cited files' blobs at the four recorded commits, read again from the clone's trees."""
    files = e.get("files") or {}
    repo = (cfg.get("paths") or {}).get("repo") or ""
    same, now = settle_rules().unchanged(
        repo,
        e.get("since") or "",
        e.get("head") or "",
        e.get("develop_then") or "",
        e.get("develop") or "",
        list(files),
    )
    if same is None:
        return "NOT RE-RUN", f"{str(now.get('missing'))[:10]} is not in the clone"
    moved = [
        p
        for p, v in (now.get("files") or {}).items()
        if list(v) != list(files.get(p) or [])
    ]
    if moved:
        return "FAIL", f"the blobs of {', '.join(moved[:3])} are not the recorded ones"
    if not same:
        return "FAIL", "the recorded files are not unchanged"
    return "PASS", f"unchanged since {str(e.get('since'))[:10]}: {len(files)} files"


def check_data(folder: pathlib.Path, e: dict) -> tuple[str, str]:
    """A fact read from the staged PR data, read again."""
    ps = settle_rules()
    parsers = {"threads": lambda text: ps.threads_value(ps.threads_state(text))}
    name = pathlib.PurePosixPath(str(e.get("file") or "")).name
    if e.get("parser") not in parsers:
        return "FAIL", f"no parser {e.get('parser')!r}"
    try:
        text = (folder / "pr" / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "FAIL", f"the staged {name} is gone"
    now = parsers[e["parser"]](text)
    if now == e.get("value"):
        return "PASS", f"{name} reads the same"
    return "FAIL", f"{name} reads {now}, not {e.get('value')}"


class Reruns:
    """The re-runs of one check, each run once whatever the number of claims that cite it, within the time budget."""

    def __init__(self, cfg: dict, folder: pathlib.Path, n: int, header: dict):
        self.s = bindings.drafter_settings(cfg)
        self.t = tree_for(cfg, folder, n, header)
        self.runs = {r["run"]: r for r in _jsonl(folder / "runs.jsonl")}
        self.left = float(self.s["check_total_s"])
        self.done: dict[str, tuple[str, str]] = {}

    def check(self, rid: str) -> tuple[str, str]:
        if rid not in self.done:
            self.done[rid] = self._check(rid)
        return self.done[rid]

    def _tree_moved(self, tool: str, r: dict) -> str:
        merged = tool in MERGED_TOOLS or (
            tool == "probe" and (r.get("args") or {}).get("tree") == "merged"
        )
        want = r.get("merged") if merged else r.get("head")
        have = self.t.sha("merged" if merged else "head")
        if want and not (have and same_commit(want, have)):
            return (
                f"the {'merged' if merged else 'head'} tree "
                + (f"moved to {have[:10]}" if have else "is gone")
                + f" (recorded {want[:10]})"
            )
        return ""

    def _check(self, rid: str) -> tuple[str, str]:
        rt = tools()
        r = self.runs.get(rid)
        if not r:
            return "FAIL", f"no run {rid} in runs.jsonl"
        tool, args = r.get("tool"), dict(r.get("args") or {})
        if tool not in ALLOWED:
            return "FAIL", f"{tool} is no tool a claim may cite"
        moved = self._tree_moved(tool, r)
        if moved:
            return "NOT RE-RUN", moved
        built = rt.build(self.t, tool, args)
        if isinstance(built, str):
            return "FAIL", f"the tool refuses the recorded arguments now: {built[:160]}"
        argv, cwd, _timeout, env = built
        if [str(a) for a in argv] != list(r.get("argv") or []):
            return "FAIL", "the recorded command is not the one the tool builds"
        if tool == "probe":
            now = rt.file_digest(self.t.probes / f"{args.get('name')}.sh")
            if not now or now != r.get("probe_digest"):
                return "FAIL", "the saved probe changed since it ran"
            return (
                "PASS",
                f"the probe and its tree are the ones that ran: {_short(r.get('norm'))}",
            )
        secs = float(r.get("secs") or 0)
        if secs > float(self.s["check_rerun_max_s"]):
            return (
                "NOT RE-RUN",
                f"not cheap: the recorded run took {secs:.0f} s (drafter.check_rerun_max_s)",
            )
        cap = min(max(30.0, 3 * secs + 10), self.left)
        if cap < 5:
            return "NOT RE-RUN", "the time budget is spent (drafter.check_total_s)"
        t0 = time.monotonic()
        res = rt.execute(argv, cwd, int(cap), env)
        self.left -= time.monotonic() - t0
        if res.rc is None:
            return "NOT RE-RUN", f"the re-run passed its {cap:.0f} s cap"
        norm = rt.normalize(tool, res.out, self.t)
        if rt.digest(norm) == r.get("digest"):
            return "PASS", f"re-run {tool}: the same output"
        return (
            "FAIL",
            f"re-run {tool}: recorded {_short(r.get('norm'))}; now {_short(norm)}",
        )


def _short(text) -> str:
    t = " ".join(str(text or "").split())
    return (t[:140] + "…") if len(t) > 140 else (t or "(nothing)")


def _id_key(cid: str) -> tuple:
    return ("NPC".find(cid[:1]), int(cid[1:]) if cid[1:].isdigit() else 0)


def _result(
    cfg: dict, folder: pathlib.Path, reruns: Reruns, e: dict
) -> tuple[str, str]:
    kind = e.get("kind")
    if kind == "read":
        return check_read(cfg, folder, e)
    if kind == "ran":
        return reruns.check(e.get("run") or "")
    if kind == "unchanged":
        return check_unchanged(cfg, e)
    if kind == "data":
        return check_data(folder, e)
    return "FAIL", f"no check for evidence of kind {kind!r}"


def _shown(e: dict) -> str:
    kind = e.get("kind")
    if kind == "ran":
        return f"ran {e.get('run')} {e.get('tool')}"
    if kind == "read":
        return f"read {e.get('file')}:{e.get('line')} ({e.get('tree')})"
    if kind == "unchanged":
        return f"unchanged since {str(e.get('since'))[:10]} ({len(e.get('files') or {})} files)"
    if kind == "data":
        return f"data {e.get('file')}"
    return str(kind)


def check(cfg: dict, n: int, folder: pathlib.Path) -> dict:
    """Check one run; its report (also written to evidence_check.md and .json in the folder), the draft marked, and
    prior_settled.md written."""
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
        found = [e for e in c["evidence"] if e.get("kind") not in ("none", None)]
        if not found:
            c["result"], c["why"] = "MISSING", "no evidence"
            continue
        results = [(_result(cfg, folder, reruns, e), e) for e in found]
        order = ("FAIL", "MISSING", "NOT RE-RUN", "PASS")
        (c["result"], c["why"]), e = min(results, key=lambda x: order.index(x[0][0]))
        c["shown"] = _shown(e)
    counts = {
        k: sum(1 for c in rows if c["result"] == k)
        for k in ("PASS", "FAIL", "NOT RE-RUN", "MISSING")
    }
    serious_missing = [
        c["id"]
        for c in rows
        if c["result"] == "MISSING" and c["severity"] in ("🔴", "🟠")
    ]
    failed = [c["id"] for c in rows if c["result"] == "FAIL"]
    result = "FAIL" if failed or serious_missing or not rows else "PASS"
    serious = f" ({len(serious_missing)} 🔴/🟠)" if serious_missing else ""
    summary = (
        f"{result}: {len(rows)} claims, {counts['PASS']} pass, {counts['FAIL']} fail, {counts['NOT RE-RUN']} "
        f"not re-run, {counts['MISSING']} missing{serious}"
        if rows
        else "FAIL: no claims recorded"
    )
    report = {
        "pr": n,
        "round": header.get("round"),
        "head": header.get("head"),
        "result": result,
        "summary": summary,
        "counts": counts,
        "failed": failed,
        "missing_serious": serious_missing,
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
    (folder / "evidence_check.md").write_text(render(report), encoding="utf-8")
    mark_draft(folder, report)
    prior_settled.write(folder, report)
    return report


def render(report: dict) -> str:
    head = str(report.get("head") or "")[:10]
    lines = [
        f"# Evidence check: #{report['pr']} {report.get('round') or ''} at {head} — {report['result']}",
        "",
        report["summary"],
        "",
        "| Claim | Sev | Evidence | Result |",
        "|---|---|---|---|",
    ]
    for c in report["claims"]:
        why = str(c.get("why") or "").replace("|", "/")
        lines.append(
            f"| {c['id']} | {c.get('severity') or '—'} | {(c.get('shown') or '—').replace('|', '/')} | "
            f"{c['result']}{': ' + why if why and c['result'] != 'PASS' else ''} |"
        )
    lines += [
        "",
        "PASS held again · FAIL contradicted (a quote not at its line, a re-run that differs, a command that "
        "is not the tool's, a blob or a staged fact that is not the recorded one) · NOT RE-RUN check it yourself "
        "· MISSING no evidence",
        "",
    ]
    return "\n".join(lines)


def mark_draft(folder: pathlib.Path, report: dict) -> int:
    """Each FAIL or MISSING claim's row in review_agent.md gets a mark; earlier marks are replaced. The rows: a
    finding's by its id (| N1 |), a prior row's by its key, a check point's by its row (draft_findings.json). The number
    of rows marked."""
    path = folder / "review_agent.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    try:
        known = json.loads((folder / "draft_findings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    keys = {p["id"]: p.get("key") for p in known.get("priors") or []}
    keys.update(known.get("checks") or {})
    lines = [MARK.sub("", ln) for ln in text.split("\n")]
    marked = 0
    for c in report["claims"]:
        if c["result"] not in ("FAIL", "MISSING"):
            continue
        what = (
            f"<br>• **evidence check FAIL**: {_short(c.get('why'))[:120]}"
            if c["result"] == "FAIL"
            else "<br>• **evidence check MISSING**: no evidence recorded"
        )
        what = what.replace("|", "/")
        start = (
            f"| {c['id']} |"
            if c["id"].startswith("N")
            else f"| {keys.get(c['id'])} |" if keys.get(c["id"]) else ""
        )
        for i, ln in enumerate(lines):
            if start and ln.startswith(start) and ln.rstrip().endswith("|"):
                lines[i] = ln.rstrip()[:-1].rstrip() + what + " |"
                marked += 1
                break
    path.write_text("\n".join(lines), encoding="utf-8")
    return marked


def keep(cfg: dict, folder: pathlib.Path, n: int, header: dict) -> pathlib.Path | None:
    """Copy the run into <paths.root>/lessons/drafts/pr<N>_<head10>/ (a shadow draft's: pr<N>_<head10>_<binding>/)."""
    root = (cfg.get("paths") or {}).get("root")
    head = str(header.get("head") or "")[:10]
    if not root or not head:
        return None
    suffix = (
        f"_{header.get('binding')}"
        if header.get("role") == "shadow_drafter" and header.get("binding")
        else ""
    )
    dest = pathlib.Path(root) / "lessons" / "drafts" / f"pr{n}_{head}{suffix}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (folder / name).is_file():
            shutil.copyfile(folder / name, dest / name)
    return dest


def cmd(cfg: dict, n: int, d: str | None = None) -> tuple[list[str], int]:
    """`rk evidence-check N [--dir DIR]`: the report's path, its summary and where the run is kept; exit 0 PASS, 1 FAIL,
    2 nothing to check."""
    folder = find(cfg, n, d)
    if folder is None:
        return [f"no draft run of #{n} to check" + (f" in {d}" if d else "")], 2
    report = check(cfg, n, folder)
    lines = [f"{folder}/evidence_check.md", report["summary"]]
    ps = prior_settled.load(folder)
    if ps:
        c = ps.get("counts") or {}
        lines.append(
            f"{folder}/{prior_settled.FILE_MD}: {c.get('rows', 0)} prior rows, {c.get('look', 0)} to look at"
            + (f", spot-check {ps['spot']}" if ps.get("spot") else "")
        )
    kept = keep(cfg, folder, n, run_header(folder))
    if kept:
        lines.append(f"kept in {kept}")
    return lines, 0 if report["result"] == "PASS" else 1
