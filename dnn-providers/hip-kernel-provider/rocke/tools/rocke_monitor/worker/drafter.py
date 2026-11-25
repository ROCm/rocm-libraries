# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tool-loop drafter: a role's model (rkmon_lib/bindings.py: the drafter, or the shadow drafter) writes the first-pass
review of one PR in the post format the verifier reads (review_agent.md), with evidence for every claim (evidence.jsonl,
runs.jsonl). It is the program of the draft stages "tool-loop" and "shadow" (rkmon_lib/draft_stage.py).

A draft runs in phases, the steps of its stage's contract (METHOD "phased"):
- settle (script, before any model): the scripted checks (the merge with develop, rule_scan, the CI row), the prior
  items, and the ones that need no model settled with evidence (prior_settle.py: approvals and the Threads row from
  data, items unchanged since our last review, closed rows that cite no file); prior_leads.md lists them at once;
- prior (model): one short task per prior item the script left open, at most drafter.prior_task_steps requests each,
  drafter.prior_parallel at once through the shared limiter, within drafter.prior_wall_min and
  drafter.prior_token_budget. A task's card holds the item and the code it cites; the task settles it FIXED, OPEN or
  PARTIAL with one piece of evidence (a quote at file:line in a tree, or a run), or SKIP (no review point). Beside the
  tasks the saved probes run (the script's): an item whose probes all give FIXED or REPRODUCES is settled by them, and
  one they do not settle goes to a task;
- hunt (model): net-new findings, at the binding's effort, from the start of the prior phase and at the same time,
  within its own drafter.hunt_wall_min and drafter.hunt_token_budget, whatever the prior phase spends. On a follow-up
  the files whose content changed since our last review (the delta, Δ) come first. A 🔴/🟠 needs its evidence before
  the hunt may finish;
- assemble (script): the post, its prior table (each row's status, its settler and its evidence id), the findings
  table, draft_findings.json.
The run stays within drafter.review_wall_min and drafter.review_token_budget (llm_client.Limiter); a phase stopped by
its own caps or the run's keeps what it recorded, and draft_run.json says why it stopped, with each phase's time,
requests and tokens.

The requests are lean, for an endpoint's per-minute limits: each starts from the repository profile's rules card and a
task card of at most about card_tokens; the PR map lists every changed file and hunk, and the model fetches hunks on
demand (diff_hunk), never the whole diff; tool results older than the last keep_results are masked in the history each
hunt request sends; the hunt's wrap-up turns, when its rounds run out, ask at mechanical_effort. Findings, prior
statuses and evidence are tool calls (add_finding, record_evidence, settle), checked at once: a wrong quote is refused,
so the model can fix it. Local work stays git and grep (review_tools.py).

What is specific to the repository comes from its profile, drafter.json and its rules card in the configuration's
profile_dir: the folder of the repository defaults for the configuration's repository and folder, where the package
has them, else none, which gives the generic profile (PROFILE_DEFAULTS and GENERIC_RULES). The profile says how the
code under review is named, the rules, the bots' logins, the logins shown under one name, the texts that hold no review
point (skip_rx), the data files the PR map lists without hunks, and the checks job's note.

Evidence: every tool run gets an id (R1, R2, ...) and a record in runs.jsonl (the tool, its arguments, the argv, the
exit code, the seconds, a digest of its output reduced to what must match on a re-run, and an excerpt). A claim's
evidence is a run, a quote of whole lines at file:line in one of the PR's trees (head, merged, base) or in the staged PR
data (staged), the blobs of its cited files (unchanged), or a fact parsed from the staged data (data); a claim left
without evidence gets a "none" record. The claims: N<k> findings, P<k> prior rows (P<k> is item I<k>'s row), C<k> the
points of the data rows (Threads, Bots / CI, Our checks). `rk evidence-check` checks it all again
(rkmon_lib/evidence.py) and writes prior_settled.md.

Outputs in the draft's folder: review_agent.md, prior_leads.md, evidence.jsonl, runs.jsonl, draft_findings.json (the
findings, prior rows and check claims, for the evidence check and the assessment), probes.txt (the probes' lines, which
the checks job reuses on the same tree), draft_run.json (the run's header and its phases), tokens_total.json (requests,
tokens, waits and wall time per attempt and per phase), agent_log.txt (every request and tool call) and
stats_agent.json.

Usage: python3 -B drafter.py <folder> <PR number> [--trees DIR] [--round First-round|Follow-up] [--line1 TEXT]
       [--role drafter|shadow_drafter] [--prio] [--config FILE]
--trees: the folder holding wt<N> and wt<N>m (default: config paths.root); --round and --line1 override pr/meta.md's
`Round:` and `Line 1:`; --role: the role whose binding drafts (default drafter); --prio ranks this run first at the
limiter (a develop fix or one of the user's priorities; the job's RKMON_DRAFT_PRIO=1 does the same); --config: a JSON
file whose keys replace the configuration's. Exit 0 with a draft written; 1 without one; REFUSED, exit 4, when the
configuration or the --config file cannot be read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import sys
import threading
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
MONITOR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MONITOR))

import llm_client as lc  # noqa: E402
import prior_settle as ps  # noqa: E402
import review_tools as rt  # noqa: E402

METHOD = "phased"  # how this program drafts; the assessment scores each method apart
light_of = ps.light_of
SEV_ORDER = {"🔴": 0, "🟠": 1, "🟡": 2, "🟢": 3, "⚪": 4, "": 5}
SERIOUS = ps.SERIOUS
TASK_STATUSES = ("FIXED", "OPEN", "PARTIAL", "SKIP")
UTILITY = {"🟢": "Positive", "🟡": "Mixed", "🔴": "Negative"}
MAX_FINDINGS = 8
PROFILE_DEFAULTS = {
    "project": "the repository",
    "rules_card": "drafter_rules.md",
    "bot_logins": r"\[bot\]$",
    "group_logins": {},
    "skip_rx": {},
    "data_paths": [],
    "data_suffixes": r"\.(json|ya?ml|csv)$",
    "data_min_lines": 300,
    "checks_note": "build and tests: the checks job's",
    "tracker_keys": [],
}
GENERIC_RULES = (
    "RULES\n- Every claim needs evidence you produced with your tools, recorded with record_evidence.\n"
    "- Net-new findings, scope first: every blocker needed to merge, each lit, no cap; significant follow-ups "
    "(not blocking) only; no nits. What others raised is a prior item.\n"
    "- Terse fragments; no links, no performance figures.\n"
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _append(path: pathlib.Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _cell(text, cap: int = 160) -> str:
    """Text for one table cell: one line, no pipe, at most `cap` characters."""
    t = re.sub(r"\s+", " ", str(text or "")).replace("|", "/").strip()
    return t[: cap - 1] + "…" if len(t) > cap else t


def _mmdd(ts: str) -> str:
    m = re.match(r"\d{4}-(\d\d)-(\d\d)", ts or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def load_profile(cfg: dict) -> dict:
    """The repository profile: drafter.json in the configuration's profile_dir over PROFILE_DEFAULTS, and its rules
    card's text; a blank profile_dir gives PROFILE_DEFAULTS and GENERIC_RULES alone."""
    given = str(cfg.get("profile_dir") or "").strip()
    d = pathlib.Path(os.path.expanduser(given)) if given else None
    try:
        raw = json.loads((_read(d / "drafter.json") if d else "") or "{}")
    except ValueError:
        raw = {}
    p = dict(
        PROFILE_DEFAULTS, **{k: v for k, v in raw.items() if not k.startswith("_")}
    )
    # The tracker's keys are the site's: the configuration's tracker_keys and JIRA project join the profile's.
    site = [
        str(k).strip().upper()
        for k in [
            *(cfg.get("tracker_keys") or []),
            (cfg.get("jira") or {}).get("project"),
        ]
        if k and str(k).strip()
    ]
    p["tracker_keys"] = list(dict.fromkeys([*(p.get("tracker_keys") or []), *site]))
    p["rules"] = (_read(d / str(p["rules_card"])) if d else "").strip() or GENERIC_RULES
    p["dir"] = str(d) if d else ""
    return p


# --- the staged PR data -----------------------------------------------------------------------------------------------


def parse_meta(text: str) -> dict:
    """pr/meta.md: number, title, author, head, base, mergeable, round, line 1, description."""
    out = {
        "title": "",
        "author": "",
        "head": "",
        "base": "",
        "mergeable": "",
        "round": "",
        "line1": "",
        "description": "",
    }
    m = re.search(r"^# PR #(\d+): (.*)$", text, re.M)
    if m:
        out["n"], out["title"] = m.group(1), m.group(2).strip()
    for key, rx in (
        ("author", r"^- Author: @?(\S+)"),
        ("head", r"^- Head: .*?@ ([0-9a-f]+); base: .*?@ ([0-9a-f]+)"),
        ("mergeable", r"^- State: .*?mergeable: (\S+)"),
        ("round", r"^- Round: (.+)$"),
        ("line1", r"^- Line 1: (.+)$"),
    ):
        m = re.search(rx, text, re.M)
        if m:
            out[key] = m.group(1).strip()
            if key == "head":
                out["base"] = m.group(2)
    from rkmon_lib.post import META_DESCRIPTION

    d = text.split(META_DESCRIPTION, 1)
    out["description"] = d[1].strip() if len(d) == 2 else ""
    return out


def strip_type(title: str) -> str:
    """A title without its `type(scope):` prefix."""
    return re.sub(r"^[A-Za-z]+(\([^)]*\))?!?:\s*", "", title or "").strip()


def parse_sections(text: str, head_rx: str) -> list[tuple[re.Match, str]]:
    """(the heading's match, the body until the next heading) for each `## ` heading matching head_rx."""
    out = []
    for part in re.split(r"(?m)^(?=## )", text):
        first, _, body = part.partition("\n")
        m = re.match(head_rx, first)
        if m:
            out.append((m, body.strip()))
    return out


def prior_items(pr: pathlib.Path, bots: str) -> tuple[list[dict], list[dict]]:
    """The other reviewers' and bots' items, from the staged reviews, threads and comments, oldest first: (items,
    approvals). Each item: kind, who, date, at (its first comment), last (its latest reply), state, where, text (the
    first comment), replies. Each approval: who, commit, date, at, and its heading line in reviews.md (line, quote).
    """
    bot = re.compile(bots, re.I)
    items, approvals = [], []
    reviews = _read(pr / "reviews.md")
    lines = reviews.splitlines()
    for m, body in parse_sections(reviews, r"## @(\S+) (\w+) at (\S+) \((\S+)\)"):
        who, state, commit, at = m.groups()
        if state == "APPROVED":
            head = m.group(0).strip()
            line = next((i for i, x in enumerate(lines, 1) if x.strip() == head), 0)
            approvals.append(
                {
                    "who": who,
                    "commit": commit,
                    "date": _mmdd(at),
                    "at": at,
                    "line": line,
                    "quote": head,
                }
            )
            continue
        if state == "DISMISSED" or (state == "COMMENTED" and body in ("", "(no body)")):
            continue
        items.append(
            {
                "kind": "review",
                "who": who,
                "state": state,
                "date": _mmdd(at),
                "at": at,
                "last": at,
                "where": "",
                "text": body,
                "replies": [],
            }
        )
    for m, body in parse_sections(
        _read(pr / "threads.md"), r"## (\S+?):(\S+) \[([^\]]+)\]"
    ):
        comments = [
            (c.group(1), c.group(2), c.group(3).strip())
            for c in re.finditer(
                r"(?ms)^### @(\S+) \((\S+)\)\n(.*?)(?=^### @|\Z)", body
            )
        ]
        if not comments:
            continue
        who, at, text = comments[0]
        items.append(
            {
                "kind": "thread",
                "who": who,
                "state": m.group(3),
                "date": _mmdd(at),
                "at": at,
                "last": max(c[1] for c in comments),
                "where": f"{m.group(1)}:{m.group(2)}",
                "text": text,
                "replies": [
                    {"who": c[0], "date": _mmdd(c[1]), "at": c[1], "text": c[2]}
                    for c in comments[1:]
                ],
            }
        )
    for m, body in parse_sections(_read(pr / "comments.md"), r"## @(\S+) \((\S+)\)"):
        who, at = m.groups()
        if bot.search(who):
            continue
        items.append(
            {
                "kind": "comment",
                "who": who,
                "state": "",
                "date": _mmdd(at),
                "at": at,
                "last": at,
                "where": "",
                "text": body,
                "replies": [],
            }
        )
    items.sort(key=lambda i: i["at"])
    return items, approvals


def parse_commits(text: str) -> list[tuple[str, str, str]]:
    """pr/commits.md: (sha8, date, subject), oldest first."""
    return [
        (m.group(1), m.group(2), m.group(3))
        for m in re.finditer(r"^- ([0-9a-f]{7,40}) (\S+) (.*)$", text, re.M)
    ]


def parse_checks(text: str) -> dict:
    """pr/checks.txt: the check runs' conclusions: counts, failing and running names with their lines."""
    out = {
        "success": 0,
        "failing": [],
        "running": [],
        "cancelled": 0,
        "total": 0,
        "lines": {},
    }
    for i, line in enumerate(text.splitlines(), 1):
        cells = line.split("\t")
        if i == 1 or len(cells) < 2:
            continue
        concl, name = cells[0].strip(), cells[1].strip()
        out["total"] += 1
        out["lines"].setdefault(concl, i)
        if concl == "success":
            out["success"] += 1
        elif concl in ("failure", "timed_out", "action_required", "startup_failure"):
            out["failing"].append((name, i))
        elif concl in ("queued", "in_progress", "pending", "waiting", "requested"):
            out["running"].append((name, i))
        elif concl == "cancelled":
            out["cancelled"] += 1
    return out


def parse_mkwt(text: str) -> dict:
    """mkwt.log: head, develop, merge base, behind/ahead, the conflicting files."""
    out = {
        "head": "",
        "develop": "",
        "merge_base": "",
        "behind": "",
        "ahead": "",
        "conflicts": [],
    }
    m = re.search(r"head=(\S+) develop=(\S+) merge-base=(\S+)", text)
    if m:
        out["head"], out["develop"], out["merge_base"] = m.groups()
    m = re.search(r"behind/ahead:\s*(\d+)\s+(\d+)", text)
    if m:
        out["behind"], out["ahead"] = m.groups()
    m = re.search(r"conflicting files vs develop:\n(.*?)(?:\n\n|\Z)", text, re.S)
    if m:
        out["conflicts"] = [x.strip() for x in m.group(1).splitlines() if x.strip()]
    return out


def tracker_tag(keys, *texts) -> str:
    """The PR's own tracker tag (KEY-NNNN, KEY one of the profile's tracker_keys): the first in its title, body or
    commits; "" for none."""
    if not keys:
        return ""
    rx = re.compile(r"\b(?:" + "|".join(re.escape(str(k)) for k in keys) + r")-\d+\b")
    for t in texts:
        m = rx.search(t or "")
        if m:
            return m.group(0)
    return ""


# --- the PR map -------------------------------------------------------------------------------------------------------

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def parse_diff(
    diff: str, data_paths=(), data_suffixes: str = r"$^", data_min_lines: int = 300
) -> list[dict]:
    """The diff's files, each with its hunks: [{id, path, plus, minus, deleted, data, hunks: [{id, old, new, plus,
    minus, ctx, lines}]}]. A file is data when a data path matches it, or a data suffix does and its change reaches
    data_min_lines."""
    paths_rx = [re.compile(p) for p in data_paths or []]
    suffix_rx = re.compile(data_suffixes)
    files = []
    for part in re.split(r"(?m)^(?=diff --git )", diff or ""):
        if not part.startswith("diff --git "):
            continue
        lines = part.split("\n")
        m = re.match(r"diff --git a/(\S+) b/(\S+)", lines[0])
        path = m.group(2) if m else lines[0][11:].strip()
        f = {
            "path": path,
            "plus": 0,
            "minus": 0,
            "hunks": [],
            "deleted": "\n+++ /dev/null" in part,
        }
        cur = None
        for ln in lines:
            m = HUNK.match(ln)
            if m:
                cur = {
                    "old": int(m.group(1)),
                    "new": int(m.group(3)),
                    "plus": 0,
                    "minus": 0,
                    "ctx": m.group(5).strip(),
                    "lines": [],
                }
                f["hunks"].append(cur)
                continue
            if cur is None or ln.startswith(("+++ ", "--- ")):
                continue
            if ln.startswith("+"):
                cur["plus"] += 1
                f["plus"] += 1
            elif ln.startswith("-"):
                cur["minus"] += 1
                f["minus"] += 1
            elif not ln.startswith((" ", "\\")):
                continue
            cur["lines"].append(ln)
        f["data"] = bool(
            any(r.search(path) for r in paths_rx)
            or (suffix_rx.search(path) and f["plus"] + f["minus"] >= data_min_lines)
        )
        files.append(f)
    for i, f in enumerate(files, 1):
        f["id"] = f"F{i}"
        for j, h in enumerate(f["hunks"], 1):
            h["id"] = f"H{i}.{j}"
    return files


def map_lines(files: list[dict], short, delta=()) -> list[str]:
    """The PR map: one line per file with its hunks (id, new start, +/-, the function the hunk is in); a file of the
    delta is marked Δ and listed first."""
    first, later = [], []
    for f in files:
        mark = " Δ" if f["path"] in delta else ""
        head = f"{f['id']}{mark} {short(f['path'])} +{f['plus']}/-{f['minus']}"
        if f["deleted"]:
            line = head + " (deleted)"
        elif f["data"]:
            line = (
                head
                + f" (data, {len(f['hunks'])} hunks; diff_hunk('{f['id']}') shows the first)"
            )
        else:
            hs = "; ".join(
                f"{h['id']} @{h['new']}+{h['plus']}/-{h['minus']}"
                + (f" {_cell(h['ctx'], 40)}" if h["ctx"] else "")
                for h in f["hunks"]
            )
            line = f"{head}: {hs}"
        (first if mark or not delta else later).append(line)
    return first + later


def fit(lines: list[str], budget: int, what: str, how: str) -> list[str]:
    """As many lines as fit in `budget` characters, then one line saying how many were left out."""
    out, used = [], 0
    for i, ln in enumerate(lines):
        if used + len(ln) + 1 > budget:
            out.append(f"[{len(lines) - i} more {what} not shown: {how}]")
            break
        out.append(ln)
        used += len(ln) + 1
    return out


def _where(where: str) -> tuple[str, int]:
    """`path:line` (a thread's, a finding's) as (path, line); line 0 when it has none."""
    path, _, line = (where or "").rpartition(":")
    return (path, int(line)) if path and line.isdigit() else (where or "", 0)


# --- the session ------------------------------------------------------------------------------------------------------


class Session:
    """One draft of one PR: its trees, data, settings, limiter, records and outputs. Its phases run in threads; the
    records are shared under a lock, and each thread has its own pass, step and task."""

    def __init__(
        self,
        ws,
        n,
        cfg: dict,
        s: dict,
        trees=None,
        round_word: str = "",
        line1: str = "",
        prio: bool = False,
        chat=None,
        clock=time.time,
        profile: dict | None = None,
        role: str = "drafter",
    ):
        self.ws, self.n = pathlib.Path(ws), str(n)
        self.cfg, self.s = cfg, dict(s)
        self.role = role
        self.clock = clock
        self.profile = profile or load_profile(cfg)
        from rkmon_lib import paths

        root = (cfg.get("paths") or {}).get("root") or str(MONITOR.parent)
        self.t = rt.Tree(self.ws, n, root, paths.repository_folder(cfg), trees=trees)
        self.pr = self.ws / "pr"
        self.meta = parse_meta(_read(self.pr / "meta.md"))
        self.round = round_word or self.meta.get("round") or "First-round"
        self.line1 = (
            line1 or self.meta.get("line1") or cfg.get("review_first_line") or ""
        )
        self.chat_fn = chat or lc.chat
        self.limiter = lc.limiter_for(
            self.s, who=f"pr{n}" + ("" if role == "drafter" else f"/{role}"), prio=prio
        )
        self._lock = threading.RLock()
        self._tl = threading.local()
        self.log = None
        self.runs: dict[str, dict] = {}
        self.evidence: list[dict] = []
        self.findings: dict[str, dict] = {}
        self.rows: dict[str, dict] = {}  # an item's id (I<k>) -> its prior row (P<k>)
        self.skipped: dict[str, str] = {}
        self.finish: dict = {}
        self.stopped = ""
        self.checks: dict[str, str] = {}  # C<k> -> the data row it is a point of
        self.merge: dict = {}
        self.rules: dict = {}
        self.threads: dict = {}
        self.answered = 0  # model answers in this run
        self.commits = parse_commits(_read(self.pr / "commits.md"))
        self.others, self.approvals = prior_items(self.pr, self.profile["bot_logins"])
        self.ours = ps.parse_ours(_read(self.pr / "ours.md"))
        self.items: list[dict] = []
        self.files = parse_diff(
            _read(self.pr / "diff.patch"),
            self.profile["data_paths"],
            self.profile["data_suffixes"],
            int(self.profile["data_min_lines"]),
        )
        self.mk = parse_mkwt(_read(self.ws / "mkwt.log"))
        self.ci = parse_checks(_read(self.pr / "checks.txt"))
        self.header: dict = {}
        self.head_sha = self.then_head = self.dev_then = self.dev_now = ""
        self.delta: list[str] = (
            []
        )  # the PR's files whose content changed since our last review's head
        self.probe_names: list[str] = []
        self.probe_tree = "head"
        self.prior_deadline = 0.0
        self.system = (
            f"You are the first-pass reviewer of one pull request to {self.profile['project']}. You work "
            "only through your tools: they read the PR's trees (head; merged with develop; the merge base), "
            "its diff and its staged review data. A stronger reviewer verifies your work, so precision "
            "beats volume: one real defect with evidence is worth more than five guesses.\n\n"
            + self.profile["rules"]
        )

    # --- each thread's own: its pass, its step, whether its pass is done, its prior task --------------------------------
    def _get(self, name, default):
        return getattr(self._tl, name, default)

    pass_ = property(
        lambda self: self._get("pass_", ""),
        lambda self, v: setattr(self._tl, "pass_", v),
    )
    step = property(
        lambda self: self._get("step", 0), lambda self, v: setattr(self._tl, "step", v)
    )
    pass_done = property(
        lambda self: self._get("pass_done", False),
        lambda self, v: setattr(self._tl, "pass_done", v),
    )
    task = property(
        lambda self: self._get("task", None),
        lambda self, v: setattr(self._tl, "task", v),
    )

    def stop(self, why: str) -> None:
        """The run's own caps were reached: what started stops, and what has not started does not start."""
        with self._lock:
            self.stopped = self.stopped or why

    # --- records ------------------------------------------------------------------------------------------------------
    def start(self) -> None:
        """A fresh run in the folder: the outputs of an earlier attempt go (its tokens stay in tokens_total.json)."""
        for name in (
            "review_agent.md",
            "evidence.jsonl",
            "runs.jsonl",
            "draft_findings.json",
            "evidence_check.md",
            "evidence_check.json",
            "stats_agent.json",
            "prior_leads.md",
            "prior_settled.md",
            "prior_settled.json",
            "probes.txt",
        ):
            (self.ws / name).unlink(missing_ok=True)
        try:
            old = json.loads(_read(self.ws / "draft_run.json") or "{}")
        except ValueError:
            old = {}
        self.head_sha = self.t.sha("head")
        self.header = {
            "pr": int(self.n),
            "round": self.round,
            "line1": self.line1,
            "model": self.s.get("model"),
            "binding": self.s.get("binding"),
            "role": self.role,
            "method": METHOD,
            "head": self.head_sha,
            "merged": self.t.sha("merged"),
            "base": self.t.sha("base"),
            "trees": str(self.t.trees),
            "ws": str(self.ws),
            "started": _now(),
            "passes": {},
            "attempt": int(old.get("attempt") or 0) + 1,
            "stopped": "",
            "draft": False,
        }
        self.save_header()
        self.log = open(self.ws / "agent_log.txt", "a", encoding="utf-8")
        self.logw(
            f"=== draft of #{self.n} ({self.round}), attempt {self.header['attempt']}, {_now()}, model "
            f"{self.s.get('name') or self.s.get('model')} ({self.role}), head {self.head_sha[:10]} ==="
        )

    def save_header(self) -> None:
        with self._lock:
            (self.ws / "draft_run.json").write_text(
                json.dumps(self.header, indent=1), encoding="utf-8"
            )

    def set_pass(self, name: str, rec: dict) -> None:
        with self._lock:
            self.header["passes"][name] = rec
        self.save_header()

    def logw(self, text: str) -> None:
        with self._lock:
            if self.log:
                self.log.write(text + "\n")
                self.log.flush()

    def record_run(
        self, name: str, args: dict, argv: list, r: rt.Result, extra: dict | None = None
    ) -> str:
        norm = rt.normalize(name, r.out, self.t)
        with self._lock:
            rid = f"R{len(self.runs) + 1}"
            rec = {
                "run": rid,
                "pass": self.pass_,
                "tool": name,
                "args": args,
                "argv": [str(a) for a in argv],
                "rc": r.rc,
                "secs": r.secs,
                "digest": rt.digest(norm),
                "norm": norm[:2000],
                "excerpt": rt.excerpt(r.out),
                "head": self.header.get("head"),
                "merged": self.header.get("merged"),
                "at": _now(),
                **(extra or {}),
            }
            self.runs[rid] = rec
            _append(self.ws / "runs.jsonl", rec)
        return rid

    def add_evidence(self, rec: dict) -> None:
        with self._lock:
            self.evidence.append(rec)
            _append(self.ws / "evidence.jsonl", rec)

    def evidenced(self) -> set[str]:
        with self._lock:
            return {
                e["id"] for e in self.evidence if e.get("kind") not in ("none", None)
            }

    def run(self, name: str, args: dict) -> tuple[str, str]:
        """A tool run: (its id, or "" for a refusal; the text the model sees)."""
        text, r, argv = rt.run_tool(self.t, name, args)
        if r is None:
            return "", text
        rid = self.record_run(name, args, argv, r)
        if name == "merge_conflicts" and r.rc == 0 and not r.out.strip():
            text = "(no conflicts)"
        return rid, f"[{rid}] {text}"

    def ran_rec(self, cid: str, light: str, claim: str, rid: str, pass_: str) -> dict:
        r = self.runs[rid]
        return {
            "id": cid,
            "pass": pass_,
            "severity": light,
            "claim": _cell(claim, 300),
            "head": self.head_sha,
            "at": _now(),
            "kind": "ran",
            "run": rid,
            "tool": r["tool"],
            "args": r["args"],
            "command": " ".join(r["argv"])[:600],
            "rc": r["rc"],
            "digest": r["digest"],
            "excerpt": r["excerpt"][:600],
        }

    # --- the prior rows -----------------------------------------------------------------------------------------------
    def shown_login(self, who: str) -> str:
        for rx, name in (self.profile.get("group_logins") or {}).items():
            if re.search(rx, who, re.I):
                return name
        return who

    def prior_key(self, its: list[dict], gist: str) -> str:
        whos = []
        for i in its:
            w = self.shown_login(i["who"])
            if w not in whos:
                whos.append(w)
        date = min((i["date"] for i in its if i["date"]), default="")
        k = f" ×{len(its)}" if len(its) > 1 else ""
        return _cell(
            (
                f"{', '.join(whos)} ({date}){k}: {gist}"
                if gist
                else f"{', '.join(whos)} ({date}){k}"
            ),
            150,
        )

    def item_light(self, it: dict) -> str:
        """The light an open item carries: our last review's when it was 🔴/🟠/🟡, else the first in its text;
        🟡 else."""
        prev = (it.get("prev") or {}).get("light")
        if prev in ("🔴", "🟠", "🟡"):
            return prev
        lt = light_of(it.get("text"))
        return lt if lt in ("🔴", "🟠", "🟡") else "🟡"

    def make_row(
        self,
        it: dict,
        light: str,
        status: str,
        sha: str,
        points: list[str],
        settler: str,
        how: str,
        ref: str = "",
        gist: str = "",
    ) -> dict:
        """Item I<k>'s row P<k>: its key (our last review's own, else who, date and gist), status, settler (script,
        model, or "" for none) and how it settled, its evidence's short reference, and whether it flips a 🔴/🟠.
        """
        prev = it.get("prev") or {}
        key = it.get("key") or self.prior_key([it], gist or ps.gist(it.get("text")))
        if it["kind"] == "approval":
            key = f"{self.shown_login(it['who'])} ({it['date']}) APPROVED at `{it['commit'][:8]}`"
        row = {
            "id": "P" + it["id"][1:],
            "item": it["id"],
            "key": _cell(key, 150),
            "light": light,
            "status": status,
            "sha": (sha or "")[:10],
            "points": [_cell(p, 140) for p in points if str(p).strip()][:4],
            "settler": settler,
            "how": how,
            "ref": ref,
            "prev_light": prev.get("light", ""),
            "prev_status": prev.get("status", ""),
            "flip": bool(settler)
            and ps.flips(prev.get("light", ""), prev.get("status", ""), light, status),
        }
        with self._lock:
            self.rows[it["id"]] = row
        return row

    def unsettled(self, it: dict, why: str) -> None:
        prev = it.get("prev") or {}
        was = (
            [f"our last review: {prev['light']} {prev['status']}".strip()]
            if prev.get("status")
            else []
        )
        self.make_row(it, "⚪", "NOT SETTLED", "", [why] + was, "", "")

    # --- the tools ----------------------------------------------------------------------------------------------------
    def t_diff_hunk(self, id: str = "", **kw) -> str:
        key = str(id or kw.get("path") or "").strip()
        f = next(
            (
                f
                for f in self.files
                if key in (f["id"], f["path"], self.t.short(f["path"]))
            ),
            None,
        )
        hunks = f["hunks"] if f else []
        if not f:
            m = re.fullmatch(r"H(\d+)\.(\d+)", key)
            if m and 1 <= int(m.group(1)) <= len(self.files):
                f = self.files[int(m.group(1)) - 1]
                hunks = [h for h in f["hunks"] if h["id"] == key]
        if not f or not hunks:
            return f"no file or hunk {key!r}: use an id of the PR map (F3, H3.2) or a changed file's path"
        out, shown = [], 0
        for h in hunks:
            body = self.hunk_text(h)
            if shown and shown + len(body) > rt.OUT_CAP * 2:
                rest = [x["id"] for x in hunks[hunks.index(h) :]]
                out.append(f"[not shown: {', '.join(rest)}: ask for them by id]")
                break
            out.append(
                f"{self.t.short(f['path'])} {h['id']} @@ -{h['old']} +{h['new']} @@ {h['ctx']}\n{body}"
            )
            shown += len(body)
        return "\n".join(out)

    @staticmethod
    def hunk_text(h: dict, cap: int = 250) -> str:
        """A hunk with the head's line numbers on context and added lines."""
        n, rows = h["new"], []
        for ln in h["lines"][:cap]:
            if ln.startswith("+"):
                rows.append(f"{n:>6}+ {ln[1:]}")
                n += 1
            elif ln.startswith("-"):
                rows.append(f"{'':>6}- {ln[1:]}")
            elif ln.startswith("\\"):
                rows.append(f"{'':>6}  {ln}")
            else:
                rows.append(f"{n:>6}  {ln[1:]}")
                n += 1
        if len(h["lines"]) > cap:
            rows.append(
                f"[{len(h['lines']) - cap} more lines of this hunk: read_lines from {n}]"
            )
        return "\n".join(rows)

    def t_read_lines(
        self, path: str = "", start=1, end=0, tree: str = "head", **_
    ) -> str:
        return rt.read_lines(self.t, path, start, end, tree or "head")

    def t_grep_code(self, pattern: str = "", path: str = ".", **_) -> str:
        return self.run(
            "grep_code", {"pattern": str(pattern), "path": str(path or ".")}
        )[1]

    def t_git_log(self, path: str = "", **_) -> str:
        return self.run("git_log", {"path": str(path or "")})[1]

    def t_read_prior(self, id: str = "", **_) -> str:
        it = next((i for i in self.items if i["id"] == str(id).strip().upper()), None)
        if not it:
            return f"no item {id!r}: the items are I1 to I{len(self.items)}"
        text = [
            f"{it['id']} {it['kind']} by {it['who']} ({it['date']}) {it.get('state') or ''} "
            f"{it.get('where') or ''}".strip(),
            it["text"],
        ]
        text += [
            f"-- reply by {r['who']} ({r['date']}): {r['text']}" for r in it["replies"]
        ]
        return rt.clip("\n".join(text), 7000)

    def quote_ok(self, file: str, line, quote: str, tree: str) -> tuple[bool, str]:
        try:
            n = int(line or 0)
        except (TypeError, ValueError):
            return False, "REJECTED: line is a line number"
        if not str(quote or "").strip():
            return False, "REJECTED: quote the line(s) exactly"
        if len(str(quote)) > rt.QUOTE_CAP:
            return (
                False,
                f"REJECTED: a quote is at most {rt.QUOTE_CAP} characters: quote fewer lines",
            )
        if tree == "staged":
            name = pathlib.PurePosixPath(str(file)).name
            if (
                not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or not (self.pr / name).is_file()
            ):
                return (
                    False,
                    "REJECTED: tree staged reads the PR's staged files (checks.txt, meta.md, ...)",
                )
            lines = _read(self.pr / name).splitlines()
        else:
            lines = rt.file_lines(self.t, file, tree)
            if isinstance(lines, str):
                return False, f"REJECTED: {lines}"
        ok, where = rt.quote_at(lines, n, quote)
        if ok:
            return True, ""
        said = (
            lines[n - 1].strip()[:160]
            if 1 <= n <= len(lines)
            else f"(the file has {len(lines)} lines)"
        )
        return False, (
            f"REJECTED: the quote is not at {file}:{n} ({tree}); that line reads {said!r}"
            + (f"; the quote is at line {where}" if where else "")
            + ". Quote whole lines exactly."
        )

    def evidence_rec(
        self,
        cid: str,
        sev: str,
        claim: str,
        run: str = "",
        file: str = "",
        line=0,
        quote: str = "",
        tree: str = "head",
    ) -> tuple[dict | None, str]:
        """A claim's evidence as the model gave it, checked: (the record, "") or (None, why it is refused)."""
        rec = {
            "id": cid,
            "pass": self.pass_,
            "severity": sev,
            "claim": _cell(claim, 300),
            "head": self.head_sha,
            "at": _now(),
        }
        if run:
            with self._lock:
                r = self.runs.get(str(run).strip().upper())
                known = ", ".join(list(self.runs)[-8:]) or "none"
            if not r:
                return None, f"REJECTED: no run {run!r} in this draft (runs: {known})"
            rec.update(
                kind="ran",
                run=r["run"],
                tool=r["tool"],
                args=r["args"],
                command=" ".join(r["argv"])[:600],
                rc=r["rc"],
                digest=r["digest"],
                excerpt=r["excerpt"][:600],
            )
            return rec, ""
        if file:
            tree = tree or "head"
            ok, why = self.quote_ok(file, line, quote, tree)
            if not ok:
                return None, why
            rel = (
                pathlib.PurePosixPath(str(file)).name
                if tree == "staged"
                else self.t.resolve(file)
            )
            rec.update(
                kind="read",
                file=rel,
                line=int(line),
                quote=str(quote),
                tree=tree,
                sha=(
                    ""
                    if tree == "staged"
                    else self.header[
                        (
                            "merged"
                            if tree == "merged"
                            else "base" if tree == "base" else "head"
                        )
                    ]
                ),
            )
            return rec, ""
        return None, "REJECTED: give run (a run id such as R3), or file, line and quote"

    @staticmethod
    def ref_of(rec: dict) -> str:
        """An evidence record's short reference for the table: its run, its quote's place, its unchanged files."""
        if rec.get("kind") == "ran":
            return f"{rec['run']} {rec['tool']}"
        if rec.get("kind") == "read":
            return f"{pathlib.PurePosixPath(rec['file']).name}:{rec['line']} ({rec['tree']})"
        if rec.get("kind") == "unchanged":
            return f"unchanged since `{rec['since'][:10]}`"
        if rec.get("kind") == "data":
            return f"{rec['file']} (data)"
        return "none"

    def t_record_evidence(
        self,
        id: str = "",
        claim: str = "",
        run: str = "",
        file: str = "",
        line=0,
        quote: str = "",
        tree: str = "head",
        severity: str = "",
        **_,
    ) -> str:
        cid = str(id or "").strip().upper()
        if not re.fullmatch(r"N\d{1,3}", cid):
            return "REJECTED: id is one of your findings' (N1)"
        with self._lock:
            f = self.findings.get(cid)
        if not f:
            return f"REJECTED: no finding {cid}: add_finding first"
        rec, why = self.evidence_rec(
            cid,
            light_of(severity) or f["severity"],
            claim,
            run,
            file,
            line,
            quote,
            tree,
        )
        if not rec:
            return why
        self.add_evidence(rec)
        return f"recorded for {cid}"

    def t_add_finding(
        self,
        severity: str = "",
        file: str = "",
        line=0,
        problem: str = "",
        impact: str = "",
        fix: str = "",
        quote: str = "",
        **_,
    ) -> str:
        sev = light_of(severity)
        if sev not in ("🔴", "🟠", "🟡"):
            return "REJECTED: severity is 🔴, 🟠 or 🟡 (red, orange, yellow)"
        with self._lock:
            live = [f for f in self.findings.values() if not f["dropped"]]
        if len(live) >= MAX_FINDINGS:
            return f"REJECTED: {MAX_FINDINGS} findings already: drop a weaker one first (drop_finding)"
        rel = self.t.resolve(file)
        lines = rt.file_lines(self.t, rel, "head")
        if isinstance(lines, str):
            return f"REJECTED: {lines} (a finding is at file:line in the head tree)"
        try:
            n = int(line or 0)
        except (TypeError, ValueError):
            n = 0
        if not 1 <= n <= len(lines):
            return f"REJECTED: line {line!r} is not in {file} ({len(lines)} lines)"
        if not str(problem or "").strip():
            return "REJECTED: problem: one short clause"
        with self._lock:
            fid = f"N{len(self.findings) + 1}"
            self.findings[fid] = {
                "id": fid,
                "severity": sev,
                "file": rel,
                "line": n,
                "problem": _cell(problem, 140),
                "impact": _cell(impact, 200),
                "fix": _cell(fix, 160),
                "dropped": False,
            }
        if str(quote or "").strip():
            rec, why = self.evidence_rec(
                fid, sev, problem, file=rel, line=n, quote=quote, tree="head"
            )
            if rec:
                self.add_evidence(rec)
                return f"{fid} added, with its quote as evidence."
            return (
                f"{fid} added; its quote was not recorded: {why} Then record_evidence(id='{fid}', ...) with an "
                f"exact quote of {self.t.short(rel)}:{n}."
            )
        return f"{fid} added. Now record_evidence(id='{fid}', ...) with an exact quote of {self.t.short(rel)}:{n}."

    def t_drop_finding(self, id: str = "", why: str = "", **_) -> str:
        with self._lock:
            f = self.findings.get(str(id).strip().upper())
            if not f:
                return f"no finding {id!r}"
            f["dropped"], f["why"] = True, _cell(why, 200)
        return f"{f['id']} dropped"

    def t_finish(
        self, reason: str = "", utility_light: str = "", utility: str = "", **_
    ) -> str:
        have = self.evidenced()
        with self._lock:
            lacking = [
                f["id"]
                for f in self.findings.values()
                if not f["dropped"] and f["severity"] in SERIOUS and f["id"] not in have
            ]
        if lacking:
            return (
                f"REJECTED: {', '.join(lacking)} has no evidence yet: record_evidence(id=...) for each, or "
                "drop_finding(id=..., why=...); every 🔴/🟠 needs its evidence"
            )
        self.finish = {
            "reason": _cell(reason, 60),
            "light": light_of(utility_light),
            "utility": _cell(utility, 120),
        }
        self.pass_done = True
        return "finished"

    def t_settle(
        self,
        status: str = "",
        light: str = "",
        gist: str = "",
        point: str = "",
        sha: str = "",
        file: str = "",
        line=0,
        quote: str = "",
        tree: str = "head",
        run: str = "",
        **_,
    ) -> str:
        it = self.task
        if not it:
            return "refused: settle belongs to a prior item's task"
        with self._lock:
            done = it["id"] in self.rows or it["id"] in self.skipped
        if done:
            self.pass_done = True
            return "already settled"
        st = str(status or "").strip().upper()
        if st not in TASK_STATUSES:
            return "REJECTED: status is FIXED, OPEN, PARTIAL or SKIP"
        if st == "SKIP":
            with self._lock:
                self.skipped[it["id"]] = "no review point of its own (the task)"
            self.pass_done = True
            return "skipped"
        lt = "🟢" if st == "FIXED" else (light_of(light) or self.item_light(it))
        if st != "FIXED" and lt not in ("🔴", "🟠", "🟡"):
            return "REJECTED: an OPEN or PARTIAL item is 🔴, 🟠 or 🟡"
        sha = str(sha or "").strip().strip("`")[:12]
        if sha and not any(
            c[0].startswith(sha[:7]) or sha.startswith(c[0][:7]) for c in self.commits
        ):
            return f"REJECTED: sha {sha!r} is not one of the PR's commits (the card's list)"
        pt = _cell(point, 120)
        if not pt:
            return "REJECTED: point: the decisive fact in 4 to 10 words, marked (read) or (ran)"
        pid = "P" + it["id"][1:]
        rec, why = self.evidence_rec(
            pid, lt, f"{st}: {pt}", run, file, line, quote, tree
        )
        if not rec:
            return why
        word = (
            "DEFERRED"
            if st == "OPEN" and (it.get("prev") or {}).get("status") == "DEFERRED"
            else st
        )
        self.make_row(
            it,
            lt,
            word,
            sha[:8],
            [pt],
            "model",
            "task",
            self.ref_of(rec),
            gist=_cell(gist, 70),
        )
        self.add_evidence(rec)
        self.pass_done = True
        return f"{pid} settled {word}"

    # --- the tool menus -----------------------------------------------------------------------------------------------
    S = {
        "diff_hunk": (
            "The PR's diff for one hunk (H3.2) or one file (F3, or its path), with the head's line numbers.",
            {"id": "string"},
            ["id"],
        ),
        "read_lines": (
            "Numbered lines of a file to read and quote. tree: head (default), merged (with develop) or "
            "base (the merge base). At most 300 lines.",
            {"path": "string", "start": "integer", "end": "integer", "tree": "string"},
            ["path"],
        ),
        "grep_code": (
            "grep the head tree (alternatives separated by |). path: a folder or file to search in.",
            {"pattern": "string", "path": "string"},
            ["pattern"],
        ),
        "git_log": (
            "The PR's commits since the merge base that touch a path (or all).",
            {"path": "string"},
            [],
        ),
        "read_prior": (
            "The whole text of a prior item, its replies included.",
            {"id": "string"},
            ["id"],
        ),
        "record_evidence": (
            "Evidence for one of your findings (N1): either run (an [R<N>] id), or file, line and quote "
            "(whole lines, exactly) with tree head, merged, base or staged.",
            {
                "id": "string",
                "claim": "string",
                "run": "string",
                "file": "string",
                "line": "integer",
                "quote": "string",
                "tree": "string",
            },
            ["id", "claim"],
        ),
        "add_finding": (
            "Add a net-new finding at file:line in the head tree: severity (🔴 🟠 🟡), problem (one short "
            "clause), impact (the concrete failure), fix (a few words), and quote (the line at file:line, "
            "exactly: it is recorded as the finding's evidence).",
            {
                "severity": "string",
                "file": "string",
                "line": "integer",
                "problem": "string",
                "impact": "string",
                "fix": "string",
                "quote": "string",
            },
            ["severity", "file", "line", "problem", "impact", "fix"],
        ),
        "drop_finding": (
            "Drop one of your findings that did not hold.",
            {"id": "string", "why": "string"},
            ["id"],
        ),
        "finish": (
            "End the hunt: reason (2 to 5 words for the header), utility_light (🟢 positive, 🟡 mixed, 🔴 "
            "negative) and utility (8 to 12 words: what the PR buys against its cost and risk). Every 🔴/🟠 "
            "needs its evidence first.",
            {"reason": "string", "utility_light": "string", "utility": "string"},
            ["reason", "utility_light", "utility"],
        ),
        "settle": (
            "Settle this prior item at head: status FIXED (fixed at head), OPEN (still there), PARTIAL (part "
            "fixed, part left) or SKIP (it has no review point of its own); light 🔴 🟠 🟡 for OPEN and "
            "PARTIAL; gist (at most 10 words); point (the decisive fact, 4 to 10 words, marked (read) or "
            "(ran)); sha (the fixing commit, from the card's list, for FIXED). Evidence for FIXED, OPEN and "
            "PARTIAL: file, line and quote (whole lines, exactly; tree head, merged or base), or run (an [R<N>] "
            "id).",
            {
                "status": "string",
                "light": "string",
                "gist": "string",
                "point": "string",
                "sha": "string",
                "file": "string",
                "line": "integer",
                "quote": "string",
                "tree": "string",
                "run": "string",
            },
            ["status"],
        ),
    }
    MENUS = {
        "prior": (
            "read_prior",
            "diff_hunk",
            "read_lines",
            "grep_code",
            "git_log",
            "settle",
        ),
        "hunt": (
            "diff_hunk",
            "read_lines",
            "grep_code",
            "git_log",
            "add_finding",
            "record_evidence",
            "drop_finding",
            "finish",
        ),
    }
    WRAP = {
        "prior": ("settle",),
        "hunt": ("add_finding", "record_evidence", "drop_finding", "finish"),
    }

    def schemas(self, names) -> list[dict]:
        return [rt.schema(n, self.S[n][0], self.S[n][1], self.S[n][2]) for n in names]

    def tool(self, name: str, args: dict, allowed) -> str:
        if name not in allowed:
            return f"refused: {name} is not available now (available: {', '.join(allowed)})"
        try:
            return getattr(self, f"t_{name}")(**args)
        except TypeError as exc:
            return f"bad arguments for {name}: {exc}"

    # --- the requests -------------------------------------------------------------------------------------------------
    def ask(self, msgs: list, tools: list, effort: str, max_tokens: int) -> dict:
        tags = {"pass": self.pass_, "step": self.step}
        if self.task:
            tags["task"] = self.task["id"]
        self.limiter.tags = tags
        msg = self.chat_fn(
            self.s, msgs, tools, effort, limiter=self.limiter, max_tokens=max_tokens
        )
        with self._lock:
            self.answered += 1
        return msg

    def mask(self, msgs: list) -> None:
        """Every tool result but the last keep_results is replaced by a stub in the history the next request sends;
        the stub names the call and its round, so the model knows what it has read."""
        keep = int(self.s["keep_results"])
        tools = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        for i in tools[: max(0, len(tools) - keep)]:
            m = msgs[i]
            if not m.get("masked"):
                m["content"] = (
                    f"[hidden: {m.get('call') or m.get('tool_name')} from round {m.get('round', '?')}, "
                    f"{len(m['content'])} characters; you read it: call again only if you need its text]"
                )
                m["masked"] = True

    def label(self) -> str:
        return self.task["id"] if self.task else self.pass_

    def call(self, call: dict, allowed, msgs: list) -> None:
        fn = call.get("function") or {}
        name, args = fn.get("name", ""), fn.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        out = (
            "bad arguments: not a JSON object"
            if "_unparsed" in args
            else self.tool(name, args, allowed)
        )
        self.logw(
            f"  [{self.label()}] TOOL {name}({json.dumps(args, ensure_ascii=False)[:200]})"
        )
        self.logw(f"  [{self.label()}] -> {out[:300].rstrip()}")
        shown = ", ".join(
            f"{v}"
            for v in args.values()
            if isinstance(v, (str, int)) and len(str(v)) <= 80
        )
        msgs.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": out,
                "call": f"{name}({shown})",
                "round": self.step,
            }
        )

    def log_step(self, name: str, msg: dict, effort: str, t0: float) -> None:
        u = msg.get("usage") or {}
        fin = msg.get("finish_reason")
        self.logw(
            f"\n=== {name} step {self.step} (t+{self.clock() - t0:.0f}s) effort={effort} "
            f"reasoning={len(msg.get('reasoning') or '')}c calls={len(msg.get('tool_calls') or [])} "
            f"tokens={u.get('prompt', '?')}+{u.get('completion', '?')}"
            + (f" cached={u['cached']}" if u.get("cached") else "")
            + (f" finish={fin}" if fin not in (None, "", "tool_calls", "stop") else "")
            + " ==="
        )

    def stopped_by(self, exc: Exception) -> str:
        """A phase's status from what stopped it; a stop by the run's caps stops the whole run."""
        if isinstance(exc, lc.OverBudget):
            why = f"stopped: {exc}"
            if getattr(exc, "scope", "run") == "run":
                self.stop(why)
            return why
        return f"failed: {type(exc).__name__}: {str(exc)[:200]}"

    # --- Phase 0: the script ------------------------------------------------------------------------------------------
    def scripted_checks(self) -> None:
        """The merge and the rule scan (R1, R2), and the claims C<k> of the Bots / CI and Our checks rows."""
        rid, text = self.run("merge_conflicts", {})
        files = (
            []
            if not rid or "(no conflicts)" in text
            else [x for x in text.split("] ", 1)[-1].splitlines() if x.strip()]
        )
        self.merge = {"rid": rid, "files": files}
        rid2, text2 = self.run("rule_scan", {})
        hits = dict(re.findall(r"^== ([\w ()-]+?): (\d+)", text2, re.M)) if rid2 else {}
        self.rules = {
            "rid": rid2,
            "hits": {k: int(v) for k, v in hits.items() if int(v)},
        }
        if rid:
            self.check_claim(
                "Our checks",
                self.ran_rec(
                    "",
                    "🟠" if files else "🟢",
                    "merge with develop: "
                    + (f"{len(files)} files conflict" if files else "clean"),
                    rid,
                    "script",
                ),
            )
        if rid2:
            self.check_claim(
                "Our checks",
                self.ran_rec(
                    "",
                    "🟡" if self.rules["hits"] else "🟢",
                    "rule_scan: "
                    + (
                        ", ".join(f"{a} {b}" for a, b in self.rules["hits"].items())
                        or "clean"
                    ),
                    rid2,
                    "script",
                ),
            )
        line = (
            self.ci["failing"][0][1]
            if self.ci["failing"]
            else (
                self.ci["running"][0][1]
                if self.ci["running"]
                else self.ci["lines"].get("success")
            )
        )
        if line:
            quote = _read(self.pr / "checks.txt").splitlines()[line - 1]
            self.check_claim(
                "Bots / CI",
                {
                    "pass": "script",
                    "severity": self.ci_light(),
                    "claim": "CI at head: " + self.ci_word(),
                    "head": self.head_sha,
                    "at": _now(),
                    "kind": "read",
                    "file": "checks.txt",
                    "line": line,
                    "quote": quote[: rt.QUOTE_CAP],
                    "tree": "staged",
                    "sha": "",
                },
            )

    def check_claim(self, row: str, rec: dict) -> str:
        """A data row's point C<k>, with its evidence."""
        with self._lock:
            cid = f"C{len(self.checks) + 1}"
            self.checks[cid] = row
        self.add_evidence(dict(rec, id=cid))
        return cid

    def build_items(self) -> None:
        """The prior items: approvals; our last review's rows and findings; the others' items with activity after that
        review (all on a first round). Each gets I<k>, the files it cites (resolved), its skip reason, the comments
        since our last review that name its labels, and its saved probes."""
        items = [
            {
                "kind": "approval",
                "who": a["who"],
                "date": a["date"],
                "at": a["at"],
                "last": a["at"],
                "state": "APPROVED",
                "where": "",
                "text": f"APPROVED at {a['commit']}",
                "replies": [],
                "commit": a["commit"],
                "line": a["line"],
                "quote": a["quote"],
            }
            for a in self.approvals
        ]
        since = (self.ours or {}).get("at") or ""
        if self.ours:
            items += ps.row_items(self.ours)
        later = [
            o for o in self.others if not since or (o.get("last") or o["at"]) > since
        ]
        items += later
        for k, it in enumerate(items, 1):
            it["id"] = f"I{k}"
        resolver = ps.Resolver(
            [f["path"] for f in self.files],
            lambda: rt.tree_files(self.t.wt, "HEAD", self.t.project_path),
            self.t.project_path,
        )
        for it in items:
            if it["kind"] == "approval":
                it.update(files=[], cite_lines=[], unresolved=[])
                continue
            cites = (
                list(it.get("cites") or [])
                if it["kind"] in ("row", "finding")
                else ps.citations(it["text"])
            )
            path, line = (
                _where(it.get("where") or "") if it["kind"] == "thread" else ("", 0)
            )
            if path:
                cites.insert(0, (path, line))
            files, lines, unresolved = [], [], []
            for tok, ln in cites:
                hit = resolver.resolve(tok)
                if not hit:
                    unresolved.append(tok)
                    continue
                files += hit
                if ln and (hit[0], ln) not in lines:
                    lines.append((hit[0], ln))
            it["files"], it["cite_lines"], it["unresolved"] = (
                sorted(set(files)),
                lines,
                sorted(set(unresolved)),
            )
        author = self.meta.get("author") or ""
        for it in items:
            if it["kind"] in ("review", "thread", "comment"):
                why = ps.skip_reason(it, author, self.profile.get("skip_rx") or {})
                if why:
                    self.skipped[it["id"]] = why
        recent = later if since else []
        for it in items:
            if it["kind"] in ("row", "finding") and it.get("labels"):
                said = []
                for o in recent:
                    text = " ".join([o["text"]] + [r["text"] for r in o["replies"]])
                    if any(
                        re.search(rf"\b{re.escape(lab)}\b", text)
                        for lab in it["labels"]
                    ):
                        said.append(o)
                it["mentions"] = said
        self.probe_names = self.t.probe_names()
        by_label: dict[str, list[str]] = {}
        for name in self.probe_names:
            by_label.setdefault(ps.probe_label(name), []).append(name)
        for it in items:
            labs = it.get("labels") or []
            if (
                it["kind"] in ("row", "finding")
                and labs
                and all(lab in by_label for lab in labs)
            ):
                it["probes"] = sorted({name for lab in labs for name in by_label[lab]})
        self.items = items

    def settle_data(self) -> None:
        """The data rows: each approval (its line in reviews.md), and the Threads row (threads.md)."""
        for it in self.items:
            if it["kind"] != "approval":
                continue
            same = it["commit"] not in ("", "?") and self.head_sha.startswith(
                it["commit"][:7]
            )
            row = self.make_row(
                it,
                "🟢",
                "APPROVED",
                it["commit"][:8],
                [] if same else ["older head; still counts on develop"],
                "script",
                "data",
                f"reviews.md:{it['line']} (staged)",
            )
            self.add_evidence(
                {
                    "id": row["id"],
                    "pass": "settle",
                    "severity": "🟢",
                    "claim": f"{it['who']} APPROVED at {it['commit'][:8]}",
                    "head": self.head_sha,
                    "at": _now(),
                    "kind": "read",
                    "file": "reviews.md",
                    "line": it["line"],
                    "quote": it["quote"][: rt.QUOTE_CAP],
                    "tree": "staged",
                    "sha": "",
                }
            )
        stt = ps.threads_state(_read(self.pr / "threads.md"))
        if stt["total"]:
            self.threads = stt
            light = "🟡" if stt["open"] else "🟢"
            self.check_claim(
                "Threads",
                {
                    "pass": "script",
                    "severity": light,
                    "claim": f"threads: {stt['open']} open of {stt['total']}",
                    "head": self.head_sha,
                    "at": _now(),
                    "kind": "data",
                    "file": "threads.md",
                    "parser": "threads",
                    "value": ps.threads_value(stt),
                },
            )

    def prepare_carry(self) -> None:
        """What carrying needs: our last review's head and the develop it merged (else the merge base then), today's
        develop, the delta (the PR's files whose content changed since that head), and the tree the probes run on (the
        merge with develop when it is clean, else the head)."""
        wt = self.t.wt
        self.dev_now = (
            rt.full_sha(wt, self.mk["develop"]) if self.mk.get("develop") else ""
        )
        if not self.dev_now:
            self.dev_now = rt.full_sha(wt, "refs/remotes/origin/develop")
        if self.ours:
            self.then_head = rt.full_sha(wt, self.ours["head"])
            if self.then_head and self.dev_now:
                then_dev = rt.full_sha(wt, self.ours.get("develop") or "")
                if (
                    then_dev
                    and rt.git_out(
                        wt, "merge-base", "--is-ancestor", then_dev, self.dev_now
                    )[0]
                ):
                    then_dev = ""
                self.dev_then = then_dev or rt.merge_base(
                    wt, self.then_head, self.dev_now
                )
            if self.then_head and self.head_sha:
                paths = [f["path"] for f in self.files]
                a, b = rt.blobs(wt, self.then_head, paths), rt.blobs(
                    wt, self.head_sha, paths
                )
                if a is not None and b is not None:
                    self.delta = [p for p in paths if a[p] != b[p]]
        clean = (
            bool(self.merge.get("rid"))
            and not self.merge.get("files")
            and self.t.wtm.exists()
        )
        self.probe_tree = "merged" if clean else "head"

    def try_carry(self, it: dict) -> bool:
        """Settle one of our last review's rows or findings by script when nothing it cites changed (unchanged), or
        when it is a closed row that cites no file (closed). Records why not in it["carry_why"].
        """
        prev = it.get("prev") or {}
        ready = bool(
            self.then_head and self.head_sha and self.dev_then and self.dev_now
        )
        same, rec = (
            ps.unchanged(
                self.t.wt,
                self.then_head,
                self.head_sha,
                self.dev_then,
                self.dev_now,
                it["files"],
            )
            if it["files"] and ready
            else (None, {})
        )
        if same is not None:
            it["changed"] = ps.changed_paths(rec)
        if it.get("mentions"):
            it["carry_why"] = "named in a comment since our last review"
            return False
        closed = ps.is_closed(prev.get("light", ""), prev.get("status", ""))
        if not it["files"]:
            if closed and it["kind"] == "row":
                row = self.make_row(
                    it,
                    prev["light"],
                    prev["status"],
                    prev.get("sha", ""),
                    ["closed in our last review; cites no file (read)"],
                    "script",
                    "closed",
                    f"ours.md:{prev['line']} (staged)",
                )
                self.add_evidence(
                    {
                        "id": row["id"],
                        "pass": "settle",
                        "severity": prev["light"],
                        "claim": "closed in our last review; cites no file",
                        "head": self.head_sha,
                        "at": _now(),
                        "kind": "read",
                        "file": "ours.md",
                        "line": prev["line"],
                        "quote": prev["quote"][:200],
                        "tree": "staged",
                        "sha": "",
                    }
                )
                return True
            it["carry_why"] = "cites no file"
            return False
        if it["unresolved"] and not closed:
            it["carry_why"] = (
                f"cites {', '.join(it['unresolved'][:3])}, not found in the tree"
            )
            return False
        if not ready:
            it["carry_why"] = "our last review's head or develop is not in the clone"
            return False
        if same is None:
            it["carry_why"] = f"{str(rec.get('missing'))[:10]} is not in the clone"
            return False
        if not same:
            it["carry_why"] = "changed since our last review: " + ", ".join(
                self.t.short(p) for p in it["changed"][:3]
            )
            return False
        names = ", ".join(self.t.short(p) for p in it["files"][:3]) + (
            f" +{len(it['files']) - 3}" if len(it["files"]) > 3 else ""
        )
        row = self.make_row(
            it,
            prev["light"],
            prev["status"],
            prev.get("sha", ""),
            [f"unchanged since `{self.then_head[:10]}`: {names} (read)"]
            + prev.get("points", [])[:2],
            "script",
            "unchanged",
            f"unchanged since `{self.then_head[:10]}`",
        )
        self.add_evidence(
            {
                "id": row["id"],
                "pass": "settle",
                "severity": prev["light"],
                "claim": _cell(f"unchanged since {self.then_head[:10]}: {names}", 300),
                "head": self.head_sha,
                "at": _now(),
                "kind": "unchanged",
                **rec,
            }
        )
        return True

    def phase_settle(self) -> None:
        """Phase 0, the script: the checks, the prior items, the data rows, and our last review's rows and findings
        that carry (those with saved probes wait for them)."""
        self.pass_ = "settle"
        t0 = self.clock()
        self.scripted_checks()
        self.build_items()
        self.settle_data()
        self.prepare_carry()
        for it in self.items:
            if (
                it["kind"] in ("row", "finding")
                and it["id"] not in self.rows
                and not it.get("probes")
            ):
                self.try_carry(it)
        self.write_leads()
        self.set_pass(
            "settle",
            {
                "status": "ok",
                "wall_s": round(self.clock() - t0, 1),
                "items": len(self.items),
                "script": sum(
                    1 for r in self.rows.values() if r["settler"] == "script"
                ),
                "skipped": len(self.skipped),
                "probes": len(self.probe_names),
            },
        )

    def write_leads(self) -> None:
        """prior_leads.md: the rows the script settled, with their evidence, for the verifier to start from."""
        with self._lock:
            rows = [
                self.rows[it["id"]]
                for it in self.items
                if it["id"] in self.rows and self.rows[it["id"]]["settler"] == "script"
            ]
        lines = [
            f"# Prior items a script settled: #{self.n} at {self.head_sha[:10]}",
            "",
            "No model: approvals from the staged reviews, our last review's rows whose cited files did not change "
            "since it, rows it closed that cite no file, and the saved probes' verdicts. Leads only: each row's "
            "evidence is its last column.",
            "",
            "| Row | Prior item | Status | How | Evidence |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            sha = f" `{r['sha']}`" if r["sha"] else ""
            lines.append(
                f"| {r['id']} | {r['key']} | {r['light']} {r['status']}{sha} | {r['how']} | "
                f"{r['ref'] or 'none'} |"
            )
        if not rows:
            lines.append("| — | none | — | — | — |")
        pending = [
            it["id"]
            for it in self.items
            if it.get("probes") and it["id"] not in self.rows
        ]
        notes = []
        if self.ours:
            how = "kept unposted" if self.ours.get("kept") else "posted"
            notes.append(
                f"Our last review: at `{self.ours['head'][:10]}`, {how} {self.ours.get('at') or '?'}; develop "
                f"then `{self.dev_then[:10] or '?'}`, today `{self.dev_now[:10] or '?'}`."
            )
        else:
            notes.append(
                "No review of ours is staged: our earlier items are not among these rows."
            )
        if self.probe_names:
            notes.append(
                f"Saved probes: {len(self.probe_names)}, on the {self.probe_tree} tree"
                + (f"; still to run for {', '.join(pending)}." if pending else ".")
            )
        (self.ws / "prior_leads.md").write_text(
            "\n".join(lines + [""] + notes) + "\n", encoding="utf-8"
        )

    # --- Phase 1: the prior tasks, and the probes beside them ----------------------------------------------------------
    def run_probe(self, name: str) -> tuple[str, str, str, str]:
        """One saved probe on the probe tree: (its run id, verdict, summary, its line as run_probes.py prints it)."""
        secs = int(self.s["probe_timeout_s"])
        if self.limiter.deadline:
            secs = max(10, min(secs, int(self.limiter.deadline - self.clock()) - 30))
        args = {"name": name, "tree": self.probe_tree, "timeout_s": secs}
        text, r, argv = rt.run_tool(self.t, "probe", args)
        if r is None:
            return "", "ERROR", text, f"{name}: ERROR rc=-: {text}"
        rid = self.record_run(
            "probe",
            args,
            argv,
            r,
            {
                "probe_digest": rt.file_digest(self.t.probes / f"{name}.sh"),
                "tree_sha": self.header[
                    "merged" if self.probe_tree == "merged" else "head"
                ],
            },
        )
        if r.rc is None:
            return (
                rid,
                "ERROR",
                f"timed out after {secs}s",
                f"{name}: ERROR rc=timeout: timed out after {secs}s",
            )
        verdict, summary = rt.probe_verdict(r.out)
        m = rt.PROBE_LINE.search(r.out or "")
        return (
            rid,
            verdict,
            summary,
            m.group(0) if m else f"{name}: {verdict}: {summary}",
        )

    def run_probes(self, submit) -> None:
        """Every saved probe, one after another (probes.txt holds their lines for the checks job); then each item whose
        probes all gave FIXED or REPRODUCES is settled by them, and any other goes to carrying, else to a task.
        """
        t0 = self.clock()
        results: dict[str, tuple] = {}
        for name in self.probe_names:
            if self.stopped:
                break
            results[name] = self.run_probe(name)
            self.logw(f"  [probe] {results[name][3][:300]}")
        if results and len(results) == len(self.probe_names):
            tree_sha = self.header["merged" if self.probe_tree == "merged" else "head"]
            (self.ws / "probes.txt").write_text(
                f"# probes of pr{self.n} at {tree_sha}: {','.join(self.probe_names)}\n"
                + "".join(r[3] + "\n" for r in results.values()),
                encoding="utf-8",
            )
        for it in self.items:
            if not it.get("probes") or it["id"] in self.rows:
                continue
            got = [results.get(name) for name in it["probes"]]
            if all(g and g[1] in ("FIXED", "REPRODUCES") for g in got):
                self.settle_by_probes(it, got)
            elif not self.try_carry(it):
                it["probe_why"] = "; ".join(
                    f"{name}: {g[1] if g else 'not run'}"
                    for name, g in zip(it["probes"], got)
                )
                submit(it)
        self.set_pass(
            "probes",
            {
                "status": "ok",
                "wall_s": round(self.clock() - t0, 1),
                "run": len(results),
                "tree": self.probe_tree,
            },
        )
        self.write_leads()

    def settle_by_probes(self, it: dict, got: list[tuple]) -> None:
        prev = it.get("prev") or {}
        if any(g[1] == "REPRODUCES" for g in got):
            light = (
                prev.get("light") if prev.get("light") in ("🔴", "🟠", "🟡") else "🟠"
            )
            word = "DEFERRED" if prev.get("status") == "DEFERRED" else "OPEN"
        else:
            light, word = "🟢", "FIXED"
        points = [
            f"probe {name}: {g[1]}" + (f": {g[2]}" if g[2] else "") + " (ran)"
            for name, g in zip(it["probes"], got)
        ]
        row = self.make_row(
            it,
            light,
            word,
            "",
            points,
            "script",
            "probe",
            ", ".join(f"{g[0]} probe" for g in got),
        )
        for name, g in zip(it["probes"], got):
            self.add_evidence(
                self.ran_rec(row["id"], light, f"probe {name}: {g[1]}", g[0], "settle")
            )

    def task_card(self, it: dict) -> str:
        """A prior item's task: the item, our last review's status of it, what changed since, the code it cites, the
        commits since it, and what to answer."""
        n = max(1, int(self.s["prior_task_steps"]))
        kind = {
            "row": "a row of our last review",
            "finding": "a finding of our last review",
            "review": "a review",
            "thread": "an inline thread",
            "comment": "a conversation comment",
        }.get(it["kind"], it["kind"])
        where = f" at `{self.t.short(it['where'])}`" if it.get("where") else ""
        state = f" [{it['state']}]" if it.get("state") else ""
        who = "" if it["kind"] in ("row", "finding") else f" by @{it['who']}"
        head = [
            f"TASK: one prior item. {self.pr_line()}",
            f"Item {it['id']}: {kind}{where}{state}{who} ({it['date']}):",
            rt.clip(it["text"], 1500),
        ]
        head += [
            f"-- reply by @{r['who']} ({r['date']}): {_cell(r['text'], 300)}"
            for r in it["replies"][-3:]
        ]
        prev = it.get("prev")
        if prev:
            head.append(
                f"Our last review (at `{(self.ours or {}).get('head', '')[:10]}`) said: {prev['light']} "
                f"{prev['status']}"
                + "".join(f"; {_cell(p, 120)}" for p in prev.get("points", [])[:3])
            )
        if it.get("changed"):
            head.append(
                "Changed since then: "
                + ", ".join(self.t.short(p) for p in it["changed"][:6])
            )
        for o in (it.get("mentions") or [])[:2]:
            head.append(
                f"Since our last review, @{o['who']} ({o['date']}): {_cell(o['text'], 300)}"
            )
        if it.get("probe_why"):
            head.append(
                f"Its saved probes settled nothing: {_cell(it['probe_why'], 200)}"
            )
        since = (
            it["at"]
            if it["kind"] not in ("row", "finding")
            else (self.ours or {}).get("at", "")
        )
        commits = [
            f"{c[0]} {c[1][:10]} {_cell(c[2], 70)}"
            for c in self.commits
            if not since or c[1] >= since
        ][-10:]
        task = (
            f"\nDecide whether this item still stands at head {self.head_sha[:10]}. The code it cites is above; "
            "read more (diff_hunk, read_lines, grep_code, git_log, read_prior) only if that does not decide it. "
            "Then call settle once: FIXED (fixed at head: quote the fix, with the fixing commit from the list when "
            "one is), OPEN (still there: quote the line that shows it), PARTIAL (part fixed: quote what is left) or "
            "SKIP (it has no review point of its own: a reply, a status note, thanks). Evidence: a quote of whole "
            "lines at file:line (tree head, merged or base) or a run id [R<N>]. You have "
            f"{n} rounds; the last offers only settle."
        )
        tail = ["Commits since (sha, date, subject):"] + (commits or ["(none)"])
        budget = (
            self.card_chars() - sum(len(x) + 1 for x in head + tail) - len(task) - 40
        )
        ctx = self.item_context(it, max(600, budget))
        return "\n".join(head + ctx + tail) + "\n" + task

    def hunk_at(self, path: str, line: int) -> str:
        """The PR's hunk that holds `line` of `path` at head, as diff_hunk shows it; "" when none does."""
        f = next((f for f in self.files if f["path"] == path), None)
        for h in (f or {}).get("hunks") or []:
            span = sum(1 for x in h["lines"] if not x.startswith(("-", "\\")))
            if h["new"] <= line < h["new"] + max(1, span):
                return (
                    f"{self.t.short(path)} {h['id']} @@ -{h['old']} +{h['new']} @@ {h['ctx']}\n"
                    + self.hunk_text(h, 60)
                )
        return ""

    def item_context(self, it: dict, budget: int) -> list[str]:
        """The code a task needs: the change since our last review to the files it cites, then the lines it cites (the
        PR's hunk that holds them, else the lines around them at head)."""
        out = []
        for path in (it.get("changed") or [])[:2]:
            d = (
                rt.diff_text(self.t.wt, self.then_head, self.head_sha, path, 1800)
                if self.then_head
                else ""
            )
            if d:
                out.append(
                    f"The change to {self.t.short(path)} since our last review:\n{d}"
                )
        for path, line in (it.get("cite_lines") or [])[:2]:
            h = self.hunk_at(path, line)
            out.append(
                h or rt.read_lines(self.t, path, max(1, line - 8), line + 8, "head")
            )
        if not out and it.get("files"):
            out.append("Cited: " + ", ".join(self.t.short(p) for p in it["files"][:6]))
        return fit(out, budget, "code excerpts", "read_lines or diff_hunk")

    def task_run(self, it: dict) -> None:
        """One prior item's task: at most prior_task_steps requests, the last offering only settle."""
        self.pass_, self.task, self.pass_done, self.step = "prior", it, False, 0
        why = ""
        left = self.prior_deadline - self.clock()
        if self.stopped or left <= 0:
            self.unsettled(
                it, self.stopped or "the prior phase's time ran out before its task"
            )
            return
        self.limiter.phase("prior", left, int(self.s.get("prior_token_budget") or 0))
        n = max(1, int(self.s["prior_task_steps"]))
        effort = self.s["prior_effort"]
        msgs = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.task_card(it)},
        ]
        t0 = self.clock()
        try:
            for step in range(n):
                self.step = step
                names = self.WRAP["prior"] if step == n - 1 else self.MENUS["prior"]
                msg = self.ask(
                    msgs,
                    self.schemas(names),
                    effort,
                    int(self.s["prior_max_tokens"]),
                )
                calls = msg.get("tool_calls") or []
                self.log_step(f"prior {it['id']}", msg, effort, t0)
                msgs.append(
                    {
                        "role": "assistant",
                        "content": (msg.get("content") or "")[:1500],
                        **({"tool_calls": calls} if calls else {}),
                    }
                )
                if not calls:
                    msgs.append(
                        {
                            "role": "user",
                            "content": "Answer with settle: FIXED, OPEN, PARTIAL or SKIP, with "
                            "its evidence.",
                        }
                    )
                    continue
                for call in calls:
                    self.call(call, names, msgs)
                if self.pass_done:
                    break
        except (lc.OverBudget, lc.GatewayError, lc.RateLimited) as exc:
            why = self.stopped_by(exc)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one task's failure leaves its item unsettled, and the draft goes on
            why = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
            self.logw(traceback.format_exc())
        finally:
            self.limiter.phase()
            self.task = None
        with self._lock:
            done = it["id"] in self.rows or it["id"] in self.skipped
        if not done:
            self.unsettled(it, why or f"no verdict in its {n} rounds")

    def phase_prior(self) -> None:
        """Phase 1: the tasks of the items Phase 0 left open, prior_parallel at once, within prior_wall_min; the saved
        probes run in this thread meanwhile, and hand the items they do not settle to tasks.
        """
        t0 = self.clock()
        self.prior_deadline = t0 + float(self.s["prior_wall_min"]) * 60
        todo = [
            it
            for it in self.items
            if it["kind"] != "approval"
            and it["id"] not in self.rows
            and it["id"] not in self.skipped
            and not it.get("probes")
        ]
        futures = []
        workers = max(1, int(self.s["prior_parallel"]))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="prior"
        ) as pool:

            def submit(it):
                futures.append(pool.submit(self.task_run, it))

            for it in todo:
                submit(it)
            if self.probe_names:
                self.pass_ = "settle"
                self.run_probes(submit)
            concurrent.futures.wait(futures)
        with self._lock:
            settled = sum(1 for r in self.rows.values() if r["how"] == "task")
            skipped = sum(
                1 for why in self.skipped.values() if why.endswith("(the task)")
            )
            left = [r["id"] for r in self.rows.values() if r["status"] == "NOT SETTLED"]
        self.set_pass(
            "prior",
            {
                "status": "ok" if not left else f"{len(left)} item(s) not settled",
                "wall_s": round(self.clock() - t0, 1),
                "tasks": len(futures),
                "settled": settled,
                "skipped": skipped,
                "not_settled": left,
                "effort": self.s.get("prior_effort"),
            },
        )

    # --- Phase 2: the hunt --------------------------------------------------------------------------------------------
    def loop(
        self, name: str, user: str, effort: str, steps: int, max_tokens: int
    ) -> None:
        self.pass_, self.pass_done, self.task = name, False, None
        menu, wrap = self.MENUS[name], self.WRAP[name]
        tools = self.schemas(menu)
        msgs = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]
        t0 = self.clock()
        status, nudged = "ok", False
        try:
            for step in range(steps):
                self.step = step
                self.limiter.progress = 0.3 + 0.6 * step / max(1, steps)
                self.mask(msgs)
                msg = self.ask(msgs, tools, effort, max_tokens)
                calls = msg.get("tool_calls") or []
                self.log_step(name, msg, effort, t0)
                msgs.append(
                    {
                        "role": "assistant",
                        "content": (msg.get("content") or "")[:1500],
                        **({"tool_calls": calls} if calls else {}),
                    }
                )
                if not calls:
                    if nudged:
                        break
                    nudged = True
                    msgs.append(
                        {
                            "role": "user",
                            "content": "Use the tools: add_finding (with its quote) for each "
                            "finding, then finish.",
                        }
                    )
                    continue
                for call in calls:
                    self.call(call, menu, msgs)
                if self.pass_done:
                    break
            if not self.pass_done:
                self.wrap_up(name, msgs, wrap)
        except (lc.OverBudget, lc.GatewayError, lc.RateLimited) as exc:
            status = self.stopped_by(exc)
            self.logw(f"  [{name}] {status}")
        self.set_pass(
            name,
            {
                "status": status,
                "steps": self.step + 1,
                "wall_s": round(self.clock() - t0, 1),
                "effort": effort,
            },
        )

    def wrap_up(self, name: str, msgs: list, wrap) -> None:
        """The pass ran out of rounds: one or two turns at the mechanical effort, with only the recording tools."""
        self.limiter.progress = 0.95
        effort = self.s["mechanical_effort"]
        msgs.append(
            {
                "role": "user",
                "content": "Your rounds are spent. Record what you can back with what you already "
                "read (" + ", ".join(wrap) + "), then call finish. Reading tools are "
                "refused now.",
            }
        )
        tools = self.schemas(wrap)
        for _ in range(2):
            self.step += 1
            self.mask(msgs)
            msg = self.ask(msgs, tools, effort, 4000)
            calls = msg.get("tool_calls") or []
            self.log_step(name + " wrap-up", msg, effort, self.clock())
            msgs.append(
                {
                    "role": "assistant",
                    "content": (msg.get("content") or "")[:800],
                    **({"tool_calls": calls} if calls else {}),
                }
            )
            for call in calls:
                self.call(call, wrap, msgs)
            if self.pass_done or not calls:
                break

    def phase_hunt(self) -> None:
        """Phase 2: the hunt, within its own hunt_wall_min and hunt_token_budget, whatever the prior phase spends."""
        self.limiter.phase(
            "hunt",
            float(self.s["hunt_wall_min"]) * 60,
            int(self.s.get("hunt_token_budget") or 0),
        )
        try:
            self.loop(
                "hunt",
                self.hunt_card(),
                self.s["effort"],
                int(self.s["hunt_steps"]),
                int(self.s["hunt_max_tokens"]),
            )
        finally:
            self.limiter.phase()

    def guarded(self, name: str, fn) -> None:
        """A phase in its thread: an unexpected error is logged and recorded as the phase's status."""
        try:
            fn()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a phase's bug must not lose what the other phases recorded
            self.logw(traceback.format_exc())
            self.set_pass(
                name, {"status": f"failed: {type(exc).__name__}: {str(exc)[:200]}"}
            )

    # --- the task cards -----------------------------------------------------------------------------------------------
    def card_chars(self) -> int:
        return int(float(self.s["card_tokens"]) * lc.CHARS_PER_TOKEN)

    def ci_light(self) -> str:
        return (
            "🔴"
            if self.ci["failing"]
            else "🟡" if self.ci["running"] else "🟢" if self.ci["total"] else "🟡"
        )

    def ci_word(self) -> str:
        return (
            "Red"
            if self.ci["failing"]
            else (
                "Running"
                if self.ci["running"]
                else "Green" if self.ci["total"] else "No CI"
            )
        )

    def short(self, path: str) -> str:
        return self.t.short(path)

    def pr_line(self) -> str:
        return (
            f"PR #{self.n} \"{strip_type(self.meta['title'])}\" by {self.meta['author']}, {self.round}. Head "
            f"{self.header['head'][:10]}, merge base {self.header['base'][:10]}"
            + (
                f", develop {self.mk['develop']}; behind {self.mk['behind']} / ahead {self.mk['ahead']}"
                if self.mk["develop"]
                else ""
            )
            + "."
        )

    def hunt_card(self) -> str:
        plus = sum(f["plus"] for f in self.files)
        minus = sum(f["minus"] for f in self.files)
        merge = (
            f"merge with develop: clean ({self.merge['rid']})"
            if self.merge.get("rid") and not self.merge["files"]
            else (
                f"merge with develop: {len(self.merge['files'])} files conflict ({self.merge['rid']})"
                if self.merge.get("rid")
                else "merge with develop: not checked"
            )
        )
        rules = (
            f"rule_scan ({self.rules['rid']}): "
            + (", ".join(f"{a} {b}" for a, b in self.rules["hits"].items()) or "clean")
            if self.rules.get("rid")
            else ""
        )
        topics = [
            it.get("key")
            or f"{self.shown_login(it['who'])} ({it['date']}): {ps.gist(it['text'], 90)}"
            for it in self.items
            if it["kind"] != "approval" and it["id"] not in self.skipped
        ]
        delta = [f["id"] for f in self.files if f["path"] in self.delta]
        task = (
            "\nFind every net-new blocker in this PR as it would land (no cap), and significant follow-ups only"
            + (
                f", the Δ files first: they changed since our last review at `{self.then_head[:10]}`"
                if delta
                else ""
            )
            + ". Read the code files' hunks first (diff_hunk by id); for each suspicious change read around it "
            "(read_lines), check sibling sites and the other implementations (grep_code), and try to disprove it "
            "(tree base, the prior items) before you add it. For each kept finding: add_finding with the exact "
            "line as its quote (its evidence; record_evidence adds more). Every 🔴/🟠 needs evidence before you "
            "finish. You may call several tools in one turn. You have about "
            f"{int(self.s['hunt_steps'])} rounds; tool results older than the last "
            f"{int(self.s['keep_results'])} are hidden behind a note of what they were."
        )
        head = [
            f"TASK: net-new findings. {self.pr_line()} {merge}."
            + (f" {rules}." if rules else ""),
            "Description: " + _cell(self.meta["description"], 700),
        ]
        if delta:
            head.append(
                f"Changed since our last review (Δ): {', '.join(delta[:40])}"
                + (f" and {len(delta) - 40} more" if len(delta) > 40 else "")
            )
        prior = (
            ["Already raised (prior items, never a net-new finding):"]
            + [f"- {_cell(t, 110)}" for t in topics[:15]]
            if topics
            else []
        )
        budget = (
            self.card_chars() - sum(len(x) + 1 for x in head + prior) - len(task) - 60
        )
        pmap = [
            f"PR map ({len(self.files)} files, +{plus}/-{minus}); diff_hunk('H1.2') shows a hunk, diff_hunk('F1') "
            "a file's hunks:"
        ] + fit(
            map_lines(self.files, self.short, self.delta),
            max(1200, budget),
            "files",
            "diff_hunk by path",
        )
        return "\n".join(head + prior + pmap) + "\n" + task

    # --- the run ------------------------------------------------------------------------------------------------------
    def run_draft(self) -> bool:
        """The whole draft: Phase 0 (script), then the prior tasks and the hunt at once, each within its own caps, then
        the assembly. False when no model answer came (nothing to draft from)."""
        self.start()
        try:
            self.guarded("settle", self.phase_settle)
            hunt = threading.Thread(
                target=self.guarded,
                args=("hunt", self.phase_hunt),
                name="hunt",
                daemon=True,
            )
            hunt.start()
            self.guarded("prior", self.phase_prior)
            end = self.limiter.deadline
            hunt.join(timeout=max(1.0, end - self.clock() + 60) if end else None)
            if hunt.is_alive():
                self.set_pass(
                    "hunt", {"status": "stopped: still running at the run's end"}
                )
            self.write_leads()
            ok = self.answered > 0
            if ok:
                self.mark_missing()
                self.write_draft()
            self.add_phase_counts()
            self.header.update(draft=ok, stopped=self.stopped, finished=_now())
            self.save_header()
            return ok
        finally:
            self.limiter.close()
            self.write_tokens()
            with self._lock:
                if self.log:
                    self.log.close()
                    self.log = None

    def add_phase_counts(self) -> None:
        """Each model phase's requests and tokens, from the limiter, into draft_run.json."""
        passes = self.limiter.summary().get("passes") or {}
        with self._lock:
            for name in ("prior", "hunt"):
                if name in passes and name in self.header["passes"]:
                    p = passes[name]
                    self.header["passes"][name].update(
                        requests=p["requests"],
                        tokens=p["tokens"],
                        reasoning_tokens=p["reasoning"],
                    )

    def mark_missing(self) -> None:
        """A finding or prior row left without evidence gets a "none" record, for the evidence check."""
        have = self.evidenced()
        with self._lock:
            claims = [
                (f["id"], f["severity"], f["problem"])
                for f in self.findings.values()
                if not f["dropped"]
            ]
            claims += [(r["id"], r["light"], r["key"]) for r in self.rows.values()]
        for cid, sev, claim in claims:
            if cid not in have:
                self.add_evidence(
                    {
                        "id": cid,
                        "pass": "draft",
                        "severity": sev,
                        "claim": claim,
                        "kind": "none",
                        "missing": True,
                        "head": self.head_sha,
                        "at": _now(),
                    }
                )

    # --- the post -----------------------------------------------------------------------------------------------------
    def settled_by(self, r: dict) -> str:
        if r["settler"] == "script":
            return f"script: {r['how']}"
        if r["settler"] == "model":
            return str(self.s.get("binding") or "model")
        return "—"

    def ids_of(self, row: str) -> str:
        return ", ".join(cid for cid, name in self.checks.items() if name == row) or "—"

    def prior_rows(self) -> list[str]:
        rows = []
        for it in self.items:
            r = self.rows.get(it["id"])
            if not r:
                continue
            st = f"{r['light']} {r['status']}" + (f" `{r['sha']}`" if r["sha"] else "")
            rows.append(
                f"| {r['key']} | {st}"
                + "".join(f"<br>• {x}" for x in r["points"])
                + f" | {self.settled_by(r)} | {r['id']}: {r['ref'] or 'none'} |"
            )
        if not rows:
            rows.append(
                "| Reviewers | 🟢 None yet<br>• no reviews or threads, only the policy bot | script: data | — |"
            )
        return rows

    def threads_row(self) -> list[str]:
        t = self.threads
        if not t:
            return []
        pts = [
            f"open: `{self.short(o['path'])}:{o['line']}` @{self.shown_login(o['who'])}"
            for o in t["open_list"][:3]
        ]
        if t["open"] > 3:
            pts.append(f"{t['open'] - 3} more open")
        pts.append(f"{t['resolved']} resolved, {t['outdated']} outdated (read)")
        word = (
            f"{t['open']} open of {t['total']}"
            if t["open"]
            else f"none open of {t['total']}"
        )
        return [
            f"| Threads | {'🟡' if t['open'] else '🟢'} {word}"
            + "".join(f"<br>• {p}" for p in pts)
            + f" | script: data | {self.ids_of('Threads')} |"
        ]

    def ci_row(self) -> str:
        pts = [
            f"fails: {_cell(n, 70)} — cause not checked"
            for n, _ in self.ci["failing"][:3]
        ]
        if len(self.ci["failing"]) > 3:
            pts.append(f"{len(self.ci['failing']) - 3} more failing")
        if self.ci["running"]:
            pts.append(f"running: {len(self.ci['running'])} jobs")
        if self.ci["total"]:
            pts.append(
                f"{self.ci['success']} ✓"
                + (
                    f"; {self.ci['cancelled']} cancelled"
                    if self.ci["cancelled"]
                    else ""
                )
                + " (read)"
            )
        if self.meta.get("mergeable"):
            pts.append(f"merge: {self.meta['mergeable'].lower()} (read)")
        return (
            f"| Bots / CI | {self.ci_light()} {self.ci_word()}"
            + "".join(f"<br>• {p}" for p in pts)
            + f" | script: data | {self.ids_of('Bots / CI')} |"
        )

    def checks_row(self) -> str:
        pts, light = [], "🟢"
        dev = self.mk["develop"] or (self.header.get("merged") or "")[:10]
        if self.merge.get("rid"):
            if self.merge["files"]:
                light = "🟠"
                pts.append(
                    f"merge vs `{dev}`: {len(self.merge['files'])} files conflict: "
                    + _cell(
                        ", ".join(self.short(f) for f in self.merge["files"][:4]), 100
                    )
                    + " (ran)"
                )
            else:
                pts.append(f"merge vs `{dev}`: clean (ran)")
        if self.rules.get("rid"):
            pts.append(
                "rules 1–3 clean (ran)"
                if not self.rules["hits"]
                else "rule_scan hints: "
                + ", ".join(f"{a} {b}" for a, b in self.rules["hits"].items())
                + " (ran)"
            )
        pts.append(
            str(self.profile.get("checks_note") or PROFILE_DEFAULTS["checks_note"])
        )
        return (
            f"| Our checks | {light} {'Conflicts' if light == '🟠' else 'Clean'}"
            + "".join(f"<br>• {p}" for p in pts)
            + f" | script: data | {self.ids_of('Our checks')} |"
        )

    def finding_rows(self) -> list[str]:
        live = sorted(
            (f for f in self.findings.values() if not f["dropped"]),
            key=lambda f: SEV_ORDER[f["severity"]],
        )
        out = []
        for f in live:
            impact = f"<br>• impact: {f['impact']}" if f["impact"] else ""
            fix = f"<br>• fix: {f['fix']}" if f["fix"] else ""
            out.append(
                f"| {f['id']} | {f['severity']} | `{self.short(f['file'])}:{f['line']}` | **{f['problem']}**"
                f"{impact}{fix} |"
            )
        return out

    def header_light(self) -> str:
        lights = [f["severity"] for f in self.findings.values() if not f["dropped"]]
        lights += [
            r["light"]
            for r in self.rows.values()
            if r["settler"] and ps.is_open(r["light"], r["status"])
        ]
        if self.merge.get("files"):
            lights.append("🟠")
        worst = min(lights, key=lambda x: SEV_ORDER.get(x, 5), default="🟢")
        return worst if worst in ("🔴", "🟠", "🟡") else "🟢"

    def write_draft(self) -> None:
        title = strip_type(self.meta["title"])
        tag = tracker_tag(
            self.profile.get("tracker_keys"),
            self.meta["title"],
            self.meta["description"],
            _read(self.pr / "commits.md"),
        )
        with self._lock:
            live = [f for f in self.findings.values() if not f["dropped"]]
        reason = self.finish.get("reason") or (
            f"{len(live)} open finding{'s' if len(live) != 1 else ''}"
            if live
            else "no net-new findings"
        )
        light = (
            self.finish.get("light") if self.finish.get("light") in UTILITY else "🟡"
        )
        utility = self.finish.get("utility") or "not judged by the drafter"
        model = self.s.get("name") or self.s.get("model") or "the drafter's model"
        rows = self.finding_rows()
        who = self.meta["author"] + (f", {tag}" if tag else "")
        lines = [
            self.line1,
            "",
            f"## PR #{self.n} — {title} ({who}) — {self.header_light()} {self.round}: {reason}",
            "",
            "| Prior item | Status | Settled by | Evidence |",
            "|---|---|---|---|",
        ]
        lines += (
            self.prior_rows()
            + self.threads_row()
            + [self.ci_row(), self.checks_row(), ""]
        )
        lines += (
            ["| # | Sev | file:line | Finding |", "|---|---|---|---|"] + rows + [""]
            if rows
            else ["**Net-new:** none.", ""]
        )
        lines += [
            f"**Utility:** {light} {UTILITY[light]} — {utility}",
            "",
            "---",
            "**Legend:** 🔴 blocking · 🟠 should-fix · 🟡 minor nit · 🟢 info/good · ⚪ retracted or not settled",
            "",
            f"_{self.round} review: first pass by {model}, verified by the orchestrator "
            f"({self.cfg.get('verifier_name') or 'the verifier'}). Checks: merge with develop, rules 1–3, "
            "prior items. Flags whether a closer look is needed; not an approval or rejection._",
        ]
        text = "\n".join(lines) + "\n"
        (self.ws / "review_agent.md").write_text(text, encoding="utf-8")
        priors = []
        for it in self.items:
            r = self.rows.get(it["id"])
            if r:
                priors.append(
                    {
                        **{
                            k: r[k]
                            for k in (
                                "id",
                                "key",
                                "light",
                                "status",
                                "settler",
                                "how",
                                "ref",
                                "prev_light",
                                "prev_status",
                                "flip",
                            )
                        },
                        "items": [it["id"]],
                        "kind": it["kind"],
                        "why": (
                            r["points"][0]
                            if r["status"] == "NOT SETTLED" and r["points"]
                            else ""
                        ),
                    }
                )
        (self.ws / "draft_findings.json").write_text(
            json.dumps(
                {
                    "pr": int(self.n),
                    "head": self.header["head"],
                    "round": self.round,
                    "binding": self.s.get("binding"),
                    "model": self.s.get("model"),
                    "role": self.role,
                    "method": METHOD,
                    "findings": [
                        {
                            k: f.get(k)
                            for k in (
                                "id",
                                "severity",
                                "file",
                                "line",
                                "problem",
                                "dropped",
                                "why",
                            )
                        }
                        for f in self.findings.values()
                    ],
                    "priors": priors,
                    "checks": self.checks,
                    "skipped": self.skipped,
                    "ours": (
                        {k: self.ours.get(k) for k in ("head", "at", "kept")}
                        if self.ours
                        else None
                    ),
                    "delta": [self.short(p) for p in self.delta],
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (self.ws / "stats_agent.json").write_text(
            json.dumps(
                {
                    "stage": "tool-loop",
                    "method": METHOD,
                    "steps": sum(
                        int(p.get("steps", 0) or 0)
                        for p in self.header["passes"].values()
                    ),
                    "tool_runs": len(self.runs),
                    "wall_s": round(self.limiter.elapsed(), 1),
                    "final_chars": len(text),
                }
            ),
            encoding="utf-8",
        )

    def write_tokens(self) -> None:
        """tokens_total.json: this attempt's counts added to the earlier attempts' in the folder."""
        path = self.ws / "tokens_total.json"
        try:
            old = json.loads(_read(path) or "{}")
        except ValueError:
            old = {}
        attempts = [a for a in (old.get("attempts") or []) if isinstance(a, dict)]
        attempts.append(
            dict(
                self.limiter.summary(),
                attempt=self.header.get("attempt"),
                at=_now(),
                stopped=self.stopped,
                draft=bool(self.header.get("draft")),
            )
        )
        keys = (
            "requests",
            "tokens",
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "waits_s",
            "rate_limited",
            "estimated",
            "wall_s",
        )
        total = {k: round(sum(float(a.get(k) or 0) for a in attempts), 1) for k in keys}
        for k in keys:
            if k not in ("waits_s", "wall_s"):
                total[k] = int(total[k])
        path.write_text(
            json.dumps(
                {
                    "pr": int(self.n),
                    "head": self.header.get("head"),
                    "model": self.s.get("model"),
                    "binding": self.s.get("binding"),
                    "method": METHOD,
                    "attempts": attempts,
                    "total": total,
                },
                indent=1,
            ),
            encoding="utf-8",
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="The tool-loop drafter: one PR's first-pass review in the post format."
    )
    ap.add_argument("ws")
    ap.add_argument("n")
    ap.add_argument("--trees", default=None)
    ap.add_argument("--round", default="", choices=("", "First-round", "Follow-up"))
    ap.add_argument("--line1", default="")
    ap.add_argument("--role", default="drafter", choices=("drafter", "shadow_drafter"))
    ap.add_argument("--prio", action="store_true")
    ap.add_argument(
        "--config",
        default=None,
        help="a JSON file whose keys replace the configuration's",
    )
    a = ap.parse_args(argv)
    from rkmon_lib import applied, bindings, config, premises
    from rkmon_lib.store import load_config

    try:
        cfg = load_config()
        if a.config:
            cfg = config.with_override(cfg, a.config)
    except config.ConfigError as exc:
        return applied.refuse(exc)
    # verifier_name by its rule: a blank one is the audit model in words
    cfg["verifier_name"] = premises.verifier_name(cfg)
    cfg["review_first_line"] = bindings.first_line(cfg, a.role)
    s = bindings.client_settings(cfg, a.role)
    if s is None:
        print(
            f"the {a.role} role has no binding ("
            + ("reviewer" if a.role == "drafter" else f"roles.{a.role}")
            + " naming a model_bindings entry)"
        )
        return 1
    prio = a.prio or os.environ.get("RKMON_DRAFT_PRIO") == "1"
    sess = Session(
        a.ws,
        a.n,
        cfg,
        s,
        trees=a.trees,
        round_word=a.round,
        line1=a.line1,
        prio=prio,
        role=a.role,
    )
    t0 = time.time()
    ok = sess.run_draft()
    last = (
        json.loads(_read(pathlib.Path(a.ws) / "tokens_total.json") or "{}").get(
            "attempts"
        )
        or [{}]
    )[-1]
    print(
        f"PR #{a.n}: draft {'written' if ok else 'NOT written'} in {time.time() - t0:.0f}s: "
        f"{last.get('requests', 0)} requests, {last.get('tokens', 0)} tokens ({last.get('reasoning_tokens', 0)} "
        f"reasoning, {last.get('cached_tokens', 0)} cached), waited {last.get('waits_s', 0)}s"
        + (f"; {sess.stopped}" if sess.stopped else "")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
