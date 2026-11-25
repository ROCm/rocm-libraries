# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The code quality pass on a binding (config roles.quality): the role's model judges one PR's code by the code quality
reviewer's instructions (the team member as the agentic install rendered it, rkmon_lib/agentic.py:
<agentic>/team_members/rocke_*quality*.md; GENERIC_CARD without one) with the drafter's tool-loop machinery
(drafter.Session: the PR's trees and staged data, the read tools, the shared limiter and its caps, the evidence and
token records), and the script writes quality.md in the pass's Output Format, which the verifier folds into the post
as it folds the orchestrator's code quality reviewer's.

One phase, "quality", within drafter.quality_wall_min and drafter.quality_token_budget (the run's caps), at the
binding's effort, in at most drafter.quality_steps rounds. Findings are tool calls, checked at once: add_finding (Q1,
Q2, ... at file:line in the head tree, its quote recorded as evidence), record_evidence (the existing home, the
standard's line, the sibling path, or a run), drop_finding, and finish (the Code quality row's points, and the notes).
A 🔴/🟠 needs evidence beside the quote of its own line before finish: the home it duplicates, the standard, the
sibling, or a run.

Its folder is <workers>/pr<N>/quality (tools/quality_run.py makes it): pr/ is the draft's staged data, mkwt.log the
staging's, overlap_section.md the checks' overlap leads (`rk overlaps N --section`). Outputs: quality.md,
quality_run.json (the run's header and its phase: status, rounds, time, requests, tokens), evidence.jsonl, runs.jsonl,
tokens_total.json and agent_log.txt.

Usage: python3 -B quality_lane.py <folder> <PR number> [--config FILE]
--config: a JSON file whose keys replace the configuration's. Exit 0 with quality.md written; 1 without; REFUSED,
exit 4, when the configuration or the --config file cannot be read.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
MONITOR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MONITOR))

import drafter  # noqa: E402
import llm_client as lc  # noqa: E402
import review_tools as rt  # noqa: E402

METHOD = "quality-lane"
ROLE = "quality"
MAX_FINDINGS = 5
WORDS = {"🟢": "Clean", "🟡": "Minor", "🟠": "Should-fix", "🔴": "Blocking"}
MEMBER_GLOB = "rocke_*quality*.md"
OVERLAP_CAP = 4000  # characters of the checks' overlap section the card carries
GENERIC_CARD = """THE CODE QUALITY PASS'S RULES
You judge whether the PR's code is well built: whether a maintainer would be glad to own it. Find where the PR's code
breaks one of these, prove each with a quoted file:line, and name a more suitable fix you checked is feasible:
1. Single responsibility: a function, class, module, test or build block does one job.
2. DRY, one home per fact: before you accept a new helper, constant, table, message or fixture, search for an existing
   home; name it (path:line) and say which copy stays.
3. Primitives first, with clear layers: a layer uses the one below through its public surface and never reaches around
   it.
4. Interfaces judged from the caller's side: write the call a user would write; names, argument order, defaults and
   errors must make it straightforward.
5. Descriptive names, in the domain's terms.
6. Every claim backed by a test that fails without the change, in a CI lane.
7. Root-cause fixes, in every engine or path that has the same code.
8. Overlapping work in other open PRs: the checks' section lists leads; one PR owns each shared piece.
Out of scope: correctness of results, CI, merge conflicts and goldens (the verifier's); formatting a formatter owns;
problems the PR neither adds nor makes worse.
Severity: 🔴 rare (the design itself breaks results or CI on merge); 🟠 a rule above clearly broken by this PR, with a
concrete fix; 🟡 names, duplication inside the PR itself, an optional cleaner alternative. One finding per root cause, at
most five, most important first. A judgement call with no clearly better alternative is not a finding."""
TOOL_NOTE = (
    "YOUR WORK HERE. The instructions above were written for an agent with a shell; here your tools are your "
    "only actions, and the script writes quality.md from your tool calls. Take their rules, checks, evidence "
    "standards and severities as yours; leave out what they say about running commands, files, lessons and "
    "the final message. Tools: diff_hunk (the PR's hunks by id), read_lines (a file at head, base or merged: "
    "the standards documents too), grep_code (search the head tree for an existing home), git_log, "
    "add_finding (Q1, Q2, ...: its quote at file:line in the head tree is recorded as its evidence), "
    "record_evidence (the home it duplicates, the standard, the sibling path: a quote at file:line in a tree, "
    "or a run id), drop_finding and finish."
)


class QualitySession(drafter.Session):
    """One code quality pass of one PR: the drafter's session with the pass's card, tools, records and output."""

    S = dict(
        drafter.Session.S,
        **{
            "add_finding": (
                "Add a quality finding at file:line in the head tree: severity (🔴 🟠 🟡), problem (one short "
                "clause), evidence (the existing home `path:line`, the standard and its section, or the "
                "sibling path, in a few words), fix (the concrete alternative, checked feasible) and quote "
                "(the line at file:line, exactly: it is recorded as the finding's evidence).",
                {
                    "severity": "string",
                    "file": "string",
                    "line": "integer",
                    "problem": "string",
                    "evidence": "string",
                    "fix": "string",
                    "quote": "string",
                },
                ["severity", "file", "line", "problem", "fix"],
            ),
            "record_evidence": (
                "Evidence for one of your findings (Q1) beside its own line: either run (an [R<N>] id), or "
                "file, line and quote (whole lines, exactly) with tree head, base or merged.",
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
            "finish": (
                "End the pass: points (1 to 3 points of 4 to 10 words for the Code quality row, separated by ';', "
                "each marked (read) or (ran)), and the notes: searched (what you searched for helpers and "
                "duplicates, and where), standards (the documents and sections you read), rechecked (a follow-up: "
                "each open quality finding of our last review, FIXED or OPEN, with its evidence), overlaps (each lead "
                "of the overlap section: same work, coincidence, or a develop pattern) and lead (one pre-existing "
                "issue at most, or none). Every 🔴/🟠 needs its evidence first.",
                {
                    "points": "string",
                    "searched": "string",
                    "standards": "string",
                    "rechecked": "string",
                    "overlaps": "string",
                    "lead": "string",
                },
                ["points"],
            ),
        },
    )
    MENUS = dict(
        drafter.Session.MENUS,
        quality=(
            "diff_hunk",
            "read_lines",
            "grep_code",
            "git_log",
            "add_finding",
            "record_evidence",
            "drop_finding",
            "finish",
        ),
    )
    WRAP = dict(
        drafter.Session.WRAP,
        quality=("add_finding", "record_evidence", "drop_finding", "finish"),
    )

    def __init__(self, ws, n, cfg: dict, s: dict, member: str = "", **kw):
        super().__init__(ws, n, cfg, s, role=ROLE, **kw)
        self.member = member
        self.notes: dict = {}
        self.system = (
            f"You run the code quality pass of the review of one pull request to {self.profile['project']}, "
            "by the instructions below. You work only through your tools: they read the PR's trees (head; "
            "merged with develop; the merge base), its diff and its staged review data. A stronger reviewer "
            "verifies your work, so precision beats volume: one real finding with evidence is worth more "
            "than five guesses.\n\n"
            + (member.strip() or GENERIC_CARD)
            + "\n\n"
            + TOOL_NOTE
        )

    # --- records ------------------------------------------------------------------------------------------------------
    def start(self) -> None:
        for name in ("quality.md", "evidence.jsonl", "runs.jsonl", "quality_run.json"):
            (self.ws / name).unlink(missing_ok=True)
        self.head_sha = self.t.sha("head")
        self.header = {
            "pr": int(self.n),
            "round": self.round,
            "model": self.s.get("model"),
            "binding": self.s.get("binding"),
            "role": ROLE,
            "method": METHOD,
            "head": self.head_sha,
            "merged": self.t.sha("merged"),
            "base": self.t.sha("base"),
            "ws": str(self.ws),
            "started": drafter._now(),
            "passes": {},
            "stopped": "",
            "written": False,
            "member": bool(self.member.strip()),
        }
        self.save_header()
        self.log = open(self.ws / "agent_log.txt", "a", encoding="utf-8")
        self.logw(
            f"=== code quality pass of #{self.n} ({self.round}), {drafter._now()}, model "
            f"{self.s.get('name') or self.s.get('model')}, head {self.head_sha[:10]} ==="
        )

    def save_header(self) -> None:
        with self._lock:
            (self.ws / "quality_run.json").write_text(
                json.dumps(self.header, indent=1), encoding="utf-8"
            )

    def write_tokens(self) -> None:
        super().write_tokens()
        path = self.ws / "tokens_total.json"
        try:
            rec = json.loads(drafter._read(path) or "{}")
        except ValueError:
            return
        rec["method"] = METHOD
        path.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    # --- the tools ----------------------------------------------------------------------------------------------------
    def own_evidence(self, qid: str) -> bool:
        """Whether finding qid has evidence beside the quote of its own line."""
        with self._lock:
            return any(
                e["id"] == qid
                and e.get("kind") not in ("none", None)
                and not e.get("own")
                for e in self.evidence
            )

    def t_add_finding(
        self,
        severity: str = "",
        file: str = "",
        line=0,
        problem: str = "",
        evidence: str = "",
        fix: str = "",
        quote: str = "",
        **_,
    ) -> str:
        sev = drafter.light_of(severity)
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
        if not str(problem or "").strip() or not str(fix or "").strip():
            return "REJECTED: problem (one short clause) and fix (the concrete alternative) are both needed"
        rec, why = (
            self.evidence_rec(
                "", sev, problem, file=rel, line=n, quote=quote, tree="head"
            )
            if str(quote or "").strip()
            else (None, "REJECTED: quote the line at file:line exactly")
        )
        if not rec:
            return why
        with self._lock:
            qid = f"Q{len(self.findings) + 1}"
            self.findings[qid] = {
                "id": qid,
                "severity": sev,
                "file": rel,
                "line": n,
                "problem": drafter._cell(problem, 140),
                "evidence": drafter._cell(evidence, 200),
                "fix": drafter._cell(fix, 160),
                "dropped": False,
            }
        self.add_evidence(dict(rec, id=qid, own=True))
        more = (
            ""
            if sev == "🟡"
            else (
                f" Now record_evidence(id='{qid}', ...): the home it duplicates, the standard, "
                "the sibling path or a run."
            )
        )
        return f"{qid} added, with its quote as evidence.{more}"

    def t_record_evidence(
        self,
        id: str = "",
        claim: str = "",
        run: str = "",
        file: str = "",
        line=0,
        quote: str = "",
        tree: str = "head",
        **_,
    ) -> str:
        qid = str(id or "").strip().upper()
        with self._lock:
            f = self.findings.get(qid)
        if not f:
            return f"REJECTED: no finding {qid or id!r}: add_finding first (ids Q1, Q2, ...)"
        rec, why = self.evidence_rec(
            qid, f["severity"], claim, run, file, line, quote, tree or "head"
        )
        if not rec:
            return why
        self.add_evidence(rec)
        return f"recorded for {qid}"

    def t_finish(
        self,
        points: str = "",
        searched: str = "",
        standards: str = "",
        rechecked: str = "",
        overlaps: str = "",
        lead: str = "",
        **_,
    ) -> str:
        with self._lock:
            serious = [
                f["id"]
                for f in self.findings.values()
                if not f["dropped"] and f["severity"] in drafter.SERIOUS
            ]
        lacking = [q for q in serious if not self.own_evidence(q)]
        if lacking:
            return (
                f"REJECTED: {', '.join(lacking)} has no evidence beside its own line: record_evidence(id=...) for "
                "each (the home it duplicates, the standard, the sibling path, or a run), or drop_finding"
            )
        pts = [
            drafter._cell(p, 90)
            for p in re.split(r";|\n", str(points or ""))
            if p.strip()
        ][:3]
        if not pts:
            return "REJECTED: points: 1 to 3 points for the Code quality row, separated by ';'"
        self.finish = {"points": pts}
        self.notes = {
            k: drafter._cell(v, 400)
            for k, v in (
                ("searched", searched),
                ("standards", standards),
                ("rechecked", rechecked),
                ("overlaps", overlaps),
                ("lead", lead),
            )
            if str(v or "").strip()
        }
        self.pass_done = True
        return "finished"

    # --- the card -----------------------------------------------------------------------------------------------------
    def our_quality(self) -> list[str]:
        """Our last review's Code quality row and its findings, for a follow-up's re-check."""
        text = drafter._read(self.pr / "ours.md")
        return [
            x
            for x in text.splitlines()
            if x.startswith(("| Code quality", "| Q", "| N"))
        ][:12]

    def card(self) -> str:
        plus = sum(f["plus"] for f in self.files)
        minus = sum(f["minus"] for f in self.files)
        overlap = drafter._read(self.ws / "overlap_section.md").strip()
        head = [
            f"TASK: code quality pass. {self.pr_line()}",
            "Description: " + drafter._cell(self.meta["description"], 700),
        ]
        ours = self.our_quality() if self.round == "Follow-up" else []
        if ours:
            head += [
                "Our last review's Code quality row and findings (re-check each open quality finding at head: "
                "FIXED or OPEN, with evidence, in finish's rechecked):"
            ] + [drafter._cell(x, 300) for x in ours]
        head += [
            "Overlapping work in other open PRs (the checks' section; leads for rule 8):",
            rt.clip(overlap, OVERLAP_CAP) if overlap else "(none listed)",
        ]
        task = (
            "\nJudge the code at head. Read the changed code's hunks first (diff_hunk by id). For each new helper, "
            "constant, table, message or fixture, search for an existing home (grep_code by name, synonyms and "
            "distinctive literals) before you accept it; read the standards documents the instructions name "
            "(read_lines, tree base); check that each behaviour the PR claims has a test. For each kept finding: "
            "add_finding with the exact line as its quote, then record_evidence for the home it duplicates, the "
            "standard's line, the sibling path or a run. Then finish with the Code quality row's points and the "
            "notes. You may call several tools in one turn. You have about "
            f"{int(self.s['quality_steps'])} rounds; tool results older than the last "
            f"{int(self.s['keep_results'])} are hidden behind a note of what they were."
        )
        budget = self.card_chars() - sum(len(x) + 1 for x in head) - len(task) - 60
        pmap = [
            f"PR map ({len(self.files)} files, +{plus}/-{minus}); diff_hunk('H1.2') shows a hunk, diff_hunk('F1') "
            "a file's hunks:"
        ] + drafter.fit(
            drafter.map_lines(self.files, self.short, self.delta),
            max(1200, budget),
            "files",
            "diff_hunk by path",
        )
        return "\n".join(head + pmap) + "\n" + task

    # --- the run ------------------------------------------------------------------------------------------------------
    def run_lane(self) -> bool:
        """The pass: one model phase within its caps, then quality.md. False when no model answer came."""
        self.start()
        try:
            self.limiter.phase(
                "quality",
                float(self.s["quality_wall_min"]) * 60,
                int(self.s.get("quality_token_budget") or 0),
            )
            self.guarded(
                "quality",
                lambda: self.loop(
                    "quality",
                    self.card(),
                    self.s["effort"],
                    int(self.s["quality_steps"]),
                    int(self.s["quality_max_tokens"]),
                ),
            )
            self.limiter.phase()
            ok = self.answered > 0
            passes = self.limiter.summary().get("passes") or {}
            with self._lock:
                if "quality" in passes and "quality" in self.header["passes"]:
                    p = passes["quality"]
                    self.header["passes"]["quality"].update(
                        requests=p["requests"],
                        tokens=p["tokens"],
                        reasoning_tokens=p["reasoning"],
                    )
            if ok:
                self.mark_missing()
                self.write_quality()
            self.header.update(
                written=ok,
                stopped=self.stopped,
                finished=drafter._now(),
                light=self.row_light(),
                findings=sum(1 for f in self.findings.values() if not f["dropped"]),
            )
            self.save_header()
            return ok
        finally:
            self.limiter.close()
            self.write_tokens()
            with self._lock:
                if self.log:
                    self.log.close()
                    self.log = None

    # --- quality.md ---------------------------------------------------------------------------------------------------
    def row_light(self) -> str:
        lights = [f["severity"] for f in self.findings.values() if not f["dropped"]]
        return min(lights, key=lambda x: drafter.SEV_ORDER.get(x, 5), default="🟢")

    def refs(self, qid: str) -> str:
        with self._lock:
            recs = [
                e
                for e in self.evidence
                if e["id"] == qid
                and not e.get("own")
                and e.get("kind") not in ("none", None)
            ]
        return ", ".join(dict.fromkeys(self.ref_of(e) for e in recs))

    def write_quality(self) -> None:
        light = self.row_light()
        q = self.header.get("passes", {}).get("quality") or {}
        status = str(q.get("status") or "")
        pts = list(self.finish.get("points") or [])
        if not self.finish:
            pts = [
                f"the pass ended unfinished ({drafter._cell(status or self.stopped or 'no finish', 70)}); what it "
                "recorded is kept"
            ]
        with self._lock:
            live = sorted(
                (f for f in self.findings.values() if not f["dropped"]),
                key=lambda f: drafter.SEV_ORDER[f["severity"]],
            )
        lines = [
            f"| Code quality | {light} {WORDS[light]}"
            + "".join(f"<br>• {p}" for p in pts)
            + " |",
            "",
        ]
        if live:
            lines += ["| # | Sev | file:line | Finding |", "|---|---|---|---|"]
            for f in live:
                ref = self.refs(f["id"])
                ev = (
                    (f["evidence"] + (f" ({ref})" if ref else ""))
                    if f["evidence"]
                    else ref
                )
                lines.append(
                    f"| {f['id']} | {f['severity']} | `{self.short(f['file'])}:{f['line']}` | "
                    f"**{f['problem']}**"
                    + (f"<br>• evidence: {ev}" if ev else "")
                    + f"<br>• fix: {f['fix']} |"
                )
            lines.append("")
        n = self.notes
        lines += [
            "notes:",
            f"- searched: {n.get('searched') or 'not said'}",
            f"- standards read: {n.get('standards') or 'not said'}",
        ]
        if self.round == "Follow-up":
            lines.append(
                f"- open quality findings re-checked (follow-ups): {n.get('rechecked') or 'not said'}"
            )
        lines += [
            f"- overlaps: {n.get('overlaps') or 'not said'}",
            f"- lead (pre-existing, not this PR's): {n.get('lead') or 'none'}",
        ]
        tot = self.limiter.summary()
        model = self.s.get("name") or self.s.get("model")
        lines.append(
            f"- the pass: the quality binding {self.s.get('binding')} ({model}), "
            f"{tot.get('requests', 0)} requests, {tot.get('tokens', 0)} tokens, "
            f"{self.limiter.elapsed() / 60:.1f} min"
            + (f"; {self.stopped}" if self.stopped else "")
        )
        (self.ws / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def member_text(cfg: dict) -> str:
    """The code quality reviewer as the agentic install rendered it (<agentic>/team_members); "" without one."""
    from rkmon_lib import paths

    try:
        found = sorted(
            (pathlib.Path(paths.get(cfg, "agentic")) / "team_members").glob(MEMBER_GLOB)
        )
    except paths.PathsError:
        return ""
    return drafter._read(found[0]) if found else ""


def settings(cfg: dict) -> dict | None:
    """The quality role's client settings, its run caps the pass's own (quality_wall_min, quality_token_budget); None
    without a binding."""
    from rkmon_lib import bindings

    s = bindings.client_settings(cfg, ROLE)
    if s is None:
        return None
    return dict(
        s,
        review_wall_min=s["quality_wall_min"],
        review_token_budget=s["quality_token_budget"],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="The code quality pass on the quality role's binding."
    )
    ap.add_argument("ws")
    ap.add_argument("n")
    ap.add_argument(
        "--config",
        default=None,
        help="a JSON file whose keys replace the configuration's",
    )
    a = ap.parse_args(argv)
    from rkmon_lib import applied, config
    from rkmon_lib.store import load_config

    try:
        cfg = load_config()
        if a.config:
            cfg = config.with_override(cfg, a.config)
    except config.ConfigError as exc:
        return applied.refuse(exc)
    s = settings(cfg)
    if s is None:
        print(
            "the quality role has no binding (roles.quality naming a model_bindings entry)"
        )
        return 1
    t0 = time.time()
    sess = QualitySession(a.ws, a.n, cfg, s, member=member_text(cfg))
    ok = sess.run_lane()
    tot = sess.limiter.summary()
    print(
        f"PR #{a.n}: quality.md {'written' if ok else 'NOT written'} in {time.time() - t0:.0f}s: "
        f"{tot.get('requests', 0)} requests, {tot.get('tokens', 0)} tokens"
        + (f"; {sess.stopped}" if sess.stopped else "")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
