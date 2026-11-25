# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The offload passes (offload.mode on or shadow): the reviewer model does a review's substance, with evidence for
every claim, in passes of the same constrained tool loop as review_agent.py (reviewer_tools.py: vetted tools, no model
shell; its own probes and mutants only in the sandbox):

  draft     its own review of the PR, by the review book: findings (N1, ...) and what it ran (D1, ...) -> draft.md
  quality   the code quality pass, by the quality team member's instructions rendered for it: Q1, ... -> quality.md
  regrade   every prior item at this head (other reviewers', bots', ours), by the verifier's rules: P1, ... -> regrade.md
  assemble  after the checks: the post, by the verifier's Output Format, from the three and the checks' report, with
            evidence for its Bots / CI and Our checks points (C1, ...) -> post_draft.md and notes_draft.md

Evidence: every tool run gets an id (R1, R2, ...) and a record in runs.jsonl: the tool, its arguments, the argv, the
exit, the seconds, a digest of its output reduced to what must match on a re-run, and an excerpt. record_evidence ties
a claim to a run, or to a file:line quote in one of the PR's trees that it checks at once (a wrong quote is refused, so
the model can fix it), in evidence.jsonl. After each pass, a claim of its output with no evidence is recorded there as
"none" for the auditor. `rk evidence-check N` checks it all again, deterministically (rkmon_lib/offload_evidence.py).

Usage: python3 offload_agent.py <review folder> <PR number> <pass name>... [--pr-dir DIR]
  The review folder is workers/pr<N> (offload on) or workers/pr<N>/offload (shadow); the PR's staged data (fetch_pr.py)
  is its pr/, or the shadow's parent's (--pr-dir). The model's settings are RKMON_WORKER_JSON's; the offload's
  RKMON_OFFLOAD_JSON's (on or as a shadow, the round, line 1, the audit model's name, the passes' budget shares, step
  and time limits); a run by hand reads both from `rk env`. Each request's tokens go to <RUN_DIR>/tokens.jsonl with its
  pass and step (reviewer_client.Budget.count), and the run's totals per pass to tokens_total.json
  (reviewer_client.TokenLog). Exit 0 when the passes asked for wrote their output; 1 when the draft pass wrote none; 2
  on a usage error.
"""

import http.client
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reviewer_client  # noqa: E402
import review_sandbox as sandbox  # noqa: E402
import reviewer_tools as rt  # noqa: E402
import rkenv  # noqa: E402

PASSES = ("draft", "quality", "regrade", "assemble")
OUTPUT = {
    "draft": "draft.md",
    "quality": "quality.md",
    "regrade": "regrade.md",
    "assemble": "post_draft.md",
}
# The budget role each pass takes (offload.budget_shares names draft, quality and regrade; the assembly is the draft's).
ROLE = {
    "draft": "draft",
    "quality": "quality",
    "regrade": "regrade",
    "assemble": "draft",
}
IDS = {"draft": "ND", "quality": "Q", "regrade": "P", "assemble": "C"}
ROW_ID = re.compile(r"^\|\s*([NDQPC]\d{1,3})\s*\|")
LIGHTS = {"🔴": "red", "🟠": "orange", "🟡": "yellow", "🟢": "green", "⚪": "white"}
WORDS = {v: k for k, v in LIGHTS.items()}
NOTES_MARK = "=====NOTES====="
# The team members whose instructions the passes take, by role: the code quality reviewer's (the quality pass) and the
# review verifier's (the re-grade and the assembly), as setup rendered them.
MEMBERS = {"quality": "rocke_*quality*.md", "verifier": "rocke_*verifier.md"}
PROMPT_CAP = 400_000  # characters of staged context one pass sends (a whole diff fits a 1M-token context)


def light(text: str) -> str:
    """The first severity light in a text ("" for none); a word (red, orange, ...) counts."""
    m = re.search("|".join(LIGHTS), text or "")
    if m:
        return m.group(0)
    w = str(text or "").strip().lower()
    return WORDS.get(w, "")


EVIDENCE_RULES = """\
EVIDENCE (required). Every claim you make needs evidence you produced with a tool in this session:
- ran: a tool run. Each run's result starts with its id, [R1], [R2], ...
- read: an exact quote of one or more whole lines of a file (read_lines shows them numbered), in the PR's head tree
  ("head"), the head merged with develop ("merged") or the merge base ("base").
Record each one with record_evidence: the claim's id (the row's id: N1 for a finding, D1 for a check you ran, Q1 for a
quality finding, P1 for a prior item, C1 for a point of the Bots / CI or Our checks rows), its severity light, the
claim in a few words, and either run=<Rk> or file, line, quote (and tree). A quote is checked at once: when it is
refused, read the lines again and quote them exactly. A claim with no evidence is marked "no evidence" for the
auditor, who drops or escalates it: never claim what you did not produce with a tool. Mark (ran) or (read) on the
point each piece of evidence backs, as the post does.
"""

TOOL_MAP = """\
The instructions below were written for an agent with a shell. Use your tools for what they say to run: read_file and
read_lines for files (tree "base" for the merge base, as `git show <merge base>:<path>`), grep_code for `git grep`,
git_diff and git_log for the PR's history, run_pytest for the repository's tests, run_mutant for a mutant (a pytest
plugin that undoes a claim; the tool runs the tests without it, then with it), write_probe and run_probe for a probe,
ci_fails and job_log for CI. Skip what is about that agent's own session (background runs, the prompt cache,
SubagentHandback, worker_grace.py, the scratch directory, SendMessage, gh): you write nothing but your final answer.
"""


def _s(desc: str, props: dict | None = None, required=()) -> dict:
    return {"description": desc, "props": props or {}, "required": list(required)}


SCHEMAS = {
    "read_file": _s(
        "Read a file from the PR head worktree (the first 8000 characters). Param: path (relative to the "
        "worktree root).",
        {"path": "string"},
        ["path"],
    ),
    "read_lines": _s(
        "Numbered lines of a file, to quote exactly. Params: path; start, end (line numbers, at most 400 "
        "lines); tree: head (default), merged (head merged with develop), base (the merge base with "
        "develop), or checks (the checks' report, path checks.md; the assemble pass only).",
        {"path": "string", "start": "integer", "end": "integer", "tree": "string"},
        ["path"],
    ),
    "grep_code": _s(
        "Recursively grep the head worktree for a literal/regex pattern (| separates alternatives). Params: "
        "pattern, optional path.",
        {"pattern": "string", "path": "string"},
        ["pattern"],
    ),
    "git_diff": _s(
        "The PR's diff from its merge base with develop to its head, for one path (or all).",
        {"path": "string"},
    ),
    "git_log": _s(
        "The PR's commits from its merge base to its head that touch a path (or all): the fixing commit of a "
        "prior item.",
        {"path": "string"},
    ),
    "run_pytest": _s(
        "Run the repo's OWN pytest file(s) or node ids (file::test) from the head worktree, as the checks "
        "run them. Param: paths (space or comma separated).",
        {"paths": "string"},
        ["paths"],
    ),
    "rule_scan": _s(
        "Scan the PR diff for repo-rule violations (tracker IDs, absolute perf numbers, kernel-selecting env "
        "reads). Hints only."
    ),
    "merge_conflicts": _s(
        "List files that conflict when the PR is merged with current develop."
    ),
    "write_probe": _s(
        "Write a probe of your own: a bash script run as `bash <probe> <tree root>` that exits 1 while the "
        "finding reproduces, 0 once it is fixed, 2 when this tree cannot show it, and prints a one-line "
        "summary last. It runs in a sandbox (no network, nothing writable but the review folder). Params: "
        "name (letters, digits, _ . -), script.",
        {"name": "string", "script": "string"},
        ["name", "script"],
    ),
    "run_probe": _s(
        "Run a probe: your own (write_probe) or a saved probe of our open findings, on the merged tree "
        "(default) or the head. Prints REPRODUCES, FIXED or ERROR and its summary. Params: name, tree.",
        {"name": "string", "tree": "string"},
        ["name"],
    ),
    "run_mutant": _s(
        "Check that a claim is backed by a test: plugin is the source of a pytest plugin module that undoes "
        "the claim when it loads (monkeypatch the function or constant the claim is about; keep the "
        "code's other contracts); tests are the test files or node ids that should catch it. The tool runs "
        "them without the plugin (the control), then with it, in a sandbox: KILLED when they fail only "
        "with it, SURVIVED when they pass with it, INVALID when the control fails. Params: name, plugin, "
        "tests.",
        {"name": "string", "plugin": "string", "tests": "string"},
        ["name", "plugin", "tests"],
    ),
    "ci_fails": _s(
        "CI at the PR's live head: each check run's latest attempt that failed (FAIL) or still runs (RUN), "
        "with its url, and the counts."
    ),
    "job_log": _s(
        "A CI job's log: its line count, then its failing lines (`<line>:<text>`). Param: job (the numeric "
        "id at the end of a job url).",
        {"job": "string"},
        ["job"],
    ),
    "read_job_log": _s(
        "Lines of a CI job's whole log (after job_log), numbered. Params: job, start, end.",
        {"job": "string", "start": "integer", "end": "integer"},
        ["job"],
    ),
    "record_evidence": _s(
        "Record the evidence for one claim (see EVIDENCE). Params: id (N1, D1, Q1, P1, C1, ...), "
        "severity (🔴 🟠 🟡 🟢 ⚪), claim (a few words), and either run (R7) or file, line, quote, "
        "tree (head, merged, base; checks in the assemble pass).",
        {
            "id": "string",
            "severity": "string",
            "claim": "string",
            "run": "string",
            "file": "string",
            "line": "integer",
            "quote": "string",
            "tree": "string",
        },
        ["id", "claim"],
    ),
}
COMMON = (
    "read_file",
    "read_lines",
    "grep_code",
    "git_diff",
    "git_log",
    "run_pytest",
    "record_evidence",
)
PASS_TOOLS = {
    "draft": COMMON
    + (
        "rule_scan",
        "merge_conflicts",
        "write_probe",
        "run_probe",
        "run_mutant",
        "ci_fails",
        "job_log",
        "read_job_log",
    ),
    "quality": COMMON + ("run_mutant",),
    "regrade": COMMON
    + ("write_probe", "run_probe", "ci_fails", "job_log", "read_job_log"),
    "assemble": (
        "read_lines",
        "merge_conflicts",
        "ci_fails",
        "job_log",
        "read_job_log",
        "record_evidence",
    ),
}
# Tools whose runs are evidence a claim may name (rk evidence-check re-runs the cheap ones).
RUNS = (
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


def schemas(names) -> list[dict]:
    return [
        rt.schema(
            n, SCHEMAS[n]["description"], SCHEMAS[n]["props"], SCHEMAS[n]["required"]
        )
        for n in names
    ]


class Session:
    """One offload run on one PR: its trees, review folder, settings, token log, runs and evidence."""

    def __init__(
        self,
        out: pathlib.Path,
        n: str,
        pr_dir: pathlib.Path | None = None,
        settings: dict | None = None,
        offload: dict | None = None,
        chat=None,
    ):
        self.out, self.n = pathlib.Path(out), str(n)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t = rt.Tree(self.out, n, pr_dir=pr_dir or self.out / "pr")
        self.ws_root = (
            self.t.pr.parent
        )  # workers/pr<N>: the checks' report is its checks/checks.md
        self.s = dict(settings if settings is not None else reviewer_client.settings())
        self.o = dict(
            offload
            if offload is not None
            else json.loads(rkenv.get("RKMON_OFFLOAD_JSON"))
        )
        self.o["round"] = self.o.get("round") or os.environ.get(
            "RKMON_OFFLOAD_ROUND", ""
        )  # the job's (worker_script)
        self.repo = rkenv.get("RKMON_REPO")
        self.agentic = pathlib.Path(
            rkenv.get("RKMON_AGENTIC", required=False) or self.out
        )
        self.chat_fn = chat or reviewer_client.chat
        self.tokens = reviewer_client.TokenLog(
            self.out,
            self.n,
            mode=self.o.get("mode") or "",
            round=self.o.get("round"),
            model=self.s.get("model"),
        )
        self.runs: dict[str, dict] = {
            r["run"]: r for r in _read_jsonl(self.out / "runs.jsonl")
        }
        self.evidence = _read_jsonl(self.out / "evidence.jsonl")
        self.pass_ = ""
        self.log = None
        self.step = 0
        self.header = self._header()

    # --- the run's header: the commits every record refers to -----------------------------------------------------
    def _header(self) -> dict:
        p = self.out / "offload_run.json"
        try:
            h = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            h = {}
        if not h.get("head"):
            h.update(
                pr=int(self.n),
                round=self.o.get("round") or "",
                mode=self.o.get("mode") or "",
                head=self.t.sha("head"),
                merged=self.t.sha("merged"),
                base=self.t.sha("base"),
                model=self.s.get("model") or "",
                started=_now(),
                passes={},
                sandbox=sandbox.kind(),
            )
            self._save_header(h)
        return h

    def _save_header(self, h: dict | None = None) -> None:
        (self.out / "offload_run.json").write_text(
            json.dumps(h or self.header, indent=1), encoding="utf-8"
        )

    # --- logs -----------------------------------------------------------------------------------------------------
    def logw(self, text: str) -> None:
        if self.log:
            self.log.write(text + "\n")
            self.log.flush()

    # --- the tools ------------------------------------------------------------------------------------------------
    def tool(self, name: str, args: dict) -> str:
        if name not in PASS_TOOLS.get(self.pass_, ()):
            return f"unknown tool {name!r} in the {self.pass_} pass"
        try:
            if name in RUNS:
                return self._run(name, args)
            return getattr(self, f"t_{name}")(**args)
        except TypeError as exc:
            return f"bad arguments for {name}: {exc}"

    def _run(self, name: str, args: dict) -> str:
        if name == "run_mutant":
            return self._mutant(args)
        # Only the parameters the tool's schema names, as plain values; a probe's time limit is the offload's.
        args = {
            k: v
            for k, v in (args or {}).items()
            if k in SCHEMAS[name]["props"] and isinstance(v, (str, int, float, bool))
        }
        if name == "run_probe":
            args["timeout"] = int(self.o["probe_timeout_s"])
        text, r, argv = rt.run_tool(self.t, name, args)
        if r is None:
            return text
        rid = self._record_run(name, args, argv, r)
        if name == "run_probe":
            text = f"{rt.probe_verdict(r)}: {text}"
        if name == "merge_conflicts" and r.rc == 0 and not r.out.strip():
            text = "(no conflicts)"
        return f"[{rid}] {text}"

    def _record_run(
        self, name: str, args: dict, argv: list, r: rt.Result, norm: str | None = None
    ) -> str:
        rid = f"R{len(self.runs) + 1}"
        norm = rt.normalize(name, r.out, self.t) if norm is None else norm
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
            "sandbox": sandbox.kind() if name in ("run_probe", "run_mutant") else "",
            "at": _now(),
        }
        self.runs[rid] = rec
        _append(self.out / "runs.jsonl", rec)
        return rid

    def _mutant(self, args: dict) -> str:
        name, plugin, tests = (
            str(args.get("name") or ""),
            str(args.get("plugin") or ""),
            str(args.get("tests") or ""),
        )
        if not rt.NAME.match(name):
            return f"bad mutant name: {name!r}"
        if not plugin.strip() or len(plugin) > rt.CODE_CAP:
            return f"plugin: the source of a pytest plugin module, at most {rt.CODE_CAP} characters"
        path = self.out / "mutants" / f"rkmut_{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plugin, encoding="utf-8")
        timeout = int(self.o["probe_timeout_s"])
        results = {}
        for control in (True, False):
            built = rt.build(
                self.t, "run_mutant", {"name": name, "tests": tests, "control": control}
            )
            if isinstance(built, str):
                return built
            argv, cwd, _t, env = built
            results[control] = (rt.execute(argv, cwd, timeout + 30, env), argv)
        (c, _), (m, argv) = results[True], results[False]
        cn, mn = json.loads(rt.normalize("run_mutant", c.out)), json.loads(
            rt.normalize("run_mutant", m.out)
        )
        cbad = cn["counts"].get("failed", 0) + cn["counts"].get("error", 0)
        mbad = mn["counts"].get("failed", 0) + mn["counts"].get("error", 0)
        verdict = (
            "INVALID (the control fails)"
            if c.rc is None or cbad or not cn["counts"].get("passed")
            else (
                "INVALID (the mutant run timed out)"
                if m.rc is None
                else (
                    f"KILLED by {', '.join(mn['failing'][:4]) or 'the tests'}"
                    if mbad
                    else "SURVIVED"
                )
            )
        )
        verdict_word = verdict.split(" ")[0]
        norm = json.dumps(
            {
                "verdict": verdict_word,
                "control": cn["counts"],
                "mutant": mn["counts"],
                "failing": mn["failing"],
            },
            sort_keys=True,
        )
        rid = self._record_run(
            "run_mutant", {"name": name, "tests": tests}, argv, m, norm=norm
        )
        return (
            f"[{rid}] {verdict}\ncontrol: {_counts(cn)}\nmutant: {_counts(mn)}\n"
            + rt.clip(m.out if m.rc is not None else c.out, 3000)
        )

    def t_read_file(self, path: str = "", **_) -> str:
        return rt.read_file(self.t, path)

    def t_read_lines(
        self, path: str = "", start=1, end=0, tree: str = "head", **_
    ) -> str:
        if tree == "checks":
            return self._checks_lines(start, end)
        return rt.read_lines(self.t, path, start, end, tree)

    def _checks_file(self) -> pathlib.Path:
        return self.ws_root / "checks" / "checks.md"

    def _checks_lines(self, start=1, end=0) -> str:
        if self.pass_ != "assemble":
            return "the checks' report is read in the assemble pass"
        try:
            lines = (
                self._checks_file()
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            return "no checks report (checks/checks.md)"
        return rt.numbered(lines, "checks.md", start, end)

    def t_write_probe(self, name: str = "", script: str = "", **_) -> str:
        if not rt.NAME.match(name or ""):
            return f"bad probe name: {name!r}"
        if not script.strip() or len(script) > rt.CODE_CAP:
            return f"script: a bash script of at most {rt.CODE_CAP} characters"
        if not sandbox.kind():
            return "refused: probes of your own run only in a sandbox, and this machine has none"
        path = self.out / "probes" / f"{name}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            script if script.endswith("\n") else script + "\n", encoding="utf-8"
        )
        return f"written {path.name}: run it with run_probe(name={name!r})"

    def t_read_job_log(self, job: str = "", start=1, end=0, **_) -> str:
        if not re.fullmatch(r"\d{1,15}", str(job or "")):
            return "job: the numeric id job_log read"
        p = self.out / "tmp" / f"joblog_{job}.txt"
        if not p.is_file():
            return f"no log for job {job}: run job_log first"
        return rt.numbered(
            p.read_text(encoding="utf-8", errors="replace").splitlines(),
            f"job {job}",
            start,
            end,
        )

    def t_record_evidence(
        self,
        severity: str = "",
        claim: str = "",
        run: str = "",
        file: str = "",
        line=0,
        quote: str = "",
        tree: str = "head",
        **a,
    ) -> str:
        cid = str(a.get("id") or "").strip().upper()
        if not re.fullmatch(r"[NDQPC]\d{1,3}", cid):
            return "REJECTED: id is the row's id: N1, D1, Q1, P1 or C1 (a letter and a number)"
        if cid[0] not in IDS[self.pass_]:
            return f"REJECTED: the {self.pass_} pass records {', '.join(IDS[self.pass_])} ids"
        rec = {
            "id": cid,
            "pass": self.pass_,
            "severity": light(severity),
            "claim": str(claim or "")[:300],
            "head": self.header.get("head"),
            "at": _now(),
        }
        if run:
            r = self.runs.get(str(run).strip().upper())
            if not r:
                return f"REJECTED: no run {run!r} in this review (runs: {', '.join(list(self.runs)[-8:]) or 'none'})"
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
        elif file:
            try:
                n = int(line or 0)
            except (TypeError, ValueError):
                return "REJECTED: line is a line number"
            if len(str(quote)) > rt.QUOTE_CAP:
                return f"REJECTED: a quote is at most {rt.QUOTE_CAP} characters: quote fewer lines"
            if tree == "checks":
                if self.pass_ != "assemble" or file not in (
                    "checks.md",
                    "checks/checks.md",
                ):
                    return "REJECTED: tree checks is the checks' report, checks.md, in the assemble pass"
                try:
                    lines = (
                        self._checks_file()
                        .read_text(encoding="utf-8", errors="replace")
                        .splitlines()
                    )
                except OSError:
                    return "REJECTED: no checks report"
            else:
                lines = rt.file_lines(self.t, file, tree)
                if isinstance(lines, str):
                    return f"REJECTED: {lines}"
            ok, where = rt.quote_at(lines, n, quote)
            if not ok:
                said = (
                    lines[n - 1].strip()[:200]
                    if 1 <= n <= len(lines)
                    else f"(the file has {len(lines)} lines)"
                )
                return (
                    f"REJECTED: the quote is not at {file}:{n} ({tree}); that line reads {said!r}"
                    + (f"; the quote is at line {where}" if where else "")
                    + ". Quote whole lines exactly."
                )
            rec.update(
                kind="read",
                file=file,
                line=n,
                quote=str(quote),
                tree=tree,
                sha=(
                    self.header.get(
                        "merged"
                        if tree == "merged"
                        else "base" if tree == "base" else "head"
                    )
                    if tree != "checks"
                    else ""
                ),
            )
        else:
            return "REJECTED: give run (a run id such as R3) or file, line and quote"
        self.evidence.append(rec)
        _append(self.out / "evidence.jsonl", rec)
        what = (
            f"ran {rec['run']} ({rec['tool']})"
            if rec["kind"] == "ran"
            else f"read {file}:{rec['line']} ({tree})"
        )
        return f"recorded for {cid}: {what}"

    # --- the loop -------------------------------------------------------------------------------------------------
    def ask(self, msgs: list, tools: list | None, think=None) -> dict:
        """One request (reviewer_client.chat), resent as review_agent.py resends: a hung local runner MAX_HANGS times, other
        failures max_resends times."""
        fails = hangs = 0
        budget = reviewer_client.budget_for(
            self.s, role=ROLE[self.pass_], shares=self.o.get("budget_shares")
        )
        budget.tags = {"pass": self.pass_, "step": self.step}
        while True:
            try:
                return self.chat_fn(
                    self.s,
                    msgs,
                    tools,
                    think,
                    budget=budget,
                    on_usage=lambda rec: self.tokens.step(rec, self.pass_, self.step),
                )
            except reviewer_client.RunnerHung as exc:
                hangs += 1
                self.logw(f"  request hung ({hangs} of {self.s['max_hangs']}): {exc}")
                if hangs >= int(self.s["max_hangs"]):
                    raise
            except (OSError, http.client.HTTPException) as exc:
                fails += 1
                self.logw(
                    f"  request {fails} of {self.s['max_resends']} gave up: {exc!r}"
                )
                if fails >= int(self.s["max_resends"]):
                    raise
            reviewer_client.unload(self.s)
            time.sleep(float(self.o.get("resend_wait_s", 10)))

    def loop(self, system: str, user: str, finalize: str) -> str:
        names = PASS_TOOLS[self.pass_]
        tools = schemas(names)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        steps = int(self.o["max_steps"] or self.s["max_steps"])
        final, t0 = "", time.time()
        for self.step in range(steps):
            msg = self.ask(msgs, tools)
            calls = msg.get("tool_calls") or []
            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    **({"tool_calls": calls} if calls else {}),
                }
            )
            self.logw(
                f"\n=== {self.pass_} step {self.step} (t+{time.time() - t0:.0f}s) "
                f"think={len(msg.get('thinking') or '')}c calls={len(calls)} ==="
            )
            if not calls:
                final = (msg.get("content") or "").strip()
                break
            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                self.logw(f"  TOOL {name}({json.dumps(args)[:160]})")
                out = self.tool(name, args if isinstance(args, dict) else {})
                self.logw(f"  -> {out[:300].rstrip()}")
                msgs.append({"role": "tool", "tool_name": name, "content": out})
        if (
            not final
        ):  # the forced finalize of review_agent.py: thinking off, tools kept in the request, calls refused
            self.logw(f"\n=== {self.pass_} finalize (t+{time.time() - t0:.0f}s) ===")
            msgs.append({"role": "user", "content": finalize})
            for _ in range(2):
                msg = self.ask(msgs, tools, think=False)
                final = (msg.get("content") or "").strip()
                calls = msg.get("tool_calls") or []
                if final or not calls:
                    break
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
                msgs += [
                    {
                        "role": "tool",
                        "tool_name": c.get("function", {}).get("name", ""),
                        "content": "Refused: no more tool calls. Write your final answer now.",
                    }
                    for c in calls
                ]
        return final

    def run_pass(self, name: str) -> bool:
        """One pass: its prompt, the loop, its output file, the claims it left without evidence. False when it wrote
        nothing (its reason is in offload_run.json)."""
        self.pass_ = name
        self.log = open(self.out / f"agent_log_{name}.txt", "w", encoding="utf-8")
        t0 = time.time()
        status = "ok"
        try:
            system, user, finalize = getattr(self, f"prompt_{name}")()
            final = self.loop(system, user, finalize)
            if not final:
                status = "no answer"
            else:
                self.write_output(name, final)
                self.mark_missing(name)
        except PassError as exc:
            status = f"not run: {exc}"
        except (OSError, http.client.HTTPException, RuntimeError, ValueError) as exc:
            status = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
        finally:
            self.log.close()
            self.log = None
        self.header["passes"][name] = {
            "status": status,
            "wall_s": round(time.time() - t0, 1),
            "steps": self.step + 1,
        }
        self._save_header()
        print(f"PR #{self.n} {name}: {status} ({time.time() - t0:.0f}s)")
        return status == "ok"

    def write_output(self, name: str, final: str) -> None:
        if name == "assemble":
            post, sep, notes = final.partition(NOTES_MARK)
            (self.out / "post_draft.md").write_text(
                post.strip() + "\n", encoding="utf-8"
            )
            (self.out / "notes_draft.md").write_text(
                (notes.strip() if sep else "(no notes block)") + "\n", encoding="utf-8"
            )
            return
        (self.out / OUTPUT[name]).write_text(final.strip() + "\n", encoding="utf-8")

    def claims(self, name: str) -> list[tuple[str, str, str]]:
        """(id, light, row) of the claims in a pass's output: its table rows with ids; for the assembly, one per point
        of its Bots / CI and Our checks rows (C1, ...)."""
        path = self.out / OUTPUT[name]
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if name == "assemble":
            out = []
            for row in text.splitlines():
                if row.startswith(("| Bots / CI |", "| Our checks |")):
                    points = row.count("<br>•")
                    for _ in range(points):
                        out.append((f"C{len(out) + 1}", "", row[:200]))
            return out
        out = []
        for row in text.splitlines():
            m = ROW_ID.match(row)
            if m and m.group(1)[0] in IDS[name]:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                out.append((m.group(1), light(" ".join(cells[1:3])), row[:300]))
        return out

    def mark_missing(self, name: str) -> None:
        have = {
            e["id"]
            for e in self.evidence
            if e.get("pass") == name and e.get("kind") != "none"
        }
        for cid, lt, row in self.claims(name):
            if cid not in have:
                rec = {
                    "id": cid,
                    "pass": name,
                    "severity": lt,
                    "claim": row,
                    "kind": "none",
                    "missing": True,
                    "head": self.header.get("head"),
                    "at": _now(),
                }
                self.evidence.append(rec)
                _append(self.out / "evidence.jsonl", rec)

    # --- the passes' prompts --------------------------------------------------------------------------------------
    def staged(self, name: str, header: str, cap: int = 60_000) -> str:
        p = self.t.pr / name
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = "(none)"
        if len(text) > cap:
            text = (
                text[:cap]
                + f"\n...[{len(text) - cap} more characters: read_file / git_diff for the rest]\n"
            )
        return f"\n\n===== {header} =====\n{text}"

    def context(self, with_diff: bool = True) -> str:
        parts = [
            self.staged("meta.md", "PR METADATA AND DESCRIPTION"),
            self.staged("reviews.md", "OTHER REVIEWERS (ours omitted)"),
            self.staged("threads.md", "INLINE REVIEW THREADS (RESOLVED/OPEN)"),
            self.staged("comments.md", "CONVERSATION COMMENTS (others)"),
            self.staged("commits.md", "COMMITS"),
            self.staged("files.txt", "CHANGED FILES"),
            self.staged("checks.txt", "CI CHECKS"),
        ]
        text = "".join(parts)
        if with_diff:
            text += self.staged(
                "diff.patch", "FULL DIFF", max(20_000, PROMPT_CAP - len(text))
            )
        return text

    def trees_line(self) -> str:
        h = self.header
        return (
            f"The PR's trees: head {str(h.get('head'))[:10]} (tree head), merge base with develop "
            f"{str(h.get('base'))[:10]} (tree base), head merged with develop {str(h.get('merged'))[:10]} (tree "
            "merged)."
        )

    def member(self, role: str) -> str:
        """A team member's instructions as setup rendered them (<WORKING_DIR>/agentic/team_members, found by its role:
        MEMBERS), copied into the review folder once as <role>_member.md."""
        dest = self.out / f"{role}_member.md"
        if not dest.is_file():
            found = sorted((self.agentic / "team_members").glob(MEMBERS[role]))
            if not found:
                raise PassError(
                    f"no rendered {role} team member ({MEMBERS[role]}) in {self.agentic}/team_members: the "
                    "container's first start puts it there (rkmon start)"
                )
            shutil.copyfile(found[0], dest)
        return dest.read_text(encoding="utf-8", errors="replace")

    def lessons(self, lane: str) -> str:
        script = self.agentic / "skills" / "rocke-monitor" / "lessons_for.py"
        if not script.is_file():
            return "(no lessons file here)"
        p = subprocess.run(
            [sys.executable, "-B", str(script), lane],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (
            rt.clip(p.stdout, 60_000)
            if p.returncode == 0
            else "(the lessons could not be read)"
        )

    def rk(self, *args, timeout: int = 300) -> str:
        """A read-only rk command's output (the job's RKMON_HOME), or a note when it fails."""
        p = subprocess.run(
            [self.t.rk(), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, TMPDIR=str(self.out / "tmp")),
        )
        return (
            p.stdout
            if p.returncode in (0, 1) and p.stdout.strip()
            else f"(rk {' '.join(args)}: rc {p.returncode})"
        )

    def round_word(self) -> str:
        return (
            "Follow-up"
            if str(self.o.get("round") or "").startswith("fu")
            else "First-round"
        )

    def prompt_draft(self):
        book = self.ws_root / "review_book.md"
        if not book.is_file():
            raise PassError(
                "no review_book.md in the review folder (review_agent_run.py copies it)"
            )
        system = (
            "You are an INDEPENDENT reviewer of one pull request of rocKE, in a public repository. Read the review "
            "book below and follow its workflow, repo rules and judgement standards. Your tools work on the PR's head "
            "worktree; USE them: read the code around every change, run the PR's changed and most relevant tests, "
            "check the merge with develop and CI, write a probe to reproduce a 🔴/🟠 when you can, and check with "
            "run_mutant that a claimed behaviour is caught by a test. Other reviewers' comments are there only so "
            "you do not repeat them: form your own findings. Do not grade prior items and do not write the Code "
            "quality row: other passes do.\n\n" + EVIDENCE_RULES + "\n"
            f"Your final answer is your draft, in exactly this form, and no tool call:\n\n"
            f"## Draft: PR #{self.n} — <title> — <light> {self.round_word()}: <2–5 word reason>\n\n"
            "| # | Sev | file:line | Finding |\n|---|---|---|---|\n"
            "| N1 | <🔴|🟠|🟡> | `<file>:<line>` | **<problem, one short clause>**<br>• impact: <the concrete failure>"
            "<br>• fix: <a few words> |\n\n"
            "| # | Check | Result |\n|---|---|---|\n"
            "| D1 | <what you ran or read, e.g. the PR's changed tests> | <light> <result, e.g. 12 passed> (ran) |\n\n"
            "**Utility:** <🟢 Positive|🟡 Mixed|🔴 Negative> — <8–12 words>\n\n"
            "Severity: 🔴 wrong results on a reachable path, a crash, a build or CI break on merge, or performance "
            "numbers in public history; 🟠 a real defect or a significant gap; 🟡 minor (at most one). With no "
            "finding, write `Net-new: none.` in place of the findings table. The light in the heading is the worst "
            "open severity.\n\n===== REVIEW BOOK =====\n"
            + book.read_text(encoding="utf-8", errors="replace")
        )
        user = (
            f"Review https://github.com/{self.repo}/pull/{self.n}. {self.trees_line()}"
            + self.context()
        )
        return (
            system,
            user,
            (
                "Stop investigating. Output ONLY your draft now, in the form the system message gives, "
                "with the evidence you recorded. No further tool calls."
            ),
        )

    def prompt_quality(self):
        member = self.member("quality")
        prep = self.prep()
        system = (
            "You run the code quality pass of the review of one rocKE PR, by the instructions below: a team "
            "member file written for another agent; take its rules, checks, evidence standards and Output "
            "Format as yours.\n" + TOOL_MAP + "\n" + EVIDENCE_RULES + "\n"
            "Your final answer is the content of quality.md, in the instructions' Output Format, with the ids "
            "Q1, Q2, ... in the table's first column, and no tool call.\n\n"
            "===== THE CODE QUALITY PASS'S INSTRUCTIONS =====\n"
            + member
            + "\n\n===== LESSONS (the code quality pass's) =====\n"
            + self.lessons("quality")
        )
        overlaps = self.rk("overlaps", self.n, "--section")
        user = (
            f"PR #{self.n} ({self.round_word()}). {self.trees_line()}\n"
            + "\n\n===== OVERLAPPING WORK IN OTHER OPEN PRS (the checks' section) =====\n"
            + overlaps
            + (
                (
                    "\n\n===== OUR LAST REVIEW (its open quality findings, for a follow-up) =====\n"
                    + _section(prep.get("facts", ""), "== our last review", "== rk abi")
                )
                if self.round_word() == "Follow-up"
                else ""
            )
            + self.staged("files.txt", "CHANGED FILES")
            + self.staged("diff.patch", "FULL DIFF", PROMPT_CAP // 2)
        )
        return (
            system,
            user,
            (
                "Stop investigating. Output ONLY the content of quality.md now, in the Output Format, "
                "with the evidence you recorded. No further tool calls."
            ),
        )

    def prompt_regrade(self):
        member = self.member("verifier")
        prep = self.prep()
        probes = sorted(
            p.stem for d in (self.t.probes,) if d.is_dir() for p in d.glob("*.sh")
        )
        system = (
            "You re-grade the prior items of one rocKE PR at its head: every item other reviewers, Copilot and "
            "bots raised, and every item of our own earlier reviews, each with a verified status. Your rules are "
            "the verifier's below: What to Check §3 (Prior items), with §2 to test an item before you call it "
            "fixed or wrong, and §6 (hygiene) for the wording; the rest of that file is another pass's.\n"
            + TOOL_MAP
            + "\n"
            + EVIDENCE_RULES
            + "\n"
            "Your final answer is the prior items table, one row per item, and no tool call:\n\n"
            "| # | Prior item | Status |\n|---|---|---|\n"
            "| P1 | <reviewer> (<MM-DD>) <item, ≤10 words> | <light> <STATUS> `<sha>`<br>• <what was fixed, or "
            "what is wrong> (ran)<br>• <what is left, or who must do it> |\n\n"
            "Statuses: FIXED `sha` (the fixing commit, or test evidence at head), OPEN, DEFERRED (quote the "
            "author's reason and judge it), APPROVED `sha` (at an older head: add \"older head; still counts "
            'on develop"), RETRACT (one of our own that proved wrong), and `🟢 None yet` when there is no prior '
            "item. After the table, one line `probes: <name> REPRODUCES|FIXED|ERROR, ...` for the saved probes "
            "you ran (or `probes: none`).\n\n===== THE VERIFIER'S RULES =====\n"
            + member
            + "\n\n===== LESSONS (the verifiers') =====\n"
            + self.lessons("verifier")
        )
        user = (
            f"PR #{self.n} ({self.round_word()}). {self.trees_line()} Saved probes of our open findings: "
            f"{', '.join(probes) or 'none'} (run_probe runs them).\n"
            "\n===== FACTS (rk facts: our reviews, the latest reviews, open threads, the operator's comments) =====\n"
            + prep.get("facts", "(not read)")
            + "\n\n===== CLASSIFICATION SINCE OUR LAST REVIEW (rk classify) =====\n"
            + prep.get("classify", "")
            + self.context(with_diff=False)
        )
        return (
            system,
            user,
            (
                "Stop investigating. Output ONLY the prior items table and the probes line now, with "
                "the evidence you recorded. No further tool calls."
            ),
        )

    def prompt_assemble(self):
        member = self.member("verifier")
        parts = {}
        for name in ("draft", "quality", "regrade"):
            try:
                parts[name] = (self.out / OUTPUT[name]).read_text(encoding="utf-8")
            except OSError:
                parts[name] = (
                    f"(no {OUTPUT[name]}: that pass wrote nothing; say so in the notes)"
                )
        if not (self.out / "draft.md").is_file():
            raise PassError("no draft.md to assemble")
        try:
            checks = self._checks_file().read_text(encoding="utf-8", errors="replace")
        except OSError:
            checks = "(no checks report)"
        line1 = self.o.get("line1") or ""
        audit = self.o.get("audit_name") or "the orchestrator"
        system = (
            "You assemble the post of one rocKE PR review from the reviewer passes' outputs and the checks' "
            "report, by the verifier's Output Format and rules below (block 1, the post; block 2, the notes). "
            "Keep every finding and prior item the passes wrote with their wording and evidence; do not "
            "invent findings. Take the Prior item rows from regrade.md, the Code quality row and Q findings "
            "from quality.md (renumbered as N rows after the draft's), the findings and utility from "
            "draft.md. Write the Bots / CI row from ci_fails and the checks' CI section, and the Our checks "
            "row from the checks' report (read_lines with tree checks).\n"
            + EVIDENCE_RULES
            + "\n"
            f"Line 1 is exactly: {line1}\nThe footer reads: _{self.round_word()} review by the reviewer model "
            f"({self.s.get('model')}), audited by {audit}. Checks: <short list, ending with \"code quality\">. "
            "Flags whether a closer look is needed; not an approval or rejection._\n"
            f"Your final answer is block 1, then a line {NOTES_MARK}, then block 2, which adds one line "
            "`map: <post id> = <source id>` per finding (N1 = N1, N5 = Q1, ...), and no tool call.\n\n"
            "===== THE VERIFIER'S OUTPUT FORMAT AND RULES =====\n" + member
        )
        user = (
            f"PR #{self.n} ({self.round_word()}). {self.trees_line()}\n"
            + "\n===== draft.md =====\n"
            + parts["draft"]
            + "\n===== quality.md =====\n"
            + parts["quality"]
            + "\n===== regrade.md =====\n"
            + parts["regrade"]
            + "\n===== EVIDENCE RECORDED (id, kind, what) =====\n"
            + self.evidence_summary()
            + "\n===== THE CHECKS' REPORT (checks.md) =====\n"
            + rt.clip(checks, 120_000)
            + self.staged("meta.md", "PR METADATA AND DESCRIPTION")
        )
        return (
            system,
            user,
            (
                f"Stop. Output ONLY block 1, the line {NOTES_MARK}, and block 2 now. No further tool "
                "calls."
            ),
        )

    def evidence_summary(self) -> str:
        rows = []
        for e in self.evidence:
            what = (
                f"ran {e.get('run')} {e.get('tool')}"
                if e.get("kind") == "ran"
                else (
                    f"read {e.get('file')}:{e.get('line')} ({e.get('tree')})"
                    if e.get("kind") == "read"
                    else "NO EVIDENCE"
                )
            )
            rows.append(
                f"{e['id']} [{e.get('pass')}] {e.get('severity') or '-'} {what}: {str(e.get('claim'))[:120]}"
            )
        return "\n".join(rows) or "(none)"

    def prep(self) -> dict:
        """rk review-prep's inputs (facts.txt, classify.txt), made once per review folder in prep/."""
        d = self.out / "prep"
        if not (d / "facts.txt").is_file():
            d.mkdir(parents=True, exist_ok=True)
            args = ["review-prep", self.n, "--out", str(d)]
            if self.o.get("round"):
                args[2:2] = ["--round", str(self.o["round"])]
            self.rk(*args, timeout=600)
        out = {}
        for key in ("facts", "classify"):
            try:
                out[key] = (d / f"{key}.txt").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                out[key] = f"(no {key}.txt: rk review-prep did not write it)"
        return out


class PassError(Exception):
    """A pass that cannot start: what it lacks."""


def _section(text: str, start: str, end: str) -> str:
    lines, on, out = (text or "").splitlines(), False, []
    for line in lines:
        if line.startswith(start):
            on = True
        elif on and line.startswith(end):
            break
        if on:
            out.append(line)
    return "\n".join(out) or "(none)"


def _counts(n: dict) -> str:
    return (
        ", ".join(f"{v} {k}" for k, v in sorted(n["counts"].items())) or "nothing ran"
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append(path: pathlib.Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    try:
        return [
            json.loads(x)
            for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    except (OSError, ValueError):
        return []


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pr_dir = None
    if "--pr-dir" in argv:
        i = argv.index("--pr-dir")
        pr_dir = pathlib.Path(argv[i + 1])
        del argv[i : i + 2]
    if len(argv) < 3 or any(p not in PASSES for p in argv[2:]):
        print(__doc__.split("\n\n")[-1])
        return 2
    s = Session(pathlib.Path(argv[0]), argv[1], pr_dir=pr_dir)
    done = {name: s.run_pass(name) for name in argv[2:]}
    s.tokens.close(
        passes_run=list(done),
        ok=[k for k, v in done.items() if v],
        head=s.header.get("head"),
    )
    return 1 if "draft" in done and not done["draft"] else 0


if __name__ == "__main__":
    sys.exit(main())
