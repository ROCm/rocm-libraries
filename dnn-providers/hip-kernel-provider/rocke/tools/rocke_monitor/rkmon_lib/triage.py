# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tick triage: the monitor's Claude session gets a turn only when a tick needs it. By default the toolkit's rule
filter decides (rkmon_lib/tickfilter.py, `rk tick-filter`, as the live tick loop does), with no model; a triage model is
an opt-in (`models.tick`) that also does the routine work. In the container, `rk orchestrate` runs `rk tick` itself every
monitor.tick_interval_minutes minutes and hands its output here (rkmon_lib/orchestrator.py).

- `parse` splits the output into its lines and its board; `classify` places each line by the skill's tick procedure
  (RULES, first match wins): CLAUDE when it needs Claude (an audit, or any outward action: a post, a notice, a hold, an
  issue, a ticket, an arm; or a judgement the skill gives Claude), USER when the skill only tells the user (the
  orchestrator's log prints it, and the board lists what needs them), QUIET when it needs nothing, UNSURE when no rule
  knows it. The incident lines are Claude's while incidents are filed (the merge queue live, with its develop guard)
  and a report before (INCIDENT); any other `!! ` line is Claude's.
- `run`: the triage model, when `models.tick` names one (config.tick_model: "worker", the local worker's model, or a
  binding's name, which needs no Ollama; never the reviewer by itself), reads the classified lines and the board,
  does the routine work with read-only rk commands only (READ_ONLY: the board and the status, a REVIEW_READY's prep,
  the CI of our recent posts), and answers in JSON: the tick's one-line summary, a class for each unsure line, the
  lines it escalates, what to tell the user, and what it did. It can add lines for Claude and never take one away: every
  CLAUDE line goes to Claude whatever it says, and an unsure line it does not place goes to Claude too (`for_claude`).
- A model that does not answer (down, timing out, rate limited past its retries), or that answers no JSON twice,
  raises Down: the orchestrator then gives Claude the whole tick, and the plain `/rocke-monitor tick` for DOWN_MIN
  minutes.
Each triage is one line of state/triage.jsonl (`record`): the tick's exit, the classes, what the model ran, its
requests and tokens. Nothing here writes anywhere else, and nothing goes outward.
"""

from __future__ import annotations

import json
import re
import subprocess

from . import config, features, paths
from .store import now

CLAUDE, USER, QUIET, UNSURE = "claude", "user", "quiet", "unsure"
# The skill's tick procedure, by how a line starts; the first rule that matches places it.
RULES = (
    # The tick's summary, the board's header and the timings: the context of the summary line.
    (
        QUIET,
        (
            "TICK ",
            "develop ",
            "EVENTS",
            "QUEUE:",
            "HOLDS:",
            "ABI ",
            "BATCHES:",
            "MERGE PLAN:",
            "PLAN (",
            "BOARD ",
            "time: ",
        ),
    ),
    # Claude's: an audit, an outward action, or a judgement the skill gives Claude.
    (
        CLAUDE,
        (
            "REVIEW_READY",
            "REVIEW_MANUAL",
            "ARM_READY",
            "QUEUE_NOTICE",
            "TRIAGE ",
            "!! DISARM_NEEDED",
            "!! AUTOMERGE_RISK",
            "!! HOLD_REVIEW_NEEDED",
            "!! HOLD_DISMISSED",
            "!! HELD_MERGED",
            "!! MERGED_OVER_RED",
            "FOLLOWUP_SCAN",
            "!! FOLLOWUP_SCAN",
            "BREAKAGE_OPEN",
            "FIX_CANDIDATE",
            "OVERLAP_NOTICE_DUE",
            "DRIFT_FAIL",
            "GPU_DONE",
            "GPU_FAIL",
            "WEEKLY REPORT",
            "LESSONS REVIEW",
            "!! WEEKLY",
            "!! LESSONS",
            "!! CRITICAL TOOL",
        ),
    ),
    # The user's: the skill only tells them (a heads-up that acts on nothing included: PRIORITY_READY_EARLY).
    (
        USER,
        (
            "PROPOSE",
            "HOLD_STALE",
            "!! AUTOMERGE_JUMP",
            "!! PRIORITY_READY_EARLY",
            "HOLD_FIXED",
            "READY_FOR_APPROVAL",
            "PRIORITY_REQUEST",
            "FIX_LISTED",
            "GPU_NOT_QUEUED",
            "GPU_PENDING_LONG",
            "GPU_NODE_BROKEN",
            "GPU_UNREACHABLE",
            "FEATURE_SKIPPED",
            "REVIEW_FAIL",
            "DRIVER: still running",
            "DRIVER: the configuration",
        ),
    ),
    # Nothing to do: the skill says so, or a step reports its own progress (an advisory line in parentheses too).
    (
        QUIET,
        (
            "ANALYSIS_START",
            "HOLD_REAPPROVED",
            "HOLD_CLEARED",
            "DEVELOP_RED_STALE",
            "DRIFT_START",
            "DRIFT ok",
            "PRIORITY_CLEARED",
            "GPU_STAGING",
            "GPU_QUEUED",
            "GPU_RUNNING",
            "GPU_CANCELLED",
            "GPU_CLEANUP",
            "REVIEW_SETTLING",
            "REVIEW_CLOSED",
            "REVIEW_WAIT",
            "REVIEW_START",
            "REVIEW_SKIP",
            "REVIEW_DROP",
            "REVIEW #",
            "RECHECK",
            "LANDED",
            "BREAKAGE_CLOSED",
            "GATE_START",
            "GATE_SKIP",
            "GATE_DONE",
            "BATCH_GATE_START",
            "BATCH_VALIDATED",
            "BATCH_FAILED",
            "BATCH_BLOCKED",
            "CARRIED",
            "RETENTION",
            "ROLE_STARTED",
            "SANITY",
            "BISECT_START",
            "BISECT_DONE",
            "INCIDENT_CLOSED",
            "CLEANUP",
            "DRIVER:",
            "(",
        ),
    ),
)
# The incident procedure's lines: Claude's while incidents are filed (the develop guard with the merge queue live), a
# report otherwise (develop is checked after every merge in prototype too).
INCIDENT = (
    "!! CRITICAL",
    "INCIDENT_OPEN",
    "CULPRIT",
    "REVERT_OFFER",
    "BISECT_NEEDED",
    "INCIDENT_AGED",
    "INCIDENT_FIX_MERGED",
)

# The rk commands the triage may run, each with the arguments it may take (a regex over " arg arg ...", or a function
# that gives it). None of them writes outward or starts work; review-prep and followup-scan write only their own local
# files.
_N = r"\d{1,7}"


def _review_prep() -> str:
    """review-prep's arguments: a PR, and a round as rk review-prep takes it (prep.ROUND). A call-time import: prep
    loads the review pipeline, and features, which the host runs, imports this module.
    """
    from .prep import ROUND

    return rf" {_N}(?: --round (?:{ROUND.pattern}))?"


READ_ONLY = {
    "status": "",
    "board": "",
    "holds": "",
    "abi": "",
    "batches": "",
    "merge-plan": "",
    "review-status": "",
    "approvals": "",
    "breakages": "",
    "incidents": "",
    "sanity-status": "",
    "priority-requests": "",
    "gpu-status": "",
    "features": "",
    "tickets": "",
    "ledger": rf"(?: {_N})?",
    "jobs": r"(?: [\w.-]{1,64})?",
    "facts": f" {_N}",
    "heads": rf"(?: {_N}){{1,20}}",
    "ci-fails": f" {_N}",
    "job-log": f" {_N}",
    "classify": f" {_N}",
    "review-prep": _review_prep,
    "overlaps": rf"(?: {_N})?(?: --section)?",
    "followup-scan": f" {_N}",
    "conflict-files": r"(?: --days \d{1,3})?",
    "ci-times": r"(?: --days \d{1,3})?",
    "usage": r"(?: --days \d{1,3})?(?: --by-review)?",
}
TOOL = {
    "type": "function",
    "function": {
        "name": "rk",
        "description": (
            "Run one read-only rk command of the monitor and read its output. Only these run: "
            + ", ".join(READ_ONLY)
            + ". Nothing you run writes to GitHub or JIRA."
        ),
        "parameters": {
            "type": "object",
            "required": ["args"],
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'the command and its arguments, such as ["ci-fails", "123"] or ["review-prep", "123"]',
                }
            },
        },
    },
}

STEPS = 12  # tool rounds before the triage must answer
RK_TIMEOUT_S = 300
OUT_MAX = 12000  # characters of one command's output the model reads
DOWN_MIN = 30  # after a triage that failed, Claude gets the plain tick this long
POSTS_H = 24  # our posts this recent get their CI re-checked
LOG = "triage.jsonl"

SYSTEM = """You are the tick triage of the rocKE monitor. The orchestrator ran `rk tick`; you read its output
before Claude does, so that Claude spends a turn only on what needs it. Nothing you do goes outward: your one tool
runs read-only rk commands, and nothing you write is posted anywhere.

Each line of the tick comes with its class: claude (Claude handles it: an audit, a post, a notice, a hold, an issue, a
ticket, an arm), user (the user is told), quiet (nothing to do) or unsure (no rule knows it). Your work:
1. Place every unsure line: claude when handling it could mean writing to GitHub or JIRA, judging a review, or a
   decision only Claude makes; user when the user only needs to know; quiet when it is progress or needs nothing. When
   in doubt, claude.
2. Do the routine work with the rk tool:
   - for each REVIEW_READY #N line, run `rk review-prep N`, so the verification's inputs are ready, and say in "done"
     which directory it wrote;
   - for each recent post listed, run `rk ci-fails N`. When the PR's live head is still the posted head and a check
     failed, add a claude line `CI_AFTER_POST #N <head commit>: <the failed checks>` (Claude judges whether the
     failure is the PR's and posts one CI update). A check still running, all green, or a newer head needs nothing;
   - check whatever else a line asks before anyone handles it, when a read-only command answers it.
   Hand Claude a line of your own for anything that looks inconsistent or wrong (numbers that do not add up, one
   state that contradicts another, a command that fails), saying what you saw, and for a problem that may deserve a
   ticket or an issue: Claude checks it, then files or drafts it.
3. Answer with one JSON object and nothing else:
{"summary": "rocke-monitor <UTC time> · develop <short sha> <CI state> · <count> events (<event types>) · queue: <head or empty> · next: <top item>",
 "unsure": [{"line": "<the unsure line, verbatim>", "class": "claude|user|quiet", "why": "<a few words>"}],
 "claude": [{"line": "<a line for Claude: a tick line verbatim, or CI_AFTER_POST ...>", "why": "<a few words>"}],
 "user": ["<a line the user should read>"],
 "done": ["<each command you ran and what it showed, one line each>"]}
The claude-class lines go to Claude anyway: do not repeat them."""
AGAIN = "Answer now with the JSON object only, as the system message says."


class Down(Exception):
    """The tick-triage model did not answer usefully: Claude gets the tick."""


def unavailable(cfg: dict) -> str:
    """Why no triage model reads the ticks ("" when one does): `models.tick` (config.tick_model) names none (the
    default: the rule filter decides), or names one that is not at hand: the local worker's (worker_gaps) or a
    binding's (its address, model and key). Meanwhile the rule filter decides.
    """
    m = config.model_for(cfg, "tick")
    if m["by"] == "binding":
        return binding_gap(m)
    if m["by"] != "worker":
        return m.get("why") or config.RULES_ONLY
    return "; ".join(worker_gaps(m))


def worker_gaps(w: dict) -> list:
    """What the local worker's settings W (config.worker's, or config.model_for's) lack before a request can go, each
    a features.Need named as the deploy settings file names it: its address and its model; once the start has checked
    the model, a pass of the test chat and the tool-call probe (config.probe_failure: a model that failed either is
    refused). Never a value.
    """
    from . import doctor
    from .features import Need

    out = []
    if not w.get("url"):
        out.append(
            Need.setting(
                "worker.url",
                "the local review loop's endpoint",
                key=doctor.worker_url_key(w),
            )
        )
    if not w.get("model"):
        out.append(Need.setting("worker.model", "the local review loop's model"))
    why = "" if out else config.probe_failure(w)
    if why:
        out.append(
            Need.setting(
                "worker.model", f"a model that passes its checks: {w['model']} {why}"
            )
        )
    return out


def binding_gap(s: dict) -> str:
    """What a binding's client settings lack before a request can go ("" when nothing), a features.Need named as the
    deploy settings file names it: its address and its model, settings; its key when it names a file: the secret its
    api_key_secret names, or the claude login for a claude binding the Claude session's key reaches
    (bindings._as_the_session). A `$NAME` url whose variable is not set is the host configuration's form: its address
    is named by its deploy settings key all the same (doctor.url_key), the variable in its words; a key file, the host
    configuration's form too (the deploy settings file gives a secret): words. Never a value.
    """
    from . import doctor, first_pass, secrets_registry as sr
    from .features import Need

    llm = first_pass.llm_client()
    name = s.get("binding") or "the binding"
    if not llm.base_url(s):
        url = s.get("url")
        return Need.setting(
            f"model_bindings.{name}.url",
            (
                f"its address, which {url!r} gives through a variable that is not set where the monitor runs"
                if url
                else "its address"
            ),
            key=doctor.url_key(name, s.get("kind") or ""),
        )
    if not s.get("model"):
        return Need.setting(f"model_bindings.{name}.model", "the model's id")
    if s.get("api_key_file") and not llm.api_key(s):
        secret = str(s.get("api_key_secret") or "").strip()
        if secret:
            # the entry the binding's key has in the registry, from the binding alone
            named = {"model_bindings": {name: {"api_key_secret": secret}}}
            return Need.secret(
                sr.entry(named, secret), f"the API key of model_bindings.{name}"
            )
        if s.get("key_headers"):
            return Need.secret(
                sr.entry({}, sr.CLAUDE_LOGIN), "the Claude session's key"
            )
        return f"model_bindings.{name}: no key in {s['api_key_file']}"
    return ""


RULE_FILTER = "the rule filter (rk tick-filter, no model)"


def describe(cfg: dict) -> str:
    """Who reads each tick first, in a few words: the triage model and where it comes from, or the rule filter (and
    why, when a model is named but not at hand)."""
    m = config.model_for(cfg, "tick")
    if m["by"] == "none":
        return RULE_FILTER
    gone = unavailable(cfg)
    if gone:
        return f"{RULE_FILTER}; the triage model waits: {gone}"
    if m["by"] == "binding":
        return f"the binding {m.get('binding')} ({m.get('name') or m.get('model')}, {m.get('provider')})"
    return f"the local review loop's model {m.get('model') or '(not set)'}"


def parse(text: str) -> tuple[list[str], str]:
    """The tick's lines, blank ones dropped, and its board: the table and the Needs you list after `BOARD (changed)` or
    `BOARD (hourly)` ("" when it printed none). The BOARD line itself stays among the lines.
    """
    lines, board, inside = [], [], False
    for line in (text or "").splitlines():
        if inside and not line.startswith("time: "):
            board.append(line)
            continue
        inside = line.startswith(("BOARD (changed)", "BOARD (hourly)"))
        if line.strip():
            lines.append(line.rstrip())
    return lines, "\n".join(board).strip()


def classify(line: str, acting: bool) -> str:
    """A tick line's class (CLAUDE, USER, QUIET or UNSURE); `acting`: incidents are filed (features.on(cfg,
    "incident-issues"))."""
    if line.startswith("!! CRITICAL TOOL"):
        return CLAUDE
    if line.startswith(INCIDENT):
        return CLAUDE if acting else USER
    for cls, starts in RULES:
        if line.startswith(starts):
            return cls
    return CLAUDE if line.startswith("!! ") else UNSURE


def classify_all(lines: list[str], acting: bool) -> list[tuple[str, str]]:
    return [(line, classify(line, acting)) for line in lines]


def allowed(args: list[str]) -> str | None:
    """Why the triage may not run `rk <args>` (None when it may): READ_ONLY's commands, with their arguments only."""
    if not args or args[0] not in READ_ONLY:
        return (
            f"refused: rk {args[0] if args else '(nothing)'} is not one of the read-only commands "
            f"({', '.join(READ_ONLY)})"
        )
    rule = READ_ONLY[args[0]]
    if not re.fullmatch(
        rule() if callable(rule) else rule, "".join(" " + a for a in args[1:])
    ):
        return f"refused: rk {args[0]} does not take {' '.join(args[1:])!r} here"
    return None


def _clip(text: str, n: int = OUT_MAX) -> str:
    return (
        text
        if len(text) <= n
        else text[: n * 2 // 3] + "\n… (cut) …\n" + text[-(n // 3) :]
    )


def rk_read(args: list[str]) -> str:
    """One allowed rk command's output (stdout and stderr, clipped) and its exit code."""
    try:
        p = subprocess.run(
            [str(paths.PACKAGE / "rk"), *args],
            capture_output=True,
            text=True,
            timeout=RK_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(rk {args[0]} did not finish: {str(exc)[:200]})"
    return _clip((p.stdout or "") + (p.stderr or "")) + f"\n(exit {p.returncode})"


def recent_posts(store, hours: int = POSTS_H) -> list[dict]:
    """Our review posts of the last `hours` (the ledger), the last one per PR: {pr, head, ts}."""
    from .store import minutes_since

    out = {}
    for r in store.ledger():
        if (
            r.get("kind") == "post"
            and minutes_since(r.get("ts") or "2000-01-01T00:00:00Z") < hours * 60
        ):
            out[r["pr"]] = {
                "pr": r["pr"],
                "head": (r.get("head") or "")[:10],
                "ts": r["ts"],
            }
    return list(out.values())


def brief(
    cfg: dict, rc: int, classes: list[tuple[str, str]], board: str, posts: list[dict]
) -> str:
    """The triage's user message: the classified lines, the board, and the posts whose CI it re-checks."""
    meaning = {0: "OK", 2: "a tool failed", 3: "critical plan items"}.get(
        rc, "unexpected"
    )
    out = [
        f"The tick ({features.headline(cfg)}; rk tick exit {rc}: {meaning}):",
        _clip("\n".join(f"[{c}] {line}" for line, c in classes), 60000),
        "",
    ]
    out.append(
        "The board:\n" + board
        if board
        else "The board did not change since it was last shown (rk board " "prints it)."
    )
    out.append("")
    out.append(
        "Recent posts (re-check their CI): "
        + ("; ".join(f"#{p['pr']} at {p['head']} ({p['ts']})" for p in posts) or "none")
    )
    return "\n".join(out)


def _args(raw) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw.split()
    if isinstance(raw, dict):
        raw = raw.get("args")
    if isinstance(raw, str):
        return raw.split()
    return [str(a) for a in raw or []]


def _json(text) -> dict | None:
    s = str(text or "")
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        v = json.loads(s[i : j + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def _verdict(v: dict) -> dict:
    """The answer in a fixed shape, whatever the model left out or got wrong."""

    def items(key):
        return (
            [x for x in (v.get(key) or []) if isinstance(x, (dict, str))]
            if isinstance(v.get(key), list)
            else []
        )

    def line(x):
        return str(x.get("line") if isinstance(x, dict) else x).strip()

    summary = str(v.get("summary") or "").strip()
    return {
        "summary": summary.splitlines()[0][:300] if summary else "",
        "unsure": {
            line(x): str(x.get("class") or "").strip().lower()
            for x in items("unsure")
            if isinstance(x, dict)
        },
        "claude": [
            {
                "line": line(x),
                "why": str(x.get("why") or "")[:200] if isinstance(x, dict) else "",
            }
            for x in items("claude")
            if line(x)
        ],
        "user": [line(x)[:400] for x in items("user") if line(x)],
        "done": [line(x)[:400] for x in items("done") if line(x)],
    }


def run(
    cfg: dict,
    rc: int,
    classes: list[tuple[str, str]],
    board: str,
    posts: list[dict],
    chat=None,
    rk=rk_read,
) -> dict:
    """The tick-triage model's pass over one tick: its verdict (`_verdict`'s shape) with `ran`, the commands it ran,
    `requests` and `tokens` ([prompt, answer]). Raises Down when the model does not answer usefully. `chat(messages,
    tools)` is the model: by default the local worker's (worker/reviewer_client.py, sharing the parallel reviews' rate
    budget), or a binding's (worker/llm_client.py, through its shared limiter)."""
    budget = None
    if chat is None:
        s = config.model_for(cfg, "tick")
        if s["by"] not in ("worker", "binding"):
            raise Down(s.get("why") or config.RULES_ONLY)
        if s["by"] == "binding":
            from . import first_pass

            llm = first_pass.llm_client()
            budget = llm.limiter_for(s, who="tick", run=False)

            def chat(messages, tools):
                return llm.chat(
                    s,
                    messages,
                    tools=tools,
                    effort=config.bound_effort(cfg, "tick", s),
                    limiter=budget,
                    max_tokens=min(8000, int(s.get("max_output_tokens") or 8000)),
                )

        else:
            from . import models

            client = models.client()
            # The tick's own effort (config.bound_effort), not worker.think (the first review pass's).
            s = dict(s, think=config.bound_effort(cfg, "tick", s))
            budget = client.Budget(
                s, paths.get(cfg, "run") or None, log=lambda text: None, who="tick"
            )

            def chat(messages, tools):
                return client.chat(
                    s, messages, tools=tools, budget=budget, rate_waits=3
                )

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": brief(cfg, rc, classes, board, posts)},
    ]
    ran, requests = [], 0

    def ask(tools):
        nonlocal requests
        requests += 1
        try:
            return chat(msgs, tools) or {}
        except (
            Exception
        ) as exc:  # noqa: BLE001 - any failure of the model: Claude gets the tick
            raise Down(f"{type(exc).__name__}: {str(exc)[:200]}") from None

    ans = None
    for _ in range(STEPS):
        ans = ask([TOOL])
        calls = ans.get("tool_calls") or []
        if not calls:
            break
        msgs.append(
            {
                "role": "assistant",
                "content": ans.get("content") or "",
                "tool_calls": calls,
                **({"blocks": ans["blocks"]} if ans.get("blocks") else {}),
            }
        )
        for tc in calls:
            fn = tc.get("function") or {}
            args = _args(fn.get("arguments"))
            why = (
                allowed(args)
                if fn.get("name") == "rk"
                else f"refused: {fn.get('name')} is not a tool here (rk is)"
            )
            ran.append("rk " + " ".join(args) + (" (refused)" if why else ""))
            msgs.append({"role": "tool", "tool_name": "rk", "content": why or rk(args)})
        ans = None
    v = _json((ans or {}).get("content"))
    if v is None:
        if ans is not None:
            msgs.append({"role": "assistant", "content": ans.get("content") or ""})
        msgs.append({"role": "user", "content": AGAIN})
        v = _json(ask(None).get("content"))
    if v is None:
        raise Down("it answered no JSON object, twice")
    out = _verdict(v)
    spent = getattr(budget, "spent", None)
    if spent is None and hasattr(
        budget, "summary"
    ):  # a binding's limiter: its prompt and completion tokens
        sm = budget.summary()
        spent = [sm["prompt_tokens"], sm["completion_tokens"]]
    out.update(ran=ran, requests=requests, tokens=list(spent or [0, 0]))
    return out


def for_claude(classes: list[tuple[str, str]], verdict: dict | None) -> list[str]:
    """The lines Claude acts on: every CLAUDE line, every unsure line the triage did not place as user or quiet, and
    the lines it escalated. Without a verdict (no triage), every CLAUDE and unsure line.
    """
    placed = (verdict or {}).get("unsure") or {}
    out = [
        line
        for line, c in classes
        if c == CLAUDE or (c == UNSURE and placed.get(line) not in (USER, QUIET))
    ]
    for x in (verdict or {}).get("claude") or []:
        if x["line"] not in out:
            out.append(x["line"])
    return out


def for_user(classes: list[tuple[str, str]], verdict: dict | None) -> list[str]:
    """What the user is told: the USER lines, the unsure lines placed as user, and the triage's own lines."""
    placed = (verdict or {}).get("unsure") or {}
    out = [
        line
        for line, c in classes
        if c == USER or (c == UNSURE and placed.get(line) == USER)
    ]
    return out + [x for x in (verdict or {}).get("user") or [] if x not in out]


def record(cfg: dict, rec: dict) -> None:
    """One line of state/triage.jsonl."""
    p = config.toolkit_state_dir(cfg) / LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict({"ts": now()}, **rec), ensure_ascii=False) + "\n")
