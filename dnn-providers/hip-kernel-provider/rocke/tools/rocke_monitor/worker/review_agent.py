# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Constrained first-round reviewer. The RAW local model (config worker_model) gets a FIXED menu of vetted,
non-shell tools to inspect the worktree and run the repo's own tests on the review machine, then
writes an independent review. The model never supplies a shell command: every tool builds
its argv from validated parameters with shell=False (reviewer_tools.py). No network, no GitHub, no writes.

Usage: python3 review_agent.py <workspace dir> <PR number>
Env: NUM_BATCH (default 256), MAX_STEPS (default 20), STALL_S (900), HANG_S (120); the paths, the repository, rocKE's
folder, the model and its server come from the job's RKMON_* environment (rkenv.py). Each request's tokens go to
<RUN_DIR>/tokens.jsonl as pass "classic" with its step (reviewer_client.Budget.count), the draft's total to tokens_total.json
(reviewer_client.TokenLog).
"""

import functools
import http.client
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reviewer_client  # noqa: E402
import reviewer_tools  # noqa: E402
import rkenv  # noqa: E402
from reviewer_client import RunnerHung  # noqa: E402

WS = pathlib.Path(sys.argv[1])
N = sys.argv[2]
# The first-pass model and its limits (config worker / models.draft, as RKMON_WORKER_JSON); the environment variables
# below still override them for a run by hand.
SETTINGS = reviewer_client.settings()
for _var, _key in (
    ("NUM_BATCH", "num_batch"),
    ("MAX_STEPS", "max_steps"),
    ("STALL_S", "stall_s"),
    ("HANG_S", "hang_s"),
):
    if os.environ.get(_var):
        SETTINGS[_key] = int(os.environ[_var])
MODEL = SETTINGS["model"]
NUM_BATCH = SETTINGS["num_batch"]
MAX_STEPS = SETTINGS["max_steps"]
# A request that sends nothing for this long is abandoned and sent again. Prompt processing sends nothing
# either, but a local runner logs its progress, and stream_chat counts that progress as an answer.
STALL_S = SETTINGS["stall_s"]
# A local runner that took the prompt and logged no progress for this long is wedged (ollama_stream.RunnerHung).
HANG_S = SETTINGS["hang_s"]
MAX_HANGS = SETTINGS["max_hangs"]
REPO = rkenv.get("RKMON_REPO")
TREE = reviewer_tools.Tree(WS, N)
PR = TREE.pr
LOG = open(WS / "agent_log.txt", "w", encoding="utf-8")
TOKENS = reviewer_client.TokenLog(WS, N, mode="classic", model=MODEL)


def logw(s):
    LOG.write(s + "\n")
    LOG.flush()


# ---- fixed, validated tools (no model-supplied shell): reviewer_tools.py ----
DISPATCH = {
    name: functools.partial(fn, TREE) for name, fn in reviewer_tools.CLASSIC.items()
}
TOOLS = reviewer_tools.CLASSIC_SCHEMAS


GOLDEN_DATA = re.compile(r"/golden/[^ ]*\.json$")
# Generated data (logic tables, instance lists, ...): collapsed only when large.
BULK_DATA = re.compile(r"\.(json|ya?ml|csv|txt|kdp)$")
BULK_MIN_LINES = 300
BULK_MIN_FILES = 50  # a data extension changed in this many files is a mass edit
DATA_KEY = re.compile(r"[+-]\s*(?:-\s*)?\"?([A-Za-z_][\w.]*)\"?\s*:")


def collapse_golden(diff):
    """Replace the hunks of golden-hash JSON files, and of large or mass-edited generated
    data files, with a one-line summary that names the most-changed keys."""
    out = []
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    per_ext = {}
    for part in parts:
        m = BULK_DATA.search(part.split("\n", 1)[0])
        if m:
            per_ext[m.group(1)] = per_ext.get(m.group(1), 0) + 1
    for part in parts:
        first = part.split("\n", 1)[0]
        body = part.split("\n")
        changed = [
            l
            for l in body
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
        ]
        golden = GOLDEN_DATA.search(first)
        bulk = BULK_DATA.search(first)
        mass = bulk and per_ext.get(bulk.group(1), 0) >= BULK_MIN_FILES
        if not golden and not (bulk and (len(changed) >= BULK_MIN_LINES or mass)):
            out.append(part)
            continue
        head = [
            l
            for l in body
            if not l.startswith(("@@", "+", "-", " ")) or l.startswith(("+++", "---"))
        ]
        plus = sum(1 for l in changed if l.startswith("+"))
        minus = len(changed) - plus
        keys = {}
        for l in changed:
            m = DATA_KEY.match(l)
            if m:
                keys[m.group(1)] = keys.get(m.group(1), 0) + 1
        top = ", ".join(
            f"{k} x{v}" for k, v in sorted(keys.items(), key=lambda kv: -kv[1])[:6]
        )
        kind = "golden data" if golden else "generated data"
        out.append(
            "\n".join(head).rstrip("\n")
            + f"\n[{kind} hunks omitted: +{plus} / -{minus} lines"
            + (f"; most-changed keys: {top}" if top else "")
            + "; use read_file / grep_code on the file to inspect entries]\n"
        )
    return "".join(out)


# The whole prompt (book + context) must fit the model's context with room to work; the diff gets what the other stages
# leave, never less than DIFF_MIN_CHARS. Both follow worker.context (reviewer_client.prompt_budget): a gateway model with a
# smaller context than a local model's gets a smaller prompt, not a refusal.
TOTAL_BUDGET_CHARS, DIFF_MIN_CHARS = reviewer_client.prompt_budget(
    SETTINGS.get("context")
)
SECTION_CAP_CHARS = 8_000  # one review, thread or comment
CODE_FIRST = re.compile(r"\.(py|cpp|cc|hpp|h|c|cmake|txt)$|CMakeLists\.txt$")


def cap_sections(text):
    """Truncate each '## ' section (one review / thread / comment) past the cap."""
    out = []
    for part in re.split(r"(?m)^(?=## )", text):
        if len(part) > SECTION_CAP_CHARS:
            part = (
                part[:SECTION_CAP_CHARS]
                + f"\n...[section truncated: {len(part) - SECTION_CAP_CHARS} more chars]\n"
            )
        out.append(part)
    return "".join(out)


def cap_diff(diff, budget):
    """Fit a huge diff in the budget: every file keeps its header; hunks are added
    code files first, then the rest, while they fit; the others are marked."""
    if len(diff) <= budget:
        return diff
    parts = re.split(r"(?m)^(?=diff --git )", diff)
    heads = []
    for part in parts:
        lines = part.split("\n")
        heads.append(
            "\n".join(
                l
                for l in lines
                if not l.startswith(("@@", "+", "-", " "))
                or l.startswith(("+++", "---"))
            ).rstrip("\n")
            + "\n"
        )
    used = sum(len(h) for h in heads)
    order = sorted(
        range(len(parts)),
        key=lambda i: (not CODE_FIRST.search(parts[i].split("\n", 1)[0]), i),
    )
    full = set()
    for i in order:
        extra = len(parts[i]) - len(heads[i])
        if used + extra <= budget:
            full.add(i)
            used += extra
    out = [
        (
            parts[i]
            if i in full
            else heads[i] + "[hunks omitted for size; use read_file / grep_code]\n"
        )
        for i in range(len(parts))
    ]
    return (
        f"[diff too large for one prompt: hunks shown for {len(full)} of {len(parts)} files, "
        "code files first; the rest list only their headers]\n" + "".join(out)
    )


def stage(name, header, diff_budget=TOTAL_BUDGET_CHARS):
    p = PR / name
    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else "(none)"
    if name == "diff.patch":
        text = cap_diff(collapse_golden(text), diff_budget)
    elif name in ("reviews.md", "threads.md", "comments.md"):
        text = cap_sections(text)
    return f"\n\n===== {header} =====\n" + text


book = (WS / "review_book.md").read_text(encoding="utf-8", errors="replace")
before_diff = "".join(
    [
        stage("meta.md", "PR METADATA AND DESCRIPTION"),
        stage("reviews.md", "OTHER REVIEWERS (ours omitted — be independent)"),
        stage("threads.md", "INLINE REVIEW THREADS (RESOLVED/OPEN)"),
        stage("comments.md", "CONVERSATION COMMENTS (others)"),
        stage("commits.md", "COMMITS"),
        stage("files.txt", "CHANGED FILES"),
        stage("checks.txt", "CI CHECKS"),
    ]
)
diff_budget = max(DIFF_MIN_CHARS, TOTAL_BUDGET_CHARS - len(book) - len(before_diff))
context = before_diff + stage("diff.patch", "FULL DIFF", diff_budget)

system = (
    "You are an INDEPENDENT first-round code reviewer. Read the review book below and follow it: its "
    "workflow, repo rules, judgement standards, and especially its section 5 output format.\n"
    "You have a fixed set of tools (read_file, grep_code, run_pytest, rule_scan, merge_conflicts) that "
    "work on the PR's head worktree on the review machine (tests that need a GPU skip when it has none, each "
    "skip printed with its reason). USE them to verify things yourself — run the "
    "relevant tests, grep the code, check merge conflicts — do not just trust the diff or other reviewers. "
    "REQUIRED before you conclude: if the PR adds or changes any test, use run_pytest to actually run the "
    "most relevant changed test file(s), and base your 'Our checks' row on what the run printed "
    "(pass/fail counts). Always run rule_scan and merge_conflicts at least once. Never state a test outcome, "
    "coverage claim, or merge result you did not produce with a tool; if you did not run it, write 'not run'. "
    "Other reviewers' comments are provided ONLY so you can check whether they were properly addressed, "
    "not to copy. Form your OWN findings and NEVER claim evidence you did not produce with a tool.\n"
    "When done, reply with ONLY the review in the book's section 5 Markdown format and no tool call.\n\n"
    "===== REVIEW BOOK =====\n" + book
)
user = (
    f"Next review is: https://github.com/{REPO}/pull/{N}/changes. Please consider the "
    "existing comments from other reviewers and whether they've been fixed. We want our own findings; we "
    "don't want to duplicate existing reporting. HOWEVER, I want to also make sure previous comments have "
    "been addressed properly. I also want a judgement call on the utility of the PR and whether the "
    "changes are positive." + context
)
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat(msgs, think=None, tools=TOOLS):
    """One streamed chat request to the first-pass model (reviewer_client: Ollama or an OpenAI-compatible endpoint); the
    chunks are joined back into one message. A local Ollama runner sometimes hangs on a fresh load: it takes the prompt
    and logs nothing more (about one fresh load in eight; each would sit until the stall timeout and cost the draft).
    stream_chat sees that in the runner's log within HANG_S and raises RunnerHung; up to
    MAX_HANGS of those are resent at once. Any other failure (nothing for STALL_S, a dropped connection) is resent up
    to max_resends - 1 more times."""
    fails = hangs = 0
    while True:
        try:
            budget = reviewer_client.budget_for(SETTINGS)
            budget.tags = {"pass": "classic", "step": step}
            return {
                "message": reviewer_client.chat(
                    SETTINGS,
                    msgs,
                    tools,
                    think,
                    budget=budget,
                    on_usage=lambda rec: TOKENS.step(rec, "classic", step),
                )
            }
        except RunnerHung as e:
            hangs += 1
            logw(f"  request hung ({hangs} of {MAX_HANGS}): {e}")
            print(
                f"runner hung ({hangs} of {MAX_HANGS}): {e}",
                file=sys.stderr,
                flush=True,
            )
            if hangs >= MAX_HANGS:
                raise
        except (OSError, http.client.HTTPException) as e:
            fails += 1
            logw(f"  request {fails} of {SETTINGS['max_resends']} gave up: {e!r}")
            if fails >= SETTINGS["max_resends"]:
                raise
        # A wedged runner stays busy after the cancel, and a resend queues behind it (and never starts). A local
        # Ollama unloads the model first, as the worker's launcher does, so the resend gets a
        # fresh runner; any other endpoint is simply asked again.
        reviewer_client.unload(SETTINGS)
        time.sleep(10)


t0, final, step = time.time(), "", 0
for step in range(MAX_STEPS):
    resp = chat(messages)
    msg = resp.get("message", {})
    calls = msg.get("tool_calls") or []
    messages.append(
        {
            "role": "assistant",
            "content": msg.get("content", ""),
            **({"tool_calls": calls} if calls else {}),
        }
    )
    logw(
        f"\n=== step {step} (t+{time.time()-t0:.0f}s) think={len(msg.get('thinking') or '')}c calls={len(calls)} ==="
    )
    if not calls:
        final = (msg.get("content") or "").strip()
        break
    for call in calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        logw(f"  TOOL {name}({json.dumps(args)[:160]})")
        out = DISPATCH[name](**args) if name in DISPATCH else f"unknown tool {name!r}"
        logw(f"  -> {out[:300].rstrip()}")
        messages.append({"role": "tool", "tool_name": name, "content": out})

# Forced finalize: if the loop ended without a review (e.g. the model spent its whole
# budget thinking, or hit MAX_STEPS), demand the review once more with thinking OFF.
# The tools stay in the request: they open the prompt, and dropping them made the runner process the whole
# conversation again (a 234k-token prompt takes about 13 min, past the stall timeout).
# A tool call asked for anyway is refused, once, from the same cached prompt.
if not final:
    logw(f"\n=== finalize (t+{time.time()-t0:.0f}s): forcing review, think off ===")
    messages.append(
        {
            "role": "user",
            "content": "Stop investigating. Output ONLY your first-round review now, in the review book's section 5 "
            "Markdown format (prior-items table ending with Bots/CI and Our checks, net-new findings table "
            "or 'Net-new: none.', utility line, legend). No preamble, no further tool calls.",
        }
    )
    for _ in range(2):
        msg = chat(messages, think=False).get("message", {})
        final = (msg.get("content") or "").strip()
        calls = msg.get("tool_calls") or []
        if final or not calls:
            break
        logw(f"  finalize asked for {len(calls)} tool call(s): refused")
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        messages += [
            {
                "role": "tool",
                "tool_name": c.get("function", {}).get("name", ""),
                "content": "Refused: no more tool calls. Output the review now.",
            }
            for c in calls
        ]
    logw(f"finalize produced {len(final)}c")

(WS / "review_agent.md").write_text(final + "\n", encoding="utf-8")
runs = sum(1 for m in messages if m.get("role") == "tool")
(WS / "stats_agent.json").write_text(
    json.dumps(
        {
            "steps": step + 1,
            "tool_runs": runs,
            "wall_s": round(time.time() - t0, 1),
            "num_batch": NUM_BATCH,
            "final_chars": len(final),
        }
    ),
    encoding="utf-8",
)
TOKENS.close(steps=step + 1, tool_runs=runs)
print(
    f"PR #{N}: {step+1} steps, {runs} tool runs, {time.time()-t0:.0f}s, review {len(final)} chars"
)
