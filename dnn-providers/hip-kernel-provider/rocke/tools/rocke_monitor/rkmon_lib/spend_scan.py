# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Claude Code's transcripts, read for review spend (rkmon_lib/spend.py): the monitor's main sessions and their
subagents, parsed once and cached by file size and modification time.

Where: config `paths.claude_projects`, else $CLAUDE_CONFIG_DIR/projects (spend.projects_root): one project's folder,
or a projects folder whose project folders are read each. A main session is `<id>.jsonl` in a project's folder, and it
is the monitor's when it holds the monitor's slash command (MONITOR). Its subagents are
`<id>/subagents/agent-<aid>.jsonl`, each beside its `.meta.json` (agentType, description, effort).

A model request is one assistant message id, counted once: its input side as recorded (uncached input, cache writes by
TTL, cache reads) and its output. A transcript records a request's output tokens at the start of its stream only, so
the output is also estimated from what the request wrote (cost_fp.py's method, the pilot's cost script):
VISIBLE_TOK tokens per character of text and tool input, plus SIGNATURE_TOK tokens per character of thinking
signature. `use` holds, per model, [input, cache write 5 m, cache write 1 h, cache read, output as recorded, output
estimated, requests]. A request is recorded by the model id its pass asked for (`asked`: the session's turns, MAIN, and
the runs of the verifier, a deep review's verifier and the code quality pass), else by the family of the model it
reports (`family`: the other runs). The model guard: a request whose reported model is of another family than the id
its pass asked for, ignoring case (a gateway reports its own spelling), is counted in its run's or turn's `off`:
{asked id: {reported model: requests}}; an asked id of no family (deploy_tree.family) is not compared. `as_asked`
gives a pass read with no ids asked for as a pass given `asked` records it, but for `off`: one read, recorded both
ways (tools/round_trip.py prices one week so).

- A subagent run: {id, session, type, desc, effort, brief, role, pr, round, start, end, use}. Its role (verifier,
  deep_verifier, quality, notice, experiment, other) comes from its agent type and description (`role_of`); its PR and
  round from its description, else its brief's start (`classified`, on every pass).
- A main-session turn: one prompt that is not a tool result, and every request until the next one: {t, cat, nums,
  roles, use}. `cat` is what woke it (`trigger`); `nums` the PRs its review actions name (`review_refs`: a review's
  directory, post, spec, facts or worker files, the review commands and scripts with a PR argument, a verifier or
  lane it starts or hears back from; a PR only listed, grepped for or read from GitHub is not one, nor one whose
  script a command only reads or uses as another PR's template, `sed 's/A/B/'` or `cp`); `roles` the
  roles of the subagents it started or heard back from.

The cache (state/spend_cache.json) keeps each file's size, modification time and parse, its requests under the keys
they were recorded by when it was read: the model id their pass asked for, or a family (a request whose pass asked for
no id, which spend.price_of prices by the family's newest id). A file whose size and time are unchanged is not read
again; a main session that grew is read from where the last pass stopped (it is append-only).
A pass reads at most `max_bytes` new bytes (the tick's budget): what is left waits for the next pass.
A pass given a time, `since`, leaves out each main session last written before it, and the session's subagents with
it, whenever they were written: none of theirs is read or returned, and their cache entries stay as they are.
"""

from __future__ import annotations

import json
import pathlib
import re
import time

from . import deploy_tree, paths, prep, whole_file

CACHE = "spend_cache.json"
MONITOR = "<command-name>/rocke-monitor</command-name>"
BRIEF_CHARS = 4000
BRIEF_KEEP = 800  # of a run's brief, kept in the cache
VERSION = 8  # the cache's layout and the turns' PRs (review_refs, agent_refs): another version is parsed again
QUIET_H = 6  # a main session unchanged this long: its subagents are not looked at again
# cost_fp.py's output estimate: tokens per character of visible output (text and tool input) and per character of a
# thinking block's signature, by model family (other families: SIGNATURE_TOK_OTHER)
VISIBLE_TOK = 0.45
SIGNATURE_TOK = {"opus": 0.266}
SIGNATURE_TOK_OTHER = 0.28
USE_LEN = 7  # input, cache write 5 m, cache write 1 h, cache read, output recorded, output estimated, requests
# A PR's number in a tool call's command or path: its digits, as many as it has, the first not 0.
PR_NUMBER = r"[1-9]\d*"
# A review directory a run's description or brief, or a tool call's command or path, names, by prep's rule for the name
# (prep.REVIEW_DIR): group 1, its PR.
REVIEW_DIR = re.compile(rf"\b{prep.REVIEW_DIR.pattern}")
# A review action on PR N in a tool call's command or path, besides its review directory (REVIEW_DIR): the post
# pr<N>_<round>.md, the scripts and files named after it, its worker folder, the review commands and scripts that take
# N, and its review notes. A GitHub read or a tree of the PR is not one: gates, merges and ticks read them too.
REVIEW_REF = re.compile(
    rf"[/\\]pr({PR_NUMBER})(?:_|\.md)|\b(?:post|spec|facts|in|prep|spot)({PR_NUMBER})(?!\d)|"
    rf"workers/pr({PR_NUMBER})/|PR_(?:FOLLOWUP|FIRST_ROUND)_({PR_NUMBER})|"
    r"\brk\s+(?:post|review-prep|dispatch-pack|spec-post|review-start|review-recheck|facts|ci-fails|stage-ours|"
    r"classify|spec|premises|"
    r"draft-lint|evidence-check|clash|mutants|ctest-labels|moved-refs|import-probe)\s+(?:-\S+\s+)*"
    rf"({PR_NUMBER})\b|"
    rf"\b(?:post_pr|post_dry|post_real|go_agent_n|cinow|failjobs)\.sh\s+({PR_NUMBER})"
)
# The roles of a review's runs: its verifier, a deep review's verifier (whose description says DEEP_REVIEW, as
# offload.md has it: it runs the deep model) and the code quality pass.
DEEP_REVIEW = "deep review"
VERIFIERS = ("verifier", "deep_verifier")
REVIEW_ROLES = (*VERIFIERS, "quality")
EXPERIMENT_WORDS = (
    "replay",
    "gold",
    "bench",
    "bake",
    "fanout",
    "fan-out",
    "mirror",
    "experiment",
    "evidence",
    "batch",
    "probe",
    "model check",
    "pilot",
    "lens",
    "eval",
    "arm ",
    "hybrid",
    "shadow",
)
# The PR a run's description or brief names: "#N" or "PR N", N of any number of digits (PR_NUMBER): group 1.
REVIEW_PR = re.compile(rf"(?:#|\bPR\s*#?)\s*({PR_NUMBER})\b", re.I)


MAIN = "orchestrator"  # the pass of a main session's turns, in `asked`


def family(model: str | None) -> str:
    """The key of a request whose pass asked for no id: the family of the model it reports (deploy_tree.family), else
    its id in lower case, or "unknown"."""
    return deploy_tree.family(model) or (model or "").lower() or "unknown"


def role_of(agent_type: str, desc: str) -> str:
    """A subagent run's role: experiment (review-cost experiments: replays, gold runs, bake-offs, fan-out lenses, the
    Sonnet copies, the hybrid arms), notice (an overlap notice's draft), verifier, deep_verifier (a verifier whose
    description says DEEP_REVIEW), quality (the code quality pass), or other (any other task).
    """
    t, d = agent_type or "", (desc or "").lower()
    if (
        t.startswith("fanout")
        or t.endswith("-s55")
        or t == "rocke-review-c"
        or any(w in d for w in EXPERIMENT_WORDS)
    ):
        return "experiment"
    if "notice" in d:
        return "notice"
    names_pr = bool(REVIEW_PR.search(desc or ""))
    if t == "rocke-review-verifier" or (
        names_pr and re.match(r"\s*(verify|re-review)\b", d)
    ):
        return "deep_verifier" if DEEP_REVIEW in d else "verifier"
    if t == "rocke-code-quality-reviewer" or (
        names_pr and ("quality lane" in d or "quality pass" in d)
    ):
        return "quality"
    return "other"


def pr_round(desc: str, brief: str = "") -> tuple[int | None, str]:
    """The PR and round a run's description names, else its brief: (pr, "fr" | "fu<k>" | "fu" | ""). The PR is the
    one it names as a PR (REVIEW_PR) or by its review directory (REVIEW_DIR); a bare number is none.
    """
    pr, rnd = None, ""
    for text in (desc or "", (brief or "")[:BRIEF_CHARS]):
        if pr is None:
            m = REVIEW_PR.search(text) or REVIEW_DIR.search(text)
            pr = int(m.group(1)) if m else None
        if not rnd:
            m = re.search(r"\b(?:fu|follow-up\s*)(\d{1,3})\b", text, re.I)
            rnd = (
                f"fu{m.group(1)}"
                if m
                else (
                    "fr"
                    if re.search(r"\bfr\b|first[- ]round", text, re.I)
                    else "fu" if re.search(r"follow-?up", text, re.I) else ""
                )
            )
    return pr, rnd


def trigger(text: str) -> str:
    """What woke a main-session turn (orch_split.py's categories)."""
    t = (text or "")[:4000].lower()
    if "<task-notification>" in t:
        for key, cat in (
            ("tick_loop", "tick loop"),
            ("tick loop", "tick loop"),
            ("review_driver", "review driver"),
            ("review driver", "review driver"),
            ("keep the wsl", "keep-alive"),
        ):
            if key in t:
                return cat
        return "agent or job done"
    if "this session is being continued" in t:
        return "compaction"
    if "<command-name>/rocke-monitor" in t:
        return "watchdog" if "watchdog" in t else "monitor command"
    return "user message"


def new_use() -> list:
    return [0] * USE_LEN


def add_use(use: dict, key: str, u: dict, out_est: float, first: bool) -> None:
    """One transcript entry of a request into `use` (under KEY, a model id or a family): its input side and recorded
    output once (`first`), its estimated output always (each entry of a request carries one more of its content blocks).
    """
    row = use.setdefault(key, new_use())
    if first:
        cc = (
            u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
        )
        written = int(u.get("cache_creation_input_tokens") or 0)
        w1h = int(cc.get("ephemeral_1h_input_tokens") or 0)
        row[0] += int(u.get("input_tokens") or 0)
        row[1] += max(written - w1h, 0)
        row[2] += w1h
        row[3] += int(u.get("cache_read_input_tokens") or 0)
        row[4] += int(u.get("output_tokens") or 0)
        row[6] += 1
    row[5] += out_est


def asked_for(asked: dict | None, role: str) -> str:
    """The model id the pass of ROLE (MAIN: a main session's turns) asks for in ASKED, "" for none: the id its requests
    are recorded under, else each under its family (`_record`)."""
    return (asked or {}).get(role) or ""


def as_asked(scan: dict, asked: dict | None) -> dict:
    """SCAN, a pass's {runs, turns, left} read with no ids asked for, as a pass given ASKED records it (`_record`): the
    requests of each run and turn whose pass asks for an id (`asked_for`) under that id, its rows by family summed;
    any other as it is. The model guard's `off` is not worked out, as a family does not name the model a request
    reports. SCAN itself is not changed."""

    def moved(item: dict, role: str) -> dict:
        want, rows = asked_for(asked, role), list((item.get("use") or {}).values())
        if not want or not rows:
            return item
        return dict(item, use={want: [sum(column) for column in zip(*rows)]})

    return dict(
        scan,
        runs=[moved(r, r.get("role") or "") for r in scan.get("runs") or []],
        turns=[moved(t, MAIN) for t in scan.get("turns") or []],
    )


def _record(
    use: dict, off: dict, want: str, model: str, u: dict, content, first: bool
) -> None:
    """One transcript entry of a request (add_use): under WANT, the id its pass asked for, else under the family of
    the model it reports; once per request, into `off` when that family is not WANT's (the model guard).
    """
    fam = family(model)
    add_use(use, want or fam, u, written(content, fam), first)
    asked = deploy_tree.family(want)
    if first and asked and asked != fam:
        seen = off.setdefault(want, {})
        seen[model] = seen.get(model, 0) + 1


def written(content, fam: str) -> float:
    """The estimated output tokens of one entry's content blocks."""
    vis = sig = 0
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            vis += len(b.get("text") or "")
        elif b.get("type") == "tool_use":
            vis += len(json.dumps(b.get("input") or {}, ensure_ascii=False))
        elif b.get("type") == "thinking":
            sig += len(b.get("signature") or "")
    return VISIBLE_TOK * vis + SIGNATURE_TOK.get(fam, SIGNATURE_TOK_OTHER) * sig


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text") or "" for b in content or [] if isinstance(b, dict))


def _refs(text: str) -> set[str]:
    text = text or ""
    return {m.group(1) for m in REVIEW_DIR.finditer(text)} | {
        g for m in REVIEW_REF.finditer(text) for g in m.groups() if g
    }


# A command's segments (`&&`, `||`, `;`, `|`, newlines), and the uses of a PR that are no review work on it: its script
# as a renumbering template (`sed 's/10101/10202/g' facts10101.sh > facts10202.sh`), as a copy's source
# (`cp facts10101.sh facts10202.sh`), or only read (`cat facts10101.sh`).
SEGMENT = re.compile(r"&&|\|\||[;|\n]")
RENUMBER = re.compile(rf"\bs([/|#,])({PR_NUMBER})\1({PR_NUMBER})\1")
READERS = {
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "grep",
    "rg",
    "file",
    "stat",
    "ls",
    "diff",
    "cmp",
}
TEMPLATE = re.compile(
    rf"(?:^|[/\\])(?:post|spec|facts|in|prep|spot)({PR_NUMBER})\w*\.(?:sh|py)$"
)


def _segment_refs(seg: str) -> set[str]:
    """One command segment's review references, without the PRs it uses only as a template or a copy's source, or
    whose script it only reads."""
    refs = _refs(seg)
    if not refs:
        return refs
    words = seg.strip().split()
    cmd = words[0].rsplit("/", 1)[-1] if words else ""
    src = {m.group(2) for m in RENUMBER.finditer(seg) if m.group(2) != m.group(3)}
    if src:  # sed 's/A/B/' A's script > B's script: A is the template
        return refs - src
    if cmd == "cp":
        args = [w for w in words[1:] if not w.startswith(("-", ">", "2>"))]
        if len(args) >= 2:
            return refs - ({r for a in args[:-1] for r in _refs(a)} - _refs(args[-1]))
    if cmd in READERS:
        return refs - {
            m.group(1)
            for w in words[1:]
            for m in [TEMPLATE.search(w.strip("'\""))]
            if m
        }
    return refs


def review_refs(text: str) -> set[str]:
    """The PRs a tool call's command or path acts on as a review (REVIEW_REF), segment by segment, without the PRs a
    segment uses only as a renumbering template or a copy's source, or whose script it only reads (a PR's facts script
    used as the template of other PRs' scripts must not book their turns to it)."""
    return set().union(*(_segment_refs(s) for s in SEGMENT.split(text or "")))


def agent_refs(desc: str, role: str) -> set[str]:
    """The PR a verifier's or lane's description names (none for another subagent)."""
    m = REVIEW_PR.search(desc or "") if role in REVIEW_ROLES else None
    return {m.group(1)} if m else set()


def _tool_refs(b: dict) -> tuple[set[str], str]:
    """(the PRs a tool call acts on as a review, the role of the subagent it starts or "")."""
    inp = b.get("input") or {}
    if not isinstance(inp, dict):
        return set(), ""
    if b.get("name") in ("Agent", "Task"):
        desc = str(inp.get("description") or "")
        role = role_of(str(inp.get("subagent_type") or ""), desc)
        return agent_refs(desc, role), role
    if b.get("name") == "Read":  # a script only read: as `cat` (_segment_refs)
        return _segment_refs("cat " + str(inp.get("file_path") or "")[:1500]), ""
    return (
        review_refs(
            " ".join(
                str(inp.get(k) or "")[:1500] for k in ("command", "file_path", "path")
            )
        ),
        "",
    )


# --- subagent runs ----------------------------------------------------------------------------------------------------


def parse_run(
    jl: pathlib.Path, meta: dict, session: str, asked: dict | None = None
) -> dict | None:
    """A subagent run from its transcript and meta (None when it made no request): its requests under the id its role
    asked for (`asked`), else by family; `off` only when one ran on another family."""
    use: dict = {}
    off: dict = {}
    seen, brief, start, end = set(), None, "", ""
    desc = str(meta.get("description") or "")
    want = asked_for(asked, role_of(str(meta.get("agentType") or ""), desc))
    try:
        f = open(jl, encoding="utf-8", errors="replace")
    except OSError:
        return None
    with f:
        for line in f:
            if brief is not None and '"usage"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            msg = e.get("message") or {}
            if brief is None and e.get("type") == "user":
                brief = _text(msg.get("content"))[:BRIEF_CHARS]
                continue
            u, model = msg.get("usage"), msg.get("model")
            if (
                e.get("type") != "assistant"
                or not isinstance(u, dict)
                or not model
                or model == "<synthetic>"
            ):
                continue
            mid = msg.get("id") or e.get("uuid")
            t = str(e.get("timestamp") or "")[:19]
            start, end = start or t, t or end
            _record(use, off, want, model, u, msg.get("content"), mid not in seen)
            seen.add(mid)
    if not seen:
        return None
    run = {
        "id": jl.stem.replace("agent-", ""),
        "session": session,
        "type": str(meta.get("agentType") or ""),
        "desc": desc[:160],
        "effort": str(meta.get("effort") or ""),
        "brief": (brief or "")[:BRIEF_KEEP],
        "start": start,
        "end": end,
        "use": use,
    }
    return classified(dict(run, off=off) if off else run)


def classified(run: dict) -> dict:
    """A run with its role, PR and round, from its type, description and kept brief (worked out on every pass, so a
    change to the rules needs no new parse)."""
    pr, rnd = pr_round(run["desc"], run.get("brief") or "")
    return dict(run, role=role_of(run["type"], run["desc"]), pr=pr, round=rnd)


# --- main sessions ----------------------------------------------------------------------------------------------------


def parse_main(
    path: pathlib.Path, ent: dict, max_bytes: int, asked: dict | None = None
) -> int:
    """Read main session `path` on from ent["offset"] (at most `max_bytes`, whole lines only) into ent: whether it is
    the monitor's (`monitor`), its turns, and the message ids of its last turn (`ids`); its requests under the id the
    session asked for (`asked`, MAIN), else by family. Returns the bytes read.
    """
    turns, ids = ent.setdefault("turns", []), set(ent.get("ids") or [])
    cut = 0
    with open(path, "rb") as f:
        f.seek(ent.get("offset", 0))
        for raw in f:
            if not raw.endswith(b"\n") or cut + len(raw) > max_bytes:
                break  # a line still being written, or past the budget: the next pass reads it
            cut += len(raw)
            _main_line(raw.decode("utf-8", errors="replace"), ent, turns, ids, asked)
    ent["offset"] = ent.get("offset", 0) + cut
    ent["ids"] = sorted(i for i in ids if i)
    return cut


def _main_line(
    line: str, ent: dict, turns: list, ids: set, asked: dict | None = None
) -> None:
    """One line of a main session into its turns (`ids`: the message ids of the last turn); a turn's `off` only when
    one of its requests ran on another family than the session asked for."""
    if not ent.get("monitor") and MONITOR in line:
        ent["monitor"] = True
    is_user = '"type":"user"' in line or '"type": "user"' in line
    if (not is_user and '"usage"' not in line) or (is_user and '"tool_result"' in line):
        return
    try:
        e = json.loads(line)
    except ValueError:
        return
    if not isinstance(e, dict) or e.get("isSidechain"):
        return
    msg = e.get("message") or {}
    if e.get("type") == "user" and not e.get("isMeta"):
        text = _text(msg.get("content"))
        cat = trigger(text)
        turn = {
            "t": str(e.get("timestamp") or "")[:19],
            "cat": cat,
            "nums": [],
            "roles": [],
            "use": {},
        }
        if (
            cat == "agent or job done"
        ):  # the notification names the run: its PR, its role
            m = re.search(r"<(?:summary|description)>(.*?)</", text, re.S)
            agent = (
                re.search(r'Agent "([^"]+)"', m.group(1)) if m else None
            )  # not a background command's
            if agent:
                role = role_of("", agent.group(1))
                turn["roles"], turn["nums"] = [role], sorted(
                    agent_refs(agent.group(1), role)
                )
        turns.append(turn)
        ids.clear()
        return
    u, model = msg.get("usage"), msg.get("model")
    if (
        e.get("type") != "assistant"
        or not isinstance(u, dict)
        or not model
        or model == "<synthetic>"
        or not turns
    ):
        return
    turn, mid = turns[-1], msg.get("id") or e.get("uuid")
    want = asked_for(asked, MAIN)
    off = turn.get("off") or {}
    _record(turn["use"], off, want, model, u, msg.get("content"), mid not in ids)
    if off:
        turn["off"] = off
    ids.add(mid)
    for b in msg.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            refs, role = _tool_refs(b)
            if refs and len(turn["nums"]) < 20:
                turn["nums"] = sorted(set(turn["nums"]) | refs)[:20]
            if role and role not in turn["roles"]:
                turn["roles"].append(role)


# --- the pass ---------------------------------------------------------------------------------------------------------


def _stat(p: pathlib.Path) -> list | None:
    try:
        s = p.stat()
    except OSError:
        return None
    return [s.st_size, s.st_mtime_ns]


def load_cache(store) -> dict:
    try:
        c = json.loads((store.dir / CACHE).read_text(encoding="utf-8"))
        return (
            c
            if isinstance(c, dict) and c.get("v") == VERSION
            else {"v": VERSION, "files": {}}
        )
    except (OSError, ValueError):
        return {"v": VERSION, "files": {}}


def save_cache(store, cache: dict) -> None:
    """CACHE written whole (whole_file.write), compact, at the mode of the files the monitor and the writer share
    (worker/perms.py)."""
    whole_file.write(
        store.dir / CACHE,
        json.dumps(cache, separators=(",", ":"), ensure_ascii=False),
        paths.worker_module("perms").SHARED_FILE_MODE,
    )


def main_sessions(root) -> list[pathlib.Path]:
    """The main sessions under `root`: its own `<id>.jsonl` (a project's folder) and those of each project folder in it
    (a Claude config's projects folder, $CLAUDE_CONFIG_DIR/projects). A session's own folder (`<id>/`, beside
    `<id>.jsonl`: its subagents and tool results) is no project folder."""
    root = pathlib.Path(root)
    if not root.is_dir():
        return []
    mains = sorted(root.glob("*.jsonl"))
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (root / f"{d.name}.jsonl").exists():
            mains += sorted(d.glob("*.jsonl"))
    return mains


def scan(
    root,
    store,
    max_bytes: int | None = None,
    asked: dict | None = None,
    since: float | None = None,
) -> dict:
    """One pass over `root` (a project's folder or a projects folder: main_sessions): {"runs", "turns", "left"}
    (`left`: bytes not read for the budget), the cache updated. Only the monitor's main sessions and their subagents
    are kept. `asked`: the model id each pass asks for, by role (MAIN for the session's turns, REVIEW_ROLES for a
    review's runs), under which what is read now is recorded. `since`: a time in seconds since the epoch, before which
    a main session last written is left out with its subagents (None: none is)."""
    cache = load_cache(store)
    files = cache["files"]
    budget = max_bytes if max_bytes is not None else 1 << 62
    left = 0
    runs, turns = [], []
    for p in main_sessions(root):
        key, stat = str(p), _stat(p)
        if stat is None or (since is not None and stat[1] < since * 1e9):
            continue
        ent = files.get(key)
        if ent is None or stat[0] < ent.get("offset", 0):
            ent = files[key] = {"offset": 0}
        if ent.get("stat") != stat:
            todo = stat[0] - ent.get("offset", 0)
            if todo > budget:
                left += todo - max(budget, 0)
            if budget > 0:
                budget -= parse_main(p, ent, budget, asked)
            if ent.get("offset", 0) >= stat[0]:
                ent["stat"] = stat
        if not ent.get("monitor"):
            ent.pop("turns", None)
            continue
        session = p.stem
        for t in ent.get("turns") or []:
            turns.append(dict(t, session=session))
        quiet = time.time() - stat[1] / 1e9 > QUIET_H * 3600
        if (
            ent.get("stat") == stat and ent.get("subs") is not None and quiet
        ):  # its runs as cached
            runs += [
                classified(files[k]["run"])
                for k in ent["subs"]
                if (files.get(k) or {}).get("run")
            ]
            continue
        sub = p.with_suffix("") / "subagents"
        if not sub.is_dir():
            continue
        subs, complete = [], True
        for mp in sorted(sub.glob("agent-*.meta.json")):
            jl = mp.with_name(mp.name[: -len(".meta.json")] + ".jsonl")
            jkey, jstat = str(jl), _stat(jl)
            if jstat is None:
                continue
            rent = files.get(jkey)
            if rent is None or rent.get("stat") != jstat:
                if jstat[0] > budget:
                    left += jstat[0]
                    complete = False
                    if rent is None:
                        continue
                else:
                    budget -= jstat[0]
                    try:
                        meta = json.loads(mp.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        meta = {}
                    rent = files[jkey] = {
                        "stat": jstat,
                        "run": parse_run(jl, meta, session, asked),
                    }
            subs.append(jkey)
            if rent.get("run"):
                runs.append(classified(rent["run"]))
        ent["subs"] = subs if complete else None
    save_cache(store, cache)
    return {"runs": runs, "turns": turns, "left": left}
