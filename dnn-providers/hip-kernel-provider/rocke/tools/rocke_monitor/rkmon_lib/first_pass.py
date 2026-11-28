# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The first passes of three tasks a Claude role owns (config models.ci_triage, models.quality_draft,
models.notice_draft): a model the operator binds does the first pass and hands the owner leads, never findings.

- ci_triage: the failing CI jobs at a PR's head (`rk ci-fails`, `rk job-log`), read for the first error that explains
  each, as leads for the verifier (`ci_triage.md` in the review directory).
- quality_draft: a first pass over a PR's diff by the code quality pass's rules, as leads for the code quality reviewer
  (`quality_leads.md` in the review directory); the verifier folds the code quality pass in.
- notice_draft: the first draft of an overlap or a drift notice, for the orchestrator, which checks it and writes the
  notice it posts (`rk notice-draft N --kind overlap|drift`).

Who does it, `models.<role>` (config.model_for): "default" (the shipped value; "", "off" and "claude" read the same): no
first pass, the owner does the whole task; "worker": the local worker's model (worker/reviewer_client.py, on the
parallel reviews' shared rate budget); or a model_bindings name (worker/llm_client.py through the binding's shared
limiter: its rpm, tpm and concurrency). The run's caps are the drafter settings' first_pass_wall_min,
first_pass_token_budget and first_pass_max_tokens. A binding to a Claude model runs at the role's effort
(config.role_effort: effort.<role>, else claude.effort); another model gets no effort.

When: ci_triage and quality_draft run in one background job per review (`rk first-pass N --head SHA`), which the
review driver starts once the PR is staged (review.first_pass_start), beside the draft and the checks; `rk review-prep`
copies the leads for the head into the review directory. notice_draft runs when the orchestrator asks for it: the
OVERLAP_NOTICE_DUE and DRIFT_FAIL lines name the command while the role is bound.

Each lead carries evidence: a quote at a line of a source the run was given (a job's log, a file at the PR's head, the
notice's facts). The quotes are checked here, with no model: a lead whose quotes are none of them at their line or
anywhere in their source is dropped, and the file says how many were. The owner checks every lead before using it.

A first pass never holds its task up: a model that is not at hand, fails, answers no JSON twice or runs past its cap is
a miss: one line in state/first_pass.jsonl and the journal, no leads file, and the owner works without it.
"""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import re
import shlex
import sys

from . import config, paths, registry
from .store import now

ROLES = ("ci_triage", "quality_draft", "notice_draft")
DRIVER_ROLES = (
    "ci_triage",
    "quality_draft",
)  # the review's own first passes (`rk first-pass N`)
OWNER = {
    "ci_triage": "the verifier",
    "quality_draft": "the code quality reviewer",
    "notice_draft": "the orchestrator",
}
FILE = {
    "ci_triage": "ci_triage.md",
    "quality_draft": "quality_leads.md",
    "notice_draft": "notice_draft.md",
}
TASK = {
    "ci_triage": "the failing CI jobs at the PR's head",
    "quality_draft": "the PR's diff, by the code quality pass's rules",
    "notice_draft": "an overlap or a drift notice's first draft",
}
LOG = "first_pass.jsonl"
FOLDER = "first_pass"  # under <workers>/pr<N>: the review job's leads
NOTICE_KINDS = ("overlap", "drift")
MAX_LEADS = 8
JOBS_MAX = 4  # the failing CI jobs ci_triage reads
LOG_TAIL = 60  # the last lines of each job's log it is shown, beside its failing lines
SHOWN_CAP = 60000  # characters of material a request carries
FILES_MAX = 40  # the PR's files whose text at the head checks quality_draft's quotes
FILE_CAP = 400000  # bytes of one such file
GRACE_S = 30  # past the run's wall-time cap, before a first pass that has not answered is a timeout

SYSTEM = (
    "You do the first pass of a task another reviewer owns. You hand that reviewer leads, never conclusions: each lead "
    "is one short claim and the exact lines that show it. Quote lines exactly as the sources show them (without the "
    "line numbers), at the line number the source gives. A lead you cannot quote is left out. At most "
    f"{MAX_LEADS} leads, most important first. Answer with one JSON object only, no prose around it: "
    '{"leads": [{"lead": "<one sentence>", "evidence": [{"source": "<a source\'s name as given>", "line": <N>, '
    '"quote": "<the line, exactly>"}]}]}'
)
DRAFT_RULE = (
    ' Add the notice itself: "draft": "<the notice in Markdown: what fails or what is duplicated, which PR owns the '
    'shared work, what the author can do>". Every fact in the draft must be one of your leads.'
)
AGAIN = "Answer now with the JSON object only, as the system message says."
ASK = {
    "ci_triage": (
        "Read the failing CI jobs below. For each failure find the first error that explains it (a compile error, a "
        "failing test and its assertion, a timeout, an infrastructure error such as a lost runner or a failed "
        "download) and say whether the lines show the PR's change or the CI's own trouble."
    ),
    "quality_draft": (
        "Read the PR's diff below by the code quality rules that follow, and give the places where the PR's code "
        "breaks one of them. Quote the line at the PR's head: the source is the file's path, the line its line "
        "number in the new file (the hunk headers give it)."
    ),
    "notice_draft": (
        "Draft the notice the facts below call for, for the PR's author. Quote the facts' lines as your evidence."
    ),
}


class Miss(Exception):
    """A first pass that gives nothing: the owner works without it."""


# --- who does it ------------------------------------------------------------------------------------------------------


def chosen(cfg: dict, role: str) -> dict:
    """Who does the role's first pass (config.model_for): {"by": "none"|"worker"|"binding", ...}."""
    return config.model_for(cfg, role)


def bound(cfg: dict, role: str) -> bool:
    return chosen(cfg, role)["by"] in ("worker", "binding")


def driver_roles(cfg: dict) -> list[str]:
    """The review's first passes a model is bound to (ci_triage, quality_draft)."""
    return [r for r in DRIVER_ROLES if bound(cfg, r)]


def gap(cfg: dict, role: str) -> str:
    """Why the role's bound model is not at hand ("" when it is, or when the role is the default): a binding's
    address, model or key; the local worker's address or model, or its last probe's refusal of it
    (config.probe_failure).
    """
    from . import triage

    m = chosen(cfg, role)
    if m["by"] == "binding":
        return triage.binding_gap(m)
    if m["by"] != "worker":
        return ""
    gone = [k for k in ("url", "model") if not m.get(k)]
    if gone:
        from . import deploy_tree, doctor

        keys = {
            "url": doctor.worker_url_key(m),
            "model": deploy_tree.name("worker.model"),
        }
        return (
            f"the local review loop's {' and '.join(gone)} ({', '.join(keys[k] for k in gone)}) "
            f"{'are' if len(gone) > 1 else 'is'} not set"
        )
    why = config.probe_failure(m)
    return f"the local review loop's model {m['model']} {why}" if why else ""


def model_name(m: dict) -> str:
    return str(m.get("name") or m.get("model_name") or m.get("model") or "")


def effort(cfg: dict, role: str, m: dict) -> str | None:
    """The effort the role's request carries (config.bound_effort): on a Claude binding, the role's effort; on another
    model, the local worker's included, the role's own, and none while it has none."""
    return config.bound_effort(cfg, role, m)


def describe(cfg: dict, role: str) -> str:
    """The role's first pass in a few words, for rk doctor's line of it (doctor.first_pass_checks)."""
    m = chosen(cfg, role)
    if m["by"] == "none":
        return f"default: no first pass, {OWNER[role]} does it alone"
    who = (
        f"the binding {m.get('binding')} ({model_name(m)}, {m.get('provider')})"
        if m["by"] == "binding"
        else f"the local review loop's model {m.get('model') or '(not set)'}"
    )
    g = gap(cfg, role)
    return f"{who}: leads for {OWNER[role]}" + (f"; not at hand: {g}" if g else "")


# --- one run ----------------------------------------------------------------------------------------------------------


def caps(cfg: dict) -> dict:
    """The run's caps: the drafter settings' first_pass_wall_min, first_pass_token_budget and first_pass_max_tokens."""
    from . import bindings

    d = bindings.drafter_settings(cfg)
    return {
        "wall_s": float(d["first_pass_wall_min"]) * 60,
        "tokens": int(d["first_pass_token_budget"]),
        "max_tokens": int(d["first_pass_max_tokens"]),
    }


def llm_client():
    """worker/llm_client.py, the bindings' model client: its folder goes on the import path once, however often the
    client is asked for."""
    folder = str(paths.PACKAGE / "worker")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import llm_client as lc

    return lc


def chat_for(cfg: dict, role: str, m: dict, limits: dict):
    """(chat(messages) -> the answer's text, usage() -> [requests, prompt tokens, answer tokens]) for the role's model:
    a binding through its shared limiter, with the run's token budget and wall-time cap; the local worker on the
    parallel reviews' shared rate budget."""
    n = {"requests": 0}
    if m["by"] == "binding":
        from . import bindings

        llm = llm_client()
        s = dict(
            m,
            review_token_budget=limits["tokens"],
            review_wall_min=limits["wall_s"] / 60,
        )
        lim = llm.limiter_for(s, who=role, run=True)
        e = effort(cfg, role, m)

        def chat(messages):
            n["requests"] += 1
            ans = llm.chat(
                s,
                messages,
                effort=e,
                limiter=lim,
                max_tokens=min(
                    limits["max_tokens"],
                    int(
                        s.get("max_output_tokens")
                        or bindings.binding_defaults()["max_output_tokens"]
                    ),
                ),
            )
            return ans.get("content") or ""

        def usage():
            sm = lim.summary()
            return [n["requests"], sm["prompt_tokens"], sm["completion_tokens"]]

        return chat, usage
    from . import models

    client = models.client()
    # The pass's own effort (`effort`), not worker.think, which is the first review pass's.
    s = dict(
        m,
        think=effort(cfg, role, m),
        max_output_tokens=min(
            limits["max_tokens"],
            int(
                m.get("max_output_tokens")
                or registry.default("worker.max_output_tokens")
            ),
        ),
    )
    budget = client.Budget(
        s, paths.get(cfg, "run") or None, log=lambda text: None, who=role
    )

    def chat(messages):
        n["requests"] += 1
        return (client.chat(s, messages, budget=budget, rate_waits=3) or {}).get(
            "content"
        ) or ""

    def usage():
        p, c = list(getattr(budget, "spent", None) or [0, 0])[:2]
        return [n["requests"], p, c]

    return chat, usage


def numbered(text: str, first: int = 1) -> str:
    return "\n".join(f"{i:>6}| {x}" for i, x in enumerate(text.splitlines(), first))


def ask_json(chat, messages: list[dict]) -> dict:
    """The model's JSON object: one request, and one more when the first answers none. Miss when neither does."""
    from .roles import json_answer

    first = chat(messages)
    try:
        return json_answer(first)
    except ValueError:
        pass
    again = messages + [
        {"role": "assistant", "content": first},
        {"role": "user", "content": AGAIN},
    ]
    try:
        return json_answer(chat(again))
    except ValueError:
        raise Miss("it answered no JSON object, twice") from None


def check(leads, sources: dict[str, list[str]]) -> tuple[list[dict], int]:
    """The leads whose evidence holds: each quote at its line of its source, or found elsewhere in it (the line is
    corrected); a lead with no quote that holds is dropped. (kept, the number dropped).
    """
    sys.path.insert(0, str(paths.PACKAGE / "worker"))
    from review_tools import quote_at

    kept, dropped = [], 0
    for raw in leads if isinstance(leads, list) else []:
        if not isinstance(raw, dict) or not str(raw.get("lead") or "").strip():
            dropped += 1
            continue
        good = []
        for ev in raw.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            src = str(ev.get("source") or "")
            lines = sources.get(src)
            if lines is None:
                continue
            ok, at = quote_at(lines, ev.get("line"), str(ev.get("quote") or ""))
            line = int(ev.get("line")) if ok else at
            if line:
                q = str(ev.get("quote") or "").strip().splitlines()
                good.append(
                    {"source": src, "line": line, "quote": q[0][:300] if q else ""}
                )
        if good:
            kept.append(
                {"lead": " ".join(str(raw["lead"]).split())[:400], "evidence": good}
            )
        else:
            dropped += 1
    return kept[:MAX_LEADS], dropped


def render(role: str, rec: dict, leads: list[dict], draft: str = "") -> str:
    """The leads file: who wrote it, that it holds leads only, each lead and its checked quotes."""
    head = f" at {rec['head'][:10]}" if rec.get("head") else ""
    out = [
        f"# First pass: {TASK[role]}{head}",
        "",
        f"Leads only, from {rec['model']} ({rec['by']}"
        + (f", effort {rec['effort']}" if rec.get("effort") else "")
        + f", {config.model_key(role)}). {OWNER[role][0].upper() + OWNER[role][1:]} checks each lead against its "
        "source before using it; a lead is never a finding by itself, and what it misses is still the owner's to "
        "find. Every quote below was found at its line"
        + (
            f"; {rec['dropped']} lead(s) whose quotes were not were dropped."
            if rec["dropped"]
            else "."
        ),
        "",
    ]
    for i, lead in enumerate(leads, 1):
        out.append(f"{i}. {lead['lead']}")
        for ev in lead["evidence"]:
            out.append(f"   - `{ev['source']}:{ev['line']}`: `{ev['quote']}`")
    if not leads:
        out.append("(no lead held its evidence)")
    if draft:
        out += ["", "## Draft notice (unchecked)", "", draft.strip()]
    return "\n".join(out) + "\n"


def record(cfg: dict, rec: dict, store=None) -> None:
    """One line of state/first_pass.jsonl; a miss also goes to the journal."""
    p = config.toolkit_state_dir(cfg) / LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict({"ts": now()}, **rec), ensure_ascii=False) + "\n")
    if store is not None and rec["status"] != "ok":
        store.journal(
            "firstpass",
            f"{rec['role']}{' #' + str(rec['pr']) if rec.get('pr') else ''}: no first pass ({rec['status']}: "
            f"{rec.get('why', '')[:200]}); {OWNER[rec['role']]} works without it",
        )


def run(
    cfg: dict,
    role: str,
    shown: list[tuple[str, str]],
    sources: dict[str, list[str]],
    out: pathlib.Path,
    head: str = "",
    pr: int = 0,
    extra: str = "",
    store=None,
    chat=None,
) -> dict:
    """The role's first pass on the material `shown` ((title, text) blocks for the model), its quotes checked against
    `sources` ({name: lines}); the leads written to `out` (and its record beside it, out.json). The record:
    {"role", "pr", "head", "by", "model", "effort", "status" (ok, default, missed, failed, timeout), "why", "leads",
    "dropped", "requests", "tokens", "file"}. Never raises: anything that goes wrong is a miss, recorded (`record`).
    """
    m = chosen(cfg, role)
    rec = {
        "role": role,
        "pr": pr,
        "head": head,
        "by": m["by"],
        "model": model_name(m),
        "effort": effort(cfg, role, m) or "",
        "status": "default",
        "why": "",
        "leads": 0,
        "dropped": 0,
        "requests": 0,
        "tokens": [0, 0],
        "file": "",
    }
    out = pathlib.Path(out)
    out.unlink(missing_ok=True)
    if m["by"] == "none":
        return rec  # the default: no first pass, nothing written, nothing recorded
    limits = caps(cfg)
    usage = None
    try:
        if chat is None:
            g = gap(cfg, role)
            if g:
                rec["status"] = "missed"
                raise Miss(g)
            chat, usage = chat_for(cfg, role, m, limits)
        material, size = [], 0
        for title, text in shown:
            block = f"== {title} ==\n{text}"
            if size + len(block) > SHOWN_CAP:
                block = block[: max(0, SHOWN_CAP - size)] + "\n...[cut]"
            material.append(block)
            size += len(block)
            if size >= SHOWN_CAP:
                break
        messages = [
            {
                "role": "system",
                "content": SYSTEM + (DRAFT_RULE if role == "notice_draft" else ""),
            },
            {
                "role": "user",
                "content": ASK[role]
                + (f"\n\n{extra}" if extra else "")
                + "\n\n"
                + "\n\n".join(material),
            },
        ]
        pool = concurrent.futures.ThreadPoolExecutor(1)
        try:
            v = pool.submit(ask_json, chat, messages).result(
                timeout=limits["wall_s"] + GRACE_S
            )
        except concurrent.futures.TimeoutError:
            rec["status"] = "timeout"
            raise Miss(f"no answer within {limits['wall_s'] / 60:g} min") from None
        finally:
            pool.shutdown(wait=False)
        leads, dropped = check(v.get("leads"), sources)
        draft = str(v.get("draft") or "").strip() if role == "notice_draft" else ""
        if role == "notice_draft" and not draft:
            raise Miss("it wrote no draft")
        rec.update(status="ok", leads=len(leads), dropped=dropped, file=str(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(role, rec, leads, draft), encoding="utf-8")
    except (
        Exception
    ) as exc:  # noqa: BLE001 - any failure is a miss: the owner works without the first pass
        if rec["status"] in ("default", "ok"):
            rec["status"] = "failed"
        rec["why"] = " ".join(str(exc).split())[:300] or type(exc).__name__
    if usage is not None:
        try:
            u = usage()
            rec.update(requests=u[0], tokens=[u[1], u[2]])
        except Exception:  # noqa: BLE001 - the counts are a record only
            pass
    if out.parent.is_dir():
        out.with_suffix(".json").write_text(json.dumps(rec), encoding="utf-8")
    record(cfg, rec, store)
    return rec


# --- ci_triage and quality_draft: the review's job -------------------------------------------------------------------


def folder(cfg: dict, n: int) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / FOLDER


JOB_URL = re.compile(r"/job/(\d+)")


def ci_inputs(cfg: dict, n: int, fails=None, job_log=None):
    """ci_triage's material: `rk ci-fails N` and, for up to JOBS_MAX failing jobs, `rk job-log ID`'s failing lines
    and the log's last lines; its sources: each job's whole log. (shown, sources, head), or Miss when no job fails.
    """
    from . import ci_runs

    fails = fails or ci_runs.fails
    job_log = job_log or ci_runs.job_log
    lines, ok = fails(cfg, n)
    if not ok:
        raise Miss("; ".join(lines)[:200])
    head = lines[0].split()[-1] if lines and lines[0].startswith("head ") else ""
    failing = [x for x in lines if x.startswith("FAIL\t")]
    if not failing:
        raise Miss("no failing CI job at the head")
    shown = [("rk ci-fails", "\n".join(lines))]
    sources = {}
    for row in failing[:JOBS_MAX]:
        m = JOB_URL.search(row)
        if not m:
            continue
        job = int(m.group(1))
        hits, got = job_log(cfg, job)
        if not got:
            continue
        try:
            whole = ci_runs.job_log_path(job).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        log = whole.split("\n")
        name = f"job {job}"
        sources[name] = log
        tail_from = max(1, len(log) - LOG_TAIL + 1)
        shown.append(
            (
                f"{name}: {row.split(chr(9))[1]} (failing lines as <N>:<the line>, then the log's last lines)",
                "\n".join(hits[1:])
                + "\n...\n"
                + numbered("\n".join(log[tail_from - 1 :]), tail_from),
            )
        )
    if not sources:
        raise Miss("no failing job's log could be read")
    return shown, sources, head


def quality_inputs(cfg: dict, n: int, head: str, base: str, repo: str):
    """quality_draft's material: the code quality rules and the PR's diff from its merge base; its sources: the PR's
    files at the head."""
    from . import gitops

    sys.path.insert(0, str(paths.PACKAGE / "worker"))
    import quality_lane

    card = quality_lane.member_text(cfg).strip() or quality_lane.GENERIC_CARD
    diff = gitops.git(repo, "diff", "--unified=3", base, head, timeout=120)
    sources = {}
    for f in gitops.git(
        repo, "diff", "--name-only", "--diff-filter=AM", base, head
    ).splitlines()[:FILES_MAX]:
        try:
            text = gitops.git(repo, "show", f"{head}:{f}", timeout=60)
        except gitops.GitError:
            continue
        if len(text) <= FILE_CAP:
            sources[f] = text.split("\n")
    return [
        ("the code quality rules", card),
        (f"the PR's diff ({base[:10]}..{head[:10]})", diff),
    ], sources


def job(cfg: dict, st: dict, store, n: int, head: str) -> list[str]:
    """`rk first-pass N --head SHA`: the review's bound first passes (ci_triage, quality_draft) at that head, each into
    <workers>/pr<N>/first_pass/; one line each."""
    from . import gitops
    from .queue import base_of

    out = []
    d = folder(cfg, n)
    d.mkdir(parents=True, exist_ok=True)
    rec = (st.get("prs") or {}).get(str(n)) or {}
    for role in driver_roles(cfg):
        dest = d / FILE[role]
        try:
            if role == "ci_triage":
                shown, sources, at = ci_inputs(cfg, n)
                if at and head and at[: len(head)] != head and head[: len(at)] != at:
                    raise Miss(f"the head moved to {at[:10]}")
            else:
                repo = paths.get(cfg, "repo")
                live = gitops.fetch_pr(repo, n)
                if head and not live.startswith(head) and not head.startswith(live):
                    raise Miss(f"the head moved to {live[:10]}")
                ref = base_of(cfg, rec, fetch=True)
                base = gitops.git(repo, "merge-base", live, ref).strip()
                shown, sources = quality_inputs(cfg, n, live, base, repo)
        except Exception as exc:  # noqa: BLE001 - no inputs: a miss, recorded
            r = {
                "role": role,
                "pr": n,
                "head": head,
                "by": chosen(cfg, role)["by"],
                "model": model_name(chosen(cfg, role)),
                "status": "missed",
                "why": " ".join(str(exc).split())[:300],
            }
            dest.unlink(missing_ok=True)
            dest.with_suffix(".json").write_text(json.dumps(r), encoding="utf-8")
            record(cfg, r, store)
            out.append(f"{role}: no first pass ({r['why']})")
            continue
        r = run(cfg, role, shown, sources, dest, head=head, pr=n, store=store)
        out.append(
            f"{role}: {r['status']}"
            + (
                f", {r['leads']} lead(s) in {dest}"
                if r["status"] == "ok"
                else f" ({r['why']})"
            )
        )
    return out


def script(cfg: dict, n: int, head: str) -> str:
    """The job's bash: `rk first-pass N --head SHA`."""
    rk = shlex.quote(str(paths.PACKAGE / "rk"))
    home = shlex.quote(str(config.home()))
    return f"{rk} --home {home} first-pass {n} --head {shlex.quote(head)}\n"


def copy(
    cfg: dict, store, rec: dict | None, n: int, out: pathlib.Path, running=False
) -> list[str]:
    """`rk review-prep`: each bound review first pass's leads for the PR's head, copied into the review directory, with
    the line that hands them to their owner; a line saying the owner works without them when there are none ([] for a
    role that is the default)."""
    lines = []
    head = (rec or {}).get("head") or ""
    for role in driver_roles(cfg):
        src = folder(cfg, n) / FILE[role]
        try:
            r = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            r = {}
        at = str(r.get("head") or "")
        same = bool(at and head) and (head.startswith(at) or at.startswith(head))
        if r.get("status") == "ok" and same and src.is_file():
            (out / FILE[role]).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )
            lines.append(
                f"{FILE[role]}: {r.get('leads', 0)} lead(s) from {r.get('model')} ({config.model_key(role)}), leads "
                f"only: put it in {OWNER[role]}'s brief; it checks each lead before using one"
            )
            continue
        why = (
            "still running"
            if running and not same
            else (
                (r.get("why") or r.get("status"))
                if same
                else f"none at the head {head[:10]}"
            )
        )
        lines.append(f"{role}: no first pass ({why}): {OWNER[role]} works without it")
        if store is not None and not same:
            record(
                cfg,
                {
                    "role": role,
                    "pr": n,
                    "head": head,
                    "by": chosen(cfg, role)["by"],
                    "status": "missed",
                    "why": f"none at review-prep ({why})",
                },
                store,
            )
    return lines


# --- notice_draft -----------------------------------------------------------------------------------------------------


def notice_hint(cfg: dict, n: int, kind: str) -> str:
    """What a notice line adds while notice_draft is bound: the command that writes the first draft ("" by default)."""
    return (
        f"; a first draft (leads only): `rk notice-draft {n} --kind {kind}`"
        if bound(cfg, "notice_draft")
        else ""
    )


def notice_facts(st: dict, store, n: int, kind: str) -> str:
    """The facts a notice of `kind` on PR n states: the overlap decisions that ask it to drop its copy, or its drift
    record (the failing tests on develop, the develop commits that may explain them)."""
    if kind == "overlap":
        from . import overlaps

        recs = {
            k: v
            for k, v in overlaps.decisions(store).items()
            if n in (v.get("members") or []) and v.get("owner") != n
        }
        if not recs:
            raise Miss(f"no overlap decision asks #{n} to drop its copy")
        return json.dumps(recs, indent=1, sort_keys=True)
    rec = ((st.get("prs") or {}).get(str(n)) or {}).get("drift") or {}
    if rec.get("status") != "fail":
        raise Miss(f"#{n} has no failing drift check")
    return json.dumps(rec, indent=1, sort_keys=True)


def notice(cfg: dict, st: dict, store, n: int, kind: str, chat=None) -> list[str]:
    """`rk notice-draft N --kind overlap|drift`: the first draft of the notice, in <tmp>/notice_draft_pr<N>_<kind>.md,
    and the line that hands it to the orchestrator; a line saying to write it without one when there is none.
    """
    out = pathlib.Path(paths.get(cfg, "tmp")) / f"notice_draft_pr{n}_{kind}.md"
    m = chosen(cfg, "notice_draft")
    if m["by"] == "none":
        return [
            f"NOTICE_DRAFT #{n}: {config.model_key('notice_draft')} is default: no first draft; write the {kind} "
            "notice yourself"
        ]
    try:
        facts = notice_facts(st, store, n, kind)
    except Miss as exc:
        rec = {
            "role": "notice_draft",
            "pr": n,
            "head": "",
            "by": m["by"],
            "model": model_name(m),
            "status": "missed",
            "why": str(exc),
        }
        record(cfg, rec, store)
        return [
            f"NOTICE_DRAFT #{n}: no first draft ({exc}): write the {kind} notice yourself"
        ]
    r = run(
        cfg,
        "notice_draft",
        [(f"the {kind} facts of PR #{n}", numbered(facts))],
        {"facts": facts.split("\n")},
        out,
        pr=n,
        extra=f'The facts\' source name is "facts". The notice is a {kind} notice on PR #{n}.',
        store=store,
        chat=chat,
    )
    if r["status"] == "ok":
        return [
            f"NOTICE_DRAFT #{n}: {out}: a first draft of the {kind} notice by {r['model']}, {r['leads']} checked "
            "lead(s): leads only. Check every claim against the facts, then write the notice you post"
        ]
    return [
        f"NOTICE_DRAFT #{n}: no first draft ({r['status']}: {r['why'][:160]}): write the {kind} notice yourself"
    ]
