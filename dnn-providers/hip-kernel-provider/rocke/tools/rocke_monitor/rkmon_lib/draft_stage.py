# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The draft stage: the review driver's contract with whichever drafter the configuration names, or none; and the
shadow stage, which drafts the same reviews with a binding under assessment.

The contract. For PR N at a head, a stage's job:
- takes the PR's staged data (<workers>/pr<N>/pr/, fetch_pr.py; our last review of the PR, posted or kept unposted,
  pr/ours.md, `rk stage-ours`), its trees wt<N> and wt<N>m (mkwt.py), the round and line 1 (REVIEW_ROUND, REVIEW_LINE1
  in the job's environment);
- writes in <workers>/pr<N>/: `staged` once those inputs are ready (the checks may start), review_agent.md (the draft,
  in the post format) and, when the stage keeps them, evidence.jsonl, runs.jsonl, draft_run.json, tokens_total.json,
  evidence_check.md and prior_settled.md (`rk evidence-check`, which the job runs before it ends, so before
  REVIEW_READY);
- ends done (exit 0: a draft), failed (no draft), lost, or past its time (the driver kills it).
Its failure modes: no staging (no trees: REVIEW_FAIL, as for any review); no draft (the stage retries `retries` times,
then `on_missing`: "skip" keeps the PR out of the picks until `rk review-unskip`, "ready" lets REVIEW_READY go ahead
with the draft marked missing).

The tool-loop program (worker/drafter.py) runs its draft in steps, each recorded in draft_run.json with its time,
requests and tokens: stage; settle (script: the prior items that need no model, prior_leads.md); prior (one short model
task per item left open, within drafter.prior_wall_min) and hunt (net-new findings, within drafter.hunt_wall_min), at
the same time; assemble (script); then the evidence check.

The stages (`for_config`: the one `choice` names, the rule of config draft_stage, a derived key: none with no first-pass
drafter, else the stage draft_stage names, where "auto" or a blank one is tool-loop when the drafter role has a binding,
else local-worker):
- local-worker: the local worker (review_agent_run.py and the local model): one at a time; no heavy job starts in its
  first PREFILL_MIN minutes, and its checks start once its trees exist and it is that old; no draft twice: skip.
- tool-loop: the drafter role's binding through worker/drafter.py (tools/draft_run.py): light locally (git and grep),
  so its checks start once it is staged and several run at once (the pool); no draft: one retry, then ready without it.
- none: no drafter: the job stages only, and REVIEW_READY has no draft.

The pool (the tool-loop stage): beside the review in flight, `pool_advance` starts drafts for the next reviews the
driver would pick, up to the ceiling (bindings.parallel_max), each once the scaler hook allows it (scaler.py). The
review driver adopts a pool draft of its PR at the same head (`adopt`); a pool draft whose PR closed, moved on, or is
covered by our review is stopped and dropped, and so is every pool draft once the live stage runs no pool.

The shadow stage (the shadow_drafter role's binding, config roles.shadow_drafter): for every live review, the review in
flight (not a merge-only head) and each review awaiting verification, `shadow_advance` drafts the same PR at the live
review's head with worker/drafter.py, in its own folder (<workers>/shadow_<binding>/pr<N>/: its own staged data, never
the live draft), once the live job's trees hold that head, through the same ceiling, scaler and limiter. The live
review never adopts a shadow draft and never waits for one: REVIEW_READY names the shadow draft when it is done
(`shadow_note`), and SHADOW_READY names one that is done later while its review awaits verification. A shadow draft that
ends after our review covered its head is still recorded, as late.

Each draft that ends (ready, missing, dropped), live or shadow, is one record in state `drafts`: its role, binding and
method (worker/drafter.py's METHOD; "single-pass" for the drafts before it), tokens, requests, waits, wall time, its
phases' times and tokens, its prior rows by settler, its status, and lateness: a live draft the review waited for after
its checks were done; a shadow draft that ended after our review covered its head.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import time
from dataclasses import dataclass

from . import bindings, config, jobs
from .snapshot import covers, same_commit
from .store import ROOT as MONITOR
from .store import now

PREFILL_MIN = 5
DRAFTS_KEPT = 500
STAGES = ("local-worker", "tool-loop", "none", "shadow")
OLD_METHOD = "single-pass"  # the tool-loop drafts recorded before draft_run.json named its method
# The statuses of a draft that came to verification: ready, or missing (its retry spent). STAGED adds "none", the
# status the record of a review staged with no drafter (the none stage) has.
VERIFIED = ("ready", "missing")
STAGED = (*VERIFIED, "none")


@dataclass(frozen=True)
class Stage:
    name: str
    retries: int  # retries after a job that left no draft
    on_missing: str  # "skip": out of the picks; "ready": REVIEW_READY without a draft; "record": recorded only
    parallel: bool  # several jobs at once (the pool)
    prefill_min: int  # no heavy job starts in this many minutes after the job starts
    timeout_min: int
    label: str  # how the start line names the job


LOCAL = Stage(
    "local-worker",
    retries=1,
    on_missing="skip",
    parallel=False,
    prefill_min=PREFILL_MIN,
    timeout_min=60,
    label="local review loop",
)
TOOL_LOOP = Stage(
    "tool-loop",
    retries=1,
    on_missing="ready",
    parallel=True,
    prefill_min=0,
    timeout_min=110,
    label="draft",
)
NONE = Stage(
    "none",
    retries=0,
    on_missing="ready",
    parallel=False,
    prefill_min=0,
    timeout_min=60,
    label="staging",
)
SHADOW = Stage(
    "shadow",
    retries=0,
    on_missing="record",
    parallel=True,
    prefill_min=0,
    timeout_min=110,
    label="shadow draft",
)


# The live stages, by name
LIVE = {stage.name: stage for stage in (LOCAL, TOOL_LOOP, NONE)}


def choice(cfg: dict) -> str:
    """The live draft stage's name, the one home of the rule: none with no first-pass drafter (bindings.no_draft: the
    default reviewer, the verifier alone), else config draft_stage ("auto" or blank: tool-loop with a drafter binding,
    else the local worker; a name that is no live stage's: the local worker)."""
    if bindings.no_draft(cfg):
        return NONE.name
    want = str(cfg.get("draft_stage") or "auto")
    if want == "auto":
        want = TOOL_LOOP.name if bindings.binding(cfg, "drafter") else LOCAL.name
    return want if want in LIVE else LOCAL.name


def for_config(cfg: dict) -> Stage:
    """The live draft stage, the one `choice` names."""
    return LIVE[choice(cfg)]


def of_pipe(pipe: dict) -> Stage:
    """The stage a review in flight runs (a pipe from before stages runs the local worker)."""
    return {"tool-loop": TOOL_LOOP, "none": NONE}.get(
        (pipe or {}).get("stage_name"), LOCAL
    )


def _workers(cfg: dict) -> pathlib.Path:
    root = (cfg.get("paths") or {}).get("root") or str(MONITOR.parent)
    return pathlib.Path((cfg.get("paths") or {}).get("workers") or f"{root}/workers")


def folder(cfg: dict, n: int) -> pathlib.Path:
    return _workers(cfg) / f"pr{n}"


def shadow_binding(cfg: dict) -> str:
    """The shadow drafter's binding name; "" when the role has none."""
    return bindings.tag(cfg, "shadow_drafter")


def shadow_folder(cfg: dict, n: int, binding: str | None = None) -> pathlib.Path:
    """A shadow draft's own folder: <workers>/shadow_<binding>/pr<N>."""
    return (
        _workers(cfg) / f"shadow_{binding or shadow_binding(cfg) or 'none'}" / f"pr{n}"
    )


def staged(cfg: dict, n: int) -> bool:
    return (folder(cfg, n) / "staged").is_file()


def staged_min(cfg: dict, n: int) -> float:
    """Minutes since PR n was staged (0 when it is not)."""
    try:
        return max(
            0.0, (time.time() - (folder(cfg, n) / "staged").stat().st_mtime) / 60.0
        )
    except OSError:
        return 0.0


def has_draft(cfg: dict, n: int, ws: pathlib.Path | None = None) -> bool:
    p = (ws or folder(cfg, n)) / "review_agent.md"
    return p.is_file() and p.stat().st_size > 0


def script(
    cfg: dict,
    stage: Stage,
    n: int,
    round_word: str,
    extra_sparse: list[str],
    retry: bool = False,
    base: str | None = None,
) -> str:
    """The job's bash for a tool-loop or none stage (the local worker's is review.worker_script). `base`: the PR's base
    branch, which mkwt.py merges (RKMON_BASE; develop when none)."""
    root = (cfg.get("paths") or {}).get("root") or str(MONITOR.parent)
    line1 = bindings.first_line(cfg)
    env = (
        f"REVIEW_ROUND={round_word} REVIEW_LINE1={shlex.quote(line1)} "
        f"SPARSE_EXTRA={shlex.quote(' '.join(extra_sparse))} RKMON_REVIEW_ROOT={shlex.quote(root)} "
        + ("" if base in (None, "", "develop") else f"RKMON_BASE={shlex.quote(base)} ")
    )
    mode = " --retry" if retry else " --stage-only" if stage.name == "none" else ""
    wipe = (
        ""
        if retry
        else f"chmod -R u+w {shlex.quote(str(folder(cfg, n)))} 2>/dev/null; "
        f"rm -rf {shlex.quote(str(folder(cfg, n)))}\n"
    )
    # The worker tools (fetch_pr.py, mkwt.py) take the monitor's settings from the job's RKMON_* environment.
    return (
        f"{config.export_line(cfg)}\n{wipe}{env}python3 -B {shlex.quote(str(MONITOR / 'tools' / 'draft_run.py'))} "
        f"{n}{mode}\n"
    )


def shadow_script(cfg: dict, n: int, head: str, round_word: str) -> str:
    """A shadow draft's bash: draft_run.py's shadow mode, with the shadow binding's line 1 (never posted)."""
    root = (cfg.get("paths") or {}).get("root") or str(MONITOR.parent)
    env = (
        f"REVIEW_ROUND={round_word} REVIEW_LINE1={shlex.quote(bindings.first_line(cfg, 'shadow_drafter'))} "
        f"RKMON_REVIEW_ROOT={shlex.quote(root)} RKMON_WORKERS={shlex.quote(str(_workers(cfg)))} "
    )
    return (
        f"{config.export_line(cfg)}\n{env}python3 -B {shlex.quote(str(MONITOR / 'tools' / 'draft_run.py'))} {n} "
        f"--shadow {shlex.quote(shadow_binding(cfg))} {shlex.quote(head)}\n"
    )


def prio(st: dict, n: int) -> bool:
    """A develop fix's or one of the user's priorities' draft: the limiter serves it first."""
    return str(n) in (st.get("develop_fixes") or {}) or str(n) in (
        st.get("user_priority") or {}
    )


def start_job(
    cfg: dict,
    st: dict,
    store,
    stage: Stage,
    n: int,
    round_word: str,
    extra_sparse: list[str],
    retry: bool = False,
) -> dict:
    base = (st.get("prs", {}).get(str(n)) or {}).get("base")
    return jobs.start(
        store,
        "draft",
        f"pr{n}",
        script(cfg, stage, n, round_word, extra_sparse, retry, base),
        timeout_min=stage.timeout_min,
        env={"RKMON_DRAFT_PRIO": "1" if prio(st, n) else "0"},
    )


# --- the pool ---------------------------------------------------------------------------------------------------------


def pool(st: dict) -> dict:
    """The pool's drafts by PR (state draft_pool); read without adding the key."""
    return st.get("draft_pool") or {}


def shadow_pool(st: dict) -> dict:
    """The shadow drafts running, by PR (state shadow_pool); read without adding the key."""
    return st.get("shadow_pool") or {}


def running(st: dict, store) -> int:
    """The draft jobs running now, live and shadow: the pool's, the shadows' and the parallel reviews' (each one's)."""
    k = sum(
        1
        for e in list(pool(st).values()) + list(shadow_pool(st).values())
        if jobs.status(store, e.get("job")) == "running"
    )
    for pipe in in_flight(st):
        if (
            of_pipe(pipe).parallel
            and jobs.status(store, pipe.get("worker")) == "running"
        ):
            k += 1
    return k


def in_flight(st: dict) -> list[dict]:
    """The parallel reviews running (state review_pipes), and a review_pipeline left by an older state."""
    return [
        p
        for p in (st.get("review_pipes") or []) + [st.get("review_pipeline") or {}]
        if p.get("pr") is not None
    ]


def adopt(st: dict, n: int, head: str) -> dict | None:
    """The review in flight takes PR n's pool draft at `head` (its job and when it started); None when there is none."""
    e = pool(st).get(str(n))
    if e and same_commit(e.get("head"), head):
        pool(st).pop(str(n))
        return e
    return None


def drop(cfg: dict, st: dict, store, n: int, why: str) -> str:
    """Stop PR n's pool draft and record it as dropped."""
    e = pool(st).pop(str(n), None)
    if not e:
        return ""
    if jobs.status(store, e.get("job")) == "running":
        jobs.kill(store, e["job"])
    record(cfg, st, store, n, e, "dropped", why=why)
    store.journal("review", f"#{n} pool draft dropped: {why}")
    return f"DRAFT_DROP #{n}: {why}"


def candidates(cfg: dict, st: dict, store, plan: list[dict]) -> list[dict]:
    """The reviews the driver would pick next, in its order, that have no draft yet: pick_next over the plan, without
    the parallel reviews, the drafts awaiting verification and the pool's PRs."""
    from . import review

    taken = {p.get("pr") for p in in_flight(st)} | {int(k) for k in pool(st)}
    out, items = [], [i for i in plan if i.get("pr") not in taken]
    for _ in range(len(items)):
        nxt = review.pick_next(cfg, st, store, items)
        if not nxt:
            break
        out.append(nxt)
        items = [i for i in items if i.get("pr") != nxt["pr"]]
    return out


def worth_drafting(cfg: dict, st: dict, nxt: dict) -> str:
    """Why a picked review gets no pool draft ("" when it does): a follow-up whose push our review covers, or that is
    merge-only (its checks decide first), is the driver's to start. A head is classified once (state draft_seen).
    """
    from . import review

    rec = st["prs"].get(str(nxt["pr"])) or {}
    if nxt["kind"] != "follow-up":
        return ""
    mc = rec.get("merge_check") or {}
    if mc.get("merge_only") and mc.get("new") == rec.get("head"):
        return "merge-only: its checks decide first"
    seen = st.setdefault("draft_seen", {})
    hit = seen.get(str(nxt["pr"])) or {}
    if hit.get("head") == rec.get("head"):
        return hit.get("why") or ""
    try:
        c = review.classify(cfg, rec, (rec.get("ours") or {}).get("head") or "")
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a git failure: no pool draft now; the driver classifies again
        return f"not classified ({str(exc)[:80]})"
    why = (
        f"{c['kind']}: the driver records it"
        if c["kind"] in review.COVERED or c["kind"] == "merge-only"
        else ""
    )
    seen[str(nxt["pr"])] = {"head": rec.get("head"), "why": why}
    return why


def pool_advance(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    """Drop the pool's stale drafts (every one when the live stage runs no pool), then start one draft for the next
    pick that has none, when the scaler allows it (stages that run several at once only). One a call, so the budget's
    use shows each start before the next."""
    stage = for_config(cfg)
    out = []
    for k, e in list(pool(st).items()):
        rec = st["prs"].get(k) or {}
        if not stage.parallel:
            out.append(
                drop(
                    cfg,
                    st,
                    store,
                    int(k),
                    f"the live stage is {stage.name}: no pool draft feeds it",
                )
            )
        elif rec.get("status") != "open":
            out.append(drop(cfg, st, store, int(k), "the PR is closed"))
        elif not same_commit(rec.get("head"), e.get("head")):
            out.append(drop(cfg, st, store, int(k), "its head moved"))
        elif covers(rec.get("ours"), e.get("head")):
            out.append(drop(cfg, st, store, int(k), "our review covers its head"))
        elif jobs.status(store, e.get("job")) == "timeout":
            jobs.kill(store, e["job"])
    if not stage.parallel:
        return [x for x in out if x]
    from . import review, scaler
    from .store import halted

    if review.tick_stale(st) or halted(st):  # a stop or a drain: no new draft
        return [x for x in out if x]
    for nxt in candidates(cfg, st, store, plan):
        why = worth_drafting(cfg, st, nxt)
        if why:
            continue
        ok, reason = scaler.may_start(
            cfg, st, store, plan, nxt["pr"], running(st, store)
        )
        if (
            ok
        ):  # a pool draft stages the PR's trees: with the RAM disk on, they need room there
            ok, why = scaler.room(cfg, st, store, "pipe", nxt["pr"])
            reason = why or reason
        if not ok:
            out.append(f"DRAFT_WAIT #{nxt['pr']}: {reason}")
            break
        rec = st["prs"][str(nxt["pr"])]
        files = review._changed_files(cfg, rec)
        extra = review.sparse_extra(cfg, files)
        job = start_job(
            cfg, st, store, stage, nxt["pr"], review.ROUNDS[nxt["kind"]], extra
        )
        st.setdefault("draft_pool", {})[str(nxt["pr"])] = {
            "pr": nxt["pr"],
            "kind": nxt["kind"],
            "head": rec["head"],
            "job": job,
            "started": now(),
            "prio": prio(st, nxt["pr"]),
        }
        store.journal(
            "review",
            f"#{nxt['pr']} pool draft started ({job['id']}) at {rec['head'][:10]}: {reason}",
        )
        out.append(
            f"DRAFT_START #{nxt['pr']} ({nxt['kind']}): pool draft at {rec['head'][:10]}"
        )
        break
    return [x for x in out if x]


# --- the shadow drafter -----------------------------------------------------------------------------------------------


def shadow_for(st: dict, n: int, head: str, binding: str) -> dict | None:
    """The last shadow draft recorded for PR n at `head` by `binding` (any status)."""
    for r in reversed(st.get("drafts") or []):
        if (
            r.get("pr") == n
            and r.get("role") == "shadow_drafter"
            and r.get("binding") == binding
            and same_commit(r.get("head"), head)
        ):
            return r
    return None


def live_reviews(st: dict) -> list[dict]:
    """The live reviews a shadow drafts: the parallel reviews (not a merge-only head, whose checks may close it), then
    the reviews awaiting verification, oldest first."""
    out = [
        {"pr": p["pr"], "kind": p["kind"], "head": p["head"]}
        for p in in_flight(st)
        if not p.get("merge_only")
    ]
    return out + [
        {"pr": r["pr"], "kind": r["kind"], "head": r["head"]}
        for r in st.get("review_ready") or []
    ]


def waiting_review(st: dict, n: int, head: str) -> str:
    """ "ready" when PR n's live review of `head` awaits verification, "flight" when it is in flight, else ""."""
    if any(
        r["pr"] == n and same_commit(r.get("head"), head)
        for r in st.get("review_ready") or []
    ):
        return "ready"
    return (
        "flight"
        if any(
            p.get("pr") == n and same_commit(p.get("head"), head) for p in in_flight(st)
        )
        else ""
    )


def shadow_collect(cfg: dict, st: dict, store) -> list[str]:
    """The shadow drafts that ended are recorded (SHADOW_READY for one whose review awaits verification), and the ones
    whose PR closed or moved on are stopped and recorded as dropped."""
    out = []
    sp = shadow_pool(st)
    for k, e in list(sp.items()):
        n, rec = int(k), st["prs"].get(k) or {}
        status = jobs.status(store, e.get("job"))
        if status == "running":
            why = (
                "the PR is closed"
                if rec.get("status") != "open"
                else (
                    "its head moved"
                    if not same_commit(rec.get("head"), e.get("head"))
                    else ""
                )
            )
            if why:
                jobs.kill(store, e["job"])
                record(cfg, st, store, n, e, "dropped", why=why, role="shadow_drafter")
                sp.pop(k)
                store.journal("review", f"#{n} shadow draft dropped: {why}")
                out.append(f"SHADOW_DROP #{n}: {why}")
            continue
        if status == "timeout":
            jobs.kill(store, e["job"])
        ws = shadow_folder(cfg, n, e.get("binding"))
        r = record(
            cfg,
            st,
            store,
            n,
            e,
            "ready" if has_draft(cfg, n, ws) else "missing",
            why=(
                ""
                if status in ("done", "timeout") or has_draft(cfg, n, ws)
                else f"the job {status}"
            ),
            role="shadow_drafter",
        )
        sp.pop(k)
        store.journal(
            "review",
            f"#{n} shadow draft {r['status']} ({status}) at {str(e.get('head'))[:10]}",
        )
        where = waiting_review(st, n, e.get("head"))
        if r["status"] == "ready" and where == "ready":
            out.append(
                f"SHADOW_READY #{n}: {ws}/review_agent.md ({r.get('evidence') or 'no evidence check'}); give it "
                f"to the review's verifier for the `{r['binding']} ...` lines once its post is final"
            )
        else:
            out.append(
                f"SHADOW_DONE #{n}: {r['status']}"
                + (" after our review (late)" if r.get("late") else "")
                + (" (the review in flight names it)" if where == "flight" else "")
            )
    return out


def shadow_start(cfg: dict, st: dict, store, plan: list[dict]) -> list[str]:
    """One shadow draft for the first live review that has none at its head, when the scaler allows it."""
    b = shadow_binding(cfg)
    if not b:
        return []
    from . import review, scaler
    from .store import halted

    if review.tick_stale(st) or halted(st):  # a stop or a drain: no new shadow draft
        return []
    sp = shadow_pool(st)
    for c in live_reviews(st):
        if str(c["pr"]) in sp or shadow_for(st, c["pr"], c["head"], b):
            continue
        ok, reason = scaler.may_start(
            cfg, st, store, plan, c["pr"], running(st, store), role="shadow_drafter"
        )
        if not ok:
            return [f"SHADOW_WAIT #{c['pr']}: {reason}"]
        job = jobs.start(
            store,
            "shadow",
            f"pr{c['pr']}",
            shadow_script(cfg, c["pr"], c["head"], review.ROUNDS[c["kind"]]),
            timeout_min=SHADOW.timeout_min,
            env={"RKMON_DRAFT_PRIO": "1" if prio(st, c["pr"]) else "0"},
        )
        st.setdefault("shadow_pool", {})[str(c["pr"])] = {
            "pr": c["pr"],
            "kind": c["kind"],
            "head": c["head"],
            "job": job,
            "started": now(),
            "binding": b,
            "prio": prio(st, c["pr"]),
        }
        store.journal(
            "review",
            f"#{c['pr']} shadow draft started ({job['id']}) at {c['head'][:10]}: {reason}",
        )
        return [f"SHADOW_START #{c['pr']} ({c['kind']}): {b} at {c['head'][:10]}"]
    return []


def shadow_note(cfg: dict, st: dict, n: int, head: str) -> str:
    """REVIEW_READY's words on the shadow draft of PR n at `head`, and on the leads its script settled."""
    b = shadow_binding(cfg)
    if not b:
        return ""
    ws = shadow_folder(cfg, n, b)
    leads = (
        f" Leads: {ws}/prior_leads.md (prior items a script settled, with evidence; re-derive every prior status "
        "yourself)."
        if (ws / "prior_leads.md").is_file()
        else ""
    )
    r = shadow_for(st, n, head, b)
    if r and r.get("status") == "ready":
        return (
            f" Shadow {b}: {ws}/review_agent.md (with evidence_check.md, prior_settled.md): once the post is "
            f"final, compare it and write the `{b} ...` lines.{leads}"
        )
    e = shadow_pool(st).get(str(n))
    if e and same_commit(e.get("head"), head):
        return (
            f" Shadow {b}: not done (running since {str(e.get('started'))[11:16]}Z): SHADOW_READY names it when "
            f"it is; still running at post time: `{b} none`.{leads}"
        )
    if r:
        return (
            f" Shadow {b}: {r['status']}"
            + (f" ({r['why']})" if r.get("why") else "")
            + f".{leads}"
        )
    return f" Shadow {b}: not started.{leads}"


# --- the records ------------------------------------------------------------------------------------------------------


def _ended(store, job: dict | None) -> float | None:
    """When a job ended (its rc file's time), None while it runs."""
    try:
        return (store.dir / "jobs" / job["id"] / "rc").stat().st_mtime
    except (OSError, KeyError, TypeError):
        return None


def _json(path: pathlib.Path) -> dict:
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def tokens_of(ws: pathlib.Path) -> dict:
    return _json(ws / "tokens_total.json").get("total") or {}


def findings_of(ws: pathlib.Path) -> dict:
    return _json(ws / "draft_findings.json")


def evidence_of(ws: pathlib.Path) -> dict:
    return _json(ws / "evidence_check.json")


def phases_of(ws: pathlib.Path) -> dict:
    """Each step's time, requests and tokens (draft_run.json's passes)."""
    out = {}
    for name, p in (_json(ws / "draft_run.json").get("passes") or {}).items():
        if isinstance(p, dict):
            out[name] = {
                k: p[k] for k in ("wall_s", "requests", "tokens", "status") if k in p
            }
    return out


def record(
    cfg: dict,
    st: dict,
    store,
    n: int,
    pipe: dict,
    status: str,
    checks: dict | None = None,
    why: str = "",
    role: str = "drafter",
) -> dict:
    """One draft that ended, into state `drafts` (the last DRAFTS_KEPT): ready, missing or dropped, with its role,
    binding and method, its tokens, requests, waits and wall time (every attempt), its phases, its findings' ids and
    lights, its prior rows by settler (prior_settled.json), its evidence check, and lateness: a live draft the review
    waited for over a minute after its checks were done; a shadow draft that ended after our review covered its head.
    """
    from . import prior_settled

    shadow = role == "shadow_drafter"
    binding = (
        (pipe.get("binding") if shadow else "")
        or bindings.binding_name(cfg, role)
        or (bindings.DEFAULT if not shadow and bindings.no_draft(cfg) else "")
    )
    ws = shadow_folder(cfg, n, binding) if shadow else folder(cfg, n)
    t = tokens_of(ws)
    f = findings_of(ws)
    ev = evidence_of(ws)
    head_rec = _json(ws / "draft_run.json")
    if shadow:
        waited = 0.0
        late = covers((st["prs"].get(str(n)) or {}).get("ours"), pipe.get("head"))
    else:
        d_end, c_end = _ended(store, pipe.get("worker") or pipe.get("job")), _ended(
            store, checks
        )
        waited = max(0.0, d_end - c_end) if d_end and c_end else 0.0
        late = waited > 60
    rec = {
        "pr": n,
        "kind": pipe.get("kind"),
        "head": pipe.get("head"),
        "at": now(),
        "status": status,
        "role": role,
        "binding": binding,
        "model": bindings.display(cfg, role) or head_rec.get("model") or "",
        "method": head_rec.get("method") or f.get("method") or OLD_METHOD,
        "attempts": 1 + int(bool(pipe.get("draft_retry"))),
        "tokens": int(t.get("tokens") or 0),
        "prompt_tokens": int(t.get("prompt_tokens") or 0),
        "cached_tokens": int(t.get("cached_tokens") or 0),
        "completion_tokens": int(t.get("completion_tokens") or 0),
        "reasoning_tokens": int(t.get("reasoning_tokens") or 0),
        "requests": int(t.get("requests") or 0),
        "waits_s": float(t.get("waits_s") or 0),
        "rate_limited": int(t.get("rate_limited") or 0),
        "wall_s": float(t.get("wall_s") or 0),
        "late": late,
        "late_s": round(waited),
        "evidence": ev.get("summary") or "",
        "evidence_result": ev.get("result") or "",
        "phases": phases_of(ws),
        "prior": prior_settled.counts(ws),
        "findings": [
            {"id": x["id"], "severity": x.get("severity")}
            for x in f.get("findings") or []
            if not x.get("dropped")
        ],
        "priors": len(f.get("priors") or []),
        **({"why": why} if why else {}),
    }
    st.setdefault("drafts", []).append(rec)
    st["drafts"] = st["drafts"][-DRAFTS_KEPT:]
    return rec


def draft_for(
    st: dict,
    n: int,
    head: str,
    binding: str | None = None,
    statuses: tuple = VERIFIED,
    role: str | None = None,
) -> dict | None:
    """The last draft recorded for PR n at `head` with one of `statuses` (by default VERIFIED: it came to
    verification; STAGED: or the review was staged with no drafter), by `binding` and in `role` (a record that names
    none is the drafter's) when they are named."""
    for r in reversed(st.get("drafts") or []):
        if (
            r.get("pr") == n
            and r.get("status") in statuses
            and same_commit(r.get("head"), head)
            and (binding is None or r.get("binding") == binding)
            and (role is None or (r.get("role") or "drafter") == role)
        ):
            return r
    return None


def ready_line(cfg: dict, n: int, kind: str, cs: str, why: str, draft: dict) -> str:
    """REVIEW_READY for a review whose stage is tool-loop or none."""
    ws = folder(cfg, n)
    tag = bindings.tag(cfg, "drafter") or "draft"
    if draft["status"] == "ready":
        what = (
            f"{draft['model'] or 'the drafter'} draft ready ({draft['tokens']} tokens, {draft['requests']} "
            f"requests, {draft['wall_s'] / 60:.0f} min; evidence check: {draft['evidence'] or 'not run'})"
        )
        inputs = (
            f"Read first: {ws}/evidence_check.md, then {ws}/prior_settled.md. Inputs: {ws}/review_agent.md (the "
            f"draft: its N1, N2, … are G1, G2, … in the `{tag} ...` assessment lines), {ws}/checks/checks.md, "
            f"{ws}/pr/meta.md, {ws}/agent_log.txt"
        )
    else:
        what = (
            "no draft (the drafter failed twice): verify without it"
            if draft["status"] == "missing"
            else "no drafter: verify without a draft"
        )
        inputs = f"Inputs: {ws}/checks/checks.md, {ws}/pr/meta.md"
    if (
        draft["status"] == "none"
    ):  # no drafter: no draft to compare, no assessment lines, no first pass to credit
        return (
            f"REVIEW_READY #{n} ({kind}): {what}, checks {cs}.{why} {inputs}. Verify alone (line 1 and the "
            f"footer name the verifier, {bindings.verifier_name(cfg)}; no first pass), then `rk post {n} FILE "
            "--premises SPEC`"
        )
    return (
        f"REVIEW_READY #{n} ({kind}): {what}, checks {cs}.{why} {inputs}. Verify blind first, then compare with "
        f"the draft (notes.md `{tag} ...` lines), then `rk post {n} FILE --premises SPEC`"
    )


# --- staging our last review ------------------------------------------------------------------------------------------


def stage_ours(cfg: dict, st: dict, n: int, out_dir: pathlib.Path) -> str:
    """pr/ours.md: our last review on PR n, for a draft's prior items: the head it covers, when it was posted (`-
    Posted:`) or kept unposted while posting is off (`- Kept unposted:`, under the title "Our last review"), and its
    body (under `## Review`). The body is read where post.our_last_review reads it. One line saying what was staged.
    """
    from . import gh
    from . import post as post_mod

    ours = (st["prs"].get(str(n)) or {}).get("ours") or {}
    if not ours.get("head"):
        return f"no review of ours on #{n}: nothing staged"
    kept = post_mod.kept_unposted(ours)
    what = "our last review" if kept else "our last post"
    try:
        body, src = post_mod.our_last_review(cfg, n, ours)
    except gh.GhError:
        body, src = "", ""
    if not body.strip():
        return f"#{n}: {what}'s body was not read: nothing staged"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title, when = ("review", "Kept unposted") if kept else ("review post", "Posted")
    text = (
        f"# Our last {title} on PR #{n}\n- Head: {ours['head']}\n- {when}: {ours.get('at') or ''}\n\n"
        f"## Review\n{body.rstrip()}\n"
    )
    (out_dir / "ours.md").write_text(text, encoding="utf-8")
    at = f"at {ours['head'][:10]}, {ours.get('at') or '?'}"
    return f"#{n}: {what} ({'kept unposted ' if kept else ''}{at}) staged from {src}"
