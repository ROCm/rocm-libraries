# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk roles`: every role's model and effort in one table (role_rows, table_lines: rk doctor prints it too), and the
two optional on-prem model roles beside the reviewer: the planner, which
makes the merge plan when its proposal passes every hard constraint (rkmon_lib/planner.py), and the analyst, which
ranks and annotates the insights and never cites a number the data does not hold (rkmon_lib/analyst.py).

- Off by default. `models.analytics` is the analytics model both roles use: "" (off), "worker" (the reviewer's own
  model), or {"model": <a model the reviewer's endpoint lists>, "share": <0.01-1>, "timeout_min": N, ...}. A role
  set on its own (`models.planner`, `models.analyst`: the same forms) uses its own model instead, "off" turns it off,
  and "" follows the analytics model (one analytics model by default, the analyst selectable apart). Model names live
  in the deployment's configuration only.
- The same endpoint as the reviewer (its address, key, headers and limits: config.worker), the role's own model, and the
  reviewer's shared rate budget, within the role's share of its limits (reviewer_client.Budget: a role over its share
  waits outside the queue, so it never holds the parallel reviews back).
- Never in the tick's way: each call is a background job (`rk role-run ROLE`, rkmon_lib/jobs.py) whose answer the
  next tick reads (`poll`). A role that is off, down, slow (past `timeout_min`), or answers something unusable changes
  nothing: the toolkit's own result stands, and the board and the journal say why. A role never writes outside its
  own folder (<RUN_DIR>/roles/<role>), and never acts: no arming, no merging, no posting.

The deploy settings' passes, by their keys (PASSES): `pass_rows` and `pass_lines`, each pass's model, provider and
effort as rk roles shows them; `pass_bindings(cfg)`, the names of the model bindings the passes run on, read by name
alone, which the secrets registry reads (a binding's key is needed while a pass runs on its binding).
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex

from . import config, jobs, paths, whole_file
from .store import now

ROLES = ("planner", "analyst")
SHARED = "analytics"  # models.analytics: the model both roles use unless one is set on its own
# A role's limits where its map leaves one out: its share of the reviewer endpoint's rate limits, and the minutes an
# answer may take. A constant, not a setting: no deployment runs these roles.
DEFAULTS = {"share": 0.1, "timeout_min": 10}
WHAT = {
    "planner": "proposes the merge plan's waves (the plan when it keeps every rule of the toolkit's own)",
    "analyst": "ranks and annotates the per-PR insights (citing only numbers the statistics hold)",
}
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def own(cfg: dict, role: str) -> bool:
    """The role has a model of its own, not the analytics model (or is "off")."""
    return (cfg.get("models") or {}).get(role, "") not in ("", None)


def wanted(cfg: dict, role: str) -> dict | None:
    """The role's settings when it is on: {"model", "share", "timeout_min", ...}, its own or the analytics model's,
    each limit it leaves out DEFAULTS'; None when it is off."""
    models = cfg.get("models") or {}
    v = (
        models.get(role, "")
        if role == SHARED or own(cfg, role)
        else models.get(SHARED, "")
    )
    if v == "worker":
        v = {"model": config.worker(cfg)["model"]}
    if not isinstance(v, dict) or not str(v.get("model") or "").strip():
        return None
    out = dict(DEFAULTS, **{k: x for k, x in v.items() if x not in (None, "")})
    out["share"] = min(1.0, max(0.01, float(out["share"])))
    out["timeout_min"] = max(1, int(out["timeout_min"]))
    return out


def on(cfg: dict, role: str) -> bool:
    return wanted(cfg, role) is not None and bool(config.worker(cfg).get("url"))


def endpoint(cfg: dict, role: str) -> dict | None:
    """The role's model settings: the reviewer's endpoint with the role's model, on the reviewer's shared rate budget
    within the role's share. None when the role is off or the reviewer has no endpoint.
    """
    want = wanted(cfg, role)
    w = config.worker(cfg)
    if not want or not w.get("url"):
        return None
    out = dict(
        w,
        model=str(want["model"]).strip(),
        model_name="",
        budget_label=f"{w['url']} {w['model']}",
        budget_role=role,
        budget_share=want["share"],
    )
    for key in ("context", "max_output_tokens"):
        if want.get(key):
            out[key] = int(want[key])
    return out


def describe(cfg: dict, role: str) -> str:
    """The setup summary's, rk roles' and rk doctor's line: off, or the model (the analytics model's, when the role
    follows it) and its share (never a key)."""
    want = wanted(cfg, role)
    if not want:
        return "off" + {
            "planner": " (the toolkit's own plan)",
            "analyst": " (no annotations)",
        }.get(role, "")
    shared = " (the analytics model)" if role != SHARED and not own(cfg, role) else ""
    return f"{want['model']}{shared} at the reviewer's endpoint, {round(want['share'] * 100)}% of its rate limits"


def json_answer(text: str) -> dict:
    """The JSON object in a model's answer: the whole text, a fenced block, or the outermost {...}. ValueError when
    there is none."""
    t = (text or "").strip()
    for cand in [t, *FENCE.findall(t)]:
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            v = json.loads(t[a : b + 1])
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    raise ValueError("the answer has no JSON object")


# --- the jobs ---------------------------------------------------------------------------------------------------------


def folder(cfg: dict, role: str) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "run")) / "roles" / role


def state(st: dict, role: str) -> dict:
    return st.setdefault("roles", {}).setdefault(role, {})


def running(st: dict, store, role: str) -> bool:
    job = state(st, role).get("job")
    return bool(job) and jobs.status(store, job) == "running"


def start(
    cfg: dict,
    st: dict,
    store,
    role: str,
    fp: str,
    messages: list[dict],
    extra: dict | None = None,
) -> str:
    """Start the role's job on these messages (one at a time per role); the line to show, "" when one is running or
    nothing new may start (a stop or a drain: store.halted)."""
    from .store import halted

    r = state(st, role)
    if running(st, store, role) or halted(st):
        return ""
    want = wanted(cfg, role)
    d = folder(cfg, role)
    d.mkdir(parents=True, exist_ok=True)
    src, out = d / f"in-{fp}.json", d / f"out-{fp}.json"
    src.write_text(
        json.dumps({"role": role, "fp": fp, "messages": messages, **(extra or {})}),
        encoding="utf-8",
    )
    out.unlink(missing_ok=True)
    rk = shlex.quote(str(paths.PACKAGE / "rk"))
    home = shlex.quote(str(config.home()))
    script = (
        f"{rk} --home {home} role-run {role} --input {shlex.quote(str(src))} "
        f"--output {shlex.quote(str(out))}"
    )
    r["job"] = jobs.start(store, "role", role, script, want["timeout_min"])
    r["fp"], r["started"] = fp, now()
    for old in sorted(d.glob("in-*.json"))[:-6] + sorted(d.glob("out-*.json"))[:-6]:
        old.unlink(missing_ok=True)  # the last few only
    return f"ROLE_STARTED {role}: {want['model']} on the facts {fp}"


def poll(cfg: dict, st: dict, store) -> list[str]:
    """Read every finished role job's answer into st["roles"][role]["last"]: {"fp", "at", "ok", "answer" | "error",
    "model"}. A job past its time is stopped and recorded as failed. The lines to show (failures only, once).
    """
    lines = []
    for role in ROLES:
        r = state(st, role)
        job = r.get("job")
        if not job:
            continue
        status = jobs.status(store, job)
        if status == "running":
            continue
        out = folder(cfg, role) / f"out-{r.get('fp')}.json"
        if status == "timeout":
            jobs.kill(store, job)
            rec = {
                "fp": r.get("fp"),
                "at": now(),
                "ok": False,
                "error": f"no answer within {job['timeout_min']} min",
            }
        else:
            try:
                rec = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                rec = {
                    "fp": r.get("fp"),
                    "at": now(),
                    "ok": False,
                    "error": f"the job ended ({status}) without an answer: {jobs.tail(store, job, 3)[-200:]}",
                }
        r.pop("job", None)
        r["last"] = rec
        if not rec.get("ok"):
            store.journal(
                "role",
                f"{role}: no usable answer ({str(rec.get('error'))[:200]}); the toolkit's own "
                "result stands",
            )
            lines.append(
                f"(the {role} failed: {str(rec.get('error'))[:160]}; the toolkit's own result stands)"
            )
    return lines


def run_job(cfg: dict, role: str, src: str, out: str) -> int:
    """`rk role-run ROLE --input FILE --output FILE`: one chat on the role's endpoint and model, the answer's JSON
    written to FILE. Exit 0 whenever FILE was written (a failure is recorded in it)."""
    from . import models

    data = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
    rec = {"fp": data.get("fp"), "at": now(), "ok": False}
    s = endpoint(cfg, role)
    try:
        if not s:
            raise RuntimeError(
                f"the {role} role is off, or the reviewer has no endpoint"
            )
        llm = models.client()
        budget = llm.Budget(s, paths.get(cfg, "run"), who=role)
        msg = llm.chat(s, data["messages"], budget=budget)
        rec.update(
            ok=True, answer=json_answer(msg.get("content") or ""), model=s["model"]
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - any failure is the role's answer: recorded, never raised into a tick
        rec.update(
            error=" ".join(str(exc).split())[:300] or type(exc).__name__,
            model=(s or {}).get("model", ""),
        )
    whole_file.write(out, json.dumps(rec, ensure_ascii=False), ANSWER_MODE)
    print(f"{role}: {'answered' if rec['ok'] else 'failed: ' + rec['error']}")
    return 0


NO_EFFORT = ("-", "not a Claude model: it takes none")
NO_MODEL = ("-", "no model")
# The deploy settings' passes, by their keys (features.review_automation.passes, and the orchestrator,
# features.review_automation.orchestrator), each with its configuration role (its effort's key) and, for an optional
# pass, its owner pass: `default` is the owner doing it, with the owner's model and effort.
FIRST_REVIEW_PASS = "first_review_pass"
PASSES = (
    ("orchestrator", "orchestrator", ""),
    ("verification_pass", "verify", ""),
    ("code_quality_pass", "quality", ""),
    (FIRST_REVIEW_PASS, "first_pass", "verification_pass"),
    ("ci_triage_pass", "ci_triage", "verification_pass"),
    ("first_code_quality_pass", "quality_draft", "code_quality_pass"),
    ("notice_draft_pass", "notice_draft", "orchestrator"),
)
ROLE = {name: role for name, role, _owner in PASSES}  # a pass's configuration role
PASS = {role: name for name, role, _owner in PASSES}  # a configuration role's pass
# A role's answer: the monitor's own (its job writes it, the tick reads it).
ANSWER_MODE = 0o600


def _claude_model(cfg: dict, name: str) -> str:
    """The model, as the deploy settings name it, of the pass NAME, which always runs on Claude."""
    return {
        "orchestrator": config.orchestrator_model(cfg),
        "verification_pass": config.claude_model(cfg, "audit"),
        "code_quality_pass": config.claude_model(cfg, "deep"),
    }[name]


def _claude_pass(cfg: dict, name: str) -> tuple[str, str]:
    """(the model, as the deploy settings name it, of the pass NAME, which always runs on Claude, its effort)."""
    return (
        _claude_model(cfg, name),
        config.role_effort(cfg, ROLE[name])[0] or config.MODEL_DEFAULT,
    )


def _bound(cfg: dict, name: str) -> dict:
    """The optional pass NAME's model: {"by": "none"} for default, {"by": "worker", ...} or {"by": "binding", ...}."""
    from . import bindings

    if name == FIRST_REVIEW_PASS:
        r = bindings.reviewer(cfg)
        if r["kind"] == bindings.DEFAULT:
            return {"by": "none"}
        if r["kind"] == bindings.WORKER:
            return dict(config.worker(cfg), by="worker")
        b = bindings.binding(cfg, "drafter")
        return dict(b, by="binding") if b else {"by": "missing", "binding": r["choice"]}
    return config.model_for(cfg, ROLE[name])


def pass_bindings(cfg: dict) -> set:
    """The names of the model bindings the passes run on (PASSES), read by name alone (no binding is resolved, so the
    secrets registry can read it): bindings.WORKER for the local worker (config `worker`), else a model_bindings name.
    A Claude pass's model is its model's name; the first review pass's, bindings.reviewer_choice; an optional pass's,
    models.<ROLE> (a map is the local worker). A name no binding has, the default among them, names none.
    """
    from . import bindings

    def model(name: str, role: str, owner: str) -> str:
        if not owner:
            return _claude_model(cfg, name)
        if name == FIRST_REVIEW_PASS:
            return bindings.reviewer_choice(cfg)
        want = config.role_setting(cfg, role)
        return bindings.WORKER if isinstance(want, dict) else str(want).strip()

    defined = cfg.get("model_bindings")
    defined = (
        {b for b, raw in (defined or {}).items() if isinstance(raw, dict)}
        if isinstance(defined, dict)
        else set()
    )
    if isinstance(cfg.get("worker"), dict):
        defined.add(bindings.WORKER)
    return {model(*p) for p in PASSES} & defined


def pass_rows(cfg: dict) -> list[tuple[str, str, str, str]]:
    """Every pass of the deploy settings, resolved: (pass, model, provider, effort). An optional pass on `default`
    shows its owner's model and effort ("default → opus", "high (from verification_pass)"), and follows a change to the
    owner. The first review pass on a model shows the effort its drafter runs at (bindings.drafter_effort); another
    pass on a model, the effort its requests carry (config.bound_effort): on a model that is not Claude, the local
    worker included, the pass's own effort, or "-" on `default`, which sends none."""
    from . import bindings

    rows = []
    for name, role, owner in PASSES:
        if not owner:
            model, effort = _claude_pass(cfg, name)
            rows.append((name, model, "claude", effort))
            continue
        m = _bound(cfg, name)
        if m["by"] == "none":
            model, effort = _claude_pass(cfg, owner)
            rows.append(
                (name, f"default → {model}", "claude", f"{effort} (from {owner})")
            )
        elif m["by"] == "missing":
            rows.append((name, f"{m['binding']} (no such model)", "-", "-"))
        elif m["by"] == "worker":
            provider = "ollama" if m.get("backend") == "ollama" else "openai-compatible"
            e = (
                bindings.drafter_effort(cfg, None)
                if name == FIRST_REVIEW_PASS
                else config.bound_effort(cfg, role, m)
            )
            rows.append(
                (
                    name,
                    f"worker → {m.get('model') or '(no model set)'}",
                    provider,
                    e or "-",
                )
            )
        else:
            kind = m.get("kind") or (
                "claude" if m.get("provider") == "anthropic" else "openai-compatible"
            )
            e = (
                m["effort"]  # the drafter's binding carries the effort it runs at
                if name == FIRST_REVIEW_PASS
                else config.bound_effort(cfg, role, m)
            )
            own = config.own_effort(cfg, role) if e else ""
            rows.append(
                (
                    name,
                    f"{m.get('binding')} → {m.get('model') or '(no model)'}",
                    kind,
                    (e + ("" if own else f" (from {owner})")) if e else "-",
                )
            )
    return rows


def pass_lines(cfg: dict) -> list[str]:
    """pass_rows as an aligned table: pass | model | provider | effort."""
    rows = [("pass", "model", "provider", "effort")] + pass_rows(cfg)
    w = [max(len(str(r[i])) for r in rows) for i in range(3)]
    return [
        f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]}".rstrip() for r in rows
    ]


def role_rows(cfg: dict) -> list[tuple[str, str, str, str]]:
    """Every role's (role, model, effort, where the effort comes from), in one place: the Claude roles at their
    resolved effort (config.role_effort), the bound roles at the effort their requests carry (config.bound_effort),
    and the draft on the local review loop at the effort its drafter sends (bindings.drafter_effort).
    The analytics model, the planner and the analyst have a row only while they are on: no deploy setting turns them
    on.
    """
    from . import bindings, deploy_tree, first_pass

    def claude(role: str, model: str) -> tuple:
        return (role, model, *config.role_effort(cfg, role))

    def bound(role: str, m: dict, none: str) -> tuple:
        if m["by"] == "none":
            return (role, none, *NO_MODEL)
        name = (
            f"{m.get('binding')} ({first_pass.model_name(m)}, "
            f"{bindings.OLD_PROVIDERS.get(m.get('provider'), m.get('provider'))})"
            if m["by"] == "binding"
            else f"worker ({m.get('model') or 'model_bindings.worker.model_id not set'})"
        )
        e = config.bound_effort(cfg, role, m)
        return (role, name) + (
            (e, config.role_effort(cfg, role)[1]) if e else NO_EFFORT
        )

    def binding_row(role: str, b: dict | None, key: str, model: str = "") -> tuple:
        if not b:
            if role == "draft" and model.startswith(f"{bindings.WORKER} "):
                # the local review loop: the effort its drafter sends (worker.think, which the first review pass's
                # effort sets), and none on the pass's default
                e = bindings.drafter_effort(cfg, None)
                where = deploy_tree.name("effort.first_pass")
                return (role, model, *((e, where) if e else NO_EFFORT))
            return (role, model or f"none ({key})", *NO_MODEL)
        return (
            role,
            model or f"{b['binding']} ({b['name']})",
            b["effort"],
            f"model_bindings.{b['binding']}.effort",
        )

    orch = config.orchestrator_model(cfg)
    r = bindings.reviewer(cfg)
    rows = [
        claude("orchestrator", orch),
        claude("verify", config.claude_model(cfg, "audit") + " (the verifier agent)"),
        claude(
            "quality", config.claude_model(cfg, "deep") + " (the code quality agent)"
        ),
        (
            "post",
            f"{orch} (the session posts)",
            *config.role_effort(cfg, "orchestrator"),
        ),
        claude("audit", config.claude_model(cfg, "audit")),
        claude("deep", config.claude_model(cfg, "deep")),
        binding_row(
            "draft",
            bindings.binding(cfg, "drafter"),
            "reviewer",
            f"{r['choice']} ({r['model'] or 'no model'})",
        ),
        binding_row(
            "shadow_drafter",
            bindings.binding(cfg, "shadow_drafter"),
            "roles.shadow_drafter",
        ),
        binding_row("quality_lane", bindings.binding(cfg, "quality"), "roles.quality"),
        bound("tick", config.model_for(cfg, "tick"), "the rule filter (no model)"),
    ]
    for role in config.FIRST_PASS_ROLES:
        rows.append(
            bound(
                role,
                config.model_for(cfg, role),
                f"default (no first pass: {first_pass.OWNER[role]} alone)",
            )
        )
    for role in (SHARED, *ROLES):
        if wanted(cfg, role):
            rows.append((role, describe(cfg, role), *NO_EFFORT))
    return rows


def table_lines(cfg: dict, store=None) -> list[str]:
    """`rk roles` (and rk doctor): the passes, resolved (pass_lines); then every other role's model and effort, and
    where the effort comes from; then the last run of the CI triage, first code quality and notice draft passes
    (config.FIRST_PASS_ROLES, state/first_pass.jsonl), each by its key.
    """
    from . import first_pass

    passes = set(PASS) | {"draft", "post"}
    rows = [("role", "model", "effort", "effort from")] + [
        r for r in role_rows(cfg) if r[0] not in passes
    ]
    w = [max(len(str(r[i])) for r in rows) for i in range(3)]
    out = (
        pass_lines(cfg)
        + ["", "other roles:"]
        + [
            f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]}".rstrip()
            for r in rows
        ]
    )
    last = {}
    try:
        p = config.toolkit_state_dir(cfg) / first_pass.LOG
        for line in p.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            last[rec.get("role")] = rec
    except (OSError, ValueError, paths.PathsError):
        pass
    for role in config.FIRST_PASS_ROLES:
        rec = last.get(role)
        if rec:
            out.append(
                f"{PASS[role]}: last run {rec.get('ts')}"
                + (f" on #{rec['pr']}" if rec.get("pr") else "")
                + f": {rec.get('status')}"
                + (
                    f", {rec.get('leads', 0)} lead(s)"
                    if rec.get("status") == "ok"
                    else f" ({str(rec.get('why', ''))[:120]})"
                )
            )
    return out


def status_lines(cfg: dict, st: dict, store) -> list[str]:
    """`rk roles`: each role that is on (no deploy setting turns one on), its model and share, its job, and its last
    answer."""
    out = []
    for role in ROLES:
        if not wanted(cfg, role):
            continue
        r = state(st, role)
        line = f"{role}: {describe(cfg, role)}"
        if r.get("job"):
            line += f"; a job runs since {r.get('started')} ({jobs.status(store, r['job'])})"
        last = r.get("last")
        if last:
            line += f"; last answer {last.get('at')}: " + (
                "ok" if last.get("ok") else f"failed: {last.get('error')}"
            )
        use = r.get("use")
        if use:
            line += f"; {use}"
        out.append(line)
    return out
