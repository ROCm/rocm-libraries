# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The reviewer (the first pass), roles and their model bindings: which model each role of the review pipeline uses, and
how to reach it.

Config `reviewer` is the one setting that chooses the first-pass reviewer (the drafter), and swaps it:
  "default"     the verifier (Claude) reviews alone, with no first-pass drafter (the package's default)
  "worker"      the local worker: the reviewer model in config `worker` (a local Ollama, or an OpenAI-compatible
                endpoint through the same settings), drafting through worker/review_agent.py
  <a name>      a binding in `model_bindings` (provider "openai": any OpenAI-compatible endpoint, gateway or local;
                provider "anthropic": Claude through the Anthropic API or the gateway Claude Code uses), drafting through
                worker/drafter.py, the first review pass on a model binding
Blank, it follows the older keys, still read: `roles.drafter`, else `worker_backend`; with neither, "worker" (a
configuration from before the setting: its local worker drafted). A new configuration holds "default" (the package's
default, the deploy settings' schema's). The older spelling "none" reads as "default". `migrate` rewrites a
configuration onto `reviewer`, and drops the older keys. Line 1 of every post and its footer
follow it (`first_line`): the default names the verifier; a drafter names its model, at the effort it runs at
(`drafter_effort`). Line 1 gives this deployment's
automated reviewer name too (rkmon_lib/reviewer_name.py, the one home of its form).

Config `model_bindings` names the bindings (the deploy settings' model_bindings, rkmon_lib/deploy_tree.py; but the one
named `worker`, which is config `worker`). Each is one model at one endpoint:
  provider      what it is (PROVIDERS): "claude", a Claude model through the Anthropic Messages API, reached as the
                Claude session reaches Claude (integrations.claude_code's address, `claude_url`, and the claude login's
                key, sent as the session sends it); "ollama",
                a model on an Ollama server (its OpenAI-compatible /v1 route); "openai-compatible", an OpenAI-compatible
                chat completions route (an API gateway, vLLM, LiteLLM). The adapter that speaks to it
                (worker/llm_client.py PROVIDERS) is "anthropic" for claude and "openai" for the others; the older
                provider names openai and anthropic are read as openai-compatible and claude. The built-in models opus,
                sonnet and haiku need no entry: a first pass that names one gets a claude binding of that name, whose
                model is the latest Claude model's id (deploy_tree.builtin), which the gateway takes
  where         "gateway" or "local" (default: local for a localhost URL, else gateway)
  url, chat_path   the base address and the route under it (/chat/completions; /v1/messages for anthropic). The url
                may be a `$NAME` reference, read from the environment where the request is made
  api_key_secret  the name of a secret that holds the key (rkmon secret NAME FILE installs it), read at each request
                and never logged, from the file where the secrets registry keeps that secret: <SECRETS_DIR>/NAME or a
                fixed secret's own place (secrets_registry.binding_key_file), which the binding takes as its
                api_key_file; an api_key_file the configuration gives is not kept
                A key comes from a secret's file alone, never from a variable. A claude binding that names no address
                of its own (or the Claude session's $ANTHROPIC_BASE_URL) or no key of its own is reached as the session
                is (`binding_by_name`), with the claude login's key in the headers the session sends it in.
  auth_header, auth_scheme, auth_mode   where the key goes: in auth_header after auth_scheme (Authorization: Bearer by
                default; x-api-key with no scheme for anthropic; a blank scheme sends the key alone); auth_mode "both"
                sends it as a bearer token too
  model, name   the id requests send, and the name line 1 and the dashboard show (default: the id)
  tpm, rpm      the endpoint's tokens and requests per minute (0: none known; its HTTP 429s still slow the callers)
  concurrency   the requests in flight at once, every caller of the binding's bucket together (0: no cap); past it,
                requests wait in the shared limiter, in rank order, instead of in the endpoint's own queue
  context, max_output_tokens   the model's limits, in tokens
  effort        the reasoning effort of the role's main work: low, medium or high (the drafter's is the first review
                pass's: `drafter_effort`)
  and the client's own: temperature (openai only), stall_s, stream, stream_usage, max_resends, resend_wait_s.

Config `roles` binds each role to a binding by name: shadow_drafter, quality, analytics, triage (and drafter, the older
name of `reviewer`).
- drafter: the live review's first pass, `reviewer` (`reviewer_choice`); the reviewer model's id (config
  worker.model) is its model id when the binding names none. With "worker" the local worker drafts (draft_stage.py).
- shadow_drafter: a binding under assessment. It drafts every live review's PR at the live review's head, in its own
  folder, beside the live draft and never in its place (draft_stage.py, the shadow stage).
- quality: the code quality pass on a binding: the review driver runs it beside the draft and the checks
  (worker/quality_lane.py, review.quality_start); with none, the orchestrator's code quality reviewer runs it.
Config `drafter` is worker/drafter.py's settings, each entry over the package's (config.defaults.json), for both
drafting roles: the run's token budget and wall-time cap; each phase's own (the prior items' tasks: prior_wall_min,
prior_token_budget; the hunt: hunt_wall_min, hunt_token_budget), so the hunt keeps its own whatever the prior tasks
spend; the prior tasks' steps, how many run at once, and their effort; the effort of the mechanical turns; the task
cards' size; the tool results kept; the hunt's rounds; the saved probes' time cap; the limiter's folder and reserve; the
restage age; the evidence check's re-run caps; the code quality pass's run (quality_wall_min, quality_token_budget,
quality_steps, quality_max_tokens); and the first passes' caps (first_pass_wall_min, first_pass_token_budget,
first_pass_max_tokens: rkmon_lib/first_pass.py, the roles models.ci_triage, models.quality_draft and
models.notice_draft, whose binding is named in `models`, not here). `draft_parallel_max` (`glm_parallel_max` is read
too) is the most drafts that run at once, live and shadow together.

The assessment scores each drafter against the verified post under a tag (`candidates`): a binding's name, or for the
local worker `worker.tag` (`worker_tag`: the leading letters of the reviewer model's id, config worker.model).

With no first-pass drafter (`no_draft`): `reviewer` "default" (an older key's "none" too), or `draft_stage` "none". The
review driver stages the PR and runs its checks with no drafter, REVIEW_READY says "no drafter", and the verifier
reviews alone: line 1 names the verifier (`review_first_line_no_draft`, `no_draft_line`: "Automated review by
<automated_reviewer_name>: Model - <verifier_name>, effort: <the verifier's effort>"; `verifier_name`: the audit model
in words), `review_first_line` (the drafter's line) waits unused, the post's footer credits no first pass, and the
assessment records no live drafter. A shadow drafter, when `roles.shadow_drafter` names one, still drafts beside it.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re

from . import config, registry, reviewer_name, secrets_registry

ROLES = ("drafter", "shadow_drafter", "quality", "analytics", "triage")
PROVIDERS = ("claude", "ollama", "openai-compatible")
# The adapter worker/llm_client.py speaks to each provider with; the older provider names it read are its own.
ADAPTER = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "ollama": "openai",
    "openai-compatible": "openai",
    "openai": "openai",
}
OLD_PROVIDERS = {"openai": "openai-compatible", "anthropic": "claude"}
# The binding settings whose default is the deploy schema's, each meaning the same in the configuration: the limits and
# the stall timeout (_schema_defaults).
FROM_SCHEMA = ("tpm", "rpm", "concurrency", "stall_s")


def _schema_defaults() -> dict:
    """The deploy schema's default of each FROM_SCHEMA setting, by its configuration key: the first form's that has
    the key (deploy_tree.node), an OpenAI-compatible binding's, the default provider's.
    """
    from . import deploy_tree

    out = {}
    for key, internal in deploy_tree.entries("model_bindings").items():
        if internal in FROM_SCHEMA:
            _path, node = deploy_tree.node(f"model_bindings.*.{key}")
            out.setdefault(internal, node["default"])
    return out


@functools.cache
def binding_defaults() -> dict:
    """What a binding has for each key it leaves out, under its provider's own (`defaults`): the FROM_SCHEMA settings
    at the deploy schema's default (_schema_defaults), read at first use, so that importing this module reads no
    schema."""
    return {
        "provider": "openai-compatible",
        "where": "",
        "url": "",
        "chat_path": "/chat/completions",
        "api_key_file": "",
        "api_key_secret": "",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
        "auth_mode": "",
        "model": "",
        "name": "",
        **_schema_defaults(),
        "context": 0,
        "max_output_tokens": 16000,
        "effort": "high",
        "temperature": 0.3,
        "stream": True,
        "stream_usage": True,
        "max_resends": 3,
        "resend_wait_s": 5,
    }


# A provider's own defaults over binding_defaults(), for the keys a binding leaves out: the Messages API's route, and the
# key in x-api-key with no scheme (Claude Code's ANTHROPIC_API_KEY; a gateway that takes ANTHROPIC_AUTH_TOKEN's bearer
# token sets auth_header Authorization and auth_scheme Bearer).
PROVIDER_DEFAULTS = {
    "anthropic": {
        "chat_path": "/v1/messages",
        "auth_header": "x-api-key",
        "auth_scheme": "",
    }
}
VAR_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
ANTHROPIC_API = "https://api.anthropic.com"  # Anthropic's own API: where Claude Code goes with no ANTHROPIC_BASE_URL
WORKER = "worker"  # reviewer: the local worker (config worker)
THINKING_OFF = "none"  # the effort of a local worker that does not think (worker.think false), as line 1 gives it
PARALLEL_MAX = 6
DEFAULT = "default"  # reviewer: no first-pass drafter, the verifier reviews alone
NONE = "none"  # the older spelling of DEFAULT (worker_backend, roles.drafter), still read; and draft_stage's "none"
NO_DRAFT_NAMES = (DEFAULT, NONE)
RESERVED = (DEFAULT, NONE, "worker")  # reviewer choices that are no binding's name


def no_draft(cfg: dict) -> bool:
    """No drafter: the reviewer is the default (`reviewer_choice`: the verifier reviews alone), or `draft_stage` is
    "none"."""
    return (
        reviewer_choice(cfg) == DEFAULT
        or str(cfg.get("draft_stage") or "").lower() == NONE
    )


def verifier_name(cfg: dict) -> str:
    """The model that verifies the posts, as the footer and line 1 with no first-pass drafter name it: config
    verifier_name, else the audit model in words (models.audit: the verifier agent's).
    """
    return str(cfg.get("verifier_name") or config.claude_name(cfg, "audit"))


def older_reviewer(cfg: dict) -> str:
    """The reviewer the older keys name: roles.drafter, else worker_backend ("" when neither is set)."""
    return (
        str((cfg.get("roles") or {}).get("drafter") or "").strip()
        or str(config.older(cfg, "worker_backend") or "").strip()
    )


def reviewer_choice(cfg: dict) -> str:
    """The first-pass reviewer: DEFAULT (the verifier alone), WORKER or a binding's name. Config `reviewer` ("none", the
    older spelling of DEFAULT, reads as DEFAULT; a name model_bindings does not define stays as it is: `problems` names
    it); blank, the older keys (older_reviewer: "none", a binding they name, or the local worker for any other name);
    with neither, WORKER (a bare configuration in code; a configuration file without the key reads as config.normalize
    resolves it: WORKER when a local worker is set up, else DEFAULT)."""
    want = str(cfg.get("reviewer") or "").strip()
    if want:
        return DEFAULT if want.lower() in NO_DRAFT_NAMES else want
    old = older_reviewer(cfg)
    if old.lower() in NO_DRAFT_NAMES:
        return DEFAULT
    if old and isinstance((cfg.get("model_bindings") or {}).get(old), dict):
        return old
    return WORKER


def local_worker_set(cfg: dict) -> bool:
    """A local worker is set up: config `worker` names an endpoint and a model, the older worker_url or worker_model
    does, or an `ollama` section is there."""
    w = cfg.get("worker") if isinstance(cfg.get("worker"), dict) else {}
    return bool(
        (str(w.get("url") or "").strip() and str(w.get("model") or "").strip())
        or str(config.older(cfg, "worker_url") or "").strip()
        or str(config.older(cfg, "worker_model") or "").strip()
        or isinstance(config.older(cfg, "ollama"), dict)
    )


def binding_name(cfg: dict, role: str) -> str:
    """The binding a role names; "" for none. The drafter's is the reviewer's (`reviewer_choice`): "" for the default
    (no drafter) and for the local worker."""
    if role == "drafter":
        choice = reviewer_choice(cfg)
        return "" if choice in (WORKER, DEFAULT) else choice
    return str((cfg.get("roles") or {}).get(role) or "").strip()


def defaults(kind: str) -> dict:
    """What a binding of the provider KIND (PROVIDERS, or an older name OLD_PROVIDERS gives) has for each key it leaves
    out: binding_defaults(), under its provider's own (PROVIDER_DEFAULTS)."""
    kind = OLD_PROVIDERS.get(kind, kind)
    return {
        **binding_defaults(),
        **PROVIDER_DEFAULTS.get(ADAPTER.get(kind, kind), {}),
    }


def binding_by_name(cfg: dict, name: str) -> dict | None:
    """A binding by its name over its provider's defaults (`defaults`), or None when model_bindings does not define
    it."""
    raw = (cfg.get("model_bindings") or {}).get(name) if name else None
    if not isinstance(raw, dict):
        return None
    kind = str(
        raw.get("provider") or raw.get("backend") or binding_defaults()["provider"]
    )
    kind = OLD_PROVIDERS.get(kind, kind)
    provider = ADAPTER.get(kind, kind)
    b = {**defaults(kind), **{k: v for k, v in raw.items() if v is not None}}
    secret = str(b.get("api_key_secret") or "").strip()
    # where the secrets registry keeps it: a file a configuration names is not kept
    b["api_key_file"] = secrets_registry.binding_key_file(cfg, secret) if secret else ""
    if kind == "claude":
        _as_the_session(cfg, b)
    b["provider"] = provider
    b["kind"] = kind
    if kind == "ollama":
        u = str(b["url"]).rstrip("/")
        b["url"] = u if u.endswith("/v1") else u + "/v1"
    if kind == "claude" and str(b["model"]) in _builtin():
        b["model"] = _builtin()[str(b["model"])]
    b["name"] = str(b["name"] or b["model"])
    if not b["where"]:
        b["where"] = "local" if config.is_local_url(str(b["url"])) else "gateway"
    b["binding"] = name
    return b


def _builtin() -> dict:
    from . import deploy_tree

    return deploy_tree.builtin()


def claude_url(cfg: dict) -> str:
    """Where a claude binding's requests go, as the Claude session's do: integrations.claude_code's address
    (orchestrator.gateway), else Anthropic's own API (ANTHROPIC_API).
    """
    from . import orchestrator

    return orchestrator.gateway(cfg) or ANTHROPIC_API


def _as_the_session(cfg: dict, b: dict) -> None:
    """A claude binding B (binding_by_name's, changed in place), reached as the Claude session is wherever it names no
    other way: a blank url, or the session's $ANTHROPIC_BASE_URL, is `claude_url`'s; with no key of its own (no
    api_key_secret), the key is the claude login's (orchestrator.key_path), in the headers the session sends it in
    (key_headers: orchestrator.request_headers of the session's variables, custom_headers.KEY_MARK in the key's place).
    """
    from . import custom_headers, orchestrator, paths

    url = str(b.get("url") or "").strip()
    ref = VAR_REF.fullmatch(url)
    if not url or (ref and ref.group(1) == "ANTHROPIC_BASE_URL"):
        b["url"] = claude_url(cfg)
    if str(b.get("api_key_secret") or "").strip():
        return
    try:
        b["api_key_file"] = str(orchestrator.key_path(cfg))
    except paths.PathsError:
        b["api_key_file"] = ""
    b["key_headers"] = orchestrator.request_headers(
        orchestrator.session_variables(cfg, custom_headers.KEY_MARK)
    )


def binding(cfg: dict, role: str) -> dict | None:
    """The role's binding (binding_by_name), or None when the role has none (no name, or a name model_bindings does
    not define). The drafter's takes the reviewer model's id (worker.model) when it names no model, and its effort is
    the first review pass's (`drafter_effort`).
    """
    b = binding_by_name(cfg, binding_name(cfg, role))
    if b is not None and role == "drafter":
        if not b["model"]:
            b["model"] = worker_model(cfg)
            b["name"] = str(
                (cfg.get("model_bindings") or {}).get(b["binding"], {}).get("name")
                or b["model"]
            )
        b["effort"] = drafter_effort(cfg, b)
    return b


def drafter_effort(cfg: dict, b: dict | None) -> str:
    """The effort the drafter's main work, its hunt, runs at: the one home of the rule, which its requests
    (`client_settings`, through `binding`) and line 1 (`drafter_line`) read, so the two never differ. It is the first
    review pass's effort (effort.first_pass) as each provider takes it:
    - B, a binding to a Claude model (a built-in model, or provider claude): the pass's own effort, else the
      verification pass's (config.role_effort), sent as the Messages API's effort;
    - B, a binding to another model: the pass's own effort, sent as reasoning_effort when the request takes it
      (deploy_tree.OTHER_EFFORTS, which worker/llm_client.py's EFFORTS equals); "default" sends none;
    - no binding (B None), the local worker: worker.think, which the pass's effort sets (deploy_tree; unset on
      "default", which sends none), sent to an Ollama server as it is (false: THINKING_OFF) and to an
      OpenAI-compatible one as reasoning_effort when the request takes it.
    "" where the requests carry none: the model takes its own default. It reads no module of the worker folder, which
    the host's rkmon does not read (rkmon confirm works the effort out too, for the draft stage).
    """
    from . import deploy_tree

    takes = deploy_tree.OTHER_EFFORTS
    if b is None:
        w = config.worker(cfg)
        think = w.get("think")
        if w.get("backend") == "ollama":
            return (
                THINKING_OFF
                if think is False
                else "" if think is True else str(think or "")
            )
        return str(think) if think in takes else ""
    if b["provider"] == ADAPTER["claude"]:
        return config.role_effort(cfg, "first_pass")[0]
    own = config.own_effort(cfg, "first_pass")
    return own if own in takes else ""


def reviewer(cfg: dict) -> dict:
    """The first-pass reviewer in one record, for rk doctor, rk features, rk drafts and the setup summary (never a key):
    {"choice", "kind": none, worker or binding, "model" (the name line 1 shows), "provider", "endpoint" (an address to
    show, or the variable a `$NAME` url names), "line1"}. A binding's name model_bindings does not define has kind
    "missing"."""
    choice = reviewer_choice(cfg)
    out = {
        "choice": choice,
        "kind": choice,
        "model": "",
        "provider": "",
        "endpoint": "",
        "line1": first_line(cfg),
    }
    if choice == DEFAULT:
        out.update(model=verifier_name(cfg), provider="the verifier alone")
    elif choice == WORKER:
        w = config.worker(cfg)
        out.update(
            model=w["model"] or "",
            provider=w["backend"],
            endpoint=config.safe_url(w["url"]),
        )
    else:
        b = binding(cfg, "drafter")
        if b is None:
            out["kind"] = "missing"
        else:
            url = str(b["url"])
            out.update(
                kind="binding",
                model=b["name"],
                provider=b.get("kind") or b["provider"],
                endpoint=url if VAR_REF.fullmatch(url) else config.safe_url(url),
            )
    return out


def reviewer_line(cfg: dict) -> str:
    """The reviewer in a few words (no key), for rk features, rk doctor and the setup summary."""
    r = reviewer(cfg)
    if r["kind"] == DEFAULT:
        return f"default: the verifier ({r['model']}) reviews alone, with no first-pass drafter"
    if r["kind"] == WORKER:
        from .doctor import worker_url_key

        url = worker_url_key({"backend": r["provider"]})
        return (
            f"worker: the local review loop, {r['provider']} model "
            f"{r['model'] or '(model_bindings.worker.model_id not set)'} "
            f"at {r['endpoint'] or '(' + url + ' not set)'}"
        )
    if r["kind"] == "missing":
        return f"{r['choice']}: model_bindings defines no such binding"
    return (
        f"{r['choice']}: {r['model'] or '(no model)'} via {_provider(r['provider'])} at "
        f"{r['endpoint'] or '(no endpoint)'}"
    )


def _provider(name: str) -> str:
    """A binding's provider as the deploy settings name it (claude, ollama, openai-compatible)."""
    return OLD_PROVIDERS.get(name, name)


def migrate(raw: dict, full: bool = True) -> tuple[dict, list[str]]:
    """A configuration rewritten onto `reviewer` (a copy) and what changed, one line each. When `reviewer` is blank it
    gets the choice the older keys make (roles.drafter, worker_backend; their "none" is "default"); with `full` (a whole
    config.json) also when neither is set (reviewer_choice: the local worker, which drafted before the setting); a
    `reviewer` spelled "none" becomes "default". Once `reviewer` is set,
    the older keys go: `reviewer` is the one place the choice lives. With `full`, models.tick "worker" (the older
    default) becomes "" (the rule filter) when the local worker is not set up: a model triage is an opt-in.
    """
    out = json.loads(json.dumps(raw or {}))
    said = []
    roles = out.get("roles") if isinstance(out.get("roles"), dict) else {}
    had = "drafter" in roles or "worker_backend" in out
    if not str(out.get("reviewer") or "").strip() and (had or full):
        out["reviewer"] = reviewer_choice(out)
        src = "roles.drafter" if roles.get("drafter") else "worker_backend"
        said.append(
            f"reviewer = {out['reviewer']}"
            + (f" (from {src})" if older_reviewer(out) else "")
        )
    if str(out.get("reviewer") or "").strip().lower() == NONE:
        out["reviewer"] = DEFAULT
        said.append("reviewer = default (the older spelling none)")
    if str(out.get("reviewer") or "").strip():
        if "drafter" in roles:
            said.append(
                f"roles.drafter ({roles.pop('drafter') or 'blank'}) removed: reviewer decides"
            )
        if "worker_backend" in out:
            said.append(
                f"worker_backend ({out.pop('worker_backend') or 'blank'}) removed: reviewer decides"
            )
    models = out.get("models") if isinstance(out.get("models"), dict) else None
    if (
        full
        and models is not None
        and models.get("tick") == WORKER
        and not worker_model(out)
    ):
        models["tick"] = ""
        said.append(
            'models.tick = "" (the rule filter: the local review loop it named is not set up)'
        )
    return out, said


def drafter_settings(cfg: dict) -> dict:
    """Config `drafter` over the package's (config.defaults.json), entry by entry; the limiter's folder defaults to
    <paths.root>/llm_limits."""
    raw = cfg.get("drafter") or {}
    out = {
        k: (raw[k] if k in raw and raw[k] not in (None, "") else v)
        for k, v in registry.default("drafter").items()
    }
    if not out["limiter_dir"]:
        root = (cfg.get("paths") or {}).get("root") or str(
            pathlib.Path(__file__).resolve().parent.parent.parent
        )
        out["limiter_dir"] = str(pathlib.Path(root) / "llm_limits")
    return out


def client_settings(cfg: dict, role: str = "drafter") -> dict:
    """What worker/llm_client.py and the drafter take for a role: its binding, the drafter's settings, the limiter's
    folder. None when the role has no binding."""
    b = binding(cfg, role)
    if b is None:
        return None
    return dict(drafter_settings(cfg), **b)


def name_settings(cfg: dict, name: str) -> dict | None:
    """What worker/llm_client.py takes for a binding by its name (the tick triage's own model, models.tick): the
    binding and the drafter's settings. None when model_bindings does not define it.
    """
    b = binding_by_name(cfg, name)
    return None if b is None else dict(drafter_settings(cfg), **b)


def parallel_max(cfg: dict) -> int:
    """The most drafts that run at once: draft_parallel_max (glm_parallel_max is read too), at least 1."""
    for key in ("draft_parallel_max", "glm_parallel_max"):
        v = cfg.get(key)
        if v not in (None, ""):
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                break
    return PARALLEL_MAX


def worker_model(cfg: dict) -> str:
    """The local worker's model id: the reviewer model's (config.worker: worker.model, or an older worker_model)."""
    return str(config.worker(cfg)["model"] or "")


def worker_tag(cfg: dict) -> str:
    """The local worker's tag in the assessment: config worker.tag, else the leading letters of its model id."""
    tag = (cfg.get("worker") or {}).get("tag")
    if tag:
        return str(tag)
    m = re.match(r"[A-Za-z]+", worker_model(cfg))
    return m.group(0).lower() if m else "worker"


def tag(cfg: dict, role: str = "") -> str:
    """A drafting role's tag: its binding's name (the local worker's tag for a drafter with none). With no role: the
    drafter under assessment, the shadow drafter's binding when there is one, else the drafter's binding ("" when
    neither has one)."""
    if role == "drafter":
        b = binding(cfg, "drafter")
        return b["binding"] if b else worker_tag(cfg)
    if role:
        b = binding(cfg, role)
        return b["binding"] if b else ""
    for r in ("shadow_drafter", "drafter"):
        b = binding(cfg, r)
        if b:
            return b["binding"]
    return ""


def candidates(cfg: dict) -> list[dict]:
    """The drafters the assessment scores against each verified post: the live drafter (a binding, or the local worker;
    none with no first-pass drafter) and the shadow drafter when there is one; each with its role, tag, binding name and
    model's name."""
    out = []
    live = binding(cfg, "drafter")
    if not no_draft(cfg):  # no drafter: nothing to score
        out.append(
            {
                "role": "drafter",
                "tag": tag(cfg, "drafter"),
                "binding": live["binding"] if live else "",
                "model": live["name"] if live else worker_model(cfg),
            }
        )
    shadow = binding(cfg, "shadow_drafter")
    if shadow:
        out.append(
            {
                "role": "shadow_drafter",
                "tag": shadow["binding"],
                "binding": shadow["binding"],
                "model": shadow["name"],
            }
        )
    return out


def display(cfg: dict, role: str = "drafter") -> str:
    """The name line 1 and the dashboard show for a role's model; "" without a binding."""
    b = binding(cfg, role)
    return b["name"] if b else ""


def first_line(cfg: dict, role: str = "drafter") -> str:
    """Line 1 of a review a role drafts, the one home of the rule: which model and effort it gives, beside this
    deployment's automated reviewer name, in reviewer_name.review_line's form. The drafter's (the reviewer's): with no
    first-pass drafter the verifier's (`no_draft_line`), else the drafter's (`drafter_line`). Another role's (a shadow
    draft, never posted): its binding's model and effort. A configuration's own review_first_line or
    review_first_line_no_draft is not read: line 1 always follows the passes and gives the name.
    """
    if role == "drafter" and no_draft(cfg):
        return no_draft_line(cfg)
    if role == "drafter":
        return drafter_line(cfg)
    b = binding(cfg, role)
    return _review_line(cfg, b["name"], b["effort"]) if b else ""


def _review_line(cfg: dict, model: str, effort: str) -> str:
    """reviewer_name.review_line, MODEL at EFFORT: config.MODEL_DEFAULT where no effort is sent ("")."""
    return reviewer_name.review_line(cfg, model, effort or config.MODEL_DEFAULT)


def no_draft_line(cfg: dict) -> str:
    """Line 1 with no first-pass drafter, the verifier's (the derived key review_first_line_no_draft): verifier_name at
    the verifier's effort (config.role_effort's for verify)."""
    return _review_line(cfg, verifier_name(cfg), config.role_effort(cfg, "verify")[0])


def drafter_line(cfg: dict) -> str:
    """The drafter's line 1 (the derived key review_first_line, unused while no first-pass drafter is set): the
    drafter's binding's name, else the local worker's model ("" with none), at the effort it runs at
    (`drafter_effort`).
    """
    b = binding(cfg, "drafter")
    if b:
        return _review_line(cfg, b["name"], b["effort"])
    model = worker_model(cfg)
    return _review_line(cfg, model, drafter_effort(cfg, None)) if model else ""


def problems(cfg: dict) -> list[str]:
    """What is wrong with the roles, the bindings and the drafter's settings in the configuration."""
    from . import deploy_tree, doctor

    out = reviewer_problems(cfg)
    bindings = cfg.get("model_bindings") or {}
    if not isinstance(bindings, dict):
        return out + ["model_bindings must map names to bindings"]
    for role, name in (cfg.get("roles") or {}).items():
        if role not in ROLES:
            out.append(f"roles.{role} is not a role; the roles are {', '.join(ROLES)}")
        elif (
            name
            and name not in bindings
            and not (role == "drafter" and name.lower() in NO_DRAFT_NAMES)
        ):
            out.append(
                f"roles.{role} names {name!r}, which model_bindings does not define"
            )
    backend = config.older(cfg, "worker_backend")
    if (
        (cfg.get("roles") or {}).get("drafter")
        and backend
        and cfg["roles"]["drafter"] != backend
    ):
        out.append(
            f"roles.drafter ({cfg['roles']['drafter']}) and worker_backend ({backend}) differ"
        )
    for name, raw in bindings.items():
        if not isinstance(raw, dict):
            out.append(f"model_bindings.{name} must be an object")
            continue
        for key in raw:
            if key not in binding_defaults() and key != "backend":
                out.append(f"model_bindings.{name}.{key} is not a binding setting")
        provider = str(
            raw.get("provider") or raw.get("backend") or binding_defaults()["provider"]
        )
        provider = OLD_PROVIDERS.get(provider, provider)
        if provider not in PROVIDERS:
            out.append(
                f"model_bindings.{name}.provider is {provider!r}; it must be one of {', '.join(PROVIDERS)}"
            )
        url = str(raw.get("url") or "").strip()
        # the deploy settings key that gives the binding its address: server_url, base_url or the Claude session's
        url_key = doctor.url_key(name, provider)
        if not url and provider != "claude":
            out.append(f"{url_key} is not set")
        elif url and not (re.match(r"https?://", url) or VAR_REF.fullmatch(url)):
            out.append(
                f"{url_key} is {url!r}; it must be an http or https address, or a variable's reference such as "
                "$ANTHROPIC_BASE_URL"
            )
        elif not VAR_REF.fullmatch(url):
            # an address, which must be one that can be read; a variable's reference is read where the request is made
            out += config.address_problems(url_key, url)
        if raw.get("api_key_secret") and not secrets_registry.NAME.fullmatch(
            str(raw["api_key_secret"])
        ):
            out.append(
                f"model_bindings.{name}.api_key_secret must be a secret's name (lowercase letters, digits and _)"
            )
        if (
            str(raw.get("effort") or binding_defaults()["effort"])
            not in deploy_tree.OTHER_EFFORTS
        ):
            out.append(
                f"model_bindings.{name}.effort must be one of {', '.join(deploy_tree.OTHER_EFFORTS)}"
            )
    b = binding(cfg, "drafter")
    if b and not b["model"]:
        out.append(
            f"the drafter's binding {b['binding']} names no model (its model, or worker.model)"
        )
    shadow = binding(cfg, "shadow_drafter")
    if shadow and b and shadow["binding"] == b["binding"]:
        out.append(
            f"roles.shadow_drafter names the live drafter's binding {b['binding']!r}: a shadow drafts beside "
            "the live draft, with another binding"
        )
    known = registry.default("drafter")
    for key in cfg.get("drafter") or {}:
        if key not in known:
            out.append(
                f"drafter.{key} is not a setting; the settings are {', '.join(known)}"
            )
    d = drafter_settings(cfg)
    try:
        wall, budget = float(d["review_wall_min"]), int(d["review_token_budget"])
        for key in ("prior_wall_min", "hunt_wall_min"):
            v = float(d[key])
            if v <= 0:
                out.append(f"drafter.{key} must be over 0")
            elif wall and v > wall:
                out.append(
                    f"drafter.{key} ({v:g}) is over drafter.review_wall_min ({wall:g}): the prior tasks and the "
                    "hunt run at once inside the run's cap"
                )
        phases = int(d["prior_token_budget"]) + int(d["hunt_token_budget"])
        if budget and phases > budget:
            out.append(
                f"drafter.prior_token_budget + drafter.hunt_token_budget ({phases}) is over "
                f"drafter.review_token_budget ({budget}): the hunt would not keep its own budget"
            )
        if int(d["prior_task_steps"]) < 1 or int(d["prior_parallel"]) < 1:
            out.append(
                "drafter.prior_task_steps and drafter.prior_parallel must be 1 or more"
            )
    except (TypeError, ValueError) as exc:
        out.append(f"the drafter's settings must be numbers: {exc}")
    return out


def reviewer_refusal(cfg: dict) -> list[str]:
    """A `reviewer` that is not none, worker or a binding model_bindings defines (one line, the key as the deploy
    settings file names it), else []."""
    from . import deploy_tree, doctor

    want = str(cfg.get("reviewer") or "").strip()
    defined = (
        cfg.get("model_bindings") if isinstance(cfg.get("model_bindings"), dict) else {}
    )
    if (
        want
        and want.lower() not in NO_DRAFT_NAMES
        and want != WORKER
        and not isinstance(defined.get(want), dict)
    ):
        return [
            f"{deploy_tree.name('reviewer')} is {want!r}, which model_bindings does not define; it must be default, "
            "worker or a "
            f"binding's name ({doctor.apply_words()})"
        ]
    return [
        f"model_bindings.{n}: {n!r} is a reviewer choice, not a binding's name: rename the binding"
        for n in defined
        if str(n).lower() in RESERVED
    ]


def reviewer_problems(cfg: dict) -> list[str]:
    """What is wrong with the reviewer's choice: a `reviewer` that is not none, worker or a binding model_bindings
    defines; an older key that disagrees with it. config.problems reads the first (reviewer_refusal), so a choice that
    names nothing stops the monitor before a review starts.
    """
    out = reviewer_refusal(cfg)
    want = str(cfg.get("reviewer") or "").strip()
    defined = (
        cfg.get("model_bindings") if isinstance(cfg.get("model_bindings"), dict) else {}
    )
    if want:
        for key, value in (
            ("roles.drafter", (cfg.get("roles") or {}).get("drafter")),
            ("worker_backend", config.older(cfg, "worker_backend")),
        ):
            old = str(value or "").strip()
            if old.lower() in NO_DRAFT_NAMES:
                same = want.lower() in NO_DRAFT_NAMES
            elif want == WORKER:
                same = not isinstance(
                    defined.get(old), dict
                )  # an older name that is no binding meant the worker
            else:
                same = old == want
            if old and not same:
                out.append(
                    f"{key} ({old}) is an older name of reviewer ({want}) and says otherwise: reviewer decides; "
                    f"remove {key}"
                )
    return out


def describe(cfg: dict) -> list[str]:
    """The reviewer, then one line per role: its binding, model, provider, the host it is at (config.safe_host; the
    reference a `$NAME` address is), where that is, and its limits (never a key)."""
    out = [
        f"first pass: {reviewer_line(cfg)}; line 1: {first_line(cfg) or '(none yet)'}"
    ]
    for role in ROLES:
        b = binding(cfg, role)
        if not b:
            out.append(
                f"{role}: "
                + (
                    f"default: no drafter, the verifier reviews alone; line 1: {first_line(cfg)}"
                    if role == "drafter" and no_draft(cfg)
                    else "no binding"
                    + (" (the local review loop drafts)" if role == "drafter" else "")
                )
            )
            continue
        url = str(b["url"])
        host = url if VAR_REF.fullmatch(url) else config.safe_host(url)
        out.append(
            f"{role}: {b['binding']} = {b['name']} via {_provider(b['provider'])} at {host} ({b['where']}), "
            f"{b['tpm'] or 'no'} tokens and {b['rpm'] or 'no'} requests a minute"
            + (f", {b['concurrency']} in flight" if b["concurrency"] else "")
            + f", effort {b['effort'] or config.MODEL_DEFAULT}, key "
            + ("file " + json.dumps(b["api_key_file"]) if b["api_key_file"] else "none")
        )
    return out


SHORT_MODEL = re.compile(r"(\w+) \d+(?:\.\d+)*$")


def reference_name(label: str) -> str:
    """The short name of the reference model a footer names: the word before its version ("Vendor Family 5.5" gives
    "Family")."""
    m = SHORT_MODEL.search(label or "")
    return m.group(1) if m else (label or "the verified post")
