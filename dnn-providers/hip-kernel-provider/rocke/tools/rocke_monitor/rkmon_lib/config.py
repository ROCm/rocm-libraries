# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The configuration: where it lives, loading, the values derived from it, and its checks.

- The monitor's home holds config.json (this machine's settings, never committed) and state/.
  `home` finds it: `rk --home DIR` or $RKMON_HOME; else the toolkit's own folder when it holds a config.json (the
  layout before the package); else the folder named in the pointer file (~/.config/rkmon/home, `pointer_path`);
  else the toolkit's own folder. The package itself is never written at run time.
- The defaults (`load_defaults`) and the descriptions (`doc`) are the registry's (rkmon_lib/registry.py), which has
  every key and its one owner: the deploy settings file's schema (schema/deploy_settings.schema.json) holds the
  default and the description of every key the file feeds (rkmon_lib/deploy_tree.py carries them onto config.json's
  keys); config.defaults.json holds the package's own values, the internal state's schema the state's defaults.
  docs/internal_settings.md is generated from the registry.
- Secrets never go in either file: they live in `paths.secrets_dir` (<WORKING_DIR>/.config, mode 700, files 600) or in
  each tool's own store (gh, the Claude Code MCP settings for JIRA and Confluence). A configuration from before the
  package keeps reading ~/.config/rkmon.

`load` is the running configuration (rkmon_lib/runtime_config.py): each key from its owner's layer, what derives from
other keys computed (every path, line 1 of a review, the code-owner org), and a required key that is blank refused,
naming each one (`problems`). `folders` is the part of it that names the folders, which the emergency brake runs with:
never the internal state or the delivered layer. `normalize` fills in the derived values of a configuration file read
on its own. `env` is what the worker tools get from a job: the RKMON_* environment. `with_override` is a worker
tool's --config: the configuration with a JSON file's keys in place of its own.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import re
from typing import NamedTuple

from . import container_layout, features, paths, registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.json"
DEFAULTS = registry.PACKAGE_FILE

BACKENDS = ("ollama", "openai")
# The reviewer model's settings (config `worker`: the first-pass model that drafts every review). The key goes in the
# header auth_header, after auth_scheme ("" sends it raw, as an API-management subscription key wants); auth_mode
# "both" also sends it as `Authorization: Bearer <key>` (a gateway that wants the key twice), "bearer" only that way
# ("": as auth_header and auth_scheme say); no other header is sent. max_parallel, rpm and tpm: what the
# endpoint allows (0: not known); probe: the start's checks (rkmon_lib/models.py). For an OpenAI-compatible endpoint:
# chat_path, its chat route under the base ("": /v1/chat/completions), models_url, where it lists its models ("":
# <base>/v1/models), and model_name, the display name of `model` (the ID requests send, such as a deployment ID). Each
# key's default is the registry's (`worker_defaults`), but for the keys the deploy settings' worker binding sets, which
# have none there: WORKER_UNSET holds what the worker takes when the binding leaves one out.
WORKER_UNSET = {
    "backend": "ollama",
    "url": "",
    "model": "",
    "api_key_secret": "",
    "auth_header": "Authorization",
    "auth_scheme": "Bearer",
    "auth_mode": "",
    "think": None,  # none: the requests carry no effort, and the model takes its own default
    "stall_s": 900,
    "max_parallel": 0,
    "rpm": 0,
    "tpm": 0,
}
# Worker keys whose blank value means something (a raw key, without a scheme), so a blank does not fall back.
WORKER_BLANK_OK = ("auth_scheme",)
AUTH_MODES = ("", "header", "bearer", "both")
CLAUDE = "claude"
CLAUDE_MARKERS = (CLAUDE, "claude-opus")
# The orchestrator model (`orchestrator.model`): the model of the Claude Code session that runs the tick loop,
# verification, the gate decisions and posting, and of the verifier and code-quality subagents. Any model Claude Code
# can run is accepted, a gateway model it reaches through an Anthropic-compatible endpoint included. The skill, its
# procedures and the agent files are known to work well with ORCHESTRATOR_KNOWN, the current Claude Opus and Sonnet
# families (an alias, or a model id, also as Bedrock or Vertex name it); any other choice gets one warning
# (`orchestrator_warning`).
CLAUDE_ALIASES = ("opus", "sonnet", "haiku", "fable", "opusplan")
ORCHESTRATOR_KNOWN = re.compile(
    r"opus|sonnet|opusplan|(?:(?:[a-z]{2,4}\.)?anthropic\.)?claude-(?:opus|sonnet)-"
    r"[a-z0-9][a-z0-9.@:-]*"
)
# Roles and who does them (config `models`; each role's default is the registry's, `role_setting`; a message names a
# role's key by `model_key`). The orchestrator (Claude, `orchestrator.model`) never hands off verification, gate
# decisions, overlap owner decisions, or anything posted: those roles are always Claude ("claude", or the marker
# "claude-opus", which means the same). `tick` (a package key: no deploy setting sets it) is the tick triage of the
# container's orchestrator (rkmon_lib/triage.py): who reads each tick before Claude does. "": the toolkit's rule filter
# (rkmon_lib/tickfilter.py, `rk tick-filter`), no model; a model is opt-in: "worker" (the local worker's model) or a
# binding's name. It never follows the reviewer by itself. quality_draft, notice_draft and ci_triage are the first
# passes of three tasks a Claude role owns (rkmon_lib/first_pass.py), each the model of its pass in the deploy
# settings: "default" (no first pass: the owner does the whole task), "worker" (the local worker's model) or a
# binding's name; the model hands the owner leads, never findings.
ROLES = (
    "draft",
    "quality_draft",
    "notice_draft",
    "ci_triage",
    "tick",
    "verify",
    "quality",
    "post",
    "analytics",
    "planner",
    "analyst",
    "audit",
    "deep",
)
CLAUDE_ONLY = ("verify", "quality", "post", "audit", "deep")
# Claude's roles that run on a model of their own (Opus at effort high by default, for the audits and the in-depth
# problem solving): `audit`, Claude's check of a review the reviewer model wrote and of anything that
# goes out; `deep`, the in-depth work when the reviewer model's output shows a big inconsistency. Each is a model as
# Claude Code names it, or "claude" for the orchestrator model; blank means the default.
CLAUDE_MODELS = ("audit", "deep")
FIRST_PASS_ROLES = ("ci_triage", "quality_draft", "notice_draft")
# The values of the tick triage's model and of a first pass's that name no model: the rule filter reads each tick, and
# a first pass's owner does the whole task.
NO_MODEL = ("", "off", "default", *CLAUDE_MARKERS)
# The optional on-prem model roles beside the reviewer (rkmon_lib/roles.py), on the reviewer's endpoint with a share of
# its rate limits: the analytics model ("" off, "worker", or {"model", "share", "timeout_min", ...}), which the planner
# and the analyst use unless one is set on its own (the same forms) or "off".
MODEL_ROLES = ("analytics", "planner", "analyst")
MODEL_NAME = re.compile(r"[A-Za-z0-9][\w.:@/\[\]-]*")
# The tokens of context at which the orchestrator's session compacts itself (`claude.auto_compact_window`, Claude
# Code's autoCompactWindow): the bounds a value must keep. The session runs on a 1M-context model.
COMPACT_RANGE = (10000, 1000000)
# The prompt cache's lifetimes Claude Code takes (`claude.prompt_cache_ttl`, `claude.subagent_prompt_cache_ttl`: its
# promptCacheTtl and subagentPromptCacheTtl settings).
CACHE_TTLS = ("5m", "1h")
# Claude's effort: each Claude role's own (`effort.<role>`, EFFORT_ROLES: the deploy settings' passes and the
# orchestrator), else, for an optional pass (OWNER_ROLE), its owner's, else the base (`claude.effort`), else the model's
# own default (no effort is passed: MODEL_DEFAULT). A configuration without one of these keys takes its default
# (the registry's).
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MODEL_DEFAULT = "the model's default"
EFFORT_ROLES = {
    "orchestrator": "the orchestrator's Claude session, which also posts (the Claude Agent SDK's effort option)",
    "verify": "the verifier agent (its agent file's effort: <VERIFY_EFFORT>)",
    "quality": "the code quality agent (its agent file's effort: <QUALITY_EFFORT>)",
    "audit": "the verifier's offload audit (<AUDIT_EFFORT>)",
    "deep": "a deep review's verifier and code quality pass (<DEEP_EFFORT>)",
    "tick": "the tick triage, when a Claude binding does it",
    "first_pass": "the first review pass's draft (a model that is not Claude takes low, medium or high)",
    "ci_triage": "CI triage's first pass, when a claude model does it",
    "notice_draft": "a notice's first draft, when a claude model does it",
    "quality_draft": "the first code quality pass, when a claude model does it",
}
# The optional passes and the role that owns each: blank, a pass takes its owner's effort.
OWNER_ROLE = {
    "first_pass": "verify",
    "ci_triage": "verify",
    "quality_draft": "quality",
    "notice_draft": "orchestrator",
}

# Keys every command needs (dotted, why). The commands that work without a configuration (rkmon.py NO_CONFIG) report
# what is missing instead.
REQUIRED = (
    ("repo", "rocKE's GitHub repository, as owner/name"),
    ("rocke_path", "rocKE's folder inside the repository"),
    (
        "operator.github_login",
        "the GitHub login the monitor posts as, and whose instructions it follows (the operator's account)",
    ),
    ("paths.working_dir", "the one folder the monitor owns"),
    ("scope.labels", "the PR labels in scope (at least one)"),
)
# The local worker's keys, needed while review automation is on and the first pass is the local worker (reviewer
# "worker"): with no first pass, or another model binding, nothing reads them.
WORKER_REQUIRED = (
    (
        "worker.model",
        "the local review loop's model (it drafts every review)",
    ),
    (
        "worker.url",
        "the local review loop's endpoint (Ollama or an OpenAI-compatible server)",
    ),
)
# paths.rocm_llvm_bin is not among them: without ROCm's LLVM tools the monitor runs. The develop guard is incomplete
# until it is set (features), and the checks' compile lanes skip with the reason, which leaves the compile gate (G7)
# pending, never passed (review.gates_from_status): a machine with no ROCm and a remote reviewer runs every rk command.


class ConfigError(ValueError):
    """A configuration that cannot be read or cannot run: MESSAGE says why, and FIX is the fix it names (a
    checks.Command, Edit or Steps)."""

    def __init__(self, message: str, fix=None):
        super().__init__(message)
        self._fix = fix

    def fix(self, program_name: str):
        """The fix this error names, as the host's command PROGRAM_NAME runs it (checks.Command, Edit or Steps);
        None when it names none. Each error of the configuration's load names one (runtime_config.load, `load`); a
        refused change to the internal state names its caller's (state_store.set)."""
        return self._fix


class NoConfiguration(ConfigError):
    """There is no configuration: no delivered deploy layer and no config.json (runtime_config.load)."""


def pointer_path() -> pathlib.Path:
    """The file that names the monitor's home: $RKMON_POINTER, else ~/.config/rkmon/home (under $XDG_CONFIG_HOME when
    set). It is not a secret."""
    if os.environ.get("RKMON_POINTER"):
        return pathlib.Path(os.path.expanduser(os.environ["RKMON_POINTER"]))
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return pathlib.Path(base) / "rkmon" / "home"


def configures_something(path: pathlib.Path) -> bool:
    """A config.json with settings in it. The package ships a blank one (only `_comment`), which names no home."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return False
    except ValueError:
        return True  # a broken configuration is still one: loading it names the problem
    return isinstance(raw, dict) and any(not k.startswith("_") for k in raw)


def home_and_why() -> tuple[pathlib.Path, str]:
    """The monitor's home (config.json, state/) and how it was found."""
    env = os.environ.get("RKMON_HOME", "").strip()
    if env:
        return pathlib.Path(os.path.expanduser(env)), "RKMON_HOME (or rk --home)"
    if configures_something(
        CONFIG
    ):  # the layout before the package: the toolkit folder holds config.json and state/
        return ROOT, "the toolkit's own folder has config.json"
    ptr = pointer_path()
    try:
        named = ptr.read_text(encoding="utf-8").strip()
    except OSError:
        named = ""
    if named:
        return pathlib.Path(os.path.expanduser(named)), f"the pointer file {ptr}"
    return ROOT, "no home is set: the toolkit's own folder"


def home() -> pathlib.Path:
    return home_and_why()[0]


def config_path() -> pathlib.Path:
    """config.json in the monitor's home."""
    return home() / "config.json"


def state_dir(home_dir=None) -> pathlib.Path:
    """state/ in the monitor's home: HOME_DIR when given, else `home()`."""
    return pathlib.Path(home() if home_dir is None else home_dir) / paths.STATE


def toolkit_state_dir(cfg: dict) -> pathlib.Path:
    """state/ in CFG's toolkit folder (paths.toolkit, the monitor's home): the files the session's runs, the tick
    triage and the first passes keep there."""
    return state_dir(paths.get(cfg, "toolkit"))


# The follow-ups folder in state/: the follow-up tickets' files (tickets_<N>.md) and, by default, the follow-up scans'
# files (followup_scan.dir, a derived key)
FOLLOWUPS = "followups"


def _merged(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _merged(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def load_defaults() -> dict:
    """Every key's default, nested: the registry's (rkmon_lib/registry.py, `defaults`)."""
    return registry.defaults()


def doc() -> dict:
    """Dotted key -> its description: the registry's (rkmon_lib/registry.py, `descriptions`); a key the deploy
    settings file feeds is described by the file key that names it, after that key."""
    return registry.descriptions()


def lookup(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# The names of a configuration file from before today's keys that the readers still take: the running configuration
# (runtime_config) holds the registry's keys only, so these are read from a configuration file alone (`older`).
OLDER_NAMES = ("worker_model", "worker_url", "worker_backend", "ollama", "scope_label")


def older(cfg: dict, name: str):
    """The value a configuration file holds under the older NAME (one of OLDER_NAMES); None when it holds none, as the
    running configuration never does. ValueError for any other name."""
    if name not in OLDER_NAMES:
        raise ValueError(f"{name} is no older name of the configuration")
    return dict.get(cfg, name) if isinstance(cfg, dict) else None


def operator_login(cfg: dict) -> str:
    """The operator's GitHub login (config `operator.github_login`): the account the monitor posts as, and whose
    instructions it follows. The skill and the review book name it `<user_login>`; the toolkit fills it in.
    """
    op = cfg.get("operator") if isinstance(cfg, dict) else None
    return (
        str((op or {}).get("github_login") or "").strip()
        if isinstance(op, dict)
        else ""
    )


def jira_accounts(cfg: dict) -> dict:
    """GitHub login -> JIRA account (an email or an account id): config `jira.accounts`."""
    a = (cfg.get("jira") or {}).get("accounts")
    return dict(a) if isinstance(a, dict) else {}


def min_free_memory_gb(cfg: dict) -> float:
    """The free memory a native build needs (config `checks.min_free_memory_gb`)."""
    return registry.number(cfg, "checks.min_free_memory_gb", float)


class Scope(NamedTuple):
    """The labels in scope (`scope`): `labels`, each of which puts a pull request in scope; `shared`, each shared label
    with its folder, from the repository's root, which puts a pull request in scope only when it changes files in that
    folder."""

    labels: tuple[str, ...]
    shared: dict[str, str]

    def searched(self) -> list[str]:
        """Every label a search for the pull requests in scope takes: the labels, then the shared labels."""
        return [*self.labels, *(x for x in self.shared if x not in self.labels)]

    def folders(self, labels) -> list[str] | None:
        """The folders a pull request with LABELS must change files in, one of them at least, to be in scope: None when
        one of LABELS puts it in scope on its own; else the folders of its shared labels, sorted (none: no label of
        LABELS is in scope)."""
        have = set(labels or ())
        if have & set(self.labels):
            return None
        return sorted({self.shared[x] for x in have & set(self.shared)})

    def takes(self, labels, changes_files_in) -> bool:
        """The one rule of scope: a pull request with LABELS is in scope when one of them puts it in scope on its own,
        or is a shared label whose folder the pull request changes files in (CHANGES_FILES_IN(folder), asked of
        `folders` in their order until one answers yes)."""
        folders = self.folders(labels)
        return folders is None or any(changes_files_in(f) for f in folders)

    def shared_text(self) -> str:
        """The shared labels, each with its folder: `needs-ci (projects/a), other (projects/b)`; blank when there are
        none."""
        return ", ".join(f"{label} ({folder})" for label, folder in self.shared.items())


def scope(cfg: dict) -> Scope:
    """The labels in scope, the one home every reader asks: config `scope.labels` (the single `scope_label` of older
    configurations), and `scope.path_labels`, the deploy settings file's repository.shared_label_folders, each shared
    label with its folder. config.json's form of the shared labels, a list of labels, gives each the repository's folder
    (paths.repository_folder), the one folder each counted for there."""
    sc = cfg.get("scope") or {}
    if sc:
        labels = sc.get("labels") or []
    else:
        label = older(cfg, "scope_label")
        labels = [label] if label else []
    shared = sc.get("path_labels") or {}
    if not isinstance(shared, dict):
        folder = paths.repository_folder(cfg)
        shared = {x: folder for x in shared}
    return Scope(tuple(labels), {x: str(f).strip("/") for x, f in shared.items()})


def worker_defaults() -> dict:
    """The reviewer model's settings at their defaults: the registry's `worker` keys, and WORKER_UNSET's."""
    return dict(WORKER_UNSET, **registry.defaults("worker"))


def worker_settings(cfg: dict) -> dict:
    """The reviewer model's settings as the configuration states them, with no file read: config `worker`, over
    `worker_defaults` (a blank or null value takes the default, but for WORKER_BLANK_OK); a configuration from before
    the package names only `worker_model` (and the local Ollama it ran, models.OLLAMA_DEFAULT).
    """
    from . import models  # a call-time import: models imports this module

    w = dict(
        worker_defaults(),
        **{
            k: v
            for k, v in (cfg.get("worker") or {}).items()
            if v is not None and (v != "" or k in WORKER_BLANK_OK)
        },
    )
    w["model"] = w.get("model") or older(cfg, "worker_model") or ""
    w["url"] = (
        w.get("url")
        or older(cfg, "worker_url")
        or (
            models.OLLAMA_DEFAULT
            if older(cfg, "worker_model") and not cfg.get("worker")
            else ""
        )
    ).rstrip("/")
    return w


def worker(cfg: dict) -> dict:
    """The reviewer model's settings, read here and nowhere else: `worker_settings`, whose key file is the place the
    load derives for the secret `api_key_secret` names (the derived key worker.api_key_file: rkmon_lib/derived.py).
    """
    return worker_settings(cfg)


REVIEWER = "the reviewer model"  # how the pages and lines name it when no model is set


def reviewer_name(cfg: dict) -> str:
    """The reviewer model as the board, the dashboards and the lines name it (never a model the package assumes): a
    drafter binding's name; the local worker's display name (worker.model_name, a gateway list's), else its ID without a
    provider prefix or an Ollama tag (provider/name:tag -> name); REVIEWER while none is set (with no first-pass drafter
    too)."""
    from . import bindings  # a call-time import: bindings imports this module

    b = None if bindings.no_draft(cfg) else bindings.binding(cfg, "drafter")
    if b:
        return b["name"] or REVIEWER
    w = worker(cfg)
    name = (w.get("model_name") or "").strip() or (w.get("model") or "").strip().rsplit(
        "/", 1
    )[-1].split(":", 1)[0]
    return name or REVIEWER


def model_for(cfg: dict, role: str) -> dict:
    """Who does a role: {"by": "worker", **settings} for the reviewer model (a dict in `models` overrides the worker
    settings for that role), {"by": "claude", "model": <its Claude model>} for Claude (`claude_model`), {"by": "none"}
    when nobody drafts it first (the orchestrator does the whole job). verify, quality, post, audit and deep are always
    Claude's, whatever `models` says. The tick triage's is `tick_model`'s, and the first passes' (FIRST_PASS_ROLES)
    `named_model`'s."""
    if role == "tick":
        return tick_model(cfg)
    if role in FIRST_PASS_ROLES:
        from .first_pass import (
            OWNER,
        )  # a call-time import: first_pass imports this module

        return named_model(cfg, role, f"{OWNER[role]} works without a first pass")
    want = role_setting(cfg, role)
    if role in CLAUDE_ONLY or want in CLAUDE_MARKERS:
        return {"by": "claude", "model": claude_model(cfg, role)}
    if isinstance(want, dict):
        return dict(
            worker(cfg),
            **{k: v for k, v in want.items() if v not in (None, "")},
            by="worker",
        )
    return dict(worker(cfg), by="worker") if want == "worker" else {"by": "none"}


def role_setting(cfg: dict, role: str):
    """What `models.<ROLE>` holds, else its default (the registry's); "" for a name that is no role."""
    return registry.value(cfg, f"models.{role}") if role in ROLES else ""


def model_key(role: str) -> str:
    """How a message names the key of ROLE's model (`models.<role>`, ROLES): its deploy key (deploy_tree.name); a
    package key, which no deploy setting sets (the tick triage's, and the fixed "claude" of verify, quality and post),
    as "the package's" key; a derived key by its name."""
    from . import deploy_tree

    key = f"models.{role}"
    if registry.key(key).owner == registry.PACKAGE:
        return f"the package's {key}"
    return deploy_tree.name(key)


RULES_ONLY = "the rule filter (rk tick-filter) reads each tick"


def tick_model(cfg: dict) -> dict:
    """The tick triage's model (models.tick), an opt-in: {"by": "worker", **the local worker's settings} (a dict in
    models.tick overrides them), {"by": "binding", **the binding's client settings} for a binding's name, else
    {"by": "none", "why": ...}: the default, "" (or "off", or "claude", which read the same), is the toolkit's rule
    filter (rkmon_lib/tickfilter.py) with no model, RULES_ONLY. The triage never follows the reviewer by itself.
    """
    return named_model(cfg, "tick", RULES_ONLY)


def named_model(cfg: dict, role: str, otherwise: str) -> dict:
    """The model a role names in `models` (the tick triage, the first passes): {"by": "worker", **the local worker's
    settings} for "worker" (a dict overrides them), {"by": "binding", **the binding's client settings} for a
    model_bindings name, else {"by": "none", "why": ...}: NO_MODEL ("default", "", "off", "claude") or a name no binding
    has; `otherwise` says what happens then. The why names the role's key by `model_key`.
    """
    from . import bindings  # a call-time import: bindings imports this module

    want = role_setting(cfg, role)
    if isinstance(want, dict):
        return dict(
            worker(cfg),
            **{k: v for k, v in want.items() if v not in (None, "")},
            by="worker",
        )
    want = str(want or "").strip()
    if want in NO_MODEL:
        return {"by": "none", "why": f"{model_key(role)} names no model: {otherwise}"}
    if want == "worker":
        return dict(worker(cfg), by="worker")
    s = bindings.name_settings(cfg, want)
    if s is None:
        return {
            "by": "none",
            "why": f"{model_key(role)} names {want!r}, which model_bindings does not define: {otherwise}",
        }
    return dict(s, by="binding")


def probe_failure(w: dict) -> str:
    """Why the reviewer model is refused for its role, from the start's last check of it (`worker.probe`): it failed
    the test chat or the tool-call probe. "" when it passed, was never checked, was not checked because the endpoint
    kept answering HTTP 429 ("not checked: ...": a rate limit is no failure of the model), or the check was of another
    endpoint or model (the settings changed since)."""
    p = w.get("probe") or {}
    if (
        not isinstance(p, dict)
        or not p
        or p.get("url") != w.get("url")
        or p.get("model") != w.get("model")
    ):
        return ""
    for step, what in (("chat", "the test chat"), ("tools", "the tool-call probe")):
        v = p.get(step)
        if v and v != "ok" and not str(v).startswith("not checked"):
            return f"failed {what}: {v}"
    return ""


def orchestrator_model(cfg: dict) -> str:
    """The orchestrator model (`orchestrator.model`), as Claude Code takes it (an alias or a model id)."""
    return str(registry.value(cfg, "orchestrator.model", registry.FALSY)).strip()


def claude_model(cfg: dict, role: str = "orchestrator") -> str:
    """A Claude role's model, as Claude Code takes it: audit and deep from `models` (CLAUDE_MODELS: "claude" is the
    orchestrator model; blank, or a value that names the reviewer model, which `problems` refuses, the default), every
    other role the orchestrator model."""
    if role not in CLAUDE_MODELS:
        return orchestrator_model(cfg)
    want = role_setting(cfg, role)
    want = (
        want.strip()
        if isinstance(want, str) and want.strip() not in ("", "worker")
        else registry.default(f"models.{role}")
    )
    return orchestrator_model(cfg) if want in CLAUDE_MARKERS else want


def claude_model_id(cfg: dict, role: str = "orchestrator") -> str:
    """The id a Claude role's model (`claude_model`) is sent by: a built-in model's id, else the model as it is named
    (deploy_tree.model_id)."""
    from . import deploy_tree

    return deploy_tree.model_id(claude_model(cfg, role))


def own_effort(cfg: dict, role: str) -> str:
    """A role's own effort (EFFORT_ROLES), `effort.<role>` (a configuration without the key: its default): one of
    EFFORTS, else "" (blank, or an optional pass's "default": no effort of its own)."""
    v = registry.value(cfg, f"effort.{role}") if role in EFFORT_ROLES else ""
    v = str(v or "").strip().lower()
    return v if v in EFFORTS else ""


def role_effort(cfg: dict, role: str) -> tuple[str, str]:
    """A Claude role's effort (EFFORT_ROLES) and where it comes from, as the deploy settings name it: its own
    (`own_effort`), else, for an optional pass, its owner's (OWNER_ROLE), else the base (`claude.effort`), else ""
    (MODEL_DEFAULT: no effort is passed, and the model takes its own default).
    """
    from . import deploy_tree

    v = own_effort(cfg, role)
    if v:
        return v, deploy_tree.name(f"effort.{role}")
    if role in OWNER_ROLE:
        e, _src = role_effort(cfg, OWNER_ROLE[role])
        return e, f"{deploy_tree.name(f'effort.{OWNER_ROLE[role]}')} (its owner's)"
    # A configuration without the key gets the shipped base (its default); only a blank one is the model's default.
    base = str(registry.value(cfg, "claude.effort") or "").strip().lower()
    if base in EFFORTS:
        return base, "claude.effort"
    return "", MODEL_DEFAULT


BINDING_EFFORTS = EFFORTS  # what a request to a Claude binding takes (worker/llm_client.py CLAUDE_EFFORTS)


def effort_problems(cfg: dict) -> list[str]:
    """The `effort` section's problems: a key that is no Claude role, a value outside EFFORTS, and a role bound to a
    Claude binding whose effort the binding cannot take."""
    own = cfg.get("effort")
    if own not in (None, "") and not isinstance(own, dict):
        return ["effort must map roles to efforts"]
    from .deploy_tree import name as N

    out = []
    for role, v in (own or {}).items():
        if role not in EFFORT_ROLES:
            out.append(
                f"effort.{role} is not a Claude role; the roles are {', '.join(EFFORT_ROLES)}"
            )
        elif v not in (None, "") and str(v).strip().lower() not in EFFORTS:
            out.append(
                f"{N('effort.' + role)} is {v!r}; it must be one of {', '.join(EFFORTS)}"
            )
    return out


def bound_effort(cfg: dict, role: str, m: dict) -> str | None:
    """The effort a role's request to its bound model M carries, in the values M's provider takes:
    - a Claude binding (provider anthropic, or a model id that names Claude): the role's effort (`role_effort`: its
      own, else its owner's, else the base);
    - any other model, the local worker (the binding named worker) among them: the role's own effort when the model
      takes it (deploy_tree.OTHER_EFFORTS: reasoning_effort, or an Ollama server's think), else None, so an optional
      pass's "default" sends none.
    None for a role no model is bound to.
    """
    from . import deploy_tree

    if m.get("by") not in ("binding", "worker"):
        return None
    claude = m.get("by") == "binding" and (
        m.get("provider") == "anthropic"
        or "claude" in str(m.get("model") or "").lower()
    )
    if claude:
        return role_effort(cfg, role)[0] or None
    own = own_effort(cfg, role)
    return own if own in deploy_tree.OTHER_EFFORTS else None


def orchestrator_known(name: str) -> bool:
    """One of the models known to drive the skill, its procedures and the agent files well (ORCHESTRATOR_KNOWN), with
    or without Claude Code's suffix in brackets (its context window: deploy_tree.base_id).
    """
    from . import deploy_tree

    return bool(ORCHESTRATOR_KNOWN.fullmatch(deploy_tree.base_id(name).lower()))


def orchestrator_warning(name: str) -> str:
    """The one warning for an orchestrator model outside ORCHESTRATOR_KNOWN; "" for one of them. Never a refusal."""
    if orchestrator_known(name):
        return ""
    return (
        f"{str(name).strip()} is not one of the models known to drive the monitor well (the current Claude Opus "
        "and Sonnet families): the skill files and the verifier agents are long and tool-heavy, so a less capable "
        "model may skip steps or misread the rules"
    )


def orchestrator_name(cfg: dict) -> str:
    """The orchestrator model in words, for a review's footer: "Claude Opus" for the alias, "Claude Opus 5.5" for
    claude-opus-5-5 (a date suffix dropped); any other model as it is named."""
    return model_words(orchestrator_model(cfg))


def claude_name(cfg: dict, role: str) -> str:
    """A Claude role's model in words (`claude_model`), as `orchestrator_name` says it."""
    return model_words(claude_model(cfg, role))


def model_words(named: str) -> str:
    """A model in words, without Claude Code's suffix in brackets (its context window: deploy_tree.base_id): "Claude
    Opus" for the alias, "Claude Opus 5.5" for claude-opus-5-5; any other model as it is named.
    """
    from . import deploy_tree

    named = deploy_tree.base_id(named)
    m = named.lower()
    if m in CLAUDE_ALIASES:
        return "Claude " + ("Opus" if m == "opusplan" else m.capitalize())
    hit = re.search(r"claude-([a-z]+)((?:-\d{1,2}(?!\d))*)", m)
    if not hit:
        return named
    version = ".".join(p for p in hit.group(2).split("-") if p)
    return f"Claude {hit.group(1).capitalize()}" + (f" {version}" if version else "")


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@functools.lru_cache(maxsize=None)
def pipes_max() -> int:
    """The most parallel reviews (`review.pipes`: first-pass drafts at once): the deploy schema's maximum of
    parallel_reviews, the file key that sets them, which is the limit the settings file's checks and the settings
    reference show. Read at the first call, never on import."""
    return _schema_maximum("review.pipes")


@functools.lru_cache(maxsize=None)
def checks_max() -> int:
    """The most heavy local jobs at once (`checks.limit`: builds and GPU tests, for review checks, gates, develop sanity
    and bisects): the deploy schema's maximum of parallel_builds, the file key that sets them, like pipes_max. Read at
    the first call, never on import."""
    return _schema_maximum("checks.limit")


def _schema_maximum(key: str) -> int:
    """The deploy schema's maximum of the file key that sets the configuration key KEY, through deploy_tree, which
    reads the schema through file_types at its first use."""
    # a call-time import: this module loads without the schema's modules
    from . import deploy_tree

    return deploy_tree.node(deploy_tree.file_key(key))[1]["maximum"]


def pipes_asked(cfg: dict) -> int:
    """`review.pipes`, held to 1..pipes_max()."""
    return max(1, min(pipes_max(), registry.number(cfg, "review.pipes", int, ())))


def checks_limit(cfg: dict) -> int:
    """`checks.limit`: the parallel builds at once, held to 1..checks_max(). The parallel reviews never change it."""
    return max(1, min(checks_max(), registry.number(cfg, "checks.limit", int, ())))


def _read_address(url) -> tuple:
    """URL read as an address: (urlsplit's parts; its host as an address shows it, in brackets for an IPv6 one; its
    port; why it cannot be read, "" when it can). An address with no scheme is read as host[:port][/path]. It cannot
    be read when urlsplit cannot read its host (an IPv6 host with no closing bracket, brackets around no IPv6
    address), when it names no host, or when its port is not a number from 0 to 65535, which leaves the port None.
    """
    import urllib.parse

    raw = str(url or "").strip()
    try:
        u = urllib.parse.urlsplit(raw if "://" in raw else "//" + raw)
        hostname = u.hostname
    except ValueError:
        return None, "", None, "its host cannot be read"
    if not hostname:
        return u, "", None, "it names no host"
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        return u, host, u.port, ""
    except ValueError:
        return u, host, None, "its port is not a number from 0 to 65535"


def safe_url(url: str) -> str:
    """An endpoint's address to show: scheme, host, port and path; never a user, password, query or fragment, where a
    credential could hide. An address with no scheme is read as host[:port][/path] (so a key in its query is never
    shown either), and a port that is not a number is left out; an address with no host that can be read reads as
    "(not an address)".
    """
    if not str(url or "").strip():
        return ""
    u, host, port = _read_address(url)[:3]
    if not host:
        return "(not an address)"
    scheme = f"{u.scheme}://" if u.scheme else ""
    return f"{scheme}{host}{f':{port}' if port else ''}{u.path}"


def safe_host(url: str) -> str:
    """The host safe_url shows of URL, an IPv6 one in brackets; "(not an address)" when it names none that can be
    read."""
    return _read_address(url)[1] or "(not an address)"


# What safe_value shows for a text in the form of a token (privacy.TOKENS): no setting holds one.
HIDDEN = "(hidden: it has the form of a token, and a secret belongs in its own file)"


def _names_a_host(text: str) -> bool:
    """Whether TEXT is an address with a scheme and a host, where a credential could hide."""
    import urllib.parse

    if "://" not in text:
        return False
    try:
        return bool(urllib.parse.urlsplit(text).hostname)
    except ValueError:
        return False


def safe_value(value):
    """VALUE, a configuration value, as it is shown (rk config show, the confirm's summary): an address without its
    user, password, query and fragment (safe_url), a text in the form of a token (privacy.TOKENS), which no setting
    holds, as HIDDEN, each value of a list or a map so, and any other value as it is."""
    from . import privacy  # a call-time import: the privacy check is a large module

    if isinstance(value, str):
        if privacy.TOKENS.search(value):
            return HIDDEN
        return safe_url(value) if _names_a_host(value) else value
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    if isinstance(value, dict):
        return {k: safe_value(v) for k, v in value.items()}
    return value


def address_problems(key: str, url) -> list[str]:
    """KEY's address URL in one line naming KEY when it cannot be read (_read_address: urlsplit cannot read its host,
    it names none, or its port is not a number from 0 to 65535); [] for one that can, and for a blank one, which KEY's
    own rules judge."""
    why = _read_address(url)[3] if str(url or "").strip() else ""
    return [f"{key} is {url!r}; {why}"] if why else []


def is_local_url(url: str) -> bool:
    """Whether URL names this host (container_layout.LOCAL_HOSTS, the one list of its names); an address that cannot
    be read does not."""
    import urllib.parse

    try:
        return (
            urllib.parse.urlsplit(url or "").hostname or ""
        ) in container_layout.LOCAL_HOSTS
    except ValueError:
        return False


def reviewer_is_local(cfg: dict) -> bool:
    """The reviewer model runs on this machine (a local Ollama, or any endpoint on the loopback address): its prompts
    load this machine's memory and CPU, so the prefill rules apply."""
    return is_local_url(worker(cfg)["url"])


def pipes(cfg: dict) -> dict:
    """The parallel reviews at the start: {"asked", "effective", "why"}. `asked` is review.pipes, the scaler's starting
    setting (rkmon_lib/scaler.py moves it within 1..pipes_max()); in effect: min(asked, what the reviewer endpoint
    allows), and the scaler never goes past that cap either. A local Ollama (on the loopback address) keeps one model in
    memory, so it allows 1 unless the operator confirmed more (`worker.max_parallel`, once they have checked the model's
    size, the free memory and Ollama's parallel setting); any other endpoint allows what the operator confirmed, else
    pipes_max(), and the parallel reviews share its rate limits (`worker.rpm`, `worker.tpm`). With no first-pass drafter
    (the verifier reviews alone) or a drafter binding, the reviewer endpoint caps nothing.
    """
    from . import deploy_tree, doctor

    asked = pipes_asked(cfg)
    from . import draft_stage  # a call-time import: draft_stage imports this module

    stage = draft_stage.for_config(cfg)
    if stage is not draft_stage.LOCAL:
        # The reviewer endpoint's limits bind only the reviewer model's own worker: with no drafter, or a drafter
        # binding (whose limits are the limiter's), they do not cap the parallel reviews.
        why = (
            "no drafter runs (the verifier reviews alone), so no model endpoint limits the parallel reviews"
            if stage is draft_stage.NONE
            else "the drafts run on the drafter binding: the shared limiter keeps to its rate limits, and the monitor "
            "runs fewer reviews at once while its endpoint is spent"
        )
        return {"asked": asked, "effective": asked, "why": why}
    w = worker(cfg)
    confirmed = max(0, _int(w.get("max_parallel"), 0))
    if w["backend"] == "ollama" and is_local_url(w["url"]):
        cap, key = confirmed or 1, deploy_tree.name("worker.max_parallel")
        why = (
            f"the reviewer is a local Ollama, and you confirmed {cap} at once ({key})"
            if cap > 1
            else "the reviewer is a local Ollama, which keeps one model in memory: once you have checked the memory, "
            + doctor.apply_words(edit=f"raise {key} in your deploy settings file")
        )
    else:
        cap = confirmed or pipes_max()
        rpm, tpm = max(0, _int(w.get("rpm"), 0)), max(0, _int(w.get("tpm"), 0))
        where = (
            "a local endpoint"
            if is_local_url(w["url"])
            else "an endpoint outside the container"
        )
        limits = " and ".join(
            x
            for x in (f"{rpm} requests" if rpm else "", f"{tpm} tokens" if tpm else "")
            if x
        )
        why = (
            f"the reviewer is {where}"
            + (
                f", allowed {confirmed} at once ({deploy_tree.name('worker.max_parallel')})"
                if confirmed
                else ""
            )
            + (
                f"; the parallel reviews share its limits of {limits} a minute"
                if limits
                else f"; no rate limits known ({deploy_tree.name('worker.rpm')}, {deploy_tree.name('worker.tpm')}), so "
                "the reviews back off when it answers 429"
            )
        )
    return {"asked": asked, "effective": min(asked, cap), "why": why}


def pipes_line(cfg: dict) -> str:
    """ "asked N, running M, because …", for rk status and the setup summary."""
    p = pipes(cfg)
    return f"asked {p['asked']}, running {p['effective']}, because {p['why']}"


def first_line(cfg: dict) -> str:
    """The drafter's line 1 (bindings.drafter_line): the line for the reviewer's model (a drafter binding's, or the
    local worker's), with the automated reviewer name. With no first-pass drafter, line 1 is bindings.first_line's.
    """
    from . import bindings  # a call-time import: bindings imports this module

    return bindings.drafter_line(cfg)


def tracker_keys(cfg: dict) -> list[str]:
    """Ticket-key prefixes that must never appear in code or in a link: `tracker_keys` and the JIRA project. Empty
    means any Jira-style ID (KEY-NNN)."""
    keys = [
        str(k).strip().upper()
        for k in (cfg.get("tracker_keys") or [])
        if str(k).strip()
    ]
    project = str((cfg.get("jira") or {}).get("project") or "").strip().upper()
    return list(dict.fromkeys(keys + ([project] if project else [])))


def first_pass(raw: dict):
    """The first pass a configuration names (`reviewer`); with neither it nor the older keys, the default (the verifier
    alone) unless a local worker is set up. normalize and runtime_config fill it in, and problems reads the same.
    """
    from . import bindings

    if str(raw.get("reviewer") or "").strip() or bindings.older_reviewer(raw):
        return raw.get("reviewer")
    return bindings.WORKER if bindings.local_worker_set(raw) else bindings.DEFAULT


def first_pass_choice(raw: dict) -> str:
    """The first pass a configuration runs, as bindings.reviewer_choice names it (DEFAULT, WORKER or a binding's name):
    the one it names, else as normalize fills it in."""
    from . import bindings

    return bindings.reviewer_choice(dict(raw, reviewer=first_pass(raw)))


def normalize(raw: dict) -> dict:
    """The configuration with the derived values filled in: every path resolved (`paths.resolve`), the weekly report's
    folder, line 1 of a review, the code-owner org (the repository's owner) and the places of the secrets and the
    logins, as the load derives them (`_places`). The input is not changed; it carries the current key names (load
    and every other reader of a file migrate them first: renames.migrate).
    """
    cfg = dict(raw, reviewer=first_pass(raw))
    cfg["paths"] = dict(raw.get("paths") or {}, **paths.resolve(raw))
    cfg["weekly_report"] = dict(
        raw.get("weekly_report") or {}, dir=cfg["paths"]["weekly_dir"]
    )
    cfg["review_first_line"] = first_line(dict(raw, reviewer=cfg.get("reviewer")))
    if not cfg.get("codeowner_org") and "/" in (raw.get("repo") or ""):
        cfg["codeowner_org"] = raw["repo"].split("/", 1)[0]
    cfg.setdefault("codeowner_teams", {})
    cfg.setdefault("hot_spots", {})
    _places(cfg)
    return cfg


def _places(cfg: dict) -> None:
    """Fill in the places of the secrets and the logins in CFG, normalize's copy, as the load does: each of
    derived.SECRET_PLACES, then worker.api_key_file, by the load's own functions (rkmon_lib/derived.py), for the layout
    this runs in. A place CFG gives is not kept (derived.py): it becomes the derived one, blank while no secrets folder
    is known. Each mapping on the way is copied, so the input's stay as they are."""
    from . import derived, doctor, registry

    layout = derived.Layout(container=doctor.deployment_image())
    values = {}
    for k in registry.keys():
        v = registry.get(cfg, k.name)
        if v is not registry.MISSING:
            values[k.name] = v
    for key in (*derived.SECRET_PLACES, "worker.api_key_file"):
        given = str(values.get(key) or "").strip()
        values[key] = place = derived.FUNCTIONS[key](values, layout)
        if not place and not given:
            continue
        *sections, leaf = key.split(".")
        cur = cfg
        for s in sections:
            cur[s] = dict(cur.get(s) or {})
            cur = cur[s]
        cur[leaf] = place


def _blank(cfg: dict, dotted: str) -> bool:
    if dotted.startswith("paths."):
        return not paths.resolve(cfg).get(dotted.split(".", 1)[1])
    if dotted.startswith("worker."):
        return not worker(cfg).get(dotted.split(".", 1)[1])
    if dotted == "scope.labels":
        return not [x for x in scope(cfg).labels if str(x).strip()]
    value = lookup(cfg, dotted)
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or value in ([], {})
    )


def problems(cfg: dict) -> list[str]:
    """What is wrong with a configuration, one line each: a blank required key, a value outside its choices, a
    relative path, an address that cannot be read (address_problems). Empty when it can run. `cfg` carries the current
    key names (renames.migrate).
    """
    from .deploy_tree import name as N

    out = []
    for dotted, why in REQUIRED:
        if _blank(cfg, dotted):
            out.append(f"{N(dotted)} is blank: {why}")
    from . import bindings  # a call-time import: bindings imports this module

    out += bindings.reviewer_refusal(cfg)
    if (
        features.on(cfg, "review-drafts")
        and first_pass_choice(cfg) == bindings.WORKER
        and not bindings.no_draft(cfg)
    ):
        for dotted, why in WORKER_REQUIRED:
            if _blank(cfg, dotted):
                out.append(
                    f"{N(dotted)} is blank: {why} (or name another model, or default, in "
                    f"{N('reviewer')})"
                )
    out += features.problems(cfg)
    if cfg.get("repo") and not re.fullmatch(r"[\w.-]+/[\w.-]+", cfg["repo"]):
        out.append(f"{N('repo')} is {cfg['repo']!r}; it must be owner/name")
    w = worker(cfg)
    if w["backend"] not in BACKENDS:
        out.append(
            f"worker.backend is {w['backend']!r}; it must be one of {', '.join(BACKENDS)}"
        )
    if w["url"] and not re.match(r"https?://", w["url"]):
        out.append(
            f"worker.url is {w['url']!r}; it must start with http:// or https://"
        )
    else:
        out += address_problems("worker.url", w["url"])
    if w.get("chat_path") and not str(w["chat_path"]).startswith("/"):
        out.append(
            f"worker.chat_path is {w['chat_path']!r}; it must be a route under worker.url, such as "
            "/chat/completions"
        )
    if w.get("models_url") and not re.match(r"https?://", str(w["models_url"])):
        out.append(
            f"worker.models_url is {w['models_url']!r}; it must be a full address (http:// or https://)"
        )
    else:
        out += address_problems("worker.models_url", w.get("models_url"))
    if w.get("auth_mode") not in AUTH_MODES:
        out.append(
            f"worker.auth_mode is {w.get('auth_mode')!r}; it must be one of header, bearer, both (or blank)"
        )
    if not re.fullmatch(r"[A-Za-z0-9-]+", str(w.get("auth_header") or "")):
        out.append(
            f"worker.auth_header is {w.get('auth_header')!r}; it must be an HTTP header name"
        )
    for key in ("max_parallel", "rpm", "tpm"):
        if _int(w.get(key), -1) < 0:
            out.append(
                f"worker.{key} is {w.get(key)!r}; it must be a whole number, 0 or more (0: not known)"
            )
    for role, want in (cfg.get("models") or {}).items():
        dflt = registry.default(f"models.{role}") if role in ROLES else ""
        if role not in ROLES:
            out.append(f"models.{role} is not a role; the roles are {', '.join(ROLES)}")
        elif role in MODEL_ROLES:
            out += _model_role_problems(role, want)
        elif role in CLAUDE_MODELS:
            if want == "worker" or isinstance(want, dict):
                out.append(
                    f'{model_key(role)} must be a Claude model, or "{CLAUDE}" for the orchestrator model: Claude\'s '
                    f"{role} never goes to the reviewer model"
                )
            elif not isinstance(want, str) or (
                want.strip() and not MODEL_NAME.fullmatch(want.strip())
            ):
                out.append(
                    f"{model_key(role)} is {want!r}; use a model as Claude Code names it (an alias such as "
                    f'{dflt}, or a model id), "{CLAUDE}" for the orchestrator model, or "" for '
                    f"{dflt}"
                )
        elif role in CLAUDE_ONLY and want not in CLAUDE_MARKERS:
            out.append(
                f"{model_key(role)} must be {CLAUDE!r}: the orchestrator (Claude) never hands off verification, "
                "gates or posting to the reviewer model"
            )
        elif role in ("tick", *FIRST_PASS_ROLES):
            named = (
                cfg.get("model_bindings")
                if isinstance(cfg.get("model_bindings"), dict)
                else {}
            )
            if not (
                isinstance(want, dict)
                or want in (*NO_MODEL, "worker")
                or (isinstance(want, str) and isinstance(named.get(want), dict))
            ):
                out.append(
                    f"{model_key(role)} is {want!r}; use "
                    + (
                        '"" (the rule filter, rk tick-filter: the default)'
                        if role == "tick"
                        else '"default" (no first pass: the default)'
                    )
                    + ', or a model: "worker" (the local review loop\'s) or a model_bindings name'
                )
            elif (
                role == "quality_draft"
                and want not in NO_MODEL
                and str((cfg.get("roles") or {}).get("quality") or "").strip()
            ):
                out.append(
                    f"{model_key(role)} and roles.quality both name a model: the quality binding's code quality pass "
                    "is final and no code quality reviewer runs, so the first pass's leads would have no reader; "
                    "keep one"
                )
        elif role not in CLAUDE_ONLY and not (
            want in ("", "worker", *CLAUDE_MARKERS) or isinstance(want, dict)
        ):
            out.append(
                f'{model_key(role)} is {want!r}; use "worker", "{CLAUDE}", "" or a backend/model object'
            )
    effort = (cfg.get("claude") or {}).get("effort")
    if effort not in (None, "") and str(effort).strip().lower() not in EFFORTS:
        out.append(
            f"claude.effort is {effort!r}; it must be one of {', '.join(EFFORTS)}"
        )
    out += effort_problems(cfg)
    window = (cfg.get("claude") or {}).get("auto_compact_window")
    if window not in (None, "") and not (
        isinstance(window, int)
        and not isinstance(window, bool)
        and COMPACT_RANGE[0] <= window <= COMPACT_RANGE[1]
    ):
        out.append(
            f"claude.auto_compact_window is {window!r}; it must be a whole number from {COMPACT_RANGE[0]} to "
            f"{COMPACT_RANGE[1]} (tokens of context)"
        )
    for key in ("prompt_cache_ttl", "subagent_prompt_cache_ttl"):
        ttl = (cfg.get("claude") or {}).get(key)
        if ttl not in (None, "") and ttl not in CACHE_TTLS:
            out.append(
                f"claude.{key} is {ttl!r}; it must be one of {', '.join(CACHE_TTLS)}"
            )
    tick = (cfg.get("claude") or {}).get("tick_min")
    if tick not in (None, "") and not (
        isinstance(tick, int) and not isinstance(tick, bool) and tick > 0
    ):
        out.append(
            f"{N('claude.tick_min')} is {tick!r}; it must be a whole number of minutes, 1 or more"
        )
    out += _dashboard_problems(cfg)
    for dotted, low, high in (
        ("review.pipes", 1, pipes_max()),
        ("checks.limit", 1, checks_max()),
    ):
        v = lookup(cfg, dotted)
        if v not in (None, "") and not (
            isinstance(v, int) and not isinstance(v, bool) and low <= v <= high
        ):
            out.append(
                f"{N(dotted)} is {v!r}; it must be a whole number from {low} to {high}"
            )
    for key, value in paths.resolve(cfg).items():
        # Not the RAM disk's folder (paths.ramdisk), which is derived, the container's, and so no setting a line could
        # name: the RAM disk's one check, ramdisk.problems, is of its size (host.resources.ramdisk_gib).
        if value and not value.startswith("/") and key != "ramdisk":
            out.append(
                f"paths.{key} is {value!r}; it must be an absolute path (or start with ~)"
            )
    out += _gh_binary_problems(cfg)
    from . import (
        gpu,
        offload,
        ramdisk,
    )  # the remote GPU tests' settings (a blank one is features.missing's)

    return out + gpu.problems(cfg) + offload.problems(cfg) + ramdisk.problems(cfg)


def _gh_binary_problems(cfg: dict) -> list[str]:
    """In the package's image, a set paths.gh: every gh call there goes through gh on PATH, the GitHub request cap's
    counter (gh.settings), so a gh the configuration names would never run."""
    from .doctor import (
        deployment_image,
    )  # a call-time import: doctor imports this module

    binary = paths.resolve(cfg).get("gh")
    if not binary or not deployment_image():
        return []
    return [
        f"paths.gh is {binary!r}; in the container every gh call goes through the image's own gh, behind the GitHub "
        "request cap's counter on PATH: blank paths.gh"
    ]


def _ipv4(value) -> bool:
    import ipaddress

    try:
        ipaddress.IPv4Address(str(value))
        return True
    except ValueError:
        return False


def _dashboard_problems(cfg: dict) -> list[str]:
    """The two dashboards' places (the deploy settings' features.dashboards): addresses are IPv4 addresses of this
    machine (127.0.0.1 the default, 0.0.0.0 all of them), ports are port numbers, an address a browser opens is a full
    http(s) one that can be read (address_problems). The admin dashboard is published on the host's 127.0.0.1 alone,
    so it has no address to publish it on. Each line names its setting by its deploy key (deploy_tree.name).
    """
    from .deploy_tree import name as N

    out = []
    for sec, addresses, ports in (
        ("admin_dashboard", ("bind",), ("port", "public_port")),
        ("team_dashboard", ("bind", "publish_host"), ("port", "public_port")),
    ):
        d = cfg.get(sec) or {}
        for key in addresses:
            if d.get(key) not in (None, "") and not _ipv4(d[key]):
                out.append(
                    f"{N(sec + '.' + key)} is {d[key]!r}; it must be an IPv4 address: {container_layout.LOOPBACK} "
                    "(this machine only), "
                    f"another address of this machine, or {container_layout.EVERY_ADDRESS} (all of them)"
                )
        for key in ports:
            low = 0 if key == "public_port" else 1
            if d.get(key) not in (None, "") and not low <= _int(d[key], -1) <= 65535:
                out.append(
                    f"{N(sec + '.' + key)} is {d[key]!r}; it must be a port number"
                    + (
                        " (0: the same as the port, or one docker picks)"
                        if key == "public_port"
                        else ""
                    )
                )
        if d.get("url") and not re.match(r"https?://[^/\s]+", str(d["url"])):
            out.append(
                f"{N(sec + '.url')} is {d['url']!r}; it must be a full address (http:// or https://)"
            )
        else:
            out += address_problems(N(sec + ".url"), d.get("url"))
    mode = (cfg.get("team_dashboard") or {}).get("mode")
    if mode not in (None, "", "publish", "serve"):
        out.append(
            f"{N('team_dashboard.mode')} is {mode!r}; it must be publish (the team web host) or serve (this monitor)"
        )
    j = cfg.get("jira") or {}
    for key in ("priority_map", "extra_fields"):
        if j.get(key) not in (None, "") and not isinstance(j[key], dict):
            out.append(f"jira.{key} must be a mapping (a name to a value)")
    return out


def _model_role_problems(role: str, want) -> list[str]:
    """models.analytics, models.planner and models.analyst: "" (analytics: off; a role: the analytics model), "off" (a
    role), "worker" (the reviewer's own model), or {"model": a model the reviewer's endpoint lists, "share": 0.01-1 of
    its rate limits, "timeout_min": 1 or more, "every_h": hours between the analyst's readings, "act": reserved,
    false}."""
    if want in ("", "worker") or (want == "off" and role != "analytics"):
        return []
    if not isinstance(want, dict):
        forms = '"" (off)' if role == "analytics" else '"" (the analytics model), "off"'
        return [
            f'models.{role} is {want!r}; use {forms}, "worker" (the reviewer\'s own model) or '
            '{"model": ...}: it runs on the reviewer\'s endpoint, never as the orchestrator'
        ]
    out = (
        []
        if str(want.get("model") or "").strip()
        else [
            f"models.{role}.model is blank: name a model the reviewer's endpoint lists, "
            f"or set models.{role} to "
            + ('"" (off)' if role == "analytics" else '"off"')
        ]
    )
    if "share" in want:  # one left out is roles.DEFAULTS'
        try:
            share = float(want["share"])
        except (TypeError, ValueError):
            share = -1.0
        if not 0.01 <= share <= 1:
            out.append(
                f"models.{role}.share is {want['share']!r}; it must be a share of the endpoint's rate limits "
                "from 0.01 to 1"
            )
    for key in ("timeout_min", "every_h"):
        if key in want and _int(want[key], 0) < 1:
            out.append(
                f"models.{role}.{key} is {want[key]!r}; it must be a whole number, 1 or more"
            )
    switch = "act"  # kept for a future switch: the role only reads
    if want.get(switch):
        out.append(
            f"models.{role}.{switch} is reserved for a future switch and must stay false: the {role} only reads, and "
            "changes nothing"
        )
    return out


def load(path: pathlib.Path | None = None, check: bool = True) -> dict:
    """The configuration the monitor runs, nested: runtime_config.load's values, each key from its owner's layer (the
    delivered deploy layer in the container, the internal state of the monitor's home, the package's defaults), and,
    while the load reads no layer, config.json at PATH (the home's by default) one more source: beside a layer it is
    never read (the one-time move splits it). NoConfiguration when there is no configuration,
    runtime_config.LayerError and state_store.StateError when the delivered layer or the internal state cannot be
    read (each a ConfigError); with CHECK, a ConfigError naming every problem when the configuration cannot run,
    its fix the settings fixed in the deploy settings file, then confirmed and applied (doctor.cannot_run_fix). Each
    error names its fix (ConfigError.fix).
    """
    # call-time imports: runtime_config and doctor import this module
    from . import doctor, runtime_config

    path = pathlib.Path(path or config_path())
    resolved = runtime_config.load(home(), config_file=path)
    if check and resolved.problems:
        raise ConfigError(
            f"{path} cannot run yet:\n  - " + "\n  - ".join(resolved.problems),
            doctor.cannot_run_fix(),
        )
    return resolved.values


def folders(path: pathlib.Path | None = None) -> dict:
    """The configuration the emergency brake runs with (rk stop, kill, undrain and stopped), nested: what names the
    folders, the container's layout and, while no layer is delivered (runtime_config.delivered), config.json at PATH
    (the home's by default) (runtime_config.folders); never the delivered layer or the internal state, so that neither
    keeps the monitor from stopping. NoConfiguration on a host without config.json, and a ConfigError when config.json
    is no JSON object, each naming its fix.
    """
    # a call-time import: runtime_config imports this module
    from . import runtime_config

    path = pathlib.Path(path or config_path())
    return runtime_config.folders(home(), config_file=path).values


def with_override(cfg: dict, path) -> dict:
    """CFG with the keys of the JSON object in the file at PATH in place of its own: a worker tool's --config
    (worker/drafter.py, worker/quality_lane.py). ConfigError naming the file and why when it cannot be read, is not
    valid JSON or holds no JSON object."""
    path = pathlib.Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(
            f"the --config file {path} cannot be read: {exc.strerror or exc}"
        ) from None
    except ValueError as exc:
        raise ConfigError(
            f"the --config file {path} is not valid JSON: {exc}"
        ) from None
    if not isinstance(raw, dict):
        raise ConfigError(f"the --config file {path} is no JSON object")
    return dict(cfg, **raw)


def env(cfg: dict) -> dict[str, str]:
    """The RKMON_* environment a job gives the worker tools (worker/*): paths and the settings they need."""
    p = paths.resolve(cfg)
    # The reviewer model's settings as the jobs get them: without who chose the model (by), and without its last probe's
    # record (probe), which stays in the internal state, as its key_sent is a digest of the headers that carry the key.
    draft = {
        k: v for k, v in model_for(cfg, "draft").items() if k not in ("by", "probe")
    }
    return {
        "RKMON_WORKING_DIR": p["working_dir"],
        "RKMON_REPO_DIR": p["repo"],
        "RKMON_TOOLKIT": p["package"],
        "RKMON_HOME": str(home()),
        "RKMON_RUN": p["run"],
        "RKMON_WORKER_TOOLS": p["worker_tools"],
        "RKMON_WORKERS": p["workers"],
        "RKMON_BUILD": p["build"],
        "RKMON_VENV": p["venv"],
        "RKMON_TMP": p["tmp"],
        "RKMON_SCRATCH": p["scratch"],
        "RKMON_LLVM_BIN": p["rocm_llvm_bin"],
        "RKMON_RAMDISK": p["ramdisk"],
        "RKMON_REVIEW_BOOK": p["review_book"],
        "RKMON_REPO": cfg.get("repo") or "",
        "RKMON_ROCKE_PATH": paths.repository_folder(cfg),
        "RKMON_SPARSE": " ".join(paths.sparse_dirs(cfg)),
        "RKMON_OPERATOR_LOGIN": operator_login(cfg),
        "RKMON_WORKER_MODEL": draft.get("model", ""),
        "RKMON_WORKER_URL": draft.get("url", ""),
        # Where the reviewer model runs, never its address: what PR code's tools/worker_grace.py reads (worker/pr_env.py)
        "RKMON_WORKER_REMOTE": (
            "1" if draft.get("url") and not is_local_url(draft["url"]) else "0"
        ),
        "RKMON_WORKER_JSON": json.dumps(draft, sort_keys=True),
        "RKMON_CLONE_SINCE": cfg.get("clone_since") or "",
        "RKMON_TRACKER_KEYS": ",".join(tracker_keys(cfg)),
        "RKMON_AGENTIC": p["agentic"],
        "RKMON_PROBES": p["probes"],
        # The read login only, and no atlassian CLI settings: jobs and the worker tools never write to GitHub and never
        # use the atlassian CLI (the session gets the same read login from orchestrator.logins_env).
        **_gh_env(cfg),
        **_offload_env(cfg),
    }


def _offload_env(cfg: dict) -> dict:
    """The review offload's mode and the passes' settings (rkmon_lib/offload.py)."""
    from . import offload

    return offload.job_env(cfg)


def _gh_env(cfg: dict) -> dict:
    """RKMON_GH (the gh binary every tool calls) and the read login's folder as GH_CONFIG_DIR (gh.settings)."""
    from . import gh

    binary, env = gh.settings(cfg)
    return {"RKMON_GH": binary, **env}


def export_line(cfg: dict) -> str:
    """`export RKMON_...=...` for a generated job script."""
    import shlex

    return "export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env(cfg).items())
