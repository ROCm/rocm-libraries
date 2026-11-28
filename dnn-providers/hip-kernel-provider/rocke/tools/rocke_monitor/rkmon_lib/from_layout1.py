# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The step from layout 1 to layout 2 of the deploy and the build settings: each file type's
`upgrades[1]` (rkmon_lib/file_types.py), which `settings_file.upgrade` runs for `rkmon migrate`.

- `deploy(values, file_type) -> (values, notes)` and `build(values, file_type) -> (values, notes)`: a layout 1
  file's values, as `settings_file.read` gives them to migrate (the header's two keys left out, each plain value
  as written: answers.Plain), in layout 2, FILE_TYPE being layout 2's file type of the kind, which `upgrade` gives
  each step: its schema's template values and its section command are the ones a note names. Each key goes to its
  new home, and the rules below say what is more than one value moved to one key. A value is carried
  as it was read, for `upgrade` to type by layout 2's schema; a value the step makes (a switch from two, a mode, a
  label list, seconds from minutes) is typed. NOTES are the lines of migrate's report, one sentence each: a move to
  a new name ("OLD is now NEW"), a value set and why, a value dropped, a value reported and not carried, and one line
  that counts the values under `advanced` equal to layout 1's defaults, which are dropped. The note for a value
  carried to a new key is a settings_file.Moved, which keeps both keys, so migrate can name the file's own key.
- Pure and offline: a step uses no network, reads no file but layout 1's frozen defaults (DEFAULTS_FILE) and
  FILE_TYPE's schema (for a placeholder), and changes neither the values it is given nor anything else.
- A value the step may neither carry nor drop raises settings_file.StepRefused, named by its key in the earlier file,
  with what the file has and the fix: an admin dashboard address other than 127.0.0.1 and two different JIRA sites,
  `advanced.jira.url` and `integrations.atlassian_cli.jira_site` (deploy), and two different Python package indexes
  (build). A bare JIRA site name, as the atlassian CLI names a cloud site, is the address it stands for
  (`https://<name>.atlassian.net`), both compared and carried (`jira_address`). A `secret_values` section is left in
  place for `upgrade` to refuse; `read` refuses it first.
- Layout 1's rules are public functions on plain values (`labels`, `guard_own`, `notices`, `pass_effort`,
  `key_sending` and the others under "layout 1's rules" below), each with its frozen constants and its contract. They
  write no note and read no document: the step calls them and writes its own notes, and a translation of layout 1's
  configuration can call the same rules.

The rules beyond one value moved to one key:

- A blank value ("" or null) is not carried, and makes no note: layout 2's template value or default takes its place,
  and means what the blank meant. But a blank or missing `container.name` is "rkmon", a blank or missing image name
  (in either file) "rocke-pr-monitor", with no note, a blank or missing image tag the container's name, a blank user
  or group id null, and a blank timeout on the binding named `worker` 900 seconds. An ollama binding's blank or
  missing `limits.max_parallel` is `limits.max_parallel_requests` 0, no cap, what it ran on; on the binding named
  `worker` at this host's Ollama, a blank, missing or 0 one is 1, as it ran one request at a time. Both are
  reported.
- Alerts, ticket creation and the team dashboard get `mode: "live"` when the file has them on (they sent whenever on),
  ticket creation while a type of ticket is on, as layout 2 has no switch for the section; one that is off is written
  as any feature that is off, with no mode. Both alerts take the alerts' switch; each type of ticket, the section's
  switch AND its trigger, a missing trigger being on. The team dashboard's port and address go to `serve` only with
  delivery "serve", and the web host's keys of `advanced` to `publish` only with delivery "publish": each delivery
  takes its own block alone (check 7), and the other block's values are dropped with a note, the web host's
  saying that their values stay in the earlier file. The page's address and priority requests are carried whatever
  the delivery; the web host's SSH key, its own key and the page's place have no key in layout 2, and each note says
  where it goes (`why_dropped`).
- With the merge queue on, its queue notices are on, and its drift notices are on while review automation is (Claude
  writes them). With its develop guard on too, that guard's labels are written as layout 1's develop guard used them:
  the first of `incident_issue.labels` (else "post-merge"), the others and the companion label "ci-red" it added on a
  red CI, and `stability_label` (else "rocke-stability"). Labels of the file's own are
  written whatever the switches; a develop guard that is off and keeps its defaults is written as its switch alone
  (`guard_labels_written`, which a translation calls too).
- Each pass's model: on an optional pass, "default", "none", "off", "claude", "claude-opus" or blank is "default";
  "worker" stays when the file binds a model named worker, else is "default", and the note says how to bind one. On
  a pass that always runs, which takes no "default", "default" or "none" is the model layout 1 ran the pass on when
  the file named none, and a note says so; a blank one is left out. Any other model is carried as it is.
  Efforts: on an optional pass, "default", blank or an effort layout 1 did not take is "default", the effort of the
  pass in charge, as layout 1 resolved it; every other effort is written as layout 1's chain resolved it: the role's
  own, else (blank or not an effort) `advanced.claude.effort`, else the package's base effort; a missing one is the
  role's default. Where that chain ended in the model's own effort, which layout 2 cannot name for a pass that always
  runs, the role's default is written, and a note says so.
- `advanced`: its human keys move to their homes, and the bindings' `auth_*` to `api_key_sent_as`,
  `api_key_header` and `api_key_scheme` by `key_sending`, so the key goes as it went. `automerge_guard_own` true is
  `on_your_pull_requests: "same_as_others"`, and false `"alert_only"`; a blank one is false, as layout 1 read it, and
  moves as false does. `update.repo` and `update.branch` are the `updates` section when the repository is a GitHub
  address or <org>/<repo>, written as <org>/<repo>, and the branch is set (`updates_section`); else the section is
  left out, and a note says why. `update.path` is dropped with a note, as the package knows its own folder. Every
  other value equal to its package default is dropped and counted; any other, a blank one too, is reported and not
  carried.

What a value meant in layout 1 is frozen beside the step, so the step means the same in every later package:
the constants below, each naming where layout 1's package read it, and layout 1's defaults, which the rule for
`advanced` compares with: DEFAULTS_FILE, layout 1's config.defaults.json beside layout 1's schemas. The build's ROCm
pins alone are this package's, which the file's are compared with: rkmon_lib/build_plan.py's, the package's build
defaults, for the file's ROCm release.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import copy
import json
import re
import urllib.parse

from . import (
    answers,
    build_plan,
    container_layout,
    host_keys,
    program,
    secrets_registry,
    settings_file,
    steps,
    wrapping,
)
from .checks import DELETE, Described, Edit, Steps
from .coherence import TEAM
from .paths import PACKAGE
from .settings_file import StepRefused

RA = "features.review_automation"
RGT = f"{RA}.remote_gpu_tests"
MQ = "features.merge_queue"
GUARD = f"{MQ}.develop_guard"
ADMIN = "features.dashboards.admin"
ALERTS = "features.alerts"
TC = "features.ticket_creation"

# Layout 1's defaults, frozen beside layout 1's schemas: config.defaults.json as layout 1's package had it, its notes
# left out. The rule for `advanced` compares each value with them.
DEFAULTS_FILE = PACKAGE / "schema" / "layout1" / "config.defaults.json"

# What a value of layout 1 meant, each frozen here apart from this package's own defaults, which may change.
# A blank container.name: layout 1's meaning, frozen: the compose project layout 1's package used.
CONTAINER_NAME = "rkmon"
# A blank image name: layout 1's meaning, frozen: the image name layout 1's package used.
IMAGE_NAME = "rocke-pr-monitor"
# A blank Python package index: layout 1's meaning, frozen: the public index layout 1's package used.
PUBLIC_PYPI = "https://pypi.org/simple"
# A clang commit that checks none: layout 1's meaning, frozen: the word layout 1's package took for it.
ANY_CLANG = "any"
# Layout 1's build settings keys that move as they are, each with its key in layout 2's: the image's, the
# operating system's, and each tool's version and download. The other build keys have rules of their own: the image's
# name, the ROCm pins (_Build.rocm_pins), the Agent SDK's version and the one package index. Layout 1's keys, frozen:
# layout 2's build schema names each in its x-internal.
BUILD_MOVES = {
    "image.tag": "image.tag",
    "image.push_registry": "image.registry",
    "image.base": "operating_system.base_image",
    "image.ca_certificates": "operating_system.ca_certificates",
    "os_packages.package_mirror": "operating_system.package_mirror",
    **{
        f"{tool}.{key}": f"{tool}.{key}"
        for tool in ("rocm", "github_cli", "atlassian_cli")
        for key in ("version", "package_url", "package_sha256")
    },
}
# A blank or 0 admin dashboard port: layout 1's meaning, frozen: layout 1's package read it from
# deployment.DEFAULT_PORT.
ADMIN_PUBLIC_PORT = 18614
# A bare JIRA site name, the atlassian CLI's name for a cloud site (one label: no scheme, no dot), and the address it
# stands for: layout 1's meaning, frozen: layout 1's package read it as <name>.atlassian.net.
SITE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?")
CLOUD_SITE = "https://{name}.atlassian.net"
# The team dashboard's deliveries, each taking its block of the same name alone (check 7): publish copies the page
# to a web host, serve listens.
PUBLISH_DELIVERY, SERVE_DELIVERY = "publish", "serve"
# A blank team dashboard delivery: layout 1's meaning, frozen: layout 1's package read it in tools/rkweb.py.
DELIVERY = "publish"
LIVE = "live"  # the mode of a feature that sends
# A missing ticket trigger: layout 1's meaning, frozen: layout 1's package read it in features.switch, from its deploy
# schema's default.
TRIGGER_DEFAULT = True
# The efforts a Claude role took: layout 1's meaning, frozen: layout 1's package read it from config.EFFORTS.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT = (
    "default"  # an optional pass's model and effort: the pass in charge does the work
)
# An optional pass's models that meant "default": layout 1's meaning, frozen: layout 1's package read it in
# deploy_tree._models_out and from config.NO_MODEL.
NO_MODEL = ("", "off", "none", "claude", "claude-opus", DEFAULT)
# "default" and its older spelling "none", which name no model: layout 1's meaning, frozen: layout 1's package read
# them from bindings.NO_DRAFT_NAMES.
DEFAULT_NAMES = (DEFAULT, "none")
# The model a pass that always runs ran on when the file named none: layout 1's meaning, frozen: layout 1's package
# read it from config.ORCHESTRATOR_DEFAULT.
ALWAYS_RUNS_MODEL = "opus"
# The base effort, a missing claude.effort: layout 1's meaning, frozen: layout 1's package read it from
# config.EFFORT_DEFAULT.
BASE_EFFORT = "high"
# Where `pass_effort` finds a role's effort: its own; the pass in charge's ("default"); the base effort; the role's
# default, where layout 1's chain ended in the model's own effort.
FROM_OWN, FROM_OWNER, FROM_BASE, FROM_ROLE = "own", "owner", "base", "role"
# The binding name that chose the local worker: layout 1's meaning, frozen: layout 1's package read it from
# deploy_tree.WORKER.
WORKER = "worker"
# What a worker binding with a blank timeout ran on: layout 1's meaning, frozen: layout 1's package read it from
# config.WORKER_DEFAULTS (stall_s).
WORKER_TIMEOUT_SECONDS = 900
# The host names of this host's Ollama: layout 1's meaning, frozen: layout 1's package read it from config.LOCAL_HOSTS.
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
# The requests at a time of the binding named worker at this host's Ollama, with a blank, missing or 0
# limits.max_parallel: layout 1's meaning, frozen: layout 1's package read it in config.pipes (one model in memory).
WORKER_PARALLEL_REQUESTS = 1
NO_CAP = 0  # max_parallel_requests: no cap, what an ollama binding with a blank or missing max_parallel ran on
GPU_KEY_SECRET = "gpu_ssh_key"  # the remote GPU tests' key, a secret in layout 2
WEB_HOST_KEY_SECRET = "web_host_ssh_key"  # the web host's SSH key, a secret in layout 2
# The label layout 1's develop guard put beside the first on a red CI: layout 1's meaning, frozen: layout 1's package
# read it from breakage.CI_RED.
CI_RED = "ci-red"
# The develop guard's labels, a missing incident_issue.labels and stability_label: layout 1's meaning, frozen: layout
# 1's package read them from config.defaults.json's incident_issue.
GUARD_LABELS = ("post-merge",)
STABILITY_LABEL = "rocke-stability"
# Why `guard_labels_written` writes a group of the develop guard's labels: they are the file's own; the develop
# guard is in use.
LABELS_OWN, GUARD_IN_USE = "own", "in use"
# The merge queue's auto_merge, and whether it armed auto-merge: layout 1's meaning, frozen: layout 1's package read
# it in features.py, which armed it on "enabled" alone.
AUTO_MERGE = {"enabled": True, "hints_only": False}
GUARD_OWN = {True: "same_as_others", False: "alert_only"}  # layout 2's values
# A blank automerge_guard_own: layout 1's meaning, frozen: layout 1's package read it in merge.automerge_guard_acts,
# where a blank one is false.
GUARD_OWN_BLANK = False
AUTH_KEYS = ("auth_header", "auth_scheme", "auth_mode")
# The header a blank auth_header named, where a bearer token went too: layout 1's meaning, frozen: layout 1's package
# read it in llm_client.auth_headers (worker/llm_client.py).
AUTH_HEADER = "Authorization"
# The auth_mode values layout 1 took; it read any other as "header": layout 1's meaning, frozen: layout 1's package read
# them from config.AUTH_MODES.
AUTH_MODES = ("", "header", "bearer", "both")
# Each Claude role: its name, its layout 1 key, its layout 2 key, and the role in charge of it, "" for a role that
# always runs. The roles in charge: layout 1's meaning, frozen: layout 1's package read them from config.OWNER_ROLE.
ROLES = (
    ("orchestrator", "monitor.orchestrator", f"{RA}.orchestrator", ""),
    ("verify", f"{RA}.passes.verify", f"{RA}.passes.verification_pass", ""),
    ("quality", f"{RA}.passes.quality", f"{RA}.passes.code_quality_pass", ""),
    (
        "first_pass",
        f"{RA}.passes.first_pass",
        f"{RA}.passes.first_review_pass",
        "verify",
    ),
    ("ci_triage", f"{RA}.passes.ci_triage", f"{RA}.passes.ci_triage_pass", "verify"),
    (
        "quality_draft",
        f"{RA}.passes.quality_draft",
        f"{RA}.passes.first_code_quality_pass",
        "quality",
    ),
    (
        "notice_draft",
        f"{RA}.passes.notice_draft",
        f"{RA}.passes.notice_draft_pass",
        "orchestrator",
    ),
)
# The effort of a role that always runs, when the file has none: layout 1's meaning, frozen: layout 1's package read
# it from config.EFFORT_ROLE_DEFAULTS.
ROLE_EFFORTS = {"orchestrator": "high", "verify": "high", "quality": "medium"}
# Each provider's layout 2 name, and its keys: (layout 1 key, layout 2 key).
PROVIDERS = {
    "claude": "claude",
    "ollama": "ollama",
    "openai-compatible": "openai_compatible",
}
# A binding's keys left out, how it sent its key among them (auth_mode, auth_header, auth_scheme): layout 1's meaning,
# frozen: layout 1's package read them from bindings.BINDING_DEFAULTS, and a claude binding's own from
# bindings.PROVIDER_DEFAULTS (anthropic).
BINDING_DEFAULTS = {
    "provider": "openai-compatible",
    "where": "",
    "url": "",
    "chat_path": "/chat/completions",
    "api_key_file": "",
    "api_key_secret": "",
    "api_key_env": "",
    "headers_env": "",
    "auth_header": "Authorization",
    "auth_scheme": "Bearer",
    "auth_mode": "",
    "model": "",
    "name": "",
    "tpm": 0,
    "rpm": 0,
    "concurrency": 0,
    "context": 0,
    "max_output_tokens": 16000,
    "effort": "high",
    "temperature": 0.3,
    "stall_s": 300,
    "stream": True,
    "stream_usage": True,
    "max_resends": 3,
    "resend_wait_s": 5,
}
CLAUDE_BINDING_DEFAULTS = {
    "chat_path": "/v1/messages",
    "auth_header": "x-api-key",
    "auth_scheme": "",
}
BINDING_KEYS = {
    "claude": (("model_id", "model_id"),),
    "ollama": (
        ("model_id", "model_id"),
        ("endpoint", "server_url"),
    ),  # its limits.max_parallel has a rule of its own: `ollama_parallel_requests`
    "openai-compatible": (
        ("model_id", "model_id"),
        ("endpoint", "base_url"),
        ("limits.requests_per_min", "limits.requests_per_minute"),
        ("limits.tokens_per_min", "limits.tokens_per_minute"),
        ("limits.max_parallel", "limits.max_parallel_requests"),
    ),
}
# A model's price fields in layout 1, and their names in layout 2: layout 1's names, frozen: layout 2's deploy
# schema names each in its x-internal.
PRICE_FIELDS = {
    "in": "input",
    "out": "output",
    "cache_read": "cache_read",
    "cache_write_5m": "cache_write_5_minutes",
    "cache_write_1h": "cache_write_1_hour",
}
# Why a field of a model's prices that PRICE_FIELDS does not name is not carried (`model_prices`), as a note says it
# after "not carried, as"; the model's prices are carried.
NOT_A_PRICE = f"a price is one of {wrapping.joined(PRICE_FIELDS)}"
# The human keys of `advanced` that move as they are, and their homes; the others have rules of their own
# below. Layout 1's keys, frozen: layout 2's deploy schema names each in its x-internal.
HUMAN = (
    ("gpu.enabled", f"{RGT}.enabled"),
    ("gpu.login_host", f"{RGT}.cluster.login_host"),
    ("gpu.ssh_port", f"{RGT}.cluster.ssh_port"),
    ("gpu.ssh_user", f"{RGT}.cluster.ssh_user"),
    ("gpu.partition", f"{RGT}.cluster.partition"),
    ("gpu.accounts", f"{RGT}.cluster.accounts"),
    ("gpu.constraint", f"{RGT}.cluster.constraint"),
    ("gpu.time_limit", f"{RGT}.cluster.time_limit"),
    ("gpu.gres", f"{RGT}.cluster.gpu_types"),
    ("gpu.default_arch", f"{RGT}.cluster.default_target"),
    ("gpu.exclude", f"{RGT}.cluster.excluded_nodes"),
    ("gpu.remote_dir", f"{RGT}.node.work_folder"),
    ("gpu.remote_python", f"{RGT}.node.python"),
    ("gpu.setup", f"{RGT}.node.setup_command"),
    ("gpu.clone_url", f"{RGT}.node.clone_url"),
    ("gpu.max_jobs_per_pr", f"{RGT}.jobs_per_review"),
    ("gpu.pending_report_min", f"{RGT}.wait_report_after_minutes"),
    ("gpu.resubmits", f"{RGT}.bad_node_resubmits"),
    ("weekly_report.enabled", "features.weekly_report.enabled"),
    ("incident_issue.repo", f"{GUARD}.issue_repository"),
    ("jira.labels", f"{TC}.labels"),
    ("jira.issue_type", f"{TC}.issue_type"),
    ("jira.extra_fields", f"{TC}.required_fields"),
)
# The web host's human keys of `advanced`, which go to the team dashboard's publish block with delivery "publish"
# alone (check 7). Layout 1's keys, frozen: layout 2's deploy schema names each in its x-internal.
WEB_HOST = (
    ("publish.html.host", "web_host"),
    ("publish.html.user", "username"),
    ("publish.html.site", "site_folder"),
)
# The web host's keys of `advanced` that layout 2 has no key for: `why_dropped` says where each one goes.
WEB_HOST_DROPPED = (
    "publish.html.key",
    "publish.html.known_hosts",
    "publish.html.host_keys",
    "publish.html.dest",
)
# The variable layout 1 gave every remote GPU job, which the repository's defaults set first in layout 2: layout
# 1's meaning, frozen: layout 1's package read it from gpu.DEFAULTS (env).
GPU_BACKEND = ("ROCKE_BACKEND", "python")
# Keys under `advanced` whose value is a map, compared whole with its default: layout 1's meaning, frozen: layout 1's
# package read it from deploy_tree.FREE_FORM.
FREE_FORM = (
    "roles",
    "drafter",
    "prices",
    "clashes",
    "hot_spots",
    "codeowner_teams",
    "gpu.gres",
    "gpu.env",
    "jira.priority_map",
    "jira.extra_fields",
    "jira.accounts",
    "worker.header_env",
    "worker.probe",
    "models.analytics",
    "models.planner",
    "models.analyst",
    "models.draft",
    "models.tick",
)
# Keys under `advanced` merged into a key of layout 2: why a value other than the package's is not carried.
MERGED = {
    "claude.effort": "each Claude role's effort is written as it resolved, and layout 2 has no base effort",
    "worker.think": "a pass's effort is its own key in layout 2",
    "tick_interval_min": "monitor.tick_interval_minutes is the one tick interval in layout 2",
    "loop_log_max_mb": "the log's size limit is the package's in layout 2",
}
PROVENANCE = ("exported_at", "package_version", "source")
# The keys of layout 1's files that layout 2 drops, and why, but host.mounts' and secrets', which name a command
# (`why_dropped`).
DROPPED = {
    "integrations.atlassian_cli.confluence_site": "no feature reads it",
    "setup": "a settings file keeps no record of the interview that wrote it",
    **{
        k: "a settings file keeps no record of the export that wrote it"
        for k in PROVENANCE
    },
    "advanced.update.path": "the package knows its own folder in the repository it comes from",
}
UPDATES = "updates"  # layout 2's section: where rkmon package update fetches newer code
# A GitHub repository as update.repo names it, which the updates section takes as <org>/<repo>: an address on
# github.com, over HTTPS (with a user in it or not) or SSH, with .git or not, or <org>/<repo> alone. An organization's
# name starts with a letter or a digit: a path that starts with /, . or ~ names none.
GITHUB_REPOSITORY = re.compile(
    r"(?:https?://(?:[^@/\s]+@)?github\.com/|ssh://git@github\.com/|git@github\.com:)?"
    r"(?P<org>[A-Za-z0-9][A-Za-z0-9-]*)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?",
    re.IGNORECASE,
)
SHOWN_MAX_CHARACTERS = 60  # how much of a value a note shows, at most


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()  # a key the earlier file does not have


# --- values ----------------------------------------------------------------------------------------------------------


def _typed(value):
    """VALUE as layout 1's reader took it: each plain value by YAML's rules (answers.typed), at any depth."""
    if isinstance(value, answers.Plain):
        return answers.typed(str(value))
    if isinstance(value, dict):
        return {str(k): _typed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_typed(v) for v in value]
    return value


def _blank(value) -> bool:
    """Whether VALUE is missing, null, or text of spaces only."""
    if value is MISSING:
        return True
    value = _typed(value)
    return value is None or (isinstance(value, str) and not value.strip())


def _text(value) -> str:
    """A scalar's text, stripped: a plain value's as written; "" for a blank one."""
    if _blank(value):
        return ""
    return str(value if isinstance(value, answers.Plain) else _typed(value)).strip()


def _same(a, b) -> bool:
    """Whether two typed values are one value: true and 1 differ; 1 and 1.0 do not."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _shown(value) -> str:
    """A value as a note shows it: as JSON, cut to 60 characters, the last of them "…"."""
    text = json.dumps(_typed(value), ensure_ascii=False, separators=(", ", ": "))
    return (
        text
        if len(text) <= SHOWN_MAX_CHARACTERS
        else text[: SHOWN_MAX_CHARACTERS - 1] + "…"
    )


def _get(values, dotted: str):
    cur = values
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def _put(values: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = values
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _pop(values: dict, dotted: str):
    """The value at DOTTED, taken out (MISSING when there is none); a section it leaves empty goes too."""
    parts = dotted.split(".")
    cur = _get(values, ".".join(parts[:-1])) if parts[:-1] else values
    if not isinstance(cur, dict) or parts[-1] not in cur:
        return MISSING
    value = cur.pop(parts[-1])
    if parts[:-1] and not cur:
        _pop(values, ".".join(parts[:-1]))
    return value


def _whole_minutes(value) -> int | None:
    """VALUE as a whole number of minutes from 1; None for any other value."""
    minutes = _typed(value)
    if isinstance(minutes, bool) or not (
        isinstance(minutes, int)
        or (isinstance(minutes, float) and minutes.is_integer())
    ):
        return None
    return int(minutes) if minutes >= 1 else None


def _command(command: str) -> str:
    """A command's name after the program's, as a note names it: "rkmon confirm"."""
    return str(steps.command_line(steps.command_name(command)))


def _layout1_defaults() -> dict:
    """Layout 1's defaults for the keys under `advanced` (DEFAULTS_FILE, its comment left out)."""
    with open(DEFAULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# --- layout 1's rules ------------------------------------------------------------------------------------------------
# What a value of layout 1 meant, beyond one value moved to one key: one function a rule. Each takes values as either
# a layout 1 file or layout 1's configuration has them (a plain value as written, or a typed one; MISSING for a key
# that is not there, where a rule tells the two apart), writes no note and reads no document: the step below writes
# its own notes, and a translation of layout 1's configuration can call the same rules.


def leaves(values: dict, prefix: str = ""):
    """(dotted key, value) for each value in VALUES: a mapping is gone into, but a FREE_FORM one, which is one value,
    compared whole; an empty mapping that is not FREE_FORM has no value."""
    for k, v in values.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict) and dotted not in FREE_FORM:
            yield from leaves(v, dotted + ".")
        else:
            yield dotted, v


def container_name(value):
    """The container's name: VALUE, or CONTAINER_NAME for a blank or missing one, what it meant: the deployment keeps
    its name, so its volumes and its state."""
    return CONTAINER_NAME if _blank(value) else value


def image_name(value):
    """The image's name, in either file: VALUE, or IMAGE_NAME for a blank or missing one, what it meant."""
    return IMAGE_NAME if _blank(value) else value


def image_tag(value, container):
    """The image's tag: VALUE, or for a blank or missing one CONTAINER, the container's name (`container_name`'s), as
    text: what a blank tag meant."""
    return _text(container) if _blank(value) else value


def labels(value) -> list | None:
    """VALUE as a list of labels: each item that is not blank, as text; [] for a blank or missing VALUE; None for a
    value that is no list of labels (no list, or a list with a list or a mapping in it).
    """
    if _blank(value):
        return []
    typed = _typed(value)
    if not isinstance(typed, list) or any(isinstance(x, (dict, list)) for x in typed):
        return None
    return [_text(x) for x in value if not _blank(x)]


def shared_label_folders(names: list, folder) -> dict:
    """repository.path_labels as repository.shared_label_folders: each label of NAMES (`labels`') with FOLDER,
    repository.folder, as its folder, the one folder it counted for."""
    return {name: folder for name in names}


def user_id(value):
    """A user or group id: None for a blank one; a whole number for a whole number or text of digits; VALUE as it is
    otherwise, for layout 2's schema to judge."""
    if _blank(value):
        return None
    typed = _typed(value)
    if isinstance(typed, int) and not isinstance(typed, bool):
        return typed
    if isinstance(typed, str) and re.fullmatch(r"[0-9]+", typed.strip()):
        return int(typed.strip())
    return value


def auto_merge_enabled(value) -> bool | None:
    """The merge queue's auto_merge as auto_merge.enabled: AUTO_MERGE's switch for "enabled" or "hints_only"; None for
    any other value, a blank one too."""
    return AUTO_MERGE.get(_text(value))


def guard_own(value) -> str | None:
    """automerge_guard_own as auto_merge_guard.on_your_pull_requests: GUARD_OWN's word for true or false, a
    blank one being GUARD_OWN_BLANK, as layout 1 read it; None for a value that is no switch.
    """
    on = GUARD_OWN_BLANK if _blank(value) else _typed(value)
    return GUARD_OWN[on] if isinstance(on, bool) else None


def ci_red_labels(names: list) -> tuple:
    """The develop guard's labels on a red CI, as layout 1's develop guard put them on a develop issue, from
    NAMES, its incident_issue.labels (`labels`'), or GUARD_LABELS when there are none: (ci_red_label, the first;
    ci_red_extra_labels, the others, then CI_RED, the label it added beside the first, when NAMES lacks it).
    """
    names = list(names) or list(GUARD_LABELS)
    return names[0], names[1:] + ([] if CI_RED in names else [CI_RED])


def stability_label(value) -> str:
    """The label layout 1's develop guard put on any other develop issue: VALUE, its incident_issue.stability_label, as
    text, or STABILITY_LABEL for a blank or missing one."""
    return _text(value) or STABILITY_LABEL


def guard_labels_written(queue_on: bool, guard_on: bool, ci_red, stability) -> tuple:
    """Whether each group of the develop guard's labels is written, and why: (the labels on a red CI, from
    CI_RED, its incident_issue.labels; the stability label, from STABILITY, its incident_issue.stability_label), each
    LABELS_OWN for labels of the file's own, other than the ones layout 1's develop guard put for none (GUARD_LABELS, as
    `labels` reads the list; STABILITY_LABEL, as `stability_label` reads the label); else GUARD_IN_USE while the develop
    guard is in use, its switch and the merge queue's on (GUARD_ON, QUEUE_ON), as it put those labels; else None: a
    develop guard that is off and keeps its defaults is written as its switch alone."""
    names = labels(ci_red)
    own = (
        bool(names) and not _same(names, list(GUARD_LABELS)),
        stability_label(stability) != STABILITY_LABEL,
    )
    in_use = queue_on and guard_on
    return tuple(
        LABELS_OWN if mine else GUARD_IN_USE if in_use else None for mine in own
    )


def notices(
    queue_on: bool, review_automation_on: bool, queue=None, drift=None
) -> tuple:
    """(queue_notices.enabled, drift_notices.enabled) of the merge queue: QUEUE and DRIFT, the source's own switches,
    when it has them (production's: its queue notices' switch, and its feature map's drift-notices); without
    them, as layout 1 posted them: queue notices while the merge queue is on (QUEUE_ON), and drift notices while
    review automation is on too (REVIEW_AUTOMATION_ON), as Claude writes them."""
    if queue is None:
        queue = queue_on
    if drift is None:
        drift = queue_on and review_automation_on
    return queue, drift


def feature_mode(on: bool) -> str | None:
    """The mode of alerts, ticket creation or the team dashboard, which had none: LIVE for one that
    is on, as it sent whenever it was on; None for one that is off, which is written as any feature that is off, with
    no mode."""
    return LIVE if on else None


def ticket_types(
    section: bool, follow_up: bool | None = None, found_issue: bool | None = None
) -> tuple:
    """(follow_up_tickets.enabled, found_issue_tickets.enabled): each type the section's switch SECTION AND its
    trigger, review_follow_up (FOLLOW_UP) or out_of_scope_issue (FOUND_ISSUE); a missing trigger (None) is
    TRIGGER_DEFAULT, as layout 1 read it. Layout 2 has no switch for the section: ticket creation is on while a type
    is, and with both off it is a feature that is off."""
    if follow_up is None:
        follow_up = TRIGGER_DEFAULT
    if found_issue is None:
        found_issue = TRIGGER_DEFAULT
    return section and follow_up, section and found_issue


def team_delivery(value) -> str:
    """The team dashboard's delivery: VALUE as text, or DELIVERY for a blank or missing one, what it meant."""
    return _text(value) or DELIVERY


def team_block(delivery: str) -> str | None:
    """The one block of the team dashboard that DELIVERY (`team_delivery`'s) takes (check 7), named as the
    delivery is: SERVE_DELIVERY's, the port and the address, as it listens; PUBLISH_DELIVERY's, the web host's keys,
    as it copies the page to a web host; None for any other delivery, which takes neither.
    """
    return delivery if delivery in (PUBLISH_DELIVERY, SERVE_DELIVERY) else None


def alert_switches(on: bool) -> dict:
    """Each alert's switch, by its key under features.alerts: ON, layout 1's one switch for both."""
    return {"develop_breaks": on, "needs_you": on}


def jira_components(value) -> list | None:
    """jira.component as ticket_creation.components: its text split at commas, each part stripped, a blank part
    left out; None for a value that is no text."""
    if not isinstance(_typed(value), str):
        return None
    return [part.strip() for part in _text(value).split(",") if part.strip()]


def same_address(a, b) -> bool:
    """Whether A and B, two addresses as text, name one place: the same but for a trailing "/"."""
    return _text(a).rstrip("/") == _text(b).rstrip("/")


def jira_address(value):
    """A JIRA site as an address, the form integrations.atlassian_cli.jira_url takes: a bare site name
    (SITE_NAME), as the atlassian CLI names a cloud site, is the address it stands for (CLOUD_SITE), whether it is
    compared with another (`same_address`) or carried alone; any other value is as it is.
    """
    text = _text(value)
    return CLOUD_SITE.format(name=text) if SITE_NAME.fullmatch(text) else value


def priority_requests(value) -> bool:
    """publish.html.requests as priority_requests.enabled: true while it names a requests file, whose place
    layout 2 works out from the site folder. False, 0, null and blank text name none, as layout 1's package read them
    (publish.takes_requests)."""
    return not _blank(value) and bool(_typed(value))


def model_prices(fields: dict) -> tuple:
    """A model's prices in layout 2, from FIELDS, its prices in layout 1: (each field that PRICE_FIELDS names,
    under its name in layout 2; each other field, by its name, which is no price)."""
    kept, other = {}, {}
    for field, value in fields.items():
        if str(field) in PRICE_FIELDS:
            kept[PRICE_FIELDS[str(field)]] = value
        else:
            other[str(field)] = value
    return kept, other


def spend_enabled(prices: dict) -> bool:
    """Whether review spend is on, by spend.on's rule (on while there are prices): PRICES, by model id, which layout 1
    counted each cost by."""
    from . import spend

    return spend.on({"prices": prices})


def gpu_environment(env: dict) -> dict:
    """The remote GPU jobs' environment in layout 2: ENV without GPU_BACKEND, which the repository's defaults
    set first."""
    return {k: v for k, v in env.items() if (str(k), _text(v)) != GPU_BACKEND}


def base_effort(value=MISSING) -> str:
    """claude.effort as layout 1's chain read it: VALUE as text, lowercased ("" for a blank one: the model's own
    effort), or BASE_EFFORT when the file has none (MISSING)."""
    return BASE_EFFORT if value is MISSING else _text(value).lower()


def pass_effort(role: str, own=MISSING, base: str = BASE_EFFORT) -> tuple:
    """The effort of the Claude role ROLE (ROLES' first column) as layout 1's chain resolved it, in layout 2: (the
    effort, where it came from). OWN, the role's own (a missing one: ROLE_EFFORTS' for a role that always runs), when
    it is one of EFFORTS: FROM_OWN. Else, on an optional pass, DEFAULT, the effort of the pass in charge, as layout 1
    read it: FROM_OWNER. Else BASE, the base effort (`base_effort`'s), when it is one: FROM_BASE. Else the chain ended
    in the model's own effort, which layout 2 cannot name for a pass that always runs: ROLE_EFFORTS', FROM_ROLE.
    """
    owner = {r[0]: r[3] for r in ROLES}[role]
    text = _text(ROLE_EFFORTS.get(role, "") if own is MISSING else own).lower()
    if text in EFFORTS:
        return text, FROM_OWN
    if owner:
        return DEFAULT, FROM_OWNER
    if base in EFFORTS:
        return base, FROM_BASE
    return ROLE_EFFORTS[role], FROM_ROLE


def pass_model(value, optional: bool, worker_bound: bool = False):
    """A pass's model in layout 2, from its model VALUE. On an optional pass (OPTIONAL): DEFAULT for one of
    NO_MODEL, a blank one too (the pass in charge does the work); WORKER for "worker" while the file binds a model
    of that name (WORKER_BOUND), else DEFAULT, as the local worker that the name chose leaves the package. On a pass
    that always runs, which takes no DEFAULT: ALWAYS_RUNS_MODEL for one of DEFAULT_NAMES, the model layout 1 ran the
    pass on when the file named none; None for a blank one, which is left out. Any other VALUE as it is.
    """
    text = _text(value)
    if not optional:
        if text in DEFAULT_NAMES:
            return ALWAYS_RUNS_MODEL
        return value if text else None
    if text in NO_MODEL:
        return DEFAULT
    if text == WORKER:
        return WORKER if worker_bound else DEFAULT
    return value


def api_key_secret_name(value):
    """A binding's api_key_secret as its api_key_secret_name, the name the one-time move copies the secret to:
    the one secrets_registry.binding_key_name gives the secret's name, when that is another (the first review pass's
    for the local worker's key; else a name that ends as every binding's key's does, _api_key, by which the ignore
    lists keep a key's file out of git and the image); else VALUE."""
    text = _text(value)
    name = secrets_registry.binding_key_name(text)
    return name if name != text else value


def binding_defaults(provider: str) -> dict:
    """Layout 1's values for the keys a binding of PROVIDER (layout 1's name) leaves out."""
    return {
        **BINDING_DEFAULTS,
        **(CLAUDE_BINDING_DEFAULTS if provider == "claude" else {}),
    }


def key_sending(mode, header, scheme) -> dict:
    """How layout 1 sent a binding's key, from its auth_mode MODE, auth_header HEADER and auth_scheme SCHEME (a missing
    one: `binding_defaults`'), as layout 2's api_key_sent_as, api_key_header and api_key_scheme
    (llm_client.auth_headers): mode "bearer", as a bearer token alone; "both", in HEADER and as a bearer token
    ("header_and_bearer"), but in HEADER alone ("header") when HEADER is AUTH_HEADER, which got it once; any other
    mode, one of AUTH_MODES or one layout 1 read as "header", in HEADER ("header"). A blank HEADER is AUTH_HEADER; a
    blank SCHEME is left out."""
    mode = _text(mode).lower()
    if mode == "bearer":
        return {"api_key_sent_as": "bearer"}
    header = _text(header) or AUTH_HEADER
    both = mode == "both" and header.lower() != AUTH_HEADER.lower()
    out = {
        "api_key_sent_as": "header_and_bearer" if both else "header",
        "api_key_header": header,
    }
    if _text(scheme):
        out["api_key_scheme"] = _text(scheme)
    return out


def stall_timeout_seconds(name: str, minutes) -> int | None:
    """The stall_timeout_seconds of the binding NAME, from its limits.timeout_min MINUTES: a whole number of
    minutes from 1, in seconds; else, on the binding named WORKER, WORKER_TIMEOUT_SECONDS, what it ran on with none;
    else None, which is left out."""
    whole = _whole_minutes(minutes)
    if whole is not None:
        return whole * 60
    return WORKER_TIMEOUT_SECONDS if name == WORKER else None


def on_this_host(url) -> bool:
    """Whether the address URL names this host (LOCAL_HOSTS), as layout 1 told this host's Ollama from another."""
    try:
        return (urllib.parse.urlsplit(_text(url)).hostname or "") in LOCAL_HOSTS
    except ValueError:
        return False


def ollama_parallel_requests(name: str, server_url, value):
    """The limits.max_parallel_requests of the ollama binding NAME, from its limits.max_parallel VALUE (MISSING
    when it has none) and its address SERVER_URL: WORKER_PARALLEL_REQUESTS on the binding named WORKER at this host's
    Ollama (`on_this_host`) for a blank, missing or 0 one, as layout 1 ran it one request at a time; NO_CAP for a
    blank or missing one on any other, what it ran on; any other VALUE as it is."""
    if name == WORKER and on_this_host(server_url):
        if _blank(value) or _same(_typed(value), 0):
            return WORKER_PARALLEL_REQUESTS
    return NO_CAP if _blank(value) else value


def why_dropped(key: str) -> str:
    """Why layout 2 drops KEY, a key of a layout 1 file that has no home, as a note says it after "dropped, as";
    "" for any other key."""
    if key == "host.mounts":
        return (
            f"the remote GPU tests' key is the secret {GPU_KEY_SECRET}, installed with: {program.name()} secret "
            f"{GPU_KEY_SECRET} FILE"
        )
    if key == "secrets":
        return f"{settings_file.NO_SECRET}; {Steps(settings_file.secret_steps())}"
    web_host_secret = f"{program.name()} secret {WEB_HOST_KEY_SECRET} FILE"
    if key == "advanced.publish.html.key":
        return f"the web host's SSH key is the secret {WEB_HOST_KEY_SECRET}, installed with: {web_host_secret}"
    if key in ("advanced.publish.html.known_hosts", "advanced.publish.html.host_keys"):
        return f"the monitor keeps the web host's own key in its state: {host_keys.record_fix(program.name())}"
    if key == "advanced.publish.html.dest":
        return (
            f"the page's place is worked out from {TEAM}.{PUBLISH_DELIVERY}.site_folder"
        )
    return DROPPED.get(key, "")


def admin_address_allowed(value) -> bool:
    """Whether the admin dashboard's address VALUE is one layout 2 has: blank, or container_layout.LOOPBACK, where it
    listens alone."""
    return _blank(value) or _text(value) == container_layout.LOOPBACK


def github_repository(value) -> str:
    """The <org>/<repo> that VALUE, update.repo, names as a GitHub repository (GITHUB_REPOSITORY): an address on
    github.com, or <org>/<repo> itself; "" for any other value: a blank or missing one, a path, a remote's name, an
    address on another host."""
    m = GITHUB_REPOSITORY.fullmatch(_text(value))
    if not m or not m["repo"].strip("."):
        return ""
    return f"{m['org']}/{m['repo']}"


def branch_name(value) -> str:
    """update.branch as the updates section takes it: the branch's name, as text; "" for a blank or missing one, and
    for a list or a map, which names no branch."""
    return "" if isinstance(_typed(value), (dict, list)) else _text(value)


def updates_section(repo, branch) -> dict:
    """update.repo and update.branch as layout 2's updates section: {"repository": the <org>/<repo>
    `github_repository` gives for REPO, "branch": `branch_name`'s for BRANCH}; {} when either of them is "", as the
    section is then left out: a blank or missing value, or a repository that is no GitHub one (a path, a remote's
    name)."""
    repository, name = github_repository(repo), branch_name(branch)
    return {"repository": repository, "branch": name} if repository and name else {}


# --- one step --------------------------------------------------------------------------------------------------------


class _Step:
    """The earlier file's values, each taken once, and what the step makes of them: the values in layout 2 and the
    notes. What is left at the end is what no rule knows."""

    def __init__(self, values: dict, file_type: settings_file.FileType):
        self.earlier = copy.deepcopy(dict(values or {}))
        self.file_type = file_type  # layout 2's, which upgrade gives the step
        self.out = {}
        self.notes = []
        self.same = 0  # values under advanced equal to the package's defaults

    def placeholder(self, dotted: str) -> str:
        """Layout 2's template value of the key DOTTED: the placeholder a person replaces."""
        for path, node, _required in settings_file.schema_keys(self.file_type.load()):
            if path == dotted:
                return settings_file.template_value(node, path)
        raise KeyError(dotted)

    def peek(self, dotted: str):
        return _get(self.earlier, dotted)

    def take(self, dotted: str):
        return _pop(self.earlier, dotted)

    def put(self, dotted: str, value) -> None:
        _put(self.out, dotted, value)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def move(self, old: str, new: str):
        """OLD's value at NEW, unless it is blank; the earlier value, MISSING when there is none."""
        value = self.take(old)
        if not _blank(value):
            self.put(new, value)
            if old != new:
                self.note(settings_file.Moved(old, new))
        return value

    def switch(self, old: str, default: bool) -> bool:
        """A switch's meaning in layout 1: its value, else DEFAULT; a value that is no switch is reported."""
        value = self.take(old)
        on = _typed(value)
        if isinstance(on, bool):
            return on
        if not _blank(value):
            self.note(
                f"{old} is {_shown(value)}: not carried, as a switch is true or false; it meant "
                f"{json.dumps(default)}, its default"
            )
        return default

    def move_image_name(self, old: str) -> None:
        """The image's name at image.name; a blank or missing one is `image_name`'s, what it meant, with no note."""
        value = self.move(old, "image.name")
        if _blank(value):
            self.put("image.name", image_name(value))

    def provenance(self) -> None:
        taken = [k for k in PROVENANCE if self.take(k) is not MISSING]
        if taken:
            self.note(f"{wrapping.joined(taken)}: dropped, as {why_dropped(taken[0])}")

    def leftovers(self) -> None:
        """A note for each value no rule took, but a blank one, which has nothing to carry."""
        for dotted, value in leaves(self.earlier):
            if _blank(value):
                continue
            self.note(
                f"{dotted} {_shown(value)}: not carried, as it is no setting of layout 1"
            )


# --- deploy settings -------------------------------------------------------------------------------------------------


class _Deploy(_Step):
    def run(self) -> None:
        self.defaults = _layout1_defaults()
        secret_values = self.take("secret_values")
        self.container()
        self.repository()
        self.host()
        self.monitor()
        self.review_automation()
        self.merge_queue()
        self.dashboards()
        self.alerts()
        self.tickets()
        self.integrations()
        self.bindings()
        self.human()
        self.updates()
        self.advanced()
        self.other_parts()
        self.leftovers()
        if self.same:
            self.note(
                f"advanced: {wrapping.counted(self.same, 'value')} the same as the package's defaults: dropped"
            )
        if secret_values is not MISSING:
            self.out["secret_values"] = secret_values  # upgrade refuses it

    # -- the parts in the earlier file's order

    def container(self) -> None:
        value = self.take("container.name")
        name = container_name(value)
        if _blank(value):
            self.note(
                f"container.name: {json.dumps(name)}, what a blank name meant: the deployment keeps its name, "
                "so its volumes and its state"
            )
        self.put("container.name", name)
        self.move_image_name("container.image.name")
        self.move("container.image.pull_registry", "image.registry")
        value = self.move("container.image.tag", "image.tag")
        if _blank(value):
            tag = image_tag(value, name)
            self.put("image.tag", tag)
            self.note(
                f"image.tag: {json.dumps(tag)}, the container's name, what a blank tag meant; "
                f"{_command(steps.DEPLOY_STEPS[2].command)} says whether that image suits this package"
            )

    def repository(self) -> None:
        self.move("repository.name", "repository.name")
        folder = self.move("repository.path", "repository.folder")
        self.move("repository.scope_labels", "repository.scope_labels")
        value = self.take("repository.path_labels")
        names = labels(value)
        if names is None:
            self.note(
                f"repository.path_labels is {_shown(value)}: not carried, as it is a list of labels"
            )
            return
        if names:
            where = _text(folder) or self.placeholder("repository.folder")
            self.put(
                "repository.shared_label_folders", shared_label_folders(names, where)
            )
            self.note(
                "repository.path_labels is now repository.shared_label_folders: each label with repository.folder "
                f"({json.dumps(where)}) as its folder, the one folder it counted for"
            )

    def host(self) -> None:
        self.move("host.resources.cpus", "host.resources.cpus")
        self.move("host.resources.memory_gb", "host.resources.memory_gib")
        self.move("host.resources.ramdisk_gb", "host.resources.ramdisk_gib")
        for k in ("monitor_uid", "monitor_gid"):
            value = self.take(f"host.users.{k}")
            if value is not MISSING:
                self.put(f"host.users.{k}", user_id(value))
        for k, user in (
            ("github_writer_uid", container_layout.WRITER),
            ("pr_runner_uid", container_layout.PR_RUNNER),
        ):
            value = self.take(f"host.users.{k}")
            if not _blank(value):
                self.note(
                    f"host.users.{k} {_shown(value)}: dropped, as that user's id is the package's ({user.uid})"
                )
        mounts = self.take("host.mounts")
        if not _blank(mounts) and mounts != []:
            n = len(mounts) if isinstance(mounts, (list, dict)) else 1
            self.note(
                f"host.mounts ({wrapping.counted(n, 'entry', 'entries')}): dropped, as {why_dropped('host.mounts')}"
            )

    def monitor(self) -> None:
        self.move("monitor.tick_min", "monitor.tick_interval_minutes")
        self.move("monitor.full_tick_min", "monitor.full_tick_max_interval_minutes")

    def review_automation(self) -> None:
        self.ra_on = self.switch(f"{RA}.enabled", True)
        self.put(f"{RA}.enabled", self.ra_on)
        self.move(f"{RA}.mode", f"{RA}.mode")
        self.move(
            f"{RA}.schedule.push_cooldown_min", f"{RA}.schedule.push_cooldown_minutes"
        )
        self.move(
            f"{RA}.schedule.follow_up_cooldown_min",
            f"{RA}.schedule.follow_up_cooldown_minutes",
        )
        self.move(f"{RA}.capacity.parallel_reviews", f"{RA}.parallel_reviews")
        self.move(f"{RA}.capacity.parallel_heavy_checks", "monitor.parallel_builds")
        self.move(
            f"{RA}.large_prs.manual_review_over_lines",
            f"{RA}.large_prs.in_depth_review_over_lines",
        )
        self.move(
            f"{RA}.large_prs.split_part_lines", f"{RA}.large_prs.split_part_lines"
        )
        worker = self.peek(f"model_bindings.{WORKER}")
        self.worker_bound = isinstance(worker, dict) and (
            _text(worker.get("provider", MISSING)) in PROVIDERS
        )
        self.worker_named = False  # an optional pass keeps the binding named worker
        own = self.peek(
            "advanced.claude.effort"
        )  # the base effort, as layout 1 resolved it, and where it came from
        self.base = (
            base_effort(own),
            (
                "the package's base effort"
                if own is MISSING
                else "advanced.claude.effort"
            ),
        )
        for role, old, new, owner in ROLES:
            model, effort = self.peek(f"{old}.model"), self.peek(f"{old}.effort")
            if not (model is MISSING and effort is MISSING):
                self.note(settings_file.Moved(old, new))
            self.model(old, new, bool(owner))
            self.effort(role, old, new, bool(owner))

    def model(self, old: str, new: str, optional: bool) -> None:
        value = self.take(f"{old}.model")
        if value is MISSING:
            return
        model = pass_model(value, optional, self.worker_bound)
        if model is None:
            return
        self.put(f"{new}.model", model)
        text = _text(value)
        if not optional:
            if text in DEFAULT_NAMES:
                self.note(
                    f"{old}.model {_shown(value)} is now {json.dumps(model)}, the model layout 1 ran it on when the "
                    "file named none: it always runs, so it takes a model of its own"
                )
            return
        if text == WORKER and model == WORKER:
            self.worker_named = True
            self.note(
                f'{new}.model: "worker", the file\'s binding of that name, an ordinary binding now, through which the '
                "pass drafts in steps; the local worker that the name chose leaves the package"
            )
        elif text == WORKER:
            section = steps.command_line(
                self.file_type.section_command, KEY="model_bindings"
            )
            self.note(
                f'{old}.model "worker" is now "default", as the file binds no model named worker: the local worker '
                f"that the name chose leaves the package. To have the pass draft, bind a model ({section} prints the "
                f"forms) and name it in {new}.model"
            )
        elif model == DEFAULT and text not in (DEFAULT, ""):
            self.note(
                f'{old}.model {_shown(value)} is now "default": the pass in charge does the work'
            )

    def effort(self, role: str, old: str, new: str, optional: bool) -> None:
        value = self.take(f"{old}.effort")
        if optional and value is MISSING:
            return
        base, source = self.base
        effort, found = pass_effort(role, value, base)
        self.put(f"{new}.effort", effort)
        own = _text(value).lower()
        if found == FROM_OWNER and own not in (DEFAULT, ""):
            self.note(
                f'{old}.effort {_shown(value)} is now "default", as layout 1 read it: the effort of the pass in '
                "charge"
            )
        elif found == FROM_BASE and own:
            self.note(
                f"{old}.effort {_shown(value)} is now {new}.effort {json.dumps(base)}, from {source}, as layout 1 "
                "took no such effort"
            )
        elif found == FROM_ROLE:
            self.note(
                f"{new}.effort: {json.dumps(effort)}, its default: the pass ran on the model's own effort, as "
                f"neither {old}.effort nor {source} named one, and layout 2 names one for a pass that always runs"
            )

    def merge_queue(self) -> None:
        queue_on = self.switch(f"{MQ}.enabled", True)
        self.put(f"{MQ}.enabled", queue_on)
        self.move(f"{MQ}.mode", f"{MQ}.mode")
        value = self.take(f"{MQ}.auto_merge")
        enabled = auto_merge_enabled(value)
        if enabled is not None:
            self.put(f"{MQ}.auto_merge.enabled", enabled)
            self.note(
                f"{MQ}.auto_merge {json.dumps(_text(value))} is now {MQ}.auto_merge.enabled "
                f"{json.dumps(enabled)}"
            )
        elif not _blank(value):
            self.note(
                f'{MQ}.auto_merge is {_shown(value)}: not carried, as layout 1 took "enabled" or "hints_only"'
            )
        self.move(f"{MQ}.auto_hold", f"{MQ}.auto_hold.enabled")
        guard = self.peek(f"{MQ}.develop_guard")
        guard_on = self.switch(f"{MQ}.develop_guard", True)
        if not _blank(guard):
            self.put(f"{GUARD}.enabled", guard_on)
            self.note(f"{MQ}.develop_guard is now {GUARD}.enabled")
        if queue_on:
            queue, drift = notices(queue_on, self.ra_on)
            self.put(f"{MQ}.queue_notices.enabled", queue)
            self.put(f"{MQ}.drift_notices.enabled", drift)
            if not drift:
                self.note(
                    f"{MQ}.drift_notices.enabled: false, as Claude writes them and review automation is off"
                )
        self.guard_labels(queue_on, guard_on)
        value = self.take("advanced.automerge_guard_own")
        if value is not MISSING:
            key = f"{MQ}.auto_merge_guard.on_your_pull_requests"
            word = guard_own(value)
            if word is not None:
                self.put(key, word)
                self.note(
                    f"advanced.automerge_guard_own {_shown(value)} is now {key} {json.dumps(word)}"
                    + (
                        f", as layout 1 read a blank one as {json.dumps(GUARD_OWN_BLANK)}"
                        if _blank(value)
                        else ""
                    )
                )
            else:
                self.note(
                    f"advanced.automerge_guard_own is {_shown(value)}: not carried, as a switch is true or false"
                )

    def guard_labels(self, queue_on: bool, guard_on: bool) -> None:
        """The develop guard's labels, as layout 1's develop guard used them: each group written while the
        develop guard is in use, or when it is the file's own (`guard_labels_written`).
        """
        value = self.take("advanced.incident_issue.labels")
        stability = self.take("advanced.incident_issue.stability_label")
        ci_red_why, stability_why = guard_labels_written(
            queue_on, guard_on, value, stability
        )
        names = labels(value)
        if names is None:
            self.note(
                f"advanced.incident_issue.labels is {_shown(value)}: not carried, as it is a list of labels"
            )
        own = ci_red_why == LABELS_OWN
        if not _blank(value) and names is not None and not own:
            self.same += 1
        if ci_red_why:
            first, extra = ci_red_labels(names or [])
            self.put(f"{GUARD}.ci_red_label", first)
            self.put(f"{GUARD}.ci_red_extra_labels", extra)
            written = f"{GUARD}.ci_red_label {json.dumps(first)} and {GUARD}.ci_red_extra_labels {json.dumps(extra)}"
            companion = (
                f"; {json.dumps(CI_RED)} is the label layout 1's develop guard added beside the first, when the "
                "repository had it"
                if CI_RED not in (names or GUARD_LABELS)
                else ""
            )
            if own:
                self.note(f"advanced.incident_issue.labels is now {written}{companion}")
            else:
                self.note(
                    f"{written}: the labels layout 1's develop guard put on a develop issue while CI was red{companion}"
                )
        if not _blank(stability) and stability_why != LABELS_OWN:
            self.same += 1
        if stability_why == LABELS_OWN:
            self.put(f"{GUARD}.stability_label", stability)
            self.note(
                settings_file.Moved(
                    "advanced.incident_issue.stability_label",
                    f"{GUARD}.stability_label",
                )
            )
        elif stability_why == GUARD_IN_USE:
            label = stability_label(stability)
            self.put(f"{GUARD}.stability_label", label)
            self.note(
                f"{GUARD}.stability_label {json.dumps(label)}: the label layout 1's develop guard put on any other "
                "develop issue"
            )

    def dashboards(self) -> None:
        self.move(f"{ADMIN}.enabled", f"{ADMIN}.enabled")
        self.move(f"{ADMIN}.port", f"{ADMIN}.port")
        bind, here = self.take(f"{ADMIN}.bind"), container_layout.LOOPBACK
        if not admin_address_allowed(bind):
            key = f"{ADMIN}.bind"
            raise StepRefused(
                key,
                f"{key} is {json.dumps(_text(bind))}, and the admin dashboard listens on this host's {here} only",
                lambda where: Steps(
                    (
                        Edit(key, where, DELETE),
                        ";",
                        'README.md, "The admin dashboard", says how to reach it from another machine',
                    )
                ),
            )
        if not _blank(bind):
            self.note(
                f"{ADMIN}.bind {json.dumps(here)}: dropped, as the admin dashboard listens on this host's "
                f"{here} only"
            )
        team_on = self.switch(f"{TEAM}.enabled", False)
        self.put(f"{TEAM}.enabled", team_on)
        self.delivery = team_delivery(self.move(f"{TEAM}.delivery", f"{TEAM}.delivery"))
        self.block = team_block(
            self.delivery
        )  # the web host's keys of advanced go with it (human)
        for old, new in (("port", "port"), ("bind", "address")):
            value = self.take(f"{TEAM}.{old}")
            if _blank(value):
                continue
            if self.block == SERVE_DELIVERY:
                self.put(f"{TEAM}.{SERVE_DELIVERY}.{new}", value)
                self.note(
                    settings_file.Moved(
                        f"{TEAM}.{old}", f"{TEAM}.{SERVE_DELIVERY}.{new}"
                    )
                )
            else:
                self.note(
                    f"{TEAM}.{old} {_shown(value)}: dropped, as the team dashboard's delivery is "
                    f"{json.dumps(self.delivery)}, and only {json.dumps(SERVE_DELIVERY)} listens"
                )
        self.mode(TEAM, team_on, "the team dashboard is")

    def mode(self, section: str, on: bool, what: str) -> None:
        """A feature's mode, from whether the file has it on: `feature_mode`'s for one that is on;
        one that is off is written as any feature that is off, and no note names a mode for it.
        """
        mode = feature_mode(on)
        if mode:
            self.put(f"{section}.mode", mode)
            self.note(
                f"{section}.mode: {json.dumps(mode)}, as {what} on in the file, and layout 1 sent whenever it was on"
            )

    def alerts(self) -> None:
        present = self.peek(f"{ALERTS}.enabled") is not MISSING
        on = self.switch(f"{ALERTS}.enabled", False)
        switches = alert_switches(on)
        for kind, kind_on in switches.items():
            self.put(f"{ALERTS}.{kind}.enabled", kind_on)
        if present:
            self.note(
                f"{ALERTS}.enabled is now "
                + wrapping.joined(f"{ALERTS}.{kind}.enabled" for kind in switches)
                + f", each {json.dumps(on)}"
            )
        self.move(f"{ALERTS}.teams_channel", f"{ALERTS}.message_label")
        self.move(
            f"{ALERTS}.message_cooldown_min", f"{ALERTS}.message_cooldown_minutes"
        )
        self.mode(ALERTS, on, "alerts are")

    def tickets(self) -> None:
        section, follow_up, found_issue = (
            f"{TC}.enabled",
            f"{TC}.triggers.review_follow_up",
            f"{TC}.triggers.out_of_scope_issue",
        )
        present = any(
            self.peek(k) is not MISSING for k in (section, follow_up, found_issue)
        )
        follow_up_on, found_issue_on = ticket_types(
            self.switch(section, False),
            self.switch(follow_up, TRIGGER_DEFAULT),
            self.switch(found_issue, TRIGGER_DEFAULT),
        )
        self.put(f"{TC}.follow_up_tickets.enabled", follow_up_on)
        self.put(f"{TC}.found_issue_tickets.enabled", found_issue_on)
        if present:
            self.note(
                f"{section} and its triggers are now {TC}.follow_up_tickets.enabled {json.dumps(follow_up_on)} and "
                f"{TC}.found_issue_tickets.enabled {json.dumps(found_issue_on)}: each the section's switch and the "
                "type's trigger"
            )
        self.move(f"{TC}.jira_project", f"{TC}.project_key")
        self.move(f"{TC}.default_epic", f"{TC}.epic_key")
        self.mode(TC, follow_up_on or found_issue_on, "ticket creation is")

    def integrations(self) -> None:
        self.move("integrations.github_cli.account", "integrations.github_cli.username")
        # the atlassian CLI's site, a name or an address: the address it stands for (`jira_address`)
        key = "integrations.atlassian_cli.jira_site"
        self.jira_site = self.take(key)
        if not _blank(self.jira_site):
            self.carry(
                key, self.jira_site, "integrations.atlassian_cli.jira_url", jira_address
            )
        key = "integrations.atlassian_cli.confluence_site"
        site = self.take(key)
        if not _blank(site):
            self.note(f"{key} {_shown(site)}: dropped, as {why_dropped(key)}")
        gateway = self.move(
            "integrations.claude_code.gateway", "integrations.claude_code.base_url"
        )
        if not _blank(gateway):
            self.note(
                "integrations.claude_code.api_key_sent_as, .api_key_header and .api_key_scheme take their defaults, "
                "the key in the x-api-key header: set them as your gateway takes the key"
            )

    def bindings(self) -> None:
        """model_bindings in layout 2, with each binding's auth_* from advanced."""
        found = self.take("model_bindings")
        earlier = found if isinstance(found, dict) else {}
        extra = self.take("advanced.model_bindings")
        extra = extra if isinstance(extra, dict) else {}
        worker_auth = {k: self.take(f"advanced.{WORKER}.{k}") for k in AUTH_KEYS}
        if not isinstance(found, dict) and not _blank(found):
            self.note(
                f"model_bindings is {_shown(found)}: not carried, as it maps names to bindings"
            )
        out = {}
        for name, binding in earlier.items():
            carried = self.binding(str(name), binding)
            if carried is not None:
                out[str(name)] = carried
        if out:
            self.put("model_bindings", out)
        self.auth(
            WORKER,
            out.get(WORKER),
            worker_auth,
            f"advanced.{WORKER}",
            self.defaults.get(WORKER) or {},
        )
        for name, keys in extra.items():
            name, prefix = str(name), f"advanced.model_bindings.{name}"
            binding = earlier.get(name)
            if not isinstance(binding, dict) or not isinstance(keys, dict):
                self.note(
                    f"{prefix} {_shown(keys)}: not carried, as the file has no binding named {name} for it"
                )
                continue
            defaults = binding_defaults(_text(binding.get("provider", MISSING)))
            auth = {k: keys.pop(k, MISSING) for k in AUTH_KEYS}
            self.auth(name, out.get(name), auth, prefix, defaults)
            for k, value in keys.items():
                self.compare(f"{prefix}.{k}", value, defaults.get(k, MISSING))

    def binding(self, name: str, binding) -> dict | None:
        key = f"model_bindings.{name}"
        if not isinstance(binding, dict):
            self.note(
                f"{key} is {_shown(binding)}: not carried, as a binding is a section"
            )
            return None
        binding = copy.deepcopy(binding)
        provider = _text(_pop(binding, "provider"))
        if provider not in PROVIDERS:
            said = json.dumps(provider) if provider else "missing"
            self.note(
                f"{key} {_shown(binding)}: not carried, as its provider is {said}, and layout 1's providers are "
                "claude, ollama and openai-compatible"
            )
            return None
        out = {"provider": PROVIDERS[provider]}
        if PROVIDERS[provider] != provider:
            self.note(
                f"{key}.provider {json.dumps(provider)} is now {json.dumps(PROVIDERS[provider])}"
            )
        for old, new in BINDING_KEYS[provider]:
            value = _pop(binding, old)
            if not _blank(value):
                _put(out, new, value)
                if old != new:
                    self.note(settings_file.Moved(f"{key}.{old}", f"{key}.{new}"))
        if provider == "openai-compatible":
            secret = _pop(binding, "api_key_secret")
            if not _blank(secret):
                secret_name = api_key_secret_name(secret)
                out["api_key_secret_name"] = secret_name
                from_key, to_key = f"{key}.api_key_secret", f"{key}.api_key_secret_name"
                text = ""
                if _text(secret_name) != _text(secret):
                    text = (
                        f"{from_key} {json.dumps(_text(secret))} is now {to_key} {json.dumps(secret_name)}, the name "
                        "the one-time move copies that secret to"
                    )
                self.note(settings_file.Moved(from_key, to_key, text))
        if provider == "ollama":
            self.parallel(name, key, binding, out)
        if provider != "claude":
            self.timeout(name, key, binding, out)
        if name == WORKER and not self.worker_named:
            self.note(
                f"{key}: an ordinary binding now; the local worker that the name chose leaves the package"
            )
        for dotted, value in leaves(binding):
            self.note(
                f"{key}.{dotted} {_shown(value)}: not carried, as a binding of provider {json.dumps(provider)} "
                "has no such key"
            )
        return out

    def parallel(self, name: str, key: str, binding: dict, out: dict) -> None:
        """An ollama binding's limits.max_parallel as limits.max_parallel_requests: `ollama_parallel_requests`,
        each case reported."""
        value = _pop(binding, "limits.max_parallel")
        requests = ollama_parallel_requests(name, out.get("server_url"), value)
        _put(out, "limits.max_parallel_requests", requests)
        old, new = f"{key}.limits.max_parallel", f"{key}.limits.max_parallel_requests"
        if not _blank(value) and _same(_typed(requests), _typed(value)):
            self.note(settings_file.Moved(old, new))
            return
        had = (
            "no limits.max_parallel"
            if _blank(value)
            else f"limits.max_parallel {_shown(value)}"
        )
        if requests == WORKER_PARALLEL_REQUESTS:
            self.note(
                f"{new}: {requests}, what the binding named {WORKER} at this host's Ollama ran on with {had}: one "
                "request at a time"
            )
        else:
            self.note(f"{new}: {requests}, what the binding ran on with {had}: no cap")

    def timeout(self, name: str, key: str, binding: dict, out: dict) -> None:
        """limits.timeout_min, in minutes, as stall_timeout_seconds; a blank one on the worker binding, 900."""
        value = _pop(binding, "limits.timeout_min")
        seconds = stall_timeout_seconds(name, value)
        if _whole_minutes(value) is not None:
            out["stall_timeout_seconds"] = seconds
            self.note(
                f"{key}.limits.timeout_min {_shown(value)} is now {key}.stall_timeout_seconds {seconds}"
            )
            return
        if not _blank(value):
            self.note(
                f"{key}.limits.timeout_min is {_shown(value)}: not carried, as it is a whole number of minutes "
                "from 1"
            )
        if seconds is not None:
            out["stall_timeout_seconds"] = seconds
            self.note(
                f"{key}.stall_timeout_seconds: {seconds}, what the binding named {WORKER} ran on with no timeout"
            )

    def auth(
        self, name: str, binding, found: dict, prefix: str, defaults: dict
    ) -> None:
        """How layout 1 sent the key of binding NAME, from FOUND's auth_header, auth_scheme and auth_mode (a missing
        one: DEFAULTS'), as its api_key_sent_as, api_key_header and api_key_scheme (`key_sending`).
        """
        given = {k: v for k, v in found.items() if v is not MISSING}
        own = {
            k: v
            for k, v in given.items()
            if not _same(_typed(v), defaults.get(k, MISSING))
        }
        self.same += len(given) - len(own)
        if not own:
            return
        if (
            not isinstance(binding, dict)
            or binding.get("provider") != PROVIDERS["openai-compatible"]
        ):
            for k, value in own.items():
                self.note(
                    f"{prefix}.{k} is {_shown(value)}: not carried, as the file has no openai_compatible binding "
                    f"named {name} to send a key"
                )
            return

        def effective(k: str):
            return given[k] if k in given else defaults.get(k, "")

        mode = _text(effective("auth_mode")).lower()
        written = key_sending(mode, effective("auth_header"), effective("auth_scheme"))
        sent_as, unused, why = written["api_key_sent_as"], [], ""
        if sent_as == "bearer":
            unused = [k for k in ("auth_header", "auth_scheme") if k in own]
        elif mode == "both" and sent_as == "header":
            why = f' (with the {AUTH_HEADER} header, "both" sent the key once)'
        elif mode not in AUTH_MODES:
            why = f' ({prefix}.auth_mode {_shown(effective("auth_mode"))} read as "header", as layout 1 read it)'
        binding.update(written)
        carried = [f"{prefix}.{k}" for k in own if k not in unused]
        if carried:
            self.note(
                f"{wrapping.joined(carried)} {'are' if len(carried) > 1 else 'is'} now "
                + wrapping.joined(
                    f"model_bindings.{name}.{k} {json.dumps(v)}"
                    for k, v in written.items()
                )
                + why
            )
        if unused:
            self.note(
                f"{wrapping.joined(prefix + '.' + k for k in unused)}: not carried, as auth_mode \"bearer\" sent "
                "the key as a bearer token alone"
            )

    def human(self) -> None:
        """The human keys of advanced that keep one value, and the merged keys with a home in layout 2."""
        for old, new in HUMAN:
            self.human_move(old, new)
        self.web_host()
        self.node_environment()
        self.prices()
        url = self.human_move("publish.html.url", f"{TEAM}.public_url")
        self.merged(
            "team_dashboard.url", url, f"{TEAM}.public_url", "advanced.publish.html.url"
        )
        value = self.take("advanced.publish.html.requests")
        enabled = priority_requests(value)
        if value is not MISSING and _same(
            _typed(value), _get(self.defaults, "publish.html.requests")
        ):
            self.same += 1
        elif enabled:
            self.put(f"{TEAM}.priority_requests.enabled", enabled)
            place = (
                f"; the requests file's place comes from {TEAM}.{PUBLISH_DELIVERY}.site_folder"
                if self.block == PUBLISH_DELIVERY
                else ""
            )
            self.note(
                f"advanced.publish.html.requests is now {TEAM}.priority_requests.enabled {json.dumps(enabled)}"
                f"{place}"
            )
        value = self.take("advanced.jira.component")
        if value is not MISSING and _same(
            _typed(value), _get(self.defaults, "jira.component")
        ):
            self.same += 1
        elif not _blank(value):
            parts = jira_components(value)
            if parts is None:
                self.note(
                    f"advanced.jira.component is {_shown(value)}: not carried, as it is text"
                )
            else:
                self.put(f"{TC}.components", parts)
                self.note(
                    f"advanced.jira.component is now {TC}.components, split at commas: {json.dumps(parts)}"
                )
        self.merged(
            "jira.url",
            self.jira_site,
            "integrations.atlassian_cli.jira_url",
            "integrations.atlassian_cli.jira_site",
            one="your JIRA site",
            address=jira_address,
        )

    def carry(self, key: str, value, new: str, address=None) -> None:
        """KEY's VALUE at NEW, with its note; ADDRESS, a rule such as `jira_address`, gives the address a value
        stands for, and the note names it when it is not the value as it was read."""
        carried = address(value) if address else value
        self.put(new, carried)
        if carried is value:
            self.note(settings_file.Moved(key, new))
        else:
            self.note(
                f"{key} {_shown(value)} is now {new} {json.dumps(carried)}, the address of the site it names"
            )

    def merged(
        self, old: str, kept, new: str, kept_from: str, one: str = "", address=None
    ) -> None:
        """advanced.OLD, an address, merged into NEW, which KEPT, the value of KEPT_FROM, may have already:
        carried when KEPT is blank; the same address (`same_address`) is one. Another is reported and not carried; with
        ONE, the place both name, which layout 2 has one address for, it is refused, naming both keys. ADDRESS,
        a rule such as `jira_address`, gives the address each value stands for, compared and carried.
        """
        key = f"advanced.{old}"
        value = self.take(key)
        if value is MISSING or _blank(value):
            if value is not MISSING and _same(_typed(value), _get(self.defaults, old)):
                self.same += 1
            return
        form = address or (lambda v: v)
        if _same(_typed(value), _get(self.defaults, old)):
            self.same += 1
        elif _blank(kept):
            self.carry(key, value, new, address)
        elif same_address(form(value), form(kept)):
            self.note(f"{key}: the same as {kept_from}, the one {new}")
        elif one:
            raise StepRefused(
                key,
                f"{key} is {_shown(value)} and {kept_from} is {_shown(kept)}, and layout 2 has one address for {one}",
                lambda where: Steps(
                    (
                        Edit(kept_from, where, Described(f"{one}'s address")),
                        ", then",
                        Edit(key, where, DELETE),
                    )
                ),
            )
        else:
            self.note(
                f"{key} is {_shown(value)}: not carried, as {new} is {_shown(kept)}, from {kept_from}"
            )

    def human_move(self, old: str, new: str | None, dropped: str = ""):
        """A human key of advanced at NEW, unless it is the package's default or blank; with DROPPED, why the file
        has no place for it, it is dropped with a note instead (NEW None: layout 2 has no key for it). Returns the
        earlier value.
        """
        value = self.take(f"advanced.{old}")
        if value is MISSING:
            return value
        if _same(_typed(value), _get(self.defaults, old)):
            self.same += 1
        elif _blank(value):
            pass
        elif dropped:
            self.note(f"advanced.{old} {_shown(value)}: dropped, as {dropped}")
        else:
            self.put(new, value)
            self.note(settings_file.Moved(f"advanced.{old}", new))
        return value

    def web_host(self) -> None:
        """The web host's keys of advanced: the team dashboard's publish block, with delivery "publish"
        alone, as check 7 takes it; with another delivery, dropped, their values staying in the earlier file. The
        keys layout 2 has no key for are dropped, each note saying where it goes (`why_dropped`).
        """
        dropped = ""
        if self.block != PUBLISH_DELIVERY:
            dropped = (
                f"the team dashboard's delivery is {json.dumps(self.delivery)}, and only "
                f"{json.dumps(PUBLISH_DELIVERY)} copies the page to a web host; the value stays in the earlier file, "
                "which migrate keeps"
            )
        for old, new in WEB_HOST:
            self.human_move(old, f"{TEAM}.{PUBLISH_DELIVERY}.{new}", dropped)
        for old in WEB_HOST_DROPPED:
            self.human_move(old, None, why_dropped(f"advanced.{old}"))

    def updates(self) -> None:
        """advanced.update.repo and .branch as the updates section (`updates_section`); without one, a note says
        why the section is left out, and how to print it. advanced.update.path is dropped with a note (`why_dropped`).
        """
        old = {k: f"advanced.update.{k}" for k in ("repo", "branch")}
        repo, branch = self.take(old["repo"]), self.take(old["branch"])
        section = updates_section(repo, branch)
        if section:
            self.put(UPDATES, section)
            new = f"{UPDATES}.repository"
            self.note(
                settings_file.Moved(
                    old["repo"],
                    new,
                    f"{old['repo']} is now {new} {json.dumps(section['repository'])}",
                )
            )
            self.note(settings_file.Moved(old["branch"], f"{UPDATES}.branch"))
        elif repo is not MISSING or branch is not MISSING:
            # Why the section is left out, and a value that fits, which goes with it.
            why, unused = [], []
            for key, value, fits, what in (
                (
                    old["repo"],
                    repo,
                    github_repository(repo),
                    "GitHub address or <org>/<repo>",
                ),
                (old["branch"], branch, branch_name(branch), "branch's name"),
            ):
                if value is MISSING:
                    why.append(f"{key} is missing")
                elif _blank(value):
                    why.append(f"{key} is blank")
                elif not fits:
                    why.append(f"{key} is no {what}")
                else:
                    unused.append(key)
            either = f", so {unused[0]} is not carried either" if unused else ""
            prints = steps.command_line(self.file_type.section_command, KEY=UPDATES)
            self.note(
                f"{UPDATES}: left out, as {wrapping.joined(why)}{either}; {prints} prints the section"
            )
        self.human_move("update.path", None, why_dropped("advanced.update.path"))

    def node_environment(self) -> None:
        """advanced.gpu.env, without the variable the repository's defaults set first: `gpu_environment`'s."""
        env = self.take("advanced.gpu.env")
        if env is MISSING:
            return
        if _same(_typed(env), _get(self.defaults, "gpu.env")):
            self.same += 1
        elif isinstance(env, dict):
            kept = gpu_environment(env)
            first = f"{GPU_BACKEND[0]}={GPU_BACKEND[1]}, which the repository's defaults set first"
            if kept:
                new = f"{RGT}.node.environment"
                self.put(new, kept)
                without = f", without {first}" if len(kept) < len(env) else ""
                self.note(
                    settings_file.Moved(
                        "advanced.gpu.env",
                        new,
                        f"advanced.gpu.env is now {new}{without}",
                    )
                )
            else:
                self.note(
                    f"advanced.gpu.env {_shown(env)}: not carried; each job has {first}"
                )
        elif not _blank(env):
            self.note(
                f"advanced.gpu.env is {_shown(env)}: not carried, as it maps variables to values"
            )

    def prices(self) -> None:
        prices = self.take("advanced.prices")
        if prices is MISSING:
            return
        if _same(_typed(prices), _get(self.defaults, "prices")):
            self.same += 1
            return
        if not isinstance(prices, dict):
            if not _blank(prices):
                self.note(
                    f"advanced.prices is {_shown(prices)}: not carried, as it maps model ids to prices"
                )
            return
        out = {}
        for model, fields in prices.items():
            key = f"advanced.prices.{model}"
            if not isinstance(fields, dict):
                self.note(
                    f"{key} is {_shown(fields)}: not carried, as a model's prices are a section"
                )
                continue
            kept, other = model_prices(fields)
            for field, value in other.items():
                self.note(
                    f"{key}.{field} {_shown(value)}: not carried, as {NOT_A_PRICE}"
                )
            out[str(model)] = kept
        enabled = spend_enabled(out)
        if enabled:
            renamed = {k: v for k, v in PRICE_FIELDS.items() if k != v}
            new = f"{RA}.spend.usd_per_million_tokens"
            self.put(f"{RA}.spend.enabled", enabled)
            self.put(new, out)
            self.note(
                settings_file.Moved(
                    "advanced.prices",
                    new,
                    f"advanced.prices is now {new}, by model id, with {wrapping.joined(renamed)} named "
                    f"{wrapping.joined(renamed.values())}; {RA}.spend.enabled {json.dumps(enabled)}",
                )
            )

    def advanced(self) -> None:
        """Every other value under advanced: equal to its package default, dropped and counted; else reported."""
        rest = self.take("advanced")
        if _blank(rest):
            return
        if not isinstance(rest, dict):
            self.note(
                f"advanced is {_shown(rest)}: not carried, as it maps the package's keys to values"
            )
            return
        for dotted, value in leaves(rest):
            self.compare(
                f"advanced.{dotted}",
                value,
                _get(self.defaults, dotted),
                MERGED.get(dotted),
            )

    def compare(self, key: str, value, default, why: str | None = None) -> None:
        if default is MISSING:
            self.note(
                f"{key} is {_shown(value)}: not carried, as the package has no such setting"
            )
        elif _same(_typed(value), default):
            self.same += 1
        else:
            self.note(
                f"{key} is {_shown(value)} (the package's default is {_shown(default)}): not carried, as "
                + (why or "layout 2 has no setting for it")
            )

    def other_parts(self) -> None:
        secrets = self.take("secrets")
        if secrets is not MISSING:
            n = (
                len(secrets)
                if isinstance(secrets, (dict, list))
                else int(not _blank(secrets))
            )
            self.note(
                f"secrets ({wrapping.counted(n, 'entry', 'entries')}): dropped, as {why_dropped('secrets')}"
            )
        if self.take("setup") is not MISSING:
            self.note(f"setup: dropped, as {why_dropped('setup')}")
        self.provenance()


def deploy(values: dict, file_type: settings_file.FileType) -> tuple:
    """Deploy settings in layout 1, in layout 2: (values, notes). See the module's docstring."""
    step = _Deploy(values, file_type)
    step.run()
    return step.out, step.notes


# --- build settings --------------------------------------------------------------------------------------------------


class _Build(_Step):
    def run(self) -> None:
        self.move_image_name("image.name")
        for old, new in BUILD_MOVES.items():
            self.move(old, new)
        self.rocm_pins()
        version = self.move("claude_code.version", "claude_agent_sdk.version")
        if not _blank(version):
            self.note(
                "claude_agent_sdk.package_url and .package_sha256 start blank: the package index serves it"
            )
        self.package_index()
        self.provenance()
        self.leftovers()

    def rocm_pins(self) -> None:
        """rocm.clang_commit and rocm.gfx_targets: dropped, as the package pins them itself; a value other than the
        package's is reported. The package's are rkmon_lib/build_plan.py's: the clang commit for the file's
        ROCm release (build_plan.ROCM_RELEASE when the file names none), and build_plan.ROCM_LIBRARY_TARGETS.
        """
        version = _text(self.out.get("rocm", {}).get("version", MISSING))
        release = version or build_plan.ROCM_RELEASE
        clang = self.take("rocm.clang_commit")
        if not _blank(clang):
            text = _text(clang)
            pinned = build_plan.clang_commit(release)
            meant = "" if text.lower() == ANY_CLANG else text
            if meant == pinned:
                self.note(
                    f"rocm.clang_commit {_shown(clang)}: dropped, the package's own"
                )
            else:
                self.note(
                    f"rocm.clang_commit {_shown(clang)}: dropped, as the package pins the clang commit itself "
                    f"({json.dumps(pinned) if pinned else 'none'} for ROCm {release})"
                )
        targets = self.take("rocm.gfx_targets")
        if not _blank(targets):
            typed = _typed(targets)
            items = typed if isinstance(typed, list) else [typed]
            text = ",".join(str(x).strip() for x in items if str(x).strip())
            package_targets = ",".join(build_plan.ROCM_LIBRARY_TARGETS)
            if (text or package_targets) == package_targets:
                self.note(
                    f"rocm.gfx_targets {_shown(targets)}: dropped, the package's own"
                )
            else:
                self.note(
                    f"rocm.gfx_targets {_shown(targets)}: dropped, as the package sets its GPU targets itself "
                    f"({package_targets})"
                )

    def package_index(self) -> None:
        """claude_code.package_index into python_packages.package_index; two different indexes refused."""
        claude = self.take("claude_code.package_index")
        python = self.take("python_packages.package_index")
        if not _blank(python):
            self.put("python_packages.package_index", python)
        if _blank(claude):
            return

        def meant(value) -> str:
            return (_text(value) or PUBLIC_PYPI).rstrip("/")

        def shown(value) -> str:
            if _blank(value):
                return f"blank, which means {PUBLIC_PYPI}"
            return json.dumps(_text(value))

        if meant(claude) == meant(python):
            if _blank(python):
                self.put("python_packages.package_index", claude)
                self.note(
                    settings_file.Moved(
                        "claude_code.package_index",
                        "python_packages.package_index",
                        "claude_code.package_index is now python_packages.package_index, the one index for every "
                        "Python package",
                    )
                )
            else:
                self.note(
                    "claude_code.package_index: the same index as python_packages.package_index, the one for every "
                    "Python package"
                )
            return
        key = "claude_code.package_index"
        raise StepRefused(
            key,
            f"{key} is {shown(claude)} and python_packages.package_index is {shown(python)}, and layout 2 has one "
            "package index for every Python package",
            lambda where: Steps(
                (
                    Edit(
                        "python_packages.package_index",
                        where,
                        Described("the one index every Python package comes from"),
                    ),
                    ", then",
                    Edit(key, where, DELETE),
                )
            ),
        )


def build(values: dict, file_type: settings_file.FileType) -> tuple:
    """Build settings in layout 1, in layout 2: (values, notes). See the module's docstring."""
    step = _Build(values, file_type)
    step.run()
    return step.out, step.notes
