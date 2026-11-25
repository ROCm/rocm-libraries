#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Production's settings as a deploy settings file of this package, and a report of what is not one value moved to one
key. Production runs the package's earlier layout in its container: its settings are the work volume's config.json,
its Claude settings and key the secret claude_env.

  python3 tools/translate_production.py --container NAME (--out FILE | --report-only)
  python3 tools/translate_production.py --container NAME --round-trip [--agreed FILE] [--out FILE]

Run it on production's host, from this package's folder. It runs a one-off container of NAME's own image, never
pulled, with no network and no log, its root read-only, as the monitor's user with no capability: NAME's work and
secrets volumes are mounted read-only, and this package's folder beside them, read-only too; of NAME's environment it
reads the monitor's ids alone (RKMON_UID and RKMON_GID). In there it reads config.json, and claude_env's shape: names
and settings, never a credential (rkmon_lib/claude_env_shape.py). It prints the file and the report,
and the host prints the report. With --out, the host writes the file as FILE, yours only (rkmon_lib/private_files.py);
with --report-only it writes nothing, and the next line runs the round trip. Nothing in
NAME or its volumes changes. A setting the file cannot carry stops it: the report names each one under Stops with
its fix, nothing is written, and it exits 4; the outcome line is followed by the fixes and the command that runs it
again.

With --round-trip it runs the round trip in the same one-off container, and writes no settings file: the file
taken back to today's names and compared with config.json and claude_env's shape, key by key and as the monitor runs
each, with the usage the monitor recorded priced both ways (tools/round_trip.py, `compare`). Its report has a row for
each difference, by its key and its kind, with both sides as the report shows a value and a header's key only as
rkmon_lib/custom_headers.py's KEY_MARK: PASS when a row of the list of intended differences lists it, with that row's
reason; FAIL, with its fix, when none does. The list is the package's (round_trip.DIFFERENCES) and, with --agreed,
the rows the owner agreed: FILE, in the list's form, read only when it is a private file. The report goes to the
terminal, or with --out to FILE, yours only, and then the terminal gets the outcome alone; FILE is checked before the
round trip runs. It passes when each difference is listed, and otherwise fails (exit 4). No usage recorded fails it
too, as no price is then compared: a FAIL row whose fix names where the monitor's transcripts are looked for
(`_usage_rows`); and so does a setting that stops the translation, as the round trip cannot run. A round trip with no
answer in ROUND_TRIP_SECONDS, or one that stops with no message of its own, is an ERROR that says how far it got: the
last step it began (TRIP_STEPS).

The one-off container has a time limit (RUN_SECONDS, and ROUND_TRIP_SECONDS for the round trip). On it, or on Ctrl-C,
this tool removes the container (`remove_one_off`), by the id docker wrote to a file in a private temporary folder
(--cidfile), never by a name: the outcome line, ERROR or INTERRUPTED (exit 130), says it is removed, and the next line
runs the command again; a container it could not remove is named in the fix, `docker rm -f` and its id. It runs beside
production with one CPU's time and ONE_OFF_MEMORY at most (ONE_OFF_CPUS); one killed (exit KILLED), as it is at that
memory limit, is an ERROR that names the limit.

- `translate(config, claude_env) -> Translation(deploy_file, report)`: the translation, pure. CONFIG is config.json's
  data, CLAUDE_ENV its shape (claude_env_shape.Shape); it reads nothing else but this package's own files (the schema
  and config.defaults.json), and writes nothing. DEPLOY_FILE is the file's text as `rkmon migrate` writes one
  (settings_file.dump): every value production has, defaults included, each at the key whose `x-internal` names it,
  and the keys in SPECIAL by the rules of this module's functions; a feature that is off and keeps its defaults is
  written as its switch alone; where production has no value for the container's memory or its CPUs, and the
  package's default is less than the file's own rules take, the least they take (`least_resources`). REPORT is its
  lines, a heading for each of SECTIONS and an item for each thing it says, production's older switches in it quoted
  as code (`review-posting true`); under BUILD, the image's keys config.json has, each with its key in a build
  settings file by the build's rules (from_layout1.BUILD_MOVES), as no build file is written; under CHECK, what the
  file's own check says of the text read back as `rkmon check` reads the file, the rules on the container's share of
  the host among it, with no figure of this host (coherence.resources). A setting it cannot carry raises
  Untranslatable, with each one and the whole report.
- A stop (`Stop`) says what was found and its fix: a key of config.json, or of claude_env, to change. A row of the
  file's own check that fails keeps its fix, each key of the file in it turned into the keys of production's it comes
  from (`Origin`), where there are some.
- Layout 1's rules, which `rkmon migrate` shares, give each value they cover: rkmon_lib/from_layout1.py's public
  functions on plain values (`labels`, `notices`, `pass_effort`, `key_sending`, `updates_section`, `why_dropped` and
  the others), which write no note. This module calls them on production's values, after its own rules of
  production's shape (the older switches, the feature map, the local worker's settings), and writes the report's
  lines.
- `one_off(docker, image, work, secrets, user, args, cidfile=None)`: the one-off container's command, this tool run
  with ARGS in it (--inside, and with it --round-trip for the round trip), its id written to CIDFILE;
  `monitor_user(ids)`, the user it runs as; `remove_one_off(docker, cidfile) -> Removal(id, gone, why)`, the container
  removed by that id.
- `main(argv)`: the command line above.

Standard library only: the translation runs inside production's image, on its own Python.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from rkmon_lib import (  # noqa: E402
    bindings,
    checks,
    claude_env_shape,
    coherence,
    config,
    container_layout,
    custom_headers,
    deploy_tree,
    deployment,
    derived,
    features,
    file_types,
    first_pass,
    from_layout1,
    host_keys,
    orchestrator,
    outcome,
    paths,
    private_files,
    program,
    registry,
    removed_keys,
    renames,
    schema,
    secrets_registry,
    settings_file,
    spend,
    steps,
    wrapping,
)
import round_trip  # noqa: E402

TOOL = "tools/translate_production.py"
# Production's configuration, as the report and the stops' fixes name it.
CONFIG_FILE = "config.json"
# What the translation says of a config.json that is not a JSON object.
NOT_SETTINGS = f"{CONFIG_FILE} is not a mapping of settings"
# The file the translation writes, as the file's own check names a file with no path: a fix keeps a key of it when no
# key of production's gives its value.
THE_FILE = f"your {file_types.DEPLOY.noun}"
RA = from_layout1.RA
MQ = from_layout1.MQ
GUARD = from_layout1.GUARD
TEAM = from_layout1.TEAM
ALERTS = from_layout1.ALERTS
TICKETS = from_layout1.TC
CLAUDE_CODE = "integrations.claude_code"
# The older secret production keeps its Claude settings in, the claude login's older name that the one-time move copies
# the key from: a stop's fix edits it.
CLAUDE_ENV = secrets_registry.CLAUDE_ENV
# The tick interval: production keeps it under each key its x-internal names (`tick_interval`).
TICK = "monitor.tick_interval_minutes"
# The team dashboard's page, as config.json names it: no deploy key has it, as the page is derived from the site folder.
PAGE = "publish.html.dest"
# The passes that always run, in the Claude session: (deploy key, its role to config.claude_model), the function that
# reads its model as today's monitor does. The role reads the key the x-internal of the pass's model names, as a test
# holds.
CLAUDE_PASSES = (
    (f"{RA}.orchestrator", "orchestrator"),
    (f"{RA}.passes.verification_pass", "audit"),
    (f"{RA}.passes.code_quality_pass", "deep"),
)
# The optional passes' deploy keys. The first review pass is translated on its own (`first_review_pass`); the others
# name a model as `optional_models` reads it.
FIRST_REVIEW_PASS = f"{RA}.passes.first_review_pass"
OPTIONAL_PASSES = (
    FIRST_REVIEW_PASS,
    f"{RA}.passes.ci_triage_pass",
    f"{RA}.passes.first_code_quality_pass",
    f"{RA}.passes.notice_draft_pass",
)
# Each pass's Claude role to layout 1's rules, by its deploy key (from_layout1.ROLES): its effort is the role's
# (from_layout1.pass_effort), and the role reads the key the x-internal of the pass's effort names, as a test holds.
ROLE = {new: role for role, _old, new, _owner in from_layout1.ROLES}
# The deploy keys the rules of this module fill in, each from production's values in its own way; every other key with
# an x-internal takes the value at that configuration key, as it is. A rule reads production's keys through the
# x-internals of the keys it fills (`_Translator.internal`), never by a name of its own.
SPECIAL = (
    "container.name",
    "image.name",
    "image.tag",
    TICK,
    "updates.repository",
    "updates.branch",
    "repository.shared_label_folders",
    "host.resources.ramdisk_gib",
    "host.users.monitor_uid",
    "host.users.monitor_gid",
    *(f"{key}.{part}" for key, _m in CLAUDE_PASSES for part in ("model", "effort")),
    *(f"{key}.{part}" for key in OPTIONAL_PASSES for part in ("model", "effort")),
    f"{RA}.remote_gpu_tests.node.environment",
    f"{RA}.spend.enabled",
    f"{RA}.spend.usd_per_million_tokens",
    f"{MQ}.auto_merge.enabled",
    f"{MQ}.auto_merge_guard.on_your_pull_requests",
    f"{MQ}.queue_notices.enabled",
    f"{MQ}.drift_notices.enabled",
    f"{GUARD}.enabled",
    f"{GUARD}.ci_red_label",
    f"{GUARD}.ci_red_extra_labels",
    f"{GUARD}.stability_label",
    "features.dashboards.admin.port",
    f"{TEAM}.mode",
    f"{TEAM}.public_url",
    f"{TEAM}.priority_requests.enabled",
    f"{TEAM}.delivery",
    f"{TEAM}.publish.web_host",
    f"{TEAM}.publish.username",
    f"{TEAM}.publish.site_folder",
    f"{TEAM}.serve.address",
    f"{TEAM}.serve.port",
    f"{ALERTS}.mode",
    f"{ALERTS}.develop_breaks.enabled",
    f"{ALERTS}.needs_you.enabled",
    f"{TICKETS}.mode",
    f"{TICKETS}.components",
    f"{TICKETS}.follow_up_tickets.enabled",
    f"{TICKETS}.found_issue_tickets.enabled",
    "integrations.atlassian_cli.jira_url",
    f"{CLAUDE_CODE}.base_url",
    f"{CLAUDE_CODE}.api_key_sent_as",
    f"{CLAUDE_CODE}.api_key_header",
    f"{CLAUDE_CODE}.api_key_scheme",
    "model_bindings",
)
# The local worker's binding: its name, as layout 1's rules name it (from_layout1.WORKER), and the secret that holds
# its key today, the first review pass's key's older name (secrets_registry.OLDER_KEY_NAMES), which the one-time move
# copies to the name from_layout1.api_key_secret_name gives it.
WORKER = from_layout1.WORKER
(WORKER_KEY,) = [
    older
    for older, name in secrets_registry.OLDER_KEY_NAMES.items()
    if name == secrets_registry.FIRST_PASS_KEY
]
# What turning the develop guard on does, which a fix that turns it on says.
GUARD_DOES = "the develop guard: it files the develop issues and the fix links"
# A model id as a model binding's name takes it: anything else in an id becomes "-".
BINDING_NAME = re.compile(r"^(?!(default|none)$)[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Today's internal keys by where each goes, until the registry of the internal keys holds them: a key
# here, or one under it, is reported in its section. The state keys the monitor finds or records:
STATE_KEYS = (
    "codeowner_teams",
    "clone_since",
    "base_branch",
    "worker.model_name",
    "worker.chat_path",
    "worker.models_url",
    "worker.probe",
    "ci_infra_site_signatures",
    "publish.html.host_keys",
    "jira.accounts",
)
# The keys computed from the file and the container's layout:
DERIVED_KEYS = (
    "codeowner_org",
    "clone_url",
    "worker.api_key_file",
    "worker.tag",
    "draft_stage",
    "profile_dir",
    "review_first_line",
    "verifier_name",
    "review_first_line_no_draft",
    "models.draft",
    "weekly_report.dir",
    "followup_scan.dir",
    "publish.html.known_hosts",
    "gpu.ssh_key",
    "gpu.known_hosts",
    "admin_dashboard.port",
    "admin_dashboard.bind",
    "admin_dashboard.access_token_file",
    "team_dashboard.port",
    "team_dashboard.bind",
    "teams.webhook_file",
    "ramdisk.path",
    "gh.read_config_dir",
    "gh.write_config_dir",
    "claude_workspace",
    *(
        f"paths.{name}"
        for name in (
            "working_dir",
            "outputs_dir",
            "toolkit",
            "repo",
            "workers",
            "tmp",
            "probes",
            "lessons",
            "cache",
            "scratch",
            "worker_tools",
            "venv",
            "build",
            "agentic",
            "secrets_dir",
            "run",
            "review_book",
            "rocm_llvm_bin",
            "dashboard",
            "html",
            "daily_dir",
            "insights_dir",
            "posted_dir",
            "posts",
            "verifier_team_file",
            "claude_projects",
            "cmake",
            "pyyaml",
            "torch_python",
            "spend_dir",
            "gh",
        )
    ),
    # the paths that are in no configuration file, which only a key added by hand puts in production's file
    *registry.DERIVED_NOT_IN_FILES,
)
# The package's own values, which no person sets: production's value is reported when it differs, never carried.
PACKAGE_KEYS = (
    "test_sparse",
    "review_checklist",
    "hot_spots",
    "rebase_noise_paths",
    "ci_test_tiers",
    "ci_absent_modules",
    "ci_infra_signatures",
    "drift_extract",
    "develop_watched_workflows",
    "develop_rocke_job_pattern",
    "incident_issue.time_limit_labels",
    "gpu.skip_pattern",
    "gpu.max_cases",
    "draft_parallel_max",
    "drafter",
    "checks.min_free_memory_gb",
    "clashes",
    "max_posts_per_hour",
    "huge_pr_lines",
    "libparity_check",
    "claude.auto_compact_window",
    "claude.prompt_cache_ttl",
    "claude.subagent_prompt_cache_ttl",
    "weekly_report.weekday",
    "weekly_report.hour_utc",
    "weekly_report.friction_window_h",
    "followup_gap_exempt",
    "loop_heartbeat_min",
    "tick_timeout_min",
    "analysis_max_wait_min",
    "revert_offer_after_min",
    "hold_stale_h",
    "drift_batch",
    "hold_label_patterns",
    "merge_plan",
    "conflicts",
    "ci_times",
    "overlaps",
    "scoreboard",
    "tracker_allow",
    "tracker_keys",
    "jira.priority_map",
    "followup_scan.defer_patterns",
    "followup_scan.request_patterns",
    "followup_scan.ticket_pattern",
    "max_host_usage_pct",
    "gh.probe_issue",
    "retention",
    "update.drain_timeout_min",
    "update.poll_s",
)
# The keys no feature reads any more, each with why: rkmon_lib/removed_keys.py, their one home.
REMOVED_KEYS, WHY_DROPPED = removed_keys.REMOVED_KEYS, removed_keys.WHY_DROPPED
# The keys whose difference is intended: production's value, and the package's in its place, each with why
# (removed_keys.PACKAGE_SET, their one home).
INTENDED_KEYS = {
    k: removed_keys.PACKAGE_SET[k] for k in ("container.write_uid", "container.pr_uid")
}
# The image's keys that config.json has under a name from before layout 1's build settings, each with the key of
# layout 1's build settings in its place, which the build's rules move on (from_layout1.BUILD_MOVES), and what that
# key takes: layout 1's package read each so.
OLDER_BUILD_KEYS = {
    "rocm.installer_base": (
        "rocm.package_url",
        "one installer's full URL, not a folder",
    ),
}
# config.json's mappings whose keys are values (a model id, a login, a label), each one key of its own.
WHOLE = (*from_layout1.FREE_FORM, "model_bindings")

# The report's sections, in order: each one's heading.
READ = "What was read"
STOPS = "Stops: settings the file cannot carry; nothing is written while there is one"
SWITCHES = "Feature switches"
FORMS = "Values that change form"
EFFORTS = "Efforts, each as the pass runs today"
FIRST = "The first review pass"
MODELS = "Models"
CLAUDE = "Claude Code: claude_env's shape, and the settings made from it"
LOGINS = "Logins and secrets"
PACKAGE = "Package keys production sets to another value: each takes the package's own"
REMOVED = "Removed keys: no feature reads them"
STATE = "State keys: the monitor finds or records them"
DERIVED = "Derived keys: computed from the file and the container's layout"
INTENDED = "Intended differences"
DEFAULTED = "Keys that take the package's default"
BUILD = "Build settings: the image's keys, each with its key in a build settings file (this translation writes none)"
UNKNOWN = "Keys this translation does not know: not carried"
CHECK = "The file's own check"
NEXT = "Next"
SECTIONS = (
    READ,
    STOPS,
    SWITCHES,
    FORMS,
    EFFORTS,
    FIRST,
    MODELS,
    CLAUDE,
    LOGINS,
    PACKAGE,
    REMOVED,
    STATE,
    DERIVED,
    INTENDED,
    DEFAULTED,
    BUILD,
    UNKNOWN,
    CHECK,
    NEXT,
)
MISSING = deploy_tree.MISSING


class Translation(NamedTuple):
    """The deploy settings file's text, and the report's lines."""

    deploy_file: str
    report: list


class Stop(NamedTuple):
    """A setting the file cannot carry: what was found, one sentence, and the fix (a checks.Fix) that lets the file
    carry it."""

    found: str
    fix: checks.Fix


class Origin(NamedTuple):
    """Where a key of the file takes its value from (`_Translator.origin`): KEYS of FILE, each set to the key's value
    when SAME, else to a value that makes it; WHAT, what setting them does, which a fix says; OFF, for a switch that is
    on while each of KEYS is, the keys a fix sets to false to turn it off (all of KEYS when empty). No KEYS: no key
    gives the value alone."""

    keys: tuple = ()
    same: bool = True
    what: str = ""
    file: str = CONFIG_FILE
    off: tuple = ()


NO_ORIGIN = Origin()


class Untranslatable(ValueError):
    """Settings the file cannot carry: STOPS, a Stop each, and REPORT, the whole report, with them under its Stops
    heading."""

    def __init__(self, stops: list, report: list):
        self.stops = list(stops)
        self.report = list(report)
        super().__init__("; ".join(s.found for s in self.stops))


def translate(config: dict, claude_env: claude_env_shape.Shape) -> Translation:
    """Production's config.json data and claude_env's shape as a deploy settings file, and the report (the module's
    docstring). Untranslatable when a setting cannot be carried."""
    return _Translator(config, claude_env).run()


# --- words ------------------------------------------------------------------------------------------------------------


def _unshown(text: str, noun: str) -> str:
    """What the report says in place of TEXT, a NOUN ("value" or "key") it never shows, as no path a settings file's
    check shows holds one (settings_file.hidden_key): one that looks like a secret, or holds half a character; "" for
    any other."""
    if not settings_file.hidden_key(text):
        return ""
    if settings_file.secret_shape(text):
        return f"(a {noun} that looks like a secret: not shown)"
    return f"(a {noun} with half a character in it: not shown)"


def _shown(value, most: int | None = 80) -> str:
    """A value as the report shows it: JSON, at most MOST characters (None: whole); never one `_unshown` hides."""
    if value is MISSING:
        return "(not set)"
    text = json.dumps(value, ensure_ascii=False)
    whole = most is None or len(text) <= most
    return _unshown(text, "value") or (text if whole else text[: most - 3] + "...")


def _text(value) -> str:
    """A value of production's as text, stripped: "" for none (MISSING) and for a blank (null, "", false)."""
    return str(value).strip() if value is not MISSING and value else ""


def _for_rule(value):
    """VALUE, as `_Translator.peek` gives it, for a rule of layout 1's (rkmon_lib/from_layout1.py): that module's own
    MISSING for a key production does not have."""
    return from_layout1.MISSING if value is MISSING else value


def _off(value) -> bool:
    """Whether production's VALUE of a removed key is blank or off: such a key is counted, not listed."""
    return value in ("", [], {}, None, False)


def _why_removed(name: str) -> str:
    """Why no feature reads production's key NAME, an entry of REMOVED_KEYS or WHY_DROPPED (removed_keys.why)."""
    return removed_keys.why(name)


def _build_key(key: str) -> str:
    """The key of a build settings file that production's KEY, a key of the image's, goes to by the build's rules
    (from_layout1.BUILD_MOVES), from the key of layout 1's in its place when it has an older name (OLDER_BUILD_KEYS),
    with what that key takes; "" for any other key."""
    older, takes = OLDER_BUILD_KEYS.get(key, (key, ""))
    new = from_layout1.BUILD_MOVES.get(older, "")
    return f"{new}, which takes {takes}" if new and takes else new


def _word(value) -> str:
    return "true" if value is True else "false" if value is False else _shown(value)


def _whole(number):
    """NUMBER as the file has it: a whole number without its ".0"."""
    return int(number) if float(number).is_integer() else number


def _older(name: str) -> str:
    """The key of the older switch NAME in config.json: its features mapping (renames.old_features)."""
    return f"features.{name}"


def _code(text: str) -> str:
    """TEXT, an older switch of production's (its name, alone or with its value after it), quoted as code: the report
    shows production's words as they are, and the word lists of the package's texts (tests/doc_words.yaml) skip text
    in backticks."""
    return f"`{text}`"


def _edit(key: str, words: str) -> checks.Edit:
    """The fix that sets config.json's KEY to a value the WORDS describe."""
    return checks.Edit(key, CONFIG_FILE, checks.Described(words))


def _one_address(first: str, second: str) -> checks.Steps:
    """The fix for two keys of config.json that name two addresses, where the file has one: either one set to the
    other's."""
    return checks.Steps(
        (
            _edit(second, f"the address {first} names"),
            ", or",
            _edit(first, f"the address {second} names"),
        )
    )


def _fix(parts: list) -> checks.Fix:
    """PARTS, Edits and Commands and the words that join them, as one fix: the one Edit or Command, or Steps."""
    return parts[0] if len(parts) == 1 else checks.Steps(tuple(parts))


def _together(fixes) -> checks.Fix:
    """FIXES as one fix: the one, or each one's steps in order, a ";" between two."""
    fixes = list(fixes)
    if len(fixes) == 1:
        return fixes[0]
    parts = []
    for fix in fixes:
        parts += [";"] if parts else []
        parts += fix.parts if isinstance(fix, checks.Steps) else [fix]
    return checks.Steps(tuple(parts))


def _key(*parts) -> str:
    """A key's path as the report shows it: its PARTS joined by dots, where a key `_unshown` hides is said, never
    shown."""
    return ".".join(_unshown(str(p), "key") or str(p) for p in parts)


def _listed(key: str, names) -> str:
    """The name in NAMES that is KEY or holds it ("" for none)."""
    for name in names:
        if key == name or key.startswith(name + "."):
            return name
    return ""


def _leaves(data: dict, prefix: tuple = ()) -> list:
    """config.json's keys, each as its parts: a mapping is one key when it is empty or its keys are values (WHOLE)."""
    out = []
    for k, v in data.items():
        parts = (*prefix, k)
        if isinstance(v, dict) and v and ".".join(map(str, parts)) not in WHOLE:
            out += _leaves(v, parts)
        else:
            out.append(parts)
    return out


def _binding_fields(deploy: dict) -> dict:
    """Each binding provider's fields: provider -> {configuration key: deploy key}: the keys of the provider's form of
    the bindings in DEPLOY, the deploy schema, each by the configuration key it feeds (deploy_tree.entries).
    """
    feeds = deploy_tree.entries("model_bindings")
    return {
        form["properties"]["provider"]["const"]: {
            feeds[key]: key for key, _node in schema.leaves(form) if key in feeds
        }
        for form in deploy["$defs"]["binding"]["oneOf"]
    }


def _internals() -> dict:
    """Each deploy key with an x-internal -> the configuration key it names, or the keys in the x-internal's order,
    for a key made of several: from deploy_tree.feeds, the one walk of the schema's x-internals.
    """
    keys: dict = {}
    for path, internal in deploy_tree.feeds():
        keys.setdefault(path, []).append(internal)
    return {path: k[0] if len(k) == 1 else k for path, k in keys.items()}


def _kind(b: dict) -> str:
    """A configuration binding's provider, as bindings.PROVIDERS names it (an older name given its current one)."""
    kind = str(
        b.get("provider") or b.get("backend") or bindings.binding_defaults()["provider"]
    )
    return bindings.OLD_PROVIDERS.get(kind, kind)


def _blank(value) -> bool:
    """Whether a value of production's is not set: none (MISSING, for null too), or text of spaces only."""
    return value is MISSING or (isinstance(value, str) and not value.strip())


def least_resources(values: dict) -> dict:
    """The container's memory and its CPUs, each by its key in the file (coherence.MEMORY, coherence.CPUS), with (the
    value the file's VALUES give it, its default where they give none; the least value the file's own rules on them
    take (coherence.resources, which `rkmon check` and `rkmon confirm` run too), as coherence.least gives it; the
    words that say why that one). VALUES are read as deployment.resources reads that configuration
    (deploy_tree.to_config): the memory has room for the RAM disk and the work's share (deployment.WORK_GB), and the
    CPUs are one for each build that may run at once, and for each review while review automation is on.
    """
    cfg = deploy_tree.to_config(values)
    r = deployment.resources(cfg)
    reviews = r["pipes"] if features.enabled(cfg, "review_automation") else 0
    takes = coherence.least(r["ramdisk_gb"], reviews, r["checks"])
    builds = f"{coherence.PARALLEL_BUILDS} ({r['checks']})"
    return {
        coherence.MEMORY: (
            _whole(r["memory_gb"]),
            _whole(takes.memory),
            f"less than {coherence.RAMDISK} ({_whole(r['ramdisk_gb'])}) plus the {deployment.WORK_GB} GiB the work "
            "needs",
        ),
        coherence.CPUS: (
            _whole(r["cpus"]),
            takes.cpus,
            (
                f"fewer than {coherence.PARALLEL_REVIEWS} ({reviews}) plus {builds}, the reviews and builds that may "
                "run at once"
                if reviews
                else f"fewer than {builds}, the builds that may run at once while review automation is off"
            ),
        ),
    }


# --- the translation ------------------------------------------------------------------------------------------------


class _Translator:
    """One translation: production's configuration (renames.moves gives each older key its current name), the deploy
    values made from it, what the rules read, the stops and the report's lines."""

    def __init__(self, raw, shape: claude_env_shape.Shape):
        if not isinstance(raw, dict):
            raise ValueError(NOT_SETTINGS)
        self.raw = copy.deepcopy(raw)
        self.cfg, self.moved = renames.moves(self.raw)
        self.shape = shape
        self.schema = file_types.DEPLOY.load()
        self.internals = _internals()
        # the package's values for production's repository and folder: its repository defaults' over the generic ones
        self.package_values = round_trip.package_for(self.cfg)
        self.package = copy.deepcopy(self.package_values.values)
        # the users' ids that are package values, which no settings file gives (container_layout)
        for key, user in (
            ("container.write_uid", container_layout.WRITER),
            ("container.pr_uid", container_layout.PR_RUNNER),
        ):
            deploy_tree.put(self.package, key, user.uid)
        self.flat = renames.old_features(self.raw)
        self.values: dict = {}
        self.read: set = set()
        # the keys renames.moves drops that production has blank or off: counted with the other removed keys
        self.dropped_off = 0
        self.stops: list = []
        self.lines = {title: [] for title in SECTIONS}
        # the file's keys that take the package's default -> each one's line among them (DEFAULTED)
        self.defaulted: dict = {}
        # the deploy keys and sections a rule leaves out of the file, each with a line of the report that says why
        self.left_out: set = set()
        # the bindings the passes name: name -> a configuration binding to add, or None for production's own
        self.models: dict = {}
        # where the keys a rule fills take their value from: key -> Origin (`came_from`); for a binding a pass made,
        # name -> the key of production's each of its fields comes from ("" for none); a key of the file a value of
        # production's names (a binding named by its model id) -> that value's key
        self.origins: dict = {}
        self.made_from: dict = {}
        self.named_by: dict = {}

    # values

    def internal(self, path: str):
        """The configuration key the x-internal of the deploy key PATH names, or the keys, for a key made of several:
        the one place a rule finds the keys it reads."""
        return self.internals[path]

    def peek(self, key: str):
        """The configuration's value at KEY (MISSING when it has none, or null)."""
        v = deploy_tree.get(self.cfg, key)
        return MISSING if v is None else v

    def take(self, key: str):
        """`peek`, and KEY is read: no other section of the report lists it."""
        self.read.add(key)
        return self.peek(key)

    def given(self, key: str):
        """The configuration's value at KEY, null too (MISSING when it has none), and KEY is read: for a rule that
        tells a null value from a key production does not have."""
        self.read.add(key)
        return deploy_tree.get(self.cfg, key)

    def put(self, path: str, value) -> None:
        """The file's key PATH holds VALUE, never MISSING (a fault of the rule: a key production does not have takes
        its default, or its rule says what it takes)."""
        if value is MISSING or (
            isinstance(value, dict) and any(v is MISSING for v in value.values())
        ):
            raise ValueError(f"no value for {path}")
        deploy_tree.put(self.values, path, value)

    def leave_out(self, path: str) -> None:
        """PATH, a key or a section, is left out of the file by a rule, which says why: the keys under it take no
        line among the keys that take the package's default."""
        self.left_out.add(path)

    def is_left_out(self, path: str) -> bool:
        return any(path == p or path.startswith(p + ".") for p in self.left_out)

    def say(self, section: str, text: str) -> None:
        self.lines[section].append(text)

    def stop(self, found: str, fix: checks.Fix) -> None:
        """A setting the file cannot carry: what was FOUND, and the FIX that lets the file carry it (a Stop)."""
        if not isinstance(fix, (checks.Command, checks.Edit, checks.Steps)):
            raise ValueError(f"a stop carries its fix: {found}")
        self.stops.append(Stop(found, fix))

    # where the file's values come from: a stop's fix changes those keys

    def came_from(
        self,
        path: str,
        *keys,
        same: bool = True,
        what: str = "",
        file: str = CONFIG_FILE,
        off: tuple = (),
    ) -> None:
        """The file's key PATH, and each key under it, take their value from KEYS (an Origin); no KEYS, from no one
        key of production's."""
        self.origins[path] = Origin(tuple(keys), same, what, file, tuple(off))

    def origin(self, path: str) -> Origin:
        """Where the file's key PATH takes its value from: the keys its rule named (`came_from`), or the key above it
        named, with the rest of PATH after each when the value is the same; for a key carried as it is, its x-internal,
        by the name production gives it (`production_key`). A list's item (`KEY[N]`, and what is under it) is its
        list's, at the same place."""
        head, bracket, tail = path.partition("[")
        if bracket:
            o = self.origin(head)
            if not o.same:
                return NO_ORIGIN
            return o._replace(
                keys=tuple(f"{k}[{tail}" for k in o.keys),
                off=tuple(f"{k}[{tail}" for k in o.off),
            )
        parts = path.split(".")
        for n in range(len(parts), 0, -1):
            head, rest = ".".join(parts[:n]), parts[n:]
            if head in self.origins:
                o = self.origins[head]
                if rest and not o.same:
                    return NO_ORIGIN
                return o._replace(
                    keys=tuple(".".join([k, *rest]) for k in o.keys),
                    off=tuple(".".join([k, *rest]) for k in o.off),
                )
            internal = self.internals.get(head)
            if head not in SPECIAL and isinstance(internal, str):
                key = self.production_key(internal)
                return Origin((".".join([key, *rest]),)) if key else NO_ORIGIN
        return NO_ORIGIN

    def production_key(self, key: str) -> str:
        """The configuration's KEY (its current name) as production's config.json has it: the older key renames.moves
        moved to it, when that one alone gives its value (a key of renames.RENAMED, or an older switch whose value is
        KEY's); "" when older keys that do not give it alone made it (the older switches, the phase, the profile).
        """
        olds = [old for old, new, _line in self.moved if new == key]
        if olds:
            old, value = olds[0], deploy_tree.get(self.cfg, key)
            alone = old in renames.RENAMED or (
                isinstance(value, bool)
                and renames.is_on(deploy_tree.get(self.raw, old)) == value
            )
            return old if len(olds) == 1 and alone else ""
        # a key production does not have: renames made it from the older switches, or it takes its default
        made = deploy_tree.get(self.raw, key) is MISSING and (
            deploy_tree.get(self.cfg, key) is not MISSING
        )
        return "" if made else key

    def in_production(self, fix: checks.Fix) -> checks.Fix:
        """FIX, a row's of the file's own check, with each edit of a key of the file turned into the edits of the keys
        it takes its value from (`origin`); an edit of a key with none stays as it is.
        """
        out = []
        for part in fix.parts if isinstance(fix, checks.Steps) else (fix,):
            o = self.origin(part.key) if isinstance(part, checks.Edit) else NO_ORIGIN
            if not o.keys or (part.value is checks.DELETE and not o.same):
                out.append(part)
                continue
            value = part.value
            if not o.same:
                said = (
                    value
                    if isinstance(value, checks.Described)
                    else json.dumps(value, ensure_ascii=False)
                )
                value = checks.Described(f"a value that makes {part.key} {said}")
            out += self.edits(o, value)
        return _fix(out)

    @staticmethod
    def edits(origin: Origin, value) -> list:
        """The steps that set each of ORIGIN's keys to VALUE (only its OFF keys, when it has them and VALUE is false),
        then what that does (its WHAT)."""
        keys = origin.off if value is False and origin.off else origin.keys
        out = []
        for i, key in enumerate(keys):
            if i:
                out.append(", and" if i == len(keys) - 1 else ",")
            out.append(checks.Edit(key, origin.file, value))
        return out + ([f"({origin.what})"] if origin.what else [])

    def flag(self, name: str) -> bool:
        return renames.is_on(self.flat.get(name))

    def flags(self, *names) -> str:
        """The older switches NAMES, as production has them, each quoted as code: "`review-posting true`, `holds
        false`"."""
        return ", ".join(
            _code(f"{n} {_word(self.flat[n])}") for n in names if n in self.flat
        )

    def on(self, path: str) -> bool:
        return deploy_tree.get(self.values, path) is True

    def got(self, path: str) -> str:
        """The file's value at PATH, as the report shows it."""
        return _word(deploy_tree.get(self.values, path))

    def feature_mode(self, feature: str, on: bool) -> str:
        """FEATURE's mode while it is ON: the configuration's (renames.moves makes one that is on with no mode live, as
        it sends today whenever it is on), or by layout 1's rule (from_layout1.feature_mode), live too, for a null one;
        one that is off is written with no mode. What the report says of it: ', mode "live"', or "" for none.
        """
        given = self.take(self.internal(f"{feature}.mode"))
        mode = (
            (from_layout1.feature_mode(on) if given is MISSING else given)
            if on
            else None
        )
        if mode is None:
            self.leave_out(f"{feature}.mode")
            return ""
        self.put(f"{feature}.mode", mode)
        return f", mode {_shown(mode)}"

    def one_address(self, path: str, keys: tuple, two: str, address=None) -> dict:
        """The file's key PATH, the one address that KEYS, two keys of production's, name: the one that is set, or the
        two when they name one place (from_layout1.same_address), without its trailing "/". ADDRESS, a rule of layout
        1's (from_layout1.jira_address), gives the address each value stands for, compared and carried. When they
        name two, a stop: they TWO. The keys that are set, with their values."""
        given = {k: v for k in keys if (v := self.take(k)) not in (MISSING, "")}
        self.came_from(path, *(given or keys[:1]))
        values = [address(v) if address else v for v in given.values()]
        if len(values) > 1 and not from_layout1.same_address(*values):
            self.stop(f"{keys[0]} and {keys[1]} {two}", _one_address(*keys))
        elif values:
            self.put(path, str(values[0]).strip().rstrip("/"))
        return given

    # the run

    def run(self) -> Translation:
        self.say(
            READ,
            f"config.json: {wrapping.counted(len(_leaves(self.raw)), 'key')}; claude_env's shape: "
            f"{wrapping.counted(len(self.shape.variables), 'variable')} and "
            f"{wrapping.counted(len(self.shape.headers), 'custom header')} (names and settings only)",
        )
        for old, new, line in self.moved:
            # the older feature switches have a section of their own
            if old.startswith("features.") or old in renames.SWITCH_KEYS:
                continue
            if old in renames.DROPPED and _off(deploy_tree.get(self.raw, old)):
                self.dropped_off += 1
            else:
                self.say(READ if new else REMOVED, line)
        self.carried()
        self.tick_interval()
        self.container()
        self.updates()
        self.repository()
        self.host()
        self.resources()
        self.review_automation()
        self.merge_queue()
        self.admin_dashboard()
        self.team_dashboard()
        self.alerts()
        self.tickets()
        self.other_switches()
        self.spend()
        self.gpu_tests()
        self.efforts()
        self.claude_models()
        self.first_review_pass()
        self.optional_models()
        self.unbound_worker()
        self.model_bindings()
        self.claude_code()
        self.logins()
        self.other_keys()
        text = self.written()
        self.say(
            NEXT,
            "for a second deployment on this host, beside production: set container.name and "
            "features.dashboards.admin.port in the file to values of its own, then run the next step",
        )
        report = self.report()
        if self.stops:
            raise Untranslatable(self.stops, report)
        return Translation(text, report)

    def carried(self) -> None:
        """Every deploy key with an x-internal and no rule of its own takes the value at that configuration key; a key
        production does not have, and a new key with no x-internal, takes the package's default.
        """
        for path, _node, _required in settings_file.schema_keys(self.schema):
            if path in settings_file.HEADER_KEYS or path in SPECIAL:
                continue
            if "*" in path.split(".") or "[]" in path:
                continue
            internal = self.internals.get(path)
            if not isinstance(internal, str):
                self.defaulted[path] = f"{path} (a new key)"
                continue
            self.carry(path, internal)

    def carry(self, path: str, internal: str) -> bool:
        """The deploy key PATH takes the value at the configuration key INTERNAL, as it is; when production does not
        have it, the package's default, and the report says so. Whether PATH took production's value.
        """
        value = self.take(internal)
        if value is MISSING:
            self.defaulted[path] = f"{path} (config.json has no {internal})"
            return False
        self.put(path, copy.deepcopy(value))
        return True

    def tick_interval(self) -> None:
        """The tick interval (TICK): the value at the first key its x-internal names (claude.tick_min, the tick the
        session runs at), as `carried` takes a key's. The keys it names after that one are merged into it
        (tick_interval_min, the tick loop's): `other_keys` names production's own value of each.
        """
        if not self.carry(TICK, self.internal(TICK)[0]):
            self.leave_out(
                TICK
            )  # its line among the keys at the package's default says why

    def container(self) -> None:
        """container.name: its x-internal (deployment.project), as text, blank meaning "rkmon"; image.name and
        image.tag: the one key both name (deployment.image), <name>:<tag>, split by deploy_tree.image_parts, a blank
        name or tag meaning what it means by layout 1's rules (from_layout1's container_name, image_name and
        image_tag)."""
        key = self.internal("container.name")
        project = self.take(key)
        name = from_layout1.container_name(_text(project))
        self.put("container.name", name)
        self.came_from("container.name", key)
        if not _text(project):
            self.say(
                FORMS,
                f"container.name {_shown(name)}: what a blank {key} means, so the deployment keeps its name, its "
                "volumes and its state",
            )
        key = self.internal("image.name")
        given = self.take(key)
        image, tag = deploy_tree.image_parts(_text(given))
        self.put("image.name", from_layout1.image_name(image))
        self.put("image.tag", from_layout1.image_tag(tag, name))
        for path in ("image.name", "image.tag"):
            self.came_from(path, key, same=False)
        if not (image and tag):
            self.say(
                FORMS,
                f"image.name {self.got('image.name')} and image.tag {self.got('image.tag')}: the image production "
                f"runs, as {key} {_shown(given)} names it",
            )

    def updates(self) -> None:
        """updates: where newer code is fetched from, from the keys its x-internals name (update.repo and
        update.branch), by layout 1's rule (from_layout1.updates_section): the <org>/<repo> of a GitHub repository,
        and the branch. When either is blank or names none, the section is left out, as the file leaves it out to
        update only from a folder or an archive, and the line says why, and which value is not carried.
        """
        repo_key = self.internal("updates.repository")
        branch_key = self.internal("updates.branch")
        repo, branch = self.take(repo_key), self.take(branch_key)
        section = from_layout1.updates_section(_for_rule(repo), _for_rule(branch))
        self.came_from("updates.repository", repo_key, same=False)
        self.came_from("updates.branch", branch_key)
        if section:
            for key, value in section.items():
                self.put(f"updates.{key}", value)
            self.say(
                FORMS,
                f"updates.repository {_shown(section['repository'])} and updates.branch "
                f"{_shown(section['branch'])}: {repo_key} {_shown(repo)}, {branch_key} {_shown(branch)}",
            )
            return
        self.leave_out("updates")
        why, left = [], []
        for key, value, named, what in (
            (
                repo_key,
                repo,
                from_layout1.github_repository(_for_rule(repo)),
                "GitHub repository",
            ),
            (branch_key, branch, from_layout1.branch_name(_for_rule(branch)), "branch"),
        ):
            if named:
                left.append(f"{key} {_shown(value)}")
            elif _blank(value):
                why.append(f"{key} is blank")
            else:
                why.append(f"{key} {_shown(value)} names no {what}")
        self.say(
            FORMS,
            f"updates: left out ({' and '.join(why)})"
            + (f"; {left[0]} is not carried" if left else ""),
        )

    def repository(self) -> None:
        """shared_label_folders: each of its x-internal's labels (scope.path_labels), with repository.folder's
        (rocke_path) as its folder, the one it counts for (from_layout1's labels and shared_label_folders).
        """
        key = self.internal("repository.shared_label_folders")
        value = self.take(key)
        if value is MISSING:
            return
        labels = from_layout1.labels(value)
        if labels is None:
            self.stop(f"{key} is no list of labels", _edit(key, "a list of labels"))
            return
        where = self.internal("repository.folder")
        folder = self.peek(where)
        if folder is MISSING:
            return  # the file's own check stops at repository.folder, which it requires
        self.put(
            "repository.shared_label_folders",
            from_layout1.shared_label_folders(labels, folder),
        )
        self.came_from("repository.shared_label_folders", key, same=False)
        if labels:
            self.say(
                FORMS,
                f"repository.shared_label_folders: {key} {wrapping.joined([_shown(x) for x in labels])}, each with "
                f"{where}'s folder {_shown(folder)}, the one folder it counts for",
            )

    def host(self) -> None:
        """ramdisk_gib: the size (ramdisk.size_gb) while the switch (ramdisk.enabled) is true, else 0, the two keys of
        its x-internal; a user or group id by layout 1's rule (from_layout1.user_id): a blank one is null, and text of
        digits a whole number."""
        switch, size_key = self.internal("host.resources.ramdisk_gib")
        enabled, size = self.take(switch), self.take(size_key)
        gib = _whole(deployment.resources(self.cfg)["ramdisk_gb"])
        self.put("host.resources.ramdisk_gib", gib)
        self.came_from("host.resources.ramdisk_gib", switch, size_key, same=False)
        if (enabled, size) != (MISSING, MISSING):
            self.say(
                FORMS,
                f"host.resources.ramdisk_gib {gib}: {switch} {_word(enabled)}, {size_key} {_word(size)}",
            )
        for path in ("host.users.monitor_uid", "host.users.monitor_gid"):
            key = self.internal(path)
            v = self.given(key)
            self.came_from(path, key)
            if v is not MISSING:
                # a value that is no id as it is, for the file's own check to judge
                self.put(path, from_layout1.user_id(v))

    def resources(self) -> None:
        """The container's memory and its CPUs, which `carried` gives production's values, or the package's defaults
        where production has none: a default less than the file's own rules take gives way to the least value they
        take (`least_resources`), and its line among the keys that take the package's default says which, and why.
        After `host`, which gives the RAM disk."""
        for path, (value, least, why) in least_resources(self.values).items():
            if path in self.defaulted and value < least:
                self.put(path, least)
                self.defaulted[path] = (
                    f"{path} {least}: {CONFIG_FILE} has no {self.internal(path)}, and the package's {value} is {why}"
                )

    # the features

    def mode(self, feature: str) -> str:
        """A feature's mode as the file has it, "" when it is off."""
        mode = deploy_tree.get(self.values, f"{feature}.mode")
        return mode if self.on(f"{feature}.enabled") and isinstance(mode, str) else ""

    def why(self, *names) -> str:
        """The older switches that decided a feature, as the report's ' (`name value`, ...)': the ones of SWITCH_KEYS
        production has, then NAMES of its feature map; "" for none."""
        found = [
            _code(f"{k} {_word(deploy_tree.get(self.raw, k))}")
            for k in renames.SWITCH_KEYS
            if deploy_tree.get(self.raw, k) is not MISSING
        ]
        found += [self.flags(*names)] if self.flags(*names) else []
        return f" ({', '.join(found)})" if found else ""

    def review_automation(self) -> None:
        """Review automation: its switch and mode (renames.moves). headless-claude and overlap-management have no key:
        the Claude session runs while review automation is on, and overlap notices go out while it is live.
        """
        self.say(
            SWITCHES,
            f"{RA}: enabled {self.got(f'{RA}.enabled')}, mode {self.got(f'{RA}.mode')}"
            + self.why("review-posting"),
        )
        self.keyless(
            "headless-claude",
            "the Claude session runs while review automation is on",
            same=self.flag("headless-claude") == self.on(f"{RA}.enabled"),
        )
        self.keyless(
            "overlap-management",
            "overlap notices go out while review automation is live",
            same=self.flag("overlap-management") == (self.mode(RA) == "live"),
        )

    def keyless(self, name: str, what: str, same=None) -> None:
        """A switch with no key in the file, WHAT saying what decides it now: the older switch NAME as production has
        it, said when production has it. SAME, for a switch a feature's mode decides today, is whether the switch says
        what the mode gives: "as today", and when it does not, today's monitor does not read the switch either.
        """
        if name not in self.flat:
            return
        line = f"{self.flags(name)}: no key; {what}"
        if same is not None:
            line += ", as today" + (
                "" if same else f"; today's monitor does not read {_code(name)} either"
            )
        self.say(SWITCHES, line)

    def merge_queue(self) -> None:
        """The merge queue, by layout 1's rules (from_layout1): auto_merge "enabled" is true and "hints_only" false
        (auto_merge_enabled); automerge_guard_own false or blank is "alert_only" and true "same_as_others" (guard_own);
        the queue and drift notices by their own rules (notices); the develop guard is incident-issues,
        stability-issues and fix-links, which move together, and its labels are the ones today's guard puts on a
        develop issue (ci_red_labels, stability_label).
        """
        key = self.internal(f"{MQ}.auto_merge.enabled")
        am = self.take(key)
        if am is not MISSING:
            enabled = from_layout1.auto_merge_enabled(am)
            if enabled is None:
                self.stop(
                    f'{key} is {_shown(am)}: "enabled" or "hints_only" carries',
                    _edit(key, '"enabled" or "hints_only"'),
                )
            else:
                self.put(f"{MQ}.auto_merge.enabled", enabled)
        self.say(
            SWITCHES,
            f"{MQ}: enabled {self.got(f'{MQ}.enabled')}, mode {self.got(f'{MQ}.mode')}; auto_merge.enabled "
            f"{self.got(f'{MQ}.auto_merge.enabled')}, auto_hold.enabled {self.got(f'{MQ}.auto_hold.enabled')}"
            + self.why("auto-merge", "holds"),
        )
        key = self.internal(f"{MQ}.auto_merge_guard.on_your_pull_requests")
        own = self.given(key)
        if own is not MISSING:
            word = from_layout1.guard_own(own)
            if word is None:
                self.stop(
                    f"{key} is {_shown(own)}: true or false carries",
                    _edit(key, "true or false"),
                )
            else:
                self.put(f"{MQ}.auto_merge_guard.on_your_pull_requests", word)
                self.say(
                    FORMS,
                    f"{MQ}.auto_merge_guard.on_your_pull_requests {_shown(word)}: {key} {_word(own)}"
                    + (
                        ""
                        if isinstance(own, bool)
                        else ", which today's monitor reads as false"
                    ),
                )
        # production's guard says what the merge queue's mode gives when it is on while the queue is on, and "act"
        # while the queue is live
        live = self.mode(MQ) == "live"
        self.keyless(
            "auto-merge-guard",
            "the auto-merge guard runs with the merge queue: it alerts in prototype and does its GitHub work in live",
            same=(self.flat.get("auto-merge-guard") == "act") == live
            and self.flag("auto-merge-guard") == bool(self.mode(MQ)),
        )
        self.notices(live)
        trio = ("incident-issues", "stability-issues", "fix-links")
        path = f"{GUARD}.enabled"
        key = self.internal(path)
        today = self.take(key)
        guard = today
        if any(n in self.flat for n in trio):
            self.came_from(path, *(_older(n) for n in trio), what=GUARD_DOES)
            values = {self.flag(n) for n in trio}
            if len(values) > 1:
                each = [
                    self.flags(n) if n in self.flat else f"{_code(n)} absent (off)"
                    for n in trio
                ]
                self.stop(
                    f"{', '.join(each)}: the three move together, as the develop guard's one switch, and they differ",
                    checks.Steps(
                        (
                            *self.edits(self.origin(path), True),
                            ", or set each of them to false",
                        )
                    ),
                )
            else:
                guard = values.pop()
                self.say(
                    SWITCHES,
                    f"{path} {_word(guard)} ({self.flags(*trio)})"
                    + (
                        f"; today's monitor does not read them: its develop guard is {_word(today)}, the merge "
                        "queue's switch"
                        if guard != today
                        else ""
                    ),
                )
        elif self.production_key(key):
            self.came_from(path, self.production_key(key), what=GUARD_DOES)
        if isinstance(guard, bool):
            self.put(path, guard)
        self.guard_labels()

    def notices(self, live: bool) -> None:
        """The queue and drift notices' switches (from_layout1.notices). The queue notices': their own switch, at its
        x-internal, which the code reads; without it, the merge queue's switch, as today's monitor posts them while
        the merge queue is live whatever its feature map's queue-notices says, so production's stay on. The drift
        notices': their own switch, at its x-internal; without it, production's own, in its feature map; without
        that, or in the shape after the map, as today's monitor posts them, with the merge queue and review
        automation too, as Claude writes them. Today's monitor posts them while the merge queue is LIVE, whatever the
        map's switch says: one that stops them is a change the report names."""
        queue_on, ra_on = self.on(f"{MQ}.enabled"), self.on(f"{RA}.enabled")
        names = ("queue-notices", "drift-notices")
        paths = (f"{MQ}.queue_notices.enabled", f"{MQ}.drift_notices.enabled")
        keys = tuple(self.internal(path) for path in paths)
        given = tuple(self.take(key) for key in keys)
        own = [
            None if given[0] is MISSING else given[0],
            (
                given[1]
                if given[1] is not MISSING
                else self.flag(names[1]) if names[1] in self.flat else None
            ),
        ]
        rules = (
            ("the merge queue's switch", "the merge queue is live"),
            (
                "the merge queue's switch and review automation's, as Claude writes them",
                "the merge queue is live and review automation is on",
            ),
        )
        values = from_layout1.notices(queue_on, ra_on, *own)
        posted = (live, live and ra_on)
        for path, key, name, mine, value, (rule, when), today, switch in zip(
            paths,
            keys,
            names,
            own,
            values,
            rules,
            posted,
            given,
        ):
            self.put(path, value)
            if switch is not MISSING:
                self.came_from(path, *(k for k in (self.production_key(key),) if k))
                self.say(SWITCHES, f"{path} {_word(value)} ({key} {_word(switch)})")
                continue
            if mine is not None:
                self.came_from(path, _older(name))
                line = f"{path} {_word(value)} ({self.flags(name)})"
            elif (
                name in self.flat
            ):  # a switch of the map's that decides nothing, as today
                line = (
                    f"{path} {_word(value)} ({self.flags(name)}): {rule}, as today"
                    + (
                        ""
                        if self.flag(name) == value
                        else f"; today's monitor does not read {_code(name)} either"
                    )
                )
            elif self.flat:
                line = f"{path} {_word(value)} ({_code(name)} absent): {rule}"
            else:
                line = f"{path} {_word(value)}: {rule}"
            if today and not value:
                line += f"; today's monitor posts them while {when}, whatever {_code(name)} says: this file stops them"
            self.say(SWITCHES, line)

    def guard_labels(self) -> None:
        """The develop guard's labels, the ones today's develop guard puts on a develop issue: on a red CI, the first of
        its x-internal's labels (incident_issue.labels), then the others with the label it adds beside the first
        (from_layout1.ci_red_labels); on any other, the stability label (from_layout1.stability_label). A blank one is
        the one the develop guard puts for none. Each group is written while the develop guard is in use, or when it is
        production's own (from_layout1.guard_labels_written); else it is left out, as a feature that is off and keeps
        its defaults is written as its switch alone."""
        # the one key both labels' x-internal names
        key = self.internal(f"{GUARD}.ci_red_label")
        value = self.take(key)
        stability_key = self.internal(f"{GUARD}.stability_label")
        stability = self.take(stability_key)
        queue_on, guard_on = self.on(f"{MQ}.enabled"), self.on(f"{GUARD}.enabled")
        ci_red_why, stability_why = from_layout1.guard_labels_written(
            queue_on, guard_on, _for_rule(value), _for_rule(stability)
        )
        # what turns the develop guard off, for the line of a group left out
        off = (
            f"{MQ}.enabled is {self.got(f'{MQ}.enabled')}"
            if not queue_on
            else f"{GUARD}.enabled is {self.got(f'{GUARD}.enabled')}"
        )
        for part in ("ci_red_label", "ci_red_extra_labels"):
            self.came_from(f"{GUARD}.{part}", key, same=False)
        names = from_layout1.labels(_for_rule(value))
        if names is None:
            self.stop(f"{key} is no list of labels", _edit(key, "a list of labels"))
        elif not ci_red_why:
            for part in ("ci_red_label", "ci_red_extra_labels"):
                self.leave_out(f"{GUARD}.{part}")
            self.say(
                FORMS,
                f"{GUARD}.ci_red_label and ci_red_extra_labels: left out, as {off} and {key} is "
                + ("not set" if value is MISSING else _shown(value))
                + ", which gives the labels today's guard puts by default; a feature that is off and keeps its "
                "defaults is written as its switch alone",
            )
        else:
            first, extra = from_layout1.ci_red_labels(names)
            self.put(f"{GUARD}.ci_red_label", first)
            self.put(f"{GUARD}.ci_red_extra_labels", extra)
            given = (
                f"{key} {_shown(value)}"
                if names
                else f"the labels today's guard puts on a develop issue, as {key} is "
                + ("not set" if value is MISSING else _shown(value))
            )
            self.say(
                FORMS,
                f"{GUARD}.ci_red_label {_shown(first)} and ci_red_extra_labels {_shown(extra)}: {given}"
                + (
                    ""
                    if from_layout1.CI_RED in names
                    else f", with {_shown(from_layout1.CI_RED)}, the label the code adds to each CI-red issue"
                ),
            )
        self.came_from(f"{GUARD}.stability_label", stability_key)
        if not stability_why:
            self.leave_out(f"{GUARD}.stability_label")
            self.say(
                FORMS,
                f"{GUARD}.stability_label: left out, as {off} and {stability_key} is "
                + ("not set" if stability is MISSING else _shown(stability))
                + ", which gives the label today's guard puts by default; a feature that is off and keeps its "
                "defaults is written as its switch alone",
            )
        elif _blank(stability):
            label = from_layout1.stability_label(_for_rule(stability))
            self.put(f"{GUARD}.stability_label", label)
            self.say(
                FORMS,
                f"{GUARD}.stability_label {_shown(label)}: the label today's guard puts on any other develop issue, "
                f"as {stability_key} is "
                + ("not set" if stability is MISSING else "blank"),
            )
        else:
            # a value that is no label as it is, for the file's own check to judge
            self.put(f"{GUARD}.stability_label", copy.deepcopy(stability))

    def admin_dashboard(self) -> None:
        """The admin dashboard: its port, 0 being the default port; it listens on 127.0.0.1 only, so another address
        production publishes it on (from_layout1.admin_address_allowed) is reported."""
        self.say(
            SWITCHES,
            f"features.dashboards.admin.enabled {self.got('features.dashboards.admin.enabled')}",
        )
        key = self.internal("features.dashboards.admin.port")
        port = self.take(key)
        self.came_from("features.dashboards.admin.port", key)
        if port in (0, ""):
            self.put("features.dashboards.admin.port", from_layout1.ADMIN_PUBLIC_PORT)
            self.say(
                FORMS,
                f"features.dashboards.admin.port {from_layout1.ADMIN_PUBLIC_PORT}: what {key} {_word(port)} means",
            )
        elif port is not MISSING:
            self.put("features.dashboards.admin.port", port)
        host = self.take("admin_dashboard.publish_host")
        if not from_layout1.admin_address_allowed(_for_rule(host)):
            why = removed_keys.PACKAGE_SET["admin_dashboard.publish_host"]
            self.say(
                FORMS, f"admin_dashboard.publish_host {_shown(host)}: no key; {why}"
            )

    def team_dashboard(self) -> None:
        """The team dashboard: live while it is on (it is pushed today whenever it is on), and with no mode while it is
        off (`feature_mode`); its address one of publish.html.url and team_dashboard.url (`one_address`); its
        delivery, publish or serve, none being publish (from_layout1.team_delivery). Each delivery writes its own block
        and leaves the other's out (from_layout1.team_block; `team_published`, `team_served`), as the file takes no
        other."""
        on = self.on(f"{TEAM}.enabled")
        mode = self.feature_mode(TEAM, on)
        # the address: one of the keys its x-internal names (publish.html.url, and team_dashboard.url, merged into it)
        self.one_address(
            f"{TEAM}.public_url",
            tuple(self.internal(f"{TEAM}.public_url")),
            "name two addresses, and the team dashboard has one",
        )
        key = self.internal(f"{TEAM}.delivery")
        delivery = self.take(key)
        self.came_from(f"{TEAM}.delivery", key)
        chosen = from_layout1.team_delivery(_for_rule(delivery))
        block = from_layout1.team_block(chosen)
        if block is None:
            self.stop(
                f'{key} is {_shown(delivery)}: "publish" or "serve" carries',
                _edit(key, '"publish" or "serve"'),
            )
        else:
            self.put(f"{TEAM}.delivery", chosen)
        if _blank(delivery):
            self.say(
                FORMS,
                f"{TEAM}.delivery {_shown(chosen)}: what {key} {_word(delivery)} means",
            )
        self.say(
            SWITCHES,
            f"{TEAM}: enabled {_word(on)}{mode}, delivery {self.got(f'{TEAM}.delivery')}"
            + (
                f" ({self.flags('team-dashboard')})"
                if "team-dashboard" in self.flat
                else ""
            )
            + ("; it is pushed today whenever it is on" if on else ""),
        )
        if block == from_layout1.SERVE_DELIVERY:
            self.team_served(on)
        else:
            self.team_published()
        # the web host's key is a secret of its own: the logins' section says how it is copied
        self.read.add("publish.html.key")

    def team_published(self) -> None:
        """The team dashboard published: the publish block (publish.html's host, user and site), its priority requests
        on while the requests file (publish.html.requests) names one, as layout 1's rule reads it
        (from_layout1.priority_requests: false and blank name none). The page (publish.html.dest) and the requests file
        are derived from the site folder (derived.site_page, derived.site_requests), so a page or a requests file at
        another place stops it, and so does a blank page, which today's monitor copies to the home folder of the web
        host's user (publish.html.user); a key production does not have is a blank one. Serve's address and port are
        left out, as today's monitor reads them only while it serves the page (in the docker/.env it writes).
        """
        for key, internal in self.block("publish"):
            value = self.take(internal)
            self.came_from(f"{TEAM}.publish.{key}", internal)
            if value is not MISSING:
                self.put(f"{TEAM}.publish.{key}", copy.deepcopy(value))
        requests_key = self.internal(f"{TEAM}.priority_requests.enabled")
        site_key = self.internal(f"{TEAM}.publish.site_folder")
        # as production has them, for the report
        given = {
            requests_key: self.take(requests_key),
            site_key: self.peek(site_key),
            PAGE: self.take(PAGE),
        }
        requests, dest, site = (_text(given[k]) for k in (requests_key, PAGE, site_key))
        self.put(
            f"{TEAM}.priority_requests.enabled",
            from_layout1.priority_requests(_for_rule(given[requests_key])),
        )
        self.came_from(f"{TEAM}.priority_requests.enabled", requests_key, same=False)
        # with no site folder, or one the report does not show, the file's own check names the key that gives it
        if site and not _unshown(json.dumps(site, ensure_ascii=False), "value"):
            page, requests_file = derived.site_page(site), derived.site_requests(site)
            if not dest:
                # today's monitor copies the page to the user's place on the web host that the page names: with none,
                # to that user's home folder
                self.stop(
                    f"{PAGE} {_shown(given[PAGE])}: today's monitor copies the page to the home folder of "
                    f"{self.internal(f'{TEAM}.publish.username')} on the web host, not to {_shown(page)}, the page "
                    f"{site_key} {_shown(given[site_key])} gives, and the page's place is derived from the site folder",
                    checks.Edit(PAGE, CONFIG_FILE, page),
                )
            elif dest != page:
                self.stop(
                    f"{PAGE} {_shown(given[PAGE])} is not {_shown(page)}, the page {site_key} "
                    f"{_shown(given[site_key])} gives, and the page's place is derived from the site folder",
                    checks.Edit(PAGE, CONFIG_FILE, page),
                )
            if requests and requests != requests_file:
                self.stop(
                    f"{requests_key} {_shown(given[requests_key])} is not {_shown(requests_file)}, the requests file "
                    f"{site_key} {_shown(given[site_key])} gives, and the requests file's place is derived from the "
                    "site folder",
                    checks.Edit(requests_key, CONFIG_FILE, requests_file),
                )
        if dest or requests:
            self.say(
                FORMS,
                f"{PAGE} {_shown(given[PAGE])} and {requests_key} {_shown(given[requests_key])}: no keys; each is "
                f"derived from {TEAM}.publish.site_folder, and {TEAM}.priority_requests.enabled is "
                f"{self.got(f'{TEAM}.priority_requests.enabled')}",
            )
        self.other_block(
            "serve",
            "publish",
            [internal for _key, internal in self.block("serve")],
            "serves",
        )

    def team_served(self, on: bool) -> None:
        """The team dashboard served here: the serve block, a blank address being the default one and port 0 the one
        it serves on inside; its priority requests on while it is on, as the page served here takes them today
        (publish.takes_requests). The publish block is left out, as today's monitor reads it only while it publishes
        the page."""
        for key, internal in self.block("serve"):
            self.came_from(f"{TEAM}.serve.{key}", internal)
        address = self.take(self.internal(f"{TEAM}.serve.address"))
        if address not in (MISSING, ""):
            self.put(f"{TEAM}.serve.address", address)
        key = self.internal(f"{TEAM}.serve.port")
        port = self.take(key)
        if port in (0, ""):
            # what production publishes for 0 (in the docker/.env it writes): the port it serves on inside
            inside = container_layout.TEAM_PORT
            self.put(f"{TEAM}.serve.port", inside)
            self.say(
                FORMS,
                f"{TEAM}.serve.port {inside}: what {key} {_word(port)} means",
            )
        elif port is not MISSING:
            self.put(f"{TEAM}.serve.port", port)
        self.put(f"{TEAM}.priority_requests.enabled", on)
        if on:
            self.say(
                FORMS,
                f"{TEAM}.priority_requests.enabled true: the page served here takes requests whenever it is on, as "
                "today",
            )
        internals = [internal for _key, internal in self.block("publish")]
        requests_key = self.internal(f"{TEAM}.priority_requests.enabled")
        self.other_block(
            "publish",
            "serve",
            (*internals, PAGE, requests_key),
            "publishes",
        )

    def block(self, name: str) -> list:
        """The team dashboard's block NAME as the schema has it: (each key under it, the configuration key its
        x-internal names)."""
        prefix = f"{TEAM}.{name}."
        return [
            (path[len(prefix) :], self.internal(path))
            for path in self.internals
            if path.startswith(prefix) and isinstance(self.internals[path], str)
        ]

    def other_block(self, block: str, delivery: str, internals, does: str) -> None:
        """The team dashboard's BLOCK, which DELIVERY takes no part of: left out of the file, and each of the
        configuration keys INTERNALS that production has is named, with why it is not carried (today's monitor reads it
        only while it DOES the page)."""
        self.leave_out(f"{TEAM}.{block}")
        had = [
            f"{internal} {_shown(value)}"
            for internal in internals
            if (value := self.take(internal)) is not MISSING
        ]
        if had:
            self.say(
                FORMS,
                f"{wrapping.joined(had)}: not carried; delivery {_shown(delivery)} takes no {block} block, and "
                f"today's monitor reads {'it' if len(had) == 1 else 'them'} only while it {does} the page",
            )

    def alerts(self) -> None:
        """Each alert's switch: chat-alerts and needs-you-alerts, or in the shape after them the one switch both
        alerts' x-internals name first (features.alerts.enabled), which turns on both (from_layout1.alert_switches),
        and each kind's own after it: a kind is on while both are, as the code reads them (renames.moves sets
        both kinds on beside a one switch that has neither). The mode is live while one is on (`feature_mode`).
        Production's monitor sends both kinds while either older switch is on: it joins the two into that one switch,
        which gates both. The file keeps production's two switches, so when they differ the report says
        what changes.
        """
        keys = {
            alert: self.internal(f"{ALERTS}.{alert}.enabled")
            for alert in ("develop_breaks", "needs_you")
        }
        # the one switch, which renames.moves sets from the older ones when production has them, then each kind's
        (key,) = {k[0] for k in keys.values()}
        own = {alert: self.take(k[1]) for alert, k in keys.items()}
        value = self.take(key)
        older = "chat-alerts" in self.flat or "needs-you-alerts" in self.flat
        if older:
            breaks, needs_you = self.flag("chat-alerts"), self.flag("needs-you-alerts")
            why = f" ({self.flags('chat-alerts', 'needs-you-alerts')})"
            for alert, name in (
                ("develop_breaks", "chat-alerts"),
                ("needs_you", "needs-you-alerts"),
            ):
                self.came_from(f"{ALERTS}.{alert}.enabled", _older(name))
        else:
            switches = from_layout1.alert_switches(value is True)
            on = {alert: switches[alert] and own[alert] is True for alert in keys}
            breaks, needs_you = on["develop_breaks"], on["needs_you"]
            # production's own keys: the one switch, and a kind's own where production has it, which alone a fix
            # that turns the kind off edits
            given = [
                k
                for k in (key, *(k[1] for k in keys.values()))
                if deploy_tree.get(self.raw, k) not in (MISSING, None)
            ]
            for alert, (one, its_own) in keys.items():
                if its_own in given:
                    self.came_from(
                        f"{ALERTS}.{alert}.enabled", one, its_own, off=(its_own,)
                    )
                else:
                    self.came_from(f"{ALERTS}.{alert}.enabled", one)
            why = (
                f" ({', '.join(f'{k} {_word(self.peek(k))}' for k in given)})"
                if given
                else ""
            )
        self.put(f"{ALERTS}.develop_breaks.enabled", breaks)
        self.put(f"{ALERTS}.needs_you.enabled", needs_you)
        mode = self.feature_mode(ALERTS, breaks or needs_you)
        if breaks != needs_you and older:
            sent = "the develop-break alerts" if breaks else "the needs-you alerts"
            as_today = f"; today's monitor sends both while either is on; this file sends only {sent}"
        else:
            as_today = (
                "; they are sent today whenever they are on"
                if breaks or needs_you
                else ""
            )
        self.say(
            SWITCHES,
            f"{ALERTS}: develop_breaks.enabled {_word(breaks)}, needs_you.enabled {_word(needs_you)}{mode}{why}"
            f"{as_today}",
        )

    def other_switches(self) -> None:
        """The other features' switches: the remote GPU tests' and the weekly report's, carried (their x-internal);
        services, whose feature leaves the package with the install without a container, and jira-mcp, whose
        feature is gone with its keys (renames.DROPPED); the keys of either are removed keys.
        """
        for path, name in (
            (f"{RA}.remote_gpu_tests.enabled", "remote-gpu-tests"),
            ("features.weekly_report.enabled", "weekly-report"),
        ):
            self.say(
                SWITCHES,
                f"{path} {self.got(path)}"
                + (f" ({self.flags(name)})" if name in self.flat else ""),
            )
        self.keyless("services", renames.OLD_FEATURES["services"][1])
        self.keyless("jira-mcp", "no feature uses the JIRA MCP server")

    def tickets(self) -> None:
        """Each ticket type's switch, by layout 1's rule (from_layout1.ticket_types): the keys of its x-internal,
        ticket creation's switch AND the type's trigger, a trigger production does not have being on, as today's
        monitor reads it; the mode live while one is on (`feature_mode`); the components split at commas
        (from_layout1.jira_components); the JIRA site one of the keys its x-internal names (atlassian.site, and
        jira.url, merged into it: `one_address`), a bare site name, the atlassian CLI's, being the
        address it stands for (from_layout1.jira_address)."""
        kinds = ("follow_up_tickets", "found_issue_tickets")
        names = ("follow-up-tickets", "found-issue-tickets")
        # each type's x-internal: ticket creation's switch, one for both, then the type's trigger
        (section,) = {self.internal(f"{TICKETS}.{t}.enabled")[0] for t in kinds}
        triggers = [self.internal(f"{TICKETS}.{t}.enabled")[1] for t in kinds]
        on = self.take(section) is True
        given = [self.take(k) for k in triggers]
        follow, found = from_layout1.ticket_types(
            on, *(None if v is MISSING else v is True for v in given)
        )
        self.put(f"{TICKETS}.follow_up_tickets.enabled", follow)
        self.put(f"{TICKETS}.found_issue_tickets.enabled", found)
        mode = self.feature_mode(TICKETS, follow or found)
        for kind, name in zip(kinds, names):
            path = f"{TICKETS}.{kind}.enabled"
            if name in self.flat:
                self.came_from(path, _older(name))
            else:
                # ticket creation's switch and the type's trigger: the trigger alone turns the type off
                keys = self.internal(path)
                self.came_from(path, *keys, off=keys[-1:])
        unset = [k for k, v in zip(triggers, given) if v is MISSING] if on else []
        self.say(
            SWITCHES,
            f"{TICKETS}: follow_up_tickets.enabled {_word(follow)}, found_issue_tickets.enabled {_word(found)}{mode}"
            + (f" ({self.flags(*names)})" if self.flags(*names) else "")
            + "".join(
                f"; {k} is not set, which today's monitor reads as on" for k in unset
            )
            + ("; tickets are filed today whenever a type is on" if mode else ""),
        )
        key = self.internal(f"{TICKETS}.components")
        component = self.take(key)
        self.came_from(f"{TICKETS}.components", key, same=False)
        if component is not MISSING:
            parts = from_layout1.jira_components(component)
            if parts is None:
                self.stop(
                    f"{key} is {_shown(component)}: the components are a text, with commas between them",
                    _edit(key, "the components: a text with commas between them"),
                )
            else:
                self.put(f"{TICKETS}.components", parts)
                if parts:
                    self.say(
                        FORMS,
                        f"{TICKETS}.components {_shown(parts)}: {key} {_shown(component)}, split at commas",
                    )
        path = "integrations.atlassian_cli.jira_url"
        keys = tuple(self.internal(path))
        sites = self.one_address(
            path,
            keys,
            "name two JIRA sites, and the file has one",
            from_layout1.jira_address,
        )
        addresses = [from_layout1.jira_address(v) for v in sites.values()]
        if len(sites) > 1 and from_layout1.same_address(*addresses):
            self.say(FORMS, f"{path}: {keys[0]} and {keys[1]}, which agree")
        if deploy_tree.get(self.values, path) is not MISSING:
            for (k, v), address in zip(sites.items(), addresses):
                if address is not v:
                    self.say(
                        FORMS,
                        f"{path} {self.got(path)}: {k} {_shown(v)}, the atlassian CLI's name for that site",
                    )
        enabled = self.take("atlassian.enabled")
        if enabled is False and sites:
            self.say(
                FORMS,
                "atlassian.enabled false: no key; the atlassian CLI is used while ticket creation is on",
            )

    def spend(self) -> None:
        """Spend, by layout 1's rules (from_layout1's model_prices and spend_enabled): on while production has prices
        (the x-internal of the prices, `prices`), each price under the same model id, its fields renamed
        (from_layout1.PRICE_FIELDS); off with none, as today each cost is unknown then (the package has no prices of
        its own). Any other field of a model's prices is a removed key, by migrate's rule (from_layout1.NOT_A_PRICE),
        beside the model's prices, carried. A note in the prices (a key that starts with _) is dropped.
        """
        key = self.internal(f"{RA}.spend.usd_per_million_tokens")
        prices = self.take(key)
        prices = {} if prices is MISSING else prices
        path = f"{RA}.spend.usd_per_million_tokens"
        self.came_from(f"{RA}.spend.enabled", key, same=False)
        self.came_from(path, key, same=False)
        if not isinstance(prices, dict):
            self.stop(
                f"{key} is no mapping of model ids to prices",
                _edit(key, "a mapping of model ids to prices"),
            )
            return
        names = from_layout1.PRICE_FIELDS
        out, notes = {}, []
        for model, price in prices.items():
            if str(model).startswith("_"):
                notes.append(_key(key, model))
                continue
            if not isinstance(price, dict):
                self.stop(
                    f"{_key(key, model)} is no mapping of prices",
                    _edit(_key(key, model), "a mapping of prices"),
                )
                continue
            # each field of the file's price, given or not, from the field of production's it is renamed from
            for field, renamed in names.items():
                self.came_from(f"{path}.{model}.{renamed}", _key(key, model, field))
            out[model], other = from_layout1.model_prices(price)
            for field, value in other.items():
                if field.startswith("_"):
                    notes.append(_key(key, model, field))
                else:
                    self.say(
                        REMOVED,
                        f"{_key(key, model, field)} {_shown(value)}: {from_layout1.NOT_A_PRICE}",
                    )
        self.put(f"{RA}.spend.enabled", from_layout1.spend_enabled(out))
        self.put(path, out)
        if out:
            self.say(
                FORMS,
                f"{RA}.spend.enabled true: production has {key} for {wrapping.counted(len(out), 'model')}; each "
                "price is the same, its fields renamed "
                f"({', '.join(f'{a} to {b}' for a, b in names.items() if a != b)})",
            )
        else:
            self.say(
                FORMS,
                f"{RA}.spend.enabled false: production has no {key}, so each cost is unknown today",
            )
        if notes:
            self.say(FORMS, f"notes in the prices, dropped: {', '.join(notes)}")

    def gpu_tests(self) -> None:
        """The remote GPU tests' environment (its x-internal, gpu.env), less the variable the repository's defaults set
        first in each job (from_layout1.gpu_environment)."""
        key = self.internal(f"{RA}.remote_gpu_tests.node.environment")
        env = self.take(key)
        self.came_from(f"{RA}.remote_gpu_tests.node.environment", key)
        if env is MISSING:
            return
        if not isinstance(env, dict):
            self.stop(
                f"{key} is no mapping of variables",
                _edit(key, "a mapping of variable names to their values"),
            )
            return
        kept = from_layout1.gpu_environment(env)
        self.put(f"{RA}.remote_gpu_tests.node.environment", kept)
        dropped = sorted(set(env) - set(kept))
        if dropped:
            self.say(
                FORMS,
                f"{RA}.remote_gpu_tests.node.environment: {', '.join(_key(k) for k in dropped)} left out, the "
                "repository's default, "
                "which each job sets first",
            )

    # the passes

    def efforts(self) -> None:
        """Each pass's effort by layout 1's chain (from_layout1.pass_effort), as today's monitor resolves it: its own
        (its x-internal; a key production does not have, its role's default on a pass that always runs); else, on an
        optional pass, "default", the effort of the pass in charge; else the base effort (claude.effort,
        from_layout1.base_effort). Where the chain ends in the model's own effort, which the file cannot name for a pass
        that always runs, its role's default, as the report says. The first review pass on the local worker takes the
        effort it ran on there (worker.think)."""
        given = self.given("claude.effort")
        base = from_layout1.base_effort(_for_rule(given))
        for key, _model in CLAUDE_PASSES:
            own = self.internal(f"{key}.effort")
            value = self.given(own)
            effort, found = from_layout1.pass_effort(ROLE[key], _for_rule(value), base)
            self.put(f"{key}.effort", effort)
            if found == from_layout1.FROM_ROLE:
                self.came_from(f"{key}.effort", own)
                self.say(
                    EFFORTS,
                    f"{key}.effort {_word(effort)}: its role's default; today the pass runs on the model's own "
                    f"effort, as neither {own} nor claude.effort names one, and the file names one for a pass that "
                    "always runs",
                )
                continue
            if found == from_layout1.FROM_BASE:
                self.came_from(f"{key}.effort", "claude.effort")
                why = (
                    "claude.effort"
                    if given is not MISSING
                    else "today's base effort, as config.json has no claude.effort"
                )
            else:
                self.came_from(f"{key}.effort", own)
                why = (
                    own
                    if value is not MISSING
                    else f"its role's default, as config.json has no {own}"
                )
            self.say(EFFORTS, f"{key}.effort {_word(effort)} ({why})")
        on_worker = config.first_pass_choice(self.cfg) == WORKER
        think = (
            str(self.worker_settings().get("think") or "").strip().lower()
            if on_worker
            else ""
        )
        for key in OPTIONAL_PASSES:
            own = self.internal(f"{key}.effort")
            value = self.take(own)
            effort, found = from_layout1.pass_effort(ROLE[key], _for_rule(value), base)
            self.came_from(f"{key}.effort", own)
            if found == from_layout1.FROM_OWN:
                why = f"{own} {_shown(value)}"
            elif key == FIRST_REVIEW_PASS and on_worker:
                effort = (
                    think
                    if think in deploy_tree.OTHER_EFFORTS
                    else from_layout1.DEFAULT
                )
                why = f"worker.think {_shown(think)}, the local worker's"
                self.came_from(f"{key}.effort", "worker.think")
            else:
                why = f"{own} {_shown(value)}" if value is not MISSING else ""
            self.put(f"{key}.effort", effort)
            self.say(
                EFFORTS, f"{key}.effort {_word(effort)}" + (f" ({why})" if why else "")
            )

    def claude_model(self, path: str, name: str, key: str) -> None:
        """A model of the Claude session at PATH, from production's KEY: a built-in name as it is; any other model id a
        claude binding of that id, named by it, or by the id with each character a binding's name does not take as "-".
        """
        self.came_from(path, key, same=False)
        if name in deploy_tree.builtin():
            self.put(path, name)
            self.say(MODELS, f"{path} {_shown(name)}")
            return
        binding = (
            name
            if BINDING_NAME.match(name)
            else re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
        )
        self.models.setdefault(binding, {"provider": "claude", "model": name})
        self.made_from.setdefault(
            binding, lambda field: key if field == "model" else ""
        )
        self.named_by.setdefault(f"model_bindings.{binding}", key)
        self.put(path, binding)
        self.say(
            MODELS,
            f"{path} {_shown(binding)}: a claude binding of the model id {_shown(name)}"
            + (
                ""
                if binding == name
                else ", named without the characters a binding's name does not take"
            ),
        )

    def claude_models(self) -> None:
        """The Claude session's models: the orchestrator's, the verification pass's and the code quality pass's
        (config.claude_model)."""
        for key, model_role in CLAUDE_PASSES:
            internal = self.internal(f"{key}.model")
            self.read.add(internal)
            self.claude_model(
                f"{key}.model",
                config.claude_model(self.cfg, model_role),
                self.production_key(internal) or internal,
            )

    def first_review_pass(self) -> None:
        """The first review pass (config.first_pass_choice, from its x-internal, reviewer, and the older keys before
        it): "default" (no draft); the local worker, as a binding named worker; or a binding of production's.
        """
        key = FIRST_REVIEW_PASS
        reviewer = self.internal(f"{key}.model")
        self.read.update(
            {reviewer, "worker_backend", "worker_url", "worker_model", "roles.drafter"}
        )
        choice = config.first_pass_choice(self.cfg)
        bound = self.peek(self.internal("model_bindings"))
        self.came_from(f"{key}.model", reviewer, same=False)
        if choice == WORKER:
            self.models[WORKER] = self.worker_binding()
            self.put(f"{key}.model", WORKER)
            self.say(
                FIRST,
                f'{key}.model "worker": the local worker leaves the package, and the pass keeps its model as a '
                "binding named worker",
            )
            return
        if (
            choice != bindings.DEFAULT
            and isinstance(bound, dict)
            and isinstance(bound.get(choice), dict)
        ):
            self.models.setdefault(choice, None)
            self.put(f"{key}.model", choice)
            self.say(
                FIRST, f"{key}.model {_shown(choice)}: the binding production names"
            )
            return
        self.put(f"{key}.model", from_layout1.DEFAULT)
        if choice == bindings.DEFAULT:
            self.say(
                FIRST,
                f'{key}.model "default": no draft; the verification pass reviews alone, as today',
            )
        else:
            self.say(
                FIRST,
                f'{key}.model "default": {reviewer} {_shown(choice)} names no binding production has: not carried',
            )

    def worker_settings(self) -> dict:
        """The local worker's settings (config.worker_settings), each read."""
        for k in (
            "backend",
            "url",
            "model",
            "auth_header",
            "auth_scheme",
            "auth_mode",
            "stall_s",
            "think",
        ):
            self.read.add(f"worker.{k}")
        for k in deploy_tree.worker_fields().values():
            self.read.add(f"worker.{k}")
        return config.worker_settings(self.cfg)

    def unbound_worker(self) -> None:
        """The local worker's settings while no pass runs on it: reported and not carried, with the
        headers its requests send; and, while the first review pass has no model, how a drafting model is bound
        later."""
        w = self.worker_settings()
        if WORKER in self.models or not bindings.local_worker_set(self.cfg):
            return
        llm = first_pass.llm_client()
        headers = llm.auth_headers(w, custom_headers.KEY_MARK)
        self.say(
            FIRST,
            "the local worker's settings, not carried: no pass runs on it. "
            + ", ".join(
                f"worker.{k} {_shown(w.get(k))}"
                for k in (
                    "backend",
                    "url",
                    "model",
                    "stall_s",
                    "max_parallel",
                    "rpm",
                    "tpm",
                    "think",
                )
            )
            + "; which provider worker.backend stands for is unverified",
        )
        self.say(
            FIRST,
            "the headers its requests send: "
            + (
                ", ".join(f"{_shown(h)}: {_shown(v)}" for h, v in headers.items())
                or "none"
            )
            + " ("
            + ", ".join(
                f"worker.{k} {_shown(w.get(k))}"
                for k in ("auth_header", "auth_scheme", "auth_mode")
            )
            + ")",
        )
        first = deploy_tree.get(self.values, f"{FIRST_REVIEW_PASS}.model")
        if first == from_layout1.DEFAULT:
            self.say(
                FIRST,
                "to bind a drafting model later: add a binding to model_bindings, name it in "
                f"{FIRST_REVIEW_PASS}.model, and install its key with {program.name()} secret",
            )

    def worker_binding(self) -> dict:
        """The local worker as a binding of config.json's form: each key the binding named worker carries
        (deploy_tree.worker_fields), a blank stall timeout what it ran on by layout 1's rule
        (from_layout1.WORKER_TIMEOUT_SECONDS); its key's secret WORKER_KEY unless it names another. Its limit on
        parallel requests is the binding's rule (`binding`)."""
        w = self.worker_settings()
        raw = self.peek("worker")
        raw = raw if isinstance(raw, dict) else {}
        backends = {v: k for k, v in deploy_tree.WORKER_BACKEND.items()}
        fields = deploy_tree.worker_fields()
        b = {binding: w.get(own) for binding, own in fields.items()}
        self.made_from[WORKER] = lambda field: f"worker.{fields.get(field, field)}"
        b["provider"] = backends.get(str(w.get("backend")), str(w.get("backend")))
        if b["provider"] == "ollama":
            b.pop("api_key_secret", None)
        else:
            b["api_key_secret"] = (
                str(w.get("api_key_secret") or "").strip() or WORKER_KEY
            )
        if raw.get("stall_s") in (None, ""):
            b["stall_s"] = from_layout1.WORKER_TIMEOUT_SECONDS
            self.say(
                FIRST,
                f"model_bindings.worker.stall_timeout_seconds {_word(b['stall_s'])}: what a blank worker.stall_s "
                "ran on",
            )
        return b

    def optional_models(self) -> None:
        """The other optional passes' models, by layout 1's rule (from_layout1.pass_model): "default" for none, or for a
        name that meant none; "worker" while a binding of that name is made or carried: the local worker's while a
        local worker is set up, else production's; a binding production has, as it is; any other, "default".
        """
        bound = self.peek(self.internal("model_bindings"))
        bound = bound if isinstance(bound, dict) else {}
        local = bindings.local_worker_set(self.cfg)
        for key in OPTIONAL_PASSES[1:]:
            internal = self.internal(f"{key}.model")
            self.came_from(f"{key}.model", internal, same=False)
            v = self.take(internal)
            if isinstance(v, dict):
                # the shape before the bindings: the pass's model on the local worker
                chose, worker_bound = WORKER, local
            else:
                chose = _for_rule(v)
                worker_bound = local or isinstance(bound.get(WORKER), dict)
            model = from_layout1.pass_model(chose, True, worker_bound)
            name = model.strip() if isinstance(model, str) else model
            if name == WORKER:
                if local:
                    if self.models.get(WORKER) is None:
                        self.models[WORKER] = self.worker_binding()
                    said = "the local worker's binding"
                else:
                    self.models.setdefault(WORKER, None)
                    said = "the binding production names"
                self.put(f"{key}.model", WORKER)
                self.say(
                    MODELS, f'{key}.model "worker": {said} ({internal} {_shown(v)})'
                )
            elif model == from_layout1.DEFAULT and (
                from_layout1.pass_model(chose, True, worker_bound=True) != WORKER
            ):
                # a model that meant none (the local worker's name would choose the worker, were one bound)
                self.put(f"{key}.model", from_layout1.DEFAULT)
            elif isinstance(name, str) and isinstance(bound.get(name), dict):
                self.models.setdefault(name, None)
                self.put(f"{key}.model", name)
                self.say(
                    MODELS,
                    f"{key}.model {_shown(name)}: the binding production names ({internal})",
                )
            else:
                self.put(f"{key}.model", from_layout1.DEFAULT)
                self.say(
                    MODELS,
                    f'{key}.model "default": {internal} {_shown(v)} names no binding production has, and no local '
                    "worker: not carried",
                )

    def model_bindings(self) -> None:
        """Every binding production has, in the file's form (`binding`), and each one made for a pass (`claude_model`,
        `worker_binding`). A binding of production's and one made with its name stop it, unless both are the same
        Claude model."""
        key = self.internal("model_bindings")
        bound = self.take(key)
        bound = {} if bound is MISSING else bound
        # production's bindings keep their names; each field's own origin is the binding's (`binding`)
        self.came_from("model_bindings", key)
        if not isinstance(bound, dict):
            self.stop(
                f"{key} is no mapping of bindings",
                _edit(key, "a mapping of binding names to bindings"),
            )
            return
        out = {}
        for name, b in bound.items():
            if not isinstance(b, dict):
                self.stop(
                    f"{_key(key, name)} is no binding",
                    _edit(_key(key, name), "a mapping of the binding's settings"),
                )
                continue
            made = self.models.get(name)
            if made is not None and (_kind(b), b.get("model")) != (
                "claude",
                made.get("model"),
            ):
                self.stop(
                    f"{_key(key, name)}: production has a binding of that name, and a pass's model needs it",
                    checks.Steps(
                        (
                            checks.Edit(_key(key, name), CONFIG_FILE, checks.DELETE),
                            ", or give it another name, and the keys that name it too",
                        )
                    ),
                )
                continue
            out[name] = self.binding(
                name, b, lambda field, name=name: _key(key, name, field)
            )
            if name not in self.models:
                self.say(MODELS, f"{_key(key, name)}: carried; no pass names it")
        for name, made in self.models.items():
            if made is not None and name not in bound:
                out[name] = self.binding(
                    name, made, self.made_from.get(name, lambda field: "")
                )
        if out:
            self.put("model_bindings", out)

    def binding(self, name: str, b: dict, source) -> dict:
        """A binding of config.json's form in the file's, by layout 1's rules (from_layout1): its fields by the
        schema's x-internal for its provider; how the key is sent, in layout 2's words (key_sending), from auth_mode,
        auth_header and auth_scheme, a missing one its provider's default (binding_defaults); a secret the one-time move
        copies, by the name it gives it (api_key_secret_name); an ollama binding's limit on parallel requests
        (ollama_parallel_requests). A field the file has no place for is reported when it is not its provider's
        default. SOURCE gives the key of production's each of its fields comes from ("" for none), where a stop's fix
        sets it."""
        kind = _kind(b)
        provider = kind.replace("-", "_")
        providers = _binding_fields(self.schema)
        fields = providers.get(provider)
        bound = self.internal("model_bindings")
        # the local worker's binding, made from its settings (`worker_binding`)
        worker = name == WORKER and b is self.models.get(WORKER)
        if fields is None:
            # the local worker's binding takes its provider from worker.backend, which names it in its own words
            names = (
                deploy_tree.WORKER_BACKEND.values()
                if worker
                else [p.replace("_", "-") for p in providers]
            )
            where = source("provider") or _key(bound, name, "provider")
            self.stop(
                f"{where} {_shown(kind)} is no provider the file has",
                _edit(
                    where,
                    f"one of {wrapping.joined([json.dumps(n) for n in names], 'or')}",
                ),
            )
            return {}
        if provider == "ollama":
            b = {**b, "concurrency": self.parallel_requests(name, b, source, worker)}
        auth = from_layout1.AUTH_KEYS
        for internal, path in fields.items():
            keys = [
                k for k in map(source, auth if internal in auth else (internal,)) if k
            ]
            renamed = internal in (*auth, "provider") or (
                internal == "api_key_secret"
                and from_layout1.api_key_secret_name(b.get(internal)) != b.get(internal)
            )
            self.came_from(f"model_bindings.{name}.{path}", *keys, same=not renamed)
        out = {"provider": provider}
        used = {"provider", "backend", "name", "where"}
        for internal, path in fields.items():
            used.add(internal)
            v = b.get(internal)
            if internal in ("provider", *auth) or v in (None, ""):
                continue
            if internal == "api_key_secret" and (
                (secret := from_layout1.api_key_secret_name(v)) != v
            ):
                self.say(
                    LOGINS,
                    f"{_key('model_bindings', name, 'api_key_secret_name')} {_shown(secret)}: the one-time move "
                    f"copies {_text(v)} to it",
                )
                v = secret
            deploy_tree.put(out, path, copy.deepcopy(v))
        defaults = from_layout1.binding_defaults(kind)
        if "auth_mode" in fields:
            today = {**defaults, **{k: v for k, v in b.items() if v is not None}}
            out.update(
                from_layout1.key_sending(
                    today["auth_mode"], today["auth_header"], today["auth_scheme"]
                )
            )
        for k, v in b.items():
            if k not in used and v != defaults.get(k, MISSING):
                self.say(
                    MODELS,
                    f"{_key(bound, name, k)} {_shown(v)}: not carried; the file has no place for it",
                )
        return out

    def parallel_requests(self, name: str, b: dict, source, worker: bool):
        """The limit on parallel requests of B, the ollama binding NAME, from its concurrency, by layout 1's rule
        (from_layout1.ollama_parallel_requests): what it ran on, which a line says when it is not the value given;
        WORKER, for the local worker's binding, whose line goes with the first review pass's. SOURCE gives the key of
        production's the value comes from (`binding`)."""
        given = b.get("concurrency")
        requests = from_layout1.ollama_parallel_requests(name, b.get("url"), given)
        if requests != given:
            where = source("concurrency") or _key(
                self.internal("model_bindings"), name, "concurrency"
            )
            had = (
                f"a blank {where}" if given in (None, "") else f"{where} {_word(given)}"
            )
            path = f"model_bindings.{name}.limits.max_parallel_requests"
            self.say(
                FIRST if worker else MODELS,
                (
                    f"{path} {requests}: no cap, what {had} ran on"
                    if requests == from_layout1.NO_CAP
                    else f"{path} {requests}: {had} on a local Ollama ran one request at a time"
                ),
            )
        return requests

    def claude_code(self) -> None:
        """integrations.claude_code from claude_env's shape (claude_env_shape.settings), the address's
        x-internal (claude.gateway, orchestrator.gateway) being the address when it is set, as it wins today; what the
        shape has that no setting carries stops it. The model variables and the nonessential-traffic switch are not
        carried. How the key is sent (api_key_sent_as, api_key_header, api_key_scheme) is the shape's, as today's
        monitor sends it; the configuration keys their x-internals name, which today's monitor does not read, are read
        so that no other section lists them."""
        key = self.internal(f"{CLAUDE_CODE}.base_url")
        gateway = orchestrator.gateway(self.cfg)
        self.read.add(key)
        shape = self.shape
        if gateway:
            shape = shape._replace(base_url="", base_url_credential="")
            self.came_from(f"{CLAUDE_CODE}.base_url", key)
        else:
            self.came_from(
                f"{CLAUDE_CODE}.base_url",
                "ANTHROPIC_BASE_URL",
                file=CLAUDE_ENV,
            )
        made, cannot = claude_env_shape.settings(shape)
        for c in cannot:
            self.stop(
                f"{CLAUDE_ENV}: {c.found}", secrets_registry.claude_env_fix(c.fix)
            )
        base = gateway or made["base_url"]
        if base:
            self.put(f"{CLAUDE_CODE}.base_url", base)
        for k in ("api_key_sent_as", "api_key_header", "api_key_scheme"):
            self.take(self.internal(f"{CLAUDE_CODE}.{k}"))
            self.put(f"{CLAUDE_CODE}.{k}", made[k])
        self.say(
            CLAUDE, f"claude_env sets {', '.join(self.shape.variables) or 'nothing'}"
        )
        for h in self.shape.headers:
            self.say(
                CLAUDE,
                f"custom header {h.name or '(a line that does not read as a header)'}: its value is "
                + ("the key" if h.value_is_key else "not the key")
                + (f", after the word {h.word}" if h.word else ""),
            )
        if gateway:
            self.say(
                CLAUDE,
                f"{CLAUDE_CODE}.base_url: {key}, which wins over ANTHROPIC_BASE_URL"
                + (
                    " (which differs)"
                    if self.shape.base_url and self.shape.base_url != gateway
                    else ""
                ),
            )
        elif base:
            self.say(CLAUDE, f"{CLAUDE_CODE}.base_url: ANTHROPIC_BASE_URL")
        self.say(
            CLAUDE,
            f"{CLAUDE_CODE}.api_key_sent_as {_shown(made['api_key_sent_as'])}, api_key_header "
            f"{_shown(made['api_key_header'])}, api_key_scheme {_shown(made['api_key_scheme'])}",
        )
        if not set(claude_env_shape.KEY_VARS) & set(self.shape.variables):
            self.say(
                CLAUDE,
                f"claude_env sets no key: the claude login is the key, which {program.name()} login claude installs",
            )
        if self.shape.models:
            self.say(
                CLAUDE,
                f"{wrapping.joined(self.shape.models)}: not carried; each call names its model by its built-in id, "
                "which the gateway takes",
            )
        switch = self.shape.nonessential_traffic
        if switch:
            self.say(
                CLAUDE,
                f"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC {_shown(switch)}: not carried; the container sets it to 1 "
                "for every session"
                + (
                    ""
                    if switch.lower() in ("1", "true")
                    else ", so the nonessential traffic production sends stops"
                ),
            )

    def logins(self) -> None:
        """Each login and secret the file's features need, and how it gets there (`login_rows`); the local worker's key
        when no binding uses it, which is not copied."""
        for name, needed, how in self.login_rows():
            if needed:
                self.say(LOGINS, f"{name}: {how}")
        if WORKER not in self.models:
            self.say(LOGINS, f"{WORKER_KEY}: not copied; no binding uses it")

    def login_rows(self) -> list:
        """The logins and secrets a deployment's features may need, each as (its name, whether the file's
        features need it, how it gets there: installed with the command that sets it, or copied once by the one-time
        move). A table of its own until the logins of rkmon login hold it."""
        team_publishes = self.on(f"{TEAM}.enabled") and deploy_tree.get(
            self.values, f"{TEAM}.delivery"
        ) in (
            from_layout1.PUBLISH_DELIVERY,
            MISSING,
        )
        key = _text(self.peek("publish.html.key"))
        prog = program.name()
        return [
            (
                "github_read",
                True,
                f"a GitHub read token of this deployment's own: {prog} login github_read",
            ),
            (
                "github_write",
                "live" in (self.mode(RA), self.mode(MQ)),
                f"the GitHub login that writes: {prog} login github_write",
            ),
            (
                "claude",
                True,
                "the key in claude_env: copied once by the one-time move",
            ),
            (
                "atlassian",
                self.on(f"{TICKETS}.follow_up_tickets.enabled")
                or self.on(f"{TICKETS}.found_issue_tickets.enabled"),
                f"the atlassian CLI's login: {prog} login atlassian, never copied",
            ),
            (
                "admin_access_token",
                True,
                f"made on this host: {prog} secret admin_access_token --generate",
            ),
            (
                "teams_webhook",
                self.on(f"{ALERTS}.develop_breaks.enabled")
                or self.on(f"{ALERTS}.needs_you.enabled"),
                "copied once by the one-time move, to the GitHub writer's volume",
            ),
            (
                "web_host_ssh_key",
                team_publishes,
                "copied once by the one-time move, from "
                + (
                    "the file publish.html.key names"
                    if key
                    else "the secrets folder's webhost_ed25519"
                )
                + f"; the web host's own key is then recorded: {host_keys.record_fix(prog)}",
            ),
            (
                "gpu_ssh_key",
                self.on(f"{RA}.remote_gpu_tests.enabled"),
                f"{prog} secret gpu_ssh_key FILE",
            ),
        ]

    def other_keys(self) -> None:
        """Every key of config.json that no rule read, by where it goes: the package's own values (listed when
        production's differs), removed keys, state, derived keys, intended differences, and the image's keys, each
        with its key in a build settings file (`_build_key`); any other is not known.
        """
        listed = {
            k: [] for k in (PACKAGE, REMOVED, STATE, DERIVED, INTENDED, BUILD, UNKNOWN)
        }
        same = {PACKAGE: 0, REMOVED: self.dropped_off}
        merged = {
            **{
                key: (TICK, deploy_tree.get(self.values, TICK))
                for key in self.internal(TICK)[1:]
            },
            "loop_log_max_mb": (
                "the package's retention.log_max_mb",
                deploy_tree.get(self.package, "retention.log_max_mb"),
            ),
        }
        for parts in _leaves(self.cfg):
            key, shown = ".".join(map(str, parts)), _key(*parts)
            if _listed(key, self.read):
                continue
            value = self.cfg
            for p in parts:
                value = value[p]
            base = deploy_tree.get(self.package, key)
            if key in merged:
                into, kept = merged[key]
                self.say(
                    FORMS,
                    f"{shown} {_shown(value)}: no key; one value with {into}, which is {_shown(kept)}"
                    + ("" if value == kept else ", so production's own is not carried"),
                )
            elif name := _listed(key, INTENDED_KEYS):
                listed[INTENDED].append(
                    f"{shown}: the package's {_shown(base)}"
                    + ("" if value == base else f", not production's {_shown(value)}")
                    + f" ({INTENDED_KEYS[name]})"
                )
            elif _listed(key, PACKAGE_KEYS):
                # the round trip reads the package's own values by the same rule (round_trip.packages_own)
                if round_trip.packages_own(key, value, self.package_values):
                    same[PACKAGE] += 1
                else:
                    listed[PACKAGE].append(
                        f"{shown}: production's {_shown(value)}, the package's {_shown(base)}"
                    )
            elif name := _listed(key, (*REMOVED_KEYS, *WHY_DROPPED)):
                if value == base or _off(value):
                    same[REMOVED] += 1
                else:
                    listed[REMOVED].append(
                        f"{shown} {_shown(value)}: {_why_removed(name)}"
                    )
            elif _listed(key, STATE_KEYS):
                listed[STATE].append(shown)
            elif _listed(key, DERIVED_KEYS):
                listed[DERIVED].append(shown)
            elif build := _build_key(key):
                listed[BUILD].append(f"{shown} {_shown(value)}: {build}")
            else:
                listed[UNKNOWN].append(shown)
        for section in (PACKAGE, REMOVED, INTENDED, BUILD, UNKNOWN):
            for line in listed[section]:
                self.say(section, line)
        other = {PACKAGE: "package key", REMOVED: "removed key"}
        for section, noun in other.items():
            if same[section]:
                # "other" beside the keys the section lists, this rule's or another's (a price field that is no price)
                noun = ("other " if self.lines[section] else "") + noun
                where = (
                    "at the package's own value"
                    if section == PACKAGE
                    else "blank, off or at the package's value"
                )
                self.say(section, f"{wrapping.counted(same[section], noun)} {where}")
        if listed[STATE]:
            self.say(STATE, ", ".join(listed[STATE]))
        if listed[DERIVED]:
            self.say(DERIVED, f"{len(listed[DERIVED])}: " + ", ".join(listed[DERIVED]))
        for path in SPECIAL:
            if self.is_left_out(path):
                continue  # its rule's line says why
            if deploy_tree.get(self.values, path) is MISSING:
                self.defaulted[path] = (
                    f"{path} (no rule sets it from production's values)"
                )
        for line in self.defaulted.values():
            self.say(DEFAULTED, line)

    def written(self) -> str:
        """The file's text, once no text in it looks like a secret (settings_file.secret_texts, the check its reader
        makes) and it passes the file's own check (settings_file.validate, with the checks across settings:
        rkmon_lib/coherence.py, and its rules on the container's share of the host, with no figure of this host) read
        back as `rkmon check` reads the file (`read_back`): the report shows that check's rows. A value the writer
        cannot write, it refuses, and the check of the values says why; the writer refuses nothing that check passes, so
        a refusal it passes is a fault of this code, never a stop. Each FAIL row is a stop with its fix in production's
        keys (`in_production`); "" when a stop is found, or was before.
        """
        for s in settings_file.secret_texts(self.values):
            where = (
                f"{s.path}'s value"
                if s.kind == "value"
                else f"a key in {s.path or 'the file'}"
            )
            self.stop(
                f"{where} looks like {s.what}, and a settings file has no secret",
                self.secret_fix(s),
            )
        if self.stops:
            self.say(CHECK, "not checked: the stops above come first")
            return ""
        doc = settings_file.Document(
            file_types.DEPLOY, file_types.DEPLOY.version, self.values
        )
        try:
            text = settings_file.dump(doc)
        except ValueError:
            text, rows = "", settings_file.validate(doc)
            if not any(row.status == checks.Status.FAIL for row in rows):
                raise
        else:
            rows = self.read_back(text)
        for row in rows:
            if row.status == checks.Status.FAIL:
                self.stop(f"{row.name}: {row.found}", self.in_production(row.fix))
            else:
                self.say(CHECK, f"{row.status} {row.name}: {row.found}")
        return "" if self.stops else text

    def read_back(self, text: str) -> list:
        """The rows of the file's own check of TEXT, read back as `rkmon check` reads the file (settings_file.loads):
        settings_file.validate's, then those of the rules on the container's share of the host, with no figure of this
        host (coherence.resources). When its reader refuses it, for a value or a key that has half a character, which
        only the reader sees and never shows, its rows, each fix of a key the file has setting that key to a value
        whose characters are whole.
        """
        try:
            doc = settings_file.loads(text, file_types.DEPLOY)
        except settings_file.Refused as exc:
            self.say(
                CHECK,
                "not checked: its reader refuses the text, as the stops above say",
            )
            whole = checks.Described("a value whose characters are whole")
            return [
                (
                    dataclasses.replace(row, fix=checks.Edit(row.name, THE_FILE, whole))
                    if deploy_tree.get(self.values, row.name) is not MISSING
                    else row
                )
                for row in exc.rows
            ]
        return settings_file.validate(doc) + coherence.resources(doc)

    def secret_fix(self, s) -> checks.Fix:
        """The fix for S (a settings_file.SecretText): the keys of production's that give the text (`origin`), each set
        to a value with no secret in it; for a key that looks like a secret, the key of production's whose value names
        it (`named_by`), or the key itself deleted from production's mapping that holds it, said, never shown. The
        file's own key where no key of production's gives the text. A value under a key that looks like a secret,
        which the check leaves out of its path, has that key's fix."""
        words = checks.Described("a value with no secret in it")
        path = s.path
        if s.kind == "value" and deploy_tree.get(self.values, path) is not MISSING:
            o = self.origin(path)
            return _fix(
                self.edits(o, words) if o.keys else [checks.Edit(path, THE_FILE, words)]
            )
        # a key that hides: at the mapping PATH, or above the value whose path leaves it out
        while True:
            node = deploy_tree.get(self.values, path) if path else self.values
            hidden = (
                [k for k in node if settings_file.hidden_key(k)]
                if isinstance(node, dict)
                else []
            )
            if hidden or not path:
                break
            path = path.rpartition(".")[0]
        o = self.origin(path) if path else NO_ORIGIN
        parts = []
        for k in hidden:
            what = settings_file.secret_shape(str(k))
            said = (
                f"(the key that looks like {what})"
                if what
                else "(the key with half a character in it)"
            )
            if f"{path}.{k}" in self.named_by:
                fix = [checks.Edit(self.named_by[f"{path}.{k}"], CONFIG_FILE, words)]
            elif o.keys and o.same:
                fix = [checks.Edit(f"{o.keys[0]}.{said}", o.file, checks.DELETE)]
            elif o.keys:
                fix = self.edits(o, words)
            else:
                fix = [
                    checks.Edit(
                        ".".join(p for p in (path, said) if p), THE_FILE, checks.DELETE
                    )
                ]
            parts += ([";"] if parts else []) + fix
        return (
            _fix(parts) if parts else checks.Edit(s.path or THE_FILE, THE_FILE, words)
        )

    def report(self) -> list:
        """The report's lines: each heading, then its items, wrapped; under Stops, each stop's fix under it, as a check
        row's (checks.FIX_LABEL)."""
        out = []
        for title in SECTIONS:
            out.append(title)
            if title == STOPS and self.stops:
                for s in self.stops:
                    out += wrapping.wrap(s.found, checks.WIDTH, "  - ", "    ")
                    out += wrapping.wrap(
                        s.fix,
                        checks.WIDTH,
                        "    " + checks.FIX_LABEL,
                        " " * (4 + len(checks.FIX_LABEL)),
                    )
                continue
            for item in self.lines[title] or ["none"]:
                out += wrapping.wrap(item, checks.WIDTH, "  - ", "    ")
        return out


# --- the command line -----------------------------------------------------------------------------------------------

# The variables that give the monitor's user and its group the deployment's ids at each start (docker/rk_ids.py; the
# compose file sets them): of the container's environment, the only ones read, as the rest may hold anything.
MONITOR_IDS = tuple(deployment.rk_ids().ENV[k] for k in ("uid", "gid"))
# What docker inspect gives: the container's image, its mounts, and the items of its environment that set MONITOR_IDS
# ("NAME=value"), then null.
INSPECT = (
    '{"image": {{json .Image}}, "mounts": {{json .Mounts}}, "ids": [{{range .Config.Env}}{{if or '
    + " ".join(f'(eq (index (split . "=") 0) "{name}")' for name in MONITOR_IDS)
    + "}}{{json .}}, {{end}}{{end}}null]}"
)
# Where the translation code is mounted in the one-off container.
CODE_DIR = "/translation"
# How long docker inspect and the one-off container may take; the translation itself takes a second or two. The round
# trip reads the monitor's transcripts of its last week or so once (round_trip.usage_of), and all of them when that
# week reaches back past those: a monitor that has run for months has many.
INSPECT_SECONDS = 60
RUN_SECONDS = 300
ROUND_TRIP_SECONDS = 1800
# The file docker run writes the one-off container's id to (--cidfile), in a private temporary folder of its own; the
# id as docker writes it; and what docker ps gives of a container: its state, "removing" while docker removes it. On
# the time limit or Ctrl-C this tool removes its one-off container by that id (`remove_one_off`).
CIDFILE = "one_off.cid"
CONTAINER_ID = re.compile("[0-9a-f]{64}")
CONTAINER_STATE = "{{.State}}"
# The one-off container's share of this host, which leaves production beside it its own: one CPU's time at most, and a
# memory limit, its /tmp in it, at which the kernel kills it (exit KILLED, 128 and SIGKILL's 9).
ONE_OFF_CPUS = "1"
ONE_OFF_MEMORY = "2g"
KILLED = 137
# Each step of the round trip in the one-off container, by the name it announces on stderr as it begins (a line of
# STEP and the name, `_round_trip`), with what it does in plain words: a run past ROUND_TRIP_SECONDS says how far it
# got.
STEP = "round trip step: "
TRIP_STEPS = {
    "translate": "translating production's settings",
    "usage": "reading the monitor's transcripts",
    "compare": "comparing the two",
}
# The name of the rows that check the file of the differences the owner agreed (--agreed).
AGREED = "agreed differences"
# The name of the report's line of the week of usage the round trip prices, and of its FAIL row when none is recorded
# (`_usage_rows`).
USAGE_ROW = "Usage"


def _answer(config: dict, claude_env: claude_env_shape.Shape) -> dict:
    """The translation as the one-off container prints it: {deploy_file, report, stops}, the file "" and the stops
    named while a setting cannot be carried, each {found, fix}, the fix in its JSON form (checks.fix_to_json).
    """
    try:
        done = translate(config, claude_env)
    except Untranslatable as exc:
        stops = [
            {"found": s.found, "fix": checks.fix_to_json(s.fix)} for s in exc.stops
        ]
        return {"deploy_file": "", "report": exc.report, "stops": stops}
    return {"deploy_file": done.deploy_file, "report": done.report, "stops": []}


def _round_trip(config: dict, claude_env: claude_env_shape.Shape, step=None) -> dict:
    """The round trip as the one-off container prints it: `_answer`'s object with its file "", as the round trip
    writes none, and, when no setting stops the translation, DIFFERENCES (round_trip.to_json) and USAGE, the usage
    they price: {root, records, days}, ROOT the folder of the monitor's transcripts as today's code finds it
    (spend.projects_root), DAYS the week priced (round_trip.week_of), RECORDS the number of runs and turns recorded on
    those days. The scan's cache goes in a temporary folder: in the one-off container, its /tmp, as all else is
    read-only. STEP, when given, is called with the name of each step (TRIP_STEPS) as it begins.
    """
    if step is not None:
        step("translate")
    done = _answer(config, claude_env)
    if done["stops"]:
        return done
    values = settings_file.loads(done["deploy_file"], file_types.DEPLOY).values
    root = spend.projects_root(round_trip.today(config))
    if step is not None:
        step("usage")
    with tempfile.TemporaryDirectory() as folder:
        usage = round_trip.usage_of(root, folder, round_trip.running(values, config))
    if step is not None:
        step("compare")
    found = round_trip.compare(config, claude_env, values, usage)
    days = round_trip.week_of(usage.today)
    return dict(
        done,
        deploy_file="",
        differences=round_trip.to_json(found),
        usage={
            "root": str(root),
            "records": len(round_trip.on_days(usage.today, days)),
            "days": list(days),
        },
    )


def _no_number(text: str):
    """json.loads' parse_constant: NaN and Infinity, which no JSON number is, are a ValueError."""
    raise ValueError(f"{CONFIG_FILE} is not JSON: {text} is no JSON number")


def _number(text: str) -> float:
    """json.loads' parse_float: a number too large for one (a float that is not finite) is a ValueError, never shown."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(
            f"{CONFIG_FILE} is not JSON: it has a number too large for one"
        )
    return value


def _inside(trip: bool = False) -> int:
    """In the one-off container: translate config.json and claude_env's shape, and print `_answer` as one JSON object;
    with TRIP, run the round trip, each step announced on stderr as it begins (`_announce`), and print
    `_round_trip`'s. What cannot be read is one line on stderr, the last, never a value (exit 1); a number JSON has
    not (NaN, Infinity, or one too large) is not read.
    """
    try:
        where = config.config_path()
        raw = json.loads(
            where.read_text(encoding="utf-8"),
            parse_constant=_no_number,
            parse_float=_number,
        )
        if not isinstance(raw, dict):
            raise ValueError(NOT_SETTINGS)
        shape = claude_env_shape.read(
            orchestrator.secret_path(raw), settings_file.secret_shape
        )
    except OSError as exc:
        print(
            f"{exc.filename or 'config.json'}: cannot be read ({exc.strerror})",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError as exc:
        print(f"config.json is not JSON: {exc.msg}, line {exc.lineno}", file=sys.stderr)
        return 1
    except (ValueError, paths.PathsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        out = json.dumps(
            _round_trip(raw, shape, _announce) if trip else _answer(raw, shape)
        )
    except Exception as exc:
        # a fault of this code: where it is, never what production's values are
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(
            f"{'the round trip' if trip else 'the translation'} failed: {type(exc).__name__} at "
            f"{pathlib.Path(frame.filename).name}:{frame.lineno}",
            file=sys.stderr,
        )
        return 1
    print(out)
    return 0


def _announce(name: str) -> None:
    """In the one-off container: the round trip's step NAME (TRIP_STEPS) on stderr as it begins, a line that names no
    value."""
    print(f"{STEP}{name}", file=sys.stderr, flush=True)


def _how_far(stderr) -> str:
    """How far the round trip in the one-off container got, by what it printed on STDERR (bytes or text): "while" and
    what the last step it announced does (TRIP_STEPS), or "before it began". No other line of it is shown.
    """
    text = (
        stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr or ""
    )
    begun = [
        line.removeprefix(STEP)
        for line in text.splitlines()
        if line.startswith(STEP) and line.removeprefix(STEP) in TRIP_STEPS
    ]
    return f"while {TRIP_STEPS[begun[-1]]}" if begun else "before it began"


def _past_the_limit(container: str, stderr) -> str:
    """The round trip's ERROR when the one-off container of CONTAINER's image gives no answer in ROUND_TRIP_SECONDS:
    the limit, and how far it got by what it printed on STDERR by then (`_how_far`). What became of the container
    follows it (`_left`)."""
    return (
        f"the round trip in the one-off container of {container}'s image reached its time limit, "
        f"{ROUND_TRIP_SECONDS} seconds (ROUND_TRIP_SECONDS in {TOOL}), {_how_far(stderr)}"
    )


class Removal(NamedTuple):
    """What became of this tool's own one-off container after its time limit or Ctrl-C (`remove_one_off`): its ID, ""
    when docker made none; whether it is GONE; and, when it is not, WHY, docker's last line.
    """

    id: str
    gone: bool
    why: str = ""


def remove_one_off(docker: str, cidfile: pathlib.Path) -> Removal:
    """This tool's own one-off container removed (docker rm -f) by the id docker run wrote to CIDFILE (--cidfile),
    which names no other container: never production's. A second Ctrl-C meanwhile is ignored, as the removal is what
    it would leave undone. When docker rm fails, docker ps tells whether the container is there: one docker's --rm
    removed first is gone, as is one docker is removing (CONTAINER_STATE)."""
    try:
        cid = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        cid = ""
    if not CONTAINER_ID.fullmatch(cid):
        return Removal("", True)

    def docker_(*args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [docker, *args], capture_output=True, text=True, timeout=INSPECT_SECONDS
        )

    before = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        p = docker_("rm", "-f", cid)
        if p.returncode == 0:
            return Removal(cid, True)
        q = docker_(
            "ps",
            "--all",
            "--no-trunc",
            "--filter",
            f"id={cid}",
            "--format",
            CONTAINER_STATE,
        )
        if q.returncode == 0 and q.stdout.strip() in ("", "removing"):
            return Removal(cid, True)
        return Removal(cid, False, f"docker rm: {_last_line(p.stderr)}")
    except subprocess.TimeoutExpired:
        return Removal(
            cid, False, f"docker gave no answer in {INSPECT_SECONDS} seconds"
        )
    finally:
        signal.signal(signal.SIGINT, before)


def _left(removal: Removal, subject: str) -> tuple:
    """(words, fix) after a time limit or Ctrl-C: what became of the one-off container (REMOVAL), SUBJECT naming it,
    then that nothing else changed; the fix, docker rm -f by its id, when it is left."""
    if not removal.id:
        return "nothing changed", None
    if removal.gone:
        return f"{subject} is removed, and nothing changed", None
    return (
        f"{subject} could not be removed ({removal.why}), and nothing else changed",
        checks.Command(("docker", "rm", "-f", removal.id)),
    )


def _mount(mounts, destination: str):
    """The --mount source of the container's mount at DESTINATION: ("volume", name) or ("bind", path); None for none."""
    for m in mounts or []:
        if m.get("Destination") == destination:
            if m.get("Type") == "volume" and m.get("Name"):
                return "volume", m["Name"]
            if m.get("Type") == "bind" and m.get("Source"):
                return "bind", m["Source"]
    return None


def monitor_user(ids) -> str:
    """docker run's --user for the monitor's user, with the ids the container's entry point gives it at each start
    (docker/rk_ids.py): IDS, the "NAME=value" items of MONITOR_IDS docker inspect gives, each one blank or unset the
    image's own user or group (container_layout.MONITOR). A ValueError names a value that is no user id (a number from
    1 to rk_ids.py's MAX_ID: never root's)."""
    most = deployment.rk_ids().MAX_ID
    given = {}
    for item in ids or []:
        if isinstance(item, str):
            name, _, value = item.partition("=")
            given[name] = value.strip()
    out = []
    for name, own in zip(
        MONITOR_IDS, (container_layout.MONITOR.name, container_layout.MONITOR.group)
    ):
        v = given.get(name, "")
        if v and not (re.fullmatch("[0-9]+", v) and 1 <= int(v) <= most):
            raise ValueError(
                f"{name} {_shown(v)} is no user id (a number from 1 to {most})"
            )
        out.append(v or own)
    return ":".join(out)


def one_off(
    docker: str, image: str, work, secrets, user: str, args, cidfile=None
) -> list:
    """The one-off container's command, this tool run with ARGS (main's: --inside, and with it --round-trip for the
    round trip): the image never pulled, no network, no log, one CPU's time and ONE_OFF_MEMORY at most, a
    read-only root, the work and secrets mounts and this package's folder read-only, as USER (monitor_user) with no
    capability and no way to gain one. The tool reads config.json and claude_env, which the monitor reads as that
    user: the entry point gives the volumes' files its ids at each start (docker/rk_ids.py). The round trip reads the
    monitor's transcripts in the work volume too, and writes its scan's cache in /tmp, a tmpfs within that memory.
    With CIDFILE, docker writes the container's id there (--cidfile), by which this tool removes it on its time limit
    or Ctrl-C (`remove_one_off`).
    """

    def mount(kind_source, dst: str) -> str:
        kind, src = kind_source
        extra = ",volume-nocopy" if kind == "volume" else ""
        return f"type={kind},src={src},dst={dst},readonly{extra}"

    return [
        docker,
        "run",
        "--rm",
        *(("--cidfile", str(cidfile)) if cidfile is not None else ()),
        "--pull",
        "never",
        "--log-driver",
        "none",
        "--network",
        "none",
        "--cpus",
        ONE_OFF_CPUS,
        "--memory",
        ONE_OFF_MEMORY,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--user",
        user,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        mount(work, container_layout.WORK_DIR),
        "--mount",
        mount(secrets, container_layout.SECRETS_DIR),
        "--mount",
        mount(("bind", str(paths.PACKAGE)), CODE_DIR),
        "--entrypoint",
        "python3",
        image,
        "-I",
        "-B",
        f"{CODE_DIR}/{TOOL}",
        *args,
    ]


def _command(container: str, *options) -> checks.Command:
    """This tool's command for production's container CONTAINER, with OPTIONS, as a next line names it."""
    return checks.Command(("python3", TOOL, "--container", container, *options))


def _again(typed: tuple) -> str:
    """The next line that runs TYPED, the command as run, again."""
    return f"{steps.NEXT_LABEL}{checks.Command(typed)}"


def _last_line(text: str) -> str:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return lines[-1] if lines else "no message"


def _listing(a, typed: tuple) -> list:
    """The round trip's list of intended differences, read before it runs: the package's rows (round_trip.DIFFERENCES),
    then those of the file --agreed names, read only when it is a private file (private_files.checked); and the FILE
    of --out checked as the report's place, as the report is written after the round trip. outcome.Stopped when a file
    is refused, with the refusal as it shows itself, or when the agreed rows break the list's form. TYPED: the command
    as run.
    """
    listing = round_trip.rows(round_trip.DIFFERENCES)
    if a.agreed:
        shown = settings_file.shown(a.agreed)
        try:
            path = private_files.checked(
                a.agreed,
                paths.PACKAGE,
                name=AGREED,
                retry=private_files.again(typed, a.agreed, "--agreed"),
                shown=shown,
            )
            listing += round_trip.rows(path, shown)
        except private_files.Refused as exc:
            raise outcome.Stopped(
                "REFUSED",
                f"{shown} is not read (the row above says why)",
                above=str(exc),
            ) from None
        except OSError as exc:
            raise outcome.Stopped(
                "ERROR", f"{shown} cannot be read: {exc.strerror}"
            ) from None
        except ValueError as exc:
            fix = checks.Steps(
                (
                    "give the file the list's form (tools/round_trip.py), then run",
                    checks.Command(typed),
                )
            )
            raise outcome.Stopped("REFUSED", str(exc), fix=fix) from None
    if a.out:
        shown = settings_file.shown(a.out)
        try:
            private_files.write(
                a.out,
                "",
                paths.PACKAGE,
                replace=True,
                retry=private_files.again(typed, a.out, "--out"),
                dry_run=True,
                shown=shown,
            )
        except private_files.Refused as exc:
            raise outcome.Stopped(
                "REFUSED",
                f"{shown} is not written, and the round trip has not run (the row above says why)",
                above=str(exc),
            ) from None
    return listing


def _side(value) -> str:
    """One side of a difference as the round trip's report shows it: whole, as `_shown` shows a value, and never a
    value `_unshown` hides; "(not set)" for none."""
    return "(not set)" if value is round_trip.MISSING else _shown(value, None)


def _trip_rows(found, listing, fix) -> list:
    """A row for each difference FOUND (round_trip.Difference), named by its key (round_trip.named) and its kind, both
    sides shown (`_side`): PASS when a row of LISTING lists it (round_trip.listed), with that row's reason; FAIL with
    FIX when none does."""
    out = []
    for d in found:
        name = f"{_key(*round_trip.named(d.key))} ({d.kind})"
        sides = f"production {_side(d.production)}, after {_side(d.after)}"
        row = round_trip.listed(d, listing)
        if row is None:
            out.append(
                checks.Check(name, checks.Status.FAIL, f"{sides}; not on the list", fix)
            )
            continue
        out.append(
            checks.Check(name, checks.Status.PASS, f"{sides}; listed: {row.why}")
        )
    return out


def _usage(data) -> dict:
    """The USAGE of `_round_trip`'s answer: ValueError when it is not of that form."""
    if not (
        isinstance(data, dict)
        and isinstance(data.get("root"), str)
        and isinstance(data.get("records"), int)
        and isinstance(data.get("days"), list)
        and all(isinstance(day, str) for day in data["days"])
    ):
        raise ValueError("no usage of the round trip's form")
    return data


def _usage_rows(usage: dict, typed: tuple) -> list:
    """The round trip's row of the USAGE it prices (`_usage`): none while a week of it is recorded; a FAIL when none
    is recorded under its root, as no price is then compared, whose fix names where the monitor's transcripts are
    looked for (spend.projects_root, in the one-off container) and runs TYPED, the command as run, again.
    """
    if usage["days"]:
        return []
    root = usage["root"]
    fix = checks.Steps(
        (
            f"the monitor's transcripts are looked for under {root}, the folder paths.claude_projects names in "
            "production's config.json, else $CLAUDE_CONFIG_DIR/projects in its container: set "
            "paths.claude_projects to their folder, then run",
            checks.Command(typed),
        )
    )
    return [
        checks.Check(USAGE_ROW, checks.Status.FAIL, f"none recorded under {root}", fix)
    ]


def _trip_report(name: str, usage: dict, rows: list) -> list:
    """The round trip's report: what it compares, the week of usage it prices (`_usage`) when one is recorded, then
    ROWS (`_usage_rows` and `_trip_rows`) as checks.render shows them."""
    days = usage["days"]
    lines = [
        f"The round trip of {name}'s settings: each difference, by its key and its kind"
    ]
    if days:
        lines.append(
            f"{USAGE_ROW}: {wrapping.counted(usage['records'], 'record')} under {usage['root']}, priced both ways over "
            f"the week {days[0]} to {days[-1]}"
        )
    out = []
    for line in lines:
        out += wrapping.wrap(line, checks.WIDTH)
    return out + ([checks.render(rows)] if rows else [])


def _round_trip_end(a, done: dict, fixes: list, listing: list, typed: tuple) -> int:
    """The round trip's report and its outcome, from DONE, the one-off container's answer (`_round_trip`), FIXES, its
    stops' fixes, and LISTING (`_listing`): the report printed, or with --out written; then the outcome line and the
    next line (the module's docstring). TYPED: the command as run."""
    shown = settings_file.shown(a.out) if a.out else ""
    where = f"in {shown}" if a.out else "above"
    again = (
        checks.Command(typed)
        if a.agreed
        else private_files.again(typed, "", "--agreed")(private_files.FILE)
    )
    fix = None
    if fixes:  # a setting stops the translation, so the round trip cannot run
        report, word = done["report"], "FAILED"
        text = (
            f"{wrapping.counted(len(fixes), 'setting')} cannot be carried (each under Stops {where}, with its fix), "
            "so the round trip has not run; nothing changed"
        )
        fix = None if a.out else _together(fixes)
        next_line = _again(typed)
    else:
        try:
            found = round_trip.from_json(done.get("differences"))
            usage = _usage(done.get("usage"))
        except ValueError:
            return outcome.finish(
                "ERROR",
                f"the one-off container of {a.container}'s image gave differences this tool does not read",
            )
        fail_fix = checks.Steps(
            (
                "explain it, then agree it, as a row of the file --agreed names (a row's form: tools/round_trip.py), "
                "or fix the translation; then run",
                again,
            )
        )
        rows = _usage_rows(usage, typed) + _trip_rows(found, listing, fail_fix)
        report = _trip_report(a.container, usage, rows)
        word, text = outcome.tally(rows)
        if word == "FAILED":
            if a.out:
                text = (
                    text.removesuffix(outcome.FIX_EACH_FAIL)
                    + f"nothing changed (fix each FAIL {where})"
                )
            next_line = f"{steps.NEXT_LABEL}{again}"
        else:
            if a.out:
                text += f"; the report is {where}"
            write = _command(a.container, "--out", private_files.FILE)
            next_line = f"{steps.NEXT_LABEL}{write}"
    if not a.out:
        print("\n".join(report))
        return outcome.finish(word, text, fix=fix, next=next_line)
    try:
        private_files.write(
            a.out,
            "\n".join(report) + "\n",
            paths.PACKAGE,
            replace=True,
            retry=private_files.again(typed, a.out, "--out"),
            shown=shown,
        )
    except private_files.Refused as exc:
        print(exc)
        return outcome.finish(
            "REFUSED", f"{shown} is not written (the row above says why)"
        )
    except OSError as exc:
        return outcome.finish("ERROR", f"{shown} cannot be written: {exc.strerror}")
    return outcome.finish(word, text, fix=fix, next=next_line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--container",
        metavar="NAME",
        help="production's container, whose image and volumes are read",
    )
    ap.add_argument(
        "--out",
        metavar="FILE",
        help="the file to write, yours only: the deploy settings file, or with --round-trip its report",
    )
    what = ap.add_mutually_exclusive_group()
    what.add_argument(
        "--report-only", action="store_true", help="print the report, and write no file"
    )
    what.add_argument(
        "--round-trip",
        action="store_true",
        help="run the round trip, and write no settings file: production's settings against the "
        "file's, taken back",
    )
    ap.add_argument(
        "--agreed",
        metavar="FILE",
        help="with --round-trip: the differences the owner agreed, as rows in the list's form "
        "(tools/round_trip.py); yours only",
    )
    ap.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.inside:
        return _inside(a.round_trip)
    if not a.container or not (a.out or a.report_only or a.round_trip):
        ap.error("give --container NAME, and --out FILE, --report-only or --round-trip")
    if a.out and a.report_only:
        ap.error("--report-only writes no file: give --out FILE or --report-only")
    if a.agreed and not a.round_trip:
        ap.error("--agreed FILE goes with --round-trip")
    typed = ("python3", TOOL, *(sys.argv[1:] if argv is None else argv))
    try:
        return _with_container(a, typed)
    except KeyboardInterrupt:
        # Ctrl-C outside the one-off container, whose own Ctrl-C removes it: before anything is written, or during a
        # write, which leaves the file as it was (private_files.write)
        return outcome.finish("INTERRUPTED", "nothing changed", next=_again(typed))


def _with_container(a, typed: tuple) -> int:
    """main's work with --container (the module's docstring), from A, the parsed command line, and TYPED, the command
    as run."""
    listing = []
    if a.round_trip:
        try:
            listing = _listing(a, typed)
        except outcome.Stopped as stop:
            return stop.finish()
    docker = shutil.which("docker")
    if not docker:
        return outcome.finish(
            "FAILED",
            "docker is not on this host's PATH, and the translation runs in a one-off container",
            earlier_step=True,
        )
    try:
        p = subprocess.run(
            [docker, "container", "inspect", "--format", INSPECT, a.container],
            capture_output=True,
            text=True,
            timeout=INSPECT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return outcome.finish(
            "ERROR",
            f"docker cannot inspect the container {a.container}: no answer in {INSPECT_SECONDS} seconds",
        )
    try:
        info = json.loads(p.stdout) if p.returncode == 0 else None
    except ValueError:
        info = None
    if not isinstance(info, dict) or not info.get("image"):
        return outcome.finish(
            "ERROR",
            f"docker cannot inspect the container {a.container}: {_last_line(p.stderr)}",
        )
    work = _mount(info.get("mounts"), container_layout.WORK_DIR)
    secrets = _mount(info.get("mounts"), container_layout.SECRETS_DIR)
    for found, dst in (
        (work, container_layout.WORK_DIR),
        (secrets, container_layout.SECRETS_DIR),
    ):
        if not found:
            return outcome.finish(
                "ERROR", f"the container {a.container} has no volume or folder at {dst}"
            )
        if "," in found[1]:
            return outcome.finish(
                "ERROR",
                f"the container {a.container}'s mount at {dst} has a comma in its name, which docker run's "
                "--mount does not take",
            )
    try:
        user = monitor_user(info.get("ids"))
    except ValueError as exc:
        return outcome.finish("ERROR", f"the container {a.container}'s {exc}")
    inside = ("--inside", "--round-trip") if a.round_trip else ("--inside",)
    seconds = ROUND_TRIP_SECONDS if a.round_trip else RUN_SECONDS
    again = _again(typed)
    with tempfile.TemporaryDirectory(prefix="translate_production-") as private:
        cidfile = pathlib.Path(private) / CIDFILE
        try:
            p = subprocess.run(
                one_off(docker, info["image"], work, secrets, user, inside, cidfile),
                capture_output=True,
                text=True,
                timeout=seconds,
            )
        except subprocess.TimeoutExpired as exc:
            words, fix = _left(remove_one_off(docker, cidfile), "the container")
            limit = (
                _past_the_limit(a.container, exc.stderr)
                if a.round_trip
                else f"the one-off container of {a.container}'s image: no answer in {seconds} seconds"
            )
            return outcome.finish("ERROR", f"{limit}; {words}", fix=fix, next=again)
        except KeyboardInterrupt:
            words, fix = _left(
                remove_one_off(docker, cidfile),
                f"the one-off container of {a.container}'s image",
            )
            return outcome.finish("INTERRUPTED", words, fix=fix, next=again)
    try:
        done = json.loads(p.stdout) if p.returncode == 0 else None
    except ValueError:
        done = None
    if not isinstance(done, dict) or not isinstance(done.get("report"), list):
        why = _last_line(p.stderr)
        # a round trip whose last line is a step stopped with no message of its own
        if a.round_trip and why.startswith(STEP):
            why = f"no message, {_how_far(p.stderr)}"
        if p.returncode == KILLED:
            why += (
                f"; killed (exit {KILLED}), as it is when it reaches its memory limit, {ONE_OFF_MEMORY} "
                f"(ONE_OFF_MEMORY in {TOOL})"
            )
        return outcome.finish(
            "ERROR", f"the one-off container of {a.container}'s image: {why}"
        )
    try:
        fixes = [checks.fix_from_json(s["fix"]) for s in done.get("stops") or []]
    except (KeyError, TypeError, ValueError):
        fixes = None
    if fixes is None or None in fixes:
        return outcome.finish(
            "ERROR",
            f"the one-off container of {a.container}'s image gave a stop with no fix this tool reads",
        )
    if a.round_trip:
        return _round_trip_end(a, done, fixes, listing, typed)
    print("\n".join(done["report"]))
    if fixes:
        word = "REFUSED" if a.out else "FAILED"
        text = (
            f"{wrapping.counted(len(fixes), 'setting')} cannot be carried (each under Stops above, with its fix); "
            "nothing written"
        )
        return outcome.finish(word, text, fix=_together(fixes), next=_again(typed))
    if a.report_only:  # the round trip comes next
        return outcome.finish(
            "PASSED",
            f"the translation carries {a.container}'s settings; nothing written",
            next=f"{steps.NEXT_LABEL}{_command(a.container, '--round-trip')}",
        )
    # the file as each line shows it, the rows of a refusal too: `~/…` under the home folder
    shown = settings_file.shown(a.out)
    try:
        private_files.write(
            a.out,
            done["deploy_file"],
            paths.PACKAGE,
            replace=True,
            retry=private_files.again(typed, a.out, "--out"),
            shown=shown,
        )
    except private_files.Refused as exc:
        print(exc)
        return outcome.finish(
            "REFUSED", f"{shown} is not written (the row above says why)"
        )
    except OSError as exc:
        return outcome.finish("ERROR", f"{shown} cannot be written: {exc.strerror}")
    return outcome.finish(
        "WROTE",
        shown,
        next=steps.next_line(steps.DEPLOY_STEPS[1], FILE=shown),
    )


if __name__ == "__main__":
    sys.exit(main())
