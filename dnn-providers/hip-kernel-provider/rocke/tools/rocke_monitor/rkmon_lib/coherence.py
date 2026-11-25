# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The checks across keys of the two settings files you write: what no one key's schema can
say, such as a feature that needs another, or a key that a value of another key requires. file_types registers
`deploy` and `build` as their file types' coherence: settings_file.validate runs them once every key fits the schema
and no <...> is left.

- `deploy(doc) -> list[Check]`: a deploy settings document's checks across keys, over its effective values (a
  key's value, or its default): a FAIL or a WARN row for each problem, each with its fix; else one PASS row,
  "coherence: the features' settings fit together"; and after them, whatever they are, check 7's PASS row while the
  team dashboard is served in prototype (below).
- `build(doc) -> list[Check]`: a build settings document's; its PASS row is "coherence: the build settings
  fit together".
- `resources(doc, host_memory_gib=0.0, host_cpus=0) -> list[Check]`: the container's share of this host (checks 8
  and 9), on this host's memory in GiB and its CPUs (0: not known, and not compared). No file type's coherence:
  validate checks a file alone, the same on every machine, and `rkmon check` and `rkmon confirm` add these rows with
  this host's own figures (deployment.host_size). Each of the three keys has one row at most:
  the RAM disk's, the memory's and the CPUs'; a value that does not fit its key's schema has validate's row alone. A
  fix offers only values this host has room for, and the RAM disk's and the memory's fixes agree: a larger RAM disk
  comes with the memory to hold it, and a smaller one is no smaller than the starting reviews and builds need, or 0.
- `least(ramdisk=0, reviews=0, builds=0) -> Least`: what the container takes at least, by the rules of checks 8 and
  9: the RAM disk REVIEWS reviews and BUILDS builds at once need at their first estimates (deployment.PIPE_GB and
  deployment.HEAVY_GB GiB each), and it in whole GiB, rounded up; the memory limit that holds a RAM disk of RAMDISK GiB
  beside the work's share (deployment.WORK_GB); and a CPU for each review and each build. The one home of those
  numbers: `resources`, deployment.sizing_checks, reach's RAM disk row and tools/translate_production.py read it.
- `skills(doc, package_dir) -> list[Check]`: check 18, the team's skills folder SKILLS_KEY names, as the
  delivery finds it on this host (deploy_plan.skills_path); no row while the key is blank. No file type's coherence
  either, as `resources`: `rkmon check` and `rkmon confirm` add these rows. Each FAIL names the key: a folder that is
  not there, is inside the package folder PACKAGE_DIR, or that you cannot read, alone; else, in one row, each thing
  in it that is no regular Markdown file at one of `skills_places`, with why: a link (never followed), a file of any
  other kind, a file or a folder at none of those places (a folder named once, with its /, and not looked into), a
  file whose name does not end in .md, or that cannot be read as UTF-8 text, as the delivery reads it
  (deploy_plan.regular_bytes); and in a row of their own, the checklist leads a verifier file there
  (verifier_file.FILE) lacks (verifier_file.missing), which a review in parts would say are missing. Else one PASS
  row: the package's files (PACKAGE_DIR's agentic/) the team's replace, and those they add.
- `skills_places()`: the places a file of a team's skills folder may be, as SKILLS_KEY's description, their one
  home, names them, "it may contain only A, B and C.": each a folder, with its /, any file in which (at any depth) is
  at that place, or a file's own path. ValueError while the description names none.

The rows, by their checks' numbers:

- A key Required when a value of another key decides it (its x-required-when: checks 4, 7 and 10, and the build file's
  check 3), left out or blank while that value requires it; and a key Required when its feature is on that the file
  gives blank or empty ("" or null, [] or {}) while that feature is on: FAIL, named by the key, with where its value
  comes from (its x-source, as every edit's words: settings_file's rule). A key missing while its feature is on is
  validate's row, by its section (check 13).
- 2: a feature that is on while a feature it needs is off: FAIL, named by the feature's title.
- 3: each pass's model is built in or a model_bindings entry, and a Claude model on a pass that is not optional; its
  effort is one its model takes. A model_bindings entry named after a built-in model with another provider than
  "claude": FAIL.
- 4: "header_and_bearer" with the header Authorization: FAIL, as the key would go twice in one header.
- 5: while spend is on, a model the passes use with no price: one WARN, "spend".
- 7: the team dashboard's delivery and its blocks: "publish" with a serve block, or "serve" with a publish block:
  FAIL, "team dashboard". A serve block left out means its defaults. With priority requests on and "publish", a
  public_url the web host's request service does not serve the page at (web_host_install.service_address: http, a
  port of 1024 or more, a host its unit takes, and no path): FAIL, "team dashboard", its fix the address to give.
  While the team dashboard is served in prototype, its page is served on this host's own address alone, whatever
  serve.address says (features.team_host): where serve.address says another, a PASS row of its own, "team
  dashboard" (TEAM_ROW), says so beside every other row, and rkmon check shows it with the FAIL and WARN rows
  (deploy_tree.read), as rkmon confirm does (confirm.checks).
- 8: an id the container refuses at each start: FAIL. For the monitor's user, the system's "nobody" user's (65534)
  or one of the container's own users' (its PR runners' and its GitHub writer's, the package's ids); for its group,
  the PR runners' group's. The monitor's group may be any other group, a system group included.
- 11: an address on the loopback in integrations.claude_code or in a model binding: WARN.
- 12: no repository defaults for repository.name and repository.folder (rkmon_lib/repository_defaults.py): one WARN,
  "repository defaults", naming the keys that take the package's generic values.
- 17: while the remote GPU tests are on, a default target that is not one of gpu_types' targets: FAIL. Each value's
  own form is its key's schema pattern, which validate checks.
- 20: a label in both repository.scope_labels and repository.shared_label_folders: one FAIL, "shared labels".
- The build file's checks 4 and 5: a blank SHA-256 where nothing else checks the download (ROCm's public installer,
  the SDK from the package index): WARN, "not checked"; a package mirror with a base image that is not Ubuntu: FAIL.

A value with a placeholder (<...>) has the placeholder's row alone (validate's): no check here adds a row for its key,
and a key whose value has one is given, never missing or blank.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import ipaddress
import math
import os
import pathlib
import re
import urllib.parse
from collections.abc import Mapping
from typing import NamedTuple

from . import (
    deploy_tree,
    deployment,
    features,
    image,
    program,
    repository_defaults,
    settings_file,
    verifier_file,
    wrapping,
)
from .checks import DELETE, Check, Command, Described, Edit, Status, Steps
from .settings_file import PLACEHOLDER

MISSING = object()  # a key the file leaves out
RA = "features.review_automation"
SPEND = RA + ".spend"
PRICES = SPEND + ".usd_per_million_tokens"
GPU = RA + ".remote_gpu_tests"
SKILLS_KEY = RA + ".skills_folder"
SKILLS = "skills folder"  # check 18's rows
TEAM_ROW = "team dashboard"  # check 7's rows
TEAM = "features.dashboards.team"
CLAUDE_CODE = "integrations.claude_code"
BINDINGS = "model_bindings"
CPUS = "host.resources.cpus"
MEMORY = "host.resources.memory_gib"
RAMDISK = "host.resources.ramdisk_gib"
PARALLEL_REVIEWS = RA + ".parallel_reviews"
PARALLEL_BUILDS = "monitor.parallel_builds"
CLAUDE = "claude"  # a Claude model's provider

# Check 2: each feature, and the feature it needs.
NEEDS = (
    ("features.ticket_creation.follow_up_tickets", RA),
    ("features.merge_queue.drift_notices", RA),
    ("features.alerts.develop_breaks", "features.merge_queue.develop_guard"),
    ("features.ticket_creation.found_issue_tickets", "features.dashboards.admin"),
    (TEAM, "features.dashboards.admin"),
)


# --- the document --------------------------------------------------------------------------------------------------


class _File:
    """One document as the checks read it: its values as the file gives them and as they take effect, and its rows."""

    def __init__(self, doc):
        self.doc = doc
        self.root = doc.file_type.load()
        self.given_values = doc.values
        self.values = settings_file.effective(doc)
        self.where = settings_file.where_of(doc)
        self.again = settings_file.read_again(doc.file_type, doc.path)
        self.rows = []

    def on(self, feature: str) -> bool:
        """Whether FEATURE is on: its switch true, and each feature around it on."""
        return settings_file.feature_on(self.root, self.given_values, feature)

    def used(self, key: str) -> bool:
        """Whether a feature that reads the key at KEY is on."""
        return settings_file.in_use(self.root, self.given_values, key)[0]

    def value(self, key: str):
        """The effective value at KEY, a path of the schema's own keys (no map's entry); None when there is none."""
        return _at(self.values, key, None)

    def given(self, key: str):
        """The value the file gives at KEY, or MISSING."""
        return _at(self.given_values, key, MISSING)

    def number(self, key: str):
        """The effective value at KEY when it is a number that fits the key's schema, else None: a value that does
        not fit has validate's row alone."""
        value = self.value(key)
        if not _is_number(value):
            return None
        return value if settings_file.fits(self.doc, key, value) else None

    def edit(self, key: str, value) -> Edit:
        return Edit(key, self.where, value)

    def fail(self, name: str, found: str, fix) -> None:
        self.rows.append(Check(name, Status.FAIL, found, fix))

    def warn(self, name: str, found: str, fix) -> None:
        self.rows.append(Check(name, Status.WARN, found, fix))

    def result(self, words: str) -> list:
        """The rows, or the one PASS row that says WORDS."""
        return self.rows or [Check("coherence", Status.PASS, words)]


def _at(tree, key: str, default):
    for part in key.split("."):
        if not isinstance(tree, Mapping) or part not in tree:
            return default
        tree = tree[part]
    return tree


def _instances(f: _File, path: str) -> list:
    """(key, the value the file gives it or MISSING, the effective mapping around it) for each key the schema's PATH
    names: each entry of a map ("*") is the file's own, by its name, but for one named by a placeholder.
    """
    out = []

    def walk(parts, prefix, given, values):
        if parts[0] == "*":
            names = [
                k
                for k in (given if isinstance(given, Mapping) else {})
                if not PLACEHOLDER.search(str(k))
            ]
        else:
            names = [parts[0]]
        for name in names:
            key = f"{prefix}.{name}" if prefix else str(name)
            below = given.get(name, MISSING) if isinstance(given, Mapping) else MISSING
            if len(parts) == 1:
                out.append((key, below, values if isinstance(values, Mapping) else {}))
            else:
                inner = values.get(name) if isinstance(values, Mapping) else None
                walk(parts[1:], key, below, inner)

    walk(path.split("."), "", f.given_values, f.values)
    return out


def _sibling(key: str, name: str) -> str:
    """The key NAME beside KEY, in the same section or map entry."""
    return f"{key.rsplit('.', 1)[0]}.{name}"


# --- values and words ----------------------------------------------------------------------------------------------


def _has_placeholder(value) -> bool:
    """Whether VALUE has a <...>: in its text, in a list's entry, or in a mapping's name or value."""
    if isinstance(value, str):
        return bool(PLACEHOLDER.search(value))
    if isinstance(value, list):
        return any(_has_placeholder(v) for v in value)
    if isinstance(value, Mapping):
        return any(
            _has_placeholder(str(k)) or _has_placeholder(v) for k, v in value.items()
        )
    return False


def _state(value) -> str:
    """How the file leaves a key's value out: "missing", "blank" ("" or null) or "empty" ([] or {}); "" when it gives
    one (a value with a placeholder is given)."""
    if value is MISSING:
        return "missing"
    if value is None or (isinstance(value, str) and value == ""):
        return "blank"
    if isinstance(value, (list, Mapping)) and not value:
        return "empty"
    return ""


def _is_text(value) -> bool:
    """A text that is not blank."""
    return isinstance(value, str) and value != ""


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _n(number) -> str:
    return f"{number:g}"


def _show(value) -> str:
    return settings_file.shown_value(value)


def _either(items) -> str:
    """ITEMS shown, joined with "or": "a", "b" or "c"."""
    return wrapping.joined([_show(i) for i in items], "or")


def _are(items) -> str:
    return wrapping.are(items)


def _all(edits: list):
    """EDITS as one fix: the edit, or each of them in turn."""
    if len(edits) == 1:
        return edits[0]
    parts = []
    for i, edit in enumerate(edits):
        parts += [", and"] if i else []
        parts.append(edit)
    return Steps(parts)


def _title(f: _File, feature: str) -> str:
    """A feature's title as a row's name: "follow-up tickets"."""
    title = str(
        (settings_file.schema_at(f.root, feature) or {}).get("title") or feature
    )
    return title[:1].lower() + title[1:]


def _schema_nodes(f: _File) -> dict:
    """Each key's schema by its path; a map entry's key, which a union's alternatives share, by its first."""
    nodes = {}
    for path, node, _ in settings_file.schema_keys(f.root):
        nodes.setdefault(path, node)
    return nodes


# --- Required when ---------------------------------------------------------------------------------------------------


def _guard_live(f: _File, key: str, around) -> str:
    if f.on("features.merge_queue.develop_guard") and (
        f.value("features.merge_queue.mode") == "live"
    ):
        return 'features.merge_queue.develop_guard is on, with features.merge_queue.mode "live"'
    return ""


def _publishing(f: _File, key: str, around) -> str:
    if f.value(TEAM + ".delivery") == "publish":
        return f'{TEAM}.delivery is "publish"'
    return ""


def _every_address(address) -> bool:
    """Whether ADDRESS is "0.0.0.0" or "::", each of this host's addresses."""
    try:
        return ipaddress.ip_address(str(address)).is_unspecified
    except ValueError:
        return False


def _public_url(f: _File, key: str, around) -> str:
    address = f.value(TEAM + ".serve.address")
    if f.value(TEAM + ".delivery") == "serve" and _every_address(address):
        return f"{TEAM}.serve.address is {_show(address)}"
    return _publishing(f, key, around)


def _header_needed(f: _File, key: str, around) -> str:
    sent_as = around.get("api_key_sent_as")
    if sent_as in ("header", "header_and_bearer"):
        return f"{_sibling(key, 'api_key_sent_as')} is {_show(sent_as)}"
    return ""


def _url_set(f: _File, key: str, around) -> str:
    if _is_text(around.get("package_url")):
        return f"{_sibling(key, 'package_url')} is set"
    return ""


# The keys Required when a value of another key decides it (x-required-when), by their schema's path, each with the
# function that says why it is required now ("" when it is not). A test checks they are the schemas' own.
DEPLOY_REQUIRED = {
    "features.merge_queue.develop_guard.ci_red_label": _guard_live,
    "features.merge_queue.develop_guard.stability_label": _guard_live,
    TEAM + ".public_url": _public_url,
    TEAM + ".publish.web_host": _publishing,
    TEAM + ".publish.username": _publishing,
    TEAM + ".publish.site_folder": _publishing,
    BINDINGS + ".*.api_key_header": _header_needed,
}
BUILD_REQUIRED = {
    "rocm.package_sha256": _url_set,
    "github_cli.package_sha256": _url_set,
    "claude_agent_sdk.package_sha256": _url_set,
}


def _required(f: _File, table: dict) -> None:
    """The keys of TABLE that are left out or blank while they are required."""
    nodes = _schema_nodes(f)
    for path, why in table.items():
        for key, given, around in _instances(f, path):
            state = _state(given)
            if not state or not f.used(key):
                continue
            reason = why(f, key, around)
            if reason:
                words = settings_file.source_of(nodes[path])
                f.fail(key, f"{state}, and {reason}", f.edit(key, words))


def _blank_while_on(f: _File) -> None:
    """The keys Required when their feature is on, given blank or empty while it is on."""
    for path, node in _schema_nodes(f).items():
        if not node.get("x-required-when") or "default" in node or "x-blank" in node:
            continue
        for key, given, _around in _instances(f, path):
            state = _state(given)
            if state in ("", "missing"):
                continue  # given, or missing: validate's row
            on = settings_file.required_by(f.root, f.given_values, key, node)
            if on:
                found = f"{state}, and {wrapping.joined(on)} {_are(on)} on"
                f.fail(key, found, f.edit(key, settings_file.source_of(node)))


# --- the deploy file -----------------------------------------------------------------------------------------------


def _dependencies(f: _File) -> None:
    """Check 2: a feature that is on while the feature it needs is off."""
    for feature, needed in NEEDS:
        if not f.on(feature) or f.on(needed):
            continue
        parts = needed.split(".")
        off = [
            switch
            for switch in (".".join(parts[: i + 1]) for i in range(len(parts)))
            if settings_file.is_feature(settings_file.schema_at(f.root, switch))
            and f.given(switch + ".enabled") is not True
        ]
        fix = [f.edit(f"{feature}.enabled", False), ", or turn on what it needs:"]
        for i, switch in enumerate(off):
            fix += [", and"] if i else []
            fix.append(f.edit(f"{switch}.enabled", True))
        found = f"{feature} is on, and it needs {needed}, which is off"
        f.fail(_title(f, feature), found, Steps(fix))


def _bindings(f: _File) -> dict:
    """The file's model bindings by name, with their effective values; an entry named by a placeholder is left out."""
    entries = f.values.get(BINDINGS) or {}
    return {
        name: entry
        for name, entry in entries.items()
        if isinstance(entry, Mapping) and not PLACEHOLDER.search(str(name))
    }


def _model(name: str, bindings: dict, builtin: dict):
    """(provider, model id) of the model NAME, or None when there is none: its binding, but where a built-in's name
    has a binding of another provider than "claude", the built-in stands (the binding has its own row).
    """
    entry = bindings.get(name)
    if entry is not None and (name not in builtin or entry.get("provider") == CLAUDE):
        return entry.get("provider"), entry.get("model_id")
    if name in builtin:
        return CLAUDE, builtin[name]
    return None


def _passes(f: _File) -> list:
    """(the pass's key, whether it is optional) for the orchestrator and each pass, in the schema's order: an optional
    pass's model is "default" until you choose one."""
    return [
        (path[: -len(".model")], node.get("default") == deploy_tree.DEFAULT_MODEL)
        for path, node in _schema_nodes(f).items()
        if path.startswith(RA + ".") and path.endswith(".model")
    ]


def _check_passes(f: _File, bindings: dict, builtin: dict) -> None:
    """Check 3: each pass's model exists, is Claude's where the pass is not optional, and takes the pass's effort."""
    claude_names = list(builtin) + [
        n
        for n, e in bindings.items()
        if e.get("provider") == CLAUDE and n not in builtin
    ]
    every_name = list(builtin) + [n for n in bindings if n not in builtin]
    for path, optional in _passes(f):
        key, effort_key = path + ".model", path + ".effort"
        name = f.value(key)
        if not isinstance(name, str) or _has_placeholder(name):
            continue
        if optional and name == deploy_tree.DEFAULT_MODEL:
            continue  # no other model: the pass in charge does the work
        model = _model(name, bindings, builtin)
        if model is None:
            choices = (
                [deploy_tree.DEFAULT_MODEL, *every_name] if optional else claude_names
            )
            found = (
                f"{_show(name)} is neither a built-in model nor an entry of {BINDINGS}"
            )
            f.fail(key, found, f.edit(key, Described(_either(choices))))
            continue
        provider = model[0]
        if not optional and provider != CLAUDE:
            found = (
                f"{_show(name)} is an entry of {BINDINGS} with provider {_show(provider)}, and this pass takes "
                "a Claude model"
            )
            f.fail(key, found, f.edit(key, Described(_either(claude_names))))
            continue
        # A model that is not Claude's runs only on an optional pass, where "default" is an effort too.
        takes = [deploy_tree.DEFAULT_MODEL, *deploy_tree.OTHER_EFFORTS]
        effort = f.value(effort_key)
        if (
            provider == CLAUDE
            or effort in takes
            or not isinstance(effort, str)
            or _has_placeholder(effort)
        ):
            continue
        found = (
            f"{_show(effort)}, and the pass's model, {_show(name)}, has provider {_show(provider)}, which takes "
            f"only {_either(deploy_tree.OTHER_EFFORTS)}"
        )
        f.fail(effort_key, found, f.edit(effort_key, Described(_either(takes))))


def _check_binding_names(f: _File, bindings: dict, builtin: dict) -> None:
    """Check 3: a built-in model's name ("opus", "sonnet", "haiku") takes only a Claude binding."""
    for name, entry in bindings.items():
        provider = entry.get("provider")
        if name in builtin and provider != CLAUDE:
            key = f"{BINDINGS}.{name}"
            found = (
                f"{_show(name)} is a built-in model's name, which takes only a binding with provider "
                f"{_show(CLAUDE)}, and this entry's provider is {_show(provider)}"
            )
            words = f"rename {key} in {f.where}, and each pass that names it, to a name of your own, then run"
            f.fail(key, found, Steps((words, f.again)))


def _check_headers(f: _File, bindings: dict) -> None:
    """Check 4: "header_and_bearer" with the header Authorization sends the key twice in one header."""
    senders = []
    if f.used(CLAUDE_CODE + ".api_key_header"):
        senders.append((CLAUDE_CODE, f.value(CLAUDE_CODE) or {}))
    senders += [(f"{BINDINGS}.{name}", entry) for name, entry in bindings.items()]
    for prefix, values in senders:
        header = values.get("api_key_header")
        if values.get("api_key_sent_as") != "header_and_bearer" or not (
            isinstance(header, str) and header.strip().lower() == "authorization"
        ):
            continue
        key, sent_as = prefix + ".api_key_header", prefix + ".api_key_sent_as"
        found = (
            f'{_show(header)}, and {sent_as} is "header_and_bearer", which sends a bearer token in that header too: '
            "the key would go twice in one header"
        )
        fix = Steps(
            (
                f.edit(sent_as, "bearer"),
                ", or",
                f.edit(key, Described("another header")),
            )
        )
        f.fail(key, found, fix)


def _check_spend(f: _File, bindings: dict, builtin: dict) -> None:
    """Check 5: while spend is on, each model the passes use has its prices."""
    if not f.on(SPEND) or _has_placeholder(f.given(PRICES)):
        return
    prices = f.value(PRICES) or {}
    used = {}  # model id -> the names the passes give it
    for path, _optional in _passes(f):
        name = f.value(path + ".model")
        if not isinstance(name, str) or _has_placeholder(name):
            continue
        model = _model(name, bindings, builtin)
        if model is None or not _is_text(model[1]) or _has_placeholder(model[1]):
            continue
        names = used.setdefault(model[1], [])
        if name not in names:
            names.append(name)
    unpriced = [i for i in used if i not in prices]
    if not unpriced:
        return
    one = len(unpriced) == 1
    ids = wrapping.joined(
        f"{i} ({wrapping.joined(_show(n) for n in used[i])})" for i in unpriced
    )
    found = f"{PRICES} has no entry for {ids}, " + (
        "a model the passes use: its cost shows as unknown"
        if one
        else "models the passes use: their cost shows as unknown"
    )
    entry = settings_file.schema_at(f.root, PRICES + ".*")
    words = str(entry.get("description") or "").strip().rstrip(".")
    words = f"{words}: {wrapping.joined(entry.get('required') or ())}"
    edits = [
        f.edit(f"{PRICES}.{i}", Described(words if not n else "that model's prices"))
        for n, i in enumerate(unpriced)
    ]
    f.warn("spend", found, _all(edits))


def _check_team(f: _File) -> None:
    """Check 7: "publish" takes no serve block, and "serve" no publish block; a serve block left out means its
    defaults (settings_file.effective); while the web host needs the request service, an address it serves.
    """
    if not f.on(TEAM):
        return
    given = f.given(TEAM)
    given = given if isinstance(given, Mapping) else {}
    delivery_key, serve, publish = (
        TEAM + ".delivery",
        TEAM + ".serve",
        TEAM + ".publish",
    )
    delivery = f.value(delivery_key)
    if delivery == "publish" and "serve" in given:
        found = f'{serve} is set, and {delivery_key} "publish" takes no serve block'
        f.fail(
            TEAM_ROW,
            found,
            Steps((f.edit(serve, DELETE), ", or", f.edit(delivery_key, "serve"))),
        )
    if delivery == "serve" and "publish" in given:
        found = f'{publish} is set, and {delivery_key} "serve" takes no publish block'
        f.fail(
            TEAM_ROW,
            found,
            Steps((f.edit(publish, DELETE), ", or", f.edit(delivery_key, "publish"))),
        )
    if delivery == "publish" and f.on(TEAM + ".priority_requests"):
        _check_served_address(f)


def _check_served_address(f: _File) -> None:
    """Check 7, while the web host needs the request service (priority requests on, delivery "publish"): public_url an
    address the service serves the page at (web_host_install.service_address), so the page the Needs you alerts link to
    is the one it serves; a blank one is the Required when row's, one with a placeholder validate's.
    """
    from . import web_host_install  # which imports this module

    key = TEAM + ".public_url"
    url = f.value(key)
    if not _is_text(url) or _has_placeholder(url):
        return
    try:
        web_host_install.service_address(url)
    except web_host_install.Unserved as exc:
        give = exc.give or Described(web_host_install.GIVE)
        f.fail(TEAM_ROW, str(exc), f.edit(key, give))


def _check_users(f: _File) -> None:
    """Check 8: no id that the container refuses at each start. docker/rk_ids.py's clashes, the one home of the ids'
    rules, finds them; this check words them (a test holds it, and the schema's minimum and maximum, to the entry
    point's). These settings give the monitor's ids alone: the GitHub writer's and the PR runners' are the package's
    (rk_ids.settings_ids), and rk_ids.SETTING names their keys. The schema's minimum keeps out the system's user ids
    below it; the monitor's group may be any group but the PR runners' (theirs and each runner's own), a system group
    included, as the entry point takes it."""
    ids = deployment.rk_ids()
    keys = ids.SETTING
    given = {k: f.value(key) for k, key in keys.items()}
    known = {k: v if _is_number(v) else None for k, v in given.items()}
    # The package's own ids never clash (tests/test_entrypoint.py): each clash is the monitor's uid's or gid's.
    for c in ids.clashes(ids.settings_ids(known["uid"], known["gid"])):
        key = keys[c.key]
        node = settings_file.schema_at(f.root, key)
        found = f"{given[c.key]} is {c.why}, which the container refuses at each start"
        words = f"{c.what}, or to null, which means {node.get('x-default-means')}"
        f.fail(key, found, f.edit(key, Described(words)))


def _on_loopback(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return False
    host = (host or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_loopback(f: _File, bindings: dict) -> None:
    """Check 11: an address on the loopback, which inside the container is the container."""
    addresses = []
    if f.used(CLAUDE_CODE + ".base_url"):
        addresses.append(
            (CLAUDE_CODE + ".base_url", f.value(CLAUDE_CODE + ".base_url"))
        )
    for name, entry in bindings.items():
        addresses += [
            (f"{BINDINGS}.{name}.{k}", entry[k])
            for k in ("base_url", "server_url")
            if k in entry
        ]
    for key, url in addresses:
        if _is_text(url) and not _has_placeholder(url) and _on_loopback(url):
            found = f"{_show(url)} is on the loopback: inside the container, localhost is the container itself"
            words = (
                "an address the container can reach, such as this host's own address"
            )
            f.warn(key, found, f.edit(key, Described(words)))


def _check_repository_defaults(f: _File) -> None:
    """Check 12: repository defaults for the watched repository and folder, else a WARN naming what is generic."""
    name, folder = f.value("repository.name"), f.value("repository.folder")
    if not (_is_text(name) and _is_text(folder)) or _has_placeholder([name, folder]):
        return
    generic = repository_defaults.generic(name, folder)
    if generic:
        shows = Command((program.name(), "rk", "config", "show", "--source"))
        f.warn(
            "repository defaults",
            repository_defaults.generic_found(name, folder, generic),
            Steps(
                (
                    shows,
                    "shows each value the monitor runs with and where it comes from",
                )
            ),
        )


def _check_gpu(f: _File) -> None:
    """Check 17: while the remote GPU tests are on, the default target one of gpu_types' targets."""
    if not f.on(GPU):
        return
    types_key = f"{GPU}.cluster.gpu_types"
    gpu_types = f.value(types_key)
    gpu_types = gpu_types if isinstance(gpu_types, Mapping) else {}
    target_key = f"{GPU}.cluster.default_target"
    target = f.value(target_key)
    if (
        _is_text(target)
        and not _has_placeholder(target)
        and gpu_types
        and not _has_placeholder(gpu_types)
        and target not in gpu_types
    ):
        blank = settings_file.schema_at(f.root, target_key).get("x-blank")
        words = f'{_either(gpu_types)}, or to "" {blank}'
        f.fail(
            target_key,
            f"{_show(target)} is not in {types_key}",
            f.edit(target_key, Described(words)),
        )


def _check_shared_labels(f: _File) -> None:
    """Check 20: no label both in scope_labels, which puts every pull request with it in scope, and in
    shared_label_folders, which puts only some in."""
    labels = f.value("repository.scope_labels")
    folders = f.value("repository.shared_label_folders")
    if not isinstance(labels, list) or not isinstance(folders, Mapping):
        return
    both = [
        label
        for label in dict.fromkeys(labels)
        if isinstance(label, str) and not _has_placeholder(label) and label in folders
    ]
    if not both:
        return
    one = len(both) == 1
    shown = wrapping.joined(_show(label) for label in both)
    found = (
        f"{shown} {_are(both)} in both repository.scope_labels and repository.shared_label_folders: the first puts "
        f"every pull request with {'it' if one else 'one of them'} in scope, the second only some"
    )
    parts = []
    # The list without them, only when it keeps a label: scope_labels needs one at least.
    if not all(label in both for label in labels):
        without = Described(f"the list without {shown}")
        parts += [f.edit("repository.scope_labels", without), ", or"]
    for i, label in enumerate(both):
        parts += [", and"] if i else []
        parts.append(f.edit(f"repository.shared_label_folders.{label}", DELETE))
    f.fail("shared labels", found, parts[0] if len(parts) == 1 else Steps(parts))


def deploy(doc) -> list:
    """A deploy settings document's checks across keys: see the module's docstring."""
    f = _File(doc)
    _required(f, DEPLOY_REQUIRED)
    _blank_while_on(f)
    _dependencies(f)
    if f.on(RA):
        bindings, builtin = _bindings(f), deploy_tree.builtin()
        _check_passes(f, bindings, builtin)
        _check_binding_names(f, bindings, builtin)
        _check_headers(f, bindings)
        _check_spend(f, bindings, builtin)
        _check_loopback(f, bindings)
    _check_team(f)
    _check_users(f)
    _check_repository_defaults(f)
    _check_gpu(f)
    _check_shared_labels(f)
    return f.result("the features' settings fit together") + _team_in_prototype(f)


def _team_in_prototype(f: _File) -> list:
    """Check 7's PASS row of its own while the team dashboard is served and its port is published on another address
    than serve.address says: in prototype, this host's own alone (features.team_host). It stands beside
    the other rows, whatever they are: none of them changes where the page is served.
    """
    address = f.value(TEAM + ".serve.address")
    published = features.team_host(f.values, address)
    if f.on(TEAM) and f.value(TEAM + ".delivery") == "serve" and published != address:
        found = (
            f"in prototype, its page is served on {published} only, not on {_show(address)} "
            f"({TEAM}.serve.address), until it is live"
        )
        return [Check(TEAM_ROW, Status.PASS, found)]
    return []


# --- the container's share of this host (checks 8 and 9) -------------------------------------------------------------


class Least(NamedTuple):
    """What the container takes at least (`least`): see the module's docstring."""

    need: float  # the RAM disk the reviews and builds need, in GiB
    about: int  # that RAM disk in whole GiB, rounded up
    memory: float  # the memory limit that holds the RAM disk asked about and the work
    cpus: int  # a CPU for each review and each build


def least(ramdisk=0, reviews: int = 0, builds: int = 0) -> Least:
    """What the container takes at least: see the module's docstring."""
    need = deployment.PIPE_GB * reviews + deployment.HEAVY_GB * builds
    return Least(need, math.ceil(need), ramdisk + deployment.WORK_GB, reviews + builds)


def _ramdisk_at_most(f: _File, room, start) -> Edit:
    """The edit that leaves the RAM disk ROOM GiB or less; or none, when ROOM is less than START, the GiB the starting
    reviews and builds need (0: not known), so the memory's fix never offers a RAM disk the RAM disk's row warns of.
    """
    if room > 0 and room >= start:
        return f.edit(RAMDISK, Described(f"{_n(room)} or less"))
    return f.edit(RAMDISK, Described("0 for no RAM disk"))


def _ramdisk(f: _File, ramdisk, start: Least, memory, host) -> None:
    """Check 8's WARN: a RAM disk smaller than the starting reviews and builds need (START, what they take). The fix
    is a RAM disk that size while the memory holds it beside the work; one that raises the memory too
    while it would not; and only none while this host has no room for both."""
    about = start.about
    room = least(about).memory  # the memory a RAM disk that size takes, with the work's
    found = f"{RAMDISK} {_n(ramdisk)} is less than the starting reviews and builds need (about {about} GiB)"
    none = f.edit(RAMDISK, Described("0 for no RAM disk"))
    if host and room > host:
        fix = none
    elif memory is None or room <= memory:
        fix = f.edit(RAMDISK, Described(f"{about} or more, or to 0 for no RAM disk"))
    else:
        larger = f.edit(RAMDISK, Described(f"{about} or more"))
        more = f.edit(MEMORY, Described(f"{_n(room)} or more"))
        fix = Steps((larger, ", and", more, "; or", none))
    f.warn(deployment.RAM_DISK_ROW, found, fix)


def _memory(f: _File, memory, ramdisk, start, host) -> None:
    """Check 8: the memory holds the RAM disk and the work, and this host has it. Each fix leads to a value that
    passes, so the advice never goes round in a circle; a smaller RAM disk is one the starting reviews and builds have
    room on (START GiB, 0: not known), else none."""
    work = deployment.WORK_GB
    want = least(ramdisk).memory
    # No value fits when the work alone needs more than this host has.
    if host and host < work:
        if memory > host:
            found = (
                f"{MEMORY} {_n(memory)} is more than this host has ({_n(host)} GiB), and the work needs {work} GiB "
                "at least"
            )
        else:
            found = f"{MEMORY} {_n(memory)} is less than the {work} GiB the work needs, and this host has {_n(host)} GiB"
        f.fail(
            "memory",
            found,
            Steps(("run", f.again, f"on a host with {work} GiB or more")),
        )
    elif host and memory > host:
        edits = [f.edit(MEMORY, Described(f"{_n(host)} or less"))]
        if want > host:  # the RAM disk takes room the work needs
            edits.append(_ramdisk_at_most(f, host - work, start))
        found = f"{MEMORY} {_n(memory)} is more than this host has ({_n(host)} GiB)"
        f.fail("memory", found, _all(edits))
    elif memory < want:
        if ramdisk:
            found = f"{MEMORY} {_n(memory)} is less than {RAMDISK} ({_n(ramdisk)}) plus the {work} GiB the work needs"
        else:
            found = f"{MEMORY} {_n(memory)} is less than the {work} GiB the work needs"
        more = f.edit(MEMORY, Described(f"{_n(want)} or more"))
        smaller = _ramdisk_at_most(f, memory - work, start)
        raise_memory = not host or want <= host
        lower_ramdisk = ramdisk and memory >= work
        if raise_memory and lower_ramdisk:
            fix = Steps((more, ", or", smaller))
        elif raise_memory:
            fix = more
        elif lower_ramdisk:
            fix = smaller
        else:  # neither alone: all of this host's memory, and a RAM disk that leaves the work its share
            fix = _all(
                [
                    f.edit(MEMORY, Described(_n(host))),
                    _ramdisk_at_most(f, host - work, start),
                ]
            )
        f.fail("memory", found, fix)
    elif host:
        found = f"the container may use {_n(memory)} of this host's {_n(host)} GiB"
        f.rows.append(Check("memory", Status.PASS, found))


def _cpus(f: _File, cpus, reviews, builds, host) -> None:
    if host and cpus > host:
        f.fail(
            "CPUs",
            f"{CPUS} {_n(cpus)} is more than this host has ({host})",
            f.edit(CPUS, Described(f"{host} or fewer")),
        )
        return
    need = least(reviews=reviews, builds=builds).cpus
    if cpus >= need:
        return
    if not host or need <= host:
        fix = f.edit(CPUS, need)
    else:  # the host has fewer CPUs than they need: fewer reviews and builds at first
        edits = [f.edit(CPUS, host)] if cpus < host else []
        fit_builds = min(builds, max(1, host - (1 if reviews else 0)))
        fit_reviews = min(reviews, max(1, host - fit_builds)) if reviews else 0
        if fit_reviews < reviews:
            edits.append(f.edit(PARALLEL_REVIEWS, fit_reviews))
        if fit_builds < builds:
            edits.append(f.edit(PARALLEL_BUILDS, fit_builds))
        if not edits:
            return  # the fewest there can be, on a host this small: the monitor adjusts them
        fix = _all(edits)
    found = f"{CPUS} {_n(cpus)} is fewer than the reviews and builds that may run at once ({_n(need)})"
    f.warn("CPUs", found, fix)


def resources(doc, host_memory_gib: float = 0.0, host_cpus: int = 0) -> list:
    """The container's share of this host: see the module's docstring."""
    f = _File(doc)
    reviews = f.number(PARALLEL_REVIEWS) if f.on(RA) else 0
    builds = f.number(PARALLEL_BUILDS)
    cpus, memory, ramdisk = (f.number(k) for k in (CPUS, MEMORY, RAMDISK))
    about = 0  # the RAM disk the starting reviews and builds need, in whole GiB (0: not known)
    if reviews is not None and builds is not None:
        start = least(reviews=reviews, builds=builds)
        about = start.about
        if ramdisk and ramdisk < start.need:
            _ramdisk(f, ramdisk, start, memory, host_memory_gib)
    if memory is not None and ramdisk is not None:
        _memory(f, memory, ramdisk, about, host_memory_gib)
    if cpus is not None and reviews is not None and builds is not None:
        _cpus(f, cpus, reviews, builds, host_cpus)
    return f.rows


# --- the team's skills folder (check 18) ----------------------------------------------------------------------------

# The sentence of SKILLS_KEY's description that names the places a file of the folder may be (skills_places), and
# the form of each place: a folder's path with its /, or a file's.
_PLACES = re.compile(r"\bmay contain only (.+?)\.(?=\s|$)")
_PLACE = re.compile(r"[\w.-]+(?:/[\w.-]+)*/?")
NOT_A_PLACE = "at none of those places"


def skills_places() -> tuple:
    """The places a file of a team's skills folder may be, as SKILLS_KEY's description names them: see the module's
    docstring."""
    m = _PLACES.search(deploy_tree.description(SKILLS_KEY))
    places = tuple(re.split(r", | and ", m.group(1))) if m else ()
    if not places or not all(_PLACE.fullmatch(p) for p in places):
        raise ValueError(
            f'{SKILLS_KEY}: its description has no "it may contain only A, B and C.", which names the places a '
            "file of the folder may be"
        )
    return places


def _at_a_place(rel: str, places) -> bool:
    """Whether the file REL, its path in the folder with /, is at one of PLACES: one of their files, or in one of their
    folders."""
    return any(rel.startswith(p) if p.endswith("/") else rel == p for p in places)


def _on_the_way(folder: str, places) -> bool:
    """Whether the folder FOLDER, its path in the folder with its /, is one of PLACES' folders, in one of them, or on
    the way to one of PLACES."""
    return any(
        p.startswith(folder) or (p.endswith("/") and folder.startswith(p))
        for p in places
    )


def _team_files(top: pathlib.Path, places) -> tuple:
    """(each Markdown file of the folder TOP at one of PLACES, by its path in TOP with / -> its text; (its path, why)
    for each other thing TOP has), in the order of their paths. A folder at none of PLACES is named once, with its /,
    and not looked into; a link is named, never followed. A file is read as the delivery reads it
    (deploy_plan.regular_bytes)."""
    from . import (
        deploy_plan,
    )  # file_types, which imports this module, imports it: read at the call

    texts, refused = {}, []

    def walk(folder: str, prefix: str) -> None:
        try:
            with os.scandir(folder) as found:
                entries = sorted(found, key=lambda e: e.name)
        except OSError as exc:
            refused.append((prefix, f"cannot be read: {exc.strerror or exc}"))
            return
        for e in entries:
            rel = prefix + e.name
            if e.is_symlink():
                refused.append((rel, "a link"))
            elif e.is_dir(follow_symlinks=False):
                if _on_the_way(rel + "/", places):
                    walk(e.path, rel + "/")
                else:
                    refused.append((rel + "/", NOT_A_PLACE))
            elif not e.is_file(follow_symlinks=False):
                refused.append((rel, "not a regular file"))
            elif not _at_a_place(rel, places):
                refused.append((rel, NOT_A_PLACE))
            elif not rel.endswith(".md"):
                refused.append((rel, "not a Markdown file"))
            else:
                data = deploy_plan.regular_bytes(pathlib.Path(e.path))
                if data is None:
                    refused.append((rel, "cannot be read"))
                    continue
                try:
                    texts[rel] = data.decode("utf-8")
                except UnicodeDecodeError:
                    refused.append((rel, "not UTF-8 text"))

    walk(str(top), "")
    return texts, refused


def _folder_row(f: _File, top: pathlib.Path, package_dir):
    """Check 18's FAIL row of the folder TOP itself, or None: it is a folder, outside the package folder PACKAGE_DIR,
    and you can read it."""
    shown = settings_file.shown(top)
    if not top.is_dir():
        found = f"{SKILLS_KEY} {shown}: {'not a folder' if top.exists() else 'no such folder'}"
        named = f.edit(SKILLS_KEY, Described("the folder your team's skills are in"))
        blank = f.edit(SKILLS_KEY, "")
        return Check(
            SKILLS,
            Status.FAIL,
            found,
            Steps((named, ", or", blank, "for the package's skills only")),
        )
    package = os.path.realpath(package_dir)
    if os.path.commonpath([os.path.realpath(top), package]) == package:
        found = f"{SKILLS_KEY} {shown}: inside the package folder, which a package update replaces"
        moved = f.edit(SKILLS_KEY, Described("its new place"))
        return Check(
            SKILLS,
            Status.FAIL,
            found,
            Steps((f"move it out of {settings_file.shown(package)}, then", moved)),
        )
    try:
        with os.scandir(top):
            pass
    except OSError as exc:
        found = f"{SKILLS_KEY} {shown}: you cannot read it ({exc.strerror or exc})"
        fix = (
            "make it and its files readable by you, such as",
            Command(("chmod", "-R", "u+rX", shown)),
            ", then run",
        )
        return Check(SKILLS, Status.FAIL, found, Steps((*fix, f.again)))
    return None


def skills(doc, package_dir) -> list:
    """Check 18, the team's skills folder on this host: see the module's docstring."""
    from . import (
        deploy_plan,
    )  # file_types, which imports this module, imports it: read at the call

    f = _File(doc)
    top = deploy_plan.skills_path(deploy_plan.settings(doc))
    if top is None or _has_placeholder(f.value(SKILLS_KEY)):
        return []
    row = _folder_row(f, top, package_dir)
    if row is not None:
        return [row]
    shown = settings_file.shown(top)
    places = skills_places()
    texts, refused = _team_files(top, places)
    if refused:
        found = (
            f"{SKILLS_KEY} {shown} may contain only Markdown files, each at {wrapping.joined(places, 'or')}, and "
            f"it has {wrapping.joined(f'{rel} ({why})' for rel, why in refused)}"
        )
        fix = (
            "take each out of",
            shown,
            "(a link: put the file itself in its place), then run",
            f.again,
        )
        f.fail(SKILLS, found, Steps(fix))
    verifier = texts.get(verifier_file.FILE)
    lacks = [] if verifier is None else verifier_file.missing(verifier)
    if lacks:
        one = len(lacks) == 1
        found = (
            f"{SKILLS_KEY} {shown}: {verifier_file.FILE} lacks the checklist {'lead' if one else 'leads'} "
            f"{wrapping.joined(f'**{lead}**' for lead in lacks)}, which a review in parts quotes from it: a part "
            f"would say {'it is' if one else 'they are'} missing"
        )
        fix = (
            f"start a bullet of its What to Check with {'it' if one else 'each'} in bold, as the package's "
            f"agentic/{verifier_file.FILE} does, then run",
            f.again,
        )
        f.fail(SKILLS, found, Steps(fix))
    if f.rows:
        return f.rows
    bundle = pathlib.Path(package_dir) / "agentic"
    replaced = [rel for rel in texts if (bundle / rel).is_file()]
    added = [rel for rel in texts if rel not in replaced]
    does = [f"replace the package's {wrapping.joined(replaced)}"] if replaced else []
    does += [f"add {wrapping.joined(added)}"] if added else []
    found = (
        f"the team's skills in {shown} {', and '.join(does)}"
        if does
        else f"{shown} has no file: the package's skills run as they are"
    )
    return [Check(SKILLS, Status.PASS, found)]


# --- the build file ------------------------------------------------------------------------------------------------


def _not_checked(f: _File) -> None:
    """Check 4: a blank SHA-256 where nothing else checks the download."""

    def source(key: str):
        return settings_file.source_of(settings_file.schema_at(f.root, key))

    if not _is_text(f.value("rocm.package_url")) and not _is_text(
        f.value("rocm.package_sha256")
    ):
        key = "rocm.package_sha256"
        found = "not checked; the build installs AMD's public installer for this version without checking it"
        f.warn(key, found, f.edit(key, source(key)))
    url, key = "claude_agent_sdk.package_url", "claude_agent_sdk.package_sha256"
    if not _is_text(f.value(url)) and not _is_text(f.value(key)):
        found = "not checked; the build installs the SDK from the package index without checking it"
        fix = (f.edit(url, source(url)), ", and", f.edit(key, source(key)))
        f.warn(key, found, Steps(fix))


def _mirror(f: _File) -> None:
    """Check 5: a package mirror is for an Ubuntu base image only."""
    mirror_key, base_key = (
        "operating_system.package_mirror",
        "operating_system.base_image",
    )
    mirror, base = f.value(mirror_key), f.value(base_key)
    if (
        not _is_text(mirror)
        or _has_placeholder(mirror)
        or not _is_text(base)
        or _has_placeholder(base)
    ):
        return
    if image.family(base) == "deb":
        return
    ubuntu = [
        b
        for b in settings_file.schema_at(f.root, base_key).get("enum") or ()
        if image.family(b) == "deb"
    ]
    found = f"set, and {base_key} {_show(base)} is not an Ubuntu image: a package mirror is for Ubuntu base images only"
    f.fail(
        mirror_key,
        found,
        Steps(
            (
                f.edit(mirror_key, DELETE),
                ", or",
                f.edit(base_key, Described(_either(ubuntu))),
            )
        ),
    )


def build(doc) -> list:
    """A build settings document's checks across keys: see the module's docstring."""
    f = _File(doc)
    _required(f, BUILD_REQUIRED)
    _not_checked(f)
    _mirror(f)
    return f.result("the build settings fit together")
