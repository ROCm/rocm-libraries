# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The round trip of production's configuration: the deploy settings file the translation makes
(tools/translate_production.py) taken back to today's names, and compared with production's config.json and the
shape of its older Claude secret (secrets_registry.CLAUDE_ENV).

The key is never shown: KEY_MARK, rkmon_lib/custom_headers.py's, stands for it wherever a header or a variable sends it,
as in the bindings' key headers, here and in the translation's report.

The inverse map is pure: it reads no file, no network and no clock.

- `inverse(values) -> Inverse(config, shape)`: the values of a deploy settings file of this layout
  (settings_file.Document.values) in today's names. CONFIG is the configuration they make (deploy_tree.to_config):
  each key's value under its x-internal, and the rules that take a value back, the image and the RAM disk among them;
  then the shared labels in today's form (`labels_back`). SHAPE is the shape of the older Claude secret that
  integrations.claude_code's settings stand for (`secret_shape`).
- `labels_back(cfg)`: CFG with its shared labels (the configuration key repository.shared_label_folders feeds) as
  today's list of labels when each label counts for the repository's own folder (paths.repository_folder), the one
  folder each counts for in today's form (config.scope); a label for another folder keeps the map, which the list
  cannot say. CFG itself is not changed.
- `secret_shape(cfg) -> claude_env_shape.Shape`: the shape of the older Claude secret that the configuration's
  Claude settings stand for (the configuration keys integrations.claude_code's keys feed): the variables the session
  gets from them (orchestrator.session_variables), KEY_MARK in place of the key, read as claude_env_shape.read reads
  the secret. It is the inverse of claude_env_shape.settings: each shape it takes comes back from the settings made
  from it, but for its model variables and its nonessential traffic switch, which no setting carries.
- `internal(path)`: the one configuration key the file key PATH feeds, from its x-internal (deploy_tree.feeds).

The comparisons are pure too. `compare(production, shape, values, usage=None) -> [Difference]`:
PRODUCTION is config.json as it is written, SHAPE its older Claude secret's shape (claude_env_shape.read),
VALUES the translated file's values, USAGE one recorded week (`Usage`). Each Difference(key, kind, production, after)
is one place where what production runs and what the file runs differ, MISSING on the side that has no value; in the
order of KINDS, then by key:

- VALUE: each configuration key the file feeds (deploy_tree.held), production's as production ran it (`ran`)
  against the inverse map's, entry by entry; a key one side leaves out at its default (registry.value), and each of
  the local worker's settings production sets and the file leaves out, while the file binds no drafting model, at
  NO_DRAFTING_MODEL, which the list's row of those settings names. The Claude session's settings (the keys
  integrations.claude_code feeds) are compared as the session gets them instead: its base address, its model
  variables and its nonessential traffic switch (SESSION_VALUES), each by its variable's name, the model variables as
  SET (the shape holds no value); and its headers (HEADER).
- FEATURE: each capability's value as rkmon_lib/features.py reads it (on or off, the auto-merge guard's and the team
  dashboard's own values), today's configuration's, with the switches production's code read as it read them
  (`switches_ran`), against the running one's.
- HEADER: the headers each request of the Claude session sends (orchestrator.request_headers): today's from
  the secret's shape and the configured gateway (`shape_variables`), the running configuration's from
  integrations.claude_code (orchestrator.session_variables); and those of the first review pass's requests
  (worker/llm_client.py auth_headers), from the settings of the model that drafts it: today's from config.json's
  `worker`, the running configuration's from the binding. Each header "Name: value", in the order sent, the key as
  KEY_MARK.
- MODEL_ID: the model id each pass and the session ask for: the Claude passes' (spend.ASKED's roles) by
  config.claude_model_id; the first review pass's and each optional pass's, by the model that runs it, None for none.
- EFFORT: each effort as the requests carry it (config.EFFORT_ROLES): a Claude role's by config.role_effort; an
  optional pass's by its model (rkmon_lib/first_pass.py effort: a Claude binding's, the role's; another model's, the
  role's own, when it has one); the first review pass's as its drafter's requests carry it (bindings.drafter_effort,
  the rule's one home, which line 1 reads too). None where none is sent.
- LINE_1: line 1 of a review, by its derived key (FIRST_LINES): the drafter's, and the verifier's, posted with no
  first-pass drafter (the verifier reviews alone). Today's is the line today's code builds from today's configuration,
  the running configuration's its own; each is compared apart from the automated reviewer name it gives (`first_lines`),
  which is a value of its own (features.review_automation.automated_reviewer_name, else container.name). So a line that
  gives another model or effort is a difference, even where the model ids and the efforts sent are the same.
- INTERNAL: each value of the running configuration that the file does not feed (the package's, the state's, the
  derived ones but line 1's, and the keys the registry no longer has), today's against the running one's, entry by
  entry; and each key of config.json whose value the deploy plan makes now (PLANNED), today's against the plan's
  (`planned`).
- PRICE: one recorded week of usage priced by today's records and by this package's (spend.usd): today's by model
  family at production's prices, the running configuration's by the model id each pass asks for at the file's
  prices; the total of each day, of each pass and of the week, to the cent's ten-thousandth. Both come from one read
  of the transcripts (`usage_of`), so they price the same requests while the monitor writes more.

Today's configuration is config.json as production ran it (`ran`), as today's code reads it (config.normalize). Its
internal values are compared as the package reads them (`at_package`): each key it leaves out at the package's value
for its repository and folder, and each entry a mapping leaves out at that value's (the readers lay a section over
the package's: registry.package_section); and each value of a package key that is the package's own (`packages_own`)
at the package's value. The running configuration is the one the file makes in the container, whose image has its
checks venv (container_layout.IMAGE_VENV), with the keys of config.json the internal state keeps (the one-time
move's: `state_of`) and config.json not read (runtime_config.from_settings): `running(values, production)`.

- `ran(production)`: PRODUCTION (config.json) as production ran it, in today's names: its older names renamed
  (renames.migrate); each key of AS_LAYOUT1_RAN at the value production ran by layout 1's rule for it
  (rkmon_lib/from_layout1.py), which the translation writes too; and each key of the team dashboard's delivery it did
  not run left out (`delivery_keys`), as production read them only while it ran that delivery.
- `today(production)` and `running(values, production)`: the two configurations; `state_of(production)`, the state
  keys production's (renamed) config.json holds.
- `switches_ran(cfg)`: CFG, today's configuration, with each switch production's code joined, or did not have, at
  what production ran by, by layout 1's rules (rkmon_lib/from_layout1.py): each alert by the alerts' one switch,
  which sent both (alert_switches); the queue notices by the merge queue's switch, and the drift notices, which
  Claude writes, by the merge queue's and review automation's (notices). features.py reads each alert's own switch,
  and each notice's (NOTICES), which production's code did not have.
- `delivery_keys(delivery)`: the configuration keys production read only while it ran the team dashboard's
  DELIVERY (from_layout1.team_block): those of the delivery's block in the file (deploy_tree.feeds), and, of the
  publish delivery, the page and the requests file, which the package derives from the site folder (FROM_THE_SITE).
- `package_for(cfg) -> Package(generic, values, theirs)`: the package's values for CFG's repository and its folder:
  GENERIC, config.defaults.json's; VALUES, the repository defaults' for them (repository_defaults.find) over GENERIC;
  THEIRS, the keys those repository defaults set.
- `packages_own(name, value, package)`: whether VALUE, production's value of the package key NAME or of an entry of
  one, is the PACKAGE's own, which the translation's report counts and does not list: the package's value; a
  blank one of a key whose blank meant the code's own values (BLANK_MEANS_PACKAGE); or, of a key the repository
  defaults set, a blank one or config.defaults.json's, which the configuration reads as theirs (runtime_config).
- `at_package(cfg)`: CFG over the package's values for its repository and folder, each value of a package key that
  is the package's own at the package's value.
- `kind_of(key)`: the kind of a difference in the configuration key KEY, or in an entry of one: VALUE for a key the
  file feeds (deploy_tree.held), INTERNAL for any other. `compare` compares line 1's keys as LINE_1 instead.
- `first_lines(now, run)`: LINE_1's two sides, line 1 by its key: today's built from NOW (today's configuration) by
  FIRST_LINES' rules, RUN's (the running configuration's) its own values; each with reviewer_name.NAME_MARK in place
  of the name it gives (reviewer_name.name_marked).
- `planned(values)`: each key of PLANNED at the value the deploy plan of a file of VALUES gives it, read from the
  variables of docker/.env the plan writes (deploy_plan.env_values).
- `shape_variables(shape, gateway)`: the variables an older Claude secret of SHAPE gives the session today, with the
  configured GATEWAY: each it sets, the key as KEY_MARK, a custom header's value that is not the key as NOT_THE_KEY.
- `named(key) -> tuple`: the parts a difference's KEY is shown by: a variable of the Claude session (SESSION_VALUES)
  under the name of the older Claude secret (secrets_registry.CLAUDE_ENV), which sets it today, so that the reader
  sees the file it came from; any other key split at its dots.
- `first_pass_headers(cfg)`, `model_ids(cfg)` and `efforts(cfg)`: one configuration's first review pass's headers
  (() when no model drafts it), the model id of each pass (by role) and each effort as sent (by role, None for none).
- `Usage(today, after)`: the items of one read's runs and turns (spend_scan.scan): TODAY recorded by family (asked
  None), AFTER the same as the running configuration's passes ask for (spend_scan.as_asked, spend.asked_ids).
- `week_of(items)`: the WEEK_DAYS days that end on the last day ITEMS record, oldest first. `on_days(items, days)`:
  the ITEMS recorded on DAYS. `totals(items, prices, days)`: the dollars of ITEMS on DAYS at PRICES
  (spend.price_table), under USAGE: each day's, each pass's (the session's turns as spend_scan.MAIN) and the week's
  (WEEK). `prices_compared(usage, today_cfg, run)`: PRICE.
- `since_of(root)`: the time the transcripts under ROOT are read from first, READ_DAYS days before the newest main
  session under it was last written (spend_scan.main_sessions); None for none. `holds(days, since)`: whether DAYS,
  the week of the main sessions written since SINCE and their subagents, is the week of all of them: it begins after
  the day after SINCE's.
- `usage_of(root, folder, run) -> Usage`: the transcripts under ROOT (spend.projects_root) read once, the main
  sessions written since `since_of` and their subagents, and all of them when the week those record does not hold
  (`holds`); recorded both ways, TODAY by family and AFTER by the ids the passes of the running configuration RUN ask
  for. The scan's cache is written in FOLDER: this is the one function here that writes files.
- `to_json(differences) -> list` and `from_json(data) -> [Difference]`: differences as JSON, which the one-off
  container prints and the host reads (tools/translate_production.py): each {key, kind, production, after}, a side
  with no value left out. `from_json(to_json(differences))` gives them back, a tuple among their values as a list.
  ValueError for data of another form, its value never shown.

The list of intended differences is data: DIFFERENCES, the package's rows, which the property test of
the running configuration reads too (tests/test_runtime_config.py), and the rows the owner agreed for the package keys
production sets to another value, in a file of the same form outside the package. A list file is a JSON object
whose `differences` is a list of rows. A row's `keys` are what it covers, each a key, which covers the key and each
one under it, or a pattern, in which * stands for any text (fnmatch); its `kind` is one of KINDS; its `why` says why
the difference is intended. Its conditions, each of which must hold:
`owner`, the registry's owner of the difference's key (of the key it is an entry of; NO_OWNER for a key the registry
does not have); `blank`, the SIDES with no value or a blank one (null, "", [] or {}); `default`, the SIDES at the key's
default (`default_of`); `extends`, the side whose value is the other's, which is not blank, and more (a text that
begins with the other's text, a list that begins with the other's items), and `by`, with it, the text that more is,
exactly; and `production` and `after`, the value of that side.

- `rows(path, shown=None) -> [Row]`: the rows of the list file PATH. ValueError, naming the file (as SHOWN, when
  given) and the row, when it breaks the form.
- `listed(difference, rows) -> Row | None`: the row of ROWS that lists DIFFERENCE: of its kind, covering its key, each
  condition holding; of those, the first that names the key itself, else the first that names a key it is under,
  else the first whose pattern matches it. None when no row lists it. `unlisted(differences, rows)`: the DIFFERENCES
  no row lists.
- `owner_of(key)` and `default_of(key)`: the registry's owner of KEY (NO_OWNER for none), and KEY's default (an
  entry's, in its key's default; MISSING for a key the registry does not have).
"""

from __future__ import annotations

import copy
import datetime
import fnmatch
import json
import pathlib
from typing import NamedTuple

from rkmon_lib import (
    bindings,
    claude_env_shape,
    config,
    container_layout,
    custom_headers,
    deploy_plan,
    deploy_tree,
    derived,
    features,
    file_types,
    first_pass,
    from_layout1,
    orchestrator,
    paths,
    registry,
    renames,
    repository_defaults,
    reviewer_name,
    runtime_config,
    secrets_registry,
    settings_file,
    spend,
    spend_scan,
)
from rkmon_lib.registry import MISSING
from rkmon_lib.store import Store

# A custom header's value that is not the key, which the shape does not hold; a model variable the older Claude secret
# sets, whose value the shape does not hold either.
NOT_THE_KEY = "(a value that is not the key)"
SET = "set"
# The file's side of each of the local worker's settings (deploy_tree.WORKER) production sets and the file leaves out,
# while the file binds no drafting model (bindings.no_draft): the verification pass reviews alone.
NO_DRAFTING_MODEL = "(no drafting model)"
# The file keys whose x-internals the inverse map reads: the shared labels and the repository's folder.
SHARED_LABELS = "repository.shared_label_folders"
FOLDER = "repository.folder"
# The file's section of the Claude session's settings, whose keys are compared as the session gets them.
CLAUDE_CODE = "integrations.claude_code"
# The variables of the older Claude secret that give the base address and the custom headers
# (claude_env_shape.CLAUDE_VARS).
BASE_URL = "ANTHROPIC_BASE_URL"
CUSTOM_HEADERS = "ANTHROPIC_CUSTOM_HEADERS"
NONESSENTIAL = "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"
# The session's variables compared as values, each by its name; the key's and the custom headers' are compared as
# headers.
SESSION_VALUES = (BASE_URL, *claude_env_shape.MODEL_VARS, NONESSENTIAL)
# The names of the two sets of headers.
SESSION_HEADERS = "claude_session.headers"
FIRST_PASS_HEADERS = "first_review_pass.headers"
# The first review pass's role (config.EFFORT_ROLES), and the passes beside it whose model `models` names.
FIRST_PASS = "first_pass"
OPTIONAL_PASSES = (*config.FIRST_PASS_ROLES, "tick")
# The names of the price totals: under USAGE, each day's (YYYY-MM-DD), each pass's (spend_scan's role, the session's
# turns MAIN's), and the week's; WEEK_DAYS days end on the last day the records hold.
USAGE = "usage"
WEEK = "week"
WEEK_DAYS = 7
# The days before the newest main session was last written whose sessions are read first (`usage_of`): the week, a day
# for its last record a day before the newest session's last write, and a day for a subagent that starts after its
# session's last write; and a day's seconds.
READ_DAYS = WEEK_DAYS + 2
DAY_SECONDS = 24 * 3600
CENTS_PLACES = 4
# The package keys whose blank value meant the code's own values in layout 1, which config.defaults.json gives them
# now: production's blank one is the package's own value.
BLANK_MEANS_PACKAGE = (
    "clashes",
    "drafter",
    "tracker_allow",
    "followup_scan.defer_patterns",
    "followup_scan.request_patterns",
    "followup_scan.ticket_pattern",
)
# The keys of the publish delivery that the package derives from the site folder: the page (derived.site_page) and
# the requests file (derived.site_requests), which production read only while it published the page.
FROM_THE_SITE = ("publish.html.dest", "publish.html.requests")
# The switches of the queue notices and of the drift notices, in the file: production's code had neither, so it read
# neither (`switches_ran`).
NOTICES = (
    f"{from_layout1.MQ}.queue_notices.enabled",
    f"{from_layout1.MQ}.drift_notices.enabled",
)

# Line 1's derived keys, each with the rule today's code builds it by (rkmon_lib/derived.py): the drafter's line, and
# the verifier's, which a deployment with no first-pass drafter (the verifier reviews alone) posts.
FIRST_LINES = {
    "review_first_line": bindings.drafter_line,
    "review_first_line_no_draft": bindings.no_draft_line,
}

# The kinds of difference, in the order the comparisons give them.
VALUE = "value"
FEATURE = "feature"
HEADER = "header"
MODEL_ID = "model id"
EFFORT = "effort"
LINE_1 = "line 1"
INTERNAL = "internal value"
PRICE = "price total"
KINDS = (VALUE, FEATURE, HEADER, MODEL_ID, EFFORT, LINE_1, INTERNAL, PRICE)

# The list of intended differences: the package's rows, beside this module. The rows the owner agreed for
# the package keys production sets to another value are in a file of the same form outside the package.
DIFFERENCES = pathlib.Path(__file__).resolve().parent / "round_trip_differences.json"
# A row's fields: those it must have, and those it may (its conditions).
ROW_NEEDS = ("keys", "kind", "why")
ROW_MAY = (
    "owner",
    "production",
    "after",
    "blank",
    "default",
    "extends",
    "by",
)
# The two sides of a difference, as a row's conditions name them; the owner of a key the registry does not have.
SIDES = ("production", "after")
NO_OWNER = "none"


class Inverse(NamedTuple):
    """A deploy settings file in today's names: CONFIG, the configuration; SHAPE, the older Claude secret's shape."""

    config: dict
    shape: claude_env_shape.Shape


class Difference(NamedTuple):
    """One difference: KEY, what differs; KIND, one of KINDS; PRODUCTION and AFTER, the two values (MISSING for
    none)."""

    key: str
    kind: str
    production: object = MISSING
    after: object = MISSING


class Usage(NamedTuple):
    """One recorded week's runs and turns: see the module's docstring."""

    today: list
    after: list


class Package(NamedTuple):
    """The package's values for one repository and its folder: see the module's docstring."""

    generic: dict
    values: dict
    theirs: list


class Row(NamedTuple):
    """One row of a list of intended differences: see the module's docstring. VALUES: (side, value) for each side
    whose value is given."""

    keys: tuple
    kind: str
    why: str
    owner: str = ""
    values: tuple = ()
    blank: tuple = ()
    default: tuple = ()
    extends: str = ""
    by: str = ""


def internal(path: str) -> str:
    """The one configuration key the file key PATH feeds: see the module's docstring."""
    (name,) = [i for p, i in deploy_tree.feeds() if p == path]
    return name


def inverse(values: dict) -> Inverse:
    """A deploy settings file's values in today's names: see the module's docstring."""
    cfg = labels_back(deploy_tree.to_config(values))
    return Inverse(cfg, secret_shape(cfg))


def labels_back(cfg: dict) -> dict:
    """CFG with its shared labels in today's form: see the module's docstring."""
    name = internal(SHARED_LABELS)
    shared = deploy_tree.get(cfg, name)
    folder = paths.repository_folder(cfg)
    if not isinstance(shared, dict) or any(
        str(f).strip("/") != folder for f in shared.values()
    ):
        return cfg
    out = copy.deepcopy(cfg)
    deploy_tree.put(out, name, list(shared))
    return out


def _header(name: str, value: str, mark: str) -> claude_env_shape.Header:
    """One custom header as the shape has it: whether its value is the key (MARK, KEY_MARK), alone or after one
    word."""
    word, _, rest = value.rpartition(" ")
    if value == mark:
        return claude_env_shape.Header(name, True, "")
    return (
        claude_env_shape.Header(name, True, word)
        if rest == mark
        else claude_env_shape.Header(name, False, "")
    )


def secret_shape(cfg: dict) -> claude_env_shape.Shape:
    """The shape of the older Claude secret the configuration's Claude settings stand for: see the module's
    docstring."""
    env = orchestrator.session_variables(cfg, custom_headers.KEY_MARK)
    return claude_env_shape.Shape(
        variables=tuple(v for v in claude_env_shape.CLAUDE_VARS if v in env),
        headers=tuple(
            _header(n, v, custom_headers.KEY_MARK)
            for n, v in custom_headers.header_lines(env.get(CUSTOM_HEADERS, ""))
        ),
        base_url=env.get(BASE_URL, ""),
    )


# --- the comparisons --------------------------------------------------------------------------------------------------


def _plain(value):
    """VALUE as plain mappings and lists (the running configuration's are registry.Strict)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _missing(value) -> bool:
    return value is MISSING or value is deploy_tree.MISSING


def _same(a, b) -> bool:
    """Whether A and B are one value, as JSON writes it (true is not 1); MISSING only as MISSING."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _leaves(value, at: tuple = ()) -> dict:
    """Each leaf of VALUE by its path, a tuple of names: a mapping's entries, an empty mapping a leaf of its own."""
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(_leaves(v, (*at, str(k))))
        return out
    return {at: value}


def _emptied(path: tuple, value, other: dict) -> bool:
    """Whether VALUE, an empty mapping at PATH, stands where the leaves OTHER have entries under PATH: those entries
    are the differences."""
    return value == {} and any(
        len(q) > len(path) and q[: len(path)] == path for q in other
    )


def _differ(kind: str, a: dict, b: dict) -> list:
    """The leaves where the leaves A and B differ, each a Difference of KIND; a mapping emptied on one side is told by
    its entries on the other."""
    out = []
    for p in sorted(set(a) | set(b)):
        x, y = a.get(p, MISSING), b.get(p, MISSING)
        if not (_same(x, y) or _emptied(p, x, b) or _emptied(p, y, a)):
            out.append(Difference(".".join(p), kind, x, y))
    return out


def _parts(names) -> list:
    """Each dotted name of NAMES as a tuple of its names."""
    return [tuple(n.split(".")) for n in names]


def _under(path: tuple, names) -> bool:
    """Whether the leaf PATH is one of NAMES (each a tuple of names: `_parts`) or inside one."""
    return any(path[: len(n)] == n for n in names)


def state_of(production: dict) -> dict:
    """The keys of PRODUCTION (config.json, renamed) the internal state keeps: each state key it holds,
    nested."""
    out: dict = {}
    for k in registry.keys():
        v = registry.get(production, k.name)
        if k.owner == registry.STATE and v is not MISSING:
            registry.put(out, k.name, copy.deepcopy(v))
    return out


def _layout1(cfg: dict, name: str):
    """CFG's value of the configuration key NAME as from_layout1's rules take it: from_layout1.MISSING for none."""
    value = registry.get(cfg, name)
    return from_layout1.MISSING if value is MISSING else value


def _container(cfg: dict):
    """The container's name production ran: a blank or missing one is from_layout1.CONTAINER_NAME."""
    return from_layout1.container_name(_layout1(cfg, "deployment.project"))


def _image(cfg: dict) -> str:
    """The image production ran, <name>:<tag> (deploy_tree.image_parts): a blank name or tag is what it meant
    (from_layout1.image_name and image_tag, the container's name for the tag)."""
    value = registry.get(cfg, "deployment.image")
    name, tag = deploy_tree.image_parts("" if value is MISSING else value)
    return f"{from_layout1.image_name(name)}:{from_layout1.image_tag(tag, _container(cfg))}"


def _labels(cfg: dict):
    """The labels production's develop guard put on a develop issue on a red CI (from_layout1.ci_red_labels): the
    first of incident_issue.labels, or the develop guard's own, then the others and the label the develop guard put
    beside the first. A value that is no list of labels is as it is (the translation stops on it).
    """
    value = registry.get(cfg, "incident_issue.labels")
    names = from_layout1.labels(_layout1(cfg, "incident_issue.labels"))
    if names is None:
        return value
    first, rest = from_layout1.ci_red_labels(names)
    return [first, *rest]


def _stability(cfg: dict):
    """The label production's develop guard put on any other develop issue: a blank or missing one is the develop
    guard's own (from_layout1.stability_label); any other as it is."""
    value = registry.get(cfg, "incident_issue.stability_label")
    if _blank(value) or (isinstance(value, str) and not value.strip()):
        return from_layout1.stability_label(
            _layout1(cfg, "incident_issue.stability_label")
        )
    return value


def _repository(cfg: dict):
    """Where production's `rk update` fetched from: a GitHub repository (from_layout1.github_repository) by the
    address the file's <org>/<repo> stands for (derived.github_address), whatever form update.repo gives it in; any
    other value as it is."""
    named = from_layout1.github_repository(_layout1(cfg, "update.repo"))
    return derived.github_address(named) if named else registry.get(cfg, "update.repo")


def _delivery(cfg: dict) -> str:
    """The team dashboard's delivery production ran: a blank or missing one is from_layout1.DELIVERY."""
    return from_layout1.team_delivery(_layout1(cfg, "team_dashboard.mode"))


def _requests(cfg: dict):
    """The requests file production's published page took other teams' priority requests from: none, "", for a
    value that names none (from_layout1.priority_requests: false, 0, null or blank); any other as it is.
    """
    value = _layout1(cfg, "publish.html.requests")
    return (
        registry.get(cfg, "publish.html.requests")
        if from_layout1.priority_requests(value)
        else ""
    )


def _worker_secret(cfg: dict):
    """The secret production's drafting model, the local worker, read its key from, by the name the one-time move
    copies it to (from_layout1.api_key_secret_name): the one worker.api_key_secret names, read as config.worker reads
    it, else the local worker's own key, which the move copies to the first review pass's
    (secrets_registry.FIRST_PASS_KEY, by OLDER_KEY_NAMES). As it is while no model drafts, and for a worker on Ollama,
    which takes no key."""
    w = config.worker_settings(cfg)
    if (
        bindings.no_draft(cfg)
        or w.get("backend") == deploy_tree.WORKER_BACKEND["ollama"]
    ):
        return registry.get(cfg, "worker.api_key_secret")
    named = str(w.get("api_key_secret") or "").strip()
    return (
        from_layout1.api_key_secret_name(named)
        if named
        else secrets_registry.FIRST_PASS_KEY
    )


# The keys the file feeds whose value production ran by a rule of layout 1's (rkmon_lib/from_layout1.py), which the
# translation writes too, each with the function of production's configuration (renamed) that gives that value: the
# container's name; the image; the develop guard's labels, on a red CI and on any other develop issue; where
# `rk update` fetched from; the team dashboard's delivery, and the requests file of the page it published; the secret
# of the drafting model's key.
AS_LAYOUT1_RAN = {
    "deployment.project": _container,
    "deployment.image": _image,
    "incident_issue.labels": _labels,
    "incident_issue.stability_label": _stability,
    "update.repo": _repository,
    "team_dashboard.mode": _delivery,
    "publish.html.requests": _requests,
    "worker.api_key_secret": _worker_secret,
}
# The keys of production's config.json whose value the deploy plan makes now, not the configuration
# (rkmon_lib/deploy_plan.py), each with the variable of docker/.env that holds it: the host address the admin
# dashboard is published on.
PLANNED = {"admin_dashboard.publish_host": "RKMON_ADMIN_HOST"}


def planned(values: dict) -> dict:
    """Each key of PLANNED at the value the deploy plan of the file's VALUES gives it: see the module's docstring."""
    doc = settings_file.Document(file_types.DEPLOY, file_types.DEPLOY.version, values)
    # the host's ids and the confirm's time, which the plan reads besides the settings, give none of these values
    variables = deploy_plan.env_values(
        deploy_plan.settings(doc), deploy_plan.Host(0, 0, "")
    )
    return {name: variables[variable] for name, variable in PLANNED.items()}


def delivery_keys(delivery: str) -> tuple:
    """The configuration keys production read only while it ran the team dashboard's DELIVERY: see the module's
    docstring."""
    block = f"{from_layout1.TEAM}.{delivery}."
    own = tuple(i for p, i in deploy_tree.feeds() if p.startswith(block))
    return own + (FROM_THE_SITE if delivery == from_layout1.PUBLISH_DELIVERY else ())


def _drop(cfg: dict, name: str) -> None:
    """CFG without the configuration key NAME, in place."""
    *path, last = name.split(".")
    holder = registry.get(cfg, ".".join(path)) if path else cfg
    if isinstance(holder, dict):
        holder.pop(last, None)


def ran(production: dict) -> dict:
    """PRODUCTION (config.json) as production ran it, in today's names: see the module's docstring."""
    renamed = renames.migrate(production)[0]
    out = copy.deepcopy(renamed)
    for name, rule in AS_LAYOUT1_RAN.items():
        value = rule(renamed)
        if not _missing(value):
            registry.put(out, name, copy.deepcopy(value))
    taken = registry.get(out, "team_dashboard.mode")
    for delivery in (from_layout1.PUBLISH_DELIVERY, from_layout1.SERVE_DELIVERY):
        if delivery != taken:
            for name in delivery_keys(delivery):
                _drop(out, name)
    return out


def running(values: dict, production: dict) -> dict:
    """The running configuration the file's VALUES make: see the module's docstring."""
    renamed = renames.migrate(production)[0]
    return _plain(
        runtime_config.from_settings(
            values, state=state_of(renamed), image_venv=container_layout.IMAGE_VENV
        ).values
    )


def today(production: dict) -> dict:
    """PRODUCTION (config.json) as production ran it, as today's code reads it: see the module's docstring."""
    return config.normalize(ran(production))


def switches_ran(cfg: dict) -> dict:
    """CFG, today's configuration, with the switches production's code joined, or did not have, at what it ran by:
    see the module's docstring."""
    out = copy.deepcopy(cfg)
    # each alert's own switch, which renames.ALERT_KINDS names by the key alert_switches gives it
    alerts = from_layout1.alert_switches(features.enabled(cfg, "alerts"))
    for name in renames.ALERT_KINDS:
        registry.put(out, name, alerts[name.rpartition(".")[2]])
    posted = from_layout1.notices(
        features.enabled(cfg, "merge_queue"), features.enabled(cfg, "review_automation")
    )
    for path, on in zip(NOTICES, posted):
        registry.put(out, internal(path), on)
    return out


def package_for(cfg: dict) -> Package:
    """The package's values for CFG's repository and its folder: see the module's docstring."""
    generic = config.load_defaults()
    values = copy.deepcopy(generic)
    found = repository_defaults.find(cfg.get("repo"), paths.repository_folder(cfg))
    theirs = repository_defaults.keys_of(found.values) if found else []
    for name in theirs:
        deploy_tree.put(values, name, copy.deepcopy(registry.get(found.values, name)))
    return Package(generic, values, theirs)


def packages_own(name: str, value, package: Package) -> bool:
    """Whether VALUE, production's of the package key NAME or of an entry of one, is the PACKAGE's own: see the
    module's docstring."""
    at = tuple(name.split("."))
    return (
        value == deploy_tree.get(package.values, name)
        or (_under(at, _parts(BLANK_MEANS_PACKAGE)) and value in ("", [], {}))
        or (
            _under(at, _parts(package.theirs))
            and not repository_defaults.own(
                value, deploy_tree.get(package.generic, name)
            )
        )
    )


def _lay(base: dict, over: dict) -> None:
    """OVER laid over BASE, in place: a mapping's entries over the same mapping's."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _lay(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def at_package(cfg: dict) -> dict:
    """CFG over the package's values for its repository and folder, its own values at the package's: see the
    module's docstring."""
    package = package_for(cfg)
    out = copy.deepcopy(package.values)
    _lay(out, cfg)
    for name in (*BLANK_MEANS_PACKAGE, *package.theirs):
        at = tuple(name.split("."))
        for path, value in _leaves(deploy_tree.get(cfg, name), at).items():
            key = ".".join(path)
            own = deploy_tree.get(package.values, key)
            if not (_missing(value) or _missing(own)) and packages_own(
                key, value, package
            ):
                deploy_tree.put(out, key, copy.deepcopy(own))
    return out


def kind_of(key: str) -> str:
    """The kind of a difference in the configuration key KEY, or in an entry of one: see the module's docstring."""
    return (
        VALUE if _under(tuple(key.split(".")), _parts(deploy_tree.held())) else INTERNAL
    )


def named(key: str) -> tuple:
    """The parts a difference's KEY is shown by: see the module's docstring."""
    return (
        (secrets_registry.CLAUDE_ENV, key)
        if key in SESSION_VALUES
        else tuple(key.split("."))
    )


def _claude_keys() -> tuple:
    """The configuration keys integrations.claude_code's keys feed."""
    return tuple(i for p, i in deploy_tree.feeds() if p.startswith(CLAUDE_CODE + "."))


def _key_value(cfg: dict, name: str):
    """The configuration key NAME's value in CFG, else its default (registry.value: MISSING when it has none)."""
    v = registry.get(cfg, name)
    if v is not MISSING:
        return v
    default = registry.default(name) if registry.kind(name) == registry.KEY else MISSING
    return default


def values_compared(production: dict, back: dict) -> list:
    """VALUE: the keys the file feeds, PRODUCTION's as production ran them (`ran`) against BACK's, the inverse map's
    configuration."""
    cfg = ran(production)
    skip = _claude_keys()
    unbound = bindings.no_draft(back)
    worker = _parts((deploy_tree.WORKER,))
    out = []
    for name in sorted(deploy_tree.held()):
        if name in skip:
            continue
        at = tuple(name.split("."))
        today_s = _leaves(_key_value(cfg, name), at)
        after = _leaves(_key_value(back, name), at)
        if unbound and _under(at, worker):
            after.update(
                (p, NO_DRAFTING_MODEL)
                for p, v in today_s.items()
                if not _missing(v) and _missing(after.get(p, MISSING))
            )
        out += _differ(VALUE, today_s, after)
    return out


def shape_variables(shape: claude_env_shape.Shape, gateway: str = "") -> dict:
    """The variables an older Claude secret of SHAPE gives the session today: see the module's docstring."""
    mark = custom_headers.KEY_MARK
    out = {}
    for name in shape.variables:
        if name in claude_env_shape.KEY_VARS:
            out[name] = mark
        elif name in claude_env_shape.MODEL_VARS:
            out[name] = SET
        elif name == BASE_URL and shape.base_url:
            out[name] = shape.base_url
        elif name == NONESSENTIAL and shape.nonessential_traffic:
            out[name] = shape.nonessential_traffic
    lines = [
        f"{h.name}: {f'{h.word} ' if h.word else ''}{mark if h.value_is_key else NOT_THE_KEY}"
        for h in shape.headers
    ]
    if lines:
        out[CUSTOM_HEADERS] = "\n".join(lines)
    if gateway:
        out[BASE_URL] = gateway
    return out


def _lines(headers: dict) -> list:
    """HEADERS ({name: value}) as a difference shows them: each "Name: value", in the order sent."""
    return [f"{n}: {v}" for n, v in headers.items()]


def _session(variables: dict) -> dict:
    """The session's variables as it runs with VARIABLES: the container's nonessential traffic switch
    (orchestrator.SESSION_ENV) unless they set one."""
    return {NONESSENTIAL: orchestrator.SESSION_ENV[NONESSENTIAL], **variables}


def _first_review_pass(cfg: dict) -> dict | None:
    """The settings of the model that drafts the first review pass: the local worker's (config.worker_settings), a
    binding's (bindings.binding_by_name), each with `by`; None when none drafts it (the verifier reviews alone).
    """
    choice = config.first_pass_choice(cfg)
    if choice in bindings.NO_DRAFT_NAMES:
        return None
    if choice == bindings.WORKER:
        return dict(config.worker_settings(cfg), by="worker")
    b = bindings.binding_by_name(cfg, choice)
    return None if b is None else dict(b, by="binding")


def first_pass_headers(cfg: dict) -> tuple:
    """The headers of the first review pass's requests: see the module's docstring; () when no model drafts it."""
    m = _first_review_pass(cfg)
    if m is None:
        return ()
    llm = first_pass.llm_client()
    return tuple(_lines(llm.auth_headers(m, custom_headers.KEY_MARK)))


def model_ids(cfg: dict) -> dict:
    """MODEL_ID: each pass's and the session's model id: see the module's docstring."""
    out = {
        role: config.claude_model_id(cfg, role)
        for role in dict.fromkeys(spend.ASKED.values())
    }
    m = _first_review_pass(cfg)
    out[FIRST_PASS] = m.get("model") if m else None
    for role in OPTIONAL_PASSES:
        m = config.model_for(cfg, role)
        out[role] = m.get("model") if m.get("by") != "none" else None
    return out


def efforts(cfg: dict) -> dict:
    """EFFORT: each effort as the requests carry it: see the module's docstring."""
    out = {}
    for role in config.EFFORT_ROLES:
        if role == FIRST_PASS:
            m = _first_review_pass(cfg)
            b = None if m is None or m["by"] == "worker" else m
            e = None if m is None else bindings.drafter_effort(cfg, b) or None
        elif role in OPTIONAL_PASSES or role in config.OWNER_ROLE:
            m = config.model_for(cfg, role)
            e = None if m.get("by") == "none" else first_pass.effort(cfg, role, m)
        else:
            e = config.role_effort(cfg, role)[0] or None
        out[role] = e
    return out


def first_lines(now: dict, run: dict) -> tuple:
    """LINE_1's two sides, line 1 by its key, each apart from the name it gives: see the module's docstring."""
    return (
        {k: reviewer_name.name_marked(rule(now)) for k, rule in FIRST_LINES.items()},
        {k: reviewer_name.name_marked(run.get(k)) for k in FIRST_LINES},
    )


def _by(kind: str, a: dict, b: dict) -> list:
    """The names where the flat mappings A and B differ, each a Difference of KIND."""
    return [
        Difference(k, kind, a.get(k, MISSING), b.get(k, MISSING))
        for k in sorted(set(a) | set(b))
        if not _same(a.get(k, MISSING), b.get(k, MISSING))
    ]


def _day(item: dict) -> str:
    return str(item.get("start") or item.get("t") or "")[:10]


def week_of(items: list) -> tuple:
    """The WEEK_DAYS days that end on the last day ITEMS record, oldest first; () when they record none."""
    days = sorted({d for d in map(_day, items) if d})
    if not days:
        return ()
    last = datetime.date.fromisoformat(days[-1])
    return tuple(
        str(last - datetime.timedelta(days=n)) for n in range(WEEK_DAYS - 1, -1, -1)
    )


def on_days(items: list, days: tuple) -> list:
    """The ITEMS recorded on DAYS, in their order."""
    return [x for x in items if _day(x) in days]


def totals(items: list, prices: dict, days: tuple) -> dict:
    """The dollars of ITEMS on DAYS at PRICES: see the module's docstring."""
    out: dict = {}
    for x in on_days(items, days):
        usd = spend.usd(prices, x.get("use") or {})
        for name in (_day(x), x.get("role") or spend_scan.MAIN, WEEK):
            key = f"{USAGE}.{name}"
            out[key] = out.get(key, 0.0) + usd
    return {k: round(v, CENTS_PLACES) for k, v in out.items()}


def prices_compared(usage: Usage, today_cfg: dict, run: dict) -> list:
    """PRICE: one recorded week priced both ways: see the module's docstring."""
    days = week_of(usage.today)
    return _by(
        PRICE,
        totals(usage.today, spend.price_table(today_cfg), days),
        totals(usage.after, spend.price_table(run), days),
    )


def compare(
    production: dict,
    shape: claude_env_shape.Shape,
    values: dict,
    usage: Usage | None = None,
) -> list:
    """Each difference between what production runs and what the file runs: see the module's docstring."""
    back = inverse(values).config
    now, run = today(production), running(values, production)
    # each configuration with its Claude session's variables: today's from the older Claude secret's shape, the running
    # one's from its settings
    sessions = (
        (now, shape_variables(shape, orchestrator.gateway(now))),
        (run, orchestrator.session_variables(run, custom_headers.KEY_MARK)),
    )
    out = values_compared(production, back)
    out += _by(
        VALUE,
        *(
            {k: v for k, v in _session(variables).items() if k in SESSION_VALUES}
            for _cfg, variables in sessions
        ),
    )
    switched = switches_ran(now)
    out += _by(
        FEATURE,
        {name: features.value(switched, name) for name in features.CAPABILITIES},
        {name: features.value(run, name) for name in features.CAPABILITIES},
    )
    out += _by(
        HEADER,
        *(
            {
                SESSION_HEADERS: _lines(orchestrator.request_headers(variables)),
                FIRST_PASS_HEADERS: list(first_pass_headers(cfg)),
            }
            for cfg, variables in sessions
        ),
    )
    out += _by(MODEL_ID, model_ids(now), model_ids(run))
    out += _by(EFFORT, efforts(now), efforts(run))
    out += _by(LINE_1, *first_lines(now, run))
    lines = _parts(FIRST_LINES)
    internal = {
        p: v
        for p, v in _leaves(run).items()
        if kind_of(".".join(p)) == INTERNAL and not _under(p, lines)
    }
    internal.update({tuple(k.split(".")): v for k, v in planned(values).items()})
    out += _differ(
        INTERNAL,
        {
            p: v
            for p, v in _leaves(at_package(now)).items()
            if kind_of(".".join(p)) == INTERNAL and not _under(p, lines)
        },
        internal,
    )
    if usage is not None:
        out += prices_compared(usage, now, run)
    return out


def since_of(root) -> float | None:
    """The time the transcripts under ROOT are read from first: see the module's docstring."""
    times = []
    for path in spend_scan.main_sessions(root):
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times) - READ_DAYS * DAY_SECONDS if times else None


def holds(days: tuple, since: float) -> bool:
    """Whether DAYS, the week (`week_of`) of the main sessions written since SINCE and their subagents, is the week of
    all of them: it begins after the day after SINCE's, in UTC as the records' days are. A main session last written
    before SINCE holds no record after it, and its subagents none a day after it, as each starts when the session
    starts it."""
    after = datetime.datetime.fromtimestamp(since, datetime.timezone.utc).date()
    return bool(days) and days[0] > str(after + datetime.timedelta(days=1))


def _items(scan: dict) -> list:
    return scan["runs"] + scan["turns"]


def usage_of(root, folder, run: dict) -> Usage:
    """The transcripts under ROOT read once and recorded both ways, the scan's cache in FOLDER: see the module's
    docstring."""
    store, since = Store(pathlib.Path(folder)), since_of(root)
    scan = spend_scan.scan(root, store, since=since)
    if since is not None and not holds(week_of(_items(scan)), since):
        # all of them: the cache holds those read already, so only the rest is read
        scan = spend_scan.scan(root, store)
    after = spend_scan.as_asked(scan, spend.asked_ids(run))
    return Usage(_items(scan), _items(after))


def to_json(differences) -> list:
    """DIFFERENCES as JSON: see the module's docstring."""
    return [
        {
            "key": d.key,
            "kind": d.kind,
            **{s: v for s, v in zip(SIDES, (d.production, d.after)) if not _missing(v)},
        }
        for d in differences
    ]


def from_json(data) -> list:
    """The differences DATA holds in `to_json`'s form: see the module's docstring."""
    if not isinstance(data, list):
        raise ValueError("the differences are no list")
    out = []
    for item in data:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("key"), str)
            or item.get("kind") not in KINDS
            or set(item) - {"key", "kind", *SIDES}
        ):
            raise ValueError(
                f"a difference is a JSON object of key, kind, {' and '.join(SIDES)}"
            )
        out.append(
            Difference(
                item["key"],
                item["kind"],
                item.get("production", MISSING),
                item.get("after", MISSING),
            )
        )
    return out


# --- the list of intended differences ---------------------------------------------------------------------------------


def _row(raw, where: str) -> Row:
    """The row RAW of a list file, WHERE naming it in a problem: ValueError when it breaks the form."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a row is a JSON object")
    unknown = sorted(set(raw) - {*ROW_NEEDS, *ROW_MAY})
    if unknown:
        raise ValueError(
            f"{where}: no field {', '.join(unknown)}: a row's fields are {', '.join((*ROW_NEEDS, *ROW_MAY))}"
        )
    missing = [f for f in ROW_NEEDS if not raw.get(f)]
    if missing:
        raise ValueError(f"{where}: no {', '.join(missing)}")
    keys = raw["keys"]
    if not isinstance(keys, list) or not all(isinstance(k, str) and k for k in keys):
        raise ValueError(f"{where}: keys is a list of keys and patterns")
    if raw["kind"] not in KINDS:
        raise ValueError(f"{where}: kind {raw['kind']!r} is none of {', '.join(KINDS)}")
    if not isinstance(raw["why"], str):
        raise ValueError(f"{where}: why is text")
    owner = raw.get("owner", "")
    if owner and owner not in (*registry.OWNERS, NO_OWNER):
        raise ValueError(
            f"{where}: owner {owner!r} is none of {', '.join((*registry.OWNERS, NO_OWNER))}"
        )
    for field in ("blank", "default"):
        sides = raw.get(field, [])
        if not isinstance(sides, list) or not set(sides) <= set(SIDES):
            raise ValueError(f"{where}: {field} is a list of sides: {', '.join(SIDES)}")
    extends = raw.get("extends", "")
    if extends and extends not in SIDES:
        raise ValueError(f"{where}: extends is one side: {' or '.join(SIDES)}")
    by = raw.get("by", "")
    if not isinstance(by, str):
        raise ValueError(f"{where}: by is text")
    if by and not extends:
        raise ValueError(f"{where}: by goes with extends, the side it extends")
    return Row(
        keys=tuple(keys),
        kind=raw["kind"],
        why=raw["why"],
        owner=owner,
        values=tuple((side, raw[side]) for side in SIDES if side in raw),
        blank=tuple(raw.get("blank", [])),
        default=tuple(raw.get("default", [])),
        extends=extends,
        by=by,
    )


def rows(path, shown=None) -> list:
    """The rows of the list file PATH, SHOWN naming it in a problem (PATH when it is None): see the module's
    docstring."""
    path = pathlib.Path(path)
    name = path if shown is None else shown
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{name}: not valid JSON: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("differences"), list):
        raise ValueError(
            f"{name}: no list of differences: a JSON object whose differences is a list of rows"
        )
    return [_row(r, f"{name}, row {n}") for n, r in enumerate(data["differences"], 1)]


def _registry_key(key: str) -> str:
    """The registry's key KEY names, or the key it is an entry of."""
    name = key
    while registry.kind(name) == registry.ENTRY:
        name = name.rsplit(".", 1)[0]
    return name


def owner_of(key: str) -> str:
    """The registry's owner of KEY (of the key it is an entry of); NO_OWNER for a key the registry does not have."""
    name = _registry_key(key)
    return registry.key(name).owner if registry.kind(name) == registry.KEY else NO_OWNER


def default_of(key: str):
    """KEY's default (an entry's, in its key's default); MISSING for a key the registry does not have."""
    name = _registry_key(key)
    if registry.kind(name) != registry.KEY:
        return MISSING
    value = registry.default(name)
    for part in key[len(name) :].split(".")[1:]:
        value = value.get(part, MISSING) if isinstance(value, dict) else MISSING
    return value


def _blank(value) -> bool:
    return _missing(value) or value in ("", None) or value == [] or value == {}


def _extends(longer, shorter, by: str = "") -> bool:
    """Whether LONGER is SHORTER, which is not blank, and more: a text that begins with SHORTER's text, a list that
    begins with SHORTER's items; a text that is SHORTER's text then BY, when BY is given.
    """
    if _blank(shorter):
        return False
    if isinstance(longer, str) and isinstance(shorter, str):
        return longer == shorter + by if by else longer.startswith(shorter)
    if isinstance(longer, list) and isinstance(shorter, list) and not by:
        return _same(longer[: len(shorter)], shorter)
    return False


def _holds(row: Row, difference: Difference) -> bool:
    """Whether each condition of ROW holds for DIFFERENCE."""
    sides = dict(zip(SIDES, (difference.production, difference.after)))
    if row.owner and owner_of(difference.key) != row.owner:
        return False
    if not all(_blank(sides[s]) for s in row.blank):
        return False
    if row.default and not all(
        _same(sides[s], default_of(difference.key)) for s in row.default
    ):
        return False
    if row.extends:
        (other,) = [s for s in SIDES if s != row.extends]
        if not _extends(sides[row.extends], sides[other], row.by):
            return False
    return all(_same(sides[s], value) for s, value in row.values)


def _closeness(row: Row, key: str) -> int | None:
    """How closely ROW covers KEY: 0 when it names KEY, 1 when it names a key KEY is under, 2 when one of its patterns
    matches KEY; None when it does not cover KEY."""
    found = []
    for name in row.keys:
        if "*" in name:
            if fnmatch.fnmatchcase(key, name):
                found.append(2)
        elif key == name:
            found.append(0)
        elif key.startswith(name + "."):
            found.append(1)
    return min(found, default=None)


def listed(difference: Difference, rows_given) -> Row | None:
    """The row of ROWS_GIVEN that lists DIFFERENCE: see the module's docstring."""
    best = None
    for row in rows_given:
        closeness = _closeness(row, difference.key)
        if (
            row.kind != difference.kind
            or closeness is None
            or not _holds(row, difference)
        ):
            continue
        if best is None or closeness < best[0]:
            best = (closeness, row)
    return best[1] if best else None


def unlisted(differences, rows_given) -> list:
    """The DIFFERENCES no row of ROWS_GIVEN lists."""
    return [d for d in differences if listed(d, rows_given) is None]
