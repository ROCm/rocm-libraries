# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The deploy path's confirm step (`rkmon confirm`; `rkmon confirm --dry-run` prints the same and asks nothing): every
deploy setting the container will run with, and the checks of the configuration they make, before anything starts; after
the first start, only what changes, each change with how it applies, and the checks the changes call for.

The configuration is the one the container's load makes of the settings the confirm delivers
(runtime_config.from_settings over deploy_plan.plan's layer): the package's defaults, the repository defaults for
its repository and folder, the settings, and what is derived from them in the container's layout.

- `headline(values)`: the container, then each feature's state (features.headline_parts), joined by " · " and
  wrapped at checks.WIDTH before a "·" (wrapping.marked).
- `sections(made, values, labels=None, rows=None, updates=None)`: every setting:
  each key of the deploy settings file under its own section, in the file's order, at the value the confirm
  delivers: MADE, what the settings make on this host (deploy_plan.plan), holds in its layer every key of the file but
  the header's (kind, schema_version) and the updates (deploy_plan.LEFT_OUT), each at its effective value, a map one
  value (settings_file.map_keys). The sections are the file's parts in the tree's order and with its titles
  (schema/deploy_settings.schema.json: Container, Image, Updates, Repository, Host, Monitor, then each feature: Review
  automation, Merge queue, Dashboards, Alerts, Ticket creation, Weekly report; then Integrations and Model bindings),
  each a heading and aligned `name: value` lines, each key named from its section, its value as `rk config show` shows
  it (config.safe_value), a text as it is and any other value as JSON; a line that would pass checks.WIDTH goes on under
  its value. Some values have plain words after them (`_worded`): the image on this host or not, the RAM disk, the
  parallel builds and reviews at the start and the scaler's range, each mode, why a feature that is off does nothing,
  where docker publishes the dashboards on this host (MADE's variables, docker/.env's: deploy_plan.env_values) and the
  weekly report; a blank registry or Claude address says what is used instead. Updates shows UPDATES, the file's own
  section, which only the host reads, or one line while the file has none (None). Review automation, while it is on,
  ends with its passes resolved (roles.pass_lines: pass | model | provider | effort, an optional pass on `default`
  showing its owner's model and effort). Then the Logins. VALUES is the configuration the container runs with those
  settings (runtime_config.from_settings over MADE's layer); LABELS, the image's labels (None while it is not on this
  host); ROWS, the logins as `rk logins --json` read them in the container (None while none can be read). Private values
  show: this is your own terminal. Nothing secret: the file's read refuses a value that looks like one
  (settings_file.secret_texts), an address shows without its user, password, query and fragment, and a login or a secret
  shows as set, missing or NOT NEEDED, never its value.
- `render(made, values, labels=None, rows=None, updates=None)`: the headline, then the sections.
- `exposure(variables)`: the exposure sentence while this host serves the team dashboard, by docker/.env's
  VARIABLES the plan makes: anyone who reaches its address and port can read it and send a priority request, which
  its server takes whenever it serves the page (publish.takes_requests); "" while it is not served.
- `exposure_changed(running, variables)`: after the first start, the exposure sentence of VARIABLES when a change
  serves the team dashboard or moves the address or port it is served on, by RUNNING, docker/.env's variables as they
  are on this host, which serve it elsewhere or not at all; "" otherwise.
- `checks(resolved, doc, rows, image=(), models=(), logins=(), claude=())`: the confirm's check rows
  (rkmon_lib/checks.py: checks.render prints them): the features' coherence (coherence.deploy's PASS row, when it
  passes), then each FAIL and WARN row of the file and its settings as the confirm read them, with check 18's PASS
  row (coherence.skills: the team's skills the confirm delivers), then what the configuration RESOLVED
  (runtime_config.Resolved) refuses to run (its problems, bindings.problems), the image's rows IMAGE (`image_rows`),
  the models' rows MODELS (`model_rows`), the logins' LOGINS (`login_skips`) and check 16's CLAUDE
  (`claude_env_rows`). A FAIL refuses the confirm.
- `image_rows(ref, code, first=True, pull=False)`: the image row: whether the image REF, the one
  image.tag names, was built from this package's image recipe: its own copies of the files an image keeps whatever
  release runs (code_volume.IMAGE_BOUND) are this package's, the one rule of whether an image suits this package
  (rkmon image check and start apply it too). CODE is None while REF is not on this host: a SKIP, which names step 4
  before the first start (FIRST), and else the command that brings the image and checks it (image pull when the
  settings name a registry, PULL; else image load). Else CODE is (the files whose copies differ, `changed_copies`, or
  None when the image's copies could not be read; why not): a PASS when none differ, else a FAIL whose fix is an
  image built from this package, releases.new_image_fix.
- CODE_PROBE: the program a one-off container of the image runs, with no network, to print the fingerprints of its own
  copies (docker/rkmon: `recipe_changes`): code_volume.fingerprints' own code, so that the image's fingerprints and
  this package's are made alike (each file's SHA-256 with its execute bit). Its arguments are the folder of the image's
  own code, then code_volume.IMAGE_BOUND as it is when the probe runs; its answer, one JSON object: each file's path in
  the package -> its fingerprint. `changed_copies(answer, package)`: the files of code_volume.IMAGE_BOUND whose
  fingerprints in ANSWER are not those of the package in the folder PACKAGE, or that one of the two lacks, sorted;
  ValueError when the answer cannot be read, names a file that is none of code_volume.IMAGE_BOUND's, or has no
  fingerprint for one.
- `model_rows(values, answers, first=True, names=None)`: the model rows: each model a pass uses
  (`models_in_use`), or each of NAMES among them, called as the container will reach it, from a one-off container of
  the image on this deployment's network. ANSWERS maps each model's address to "" when it answered there (any HTTP
  answer: a 401 or a 404 is one), else why not: a PASS, or a WARN that names the address. No key is sent: a claude
  model's PASS says the gateway answers at its address, and that the claude login's command and rkmon doctor check
  the key, which they send as the session does (a 401 then fails, naming the login: rkmon_lib/reach.py claude_test).
  ANSWERS is None while no such call can run (the image is not on this host, or this deployment has no network yet):
  a SKIP that names where the container calls it, step 7 before the first start (FIRST), else rkmon doctor; so is a
  model the settings give no address for (a claude model with no gateway, which the session reaches at Anthropic's
  own API, or an address a variable names, read where the monitor runs).
- MODEL_PROBE: the program that one-off container runs to call each address (docker/rkmon: `model_answers`): its
  arguments are the seconds each call may take (REACH_TIMEOUT_S), then the addresses; its answer, one JSON object:
  each address -> "" or why not. `model_answers(code, printed, urls)`: each address of URLS -> its answer, as the
  one-off container with docker's exit code CODE printed it (PRINTED); a call that did not run, or an answer that
  cannot be read, is each address's why.
- HELD_PROBE: the program a one-off container of the image runs before the first start, with no network and the
  secrets volumes that exist mounted read-only, to tell which logins and secrets they hold (docker/rkmon:
  `logins_unset`): secrets_registry.held's own code, so that the volumes are read as `rk logins` reads them. Its
  argument, one JSON list: `login_places`'; its answer, one JSON list: the names none of whose places holds anything,
  each once. `unheld(answer, places)`: those names, in PLACES' order, each once; ValueError when the answer cannot be
  read or names one that is none of PLACES'.
- Check 16, while this deployment's secrets keep claude_env, an older package's Claude settings
  (the claude login's older name, whose key the one-time move copies). `claude_env_places(values)`: where
  the container keeps it and the claude login, as the registry names them (secrets_registry.older_places):
  Places(its name, its place, the login's place); None when no place is known. SHAPE_PROBE: the program a one-off
  container of the image runs, with no network and the secrets volume alone mounted read-only (docker/rkmon:
  `claude_env_checks`): the image's own rkmon_lib/claude_env_shape.py reads the file, judged by
  settings_file.secret_shape's own code, and secrets_registry.held's own code reads the login. Its arguments are the
  folder of the image's own code, then the two places; its answer, one JSON line: null while there is no such file,
  else the file's shape (claude_env_shape.to_json) with what root's own reading of it says, with no judgement
  (claude_env_shape.move: why the one-time move copies no key, "" when it copies it), or why its reader refused the
  file; and whether the login is set. `claude_env_answer(code, printed, places)`: that answer, as the container
  printed it (PRINTED; docker's exit code CODE): None, or Found(name, shape, move, login), MOVE the refusal when SHAPE
  is None; ValueError, with why, when the container failed or its answer cannot be read. `claude_env_rows(values, doc,
  found)`: what FOUND says against integrations.claude_code in the configuration VALUES (none for None): while the
  claude login is not set, a WARN when the move copies no key, in root's words, whose fix is `rkmon login claude`; a
  WARN for each thing the shape has that no setting carries, naming what the session will not send, whose
  fix is the file's edit (secrets_registry.claude_env_fix), which clears the row, while the login is set, and else
  names both ways, since check 16 cannot tell whether the move has run: the file's edit before the first start, or
  `rkmon login claude` (secrets_registry.claude_env_or_login_fix); a FAIL for each of its settings that sends the
  session or the key otherwise than the file does, whose fix, an edit of DOC's file, sets the file's; a PASS when
  none does; and a PASS that names the model variables the file sets, which the move leaves behind.
  Only where the file sets an address that holds no credential is the address compared, and the header and the
  word before the key only where it sends the key in a header; nothing is compared while the settings cannot carry
  the rest.
  `claude_env_skip(places, why, first=LOAD_IMAGE)`: the SKIP while no one-off container read the secrets: WHY says why
  not, then what to do before `rkmon confirm` runs again, FIRST (by default, load the image: deploy step 4).
  `claude_env_unread(places, ref, code, why)`: that SKIP when a one-off container of the image REF read nothing, by
  docker's exit code CODE, as docker/rkmon's ask_docker gives it, and WHY: TIMED_OUT, no answer in time, then check
  that docker runs containers; NOT_RUN, docker could not run it, then fix what docker says; any other, the image's own
  code could not read them or gave an answer that cannot be read, then load this package's image (step 4).

What needs each login is read as the container reads it (`login_rows`; `logins_needed(values)`: the names of those the
configuration VALUES needs; `login_places(values)`: where the container keeps each of them, as (its name, its place,
the files that hold it there: secrets_registry.held_files), its secrets folder container_layout.SECRETS_DIR, as
the load takes it there (rkmon_lib/derived.py), "" for a place that cannot be known; then a row for each place an older name
keeps it in that the one-time move copies to its own (secrets_registry.older_places), which counts as set until the
move ends: state 7): the Claude session, say, runs in the container while review automation is enabled.

The login walk's list, which deploy step 5 shows and state 7 reads (`walk_list(values, rows)`): each entry
the walk lists (secrets_registry.walk_entries), as {name, what, needed_by, waiting, use, state, found, command,
host_key}. What needs it is read from the configuration VALUES, your confirmed settings, here on the host, as the
container reads them; its state from ROWS, `rk logins --json` in the container, which reads the volumes. NOT NEEDED
while nothing that is on needs it and it is not set; MISSING when the container's own reading did not need it (its NOT
NEEDED: not set); UNKNOWN, and why, while ROWS does not list it. WAITING: the modes it waits for while nothing that is
on needs it (secrets_registry.waiting_words); USE: what it is used for meanwhile, with the program's name
(secrets_registry's prototype_use). COMMAND: the command that sets it, after the program's name, but `login` for the
ssh key of a host whose own key is not recorded (HOST_KEY, from ROWS). VALUES None: ROWS as they are. `UNSET`: the
states of one that is needed and not set as it must be, MISSING and WRONG: the walk's FAIL rows, and the logins state 7
says are missing.

After the first start:
- `change_lines(plan, values=None) -> (lines, count)`: what changes, from PLAN (change_plan.ChangePlan), and how
  many changes that is: after a package update, first `This package: A -> B`, or `This package: A -> A (OTHER_CODE)`
  for other code under the same version; then the parts "Your changes", "Changed outside your file" (none after a
  package update: apply writes every file anew), "Deploy keys you left blank whose default changed" and "Package
  defaults that changed", each `none` when it is empty. A change's line is its mark (`~` changed, `+` added, `-`
  removed), its key, and what runs now -> what apply makes it (JSON: a text in quotes, a list whole; the new value
  alone for one added, the old alone for one removed, none for a mapping), then a line that says how it applies
  (APPLIES, by its class; OUTSIDE_APPLIES for a change outside your file). A change of a key the web host's request
  service is made from (web_host_install.deploy_keys) has WEB_HOST's line after that one, while VALUES, the
  configuration the confirmed settings make, needs the service there (web_host_install.nothing_to_install): the
  service takes the change only when `rkmon web-host install` runs again, after apply. docker/.env is shown
  variable by variable, each named by the deploy key it is made from (deploy_plan.ENV_KEYS) "as docker/.env has it",
  any other file as a whole (BY_HAND). The count is each change's line but the package defaults', which are for your
  information: no key of your file sets them. With no change: one line that says so, and 0.
- `apply_lines(count, restarts, drain_minutes, stopped=False, first=False)`: what `rkmon apply` will do for them,
  wrapped at checks.WIDTH: let the work in flight finish (up to DRAIN_MINUTES minutes; None: stop it, as apply
  --no-drain does) while one of them restarts (RESTARTS), write the new settings, keeping the previous ones, make the
  container again (RESTARTS), run the doctor, and put the previous settings back on a new failure, or, for the FIRST
  apply over an older package, stop the new container, which makes a STOPPED container again too, with no drain;
  else, on a STOPPED container, write the new settings and leave it stopped.
- `rollback_lines(code, drain_minutes, stopped=False)`: what `rkmon rollback` will do, wrapped at checks.WIDTH: let
  the work in flight finish (up to DRAIN_MINUTES minutes), put back the settings from before the last apply, and
  its code with CODE, make the container again and run the doctor; on a STOPPED container, put them back and leave it
  stopped.
- `models_added(values, ran)`: the names of the models the configuration VALUES uses that RAN, the configuration that
  runs, does not use by the same provider at the same address: the ones confirm calls.
- `login_skips(values, rows)`: a SKIP for each login and secret the configuration VALUES needs that is not set yet:
  ROWS, `rk logins --json` in the container, list it as NOT NEEDED by what runs (so not set); `rkmon login` sets it
  before apply. None while ROWS is None.

docker/rkmon runs this on the host: the standard library only.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import pathlib
import re
from typing import NamedTuple

from . import (
    change_plan,
    claude_env_shape,
    code_volume,
    deploy_tree,
    container_layout,
    deployment as d,
    program,
    releases,
    secrets_registry,
    settings_file,
    steps,
    wrapping,
)
from .checks import PROBLEMS, WIDTH, Check, Described, Edit, Status, Steps
from .from_layout1 import UPDATES

REACH_TIMEOUT_S = 5
CODE_PROBE = (
    "import hashlib, json, pathlib, stat, sys\n"
    f"RELEASE_FILE = {code_volume.RELEASE_FILE!r}\n"
    f"{inspect.getsource(code_volume.fingerprints)}"
    "print(json.dumps(fingerprints(sys.argv[1], sys.argv[2:])))\n"
)
# A file's fingerprint, as code_volume.fingerprints makes it: a SHA-256 in hex.
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
MODEL_PROBE = """\
import json, sys, urllib.error, urllib.request
timeout = float(sys.argv[1])
out = {}
for url in sys.argv[2:]:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout)
        out[url] = ""
    except urllib.error.HTTPError:
        out[url] = ""
    except Exception as exc:
        out[url] = str(getattr(exc, "reason", exc))[:160] or type(exc).__name__
print(json.dumps(out))
"""
# Where a model row's call is made from: a one-off container of the image on this deployment's network.
CALLED = "called from a one-off container on this deployment's network"
HELD_PROBE = (
    "import json, pathlib, sys\n"
    f"{inspect.getsource(secrets_registry.held)}"
    "places = json.loads(sys.argv[1])\n"
    "found = {n for n, p, f in places if held(p, f)}\n"
    "print(json.dumps([n for n in dict.fromkeys(n for n, p, f in places) if n not in found]))\n"
)


@contextlib.contextmanager
def _as_the_container():
    """The capabilities read as the container reads them (features reads RKMON_CONTAINER=1 as the container's)."""
    old = os.environ.get("RKMON_CONTAINER")
    os.environ["RKMON_CONTAINER"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RKMON_CONTAINER", None)
        else:
            os.environ["RKMON_CONTAINER"] = old


def _shown(value) -> str:
    """A setting's value as the summary shows it: as rk config show shows it (config.safe_value), a text as it is,
    any other value, a blank text too, as JSON (a list or a map whole)."""
    from . import config

    value = config.safe_value(value)
    return (
        value
        if isinstance(value, str) and value
        else json.dumps(value, ensure_ascii=False)
    )


def _title(dotted: str) -> str:
    from . import file_types, schema

    node = schema.node(file_types.DEPLOY.load(), dotted) or {}
    return str(node.get("title") or dotted)


def headline(values: dict) -> str:
    """The container, then each feature's state, joined by " · " and wrapped at checks.WIDTH before a "·":

    container rocke-monitor-2 · review automation prototype · merge queue off · admin dashboard on
    · team dashboard off · alerts off · ticket creation off
    """
    from . import features

    parts = [f"container {d.project(values)}", *features.headline_parts(values)]
    return "\n".join(wrapping.wrap(wrapping.marked(parts), WIDTH))


def _parts() -> list[str]:
    """The summary's sections in the file's order, each by its key: the file's parts (deploy_tree.parts), each feature
    a section of its own."""
    from . import file_types

    out = []
    for part in deploy_tree.parts():
        if part == "features":
            names = file_types.DEPLOY.load()["properties"][part]["properties"]
            out += [f"{part}.{name}" for name in names]
        else:
            out.append(part)
    return out


def _section(part: str, rows: list) -> list[str]:
    """The section of the key PART: its title, then `name: text` for each (key, text) of ROWS, each key named from
    PART (PART's own key by its last name), aligned; a line that would pass WIDTH goes on under its text.
    """
    names = [
        k[len(part) + 1 :] if k != part else k.rpartition(".")[2] for k, _t in rows
    ]
    w = max(len(n) for n in names) + 1
    out = [_title(part)]
    for name, (_k, text) in zip(names, rows):
        lead = f"  {name + ':':<{w}} "
        out += wrapping.wrap(text, WIDTH, lead, " " * len(lead))
    return out


def _worded(made, values: dict, labels: dict | None) -> dict:
    """The text the summary shows for each key whose value it says in plain words too, by the key: see the module's
    docstring."""
    from . import config, features, image, registry

    settings, variables = made.layer["settings"], made.variables
    ra, mq, db = (
        f"features.{f}" for f in ("review_automation", "merge_queue", "dashboards")
    )
    r = d.resources(values)
    out = {}

    def note(key: str, words: str) -> None:
        out[key] = f"{_shown(registry.get(settings, key))} ({words})"

    if not registry.get(settings, "image.registry"):
        out["image.registry"] = "none (the image comes from a file)"
    if labels is not None:
        note("image.tag", f"on this host: {image.describe(labels)}")
    else:
        note("image.tag", "not on this host yet: step 4 brings it")
    note(
        "host.resources.ramdisk_gib",
        f"at {r['ramdisk_path']}" if r["ramdisk"] else "none",
    )
    for key in ("host.users.monitor_uid", "host.users.monitor_gid"):
        if registry.get(settings, key) is None:
            note(key, "yours")
    note("monitor.parallel_builds", f"at the start, within 1-{config.checks_max()}")
    if features.enabled(values, "review_automation"):
        if features.live(values, "review_automation"):
            note(f"{ra}.mode", "verified reviews and overlap notices are posted")
        else:
            note(f"{ra}.mode", "reviews are prepared and kept here, nothing is posted")
        note(f"{ra}.parallel_reviews", f"at the start, within 1-{config.pipes_max()}")
    else:
        note(f"{ra}.enabled", "no reviews and no Claude session")
    if features.enabled(values, "merge_queue"):
        if features.live(values, "merge_queue"):
            note(f"{mq}.mode", "the monitor carries out the plan")
        else:
            note(
                f"{mq}.mode",
                "the plan, gates and order are shown, nothing is armed, held or filed",
            )
        if features.switch(
            values, "merge_queue.auto_merge"
        ) == "enabled" and not features.live(values, "merge_queue"):
            note(
                f"{mq}.auto_merge.enabled", "arms nothing until the merge queue is live"
            )
        if features.on(values, "incident-issues") and features.live(
            values, "merge_queue"
        ):
            note(f"{mq}.develop_guard.enabled", "files issues and incidents")
    else:
        note(f"{mq}.enabled", "no queue, gates, plan or develop guard")
    if features.enabled(values, "dashboards.admin"):
        # Where docker publishes each dashboard on this host: docker/.env's variables, the plan's
        note(
            f"{db}.admin.port",
            f"{variables['RKMON_ADMIN_HOST']}:{variables['RKMON_PORT']} on this host",
        )
        served, host = _team_at(variables), d.get(values, "publish.html.host", "")
        if served:
            note(f"{db}.team.enabled", f"served here, {served}")
        elif features.enabled(values, "dashboards.team") and host:
            once = "" if features.live(values, "dashboards.team") else " once live"
            note(f"{db}.team.enabled", f"published to {host}{once}")
    else:
        note(f"{db}.admin.enabled", "no web server")
    # The modes of the other features that send: in prototype each does its work and sends nothing.
    for feature, kept, sent in (
        (
            "alerts",
            "messages are written and kept here, nothing is sent",
            "messages are sent",
        ),
        (
            "dashboards.team",
            "the page is made every tick, nothing is pushed to a web host",
            "the page is delivered every tick",
        ),
        (
            "ticket_creation",
            "tickets are written and kept here, nothing is filed",
            "tickets are filed",
        ),
    ):
        if features.enabled(values, feature):
            note(
                f"features.{feature}.mode",
                sent if features.live(values, feature) else kept,
            )
    if features.on(values, "weekly-report"):
        note(
            "features.weekly_report.enabled",
            "every week, a statistics report and notes for a lessons-learned review",
        )
    if not registry.get(settings, "integrations.claude_code.base_url"):
        out["integrations.claude_code.base_url"] = "none (Anthropic's own API)"
    return out


def models_lines(values: dict) -> list[str]:
    """The passes, resolved (roles.pass_lines: their one home): pass | model | provider | effort."""
    from . import roles

    return roles.pass_lines(values)


def login_rows(values: dict, rows: list | None) -> list:
    """(login, state) for each login and secret the login walk lists (secrets_registry.walk_entries), with what needs
    it read as the container reads it: NOT NEEDED, or its state as `rk logins` read it in the container (set, missing,
    wrong, unknown), or what needs it while no image can tell. Never a value."""
    from . import secrets_registry as sr

    have = {r.get("name"): str(r.get("state") or "") for r in rows or []}
    with _as_the_container():
        needs = [(e.name, sr.needed_by(values, e)) for e in sr.walk_entries(values)]
    out = []
    for name, need in needs:
        if not need or have.get(name) == "NOT NEEDED":
            out.append((name, "NOT NEEDED"))
        elif name in have:
            out.append((name, have[name].lower()))
        else:
            out.append(
                (name, f"needed by {', '.join(need)} (checked at step 5, login)")
            )
    return out


# The states of a login or secret that something that is on needs and that is not set as it must be: the walk's FAIL
# rows, and the logins state 7 says are missing (walk_list).
UNSET = ("MISSING", "WRONG")


def walk_list(values: dict | None, rows: list | None) -> list[dict]:
    """The login walk's list, which deploy step 5 shows and state 7 reads: see the module's docstring."""
    sr = secrets_registry
    rows = [r for r in rows or [] if isinstance(r, dict)]
    if values is None:
        return [{"waiting": "", "use": "", **r} for r in rows]
    have = {r.get("name"): r for r in rows}
    out = []
    with _as_the_container():
        for e in sr.walk_entries(values):
            row = have.get(e.name)
            needs = sr.needed_by(values, e)
            if row is None:
                state, found = "UNKNOWN", "the image's rk logins does not list it"
            else:
                state = str(row.get("state") or "UNKNOWN")
                found = str(row.get("found") or "")
            if not needs and state != "SET":
                state = "NOT NEEDED"
            elif needs and state == "NOT NEEDED":
                state = "MISSING"
            waiting = "" if needs else sr.waiting_words(values, e)
            host_key = state == "WRONG" and bool(row and row.get("host_key"))
            use = e.prototype_use.format(program=program.name()) if waiting else ""
            out.append(
                {
                    "name": e.name,
                    "what": e.what,
                    "needed_by": needs,
                    "waiting": waiting,
                    "use": use,
                    "state": state,
                    "found": found,
                    "command": "login" if host_key else " ".join(sr.set_command(e)),
                    "host_key": host_key,
                }
            )
    return out


def logins_needed(values: dict) -> list:
    """The names of the logins and secrets the configuration VALUES needs (secrets_registry.needed), read as the
    container reads it."""
    from . import secrets_registry as sr

    with _as_the_container():
        return [e.name for e in sr.needed(values)]


def _in_the_container(values: dict) -> tuple:
    """(the configuration VALUES with the container's secrets folder, container_layout.SECRETS_DIR, as the load
    takes it there (rkmon_lib/derived.py); the environment the container's places are read in).
    """
    cfg = json.loads(json.dumps(values))
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    cfg["paths"] = {**paths, "secrets_dir": container_layout.SECRETS_DIR}
    return cfg, {"RKMON_CONTAINER": "1"}


def login_places(values: dict) -> list[tuple[str, str, list | None]]:
    """Where the container keeps each login and secret the configuration VALUES needs: see the module's docstring."""
    sr = secrets_registry
    cfg, environ = _in_the_container(values)
    out, older = [], []
    with _as_the_container():
        for e in sr.needed(cfg):
            try:
                place = sr.path(cfg, e, environ)
            except Exception:  # noqa: BLE001 - no place is known for it
                place = ""
            out.append((e.name, place, sr.held_files(e)))
            try:
                copies = sr.older_places(cfg, e, environ)
            except Exception:  # noqa: BLE001 - no place is known for it
                copies = []
            older += [(e.name, p, sr.held_files(e, o)) for o, p in copies if o.copied]
    return out + older


def unheld(answer: str, places: list) -> tuple:
    """The names HELD_PROBE's ANSWER gives, in PLACES' order: see the module's docstring."""
    names = json.loads(answer)
    known = list(dict.fromkeys(p[0] for p in places))
    if not isinstance(names, list) or any(n not in known for n in names):
        raise ValueError(
            "the answer about the secrets volumes names none of the logins and secrets"
        )
    return tuple(n for n in known if n in names)


def models_in_use(values: dict) -> list[tuple[str, str, str, str]]:
    """Each model a pass uses, once: (its name, its provider, the address it is reached at, the passes that use it).
    A claude model is reached through the gateway (integrations.claude_code.base_url, config claude.gateway; "" when
    the file names none).
    """
    from . import bindings, config, roles

    out: dict = {}
    gateway = str((values.get("claude") or {}).get("gateway") or "").strip()
    for name, _role, owner in roles.PASSES:
        if not owner:
            model = roles._claude_pass(values, name)[0]
            out.setdefault(model, ["claude", gateway, []])[2].append(name)
            continue
        m = roles._bound(values, name)
        if m["by"] == "none":
            continue
        if m["by"] == "worker":
            w = config.worker(values)
            key = f"worker ({w.get('model') or '?'})"
            provider = "ollama" if w.get("backend") == "ollama" else "openai-compatible"
            out.setdefault(key, [provider, w.get("url") or "", []])[2].append(name)
        elif m["by"] == "binding":
            url = str(m.get("url") or "")
            url = "" if bindings.VAR_REF.fullmatch(url) else url
            out.setdefault(str(m.get("binding")), [m.get("kind") or "?", url, []])[
                2
            ].append(name)
    return [(k, v[0], v[1], ", ".join(v[2])) for k, v in out.items()]


def model_answers(code: int, printed: str, urls) -> dict:
    """Each address of URLS -> "" when it answered the one-off container that ran MODEL_PROBE, else why not: see the
    module's docstring."""
    import json

    said = None
    if code == 0:
        try:
            said = json.loads(printed)
        except ValueError:
            pass
    if not isinstance(said, dict):
        last = (printed.strip().splitlines() or [f"docker exited {code}"])[-1][:160]
        why = (
            f"its answer could not be read ({last})"
            if code == 0
            else f"the one-off container did not run ({last})"
        )
        return {url: why for url in urls}
    return {
        url: said[url][:160] if isinstance(said.get(url), str) else "no answer"
        for url in urls
    }


def model_rows(values: dict, answers, first: bool = True, names=None) -> list:
    """The model rows of the configuration VALUES, of the models NAMES (all by default): see the module's docstring."""
    from .config import safe_url

    doctor = steps.command_line("doctor")
    later = (
        "checked at step 7, from the container"
        if first
        else f"{doctor} calls it from the container"
    )
    login = (
        " (it needs the claude login, step 5)"
        if first
        else " (it needs the claude login)"
    )
    out = []
    for name, provider, url, _used in models_in_use(values):
        if names is not None and name not in names:
            continue
        row = f"model {name}"
        if not url or answers is None:
            out.append(
                Check(row, Status.SKIP, later + (login if provider == "claude" else ""))
            )
            continue
        why = re.sub(
            r"//[^/@\s]*@",
            "//",
            str(answers.get(url, "no answer")).replace(url, safe_url(url)),
        )
        if not why and provider == "claude":  # the gateway: no key is sent
            sr = secrets_registry
            login = sr.set_fix(sr.entry(values, sr.CLAUDE_LOGIN)).commands[0]
            said = f"the gateway answers at {safe_url(url)}, {CALLED} (no key is sent: {login} and {doctor} check it)"
            out.append(Check(row, Status.PASS, said))
            continue
        if not why:
            out.append(Check(row, Status.PASS, f"answers, {CALLED}"))
            continue
        out.append(
            Check(
                row,
                Status.WARN,
                f"no answer at {safe_url(url)}, {CALLED} ({why})",
                Steps(
                    (
                        "check the address, and that this deployment's network reaches it;",
                        doctor,
                        "calls it again from the container",
                    )
                ),
            )
        )
    return out


def models_added(values: dict, ran: dict) -> set:
    """The models the configuration VALUES uses that RAN does not use: see the module's docstring."""
    before = {(n, p, u) for n, p, u, _used in models_in_use(ran)}
    return {n for n, p, u, _used in models_in_use(values) if (n, p, u) not in before}


def login_skips(values: dict, rows) -> list:
    """The SKIP rows of the logins and secrets the configuration VALUES needs that are not set yet: see the module's
    docstring."""
    if rows is None:
        return []
    unset = {r.get("name") for r in rows if r.get("state") == "NOT NEEDED"}
    sets = f"{steps.command_line('login')} sets it before apply"
    return [
        Check(name, Status.SKIP, f"not set yet; {sets}")
        for name in logins_needed(values)
        if name in unset
    ]


def sections(
    made,
    values: dict,
    labels: dict | None = None,
    rows: list | None = None,
    updates: dict | None = None,
) -> list[str]:
    """Every setting MADE delivers, each key under its own section in the file's order, then the logins: see the
    module's docstring."""
    from . import features, file_types, registry

    settings = dict(made.layer["settings"])
    if updates:
        settings[UPDATES] = updates
    keys = registry.leaves(settings, settings_file.map_keys(file_types.DEPLOY.load()))
    worded = _worded(made, values, labels)
    out = []
    for part in _parts():
        if part == UPDATES and not updates:
            out.append(
                f"{_title(part)}: none (package updates only from a folder or an archive: "
                f"{steps.command_line(settings_file.PACKAGE_UPDATE)})"
            )
            continue
        mine = [k for k in keys if k == part or k.startswith(f"{part}.")]
        out += _section(
            part,
            [(k, worded.get(k) or _shown(registry.get(settings, k))) for k in mine],
        )
        if part == "features.review_automation" and features.enabled(
            values, "review_automation"
        ):
            out += [f"  {x}" for x in models_lines(values)]
    logins = login_rows(values, rows)
    w = max(len(k) for k, _ in logins) + 1
    out += ["Logins"] + [f"  {k + ':':<{w}} {v}" for k, v in logins]
    return out


def render(
    made,
    values: dict,
    labels: dict | None = None,
    rows: list | None = None,
    updates: dict | None = None,
) -> str:
    return "\n".join(
        [headline(values), ""] + sections(made, values, labels, rows, updates)
    )


def _team_at(variables: dict) -> str:
    """ADDRESS:PORT where docker publishes the team dashboard on this host, by docker/.env's VARIABLES the plan makes
    (RKMON_TEAM_HOST, RKMON_TEAM_PORT); "" while this host does not serve it."""
    if "RKMON_TEAM_PORT" not in variables:
        return ""
    return f"{variables.get('RKMON_TEAM_HOST', '')}:{variables['RKMON_TEAM_PORT']}"


def exposure(variables: dict) -> str:
    """The exposure sentence while this host serves the team dashboard, by VARIABLES; "" while it does not."""
    at = _team_at(variables)
    if not at:
        return ""
    return f"anyone who reaches {at} can read the team dashboard and send a priority request"


def exposure_changed(running: dict, variables: dict) -> str:
    """The exposure sentence of VARIABLES when a change serves the team dashboard or moves where it is served, by
    RUNNING: see the module's docstring."""
    at = _team_at(variables)
    return exposure(variables) if at and at != _team_at(running) else ""


# --- the image --------------------------------------------------------------------------------------------------------


def changed_copies(answer: str, package) -> list:
    """The files whose copies in the image, by CODE_PROBE's ANSWER, are not PACKAGE's: see the module's docstring."""
    try:
        held = json.loads(answer)
    except ValueError:
        raise ValueError("its answer is no JSON") from None
    if not isinstance(held, dict):
        raise ValueError("its answer is no JSON object")
    bound = code_volume.IMAGE_BOUND
    for rel, fingerprint in held.items():
        path = pathlib.PurePosixPath(rel)
        kept = any(
            rel == entry or (entry.endswith("/") and rel.startswith(entry))
            for entry in bound
        )
        if path.is_absolute() or ".." in path.parts or not kept:
            raise ValueError(f"its answer names {rel}, none of the files it keeps")
        if not isinstance(fingerprint, str) or not FINGERPRINT.fullmatch(fingerprint):
            raise ValueError(f"its answer has no fingerprint of {rel}")
    ours = code_volume.fingerprints(package, bound)
    return sorted(f for f in ours.keys() | held.keys() if ours.get(f) != held.get(f))


def image_rows(ref: str, code, first: bool = True, pull: bool = False) -> list:
    """The image row of the image REF: see the module's docstring."""
    if code is None:
        if first:
            return [
                Check(
                    "image", Status.SKIP, "checked at step 4, once it is on this host"
                )
            ]
        bring = steps.command_line("image pull" if pull else "image load IMAGE_FILE")
        return [
            Check("image", Status.SKIP, f"{ref} is not on this host; {bring} checks it")
        ]
    changes, why = code
    if changes is None:
        found = f"{ref}: its own code could not be read ({why})"
    elif changes:
        found = (
            f"{ref} was not built from this package's image recipe: its copies of {', '.join(changes)} are not this "
            "package's"
        )
    else:
        return [
            Check(
                "image",
                Status.PASS,
                f"{ref} was built from this package's image recipe",
            )
        ]
    return [Check("image", Status.FAIL, found, releases.new_image_fix())]


# --- what changes, after the first start ------------------------------------------------------------------------------

# How a change applies, by its class, as the line under it says it; a key of no class restarts (container.name:
# check 14 refuses a new one while the container exists).
APPLIES = {
    change_plan.NO_RESTART: "no restart: read at its next use",
    change_plan.NEW_CONTAINER: "restart: the work in flight finishes first",
}
# How a change found outside your file applies: docker reads those files only when it makes the container.
OUTSIDE_APPLIES = "restart"
# A file found unlike what apply writes, by its change's mark ("what runs now -> what apply makes it").
BY_HAND = {"~": "edited by hand", "+": "removed by hand", "-": "added by hand"}
# Where the line that says how a change applies starts.
APPLIES_INDENT = " " * 8
# The line under a change of a key the web host's request service is made from (web_host_install.deploy_keys), while
# the confirmed settings need that service there: the service takes the change only when the install runs again, from
# the settings apply delivers. {command}: the install's command.
WEB_HOST = (
    "web host: the request service there takes it when you run {command} after apply"
)
# What the package line adds when the package's code changed and its version did not (change_plan: a package is its
# version and its code).
OTHER_CODE = "other code under the same version"


def _value(value) -> str:
    """A setting's value as a change's line shows it: JSON; "" for a mapping, or for none."""
    if value is deploy_tree.MISSING or isinstance(value, dict):
        return ""
    return json.dumps(value, ensure_ascii=False)


def _env_value(value) -> str:
    """A docker/.env value as a change's line shows it: as the file has it; "" for none."""
    if value is None:
        return ""
    return value or '""'


def _line(mark: str, key: str, old: str, new: str) -> str:
    """A change's line: its MARK and KEY, then OLD -> NEW; the new value alone for one added, the old alone for one
    removed; nothing more where a value is "" (a mapping)."""
    if mark == "~":
        values = f"{old} -> {new}" if old and new else ""
    else:
        values = new if mark == "+" else old
    return f"  {mark} {key}" + (f"  {values}" if values else "")


def _setting(change, web_host=frozenset()) -> tuple:
    """A change of the settings: its line, then how it applies; then, for a key of WEB_HOST (_web_host_keys), WEB_HOST's
    line."""
    applies = APPLIES.get(change.applies, APPLIES[change_plan.NEW_CONTAINER])
    line = _line(change.mark, change.key, _value(change.old), _value(change.new))
    if change.key not in web_host:
        return line, APPLIES_INDENT + applies
    install = WEB_HOST.format(command=steps.command_line("web-host install"))
    return line, APPLIES_INDENT + applies, APPLIES_INDENT + install


def _web_host_keys(values) -> frozenset:
    """The keys the web host's request service is made from (web_host_install.deploy_keys) while VALUES, the
    configuration the confirmed settings make, needs that service there (web_host_install.nothing_to_install); none
    else, and for VALUES None."""
    from . import web_host_install

    if values is None or web_host_install.nothing_to_install(values):
        return frozenset()
    return frozenset(web_host_install.deploy_keys())


def _outside(change) -> list:
    """A file found outside your file (change_plan's `outside`), as changes: docker/.env variable by variable, else the
    file whole: see the module's docstring."""
    from . import deliver, deploy_plan, image

    out = []
    env = deliver.ENV_FILE
    if change.found_in == env and deploy_tree.MISSING not in (change.old, change.new):
        found, written = image.dotenv_text(change.old), image.dotenv_text(change.new)
        for name in [*written, *(k for k in found if k not in written)]:
            old, new = found.get(name), written.get(name)
            if old == new:
                continue
            key = (deploy_plan.ENV_KEYS.get(name) or (name,))[0]
            mark = "+" if old is None else "-" if new is None else "~"
            where = {
                "~": f"as {env} has it",
                "+": f"removed from {env} by hand",
                "-": f"added to {env} by hand",
            }[mark]
            line = _line(mark, f"{key}, {where}", _env_value(old), _env_value(new))
            out.append((line, APPLIES_INDENT + OUTSIDE_APPLIES))
    if out:
        return out
    line = f"  {change.mark} {change.found_in}, {BY_HAND[change.mark]}"
    return [(line, APPLIES_INDENT + OUTSIDE_APPLIES)]


def _part(title: str, changes: list, about: str = "") -> list:
    """A part of what changes: TITLE (ABOUT after it), then each change's lines; "TITLE: none" for no change."""
    if not changes:
        return [f"{title}: none"]
    return [f"{title}{about}:"] + [line for change in changes for line in change]


def change_lines(plan, values=None) -> tuple:
    """(the lines that say what changes, how many changes) of PLAN, a change_plan.ChangePlan, with VALUES, the
    configuration the confirmed settings make (None: not known): see the module's docstring.
    """
    out, counted = [], []
    web_host = _web_host_keys(values)
    if plan.package is not None:
        then, now = plan.package
        other = f" ({OTHER_CODE})" if then == now else ""
        out.append(f"This package: {then} -> {now}{other}")
        out.append(APPLIES_INDENT + APPLIES[change_plan.NEW_CONTAINER])
        counted.append(plan.package)
    yours = [_setting(c, web_host) for c in plan.yours]
    out += _part("Your changes", yours)
    counted += yours
    if plan.package is None:
        outside = [x for c in plan.outside for x in _outside(c)]
        out += _part(
            "Changed outside your file",
            outside,
            " (apply puts your file's value back)",
        )
        counted += outside
    blank = [_setting(c, web_host) for c in plan.blank_defaults]
    out += _part("Deploy keys you left blank whose default changed", blank)
    counted += blank
    defaults = [
        (_line(c.mark, c.key, _value(c.old), _value(c.new)),)
        for c in plan.package_defaults
    ]
    out += _part(
        "Package defaults that changed",
        defaults,
        ", for your information (no key of your file sets them)",
    )
    if not counted:
        return ["No change: the deployment has these settings already."], 0
    return out, len(counted)


def apply_lines(
    count: int,
    restarts: bool,
    drain_minutes,
    stopped: bool = False,
    first: bool = False,
) -> list:
    """What apply will do for COUNT changes: see the module's docstring."""
    does = ["write the new settings (the previous ones kept)"]
    if stopped and not first:
        does.append("and leave the container stopped")
    else:
        if restarts:
            drain = [] if stopped else [_drain(drain_minutes)]
            does = [*drain, *does, "make the container again"]
        does += [
            "run the doctor",
            (
                "and stop the new container on a failure"
                if first
                else "and put the previous settings back on a new failure"
            ),
        ]
    text = f"{wrapping.counted(count, 'change')}. apply will: {', '.join(does)}."
    return wrapping.wrap(text, WIDTH)


def rollback_lines(code: bool, drain_minutes, stopped: bool = False) -> list:
    """What rollback will do: see the module's docstring."""
    what = "the code and settings" if code else "the settings"
    does = [f"put back {what} from before your last apply"]
    if stopped:
        does.append("and leave the container stopped")
    else:
        does = [
            _drain(drain_minutes),
            *does,
            "make the container again",
            "and run the doctor",
        ]
    return wrapping.wrap(f"rollback will: {', '.join(does)}.", WIDTH)


def _drain(drain_minutes) -> str:
    """The drain as apply and rollback say it: the work in flight finishes, for DRAIN_MINUTES minutes at most; None:
    it stops."""
    if drain_minutes is None:
        return "stop the work in flight (it starts again later)"
    return f"let the work in flight finish (up to {drain_minutes:g} minutes)"


# --- check 16: the older Claude secret --------------------------------------------------------------------------------

# Check 16's rows' name.
CLAUDE_ROW = "Claude Code"
# The settings check 16 compares, by their section in the deploy settings file.
CLAUDE_SECTION = "integrations.claude_code"
# How the session sends the key, by api_key_sent_as, as check 16's rows say it.
SENT_AS = {
    "header": "in a header",
    "bearer": "as a bearer token",
    "header_and_bearer": "in a header and as a bearer token",
}
# What check 16's SKIP asks for before `rkmon confirm` runs again, by why no one-off container read the secrets: the
# image is not on this host; docker gave no answer in time; docker could not run the container; the image's own code
# could not read them (claude_env_skip, claude_env_unread).
LOAD_IMAGE = "load the image (step 4)"
DOCKER_RUNS = "check that docker runs containers"
DOCKER_SAYS = "fix what docker says"
LOAD_OURS = "load this package's image (step 4)"
# docker's exit codes, as docker/rkmon's ask_docker gives them: no answer in time, and docker could not run the
# container (any other is the container's own).
TIMED_OUT, NOT_RUN = 124, 125


def _compiled(name: str, pattern) -> str:
    """A line of a program of its own that makes the compiled PATTERN again, as NAME."""
    return f"{name} = re.compile({pattern.pattern!r}, {int(pattern.flags)})\n"


SHAPE_PROBE = (
    "import json, os, pathlib, re, runpy, sys, urllib.parse\n"
    + "".join(
        _compiled(n, getattr(settings_file, n))
        for n in ("TOKENS", "PEM", "JWT", "ADDRESS")
    )
    + f"CREDENTIAL_NAMES = {sorted(settings_file.CREDENTIAL_NAMES)!r}\n"
    + f"CREDENTIAL_ENDS = {settings_file.CREDENTIAL_ENDS!r}\n"
    + inspect.getsource(settings_file.secret_shape)
    + inspect.getsource(secrets_registry.held)
    + "code, older, login = sys.argv[1:]\n"
    "out = None\n"
    "if os.path.lexists(older):\n"
    "    reader = runpy.run_path(os.path.join(code, 'rkmon_lib', 'claude_env_shape.py'))\n"
    "    try:\n"
    "        shape = reader['to_json'](reader['read'](older, secret_shape))\n"
    "        out = {'shape': shape, 'move': reader['move'](reader['read'](older))}\n"
    "    except (OSError, ValueError) as exc:\n"
    "        out = {'refused': str(exc)}\n"
    "    out['login'] = held(login)\n"
    "print(json.dumps(out))\n"
)


class Places(NamedTuple):
    """Where check 16 reads, in the container: see the module's docstring."""

    name: str
    older: str
    login: str


class Found(NamedTuple):
    """What check 16's one-off container found: see the module's docstring. SHAPE is a claude_env_shape.Shape, None
    when its reader refused the file; MOVE says why the one-time move copies no key from it, as root's own reading
    gives it (claude_env_shape.move): why its reader refused it, while SHAPE is None, and "" when the move copies the
    key."""

    name: str
    shape: object
    move: str
    login: bool


def claude_env_places(values: dict):
    """Where the container keeps claude_env and the claude login, for the configuration VALUES: see the module's
    docstring."""
    sr = secrets_registry
    cfg, environ = _in_the_container(values)
    with _as_the_container():
        entry = sr.entry(cfg, sr.CLAUDE_LOGIN)
        try:
            copies = [
                (o, p) for o, p in sr.older_places(cfg, entry, environ) if o.copied
            ]
            login = sr.path(cfg, entry, environ)
        except Exception:  # noqa: BLE001 - no place is known for it
            return None
    if not copies:
        return None
    older, place = copies[0]
    return Places(older.name, place, login)


def claude_env_answer(code: int, printed: str, places: Places):
    """SHAPE_PROBE's answer about PLACES, as the one-off container printed it: see the module's docstring."""
    lines = printed.strip().splitlines()
    if code != 0:
        raise ValueError(lines[-1][:200] if lines else f"docker exited {code}")
    try:
        data = json.loads(lines[-1] if lines else "")
    except ValueError:
        raise ValueError("its answer is no JSON") from None
    if data is None:
        return None
    if (
        not isinstance(data, dict)
        or set(data) not in ({"shape", "move", "login"}, {"refused", "login"})
        or not isinstance(data["login"], bool)
    ):
        raise ValueError("its answer is no JSON object of what it found")
    if "shape" in data:
        shape = claude_env_shape.from_json(data["shape"], settings_file.secret_shape)
        move = _said(data["move"], "why the move copies no key", blank=True)
        return Found(places.name, shape, move, data["login"])
    refused = _said(data["refused"], "why it refused the file")
    return Found(places.name, None, refused, data["login"])


def _said(words, what: str, blank: bool = False) -> str:
    """WORDS, a part of SHAPE_PROBE's answer that says WHAT, when a row may show them: one line that looks like no
    secret, or "" when BLANK; else ValueError."""
    if not (
        isinstance(words, str)
        and (len(words.splitlines()) == 1 or (blank and not words))
        and not settings_file.secret_shape(words)
    ):
        raise ValueError(f"its answer says {what} in words it cannot show")
    return words


def claude_env_skip(places: Places, why: str, first: str = LOAD_IMAGE) -> list:
    """Check 16's SKIP while no one-off container read the secrets: see the module's docstring."""
    found = (
        f"this deployment's secrets are not read for {places.name}, an older package's Claude settings: {why}; "
        f"{first}, then run {steps.command_line('confirm')} again"
    )
    return [Check(CLAUDE_ROW, Status.SKIP, found)]


def claude_env_unread(places: Places, ref: str, code: int, why: str) -> list:
    """Check 16's SKIP when a one-off container of the image REF read nothing: see the module's docstring."""
    if code == TIMED_OUT:
        return claude_env_skip(
            places, f"a one-off container of {ref} {why}", DOCKER_RUNS
        )
    if code == NOT_RUN:
        return claude_env_skip(
            places,
            f"docker could not run a one-off container of {ref} ({why})",
            DOCKER_SAYS,
        )
    return claude_env_skip(
        places, f"a one-off container of {ref} could not read them ({why})", LOAD_OURS
    )


def _claude_code(values: dict) -> dict:
    """integrations.claude_code's settings as the configuration VALUES has them (deploy_tree.feeds names where), by
    their keys, each a text, blank space aside; how the key is sent, its default when it is blank (registry.default),
    as the session takes it (orchestrator.session_variables)."""
    from . import registry

    lead = CLAUDE_SECTION + "."
    out = {}
    for key, internal in deploy_tree.feeds():
        if key.startswith(lead):
            value = deploy_tree.get(values, internal)
            missing = value is deploy_tree.MISSING or value is None
            out[key[len(lead) :]] = "" if missing else str(value).strip()
    out["api_key_sent_as"] = out.get("api_key_sent_as") or registry.default(
        "claude.api_key_sent_as"
    )
    return out


def _compared(shape, made: dict) -> list:
    """The settings of MADE, integrations.claude_code's as SHAPE makes them, that a difference would change: where the
    session sends, while SHAPE sets an address; with a key, how it is sent, then, but as a bearer token, in which
    header, then, in a header of its own, after which word."""
    keys = ["base_url"] if shape.base_url else []
    if any(v in shape.variables for v in claude_env_shape.KEY_VARS):
        keys.append("api_key_sent_as")
        if made["api_key_sent_as"] != "bearer":
            keys.append("api_key_header")
            if made["api_key_header"]:
                keys.append("api_key_scheme")
    return keys


def _does(key: str, value: str) -> str:
    """What a claude_env does that makes integrations.claude_code's KEY VALUE, in words."""
    if key == "base_url":
        return f"sends the session to {value}"
    if key == "api_key_sent_as":
        return f"sends the key {SENT_AS[value]}"
    if key == "api_key_header":
        return (
            f"sends the key in the custom header {value}"
            if value
            else "sends the key in no custom header"
        )
    return (
        f"sends the word {value} before the key"
        if value
        else "sends no word before the key"
    )


def claude_env_rows(values: dict, doc, found) -> list:
    """Check 16's rows of FOUND, against the configuration VALUES and DOC's file: see the module's docstring."""
    if found is None:
        return []
    sr, name, out = secrets_registry, found.name, []
    login = sr.entry({}, sr.CLAUDE_LOGIN)
    if found.move and not found.login:
        words = f"the one-time move copies no key from {name} to the {login.name} login: {found.move}"
        out.append(Check(CLAUDE_ROW, Status.WARN, words, sr.set_fix(login)))
    shape = found.shape
    if shape is None:
        return out
    made, cannot = claude_env_shape.settings(shape)
    fix = sr.claude_env_fix if found.login else sr.claude_env_or_login_fix
    out += [
        Check(
            CLAUDE_ROW,
            Status.WARN,
            f"the session will not send all that {name} sends: {c.found}",
            fix(c.fix),
        )
        for c in cannot
    ]
    # An address with a credential has its own row: the rest is compared while the settings carry it.
    rest = claude_env_shape.settings(shape._replace(base_url_credential=""))[1]
    compared = [] if rest else _compared(shape, made)
    have, file = _claude_code(values), _file(doc)
    fails = [
        Check(
            CLAUDE_ROW,
            Status.FAIL,
            f"{name} {_does(k, made[k])}, and {CLAUDE_SECTION}.{k} says {json.dumps(have.get(k, ''))}",
            Edit(f"{CLAUDE_SECTION}.{k}", file, made[k]),
        )
        for k in compared
        if have.get(k, "") != made[k]
    ]
    out += fails
    if compared and not fails:
        out.append(
            Check(
                CLAUDE_ROW,
                Status.PASS,
                f"{CLAUDE_SECTION} sends the {'key' if cannot else 'session'} as {name} does",
            )
        )
    if shape.models:
        them = "them" if len(shape.models) > 1 else "it"
        out.append(
            Check(
                CLAUDE_ROW,
                Status.PASS,
                f"{name} sets {wrapping.joined(shape.models)}; the move leaves {them} behind, since each pass asks "
                "for its model (features.review_automation) by its built-in id",
            )
        )
    return out


# --- the checks -------------------------------------------------------------------------------------------------------


def _file(doc) -> str:
    """The deploy settings file a fix edits: DOC's, as a message shows it; deploy_tree.FILE's words when it has no
    path."""
    return settings_file.shown(doc.path) if doc.path else deploy_tree.FILE


def checks(
    resolved, doc, rows: list, image=(), models=(), logins=(), claude=()
) -> list:
    """The confirm's check rows of DOC, the deploy settings document, and RESOLVED, the configuration the container
    runs with it (runtime_config.Resolved): the features' coherence (coherence.deploy's PASS row, when it passes); each
    FAIL and WARN row of ROWS, the file's and its settings' own as the confirm read them (deployment.read, then
    deployment.confirm_rows: deployment.settings_checks, coherence.resources and coherence.skills), and check 18's
    PASS row among them, which names the team's skills the confirm delivers, and check 7's PASS row, which says where
    the team dashboard in prototype is served; each line of what the configuration refuses to run that no row says
    already; the image's rows IMAGE (`image_rows`); the models' rows MODELS (`model_rows`); the logins' rows LOGINS
    (`login_skips`); and check 16's rows CLAUDE (`claude_env_rows`). A FAIL refuses the confirm.
    """
    from . import bindings, features
    from .coherence import SKILLS, TEAM_ROW

    values = resolved.values
    file = _file(doc)
    own = settings_file.validate(doc)
    out = [c for c in own if c.name == "coherence" and c.status is Status.PASS]
    out += [c for c in rows if c.status in PROBLEMS or c.name in (SKILLS, TEAM_ROW)]
    shown = {c.found for c in out}
    required = {k for k, _what, _line in d.REQUIRED_SETTINGS}
    names = {d.N(k) for k in required}
    coherence = {what for _s, what, _f, _st in features.coherence(values)}
    seen = set()
    # the file's own rows name these keys already
    refused = {c.name for c in own if c.status in PROBLEMS}
    for line in list(resolved.problems) + bindings.problems(values):
        key = line.split(" ", 1)[0]
        tree = d.N(key)
        if any(
            r == tree or r.startswith(tree + ".") or tree.startswith(r + ".")
            for r in refused
        ):
            continue
        if (
            line in seen
            or line in shown
            or line in coherence
            or key in required | names
        ):
            continue
        seen.add(line)
        out.append(
            Check(
                "configuration",
                Status.FAIL,
                line,
                Edit(tree, file, Described("a value it takes")),
            )
        )
    return out + list(image) + list(models) + list(logins) + list(claude)
