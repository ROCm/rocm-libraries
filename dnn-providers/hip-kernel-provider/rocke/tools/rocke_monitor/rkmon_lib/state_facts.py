# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The state table's facts: `gather` makes states.Facts for a package folder from its
host's records, the settings file as it is now, this package, and docker's answers, which the caller's DOCKER gives,
so that the facts ask docker nothing of their own.

- `gather(package_dir, docker, *, now=None, package=None, carried=False) -> states.Facts`: the
  facts of the package in the folder PACKAGE, with PACKAGE_DIR's records. With no PACKAGE, the package is this one:
  PACKAGE_DIR's VERSION, and the layout of the deploy file this code reads; else PACKAGE's VERSION, and the layout
  its own deploy schema fixes (file_types.DEPLOY.layout_in; `NoLayout`, before anything is read, when it fixes
  none): an update asks for the facts of the package it puts, or would put, in PACKAGE_DIR, while the code that runs
  it is often the package it replaces (rkmon_lib/package_update.py); its dry run reads the records as its carry
  leaves them (CARRIED: records.read). The facts come in the table's order, each read or asked at most once, and
  only while the state is not decided before it:
  1. The records (records.read), and whether they keep a first tick (records.FirstTick: deploy step 8 passed after
     a start). With no confirm recorded: state 2's facts, and nothing else is read or asked.
  2. Whose the confirmed container.name is (DOCKER.owner); the confirmed settings file: whether it is there, its
     layout (a settings_file.Refused for another layout gives it), and whether its own values changed
     (deploy_plan.file_sha256 against the records'; a file that cannot be read as settings has changed); whether
     the package changed: its version (image.package_version), or, under the same version, its code
     (image.code_sha256: other code, such as a later commit of updates.branch, is a change too; an OSError when the
     package has no .dockerignore); an apply or rollback under way, and whether it still runs (records.runs); the
     change put back, while its fingerprint is the confirmed one (PUT_BACK's sentence); and whether the change under
     way, or the failed change kept, is a first apply over an older package (records.first_apply).
  3. Docker's answers, only while the name is not another deployment's or unknown (the owner is identity's NONE or
     OURS: asks_docker) and the state is none of those the records decide alone (states.RECORDS_ONLY), in this
     order: the confirmed image (DOCKER.image_here, by its reference, records.Image; none here, state 6); with it,
     the logins missing or wrong (DOCKER.logins_missing; state 7); the container (DOCKER.container; none, state 8);
     while it runs, whether its work volume is set up (DOCKER.set_up; not set up, state 8 too); then the delivered
     layer (DOCKER.layer), whose fingerprint is the delivered one; and while the container runs, the monitor's answer
     (DOCKER.monitor).
  4. With the delivered layer, the files deliver writes, docker/.env and the override as they are (deliver.on_host)
     and that layer, against the fingerprints of what deliver last wrote there, which this folder's records keep
     (records.Written): those that differ, by their names in deliver.WRITES (deliver.changed_outside, state 9). A
     file with no fingerprint kept counts as unchanged.
  A fact not asked reads as nothing asked, its empty value: no image here, no login missing, no container (and no
  work volume found not set up), nothing delivered or loaded, standby and pause off. So, while docker cannot say whose
  the name is (an owner that is none of identity's words), the state the facts give is known only when it is one of
  RECORDS_ONLY: the caller says that the rest is unknown. NOW, an aware time (now by default), gives how long the
  container has run.
  private_files.Refused when the records, docker/.env or the override break a private file's rules (its rows say
  why, with the fix), and records.Unreadable when the records are none this package reads.
- DOCKER, the caller's, answers for the confirmed container.name, PROJECT:
  - `owner(project) -> str`: whose the name is on this host: identity's NONE, OURS or OTHER; any other word when
    docker cannot say.
  - `image_here(image) -> bool`: whether the image IMAGE, NAME:TAG, is on this host.
  - `logins_missing(image) -> tuple`: the names of the logins and secrets the confirmed settings need that are missing
    or wrong (confirm.UNSET), as a container of IMAGE lists them.
  - `container(project) -> (status, started)`: the container: states.NO_CONTAINER, STOPPED or RUNNING (a stop that
    `rk stop` left in a running container reads as STOPPED); STARTED, when it started, an aware time, while it runs,
    else None.
  - `set_up(project) -> bool`: whether the running container's work volume is set up, as `rk provision --check` in it
    says (rkmon_lib/provision.py); False while it waits for its first start (it was started some other way).
  - `layer(project) -> str | None`: the delivered layer's text, deliver.LAYER_FILE in the settings volume; None when
    none is delivered.
  - `monitor(project) -> Monitor(loaded_fingerprint, standby, paused)`: what the running monitor says: the
    fingerprint of the layer it last loaded, its applied record's ("" for none), and whether standby and a pause are
    on.
- `asks_docker(owner) -> bool`: whether docker's answers are asked (step 3 above): while OWNER, whose the confirmed
  container.name is, is identity's NONE or OURS; never while it is another deployment's or unknown.
- `monitor_asked(facts) -> bool`: whether FACTS hold the running monitor's answer (DOCKER.monitor: what it loaded, and
  whether standby and a pause are on), as `gather` asks it: the container runs and its work volume is set up. Until
  then those facts read as nothing asked: no standby, no pause.
- PUT_BACK: state 10's sentence (states.Facts.put_back), by records.REASONS; `rollback`'s names the command. FAILED:
  what follows it when something failed (records.PutBack.failure), by reason, `: FAILURE` for any other: the line
  above state 10's next line names the failure.
- `layer_fingerprint(layer_text) -> str`: the fingerprint the delivered layer's text LAYER_TEXT was made from, which
  state 11 holds against the confirmed one; "" for none, or a layer that does not say.
- `NoLayout`: a package whose deploy schema fixes no layout of its deploy file (a ValueError); its message names the
  package's folder.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib
from typing import NamedTuple

from . import (
    deliver,
    deploy_plan,
    file_types,
    identity,
    image,
    progress,
    records,
    settings_file,
    states,
    steps,
)

PUT_BACK = {
    "rollback": "the last change you confirmed was put back by {}",
    "restart": "the last change you confirmed was put back after a failed restart",
    "load": "the last change you confirmed was put back after a failed load",
    "doctor": "the last change you confirmed was put back after a failed doctor check",
    records.FIRST_APPLY: "the first apply over an older package failed and stopped the new container",
}
FAILED = {"rollback": ", which ended with a failure: {}"}


class NoLayout(ValueError):
    """A package that says no layout of its deploy file: see the module's docstring."""


class Monitor(NamedTuple):
    """What the running monitor says: see the module's docstring."""

    loaded_fingerprint: str
    standby: bool
    paused: bool


def monitor_asked(facts) -> bool:
    """Whether FACTS hold the running monitor's answer: see the module's docstring."""
    return facts.container == states.RUNNING and not facts.not_set_up


def asks_docker(owner: str) -> bool:
    """Whether docker's answers are asked: see the module's docstring."""
    return owner in (identity.NONE, identity.OURS)


def _nothing(
    name_owner: str, kept: records.Records, package_layout: int
) -> states.Facts:
    """The facts with nothing else read or asked than the records KEPT: state 2's, for a package folder with no
    confirm, else the confirm's own, the package reading the layout PACKAGE_LAYOUT, each other fact at its empty value
    but the first tick the records keep.
    """
    deploy = file_types.DEPLOY
    c = kept.confirmed
    when = records.parsed(c.at) if c else None
    return states.Facts(
        name_owner=name_owner,
        container_name=c.container_name if c else "",
        settings_file=c.settings_file if c else "",
        file_exists=False,
        under_way="",
        under_way_runs=False,
        first_apply=False,
        file_layout=deploy.version,
        package_layout=package_layout,
        file_changed=False,
        package_changed=False,
        image=c.image.reference if c else "",
        image_here=False,
        registry=bool(c.image.registry) if c else False,
        logins_missing=(),
        container=states.NO_CONTAINER,
        running_minutes=0.0,
        not_set_up=False,
        put_back="",
        put_back_fix=None,
        confirmed_fingerprint=c.fingerprint if c else "",
        confirmed_at=progress.shown_time(when) if when else "",
        waiting_changes=c.changes if c else 0,
        delivered_fingerprint="",
        loaded_fingerprint="",
        hand_edited=(),
        standby=False,
        paused=False,
        first_tick_recorded=kept.first_tick is not None,
    )


def _settings(path: str, confirmed) -> tuple:
    """(its layout; whether it changed after the confirm, as a file that cannot be read as settings has) of the
    settings file at PATH."""
    deploy = file_types.DEPLOY
    try:
        doc = settings_file.read(path, deploy)
    except settings_file.Refused as refused:
        return (refused.layouts[0] if refused.layouts else deploy.version), True
    except OSError:
        return deploy.version, True
    return deploy.version, deploy_plan.file_sha256(doc) != confirmed.file_sha256


def _put_back(record, confirmed) -> tuple:
    """(PUT_BACK's sentence, then what failed (FAILED), the fix) for the change put back, while it is the confirmed
    one; ("", None) else."""
    if record is None or record.fingerprint != confirmed.fingerprint:
        return "", None
    words = PUT_BACK[record.reason]
    if record.reason == "rollback":
        words = words.format(steps.command_line("rollback"))
    if record.failure:
        words += FAILED.get(record.reason, ": {}").format(record.failure)
    return words, record.fix


def _package_changed(package, confirmed) -> bool:
    """Whether the package in the folder PACKAGE is not the one the record CONFIRMED was confirmed with: its version,
    or under the same version its code."""
    if image.package_version(package) != confirmed.package_version:
        return True
    return image.code_sha256(package) != confirmed.code_sha256


def layer_fingerprint(layer_text) -> str:
    """The fingerprint the delivered layer LAYER_TEXT was made from; "" for none, or a layer that does not say."""
    found = deploy_plan.read_layer(layer_text).get("fingerprint")
    return found if isinstance(found, str) else ""


def _reads(package) -> int:
    """The layout of the deploy file that the package in the folder PACKAGE reads, which its own deploy schema fixes;
    NoLayout when it fixes none."""
    layout = file_types.DEPLOY.layout_in(package)
    if layout is None:
        shown = settings_file.shown(package)
        raise NoLayout(f"the package in {shown} says no layout of its deploy file")
    return layout


def gather(
    package_dir, docker, *, now=None, package=None, carried=False
) -> states.Facts:
    """The state table's facts: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    if package is None:
        package, reads = pkg, file_types.DEPLOY.version
    else:
        reads = _reads(package)
    kept = records.read(pkg, carried=carried)
    c = kept.confirmed
    if c is None:
        return _nothing(identity.NONE, kept, reads)
    owner = docker.owner(c.container_name)
    path = os.path.expanduser(c.settings_file)
    exists = os.path.isfile(path)
    layout, changed = file_types.DEPLOY.version, False
    if exists:
        layout, changed = _settings(path, c)
    under_way = kept.under_way.command if kept.under_way else ""
    put_back, fix = _put_back(kept.put_back, c)
    facts = dataclasses.replace(
        _nothing(owner, kept, reads),
        file_exists=exists,
        under_way=under_way,
        under_way_runs=bool(under_way) and records.runs(pkg),
        first_apply=records.first_apply(kept),
        file_layout=layout,
        file_changed=changed,
        package_changed=_package_changed(package, c),
        put_back=put_back,
        put_back_fix=fix,
    )
    if not asks_docker(owner):
        return facts
    if states.current(facts).n in states.RECORDS_ONLY:
        return facts
    facts = dataclasses.replace(
        facts, image_here=bool(docker.image_here(c.image.reference))
    )
    if not facts.image_here:
        return facts
    missing = tuple(docker.logins_missing(c.image.reference))
    facts = dataclasses.replace(facts, logins_missing=missing)
    if missing:
        return facts
    status, started = docker.container(c.container_name)
    minutes = 0.0
    if started is not None:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        minutes = max(0.0, (now - started).total_seconds() / 60)
    facts = dataclasses.replace(facts, container=status, running_minutes=minutes)
    if status == states.NO_CONTAINER:
        return facts
    if status == states.RUNNING and not docker.set_up(c.container_name):
        return dataclasses.replace(facts, not_set_up=True)
    layer_text = docker.layer(c.container_name)
    edited = deliver.changed_outside(kept.written, *deliver.on_host(pkg), layer_text)
    facts = dataclasses.replace(
        facts, hand_edited=edited, delivered_fingerprint=layer_fingerprint(layer_text)
    )
    if status == states.RUNNING:
        answer = docker.monitor(c.container_name)
        facts = dataclasses.replace(
            facts,
            loaded_fingerprint=answer.loaded_fingerprint,
            standby=bool(answer.standby),
            paused=bool(answer.paused),
        )
    return facts
