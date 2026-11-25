# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What changes on a running deployment: what confirm shows,
each change as "what runs now -> what apply makes it", with how it applies.

- `Applied(layer, env, override)`: what runs now, each a whole text as it is: the delivered layer's
  (deliver.LAYER_FILE in the settings volume), and on this host docker/.env's and the override's, None for none
  (deliver.on_host).
- `Confirmed(doc, host)`: what apply makes: the deploy settings as you confirm them, on HOST (deploy_plan.Host: your
  ids on this host, which a blank id stands for, and the confirm's time). Their plan with this package
  (deploy_plan.plan) is what apply delivers.
- `Package(version, deploy, defaults, code)`: a package as the change plan reads it: its version; its deploy
  settings' file type (file_types.DEPLOY as that package has it), whose schema gives each deploy key's default and
  class; its package defaults, dotted key -> value, which no file key sets; and the SHA-256 of its code
  (image.code_sha256).
- `Change(key, old, new, applies, found_in="")`: one change. OLD is what runs now and NEW what apply makes it,
  deploy_tree.MISSING for none: an added key has no OLD, a removed one no NEW. `mark`: "+" added, "-" removed, "~"
  changed.
  - A change of the settings: KEY is a deploy key, dotted; a map's entry that is added or removed is one change, named
    by the entry (model_bindings.fast), and a section that is added or removed is one change for each key in it.
    APPLIES is its class: the x-applies of the key, or of the nearest section or map around it, in this package's
    schema (for a key it removes, in the schema of the package then); "" for a key with none (container.name).
    FOUND_IN is "".
  - A change found outside your file: KEY and FOUND_IN are the file's name (deliver.WRITES: docker/.env, the
    override or the layer), OLD its whole text as it is and NEW that text with only its lines edited by hand as apply
    writes them (`hand_edits_applied`; MISSING: no file). Each is NEW_CONTAINER: docker reads the two files only when
    it makes the container.
- `plan(applied, confirmed, package_then, package_now) -> ChangePlan(package, yours, outside, blank_defaults,
  package_defaults)`: PACKAGE_THEN is the package that made the delivered layer (a ValueError when the layer names
  another version), PACKAGE_NOW this one.
  - `package`: (the version then, this one) when PACKAGE_THEN is not this package, by its version or, under the same
    version, by its code (as after a package update to a later commit of updates.branch), else None: "This package:
    A -> B", a restart.
  - `yours`: your changes: each deploy key whose value changes because your file changed: its value as PACKAGE_THEN
    reads your file is not what runs now; for a key PACKAGE_THEN does not know, your file sets it.
  - `blank_defaults`: the other deploy keys whose value changes, which a new package changes: those you left blank
    whose default changed, and those it adds or removes.
  - `outside`: the changes outside your file, which apply puts back: each file deliver writes whose whole text is
    not what deliver writes for the layer that runs (deliver.differing), against the plan of that layer, made again
    from its own settings on HOST's ids (deploy_plan.from_layer), each with its lines edited by hand as the confirmed
    settings' plan (deploy_plan.plan) has them, so that a row holds the hand edit alone and your own changes show
    only in `yours`; a file whose hand edit apply writes as it is, as your file now makes it, is no change. None
    after a package update: the package line says that apply writes every file anew. State 9 finds a hand edit by
    another rule: the fingerprints of what deliver last wrote (deliver.changed_outside). A hand edit of the layer's
    settings changes what runs: it shows in `yours`, and outside only by the layer's fingerprint, which it leaves
    unlike its settings, when apply writes another.
  - `package_defaults`: the package defaults that changed, for your information, each NEW_CONTAINER.
  - `total`: how many changes: the package line, `yours`, `outside` (one for each file) and `blank_defaults`.
    `restarts`: whether one of them makes the container again: a new package, or a change whose class is not
    NO_RESTART.
  The settings' changes are in the schema's order, a key a change removes after the others of its section; the
  files' in the order of deliver.WRITES.
- `hand_edits_applied(found, ran, made)`: the text FOUND, a file as it is, with only its lines edited by hand, the
  runs of lines where it is not the text RAN, as the text MADE has them. A run is difflib's, taken line by line when
  it changes as many lines as it holds. A run of lines where MADE is not RAN stays as found, unless a hand edit
  overlaps it: then the two take MADE's lines together. A file added or removed, by hand or by apply (None: no
  file), is MADE whole.
- `package_in(folder) -> Package`: the package in the folder FOLDER, as `plan` reads one: its VERSION
  (image.package_version), its deploy settings' file type as its own deploy schema has it (file_types.DEPLOY's kind,
  in the layout that schema fixes), its package defaults, its own registry.PACKAGE_FILE, and its code's SHA-256
  (image.code_sha256). A ValueError when FOLDER has no such package. After `rkmon package update`, the package that
  made the delivered layer is the one it keeps beside the package folder (`<folder>.prev`).
- NO_RESTART and NEW_CONTAINER: the two classes (CLASSES).

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import pathlib
from typing import NamedTuple, Optional

from . import deliver, deploy_plan, settings_file
from .deploy_tree import MISSING, get

NO_RESTART = "no_restart"
NEW_CONTAINER = "new_container"
CLASSES = (NO_RESTART, NEW_CONTAINER)


class Applied(NamedTuple):
    """What runs now: see the module's docstring."""

    layer: str
    env: Optional[str]
    override: Optional[str]


class Confirmed(NamedTuple):
    """What apply makes: see the module's docstring."""

    doc: settings_file.Document
    host: deploy_plan.Host


class Package(NamedTuple):
    """A package as the change plan reads it: see the module's docstring."""

    version: str
    deploy: settings_file.FileType
    defaults: dict
    code: str


class Change(NamedTuple):
    """One change: see the module's docstring."""

    key: str
    old: object
    new: object
    applies: str
    found_in: str = ""

    @property
    def mark(self) -> str:
        if self.old is MISSING:
            return "+"
        return "-" if self.new is MISSING else "~"


class ChangePlan(NamedTuple):
    """What changes: see the module's docstring."""

    package: Optional[tuple]
    yours: tuple
    outside: tuple
    blank_defaults: tuple
    package_defaults: tuple

    @property
    def total(self) -> int:
        counted = (self.yours, self.outside, self.blank_defaults)
        return (self.package is not None) + sum(len(x) for x in counted)

    @property
    def restarts(self) -> bool:
        counted = (*self.yours, *self.outside, *self.blank_defaults)
        return self.package is not None or any(c.applies != NO_RESTART for c in counted)


def package_in(folder) -> Package:
    """The package in the folder FOLDER, as `plan` reads one: see the module's docstring."""
    from . import file_types, image, registry
    from .paths import PACKAGE

    folder = pathlib.Path(folder)
    deploy = file_types.DEPLOY
    layout, version = deploy.layout_in(folder), image.package_version(folder)
    if layout is None or not version:
        raise ValueError(f"{settings_file.shown(folder)} has no package")
    defaults = folder / registry.PACKAGE_FILE.relative_to(PACKAGE)
    try:
        values = json.loads(defaults.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"{settings_file.shown(defaults)} cannot be read") from None
    if not isinstance(values, dict):
        raise ValueError(f"{settings_file.shown(defaults)} has no package defaults")
    schema = folder / pathlib.Path(deploy.schema).relative_to(PACKAGE)
    try:
        code = image.code_sha256(folder)
    except OSError:
        raise ValueError(f"{settings_file.shown(folder)} has no package") from None
    file_type = settings_file.FileType(deploy.kind, layout, schema)
    return Package(version, file_type, values, code)


def _section(root: dict, dotted: str) -> bool:
    """Whether DOTTED names a section in the schema ROOT: a property of the section around it (no map's entry), with
    properties of its own."""
    around, _, name = dotted.rpartition(".")
    node = settings_file.schema_at(root, around) if around else root
    names = node.get("properties") if isinstance(node, dict) else None
    if not isinstance(names, dict) or name not in names:
        return False
    node = settings_file.schema_at(root, dotted)
    return isinstance(node, dict) and isinstance(node.get("properties"), dict)


def _walked(old: dict, new: dict, is_section, path: str = ""):
    """(key, old, new) for each value that differs between the nested mappings OLD and NEW, in NEW's order, then OLD's
    for the keys NEW lacks: a mapping both have is walked, and so is a section one of them lacks (IS_SECTION(key));
    any other value, a map's entry among them, is compared whole."""
    for key in [*new, *(k for k in old if k not in new)]:
        dotted = f"{path}.{key}" if path else str(key)
        a, b = old.get(key, MISSING), new.get(key, MISSING)
        if a is MISSING and isinstance(b, dict) and is_section(dotted):
            a = {}
        elif b is MISSING and isinstance(a, dict) and is_section(dotted):
            b = {}
        if isinstance(a, dict) and isinstance(b, dict):
            yield from _walked(a, b, is_section, dotted)
        elif a != b:
            yield dotted, a, b


def _class(root: dict, dotted: str) -> str:
    """The x-applies of the key DOTTED, or of the nearest section or map around it, in ROOT; "" for none."""
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        node = settings_file.schema_at(root, ".".join(parts[:end]))
        if isinstance(node, dict) and "x-applies" in node:
            return node["x-applies"]
    return ""


def _left_blank(values: dict, dotted: str) -> bool:
    """Whether a file's own VALUES leave the key DOTTED blank (settings_file.blank), or out."""
    value = get(values, dotted)
    return settings_file.blank(None if value is MISSING else value)


def _ordered(values, node):
    """VALUES, the delivered layer's settings, each section's keys in the order of its properties in the schema NODE,
    any other key after them as it was: the layer's file keeps them sorted."""
    if not isinstance(values, dict):
        return values
    names = node.get("properties") if isinstance(node, dict) else None
    names = names if isinstance(names, dict) else {}
    keys = [k for k in names if k in values] + [k for k in values if k not in names]
    return {k: _ordered(values[k], names.get(k)) for k in keys}


def _runs(a: list, b: list) -> list:
    """The runs of lines where B is not A, each (start, end, lines): A[start:end] is B's LINES there (difflib's, line
    by line where a run changes as many lines as it holds: no line of a run is one of the lines it replaces).
    """
    out = []
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if tag != "equal" and a2 - a1 == b2 - b1:
            out += [(i, i + 1, [b[b1 + i - a1]]) for i in range(a1, a2)]
        elif tag != "equal":
            out.append((a1, a2, b[b1:b2]))
    return out


def _overlaps(start: int, end: int, a1: int, a2: int) -> bool:
    """Whether the run A1:A2, which starts no sooner than the lines START:END, overlaps them: it starts inside them,
    or it adds lines at the very place where they add theirs."""
    return a1 < end or a1 == a2 == start == end


def hand_edits_applied(found, ran, made):
    """FOUND with only its lines edited by hand as MADE has them: see the module's docstring."""
    if None in (found, ran, made):
        return made
    ran_lines = ran.splitlines(keepends=True)
    runs = sorted(
        [(*r, True) for r in _runs(ran_lines, found.splitlines(keepends=True))]
        + [(*r, False) for r in _runs(ran_lines, made.splitlines(keepends=True))],
        key=lambda r: r[:2],
    )
    # [start, end, edited by hand, MADE's runs]: the runs that overlap, as one
    merged = []
    for a1, a2, lines, by_hand in runs:
        if not (merged and _overlaps(merged[-1][0], merged[-1][1], a1, a2)):
            merged.append([a1, a2, False, []])
        last = merged[-1]
        last[1] = max(last[1], a2)
        if by_hand:
            last[2] = True
        else:
            last[3].append((a1, a2, lines))
    # Outside the runs edited by hand, FOUND is RAN; in them, it takes MADE's lines
    out, at = [], 0
    for start, end, by_hand, made_runs in merged:
        out += ran_lines[at:start]
        at = start
        for a1, a2, lines in made_runs if by_hand else ():
            out += ran_lines[at:a1] + lines
            at = a2
        out += ran_lines[at:end]
        at = end
    return "".join(out + ran_lines[at:])


def _outside(
    applied: Applied, layer: dict, confirmed: Confirmed, package: Package
) -> list:
    """The changes outside your file, PACKAGE being this package: see `outside` in the module's docstring."""
    host = confirmed.host
    ran = deploy_plan.from_layer(layer, host.uid, host.gid)
    made = deploy_plan.plan(confirmed.doc, package.version, package.code, host)
    found = (applied.env, applied.override, applied.layer)
    olds = dict(zip(deliver.WRITES, found))
    rans = dict(zip(deliver.WRITES, deliver.texts(ran)))
    news = dict(zip(deliver.WRITES, deliver.texts(made)))
    out = []
    for name in deliver.differing(ran, *found):
        new = hand_edits_applied(olds[name], rans[name], news[name])
        if new != olds[name]:
            old, new = (MISSING if x is None else x for x in (olds[name], new))
            out.append(Change(name, old, new, NEW_CONTAINER, name))
    return out


def plan(
    applied: Applied,
    confirmed: Confirmed,
    package_then: Package,
    package_now: Package,
) -> ChangePlan:
    """What changes: see the module's docstring."""
    layer = json.loads(applied.layer)
    made_by = layer.get("package_version") if isinstance(layer, dict) else None
    if made_by != package_then.version:
        raise ValueError(
            f"the delivered layer was made by package {made_by}, not {package_then.version}"
        )
    now, then = package_now.deploy.load(), package_then.deploy.load()
    # your file as PACKAGE_THEN reads it: where that is not what runs, your file changed
    read_then = deploy_plan.settings(
        dataclasses.replace(confirmed.doc, file_type=package_then.deploy)
    )
    yours, blank = [], []
    walk = _walked(
        _ordered(layer["settings"], then),
        deploy_plan.settings(confirmed.doc),
        lambda key: _section(now, key) or _section(then, key),
    )
    for key, old, new in walk:
        applies = _class(then if new is MISSING else now, key)
        was = get(read_then, key)
        if old is MISSING and was is MISSING:  # a key PACKAGE_THEN does not know
            mine = not _left_blank(confirmed.doc.values, key)
        else:
            mine = old != was
        (yours if mine else blank).append(Change(key, old, new, applies))
    package_defaults = [
        Change(key, old, new, NEW_CONTAINER)
        for key, old, new in _walked(
            package_then.defaults, package_now.defaults, lambda key: False
        )
    ]
    package, outside = None, ()
    if (package_then.version, package_then.code) != (
        package_now.version,
        package_now.code,
    ):
        package = (package_then.version, package_now.version)
    else:
        outside = tuple(_outside(applied, layer, confirmed, package_now))
    return ChangePlan(
        package,
        tuple(yours),
        outside,
        tuple(blank),
        tuple(package_defaults),
    )
