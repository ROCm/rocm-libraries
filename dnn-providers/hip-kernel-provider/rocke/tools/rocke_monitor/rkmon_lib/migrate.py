# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rkmon migrate --settings FILE [--dry-run]`: a deploy or build settings file in an earlier
layout, written again in this package's layout, in place, with a report of what moved.

- `run(argv, given, path, dry_run=False, out=None) -> int`: migrates the settings file at PATH, the absolute path of
  GIVEN, the file as you typed it in ARGV, the command line you ran (its program first, as you ran it); prints on OUT
  (stdout) and returns the exit code (rkmon_lib/outcome.py). It needs no docker and no deployment, and uses no
  network, so it runs in any state. In order:
  1. its banner, `== migrate: WHAT`, WHAT being the summary of its row in the command table (commands.TABLE);
  2. the file's own checks, as every reader of a private file makes them (`private_files.checked`), their rows
     showing the file as migrate's own lines do (`settings_file.shown`: `~/…` in your home folder);
  3. the file, read as the first of FILE_TYPES whose kind it has (`settings_file.read`, OLDER); a file in this
     layout already is NOTHING TO DO, and its next line is check's;
  4. its upgrade, through each step to this layout (`settings_file.upgrade`: the file type's `upgrades`); a value
     that its key in this layout cannot take as written (answers.Plain) is written as the text it is;
  5. the profile its title line names (`settings_file.profile_of`), and the new file's text (`settings_file.dump`,
     the one writer);
  6. the report: `FILE: KIND, from layout N to layout M`, then each note of the steps; then a note for each value
     the new file holds that its key does not take (`settings_file.misfits`), which is carried as it is for check to
     fail with its fix, and one for each other value written as text; then the profile's; one to a line;
  7. the write (`settings_file.rewrite`): the earlier file kept as FILE.v<N>, N its layout, then the new file, each
     through rkmon_lib/private_files.py, with the retry made from ARGV and GIVEN (`private_files.again`) for a
     refusal's fix that names another file. Under DRY_RUN every check runs, nothing is written, and the new file is
     printed after the report;
  8. the outcome: `MIGRATED: FILE is in layout M (the earlier file is kept as FILE.vN)`, then `next: PROGRAM check
     --settings FILE`; under DRY_RUN, PASSED, and the next line runs migrate.
  A refusal writes nothing and exits 4. A private file's refusal is printed as it shows itself (private_files.Refused:
  its rows, then the line that names the rules), then `REFUSED: FILE is not migrated; nothing changed (fix each FAIL
  above)` and `next: PROGRAM migrate --settings FILE`, which runs migrate again. A settings file's one FAIL row
  is the outcome itself: `REFUSED: FOUND; nothing changed`, its fix, then that next line; its several rows are
  printed first (checks.render), then that outcome and next line.
  A file in a newer layout ends with state 4's next line instead (`states.layout_next_line`): the package that wrote
  it, since migrate again would refuse it again. A file without a header, one in the layout before layout 1
  among them (its first key one of file_types.OLD_HEADERS), is refused with a new file of either kind (of its own
  kind when its kind line is there) as its fix (`settings_file.start_over`), and `next: PROGRAM check --settings
  NEW_FILE`, NEW_FILE being settings_file.NEW_FILE. A file of neither kind is refused with a row of its own,
  named "settings file", as the private file's rows are; so is a file without a kind line. A value the writer cannot
  write, a file that changes while migrate runs, and a read or a write that fails are ERROR (exit 1), with the next
  line that runs migrate again. The first names the value's key as the file has it and, when a step moved it, its new
  home (`settings_file.earlier_key`: "your monitor.tick_min, now monitor.tick_interval_minutes.every"), with the fix
  that corrects that value in FILE, after a report of the steps' notes alone, since no value is written.
- `before_layout_1(first_key, where, name=NAME)`: the refusal of a file in the layout before layout 1, whose first
  key, FIRST_KEY, is one of file_types.OLD_HEADERS: a FAIL row named NAME that shows the file as WHERE, with a
  new file of either kind as its fix (`settings_file.start_over`); None for any other first key. Every reader of a
  settings file refuses such a file with it, under the name of its other rows: migrate, deploy_tree.read, and
  docker/rkmon's build settings.
- FILE_TYPES: the file types migrate reads, in the order it tries them: deploy settings, then build settings.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping

from . import (
    commands,
    file_types,
    outcome,
    private_files,
    settings_file,
    states,
    steps,
    wrapping,
)
from .answers import Plain
from .checks import WIDTH, Check, Status, Steps, render
from .paths import PACKAGE

FILE_TYPES = (file_types.DEPLOY, file_types.BUILD)
NAME = "settings file"  # the name of the rows that check the file


def _say(out, lines) -> None:
    out.write("".join(f"{line}\n" for line in lines))
    out.flush()


def _read(path, where: str, retry) -> settings_file.Document:
    """The file as a document of the first of FILE_TYPES whose kind it has. A file without a header is refused with a
    new file as its fix (`_no_header`); any other refusal is that file type's."""
    kinds = wrapping.joined([f.noun for f in FILE_TYPES], "or")
    for file_type in FILE_TYPES:
        try:
            return settings_file.read(path, file_type, older=True)
        except settings_file.Refused as exc:
            if exc.reason == "header":
                raise _no_header(exc, where, kinds) from None
            if exc.reason != "kind":
                raise
    found = f"{where} is not a {kinds} file: its kind line names another kind"
    fix = Steps(("name your settings file instead:", retry(private_files.FILE)))
    raise settings_file.Refused([Check(NAME, Status.FAIL, found, fix)], "kind")


def _no_header(
    exc: settings_file.Refused, where: str, kinds: str
) -> settings_file.Refused:
    """EXC, the refusal of a file whose header a file type cannot read, with a new file as its fix. A file with
    a kind line but no layout keeps EXC's row, whose fix is a new file of its kind. A file without a kind line, of
    either of KINDS, is in the layout before layout 1 when it starts with that layout's first key
    (file_types.OLD_HEADERS), and the fix names a new file of each kind.
    """
    if exc.first_key is None:
        return exc
    row = before_layout_1(exc.first_key, where)
    if row is None:
        found = f"{where} has no kind line, so it is not a {kinds} file"
        row = Check(NAME, Status.FAIL, found, settings_file.start_over(*FILE_TYPES))
    return settings_file.Refused([row], "header")


def before_layout_1(first_key, where: str, name: str = NAME) -> Check | None:
    """The refusal of a file in the layout before layout 1, which this package does not read: see the module's
    docstring."""
    if first_key not in file_types.OLD_HEADERS:
        return None
    found = (
        f"{where} is in the layout before layout 1 (it starts with {first_key}), which this package does not "
        "read"
    )
    return Check(name, Status.FAIL, found, settings_file.start_over(*FILE_TYPES))


def _as_text(values: dict) -> tuple:
    """(VALUES with each plain value that its key could not take as written written as the text it is, {the path of
    each: that text})."""
    texts = {}

    def visit(value, path: str):
        if isinstance(value, Plain):
            texts[path] = str(value)
            return str(value)
        if isinstance(value, Mapping):
            return {
                k: visit(v, f"{path}.{k}" if path else str(k)) for k, v in value.items()
            }
        if isinstance(value, list):
            return [visit(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return value

    return visit(values, ""), texts


def _value_notes(doc: settings_file.Document, texts: dict) -> list:
    """A note for each value DOC holds that its key does not take, and for each other value of TEXTS, the values
    written as text (`_as_text`)."""
    rows = settings_file.misfits(doc)
    notes = [
        f"{r.name}: {r.found}; carried as it is, and check fails it with its fix"
        for r in rows
    ]
    named = {r.name for r in rows}
    return notes + [
        f"{path}: {text} is written as the text {json.dumps(text, ensure_ascii=False)}"
        for path, text in texts.items()
        if path not in named
    ]


def _report(out, head: str, notes) -> None:
    """The report: HEAD, then each note on a line of its own, wrapped at WIDTH."""
    lines = [head]
    for note in notes:
        lines += wrapping.wrap(note, WIDTH, "  ", "    ")
    _say(out, lines)


def _refused(exc, where: str, again: str, out) -> int:
    """A refusal's output from EXC, a settings_file.Refused or private_files.Refused, whose rows are FAIL rows: a
    private file's refusal as it shows itself (its rows, then the line that names the rules), then the outcome; a
    settings file's one row as the outcome itself, and its several printed, then the outcome. Its next line is AGAIN,
    but state 4's for a file in a newer layout, and check's of the new file for a file without a header.
    """
    rows = exc.rows
    if isinstance(exc, settings_file.Refused) and exc.reason == "newer":
        again = states.layout_next_line(where, *exc.layouts)
    elif isinstance(exc, settings_file.Refused) and exc.reason == "header":
        check = steps.command_line(settings_file.CHECK, FILE=settings_file.NEW_FILE)
        again = f"{steps.NEXT_LABEL}{check}"
    if isinstance(exc, settings_file.Refused) and len(rows) == 1:
        return outcome.finish(
            "REFUSED",
            f"{rows[0].found}; nothing changed",
            fix=rows[0].fix,
            next=again,
            out=out,
        )
    private = isinstance(exc, private_files.Refused)
    _say(out, [str(exc) if private else render(rows, "problems")])
    return outcome.finish(
        "REFUSED",
        f"{where} is not migrated; {outcome.FIX_EACH_FAIL}",
        next=again,
        out=out,
    )


def _error(where: str, exc: Exception, again: str, out) -> int:
    """A read or a write that failed, or a file that changed while migrate ran."""
    what = str(exc)
    if isinstance(exc, OSError) and exc.strerror:
        what = exc.strerror
        if exc.filename:
            what = f"{settings_file.shown(exc.filename)}: {what}"
    return outcome.finish(
        "ERROR", f"{where} is not migrated: {what}", next=again, out=out
    )


def run(argv, given: str, path, dry_run: bool = False, out=None) -> int:
    """Migrate the settings file at PATH: see the module's docstring."""
    out = sys.stdout if out is None else out
    row = commands.find(("migrate",), commands.TABLE)
    _say(out, [steps.command_banner(row.name, row.summary, dry_run)])
    retry = private_files.again(argv, given)
    where = settings_file.shown(path)
    command = steps.command_line(settings_file.MIGRATE, FILE=where)
    again = f"{steps.NEXT_LABEL}{command}"
    try:
        private_files.checked(path, PACKAGE, NAME, retry=retry, shown=where)
        earlier = _read(path, where, retry)
        if earlier.version == earlier.file_type.version:
            check = steps.command_line(settings_file.CHECK, FILE=where)
            return outcome.finish(
                "NOTHING TO DO",
                f"{where} is in layout {earlier.version} already",
                next=f"{steps.NEXT_LABEL}{check}",
                out=out,
            )
        doc, notes = settings_file.upgrade(earlier)
    except (private_files.Refused, settings_file.Refused) as exc:
        return _refused(exc, where, again, out)
    except (OSError, ValueError) as exc:
        return _error(where, exc, again, out)
    values, texts = _as_text(doc.values)
    profile, why = settings_file.profile_of(dataclasses.replace(doc, values=values))
    doc = dataclasses.replace(doc, values=values, profile=profile)
    head = f"{where}: {doc.file_type.noun}, from layout {earlier.version} to layout {doc.version}"
    try:
        text = settings_file.dump(doc)
    except settings_file.Unwritable as exc:
        _report(out, head, notes)
        key = settings_file.earlier_key(doc, exc.path)
        yours = f"your {key}" if key == exc.path else f"your {key}, now {exc.path}"
        fix = Steps((f"correct that value in {where}, then run", command))
        return outcome.finish(
            "ERROR",
            f"{where} is not migrated, as its new file cannot be written: {yours}: {exc.why}",
            fix=fix,
            next=again,
            out=out,
        )
    _report(out, head, [*notes, *_value_notes(doc, texts), why])
    try:
        kept = settings_file.rewrite(path, doc, retry, dry_run=dry_run)
    except (private_files.Refused, settings_file.Refused) as exc:
        return _refused(exc, where, again, out)
    except (OSError, ValueError) as exc:
        return _error(where, exc, again, out)
    kept = settings_file.shown(kept)
    if dry_run:
        _say(out, ["", text])
        return outcome.finish(
            "PASSED",
            f"migrate can write {where} in layout {doc.version} and keep the earlier file as {kept}; nothing "
            "changed",
            next=again,
            out=out,
        )
    check = steps.command_line(settings_file.CHECK, FILE=where)
    return outcome.finish(
        "MIGRATED",
        f"{where} is in layout {doc.version} (the earlier file is kept as {kept})",
        next=f"{steps.NEXT_LABEL}{check}",
        out=out,
    )
