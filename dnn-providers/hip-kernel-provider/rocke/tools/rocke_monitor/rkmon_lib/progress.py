# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where a deployment, or a build, is on its path, and the one command to run next:
what `rkmon next` prints, the deploy path's rows that `rkmon doctor` starts with, and the state's lines that `rkmon
status` starts with.

- DONE, TODO and REDO: a step's status; REDO is a step done once and out of date now. A row says UNKNOWN of what is
  not asked (this folder's container, while its name is another deployment's), `unknown (WHY)` while docker cannot
  say whose that name is, and NOTHING_RECORDED of what nothing records yet.
- `Row(step, status, found, known=True)`: one step's line, `  N. STATUS  NAME: FOUND`, the name at NAME_COLUMN and
  the found text's further lines there too, wrapped at checks.WIDTH. KNOWN is False for a row whose fact was not
  read: it holds back no later step.
- `Report(path, rows, notes, next, also="")`: a path's rows ("build" or "deploy", steps.PATHS), the lines before the
  next line (`note: …`, a state's `fix: …`), the next line (`next: …` or `done: …`), and a line after it.
- NEEDS: the steps each step needs done first. A TODO row (but an unknown one, UNKNOWN alone or with why) whose need
  is not done reads `after step N`, N the first such need: one that is REDO, or TODO and known.
- `Judged(file, exists, layout, checks, recorded)`: what `rkmon next --settings FILE` found of FILE, a deploy
  settings file: FILE as messages show it, whether it is there, its layout, the deploy check's rows (checks.Check),
  and whether it is the file this folder recorded. Steps 1 and 2 are then FILE's.
- `deploy(facts, judged=None, confirmed_once=False, logins_read=False, started="", tick=None, note="", unknown_why="",
  doctor=None)`: the deploy path's eight steps from the state table's facts (rkmon_lib/states.py), each step's row from
  its states' conditions and words: CONFIRMED_ONCE, a confirm was recorded once (a change since is REDO, else TODO; a
  confirm that is current is DONE, its row saying state 11's sentence while a change waits, else that your file and this
  package are what you confirmed: states.as_confirmed); LOGINS_READ, the logins were read (in the container, or before
  the first start in the volumes that keep them; else step 5's row is LOGINS_NOT_READ); STARTED, when the running
  container started, as messages show a time; TICK, the first tick since the start (its time; "" for none yet; None when
  unknown); DOCTOR, the last doctor result since the start (records.Doctor; None for none), step 7's row while the
  container runs, not waiting for its first start (state 8): DONE when no check failed, else TODO with how many failed
  and their names; none: TODO, NOT_SINCE_START; NOTE, what this folder is told beside its state (why docker cannot
  say whose its container name is); UNKNOWN_WHY, why docker cannot say whose this folder's container name is, in short
  ("" when it can): a recorded deployment's steps 4 to 8 then read `unknown (UNKNOWN_WHY)`, and the
  state is known only in the states the records decide alone (states.RECORDS_ONLY). Its next line: for a judged file,
  the first of steps 1 and 2 to do for FILE, and step 3 when this folder has not recorded FILE; on a folder with no
  settings file, step 1's with states.SUGGESTED_FILE, and a line that offers `next --settings FILE`; while the state is
  not known, step 2's for the recorded file, whose check says what docker answers; else the state's, as
  states.next_line gives it from what has passed since the start (since_start of DOCTOR and TICK): in standby's state
  and state 16 (states.WALKS_7_AND_8) while the container runs, step 7's until the doctor has passed since the start,
  then step 8's while no tick has run since then. Each of these lines names a deploy step's command with its step's
  number and name until deploy step 8 has passed after a start (states.step_8_passed). The notes say NOTE, then, in a
  known state, the state's sentence when no row says it (NOTED_STATES) and the state's fix.
- `since_start(doctor, tick)`: what has passed since the container's start (states.SinceStart), from DOCTOR and TICK
  as deploy takes them: the doctor has passed when no check of it failed, as step 7's row is then DONE.
- `state_lines(facts, note="", unknown_why="")`: what `rkmon status` says of the deployment's state, before
  the next line `deploy` gives: `state: SENTENCE`, wrapped at checks.WIDTH with no indent, in a state that is known
  (`state: unknown (UNKNOWN_WHY)` in any other), NOTE as a note, then the state's fix.
- `build(file, checks, image, image_here)`: the build path's five steps of FILE, a build settings file as messages
  show it: its check's rows, the image it builds (NAME:TAG, "" when its rows do not say) and whether that image is on
  this host. Steps 4 and 5 are NOTHING_RECORDED. Its next line is the first step's that is not done, for FILE.
- `render(report, next=True, notes=True)`: the title (TITLES) and the report's lines, the next line as an outcome
  prints it (outcome.next_lines) and the line after it wrapped as well; NEXT False leaves out the next line and the
  line after it, for a command that ends with its own, and NOTES False the notes, which a state's row says (rkmon
  doctor: states.row).
- `shown_time(when)`: a time as the rows show it, `2026-11-15 09:12 UTC` (TIME_FORMAT): STARTED's and TICK's.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from . import outcome, records, settings_file, states, steps, wrapping
from .checks import RAN, WIDTH, Status

DONE = "DONE"
TODO = "TODO"
REDO = "REDO"
UNKNOWN = "unknown"
NOTHING_RECORDED = "nothing recorded"
TITLES = {
    "build": "The build path (QUICK_BUILD_GUIDE.md):",
    "deploy": "The deploy path (QUICK_DEPLOY_GUIDE.md):",
}
NEEDS = {
    "build": {1: (), 2: (1,), 3: (2,), 4: (3,), 5: (3,)},
    "deploy": {
        1: (),
        2: (1,),
        3: (2,),
        4: (3,),
        5: (3, 4),
        6: (3, 4, 5),
        7: (6,),
        8: (6,),
    },
}
NAME_COLUMN = len("  1. TODO  ")
NOTE = "note: "
STATE = "state: "  # the words before the state's sentence in rkmon status (state_lines)
NOTED_STATES = (3, 9, 10, 13, 14, 15)
# what a row of a step not yet reached says, until `after step N` takes its place
NOT_YET = "not yet"
LOGINS_NOT_READ = "unknown: the logins and secrets could not be read"
NOT_SINCE_START = "not run since the start"  # step 7's row while no doctor result is recorded since the start
TIME_FORMAT = "%Y-%m-%d %H:%M UTC"
TEMPLATE, CHECK, CONFIRM, IMAGE, LOGIN, START, DOCTOR, FIRST_TICK = steps.DEPLOY_STEPS
BUILD_TEMPLATE, BUILD_CHECK, BUILD, TEST, SHIP = steps.BUILD_STEPS


@dataclass(frozen=True)
class Row:
    """One step's line."""

    step: steps.Step
    status: str
    found: str
    known: bool = True


@dataclass(frozen=True)
class Report:
    """A path's rows, the lines before the next line, the next line and a line after it."""

    path: str
    rows: tuple
    notes: tuple
    next: str
    also: str = ""


@dataclass(frozen=True)
class Judged:
    """What rkmon next --settings FILE found of FILE, a deploy settings file."""

    file: str
    exists: bool
    layout: int
    checks: tuple
    recorded: bool


def _state(n: int) -> states.State:
    state = states.TABLE[n - 1]
    assert state.n == n, (state.n, n)
    return state


def _counted(checks) -> tuple:
    """(DONE or TODO, what the check found): how many of its checks failed, or how many passed."""
    ran = [c for c in checks if c.status in RAN]
    failed = sum(1 for c in ran if c.status == Status.FAIL)
    warned = sum(1 for c in ran if c.status == Status.WARN)
    if failed:
        return TODO, f"{failed} of {wrapping.counted(len(ran), 'check')} fail"
    text = f"{wrapping.counted(len(ran), 'check')} passed"
    return DONE, text + (f" ({wrapping.counted(warned, 'warning')})" if warned else "")


def _known(state: states.State, unknown_why: str) -> bool:
    """Whether the state is known: docker can say whose the container name is (no UNKNOWN_WHY), or the records decide
    the state alone."""
    return not unknown_why or state.n in states.RECORDS_ONLY


def _unknown(why: str = "") -> str:
    """What a row says of a fact not asked: UNKNOWN, and WHY in brackets when there is one."""
    return f"{UNKNOWN} ({why})" if why else UNKNOWN


def _after(path: str, rows: list) -> tuple:
    """The rows, each TODO one (but an unknown one) whose need is not done reading `after step N`."""
    out = []
    for row in rows:
        unknown = row.found == UNKNOWN or row.found.startswith(f"{UNKNOWN} (")
        if row.status == TODO and not unknown:
            for n in NEEDS[path][row.step.n]:
                need = out[n - 1]
                if need.status == REDO or (need.status == TODO and need.known):
                    row = Row(row.step, TODO, f"after step {n}")
                    break
        out.append(row)
    return tuple(out)


def _file_rows(f: states.Facts, judged, recorded: bool, confirmed_once: bool) -> list:
    """Steps 1 to 3: the settings file, its check, and the confirm."""
    file = judged.file if judged else f.settings_file
    if judged and not judged.exists:
        one = Row(TEMPLATE, TODO, f"no file {file}")
    elif judged and judged.layout != f.package_layout:
        found = settings_file.layout_found(file, judged.layout, f.package_layout)
        one = Row(TEMPLATE, REDO, found)
    elif judged:
        one = Row(TEMPLATE, DONE, file)
    elif not f.settings_file:
        one = Row(TEMPLATE, TODO, "no deploy settings file named yet")
    elif _state(2).when(f):
        one = Row(TEMPLATE, REDO, _state(2).words(f))
    elif _state(4).when(f):
        one = Row(TEMPLATE, REDO, _state(4).words(f))
    else:
        one = Row(TEMPLATE, DONE, file)
    if judged and judged.exists:
        two = Row(CHECK, *_counted(judged.checks))
    elif recorded:
        two = Row(
            CHECK, DONE, "your settings passed their checks when they were recorded"
        )
    else:
        two = Row(CHECK, TODO, NOT_YET)
    if judged and not judged.recorded and f.settings_file:
        three = Row(CONFIRM, TODO, f"this folder's settings file is {f.settings_file}")
    elif not recorded:
        three = Row(CONFIRM, TODO, "not confirmed yet")
    elif _state(1).when(f):
        three = Row(CONFIRM, REDO, _state(1).words(f))
    elif _state(5).when(f) and confirmed_once:
        three = Row(CONFIRM, REDO, _state(5).words(f))
    elif _state(5).when(f):
        three = Row(CONFIRM, TODO, "not confirmed yet")
    elif states.current(f).n == 11:
        # the confirm is done; the change waits for apply, the state's next line
        three = Row(CONFIRM, DONE, _state(11).words(f))
    else:
        three = Row(CONFIRM, DONE, states.as_confirmed(f.settings_file, f.confirmed_at))
    return [one, two, three]


def since_start(doctor, tick) -> states.SinceStart:
    """What has passed since the container's start, from the doctor's last result since then and the last tick since
    then: see the module's docstring."""
    return states.SinceStart(doctor is not None and not doctor.failing, tick)


def _monitor_runs(f: states.Facts) -> bool:
    """The container runs, and not while it waits for its first start (its work volume not set up: state 8)."""
    return f.container == states.RUNNING and not f.not_set_up


def _doctor_row(f: states.Facts, doctor) -> Row:
    """Step 7: the last doctor result since the start, DOCTOR (records.Doctor, or None), while the monitor runs."""
    if not _monitor_runs(f):
        return Row(DOCTOR, TODO, NOT_YET)
    if doctor is None:
        return Row(DOCTOR, TODO, NOT_SINCE_START)
    at = records.parsed(doctor.at)
    when = f" at {shown_time(at)}" if at is not None else ""
    ran = wrapping.counted(doctor.checks, "check")
    if doctor.failing:
        failed = f"{len(doctor.failing)} of {ran} failed{when}"
        return Row(DOCTOR, TODO, f"{failed}: {wrapping.joined(list(doctor.failing))}")
    return Row(DOCTOR, DONE, f"{ran} passed{when}")


def _container_rows(
    f: states.Facts,
    recorded: bool,
    logins_read: bool,
    started: str,
    tick,
    unknown_why: str = "",
    doctor=None,
) -> list:
    """Steps 4 to 8: the image, the logins and this folder's container; each unknown, and why, while docker cannot
    say whose the container name is (UNKNOWN_WHY)."""
    if not recorded:
        return [Row(s, TODO, NOT_YET) for s in steps.DEPLOY_STEPS[3:]]
    if unknown_why:
        return [
            Row(s, TODO, _unknown(unknown_why), known=False)
            for s in steps.DEPLOY_STEPS[3:]
        ]
    if _state(6).when(f):
        rows = [Row(IMAGE, TODO, _state(6).words(f))]
    else:
        rows = [Row(IMAGE, DONE, f"{f.image} is on this host")]
    if _state(1).when(f):
        return rows + [
            Row(s, TODO, _unknown(), known=False) for s in steps.DEPLOY_STEPS[4:]
        ]
    if not logins_read:
        rows.append(Row(LOGIN, TODO, LOGINS_NOT_READ, known=False))
    elif _state(7).when(f):
        rows.append(Row(LOGIN, TODO, _state(7).words(f)))
    else:
        rows.append(
            Row(LOGIN, DONE, "every login and secret your settings need is set")
        )
    if _state(8).when(f):
        rows.append(Row(START, TODO, _state(8).words(f)))
    elif _state(12).when(f):
        rows.append(Row(START, REDO, _state(12).words(f)))
    else:
        rows.append(
            Row(START, DONE, f"running since {started}" if started else "running")
        )
    rows.append(_doctor_row(f, doctor))
    if not _monitor_runs(f):
        rows.append(Row(FIRST_TICK, TODO, NOT_YET))
    elif tick is None:
        rows.append(Row(FIRST_TICK, TODO, UNKNOWN, known=False))
    elif tick:
        rows.append(Row(FIRST_TICK, DONE, f"at {tick}"))
    else:
        rows.append(Row(FIRST_TICK, TODO, "none since the start"))
    return rows


def _judged_next(
    rows, judged: Judged, package_layout: int, last: int, plain: bool
) -> str:
    """The line of the first of steps 1 to LAST to do for the judged file, PLAIN without its step (steps.next_line);
    "" when they are done."""
    one = rows[0]
    if one.status == TODO:
        return steps.next_line(TEMPLATE, FILE=judged.file, plain=plain)
    if one.status == REDO:
        fix = settings_file.layout_fix(judged.file, judged.layout, package_layout)
        return f"{steps.NEXT_LABEL}{fix}"
    for step in (CHECK, CONFIRM)[: last - 1]:
        if rows[step.n - 1].status != DONE:
            return steps.next_line(step, FILE=judged.file, plain=plain)
    return ""


def deploy(
    facts: states.Facts,
    judged: Judged | None = None,
    confirmed_once: bool = False,
    logins_read: bool = False,
    started: str = "",
    tick: str | None = None,
    note: str = "",
    unknown_why: str = "",
    doctor=None,
) -> Report:
    """The deploy path's report: see the module's docstring."""
    state = states.current(facts)
    known = _known(state, unknown_why)
    ours = (
        judged is None or judged.recorded
    )  # the report is of this folder's settings file
    recorded = bool(facts.settings_file) and facts.file_exists and ours
    rows = _after(
        "deploy",
        _file_rows(facts, judged, recorded, confirmed_once)
        + _container_rows(
            facts, recorded, logins_read, started, tick, unknown_why, doctor
        ),
    )
    nxt, also = "", ""
    since = since_start(doctor, tick)
    plain = states.step_8_passed(facts, since)
    if judged:
        last = 2 if recorded else 3
        nxt = _judged_next(rows, judged, facts.package_layout, last, plain)
    if nxt:
        pass
    elif state.n == 2:
        nxt = steps.next_line(TEMPLATE, FILE=states.SUGGESTED_FILE, plain=plain)
        also = f"(have a file already? {steps.command_line('next --settings FILE')})"
    elif not known:
        nxt = steps.next_line(CHECK, FILE=facts.settings_file, plain=plain)
    else:
        nxt = states.next_line(state, facts, since)
    notes = []
    said = (
        states.sentence(state, facts)
        if ours and known and state.n in NOTED_STATES
        else ""
    )
    for text in (note, said):
        if text:
            notes += wrapping.wrap(text, WIDTH, NOTE, " " * len(NOTE))
    fix = state.fix(facts) if ours and known and state.fix else None
    if fix is not None:
        notes += outcome.fix_lines(fix)
    return Report("deploy", rows, tuple(notes), nxt, also)


def state_lines(facts: states.Facts, note: str = "", unknown_why: str = "") -> list:
    """What rkmon status says of the deployment's state: see the module's docstring."""
    state = states.current(facts)
    if _known(state, unknown_why):
        said = states.sentence(state, facts)
        fix = state.fix(facts) if state.fix else None
    else:
        said, fix = _unknown(unknown_why), None
    lines = wrapping.wrap(said, WIDTH, STATE)
    if note:
        lines += wrapping.wrap(note, WIDTH, NOTE, " " * len(NOTE))
    if fix is not None:
        lines += outcome.fix_lines(fix)
    return lines


def build(file: str, checks, image: str, image_here: bool) -> Report:
    """The build path's report: see the module's docstring."""
    status, found = _counted(checks)
    rows = [Row(BUILD_TEMPLATE, DONE, file), Row(BUILD_CHECK, status, found)]
    if status != DONE or not image:
        rows.append(Row(BUILD, TODO, NOT_YET))
    elif image_here:
        rows.append(Row(BUILD, DONE, f"{image} is on this host"))
    else:
        rows.append(Row(BUILD, TODO, f"the image {image} is not on this host"))
    rows += [Row(s, TODO, NOTHING_RECORDED, known=False) for s in (TEST, SHIP)]
    rows = _after("build", rows)
    step = next(r.step for r in rows if r.status != DONE)
    return Report("build", rows, (), steps.next_line(step, FILE=file))


def render(report: Report, next: bool = True, notes: bool = True) -> str:
    """The report as the terminal shows it: see the module's docstring."""
    lines = [TITLES[report.path]]
    for row in report.rows:
        lines += wrapping.wrap(
            f"{row.step.name}: {row.found}",
            WIDTH,
            f"  {row.step.n}. {row.status:<4}  ",
            " " * NAME_COLUMN,
        )
    if notes:
        lines += list(report.notes)
    if next:
        lines += outcome.next_lines(report.next)
        if report.also:
            lines += wrapping.wrap(report.also, WIDTH)
    return "\n".join(lines)


def shown_time(when: datetime.datetime) -> str:
    """WHEN, an aware time, as the rows show it."""
    return when.astimezone(datetime.timezone.utc).strftime(TIME_FORMAT)
