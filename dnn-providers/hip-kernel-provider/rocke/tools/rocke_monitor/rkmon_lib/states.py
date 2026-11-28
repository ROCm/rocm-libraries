# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one state table of a deployment. It decides what `rkmon status`, `rkmon next` and `rkmon doctor`
report, which commands run, and every refusal's sentence and next line on the deploy side, so that every command
names the same next command for the same state.

- `Facts(...)`: what the table reads, which the caller gathers: this folder's records, the settings file as it is
  now, this package, docker, the delivered layer and, while the container runs, the monitor's applied record,
  standby and pause. Each field says what it holds; none has a default, so that no fact is taken as fine unread.
- `State(n, name, when, next_command, words, fix=None, step=None)`: one row: its number and name;
  its condition, `when(facts) -> bool`; its next command, `next_command(facts)`, what the state asks you to run as a
  fix (a checks.Command, or Steps of commands and the words that join them), which shows after steps.NEXT_LABEL
  (None for state 16, whose line is `done:`); its sentence, `words(facts) -> str`; the fix a refusal in it shows,
  `fix(facts)`, a checks.Command, Edit or Steps (None for none), or None; and the deploy step whose command the next
  command is (steps.DEPLOY_STEPS), or None. Each names the program as rkmon was run by (rkmon_lib/program.py). A next
  command that is a deploy step's is that step's command, typed (rkmon_lib/steps.py); state 2's starts the deploy
  path, before any state, and names step 1 itself, with its number and name until deploy step 8 has passed after a
  start (step_8_passed). Every command a next line or a sentence names is a checks.Command as text: each word quoted
  as the shell needs, a leading ~/ outside the quotes.
- TABLE: the 16 states, in order. `current(facts) -> State`: the first state whose condition is true; state
  16's always is. State 3's next command is the command that stopped part-way again, in the form `rkmon help` names
  it (commands.form_of), with what it finishes (FINISHES); for a first apply over an older package (Facts'
  first_apply), what it does in its place, which puts nothing back (FIRST_APPLY_FINISHES).
- `runs_here(command, where, n, under_way="") -> bool`: whether the command COMMAND (its words), which runs in the
  states WHERE (its row's: commands.Row.states), runs in state N: N is one of them, and in state 3 (PART_WAY) a
  command the records keep as under way (records.COMMANDS) runs only where it takes the place of UNDER_WAY, the
  command that stopped part-way (records.takes_place), so that each finishes its own. rkmon's guard asks it.
- `sentence(state, facts)`: the state's words; while standby is on, every state's say so (STANDBY), which
  state 14's words do themselves.
- `in_standby()`: those words, STANDBY with the command that ends standby. `with_standby(text, facts)`: TEXT, what a
  start or an apply says the monitor runs, then those words while FACTS say standby is on (`STARTED: the
  monitor runs your confirmed settings, in standby: it sends nothing until you run rkmon standby off`).
- `standby_fix(run=...) -> Steps`: what ends standby: `rkmon standby off`, once `rkmon status` and the dashboards
  read right; state 14's next command, and the fix of the doctor's standby row (rkmon_lib/doctor.py, standby_checks).
  RUN makes each command from its words, as typed: the program's name before them, by default.
- `rkmon standby off`'s words, the host's and rk's (rkmon_lib/standby.py): RELEASED, what its RELEASED outcome
  says; NOT_IN_STANDBY, what its NOTHING TO DO says while the monitor is not in standby; `stays_on(state, facts)`,
  its refusal in a state other than 14 while standby is on, in the place of the state's sentence (`refuse`: the
  state's fix and next line follow): STAYS_ON, then the state's own words, which say why; not its `sentence`, whose
  standby words STAYS_ON says itself.
- `SinceStart(doctor_passed, tick)`: what has passed since this folder's container started, as rkmon next reads deploy
  steps 7 and 8 (progress.since_start makes it): the doctor, with no check failing; a tick, the last
  one's time as messages show it ("" for none yet; None while the container does not run, or did not say). Deploy
  step 8 has passed once a tick has run (`ticked`). NOT_STARTED: nothing has (no container runs, or none was asked).
- `next_line(state, facts, since)`: the state's next line, the one every command on the deploy side ends with:
  `next: COMMAND`, or state 16's `done: the monitor runs your confirmed settings`; in the states of WALKS_7_AND_8
  (standby's and state 16) while the container runs, step 7's command until the doctor has passed since the start
  (SINCE, a SinceStart), then step 8's while no tick has run since the start. Until deploy step 8
  has passed after a start (step_8_passed), a command that is a deploy step's carries that step's number and name:
  `next: step 5, login: rkmon login`; after it, `next: COMMAND`. `reads_since(state, facts)`: whether that line reads
  SINCE: the state's next command is a deploy step's, or it is one of WALKS_7_AND_8 while the container runs. A
  caller asks the container what has passed only then.
- `step_8_passed(facts, since=NOT_STARTED)`: whether deploy step 8 has passed after a start, so that a next line
  names a deploy step's command without its step: this folder's records keep a first tick (FACTS'
  first_tick_recorded), or SINCE has a tick since this start, which the first command to see one records.
- `row(state, facts) -> checks.Check`: the state as a check row, ROW_NAME's, the doctor's first: PASS in state
  16, the monitor runs your confirmed settings; WARN in RUNS_CONFIRMED's others, where it runs them and, as you chose,
  sends nothing or starts no new work; else FAIL. A WARN's or a FAIL's fix is the state's own fix, else its next
  command.
- `layout_next_line(where, layout, current)`: state 4's next line for the settings file WHERE in LAYOUT, this package
  reading CURRENT, as `next_line` gives it: for a command that reads that file alone, without the deployment's
  facts (`rkmon migrate`'s refusal of a newer layout).
- `as_confirmed(file, at="")`: what a confirm that is current says, state 5's opposite: FILE, the settings file as
  messages show it, and this package are what you confirmed, at AT (a time as messages show it) when there is one.
  Confirm's NOTHING TO DO says it, and step 3's DONE row in rkmon next.
- EARLIER_STEPS: the states in which a refusal exits 3, an earlier step not done (2 and 4 to 8); in any other, 4.
- RECORDS_ONLY: the states whose conditions, next commands, words and fixes read none of what docker or the
  container answers (2 to 5), so each is known while docker cannot say whose the container name is; the state after
  them reads docker's answer.
- BEFORE_CONTAINER: the states whose conditions read nothing a container answers (1 to 6: docker's own answers, on
  the container name and the image, aside), so each is known while no container is asked, as rkmon's guard first
  asks none; every later state then reads as state 8.
- `name_taken(name)`: state 1's sentence, the container name NAME is used by another deployment on this host.
- `name_fix(name, file)`: the fix for a container name NAME that another deployment on this host uses, state 1's
  and the deploy check's: a name of its own in FILE, one to suggest (the name with -2 at its end, or -3 for
  -2), and the command that lists the names in use.
- `refuse(state, facts, need="", out=None, fix=None, *, since) -> int`: a command that does not run in this state:
  `REFUSED: SENTENCE; nothing changed` (the command's own need in place of the state's sentence, when that is what
  failed), the state's fix (FIX, the need's own fix, in its place), and its next line (next_line, with SINCE);
  returns the exit code.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from . import outcome, program, records, settings_file, steps, wrapping
from .checks import Check, Command, Described, Edit, Fix, Status, Steps
from .identity import OTHER

NO_CONTAINER = "none"
STOPPED = "stopped"
RUNNING = "running"
# how long a running container has to load the delivered layer (state 13)
LOAD_WAIT_MINUTES = 2
SUGGESTED_FILE = "~/rkmon-deploy.yaml"  # the file name state 2's next line suggests, as the guides do
EARLIER_STEPS = (2, 4, 5, 6, 7, 8)
PART_WAY = 3  # an apply, rollback or adopt stopped part-way: the state of the records' command under way (runs_here)
# what state 3's next command finishes, by the command that stopped part-way (_under_way_next)
FINISHES = {
    "apply": "(it finishes the change, or puts the previous settings back on a new failure)",
    "rollback": "(it finishes putting back the code and settings from before the last apply)",
    "adopt": "(from the same copy, it finishes putting the earlier monitor's state in place)",
}
# what it finishes for a first apply over an older package, which never puts the settings before it back
FIRST_APPLY_FINISHES = (
    "(it finishes the first apply, or, on a new failure, stops the new container and keeps the older monitor's "
    "files)"
)
# the states this folder's records, the settings file and this package decide alone: known while docker cannot say
# whose the container name is (rkmon next)
RECORDS_ONLY = (2, 3, 4, 5)
# the states whose conditions read nothing a container answers (docker's own answers, on the container name and the
# image, aside): known while no container is asked, which reads every later state as 8 (no login missing, none made)
BEFORE_CONTAINER = (1, 2, 3, 4, 5, 6)
# the states in which the monitor runs your confirmed settings: in standby, paused, and as they are (row)
RUNS_CONFIRMED = (14, 15, 16)
# the states whose next line walks deploy steps 7 and 8 while the container runs (next_line): standby's, as a start
# in standby ends with step 7, and state 16
WALKS_7_AND_8 = (14, 16)
ROW_NAME = "state"  # the name of the state's check row (row)
# what every state's sentence says while standby is on, {} the command that turns it off (_standby)
STANDBY = "in standby: it sends nothing until you run {}"
# rkmon standby off's words: its RELEASED outcome, its NOTHING TO DO, and its refusal outside state 14, which
# the state's own words follow (stays_on)
RELEASED = "the monitor sends again, as your settings allow"
NOT_IN_STANDBY = "the monitor is not in standby"
STAYS_ON = "the monitor does not run the settings you confirmed, so standby stays on"
# the deploy steps whose commands the next lines name
TEMPLATE, CHECK, CONFIRM, IMAGE, LOGIN, START, DOCTOR, FIRST_TICK = steps.DEPLOY_STEPS


@dataclass(frozen=True)
class Facts:
    """What the state table reads."""

    # identity.identity() of the confirmed container.name: NONE, OURS or OTHER
    name_owner: str
    container_name: str  # the confirmed container.name
    settings_file: str  # the confirmed settings file, as messages show it; "" when none is confirmed
    file_exists: bool  # the confirmed settings file is still there
    under_way: str  # "apply", "rollback" or "adopt" when the records say one is under way; "" when none is
    under_way_runs: bool  # that command still runs
    # the change under way, or the failed change the records keep, is a first apply over an older package
    # (records.first_apply)
    first_apply: bool
    file_layout: int  # the settings file's schema_version
    package_layout: int  # the layout this package reads
    file_changed: bool  # the settings file changed after the last confirm
    package_changed: bool  # this package changed after the last confirm
    image: str  # the image the confirmed settings name, NAME:TAG
    image_here: bool  # that image is on this host
    registry: bool  # image.registry is set
    # the logins and secrets the confirmed settings need that are missing or wrong (confirm.UNSET)
    logins_missing: tuple
    container: str  # this folder's container: NO_CONTAINER, STOPPED or RUNNING
    running_minutes: float  # how long the container has run
    # rk provision --check in the running container says its work volume is not set up: it was started some other way
    # than the start, and waits for its first start (False while that is not asked)
    not_set_up: bool
    # what put the last confirmed change back, in a sentence: "the last change you confirmed was put back by rkmon
    # rollback", or "... after a failed doctor check: model fast: no answer from the binding's base_url", or "the first
    # apply over an older package failed and stopped the new container: ..."; "" when nothing did
    put_back: str
    put_back_fix: Fix | None  # the fix for what failed, when something failed
    confirmed_fingerprint: str  # recorded at the last confirm
    confirmed_at: str  # when, as messages show a time ("2026-01-02 03:04 UTC")
    # how many changes the last confirm showed; 0 when it showed every setting (it could not read what runs)
    waiting_changes: int
    delivered_fingerprint: str  # the fingerprint the delivered layer was made from
    loaded_fingerprint: str  # the fingerprint the monitor last loaded (its applied record); "" when none
    # the files deliver writes that differ from what it last wrote (deliver.changed_outside), by deliver.WRITES' names
    hand_edited: tuple
    standby: bool
    paused: bool
    # this folder's records keep a first tick (records.FirstTick): deploy step 8 passed after a start
    first_tick_recorded: bool


@dataclass(frozen=True)
class SinceStart:
    """What has passed since this folder's container started, as rkmon next reads deploy steps 7 and 8."""

    doctor_passed: bool  # the doctor ran since the start, and no check of it failed
    # the last tick since the start, as messages show a time; "" for none yet; None while the container does not run,
    # or did not say
    tick: Optional[str]

    @property
    def ticked(self) -> bool:
        """Deploy step 8 has passed since the start: a tick has run."""
        return bool(self.tick)


# nothing has passed since the start: no container runs, or none was asked
NOT_STARTED = SinceStart(False, None)


@dataclass(frozen=True)
class State:
    """One row of the state table."""

    n: int
    name: str
    when: Callable[[Facts], bool]
    next_command: Callable[[Facts], Optional[Fix]]
    words: Callable[[Facts], str]
    fix: Optional[Callable[[Facts], Optional[Fix]]] = None
    # the deploy step whose command the next command is
    step: Optional[steps.Step] = None


def _run(command: str, **words) -> Command:
    """`PROGRAM COMMAND`, the command typed (steps.command_line), as the shell takes it."""
    return steps.command_line(command, **words)


def in_standby() -> str:
    """What every state's sentence says while standby is on: see the module's docstring."""
    return STANDBY.format(_run("standby off"))


def with_standby(text: str, facts: Facts) -> str:
    """TEXT, then standby's words while FACTS say standby is on: see the module's docstring."""
    return f"{text}, {in_standby()}" if facts.standby else text


def standby_fix(run: Callable[[str], Command] = _run) -> Steps:
    """What ends standby: see the module's docstring."""
    return Steps(
        (
            run("standby off"),
            ", once",
            run("status"),
            "and the dashboards read right",
        )
    )


def stays_on(state: State, facts: Facts) -> str:
    """rkmon standby off's refusal in STATE, one other than 14, while FACTS say standby is on: see the module's
    docstring."""
    return f"{STAYS_ON}; {state.words(facts)}"


def _another_name(name: str) -> str:
    """A name of its own to suggest: rocke-monitor-2 for rocke-monitor, -3 for -2."""
    m = re.fullmatch(r"(.*)-(\d+)", name)
    return f"{m.group(1)}-{int(m.group(2)) + 1}" if m else f"{name}-2"


def name_taken(name: str) -> str:
    """State 1's sentence: the container name NAME is another deployment's on this host."""
    return f"the container name {name} is used by another deployment on this host"


def name_fix(name: str, file: str) -> Steps:
    """The fix for a container name another deployment uses: see the module's docstring."""
    own = Described(f"a name of its own, such as {_another_name(name)}")
    return Steps(
        (
            Edit("container.name", file, own),
            "(",
            Command(("docker", "compose", "ls", "--all")),
            "lists the names in use)",
        )
    )


def _not_confirmed(f: Facts) -> str:
    which = [
        w
        for w, changed in (
            ("your settings file", f.file_changed),
            ("this package", f.package_changed),
        )
        if changed
    ]
    return f"{wrapping.joined(which)} changed after the last confirm"


def as_confirmed(file: str, at: str = "") -> str:
    """What a confirm that is current says: see the module's docstring."""
    when = f" at {at}" if at else ""
    return f"{file} and this package are what you confirmed{when}"


def _layout(f: Facts) -> str:
    return settings_file.layout_found(f.settings_file, f.file_layout, f.package_layout)


def _layout_command(where: str, layout: int, current: int) -> Fix:
    """State 4's next command for the settings file WHERE in LAYOUT, this package reading CURRENT: migrate an older
    file; a newer one needs the package that wrote it (settings_file.layout_fix)."""
    return settings_file.layout_fix(where, layout, current)


def _waits(f: Facts) -> str:
    when = f" at {f.confirmed_at}" if f.confirmed_at else ""
    them = "it" if f.waiting_changes == 1 else "them"
    has = "the monitor runs" if f.container == RUNNING else "the stopped container has"
    what = (
        wrapping.counted(f.waiting_changes, "change")
        if f.waiting_changes
        else "new settings"
    )
    return f"a change waits: you confirmed {what}{when}, and {has} the settings before {them}"


def _never_started(f: Facts) -> str:
    if f.container == NO_CONTAINER:
        return "this deployment's container does not exist yet"
    return (
        "this deployment's container runs, but its work volume is not set up (it was started some other way than "
        f"{_run(START.command)})"
    )


def _edited(f: Facts) -> str:
    files = wrapping.joined(f.hand_edited)
    differ = "differs" if len(f.hand_edited) == 1 else "differ"
    by = Command((program.name(),))
    return (
        f"changed outside your file: {files} {differ} from what {by} last wrote there"
    )


def _under_way_next(f: Facts) -> Steps:
    """State 3's next command: see the module's docstring."""
    # a call-time import: the command table is read only for this line
    from . import commands

    finishes = FIRST_APPLY_FINISHES if f.first_apply else FINISHES[f.under_way]
    return Steps((_run(commands.form_of(f.under_way)), finishes))


def runs_here(command: str, where, n: int, under_way: str = "") -> bool:
    """Whether COMMAND, which runs in the states WHERE, runs in state N: see the module's docstring."""
    if n not in where:
        return False
    return (
        n != PART_WAY
        or command not in records.COMMANDS
        or records.takes_place(command, under_way)
    )


TABLE = (
    State(
        1,
        "Another deployment's container name",
        lambda f: f.name_owner == OTHER,
        lambda f: _run(CONFIRM.command, FILE=f.settings_file),
        lambda f: name_taken(f.container_name),
        lambda f: name_fix(f.container_name, f.settings_file),
        step=CONFIRM,
    ),
    State(
        2,
        "No settings confirmed",
        lambda f: not f.settings_file or not f.file_exists,
        lambda f: Steps(
            (
                *steps.step_fix(
                    TEMPLATE, FILE=SUGGESTED_FILE, plain=step_8_passed(f)
                ).parts,
                "(with a file of your own:",
                _run(CHECK.command),
                ")",
            )
        ),
        lambda f: (
            f"the settings file you confirmed, {f.settings_file}, is gone"
            if f.settings_file
            else "no deploy settings are confirmed in this package folder"
        ),
    ),
    State(
        PART_WAY,
        "An apply, rollback or adopt stopped part-way",
        lambda f: bool(f.under_way) and not f.under_way_runs,
        _under_way_next,
        lambda f: f"{_run(f.under_way)} stopped part-way",
    ),
    State(
        4,
        "Your file is in another layout",
        lambda f: f.file_layout != f.package_layout,
        lambda f: _layout_command(f.settings_file, f.file_layout, f.package_layout),
        lambda f: _layout(f),
    ),
    State(
        5,
        "Not confirmed",
        lambda f: f.file_changed or f.package_changed,
        lambda f: _run(steps.command_name(CONFIRM.command)),
        lambda f: _not_confirmed(f),
        step=CONFIRM,
    ),
    State(
        6,
        "No image",
        lambda f: not f.image_here,
        # image pull, or image load IMAGE_FILE
        lambda f: _run(IMAGE.others[0]) if f.registry else _run(IMAGE.command),
        lambda f: f"the image {f.image} is not on this host",
        step=IMAGE,
    ),
    State(
        7,
        "Logins missing",
        lambda f: bool(f.logins_missing),
        lambda f: _run(LOGIN.command),
        lambda f: (
            f"your confirmed settings need {wrapping.joined(f.logins_missing)}, which "
            f"{'is' if len(f.logins_missing) == 1 else 'are'} missing or wrong"
        ),
        step=LOGIN,
    ),
    State(
        8,
        "Never started",
        lambda f: f.container == NO_CONTAINER
        or (f.container == RUNNING and f.not_set_up),
        lambda f: _run(START.command),
        lambda f: _never_started(f),
        step=START,
    ),
    State(
        9,
        "Changed outside your file",
        lambda f: bool(f.hand_edited),
        lambda f: Steps((_run("apply"), "(--dry-run shows what it puts back)")),
        lambda f: _edited(f),
    ),
    State(
        10,
        "Put back",
        lambda f: bool(f.put_back),
        lambda f: (
            Steps(
                (
                    _run("apply"),
                    ", once what failed is fixed (a fix in your file:",
                    _run("confirm"),
                    "first)",
                )
            )
            if f.put_back_fix is not None
            else _run("apply")
        ),
        lambda f: f.put_back,
        lambda f: f.put_back_fix,
    ),
    State(
        11,
        "A change waits",
        lambda f: f.confirmed_fingerprint != f.delivered_fingerprint,
        lambda f: _run("apply"),
        lambda f: _waits(f),
    ),
    State(
        12,
        "Stopped",
        lambda f: f.container == STOPPED,
        lambda f: _run(START.command),
        lambda f: "the container is stopped",
        step=START,
    ),
    State(
        13,
        "Not loaded",
        lambda f: f.container == RUNNING
        and f.running_minutes >= LOAD_WAIT_MINUTES
        and f.loaded_fingerprint != f.delivered_fingerprint,
        lambda f: Steps(
            (_run("apply"), "(or", _run("rollback"), ", for the version before)")
        ),
        lambda f: (
            f"the container has run for {LOAD_WAIT_MINUTES} minutes or more, and the monitor has not loaded your "
            "confirmed settings"
        ),
    ),
    State(
        14,
        "Standby",
        lambda f: f.standby,
        lambda f: standby_fix(),
        lambda f: "the monitor runs your confirmed settings, " + in_standby(),
    ),
    State(
        15,
        "Paused",
        lambda f: f.paused,
        lambda f: _run("resume"),
        lambda f: "you paused the monitor: it starts no new work",
    ),
    State(
        16,
        "Runs your confirmed settings",
        lambda f: True,
        lambda f: None,
        lambda f: "the monitor runs your confirmed settings",
    ),
)


def current(facts: Facts) -> State:
    """The deployment's state: the first, in the table's order, whose condition is true."""
    return next(s for s in TABLE if s.when(facts))


def sentence(state: State, facts: Facts) -> str:
    """The state's sentence, which says so while standby is on."""
    words = state.words(facts)
    if facts.standby and state.n != 14:
        words += "; " + in_standby()
    return words


def _walks_7_and_8(state: State, facts: Facts) -> bool:
    """A state of WALKS_7_AND_8 while the container runs: its next line names deploy step 7, then 8, until each has
    passed."""
    return state.n in WALKS_7_AND_8 and facts.container == RUNNING


def reads_since(state: State, facts: Facts) -> bool:
    """Whether the state's next line reads what has passed since the start: see the module's docstring."""
    return state.step is not None or _walks_7_and_8(state, facts)


def step_8_passed(facts: Facts, since: SinceStart = NOT_STARTED) -> bool:
    """Whether deploy step 8 has passed after a start: see the module's docstring."""
    return facts.first_tick_recorded or since.ticked


def next_line(state: State, facts: Facts, since: SinceStart) -> str:
    """The state's next line, SINCE what has passed since the start: see the module's docstring."""
    step, command = state.step, state.next_command(facts)
    if _walks_7_and_8(state, facts):
        if not since.doctor_passed:
            step, command = DOCTOR, _run(DOCTOR.command)
        elif since.tick == "":  # no tick since the start yet
            step, command = FIRST_TICK, _run(FIRST_TICK.command)
    if command is None:
        return steps.done_line(state.words(facts))
    if step is None or step_8_passed(facts, since):
        return f"{steps.NEXT_LABEL}{command}"
    return f"{steps.NEXT_LABEL}{steps.step_words(step)} {command}"


def row(state: State, facts: Facts) -> Check:
    """The state as a check row: see the module's docstring."""
    said = sentence(state, facts)
    if state.n == TABLE[-1].n:
        return Check(ROW_NAME, Status.PASS, said)
    status = Status.WARN if state.n in RUNS_CONFIRMED else Status.FAIL
    fix = state.fix(facts) if state.fix else None
    return Check(ROW_NAME, status, said, fix or state.next_command(facts))


def layout_next_line(where: str, layout: int, current: int) -> str:
    """State 4's next line for the settings file WHERE in LAYOUT, this package reading CURRENT, as `next_line` gives
    it from those facts: for a command that reads that file alone, without the deployment's facts (migrate).
    """
    return f"{steps.NEXT_LABEL}{_layout_command(where, layout, current)}"


def refuse(
    state: State,
    facts: Facts,
    need: str = "",
    out=None,
    fix: Optional[Fix] = None,
    *,
    since: SinceStart,
) -> int:
    """Print a refusal in this state and return its exit code: 3 in EARLIER_STEPS, else 4. FIX: the fix of the
    command's own NEED, in the place of the state's. SINCE: what has passed since the start (next_line).
    """
    text = need or sentence(state, facts)
    if fix is None and state.fix:
        fix = state.fix(facts)
    return outcome.finish(
        "REFUSED",
        f"{text}; nothing changed",
        next=next_line(state, facts, since),
        fix=fix,
        earlier_step=state.n in EARLIER_STEPS,
        out=out,
    )
