# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The state table (rkmon_lib/states.py): on synthetic facts, `current` gives the first state whose
condition is true; each state's next line, with a deploy step's number and name until step 8 has passed after a
start, and for good once this folder's records keep that it has, and the walk of steps 7 and 8 in standby's
state and state 16; a refusal's text and exit code, a taken container name's among them; standby, which every
state's line names; the states the records decide alone, which read nothing docker answers; and the graph test,
over the state table and the command table.
"""

import contextlib
import dataclasses
import io
import pathlib
import re
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import commands, program, states, steps  # noqa: E402
from rkmon_lib.checks import Command, Described, Edit, Status, Steps  # noqa: E402
from rkmon_lib.identity import NONE, OTHER, OURS  # noqa: E402

FILE = "~/rkmon-deploy.yaml"

# A deployment that runs its confirmed settings: state 16.
RUNS = states.Facts(
    name_owner=OURS,
    container_name="rocke-monitor",
    settings_file=FILE,
    file_exists=True,
    under_way="",
    under_way_runs=False,
    first_apply=False,
    file_layout=2,
    package_layout=2,
    file_changed=False,
    package_changed=False,
    image="rocke-monitor:2026.11.15",
    image_here=True,
    registry=False,
    logins_missing=(),
    container=states.RUNNING,
    running_minutes=90.0,
    not_set_up=False,
    put_back="",
    put_back_fix=None,
    confirmed_fingerprint="sha256:" + "1" * 64,
    confirmed_at="13:58",
    waiting_changes=2,
    delivered_fingerprint="sha256:" + "1" * 64,
    loaded_fingerprint="sha256:" + "1" * 64,
    hand_edited=(),
    standby=False,
    paused=False,
    first_tick_recorded=False,
)

# For each state, facts that make it the deployment's state.
CASES = {
    1: dict(name_owner=OTHER),
    2: dict(settings_file="", name_owner=NONE, container=states.NO_CONTAINER),
    3: dict(under_way="apply"),
    4: dict(file_layout=1),
    5: dict(file_changed=True),
    6: dict(image_here=False),
    7: dict(logins_missing=("claude", "admin_access_token")),
    8: dict(container=states.NO_CONTAINER, running_minutes=0.0, loaded_fingerprint=""),
    9: dict(hand_edited=("docker/.env",)),
    10: dict(
        put_back="the last change you confirmed was put back after a failed doctor check: model fast: no answer "
        "from the binding's base_url",
        put_back_fix=Edit(
            "model_bindings.fast.base_url",
            FILE,
            Described("an address the container reaches"),
        ),
        delivered_fingerprint="sha256:" + "0" * 64,
    ),
    11: dict(
        delivered_fingerprint="sha256:" + "2" * 64,
        loaded_fingerprint="sha256:" + "2" * 64,
    ),
    12: dict(container=states.STOPPED, running_minutes=0.0),
    13: dict(running_minutes=2.0, loaded_fingerprint="sha256:" + "0" * 64),
    14: dict(standby=True),
    15: dict(paused=True),
    16: dict(),
}


# What has passed since the container's start, once deploy step 8 has: the doctor passed, and a tick ran. Next lines
# then name a command alone. TICK: an invented tick's time, as messages show one.
TICK = "2026-11-15 09:20 UTC"
RAN = states.SinceStart(True, TICK)


def facts(**changes):
    return dataclasses.replace(RUNS, **changes)


# The graph test: what each next command does to the facts the state table reads, run with every check passing.
# A confirm records CONFIRMED; a rollback puts back the settings the last apply kept, KEPT.
CONFIRMED = "sha256:" + "c" * 64
KEPT = "sha256:" + "0" * 64


def _confirm(f):
    """confirm: your file and this package are what you confirmed. Its check of the container name passes: after state
    1's fix, the name is one of its own, which no container has yet."""
    if f.name_owner == OTHER:
        f = dataclasses.replace(
            f,
            name_owner=NONE,
            container=states.NO_CONTAINER,
            running_minutes=0.0,
            loaded_fingerprint="",
        )
    return dataclasses.replace(
        f, file_changed=False, package_changed=False, confirmed_fingerprint=CONFIRMED
    )


def _deploy_path(f):
    """The deploy path's steps 1 to 3 as one move: the template in this package's layout, your answers, check and
    confirm."""
    return _confirm(
        dataclasses.replace(
            f, settings_file=FILE, file_exists=True, file_layout=f.package_layout
        )
    )


def _migrate(f):
    """migrate: your file rewritten in this package's layout, so it changed after the last confirm."""
    return dataclasses.replace(f, file_layout=f.package_layout, file_changed=True)


def _image(f):
    """image pull, or image load: the image the confirmed settings name, on this host."""
    return dataclasses.replace(f, image_here=True)


def _login(f):
    """login: each login and secret the confirmed settings need, set."""
    return dataclasses.replace(f, logins_missing=())


def _start(f):
    """start: the container runs, and the monitor loads what is delivered; the first start delivers your confirmed
    settings."""
    delivered = (
        f.confirmed_fingerprint
        if f.container == states.NO_CONTAINER
        else f.delivered_fingerprint
    )
    return dataclasses.replace(
        f,
        container=states.RUNNING,
        running_minutes=0.0,
        delivered_fingerprint=delivered,
        loaded_fingerprint=delivered,
    )


def _apply(f):
    """apply: an apply stopped part-way finished; your confirmed settings delivered over any hand edit, and the record
    of a change put back cleared; the running container made again with them, which the monitor loads; a stopped one
    stays stopped."""
    running = f.container == states.RUNNING
    return dataclasses.replace(
        f,
        under_way="",
        hand_edited=(),
        put_back="",
        put_back_fix=None,
        delivered_fingerprint=f.confirmed_fingerprint,
        loaded_fingerprint=(
            f.confirmed_fingerprint if running else f.loaded_fingerprint
        ),
        running_minutes=0.0 if running else f.running_minutes,
    )


def _rollback(f):
    """rollback: a rollback stopped part-way finished: the settings the last apply kept put back, which the records keep
    as the change put back by rollback; the running container made again with them, which the monitor loads.
    """
    running = f.container == states.RUNNING
    return dataclasses.replace(
        f,
        under_way="",
        hand_edited=(),
        put_back=f"the last change you confirmed was put back by {program.name()} rollback",
        put_back_fix=None,
        delivered_fingerprint=KEPT,
        loaded_fingerprint=KEPT if running else f.loaded_fingerprint,
        running_minutes=0.0 if running else f.running_minutes,
    )


def _resume(f):
    """resume: the pause ends."""
    return dataclasses.replace(f, paused=False)


def _package_update(f):
    """package update, with the package that wrote your file: this folder's package reads its layout, and it changed
    after the last confirm."""
    return dataclasses.replace(f, package_layout=f.file_layout, package_changed=True)


def _standby_off(f):
    """standby off: standby ends, and the monitor sends again."""
    return dataclasses.replace(f, standby=False)


def _adopt(f):
    """adopt: an adopt stopped part-way finished, from the same copy: the earlier monitor's state in place, in
    standby."""
    return dataclasses.replace(f, under_way="", standby=True)


# Each next command's words in the command table, and what it does.
EFFECTS = {
    ("template", "deploy"): _deploy_path,
    ("confirm",): _confirm,
    ("migrate",): _migrate,
    ("image", "pull"): _image,
    ("image", "load"): _image,
    ("login",): _login,
    ("start",): _start,
    ("apply",): _apply,
    ("rollback",): _rollback,
    ("resume",): _resume,
    ("package", "update"): _package_update,
    ("standby", "off"): _standby_off,
    ("adopt",): _adopt,
}
# The facts of each state in its other forms, beyond CASES: the walks start from these too.
OTHER_FORMS = (
    (2, dict(name_owner=NONE, container=states.NO_CONTAINER, file_exists=False)),
    (3, dict(under_way="rollback")),
    (3, dict(under_way="apply", container=states.STOPPED, running_minutes=0.0)),
    (3, dict(under_way="adopt", container=states.STOPPED, running_minutes=0.0)),
    (3, dict(CASES[8], under_way="adopt")),
    (6, dict(image_here=False, registry=True)),
    (
        10,
        dict(
            put_back=f"the last change you confirmed was put back by {program.name()} rollback",
            delivered_fingerprint=KEPT,
        ),
    ),
    (10, dict(CASES[10], container=states.STOPPED, running_minutes=0.0)),
    (11, dict(CASES[11], container=states.STOPPED, running_minutes=0.0)),
    (4, dict(file_layout=3)),
)


def runs_in(row, n: int, f) -> bool:
    """The command of ROW runs in state N of the deployment in F, as the command table says: one of its states, or
    none named, in state 3 where it finishes the command that stopped part-way (states.runs_here), and what it needs
    of this folder's container.
    """
    if row.states and not states.runs_here(row.name, row.states, n, f.under_way):
        return False
    if row.container == commands.RUNS:
        return f.container == states.RUNNING
    if row.container == commands.EXISTS:
        return f.container != states.NO_CONTAINER
    return True


def first_command(fix) -> Command:
    """The command a next line names first: the fix itself, or the first command of its steps."""
    if isinstance(fix, Command):
        return fix
    return next(p for p in fix.parts if isinstance(p, Command))


def parsed(command: Command):
    """The command table's row of COMMAND, as rkmon parses it; None for a command it does not parse."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            ns = commands.table_parser().parse_args(list(command.argv[1:]))
        except SystemExit:
            return None
    return ns.row


def refused(state, f, since=RAN, **kwargs):
    out = io.StringIO()
    rc = states.refuse(state, f, out=out, since=since, **kwargs)
    return rc, out.getvalue()


class TestTable(unittest.TestCase):
    def test_the_sixteen_states_in_order(self):
        self.assertEqual(
            [(s.n, s.name) for s in states.TABLE],
            [
                (1, "Another deployment's container name"),
                (2, "No settings confirmed"),
                (3, "An apply, rollback or adopt stopped part-way"),
                (4, "Your file is in another layout"),
                (5, "Not confirmed"),
                (6, "No image"),
                (7, "Logins missing"),
                (8, "Never started"),
                (9, "Changed outside your file"),
                (10, "Put back"),
                (11, "A change waits"),
                (12, "Stopped"),
                (13, "Not loaded"),
                (14, "Standby"),
                (15, "Paused"),
                (16, "Runs your confirmed settings"),
            ],
        )

    def test_no_fact_is_taken_as_fine_unread(self):
        with self.assertRaises(TypeError):
            states.Facts()

    def test_the_states_the_records_decide_alone(self):
        """RECORDS_ONLY: each one's condition, next command, words and fix read none of the facts docker or the
        container answers, so each is known while docker cannot say whose the container name is; the state after them
        reads one."""

        class Unasked:
            """Facts whose answers from docker or the container raise when they are read."""

            ASKED = {
                "name_owner",
                "image_here",
                "logins_missing",
                "container",
                "running_minutes",
                "not_set_up",
                "loaded_fingerprint",
                "hand_edited",
                "standby",
                "paused",
            }

            def __init__(self, f):
                self.f = f

            def __getattr__(self, name):
                if name in self.ASKED:
                    raise AssertionError(f"{name} is docker's answer")
                return getattr(self.f, name)

        self.assertEqual(states.RECORDS_ONLY, (2, 3, 4, 5))
        for n in states.RECORDS_ONLY:
            with self.subTest(state=n):
                state, f = states.TABLE[n - 1], Unasked(facts(**CASES[n]))
                self.assertTrue(state.when(f))
                self.assertTrue(state.next_command(f))
                self.assertTrue(state.words(f))
                if state.fix:
                    state.fix(f)
        after = states.TABLE[max(states.RECORDS_ONLY)]
        with self.assertRaisesRegex(AssertionError, "is docker's answer"):
            after.when(Unasked(facts(**CASES[after.n])))


class TestRunsHere(unittest.TestCase):
    def test_in_state_3_each_command_the_records_keep_finishes_its_own(self):
        """runs_here: a command runs in the states its row names; in state 3, one the records keep as under way
        (records.COMMANDS) only where it takes the place of the one that stopped part-way, an apply or a rollback of
        either, an adopt of an adopt; any other command as its states say."""
        where = (3, 12)
        for command, under_way, runs in (
            ("apply", "apply", True),
            ("rollback", "apply", True),
            ("apply", "rollback", True),
            ("adopt", "adopt", True),
            ("apply", "adopt", False),
            ("rollback", "adopt", False),
            ("adopt", "apply", False),
            ("adopt", "rollback", False),
            ("remove", "adopt", True),
            ("remove", "apply", True),
        ):
            with self.subTest(command=command, under_way=under_way):
                self.assertIs(states.runs_here(command, where, 3, under_way), runs)
        self.assertTrue(states.runs_here("adopt", where, 12))
        self.assertFalse(states.runs_here("adopt", where, 8))


class TestCurrent(unittest.TestCase):
    def test_each_state_in_its_case(self):
        for n, changes in CASES.items():
            self.assertEqual(states.current(facts(**changes)).n, n, changes)

    def test_the_first_state_whose_condition_is_true(self):
        # Each case with the facts of every later case: the earlier state wins.
        for n in CASES:
            changes = {}
            for later in sorted(CASES, reverse=True):
                if later >= n:
                    changes.update(CASES[later])
            self.assertEqual(states.current(facts(**changes)).n, n, changes)

    def test_the_conditions(self):
        self.assertEqual(states.current(facts(file_exists=False)).n, 2)
        self.assertEqual(states.current(facts(under_way="rollback")).n, 3)
        self.assertEqual(
            states.current(facts(under_way="apply", under_way_runs=True)).n, 16
        )
        self.assertEqual(states.current(facts(file_layout=3)).n, 4)
        self.assertEqual(states.current(facts(package_changed=True)).n, 5)
        # A change delivered to a stopped container is not a change that waits.
        self.assertEqual(states.current(facts(container=states.STOPPED)).n, 12)
        self.assertEqual(
            states.current(
                facts(
                    container=states.STOPPED, delivered_fingerprint="sha256:" + "2" * 64
                )
            ).n,
            11,
        )
        # A hand edit comes first: while a change waits, and after a put-back.
        for n in (10, 11):
            edited = facts(**CASES[n], hand_edited=("docker/.env",))
            self.assertEqual(states.current(edited).n, 9, n)
        # A container that runs, its work volume not set up (it was started some other way): never started, and only
        # while it runs.
        self.assertEqual(states.current(facts(not_set_up=True)).n, 8)
        self.assertEqual(
            states.current(facts(not_set_up=True, container=states.STOPPED)).n, 12
        )
        # Not loaded only after two minutes running.
        self.assertEqual(
            states.current(facts(running_minutes=1.5, loaded_fingerprint="")).n, 16
        )
        self.assertEqual(
            states.current(facts(running_minutes=2.5, loaded_fingerprint="")).n, 13
        )


# Each state's sentence and next line, in its case, once deploy step 8 has passed since the start (RAN).
WORDS = {
    1: (
        "the container name rocke-monitor is used by another deployment on this host",
        "next: rkmon confirm --settings ~/rkmon-deploy.yaml",
    ),
    2: (
        "no deploy settings are confirmed in this package folder",
        "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check "
        "--settings FILE)",
    ),
    3: (
        "rkmon apply stopped part-way",
        "next: rkmon apply (it finishes the change, or puts the previous settings back on a new failure)",
    ),
    4: (
        "~/rkmon-deploy.yaml is in layout 1, older than this package's layout 2",
        "next: rkmon migrate --settings ~/rkmon-deploy.yaml",
    ),
    5: ("your settings file changed after the last confirm", "next: rkmon confirm"),
    6: (
        "the image rocke-monitor:2026.11.15 is not on this host",
        "next: rkmon image load IMAGE_FILE",
    ),
    7: (
        "your confirmed settings need claude and admin_access_token, which are missing or wrong",
        "next: rkmon login",
    ),
    8: ("this deployment's container does not exist yet", "next: rkmon start"),
    9: (
        "changed outside your file: docker/.env differs from what rkmon last wrote there",
        "next: rkmon apply (--dry-run shows what it puts back)",
    ),
    10: (
        "the last change you confirmed was put back after a failed doctor check: model fast: no answer from the "
        "binding's base_url",
        "next: rkmon apply, once what failed is fixed (a fix in your file: rkmon confirm first)",
    ),
    11: (
        "a change waits: you confirmed 2 changes at 13:58, and the monitor runs the settings before them",
        "next: rkmon apply",
    ),
    12: ("the container is stopped", "next: rkmon start"),
    13: (
        "the container has run for 2 minutes or more, and the monitor has not loaded your confirmed settings",
        "next: rkmon apply (or rkmon rollback, for the version before)",
    ),
    14: (
        "the monitor runs your confirmed settings, in standby: it sends nothing until you run rkmon standby off",
        "next: rkmon standby off, once rkmon status and the dashboards read right",
    ),
    15: ("you paused the monitor: it starts no new work", "next: rkmon resume"),
    16: (
        "the monitor runs your confirmed settings",
        "done: the monitor runs your confirmed settings",
    ),
}


class TestLines(unittest.TestCase):
    def line(self, n, since=RAN, **changes):
        f = facts(**{**CASES[n], **changes})
        state = states.current(f)
        self.assertEqual(state.n, n)
        return states.next_line(state, f, since)

    def sentence(self, n, **changes):
        f = facts(**{**CASES[n], **changes})
        state = states.current(f)
        self.assertEqual(state.n, n)
        return states.sentence(state, f)

    def test_each_states_sentence_and_next_line(self):
        self.assertEqual(sorted(WORDS), [s.n for s in states.TABLE])
        for n, words in WORDS.items():
            self.assertEqual((self.sentence(n), self.line(n)), words, n)

    def test_never_started_while_a_container_started_some_other_way_waits(self):
        f = facts(not_set_up=True)
        state = states.current(f)
        self.assertEqual(
            states.sentence(state, f),
            "this deployment's container runs, but its work volume is not set up (it was started some other way "
            "than rkmon start)",
        )
        self.assertEqual(
            states.next_line(state, f, states.NOT_STARTED),
            "next: step 6, start: rkmon start",
        )

    def test_each_command_as_the_shell_takes_it(self):
        # A word with a space is quoted, a leading ~/ left outside the quotes: the file, and the program's name. With
        # the samples' words put back, each state's sentence and next line are WORDS'.
        mine, prog = "~/my settings.yaml", "/opt/my tools/rkmon"

        def as_sampled(text):
            return text.replace(f"'{prog}'", "rkmon").replace(
                "~/'my settings.yaml'", FILE
            )

        with mock.patch.object(program, "name", return_value=prog):
            for n, words in WORDS.items():
                f = facts(**{"settings_file": mine, **CASES[n]})
                state = states.current(f)
                with self.subTest(n=n):
                    self.assertEqual(state.n, n)
                    sentence = as_sampled(states.sentence(state, f)).replace(mine, FILE)
                    line = as_sampled(states.next_line(state, f, RAN))
                    self.assertEqual((sentence, line), words)

    def test_next_lines_that_follow_the_facts(self):
        self.assertEqual(self.line(6, registry=True), "next: rkmon image pull")
        self.assertEqual(
            self.line(
                10,
                put_back="the last change you confirmed was put back by rkmon rollback",
                put_back_fix=None,
            ),
            "next: rkmon apply",
        )
        self.assertTrue(
            self.line(4, file_layout=3).startswith(
                "next: rkmon package update --from DIR"
            )
        )
        self.assertTrue(
            self.line(3, under_way="rollback").startswith("next: rkmon rollback ")
        )
        self.assertEqual(
            self.sentence(3, under_way="adopt"), "rkmon adopt stopped part-way"
        )
        self.assertEqual(
            self.line(3, under_way="adopt"),
            "next: rkmon adopt --from DIR (from the same copy, it finishes putting the earlier monitor's state in "
            "place)",
        )
        # a first apply over an older package puts nothing back: its line says what apply does in its place
        self.assertEqual(
            self.sentence(3, first_apply=True), "rkmon apply stopped part-way"
        )
        self.assertEqual(
            self.line(3, first_apply=True),
            "next: rkmon apply (it finishes the first apply, or, on a new failure, stops the new container and keeps "
            "the older monitor's files)",
        )

    def test_state_4s_next_line_for_a_file_read_alone(self):
        # A command that reads the file without the deployment's facts (migrate) names state 4's line for its layout.
        for layout, command in (
            (1, "rkmon migrate --settings ~/rkmon-deploy.yaml"),
            (
                3,
                "rkmon package update --from DIR, with DIR the package that wrote your file, or a newer one",
            ),
        ):
            with self.subTest(layout=layout):
                line = states.layout_next_line(FILE, layout, 2)
                self.assertEqual(line, f"next: {command}")
                self.assertEqual(line, self.line(4, file_layout=layout))

    def test_a_deploy_steps_command_is_the_steps(self):
        # Change a step's command, and each next line that names it follows.
        moved = {
            "TEMPLATE": ("template deploy FILE --new", (2,)),
            "CHECK": ("check --file FILE", (2,)),
            "CONFIRM": ("confirm-it --settings FILE", (1, 5)),
            "IMAGE": ("image fetch IMAGE_FILE", (6,)),
            "LOGIN": ("login --all", (7,)),
            "START": ("start --now", (8, 12)),
        }
        for name, (command, ns) in moved.items():
            step = dataclasses.replace(getattr(states, name), command=command)
            with mock.patch.object(states, name, step):
                for n in ns:
                    self.assertNotEqual(self.line(n), WORDS[n][1], (name, n))
                    self.assertIn(command.split()[0], self.line(n), (name, n))

    def test_the_status_sample(self):
        # The state's sentence as rkmon status shows it: a change waits, in standby.
        f = facts(**CASES[11], standby=True)
        state = states.current(f)
        self.assertEqual(
            states.sentence(state, f),
            "a change waits: you confirmed 2 changes at 13:58, and the monitor runs the settings before them; in "
            "standby: it sends nothing until you run rkmon standby off",
        )
        # Apply is no deploy step: its line is the same whatever has passed since the start.
        for since in (RAN, states.NOT_STARTED):
            self.assertEqual(states.next_line(state, f, since), "next: rkmon apply")

    def test_a_change_confirmed_with_every_setting_shown(self):
        # A confirm that could not read what runs showed every setting, and counted no change.
        self.assertEqual(
            self.sentence(11, waiting_changes=0),
            "a change waits: you confirmed new settings at 13:58, and the monitor runs the settings before them",
        )
        self.assertEqual(
            self.sentence(11, waiting_changes=1, container=states.STOPPED),
            "a change waits: you confirmed 1 change at 13:58, and the stopped container has the settings before it",
        )

    def test_a_start_or_an_apply_says_it_runs_in_standby(self):
        """The outcome of a start, or an apply, while the monitor runs in standby."""
        text = "the monitor runs your confirmed settings"
        self.assertEqual(states.with_standby(text, facts()), text)
        self.assertEqual(
            states.with_standby(text, facts(**CASES[14])),
            "the monitor runs your confirmed settings, in standby: it sends nothing until you run rkmon standby off",
        )

    def test_standby_is_named_once(self):
        f = facts(**CASES[14])
        sentence = states.sentence(states.current(f), f)
        self.assertEqual(sentence.count("in standby"), 1)
        for n in (2, 5, 12, 13):
            f = facts(**CASES[n], standby=True)
            self.assertIn(
                "; in standby: it sends nothing until you run rkmon standby off",
                states.sentence(states.current(f), f),
            )

    def test_one_login_or_more(self):
        self.assertEqual(
            self.sentence(7, logins_missing=("claude",)),
            "your confirmed settings need claude, which is missing or wrong",
        )
        self.assertEqual(
            self.sentence(7, logins_missing=("a", "b", "c")),
            "your confirmed settings need a, b and c, which are missing or wrong",
        )

    def test_put_back_by_a_rollback(self):
        self.assertEqual(
            self.sentence(
                10,
                put_back="the last change you confirmed was put back by rkmon rollback",
                put_back_fix=None,
            ),
            "the last change you confirmed was put back by rkmon rollback",
        )

    def test_which_changed(self):
        sentence = states.sentence(
            states.TABLE[4],
            facts(file_changed=True, package_changed=True),
        )
        self.assertEqual(
            sentence,
            "your settings file and this package changed after the last confirm",
        )
        sentence = states.sentence(states.TABLE[4], facts(package_changed=True))
        self.assertEqual(sentence, "this package changed after the last confirm")

    def test_a_confirm_that_is_current(self):
        # State 5's opposite, as confirm's NOTHING TO DO and step 3's DONE row say it.
        self.assertEqual(
            states.as_confirmed(FILE, "13:58"),
            "~/rkmon-deploy.yaml and this package are what you confirmed at 13:58",
        )
        self.assertEqual(
            states.as_confirmed(FILE),
            "~/rkmon-deploy.yaml and this package are what you confirmed",
        )


class TestStepWords(unittest.TestCase):
    """Until deploy step 8 has passed after a start, a next command that is a deploy step's carries that step's
    number and name; after it, the line is the command alone (WORDS), and for good once this folder's records keep a
    first tick. While the container runs, the lines of standby's state and state 16 walk steps 7 and 8.
    """

    # Each state's next line before step 8 has passed, with no doctor since the start: the states whose next command
    # is a deploy step's, and the walk's first step in states 14 and 16. The others' are WORDS' (state 2's names step
    # 1 itself).
    BEFORE = {
        1: "next: step 3, confirm: rkmon confirm --settings ~/rkmon-deploy.yaml",
        5: "next: step 3, confirm: rkmon confirm",
        6: "next: step 4, image: rkmon image load IMAGE_FILE",
        7: "next: step 5, login: rkmon login",
        8: "next: step 6, start: rkmon start",
        12: "next: step 6, start: rkmon start",
        14: "next: step 7, doctor: rkmon doctor",
        16: "next: step 7, doctor: rkmon doctor",
    }
    # the states whose line walks steps 7 and 8 while the container runs
    WALKERS = (14, 16)

    def line(self, n, since, **changes):
        f = facts(**{**CASES[n], **changes})
        state = states.current(f)
        self.assertEqual(state.n, n)
        return states.next_line(state, f, since)

    def test_until_step_8_has_passed_a_deploy_steps_command_carries_its_step(self):
        for since in (states.NOT_STARTED, states.SinceStart(True, "")):
            for n in CASES:
                with self.subTest(state=n, since=since):
                    want = self.BEFORE.get(n, WORDS[n][1])
                    if n in self.WALKERS and since.doctor_passed:
                        want = "next: step 8, first tick: rkmon first-tick"
                    self.assertEqual(self.line(n, since), want)
        self.assertEqual(
            self.line(6, states.NOT_STARTED, registry=True),
            "next: step 4, image: rkmon image pull",
        )

    def test_once_step_8_has_passed_each_line_is_the_command_alone(self):
        # A tick since the start: whether the doctor's last run passed or not, no line names a step.
        for since in (RAN, states.SinceStart(False, TICK)):
            for n in CASES:
                with self.subTest(state=n, since=since):
                    want = WORDS[n][1]
                    if n in self.WALKERS and not since.doctor_passed:
                        want = "next: rkmon doctor"
                    self.assertEqual(self.line(n, since), want)

    def test_once_the_records_keep_a_first_tick_no_line_names_a_step_again(self):
        # A later start walks steps 7 and 8 again in states 14 and 16, each command alone, then the state's own
        # line once a tick has run (None in WALK); state 2's names step 1's.
        plain = {
            2: "next: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check --settings "
            "FILE)"
        }
        walk = {
            states.NOT_STARTED: "next: rkmon doctor",
            states.SinceStart(False, ""): "next: rkmon doctor",
            states.SinceStart(True, ""): "next: rkmon first-tick",
            RAN: None,
        }
        for since in walk:
            for n in CASES:
                with self.subTest(state=n, since=since):
                    want = plain.get(n, WORDS[n][1])
                    if n in self.WALKERS:
                        want = walk[since] or want
                    self.assertEqual(
                        self.line(n, since, first_tick_recorded=True), want
                    )

    def test_step_8_has_passed_once_a_tick_has_run_since_a_start(self):
        for recorded, since, want in (
            (False, states.NOT_STARTED, False),
            (False, states.SinceStart(True, ""), False),
            (False, states.SinceStart(False, TICK), True),
            (True, states.NOT_STARTED, True),
            (True, states.SinceStart(True, ""), True),
        ):
            with self.subTest(recorded=recorded, since=since):
                f = facts(first_tick_recorded=recorded)
                self.assertEqual(states.step_8_passed(f, since), want)
        # Without what has passed since the start, the records alone say.
        self.assertFalse(states.step_8_passed(RUNS))
        self.assertTrue(states.step_8_passed(facts(first_tick_recorded=True)))

    def test_standby_and_state_16_walk_the_doctor_then_the_first_tick(self):
        # Step 7 until the doctor has passed since the start, then step 8 while no tick has run since then; the state's
        # own line once one has, or while the container did not say: state 16's done line, standby's standby off. A
        # start in standby ends with step 7.
        for n in self.WALKERS:
            own = WORDS[n][1]
            for (passed, tick), want in (
                ((False, None), "next: step 7, doctor: rkmon doctor"),
                ((False, ""), "next: step 7, doctor: rkmon doctor"),
                ((True, ""), "next: step 8, first tick: rkmon first-tick"),
                ((True, None), own),
                ((True, TICK), own),
                ((False, TICK), "next: rkmon doctor"),
            ):
                with self.subTest(state=n, passed=passed, tick=tick):
                    self.assertEqual(
                        self.line(n, states.SinceStart(passed, tick)), want
                    )

    def test_the_states_whose_line_reads_what_has_passed_since_the_start(self):
        # A caller asks the container what has passed only for these: a deploy step's command, or the walk of states
        # 14 and 16.
        for n, changes in CASES.items():
            f = facts(**changes)
            with self.subTest(state=n):
                self.assertEqual(
                    states.reads_since(states.current(f), f),
                    n in (1, 5, 6, 7, 8, 12, 14, 16),
                )

    def test_a_refusal_ends_with_the_states_line(self):
        f = facts(**CASES[7])
        rc, text = refused(states.current(f), f, since=states.NOT_STARTED)
        self.assertEqual(rc, 3)
        self.assertEqual(
            text.splitlines(),
            [
                "REFUSED: your confirmed settings need claude and admin_access_token, which are missing or wrong; "
                "nothing changed",
                "next: step 5, login: rkmon login",
            ],
        )


class TestRow(unittest.TestCase):
    def test_each_states_next_command_is_a_fix(self):
        # What each state asks you to run is a fix, a command or steps of them (None in state 16, whose line is done:),
        # and its next line shows it as the words after "next: " once deploy step 8 has passed.
        for n, changes in CASES.items():
            f = facts(**changes)
            state = states.current(f)
            with self.subTest(state=n):
                command = state.next_command(f)
                if n == 16:
                    self.assertIsNone(command)
                    continue
                self.assertIsInstance(command, (Command, Steps))
                self.assertEqual(states.next_line(state, f, RAN), f"next: {command}")

    def test_the_state_as_the_doctors_row(self):
        # The doctor's state row: PASS while the monitor runs your confirmed settings; WARN in standby and
        # paused, which you chose; FAIL in any other state. What it found is the state's sentence; a WARN's or a
        # FAIL's fix is the state's own fix (state 1's and 10's), else what its next line asks you to run.
        own = {
            1: states.name_fix("rocke-monitor", FILE),
            10: CASES[10]["put_back_fix"],
        }
        for n, changes in CASES.items():
            f = facts(**changes)
            state = states.current(f)
            with self.subTest(state=n):
                row = states.row(state, f)
                self.assertEqual(
                    (row.name, row.found), ("state", states.sentence(state, f))
                )
                if n == 16:
                    self.assertEqual(row.status, Status.PASS)
                    continue
                self.assertEqual(
                    row.status, Status.WARN if n in (14, 15) else Status.FAIL
                )
                self.assertEqual(row.fix, own.get(n) or state.next_command(f))
        f = facts(**CASES[15])
        self.assertEqual(str(states.row(states.current(f), f).fix), "rkmon resume")


class TestRefuse(unittest.TestCase):
    def test_the_container_name_refusal(self):
        # While the name is another deployment's, no container of this folder's can be asked what has passed since
        # its start, so confirm's command carries its step.
        f = facts(**CASES[1])
        rc, text = refused(states.current(f), f, since=states.NOT_STARTED)
        self.assertEqual(rc, 4)
        self.assertEqual(
            text,
            "REFUSED: the container name rocke-monitor is used by another deployment on this host; nothing changed\n"
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rocke-monitor-2 (docker compose "
            "ls --all\n"
            "     lists the names in use)\n"
            "next: step 3, confirm: rkmon confirm --settings ~/rkmon-deploy.yaml\n",
        )

    def test_exit_3_when_an_earlier_step_is_not_done(self):
        for n in CASES:
            f = facts(**CASES[n])
            rc, text = refused(states.current(f), f)
            self.assertEqual(rc, 3 if n in (2, 4, 5, 6, 7, 8) else 4, n)
            self.assertTrue(text.startswith("REFUSED: "), text)
            # The outcome line, on the lines it is wrapped on, ends with what changed.
            refusal = re.split(r"\n(?:fix|next|done): ", text)[0]
            self.assertTrue(
                refusal.replace("\n", " ").endswith("; nothing changed"), text
            )

    def test_a_hand_edit(self):
        # State 9 refuses with apply as its next line, exit 4: no earlier step is undone.
        f = facts(hand_edited=("docker/.env", "the settings volume's deploy.json"))
        rc, text = refused(states.current(f), f)
        self.assertEqual(rc, 4)
        self.assertEqual(
            text.splitlines(),
            [
                "REFUSED: changed outside your file: docker/.env and the settings volume's deploy.json differ from "
                "what",
                "rkmon last wrote there; nothing changed",
                "next: rkmon apply (--dry-run shows what it puts back)",
            ],
        )

    def test_the_put_back_fix(self):
        f = facts(**CASES[10])
        rc, text = refused(states.current(f), f)
        self.assertEqual(
            text.splitlines(),
            [
                "REFUSED: the last change you confirmed was put back after a failed doctor check: model fast: no "
                "answer from the",
                "binding's base_url; nothing changed",
                "fix: set model_bindings.fast.base_url in ~/rkmon-deploy.yaml to an address the container reaches",
                "next: rkmon apply, once what failed is fixed (a fix in your file: rkmon confirm first)",
            ],
        )

    def test_a_commands_own_need(self):
        f = facts(**CASES[12])
        rc, text = refused(
            states.current(f),
            f,
            need="rkmon pause needs a running container, and it is stopped",
        )
        self.assertEqual(rc, 4)
        self.assertEqual(
            text,
            "REFUSED: rkmon pause needs a running container, and it is stopped; nothing changed\nnext: rkmon start\n",
        )

    def test_a_commands_own_need_with_its_own_fix(self):
        # The need's fix in the place of the state's: state 10's own (the put back failure's) is not what it asks for.
        stop = Steps(
            (
                Command(("rkmon", "rk", "stop", "--drain")),
                ", then",
                Command(("rkmon", "stop")),
            )
        )
        for n, last in (
            (
                10,
                "next: rkmon apply, once what failed is fixed (a fix in your file: rkmon confirm first)",
            ),
            (16, "done: the monitor runs your confirmed settings"),
        ):
            f = facts(**CASES[n])
            rc, text = refused(
                states.current(f), f, need="the container is running", fix=stop
            )
            self.assertEqual(rc, 4)
            self.assertEqual(
                text.splitlines(),
                [
                    "REFUSED: the container is running; nothing changed",
                    "fix: rkmon rk stop --drain, then rkmon stop",
                    last,
                ],
            )

    def test_the_name_rkmon_was_run_by(self):
        f = facts(**CASES[14])
        with mock.patch.object(program, "name", return_value="./rkmon"):
            rc, text = refused(states.current(f), f)
        self.assertIn("until you run ./rkmon standby off", text)
        self.assertTrue(
            text.endswith(
                "next: ./rkmon standby off, once ./rkmon status and the dashboards read right\n"
            ),
            text,
        )

    def test_another_name_to_suggest(self):
        f = facts(name_owner=OTHER, container_name="rocke-monitor-2")
        self.assertIn("such as rocke-monitor-3 ", str(states.TABLE[0].fix(f)))

    def test_the_name_taken_sentence(self):
        f = facts(**CASES[1])
        self.assertEqual(
            states.sentence(states.TABLE[0], f), states.name_taken(f.container_name)
        )

    def test_the_name_fix_is_state_1s_for_any_file(self):
        f = facts(**CASES[1])
        self.assertEqual(
            states.name_fix(f.container_name, f.settings_file), states.TABLE[0].fix(f)
        )
        self.assertEqual(
            str(states.name_fix("rkmon", "your deploy settings file")),
            "set container.name in your deploy settings file to a name of its own, such as rkmon-2 (docker "
            "compose ls --all lists the names in use)",
        )


class TestGraph(unittest.TestCase):
    """The graph test, from the two tables: for each state, its next command, run with every check passing,
    runs in that state (the command table's states, and what it needs of the container), and leads to a state with a
    higher number, or to 16, so that each walk of next lines ends at 16. State 2's move is the deploy path's steps 1
    to 3 (the template, your answers, check and confirm), which leads to state 6 or later. EFFECTS says what each
    command does."""

    def move(self, f) -> tuple:
        """(the state of the deployment in F, the command its next line names first, the facts after that command)."""
        state = states.current(f)
        command = first_command(state.next_command(f))
        row = parsed(command)
        self.assertIsNotNone(row, f"state {state.n}: {command} does not parse")
        moves = [command]
        if row.words == ("template", "deploy"):
            moves += [
                steps.command_line(step.command, FILE=FILE)
                for step in steps.DEPLOY_STEPS[1:3]
            ]
        for typed in moves:
            then = parsed(typed)
            self.assertIsNotNone(then, f"state {state.n}: {typed} does not parse")
            self.assertTrue(
                runs_in(then, state.n, f), f"{typed} does not run in state {state.n}"
            )
        return state, command, EFFECTS[row.words](f)

    def walk(self, f) -> list:
        """The states a walk of next lines from the deployment in F passes, to 16."""
        seen = [states.current(f).n]
        while seen[-1] != 16:
            state, command, f = self.move(f)
            after = states.current(f).n
            self.assertGreater(
                after, state.n, f"{command} in state {state.n} leads to state {after}"
            )
            if state.n == 2:
                self.assertGreaterEqual(after, 6, "the deploy path's steps 1 to 3")
            seen.append(after)
        return seen

    def test_each_next_command_runs_and_leads_on_to_16(self):
        for n, changes in (*CASES.items(), *OTHER_FORMS):
            f = facts(**changes)
            with self.subTest(state=n, changes=changes):
                self.assertEqual(states.current(f).n, n)
                self.assertEqual(self.walk(f)[-1], 16)

    def test_the_walks(self):
        # What the walks pass, as the state table's order and EFFECTS make them.
        stopped = dict(container=states.STOPPED, running_minutes=0.0)
        for changes, walk in (
            (CASES[1], [1, 8, 16]),
            (CASES[2], [2, 8, 16]),
            (CASES[4], [4, 5, 11, 16]),
            # a new layout, with the container stopped: apply leaves it stopped (12), start runs it
            (dict(CASES[4], **stopped), [4, 5, 11, 12, 16]),
            # a settings file newer than this package: the package that wrote it first
            (dict(file_layout=3), [4, 5, 11, 16]),
            (dict(under_way="rollback"), [3, 10, 16]),
            # an adopt stopped part-way, before the first start, or with the container stopped: adopt again, from
            # the same copy, then the start, in standby, then standby off
            (dict(CASES[8], under_way="adopt"), [3, 8, 14, 16]),
            (dict(under_way="adopt", **stopped), [3, 12, 14, 16]),
            (dict(CASES[10], **stopped), [10, 12, 16]),
            (dict(CASES[11], **stopped), [11, 12, 16]),
            (CASES[13], [13, 16]),
            (CASES[14], [14, 16]),
            # in standby: apply delivers your confirmed settings, then standby off ends standby
            (dict(CASES[11], standby=True), [11, 14, 16]),
            # in standby and paused: standby off, then resume
            (dict(CASES[15], standby=True), [14, 15, 16]),
            (CASES[15], [15, 16]),
        ):
            with self.subTest(walk=walk):
                self.assertEqual(self.walk(facts(**changes)), walk)


if __name__ == "__main__":
    unittest.main()
