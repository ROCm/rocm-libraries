# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The command table's generators (rkmon_lib/commands.py): on a table made here, `rkmon help`
and the help of confirm and apply print the texts below; the parser takes each command, its arguments and options,
and --dry-run before or after it; -h, --help and --version print and exit 0; a usage error is one line and exit 2;
every text names the program as it was invoked. rkmon's own table (commands.TABLE): its help fits 80 columns, every
option says what it does and its default, every step, example and replacing command parses, and so does every command
a doc writes, a fix names or the code types for a next line: the commands' words have one home, the table.
"""

import ast
import collections
import contextlib
import io
import pathlib
import re
import shlex
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import commands, outcome, program, steps  # noqa: E402
from rkmon_lib.commands import Arg, Group, Option, Path, Row  # noqa: E402

CONFIRM = Row(
    "confirm",
    "check your deploy settings, show them, and confirm them",
    options=(
        Option(
            ("--settings",),
            "your deploy settings file (the first confirm records it)",
            metavar="FILE",
        ),
        Option(("--yes",), "answer yes (a failed check still refuses)"),
        Option(("--dry-run",), "ask nothing, record nothing"),
    ),
    details=(
        "Before the first start, it shows every setting. After it, only what changes and how each change applies: "
        "no restart, or a restart. Confirming changes nothing that runs; {program} apply does that."
    ),
    examples=(
        ("confirm --settings ~/rkmon-deploy.yaml", "the first time (deploy step 3)"),
        ("confirm", "after you edit that file"),
        ("confirm --dry-run", "the summary and checks only"),
    ),
    exit=(
        "0 confirmed, or nothing to confirm; 1 you said no; 2 usage error, or no terminal; 3 an earlier step is not "
        "done; 4 a check failed, or refused."
    ),
    group="change",
    brief="show what changes and how; confirm it",
)
APPLY = Row(
    "apply",
    "make the confirmed change on this deployment",
    options=(
        Option(
            ("--timeout-minutes",),
            "minutes the work in flight may take (default: 60)",
            metavar="N",
        ),
        Option(("--no-drain",), "stop the work in flight; it starts again later"),
        Option(("--dry-run",), "print the steps; run none of them"),
    ),
    details=(
        "It makes what {program} confirm showed: new settings, new code from a package update, or a new image. A "
        "restart first lets the work in flight finish, for at most --timeout-minutes (60 by default); --no-drain "
        "stops it instead, and it starts again later. The doctor runs before and after; a new failure puts the "
        "previous code and settings back (ROLLED BACK, exit 1). On a stopped container it puts everything in place, "
        "and the container stays stopped.\n\n"
        "A drain can take up to --timeout-minutes. On a remote host, run apply inside tmux or screen; if it is "
        "interrupted, run it again."
    ),
    examples=(
        ("apply", "make the change {program} confirm showed"),
        ("apply --dry-run", "show the steps; change nothing"),
    ),
    exit=(
        "0 applied, or nothing to apply; 1 rolled back, or an error; 2 usage error; 3 an earlier step is not done; "
        "4 refused; 130 interrupted."
    ),
    group="change",
    brief="make the confirmed change; put it back on failure",
)
ROWS = (
    Row(
        "template section",
        "one section of the deploy file in full",
        args=(Arg("KEY", optional=True),),
        group="change",
        form="template section KEY",
        brief="a section of the deploy file in full, every key with its comment (to turn on a feature)",
    ),
    CONFIRM,
    APPLY,
    Row(
        "rollback",
        "go back",
        group="change",
        brief="go back to the code and settings before the last apply",
    ),
    Row(
        "status",
        "the state",
        group="day-to-day",
        brief="the deployment's state, and the monitor's status",
    ),
    Row(
        "logs",
        "the container's log",
        options=(Option(("-f", "--follow"), "follow it"),),
        group="day-to-day",
        form="logs [-f]",
        brief="the container's log (-f: follow it)",
    ),
    Row(
        "pause",
        "pause",
        group="day-to-day",
        brief="start no new work; {program} resume starts it again",
    ),
    Row(
        "stop",
        "stop",
        group="day-to-day",
        form="stop [--drain]",
        brief="stop the container (--drain: reviews finish first)",
    ),
    Row("start", "start", group="day-to-day", brief="start it again"),
    Row(
        "login",
        "set each login and secret your settings need",
        args=(Arg("NAME", optional=True),),
        group="day-to-day",
        form="login NAME",
        brief="one login: github_read, github_write, claude, atlassian",
    ),
    Row(
        "secret",
        "one secret",
        args=(Arg("NAME"), Arg("FILE", optional=True)),
        group="day-to-day",
        brief="one secret, from FILE or typed; never shown",
    ),
    Row(
        "standby off",
        "standby off",
        group="day-to-day",
        brief="let a monitor in standby send again",
    ),
    Row("shell", "a shell", group="day-to-day", brief="a shell in the container"),
    Row(
        "rk",
        "an rk command",
        args=(Arg("ARGS", rest=True),),
        group="day-to-day",
        brief="any rk command, in the container",
    ),
    Row(
        "package update",
        "update",
        options=(
            Option(
                ("--from",), "a newer package", metavar="DIR|TARBALL", required=True
            ),
        ),
        group="other",
        brief="this package folder, from a newer package",
    ),
    Row(
        "migrate",
        "migrate",
        options=(Option(("--settings",), "the file", metavar="FILE", required=True),),
        group="other",
        brief="a settings file in an earlier layout, rewritten",
    ),
    Row(
        "adopt",
        "adopt",
        options=(
            Option(
                ("--from",),
                "a copy of an earlier monitor's /work",
                metavar="DIR",
                required=True,
            ),
            Option(
                ("--map",), "a login mapped to another", metavar="OLD=NEW", repeat=True
            ),
        ),
        group="other",
        brief="a monitor's state, from a copy of an earlier one's",
    ),
    Row(
        "web-host install",
        "web host",
        group="other",
        brief="the request service, on the team dashboard's web host",
    ),
    Row(
        "image ls", "images", group="other", brief="this package's images on this host"
    ),
    Row("image load", "load", args=(Arg("IMAGE_FILE"),)),
    Row(
        "remove",
        "remove",
        group="other",
        form="remove [--delete-data]",
        brief="remove the container (and its data: asked first)",
    ),
)
# The paths' steps, as the overview reads them (rkmon_lib/steps.py has the package's).
Step = collections.namedtuple(
    "Step", "n name command others skip check", defaults=("", "")
)
BUILD_STEPS = (
    Step(1, "template", "template build FILE", ()),
    Step(2, "check", "check --settings FILE", ()),
    Step(3, "build", "build --settings FILE", ()),
    Step(4, "test", "image test --settings FILE", ()),
    Step(
        5,
        "ship",
        "image save IMAGE_FILE --settings FILE",
        ("image push --settings FILE",),
    ),
)
DEPLOY_STEPS = (
    Step(1, "template", "template deploy [--profile PROFILE] FILE", ()),
    Step(2, "check", "check --settings FILE", ()),
    Step(3, "confirm", "confirm --settings FILE", ()),
    Step(
        4,
        "image",
        "image load IMAGE_FILE",
        ("image pull",),
        "nothing when the image is here",
        "image check",
    ),
    Step(5, "login", "login", ()),
    Step(6, "start", "start", ()),
    Step(7, "doctor", "doctor", ()),
    Step(8, "first tick", "first-tick", ()),
)
PATHS = (
    Path("build-path", "The build path, on any machine", BUILD_STEPS),
    Path(
        "deploy-path",
        "The deploy path, on the machine that runs it",
        DEPLOY_STEPS,
        last="Where you are, and the next command: {program} next",
    ),
)
GROUPS = (
    Group(
        "change",
        "Change a deployment",
        note='the steps: QUICK_USER_GUIDE.md, "Change a setting"',
    ),
    Group("day-to-day", "Day to day"),
    Group("other", "Newer versions, other monitors, and removal", topic=False),
)
ABOUT = "build the rocKE monitor's image, then deploy and run its container."
ENVIRONMENT = (
    ("CONTAINER_ENGINE=podman", "run podman instead of docker"),
    ("NO_COLOR=1", "no dim echo of the docker commands"),
)

# rkmon help, as the table made here prints it.
OVERVIEW = """\
rkmon: build the rocKE monitor's image, then deploy and run its container.

Usage:  rkmon COMMAND [OPTIONS]
        rkmon help [COMMAND | build-path | deploy-path | change | day-to-day |
                    exit-codes]

The build path, on any machine (rkmon help build-path):
  1 template    rkmon template build FILE
  2 check       rkmon check --settings FILE
  3 build       rkmon build --settings FILE
  4 test        rkmon image test --settings FILE
  5 ship        rkmon image save IMAGE_FILE --settings FILE
                or: rkmon image push --settings FILE

The deploy path, on the machine that runs it (rkmon help deploy-path):
  1 template    rkmon template deploy [--profile PROFILE] FILE
  2 check       rkmon check --settings FILE
  3 confirm     rkmon confirm --settings FILE
  4 image       rkmon image load IMAGE_FILE
                or: rkmon image pull; nothing when the image is here
                (rkmon image check checks it)
  5 login       rkmon login
  6 start       rkmon start
  7 doctor      rkmon doctor
  8 first tick  rkmon first-tick
  Where you are, and the next command: rkmon next

Change a deployment (rkmon help change; the steps: QUICK_USER_GUIDE.md,
"Change a setting"):
  rkmon template section KEY
                            a section of the deploy file in full, every key
                            with its comment (to turn on a feature)
  rkmon confirm             show what changes and how; confirm it
  rkmon apply               make the confirmed change; put it back on failure
  rkmon rollback            go back to the code and settings before the last
                            apply

Day to day (rkmon help day-to-day):
  rkmon status              the deployment's state, and the monitor's status
  rkmon logs [-f]           the container's log (-f: follow it)
  rkmon pause               start no new work; rkmon resume starts it again
  rkmon stop [--drain]      stop the container (--drain: reviews finish first)
  rkmon start               start it again
  rkmon login NAME          one login: github_read, github_write, claude,
                            atlassian
  rkmon secret NAME [FILE]  one secret, from FILE or typed; never shown
  rkmon standby off         let a monitor in standby send again
  rkmon shell               a shell in the container
  rkmon rk ARGS             any rk command, in the container

Newer versions, other monitors, and removal:
  rkmon package update --from DIR|TARBALL
                            this package folder, from a newer package
  rkmon migrate --settings FILE
                            a settings file in an earlier layout, rewritten
  rkmon adopt --from DIR    a monitor's state, from a copy of an earlier one's
  rkmon web-host install    the request service, on the team dashboard's web
                            host
  rkmon image ls            this package's images on this host
  rkmon remove [--delete-data]
                            remove the container (and its data: asked first)

Options for every command:
  --dry-run     show what the command would do; change nothing
  -h, --help    the command's help, with examples
  --version     this package's version

Environment:
  CONTAINER_ENGINE=podman   run podman instead of docker
  NO_COLOR=1                no dim echo of the docker commands

Exit codes: 0 done; 1 an error, a roll back, or you said no; 2 usage error;
3 an earlier step is not done; 4 refused or a check failed (nothing changed);
130 interrupted. More: rkmon help exit-codes.
"""

# The help of confirm and apply, as the rows made here print it.
CONFIRM_HELP = """\
rkmon confirm: check your deploy settings, show them, and confirm them.

Usage:  rkmon confirm [--settings FILE] [--yes] [--dry-run]

Before the first start, it shows every setting. After it, only what changes and
how each change applies: no restart, or a restart. Confirming changes nothing
that runs; rkmon apply does that.

Examples:
  rkmon confirm --settings ~/rkmon-deploy.yaml   the first time (deploy step 3)
  rkmon confirm                                  after you edit that file
  rkmon confirm --dry-run                        the summary and checks only

Options:
  --settings FILE  your deploy settings file (the first confirm records it)
  --yes            answer yes (a failed check still refuses)
  --dry-run        ask nothing, record nothing

Exit: 0 confirmed, or nothing to confirm; 1 you said no; 2 usage error, or no
terminal; 3 an earlier step is not done; 4 a check failed, or refused.
"""

APPLY_HELP = """\
rkmon apply: make the confirmed change on this deployment.

Usage:  rkmon apply [--timeout-minutes N] [--no-drain] [--dry-run]

It makes what rkmon confirm showed: new settings, new code from a package
update, or a new image. A restart first lets the work in flight finish, for at
most --timeout-minutes (60 by default); --no-drain stops it instead, and it
starts again later. The doctor runs before and after; a new failure puts the
previous code and settings back (ROLLED BACK, exit 1). On a stopped container
it puts everything in place, and the container stays stopped.

A drain can take up to --timeout-minutes. On a remote host, run apply inside
tmux or screen; if it is interrupted, run it again.

Examples:
  rkmon apply             make the change rkmon confirm showed
  rkmon apply --dry-run   show the steps; change nothing

Options:
  --timeout-minutes N  minutes the work in flight may take (default: 60)
  --no-drain           stop the work in flight; it starts again later
  --dry-run            print the steps; run none of them

Exit: 0 applied, or nothing to apply; 1 rolled back, or an error; 2 usage
error; 3 an earlier step is not done; 4 refused; 130 interrupted.
"""


def overview():
    return commands.overview(ROWS, PATHS, GROUPS, ABOUT, ENVIRONMENT)


def parse(*argv):
    """(exit code or None, namespace or None, stdout, stderr) of a parse."""
    out, err = io.StringIO(), io.StringIO()
    p = commands.parser(ROWS, "2026.11.15", PATHS, GROUPS, ABOUT, ENVIRONMENT)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            ns = p.parse_args(list(argv))
        except SystemExit as exc:
            return exc.code, None, out.getvalue(), err.getvalue()
    return None, ns, out.getvalue(), err.getvalue()


class TestHelp(unittest.TestCase):
    def test_rkmon_help_word_for_word(self):
        self.assertEqual(overview(), OVERVIEW)

    def test_the_help_of_confirm_and_apply_word_for_word(self):
        self.assertEqual(commands.help_text(CONFIRM), CONFIRM_HELP)
        self.assertEqual(commands.help_text(APPLY), APPLY_HELP)

    def test_every_line_stops_short_of_80_columns(self):
        for text in (OVERVIEW, CONFIRM_HELP, APPLY_HELP):
            self.assertLessEqual(
                max(len(line) for line in text.splitlines()), commands.WIDTH
            )

    def test_the_name_it_was_invoked_with(self):
        with mock.patch.object(program, "name", return_value="./rkmon"):
            text = overview()
            apply_help = commands.help_text(APPLY)
            _, _, _, err = parse("confirm", "--nope")
        self.assertIn("  1 template    ./rkmon template build FILE\n", text)
        self.assertIn("start no new work; ./rkmon resume starts it again", text)
        self.assertIn("More: ./rkmon help exit-codes.", text)
        self.assertNotIn(" rkmon ", text.replace("./rkmon", "PROGRAM"))
        self.assertIn("It makes what ./rkmon confirm showed", apply_help)
        self.assertTrue(apply_help.startswith("./rkmon apply: "), apply_help)
        self.assertTrue(err.startswith("./rkmon confirm: "), err)

    def test_the_generated_usage(self):
        adopt = commands.find(("adopt",), ROWS)
        self.assertIn(
            "Usage:  rkmon adopt --from DIR [--map OLD=NEW]\n",
            commands.help_text(adopt),
        )
        logs = commands.find(("logs",), ROWS)
        self.assertIn("Usage:  rkmon logs [--follow]\n", commands.help_text(logs))
        secret = commands.find(("secret",), ROWS)
        self.assertIn(
            "Usage:  rkmon secret NAME [FILE]\n",
            commands.help_text(secret),
        )

    def test_what_an_example_does_that_would_not_fit_beside_the_longest(self):
        """What each example does goes three columns after the longest example; where the longest of those would not
        fit there, each goes on the line after its example, at column EXAMPLE_BELOW."""
        long = "save " + "x" * 60
        for examples, shown in (
            (
                (("save y", "one"), ("save yy", "another")),
                "  rkmon save y    one\n  rkmon save yy   another\n",
            ),
            (
                ((long, "build step 5"), ("save y", "another")),
                f"  rkmon {long}\n      build step 5\n  rkmon save y\n      another\n",
            ),
        ):
            with self.subTest(examples=examples):
                row = Row("save", "save it", examples=examples, exit="0 saved.")
                text = commands.help_text(row)
                self.assertIn("\nExamples:\n" + shown, text)
                self.assertLessEqual(
                    max(len(x) for x in text.splitlines()), commands.WIDTH
                )

    def test_the_usage_on_one_line_is_the_help_s(self):
        """commands.usage_line: the Usage line's words on one line, the row's usage or the generated one, as the
        program was run by."""
        status = Row("status", "the state")
        for row, line in (
            (commands.find(("adopt",), ROWS), "rkmon adopt --from DIR [--map OLD=NEW]"),
            (APPLY, "rkmon apply [--timeout-minutes N] [--no-drain] [--dry-run]"),
            (status, "rkmon status"),
        ):
            with self.subTest(row=row.name):
                self.assertEqual(commands.usage_line(row), line)
                usage = commands.help_text(row).split("\n\n")[1]
                self.assertEqual(" ".join(usage.split()), f"Usage: {line}")
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(commands.usage_line(status), "./rkmon status")

    def test_a_command_s_form_is_the_one_the_overview_prints(self):
        """commands.form: a command as `rkmon help` names it, the row's own form, else its name, its required options
        and its arguments; the overview prints it after the program's name."""
        for words, shown in (
            (("logs",), "logs [-f]"),
            (("migrate",), "migrate --settings FILE"),
            (("adopt",), "adopt --from DIR"),
            (("secret",), "secret NAME [FILE]"),
            (("image", "load"), "image load IMAGE_FILE"),
        ):
            with self.subTest(words=words):
                self.assertEqual(commands.form(commands.find(words, ROWS)), shown)
        text = "\n" + overview()
        for row in ROWS:
            if row.group:
                self.assertIn(f"\n  rkmon {commands.form(row)}", text, row.name)

    def test_topics(self):
        self.assertEqual(
            commands.topics(PATHS, GROUPS),
            ["build-path", "deploy-path", "change", "day-to-day", "exit-codes"],
        )

    def test_each_topic_is_its_part_of_the_overview(self):
        for key in ("build-path", "deploy-path", "change", "day-to-day"):
            part = commands.section(key, ROWS, PATHS, GROUPS)
            self.assertIn("\n" + part, "\n" + OVERVIEW + "\n", key)
        with self.assertRaises(KeyError):
            commands.section("other", ROWS, PATHS, GROUPS)

    def test_the_exit_codes_are_outcomes_sentence(self):
        with mock.patch.object(outcome, "exit_summary", return_value="0 done."):
            text = overview()
        self.assertTrue(
            text.endswith("\n\nExit codes: 0 done. More: rkmon help exit-codes.\n"),
            text[-200:],
        )

    def test_exit_codes_in_full(self):
        text = commands.section("exit-codes", ROWS, PATHS, GROUPS)
        self.assertEqual(text.splitlines()[0], "Exit codes:")
        self.assertIn("  0     done, or nothing to do\n", text)
        self.assertIn("  130   interrupted\n", text)

    def test_help_for(self):
        self.assertEqual(
            commands.help_for((), ROWS, PATHS, GROUPS, ABOUT, ENVIRONMENT),
            OVERVIEW,
        )
        self.assertEqual(
            commands.help_for(("confirm",), ROWS, PATHS, GROUPS),
            CONFIRM_HELP,
        )
        self.assertTrue(
            commands.help_for(("image", "load"), ROWS).startswith(
                "rkmon image load: load.\n"
            )
        )
        self.assertTrue(
            commands.help_for(("change",), ROWS, PATHS, GROUPS).startswith("Change a ")
        )
        for unknown in (("build",), ("image", "nope")):
            with self.assertRaises(KeyError) as caught:
                commands.help_for(unknown, ROWS, PATHS, GROUPS)
            self.assertEqual(
                caught.exception.args[0],
                f': no command or topic "{" ".join(unknown)}": rkmon help lists them',
            )

    def test_help_for_a_replaced_command_names_the_one_that_replaces_it(self):
        """A command of GONE is help's usage error too, the parser's words; a command whose option went keeps its
        help."""
        gone = (
            commands.Gone("preflight", "check --settings FILE"),
            commands.Gone("confirm --check", "check --settings FILE"),
        )
        with self.assertRaises(KeyError) as caught:
            commands.help_for(("preflight",), ROWS, PATHS, GROUPS, gone=gone)
        self.assertEqual(
            caught.exception.args[0],
            ': unknown command "preflight": rkmon check --settings FILE replaces it',
        )
        self.assertEqual(
            commands.help_for(("confirm",), ROWS, PATHS, GROUPS, gone=gone),
            CONFIRM_HELP,
        )
        for g in commands.GONE:
            if not g.option:
                with self.subTest(gone=g.words), self.assertRaises(KeyError) as caught:
                    commands.table_help(g.command)
                self.assertIn(g.replacement("rkmon"), caught.exception.args[0])


class TestParser(unittest.TestCase):
    def test_a_command_its_options_and_arguments(self):
        rc, ns, _, _ = parse("confirm", "--settings", "~/rkmon-deploy.yaml", "--yes")
        self.assertIsNone(rc)
        self.assertIs(ns.row, CONFIRM)
        self.assertEqual(
            (ns.settings, ns.yes, ns.dry_run), ("~/rkmon-deploy.yaml", True, False)
        )
        _, ns, _, _ = parse("secret", "claude")
        self.assertEqual((ns.row.name, ns.name, ns.file), ("secret", "claude", None))
        _, ns, _, _ = parse("image", "load", "monitor.tar")
        self.assertEqual((ns.row.name, ns.image_file), ("image load", "monitor.tar"))

    def test_repeat_and_rest(self):
        _, ns, _, _ = parse("adopt", "--from", "d", "--map", "a=b", "--map", "c=d")
        self.assertEqual((ns.from_, ns.map), ("d", ["a=b", "c=d"]))
        _, ns, _, _ = parse("rk", "status", "--json")
        self.assertEqual(ns.args, ["status", "--json"])

    def test_dry_run_before_or_after_the_command(self):
        for argv in (
            ("--dry-run", "apply"),
            ("apply", "--dry-run"),
            ("--dry-run", "confirm"),
            ("confirm", "--dry-run"),
        ):
            rc, ns, _, _ = parse(*argv)
            self.assertIsNone(rc, argv)
            self.assertTrue(ns.dry_run, argv)
        _, ns, _, _ = parse("apply")
        self.assertFalse(ns.dry_run)

    def test_a_command_s_own_option_with_the_program_s_flag(self):
        """A row's --dry-run with a name of its own is the command's (as adopt's): after the command it sets that
        name, and dry_run stays the program's, which --dry-run before the command sets.
        """
        own = Option(("--dry-run",), "show the plan; apply nothing", name="plan")
        rows = (Row("adopt", "adopt", options=(own,)),)
        p = commands.parser(rows)
        ns = p.parse_args(["adopt", "--dry-run"])
        self.assertEqual((ns.dry_run, ns.plan), (False, True))
        ns = p.parse_args(["--dry-run", "adopt"])
        self.assertEqual((ns.dry_run, ns.plan), (True, False))
        self.assertIn(
            "  --dry-run  show the plan; apply nothing\n", commands.help_text(rows[0])
        )

    def test_no_command(self):
        rc, ns, _, _ = parse()
        self.assertIsNone(rc)
        self.assertIsNone(ns.row)
        self.assertFalse(ns.dry_run)

    def test_help_and_version_print_and_exit_0(self):
        for argv in (("confirm", "-h"), ("confirm", "--help")):
            self.assertEqual(parse(*argv)[::2], (0, CONFIRM_HELP))
        self.assertEqual(parse("-h")[::2], (0, OVERVIEW))
        self.assertEqual(parse("--version")[::2], (0, "2026.11.15\n"))
        self.assertEqual(parse("apply", "--version")[::2], (0, "2026.11.15\n"))

    def test_a_usage_error_is_one_line_and_exit_2(self):
        rc, _, out, err = parse("confirm", "--nope")
        self.assertEqual((rc, out), (2, ""))
        self.assertEqual(len(err.splitlines()), 1)
        self.assertTrue(err.startswith("rkmon confirm: "), err)
        rc, _, _, err = parse("secret")
        self.assertEqual(rc, 2)
        self.assertTrue(err.startswith("rkmon secret: "), err)
        rc, _, _, err = parse("image")
        self.assertEqual(rc, 2)
        self.assertTrue(err.startswith("rkmon image: "), err)
        rc, _, _, err = parse("nope")
        self.assertEqual(rc, 2)
        self.assertTrue(err.startswith("rkmon: "), err)

    def test_an_unknown_option_s_value_is_not_shown(self):
        """An option the command does not take is named without the value given after its =, which may be a
        secret; any other word it does not take is named as given."""
        rc, _, out, err = parse("confirm", "--token=made-up", "--nope", "extra")
        self.assertEqual(
            (rc, out, err),
            (2, "", "rkmon confirm: unrecognized arguments: --token --nope extra\n"),
        )

    def test_an_unknown_command_says_where_the_commands_are(self):
        """A command the table does not have is a usage error that says where the commands are listed, with nothing
        on stdout; a word that starts several commands, alone, names them; --help after it is still the help.
        """
        rc, _, out, err = parse("rebuild", "--settings", "f")
        self.assertEqual(
            (rc, out, err),
            (
                2,
                "",
                'rkmon: unknown command "rebuild": rkmon help lists the commands\n',
            ),
        )
        rc, _, _, err = parse("--dry-run", "image", "nothing")
        self.assertEqual(
            (rc, err),
            (
                2,
                'rkmon image: unknown command "image nothing": rkmon help lists the '
                "commands\n",
            ),
        )
        rc, _, _, err = parse("image")
        self.assertEqual(
            (rc, err),
            (2, "rkmon image: name one of its commands: image ls, image load\n"),
        )
        self.assertEqual(parse("image", "--help")[::2], (0, OVERVIEW))
        self.assertEqual(parse("image", "load", "-h")[0], 0)
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(
                parse("rebuild")[3],
                './rkmon: unknown command "rebuild": ./rkmon help lists the commands\n',
            )

    def test_no_abbreviation(self):
        self.assertEqual(parse("confirm", "--sett", "f")[0], 2)

    def test_a_replaced_command_names_the_one_that_replaces_it(self):
        settings = Option(("--settings",), "the file", metavar="FILE")
        rows = (
            Row("check", "check", options=(settings,)),
            Row("build", "build", options=(settings,)),
            Row("status", "the state"),
        )
        gone = (
            commands.Gone("preflight", "check --settings FILE"),
            commands.Gone("image import", "image load IMAGE_FILE"),
            commands.Gone("build --check", "check --settings FILE"),
            commands.Gone(
                "setup", in_words="your file and {program} status replace it"
            ),
        )
        home = Option(("--home",), "a home (default: yours)", metavar="DIR")

        def run(*argv):
            p = commands.parser(rows, gone=gone, leading=(home,))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                try:
                    p.parse_args(list(argv))
                except SystemExit as exc:
                    return exc.code, err.getvalue()
            return None, err.getvalue()

        preflight = (
            2,
            'rkmon: unknown command "preflight": rkmon check --settings FILE replaces it\n',
        )
        self.assertEqual(run("preflight", "--settings", "f"), preflight)
        self.assertEqual(run("--dry-run", "preflight"), preflight)
        self.assertEqual(
            run("image", "import", "monitor.tar"),
            (
                2,
                'rkmon: unknown command "image import": rkmon image load IMAGE_FILE replaces it\n',
            ),
        )
        option = (
            2,
            'rkmon build: unknown option "--check": rkmon check --settings FILE replaces it\n',
        )
        self.assertEqual(run("build", "--settings", "f", "--check"), option)
        self.assertEqual(run("build", "--check=yes"), option)
        self.assertEqual(run("build", "--settings", "f"), (None, ""))
        # a leading option's value is no command
        self.assertEqual(run("--home", "preflight", "status"), (None, ""))
        # where no one command replaces it, the row's own words, the program's name in them, whatever the options
        self.assertEqual(
            run("setup", "--section", "x"),
            (
                2,
                'rkmon: unknown command "setup": your file and rkmon status replace it\n',
            ),
        )
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertTrue(
                run("preflight")[1].startswith(
                    './rkmon: unknown command "preflight": ./rkmon check '
                )
            )
            self.assertEqual(
                run("setup")[1],
                './rkmon: unknown command "setup": your file and ./rkmon status replace it\n',
            )

    def test_a_replaced_command_s_words_and_option(self):
        for words, command, option in (
            ("preflight", ("preflight",), ""),
            ("image import", ("image", "import"), ""),
            ("build --check", ("build",), "--check"),
            ("swap --timeout", ("swap",), "--timeout"),
        ):
            with self.subTest(words=words):
                g = commands.Gone(words, "check --settings FILE")
                self.assertEqual((g.command, g.option), (command, option))

    def test_an_option_before_the_command(self):
        lead = Option(
            ("--image-code",),
            "before the command: rk runs the image's own code (default: the release installed over it)",
        )
        rows = (
            Row("rk", "an rk command", args=(Arg("ARGS", rest=True),)),
            Row("status", "the state"),
        )
        p = commands.parser(rows, leading=(lead,))
        ns = p.parse_args(["--image-code", "rk", "doctor"])
        self.assertEqual((ns.image_code, ns.args), (True, ["doctor"]))
        ns = p.parse_args(["rk", "doctor", "--image-code"])
        self.assertEqual((ns.image_code, ns.args), (False, ["doctor", "--image-code"]))
        self.assertIsNone(p.parse_args([]).row)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                p.parse_args(["status", "--image-code"])
        self.assertEqual(caught.exception.code, 2)
        text = commands.overview(rows, leading=(lead,))
        self.assertIn(
            "  --version     this package's version\n"
            "  --image-code  before the command: rk runs the image's own code (default: the\n"
            "                release installed over it)\n",
            text,
        )
        self.assertNotIn("--image-code", commands.overview(rows))

    def test_a_table_names_each_command_once(self):
        with self.assertRaises(ValueError):
            commands.parser((Row("a", "x"), Row("a", "y")))
        with self.assertRaises(ValueError):
            commands.parser((Row("image", "x"), Row("image load", "y")))

    def test_an_option_s_line_may_be_read_when_it_is_printed(self):
        """An option's help may be a function that returns its line: called when the help is printed, never when the
        table is made."""
        read = []

        def line():
            read.append(1)
            return "one of the kinds: a or b (default: a)"

        kind = Option(("--kind",), line, metavar="KIND")
        row = Row("make", "make one", options=(kind,))
        self.assertEqual(read, [])
        self.assertIn(
            "  --kind KIND  one of the kinds: a or b (default: a)\n",
            commands.help_text(row),
        )
        self.assertIn(
            "  --kind KIND   one of the kinds: a or b (default: a)\n",
            commands.overview((row,), leading=(kind,)),
        )

    def test_an_option_says_what_it_does(self):
        with self.assertRaises(ValueError):
            Option(("--x",), " ")
        with self.assertRaises(ValueError):
            Option(("x",), "what")


def run(*argv):
    """(exit code or None, namespace or None, stdout, stderr) of a parse by rkmon's own table."""
    out, err = io.StringIO(), io.StringIO()
    p = commands.table_parser("2026.11.15")
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            ns = p.parse_args(list(argv))
        except SystemExit as exc:
            return exc.code, None, out.getvalue(), err.getvalue()
    return None, ns, out.getvalue(), err.getvalue()


def options_part(row) -> dict:
    """{an option's flags and value word: its line, as the command's help prints it}."""
    text = commands.help_text(row)
    if "\nOptions:\n" not in text:
        return {}
    part = text.split("\nOptions:\n", 1)[1].split("\n\n", 1)[0]
    entries = []
    for line in part.splitlines():
        if line.startswith("  -"):
            entries.append(line.strip())
        else:
            entries[-1] += " " + line.strip()
    lines = {}
    for o in row.options:
        entry = next(e for e in entries if e == o.shown or e.startswith(o.shown + "  "))
        lines[o.shown] = entry[len(o.shown) :].strip()
    return lines


def profiles():
    from rkmon_lib import file_types, settings_file

    return settings_file.profiles(file_types.DEPLOY)


# A word for each placeholder of a step's command.
WORDS = {"FILE": "f.yaml", "IMAGE_FILE": "m.tgz", "TAG": "t", "N": "5"}
# The words a command of the program starts with: the table's, and those of the commands another replaced.
COMMAND_WORDS = frozenset(
    {row.name.split()[0] for row in commands.TABLE}
    | {gone.command[0] for gone in commands.GONE}
)

# --- the commands the docs write --------------------------------------------------------------------------------------

# The docs, by their paths in the package: the guides, the README, docs/, the agentic files and the setup skill.
DOCS = ("*.md", "docs/*.md", "agentic/**/*.md", ".claude/skills/**/*.md")
# A command of the program as a doc writes it: its name, ./ or docker/ before it, then a space or the end.
WRITTEN = re.compile(rf"(?:\./|docker/)?{re.escape(program.NAME)}(?= |$)")
FENCE = re.compile(r" {0,3}(?:```|~~~)")
SPAN = re.compile(r"`([^`\n]+)`")
# Where a shell line goes on past its command: a comment, a list of commands, a pipe, a redirection.
SHELL = (" #", " &&", " ||", ";", " | ", " >", " 2>", " < ")
# What the parser says of a command a doc names without the arguments it requires: given them, it must parse.
REQUIRED = re.compile(r"the following arguments are required: (.+)")
# What it says of an option a doc names without its value, or of a group of commands named alone.
NAMED_ONLY = ("expected one argument", "name one of its commands")


def written(text: str) -> list:
    """(line, text) of each code span, and each line of a fenced block (its `$ ` prompt aside), that starts with the
    program's name."""
    out, fenced = [], False
    for n, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            bodies = [re.sub(r"^\$ ", "", line.strip())]
        else:
            bodies = [m.group(1).strip() for m in SPAN.finditer(line)]
        out += [(n, body) for body in bodies if WRITTEN.match(body)]
    return out


def command_part(text: str):
    """The command of TEXT alone, up to where the shell line goes on (SHELL, outside brackets), a table cell's \\| read
    as |; None for a message (a word, then a colon) or a placeholder for any command (`rkmon <...>`).
    """
    text, depth = text.replace("\\|", "|"), 0
    for i, ch in enumerate(text):
        depth += (ch in "[(") - (ch in "])")
        if depth == 0 and text.startswith(SHELL, i):
            text = text[:i]
            break
    text = text.strip()
    if re.search(r"\w: |:$", text) or re.match(r"\S+ <", text):
        return None
    return text


def alternatives(text: str) -> list:
    """TEXT once for each alternative of each of its a|b (a | b in a group)."""
    m = re.search(r"(\S+) ?\| ?(\S+)", text)
    if not m:
        return [" ".join(text.split())]
    return [
        alt
        for word in m.groups()
        for alt in alternatives(text[: m.start()] + word + text[m.end() :])
    ]


def forms(text: str) -> list:
    """Each form of a command written as its usage, without ... (more of the same): without its optional groups
    ([...]), then with each one alone, in each of its alternatives."""
    text = re.sub(r"\.\.\.|…", " ", text)
    groups, base = [], text
    while re.search(r"\[[^\[\]]*\]", base):
        groups += re.findall(r"\[([^\[\]]*)\]", base)
        base = re.sub(r"\s*\[[^\[\]]*\]", "", base)
    return [
        alt
        for form in (base, *(f"{base} {g}" for g in groups))
        for alt in alternatives(form)
    ]


def doc_problem(form: str, words: dict) -> str:
    """The parser's error for the command FORM, its placeholders given WORDS (a word in angle brackets any word); ""
    when it parses. A doc may name a command without the arguments it requires (REQUIRED: with them, it parses), an
    option without its value, or a group of commands alone (NAMED_ONLY).
    """
    argv = [words.get(w, w) for w in shlex.split(re.sub(r"<[^<>]*>", "x", form))[1:]]
    if not argv:
        return ""
    rc, _, _, err = run(*argv)
    required = REQUIRED.search(err)
    if required:
        for name in required.group(1).split(", "):
            argv += [name.split("/")[-1], "x"] if name.startswith("-") else ["x"]
        rc, _, _, err = run(*argv)
    return "" if rc is None or any(n in err for n in NAMED_ONLY) else err.strip()


# --- the commands the code names in a fix or a next line --------------------------------------------------------------

CODE = [ROOT / "docker" / "rkmon", *sorted((ROOT / "rkmon_lib").glob("*.py"))]
# The commands the code names before the table has them: (the words, why). Each one must not
# parse: once the table has it, its entry goes.
LATER = ()
# What gives the program's name where the code names one of its commands: program on the host, doctor's wrapper_name
# in the container, docker/rkmon's own, and a function's argument that carries it.
PROGRAM_NAME = re.compile(
    r"(?:program\.name|(?:doctor\.)?wrapper_name)\(.*\)|program_name|self\.program"
)


def _program_names(tree) -> set:
    """The names the module TREE sets to the program's name (`w = wrapper_name(environ)`)."""
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and PROGRAM_NAME.fullmatch(ast.unparse(node.value))
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _module(rel: str) -> str:
    """The name the code calls the module at the path REL by: rkmon_lib/states.py is states."""
    return pathlib.PurePosixPath(rel).stem


def _constants(tree) -> dict:
    """{NAME: text} of each string constant the module TREE sets at its top level."""
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _text(node, constants: dict, module: str):
    """The text NODE gives, or None: a literal; a string constant of the module MODULE (NAME) or of another
    (other.NAME), CONSTANTS being {a module's name: its constants}."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(module, {}).get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return constants.get(node.value.id, {}).get(node.attr)
    return None


def _argvs(elts) -> list:
    """Each argv that the words ELTS, as the code writes them, give: a literal as it is (WORDS for a placeholder), a
    value for any other word, both words of `a if c else b`; up to a *starred word."""
    argvs = [[]]
    for e in elts:
        if isinstance(e, ast.Starred):
            break
        if isinstance(e, ast.IfExp) and all(
            isinstance(x, ast.Constant) for x in (e.body, e.orelse)
        ):
            argvs = [a + [str(x.value)] for a in argvs for x in (e.body, e.orelse)]
        else:
            word = str(e.value) if isinstance(e, ast.Constant) else "VALUE"
            argvs = [a + [WORDS.get(word, word)] for a in argvs]
    return argvs


def code_commands(sources: dict) -> list:
    """(file:line, argv) of each command of the program that the modules SOURCES ({path: source}) name in a fix or type
    for a next line. A checks.Command written with its words, a literal word second: the program's name first as the
    code gives it (PROGRAM_NAME, or a name the module gives it to), or any first word but a literal one (a literal name
    is test_program's) and one of the program's commands second. And a command steps types (steps.command_line and
    steps.typed, and states._run, which calls command_line) from a literal or a module's string constant.
    """
    trees = {rel: ast.parse(source) for rel, source in sources.items()}
    constants = {_module(rel): _constants(tree) for rel, tree in trees.items()}
    out = []
    for rel, tree in trees.items():
        names = _program_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            called = getattr(node.func, "attr", getattr(node.func, "id", ""))
            first, where = node.args[0], f"{rel}:{node.lineno}"
            if called == "Command" and isinstance(first, (ast.Tuple, ast.List)):
                elts = first.elts
                if (
                    len(elts) > 1
                    and not isinstance(elts[0], ast.Constant)
                    and isinstance(elts[1], ast.Constant)
                    and (
                        PROGRAM_NAME.fullmatch(ast.unparse(elts[0]))
                        or getattr(elts[0], "id", None) in names
                        or elts[1].value in COMMAND_WORDS
                    )
                ):
                    out += [(where, argv) for argv in _argvs(elts[1:])]
            elif called in ("command_line", "typed") or (
                called == "_run" and rel.endswith("states.py")
            ):
                text = _text(first, constants, _module(rel))
                if isinstance(text, str):
                    out.append((where, list(steps.typed(text, **WORDS))))
    return out


def later(argv, entries=LATER) -> bool:
    """Whether ARGV is a command ENTRIES names, LATER's by default."""
    return any(argv[: len(w.split())] == w.split() for w, _ in entries)


class TestTheTable(unittest.TestCase):
    """rkmon's own table: every help text fits 80 columns, every option says what it does and
    its default, every step and example parses, and a replaced command names one that parses; so does every command a
    doc writes, a fix names or the code types for a next line, but those LATER names.
    """

    def test_every_help_text_stops_short_of_80_columns(self):
        texts = [("the overview", commands.table_help())]
        for topic in commands.topics(commands.PATHS, commands.GROUPS):
            texts.append((topic, commands.table_help((topic,))))
        texts += [(row.name, commands.help_text(row)) for row in commands.TABLE]
        for name in ("rkmon", "./rkmon", "../../package/docker/rkmon"):
            with mock.patch.object(program, "name", return_value=name):
                texts.append(("the overview as " + name, commands.table_help()))
        for what, text in texts:
            for line in text.splitlines():
                self.assertLessEqual(len(line), commands.WIDTH, (what, line))

    def test_apply_s_help_with_the_package_s_drain_time(self):
        """With the package's default drain time (commands.drain_timeout_minutes), which confirm's summary names."""
        self.assertEqual(commands.drain_timeout_minutes(), 60)
        apply = commands.find(("apply",), commands.TABLE)
        self.assertEqual(commands.help_text(apply), APPLY_HELP)

    def test_each_path_s_help_is_part_of_rkmon_help(self):
        for topic in ("build-path", "deploy-path"):
            part = commands.table_help((topic,))
            self.assertIn("\n" + part, "\n" + OVERVIEW + "\n", topic)

    def test_every_option_says_what_it_does_and_its_default(self):
        for row in commands.TABLE:
            for shown, line in options_part(row).items():
                self.assertTrue(line, (row.name, shown))
                o = next(o for o in row.options if o.shown == shown)
                if o.metavar and not o.required:
                    self.assertIn("(default:", line, (row.name, shown))

    def test_the_profile_line_names_each_profile_and_the_first_as_the_default(self):
        line = options_part(commands.find(("template", "deploy"), commands.TABLE))[
            "--profile PROFILE"
        ]
        found = profiles()
        self.assertIn(f"(default: {found[0].name})", line)
        for p in found:
            self.assertIn(f"{p.name} for {p.what}", line)

    def test_each_command_has_examples_and_an_exit_line(self):
        for row in commands.TABLE:
            self.assertTrue(row.examples, row.name)
            self.assertTrue(row.exit.endswith("."), row.name)

    def test_stop_s_exit_line_names_each_way_it_exits_1(self):
        """rkmon stop exits 1 (ERROR) when docker cannot stop the container, when rk drain or rk stop fails in it,
        when the work in flight does not finish in time, and when another command lifts the drain (docker/rkmon
        stop_drained, drain): its exit line names each."""
        exit_ = commands.find(("stop",), commands.TABLE).exit
        one = exit_.split("; 1 ")[1].split("; 2 ")[0]
        for cause in (
            "docker could not stop it",
            "rk drain",
            "rk stop",
            "did not finish in time",
            "another command lifted the drain",
        ):
            self.assertIn(cause, one)

    def test_a_usage_written_out_names_each_option_of_its_command(self):
        """A row's own usage, for one the generated usage does not say right, names each option the command takes;
        the generated one names them in the table's order: rkmon build's ends with --dry-run and --no-cache,
        and its help says how the dry run's docker build differs from the build's."""
        for row in commands.TABLE:
            for o in row.options if row.usage else ():
                flag = next((f for f in o.flags if f.startswith("--")), o.flags[0])
                self.assertIn(flag, row.usage, row.name)
        build = commands.find(("build",), commands.TABLE)
        line = commands.usage_line(build)
        self.assertTrue(line.startswith("rkmon build --settings FILE "), line)
        self.assertTrue(line.endswith(" [--dry-run] [--no-cache]"), line)
        self.assertIn(
            "it prints the docker build it would run, the same as the build's but for the label that says when it "
            "was built.",
            " ".join(commands.help_text(build).split()),
        )

    def test_image_load_s_example_loads_the_file_image_save_s_writes(self):
        """image save's example writes the image file, ~/rocke-monitor.tar.gz, and image load's example loads it."""
        save = commands.find(("image", "save"), commands.TABLE)
        load = commands.find(("image", "load"), commands.TABLE)
        saved = shlex.split(save.examples[0][0])[2]
        self.assertEqual(saved, "~/rocke-monitor.tar.gz")
        self.assertEqual(shlex.split(load.examples[0][0])[2], saved)

    def test_image_test_s_help_lists_every_stage_of_the_test(self):
        """`image test --help` lists the stages of the image's test, the build guide's list of them, as the test runs
        them (steps.IMAGE_TEST_STAGES): each in order, with what it checks in a clause of its own.
        """
        row = commands.find(("image", "test"), commands.TABLE)
        listed = "; ".join(
            f"{name} ({what})" for name, what in steps.IMAGE_TEST_STAGES.items()
        )
        self.assertIn(
            f"Its stages, in order: {listed}.",
            " ".join(commands.help_text(row).split()),
        )
        for name, what in steps.IMAGE_TEST_STAGES.items():
            with self.subTest(stage=name):
                self.assertEqual(set("();") & set(what), set(), what)

    def test_every_example_parses_as_its_command(self):
        names = [p.name for p in profiles()]
        for row in commands.TABLE:
            for words, _ in row.examples:
                argv = shlex.split(words)
                rc, ns, _, err = run(*argv)
                self.assertIsNone(rc, (words, err))
                self.assertIs(ns.row, row, words)
                if "--profile" in argv:
                    self.assertIn(argv[argv.index("--profile") + 1], names, words)

    def test_every_step_command_parses(self):
        name = profiles()[-1].name
        for step in steps.BUILD_STEPS + steps.DEPLOY_STEPS:
            for command in (step.command, *step.others):
                for argv in (
                    steps.typed(command, **WORDS),
                    steps.typed(command, PROFILE=name, **WORDS),
                ):
                    rc, ns, _, err = run(*argv)
                    self.assertIsNone(rc, (argv, err))
                    self.assertEqual(ns.row.name, steps.command_name(command), argv)

    def test_a_path_step_s_summary_is_its_step_s(self):
        # A command of two steps (check), of a step of several commands (image save, image load), or that does more
        # than its step (login NAME) has a summary of its own.
        own = {"check", "image save", "image load", "login"}
        for step in steps.BUILD_STEPS + steps.DEPLOY_STEPS:
            name = steps.command_name(step.command)
            row = commands.find(name.split(), commands.TABLE)
            if name not in own:
                self.assertEqual(row.summary, step.what, step.command)

    def test_each_topic_and_each_command_has_its_help(self):
        for topic in commands.topics(commands.PATHS, commands.GROUPS):
            self.assertTrue(commands.table_help((topic,)))
        for row in commands.TABLE:
            self.assertEqual(commands.table_help(row.words), commands.help_text(row))
            self.assertEqual(
                run(*row.words, "--help")[::2], (0, commands.help_text(row))
            )
        for words in (("nothing",), ("image", "nothing"), ("other",)):
            with self.assertRaises(KeyError):
                commands.table_help(words)
        self.assertEqual(run("--help")[::2], (0, commands.table_help()))
        self.assertEqual(run("--version")[::2], (0, "2026.11.15\n"))

    def test_a_replaced_command_names_one_that_parses(self):
        for g in commands.GONE:
            rc, _, _, err = run(*g.words.split())
            self.assertEqual(rc, 2, g.words)
            self.assertIn(f": {g.replacement('rkmon')}\n", err)
            if g.in_words:
                continue  # its own words name what replaces it: test_setup_names_what_replaces_it
            argv = steps.typed(g.instead, **WORDS)
            rc, ns, _, err = run(*argv)
            self.assertIsNone(rc, (argv, err))
            self.assertEqual(ns.row.name, steps.command_name(g.instead), argv)

    def test_setup_names_what_replaces_it(self):
        """setup, with any of its options, is an unknown command whose usage error names your deploy
        settings file, rkmon login and rkmon secret, the deploy path's help, and, with Claude Code, /rkmon-setup; each
        command it names is the table's, and help says the same."""
        said = (
            'rkmon: unknown command "setup": set up a deployment with your deploy settings file, rkmon login and '
            "rkmon secret, as rkmon help deploy-path shows; with Claude Code, /rkmon-setup helps you write the file\n"
        )
        for argv in (
            ("setup",),
            ("--dry-run", "setup"),
            ("setup", "--section", "gh"),
            ("setup", "--settings", "f.yaml"),
            ("setup", "--overwrite"),
        ):
            with self.subTest(argv=argv):
                rc, _, out, err = run(*argv)
                self.assertEqual((rc, out, err), (2, "", said))
        for name in ("login", "secret", "help"):
            self.assertIsNotNone(commands.find((name,), commands.TABLE), name)
        self.assertIn("deploy-path", commands.topics(commands.PATHS, commands.GROUPS))
        with self.assertRaises(KeyError) as caught:
            commands.table_help(("setup",))
        self.assertEqual(caught.exception.args[0], said[len("rkmon") : -1])

    def test_update_and_refresh_are_gone(self):
        """package update, then confirm and apply, brings newer code, and rollback goes back; update and refresh
        are unknown commands, whose usage error says where the commands are listed."""
        for argv in (("update",), ("update", "--rollback"), ("refresh", "--from", "x")):
            with self.subTest(argv=argv):
                rc, _, out, err = run(*argv)
                self.assertEqual(
                    (rc, out, err),
                    (
                        2,
                        "",
                        f'rkmon: unknown command "{argv[0]}": rkmon help lists the commands\n',
                    ),
                )

    def test_the_timeouts_default_to_the_table_s(self):
        self.assertEqual(
            run("first-tick")[1].timeout_minutes, commands.FIRST_TICK_TIMEOUT_MINUTES
        )
        self.assertIsNone(run("apply")[1].timeout_minutes)

    def test_the_commands_that_hold_the_package_folder_s_lock(self):
        """In the table, the lock's one home: package update holds the package folder's lock exclusive; help, template
        section, status, logs, next and image ls hold none, since they read and end at once, or follow output; every
        other command holds it shared, for its whole run."""
        locks = {r.name: r.lock for r in commands.TABLE}
        self.assertEqual(
            {n for n, lock in locks.items() if lock == commands.EXCLUSIVE},
            {"package update"},
        )
        self.assertEqual(
            {n for n, lock in locks.items() if lock == commands.NO_LOCK},
            {"help", "template section", "status", "logs", "next", "image ls"},
        )
        self.assertEqual(
            set(locks.values()), {commands.SHARED, commands.EXCLUSIVE, commands.NO_LOCK}
        )
        self.assertEqual(Row("x", "y").lock, commands.SHARED)

    def test_no_text_of_the_table_names_the_program(self):
        """Each text writes {program} where the program's name goes: rkmon may be run as ./rkmon or by a link."""
        texts = [commands.ABOUT, *(p.last for p in commands.PATHS)]
        texts += [g.title for g in commands.GROUPS]
        texts += [what for _, what in commands.ENVIRONMENT]
        for row in commands.TABLE:
            texts += [
                row.summary,
                row.details() if callable(row.details) else row.details,
                row.exit,
                row.brief,
                row.form,
                row.usage,
            ]
            texts += [part for example in row.examples for part in example]
            texts += [o.help for o in row.options if not callable(o.help)]
        texts += [o.help for o in commands.LEADING]
        for text in texts:
            self.assertNotRegex(text, r"\brkmon\b(?!-)", text)
        with mock.patch.object(program, "name", return_value="./rkmon"):
            text = commands.table_help() + "".join(
                commands.help_text(r) for r in commands.TABLE
            )
        self.assertNotRegex(text.replace("./rkmon", "PROGRAM"), r"\brkmon\b(?!-)")

    def test_every_command_a_doc_writes_parses(self):
        """In each of its forms; a doc may name a command without the arguments it requires, an option without its
        value, or a group of commands alone (doc_problem)."""
        words = {**WORDS, "PROFILE": profiles()[0].name}
        bad, read = [], 0
        for pattern in DOCS:
            for doc in sorted(ROOT.glob(pattern)):
                rel = doc.relative_to(ROOT).as_posix()
                for line, text in written(doc.read_text(encoding="utf-8")):
                    part = command_part(text)
                    for form in forms(part) if part else ():
                        read += 1
                        problem = doc_problem(form, words)
                        if problem:
                            bad.append(f"{rel}:{line}: {text}: {problem}")
        self.assertGreater(read, 100, "the commands read")
        self.assertEqual(bad, [])

    def test_what_a_doc_s_command_is_read_as(self):
        self.assertEqual(
            written(
                "Run `rkmon next`, then `rk status`; `rkmon_lib` is code.\n"
                "```\n$ ./rkmon start\ndocker ps\n```\n"
                "and `docker/rkmon logs -f`\n"
            ),
            [(1, "rkmon next"), (3, "./rkmon start"), (6, "docker/rkmon logs -f")],
        )
        for text, part in (
            (
                "rkmon check --settings f.yaml && rkmon confirm",
                "rkmon check --settings f.yaml",
            ),
            ("rkmon logs -f | grep x", "rkmon logs -f"),
            ("rkmon status  # where it is", "rkmon status"),
            (
                "./rkmon login github_read --with-token < <token file>",
                "./rkmon login github_read --with-token",
            ),
            ("rkmon help [COMMAND \\| TOPIC]", "rkmon help [COMMAND | TOPIC]"),
            ("rkmon login: --with-token goes with github_write or github_read", None),
            ("rkmon <command>", None),
        ):
            with self.subTest(text=text):
                self.assertEqual(command_part(text), part)
        for text, found in (
            (
                "rkmon template deploy [--profile PROFILE] FILE",
                [
                    "rkmon template deploy FILE",
                    "rkmon template deploy FILE --profile PROFILE",
                ],
            ),
            (
                "rkmon help [COMMAND | TOPIC]",
                ["rkmon help", "rkmon help COMMAND", "rkmon help TOPIC"],
            ),
            (
                "rkmon image load|save IMAGE_FILE",
                ["rkmon image load IMAGE_FILE", "rkmon image save IMAGE_FILE"],
            ),
            ("rkmon rk ...", ["rkmon rk"]),
        ):
            with self.subTest(text=text):
                self.assertEqual(forms(text), found)
        for form in (
            "rkmon preflight",
            "rkmon first-tick --timeout 5",
            "rkmon nothing",
            "rkmon build --check",
            "rkmon build --setings FILE",
            "rkmon down -v",
        ):
            with self.subTest(form=form):
                self.assertTrue(doc_problem(form, WORDS))
        for form in (
            "rkmon image load",
            "rkmon build",
            "rkmon template deploy --profile",
            "rkmon image",
            "rkmon remove --delete-data",
            "rkmon secret <name>",
            "./rkmon rk config show",
            "rkmon",
        ):
            with self.subTest(form=form):
                self.assertEqual(doc_problem(form, WORDS), "")

    def test_every_command_the_code_names_in_a_fix_or_a_next_line_parses(self):
        found = code_commands(
            {
                f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8")
                for f in CODE
            }
        )
        self.assertGreater(len(found), 40, "the commands read")
        bad = []
        for where, argv in found:
            rc, _, _, err = run(*argv)
            if rc is not None and not later(argv):
                bad.append(f"{where}: {' '.join(argv)}: {err.strip()}")
        self.assertEqual(bad, [])
        for words, _why in LATER:
            with self.subTest(later=words):
                self.assertEqual(
                    run(*words.split())[0], 2, "the table has it: its LATER entry goes"
                )

    def test_the_command_that_sets_each_secret_parses(self):
        """The secrets registry's commands, which its fixes name after the program (secrets_registry.set_command)."""
        from rkmon_lib import secrets_registry

        cfg = {"model_bindings": {"b": {"api_key_secret": "b_api_key"}}}
        for entry in secrets_registry.entries(cfg):
            argv = secrets_registry.set_command(entry)
            with self.subTest(argv=argv):
                rc, ns, _, err = run(*argv)
                self.assertIsNone(rc, err)
                self.assertEqual(ns.row.name, " ".join(argv[: len(ns.row.words)]))

    def test_the_commands_the_code_names_are_found(self):
        sources = {
            "rkmon_lib/x.py": (
                "from . import doctor, program, steps\n"
                "A = Command((program.name(), 'check', '--settings', 'FILE'))\n"
                "B = Command((doctor.wrapper_name(), 'login', 'github_read' if split else 'github_write'))\n"
                "C = checks.Command((self.program, 'confirm', '--settings', file, '--yes'))\n"
                "D = Command((program.name(), *words))\n"
                "E = Command(('docker', 'compose', 'ls'))\n"
                "F = Command((binary, 'auth', 'login'))\n"
                "G = steps.command_line('image load IMAGE_FILE')\n"
                "H = steps.command_line(MIGRATE)\n"
                "MIGRATE = 'migrate --settings FILE'\n"
                "I = steps.command_line(states.NEXT)\n"
                "J = steps.command_line(settings_file.CHECK, FILE=where)\n"
                "K = steps.command_line(CHECK)\n"
                "w = doctor.wrapper_name(environ)\n"
                "L = Command((w, 'chek', 'FILE'))\n"
                "M = Command((self.program, 'remove'))\n"
            ),
            "rkmon_lib/states.py": "NEXT = 'next'\nCHECK = 'CHECK:'\nx = _run('apply')\n",
            "rkmon_lib/gh.py": "x = _run('auth status')\n",
        }
        # A constant is its own module's: settings_file.CHECK, and x's CHECK, are not states' CHECK. After the program's
        # name, a word the table does not have (chek, remove) is read too, for the parser to refuse.
        self.assertEqual(
            code_commands(sources),
            [
                ("rkmon_lib/x.py:2", ["check", "--settings", "f.yaml"]),
                ("rkmon_lib/x.py:3", ["login", "github_read"]),
                ("rkmon_lib/x.py:3", ["login", "github_write"]),
                ("rkmon_lib/x.py:4", ["confirm", "--settings", "VALUE", "--yes"]),
                ("rkmon_lib/x.py:8", ["image", "load", "m.tgz"]),
                ("rkmon_lib/x.py:9", ["migrate", "--settings", "f.yaml"]),
                ("rkmon_lib/x.py:11", ["next"]),
                ("rkmon_lib/x.py:15", ["chek", "f.yaml"]),
                ("rkmon_lib/x.py:16", ["remove"]),
                ("rkmon_lib/states.py:3", ["apply"]),
            ],
        )
        entries = (("standby off", "an entry of LATER's shape"),)
        self.assertTrue(later(["standby", "off", "x"], entries))
        # fewer words than the entry's, and commands it does not name
        self.assertFalse(
            later(["standby"], entries)
            or later(["start"], entries)
            or later(["resume"], entries)
            or later(["package", "update", "x"], entries)
        )


if __name__ == "__main__":
    unittest.main()
