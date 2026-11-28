# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The command table: one row per command, from which the argument parser, `rkmon help` and each command's help are
generated. This module is the table's shape, its generators and the table.

Every text names the program as rkmon was run by (rkmon_lib/program.py): a text of the table writes `{program}` where
the name goes.

- `Option(flags, help, metavar="", default=None, repeat=False, required=False, name="")`: an option: its flags; one
  line that says what it does, and its default; and the word for its value (none for a switch). `repeat`: it
  may be given more than once, and its value is the list of what was given. `name`: its name in the namespace, for a
  command's own option whose flag the program has too: a row's `--dry-run` with a name is the command's own (`rkmon
  adopt --dry-run` shows its plan), and sets that name, never `dry_run`. Its help may be a function of no arguments
  that returns the line, for a line whose words have their home elsewhere (the deploy schema's profiles): it is
  called when the help is printed, never when the table is made.
- `Arg(name, help="", optional=False, rest=False)`: a positional argument: its word in the usage (FILE, KEY) and what
  it is; `rest` takes every word after it, as `rk ARGS` does.
- `Row(name, summary, ...)`: one command: its words after the program's name ("confirm", "image load"); its summary
  (its help's first line, and the WHAT of its banner); its arguments and options; the usage, when the generated one
  (the options, then the arguments) does not say it right; its help's paragraphs (a blank line between two; or a
  function of no arguments that returns them, for words whose home is elsewhere, called as the help is printed),
  examples ((the words after the program's name, what they do)) and Exit line; the overview's group, line (`brief`, the
  summary when blank) and form of the command (`form`; when blank, its name, required options and arguments); the
  deployment states it runs in (none: it needs no deployment); what it needs of
  this folder's container in them (`container`: EXISTS, that it exists; RUNS, that it runs; "", nothing); and how it
  holds the package folder's lock for its whole run (`lock`, rkmon_lib/package_lock.py): SHARED, as each command
  that changes the deployment or reads the package's files after it starts does; EXCLUSIVE, package update's, which
  replaces a package folder; NO_LOCK, none, for a command that reads and ends at once, or only follows output.
- `Group(key, title, topic=True, note="")`: a part of the overview, with the rows whose group is its key, in the
  table's order. When it is a topic, `rkmon help KEY` prints it, and its title says so, with the note after it.
  `Path(key, title, steps, last="")`: a numbered path of the overview (steps.BUILD_STEPS, steps.DEPLOY_STEPS), always
  a topic, with a last line of its own.
- `Gone(words, instead="", in_words="")`: a command, or an option of one, that another command replaced, with no
  alias: its words as typed after the program's name ("preflight", "image import"; "build --check" is the
  option --check of build), and the command that replaces it, as the table names it ("check --settings FILE"); or,
  where no one command does, `in_words`: what does, as the usage error says it, {program} for the program's name
  (setup's: the deploy settings file, login and secret). Its `command` is its command's words, its `option` the
  option that went ("" when the whole command went), and `replacement(name)` the usage error's words after the
  command.
- GLOBAL_OPTIONS: `--dry-run`, `-h`/`--help` and `--version`, which every command takes. A row may give `--dry-run`
  its own help line. A program may take options of its own before the command (`leading`: Options), which the
  overview lists with these.
- rkmon's own table: TABLE, a row for every command docker/rkmon runs, each path step's summary its step's WHAT;
  PATHS, the build and deploy paths of rkmon_lib/steps.py; GROUPS, the overview's other parts (change, day-to-day,
  and newer versions, other monitors and removal); GONE, the commands and options another command replaced;
  LEADING, the option docker/rkmon takes before the command (--image-code); ABOUT and ENVIRONMENT, the overview's
  first line and its environment settings. NEEDS: the sentence of a refusal whose need (EXISTS, RUNS) fails, in the
  place of the state's. FIRST_TICK_TIMEOUT_MINUTES: the default of first-tick's --timeout-minutes.
  `drain_timeout_minutes()`: how long apply and stop --drain let the work in flight finish without
  --timeout-minutes, the package's update.drain_timeout_min, which confirm's summary names too. The --profile line of
  `template deploy`, and the help of apply and stop with that default, are read when the help is printed.
  `table_parser(version="")`: the parser of that table (VERSION: what --version prints); `table_help(words=())`: what
  `rkmon help WORDS` prints from it, a KeyError for a command or topic it does not have.
- `overview(rows, paths=(), groups=(), about="", environment=(), leading=())`: `rkmon help`: what rkmon is, the
  usage, each path's steps (each command at column 16, and on the lines after it what steps.aside gives), each
  group's commands (each line at column 28, or on the next line when the command reaches it), the options for every
  command (GLOBAL_OPTIONS, then `leading`), the environment ((setting, what it does) pairs) and the exit codes.
- `section(key, rows, paths=(), groups=())`: one topic of `rkmon help`: a path, a group, or exit-codes (each exit
  code and what it means). `topics(paths, groups)`: the topics, in the overview's order.
- `help_text(row)`: a command's help: `PROGRAM NAME: SUMMARY.`, then the usage, the paragraphs, the examples
  (each command at column 2, what it does three columns after the longest, or, where the longest of those would not
  fit there, on the line after each command, at column EXAMPLE_BELOW), the options (each one's line two columns
  after the longest flags) and the Exit line. `usage_line(row)`: its Usage line's words on one line, `PROGRAM NAME
  USAGE` (the row's usage, else the generated one), the one a command's own usage error quotes.
- `form(row)`: a command as `rkmon help` names it after the program's name, the one home of that form: the row's
  form, else its name, its required options and its arguments ("migrate --settings FILE", "logs [-f]").
  `form_of(words)`: that form of the command WORDS in rkmon's own table, TABLE ("adopt --from DIR" for "adopt").
- `find(words, rows)`: the row of a command's words, or None. `help_for(words, rows, paths=(), groups=(), about="",
  environment=(), leading=(), gone=())`: what `rkmon help WORDS` prints: the overview for no words, a command's
  help, or a topic; for anything else a KeyError, its text the usage error after `PROGRAM help`: a command of `gone`
  names the command that replaces it, as the parser's error does (`: unknown command "preflight": PROGRAM check
  --settings FILE replaces it`), any other where they are listed (`: no command or topic "rebuild": PROGRAM help
  lists them`).
- `parser(rows, version="", paths=(), groups=(), about="", environment=(), gone=(), leading=())`: the argparse
  parser. Its namespace has `row` (the command's row; None when no command is given), `dry_run`, each leading
  option, and the command's arguments and options, each by its name in lowercase (`--from` as from_). `-h` and
  `--help` print the command's help on stdout (the overview, before any command), and `--version` the version; each
  exits 0. A usage error prints `PROGRAM COMMAND: what is wrong` on stderr and exits 2; an option the command does
  not take is named without the value after its =, which may be a secret. A command or option of `gone`
  is one, which names the command that replaces it: `PROGRAM: unknown command "preflight": PROGRAM check --settings
  FILE replaces it`, `PROGRAM build: unknown option "--check": ...`. A command the table does not have is one too,
  which says where the commands are listed (`PROGRAM: unknown command "rebuild": PROGRAM help lists the commands`),
  and a word that starts several commands, alone, names them (`PROGRAM image: name one of its commands: image load,
  image ls`). The leading options are taken before the command only. A command named twice in the table, or one whose
  words start another's, is a ValueError.

Every text is at most WIDTH columns, no word broken (rkmon_lib/wrapping.py).

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import argparse
import keyword
import sys
from dataclasses import dataclass

from . import container_layout, outcome, program, steps, wrapping

WIDTH = 79  # help is at most 80 columns wide; its lines stop short of the 80th
STEP_COLUMN = 16  # a path step's command, and the line of each option for every command
BRIEF_COLUMN = 28  # a group command's line, and what each environment setting does
EXAMPLE_GAP = 3  # between the longest example and what each example does
# what each example does, on the line after it, when it would not fit three columns after the longest
EXAMPLE_BELOW = 6
OPTION_GAP = 2  # between the longest flags and what each option does
# what a command of the deploy side needs of this folder's container in its states (Row.container)
EXISTS, RUNS = "exists", "runs"
NEEDS = {
    EXISTS: "this deployment's container does not exist",
    RUNS: "the container is not running",
}
# how a command holds the package folder's lock for its whole run (Row.lock; rkmon_lib/package_lock.py)
SHARED, EXCLUSIVE, NO_LOCK = "shared", "exclusive", ""


@dataclass(frozen=True)
class Option:
    """An option of a command."""

    flags: tuple
    help: str
    metavar: str = ""
    default: object = None
    repeat: bool = False
    required: bool = False
    name: str = ""

    def __post_init__(self):
        flags = (self.flags,) if isinstance(self.flags, str) else tuple(self.flags)
        if not flags or not all(f.startswith("-") for f in flags):
            raise ValueError(f"an option's flags start with -: {self.flags!r}")
        if not callable(self.help) and not str(self.help).strip():
            raise ValueError(f"an option says what it does: {flags[0]}")
        object.__setattr__(self, "flags", flags)

    @property
    def dest(self) -> str:
        """Its name in the namespace: `name`, else its long flag's name, each - made _, and a Python keyword with _
        after it."""
        if self.name:
            return self.name
        long = [f for f in self.flags if f.startswith("--")] or list(self.flags)
        name = long[0].lstrip("-").replace("-", "_")
        return name + "_" if keyword.iskeyword(name) else name

    @property
    def shown(self) -> str:
        """Its flags and value word, as help lists them: `-f, --follow`, `--settings FILE`."""
        return ", ".join(self.flags) + (f" {self.metavar}" if self.metavar else "")


@dataclass(frozen=True)
class Arg:
    """A positional argument of a command."""

    name: str
    help: str = ""
    optional: bool = False
    rest: bool = False

    @property
    def dest(self) -> str:
        return self.name.lower().replace("-", "_")

    @property
    def shown(self) -> str:
        return f"[{self.name}]" if self.optional else self.name


@dataclass(frozen=True)
class Row:
    """One command of the table."""

    name: str
    summary: str
    args: tuple = ()
    options: tuple = ()
    usage: str = ""
    details: str = ""
    examples: tuple = ()
    exit: str = ""
    group: str = ""
    brief: str = ""
    form: str = ""
    states: tuple = ()
    container: str = ""
    lock: str = SHARED

    @property
    def words(self) -> tuple:
        return tuple(self.name.split())


@dataclass(frozen=True)
class Gone:
    """A command, or an option of one, that another command replaced, or (`in_words`) that more than one thing did."""

    words: str
    instead: str = ""
    in_words: str = ""

    @property
    def command(self) -> tuple:
        """The command's words, its option left out: ("build",) of "build --check"."""
        return tuple(w for w in self.words.split() if not w.startswith("-"))

    @property
    def option(self) -> str:
        """The option that went ("--check" of "build --check"), or "" when the whole command went."""
        return next((w for w in self.words.split() if w.startswith("-")), "")

    def replacement(self, name: str) -> str:
        """What the usage error says replaces it, NAME the program's name: `NAME INSTEAD replaces it`, or IN_WORDS with
        NAME for each {program} in it."""
        if self.in_words:
            return self.in_words.format(program=name)
        return f"{name} {self.instead} replaces it"


@dataclass(frozen=True)
class Group:
    """A part of the overview: the rows whose group is its key."""

    key: str
    title: str
    topic: bool = True
    note: str = ""


@dataclass(frozen=True)
class Path:
    """A numbered path of the overview."""

    key: str
    title: str
    steps: tuple
    last: str = ""


GLOBAL_OPTIONS = (
    Option(("--dry-run",), "show what the command would do; change nothing"),
    Option(("-h", "--help"), "the command's help, with examples"),
    Option(("--version",), "this package's version"),
)


def _wrap(text: str, first: str = "", rest: str = "") -> list:
    return wrapping.wrap(text, WIDTH, first, rest)


def _columns(left: str, right: str, column: int) -> list:
    """`  LEFT`, and RIGHT at `column`: on the same line when two spaces fit between them, else on the next."""
    lead = f"  {left}"
    if len(lead) + 2 > column:
        return [lead] + _wrap(right, " " * column, " " * column)
    return _wrap(right, lead.ljust(column), " " * column)


def _shown_in_usage(o: Option) -> str:
    """An option as a usage names it: its long flag (else its first), and its value word."""
    flag = next((f for f in o.flags if f.startswith("--")), o.flags[0])
    return flag + (f" {o.metavar}" if o.metavar else "")


def form(row: Row) -> str:
    """A command as `rkmon help` names it after the program's name: the row's form, else its name, its required
    options and its arguments."""
    if row.form:
        return program.named(row.form)
    required = [_shown_in_usage(o) for o in row.options if o.required]
    return " ".join([row.name, *required, *[a.shown for a in row.args]])


def form_of(words: str) -> str:
    """The form of the command WORDS in rkmon's own table: see the module's docstring."""
    return form(find(tuple(words.split()), TABLE))


def _line(o: Option) -> str:
    """An option's help line, its function called when it has one, with the program's name in it."""
    return program.named(o.help() if callable(o.help) else o.help)


def _usage(row: Row) -> str:
    """The usage after the command's name: the row's, else the generated one (its options, then its arguments)."""
    if row.usage:
        return program.named(row.usage)
    parts = []
    for o in row.options:
        shown = _shown_in_usage(o)
        parts.append(shown if o.required else f"[{shown}]")
    parts += [a.shown for a in row.args]
    return " ".join(parts)


def usage_line(row: Row) -> str:
    """A command's usage on one line, `PROGRAM NAME USAGE`, as its help's Usage line says it."""
    return " ".join(w for w in (program.name(), row.name, _usage(row)) if w)


def help_text(row: Row) -> str:
    """A command's help, as -h, --help and `rkmon help COMMAND` print it."""
    prog = program.name()
    out = _wrap(f"{prog} {row.name}: {program.named(row.summary)}.")
    lead = f"Usage:  {prog} {row.name}"
    usage = _usage(row)
    out += [""] + (_wrap(usage, lead + " ", " " * (len(lead) + 1)) if usage else [lead])
    details = row.details() if callable(row.details) else row.details
    for para in program.named(details).split("\n\n"):
        if para.strip():
            out += [""] + _wrap(" ".join(para.split()))
    if row.examples:
        commands = [
            f"{prog} {program.named(words)}".rstrip() for words, _ in row.examples
        ]
        whats = [program.named(what) for _, what in row.examples]
        column = 2 + max(len(c) for c in commands) + EXAMPLE_GAP
        if column + max(len(w) for w in whats) > WIDTH:
            column = EXAMPLE_BELOW
        out += ["", "Examples:"]
        for command, what in zip(commands, whats):
            out += _columns(command, what, column)
    if row.options:
        column = 2 + max(len(o.shown) for o in row.options) + OPTION_GAP
        out += ["", "Options:"]
        for o in row.options:
            out += _columns(o.shown, _line(o), column)
    if row.exit:
        out += [""] + _wrap(f"Exit: {program.named(row.exit)}")
    return "\n".join(out) + "\n"


def topics(paths=(), groups=()) -> list:
    """The topics `rkmon help TOPIC` takes, in the overview's order."""
    return [p.key for p in paths] + [g.key for g in groups if g.topic] + ["exit-codes"]


def _title(title: str, key: str, note: str) -> str:
    """A part's title, with the help command that prints it alone: `TITLE (PROGRAM help KEY; NOTE):`."""
    see = f"{program.name()} help {key}" + (f"; {program.named(note)}" if note else "")
    return f"{program.named(title)} ({see}):"


def _path_lines(path: Path) -> list:
    prog = program.name()
    out = _wrap(_title(path.title, path.key, ""))
    for s in path.steps:
        lead = f"  {s.n} {s.name}".ljust(STEP_COLUMN)
        out += _wrap(f"{prog} {s.command}", lead, " " * STEP_COLUMN)
        for line in steps.aside(s):
            if line:
                out += _wrap(line, " " * STEP_COLUMN, " " * STEP_COLUMN)
    if path.last:
        out += _wrap(program.named(path.last), "  ", "  ")
    return out


def _group_lines(group: Group, rows) -> list:
    if group.topic:
        out = _wrap(_title(group.title, group.key, group.note))
    else:
        out = _wrap(f"{program.named(group.title)}:")
    for row in rows:
        if row.group == group.key:
            brief = program.named(row.brief or row.summary)
            out += _columns(f"{program.name()} {form(row)}", brief, BRIEF_COLUMN)
    return out


def _exit_lines() -> list:
    """The exit codes in one sentence (outcome's), each code with its words on one line."""
    summary = outcome.exit_summary(held=True)
    return _wrap(f"Exit codes: {summary} More: {program.name()} help exit-codes.")


def overview(
    rows, paths=(), groups=(), about: str = "", environment=(), leading=()
) -> str:
    """`rkmon help`: what rkmon is, the usage, every path and group, the options for every command, the environment
    and the exit codes."""
    prog = program.name()
    parts = []
    if about:
        parts.append(_wrap(f"{prog}: {program.named(about)}"))
    head = f"        {prog} help ["
    which = " | ".join(["COMMAND"] + topics(paths, groups)) + "]"
    parts.append(
        [f"Usage:  {prog} COMMAND [OPTIONS]"] + _wrap(which, head, " " * len(head))
    )
    parts += [_path_lines(p) for p in paths]
    parts += [_group_lines(g, rows) for g in groups]
    lines = ["Options for every command:"]
    for o in (*GLOBAL_OPTIONS, *leading):
        lines += _columns(o.shown, _line(o), STEP_COLUMN)
    parts.append(lines)
    if environment:
        lines = ["Environment:"]
        for setting, what in environment:
            lines += _columns(setting, program.named(what), BRIEF_COLUMN)
        parts.append(lines)
    parts.append(_exit_lines())
    return "\n\n".join("\n".join(p) for p in parts) + "\n"


def section(key: str, rows, paths=(), groups=()) -> str:
    """One topic of `rkmon help`: a path, a group, or the exit codes. A KeyError for any other key."""
    for p in paths:
        if p.key == key:
            return "\n".join(_path_lines(p)) + "\n"
    for g in groups:
        if g.key == key and g.topic:
            return "\n".join(_group_lines(g, rows)) + "\n"
    if key == "exit-codes":
        lines = ["Exit codes:"]
        for n, _, long in outcome.EXIT_CODES:
            lines += _columns(str(n), long, 8)
        return "\n".join(lines) + "\n"
    raise KeyError(key)


def find(words, rows):
    """The row whose name is these words, or None."""
    words = tuple(words)
    return next((r for r in rows if r.words == words), None)


def help_for(
    words,
    rows,
    paths=(),
    groups=(),
    about: str = "",
    environment=(),
    leading=(),
    gone=(),
) -> str:
    """What `rkmon help WORDS` prints: the overview, a command's help, or a topic. Anything else is a KeyError whose
    text is the usage error after `PROGRAM help`: for a command of GONE, the parser's, which names the command that
    replaces it; else that the program's help lists the commands and topics."""
    words = tuple(words)
    if not words:
        return overview(rows, paths, groups, about, environment, leading)
    replaced = _replaced(list(words), gone)
    if replaced:
        raise KeyError(replaced)
    row = find(words, rows)
    if row is not None:
        return help_text(row)
    if len(words) == 1 and words[0] in topics(paths, groups):
        return section(words[0], rows, paths, groups)
    shown = " ".join(words)
    raise KeyError(f': no command or topic "{shown}": {program.name()} help lists them')


class _Parser(argparse.ArgumentParser):
    """A parser whose usage error is one line on stderr, `PROGRAM COMMAND: what is wrong`, and exit code 2. An
    argument it does not know is its own usage error, named with its command; an option's value given after its = is
    not shown, as it may be a secret (never from a flag's value)."""

    def error(self, message):
        self.exit(outcome.USAGE, f"{self.prog}: {message}\n")

    def parse_known_args(self, args=None, namespace=None):
        ns, extra = super().parse_known_args(args, namespace)
        if extra:
            shown = [a.split("=", 1)[0] if a.startswith("-") else a for a in extra]
            self.error(f"unrecognized arguments: {' '.join(shown)}")
        return ns, extra


def _replaced(args, gone, valued=()) -> str:
    """The usage error for a command or an option of `gone` in `args`, after the program's name and before its colon
    (`: unknown command ...`, ` build: unknown option ...`), or "". `valued`: the leading options that take a value.
    """
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in valued else 1
    typed = args[i:]
    for g in gone:
        name = list(g.command)
        if typed[: len(name)] != name:
            continue
        instead = g.replacement(program.name())
        if not g.option:
            return f': unknown command "{g.words}": {instead}'
        if any(a.split("=", 1)[0] == g.option for a in typed[len(name) :]):
            return f' {" ".join(name)}: unknown option "{g.option}": {instead}'
    return ""


def _unknown(args, names, valued=()) -> str:
    """The usage error for a command the table does not have in `args`, after the program's name and before its
    colon, or "": a word no command starts with, or a word that starts several commands with no next word that names
    one of them. NAMES: each command's words, in the table's order; `valued`: the leading options that take a value.
    """
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in valued else 1
    typed = args[i:]
    for depth in range(1, len(typed) + 1):
        words = tuple(typed[:depth])
        if words in names:
            return ""
        if not any(n[:depth] == words for n in names):
            where = f" {' '.join(words[:-1])}" if depth > 1 else ""
            return f'{where}: unknown command "{" ".join(words)}": {program.name()} help lists the commands'
        if depth == len(typed):
            under = [" ".join(n) for n in names if n[:depth] == words]
            return f" {' '.join(words)}: name one of its commands: {', '.join(under)}"
        if typed[depth].startswith("-"):
            return ""  # an option after the word: the parser's to read
    return ""


class _Top(_Parser):
    """The program's parser: `row` and `dry_run` are always in its namespace; a command or option of `gone` is a usage
    error that names the command that replaces it, and a command the table does not have one that says where the
    commands are listed."""

    gone = ()
    valued = ()
    names = ()

    def parse_known_args(self, args=None, namespace=None):
        args = sys.argv[1:] if args is None else list(args)
        why = _replaced(args, self.gone, self.valued) or _unknown(
            args, self.names, self.valued
        )
        if why:
            self.exit(outcome.USAGE, f"{self.prog}{why}\n")
        ns, extra = super().parse_known_args(args, namespace)
        if not hasattr(ns, "dry_run"):
            ns.dry_run = False
        if not hasattr(ns, "row"):
            ns.row = None
        return ns, extra


class _Show(argparse.Action):
    """An option that prints a text and exits 0: -h, --help and --version."""

    def __init__(self, option_strings, dest, text=None, **kwargs):
        kwargs.pop("default", None)
        super().__init__(
            option_strings,
            dest=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
            nargs=0,
            **kwargs,
        )
        self.text = text

    def __call__(self, parser, namespace, values, option_string=None):
        sys.stdout.write(self.text())
        sys.stdout.flush()
        parser.exit(0)


def _add_dry_run(p: argparse.ArgumentParser) -> None:
    # No default: a --dry-run before the command stays when the command's own parser runs.
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=argparse.SUPPRESS
    )


def _add_globals(
    p: argparse.ArgumentParser, text, version: str, dry_run: bool = True
) -> None:
    """The options every command takes: --dry-run (unless the row gives its own), -h, --help and --version."""
    if dry_run:
        _add_dry_run(p)
    p.add_argument("-h", "--help", action=_Show, text=text)
    p.add_argument("--version", action=_Show, text=lambda: f"{version}\n")


def _add_row(p: argparse.ArgumentParser, row: Row) -> None:
    for o in row.options:
        if o.flags == ("--dry-run",) and not o.name:
            _add_dry_run(p)
        elif o.metavar:
            p.add_argument(
                *o.flags,
                dest=o.dest,
                metavar=o.metavar,
                default=o.default,
                required=o.required,
                action="append" if o.repeat else "store",
            )
        else:
            p.add_argument(*o.flags, dest=o.dest, action="store_true")
    for a in row.args:
        if a.rest:
            p.add_argument(a.dest, metavar=a.name, nargs=argparse.REMAINDER)
        else:
            p.add_argument(a.dest, metavar=a.name, nargs="?" if a.optional else None)
    p.set_defaults(row=row)


def parser(
    rows,
    version: str = "",
    paths=(),
    groups=(),
    about: str = "",
    environment=(),
    gone=(),
    leading=(),
) -> argparse.ArgumentParser:
    """The argument parser of a command table."""

    def whole():
        return overview(rows, paths, groups, about, environment, leading)

    names = [r.words for r in rows]
    for words in names:
        if names.count(words) > 1:
            raise ValueError(f"the command {' '.join(words)!r} is in the table twice")
        if any(other[: len(words)] == words for other in names if other != words):
            raise ValueError(
                f"the command {' '.join(words)!r} is also the start of another"
            )
    top = _Top(prog=program.name(), add_help=False, allow_abbrev=False)
    top.gone = tuple(gone)
    top.valued = tuple(f for o in leading if o.metavar for f in o.flags)
    top.names = tuple(names)
    _add_globals(top, whole, version)
    _add_row(top, Row("", "", options=tuple(leading)))
    top.set_defaults(row=None)
    branches = {(): top.add_subparsers(metavar="COMMAND", parser_class=_Parser)}
    for row in rows:
        words = row.words
        for depth in range(1, len(words)):
            key = words[:depth]
            if key not in branches:
                mid = branches[key[:-1]].add_parser(
                    key[-1], add_help=False, allow_abbrev=False
                )
                _add_globals(mid, whole, version)
                branches[key] = mid.add_subparsers(
                    metavar="COMMAND", parser_class=_Parser
                )
                branches[key].required = True
        p = branches[words[:-1]].add_parser(
            words[-1], add_help=False, allow_abbrev=False
        )
        own = any(o.flags == ("--dry-run",) for o in row.options)
        _add_globals(p, lambda row=row: help_text(row), version, dry_run=not own)
        _add_row(p, row)
    return top


def table_parser(version: str = "") -> argparse.ArgumentParser:
    """The argument parser of rkmon's own table (VERSION: what --version prints)."""
    return parser(TABLE, version, PATHS, GROUPS, ABOUT, ENVIRONMENT, GONE, LEADING)


def table_help(words=()) -> str:
    """What `rkmon help WORDS` prints from rkmon's own table: a KeyError for a command or topic it does not have, a
    command of GONE too (help_for)."""
    return help_for(words, TABLE, PATHS, GROUPS, ABOUT, ENVIRONMENT, LEADING, GONE)


# --- rkmon's own table ------------------------------------------------------------------------------------------------

FIRST_TICK_TIMEOUT_MINUTES = (
    25  # how long rkmon first-tick waits for the first tick, at most
)
# The deployment states the commands of the deploy side run in (rkmon_lib/states.py)
_CONFIRM_STATES = (
    1,
    2,
    *range(5, 17),
)  # all but an apply, rollback or adopt stopped part-way, and another layout
_IMAGE_STATES = tuple(
    range(6, 17)
)  # once the confirmed settings are this package's to run
_LOGIN_STATES = tuple(range(7, 17))  # once the image is here
# all but another deployment's container name, no confirm, and never started
_CONTAINER_STATES = (*range(3, 8), *range(9, 17))
# an adopt stopped part-way (states.runs_here: no apply or rollback), before the first start, or with the container
# stopped
_ADOPT_STATES = (3, 7, 8, 12)
# once the logins are set, but not in standby: the monitor you moved from may still use the web host
_WEB_HOST_STATES = (*range(8, 14), 15, 16)
_START_STATES = (8, 10, 12)  # never started, put back, or stopped
# an apply or rollback stopped part-way (states.runs_here: no adopt), and every state once the container exists and
# its logins are set; in 12 and 14 to 16 apply has nothing to do, nor before the first start, in 8 and in 7 with no
# container yet, outside these, as the first start takes your confirmed settings
_APPLY_STATES = (3, *range(9, 17))
# every state but an apply, rollback or adopt stopped part-way, which finishes first
_PACKAGE_UPDATE_STATES = (1, 2, *range(4, 17))
# all but another deployment's container name and no confirm: before the first start too, for the volumes rkmon login
# makes there
_REMOVE_STATES = tuple(range(3, 17))


def drain_timeout_minutes():
    """How long apply and stop --drain let the work in flight finish without --timeout-minutes: see the module's
    docstring."""
    from . import registry  # the package's defaults, read when they are asked for

    return registry.package_section("update")["drain_timeout_min"]


def _apply_details() -> str:
    """apply's paragraphs, with the drain's default (drain_timeout_minutes)."""
    return (
        "It makes what {program} confirm showed: new settings, new code from a package update, or a new image. A "
        "restart first lets the work in flight finish, for at most --timeout-minutes "
        f"({drain_timeout_minutes():g} by default); --no-drain stops it instead, and it starts again later. The "
        "doctor runs before and after; a new failure puts the previous code and settings back (ROLLED BACK, exit 1). "
        "On a stopped container it puts everything in place, and the container stays stopped.\n\n"
        "A drain can take up to --timeout-minutes. On a remote host, run apply inside tmux or screen; if it is "
        "interrupted, run it again."
    )


def _profile_line() -> str:
    """--profile's line: the deploy schema's profiles, each with what it is for; the first is the default."""
    from . import (
        file_types,
        settings_file,
    )  # each imports this module: read when the help is printed

    found = settings_file.profiles(file_types.DEPLOY)
    each = "; ".join(f"{p.name} for {p.what}" for p in found)
    return f"the kind of deployment, which sets the features that are on (default: {found[0].name}): {each}"


def _image_exit(docker: str = "", refused: str = "", earlier: bool = True) -> str:
    """The Exit line of the deploy path's step 4. DOCKER: the docker step that may fail before the checks (exit 1);
    REFUSED: what else the command refuses, besides an image that does not suit (exit 4); EARLIER: the command needs
    confirmed deploy settings (exit 3 without them)."""
    failed = f"1 docker could not {docker} it; " if docker else ""
    return (
        f"0 the image is here and suits this package; {failed}2 usage error; "
        + ("3 an earlier step is not done; " if earlier else "")
        + f"4 {refused}the image does not suit this package, or this network's TLS fails inside it."
    )


_TEMPLATE_FILE = Arg("FILE", "the file to write; - prints the template instead")
_TEMPLATE_DRY_RUN = Option(
    ("--dry-run",), "check that FILE can be written; write nothing"
)
_TEMPLATE_WRITES = (
    "It never writes over a file, nor into the package, a git work tree or a cloud-synced folder. With - in place of "
    "FILE, it prints the template, alone."
)
_TEMPLATE_EXIT = (
    "0 written, or printed; 2 usage error; 4 refused: FILE exists, or lies inside the package, a git work tree or a "
    "cloud-synced folder."
)
_BUILT_SETTINGS = Option(
    ("--settings",),
    "your build settings file, which names the image (default: this deployment's image)",
    metavar="FILE",
)
_WALK_CODES = (
    "0 every login and secret your settings need is set; 1 you said no to each one left, or the logins could not be "
    "read; 2 usage error; 3 there is no image yet; 4 a login or secret is missing or wrong"
)

TABLE = (
    Row(
        "help",
        "the paths and the commands, one command's help, or one topic",
        args=(
            Arg("COMMAND", "a command's words, or a topic", optional=True, rest=True),
        ),
        usage="[COMMAND | TOPIC]",
        details=(
            "With no COMMAND, it prints the overview: the build path, the deploy path, the other commands, the "
            "options for every command and the exit codes. With a command, it prints that command's help, as -h "
            "after the command does; with a topic, which the overview's usage lists, that part of the overview."
        ),
        examples=(
            ("help", "the overview"),
            ("help deploy-path", "the deploy path's steps"),
            ("help image load", "the help of image load"),
        ),
        exit="0 the help printed; 2 no such command or topic.",
        lock=NO_LOCK,
    ),
    Row(
        "template build",
        steps.BUILD_STEPS[0].what,
        args=(_TEMPLATE_FILE,),
        options=(_TEMPLATE_DRY_RUN,),
        details=(
            "It writes FILE, a new file that only you can read (permissions 600), with the build settings' keys, "
            "each with its comment; each value for you to fill in is marked <...>. "
            + _TEMPLATE_WRITES
        ),
        examples=(
            ("template build ~/rkmon-build.yaml", "start your build settings"),
            ("template build -", "print the template"),
        ),
        exit=_TEMPLATE_EXIT,
    ),
    Row(
        "template deploy",
        steps.DEPLOY_STEPS[0].what,
        args=(_TEMPLATE_FILE,),
        options=(
            Option(("--profile",), _profile_line, metavar="PROFILE"),
            _TEMPLATE_DRY_RUN,
        ),
        details=(
            "It writes FILE, a new file that only you can read (permissions 600), with the deploy settings that "
            "kind of deployment needs, each with its comment; each value for you to fill in is marked <...>. A "
            "feature that starts off has its switch alone: {program} template section KEY prints its section in "
            "full. " + _TEMPLATE_WRITES
        ),
        examples=(
            ("template deploy ~/rkmon-deploy.yaml", "a full deployment's settings"),
            ("template deploy --profile review-only -", "the review-only template"),
        ),
        exit=_TEMPLATE_EXIT,
    ),
    Row(
        "template section",
        "print one section of the deploy file in full, every key with its comment",
        args=(
            Arg(
                "KEY",
                "the section's name, its parents' names before it, joined by dots; none: the list of sections",
                optional=True,
            ),
        ),
        details=(
            "It prints the section as the deploy file's every-key template writes it: its title comment, then every "
            "key with its comment, indented as in the file, so that you can paste it over the section's switch "
            "alone in your deploy settings file. KEY is the section's name, with its parents' names before it, "
            "joined by dots, such as features.alerts. With no KEY, it lists each section's KEY and title. It writes "
            "nothing."
        ),
        examples=(
            (
                "template section features.alerts",
                "the alerts, every key with its comment",
            ),
            ("template section", "each section's KEY and title"),
        ),
        exit="0 the text printed; 2 usage error, or no such section.",
        group="change",
        brief="a section of the deploy file in full, every key with its comment (to turn on a feature)",
        form="template section KEY",
        lock=NO_LOCK,
    ),
    Row(
        "check",
        "check this machine and your settings; change nothing but the ticket fields JIRA requires, added on your yes",
        options=(
            Option(
                ("--settings",),
                "your build or deploy settings file (default: the deploy settings file this deployment recorded)",
                metavar="FILE",
            ),
        ),
        details=(
            "A build settings file gets build step 2: docker and buildx; the file, which only you may read, outside "
            "the package, git and cloud-synced folders; each key against the build schema; your network's TLS "
            "against the certificates the file names; what the build reads over the network, with a WARN for each "
            "blank version; then the build's parameters. A deploy settings file gets deploy step 2: docker and "
            "compose; disk, memory, CPUs and the ports; the file, and each key against the deploy schema; the "
            "container's name; and, while the deployment's container exists, the logins. With ticket creation on, the "
            "ticket creation check reads, with the atlassian login, which fields your JIRA project requires on a new "
            "ticket (a read: no ticket is made); {program} confirm and {program} doctor run it too. Without the "
            "atlassian login it reads nothing: its row is a SKIP until the deployment can have the login (deploy "
            "step 5), then a FAIL while ticket creation is live. With the login, on a terminal, it then asks for a "
            "value for each required field your file lacks, and offers to add their lines under "
            "features.ticket_creation.required_fields: the one write {program} check makes, and only on your yes. "
            "Every other line and comment of the file stays as it was, and the file as it was is kept beside it as "
            "FILE.bak; without a terminal, on a no, or with --dry-run, nothing is written. Each check is a row, and "
            "each WARN and FAIL comes with its fix."
        ),
        examples=(
            ("check --settings ~/rkmon-build.yaml", "build step 2"),
            ("check --settings ~/rkmon-deploy.yaml", "deploy step 2"),
            ("check", "the file confirm recorded"),
        ),
        exit=(
            "0 every check passed; 1 an error, such as lines you said yes to that could not be written; 2 usage "
            "error; 3 docker, its compose or buildx is missing, or no settings file is named or recorded; 4 a check "
            "failed."
        ),
    ),
    Row(
        "build",
        steps.BUILD_STEPS[2].what,
        options=(
            Option(
                ("--settings",),
                "your build settings file",
                metavar="FILE",
                required=True,
            ),
            Option(
                ("--dry-run",),
                "run every check and show the docker build; build nothing",
            ),
            Option(
                ("--no-cache",),
                "build every layer again, none from docker's build cache",
            ),
        ),
        details=(
            "One docker build: each value of your build settings becomes a build argument of the Dockerfile, and each "
            "download location a build secret, in no layer, history or label; the image's labels and the build record "
            "it carries are the build's own, and docker is given no other flag. It runs build step 2's checks of the "
            "file first, reads what only the network gives (the newest build of a ROCm release, the newest release "
            "for a blank version, the Claude Code version the SDK carries), and builds nothing when one fails. With "
            "--dry-run, it prints the docker build it would run, the same as the build's but for the label that says "
            "when it was built."
        ),
        examples=(
            ("build --settings ~/rkmon-build.yaml", "build step 3"),
            ("build --settings ~/rkmon-build.yaml --dry-run", "the command, not run"),
        ),
        exit=(
            "0 built; 1 the docker build failed; 2 usage error; 4 a check failed, or a value the build needs cannot be "
            "read, such as the build of a ROCm release AMD's public folder does not list."
        ),
    ),
    Row(
        "image test",
        steps.BUILD_STEPS[3].what,
        options=(_BUILT_SETTINGS,),
        details=(
            "It runs tests/container/image.py on the image. Each stage runs in throwaway containers with no network, "
            "except deploy-path, no-restart, web-host, one-time-move, adopt, team-prototype, several-deployments and "
            "first-apply, each of whose deployments runs on an internal network that reaches only stand-ins on this "
            "host, so nothing leaves the host. Its stages, in order: "
            + "; ".join(
                f"{name} ({what})" for name, what in steps.IMAGE_TEST_STAGES.items()
            )
            + "."
        ),
        examples=(("image test --settings ~/rkmon-build.yaml", "build step 4"),),
        exit=(
            "0 every test passed; 1 the tests did not start; 2 usage error; 3 no such image; 4 a test failed, or a "
            "check of your build settings."
        ),
    ),
    Row(
        "image save",
        "save the image to a file, to carry to the deploy machine",
        args=(Arg("IMAGE_FILE", "the file to write"),),
        options=(_BUILT_SETTINGS,),
        usage="IMAGE_FILE [--settings FILE]",
        details=(
            "docker save, compressed with gzip, or pigz when this machine has it. On the deploy machine, {program} "
            "image load IMAGE_FILE loads it."
        ),
        examples=(
            (
                "image save ~/rocke-monitor.tar.gz --settings ~/rkmon-build.yaml",
                "build step 5",
            ),
        ),
        exit=(
            "0 saved; 1 docker save or the compression failed; 2 usage error; 3 no such image; 4 a check of your "
            "build settings failed."
        ),
    ),
    Row(
        "image push",
        "push the image to the registry your build settings name",
        options=(
            Option(
                ("--settings",),
                "your build settings file, which names the image and the registry (image.registry)",
                metavar="FILE",
                required=True,
            ),
        ),
        details=(
            "It tags the image for the registry, pushes it, then removes that extra tag. On the deploy machine, "
            "{program} image pull pulls it."
        ),
        examples=(("image push --settings ~/rkmon-build.yaml", "build step 5"),),
        exit=(
            "0 pushed; 1 a docker command failed; 2 usage error; 3 no such image; 4 your build settings name no "
            "registry, or a check of them failed."
        ),
    ),
    Row(
        "image load",
        "load the image from a file, and check that it suits this package",
        args=(Arg("IMAGE_FILE", "the file image save wrote"),),
        details=(
            "docker load, then the checks of {program} image check on the image it loaded. Before a confirm has "
            "recorded your deploy settings, it loads the image and stops there."
        ),
        examples=(("image load ~/rocke-monitor.tar.gz", "deploy step 4"),),
        exit=_image_exit(
            "load",
            "the file has no image of the name your settings give, ",
            earlier=False,
        ),
    ),
    Row(
        "image pull",
        "pull the image from the registry your deploy settings name, and check it",
        details=(
            "docker pull from that registry, the image tagged as your deploy settings name it, then the checks of "
            "{program} image check."
        ),
        examples=(("image pull", "deploy step 4, from your registry"),),
        exit=_image_exit("pull", "your settings name no registry, "),
        states=_IMAGE_STATES,
    ),
    Row(
        "image check",
        "check that the image is on this host and suits this package",
        details=(
            "The image your deploy settings name must be on this host, built from this package's image recipe (its own "
            "copies of what it is built from, and of the scripts it runs as root or through sudo, are this "
            "package's), and trust this network's TLS from inside it. It changes nothing."
        ),
        examples=(("image check", "deploy step 4, an image already here"),),
        exit=_image_exit(),
        states=_IMAGE_STATES,
    ),
    Row(
        "confirm",
        steps.DEPLOY_STEPS[2].what,
        options=(
            Option(
                ("--settings",),
                "your deploy settings file (default: the one recorded before; the first confirm records it)",
                metavar="FILE",
            ),
            Option(("--yes",), "answer yes (a failed check still refuses)"),
            Option(("--dry-run",), "ask nothing, record nothing"),
        ),
        details=(
            "Before the first start, it shows every setting. After it, only what changes and how each change "
            "applies: no restart, or a restart. Confirming changes nothing that runs; {program} apply does that."
        ),
        examples=(
            (
                "confirm --settings ~/rkmon-deploy.yaml",
                "the first time (deploy step 3)",
            ),
            ("confirm", "after you edit that file"),
            ("confirm --dry-run", "the summary and checks only"),
        ),
        exit=(
            "0 confirmed, or nothing to confirm; 1 you said no; 2 usage error, or no terminal; 3 an earlier step is "
            "not done; 4 a check failed, or refused."
        ),
        states=_CONFIRM_STATES,
        group="change",
        brief="record your settings, show them; confirm them",
    ),
    Row(
        "apply",
        "make the confirmed change on this deployment",
        options=(
            Option(
                ("--timeout-minutes",),
                lambda: f"minutes the work in flight may take (default: {drain_timeout_minutes():g})",
                metavar="N",
            ),
            Option(("--no-drain",), "stop the work in flight; it starts again later"),
            Option(("--dry-run",), "print the steps; run none of them"),
        ),
        details=_apply_details,
        examples=(
            ("apply", "make the change {program} confirm showed"),
            ("apply --dry-run", "show the steps; change nothing"),
        ),
        exit=(
            "0 applied, or nothing to apply; 1 rolled back, or an error; 2 usage error; 3 an earlier step is not done; "
            "4 refused; 130 interrupted."
        ),
        states=_APPLY_STATES,
        group="change",
        brief="make the confirmed change; put it back on failure",
    ),
    Row(
        "rollback",
        "go back to the code and settings before the last apply",
        options=(Option(("--dry-run",), "print the steps; run none of them"),),
        details=(
            "It puts back what the last {program} apply kept: the code, docker/.env, the override and your confirmed "
            "settings from before it, with apply's drain, restart and doctor; on a stopped container it puts them in "
            "place, and the container stays stopped. Your settings file and the package stay as they are: to stay on "
            "the earlier version, put your file back as it was, then {program} confirm."
        ),
        examples=(
            ("rollback", "go back to the version before the last apply"),
            ("rollback --dry-run", "show the steps; change nothing"),
        ),
        exit=(
            "0 restored, or nothing to put back; 1 an error; 2 usage error; 3 an earlier step is not done; 4 refused; "
            "130 interrupted."
        ),
        states=_CONTAINER_STATES,
        group="change",
        brief="go back to the code and settings before the last apply",
    ),
    Row(
        "status",
        "the deployment's state, then the monitor's status",
        options=(Option(("--queue",), "recompute the merge queue first (git work)"),),
        details=(
            "The deployment's state first: this folder's deployment, its container and the features your confirmed "
            "settings turn on; the state, with its fix; and the one command to run next, as {program} next ends. "
            "Then, while the container runs, rk status in it. While this folder's container name is another "
            "deployment's, or docker cannot say whose it is, nothing of the container is asked."
        ),
        examples=(("status", "the deployment's state and the monitor's"),),
        exit="0 always.",
        group="day-to-day",
        brief="the deployment's state, and the monitor's status",
        lock=NO_LOCK,
    ),
    Row(
        "logs",
        "the container's log",
        options=(
            Option(("-f", "--follow"), "follow it"),
            Option(
                ("--tail",), "only its last N lines (default: all of them)", metavar="N"
            ),
            Option(
                ("--since",),
                "only its lines since TIME: a time, such as 2026-11-15T09:00:00Z, or a span back from now, such as "
                "42m (default: all of them)",
                metavar="TIME",
            ),
        ),
        examples=(
            ("logs -f", "follow the log"),
            ("logs --tail 100", "its last 100 lines"),
            ("logs --since 1h", "its lines of the last hour"),
        ),
        exit="docker compose logs' code; 3 an earlier step is not done, such as the start; 4 refused.",
        states=_CONTAINER_STATES,
        container=EXISTS,
        group="day-to-day",
        form="logs [-f]",
        brief="the container's log (-f: follow it)",
        lock=NO_LOCK,
    ),
    Row(
        "pause",
        "start no new work; the work in flight finishes",
        details=(
            "rk pause in the running container: the monitor decides nothing and starts nothing new, and the work in "
            "flight finishes. {program} resume ends the pause; once the container is stopped, {program} start --on "
            "ends it as the container starts, and a start without --on keeps it."
        ),
        examples=(("pause", "start no new work"),),
        exit=(
            "0 paused, or nothing to do (the monitor is paused); 1 rk pause failed; 3 an earlier step is not done, "
            "such as the start; 4 refused: the container is stopped."
        ),
        states=_CONTAINER_STATES,
        container=RUNS,
        group="day-to-day",
        brief="start no new work; {program} resume starts it again",
    ),
    Row(
        "resume",
        "start new work again, after a pause",
        details="rk resume in the running container: the monitor decides and starts new work again.",
        examples=(("resume", "end the pause"),),
        exit=(
            "0 resumed, or nothing to do (the monitor is not paused); 1 rk resume failed; 3 an earlier step is not "
            "done, such as the start; 4 refused: the container is stopped."
        ),
        states=_CONTAINER_STATES,
        container=RUNS,
    ),
    Row(
        "stop",
        "stop the container",
        options=(
            Option(("--drain",), "let the work in flight finish first"),
            Option(
                ("--timeout-minutes",),
                lambda: f"with --drain, minutes the work in flight may take (default: {drain_timeout_minutes():g})",
                metavar="N",
            ),
        ),
        usage="[--drain [--timeout-minutes N]]",
        details=(
            "docker compose stop; {program} start brings it back. --drain first lets the work in flight finish, "
            "starting nothing new, for at most --timeout-minutes; then the monitor in the container stops, then the "
            "container. When it does not finish in time, nothing is stopped.\n\n"
            "Ctrl-C during the drain lifts it. Once the monitor or the container is stopping, Ctrl-C waits for that "
            "stop to end, then starts the monitor again while the container runs."
        ),
        examples=(
            ("stop", "stop the container"),
            ("stop --drain", "let the work in flight finish, then stop it"),
        ),
        exit=(
            "0 stopped, or nothing to do (it does not run); 1 docker could not stop it, rk drain or rk stop failed in "
            "the container, the work in flight did not finish in time, or another command lifted the drain; 2 "
            "usage error; 3 an earlier step is not done, such as the start; 4 refused; 130 interrupted."
        ),
        states=_CONTAINER_STATES,
        container=EXISTS,
        group="day-to-day",
        form="stop [--drain]",
        brief="stop the container (--drain: the work in flight finishes first)",
    ),
    Row(
        "start",
        steps.DEPLOY_STEPS[5].what,
        options=(
            Option(("--on",), "resume a monitor paused with {program} pause, too"),
        ),
        details=(
            "The first start puts your confirmed settings in place (docker/.env, the compose override and the settings "
            "volume), sets the work volume up from them (the clone, the virtual environments and the agentic files, "
            "with no question asked), and rk init seeds the monitor's state, the monitor on; a first start cut short "
            "is finished by the next. After it, a start starts what is in place "
            "and never applies a change: with a confirmed change waiting, it refuses and names {program} apply. A "
            "start puts this package's code in the container when the image's own code is not it (an image built "
            "from this package's image recipe may carry an older package's code), except after {program} rollback or "
            "an apply that rolled back: the code they put back runs. It "
            "refuses outside its states, such as settings you have not confirmed, no image or a login missing; an "
            "image that was not built from this package's image recipe; a container name, compose project or volume "
            "of another deployment; a busy port; and a private file in the package folder that is not its own. A "
            "stop or drain left behind is "
            "cleared, and a container that runs with one left in it is made again when code went in. A monitor "
            "paused with {program} pause stays paused, unless --on. While the container runs, it "
            "has nothing to do, but when it was started some other way and waits for its first start: it stops that "
            "container (nothing runs in it yet) and makes the first start."
        ),
        examples=(
            ("start", "deploy step 6, or after a stop"),
            ("start --on", "and end a pause"),
        ),
        exit=(
            "0 started, or nothing to do (the container runs); 1 a command on the way failed, or the container did "
            "not stay up; 2 usage error; 3 an earlier step is not done, such as the confirm, the image or a login; 4 "
            "refused, such as a change that waits for {program} apply, a busy port, or another deployment's "
            "container name, compose project or volume."
        ),
        states=_START_STATES,
        group="day-to-day",
        brief="start it again",
    ),
    Row(
        "login",
        "set each login and secret your settings need, or one login",
        args=(Arg("NAME", "one login, by name", optional=True),),
        options=(
            Option(
                ("--no-input",),
                "with no NAME: ask nothing and set nothing",
            ),
            Option(
                ("--with-token",),
                "a GitHub login's token, from stdin or typed without echo: github_write's in place of its browser "
                "flow; github_read takes one either way",
            ),
        ),
        usage="[--no-input | NAME [--with-token]]",
        details=(
            "With no NAME, it is deploy step 5: each login and secret something that is on may need, in the secrets "
            "registry's order, with the features that need it, its state and the command that sets it. What needs "
            "each one comes from your confirmed settings, read here, and its state from the volumes. One that only a "
            "feature in prototype would need is off, and its row says prototype; but one used meanwhile, the "
            "atlassian login while ticket creation is in prototype, is a warning while it is not set, and its row "
            "says what for. On a terminal, unless --no-input, it offers to set each missing one, then shows its row "
            "again; for an ssh key that is set while its host's own key is not "
            "recorded, as one the one-time move copies, it shows that key, and records it once you accept it. Its rk "
            "commands run in a one-off container of the image "
            "your confirmed settings name, handed those settings. Then, once the first start has made the container, "
            "come the doctor's findings on them, while it runs your confirmed settings; until you apply them, a line "
            "says that those findings come with {program} apply, as the container still runs your earlier ones. With "
            "NAME, it runs that login's own flow, in a one-off container; "
            "claude's then makes the login's test call, one message sent as the Claude session sends its requests, "
            "and shows its row. gh's write login and the atlassian CLI's run as the writer (in the running container "
            "when it is up) and stay in the writer's volume, which no other user reads. github_read takes a token "
            "alone, typed or from stdin, and refuses one that can write: the probe asks in a folder of its own before "
            "the read login changes. The outcome says when the monitor starts using the login."
        ),
        examples=(
            ("login", "deploy step 5"),
            ("login --no-input", "deploy step 5, asking nothing and setting nothing"),
            (
                "login github_read",
                "the read login: a token that can only read, typed or from stdin",
            ),
            ("login github_write --with-token", "the write login, from a token"),
        ),
        exit=(
            _WALK_CODES
            + "; with NAME: 0 the login is set; 1 its flow failed, no token typed, or claude's test call did not pass "
            "(the key is set); 4 no token on stdin, or one for github_read that can write. Refused: 3 "
            "an earlier step is not done, such as the image; else 4."
        ),
        states=_LOGIN_STATES,
        group="day-to-day",
        form="login NAME",
        brief="one login, by name ({program} login lists each)",
    ),
    Row(
        "secret",
        "set one secret, from FILE, stdin or typed; never shown",
        args=(
            Arg("NAME", "the secret's name"),
            Arg(
                "FILE",
                "a file of yours with its value; - reads stdin",
                optional=True,
            ),
        ),
        options=(
            Option(
                ("--generate",),
                "make its value here and show it once, on your terminal, for a secret the registry makes",
            ),
            Option(("--remove",), "remove its file"),
        ),
        usage="NAME [FILE | -] [--generate | --remove]",
        details=(
            "NAME is a secret the secrets registry lists ({program} login lists each), or the secret a model binding's "
            "api_key_secret_name names in your confirmed settings; a login's name, or an older name, is refused, with "
            "the command that sets it. Its value comes from FILE (yours only, outside the package and git), from "
            "stdin (-, or another command's output), or typed without echo; never from an option's value. It is never "
            "printed, nor its length. One the writer keeps (the Teams webhook, the web host's key) goes in as the "
            "writer: from here only, never from the container. Once the ssh key of a host is set (web_host_ssh_key, "
            "gpu_ssh_key), it shows the host's own keys as the container reads them, by their fingerprints, and "
            "records them once you accept them: ssh trusts the host by them alone."
        ),
        examples=(
            ("secret teams_webhook ~/webhook.txt", "from a file of yours"),
            ("secret admin_access_token --generate", "make it here; show it once"),
        ),
        exit=(
            "0 set, or removed; 1 no value typed, a host's own keys not accepted, or an error; 2 usage error; 3 an "
            "earlier step is not done, such as the image; 4 refused: the deployment's state, FILE is not yours only or "
            "lies inside the package or git, the value is empty, or rk secret-set refuses it."
        ),
        states=_LOGIN_STATES,
        group="day-to-day",
        form="secret NAME [FILE]",
        brief="one secret, from FILE or typed; never shown",
    ),
    Row(
        "standby off",
        "let a monitor in standby send again",
        details=(
            "rk standby-off in the running container, in state 14 only, where the monitor runs your confirmed "
            "settings in standby and sends nothing: it ends the one-time move from an older install, if one is open "
            "(the older install's config.json is renamed, read-only, and the older copies of the secrets are "
            'removed), and the monitor sends again, as your settings allow. QUICK_USER_GUIDE.md, "Standby", says '
            "when to run it. It reads nothing of any other deployment."
        ),
        examples=(("standby off", "the monitor sends again"),),
        exit=(
            "0 released, or nothing to do (the monitor is not in standby); 1 an error part-way, which says what was "
            "done; 3 an earlier step is not done, such as the start; 4 refused: the deployment's state, or the move "
            "from an older install cannot end yet."
        ),
        states=(14,),
        container=RUNS,
        group="day-to-day",
    ),
    Row(
        "shell",
        "a shell in the container, with the tools' settings",
        details="In the running container.",
        examples=(("shell", "a shell in the container"),),
        exit="the shell's code; 3 an earlier step is not done, such as the start; 4 refused: the container is stopped.",
        states=_CONTAINER_STATES,
        container=RUNS,
        group="day-to-day",
        brief="a shell in the container",
    ),
    Row(
        "rk",
        "run an rk command in the container",
        args=(Arg("ARGS", "the rk command and its arguments", rest=True),),
        details=(
            "In the running container. With --image-code before the command, rk runs the image's own code."
        ),
        examples=(
            ("rk pause", "start no new work"),
            ("rk resume", "start new work again"),
            ("rk features", "each feature, and whether it is on"),
        ),
        exit="rk's code; 3 an earlier step is not done, such as the start; 4 refused: the container is stopped.",
        states=_CONTAINER_STATES,
        container=RUNS,
        group="day-to-day",
        brief="any rk command, in the container",
    ),
    Row(
        "next",
        "where this deployment is on its path, and the command to run next",
        options=(
            Option(
                ("--settings",),
                "a settings file to judge too: a deploy settings file's steps 1 and 2, or a build settings file's "
                "build path (default: none)",
                metavar="FILE",
            ),
        ),
        details=(
            "Each step of the deploy path, DONE, TODO or REDO, and the one command to run next, from this folder's "
            "records and its own container. It changes nothing but one record: once it sees a tick since a start, "
            "this folder's records keep that deploy step 8 has passed, and from then on next lines name each "
            "command without its step."
        ),
        examples=(
            ("next", "where you are"),
            ("next --settings ~/rkmon-build.yaml", "the build path, for that file"),
        ),
        exit="0 always; 2 usage error.",
        lock=NO_LOCK,
    ),
    Row(
        "doctor",
        steps.DEPLOY_STEPS[6].what,
        options=(
            Option(
                ("--fresh",), "check each service again, not from the few-minute cache"
            ),
            Option(
                ("--fails",), "only the names of the checks that fail, one per line"
            ),
        ),
        details=(
            "Where this deployment is on the deploy path and its state, as a check row, then rk doctor in the "
            "container: each check, with its fix. Its result is kept on this host: {program} next reads it."
        ),
        examples=(
            ("doctor", "deploy step 7"),
            ("doctor --fresh", "each service checked again"),
        ),
        exit=(
            "0 every check passed; 1 rk doctor did not run to its end; 3 a prerequisite is missing, or an earlier step "
            "is not done, such as the start; 4 a check failed, or refused."
        ),
        states=_CONTAINER_STATES,
        container=EXISTS,
    ),
    Row(
        "first-tick",
        steps.DEPLOY_STEPS[7].what,
        options=(
            Option(
                ("--timeout-minutes",),
                f"how long to wait, in minutes (default: {FIRST_TICK_TIMEOUT_MINUTES})",
                metavar="N",
                default=FIRST_TICK_TIMEOUT_MINUTES,
            ),
        ),
        details=(
            "It waits for the first tick after the start, which it asks the tick loop for at once while none has "
            "run since the start, then shows rk status."
        ),
        examples=(
            ("first-tick", "deploy step 8"),
            ("first-tick --timeout-minutes 40", "wait 40 minutes at most"),
        ),
        exit=(
            "0 the first tick ran; 2 usage error; 3 an earlier step is not done, such as the start; 4 the container, "
            "or the monitor in it, is stopped, or no tick ran in time."
        ),
        states=_CONTAINER_STATES,
        container=RUNS,
    ),
    Row(
        "package update",
        "replace a package folder with a newer package",
        options=(
            Option(
                ("--from",),
                "the newer package: a folder, or a tarball the package build made; FOLDER.prev goes back to the "
                "package it replaced (default: the newest commit of the branch your settings file's updates section "
                "names)",
                metavar="DIR|TARBALL",
            ),
            Option(
                ("--folder",),
                "the package folder to update, such as an older install's, from this package's folder (default: "
                "this package folder)",
                metavar="FOLDER",
            ),
            Option(
                ("--dry-run",),
                "check the update and show what it carries; change nothing",
            ),
        ),
        details=(
            "It puts the newer package in the folder's place and keeps the folder it replaced beside it, as "
            "FOLDER.prev: the way back. Your docker/.env, the override, the host's records and the other local files "
            "of docker/ come along as they are. With no --from, git on this host fetches the package's folder alone, "
            "from the newest commit of the branch the updates section of your settings file names ({program} "
            "template section updates prints it).\n\n"
            "It says when the new package needs a new image. The new code runs once {program} confirm and {program} "
            "apply have put it in place: the next line names the command to run first."
        ),
        examples=(
            ("package update", "from the branch your settings file names"),
            ("package update --from ~/package.tgz", "from a newer package's tarball"),
            (
                "package update --from ~/rocke_monitor.prev",
                "back to the package it replaced",
            ),
            (
                "package update --from .. --folder ~/rocke_monitor",
                "an older install, from this package",
            ),
        ),
        exit=(
            "0 updated, or nothing to do; 1 an error, such as a fetch that failed; 2 usage error; 3 no --from and no "
            "updates section, or no git on this host; 4 refused, such as a private file in the new code; 130 "
            "interrupted."
        ),
        states=_PACKAGE_UPDATE_STATES,
        group="other",
        form="package update [--from DIR|TARBALL]",
        brief="this package folder, from a newer package",
        lock=EXCLUSIVE,
    ),
    Row(
        "migrate",
        "rewrite a settings file in this package's layout",
        options=(
            Option(
                ("--settings",),
                "your deploy or build settings file, in an earlier layout",
                metavar="FILE",
                required=True,
            ),
            Option(("--dry-run",), "print the report and the new file; write nothing"),
        ),
        details=(
            "It writes FILE again in place, in this package's layout: the template's comments, with your values. A "
            "feature that is off is written as its switch alone, unless a key under it has a value other than its "
            "default. The earlier file stays beside it as FILE.v<N>, N being its layout, yours only, with your own "
            "comments. The report says what moved, and each value dropped or reported.\n\n"
            f"A file with a secret_values section, or an admin dashboard address other than "
            f"{container_layout.LOOPBACK}, is refused, "
            "and nothing changes. It needs no docker and no deployment, and uses no network."
        ),
        examples=(
            ("migrate --settings ~/rkmon-deploy.yaml", "your deploy settings file"),
            ("migrate --settings ~/rkmon-build.yaml", "your build settings file"),
        ),
        exit="0 migrated, or nothing to migrate; 1 an error; 2 usage error; 4 refused (nothing changed).",
        group="other",
        brief="a settings file in an earlier layout, rewritten",
    ),
    Row(
        "adopt",
        "move an earlier monitor's state into this deployment",
        options=(
            Option(
                ("--from",),
                f"a copy of the earlier monitor's {container_layout.WORK_DIR} "
                f"(docker cp NAME:{container_layout.WORK_DIR} DIR)",
                metavar="DIR",
                required=True,
            ),
            Option(
                ("--map",),
                "a path of the existing monitor, and the one it becomes here; once for each path (default: none)",
                metavar="OLD=NEW",
                repeat=True,
            ),
            Option(("--dry-run",), "show the plan; apply nothing", name="plan"),
            Option(("--yes",), "apply the plan without asking"),
        ),
        details=(
            "Its state and its config.json are staged from DIR, its secrets, its locks and its Claude session left "
            "behind (a Claude conversation stays on the host that made it: the orchestrator here starts a new one), "
            "and standby set in it; the plan is shown, then applied, this deployment's own copies set aside. Paths "
            "are rewritten from both configurations and each --map. The next start splits that config.json by owner, "
            "once (the one-time move), and the deployment starts in standby: {program} standby off ends it, once the "
            "earlier monitor has stopped. It runs before the first start, or with the container stopped, and again, "
            "from the same copy, after an adopt that stopped part-way; it refuses while the container runs, and "
            "before deploy steps 1 to 4."
        ),
        examples=(
            ("adopt --from ~/monitor-copy --dry-run", "the plan only"),
            ("adopt --from ~/monitor-copy", "the plan, then the move"),
        ),
        exit=(
            "0 adopted, or the plan shown; 1 you said no, or an error; 2 usage error, or no terminal and no --yes. "
            "Refused, nothing changed: 3 an earlier step is not done, such as the image; else 4, among them the "
            "container running, a copy adopt refuses, and an apply stopped before anything moved."
        ),
        states=_ADOPT_STATES,
        group="other",
        form="adopt --from DIR",
        brief="a monitor's state, from a copy of an earlier one's",
    ),
    Row(
        "web-host install",
        "install the request service on the team dashboard's web host",
        details=(
            "With the team dashboard published to a web host, its priority box needs the request service there "
            "(tools/webhost_server.py), which serves the page and takes the requests other teams make on it. This "
            "installs it, or restarts it, on the web host your confirmed settings name, as a user service of the web "
            "host's login that keeps running after logout, over SSH with web_host_ssh_key, trusting the web host by "
            "its own keys alone, those recorded when you set the key. The writer, the key's keeper, runs it in a "
            "one-off container of the image your confirmed settings name, handed those settings, and reads the web "
            "host, its login and its paths from them alone. It first looks at what serves the page's port, and "
            "changes nothing while another program listens on it. In prototype it readies the web host for live. "
            "With the team dashboard served by this monitor, or the priority box off, there is nothing to install. "
            "In standby it is refused: the monitor you moved from may still use the web host. {program} doctor "
            'checks that the service answers. What the web host needs: README.md, "The team dashboard".'
        ),
        examples=(
            ("web-host install", "install the request service, or restart it"),
            ("web-host install --dry-run", "the docker command only; nothing runs"),
        ),
        exit=(
            "0 installed, or nothing to install; 1 an error part-way, the service not up among them; 2 usage error; "
            "3 an earlier step is not done, such as the web host's key; 4 refused, in standby among them, or a step "
            "the web host refused (nothing changed)."
        ),
        states=_WEB_HOST_STATES,
        group="other",
        brief="the request service, on the team dashboard's web host",
    ),
    Row(
        "image ls",
        "this package's images on this host, whatever their names",
        details=(
            "Each image built by this package's Dockerfile, with its labels: the package version, the build date, the "
            "base and ROCm."
        ),
        examples=(("image ls", "this package's images here"),),
        exit="0 always.",
        group="other",
        brief="this package's images on this host",
        lock=NO_LOCK,
    ),
    Row(
        "remove",
        "remove the container; its volumes stay, unless --delete-data",
        options=(
            Option(
                ("--delete-data",),
                "remove its volumes too: its state, its clone, its secrets and its logins, for good (asked first)",
            ),
            Option(("--yes",), "answer yes to that question"),
        ),
        details=(
            "docker compose down, and with --delete-data, docker volume rm of each volume of this deployment's. A "
            "volume of another deployment stays: one that another compose project made, or that a container of one "
            "mounts, as {program} start reads it. "
            "Before the first start there is no container, and there may be volumes: {program} login makes the ones "
            "that keep the logins, and a start cut short may make the others. With neither a container nor a volume "
            "of this deployment's on this host, it has nothing to do."
        ),
        examples=(
            ("remove", "remove the container; keep its data"),
            ("remove --delete-data", "and its data, asked first"),
        ),
        exit=(
            "0 removed, or nothing to do; 1 you said no, or docker could not remove it; 2 usage error, or no terminal "
            "and no --yes; 3 an earlier step is not done, such as the confirm; 4 refused."
        ),
        states=_REMOVE_STATES,
        group="other",
        form="remove [--delete-data]",
        brief="remove the container (and its data: asked first)",
    ),
)
PATHS = (
    Path("build-path", "The build path, on any machine", steps.BUILD_STEPS),
    Path(
        "deploy-path",
        "The deploy path, on the machine that runs it",
        steps.DEPLOY_STEPS,
        last="Where you are, and the next command: {program} next",
    ),
)
GROUPS = (
    Group("change", "Change a deployment"),
    Group("day-to-day", "Day to day"),
    Group("other", "Newer versions, other monitors, and removal", topic=False),
)
GONE = (
    Gone("preflight", "check --settings FILE"),
    Gone("settings --check", "check --settings FILE"),
    Gone("build --check", "check --settings FILE"),
    Gone("build --template", "template build FILE"),
    Gone("build --print", "build --settings FILE --dry-run"),
    Gone("settings --template", "template deploy [--profile PROFILE] FILE"),
    Gone("image import", "image load IMAGE_FILE"),
    Gone("image export", "image save IMAGE_FILE"),
    Gone("guide", "help [day-to-day]"),
    Gone("first-tick --timeout", "first-tick --timeout-minutes N"),
    Gone("settings", "confirm --settings FILE"),
    Gone(
        "setup",
        in_words="set up a deployment with your deploy settings file, {program} login and {program} secret, as "
        "{program} help deploy-path shows; with Claude Code, /rkmon-setup helps you write the file",
    ),
    Gone("down", "remove [--delete-data]"),
    Gone("logins --check", "login --no-input"),
    Gone("logins", "login"),
)
LEADING = (
    Option(
        ("--image-code",),
        "before the command: rk runs the image's own code, not the release {program} apply installed over it",
    ),
)
ABOUT = "build the rocKE monitor's image, then deploy and run its container."
ENVIRONMENT = (
    ("CONTAINER_ENGINE=podman", "run podman instead of docker"),
    ("NO_COLOR=1", "no dim echo of the docker commands"),
)
