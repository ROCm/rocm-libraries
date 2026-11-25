# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One row shape for every check, and the one way to print rows.

A check answers one question about one thing: its name, its status, what was found and, when something must change,
the fix.

- `Status`: PASS (as it should be); WARN (it works, and needs a look); FAIL (it must be fixed first); OFF (a feature
  that is off needs nothing here); SKIP (checked at a later step, which the row names). A status is its word:
  `str(Status.FAIL)` and `f"{Status.FAIL}"` are "FAIL", and `Status.FAIL == "FAIL"`. RAN: the statuses of the rows
  that ran, the ones an outcome counts (PASS, WARN and FAIL); PROBLEMS: those whose rows carry a fix (WARN and FAIL).
- A fix (`Fix`) is a command to run (`Command(argv)`), an edit that names the key, the file and the value
  (`Edit(key, file, value)`) or deletes the key (`Edit(key, file, DELETE)`), or steps of them in order
  (`Steps(parts)`). A fix is never empty and never words alone: any other fix, plain text among them, is refused.
  - A Command shows as a shell line, each word quoted as the shell needs; a word that starts with `~/` keeps it
    unquoted, so the shell puts your home folder there.
  - An Edit shows as `set KEY in FILE to VALUE`, the value written as JSON (a string in double quotes, a number as it
    is). A value only you know is `Described(words)`: the words say where it comes from, and show as they are
    (`set atlassian_cli.package_url in ~/rkmon-build.yaml to the address of the atlassian tool's Linux binary`). An
    edit whose value is DELETE shows as `delete KEY from FILE`. An edit that adds entries to a mapping KEY has the
    lines to add as its value, `Added(lines)`, and shows as `add under KEY in FILE:`, then each line on a line of its
    own, two spaces in (`add under features.ticket_creation.required_fields in ~/rkmon-deploy.yaml:`).
  - Steps hold Commands and Edits in order, at least one of them (ValueError otherwise), and, as text, the few words
    that join them; words after the last step may name the doc section that says more. The parts show joined by a
    space, but for none before words that start with `,` `.` `;` `:` `)` or a line break, and none after words that
    end with `(` or a line break (a command that starts with `./` keeps its space).
  - A fix names the commands it gives, as they show: `commands`, the Command itself, each Command of Steps, none for
    an Edit. `wrapping.wrap` keeps each of them that fits on a line whole.
- `fix_to_json(fix)` and `fix_from_json(data)`: a fix as JSON and back, for a file that keeps rows (rk doctor's cache
  of the service checks). A Command is `{"command": [WORD, ...]}`; an Edit is `{"edit": [KEY, FILE]}` with its value
  as `"value": VALUE`, `"described": WORDS`, `"added": [LINE, ...]` or `"delete": true`; Steps are
  `{"steps": [PART, ...]}`, each part a fix
  in this form or its words as a string; no fix is null. `fix_from_json(fix_to_json(fix))` is the same fix.
- `Check(name, status, found, fix=None)`: a row, checked as it is made (ValueError): a name and what was found, never
  empty; a status that is one of the five (its word is taken too); a fix that is a Fix, or None; a WARN and a FAIL
  carry their fix. A PASS, an OFF and a SKIP may carry one; it is never shown.
- `check_to_json(check)` and `check_from_json(data)`: a row as JSON and back, for a file or an answer that keeps rows
  (rk doctor's cache of the service checks, `rk doctor --json`): `{"name": NAME, "status": STATUS, "found": FOUND,
  "fix": FIX}`, its fix in fix_to_json's form. `check_from_json` makes the row as Check does, checked again
  (ValueError), and a form that is not a row's is a KeyError or a TypeError.
- `failed(checks)`: whether a row of them is a FAIL, which refuses what they check.
- `render(checks, rows="all", fixes=True)`: the rows as the terminal shows them, one after the other ("problems": only
  the WARN and FAIL rows, "" when there are none). A row is `  STATUS  NAME: FOUND`, the name at column 8, with a
  WARN's or a FAIL's fix below it as `        fix: FIX`, unless FIXES is false: the caller shows the fix elsewhere,
  after an outcome line, say. Lines are wrapped at WIDTH columns by `wrapping.wrap`, no word broken,
  and no quoted phrase or command that fits on a line split: the found text goes on at column 8, the fix at column
  13. A line break in the found text or the fix starts a line at that same column, which keeps its own indent after
  it:

      PASS  settings file: ~/rkmon-build.yaml is yours only, outside the package, git and cloud sync
      FAIL  alerts: features.alerts.develop_breaks is on, and features.alerts.mode is missing
            fix: run rkmon template section features.alerts, paste its text over features.alerts in ~/rkmon-deploy.yaml,
                 then set develop_breaks.enabled back to true and fill in each key whose comment says "Required when"

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import enum
import json
import shlex
from dataclasses import dataclass
from typing import Union

from . import wrapping

WIDTH = 117  # the longest a row's line gets (a word longer than a line goes on a line of its own)
NAME_COLUMN = 8  # where a row's name starts, and the found text's further lines
FIX_LABEL = "fix: "  # the words before a fix, in a row and after an outcome line
FIX_COLUMN = NAME_COLUMN + len(FIX_LABEL)  # where the fix's further lines start


class Status(str, enum.Enum):
    """A check's answer, shown as its word."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    OFF = "OFF"
    SKIP = "SKIP"

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec)


PROBLEMS = (
    Status.WARN,
    Status.FAIL,
)  # the statuses whose rows carry a fix, and the rows "problems" shows
RAN = (
    Status.PASS,
    *PROBLEMS,
)  # the statuses of the rows that ran: an OFF or a SKIP row checked nothing here


def _word(word: str) -> str:
    """A command's word as the shell needs it: quoted where it must be, with a leading ~/ left as it is."""
    if word.startswith("~/"):
        return "~/" + (shlex.quote(word[2:]) if word[2:] else "")
    return shlex.quote(word)


@dataclass(frozen=True)
class Command:
    """A fix that is a command to run: its words, the program first."""

    argv: tuple

    def __post_init__(self):
        argv = tuple(str(a) for a in self.argv)
        if not argv or not argv[0].strip():
            raise ValueError("a command names its program")
        object.__setattr__(self, "argv", argv)

    def __str__(self) -> str:
        return " ".join(_word(a) for a in self.argv)

    @property
    def commands(self) -> tuple:
        """The command, as it shows: a wrap keeps it whole when it fits on a line."""
        return (str(self),)


@dataclass(frozen=True)
class Described:
    """An edit's value that only you know: the words that say where it comes from."""

    words: str

    def __post_init__(self):
        object.__setattr__(self, "words", str(self.words).strip())
        if not self.words:
            raise ValueError("a described value says where the value comes from")

    def __str__(self) -> str:
        return self.words


ADDED_INDENT = "  "  # how far in an Added line shows, under its key


@dataclass(frozen=True)
class Added:
    """An edit's value that adds entries to a mapping: the lines to add under its key, each as the file is to have it,
    without the indent the key's entries take there."""

    lines: tuple

    def __post_init__(self):
        lines = tuple(str(x) for x in self.lines)
        if not lines or not all(x.strip() and "\n" not in x for x in lines):
            raise ValueError("the lines added under a key are lines of text")
        object.__setattr__(self, "lines", lines)

    def __str__(self) -> str:
        return "\n".join(ADDED_INDENT + x for x in self.lines)


class _Delete:
    """The value of an edit that deletes its key: there is one, DELETE."""

    def __repr__(self) -> str:
        return "DELETE"


DELETE = _Delete()  # an Edit's value that deletes the key: Edit(key, file, DELETE)


@dataclass(frozen=True)
class Edit:
    """A fix that is an edit: the key to set, the file it is in, and the value to give it (a Described one when only
    you know it), the lines to add under it (Added), or DELETE, which deletes the key.
    """

    key: str
    file: str
    value: object

    def __post_init__(self):
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "file", str(self.file))
        if not self.key.strip() or not self.file.strip():
            raise ValueError("an edit names its key and its file")

    def __str__(self) -> str:
        if self.value is DELETE:
            return f"delete {self.key} from {self.file}"
        if isinstance(self.value, Added):
            return f"add under {self.key} in {self.file}:\n{self.value}"
        value = (
            str(self.value)
            if isinstance(self.value, Described)
            else json.dumps(self.value, ensure_ascii=False)
        )
        return f"set {self.key} in {self.file} to {value}"

    @property
    def commands(self) -> tuple:
        """None: an edit gives no command."""
        return ()


NO_SPACE_BEFORE = (
    ",",
    ".",
    ";",
    ":",
    ")",
    "\n",
)  # words of Steps that start so join the part before them
NO_SPACE_AFTER = ("(", "\n")  # words of Steps that end so join the part after them


@dataclass(frozen=True)
class Steps:
    """A fix of steps: Commands and Edits in order, with the few words that join them and, after the last, the doc
    section that says more (text)."""

    parts: tuple

    def __post_init__(self):
        parts = tuple(self.parts)
        for p in parts:
            if not isinstance(p, (Command, Edit, str)):
                raise ValueError(
                    f"a step is a Command or an Edit, or the words that join steps, not {p!r}"
                )
        if not any(isinstance(p, (Command, Edit)) for p in parts):
            raise ValueError(
                "steps hold a Command or an Edit: a fix is never words alone"
            )
        object.__setattr__(self, "parts", parts)

    def __str__(self) -> str:
        out, joins = (
            "",
            False,
        )  # joins: the words before end so that the next part follows with no space
        for p in self.parts:
            words = isinstance(p, str)
            text = p.strip(" ") if words else str(p)
            if not text:
                continue
            if out and not (joins or (words and text.startswith(NO_SPACE_BEFORE))):
                out += " "
            out += text
            joins = words and text.endswith(NO_SPACE_AFTER)
        return out

    @property
    def commands(self) -> tuple:
        """Each Command of the steps, as it shows: a wrap keeps each whole that fits on a line."""
        return tuple(str(p) for p in self.parts if isinstance(p, Command))


Fix = Union[Command, Edit, Steps]


def fix_to_json(fix: Fix | None):
    """FIX in its JSON form (the module's docstring shows it); None for no fix."""
    if fix is None:
        return None
    if isinstance(fix, Command):
        return {"command": list(fix.argv)}
    if isinstance(fix, Edit):
        v = fix.value
        if v is DELETE:
            said = {"delete": True}
        elif isinstance(v, Described):
            said = {"described": v.words}
        elif isinstance(v, Added):
            said = {"added": list(v.lines)}
        else:
            said = {"value": v}
        return {"edit": [fix.key, fix.file], **said}
    return {"steps": [p if isinstance(p, str) else fix_to_json(p) for p in fix.parts]}


def fix_from_json(data) -> Fix | None:
    """The fix whose JSON form (fix_to_json) DATA is; None for none."""
    if data is None:
        return None
    if "command" in data:
        return Command(tuple(data["command"]))
    if "edit" in data:
        key, file = data["edit"]
        if data.get("delete"):
            value = DELETE
        elif "described" in data:
            value = Described(data["described"])
        elif "added" in data:
            value = Added(tuple(data["added"]))
        else:
            value = data.get("value")
        return Edit(key, file, value)
    return Steps(
        tuple(p if isinstance(p, str) else fix_from_json(p) for p in data["steps"])
    )


@dataclass(frozen=True)
class Check:
    """One check's row: what was checked, its status, what was found and, for a WARN or a FAIL, the fix."""

    name: str
    status: Status
    found: str
    fix: Fix | None = None

    def __post_init__(self):
        object.__setattr__(self, "status", Status(self.status))
        if not str(self.name).strip() or not str(self.found).strip():
            raise ValueError("a check row names what it checked and what it found")
        if self.fix is not None and not isinstance(self.fix, (Command, Edit, Steps)):
            raise ValueError(
                f"a fix is a Command, an Edit or Steps, never words alone: {self.fix!r}"
            )
        if self.fix is None and self.status in PROBLEMS:
            raise ValueError(f"a {self.status} row carries its fix: {self.name}")


def check_to_json(check: Check) -> dict:
    """CHECK in its JSON form (the module's docstring shows it)."""
    return {
        "name": check.name,
        "status": str(check.status),
        "found": check.found,
        "fix": fix_to_json(check.fix),
    }


def check_from_json(data) -> Check:
    """The row whose JSON form (check_to_json) DATA is, checked as Check checks a row it makes."""
    return Check(
        data["name"], data["status"], data["found"], fix_from_json(data.get("fix"))
    )


def failed(checks) -> bool:
    """Whether a row of CHECKS is a FAIL."""
    return any(c.status == Status.FAIL for c in checks)


def render(checks, rows: str = "all", fixes: bool = True) -> str:
    """The rows as the terminal shows them: every row ("all"), or the WARN and FAIL rows only ("problems"); each
    WARN's and FAIL's fix below it, unless FIXES is false."""
    if rows not in ("all", "problems"):
        raise ValueError(f'rows is "all" or "problems", not {rows!r}')
    lines = []
    for c in checks:
        if rows == "problems" and c.status not in PROBLEMS:
            continue
        lines += wrapping.wrap(
            f"{c.name}: {c.found}",
            WIDTH,
            f"  {c.status:<{NAME_COLUMN - 2}}",
            " " * NAME_COLUMN,
        )
        if fixes and c.status in PROBLEMS:
            lines += wrapping.wrap(
                c.fix, WIDTH, " " * NAME_COLUMN + FIX_LABEL, " " * FIX_COLUMN
            )
    return "\n".join(lines)
