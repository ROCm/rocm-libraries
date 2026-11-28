# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One way to wrap plain English and to join words: for help, for an outcome's fix, for messages and for the comments
of a settings file.

- `wrap(text, width, first="", rest="") -> list[str]`: `text` in lines of at most `width` columns, the first line
  after `first`, the others after `rest`. No word is broken and no line breaks at a hyphen: a word longer than a line
  goes on a line of its own. A quoted phrase that fits on a line, such as "Change a setting", is never split, and
  neither is a command that fits on a line, so that a line copied runs the whole command: where the line would pass
  the width, the command moves whole to the next line. A quoted phrase or a command longer than a line wraps as its
  words do. The commands are those the text names as its `commands`, as a fix does (rkmon_lib/checks.py), and in a
  text that names none, such as a next line, those of the program (`program_commands`). A line break in the text
  starts a new line after `rest`, and that line keeps its own indent after it. An empty text, or an empty line of it,
  is its lead alone, without trailing spaces.
- `program_commands(text) -> list[tuple[int, int]]`: where each command of the program in `text` starts and ends: the
  program's name or a path to it (PROGRAM_WORD; a `(` before it is no part of it), then each word up to one that
  starts with `(` or joins steps (JOINS), and through one that ends with a mark of COMMAND_END, which is no part of
  it. tools/doc_lints.py reads the commands of a doc's quote the same way.
- HOLD: a no-break space (U+00A0). Words joined by it stay on one line, and it prints as a space: a caller holds an
  item together with it, such as each exit code and its words in `rkmon help`.
- `marked(items, mark="·") -> str`: the items joined by " MARK ", each held together with the mark before it
  (HOLD), so that `wrap` breaks the line only before a mark, which starts the next line: the features' line of deploy
  steps 2 and 3.
- `joined(items, word="and") -> str`: the items in a sentence: "a", "a and b", "a, b and c"; WORD in place of "and"
  ("a, b or c"). No item is a ValueError.
- `are(items) -> str`: the verb after the items joined: "is" for one item, "are" for more ("a is", "a and b are").
- `counted(n, one, many=None) -> str`: a number and its noun: "1 entry", "2 entries"; MANY is the noun for any number
  but 1, ONE with an s when it is not given ("0 checks").

Standard library and rkmon_lib/program.py only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the
host.
"""

from __future__ import annotations

import re
import textwrap

from . import program

HOLD = "\u00a0"
QUOTED = re.compile(r'"[^"\n]*"')
# The marks that end a command of the program after its last word: no part of the command.
COMMAND_END = ",;:.)"
# The words that join the steps of a fix or a next line, which no command of the program takes.
JOINS = ("and", "or", "then")
# A word that is the program: a ( before it, its name or a path that ends with it (docker/rkmon), and the marks that
# end a command (COMMAND_END) after it, if any.
PROGRAM_WORD = re.compile(
    rf"(\(?)((?:\S*/)?{re.escape(program.NAME)})([{re.escape(COMMAND_END)}]*)"
)


def held(text: str) -> str:
    """`text` with its spaces made no-break spaces, so that it stays on one line."""
    return str(text).replace(" ", HOLD)


def program_commands(text: str) -> list:
    """Where each command of the program in TEXT starts and ends: see the module's docstring."""
    out, words, i = [], list(re.finditer(r"\S+", text)), 0
    while i < len(words):
        name = PROGRAM_WORD.fullmatch(words[i].group())
        i += 1
        if not name:
            continue
        start = words[i - 1].start() + len(name.group(1))
        end = start + len(name.group(2))
        while i < len(words) and not name.group(3):
            word = words[i].group()
            if word.startswith("(") or word in JOINS:
                break
            core = word.rstrip(COMMAND_END)
            if core:
                end = words[i].start() + len(core)
            i += 1
            if core != word:
                break
        out.append((start, end))
    return out


def _commands(text, line: str) -> list:
    """Where each command in LINE, a line of TEXT, starts and ends: each that TEXT names as its `commands`, where it
    stands as words of their own (after a space or a `(`, before a space or a mark of COMMAND_END); in a text that names
    none, each of the program (program_commands)."""
    named = getattr(text, "commands", None)
    if named is None:
        return program_commands(line)
    found = []
    for command in named:
        alone = rf"(?<![^\s(]){re.escape(command)}(?![^\s{re.escape(COMMAND_END)}])"
        found += [m.span() for m in re.finditer(alone, line)]
    return found


def _moved(line: str, start: int, end: int) -> int:
    """How long the part of LINE between spaces is that holds START to END: what the wrap moves with a command, the
    marks and the ( next to it too."""
    before = re.search(r"[^ \t]*$", line[:start]).group()
    after = re.match(r"[^ \t]*", line[end:]).group()
    return len(before) + end - start + len(after)


def wrap(text, width: int, first: str = "", rest: str = "") -> list[str]:
    """`text`, a string or a text that names its `commands`, in lines of at most `width` columns: see the module's
    docstring."""
    out = []
    for i, part in enumerate(str(text).split("\n")):
        words = part.lstrip(" ")
        own = part[: len(part) - len(words)]
        lead = first if i == 0 else rest + own
        more = rest if i == 0 else rest + own
        room = width - len(more)
        for start, end in _commands(text, words):
            if _moved(words, start, end) <= room:
                words = words[:start] + held(words[start:end]) + words[end:]
        words = QUOTED.sub(
            lambda m: held(m.group(0)) if len(m.group(0)) <= room else m.group(0),
            words,
        )
        lines = textwrap.wrap(
            words,
            width,
            initial_indent=lead,
            subsequent_indent=more,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out += [line.replace(HOLD, " ") for line in lines] or [lead.rstrip()]
    return out


def marked(items, mark: str = "·") -> str:
    """The items joined by " MARK ", each held together with the mark before it: see the module's docstring."""
    return " ".join(
        held(item if n == 0 else f"{mark} {item}") for n, item in enumerate(items)
    )


def joined(items, word: str = "and") -> str:
    """The items in a sentence: "a", "a and b", "a, b and c"."""
    items = [str(i) for i in items]
    if not items:
        raise ValueError("joined names at least one item")
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {word} {items[-1]}"


def are(items) -> str:
    """The verb after the items joined: "is" for one, "are" for more."""
    return "is" if len(items) == 1 else "are"


def counted(n: int, one: str, many: str | None = None) -> str:
    """A number and its noun: "1 entry", "2 entries"."""
    return f"{n} {one if n == 1 else many or one + 's'}"
