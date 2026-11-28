# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One way to finish a command: one outcome line, a capitalized word and what happened, then one
`next:` or `done:` line, and the exit code that goes with the word.

- WORDS: each outcome word and its exit code. PASSED, NOTHING TO DO and the words of a thing done (WROTE, BUILT,
  SAVED, PUSHED, LOADED, PULLED, CONFIRMED, APPLIED, RESTORED, STARTED, STOPPED, PAUSED, RESUMED, SET, REMOVED,
  UPDATED, MIGRATED, ADOPTED, RELEASED, INSTALLED) exit 0; CANCELLED, ERROR and ROLLED BACK exit 1; FAILED and REFUSED
  exit 4, or 3 when what is missing is an earlier step (docker, buildx, a settings file, a confirm, an image, a login,
  a first start); INTERRUPTED exits 130. No other code is used: 2, a usage error, is the argument parser's.
- The words of a thing made or moved (WROTE, BUILT, SAVED, PUSHED, LOADED, PULLED) are followed by the thing:
  `BUILT rocke-monitor:2026.11.15`. Every other word by a colon: `PASSED: 9 checks`.
- `finish(word, text, next=None, fix=None, earlier_step=False, out=None) -> int`: prints the outcome line, the fix
  (`fix: …`, its further lines at column 5), then the next line (`next: …` or `done: …`, as steps and states make
  it), each at most as wide as a check row (WIDTH), the outcome's and the next line's further lines at column 0
  (rkmon_lib/wrapping.py); returns the exit code. A fix is a check row's: a checks.Command,
  Edit or Steps, never words alone. A word it does not know, `earlier_step` on a word other than FAILED or REFUSED, a
  fix of words alone, or a next line that is neither, is a ValueError: the caller's mistake, not the user's.
- `next_lines(next) -> [line]`: the next line NEXT as `finish` prints it: at most WIDTH columns, its further lines
  at column 0; one that starts with neither `next: ` nor `done: ` (steps.NEXT_LABEL, steps.DONE_LABEL) is a
  ValueError. A report that ends with a next line prints it so (rkmon_lib/progress.py).
- `Stopped(word, text, fix=None, earlier_step=False, rows=(), above="")`: a command that ends short, raised where it
  stops and finished where it is caught: its outcome word, what happened, the fix, whether what is missing is an
  earlier step, and what says why: the rows (checks.Check), or ABOVE, a refusal as it shows itself (the message of a
  private_files.Refused: its rows, then the line that names the rules). Its `finish(out=None, next=None) -> int`
  prints the rows as checks.render shows them, then ABOVE, then the outcome line, the fix and NEXT, the next line
  the caller gives, as `finish` does, and returns the exit code.
- `tally(checks, unchanged="nothing changed") -> (word, text)`: the outcome of a list of check rows: PASSED with how
  many checks and warnings, or FAILED with how many of them failed, then UNCHANGED, what the failure left as it was
  (`FAILED: 3 of 7 checks failed; nothing changed (fix each FAIL above)`; a confirm that recorded your file
  first: "nothing confirmed"). An OFF or SKIP row is no check that ran, and is not counted.
- AFTER_FAILS: the end of every outcome line that follows FAIL rows, what was left as it was in its {}, then that
  each row has its fix; FIX_EACH_FAIL is that end when nothing changed.
- EXIT_CODES: what each exit code means, in the words help prints. `exit_summary(held=False)`: the exit codes
  in one sentence, as `rkmon help` ends with them; HELD holds each code and its words together (wrapping.held), for
  a caller that wraps the sentence.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import sys
from typing import get_args

from . import steps, wrapping
from .checks import FIX_LABEL, WIDTH, Fix, Status, render

# The end of an outcome line that follows FAIL rows: what was left as it was ({}), and each row has its fix.
AFTER_FAILS = "{} (fix each FAIL above)"
FIX_EACH_FAIL = AFTER_FAILS.format("nothing changed")
DONE_WORDS = (
    "WROTE",
    "BUILT",
    "SAVED",
    "PUSHED",
    "LOADED",
    "PULLED",
    "CONFIRMED",
    "APPLIED",
    "RESTORED",
    "STARTED",
    "STOPPED",
    "PAUSED",
    "RESUMED",
    "SET",
    "REMOVED",
    "UPDATED",
    "MIGRATED",
    "ADOPTED",
    "RELEASED",
    "INSTALLED",
)
WORDS = {
    "PASSED": 0,
    "NOTHING TO DO": 0,
    **{w: 0 for w in DONE_WORDS},
    "CANCELLED": 1,
    "ERROR": 1,
    "ROLLED BACK": 1,
    "FAILED": 4,
    "REFUSED": 4,
    "INTERRUPTED": 130,
}
OBJECT_WORDS = ("WROTE", "BUILT", "SAVED", "PUSHED", "LOADED", "PULLED")
EARLIER_STEP = 3  # FAILED or REFUSED because an earlier step is not done
USAGE = 2  # a usage error: the argument parser's code
# (code, its words in one sentence, its words in a list)
EXIT_CODES = (
    (0, "done", "done, or nothing to do"),
    (
        1,
        "an error, a roll back, or you said no",
        "an error part-way, a roll back, or you said no",
    ),
    (
        2,
        "usage error",
        "a usage error, including a question with no terminal and no --yes",
    ),
    (
        3,
        "an earlier step is not done",
        "an earlier step is not done (docker, buildx, a settings file, a confirm, an image, a login, a first start)",
    ),
    (
        4,
        "refused or a check failed (nothing changed)",
        "refused, or a check failed, and nothing changed",
    ),
    (130, "interrupted", "interrupted"),
)


def line(word: str, text: str = "") -> str:
    """The outcome line: the word, then what happened."""
    if word not in WORDS:
        raise ValueError(f"no outcome word {word!r}")
    if not str(text).strip():
        return word
    return f"{word} {text}" if word in OBJECT_WORDS else f"{word}: {text}"


def code(word: str, earlier_step: bool = False) -> int:
    """The exit code of an outcome word: 3 for FAILED or REFUSED when an earlier step is not done."""
    if word not in WORDS:
        raise ValueError(f"no outcome word {word!r}")
    if earlier_step:
        if word not in ("FAILED", "REFUSED"):
            raise ValueError(
                f"{word} is never an earlier step's: only FAILED and REFUSED are"
            )
        return EARLIER_STEP
    return WORDS[word]


def fix_lines(fix: Fix) -> list[str]:
    """A fix as its lines: `fix: …` (a check row's FIX_LABEL), at most WIDTH columns (a check row's width), its
    further lines under the fix's first word. A fix of words alone is a ValueError."""
    if not isinstance(fix, get_args(Fix)):
        raise ValueError(
            f"a fix is a Command, an Edit or Steps, never words alone: {fix!r}"
        )
    return wrapping.wrap(fix, WIDTH, FIX_LABEL, " " * len(FIX_LABEL))


def next_lines(next) -> list[str]:
    """The next line NEXT as its lines, as `finish` prints it: see the module's docstring."""
    labels = (steps.NEXT_LABEL, steps.DONE_LABEL)
    if not str(next).startswith(labels):
        said = " or ".join(label.strip() for label in labels)
        raise ValueError(f"a next line starts with {said}, not {next!r}")
    return wrapping.wrap(str(next), WIDTH)


def finish(
    word: str, text: str = "", next=None, fix=None, earlier_step: bool = False, out=None
) -> int:
    """Print the outcome line, the fix if any, and the next line, each wrapped at WIDTH; return the exit code."""
    after = [] if next is None else next_lines(next)
    rc = code(word, earlier_step)
    lines = wrapping.wrap(line(word, text), WIDTH)
    if fix is not None:
        lines += fix_lines(fix)
    lines += after
    out = sys.stdout if out is None else out
    out.write("\n".join(lines) + "\n")
    out.flush()
    return rc


class Stopped(Exception):
    """A command that ends short: see the module's docstring."""

    def __init__(
        self,
        word: str,
        text: str,
        fix=None,
        earlier_step: bool = False,
        rows=(),
        above: str = "",
    ):
        super().__init__(text)
        self.word, self.text, self.fix = word, text, fix
        self.earlier_step, self.rows, self.above = earlier_step, tuple(rows), above

    def finish(self, out=None, next=None) -> int:
        """Print the rows and ABOVE, then the outcome line, the fix and NEXT; return the exit code: see the module's
        docstring."""
        out = sys.stdout if out is None else out
        if self.rows:
            out.write(render(self.rows) + "\n")
        if self.above:
            out.write(self.above + "\n")
        return finish(
            self.word,
            self.text,
            next=next,
            fix=self.fix,
            earlier_step=self.earlier_step,
            out=out,
        )


def tally(checks, unchanged: str = "nothing changed") -> tuple:
    """(word, text) for the outcome of check rows: PASSED with the counts, or FAILED with how many failed and
    UNCHANGED, what the failure left as it was."""
    rows = list(checks)
    failed = sum(1 for c in rows if c.status == Status.FAIL)
    warned = sum(1 for c in rows if c.status == Status.WARN)
    ran = [c for c in rows if c.status in (Status.PASS, Status.WARN, Status.FAIL)]
    if failed:
        return (
            "FAILED",
            f"{failed} of {len(ran)} checks failed; {AFTER_FAILS.format(unchanged)}",
        )
    text = wrapping.counted(len(ran), "check")
    if warned:
        text += f" ({wrapping.counted(warned, 'warning')})"
    return "PASSED", text


def exit_summary(held: bool = False) -> str:
    """The exit codes in one sentence, as rkmon help ends with them; HELD: each code held with its words."""
    items = (f"{n} {short}" for n, short, _ in EXIT_CODES)
    return "; ".join(wrapping.held(i) if held else i for i in items) + "."
