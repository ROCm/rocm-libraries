# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Conservative CLI-flag extraction from the ingestor-engine skill's markdown.

The sibling `skill_paths.py` makes "every path the skill cites resolves" a
permanent check. This is the same idea for the other half of a copy-pasteable
instruction: **every FLAG the skill cites exists in the tool's argparse.**

The failure this prevents has already happened twice on this branch. `extend.md`
told an agent to run `variant_reachability.py ... --score`, a flag that has never
existed (the real ones are `--score-field` / `--score-prefer`), and RUNBOOK step
4a-3 documented a `dispatch_parity.py` capability the tool did not have until
`--knobs` was added. A wrong flag is worse than a wrong path: argparse exits 2 with
a usage string, which reads as "the agent typed it wrong" rather than "the document
is lying", so it gets worked around instead of fixed.

WHY THE EXTRACTOR IS THIS FUSSY. A flag token means nothing on its own -- the skill
legitimately cites `--expect-engine` (a C++ binary's flag), `-DGPU_TARGETS`,
`ctest -L`, `git log --all`. So a flag is only checked when it is bound to a
KNOWN PYTHON TOOL, and binding means: the same shell command names that tool.
Specifically, inside a fenced code block, with backslash continuations joined,
we take the last `<name>.py` token on the logical line and check the flags that
follow it. Everything else -- prose, other binaries, flags appearing before any
tool name -- is deliberately out of scope.

That rule has one consequence worth stating, because it looks like a gap: a flag
cited in PROSE is not checked. `SKILL.md` writes out
``hipdnn_validate_descriptors ... [--json]`` in prose near a sentence mentioning
`generate.py`, and a proximity-based extractor flags it as `generate.py --json` --
a false positive on a correct document. A check that cries wolf gets deleted, so
the binding is syntactic rather than proximate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: A fenced code block, whatever the info string.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

#: `add_argument("--foo"` -- the declaration form every tool in this repo uses.
#: Single-quoted spellings are absent from the tree today (asserted by a test),
#: so the pattern stays narrow rather than guessing at a style nobody writes.
_ADD_ARGUMENT_RE = re.compile(r"""add_argument\(\s*["'](--[a-z0-9][a-z0-9-]*)["']""")

#: A long flag as it appears in a command line. Deliberately long-form only:
#: short flags are ambiguous across binaries and carry no signal here.
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")

#: A python tool invocation, e.g. `dispatch_parity.py` or `$GEN/tools/x.py`.
_TOOL_RE = re.compile(r"([a-z_][a-z0-9_]*\.py)\b")

#: A shell line's trailing backslash continuation.
_CONTINUATION_RE = re.compile(r"\\\s*\n\s*")


@dataclass(frozen=True)
class CitedFlag:
    file: str  # skill markdown file the citation was found in
    line: int  # 1-indexed line of the fenced block the command sits in
    tool: str  # the .py basename the flag was bound to
    flag: str  # the long flag as cited
    command: str  # the logical command line, for a readable failure message


def tool_flags(tool_path: Path) -> set[str]:
    """Every long flag `tool_path`'s argparse declares."""
    return set(_ADD_ARGUMENT_RE.findall(tool_path.read_text()))


def discover_tools(*roots: Path) -> dict[str, Path]:
    """`{basename: path}` for every python tool the skill could cite.

    A dict keyed on basename because that is how the skill writes them --
    `$GEN/tools/dispatch_parity.py` and a bare `dispatch_parity.py` are the same
    tool, and the prefix is a variable the reader substitutes.
    """
    found: dict[str, Path] = {}
    for root in roots:
        if root.is_file():
            found[root.name] = root
            continue
        for path in sorted(root.glob("*.py")):
            found[path.name] = path
    return found


def extract_cited_flags(text: str, filename: str, known: set[str]) -> list[CitedFlag]:
    """Every (tool, flag) pair the fenced commands in `text` bind together.

    `known` is the set of tool basenames to bind against; a `.py` file the caller
    does not know about is ignored rather than guessed at, so an unrelated script
    named in an example cannot manufacture failures.
    """
    cited: list[CitedFlag] = []
    for block in _FENCE_RE.finditer(text):
        block_line = text[: block.start()].count("\n") + 1
        joined = _CONTINUATION_RE.sub(" ", block.group(1))
        for raw in joined.split("\n"):
            command = raw.strip()
            if not command or command.startswith("#"):
                continue
            # The LAST tool named on the line owns the flags after it: a line
            # like `foo.py --a | bar.py --b` binds --b to bar.py, and taking the
            # first would attribute it to foo.py.
            matches = [m for m in _TOOL_RE.finditer(command) if m.group(1) in known]
            if not matches:
                continue
            owner = matches[-1]
            for flag in _FLAG_RE.findall(command[owner.end() :]):
                cited.append(
                    CitedFlag(
                        file=filename,
                        line=block_line,
                        tool=owner.group(1),
                        flag=flag,
                        command=command,
                    )
                )
    return cited
