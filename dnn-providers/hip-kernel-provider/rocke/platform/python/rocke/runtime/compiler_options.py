# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which LLVM backend options (``-mllvm <option>``) each compiler accepts.

A ``-mllvm`` option is handed to LLVM's own command-line parser inside COMGR.
That parser does not report an option it does not know as an error the caller
can catch: it prints ``Unknown command line argument`` and exits the whole
host process. So the only safe time to find out is before COMGR is called.

The answers are measured, never assumed from a version number:
``tools/gen_compiler_options.py`` compiles an empty kernel through the loaded
COMGR in a child process, once per option, and records in
``data/compiler_options.json`` whether that compiler accepted it. A host can
only measure its own compiler, so the table has one column per LLVM major
version measured. This module is the only reader and writer of that file's
format (:func:`parse_table`, :func:`format_table`); the tool only measures.

    data/compiler_options.json
      options
        "-amdgpu-mfma-vgpr-form"
          "20": accepted false   (LLVM 20.0.0)  evidence: Unknown command ...
          "22": accepted true    (LLVM 22.0.0)

A compile that needs an option the loaded compiler does not accept raises
:class:`CompilerOptionNotAccepted`. Leaving the option out instead would
compile a different kernel from the one asked for, so it is never done here.
A compiler or an option with no measured answer raises ``ValueError``: the
answer is unknown, and guessing it wrong ends the process.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .comgr import CompilerInfo

__all__ = [
    "SCHEMA",
    "TABLE_PATH",
    "Answer",
    "CompilerOptionNotAccepted",
    "backend_options",
    "column_drift",
    "format_table",
    "lookup",
    "measured_answers",
    "parse_table",
    "require_accepted",
]

SCHEMA = "rocke.compiler_options/v1"

# Anchored on __file__ so it resolves in an installed wheel (pyproject.toml
# packages every ``**/*.json``).
TABLE_PATH = Path(__file__).parent / "data" / "compiler_options.json"

# The tool that measures the table, for the error messages.
_MEASURE_TOOL = "tools/gen_compiler_options.py"


class CompilerOptionNotAccepted(ValueError):
    """The loaded compiler does not accept a ``-mllvm`` option a compile needs."""


class Answer(NamedTuple):
    """One measured (option, LLVM major) result.

    ``evidence`` is the compiler's first diagnostic line when it refused the
    option, empty when it accepted it. ``verified_on`` names the exact
    compiler the answer was measured on.
    """

    accepted: bool
    evidence: str
    verified_on: str


def backend_options(options: Sequence[str]) -> tuple[str, ...]:
    """The LLVM backend option names in a COMGR option list, in order.

    COMGR takes a backend option in two spellings, ``-mllvm <option>`` and
    ``-mllvm=<option>``; both reach LLVM's option parser, so both are read.
    ``["-O3", "-mllvm", "-amdgpu-mfma-vgpr-form=1"]`` and
    ``["-O3", "-mllvm=-amdgpu-mfma-vgpr-form=1"]`` both give
    ``("-amdgpu-mfma-vgpr-form",)``: the value after the option's own ``=`` is
    not part of what the parser has to know.

    COMGR reads each option as a C string, so it loses any text after a NUL
    character, and the names read here could differ from the ones COMGR sees
    (``"-mllvm\\0"`` reaches it as ``-mllvm``). A list with a NUL in any option
    raises ``ValueError``.
    """
    with_nul = [option for option in options if "\0" in option]
    if with_nul:
        raise ValueError(
            f"COMGR options {with_nul} contain a NUL character; COMGR reads each "
            "option as a C string and would see only the text before it: "
            f"{list(options)}"
        )
    names = []
    for i, option in enumerate(options):
        if option.startswith("-mllvm="):
            value = option[len("-mllvm=") :]
        elif option == "-mllvm":
            if i + 1 >= len(options):
                raise ValueError(
                    f"'-mllvm' is the last COMGR option; it needs a value: {list(options)}"
                )
            value = options[i + 1]
        else:
            continue
        if not value:
            raise ValueError(f"'-mllvm=' names no option: {list(options)}")
        names.append(value.split("=", 1)[0])
    return tuple(names)


def _parse_answer(cell: object, *, where: str) -> Answer:
    """One ``{accepted, evidence, verified_on}`` cell, with its types checked.

    The types are checked rather than coerced: ``bool("false")`` is ``True``,
    so a hand-edited ``"accepted": "false"`` would otherwise read as accepted.
    A wrong type here is a bad value in the file, not a bad argument, so it is
    a ``ValueError`` like every other table error (hence the ``noqa``).
    """
    if not isinstance(cell, dict):
        raise ValueError(  # noqa: TRY004
            f"{where} is {cell!r}, expected an object with the keys "
            f"{', '.join(Answer._fields)}"
        )
    missing = [key for key in Answer._fields if key not in cell]
    unexpected = [key for key in cell if key not in Answer._fields]
    if missing or unexpected:
        raise ValueError(
            f"{where} has missing keys {missing} and unexpected keys {unexpected}; "
            f"expected exactly {', '.join(Answer._fields)}"
        )
    answer = Answer(**cell)
    if not isinstance(answer.accepted, bool):
        raise ValueError(  # noqa: TRY004
            f"{where} accepted={answer.accepted!r}, expected true or false"
        )
    if not isinstance(answer.evidence, str) or not isinstance(answer.verified_on, str):
        raise ValueError(  # noqa: TRY004
            f"{where} evidence={answer.evidence!r} and "
            f"verified_on={answer.verified_on!r}, expected strings"
        )
    return answer


def parse_table(
    text: str, *, source: str | os.PathLike[str] = TABLE_PATH
) -> dict[str, dict[int, Answer]]:
    """Read a ``compiler_options.json`` document: option -> LLVM major -> answer.

    ``source`` names where ``text`` came from (by default the committed
    table). Raises ``ValueError`` naming ``source`` and the offending entry
    when the text is not JSON, is not a :data:`SCHEMA` table, or an entry has
    the wrong shape or type.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: not valid JSON: {exc}") from exc
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema != SCHEMA:
        raise ValueError(
            f"{source}: schema={schema!r}, expected {SCHEMA!r}; "
            f"regenerate it with {_MEASURE_TOOL}"
        )
    options = doc.get("options")
    if not isinstance(options, dict):
        raise ValueError(  # noqa: TRY004
            f"{source}: options={options!r}, expected an object "
            "mapping each option to its measured LLVM majors"
        )
    answers: dict[str, dict[int, Answer]] = {}
    for option, column in options.items():
        if not isinstance(column, dict):
            raise ValueError(  # noqa: TRY004
                f"{source}: {option!r} is {column!r}, expected an object keyed "
                "by LLVM major version"
            )
        not_majors = [key for key in column if not key.isdigit()]
        if not_majors:
            raise ValueError(
                f"{source}: {option!r} has the keys {not_majors}, expected LLVM "
                'major versions such as "20"'
            )
        answers[option] = {
            int(major): _parse_answer(
                cell, where=f"{source}: {option!r} on LLVM {major}"
            )
            for major, cell in column.items()
        }
    return answers


def format_table(answers: Mapping[str, Mapping[int, Answer]]) -> str:
    """The ``compiler_options.json`` text for ``answers``, as :func:`parse_table`
    reads it: options in the given order, LLVM majors ascending."""
    doc = {
        "schema": SCHEMA,
        "options": {
            option: {
                str(major): answer._asdict() for major, answer in sorted(column.items())
            }
            for option, column in answers.items()
        },
    }
    return json.dumps(doc, indent=2) + "\n"


def column_drift(
    committed: Mapping[str, Mapping[int, Answer]],
    measured: Mapping[str, bool],
    *,
    llvm_major: int,
) -> list[str]:
    """How the ``llvm_major`` column of ``committed`` differs from ``measured``.

    ``measured`` maps each option to whether the loaded compiler (LLVM
    ``llvm_major``) accepted it. Returns one line per measured option whose
    committed answer for that major is missing or says the opposite, in
    ``measured`` order; an empty list means the column reproduces.
    ``tools/gen_compiler_options.py --check`` fails exactly when it is
    non-empty.
    """
    drift = []
    for option, accepted in measured.items():
        answer = committed.get(option, {}).get(llvm_major)
        if answer is None:
            drift.append(f"{option} on LLVM {llvm_major}: no committed answer")
        elif answer.accepted != accepted:
            drift.append(
                f"{option} on LLVM {llvm_major}: committed accepted={answer.accepted}, "
                f"measured accepted={accepted}"
            )
    return drift


@functools.cache
def _table() -> dict[str, dict[int, Answer]]:
    return parse_table(TABLE_PATH.read_text(encoding="utf-8"))


def measured_answers() -> dict[str, dict[int, Answer]]:
    """Every committed answer, option -> LLVM major -> :class:`Answer`, in file
    order. The dictionaries are a fresh copy the caller may change."""
    return {option: dict(column) for option, column in _table().items()}


def lookup(option: str, llvm_major: int) -> Answer | None:
    """The measured answer for ``option`` on LLVM ``llvm_major``, or ``None``
    when that pair was never measured."""
    return _table().get(option, {}).get(llvm_major)


def require_accepted(
    options: Sequence[str], *, compiler: CompilerInfo, needed_by: str
) -> None:
    """Raise unless ``compiler`` accepts every ``-mllvm`` option in ``options``.

    ``needed_by`` names what is being compiled (for example
    ``"kernel 'gemm_f16'"``) so the error says which kernel needs the option.
    Raises :class:`CompilerOptionNotAccepted` when the table records that the
    compiler refuses an option, and ``ValueError`` when the table has no answer
    for the option or for this compiler's LLVM major version, or when
    ``compiler`` has no LLVM version. For the loaded compiler, call
    :func:`rocke.runtime.comgr.require_options_accepted` instead: it raises
    ``ComgrError`` before this when that compiler's version cannot be queried.
    """
    names = backend_options(options)
    if not names:
        return
    if compiler.llvm_version is None:
        raise ValueError(
            f"{needed_by} needs the compiler option(s) {list(names)}, and the loaded "
            f"compiler's LLVM version is unknown ({compiler.describe()}), so "
            "whether it accepts them cannot be looked up"
        )
    major = compiler.llvm_version[0]
    for name in names:
        if name not in _table():
            raise ValueError(
                f"{needed_by} needs the compiler option '-mllvm {name}', which has "
                "never been measured on any compiler. Measure it first: "
                f"python {_MEASURE_TOOL} --option={name}"
            )
        answer = lookup(name, major)
        if answer is None:
            raise ValueError(
                f"{needed_by} needs the compiler option '-mllvm {name}', which has "
                f"not been measured on LLVM {major} ({compiler.describe()}). Measure "
                f"it with this compiler loaded: python {_MEASURE_TOOL}"
            )
        if not answer.accepted:
            accepting = sorted(
                major for major, cell in _table()[name].items() if cell.accepted
            )
            raise CompilerOptionNotAccepted(
                f"{needed_by} needs the compiler option '-mllvm {name}', which the "
                f"loaded compiler does not accept ({compiler.describe()}; measured "
                f"on {answer.verified_on}: {answer.evidence.rstrip('.')}). Compiling without "
                "it would build a different kernel, so it is not compiled. Load a "
                "compiler that accepts it (measured: LLVM "
                f"{', '.join(map(str, accepting)) or 'none'}), or change the kernel "
                "so it does not need it."
            )
