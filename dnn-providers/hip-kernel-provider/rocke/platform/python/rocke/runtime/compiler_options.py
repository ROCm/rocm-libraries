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
version measured. This module is the only reader.

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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .comgr import CompilerInfo

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


def backend_options(options: Sequence[str]) -> tuple:
    """The LLVM backend option names in a COMGR option list, in order.

    COMGR takes a backend option in two spellings, ``-mllvm <option>`` and
    ``-mllvm=<option>``; both reach LLVM's option parser, so both are read.
    ``["-O3", "-mllvm", "-amdgpu-mfma-vgpr-form=1"]`` and
    ``["-O3", "-mllvm=-amdgpu-mfma-vgpr-form=1"]`` both give
    ``("-amdgpu-mfma-vgpr-form",)``: the value after the option's own ``=`` is
    not part of what the parser has to know.
    """
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


@functools.cache
def _table() -> dict:
    doc = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(
            f"{TABLE_PATH} is not a {SCHEMA} table; regenerate it with {_MEASURE_TOOL}"
        )
    options = doc.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"{TABLE_PATH} has no `options` mapping")  # noqa: TRY004
    return options


def measured_options() -> tuple:
    """Every option the table has an answer for, in file order."""
    return tuple(_table())


def lookup(option: str, llvm_major: int) -> Answer | None:
    """The measured answer for ``option`` on LLVM ``llvm_major``, or ``None``
    when that pair was never measured."""
    cell = _table().get(option, {}).get(str(llvm_major))
    if cell is None:
        return None
    return Answer(
        accepted=bool(cell["accepted"]),
        evidence=str(cell.get("evidence", "")),
        verified_on=str(cell.get("verified_on", "")),
    )


def require_accepted(
    options: Sequence[str], *, compiler: CompilerInfo, needed_by: str
) -> None:
    """Raise unless ``compiler`` accepts every ``-mllvm`` option in ``options``.

    ``needed_by`` names what is being compiled (for example
    ``"kernel 'gemm_f16'"``) so the error says which kernel needs the option.
    Raises :class:`CompilerOptionNotAccepted` when the table records that the
    compiler refuses an option, and ``ValueError`` when the table has no answer
    for the option or for this compiler's LLVM major version.
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
                int(m) for m, cell in _table()[name].items() if cell["accepted"]
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
