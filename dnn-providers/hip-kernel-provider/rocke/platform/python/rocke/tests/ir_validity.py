# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: judge whether lowered LLVM IR compiles and links for one target.

It reuses rocke's IR-validity tool (``platform/tools/check_ir_validity.py``) so a
test and the corpus gate use the same compiler, the same flags and the same
definition of "valid": the module must compile *and link* to a code object. The
link is what catches a call to an intrinsic that does not exist; see that tool's
module docstring.

``arbiter_verdict`` reports one of three verdicts:

- ``"accepted"``: clang compiled and linked the module for the target.
- ``"rejected"``: clang read the module and refused it; the text is clang's first
  error line.
- ``"cannot_judge"``: no verdict is possible here (no tools, no clang, a clang that
  does not know the target, or a clang that cannot run its linker); the text says
  which.

``assert_target_accepts`` turns ``"cannot_judge"`` into a named pytest skip, so a
host whose toolchain predates a target reports the gap instead of passing.
``arbiter_llvm_flavor`` names the LLVM flavor of the clang the arbiter compiles
with, for a test that must lower its module at that flavor
(``require_arbiter_llvm_flavor`` skips when there is none), and
``assert_kernel_target_accepts`` lowers a kernel at that flavor and requires the
module to compile and link.
``target_instructions`` compiles the same way and returns the code object's
disassembled instruction lines, and ``target_mnemonics`` their mnemonics, for a
test that pins the instruction LLVM selected and its operands.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from rocke.analysis.isa import _extract_mnemonic, _instruction_lines
from rocke.assets import platform_root
from rocke.core.ir import KernelDef

Verdict = Literal["accepted", "rejected", "cannot_judge"]

__all__ = [
    "Verdict",
    "arbiter_llvm_flavor",
    "arbiter_verdict",
    "assert_kernel_target_accepts",
    "assert_target_accepts",
    "require_arbiter_llvm_flavor",
    "target_instructions",
    "target_mnemonics",
]


def _validity_tool() -> ModuleType | None:
    """The IR-validity tool module, or None when the platform tools are not
    shipped alongside this package (e.g. an installed wheel)."""
    tools = platform_root() / "tools"
    if not (tools / "check_ir_validity.py").is_file():
        return None
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import check_ir_validity

    return check_ir_validity


def arbiter_llvm_flavor() -> tuple[str | None, str]:
    """The LLVM flavor of the clang :func:`arbiter_verdict` compiles with, as ``(flavor, "")``.

    A module the arbiter judges must be lowered at its clang's flavor. ``(None, reason)`` when
    there is no such clang here, or its ``--version`` banner names no flavor rocke lowers for.
    """
    from rocke.core.lower_llvm import LLVM_FLAVORS

    tool = _validity_tool()
    if tool is None:
        return None, "rocke platform tools not found"
    clang = tool.llvm_tool("clang")
    if clang is None:
        return None, "no host clang to compile lowered IR"
    flavor = tool.flavor_of_clang(tool.tool_identity(clang))
    if flavor not in LLVM_FLAVORS:
        return None, f"host clang {clang} reports no LLVM flavor rocke lowers for"
    return flavor, ""


def arbiter_verdict(ir_text: str, target: str) -> tuple[Verdict, str]:
    """Compile and link ``ir_text`` for ``target`` with the host clang.

    Returns ``(verdict, text)``; ``text`` is empty when the module is accepted.
    """
    with tempfile.TemporaryDirectory() as work:
        return _compile_in(ir_text, target, Path(work))


def _compile_in(ir_text: str, target: str, work: Path) -> tuple[Verdict, str]:
    """:func:`arbiter_verdict`, leaving the code object at ``work/module.hsaco``."""
    tool = _validity_tool()
    if tool is None:
        return "cannot_judge", "rocke platform tools not found"
    clang = tool.llvm_tool("clang")
    if clang is None:
        return "cannot_judge", "no host clang to compile lowered IR"
    source = work / "module.ll"
    source.write_text(ir_text)
    accepted, diagnostic = tool.compile_code_object(
        clang, source, target, work / "module.hsaco"
    )
    if accepted:
        return "accepted", ""
    if tool.target_unsupported(diagnostic):
        return "cannot_judge", f"host clang cannot target {target}"
    if _toolchain_incomplete(diagnostic):
        return (
            "cannot_judge",
            f"host clang {clang} cannot link: {diagnostic.splitlines()[0]}",
        )
    return "rejected", tool.first_diagnostic(diagnostic)


def _toolchain_incomplete(diagnostic: str) -> bool:
    """Whether clang failed because a tool it drives is missing, before judging the module."""
    return "unable to execute command" in diagnostic


def assert_target_accepts(ir_text: str, target: str) -> None:
    """Fail unless ``ir_text`` compiles and links for ``target``; skip, naming
    the reason, when no verdict is possible on this host."""
    _require_accepted(*arbiter_verdict(ir_text, target), target)


def require_arbiter_llvm_flavor() -> str:
    """:func:`arbiter_llvm_flavor`'s flavor; skip the calling test, naming the reason, when there
    is none."""
    flavor, why_not = arbiter_llvm_flavor()
    if flavor is None:
        pytest.skip(why_not)
    return flavor


def assert_kernel_target_accepts(kernel: KernelDef, target: str) -> None:
    """Lower ``kernel`` for ``target`` with the Python engine, whose IR the C++ engine must match,
    at the flavor of the arbiter's clang, and fail unless the module compiles and links; skip,
    naming the reason, when no verdict is possible on this host."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    flavor = require_arbiter_llvm_flavor()
    assert_target_accepts(
        _lower_kernel_to_llvm_python(kernel, arch=target, llvm_flavor=flavor), target
    )


def _require_accepted(verdict: Verdict, text: str, target: str) -> None:
    if verdict == "cannot_judge":
        pytest.skip(text)
    assert verdict == "accepted", f"LLVM rejected the lowered IR for {target}: {text}"


def target_mnemonics(ir_text: str, target: str) -> list[str]:
    """The instruction mnemonics of ``ir_text`` compiled and linked for ``target``.

    Fails and skips as :func:`target_instructions` does.
    """
    return [_extract_mnemonic(line) for line in target_instructions(ir_text, target)]


def target_instructions(ir_text: str, target: str) -> list[str]:
    """The disassembled instruction lines of ``ir_text`` compiled and linked for
    ``target``, each with its operands and modifiers.

    Fails when LLVM rejects the module, like :func:`assert_target_accepts`;
    skips, naming the reason, when no verdict is possible or no
    ``llvm-objdump`` is found next to the host clang.
    """
    with tempfile.TemporaryDirectory() as work:
        _require_accepted(*_compile_in(ir_text, target, Path(work)), target)
        objdump = _validity_tool().llvm_tool("llvm-objdump")
        if objdump is None:
            pytest.skip("no llvm-objdump to disassemble the code object")
        listing = subprocess.run(
            [objdump, "-d", str(Path(work) / "module.hsaco")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return list(_instruction_lines(listing))
