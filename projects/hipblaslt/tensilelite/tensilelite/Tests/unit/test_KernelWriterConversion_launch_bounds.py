# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The output conversion (PostGSU) kernels declare the host's workgroup size.

The generator emits __launch_bounds__(KernelWriterConversion.WorkGroupSize) and
the host launches every conversion kernel with OutputConversionWorkGroupSize
threads. A launch with more threads than the bound fails, and a bound larger
than the launch costs registers, so the two must agree.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensilelite.KernelWriterConversion import KernelWriterConversion

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HEADER = _PROJECT_ROOT / "include" / "Tensile" / "ContractionSolution.hpp"
_SOURCE = _PROJECT_ROOT / "src" / "ContractionSolution.cpp"


def _blankCommentsAndStrings(source):
    """source with comments removed and string and char literals emptied, so
    braces and calls inside them are not seen."""
    literal = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
    return re.sub(literal, lambda m: m.group(0)[0] * 2 if m.group(0)[0] in "\"'" else "",
                  source, flags=re.S)


def _matching(code, openIndex):
    """Index of the bracket closing the one at code[openIndex]."""
    pair = {"(": ")", "{": "}"}[code[openIndex]]
    depth = 0
    for i in range(openIndex, len(code)):
        if code[i] == code[openIndex]:
            depth += 1
        elif code[i] == pair:
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError(f"unbalanced {code[openIndex]} at offset {openIndex} in {_SOURCE.name}")


def _memberFunctions(source):
    """{name: [body, ...]} of every ContractionSolution member defined in source."""
    code = _blankCommentsAndStrings(source)
    functions = {}
    for m in re.finditer(r"\bContractionSolution::(\w+)\s*\(", code):
        params = _matching(code, m.end() - 1)
        body = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", code[params + 1:])
        if not body:
            continue  # a call or a declaration
        brace = params + body.end()
        functions.setdefault(m.group(1), []).append(code[brace:_matching(code, brace) + 1])
    return functions


_FUNCTIONS = _memberFunctions(_SOURCE.read_text())

# Host functions that launch a conversion kernel: those that name one, plus
# those named like a conversion launcher, because
# updateUserArgsOutputConversionCallGroupedGemm has its outputConversionKernelName
# call commented out (it has no problem to name the kernel from) but still sets
# the launch size.
_LAUNCHERS = sorted(name for name, bodies in _FUNCTIONS.items()
                    if "OutputConversionCall" in name
                    or any("outputConversionKernelName(" in body for body in bodies))


def test_every_known_launcher_is_found():
    assert {"generateOutputConversionCall", "generateOutputConversionCallGroupedGemm",
            "updateUserArgsOutputConversionCallGroupedGemm"} <= set(_LAUNCHERS), _LAUNCHERS


def test_host_constant_matches_generator():
    match = re.search(r"constexpr uint32_t OutputConversionWorkGroupSize = (\d+);",
                      _HEADER.read_text())
    assert match, f"OutputConversionWorkGroupSize not found in {_HEADER.name}"
    assert int(match.group(1)) == KernelWriterConversion.WorkGroupSize


@pytest.mark.parametrize("name", _LAUNCHERS)
def test_host_launches_use_the_constant(name):
    for body in _FUNCTIONS[name]:
        sizes = re.findall(r"workGroupSize\.x\s*=\s*([^;]+);", body)
        assert sizes == ["OutputConversionWorkGroupSize"], sizes


@pytest.mark.parametrize("grouped", [False, True])
def test_signature_declares_launch_bounds(grouped):
    # functionSignature reads only these; a real writer needs rocisa and isaInfoMap.
    writer = SimpleNamespace(
        kernelName="Cijk_SB_PostGSU16_VW4",
        state={"ProblemType": {"GroupedGemm": grouped}},
        endLine="\n",
        WorkGroupSize=KernelWriterConversion.WorkGroupSize,
    )
    signature = KernelWriterConversion.functionSignature(writer)
    assert "__global__ __launch_bounds__(%d) void Cijk_SB_PostGSU16_VW4(" \
        % KernelWriterConversion.WorkGroupSize in signature
