# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check LDS operand readiness on the handwritten gfx950 optimized final path."""

from collections import deque
import re

import pytest

from Tensile.CustomKernels import getCustomKernelContents

pytestmark = pytest.mark.unit

KERNELS = [
    f"Custom_Cijk_Alik_Bljk_{dtype}_BH_MT256x256x64_MI16x16x1_UserArgs_shortname{variant}_gfx950"
    for dtype, variant in [("BBS", 0), ("BBS", 1), ("HHS", 0)]
]


def _pending_final_loop_operands(source):
    """Follow the beta=0, alpha=1, GSU=1, full-tile fallthrough path.

    Start at the preceding LDS drain, track the subsequent reads, and check
    their MFMA consumers across the final-loop boundary. These kernels use
    only ds_read_b128 for LGKM operations in this interval. FIFO retirement
    accounts for gfx950's 15 outstanding LGKM operations as well as waits.
    This is a focused dependency check, not a general assembly interpreter.
    """
    definitions = dict(re.findall(r"^\.set\s+(\w+),\s*([^/\n]+)", source, re.M))

    def value(expression):
        # Register expressions in these kernels are sums of constants and
        # .set symbols. Fail on unsupported syntax instead of evaluating code.
        terms = expression.strip().split("+")
        result = 0
        for term in terms:
            term = term.strip()
            if re.fullmatch(r"0x[0-9a-fA-F]+|[0-9]+", term):
                result += int(term, 0)
            else:
                assert re.fullmatch(r"\w+", term), term
                result += value(definitions[term])
        return result

    def registers(operand):
        bounds = operand.split(":")
        first, last = value(bounds[0]), value(bounds[-1])
        return set(range(first, last + 1))

    boundary = source.index("label_toPGR1:")
    start = source.rindex("s_waitcnt lgkmcnt(0)", 0, boundary)
    end = source.index("label_OptNLL_End:", boundary)
    # All branches selecting the optimized path fall through. Its final
    # branch exits to the epilogue after the compute instructions checked here.
    lines = source[start:end].splitlines()
    pending = deque()
    violations = []
    reads = mfmas = 0
    for line in lines:
        instruction = line.split("//", 1)[0].strip()
        assert not instruction.startswith(("s_load_", "ds_write_")), instruction
        wait = re.match(r"s_waitcnt\s+lgkmcnt\((\d+)\)", instruction)
        if wait:
            while len(pending) > int(wait[1]):
                pending.popleft()
        elif instruction.startswith("ds_read_"):
            assert instruction.startswith("ds_read_b128 "), instruction
            destination = re.match(r"ds_read_b128\s+v\[([^\]]+)\]", instruction)
            assert destination, instruction
            pending.append(registers(destination[1]))
            reads += 1
            if len(pending) > 15:
                pending.popleft()
        elif instruction.startswith("v_mfma_"):
            operands = re.findall(r"v\[([^\]]+)\]", instruction)
            assert len(operands) == 2, instruction
            used = registers(operands[0]) | registers(operands[1])
            outstanding = set().union(*pending)
            if used & outstanding:
                violations.append((sorted(used & outstanding), instruction))
            mfmas += 1
    assert reads >= 16 and mfmas >= 128, "The expected compute path was not checked"
    return violations


@pytest.mark.parametrize("kernel", KERNELS)
def test_optimized_final_loop_consumes_completed_lds_reads(kernel):
    violations = _pending_final_loop_operands(getCustomKernelContents(kernel))
    assert not violations, f"{kernel}: pending LDS operands at {violations}"
