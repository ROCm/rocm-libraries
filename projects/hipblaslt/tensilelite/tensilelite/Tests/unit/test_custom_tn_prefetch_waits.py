# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check DirectToLDS completion before the handwritten gfx950 TN local reads.

The PGR2 waits count the younger second prefetch. On the one-DepthU path,
that prefetch is skipped, so the first prefetch needs its own completion
before the barrier makes its LDS writes visible to the local reads.
"""

import re

import pytest

from Tensile.resources import custom_kernel_text

pytestmark = pytest.mark.unit

KERNELS = [
    f"Custom_Cijk_Alik_Bljk_{dtype}_BH_MT256x256x64_MI16x16x1_UserArgs_shortname{variant}_gfx950"
    for dtype, variant in [("BBS", 0), ("BBS", 1), ("HHS", 0)]
]


def _prefetch_hazards(source, loop_count):
    # No VM wait precedes the PGR2 branch: the first A/B prefetch is pending.
    start = source.index("/* prefetch: global -> local */")
    branch = source.index("s_cmp_eq_u32 s[sgprLoopCounterL], 0x1", start)
    prefix = source[start:branch]
    end = source.index("label_openLoopL:", branch)
    code = [line.split("//", 1)[0].strip() for line in source[branch:end].splitlines()]
    labels = {line[:-1]: i for i, line in enumerate(code) if line.endswith(":")}
    pending, issued, completed, visible = [], set(), set(), set()
    prefetch_counts = {"A": 0, "B": 0}

    def issue(instruction):
        match = re.search(r"s\[sgprSrd([AB]):", instruction)
        assert match and "lds" in instruction
        tensor = match.group(1)
        index = prefetch_counts[tensor]
        event = (tensor, index // 8, index % 8)
        prefetch_counts[tensor] += 1
        pending.append(event)
        issued.add(event)

    for instruction in prefix.splitlines():
        if instruction.startswith("buffer_load_"):
            issue(instruction)
        assert not re.search(r"s_waitcnt.*vmcnt", instruction)
    assert prefetch_counts == {"A": 8, "B": 8}
    ip, scc, hazards, reads = 0, False, [], 0
    while ip < len(code):
        instruction = code[ip]
        if instruction.startswith("s_cmp_eq_u32 s[sgprLoopCounterL], 0x1"):
            scc = loop_count == 1
        elif instruction.startswith("s_cbranch_scc"):
            operation, target = instruction.split()
            if scc == operation.endswith("1"):
                ip = labels[target]
                continue
        elif instruction.startswith("s_branch "):
            ip = labels[instruction.split()[1]]
            continue
        elif instruction.startswith("buffer_load_"):
            issue(instruction)
        elif instruction.startswith("s_waitcnt "):
            match = re.search(r"vmcnt\((\d+)\)", instruction)
            if match:
                threshold = int(match.group(1))
                retire = max(0, len(pending) - threshold)
                completed.update(pending[:retire])
                del pending[:retire]
        elif instruction.startswith("s_barrier"):
            visible.update(completed)
        elif instruction.startswith("ds_read_b128 "):
            tensor = re.search(r"v\[vgprLocalReadAddr([AB])\]", instruction).group(1)
            first_prefetch = {event for event in issued if event[0:2] == (tensor, 0)}
            if not first_prefetch <= visible:
                hazards.append((ip, tensor, sorted(first_prefetch - visible)))
            reads += 1
        ip += 1
    assert reads == 16
    assert prefetch_counts == {"A": 8 * min(loop_count, 2), "B": 8 * min(loop_count, 2)}
    return hazards


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("loop_count", [1, 2])
def test_first_prefetch_is_visible_before_local_reads(kernel, loop_count):
    assert not _prefetch_hazards(custom_kernel_text(kernel), loop_count)
