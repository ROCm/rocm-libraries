# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 TDMPlusLdsBuf characterization: the 3-buffer LDS rotation under UseLoopModel.

Target code: Lowering/gir/nodes.read_pointer_unit, SwapRegions, and the IncLdsBufSwitch arms of
KernelWriterAssembly.localReadSwapOffsets / tdmSwapLdsOffset.

At three or more LDS buffers every address is rebuilt from one shared increment SGPR instead of
toggling with an xor, so the increment must be advanced before any address is built from it.
SwapRegions places one swap per PHYSICAL pointer, which is why that holds whichever operand the
solver puts first. pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")

_ADV_READ = re.compile(r"s_add_u32 s\[sgprLDSBufferReadInc\]")
_USE_READ = re.compile(r"v_add_nc_u32 v\[vgprLocalReadAddr\w+\], s\[sgprLDSBufferReadInc\]")
_ADV_WRITE = re.compile(r"s_add_u32 s\[sgprLDSBufferWriteInc\]")
_USE_WRITE = re.compile(r"s_cmp_eq_u32 s\[sgprLDSBufferWriteInc\], 0")


def _loop_body(source):
    lines = source.splitlines()
    begin = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
    end = next(i for i, l in enumerate(lines) if l.startswith("label_LoopEndL:"))
    return lines[begin:end + 1]


def _sequence(body, advance, use):
    out = []
    for line in body:
        if advance.search(line):
            out.append("ADV")
        elif use.search(line):
            out.append("USE")
    return out


@pytest.fixture(scope="module")
def kernels():
    return emit_kernels_from_config(os.path.join(_DESIGNED, "loopmodel_tdmpluslds.yaml"),
                                    limit=8, arch="gfx1250")


def test_three_lds_buffers_rotate_off_one_advanced_increment(kernels):
    """One physical pointer advances ONCE a trip, before every address built from it.

    Two advances means the operands were placed as two pointers over one register, so the ring
    turns twice a trip; a use before the advance means an address was built from the previous
    trip's value.
    """
    seen = 0
    for base, source, err in kernels:
        assert err == 0, base
        if "sgprLDSBufferReadInc" not in source:
            continue
        seen += 1
        body = _loop_body(source)
        assert int(re.search(r"\.set LdsBlockEndSize, (\d+)", source).group(1)) == \
            3 * int(re.search(r"\.set LdsOneBlockSize, (\d+)", source).group(1)), base
        for advance, use in ((_ADV_READ, _USE_READ), (_ADV_WRITE, _USE_WRITE)):
            sequence = _sequence(body, advance, use)
            assert sequence.count("ADV") == 1, "%s: %r" % (base, sequence)
            assert sequence.index("ADV") == 0, "%s: %r" % (base, sequence)
        assert_assembles(source, base)
    assert seen, "no kernel used the 3-buffer rotation; the fixture no longer exercises it"
