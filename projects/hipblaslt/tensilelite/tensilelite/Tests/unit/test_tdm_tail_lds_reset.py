# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Execute emitted tail resets and check the physical LDS destination."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.container import sgpr
from rocisa.label import LabelManager

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


class _Writer(KernelWriterAssembly):
    def __init__(self, blocks=2):
        self.states = SimpleNamespace(numLDSBlk=blocks)
        self.labels = LabelManager()

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        assert num == 1
        yield SimpleNamespace(idx=100)


def _execute(module, registers):
    """Interpret the scalar arithmetic and forward branches emitted by the reset."""
    registers = dict(registers)
    lines = [line.split("//")[0].strip() for line in str(module).splitlines()]
    lines = [line for line in lines if line]
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
    pc, scc = 0, False

    def value(operand):
        return registers[operand] if operand in registers else int(operand, 0)

    while pc < len(lines):
        line = lines[pc]
        pc += 1
        if line.endswith(":"):
            continue
        op, args = line.split(None, 1)
        args = [arg.strip() for arg in args.split(",")]
        if op == "s_cmp_lt_u32":
            scc = value(args[0]) < value(args[1])
        elif op == "s_cbranch_scc1":
            if scc:
                pc = labels[args[0]] + 1
        else:
            dst, a, b = args[0], value(args[1]), value(args[2])
            if op == "s_xor_b32":
                result = a ^ b
            elif op == "s_min_u32":
                result = min(a, b)
            elif op == "s_sub_u32":
                result = a - b
            else:
                pytest.fail(f"Unsupported instruction in tail reset: {line}")
            registers[dst] = result & 0xFFFFFFFF
    return registers


# Padded component pairs: A0, B0, A1, B1, then separate MX scale pairs.
# These include pong below the old block boundary, ping above it, and a
# wave's nonzero offset within a component. Scale strides differ from data.
@pytest.mark.parametrize("tc,ping,stride", [
    pytest.param("A", 0, 33024, id="A0"),
    pytest.param("A", 132096, 33024, id="A1"),
    pytest.param("A", 148608, 33024, id="A1-wave-offset"),
    pytest.param("B", 66048, 33024, id="B0"),
    pytest.param("B", 198144, 33024, id="B1"),
    pytest.param("MXSA", 264192, 1280, id="scale-A"),
    pytest.param("MXSB", 266752, 2304, id="scale-B"),
])
@pytest.mark.parametrize("phase", [0, 1], ids=["ping", "pong"])
def test_paired_tail_reset_selects_own_ping_idempotently(tc, ping, stride, phase):
    kernel = {"LdsOffsetA_Blk": 131072,
              "LDSSegInterleaveOffsets": {"pairedBuffers": True}}
    addr = str(sgpr(f"tdm{tc}Group0+1"))
    mask = "PairedTDMScaleSwap" if tc.startswith("MXS") else "PairedTDMSwap"
    # Give the other descriptor set a distinct address and mask: resetting this
    # set must not change or consult it.
    other = "tdmAGroup0+1" if tc.startswith("MXS") else "tdmMXSAGroup0+1"
    registers = {
        addr: ping + phase * stride,
        str(sgpr(other)): 76544,
        str(sgpr("PairedTDMSwap")): 0xBAD00,
        str(sgpr("PairedTDMScaleSwap")): 0xBAD00,
    }
    registers[str(sgpr(mask))] = ping ^ (ping + stride)
    module = _Writer().tdmResetTailLdsBuffer(kernel, f"tdm{tc}Group0+1")
    for _ in range(2):
        registers = _execute(module, registers)
        assert registers[addr] == ping
        assert registers[str(sgpr(other))] == 76544
        assert registers[str(sgpr(mask))] == ping ^ (ping + stride)


@pytest.mark.parametrize("blocks,stride", [(2, 131072), (2, 132096), (3, 66048)])
def test_unpaired_tail_reset_preserves_tensor_base(blocks, stride):
    kernel = {"LdsOffsetA_Blk": stride}
    writer = _Writer(blocks)
    addr = str(sgpr("tdmBGroup0+1"))
    module = writer.tdmResetTailLdsBuffer(kernel, "tdmBGroup0+1")
    for base in (0, 33024, stride - 256):
        for phase in range(blocks):
            registers = {addr: base + phase * stride}
            for _ in range(2):
                registers = _execute(module, registers)
                assert registers[addr] == base


def test_tail_reset_without_buffer_stride_preserves_address():
    module = _Writer().tdmResetTailLdsBuffer({"LdsOffsetA_Blk": 0}, "tdmAGroup0+1")
    registers = {str(sgpr("tdmAGroup0+1")): 33024}
    assert _execute(module, registers) == registers
