# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Execute emitted descriptor addresses against descriptor-relative pad arithmetic."""
from contextlib import contextmanager
import re

import pytest
from rocisa.container import ContinuousRegister
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


class _Writer:
    tdmWaveLdsAddr = KernelWriterAssembly.tdmWaveLdsAddr

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield ContinuousRegister(100, num)


def _execute(module, component):
    registers = {"s0": component}

    def read(operand):
        if operand in registers:
            return registers[operand]
        pair = re.fullmatch(r"s\[(\d+):(\d+)\]", operand)
        if pair:
            return registers['s' + pair[1]] | (registers['s' + pair[2]] << 32)
        return int(operand, 0)

    for raw in str(module).splitlines():
        line = raw.split('//')[0].strip()
        if not line:
            continue
        opcode, operands = line.split(None, 1)
        dst, *src = [s.strip() for s in operands.split(',')]
        values = [read(s) for s in src]
        if opcode == 's_mul_i32':
            result = values[0] * values[1]
        elif opcode == 's_mul_hi_u32':
            result = (values[0] * values[1]) >> 32
        elif opcode == 's_add_u32':
            result = values[0] + values[1]
        elif opcode in ('s_lshr_b32', 's_lshr_b64'):
            result = values[0] >> values[1]
        elif opcode == 's_mov_b32':
            result = values[0]
        else:
            pytest.fail(f'Unsupported descriptor instruction: {line}')
        pair = re.fullmatch(r"s\[(\d+):(\d+)\]", dst)
        if pair:
            registers['s' + pair[1]] = result & 0xffffffff
            registers['s' + pair[2]] = (result >> 32) & 0xffffffff
        else:
            registers[dst] = result & 0xffffffff
    return registers['s0']


@pytest.mark.parametrize('tc,base', [('A', 0), ('B', 8448)])
@pytest.mark.parametrize('block,piece', [(512, 24576), (1024, 49152),
                                        (1536, 24576), (3072, 49152)])
@pytest.mark.parametrize('component', [0, 1, 2, 3])
def test_descriptor_component_padding_uses_exact_division(tc, base, block, piece, component):
    module = _Writer().tdmWaveLdsAddr(
        {'LDSSegmentInterleave': 0, 'NumWaves': 4}, tc, 2, piece, base,
        block, 16, 0, 1)
    raw = component * piece
    assert _execute(module, component) == base + raw + (raw // block) * 16
    if block & (block - 1) == 0:
        assert 's_mul_hi_u32' not in str(module)
