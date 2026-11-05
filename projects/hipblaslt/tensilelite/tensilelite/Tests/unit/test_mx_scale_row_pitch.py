# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Execute the emitted MX scale K-row pitch on the CPU.

TDM wave/GSU offsets and GSU SRD increments scale by the K-row pitch of the scale
tensor: Size for 1D, ceil(Size / MXBlockFree) rows for 2D, as the host allocates.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.container import sgpr
from rocisa.instruction import SAddU32, SLShiftRightB32, SMulI32

from tensilelite.Common.MxScaleLayout import mulScaleRows
from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401 (component registry)
from tensilelite.Components.TensorDataMover import TensorDataMoverLoad

pytestmark = pytest.mark.unit

_MASK32 = (1 << 32) - 1


def _execute(module, registers):
    registers = dict(registers)

    def value(operand):
        text = str(operand)
        return registers[text] if text in registers else int(text, 0)

    for inst in module.flatitems():
        if not hasattr(inst, "getParams"):
            continue
        params = inst.getParams()
        a, b = value(params[1]), value(params[2])
        if isinstance(inst, SAddU32):
            result = a + b
        elif isinstance(inst, SLShiftRightB32):
            result = a >> b  # getParams() is (dst, src, shift)
        elif isinstance(inst, SMulI32):
            result = a * b
        else:
            pytest.fail(f"Unsupported instruction: {inst}")
        registers[str(params[0])] = result & _MASK32
    return registers


def _pitch(size, mx_tile):
    return -(-size // mx_tile)


_SIZES = [128, 256, 264, 200, 1, 4096 + 7]


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("mx_tile", [1, 32, 128])
def test_constant_factor(size, mx_tile):
    # GSU SRD increment: dst = ceil(Size / MXBlockFree) * (K bytes per step).
    dst, size_reg = sgpr("Inc"), sgpr("SizeI")
    module = mulScaleRows(dst, 2, size_reg, mx_tile)
    regs = _execute(module, {str(size_reg): size})
    assert regs[str(dst)] == 2 * _pitch(size, mx_tile)


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("mx_tile", [1, 32, 128])
def test_register_factor_in_place(size, mx_tile):
    # TDM offsets: dst *= pitch, with the 2D row count in a separate sgpr.
    dst, rows, size_reg = sgpr("Off"), sgpr("Rows"), sgpr("SizeJ")
    module = mulScaleRows(dst, dst, size_reg, mx_tile, rows)
    regs = _execute(module, {str(size_reg): size, str(dst): 3})
    assert regs[str(dst)] == 3 * _pitch(size, mx_tile)


def test_1d_is_a_single_multiply():
    # 1D keeps the pre-2D emission: one s_mul by Size, no extra SALU.
    dst, size_reg = sgpr("Off"), sgpr("SizeI")
    items = [i for i in mulScaleRows(dst, dst, size_reg, 1).flatitems() if hasattr(i, "getParams")]
    assert [type(i) for i in items] == [SMulI32]


class _Writer:
    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield SimpleNamespace(idx=90)


@pytest.mark.parametrize("size", [256, 264])
def test_tdm_offset_uses_ceil_rows(size):
    # 2D MXSA: M = 264 needs 3 scale rows, so one K row of offset is 3 bytes, not 2.
    kernel = {"ProblemType": {"MXBlockFreeA": 128}}
    module = TensorDataMoverLoad.mulScaleKRowPitch(None, _Writer(), kernel, "MXSA", 0, 80, "test")
    regs = _execute(module, {str(sgpr("SizeI")): size, str(sgpr(80)): 1})
    assert regs[str(sgpr(80))] == _pitch(size, 128)
