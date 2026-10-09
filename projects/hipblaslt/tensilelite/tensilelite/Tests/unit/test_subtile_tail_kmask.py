# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The subtile tail-loop K mask must match the K layout the local read loads.

The tail loop reads a full MatrixInstK slice per MMA tile and zeroes, per lane,
the bf16/fp16 elements at K >= the remaining K.  Which K each VGPR half holds is set
by the local read (_computeLROffset + emitSingleDsRead), so this test runs both
the production LR offset code and the production mask code through a small
per-lane interpreter and checks, for every remainder, that the mask keeps
exactly the elements whose loaded K is below it.

Wave32 bf16 (gfx1250 WMMA) splits each tile into two 16-byte reads that
interleave across the lane groups, which is the case this guards; wave64 bf16
(gfx950) is one read per tile.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state
from Tensile.Common.RegisterPool import allocTmpGpr
from Tensile.Components.Subtile.LogicalScheduler import ReadGranularity, SchedulerConfig
from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter
from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, AB_B16_W32, CD_F32, CD_F32_W32,
)
from Tensile.Components.Subtile.SubtileLREmit import _computeLROffset, emitSingleDsRead

MASK32 = 0xFFFFFFFF
DEPTH_U = 64
MI_K = 32
NUM_SUB_ITER_K = DEPTH_U // MI_K

CASES = [
    pytest.param(("gfx1250", 32, AB_B16_W32, CD_F32_W32), id="gfx1250-w32-two-reads"),
    pytest.param(("gfx950", 64, AB_B16, CD_F32), id="gfx950-w64-one-read"),
]


# ── Per-lane interpreter for the handful of opcodes these emitters use ──────

def _signed(x):
    x &= MASK32
    return x - (1 << 32) if x & 0x80000000 else x


def _binop(op):
    return lambda a, b: op(a, b) & MASK32


_VOP2 = {
    "v_and_b32": _binop(lambda a, b: a & b),
    "v_add_u32": _binop(lambda a, b: a + b),
    "v_add_nc_u32": _binop(lambda a, b: a + b),
    "v_sub_i32": _binop(lambda a, b: a - b),
    "v_sub_nc_i32": _binop(lambda a, b: a - b),
    "v_lshlrev_b32": _binop(lambda s, x: x << s),
    "v_lshrrev_b32": _binop(lambda s, x: x >> s),
}
_VCMP = {
    "v_cmp_lt_i32": lambda a, b: _signed(a) < _signed(b),
    "v_cmp_gt_i32": lambda a, b: _signed(a) > _signed(b),
    "v_cmp_le_i32": lambda a, b: _signed(a) <= _signed(b),
}


class _Lanes:
    """Registers as name -> per-lane list (vector) or int (scalar)."""

    def __init__(self, wave, scalars=None, vectors=None):
        self.wave = wave
        self.regs = dict(scalars or {})
        self.regs.update(vectors or {})

    def read(self, tok, lane):
        if tok in self.regs:
            val = self.regs[tok]
            return val[lane] if isinstance(val, list) else val
        return int(tok, 0) & MASK32

    def run(self, insts):
        for inst in insts:
            text = str(inst).split("//")[0].strip()
            if not text:
                continue
            mnemonic, _, rest = text.partition(" ")
            ops = [o.strip() for o in rest.split(",")]
            lanes = range(self.wave)
            if mnemonic in _VOP2:
                fn = _VOP2[mnemonic]
                self.regs[ops[0]] = [fn(self.read(ops[1], l), self.read(ops[2], l)) for l in lanes]
            elif mnemonic in _VCMP:
                fn = _VCMP[mnemonic]
                self.regs[ops[0]] = [fn(self.read(ops[1], l), self.read(ops[2], l)) for l in lanes]
            elif mnemonic == "v_cndmask_b32":
                self.regs[ops[0]] = [self.read(ops[2], l) if self.read(ops[3], l) else self.read(ops[1], l)
                                     for l in lanes]
            elif mnemonic == "v_mov_b32":
                self.regs[ops[0]] = [self.read(ops[1], l) for l in lanes]
            elif mnemonic == "s_mov_b32":
                self.regs[ops[0]] = self.read(ops[1], 0)
            else:
                raise AssertionError(f"interpreter does not model {text!r}")


# ── Harness ─────────────────────────────────────────────────────────────────

def _16bit_dtype(name):
    dtype = MagicMock()
    dtype.numBytes.return_value = 2
    dtype.isBFloat16.return_value = name == "bf16"
    dtype.isHalf.return_value = name == "fp16"
    dtype.is8bitFloat.return_value = False
    return dtype


def _make_emitter(wave, abGeometry, dGeometry, dtypeName):
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    dtype = _16bit_dtype(dtypeName)
    kernel = {
        "MacroTile0": 16, "MacroTile1": 16, "MacroTileA": 16, "MacroTileB": 16,
        "DepthU": DEPTH_U, "_DepthUA": DEPTH_U, "_DepthUB": DEPTH_U,
        "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": MI_K, "MatrixInstB": 1,
        "MIWaveGroup": [1, 1], "WavefrontSize": wave, "MIArchVgpr": True,
        "ProblemType": {"DataTypeA": dtype, "DataTypeB": dtype, "MXBlockA": 0, "MXBlockB": 0},
    }
    writer = SimpleNamespace()
    for name, regType in (("vgprPool", RegisterType.Vgpr), ("sgprPool", RegisterType.Sgpr),
                          ("agprPool", RegisterType.Accvgpr)):
        setattr(writer, name, RegisterPool(0, regType, defaultPreventOverflow=False, printRP=False))
    writer.states = SimpleNamespace(
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4}, asmCaps={},
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        subtileLdsSwizzle=False, laneSGPRCount=wave // 32, unrollIdx=0)
    writer.loopCounterName = lambda kernel, loopIdx: "LoopCounterL"
    writer.allocTmpSgpr = lambda num, alignment=None, tag=None: \
        allocTmpGpr(writer.sgprPool, num, writer.states.regCaps["MaxSgpr"], alignment, tag)

    tileInfoA = TileInfo(abGeometry, 'A', writer, kernel)
    tileInfoB = TileInfo(abGeometry, 'B', writer, kernel)
    dtileInfo = TileInfo(dGeometry, 'D', writer, kernel)
    for ti in (tileInfoA, tileInfoB, dtileInfo):
        ti.allocVgprTileRegisters_legacy(writer, kernel)
    config = SchedulerConfig(
        numMFMATilesM=1, numMFMATilesN=1, numSubIterK=NUM_SUB_ITER_K,
        lrA=ReadGranularity(mn=1, k=1), lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=NUM_SUB_ITER_K), grB=ReadGranularity(mn=1, k=NUM_SUB_ITER_K))
    emitter = InstructionEmitter(writer, kernel, config, tileInfoA, tileInfoB, dtileInfo,
                                 tileInfoA.vgprTiles, tileInfoB.vgprTiles)
    return writer, emitter, (tileInfoA, tileInfoB)


_DS_LOAD_RE = re.compile(r"ds_(?:load|read)_b128 v\[(\d+):(\d+)\], (v\d+) offset:(\d+)")


def _loaded_k(writer, tileInfo, wave, subIterK):
    """{(vgpr, half): per-lane K} for one MMA tile, from the production LR code."""
    from rocisa.code import Module

    tileInfo.lr.sharedVgprLROffset = [writer.vgprPool.checkOut(1)
                                      for _ in range(tileInfo.numLRPerSubtile)]
    colOffset = writer.vgprPool.checkOut(1)
    rowOffset = writer.vgprPool.checkOut(1)
    module = Module()
    _computeLROffset(module, tileInfo, colOffset, rowOffset, swizzled=False)
    lanes = _Lanes(wave, vectors={
        f"v{colOffset}": [(l % wave) // 16 for l in range(wave)],  # lane16Group
        f"v{rowOffset}": [0] * wave,
    })
    lanes.run(module.flatitems())

    bpe = int(tileInfo.bpe)
    layout = {}
    reads = emitSingleDsRead(tileInfo, 0, 0, subIterK, tileInfo.vgprTiles[0], swizzled=False)
    for inst in reads.flatitems():
        m = _DS_LOAD_RE.search(str(inst))
        assert m, f"unexpected local read {inst}"
        first, last, addrReg, offset = int(m[1]), int(m[2]), m[3], int(m[4])
        for v in range(first, last + 1):
            for half in range(2):
                elem = (v - first) * 2 + half
                layout[(v, half)] = [(lanes.read(addrReg, l) + offset) // bpe + elem
                                     for l in range(wave)]
    return layout


@pytest.fixture(params=["bf16", "fp16"])
def dtypeName(request):
    return request.param


@pytest.fixture(params=CASES)
def case(request, dtypeName):
    target, wave, abGeometry, dGeometry = request.param
    from gpu_test_helpers import init_rocisa
    from rocisa import rocIsa

    with preserve_rocisa_kernel_state():
        init_rocisa(target=target, wavesize=wave)
        if target == "gfx1250" and not rocIsa.getInstance().getAsmCaps().get("s_add_u64", 0):
            pytest.skip("assembler does not support gfx1250 instructions")
        yield (wave,) + _make_emitter(wave, abGeometry, dGeometry, dtypeName)


def test_tail_kmask_keeps_exactly_the_loaded_k_below_remainder(case):
    wave, writer, emitter, tileInfos = case
    initInsts = emitter.emit_mask_k_init()
    for subIterK in range(NUM_SUB_ITER_K):
        source = SimpleNamespace(subIterK=subIterK,
                                 vgpr_tile_map={'A': [{0: 0}], 'B': [{0: 0}]})
        maskInsts = emitter.emit_mask_k(source)
        layouts = [_loaded_k(writer, ti, wave, subIterK) for ti in tileInfos]
        tileRegs = [f"v{v}" for v, _ in layouts[0]] + [f"v{v}" for v, _ in layouts[1]]
        for rem in range(1, DEPTH_U):
            lanes = _Lanes(wave,
                           scalars={"s[sgprLoopCounterL]": rem},
                           vectors={"v[vgprSerial]": list(range(wave)),
                                    **{r: [MASK32] * wave for r in tileRegs}})
            lanes.run(initInsts)
            lanes.run(maskInsts)
            for layout in layouts:
                for (v, half), ks in layout.items():
                    for lane in range(wave):
                        kept = ((lanes.read(f"v{v}", lane) >> (16 * half)) & 0xFFFF) == 0xFFFF
                        assert kept == (ks[lane] < rem), (
                            f"rem={rem} subIterK={subIterK} lane={lane} v{v}.{half} "
                            f"holds K={ks[lane]} but mask {'kept' if kept else 'zeroed'} it")
