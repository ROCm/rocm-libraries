#!/usr/bin/env python3
################################################################################
# Codegen tests for gfx1250 (wave32) subtile paths.
#
# Exercises TileInfo, GR/LR emit, and kernel helpers with wave32 kernel dicts.
# No GPU hardware required -- tests run against the Python codegen layer only.
#
# Usage:
#   pytest test_subtile_gfx1250_codegen.py -v
################################################################################

import os
import sys
import shutil

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)


GFX1250_ISA = (12, 5, 0)
WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=None, depth_u=64):
    dtype = _mock_dtype(2)
    if mi_wave_group is None:
        mi_wave_group = [1, 1]
    return {
        "DepthU": depth_u,
        "_DepthU": depth_u,
        "_DepthUA": depth_u,
        "_DepthUB": depth_u,
        "MacroTileA": mt_a,
        "MacroTileB": mt_b,
        "MacroTile0": mt_a,
        "MacroTile1": mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": mi_wave_group,
        "WavefrontSize": WAVESIZE_32,
        "UseSubtileImpl": True,
        "ISA": GFX1250_ISA,
        "MIArchVgpr": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def _create_writer_gfx1250(kernel):
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16_W32

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.vgprPool.checkOut(1)  # v0 = Serial

    tiA = TileInfo(AB_B16_W32, 'A', writer, kernel)
    tiB = TileInfo(AB_B16_W32, 'B', writer, kernel)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tiA),
        b=SimpleNamespace(tileInfo=tiB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
        asmCaps={"HasMFMA": False, "HasWMMA_AccImmZero": True},
        subtileLdsSwizzle=False,
    )
    readSize = 2 * tiA.subtileSize
    numASubtiles = tiA.globalSubtileGrid[0] * tiA.globalSubtileGrid[1]
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = int(((numASubtiles * tiA.subtileSize + readSize - 1) // readSize) * readSize)

    return writer, tiA, tiB


def _setup_sgprs(writer):
    """Reserve base SGPRs and TDM descriptor SGPRs."""
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    for tc in ['A', 'B']:
        writer.sgprs["tdm%sGroup0" % tc] = writer.sgprPool.checkOutAligned(4, 4, preventOverflow=False)
        writer.sgprs["tdm%sGroup1" % tc] = writer.sgprPool.checkOutAligned(8, 4, preventOverflow=False)
        writer.sgprs["tdmLdsAddr%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["tdmLdsSwapMask%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["Address%s" % tc] = writer.sgprPool.checkOutAligned(2, 2, preventOverflow=False)


CONFIGS_1x1 = [
    (32, 32, [1, 1]),
    (64, 64, [1, 1]),
]

CONFIGS_MULTI_WAVE = [
    (64, 64,  [2, 2]),
    (128, 32, [4, 1]),
    (32, 128, [1, 4]),
]


class TestGfx1250SubtileCodegen:
    """Codegen tests for gfx1250 subtile paths."""

    # -- TDM GR offset skip --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_1x1 + CONFIGS_MULTI_WAVE,
                             ids=[f"{a}x{b}_wg{w[0]}x{w[1]}" for a, b, w in CONFIGS_1x1 + CONFIGS_MULTI_WAVE])
    def test_tdm_skips_gr_offset_alloc(self, mt_a, mt_b, wg):
        """TDM-enabled kernel: GR offset registers should not be allocated."""
        _init_rocisa_gfx1250()
        kernel = _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=wg)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        tiA.allocOffsetRegisters(writer, kernel)
        assert tiA.sharedVgprGROffset == []

    # -- LR tile assignment (covers wave partition, rotation skip) --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_MULTI_WAVE,
                             ids=[f"{a}x{b}_wg{w[0]}x{w[1]}" for a, b, w in CONFIGS_MULTI_WAVE])
    def test_lr_tile_assignment_multi_wave(self, mt_a, mt_b, wg):
        """LR tile assignment with multi-wave TDM produces valid assembly."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment
        kernel = _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=wg)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        module = lraTileAssignment(writer, kernel)
        asm = str(module)
        assert "TDM wave partition" in asm
        assert "rotation" not in asm.lower()

    # -- emitSingleDsRead dual load for 8-VGPR tiles --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_1x1,
                             ids=[f"{a}x{b}" for a, b, _ in CONFIGS_1x1])
    def test_ds_read_dual_load(self, mt_a, mt_b, wg):
        """Wave32 8-VGPR tiles emit two DSLoadB128 (lo + hi K-halves)."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileLREmit import emitSingleDsRead
        kernel = _create_gfx1250_kernel(mt_a, mt_b)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        tiA.allocVgprTileRegisters_legacy(writer, kernel)
        tile = tiA.vgprTiles[0]
        assert len(tile.regList.indices) == 8
        result = emitSingleDsRead(tiA, 0, 0, 0, tile)
        asm = str(result)
        assert asm.count("ds_load_b128") == 2
        assert "read=0" in asm
        assert "read=1" in asm

    # -- selectDGeometry wave32 --

    def test_select_d_geometry_wave32(self):
        """selectDGeometry returns CD_F32_W32 for wave32 kernels."""
        from Tensile.Components.Subtile.Kernel import selectDGeometry, CD_F32_W32
        kernel = _create_gfx1250_kernel(64, 64)
        assert selectDGeometry(kernel) is CD_F32_W32

    # -- initVgprTilesToZero scalar fallback --

    def test_zero_tiles_wmma(self):
        """gfx1250 tile zeroing uses v_wmma_f32_16x16x4_f32 with acc2_imm=0."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.Kernel import initVgprTilesToZero
        kernel = _create_gfx1250_kernel(32, 32)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiA.allocVgprTileRegisters_legacy(writer, kernel)
        module = initVgprTilesToZero(writer, kernel, tiA)
        asm = str(module)
        assert "v_wmma_f32_16x16x4_f32" in asm
        assert ", 0" in asm  # acc2_imm=0

    # -- globalReadLDSBufferSwap TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_gr_lds_buffer_swap_tdm(self, tc):
        """TDM LDS buffer swap emits XOR on tracking SGPR."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import globalReadLDSBufferSwap
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        module = globalReadLDSBufferSwap(tc, writer, kernel)
        asm = str(module)
        assert "s_xor_b32" in asm
        assert "sync descriptor LDS addr" in asm

    # -- globalReadPtrUpdates TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_gr_ptr_updates_tdm(self, tc):
        """TDM pointer update increments Address and syncs descriptor."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import globalReadPtrUpdates
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        module = globalReadPtrUpdates(tc, writer, kernel)
        asm = str(module)
        assert "s_add_u64" in asm
        assert "sync descriptor global addr" in asm

    # -- emitSingleBufferLoad TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_buffer_load_tdm(self, tc):
        """TDM emitSingleBufferLoad emits tensor_load_to_lds."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import emitSingleBufferLoad
        kernel = _create_gfx1250_kernel(64, 64)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        ti = tiA if tc == 'A' else tiB
        module = emitSingleBufferLoad(ti, kernel, 0, 0)
        asm = str(module)
        assert "tensor_load_to_lds" in asm


def _stinky_kernel():
    """Minimal kernel dict carrying the keys the StinkyTofu option builders read."""
    return {
        "_StinkyTofuOptLevel": 3,
        "EnableStinkyTofuESM2": False,
        "ThreadTile0": 1,
        "ThreadTile1": 1,
        "MacroTile0": 64,
        "WavefrontSize": WAVESIZE_32,
        "SubGroup0": 16,
        "SubGroup1": 16,
        "MIWaveGroup": [1, 1],
        "VectorWidthA": 4,
        "VectorWidthB": 4,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "DirectToLdsA": 0,
        "DirectToLdsB": 0,
        "_UseSgprForGRO": False,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ClusterBarrier": True,
    }


def _stinky_mock_writer():
    """Mock KernelWriter self exposing only what the StinkyTofu helpers touch."""
    return SimpleNamespace(
        sgprs={},
        states=SimpleNamespace(version=GFX1250_ISA, kernelName="kernel_name"),
    )


def _build_minimal_kernel_body():
    """A tiny but valid gfx1250 KernelBody + signature StinkyTofu can convert."""
    from rocisa.code import Module, KernelBody, SignatureBase, Label
    from rocisa.instruction import SEndpgm, SNop

    body = Module("body")
    body.add(Label("ASM_Start", "start"))
    body.add(SNop(waitState=0, comment="nop"))
    body.add(SEndpgm(comment="end"))

    sig = SignatureBase("kernel_name", 1, "V5", 0, [0, 1, 2], 0, 256,
                        totalVgprs=4, totalSgprs=16)

    kb = KernelBody("kernelBody")
    kb.addSignature(sig)
    kb.addBody(body)
    return kb, sig


class TestSubtileStinkyTofu:
    """StinkyTofu wiring for the gfx1250 subtile path (Phase 0 + Phase 1)."""

    # -- Phase 1: subtile options are a neutral pass-through --

    def test_subtile_options_are_neutral_passthrough(self):
        from Tensile.Components.Subtile.StinkyTofu import buildSubtileStinkyTofuOptions
        kernel = _stinky_kernel()
        opts = buildSubtileStinkyTofuOptions(kernel, kernel["_StinkyTofuOptLevel"],
                                             _stinky_mock_writer())
        # Pass-through: StinkyTofu must not insert/strip waits or barriers yet.
        assert opts["EnableWaitCntInsertion"] is False
        assert opts["ClusterBarrier"] is False
        # Subtile forces unit vector widths regardless of the kernel values.
        assert opts["VectorWidthA"] == 1
        assert opts["VectorWidthB"] == 1
        assert opts["OptLevel"] == 3

    def test_subtile_options_mirror_classic_keys(self):
        from Tensile.KernelWriter import KernelWriter
        from Tensile.Components.Subtile.StinkyTofu import buildSubtileStinkyTofuOptions
        kernel = _stinky_kernel()
        writer = _stinky_mock_writer()
        classic = KernelWriter._classicStinkyTofuOptions(writer, kernel,
                                                         kernel["_StinkyTofuOptLevel"])
        subtile = buildSubtileStinkyTofuOptions(kernel, kernel["_StinkyTofuOptLevel"],
                                                writer)
        # Same option surface; only the neutral/forced values differ.
        assert set(subtile.keys()) == set(classic.keys())
        differing = {k for k in classic if classic[k] != subtile[k]}
        assert differing <= {"EnableWaitCntInsertion", "ClusterBarrier",
                             "VectorWidthA", "VectorWidthB"}

    # -- Phase 1: subtile drives the helper and emits parseable asm --

    def test_helper_emits_assembly_for_subtile(self):
        _init_rocisa_gfx1250()
        from Tensile.KernelWriter import KernelWriter
        from Tensile.Components.Subtile.StinkyTofu import buildSubtileStinkyTofuOptions
        kernel = _stinky_kernel()
        writer = _stinky_mock_writer()
        kb, sig = _build_minimal_kernel_body()
        opts = buildSubtileStinkyTofuOptions(kernel, kernel["_StinkyTofuOptLevel"], writer)
        asm = KernelWriter._maybeRunStinkyTofu(writer, kernel, kb, sig,
                                               stinky_module_options=opts)
        assert asm is not None
        assert len(asm) > 0
        assert 'amdgcn-amd-amdhsa--gfx1250' in asm
        assert 'kernel_name' in asm

    # -- Phase 0: helper falls back (returns None) when StinkyTofu is off --

    def test_helper_returns_none_when_disabled(self):
        _init_rocisa_gfx1250()
        from Tensile.KernelWriter import KernelWriter
        kernel = _stinky_kernel()
        kernel["_StinkyTofuOptLevel"] = None  # SIA != 4: classic str() fallback
        writer = _stinky_mock_writer()
        kb, sig = _build_minimal_kernel_body()
        assert KernelWriter._maybeRunStinkyTofu(writer, kernel, kb, sig) is None

    # -- Phase 0: classic option set is unchanged by the refactor --

    def test_classic_options_unchanged(self):
        from Tensile.KernelWriter import KernelWriter
        kernel = _stinky_kernel()
        writer = _stinky_mock_writer()
        opts = KernelWriter._classicStinkyTofuOptions(writer, kernel,
                                                     kernel["_StinkyTofuOptLevel"])
        expected = {
            "OptLevel": 3,
            "EnableRemarks": False,
            "DebugLevel": 0,
            "PrintBeforePass": "",
            "PrintAfterPass": "",
            "DebugPass": "",
            "PassOrderSnapshotJson": "",
            # OptLevel != 0 -> wait-count insertion forced on for classic.
            "EnableWaitCntInsertion": True,
            "EnableESM2": False,
            "TileA0": 1,
            "TileB0": 1,
            "TileM0": 64,
            "wavefrontSize": WAVESIZE_32,
            "SubGroup0": 16,
            "SubGroup1": 16,
            "WaveGroup0": 1,
            "WaveGroup1": 1,
            "VectorWidthA": 4,
            "VectorWidthB": 4,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
            "DirectToLdsA": False,
            "DirectToLdsB": False,
            "UseSgprForGRO": False,
            "SwPrefetchScratchSgpr": -1,
            "ClusterBarrier": True,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
        }
        assert opts == expected


def _make_token_emitter(kernel_overrides=None):
    """Build a minimal InstructionEmitter exposing the MemToken tagging logic.

    Only the attributes that __init__ and the tagging helpers touch are
    populated; the heavy tile/scheduler state is mocked.
    """
    from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter
    kernel = {"1LDSBuffer": 0, "enableTDMA": True, "enableTDMB": True}
    if kernel_overrides:
        kernel.update(kernel_overrides)
    tileInfoA = SimpleNamespace(subtileShape=[1, 2])
    tileInfoB = SimpleNamespace(subtileShape=[1, 2])
    return InstructionEmitter(
        writer=SimpleNamespace(states=SimpleNamespace(subtileLdsSwizzle=False)),
        kernel=kernel, config=SimpleNamespace(),
        tileInfoA=tileInfoA, tileInfoB=tileInfoB, dtileInfo=None,
        vgprTilesA=[], vgprTilesB=[])


def _make_tensor_load():
    from rocisa.instruction import TensorLoadToLds
    from rocisa.container import sgpr
    return TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), None, None,
                           comment="TDM: global->LDS")


def _make_ds_read(dst=8, addr=16):
    from rocisa.instruction import DSLoadB128
    from rocisa.container import vgpr, DSModifiers
    return DSLoadB128(dst=vgpr(dst, 4), src=vgpr(addr),
                      ds=DSModifiers(offset=0), comment="ds_read")


def _stub_swap_emitters(monkeypatch):
    """Replace the heavy GR/LR swap emitters with no-op Modules.

    Lets emit_gr_inc/emit_lr_inc exercise the token toggle without a full
    writer/tileInfo state.
    """
    from rocisa.code import Module
    import Tensile.Components.Subtile.InstructionEmitter as IE
    stub = lambda *a, **k: Module()
    for name in ("globalReadPtrUpdates", "globalReadLDSBufferSwap",
                 "globalReadScalePtrUpdates", "localReadLDSBufferSwap"):
        monkeypatch.setattr(IE, name, stub)


def _candidate_tokens(insts):
    """Mirror the rocisa MemTokenConsistencyCheck: per-region all-or-none.

    Returns (has_tagged, has_untagged) over ds_read/ds_write/tensor_load.
    """
    from Tensile.Components.Subtile.MemToken import isLdsProducer, isLdsConsumer
    has_tagged = has_untagged = False
    for inst in insts:
        if isLdsProducer(inst) or isLdsConsumer(inst):
            if inst.getMemToken() is not None:
                has_tagged = True
            else:
                has_untagged = True
    return has_tagged, has_untagged


class TestSubtileMemToken:
    """Phase 2: MemTokenData tagging for the gfx1250 subtile path."""

    # -- token-ID scheme: double-buffer toggling --

    def test_tracker_double_buffer_toggle(self):
        from Tensile.Components.Subtile.MemToken import SubtileMemTokenTracker
        t = SubtileMemTokenTracker({"1LDSBuffer": 0})
        assert (t.writeTokenIdx, t.readTokenIdx) == (0, 0)
        assert t.writeToken().tokens == [0]
        assert t.readToken().tokens == [0]
        # Before any swap both buffers coincide -> single-id barrier.
        assert t.barrierToken().tokens == [0]
        t.swapWrite()
        assert t.writeTokenIdx == 1
        assert t.writeToken().tokens == [1]
        # Producer now on buffer1, consumer still buffer0 -> barrier spans both.
        assert t.barrierToken().tokens == [0, 1]
        t.swapRead()
        assert t.readTokenIdx == 1
        assert t.barrierToken().tokens == [1]
        t.swapWrite()
        assert t.writeTokenIdx == 0

    def test_tracker_one_lds_buffer_collapse(self):
        from Tensile.Components.Subtile.MemToken import SubtileMemTokenTracker
        t = SubtileMemTokenTracker({"1LDSBuffer": 1})
        assert (t.buffer0, t.buffer1) == (0, 0)
        t.swapWrite()
        t.swapRead()
        # With a single LDS buffer every token id stays 0.
        assert t.writeToken().tokens == [0]
        assert t.readToken().tokens == [0]
        assert t.barrierToken().tokens == [0]

    # -- producer/consumer classification --

    def test_producer_consumer_classification(self):
        _init_rocisa_gfx1250()
        from rocisa.instruction import SNop
        from Tensile.Components.Subtile.MemToken import isLdsProducer, isLdsConsumer
        tl = _make_tensor_load()
        rd = _make_ds_read()
        nop = SNop(waitState=0, comment="nop")
        assert isLdsProducer(tl) and not isLdsConsumer(tl)
        assert isLdsConsumer(rd) and not isLdsProducer(rd)
        assert not isLdsProducer(nop) and not isLdsConsumer(nop)

    # -- tagging: producers/consumers carry expected tokens --

    def test_tag_lds_tokens_producer_consumer(self):
        _init_rocisa_gfx1250()
        from rocisa.instruction import SNop
        em = _make_token_emitter()
        tl, rd, nop = _make_tensor_load(), _make_ds_read(), SNop(waitState=0)
        em._tagLdsTokens([tl, rd, nop])
        assert tl.getMemToken().tokens == [0]   # producer -> write buffer
        assert rd.getMemToken().tokens == [0]   # consumer -> read buffer
        assert nop.getMemToken() is None        # non-candidate untouched

    # -- tagging is consistent (no partial tagging in a region) --

    def test_tagging_is_consistent_within_region(self):
        _init_rocisa_gfx1250()
        from rocisa.instruction import SNop
        em = _make_token_emitter()
        region = [_make_tensor_load(), _make_ds_read(), SNop(waitState=0),
                  _make_tensor_load(), _make_ds_read()]
        em._tagLdsTokens(region)
        has_tagged, has_untagged = _candidate_tokens(region)
        assert has_tagged and not has_untagged

    # -- tokens toggle across an LDS swap --

    def test_tokens_toggle_across_swap(self, monkeypatch):
        _init_rocisa_gfx1250()
        _stub_swap_emitters(monkeypatch)
        from Tensile.Components.Subtile.LogicalScheduler import GRIncOp, LRIncOp
        em = _make_token_emitter()
        before_prod = em._tagLdsTokens([_make_tensor_load()])[0]
        before_cons = em._tagLdsTokens([_make_ds_read()])[0]
        assert before_prod.getMemToken().tokens == [0]
        assert before_cons.getMemToken().tokens == [0]
        # A global-read swap flips the producer buffer; local-read flips reader.
        em.emit_gr_inc(GRIncOp(tensor='A'))
        em.emit_lr_inc(LRIncOp(tensor='A'))
        after_prod = em._tagLdsTokens([_make_tensor_load()])[0]
        after_cons = em._tagLdsTokens([_make_ds_read()])[0]
        assert after_prod.getMemToken().tokens == [1]
        assert after_cons.getMemToken().tokens == [1]

    def test_swap_toggles_once_per_event_not_per_tensor(self, monkeypatch):
        _init_rocisa_gfx1250()
        _stub_swap_emitters(monkeypatch)
        from Tensile.Components.Subtile.LogicalScheduler import GRIncOp
        em = _make_token_emitter()
        # A swap event issues an inc per tensor (A, B); only 'A' flips the
        # shared parity so A+B together produce a single net toggle.
        em.emit_gr_inc(GRIncOp(tensor='A'))
        em.emit_gr_inc(GRIncOp(tensor='B'))
        assert em.memToken.writeTokenIdx == 1
        prod = em._tagLdsTokens([_make_tensor_load()])[0]
        assert prod.getMemToken().tokens == [1]

    # -- barrier carries both buffers it separates --

    def test_emit_sync_barrier_token(self, monkeypatch):
        _init_rocisa_gfx1250()
        _stub_swap_emitters(monkeypatch)
        from Tensile.Components.Subtile.LogicalScheduler import GRIncOp
        em = _make_token_emitter()
        barrier0 = em.emit_sync()[0]
        assert barrier0.getMemToken().tokens == [0]
        em.emit_gr_inc(GRIncOp(tensor='A'))  # producer now on buffer1
        barrier1 = em.emit_sync()[0]
        assert barrier1.getMemToken().tokens == [0, 1]

    # -- StinkyTofu pipeline accepts a fully-tagged subtile body --

    def test_tagged_body_passes_consistency_check(self):
        _init_rocisa_gfx1250()
        from Tensile.KernelWriter import KernelWriter
        from Tensile.Components.Subtile.StinkyTofu import buildSubtileStinkyTofuOptions
        from rocisa.code import Module, KernelBody, SignatureBase, Label
        from rocisa.instruction import SEndpgm, SBarrier

        em = _make_token_emitter()
        # One region with a fully-tagged producer -> barrier -> consumer chain.
        tl = _make_tensor_load()
        rd = _make_ds_read()
        em._tagLdsTokens([tl, rd])
        barrier = SBarrier(comment="Barrier")
        barrier.setMemToken(em.memToken.barrierToken())

        body = Module("body")
        body.add(Label("ASM_Start", "start"))
        body.add(tl)
        body.add(barrier)
        body.add(rd)
        body.add(SEndpgm(comment="end"))
        sig = SignatureBase("kernel_name", 1, "V5", 0, [0, 1, 2], 0, 256,
                            totalVgprs=32, totalSgprs=32)
        kb = KernelBody("kernelBody")
        kb.addSignature(sig)
        kb.addBody(body)

        kernel = _stinky_kernel()
        opts = buildSubtileStinkyTofuOptions(kernel, kernel["_StinkyTofuOptLevel"],
                                             _stinky_mock_writer())
        # MemTokenConsistencyCheck runs at kernel scope; consistent tags must
        # not abort and must yield assembly.
        asm = KernelWriter._maybeRunStinkyTofu(_stinky_mock_writer(), kernel, kb,
                                               sig, stinky_module_options=opts)
        assert asm is not None and 'amdgcn-amd-amdhsa--gfx1250' in asm
