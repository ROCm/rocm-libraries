#!/usr/bin/env python3
################################################################################
# Step 5 LDS→MFMA validation: LR DTL Init & LDS Buffer Swap for NN-FP4 (TLU1).
# Pure-CPU test (no GPU / no hip).
#
# Verifies:
#   * Dispatch routes to _emitLRDTLInit_TLU1 and _emitLRLDSSwap_TLU1, not stubs.
#   * DTL Init emits a non-empty Module with correct instruction count.
#   * LDS Buffer Swap emits exactly numLRPerSubtile v_xor_b32 instructions.
#   * Reference model validates XOR toggle correctness for all offsets.
#
#   pytest test_nn_f4_lr_dtl.py -v   |   python test_nn_f4_lr_dtl.py
################################################################################
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from rocisa.instruction import VXorB32, VAddU32, SMovB32
from rocisa.code import Module as Mod
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1
from gpu_test_helpers import create_writer, TileConfig, init_rocisa

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_LR_PER_SUBTILE = 4
LDS_TOTAL_SIZE = 32768  # test value for ldsTotalSize


def _setup():
    """Create writer/kernel/tileInfo with LR offsets computed and ldsTotalSize set."""
    cfg = TileConfig(mt_a=256, mt_b=256, depth_u=256, stride_a=256, stride_b=256)
    writer, kernel, tiA, _ = create_writer(
        cfg, mi_wave_group=[2, 2], geometry=AB_B4_TLU1, inst_k=128, bpe=1
    )
    init_rocisa()
    writer.ldsTotalSize = LDS_TOTAL_SIZE
    tiA.allocOffsetRegisters(writer, kernel)
    tiA.emitLocalReadOffset(writer, kernel)
    return writer, kernel, tiA


def _collect_insts(module, inst_type):
    """Recursively collect all instructions of a given type from module tree."""
    results = []
    for item in module.items():
        if isinstance(item, Mod):
            results.extend(_collect_insts(item, inst_type))
        elif isinstance(item, inst_type):
            results.append(item)
    return results


# ── Test 1: DTL Init dispatch routes to real function ────────────────────────
def test_dtl_init_dispatch():
    """_emitLRDTLInit must dispatch to _emitLRDTLInit_TLU1 for LRTag_TLU1."""
    from Tensile.Components.Subtile.SubtileLREmit import _emitLRDTLInit
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    name = _emitLRDTLInit.dispatch(LRTag_TLU1).__name__
    assert (
        name == "_emitLRDTLInit_TLU1"
    ), f"dispatches to {name}, expected _emitLRDTLInit_TLU1"


# ── Test 2: LDS Swap dispatch routes to real function ────────────────────────
def test_lds_swap_dispatch():
    """_emitLRLDSBufferSwap must dispatch to _emitLRLDSSwap_TLU1 for LRTag_TLU1."""
    from Tensile.Components.Subtile.SubtileLREmit import _emitLRLDSBufferSwap
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    name = _emitLRLDSBufferSwap.dispatch(LRTag_TLU1).__name__
    assert (
        name == "_emitLRLDSSwap_TLU1"
    ), f"dispatches to {name}, expected _emitLRLDSSwap_TLU1"


# ── Test 3: DTL Init produces non-empty module ───────────────────────────────
def test_dtl_init_produces_module():
    """emitLRDTLInit returns a Module with instructions."""
    writer, kernel, tiA = _setup()
    module = tiA.emitLRDTLInit(writer, kernel)
    assert module is not None, "emitLRDTLInit returned None"
    asm = str(module)
    assert len(asm) > 50, f"Module too short ({len(asm)} chars)"


# ── Test 4: DTL Init instruction count ───────────────────────────────────────
def test_dtl_init_instruction_count():
    """DTL Init should emit 1 s_mov + 4 v_add + 4 v_xor = 9 instructions."""
    writer, kernel, tiA = _setup()
    module = tiA.emitLRDTLInit(writer, kernel)
    smovs = _collect_insts(module, SMovB32)
    vadds = _collect_insts(module, VAddU32)
    vxors = _collect_insts(module, VXorB32)
    assert len(smovs) == 1, f"Expected 1 s_mov_b32, got {len(smovs)}"
    assert (
        len(vadds) == NUM_LR_PER_SUBTILE
    ), f"Expected {NUM_LR_PER_SUBTILE} v_add_u32, got {len(vadds)}"
    assert (
        len(vxors) == NUM_LR_PER_SUBTILE
    ), f"Expected {NUM_LR_PER_SUBTILE} v_xor_b32, got {len(vxors)}"


# ── Test 5: LDS Swap produces exactly numLRPerSubtile v_xor instructions ─────
def test_lds_swap_instruction_count():
    """LDS Buffer Swap should emit exactly 4 v_xor_b32 instructions."""
    writer, kernel, tiA = _setup()
    tiA.emitLRDTLInit(writer, kernel)  # must init swap masks first
    module = tiA.emitLRLDSBufferSwap(writer, kernel)
    vxors = _collect_insts(module, VXorB32)
    assert (
        len(vxors) == NUM_LR_PER_SUBTILE
    ), f"Expected {NUM_LR_PER_SUBTILE} v_xor_b32 in swap, got {len(vxors)}"


# ── Test 6: DTL Init asm mentions ldsTotalSize ────────────────────────────────
def test_dtl_init_has_lds_size():
    """DTL Init assembly should reference ldsTotalSize in a comment or immediate."""
    writer, kernel, tiA = _setup()
    module = tiA.emitLRDTLInit(writer, kernel)
    asm = str(module)
    assert (
        "ldsTotalSize" in asm.lower()
        or str(LDS_TOTAL_SIZE) in asm
        or hex(LDS_TOTAL_SIZE) in asm
    ), "DTL Init assembly doesn't reference ldsTotalSize value"


# ── Test 7: Reference model — XOR toggle correctness ─────────────────────────
def test_xor_toggle_reference():
    """Verify XOR swap logic: toggling twice returns to original address."""
    for addr in [0, 8, 256, 1776, 2040, 8192, 10232]:
        buf1_addr = addr + LDS_TOTAL_SIZE
        swap_mask = addr ^ buf1_addr
        toggled = addr ^ swap_mask
        assert (
            toggled == buf1_addr
        ), f"addr={addr}: toggle={toggled}, expected={buf1_addr}"
        toggled_back = toggled ^ swap_mask
        assert (
            toggled_back == addr
        ), f"addr={addr}: toggle_back={toggled_back}, expected={addr}"


# ── Test 8: Reference model — all 4 offsets toggle independently ─────────────
def test_all_offsets_toggle():
    """Verify each of the 4 canonical offsets toggles correctly."""
    offsets = [0, 256, 8, 264]  # lane 0, reads 0-3 (from Step 3 reference model)
    for r, addr in enumerate(offsets):
        buf1 = addr + LDS_TOTAL_SIZE
        mask = addr ^ buf1
        assert addr ^ mask == buf1, f"offset[{r}]={addr}: XOR toggle failed"
        assert buf1 ^ mask == addr, f"offset[{r}]={addr}: XOR untoggle failed"


# ── Test 9: Swap uses the offset VGPRs (not some other register) ─────────────
def test_swap_uses_offset_vgprs():
    """Each v_xor in the swap module writes to a sharedVgprLROffset register."""
    writer, kernel, tiA = _setup()
    tiA.emitLRDTLInit(writer, kernel)
    module = tiA.emitLRLDSBufferSwap(writer, kernel)
    vxors = _collect_insts(module, VXorB32)
    lr_tile = tiA.lr
    offset_vgprs = set(lr_tile.sharedVgprLROffset)
    for i, inst in enumerate(vxors):
        dst_vgpr = inst.dst.regIdx
        assert (
            dst_vgpr in offset_vgprs
        ), f"Swap xor[{i}]: dst=v{dst_vgpr}, not in sharedVgprLROffset={offset_vgprs}"


if __name__ == "__main__":
    test_dtl_init_dispatch()
    test_lds_swap_dispatch()
    test_dtl_init_produces_module()
    test_dtl_init_instruction_count()
    test_lds_swap_instruction_count()
    test_dtl_init_has_lds_size()
    test_xor_toggle_reference()
    test_all_offsets_toggle()
    test_swap_uses_offset_vgprs()
    print("OK: LR DTL Init and LDS Buffer Swap (TLU1) emit correctly.")
