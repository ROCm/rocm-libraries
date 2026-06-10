#!/usr/bin/env python3
################################################################################
# Step 4 LDS→MFMA validation: LR ds_read_b64_tr_b4 emission for NN-FP4 (TLU1).
# Pure-CPU test (no GPU / no hip).
#
# Verifies:
#   * Dispatch routes to _emitLR_TLU1, not stub.
#   * Emit produces a non-empty Module with assembly.
#   * Correct total instruction count (localGrid0 * localGrid1 * numLRPerSubtile).
#   * Each instruction is ds_read_b64_tr_b4 (verified via class type).
#   * DS offset values match the expected formula.
#   * Destination VGPRs cover the full vgprTile array without gaps or overlap.
#
#   pytest test_nn_f4_lr_load.py -v   |   python test_nn_f4_lr_load.py
################################################################################
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
from rocisa.instruction import DSLoadB64TrB4
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B4_TLU1
from gpu_test_helpers import create_writer, TileConfig, init_rocisa

# ── Constants matching AB_B4_TLU1 with MT=256, DU=256, MIWaveGroup=(2,2) ─────
SUBTILE_SIZE = 2048
GLOBAL_GRID_0 = 8  # globalSubtileGrid[0]
LOCAL_GRID_0 = 4  # lrLocalSubtileGrid[0]
LOCAL_GRID_1 = 2  # lrLocalSubtileGrid[1]
NUM_LR_PER_SUBTILE = 4
TILES_PER_SUBTILE = 2
READS_PER_TILE = NUM_LR_PER_SUBTILE // TILES_PER_SUBTILE  # 2
REGS_PER_READ = 2
TOTAL_READS = LOCAL_GRID_0 * LOCAL_GRID_1 * NUM_LR_PER_SUBTILE  # 32


# ── Helpers for parsing DS instruction assembly text ──────────────────────────
def _parse_ds_offset(inst):
    """Extract DS offset from instruction assembly text (e.g. 'offset:1024')."""
    m = re.search(r"offset:(\d+)", str(inst))
    return int(m.group(1)) if m else 0


def _parse_src_vgpr(inst):
    """Extract source (address) VGPR index from assembly text (e.g. ', v5')."""
    m = re.search(r",\s*v(\d+)", str(inst))
    return int(m.group(1)) if m else -1


def _setup():
    """Create writer/kernel/tileInfo with all registers allocated."""
    cfg = TileConfig(mt_a=256, mt_b=256, depth_u=256, stride_a=256, stride_b=256)
    writer, kernel, tiA, _ = create_writer(
        cfg, mi_wave_group=[2, 2], geometry=AB_B4_TLU1, inst_k=128, bpe=1
    )
    init_rocisa()
    tiA.allocOffsetRegisters(writer, kernel)
    tiA.allocVgprTileRegisters_legacy(writer, kernel)
    tiA.emitLocalReadOffset(writer, kernel)
    module = tiA.emitLocalRead(writer, kernel)
    return module, tiA


def _collect_ds_reads(module):
    """Recursively collect all DSLoadB64TrB4 instructions from module tree."""
    from rocisa.code import Module as Mod

    results = []
    for item in module.items():
        if isinstance(item, Mod):
            results.extend(_collect_ds_reads(item))
        elif isinstance(item, DSLoadB64TrB4):
            results.append(item)
    return results


# ── Test 1: Dispatch routes to real function ─────────────────────────────────
def test_dispatch_not_stub():
    """_emitLocalRead must dispatch to _emitLR_TLU1 for LRTag_TLU1."""
    from Tensile.Components.Subtile.SubtileLREmit import _emitLocalRead
    from Tensile.Components.Subtile.SubtileGeometry import LRTag_TLU1

    name = _emitLocalRead.dispatch(LRTag_TLU1).__name__
    assert name == "_emitLR_TLU1", f"dispatches to {name}, expected _emitLR_TLU1"


# ── Test 2: Emit produces a non-empty Module ─────────────────────────────────
def test_emit_produces_module():
    """Calling emitLocalRead returns a Module with ds_read instructions."""
    module, _ = _setup()
    assert module is not None, "emitLocalRead returned None (still stub?)"
    asm = str(module)
    assert len(asm) > 100, f"Module too short ({len(asm)} chars), likely empty"
    assert (
        "ds_read" in asm.lower() or "ds_load" in asm.lower()
    ), "No ds_read/ds_load instruction found in emitted asm"


# ── Test 3: Correct total instruction count ──────────────────────────────────
def test_instruction_count():
    """Must emit exactly LOCAL_GRID_0 * LOCAL_GRID_1 * NUM_LR_PER_SUBTILE reads."""
    module, _ = _setup()
    reads = _collect_ds_reads(module)
    assert (
        len(reads) == TOTAL_READS
    ), f"Expected {TOTAL_READS} ds_read_b64_tr_b4 instructions, got {len(reads)}"


# ── Test 4: DS offset values match formula ───────────────────────────────────
def test_ds_offsets():
    """Each ds_read's DS offset must equal i*subtileSize + j*globalGrid0*subtileSize."""
    module, _ = _setup()
    reads = _collect_ds_reads(module)
    expected_offsets = []
    for i in range(LOCAL_GRID_0):
        for j in range(LOCAL_GRID_1):
            ds_off = i * SUBTILE_SIZE + j * GLOBAL_GRID_0 * SUBTILE_SIZE
            for _ in range(NUM_LR_PER_SUBTILE):
                expected_offsets.append(ds_off)
    for idx, (inst, expected) in enumerate(zip(reads, expected_offsets)):
        actual = _parse_ds_offset(inst)
        assert (
            actual == expected
        ), f"Read[{idx}]: ds_offset={actual}, expected={expected}"


# ── Test 5: Destination VGPRs cover all tiles without overlap ────────────────
def test_dst_vgpr_coverage():
    """Each read targets a unique 2-VGPR slot; all MMA tile VGPRs are written."""
    module, tiA = _setup()
    reads = _collect_ds_reads(module)
    dst_ranges = set()
    for inst in reads:
        dst_start = inst.dst.regIdx
        dst_ranges.add(dst_start)
    total_tiles = LOCAL_GRID_0 * LOCAL_GRID_1 * TILES_PER_SUBTILE
    expected_slots = total_tiles * READS_PER_TILE
    assert (
        len(dst_ranges) == expected_slots
    ), f"Expected {expected_slots} unique dst starts, got {len(dst_ranges)}"


# ── Test 6: Destination VGPRs align with vgprTiles ───────────────────────────
def test_dst_matches_vgpr_tiles():
    """Each read's dst must be a valid offset within the corresponding vgprTile."""
    module, tiA = _setup()
    reads = _collect_ds_reads(module)
    read_idx = 0
    for i in range(LOCAL_GRID_0):
        for j in range(LOCAL_GRID_1):
            for r in range(NUM_LR_PER_SUBTILE):
                mfmaId = r // READS_PER_TILE
                readInTile = r % READS_PER_TILE
                tileIdx = tiA.lrTileIndexForSubtile(i, j, mfmaId)
                expected_start = (
                    tiA.vgprTiles[tileIdx].regList.indices[0]
                    + readInTile * REGS_PER_READ
                )
                actual_start = reads[read_idx].dst.regIdx
                assert actual_start == expected_start, (
                    f"Read[{read_idx}] (i={i},j={j},r={r}): "
                    f"dstVgpr={actual_start}, expected={expected_start}"
                )
                read_idx += 1


# ── Test 7: Address VGPRs cycle through sharedVgprLROffset correctly ─────────
def test_addr_vgpr_cycling():
    """Address VGPR for read r must be sharedVgprLROffset[r] (cycles per subtile)."""
    module, tiA = _setup()
    reads = _collect_ds_reads(module)
    lr_tile = tiA.lr  # ABLRTile instance
    read_idx = 0
    for i in range(LOCAL_GRID_0):
        for j in range(LOCAL_GRID_1):
            for r in range(NUM_LR_PER_SUBTILE):
                expected_addr = lr_tile.sharedVgprLROffset[r]
                actual_addr = _parse_src_vgpr(reads[read_idx])
                assert actual_addr == expected_addr, (
                    f"Read[{read_idx}] (i={i},j={j},r={r}): "
                    f"addrVgpr={actual_addr}, expected={expected_addr}"
                )
                read_idx += 1


# ── Test 8: Max LDS address stays within 64KB ────────────────────────────────
def test_max_lds_address():
    """Maximum possible LDS address (vgpr + ds_offset) must be < 65536."""
    max_vgpr_addr = (
        2040 + (2 - 1) * LOCAL_GRID_0 * SUBTILE_SIZE
    )  # wave 1, lane 63, read 3
    max_ds_offset = (LOCAL_GRID_0 - 1) * SUBTILE_SIZE + (
        LOCAL_GRID_1 - 1
    ) * GLOBAL_GRID_0 * SUBTILE_SIZE
    total = max_vgpr_addr + max_ds_offset
    assert total < 65536, f"Max LDS addr = {total} >= 65536 (overflow!)"


if __name__ == "__main__":
    test_dispatch_not_stub()
    test_emit_produces_module()
    test_instruction_count()
    test_ds_offsets()
    test_dst_vgpr_coverage()
    test_dst_matches_vgpr_tiles()
    test_addr_vgpr_cycling()
    test_max_lds_address()
    print(
        "OK: LR Load TLU1 emits 32 ds_read_b64_tr_b4 with correct offsets and destinations."
    )
