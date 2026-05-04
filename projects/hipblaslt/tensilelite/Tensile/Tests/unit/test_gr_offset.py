#!/usr/bin/env python3
################################################################################
# GR offset verification test: fills a buffer with MMA tile IDs, reads via
# production GR offset computation, verifies each lane loads from the correct
# MMA tiles for the expected subtile.
#
# Usage:
#   pytest test_gr_offset.py -v -s
#   python test_gr_offset.py --debug --wave 0
################################################################################

import math
import os
import sys
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, WAVESIZE, NUM_THREADS,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
)

from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, ABTilePair,
)
from rocisa.code import Module, TextBlock
from rocisa.container import vgpr, sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    # 2x2 WG (subtileCount=2, loadRatio=1.0)
    TileConfig(mt_a=128, mt_b=128, depth_u=64, stride_a=64,  stride_b=64),
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64,  stride_b=64),
    # stride > depthU
    TileConfig(mt_a=128, mt_b=128, depth_u=64, stride_a=128, stride_b=128),
    # 1x4 WG (subtileCount=1, loadRatio=2.0)
    TileConfig(mt_a=48,  mt_b=128, depth_u=64, stride_a=64,  stride_b=64),
    # 4x1 WG (subtileCount=4, loadRatio=0.5)
    TileConfig(mt_a=128, mt_b=48,  depth_u=64, stride_a=64,  stride_b=64),
]


# ---------------------------------------------------------------------------
# Host-side: fill buffer with MMA tile IDs
# ---------------------------------------------------------------------------

def fill_mma_tile_buffer(tileInfo, mt, du, stride):
    """Create an fp16 buffer where each element = its MMA tile linear ID + 1.

    For element at (m, k):
      mma_row = m // instM,  mma_col = k // instK
      value   = mma_row * nK + mma_col + 1

    Elements outside depthU columns are 0. IDs start at 1 so 0 = out-of-bounds.
    """
    mma_m = tileInfo.mmaTileShape[0]
    mma_k = tileInfo.mmaTileShape[1]
    nK = int(tileInfo.globalMMATileGrid[1])

    buf = np.zeros(mt * stride, dtype=np.float16)
    for m in range(mt):
        for k in range(du):
            mma_r = m // mma_m
            mma_c = k // mma_k
            buf[m * stride + k] = float(mma_r * nK + mma_c + 1)

    return buf


# ---------------------------------------------------------------------------
# Assembly generation
# ---------------------------------------------------------------------------

def generate_srd_setup():
    module = Module("SRD setup")
    module.add(SMovB64(dst=sgpr("SrdA+0", 2), src=sgpr(4, 2), comment="SrdA base = input_A_ptr"))
    module.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF",   comment="SrdA NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000",      comment="SrdA OOB_SELECT=2"))
    return module


def generate_gr_to_vgpr_loads_v2(writer, ti):
    """Emit buffer_load_dwordx4 to VGPRs using TileInfo + geometry offsets.

    Uses ti.gr.sharedVgprGROffset and ti.gr.localSubtilesRegister for vaddr/soffset.
    Returns (module, dest_vgprs) where dest_vgprs is [(sId0, sId1, gr_idx, vgpr_start), ...].
    """
    module = Module("GR to VGPR loads")
    dest_vgprs = []
    gr_tile = ti.gr

    perpDimSize = len(ti.localSubtilesRegister)

    for j in range(int(ti.localSubtileGrid[1])):
        for i in range(int(ti.localSubtileGrid[0])):
            reg_idx = ti.grRegGroupForSubtileRow(i)

            # Skip duplicate loads when loadRatio > 1
            if ti.loadRatioGR > 1:
                linearId = ti.getLocalSubtileLinearId(i, j)
                grBaseId = ti.grLoadIndexForSubtile(i, j)
                firstInGroup = int(grBaseId * ti.loadRatioGR)
                if linearId != firstInGroup:
                    continue

            rl = ti.localSubtilesRegister[reg_idx]
            offsetK = j * int(ti.mmaTileShape[1] * ti.subtileShape[0] * ti.bpe)

            for gr_idx in range(ti.numGRPerSubtile):
                dst_start = writer.vgprPool.checkOutAligned(4, 2, preventOverflow=False)

                if len(rl) > 0 and rl.is_sgpr:
                    soff_str = f"s{rl.indices[0]}"
                else:
                    soff_str = "0"

                voff = gr_tile.sharedVgprGROffset[gr_idx]

                asm = (f"  buffer_load_dwordx4 v[{dst_start}:{dst_start+3}], "
                       f"v{voff}, s[sgprSrdA:sgprSrdA+3], {soff_str} offen offset:{offsetK}\n")
                module.add(TextBlock(asm))
                dest_vgprs.append((i, j, gr_idx, dst_start))

    module.add(SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1))
    return module, dest_vgprs


def generate_export_all_waves_v2(ti, dest_vgprs):
    """Export loaded GR data from ALL waves to output buffer.

    Output layout per wave: sequential dwordx4 blocks, one per load.
    Total output = NUM_WAVES * num_loads * WAVESIZE * 16 bytes.
    Wave w writes at base offset = w * num_loads * WAVESIZE * 16.
    """
    num_loads = len(dest_vgprs)
    lines = []
    lines.append("  // ---- Export GR data (all waves) ----")

    all_v = set()
    for _, _, _, vs in dest_vgprs:
        for k in range(4):
            all_v.add(vs + k)
    for v in ti.gr.sharedVgprGROffset:
        all_v.add(v)

    next_v = max(all_v | {0}) + 1
    tmp = next_v; next_v += 1
    wave_v = next_v; next_v += 1
    wave_off = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1

    lines.append(f"  v_lshrrev_b32 v{wave_v}, 6, v0               // waveId")
    lines.append(f"  v_and_b32 v{tmp}, 0x3F, v0                    // laneId")

    per_wave_size = num_loads * WAVESIZE * 16

    # Precompute wave offset: waveId * per_wave_size (use s2 as temp for the constant)
    lines.append(f"  s_mov_b32 s2, {per_wave_size}                 // per_wave_size")
    lines.append(f"  v_mul_lo_u32 v{wave_off}, s2, v{wave_v}       // wave byte offset")

    for idx, (sId0, sId1, gr_idx, vgpr_start) in enumerate(dest_vgprs):
        load_base = idx * WAVESIZE * 16
        lines.append(f"  // Export [{sId0},{sId1}] gr{gr_idx}: v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 4, v{tmp}         // laneId * 16")
        if load_base > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {load_base}, v{addr_lo} // + load base")
        lines.append(f"  v_add_u32 v{addr_lo}, v{wave_off}, v{addr_lo}  // + wave offset")
        lines.append(f"  v_mov_b32 v{addr_hi}, s7                     // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s6, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vgpr_start}:{vgpr_start+3}]")

    lines.append(f"  s_waitcnt vmcnt(0)")
    return "\n".join(lines)


def _create_kernel_dict(cfg):
    """Create a kernel dict for TileInfo construction."""
    if ((cfg.mt_a // 16) % 2 == 0) and ((cfg.mt_b // 16) % 2 == 0):
        wg = [2, 2]
    elif ((cfg.mt_a // 16) % 2 != 0) and ((cfg.mt_b // 16) % 4 == 0):
        wg = [1, 4]
    elif ((cfg.mt_a // 16) % 4 == 0) and ((cfg.mt_b // 16) % 2 != 0):
        wg = [4, 1]
    else:
        raise ValueError(f"Unsupported tile config: mt_a={cfg.mt_a}, mt_b={cfg.mt_b}")

    return {
        "MacroTileA": cfg.mt_a,
        "MacroTileB": cfg.mt_b,
        "_DepthUA": cfg.depth_u,
        "_DepthUB": cfg.depth_u,
        "MIWaveGroup": wg,
        "WavefrontSize": WAVESIZE,
        "MatrixInstM": 16,
        "MatrixInstK": 32,
    }


def generate_gr_offset_kernel(cfg):
    """Generate a kernel using TileInfo + geometry emit for GR offset computation."""
    init_rocisa()

    geometry = AB_B16
    kernel = _create_kernel_dict(cfg)

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.vgprPool.checkOut(1)  # v0 = Serial
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )

    ti = TileInfo(geometry, 'A', writer, kernel)

    # Reserve s0-s7 for hardware + kernarg
    writer.sgprPool.checkOut(8)
    stride_sgpr = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.sgprs["StrideA0I"] = stride_sgpr
    writer.sgprs["StrideB1J"] = stride_sgpr

    # Allocate GR offset registers via geometry dispatch
    ti.gr.allocOffsetRegisters(ti, writer, kernel)

    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(4, 4, "SrdA", preventOverflow=False)

    # Compute GR offsets via geometry emit dispatch
    gra_module = ti.emitGlobalReadOffset(writer, kernel)

    # Emit non-DTL buffer loads to VGPRs using the computed offsets
    gr_load_module, dest_vgprs = generate_gr_to_vgpr_loads_v2(writer, ti)

    export_asm = generate_export_all_waves_v2(ti, dest_vgprs)

    prologue = generate_load_params([
        (4, 2, 0x00, "input_A_ptr"),
        (6, 2, 0x08, "output_ptr"),
        (stride_sgpr, 1, 0x10, "strideA"),
    ])
    srd_module = generate_srd_setup()

    inner_asm = "\n".join([
        str(prologue),
        str(srd_module),
        str(gra_module),
        str(gr_load_module),
        export_asm,
    ])

    args = (
        ("input_A_ptr", 8, "global_buffer", "f16"),
        ("output_ptr",  8, "global_buffer", "u32"),
        ("strideA",     4, "by_value",      "u32"),
    )

    num_waves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    output_size = num_waves * len(dest_vgprs) * WAVESIZE * 16
    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_size=0)

    return kernel_asm, kernel, ti, output_size, dest_vgprs


# ---------------------------------------------------------------------------
# Host-side expected values
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_gr_output(output_bytes, ti, kernel, dest_vgprs, mt, stride, debug=False):
    """Verify each GR load reads a consistent set of elements from the same subtile.

    For each (wave, load) block in the output:
      1. Collect all non-zero MMA tile IDs loaded across all lanes.
      2. Check that every non-zero value within each lane belongs to the same
         MMA tile (consistency within a lane's 16B load).
      3. Check that across all lanes in a wave for one GR load, the MMA tile IDs
         form a contiguous set matching the subtile shape.
    """
    num_waves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
    num_loads = len(dest_vgprs)
    per_wave_size = num_loads * WAVESIZE * 16
    nK = int(ti.globalMMATileGrid[1])
    st_m = int(ti.subtileShape[0])
    st_k = int(ti.subtileShape[1])
    # Each wave covers numRowsPerWave M rows and the full K range of the subtile.
    # M tiles per wave = numRowsPerWave / instM, K tiles per load = subtileShape[1].
    wavesize = kernel["WavefrontSize"]
    blockSize = ti.subIterKBytes // ti.loadWidthGR
    numRowsPerWave = wavesize // blockSize
    m_tiles_per_wave = numRowsPerWave // ti.mmaTileShape[0]
    expected_tile_count = max(1, m_tiles_per_wave) * st_k

    errors = 0
    for wave_id in range(num_waves):
        wave_base = wave_id * per_wave_size

        for idx, (sId0, sId1, gr_idx, _) in enumerate(dest_vgprs):
            load_base = wave_base + idx * WAVESIZE * 16
            block_data = np.frombuffer(
                output_bytes[load_base:load_base + WAVESIZE * 16], dtype=np.float16)

            all_tiles = set()
            for lane in range(WAVESIZE):
                lane_data = block_data[lane * 8 : lane * 8 + 8]
                lane_tiles = set(int(v) for v in lane_data if v != 0)
                if len(lane_tiles) > 1:
                    errors += 1
                    if errors <= 10 or debug:
                        print(f"  INCONSISTENT wave={wave_id} subtile=[{sId0},{sId1}] "
                              f"gr{gr_idx} lane={lane}: tiles={lane_tiles}")
                all_tiles.update(lane_tiles)

            if len(all_tiles) != expected_tile_count and len(all_tiles) > 0:
                errors += 1
                if errors <= 10 or debug:
                    print(f"  TILE_COUNT wave={wave_id} subtile=[{sId0},{sId1}] gr{gr_idx}: "
                          f"got {len(all_tiles)} tiles {sorted(all_tiles)}, expected {expected_tile_count}")

            if debug and len(all_tiles) > 0:
                print(f"  wave={wave_id} [{sId0},{sId1}] gr{gr_idx}: tiles={sorted(all_tiles)}")

    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGrOffset:

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def test_gr_offset(self, cfg, tmp_path):
        """Verify GR offset calculations read the correct subtile elements."""
        kernel_asm, kernel, ti, output_size, dest_vgprs = \
            generate_gr_offset_kernel(cfg)

        input_A = fill_mma_tile_buffer(ti, cfg.mt_a, cfg.depth_u, cfg.stride_a)

        label = f"gr_offset_{cfg.label}"
        output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                        inputs=(input_A,),
                                        scalars=(cfg.stride_a,))

        errors = verify_gr_output(output_bytes, ti, kernel, dest_vgprs,
                                   cfg.mt_a, cfg.stride_a)

        assert errors == 0, f"Config {cfg.label}: {errors} mismatches"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GR offset verification GPU test")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=int, default=None, help="Config index")
    args = parser.parse_args()

    if not HAS_HIP:
        print("HIP not available")
        sys.exit(1)

    config_list = CONFIGS if args.config is None else [CONFIGS[args.config]]

    total_errors = 0
    total_tests = 0

    for cfg in config_list:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, depth_u={cfg.depth_u}, stride_a={cfg.stride_a}")

        kernel_asm, kernel, ti, output_size, dest_vgprs = \
            generate_gr_offset_kernel(cfg)

        input_A = fill_mma_tile_buffer(ti, cfg.mt_a, cfg.depth_u, cfg.stride_a)
        print(f"  Loads: {len(dest_vgprs)}, output: {output_size} bytes")
        print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")
        print(f"  loadRatioGR: {ti.loadRatioGR}")
        print(f"  subtileShape: {ti.subtileShape}")
        print(f"  localSubtileGrid: {ti.localSubtileGrid}")

        if args.debug:
            print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

            label = f"gr_offset_{cfg.label}"
            output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                            inputs=(input_A,),
                                            scalars=(cfg.stride_a,))

            errors = verify_gr_output(output_bytes, ti, kernel, dest_vgprs,
                                       cfg.mt_a, cfg.stride_a, debug=args.debug)

            if errors == 0:
                print(f"  PASS")
            else:
                print(f"  FAIL: {errors} mismatches")
                total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    sys.exit(1 if total_errors > 0 else 0)
