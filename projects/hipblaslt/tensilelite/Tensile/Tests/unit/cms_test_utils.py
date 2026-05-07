################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Mock builders for CMS Validator unit tests.

These create minimal rocisa instruction objects that satisfy the validator's
type inspection and register extraction without requiring a full kernel writer.

Register spaces are widely separated to avoid collisions:
    LRA:               v[1000..]   (LR[i].dst = v[1000 + i*4 : +4])
    LRB:               v[2000..]
    PackA output:      v[5000..]   (Pack dst -> MFMA .a)
    PackB output:      v[6000..]   (Pack dst -> MFMA .b)
    MFMA acc:          v[7000..]
"""

from rocisa.instruction import (
    MFMAInstruction, VPermB32, DSLoadB128, SWaitCnt, SBarrier, SNop,
    SMovB32, SAddU32, BufferLoadB128, SCmpEQU32, SCSelectB32,
    SAddCU32, SSubU32,
)
from rocisa.container import vgpr, sgpr
from rocisa.enum import InstType

# --- Register space constants ---
LRA_BASE = 1000
LRB_BASE = 2000
PACK_A_DST_BASE = 5000
PACK_B_DST_BASE = 6000
MFMA_ACC_BASE = 7000


def make_mock_mfma_code(num_mfma) -> list:
    """Build a list of mock MFMAInstruction objects.

    Each MFMA reads from PackA output space (.a) and PackB output space (.b),
    matching the register chains set up by _make_mock_packs_*.
    """
    if not num_mfma:
        return []
    mfmas = []
    for i in range(int(num_mfma)):
        acc_start = MFMA_ACC_BASE + i * 4
        a_start = PACK_A_DST_BASE + i * 4
        b_start = PACK_B_DST_BASE + i * 4
        mfmas.append(MFMAInstruction(
            instType=InstType.INST_BF16, accType=InstType.INST_F32,
            variant=[16, 16, 32, 1], mfma1k=False,
            acc=vgpr(acc_start, 4), a=vgpr(a_start, 4),
            b=vgpr(b_start, 4), acc2=vgpr(acc_start, 4),
        ))
    return mfmas


def _make_mock_lr(count: int, base_reg: int = LRA_BASE) -> list:
    """Create mock DSLoadB128 instructions for local reads.

    LR[i] writes 4 VGPRs: v[base_reg + i*4 : base_reg + i*4 + 4].
    """
    return [DSLoadB128(dst=vgpr(base_reg + i * 4, 4), src=vgpr(0, 1))
            for i in range(count)]


def _make_mock_packs_bf16(count: int, pack_dst_base: int = PACK_A_DST_BASE,
                          lr_base: int = LRA_BASE, num_lrs: int = 8) -> list:
    """Create mock VPermB32 instructions for BF16 packs.

    Each pack reads from TWO different LRs (D0 and D1 dimensions), matching
    the positional code's element_idx mapping:
      element_idx = i % num_element_pairs
      LR indices: element_idx*2 (D0) and element_idx*2+1 (D1)

    Pack[i]: src0 = one VGPR from LR[element_idx*2 + 1]
             src1 = one VGPR from LR[element_idx*2]
             dst  = v[pack_dst_base + i]
    """
    num_element_pairs = max(num_lrs // 2, 1)
    items = []
    for i in range(count):
        element_idx = i % num_element_pairs
        # src0 from LR[element_idx*2 + 1], src1 from LR[element_idx*2]
        # Each LR occupies 4 VGPRs at lr_base + lr_idx*4
        lr_idx_0 = element_idx * 2      # D0 LR
        lr_idx_1 = element_idx * 2 + 1  # D1 LR
        items.append(VPermB32(
            dst=vgpr(pack_dst_base + i, 1),
            src0=vgpr(lr_base + lr_idx_1 * 4, 1),  # one reg from D1 LR
            src1=vgpr(lr_base + lr_idx_0 * 4, 1),  # one reg from D0 LR
            src2=sgpr(0, 1),
        ))
    return items


def _make_mock_gr(count: int, dtl: bool = True) -> list:
    """Create mock GR instruction sequences.

    For DTL=1: interleaved [SMovB32, BufferLoadB128] pairs (count must be even).
    For DTL=0: all BufferLoadB128 (any count).
    """
    items = []
    if dtl:
        # DTL=1: interleaved SMovB32/SAddU32 + BufferLoadB128 pairs
        for i in range(count):
            if i % 2 == 0:
                if i == 0:
                    items.append(SMovB32(dst=sgpr(0, 1), src=sgpr(1, 1)))
                else:
                    items.append(SAddU32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=4096))
            else:
                items.append(BufferLoadB128(dst=None, vaddr=vgpr(i, 1),
                                           saddr=sgpr(0, 4), soffset=0))
    else:
        # Non-DTL: all are buffer loads
        for i in range(count):
            items.append(BufferLoadB128(dst=None, vaddr=vgpr(i, 1),
                                       saddr=sgpr(0, 4), soffset=0))
    return items


def _make_mock_grinc(count: int) -> list:
    """Create mock scalar instructions for GRInc sequences."""
    items = []
    for i in range(count):
        items.append(SCmpEQU32(src0=sgpr(0, 1), src1=sgpr(1, 1)))
    return items


def _make_mock_swap(count: int) -> list:
    """Mock LRSA/LRSB pointer-flip ops as VXorB32 (the rocisa class CMS
    actually emits for LR-side swap operations)."""
    from rocisa.instruction import VXorB32
    return [VXorB32(dst=vgpr(0, 1), src0=vgpr(0, 1), src1=vgpr(1, 1))
            for _ in range(count)]


def _make_mock_lw_swap(count: int) -> list:
    """Create mock SXorB32 for LW swap operations."""
    from rocisa.instruction import SXorB32
    return [SXorB32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=sgpr(1, 1))
            for _ in range(count)]


def _make_mock_lcc() -> list:
    """Create mock loop counter code (SSubU32 + SCmpEQI32)."""
    from rocisa.instruction import SCmpEQI32
    return [
        SSubU32(dst=sgpr(0, 1), src0=sgpr(0, 1), src1=1),
        SCmpEQI32(src0=sgpr(0, 1), src1=2),
    ]



def make_mock_id_map(schedule_info, kernel=None) -> dict:
    """Build a minimal mock id_map from a ScheduleInfo.

    For each key in optSchedule, creates a list of mock rocisa instruction
    objects of the correct length with the correct types for the validator's
    type resolution to work.
    """
    id_map = {}
    opt = schedule_info.optSchedule

    for key in opt:
        # Get the number of instructions from the first code path
        val = opt[key]
        if not val:
            count = 0
        elif isinstance(val[0], list):
            count = len(val[0])
        else:
            count = len(val)  # Single code path — val is the list directly

        if key == "SYNC":
            id_map[key] = list(schedule_info.syncCode) if schedule_info.syncCode else []
        elif key == "SNOP":
            id_map[key] = list(schedule_info.snopCode) if schedule_info.snopCode else []
        elif key.startswith("LRA") and not key.startswith("LRS"):
            id_map[key] = _make_mock_lr(count, base_reg=LRA_BASE)
        elif key.startswith("LRB") and not key.startswith("LRS"):
            id_map[key] = _make_mock_lr(count, base_reg=LRB_BASE)
        elif key == "GRA":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRB":
            id_map[key] = _make_mock_gr(count, dtl=kernel.get("DirectToLds", True) if kernel else True)
        elif key == "GRIncA":
            id_map[key] = _make_mock_grinc(count)
        elif key == "GRIncB":
            id_map[key] = _make_mock_grinc(count)
        elif key.startswith("Pack"):
            is_a = key.startswith("PackA")
            lr_base = LRA_BASE if is_a else LRB_BASE
            pack_dst_base = PACK_A_DST_BASE if is_a else PACK_B_DST_BASE
            # Find corresponding LR count for element_idx mapping
            lr_name = key.replace("Pack", "LR")
            lr_val = opt.get(lr_name, [])
            if lr_val and isinstance(lr_val[0], list):
                num_lrs = len(lr_val[0])
            elif lr_val:
                num_lrs = len(lr_val)
            else:
                num_lrs = 8  # default
            id_map[key] = _make_mock_packs_bf16(count,
                                                 pack_dst_base=pack_dst_base,
                                                 lr_base=lr_base,
                                                 num_lrs=num_lrs)
        elif key == "LRSA" or key == "LRSB":
            id_map[key] = _make_mock_swap(count)
        elif key == "LWSA" or key == "LWSB":
            id_map[key] = _make_mock_lw_swap(count)
        elif key == "LWA" or key == "LWB":
            id_map[key] = []  # Empty for DTL=1
        elif key == "LCC":
            # Create enough LCC entries — repeat the pattern if count > 2
            base = _make_mock_lcc()
            id_map[key] = (base * ((count // len(base)) + 1))[:count]
        else:
            # Unknown key — create empty list
            id_map[key] = [None] * count

    return id_map


# --- Helpers for transitioning tests to real idMap ---

def kernel_to_solution_config(kernel):
    """Convert a test kernel dict to a Solution-compatible config.

    Test kernels use a 4-element MatrixInstruction plus separate MIWaveTileA/B
    and MIWaveGroup fields, and mock DataType objects. Solutions need a 9-element
    MatrixInstruction and real ProblemType dict strings.

    Many test kernels have intentionally minimal configs (e.g., MIWaveTileA=2,
    GlobalReadVectorWidthA=0) that don't correspond to any real CMS-supported
    kernel. This function maps them to a known-good config that:
    1. Produces a valid Solution with CMS support
    2. Has the same datatype (BF16/TF32/FP32) and layout (TN/NN/etc.)
    3. Has enough instructions for subsetting
    """
    pt = kernel.get("ProblemType", {})
    transA = pt.get("TransposeA", False)
    transB = pt.get("TransposeB", False)
    is_tf32 = kernel.get("UseF32XEmulation", False)

    # Determine data type. TF32 kernels use FP32 with F32XdlMathOp.
    # Non-TF32 FP32 kernels have no CMS schedules — use Half instead,
    # which produces the same v_perm_b32 Pack instructions.
    dt = pt.get("DataType")
    if is_tf32:
        dtype = "S"
    elif dt and hasattr(dt, 'numBytes') and dt.numBytes() == 2:
        dtype = "H"
    else:
        # FP32 without TF32 → use Half for CMS schedule compatibility
        dtype = "H"

    # Build ProblemType
    pt_config = {
        "OperationType": "GEMM", "UseBeta": True, "Batched": True,
        "TransposeA": transA, "TransposeB": transB,
        "DataType": dtype, "DestDataType": dtype,
    }
    if is_tf32:
        pt_config["F32XdlMathOp"] = "X"
    if dtype == "H":
        pt_config["HighPrecisionAccumulate"] = True

    # Use the test's actual MI/tile config if it would produce a valid CMS kernel,
    # otherwise use a known-good config.
    mi4 = kernel.get("MatrixInstruction", [16, 16, 32, 1])
    wave_tile_a = kernel.get("MIWaveTileA", 4)
    wave_tile_b = kernel.get("MIWaveTileB", 4)
    wave_group = kernel.get("MIWaveGroup", [2, 2])
    if not wave_group:
        wave_group = [2, 2]
    depth_u = kernel.get("DepthU", 64)
    vw_a = kernel.get("VectorWidthA", 1)
    vw_b = kernel.get("VectorWidthB", 1)

    # If the tile is too small for CMS (no registered schedule), scale up.
    # Use TN layout (most schedules are registered for TN) and a known-good tile.
    macro_tile_0 = mi4[0] * wave_group[0] * wave_tile_a
    macro_tile_1 = mi4[1] * wave_group[1] * wave_tile_b
    if macro_tile_0 < 128 or macro_tile_1 < 128:
        if is_tf32:
            wave_tile_a = max(wave_tile_a, 4)
            wave_tile_b = max(wave_tile_b, 4)
        else:
            wave_tile_a = max(wave_tile_a, 8)
            wave_tile_b = max(wave_tile_b, 8)
        # Force TN layout — most CMS schedules are registered for TN
        transA = True
        transB = False
        pt_config["TransposeA"] = True
        pt_config["TransposeB"] = False

    # Use `... if key in kernel else default` rather than `kernel.get(key) or default`
    # because `or` clobbers valid 0 / False values (e.g., TransposeLDS=0 for NT layouts).
    def _override(key, default):
        return kernel[key] if key in kernel else default

    config = {
        "ProblemType": pt_config,
        "MatrixInstruction": mi4 + [1, wave_tile_a, wave_tile_b] + wave_group,
        "DepthU": depth_u,
        "PrefetchGlobalRead": _override("PrefetchGlobalRead", 2),
        "PrefetchLocalRead": _override("PrefetchLocalRead", 1),
        "DirectToLds": _override("DirectToLds", 1),
        "TransposeLDS": _override("TransposeLDS", 1),
        "LocalReadVectorWidth": _override("LocalReadVectorWidth", 8 if dtype == "H" else 4),
        "GlobalReadVectorWidthA": _override("GlobalReadVectorWidthA", 8 if dtype == "H" else 4),
        "GlobalReadVectorWidthB": _override("GlobalReadVectorWidthB", 8 if dtype == "H" else 4),
        "UseCustomMainLoopSchedule": 1,
        "StreamK": _override("StreamK", 0),
        "SourceSwap": _override("SourceSwap", 1),
        "ExpandPointerSwap": _override("ExpandPointerSwap", 0),
    }

    if "ISA" in kernel:
        config["ISA"] = kernel["ISA"]

    # Copy additional parameters that affect schedule generation
    for key in ["LDSTrInst", "StaggerU", "1LDSBuffer", "VectorWidthA", "VectorWidthB",
                "ForceUnrollSubIter", "ClusterLocalRead", "LdsPadA", "LdsPadB",
                "UseSgprForGRO", "NonTemporalA", "NonTemporalB", "NonTemporalD",
                "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "MIArchVgpr"]:
        if key in kernel:
            config[key] = kernel[key]

    return config


def subset_id_map(real_id_map, optSchedule, syncCode=None, snopCode=None,
                  fallback_id_map=None):
    """Subset a real idMap to match a custom schedule's instruction counts.

    For each key in optSchedule, truncates the idMap entry to the number
    of instructions in the schedule's first code path.

    SYNC and SNOP are special: they come from the test's syncCode/snopCode
    arguments, not from the real idMap (which has the production schedule's
    SWaitCnt objects with different dscnt values).

    Keys in real_id_map not in optSchedule are kept as-is.

    If fallback_id_map is provided, any optSchedule key missing from
    real_id_map is sourced from fallback_id_map (typically the mock id_map).
    This is the escape hatch for tests whose schedules reference
    sub-iteration keys (e.g. PackA3/PackB3 from ForceUnrollSubIter) that no
    registered CMS solution produces — the swap-pack registers in the keys
    that DO match are real, while auxiliary keys fall back to mocks.
    """
    id_map = dict(real_id_map)

    for key, val in optSchedule.items():
        if key == "SYNC":
            id_map["SYNC"] = list(syncCode) if syncCode else []
            continue
        if key == "SNOP":
            id_map["SNOP"] = list(snopCode) if snopCode else []
            continue

        # Determine count from first code path
        if not val:
            count = 0
        elif isinstance(val[0], list):
            count = len(val[0])
        else:
            count = len(val)

        if key in id_map:
            id_map[key] = id_map[key][:count]
        elif fallback_id_map is not None and key in fallback_id_map:
            id_map[key] = fallback_id_map[key][:count]
        # else: leave the key absent; validator will error if it needs it.

    return id_map


def _frozen_config_key(config):
    """Create a hashable cache key from a Solution config dict."""
    import json

    def _make_serializable(obj):
        if hasattr(obj, '_asdict'):
            return tuple(obj)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        return obj

    return json.dumps(_make_serializable(config), sort_keys=True)


# --- Real idMap generation from kernel writer ---

def _make_solution(kernel_config, asm, isaInfoMap):
    """Build a Solution from `kernel_config`, deriving the 9-element
    MI parameters when the input carries the Tensile 4-element shape and
    falling back to the first ISA in `isaInfoMap` if the config doesn't
    pin one. Asserts the resulting Solution is Valid (rather than letting
    a silently-rejected solution propagate to the kernel writer)."""
    from copy import deepcopy
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Problem import ProblemType
    from Tensile.TensileLogic.HandleCustomKernel import matrixInstructionToMIParameters

    requested_isa = kernel_config.get('ISA')
    if requested_isa is not None:
        if requested_isa not in isaInfoMap:
            raise ValueError(
                f"Requested ISA {requested_isa} not present in isaInfoMap "
                f"(have {list(isaInfoMap.keys())}). Update conftest.py "
                f"isa_infrastructure to probe this ISA."
            )
        isa = requested_isa
    else:
        isa = next(iter(isaInfoMap.keys()))

    # Build ProblemType if given as a dict
    pt_config = kernel_config.get('ProblemType', {})
    if isinstance(pt_config, dict):
        pt = ProblemType(pt_config, False)
        config = dict(kernel_config)
        config['ProblemType'] = deepcopy(pt.state)
    else:
        config = dict(kernel_config)

    config['ISA'] = isa
    config.setdefault('KernelLanguage', 'Assembly')
    config.setdefault('WavefrontSize', 64)
    config.setdefault('WorkGroup', [32, 8, 1])

    # Derive MI parameters if MatrixInstruction has 9 elements
    mi = config.get('MatrixInstruction', [])
    if len(mi) == 9:
        mi_params = matrixInstructionToMIParameters(
            mi, isa, config['WavefrontSize'],
            config['ProblemType'], config['WorkGroup'], isaInfoMap)
        config.update(mi_params)

    sol = Solution(config, splitGSU=False, printSolutionRejectionReason=True,
                   printIndexAssignmentInfo=False, assembler=asm,
                   isaInfoMap=isaInfoMap)
    assert sol["Valid"], f"Solution is not valid for config: {kernel_config}"
    return sol


def generate_real_idmap(kernel_config, asm, isaInfoMap):
    """Run the full kernel-writer pipeline so `customMainLoopSchedule`
    populates `writer._last_id_map` / `writer._last_mfma_code`, then
    return those alongside the Solution. The kernel-writer side-channel
    is the only way to get rocisa instructions with real (writer-assigned)
    register names, which the swap-pack and pack-MFMA tests need.

    Re-raises with a hint when generation fails — global state from
    earlier tests can interfere with `hasCustomSchedule`, so the
    "run in isolation" suggestion saves debugging time."""
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    solution = _make_solution(kernel_config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        writer._getKernelSource(solution)
    except Exception as e:
        raise RuntimeError(
            f"Kernel generation failed. This can happen if earlier tests modified "
            f"global state (e.g. via hasCustomSchedule). Try running this test in "
            f"isolation: pytest <file>::<class>::<test> -v\n"
            f"Original error: {e}"
        ) from e

    if not hasattr(writer, '_last_id_map'):
        raise RuntimeError(
            "Kernel writer did not produce an idMap. This means "
            "customMainLoopSchedule was not called — either UseCustomMainLoopSchedule "
            "is not enabled for this config, or global state from earlier tests "
            "caused the schedule to not match. Try running in isolation."
        )

    return writer._last_id_map, writer._last_mfma_code, solution
