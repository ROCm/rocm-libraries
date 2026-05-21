#!/usr/bin/env python3
################################################################################
# GPU test helpers for subtile-based kernel unit tests.
#
# Provides:
#   - TileConfig dataclass for parameterized tile configurations
#   - Mock/kernel creation helpers (_mock_dtype, _create_kernel, create_writer)
#   - rocIsa initialization (init_rocisa)
#   - Unified kernel asm generator (generate_kernel_asm)
#   - Prologue builder (generate_load_params) and export epilogue (generate_export_epilogue)
#   - Assembly & GPU execution (assemble_kernel, assemble_and_run)
#   - Debug utilities (print_offset_grid)
#   - Roundtrip kernel helpers (generate_srd_setup, setup_roundtrip_writer, build_roundtrip_inner_asm)
################################################################################

import ctypes
import os
import re
import sys
import struct
import subprocess
import tempfile

# Add tensilelite to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from hip import hip, hiprtc  # type: ignore

from unittest.mock import MagicMock
from types import SimpleNamespace
from dataclasses import dataclass

from rocisa.code import Module, TextBlock
from rocisa.container import vgpr, sgpr
from rocisa.instruction import SLoadB32, SLoadB64, SLoadB128, SMovB32, SMovB64, SWaitCnt, SBarrier, VLShiftLeftB32, VMovB32
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16, AB_B8
from Tensile.Components.Subtile.SubtileGREmit import graTileAssignment, globalReadDTLInitCommonSgpr, globalReadDoSubtile
from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment, localReadDoSubtile

# ---- GPU target detection ----
def _detect_gfx_target():
    """Detect the GPU architecture from the current device.

    Detection order:
    1. TENSILE_GPU_TARGET env var (explicit override)
    2. rocm_agent_enumerator (GPU hardware detection)

    Returns None if no GPU can be detected.
    """
    override = os.environ.get("TENSILE_GPU_TARGET")
    if override:
        return override

    rocmpath = os.environ.get("ROCM_PATH", "/opt/rocm")
    enumerator = os.path.join(rocmpath, "bin", "rocm_agent_enumerator")
    if os.path.exists(enumerator):
        try:
            output = subprocess.check_output(
                [enumerator, "-t", "GPU"], stderr=subprocess.DEVNULL
            )
            archs = [
                line.strip()
                for line in output.decode().splitlines()
                if line.strip() and "gfx000" not in line
            ]
            if archs:
                return archs[0]
        except subprocess.CalledProcessError:
            pass

    return None


# ---- Constants ----
GFX_TARGET = _detect_gfx_target()
HAS_GFX950 = GFX_TARGET == "gfx950"
WAVESIZE   = 64
NUM_WAVES  = 4
NUM_THREADS = WAVESIZE * NUM_WAVES  # 256
BPE        = 2      # fp16
LOAD_WIDTH = 16     # dwordx4


@dataclass
class TileConfig:
    """Parameterized tile configuration for testing."""
    mt_a: int       # MacroTileA
    mt_b: int       # MacroTileB
    depth_u: int    # DepthU
    stride_a: int = 0   # StrideA0I (in elements), only needed by GRA tests
    stride_b: int = 0   # StrideB1J (in elements), only needed by GRA tests
    use_swizzling: bool = False  # Whether to enable swizzling, only needed by GRA tests

    @property
    def label(self):
        swz = "_swz" if self.use_swizzling else ""
        stride = ""
        if self.stride_a and self.stride_a != self.depth_u:
            stride = f"_s{self.stride_a}"
        return f"{self.mt_a}x{self.mt_b}x{self.depth_u}{swz}{stride}"

# ---- HIP helpers ----

def hip_check(result):
    """Check HIP call result."""
    if isinstance(result, tuple):
        err = result[0]
        if err != 0:
            raise RuntimeError(f"HIP error {err}")
        return result[1] if len(result) == 2 else result[1:]
    if result != 0:
        raise RuntimeError(f"HIP error {result}")


# ---- Mock / setup helpers ----

def _mock_dtype(num_bytes=2):
    """Create a mock DataType that returns numBytes()."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_kernel(cfg, mi_wave_group=None, inst_k=32, bpe=2):
    """Create a minimal kernel dict matching the given tile config."""
    dtype = _mock_dtype(bpe)
    if mi_wave_group is not None:
        MIWaveGroup = mi_wave_group
    elif ((cfg.mt_a//16) % 2 == 0) and ((cfg.mt_b//16) % 2 == 0):
        MIWaveGroup = [2,2]
    elif ((cfg.mt_a//16) % 2 != 0) and ((cfg.mt_b//16) % 4 == 0):
        MIWaveGroup = [1,4]
    elif ((cfg.mt_a//16) % 4 == 0) and ((cfg.mt_b//16) % 2 != 0):
        MIWaveGroup = [4,1]
    else:
        raise ValueError(f"Unsupported tile config for wave grouping: mt_a={cfg.mt_a}, mt_b={cfg.mt_b}")

    return {
        "DepthU": cfg.depth_u,
        "_DepthU": cfg.depth_u,
        "_DepthUA": cfg.depth_u,
        "_DepthUB": cfg.depth_u,
        "MacroTileA": cfg.mt_a,
        "MacroTileB": cfg.mt_b,
        "MacroTile0": cfg.mt_a,
        "MacroTile1": cfg.mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": inst_k,
        "MIWaveGroup": MIWaveGroup,
        "WavefrontSize": WAVESIZE,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def compute_lds_start_offset_b(tileInfoA):
    """Compute LDS start offset for B (A subtiles followed by B, aligned to readSize)."""
    readSize = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    return int(((numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize) * readSize)


def create_writer(cfg, mi_wave_group=None, geometry=None, inst_k=32, bpe=2):
    """Create a minimal mock writer with register pools, kernel dict, and TileInfo.

    Sets up the base writer that all tests need:
      - vgprPool/sgprPool (empty)
      - v0 reserved for Serial (hardware workitem_id)
      - kernel dict from cfg
      - TileInfo A/B
      - writer.states with tileInfo refs and register caps

    Args:
        cfg:           TileConfig with macro tile / depthU / stride parameters.
        mi_wave_group: Override MIWaveGroup; auto-detected from cfg if None.
        geometry:      ABTilePair geometry (e.g. AB_B16, AB_B8). Defaults to AB_B16.
        inst_k:        MatrixInstK value (32 for FP16, 128 for FP8).
        bpe:           Bytes per element for A/B data type (2 for FP16, 1 for FP8).

    Each test is responsible for reserving sgprs, defining named sgprs
    (writer.sgprs), and calling allocOffsetRegisters etc.

    Returns:
        (writer, kernel, tileInfoA, tileInfoB)
    """
    if geometry is None:
        geometry = AB_B16

    writer = SimpleNamespace()

    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}

    # Reserve v0 for Serial (hardware workitem_id)
    writer.vgprPool.checkOut(1)

    # Build kernel and TileInfo
    kernel = _create_kernel(cfg, mi_wave_group=mi_wave_group, inst_k=inst_k, bpe=bpe)

    tileInfoA = TileInfo(geometry, 'A', writer, kernel)
    tileInfoB = TileInfo(geometry, 'B', writer, kernel)

    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                    defaultPreventOverflow=False, printRP=False)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tileInfoA),
        b=SimpleNamespace(tileInfo=tileInfoB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
    )
    # LDS layout: A subtiles followed by B subtiles, aligned to readSize
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = compute_lds_start_offset_b(tileInfoA)

    return writer, kernel, tileInfoA, tileInfoB


def _generate_tile_asm(cfg, emitter, geometry=None, inst_k=32, bpe=2):
    """Common setup for GR/LR tile assignment asm generation.

    Args:
        cfg:     TileConfig with tile geometry and stride parameters.
        emitter: Callable (writer, kernel) -> module that generates the tile asm.
        geometry, inst_k, bpe: passed to create_writer.

    Returns:
        (asm, writer, tileInfoA, tileInfoB, kernel)
    """
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg, geometry=geometry, inst_k=inst_k, bpe=bpe)
    init_rocisa()
    # Reserve s0-s11: s[0:1]=kernarg ptr (HW), s[2:3]=workgroup IDs (HW),
    # s[4:5]=output ptr, s[6:9]=padding, s10=StrideA0I, s11=StrideB1J
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)
    # Kernarg layout (same for all data types — strides are always u32, ptr is u64):
    # (dst_sgpr, count_dwords, byte_offset, comment)
    load_params = (
        (4,           2, 0x00, "output_ptr"),  # s[4:5] = 64-bit output buffer pointer
        ("StrideA0I", 1, 0x08, "strideA"),    # s10    = stride for A (elements)
        ("StrideB1J", 1, 0x0c, "strideB"),    # s11    = stride for B (elements)
    )
    prologue = generate_load_params(load_params)
    module = emitter(writer, kernel)
    asm = f"{prologue}\n{module}"
    return asm, writer, tileInfoA, tileInfoB, kernel


def generate_gra_asm(cfg, geometry=None, inst_k=32, bpe=2):
    """Run graTileAssignment and return (gra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    return _generate_tile_asm(
        cfg,
        lambda w, k: graTileAssignment(w, k, useSwizzling=cfg.use_swizzling),
        geometry=geometry, inst_k=inst_k, bpe=bpe,
    )


def generate_lra_asm(cfg, geometry=None, inst_k=32, bpe=2):
    """Run lraTileAssignment and return (lra_asm, writer, tileInfoA, tileInfoB, kernel)."""
    return _generate_tile_asm(cfg, lraTileAssignment, geometry=geometry, inst_k=inst_k, bpe=bpe)


def compute_expected_subtile(regId, stride, tileInfo, bpe):
    """Expected subtile soffset register value: rowOffset * bpe * regId * stride.

    rowOffset is 2*subtileSize when loadRatioGR==2.0 (two subtiles per GR), else subtileSize.
    """
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    rowOffset = 2 * subtileSize if tileInfo.loadRatioGR == 2.0 else subtileSize
    return rowOffset * bpe * regId * stride


def export_register(writer, test_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Generate export kernel, assemble, run, return per-thread u32 results."""
    # Must match load_params layout in generate_gra_asm (same for all data types)
    # (name, size_bytes, value_kind, value_type)
    export_args = (
        ("output_ptr", 8, "global_buffer", "u32"),  # 64-bit pointer to output buffer
        ("strideA",    4, "by_value",      "u32"),  # stride for A in elements
        ("strideB",    4, "by_value",      "u32"),  # stride for B in elements
    )
    epilogue, allocated = generate_export_epilogue(writer, export_reg, is_sgpr)
    kernel_asm = generate_kernel_asm(f"{test_asm}\n{epilogue}", writer, export_args)
    for v in allocated:
        writer.vgprPool.checkIn(v)
    raw = assemble_and_run(kernel_asm, tmp_path, label, NUM_THREADS * 4,
                           scalars=(cfg.stride_a, cfg.stride_b))
    return struct.unpack(f"{NUM_THREADS}I", raw)


def generate_set_directives(sgprs):
    """Generate .set directives mapping symbolic names to sgpr indices.

    Args:
        sgprs: dict mapping name -> index (e.g. writer.sgprs).
    """
    return "\n".join(f".set sgpr{name}, {idx}" for name, idx in sgprs.items())


def init_rocisa(target=None):
    """Initialize rocIsa singleton for the detected GPU target.
    Args:
        target: Optional gfx string (e.g. 'gfx950'). When None (default), we
                use GFX_TARGET as detected from the host via
                rocm_agent_enumerator. Passing an explicit target is required
                only by tests whose assertions are tied to a specific ISA's
                opcode mnemonics (mfma vs wmma).
    Always calls ri.init() because other module imports (e.g. KernelWriter)
    may have already initialized the singleton with a different target.
    """
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    gfx_target = target if target is not None else GFX_TARGET
    if not gfx_target:
        raise ValueError(f"Invalid GPU target: '{gfx_target}'")
    isa = gfxToIsa(gfx_target)
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE)


# ---- Kernel assembly generator ----

def _generate_args_metadata(args):
    """Generate YAML args metadata and kernarg_segment_size from an args list.

    Each arg is a tuple: (name, size, value_kind, value_type).
    For global_buffer args, address_space is set to "global".

    Returns:
        (args_yaml_str, kernarg_size) tuple.
    """
    lines = []
    offset = 0
    for name, size, value_kind, value_type in args:
        lines.append(f"      - .name:            {name}")
        lines.append(f"        .size:            {size}")
        lines.append(f"        .offset:          {offset}")
        lines.append(f"        .value_kind:      {value_kind}")
        lines.append(f"        .value_type:      {value_type}")
        if value_kind == "global_buffer":
            lines.append(f"        .address_space:   global")
        offset += size
    kernarg_size = (offset + 7) & ~7  # align to 8
    return "\n".join(lines), kernarg_size


def _scan_register_indices(*texts):
    """Extract vgpr, agpr, and sgpr index sets from assembly text.

    Handles bare references (s10, v3, a5) and range references (s[4:7], v[0:3], a[0:3]).
    Also extracts indices from .set directives (.set sgprFoo, 12).
    Returns (vgprs, agprs, sgprs).
    """
    vgprs = set()
    agprs = set()
    sgprs = set()
    for text in texts:
        vgprs.update(int(m) for m in re.findall(r'\bv(\d+)\b', text))
        agprs.update(int(m) for m in re.findall(r'\ba(\d+)\b', text))
        sgprs.update(int(m) for m in re.findall(r'\bs(\d+)\b', text))
        for start, end in re.findall(r'\bs\[(\d+):(\d+)\]', text):
            sgprs.update(range(int(start), int(end) + 1))
        for start, end in re.findall(r'\bv\[(\d+):(\d+)\]', text):
            vgprs.update(range(int(start), int(end) + 1))
        for start, end in re.findall(r'\ba\[(\d+):(\d+)\]', text):
            agprs.update(range(int(start), int(end) + 1))
        # .set sgprFoo, N  /  .set vgprBar, N
        sgprs.update(int(m) for m in re.findall(r'\.set\s+sgpr\w+,\s*(\d+)', text))
        vgprs.update(int(m) for m in re.findall(r'\.set\s+vgpr\w+,\s*(\d+)', text))
    return vgprs, agprs, sgprs


def generate_kernel_asm(inner_asm, writer, args, lds_size=0, num_threads=None):
    """Generate a complete AMDHSA kernel wrapping inner_asm.

    Args:
        inner_asm:  Assembly code including prologue, test logic, and epilogue.
        writer:     Object with .sgprs dict (name -> index) for .set directives.
        args:       Kernarg descriptors — sequence of
                    (name, size, value_kind, value_type) tuples.
        lds_size:   LDS allocation size in bytes.

    Returns:
        Assembly source string.
    """
    if num_threads is None:
        num_threads = NUM_THREADS
    lds_size = int(lds_size)
    set_directives = generate_set_directives(writer.sgprs)
    args_metadata, kernarg_size = _generate_args_metadata(args)

    # Auto-compute register counts from inner_asm + set directives
    vgpr_indices, agpr_indices, sgpr_indices = _scan_register_indices(
        inner_asm, set_directives)

    max_vgpr = max(vgpr_indices | {0}) + 1
    max_sgpr = max(sgpr_indices | {0}) + 1
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)  # align to 4 for accum_offset
    # On gfx950, VGPRs and accVGPRs share the physical register file.
    # next_free_vgpr must cover accum_offset (= max_vgpr) + all accvgprs used.
    num_agprs = max(agpr_indices | {-1}) + 1  # 0 if no agprs used
    next_free_vgpr = max_vgpr + num_agprs

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"

// Register name mappings
.set vgprSerial, 0
{set_directives}

.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset {max_vgpr}
  .amdhsa_next_free_vgpr {next_free_vgpr}
  .amdhsa_next_free_sgpr {max_sgpr}
  .amdhsa_group_segment_fixed_size {lds_size}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.text
test_kernel:
{inner_asm}
  s_waitcnt vmcnt(0)
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: test_kernel
    .symbol: 'test_kernel.kd'
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .args:
{args_metadata}
    .kernarg_segment_size: {kernarg_size}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds_size}
    .private_segment_fixed_size: 0
    .wavefront_size: {WAVESIZE}
    .sgpr_count: {max_sgpr}
    .vgpr_count: {next_free_vgpr}
    .max_flat_workgroup_size: {num_threads}
...
.end_amdgpu_metadata
"""


def generate_load_params(loads):
    """Generate prologue module that loads kernel arguments from kernarg segment.

    Args:
        loads: sequence of (dst, count, offset, comment) tuples.
            dst:     sgpr index (int) or symbolic name (str).
            count:   number of dwords to load (1, 2, or 4).
            offset:  byte offset into kernarg segment.
            comment: comment string for the instruction.

    Returns:
        Module with SLoad instructions + SWaitCnt.
    """
    load_cls = {1: SLoadB32, 2: SLoadB64, 4: SLoadB128}
    module = Module("prologue")
    for dst, count, offset, comment in loads:
        cls = load_cls[count]
        dst_sgpr = sgpr(dst, count) if count > 1 else sgpr(dst)
        module.add(cls(dst=dst_sgpr, base=sgpr(0, 2), soffset=offset, comment=comment))
    module.add(SWaitCnt(dscnt=0))
    return module


def generate_export_epilogue(writer, export_reg, is_sgpr=False):
    """Generate epilogue module that exports a single register via global_store.

    Allocates tmp vgprs from writer.vgprPool. Exports export_reg (vgpr or sgpr)
    to the output buffer at s[4:5].

    Args:
        writer:     Writer with vgprPool for register allocation.
        export_reg: Register index to export (e.g. 3 for v3 or s3).
        is_sgpr:    True to export an sgpr, False for a vgpr.

    Returns:
        (module, allocated_vgprs) — Module and list of vgpr indices to checkIn later.
    """
    allocated = []
    tmp_vgpr = writer.vgprPool.checkOut(1, "export_addr", preventOverflow=False)
    allocated.append(tmp_vgpr)

    module = Module("export epilogue")
    module.add(VLShiftLeftB32(dst=vgpr(tmp_vgpr), shiftHex=2, src=vgpr(0),
                              comment="byte offset = tid * 4"))

    if is_sgpr:
        data_vgpr = writer.vgprPool.checkOut(1, "export_data", preventOverflow=False)
        allocated.append(data_vgpr)
        module.add(VMovB32(dst=vgpr(data_vgpr), src=sgpr(export_reg),
                           comment=f"copy s{export_reg} to v{data_vgpr}"))
    else:
        data_vgpr = export_reg

    module.add(TextBlock(f"  global_store_dword v{tmp_vgpr}, v{data_vgpr}, s[4:5]\n"))
    return module, allocated


# ---- Assemble / run ----

def assemble_kernel(asm_source, output_path):
    """Assemble .s source to .co code object."""
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name

    obj_path = asm_path.replace(".s", ".o")

    try:
        subprocess.check_call([
            "amdclang++", "-x", "assembler",
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={GFX_TARGET}",
            "-mwavefrontsize64",
            "-mcode-object-version=5",
            "-o", obj_path,
            asm_path
        ])
        os.rename(obj_path, output_path)
    finally:
        if os.path.exists(asm_path):
            os.unlink(asm_path)
        if os.path.exists(obj_path) and obj_path != output_path:
            os.unlink(obj_path)


def run_on_gpu(co_path, output_size, inputs=(), scalars=(), lds_size=0, num_threads=None):
    """Load code object, launch kernel, read output buffer.

    Builds the kernarg struct dynamically: one u64 per input buffer pointer,
    one u64 for the output pointer, then one u32 per scalar.

    Args:
        co_path:     Path to assembled .co file.
        output_size: Output buffer size in bytes.
        inputs:      Sequence of numpy arrays (or bytes-like) to upload as
                     input buffers. Each becomes a u64 pointer in kernargs.
        scalars:     Sequence of u32 scalar values appended after pointers.
        lds_size:    Shared memory (LDS) size in bytes.

    Returns:
        bytes: Raw output buffer contents.
    """
    if num_threads is None:
        num_threads = NUM_THREADS
    hip_check(hip.hipInit(0))

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    # Upload input buffers
    d_inputs = []
    for inp in inputs:
        data = inp.tobytes() if hasattr(inp, 'tobytes') else bytes(inp)
        d_buf = hip_check(hip.hipMalloc(len(data)))
        hip_check(hip.hipMemcpyHtoD(d_buf, data, len(data)))
        d_inputs.append(d_buf)

    # Allocate output buffer
    d_output = hip_check(hip.hipMalloc(output_size))
    hip_check(hip.hipMemset(d_output, 0xff, output_size))

    # Build kernarg struct: [input_ptrs...] + [output_ptr] + [scalars...]
    fields = []
    for i in range(len(d_inputs)):
        fields.append((f"input_{i}", ctypes.c_uint64))
    fields.append(("output_ptr", ctypes.c_uint64))
    for i in range(len(scalars)):
        fields.append((f"scalar_{i}", ctypes.c_uint32))

    KernelArgs = type("KernelArgs", (ctypes.Structure,), {"_fields_": fields})
    values = [int(d) for d in d_inputs] + [int(d_output)] + list(scalars)
    kargs = KernelArgs(*values)
    kargs_size = ctypes.c_size_t(ctypes.sizeof(kargs))

    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(0x01),  # HIP_LAUNCH_PARAM_BUFFER_POINTER
        ctypes.c_void_p(ctypes.addressof(kargs)),
        ctypes.c_void_p(0x02),  # HIP_LAUNCH_PARAM_BUFFER_SIZE
        ctypes.c_void_p(ctypes.addressof(kargs_size)),
        ctypes.c_void_p(0x03),  # HIP_LAUNCH_PARAM_END
    )

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        1, 1, 1,
        num_threads, 1, 1,
        lds_size,
        None,
        None,
        extra
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_output = bytearray(output_size)
    hip_check(hip.hipMemcpyDtoH(h_output, d_output, output_size))

    for d_buf in d_inputs:
        hip_check(hip.hipFree(d_buf))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(module))

    return bytes(h_output)


def assemble_and_run(asm, tmp_path, label, output_size, inputs=(), scalars=(), lds_size=0, num_threads=None):
    """Write asm to file, assemble to code object, run on GPU, return raw bytes."""
    co_path = str(tmp_path / f"test_{label}.co")
    asm_path = str(tmp_path / f"test_{label}.s")
    with open(asm_path, "w") as f:
        f.write(asm)
    assemble_kernel(asm, co_path)
    return run_on_gpu(co_path, output_size, inputs=inputs, scalars=scalars, lds_size=lds_size, num_threads=num_threads)



# ---- Roundtrip kernel helpers ----

def generate_srd_setup():
    """Generate SRD buffer descriptor setup for A and B.

    Kernarg layout (set by build_roundtrip_inner_asm prologue):
      s[4:5] = input_A_ptr, s[6:7] = input_B_ptr, s[8:9] = output_ptr
    SRD descriptor fields (4 dwords):
      [0:1] = base address (64-bit), [2] = NumRecords, [3] = descriptor flags
    """
    module = Module("SRD setup")
    module.add(SMovB64(dst=sgpr("SrdA+0", 2), src=sgpr(4, 2), comment="SrdA base = s[4:5] = input_A_ptr"))
    module.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF",   comment="SrdA NumRecords = max (no bounds check)"))
    module.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000",      comment="SrdA OOB_SELECT=2 (raw buffer)"))
    module.add(SMovB64(dst=sgpr("SrdB+0", 2), src=sgpr(6, 2), comment="SrdB base = s[6:7] = input_B_ptr"))
    module.add(SMovB32(dst=sgpr("SrdB+2"), src="0xFFFFFFFF",   comment="SrdB NumRecords = max (no bounds check)"))
    module.add(SMovB32(dst=sgpr("SrdB+3"), src="0x20000",      comment="SrdB OOB_SELECT=2 (raw buffer)"))
    return module


def setup_roundtrip_writer(cfg, geometry=None, inst_k=32, bpe=2):
    """Create and configure a writer for roundtrip kernel tests.

    Calls create_writer, reserves sgprs for HW regs + strides + SRD + DTL + swap,
    computes LDS size, and allocates vgprTile registers.

    Returns:
        (writer, kernel, tileInfoA, tileInfoB, lds_size)
    """
    init_rocisa()
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg, geometry=geometry, inst_k=inst_k, bpe=bpe)

    # Reserve s0-s11: s[0:1]=kernarg ptr (HW), s[2:3]=workgroup IDs (HW),
    # s[4:5]=input_A_ptr, s[6:7]=input_B_ptr, s[8:9]=output_ptr,
    # s10=StrideA0I, s11=StrideB1J
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    # SRD descriptors: 4 dwords each, aligned to 4-dword boundary
    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(4, 4, "SrdA", preventOverflow=False)
    writer.sgprs["SrdB"] = writer.sgprPool.checkOutAligned(4, 4, "SrdB", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrA"]  = writer.sgprPool.checkOut(1, "LocalWriteBaseAddrA",  preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetA"] = writer.sgprPool.checkOut(1, "LocalWriteDTLOffsetA", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrB"]  = writer.sgprPool.checkOut(1, "LocalWriteBaseAddrB",  preventOverflow=False)
    writer.sgprs["LocalWriteDTLOffsetB"] = writer.sgprPool.checkOut(1, "LocalWriteDTLOffsetB", preventOverflow=False)
    writer.sgprs["SwapA"] = writer.sgprPool.checkOut(1, "SwapA", preventOverflow=False)
    writer.sgprs["SwapB"] = writer.sgprPool.checkOut(1, "SwapB", preventOverflow=False)

    # LDS layout: align A and B to readSize = 2*subtileSize (DTL reads 2 subtiles at once).
    # Matches production formula in KernelWriter.py.
    readSize     = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    numBSubtiles = tileInfoB.globalSubtileGrid[0] * tileInfoB.globalSubtileGrid[1]
    sizeA = ((numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize) * readSize
    sizeB = ((numBSubtiles * tileInfoB.subtileSize + readSize - 1) // readSize) * readSize
    lds_size = int(sizeA + sizeB)
    writer.ldsTotalSize = lds_size

    tileInfoA.allocVgprTileRegisters_legacy(writer, kernel)
    tileInfoB.allocVgprTileRegisters_legacy(writer, kernel)

    return writer, kernel, tileInfoA, tileInfoB, lds_size


def collect_tile_vgprs(tileInfoA, tileInfoB):
    """Return the set of all vgpr indices used by tile and offset registers."""
    used = set()
    for tileInfo in (tileInfoA, tileInfoB):
        for t in tileInfo.vgprTiles:
            used.update(t.regList.indices)
        used.update(tileInfo.sharedVgprGROffset)
        used.update(tileInfo.sharedVgprLROffset)
    return used


def alloc_export_vgprs(tileInfoA, tileInfoB):
    """Find free vgpr temporaries for export assembly (tmp, addr_lo, addr_hi).

    Scans all vgprTile registers plus GR/LR offset registers to find the
    next free vgpr, then allocates tmp (any), addr_lo (even-aligned), addr_hi.

    Returns:
        (tmp, addr_lo, addr_hi, next_v)
    """
    all_used = collect_tile_vgprs(tileInfoA, tileInfoB)

    next_v = max(all_used | {0}) + 1
    tmp = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1
    return tmp, addr_lo, addr_hi, next_v


def build_roundtrip_inner_asm(writer, kernel, export_asm):
    """Generate the inner assembly for a GR->LDS->LR roundtrip kernel.

    Emits: prologue, SRD setup, GRA, LRA, DTL init, global reads (A+B),
    wait, barrier, local reads (A+B), wait, then the provided export_asm.

    Args:
        writer:     Configured writer (from setup_roundtrip_writer).
        kernel:     Kernel dict.
        export_asm: Assembly string for exporting tile registers.

    Returns:
        inner_asm string.
    """
    gra_module  = graTileAssignment(writer, kernel, useSwizzling=True)
    lra_module  = lraTileAssignment(writer, kernel)
    dtl_module  = globalReadDTLInitCommonSgpr(writer, kernel)
    gr_a_module = globalReadDoSubtile('A', writer, kernel)
    gr_b_module = globalReadDoSubtile('B', writer, kernel)
    wait_gr     = SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1)   # wait for global reads
    barrier     = SBarrier()
    lr_a_module = localReadDoSubtile('A', writer, kernel)
    lr_b_module = localReadDoSubtile('B', writer, kernel)
    wait_lr     = SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1)   # wait for local reads

    # Kernarg layout (two s_load_dwordx4 covering all 5 args):
    #   offset 0x00: s[4:5]=input_A_ptr (8B), s[6:7]=input_B_ptr (8B)
    #   offset 0x10: s[8:9]=output_ptr (8B), s10=strideA (4B), s11=strideB (4B)
    prologue   = generate_load_params([
        (4, 4, 0x00, "input_A_ptr + input_B_ptr"),
        (8, 4, 0x10, "output_ptr + strideA + strideB"),
    ])
    srd_module = generate_srd_setup()

    return "\n".join([
        str(prologue),
        str(srd_module),
        str(gra_module),
        str(lra_module),
        str(dtl_module),
        str(gr_a_module),
        str(gr_b_module),
        str(wait_gr),
        str(barrier),
        str(lr_a_module),
        str(lr_b_module),
        str(wait_lr),
        str(export_asm),
    ])


# ---- Utilities ----

def print_offset_grid(label, results, wavesize, num_waves):
    """Print offsets as a 2D grid: rows = waves, columns = lanes."""
    print(f"\n--- {label} offsets (rows=waves, cols=lanes) ---")
    print(f"{'wave':>6}", end="")
    for lane in range(wavesize):
        print(f" {lane:>6}", end="")
    print()
    print("-" * (7 + 7 * wavesize))
    for w in range(num_waves):
        print(f"{w:>6}", end="")
        for lane in range(wavesize):
            tid = w * wavesize + lane
            print(f" {results[tid]:>6}", end="")
        print()
