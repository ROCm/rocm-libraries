# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Integration tests: Tensile CUOccupancy vs HIP hardware ground truth.

Assembles minimal kernels, loads them via HIP, and asserts that
getOccupancy() / getVgprOccupancy() match
hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
Requires hip-python and a GPU (skip otherwise).
"""

import os
import shutil
import subprocess
import tempfile
from math import ceil
from types import SimpleNamespace

import pytest

# The coverage tox env runs `pytest -m unit Tensile/Tests/unit`; without this
# explicit declaration the file is silently deselected and reports 0% coverage.
# GPU-requiring tests within this file are individually skip-gated via
# @pytest.mark.skipif, so the unit marker is safe here.
pytestmark = pytest.mark.unit

# ── HIP availability and helpers (test-layer only) ─────────────────────────────
from occupancy_hip_testutil import HIP_AVAILABLE, _hip, _hip_check

# ── rocisa / Tensile imports ────────────────────────────────────────────────────
from rocisa import rocIsa
from Tensile.Common.Architectures import gfxToIsa
from Tensile.KernelWriterAssembly import KernelWriterAssembly


# ── GPU detection ───────────────────────────────────────────────────────────────

def _detect_gpu() -> str | None:
    """Return the first detected GPU arch string, or None."""
    override = os.environ.get("TENSILE_GPU_TARGET")
    if override:
        return override
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    enumerator = os.path.join(rocm_path, "bin", "rocm_agent_enumerator")
    if os.path.exists(enumerator):
        try:
            out = subprocess.check_output(
                [enumerator, "-t", "GPU"], stderr=subprocess.DEVNULL
            )
            archs = [
                line.strip()
                for line in out.decode().splitlines()
                if line.strip() and "gfx000" not in line
            ]
            return archs[0] if archs else None
        except subprocess.CalledProcessError:
            pass
    return None


GFX_TARGET: str | None = _detect_gpu()

# ── rocisa helpers ──────────────────────────────────────────────────────────────

def _init_rocisa(gfx: str):
    """Initialise the rocisa singleton for *gfx* and return the instance."""
    ri = rocIsa.getInstance()
    isa = gfxToIsa(gfx)
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    asmpath = shutil.which("amdclang++", path=search_path) or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)
    return ri


def _make_writer(ri) -> KernelWriterAssembly:
    """Construct a minimal KernelWriterAssembly with arch/reg caps wired in."""
    kw = object.__new__(KernelWriterAssembly)
    kw.states = SimpleNamespace(
        archCaps=ri.getArchCaps(),
        regCaps=ri.getRegCaps(),
    )
    return kw


# ── Assembly / compilation helpers ─────────────────────────────────────────────

def _build_minimal_kernel_asm(
    gfx: str,
    num_vgprs: int,
    num_sgprs: int,
    lds_bytes: int,
    num_threads: int,
    wave_size: int = 64,
) -> str:
    """
    Build a complete AMDHSA kernel source with the given resource allocations.

    The kernel body is just *s_endpgm* – we only care that the kernel descriptor
    reports the exact register/LDS numbers so that the HIP occupancy API can
    compute an accurate answer.

    VGPR alignment (4 per gfx9) is applied exactly as the hardware requires.
    On gfx950 with unified registers the same physical pool (PhysicalMaxVgpr=512)
    is used for both regular VGPRs and accVGPRs; we allocate 0 accVGPRs here.
    """
    # Hardware granule: VGPRs are allocated in multiples of 4 per lane (gfx9)
    vgpr_granule = 4
    aligned_vgprs = ((num_vgprs + vgpr_granule - 1) // vgpr_granule) * vgpr_granule
    # accum_offset = first acc VGPR slot = same as next_free regular VGPR
    # (0 acc VGPRs used in these minimal kernels)
    accum_offset = aligned_vgprs
    next_free_vgpr = aligned_vgprs  # no acc VGPRs

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{gfx}"

.text
.protected minimal_kernel
.globl    minimal_kernel
.p2align  8
.type     minimal_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel minimal_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset              {accum_offset}
  .amdhsa_next_free_vgpr            {next_free_vgpr}
  .amdhsa_next_free_sgpr            {num_sgprs}
  .amdhsa_group_segment_fixed_size  {lds_bytes}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id    0
  .amdhsa_float_denorm_mode_32      3
  .amdhsa_float_denorm_mode_16_64   3
.end_amdhsa_kernel

.text
minimal_kernel:
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name:                    minimal_kernel
    .symbol:                  'minimal_kernel.kd'
    .language:                OpenCL C
    .language_version:
      - 2
      - 0
    .args: []
    .kernarg_segment_size:    8
    .kernarg_segment_align:   8
    .group_segment_fixed_size: {lds_bytes}
    .private_segment_fixed_size: 0
    .wavefront_size:          {wave_size}
    .sgpr_count:              {num_sgprs}
    .vgpr_count:              {next_free_vgpr}
    .max_flat_workgroup_size: {num_threads}
...
.end_amdgpu_metadata
"""


def _assemble_to_co(gfx: str, asm_source: str, output_path: str) -> None:
    """Assemble *asm_source* to a code object at *output_path*."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    amdclang = shutil.which("amdclang++", path=search_path) or "/usr/bin/amdclang++"

    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name

    try:
        result = subprocess.run(
            [
                amdclang, "-x", "assembler",
                "--target=amdgcn-amd-amdhsa",
                f"-mcpu={gfx}",
                "-mwavefrontsize64",
                "-mcode-object-version=5",
                "-o", output_path,
                asm_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Assembly failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
    finally:
        if os.path.exists(asm_path):
            os.unlink(asm_path)


# ── HIP query for this test's minimal kernels ────────────────────────────────────

def _query_hip_occupancy(co_path: str, num_threads: int) -> int:
    """
    Load *co_path* and return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
    for *minimal_kernel* with the given thread count.

    dynSharedMemPerBlk is always 0: all LDS in our minimal kernels is static and
    already encoded in the kernel descriptor's group_segment_fixed_size field.
    The HIP runtime reads that value directly from the code object; passing non-
    zero dynamic LDS here would double-count the static allocation.

    Returns the number of max active workgroups (blocks) per CU.
    """
    _hip_check(_hip.hipInit(0))
    module = _hip_check(_hip.hipModuleLoad(
        co_path.encode() if isinstance(co_path, str) else co_path
    ))
    func = _hip_check(_hip.hipModuleGetFunction(module, b"minimal_kernel"))

    err, blocks_per_cu = _hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        func, num_threads, 0   # dynSharedMemPerBlk=0; static LDS is in the descriptor
    )
    _hip_check(err)
    _hip_check(_hip.hipModuleUnload(module))
    return int(blocks_per_cu)


# ── Conversion formula ──────────────────────────────────────────────────────────

def tensile_occ_to_hip_blocks(cu_occupancy: int, num_threads: int,
                               wave_size: int = 64, num_simds: int = 4) -> int:
    """Return expected HIP blocks-per-CU given Tensile's CUOccupancy.

    For gfx9 (4 SIMDs × 64-wide wavefront), Tensile's CUOccupancy equals
    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor (1:1 relationship).
    """
    _ = (num_threads, wave_size, num_simds)
    return cu_occupancy


# ── Test parametrization ────────────────────────────────────────────────────────
#
# Each entry: (required_gfx, num_vgprs, lds_bytes, num_threads, label)
# VGPRs are multiples of 8 so Tensile's 8-dword alignment and HIP's 4-dword granule agree.
# _MIN_LDS=256 avoids getLdsLimitedOccupancy divide-by-zero (represents no LDS constraint).

_MIN_LDS = 256   # minimum LDS allocation granule; getLdsLimitedOccupancy crashes on 0

OCCUPANCY_CASES = [
    # ── gfx950 regression case ─────────────────────────────────────────────────
    pytest.param(
        "gfx950", 8, _MIN_LDS, 256,
        "gfx950 low-VGPR 256t → occupancy was over-reported as 10, now fixed to 8",
        id="gfx950-low-vgpr-256t",
    ),
    pytest.param(
        "gfx950", 8, _MIN_LDS, 512,
        "gfx950 low-VGPR 512t → maxOcc per SIMD = 4",
        id="gfx950-low-vgpr-512t",
    ),
    # ── gfx950 VGPR-limited cases ──────────────────────────────────────────────
    pytest.param(
        "gfx950", 128, _MIN_LDS, 256,
        "gfx950 128-VGPR 256t → VGPR-limited, occupancy=4",
        id="gfx950-128-vgpr-256t",
    ),
    pytest.param(
        "gfx950", 256, _MIN_LDS, 256,
        "gfx950 256-VGPR 256t → VGPR-limited, occupancy=2",
        id="gfx950-256-vgpr-256t",
    ),
    # ── gfx950 LDS-limited case ────────────────────────────────────────────────
    # LDS per CU = 163840 B, 163840//65536 = 2 blocks
    # The kernel descriptor holds group_segment_fixed_size=65536; dynSharedMemPerBlk=0.
    pytest.param(
        "gfx950", 8, 65536, 256,
        "gfx950 low-VGPR large-LDS 256t → LDS-limited, occupancy=2",
        id="gfx950-low-vgpr-lds-limited-256t",
    ),
    # ── gfx908 baseline (MaxWavesPerSimd=10, was already correct) ─────────────
    pytest.param(
        "gfx908", 8, _MIN_LDS, 256,
        "gfx908 low-VGPR 256t → occupancy=10 (was always correct)",
        id="gfx908-low-vgpr-256t",
    ),
    # ── gfx942 (ArchAccUnifiedRegs, MaxWavesPerSimd=8, same family as gfx950) ─
    pytest.param(
        "gfx942", 8, _MIN_LDS, 256,
        "gfx942 low-VGPR 256t → occupancy=8",
        id="gfx942-low-vgpr-256t",
    ),
]


# ── Main test ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HIP_AVAILABLE, reason="hip-python not installed")
@pytest.mark.skipif(GFX_TARGET is None, reason="No GPU detected")
@pytest.mark.parametrize(
    "required_gfx,num_vgprs,lds_bytes,num_threads,description",
    OCCUPANCY_CASES,
)
def test_hip_occupancy_matches_tensile(
    required_gfx, num_vgprs, lds_bytes, num_threads, description, tmp_path
):
    """HIP ground-truth: assemble a minimal kernel, query HIP, assert equality with Tensile."""
    if GFX_TARGET != required_gfx:
        pytest.skip(
            f"Test requires {required_gfx}, detected {GFX_TARGET}"
        )

    # ── 1. Tensile prediction ─────────────────────────────────────────────────
    ri = _init_rocisa(required_gfx)
    kw = _make_writer(ri)
    arch_caps = ri.getArchCaps()

    # doubleVgpr=True on ArchAccUnifiedRegs architectures (gfx90a / gfx942 / gfx950)
    double_vgpr = bool(arch_caps.get("ArchAccUnifiedRegs", 0))
    num_sgprs = 16   # minimal; well within the SGPR limit
    acc_vgprs = 0    # no accumulator VGPRs in these minimal kernels

    tensile_occ = kw.getOccupancy(
        numThreads=num_threads,
        vgprs=num_vgprs,
        sgprs=num_sgprs,
        ldsSize=lds_bytes,
        accvgprs=acc_vgprs,
        doubleVgpr=double_vgpr,
    )

    # ── 2. Build and assemble minimal kernel ──────────────────────────────────
    wave_size = 64
    asm_src = _build_minimal_kernel_asm(
        required_gfx, num_vgprs, num_sgprs, lds_bytes, num_threads, wave_size
    )
    co_path = str(tmp_path / f"occ_{required_gfx}_{num_vgprs}v_{lds_bytes}lds_{num_threads}t.co")
    _assemble_to_co(required_gfx, asm_src, co_path)
    assert os.path.exists(co_path), "Assembler produced no output"

    # ── 3. HIP hardware occupancy ─────────────────────────────────────────────
    hip_blocks_per_cu = _query_hip_occupancy(co_path, num_threads)

    # ── 4. Convert and compare ────────────────────────────────────────────────
    # Tensile CUOccupancy == hipModuleOccupancyMaxActiveBlocksPerMultiprocessor (1:1)
    expected_blocks_per_cu = tensile_occ_to_hip_blocks(
        tensile_occ, num_threads, wave_size=wave_size
    )

    print(f"\n  [{description}]")
    print(f"  arch:                    {required_gfx}")
    print(f"  ArchAccUnifiedRegs:      {double_vgpr}")
    print(f"  MaxWavesPerSimd:         {arch_caps.get('MaxWavesPerSimd', '?')}")
    print(f"  num_vgprs:               {num_vgprs}")
    print(f"  lds_bytes:               {lds_bytes}")
    print(f"  num_threads:             {num_threads}")
    print(f"  Tensile CUOccupancy:     {tensile_occ} blocks/CU")
    print(f"  HIP blocks/CU:           {hip_blocks_per_cu}")
    print(f"  Expected blocks/CU:      {expected_blocks_per_cu}")

    assert hip_blocks_per_cu == expected_blocks_per_cu, (
        f"Tensile CUOccupancy={tensile_occ} blocks/CU, "
        f"but HIP reports {hip_blocks_per_cu} blocks/CU.\n"
        f"  arch={required_gfx}, MaxWavesPerSimd={arch_caps.get('MaxWavesPerSimd')}, "
        f"num_vgprs={num_vgprs}, lds_bytes={lds_bytes}, num_threads={num_threads}.\n"
        f"  If Tensile over-reports, check that fix/gfx950-tensile-occupancy is applied "
        f"and rocisa is rebuilt."
    )


# ── Regression: gfx950 old behavior would have returned 10 ─────────────────────

@pytest.mark.skipif(not HIP_AVAILABLE, reason="hip-python not installed")
@pytest.mark.skipif(GFX_TARGET != "gfx950", reason="Requires gfx950 GPU")
def test_gfx950_occupancy_not_overcounted(tmp_path):
    """
    Explicit regression test: Tensile must NOT return CUOccupancy > 8 for gfx950.

    Before fix/gfx950-tensile-occupancy, getVgprOccupancy() used a hardcoded
    maxOccupancy=10 (the gfx908 value), causing CUOccupancy=10 for any kernel
    with low VGPR usage on gfx950.  The hardware cap is MaxWavesPerSimd=8.
    This test fails if the fix is absent or reverted.
    """
    ri = _init_rocisa("gfx950")
    kw = _make_writer(ri)
    arch_caps = ri.getArchCaps()

    assert arch_caps["MaxWavesPerSimd"] == 8, (
        f"Expected MaxWavesPerSimd=8 for gfx950, got {arch_caps['MaxWavesPerSimd']}. "
        "Check that hardware_caps.hpp has the ArchAccUnifiedRegs branch."
    )

    # Low-VGPR case: VGPR count is nowhere near limiting (pure wave-cap limited)
    cu_occ = kw.getVgprOccupancy(numThreads=256, vgprs=1, doubleVgpr=False)
    assert cu_occ <= 8, (
        f"gfx950 getVgprOccupancy returned {cu_occ} > 8.  "
        "MaxWavesPerSimd must be 8 for ArchAccUnifiedRegs architectures; "
        "the old hardcoded value of 10 is incorrect for gfx950."
    )

    # Verify the HIP ground truth agrees
    asm = _build_minimal_kernel_asm("gfx950", 8, 16, _MIN_LDS, 256)
    co_path = str(tmp_path / "regression_gfx950.co")
    _assemble_to_co("gfx950", asm, co_path)
    hip_blocks = _query_hip_occupancy(co_path, 256)

    # For 256 threads, gfx9: blocks/CU == waves/SIMD (1:1 factor)
    assert hip_blocks <= 8, (
        f"Hardware reports {hip_blocks} blocks/CU > 8 for gfx950; "
        "unexpected – MaxWavesPerSimd=8 should cap this."
    )
    assert cu_occ == hip_blocks, (
        f"Tensile CUOccupancy={cu_occ} != HIP blocks/CU={hip_blocks}. "
        "Tensile is over- or under-reporting occupancy for gfx950."
    )
