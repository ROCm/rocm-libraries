#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# On-simulator (FFM gfx1250) GPU test for the StreamK workgroup-cluster
# reduction fast-path handshake.
#
# Mirrors the structure of test_gr_lr_roundtrip.py (assemble a small gfx1250
# kernel, run it on the device, compare to a numpy reference), but exercises the
# *reduction synchronization sequence* the StreamKClusterReduction fast path
# emits in Tensile/Components/StreamK.py, run on the FFM mi450 simulator:
#
#     peer :  write partial -> global_wb scope:SCOPE_DEV (release)
#                            -> wave-0-elected s_barrier_signal -3 (arrive)
#     owner:  s_barrier_wait -3 (all cluster members arrived)
#                            -> global_inv scope:SCOPE_DEV (acquire)
#                            -> accumulate peers' partials (v_add_f32, as
#                               fixupStep does)
#
# The kernel is assembled with wavefront_size=None (gfx1250 is wave32 and its
# assembler rejects -mwavefrontsize32) and validated for numerical correctness
# of the accumulated reduction, for C in {2, 4}. A SIGALRM watchdog turns a
# barrier deadlock into a test failure instead of an indefinite hang.
#
# Two launch shapes are covered:
#   * single workgroup (a degenerate cluster of one WG) -- proves the exact
#     store/release/signal/wait/acquire/accumulate sequence runs on FFM,
#     computes the right reduction, and does not hang; and
#   * C concurrent workgroups, each an independent cluster-of-one over its own
#     partial slice -- proves the -3 signal/wait handshake replicated across
#     many workgroups completes without deadlock and every workgroup's local
#     reduction is correct.
#
# NOTE (scope / binding limitation): the *co-resident* multi-WG cluster launch
# the production host uses (hipDrvLaunchKernelEx + hipLaunchAttributeCluster-
# Dimension, see src/hip/HipSolutionAdapter.cpp) cannot be issued through the
# installed hip-python binding -- it exposes neither the cluster launch
# attribute id nor a clusterDim field on hipLaunchAttributeValue. So the true
# owner-waits-for-C-1-peers co-residency is not exercised here; that requires
# the C++ tensilelite-client cluster launch path. What is proven on-sim is that
# the emitted instruction/fence sequence assembles and executes on gfx1250
# without hanging and with correct reduction arithmetic.
#
# Usage:
#   source /opt/ffm/mi450/setenv.sh
#   TENSILE_GPU_TARGET=gfx1250 pytest test_streamk_cluster_reduction_gpu.py -v
################################################################################

import os
import signal
import struct
import sys

import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gpu_test_helpers import (  # noqa: E402
    GFX_TARGET,
    assemble_kernel,
    run_on_gpu,
    requires_gpu_gfx1250,
)

pytestmark = pytest.mark.unit

_RUN_TIMEOUT_S = 60  # per-run deadlock watchdog

WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    """Initialize the rocIsa singleton for gfx1250 (wave32) so the emitters
    render the gfx1250 encodings (s_barrier_signal/-wait -3, v_add_nc_u32,
    s_wait_loadcnt/storecnt, global_wb/global_inv scope:SCOPE_DEV)."""
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


class _Timeout:
    """SIGALRM watchdog: convert a barrier deadlock into a failure, not a hang."""

    def __init__(self, seconds, what):
        self.seconds = seconds
        self.what = what

    def __enter__(self):
        def _raise(signum, frame):
            raise TimeoutError(f"{self.what} did not complete within {self.seconds}s "
                               "(possible cluster-barrier deadlock)")
        self._old = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


def _emit_reduction_module(C):
    """Emit the fast-path reduction body (rocisa) for a cluster of size C.

    Returns the assembly text for the kernel body. Uses the same rocisa
    emitters and instruction shapes the StreamKClusterReduction fast path uses
    (SBarrier cluster signal/wait, global_wb/global_inv SCOPE_DEV fences,
    v_add_f32 accumulation), so the on-sim run exercises real gfx1250 encodings.

    Layout over the ``ws`` buffer (base 0, independent of workgroup id): lane
    ``l`` (0 <= l < C) publishes partial ``l + 1`` into slot ``l``; after the
    cluster handshake every lane accumulates all C slots and lane 0 stores the
    sum into ``out[0]``. The published partials are idempotent (every workgroup
    writes the identical value into each slot), so launching this kernel over a
    grid of C workgroups is a benign race that still exercises the cross-WG
    ``s_barrier_signal/-wait -3`` handshake and yields the same verifiable sum
    regardless of which workgroup writes last -- deliberately avoiding a
    dependence on the workgroup-id SGPR, whose value is unreliable on the FFM
    hand-rolled-kernel path.
    """
    from rocisa.code import Module
    from rocisa.container import vgpr, sgpr
    from rocisa.enum import CacheScope
    from rocisa.instruction import (
        SLoadB64, SWaitCnt, SCmpEQU32, SCBranchSCC0,
        VLShiftLeftB32, VAddU32, VMovB32, VCvtU32toF32, VReadfirstlaneB32,
        GlobalStoreB32, GlobalLoadB32, GlobalWb, GlobalInv, VAddF32,
        SBarrier,
    )
    from rocisa.code import Label

    _init_rocisa_gfx1250()

    m = Module("streamk cluster reduction on-sim body")
    # --- kernarg load: s[4:5]=ws_ptr, s[6:7]=out_ptr ---
    m.add(SLoadB64(dst=sgpr(4, 2), base=sgpr(0, 2), soffset=0x0, comment="ws_ptr"))
    m.add(SLoadB64(dst=sgpr(6, 2), base=sgpr(0, 2), soffset=0x8, comment="out_ptr"))
    m.add(SWaitCnt(kmcnt=0, comment="wait kernargs"))

    # --- peer publish: ws[tid] = float(tid + 1) ---
    m.add(VLShiftLeftB32(dst=vgpr(1), shiftHex=2, src=vgpr(0), comment="tid*4 -> voffset"))
    m.add(VAddU32(dst=vgpr(2), src0=1, src1=vgpr(0), comment="tid + 1 (u32)"))
    m.add(VCvtU32toF32(dst=vgpr(2), src=vgpr(2), comment="partial = float(tid + 1)"))
    m.add(GlobalStoreB32(vgpr(1), vgpr(2), sgpr(4, 2), comment="publish partial to ws"))
    m.add(SWaitCnt(vscnt=0, comment="drain the partial store"))
    # release: publish the partial before signalling (SCOPE_DEV, as the fast path keeps)
    m.add(GlobalWb(CacheScope.SCOPE_DEV, comment="releaseFence: partial visible"))

    # --- wave-0-elected cluster arrive (s_barrier_signal -3) ---
    skip = Label(label="skip_cluster_signal", comment="")
    m.add(VReadfirstlaneB32(dst=sgpr(10), src=vgpr(0), comment="wave 0 signals the cluster"))
    m.add(SCmpEQU32(src0=sgpr(10), src1=0, comment="check for wave 0"))
    m.add(SCBranchSCC0(labelName=skip.getLabelName(), comment="only wave 0 signals"))
    m.add(SBarrier(True, False, True, comment="cluster_barrier signal (arrive)"))
    m.add(skip)

    # --- owner cluster wait (s_barrier_wait -3) + acquire ---
    m.add(SBarrier(True, True, True, comment="cluster_barrier wait (all peers arrived)"))
    m.add(GlobalInv(CacheScope.SCOPE_DEV, comment="acquireFence: observe peers' partials"))

    # --- accumulate all C partials (fixupStep-style v_add_f32) ---
    m.add(VMovB32(dst=vgpr(9), src=0, comment="accumulator = 0"))
    for i in range(C):
        m.add(VMovB32(dst=vgpr(3), src=i * 4, comment=f"slot {i} byte offset"))
        m.add(GlobalLoadB32(vgpr(4), vgpr(3), sgpr(4, 2), comment=f"load peer partial {i}"))
        m.add(SWaitCnt(vlcnt=0, comment="wait partial load"))
        m.add(VAddF32(dst=vgpr(9), src0=vgpr(9), src1=vgpr(4), comment="accumulate"))

    # --- store the reduction result to out[0] (every lane writes the same sum) ---
    m.add(VMovB32(dst=vgpr(5), src=0, comment="out[0] byte offset"))
    m.add(GlobalStoreB32(vgpr(5), vgpr(9), sgpr(6, 2), comment="out[0] = sum"))
    m.add(SWaitCnt(vscnt=0, comment="drain result store"))
    return str(m)


def _build_kernel(C):
    """Wrap the reduction body in a complete gfx1250 (wave32) AMDHSA kernel."""
    body = _emit_reduction_module(C)
    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"
.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 16
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
test_kernel:
{body}
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name: test_kernel
    .symbol: 'test_kernel.kd'
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            ws
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   global
      - .name:            out
        .size:            8
        .offset:          8
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   global
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 16
    .vgpr_count: 16
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
"""


def _run_reduction(C, num_wgs, tmp_path, label):
    """Assemble + run the cluster-reduction kernel; return the single-slot
    reduction result as a numpy float32 scalar.

    ``num_wgs`` workgroups of C lanes each run the -3 handshake over the same
    (base 0) ws region; the published partials are identical across workgroups,
    so out[0] holds the reduction sum no matter which workgroup writes last."""
    asm = _build_kernel(C)
    co_path = str(tmp_path / f"{label}.co")
    with open(str(tmp_path / f"{label}.s"), "w") as f:
        f.write(asm)
    # gfx1250 is wave32: assemble with wavefront_size=None (no -mwavefrontsize32).
    assemble_kernel(asm, co_path, wavefront_size=None)

    out_bytes = 4
    # A scratch ws buffer (input) holding the C partial slots, plus a single-slot
    # output. run_on_gpu treats the first input as ws_ptr (s[4:5]) and the output
    # buffer as out_ptr (s[6:7]).
    ws_init = np.zeros(C, dtype=np.float32)
    with _Timeout(_RUN_TIMEOUT_S, f"reduction C={C} num_wgs={num_wgs}"):
        raw = run_on_gpu(
            co_path, out_bytes, inputs=(ws_init,), scalars=(),
            num_threads=C, grid=num_wgs,
        )
    return np.array(struct.unpack("I", raw), dtype=np.uint32).view(np.float32)[0]


@requires_gpu_gfx1250
class TestStreamKClusterReductionOnSim:
    """FFM gfx1250 execution of the cluster reduction fast-path handshake."""

    @pytest.fixture(params=[2, 4], ids=lambda c: f"C{c}")
    def cluster_size(self, request):
        return request.param

    def test_single_workgroup_reduction(self, cluster_size, tmp_path):
        """One workgroup: the store/release/signal/wait/acquire/accumulate
        sequence runs, does not hang, and reduces to sum(l+1 for l in range(C))."""
        C = cluster_size
        out = _run_reduction(C, num_wgs=1, tmp_path=tmp_path, label=f"skcls_1wg_C{C}")
        expected = float(sum(l + 1 for l in range(C)))  # C*(C+1)/2
        assert out == pytest.approx(expected), (
            f"C={C}: cluster reduction = {out}, expected {expected}"
        )

    def test_multi_workgroup_no_deadlock(self, cluster_size, tmp_path):
        """C concurrent workgroups run the -3 signal/wait handshake: all
        complete (no deadlock/hang under the watchdog) and the reduction is
        numerically correct."""
        C = cluster_size
        num_wgs = C
        out = _run_reduction(C, num_wgs=num_wgs, tmp_path=tmp_path,
                             label=f"skcls_{num_wgs}wg_C{C}")
        expected = float(sum(l + 1 for l in range(C)))
        assert out == pytest.approx(expected), (
            f"C={C}, num_wgs={num_wgs}: reduction = {out}, expected {expected}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
