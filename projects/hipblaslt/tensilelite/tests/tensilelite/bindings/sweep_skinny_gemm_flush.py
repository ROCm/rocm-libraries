# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LLC flush of the skinny GEMM sweep.

Same contract as bench_skinny_gemm.LlcFlush (a read-only pass over a 256 MiB
buffer, then a device synchronisation, before every timed launch), but the kernel
reads 16 bytes per lane with eight loads in flight per wait instead of one dword
per wait: the flush takes about 50 us instead of 400 us, which dominated the wall
time of the sweep (the timed kernels take 10 to 200 us). The cold-LLC time of a
kernel is the same within noise as with the original flush.
"""

import skinny_gemm_support as support
from skinny_gemm_support import LoadedKernel, hipCheck

FLUSH_MB = 256
KERNEL_NAME = "fast_flush_llc"
THREADS = 256
LOADS_IN_FLIGHT = 8
LOAD_BYTES = 16

ASM_HEAD = """
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl {name}
  .p2align 8
  .type {name},@function
{name}:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  s_lshl_b32 s2, s6, 8
  v_add_u32 v1, s2, v0
  v_lshlrev_b32 v1, 4, v1
  s_mov_b32 s7, {stride}
  s_mov_b32 s2, {rounds}
.Lloop:
"""
# Consecutive lanes read consecutive 16-byte chunks, and every load advances by
# the stride of the whole grid.
ASM_LOAD = """  global_load_dwordx4 v[{low}:{high}], v1, s[4:5]
  v_add_u32 v1, v1, s7
"""
ASM_TAIL = """  s_waitcnt vmcnt(0)
  s_sub_u32 s2, s2, 1
  s_cmp_gt_u32 s2, 0
  s_cbranch_scc1 .Lloop
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel {name}
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr {vgprs}
    .amdhsa_next_free_sgpr 12
    .amdhsa_accum_offset {vgprs}
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size {name}, .Lfunc_end0-{name}

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: {name}
    .private_segment_fixed_size: 0
    .sgpr_count: 12
    .sgpr_spill_count: 0
    .symbol: {name}.kd
    .vgpr_count: {vgprs}
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
---
  .end_amdgpu_metadata
"""


def flush_asm(workgroups: int, rounds: int) -> str:
    vgprs = 4 + LOADS_IN_FLIGHT * 4
    loads = "".join(
        ASM_LOAD.format(low=4 + 4 * i, high=7 + 4 * i) for i in range(LOADS_IN_FLIGHT)
    )
    stride = workgroups * THREADS * LOAD_BYTES
    head = ASM_HEAD.format(name=KERNEL_NAME, stride=stride, rounds=rounds)
    return head + loads + ASM_TAIL.format(name=KERNEL_NAME, vgprs=vgprs)


class FastLlcFlush:
    """Empties the LLC by reading a buffer larger than it with one workgroup per
    CU, then waits for the device; a context manager that frees the buffer."""

    def __init__(self):
        import tensilelite_jit
        from bench_skinny_gemm import packArgs

        hip = support._hip()
        props = support.getDeviceProperties()
        self.workgroups = props.multiProcessorCount
        per_round = self.workgroups * THREADS * LOAD_BYTES * LOADS_IN_FLIGHT
        # Round the buffer up so that every load is in bounds.
        rounds = support.ceilDiv(FLUSH_MB * 2**20, per_round)
        asm = flush_asm(self.workgroups, rounds)
        hsaco = tensilelite_jit.assembleToHSACO(asm, support.TARGET)
        self.kernel = LoadedKernel(hsaco, KERNEL_NAME)
        self.buffer = hipCheck(hip.hipMalloc(rounds * per_round))
        self.args = packArgs((self.buffer.as_c_void_p(),))

    def flush(self) -> None:
        hip = support._hip()
        hipCheck(
            hip.hipModuleLaunchKernel(
                self.kernel.function,
                self.workgroups,
                1,
                1,
                THREADS,
                1,
                1,
                sharedMemBytes=0,
                stream=None,
                kernelParams=None,
                extra=self.args,
            )
        )
        hipCheck(hip.hipDeviceSynchronize())

    def close(self) -> None:
        if self.buffer is None:
            return
        hipCheck(support._hip().hipFree(self.buffer))
        self.buffer = None
        self.kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.close()
