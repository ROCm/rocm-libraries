import struct

import pytest

from gpu_test_helpers import GFX_TARGET, HAS_GFX950, assemble_and_run


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_GFX950, reason=f"requires gfx950, found {GFX_TARGET}"),
]


def _all_vgpr_no_agpr_kernel(vgpr_count, sentinel):
    """Return a tiny kernel that reserves vgpr_count VGPRs and zero AGPRs."""
    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256
  .amdhsa_next_free_vgpr {vgpr_count}
  .amdhsa_next_free_sgpr 6
  .amdhsa_group_segment_fixed_size 0
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
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32 v2, {sentinel}
  v_mov_b32 v1, 0
  global_store_dword v1, v2, s[4:5]
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
      - .name:            output
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      struct
        .address_space:   global
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 6
    .vgpr_count: {vgpr_count}
    .max_flat_workgroup_size: 1
...
.end_amdgpu_metadata
"""


def test_gfx950_can_execute_max_vgpr_without_agpr(tmp_path):
    sentinel = 0x13579BDF
    asm = _all_vgpr_no_agpr_kernel(vgpr_count=512, sentinel=sentinel)

    assert "acc[" not in asm
    assert "v_accvgpr" not in asm
    assert ".amdhsa_accum_offset 256" in asm
    assert ".amdhsa_next_free_vgpr 512" in asm
    assert ".vgpr_count: 512" in asm

    raw = assemble_and_run(
        asm,
        tmp_path,
        "max_vgpr_no_agpr",
        output_size=4,
        num_threads=1,
    )

    assert struct.unpack("I", raw)[0] == sentinel
