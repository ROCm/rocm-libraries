.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

# gfx1250 probe for VA_VDST ordering across the XDL and VALU
# destination pipes.  Each kernel executes:
#
#   XDL write -> unrelated VALU write -> [wait] -> DS reader of the XDL result
#
# A globally ordered VA_VDST makes depctr_va_vdst(1) sufficient: the younger
# VALU write may remain outstanding, but the older XDL write must have retired.
# A per-pipe counter does not: each pipe has only one outstanding write and the
# wait can return immediately.  va_vdst_cross_pipe_nowait is a sensitivity
# control; va_vdst_cross_pipe_wait0 is the known-safe reference.

.macro PROBE_BODY wait_kind
  s_mov_b32 s2, ttmp9
  s_load_b64 s[4:5], s[0:1], 0
  s_load_b64 s[6:7], s[0:1], 8
  s_wait_kmcnt 0

  # One 32-byte BF16 A/B record per lane.
  v_lshlrev_b32 v1, 5, v0
  global_load_b128 v[8:11], v1, s[6:7]
  global_load_b128 v[12:15], v1, s[6:7] offset:16
  global_load_b128 v[24:27], v1, s[6:7] offset:32
  global_load_b128 v[28:31], v1, s[6:7] offset:48
  s_wait_loadcnt 0

  v_mov_b32 v40, 0
  v_mov_b32 v41, 0
  v_mov_b32 v42, 0
  v_mov_b32 v43, 0
  v_mov_b32 v44, 0
  v_mov_b32 v45, 0
  v_mov_b32 v46, 0
  v_mov_b32 v47, 0

  # Four waves, 32 lanes/wave, 32 bytes/lane.
  v_lshrrev_b32 v2, 5, v0
  v_and_b32 v3, 31, v0
  v_lshlrev_b32 v4, 10, v2
  v_lshl_add_u32 v4, v3, 5, v4

  s_wait_alu 0
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15], v[24:31], v[40:47]
  v_add_nc_u32 v100, 1, v0
.if \wait_kind == 1
  s_wait_alu depctr_va_vdst(1)
.elseif \wait_kind == 2
  s_wait_alu depctr_va_vdst(0)
.endif
  ds_store_b128 v4, v[40:43]
  ds_store_b128 v4, v[44:47] offset:16
  s_wait_dscnt 0
  ds_load_b128 v[48:51], v4
  ds_load_b128 v[52:55], v4 offset:16
  s_wait_dscnt 0
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 0

  # Unique output record per workgroup/wave/lane.
  s_lshl_b32 s8, s2, 12
  v_add_nc_u32 v5, s8, v4
  global_store_b128 v5, v[48:51], s[4:5]
  global_store_b128 v5, v[52:55], s[4:5] offset:16
  s_wait_storecnt 0
  s_endpgm
.endm

.macro PROBE_KERNEL name, wait_kind
.text
.protected \name
.globl \name
.p2align 8
.type \name,@function
\name:
  PROBE_BODY \wait_kind
.p2align 6
.amdhsa_kernel \name
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 101
  .amdhsa_next_free_sgpr 9
  .amdhsa_group_segment_fixed_size 4096
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.endm

PROBE_KERNEL va_vdst_cross_pipe_nowait, 0
PROBE_KERNEL va_vdst_cross_pipe_wait1, 1
PROBE_KERNEL va_vdst_cross_pipe_wait0, 2

.amdgpu_metadata
---
amdhsa.version: [1, 1]
amdhsa.kernels:
  - .name: va_vdst_cross_pipe_nowait
    .symbol: va_vdst_cross_pipe_nowait.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 9
    .vgpr_count: 101
    .max_flat_workgroup_size: 128
    .args:
      - { .name: out, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .name: in, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: va_vdst_cross_pipe_wait1
    .symbol: va_vdst_cross_pipe_wait1.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 9
    .vgpr_count: 101
    .max_flat_workgroup_size: 128
    .args:
      - { .name: out, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .name: in, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
  - .name: va_vdst_cross_pipe_wait0
    .symbol: va_vdst_cross_pipe_wait0.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 9
    .vgpr_count: 101
    .max_flat_workgroup_size: 128
    .args:
      - { .name: out, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .name: in, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
...
.end_amdgpu_metadata
