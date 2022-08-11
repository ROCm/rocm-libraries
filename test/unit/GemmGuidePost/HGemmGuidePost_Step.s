/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

// Component.Signature.SignatureCOV3
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
.text
.protected Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1
.globl Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1
.p2align 8
.type Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 128 // accvgpr offset
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 65 // sgprs
  .amdhsa_group_segment_fixed_size 28672 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 32 x 4 */
/* SubGroup= 4 x 64 */
/* VectorWidth=2 */
/* GlobalLoadVectorWidthA=8, GlobalLoadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1
    .symbol: 'Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            sizeC
        .size:            8
        .offset:          0
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeA
        .size:            8
        .offset:          8
        .value_kind:      by_value
        .value_type:      u64
      - .name:            sizeB
        .size:            8
        .offset:          16
        .value_kind:      by_value
        .value_type:      u64
      - .name:            D
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            alpha
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      f32
      - .name:            strideD0
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          104
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OrigStaggerUIter
        .size:            4
        .offset:          108
        .value_kind:      by_value
        .value_type:      i32
      - .name:            NumWorkGroups0
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumWorkGroups1
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      u32
      - .name:            NumFullBlocks
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            WgmRemainder1
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            MagicNumberWgmRemainder1
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetD
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetC
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetA
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
      - .name:            OffsetB
        .size:            4
        .offset:          144
        .value_kind:      by_value
        .value_type:      u32
      - .name:            padding
        .size:            4
        .offset:          148
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   28672
    .kernarg_segment_align:      8
    .kernarg_segment_size:       152
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 65
    .sgpr_spill_count:           0
    .vgpr_count:                 128
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bjlk_HHS_H_MT128x256x16_MI32x32x8x1_SE_K1:

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_i32 dst:req, src0:req, src1:req, dpp=
   v_add_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_i32 dst:req, src0:req, src1:req, dpp=
   v_sub_i32 \dst, \src0, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_or_b32 dst:req, src0:req, shiftCnt:req, src1:req
    v_lshl_or_b32 \dst, \src0, \shiftCnt, \src1
.endm

.macro _v_dot2acc_f32_f16 dst, src0, src1
v_dot2c_f32_f16 \dst, \src0, \src1
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i16 dst, src0, src1=
   v_cmpx_ne_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i32 dst, src0, src1=
   v_cmpx_ne_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_i64 dst, src0, src1=
   v_cmpx_ne_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u16 dst, src0, src1=
   v_cmpx_ne_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u32 dst, src0, src1=
   v_cmpx_ne_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ne_u64 dst, src0, src1=
   v_cmpx_ne_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst, \src0, \src1 
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst, \src0, \src1 
.endm
.macro _v_mac_f32 c:req, a:req, b:req
    v_fmac_f32 \c, \a, \b
.endmacro

/* scale global load macros */
.macro _s_load_b32 dst base offset
    s_load_dword \dst \base \offset
.endm

.macro _s_load_b64 dst base offset
    s_load_dwordx2 \dst \base \offset
.endm

.macro _s_load_b128 dst base offset
    s_load_dwordx4 \dst \base \offset
.endm

.macro _s_load_b256 dst base offset
    s_load_dwordx8 \dst \base \offset
.endm

.macro _s_load_b512 dst base offset
    s_load_dwordx16 \dst \base \offset
.endm


/* ds operation macros */
.macro _ds_load_u8 dst src offset
    ds_read_u8 \dst \src \offset
.endm

.macro _ds_load_u8_d16_hi dst src offset
    ds_read_u8_d16_hi \dst \src \offset
.endm

.macro _ds_load_u16 dst src offset
    ds_read_u16 \dst \src \offset
.endm

.macro _ds_load_u16_d16_hi dst src offset
    ds_read_u16_d16_hi \dst \src \offset
.endm

.macro _ds_load_b32 dst src offset
    ds_read_b32 \dst \src \offset
.endm

.macro _ds_load_b64 dst src offset
    ds_read_b64 \dst \src \offset
.endm

.macro _ds_load_b128 dst src offset
    ds_read_b128 \dst \src \offset
.endm

.macro _ds_store_b8 dst src offset
    ds_write_b8 \dst \src \offset
.endm

.macro _ds_store_b8_d16_hi dst src offset
    ds_write_b8_d16_hi \dst \src \offset
.endm

.macro _ds_store_b16 dst src offset
    ds_write_b16 \dst \src \offset
.endm

.macro _ds_store_b16_d16_hi dst src offset
    ds_write_b16_d16_hi \dst \src \offset
.endm

.macro _ds_store_b32 dst src offset
    ds_write_b32 \dst \src \offset
.endm

.macro _ds_store_b64 dst src offset
    ds_write_b64 \dst \src \offset
.endm

.macro _ds_store_b128 dst src offset
    ds_write_b128 \dst \src \offset
.endm

.macro _ds_load2_b32 dst src offset1 offset2
    ds_read2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_load2_b64 dst src offset1 offset2
    ds_read2_b64 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b32 dst src offset1 offset2
    ds_write2_b32 \dst \src \offset1 \offset2
.endm

.macro _ds_store2_b64 dst src offset1 offset2
    ds_write2_b64 \dst \src \offset1 \offset2
.endm


/* buffer memory operation macros */
.macro _buffer_load_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dword \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx2 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b96 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx3 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_b128 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_dwordx4 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_short_d16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_b16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_short_d16_hi \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ubyte_d16 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_d16_hi_u8 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ubyte_d16_hi \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_load_u16 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_load_ushort \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b32 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dword \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b64 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx2 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b96 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx3 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b128 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_dwordx4 \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_short \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b16 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_short_d16_hi \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_byte \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_store_d16_hi_b8 src voffset base soffset offen ioffset md0 md1 md2
    buffer_store_byte_d16_hi \src \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b32 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm

.macro _buffer_atomic_cmpswap_b64 dst voffset base soffset offen ioffset md0 md1 md2
    buffer_atomic_cmpswap_x2 \dst \voffset \base \soffset \offen \ioffset \md0 \md1 \md2
.endm


/* buffer memory operation macros */
.macro _flat_load_b32 dst base md0 md1 md2
    flat_load_dword \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_b32 base src md0 md1 md2
    flat_store_dword \base \src \md0 \md1 \md2
.endm

.macro _flat_load_b64 dst base md0 md1 md2
    flat_load_dwordx2 \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_b64 base src md0 md1 md2
    flat_store_dwordx2 \base \src \md0 \md1 \md2
.endm

.macro _flat_load_b96 dst base md0 md1 md2
    flat_load_dwordx3 \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_b96 base src md0 md1 md2
    flat_store_dwordx3 \base \src \md0 \md1 \md2
.endm

.macro _flat_load_b128 dst base md0 md1 md2
    flat_load_dwordx4 \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_b128 base src md0 md1 md2
    flat_store_dwordx4 \base \src \md0 \md1 \md2
.endm

.macro _flat_load_d16_b16 dst base md0 md1 md2
    flat_load_short_d16 \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_d16_b16 base src md0 md1 md2
    flat_store_short_d16 \base \src \md0 \md1 \md2
.endm

.macro _flat_load_d16_hi_b16 dst base md0 md1 md2
    flat_load_short_d16_hi \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_d16_hi_b16 base src md0 md1 md2
    flat_store_short_d16_hi \base \src \md0 \md1 \md2
.endm

.macro _flat_load_d16_u8 dst base md0 md1 md2
    flat_load_ubyte_d16 \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_d16_u8 base src md0 md1 md2
    flat_store_ubyte_d16 \base \src \md0 \md1 \md2
.endm

.macro _flat_load_d16_hi_u8 dst base md0 md1 md2
    flat_load_ubyte_d16_hi \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_d16_hi_u8 base src md0 md1 md2
    flat_store_ubyte_d16_hi \base \src \md0 \md1 \md2
.endm

.macro _flat_load_u16 dst base md0 md1 md2
    flat_load_ushort \dst \base \md0 \md1 \md2
.endm

.macro _flat_store_u16 base src md0 md1 md2
    flat_store_ushort \base \src \md0 \md1 \md2
.endm

.macro _flat_atomic_cmpswap_b32 tmp base data md
    flat_atomic_cmpswap \tmp \base \data \md
.endm


/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    _v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 0
.set vgprValuA_X1_I0, 4
.set vgprG2LA, 30
.set vgprValuB_X0_I0, 8
.set vgprValuB_X1_I0, 16
.set vgprG2LB, 34
.set vgprLocalWriteAddrA, 24
.set vgprLocalWriteAddrB, 25
.set vgprGlobalReadOffsetA, 26
.set vgprGlobalReadOffsetB, 27
.set vgprLocalReadAddrA, 42
.set vgprLocalReadAddrB, 43
.set vgprSerial, 44
/* Num VGPR=128 */
/* Num AccVGPR=128 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprLoopCounterL, 5
.set sgprOrigLoopCounter, 6
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeA, 24
.set sgprTensor2dSizeB, 26
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprAlpha, 36
.set sgprStridesD, 37
.set sgprStridesC, 39
.set sgprStridesA, 41
.set sgprStridesB, 43
.set sgprSizesFree, 45
.set sgprSizesSum, 48
.set sgprOrigStaggerUIter, 49
.set sgprNumWorkGroups0, 50
.set sgprNumWorkGroups1, 51
.set sgprNumFullBlocks, 52
.set sgprWgmRemainder1, 53
.set sgprMagicNumberWgmRemainder1, 54
.set sgprOffsetD, 55
.set sgprOffsetC, 56
.set sgprOffsetA, 57
.set sgprOffsetB, 58
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprGlobalReadIncsA, 7
.set sgprGlobalReadIncsB, 49
/* max SGPR=65 */

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideB1J, 1
.set sgprStrideBL, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 256
.set DepthU, 16
.set GSU, 1
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideAL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset0I], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffset1J:req vgprOffsetL:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideBL], v[\vgprOffsetL] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffset1J], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x8, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x1, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          // 
v_rcp_f32 v[\vQuotient], v[\vQuotient]             // 
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] // 
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         // 
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] // 
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]  // 
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]      // 
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder] // 
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder] // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] // 
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder] // 
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder] // 
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient] // 
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]    // 
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]          // 
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc               // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc // 
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1] // 
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]                // 
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder] // final result
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x7000                               // LDS clamp at 28672 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Load Kernel Args */
_s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // 
_s_load_b512 s[40:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48 // 
_s_load_b64 s[56:57], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88 // 
_s_load_b32 s58, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // 
s_waitcnt lgkmcnt(0)                               // wait for 148 bytes of kern args
s_lshl_b32 s[sgprOffsetD], s[sgprOffsetD], 0x1     // elements offset to bytes offset
s_add_u32 s[sgprAddressD+0], s[sgprAddressD+0], s[sgprOffsetD] // add offset to buffer address
s_addc_u32 s[sgprAddressD+1], s[sgprAddressD+1], 0 // add offset to buffer address
s_lshl_b32 s[sgprOffsetC], s[sgprOffsetC], 0x1     // elements offset to bytes offset
s_add_u32 s[sgprAddressC+0], s[sgprAddressC+0], s[sgprOffsetC] // add offset to buffer address
s_addc_u32 s[sgprAddressC+1], s[sgprAddressC+1], 0 // add offset to buffer address
s_lshl_b32 s[sgprOffsetA], s[sgprOffsetA], 0x1     // elements offset to bytes offset
s_add_u32 s[sgprAddressA+0], s[sgprAddressA+0], s[sgprOffsetA] // add offset to buffer address
s_addc_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // add offset to buffer address
s_lshl_b32 s[sgprOffsetB], s[sgprOffsetB], 0x1     // elements offset to bytes offset
s_add_u32 s[sgprAddressB+0], s[sgprAddressB+0], s[sgprOffsetB] // add offset to buffer address
s_addc_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // add offset to buffer address
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

.set OffsetD, UNDEF
.set OffsetC, UNDEF
.set OffsetA, UNDEF
.set OffsetB, UNDEF


/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a/b */

/*lr0I*/
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 31, v2                               // 1. N offset: nIdx = wtid % MI_N(32)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 3. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 5, v2                            // 4. K offset: kIdx = wtid / (MIN(32) * MIBB(1))
v_lshlrev_b32 v2, 0x9, v2                          // 4. K offset: lrKOffset = kIdx * mStride(512)
_v_add_u32 v1, v2, v1                              // 5. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // 6. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v0, 1, v0                                // 6. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v0, 0x5, v0                          // 6. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
_v_add_u32 v1, v0, v1                              // 7. final local read offset: flrOffset = lrOffset + WOffset
/*lr1J*/
v_and_b32 v3, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v2, 31, v3                               // 1. N offset: nIdx = wtid % MI_N(32)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
                                                   // 2. block offset: bnIdx = bnIdx % num1DBlocks(1) is 0. do nothing
                                                   // 3. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v3, 5, v3                            // 4. K offset: kIdx = wtid / (MIN(32) * MIBB(1))
v_lshlrev_b32 v3, 0xa, v3                          // 4. K offset: lrKOffset = kIdx * mStride(1024)
_v_add_u32 v2, v3, v2                              // 5. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v0, 7, v[vgprSerial]                 // 6. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v0, 1, v0                                // 6. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v0, 0x5, v0                          // 6. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
_v_add_u32 v2, v0, v2                              // 7. final local read offset: flrOffset = lrOffset + WOffset


/* local read addresses: final offsets a */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
s_mov_b32 s55, 128                                 // LSU offset: stride = MT0(128) + PAD0(0)
v_mul_lo_u32 v0, s55, v0                           // LSU offset: lsuoffset = sgid*(MT0+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x1 // Final Offset: offset = (lro0*VW+lsuoffset)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // LSU offset: sgid = Serial / subGroup(256)
s_mov_b32 s55, 256                                 // LSU offset: stride = MT1(256) + PAD1(0)
v_mul_lo_u32 v0, s55, v0                           // LSU offset: lsuoffset = sgid*(MT1+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v2, 0x1 // Final Offset: offset = (lro1*VW+lsuoffset)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x1000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)



/******************************************/
/* Begin setupNewTile, isPap=False           */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */
s_mov_b32 s63, 0x8888889L                          // magic number for WGM==15
s_mul_hi_u32 s61, s[sgprWorkGroup1], s63           // s_magic mul
s_mul_i32 s60, s[sgprWorkGroup1], s63              // s_magic mul
s_lshr_b64 s[60:61], s[60:61], 31                  // sMagicDiv
s_mul_i32 s61, s60, 15                             // quotient * non-magic divisor
s_sub_u32 s61, s[sgprWorkGroup1], s61              // WorkGroup1=remainder
s_mul_i32 s61, s61, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s61, s61, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
s_cmp_ge_u32 s60, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s63, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s62, s[sgprWgmRemainder1], 15        // 
s_mul_hi_u32 s3, s61, s63                          // s_magic mul
s_mul_i32 s2, s61, s63                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s62 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s61, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s60, s60, 15                             // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s60 // wg1 += blockId * WGM


/* global read addresses: tile offset assignment a */

/* LVCA = 16 */
/* v0 = (local)groA-tile = serial%LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial/LVCA */
v_and_b32 v4, 63, v[vgprSerial]                    // v4 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v4                            // v1 = v4 / 16
v_and_b32 v0, 15, v4                               // v0 = v4 % 16
v_readfirstlane_b32 s55, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s55, s55, 0x6                           // WaveId
s_mul_i32 s55, s55, 4                              // Global Read Wave: each wave loads continuous lsp(4)*nrp(1) columns
_v_add_u32 v1, s55, v1                             // Global Read Wave: add back to column index
/* gro-tile *= glvw */
v_lshlrev_b32 v0, 0x3, v0                          // v0 = v0 * 8


/* global read addresses: tile offset assignment b */

/* LVCB = 32 */
/* v2 = (local)groB-tile = serial%LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial/LVCB */
v_and_b32 v6, 63, v[vgprSerial]                    // v6 = v[vgprSerial] % 64
v_lshrrev_b32 v3, 5, v6                            // v3 = v6 / 32
v_and_b32 v2, 31, v6                               // v2 = v6 % 32
v_readfirstlane_b32 s55, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s55, s55, 0x6                           // WaveId
s_mul_i32 s55, s55, 4                              // Global Read Wave: each wave loads continuous lsp(2)*nrp(2) columns
_v_add_u32 v3, s55, v3                             // Global Read Wave: add back to column index
/* gro-tile *= glvw */
v_lshlrev_b32 v2, 0x3, v2                          // v2 = v2 * 8


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other free assignments */

/* s[sgprWorkGroup2] */


/* global read addresses: tile offsets a */

v_mov_b32 v4, v0                                   // groA0I_0


/* global read addresses: tile offsets b */

v_mov_b32 v5, v2                                   // groB1J_0


/* global read addresses: unroll offsets a */

v_mov_b32 v6, v1                                   // groAL_0


/* global read addresses: unroll offsets b */

v_mov_b32 v7, v3                                   // groBL_0
_v_add_co_u32 v8, vcc, 2, v7                       // groBL_1 + LSPB


/* global read addresses: shift b */

s_mul_i32 s55, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_sub_u32 s55, s[sgprSizeJ], s55                   // edge = Size1J - WG*MT
s_sub_u32 s55, s55, 8                              // edge -= margin(8)
v_mov_b32 v9, s55                                  // edge vgpr = Size1J- WG*MT - margin(8)
v_min_i32 v5, v9, v5                               // offset = (offset < edge) ? offset(v5) : edge(v9)


/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  4,  6, 9 // gROA_0_0_0_0


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  5,  7, 9 // gROB_0_0_0_0
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+1,  5,  8, 9 // gROB_0_0_1_0


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s63, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s62, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s62 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s63 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s61, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s60, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s62, s62, s60                            // accum wg term to tilestart
s_addc_u32 s63, s63, s61                           // accum wg term to tilestart
s_lshl_b64 s[62:63], s[62:63], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s62    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s63   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s63, s[sgprWorkGroup1], 256           // WorkGroup[01] * MT
s_mul_i32 s62, s[sgprWorkGroup1], 256              // WorkGroup[01] * MT
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s62 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s63 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s61, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s60, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s62, s62, s60                            // accum wg term to tilestart
s_addc_u32 s63, s63, s61                           // accum wg term to tilestart
s_lshl_b64 s[62:63], s[62:63], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s62    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s63   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mul_i32 s[sgprGlobalReadIncsA+0], DepthU*BpeA, s[sgprStrideAL] // incrA unrollIdx)


/* global read addresses: increments b */

s_mul_i32 s[sgprGlobalReadIncsB+0], DepthU*BpeB, s[sgprStrideBL] // incrB unrollIdx)


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v1 */

/* lwaUnrollAssignmentB = v3 */


/* local write addresses: first offset a */

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v1     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x1 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x100, v3    // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x1000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048*2







/* declare loop num iterations */


s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 4 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 16
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/* prefetch: global -> local */

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 ShadowInitStart_10                  // skip to ShadowInitStart iter b/c numIter==0


_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0


_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0


/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/******************************************/
/* End setupNewTile, isPap=False             */
/******************************************/

ShadowInitStart_10: // 

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000                // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000                // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s62, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s61, s62, s[sgprStrideC1J]            // CScale s62 by Stride
s_mul_i32 s60, s62, s[sgprStrideC1J]               // CScale s62 by Stride
s_lshl_b64 s[60:61], s[60:61], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s60    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s61   // add hi to SRD
s_mul_hi_u32 s61, s62, s[sgprStrideD1J]            // Scale s62 by Stride
s_mul_i32 s60, s62, s[sgprStrideD1J]               // Scale s62 by Stride
s_lshl_b64 s[60:61], s[60:61], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s60    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s61   // add hi to SRD

s_mul_hi_u32 s61, s[sgprWorkGroup2], s[sgprStrideCK] // CScale s[sgprWorkGroup2] by Stride
s_mul_i32 s60, s[sgprWorkGroup2], s[sgprStrideCK]  // CScale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[60:61], s[60:61], 1                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s60        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s61       // add hi to SRD
s_mul_hi_u32 s61, s[sgprWorkGroup2], s[sgprStrideDK] // Scale s[sgprWorkGroup2] by Stride
s_mul_i32 s60, s[sgprWorkGroup2], s[sgprStrideDK]  // Scale s[sgprWorkGroup2] by Stride
s_lshl_b64 s[60:61], s[60:61], 1                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s60        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s61       // add hi to SRD



/* initC: remove C-tile 0-0 from pool */

/* initC: remove AB-tile 0-24 from pool */
v_accvgpr_write acc0, 0x0                          // initC
v_accvgpr_write acc1, 0x0                          // initC
v_accvgpr_write acc2, 0x0                          // initC
v_accvgpr_write acc3, 0x0                          // initC
v_accvgpr_write acc4, 0x0                          // initC
v_accvgpr_write acc5, 0x0                          // initC
v_accvgpr_write acc6, 0x0                          // initC
v_accvgpr_write acc7, 0x0                          // initC
v_accvgpr_write acc8, 0x0                          // initC
v_accvgpr_write acc9, 0x0                          // initC
v_accvgpr_write acc10, 0x0                         // initC
v_accvgpr_write acc11, 0x0                         // initC
v_accvgpr_write acc12, 0x0                         // initC
v_accvgpr_write acc13, 0x0                         // initC
v_accvgpr_write acc14, 0x0                         // initC
v_accvgpr_write acc15, 0x0                         // initC
v_accvgpr_write acc16, 0x0                         // initC
v_accvgpr_write acc17, 0x0                         // initC
v_accvgpr_write acc18, 0x0                         // initC
v_accvgpr_write acc19, 0x0                         // initC
v_accvgpr_write acc20, 0x0                         // initC
v_accvgpr_write acc21, 0x0                         // initC
v_accvgpr_write acc22, 0x0                         // initC
v_accvgpr_write acc23, 0x0                         // initC
v_accvgpr_write acc24, 0x0                         // initC
v_accvgpr_write acc25, 0x0                         // initC
v_accvgpr_write acc26, 0x0                         // initC
v_accvgpr_write acc27, 0x0                         // initC
v_accvgpr_write acc28, 0x0                         // initC
v_accvgpr_write acc29, 0x0                         // initC
v_accvgpr_write acc30, 0x0                         // initC
v_accvgpr_write acc31, 0x0                         // initC
v_accvgpr_write acc32, 0x0                         // initC
v_accvgpr_write acc33, 0x0                         // initC
v_accvgpr_write acc34, 0x0                         // initC
v_accvgpr_write acc35, 0x0                         // initC
v_accvgpr_write acc36, 0x0                         // initC
v_accvgpr_write acc37, 0x0                         // initC
v_accvgpr_write acc38, 0x0                         // initC
v_accvgpr_write acc39, 0x0                         // initC
v_accvgpr_write acc40, 0x0                         // initC
v_accvgpr_write acc41, 0x0                         // initC
v_accvgpr_write acc42, 0x0                         // initC
v_accvgpr_write acc43, 0x0                         // initC
v_accvgpr_write acc44, 0x0                         // initC
v_accvgpr_write acc45, 0x0                         // initC
v_accvgpr_write acc46, 0x0                         // initC
v_accvgpr_write acc47, 0x0                         // initC
v_accvgpr_write acc48, 0x0                         // initC
v_accvgpr_write acc49, 0x0                         // initC
v_accvgpr_write acc50, 0x0                         // initC
v_accvgpr_write acc51, 0x0                         // initC
v_accvgpr_write acc52, 0x0                         // initC
v_accvgpr_write acc53, 0x0                         // initC
v_accvgpr_write acc54, 0x0                         // initC
v_accvgpr_write acc55, 0x0                         // initC
v_accvgpr_write acc56, 0x0                         // initC
v_accvgpr_write acc57, 0x0                         // initC
v_accvgpr_write acc58, 0x0                         // initC
v_accvgpr_write acc59, 0x0                         // initC
v_accvgpr_write acc60, 0x0                         // initC
v_accvgpr_write acc61, 0x0                         // initC
v_accvgpr_write acc62, 0x0                         // initC
v_accvgpr_write acc63, 0x0                         // initC
v_accvgpr_write acc64, 0x0                         // initC
v_accvgpr_write acc65, 0x0                         // initC
v_accvgpr_write acc66, 0x0                         // initC
v_accvgpr_write acc67, 0x0                         // initC
v_accvgpr_write acc68, 0x0                         // initC
v_accvgpr_write acc69, 0x0                         // initC
v_accvgpr_write acc70, 0x0                         // initC
v_accvgpr_write acc71, 0x0                         // initC
v_accvgpr_write acc72, 0x0                         // initC
v_accvgpr_write acc73, 0x0                         // initC
v_accvgpr_write acc74, 0x0                         // initC
v_accvgpr_write acc75, 0x0                         // initC
v_accvgpr_write acc76, 0x0                         // initC
v_accvgpr_write acc77, 0x0                         // initC
v_accvgpr_write acc78, 0x0                         // initC
v_accvgpr_write acc79, 0x0                         // initC
v_accvgpr_write acc80, 0x0                         // initC
v_accvgpr_write acc81, 0x0                         // initC
v_accvgpr_write acc82, 0x0                         // initC
v_accvgpr_write acc83, 0x0                         // initC
v_accvgpr_write acc84, 0x0                         // initC
v_accvgpr_write acc85, 0x0                         // initC
v_accvgpr_write acc86, 0x0                         // initC
v_accvgpr_write acc87, 0x0                         // initC
v_accvgpr_write acc88, 0x0                         // initC
v_accvgpr_write acc89, 0x0                         // initC
v_accvgpr_write acc90, 0x0                         // initC
v_accvgpr_write acc91, 0x0                         // initC
v_accvgpr_write acc92, 0x0                         // initC
v_accvgpr_write acc93, 0x0                         // initC
v_accvgpr_write acc94, 0x0                         // initC
v_accvgpr_write acc95, 0x0                         // initC
v_accvgpr_write acc96, 0x0                         // initC
v_accvgpr_write acc97, 0x0                         // initC
v_accvgpr_write acc98, 0x0                         // initC
v_accvgpr_write acc99, 0x0                         // initC
v_accvgpr_write acc100, 0x0                        // initC
v_accvgpr_write acc101, 0x0                        // initC
v_accvgpr_write acc102, 0x0                        // initC
v_accvgpr_write acc103, 0x0                        // initC
v_accvgpr_write acc104, 0x0                        // initC
v_accvgpr_write acc105, 0x0                        // initC
v_accvgpr_write acc106, 0x0                        // initC
v_accvgpr_write acc107, 0x0                        // initC
v_accvgpr_write acc108, 0x0                        // initC
v_accvgpr_write acc109, 0x0                        // initC
v_accvgpr_write acc110, 0x0                        // initC
v_accvgpr_write acc111, 0x0                        // initC
v_accvgpr_write acc112, 0x0                        // initC
v_accvgpr_write acc113, 0x0                        // initC
v_accvgpr_write acc114, 0x0                        // initC
v_accvgpr_write acc115, 0x0                        // initC
v_accvgpr_write acc116, 0x0                        // initC
v_accvgpr_write acc117, 0x0                        // initC
v_accvgpr_write acc118, 0x0                        // initC
v_accvgpr_write acc119, 0x0                        // initC
v_accvgpr_write acc120, 0x0                        // initC
v_accvgpr_write acc121, 0x0                        // initC
v_accvgpr_write acc122, 0x0                        // initC
v_accvgpr_write acc123, 0x0                        // initC
v_accvgpr_write acc124, 0x0                        // initC
v_accvgpr_write acc125, 0x0                        // initC
v_accvgpr_write acc126, 0x0                        // initC
v_accvgpr_write acc127, 0x0                        // initC

s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_11                   // Only branch on scc1
s_getpc_B64 s[60:61]                               // addr of next instr
s_add_i32 s62, LoopEndL_2, 0x4                     // target branch offset
s_add_u32 s60, s60, s62                            // add target branch offset
s_addc_u32 s61, s61, 0                             // add high and carry
s_setpc_b64 s[60:61]                               // branch to LoopEndL_2
label_NoBranch_11:

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=08wait for global read


/* local write a */
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local write b */
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1024 // lwoB_0_0_1_0 = (0*LSCB) + (1*LSPB)(*MT1J+PAD) = 1024


/* local write swap a */

v_xor_b32 v[vgprLocalWriteAddrA+0], 0x4000, v[vgprLocalWriteAddrA+0] // swap Red Blk


/* local write swap b */

v_xor_b32 v[vgprLocalWriteAddrB+0], 0x4000, v[vgprLocalWriteAddrB+0] // swap Red Blk



s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-10prefetch wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* local read prefetch a */

_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v45, v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v46, v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v47, v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v48, v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0


/* local read prefetch b */

_ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v49, v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v50, v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v51, v[vgprLocalReadAddrB] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v52, v[vgprLocalReadAddrB] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+4], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v53, v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+5], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v54, v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+6], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v55, v[vgprLocalReadAddrB] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+7], v[vgprLocalReadAddrB] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v56, v[vgprLocalReadAddrB] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0


/* local read inc a */

/* N/A, lro->1024 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */


/* local read inc b */

/* N/A, lro->2048 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */



/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_12:
s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 LoopEndL_2                          // do not enter LoopL
LoopBeginL_1:


/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

label_0013: // LoopCopy1 


/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */



s_cmp_eq_i32 s[sgprLoopCounterL], 1                // is this the last iteration
s_cmov_b32 s[sgprSrdA+2], 0                        // Set limit to 0 for last iteration
s_cmov_b32 s[sgprSrdB+2], 0                        // Set limit to 0 for last iteration


/* iter 0 (reset local read pointers iteration)  (swap local read pointers iteration)  */

/*  grEndMfmaIndex:2, lwStartMfmaIndex:7, lwEndMfmaIndex:9  */
/*  numMfmaForLR:4, barrierMfmaIndex:11  */
/*  mfmaIndex:0  */
_buffer_load_b128 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v45 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v46 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v49 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+1], v50 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v47 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v48 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[0+0:15+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:15]
/*  mfmaIndex:1  */
_ds_load_u16 v[vgprValuA_X1_I0+0], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v57, v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuA_X1_I0+1], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v58, v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuB_X1_I0+0], v[vgprLocalReadAddrB] offset:4096 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v61, v[vgprLocalReadAddrB] offset:4608 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
_buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_or_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+2], v51 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+3], v52 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+4], v53 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0+5], v54 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[16+0:31+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[16:31]
/*  mfmaIndex:2  */
_ds_load_u16 v[vgprValuB_X1_I0+1], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v62, v[vgprLocalReadAddrB] offset:5632 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuA_X1_I0+2], v[vgprLocalReadAddrA] offset:2176 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v59, v[vgprLocalReadAddrA] offset:2432 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuA_X1_I0+3], v[vgprLocalReadAddrA] offset:2688 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v60, v[vgprLocalReadAddrA] offset:2944 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=1 iui=0
_buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_1_0
/* pack scheduling: packAIdx:4, packBIdx:4 */
v_or_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+6], v55 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+7], v56 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[32+0:47+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], a[32:47]
/*  mfmaIndex:3  */
_ds_load_u16 v[vgprValuB_X1_I0+2], v[vgprLocalReadAddrB] offset:4224 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v63, v[vgprLocalReadAddrB] offset:4736 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuB_X1_I0+3], v[vgprLocalReadAddrB] offset:5248 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v64, v[vgprLocalReadAddrB] offset:5760 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuB_X1_I0+4], v[vgprLocalReadAddrB] offset:4352 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=2 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v65, v[vgprLocalReadAddrB] offset:4864 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=2 rIdx=1 oIdx=0 buffer=1 iui=0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
v_mfma_f32_32x32x8f16 a[48+0:63+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], a[48:63]
/*  mfmaIndex:4  */
_ds_load_u16 v[vgprValuB_X1_I0+5], v[vgprLocalReadAddrB] offset:5376 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=2 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v66, v[vgprLocalReadAddrB] offset:5888 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=2 rIdx=3 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuB_X1_I0+6], v[vgprLocalReadAddrB] offset:4480 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=3 rIdx=0 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v67, v[vgprLocalReadAddrB] offset:4992 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=3 rIdx=1 oIdx=0 buffer=1 iui=0
_ds_load_u16 v[vgprValuB_X1_I0+7], v[vgprLocalReadAddrB] offset:5504 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=3 rIdx=2 oIdx=0 buffer=1 iui=0
_ds_load_u16_d16_hi v68, v[vgprLocalReadAddrB] offset:6016 // L -> Reg lro=2048 swapByteOffset=0 ti=64 vIdx=3 rIdx=3 oIdx=0 buffer=1 iui=0
/* localReadsVacancy: latencyLeft 1 */
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32
v_mfma_f32_32x32x8f16 a[64+0:79+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[64:79]
/*  mfmaIndex:5  */
/* localReadsVacancy: latencyLeft 13 */
v_mfma_f32_32x32x8f16 a[80+0:95+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[80:95]
/*  mfmaIndex:6  */
/* localReadsVacancy: latencyLeft 13 */
v_mfma_f32_32x32x8f16 a[96+0:111+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], a[96:111]
/*  mfmaIndex:7  */
/* localReadsVacancy: latencyLeft 3 */
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // lgkmcnt=-1 vmcnt=2wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], 0x4000, v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], 0x4000, v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_f32_32x32x8f16 a[112+0:127+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], a[112:127]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=16 */


/* iter 1 (swap and reset local write pointers iteration)  */

/*  grEndMfmaIndex:2, lwStartMfmaIndex:7, lwEndMfmaIndex:9  */
/*  numMfmaForLR:4, barrierMfmaIndex:11  */
/*  mfmaIndex:8  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // lgkmcnt=-1 vmcnt=1wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0
s_waitcnt lgkmcnt(2)                               // lgkmcnt=0 vmcnt=-1wait for prior local read local write old=0, new=2 newLW=2 newLR=0
/* pack scheduling: packAIdx:2, packBIdx:2 */
v_or_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0+0], v57 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0+1], v58 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+0], v[vgprValuB_X1_I0+0], v61 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+1], v[vgprValuB_X1_I0+1], v62 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0+2], v59 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0+3], v60 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[0+0:15+0], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], a[0:15]
/*  mfmaIndex:9  */
/* sched write - iter 1 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=0wait for global read before writing to local
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1024 // lwoB_0_0_1_0 = (0*LSCB) + (1*LSPB)(*MT1J+PAD) = 1024

/* local write swap offsets a */
v_xor_b32 v[vgprLocalWriteAddrA+0], 0x4000, v[vgprLocalWriteAddrA+0] // swap Red Blk

/* local write swap offsets b */
v_xor_b32 v[vgprLocalWriteAddrB+0], 0x4000, v[vgprLocalWriteAddrB+0] // swap Red Blk
/* pack scheduling: packAIdx:4, packBIdx:2 */
v_or_b32 v[vgprValuB_X1_I0+2], v[vgprValuB_X1_I0+2], v63 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+3], v[vgprValuB_X1_I0+3], v64 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+4], v[vgprValuB_X1_I0+4], v65 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+5], v[vgprValuB_X1_I0+5], v66 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[16+0:31+0], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[vgprValuB_X1_I0+0+0+0:vgprValuB_X1_I0+0+0+0+1], a[16:31]
/*  mfmaIndex:10  */
/* pack scheduling: packAIdx:4, packBIdx:4 */
v_or_b32 v[vgprValuB_X1_I0+6], v[vgprValuB_X1_I0+6], v67 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X1_I0+7], v[vgprValuB_X1_I0+7], v68 // pack two half Vgpr to one Vgpr
v_mfma_f32_32x32x8f16 a[32+0:47+0], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], a[32:47]
/*  mfmaIndex:11  */
s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-13wait for local write
s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //
v_mfma_f32_32x32x8f16 a[48+0:63+0], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[vgprValuB_X1_I0+2+0+0:vgprValuB_X1_I0+2+0+0+1], a[48:63]
/*  mfmaIndex:12  */
_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v45, v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v46, v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v49, v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
v_mfma_f32_32x32x8f16 a[64+0:79+0], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], a[64:79]
/*  mfmaIndex:13  */
_ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v50, v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v47, v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v48, v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
v_mfma_f32_32x32x8f16 a[80+0:95+0], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[vgprValuB_X1_I0+4+0+0:vgprValuB_X1_I0+4+0+0+1], a[80:95]
/*  mfmaIndex:14  */
_ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v51, v[vgprLocalReadAddrB] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v52, v[vgprLocalReadAddrB] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+4], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v53, v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
v_mfma_f32_32x32x8f16 a[96+0:111+0], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], a[96:111]
/*  mfmaIndex:15  */
_ds_load_u16 v[vgprValuB_X0_I0+5], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v54, v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+6], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v55, v[vgprLocalReadAddrB] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+7], v[vgprLocalReadAddrB] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v56, v[vgprLocalReadAddrB] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0
v_mfma_f32_32x32x8f16 a[112+0:127+0], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], v[vgprValuB_X1_I0+6+0+0:vgprValuB_X1_I0+6+0+0+1], a[112:127]
/* numPrefetchIter=1 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=16 */




/******************************************/
/* Unrolled Loop - End                    */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 LoopBeginL_1                        // restart LoopL
LoopEndL_2:


/* Before NLL: Check VGPR.checkin for INT8 LW */


/******************************************/
/* Tail Loop                              */
/******************************************/


/* local write reset offsets a */

v_and_b32 v[vgprLocalWriteAddrA], 0xf03fff, v[vgprLocalWriteAddrA] // reset to Red


/* local write reset offsets b */

v_and_b32 v[vgprLocalWriteAddrB], 0xf03fff, v[vgprLocalWriteAddrB] // reset to Red


s_cmp_eq_u32 s[sgprOrigLoopCounter], 0             // completely skipped unroll loop?
s_cselect_b32 s60, 0, s[sgprGlobalReadIncsA]       // force to 0?
s_cselect_b32 s61, 0, s[sgprGlobalReadIncsB]       // force to 0?
s_sub_u32  s[sgprSrdA+0], s[sgprSrdA+0], s60       // gra SRD -= inc(lower)
s_subb_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD -= inc(upper)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s60 // limit -= inc)
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

s_sub_u32  s[sgprSrdB+0], s[sgprSrdB+0], s61       // gra SRD -= inc(lower)
s_subb_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD -= inc(upper)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s61 // limit -= inc)
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 15, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 16
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 SkipTailLoopL_8                     // skip to end of tail loop b/c numIter==0


/* Update M0 for DTLDS */



/* global read a */

/* g2l=0, load component 0 */
_buffer_load_d16_b16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:2 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v45 // HasEccHalf: pack
/* g2l=0, load component 2 */
_buffer_load_d16_b16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:6 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v45 // HasEccHalf: pack
/* g2l=0, load component 4 */
_buffer_load_d16_b16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:10 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v45 // HasEccHalf: pack
/* g2l=0, load component 6 */
_buffer_load_d16_b16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v45 // HasEccHalf: pack


/* Update M0 for DTLDS */



/* global read b */

/* g2l=0, load component 0 */
_buffer_load_d16_b16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:2 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v45 // HasEccHalf: pack
/* g2l=0, load component 2 */
_buffer_load_d16_b16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:6 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v45 // HasEccHalf: pack
/* g2l=0, load component 4 */
_buffer_load_d16_b16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:10 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v45 // HasEccHalf: pack
/* g2l=0, load component 6 */
_buffer_load_d16_b16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v45 // HasEccHalf: pack
/* g2l=4, load component 0 */
_buffer_load_d16_b16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:2 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v45 // HasEccHalf: pack
/* g2l=4, load component 2 */
_buffer_load_d16_b16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=4, load component 3 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:6 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v45 // HasEccHalf: pack
/* g2l=4, load component 4 */
_buffer_load_d16_b16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:8 // load one buffer value
/* g2l=4, load component 5 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:10 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v45 // HasEccHalf: pack
/* g2l=4, load component 6 */
_buffer_load_d16_b16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:12 // load one buffer value
/* g2l=4, load component 7 */
_buffer_load_d16_hi_b16 v45, v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v45 // HasEccHalf: pack

s_waitcnt vmcnt(0)                                 // lgkmcnt=-1 vmcnt=02wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //




/* local write a */

_ds_store_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0


/* local write b */

_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0
_ds_store_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:1024 // lwoB_0_0_1_0 = (0*LSCB) + (1*LSPB)(*MT1J+PAD) = 1024


/* Recalc local read offsets */


s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-15wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* local read reset offsets a */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x3fff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red


/* local read reset offsets b */


/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x3fff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red


/* local read init pointers a */


/* localReadInitPointers */


/* local read init pointers b */


/* localReadInitPointers */


/* tail loop: macs */

TailLoopBeginL_6:


/* local read a */

_ds_load_u16 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v45, v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v46, v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+2], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v47, v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuA_X0_I0+3], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v48, v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0


/* local read b */

_ds_load_u16 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v49, v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+1], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v50, v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+2], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v51, v[vgprLocalReadAddrB] offset:640 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+3], v[vgprLocalReadAddrB] offset:1152 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v52, v[vgprLocalReadAddrB] offset:1664 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=1 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+4], v[vgprLocalReadAddrB] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v53, v[vgprLocalReadAddrB] offset:768 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+5], v[vgprLocalReadAddrB] offset:1280 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v54, v[vgprLocalReadAddrB] offset:1792 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=2 rIdx=3 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+6], v[vgprLocalReadAddrB] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=0 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v55, v[vgprLocalReadAddrB] offset:896 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=1 oIdx=0 buffer=0 iui=0
_ds_load_u16 v[vgprValuB_X0_I0+7], v[vgprLocalReadAddrB] offset:1408 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=2 oIdx=0 buffer=0 iui=0
_ds_load_u16_d16_hi v56, v[vgprLocalReadAddrB] offset:1920 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=3 rIdx=3 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s55, 0x800                               // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s55, v[vgprLocalReadAddrA] // lrA += 2048 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s55, 0x1000                              // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s55, v[vgprLocalReadAddrB] // lrB += 4096 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // lgkmcnt=0 vmcnt=-14wait for local read

v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v45 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v46 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0+2], v47 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0+3], v48 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+0], v[vgprValuB_X0_I0+0], v49 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+1], v[vgprValuB_X0_I0+1], v50 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+2], v[vgprValuB_X0_I0+2], v51 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+3], v[vgprValuB_X0_I0+3], v52 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+4], v[vgprValuB_X0_I0+4], v53 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+5], v[vgprValuB_X0_I0+5], v54 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+6], v[vgprValuB_X0_I0+6], v55 // pack two half Vgpr to one Vgpr
v_or_b32 v[vgprValuB_X0_I0+7], v[vgprValuB_X0_I0+7], v56 // pack two half Vgpr to one Vgpr

v_and_b32 v45, 63, v[vgprSerial]                   // v45 = v[vgprSerial] % 64
v_lshrrev_b32 v45, 5, v45                          // v45 = v45 / 32
v_lshlrev_b32 v45, 0x2, v45                        // v45 = v45 * 4
v_cmp_ge_i32 s[60:61], v45, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0], v[vgprValuA_X0_I0+0+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+0], v[vgprValuA_X0_I0+2+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0], v[vgprValuB_X0_I0+0+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0], v[vgprValuB_X0_I0+2+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0], v[vgprValuB_X0_I0+4+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+0], v[vgprValuB_X0_I0+6+0], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+1], v[vgprValuA_X0_I0+0+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+2+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+1], v[vgprValuB_X0_I0+0+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+1], v[vgprValuB_X0_I0+2+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+1], v[vgprValuB_X0_I0+4+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+6+1], v[vgprValuB_X0_I0+6+1], 0x0, s[60:61] // set 0 if K_idx >= sizeL
_v_sub_u32 v45, s[sgprLoopCounterL], v45           // get distance between size and k index
v_cmp_lt_i32 s[60:61], v45, 4                      // set partial 0 if distance less than input per thread
s_and_b32 s62, s[sgprLoopCounterL], 3              // get inputs for edge thread
s_sub_u32 s62, 4, s62                              // use shift to fill 0 for outside element
s_lshl_b32 s62, s62, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[46:47], s62, v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1] // 
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v47, s[60:61] // 
v_lshlrev_b64 v[46:47], s62, v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1] // 
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+0], v[vgprValuA_X0_I0+2+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0+1], v47, s[60:61] // 
v_lshlrev_b64 v[46:47], s62, v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1] // 
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v47, s[60:61] // 
v_lshlrev_b64 v[46:47], s62, v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1] // 
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+0], v[vgprValuB_X0_I0+2+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0+1], v47, s[60:61] // 
v_lshlrev_b64 v[46:47], s62, v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1] // 
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], v47, s[60:61] // 
v_lshlrev_b64 v[46:47], s62, v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1] // 
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+0], v[vgprValuB_X0_I0+6+0+0+0], v46, s[60:61] // 
v_cndmask_b32 v[vgprValuB_X0_I0+6+0+0+1], v[vgprValuB_X0_I0+6+0+0+1], v47, s[60:61] // 
s_nop 1
v_mfma_f32_32x32x8f16 a[0+0:15+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[0:15]
v_mfma_f32_32x32x8f16 a[16+0:31+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], a[16:31]
v_mfma_f32_32x32x8f16 a[32+0:47+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], a[32:47]
v_mfma_f32_32x32x8f16 a[48+0:63+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], a[48:63]
v_mfma_f32_32x32x8f16 a[64+0:79+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[64:79]
v_mfma_f32_32x32x8f16 a[80+0:95+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], a[80:95]
v_mfma_f32_32x32x8f16 a[96+0:111+0], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], a[96:111]
v_mfma_f32_32x32x8f16 a[112+0:127+0], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+6+0+0:vgprValuB_X0_I0+6+0+0+1], a[112:127]


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x8 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x8 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 TailLoopBeginL_6                    // restart LoopL
TailLoopEndL_7:

SkipTailLoopL_8:

Summation_End_15:
/* endSummation: add vgpr [0...42) to pool */
.set NumFullBlocks, UNDEF
.set WgmRemainder1, UNDEF
.set MagicNumberWgmRemainder1, UNDEF
.set ShadowLimitA, UNDEF
.set ShadowLimitB, UNDEF
s_waitcnt lgkmcnt(0) & vmcnt(0)                    // wait for all summation activity

/* Mapping of Acc register -> C Vgpr register */


/* shift vector components d1 */




/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v1, 1, v4                            // v1 = v4 / 2
v_mul_lo_u32 v1, 0x20, v1                          // wave coordination offset 1
v_and_b32 v5, 31, v[vgprSerial]                    // v5 = v[vgprSerial] % 32
_v_add_u32 v1, v5, v1                              // coordination 1 = wave_id1 + tid1
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v5, 1, v4                                // v5 = v4 % 2
v_mul_lo_u32 v5, 0x20, v5                          // wave coordination offset 0
v_and_b32 v0, 63, v[vgprSerial]                    // v0 = v[vgprSerial] % 64
v_lshrrev_b32 v0, 5, v0                            // v0 = v0 / 32
v_lshlrev_b32 v0, 0x2, v0                          // thread0 * continuous_output
_v_add_u32 v0, v5, v0                              // coordination 0 = wave_id0 + tid0
s_mul_i32 s52, 128, s[sgprWorkGroup0]              // wgp0 * MT0
_v_add_u32 v0, s52, v0                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s52, 256, s[sgprWorkGroup1]              // wgp1 * MT1
_v_add_u32 v1, s52, v1                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1
/* Store Remap Local Write address */
v_lshrrev_b32 v5, 7, v[vgprSerial]                 // v5 = v[vgprSerial] / 128
v_and_b32 v4, 127, v[vgprSerial]                   // v4 = v[vgprSerial] % 128
v_mul_lo_u32 v13, 0x20, v5                         // coord1 offset of LDS for each Wave
v_and_b32 v5, 0x1f, v[vgprSerial]                  // coord1 offset of LDS for each thread
_v_add_u32 v5, v13, v5                             // coord1 offset in MacroTile
v_mov_b32 v11, 0x84                                // lds stride = MT0 + PAD
v_mul_lo_u32 v9, v5, v11                           // lds coord1 offset = Col-id* lds stride
v_lshrrev_b32 v10, 6, v4                           // v10 = v4 / 64
v_and_b32 v4, 63, v4                               // v4 = v4 % 64
v_lshrrev_b32 v12, 0x5, v4                         // tid / matrixInstN
v_lshlrev_b32 v12, 0x2, v12                        // lds coord0 offset *= 4 (each thread hold 4 element)
v_mad_u32_u24 v12, 32, v10, v12                    // coord0 += waveCoord0 * wave M shape(blockM*MiM)
_v_add_lshl_u32 v7, v9, v12, 0x1                   // local write C address

/* Store Remap Local Read address */
v_lshrrev_b32 v5, 6, v[vgprSerial]                 // v5 = v[vgprSerial] / 64
v_and_b32 v4, 63, v[vgprSerial]                    // v4 = v[vgprSerial] % 64
v_mul_lo_u32 v13, 0x10, v5                         // coord1 offset of LDS for each Wave
v_lshrrev_b32 v10, 0x5, v4                         // tid / nThreadPerCol
_v_add_u32 v6, v13, v10                            // coord1 offset in MacroTile
v_mul_lo_u32 v9, v6, v11                           // lds coord1 offset = Col-id* lds stride
v_and_b32 v12, 0x1f, v4                            // coord0 offset of LDS for each thread
v_lshlrev_b32 v12, 0x2, v12                        // lds coord0 offset *= gwvw (each thread hold gwvw element)
_v_add_lshl_u32 v8, v9, v12, 0x1                   // local read C address

/* Store Remap global write coord0 and coord1 */
v_lshrrev_b32 v5, 7, v[vgprSerial]                 // v5 = v[vgprSerial] / 128
v_and_b32 v4, 127, v[vgprSerial]                   // v4 = v[vgprSerial] % 128
v_mul_lo_u32 v13, 0x20, v5                         // coord1 offset of global memory for each Wave
v_lshrrev_b32 v5, 6, v4                            // v5 = v4 / 64
v_and_b32 v4, 63, v4                               // v4 = v4 % 64
v_mad_u32_u24 v13, 16, v5, v13                     // waveCoord1 += waveCoord0 * MiN / WaveGroupM
v_lshrrev_b32 v10, 0x5, v4                         // tid / nThreadPerCol
_v_add_u32 v6, v13, v10                            // coord1 offset in MacroTile
s_mul_i32 s52, 0x80, s[sgprWorkGroup0]             // s52 = wg0*MT0
_v_add_co_u32 v4, vcc, s52, v12                    // coord0 = coord0 + wg0 * MT0
s_mul_i32 s53, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
_v_add_co_u32 v5, vcc, s53, v6                     // coord1 = tid1*VW + wg1*MT1

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //StoreRemap Start


/* not-LocalSplitU: global write */

GW_B0_E0_18:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=8 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4); (0,4,0,0:vw4); (0,5,0,0:vw4); (0,6,0,0:vw4); (0,7,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_lshl_u32 v9, v3, v0, 0x1                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+12], acc0 // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1 // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2 // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3 // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+16], acc4 // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+17], acc5 // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+18], acc6 // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+19], acc7 // copy acc to vreg[7]
v_accvgpr_read_b32 v[vgprValuC+20], acc8 // copy acc to vreg[8]
v_accvgpr_read_b32 v[vgprValuC+21], acc9 // copy acc to vreg[9]
v_accvgpr_read_b32 v[vgprValuC+22], acc10 // copy acc to vreg[10]
v_accvgpr_read_b32 v[vgprValuC+23], acc11 // copy acc to vreg[11]
v_accvgpr_read_b32 v[vgprValuC+24], acc12 // copy acc to vreg[12]
v_accvgpr_read_b32 v[vgprValuC+25], acc13 // copy acc to vreg[13]
v_accvgpr_read_b32 v[vgprValuC+26], acc14 // copy acc to vreg[14]
v_accvgpr_read_b32 v[vgprValuC+27], acc15 // copy acc to vreg[15]
v_accvgpr_read_b32 v[vgprValuC+28], acc16 // copy acc to vreg[16]
v_accvgpr_read_b32 v[vgprValuC+29], acc17 // copy acc to vreg[17]
v_accvgpr_read_b32 v[vgprValuC+30], acc18 // copy acc to vreg[18]
v_accvgpr_read_b32 v[vgprValuC+31], acc19 // copy acc to vreg[19]
v_accvgpr_read_b32 v[vgprValuC+32], acc20 // copy acc to vreg[20]
v_accvgpr_read_b32 v[vgprValuC+33], acc21 // copy acc to vreg[21]
v_accvgpr_read_b32 v[vgprValuC+34], acc22 // copy acc to vreg[22]
v_accvgpr_read_b32 v[vgprValuC+35], acc23 // copy acc to vreg[23]
v_accvgpr_read_b32 v[vgprValuC+36], acc24 // copy acc to vreg[24]
v_accvgpr_read_b32 v[vgprValuC+37], acc25 // copy acc to vreg[25]
v_accvgpr_read_b32 v[vgprValuC+38], acc26 // copy acc to vreg[26]
v_accvgpr_read_b32 v[vgprValuC+39], acc27 // copy acc to vreg[27]
v_accvgpr_read_b32 v[vgprValuC+48], acc28 // copy acc to vreg[28]
v_accvgpr_read_b32 v[vgprValuC+49], acc29 // copy acc to vreg[29]
v_accvgpr_read_b32 v[vgprValuC+50], acc30 // copy acc to vreg[30]
v_accvgpr_read_b32 v[vgprValuC+51], acc31 // copy acc to vreg[31]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0), (0, 4, 0, 0), (0, 5, 0, 0), (0, 6, 0, 0), (0, 7, 0, 0)] */

/* apply mask, calc new C and issue writes */
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v13, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
_ds_store_b64 v7, v[12:13], offset:0               // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v17, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
_ds_store_b64 v7, v[16:17], offset:16              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v21, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
_ds_store_b64 v7, v[20:21], offset:32              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+24], v[vgprValuC+24]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+25], v[vgprValuC+25]     // convert C to fp16
v_pack_b32_f16 v24, v[vgprValuC+24], v[vgprValuC+25] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+26], v[vgprValuC+26]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+27], v[vgprValuC+27]     // convert C to fp16
v_pack_b32_f16 v25, v[vgprValuC+26], v[vgprValuC+27] // Pack with neighbor
_ds_store_b64 v7, v[24:25], offset:48              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+28], v[vgprValuC+28]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+29], v[vgprValuC+29]     // convert C to fp16
v_pack_b32_f16 v28, v[vgprValuC+28], v[vgprValuC+29] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+30], v[vgprValuC+30]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+31], v[vgprValuC+31]     // convert C to fp16
v_pack_b32_f16 v29, v[vgprValuC+30], v[vgprValuC+31] // Pack with neighbor
_ds_store_b64 v7, v[28:29], offset:128             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+32], v[vgprValuC+32]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+33], v[vgprValuC+33]     // convert C to fp16
v_pack_b32_f16 v32, v[vgprValuC+32], v[vgprValuC+33] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+34], v[vgprValuC+34]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+35], v[vgprValuC+35]     // convert C to fp16
v_pack_b32_f16 v33, v[vgprValuC+34], v[vgprValuC+35] // Pack with neighbor
_ds_store_b64 v7, v[32:33], offset:144             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+36], v[vgprValuC+36]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+37], v[vgprValuC+37]     // convert C to fp16
v_pack_b32_f16 v36, v[vgprValuC+36], v[vgprValuC+37] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+38], v[vgprValuC+38]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+39], v[vgprValuC+39]     // convert C to fp16
v_pack_b32_f16 v37, v[vgprValuC+38], v[vgprValuC+39] // Pack with neighbor
_ds_store_b64 v7, v[36:37], offset:160             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+48], v[vgprValuC+48]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+49], v[vgprValuC+49]     // convert C to fp16
v_pack_b32_f16 v48, v[vgprValuC+48], v[vgprValuC+49] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+50], v[vgprValuC+50]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+51], v[vgprValuC+51]     // convert C to fp16
v_pack_b32_f16 v49, v[vgprValuC+50], v[vgprValuC+51] // Pack with neighbor
_ds_store_b64 v7, v[48:49], offset:176             // storeRemap lw

/* Handle local read and global write */
s_waitcnt lgkmcnt(0)                               // wait for LDS write
s_barrier //wait all lds write finished

_ds_load_b64 v[12:13], v8, offset:0                // storeRemap lr
_ds_load_b64 v[14:15], v8, offset:528              // storeRemap lr
_ds_load_b64 v[16:17], v8, offset:1056             // storeRemap lr
_ds_load_b64 v[18:19], v8, offset:1584             // storeRemap lr
_ds_load_b64 v[20:21], v8, offset:2112             // storeRemap lr
_ds_load_b64 v[22:23], v8, offset:2640             // storeRemap lr
_ds_load_b64 v[24:25], v8, offset:3168             // storeRemap lr
_ds_load_b64 v[26:27], v8, offset:3696             // storeRemap lr

v_mov_b32 v28, v6                                  // coord1
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(7)                               // wait for LDS read
_buffer_store_b64 v[12:13], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 2                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(6)                               // wait for LDS read
_buffer_store_b64 v[14:15], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 4                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(5)                               // wait for LDS read
_buffer_store_b64 v[16:17], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 6                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(4)                               // wait for LDS read
_buffer_store_b64 v[18:19], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 8                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(3)                               // wait for LDS read
_buffer_store_b64 v[20:21], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 10                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(2)                               // wait for LDS read
_buffer_store_b64 v[22:23], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 12                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(1)                               // wait for LDS read
_buffer_store_b64 v[24:25], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 14                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(0)                               // wait for LDS read
_buffer_store_b64 v[26:27], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_barrier //wait all lds read finished
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (1,0,0,0:vw4); (1,1,0,0:vw4); (1,2,0,0:vw4); (1,3,0,0:vw4); (1,4,0,0:vw4); (1,5,0,0:vw4); (1,6,0,0:vw4); (1,7,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,1,0) */
/* (d1,vc1,d0,vc0)=(1,0,2,0) */
/* (d1,vc1,d0,vc0)=(1,0,3,0) */
/* (d1,vc1,d0,vc0)=(1,0,4,0) */
/* (d1,vc1,d0,vc0)=(1,0,5,0) */
/* (d1,vc1,d0,vc0)=(1,0,6,0) */
/* (d1,vc1,d0,vc0)=(1,0,7,0) */
v_accvgpr_read_b32 v[vgprValuC+12], acc32 // copy acc to vreg[32]
v_accvgpr_read_b32 v[vgprValuC+13], acc33 // copy acc to vreg[33]
v_accvgpr_read_b32 v[vgprValuC+14], acc34 // copy acc to vreg[34]
v_accvgpr_read_b32 v[vgprValuC+15], acc35 // copy acc to vreg[35]
v_accvgpr_read_b32 v[vgprValuC+16], acc36 // copy acc to vreg[36]
v_accvgpr_read_b32 v[vgprValuC+17], acc37 // copy acc to vreg[37]
v_accvgpr_read_b32 v[vgprValuC+18], acc38 // copy acc to vreg[38]
v_accvgpr_read_b32 v[vgprValuC+19], acc39 // copy acc to vreg[39]
v_accvgpr_read_b32 v[vgprValuC+20], acc40 // copy acc to vreg[40]
v_accvgpr_read_b32 v[vgprValuC+21], acc41 // copy acc to vreg[41]
v_accvgpr_read_b32 v[vgprValuC+22], acc42 // copy acc to vreg[42]
v_accvgpr_read_b32 v[vgprValuC+23], acc43 // copy acc to vreg[43]
v_accvgpr_read_b32 v[vgprValuC+24], acc44 // copy acc to vreg[44]
v_accvgpr_read_b32 v[vgprValuC+25], acc45 // copy acc to vreg[45]
v_accvgpr_read_b32 v[vgprValuC+26], acc46 // copy acc to vreg[46]
v_accvgpr_read_b32 v[vgprValuC+27], acc47 // copy acc to vreg[47]
v_accvgpr_read_b32 v[vgprValuC+28], acc48 // copy acc to vreg[48]
v_accvgpr_read_b32 v[vgprValuC+29], acc49 // copy acc to vreg[49]
v_accvgpr_read_b32 v[vgprValuC+30], acc50 // copy acc to vreg[50]
v_accvgpr_read_b32 v[vgprValuC+31], acc51 // copy acc to vreg[51]
v_accvgpr_read_b32 v[vgprValuC+32], acc52 // copy acc to vreg[52]
v_accvgpr_read_b32 v[vgprValuC+33], acc53 // copy acc to vreg[53]
v_accvgpr_read_b32 v[vgprValuC+34], acc54 // copy acc to vreg[54]
v_accvgpr_read_b32 v[vgprValuC+35], acc55 // copy acc to vreg[55]
v_accvgpr_read_b32 v[vgprValuC+36], acc56 // copy acc to vreg[56]
v_accvgpr_read_b32 v[vgprValuC+37], acc57 // copy acc to vreg[57]
v_accvgpr_read_b32 v[vgprValuC+38], acc58 // copy acc to vreg[58]
v_accvgpr_read_b32 v[vgprValuC+39], acc59 // copy acc to vreg[59]
v_accvgpr_read_b32 v[vgprValuC+48], acc60 // copy acc to vreg[60]
v_accvgpr_read_b32 v[vgprValuC+49], acc61 // copy acc to vreg[61]
v_accvgpr_read_b32 v[vgprValuC+50], acc62 // copy acc to vreg[62]
v_accvgpr_read_b32 v[vgprValuC+51], acc63 // copy acc to vreg[63]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0), (1, 5, 0, 0), (1, 6, 0, 0), (1, 7, 0, 0)] */

/* apply mask, calc new C and issue writes */

/* StoreRemap: shift coord1 address */
s_mul_i32 s52, s[sgprStrideD1J], 128               // scale StrideD *= numRows(64) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s52       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
v_mov_b32 v10, 64                                  // set shift rows
_v_add_u32 v5, v5, v10                             // shift storeRemap coord1
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v13, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
_ds_store_b64 v7, v[12:13], offset:0               // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v17, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
_ds_store_b64 v7, v[16:17], offset:16              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v21, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
_ds_store_b64 v7, v[20:21], offset:32              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+24], v[vgprValuC+24]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+25], v[vgprValuC+25]     // convert C to fp16
v_pack_b32_f16 v24, v[vgprValuC+24], v[vgprValuC+25] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+26], v[vgprValuC+26]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+27], v[vgprValuC+27]     // convert C to fp16
v_pack_b32_f16 v25, v[vgprValuC+26], v[vgprValuC+27] // Pack with neighbor
_ds_store_b64 v7, v[24:25], offset:48              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+28], v[vgprValuC+28]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+29], v[vgprValuC+29]     // convert C to fp16
v_pack_b32_f16 v28, v[vgprValuC+28], v[vgprValuC+29] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+30], v[vgprValuC+30]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+31], v[vgprValuC+31]     // convert C to fp16
v_pack_b32_f16 v29, v[vgprValuC+30], v[vgprValuC+31] // Pack with neighbor
_ds_store_b64 v7, v[28:29], offset:128             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+32], v[vgprValuC+32]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+33], v[vgprValuC+33]     // convert C to fp16
v_pack_b32_f16 v32, v[vgprValuC+32], v[vgprValuC+33] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+34], v[vgprValuC+34]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+35], v[vgprValuC+35]     // convert C to fp16
v_pack_b32_f16 v33, v[vgprValuC+34], v[vgprValuC+35] // Pack with neighbor
_ds_store_b64 v7, v[32:33], offset:144             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+36], v[vgprValuC+36]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+37], v[vgprValuC+37]     // convert C to fp16
v_pack_b32_f16 v36, v[vgprValuC+36], v[vgprValuC+37] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+38], v[vgprValuC+38]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+39], v[vgprValuC+39]     // convert C to fp16
v_pack_b32_f16 v37, v[vgprValuC+38], v[vgprValuC+39] // Pack with neighbor
_ds_store_b64 v7, v[36:37], offset:160             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+48], v[vgprValuC+48]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+49], v[vgprValuC+49]     // convert C to fp16
v_pack_b32_f16 v48, v[vgprValuC+48], v[vgprValuC+49] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+50], v[vgprValuC+50]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+51], v[vgprValuC+51]     // convert C to fp16
v_pack_b32_f16 v49, v[vgprValuC+50], v[vgprValuC+51] // Pack with neighbor
_ds_store_b64 v7, v[48:49], offset:176             // storeRemap lw

/* Handle local read and global write */
s_waitcnt lgkmcnt(0)                               // wait for LDS write
s_barrier //wait all lds write finished

_ds_load_b64 v[12:13], v8, offset:0                // storeRemap lr
_ds_load_b64 v[14:15], v8, offset:528              // storeRemap lr
_ds_load_b64 v[16:17], v8, offset:1056             // storeRemap lr
_ds_load_b64 v[18:19], v8, offset:1584             // storeRemap lr
_ds_load_b64 v[20:21], v8, offset:2112             // storeRemap lr
_ds_load_b64 v[22:23], v8, offset:2640             // storeRemap lr
_ds_load_b64 v[24:25], v8, offset:3168             // storeRemap lr
_ds_load_b64 v[26:27], v8, offset:3696             // storeRemap lr

v_mov_b32 v28, v6                                  // coord1
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(7)                               // wait for LDS read
_buffer_store_b64 v[12:13], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 2                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(6)                               // wait for LDS read
_buffer_store_b64 v[14:15], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 4                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(5)                               // wait for LDS read
_buffer_store_b64 v[16:17], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 6                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(4)                               // wait for LDS read
_buffer_store_b64 v[18:19], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 8                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(3)                               // wait for LDS read
_buffer_store_b64 v[20:21], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 10                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(2)                               // wait for LDS read
_buffer_store_b64 v[22:23], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 12                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(1)                               // wait for LDS read
_buffer_store_b64 v[24:25], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 14                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(0)                               // wait for LDS read
_buffer_store_b64 v[26:27], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_barrier //wait all lds read finished
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (2,0,0,0:vw4); (2,1,0,0:vw4); (2,2,0,0:vw4); (2,3,0,0:vw4); (2,4,0,0:vw4); (2,5,0,0:vw4); (2,6,0,0:vw4); (2,7,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,1,0) */
/* (d1,vc1,d0,vc0)=(2,0,2,0) */
/* (d1,vc1,d0,vc0)=(2,0,3,0) */
/* (d1,vc1,d0,vc0)=(2,0,4,0) */
/* (d1,vc1,d0,vc0)=(2,0,5,0) */
/* (d1,vc1,d0,vc0)=(2,0,6,0) */
/* (d1,vc1,d0,vc0)=(2,0,7,0) */
v_accvgpr_read_b32 v[vgprValuC+12], acc64 // copy acc to vreg[64]
v_accvgpr_read_b32 v[vgprValuC+13], acc65 // copy acc to vreg[65]
v_accvgpr_read_b32 v[vgprValuC+14], acc66 // copy acc to vreg[66]
v_accvgpr_read_b32 v[vgprValuC+15], acc67 // copy acc to vreg[67]
v_accvgpr_read_b32 v[vgprValuC+16], acc68 // copy acc to vreg[68]
v_accvgpr_read_b32 v[vgprValuC+17], acc69 // copy acc to vreg[69]
v_accvgpr_read_b32 v[vgprValuC+18], acc70 // copy acc to vreg[70]
v_accvgpr_read_b32 v[vgprValuC+19], acc71 // copy acc to vreg[71]
v_accvgpr_read_b32 v[vgprValuC+20], acc72 // copy acc to vreg[72]
v_accvgpr_read_b32 v[vgprValuC+21], acc73 // copy acc to vreg[73]
v_accvgpr_read_b32 v[vgprValuC+22], acc74 // copy acc to vreg[74]
v_accvgpr_read_b32 v[vgprValuC+23], acc75 // copy acc to vreg[75]
v_accvgpr_read_b32 v[vgprValuC+24], acc76 // copy acc to vreg[76]
v_accvgpr_read_b32 v[vgprValuC+25], acc77 // copy acc to vreg[77]
v_accvgpr_read_b32 v[vgprValuC+26], acc78 // copy acc to vreg[78]
v_accvgpr_read_b32 v[vgprValuC+27], acc79 // copy acc to vreg[79]
v_accvgpr_read_b32 v[vgprValuC+28], acc80 // copy acc to vreg[80]
v_accvgpr_read_b32 v[vgprValuC+29], acc81 // copy acc to vreg[81]
v_accvgpr_read_b32 v[vgprValuC+30], acc82 // copy acc to vreg[82]
v_accvgpr_read_b32 v[vgprValuC+31], acc83 // copy acc to vreg[83]
v_accvgpr_read_b32 v[vgprValuC+32], acc84 // copy acc to vreg[84]
v_accvgpr_read_b32 v[vgprValuC+33], acc85 // copy acc to vreg[85]
v_accvgpr_read_b32 v[vgprValuC+34], acc86 // copy acc to vreg[86]
v_accvgpr_read_b32 v[vgprValuC+35], acc87 // copy acc to vreg[87]
v_accvgpr_read_b32 v[vgprValuC+36], acc88 // copy acc to vreg[88]
v_accvgpr_read_b32 v[vgprValuC+37], acc89 // copy acc to vreg[89]
v_accvgpr_read_b32 v[vgprValuC+38], acc90 // copy acc to vreg[90]
v_accvgpr_read_b32 v[vgprValuC+39], acc91 // copy acc to vreg[91]
v_accvgpr_read_b32 v[vgprValuC+48], acc92 // copy acc to vreg[92]
v_accvgpr_read_b32 v[vgprValuC+49], acc93 // copy acc to vreg[93]
v_accvgpr_read_b32 v[vgprValuC+50], acc94 // copy acc to vreg[94]
v_accvgpr_read_b32 v[vgprValuC+51], acc95 // copy acc to vreg[95]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0), (2, 3, 0, 0), (2, 4, 0, 0), (2, 5, 0, 0), (2, 6, 0, 0), (2, 7, 0, 0)] */

/* apply mask, calc new C and issue writes */

/* StoreRemap: shift coord1 address */
s_mul_i32 s52, s[sgprStrideD1J], 128               // scale StrideD *= numRows(64) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s52       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
v_mov_b32 v10, 64                                  // set shift rows
_v_add_u32 v5, v5, v10                             // shift storeRemap coord1
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v13, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
_ds_store_b64 v7, v[12:13], offset:0               // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v17, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
_ds_store_b64 v7, v[16:17], offset:16              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v21, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
_ds_store_b64 v7, v[20:21], offset:32              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+24], v[vgprValuC+24]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+25], v[vgprValuC+25]     // convert C to fp16
v_pack_b32_f16 v24, v[vgprValuC+24], v[vgprValuC+25] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+26], v[vgprValuC+26]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+27], v[vgprValuC+27]     // convert C to fp16
v_pack_b32_f16 v25, v[vgprValuC+26], v[vgprValuC+27] // Pack with neighbor
_ds_store_b64 v7, v[24:25], offset:48              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+28], v[vgprValuC+28]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+29], v[vgprValuC+29]     // convert C to fp16
v_pack_b32_f16 v28, v[vgprValuC+28], v[vgprValuC+29] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+30], v[vgprValuC+30]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+31], v[vgprValuC+31]     // convert C to fp16
v_pack_b32_f16 v29, v[vgprValuC+30], v[vgprValuC+31] // Pack with neighbor
_ds_store_b64 v7, v[28:29], offset:128             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+32], v[vgprValuC+32]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+33], v[vgprValuC+33]     // convert C to fp16
v_pack_b32_f16 v32, v[vgprValuC+32], v[vgprValuC+33] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+34], v[vgprValuC+34]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+35], v[vgprValuC+35]     // convert C to fp16
v_pack_b32_f16 v33, v[vgprValuC+34], v[vgprValuC+35] // Pack with neighbor
_ds_store_b64 v7, v[32:33], offset:144             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+36], v[vgprValuC+36]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+37], v[vgprValuC+37]     // convert C to fp16
v_pack_b32_f16 v36, v[vgprValuC+36], v[vgprValuC+37] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+38], v[vgprValuC+38]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+39], v[vgprValuC+39]     // convert C to fp16
v_pack_b32_f16 v37, v[vgprValuC+38], v[vgprValuC+39] // Pack with neighbor
_ds_store_b64 v7, v[36:37], offset:160             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+48], v[vgprValuC+48]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+49], v[vgprValuC+49]     // convert C to fp16
v_pack_b32_f16 v48, v[vgprValuC+48], v[vgprValuC+49] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+50], v[vgprValuC+50]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+51], v[vgprValuC+51]     // convert C to fp16
v_pack_b32_f16 v49, v[vgprValuC+50], v[vgprValuC+51] // Pack with neighbor
_ds_store_b64 v7, v[48:49], offset:176             // storeRemap lw

/* Handle local read and global write */
s_waitcnt lgkmcnt(0)                               // wait for LDS write
s_barrier //wait all lds write finished

_ds_load_b64 v[12:13], v8, offset:0                // storeRemap lr
_ds_load_b64 v[14:15], v8, offset:528              // storeRemap lr
_ds_load_b64 v[16:17], v8, offset:1056             // storeRemap lr
_ds_load_b64 v[18:19], v8, offset:1584             // storeRemap lr
_ds_load_b64 v[20:21], v8, offset:2112             // storeRemap lr
_ds_load_b64 v[22:23], v8, offset:2640             // storeRemap lr
_ds_load_b64 v[24:25], v8, offset:3168             // storeRemap lr
_ds_load_b64 v[26:27], v8, offset:3696             // storeRemap lr

v_mov_b32 v28, v6                                  // coord1
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(7)                               // wait for LDS read
_buffer_store_b64 v[12:13], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 2                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(6)                               // wait for LDS read
_buffer_store_b64 v[14:15], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 4                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(5)                               // wait for LDS read
_buffer_store_b64 v[16:17], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 6                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(4)                               // wait for LDS read
_buffer_store_b64 v[18:19], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 8                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(3)                               // wait for LDS read
_buffer_store_b64 v[20:21], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 10                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(2)                               // wait for LDS read
_buffer_store_b64 v[22:23], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 12                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(1)                               // wait for LDS read
_buffer_store_b64 v[24:25], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 14                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(0)                               // wait for LDS read
_buffer_store_b64 v[26:27], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_barrier //wait all lds read finished
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (3,0,0,0:vw4); (3,1,0,0:vw4); (3,2,0,0:vw4); (3,3,0,0:vw4); (3,4,0,0:vw4); (3,5,0,0:vw4); (3,6,0,0:vw4); (3,7,0,0:vw4) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(3,0,0,0) */
/* (d1,vc1,d0,vc0)=(3,0,1,0) */
/* (d1,vc1,d0,vc0)=(3,0,2,0) */
/* (d1,vc1,d0,vc0)=(3,0,3,0) */
/* (d1,vc1,d0,vc0)=(3,0,4,0) */
/* (d1,vc1,d0,vc0)=(3,0,5,0) */
/* (d1,vc1,d0,vc0)=(3,0,6,0) */
/* (d1,vc1,d0,vc0)=(3,0,7,0) */
v_accvgpr_read_b32 v[vgprValuC+12], acc96 // copy acc to vreg[96]
v_accvgpr_read_b32 v[vgprValuC+13], acc97 // copy acc to vreg[97]
v_accvgpr_read_b32 v[vgprValuC+14], acc98 // copy acc to vreg[98]
v_accvgpr_read_b32 v[vgprValuC+15], acc99 // copy acc to vreg[99]
v_accvgpr_read_b32 v[vgprValuC+16], acc100 // copy acc to vreg[100]
v_accvgpr_read_b32 v[vgprValuC+17], acc101 // copy acc to vreg[101]
v_accvgpr_read_b32 v[vgprValuC+18], acc102 // copy acc to vreg[102]
v_accvgpr_read_b32 v[vgprValuC+19], acc103 // copy acc to vreg[103]
v_accvgpr_read_b32 v[vgprValuC+20], acc104 // copy acc to vreg[104]
v_accvgpr_read_b32 v[vgprValuC+21], acc105 // copy acc to vreg[105]
v_accvgpr_read_b32 v[vgprValuC+22], acc106 // copy acc to vreg[106]
v_accvgpr_read_b32 v[vgprValuC+23], acc107 // copy acc to vreg[107]
v_accvgpr_read_b32 v[vgprValuC+24], acc108 // copy acc to vreg[108]
v_accvgpr_read_b32 v[vgprValuC+25], acc109 // copy acc to vreg[109]
v_accvgpr_read_b32 v[vgprValuC+26], acc110 // copy acc to vreg[110]
v_accvgpr_read_b32 v[vgprValuC+27], acc111 // copy acc to vreg[111]
v_accvgpr_read_b32 v[vgprValuC+28], acc112 // copy acc to vreg[112]
v_accvgpr_read_b32 v[vgprValuC+29], acc113 // copy acc to vreg[113]
v_accvgpr_read_b32 v[vgprValuC+30], acc114 // copy acc to vreg[114]
v_accvgpr_read_b32 v[vgprValuC+31], acc115 // copy acc to vreg[115]
v_accvgpr_read_b32 v[vgprValuC+32], acc116 // copy acc to vreg[116]
v_accvgpr_read_b32 v[vgprValuC+33], acc117 // copy acc to vreg[117]
v_accvgpr_read_b32 v[vgprValuC+34], acc118 // copy acc to vreg[118]
v_accvgpr_read_b32 v[vgprValuC+35], acc119 // copy acc to vreg[119]
v_accvgpr_read_b32 v[vgprValuC+36], acc120 // copy acc to vreg[120]
v_accvgpr_read_b32 v[vgprValuC+37], acc121 // copy acc to vreg[121]
v_accvgpr_read_b32 v[vgprValuC+38], acc122 // copy acc to vreg[122]
v_accvgpr_read_b32 v[vgprValuC+39], acc123 // copy acc to vreg[123]
v_accvgpr_read_b32 v[vgprValuC+48], acc124 // copy acc to vreg[124]
v_accvgpr_read_b32 v[vgprValuC+49], acc125 // copy acc to vreg[125]
v_accvgpr_read_b32 v[vgprValuC+50], acc126 // copy acc to vreg[126]
v_accvgpr_read_b32 v[vgprValuC+51], acc127 // copy acc to vreg[127]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(3, 0, 0, 0), (3, 1, 0, 0), (3, 2, 0, 0), (3, 3, 0, 0), (3, 4, 0, 0), (3, 5, 0, 0), (3, 6, 0, 0), (3, 7, 0, 0)] */

/* apply mask, calc new C and issue writes */

/* StoreRemap: shift coord1 address */
s_mul_i32 s52, s[sgprStrideD1J], 128               // scale StrideD *= numRows(64) * bpe
s_add_u32  s[sgprSrdD+0], s[sgprSrdD+0], s52       // incToNextRow: gra SRD += inc(lower)
s_addc_u32  s[sgprSrdD+1], s[sgprSrdD+1], 0        // incToNextRow: gra SRD += inc(upper)
v_mov_b32 v10, 64                                  // set shift rows
_v_add_u32 v5, v5, v10                             // shift storeRemap coord1
v_cvt_f16_f32 v[vgprValuC+12], v[vgprValuC+12]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+13], v[vgprValuC+13]     // convert C to fp16
v_pack_b32_f16 v12, v[vgprValuC+12], v[vgprValuC+13] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v13, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
_ds_store_b64 v7, v[12:13], offset:0               // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v17, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
_ds_store_b64 v7, v[16:17], offset:16              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v21, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
_ds_store_b64 v7, v[20:21], offset:32              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+24], v[vgprValuC+24]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+25], v[vgprValuC+25]     // convert C to fp16
v_pack_b32_f16 v24, v[vgprValuC+24], v[vgprValuC+25] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+26], v[vgprValuC+26]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+27], v[vgprValuC+27]     // convert C to fp16
v_pack_b32_f16 v25, v[vgprValuC+26], v[vgprValuC+27] // Pack with neighbor
_ds_store_b64 v7, v[24:25], offset:48              // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+28], v[vgprValuC+28]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+29], v[vgprValuC+29]     // convert C to fp16
v_pack_b32_f16 v28, v[vgprValuC+28], v[vgprValuC+29] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+30], v[vgprValuC+30]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+31], v[vgprValuC+31]     // convert C to fp16
v_pack_b32_f16 v29, v[vgprValuC+30], v[vgprValuC+31] // Pack with neighbor
_ds_store_b64 v7, v[28:29], offset:128             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+32], v[vgprValuC+32]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+33], v[vgprValuC+33]     // convert C to fp16
v_pack_b32_f16 v32, v[vgprValuC+32], v[vgprValuC+33] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+34], v[vgprValuC+34]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+35], v[vgprValuC+35]     // convert C to fp16
v_pack_b32_f16 v33, v[vgprValuC+34], v[vgprValuC+35] // Pack with neighbor
_ds_store_b64 v7, v[32:33], offset:144             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+36], v[vgprValuC+36]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+37], v[vgprValuC+37]     // convert C to fp16
v_pack_b32_f16 v36, v[vgprValuC+36], v[vgprValuC+37] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+38], v[vgprValuC+38]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+39], v[vgprValuC+39]     // convert C to fp16
v_pack_b32_f16 v37, v[vgprValuC+38], v[vgprValuC+39] // Pack with neighbor
_ds_store_b64 v7, v[36:37], offset:160             // storeRemap lw
v_cvt_f16_f32 v[vgprValuC+48], v[vgprValuC+48]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+49], v[vgprValuC+49]     // convert C to fp16
v_pack_b32_f16 v48, v[vgprValuC+48], v[vgprValuC+49] // Pack with neighbor
v_cvt_f16_f32 v[vgprValuC+50], v[vgprValuC+50]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+51], v[vgprValuC+51]     // convert C to fp16
v_pack_b32_f16 v49, v[vgprValuC+50], v[vgprValuC+51] // Pack with neighbor
_ds_store_b64 v7, v[48:49], offset:176             // storeRemap lw

/* Handle local read and global write */
s_waitcnt lgkmcnt(0)                               // wait for LDS write
s_barrier //wait all lds write finished

_ds_load_b64 v[12:13], v8, offset:0                // storeRemap lr
_ds_load_b64 v[14:15], v8, offset:528              // storeRemap lr
_ds_load_b64 v[16:17], v8, offset:1056             // storeRemap lr
_ds_load_b64 v[18:19], v8, offset:1584             // storeRemap lr
_ds_load_b64 v[20:21], v8, offset:2112             // storeRemap lr
_ds_load_b64 v[22:23], v8, offset:2640             // storeRemap lr
_ds_load_b64 v[24:25], v8, offset:3168             // storeRemap lr
_ds_load_b64 v[26:27], v8, offset:3696             // storeRemap lr

v_mov_b32 v28, v6                                  // coord1
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(7)                               // wait for LDS read
_buffer_store_b64 v[12:13], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 2                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(6)                               // wait for LDS read
_buffer_store_b64 v[14:15], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 4                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(5)                               // wait for LDS read
_buffer_store_b64 v[16:17], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 6                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(4)                               // wait for LDS read
_buffer_store_b64 v[18:19], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 8                              // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(3)                               // wait for LDS read
_buffer_store_b64 v[20:21], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 10                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(2)                               // wait for LDS read
_buffer_store_b64 v[22:23], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 12                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(1)                               // wait for LDS read
_buffer_store_b64 v[24:25], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D
_v_add_u32 v28, v6, 14                             // coord1 += nColPerLoad
v_mul_lo_u32 v28, v28, s[sgprStrideD1J]            // coord1 offset =  coord1 * StrideD
_v_add_lshl_u32 v28, v28, v4, 0x1                  // global write D address
s_waitcnt lgkmcnt(0)                               // wait for LDS read
_buffer_store_b64 v[26:27], v28, s[sgprSrdD:sgprSrdD+3], 0, offen, offset:0 // store D

s_barrier //wait all lds read finished
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_20                           // jump to end
label_GW_End_20:

label_0022:  /// KernelEnd
s_endpgm                                           // Kernel End


