/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
; generated by igemm_codegen.py (2c97a12e9f49260c2ce1b87c12cf7e05adcaf403)
;
.include "igemm_fwd_gtcx35_nhwc_int8_utils.inc"

;----------------------------------------------------------
; starting of kernel igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64
; tensor_layout              : 'nhwc'
; gemm_m_per_block           : 64
; gemm_n_per_block           : 128
; gemm_k_per_block           : 64
; wave_tile_m                : 32
; wave_step_m                : 1
; wave_repeat_m              : 2
; wave_tile_n                : 32
; wave_step_n                : 1
; wave_repeat_n              : 1
; wave_tile_k                : 16
; tensor_a_thread_lengths    : [1, 16, 1, 1]
; tensor_a_cluster_lengths   : [1, 4, 1, 64]
; tensor_b_thread_lengths    : [1, 16, 2, 1]
; tensor_b_cluster_lengths   : [1, 4, 1, 64]
; direction                  : 'fwd'
; precision                  : 'int8'
; nxb                        : 0
; nxe                        : 0
; vector_c                   : 1
; 
; block_size                 : 256
; lds_total                  : 16384
; lds_buffer_num             : 1
; 
.set k_p_in, 0
.set k_p_wei, 8
.set k_p_out, 16
.set k_hi, 24
.set k_wi, 28
.set k_n, 32
.set k_k, 36
.set k_c, 40
.set k_ho, 44
.set k_wo, 48
.set k_stride_h, 52
.set k_stride_w, 56
.set k_dilation_h, 60
.set k_dilation_w, 64
.set k_pad_h, 68
.set k_pad_w, 72
.set k_y, 76
.set k_x, 80
.set k_group, 84
.set k_magic_0, 88
.set k_magic_1, 92
.set k_magic_2, 96
.set k_magic_3, 100
.set k_magic_4, 104
.set k_magic_5, 108
.set k_shift_pack_0, 112
.set k_shift_pack_1, 116
.set k_gemm_k_global_split, 120
.set k__pack_0, 124
.set k_end, 128
.set k_gload_in_c_stride, 16

.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_p_in, 4
.set s_p_wei, 8
.set s_p_out, 12
.set s_hi, 16
.set s_wi, 17
.set s_n, 18
.set s_k, 19
.set s_c, 20
.set s_group, 21
.set s_in_stride_wi, 22
.set s_in_stride_n, 23
.set s_wei_stride_k0, 24
.set s_wei_stride_k, 25
.set s_out_stride_wo, 26
.set s_out_stride_n, 27
.set s_block_gtc_ig, 28
.set s_block_gtc_ik, 29
.set s_block_gtc_inb, 30
.set s_move_slice_k_stride_c, 31
.set s_knum, 3
.set s_dim_br, 32
.set s_dim_mp, 33
.set s_dim_mr, 34
.set s_dim_np, 35
.set s_gemm_k_num_c, 35
.set s_in_diff_hi, 29
.set s_in_diff_wi, 28
.set s_dilation_w_x, 36
.set s_move_slice_k_ix, 32
.set s_flag_need_acc_yx, 33
.set s_kitr, 1
.set s_0xff, 37
.set s_in_offset, 38
.set s_wei_offset, 39
.set s_magic_0, 6
.set s_magic_1, 7
.set s_magic_2, 14
.set s_magic_3, 15
.set s_shift_pack_0, 39
.set s_tmp, 40
.set s_end, 46

.set v_c, 0  ; coalescing:32, needed:0, resuable:33
.set v_a, 0
.set v_b, 8
.set v_gld_a, 12
.set v_gld_b, 16
.set v_sst_a_os, 24
.set v_sld_a_os, 25
.set v_sst_b_os, 26
.set v_sld_b_os, 27
.set v_in_os, 28
.set v_in_ihi_list, 29
.set v_in_iwi_list, 30
.set v_in_flag, 31
.set v_in_flag_n, 32
.set v_wei_os, 33
.set v_out_os, 34
.set v_gtc_ic, 35
.set v_in_inb, 36
.set v_in_in, 37
.set v_wei_ik, 38
.set v_co_sst, 37
.set v_co_sld, 39
.set v_out_flag, 38
.set v_out_inb, 36
.set v_gemm_in, 40
.set v_gemm_im, 41
.set v_co_sub_m_index, 41
.set v_co_sub_n_index, 40
.set v_tmp, 42
.set v_wei_tmp_pack, 11
.set v_wei_flag, 42
.set v_end, 80

.set a_c, 48
.set a_end, 80

.text
.globl igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64
.p2align 8
.type igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64,@function
igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64:
    s_load_dwordx2  s[s_p_in+0:s_p_in+1],    s[s_ka+0:s_ka+1],    0+k_p_in
    s_load_dwordx2  s[s_p_wei+0:s_p_wei+1],   s[s_ka+0:s_ka+1],    0+k_p_wei
    s_load_dwordx2  s[s_p_out+0:s_p_out+1],   s[s_ka+0:s_ka+1],    0+k_p_out
    s_load_dwordx4 s[s_hi+0:s_hi+3],    s[s_ka+0:s_ka+1],    0+k_hi
    s_load_dword s[s_c],    s[s_ka+0:s_ka+1],    0+k_c
    s_load_dword s[s_group],    s[s_ka+0:s_ka+1],    0+k_group
    s_load_dwordx2 s[s_magic_0+0:s_magic_0+1],  s[s_ka+0:s_ka+1],  0+k_magic_0
    s_load_dwordx2 s[s_magic_2+0:s_magic_2+1],  s[s_ka+0:s_ka+1],  0+k_magic_2
    s_load_dword s[s_shift_pack_0], s[s_ka+0:s_ka+1],  0+k_shift_pack_0
    ; in(e, c, nb0, nb1) thread_lengths: 1x16x1x1, cluster_length: 1x4x1x64, k_pack:16
    v_mov_b32 v[v_tmp], v0
    v_and_b32 v[v_gtc_ic], 3, v[v_tmp]
    v_lshlrev_b32 v[v_gtc_ic], 4, v[v_gtc_ic]
    v_lshrrev_b32 v[v_tmp], 2, v[v_tmp]
    v_and_b32 v[v_in_inb], 63, v[v_tmp]
    ; wei(e, c, k0, k1) thread_length: 1x16x2x1, cluster_length: 1x4x1x64, k_pack:16
    v_lshrrev_b32 v[v_tmp], 2, v0
    v_and_b32 v[v_wei_ik], 63, v[v_tmp]

    s_mov_b32 s[s_0xff], 0xff
    s_waitcnt lgkmcnt(0)

    ; calculate index
    s_mul_i32 s[s_in_stride_wi], s[s_c], s[s_group]
    s_mul_i32 s[s_tmp+2], s[s_wi], s[s_in_stride_wi]
    s_mul_i32 s[s_in_stride_n], s[s_hi], s[s_tmp+2]
    s_mov_b32 s[s_wei_stride_k], s[s_c]
    s_lshl_b32 s[s_wei_stride_k0], s[s_wei_stride_k], 6
    s_mul_i32 s[s_out_stride_wo], s[s_k], s[s_group]
    s_mul_i32 s[s_tmp+1], s[s_wi], s[s_out_stride_wo]
    s_mul_i32 s[s_out_stride_n], s[s_hi], s[s_tmp+1]
    s_mul_i32  s[s_tmp], s[s_n], s[s_in_stride_n]
    s_mul_i32  s[s_tmp+1], s[s_n], s[s_out_stride_n]
    s_lshl_b32 s[s_tmp+4], s[s_tmp], 0
    s_lshl_b32 s[s_tmp+5], s[s_tmp+1], 0
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+4]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+4]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]
    s_mul_i32 s[s_tmp], s[s_by], s[s_tmp+5]
    s_mul_hi_u32 s[s_tmp+1], s[s_by], s[s_tmp+5]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]
    s_mov_b32 s[s_knum], s[s_wei_stride_k]
    s_mul_i32 s[s_dim_br], s[s_hi], s[s_wi]
    s_mul_i32 s[s_dim_mr], s[s_n], s[s_dim_br]
    s_add_u32 s[s_tmp], 63, s[s_dim_mr]
    s_lshr_b32 s[s_tmp+1], s[s_tmp], 6
    s_lshl_b32 s[s_dim_mp], s[s_tmp+1], 6
    s_add_u32 s[s_tmp], 127, s[s_k]
    s_lshr_b32 s[s_tmp+1], s[s_tmp], 7
    s_lshl_b32 s[s_dim_np], s[s_tmp+1], 7

    ; gemm_m_per_block:64, gemm_n_per_block:128, source_access_order:0
    s_lshr_b32 s[s_tmp], s[s_dim_mp], 6
    s_lshr_b32 s[s_tmp+1], s[s_dim_np], 7
    s_mul_i32 s[0], s[s_tmp+1], s[s_tmp]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080018 ; offset:24, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_block_gtc_ig,s_bx,s_magic_3,s_tmp+3,0,s_tmp
    s_mov_b32 s[s_bx], s[s_tmp+4]
    s_lshr_b32 s[0], s[s_dim_np], 7
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080000 ; offset:0, width:8
    .mdiv_u32_rem_ss s_tmp+4,s_tmp+5,s_bx,s_magic_0,s_tmp+3,0,s_tmp
    ; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im
    s_lshl_b32 s[s_block_gtc_ik], s[s_tmp+4], 7
    s_lshl_b32 s[s_block_gtc_inb], s[s_tmp+5], 6
    v_add_u32 v[v_tmp+5], s[s_block_gtc_inb], v[v_in_inb]
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080008 ; offset:8, width:8
    .mdiv_u32_rem_vs v_tmp+4,v_in_in,v_tmp+5,s_magic_1,s_tmp+3,s_dim_br,v_tmp
    s_bfe_u32 s[s_tmp+3], s[s_shift_pack_0], 0x00080010 ; offset:16, width:8
    .mdiv_u32_rem_vs v_in_iwi_list,v_in_ihi_list,v_tmp+4,s_magic_2,s_tmp+3,s_wi,v_tmp
    v_cmp_gt_u32 vcc, s[s_n], v[v_in_in]
    v_cndmask_b32 v[v_tmp], 0, 1, vcc
    v_lshlrev_b32 v[v_in_flag_n], 0, v[v_tmp]
    s_lshl_b32 s[s_block_gtc_ig], s[s_block_gtc_ig], 0
    ; calculate wei offset
    s_mul_i32 s[s_tmp+2], s[s_k], s[s_wei_stride_k]
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_tmp+2]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_tmp+2]
    s_add_u32 s[s_p_wei], s[s_p_wei], s[s_tmp]
    s_addc_u32 s[s_p_wei+1], s[s_p_wei+1], s[s_tmp+1]
    v_add_u32 v[v_tmp+5], s[s_block_gtc_ik], v[v_wei_ik]
    v_mul_lo_u32 v[v_tmp], s[s_wei_stride_k], v[v_tmp+5]
    v_add_lshl_u32 v[v_wei_os], v[v_tmp], v[v_gtc_ic], 0
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp+5]
    v_cndmask_b32 v[v_wei_flag], 0, 1, vcc
    v_mov_b32 v[v_wei_tmp_pack], v[v_wei_flag]
    s_mov_b32 s[s_tmp], 64
    v_add_u32 v[v_tmp+5], s[s_tmp], v[v_tmp+5]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp+5]
    v_cndmask_b32 v[v_wei_flag+1], 0, 1, vcc
    v_lshl_or_b32 v[v_wei_tmp_pack], v[v_wei_flag+1], 1, v[v_wei_tmp_pack]

    s_lshl_b32 s[s_wei_stride_k0], s[s_wei_stride_k0], 0

    
    .v_clear_nc v_gld_b, 8
    s_mov_b32 s[s_p_wei+2], 0xffffffff
    s_mov_b32 s[s_p_wei+3], 0x27000
    ; load weight
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag]
    buffer_load_dwordx4 v[v_gld_b:v_gld_b+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    s_mov_b64 exec, -1
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag+1]
    buffer_load_dwordx4 v[v_gld_b+4:v_gld_b+4+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k0] offen offset:0
    s_mov_b64 exec, -1

    ; calculate in offset
    s_mov_b32 s[s_in_offset], 0
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_c]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_c]
    s_add_u32 s[s_p_in], s[s_p_in], s[s_tmp]
    s_addc_u32 s[s_p_in+1], s[s_p_in+1], s[s_tmp+1]

    v_mul_lo_u32 v[v_tmp+1], s[s_in_stride_n], v[v_in_in]
    s_lshl_b32 s[s_in_stride_wi], s[s_in_stride_wi], 0
    v_add_lshl_u32 v[v_tmp+4], v[v_gtc_ic], v[v_tmp+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_wi], v[v_in_ihi_list]
    v_add_u32 v[v_tmp], v[v_in_iwi_list], v[v_tmp]
    v_mul_lo_u32 v[v_tmp], s[s_in_stride_wi], v[v_tmp]
    v_add_u32 v[v_in_os], v[v_tmp+4], v[v_tmp]
    v_bfe_u32 v[v_tmp+1], v[v_in_flag_n],  0, 1
    v_cmp_gt_u32 vcc, s[s_hi], v[v_in_ihi_list]
    v_cndmask_b32 v[v_in_flag], 0, v[v_tmp+1], vcc
    v_cmp_gt_u32 vcc, s[s_wi], v[v_in_iwi_list]
    v_cndmask_b32 v[v_in_flag], 0, v[v_in_flag], vcc

    s_mov_b32 s[s_p_in+2], 0xffffffff
    s_mov_b32 s[s_p_in+3], 0x27000
    ; load input, nxe:0
    .v_clear_nc v_gld_a, 4
    v_cmpx_le_u32 vcc, 1, v[v_in_flag]
    buffer_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset] offen offset:0
    s_mov_b64 exec, -1

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get source matrix gemm index, k_pack:16, v_pack:1, k_pack_per_thread:2
    v_and_b32 v[v_gemm_in], 31, v[v_tmp+5]           ; block_n index 
    v_and_b32 v[v_gemm_im], 31, v[v_tmp+5]           ; block_m index 
    v_lshlrev_b32 v[v_gemm_in], 4, v[v_gemm_in]   ; shift left k_pack:16
    v_lshlrev_b32 v[v_gemm_im], 4, v[v_gemm_im]   ; shift left k_pack:16
    v_lshrrev_b32 v[v_tmp+5], 5, v[v_tmp+5]
    v_and_b32 v[v_tmp + 0], 1, v[v_tmp+5]          ; block_k_per_wave index
    v_lshl_or_b32 v[v_gemm_in],  v[v_tmp + 0], 3, v[v_gemm_in]  ; or lanegroup_k_per_thread:8
    v_lshl_or_b32 v[v_gemm_im],  v[v_tmp + 0], 3, v[v_gemm_im]  ; or lanegroup_k_per_thread:8
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_and_b32 v[v_tmp + 2], 3, v[v_tmp+5]  ; waves_per_n index
    v_lshl_or_b32 v[v_gemm_in], v[v_tmp + 2], 9, v[v_gemm_in]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]

    v_mov_b32 v[v_tmp+5], v0
    ; xdlops mapping, get dst matrix gemm index
    v_and_b32 v[v_tmp+0], 31, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 5, v[v_tmp+5]
    v_and_b32 v[v_tmp+1], 1, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 1, v[v_tmp+5]
    v_mov_b32 v[v_co_sst], v[v_tmp+0]
    v_lshlrev_b32 v[v_co_sld], 2, v[v_tmp+1]
    v_and_b32 v[v_tmp+0], 3, v[v_tmp+5]
    v_lshrrev_b32 v[v_tmp+5], 2, v[v_tmp+5]
    v_lshl_or_b32 v[v_co_sst], v[v_tmp+0], 5, v[v_co_sst]

    ; LDS store, in: e,c,nb0,nb1: 1x16x1x1, 1x4x1x64, k_pack:16, k_pack_gld_a:16, int8
    v_lshlrev_b32 v[v_tmp+2], 4,  v[v_in_inb]
    v_lshrrev_b32 v[v_tmp+1], 4,  v[v_gtc_ic]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+1], 10, v[v_tmp+2]
    v_lshlrev_b32 v[v_sst_a_os], 0, v[v_tmp]

    v_lshlrev_b32 v[v_sld_a_os], 0, v[v_gemm_im] ; LDS load in
    ; LDS store, wei: e,c,k: 1x16x2x1, 1x4x1x64, k_pack:16, k_pack_gld_b:16, int8
    v_lshlrev_b32 v[v_tmp+2], 4,  v[v_wei_ik]
    v_lshrrev_b32 v[v_tmp+1], 4,  v[v_gtc_ic]
    v_lshl_or_b32 v[v_tmp], v[v_tmp+1], 11, v[v_tmp+2]
    v_lshlrev_b32 v[v_sst_b_os], 0, v[v_tmp]
    v_add_u32 v[v_sst_b_os], 4096, v[v_sst_b_os]

    v_lshlrev_b32 v[v_sld_b_os], 0, v[v_gemm_in] ; LDS load wei
    v_add_u32 v[v_sld_b_os], 4096, v[v_sld_b_os]
    v_mov_b32 v[v_gemm_in], v[v_co_sst]
    v_mov_b32 v[v_gemm_im], v[v_co_sld]
    ; init_co_lds_offset for xdlops
    v_lshrrev_b32 v[v_tmp], 2, v[v_gemm_im]
    v_and_b32 v[v_tmp],  1 v[v_tmp]   ; thread id of lanegroup_m_per_cluster
    v_lshlrev_b32 v[v_co_sst], 2, v[v_tmp]
    v_lshl_or_b32 v[v_co_sst], v[v_co_sst], 7, v[v_gemm_in]
    v_lshlrev_b32 v[v_co_sst], 0, v[v_co_sst]
    v_lshlrev_b32 v[v_co_sld], 4, v[0]
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x128 sub_m_index:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:2, l_ms:1, l_mw:1, l_mb:4, l_mt:4 | n_mc:2, n_ml:1, n_mv:1
    ; nd_stride:[4, 2, 1, 4, 1, 1, 1, 1]
    v_lshlrev_b32 v[v_tmp], 4, v[0]
    v_lshrrev_b32 v[v_co_sub_m_index], 7, v[v_tmp]  ; get tid along m
    v_and_b32 v[v_tmp+0], 3, v[v_co_sub_m_index]                   ; => x_mt
    v_lshrrev_b32 v[v_co_sub_m_index], 2  ,v[v_co_sub_m_index]
    v_and_b32 v[v_tmp+1], 1, v[v_co_sub_m_index]                   ; => x_mc
    v_lshrrev_b32 v[v_co_sub_m_index], 1  ,v[v_co_sub_m_index]
    v_and_b32 v[v_tmp+2], 3, v[v_co_sub_m_index]                   ; => x_mb
    v_mov_b32 v[v_co_sub_m_index], v[v_tmp+0]      ; => accumulate x_mt
    v_lshl_or_b32 v[v_co_sub_m_index], v[v_tmp+1], 2, v[v_co_sub_m_index]      ; => accumulate x_mc
    v_lshl_or_b32 v[v_co_sub_m_index], v[v_tmp+2], 3, v[v_co_sub_m_index]      ; => accumulate x_mb
    ; init_co_sub_n_index xdlops
    v_lshlrev_b32 v[v_tmp], 4, v[0]
    v_and_b32 v[v_co_sub_n_index], 127, v[v_tmp]

    v_add_u32 v[v_tmp], s[s_block_gtc_ik], v[v_co_sub_n_index]
    v_cmp_gt_u32 vcc, s[s_k], v[v_tmp]
    v_cndmask_b32 v[v_out_flag], 0, 1, vcc
    ; output offset
    s_mul_i32 s[s_tmp], s[s_block_gtc_ig], s[s_k]
    s_mul_hi_u32 s[s_tmp+1], s[s_block_gtc_ig], s[s_k]
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], s[s_tmp+1]

    s_lshl_b32 s[s_tmp+3], s[s_block_gtc_ik], 0
    s_add_u32 s[s_p_out], s[s_p_out], s[s_tmp+3]
    s_addc_u32 s[s_p_out+1], s[s_p_out+1], 0

    s_lshl_b32 s[s_out_stride_wo], s[s_out_stride_wo], 0
    v_add_u32 v[v_out_inb], s[s_block_gtc_inb], v[v_co_sub_m_index]   ; total n*ho*wo
    v_mul_lo_u32 v[v_out_os], s[s_out_stride_wo], v[v_out_inb]
    v_lshlrev_b32 v[v_tmp], 0, v[v_co_sub_n_index]
    v_add_u32 v[v_out_os], v[v_out_os], v[v_tmp]
    ; move slice stride
    s_lshl_b32 s[s_gemm_k_num_c], s[s_c], 0
    v_bfe_u32 v[v_wei_flag], v[v_wei_tmp_pack], 0, 1
    s_mov_b32 s[s_move_slice_k_stride_c], 64
    v_bfe_u32 v[v_wei_flag+1], v[v_wei_tmp_pack], 1, 1

    s_mov_b32 s[s_p_out+2], 0xffffffff
    s_mov_b32 s[s_p_out+3], 0x27000
    ; start MFMA loop, 32x32 wave tile with 2x1 repeat, 1x1 step, k_pack:16
    s_waitcnt vmcnt(1)
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+0:v_gld_b+0+3] 
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+4:v_gld_b+4+3] offset:1024

    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+3] 

    .v_clear_nc a_c, 32
    ; make sure acc WAR harzard, at least 1 nop for src_c
    s_sub_i32 s[s_kitr], s[s_knum], 64
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_end

    s_add_u32 s[s_in_offset],  s[s_move_slice_k_stride_c], s[s_in_offset]
    v_add_u32 v[v_wei_os], s[s_move_slice_k_stride_c], v[v_wei_os]

    
    s_waitcnt lgkmcnt(0)
    s_barrier
L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_body:
    ; do fma accumulate with unroll 64
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:0
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] offset:0
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:512
    s_waitcnt lgkmcnt(1)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag]
    buffer_load_dwordx4 v[v_gld_b:v_gld_b+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], 0 offen offset:0
    s_mov_b64 exec, -1
    v_cmpx_le_u32 vcc, 1, v[v_wei_flag+1]
    buffer_load_dwordx4 v[v_gld_b+4:v_gld_b+4+3], v[v_wei_os], s[s_p_wei:s_p_wei+3], s[s_wei_stride_k0] offen offset:0
    s_mov_b64 exec, -1
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:2048 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:1536 ; load i_k:1 into local buffer 1, repeat 1
    s_waitcnt lgkmcnt(3)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    .v_clear_nc v_gld_a, 4
    v_cmpx_le_u32 vcc, 1, v[v_in_flag]
    buffer_load_dwordx4 v[v_gld_a:v_gld_a+3], v[v_in_os], s[s_p_in:s_p_in+3], s[s_in_offset] offen offset:0
    s_mov_b64 exec, -1
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:4096 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0
    s_waitcnt lgkmcnt(3)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    s_add_u32 s[s_in_offset],  s[s_move_slice_k_stride_c], s[s_in_offset]
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:2560 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0
    s_waitcnt lgkmcnt(4)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    v_add_u32 v[v_wei_os], s[s_move_slice_k_stride_c], v[v_wei_os]
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:6144 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:3584 ; load i_k:3 into local buffer 1, repeat 1
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt vmcnt(1)
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+0:v_gld_b+0+3]
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    ds_write_b128 v[v_sst_b_os], v[v_gld_b+4:v_gld_b+4+3] offset:1024
    s_barrier
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_sst_a_os], v[v_gld_a+0:v_gld_a+0+3]
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    s_sub_i32 s[s_kitr], s[s_kitr], 64
    s_cmp_gt_i32 s[s_kitr], 0
    s_cbranch_scc0 L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_finishing
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_branch L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_body
L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_finishing:
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16

L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_mfma_end:
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:0
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] offset:0
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:512
    ; k iteration : 0
    s_waitcnt lgkmcnt(1)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:2048 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:1024 ; load i_k:1 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:1536 ; load i_k:1 into local buffer 1, repeat 1

    s_waitcnt lgkmcnt(3)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_b:v_b+1], v[v_sld_b_os] offset:4096 ; load i_k:2 into local buffer 0, repeat 0
    ds_read_b64 v[v_a:v_a+1], v[v_sld_a_os] offset:2048 ; load i_k:2 into local buffer 0, repeat 0

    ; k iteration : 16
    s_waitcnt lgkmcnt(3)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_a+2:v_a+2+1], v[v_sld_a_os] offset:2560 ; load i_k:2 into local buffer 0, repeat 1
    ds_read_b64 v[v_a+4:v_a+4+1], v[v_sld_a_os] offset:3072 ; load i_k:3 into local buffer 1, repeat 0

    s_waitcnt lgkmcnt(4)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16
    ds_read_b64 v[v_b+2:v_b+2+1], v[v_sld_b_os] offset:6144 ; load i_k:3 into local buffer 1, repeat 0
    ds_read_b64 v[v_a+6:v_a+6+1], v[v_sld_a_os] offset:3584 ; load i_k:3 into local buffer 1, repeat 1

    ; k iteration : 32
    s_waitcnt lgkmcnt(4)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+0:v_a+1], v[v_b+0:v_b+1], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16

    s_waitcnt lgkmcnt(3)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+2:v_a+3], v[v_b+0:v_b+1], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16

    ; k iteration : 48
    s_waitcnt lgkmcnt(1)
    v_mfma_i32_32x32x16i8 v[a_c+0:a_c+15], v[v_a+4:v_a+5], v[v_b+2:v_b+3], v[a_c+0:a_c+15]     ; repeat:0x0, step:0x0, num_a_c:16

    s_waitcnt lgkmcnt(0)
    v_mfma_i32_32x32x16i8 v[a_c+16:a_c+31], v[v_a+6:v_a+7], v[v_b+2:v_b+3], v[a_c+16:a_c+31]     ; repeat:1x0, step:0x0, num_a_c:16

    s_nop 9
    ; coalescing store, mapping:mt_m:64, mt_n:128, wt_m:32, wt_n:32, ws:4, r_m:2, r_n:1, s_m:1, s_n:1 | 32x32x16, lanegroup_m_tcbw:4x2x4x1, lanegroup_n_tcbw:1x32x1x1
    ; coalescing_groups:1, num_dword_per_group:32
    ; init_co_sub_m_index xdlops, block_size:256, macro-tile:64x128 sub_m_index:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ; g_mr:1, g_ms:1, g_mw:1, g_mb:1, g_mt:1 | l_mr:2, l_ms:1, l_mw:1, l_mb:4, l_mt:4 | n_mc:2, n_ml:1, n_mv:1
    ; nd_stride:[2, 1, 4, 1, 1, 1, 1]
    ; start group 0, i_g_mr:0, i_g_ms:0, i_g_mw:0, i_g_mb:0, i_g_mt:0, m index start from 0
    s_barrier
    ds_write_b8 v[v_co_sst], v[a_c]  ; idword:0(0,0), 0x0, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+1] offset:128 ; idword:0(0,0), 0x0, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+2] offset:256 ; idword:0(0,0), 0x0, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+3] offset:384 ; idword:0(0,0), 0x0, i_mr:0, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+4] offset:1024 ; idword:1024(8,0), 8x0, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+5] offset:1152 ; idword:1024(8,0), 8x0, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+6] offset:1280 ; idword:1024(8,0), 8x0, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+7] offset:1408 ; idword:1024(8,0), 8x0, i_mr:0, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+8] offset:2048 ; idword:2048(16,0), 16x0, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+9] offset:2176 ; idword:2048(16,0), 16x0, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+10] offset:2304 ; idword:2048(16,0), 16x0, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+11] offset:2432 ; idword:2048(16,0), 16x0, i_mr:0, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+12] offset:3072 ; idword:3072(24,0), 24x0, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+13] offset:3200 ; idword:3072(24,0), 24x0, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+14] offset:3328 ; idword:3072(24,0), 24x0, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+15] offset:3456 ; idword:3072(24,0), 24x0, i_mr:0, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+16] offset:4096 ; idword:4096(32,0), 32x0, i_mr:1, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+17] offset:4224 ; idword:4096(32,0), 32x0, i_mr:1, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+18] offset:4352 ; idword:4096(32,0), 32x0, i_mr:1, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+19] offset:4480 ; idword:4096(32,0), 32x0, i_mr:1, i_ms:0, i_mw:0, i_mb:0  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+20] offset:5120 ; idword:5120(40,0), 40x0, i_mr:1, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+21] offset:5248 ; idword:5120(40,0), 40x0, i_mr:1, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+22] offset:5376 ; idword:5120(40,0), 40x0, i_mr:1, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+23] offset:5504 ; idword:5120(40,0), 40x0, i_mr:1, i_ms:0, i_mw:0, i_mb:1  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+24] offset:6144 ; idword:6144(48,0), 48x0, i_mr:1, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+25] offset:6272 ; idword:6144(48,0), 48x0, i_mr:1, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+26] offset:6400 ; idword:6144(48,0), 48x0, i_mr:1, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+27] offset:6528 ; idword:6144(48,0), 48x0, i_mr:1, i_ms:0, i_mw:0, i_mb:2  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+28] offset:7168 ; idword:7168(56,0), 56x0, i_mr:1, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+29] offset:7296 ; idword:7168(56,0), 56x0, i_mr:1, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+30] offset:7424 ; idword:7168(56,0), 56x0, i_mr:1, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    ds_write_b8 v[v_co_sst], v[a_c+31] offset:7552 ; idword:7168(56,0), 56x0, i_mr:1, i_ms:0, i_mw:0, i_mb:3  x  i_nr:0, i_ns:0, i_nw:0
    s_mov_b32 s[s_tmp], 0   ; i_m:0(i_m0:0,i_m1:0)
    v_add_u32 v[v_out_inb], s[s_block_gtc_inb], v[v_co_sub_m_index]
    v_mov_b32 v[v_tmp], v[v_out_inb]
    s_waitcnt lgkmcnt(0)
    s_barrier
    ;   load from lds, i_ssgroup:0, num_sld_per_ssgroup:2
    ds_read_b128 v[v_c:v_c+3], v[v_co_sld] offset:0
    ds_read_b128 v[v_c+4:v_c+4+3], v[v_co_sld] offset:4096
    v_cmpx_eq_u32 vcc, 1, v[v_out_flag]
    ;   store to global, m index start from 0, m0:0, m1:0
    s_waitcnt lgkmcnt(1)
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dwordx4_m v[v_c:v_c+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mul_i32 s[s_tmp], 32, s[s_out_stride_wo]   ; i_m:32(i_m0:0,i_m1:32)
    v_add_u32 v[v_tmp], 32, v[v_out_inb]
    s_waitcnt lgkmcnt(0)
    v_cmp_gt_u32 vcc, s[s_dim_mr], v[v_tmp]
    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc
    buffer_store_dwordx4_m v[v_c+4:v_c+4+3], v[v_out_os], s[s_p_out:s_p_out+3], s[s_tmp] offen offset:0
    s_or_b64 exec, exec, s[s_tmp+4:s_tmp+5]
    s_mov_b64 exec, -1
L_igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64_out:
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 46
    .amdhsa_ieee_mode 1
    .amdhsa_dx10_clamp 1
    .amdhsa_float_round_mode_32 3
    .amdhsa_float_round_mode_16_64 3
    .amdhsa_tg_split 0
    .amdhsa_accum_offset 48
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64
    .symbol: igemm_fwd_gtcx35_nhwc_int8_bx0_ex0_bt64x128x64_wt32x32x16_ws1x1_wr2x1_ta1x16x1x1_1x4x1x64_tb1x16x2x1_1x4x1x64.kd
    .sgpr_count: 52
    .vgpr_count: 80
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: p_in_     , .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_wei_    , .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: p_out_    , .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: hi_       , .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: wi_       , .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n_        , .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k_        , .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: c_        , .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ho_       , .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: wo_       , .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: stride_h_ , .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: stride_w_ , .size: 4, .offset:  56, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_h_, .size: 4, .offset:  60, .value_kind: by_value, .value_type: i32}
    - { .name: dilation_w_, .size: 4, .offset:  64, .value_kind: by_value, .value_type: i32}
    - { .name: pad_h_    , .size: 4, .offset:  68, .value_kind: by_value, .value_type: i32}
    - { .name: pad_w_    , .size: 4, .offset:  72, .value_kind: by_value, .value_type: i32}
    - { .name: y_        , .size: 4, .offset:  76, .value_kind: by_value, .value_type: i32}
    - { .name: x_        , .size: 4, .offset:  80, .value_kind: by_value, .value_type: i32}
    - { .name: group_    , .size: 4, .offset:  84, .value_kind: by_value, .value_type: i32}
    - { .name: magic_0_  , .size: 4, .offset:  88, .value_kind: by_value, .value_type: i32}
    - { .name: magic_1_  , .size: 4, .offset:  92, .value_kind: by_value, .value_type: i32}
    - { .name: magic_2_  , .size: 4, .offset:  96, .value_kind: by_value, .value_type: i32}
    - { .name: magic_3_  , .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    - { .name: magic_4_  , .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    - { .name: magic_5_  , .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_0_, .size: 4, .offset: 112, .value_kind: by_value, .value_type: i32}
    - { .name: shift_pack_1_, .size: 4, .offset: 116, .value_kind: by_value, .value_type: i32}
    - { .name: gemm_k_split_, .size: 4, .offset: 120, .value_kind: by_value, .value_type: i32}
    - { .name: __pack_0_ , .size: 4, .offset: 124, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
