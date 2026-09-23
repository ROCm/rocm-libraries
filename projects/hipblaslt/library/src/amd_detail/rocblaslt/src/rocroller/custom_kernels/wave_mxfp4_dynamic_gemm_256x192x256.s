.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text

.protected wave_mxfp4_dynamic_gemm_256x192x256
.globl wave_mxfp4_dynamic_gemm_256x192x256
.p2align 8
.type wave_mxfp4_dynamic_gemm_256x192x256,@function
wave_mxfp4_dynamic_gemm_256x192x256:
  ; SRD setup prologue
  s_load_dwordx2 s[2:3], s[0:1], 0
  s_load_dwordx2 s[4:5], s[0:1], 8
  s_load_dwordx2 s[6:7], s[0:1], 16
  s_load_dwordx2 s[8:9], s[0:1], 24
  s_load_dwordx2 s[10:11], s[0:1], 32
  s_load_dwordx2 s[12:13], s[0:1], 40
  s_load_dwordx2 s[14:15], s[0:1], 48
  s_branch .L_wave_mxfp4_dynamic_gemm_256x192x256_main
  .p2align 8
  .L_wave_mxfp4_dynamic_gemm_256x192x256_main:
  s_load_dwordx2 s[40:41], s[0:1], 56
  s_load_dwordx2 s[42:43], s[0:1], 64
  s_load_dwordx2 s[44:45], s[0:1], 72
  s_load_dwordx2 s[46:47], s[0:1], 80
  s_load_dwordx2 s[48:49], s[0:1], 88
  s_load_dwordx2 s[50:51], s[0:1], 96
  s_waitcnt lgkmcnt(0)
  s_mov_b64 s[20:21], s[2:3]
  s_mov_b32 s22, 0x7FFFFFFE
  s_mov_b32 s23, 0x20000
  s_mov_b64 s[24:25], s[4:5]
  s_mov_b32 s26, 0x7FFFFFFE
  s_mov_b32 s27, 0x20000
  s_mov_b64 s[28:29], s[6:7]
  s_mov_b32 s30, 0x7FFFFFFE
  s_mov_b32 s31, 0x20000
  s_mov_b64 s[32:33], s[8:9]
  s_mov_b32 s34, 0x7FFFFFFE
  s_mov_b32 s35, 0x20000
  s_mov_b64 s[36:37], s[10:11]
  s_mov_b32 s38, 0x7FFFFFFC
  s_mov_b32 s39, 0x20000
  s_mov_b32 s52, s12
  s_mov_b32 s53, s14
  s_mov_b32 s54, s40
  s_mov_b32 s55, s42
  s_mov_b32 s56, s44
  s_mov_b32 s57, s46
  s_mov_b32 s58, s48
  s_mov_b32 s59, s50
  ; End SRD setup
  v_and_b32 v253, 1023, v0
  v_bfe_u32 v249, v0, 10, 10
  s_lshr_b32 s2, s54, 1
  s_lshr_b32 s3, s54, 5
  v_lshlrev_b32 v250, 4, v249
  v_lshrrev_b32 v247, 6, v253
  v_lshlrev_b32 v252, 3, v247
  v_lshl_or_b32 v240, v249, 4, v252
  v_lshl_or_b32 v252, v249, 1, v247
  v_lshlrev_b32 v248, 8, 0
  v_sub_u32 v251, v240, v248
  s_nop 0
  v_readfirstlane_b32 s4, v251
  s_mul_hi_u32 s5, s53, 2863311531
  s_lshr_b32 s6, s5, 7
  s_mul_i32 s5, s6, 192
  s_sub_u32 s7, s53, s5
  s_cmp_lg_u32 s7, 0
  s_addc_u32 s5, s6, 0
  s_and_b32 s6, s5, 31
  s_max_i32 s8, s6, 1
  v_cmp_gt_i32 vcc, s6, 0
  v_mov_b32 v240, 1
  v_cndmask_b32 v239, 0, v240, vcc
  s_lshr_b32 s6, s52, 8
  s_and_b32 s9, s52, 255
  s_cmp_lg_u32 s9, 0
  s_addc_u32 s10, s6, 0
  s_mul_i32 s6, s10, s17
  s_add_u32 s11, s16, s6
  s_cmp_lg_u32 s7, 0
  s_lshr_b32 s12, s5, 5
  s_cmp_lg_u32 s9, 0
  s_mul_i32 s13, s12, s10
  s_lshl_b32 s12, s13, 5
  v_mov_b32 v240, s11
  v_cmp_ge_i32 vcc, v240, s12
  v_mov_b32 v240, 1
  v_cndmask_b32 v245, 0, v240, vcc
  v_and_b32 v240, v239, v245
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  s_mul_i32 s12, s13, -32
  s_add_u32 s14, s16, s12
  s_cmp_lg_u32 s9, 0
  s_add_u32 s15, s14, s6
  v_mov_b32 v245, s8
  v_cvt_f32_u32 v246, v245
  v_rcp_f32 v245, v246
  v_mov_b32 v15, 1333788670
  v_mul_f32 v246, v15, v245
  v_cvt_u32_f32 v245, v246
  v_mov_b32 v246, s8
  v_sub_u32 v237, 0, v246
  v_mul_lo_u32 v246, v237, v245
  v_mul_hi_u32 v237, v245, v246
  v_add_u32 v246, v245, v237
  v_mov_b32 v245, s15
  v_mul_hi_u32 v237, v245, v246
  v_mov_b32 v245, s8
  v_mul_lo_u32 v236, v237, v245
  v_mov_b32 v245, s15
  v_sub_u32 v244, v245, v236
  v_mov_b32 v245, s8
  v_subrev_u32 v236, v245, v244
  v_mov_b32 v245, s8
  v_cmp_ge_u32 vcc, v244, v245
  v_mov_b32 v245, 1
  v_cndmask_b32 v238, 0, v245, vcc
  v_add_u32 v245, v237, v238
  v_cndmask_b32 v238, v244, v236
  v_mov_b32 v241, s8
  v_cmp_ge_u32 vcc, v238, v241
  v_mov_b32 v238, 1
  v_cndmask_b32 v241, 0, v238, vcc
  v_add_u32 v238, v245, v241
  s_nop 0
  v_readfirstlane_b32 s14, v238
  s_cmp_lg_u32 s9, 0
  s_lshr_b32 s18, s11, 5
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v245, s10
  v_cvt_f32_u32 v238, v245
  v_rcp_f32 v245, v238
  s_nop 0
  v_mul_f32 v238, v15, v245
  v_cvt_u32_f32 v245, v238
  v_mov_b32 v238, s10
  v_sub_u32 v241, 0, v238
  v_mul_lo_u32 v238, v241, v245
  v_mul_hi_u32 v241, v245, v238
  v_add_u32 v238, v245, v241
  v_mov_b32 v245, s18
  v_mul_hi_u32 v241, v245, v238
  v_mov_b32 v245, s10
  v_mul_lo_u32 v238, v241, v245
  v_mov_b32 v245, s18
  v_sub_u32 v242, v245, v238
  v_mov_b32 v245, s10
  v_subrev_u32 v238, v245, v242
  v_mov_b32 v245, s10
  v_cmp_ge_u32 vcc, v242, v245
  v_mov_b32 v245, 1
  v_cndmask_b32 v243, 0, v245, vcc
  v_add_u32 v245, v241, v243
  v_cndmask_b32 v243, v242, v238
  v_mov_b32 v235, s10
  v_cmp_ge_u32 vcc, v243, v235
  v_mov_b32 v243, 1
  v_cndmask_b32 v235, 0, v243, vcc
  v_add_u32 v243, v245, v235
  s_nop 0
  v_readfirstlane_b32 s19, v243
  s_mul_i32 s40, s19, s10
  s_sub_u32 s19, s18, s40
  v_cmp_ne_u32 vcc, v240, 0
  v_mov_b32 v245, s14
  v_mov_b32 v243, s19
  v_cndmask_b32 v235, v243, v245
  v_lshlrev_b32 v245, 8, v235
  v_lshl_or_b32 v243, v249, 4, v245
  v_lshrrev_b32 v245, 3, v253
  v_or_b32 v233, v243, v245
  v_sub_u32 v243, v233, v248
  v_and_b32 v233, v245, 7
  v_and_b32 v232, v253, 7
  v_xor_b32 v231, v232, v233
  v_lshlrev_b32 v233, 4, v231
  v_mul_lo_u32 v234, v243, s2
  v_add_u32 v230, v234, v233
  s_mul_i32 s14, s52, s54
  s_lshr_b32 s18, s14, 1
  s_and_b32 s21, s21, 0xffff
  s_or_b32 s21, s21, 0x40000000
  s_add_u32 s22, s18, 0
  s_mov_b32 s23, 0x27000
  v_lshlrev_b32 v229, 5, v231
  v_cmp_lt_i32 vcc, v229, s54
  v_mov_b32 v231, 1
  v_cndmask_b32 v228, 0, v231, vcc
  v_cmp_lt_i32 vcc, v243, s52
  v_mov_b32 v231, 1
  v_cndmask_b32 v227, 0, v231, vcc
  v_and_b32 v231, v228, v227
  v_cmp_ne_u32 vcc, v231, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v226, v15, v230
  s_lshl_b32 s40, s4, 7
  s_add_u32 s4, s40, 36864
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v226, s[20:23], 0 offen lds
  v_or_b32 v231, v252, 4
  v_add_u32 v230, v251, 32
  s_nop 0
  v_readfirstlane_b32 s41, v230
  v_add_u32 v230, v243, 32
  v_mul_lo_u32 v226, v230, s2
  v_add_u32 v225, v226, v233
  v_cmp_lt_i32 vcc, v230, s52
  v_mov_b32 v230, 1
  v_cndmask_b32 v224, 0, v230, vcc
  v_and_b32 v230, v228, v224
  v_cmp_ne_u32 vcc, v230, 0
  v_cndmask_b32 v220, v15, v225, vcc
  s_lshl_b32 s42, s41, 7
  s_add_u32 s41, s42, 36864
  s_mov_b32 m0, s41
  buffer_load_dwordx4 v220, s[20:23], 0 offen lds
  v_add_u32 v230, v251, 64
  s_nop 0
  v_readfirstlane_b32 s43, v230
  v_add_u32 v230, v243, 64
  v_mul_lo_u32 v225, v230, s2
  v_add_u32 v220, v225, v233
  v_cmp_lt_i32 vcc, v230, s52
  v_mov_b32 v230, 1
  v_cndmask_b32 v221, 0, v230, vcc
  v_and_b32 v230, v228, v221
  v_cmp_ne_u32 vcc, v230, 0
  v_cndmask_b32 v222, v15, v220, vcc
  s_lshl_b32 s44, s43, 7
  s_add_u32 s43, s44, 36864
  s_mov_b32 m0, s43
  buffer_load_dwordx4 v222, s[20:23], 0 offen lds
  v_mov_b32 v15, 96
  v_add_u32 v230, v251, v15
  s_nop 0
  v_readfirstlane_b32 s45, v230
  v_add_u32 v230, v243, v15
  v_mul_lo_u32 v220, v230, s2
  v_add_u32 v222, v220, v233
  v_cmp_lt_i32 vcc, v230, s52
  v_mov_b32 v230, 1
  v_cndmask_b32 v223, 0, v230, vcc
  v_and_b32 v230, v228, v223
  v_cmp_ne_u32 vcc, v230, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v219, v15, v222
  s_lshl_b32 s46, s45, 7
  s_add_u32 s45, s46, 36864
  s_mov_b32 m0, s45
  buffer_load_dwordx4 v219, s[20:23], 0 offen lds
  v_min_i32 v230, v252, 7
  v_lshlrev_b32 v252, 8, v230
  s_nop 0
  v_readfirstlane_b32 s47, v252
  v_lshlrev_b32 v252, 7, v253
  v_lshlrev_b32 v222, 3, v235
  v_add_u32 v235, v222, v230
  v_mul_lo_u32 v219, v235, s54
  v_lshl_add_u32 v235, v219, 5, v252
  v_lshlrev_b32 v219, 13, v247
  v_sub_u32 v218, v235, v219
  v_mov_b32 v235, s54
  v_cvt_f32_u32 v217, v235
  v_rcp_f32 v235, v217
  s_nop 0
  v_mov_b32 v15, 1333788670
  v_mul_f32 v217, v15, v235
  v_cvt_u32_f32 v235, v217
  v_mov_b32 v217, s54
  v_sub_u32 v216, 0, v217
  v_mul_lo_u32 v217, v216, v235
  v_mul_hi_u32 v216, v235, v217
  v_add_u32 v217, v235, v216
  v_mul_hi_u32 v235, v218, v217
  v_mov_b32 v216, s54
  v_mul_lo_u32 v212, v235, v216
  v_sub_u32 v216, v218, v212
  v_mov_b32 v212, s54
  v_subrev_u32 v213, v212, v216
  v_mov_b32 v212, s54
  v_cmp_ge_u32 vcc, v216, v212
  v_mov_b32 v212, 1
  v_cndmask_b32 v214, 0, v212, vcc
  v_add_u32 v212, v235, v214
  v_cndmask_b32 v214, v216, v213
  v_mov_b32 v215, s54
  v_cmp_ge_u32 vcc, v214, v215
  v_mov_b32 v214, 1
  v_cndmask_b32 v215, 0, v214, vcc
  v_add_u32 v214, v212, v215
  v_mov_b32 v212, s54
  v_cmp_ge_u32 vcc, v216, v212
  v_mov_b32 v212, 1
  v_cndmask_b32 v215, 0, v212, vcc
  v_add_u32 v212, v235, v215
  v_cndmask_b32 v235, v216, v213
  v_mov_b32 v216, s54
  v_cmp_ge_u32 vcc, v235, v216
  v_mov_b32 v235, 1
  v_cndmask_b32 v216, 0, v235, vcc
  v_add_u32 v235, v212, v216
  v_mul_lo_u32 v216, v235, s54
  v_sub_u32 v235, v218, v216
  v_lshrrev_b32 v216, 5, v235
  v_mul_lo_u32 v235, v214, s3
  v_add_u32 v212, v235, v216
  s_lshr_b32 s48, s14, 5
  s_and_b32 s25, s25, 0xffff
  s_or_b32 s25, s25, 0x40000000
  s_add_u32 s26, s48, 0
  s_mov_b32 s27, 0x27000
  s_add_u32 s14, s47, 2048
  s_mov_b32 m0, s14
  buffer_load_dword v212, s[24:27], 0 offen lds
  v_mov_b32 v15, 128
  v_add_u32 v235, v251, v15
  s_nop 0
  v_readfirstlane_b32 s49, v235
  v_add_u32 v235, v243, v15
  v_mul_lo_u32 v216, v235, s2
  v_add_u32 v212, v216, v233
  v_cmp_lt_i32 vcc, v235, s52
  v_mov_b32 v235, 1
  v_cndmask_b32 v213, 0, v235, vcc
  v_and_b32 v235, v228, v213
  v_cmp_ne_u32 vcc, v235, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v214, v15, v212
  s_lshl_b32 s50, s49, 7
  s_add_u32 s49, s50, 36864
  s_mov_b32 m0, s49
  buffer_load_dwordx4 v214, s[20:23], 0 offen lds
  v_mov_b32 v15, 160
  v_add_u32 v235, v251, v15
  s_nop 0
  v_readfirstlane_b32 s51, v235
  v_add_u32 v235, v243, v15
  v_mul_lo_u32 v212, v235, s2
  v_add_u32 v214, v212, v233
  v_cmp_lt_i32 vcc, v235, s52
  v_mov_b32 v235, 1
  v_cndmask_b32 v215, 0, v235, vcc
  v_and_b32 v235, v228, v215
  v_cmp_ne_u32 vcc, v235, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v211, v15, v214
  s_lshl_b32 s64, s51, 7
  s_add_u32 s51, s64, 36864
  s_mov_b32 m0, s51
  buffer_load_dwordx4 v211, s[20:23], 0 offen lds
  v_mov_b32 v15, 192
  v_add_u32 v235, v251, v15
  s_nop 0
  v_readfirstlane_b32 s65, v235
  v_add_u32 v235, v243, v15
  v_mul_lo_u32 v214, v235, s2
  v_add_u32 v211, v214, v233
  v_cmp_lt_i32 vcc, v235, s52
  v_mov_b32 v235, 1
  v_cndmask_b32 v210, 0, v235, vcc
  v_and_b32 v235, v228, v210
  v_cmp_ne_u32 vcc, v235, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v209, v15, v211
  s_lshl_b32 s66, s65, 7
  s_add_u32 s65, s66, 36864
  s_mov_b32 m0, s65
  buffer_load_dwordx4 v209, s[20:23], 0 offen lds
  v_mov_b32 v15, 224
  v_add_u32 v235, v251, v15
  s_nop 0
  v_readfirstlane_b32 s67, v235
  v_add_u32 v251, v243, v15
  v_mul_lo_u32 v243, v251, s2
  v_add_u32 v235, v243, v233
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v211, 0, v251, vcc
  v_and_b32 v251, v228, v211
  v_cmp_ne_u32 vcc, v251, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v228, v15, v235
  s_lshl_b32 s68, s67, 7
  s_add_u32 s67, s68, 36864
  s_mov_b32 m0, s67
  buffer_load_dwordx4 v228, s[20:23], 0 offen lds
  v_min_i32 v251, v231, 7
  v_lshlrev_b32 v235, 8, v251
  s_nop 0
  v_readfirstlane_b32 s69, v235
  v_add_u32 v235, v222, v251
  v_mul_lo_u32 v231, v235, s54
  v_lshl_add_u32 v235, v231, 5, v252
  v_sub_u32 v252, v235, v219
  v_mul_hi_u32 v235, v252, v217
  v_mov_b32 v231, s54
  v_mul_lo_u32 v228, v235, v231
  v_sub_u32 v231, v252, v228
  v_mov_b32 v228, s54
  v_subrev_u32 v222, v228, v231
  v_mov_b32 v228, s54
  v_cmp_ge_u32 vcc, v231, v228
  v_mov_b32 v228, 1
  v_cndmask_b32 v219, 0, v228, vcc
  v_add_u32 v228, v235, v219
  v_cndmask_b32 v219, v231, v222
  v_mov_b32 v209, s54
  v_cmp_ge_u32 vcc, v219, v209
  v_mov_b32 v219, 1
  v_cndmask_b32 v209, 0, v219, vcc
  v_add_u32 v219, v228, v209
  v_mov_b32 v228, s54
  v_cmp_ge_u32 vcc, v231, v228
  v_mov_b32 v228, 1
  v_cndmask_b32 v209, 0, v228, vcc
  v_add_u32 v228, v235, v209
  v_cndmask_b32 v235, v231, v222
  v_mov_b32 v231, s54
  v_cmp_ge_u32 vcc, v235, v231
  v_mov_b32 v235, 1
  v_cndmask_b32 v231, 0, v235, vcc
  v_add_u32 v235, v228, v231
  v_mul_lo_u32 v231, v235, s54
  v_sub_u32 v235, v252, v231
  v_lshrrev_b32 v231, 5, v235
  v_mul_lo_u32 v235, v219, s3
  v_add_u32 v228, v235, v231
  s_add_u32 s70, s69, 2048
  s_mov_b32 m0, s70
  buffer_load_dword v228, s[24:27], 0 offen lds
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s7, 0
  s_and_b32 s71, s5, 4294967264
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v235, s8
  v_cmp_ge_u32 vcc, v244, v235
  v_mov_b32 v235, 1
  v_cndmask_b32 v231, 0, v235, vcc
  v_add_u32 v235, v237, v231
  v_cndmask_b32 v237, v244, v236
  v_mov_b32 v236, s8
  v_cmp_ge_u32 vcc, v237, v236
  v_mov_b32 v237, 1
  v_cndmask_b32 v236, 0, v237, vcc
  v_add_u32 v237, v235, v236
  s_nop 0
  v_readfirstlane_b32 s5, v237
  s_mul_i32 s72, s5, s8
  s_sub_u32 s5, s15, s72
  s_add_u32 s15, s71, s5
  s_cmp_lg_u32 s9, 0
  s_and_b32 s5, s11, 31
  s_cmp_lg_u32 s9, 0
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v237, s10
  v_cmp_ge_u32 vcc, v242, v237
  v_mov_b32 v237, 1
  v_cndmask_b32 v236, 0, v237, vcc
  v_cndmask_b32 v237, v242, v238
  v_mov_b32 v236, s10
  v_cmp_ge_u32 vcc, v237, v236
  v_mov_b32 v237, 1
  v_cndmask_b32 v236, 0, v237, vcc
  s_cmp_lg_u32 s9, 0
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v237, s10
  v_cmp_ge_u32 vcc, v242, v237
  v_mov_b32 v237, 1
  v_cndmask_b32 v236, 0, v237, vcc
  v_add_u32 v237, v241, v236
  v_cndmask_b32 v236, v242, v238
  v_mov_b32 v244, s10
  v_cmp_ge_u32 vcc, v236, v244
  v_mov_b32 v236, 1
  v_cndmask_b32 v244, 0, v236, vcc
  v_add_u32 v236, v237, v244
  s_nop 0
  v_readfirstlane_b32 s10, v236
  v_mov_b32 v237, s10
  v_mov_b32 v236, s5
  v_lshl_or_b32 v244, v237, 5, v236
  v_cmp_ne_u32 vcc, v240, 0
  v_mov_b32 v237, s15
  v_cndmask_b32 v236, v244, v237
  v_mov_b32 v15, 96
  v_mul_lo_u32 v237, v249, v15
  v_mov_b32 v15, 192
  v_mul_lo_u32 v244, v236, v15
  v_add_u32 v238, v237, v244
  v_and_b32 v241, v253, 63
  v_lshrrev_b32 v242, 4, v241
  v_lshlrev_b32 v241, 8, v242
  v_lshl_add_u32 v235, v253, 4, v241
  v_lshrrev_b32 v241, 4, v253
  v_lshlrev_b32 v231, 8, v241
  v_sub_u32 v228, v235, v231
  v_mov_b32 v235, s2
  v_cvt_f32_u32 v231, v235
  v_rcp_f32 v235, v231
  s_nop 0
  v_mov_b32 v15, 1333788670
  v_mul_f32 v231, v15, v235
  v_cvt_u32_f32 v235, v231
  v_mov_b32 v231, s2
  v_sub_u32 v222, 0, v231
  v_mul_lo_u32 v231, v222, v235
  v_mul_hi_u32 v222, v235, v231
  v_add_u32 v231, v235, v222
  v_mul_hi_u32 v235, v228, v231
  v_mov_b32 v222, s2
  v_mul_lo_u32 v219, v235, v222
  v_sub_u32 v222, v228, v219
  v_mov_b32 v219, s2
  v_subrev_u32 v209, v219, v222
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v208, 0, v219, vcc
  v_add_u32 v219, v235, v208
  v_cndmask_b32 v208, v222, v209
  v_mov_b32 v204, s2
  v_cmp_ge_u32 vcc, v208, v204
  v_mov_b32 v208, 1
  v_cndmask_b32 v204, 0, v208, vcc
  v_add_u32 v208, v219, v204
  v_add_u32 v219, v238, v208
  v_mov_b32 v208, s2
  v_cmp_ge_u32 vcc, v222, v208
  v_mov_b32 v208, 1
  v_cndmask_b32 v204, 0, v208, vcc
  v_add_u32 v208, v235, v204
  v_cndmask_b32 v204, v222, v209
  v_mov_b32 v205, s2
  v_cmp_ge_u32 vcc, v204, v205
  v_mov_b32 v204, 1
  v_cndmask_b32 v205, 0, v204, vcc
  v_add_u32 v204, v208, v205
  v_mul_lo_u32 v208, v204, s2
  v_sub_u32 v204, v228, v208
  v_mul_lo_u32 v208, v219, s57
  v_add_u32 v219, v208, v204
  s_mul_i32 s5, s53, s54
  s_lshr_b32 s10, s5, 1
  s_add_u32 s30, s10, 0
  buffer_load_dwordx4 v[200:203], v219, s[28:31], 0 offen
  v_mov_b32 v15, 1024
  v_add_u32 v219, v228, v15
  v_mul_hi_u32 v208, v219, v231
  v_mov_b32 v205, s2
  v_mul_lo_u32 v206, v208, v205
  v_sub_u32 v205, v219, v206
  v_mov_b32 v206, s2
  v_subrev_u32 v207, v206, v205
  v_mov_b32 v206, s2
  v_cmp_ge_u32 vcc, v205, v206
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v208, v159
  v_cndmask_b32 v159, v205, v207
  v_mov_b32 v158, s2
  v_cmp_ge_u32 vcc, v159, v158
  v_mov_b32 v159, 1
  v_cndmask_b32 v158, 0, v159, vcc
  v_add_u32 v159, v206, v158
  v_add_u32 v206, v238, v159
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v205, v159
  v_mov_b32 v159, 1
  v_cndmask_b32 v158, 0, v159, vcc
  v_add_u32 v159, v208, v158
  v_cndmask_b32 v158, v205, v207
  v_mov_b32 v157, s2
  v_cmp_ge_u32 vcc, v158, v157
  v_mov_b32 v158, 1
  v_cndmask_b32 v157, 0, v158, vcc
  v_add_u32 v158, v159, v157
  v_mul_lo_u32 v159, v158, s2
  v_sub_u32 v158, v219, v159
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v158
  buffer_load_dwordx4 v[196:199], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v235, v206
  v_cndmask_b32 v206, v222, v209
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 16
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v204
  buffer_load_dwordx4 v[192:195], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v205, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v208, v206
  v_cndmask_b32 v206, v205, v207
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 16
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v158
  buffer_load_dwordx4 v[188:191], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v235, v206
  v_cndmask_b32 v206, v222, v209
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 32
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v204
  buffer_load_dwordx4 v[184:187], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v205, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v208, v206
  v_cndmask_b32 v206, v205, v207
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 32
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v158
  buffer_load_dwordx4 v[180:183], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v235, v206
  v_cndmask_b32 v206, v222, v209
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 48
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v204
  buffer_load_dwordx4 v[176:179], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v205, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v208, v206
  v_cndmask_b32 v206, v205, v207
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 48
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v158
  buffer_load_dwordx4 v[172:175], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v235, v206
  v_cndmask_b32 v206, v222, v209
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 64
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v204
  buffer_load_dwordx4 v[168:171], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v205, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v208, v206
  v_cndmask_b32 v206, v205, v207
  v_mov_b32 v159, s2
  v_cmp_ge_u32 vcc, v206, v159
  v_mov_b32 v206, 1
  v_cndmask_b32 v159, 0, v206, vcc
  v_add_u32 v206, v219, v159
  v_add_u32 v219, v238, v206
  v_add_u32 v206, v219, 64
  v_mul_lo_u32 v219, v206, s57
  v_add_u32 v206, v219, v158
  buffer_load_dwordx4 v[164:167], v206, s[28:31], 0 offen
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v219, 1
  v_cndmask_b32 v206, 0, v219, vcc
  v_add_u32 v219, v235, v206
  v_cndmask_b32 v235, v222, v209
  v_mov_b32 v222, s2
  v_cmp_ge_u32 vcc, v235, v222
  v_mov_b32 v235, 1
  v_cndmask_b32 v222, 0, v235, vcc
  v_add_u32 v235, v219, v222
  v_add_u32 v222, v238, v235
  v_mov_b32 v15, 80
  v_add_u32 v235, v222, v15
  v_mul_lo_u32 v222, v235, s57
  v_add_u32 v235, v222, v204
  buffer_load_dwordx4 v[160:163], v235, s[28:31], 0 offen
  v_mov_b32 v235, s2
  v_cmp_ge_u32 vcc, v205, v235
  v_mov_b32 v235, 1
  v_cndmask_b32 v222, 0, v235, vcc
  v_add_u32 v235, v208, v222
  v_cndmask_b32 v222, v205, v207
  v_mov_b32 v219, s2
  v_cmp_ge_u32 vcc, v222, v219
  v_mov_b32 v222, 1
  v_cndmask_b32 v219, 0, v222, vcc
  v_add_u32 v222, v235, v219
  v_add_u32 v235, v238, v222
  v_add_u32 v222, v235, v15
  v_mul_lo_u32 v235, v222, s57
  v_add_u32 v222, v235, v158
  buffer_load_dwordx4 v[204:207], v222, s[28:31], 0 offen
  v_lshlrev_b32 v235, 2, v253
  v_lshlrev_b32 v222, 6, v241
  v_sub_u32 v241, v235, v222
  s_add_u32 s10, s3, 7
  s_lshr_b32 s11, s10, 3
  v_mul_lo_u32 v222, v236, 6
  v_mul_lo_u32 v236, v249, 3
  v_add_u32 v249, v222, v236
  v_mul_lo_u32 v236, s11, v249
  v_lshl_add_u32 v222, v236, 8, v241
  v_lshlrev_b32 v236, 6, v242
  v_add_u32 v219, v222, v236
  s_and_b32 s15, s10, 4294967288
  v_mov_b32 v222, s15
  v_cvt_f32_u32 v209, v222
  v_rcp_f32 v222, v209
  s_nop 0
  v_mov_b32 v15, 1333788670
  v_mul_f32 v209, v15, v222
  v_cvt_u32_f32 v222, v209
  v_mov_b32 v209, s15
  v_sub_u32 v208, 0, v209
  v_mul_lo_u32 v209, v208, v222
  v_mul_hi_u32 v208, v222, v209
  v_add_u32 v209, v222, v208
  v_mul_hi_u32 v222, v219, v209
  v_mov_b32 v208, s15
  v_mul_lo_u32 v159, v222, v208
  v_sub_u32 v208, v219, v159
  v_mov_b32 v159, s15
  v_subrev_u32 v158, v159, v208
  v_mov_b32 v159, s15
  v_cmp_ge_u32 vcc, v208, v159
  v_mov_b32 v159, 1
  v_cndmask_b32 v157, 0, v159, vcc
  v_add_u32 v159, v222, v157
  v_cndmask_b32 v222, v208, v158
  v_mov_b32 v208, s15
  v_cmp_ge_u32 vcc, v222, v208
  v_mov_b32 v222, 1
  v_cndmask_b32 v208, 0, v222, vcc
  v_add_u32 v222, v159, v208
  v_add_u32 v208, v241, v236
  v_mul_hi_u32 v159, v208, v209
  v_mov_b32 v158, s15
  v_mul_lo_u32 v157, v159, v158
  v_sub_u32 v158, v208, v157
  v_mov_b32 v157, s15
  v_subrev_u32 v156, v157, v158
  v_mov_b32 v157, s15
  v_cmp_ge_u32 vcc, v158, v157
  v_mov_b32 v157, 1
  v_cndmask_b32 v152, 0, v157, vcc
  v_add_u32 v157, v159, v152
  v_cndmask_b32 v159, v158, v156
  v_mov_b32 v158, s15
  v_cmp_ge_u32 vcc, v159, v158
  v_mov_b32 v159, 1
  v_cndmask_b32 v158, 0, v159, vcc
  v_add_u32 v159, v157, v158
  v_mul_lo_u32 v158, v159, s15
  v_sub_u32 v159, v208, v158
  v_mul_lo_u32 v158, v222, s58
  v_add_u32 v222, v158, v159
  s_lshr_b32 s10, s5, 5
  s_add_u32 s34, s10, 0
  buffer_load_dword v158, v222, s[32:35], 0 offen
  v_add_u32 v222, v249, 1
  v_mul_lo_u32 v157, s11, v222
  v_lshl_add_u32 v222, v157, 8, v241
  v_add_u32 v157, v222, v236
  v_mul_hi_u32 v222, v157, v209
  v_mov_b32 v156, s15
  v_mul_lo_u32 v152, v222, v156
  v_sub_u32 v156, v157, v152
  v_mov_b32 v157, s15
  v_subrev_u32 v152, v157, v156
  v_mov_b32 v157, s15
  v_cmp_ge_u32 vcc, v156, v157
  v_mov_b32 v157, 1
  v_cndmask_b32 v153, 0, v157, vcc
  v_add_u32 v157, v222, v153
  v_cndmask_b32 v222, v156, v152
  v_mov_b32 v156, s15
  v_cmp_ge_u32 vcc, v222, v156
  v_mov_b32 v222, 1
  v_cndmask_b32 v156, 0, v222, vcc
  v_add_u32 v222, v157, v156
  v_mul_lo_u32 v157, v222, s58
  v_add_u32 v222, v157, v159
  buffer_load_dword v157, v222, s[32:35], 0 offen
  v_add_u32 v222, v249, 2
  v_mul_lo_u32 v249, s11, v222
  v_lshl_add_u32 v222, v249, 8, v241
  v_add_u32 v249, v222, v236
  v_mul_hi_u32 v236, v249, v209
  v_mov_b32 v241, s15
  v_mul_lo_u32 v222, v236, v241
  v_sub_u32 v241, v249, v222
  v_mov_b32 v249, s15
  v_subrev_u32 v222, v249, v241
  v_mov_b32 v249, s15
  v_cmp_ge_u32 vcc, v241, v249
  v_mov_b32 v249, 1
  v_cndmask_b32 v156, 0, v249, vcc
  v_add_u32 v249, v236, v156
  v_cndmask_b32 v236, v241, v222
  v_mov_b32 v241, s15
  v_cmp_ge_u32 vcc, v236, v241
  v_mov_b32 v236, 1
  v_cndmask_b32 v241, 0, v236, vcc
  v_add_u32 v236, v249, v241
  v_mul_lo_u32 v249, v236, s58
  v_add_u32 v236, v249, v159
  buffer_load_dword v249, v236, s[32:35], 0 offen
  v_mov_b32 v15, 128
  v_add_u32 v236, v233, v15
  v_add_u32 v241, v234, v236
  v_mov_b32 v15, 256
  v_add_u32 v234, v229, v15
  v_cmp_lt_i32 vcc, v234, s54
  v_mov_b32 v234, 1
  v_cndmask_b32 v229, 0, v234, vcc
  v_and_b32 v234, v229, v227
  v_cmp_ne_u32 vcc, v234, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v227, v15, v241
  s_add_u32 s5, s40, 4096
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v226, v236
  v_and_b32 v234, v229, v224
  v_cmp_ne_u32 vcc, v234, 0
  v_cndmask_b32 v227, v15, v241, vcc
  s_add_u32 s10, s42, 4096
  s_mov_b32 m0, s10
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v225, v236
  v_and_b32 v234, v229, v221
  v_cmp_ne_u32 vcc, v234, 0
  v_cndmask_b32 v227, v15, v241, vcc
  s_add_u32 s11, s44, 4096
  s_mov_b32 m0, s11
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v220, v236
  v_and_b32 v234, v229, v223
  v_cmp_ne_u32 vcc, v234, 0
  v_cndmask_b32 v227, v15, v241, vcc
  s_add_u32 s40, s46, 4096
  s_mov_b32 m0, s40
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_mov_b32 v15, 8192
  v_add_u32 v241, v218, v15
  v_mul_hi_u32 v234, v241, v217
  v_mov_b32 v227, s54
  v_mul_lo_u32 v226, v234, v227
  v_sub_u32 v227, v241, v226
  v_mov_b32 v226, s54
  v_subrev_u32 v225, v226, v227
  v_mov_b32 v226, s54
  v_cmp_ge_u32 vcc, v227, v226
  v_mov_b32 v226, 1
  v_cndmask_b32 v224, 0, v226, vcc
  v_add_u32 v226, v234, v224
  v_cndmask_b32 v224, v227, v225
  v_mov_b32 v220, s54
  v_cmp_ge_u32 vcc, v224, v220
  v_mov_b32 v224, 1
  v_cndmask_b32 v220, 0, v224, vcc
  v_add_u32 v224, v226, v220
  v_mov_b32 v226, s54
  v_cmp_ge_u32 vcc, v227, v226
  v_mov_b32 v226, 1
  v_cndmask_b32 v220, 0, v226, vcc
  v_add_u32 v226, v234, v220
  v_cndmask_b32 v234, v227, v225
  v_mov_b32 v227, s54
  v_cmp_ge_u32 vcc, v234, v227
  v_mov_b32 v234, 1
  v_cndmask_b32 v227, 0, v234, vcc
  v_add_u32 v234, v226, v227
  v_mul_lo_u32 v227, v234, s54
  v_sub_u32 v234, v241, v227
  v_lshrrev_b32 v241, 5, v234
  v_mul_lo_u32 v234, v224, s3
  v_add_u32 v227, v234, v241
  s_mov_b32 m0, s47
  buffer_load_dword v227, s[24:27], 0 offen lds
  v_add_u32 v241, v216, v236
  v_and_b32 v234, v229, v213
  v_cmp_ne_u32 vcc, v234, 0
  v_mov_b32 v15, 2147483647
  v_cndmask_b32 v227, v15, v241
  s_add_u32 s42, s50, 4096
  s_mov_b32 m0, s42
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v212, v236
  v_and_b32 v234, v229, v215
  v_cmp_ne_u32 vcc, v234, 0
  v_cndmask_b32 v227, v15, v241, vcc
  s_add_u32 s44, s64, 4096
  s_mov_b32 m0, s44
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v214, v236
  v_and_b32 v234, v229, v210
  v_cmp_ne_u32 vcc, v234, 0
  v_cndmask_b32 v227, v15, v241, vcc
  s_add_u32 s46, s66, 4096
  s_mov_b32 m0, s46
  buffer_load_dwordx4 v227, s[20:23], 0 offen lds
  v_add_u32 v241, v243, v236
  v_and_b32 v236, v229, v211
  v_cmp_ne_u32 vcc, v236, 0
  v_cndmask_b32 v243, v15, v241, vcc
  s_add_u32 s50, s68, 4096
  s_mov_b32 m0, s50
  buffer_load_dwordx4 v243, s[20:23], 0 offen lds
  v_mov_b32 v15, 8192
  v_add_u32 v236, v252, v15
  v_mul_hi_u32 v252, v236, v217
  v_mov_b32 v241, s54
  v_mul_lo_u32 v243, v252, v241
  v_sub_u32 v241, v236, v243
  v_mov_b32 v243, s54
  v_subrev_u32 v234, v243, v241
  v_mov_b32 v243, s54
  v_cmp_ge_u32 vcc, v241, v243
  v_mov_b32 v243, 1
  v_cndmask_b32 v229, 0, v243, vcc
  v_add_u32 v243, v252, v229
  v_cndmask_b32 v229, v241, v234
  v_mov_b32 v227, s54
  v_cmp_ge_u32 vcc, v229, v227
  v_mov_b32 v229, 1
  v_cndmask_b32 v227, 0, v229, vcc
  v_add_u32 v229, v243, v227
  v_mov_b32 v243, s54
  v_cmp_ge_u32 vcc, v241, v243
  v_mov_b32 v243, 1
  v_cndmask_b32 v227, 0, v243, vcc
  v_add_u32 v243, v252, v227
  v_cndmask_b32 v252, v241, v234
  v_mov_b32 v241, s54
  v_cmp_ge_u32 vcc, v252, v241
  v_mov_b32 v252, 1
  v_cndmask_b32 v241, 0, v252, vcc
  v_add_u32 v252, v243, v241
  v_mul_lo_u32 v241, v252, s54
  v_sub_u32 v252, v236, v241
  v_lshrrev_b32 v236, 5, v252
  v_mul_lo_u32 v252, v229, s3
  v_add_u32 v241, v252, v236
  s_mov_b32 m0, s69
  buffer_load_dword v241, s[24:27], 0 offen lds
  s_waitcnt vmcnt(30)
  s_barrier
  ; sched_barrier mask=0
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  v_cmp_le_i32 vcc, s13, 0
  v_mov_b32 v252, 1
  v_cndmask_b32 v236, 0, v252, vcc
  v_and_b32 v252, v236, v239
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v239, s12
  v_mul_hi_u32 v236, v239, v246
  v_mov_b32 v246, s8
  v_mul_lo_u32 v241, v236, v246
  v_sub_u32 v246, v239, v241
  v_mov_b32 v239, s8
  v_subrev_u32 v241, v239, v246
  v_mov_b32 v239, s8
  v_cmp_ge_u32 vcc, v246, v239
  v_mov_b32 v239, 1
  v_cndmask_b32 v243, 0, v239, vcc
  v_add_u32 v239, v236, v243
  v_cndmask_b32 v243, v246, v241
  v_mov_b32 v234, s8
  v_cmp_ge_u32 vcc, v243, v234
  v_mov_b32 v243, 1
  v_cndmask_b32 v234, 0, v243, vcc
  v_add_u32 v243, v239, v234
  s_nop 0
  v_readfirstlane_b32 s64, v243
  v_cmp_ne_u32 vcc, v252, 0
  v_mov_b32 v239, s64
  v_cndmask_b32 v243, 0, v239, vcc
  v_lshl_or_b32 v239, v243, 8, v253
  v_and_b32 v234, 4294967280, v253
  v_sub_u32 v229, v239, v234
  v_lshlrev_b32 v239, 7, v247
  v_add_u32 v227, v229, v239
  v_xor_b32 v229, v242, v232
  v_lshlrev_b32 v226, 4, v229
  v_lshlrev_b32 v229, 7, v227
  v_add_u32 v227, v229, v226
  ds_read_b128 v[220:223], v227 offset:36864
  v_or_b32 v226, v242, 4
  v_xor_b32 v225, v226, v232
  v_lshlrev_b32 v232, 4, v225
  v_add_u32 v226, v229, v232
  ds_read_b128 v[212:215], v226 offset:36864
  ds_read_b128 v[152:155], v227 offset:38912
  ds_read_b128 v[148:151], v226 offset:38912
  ds_read_b128 v[144:147], v227 offset:40960
  ds_read_b128 v[140:143], v226 offset:40960
  ds_read_b128 v[136:139], v227 offset:43008
  ds_read_b128 v[132:135], v226 offset:43008
  v_lshl_or_b32 v232, v243, 11, v235
  v_mov_b32 v15, 768
  v_mul_lo_u32 v229, v247, v15
  v_add_u32 v225, v232, v229
  ds_read_b32 v232, v225 offset:2048
  ds_read_b32 v229, v225 offset:2304
  s_lshr_b32 s64, s54, 8
  s_and_b32 s66, s54, 255
  s_cmp_lg_u32 s66, 0
  s_addc_u32 s66, s64, 0
  v_mov_b32 v15, 2048
  v_add_u32 v224, v228, v15
  v_mul_hi_u32 v228, v224, v231
  v_mov_b32 v231, s2
  v_mul_lo_u32 v218, v228, v231
  v_sub_u32 v231, v224, v218
  v_mov_b32 v218, s2
  v_subrev_u32 v217, v218, v231
  v_mov_b32 v218, s2
  v_cmp_ge_u32 vcc, v231, v218
  v_mov_b32 v218, 1
  v_cndmask_b32 v216, 0, v218, vcc
  v_add_u32 v218, v228, v216
  v_cndmask_b32 v216, v231, v217
  v_mov_b32 v211, s2
  v_cmp_ge_u32 vcc, v216, v211
  v_mov_b32 v216, 1
  v_cndmask_b32 v211, 0, v216, vcc
  v_add_u32 v216, v218, v211
  v_add_u32 v218, v238, v216
  v_mul_lo_u32 v238, v218, s2
  v_mov_b32 v218, s2
  v_cmp_ge_u32 vcc, v231, v218
  v_mov_b32 v218, 1
  v_cndmask_b32 v216, 0, v218, vcc
  v_add_u32 v218, v228, v216
  v_cndmask_b32 v228, v231, v217
  v_mov_b32 v231, s2
  v_cmp_ge_u32 vcc, v228, v231
  v_mov_b32 v231, 1
  v_cndmask_b32 v228, 0, v231, vcc
  v_add_u32 v231, v218, v228
  v_mul_lo_u32 v228, v231, s2
  v_sub_u32 v231, v224, v228
  v_add_u32 v228, v238, v231
  v_mov_b32 v15, 256
  v_add_u32 v238, v219, v15
  v_mul_hi_u32 v231, v238, v209
  v_mov_b32 v224, s15
  v_mul_lo_u32 v218, v231, v224
  v_sub_u32 v224, v238, v218
  v_mov_b32 v238, s15
  v_subrev_u32 v218, v238, v224
  v_mov_b32 v238, s15
  v_cmp_ge_u32 vcc, v224, v238
  v_mov_b32 v238, 1
  v_cndmask_b32 v217, 0, v238, vcc
  v_add_u32 v238, v231, v217
  v_cndmask_b32 v231, v224, v218
  v_mov_b32 v224, s15
  v_cmp_ge_u32 vcc, v231, v224
  v_mov_b32 v231, 1
  v_cndmask_b32 v224, 0, v231, vcc
  v_add_u32 v231, v238, v224
  v_mul_lo_u32 v238, v231, s3
  v_add_u32 v231, v208, v15
  v_mul_hi_u32 v224, v231, v209
  v_mov_b32 v218, s15
  v_mul_lo_u32 v217, v224, v218
  v_sub_u32 v218, v231, v217
  v_mov_b32 v217, s15
  v_subrev_u32 v216, v217, v218
  v_mov_b32 v217, s15
  v_cmp_ge_u32 vcc, v218, v217
  v_mov_b32 v217, 1
  v_cndmask_b32 v211, 0, v217, vcc
  v_add_u32 v217, v224, v211
  v_cndmask_b32 v224, v218, v216
  v_mov_b32 v218, s15
  v_cmp_ge_u32 vcc, v224, v218
  v_mov_b32 v224, 1
  v_cndmask_b32 v218, 0, v224, vcc
  v_add_u32 v224, v217, v218
  v_mul_lo_u32 v218, v224, s15
  v_sub_u32 v224, v231, v218
  v_add_u32 v231, v238, v224
  s_lshl_b32 s64, s54, 3
  v_add_u32 v238, v228, s64
  s_add_u32 s68, s64, 1024
  v_add_u32 v224, v228, s68
  s_lshl_b32 s68, s54, 4
  v_add_u32 v218, v228, s68
  s_add_u32 s72, s68, 1024
  v_add_u32 v217, v228, s72
  s_mul_i32 s72, s54, 24
  v_add_u32 v216, v228, s72
  s_add_u32 s73, s72, 1024
  v_add_u32 v211, v228, s73
  s_lshl_b32 s73, s54, 5
  v_add_u32 v210, v228, s73
  s_add_u32 s74, s73, 1024
  v_add_u32 v159, v228, s74
  s_mul_i32 s74, s54, 40
  v_add_u32 v156, v228, s74
  s_add_u32 s75, s74, 1024
  v_add_u32 v131, v228, s75
  v_add_u32 v130, v231, s54
  s_lshl_b32 s75, s54, 1
  v_add_u32 v129, v231, s75
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  s_mul_i32 s76, s13, s8
  s_mul_i32 s13, s76, -32
  s_cmp_lg_u32 s9, 0
  s_mul_i32 s76, s6, s8
  s_add_u32 s6, s13, s76
  s_mul_i32 s13, s6, s8
  s_mul_i32 s6, s8, s8
  s_mul_i32 s76, s16, s6
  s_add_u32 s77, s13, s76
  s_mul_i32 s13, s6, s8
  v_mov_b32 v127, s13
  v_cvt_f32_u32 v126, v127
  v_rcp_f32 v127, v126
  s_nop 0
  v_mov_b32 v15, 1333788670
  v_mul_f32 v126, v15, v127
  v_cvt_u32_f32 v127, v126
  v_mov_b32 v126, s13
  v_sub_u32 v128, 0, v126
  v_mul_lo_u32 v126, v128, v127
  v_mul_hi_u32 v128, v127, v126
  v_add_u32 v126, v127, v128
  v_mov_b32 v127, s77
  v_mul_hi_u32 v128, v127, v126
  v_mov_b32 v127, s13
  v_mul_lo_u32 v126, v128, v127
  v_mov_b32 v127, s77
  v_sub_u32 v125, v127, v126
  v_mov_b32 v127, s13
  v_subrev_u32 v126, v127, v125
  v_mov_b32 v127, s13
  v_cmp_ge_u32 vcc, v125, v127
  v_mov_b32 v127, 1
  v_cndmask_b32 v124, 0, v127, vcc
  v_add_u32 v127, v128, v124
  v_cndmask_b32 v128, v125, v126
  v_mov_b32 v126, s13
  v_cmp_ge_u32 vcc, v128, v126
  v_mov_b32 v126, 1
  v_cndmask_b32 v128, 0, v126, vcc
  v_add_u32 v126, v127, v128
  s_nop 0
  v_readfirstlane_b32 s6, v126
  v_cmp_ne_u32 vcc, v240, 0
  v_mov_b32 v127, s6
  v_mov_b32 v126, s19
  v_cndmask_b32 v128, v126, v127
  v_lshl_or_b32 v240, v128, 8, v250
  v_or_b32 v250, v240, v245
  v_sub_u32 v240, v250, v248
  v_mul_lo_u32 v250, v240, s2
  v_mov_b32 v15, 256
  v_add_u32 v248, v233, v15
  v_add_u32 v245, v250, v248
  v_add_u32 v127, v240, 32
  v_mul_lo_u32 v126, v127, s2
  v_add_u32 v127, v126, v248
  v_add_u32 v125, v240, 64
  v_mul_lo_u32 v124, v125, s2
  v_add_u32 v125, v124, v248
  v_mov_b32 v15, 96
  v_add_u32 v123, v240, v15
  v_mul_lo_u32 v122, v123, s2
  v_add_u32 v123, v122, v248
  v_mul_lo_u32 v121, v128, s54
  v_lshlrev_b32 v120, 3, v121
  v_mul_lo_u32 v121, s54, v230
  v_add_u32 v230, v120, v121
  v_add_u32 v121, v230, v235
  v_lshlrev_b32 v230, 8, v247
  v_sub_u32 v247, v121, v230
  v_mov_b32 v15, 512
  v_add_u32 v121, v247, v15
  v_mov_b32 v15, 128
  v_add_u32 v119, v240, v15
  v_mul_lo_u32 v118, v119, s2
  v_add_u32 v119, v118, v248
  v_mov_b32 v15, 160
  v_add_u32 v117, v240, v15
  v_mul_lo_u32 v116, v117, s2
  v_add_u32 v117, v116, v248
  v_mov_b32 v15, 192
  v_add_u32 v115, v240, v15
  v_mul_lo_u32 v114, v115, s2
  v_add_u32 v115, v114, v248
  v_mov_b32 v15, 224
  v_add_u32 v113, v240, v15
  v_mul_lo_u32 v240, v113, s2
  v_add_u32 v113, v240, v248
  v_mul_lo_u32 v248, s54, v251
  v_add_u32 v251, v120, v248
  v_add_u32 v248, v251, v235
  v_sub_u32 v251, v248, v230
  v_mov_b32 v15, 512
  v_add_u32 v248, v251, v15
  v_add_u32 v235, v219, v15
  v_mul_hi_u32 v230, v235, v209
  v_mov_b32 v219, s15
  v_mul_lo_u32 v120, v230, v219
  v_sub_u32 v219, v235, v120
  v_mov_b32 v235, s15
  v_subrev_u32 v120, v235, v219
  v_mov_b32 v235, s15
  v_cmp_ge_u32 vcc, v219, v235
  v_mov_b32 v235, 1
  v_cndmask_b32 v112, 0, v235, vcc
  v_add_u32 v235, v230, v112
  v_cndmask_b32 v230, v219, v120
  v_mov_b32 v219, s15
  v_cmp_ge_u32 vcc, v230, v219
  v_mov_b32 v230, 1
  v_cndmask_b32 v219, 0, v230, vcc
  v_add_u32 v230, v235, v219
  v_mul_lo_u32 v235, v230, s3
  v_add_u32 v230, v208, v15
  v_mul_hi_u32 v219, v230, v209
  v_mov_b32 v209, s15
  v_mul_lo_u32 v208, v219, v209
  v_sub_u32 v209, v230, v208
  v_mov_b32 v208, s15
  v_subrev_u32 v120, v208, v209
  v_mov_b32 v208, s15
  v_cmp_ge_u32 vcc, v209, v208
  v_mov_b32 v208, 1
  v_cndmask_b32 v112, 0, v208, vcc
  v_add_u32 v208, v219, v112
  v_cndmask_b32 v219, v209, v120
  v_mov_b32 v209, s15
  v_cmp_ge_u32 vcc, v219, v209
  v_mov_b32 v219, 1
  v_cndmask_b32 v209, 0, v219, vcc
  v_add_u32 v219, v208, v209
  v_mul_lo_u32 v209, v219, s15
  v_sub_u32 v219, v230, v209
  v_add_u32 v230, v235, v219
  s_add_u32 s2, s64, 2048
  v_add_u32 v235, v228, s2
  s_add_u32 s2, s64, 3072
  v_add_u32 v219, v228, s2
  s_add_u32 s2, s68, 2048
  v_add_u32 v209, v228, s2
  s_add_u32 s2, s68, 3072
  v_add_u32 v208, v228, s2
  s_add_u32 s2, s72, 2048
  v_add_u32 v120, v228, s2
  s_add_u32 s2, s72, 3072
  v_add_u32 v112, v228, s2
  s_add_u32 s2, s73, 2048
  v_add_u32 v111, v228, s2
  s_add_u32 s2, s73, 3072
  v_add_u32 v110, v228, s2
  s_add_u32 s2, s74, 2048
  v_add_u32 v109, v228, s2
  s_add_u32 s2, s74, 3072
  v_add_u32 v108, v228, s2
  v_add_u32 v14, v230, s54
  v_add_u32 v13, v230, s75
  v_mov_b32 v15, 384
  v_add_u32 v12, v233, v15
  v_add_u32 v233, v250, v12
  v_add_u32 v250, v126, v12
  v_add_u32 v126, v124, v12
  v_add_u32 v124, v122, v12
  v_mov_b32 v15, 768
  v_add_u32 v122, v247, v15
  v_add_u32 v247, v118, v12
  v_add_u32 v118, v116, v12
  v_add_u32 v116, v114, v12
  v_add_u32 v114, v240, v12
  v_add_u32 v240, v251, v15
  v_accvgpr_write_b32 a0, 0
  v_accvgpr_write_b32 a1, 0
  v_accvgpr_write_b32 a2, 0
  v_accvgpr_write_b32 a3, 0
  s_mov_b32 s2, 0
  s_mul_i32 s3, s2, 2048
  s_mov_b32 s2, 0
  s_mul_i32 s6, s2, 256
  v_add_u32 v251, v238, s3
  v_add_u32 v238, v224, s3
  v_add_u32 v224, v218, s3
  v_add_u32 v218, v217, s3
  v_add_u32 v217, v216, s3
  v_add_u32 v216, v211, s3
  v_add_u32 v211, v210, s3
  v_add_u32 v210, v159, s3
  v_add_u32 v159, v156, s3
  v_add_u32 v156, v131, s3
  v_add_u32 v131, v130, s6
  v_add_u32 v130, v129, s6
  s_mov_b32 s2, 0
  s_mul_i32 s13, s2, 128
  v_add_u32 v129, v127, s13
  v_add_u32 v127, v125, s13
  v_add_u32 v125, v123, s13
  v_add_u32 v123, v119, s13
  v_add_u32 v119, v117, s13
  v_add_u32 v117, v115, s13
  v_add_u32 v115, v113, s13
  v_add_u32 v113, v248, s6
  v_add_u32 v248, v230, s6
  v_add_u32 v230, v235, s3
  v_add_u32 v235, v219, s3
  v_add_u32 v219, v209, s3
  v_add_u32 v209, v208, s3
  v_add_u32 v208, v120, s3
  v_add_u32 v120, v112, s3
  v_add_u32 v112, v111, s3
  v_add_u32 v111, v110, s3
  v_add_u32 v110, v109, s3
  v_add_u32 v109, v108, s3
  v_add_u32 v108, v14, s6
  v_add_u32 v14, v13, s6
  v_add_u32 v13, v250, s13
  v_add_u32 v250, v126, s13
  v_add_u32 v126, v124, s13
  v_add_u32 v124, v247, s13
  v_add_u32 v247, v118, s13
  v_add_u32 v118, v116, s13
  v_add_u32 v116, v114, s13
  v_add_u32 v114, v240, s6
  v_add_u32 v240, v228, s3
  v_accvgpr_write_b32 a4, 0
  v_accvgpr_write_b32 a5, 0
  v_accvgpr_write_b32 a6, 0
  v_accvgpr_write_b32 a7, 0
  v_accvgpr_write_b32 a8, 0
  v_accvgpr_write_b32 a9, 0
  v_accvgpr_write_b32 a10, 0
  v_accvgpr_write_b32 a11, 0
  v_accvgpr_write_b32 a12, 0
  v_accvgpr_write_b32 a13, 0
  v_accvgpr_write_b32 a14, 0
  v_accvgpr_write_b32 a15, 0
  v_accvgpr_write_b32 a16, 0
  v_accvgpr_write_b32 a17, 0
  v_accvgpr_write_b32 a18, 0
  v_accvgpr_write_b32 a19, 0
  v_accvgpr_write_b32 a20, 0
  v_accvgpr_write_b32 a21, 0
  v_accvgpr_write_b32 a22, 0
  v_accvgpr_write_b32 a23, 0
  v_accvgpr_write_b32 a24, 0
  v_accvgpr_write_b32 a25, 0
  v_accvgpr_write_b32 a26, 0
  v_accvgpr_write_b32 a27, 0
  v_accvgpr_write_b32 a28, 0
  v_accvgpr_write_b32 a29, 0
  v_accvgpr_write_b32 a30, 0
  v_accvgpr_write_b32 a31, 0
  v_accvgpr_write_b32 a32, 0
  v_accvgpr_write_b32 a33, 0
  v_accvgpr_write_b32 a34, 0
  v_accvgpr_write_b32 a35, 0
  v_accvgpr_write_b32 a36, 0
  v_accvgpr_write_b32 a37, 0
  v_accvgpr_write_b32 a38, 0
  v_accvgpr_write_b32 a39, 0
  v_accvgpr_write_b32 a40, 0
  v_accvgpr_write_b32 a41, 0
  v_accvgpr_write_b32 a42, 0
  v_accvgpr_write_b32 a43, 0
  v_accvgpr_write_b32 a44, 0
  v_accvgpr_write_b32 a45, 0
  v_accvgpr_write_b32 a46, 0
  v_accvgpr_write_b32 a47, 0
  v_accvgpr_write_b32 a48, 0
  v_accvgpr_write_b32 a49, 0
  v_accvgpr_write_b32 a50, 0
  v_accvgpr_write_b32 a51, 0
  v_accvgpr_write_b32 a52, 0
  v_accvgpr_write_b32 a53, 0
  v_accvgpr_write_b32 a54, 0
  v_accvgpr_write_b32 a55, 0
  v_accvgpr_write_b32 a56, 0
  v_accvgpr_write_b32 a57, 0
  v_accvgpr_write_b32 a58, 0
  v_accvgpr_write_b32 a59, 0
  v_accvgpr_write_b32 a60, 0
  v_accvgpr_write_b32 a61, 0
  v_accvgpr_write_b32 a62, 0
  v_accvgpr_write_b32 a63, 0
  v_accvgpr_write_b32 a64, 0
  v_accvgpr_write_b32 a65, 0
  v_accvgpr_write_b32 a66, 0
  v_accvgpr_write_b32 a67, 0
  v_accvgpr_write_b32 a68, 0
  v_accvgpr_write_b32 a69, 0
  v_accvgpr_write_b32 a70, 0
  v_accvgpr_write_b32 a71, 0
  v_accvgpr_write_b32 a72, 0
  v_accvgpr_write_b32 a73, 0
  v_accvgpr_write_b32 a74, 0
  v_accvgpr_write_b32 a75, 0
  v_accvgpr_write_b32 a76, 0
  v_accvgpr_write_b32 a77, 0
  v_accvgpr_write_b32 a78, 0
  v_accvgpr_write_b32 a79, 0
  v_accvgpr_write_b32 a80, 0
  v_accvgpr_write_b32 a81, 0
  v_accvgpr_write_b32 a82, 0
  v_accvgpr_write_b32 a83, 0
  v_accvgpr_write_b32 a84, 0
  v_accvgpr_write_b32 a85, 0
  v_accvgpr_write_b32 a86, 0
  v_accvgpr_write_b32 a87, 0
  v_accvgpr_write_b32 a88, 0
  v_accvgpr_write_b32 a89, 0
  v_accvgpr_write_b32 a90, 0
  v_accvgpr_write_b32 a91, 0
  v_accvgpr_write_b32 a92, 0
  v_accvgpr_write_b32 a93, 0
  v_accvgpr_write_b32 a94, 0
  v_accvgpr_write_b32 a95, 0
  v_accvgpr_write_b32 a96, 0
  v_accvgpr_write_b32 a97, 0
  v_accvgpr_write_b32 a98, 0
  v_accvgpr_write_b32 a99, 0
  v_accvgpr_write_b32 a100, 0
  v_accvgpr_write_b32 a101, 0
  v_accvgpr_write_b32 a102, 0
  v_accvgpr_write_b32 a103, 0
  v_accvgpr_write_b32 a104, 0
  v_accvgpr_write_b32 a105, 0
  v_accvgpr_write_b32 a106, 0
  v_accvgpr_write_b32 a107, 0
  v_accvgpr_write_b32 a108, 0
  v_accvgpr_write_b32 a109, 0
  v_accvgpr_write_b32 a110, 0
  v_accvgpr_write_b32 a111, 0
  v_accvgpr_write_b32 a112, 0
  v_accvgpr_write_b32 a113, 0
  v_accvgpr_write_b32 a114, 0
  v_accvgpr_write_b32 a115, 0
  v_accvgpr_write_b32 a116, 0
  v_accvgpr_write_b32 a117, 0
  v_accvgpr_write_b32 a118, 0
  v_accvgpr_write_b32 a119, 0
  v_accvgpr_write_b32 a120, 0
  v_accvgpr_write_b32 a121, 0
  v_accvgpr_write_b32 a122, 0
  v_accvgpr_write_b32 a123, 0
  v_accvgpr_write_b32 a124, 0
  v_accvgpr_write_b32 a125, 0
  v_accvgpr_write_b32 a126, 0
  v_accvgpr_write_b32 a127, 0
  v_accvgpr_write_b32 a128, 0
  v_accvgpr_write_b32 a129, 0
  v_accvgpr_write_b32 a130, 0
  v_accvgpr_write_b32 a131, 0
  v_accvgpr_write_b32 a132, 0
  v_accvgpr_write_b32 a133, 0
  v_accvgpr_write_b32 a134, 0
  v_accvgpr_write_b32 a135, 0
  v_accvgpr_write_b32 a136, 0
  v_accvgpr_write_b32 a137, 0
  v_accvgpr_write_b32 a138, 0
  v_accvgpr_write_b32 a139, 0
  v_accvgpr_write_b32 a140, 0
  v_accvgpr_write_b32 a141, 0
  v_accvgpr_write_b32 a142, 0
  v_accvgpr_write_b32 a143, 0
  v_accvgpr_write_b32 a144, 0
  v_accvgpr_write_b32 a145, 0
  v_accvgpr_write_b32 a146, 0
  v_accvgpr_write_b32 a147, 0
  v_accvgpr_write_b32 a148, 0
  v_accvgpr_write_b32 a149, 0
  v_accvgpr_write_b32 a150, 0
  v_accvgpr_write_b32 a151, 0
  v_accvgpr_write_b32 a152, 0
  v_accvgpr_write_b32 a153, 0
  v_accvgpr_write_b32 a154, 0
  v_accvgpr_write_b32 a155, 0
  v_accvgpr_write_b32 a156, 0
  v_accvgpr_write_b32 a157, 0
  v_accvgpr_write_b32 a158, 0
  v_accvgpr_write_b32 a159, 0
  v_accvgpr_write_b32 a160, 0
  v_accvgpr_write_b32 a161, 0
  v_accvgpr_write_b32 a162, 0
  v_accvgpr_write_b32 a163, 0
  v_accvgpr_write_b32 a164, 0
  v_accvgpr_write_b32 a165, 0
  v_accvgpr_write_b32 a166, 0
  v_accvgpr_write_b32 a167, 0
  v_accvgpr_write_b32 a168, 0
  v_accvgpr_write_b32 a169, 0
  v_accvgpr_write_b32 a170, 0
  v_accvgpr_write_b32 a171, 0
  v_accvgpr_write_b32 a172, 0
  v_accvgpr_write_b32 a173, 0
  v_accvgpr_write_b32 a174, 0
  v_accvgpr_write_b32 a175, 0
  v_accvgpr_write_b32 a176, 0
  v_accvgpr_write_b32 a177, 0
  v_accvgpr_write_b32 a178, 0
  v_accvgpr_write_b32 a179, 0
  v_accvgpr_write_b32 a180, 0
  v_accvgpr_write_b32 a181, 0
  v_accvgpr_write_b32 a182, 0
  v_accvgpr_write_b32 a183, 0
  v_accvgpr_write_b32 a184, 0
  v_accvgpr_write_b32 a185, 0
  v_accvgpr_write_b32 a186, 0
  v_accvgpr_write_b32 a187, 0
  v_accvgpr_write_b32 a188, 0
  v_accvgpr_write_b32 a189, 0
  v_accvgpr_write_b32 a190, 0
  v_accvgpr_write_b32 a191, 0
  s_mov_b32 s2, 0
  s_mov_b32 s3, 0
  s_mov_b32 s6, 0
  s_mov_b32 s13, 0
  s_mov_b32 s15, 0
  s_mov_b32 s19, 0
  s_mov_b32 s64, 0
L_loop_0:
  s_waitcnt vmcnt(10) & lgkmcnt(0)
  s_barrier
  ; sched_barrier mask=0
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[220:223], v[200:203], a[0:3], v232, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[96:99], v228, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[212:215], v[196:199], a[0:3], v232, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[104:107], v227 offset:45056
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[220:223], v[192:195], a[4:7], v232, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[92:95], v240, s[28:31], s64 offen offset:1024
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[212:215], v[188:191], a[4:7], v232, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v12, v231, s[32:35], s2 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[220:223], v[184:187], a[8:11], v232, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[88:91], v251, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[212:215], v[180:183], a[8:11], v232, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[100:103], v226 offset:45056
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[220:223], v[176:179], a[12:15], v232, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[84:87], v238, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[212:215], v[172:175], a[12:15], v232, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[220:223], v[168:171], a[16:19], v232, v249 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[80:83], v224, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[212:215], v[164:167], a[16:19], v232, v249 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[16:19], v227 offset:47104
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[220:223], v[160:163], a[20:23], v232, v249 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[76:79], v218, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[212:215], v[204:207], a[20:23], v232, v249 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[200:203], a[24:27], v232, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[72:75], v217, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[196:199], a[24:27], v232, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[48:51], v226 offset:47104
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[192:195], a[28:31], v232, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[68:71], v216, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[188:191], a[28:31], v232, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[152:155], v[184:187], a[32:35], v232, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[64:67], v211, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:151], v[180:183], a[32:35], v232, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[36:39], v227 offset:49152
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[176:179], a[36:39], v232, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[60:63], v210, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:151], v[172:175], a[36:39], v232, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[168:171], a[40:43], v232, v249 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[56:59], v159, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[164:167], a[40:43], v232, v249 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[32:35], v226 offset:49152
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[160:163], a[44:47], v232, v249 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[52:55], v156, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[204:207], a[44:47], v232, v249 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[200:203], a[48:51], v229, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[196:199], a[48:51], v229, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[44:47], v227 offset:51200
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[192:195], a[52:55], v229, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[188:191], a[52:55], v229, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v3, v131, s[32:35], s2 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[184:187], a[56:59], v229, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[180:183], a[56:59], v229, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[40:43], v226 offset:51200
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[176:179], a[60:63], v229, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[172:175], a[60:63], v229, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v0, v225 offset:2560
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[168:171], a[64:67], v229, v249 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[164:167], a[64:67], v229, v249 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[160:163], a[68:71], v229, v249 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[204:207], a[68:71], v229, v249 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:139], v[200:203], a[72:75], v229, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[196:199], a[72:75], v229, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:139], v[192:195], a[76:79], v229, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[188:191], a[76:79], v229, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[184:187], a[80:83], v229, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[132:135], v[180:183], a[80:83], v229, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[176:179], a[84:87], v229, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[172:175], a[84:87], v229, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v1, v225 offset:2816
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[168:171], a[88:91], v229, v249 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[164:167], a[88:91], v229, v249 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[160:163], a[92:95], v229, v249 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[204:207], a[92:95], v229, v249 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v2, v130, s[32:35], s2 offen
  ; sched_barrier mask=0
  s_waitcnt vmcnt(20) & lgkmcnt(0)
  s_barrier
  ; sched_barrier mask=0
  ; sched_barrier mask=0
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[200:203], a[96:99], v0, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_add_u32 s68, s19, 2
  s_cmp_lt_i32 s68, s66
  s_and_b32 s21, s21, 0xffff
  s_or_b32 s21, s21, 0x40000000
  s_cmp_lt_i32 s68, s66
  s_cselect_b32 s22, s18, 0
  s_mov_b32 s23, 0x27000
  s_mov_b32 m0, s4
  buffer_load_dwordx4 v245, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[100:103], v[196:199], a[96:99], v0, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[24:27], v227 offset:4096
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[192:195], a[100:103], v0, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[100:103], v[188:191], a[100:103], v0, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s41
  buffer_load_dwordx4 v129, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[184:187], a[104:107], v0, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[20:23], v226 offset:4096
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[100:103], v[180:183], a[104:107], v0, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[176:179], a[108:111], v0, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s43
  buffer_load_dwordx4 v127, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[100:103], v[172:175], a[108:111], v0, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[28:31], v227 offset:6144
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[168:171], a[112:115], v0, v249 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[100:103], v[164:167], a[112:115], v0, v249 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s45
  buffer_load_dwordx4 v125, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[160:163], a[116:119], v0, v249 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[104:107], v226 offset:6144
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[100:103], v[204:207], a[116:119], v0, v249 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[16:19], v[200:203], a[120:123], v0, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_cmp_lt_i32 s68, s66
  s_and_b32 s25, s25, 0xffff
  s_or_b32 s25, s25, 0x40000000
  s_cmp_lt_i32 s68, s66
  s_cselect_b32 s26, s48, 0
  s_mov_b32 s27, 0x27000
  s_mov_b32 m0, s14
  buffer_load_dword v121, s[24:27], s6 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[48:51], v[196:199], a[120:123], v0, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[100:103], v227 offset:8192
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[16:19], v[192:195], a[124:127], v0, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[48:51], v[188:191], a[124:127], v0, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s49
  buffer_load_dwordx4 v123, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[16:19], v[184:187], a[128:131], v0, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[4:7], v226 offset:8192
  v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[48:51], v[180:183], a[128:131], v0, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[16:19], v[176:179], a[132:135], v0, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s51
  buffer_load_dwordx4 v119, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[48:51], v[172:175], a[132:135], v0, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[8:11], v227 offset:10240
  v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[16:19], v[168:171], a[136:139], v0, v249 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[48:51], v[164:167], a[136:139], v0, v249 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s65
  buffer_load_dwordx4 v117, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[16:19], v[160:163], a[140:143], v0, v249 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[16:19], v226 offset:10240
  v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[48:51], v[204:207], a[140:143], v0, v249 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[36:39], v[200:203], a[144:147], v1, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s67
  buffer_load_dwordx4 v115, s[20:23], s3 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[32:35], v[196:199], a[144:147], v1, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v48, v225
  v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[36:39], v[192:195], a[148:151], v1, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[32:35], v[188:191], a[148:151], v1, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s70
  buffer_load_dword v113, s[24:27], s6 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[36:39], v[184:187], a[152:155], v1, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[32:35], v[180:183], a[152:155], v1, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[36:39], v[176:179], a[156:159], v1, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[32:35], v[172:175], a[156:159], v1, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[36:39], v[168:171], a[160:163], v1, v249 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[32:35], v[164:167], a[160:163], v1, v249 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[36:39], v[160:163], a[164:167], v1, v249 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[32:35], v[204:207], a[164:167], v1, v249 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[44:47], v[200:203], a[168:171], v1, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[40:43], v[196:199], a[168:171], v1, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v49, v225 offset:256
  v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[44:47], v[192:195], a[172:175], v1, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[40:43], v[188:191], a[172:175], v1, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[44:47], v[184:187], a[176:179], v1, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[40:43], v[180:183], a[176:179], v1, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[44:47], v[176:179], a[180:183], v1, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[40:43], v[172:175], a[180:183], v1, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[44:47], v[168:171], a[184:187], v1, v249 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[40:43], v[164:167], a[184:187], v1, v249 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[44:47], v[160:163], a[188:191], v1, v249 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[40:43], v[204:207], a[188:191], v1, v249 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ; sched_barrier mask=0
  s_waitcnt vmcnt(10) & lgkmcnt(0)
  s_barrier
  ; sched_barrier mask=0
  ; sched_barrier mask=0
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[96:99], a[0:3], v48, v12 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[200:203], v240, s[28:31], s64 offen offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[92:95], a[0:3], v48, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[36:39], v227 offset:12288
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:27], v[88:91], a[4:7], v48, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[196:199], v240, s[28:31], s64 offen offset:3072
  v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[84:87], a[4:7], v48, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v158, v248, s[32:35], s2 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:27], v[80:83], a[8:11], v48, v3 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[192:195], v230, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[76:79], a[8:11], v48, v3 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[32:35], v226 offset:12288
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:27], v[72:75], a[12:15], v48, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[188:191], v235, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[68:71], a[12:15], v48, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:27], v[64:67], a[16:19], v48, v2 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[184:187], v219, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:23], v[60:63], a[16:19], v48, v2 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[44:47], v227 offset:14336
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:27], v[56:59], a[20:23], v48, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[180:183], v209, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:23], v[52:55], a[20:23], v48, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[28:31], v[96:99], a[24:27], v48, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[176:179], v208, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[92:95], a[24:27], v48, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[40:43], v226 offset:14336
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[28:31], v[88:91], a[28:31], v48, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[172:175], v120, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:107], v[84:87], a[28:31], v48, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[28:31], v[80:83], a[32:35], v48, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[168:171], v112, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[76:79], a[32:35], v48, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[24:27], v227 offset:16384
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[28:31], v[72:75], a[36:39], v48, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[164:167], v111, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[68:71], a[36:39], v48, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[28:31], v[64:67], a[40:43], v48, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[160:163], v110, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[60:63], a[40:43], v48, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[20:23], v226 offset:16384
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[28:31], v[56:59], a[44:47], v48, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  buffer_load_dwordx4 v[204:207], v109, s[28:31], s64 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[52:55], a[44:47], v48, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:103], v[96:99], a[48:51], v49, v12 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[4:7], v[92:95], a[48:51], v49, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[104:107], v227 offset:18432
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:103], v[88:91], a[52:55], v49, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[4:7], v[84:87], a[52:55], v49, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v157, v108, s[32:35], s2 offen
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[80:83], a[56:59], v49, v3 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[4:7], v[76:79], a[56:59], v49, v3 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[28:31], v226 offset:18432
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[72:75], a[60:63], v49, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[4:7], v[68:71], a[60:63], v49, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v48, v225 offset:512
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[64:67], a[64:67], v49, v2 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[4:7], v[60:63], a[64:67], v49, v2 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[56:59], a[68:71], v49, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[4:7], v[52:55], a[68:71], v49, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[8:11], v[96:99], a[72:75], v49, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[16:19], v[92:95], a[72:75], v49, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[8:11], v[88:91], a[76:79], v49, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[16:19], v[84:87], a[76:79], v49, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[8:11], v[80:83], a[80:83], v49, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[16:19], v[76:79], a[80:83], v49, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[8:11], v[72:75], a[84:87], v49, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[16:19], v[68:71], a[84:87], v49, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v100, v225 offset:768
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[8:11], v[64:67], a[88:91], v49, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:19], v[60:63], a[88:91], v49, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[8:11], v[56:59], a[92:95], v49, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[16:19], v[52:55], a[92:95], v49, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  buffer_load_dword v249, v14, s[32:35], s2 offen
  ; sched_barrier mask=0
  s_waitcnt vmcnt(20) & lgkmcnt(0)
  s_barrier
  ; sched_barrier mask=0
  ; sched_barrier mask=0
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[36:39], v[96:99], a[96:99], v48, v12 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_add_u32 s72, s19, 3
  s_cmp_lt_i32 s72, s66
  s_and_b32 s21, s21, 0xffff
  s_or_b32 s21, s21, 0x40000000
  s_cmp_lt_i32 s72, s66
  s_cselect_b32 s22, s18, 0
  s_mov_b32 s23, 0x27000
  s_mov_b32 m0, s5
  buffer_load_dwordx4 v233, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[32:35], v[92:95], a[96:99], v48, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[220:223], v227 offset:36864
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[36:39], v[88:91], a[100:103], v48, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[32:35], v[84:87], a[100:103], v48, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s10
  buffer_load_dwordx4 v13, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[36:39], v[80:83], a[104:107], v48, v3 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[212:215], v226 offset:36864
  v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[32:35], v[76:79], a[104:107], v48, v3 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[36:39], v[72:75], a[108:111], v48, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s11
  buffer_load_dwordx4 v250, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[32:35], v[68:71], a[108:111], v48, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[152:155], v227 offset:38912
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[36:39], v[64:67], a[112:115], v48, v2 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[32:35], v[60:63], a[112:115], v48, v2 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s40
  buffer_load_dwordx4 v126, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[36:39], v[56:59], a[116:119], v48, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[148:151], v226 offset:38912
  v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[32:35], v[52:55], a[116:119], v48, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[44:47], v[96:99], a[120:123], v48, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_cmp_lt_i32 s72, s66
  s_and_b32 s25, s25, 0xffff
  s_or_b32 s25, s25, 0x40000000
  s_cmp_lt_i32 s72, s66
  s_cselect_b32 s26, s48, 0
  s_mov_b32 s27, 0x27000
  s_mov_b32 m0, s47
  buffer_load_dword v122, s[24:27], s15 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[40:43], v[92:95], a[120:123], v48, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[144:147], v227 offset:40960
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[44:47], v[88:91], a[124:127], v48, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[40:43], v[84:87], a[124:127], v48, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s42
  buffer_load_dwordx4 v124, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[44:47], v[80:83], a[128:131], v48, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[140:143], v226 offset:40960
  v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[40:43], v[76:79], a[128:131], v48, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[44:47], v[72:75], a[132:135], v48, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s44
  buffer_load_dwordx4 v247, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[40:43], v[68:71], a[132:135], v48, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b128 v[136:139], v227 offset:43008
  v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[44:47], v[64:67], a[136:139], v48, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[40:43], v[60:63], a[136:139], v48, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s46
  buffer_load_dwordx4 v118, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[44:47], v[56:59], a[140:143], v48, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  ds_read_b128 v[132:135], v226 offset:43008
  v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[40:43], v[52:55], a[140:143], v48, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[24:27], v[96:99], a[144:147], v100, v12 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  s_mov_b32 m0, s50
  buffer_load_dwordx4 v116, s[20:23], s13 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[20:23], v[92:95], a[144:147], v100, v12 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v232, v225 offset:2048
  v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[24:27], v[88:91], a[148:151], v100, v12 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[20:23], v[84:87], a[148:151], v100, v12 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  s_mov_b32 m0, s69
  buffer_load_dword v114, s[24:27], s15 offen lds
  v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[24:27], v[80:83], a[152:155], v100, v3 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[20:23], v[76:79], a[152:155], v100, v3 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[24:27], v[72:75], a[156:159], v100, v3 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[20:23], v[68:71], a[156:159], v100, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[24:27], v[64:67], a[160:163], v100, v2 op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[20:23], v[60:63], a[160:163], v100, v2 op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[24:27], v[56:59], a[164:167], v100, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[20:23], v[52:55], a[164:167], v100, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[104:107], v[96:99], a[168:171], v100, v12 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[28:31], v[92:95], a[168:171], v100, v12 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ds_read_b32 v229, v225 offset:2304
  v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:107], v[88:91], a[172:175], v100, v12 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[28:31], v[84:87], a[172:175], v100, v12 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[104:107], v[80:83], a[176:179], v100, v3 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[28:31], v[76:79], a[176:179], v100, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[104:107], v[72:75], a[180:183], v100, v3 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[28:31], v[68:71], a[180:183], v100, v3 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:107], v[64:67], a[184:187], v100, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[28:31], v[60:63], a[184:187], v100, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:107], v[56:59], a[188:191], v100, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
  v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[28:31], v[52:55], a[188:191], v100, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
  ; sched_barrier mask=0
  s_waitcnt vmcnt(10) & lgkmcnt(0)
  s_barrier
  ; sched_barrier mask=0
  s_add_u32 s64, s64, 4096
  s_add_u32 s2, s2, 512
  s_add_u32 s3, s3, 256
  s_add_u32 s6, s6, 512
  s_add_u32 s13, s13, 256
  s_add_u32 s15, s15, 512
  s_cmp_lt_u32 s68, s66
  s_mov_b32 s19, s68
  s_cbranch_scc1 L_loop_0
  v_add_u32 v249, v253, v244
  v_add_u32 v250, v249, v237
  v_sub_u32 v249, v250, v234
  v_cmp_lt_i32 vcc, v249, s53
  v_mov_b32 v250, 1
  v_cndmask_b32 v247, 0, v250, vcc
  v_lshl_or_b32 v250, v128, 8, v239
  v_lshl_or_b32 v240, v242, 2, v250
  v_cmp_lt_i32 vcc, v240, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_lshlrev_b32 v251, 8, v243
  v_sub_u32 v245, 0, v251
  v_lshl_add_u32 v251, v128, 8, v245
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s7, 0
  s_cmp_lg_u32 s9, 0
  v_mov_b32 v245, s8
  v_cmp_ge_u32 vcc, v246, v245
  v_mov_b32 v245, 1
  v_cndmask_b32 v238, 0, v245, vcc
  v_add_u32 v245, v236, v238
  v_cndmask_b32 v236, v246, v241
  v_mov_b32 v246, s8
  v_cmp_ge_u32 vcc, v236, v246
  v_mov_b32 v246, 1
  v_cndmask_b32 v236, 0, v246, vcc
  v_add_u32 v246, v245, v236
  s_nop 0
  v_readfirstlane_b32 s2, v246
  s_mul_i32 s3, s2, s8
  s_sub_u32 s2, s12, s3
  s_add_u32 s3, s71, s2
  v_cmp_ne_u32 vcc, v252, 0
  v_mov_b32 v245, s3
  v_cndmask_b32 v246, 0, v245, vcc
  v_mov_b32 v15, -192
  v_mul_lo_u32 v252, v246, v15
  v_add_u32 v245, v244, v252
  v_lshl_or_b32 v252, v243, 8, v239
  v_lshl_or_b32 v239, v242, 2, v252
  v_mov_b32 v15, 192
  v_mul_lo_u32 v252, v246, v15
  v_add_u32 v246, v253, v252
  v_add_u32 v253, v246, v237
  v_sub_u32 v252, v253, v234
  v_mul_lo_u32 v253, v251, s59
  v_mul_lo_u32 v251, v239, s59
  v_add_u32 v246, v253, v245
  v_add_u32 v253, v251, v252
  s_mov_b64 s[60:61], s[36:37]
  v_readfirstlane_b32 s63, v246
  s_mul_hi_i32 s62, s63, 2
  s_mul_i32 s64, s63, 2
  s_add_u32 s60, s60, s64
  s_addc_u32 s61, s61, s62
  s_mov_b32 s62, 0x7FFFFFFC
  s_mov_b32 s63, 0x20000
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v253
  v_accvgpr_read_b32 v253, a0
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_or_b32 v253, v240, 1
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v251, 0, v253, vcc
  v_and_b32 v253, v247, v251
  v_or_b32 v245, v239, 1
  v_mul_lo_u32 v246, v245, s59
  v_add_u32 v245, v246, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v245
  v_accvgpr_read_b32 v245, a1
  v_cvt_pk_bf16_f32 v245, v245, 0
  buffer_store_short v245, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_or_b32 v245, v240, 2
  v_cmp_lt_i32 vcc, v245, s52
  v_mov_b32 v245, 1
  v_cndmask_b32 v246, 0, v245, vcc
  v_and_b32 v245, v247, v246
  v_or_b32 v237, v239, 2
  v_mul_lo_u32 v236, v237, s59
  v_add_u32 v237, v236, v252
  v_cmp_ne_u32 vcc, v245, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v245, 1, v237
  v_accvgpr_read_b32 v237, a2
  v_cvt_pk_bf16_f32 v237, v237, 0
  buffer_store_short v237, v245, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_or_b32 v237, v240, 3
  v_cmp_lt_i32 vcc, v237, s52
  v_mov_b32 v237, 1
  v_cndmask_b32 v236, 0, v237, vcc
  v_and_b32 v237, v247, v236
  v_or_b32 v244, v239, 3
  v_mul_lo_u32 v238, v244, s59
  v_add_u32 v244, v238, v252
  v_cmp_ne_u32 vcc, v237, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v237, 1, v244
  v_accvgpr_read_b32 v244, a3
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v237, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v244, v249, 16
  v_cmp_lt_i32 vcc, v244, s53
  v_mov_b32 v244, 1
  v_cndmask_b32 v238, 0, v244, vcc
  v_and_b32 v244, v238, v248
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a4
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v238, v251
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a5
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v238, v246
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a6
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v245, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v238, v236
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a7
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v237, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_add_u32 v244, v249, 32
  v_cmp_lt_i32 vcc, v244, s53
  v_mov_b32 v244, 1
  v_cndmask_b32 v241, 0, v244, vcc
  v_and_b32 v244, v241, v248
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a8
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v241, v251
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a9
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v241, v246
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a10
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v245, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v241, v236
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a11
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v237, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_add_u32 v244, v249, 48
  v_cmp_lt_i32 vcc, v244, s53
  v_mov_b32 v244, 1
  v_cndmask_b32 v242, 0, v244, vcc
  v_and_b32 v244, v242, v248
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a12
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v242, v251
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a13
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v242, v246
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a14
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v245, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v242, v236
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a15
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v237, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_add_u32 v244, v249, 64
  v_cmp_lt_i32 vcc, v244, s53
  v_mov_b32 v244, 1
  v_cndmask_b32 v243, 0, v244, vcc
  v_and_b32 v244, v243, v248
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a16
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v243, v251
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a17
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v243, v246
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a18
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v245, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v244, v243, v236
  v_cmp_ne_u32 vcc, v244, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v244, a19
  v_cvt_pk_bf16_f32 v244, v244, 0
  buffer_store_short v244, v237, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 80
  v_add_u32 v244, v249, v15
  v_cmp_lt_i32 vcc, v244, s53
  v_mov_b32 v249, 1
  v_cndmask_b32 v244, 0, v249, vcc
  v_and_b32 v249, v244, v248
  v_cmp_ne_u32 vcc, v249, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a20
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v249, v244, v251
  v_cmp_ne_u32 vcc, v249, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a21
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v246
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a22
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v245, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v236
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a23
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v237, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_add_u32 v253, v240, 16
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, 16
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a24
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v250, v240, 17
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, 17
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a25
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v251, v240, 18
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, 18
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a26
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v246, v240, 19
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, 19
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a27
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a28
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a29
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a30
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a31
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a32
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a33
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a34
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a35
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a36
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a37
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a38
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a39
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a40
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a41
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a42
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a43
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a44
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a45
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a46
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a47
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_add_u32 v253, v240, 32
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, 32
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a48
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v250, v240, 33
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, 33
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a49
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v251, v240, 34
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, 34
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a50
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v246, v240, 35
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, 35
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a51
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a52
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a53
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a54
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a55
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a56
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a57
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a58
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a59
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a60
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a61
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a62
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a63
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a64
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a65
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a66
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a67
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a68
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a69
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a70
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a71
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_add_u32 v253, v240, 48
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, 48
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a72
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v250, v240, 49
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, 49
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a73
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v251, v240, 50
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, 50
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a74
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_add_u32 v246, v240, 51
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, 51
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a75
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a76
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a77
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a78
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a79
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a80
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a81
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a82
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a83
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a84
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a85
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a86
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a87
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a88
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a89
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a90
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a91
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a92
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a93
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a94
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a95
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_add_u32 v253, v240, 64
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, 64
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a96
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 65
  v_add_u32 v250, v240, v15
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, v15
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a97
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 66
  v_add_u32 v251, v240, v15
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, v15
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a98
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 67
  v_add_u32 v246, v240, v15
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, v15
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a99
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a100
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a101
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a102
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a103
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a104
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a105
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a106
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a107
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a108
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a109
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a110
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a111
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a112
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a113
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a114
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a115
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a116
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a117
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a118
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a119
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 80
  v_add_u32 v253, v240, v15
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, v15
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a120
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 81
  v_add_u32 v250, v240, v15
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, v15
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a121
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 82
  v_add_u32 v251, v240, v15
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, v15
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a122
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 83
  v_add_u32 v246, v240, v15
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, v15
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a123
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a124
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a125
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a126
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a127
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a128
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a129
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a130
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a131
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a132
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a133
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a134
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a135
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a136
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a137
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a138
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a139
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a140
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a141
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a142
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a143
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 96
  v_add_u32 v253, v240, v15
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, v15
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a144
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 97
  v_add_u32 v250, v240, v15
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, v15
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a145
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 98
  v_add_u32 v251, v240, v15
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, v15
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a146
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 99
  v_add_u32 v246, v240, v15
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v246, 1
  v_cndmask_b32 v237, 0, v246, vcc
  v_and_b32 v246, v247, v237
  v_add_u32 v236, v239, v15
  v_mul_lo_u32 v235, v236, s59
  v_add_u32 v236, v235, v252
  v_cmp_ne_u32 vcc, v246, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v246, 1, v236
  v_accvgpr_read_b32 v236, a147
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a148
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a149
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a150
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v238, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a151
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a152
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a153
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a154
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v241, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a155
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a156
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a157
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a158
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v242, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a159
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a160
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v248
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a161
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v245
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a162
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v243, v237
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v236, a163
  v_cvt_pk_bf16_f32 v236, v236, 0
  buffer_store_short v236, v246, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v236, v244, v249
  v_cmp_ne_u32 vcc, v236, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a164
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a165
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a166
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v237
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a167
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v246, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 112
  v_add_u32 v253, v240, v15
  v_cmp_lt_i32 vcc, v253, s52
  v_mov_b32 v253, 1
  v_cndmask_b32 v249, 0, v253, vcc
  v_and_b32 v253, v247, v249
  v_add_u32 v250, v239, v15
  v_mul_lo_u32 v248, v250, s59
  v_add_u32 v250, v248, v252
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v253, 1, v250
  v_accvgpr_read_b32 v250, a168
  v_cvt_pk_bf16_f32 v250, v250, 0
  buffer_store_short v250, v253, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 113
  v_add_u32 v250, v240, v15
  v_cmp_lt_i32 vcc, v250, s52
  v_mov_b32 v250, 1
  v_cndmask_b32 v248, 0, v250, vcc
  v_and_b32 v250, v247, v248
  v_add_u32 v251, v239, v15
  v_mul_lo_u32 v245, v251, s59
  v_add_u32 v251, v245, v252
  v_cmp_ne_u32 vcc, v250, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v250, 1, v251
  v_accvgpr_read_b32 v251, a169
  v_cvt_pk_bf16_f32 v251, v251, 0
  buffer_store_short v251, v250, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 114
  v_add_u32 v251, v240, v15
  v_cmp_lt_i32 vcc, v251, s52
  v_mov_b32 v251, 1
  v_cndmask_b32 v245, 0, v251, vcc
  v_and_b32 v251, v247, v245
  v_add_u32 v246, v239, v15
  v_mul_lo_u32 v237, v246, s59
  v_add_u32 v246, v237, v252
  v_cmp_ne_u32 vcc, v251, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v251, 1, v246
  v_accvgpr_read_b32 v246, a170
  v_cvt_pk_bf16_f32 v246, v246, 0
  buffer_store_short v246, v251, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_mov_b32 v15, 115
  v_add_u32 v246, v240, v15
  v_cmp_lt_i32 vcc, v246, s52
  v_mov_b32 v240, 1
  v_cndmask_b32 v246, 0, v240, vcc
  v_and_b32 v240, v247, v246
  v_add_u32 v247, v239, v15
  v_mul_lo_u32 v239, v247, s59
  v_add_u32 v247, v239, v252
  v_cmp_ne_u32 vcc, v240, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_lshlrev_b32 v252, 1, v247
  v_accvgpr_read_b32 v247, a171
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v252, s[60:63], 0 offen
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v238, v249
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a172
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v253, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v238, v248
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a173
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v250, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v238, v245
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a174
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v251, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v238, v246
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a175
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v252, s[60:63], 0 offen offset:32
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v241, v249
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a176
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v253, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v241, v248
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a177
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v250, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v241, v245
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a178
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v251, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v241, v246
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a179
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v252, s[60:63], 0 offen offset:64
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v242, v249
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a180
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v253, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v242, v248
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a181
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v250, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v242, v245
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a182
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v251, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v242, v246
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a183
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v252, s[60:63], 0 offen offset:96
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v243, v249
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a184
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v253, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v243, v248
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a185
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v250, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v243, v245
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a186
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v251, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v243, v246
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v247, a187
  v_cvt_pk_bf16_f32 v247, v247, 0
  buffer_store_short v247, v252, s[60:63], 0 offen offset:128
  s_mov_b64 exec, s[2:3]
  v_and_b32 v247, v244, v249
  v_cmp_ne_u32 vcc, v247, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v249, a188
  v_cvt_pk_bf16_f32 v249, v249, 0
  buffer_store_short v249, v253, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v248
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a189
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v250, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v245
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a190
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v251, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  v_and_b32 v253, v244, v246
  v_cmp_ne_u32 vcc, v253, 0
  s_and_saveexec_b64 s[2:3], vcc
  v_accvgpr_read_b32 v253, a191
  v_cvt_pk_bf16_f32 v253, v253, 0
  buffer_store_short v253, v252, s[60:63], 0 offen offset:160
  s_mov_b64 exec, s[2:3]
  s_endpgm

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel wave_mxfp4_dynamic_gemm_256x192x256
  .amdhsa_group_segment_fixed_size 69632
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 208
  .amdhsa_user_sgpr_count 16
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_kernarg_preload_length 14
  .amdhsa_user_sgpr_kernarg_preload_offset 0
  .amdhsa_user_sgpr_private_segment_size 0
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_enable_private_segment 0
  .amdhsa_accum_offset 256
  .amdhsa_next_free_vgpr 448
  .amdhsa_next_free_sgpr 80
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 1
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: wave_mxfp4_dynamic_gemm_256x192x256
    .symbol: wave_mxfp4_dynamic_gemm_256x192x256.kd
    .args:
      - .name:       arg0_ptr
        .offset:     0
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg1_ptr
        .offset:     8
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg2_ptr
        .offset:     16
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg3_ptr
        .offset:     24
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg4_ptr
        .offset:     32
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg5_ptr
        .offset:     40
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg6_ptr
        .offset:     48
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg7_ptr
        .offset:     56
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg8_ptr
        .offset:     64
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg9_ptr
        .offset:     72
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg10_ptr
        .offset:     80
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg11_ptr
        .offset:     88
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
      - .name:       arg12_ptr
        .offset:     96
        .size:       8
        .value_kind: global_buffer
        .value_type: 'i8*'
    .kernarg_segment_size: 208
    .group_segment_fixed_size: 69632
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 78
    .vgpr_count: 254
    .agpr_count: 192
    .max_flat_workgroup_size: 256
...
.end_amdgpu_metadata
