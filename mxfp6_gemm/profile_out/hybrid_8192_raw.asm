; ============================================================================
; KERNEL: _ZN5mxfp615lds_gemm_hybridILi256ELi256ELi192ELi2ELi2ELi1ELi0ELb1E6__halfLi6ELb1EEE...
;   lds_gemm_hybrid<M_TILE=256, N_TILE=256, K_TILE=192, WM=2, WN=2,
;                   MIN_OCC=1, SWZ=0, DB=true, OutT=__half, PFD=6, SHUF=true>
; TARGET: gfx950 (MI350X / CDNA4)
; RESOURCES: vgpr_count=508 (252 arch + 256 acc), agpr=256, sgpr=32,
;            LDS=73728 B (A double-buffered only; B streamed direct HBM->VGPR), spill=0
; PROBLEM:  8192^3, Kp=8256, k_iters=129, k_tiles=43, grid 32x32, 4 waves/WG
; ----------------------------------------------------------------------------
; DERIVED CONSTANTS (from template params):
;   KT_BYTES = K_TILE*6/8 = 144      ROW_CHUNKS = 144/16 = 9
;   K64_PER_TILE = 192/64 = 3 (= "sub" count)
;   M_BLKS=8 N_BLKS=8  M_PW=4 N_PW=4 (=> per-wave 4x4 = 16 acc blocks = 256 AccVGPR)
;   NB = K64_PER_TILE * N_PW = 3*4 = 12  (b-stream positions / tile)
;   Per tile: A coop load = 256*9/256 = 9 buffer_load_dwordx4 (HBM->LDS, M0+lane*16)
;             B direct     = NB=12 operands, each = global_load_dwordx4(16B,lane*16)
;                            + global_load_dwordx2(8B,1024+lane*8) = 24B/32 FP6  [SHUF]
;             scales       = 2 global_load_dwordx3 (sA,sB; K64_PER_TILE=3 dwords)
;             A ds_read    = per sub(=4 M-blocks): 4 ds_read_b128 (the 4 a[mi] operands,
;                            one b128 each since v6i 24B spans 2 reads? -> see body)
;                            + 2 ds_read2st64_b64 (the NEXT sub's a[] prefetch, 2-half)
;             compute      = NB(12) x M_PW(4) = 48 MFMA; each p=(sub,ni), inner mi=0..3
;                            writes acc[mi][ni] sharing b_cur = bring[p] (the p-th B slot)
; ============================================================================
; NOTE: byte-code columns stripped. Format: <vaddr>:  <mnemonic> <operands>
; ============================================================================


0000000000001d00 <_ZN5mxfp615lds_gemm_hybridILi256ELi256ELi192ELi2ELi2ELi1ELi0ELb1E6__halfLi6ELb1EEEvPKvS3_PKhS5_PT7_iiii>:
000000001D00:	s_load_dwordx8 s[4:11], s[0:1], 0x0
000000001D08:	s_load_dwordx2 s[20:21], s[0:1], 0x20
000000001D10:	s_load_dwordx4 s[12:15], s[0:1], 0x28
000000001D18:	s_lshl_b32 s0, s2, 8
000000001D1C:	v_lshrrev_b32_e32 v2, 7, v0
000000001D20:	v_lshl_or_b32 v2, s2, 1, v2
000000001D28:	v_and_b32_e32 v75, 63, v0
000000001D2C:	s_waitcnt lgkmcnt(0)
000000001D30:	s_mul_i32 s15, s14, s0
000000001D34:	s_mul_hi_i32 s1, s14, s0
000000001D38:	s_add_u32 s16, s4, s15
000000001D3C:	s_addc_u32 s5, s5, s1
000000001D40:	s_mul_hi_i32 s1, s13, 0x55555556
000000001D48:	s_lshr_b32 s2, s1, 31
000000001D4C:	s_add_i32 s1, s1, s2
000000001D50:	v_mul_lo_u32 v2, s1, v2
000000001D58:	v_lshlrev_b32_e32 v79, 6, v2
000000001D5C:	v_or_b32_e32 v2, v79, v75
000000001D60:	v_mad_i64_i32 v[2:3], s[18:19], v2, 12, s[8:9]
000000001D68:	global_load_dwordx3 v[72:74], v[2:3], off
000000001D70:	v_mul_u32_u24_e32 v2, 0x1c72, v0
000000001D78:	v_lshrrev_b32_e32 v3, 16, v2
000000001D7C:	v_mul_lo_u16_e32 v2, 9, v3
000000001D80:	v_bfe_u32 v1, v0, 6, 1
000000001D88:	s_lshl_b32 s4, s3, 1
000000001D8C:	v_sub_u16_e32 v2, v0, v2
000000001D90:	v_or_b32_e32 v4, s4, v1
000000001D94:	v_lshlrev_b16_e32 v2, 4, v2
000000001D98:	v_mul_lo_u32 v4, s1, v4
000000001DA0:	v_mad_u64_u32 v[88:89], s[22:23], s14, v3, v[2:3]
000000001DA8:	v_or_b32_e32 v2, 0x100, v0
000000001DB0:	v_lshl_or_b32 v4, v4, 6, v75
000000001DB8:	v_mul_u32_u24_e32 v3, 0x1c72, v2
000000001DC0:	v_mad_i64_i32 v[4:5], s[18:19], v4, 12, s[10:11]
000000001DC8:	v_lshrrev_b32_e32 v3, 16, v3
000000001DCC:	global_load_dwordx3 v[68:70], v[4:5], off
000000001DD4:	v_mul_lo_u16_e32 v4, 9, v3
000000001DD8:	v_sub_u16_e32 v2, v2, v4
000000001DDC:	v_lshlrev_b16_e32 v2, 4, v2
000000001DE0:	v_mad_u64_u32 v[90:91], s[22:23], s14, v3, v[2:3]
000000001DE8:	v_or_b32_e32 v2, 0x200, v0
000000001DF0:	v_mul_u32_u24_e32 v3, 0x1c72, v2
000000001DF8:	v_lshrrev_b32_e32 v3, 16, v3
000000001DFC:	v_mul_lo_u16_e32 v4, 9, v3
000000001E00:	v_sub_u16_e32 v2, v2, v4
000000001E04:	v_lshlrev_b16_e32 v2, 4, v2
000000001E08:	v_mad_u64_u32 v[92:93], s[22:23], s14, v3, v[2:3]
000000001E10:	v_or_b32_e32 v2, 0x300, v0
000000001E18:	v_mul_u32_u24_e32 v3, 0x1c72, v2
000000001E20:	v_lshrrev_b32_e32 v3, 16, v3
000000001E24:	v_mul_lo_u16_e32 v4, 9, v3
000000001E28:	v_sub_u16_e32 v2, v2, v4
000000001E2C:	v_lshlrev_b16_e32 v2, 4, v2
000000001E30:	v_mad_u64_u32 v[94:95], s[22:23], s14, v3, v[2:3]
000000001E38:	v_or_b32_e32 v2, 0x400, v0
000000001E40:	v_lshrrev_b32_e32 v6, 6, v0
000000001E44:	v_mul_u32_u24_e32 v3, 0x1c72, v2
000000001E4C:	v_lshlrev_b32_e32 v83, 10, v6
000000001E50:	v_lshrrev_b32_e32 v3, 16, v3
000000001E54:	v_readfirstlane_b32 s2, v83
000000001E58:	v_or_b32_e32 v89, 0x1000, v83
000000001E60:	v_mul_lo_u16_e32 v4, 9, v3
000000001E64:	s_mov_b32 m0, s2
000000001E68:	v_readfirstlane_b32 s2, v89
000000001E6C:	v_or_b32_e32 v91, 0x2000, v83
000000001E74:	v_sub_u16_e32 v4, v2, v4
000000001E78:	s_and_b32 s17, s5, 0xffff
000000001E80:	s_mov_b32 s19, 0x20000
000000001E88:	s_brev_b32 s18, -2
000000001E8C:	buffer_load_dwordx4 v88, s[16:19], 0 offen lds
000000001E94:	s_mov_b32 m0, s2
000000001E98:	v_readfirstlane_b32 s2, v91
000000001E9C:	v_or_b32_e32 v93, 0x3000, v83
000000001EA4:	v_lshlrev_b16_e32 v4, 4, v4
000000001EA8:	buffer_load_dwordx4 v90, s[16:19], 0 offen lds
000000001EB0:	s_mov_b32 m0, s2
000000001EB4:	v_readfirstlane_b32 s2, v93
000000001EB8:	v_mad_u64_u32 v[4:5], s[22:23], s14, v3, v[4:5]
000000001EC0:	v_or_b32_e32 v95, 0x4000, v83
000000001EC8:	v_or_b32_e32 v3, 0x500, v0
000000001ED0:	buffer_load_dwordx4 v92, s[16:19], 0 offen lds
000000001ED8:	s_mov_b32 m0, s2
000000001EDC:	buffer_load_dwordx4 v94, s[16:19], 0 offen lds
000000001EE4:	v_readfirstlane_b32 s2, v95
000000001EE8:	s_mov_b32 m0, s2
000000001EEC:	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
000000001EF4:	v_mul_u32_u24_e32 v4, 0x1c72, v3
000000001EFC:	v_lshrrev_b32_e32 v5, 16, v4
000000001F00:	v_mul_lo_u16_e32 v4, 9, v5
000000001F04:	v_sub_u16_e32 v3, v3, v4
000000001F08:	v_lshlrev_b16_e32 v4, 4, v3
000000001F0C:	v_or_b32_e32 v3, 0x600, v0
000000001F14:	v_mad_u64_u32 v[96:97], s[22:23], s14, v5, v[4:5]
000000001F1C:	v_mul_u32_u24_e32 v4, 0x1c72, v3
000000001F24:	v_lshrrev_b32_e32 v5, 16, v4
000000001F28:	v_mul_lo_u16_e32 v4, 9, v5
000000001F2C:	v_sub_u16_e32 v3, v3, v4
000000001F30:	v_lshlrev_b16_e32 v4, 4, v3
000000001F34:	v_or_b32_e32 v3, 0x700, v0
000000001F3C:	v_mad_u64_u32 v[98:99], s[22:23], s14, v5, v[4:5]
000000001F44:	v_mul_u32_u24_e32 v4, 0x1c72, v3
000000001F4C:	v_lshrrev_b32_e32 v5, 16, v4
000000001F50:	v_mul_lo_u16_e32 v4, 9, v5
000000001F54:	v_sub_u16_e32 v3, v3, v4
000000001F58:	v_lshlrev_b16_e32 v4, 4, v3
000000001F5C:	v_or_b32_e32 v3, 0x800, v0
000000001F64:	v_mad_u64_u32 v[100:101], s[22:23], s14, v5, v[4:5]
000000001F6C:	v_mul_u32_u24_e32 v4, 0x1c72, v3
000000001F74:	v_lshrrev_b32_e32 v5, 16, v4
000000001F78:	v_or_b32_e32 v99, 0x6000, v83
000000001F80:	v_mul_lo_u16_e32 v4, 9, v5
000000001F84:	v_or_b32_e32 v97, 0x5000, v83
000000001F8C:	v_readfirstlane_b32 s15, v99
000000001F90:	v_or_b32_e32 v101, 0x7000, v83
000000001F98:	v_sub_u16_e32 v4, v3, v4
000000001F9C:	v_readfirstlane_b32 s2, v97
000000001FA0:	s_mov_b32 m0, s2
000000001FA4:	buffer_load_dwordx4 v96, s[16:19], 0 offen lds
000000001FAC:	s_mov_b32 m0, s15
000000001FB0:	v_readfirstlane_b32 s15, v101
000000001FB4:	v_lshlrev_b16_e32 v4, 4, v4
000000001FB8:	v_or_b32_e32 v108, 0x8000, v83
000000001FC0:	buffer_load_dwordx4 v98, s[16:19], 0 offen lds
000000001FC8:	s_mov_b32 m0, s15
000000001FCC:	buffer_load_dwordx4 v100, s[16:19], 0 offen lds
000000001FD4:	v_mad_u64_u32 v[4:5], s[22:23], s14, v5, v[4:5]
000000001FDC:	v_readfirstlane_b32 s15, v108
000000001FE0:	s_mov_b32 m0, s15
000000001FE4:	buffer_load_dwordx4 v4, s[16:19], 0 offen lds
000000001FEC:	s_movk_i32 s5, 0x1c72
000000001FF0:	s_movk_i32 s2, 0x600
000000001FF4:	s_cmp_lt_i32 s13, 6
000000001FF8:	v_lshlrev_b32_e32 v86, 4, v75
000000001FFC:	v_lshlrev_b32_e32 v84, 3, v75
000000002000:	v_lshrrev_b32_e32 v71, 5, v75
000000002004:	s_cbranch_scc1 1639
000000002008:	v_lshlrev_b32_e32 v4, 2, v1
00000000200C:	v_lshl_or_b32 v4, s3, 3, v4
000000002014:	v_mul_lo_u32 v120, s13, v4
00000000201C:	v_mov_b32_e32 v8, 9
000000002020:	s_mov_b32 s15, 0x1c71c71d
000000002028:	v_or_b32_e32 v7, 3, v4
00000000202C:	v_mul_lo_u32 v118, s13, v7
000000002034:	v_or_b32_e32 v7, 2, v4
000000002038:	v_and_b32_e32 v5, 0x9f, v0
000000002040:	v_mul_lo_u32 v119, s13, v7
000000002048:	v_mul_hi_u32 v7, v3, s15
000000002050:	v_mul_u32_u24_e32 v5, 0x90, v5
000000002058:	v_mul_u32_u24_sdwa v4, v3, s5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
000000002060:	v_mul_lo_u16_sdwa v4, v4, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
000000002068:	v_sub_u16_e32 v3, v3, v4
00000000206C:	v_lshlrev_b16_e32 v4, 4, v3
000000002070:	v_mad_u64_u32 v[102:103], s[22:23], s14, v7, v[4:5]
000000002078:	v_mul_u32_u24_sdwa v4, v2, s5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
000000002080:	v_mul_lo_u16_sdwa v4, v4, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
000000002088:	v_mov_b32_e32 v87, 0
00000000208C:	v_mad_u32_u24 v6, v71, 24, 0
000000002094:	v_or_b32_e32 v109, 0x9000, v83
00000000209C:	v_mul_hi_u32 v3, v2, s15
0000000020A4:	v_sub_u16_e32 v2, v2, v4
0000000020A8:	v_lshlrev_b16_e32 v2, 4, v2
0000000020AC:	v_mad_u64_u32 v[104:105], s[14:15], s14, v3, v[2:3]
0000000020B4:	v_or_b32_e32 v110, 0xa000, v83
0000000020BC:	v_or_b32_e32 v111, 0xb000, v83
0000000020C4:	v_add_u32_e32 v2, s4, v1
0000000020C8:	v_mul_lo_u32 v2, s1, v2
0000000020D0:	v_or_b32_e32 v112, 0xc000, v83
0000000020D8:	v_or_b32_e32 v113, 0xd000, v83
0000000020E0:	v_or_b32_e32 v114, 0xe000, v83
0000000020E8:	v_or_b32_e32 v115, 0xf000, v83
0000000020F0:	v_or_b32_e32 v116, 0x10000, v83
0000000020F8:	v_or_b32_e32 v117, 0x11000, v83
000000002100:	v_mov_b32_e32 v85, v87
000000002104:	v_add_u32_e32 v121, s13, v120
000000002108:	v_lshlrev_b32_e32 v103, 6, v2
00000000210C:	s_mov_b32 s5, 0
000000002110:	v_accvgpr_write_b32 a127, 0
000000002118:	v_accvgpr_write_b32 a126, 0
000000002120:	v_accvgpr_write_b32 a125, 0
000000002128:	v_accvgpr_write_b32 a124, 0
000000002130:	v_accvgpr_write_b32 a123, 0
000000002138:	v_accvgpr_write_b32 a122, 0
000000002140:	v_accvgpr_write_b32 a121, 0
000000002148:	v_accvgpr_write_b32 a120, 0
000000002150:	v_accvgpr_write_b32 a119, 0
000000002158:	v_accvgpr_write_b32 a118, 0
000000002160:	v_accvgpr_write_b32 a117, 0
000000002168:	v_accvgpr_write_b32 a116, 0
000000002170:	v_accvgpr_write_b32 a115, 0
000000002178:	v_accvgpr_write_b32 a114, 0
000000002180:	v_accvgpr_write_b32 a113, 0
000000002188:	v_accvgpr_write_b32 a112, 0
000000002190:	v_accvgpr_write_b32 a175, 0
000000002198:	v_accvgpr_write_b32 a174, 0
0000000021A0:	v_accvgpr_write_b32 a173, 0
0000000021A8:	v_accvgpr_write_b32 a172, 0
0000000021B0:	v_accvgpr_write_b32 a171, 0
0000000021B8:	v_accvgpr_write_b32 a170, 0
0000000021C0:	v_accvgpr_write_b32 a169, 0
0000000021C8:	v_accvgpr_write_b32 a168, 0
0000000021D0:	v_accvgpr_write_b32 a167, 0
0000000021D8:	v_accvgpr_write_b32 a166, 0
0000000021E0:	v_accvgpr_write_b32 a165, 0
0000000021E8:	v_accvgpr_write_b32 a164, 0
0000000021F0:	v_accvgpr_write_b32 a163, 0
0000000021F8:	v_accvgpr_write_b32 a162, 0
000000002200:	v_accvgpr_write_b32 a161, 0
000000002208:	v_accvgpr_write_b32 a160, 0
000000002210:	v_accvgpr_write_b32 a223, 0
000000002218:	v_accvgpr_write_b32 a222, 0
000000002220:	v_accvgpr_write_b32 a221, 0
000000002228:	v_accvgpr_write_b32 a220, 0
000000002230:	v_accvgpr_write_b32 a219, 0
000000002238:	v_accvgpr_write_b32 a218, 0
000000002240:	v_accvgpr_write_b32 a217, 0
000000002248:	v_accvgpr_write_b32 a216, 0
000000002250:	v_accvgpr_write_b32 a215, 0
000000002258:	v_accvgpr_write_b32 a214, 0
000000002260:	v_accvgpr_write_b32 a213, 0
000000002268:	v_accvgpr_write_b32 a212, 0
000000002270:	v_accvgpr_write_b32 a211, 0
000000002278:	v_accvgpr_write_b32 a210, 0
000000002280:	v_accvgpr_write_b32 a209, 0
000000002288:	v_accvgpr_write_b32 a208, 0
000000002290:	v_accvgpr_write_b32 a255, 0
000000002298:	v_accvgpr_write_b32 a254, 0
0000000022A0:	v_accvgpr_write_b32 a253, 0
0000000022A8:	v_accvgpr_write_b32 a252, 0
0000000022B0:	v_accvgpr_write_b32 a251, 0
0000000022B8:	v_accvgpr_write_b32 a250, 0
0000000022C0:	v_accvgpr_write_b32 a249, 0
0000000022C8:	v_accvgpr_write_b32 a248, 0
0000000022D0:	v_accvgpr_write_b32 a247, 0
0000000022D8:	v_accvgpr_write_b32 a246, 0
0000000022E0:	v_accvgpr_write_b32 a245, 0
0000000022E8:	v_accvgpr_write_b32 a244, 0
0000000022F0:	v_accvgpr_write_b32 a243, 0
0000000022F8:	v_accvgpr_write_b32 a242, 0
000000002300:	v_accvgpr_write_b32 a241, 0
000000002308:	v_accvgpr_write_b32 a240, 0
000000002310:	v_accvgpr_write_b32 a95, 0
000000002318:	v_accvgpr_write_b32 a94, 0
000000002320:	v_accvgpr_write_b32 a93, 0
000000002328:	v_accvgpr_write_b32 a92, 0
000000002330:	v_accvgpr_write_b32 a91, 0
000000002338:	v_accvgpr_write_b32 a90, 0
000000002340:	v_accvgpr_write_b32 a89, 0
000000002348:	v_accvgpr_write_b32 a88, 0
000000002350:	v_accvgpr_write_b32 a87, 0
000000002358:	v_accvgpr_write_b32 a86, 0
000000002360:	v_accvgpr_write_b32 a85, 0
000000002368:	v_accvgpr_write_b32 a84, 0
000000002370:	v_accvgpr_write_b32 a83, 0
000000002378:	v_accvgpr_write_b32 a82, 0
000000002380:	v_accvgpr_write_b32 a81, 0
000000002388:	v_accvgpr_write_b32 a80, 0
000000002390:	v_accvgpr_write_b32 a143, 0
000000002398:	v_accvgpr_write_b32 a142, 0
0000000023A0:	v_accvgpr_write_b32 a141, 0
0000000023A8:	v_accvgpr_write_b32 a140, 0
0000000023B0:	v_accvgpr_write_b32 a139, 0
0000000023B8:	v_accvgpr_write_b32 a138, 0
0000000023C0:	v_accvgpr_write_b32 a137, 0
0000000023C8:	v_accvgpr_write_b32 a136, 0
0000000023D0:	v_accvgpr_write_b32 a135, 0
0000000023D8:	v_accvgpr_write_b32 a134, 0
0000000023E0:	v_accvgpr_write_b32 a133, 0
0000000023E8:	v_accvgpr_write_b32 a132, 0
0000000023F0:	v_accvgpr_write_b32 a131, 0
0000000023F8:	v_accvgpr_write_b32 a130, 0
000000002400:	v_accvgpr_write_b32 a129, 0
000000002408:	v_accvgpr_write_b32 a128, 0
000000002410:	v_accvgpr_write_b32 a191, 0
000000002418:	v_accvgpr_write_b32 a190, 0
000000002420:	v_accvgpr_write_b32 a189, 0
000000002428:	v_accvgpr_write_b32 a188, 0
000000002430:	v_accvgpr_write_b32 a187, 0
000000002438:	v_accvgpr_write_b32 a186, 0
000000002440:	v_accvgpr_write_b32 a185, 0
000000002448:	v_accvgpr_write_b32 a184, 0
000000002450:	v_accvgpr_write_b32 a183, 0
000000002458:	v_accvgpr_write_b32 a182, 0
000000002460:	v_accvgpr_write_b32 a181, 0
000000002468:	v_accvgpr_write_b32 a180, 0
000000002470:	v_accvgpr_write_b32 a179, 0
000000002478:	v_accvgpr_write_b32 a178, 0
000000002480:	v_accvgpr_write_b32 a177, 0
000000002488:	v_accvgpr_write_b32 a176, 0
000000002490:	v_accvgpr_write_b32 a239, 0
000000002498:	v_accvgpr_write_b32 a238, 0
0000000024A0:	v_accvgpr_write_b32 a237, 0
0000000024A8:	v_accvgpr_write_b32 a236, 0
0000000024B0:	v_accvgpr_write_b32 a235, 0
0000000024B8:	v_accvgpr_write_b32 a234, 0
0000000024C0:	v_accvgpr_write_b32 a233, 0
0000000024C8:	v_accvgpr_write_b32 a232, 0
0000000024D0:	v_accvgpr_write_b32 a231, 0
0000000024D8:	v_accvgpr_write_b32 a230, 0
0000000024E0:	v_accvgpr_write_b32 a229, 0
0000000024E8:	v_accvgpr_write_b32 a228, 0
0000000024F0:	v_accvgpr_write_b32 a227, 0
0000000024F8:	v_accvgpr_write_b32 a226, 0
000000002500:	v_accvgpr_write_b32 a225, 0
000000002508:	v_accvgpr_write_b32 a224, 0
000000002510:	v_accvgpr_write_b32 a47, 0
000000002518:	v_accvgpr_write_b32 a46, 0
000000002520:	v_accvgpr_write_b32 a45, 0
000000002528:	v_accvgpr_write_b32 a44, 0
000000002530:	v_accvgpr_write_b32 a43, 0
000000002538:	v_accvgpr_write_b32 a42, 0
000000002540:	v_accvgpr_write_b32 a41, 0
000000002548:	v_accvgpr_write_b32 a40, 0
000000002550:	v_accvgpr_write_b32 a39, 0
000000002558:	v_accvgpr_write_b32 a38, 0
000000002560:	v_accvgpr_write_b32 a37, 0
000000002568:	v_accvgpr_write_b32 a36, 0
000000002570:	v_accvgpr_write_b32 a35, 0
000000002578:	v_accvgpr_write_b32 a34, 0
000000002580:	v_accvgpr_write_b32 a33, 0
000000002588:	v_accvgpr_write_b32 a32, 0
000000002590:	v_accvgpr_write_b32 a111, 0
000000002598:	v_accvgpr_write_b32 a110, 0
0000000025A0:	v_accvgpr_write_b32 a109, 0
0000000025A8:	v_accvgpr_write_b32 a108, 0
0000000025B0:	v_accvgpr_write_b32 a107, 0
0000000025B8:	v_accvgpr_write_b32 a106, 0
0000000025C0:	v_accvgpr_write_b32 a105, 0
0000000025C8:	v_accvgpr_write_b32 a104, 0
0000000025D0:	v_accvgpr_write_b32 a103, 0
0000000025D8:	v_accvgpr_write_b32 a102, 0
0000000025E0:	v_accvgpr_write_b32 a101, 0
0000000025E8:	v_accvgpr_write_b32 a100, 0
0000000025F0:	v_accvgpr_write_b32 a99, 0
0000000025F8:	v_accvgpr_write_b32 a98, 0
000000002600:	v_accvgpr_write_b32 a97, 0
000000002608:	v_accvgpr_write_b32 a96, 0
000000002610:	v_accvgpr_write_b32 a159, 0
000000002618:	v_accvgpr_write_b32 a158, 0
000000002620:	v_accvgpr_write_b32 a157, 0
000000002628:	v_accvgpr_write_b32 a156, 0
000000002630:	v_accvgpr_write_b32 a155, 0
000000002638:	v_accvgpr_write_b32 a154, 0
000000002640:	v_accvgpr_write_b32 a153, 0
000000002648:	v_accvgpr_write_b32 a152, 0
000000002650:	v_accvgpr_write_b32 a151, 0
000000002658:	v_accvgpr_write_b32 a150, 0
000000002660:	v_accvgpr_write_b32 a149, 0
000000002668:	v_accvgpr_write_b32 a148, 0
000000002670:	v_accvgpr_write_b32 a147, 0
000000002678:	v_accvgpr_write_b32 a146, 0
000000002680:	v_accvgpr_write_b32 a145, 0
000000002688:	v_accvgpr_write_b32 a144, 0
000000002690:	v_accvgpr_write_b32 a207, 0
000000002698:	v_accvgpr_write_b32 a206, 0
0000000026A0:	v_accvgpr_write_b32 a205, 0
0000000026A8:	v_accvgpr_write_b32 a204, 0
0000000026B0:	v_accvgpr_write_b32 a203, 0
0000000026B8:	v_accvgpr_write_b32 a202, 0
0000000026C0:	v_accvgpr_write_b32 a201, 0
0000000026C8:	v_accvgpr_write_b32 a200, 0
0000000026D0:	v_accvgpr_write_b32 a199, 0
0000000026D8:	v_accvgpr_write_b32 a198, 0
0000000026E0:	v_accvgpr_write_b32 a197, 0
0000000026E8:	v_accvgpr_write_b32 a196, 0
0000000026F0:	v_accvgpr_write_b32 a195, 0
0000000026F8:	v_accvgpr_write_b32 a194, 0
000000002700:	v_accvgpr_write_b32 a193, 0
000000002708:	v_accvgpr_write_b32 a192, 0
000000002710:	v_accvgpr_write_b32 a31, 0
000000002718:	v_accvgpr_write_b32 a30, 0
000000002720:	v_accvgpr_write_b32 a29, 0
000000002728:	v_accvgpr_write_b32 a28, 0
000000002730:	v_accvgpr_write_b32 a27, 0
000000002738:	v_accvgpr_write_b32 a26, 0
000000002740:	v_accvgpr_write_b32 a25, 0
000000002748:	v_accvgpr_write_b32 a24, 0
000000002750:	v_accvgpr_write_b32 a23, 0
000000002758:	v_accvgpr_write_b32 a22, 0
000000002760:	v_accvgpr_write_b32 a21, 0
000000002768:	v_accvgpr_write_b32 a20, 0
000000002770:	v_accvgpr_write_b32 a19, 0
000000002778:	v_accvgpr_write_b32 a18, 0
000000002780:	v_accvgpr_write_b32 a17, 0
000000002788:	v_accvgpr_write_b32 a16, 0
000000002790:	v_accvgpr_write_b32 a79, 0
000000002798:	v_accvgpr_write_b32 a78, 0
0000000027A0:	v_accvgpr_write_b32 a77, 0
0000000027A8:	v_accvgpr_write_b32 a76, 0
0000000027B0:	v_accvgpr_write_b32 a75, 0
0000000027B8:	v_accvgpr_write_b32 a74, 0
0000000027C0:	v_accvgpr_write_b32 a73, 0
0000000027C8:	v_accvgpr_write_b32 a72, 0
0000000027D0:	v_accvgpr_write_b32 a71, 0
0000000027D8:	v_accvgpr_write_b32 a70, 0
0000000027E0:	v_accvgpr_write_b32 a69, 0
0000000027E8:	v_accvgpr_write_b32 a68, 0
0000000027F0:	v_accvgpr_write_b32 a67, 0
0000000027F8:	v_accvgpr_write_b32 a66, 0
000000002800:	v_accvgpr_write_b32 a65, 0
000000002808:	v_accvgpr_write_b32 a64, 0
000000002810:	v_accvgpr_write_b32 a63, 0
000000002818:	v_accvgpr_write_b32 a62, 0
000000002820:	v_accvgpr_write_b32 a61, 0
000000002828:	v_accvgpr_write_b32 a60, 0
000000002830:	v_accvgpr_write_b32 a59, 0
000000002838:	v_accvgpr_write_b32 a58, 0
000000002840:	v_accvgpr_write_b32 a57, 0
000000002848:	v_accvgpr_write_b32 a56, 0
000000002850:	v_accvgpr_write_b32 a55, 0
000000002858:	v_accvgpr_write_b32 a54, 0
000000002860:	v_accvgpr_write_b32 a53, 0
000000002868:	v_accvgpr_write_b32 a52, 0
000000002870:	v_accvgpr_write_b32 a51, 0
000000002878:	v_accvgpr_write_b32 a50, 0
000000002880:	v_accvgpr_write_b32 a49, 0
000000002888:	v_accvgpr_write_b32 a48, 0
000000002890:	v_accvgpr_write_b32 a15, 0
000000002898:	v_accvgpr_write_b32 a14, 0
0000000028A0:	v_accvgpr_write_b32 a13, 0
0000000028A8:	v_accvgpr_write_b32 a12, 0
0000000028B0:	v_accvgpr_write_b32 a11, 0
0000000028B8:	v_accvgpr_write_b32 a10, 0
0000000028C0:	v_accvgpr_write_b32 a9, 0
0000000028C8:	v_accvgpr_write_b32 a8, 0
0000000028D0:	v_accvgpr_write_b32 a7, 0
0000000028D8:	v_accvgpr_write_b32 a6, 0
0000000028E0:	v_accvgpr_write_b32 a5, 0
0000000028E8:	v_accvgpr_write_b32 a4, 0
0000000028F0:	v_accvgpr_write_b32 a3, 0
0000000028F8:	v_accvgpr_write_b32 a2, 0
000000002900:	v_accvgpr_write_b32 a1, 0
000000002908:	v_accvgpr_write_b32 a0, 0
000000002910:	v_add_u32_e32 v105, v6, v5
000000002914:	s_mov_b32 s14, 0
000000002918:	s_mov_b32 s4, 0
00000000291C:	s_branch 457
000000002920:	v_add_u32_e32 v2, 3, v125
000000002924:	v_mov_b64_e32 v[26:27], s[6:7]
000000002928:	v_add_u32_e32 v8, 3, v124
00000000292C:	v_mad_i64_i32 v[2:3], s[22:23], v2, s2, v[26:27]
000000002934:	v_mad_i64_i32 v[8:9], s[22:23], v8, s2, v[26:27]
00000000293C:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000002944:	v_lshl_add_u64 v[6:7], v[2:3], 0, v[84:85]
00000000294C:	v_lshl_add_u64 v[10:11], v[8:9], 0, v[86:87]
000000002954:	v_lshl_add_u64 v[8:9], v[8:9], 0, v[84:85]
00000000295C:	global_load_dwordx4 v[2:5], v[4:5], off
000000002964:	s_nop 0
000000002968:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
000000002970:	s_nop 0
000000002974:	global_load_dwordx4 v[20:23], v[10:11], off
00000000297C:	global_load_dwordx2 v[24:25], v[8:9], off offset:1024
000000002984:	v_add_u32_e32 v8, 3, v123
000000002988:	v_mad_i64_i32 v[8:9], s[22:23], v8, s2, v[26:27]
000000002990:	v_lshl_add_u64 v[10:11], v[8:9], 0, v[86:87]
000000002998:	v_lshl_add_u64 v[8:9], v[8:9], 0, v[84:85]
0000000029A0:	global_load_dwordx4 v[14:17], v[10:11], off
0000000029A8:	global_load_dwordx2 v[18:19], v[8:9], off offset:1024
0000000029B0:	v_add_u32_e32 v8, 3, v122
0000000029B4:	v_mad_i64_i32 v[8:9], s[22:23], v8, s2, v[26:27]
0000000029BC:	v_lshl_add_u64 v[10:11], v[8:9], 0, v[86:87]
0000000029C4:	v_lshl_add_u64 v[12:13], v[8:9], 0, v[84:85]
0000000029CC:	global_load_dwordx4 v[8:11], v[10:11], off
0000000029D4:	s_nop 0
0000000029D8:	global_load_dwordx2 v[12:13], v[12:13], off offset:1024
0000000029E0:	v_add_u32_e32 v28, 4, v125
0000000029E4:	v_mad_i64_i32 v[28:29], s[22:23], v28, s2, v[26:27]
0000000029EC:	v_lshl_add_u64 v[30:31], v[28:29], 0, v[86:87]
0000000029F4:	ds_read2st64_b64 v[42:45], v137 offset0:72 offset1:81
0000000029FC:	ds_read2st64_b64 v[48:51], v137 offset0:90 offset1:99
000000002A04:	v_lshl_add_u64 v[28:29], v[28:29], 0, v[84:85]
000000002A0C:	global_load_dwordx4 v[52:55], v[30:31], off
000000002A14:	global_load_dwordx2 v[56:57], v[28:29], off offset:1024
000000002A1C:	v_add_u32_e32 v34, 4, v124
000000002A20:	v_add_u32_e32 v35, 4, v123
000000002A24:	v_add_u32_e32 v36, 4, v122
000000002A28:	v_mad_i64_i32 v[28:29], s[22:23], v34, s2, v[26:27]
000000002A30:	v_mad_i64_i32 v[30:31], s[22:23], v35, s2, v[26:27]
000000002A38:	v_mad_i64_i32 v[34:35], s[22:23], v36, s2, v[26:27]
000000002A40:	v_lshl_add_u64 v[36:37], v[28:29], 0, v[86:87]
000000002A48:	v_lshl_add_u64 v[28:29], v[28:29], 0, v[84:85]
000000002A50:	global_load_dwordx4 v[58:61], v[36:37], off
000000002A58:	global_load_dwordx2 v[62:63], v[28:29], off offset:1024
000000002A60:	s_waitcnt lgkmcnt(1)
000000002A64:	v_mov_b32_e32 v130, v42
000000002A68:	v_mov_b32_e32 v131, v43
000000002A6C:	ds_read_b128 v[126:129], v105 offset:36864
000000002A74:	ds_read_b128 v[40:43], v105 offset:41472
000000002A7C:	s_waitcnt lgkmcnt(2)
000000002A80:	v_mov_b32_e32 v136, v48
000000002A84:	v_mov_b32_e32 v137, v49
000000002A88:	ds_read_b128 v[132:135], v105 offset:46080
000000002A90:	ds_read_b128 v[46:49], v105 offset:50688
000000002A98:	v_add_u32_e32 v125, 5, v125
000000002A9C:	v_lshl_add_u64 v[64:65], v[30:31], 0, v[86:87]
000000002AA4:	v_and_b32_e32 v33, 0xff, v76
000000002AAC:	v_and_b32_e32 v107, 0xff, v80
000000002AB4:	v_bfe_u32 v150, v80, 8, 8
000000002ABC:	v_bfe_u32 v151, v80, 16, 8
000000002AC4:	v_lshrrev_b32_e32 v80, 24, v80
000000002AC8:	v_lshl_add_u64 v[30:31], v[30:31], 0, v[84:85]
000000002AD0:	global_load_dwordx4 v[138:141], v[64:65], off
000000002AD8:	global_load_dwordx2 v[142:143], v[30:31], off offset:1024
000000002AE0:	v_lshl_add_u64 v[66:67], v[34:35], 0, v[86:87]
000000002AE8:	v_bfe_u32 v39, v76, 8, 8
000000002AF0:	v_bfe_u32 v106, v76, 16, 8
000000002AF8:	v_lshl_add_u64 v[34:35], v[34:35], 0, v[84:85]
000000002B00:	v_lshrrev_b32_e32 v76, 24, v76
000000002B04:	v_add_u32_e32 v28, 5, v123
000000002B08:	v_mad_i64_i32 v[28:29], s[22:23], v28, s2, v[26:27]
000000002B10:	v_lshl_add_u64 v[30:31], v[28:29], 0, v[86:87]
000000002B18:	v_lshl_add_u64 v[28:29], v[28:29], 0, v[84:85]
000000002B20:	v_and_b32_e32 v64, 0xff, v81
000000002B28:	v_bfe_u32 v65, v81, 8, 8
000000002B30:	s_add_i32 s14, s14, 6
000000002B34:	s_addk_i32 s5, 0x120
000000002B38:	s_add_i32 s15, s4, 1
000000002B3C:	v_add_u32_e32 v103, 0x80, v103
000000002B44:	s_cmp_lt_i32 s15, s1
000000002B48:	v_add_u32_e32 v79, 0x80, v79
000000002B50:	s_waitcnt vmcnt(12) lgkmcnt(3)
000000002B54:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[2:7], v[126:131], a[112:127], v33, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002B64:	s_waitcnt lgkmcnt(2)
000000002B68:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[2:7], v[40:45], a[80:95], v33, v150 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002B78:	s_waitcnt lgkmcnt(1)
000000002B7C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[2:7], v[132:137], a[32:47], v33, v151 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002B8C:	s_waitcnt lgkmcnt(0)
000000002B90:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[2:7], v[46:51], a[16:31], v33, v80 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002BA0:	v_mad_i64_i32 v[2:3], s[22:23], v125, s2, v[26:27]
000000002BA8:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000002BB0:	v_lshl_add_u64 v[6:7], v[2:3], 0, v[84:85]
000000002BB8:	global_load_dwordx4 v[144:147], v[66:67], off
000000002BC0:	global_load_dwordx2 v[148:149], v[34:35], off offset:1024
000000002BC8:	s_waitcnt vmcnt(12)
000000002BCC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[20:25], v[126:131], a[160:175], v39, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002BDC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[20:25], v[40:45], a[128:143], v39, v150 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002BEC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[20:25], v[132:137], a[96:111], v39, v151 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002BFC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[20:25], v[46:51], a[64:79], v39, v80 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002C0C:	global_load_dwordx4 v[2:5], v[4:5], off
000000002C14:	s_nop 0
000000002C18:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
000000002C20:	s_waitcnt vmcnt(12)
000000002C24:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[14:19], v[126:131], a[208:223], v106, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002C34:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[14:19], v[40:45], a[176:191], v106, v150 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002C44:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[14:19], v[132:137], a[144:159], v106, v151 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002C54:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[14:19], v[46:51], a[48:63], v106, v80 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002C64:	v_add_u32_e32 v14, 5, v124
000000002C68:	v_mad_i64_i32 v[14:15], s[22:23], v14, s2, v[26:27]
000000002C70:	v_lshl_add_u64 v[16:17], v[14:15], 0, v[86:87]
000000002C78:	v_lshl_add_u64 v[18:19], v[14:15], 0, v[84:85]
000000002C80:	global_load_dwordx4 v[14:17], v[16:17], off
000000002C88:	s_nop 0
000000002C8C:	global_load_dwordx2 v[18:19], v[18:19], off offset:1024
000000002C94:	s_waitcnt vmcnt(12)
000000002C98:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[8:13], v[126:131], a[240:255], v76, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002CA8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[8:13], v[40:45], a[224:239], v76, v150 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002CB8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[8:13], v[132:137], a[192:207], v76, v151 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002CC8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[8:13], v[46:51], a[0:15], v76, v80 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002CD8:	ds_read2st64_b64 v[10:13], v38 offset0:72 offset1:81
000000002CE0:	ds_read2st64_b64 v[36:39], v38 offset0:90 offset1:99
000000002CE8:	v_and_b32_e32 v33, 0xff, v77
000000002CF0:	v_bfe_u32 v66, v81, 16, 8
000000002CF8:	v_lshrrev_b32_e32 v67, 24, v81
000000002CFC:	s_waitcnt lgkmcnt(1)
000000002D00:	v_mov_b32_e32 v24, v10
000000002D04:	v_mov_b32_e32 v25, v11
000000002D08:	ds_read_b128 v[20:23], v105 offset:36912
000000002D10:	ds_read_b128 v[8:11], v105 offset:41520
000000002D18:	s_waitcnt lgkmcnt(2)
000000002D1C:	v_mov_b32_e32 v44, v36
000000002D20:	v_mov_b32_e32 v45, v37
000000002D24:	ds_read_b128 v[40:43], v105 offset:46128
000000002D2C:	ds_read_b128 v[34:37], v105 offset:50736
000000002D34:	global_load_dwordx4 v[46:49], v[30:31], off
000000002D3C:	global_load_dwordx2 v[50:51], v[28:29], off offset:1024
000000002D44:	v_add_u32_e32 v28, 5, v122
000000002D48:	v_mad_i64_i32 v[26:27], s[22:23], v28, s2, v[26:27]
000000002D50:	v_lshl_add_u64 v[28:29], v[26:27], 0, v[86:87]
000000002D58:	v_lshl_add_u64 v[30:31], v[26:27], 0, v[84:85]
000000002D60:	s_waitcnt vmcnt(12) lgkmcnt(3)
000000002D64:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[52:57], v[20:25], a[112:127], v33, v64 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002D74:	s_waitcnt lgkmcnt(2)
000000002D78:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[52:57], v[8:13], a[80:95], v33, v65 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002D88:	s_waitcnt lgkmcnt(1)
000000002D8C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[52:57], v[40:45], a[32:47], v33, v66 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002D9C:	s_waitcnt lgkmcnt(0)
000000002DA0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[52:57], v[34:39], a[16:31], v33, v67 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002DB0:	global_load_dwordx4 v[26:29], v[28:29], off
000000002DB8:	s_nop 0
000000002DBC:	global_load_dwordx2 v[30:31], v[30:31], off offset:1024
000000002DC4:	v_bfe_u32 v33, v77, 8, 8
000000002DCC:	v_bfe_u32 v52, v77, 16, 8
000000002DD4:	v_lshrrev_b32_e32 v53, 24, v77
000000002DD8:	s_waitcnt vmcnt(12)
000000002DDC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[58:63], v[20:25], a[160:175], v33, v64 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002DEC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[58:63], v[8:13], a[128:143], v33, v65 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002DFC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[58:63], v[40:45], a[96:111], v33, v66 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E0C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[58:63], v[34:39], a[64:79], v33, v67 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E1C:	s_waitcnt vmcnt(10)
000000002E20:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[138:143], v[20:25], a[208:223], v52, v64 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E30:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[138:143], v[8:13], a[176:191], v52, v65 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E40:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[138:143], v[40:45], a[144:159], v52, v66 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E50:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[138:143], v[34:39], a[48:63], v52, v67 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E60:	v_and_b32_e32 v54, 0xff, v78
000000002E68:	v_bfe_u32 v55, v78, 8, 8
000000002E70:	v_bfe_u32 v56, v78, 16, 8
000000002E78:	v_bfe_u32 v52, v82, 16, 8
000000002E80:	s_waitcnt vmcnt(8)
000000002E84:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[144:149], v[20:25], a[240:255], v53, v64 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002E94:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[144:149], v[8:13], a[224:239], v53, v65 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002EA4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[144:149], v[40:45], a[192:207], v53, v66 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002EB4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[144:149], v[34:39], a[0:15], v53, v67 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002EC4:	ds_read2st64_b64 v[10:13], v32 offset0:72 offset1:81
000000002ECC:	ds_read2st64_b64 v[22:25], v32 offset0:90 offset1:99
000000002ED4:	v_and_b32_e32 v44, 0xff, v82
000000002EDC:	v_bfe_u32 v45, v82, 8, 8
000000002EE4:	v_lshrrev_b32_e32 v53, 24, v82
000000002EE8:	s_waitcnt lgkmcnt(1)
000000002EEC:	v_mov_b32_e32 v36, v10
000000002EF0:	v_mov_b32_e32 v37, v11
000000002EF4:	ds_read_b128 v[32:35], v105 offset:36960
000000002EFC:	ds_read_b128 v[8:11], v105 offset:41568
000000002F04:	s_waitcnt lgkmcnt(2)
000000002F08:	v_mov_b32_e32 v42, v22
000000002F0C:	v_mov_b32_e32 v43, v23
000000002F10:	ds_read_b128 v[38:41], v105 offset:46176
000000002F18:	ds_read_b128 v[20:23], v105 offset:50784
000000002F20:	s_waitcnt vmcnt(6) lgkmcnt(3)
000000002F24:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[2:7], v[32:37], a[112:127], v54, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F34:	s_waitcnt lgkmcnt(2)
000000002F38:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[2:7], v[8:13], a[80:95], v54, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F48:	s_waitcnt lgkmcnt(1)
000000002F4C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[2:7], v[38:43], a[32:47], v54, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F5C:	s_waitcnt lgkmcnt(0)
000000002F60:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[2:7], v[20:25], a[16:31], v54, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F70:	s_waitcnt vmcnt(4)
000000002F74:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[14:19], v[32:37], a[160:175], v55, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F84:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[14:19], v[8:13], a[128:143], v55, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002F94:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[14:19], v[38:43], a[96:111], v55, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FA4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[14:19], v[20:25], a[64:79], v55, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FB4:	s_waitcnt vmcnt(2)
000000002FB8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[46:51], v[32:37], a[208:223], v56, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FC8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[46:51], v[8:13], a[176:191], v56, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FD8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[46:51], v[38:43], a[144:159], v56, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FE8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[46:51], v[20:25], a[48:63], v56, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000002FF8:	v_lshrrev_b32_e32 v2, 24, v78
000000002FFC:	s_waitcnt vmcnt(0)
000000003000:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[26:31], v[32:37], a[240:255], v2, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003010:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v2, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003020:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[26:31], v[38:43], a[192:207], v2, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003030:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:31], v[20:25], a[0:15], v2, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003040:	s_cbranch_scc0 858
000000003044:	v_add_u32_e32 v130, v75, v79
000000003048:	v_add_u32_e32 v2, 64, v130
00000000304C:	v_mad_i64_i32 v[2:3], s[22:23], v2, 12, s[8:9]
000000003054:	v_add_u32_e32 v132, v75, v103
000000003058:	v_add_u32_e32 v126, s5, v88
00000000305C:	s_barrier
000000003060:	v_add_u32_e32 v4, 64, v132
000000003064:	global_load_dwordx3 v[80:82], v[2:3], off
00000000306C:	v_add_u32_e32 v2, 0x90, v126
000000003074:	v_add_u32_e32 v127, s5, v90
000000003078:	v_mad_i64_i32 v[4:5], s[22:23], v4, 12, s[10:11]
000000003080:	global_load_dwordx3 v[76:78], v[4:5], off
000000003088:	v_readfirstlane_b32 s15, v109
00000000308C:	s_mov_b32 m0, s15
000000003090:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003098:	v_add_u32_e32 v2, 0x90, v127
0000000030A0:	v_add_u32_e32 v128, s5, v92
0000000030A4:	v_readfirstlane_b32 s15, v110
0000000030A8:	s_mov_b32 m0, s15
0000000030AC:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000030B4:	v_add_u32_e32 v2, 0x90, v128
0000000030BC:	v_add_u32_e32 v129, s5, v94
0000000030C0:	v_readfirstlane_b32 s15, v111
0000000030C4:	s_mov_b32 m0, s15
0000000030C8:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000030D0:	v_add_u32_e32 v2, 0x90, v129
0000000030D8:	v_add_u32_e32 v131, s5, v104
0000000030DC:	v_readfirstlane_b32 s15, v112
0000000030E0:	s_mov_b32 m0, s15
0000000030E4:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000030EC:	v_add_u32_e32 v2, 0x90, v131
0000000030F4:	v_add_u32_e32 v133, s5, v96
0000000030F8:	v_readfirstlane_b32 s15, v113
0000000030FC:	s_mov_b32 m0, s15
000000003100:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003108:	v_add_u32_e32 v2, 0x90, v133
000000003110:	v_add_u32_e32 v134, s5, v98
000000003114:	v_readfirstlane_b32 s15, v114
000000003118:	s_mov_b32 m0, s15
00000000311C:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003124:	v_add_u32_e32 v2, 0x90, v134
00000000312C:	v_add_u32_e32 v135, s5, v100
000000003130:	v_readfirstlane_b32 s15, v115
000000003134:	s_mov_b32 m0, s15
000000003138:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003140:	v_add_u32_e32 v2, 0x90, v135
000000003148:	v_add_u32_e32 v136, s5, v102
00000000314C:	v_readfirstlane_b32 s15, v116
000000003150:	s_mov_b32 m0, s15
000000003154:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
00000000315C:	v_add_u32_e32 v2, 0x90, v136
000000003164:	v_add_u32_e32 v125, s14, v120
000000003168:	v_mov_b64_e32 v[106:107], s[6:7]
00000000316C:	v_readfirstlane_b32 s15, v117
000000003170:	s_mov_b32 m0, s15
000000003174:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
00000000317C:	v_mad_i64_i32 v[2:3], s[22:23], v125, s2, v[106:107]
000000003184:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
00000000318C:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003194:	global_load_dwordx4 v[32:35], v[4:5], off
00000000319C:	global_load_dwordx2 v[36:37], v[2:3], off offset:1024
0000000031A4:	v_add_u32_e32 v124, s14, v121
0000000031A8:	v_mad_i64_i32 v[2:3], s[22:23], v124, s2, v[106:107]
0000000031B0:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
0000000031B8:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
0000000031C0:	global_load_dwordx4 v[50:53], v[4:5], off
0000000031C8:	global_load_dwordx2 v[54:55], v[2:3], off offset:1024
0000000031D0:	v_add_u32_e32 v123, s14, v119
0000000031D4:	v_mad_i64_i32 v[2:3], s[22:23], v123, s2, v[106:107]
0000000031DC:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
0000000031E4:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
0000000031EC:	global_load_dwordx4 v[26:29], v[4:5], off
0000000031F4:	global_load_dwordx2 v[30:31], v[2:3], off offset:1024
0000000031FC:	v_add_u32_e32 v122, s14, v118
000000003200:	v_mad_i64_i32 v[2:3], s[22:23], v122, s2, v[106:107]
000000003208:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003210:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003218:	global_load_dwordx4 v[14:17], v[4:5], off
000000003220:	global_load_dwordx2 v[18:19], v[2:3], off offset:1024
000000003228:	v_add_u32_e32 v2, 1, v125
00000000322C:	v_add_u32_e32 v8, 1, v124
000000003230:	v_mad_i64_i32 v[6:7], s[22:23], v2, s2, v[106:107]
000000003238:	v_mad_i64_i32 v[8:9], s[22:23], v8, s2, v[106:107]
000000003240:	v_lshl_add_u64 v[2:3], v[6:7], 0, v[86:87]
000000003248:	v_lshl_add_u64 v[6:7], v[6:7], 0, v[84:85]
000000003250:	v_add_u32_e32 v137, 16, v105
000000003254:	v_lshl_add_u64 v[10:11], v[8:9], 0, v[86:87]
00000000325C:	v_lshl_add_u64 v[12:13], v[8:9], 0, v[84:85]
000000003264:	global_load_dwordx4 v[2:5], v[2:3], off
00000000326C:	ds_read2st64_b64 v[46:49], v137 offset1:9
000000003274:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
00000000327C:	s_nop 0
000000003280:	global_load_dwordx4 v[8:11], v[10:11], off
000000003288:	s_nop 0
00000000328C:	global_load_dwordx2 v[12:13], v[12:13], off offset:1024
000000003294:	ds_read2st64_b64 v[40:43], v137 offset0:18 offset1:27
00000000329C:	v_add_u32_e32 v20, 1, v123
0000000032A0:	s_waitcnt lgkmcnt(1)
0000000032A4:	v_mov_b32_e32 v66, v46
0000000032A8:	v_mov_b32_e32 v67, v47
0000000032AC:	ds_read_b128 v[62:65], v105
0000000032B4:	ds_read_b128 v[44:47], v105 offset:4608
0000000032BC:	s_waitcnt lgkmcnt(2)
0000000032C0:	v_mov_b32_e32 v60, v40
0000000032C4:	v_mov_b32_e32 v61, v41
0000000032C8:	ds_read_b128 v[56:59], v105 offset:9216
0000000032D0:	ds_read_b128 v[38:41], v105 offset:13824
0000000032D8:	v_mad_i64_i32 v[20:21], s[22:23], v20, s2, v[106:107]
0000000032E0:	v_lshl_add_u64 v[22:23], v[20:21], 0, v[86:87]
0000000032E8:	v_lshl_add_u64 v[24:25], v[20:21], 0, v[84:85]
0000000032F0:	v_and_b32_e32 v138, 0xff, v68
0000000032F8:	v_and_b32_e32 v142, 0xff, v72
000000003300:	v_bfe_u32 v143, v72, 8, 8
000000003308:	v_bfe_u32 v144, v72, 16, 8
000000003310:	v_lshrrev_b32_e32 v145, 24, v72
000000003314:	global_load_dwordx4 v[20:23], v[22:23], off
00000000331C:	s_nop 0
000000003320:	global_load_dwordx2 v[24:25], v[24:25], off offset:1024
000000003328:	s_waitcnt vmcnt(12) lgkmcnt(3)
00000000332C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[62:67], a[112:127], v138, v142 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000333C:	s_waitcnt lgkmcnt(2)
000000003340:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[32:37], v[44:49], a[80:95], v138, v143 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003350:	s_waitcnt lgkmcnt(1)
000000003354:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[32:37], v[56:61], a[32:47], v138, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003364:	s_waitcnt lgkmcnt(0)
000000003368:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[38:43], a[16:31], v138, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003378:	v_add_u32_e32 v32, 1, v122
00000000337C:	v_mad_i64_i32 v[32:33], s[22:23], v32, s2, v[106:107]
000000003384:	v_lshl_add_u64 v[34:35], v[32:33], 0, v[86:87]
00000000338C:	v_lshl_add_u64 v[36:37], v[32:33], 0, v[84:85]
000000003394:	v_bfe_u32 v139, v68, 8, 8
00000000339C:	global_load_dwordx4 v[32:35], v[34:35], off
0000000033A4:	s_nop 0
0000000033A8:	global_load_dwordx2 v[36:37], v[36:37], off offset:1024
0000000033B0:	s_waitcnt vmcnt(12)
0000000033B4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[50:55], v[62:67], a[160:175], v139, v142 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000033C4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[50:55], v[44:49], a[128:143], v139, v143 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000033D4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[50:55], v[56:61], a[96:111], v139, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000033E4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[50:55], v[38:43], a[64:79], v139, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000033F4:	v_add_u32_e32 v50, 2, v125
0000000033F8:	v_mad_i64_i32 v[50:51], s[22:23], v50, s2, v[106:107]
000000003400:	v_lshl_add_u64 v[52:53], v[50:51], 0, v[86:87]
000000003408:	v_lshl_add_u64 v[54:55], v[50:51], 0, v[84:85]
000000003410:	v_bfe_u32 v140, v68, 16, 8
000000003418:	global_load_dwordx4 v[50:53], v[52:53], off
000000003420:	s_nop 0
000000003424:	global_load_dwordx2 v[54:55], v[54:55], off offset:1024
00000000342C:	s_waitcnt vmcnt(12)
000000003430:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[26:31], v[62:67], a[208:223], v140, v142 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003440:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[26:31], v[44:49], a[176:191], v140, v143 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003450:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[26:31], v[56:61], a[144:159], v140, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003460:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[38:43], a[48:63], v140, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003470:	v_add_u32_e32 v26, 2, v124
000000003474:	v_mad_i64_i32 v[26:27], s[22:23], v26, s2, v[106:107]
00000000347C:	v_lshl_add_u64 v[28:29], v[26:27], 0, v[86:87]
000000003484:	v_lshl_add_u64 v[30:31], v[26:27], 0, v[84:85]
00000000348C:	v_lshrrev_b32_e32 v141, 24, v68
000000003490:	global_load_dwordx4 v[26:29], v[28:29], off
000000003498:	s_nop 0
00000000349C:	global_load_dwordx2 v[30:31], v[30:31], off offset:1024
0000000034A4:	s_waitcnt vmcnt(12)
0000000034A8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[14:19], v[62:67], a[240:255], v141, v142 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000034B8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[14:19], v[44:49], a[224:239], v141, v143 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000034C8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[14:19], v[56:61], a[192:207], v141, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000034D8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[14:19], v[38:43], a[0:15], v141, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000034E8:	v_add_u32_e32 v38, 64, v105
0000000034EC:	ds_read2st64_b64 v[16:19], v38 offset1:9
0000000034F4:	ds_read2st64_b64 v[58:61], v38 offset0:18 offset1:27
0000000034FC:	v_add_u32_e32 v46, 2, v123
000000003500:	v_mad_i64_i32 v[46:47], s[22:23], v46, s2, v[106:107]
000000003508:	s_waitcnt lgkmcnt(1)
00000000350C:	v_mov_b32_e32 v44, v16
000000003510:	v_mov_b32_e32 v45, v17
000000003514:	ds_read_b128 v[40:43], v105 offset:48
00000000351C:	ds_read_b128 v[14:17], v105 offset:4656
000000003524:	s_waitcnt lgkmcnt(2)
000000003528:	v_mov_b32_e32 v66, v58
00000000352C:	v_mov_b32_e32 v67, v59
000000003530:	ds_read_b128 v[62:65], v105 offset:9264
000000003538:	ds_read_b128 v[56:59], v105 offset:13872
000000003540:	v_lshl_add_u64 v[48:49], v[46:47], 0, v[86:87]
000000003548:	v_and_b32_e32 v39, 0xff, v69
000000003550:	v_and_b32_e32 v144, 0xff, v73
000000003558:	v_bfe_u32 v145, v73, 8, 8
000000003560:	v_bfe_u32 v146, v73, 16, 8
000000003568:	v_lshrrev_b32_e32 v147, 24, v73
00000000356C:	v_lshl_add_u64 v[46:47], v[46:47], 0, v[84:85]
000000003574:	global_load_dwordx4 v[138:141], v[48:49], off
00000000357C:	global_load_dwordx2 v[142:143], v[46:47], off offset:1024
000000003584:	s_waitcnt vmcnt(12) lgkmcnt(3)
000000003588:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[2:7], v[40:45], a[112:127], v39, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003598:	s_waitcnt lgkmcnt(2)
00000000359C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[2:7], v[14:19], a[80:95], v39, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000035AC:	s_waitcnt lgkmcnt(1)
0000000035B0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[2:7], v[62:67], a[32:47], v39, v146 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000035C0:	s_waitcnt lgkmcnt(0)
0000000035C4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[2:7], v[56:61], a[16:31], v39, v147 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000035D4:	v_add_u32_e32 v2, 2, v122
0000000035D8:	v_mad_i64_i32 v[2:3], s[22:23], v2, s2, v[106:107]
0000000035E0:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
0000000035E8:	v_lshl_add_u64 v[6:7], v[2:3], 0, v[84:85]
0000000035F0:	global_load_dwordx4 v[2:5], v[4:5], off
0000000035F8:	s_nop 0
0000000035FC:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
000000003604:	v_bfe_u32 v39, v69, 8, 8
00000000360C:	v_bfe_u32 v46, v69, 16, 8
000000003614:	v_lshrrev_b32_e32 v47, 24, v69
000000003618:	s_waitcnt vmcnt(12)
00000000361C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[8:13], v[40:45], a[160:175], v39, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000362C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[8:13], v[14:19], a[128:143], v39, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000363C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[8:13], v[62:67], a[96:111], v39, v146 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000364C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[8:13], v[56:61], a[64:79], v39, v147 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000365C:	s_waitcnt vmcnt(10)
000000003660:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[20:25], v[40:45], a[208:223], v46, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003670:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[20:25], v[14:19], a[176:191], v46, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003680:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[20:25], v[62:67], a[144:159], v46, v146 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003690:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[20:25], v[56:61], a[48:63], v46, v147 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000036A0:	s_waitcnt vmcnt(8)
0000000036A4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[40:45], a[240:255], v47, v144 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000036B4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[32:37], v[14:19], a[224:239], v47, v145 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000036C4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[32:37], v[62:67], a[192:207], v47, v146 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000036D4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[56:61], a[0:15], v47, v147 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000036E4:	v_add_u32_e32 v32, 0x70, v105
0000000036EC:	ds_read2st64_b64 v[10:13], v32 offset1:9
0000000036F4:	ds_read2st64_b64 v[22:25], v32 offset0:18 offset1:27
0000000036FC:	s_add_i32 s4, s4, 2
000000003700:	s_cmp_ge_i32 s4, s1
000000003704:	v_and_b32_e32 v33, 0xff, v70
00000000370C:	s_waitcnt lgkmcnt(1)
000000003710:	v_mov_b32_e32 v18, v10
000000003714:	v_mov_b32_e32 v19, v11
000000003718:	ds_read_b128 v[14:17], v105 offset:96
000000003720:	ds_read_b128 v[8:11], v105 offset:4704
000000003728:	s_waitcnt lgkmcnt(2)
00000000372C:	v_mov_b32_e32 v44, v22
000000003730:	v_mov_b32_e32 v45, v23
000000003734:	ds_read_b128 v[40:43], v105 offset:9312
00000000373C:	ds_read_b128 v[20:23], v105 offset:13920
000000003744:	v_bfe_u32 v34, v70, 8, 8
00000000374C:	v_bfe_u32 v35, v70, 16, 8
000000003754:	v_and_b32_e32 v36, 0xff, v74
00000000375C:	v_bfe_u32 v37, v74, 8, 8
000000003764:	v_bfe_u32 v39, v74, 16, 8
00000000376C:	v_lshrrev_b32_e32 v46, 24, v74
000000003770:	s_waitcnt vmcnt(6) lgkmcnt(3)
000000003774:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[50:55], v[14:19], a[112:127], v33, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003784:	s_waitcnt lgkmcnt(2)
000000003788:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[50:55], v[8:13], a[80:95], v33, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003798:	s_waitcnt lgkmcnt(1)
00000000379C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[50:55], v[40:45], a[32:47], v33, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000037AC:	s_waitcnt lgkmcnt(0)
0000000037B0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[50:55], v[20:25], a[16:31], v33, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000037C0:	s_waitcnt vmcnt(4)
0000000037C4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[14:19], a[160:175], v34, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000037D4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[8:13], a[128:143], v34, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000037E4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[40:45], a[96:111], v34, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000037F4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[26:31], v[20:25], a[64:79], v34, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003804:	s_waitcnt vmcnt(2)
000000003808:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[138:143], v[14:19], a[208:223], v35, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003818:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[138:143], v[8:13], a[176:191], v35, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003828:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[138:143], v[40:45], a[144:159], v35, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003838:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[138:143], v[20:25], a[48:63], v35, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003848:	v_lshrrev_b32_e32 v26, 24, v70
00000000384C:	s_waitcnt vmcnt(0)
000000003850:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[2:7], v[14:19], a[240:255], v26, v36 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003860:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[2:7], v[8:13], a[224:239], v26, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003870:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[2:7], v[40:45], a[192:207], v26, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003880:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[2:7], v[20:25], a[0:15], v26, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003890:	s_barrier
000000003894:	s_cbranch_scc1 64546
000000003898:	v_add_u32_e32 v2, 0x80, v130
0000000038A0:	v_mad_i64_i32 v[2:3], s[22:23], v2, 12, s[8:9]
0000000038A8:	v_add_u32_e32 v4, 0x80, v132
0000000038B0:	global_load_dwordx3 v[72:74], v[2:3], off
0000000038B8:	v_add_u32_e32 v2, 0x120, v126
0000000038C0:	v_readfirstlane_b32 s15, v83
0000000038C4:	v_mad_i64_i32 v[4:5], s[22:23], v4, 12, s[10:11]
0000000038CC:	global_load_dwordx3 v[68:70], v[4:5], off
0000000038D4:	s_mov_b32 m0, s15
0000000038D8:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000038E0:	v_add_u32_e32 v2, 0x120, v127
0000000038E8:	v_readfirstlane_b32 s15, v89
0000000038EC:	s_mov_b32 m0, s15
0000000038F0:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000038F8:	v_add_u32_e32 v2, 0x120, v128
000000003900:	v_readfirstlane_b32 s15, v91
000000003904:	s_mov_b32 m0, s15
000000003908:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003910:	v_add_u32_e32 v2, 0x120, v129
000000003918:	v_readfirstlane_b32 s15, v93
00000000391C:	s_mov_b32 m0, s15
000000003920:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003928:	v_add_u32_e32 v2, 0x120, v131
000000003930:	v_readfirstlane_b32 s15, v95
000000003934:	s_mov_b32 m0, s15
000000003938:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003940:	v_add_u32_e32 v2, 0x120, v133
000000003948:	v_readfirstlane_b32 s15, v97
00000000394C:	s_mov_b32 m0, s15
000000003950:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003958:	v_add_u32_e32 v2, 0x120, v134
000000003960:	v_readfirstlane_b32 s15, v99
000000003964:	s_mov_b32 m0, s15
000000003968:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003970:	v_add_u32_e32 v2, 0x120, v135
000000003978:	v_readfirstlane_b32 s15, v101
00000000397C:	s_mov_b32 m0, s15
000000003980:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
000000003988:	v_add_u32_e32 v2, 0x120, v136
000000003990:	v_readfirstlane_b32 s15, v108
000000003994:	s_mov_b32 m0, s15
000000003998:	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
0000000039A0:	s_branch 64479
0000000039A4:	v_accvgpr_write_b32 a0, 0
0000000039AC:	s_mov_b32 s4, 0
0000000039B0:	v_accvgpr_mov_b32 a1, a0
0000000039B4:	v_accvgpr_mov_b32 a2, a0
0000000039B8:	v_accvgpr_mov_b32 a3, a0
0000000039BC:	v_accvgpr_mov_b32 a4, a0
0000000039C0:	v_accvgpr_mov_b32 a5, a0
0000000039C4:	v_accvgpr_mov_b32 a6, a0
0000000039C8:	v_accvgpr_mov_b32 a7, a0
0000000039CC:	v_accvgpr_mov_b32 a8, a0
0000000039D0:	v_accvgpr_mov_b32 a9, a0
0000000039D4:	v_accvgpr_mov_b32 a10, a0
0000000039D8:	v_accvgpr_mov_b32 a11, a0
0000000039DC:	v_accvgpr_mov_b32 a12, a0
0000000039E0:	v_accvgpr_mov_b32 a13, a0
0000000039E4:	v_accvgpr_mov_b32 a14, a0
0000000039E8:	v_accvgpr_mov_b32 a15, a0
0000000039EC:	v_accvgpr_mov_b32 a48, a0
0000000039F0:	v_accvgpr_mov_b32 a49, a0
0000000039F4:	v_accvgpr_mov_b32 a50, a0
0000000039F8:	v_accvgpr_mov_b32 a51, a0
0000000039FC:	v_accvgpr_mov_b32 a52, a0
000000003A00:	v_accvgpr_mov_b32 a53, a0
000000003A04:	v_accvgpr_mov_b32 a54, a0
000000003A08:	v_accvgpr_mov_b32 a55, a0
000000003A0C:	v_accvgpr_mov_b32 a56, a0
000000003A10:	v_accvgpr_mov_b32 a57, a0
000000003A14:	v_accvgpr_mov_b32 a58, a0
000000003A18:	v_accvgpr_mov_b32 a59, a0
000000003A1C:	v_accvgpr_mov_b32 a60, a0
000000003A20:	v_accvgpr_mov_b32 a61, a0
000000003A24:	v_accvgpr_mov_b32 a62, a0
000000003A28:	v_accvgpr_mov_b32 a63, a0
000000003A2C:	v_accvgpr_mov_b32 a64, a0
000000003A30:	v_accvgpr_mov_b32 a65, a0
000000003A34:	v_accvgpr_mov_b32 a66, a0
000000003A38:	v_accvgpr_mov_b32 a67, a0
000000003A3C:	v_accvgpr_mov_b32 a68, a0
000000003A40:	v_accvgpr_mov_b32 a69, a0
000000003A44:	v_accvgpr_mov_b32 a70, a0
000000003A48:	v_accvgpr_mov_b32 a71, a0
000000003A4C:	v_accvgpr_mov_b32 a72, a0
000000003A50:	v_accvgpr_mov_b32 a73, a0
000000003A54:	v_accvgpr_mov_b32 a74, a0
000000003A58:	v_accvgpr_mov_b32 a75, a0
000000003A5C:	v_accvgpr_mov_b32 a76, a0
000000003A60:	v_accvgpr_mov_b32 a77, a0
000000003A64:	v_accvgpr_mov_b32 a78, a0
000000003A68:	v_accvgpr_mov_b32 a79, a0
000000003A6C:	v_accvgpr_mov_b32 a16, a0
000000003A70:	v_accvgpr_mov_b32 a17, a0
000000003A74:	v_accvgpr_mov_b32 a18, a0
000000003A78:	v_accvgpr_mov_b32 a19, a0
000000003A7C:	v_accvgpr_mov_b32 a20, a0
000000003A80:	v_accvgpr_mov_b32 a21, a0
000000003A84:	v_accvgpr_mov_b32 a22, a0
000000003A88:	v_accvgpr_mov_b32 a23, a0
000000003A8C:	v_accvgpr_mov_b32 a24, a0
000000003A90:	v_accvgpr_mov_b32 a25, a0
000000003A94:	v_accvgpr_mov_b32 a26, a0
000000003A98:	v_accvgpr_mov_b32 a27, a0
000000003A9C:	v_accvgpr_mov_b32 a28, a0
000000003AA0:	v_accvgpr_mov_b32 a29, a0
000000003AA4:	v_accvgpr_mov_b32 a30, a0
000000003AA8:	v_accvgpr_mov_b32 a31, a0
000000003AAC:	v_accvgpr_mov_b32 a192, a0
000000003AB0:	v_accvgpr_mov_b32 a193, a0
000000003AB4:	v_accvgpr_mov_b32 a194, a0
000000003AB8:	v_accvgpr_mov_b32 a195, a0
000000003ABC:	v_accvgpr_mov_b32 a196, a0
000000003AC0:	v_accvgpr_mov_b32 a197, a0
000000003AC4:	v_accvgpr_mov_b32 a198, a0
000000003AC8:	v_accvgpr_mov_b32 a199, a0
000000003ACC:	v_accvgpr_mov_b32 a200, a0
000000003AD0:	v_accvgpr_mov_b32 a201, a0
000000003AD4:	v_accvgpr_mov_b32 a202, a0
000000003AD8:	v_accvgpr_mov_b32 a203, a0
000000003ADC:	v_accvgpr_mov_b32 a204, a0
000000003AE0:	v_accvgpr_mov_b32 a205, a0
000000003AE4:	v_accvgpr_mov_b32 a206, a0
000000003AE8:	v_accvgpr_mov_b32 a207, a0
000000003AEC:	v_accvgpr_mov_b32 a144, a0
000000003AF0:	v_accvgpr_mov_b32 a145, a0
000000003AF4:	v_accvgpr_mov_b32 a146, a0
000000003AF8:	v_accvgpr_mov_b32 a147, a0
000000003AFC:	v_accvgpr_mov_b32 a148, a0
000000003B00:	v_accvgpr_mov_b32 a149, a0
000000003B04:	v_accvgpr_mov_b32 a150, a0
000000003B08:	v_accvgpr_mov_b32 a151, a0
000000003B0C:	v_accvgpr_mov_b32 a152, a0
000000003B10:	v_accvgpr_mov_b32 a153, a0
000000003B14:	v_accvgpr_mov_b32 a154, a0
000000003B18:	v_accvgpr_mov_b32 a155, a0
000000003B1C:	v_accvgpr_mov_b32 a156, a0
000000003B20:	v_accvgpr_mov_b32 a157, a0
000000003B24:	v_accvgpr_mov_b32 a158, a0
000000003B28:	v_accvgpr_mov_b32 a159, a0
000000003B2C:	v_accvgpr_mov_b32 a96, a0
000000003B30:	v_accvgpr_mov_b32 a97, a0
000000003B34:	v_accvgpr_mov_b32 a98, a0
000000003B38:	v_accvgpr_mov_b32 a99, a0
000000003B3C:	v_accvgpr_mov_b32 a100, a0
000000003B40:	v_accvgpr_mov_b32 a101, a0
000000003B44:	v_accvgpr_mov_b32 a102, a0
000000003B48:	v_accvgpr_mov_b32 a103, a0
000000003B4C:	v_accvgpr_mov_b32 a104, a0
000000003B50:	v_accvgpr_mov_b32 a105, a0
000000003B54:	v_accvgpr_mov_b32 a106, a0
000000003B58:	v_accvgpr_mov_b32 a107, a0
000000003B5C:	v_accvgpr_mov_b32 a108, a0
000000003B60:	v_accvgpr_mov_b32 a109, a0
000000003B64:	v_accvgpr_mov_b32 a110, a0
000000003B68:	v_accvgpr_mov_b32 a111, a0
000000003B6C:	v_accvgpr_mov_b32 a32, a0
000000003B70:	v_accvgpr_mov_b32 a33, a0
000000003B74:	v_accvgpr_mov_b32 a34, a0
000000003B78:	v_accvgpr_mov_b32 a35, a0
000000003B7C:	v_accvgpr_mov_b32 a36, a0
000000003B80:	v_accvgpr_mov_b32 a37, a0
000000003B84:	v_accvgpr_mov_b32 a38, a0
000000003B88:	v_accvgpr_mov_b32 a39, a0
000000003B8C:	v_accvgpr_mov_b32 a40, a0
000000003B90:	v_accvgpr_mov_b32 a41, a0
000000003B94:	v_accvgpr_mov_b32 a42, a0
000000003B98:	v_accvgpr_mov_b32 a43, a0
000000003B9C:	v_accvgpr_mov_b32 a44, a0
000000003BA0:	v_accvgpr_mov_b32 a45, a0
000000003BA4:	v_accvgpr_mov_b32 a46, a0
000000003BA8:	v_accvgpr_mov_b32 a47, a0
000000003BAC:	v_accvgpr_mov_b32 a224, a0
000000003BB0:	v_accvgpr_mov_b32 a225, a0
000000003BB4:	v_accvgpr_mov_b32 a226, a0
000000003BB8:	v_accvgpr_mov_b32 a227, a0
000000003BBC:	v_accvgpr_mov_b32 a228, a0
000000003BC0:	v_accvgpr_mov_b32 a229, a0
000000003BC4:	v_accvgpr_mov_b32 a230, a0
000000003BC8:	v_accvgpr_mov_b32 a231, a0
000000003BCC:	v_accvgpr_mov_b32 a232, a0
000000003BD0:	v_accvgpr_mov_b32 a233, a0
000000003BD4:	v_accvgpr_mov_b32 a234, a0
000000003BD8:	v_accvgpr_mov_b32 a235, a0
000000003BDC:	v_accvgpr_mov_b32 a236, a0
000000003BE0:	v_accvgpr_mov_b32 a237, a0
000000003BE4:	v_accvgpr_mov_b32 a238, a0
000000003BE8:	v_accvgpr_mov_b32 a239, a0
000000003BEC:	v_accvgpr_mov_b32 a176, a0
000000003BF0:	v_accvgpr_mov_b32 a177, a0
000000003BF4:	v_accvgpr_mov_b32 a178, a0
000000003BF8:	v_accvgpr_mov_b32 a179, a0
000000003BFC:	v_accvgpr_mov_b32 a180, a0
000000003C00:	v_accvgpr_mov_b32 a181, a0
000000003C04:	v_accvgpr_mov_b32 a182, a0
000000003C08:	v_accvgpr_mov_b32 a183, a0
000000003C0C:	v_accvgpr_mov_b32 a184, a0
000000003C10:	v_accvgpr_mov_b32 a185, a0
000000003C14:	v_accvgpr_mov_b32 a186, a0
000000003C18:	v_accvgpr_mov_b32 a187, a0
000000003C1C:	v_accvgpr_mov_b32 a188, a0
000000003C20:	v_accvgpr_mov_b32 a189, a0
000000003C24:	v_accvgpr_mov_b32 a190, a0
000000003C28:	v_accvgpr_mov_b32 a191, a0
000000003C2C:	v_accvgpr_mov_b32 a128, a0
000000003C30:	v_accvgpr_mov_b32 a129, a0
000000003C34:	v_accvgpr_mov_b32 a130, a0
000000003C38:	v_accvgpr_mov_b32 a131, a0
000000003C3C:	v_accvgpr_mov_b32 a132, a0
000000003C40:	v_accvgpr_mov_b32 a133, a0
000000003C44:	v_accvgpr_mov_b32 a134, a0
000000003C48:	v_accvgpr_mov_b32 a135, a0
000000003C4C:	v_accvgpr_mov_b32 a136, a0
000000003C50:	v_accvgpr_mov_b32 a137, a0
000000003C54:	v_accvgpr_mov_b32 a138, a0
000000003C58:	v_accvgpr_mov_b32 a139, a0
000000003C5C:	v_accvgpr_mov_b32 a140, a0
000000003C60:	v_accvgpr_mov_b32 a141, a0
000000003C64:	v_accvgpr_mov_b32 a142, a0
000000003C68:	v_accvgpr_mov_b32 a143, a0
000000003C6C:	v_accvgpr_mov_b32 a80, a0
000000003C70:	v_accvgpr_mov_b32 a81, a0
000000003C74:	v_accvgpr_mov_b32 a82, a0
000000003C78:	v_accvgpr_mov_b32 a83, a0
000000003C7C:	v_accvgpr_mov_b32 a84, a0
000000003C80:	v_accvgpr_mov_b32 a85, a0
000000003C84:	v_accvgpr_mov_b32 a86, a0
000000003C88:	v_accvgpr_mov_b32 a87, a0
000000003C8C:	v_accvgpr_mov_b32 a88, a0
000000003C90:	v_accvgpr_mov_b32 a89, a0
000000003C94:	v_accvgpr_mov_b32 a90, a0
000000003C98:	v_accvgpr_mov_b32 a91, a0
000000003C9C:	v_accvgpr_mov_b32 a92, a0
000000003CA0:	v_accvgpr_mov_b32 a93, a0
000000003CA4:	v_accvgpr_mov_b32 a94, a0
000000003CA8:	v_accvgpr_mov_b32 a95, a0
000000003CAC:	v_accvgpr_mov_b32 a240, a0
000000003CB0:	v_accvgpr_mov_b32 a241, a0
000000003CB4:	v_accvgpr_mov_b32 a242, a0
000000003CB8:	v_accvgpr_mov_b32 a243, a0
000000003CBC:	v_accvgpr_mov_b32 a244, a0
000000003CC0:	v_accvgpr_mov_b32 a245, a0
000000003CC4:	v_accvgpr_mov_b32 a246, a0
000000003CC8:	v_accvgpr_mov_b32 a247, a0
000000003CCC:	v_accvgpr_mov_b32 a248, a0
000000003CD0:	v_accvgpr_mov_b32 a249, a0
000000003CD4:	v_accvgpr_mov_b32 a250, a0
000000003CD8:	v_accvgpr_mov_b32 a251, a0
000000003CDC:	v_accvgpr_mov_b32 a252, a0
000000003CE0:	v_accvgpr_mov_b32 a253, a0
000000003CE4:	v_accvgpr_mov_b32 a254, a0
000000003CE8:	v_accvgpr_mov_b32 a255, a0
000000003CEC:	v_accvgpr_mov_b32 a208, a0
000000003CF0:	v_accvgpr_mov_b32 a209, a0
000000003CF4:	v_accvgpr_mov_b32 a210, a0
000000003CF8:	v_accvgpr_mov_b32 a211, a0
000000003CFC:	v_accvgpr_mov_b32 a212, a0
000000003D00:	v_accvgpr_mov_b32 a213, a0
000000003D04:	v_accvgpr_mov_b32 a214, a0
000000003D08:	v_accvgpr_mov_b32 a215, a0
000000003D0C:	v_accvgpr_mov_b32 a216, a0
000000003D10:	v_accvgpr_mov_b32 a217, a0
000000003D14:	v_accvgpr_mov_b32 a218, a0
000000003D18:	v_accvgpr_mov_b32 a219, a0
000000003D1C:	v_accvgpr_mov_b32 a220, a0
000000003D20:	v_accvgpr_mov_b32 a221, a0
000000003D24:	v_accvgpr_mov_b32 a222, a0
000000003D28:	v_accvgpr_mov_b32 a223, a0
000000003D2C:	v_accvgpr_mov_b32 a160, a0
000000003D30:	v_accvgpr_mov_b32 a161, a0
000000003D34:	v_accvgpr_mov_b32 a162, a0
000000003D38:	v_accvgpr_mov_b32 a163, a0
000000003D3C:	v_accvgpr_mov_b32 a164, a0
000000003D40:	v_accvgpr_mov_b32 a165, a0
000000003D44:	v_accvgpr_mov_b32 a166, a0
000000003D48:	v_accvgpr_mov_b32 a167, a0
000000003D4C:	v_accvgpr_mov_b32 a168, a0
000000003D50:	v_accvgpr_mov_b32 a169, a0
000000003D54:	v_accvgpr_mov_b32 a170, a0
000000003D58:	v_accvgpr_mov_b32 a171, a0
000000003D5C:	v_accvgpr_mov_b32 a172, a0
000000003D60:	v_accvgpr_mov_b32 a173, a0
000000003D64:	v_accvgpr_mov_b32 a174, a0
000000003D68:	v_accvgpr_mov_b32 a175, a0
000000003D6C:	v_accvgpr_mov_b32 a112, a0
000000003D70:	v_accvgpr_mov_b32 a113, a0
000000003D74:	v_accvgpr_mov_b32 a114, a0
000000003D78:	v_accvgpr_mov_b32 a115, a0
000000003D7C:	v_accvgpr_mov_b32 a116, a0
000000003D80:	v_accvgpr_mov_b32 a117, a0
000000003D84:	v_accvgpr_mov_b32 a118, a0
000000003D88:	v_accvgpr_mov_b32 a119, a0
000000003D8C:	v_accvgpr_mov_b32 a120, a0
000000003D90:	v_accvgpr_mov_b32 a121, a0
000000003D94:	v_accvgpr_mov_b32 a122, a0
000000003D98:	v_accvgpr_mov_b32 a123, a0
000000003D9C:	v_accvgpr_mov_b32 a124, a0
000000003DA0:	v_accvgpr_mov_b32 a125, a0
000000003DA4:	v_accvgpr_mov_b32 a126, a0
000000003DA8:	v_accvgpr_mov_b32 a127, a0
000000003DAC:	s_cmp_ge_i32 s4, s1
000000003DB0:	s_cbranch_scc1 475
000000003DB4:	s_lshl_b32 s1, s3, 3
000000003DB8:	v_lshl_or_b32 v2, v1, 2, s1
000000003DC0:	s_mul_i32 s2, s4, 3
000000003DC4:	v_mul_lo_u32 v64, s13, v2
000000003DCC:	v_mov_b32_e32 v87, 0
000000003DD0:	v_add_u32_e32 v2, s2, v64
000000003DD4:	s_movk_i32 s1, 0x600
000000003DD8:	v_mov_b64_e32 v[44:45], s[6:7]
000000003DDC:	v_mov_b32_e32 v85, v87
000000003DE0:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000003DE8:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003DF0:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003DF8:	v_add_u32_e32 v65, s13, v64
000000003DFC:	s_waitcnt vmcnt(0)
000000003E00:	s_barrier
000000003E04:	global_load_dwordx4 v[14:17], v[4:5], off
000000003E0C:	global_load_dwordx2 v[18:19], v[2:3], off offset:1024
000000003E14:	v_add_u32_e32 v2, s2, v65
000000003E18:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000003E20:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003E28:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003E30:	v_add_u32_e32 v66, s13, v65
000000003E34:	global_load_dwordx4 v[46:49], v[4:5], off
000000003E3C:	global_load_dwordx2 v[50:51], v[2:3], off offset:1024
000000003E44:	v_add_u32_e32 v2, s2, v66
000000003E48:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000003E50:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003E58:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003E60:	v_add_u32_e32 v75, s13, v66
000000003E64:	global_load_dwordx4 v[20:23], v[4:5], off
000000003E6C:	global_load_dwordx2 v[24:25], v[2:3], off offset:1024
000000003E74:	v_add_u32_e32 v2, s2, v75
000000003E78:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000003E80:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003E88:	v_lshl_add_u64 v[2:3], v[2:3], 0, v[84:85]
000000003E90:	global_load_dwordx4 v[32:35], v[4:5], off
000000003E98:	global_load_dwordx2 v[36:37], v[2:3], off offset:1024
000000003EA0:	s_or_b32 s6, s2, 1
000000003EA4:	v_add_u32_e32 v2, s6, v64
000000003EA8:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000003EB0:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000003EB8:	v_lshl_add_u64 v[6:7], v[2:3], 0, v[84:85]
000000003EC0:	global_load_dwordx4 v[2:5], v[4:5], off
000000003EC8:	s_nop 0
000000003ECC:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
000000003ED4:	v_and_b32_e32 v10, 0x9f, v0
000000003EDC:	v_add_u32_e32 v8, s6, v65
000000003EE0:	v_mul_u32_u24_e32 v11, 24, v71
000000003EE4:	v_mul_u32_u24_e32 v10, 0x90, v10
000000003EEC:	v_mad_i64_i32 v[8:9], s[4:5], v8, s1, v[44:45]
000000003EF4:	v_add3_u32 v71, 0, v11, v10
000000003EFC:	v_add_u32_e32 v26, 16, v71
000000003F00:	v_lshl_add_u64 v[10:11], v[8:9], 0, v[86:87]
000000003F08:	v_lshl_add_u64 v[12:13], v[8:9], 0, v[84:85]
000000003F10:	ds_read2st64_b64 v[54:57], v26 offset1:9
000000003F18:	global_load_dwordx4 v[8:11], v[10:11], off
000000003F20:	s_nop 0
000000003F24:	global_load_dwordx2 v[12:13], v[12:13], off offset:1024
000000003F2C:	ds_read2st64_b64 v[78:81], v26 offset0:18 offset1:27
000000003F34:	v_add_u32_e32 v26, s6, v66
000000003F38:	v_mad_i64_i32 v[26:27], s[4:5], v26, s1, v[44:45]
000000003F40:	s_waitcnt lgkmcnt(1)
000000003F44:	v_mov_b32_e32 v62, v54
000000003F48:	v_mov_b32_e32 v63, v55
000000003F4C:	ds_read_b128 v[58:61], v71
000000003F54:	ds_read_b128 v[52:55], v71 offset:4608
000000003F5C:	s_waitcnt lgkmcnt(2)
000000003F60:	v_mov_b32_e32 v92, v78
000000003F64:	v_mov_b32_e32 v93, v79
000000003F68:	ds_read_b128 v[88:91], v71 offset:9216
000000003F70:	ds_read_b128 v[76:79], v71 offset:13824
000000003F78:	v_lshl_add_u64 v[28:29], v[26:27], 0, v[86:87]
000000003F80:	v_lshl_add_u64 v[30:31], v[26:27], 0, v[84:85]
000000003F88:	v_and_b32_e32 v38, 0xff, v68
000000003F90:	v_and_b32_e32 v83, 0xff, v72
000000003F98:	v_bfe_u32 v94, v72, 8, 8
000000003FA0:	v_bfe_u32 v95, v72, 16, 8
000000003FA8:	v_lshrrev_b32_e32 v72, 24, v72
000000003FAC:	global_load_dwordx4 v[26:29], v[28:29], off
000000003FB4:	s_nop 0
000000003FB8:	global_load_dwordx2 v[30:31], v[30:31], off offset:1024
000000003FC0:	s_add_i32 s2, s2, 2
000000003FC4:	v_bfe_u32 v67, v68, 8, 8
000000003FCC:	v_bfe_u32 v82, v68, 16, 8
000000003FD4:	v_lshrrev_b32_e32 v68, 24, v68
000000003FD8:	s_waitcnt vmcnt(12) lgkmcnt(3)
000000003FDC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[14:19], v[58:63], a[112:127], v38, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000003FEC:	s_waitcnt lgkmcnt(2)
000000003FF0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[14:19], v[52:57], a[80:95], v38, v94 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004000:	s_waitcnt lgkmcnt(1)
000000004004:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[14:19], v[88:93], a[32:47], v38, v95 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004014:	s_waitcnt lgkmcnt(0)
000000004018:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[14:19], v[76:81], a[16:31], v38, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004028:	v_add_u32_e32 v14, s6, v75
00000000402C:	v_mad_i64_i32 v[14:15], s[4:5], v14, s1, v[44:45]
000000004034:	v_lshl_add_u64 v[16:17], v[14:15], 0, v[86:87]
00000000403C:	v_lshl_add_u64 v[14:15], v[14:15], 0, v[84:85]
000000004044:	global_load_dwordx4 v[38:41], v[16:17], off
00000000404C:	global_load_dwordx2 v[42:43], v[14:15], off offset:1024
000000004054:	v_add_u32_e32 v14, s2, v64
000000004058:	v_mad_i64_i32 v[14:15], s[4:5], v14, s1, v[44:45]
000000004060:	v_lshl_add_u64 v[16:17], v[14:15], 0, v[86:87]
000000004068:	v_lshl_add_u64 v[18:19], v[14:15], 0, v[84:85]
000000004070:	s_waitcnt vmcnt(12)
000000004074:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[46:51], v[58:63], a[160:175], v67, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004084:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[46:51], v[52:57], a[128:143], v67, v94 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004094:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[46:51], v[88:93], a[96:111], v67, v95 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000040A4:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[46:51], v[76:81], a[64:79], v67, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000040B4:	global_load_dwordx4 v[14:17], v[16:17], off
0000000040BC:	s_nop 0
0000000040C0:	global_load_dwordx2 v[18:19], v[18:19], off offset:1024
0000000040C8:	s_waitcnt vmcnt(12)
0000000040CC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[20:25], v[58:63], a[208:223], v82, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000040DC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[20:25], v[52:57], a[176:191], v82, v94 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000040EC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[20:25], v[88:93], a[144:159], v82, v95 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000040FC:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[20:25], v[76:81], a[48:63], v82, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000410C:	v_add_u32_e32 v20, s2, v65
000000004110:	v_mad_i64_i32 v[20:21], s[4:5], v20, s1, v[44:45]
000000004118:	v_lshl_add_u64 v[22:23], v[20:21], 0, v[86:87]
000000004120:	v_lshl_add_u64 v[24:25], v[20:21], 0, v[84:85]
000000004128:	global_load_dwordx4 v[20:23], v[22:23], off
000000004130:	s_nop 0
000000004134:	global_load_dwordx2 v[24:25], v[24:25], off offset:1024
00000000413C:	s_waitcnt vmcnt(12)
000000004140:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[58:63], a[240:255], v68, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004150:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[32:37], v[52:57], a[224:239], v68, v94 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004160:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[32:37], v[88:93], a[192:207], v68, v95 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004170:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[76:81], a[0:15], v68, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004180:	v_add_u32_e32 v32, 64, v71
000000004184:	ds_read2st64_b64 v[34:37], v32 offset1:9
00000000418C:	ds_read2st64_b64 v[54:57], v32 offset0:18 offset1:27
000000004194:	v_add_u32_e32 v64, s2, v66
000000004198:	v_mad_i64_i32 v[64:65], s[4:5], v64, s1, v[44:45]
0000000041A0:	s_waitcnt lgkmcnt(1)
0000000041A4:	v_mov_b32_e32 v50, v34
0000000041A8:	v_mov_b32_e32 v51, v35
0000000041AC:	ds_read_b128 v[46:49], v71 offset:48
0000000041B4:	ds_read_b128 v[32:35], v71 offset:4656
0000000041BC:	s_waitcnt lgkmcnt(2)
0000000041C0:	v_mov_b32_e32 v62, v54
0000000041C4:	v_mov_b32_e32 v63, v55
0000000041C8:	ds_read_b128 v[58:61], v71 offset:9264
0000000041D0:	ds_read_b128 v[52:55], v71 offset:13872
0000000041D8:	v_lshl_add_u64 v[66:67], v[64:65], 0, v[86:87]
0000000041E0:	v_and_b32_e32 v68, 0xff, v69
0000000041E8:	v_and_b32_e32 v72, 0xff, v73
0000000041F0:	v_bfe_u32 v82, v73, 8, 8
0000000041F8:	v_bfe_u32 v83, v73, 16, 8
000000004200:	v_lshrrev_b32_e32 v73, 24, v73
000000004204:	v_lshl_add_u64 v[64:65], v[64:65], 0, v[84:85]
00000000420C:	global_load_dwordx4 v[76:79], v[66:67], off
000000004214:	global_load_dwordx2 v[80:81], v[64:65], off offset:1024
00000000421C:	s_waitcnt vmcnt(12) lgkmcnt(3)
000000004220:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[2:7], v[46:51], a[112:127], v68, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004230:	s_waitcnt lgkmcnt(2)
000000004234:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[2:7], v[32:37], a[80:95], v68, v82 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004244:	s_waitcnt lgkmcnt(1)
000000004248:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[2:7], v[58:63], a[32:47], v68, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004258:	s_waitcnt lgkmcnt(0)
00000000425C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[2:7], v[52:57], a[16:31], v68, v73 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000426C:	v_add_u32_e32 v2, s2, v75
000000004270:	v_mad_i64_i32 v[2:3], s[4:5], v2, s1, v[44:45]
000000004278:	v_lshl_add_u64 v[4:5], v[2:3], 0, v[86:87]
000000004280:	v_lshl_add_u64 v[6:7], v[2:3], 0, v[84:85]
000000004288:	global_load_dwordx4 v[2:5], v[4:5], off
000000004290:	s_nop 0
000000004294:	global_load_dwordx2 v[6:7], v[6:7], off offset:1024
00000000429C:	v_bfe_u32 v44, v69, 8, 8
0000000042A4:	s_waitcnt vmcnt(12)
0000000042A8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[8:13], v[46:51], a[160:175], v44, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000042B8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[8:13], v[32:37], a[128:143], v44, v82 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000042C8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[8:13], v[58:63], a[96:111], v44, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000042D8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[8:13], v[52:57], a[64:79], v44, v73 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000042E8:	v_add_u32_e32 v8, 0x70, v71
0000000042F0:	v_bfe_u32 v45, v69, 16, 8
0000000042F8:	v_lshrrev_b32_e32 v64, 24, v69
0000000042FC:	s_waitcnt vmcnt(10)
000000004300:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[26:31], v[46:51], a[208:223], v45, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004310:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[26:31], v[32:37], a[176:191], v45, v82 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004320:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[26:31], v[58:63], a[144:159], v45, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004330:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[52:57], a[48:63], v45, v73 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004340:	s_waitcnt vmcnt(8)
000000004344:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[38:43], v[46:51], a[240:255], v64, v72 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004354:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[38:43], v[32:37], a[224:239], v64, v82 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004364:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[58:63], a[192:207], v64, v83 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004374:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[52:57], a[0:15], v64, v73 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004384:	ds_read2st64_b64 v[10:13], v8 offset1:9
00000000438C:	ds_read2st64_b64 v[34:37], v8 offset0:18 offset1:27
000000004394:	v_and_b32_e32 v44, 0xff, v70
00000000439C:	v_bfe_u32 v45, v70, 8, 8
0000000043A4:	v_bfe_u32 v46, v70, 16, 8
0000000043AC:	s_waitcnt lgkmcnt(1)
0000000043B0:	v_mov_b32_e32 v30, v10
0000000043B4:	v_mov_b32_e32 v31, v11
0000000043B8:	ds_read_b128 v[26:29], v71 offset:96
0000000043C0:	ds_read_b128 v[8:11], v71 offset:4704
0000000043C8:	s_waitcnt lgkmcnt(2)
0000000043CC:	v_mov_b32_e32 v42, v34
0000000043D0:	v_mov_b32_e32 v43, v35
0000000043D4:	ds_read_b128 v[38:41], v71 offset:9312
0000000043DC:	ds_read_b128 v[32:35], v71 offset:13920
0000000043E4:	v_and_b32_e32 v47, 0xff, v74
0000000043EC:	v_bfe_u32 v48, v74, 8, 8
0000000043F4:	v_bfe_u32 v49, v74, 16, 8
0000000043FC:	v_lshrrev_b32_e32 v50, 24, v74
000000004400:	s_waitcnt vmcnt(6) lgkmcnt(3)
000000004404:	v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[14:19], v[26:31], a[112:127], v44, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004414:	s_waitcnt lgkmcnt(2)
000000004418:	v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[14:19], v[8:13], a[80:95], v44, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004428:	s_waitcnt lgkmcnt(1)
00000000442C:	v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[14:19], v[38:43], a[32:47], v44, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
00000000443C:	s_waitcnt lgkmcnt(0)
000000004440:	v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[14:19], v[32:37], a[16:31], v44, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004450:	s_waitcnt vmcnt(4)
000000004454:	v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[20:25], v[26:31], a[160:175], v45, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004464:	v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[20:25], v[8:13], a[128:143], v45, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004474:	v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[20:25], v[38:43], a[96:111], v45, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004484:	v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[20:25], v[32:37], a[64:79], v45, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004494:	s_waitcnt vmcnt(2)
000000004498:	v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[76:81], v[26:31], a[208:223], v46, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000044A8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[76:81], v[8:13], a[176:191], v46, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000044B8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[76:81], v[38:43], a[144:159], v46, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000044C8:	v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[76:81], v[32:37], a[48:63], v46, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000044D8:	v_lshrrev_b32_e32 v14, 24, v70
0000000044DC:	s_waitcnt vmcnt(0)
0000000044E0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[2:7], v[26:31], a[240:255], v14, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2
0000000044F0:	v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[2:7], v[8:13], a[224:239], v14, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004500:	v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[2:7], v[38:43], a[192:207], v14, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004510:	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[2:7], v[32:37], a[0:15], v14, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2
000000004520:	v_and_b32_e32 v242, 0x80, v0
000000004528:	v_add_u32_e32 v243, s0, v242
00000000452C:	s_lshl_b32 s0, s3, 8
000000004530:	v_lshl_or_b32 v242, v1, 7, s0
000000004538:	v_and_or_b32 v250, v0, 31, v243
000000004540:	v_mad_i64_i32 v[244:245], s[0:1], v250, s12, 0
000000004548:	v_ashrrev_i32_e32 v243, 31, v242
00000000454C:	v_accvgpr_read_b32 v18, a112
000000004554:	v_accvgpr_read_b32 v194, a240
00000000455C:	v_lshl_add_u64 v[244:245], v[244:245], 1, s[20:21]
000000004564:	v_lshlrev_b64 v[242:243], 1, v[242:243]
00000000456C:	v_lshrrev_b32_e32 v0, 2, v0
000000004570:	v_accvgpr_read_b32 v19, a113
000000004578:	v_accvgpr_read_b32 v20, a114
000000004580:	v_accvgpr_read_b32 v21, a115
000000004588:	v_accvgpr_read_b32 v22, a116
000000004590:	v_accvgpr_read_b32 v23, a117
000000004598:	v_accvgpr_read_b32 v24, a118
0000000045A0:	v_accvgpr_read_b32 v25, a119
0000000045A8:	v_accvgpr_read_b32 v26, a120
0000000045B0:	v_accvgpr_read_b32 v27, a121
0000000045B8:	v_accvgpr_read_b32 v28, a122
0000000045C0:	v_accvgpr_read_b32 v29, a123
0000000045C8:	v_accvgpr_read_b32 v195, a241
0000000045D0:	v_accvgpr_read_b32 v196, a242
0000000045D8:	v_accvgpr_read_b32 v197, a243
0000000045E0:	v_accvgpr_read_b32 v198, a244
0000000045E8:	v_accvgpr_read_b32 v199, a245
0000000045F0:	v_accvgpr_read_b32 v200, a246
0000000045F8:	v_accvgpr_read_b32 v201, a247
000000004600:	v_lshl_add_u64 v[244:245], v[244:245], 0, v[242:243]
000000004608:	v_and_b32_e32 v0, 8, v0
00000000460C:	v_mov_b32_e32 v1, 0
000000004610:	v_accvgpr_read_b32 v30, a124
000000004618:	v_accvgpr_read_b32 v31, a125
000000004620:	v_accvgpr_read_b32 v32, a126
000000004628:	v_accvgpr_read_b32 v33, a127
000000004630:	v_accvgpr_read_b32 v202, a248
000000004638:	v_accvgpr_read_b32 v203, a249
000000004640:	v_accvgpr_read_b32 v204, a250
000000004648:	v_accvgpr_read_b32 v205, a251
000000004650:	v_lshl_add_u64 v[244:245], v[244:245], 0, v[0:1]
000000004658:	v_cvt_pk_f16_f32 v21, v20, v21
000000004660:	v_cvt_pk_f16_f32 v20, v18, v19
000000004668:	v_cvt_pk_f16_f32 v19, v24, v25
000000004670:	v_cvt_pk_f16_f32 v18, v22, v23
000000004678:	v_cvt_pk_f16_f32 v247, v28, v29
000000004680:	v_cvt_pk_f16_f32 v246, v26, v27
000000004688:	v_cvt_pk_f16_f32 v197, v196, v197
000000004690:	v_cvt_pk_f16_f32 v196, v194, v195
000000004698:	v_cvt_pk_f16_f32 v195, v200, v201
0000000046A0:	v_cvt_pk_f16_f32 v194, v198, v199
0000000046A8:	v_accvgpr_read_b32 v206, a252
0000000046B0:	v_accvgpr_read_b32 v207, a253
0000000046B8:	v_accvgpr_read_b32 v208, a254
0000000046C0:	v_accvgpr_read_b32 v209, a255
0000000046C8:	global_store_dwordx2 v[244:245], v[20:21], off
0000000046D0:	global_store_dwordx2 v[244:245], v[18:19], off offset:16
0000000046D8:	v_cvt_pk_f16_f32 v249, v32, v33
0000000046E0:	v_cvt_pk_f16_f32 v248, v30, v31
0000000046E8:	global_store_dwordx2 v[244:245], v[246:247], off offset:32
0000000046F0:	global_store_dwordx2 v[244:245], v[248:249], off offset:48
0000000046F8:	global_store_dwordx2 v[244:245], v[194:195], off offset:208
000000004700:	v_cvt_pk_f16_f32 v195, v204, v205
000000004708:	v_cvt_pk_f16_f32 v194, v202, v203
000000004710:	global_store_dwordx2 v[244:245], v[194:195], off offset:224
000000004718:	v_cvt_pk_f16_f32 v195, v208, v209
000000004720:	v_cvt_pk_f16_f32 v194, v206, v207
000000004728:	v_accvgpr_read_b32 v241, a175
000000004730:	v_accvgpr_read_b32 v225, a223
000000004738:	global_store_dwordx2 v[244:245], v[194:195], off offset:240
000000004740:	v_or_b32_e32 v194, 32, v250
000000004744:	v_accvgpr_read_b32 v233, a167
00000000474C:	v_accvgpr_read_b32 v232, a166
000000004754:	v_accvgpr_read_b32 v231, a165
00000000475C:	v_accvgpr_read_b32 v230, a164
000000004764:	v_accvgpr_read_b32 v229, a163
00000000476C:	v_accvgpr_read_b32 v228, a162
000000004774:	v_accvgpr_read_b32 v227, a161
00000000477C:	v_accvgpr_read_b32 v226, a160
000000004784:	v_accvgpr_read_b32 v217, a215
00000000478C:	v_accvgpr_read_b32 v216, a214
000000004794:	v_accvgpr_read_b32 v215, a213
00000000479C:	v_accvgpr_read_b32 v214, a212
0000000047A4:	v_accvgpr_read_b32 v213, a211
0000000047AC:	v_accvgpr_read_b32 v212, a210
0000000047B4:	v_accvgpr_read_b32 v211, a209
0000000047BC:	v_accvgpr_read_b32 v210, a208
0000000047C4:	v_mad_i64_i32 v[194:195], s[0:1], v194, s12, 0
0000000047CC:	v_accvgpr_read_b32 v237, a171
0000000047D4:	v_accvgpr_read_b32 v236, a170
0000000047DC:	v_accvgpr_read_b32 v235, a169
0000000047E4:	v_accvgpr_read_b32 v234, a168
0000000047EC:	v_accvgpr_read_b32 v221, a219
0000000047F4:	v_accvgpr_read_b32 v220, a218
0000000047FC:	v_accvgpr_read_b32 v219, a217
000000004804:	v_accvgpr_read_b32 v218, a216
00000000480C:	v_accvgpr_read_b32 v130, a224
000000004814:	v_cvt_pk_f16_f32 v229, v228, v229
00000000481C:	v_cvt_pk_f16_f32 v228, v226, v227
000000004824:	v_cvt_pk_f16_f32 v227, v232, v233
00000000482C:	v_cvt_pk_f16_f32 v226, v230, v231
000000004834:	v_cvt_pk_f16_f32 v213, v212, v213
00000000483C:	v_cvt_pk_f16_f32 v212, v210, v211
000000004844:	v_cvt_pk_f16_f32 v211, v216, v217
00000000484C:	v_cvt_pk_f16_f32 v210, v214, v215
000000004854:	v_lshl_add_u64 v[194:195], v[194:195], 1, s[20:21]
00000000485C:	v_accvgpr_read_b32 v240, a174
000000004864:	v_accvgpr_read_b32 v239, a173
00000000486C:	v_accvgpr_read_b32 v238, a172
000000004874:	v_accvgpr_read_b32 v224, a222
00000000487C:	v_accvgpr_read_b32 v223, a221
000000004884:	v_accvgpr_read_b32 v222, a220
00000000488C:	v_accvgpr_read_b32 v131, a225
000000004894:	v_accvgpr_read_b32 v132, a226
00000000489C:	v_accvgpr_read_b32 v133, a227
0000000048A4:	v_accvgpr_read_b32 v134, a228
0000000048AC:	v_accvgpr_read_b32 v135, a229
0000000048B4:	v_accvgpr_read_b32 v136, a230
0000000048BC:	v_accvgpr_read_b32 v137, a231
0000000048C4:	global_store_dwordx2 v[244:245], v[226:227], off offset:80
0000000048CC:	v_cvt_pk_f16_f32 v227, v236, v237
0000000048D4:	v_cvt_pk_f16_f32 v226, v234, v235
0000000048DC:	global_store_dwordx2 v[244:245], v[210:211], off offset:144
0000000048E4:	v_cvt_pk_f16_f32 v211, v220, v221
0000000048EC:	v_cvt_pk_f16_f32 v210, v218, v219
0000000048F4:	v_lshl_add_u64 v[194:195], v[194:195], 0, v[242:243]
0000000048FC:	v_accvgpr_read_b32 v138, a232
000000004904:	v_accvgpr_read_b32 v139, a233
00000000490C:	v_accvgpr_read_b32 v140, a234
000000004914:	v_accvgpr_read_b32 v141, a235
00000000491C:	global_store_dwordx2 v[244:245], v[226:227], off offset:96
000000004924:	v_cvt_pk_f16_f32 v227, v240, v241
00000000492C:	v_cvt_pk_f16_f32 v226, v238, v239
000000004934:	global_store_dwordx2 v[244:245], v[210:211], off offset:160
00000000493C:	v_cvt_pk_f16_f32 v211, v224, v225
000000004944:	v_cvt_pk_f16_f32 v210, v222, v223
00000000494C:	v_lshl_add_u64 v[194:195], v[194:195], 0, v[0:1]
000000004954:	v_cvt_pk_f16_f32 v133, v132, v133
00000000495C:	v_cvt_pk_f16_f32 v132, v130, v131
000000004964:	v_cvt_pk_f16_f32 v131, v136, v137
00000000496C:	v_cvt_pk_f16_f32 v130, v134, v135
000000004974:	v_accvgpr_read_b32 v142, a236
00000000497C:	v_accvgpr_read_b32 v143, a237
000000004984:	v_accvgpr_read_b32 v144, a238
00000000498C:	v_accvgpr_read_b32 v145, a239
000000004994:	global_store_dwordx2 v[244:245], v[228:229], off offset:64
00000000499C:	global_store_dwordx2 v[244:245], v[226:227], off offset:112
0000000049A4:	global_store_dwordx2 v[244:245], v[212:213], off offset:128
0000000049AC:	global_store_dwordx2 v[244:245], v[210:211], off offset:176
0000000049B4:	global_store_dwordx2 v[244:245], v[196:197], off offset:192
0000000049BC:	global_store_dwordx2 v[194:195], v[130:131], off offset:208
0000000049C4:	v_cvt_pk_f16_f32 v131, v140, v141
0000000049CC:	v_cvt_pk_f16_f32 v130, v138, v139
0000000049D4:	global_store_dwordx2 v[194:195], v[130:131], off offset:224
0000000049DC:	v_cvt_pk_f16_f32 v131, v144, v145
0000000049E4:	v_cvt_pk_f16_f32 v130, v142, v143
0000000049EC:	v_accvgpr_read_b32 v193, a95
0000000049F4:	v_accvgpr_read_b32 v177, a143
0000000049FC:	v_accvgpr_read_b32 v146, a176
000000004A04:	global_store_dwordx2 v[194:195], v[130:131], off offset:240
000000004A0C:	v_or_b32_e32 v130, 64, v250
000000004A10:	v_accvgpr_read_b32 v185, a87
000000004A18:	v_accvgpr_read_b32 v184, a86
000000004A20:	v_accvgpr_read_b32 v183, a85
000000004A28:	v_accvgpr_read_b32 v182, a84
000000004A30:	v_accvgpr_read_b32 v181, a83
000000004A38:	v_accvgpr_read_b32 v180, a82
000000004A40:	v_accvgpr_read_b32 v179, a81
000000004A48:	v_accvgpr_read_b32 v178, a80
000000004A50:	v_accvgpr_read_b32 v169, a135
000000004A58:	v_accvgpr_read_b32 v168, a134
000000004A60:	v_accvgpr_read_b32 v167, a133
000000004A68:	v_accvgpr_read_b32 v166, a132
000000004A70:	v_accvgpr_read_b32 v165, a131
000000004A78:	v_accvgpr_read_b32 v164, a130
000000004A80:	v_accvgpr_read_b32 v163, a129
000000004A88:	v_accvgpr_read_b32 v162, a128
000000004A90:	v_accvgpr_read_b32 v147, a177
000000004A98:	v_accvgpr_read_b32 v148, a178
000000004AA0:	v_accvgpr_read_b32 v149, a179
000000004AA8:	v_accvgpr_read_b32 v150, a180
000000004AB0:	v_accvgpr_read_b32 v151, a181
000000004AB8:	v_accvgpr_read_b32 v152, a182
000000004AC0:	v_accvgpr_read_b32 v153, a183
000000004AC8:	v_mad_i64_i32 v[130:131], s[0:1], v130, s12, 0
000000004AD0:	v_accvgpr_read_b32 v189, a91
000000004AD8:	v_accvgpr_read_b32 v188, a90
000000004AE0:	v_accvgpr_read_b32 v187, a89
000000004AE8:	v_accvgpr_read_b32 v186, a88
000000004AF0:	v_accvgpr_read_b32 v173, a139
000000004AF8:	v_accvgpr_read_b32 v172, a138
000000004B00:	v_accvgpr_read_b32 v171, a137
000000004B08:	v_accvgpr_read_b32 v170, a136
000000004B10:	v_accvgpr_read_b32 v154, a184
000000004B18:	v_accvgpr_read_b32 v155, a185
000000004B20:	v_accvgpr_read_b32 v156, a186
000000004B28:	v_accvgpr_read_b32 v157, a187
000000004B30:	v_accvgpr_read_b32 v66, a192
000000004B38:	v_cvt_pk_f16_f32 v181, v180, v181
000000004B40:	v_cvt_pk_f16_f32 v180, v178, v179
000000004B48:	v_cvt_pk_f16_f32 v179, v184, v185
000000004B50:	v_cvt_pk_f16_f32 v178, v182, v183
000000004B58:	v_cvt_pk_f16_f32 v165, v164, v165
000000004B60:	v_cvt_pk_f16_f32 v164, v162, v163
000000004B68:	v_cvt_pk_f16_f32 v163, v168, v169
000000004B70:	v_cvt_pk_f16_f32 v162, v166, v167
000000004B78:	v_cvt_pk_f16_f32 v149, v148, v149
000000004B80:	v_cvt_pk_f16_f32 v148, v146, v147
000000004B88:	v_cvt_pk_f16_f32 v147, v152, v153
000000004B90:	v_cvt_pk_f16_f32 v146, v150, v151
000000004B98:	v_lshl_add_u64 v[130:131], v[130:131], 1, s[20:21]
000000004BA0:	v_accvgpr_read_b32 v192, a94
000000004BA8:	v_accvgpr_read_b32 v191, a93
000000004BB0:	v_accvgpr_read_b32 v190, a92
000000004BB8:	v_accvgpr_read_b32 v176, a142
000000004BC0:	v_accvgpr_read_b32 v175, a141
000000004BC8:	v_accvgpr_read_b32 v174, a140
000000004BD0:	v_accvgpr_read_b32 v158, a188
000000004BD8:	v_accvgpr_read_b32 v159, a189
000000004BE0:	v_accvgpr_read_b32 v160, a190
000000004BE8:	v_accvgpr_read_b32 v161, a191
000000004BF0:	v_accvgpr_read_b32 v67, a193
000000004BF8:	v_accvgpr_read_b32 v68, a194
000000004C00:	v_accvgpr_read_b32 v69, a195
000000004C08:	v_accvgpr_read_b32 v70, a196
000000004C10:	v_accvgpr_read_b32 v71, a197
000000004C18:	v_accvgpr_read_b32 v72, a198
000000004C20:	v_accvgpr_read_b32 v73, a199
000000004C28:	global_store_dwordx2 v[194:195], v[178:179], off offset:16
000000004C30:	v_cvt_pk_f16_f32 v179, v188, v189
000000004C38:	v_cvt_pk_f16_f32 v178, v186, v187
000000004C40:	global_store_dwordx2 v[194:195], v[162:163], off offset:80
000000004C48:	v_cvt_pk_f16_f32 v163, v172, v173
000000004C50:	v_cvt_pk_f16_f32 v162, v170, v171
000000004C58:	global_store_dwordx2 v[194:195], v[146:147], off offset:144
000000004C60:	v_cvt_pk_f16_f32 v147, v156, v157
000000004C68:	v_cvt_pk_f16_f32 v146, v154, v155
000000004C70:	v_lshl_add_u64 v[130:131], v[130:131], 0, v[242:243]
000000004C78:	v_accvgpr_read_b32 v74, a200
000000004C80:	v_accvgpr_read_b32 v75, a201
000000004C88:	v_accvgpr_read_b32 v76, a202
000000004C90:	v_accvgpr_read_b32 v77, a203
000000004C98:	global_store_dwordx2 v[194:195], v[178:179], off offset:32
000000004CA0:	v_cvt_pk_f16_f32 v179, v192, v193
000000004CA8:	v_cvt_pk_f16_f32 v178, v190, v191
000000004CB0:	global_store_dwordx2 v[194:195], v[162:163], off offset:96
000000004CB8:	v_cvt_pk_f16_f32 v163, v176, v177
000000004CC0:	v_cvt_pk_f16_f32 v162, v174, v175
000000004CC8:	global_store_dwordx2 v[194:195], v[146:147], off offset:160
000000004CD0:	v_cvt_pk_f16_f32 v147, v160, v161
000000004CD8:	v_cvt_pk_f16_f32 v146, v158, v159
000000004CE0:	v_lshl_add_u64 v[130:131], v[130:131], 0, v[0:1]
000000004CE8:	v_cvt_pk_f16_f32 v69, v68, v69
000000004CF0:	v_cvt_pk_f16_f32 v68, v66, v67
000000004CF8:	v_cvt_pk_f16_f32 v67, v72, v73
000000004D00:	v_cvt_pk_f16_f32 v66, v70, v71
000000004D08:	v_accvgpr_read_b32 v78, a204
000000004D10:	v_accvgpr_read_b32 v79, a205
000000004D18:	v_accvgpr_read_b32 v80, a206
000000004D20:	v_accvgpr_read_b32 v81, a207
000000004D28:	global_store_dwordx2 v[194:195], v[180:181], off
000000004D30:	global_store_dwordx2 v[194:195], v[178:179], off offset:48
000000004D38:	global_store_dwordx2 v[194:195], v[164:165], off offset:64
000000004D40:	global_store_dwordx2 v[194:195], v[162:163], off offset:112
000000004D48:	global_store_dwordx2 v[194:195], v[148:149], off offset:128
000000004D50:	global_store_dwordx2 v[194:195], v[146:147], off offset:176
000000004D58:	global_store_dwordx2 v[194:195], v[132:133], off offset:192
000000004D60:	global_store_dwordx2 v[130:131], v[66:67], off offset:208
000000004D68:	v_cvt_pk_f16_f32 v67, v76, v77
000000004D70:	v_cvt_pk_f16_f32 v66, v74, v75
000000004D78:	global_store_dwordx2 v[130:131], v[66:67], off offset:224
000000004D80:	v_cvt_pk_f16_f32 v67, v80, v81
000000004D88:	v_cvt_pk_f16_f32 v66, v78, v79
000000004D90:	v_accvgpr_read_b32 v129, a47
000000004D98:	v_accvgpr_read_b32 v113, a111
000000004DA0:	v_accvgpr_read_b32 v82, a144
000000004DA8:	global_store_dwordx2 v[130:131], v[66:67], off offset:240
000000004DB0:	v_or_b32_e32 v66, 0x60, v250
000000004DB8:	v_accvgpr_read_b32 v121, a39
000000004DC0:	v_accvgpr_read_b32 v120, a38
000000004DC8:	v_accvgpr_read_b32 v119, a37
000000004DD0:	v_accvgpr_read_b32 v118, a36
000000004DD8:	v_accvgpr_read_b32 v117, a35
000000004DE0:	v_accvgpr_read_b32 v116, a34
000000004DE8:	v_accvgpr_read_b32 v115, a33
000000004DF0:	v_accvgpr_read_b32 v114, a32
000000004DF8:	v_accvgpr_read_b32 v105, a103
000000004E00:	v_accvgpr_read_b32 v104, a102
000000004E08:	v_accvgpr_read_b32 v103, a101
000000004E10:	v_accvgpr_read_b32 v102, a100
000000004E18:	v_accvgpr_read_b32 v101, a99
000000004E20:	v_accvgpr_read_b32 v100, a98
000000004E28:	v_accvgpr_read_b32 v99, a97
000000004E30:	v_accvgpr_read_b32 v98, a96
000000004E38:	v_accvgpr_read_b32 v83, a145
000000004E40:	v_accvgpr_read_b32 v84, a146
000000004E48:	v_accvgpr_read_b32 v85, a147
000000004E50:	v_accvgpr_read_b32 v86, a148
000000004E58:	v_accvgpr_read_b32 v87, a149
000000004E60:	v_accvgpr_read_b32 v88, a150
000000004E68:	v_accvgpr_read_b32 v89, a151
000000004E70:	v_mad_i64_i32 v[66:67], s[0:1], v66, s12, 0
000000004E78:	v_accvgpr_read_b32 v125, a43
000000004E80:	v_accvgpr_read_b32 v124, a42
000000004E88:	v_accvgpr_read_b32 v123, a41
000000004E90:	v_accvgpr_read_b32 v122, a40
000000004E98:	v_accvgpr_read_b32 v109, a107
000000004EA0:	v_accvgpr_read_b32 v108, a106
000000004EA8:	v_accvgpr_read_b32 v107, a105
000000004EB0:	v_accvgpr_read_b32 v106, a104
000000004EB8:	v_accvgpr_read_b32 v90, a152
000000004EC0:	v_accvgpr_read_b32 v91, a153
000000004EC8:	v_accvgpr_read_b32 v92, a154
000000004ED0:	v_accvgpr_read_b32 v93, a155
000000004ED8:	v_accvgpr_read_b32 v2, a48
000000004EE0:	v_cvt_pk_f16_f32 v117, v116, v117
000000004EE8:	v_cvt_pk_f16_f32 v116, v114, v115
000000004EF0:	v_cvt_pk_f16_f32 v115, v120, v121
000000004EF8:	v_cvt_pk_f16_f32 v114, v118, v119
000000004F00:	v_cvt_pk_f16_f32 v101, v100, v101
000000004F08:	v_cvt_pk_f16_f32 v100, v98, v99
000000004F10:	v_cvt_pk_f16_f32 v99, v104, v105
000000004F18:	v_cvt_pk_f16_f32 v98, v102, v103
000000004F20:	v_cvt_pk_f16_f32 v85, v84, v85
000000004F28:	v_cvt_pk_f16_f32 v84, v82, v83
000000004F30:	v_cvt_pk_f16_f32 v83, v88, v89
000000004F38:	v_cvt_pk_f16_f32 v82, v86, v87
000000004F40:	v_lshl_add_u64 v[66:67], v[66:67], 1, s[20:21]
000000004F48:	v_accvgpr_read_b32 v128, a46
000000004F50:	v_accvgpr_read_b32 v127, a45
000000004F58:	v_accvgpr_read_b32 v126, a44
000000004F60:	v_accvgpr_read_b32 v112, a110
000000004F68:	v_accvgpr_read_b32 v111, a109
000000004F70:	v_accvgpr_read_b32 v110, a108
000000004F78:	v_accvgpr_read_b32 v94, a156
000000004F80:	v_accvgpr_read_b32 v95, a157
000000004F88:	v_accvgpr_read_b32 v96, a158
000000004F90:	v_accvgpr_read_b32 v97, a159
000000004F98:	v_accvgpr_read_b32 v3, a49
000000004FA0:	v_accvgpr_read_b32 v4, a50
000000004FA8:	v_accvgpr_read_b32 v5, a51
000000004FB0:	v_accvgpr_read_b32 v6, a52
000000004FB8:	v_accvgpr_read_b32 v7, a53
000000004FC0:	v_accvgpr_read_b32 v8, a54
000000004FC8:	v_accvgpr_read_b32 v9, a55
000000004FD0:	global_store_dwordx2 v[130:131], v[114:115], off offset:16
000000004FD8:	v_cvt_pk_f16_f32 v115, v124, v125
000000004FE0:	v_cvt_pk_f16_f32 v114, v122, v123
000000004FE8:	global_store_dwordx2 v[130:131], v[98:99], off offset:80
000000004FF0:	v_cvt_pk_f16_f32 v99, v108, v109
000000004FF8:	v_cvt_pk_f16_f32 v98, v106, v107
000000005000:	global_store_dwordx2 v[130:131], v[82:83], off offset:144
000000005008:	v_cvt_pk_f16_f32 v83, v92, v93
000000005010:	v_cvt_pk_f16_f32 v82, v90, v91
000000005018:	v_lshl_add_u64 v[66:67], v[66:67], 0, v[242:243]
000000005020:	v_accvgpr_read_b32 v10, a56
000000005028:	v_accvgpr_read_b32 v11, a57
000000005030:	v_accvgpr_read_b32 v12, a58
000000005038:	v_accvgpr_read_b32 v13, a59
000000005040:	global_store_dwordx2 v[130:131], v[114:115], off offset:32
000000005048:	v_cvt_pk_f16_f32 v115, v128, v129
000000005050:	v_cvt_pk_f16_f32 v114, v126, v127
000000005058:	global_store_dwordx2 v[130:131], v[98:99], off offset:96
000000005060:	v_cvt_pk_f16_f32 v99, v112, v113
000000005068:	v_cvt_pk_f16_f32 v98, v110, v111
000000005070:	global_store_dwordx2 v[130:131], v[82:83], off offset:160
000000005078:	v_cvt_pk_f16_f32 v83, v96, v97
000000005080:	v_cvt_pk_f16_f32 v82, v94, v95
000000005088:	v_lshl_add_u64 v[0:1], v[66:67], 0, v[0:1]
000000005090:	v_cvt_pk_f16_f32 v5, v4, v5
000000005098:	v_cvt_pk_f16_f32 v4, v2, v3
0000000050A0:	v_cvt_pk_f16_f32 v3, v8, v9
0000000050A8:	v_cvt_pk_f16_f32 v2, v6, v7
0000000050B0:	v_accvgpr_read_b32 v14, a60
0000000050B8:	v_accvgpr_read_b32 v15, a61
0000000050C0:	v_accvgpr_read_b32 v16, a62
0000000050C8:	v_accvgpr_read_b32 v17, a63
0000000050D0:	v_accvgpr_read_b32 v33, a15
0000000050D8:	global_store_dwordx2 v[130:131], v[116:117], off
0000000050E0:	global_store_dwordx2 v[130:131], v[114:115], off offset:48
0000000050E8:	global_store_dwordx2 v[130:131], v[100:101], off offset:64
0000000050F0:	global_store_dwordx2 v[130:131], v[98:99], off offset:112
0000000050F8:	global_store_dwordx2 v[130:131], v[84:85], off offset:128
000000005100:	global_store_dwordx2 v[130:131], v[82:83], off offset:176
000000005108:	global_store_dwordx2 v[130:131], v[68:69], off offset:192
000000005110:	global_store_dwordx2 v[0:1], v[2:3], off offset:144
000000005118:	v_cvt_pk_f16_f32 v3, v12, v13
000000005120:	v_cvt_pk_f16_f32 v2, v10, v11
000000005128:	v_accvgpr_read_b32 v65, a31
000000005130:	v_accvgpr_read_b32 v34, a64
000000005138:	v_accvgpr_read_b32 v21, a3
000000005140:	v_accvgpr_read_b32 v20, a2
000000005148:	v_accvgpr_read_b32 v19, a1
000000005150:	v_accvgpr_read_b32 v18, a0
000000005158:	global_store_dwordx2 v[0:1], v[2:3], off offset:160
000000005160:	v_cvt_pk_f16_f32 v3, v16, v17
000000005168:	v_cvt_pk_f16_f32 v2, v14, v15
000000005170:	v_accvgpr_read_b32 v57, a23
000000005178:	v_accvgpr_read_b32 v56, a22
000000005180:	v_accvgpr_read_b32 v55, a21
000000005188:	v_accvgpr_read_b32 v54, a20
000000005190:	v_accvgpr_read_b32 v53, a19
000000005198:	v_accvgpr_read_b32 v52, a18
0000000051A0:	v_accvgpr_read_b32 v51, a17
0000000051A8:	v_accvgpr_read_b32 v50, a16
0000000051B0:	v_accvgpr_read_b32 v35, a65
0000000051B8:	v_accvgpr_read_b32 v36, a66
0000000051C0:	v_accvgpr_read_b32 v37, a67
0000000051C8:	v_accvgpr_read_b32 v38, a68
0000000051D0:	v_accvgpr_read_b32 v39, a69
0000000051D8:	v_accvgpr_read_b32 v40, a70
0000000051E0:	v_accvgpr_read_b32 v41, a71
0000000051E8:	v_accvgpr_read_b32 v25, a7
0000000051F0:	v_accvgpr_read_b32 v24, a6
0000000051F8:	v_accvgpr_read_b32 v23, a5
000000005200:	v_accvgpr_read_b32 v22, a4
000000005208:	global_store_dwordx2 v[0:1], v[2:3], off offset:176
000000005210:	v_cvt_pk_f16_f32 v3, v20, v21
000000005218:	v_cvt_pk_f16_f32 v2, v18, v19
000000005220:	v_accvgpr_read_b32 v61, a27
000000005228:	v_accvgpr_read_b32 v60, a26
000000005230:	v_accvgpr_read_b32 v59, a25
000000005238:	v_accvgpr_read_b32 v58, a24
000000005240:	v_accvgpr_read_b32 v42, a72
000000005248:	v_accvgpr_read_b32 v43, a73
000000005250:	v_accvgpr_read_b32 v44, a74
000000005258:	v_accvgpr_read_b32 v45, a75
000000005260:	v_accvgpr_read_b32 v29, a11
000000005268:	v_accvgpr_read_b32 v28, a10
000000005270:	v_accvgpr_read_b32 v27, a9
000000005278:	v_accvgpr_read_b32 v26, a8
000000005280:	v_cvt_pk_f16_f32 v53, v52, v53
000000005288:	v_cvt_pk_f16_f32 v52, v50, v51
000000005290:	v_cvt_pk_f16_f32 v51, v56, v57
000000005298:	v_cvt_pk_f16_f32 v50, v54, v55
0000000052A0:	v_cvt_pk_f16_f32 v37, v36, v37
0000000052A8:	v_cvt_pk_f16_f32 v36, v34, v35
0000000052B0:	v_cvt_pk_f16_f32 v35, v40, v41
0000000052B8:	v_cvt_pk_f16_f32 v34, v38, v39
0000000052C0:	global_store_dwordx2 v[0:1], v[2:3], off offset:192
0000000052C8:	v_cvt_pk_f16_f32 v3, v24, v25
0000000052D0:	v_cvt_pk_f16_f32 v2, v22, v23
0000000052D8:	v_accvgpr_read_b32 v64, a30
0000000052E0:	v_accvgpr_read_b32 v63, a29
0000000052E8:	v_accvgpr_read_b32 v62, a28
0000000052F0:	v_accvgpr_read_b32 v46, a76
0000000052F8:	v_accvgpr_read_b32 v47, a77
000000005300:	v_accvgpr_read_b32 v48, a78
000000005308:	v_accvgpr_read_b32 v49, a79
000000005310:	v_accvgpr_read_b32 v32, a14
000000005318:	v_accvgpr_read_b32 v31, a13
000000005320:	v_accvgpr_read_b32 v30, a12
000000005328:	global_store_dwordx2 v[0:1], v[50:51], off offset:16
000000005330:	v_cvt_pk_f16_f32 v51, v60, v61
000000005338:	v_cvt_pk_f16_f32 v50, v58, v59
000000005340:	global_store_dwordx2 v[0:1], v[34:35], off offset:80
000000005348:	v_cvt_pk_f16_f32 v35, v44, v45
000000005350:	v_cvt_pk_f16_f32 v34, v42, v43
000000005358:	global_store_dwordx2 v[0:1], v[2:3], off offset:208
000000005360:	v_cvt_pk_f16_f32 v3, v28, v29
000000005368:	v_cvt_pk_f16_f32 v2, v26, v27
000000005370:	global_store_dwordx2 v[0:1], v[50:51], off offset:32
000000005378:	v_cvt_pk_f16_f32 v51, v64, v65
000000005380:	v_cvt_pk_f16_f32 v50, v62, v63
000000005388:	global_store_dwordx2 v[0:1], v[34:35], off offset:96
000000005390:	v_cvt_pk_f16_f32 v35, v48, v49
000000005398:	v_cvt_pk_f16_f32 v34, v46, v47
0000000053A0:	global_store_dwordx2 v[0:1], v[2:3], off offset:224
0000000053A8:	v_cvt_pk_f16_f32 v3, v32, v33
0000000053B0:	v_cvt_pk_f16_f32 v2, v30, v31
0000000053B8:	global_store_dwordx2 v[0:1], v[52:53], off
0000000053C0:	global_store_dwordx2 v[0:1], v[50:51], off offset:48
0000000053C8:	global_store_dwordx2 v[0:1], v[36:37], off offset:64
0000000053D0:	global_store_dwordx2 v[0:1], v[34:35], off offset:112
0000000053D8:	global_store_dwordx2 v[0:1], v[4:5], off offset:128
0000000053E0:	global_store_dwordx2 v[0:1], v[2:3], off offset:240
0000000053E8:	s_endpgm
