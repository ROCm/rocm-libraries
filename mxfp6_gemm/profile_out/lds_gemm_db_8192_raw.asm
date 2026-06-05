; lds_gemm_db<256,256,192,2,2,occ1,SWZ=0,DB,__half> buffer_load_lds (MXFP6_M0_NOP=0) @8192^3
; ISA from RCV code.json, 2151 instr; cols: hit lat stall idle (summed over sampled waves)
; _ZN5mxfp611lds_gemm_dbILi256ELi256ELi192ELi2ELi2ELi1ELi0ELb1E6__halfEEvPKvS3_PKhS5_PT7_iiiiS5_S5_
s_load_dwordx8 s[4:11], s[0:1], 0x0                        ; hit=128 lat=512 stall=0 idle=28512
s_load_dwordx2 s[20:21], s[0:1], 0x20                      ; hit=128 lat=512 stall=0 idle=0
s_load_dwordx4 s[12:15], s[0:1], 0x28                      ; hit=128 lat=524 stall=0 idle=0
v_mul_u32_u24_e32 v2, 0x1c72, v0                           ; hit=128 lat=892 stall=380 idle=0
s_lshl_b32 s22, s2, 8                                      ; hit=128 lat=532 stall=0 idle=0
v_lshrrev_b32_e32 v5, 16, v2                               ; hit=128 lat=1200 stall=652 idle=0
v_mul_lo_u16_e32 v2, 9, v5                                 ; hit=128 lat=952 stall=424 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=128 lat=37548 stall=37548 idle=0
s_mul_i32 s17, s14, s22                                    ; hit=128 lat=512 stall=0 idle=0
s_mul_hi_i32 s16, s14, s22                                 ; hit=128 lat=512 stall=0 idle=0
s_add_u32 s4, s4, s17                                      ; hit=128 lat=512 stall=0 idle=0
v_sub_u16_e32 v2, v0, v2                                   ; hit=128 lat=1480 stall=964 idle=0
v_or_b32_e32 v92, 0x100, v0                                ; hit=128 lat=1456 stall=944 idle=0
s_addc_u32 s5, s5, s16                                     ; hit=128 lat=512 stall=0 idle=0
s_lshl_b32 s23, s3, 8                                      ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v4, 4, v2                                ; hit=128 lat=1484 stall=964 idle=0
v_mul_u32_u24_e32 v2, 0x1c72, v92                          ; hit=128 lat=1232 stall=720 idle=0
s_mul_i32 s17, s15, s23                                    ; hit=128 lat=520 stall=0 idle=0
v_lshrrev_b32_e32 v7, 16, v2                               ; hit=128 lat=744 stall=232 idle=0
v_lshrrev_b32_e32 v1, 6, v0                                ; hit=128 lat=1036 stall=524 idle=0
s_mul_hi_i32 s16, s15, s23                                 ; hit=128 lat=512 stall=0 idle=0
s_add_u32 s24, s6, s17                                     ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u16_e32 v2, 9, v7                                 ; hit=128 lat=1388 stall=876 idle=0
s_addc_u32 s18, s7, s16                                    ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[56:57], s[16:17], s14, v5, v[4:5]          ; hit=128 lat=1140 stall=628 idle=0
v_lshlrev_b32_e32 v1, 10, v1                               ; hit=128 lat=812 stall=300 idle=0
v_sub_u16_e32 v2, v92, v2                                  ; hit=128 lat=660 stall=148 idle=0
v_readfirstlane_b32 s16, v1                                ; hit=128 lat=816 stall=304 idle=0
v_lshlrev_b16_e32 v6, 4, v2                                ; hit=128 lat=520 stall=8 idle=0
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=1528
v_mad_u64_u32 v[58:59], s[16:17], s14, v7, v[6:7]          ; hit=128 lat=524 stall=12 idle=0
v_or_b32_e32 v59, 0x200, v0                                ; hit=128 lat=512 stall=0 idle=0
v_mul_u32_u24_e32 v2, 0x1c72, v59                          ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v9, 16, v2                               ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u16_e32 v2, 9, v9                                 ; hit=128 lat=512 stall=0 idle=0
v_sub_u16_e32 v2, v59, v2                                  ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v8, 4, v2                                ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v2, 0x300, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_mul_u32_u24_e32 v3, 0x1c72, v2                           ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v11, 16, v3                              ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u16_e32 v3, 9, v11                                ; hit=128 lat=512 stall=0 idle=0
v_sub_u16_e32 v2, v2, v3                                   ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v41, 0x1000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v10, 4, v2                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v2, 0x400, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s16, v41                               ; hit=128 lat=512 stall=0 idle=0
v_mul_u32_u24_e32 v3, 0x1c72, v2                           ; hit=128 lat=512 stall=0 idle=0
s_and_b32 s5, s5, 0xffff                                   ; hit=128 lat=512 stall=0 idle=1536
s_mov_b32 s7, 0x20000                                      ; hit=128 lat=512 stall=0 idle=0
s_brev_b32 s6, -2                                          ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v56, s[4:7], 0 offen lds               ; hit=128 lat=1116 stall=604 idle=512
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[60:61], s[16:17], s14, v9, v[8:9]          ; hit=128 lat=896 stall=384 idle=0
v_or_b32_e32 v45, 0x2000, v1                               ; hit=128 lat=604 stall=92 idle=0
v_lshrrev_b32_e32 v13, 16, v3                              ; hit=128 lat=692 stall=180 idle=0
v_readfirstlane_b32 s16, v45                               ; hit=128 lat=620 stall=108 idle=0
v_mul_lo_u16_e32 v3, 9, v13                                ; hit=128 lat=756 stall=244 idle=0
buffer_load_dwordx4 v58, s[4:7], 0 offen lds               ; hit=128 lat=532 stall=20 idle=1024
s_mov_b32 m0, s16                                          ; hit=128 lat=540 stall=0 idle=0
v_mad_u64_u32 v[62:63], s[16:17], s14, v11, v[10:11]       ; hit=128 lat=792 stall=280 idle=0
v_or_b32_e32 v49, 0x3000, v1                               ; hit=128 lat=708 stall=196 idle=0
v_sub_u16_e32 v3, v2, v3                                   ; hit=128 lat=856 stall=344 idle=0
v_readfirstlane_b32 s16, v49                               ; hit=128 lat=724 stall=212 idle=0
v_lshlrev_b16_e32 v12, 4, v3                               ; hit=128 lat=632 stall=120 idle=0
buffer_load_dwordx4 v60, s[4:7], 0 offen lds               ; hit=128 lat=584 stall=72 idle=1024
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[14:15], s[16:17], s14, v13, v[12:13]       ; hit=128 lat=600 stall=88 idle=0
v_or_b32_e32 v53, 0x4000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v3, 0x500, v0                                 ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v62, s[4:7], 0 offen lds               ; hit=128 lat=11684 stall=11172 idle=1024
v_readfirstlane_b32 s16, v53                               ; hit=128 lat=640 stall=0 idle=0
s_mov_b32 m0, s16                                          ; hit=128 lat=544 stall=0 idle=1920
buffer_load_dwordx4 v14, s[4:7], 0 offen lds               ; hit=128 lat=12636 stall=12116 idle=0
v_mul_u32_u24_e32 v14, 0x1c72, v3                          ; hit=128 lat=1256 stall=740 idle=0
v_lshrrev_b32_e32 v15, 16, v14                             ; hit=128 lat=652 stall=140 idle=0
v_mul_lo_u16_e32 v14, 9, v15                               ; hit=128 lat=604 stall=92 idle=0
v_sub_u16_e32 v3, v3, v14                                  ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v14, 4, v3                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v3, 0x600, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_mul_u32_u24_e32 v16, 0x1c72, v3                          ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v17, 16, v16                             ; hit=128 lat=544 stall=32 idle=0
v_mul_lo_u16_e32 v16, 9, v17                               ; hit=128 lat=648 stall=136 idle=0
v_sub_u16_e32 v3, v3, v16                                  ; hit=128 lat=836 stall=324 idle=0
v_lshlrev_b16_e32 v16, 4, v3                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v3, 0x700, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_mul_u32_u24_e32 v18, 0x1c72, v3                          ; hit=128 lat=596 stall=84 idle=0
v_lshrrev_b32_e32 v19, 16, v18                             ; hit=128 lat=592 stall=80 idle=0
v_mul_lo_u16_e32 v18, 9, v19                               ; hit=128 lat=708 stall=196 idle=0
v_sub_u16_e32 v3, v3, v18                                  ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[64:65], s[16:17], s14, v15, v[14:15]       ; hit=128 lat=572 stall=60 idle=0
v_or_b32_e32 v57, 0x5000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v18, 4, v3                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v3, 0x800, v0                                 ; hit=128 lat=636 stall=124 idle=9728
v_readfirstlane_b32 s16, v57                               ; hit=128 lat=624 stall=112 idle=0
v_mul_u32_u24_e32 v20, 0x1c72, v3                          ; hit=128 lat=712 stall=200 idle=0
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=1456
v_mad_u64_u32 v[66:67], s[16:17], s14, v17, v[16:17]       ; hit=128 lat=600 stall=88 idle=0
v_or_b32_e32 v63, 0x6000, v1                               ; hit=128 lat=612 stall=100 idle=0
v_lshrrev_b32_e32 v21, 16, v20                             ; hit=128 lat=584 stall=72 idle=0
v_readfirstlane_b32 s16, v63                               ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u16_e32 v20, 9, v21                               ; hit=128 lat=516 stall=4 idle=0
buffer_load_dwordx4 v64, s[4:7], 0 offen lds               ; hit=128 lat=524 stall=12 idle=1024
s_mov_b32 m0, s16                                          ; hit=128 lat=516 stall=0 idle=0
v_mad_u64_u32 v[68:69], s[16:17], s14, v19, v[18:19]       ; hit=128 lat=624 stall=108 idle=0
v_or_b32_e32 v65, 0x7000, v1                               ; hit=128 lat=648 stall=136 idle=0
v_sub_u16_e32 v20, v3, v20                                 ; hit=128 lat=548 stall=36 idle=0
v_readfirstlane_b32 s16, v65                               ; hit=128 lat=540 stall=28 idle=0
v_lshlrev_b16_e32 v20, 4, v20                              ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v66, s[4:7], 0 offen lds               ; hit=128 lat=1252 stall=740 idle=1024
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[22:23], s[16:17], s14, v21, v[20:21]       ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v67, 0x8000, v1                               ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v68, s[4:7], 0 offen lds               ; hit=128 lat=6052 stall=5540 idle=1024
s_and_b32 s26, s18, 0xffff                                 ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s16, v67                               ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 m0, s16                                          ; hit=128 lat=512 stall=0 idle=2048
s_mov_b64 s[18:19], s[6:7]                                 ; hit=128 lat=512 stall=0 idle=0
s_mov_b64 s[16:17], s[4:5]                                 ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 s17, s26                                         ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[70:71], s[26:27], s15, v5, v[4:5]          ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v69, 0x9000, v1                               ; hit=128 lat=628 stall=116 idle=0
s_mov_b32 s16, s24                                         ; hit=128 lat=512 stall=0 idle=1492
v_readfirstlane_b32 s24, v69                               ; hit=128 lat=524 stall=12 idle=0
v_mad_u64_u32 v[72:73], s[26:27], s15, v7, v[6:7]          ; hit=128 lat=552 stall=40 idle=0
v_or_b32_e32 v71, 0xa000, v1                               ; hit=128 lat=616 stall=104 idle=0
buffer_load_dwordx4 v22, s[4:7], 0 offen lds               ; hit=128 lat=18392 stall=17880 idle=1024
s_mov_b32 m0, s24                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[74:75], s[26:27], s15, v9, v[8:9]          ; hit=128 lat=532 stall=20 idle=0
v_readfirstlane_b32 s24, v71                               ; hit=128 lat=532 stall=20 idle=0
v_or_b32_e32 v73, 0xb000, v1                               ; hit=128 lat=528 stall=16 idle=0
buffer_load_dwordx4 v70, s[16:19], 0 offen lds             ; hit=128 lat=18044 stall=17532 idle=1024
s_mov_b32 m0, s24                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[76:77], s[26:27], s15, v11, v[10:11]       ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s24, v73                               ; hit=128 lat=516 stall=4 idle=0
v_or_b32_e32 v75, 0xc000, v1                               ; hit=128 lat=536 stall=24 idle=0
buffer_load_dwordx4 v72, s[16:19], 0 offen lds             ; hit=128 lat=14292 stall=13780 idle=1024
s_mov_b32 m0, s24                                          ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v77, 0xd000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s24, v75                               ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[78:79], s[28:29], s15, v15, v[14:15]       ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v74, s[16:19], 0 offen lds             ; hit=128 lat=22412 stall=21900 idle=1024
s_mov_b32 m0, s24                                          ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[4:5], s[26:27], s15, v13, v[12:13]         ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s24, v77                               ; hit=128 lat=520 stall=8 idle=0
v_or_b32_e32 v79, 0xe000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[80:81], s[28:29], s15, v17, v[16:17]       ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v76, s[16:19], 0 offen lds             ; hit=128 lat=26100 stall=25588 idle=1024
s_mov_b32 m0, s24                                          ; hit=128 lat=512 stall=0 idle=0
s_mul_hi_i32 s24, s13, 0x55555556                          ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s27, v79                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v81, 0xf000, v1                               ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[82:83], s[28:29], s15, v19, v[18:19]       ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v55, 7, v0                               ; hit=128 lat=512 stall=0 idle=0
buffer_load_dwordx4 v4, s[16:19], 0 offen lds              ; hit=128 lat=30996 stall=30484 idle=1024
s_lshr_b32 s26, s24, 31                                    ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s27, v81                               ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v83, 0x10000, v1                              ; hit=128 lat=512 stall=0 idle=0
v_lshl_or_b32 v22, s2, 1, v55                              ; hit=128 lat=512 stall=0 idle=0
s_add_i32 s24, s24, s26                                    ; hit=128 lat=512 stall=0 idle=1024
buffer_load_dwordx4 v78, s[16:19], 0 offen lds             ; hit=128 lat=19244 stall=18732 idle=512
s_mov_b32 m0, s27                                          ; hit=128 lat=512 stall=0 idle=0
v_readfirstlane_b32 s27, v83                               ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[4:5], s[28:29], s15, v21, v[20:21]         ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v93, 0x11000, v1                              ; hit=128 lat=536 stall=24 idle=0
v_bfe_u32 v242, v0, 6, 1                                   ; hit=128 lat=512 stall=0 idle=0
s_lshl_b32 s25, s3, 1                                      ; hit=128 lat=512 stall=0 idle=1016
buffer_load_dwordx4 v80, s[16:19], 0 offen lds             ; hit=128 lat=20004 stall=19492 idle=512
s_mov_b32 m0, s27                                          ; hit=128 lat=640 stall=0 idle=0
buffer_load_dwordx4 v82, s[16:19], 0 offen lds             ; hit=128 lat=29472 stall=28960 idle=0
v_readfirstlane_b32 s27, v93                               ; hit=128 lat=760 stall=92 idle=0
s_mov_b32 m0, s27                                          ; hit=128 lat=628 stall=0 idle=1892
buffer_load_dwordx4 v4, s[16:19], 0 offen lds              ; hit=128 lat=48688 stall=48176 idle=0
v_mul_lo_u32 v4, s24, v22                                  ; hit=128 lat=564 stall=40 idle=0
v_and_b32_e32 v54, 63, v0                                  ; hit=128 lat=584 stall=68 idle=0
v_or_b32_e32 v6, s25, v242                                 ; hit=128 lat=576 stall=64 idle=0
v_lshlrev_b32_e32 v94, 6, v4                               ; hit=128 lat=548 stall=36 idle=0
v_or_b32_e32 v4, v94, v54                                  ; hit=128 lat=596 stall=84 idle=0
v_mul_lo_u32 v6, s24, v6                                   ; hit=128 lat=688 stall=176 idle=0
v_mad_i64_i32 v[4:5], s[28:29], v4, 12, s[8:9]             ; hit=128 lat=776 stall=264 idle=0
v_lshl_or_b32 v6, v6, 6, v54                               ; hit=128 lat=564 stall=52 idle=0
global_load_dwordx3 v[42:44], v[4:5], off                  ; hit=128 lat=38196 stall=37648 idle=1024
v_mad_i64_i32 v[6:7], s[28:29], v6, 12, s[10:11]           ; hit=128 lat=1184 stall=532 idle=0
global_load_dwordx3 v[38:40], v[6:7], off                  ; hit=128 lat=28876 stall=28356 idle=884
s_movk_i32 s26, 0x1c72                                     ; hit=128 lat=512 stall=0 idle=40
s_cmp_lt_i32 s13, 6                                        ; hit=128 lat=524 stall=0 idle=0
v_and_b32_e32 v243, 31, v0                                 ; hit=128 lat=1068 stall=536 idle=0
v_lshrrev_b32_e32 v61, 5, v54                              ; hit=128 lat=848 stall=336 idle=0
s_cbranch_scc1 1619                                        ; hit=128 lat=512 stall=0 idle=0
v_and_b32_e32 v4, 31, v0                                   ; hit=128 lat=768 stall=256 idle=0
v_lshl_or_b32 v4, v242, 7, v4                              ; hit=128 lat=668 stall=156 idle=0
v_mul_u32_u24_e32 v115, 0x90, v4                           ; hit=128 lat=568 stall=56 idle=0
v_add_u32_e32 v4, s25, v242                                ; hit=128 lat=700 stall=188 idle=424
v_and_b32_e32 v5, 0x9f, v0                                 ; hit=128 lat=664 stall=152 idle=0
s_add_i32 s27, 0, 0x12000                                  ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u32 v4, s24, v4                                   ; hit=128 lat=716 stall=204 idle=0
v_mul_u32_u24_e32 v114, 0x90, v5                           ; hit=128 lat=648 stall=132 idle=0
v_mad_u32_u24 v5, v61, 24, s27                             ; hit=128 lat=608 stall=96 idle=0
s_add_i32 s27, 0, 0x1b000                                  ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b32_e32 v116, 6, v4                              ; hit=128 lat=720 stall=208 idle=0
v_mul_u32_u24_sdwa v4, v3, s26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD ; hit=128 lat=528 stall=16 idle=0
v_mov_b32_e32 v12, 9                                       ; hit=128 lat=608 stall=96 idle=4
v_mad_u32_u24 v6, v61, 24, s27                             ; hit=128 lat=556 stall=44 idle=0
s_add_i32 s27, 0, 0x12030                                  ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 s25, 0x1c71c71d                                  ; hit=128 lat=512 stall=0 idle=0
v_mul_lo_u16_sdwa v4, v4, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD ; hit=128 lat=632 stall=120 idle=0
v_mul_u32_u24_sdwa v13, v2, s26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD ; hit=128 lat=544 stall=32 idle=0
v_mad_u32_u24 v7, v61, 24, s27                             ; hit=128 lat=544 stall=32 idle=0
s_add_i32 s27, 0, 0x1b030                                  ; hit=128 lat=512 stall=0 idle=0
v_mul_hi_u32 v11, v3, s25                                  ; hit=128 lat=572 stall=60 idle=496
v_sub_u16_e32 v3, v3, v4                                   ; hit=128 lat=540 stall=28 idle=0
v_mul_lo_u16_sdwa v12, v13, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD ; hit=128 lat=512 stall=0 idle=0
v_mad_u32_u24 v8, v61, 24, s27                             ; hit=128 lat=512 stall=0 idle=0
s_add_i32 s27, 0, 0x12060                                  ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v4, 4, v3                                ; hit=128 lat=560 stall=48 idle=0
v_mad_u32_u24 v9, v61, 24, s27                             ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[84:85], s[28:29], s15, v11, v[4:5]         ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v95, 0x12000, v1                              ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v96, 0x13000, v1                              ; hit=128 lat=516 stall=0 idle=0
s_add_i32 s27, 0, 0x1b060                                  ; hit=128 lat=512 stall=0 idle=1020
v_mad_u32_u24 v10, v61, 24, s27                            ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v97, 0x14000, v1                              ; hit=128 lat=512 stall=0 idle=0
v_mul_hi_u32 v3, v2, s25                                   ; hit=128 lat=520 stall=8 idle=0
v_sub_u16_e32 v2, v2, v12                                  ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b16_e32 v2, 4, v2                                ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[90:91], s[26:27], s14, v3, v[2:3]          ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v98, 0x15000, v1                              ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[88:89], s[26:27], s14, v11, v[4:5]         ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v99, 0x16000, v1                              ; hit=128 lat=512 stall=0 idle=0
v_mad_u64_u32 v[86:87], s[26:27], s15, v3, v[2:3]          ; hit=128 lat=524 stall=12 idle=0
v_or_b32_e32 v100, 0x17000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v101, 0x18000, v1                             ; hit=128 lat=512 stall=0 idle=508
v_or_b32_e32 v102, 0x19000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v103, 0x1a000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v104, 0x1b000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v105, 0x1c000, v1                             ; hit=128 lat=512 stall=0 idle=516
v_or_b32_e32 v106, 0x1d000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v107, 0x1e000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v108, 0x1f000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v109, 0x20000, v1                             ; hit=128 lat=512 stall=0 idle=520
v_or_b32_e32 v110, 0x21000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v111, 0x22000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v112, 0x23000, v1                             ; hit=128 lat=512 stall=0 idle=0
v_mad_u32_u24 v113, v61, 24, 0                             ; hit=128 lat=512 stall=0 idle=512
s_mov_b32 s26, 0                                           ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a111, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a110, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a109, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a108, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a107, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a106, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a105, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a104, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a103, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a102, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a101, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a100, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a99, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a98, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a97, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a96, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a175, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a174, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a173, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a172, 0                                ; hit=128 lat=512 stall=0 idle=8
v_accvgpr_write_b32 a171, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a170, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a169, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a168, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a167, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a166, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a165, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a164, 0                                ; hit=128 lat=512 stall=0 idle=4
v_accvgpr_write_b32 a163, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a162, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a161, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a160, 0                                ; hit=128 lat=512 stall=0 idle=4
v_accvgpr_write_b32 a223, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a222, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a221, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a220, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a219, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a218, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a217, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a216, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a215, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a214, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a213, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a212, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a211, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a210, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a209, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a208, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a255, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a254, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a253, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a252, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a251, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a250, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a249, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a248, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a247, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a246, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a245, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a244, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a243, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a242, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a241, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a240, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a79, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a78, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a77, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a76, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a75, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a74, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a73, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a72, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a71, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a70, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a69, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a68, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a67, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a66, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a65, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a64, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a143, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a142, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a141, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a140, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a139, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a138, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a137, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a136, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a135, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a134, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a133, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a132, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a131, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a130, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a129, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a128, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a191, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a190, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a189, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a188, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a187, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a186, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a185, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a184, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a183, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a182, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a181, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a180, 0                                ; hit=128 lat=512 stall=0 idle=4
v_accvgpr_write_b32 a179, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a178, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a177, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a176, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a239, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a238, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a237, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a236, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a235, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a234, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a233, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a232, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a231, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a230, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a229, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a228, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a227, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a226, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a225, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a224, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a31, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a30, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a29, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a28, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a27, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a26, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a25, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a24, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a23, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a22, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a21, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a20, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a19, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a18, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a17, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a16, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a95, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a94, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a93, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a92, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a91, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a90, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a89, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a88, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a87, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a86, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a85, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a84, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a83, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a82, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a81, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a80, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a159, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a158, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a157, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a156, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a155, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a154, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a153, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a152, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a151, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a150, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a149, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a148, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a147, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a146, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a145, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a144, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a207, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a206, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a205, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a204, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a203, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a202, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a201, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a200, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a199, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a198, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a197, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a196, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a195, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a194, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a193, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a192, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a15, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a14, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a13, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a12, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a11, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a10, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a9, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a8, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a7, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a6, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a5, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a4, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a3, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a2, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a1, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a0, 0                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a63, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a62, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a61, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a60, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a59, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a58, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a57, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a56, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a55, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a54, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a53, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a52, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a51, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a50, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a49, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a48, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a127, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a126, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a125, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a124, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a123, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a122, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a121, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a120, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a119, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a118, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a117, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a116, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a115, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a114, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a113, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a112, 0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a47, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a46, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a45, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a44, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a43, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a42, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a41, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a40, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a39, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a38, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a37, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a36, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a35, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a34, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a33, 0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_write_b32 a32, 0                                 ; hit=128 lat=512 stall=0 idle=4
v_add_u32_e32 v85, v5, v114                                ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v87, v6, v115                                ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v89, v7, v114                                ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v91, v8, v115                                ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v117, v9, v114                               ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v118, v10, v115                              ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 s25, 0                                           ; hit=128 lat=512 stall=0 idle=0
s_branch 368                                               ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v2, 16, v85                                  ; hit=2688 lat=13712 stall=212 idle=40324
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=2688 lat=10844 stall=72 idle=18756
ds_read2st64_b64 v[10:13], v2 offset0:18 offset1:27        ; hit=2688 lat=10936 stall=160 idle=0
v_add_u32_e32 v26, 16, v87                                 ; hit=2688 lat=10964 stall=208 idle=0
v_and_b32_e32 v119, 0xff, v50                              ; hit=2688 lat=11076 stall=324 idle=0
v_bfe_u32 v132, v50, 8, 8                                  ; hit=2688 lat=11184 stall=428 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=224940 stall=224940 idle=0
v_mov_b32_e32 v18, v4                                      ; hit=2688 lat=11696 stall=720 idle=0
v_mov_b32_e32 v19, v5                                      ; hit=2688 lat=12480 stall=1488 idle=0
ds_read_b128 v[14:17], v85                                 ; hit=2688 lat=11708 stall=948 idle=21264
ds_read_b128 v[2:5], v85 offset:4608                       ; hit=2688 lat=13532 stall=2776 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=32004 stall=32004 idle=0
v_mov_b32_e32 v24, v10                                     ; hit=2688 lat=11108 stall=308 idle=0
v_mov_b32_e32 v25, v11                                     ; hit=2688 lat=12048 stall=1204 idle=0
ds_read2st64_b64 v[28:31], v26 offset1:9                   ; hit=2688 lat=10828 stall=48 idle=21412
ds_read_b128 v[20:23], v85 offset:9216                     ; hit=2688 lat=34708 stall=23928 idle=0
ds_read_b128 v[8:11], v85 offset:13824                     ; hit=2688 lat=96084 stall=85304 idle=0
ds_read2st64_b64 v[122:125], v26 offset0:18 offset1:27     ; hit=2688 lat=40944 stall=30192 idle=0
v_bfe_u32 v133, v50, 16, 8                                 ; hit=2688 lat=11208 stall=456 idle=0
v_lshrrev_b32_e32 v50, 24, v50                             ; hit=2688 lat=11868 stall=1116 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=690940 stall=690940 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=11032 stall=248 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=11592 stall=808 idle=0
ds_read_b128 v[32:35], v87                                 ; hit=2688 lat=41692 stall=30940 idle=21472
ds_read_b128 v[26:29], v87 offset:4608                     ; hit=2688 lat=97548 stall=86796 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=284736 stall=284736 idle=0
v_mov_b32_e32 v130, v122                                   ; hit=2688 lat=10756 stall=4 idle=0
v_mov_b32_e32 v131, v123                                   ; hit=2688 lat=10768 stall=12 idle=0
ds_read_b128 v[126:129], v87 offset:9216                   ; hit=2688 lat=10752 stall=0 idle=21500
ds_read_b128 v[120:123], v87 offset:13824                  ; hit=2688 lat=36968 stall=26216 idle=0
v_and_b32_e32 v134, 0xff, v46                              ; hit=2688 lat=10780 stall=28 idle=0
v_bfe_u32 v135, v46, 8, 8                                  ; hit=2688 lat=11320 stall=568 idle=0
v_bfe_u32 v136, v46, 16, 8                                 ; hit=2688 lat=11716 stall=964 idle=0
v_lshrrev_b32_e32 v46, 24, v46                             ; hit=2688 lat=12588 stall=1836 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=694308 stall=694308 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[14:19], a[96:111], v134, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=22980 stall=1992 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v134, v132 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86020 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v134, v133 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86020 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[8:13], a[0:15], v134, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86028 stall=75276 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=10760 stall=10760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[14:19], a[160:175], v135, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75276 stall=64520 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v135, v132 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75260 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v135, v133 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75260 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[8:13], a[48:63], v135, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86012 stall=75260 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=15904 stall=15904 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[126:131], v[14:19], a[208:223], v136, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=74096 stall=63176 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[126:131], v[2:7], a[176:191], v136, v132 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[126:131], v[20:25], a[144:159], v136, v133 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[126:131], v[8:13], a[112:127], v136, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[120:125], v[14:19], a[240:255], v46, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[120:125], v[2:7], a[224:239], v46, v132 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_add_u32_e32 v2, 16, v89                                  ; hit=2688 lat=21504 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[120:125], v[20:25], a[192:207], v46, v133 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=64512 stall=53760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[120:125], v[8:13], a[32:47], v46, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=96768 stall=75264 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=2688 lat=11136 stall=384 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=2688 lat=11332 stall=580 idle=0
v_add_u32_e32 v26, 16, v91                                 ; hit=2688 lat=10752 stall=0 idle=0
v_and_b32_e32 v46, 0xff, v51                               ; hit=2688 lat=10752 stall=0 idle=0
v_bfe_u32 v50, v51, 8, 8                                   ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=428128 stall=428128 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=2688 lat=11888 stall=1136 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=2688 lat=10796 stall=44 idle=0
ds_read_b128 v[8:11], v89                                  ; hit=2688 lat=10752 stall=0 idle=21504
ds_read_b128 v[2:5], v89 offset:4608                       ; hit=2688 lat=10764 stall=12 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=62084 stall=62084 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=2688 lat=11000 stall=248 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=2688 lat=11300 stall=548 idle=0
ds_read2st64_b64 v[28:31], v26 offset1:9                   ; hit=2688 lat=10764 stall=12 idle=21504
ds_read_b128 v[20:23], v89 offset:9216                     ; hit=2688 lat=11688 stall=936 idle=0
ds_read_b128 v[14:17], v89 offset:13824                    ; hit=2688 lat=13048 stall=2296 idle=0
ds_read2st64_b64 v[122:125], v26 offset0:18 offset1:27     ; hit=2688 lat=29240 stall=18488 idle=0
v_bfe_u32 v119, v51, 16, 8                                 ; hit=2688 lat=10752 stall=0 idle=0
v_lshrrev_b32_e32 v51, 24, v51                             ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=622216 stall=622216 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[32:35], v91                                 ; hit=2688 lat=12160 stall=1408 idle=21504
ds_read_b128 v[26:29], v91 offset:4608                     ; hit=2688 lat=15700 stall=4948 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=341968 stall=341968 idle=0
v_mov_b32_e32 v130, v122                                   ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v131, v123                                   ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[126:129], v91 offset:9216                   ; hit=2688 lat=10788 stall=36 idle=21504
ds_read_b128 v[120:123], v91 offset:13824                  ; hit=2688 lat=11680 stall=928 idle=0
v_and_b32_e32 v132, 0xff, v47                              ; hit=2688 lat=10756 stall=4 idle=0
v_bfe_u32 v133, v47, 8, 8                                  ; hit=2688 lat=10752 stall=0 idle=0
v_bfe_u32 v134, v47, 16, 8                                 ; hit=2688 lat=11216 stall=464 idle=0
v_lshrrev_b32_e32 v47, 24, v47                             ; hit=2688 lat=11460 stall=708 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=167060 stall=167060 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v132, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=21572 stall=80 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v132, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v132, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v132, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86104 stall=75352 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v133, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v133, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v133, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v133, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=23292 stall=23292 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[126:131], v[8:13], a[208:223], v134, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=71572 stall=60560 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[126:131], v[2:7], a[176:191], v134, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86116 stall=75364 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[126:131], v[20:25], a[144:159], v134, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[126:131], v[14:19], a[112:127], v134, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[120:125], v[8:13], a[240:255], v47, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[120:125], v[2:7], a[224:239], v47, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_add_u32_e32 v2, 16, v117                                 ; hit=2688 lat=21504 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[120:125], v[20:25], a[192:207], v47, v119 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=64512 stall=53760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[120:125], v[14:19], a[32:47], v47, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=96768 stall=75264 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=2688 lat=10952 stall=200 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=2688 lat=11072 stall=320 idle=0
v_add_u32_e32 v26, 16, v118                                ; hit=2688 lat=10752 stall=0 idle=0
s_addk_i32 s26, 0x120                                      ; hit=2688 lat=10752 stall=0 idle=0
s_add_i32 s27, s25, 1                                      ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=592416 stall=592416 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=2688 lat=10844 stall=92 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=2688 lat=10796 stall=44 idle=0
ds_read_b128 v[8:11], v117                                 ; hit=2688 lat=10752 stall=0 idle=21504
ds_read_b128 v[2:5], v117 offset:4608                      ; hit=2688 lat=10752 stall=0 idle=252
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=109432 stall=109432 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=2688 lat=10768 stall=16 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=2688 lat=10752 stall=0 idle=0
ds_read2st64_b64 v[28:31], v26 offset1:9                   ; hit=2688 lat=10780 stall=28 idle=21504
ds_read_b128 v[20:23], v117 offset:9216                    ; hit=2688 lat=10844 stall=92 idle=0
ds_read_b128 v[14:17], v117 offset:13824                   ; hit=2688 lat=10820 stall=68 idle=0
ds_read2st64_b64 v[122:125], v26 offset0:18 offset1:27     ; hit=2688 lat=15084 stall=4332 idle=0
v_add_u32_e32 v116, 0x80, v116                             ; hit=2688 lat=10752 stall=0 idle=0
s_cmp_lt_i32 s27, s24                                      ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=618700 stall=618700 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[32:35], v118                                ; hit=2688 lat=10792 stall=40 idle=21504
ds_read_b128 v[26:29], v118 offset:4608                    ; hit=2688 lat=10992 stall=240 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=356200 stall=356200 idle=0
v_mov_b32_e32 v130, v122                                   ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v131, v123                                   ; hit=2688 lat=10820 stall=68 idle=0
ds_read_b128 v[126:129], v118 offset:9216                  ; hit=2688 lat=10756 stall=4 idle=21504
ds_read_b128 v[120:123], v118 offset:13824                 ; hit=2688 lat=10844 stall=92 idle=0
v_add_u32_e32 v94, 0x80, v94                               ; hit=2688 lat=10752 stall=0 idle=0
v_and_b32_e32 v46, 0xff, v52                               ; hit=2688 lat=11028 stall=276 idle=0
v_bfe_u32 v47, v52, 8, 8                                   ; hit=2688 lat=10880 stall=128 idle=0
v_bfe_u32 v50, v52, 16, 8                                  ; hit=2688 lat=10816 stall=64 idle=0
v_lshrrev_b32_e32 v51, 24, v52                             ; hit=2688 lat=11116 stall=364 idle=0
v_and_b32_e32 v52, 0xff, v48                               ; hit=2688 lat=10836 stall=84 idle=0
v_bfe_u32 v119, v48, 8, 8                                  ; hit=2688 lat=10764 stall=12 idle=8
v_bfe_u32 v132, v48, 16, 8                                 ; hit=2688 lat=10816 stall=64 idle=0
v_lshrrev_b32_e32 v48, 24, v48                             ; hit=2688 lat=10868 stall=116 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=45564 stall=45564 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v52, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=21652 stall=180 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v52, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86028 stall=75276 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v52, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v52, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=10820 stall=10820 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v119, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75220 stall=64464 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v119, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v119, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v119, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=13076 stall=13076 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[126:131], v[8:13], a[208:223], v132, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=73344 stall=62560 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[126:131], v[2:7], a[176:191], v132, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[126:131], v[20:25], a[144:159], v132, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[126:131], v[14:19], a[112:127], v132, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[120:125], v[8:13], a[240:255], v48, v46 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[120:125], v[2:7], a[224:239], v48, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[120:125], v[20:25], a[192:207], v48, v50 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[120:125], v[14:19], a[32:47], v48, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_cbranch_scc0 886                                         ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v119, s26, v56                               ; hit=2688 lat=10752 stall=0 idle=5740
v_add_u32_e32 v2, 0x90, v119                               ; hit=2688 lat=10760 stall=8 idle=0
v_add_u32_e32 v120, s26, v58                               ; hit=2688 lat=10756 stall=4 idle=0
s_barrier                                                  ; hit=2688 lat=1178892 stall=1178892 idle=0
v_readfirstlane_b32 s27, v95                               ; hit=2688 lat=10852 stall=100 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10752 stall=0 idle=43008
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=47208 stall=35380 idle=0
v_add_u32_e32 v2, 0x90, v120                               ; hit=2688 lat=23660 stall=12204 idle=0
v_add_u32_e32 v121, s26, v60                               ; hit=2688 lat=19848 stall=8660 idle=0
v_readfirstlane_b32 s27, v96                               ; hit=2688 lat=20572 stall=9364 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10768 stall=0 idle=42552
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=15432 stall=3520 idle=0
v_add_u32_e32 v2, 0x90, v121                               ; hit=2688 lat=25404 stall=14604 idle=0
v_add_u32_e32 v122, s26, v62                               ; hit=2688 lat=15920 stall=5156 idle=0
v_readfirstlane_b32 s27, v97                               ; hit=2688 lat=13776 stall=2972 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10796 stall=0 idle=42956
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=31124 stall=19212 idle=0
v_add_u32_e32 v2, 0x90, v122                               ; hit=2688 lat=31156 stall=19996 idle=0
v_add_u32_e32 v123, s26, v90                               ; hit=2688 lat=16572 stall=5796 idle=0
v_readfirstlane_b32 s27, v98                               ; hit=2688 lat=18820 stall=7500 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10796 stall=0 idle=42440
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=21500 stall=10292 idle=0
v_add_u32_e32 v2, 0x90, v123                               ; hit=2688 lat=11992 stall=1084 idle=0
v_add_u32_e32 v124, s26, v64                               ; hit=2688 lat=11284 stall=532 idle=0
v_readfirstlane_b32 s27, v99                               ; hit=2688 lat=11128 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10836 stall=0 idle=42632
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=22492 stall=11740 idle=0
v_add_u32_e32 v2, 0x90, v124                               ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v125, s26, v66                               ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v100                              ; hit=2688 lat=11140 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10844 stall=0 idle=42620
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=26628 stall=15876 idle=0
v_add_u32_e32 v2, 0x90, v125                               ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v126, s26, v68                               ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v101                              ; hit=2688 lat=10864 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11752 stall=0 idle=42896
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=76764 stall=66012 idle=0
v_add_u32_e32 v2, 0x90, v126                               ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v127, s26, v88                               ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v102                              ; hit=2688 lat=10800 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12940 stall=0 idle=42960
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=238476 stall=227720 idle=0
v_add_u32_e32 v2, 0x90, v127                               ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v128, s26, v70                               ; hit=2688 lat=10772 stall=12 idle=0
v_readfirstlane_b32 s27, v103                              ; hit=2688 lat=10940 stall=180 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13148 stall=0 idle=43000
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=510404 stall=499648 idle=0
v_add_u32_e32 v2, 0x90, v128                               ; hit=2688 lat=10840 stall=88 idle=0
v_add_u32_e32 v129, s26, v72                               ; hit=2688 lat=10776 stall=16 idle=0
v_readfirstlane_b32 s27, v104                              ; hit=2688 lat=10868 stall=108 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13196 stall=0 idle=43000
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=376336 stall=365580 idle=0
v_add_u32_e32 v2, 0x90, v129                               ; hit=2688 lat=10764 stall=12 idle=0
v_add_u32_e32 v130, s26, v74                               ; hit=2688 lat=10756 stall=0 idle=0
v_readfirstlane_b32 s27, v105                              ; hit=2688 lat=10836 stall=80 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13180 stall=0 idle=43004
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=265404 stall=254652 idle=0
v_add_u32_e32 v2, 0x90, v130                               ; hit=2688 lat=10772 stall=20 idle=0
v_add_u32_e32 v131, s26, v76                               ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v106                              ; hit=2688 lat=10756 stall=4 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13232 stall=0 idle=43008
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=304428 stall=293676 idle=0
v_add_u32_e32 v2, 0x90, v131                               ; hit=2688 lat=10760 stall=8 idle=0
v_add_u32_e32 v132, s26, v86                               ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v107                              ; hit=2688 lat=10824 stall=72 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13144 stall=0 idle=43008
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=386456 stall=375704 idle=0
v_add_u32_e32 v2, 0x90, v132                               ; hit=2688 lat=10800 stall=48 idle=0
v_add_u32_e32 v133, s26, v78                               ; hit=2688 lat=10756 stall=0 idle=0
v_readfirstlane_b32 s27, v108                              ; hit=2688 lat=10908 stall=152 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13200 stall=0 idle=43004
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=394692 stall=383928 idle=0
v_add_u32_e32 v2, 0x90, v133                               ; hit=2688 lat=10816 stall=64 idle=0
v_add_u32_e32 v134, s26, v80                               ; hit=2688 lat=10756 stall=0 idle=0
v_readfirstlane_b32 s27, v109                              ; hit=2688 lat=10848 stall=92 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13164 stall=0 idle=43004
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=367136 stall=356380 idle=0
v_add_u32_e32 v2, 0x90, v134                               ; hit=2688 lat=10884 stall=132 idle=0
v_add_u32_e32 v135, s26, v82                               ; hit=2688 lat=10756 stall=0 idle=0
v_readfirstlane_b32 s27, v110                              ; hit=2688 lat=10936 stall=180 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13204 stall=0 idle=43004
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=344440 stall=333684 idle=0
v_add_u32_e32 v2, 0x90, v135                               ; hit=2688 lat=10784 stall=32 idle=0
v_add_u32_e32 v136, s26, v84                               ; hit=2688 lat=10776 stall=12 idle=0
v_readfirstlane_b32 s27, v111                              ; hit=2688 lat=10856 stall=92 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13212 stall=0 idle=42996
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=488724 stall=477972 idle=0
v_add_u32_e32 v2, 0x90, v136                               ; hit=2688 lat=10784 stall=32 idle=0
v_add_u32_e32 v137, v54, v94                               ; hit=2688 lat=10760 stall=0 idle=0
v_readfirstlane_b32 s27, v112                              ; hit=2688 lat=10900 stall=140 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=13184 stall=0 idle=43000
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=604944 stall=594192 idle=0
v_add_u32_e32 v2, 64, v137                                 ; hit=2688 lat=10804 stall=52 idle=0
v_add_u32_e32 v138, v54, v116                              ; hit=2688 lat=10760 stall=8 idle=0
v_mad_i64_i32 v[2:3], s[28:29], v2, 12, s[8:9]             ; hit=2688 lat=10772 stall=20 idle=0
v_add_u32_e32 v4, 64, v138                                 ; hit=2688 lat=10752 stall=0 idle=0
global_load_dwordx3 v[50:52], v[2:3], off                  ; hit=2688 lat=428052 stall=417292 idle=21504
v_add_u32_e32 v139, v113, v114                             ; hit=2688 lat=10864 stall=28 idle=0
v_mad_i64_i32 v[4:5], s[28:29], v4, 12, s[10:11]           ; hit=2688 lat=15096 stall=412 idle=0
global_load_dwordx3 v[46:48], v[4:5], off                  ; hit=2688 lat=341344 stall=330532 idle=17572
v_add_u32_e32 v2, 16, v139                                 ; hit=2688 lat=13384 stall=668 idle=0
ds_read2st64_b64 v[10:13], v2 offset1:9                    ; hit=2688 lat=10988 stall=188 idle=19540
ds_read2st64_b64 v[4:7], v2 offset0:18 offset1:27          ; hit=2688 lat=10996 stall=208 idle=0
v_add_u32_e32 v140, v113, v115                             ; hit=2688 lat=11416 stall=504 idle=0
v_add_u32_e32 v26, 16, v140                                ; hit=2688 lat=11708 stall=896 idle=0
v_and_b32_e32 v141, 0xff, v42                              ; hit=2688 lat=12040 stall=1176 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=228020 stall=228020 idle=0
v_mov_b32_e32 v24, v10                                     ; hit=2688 lat=12052 stall=1272 idle=0
v_mov_b32_e32 v25, v11                                     ; hit=2688 lat=12588 stall=1512 idle=0
ds_read_b128 v[20:23], v139                                ; hit=2688 lat=10928 stall=176 idle=21180
ds_read_b128 v[8:11], v139 offset:4608                     ; hit=2688 lat=13528 stall=2776 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=39984 stall=39984 idle=0
v_mov_b32_e32 v18, v4                                      ; hit=2688 lat=11252 stall=472 idle=0
v_mov_b32_e32 v19, v5                                      ; hit=2688 lat=11812 stall=984 idle=0
ds_read2st64_b64 v[144:147], v26 offset0:72 offset1:81     ; hit=2688 lat=10880 stall=128 idle=21428
ds_read_b128 v[14:17], v139 offset:9216                    ; hit=2688 lat=26172 stall=15420 idle=0
ds_read_b128 v[2:5], v139 offset:13824                     ; hit=2688 lat=82696 stall=71944 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:90 offset1:99       ; hit=2688 lat=42532 stall=31780 idle=0
v_bfe_u32 v154, v42, 8, 8                                  ; hit=2688 lat=11108 stall=336 idle=0
v_bfe_u32 v155, v42, 16, 8                                 ; hit=2688 lat=12096 stall=1304 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=684908 stall=684908 idle=0
v_mov_b32_e32 v152, v144                                   ; hit=2688 lat=10964 stall=204 idle=0
v_mov_b32_e32 v153, v145                                   ; hit=2688 lat=11704 stall=912 idle=0
ds_read_b128 v[148:151], v140 offset:36864                 ; hit=2688 lat=36164 stall=25412 idle=21464
ds_read_b128 v[142:145], v140 offset:41472                 ; hit=2688 lat=91936 stall=81184 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=280620 stall=280620 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=10772 stall=12 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=10884 stall=108 idle=0
ds_read_b128 v[32:35], v140 offset:46080                   ; hit=2688 lat=10872 stall=120 idle=21480
ds_read_b128 v[26:29], v140 offset:50688                   ; hit=2688 lat=33468 stall=22716 idle=0
v_lshrrev_b32_e32 v156, 24, v42                            ; hit=2688 lat=10852 stall=100 idle=0
v_and_b32_e32 v157, 0xff, v38                              ; hit=2688 lat=11308 stall=556 idle=0
v_bfe_u32 v158, v38, 8, 8                                  ; hit=2688 lat=11116 stall=364 idle=0
v_bfe_u32 v159, v38, 16, 8                                 ; hit=2688 lat=11492 stall=740 idle=0
v_lshrrev_b32_e32 v160, 24, v38                            ; hit=2688 lat=11524 stall=772 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=637512 stall=637512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[148:153], v[20:25], a[96:111], v157, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=21996 stall=648 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[148:153], v[8:13], a[64:79], v157, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86068 stall=75296 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[148:153], v[14:19], a[16:31], v157, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86044 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[148:153], v[2:7], a[0:15], v157, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86008 stall=75256 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=10772 stall=10772 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[142:147], v[20:25], a[160:175], v158, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75280 stall=64488 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[142:147], v[8:13], a[128:143], v158, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=85996 stall=75228 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[142:147], v[14:19], a[80:95], v158, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86012 stall=75248 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[142:147], v[2:7], a[48:63], v158, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86004 stall=75252 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=18144 stall=18144 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[32:37], v[20:25], a[208:223], v159, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=73816 stall=62848 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[8:13], a[176:191], v159, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75256 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[32:37], v[14:19], a[144:159], v159, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86008 stall=75256 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[2:7], a[112:127], v159, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10760 stall=10760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[26:31], v[20:25], a[240:255], v160, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75256 stall=64504 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v160, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[26:31], v[14:19], a[192:207], v160, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[2:7], a[32:47], v160, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_add_u32_e32 v2, 64, v139                                 ; hit=2688 lat=21504 stall=10752 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=2688 lat=10876 stall=124 idle=21504
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=2688 lat=11316 stall=564 idle=0
v_add_u32_e32 v26, 64, v140                                ; hit=2688 lat=10752 stall=0 idle=0
v_and_b32_e32 v141, 0xff, v43                              ; hit=2688 lat=10752 stall=0 idle=0
v_bfe_u32 v154, v43, 8, 8                                  ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=428612 stall=428612 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=2688 lat=11136 stall=384 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=2688 lat=10792 stall=40 idle=0
ds_read_b128 v[8:11], v139 offset:48                       ; hit=2688 lat=10776 stall=24 idle=21504
ds_read_b128 v[2:5], v139 offset:4656                      ; hit=2688 lat=10920 stall=168 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=79220 stall=79220 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=2688 lat=10848 stall=96 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81       ; hit=2688 lat=10760 stall=8 idle=21504
ds_read_b128 v[20:23], v139 offset:9264                    ; hit=2688 lat=11536 stall=784 idle=0
ds_read_b128 v[14:17], v139 offset:13872                   ; hit=2688 lat=13456 stall=2704 idle=0
ds_read2st64_b64 v[144:147], v26 offset0:90 offset1:99     ; hit=2688 lat=26024 stall=15272 idle=0
v_bfe_u32 v155, v43, 16, 8                                 ; hit=2688 lat=10764 stall=12 idle=0
v_lshrrev_b32_e32 v156, 24, v43                            ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=617592 stall=617592 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[32:35], v140 offset:36912                   ; hit=2688 lat=13736 stall=2984 idle=21504
ds_read_b128 v[26:29], v140 offset:41520                   ; hit=2688 lat=22236 stall=11484 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=326556 stall=326556 idle=0
v_mov_b32_e32 v152, v144                                   ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v153, v145                                   ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[148:151], v140 offset:46128                 ; hit=2688 lat=10752 stall=0 idle=21504
ds_read_b128 v[142:145], v140 offset:50736                 ; hit=2688 lat=11852 stall=1100 idle=0
v_and_b32_e32 v157, 0xff, v39                              ; hit=2688 lat=10756 stall=4 idle=0
v_bfe_u32 v158, v39, 8, 8                                  ; hit=2688 lat=10804 stall=52 idle=0
v_bfe_u32 v159, v39, 16, 8                                 ; hit=2688 lat=10936 stall=184 idle=0
v_lshrrev_b32_e32 v160, 24, v39                            ; hit=2688 lat=11112 stall=360 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=190440 stall=190440 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v157, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=21540 stall=52 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v157, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v157, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v157, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86072 stall=75320 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=10760 stall=10760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v158, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75260 stall=64504 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v158, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75260 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v158, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75260 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v158, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86012 stall=75260 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=19364 stall=19364 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[148:153], v[8:13], a[208:223], v159, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=71204 stall=60308 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[148:153], v[2:7], a[176:191], v159, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[148:153], v[20:25], a[144:159], v159, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[148:153], v[14:19], a[112:127], v159, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[142:147], v[8:13], a[240:255], v160, v141 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[142:147], v[2:7], a[224:239], v160, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_add_u32_e32 v2, 0x70, v139                               ; hit=2688 lat=21504 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[142:147], v[20:25], a[192:207], v160, v155 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=64512 stall=53760 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[142:147], v[14:19], a[32:47], v160, v156 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=96768 stall=75264 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=2688 lat=10968 stall=216 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=2688 lat=11216 stall=464 idle=0
v_add_u32_e32 v26, 0x70, v140                              ; hit=2688 lat=10752 stall=0 idle=0
s_add_i32 s25, s25, 2                                      ; hit=2688 lat=10752 stall=0 idle=0
s_cmp_ge_i32 s25, s24                                      ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=583388 stall=583388 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=2688 lat=11020 stall=268 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=2688 lat=10824 stall=72 idle=0
ds_read_b128 v[8:11], v139 offset:96                       ; hit=2688 lat=10752 stall=0 idle=21504
ds_read_b128 v[2:5], v139 offset:4704                      ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=113668 stall=113668 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=2688 lat=10760 stall=8 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=2688 lat=10760 stall=8 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81       ; hit=2688 lat=10756 stall=4 idle=21504
ds_read_b128 v[20:23], v139 offset:9312                    ; hit=2688 lat=10788 stall=36 idle=0
ds_read_b128 v[14:17], v139 offset:13920                   ; hit=2688 lat=11020 stall=268 idle=0
ds_read2st64_b64 v[142:145], v26 offset0:90 offset1:99     ; hit=2688 lat=13268 stall=2516 idle=0
v_and_b32_e32 v152, 0xff, v44                              ; hit=2688 lat=10752 stall=0 idle=0
v_bfe_u32 v153, v44, 8, 8                                  ; hit=2688 lat=10752 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=585300 stall=585300 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=2688 lat=10752 stall=0 idle=0
ds_read_b128 v[32:35], v140 offset:36960                   ; hit=2688 lat=10880 stall=128 idle=21504
ds_read_b128 v[26:29], v140 offset:41568                   ; hit=2688 lat=11588 stall=836 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=348740 stall=348740 idle=0
v_mov_b32_e32 v150, v142                                   ; hit=2688 lat=10752 stall=0 idle=0
v_mov_b32_e32 v151, v143                                   ; hit=2688 lat=10764 stall=12 idle=0
ds_read_b128 v[146:149], v140 offset:46176                 ; hit=2688 lat=10752 stall=0 idle=21504
ds_read_b128 v[140:143], v140 offset:50784                 ; hit=2688 lat=10808 stall=56 idle=0
v_bfe_u32 v154, v44, 16, 8                                 ; hit=2688 lat=10752 stall=0 idle=0
v_lshrrev_b32_e32 v139, 24, v44                            ; hit=2688 lat=10972 stall=220 idle=0
v_and_b32_e32 v155, 0xff, v40                              ; hit=2688 lat=10924 stall=172 idle=0
v_bfe_u32 v156, v40, 8, 8                                  ; hit=2688 lat=10960 stall=208 idle=0
v_bfe_u32 v157, v40, 16, 8                                 ; hit=2688 lat=10932 stall=180 idle=0
v_lshrrev_b32_e32 v158, 24, v40                            ; hit=2688 lat=10828 stall=76 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=2688 lat=72148 stall=72148 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v155, v152 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=21544 stall=60 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v155, v153 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v155, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v155, v139 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=2688 lat=11224 stall=11224 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v156, v152 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75032 stall=64240 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v156, v153 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v156, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v156, v139 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=2688 lat=13124 stall=13124 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[146:151], v[8:13], a[208:223], v157, v152 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=73264 stall=62464 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[146:151], v[2:7], a[176:191], v157, v153 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[146:151], v[20:25], a[144:159], v157, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[146:151], v[14:19], a[112:127], v157, v139 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=2688 lat=10752 stall=10752 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[140:145], v[8:13], a[240:255], v158, v152 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=75264 stall=64512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[140:145], v[2:7], a[224:239], v158, v153 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[140:145], v[20:25], a[192:207], v158, v154 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[140:145], v[14:19], a[32:47], v158, v139 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=2688 lat=86016 stall=75264 idle=0
s_barrier                                                  ; hit=2688 lat=1241536 stall=1241536 idle=0
s_cbranch_scc1 64661                                       ; hit=2688 lat=10752 stall=0 idle=0
v_add_u32_e32 v2, 0x120, v119                              ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v1                                ; hit=2688 lat=10752 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10752 stall=0 idle=43008
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=44688 stall=32624 idle=0
v_add_u32_e32 v2, 0x120, v120                              ; hit=2688 lat=15876 stall=5120 idle=0
v_readfirstlane_b32 s27, v41                               ; hit=2688 lat=14908 stall=1828 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10752 stall=0 idle=40680
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=10828 stall=0 idle=0
v_add_u32_e32 v2, 0x120, v121                              ; hit=2688 lat=11552 stall=800 idle=0
v_readfirstlane_b32 s27, v45                               ; hit=2688 lat=14232 stall=208 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10752 stall=0 idle=39736
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=10972 stall=0 idle=0
v_add_u32_e32 v2, 0x120, v122                              ; hit=2688 lat=11680 stall=520 idle=0
v_readfirstlane_b32 s27, v49                               ; hit=2688 lat=14848 stall=60 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10752 stall=0 idle=38972
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=11348 stall=0 idle=0
v_add_u32_e32 v2, 0x120, v123                              ; hit=2688 lat=12336 stall=1584 idle=0
v_readfirstlane_b32 s27, v53                               ; hit=2688 lat=14244 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=10892 stall=0 idle=39516
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=11732 stall=752 idle=0
v_add_u32_e32 v2, 0x120, v124                              ; hit=2688 lat=11376 stall=624 idle=0
v_readfirstlane_b32 s27, v57                               ; hit=2688 lat=14784 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12992 stall=0 idle=38976
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=68452 stall=57700 idle=0
v_add_u32_e32 v2, 0x120, v125                              ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v63                               ; hit=2688 lat=12648 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11200 stall=0 idle=41112
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=114940 stall=104188 idle=0
v_add_u32_e32 v2, 0x120, v126                              ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v65                               ; hit=2688 lat=14232 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12992 stall=0 idle=39528
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=147460 stall=136708 idle=0
v_add_u32_e32 v2, 0x120, v127                              ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v67                               ; hit=2688 lat=12648 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11200 stall=0 idle=41112
buffer_load_dwordx4 v2, s[4:7], 0 offen lds                ; hit=2688 lat=564248 stall=553488 idle=0
v_add_u32_e32 v2, 0x120, v128                              ; hit=2688 lat=10796 stall=44 idle=0
v_readfirstlane_b32 s27, v69                               ; hit=2688 lat=14232 stall=16 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12984 stall=0 idle=39544
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=354992 stall=344236 idle=0
v_add_u32_e32 v2, 0x120, v129                              ; hit=2688 lat=10772 stall=20 idle=0
v_readfirstlane_b32 s27, v71                               ; hit=2688 lat=12672 stall=8 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11212 stall=0 idle=41096
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=286588 stall=275832 idle=0
v_add_u32_e32 v2, 0x120, v130                              ; hit=2688 lat=10776 stall=24 idle=0
v_readfirstlane_b32 s27, v73                               ; hit=2688 lat=14232 stall=20 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12980 stall=0 idle=39548
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=285256 stall=274504 idle=0
v_add_u32_e32 v2, 0x120, v131                              ; hit=2688 lat=10780 stall=28 idle=0
v_readfirstlane_b32 s27, v75                               ; hit=2688 lat=12672 stall=8 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11212 stall=0 idle=41096
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=502708 stall=491956 idle=0
v_add_u32_e32 v2, 0x120, v132                              ; hit=2688 lat=10804 stall=52 idle=0
v_readfirstlane_b32 s27, v77                               ; hit=2688 lat=14220 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12980 stall=0 idle=39540
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=389620 stall=378868 idle=0
v_add_u32_e32 v2, 0x120, v133                              ; hit=2688 lat=10752 stall=0 idle=0
v_readfirstlane_b32 s27, v79                               ; hit=2688 lat=12660 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11212 stall=0 idle=41100
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=354684 stall=343932 idle=0
v_add_u32_e32 v2, 0x120, v134                              ; hit=2688 lat=10768 stall=16 idle=0
v_readfirstlane_b32 s27, v81                               ; hit=2688 lat=14220 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12980 stall=0 idle=39540
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=322788 stall=312036 idle=0
v_add_u32_e32 v2, 0x120, v135                              ; hit=2688 lat=10756 stall=4 idle=0
v_readfirstlane_b32 s27, v83                               ; hit=2688 lat=12660 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=11212 stall=0 idle=41100
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=461212 stall=450456 idle=0
v_add_u32_e32 v2, 0x120, v136                              ; hit=2688 lat=10780 stall=28 idle=0
v_readfirstlane_b32 s27, v93                               ; hit=2688 lat=14216 stall=0 idle=0
s_mov_b32 m0, s27                                          ; hit=2688 lat=12976 stall=0 idle=39544
buffer_load_dwordx4 v2, s[16:19], 0 offen lds              ; hit=2688 lat=631004 stall=620252 idle=0
v_add_u32_e32 v2, 0x80, v137                               ; hit=2688 lat=10752 stall=0 idle=0
v_mad_i64_i32 v[2:3], s[28:29], v2, 12, s[8:9]             ; hit=2688 lat=10756 stall=0 idle=0
v_add_u32_e32 v4, 0x80, v138                               ; hit=2688 lat=10764 stall=8 idle=0
global_load_dwordx3 v[42:44], v[2:3], off                  ; hit=2688 lat=441856 stall=431104 idle=21500
v_mad_i64_i32 v[4:5], s[28:29], v4, 12, s[10:11]           ; hit=2688 lat=13440 stall=0 idle=0
global_load_dwordx3 v[38:40], v[4:5], off                  ; hit=2688 lat=355108 stall=344356 idle=18816
s_branch 64540                                             ; hit=2688 lat=13440 stall=0 idle=0
v_accvgpr_write_b32 a32, 0                                 ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 s25, 0                                           ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a33, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a34, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a35, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a36, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a37, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a38, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a39, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a40, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a41, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a42, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a43, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a44, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a45, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a46, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a47, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a112, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a113, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a114, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a115, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a116, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a117, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a118, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a119, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a120, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a121, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a122, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a123, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a124, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a125, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a126, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a127, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a48, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a49, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a50, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a51, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a52, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a53, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a54, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a55, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a56, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a57, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a58, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a59, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a60, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a61, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a62, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a63, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a0, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a1, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a2, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a3, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a4, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a5, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a6, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a7, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a8, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a9, a32                                  ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a10, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a11, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a12, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a13, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a14, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a15, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a192, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a193, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a194, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a195, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a196, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a197, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a198, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a199, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a200, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a201, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a202, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a203, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a204, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a205, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a206, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a207, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a144, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a145, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a146, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a147, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a148, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a149, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a150, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a151, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a152, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a153, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a154, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a155, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a156, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a157, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a158, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a159, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a80, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a81, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a82, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a83, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a84, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a85, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a86, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a87, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a88, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a89, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a90, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a91, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a92, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a93, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a94, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a95, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a16, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a17, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a18, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a19, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a20, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a21, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a22, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a23, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a24, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a25, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a26, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a27, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a28, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a29, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a30, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a31, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a224, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a225, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a226, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a227, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a228, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a229, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a230, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a231, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a232, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a233, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a234, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a235, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a236, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a237, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a238, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a239, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a176, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a177, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a178, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a179, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a180, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a181, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a182, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a183, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a184, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a185, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a186, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a187, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a188, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a189, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a190, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a191, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a128, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a129, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a130, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a131, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a132, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a133, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a134, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a135, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a136, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a137, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a138, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a139, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a140, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a141, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a142, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a143, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a64, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a65, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a66, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a67, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a68, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a69, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a70, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a71, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a72, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a73, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a74, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a75, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a76, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a77, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a78, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a79, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a240, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a241, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a242, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a243, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a244, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a245, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a246, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a247, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a248, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a249, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a250, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a251, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a252, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a253, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a254, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a255, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a208, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a209, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a210, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a211, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a212, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a213, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a214, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a215, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a216, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a217, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a218, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a219, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a220, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a221, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a222, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a223, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a160, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a161, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a162, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a163, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a164, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a165, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a166, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a167, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a168, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a169, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a170, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a171, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a172, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a173, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a174, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a175, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a96, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a97, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a98, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a99, a32                                 ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a100, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a101, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a102, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a103, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a104, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a105, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a106, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a107, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a108, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a109, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a110, a32                                ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_mov_b32 a111, a32                                ; hit=0 lat=0 stall=0 idle=0
s_load_dwordx4 s[8:11], s[0:1], 0x38                       ; hit=128 lat=512 stall=0 idle=2048
s_cmp_ge_i32 s25, s24                                      ; hit=128 lat=512 stall=0 idle=0
s_cbranch_scc1 377                                         ; hit=128 lat=512 stall=0 idle=512
v_and_b32_e32 v2, 0x9f, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_mad_u32_u24 v14, v61, 24, 0                              ; hit=128 lat=512 stall=0 idle=0
s_movk_i32 s0, 0x90                                        ; hit=128 lat=512 stall=0 idle=1428
v_mad_u32_u24 v46, v2, s0, v14                             ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v2, 16, v46                                  ; hit=128 lat=512 stall=0 idle=0
s_waitcnt vmcnt(0)                                         ; hit=128 lat=512 stall=512 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=128 lat=720 stall=720 idle=0
s_barrier                                                  ; hit=128 lat=59152 stall=59152 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=128 lat=756 stall=244 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=128 lat=1024 stall=512 idle=0
v_and_b32_e32 v15, 31, v0                                  ; hit=128 lat=1508 stall=996 idle=0
v_lshl_or_b32 v15, v242, 7, v15                            ; hit=128 lat=1488 stall=976 idle=0
v_mad_u32_u24 v50, v15, s0, v14                            ; hit=128 lat=828 stall=316 idle=0
v_add_u32_e32 v26, 16, v50                                 ; hit=128 lat=584 stall=72 idle=2896
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=4648 stall=4648 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=128 lat=632 stall=120 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=128 lat=588 stall=76 idle=0
ds_read_b128 v[8:11], v46                                  ; hit=128 lat=516 stall=4 idle=1024
ds_read_b128 v[2:5], v46 offset:4608                       ; hit=128 lat=516 stall=4 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=4236 stall=4236 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=128 lat=804 stall=292 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=128 lat=624 stall=112 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81       ; hit=128 lat=584 stall=72 idle=1024
ds_read_b128 v[20:23], v46 offset:9216                     ; hit=128 lat=2204 stall=1692 idle=0
ds_read_b128 v[14:17], v46 offset:13824                    ; hit=128 lat=4512 stall=4000 idle=0
ds_read2st64_b64 v[64:67], v26 offset0:90 offset1:99       ; hit=128 lat=4692 stall=4180 idle=0
v_and_b32_e32 v47, 0xff, v42                               ; hit=128 lat=664 stall=152 idle=0
v_bfe_u32 v48, v42, 8, 8                                   ; hit=128 lat=548 stall=36 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=37744 stall=37744 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=128 lat=1412 stall=900 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=128 lat=592 stall=80 idle=0
ds_read_b128 v[32:35], v50 offset:36864                    ; hit=128 lat=628 stall=116 idle=1024
ds_read_b128 v[26:29], v50 offset:41472                    ; hit=128 lat=744 stall=232 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=17660 stall=17660 idle=0
v_mov_b32_e32 v72, v64                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v73, v65                                     ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[68:71], v50 offset:46080                    ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[62:65], v50 offset:50688                    ; hit=128 lat=572 stall=60 idle=0
v_bfe_u32 v51, v42, 16, 8                                  ; hit=128 lat=520 stall=8 idle=0
v_lshrrev_b32_e32 v42, 24, v42                             ; hit=128 lat=916 stall=404 idle=0
v_and_b32_e32 v52, 0xff, v38                               ; hit=128 lat=660 stall=148 idle=0
v_bfe_u32 v56, v38, 8, 8                                   ; hit=128 lat=1484 stall=972 idle=0
v_bfe_u32 v58, v38, 16, 8                                  ; hit=128 lat=588 stall=76 idle=0
v_lshrrev_b32_e32 v38, 24, v38                             ; hit=128 lat=532 stall=20 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=28444 stall=28444 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v52, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=1732 stall=880 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v52, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v52, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v52, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v56, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3584 stall=3072 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v56, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v56, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v56, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=736 stall=736 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[68:73], v[8:13], a[208:223], v58, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3456 stall=2920 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[68:73], v[2:7], a[176:191], v58, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[68:73], v[20:25], a[144:159], v58, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[68:73], v[14:19], a[112:127], v58, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[62:67], v[8:13], a[240:255], v38, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3584 stall=3072 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[62:67], v[2:7], a[224:239], v38, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_add_u32_e32 v2, 64, v46                                  ; hit=128 lat=1024 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[62:67], v[20:25], a[192:207], v38, v51 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3072 stall=2560 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[62:67], v[14:19], a[32:47], v38, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4608 stall=3584 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=128 lat=512 stall=0 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=128 lat=512 stall=0 idle=0
v_add_u32_e32 v26, 64, v50                                 ; hit=128 lat=512 stall=0 idle=0
v_and_b32_e32 v38, 0xff, v43                               ; hit=128 lat=512 stall=0 idle=0
v_bfe_u32 v42, v43, 8, 8                                   ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=16604 stall=16604 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=128 lat=736 stall=224 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=128 lat=520 stall=8 idle=0
ds_read_b128 v[8:11], v46 offset:48                        ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[2:5], v46 offset:4656                       ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=2172 stall=2172 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=128 lat=544 stall=32 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81       ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[20:23], v46 offset:9264                     ; hit=128 lat=556 stall=44 idle=0
ds_read_b128 v[14:17], v46 offset:13872                    ; hit=128 lat=848 stall=336 idle=0
ds_read2st64_b64 v[64:67], v26 offset0:90 offset1:99       ; hit=128 lat=1756 stall=1244 idle=0
v_bfe_u32 v47, v43, 16, 8                                  ; hit=128 lat=540 stall=28 idle=372
v_lshrrev_b32_e32 v43, 24, v43                             ; hit=128 lat=580 stall=68 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=32844 stall=32844 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=128 lat=520 stall=8 idle=0
ds_read_b128 v[32:35], v50 offset:36912                    ; hit=128 lat=700 stall=188 idle=1024
ds_read_b128 v[26:29], v50 offset:41520                    ; hit=128 lat=920 stall=408 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=16516 stall=16516 idle=0
v_mov_b32_e32 v72, v64                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v73, v65                                     ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[68:71], v50 offset:46128                    ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[62:65], v50 offset:50736                    ; hit=128 lat=616 stall=104 idle=0
v_and_b32_e32 v48, 0xff, v39                               ; hit=128 lat=516 stall=4 idle=0
v_bfe_u32 v51, v39, 8, 8                                   ; hit=128 lat=532 stall=20 idle=0
v_bfe_u32 v52, v39, 16, 8                                  ; hit=128 lat=612 stall=100 idle=0
v_lshrrev_b32_e32 v39, 24, v39                             ; hit=128 lat=552 stall=40 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=9464 stall=9464 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v48, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=1024 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v48, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v48, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v48, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v51, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3588 stall=3076 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v51, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v51, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v51, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=632 stall=632 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[68:73], v[8:13], a[208:223], v52, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3496 stall=2976 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[68:73], v[2:7], a[176:191], v52, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[68:73], v[20:25], a[144:159], v52, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[68:73], v[14:19], a[112:127], v52, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[62:67], v[8:13], a[240:255], v39, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3584 stall=3072 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[62:67], v[2:7], a[224:239], v39, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_add_u32_e32 v2, 0x70, v46                                ; hit=128 lat=1024 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[62:67], v[20:25], a[192:207], v39, v47 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3072 stall=2048 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[62:67], v[14:19], a[32:47], v39, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4608 stall=3584 idle=0
ds_read2st64_b64 v[4:7], v2 offset1:9                      ; hit=128 lat=512 stall=0 idle=0
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27        ; hit=128 lat=516 stall=4 idle=0
v_add_u32_e32 v26, 0x70, v50                               ; hit=128 lat=512 stall=0 idle=0
v_and_b32_e32 v38, 0xff, v44                               ; hit=128 lat=512 stall=0 idle=0
v_bfe_u32 v39, v44, 8, 8                                   ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=26192 stall=26192 idle=0
v_mov_b32_e32 v12, v4                                      ; hit=128 lat=540 stall=28 idle=0
v_mov_b32_e32 v13, v5                                      ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[8:11], v46 offset:96                        ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[2:5], v46 offset:4704                       ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=5540 stall=5540 idle=0
v_mov_b32_e32 v24, v16                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v25, v17                                     ; hit=128 lat=512 stall=0 idle=0
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81       ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[20:23], v46 offset:9312                     ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[14:17], v46 offset:13920                    ; hit=128 lat=512 stall=0 idle=0
ds_read2st64_b64 v[64:67], v26 offset0:90 offset1:99       ; hit=128 lat=560 stall=48 idle=0
v_bfe_u32 v42, v44, 16, 8                                  ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v43, 24, v44                             ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=32492 stall=32492 idle=0
v_mov_b32_e32 v36, v28                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v37, v29                                     ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[32:35], v50 offset:36960                    ; hit=128 lat=644 stall=132 idle=1024
ds_read_b128 v[26:29], v50 offset:41568                    ; hit=128 lat=932 stall=420 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=17068 stall=17068 idle=0
v_mov_b32_e32 v72, v64                                     ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v73, v65                                     ; hit=128 lat=512 stall=0 idle=0
ds_read_b128 v[68:71], v50 offset:46176                    ; hit=128 lat=512 stall=0 idle=1024
ds_read_b128 v[62:65], v50 offset:50784                    ; hit=128 lat=512 stall=0 idle=0
v_and_b32_e32 v44, 0xff, v40                               ; hit=128 lat=512 stall=0 idle=0
v_bfe_u32 v46, v40, 8, 8                                   ; hit=128 lat=512 stall=0 idle=0
v_bfe_u32 v47, v40, 16, 8                                  ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v40, 24, v40                             ; hit=128 lat=512 stall=0 idle=0
s_waitcnt lgkmcnt(3)                                       ; hit=128 lat=3288 stall=3288 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[8:13], a[96:111], v44, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=1024 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[2:7], a[64:79], v44, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4112 stall=3600 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[20:25], a[16:31], v44, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[14:19], a[0:15], v44, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4100 stall=3588 idle=0
s_waitcnt lgkmcnt(2)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[8:13], a[160:175], v46, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3584 stall=3072 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[26:31], v[2:7], a[128:143], v46, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v46, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[26:31], v[14:19], a[48:63], v46, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(1)                                       ; hit=128 lat=528 stall=528 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[68:73], v[8:13], a[208:223], v47, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3568 stall=3056 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[68:73], v[2:7], a[176:191], v47, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[68:73], v[20:25], a[144:159], v47, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[68:73], v[14:19], a[112:127], v47, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=128 lat=512 stall=512 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[62:67], v[8:13], a[240:255], v40, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=3584 stall=3072 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[62:67], v[2:7], a[224:239], v40, v39 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4096 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[62:67], v[20:25], a[192:207], v40, v42 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4100 stall=3584 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[62:67], v[14:19], a[32:47], v40, v43 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=128 lat=4092 stall=3580 idle=0
s_mul_i32 s25, s24, 3                                      ; hit=128 lat=512 stall=0 idle=0
s_cmp_lt_i32 s25, s13                                      ; hit=128 lat=512 stall=0 idle=0
s_cbranch_scc1 4                                           ; hit=128 lat=512 stall=0 idle=512
v_and_b32_e32 v2, 31, v0                                   ; hit=128 lat=512 stall=0 idle=2048
s_cbranch_execz 2                                          ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v243, v2                                     ; hit=128 lat=512 stall=0 idle=0
s_branch 357                                               ; hit=128 lat=512 stall=0 idle=0
s_mov_b32 s0, 0x55555556                                   ; hit=0 lat=0 stall=0 idle=0
v_mul_hi_u32 v2, v0, s0                                    ; hit=0 lat=0 stall=0 idle=0
v_mul_hi_u32 v3, v92, s0                                   ; hit=0 lat=0 stall=0 idle=0
v_mul_hi_u32 v4, v59, s0                                   ; hit=0 lat=0 stall=0 idle=0
s_lshl_b32 s0, s2, 3                                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_or_b32 v9, v242, 7, v243                            ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v17, 3, v4                               ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v23, s15, v4                                  ; hit=0 lat=0 stall=0 idle=0
v_lshl_or_b32 v8, v55, 2, s0                               ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v24, 48, v9                              ; hit=0 lat=0 stall=0 idle=0
v_and_b32_e32 v9, 0x9f, v0                                 ; hit=0 lat=0 stall=0 idle=0
s_mulk_i32 s24, 0x90                                       ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v18, 3, v3                               ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v21, s14, v3                                  ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v20, s14, v4                                  ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v22, s15, v3                                  ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b32_e32 v12, 2, v242                             ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v26, 48, v9                              ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v8, v8, s13                                   ; hit=0 lat=0 stall=0 idle=0
v_or_b32_e32 v9, 0x60, v0                                  ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v23, s24, v23                                ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b32_e32 v28, 4, v17                              ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v6, 3, v2                                ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v7, s14, v2                                   ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v19, s15, v2                                  ; hit=0 lat=0 stall=0 idle=0
v_mul_u32_u24_e32 v27, 48, v9                              ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v9, s13, v8                                  ; hit=0 lat=0 stall=0 idle=0
v_lshl_or_b32 v14, s3, 3, v12                              ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v17, v23, v28                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v22, s24, v22                                ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b32_e32 v23, 4, v18                              ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v20, s24, v20                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v21, s24, v21                                ; hit=0 lat=0 stall=0 idle=0
v_mad_u32_u24 v25, v61, 24, 0                              ; hit=0 lat=0 stall=0 idle=0
v_mov_b32_e32 v55, 0                                       ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v10, s13, v9                                 ; hit=0 lat=0 stall=0 idle=0
v_or_b32_e32 v12, 3, v14                                   ; hit=0 lat=0 stall=0 idle=0
v_or_b32_e32 v13, 2, v14                                   ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v14, s13, v14                                 ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v18, v22, v23                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v19, s24, v19                                ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b32_e32 v6, 4, v6                                ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v20, v20, v28                                ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v21, v21, v23                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v7, s24, v7                                  ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[2:3], s[8:9], 0, v[54:55]                 ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[4:5], s[10:11], 0, v[54:55]               ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v11, s13, v10                                ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v12, s13, v12                                 ; hit=0 lat=0 stall=0 idle=0
v_mul_lo_u32 v13, s13, v13                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v15, s13, v14                                ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b32_e32 v16, 4, v0                               ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v17, 0x2000, v17                             ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v18, 0x1000, v18                             ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v19, v19, v6                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v20, 0x2000, v20                             ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v21, 0x1000, v21                             ; hit=0 lat=0 stall=0 idle=0
v_sub_u32_e32 v22, v7, v6                                  ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v23, v25, v26                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v24, v25, v24                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v25, v25, v27                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v7, v16, v22                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v6, s25, v8                                  ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s0, v1                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v27, v16, v21                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v26, s25, v14                                ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s0                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v7, s[4:7], 0 offen lds                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v7, 31, v6                               ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s1, v41                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v28, v16, v20                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v29, v16, v19                                ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s1                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v27, s[4:7], 0 offen lds               ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v27, 31, v26                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[6:7], 6, v[6:7]                            ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s2, v45                                ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s3, v49                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v30, v16, v18                                ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s8, v53                                ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v31, v16, v17                                ; hit=0 lat=0 stall=0 idle=0
v_readfirstlane_b32 s9, v57                                ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s2                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v28, s[4:7], 0 offen lds               ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s3                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v29, s[16:19], 0 offen lds             ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s8                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v30, s[16:19], 0 offen lds             ; hit=0 lat=0 stall=0 idle=0
s_mov_b32 m0, s9                                           ; hit=0 lat=0 stall=0 idle=0
buffer_load_dwordx4 v31, s[16:19], 0 offen lds             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[26:27], 6, v[26:27]                        ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[28:29], v[2:3], 0, v[6:7]                 ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0)                                         ; hit=0 lat=0 stall=0 idle=0
s_barrier                                                  ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[6:7], v[4:5], 0, v[26:27]                 ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v40, v[28:29], off                       ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v42, v[6:7], off                         ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[26:29], v23                                 ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[30:31], v23 offset:16                        ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:12288                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:12304                     ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v38, s25, v15                                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v39, 31, v38                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[38:39], 6, v[38:39]                        ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[4:5], 0, v[38:39]               ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v17, 48, v17                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v18, 48, v18                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v19, 48, v19                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v20, 48, v20                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v21, 48, v21                                 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v22, 48, v22                                 ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[32:37], v[26:31], a[96:111], v42, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v32, s25, v13                                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v33, 31, v32                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[32:33], 6, v[32:33]                        ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v43, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[4:5], 0, v[32:33]               ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:13824                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:13840                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[32:37], v[26:31], a[160:175], v43, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v32, s25, v12                                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v33, 31, v32                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[32:33], 6, v[32:33]                        ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v44, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[4:5], 0, v[32:33]               ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:15360                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:15376                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[32:37], v[26:31], a[208:223], v44, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v32, s25, v9                                 ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v33, 31, v32                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[32:33], 6, v[32:33]                        ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v46, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[2:3], 0, v[32:33]               ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:16896                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:16912                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[26:31], a[240:255], v46, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v26, s25, v10                                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v27, 31, v26                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[26:27], 6, v[26:27]                        ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v40, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[2:3], 0, v[26:27]               ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[26:29], v23 offset:1536                     ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[30:31], v23 offset:1552                      ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:12288                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:12304                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[32:37], v[26:31], a[64:79], v42, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:13824                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:13840                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[32:37], v[26:31], a[128:143], v43, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:15360                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:15376                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[26:31], a[176:191], v44, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:16896                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:16912                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[32:37], v[26:31], a[224:239], v46, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
v_add_u32_e32 v26, s25, v11                                ; hit=0 lat=0 stall=0 idle=0
v_ashrrev_i32_e32 v27, 31, v26                             ; hit=0 lat=0 stall=0 idle=0
v_lshlrev_b64 v[26:27], 6, v[26:27]                        ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v40, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
v_lshl_add_u64 v[38:39], v[2:3], 0, v[26:27]               ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[26:29], v23 offset:3072                     ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[30:31], v23 offset:3088                      ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:12288                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:12304                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[32:37], v[26:31], a[16:31], v42, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:13824                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:13840                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[32:37], v[26:31], a[80:95], v43, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:15360                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:15376                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[32:37], v[26:31], a[144:159], v44, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:16896                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:16912                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[32:37], v[26:31], a[192:207], v46, v40 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v38, v[38:39], off                       ; hit=0 lat=0 stall=0 idle=0
s_nop 0                                                    ; hit=0 lat=0 stall=0 idle=0
global_load_ubyte v6, v[6:7], off                          ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[26:29], v25                                 ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[30:31], v25 offset:16                        ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:12288                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:12304                     ; hit=0 lat=0 stall=0 idle=0
s_add_i32 s25, s25, 1                                      ; hit=0 lat=0 stall=0 idle=0
s_cmp_ge_i32 s25, s13                                      ; hit=0 lat=0 stall=0 idle=0
s_waitcnt vmcnt(0) lgkmcnt(0)                              ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[32:37], v[26:31], a[0:15], v6, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:13824                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:13840                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[26:31], a[48:63], v43, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:15360                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:15376                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[26:31], a[112:127], v44, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
ds_read_b128 v[32:35], v24 offset:16896                    ; hit=0 lat=0 stall=0 idle=0
ds_read_b64 v[36:37], v24 offset:16912                     ; hit=0 lat=0 stall=0 idle=0
s_waitcnt lgkmcnt(0)                                       ; hit=0 lat=0 stall=0 idle=0
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[32:37], v[26:31], a[32:47], v46, v38 op_sel_hi:[0,0,0] cbsz:2 blgp:2 ; hit=0 lat=0 stall=0 idle=0
s_barrier                                                  ; hit=0 lat=0 stall=0 idle=0
s_cbranch_scc0 65267                                       ; hit=0 lat=0 stall=0 idle=0
v_accvgpr_read_b32 v2, a96                                 ; hit=128 lat=536 stall=24 idle=3128
v_and_b32_e32 v1, 0x80, v0                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v6, a100                                ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v7, a101                                ; hit=128 lat=512 stall=0 idle=504
v_accvgpr_read_b32 v8, a102                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v9, a103                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v10, a104                               ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v11, a105                               ; hit=128 lat=512 stall=0 idle=508
v_accvgpr_read_b32 v14, a108                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v15, a109                               ; hit=128 lat=520 stall=8 idle=0
v_add_u32_e32 v1, s22, v1                                  ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v3, a97                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v4, a98                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v5, a99                                 ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v9, v8, v9                                ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v8, v6, v7                                ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v10, v11                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v10, v14, v15                             ; hit=128 lat=512 stall=0 idle=0
v_or_b32_e32 v14, v1, v243                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v12, a106                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v13, a107                               ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v5, v4, v5                                ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v2, v3                                ; hit=128 lat=512 stall=0 idle=0
v_mad_i64_i32 v[2:3], s[0:1], v14, s12, 0                  ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v7, v12, v13                              ; hit=128 lat=512 stall=0 idle=512
v_lshl_add_u64 v[12:13], v[2:3], 1, s[20:21]               ; hit=128 lat=512 stall=0 idle=0
v_lshl_or_b32 v2, v242, 7, s23                             ; hit=128 lat=512 stall=0 idle=0
v_ashrrev_i32_e32 v3, 31, v2                               ; hit=128 lat=512 stall=0 idle=0
v_lshlrev_b64 v[2:3], 1, v[2:3]                            ; hit=128 lat=512 stall=0 idle=0
v_lshrrev_b32_e32 v0, 2, v0                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v241, a175                              ; hit=128 lat=512 stall=0 idle=0
v_lshl_add_u64 v[12:13], v[12:13], 0, v[2:3]               ; hit=128 lat=512 stall=0 idle=0
v_and_b32_e32 v0, 8, v0                                    ; hit=128 lat=512 stall=0 idle=0
v_mov_b32_e32 v1, 0                                        ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v229, a163                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v228, a162                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v227, a161                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v226, a160                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v16, a110                               ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v17, a111                               ; hit=128 lat=512 stall=0 idle=0
v_lshl_add_u64 v[12:13], v[12:13], 0, v[0:1]               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v233, a167                              ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v232, a166                              ; hit=128 lat=512 stall=0 idle=508
v_accvgpr_read_b32 v231, a165                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v230, a164                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v11, v16, v17                             ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off                 ; hit=128 lat=17796 stall=17156 idle=1024
global_store_dwordx2 v[12:13], v[8:9], off offset:16       ; hit=128 lat=5024 stall=4384 idle=0
global_store_dwordx2 v[12:13], v[6:7], off offset:32       ; hit=128 lat=5604 stall=4964 idle=0
global_store_dwordx2 v[12:13], v[10:11], off offset:48     ; hit=128 lat=5380 stall=4868 idle=0
v_cvt_pk_f16_f32 v5, v228, v229                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v226, v227                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v237, a171                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v236, a170                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v235, a169                              ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v234, a168                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:64       ; hit=128 lat=25392 stall=24880 idle=1024
v_cvt_pk_f16_f32 v5, v232, v233                            ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v230, v231                            ; hit=128 lat=584 stall=72 idle=0
v_accvgpr_read_b32 v225, a223                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v240, a174                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v239, a173                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v238, a172                              ; hit=128 lat=512 stall=0 idle=512
global_store_dwordx2 v[12:13], v[4:5], off offset:80       ; hit=128 lat=19364 stall=18852 idle=1024
v_cvt_pk_f16_f32 v5, v236, v237                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v234, v235                            ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v213, a211                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v212, a210                              ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v211, a209                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v210, a208                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:96       ; hit=128 lat=18864 stall=18352 idle=1024
v_cvt_pk_f16_f32 v5, v240, v241                            ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v238, v239                            ; hit=128 lat=544 stall=32 idle=0
v_accvgpr_read_b32 v217, a215                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v216, a214                              ; hit=128 lat=512 stall=0 idle=492
v_accvgpr_read_b32 v215, a213                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v214, a212                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:112      ; hit=128 lat=19588 stall=19076 idle=1024
v_cvt_pk_f16_f32 v5, v212, v213                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v210, v211                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v221, a219                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v220, a218                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v219, a217                              ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v218, a216                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:128      ; hit=128 lat=20428 stall=19916 idle=1024
v_cvt_pk_f16_f32 v5, v216, v217                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v214, v215                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v194, a240                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v224, a222                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v223, a221                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v222, a220                              ; hit=128 lat=512 stall=0 idle=512
global_store_dwordx2 v[12:13], v[4:5], off offset:144      ; hit=128 lat=23124 stall=22612 idle=1024
v_cvt_pk_f16_f32 v5, v220, v221                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v218, v219                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v195, a241                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v196, a242                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v197, a243                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:160      ; hit=128 lat=24644 stall=24132 idle=1024
v_cvt_pk_f16_f32 v5, v224, v225                            ; hit=128 lat=516 stall=4 idle=0
v_cvt_pk_f16_f32 v4, v222, v223                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v198, a244                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v199, a245                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v200, a246                              ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v201, a247                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:176      ; hit=128 lat=25300 stall=24788 idle=1024
v_cvt_pk_f16_f32 v5, v196, v197                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v194, v195                            ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v202, a248                              ; hit=128 lat=540 stall=28 idle=0
v_accvgpr_read_b32 v203, a249                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v204, a250                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v205, a251                              ; hit=128 lat=512 stall=0 idle=504
global_store_dwordx2 v[12:13], v[4:5], off offset:192      ; hit=128 lat=31700 stall=31188 idle=1024
v_cvt_pk_f16_f32 v5, v200, v201                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v198, v199                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v206, a252                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v207, a253                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v208, a254                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v209, a255                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:208      ; hit=128 lat=39872 stall=39360 idle=1024
v_cvt_pk_f16_f32 v5, v204, v205                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v202, v203                            ; hit=128 lat=640 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:224      ; hit=128 lat=40900 stall=40388 idle=896
v_cvt_pk_f16_f32 v5, v208, v209                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v4, v206, v207                            ; hit=128 lat=640 stall=0 idle=0
global_store_dwordx2 v[12:13], v[4:5], off offset:240      ; hit=128 lat=44256 stall=43744 idle=896
v_or_b32_e32 v4, 32, v14                                   ; hit=128 lat=512 stall=0 idle=0
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0                   ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v193, a79                               ; hit=128 lat=516 stall=4 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 1, s[20:21]                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v181, a67                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v180, a66                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v179, a65                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v178, a64                               ; hit=128 lat=512 stall=0 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]                   ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v185, a71                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v184, a70                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v183, a69                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v182, a68                               ; hit=128 lat=512 stall=0 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]                   ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v7, v180, v181                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v178, v179                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v189, a75                               ; hit=128 lat=604 stall=92 idle=0
v_accvgpr_read_b32 v188, a74                               ; hit=128 lat=724 stall=212 idle=0
v_accvgpr_read_b32 v187, a73                               ; hit=128 lat=724 stall=212 idle=0
v_accvgpr_read_b32 v186, a72                               ; hit=128 lat=624 stall=112 idle=0
global_store_dwordx2 v[4:5], v[6:7], off                   ; hit=128 lat=30184 stall=29672 idle=1024
v_cvt_pk_f16_f32 v7, v184, v185                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v182, v183                            ; hit=128 lat=532 stall=16 idle=0
v_accvgpr_read_b32 v177, a143                              ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v192, a78                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v191, a77                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v190, a76                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:16         ; hit=128 lat=35740 stall=35228 idle=1024
v_cvt_pk_f16_f32 v7, v188, v189                            ; hit=128 lat=528 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v186, v187                            ; hit=128 lat=540 stall=28 idle=0
v_accvgpr_read_b32 v165, a131                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v164, a130                              ; hit=128 lat=540 stall=28 idle=0
v_accvgpr_read_b32 v163, a129                              ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v162, a128                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:32         ; hit=128 lat=36596 stall=36084 idle=1024
v_cvt_pk_f16_f32 v7, v192, v193                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v190, v191                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v169, a135                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v168, a134                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v167, a133                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v166, a132                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:48         ; hit=128 lat=37280 stall=36768 idle=1024
v_cvt_pk_f16_f32 v7, v164, v165                            ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v162, v163                            ; hit=128 lat=520 stall=4 idle=0
v_accvgpr_read_b32 v173, a139                              ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v172, a138                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v171, a137                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v170, a136                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:64         ; hit=128 lat=36240 stall=35728 idle=1024
v_cvt_pk_f16_f32 v7, v168, v169                            ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v166, v167                            ; hit=128 lat=528 stall=12 idle=0
v_accvgpr_read_b32 v146, a176                              ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v176, a142                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v175, a141                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v174, a140                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:80         ; hit=128 lat=35616 stall=35104 idle=1024
v_cvt_pk_f16_f32 v7, v172, v173                            ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v170, v171                            ; hit=128 lat=552 stall=24 idle=0
v_accvgpr_read_b32 v147, a177                              ; hit=128 lat=544 stall=32 idle=0
v_accvgpr_read_b32 v148, a178                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v149, a179                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:96         ; hit=128 lat=36872 stall=36360 idle=1024
v_cvt_pk_f16_f32 v7, v176, v177                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v174, v175                            ; hit=128 lat=552 stall=32 idle=0
v_accvgpr_read_b32 v150, a180                              ; hit=128 lat=600 stall=88 idle=0
v_accvgpr_read_b32 v151, a181                              ; hit=128 lat=556 stall=44 idle=0
v_accvgpr_read_b32 v152, a182                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v153, a183                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:112        ; hit=128 lat=36320 stall=35808 idle=1024
v_cvt_pk_f16_f32 v7, v148, v149                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v146, v147                            ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v154, a184                              ; hit=128 lat=528 stall=12 idle=0
v_accvgpr_read_b32 v155, a185                              ; hit=128 lat=532 stall=20 idle=0
v_accvgpr_read_b32 v156, a186                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v157, a187                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:128        ; hit=128 lat=37036 stall=36524 idle=1024
v_cvt_pk_f16_f32 v7, v152, v153                            ; hit=128 lat=528 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v150, v151                            ; hit=128 lat=596 stall=80 idle=0
v_accvgpr_read_b32 v130, a224                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v158, a188                              ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v159, a189                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v160, a190                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v161, a191                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:144        ; hit=128 lat=36116 stall=35604 idle=1024
v_cvt_pk_f16_f32 v7, v156, v157                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v154, v155                            ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v131, a225                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v132, a226                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v133, a227                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:160        ; hit=128 lat=36752 stall=36240 idle=1024
v_cvt_pk_f16_f32 v7, v160, v161                            ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v158, v159                            ; hit=128 lat=532 stall=12 idle=0
v_accvgpr_read_b32 v134, a228                              ; hit=128 lat=544 stall=28 idle=0
v_accvgpr_read_b32 v135, a229                              ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v136, a230                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v137, a231                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:176        ; hit=128 lat=35724 stall=35212 idle=1024
v_cvt_pk_f16_f32 v7, v132, v133                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v130, v131                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v138, a232                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v139, a233                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v140, a234                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v141, a235                              ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:192        ; hit=128 lat=37092 stall=36580 idle=1024
v_cvt_pk_f16_f32 v7, v136, v137                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v134, v135                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v142, a236                              ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v143, a237                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v144, a238                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v145, a239                              ; hit=128 lat=524 stall=12 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:208        ; hit=128 lat=38604 stall=38092 idle=1024
v_cvt_pk_f16_f32 v7, v140, v141                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v138, v139                            ; hit=128 lat=640 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:224        ; hit=128 lat=38580 stall=38068 idle=896
v_cvt_pk_f16_f32 v7, v144, v145                            ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v142, v143                            ; hit=128 lat=644 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:240        ; hit=128 lat=39796 stall=39284 idle=892
v_or_b32_e32 v4, 64, v14                                   ; hit=128 lat=512 stall=0 idle=0
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0                   ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v129, a31                               ; hit=128 lat=532 stall=20 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 1, s[20:21]                 ; hit=128 lat=544 stall=32 idle=0
v_accvgpr_read_b32 v117, a19                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v116, a18                               ; hit=128 lat=512 stall=0 idle=496
v_accvgpr_read_b32 v115, a17                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v114, a16                               ; hit=128 lat=512 stall=0 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]                   ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v121, a23                               ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v120, a22                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v119, a21                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v118, a20                               ; hit=128 lat=516 stall=4 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]                   ; hit=128 lat=512 stall=0 idle=508
v_cvt_pk_f16_f32 v7, v116, v117                            ; hit=128 lat=632 stall=120 idle=0
v_cvt_pk_f16_f32 v6, v114, v115                            ; hit=128 lat=684 stall=172 idle=0
v_accvgpr_read_b32 v125, a27                               ; hit=128 lat=644 stall=128 idle=0
v_accvgpr_read_b32 v124, a26                               ; hit=128 lat=616 stall=104 idle=404
v_accvgpr_read_b32 v123, a25                               ; hit=128 lat=584 stall=72 idle=0
v_accvgpr_read_b32 v122, a24                               ; hit=128 lat=612 stall=96 idle=0
global_store_dwordx2 v[4:5], v[6:7], off                   ; hit=128 lat=29880 stall=29368 idle=1020
v_cvt_pk_f16_f32 v7, v120, v121                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v118, v119                            ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v113, a95                               ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v128, a30                               ; hit=128 lat=532 stall=20 idle=0
v_accvgpr_read_b32 v127, a29                               ; hit=128 lat=512 stall=0 idle=500
v_accvgpr_read_b32 v126, a28                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:16         ; hit=128 lat=36924 stall=36412 idle=1024
v_cvt_pk_f16_f32 v7, v124, v125                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v122, v123                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v101, a83                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v100, a82                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v99, a81                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v98, a80                                ; hit=128 lat=512 stall=0 idle=512
global_store_dwordx2 v[4:5], v[6:7], off offset:32         ; hit=128 lat=36800 stall=36288 idle=1024
v_cvt_pk_f16_f32 v7, v128, v129                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v126, v127                            ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v105, a87                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v104, a86                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v103, a85                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v102, a84                               ; hit=128 lat=556 stall=44 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:48         ; hit=128 lat=37952 stall=37440 idle=1024
v_cvt_pk_f16_f32 v7, v100, v101                            ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v98, v99                              ; hit=128 lat=516 stall=0 idle=0
v_accvgpr_read_b32 v109, a91                               ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v108, a90                               ; hit=128 lat=512 stall=0 idle=508
v_accvgpr_read_b32 v107, a89                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v106, a88                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:64         ; hit=128 lat=34964 stall=34452 idle=1024
v_cvt_pk_f16_f32 v7, v104, v105                            ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v102, v103                            ; hit=128 lat=536 stall=12 idle=0
v_accvgpr_read_b32 v82, a144                               ; hit=128 lat=580 stall=68 idle=0
v_accvgpr_read_b32 v112, a94                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v111, a93                               ; hit=128 lat=512 stall=0 idle=492
v_accvgpr_read_b32 v110, a92                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:80         ; hit=128 lat=35432 stall=34920 idle=1024
v_cvt_pk_f16_f32 v7, v108, v109                            ; hit=128 lat=532 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v106, v107                            ; hit=128 lat=568 stall=36 idle=0
v_accvgpr_read_b32 v83, a145                               ; hit=128 lat=576 stall=64 idle=0
v_accvgpr_read_b32 v84, a146                               ; hit=128 lat=536 stall=24 idle=0
v_accvgpr_read_b32 v85, a147                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:96         ; hit=128 lat=36648 stall=36136 idle=1024
v_cvt_pk_f16_f32 v7, v112, v113                            ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v110, v111                            ; hit=128 lat=544 stall=16 idle=0
v_accvgpr_read_b32 v86, a148                               ; hit=128 lat=584 stall=68 idle=0
v_accvgpr_read_b32 v87, a149                               ; hit=128 lat=540 stall=28 idle=484
v_accvgpr_read_b32 v88, a150                               ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v89, a151                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:112        ; hit=128 lat=35936 stall=35424 idle=1024
v_cvt_pk_f16_f32 v7, v84, v85                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v82, v83                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v90, a152                               ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v91, a153                               ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v92, a154                               ; hit=128 lat=512 stall=0 idle=504
v_accvgpr_read_b32 v93, a155                               ; hit=128 lat=528 stall=16 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:128        ; hit=128 lat=36240 stall=35728 idle=1024
v_cvt_pk_f16_f32 v7, v88, v89                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v86, v87                              ; hit=128 lat=572 stall=52 idle=0
v_accvgpr_read_b32 v66, a192                               ; hit=128 lat=612 stall=100 idle=0
v_accvgpr_read_b32 v94, a156                               ; hit=128 lat=544 stall=32 idle=0
v_accvgpr_read_b32 v95, a157                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v96, a158                               ; hit=128 lat=512 stall=0 idle=468
v_accvgpr_read_b32 v97, a159                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:144        ; hit=128 lat=35428 stall=34916 idle=1024
v_cvt_pk_f16_f32 v7, v92, v93                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v90, v91                              ; hit=128 lat=536 stall=24 idle=0
v_accvgpr_read_b32 v67, a193                               ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v68, a194                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v69, a195                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:160        ; hit=128 lat=36108 stall=35596 idle=1024
v_cvt_pk_f16_f32 v7, v96, v97                              ; hit=128 lat=520 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v94, v95                              ; hit=128 lat=548 stall=36 idle=0
v_accvgpr_read_b32 v70, a196                               ; hit=128 lat=560 stall=44 idle=0
v_accvgpr_read_b32 v71, a197                               ; hit=128 lat=528 stall=16 idle=484
v_accvgpr_read_b32 v72, a198                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v73, a199                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:176        ; hit=128 lat=35168 stall=34656 idle=1024
v_cvt_pk_f16_f32 v7, v68, v69                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v66, v67                              ; hit=128 lat=520 stall=4 idle=0
v_accvgpr_read_b32 v74, a200                               ; hit=128 lat=532 stall=20 idle=0
v_accvgpr_read_b32 v75, a201                               ; hit=128 lat=516 stall=4 idle=0
v_accvgpr_read_b32 v76, a202                               ; hit=128 lat=512 stall=0 idle=500
v_accvgpr_read_b32 v77, a203                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:192        ; hit=128 lat=36288 stall=35776 idle=1024
v_cvt_pk_f16_f32 v7, v72, v73                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v70, v71                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v78, a204                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v79, a205                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v80, a206                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v81, a207                               ; hit=128 lat=512 stall=0 idle=512
global_store_dwordx2 v[4:5], v[6:7], off offset:208        ; hit=128 lat=37164 stall=36652 idle=1024
v_cvt_pk_f16_f32 v7, v76, v77                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v74, v75                              ; hit=128 lat=648 stall=4 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:224        ; hit=128 lat=38916 stall=38404 idle=892
v_cvt_pk_f16_f32 v7, v80, v81                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v6, v78, v79                              ; hit=128 lat=644 stall=0 idle=0
global_store_dwordx2 v[4:5], v[6:7], off offset:240        ; hit=128 lat=38932 stall=38420 idle=892
v_or_b32_e32 v4, 0x60, v14                                 ; hit=128 lat=512 stall=0 idle=0
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0                   ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v65, a15                                ; hit=128 lat=556 stall=44 idle=0
v_lshl_add_u64 v[4:5], v[4:5], 1, s[20:21]                 ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v53, a3                                 ; hit=128 lat=512 stall=0 idle=492
v_accvgpr_read_b32 v52, a2                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v51, a1                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v50, a0                                 ; hit=128 lat=528 stall=16 idle=0
v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]                   ; hit=128 lat=512 stall=0 idle=508
v_accvgpr_read_b32 v57, a7                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v56, a6                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v55, a5                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v54, a4                                 ; hit=128 lat=512 stall=0 idle=512
v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]                   ; hit=128 lat=520 stall=8 idle=0
v_cvt_pk_f16_f32 v3, v52, v53                              ; hit=128 lat=584 stall=68 idle=0
v_cvt_pk_f16_f32 v2, v50, v51                              ; hit=128 lat=648 stall=132 idle=0
v_accvgpr_read_b32 v61, a11                                ; hit=128 lat=668 stall=156 idle=436
v_accvgpr_read_b32 v60, a10                                ; hit=128 lat=648 stall=136 idle=0
v_accvgpr_read_b32 v59, a9                                 ; hit=128 lat=600 stall=88 idle=0
v_accvgpr_read_b32 v58, a8                                 ; hit=128 lat=604 stall=92 idle=0
global_store_dwordx2 v[0:1], v[2:3], off                   ; hit=128 lat=29144 stall=28632 idle=1024
v_cvt_pk_f16_f32 v3, v56, v57                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v54, v55                              ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v34, a48                                ; hit=128 lat=536 stall=24 idle=0
v_accvgpr_read_b32 v64, a14                                ; hit=128 lat=512 stall=0 idle=504
v_accvgpr_read_b32 v63, a13                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v62, a12                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:16         ; hit=128 lat=35720 stall=35208 idle=1024
v_cvt_pk_f16_f32 v3, v60, v61                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v58, v59                              ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v35, a49                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v36, a50                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v37, a51                                ; hit=128 lat=536 stall=24 idle=504
global_store_dwordx2 v[0:1], v[2:3], off offset:32         ; hit=128 lat=36884 stall=36372 idle=1024
v_cvt_pk_f16_f32 v3, v64, v65                              ; hit=128 lat=524 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v62, v63                              ; hit=128 lat=532 stall=8 idle=0
v_accvgpr_read_b32 v38, a52                                ; hit=128 lat=548 stall=36 idle=0
v_accvgpr_read_b32 v39, a53                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v40, a54                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v41, a55                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:48         ; hit=128 lat=36304 stall=35792 idle=1024
v_cvt_pk_f16_f32 v3, v36, v37                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v34, v35                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v42, a56                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v43, a57                                ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v44, a58                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v45, a59                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:64         ; hit=128 lat=33824 stall=33312 idle=1024
v_cvt_pk_f16_f32 v3, v40, v41                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v38, v39                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v18, a112                               ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v46, a60                                ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v47, a61                                ; hit=128 lat=512 stall=0 idle=500
v_accvgpr_read_b32 v48, a62                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v49, a63                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:80         ; hit=128 lat=31432 stall=30920 idle=1024
v_cvt_pk_f16_f32 v3, v44, v45                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v42, v43                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v19, a113                               ; hit=128 lat=524 stall=12 idle=0
v_accvgpr_read_b32 v20, a114                               ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v21, a115                               ; hit=128 lat=512 stall=0 idle=504
global_store_dwordx2 v[0:1], v[2:3], off offset:96         ; hit=128 lat=30652 stall=30140 idle=1024
v_cvt_pk_f16_f32 v3, v48, v49                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v46, v47                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v22, a116                               ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v23, a117                               ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v24, a118                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v25, a119                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:112        ; hit=128 lat=28880 stall=28368 idle=1024
v_cvt_pk_f16_f32 v3, v20, v21                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v18, v19                              ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v26, a120                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v27, a121                               ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v28, a122                               ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v29, a123                               ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:128        ; hit=128 lat=26624 stall=26112 idle=1024
v_cvt_pk_f16_f32 v3, v24, v25                              ; hit=128 lat=532 stall=20 idle=0
v_cvt_pk_f16_f32 v2, v22, v23                              ; hit=128 lat=536 stall=20 idle=0
v_accvgpr_read_b32 v30, a124                               ; hit=128 lat=572 stall=60 idle=0
v_accvgpr_read_b32 v31, a125                               ; hit=128 lat=544 stall=32 idle=0
v_accvgpr_read_b32 v32, a126                               ; hit=128 lat=512 stall=0 idle=480
v_accvgpr_read_b32 v33, a127                               ; hit=128 lat=528 stall=16 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:144        ; hit=128 lat=25940 stall=25428 idle=1024
v_cvt_pk_f16_f32 v3, v28, v29                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v26, v27                              ; hit=128 lat=520 stall=8 idle=0
v_accvgpr_read_b32 v4, a32                                 ; hit=128 lat=524 stall=12 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:160        ; hit=128 lat=27004 stall=26492 idle=1024
v_cvt_pk_f16_f32 v3, v32, v33                              ; hit=128 lat=516 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v30, v31                              ; hit=128 lat=528 stall=12 idle=0
v_accvgpr_read_b32 v5, a33                                 ; hit=128 lat=540 stall=28 idle=0
v_accvgpr_read_b32 v6, a34                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v7, a35                                 ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:176        ; hit=128 lat=26196 stall=25684 idle=1024
v_accvgpr_read_b32 v8, a36                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v9, a37                                 ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v10, a38                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v11, a39                                ; hit=128 lat=528 stall=16 idle=512
v_cvt_pk_f16_f32 v3, v6, v7                                ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v4, v5                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v12, a40                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v13, a41                                ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v14, a42                                ; hit=128 lat=528 stall=16 idle=0
v_accvgpr_read_b32 v15, a43                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:192        ; hit=128 lat=23220 stall=22708 idle=1024
v_cvt_pk_f16_f32 v3, v10, v11                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v8, v9                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v16, a44                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v17, a45                                ; hit=128 lat=512 stall=0 idle=0
v_accvgpr_read_b32 v18, a46                                ; hit=128 lat=512 stall=0 idle=512
v_accvgpr_read_b32 v19, a47                                ; hit=128 lat=512 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:208        ; hit=128 lat=24872 stall=24360 idle=1024
v_cvt_pk_f16_f32 v3, v14, v15                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v12, v13                              ; hit=128 lat=580 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:224        ; hit=128 lat=26144 stall=25632 idle=956
v_cvt_pk_f16_f32 v3, v18, v19                              ; hit=128 lat=512 stall=0 idle=0
v_cvt_pk_f16_f32 v2, v16, v17                              ; hit=128 lat=580 stall=0 idle=0
global_store_dwordx2 v[0:1], v[2:3], off offset:240        ; hit=128 lat=24188 stall=23548 idle=956
s_endpgm                                                   ; hit=128 lat=190852 stall=0 idle=0
