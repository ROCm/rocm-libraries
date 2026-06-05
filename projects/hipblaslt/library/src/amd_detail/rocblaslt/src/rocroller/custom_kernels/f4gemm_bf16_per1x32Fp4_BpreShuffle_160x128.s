// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128:
	
	s_and_b32 s1, s1, 0xffff                                   
	s_load_dwordx2 s[4:5], s[0:1], 0x0                         
	s_load_dwordx2 s[8:9], s[0:1], 0x10                        
	s_load_dwordx2 s[12:13], s[0:1], 0x20                      
	s_load_dwordx2 s[16:17], s[0:1], 0x30                      
	s_load_dword s41, s[0:1], 0x40                             
	s_load_dword s42, s[0:1], 0x50                             
	s_load_dword s36, s[0:1], 0x80                             
	s_load_dword s37, s[0:1], 0xa0                             
	s_load_dword s38, s[0:1], 0xc0                             
	s_load_dword s43, s[0:1], 0xe0                             
	s_load_dword s44, s[0:1], 0xf0                             
	s_load_dword s45, s[0:1], 0x100                            
	s_load_dwordx2 s[20:21], s[0:1], 0x110                     
	s_load_dwordx2 s[24:25], s[0:1], 0x120                     
	s_load_dword s39, s[0:1], 0x130                            
	s_load_dword s40, s[0:1], 0x150                            
	v_lshrrev_b32_e32 v1, 10, v0                               
	v_lshrrev_b32_e32 v2, 10, v1                               
	v_and_b32_e32 v2, 0x3ff, v2                                
	v_and_b32_e32 v1, 0x3ff, v1                                
	v_and_b32_e32 v0, 0x3ff, v0                                
	v_lshrrev_b32_e32 v3, 6, v0                                
	v_and_b32_e32 v0, 63, v0                                   
	s_mov_b32 s47, s2                                          
	s_mov_b32 s48, s3                                          
	v_readfirstlane_b32 s46, v3                                
	s_waitcnt lgkmcnt(0)                                       
	s_add_u32 s51, s44, 0x7f                                   
	s_lshr_b32 s50, s51, 7                                     
	s_mul_i32 s49, s50, s48                                    
	s_add_i32 s49, s49, s47                                    
	s_add_u32 s51, s43, 0x9f                                   
	s_mov_b32 s63, 0xa0                                        
	v_cvt_f32_u32_e32 v4, s63                                  
	s_sub_i32 s62, 0, s63                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s62, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s51, v4                                   
	v_mul_lo_u32 v5, v4, s63                                   
	v_sub_u32_e32 v7, s51, v5                                  
	v_add_u32_e32 v6, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s63, v7                              
	v_subrev_u32_e32 v5, s63, v7                               
	s_nop 0                                                    
	v_cndmask_b32_e32 v4, v4, v6, vcc                          
	v_cndmask_b32_e32 v7, v7, v5, vcc                          
	v_add_u32_e32 v5, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s63, v7                              
	s_nop 1                                                    
	v_cndmask_b32_e32 v7, v4, v5, vcc                          
	s_nop 3                                                    
	v_readfirstlane_b32 s62, v7                                
	s_nop 3                                                    
	s_lshl_b32 s62, s62, 5                                     
	s_mov_b32 s47, 0                                           
	
label_0059:
	s_cmp_lt_i32 s49, s62                                      
	s_cbranch_scc1 label_005E                                  
	s_sub_i32 s49, s49, s62                                    
	s_add_i32 s47, s47, 32                                     
	s_branch label_0059                                        
	
label_005E:
	s_sub_i32 s50, s50, s47                                    
	s_cmp_lt_i32 s50, 32                                       
	s_cbranch_scc1 label_0064                                  
	s_lshr_b32 s48, s49, 5                                     
	s_and_b32 s62, s49, 31                                     
	s_branch label_0084                                        
	
label_0064:
	v_cvt_f32_u32_e32 v4, s50                                  
	s_sub_i32 s48, 0, s50                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s48, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s49, v4                                   
	v_mul_lo_u32 v5, v4, s50                                   
	v_sub_u32_e32 v7, s49, v5                                  
	v_add_u32_e32 v6, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s50, v7                              
	v_subrev_u32_e32 v5, s50, v7                               
	s_nop 0                                                    
	v_cndmask_b32_e32 v4, v4, v6, vcc                          
	v_cndmask_b32_e32 v7, v7, v5, vcc                          
	v_add_u32_e32 v5, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s50, v7                              
	s_nop 1                                                    
	v_cndmask_b32_e32 v7, v4, v5, vcc                          
	s_nop 3                                                    
	v_readfirstlane_b32 s48, v7                                
	s_nop 3                                                    
	s_mul_i32 s62, s50, s48                                    
	s_sub_i32 s62, s49, s62                                    
	
label_0084:
	s_add_i32 s47, s62, s47                                    
	s_lshr_b32 s37, s37, 1                                     
	s_mul_i32 s62, s48, 0xa0                                   
	s_mul_hi_u32 s63, s37, s62                                 
	s_add_u32 s13, s13, s63                                    
	s_mul_i32 s63, s37, s62                                    
	s_add_u32 s12, s12, s63                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_i32 s63, s43, s62                                    
	s_cmp_lt_u32 s63, 0xa0                                     
	s_cselect_b32 s62, s63, 0xa0                               
	s_mul_i32 s14, s37, s62                                    
	s_mov_b32 s15, 0x20000                                     
	v_lshrrev_b32_e32 v4, 3, v0                                
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_and_b32_e32 v4, 3, v4                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_lshlrev_b32_e32 v6, 2, v6                                
	v_add_u32_e32 v5, v5, v6                                   
	v_and_b32_e32 v4, 1, v4                                    
	v_add_u32_e32 v5, v5, v4                                   
	v_mul_lo_u32 v161, s37, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v161, v4, v161                               
	s_lshr_b32 s62, s46, 1                                     
	s_mul_i32 s62, s62, 8                                      
	s_and_b32 s63, s46, 1                                      
	s_mul_i32 s63, s63, 2                                      
	s_add_u32 s62, s62, s63                                    
	s_mul_i32 s62, s37, s62                                    
	v_add_u32_e32 v161, s62, v161                              
	s_mul_i32 s62, s37, 32                                     
	v_add_u32_e32 v162, s62, v161                              
	v_add_u32_e32 v163, s62, v162                              
	v_add_u32_e32 v164, s62, v163                              
	v_add_u32_e32 v165, s62, v164                              
	s_mul_i32 s64, 0x420, s46                                  
	s_add_u32 s64, 0x2000, s64                                 
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v166, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v166, v5, v166                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v166, v6, v166                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v166, v4, v166                               
	v_add_u32_e32 v166, 0x2000, v166                           
	v_add_u32_e32 v167, 0x5280, v166                           
	v_add_u32_e32 v168, 0x5280, v167                           
	v_add_u32_e32 v169, 0x5280, v168                           
	s_mul_i32 s62, s48, 0xa0                                   
	s_mul_hi_u32 s63, s39, s62                                 
	s_add_u32 s21, s21, s63                                    
	s_mul_i32 s63, s39, s62                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_add_u32 s63, s43, 31                                     
	s_lshr_b32 s63, s63, 5                                     
	s_lshl_b32 s63, s63, 5                                     
	s_sub_i32 s63, s63, s62                                    
	s_cmp_lt_u32 s63, 0xa0                                     
	s_cselect_b32 s62, s63, 0xa0                               
	s_mul_i32 s22, s39, s62                                    
	s_mov_b32 s23, 0x20000                                     
	v_lshlrev_b32_e32 v170, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v170, s63, v170                              
	s_mul_i32 s63, 0x80, s39                                   
	v_add_u32_e32 v171, s63, v170                              
	s_mul_i32 s65, s46, 0x100                                  
	s_add_i32 s65, s65, 0                                      
	v_lshlrev_b32_e32 v172, 2, v0                              
	v_add_u32_e32 v172, 0, v172                                
	s_lshr_b32 s38, s38, 1                                     
	s_mul_i32 s62, s47, 0x80                                   
	s_mul_hi_u32 s63, s38, s62                                 
	s_add_u32 s17, s17, s63                                    
	s_mul_i32 s63, s38, s62                                    
	s_add_u32 s16, s16, s63                                    
	s_addc_u32 s17, s17, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x80                                     
	s_cselect_b32 s62, s63, 0x80                               
	s_mul_i32 s18, s38, s62                                    
	s_mov_b32 s19, 0x20000                                     
	v_lshlrev_b32_e32 v173, 4, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v173, s62, v173                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v174, s62, v173                              
	s_mul_i32 s62, s47, 0x80                                   
	s_mul_hi_u32 s63, s40, s62                                 
	s_add_u32 s25, s25, s63                                    
	s_mul_i32 s63, s40, s62                                    
	s_add_u32 s24, s24, s63                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x80                                     
	s_cselect_b32 s62, s63, 0x80                               
	s_mul_i32 s26, s40, s62                                    
	s_mov_b32 s27, 0x20000                                     
	v_lshlrev_b32_e32 v175, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v175, s63, v175                              
	s_mov_b32 s66, 0x80                                        
	s_mov_b32 s67, 0x800                                       
	s_mov_b32 s68, 0x100                                       
	s_mov_b32 s69, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	buffer_load_dwordx4 v[88:91], v173, s[16:19], 0 offen      
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	buffer_load_dwordx4 v[92:95], v174, s[16:19], 0 offen      
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	buffer_load_dwordx4 v[96:99], v173, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	buffer_load_dwordx4 v[100:103], v174, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	buffer_load_dword v157, v175, s[24:27], 0 offen            
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	s_add_u32 s62, 0x100, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	s_add_u32 s63, 0x100, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	buffer_load_dword v158, v175, s[24:27], 0 offen            
	s_add_u32 s62, 0x200, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	s_add_u32 s63, 0x200, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_add_u32 m0, 0x1000, s65                                  
	buffer_load_dword v170, s[20:23], 0 offen lds              
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v171, s[20:23], 0 offen lds              
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	buffer_load_dwordx4 v[120:123], v173, s[16:19], 0 offen    
	buffer_load_dwordx4 v[124:127], v174, s[16:19], 0 offen    
	buffer_load_dwordx4 v[128:131], v173, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[132:135], v174, s[16:19], 0 offen offset:1024
	buffer_load_dword v159, v175, s[24:27], 0 offen            
	s_add_u32 s62, 0x300, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	s_add_u32 s63, 0x300, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_waitcnt vmcnt(32)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v166                                 
	ds_read_b128 v[16:19], v166 offset:64                      
	ds_read_b128 v[12:15], v166 offset:512                     
	ds_read_b128 v[20:23], v166 offset:576                     
	ds_read_b32 v152, v172                                     
	ds_read_b128 v[24:27], v166 offset:4224                    
	ds_read_b128 v[32:35], v166 offset:4288                    
	ds_read_b128 v[28:31], v166 offset:4736                    
	ds_read_b128 v[36:39], v166 offset:4800                    
	ds_read_b32 v153, v172 offset:256                          
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_lshl_b32 s36, s36, 1                                     
	s_mul_i32 s62, s48, 0xa0                                   
	s_mul_hi_u32 s63, s36, s62                                 
	s_add_u32 s5, s5, s63                                      
	s_mul_i32 s63, s36, s62                                    
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_mul_i32 s63, s47, 0x80                                   
	s_lshl_b32 s63, s63, 1                                     
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_sub_i32 s62, s43, s62                                    
	s_cmp_lt_u32 s62, 0xa0                                     
	s_cselect_b32 s62, s62, 0xa0                               
	s_mul_i32 s62, s36, s62                                    
	s_sub_i32 s6, s62, s63                                     
	s_mov_b32 s7, 0x20000                                      
	s_mul_i32 s62, s46, 32                                     
	s_lshl_b32 s62, s62, 1                                     
	v_lshrrev_b32_e32 v4, 5, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_lshrrev_b32_e32 v5, 4, v0                                
	v_and_b32_e32 v5, 1, v5                                    
	v_mul_i32_i24_e32 v5, 32, v5                               
	v_add_u32_e32 v4, v4, v5                                   
	v_and_b32_e32 v5, 15, v0                                   
	v_mul_lo_u32 v176, s36, v5                                 
	v_add_u32_e32 v176, s62, v176                              
	v_add_u32_e32 v176, v4, v176                               
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_0711                                  
	
label_02B2:
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[8:11], a[0:3], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v166 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[12:15], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1800, s65                                  
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[8:11], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v166 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[12:15], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1c00, s65                                  
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[16:19], a[0:3], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v166 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[20:23], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[16:19], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v166 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[20:23], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x10800, s64                                 
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	ds_read_b32 v154, v172 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[24:27], a[16:19], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v166 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[28:31], a[20:23], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x11880, s64                                 
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[24:27], a[24:27], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v166 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[28:31], a[28:31], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x12900, s64                                 
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[32:35], a[16:19], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v166 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[36:39], a[20:23], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x13980, s64                                 
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[32:35], a[24:27], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[68:71], v166 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[36:39], a[28:31], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v155, v172 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[40:43], a[32:35], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[72:75], v166 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[44:47], a[36:39], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[136:139], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[92:95], v[40:43], a[40:43], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[80:83], v166 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:95], v[44:47], a[44:47], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[48:51], a[32:35], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[76:79], v166 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[52:55], a[36:39], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[140:143], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[48:51], a[40:43], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[84:87], v166 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[52:55], a[44:47], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v156, v172 offset:1024                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[88:91], v[56:59], a[48:51], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v167                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[88:91], v[60:63], a[52:55], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[56:59], a[56:59], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v167 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[60:63], a[60:63], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[64:67], a[48:51], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v167 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[68:71], a[52:55], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[64:67], a[56:59], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v167 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[68:71], a[60:63], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v172 offset:2048                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[72:75], a[64:67], v157, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v167 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[76:79], a[68:71], v157, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v160, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[72:75], a[72:75], v157, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v167 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[76:79], a[76:79], v157, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:99], v[80:83], a[64:67], v157, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v167 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[96:99], v[84:87], a[68:71], v157, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[80:83], a[72:75], v157, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v167 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[84:87], a[76:79], v157, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v153, v172 offset:2304                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[8:11], a[0:3], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v167 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[12:15], a[4:7], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[8:11], a[8:11], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v167 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[12:15], a[12:15], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[16:19], a[0:3], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v167 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[20:23], a[4:7], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[16:19], a[8:11], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v167 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[20:23], a[12:15], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	ds_read_b32 v154, v172 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[24:27], a[16:19], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v167 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[28:31], a[20:23], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[24:27], a[24:27], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v167 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[28:31], a[28:31], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[32:35], a[16:19], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v167 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[36:39], a[20:23], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[32:35], a[24:27], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[68:71], v167 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:119], v[36:39], a[28:31], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v155, v172 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[40:43], a[32:35], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[72:75], v167 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[44:47], a[36:39], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[88:91], v173, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[40:43], a[40:43], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[80:83], v167 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[44:47], a[44:47], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[48:51], a[32:35], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[76:79], v167 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[52:55], a[36:39], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[92:95], v174, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[48:51], a[40:43], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[84:87], v167 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[52:55], a[44:47], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v156, v172 offset:3072                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[56:59], a[48:51], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v168                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[60:63], a[52:55], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[56:59], a[56:59], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v168 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[60:63], a[60:63], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[64:67], a[48:51], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v168 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[68:71], a[52:55], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[64:67], a[56:59], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v168 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[68:71], a[60:63], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v172 offset:4096                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[72:75], a[64:67], v158, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v168 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[76:79], a[68:71], v158, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v157, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[72:75], a[72:75], v158, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v168 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[76:79], a[76:79], v158, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[80:83], a[64:67], v158, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v168 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[112:115], v[84:87], a[68:71], v158, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[80:83], a[72:75], v158, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v168 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[84:87], a[76:79], v158, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v153, v172 offset:4352                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[8:11], a[0:3], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v168 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[12:15], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[8:11], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v168 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[12:15], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[16:19], a[0:3], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v168 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[20:23], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[16:19], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v168 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[20:23], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	ds_read_b32 v154, v172 offset:4608                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[24:27], a[16:19], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v168 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[28:31], a[20:23], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[24:27], a[24:27], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v168 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[28:31], a[28:31], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[32:35], a[16:19], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v168 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[36:39], a[20:23], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[32:35], a[24:27], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[68:71], v168 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[36:39], a[28:31], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v155, v172 offset:4864                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:123], v[40:43], a[32:35], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[72:75], v168 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[44:47], a[36:39], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[40:43], a[40:43], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[80:83], v168 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[44:47], a[44:47], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:131], v[48:51], a[32:35], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[76:79], v168 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[52:55], a[36:39], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[48:51], a[40:43], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[84:87], v168 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[52:55], a[44:47], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v156, v172 offset:5120                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[56:59], a[48:51], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v169                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[60:63], a[52:55], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[56:59], a[56:59], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v169 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[60:63], a[60:63], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[64:67], a[48:51], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v169 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[68:71], a[52:55], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[64:67], a[56:59], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v169 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[68:71], a[60:63], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v172 offset:6144                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[72:75], a[64:67], v159, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v169 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[76:79], a[68:71], v159, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v158, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[72:75], a[72:75], v159, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v169 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[76:79], a[76:79], v159, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[80:83], a[64:67], v159, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v169 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[84:87], a[68:71], v159, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[80:83], a[72:75], v159, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v169 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[84:87], a[76:79], v159, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v153, v172 offset:6400                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v160, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v169 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v160, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1000, s65                                  
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v160, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v169 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v160, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v160, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v169 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v160, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v160, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v169 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v160, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	ds_read_b32 v154, v172 offset:6656                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v160, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v169 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v160, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[24:27], a[24:27], v160, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v169 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[28:31], a[28:31], v160, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[32:35], a[16:19], v160, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v169 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[36:39], a[20:23], v160, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[32:35], a[24:27], v160, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[68:71], v169 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[36:39], a[28:31], v160, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v155, v172 offset:6912                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[40:43], a[32:35], v160, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[72:75], v169 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[44:47], a[36:39], v160, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[120:123], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[40:43], a[40:43], v160, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[80:83], v169 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[44:47], a[44:47], v160, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[48:51], a[32:35], v160, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[76:79], v169 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[52:55], a[36:39], v160, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[124:127], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[48:51], a[40:43], v160, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[84:87], v169 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[52:55], a[44:47], v160, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v156, v172 offset:7168                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[56:59], a[48:51], v160, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v166                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[60:63], a[52:55], v160, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[56:59], a[56:59], v160, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v166 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[60:63], a[60:63], v160, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[64:67], a[48:51], v160, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v166 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[68:71], a[52:55], v160, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[64:67], a[56:59], v160, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v166 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[68:71], a[60:63], v160, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v172                                     
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[72:75], a[64:67], v160, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v166 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[76:79], a[68:71], v160, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v159, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[72:75], a[72:75], v160, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v166 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[76:79], a[76:79], v160, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[80:83], a[64:67], v160, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v166 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[84:87], a[68:71], v160, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[80:83], a[72:75], v160, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v166 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[84:87], a[76:79], v160, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v153, v172 offset:256                          
	s_cbranch_scc0 label_0B70                                  
	s_branch label_02B2                                        
	
label_0711:
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[8:11], a[0:3], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1800, s65                                  
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[12:15], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v166 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[8:11], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1c00, s65                                  
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[12:15], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v166 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[16:19], a[0:3], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[20:23], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v166 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[16:19], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x10800, s64                                 
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[20:23], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v166 offset:9024                    
	ds_read_b32 v154, v172 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[24:27], a[16:19], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x11880, s64                                 
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[28:31], a[20:23], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v166 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[24:27], a[24:27], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x12900, s64                                 
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[28:31], a[28:31], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v166 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[32:35], a[16:19], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x13980, s64                                 
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[36:39], a[20:23], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[60:63], v166 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[32:35], a[24:27], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[36:39], a[28:31], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[68:71], v166 offset:13248                   
	ds_read_b32 v155, v172 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[40:43], a[32:35], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[136:139], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[44:47], a[36:39], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[72:75], v166 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[92:95], v[40:43], a[40:43], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:95], v[44:47], a[44:47], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[80:83], v166 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[48:51], a[32:35], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[140:143], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[52:55], a[36:39], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[76:79], v166 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[48:51], a[40:43], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[52:55], a[44:47], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[84:87], v166 offset:17472                   
	ds_read_b32 v156, v172 offset:1024                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[88:91], v[56:59], a[48:51], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[88:91], v[60:63], a[52:55], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v167                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[56:59], a[56:59], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[60:63], a[60:63], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v167 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[64:67], a[48:51], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[68:71], a[52:55], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v167 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[64:67], a[56:59], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[68:71], a[60:63], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v167 offset:576                     
	ds_read_b32 v152, v172 offset:2048                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[72:75], a[64:67], v157, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v160, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[76:79], a[68:71], v157, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v167 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[72:75], a[72:75], v157, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[76:79], a[76:79], v157, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v167 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:99], v[80:83], a[64:67], v157, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[96:99], v[84:87], a[68:71], v157, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v167 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[80:83], a[72:75], v157, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[84:87], a[76:79], v157, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v167 offset:4800                    
	ds_read_b32 v153, v172 offset:2304                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[8:11], a[0:3], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[12:15], a[4:7], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v167 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[8:11], a[8:11], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[12:15], a[12:15], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v167 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[16:19], a[0:3], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[20:23], a[4:7], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v167 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[16:19], a[8:11], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[20:23], a[12:15], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v167 offset:9024                    
	ds_read_b32 v154, v172 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[24:27], a[16:19], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[28:31], a[20:23], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v167 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[24:27], a[24:27], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[28:31], a[28:31], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v167 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[32:35], a[16:19], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[36:39], a[20:23], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[60:63], v167 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[32:35], a[24:27], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:119], v[36:39], a[28:31], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[68:71], v167 offset:13248                   
	ds_read_b32 v155, v172 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[40:43], a[32:35], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[88:91], v173, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[44:47], a[36:39], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[72:75], v167 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[40:43], a[40:43], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[44:47], a[44:47], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[80:83], v167 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[48:51], a[32:35], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[92:95], v174, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[52:55], a[36:39], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[76:79], v167 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[48:51], a[40:43], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[52:55], a[44:47], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[84:87], v167 offset:17472                   
	ds_read_b32 v156, v172 offset:3072                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[56:59], a[48:51], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[60:63], a[52:55], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v168                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[56:59], a[56:59], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[60:63], a[60:63], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v168 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[64:67], a[48:51], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[68:71], a[52:55], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v168 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[64:67], a[56:59], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[68:71], a[60:63], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v168 offset:576                     
	ds_read_b32 v152, v172 offset:4096                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[72:75], a[64:67], v158, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v157, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[76:79], a[68:71], v158, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v168 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[72:75], a[72:75], v158, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[76:79], a[76:79], v158, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v168 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[80:83], a[64:67], v158, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[112:115], v[84:87], a[68:71], v158, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v168 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[80:83], a[72:75], v158, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[84:87], a[76:79], v158, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v168 offset:4800                    
	ds_read_b32 v153, v172 offset:4352                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[8:11], a[0:3], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[12:15], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v168 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[8:11], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[12:15], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v168 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[16:19], a[0:3], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[20:23], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v168 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[16:19], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[20:23], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v168 offset:9024                    
	ds_read_b32 v154, v172 offset:4608                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[24:27], a[16:19], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[28:31], a[20:23], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v168 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[24:27], a[24:27], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[28:31], a[28:31], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v168 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[32:35], a[16:19], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[36:39], a[20:23], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[60:63], v168 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[32:35], a[24:27], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[36:39], a[28:31], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[68:71], v168 offset:13248                   
	ds_read_b32 v155, v172 offset:4864                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:123], v[40:43], a[32:35], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[44:47], a[36:39], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[72:75], v168 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[40:43], a[40:43], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[44:47], a[44:47], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[80:83], v168 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:131], v[48:51], a[32:35], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[52:55], a[36:39], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[76:79], v168 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[48:51], a[40:43], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[52:55], a[44:47], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[84:87], v168 offset:17472                   
	ds_read_b32 v156, v172 offset:5120                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[56:59], a[48:51], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[60:63], a[52:55], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v169                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[56:59], a[56:59], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[60:63], a[60:63], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v169 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[64:67], a[48:51], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[68:71], a[52:55], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v169 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[64:67], a[56:59], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[68:71], a[60:63], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v169 offset:576                     
	ds_read_b32 v152, v172 offset:6144                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[72:75], a[64:67], v159, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v158, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[76:79], a[68:71], v159, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v169 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[72:75], a[72:75], v159, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[76:79], a[76:79], v159, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v169 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[80:83], a[64:67], v159, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[84:87], a[68:71], v159, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v169 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[80:83], a[72:75], v159, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[84:87], a[76:79], v159, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v169 offset:4800                    
	ds_read_b32 v153, v172 offset:6400                         
	s_cbranch_scc0 label_0B70                                  
	s_waitcnt vmcnt(24) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v160, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1000, s65                                  
	buffer_load_dword v170, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v160, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v169 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v160, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v171, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v160, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v169 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v160, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v160, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v169 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v160, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v160, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v169 offset:9024                    
	ds_read_b32 v154, v172 offset:6656                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v160, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v160, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v169 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[24:27], a[24:27], v160, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[28:31], a[28:31], v160, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v169 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[32:35], a[16:19], v160, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[36:39], a[20:23], v160, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[60:63], v169 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[32:35], a[24:27], v160, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[36:39], a[28:31], v160, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[68:71], v169 offset:13248                   
	ds_read_b32 v155, v172 offset:6912                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[40:43], a[32:35], v160, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[120:123], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[44:47], a[36:39], v160, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[72:75], v169 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[40:43], a[40:43], v160, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[44:47], a[44:47], v160, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[80:83], v169 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[48:51], a[32:35], v160, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[124:127], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[52:55], a[36:39], v160, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[76:79], v169 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[48:51], a[40:43], v160, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[52:55], a[44:47], v160, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[84:87], v169 offset:17472                   
	ds_read_b32 v156, v172 offset:7168                         
	s_waitcnt vmcnt(29) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[56:59], a[48:51], v160, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[60:63], a[52:55], v160, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v166                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[56:59], a[56:59], v160, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[60:63], a[60:63], v160, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v166 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[64:67], a[48:51], v160, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[68:71], a[52:55], v160, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v166 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[64:67], a[56:59], v160, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[68:71], a[60:63], v160, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v166 offset:576                     
	ds_read_b32 v152, v172                                     
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[72:75], a[64:67], v160, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v159, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[76:79], a[68:71], v160, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v166 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[72:75], a[72:75], v160, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[76:79], a[76:79], v160, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v166 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[80:83], a[64:67], v160, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[84:87], a[68:71], v160, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v166 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[80:83], a[72:75], v160, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[84:87], a[76:79], v160, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v166 offset:4800                    
	ds_read_b32 v153, v172 offset:256                          
	s_cbranch_scc0 label_0B70                                  
	s_branch label_0711                                        
	
label_0B70:
	s_waitcnt lgkmcnt(0)                                       
	s_mul_i32 s62, s47, 0x80                                   
	s_mul_i32 s63, s46, 32                                     
	s_add_u32 s60, s62, s63                                    
	s_add_u32 s62, s60, 32                                     
	s_cmp_lt_i32 s44, s62                                      
	s_cbranch_scc1 label_0CBB                                  
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v180, 0, v176                                
	v_accvgpr_read_b32 v8, a0                                  
	v_accvgpr_read_b32 v9, a1                                  
	v_accvgpr_read_b32 v10, a2                                 
	v_accvgpr_read_b32 v11, a3                                 
	v_accvgpr_read_b32 v12, a8                                 
	v_accvgpr_read_b32 v13, a9                                 
	v_accvgpr_read_b32 v14, a10                                
	v_accvgpr_read_b32 v15, a11                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a4                                  
	v_accvgpr_read_b32 v9, a5                                  
	v_accvgpr_read_b32 v10, a6                                 
	v_accvgpr_read_b32 v11, a7                                 
	v_accvgpr_read_b32 v12, a12                                
	v_accvgpr_read_b32 v13, a13                                
	v_accvgpr_read_b32 v14, a14                                
	v_accvgpr_read_b32 v15, a15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a16                                 
	v_accvgpr_read_b32 v9, a17                                 
	v_accvgpr_read_b32 v10, a18                                
	v_accvgpr_read_b32 v11, a19                                
	v_accvgpr_read_b32 v12, a24                                
	v_accvgpr_read_b32 v13, a25                                
	v_accvgpr_read_b32 v14, a26                                
	v_accvgpr_read_b32 v15, a27                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a20                                 
	v_accvgpr_read_b32 v9, a21                                 
	v_accvgpr_read_b32 v10, a22                                
	v_accvgpr_read_b32 v11, a23                                
	v_accvgpr_read_b32 v12, a28                                
	v_accvgpr_read_b32 v13, a29                                
	v_accvgpr_read_b32 v14, a30                                
	v_accvgpr_read_b32 v15, a31                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a32                                 
	v_accvgpr_read_b32 v9, a33                                 
	v_accvgpr_read_b32 v10, a34                                
	v_accvgpr_read_b32 v11, a35                                
	v_accvgpr_read_b32 v12, a40                                
	v_accvgpr_read_b32 v13, a41                                
	v_accvgpr_read_b32 v14, a42                                
	v_accvgpr_read_b32 v15, a43                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a36                                 
	v_accvgpr_read_b32 v9, a37                                 
	v_accvgpr_read_b32 v10, a38                                
	v_accvgpr_read_b32 v11, a39                                
	v_accvgpr_read_b32 v12, a44                                
	v_accvgpr_read_b32 v13, a45                                
	v_accvgpr_read_b32 v14, a46                                
	v_accvgpr_read_b32 v15, a47                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a48                                 
	v_accvgpr_read_b32 v9, a49                                 
	v_accvgpr_read_b32 v10, a50                                
	v_accvgpr_read_b32 v11, a51                                
	v_accvgpr_read_b32 v12, a56                                
	v_accvgpr_read_b32 v13, a57                                
	v_accvgpr_read_b32 v14, a58                                
	v_accvgpr_read_b32 v15, a59                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a52                                 
	v_accvgpr_read_b32 v9, a53                                 
	v_accvgpr_read_b32 v10, a54                                
	v_accvgpr_read_b32 v11, a55                                
	v_accvgpr_read_b32 v12, a60                                
	v_accvgpr_read_b32 v13, a61                                
	v_accvgpr_read_b32 v14, a62                                
	v_accvgpr_read_b32 v15, a63                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a64                                 
	v_accvgpr_read_b32 v9, a65                                 
	v_accvgpr_read_b32 v10, a66                                
	v_accvgpr_read_b32 v11, a67                                
	v_accvgpr_read_b32 v12, a72                                
	v_accvgpr_read_b32 v13, a73                                
	v_accvgpr_read_b32 v14, a74                                
	v_accvgpr_read_b32 v15, a75                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a68                                 
	v_accvgpr_read_b32 v9, a69                                 
	v_accvgpr_read_b32 v10, a70                                
	v_accvgpr_read_b32 v11, a71                                
	v_accvgpr_read_b32 v12, a76                                
	v_accvgpr_read_b32 v13, a77                                
	v_accvgpr_read_b32 v14, a78                                
	v_accvgpr_read_b32 v15, a79                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	s_branch label_0E00                                        
	
label_0CBB:
	s_mul_i32 s62, s36, 16                                     
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_0E00                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v180, 0, v176                                
	v_accvgpr_read_b32 v8, a0                                  
	v_accvgpr_read_b32 v9, a1                                  
	v_accvgpr_read_b32 v10, a2                                 
	v_accvgpr_read_b32 v11, a3                                 
	v_accvgpr_read_b32 v12, a8                                 
	v_accvgpr_read_b32 v13, a9                                 
	v_accvgpr_read_b32 v14, a10                                
	v_accvgpr_read_b32 v15, a11                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a4                                  
	v_accvgpr_read_b32 v9, a5                                  
	v_accvgpr_read_b32 v10, a6                                 
	v_accvgpr_read_b32 v11, a7                                 
	v_accvgpr_read_b32 v12, a12                                
	v_accvgpr_read_b32 v13, a13                                
	v_accvgpr_read_b32 v14, a14                                
	v_accvgpr_read_b32 v15, a15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a16                                 
	v_accvgpr_read_b32 v9, a17                                 
	v_accvgpr_read_b32 v10, a18                                
	v_accvgpr_read_b32 v11, a19                                
	v_accvgpr_read_b32 v12, a24                                
	v_accvgpr_read_b32 v13, a25                                
	v_accvgpr_read_b32 v14, a26                                
	v_accvgpr_read_b32 v15, a27                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a20                                 
	v_accvgpr_read_b32 v9, a21                                 
	v_accvgpr_read_b32 v10, a22                                
	v_accvgpr_read_b32 v11, a23                                
	v_accvgpr_read_b32 v12, a28                                
	v_accvgpr_read_b32 v13, a29                                
	v_accvgpr_read_b32 v14, a30                                
	v_accvgpr_read_b32 v15, a31                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a32                                 
	v_accvgpr_read_b32 v9, a33                                 
	v_accvgpr_read_b32 v10, a34                                
	v_accvgpr_read_b32 v11, a35                                
	v_accvgpr_read_b32 v12, a40                                
	v_accvgpr_read_b32 v13, a41                                
	v_accvgpr_read_b32 v14, a42                                
	v_accvgpr_read_b32 v15, a43                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a36                                 
	v_accvgpr_read_b32 v9, a37                                 
	v_accvgpr_read_b32 v10, a38                                
	v_accvgpr_read_b32 v11, a39                                
	v_accvgpr_read_b32 v12, a44                                
	v_accvgpr_read_b32 v13, a45                                
	v_accvgpr_read_b32 v14, a46                                
	v_accvgpr_read_b32 v15, a47                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a48                                 
	v_accvgpr_read_b32 v9, a49                                 
	v_accvgpr_read_b32 v10, a50                                
	v_accvgpr_read_b32 v11, a51                                
	v_accvgpr_read_b32 v12, a56                                
	v_accvgpr_read_b32 v13, a57                                
	v_accvgpr_read_b32 v14, a58                                
	v_accvgpr_read_b32 v15, a59                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a52                                 
	v_accvgpr_read_b32 v9, a53                                 
	v_accvgpr_read_b32 v10, a54                                
	v_accvgpr_read_b32 v11, a55                                
	v_accvgpr_read_b32 v12, a60                                
	v_accvgpr_read_b32 v13, a61                                
	v_accvgpr_read_b32 v14, a62                                
	v_accvgpr_read_b32 v15, a63                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a64                                 
	v_accvgpr_read_b32 v9, a65                                 
	v_accvgpr_read_b32 v10, a66                                
	v_accvgpr_read_b32 v11, a67                                
	v_accvgpr_read_b32 v12, a72                                
	v_accvgpr_read_b32 v13, a73                                
	v_accvgpr_read_b32 v14, a74                                
	v_accvgpr_read_b32 v15, a75                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	v_accvgpr_read_b32 v8, a68                                 
	v_accvgpr_read_b32 v9, a69                                 
	v_accvgpr_read_b32 v10, a70                                
	v_accvgpr_read_b32 v11, a71                                
	v_accvgpr_read_b32 v12, a76                                
	v_accvgpr_read_b32 v13, a77                                
	v_accvgpr_read_b32 v14, a78                                
	v_accvgpr_read_b32 v15, a79                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v180, s[4:7], 0 offen       
	v_add_u32_e32 v180, s62, v180                              
	
label_0E00:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128
  .amdhsa_next_free_vgpr 512
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 163840
  .amdhsa_accum_offset 256
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
.end_amdhsa_kernel

// ===== AMDGPU Metadata =====
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_write
        .address_space:  global
        .name:           D
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         8
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           C
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         24
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           A
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         40
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           B
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         56
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           alpha
        .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           beta
        .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideD0
        .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideD1
        .offset:         112
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         116
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         120
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         124
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideC0
        .offset:         128
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         132
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         136
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         140
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideC1
        .offset:         144
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         148
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         152
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         156
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideA0
        .offset:         160
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         164
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         168
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         172
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideA1
        .offset:         176
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         180
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         184
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         188
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideB0
        .offset:         192
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         196
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         200
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         204
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideB1
        .offset:         208
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         212
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         216
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         220
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Mdim
        .offset:         224
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         228
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         232
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         236
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Ndim
        .offset:         240
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         244
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         248
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         252
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Kdim
        .offset:         256
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         260
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         264
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         268
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           ScaleA
        .offset:         272
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         280
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           ScaleB
        .offset:         288
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         296
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideScaleA0
        .offset:         304
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         308
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         312
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         316
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideScaleA1
        .offset:         320
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         324
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         328
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         332
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideScaleB0
        .offset:         336
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         340
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         344
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         348
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           strideScaleB1
        .offset:         352
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         356
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         360
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         364
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           log2_k_split
        .offset:         368
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         372
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         376
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         380
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 4
    .kernarg_segment_size: 384
    .max_flat_workgroup_size: 256
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

