// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256:
	
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
	s_add_u32 s51, s44, 0xff                                   
	s_lshr_b32 s50, s51, 8                                     
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
	s_add_u32 s64, 0x1000, s64                                 
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
	v_add_u32_e32 v166, 0x1000, v166                           
	v_add_u32_e32 v167, 0x5280, v166                           
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
	v_lshlrev_b32_e32 v168, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v168, s63, v168                              
	s_mul_i32 s63, 0x80, s39                                   
	v_add_u32_e32 v169, s63, v168                              
	s_mul_i32 s65, s46, 0x100                                  
	s_add_i32 s65, s65, 0                                      
	v_lshlrev_b32_e32 v170, 2, v0                              
	v_add_u32_e32 v170, 0, v170                                
	s_lshr_b32 s38, s38, 1                                     
	s_mul_i32 s62, s47, 0x100                                  
	s_mul_hi_u32 s63, s38, s62                                 
	s_add_u32 s17, s17, s63                                    
	s_mul_i32 s63, s38, s62                                    
	s_add_u32 s16, s16, s63                                    
	s_addc_u32 s17, s17, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x100                                    
	s_cselect_b32 s62, s63, 0x100                              
	s_mul_i32 s18, s38, s62                                    
	s_mov_b32 s19, 0x20000                                     
	v_lshlrev_b32_e32 v171, 4, v0                              
	s_mul_i32 s63, s46, 64                                     
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v171, s62, v171                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v172, s62, v171                              
	v_add_u32_e32 v173, s62, v172                              
	v_add_u32_e32 v174, s62, v173                              
	s_mul_i32 s62, s47, 0x100                                  
	s_mul_hi_u32 s63, s40, s62                                 
	s_add_u32 s25, s25, s63                                    
	s_mul_i32 s63, s40, s62                                    
	s_add_u32 s24, s24, s63                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x100                                    
	s_cselect_b32 s62, s63, 0x100                              
	s_mul_i32 s26, s40, s62                                    
	s_mov_b32 s27, 0x20000                                     
	v_lshlrev_b32_e32 v175, 2, v0                              
	s_mul_i32 s63, s46, 64                                     
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v175, s63, v175                              
	s_mul_i32 s62, 32, s40                                     
	v_add_u32_e32 v176, s62, v175                              
	s_mov_b32 s66, 0x80                                        
	s_mov_b32 s67, 0x800                                       
	s_mov_b32 s68, 0x100                                       
	s_mov_b32 s69, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
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
	buffer_load_dwordx4 v[88:91], v171, s[16:19], 0 offen      
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	buffer_load_dwordx4 v[92:95], v172, s[16:19], 0 offen      
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	buffer_load_dwordx4 v[96:99], v171, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	buffer_load_dwordx4 v[100:103], v172, s[16:19], 0 offen offset:1024
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
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	buffer_load_dword v158, v176, s[24:27], 0 offen            
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
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
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a102, 0                                
	v_accvgpr_write_b32 a103, 0                                
	v_accvgpr_write_b32 a104, 0                                
	v_accvgpr_write_b32 a105, 0                                
	v_accvgpr_write_b32 a106, 0                                
	v_accvgpr_write_b32 a107, 0                                
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a108, 0                                
	v_accvgpr_write_b32 a109, 0                                
	v_accvgpr_write_b32 a110, 0                                
	v_accvgpr_write_b32 a111, 0                                
	v_accvgpr_write_b32 a112, 0                                
	v_accvgpr_write_b32 a113, 0                                
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a114, 0                                
	v_accvgpr_write_b32 a115, 0                                
	v_accvgpr_write_b32 a116, 0                                
	v_accvgpr_write_b32 a117, 0                                
	v_accvgpr_write_b32 a118, 0                                
	v_accvgpr_write_b32 a119, 0                                
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a120, 0                                
	v_accvgpr_write_b32 a121, 0                                
	v_accvgpr_write_b32 a122, 0                                
	v_accvgpr_write_b32 a123, 0                                
	v_accvgpr_write_b32 a124, 0                                
	v_accvgpr_write_b32 a125, 0                                
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a126, 0                                
	v_accvgpr_write_b32 a127, 0                                
	v_accvgpr_write_b32 a128, 0                                
	v_accvgpr_write_b32 a129, 0                                
	v_accvgpr_write_b32 a130, 0                                
	v_accvgpr_write_b32 a131, 0                                
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a132, 0                                
	v_accvgpr_write_b32 a133, 0                                
	v_accvgpr_write_b32 a134, 0                                
	v_accvgpr_write_b32 a135, 0                                
	v_accvgpr_write_b32 a136, 0                                
	v_accvgpr_write_b32 a137, 0                                
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a138, 0                                
	v_accvgpr_write_b32 a139, 0                                
	v_accvgpr_write_b32 a140, 0                                
	v_accvgpr_write_b32 a141, 0                                
	v_accvgpr_write_b32 a142, 0                                
	v_accvgpr_write_b32 a143, 0                                
	buffer_load_dwordx4 v[120:123], v171, s[16:19], 0 offen    
	v_accvgpr_write_b32 a144, 0                                
	v_accvgpr_write_b32 a145, 0                                
	v_accvgpr_write_b32 a146, 0                                
	v_accvgpr_write_b32 a147, 0                                
	v_accvgpr_write_b32 a148, 0                                
	v_accvgpr_write_b32 a149, 0                                
	buffer_load_dwordx4 v[124:127], v172, s[16:19], 0 offen    
	v_accvgpr_write_b32 a150, 0                                
	v_accvgpr_write_b32 a151, 0                                
	v_accvgpr_write_b32 a152, 0                                
	v_accvgpr_write_b32 a153, 0                                
	v_accvgpr_write_b32 a154, 0                                
	v_accvgpr_write_b32 a155, 0                                
	buffer_load_dwordx4 v[128:131], v171, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a156, 0                                
	v_accvgpr_write_b32 a157, 0                                
	v_accvgpr_write_b32 a158, 0                                
	v_accvgpr_write_b32 a159, 0                                
	buffer_load_dwordx4 v[132:135], v172, s[16:19], 0 offen offset:1024
	buffer_load_dword v159, v175, s[24:27], 0 offen            
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
	s_waitcnt vmcnt(25)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v166                                 
	ds_read_b128 v[16:19], v166 offset:64                      
	ds_read_b128 v[12:15], v166 offset:512                     
	ds_read_b128 v[20:23], v166 offset:576                     
	ds_read_b32 v152, v170                                     
	ds_read_b128 v[24:27], v166 offset:4224                    
	ds_read_b128 v[32:35], v166 offset:4288                    
	ds_read_b128 v[28:31], v166 offset:4736                    
	ds_read_b128 v[36:39], v166 offset:4800                    
	ds_read_b32 v153, v170 offset:256                          
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
	s_mul_i32 s63, s47, 0x100                                  
	s_lshl_b32 s63, s63, 1                                     
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_sub_i32 s62, s43, s62                                    
	s_cmp_lt_u32 s62, 0xa0                                     
	s_cselect_b32 s62, s62, 0xa0                               
	s_mul_i32 s62, s36, s62                                    
	s_sub_i32 s6, s62, s63                                     
	s_mov_b32 s7, 0x20000                                      
	s_mul_i32 s62, s46, 64                                     
	s_lshl_b32 s62, s62, 1                                     
	v_lshrrev_b32_e32 v4, 5, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_lshrrev_b32_e32 v5, 4, v0                                
	v_and_b32_e32 v5, 1, v5                                    
	v_mul_i32_i24_e32 v5, 32, v5                               
	v_add_u32_e32 v4, v4, v5                                   
	v_and_b32_e32 v5, 15, v0                                   
	v_mul_lo_u32 v177, s36, v5                                 
	v_add_u32_e32 v177, s62, v177                              
	v_add_u32_e32 v177, v4, v177                               
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_069E                                  
	
label_0315:
	s_waitcnt vmcnt(17) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[8:11], a[0:3], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	ds_read_b128 v[40:43], v166 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[12:15], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	buffer_load_dwordx4 v[136:139], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[8:11], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b128 v[48:51], v166 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[12:15], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[16:19], a[0:3], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v166 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[20:23], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[16:19], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v166 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[20:23], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v154, v170 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[24:27], a[32:35], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v166 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[28:31], a[36:39], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[92:95], v[24:27], a[40:43], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v166 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:95], v[28:31], a[44:47], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[32:35], a[32:35], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v166 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[36:39], a[36:39], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[32:35], a[40:43], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v166 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[36:39], a[44:47], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v155, v170 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[40:43], a[64:67], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v166 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[44:47], a[68:71], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v160, v176, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[40:43], a[72:75], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[80:83], v166 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[44:47], a[76:79], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:99], v[48:51], a[64:67], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[76:79], v166 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[96:99], v[52:55], a[68:71], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[48:51], a[72:75], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v166 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[52:55], a[76:79], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v156, v170 offset:1024                         
	s_waitcnt lgkmcnt(5)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[56:59], a[96:99], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[60:63], a[100:103], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[56:59], a[104:107], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[60:63], a[108:111], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[64:67], a[96:99], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[68:71], a[100:103], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[100:103], v[64:67], a[104:107], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[100:103], v[68:71], a[108:111], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[72:75], a[128:131], v157, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[76:79], a[132:135], v157, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[72:75], a[136:139], v157, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[76:79], a[140:143], v157, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[80:83], a[128:131], v157, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[84:87], a[132:135], v157, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[100:103], v[80:83], a[136:139], v157, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[100:103], v[84:87], a[140:143], v157, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(20)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[8:11], a[16:19], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[12:15], a[20:23], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[8:11], a[24:27], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[12:15], a[28:31], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[16:19], a[16:19], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[20:23], a[20:23], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[16:19], a[24:27], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:119], v[20:23], a[28:31], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[24:27], a[48:51], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[28:31], a[52:55], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[24:27], a[56:59], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[28:31], a[60:63], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[32:35], a[48:51], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[36:39], a[52:55], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[32:35], a[56:59], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x300, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[36:39], a[60:63], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:107], v[40:43], a[80:83], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:107], v[44:47], a[84:87], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	buffer_load_dwordx4 v[88:91], v171, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[40:43], a[88:91], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[44:47], a[92:95], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[48:51], a[80:83], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[52:55], a[84:87], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v172, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:119], v[48:51], a[88:91], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[52:55], a[92:95], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(22)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[56:59], a[112:115], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v167                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[60:63], a[116:119], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v171, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[108:111], v[56:59], a[120:123], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v167 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[60:63], a[124:127], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[64:67], a[112:115], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v167 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[68:71], a[116:119], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v172, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:119], v[64:67], a[120:123], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v167 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:119], v[68:71], a[124:127], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v170 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[104:107], v[72:75], a[144:147], v158, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v167 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[104:107], v[76:79], a[148:151], v158, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v157, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[72:75], a[152:155], v158, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v167 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[76:79], a[156:159], v158, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[112:115], v[80:83], a[144:147], v158, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v167 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[112:115], v[84:87], a[148:151], v158, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[116:119], v[80:83], a[152:155], v158, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v167 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:119], v[84:87], a[156:159], v158, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v153, v170 offset:2304                         
	s_cbranch_scc0 label_0A27                                  
	s_waitcnt vmcnt(17) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[8:11], a[0:3], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	ds_read_b128 v[40:43], v167 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[12:15], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[8:11], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b128 v[48:51], v167 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[12:15], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[16:19], a[0:3], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v167 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[20:23], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[16:19], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v167 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[20:23], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v154, v170 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:123], v[24:27], a[32:35], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v167 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[28:31], a[36:39], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[24:27], a[40:43], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v167 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[28:31], a[44:47], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:131], v[32:35], a[32:35], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v167 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[36:39], a[36:39], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[32:35], a[40:43], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v167 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[36:39], a[44:47], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v155, v170 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[40:43], a[64:67], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v167 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[44:47], a[68:71], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v158, v176, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[40:43], a[72:75], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[80:83], v167 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[44:47], a[76:79], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[48:51], a[64:67], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[76:79], v167 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[52:55], a[68:71], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[48:51], a[72:75], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v167 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[52:55], a[76:79], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v156, v170 offset:3072                         
	s_waitcnt lgkmcnt(5)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[56:59], a[96:99], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[60:63], a[100:103], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:127], v[56:59], a[104:107], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[60:63], a[108:111], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[64:67], a[96:99], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[68:71], a[100:103], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[64:67], a[104:107], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[132:135], v[68:71], a[108:111], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[120:123], v[72:75], a[128:131], v159, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[120:123], v[76:79], a[132:135], v159, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[72:75], a[136:139], v159, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[76:79], a[140:143], v159, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[128:131], v[80:83], a[128:131], v159, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[128:131], v[84:87], a[132:135], v159, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[132:135], v[80:83], a[136:139], v159, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[132:135], v[84:87], a[140:143], v159, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(20)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[8:11], a[16:19], v160, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[12:15], a[20:23], v160, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[8:11], a[24:27], v160, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[12:15], a[28:31], v160, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[16:19], a[16:19], v160, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[20:23], a[20:23], v160, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[16:19], a[24:27], v160, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[20:23], a[28:31], v160, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[24:27], a[48:51], v160, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[28:31], a[52:55], v160, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[24:27], a[56:59], v160, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[28:31], a[60:63], v160, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[32:35], a[48:51], v160, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[36:39], a[52:55], v160, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[32:35], a[56:59], v160, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x300, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[36:39], a[60:63], v160, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[40:43], a[80:83], v160, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[44:47], a[84:87], v160, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	buffer_load_dwordx4 v[120:123], v171, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[40:43], a[88:91], v160, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[44:47], a[92:95], v160, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[48:51], a[80:83], v160, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[52:55], a[84:87], v160, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v172, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[48:51], a[88:91], v160, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[52:55], a[92:95], v160, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(22)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[56:59], a[112:115], v160, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v166                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[60:63], a[116:119], v160, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v171, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[56:59], a[120:123], v160, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v166 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[60:63], a[124:127], v160, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[64:67], a[112:115], v160, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v166 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[68:71], a[116:119], v160, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v172, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[64:67], a[120:123], v160, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v166 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[68:71], a[124:127], v160, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v152, v170                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[72:75], a[144:147], v160, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v166 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[76:79], a[148:151], v160, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v159, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[72:75], a[152:155], v160, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v166 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[76:79], a[156:159], v160, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[80:83], a[144:147], v160, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v166 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[84:87], a[148:151], v160, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[80:83], a[152:155], v160, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v166 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[84:87], a[156:159], v160, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v153, v170 offset:256                          
	s_cbranch_scc0 label_0A27                                  
	s_branch label_0315                                        
	
label_069E:
	s_waitcnt vmcnt(17) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[8:11], a[0:3], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	buffer_load_dwordx4 v[136:139], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[12:15], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	ds_read_b128 v[40:43], v166 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[8:11], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[12:15], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v166 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[16:19], a[0:3], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[20:23], a[4:7], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v166 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[16:19], a[8:11], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[20:23], a[12:15], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v166 offset:9024                    
	ds_read_b32 v154, v170 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[24:27], a[32:35], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[28:31], a[36:39], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v166 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[92:95], v[24:27], a[40:43], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[92:95], v[28:31], a[44:47], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v166 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[32:35], a[32:35], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[36:39], a[36:39], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v166 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[32:35], a[40:43], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[36:39], a[44:47], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v166 offset:13248                   
	ds_read_b32 v155, v170 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[40:43], a[64:67], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v160, v176, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[44:47], a[68:71], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[72:75], v166 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[40:43], a[72:75], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[44:47], a[76:79], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[80:83], v166 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[96:99], v[48:51], a[64:67], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[96:99], v[52:55], a[68:71], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v166 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[48:51], a[72:75], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[52:55], a[76:79], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v166 offset:17472                   
	ds_read_b32 v156, v170 offset:1024                         
	s_waitcnt lgkmcnt(5)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[56:59], a[96:99], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[60:63], a[100:103], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[56:59], a[104:107], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[60:63], a[108:111], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[64:67], a[96:99], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[68:71], a[100:103], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[100:103], v[64:67], a[104:107], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[100:103], v[68:71], a[108:111], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[72:75], a[128:131], v157, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[76:79], a[132:135], v157, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[72:75], a[136:139], v157, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[76:79], a[140:143], v157, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[80:83], a[128:131], v157, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[84:87], a[132:135], v157, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[100:103], v[80:83], a[136:139], v157, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[100:103], v[84:87], a[140:143], v157, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(20)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[8:11], a[16:19], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[12:15], a[20:23], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[8:11], a[24:27], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[12:15], a[28:31], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[16:19], a[16:19], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[20:23], a[20:23], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[16:19], a[24:27], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[116:119], v[20:23], a[28:31], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[24:27], a[48:51], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[28:31], a[52:55], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[24:27], a[56:59], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[28:31], a[60:63], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[32:35], a[48:51], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[36:39], a[52:55], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x300, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[32:35], a[56:59], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[36:39], a[60:63], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[104:107], v[40:43], a[80:83], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	buffer_load_dwordx4 v[88:91], v171, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:107], v[44:47], a[84:87], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[40:43], a[88:91], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[44:47], a[92:95], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[48:51], a[80:83], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v172, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[52:55], a[84:87], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:119], v[48:51], a[88:91], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[52:55], a[92:95], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(22)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[56:59], a[112:115], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v171, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[60:63], a[116:119], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v167                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[108:111], v[56:59], a[120:123], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[60:63], a[124:127], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v167 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[64:67], a[112:115], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v172, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[68:71], a[116:119], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v167 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:119], v[64:67], a[120:123], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:119], v[68:71], a[124:127], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v167 offset:576                     
	ds_read_b32 v152, v170 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[104:107], v[72:75], a[144:147], v158, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v157, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[104:107], v[76:79], a[148:151], v158, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v167 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[72:75], a[152:155], v158, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[76:79], a[156:159], v158, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v167 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[112:115], v[80:83], a[144:147], v158, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[112:115], v[84:87], a[148:151], v158, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v167 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[116:119], v[80:83], a[152:155], v158, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[116:119], v[84:87], a[156:159], v158, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v167 offset:4800                    
	ds_read_b32 v153, v170 offset:2304                         
	s_cbranch_scc0 label_0A27                                  
	s_waitcnt vmcnt(17) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[8:11], a[0:3], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	buffer_load_dwordx4 v[104:107], v173, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[12:15], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	ds_read_b128 v[40:43], v167 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[8:11], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[12:15], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v167 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[16:19], a[0:3], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v174, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[20:23], a[4:7], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v167 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[16:19], a[8:11], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[20:23], a[12:15], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v167 offset:9024                    
	ds_read_b32 v154, v170 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:123], v[24:27], a[32:35], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v173, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[28:31], a[36:39], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v167 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[24:27], a[40:43], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[28:31], a[44:47], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v167 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:131], v[32:35], a[32:35], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v174, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[36:39], a[36:39], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v167 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[32:35], a[40:43], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[36:39], a[44:47], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v167 offset:13248                   
	ds_read_b32 v155, v170 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[40:43], a[64:67], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v158, v176, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[44:47], a[68:71], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[72:75], v167 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[40:43], a[72:75], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[44:47], a[76:79], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[80:83], v167 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[48:51], a[64:67], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[52:55], a[68:71], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v167 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[48:51], a[72:75], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[52:55], a[76:79], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v167 offset:17472                   
	ds_read_b32 v156, v170 offset:3072                         
	s_waitcnt lgkmcnt(5)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[56:59], a[96:99], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v168, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[60:63], a[100:103], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:127], v[56:59], a[104:107], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[60:63], a[108:111], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[64:67], a[96:99], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v169, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[68:71], a[100:103], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[64:67], a[104:107], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[132:135], v[68:71], a[108:111], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[120:123], v[72:75], a[128:131], v159, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v161, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[120:123], v[76:79], a[132:135], v159, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[72:75], a[136:139], v159, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[76:79], a[140:143], v159, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[128:131], v[80:83], a[128:131], v159, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[128:131], v[84:87], a[132:135], v159, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[132:135], v[80:83], a[136:139], v159, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[132:135], v[84:87], a[140:143], v159, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(20)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[8:11], a[16:19], v160, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v162, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[12:15], a[20:23], v160, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[8:11], a[24:27], v160, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[12:15], a[28:31], v160, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[16:19], a[16:19], v160, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v163, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[20:23], a[20:23], v160, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[16:19], a[24:27], v160, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[20:23], a[28:31], v160, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[24:27], a[48:51], v160, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v164, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[28:31], a[52:55], v160, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[24:27], a[56:59], v160, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[28:31], a[60:63], v160, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[32:35], a[48:51], v160, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v165, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[36:39], a[52:55], v160, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x300, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[32:35], a[56:59], v160, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	s_cselect_b32 s68, s68, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[36:39], a[60:63], v160, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[40:43], a[80:83], v160, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	buffer_load_dwordx4 v[120:123], v171, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[44:47], a[84:87], v160, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s68                                    
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[40:43], a[88:91], v160, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[44:47], a[92:95], v160, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[48:51], a[80:83], v160, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v172, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[52:55], a[84:87], v160, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[48:51], a[88:91], v160, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[52:55], a[92:95], v160, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(22)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[56:59], a[112:115], v160, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v171, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[60:63], a[116:119], v160, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v166                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[56:59], a[120:123], v160, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[60:63], a[124:127], v160, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v166 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[64:67], a[112:115], v160, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v172, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[68:71], a[116:119], v160, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v166 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[64:67], a[120:123], v160, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[68:71], a[124:127], v160, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v166 offset:576                     
	ds_read_b32 v152, v170                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[72:75], a[144:147], v160, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v159, v175, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[76:79], a[148:151], v160, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v166 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[72:75], a[152:155], v160, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[76:79], a[156:159], v160, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v166 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[80:83], a[144:147], v160, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[84:87], a[148:151], v160, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v166 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[80:83], a[152:155], v160, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[84:87], a[156:159], v160, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v166 offset:4800                    
	ds_read_b32 v153, v170 offset:256                          
	s_cbranch_scc0 label_0A27                                  
	s_branch label_069E                                        
	
label_0A27:
	s_waitcnt lgkmcnt(0)                                       
	s_mul_i32 s62, s47, 0x100                                  
	s_mul_i32 s63, s46, 64                                     
	s_add_u32 s60, s62, s63                                    
	s_add_u32 s62, s60, 64                                     
	s_cmp_lt_i32 s44, s62                                      
	s_cbranch_scc1 label_0CB3                                  
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v181, 0, v177                                
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a96                                 
	v_accvgpr_read_b32 v9, a97                                 
	v_accvgpr_read_b32 v10, a98                                
	v_accvgpr_read_b32 v11, a99                                
	v_accvgpr_read_b32 v12, a104                               
	v_accvgpr_read_b32 v13, a105                               
	v_accvgpr_read_b32 v14, a106                               
	v_accvgpr_read_b32 v15, a107                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a100                                
	v_accvgpr_read_b32 v9, a101                                
	v_accvgpr_read_b32 v10, a102                               
	v_accvgpr_read_b32 v11, a103                               
	v_accvgpr_read_b32 v12, a108                               
	v_accvgpr_read_b32 v13, a109                               
	v_accvgpr_read_b32 v14, a110                               
	v_accvgpr_read_b32 v15, a111                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a128                                
	v_accvgpr_read_b32 v9, a129                                
	v_accvgpr_read_b32 v10, a130                               
	v_accvgpr_read_b32 v11, a131                               
	v_accvgpr_read_b32 v12, a136                               
	v_accvgpr_read_b32 v13, a137                               
	v_accvgpr_read_b32 v14, a138                               
	v_accvgpr_read_b32 v15, a139                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a132                                
	v_accvgpr_read_b32 v9, a133                                
	v_accvgpr_read_b32 v10, a134                               
	v_accvgpr_read_b32 v11, a135                               
	v_accvgpr_read_b32 v12, a140                               
	v_accvgpr_read_b32 v13, a141                               
	v_accvgpr_read_b32 v14, a142                               
	v_accvgpr_read_b32 v15, a143                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_add_u32_e32 v181, 64, v177                               
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a80                                 
	v_accvgpr_read_b32 v9, a81                                 
	v_accvgpr_read_b32 v10, a82                                
	v_accvgpr_read_b32 v11, a83                                
	v_accvgpr_read_b32 v12, a88                                
	v_accvgpr_read_b32 v13, a89                                
	v_accvgpr_read_b32 v14, a90                                
	v_accvgpr_read_b32 v15, a91                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a84                                 
	v_accvgpr_read_b32 v9, a85                                 
	v_accvgpr_read_b32 v10, a86                                
	v_accvgpr_read_b32 v11, a87                                
	v_accvgpr_read_b32 v12, a92                                
	v_accvgpr_read_b32 v13, a93                                
	v_accvgpr_read_b32 v14, a94                                
	v_accvgpr_read_b32 v15, a95                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a112                                
	v_accvgpr_read_b32 v9, a113                                
	v_accvgpr_read_b32 v10, a114                               
	v_accvgpr_read_b32 v11, a115                               
	v_accvgpr_read_b32 v12, a120                               
	v_accvgpr_read_b32 v13, a121                               
	v_accvgpr_read_b32 v14, a122                               
	v_accvgpr_read_b32 v15, a123                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a116                                
	v_accvgpr_read_b32 v9, a117                                
	v_accvgpr_read_b32 v10, a118                               
	v_accvgpr_read_b32 v11, a119                               
	v_accvgpr_read_b32 v12, a124                               
	v_accvgpr_read_b32 v13, a125                               
	v_accvgpr_read_b32 v14, a126                               
	v_accvgpr_read_b32 v15, a127                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a144                                
	v_accvgpr_read_b32 v9, a145                                
	v_accvgpr_read_b32 v10, a146                               
	v_accvgpr_read_b32 v11, a147                               
	v_accvgpr_read_b32 v12, a152                               
	v_accvgpr_read_b32 v13, a153                               
	v_accvgpr_read_b32 v14, a154                               
	v_accvgpr_read_b32 v15, a155                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a148                                
	v_accvgpr_read_b32 v9, a149                                
	v_accvgpr_read_b32 v10, a150                               
	v_accvgpr_read_b32 v11, a151                               
	v_accvgpr_read_b32 v12, a156                               
	v_accvgpr_read_b32 v13, a157                               
	v_accvgpr_read_b32 v14, a158                               
	v_accvgpr_read_b32 v15, a159                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	s_branch label_0F3C                                        
	
label_0CB3:
	s_mul_i32 s62, s36, 16                                     
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_0F3C                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v181, 0, v177                                
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a96                                 
	v_accvgpr_read_b32 v9, a97                                 
	v_accvgpr_read_b32 v10, a98                                
	v_accvgpr_read_b32 v11, a99                                
	v_accvgpr_read_b32 v12, a104                               
	v_accvgpr_read_b32 v13, a105                               
	v_accvgpr_read_b32 v14, a106                               
	v_accvgpr_read_b32 v15, a107                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a100                                
	v_accvgpr_read_b32 v9, a101                                
	v_accvgpr_read_b32 v10, a102                               
	v_accvgpr_read_b32 v11, a103                               
	v_accvgpr_read_b32 v12, a108                               
	v_accvgpr_read_b32 v13, a109                               
	v_accvgpr_read_b32 v14, a110                               
	v_accvgpr_read_b32 v15, a111                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a128                                
	v_accvgpr_read_b32 v9, a129                                
	v_accvgpr_read_b32 v10, a130                               
	v_accvgpr_read_b32 v11, a131                               
	v_accvgpr_read_b32 v12, a136                               
	v_accvgpr_read_b32 v13, a137                               
	v_accvgpr_read_b32 v14, a138                               
	v_accvgpr_read_b32 v15, a139                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a132                                
	v_accvgpr_read_b32 v9, a133                                
	v_accvgpr_read_b32 v10, a134                               
	v_accvgpr_read_b32 v11, a135                               
	v_accvgpr_read_b32 v12, a140                               
	v_accvgpr_read_b32 v13, a141                               
	v_accvgpr_read_b32 v14, a142                               
	v_accvgpr_read_b32 v15, a143                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_0F3C                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v181, 64, v177                               
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
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
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a80                                 
	v_accvgpr_read_b32 v9, a81                                 
	v_accvgpr_read_b32 v10, a82                                
	v_accvgpr_read_b32 v11, a83                                
	v_accvgpr_read_b32 v12, a88                                
	v_accvgpr_read_b32 v13, a89                                
	v_accvgpr_read_b32 v14, a90                                
	v_accvgpr_read_b32 v15, a91                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a84                                 
	v_accvgpr_read_b32 v9, a85                                 
	v_accvgpr_read_b32 v10, a86                                
	v_accvgpr_read_b32 v11, a87                                
	v_accvgpr_read_b32 v12, a92                                
	v_accvgpr_read_b32 v13, a93                                
	v_accvgpr_read_b32 v14, a94                                
	v_accvgpr_read_b32 v15, a95                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a112                                
	v_accvgpr_read_b32 v9, a113                                
	v_accvgpr_read_b32 v10, a114                               
	v_accvgpr_read_b32 v11, a115                               
	v_accvgpr_read_b32 v12, a120                               
	v_accvgpr_read_b32 v13, a121                               
	v_accvgpr_read_b32 v14, a122                               
	v_accvgpr_read_b32 v15, a123                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a116                                
	v_accvgpr_read_b32 v9, a117                                
	v_accvgpr_read_b32 v10, a118                               
	v_accvgpr_read_b32 v11, a119                               
	v_accvgpr_read_b32 v12, a124                               
	v_accvgpr_read_b32 v13, a125                               
	v_accvgpr_read_b32 v14, a126                               
	v_accvgpr_read_b32 v15, a127                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a144                                
	v_accvgpr_read_b32 v9, a145                                
	v_accvgpr_read_b32 v10, a146                               
	v_accvgpr_read_b32 v11, a147                               
	v_accvgpr_read_b32 v12, a152                               
	v_accvgpr_read_b32 v13, a153                               
	v_accvgpr_read_b32 v14, a154                               
	v_accvgpr_read_b32 v15, a155                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	v_accvgpr_read_b32 v8, a148                                
	v_accvgpr_read_b32 v9, a149                                
	v_accvgpr_read_b32 v10, a150                               
	v_accvgpr_read_b32 v11, a151                               
	v_accvgpr_read_b32 v12, a156                               
	v_accvgpr_read_b32 v13, a157                               
	v_accvgpr_read_b32 v14, a158                               
	v_accvgpr_read_b32 v15, a159                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v181, s[4:7], 0 offen       
	v_add_u32_e32 v181, s62, v181                              
	
label_0F3C:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

