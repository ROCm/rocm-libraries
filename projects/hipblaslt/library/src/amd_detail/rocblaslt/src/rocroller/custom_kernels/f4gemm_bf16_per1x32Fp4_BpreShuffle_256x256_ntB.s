// Auto-generated from FP4_GEMM_1TG_4W_MT256x256x256_B_Shuffle_Epilogue.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB:
	
	s_and_b32 s1, s1, 0xffff                                   
	s_load_dwordx2 s[4:5], s[0:1], 0x0                         
	s_load_dwordx2 s[8:9], s[0:1], 0x10                        
	s_load_dwordx2 s[12:13], s[0:1], 0x20                      
	s_load_dwordx2 s[16:17], s[0:1], 0x30                      
	s_load_dword s32, s[0:1], 0x40                             
	s_load_dword s33, s[0:1], 0x50                             
	s_load_dword s34, s[0:1], 0x80                             
	s_load_dword s35, s[0:1], 0xa0                             
	s_load_dword s36, s[0:1], 0xc0                             
	s_load_dword s39, s[0:1], 0xe0                             
	s_load_dword s40, s[0:1], 0xf0                             
	s_load_dword s41, s[0:1], 0x100                            
	s_load_dwordx2 s[20:21], s[0:1], 0x110                     
	s_load_dwordx2 s[24:25], s[0:1], 0x120                     
	s_load_dword s37, s[0:1], 0x130                            
	s_load_dword s38, s[0:1], 0x150                            
	v_lshrrev_b32_e32 v1, 10, v0                               
	v_lshrrev_b32_e32 v2, 10, v1                               
	v_and_b32_e32 v2, 0x3ff, v2                                
	v_and_b32_e32 v1, 0x3ff, v1                                
	v_and_b32_e32 v0, 0x3ff, v0                                
	v_lshrrev_b32_e32 v3, 6, v0                                
	v_and_b32_e32 v0, 63, v0                                   
	s_mov_b32 s53, s2                                          
	s_mov_b32 s54, s3                                          
	v_readfirstlane_b32 s42, v3                                
	s_mov_b32 s68, 0x80                                        
	s_mov_b32 s69, 0x800                                       
	s_mov_b32 s70, 0x100                                       
	s_mov_b32 s71, 0x100                                       
	s_waitcnt lgkmcnt(0)                                       
	s_add_u32 s46, s40, 0xff                                   
	s_lshr_b32 s46, s46, 8                                     
	s_mul_i32 s48, s46, s54                                    
	s_add_i32 s48, s48, s53                                    
	s_lshr_b32 s49, s46, 5                                     
	s_lshl_b32 s49, s49, 5                                     
	s_sub_i32 s50, s46, s49                                    
	s_add_u32 s51, s39, 0xff                                   
	s_lshr_b32 s51, s51, 8                                     
	s_mul_i32 s52, s49, s51                                    
	s_cmp_lt_i32 s48, s52                                      
	s_cbranch_scc0 label_006B                                  
	s_lshr_b32 s46, s48, 5                                     
	v_cvt_f32_u32_e32 v4, s51                                  
	s_sub_i32 s47, 0, s51                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s47, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s46, v4                                   
	v_mul_lo_u32 v5, v4, s51                                   
	v_sub_u32_e32 v7, s46, v5                                  
	v_add_u32_e32 v6, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s51, v7                              
	v_subrev_u32_e32 v5, s51, v7                               
	s_nop 0                                                    
	v_cndmask_b32_e32 v4, v4, v6, vcc                          
	v_cndmask_b32_e32 v7, v7, v5, vcc                          
	v_add_u32_e32 v5, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s51, v7                              
	s_nop 1                                                    
	v_cndmask_b32_e32 v7, v4, v5, vcc                          
	s_nop 3                                                    
	v_readfirstlane_b32 s47, v7                                
	s_nop 3                                                    
	s_mul_i32 s54, s51, s47                                    
	s_sub_i32 s54, s46, s54                                    
	s_and_b32 s46, s48, 31                                     
	s_lshl_b32 s53, s47, 5                                     
	s_add_i32 s53, s53, s46                                    
	s_branch label_008D                                        
	
label_006B:
	s_sub_i32 s46, s48, s52                                    
	v_cvt_f32_u32_e32 v4, s50                                  
	s_sub_i32 s54, 0, s50                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s54, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s46, v4                                   
	v_mul_lo_u32 v5, v4, s50                                   
	v_sub_u32_e32 v7, s46, v5                                  
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
	v_readfirstlane_b32 s54, v7                                
	s_nop 3                                                    
	s_mul_i32 s47, s50, s54                                    
	s_sub_i32 s53, s46, s47                                    
	s_add_i32 s53, s49, s53                                    
	
label_008D:
	s_lshr_b32 s35, s35, 1                                     
	s_mov_b32 s15, 0x20000                                     
	s_and_b32 s13, s13, 0xffff                                 
	s_or_b32 s13, s13, 0x40000                                 
	s_mul_i32 s14, s35, s39                                    
	s_mov_b32 s23, 0x20000                                     
	s_and_b32 s21, s21, 0xffff                                 
	s_or_b32 s21, s21, 0x40000                                 
	s_add_u32 s46, s39, 31                                     
	s_lshr_b32 s46, s46, 5                                     
	s_lshl_b32 s46, s46, 5                                     
	s_mul_i32 s22, s46, s37                                    
	v_lshrrev_b32_e32 v4, 3, v0                                
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_and_b32_e32 v4, 3, v4                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_lshlrev_b32_e32 v6, 2, v6                                
	v_add_u32_e32 v5, v5, v6                                   
	v_and_b32_e32 v4, 1, v4                                    
	v_add_u32_e32 v5, v5, v4                                   
	v_mul_lo_u32 v212, s35, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v212, v212, v4                               
	s_lshr_b32 s46, s42, 1                                     
	s_lshl_b32 s46, s46, 3                                     
	s_and_b32 s47, s42, 1                                      
	s_lshl_b32 s47, s47, 1                                     
	s_add_u32 s46, s46, s47                                    
	s_lshl_b32 s47, s54, 8                                     
	s_add_u32 s46, s46, s47                                    
	s_mul_i32 s46, s35, s46                                    
	v_add_u32_e32 v212, s46, v212                              
	s_lshl_b32 s46, s35, 5                                     
	v_add_u32_e32 v213, s46, v212                              
	v_add_u32_e32 v214, s46, v213                              
	v_add_u32_e32 v215, s46, v214                              
	v_add_u32_e32 v216, s46, v215                              
	v_add_u32_e32 v217, s46, v216                              
	v_add_u32_e32 v218, s46, v217                              
	v_add_u32_e32 v219, s46, v218                              
	s_mul_i32 s76, 0x420, s42                                  
	s_add_u32 s76, 0x1000, s76                                 
	s_lshl_b32 s46, s54, 8                                     
	s_lshl_b32 s47, s42, 5                                     
	s_add_i32 s47, s47, s46                                    
	s_mul_i32 s47, s47, s37                                    
	v_lshlrev_b32_e32 v222, 2, v0                              
	v_add_u32_e32 v222, s47, v222                              
	s_lshl_b32 s47, s37, 7                                     
	v_add_u32_e32 v223, s47, v222                              
	s_lshl_b32 s77, s42, 8                                     
	s_add_i32 s77, s77, 0                                      
	s_mov_b32 s33, s32                                         
	s_mov_b32 s43, 0                                           
	s_mov_b32 s44, s41                                         
	s_add_u32 m0, 0, s76                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	s_add_u32 m0, 0x1080, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	s_add_u32 m0, 0x2100, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	s_add_u32 m0, 0x3180, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	s_add_u32 m0, 0, s77                                       
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	s_add_u32 m0, 0x4200, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	s_add_u32 m0, 0x5280, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	s_add_u32 m0, 0x6300, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	s_add_u32 m0, 0x7380, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	s_add_u32 m0, 0x400, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	s_add_u32 s45, 0x100, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	s_lshr_b32 s36, s36, 1                                     
	s_mov_b32 s19, 0x20000                                     
	s_and_b32 s17, s17, 0xffff                                 
	s_or_b32 s17, s17, 0x40000                                 
	s_mul_i32 s18, s36, s40                                    
	s_mov_b32 s27, 0x20000                                     
	s_and_b32 s25, s25, 0xffff                                 
	s_or_b32 s25, s25, 0x40000                                 
	s_mul_i32 s26, s38, s40                                    
	s_lshl_b32 s46, s53, 8                                     
	s_lshl_b32 s47, s42, 6                                     
	s_add_i32 s46, s46, s47                                    
	s_mul_i32 s46, s46, s36                                    
	v_lshlrev_b32_e32 v225, 4, v0                              
	v_add_u32_e32 v225, s46, v225                              
	s_lshl_b32 s46, s36, 4                                     
	v_add_u32_e32 v226, s46, v225                              
	v_add_u32_e32 v227, s46, v226                              
	v_add_u32_e32 v228, s46, v227                              
	v_add_u32_e32 v229, 0x400, v225                            
	v_add_u32_e32 v230, 0x400, v226                            
	v_add_u32_e32 v231, 0x400, v227                            
	v_add_u32_e32 v232, 0x400, v228                            
	s_lshl_b32 s46, s53, 8                                     
	s_lshl_b32 s47, s42, 6                                     
	s_add_i32 s46, s46, s47                                    
	s_mul_i32 s46, s46, s38                                    
	v_lshlrev_b32_e32 v233, 2, v0                              
	v_add_u32_e32 v233, s46, v233                              
	s_mul_i32 s47, 32, s38                                     
	v_add_u32_e32 v234, s47, v233                              
	s_lshl_b32 s34, s34, 1                                     
	s_mov_b32 s7, 0x20000                                      
	s_and_b32 s5, s5, 0xffff                                   
	s_or_b32 s5, s5, 0x40000                                   
	s_mul_i32 s6, s34, s39                                     
	s_lshl_b32 s46, s53, 8                                     
	s_lshl_b32 s47, s42, 6                                     
	s_add_i32 s55, s46, s47                                    
	s_lshl_b32 s46, s55, 1                                     
	s_lshl_b32 s47, s54, 8                                     
	s_mul_i32 s47, s47, s34                                    
	s_add_u32 s46, s46, s47                                    
	s_add_u32 s4, s4, s46                                      
	s_addc_u32 s5, s5, 0                                       
	s_sub_u32 s6, s6, s46                                      
	v_and_b32_e64 v4, v0, 15                                   
	v_mul_lo_u32 v4, v4, s34                                   
	v_lshrrev_b32_e32 v5, 5, v0                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_lshrrev_b32_e32 v6, 4, v0                                
	v_and_b32_e32 v6, 1, v6                                    
	v_lshlrev_b32_e32 v6, 5, v6                                
	v_add3_u32 v235, v4, v5, v6                                
	s_lshl_b32 s47, s34, 4                                     
	v_add_u32_e32 v236, s47, v235                              
	v_add_u32_e32 v237, s47, v236                              
	v_add_u32_e32 v238, s47, v237                              
	v_add_u32_e32 v239, s47, v238                              
	v_add_u32_e32 v240, s47, v239                              
	v_add_u32_e32 v241, s47, v240                              
	v_add_u32_e32 v242, s47, v241                              
	v_add_u32_e32 v243, s47, v242                              
	v_add_u32_e32 v244, s47, v243                              
	v_add_u32_e32 v245, s47, v244                              
	v_add_u32_e32 v246, s47, v245                              
	v_add_u32_e32 v247, s47, v246                              
	v_add_u32_e32 v248, s47, v247                              
	v_add_u32_e32 v249, s47, v248                              
	v_add_u32_e32 v250, s47, v249                              
	buffer_load_dwordx4 v[136:139], v225, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	buffer_load_dwordx4 v[140:143], v226, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
	v_accvgpr_write_b32 a102, 0                                
	v_accvgpr_write_b32 a103, 0                                
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a104, 0                                
	v_accvgpr_write_b32 a105, 0                                
	v_accvgpr_write_b32 a106, 0                                
	v_accvgpr_write_b32 a107, 0                                
	v_accvgpr_write_b32 a108, 0                                
	v_accvgpr_write_b32 a109, 0                                
	v_accvgpr_write_b32 a110, 0                                
	v_accvgpr_write_b32 a111, 0                                
	buffer_load_dwordx4 v[152:155], v229, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a112, 0                                
	v_accvgpr_write_b32 a113, 0                                
	v_accvgpr_write_b32 a114, 0                                
	v_accvgpr_write_b32 a115, 0                                
	v_accvgpr_write_b32 a116, 0                                
	v_accvgpr_write_b32 a117, 0                                
	v_accvgpr_write_b32 a118, 0                                
	v_accvgpr_write_b32 a119, 0                                
	buffer_load_dwordx4 v[156:159], v230, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a120, 0                                
	v_accvgpr_write_b32 a121, 0                                
	v_accvgpr_write_b32 a122, 0                                
	v_accvgpr_write_b32 a123, 0                                
	v_accvgpr_write_b32 a124, 0                                
	v_accvgpr_write_b32 a125, 0                                
	v_accvgpr_write_b32 a126, 0                                
	v_accvgpr_write_b32 a127, 0                                
	buffer_load_dwordx4 v[160:163], v231, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a128, 0                                
	v_accvgpr_write_b32 a129, 0                                
	v_accvgpr_write_b32 a130, 0                                
	v_accvgpr_write_b32 a131, 0                                
	v_accvgpr_write_b32 a132, 0                                
	v_accvgpr_write_b32 a133, 0                                
	v_accvgpr_write_b32 a134, 0                                
	v_accvgpr_write_b32 a135, 0                                
	buffer_load_dwordx4 v[164:167], v232, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a136, 0                                
	v_accvgpr_write_b32 a137, 0                                
	v_accvgpr_write_b32 a138, 0                                
	v_accvgpr_write_b32 a139, 0                                
	v_accvgpr_write_b32 a140, 0                                
	v_accvgpr_write_b32 a141, 0                                
	v_accvgpr_write_b32 a142, 0                                
	v_accvgpr_write_b32 a143, 0                                
	buffer_load_dword v208, v233, s[24:27], 0 offen nt            
	v_accvgpr_write_b32 a144, 0                                
	v_accvgpr_write_b32 a145, 0                                
	v_accvgpr_write_b32 a146, 0                                
	v_accvgpr_write_b32 a147, 0                                
	v_accvgpr_write_b32 a148, 0                                
	v_accvgpr_write_b32 a149, 0                                
	v_accvgpr_write_b32 a150, 0                                
	v_accvgpr_write_b32 a151, 0                                
	buffer_load_dword v209, v234, s[24:27], 0 offen nt            
	v_accvgpr_write_b32 a152, 0                                
	v_accvgpr_write_b32 a153, 0                                
	v_accvgpr_write_b32 a154, 0                                
	v_accvgpr_write_b32 a155, 0                                
	v_accvgpr_write_b32 a156, 0                                
	v_accvgpr_write_b32 a157, 0                                
	v_accvgpr_write_b32 a158, 0                                
	v_accvgpr_write_b32 a159, 0                                
	s_add_u32 s45, 0x100, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	s_add_u32 s16, s16, s69                                    
	s_addc_u32 s17, s17, 0                                     
	s_add_u32 s24, s24, s71                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_u32 s18, s18, s69                                    
	s_sub_u32 s26, s26, s71                                    
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v220, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v220, v5, v220                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v220, v6, v220                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v4, 0x1000, v4                               
	v_add_u32_e32 v220, v4, v220                               
	v_add_u32_e32 v221, 0x8400, v220                           
	v_lshlrev_b32_e32 v224, 2, v0                              
	v_add_u32_e32 v224, 0, v224                                
	s_waitcnt vmcnt(15)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v220                                 
	ds_read_b128 v[40:43], v220 offset:64                      
	s_add_u32 m0, 0x8400, s76                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a160, 0                                
	v_accvgpr_write_b32 a161, 0                                
	v_accvgpr_write_b32 a162, 0                                
	v_accvgpr_write_b32 a163, 0                                
	v_accvgpr_write_b32 a164, 0                                
	v_accvgpr_write_b32 a165, 0                                
	v_accvgpr_write_b32 a166, 0                                
	v_accvgpr_write_b32 a167, 0                                
	ds_read_b128 v[12:15], v220 offset:512                     
	ds_read_b128 v[44:47], v220 offset:576                     
	s_add_u32 m0, 0x9480, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a168, 0                                
	v_accvgpr_write_b32 a169, 0                                
	v_accvgpr_write_b32 a170, 0                                
	v_accvgpr_write_b32 a171, 0                                
	v_accvgpr_write_b32 a172, 0                                
	v_accvgpr_write_b32 a173, 0                                
	v_accvgpr_write_b32 a174, 0                                
	v_accvgpr_write_b32 a175, 0                                
	ds_read_b128 v[16:19], v220 offset:4224                    
	ds_read_b128 v[48:51], v220 offset:4288                    
	s_add_u32 m0, 0xa500, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a176, 0                                
	v_accvgpr_write_b32 a177, 0                                
	v_accvgpr_write_b32 a178, 0                                
	v_accvgpr_write_b32 a179, 0                                
	v_accvgpr_write_b32 a180, 0                                
	v_accvgpr_write_b32 a181, 0                                
	v_accvgpr_write_b32 a182, 0                                
	v_accvgpr_write_b32 a183, 0                                
	ds_read_b128 v[20:23], v220 offset:4736                    
	ds_read_b128 v[52:55], v220 offset:4800                    
	s_add_u32 m0, 0xb580, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a184, 0                                
	v_accvgpr_write_b32 a185, 0                                
	v_accvgpr_write_b32 a186, 0                                
	v_accvgpr_write_b32 a187, 0                                
	v_accvgpr_write_b32 a188, 0                                
	v_accvgpr_write_b32 a189, 0                                
	v_accvgpr_write_b32 a190, 0                                
	v_accvgpr_write_b32 a191, 0                                
	ds_read_b128 v[24:27], v220 offset:8448                    
	ds_read_b128 v[56:59], v220 offset:8512                    
	s_add_u32 m0, 0x800, s77                                   
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a192, 0                                
	v_accvgpr_write_b32 a193, 0                                
	v_accvgpr_write_b32 a194, 0                                
	v_accvgpr_write_b32 a195, 0                                
	v_accvgpr_write_b32 a196, 0                                
	v_accvgpr_write_b32 a197, 0                                
	v_accvgpr_write_b32 a198, 0                                
	v_accvgpr_write_b32 a199, 0                                
	ds_read_b128 v[28:31], v220 offset:8960                    
	ds_read_b128 v[60:63], v220 offset:9024                    
	s_add_u32 m0, 0xc600, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a200, 0                                
	v_accvgpr_write_b32 a201, 0                                
	v_accvgpr_write_b32 a202, 0                                
	v_accvgpr_write_b32 a203, 0                                
	v_accvgpr_write_b32 a204, 0                                
	v_accvgpr_write_b32 a205, 0                                
	v_accvgpr_write_b32 a206, 0                                
	v_accvgpr_write_b32 a207, 0                                
	ds_read_b128 v[32:35], v220 offset:12672                   
	ds_read_b128 v[64:67], v220 offset:12736                   
	s_add_u32 m0, 0xd680, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a208, 0                                
	v_accvgpr_write_b32 a209, 0                                
	v_accvgpr_write_b32 a210, 0                                
	v_accvgpr_write_b32 a211, 0                                
	v_accvgpr_write_b32 a212, 0                                
	v_accvgpr_write_b32 a213, 0                                
	v_accvgpr_write_b32 a214, 0                                
	v_accvgpr_write_b32 a215, 0                                
	ds_read_b128 v[36:39], v220 offset:13184                   
	ds_read_b128 v[68:71], v220 offset:13248                   
	s_add_u32 m0, 0xe700, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a216, 0                                
	v_accvgpr_write_b32 a217, 0                                
	v_accvgpr_write_b32 a218, 0                                
	v_accvgpr_write_b32 a219, 0                                
	v_accvgpr_write_b32 a220, 0                                
	v_accvgpr_write_b32 a221, 0                                
	v_accvgpr_write_b32 a222, 0                                
	v_accvgpr_write_b32 a223, 0                                
	ds_read_b32 v200, v224                                     
	ds_read_b32 v201, v224 offset:256                          
	s_add_u32 m0, 0xf780, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a224, 0                                
	v_accvgpr_write_b32 a225, 0                                
	v_accvgpr_write_b32 a226, 0                                
	v_accvgpr_write_b32 a227, 0                                
	v_accvgpr_write_b32 a228, 0                                
	v_accvgpr_write_b32 a229, 0                                
	v_accvgpr_write_b32 a230, 0                                
	v_accvgpr_write_b32 a231, 0                                
	ds_read_b32 v202, v224 offset:512                          
	ds_read_b32 v203, v224 offset:768                          
	s_add_u32 m0, 0xc00, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a232, 0                                
	v_accvgpr_write_b32 a233, 0                                
	v_accvgpr_write_b32 a234, 0                                
	v_accvgpr_write_b32 a235, 0                                
	v_accvgpr_write_b32 a236, 0                                
	v_accvgpr_write_b32 a237, 0                                
	v_accvgpr_write_b32 a238, 0                                
	v_accvgpr_write_b32 a239, 0                                
	s_add_u32 s45, 0x200, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	v_accvgpr_write_b32 a240, 0                                
	v_accvgpr_write_b32 a241, 0                                
	v_accvgpr_write_b32 a242, 0                                
	v_accvgpr_write_b32 a243, 0                                
	v_accvgpr_write_b32 a244, 0                                
	v_accvgpr_write_b32 a245, 0                                
	v_accvgpr_write_b32 a246, 0                                
	v_accvgpr_write_b32 a247, 0                                
	v_accvgpr_write_b32 a248, 0                                
	v_accvgpr_write_b32 a249, 0                                
	v_accvgpr_write_b32 a250, 0                                
	v_accvgpr_write_b32 a251, 0                                
	v_accvgpr_write_b32 a252, 0                                
	v_accvgpr_write_b32 a253, 0                                
	v_accvgpr_write_b32 a254, 0                                
	v_accvgpr_write_b32 a255, 0                                
	s_mov_b32 s56, 0                                           
	s_add_u32 s46, s43, 0x100                                  
	s_cmp_lt_i32 s46, s44                                      
	s_cbranch_scc0 label_0EA1                                  
	s_cmp_lt_i32 s42, 2                                        
	s_cbranch_scc0 label_094A                                  
	
label_03F3:
	s_waitcnt vmcnt(10) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x200, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v220 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[8:11], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[12:15], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v220 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v220 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[16:19], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[20:23], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v220 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v220 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[24:27], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[28:31], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v220 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[32:35], a[24:27], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[36:39], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v220 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[32:35], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[36:39], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v220 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[8:11], a[64:67], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[12:15], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v220 offset:25344                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:151], v[8:11], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[184:187], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[12:15], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v220 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:147], v[16:19], a[72:75], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:147], v[20:23], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v220 offset:25856                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[16:19], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[20:23], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[124:127], v220 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[24:27], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[28:31], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v220 offset:29568                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:151], v[24:27], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[28:31], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[128:131], v220 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[32:35], a[88:91], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[36:39], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v220 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[32:35], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[36:39], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v220 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[40:43], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[44:47], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v224 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:159], v[40:43], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v210, v233, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[44:47], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v224 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[48:51], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[52:55], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v206, v224 offset:1536                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[48:51], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v211, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[52:55], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v207, v224 offset:1792                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[56:59], a[16:19], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s69                                    
	s_addc_u32 s17, s17, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[60:63], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s71                                    
	s_addc_u32 s25, s25, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[56:59], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s69                                    
	s_sub_u32 s26, s26, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[60:63], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[64:67], a[24:27], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[68:71], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[40:43], a[64:67], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[44:47], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[164:167], v[40:43], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[164:167], v[44:47], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[48:51], a[72:75], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[52:55], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[48:51], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[52:55], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[56:59], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[60:63], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[56:59], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[60:63], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[64:67], a[88:91], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[160:163], v[68:71], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v221                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[72:75], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s76                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[76:79], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v221 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[80:83], a[136:139], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[84:87], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v221 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[80:83], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[84:87], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v221 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[88:91], a[144:147], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[92:95], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v221 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[88:91], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[92:95], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v221 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[96:99], a[152:155], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[100:103], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v221 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[96:99], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[100:103], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v221 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[72:75], a[192:195], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[76:79], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v221 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:151], v[72:75], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s77                                       
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[148:151], v[76:79], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v221 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:147], v[80:83], a[200:203], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[84:87], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v221 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[80:83], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[84:87], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v221 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:147], v[88:91], a[208:211], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[144:147], v[92:95], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v221 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[88:91], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[92:95], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v221 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[96:99], a[216:219], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[144:147], v[100:103], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v221 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[96:99], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[100:103], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v221 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[104:107], a[128:131], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[108:111], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v224 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:159], v[104:107], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:159], v[108:111], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v224 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[112:115], a[136:139], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[116:119], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v224 offset:2560                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:159], v[112:115], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:159], v[116:119], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v224 offset:2816                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[120:123], a[144:147], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x300, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[124:127], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:159], v[120:123], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:159], v[124:127], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[128:131], a[152:155], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[132:135], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s56, 1                                           
	s_addk_i32 s43, 0x100                                      
	s_add_u32 s45, s43, 0x100                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:159], v[128:131], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:159], v[132:135], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:163], v[104:107], a[192:195], v209, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:163], v[108:111], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[164:167], v[104:107], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[164:167], v[108:111], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:163], v[112:115], a[200:203], v209, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:163], v[116:119], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[112:115], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[116:119], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[120:123], a[208:211], v209, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[124:127], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:167], v[120:123], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:167], v[124:127], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:163], v[128:131], a[216:219], v209, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:163], v[132:135], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:167], v[128:131], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:167], v[132:135], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0EA1                                  
	s_waitcnt vmcnt(10) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[8:11], a[0:3], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x200, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[12:15], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v221 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:175], v[8:11], a[32:35], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[12:15], a[36:39], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v221 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[16:19], a[8:11], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[168:171], v[20:23], a[12:15], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v221 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[16:19], a[40:43], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[20:23], a[44:47], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v221 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[168:171], v[24:27], a[16:19], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:171], v[28:31], a[20:23], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v221 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[172:175], v[24:27], a[48:51], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:175], v[28:31], a[52:55], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v221 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[168:171], v[32:35], a[24:27], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[168:171], v[36:39], a[28:31], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v221 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:175], v[32:35], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:175], v[36:39], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v221 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[8:11], a[64:67], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[176:179], v[12:15], a[68:71], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v221 offset:25344                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[180:183], v[8:11], a[96:99], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:183], v[12:15], a[100:103], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v221 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[176:179], v[16:19], a[72:75], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[176:179], v[20:23], a[76:79], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v221 offset:25856                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[16:19], a[104:107], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[20:23], a[108:111], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[124:127], v221 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[24:27], a[80:83], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[176:179], v[28:31], a[84:87], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v221 offset:29568                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:183], v[24:27], a[112:115], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[28:31], a[116:119], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[128:131], v221 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[176:179], v[32:35], a[88:91], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[176:179], v[36:39], a[92:95], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v221 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[32:35], a[120:123], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[36:39], a[124:127], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v221 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[184:187], v[40:43], a[0:3], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[44:47], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v224 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[188:191], v[40:43], a[32:35], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v208, v233, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[188:191], v[44:47], a[36:39], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v224 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[184:187], v[48:51], a[8:11], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[184:187], v[52:55], a[12:15], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v206, v224 offset:3584                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[188:191], v[48:51], a[40:43], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v209, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[52:55], a[44:47], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v207, v224 offset:3840                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[184:187], v[56:59], a[16:19], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s69                                    
	s_addc_u32 s17, s17, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[60:63], a[20:23], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s71                                    
	s_addc_u32 s25, s25, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[188:191], v[56:59], a[48:51], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s69                                    
	s_sub_u32 s26, s26, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[188:191], v[60:63], a[52:55], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[184:187], v[64:67], a[24:27], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[184:187], v[68:71], a[28:31], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[64:67], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[188:191], v[68:71], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:195], v[40:43], a[64:67], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[44:47], a[68:71], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[196:199], v[40:43], a[96:99], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[196:199], v[44:47], a[100:103], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[192:195], v[48:51], a[72:75], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[192:195], v[52:55], a[76:79], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[196:199], v[48:51], a[104:107], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[196:199], v[52:55], a[108:111], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[192:195], v[56:59], a[80:83], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[192:195], v[60:63], a[84:87], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:199], v[56:59], a[112:115], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[196:199], v[60:63], a[116:119], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[192:195], v[64:67], a[88:91], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[192:195], v[68:71], a[92:95], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[64:67], a[120:123], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[68:71], a[124:127], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:171], v[72:75], a[128:131], v210, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:171], v[76:79], a[132:135], v210, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v220                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[172:175], v[72:75], a[160:163], v210, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s76                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[172:175], v[76:79], a[164:167], v210, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v220 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:171], v[80:83], a[136:139], v210, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:171], v[84:87], a[140:143], v210, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v220 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[172:175], v[80:83], a[168:171], v210, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[172:175], v[84:87], a[172:175], v210, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v220 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:171], v[88:91], a[144:147], v210, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:171], v[92:95], a[148:151], v210, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v220 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[172:175], v[88:91], a[176:179], v210, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[172:175], v[92:95], a[180:183], v210, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v220 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:171], v[96:99], a[152:155], v210, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:171], v[100:103], a[156:159], v210, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v220 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[172:175], v[96:99], a[184:187], v210, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[172:175], v[100:103], a[188:191], v210, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v220 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:179], v[72:75], a[192:195], v211, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[176:179], v[76:79], a[196:199], v211, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v220 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[180:183], v[72:75], a[224:227], v211, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s77                                   
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[180:183], v[76:79], a[228:231], v211, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v220 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:179], v[80:83], a[200:203], v211, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[176:179], v[84:87], a[204:207], v211, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v220 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[80:83], a[232:235], v211, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[84:87], a[236:239], v211, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v220 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:179], v[88:91], a[208:211], v211, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[176:179], v[92:95], a[212:215], v211, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v220 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[180:183], v[88:91], a[240:243], v211, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[180:183], v[92:95], a[244:247], v211, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v220 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[176:179], v[96:99], a[216:219], v211, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[176:179], v[100:103], a[220:223], v211, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v220 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[180:183], v[96:99], a[248:251], v211, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:183], v[100:103], a[252:255], v211, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v220 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[184:187], v[104:107], a[128:131], v210, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[184:187], v[108:111], a[132:135], v210, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v224                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[188:191], v[104:107], a[160:163], v210, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[188:191], v[108:111], a[164:167], v210, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v224 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[184:187], v[112:115], a[136:139], v210, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[184:187], v[116:119], a[140:143], v210, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v224 offset:512                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[188:191], v[112:115], a[168:171], v210, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[188:191], v[116:119], a[172:175], v210, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v224 offset:768                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[184:187], v[120:123], a[144:147], v210, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x300, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[184:187], v[124:127], a[148:151], v210, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[188:191], v[120:123], a[176:179], v210, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[188:191], v[124:127], a[180:183], v210, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[184:187], v[128:131], a[152:155], v210, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[184:187], v[132:135], a[156:159], v210, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s56, 0                                           
	s_addk_i32 s43, 0x100                                      
	s_add_u32 s45, s43, 0x100                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[188:191], v[128:131], a[184:187], v210, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[188:191], v[132:135], a[188:191], v210, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:195], v[104:107], a[192:195], v211, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[192:195], v[108:111], a[196:199], v211, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[196:199], v[104:107], a[224:227], v211, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[196:199], v[108:111], a[228:231], v211, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[192:195], v[112:115], a[200:203], v211, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[192:195], v[116:119], a[204:207], v211, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[196:199], v[112:115], a[232:235], v211, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[196:199], v[116:119], a[236:239], v211, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[192:195], v[120:123], a[208:211], v211, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[192:195], v[124:127], a[212:215], v211, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[196:199], v[120:123], a[240:243], v211, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[196:199], v[124:127], a[244:247], v211, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[192:195], v[128:131], a[216:219], v211, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[192:195], v[132:135], a[220:223], v211, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[128:131], a[248:251], v211, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[132:135], a[252:255], v211, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0EA1                                  
	s_branch label_03F3                                        
	
label_094A:
	s_waitcnt vmcnt(10) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v220 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[8:11], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v220 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[12:15], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x200, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v220 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[16:19], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v220 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[20:23], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v220 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[24:27], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v220 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[28:31], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[32:35], a[24:27], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v220 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[36:39], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[32:35], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v220 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[36:39], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[8:11], a[64:67], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v220 offset:25344                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[12:15], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[184:187], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:151], v[8:11], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v220 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[12:15], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:147], v[16:19], a[72:75], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v220 offset:25856                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:147], v[20:23], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[16:19], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[124:127], v220 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[20:23], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[24:27], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v220 offset:29568                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[28:31], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:151], v[24:27], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[128:131], v220 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[28:31], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[32:35], a[88:91], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v220 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[36:39], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[32:35], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v220 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[36:39], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[40:43], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v224 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[44:47], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v210, v233, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:159], v[40:43], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v224 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[44:47], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[48:51], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v206, v224 offset:1536                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[52:55], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v211, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[48:51], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v207, v224 offset:1792                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[52:55], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s69                                    
	s_addc_u32 s17, s17, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[56:59], a[16:19], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s71                                    
	s_addc_u32 s25, s25, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[60:63], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s69                                    
	s_sub_u32 s26, s26, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[56:59], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[60:63], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[64:67], a[24:27], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[68:71], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[40:43], a[64:67], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[44:47], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[164:167], v[40:43], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[164:167], v[44:47], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[48:51], a[72:75], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[52:55], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[48:51], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[52:55], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[56:59], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[60:63], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[56:59], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[60:63], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[64:67], a[88:91], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[160:163], v[68:71], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v221                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s76                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[72:75], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v221 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[76:79], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[80:83], a[136:139], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v221 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[84:87], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[80:83], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v221 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[84:87], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[88:91], a[144:147], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v221 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[92:95], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[88:91], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v221 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[92:95], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[96:99], a[152:155], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v221 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[100:103], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[96:99], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v221 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[100:103], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[72:75], a[192:195], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v221 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[76:79], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s77                                       
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:151], v[72:75], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v221 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[148:151], v[76:79], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:147], v[80:83], a[200:203], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v221 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[84:87], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[80:83], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v221 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[84:87], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:147], v[88:91], a[208:211], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v221 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[144:147], v[92:95], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[88:91], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v221 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[92:95], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[96:99], a[216:219], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v221 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[144:147], v[100:103], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[96:99], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v221 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[100:103], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[104:107], a[128:131], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v224 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[108:111], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:159], v[104:107], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v224 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:159], v[108:111], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[112:115], a[136:139], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v224 offset:2560                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[116:119], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:159], v[112:115], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v224 offset:2816                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:159], v[116:119], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x300, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[120:123], a[144:147], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[124:127], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:159], v[120:123], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:159], v[124:127], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[128:131], a[152:155], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s56, 1                                           
	s_addk_i32 s43, 0x100                                      
	s_add_u32 s45, s43, 0x100                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[132:135], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:159], v[128:131], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:159], v[132:135], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:163], v[104:107], a[192:195], v209, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:163], v[108:111], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[164:167], v[104:107], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[164:167], v[108:111], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:163], v[112:115], a[200:203], v209, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:163], v[116:119], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[112:115], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[116:119], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[120:123], a[208:211], v209, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[124:127], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:167], v[120:123], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:167], v[124:127], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:163], v[128:131], a[216:219], v209, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:163], v[132:135], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:167], v[128:131], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:167], v[132:135], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0EA1                                  
	s_waitcnt vmcnt(10) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[8:11], a[0:3], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v221 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[12:15], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:175], v[8:11], a[32:35], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v221 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[12:15], a[36:39], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x200, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[16:19], a[8:11], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v221 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[168:171], v[20:23], a[12:15], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[16:19], a[40:43], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v221 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[20:23], a[44:47], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[168:171], v[24:27], a[16:19], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v221 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:171], v[28:31], a[20:23], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[172:175], v[24:27], a[48:51], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v221 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:175], v[28:31], a[52:55], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[168:171], v[32:35], a[24:27], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v221 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[168:171], v[36:39], a[28:31], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:175], v[32:35], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v221 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:175], v[36:39], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[8:11], a[64:67], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v221 offset:25344                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[176:179], v[12:15], a[68:71], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[180:183], v[8:11], a[96:99], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v221 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:183], v[12:15], a[100:103], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[176:179], v[16:19], a[72:75], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v221 offset:25856                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[176:179], v[20:23], a[76:79], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[16:19], a[104:107], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[124:127], v221 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[20:23], a[108:111], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[24:27], a[80:83], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v221 offset:29568                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[176:179], v[28:31], a[84:87], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:183], v[24:27], a[112:115], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[128:131], v221 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[28:31], a[116:119], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[176:179], v[32:35], a[88:91], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v221 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[176:179], v[36:39], a[92:95], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[32:35], a[120:123], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v221 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[36:39], a[124:127], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[184:187], v[40:43], a[0:3], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v224 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[44:47], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v208, v233, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[188:191], v[40:43], a[32:35], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v224 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[188:191], v[44:47], a[36:39], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[184:187], v[48:51], a[8:11], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v206, v224 offset:3584                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[184:187], v[52:55], a[12:15], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v209, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[188:191], v[48:51], a[40:43], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v207, v224 offset:3840                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[52:55], a[44:47], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s69                                    
	s_addc_u32 s17, s17, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[184:187], v[56:59], a[16:19], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s71                                    
	s_addc_u32 s25, s25, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[60:63], a[20:23], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s69                                    
	s_sub_u32 s26, s26, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[188:191], v[56:59], a[48:51], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[188:191], v[60:63], a[52:55], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[184:187], v[64:67], a[24:27], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[184:187], v[68:71], a[28:31], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[64:67], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[188:191], v[68:71], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:195], v[40:43], a[64:67], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[44:47], a[68:71], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[196:199], v[40:43], a[96:99], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[196:199], v[44:47], a[100:103], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[192:195], v[48:51], a[72:75], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[192:195], v[52:55], a[76:79], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[196:199], v[48:51], a[104:107], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[196:199], v[52:55], a[108:111], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[192:195], v[56:59], a[80:83], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[192:195], v[60:63], a[84:87], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:199], v[56:59], a[112:115], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[196:199], v[60:63], a[116:119], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[192:195], v[64:67], a[88:91], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[192:195], v[68:71], a[92:95], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[64:67], a[120:123], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[68:71], a[124:127], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:171], v[72:75], a[128:131], v210, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v220                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[168:171], v[76:79], a[132:135], v210, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s76                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[172:175], v[72:75], a[160:163], v210, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v220 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[172:175], v[76:79], a[164:167], v210, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:171], v[80:83], a[136:139], v210, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v220 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[168:171], v[84:87], a[140:143], v210, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s76                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[172:175], v[80:83], a[168:171], v210, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v220 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[172:175], v[84:87], a[172:175], v210, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:171], v[88:91], a[144:147], v210, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v220 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:171], v[92:95], a[148:151], v210, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s76                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[172:175], v[88:91], a[176:179], v210, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v220 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[172:175], v[92:95], a[180:183], v210, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:171], v[96:99], a[152:155], v210, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v220 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:171], v[100:103], a[156:159], v210, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s76                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[172:175], v[96:99], a[184:187], v210, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v220 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[172:175], v[100:103], a[188:191], v210, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:179], v[72:75], a[192:195], v211, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v220 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[176:179], v[76:79], a[196:199], v211, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s77                                   
	buffer_load_dword v222, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[180:183], v[72:75], a[224:227], v211, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v220 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[180:183], v[76:79], a[228:231], v211, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:179], v[80:83], a[200:203], v211, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v220 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[176:179], v[84:87], a[204:207], v211, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s76                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[80:83], a[232:235], v211, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v220 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[84:87], a[236:239], v211, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:179], v[88:91], a[208:211], v211, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v220 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[176:179], v[92:95], a[212:215], v211, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s76                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[180:183], v[88:91], a[240:243], v211, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v220 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[180:183], v[92:95], a[244:247], v211, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[176:179], v[96:99], a[216:219], v211, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v220 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[176:179], v[100:103], a[220:223], v211, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s76                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[180:183], v[96:99], a[248:251], v211, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v220 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:183], v[100:103], a[252:255], v211, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[184:187], v[104:107], a[128:131], v210, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v224                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[184:187], v[108:111], a[132:135], v210, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s76                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[188:191], v[104:107], a[160:163], v210, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v224 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[188:191], v[108:111], a[164:167], v210, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[184:187], v[112:115], a[136:139], v210, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v224 offset:512                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[184:187], v[116:119], a[140:143], v210, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s77                                   
	buffer_load_dword v223, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[188:191], v[112:115], a[168:171], v210, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v224 offset:768                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[188:191], v[116:119], a[172:175], v210, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s45, 0x300, s43                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[184:187], v[120:123], a[144:147], v210, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[184:187], v[124:127], a[148:151], v210, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s68                                    
	s_addc_u32 s13, s13, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[188:191], v[120:123], a[176:179], v210, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s70                                    
	s_addc_u32 s21, s21, 0                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[188:191], v[124:127], a[180:183], v210, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s68                                    
	s_sub_u32 s22, s22, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[184:187], v[128:131], a[152:155], v210, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s56, 0                                           
	s_addk_i32 s43, 0x100                                      
	s_add_u32 s45, s43, 0x100                                  
	s_cmp_lt_i32 s45, s44                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[184:187], v[132:135], a[156:159], v210, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[188:191], v[128:131], a[184:187], v210, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[188:191], v[132:135], a[188:191], v210, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[192:195], v[104:107], a[192:195], v211, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[192:195], v[108:111], a[196:199], v211, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[196:199], v[104:107], a[224:227], v211, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[196:199], v[108:111], a[228:231], v211, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[192:195], v[112:115], a[200:203], v211, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[192:195], v[116:119], a[204:207], v211, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[196:199], v[112:115], a[232:235], v211, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[196:199], v[116:119], a[236:239], v211, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[192:195], v[120:123], a[208:211], v211, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[192:195], v[124:127], a[212:215], v211, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[196:199], v[120:123], a[240:243], v211, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[196:199], v[124:127], a[244:247], v211, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[192:195], v[128:131], a[216:219], v211, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[192:195], v[132:135], a[220:223], v211, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[128:131], a[248:251], v211, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[132:135], a[252:255], v211, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0EA1                                  
	s_branch label_094A                                        
	
label_0EA1:
	s_mul_i32 s46, s56, 0x8400                                 
	v_add_u32_e32 v220, s46, v220                              
	s_lshl_b32 s46, s56, 11                                    
	v_add_u32_e32 v224, s46, v224                              
	s_waitcnt vmcnt(10)                                        
	s_barrier                                                  
	s_cmp_eq_u32 s56, 0                                        
	s_cbranch_scc1 label_0ECC                                  
	v_mov_b32_e32 v136, v168                                   
	v_mov_b32_e32 v137, v169                                   
	v_mov_b32_e32 v138, v170                                   
	v_mov_b32_e32 v139, v171                                   
	v_mov_b32_e32 v140, v172                                   
	v_mov_b32_e32 v141, v173                                   
	v_mov_b32_e32 v142, v174                                   
	v_mov_b32_e32 v143, v175                                   
	v_mov_b32_e32 v144, v176                                   
	v_mov_b32_e32 v145, v177                                   
	v_mov_b32_e32 v146, v178                                   
	v_mov_b32_e32 v147, v179                                   
	v_mov_b32_e32 v148, v180                                   
	v_mov_b32_e32 v149, v181                                   
	v_mov_b32_e32 v150, v182                                   
	v_mov_b32_e32 v151, v183                                   
	v_mov_b32_e32 v152, v184                                   
	v_mov_b32_e32 v153, v185                                   
	v_mov_b32_e32 v154, v186                                   
	v_mov_b32_e32 v155, v187                                   
	v_mov_b32_e32 v156, v188                                   
	v_mov_b32_e32 v157, v189                                   
	v_mov_b32_e32 v158, v190                                   
	v_mov_b32_e32 v159, v191                                   
	v_mov_b32_e32 v160, v192                                   
	v_mov_b32_e32 v161, v193                                   
	v_mov_b32_e32 v162, v194                                   
	v_mov_b32_e32 v163, v195                                   
	v_mov_b32_e32 v164, v196                                   
	v_mov_b32_e32 v165, v197                                   
	v_mov_b32_e32 v166, v198                                   
	v_mov_b32_e32 v167, v199                                   
	v_mov_b32_e32 v208, v210                                   
	v_mov_b32_e32 v209, v211                                   
	
label_0ECC:
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v220 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[40:43], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[8:11], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v220 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:159], v[40:43], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[8:11], a[64:67], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v220 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[40:43], a[64:67], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:151], v[8:11], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v220 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[164:167], v[40:43], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a0                                
	v_accvgpr_read_b32 v213, a1                                
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a2                                
	v_accvgpr_read_b32 v215, a3                                
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a32                               
	v_accvgpr_read_b32 v217, a33                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a34                               
	v_accvgpr_read_b32 v219, a35                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v235, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a64                               
	v_accvgpr_read_b32 v213, a65                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a66                               
	v_accvgpr_read_b32 v215, a67                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a96                               
	v_accvgpr_read_b32 v217, a97                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a98                               
	v_accvgpr_read_b32 v219, a99                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v235, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v220 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[44:47], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[12:15], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v220 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[44:47], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[12:15], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v220 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[44:47], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[12:15], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v220 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[164:167], v[44:47], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a4                                
	v_accvgpr_read_b32 v213, a5                                
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a6                                
	v_accvgpr_read_b32 v215, a7                                
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a36                               
	v_accvgpr_read_b32 v217, a37                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a38                               
	v_accvgpr_read_b32 v219, a39                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v236, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a68                               
	v_accvgpr_read_b32 v213, a69                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a70                               
	v_accvgpr_read_b32 v215, a71                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a100                              
	v_accvgpr_read_b32 v217, a101                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a102                              
	v_accvgpr_read_b32 v219, a103                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v236, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v220 offset:25344                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[48:51], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[16:19], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v220 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[48:51], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:147], v[16:19], a[72:75], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v220 offset:25856                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[48:51], a[72:75], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[16:19], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[124:127], v220 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[48:51], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a8                                
	v_accvgpr_read_b32 v213, a9                                
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a10                               
	v_accvgpr_read_b32 v215, a11                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a40                               
	v_accvgpr_read_b32 v217, a41                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a42                               
	v_accvgpr_read_b32 v219, a43                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v237, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a72                               
	v_accvgpr_read_b32 v213, a73                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a74                               
	v_accvgpr_read_b32 v215, a75                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a104                              
	v_accvgpr_read_b32 v217, a105                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a106                              
	v_accvgpr_read_b32 v219, a107                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v237, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v220 offset:29568                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[52:55], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[20:23], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[128:131], v220 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[52:55], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:147], v[20:23], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v220 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[52:55], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[20:23], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v220 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[52:55], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a12                               
	v_accvgpr_read_b32 v213, a13                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a14                               
	v_accvgpr_read_b32 v215, a15                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a44                               
	v_accvgpr_read_b32 v217, a45                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a46                               
	v_accvgpr_read_b32 v219, a47                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v238, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a76                               
	v_accvgpr_read_b32 v213, a77                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a78                               
	v_accvgpr_read_b32 v215, a79                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a108                              
	v_accvgpr_read_b32 v217, a109                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a110                              
	v_accvgpr_read_b32 v219, a111                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v238, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v204, v224 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[56:59], a[16:19], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[24:27], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v205, v224 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[56:59], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[24:27], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v206, v224 offset:1536                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[56:59], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:151], v[24:27], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v207, v224 offset:1792                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[56:59], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a16                               
	v_accvgpr_read_b32 v213, a17                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a18                               
	v_accvgpr_read_b32 v215, a19                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a48                               
	v_accvgpr_read_b32 v217, a49                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a50                               
	v_accvgpr_read_b32 v219, a51                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v239, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a80                               
	v_accvgpr_read_b32 v213, a81                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a82                               
	v_accvgpr_read_b32 v215, a83                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a112                              
	v_accvgpr_read_b32 v217, a113                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a114                              
	v_accvgpr_read_b32 v219, a115                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v239, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[60:63], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[28:31], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[60:63], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[28:31], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[60:63], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[28:31], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[60:63], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a20                               
	v_accvgpr_read_b32 v213, a21                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a22                               
	v_accvgpr_read_b32 v215, a23                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a52                               
	v_accvgpr_read_b32 v217, a53                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a54                               
	v_accvgpr_read_b32 v219, a55                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v240, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a84                               
	v_accvgpr_read_b32 v213, a85                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a86                               
	v_accvgpr_read_b32 v215, a87                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a116                              
	v_accvgpr_read_b32 v217, a117                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a118                              
	v_accvgpr_read_b32 v219, a119                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v240, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[32:35], a[24:27], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[64:67], a[24:27], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[32:35], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[32:35], a[88:91], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[64:67], a[88:91], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[32:35], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a24                               
	v_accvgpr_read_b32 v213, a25                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a26                               
	v_accvgpr_read_b32 v215, a27                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a56                               
	v_accvgpr_read_b32 v217, a57                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a58                               
	v_accvgpr_read_b32 v219, a59                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v241, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a88                               
	v_accvgpr_read_b32 v213, a89                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a90                               
	v_accvgpr_read_b32 v215, a91                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a120                              
	v_accvgpr_read_b32 v217, a121                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a122                              
	v_accvgpr_read_b32 v219, a123                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v241, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[36:39], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[68:71], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[36:39], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[36:39], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[160:163], v[68:71], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[36:39], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a28                               
	v_accvgpr_read_b32 v213, a29                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a30                               
	v_accvgpr_read_b32 v215, a31                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a60                               
	v_accvgpr_read_b32 v217, a61                               
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a62                               
	v_accvgpr_read_b32 v219, a63                               
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v242, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a92                               
	v_accvgpr_read_b32 v213, a93                               
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a94                               
	v_accvgpr_read_b32 v215, a95                               
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a124                              
	v_accvgpr_read_b32 v217, a125                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a126                              
	v_accvgpr_read_b32 v219, a127                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v242, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[104:107], a[128:131], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[72:75], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:159], v[104:107], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[72:75], a[192:195], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[160:163], v[104:107], a[192:195], v209, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:151], v[72:75], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[164:167], v[104:107], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a128                              
	v_accvgpr_read_b32 v213, a129                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a130                              
	v_accvgpr_read_b32 v215, a131                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a160                              
	v_accvgpr_read_b32 v217, a161                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a162                              
	v_accvgpr_read_b32 v219, a163                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v243, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a192                              
	v_accvgpr_read_b32 v213, a193                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a194                              
	v_accvgpr_read_b32 v215, a195                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a224                              
	v_accvgpr_read_b32 v217, a225                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a226                              
	v_accvgpr_read_b32 v219, a227                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v243, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[108:111], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[76:79], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:159], v[108:111], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[76:79], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[160:163], v[108:111], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[148:151], v[76:79], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[164:167], v[108:111], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a132                              
	v_accvgpr_read_b32 v213, a133                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a134                              
	v_accvgpr_read_b32 v215, a135                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a164                              
	v_accvgpr_read_b32 v217, a165                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a166                              
	v_accvgpr_read_b32 v219, a167                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v244, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a196                              
	v_accvgpr_read_b32 v213, a197                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a198                              
	v_accvgpr_read_b32 v215, a199                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a228                              
	v_accvgpr_read_b32 v217, a229                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a230                              
	v_accvgpr_read_b32 v219, a231                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v244, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[80:83], a[136:139], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[112:115], a[136:139], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[80:83], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:159], v[112:115], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:147], v[80:83], a[200:203], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[160:163], v[112:115], a[200:203], v209, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[80:83], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[112:115], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a136                              
	v_accvgpr_read_b32 v213, a137                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a138                              
	v_accvgpr_read_b32 v215, a139                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a168                              
	v_accvgpr_read_b32 v217, a169                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a170                              
	v_accvgpr_read_b32 v219, a171                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v245, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a200                              
	v_accvgpr_read_b32 v213, a201                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a202                              
	v_accvgpr_read_b32 v215, a203                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a232                              
	v_accvgpr_read_b32 v217, a233                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a234                              
	v_accvgpr_read_b32 v219, a235                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v245, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[84:87], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[116:119], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[84:87], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:159], v[116:119], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[84:87], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[160:163], v[116:119], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[84:87], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[116:119], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a140                              
	v_accvgpr_read_b32 v213, a141                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a142                              
	v_accvgpr_read_b32 v215, a143                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a172                              
	v_accvgpr_read_b32 v217, a173                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a174                              
	v_accvgpr_read_b32 v219, a175                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v246, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a204                              
	v_accvgpr_read_b32 v213, a205                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a206                              
	v_accvgpr_read_b32 v215, a207                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a236                              
	v_accvgpr_read_b32 v217, a237                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a238                              
	v_accvgpr_read_b32 v219, a239                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v246, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[88:91], a[144:147], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[120:123], a[144:147], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[88:91], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:159], v[120:123], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:147], v[88:91], a[208:211], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[120:123], a[208:211], v209, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[88:91], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:167], v[120:123], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a144                              
	v_accvgpr_read_b32 v213, a145                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a146                              
	v_accvgpr_read_b32 v215, a147                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a176                              
	v_accvgpr_read_b32 v217, a177                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a178                              
	v_accvgpr_read_b32 v219, a179                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v247, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a208                              
	v_accvgpr_read_b32 v213, a209                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a210                              
	v_accvgpr_read_b32 v215, a211                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a240                              
	v_accvgpr_read_b32 v217, a241                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a242                              
	v_accvgpr_read_b32 v219, a243                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v247, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[92:95], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[124:127], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[92:95], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:159], v[124:127], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[144:147], v[92:95], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[124:127], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[92:95], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:167], v[124:127], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a148                              
	v_accvgpr_read_b32 v213, a149                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a150                              
	v_accvgpr_read_b32 v215, a151                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a180                              
	v_accvgpr_read_b32 v217, a181                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a182                              
	v_accvgpr_read_b32 v219, a183                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v248, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a212                              
	v_accvgpr_read_b32 v213, a213                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a214                              
	v_accvgpr_read_b32 v215, a215                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a244                              
	v_accvgpr_read_b32 v217, a245                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a246                              
	v_accvgpr_read_b32 v219, a247                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v248, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[96:99], a[152:155], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[128:131], a[152:155], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[96:99], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[156:159], v[128:131], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[96:99], a[216:219], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[160:163], v[128:131], a[216:219], v209, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[96:99], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:167], v[128:131], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a152                              
	v_accvgpr_read_b32 v213, a153                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a154                              
	v_accvgpr_read_b32 v215, a155                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a184                              
	v_accvgpr_read_b32 v217, a185                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a186                              
	v_accvgpr_read_b32 v219, a187                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v249, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a216                              
	v_accvgpr_read_b32 v213, a217                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a218                              
	v_accvgpr_read_b32 v215, a219                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a248                              
	v_accvgpr_read_b32 v217, a249                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a250                              
	v_accvgpr_read_b32 v219, a251                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v249, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[100:103], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[132:135], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[100:103], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[156:159], v[132:135], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[144:147], v[100:103], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[160:163], v[132:135], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[100:103], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:167], v[132:135], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s46, s55                                         
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a156                              
	v_accvgpr_read_b32 v213, a157                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a158                              
	v_accvgpr_read_b32 v215, a159                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a188                              
	v_accvgpr_read_b32 v217, a189                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a190                              
	v_accvgpr_read_b32 v219, a191                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v250, 0                                  
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_cmp_lt_i32 s46, s40                                      
	s_cselect_b32 s74, -1, 0                                   
	s_cselect_b32 s75, -1, 0                                   
	s_and_saveexec_b64 s[72:73], s[74:75]                      
	v_accvgpr_read_b32 v212, a220                              
	v_accvgpr_read_b32 v213, a221                              
	v_pk_mul_f32 v[212:213], s[32:33], v[212:213]              
	v_accvgpr_read_b32 v214, a222                              
	v_accvgpr_read_b32 v215, a223                              
	v_pk_mul_f32 v[214:215], s[32:33], v[214:215]              
	v_accvgpr_read_b32 v216, a252                              
	v_accvgpr_read_b32 v217, a253                              
	v_pk_mul_f32 v[216:217], s[32:33], v[216:217]              
	v_accvgpr_read_b32 v218, a254                              
	v_accvgpr_read_b32 v219, a255                              
	v_pk_mul_f32 v[218:219], s[32:33], v[218:219]              
	v_cvt_pk_bf16_f32 v226, v212, v213                         
	v_cvt_pk_bf16_f32 v227, v214, v215                         
	v_cvt_pk_bf16_f32 v228, v216, v217                         
	v_cvt_pk_bf16_f32 v229, v218, v219                         
	s_nop 1                                                    
	.long 0x7fc4b3e4                                           
	s_nop 1                                                    
	.long 0x7fc6b3e5                                           
	s_nop 1                                                    
	v_add_u32_e64 v4, v250, 64                                 
	buffer_store_dwordx4 v[226:229], v4, s[4:7], 0 offen       
	s_mov_b64 exec, s[72:73]                                   
	s_addk_i32 s46, 0x20                                       
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_ntB.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

