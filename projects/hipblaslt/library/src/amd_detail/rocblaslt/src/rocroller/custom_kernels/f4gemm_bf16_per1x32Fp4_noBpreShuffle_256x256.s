// Auto-generated from f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256
.p2align 8
.type f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256,@function

f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256:
	
	s_and_b32 s1, s1, 0xffff                                   
	s_load_dword s25, s[0:1], 0xe0                             
	s_load_dword s26, s[0:1], 0xf0                             
	s_load_dword s27, s[0:1], 0x100                            
	s_load_dword s28, s[0:1], 0xa0                             
	s_load_dword s29, s[0:1], 0xc0                             
	s_load_dword s30, s[0:1], 0x80                             
	s_load_dword s20, s[0:1], 0x40                             
	s_load_dword s21, s[0:1], 0x50                             
	s_load_dwordx2 s[4:5], s[0:1], 0x20                        
	s_load_dwordx2 s[8:9], s[0:1], 0x30                        
	s_load_dwordx2 s[12:13], s[0:1], 0x10                      
	s_load_dwordx2 s[16:17], s[0:1], 0x0                       
	s_load_dwordx2 s[76:77], s[0:1], 0x110                     
	s_load_dwordx2 s[80:81], s[0:1], 0x120                     
	s_load_dword s93, s[0:1], 0x130                            
	s_load_dword s94, s[0:1], 0x150                            
	v_lshrrev_b32_e32 v1, 10, v0                               
	v_lshrrev_b32_e32 v2, 10, v1                               
	v_and_b32_e32 v2, 0x3ff, v2                                
	v_and_b32_e32 v1, 0x3ff, v1                                
	v_and_b32_e32 v0, 0x3ff, v0                                
	v_lshrrev_b32_e32 v3, 6, v0                                
	v_and_b32_e32 v0, 63, v0                                   
	s_mov_b32 s22, s2                                          
	s_mov_b32 s23, s3                                          
	v_readfirstlane_b32 s24, v3                                
	s_waitcnt lgkmcnt(0)                                       
	s_mov_b32 s18, -16                                         
	s_mov_b32 s14, -16                                         
	s_mov_b32 s10, -16                                         
	s_mov_b32 s6, -16                                          
	s_add_u32 s32, s25, 0xff                                   
	s_lshr_b32 s32, s32, 8                                     
	s_lshl_b32 s32, s32, 8                                     
	s_mul_i32 s31, s32, s93                                    
	s_mov_b32 s78, s31                                         
	s_mul_i32 s31, s26, s94                                    
	s_mov_b32 s82, s31                                         
	s_mov_b32 s19, 0x20000                                     
	s_mov_b32 s15, 0x20000                                     
	s_mov_b32 s11, 0x20000                                     
	s_mov_b32 s7, 0x20000                                      
	s_mov_b32 s79, 0x20000                                     
	s_mov_b32 s83, 0x20000                                     
	s_and_b32 s17, s17, 0xffff                                 
	s_and_b32 s13, s13, 0xffff                                 
	s_and_b32 s9, s9, 0xffff                                   
	s_and_b32 s5, s5, 0xffff                                   
	s_and_b32 s77, s77, 0xffff                                 
	s_and_b32 s81, s81, 0xffff                                 
	s_or_b32 s17, s17, 0x40000                                 
	s_or_b32 s13, s13, 0x40000                                 
	s_or_b32 s9, s9, 0x40000                                   
	s_or_b32 s5, s5, 0x40000                                   
	s_or_b32 s77, s77, 0x40000                                 
	s_or_b32 s81, s81, 0x40000                                 
	s_lshr_b32 s28, s28, 1                                     
	s_mul_i32 s31, s23, 0x100                                  
	s_mul_i32 s56, s31, s28                                    
	s_mul_i32 s31, s28, s25                                    
	s_mov_b32 s6, s31                                          
	v_lshrrev_b32_e32 v4, 3, v0                                
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_and_b32_e32 v4, 3, v4                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_lshlrev_b32_e32 v6, 2, v6                                
	v_add_u32_e32 v5, v5, v6                                   
	v_and_b32_e32 v4, 1, v4                                    
	v_add_u32_e32 v5, v5, v4                                   
	v_mul_lo_u32 v140, s28, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v140, v140, v4                               
	s_lshr_b32 s31, s24, 1                                     
	s_mul_i32 s31, s31, 8                                      
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 2                                      
	s_add_u32 s31, s31, s32                                    
	s_mul_i32 s31, s28, s31                                    
	s_add_u32 s56, s56, s31                                    
	v_add_u32_e32 v140, s56, v140                              
	s_mul_i32 s31, s28, 32                                     
	v_add_u32_e32 v141, s31, v140                              
	s_mul_i32 s31, s28, 0x80                                   
	v_add_u32_e32 v142, s31, v140                              
	v_add_u32_e32 v143, s31, v141                              
	s_mul_i32 s57, 0x420, s24                                  
	s_add_u32 s57, 0x2000, s57                                 
	s_add_u32 s58, 0x4200, s57                                 
	s_add_u32 s59, 0x8400, s57                                 
	s_add_u32 s60, 0x8400, s58                                 
	s_mov_b32 s61, 0x80                                        
	s_mul_i32 s62, 64, s28                                     
	s_mov_b32 s71, s4                                          
	s_mov_b32 s72, s5                                          
	s_mov_b32 s12, s4                                          
	s_mov_b32 s13, s5                                          
	s_mov_b32 s14, s6                                          
	s_mov_b32 s15, s7                                          
	s_cmp_le_u32 s14, s62                                      
	s_cselect_b32 s21, 0, s61                                  
	s_cselect_b32 s31, s14, s62                                
	s_add_u32 s12, s12, s62                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s31                                    
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v144, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v144, v5, v144                               
	v_add_u32_e32 v144, v6, v144                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v144, v4, v144                               
	s_mov_b32 s31, 0x2000                                      
	v_add_u32_e64 v144, v144, s31                              
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 0x2100                                 
	v_add_u32_e64 v144, v144, s32                              
	v_mov_b32_e32 v145, v144                                   
	s_lshr_b32 s29, s29, 1                                     
	s_mul_i32 s31, s22, 0x100                                  
	s_mul_i32 s63, s31, s29                                    
	s_mul_i32 s31, s29, s26                                    
	s_mov_b32 s10, s31                                         
	v_lshrrev_b32_e32 v4, 3, v0                                
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_and_b32_e32 v4, 3, v4                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_lshlrev_b32_e32 v6, 2, v6                                
	v_add_u32_e32 v5, v5, v6                                   
	v_and_b32_e32 v4, 1, v4                                    
	v_add_u32_e32 v5, v5, v4                                   
	v_mul_lo_u32 v146, s29, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v146, v146, v4                               
	s_lshr_b32 s31, s24, 1                                     
	s_mul_i32 s31, s31, 8                                      
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 2                                      
	s_add_u32 s31, s31, s32                                    
	s_mul_i32 s31, s29, s31                                    
	s_add_u32 s63, s63, s31                                    
	v_add_u32_e32 v146, s63, v146                              
	s_mul_i32 s31, s29, 32                                     
	v_add_u32_e32 v147, s31, v146                              
	s_mul_i32 s31, s29, 0x80                                   
	v_add_u32_e32 v148, s31, v146                              
	v_add_u32_e32 v149, s31, v147                              
	s_mul_i32 s65, 0x420, s24                                  
	s_add_u32 s65, 0x12800, s65                                
	s_add_u32 s66, 0x4200, s65                                 
	s_add_u32 s67, 0x8400, s65                                 
	s_add_u32 s68, 0x8400, s66                                 
	s_mov_b32 s69, 0x80                                        
	s_mul_i32 s70, 64, s29                                     
	s_mov_b32 s73, s8                                          
	s_mov_b32 s74, s9                                          
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v150, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v150, v5, v150                               
	v_add_u32_e32 v150, v6, v150                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v150, v4, v150                               
	s_mov_b32 s31, 0x12800                                     
	v_add_u32_e64 v150, v150, s31                              
	s_lshr_b32 s32, s24, 1                                     
	s_mul_i32 s32, s32, 0x2100                                 
	v_add_u32_e64 v150, v150, s32                              
	v_mov_b32_e32 v151, v150                                   
	s_lshl_b32 s30, s30, 1                                     
	s_mul_i32 s31, s23, 0x100                                  
	s_mul_hi_u32 s32, s31, s30                                 
	s_add_u32 s17, s17, s32                                    
	s_mul_i32 s32, s31, s30                                    
	s_add_u32 s16, s16, s32                                    
	s_addc_u32 s17, 0, s17                                     
	s_mov_b32 s31, 0                                           
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 0x80                                   
	s_add_i32 s31, s32, s31                                    
	s_mul_i32 s35, s31, s30                                    
	s_mul_i32 s31, s22, 0x100                                  
	s_lshr_b32 s32, s24, 1                                     
	s_mul_i32 s32, s32, 0x80                                   
	s_add_i32 s31, s32, s31                                    
	s_lshl_b32 s31, s31, 1                                     
	s_add_i32 s35, s31, s35                                    
	s_mul_i32 s32, s30, 16                                     
	s_add_i32 s36, s35, s32                                    
	s_add_i32 s37, s36, s32                                    
	s_add_i32 s38, s37, s32                                    
	s_add_i32 s39, s38, s32                                    
	s_add_i32 s40, s39, s32                                    
	s_add_i32 s41, s40, s32                                    
	s_add_i32 s42, s41, s32                                    
	s_mov_b32 s53, s35                                         
	v_and_b32_e64 v7, v0, 15                                   
	v_mul_lo_u32 v7, v7, s30                                   
	v_lshrrev_b32_e32 v3, 5, v0                                
	v_mul_i32_i24_e32 v3, 16, v3                               
	v_add_u32_e32 v7, v3, v7                                   
	v_and_b32_e32 v3, 31, v0                                   
	v_lshrrev_b32_e32 v3, 4, v3                                
	v_mul_i32_i24_e32 v3, 32, v3                               
	v_add_u32_e32 v7, v3, v7                                   
	v_lshrrev_b32_e32 v7, 2, v7                                
	v_and_b32_e64 v8, v0, 15                                   
	v_mul_lo_u32 v8, v8, s30                                   
	v_lshrrev_b32_e32 v3, 4, v0                                
	v_mul_i32_i24_e32 v3, 8, v3                                
	v_add_u32_e32 v8, v3, v8                                   
	v_lshrrev_b32_e32 v8, 2, v8                                
	v_lshlrev_b32_e32 v160, 2, v0                              
	v_lshlrev_b32_e32 v162, 2, v0                              
	s_mul_i32 s31, s23, 0x100                                  
	s_lshr_b32 s32, s24, 1                                     
	s_mul_i32 s32, s32, 0x80                                   
	s_add_i32 s31, s32, s31                                    
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 32                                     
	s_add_i32 s31, s32, s31                                    
	s_mul_i32 s32, s31, s93                                    
	v_add_u32_e32 v160, s32, v160                              
	s_mul_i32 s32, 64, s93                                     
	v_add_u32_e32 v161, s32, v160                              
	v_lshlrev_b32_e32 v164, 2, v0                              
	s_mov_b32 s31, 0                                           
	v_add_u32_e64 v164, v164, s31                              
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 0x200                                  
	v_add_u32_e64 v164, v164, s32                              
	s_mul_i32 s75, s24, 0x100                                  
	s_add_i32 s75, s75, 0                                      
	s_mul_i32 s31, s22, 0x100                                  
	s_lshr_b32 s32, s24, 1                                     
	s_mul_i32 s32, s32, 0x80                                   
	s_add_i32 s31, s32, s31                                    
	s_and_b32 s32, s24, 1                                      
	s_mul_i32 s32, s32, 32                                     
	s_add_i32 s31, s32, s31                                    
	s_mul_i32 s32, s31, s94                                    
	v_add_u32_e32 v162, s32, v162                              
	s_mul_i32 s32, 64, s94                                     
	v_add_u32_e32 v163, s32, v162                              
	v_lshlrev_b32_e32 v165, 2, v0                              
	s_mov_b32 s31, 0x1000                                      
	v_add_u32_e64 v165, v165, s31                              
	s_lshr_b32 s32, s24, 1                                     
	s_mul_i32 s32, s32, 0x200                                  
	v_add_u32_e64 v165, v165, s32                              
	s_mul_i32 s95, s24, 0x100                                  
	s_add_i32 s95, s95, 0x1000                                 
	s_mov_b32 s34, s27                                         
	s_mov_b32 s33, 0                                           
	s_mov_b32 m0, s57                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x1080, s57                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x2100, s57                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x3180, s57                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	s_add_u32 m0, s75, 0                                       
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	s_mov_b32 m0, s65                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x1080, s65                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x2100, s65                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x3180, s65                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	s_add_u32 m0, s95, 0                                       
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
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
	s_mov_b32 m0, s58                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x1080, s58                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x2100, s58                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x3180, s58                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s21                                    
	s_add_u32 m0, s75, 0x400                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	s_mov_b32 m0, s66                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x1080, s66                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x2100, s66                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x3180, s66                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	s_add_u32 m0, s95, 0x400                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
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
	s_cmp_eq_u32 s34, 0x200                                    
	s_cbranch_scc0 label_02DA                                  
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s61, s61, 0                                  
	s_cselect_b32 s21, s21, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	
label_02DA:
	s_mov_b32 m0, s59                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x1080, s59                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x2100, s59                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	s_add_u32 m0, 0x3180, s59                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	s_add_u32 m0, s75, 0x800                                   
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	s_mov_b32 m0, s67                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x1080, s67                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x2100, s67                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x3180, s67                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	s_add_u32 m0, s95, 0x800                                   
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
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
	s_mov_b32 m0, s60                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s21                                    
	s_add_u32 m0, s75, 0xc00                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	s_mov_b32 m0, s68                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x1080, s68                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x2100, s68                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	s_add_u32 m0, 0x3180, s68                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	s_add_u32 m0, s95, 0xc00                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
	v_accvgpr_write_b32 a192, 0                                
	v_accvgpr_write_b32 a193, 0                                
	v_accvgpr_write_b32 a194, 0                                
	v_accvgpr_write_b32 a195, 0                                
	v_accvgpr_write_b32 a196, 0                                
	v_accvgpr_write_b32 a197, 0                                
	v_accvgpr_write_b32 a198, 0                                
	v_accvgpr_write_b32 a199, 0                                
	v_accvgpr_write_b32 a200, 0                                
	v_accvgpr_write_b32 a201, 0                                
	v_accvgpr_write_b32 a202, 0                                
	v_accvgpr_write_b32 a203, 0                                
	v_accvgpr_write_b32 a204, 0                                
	v_accvgpr_write_b32 a205, 0                                
	v_accvgpr_write_b32 a206, 0                                
	v_accvgpr_write_b32 a207, 0                                
	v_accvgpr_write_b32 a208, 0                                
	v_accvgpr_write_b32 a209, 0                                
	v_accvgpr_write_b32 a210, 0                                
	v_accvgpr_write_b32 a211, 0                                
	v_accvgpr_write_b32 a212, 0                                
	v_accvgpr_write_b32 a213, 0                                
	v_accvgpr_write_b32 a214, 0                                
	v_accvgpr_write_b32 a215, 0                                
	v_accvgpr_write_b32 a216, 0                                
	v_accvgpr_write_b32 a217, 0                                
	v_accvgpr_write_b32 a218, 0                                
	v_accvgpr_write_b32 a219, 0                                
	v_accvgpr_write_b32 a220, 0                                
	v_accvgpr_write_b32 a221, 0                                
	v_accvgpr_write_b32 a222, 0                                
	v_accvgpr_write_b32 a223, 0                                
	v_accvgpr_write_b32 a224, 0                                
	v_accvgpr_write_b32 a225, 0                                
	v_accvgpr_write_b32 a226, 0                                
	v_accvgpr_write_b32 a227, 0                                
	v_accvgpr_write_b32 a228, 0                                
	v_accvgpr_write_b32 a229, 0                                
	v_accvgpr_write_b32 a230, 0                                
	v_accvgpr_write_b32 a231, 0                                
	v_accvgpr_write_b32 a232, 0                                
	v_accvgpr_write_b32 a233, 0                                
	v_accvgpr_write_b32 a234, 0                                
	v_accvgpr_write_b32 a235, 0                                
	v_accvgpr_write_b32 a236, 0                                
	v_accvgpr_write_b32 a237, 0                                
	v_accvgpr_write_b32 a238, 0                                
	v_accvgpr_write_b32 a239, 0                                
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
	s_waitcnt vmcnt(20)                                        
	s_barrier                                                  
	ds_read_b128 v[12:15], v144                                
	ds_read_b128 v[28:31], v144 offset:64                      
	ds_read_b128 v[16:19], v144 offset:512                     
	ds_read_b128 v[32:35], v144 offset:576                     
	ds_read_b128 v[20:23], v144 offset:4224                    
	ds_read_b128 v[36:39], v144 offset:4288                    
	ds_read_b128 v[24:27], v144 offset:4736                    
	ds_read_b128 v[40:43], v144 offset:4800                    
	ds_read_b32 v152, v164                                     
	ds_read_b32 v153, v164 offset:256                          
	ds_read_b128 v[76:79], v150                                
	ds_read_b128 v[92:95], v150 offset:64                      
	ds_read_b128 v[80:83], v150 offset:512                     
	ds_read_b128 v[96:99], v150 offset:576                     
	ds_read_b128 v[84:87], v150 offset:4224                    
	ds_read_b128 v[100:103], v150 offset:4288                  
	ds_read_b128 v[88:91], v150 offset:4736                    
	ds_read_b128 v[104:107], v150 offset:4800                  
	ds_read_b32 v156, v165                                     
	ds_read_b32 v157, v165 offset:256                          
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	s_cmp_lt_i32 s24, 2                                        
	s_cbranch_scc0 label_09F6                                  
	
label_046D:
	s_cmp_lt_i32 s33, s34                                      
	s_cbranch_scc0 label_0F7F                                  
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[12:15], a[0:3], v156, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v144 offset:16896                   
	ds_read_b128 v[60:63], v144 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[16:19], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s57                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[12:15], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v144 offset:17408                   
	ds_read_b128 v[64:67], v144 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[16:19], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v154, v164 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[12:15], a[64:67], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v144 offset:21120                   
	ds_read_b128 v[68:71], v144 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[16:19], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v155, v164 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[12:15], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v144 offset:21632                   
	ds_read_b128 v[72:75], v144 offset:21696                   
	v_add_u32_e32 v144, 0x8400, v144                           
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[16:19], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v158, v165 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[20:23], a[72:75], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v150 offset:16896                 
	ds_read_b128 v[124:127], v150 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[24:27], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s57                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[20:23], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v150 offset:17408                 
	ds_read_b128 v[128:131], v150 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[24:27], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v159, v165 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[20:23], a[8:11], v156, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v150 offset:21120                 
	ds_read_b128 v[132:135], v150 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[24:27], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0                                       
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[20:23], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v150 offset:21632                 
	ds_read_b128 v[136:139], v150 offset:21696                 
	v_add_u32_e32 v150, 0x8400, v150                           
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[24:27], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:95], v[28:31], a[0:3], v156, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:95], v[32:35], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s57                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[28:31], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[32:35], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[28:31], a[64:67], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[32:35], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[28:31], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[32:35], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[36:39], a[72:75], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[40:43], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s57                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[36:39], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[40:43], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[36:39], a[8:11], v156, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[40:43], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[36:39], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[40:43], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[12:15], a[128:131], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[16:19], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s65                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[12:15], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[16:19], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0                                       
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[12:15], a[192:195], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[16:19], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:123], v[12:15], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[16:19], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[20:23], a[200:203], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[24:27], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s65                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[20:23], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[24:27], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[20:23], a[136:139], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[24:27], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[112:115], v[20:23], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[112:115], v[24:27], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[28:31], a[128:131], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[32:35], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s65                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[28:31], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[32:35], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[28:31], a[192:195], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:135], v[32:35], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[136:139], v[28:31], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[32:35], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[36:39], a[200:203], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[132:135], v[40:43], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s65                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[36:39], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[40:43], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[36:39], a[136:139], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[40:43], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[36:39], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[40:43], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[44:47], a[16:19], v156, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[48:51], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s58                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[44:47], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[48:51], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0x400                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[44:47], a[80:83], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[48:51], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[44:47], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[48:51], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[52:55], a[88:91], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[56:59], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s58                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[52:55], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[56:59], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[52:55], a[24:27], v156, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[52:55], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[56:59], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[60:63], a[16:19], v156, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[64:67], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s58                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[60:63], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[64:67], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:103], v[60:63], a[80:83], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:103], v[64:67], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[60:63], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[64:67], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:103], v[68:71], a[88:91], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:103], v[72:75], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s58                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[68:71], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s61, s61, 0                                  
	s_cselect_b32 s21, s21, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[72:75], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[68:71], a[24:27], v156, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s21                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[72:75], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[68:71], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[72:75], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[108:111], v[44:47], a[144:147], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v144                                
	ds_read_b128 v[28:31], v144 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:111], v[48:51], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s66                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[44:47], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v144 offset:512                     
	ds_read_b128 v[32:35], v144 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[112:115], v[48:51], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v152, v164 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[44:47], a[208:211], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v144 offset:4224                    
	ds_read_b128 v[36:39], v144 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:119], v[48:51], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v153, v164 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[44:47], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v144 offset:4736                    
	ds_read_b128 v[40:43], v144 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[48:51], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v156, v165 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:119], v[52:55], a[216:219], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v150                                
	ds_read_b128 v[92:95], v150 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:119], v[56:59], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s66                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[52:55], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v150 offset:512                     
	ds_read_b128 v[96:99], v150 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[56:59], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v157, v165 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[52:55], a[152:155], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v150 offset:4224                    
	ds_read_b128 v[100:103], v150 offset:4288                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[56:59], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0x400                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[112:115], v[52:55], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v150 offset:4736                    
	ds_read_b128 v[104:107], v150 offset:4800                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[112:115], v[56:59], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s66                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[60:63], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[64:67], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[132:135], v[60:63], a[208:211], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[132:135], v[64:67], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[60:63], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[64:67], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:135], v[68:71], a[216:219], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:135], v[72:75], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s66                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[68:71], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[72:75], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[68:71], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[72:75], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s33, 0x100                                      
	s_cmp_lt_i32 s33, s34                                      
	s_cbranch_scc0 label_0F7F                                  
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[12:15], a[0:3], v156, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v144 offset:16896                   
	ds_read_b128 v[60:63], v144 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[16:19], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s59                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[12:15], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v144 offset:17408                   
	ds_read_b128 v[64:67], v144 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[16:19], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v154, v164 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[12:15], a[64:67], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v144 offset:21120                   
	ds_read_b128 v[68:71], v144 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[16:19], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v155, v164 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[12:15], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v144 offset:21632                   
	ds_read_b128 v[72:75], v144 offset:21696                   
	v_mov_b32_e32 v144, v145                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[16:19], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v158, v165 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[20:23], a[72:75], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v150 offset:16896                 
	ds_read_b128 v[124:127], v150 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[24:27], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s59                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[20:23], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v150 offset:17408                 
	ds_read_b128 v[128:131], v150 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[24:27], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v159, v165 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[20:23], a[8:11], v156, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v150 offset:21120                 
	ds_read_b128 v[132:135], v150 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[24:27], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0x800                                   
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[20:23], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v150 offset:21632                 
	ds_read_b128 v[136:139], v150 offset:21696                 
	v_mov_b32_e32 v150, v151                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[24:27], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:95], v[28:31], a[0:3], v156, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:95], v[32:35], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s59                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[28:31], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[32:35], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[28:31], a[64:67], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[32:35], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[28:31], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[32:35], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[36:39], a[72:75], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[40:43], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s59                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[36:39], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[40:43], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[36:39], a[8:11], v156, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[40:43], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[36:39], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[40:43], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[12:15], a[128:131], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[16:19], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s67                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[12:15], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[16:19], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0x800                                   
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[12:15], a[192:195], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[16:19], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:123], v[12:15], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[16:19], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[20:23], a[200:203], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[24:27], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s67                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[20:23], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[24:27], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[20:23], a[136:139], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[24:27], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[112:115], v[20:23], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[112:115], v[24:27], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[28:31], a[128:131], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[32:35], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s67                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[28:31], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[32:35], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[28:31], a[192:195], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:135], v[32:35], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[136:139], v[28:31], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[32:35], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[36:39], a[200:203], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[132:135], v[40:43], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s67                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[36:39], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[40:43], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[36:39], a[136:139], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[40:43], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[36:39], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[40:43], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[44:47], a[16:19], v156, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[48:51], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s60                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[44:47], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[48:51], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0xc00                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[44:47], a[80:83], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[48:51], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[44:47], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[48:51], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[52:55], a[88:91], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[56:59], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[52:55], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[56:59], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[52:55], a[24:27], v156, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[52:55], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[56:59], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[60:63], a[16:19], v156, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[64:67], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[60:63], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[64:67], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:103], v[60:63], a[80:83], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:103], v[64:67], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[60:63], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[64:67], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:103], v[68:71], a[88:91], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:103], v[72:75], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[68:71], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s61, s61, 0                                  
	s_cselect_b32 s21, s21, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[72:75], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[68:71], a[24:27], v156, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s21                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[72:75], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[68:71], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[72:75], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[108:111], v[44:47], a[144:147], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v144                                
	ds_read_b128 v[28:31], v144 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:111], v[48:51], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s68                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[44:47], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v144 offset:512                     
	ds_read_b128 v[32:35], v144 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[112:115], v[48:51], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v152, v164                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[44:47], a[208:211], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v144 offset:4224                    
	ds_read_b128 v[36:39], v144 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:119], v[48:51], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v153, v164 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[44:47], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v144 offset:4736                    
	ds_read_b128 v[40:43], v144 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[48:51], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v156, v165                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:119], v[52:55], a[216:219], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v150                                
	ds_read_b128 v[92:95], v150 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:119], v[56:59], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s68                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[52:55], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v150 offset:512                     
	ds_read_b128 v[96:99], v150 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[56:59], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v157, v165 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[52:55], a[152:155], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v150 offset:4224                    
	ds_read_b128 v[100:103], v150 offset:4288                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[56:59], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0xc00                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[112:115], v[52:55], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v150 offset:4736                    
	ds_read_b128 v[104:107], v150 offset:4800                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[112:115], v[56:59], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s68                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[60:63], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[64:67], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[132:135], v[60:63], a[208:211], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[132:135], v[64:67], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[60:63], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[64:67], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:135], v[68:71], a[216:219], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:135], v[72:75], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s68                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[68:71], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[72:75], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[68:71], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[72:75], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s33, 0x100                                      
	s_branch label_046D                                        
	
label_09F6:
	s_cmp_lt_i32 s33, s34                                      
	s_cbranch_scc0 label_0F7F                                  
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[12:15], a[0:3], v156, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s57                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[16:19], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v144 offset:16896                   
	ds_read_b128 v[60:63], v144 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[12:15], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v154, v164 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[16:19], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v144 offset:17408                   
	ds_read_b128 v[64:67], v144 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[12:15], a[64:67], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v155, v164 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[16:19], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v144 offset:21120                   
	ds_read_b128 v[68:71], v144 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[12:15], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v158, v165 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[16:19], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v144 offset:21632                   
	ds_read_b128 v[72:75], v144 offset:21696                   
	v_add_u32_e32 v144, 0x8400, v144                           
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[20:23], a[72:75], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s57                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[24:27], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v150 offset:16896                 
	ds_read_b128 v[124:127], v150 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[20:23], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v159, v165 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[24:27], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v150 offset:17408                 
	ds_read_b128 v[128:131], v150 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[20:23], a[8:11], v156, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0                                       
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[24:27], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v150 offset:21120                 
	ds_read_b128 v[132:135], v150 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[20:23], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[24:27], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v150 offset:21632                 
	ds_read_b128 v[136:139], v150 offset:21696                 
	v_add_u32_e32 v150, 0x8400, v150                           
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:95], v[28:31], a[0:3], v156, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s57                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:95], v[32:35], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[28:31], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[32:35], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[28:31], a[64:67], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[32:35], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[28:31], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[32:35], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[36:39], a[72:75], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s57                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[40:43], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[36:39], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[40:43], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[36:39], a[8:11], v156, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[40:43], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[36:39], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[40:43], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[12:15], a[128:131], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s65                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[16:19], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[12:15], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0                                       
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[16:19], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[12:15], a[192:195], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[16:19], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:123], v[12:15], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[16:19], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[20:23], a[200:203], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s65                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[24:27], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[20:23], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[24:27], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[20:23], a[136:139], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[24:27], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[112:115], v[20:23], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[112:115], v[24:27], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[28:31], a[128:131], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s65                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[32:35], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[28:31], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[32:35], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[28:31], a[192:195], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:135], v[32:35], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[136:139], v[28:31], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[32:35], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[36:39], a[200:203], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s65                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[132:135], v[40:43], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[36:39], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[40:43], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[36:39], a[136:139], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[40:43], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[36:39], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[40:43], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[44:47], a[16:19], v156, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s58                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[48:51], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[44:47], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0x400                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[48:51], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[44:47], a[80:83], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[48:51], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[44:47], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[48:51], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[52:55], a[88:91], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s58                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[56:59], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[52:55], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[56:59], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[52:55], a[24:27], v156, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[52:55], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[56:59], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[60:63], a[16:19], v156, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s58                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[64:67], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[60:63], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[64:67], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:103], v[60:63], a[80:83], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:103], v[64:67], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[60:63], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[64:67], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:103], v[68:71], a[88:91], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s58                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:103], v[72:75], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s61, s61, 0                                  
	s_cselect_b32 s21, s21, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[68:71], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[72:75], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s21                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[68:71], a[24:27], v156, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[72:75], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[68:71], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[72:75], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[108:111], v[44:47], a[144:147], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s66                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:111], v[48:51], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v144                                
	ds_read_b128 v[28:31], v144 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[44:47], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v152, v164 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[112:115], v[48:51], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v144 offset:512                     
	ds_read_b128 v[32:35], v144 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[44:47], a[208:211], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v153, v164 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:119], v[48:51], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v144 offset:4224                    
	ds_read_b128 v[36:39], v144 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[44:47], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v156, v165 offset:2048                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[48:51], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v144 offset:4736                    
	ds_read_b128 v[40:43], v144 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:119], v[52:55], a[216:219], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s66                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:119], v[56:59], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v150                                
	ds_read_b128 v[92:95], v150 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[52:55], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v157, v165 offset:2304                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[56:59], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v150 offset:512                     
	ds_read_b128 v[96:99], v150 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[52:55], a[152:155], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0x400                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[56:59], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v150 offset:4224                    
	ds_read_b128 v[100:103], v150 offset:4288                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[112:115], v[52:55], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[112:115], v[56:59], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v150 offset:4736                    
	ds_read_b128 v[104:107], v150 offset:4800                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s66                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[60:63], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[64:67], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[132:135], v[60:63], a[208:211], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[132:135], v[64:67], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[60:63], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[64:67], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:135], v[68:71], a[216:219], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s66                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:135], v[72:75], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[68:71], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[72:75], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[68:71], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[72:75], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s33, 0x100                                      
	s_cmp_lt_i32 s33, s34                                      
	s_cbranch_scc0 label_0F7F                                  
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[12:15], a[0:3], v156, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s59                                          
	buffer_load_dwordx4 v140, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[16:19], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v144 offset:16896                   
	ds_read_b128 v[60:63], v144 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[12:15], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v154, v164 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[16:19], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v144 offset:17408                   
	ds_read_b128 v[64:67], v144 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[12:15], a[64:67], v157, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v155, v164 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[16:19], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v144 offset:21120                   
	ds_read_b128 v[68:71], v144 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[12:15], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v158, v165 offset:3072                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[16:19], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v144 offset:21632                   
	ds_read_b128 v[72:75], v144 offset:21696                   
	v_mov_b32_e32 v144, v145                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[20:23], a[72:75], v157, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s59                                  
	buffer_load_dwordx4 v141, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[24:27], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v150 offset:16896                 
	ds_read_b128 v[124:127], v150 offset:16960                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[20:23], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v159, v165 offset:3328                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[24:27], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v150 offset:17408                 
	ds_read_b128 v[128:131], v150 offset:17472                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[20:23], a[8:11], v156, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0x800                                   
	buffer_load_dword v160, s[76:79], 0 offen lds              
	v_add_u32_e32 v160, 0x100, v160                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[24:27], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v150 offset:21120                 
	ds_read_b128 v[132:135], v150 offset:21184                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[20:23], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[24:27], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v150 offset:21632                 
	ds_read_b128 v[136:139], v150 offset:21696                 
	v_mov_b32_e32 v150, v151                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:95], v[28:31], a[0:3], v156, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s59                                  
	buffer_load_dwordx4 v142, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[92:95], v[32:35], a[4:7], v156, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[28:31], a[32:35], v156, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[32:35], a[36:39], v156, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[100:103], v[28:31], a[64:67], v157, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[32:35], a[68:71], v157, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[28:31], a[96:99], v157, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[32:35], a[100:103], v157, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[100:103], v[36:39], a[72:75], v157, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s59                                  
	buffer_load_dwordx4 v143, s[4:7], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[100:103], v[40:43], a[76:79], v157, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s4, s61, s4                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_u32 s6, s6, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[36:39], a[104:107], v157, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[40:43], a[108:111], v157, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[36:39], a[8:11], v156, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[40:43], a[12:15], v156, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[36:39], a[40:43], v156, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[40:43], a[44:47], v156, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[12:15], a[128:131], v158, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s67                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[16:19], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[12:15], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0x800                                   
	buffer_load_dword v162, s[80:83], 0 offen lds              
	v_add_u32_e32 v162, 0x100, v162                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[16:19], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[12:15], a[192:195], v159, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[16:19], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[120:123], v[12:15], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[16:19], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[20:23], a[200:203], v159, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s67                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[24:27], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[20:23], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[24:27], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[20:23], a[136:139], v158, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[108:111], v[24:27], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[112:115], v[20:23], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[112:115], v[24:27], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[28:31], a[128:131], v158, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s67                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[32:35], a[132:135], v158, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[28:31], a[160:163], v158, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[32:35], a[164:167], v158, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[28:31], a[192:195], v159, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[132:135], v[32:35], a[196:199], v159, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[136:139], v[28:31], a[224:227], v159, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[32:35], a[228:231], v159, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[36:39], a[200:203], v159, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s67                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[132:135], v[40:43], a[204:207], v159, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s8, s70, s8                                      
	s_addc_u32 s9, 0, s9                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[36:39], a[232:235], v159, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[40:43], a[236:239], v159, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[36:39], a[136:139], v158, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[40:43], a[140:143], v158, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[36:39], a[168:171], v158, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[40:43], a[172:175], v158, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[44:47], a[16:19], v156, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s60                                          
	buffer_load_dwordx4 v140, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[48:51], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[44:47], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s75, 0xc00                                   
	buffer_load_dword v161, s[76:79], 0 offen lds              
	v_add_u32_e32 v161, 0x100, v161                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[48:51], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[44:47], a[80:83], v157, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[48:51], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[44:47], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[48:51], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[52:55], a[88:91], v157, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v141, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[56:59], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[52:55], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[56:59], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[52:55], a[24:27], v156, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[56:59], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[52:55], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[56:59], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[60:63], a[16:19], v156, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v142, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[64:67], a[20:23], v156, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[96:99], v[60:63], a[48:51], v156, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[96:99], v[64:67], a[52:55], v156, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:103], v[60:63], a[80:83], v157, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:103], v[64:67], a[84:87], v157, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[60:63], a[112:115], v157, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[64:67], a[116:119], v157, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:103], v[68:71], a[88:91], v157, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v143, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:103], v[72:75], a[92:95], v157, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s61, s61, 0                                  
	s_cselect_b32 s21, s21, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[68:71], a[120:123], v157, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s61, s12                                    
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[72:75], a[124:127], v157, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s21                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[68:71], a[24:27], v156, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[72:75], a[28:31], v156, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[96:99], v[68:71], a[56:59], v156, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[96:99], v[72:75], a[60:63], v156, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[108:111], v[44:47], a[144:147], v158, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s68                                          
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[108:111], v[48:51], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v144                                
	ds_read_b128 v[28:31], v144 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[44:47], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v152, v164                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[112:115], v[48:51], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v144 offset:512                     
	ds_read_b128 v[32:35], v144 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[44:47], a[208:211], v159, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v153, v164 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[116:119], v[48:51], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v144 offset:4224                    
	ds_read_b128 v[36:39], v144 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[44:47], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v156, v165                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[48:51], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v144 offset:4736                    
	ds_read_b128 v[40:43], v144 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[116:119], v[52:55], a[216:219], v159, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s68                                  
	buffer_load_dwordx4 v147, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[116:119], v[56:59], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v150                                
	ds_read_b128 v[92:95], v150 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[52:55], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v157, v165 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[56:59], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v150 offset:512                     
	ds_read_b128 v[96:99], v150 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[52:55], a[152:155], v158, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, s95, 0xc00                                   
	buffer_load_dword v163, s[80:83], 0 offen lds              
	v_add_u32_e32 v163, 0x100, v163                            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[108:111], v[56:59], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v150 offset:4224                    
	ds_read_b128 v[100:103], v150 offset:4288                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[112:115], v[52:55], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[112:115], v[56:59], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v150 offset:4736                    
	ds_read_b128 v[104:107], v150 offset:4800                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v158, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s68                                  
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v158, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[60:63], a[176:179], v158, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[64:67], a[180:183], v158, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[132:135], v[60:63], a[208:211], v159, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[132:135], v[64:67], a[212:215], v159, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[60:63], a[240:243], v159, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[64:67], a[244:247], v159, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[132:135], v[68:71], a[216:219], v159, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s68                                  
	buffer_load_dwordx4 v149, s[8:11], 0 offen lds             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[132:135], v[72:75], a[220:223], v159, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s31, 0x300, s33                                  
	s_cmp_lt_u32 s31, s34                                      
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[68:71], a[248:251], v159, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s73, s73, s69                                    
	s_addc_u32 s74, 0, s74                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[72:75], a[252:255], v159, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_mov_b32 s8, s73                                          
	s_mov_b32 s9, s74                                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v158, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v158, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[68:71], a[184:187], v158, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[72:75], a[188:191], v158, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s33, 0x100                                      
	s_branch label_09F6                                        
	
label_0F7F:
	s_waitcnt lgkmcnt(0)                                       
	s_mov_b64 s[88:89], exec                                   
	s_mov_b32 s90, 0xffff                                      
	s_mov_b32 s91, 0xffff0000                                  
	s_mov_b32 s84, 1                                           
	s_mov_b32 s86, 8                                           
	s_mov_b32 s85, 0                                           
	s_mov_b32 s92, 0                                           
	s_mul_i32 s31, s30, 16                                     
	v_mov_b32_e32 v3, 1.0                                      
	
label_0F8B:
	s_cmp_lt_i32 s85, s84                                      
	s_cbranch_scc0 label_103F                                  
	s_mov_b32 s87, 0                                           
	s_mul_i32 s32, 0x80, s85                                   
	s_add_i32 s35, s35, s32                                    
	s_add_i32 s36, s36, s32                                    
	s_add_i32 s37, s37, s32                                    
	s_add_i32 s38, s38, s32                                    
	s_add_i32 s39, s39, s32                                    
	s_add_i32 s40, s40, s32                                    
	s_add_i32 s41, s41, s32                                    
	s_add_i32 s42, s42, s32                                    
	s_mov_b32 s51, s35                                         
	
label_0F99:
	s_cmp_lt_i32 s87, s86                                      
	s_cbranch_scc0 label_103D                                  
	s_add_u32 s87, 1, s87                                      
	s_set_gpr_idx_on s92, gpr_idx(SRC0)                        
	v_accvgpr_read_b32 v12, a0                                 
	v_mul_f32_e32 v12, s20, v12                                
	v_accvgpr_read_b32 v13, a1                                 
	v_mul_f32_e32 v13, s20, v13                                
	v_accvgpr_read_b32 v14, a2                                 
	v_mul_f32_e32 v14, s20, v14                                
	v_accvgpr_read_b32 v15, a3                                 
	v_mul_f32_e32 v15, s20, v15                                
	v_accvgpr_read_b32 v16, a32                                
	v_mul_f32_e32 v16, s20, v16                                
	v_accvgpr_read_b32 v17, a33                                
	v_mul_f32_e32 v17, s20, v17                                
	v_accvgpr_read_b32 v18, a34                                
	v_mul_f32_e32 v18, s20, v18                                
	v_accvgpr_read_b32 v19, a35                                
	v_mul_f32_e32 v19, s20, v19                                
	v_accvgpr_read_b32 v20, a64                                
	v_mul_f32_e32 v20, s20, v20                                
	v_accvgpr_read_b32 v21, a65                                
	v_mul_f32_e32 v21, s20, v21                                
	v_accvgpr_read_b32 v22, a66                                
	v_mul_f32_e32 v22, s20, v22                                
	v_accvgpr_read_b32 v23, a67                                
	v_mul_f32_e32 v23, s20, v23                                
	v_accvgpr_read_b32 v24, a96                                
	v_mul_f32_e32 v24, s20, v24                                
	v_accvgpr_read_b32 v25, a97                                
	v_mul_f32_e32 v25, s20, v25                                
	v_accvgpr_read_b32 v26, a98                                
	v_mul_f32_e32 v26, s20, v26                                
	v_accvgpr_read_b32 v27, a99                                
	v_mul_f32_e32 v27, s20, v27                                
	v_accvgpr_read_b32 v28, a128                               
	v_mul_f32_e32 v28, s20, v28                                
	v_accvgpr_read_b32 v29, a129                               
	v_mul_f32_e32 v29, s20, v29                                
	v_accvgpr_read_b32 v30, a130                               
	v_mul_f32_e32 v30, s20, v30                                
	v_accvgpr_read_b32 v31, a131                               
	v_mul_f32_e32 v31, s20, v31                                
	v_accvgpr_read_b32 v32, a160                               
	v_mul_f32_e32 v32, s20, v32                                
	v_accvgpr_read_b32 v33, a161                               
	v_mul_f32_e32 v33, s20, v33                                
	v_accvgpr_read_b32 v34, a162                               
	v_mul_f32_e32 v34, s20, v34                                
	v_accvgpr_read_b32 v35, a163                               
	v_mul_f32_e32 v35, s20, v35                                
	v_accvgpr_read_b32 v36, a192                               
	v_mul_f32_e32 v36, s20, v36                                
	v_accvgpr_read_b32 v37, a193                               
	v_mul_f32_e32 v37, s20, v37                                
	v_accvgpr_read_b32 v38, a194                               
	v_mul_f32_e32 v38, s20, v38                                
	v_accvgpr_read_b32 v39, a195                               
	v_mul_f32_e32 v39, s20, v39                                
	v_accvgpr_read_b32 v40, a224                               
	v_mul_f32_e32 v40, s20, v40                                
	v_accvgpr_read_b32 v41, a225                               
	v_mul_f32_e32 v41, s20, v41                                
	v_accvgpr_read_b32 v42, a226                               
	v_mul_f32_e32 v42, s20, v42                                
	v_accvgpr_read_b32 v43, a227                               
	v_mul_f32_e32 v43, s20, v43                                
	s_set_gpr_idx_off                                          
	s_addk_i32 s92, 0x4                                        
	v_cvt_pk_bf16_f32 v44, v12, v13                            
	v_cvt_pk_bf16_f32 v45, v14, v15                            
	v_cvt_pk_bf16_f32 v46, v16, v17                            
	v_cvt_pk_bf16_f32 v47, v18, v19                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v44, v46                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v45, v47                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[44:47], v7, s[16:19], s51 idxen     
	v_cvt_pk_bf16_f32 v44, v20, v21                            
	v_cvt_pk_bf16_f32 v45, v22, v23                            
	v_cvt_pk_bf16_f32 v46, v24, v25                            
	v_cvt_pk_bf16_f32 v47, v26, v27                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v44, v46                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v45, v47                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[44:47], v7, s[16:19], s51 idxen offset:64
	v_cvt_pk_bf16_f32 v44, v28, v29                            
	v_cvt_pk_bf16_f32 v45, v30, v31                            
	v_cvt_pk_bf16_f32 v46, v32, v33                            
	v_cvt_pk_bf16_f32 v47, v34, v35                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v44, v46                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v45, v47                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[44:47], v7, s[16:19], s51 idxen offset:128
	v_cvt_pk_bf16_f32 v44, v36, v37                            
	v_cvt_pk_bf16_f32 v45, v38, v39                            
	v_cvt_pk_bf16_f32 v46, v40, v41                            
	v_cvt_pk_bf16_f32 v47, v42, v43                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v44, v46                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v45, v47                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[44:47], v7, s[16:19], s51 idxen offset:192
	s_add_i32 s51, s51, s31                                    
	s_branch label_0F99                                        
	
label_103D:
	s_add_u32 s85, 1, s85                                      
	s_branch label_0F8B                                        
	
label_103F:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256
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
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 4
    .kernarg_segment_size: 368
    .max_flat_workgroup_size: 256
    .name:           f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

