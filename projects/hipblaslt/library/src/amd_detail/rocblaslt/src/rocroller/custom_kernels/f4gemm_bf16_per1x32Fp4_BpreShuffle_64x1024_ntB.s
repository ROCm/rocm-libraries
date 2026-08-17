// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB:
	
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
	s_add_u32 s51, s44, 0x3ff                                  
	s_lshr_b32 s50, s51, 10                                    
	s_mul_i32 s49, s50, s48                                    
	s_add_i32 s49, s49, s47                                    
	s_add_u32 s51, s43, 63                                     
	s_lshr_b32 s62, s51, 6                                     
	s_lshl_b32 s62, s62, 5                                     
	s_mov_b32 s47, 0                                           
	
label_0039:
	s_cmp_lt_i32 s49, s62                                      
	s_cbranch_scc1 label_003E                                  
	s_sub_i32 s49, s49, s62                                    
	s_add_i32 s47, s47, 32                                     
	s_branch label_0039                                        
	
label_003E:
	s_sub_i32 s50, s50, s47                                    
	s_cmp_lt_i32 s50, 32                                       
	s_cbranch_scc1 label_0044                                  
	s_lshr_b32 s48, s49, 5                                     
	s_and_b32 s62, s49, 31                                     
	s_branch label_0064                                        
	
label_0044:
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
	
label_0064:
	s_add_i32 s47, s62, s47                                    
	s_lshr_b32 s37, s37, 1                                     
	s_mul_i32 s62, s48, 64                                     
	s_mul_hi_u32 s63, s37, s62                                 
	s_add_u32 s13, s13, s63                                    
	s_mul_i32 s63, s37, s62                                    
	s_add_u32 s12, s12, s63                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_i32 s63, s43, s62                                    
	s_cmp_lt_u32 s63, 64                                       
	s_cselect_b32 s62, s63, 64                                 
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
	v_mul_lo_u32 v212, s37, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v212, v4, v212                               
	s_lshr_b32 s62, s46, 1                                     
	s_mul_i32 s62, s62, 8                                      
	s_and_b32 s63, s46, 1                                      
	s_mul_i32 s63, s63, 2                                      
	s_add_u32 s62, s62, s63                                    
	s_mul_i32 s62, s37, s62                                    
	v_add_u32_e32 v212, s62, v212                              
	s_mul_i32 s62, s37, 32                                     
	v_add_u32_e32 v213, s62, v212                              
	s_mul_i32 s64, 0x420, s46                                  
	s_add_u32 s64, 0x800, s64                                  
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v214, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v214, v5, v214                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v214, v6, v214                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v214, v4, v214                               
	v_add_u32_e32 v214, 0x800, v214                            
	v_add_u32_e32 v215, 0x2100, v214                           
	s_mul_i32 s62, s48, 64                                     
	s_mul_hi_u32 s63, s39, s62                                 
	s_add_u32 s21, s21, s63                                    
	s_mul_i32 s63, s39, s62                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_add_u32 s63, s43, 31                                     
	s_lshr_b32 s63, s63, 5                                     
	s_lshl_b32 s63, s63, 5                                     
	s_sub_i32 s63, s63, s62                                    
	s_cmp_lt_u32 s63, 64                                       
	s_cselect_b32 s62, s63, 64                                 
	s_mul_i32 s22, s39, s62                                    
	s_mov_b32 s23, 0x20000                                     
	v_lshlrev_b32_e32 v216, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v216, s63, v216                              
	s_mul_i32 s65, s46, 0x100                                  
	s_add_i32 s65, s65, 0                                      
	v_lshlrev_b32_e32 v217, 2, v0                              
	v_add_u32_e32 v217, 0, v217                                
	s_lshr_b32 s38, s38, 1                                     
	s_mul_i32 s62, s47, 0x400                                  
	s_mul_hi_u32 s63, s38, s62                                 
	s_add_u32 s17, s17, s63                                    
	s_mul_i32 s63, s38, s62                                    
	s_add_u32 s16, s16, s63                                    
	s_addc_u32 s17, s17, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x400                                    
	s_cselect_b32 s62, s63, 0x400                              
	s_mul_i32 s18, s38, s62                                    
	s_mov_b32 s19, 0x20000                                     
	v_lshlrev_b32_e32 v218, 4, v0                              
	s_mul_i32 s63, s46, 0x100                                  
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v218, s62, v218                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v219, s62, v218                              
	v_add_u32_e32 v220, s62, v219                              
	v_add_u32_e32 v221, s62, v220                              
	v_add_u32_e32 v222, s62, v221                              
	v_add_u32_e32 v223, s62, v222                              
	v_add_u32_e32 v224, s62, v223                              
	v_add_u32_e32 v225, s62, v224                              
	v_add_u32_e32 v226, s62, v225                              
	v_add_u32_e32 v227, s62, v226                              
	v_add_u32_e32 v228, s62, v227                              
	v_add_u32_e32 v229, s62, v228                              
	v_add_u32_e32 v230, s62, v229                              
	v_add_u32_e32 v231, s62, v230                              
	v_add_u32_e32 v232, s62, v231                              
	v_add_u32_e32 v233, s62, v232                              
	s_mul_i32 s62, s47, 0x400                                  
	s_mul_hi_u32 s63, s40, s62                                 
	s_add_u32 s25, s25, s63                                    
	s_mul_i32 s63, s40, s62                                    
	s_add_u32 s24, s24, s63                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x400                                    
	s_cselect_b32 s62, s63, 0x400                              
	s_mul_i32 s26, s40, s62                                    
	s_mov_b32 s27, 0x20000                                     
	v_lshlrev_b32_e32 v234, 2, v0                              
	s_mul_i32 s63, s46, 0x100                                  
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v234, s63, v234                              
	s_mul_i32 s62, 32, s40                                     
	v_add_u32_e32 v235, s62, v234                              
	v_add_u32_e32 v236, s62, v235                              
	v_add_u32_e32 v237, s62, v236                              
	v_add_u32_e32 v238, s62, v237                              
	v_add_u32_e32 v239, s62, v238                              
	v_add_u32_e32 v240, s62, v239                              
	v_add_u32_e32 v241, s62, v240                              
	s_mov_b32 s66, 0x80                                        
	s_mov_b32 s67, 0x800                                       
	s_mov_b32 s68, 0x100                                       
	s_mov_b32 s69, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v216, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	buffer_load_dwordx4 v[72:75], v218, s[16:19], 0 offen nt      
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	buffer_load_dwordx4 v[76:79], v219, s[16:19], 0 offen nt      
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	buffer_load_dwordx4 v[80:83], v218, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	buffer_load_dwordx4 v[84:87], v219, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	buffer_load_dword v204, v234, s[24:27], 0 offen nt            
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
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
	buffer_load_dwordx4 v[88:91], v220, s[16:19], 0 offen nt      
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	buffer_load_dwordx4 v[92:95], v221, s[16:19], 0 offen nt      
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	buffer_load_dwordx4 v[96:99], v220, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	buffer_load_dwordx4 v[100:103], v221, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	buffer_load_dword v205, v235, s[24:27], 0 offen nt            
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	buffer_load_dwordx4 v[104:107], v222, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	buffer_load_dwordx4 v[108:111], v223, s[16:19], 0 offen nt    
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
	buffer_load_dwordx4 v[112:115], v222, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	buffer_load_dwordx4 v[116:119], v223, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
	buffer_load_dword v206, v236, s[24:27], 0 offen nt            
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
	s_waitcnt vmcnt(15)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v214                                 
	ds_read_b128 v[16:19], v214 offset:64                      
	ds_read_b128 v[12:15], v214 offset:512                     
	ds_read_b128 v[20:23], v214 offset:576                     
	ds_read_b32 v200, v217                                     
	ds_read_b128 v[24:27], v214 offset:4224                    
	ds_read_b128 v[32:35], v214 offset:4288                    
	ds_read_b128 v[28:31], v214 offset:4736                    
	ds_read_b128 v[36:39], v214 offset:4800                    
	ds_read_b32 v201, v217 offset:256                          
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_lshl_b32 s36, s36, 1                                     
	s_mul_i32 s62, s48, 64                                     
	s_mul_hi_u32 s63, s36, s62                                 
	s_add_u32 s5, s5, s63                                      
	s_mul_i32 s63, s36, s62                                    
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_mul_i32 s63, s47, 0x400                                  
	s_lshl_b32 s63, s63, 1                                     
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_sub_i32 s62, s43, s62                                    
	s_cmp_lt_u32 s62, 64                                       
	s_cselect_b32 s62, s62, 64                                 
	s_mul_i32 s62, s36, s62                                    
	s_sub_i32 s6, s62, s63                                     
	s_mov_b32 s7, 0x20000                                      
	s_mul_i32 s62, s46, 0x100                                  
	s_lshl_b32 s62, s62, 1                                     
	v_lshrrev_b32_e32 v4, 5, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_lshrrev_b32_e32 v5, 4, v0                                
	v_and_b32_e32 v5, 1, v5                                    
	v_mul_i32_i24_e32 v5, 32, v5                               
	v_add_u32_e32 v4, v4, v5                                   
	v_and_b32_e32 v5, 15, v0                                   
	v_mul_lo_u32 v242, s36, v5                                 
	v_add_u32_e32 v242, s62, v242                              
	v_add_u32_e32 v242, v4, v242                               
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_089E                                  
	
label_0379:
	s_waitcnt vmcnt(10) lgkmcnt(5)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v204, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v224, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[8:11], a[8:11], v204, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[12:15], a[12:15], v204, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[16:19], a[0:3], v204, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[20:23], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v224, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[16:19], a[8:11], v204, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[20:23], a[12:15], v204, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v225, s[16:19], 0 offen offset:1024
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[24:27], a[128:131], v204, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[28:31], a[132:135], v204, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v237, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:79], v[24:27], a[136:139], v204, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[76:79], v[28:31], a[140:143], v204, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[32:35], a[128:131], v204, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[36:39], a[132:135], v204, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[32:35], a[136:139], v204, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[36:39], a[140:143], v204, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[8:11], a[16:19], v205, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[12:15], a[20:23], v205, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[8:11], a[24:27], v205, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[12:15], a[28:31], v205, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[16:19], a[16:19], v205, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[20:23], a[20:23], v205, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v226, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[16:19], a[24:27], v205, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[20:23], a[28:31], v205, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:91], v[24:27], a[144:147], v205, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:91], v[28:31], a[148:151], v205, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v238, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[24:27], a[152:155], v205, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[28:31], a[156:159], v205, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[32:35], a[144:147], v205, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[36:39], a[148:151], v205, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[32:35], a[152:155], v205, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[36:39], a[156:159], v205, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[8:11], a[32:35], v206, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[12:15], a[36:39], v206, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[8:11], a[40:43], v206, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[12:15], a[44:47], v206, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[16:19], a[32:35], v206, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[20:23], a[36:39], v206, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[16:19], a[40:43], v206, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[20:23], a[44:47], v206, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v229, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[24:27], a[160:163], v206, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[28:31], a[164:167], v206, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v239, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[24:27], a[168:171], v206, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[28:31], a[172:175], v206, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[32:35], a[160:163], v206, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[36:39], a[164:167], v206, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[32:35], a[168:171], v206, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[36:39], a[172:175], v206, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[8:11], a[48:51], v207, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[12:15], a[52:55], v207, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[8:11], a[56:59], v207, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[12:15], a[60:63], v207, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[16:19], a[48:51], v207, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[20:23], a[52:55], v207, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[16:19], a[56:59], v207, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[20:23], a[60:63], v207, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[24:27], a[176:179], v207, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[28:31], a[180:183], v207, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v240, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[24:27], a[184:187], v207, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[28:31], a[188:191], v207, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[32:35], a[176:179], v207, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[36:39], a[180:183], v207, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[32:35], a[184:187], v207, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[36:39], a[188:191], v207, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[8:11], a[64:67], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x100, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[12:15], a[68:71], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[8:11], a[72:75], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[12:15], a[76:79], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[16:19], a[64:67], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[20:23], a[68:71], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v232, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[16:19], a[72:75], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[20:23], a[76:79], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v233, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[24:27], a[192:195], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[28:31], a[196:199], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v241, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:143], v[24:27], a[200:203], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[28:31], a[204:207], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[32:35], a[192:195], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[36:39], a[196:199], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v216, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[32:35], a[200:203], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[36:39], a[204:207], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(11)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[8:11], a[80:83], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[12:15], a[84:87], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[8:11], a[88:91], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[12:15], a[92:95], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[16:19], a[80:83], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[20:23], a[84:87], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	buffer_load_dwordx4 v[72:75], v218, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[16:19], a[88:91], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[20:23], a[92:95], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[76:79], v219, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[24:27], a[208:211], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[28:31], a[212:215], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	buffer_load_dwordx4 v[80:83], v218, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[24:27], a[216:219], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[28:31], a[220:223], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v219, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[32:35], a[208:211], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[36:39], a[212:215], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v204, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:167], v[32:35], a[216:219], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:167], v[36:39], a[220:223], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[8:11], a[96:99], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[12:15], a[100:103], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v220, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[8:11], a[104:107], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[12:15], a[108:111], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v221, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[16:19], a[96:99], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[20:23], a[100:103], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v220, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[16:19], a[104:107], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[20:23], a[108:111], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v221, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:171], v[24:27], a[224:227], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:171], v[28:31], a[228:231], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v235, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:175], v[24:27], a[232:235], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:175], v[28:31], a[236:239], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[32:35], a[224:227], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[36:39], a[228:231], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[32:35], a[232:235], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[36:39], a[236:239], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[8:11], a[112:115], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[40:43], v215                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[12:15], a[116:119], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	buffer_load_dwordx4 v[104:107], v222, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[8:11], a[120:123], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v215 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[12:15], a[124:127], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v223, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[16:19], a[112:115], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v215 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[20:23], a[116:119], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v222, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[16:19], a[120:123], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v215 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[20:23], a[124:127], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v223, s[16:19], 0 offen offset:1024
	ds_read_b32 v202, v217 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[184:187], v[24:27], a[240:243], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v215 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[184:187], v[28:31], a[244:247], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v236, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[188:191], v[24:27], a[248:251], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v215 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[188:191], v[28:31], a[252:255], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[192:195], v[32:35], a[240:243], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v215 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[192:195], v[36:39], a[244:247], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[32:35], a[248:251], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v215 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[36:39], a[252:255], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v217 offset:1280                         
	s_cbranch_scc0 label_0DC3                                  
	s_waitcnt vmcnt(10) lgkmcnt(5)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v204, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v224, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[40:43], a[8:11], v204, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[44:47], a[12:15], v204, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[48:51], a[0:3], v204, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[52:55], a[4:7], v204, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v224, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[48:51], a[8:11], v204, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[52:55], a[12:15], v204, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v225, s[16:19], 0 offen offset:1024
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[56:59], a[128:131], v204, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[60:63], a[132:135], v204, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v237, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:79], v[56:59], a[136:139], v204, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[76:79], v[60:63], a[140:143], v204, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[64:67], a[128:131], v204, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[68:71], a[132:135], v204, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[64:67], a[136:139], v204, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[68:71], a[140:143], v204, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[40:43], a[16:19], v205, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[44:47], a[20:23], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[40:43], a[24:27], v205, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[44:47], a[28:31], v205, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[48:51], a[16:19], v205, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[52:55], a[20:23], v205, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v226, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[48:51], a[24:27], v205, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[52:55], a[28:31], v205, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:91], v[56:59], a[144:147], v205, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:91], v[60:63], a[148:151], v205, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v238, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[56:59], a[152:155], v205, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[60:63], a[156:159], v205, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[64:67], a[144:147], v205, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[68:71], a[148:151], v205, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[64:67], a[152:155], v205, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[68:71], a[156:159], v205, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[40:43], a[32:35], v206, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[44:47], a[36:39], v206, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[40:43], a[40:43], v206, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[44:47], a[44:47], v206, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[48:51], a[32:35], v206, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[52:55], a[36:39], v206, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[48:51], a[40:43], v206, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[52:55], a[44:47], v206, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v229, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[56:59], a[160:163], v206, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[60:63], a[164:167], v206, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v239, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[56:59], a[168:171], v206, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[60:63], a[172:175], v206, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[64:67], a[160:163], v206, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[68:71], a[164:167], v206, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[64:67], a[168:171], v206, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[68:71], a[172:175], v206, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[40:43], a[48:51], v207, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[44:47], a[52:55], v207, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[40:43], a[56:59], v207, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[44:47], a[60:63], v207, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[48:51], a[48:51], v207, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[52:55], a[52:55], v207, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[48:51], a[56:59], v207, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[52:55], a[60:63], v207, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[56:59], a[176:179], v207, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[60:63], a[180:183], v207, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v240, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[56:59], a[184:187], v207, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[60:63], a[188:191], v207, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[64:67], a[176:179], v207, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[68:71], a[180:183], v207, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[64:67], a[184:187], v207, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[68:71], a[188:191], v207, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[40:43], a[64:67], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x100, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[44:47], a[68:71], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s67, s67, 0                                  
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[40:43], a[72:75], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[44:47], a[76:79], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[48:51], a[64:67], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[52:55], a[68:71], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v232, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[48:51], a[72:75], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[52:55], a[76:79], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v233, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[56:59], a[192:195], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[60:63], a[196:199], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v241, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:143], v[56:59], a[200:203], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[60:63], a[204:207], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[64:67], a[192:195], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[68:71], a[196:199], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v216, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[64:67], a[200:203], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[68:71], a[204:207], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(11)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[40:43], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[44:47], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[40:43], a[88:91], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[44:47], a[92:95], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[48:51], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[52:55], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	buffer_load_dwordx4 v[72:75], v218, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[48:51], a[88:91], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[52:55], a[92:95], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[76:79], v219, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[56:59], a[208:211], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[60:63], a[212:215], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	buffer_load_dwordx4 v[80:83], v218, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[56:59], a[216:219], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[60:63], a[220:223], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v219, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[64:67], a[208:211], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[68:71], a[212:215], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v204, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:167], v[64:67], a[216:219], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:167], v[68:71], a[220:223], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[40:43], a[96:99], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[44:47], a[100:103], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v220, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[40:43], a[104:107], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[44:47], a[108:111], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v221, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[48:51], a[96:99], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[52:55], a[100:103], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v220, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[48:51], a[104:107], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[52:55], a[108:111], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v221, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:171], v[56:59], a[224:227], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:171], v[60:63], a[228:231], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v235, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:175], v[56:59], a[232:235], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:175], v[60:63], a[236:239], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[64:67], a[224:227], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[68:71], a[228:231], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[64:67], a[232:235], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[68:71], a[236:239], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[40:43], a[112:115], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[8:11], v214                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[44:47], a[116:119], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	buffer_load_dwordx4 v[104:107], v222, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[40:43], a[120:123], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v214 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[44:47], a[124:127], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v223, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[48:51], a[112:115], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v214 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[52:55], a[116:119], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v222, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[48:51], a[120:123], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v214 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[52:55], a[124:127], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v223, s[16:19], 0 offen offset:1024
	ds_read_b32 v200, v217                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[184:187], v[56:59], a[240:243], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v214 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[184:187], v[60:63], a[244:247], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v236, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[188:191], v[56:59], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v214 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[188:191], v[60:63], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[192:195], v[64:67], a[240:243], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v214 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[192:195], v[68:71], a[244:247], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[64:67], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v214 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[68:71], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v217 offset:256                          
	s_cbranch_scc0 label_0DC3                                  
	s_branch label_0379                                        
	
label_089E:
	s_waitcnt vmcnt(10) lgkmcnt(5)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v204, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v224, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[8:11], a[8:11], v204, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[12:15], a[12:15], v204, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[16:19], a[0:3], v204, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v224, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[20:23], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[16:19], a[8:11], v204, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v225, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[20:23], a[12:15], v204, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[24:27], a[128:131], v204, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v237, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[28:31], a[132:135], v204, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:79], v[24:27], a[136:139], v204, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[76:79], v[28:31], a[140:143], v204, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[32:35], a[128:131], v204, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[36:39], a[132:135], v204, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[32:35], a[136:139], v204, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[36:39], a[140:143], v204, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[8:11], a[16:19], v205, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[12:15], a[20:23], v205, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[8:11], a[24:27], v205, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[12:15], a[28:31], v205, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[16:19], a[16:19], v205, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v226, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[20:23], a[20:23], v205, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[16:19], a[24:27], v205, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[20:23], a[28:31], v205, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:91], v[24:27], a[144:147], v205, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v238, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:91], v[28:31], a[148:151], v205, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[24:27], a[152:155], v205, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[28:31], a[156:159], v205, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[32:35], a[144:147], v205, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[36:39], a[148:151], v205, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[32:35], a[152:155], v205, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[36:39], a[156:159], v205, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[8:11], a[32:35], v206, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[12:15], a[36:39], v206, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[8:11], a[40:43], v206, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[12:15], a[44:47], v206, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[16:19], a[32:35], v206, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[20:23], a[36:39], v206, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[16:19], a[40:43], v206, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v229, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[20:23], a[44:47], v206, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[24:27], a[160:163], v206, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v239, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[28:31], a[164:167], v206, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[24:27], a[168:171], v206, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[28:31], a[172:175], v206, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[32:35], a[160:163], v206, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[36:39], a[164:167], v206, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[32:35], a[168:171], v206, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[36:39], a[172:175], v206, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[8:11], a[48:51], v207, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[12:15], a[52:55], v207, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[8:11], a[56:59], v207, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[12:15], a[60:63], v207, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[16:19], a[48:51], v207, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[20:23], a[52:55], v207, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[16:19], a[56:59], v207, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[20:23], a[60:63], v207, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[24:27], a[176:179], v207, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v240, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[28:31], a[180:183], v207, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[24:27], a[184:187], v207, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[28:31], a[188:191], v207, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[32:35], a[176:179], v207, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[36:39], a[180:183], v207, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[32:35], a[184:187], v207, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[36:39], a[188:191], v207, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[8:11], a[64:67], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x100, s60                                  
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[12:15], a[68:71], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[8:11], a[72:75], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[12:15], a[76:79], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[16:19], a[64:67], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v232, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[20:23], a[68:71], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[16:19], a[72:75], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v233, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[20:23], a[76:79], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[24:27], a[192:195], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v241, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[28:31], a[196:199], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:143], v[24:27], a[200:203], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[28:31], a[204:207], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[32:35], a[192:195], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v216, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[36:39], a[196:199], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[32:35], a[200:203], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[36:39], a[204:207], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(11)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[8:11], a[80:83], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[12:15], a[84:87], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[8:11], a[88:91], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[12:15], a[92:95], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[16:19], a[80:83], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	buffer_load_dwordx4 v[72:75], v218, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[20:23], a[84:87], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[16:19], a[88:91], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[76:79], v219, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[20:23], a[92:95], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[24:27], a[208:211], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	buffer_load_dwordx4 v[80:83], v218, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[28:31], a[212:215], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[24:27], a[216:219], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v219, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[28:31], a[220:223], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[32:35], a[208:211], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v204, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[36:39], a[212:215], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:167], v[32:35], a[216:219], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:167], v[36:39], a[220:223], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[8:11], a[96:99], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v220, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[12:15], a[100:103], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[8:11], a[104:107], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v221, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[12:15], a[108:111], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[16:19], a[96:99], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v220, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[20:23], a[100:103], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[16:19], a[104:107], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v221, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[20:23], a[108:111], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:171], v[24:27], a[224:227], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v235, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:171], v[28:31], a[228:231], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:175], v[24:27], a[232:235], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:175], v[28:31], a[236:239], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[32:35], a[224:227], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[36:39], a[228:231], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[32:35], a[232:235], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[36:39], a[236:239], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[8:11], a[112:115], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	buffer_load_dwordx4 v[104:107], v222, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[12:15], a[116:119], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b128 v[40:43], v215                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[8:11], a[120:123], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v223, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[12:15], a[124:127], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v215 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[16:19], a[112:115], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v222, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[20:23], a[116:119], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v215 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[16:19], a[120:123], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v223, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[20:23], a[124:127], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v215 offset:576                     
	ds_read_b32 v202, v217 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[184:187], v[24:27], a[240:243], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v236, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[184:187], v[28:31], a[244:247], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v215 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[188:191], v[24:27], a[248:251], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[188:191], v[28:31], a[252:255], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v215 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[192:195], v[32:35], a[240:243], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[192:195], v[36:39], a[244:247], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v215 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[32:35], a[248:251], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[36:39], a[252:255], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v215 offset:4800                    
	ds_read_b32 v203, v217 offset:1280                         
	s_cbranch_scc0 label_0DC3                                  
	s_waitcnt vmcnt(10) lgkmcnt(5)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v204, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v224, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[40:43], a[8:11], v204, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v225, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[44:47], a[12:15], v204, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[48:51], a[0:3], v204, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v224, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[52:55], a[4:7], v204, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[48:51], a[8:11], v204, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v225, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[52:55], a[12:15], v204, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[56:59], a[128:131], v204, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v237, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[60:63], a[132:135], v204, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[76:79], v[56:59], a[136:139], v204, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[76:79], v[60:63], a[140:143], v204, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[64:67], a[128:131], v204, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[68:71], a[132:135], v204, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[64:67], a[136:139], v204, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[68:71], a[140:143], v204, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[40:43], a[16:19], v205, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v226, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[44:47], a[20:23], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[40:43], a[24:27], v205, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v227, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[44:47], a[28:31], v205, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[96:99], v[48:51], a[16:19], v205, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v226, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[96:99], v[52:55], a[20:23], v205, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[100:103], v[48:51], a[24:27], v205, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[100:103], v[52:55], a[28:31], v205, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[88:91], v[56:59], a[144:147], v205, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v238, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[88:91], v[60:63], a[148:151], v205, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[56:59], a[152:155], v205, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[60:63], a[156:159], v205, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[64:67], a[144:147], v205, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[68:71], a[148:151], v205, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[64:67], a[152:155], v205, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[68:71], a[156:159], v205, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[40:43], a[32:35], v206, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v228, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[44:47], a[36:39], v206, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[40:43], a[40:43], v206, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v229, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[44:47], a[44:47], v206, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[48:51], a[32:35], v206, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[52:55], a[36:39], v206, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[48:51], a[40:43], v206, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v229, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[52:55], a[44:47], v206, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[104:107], v[56:59], a[160:163], v206, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v239, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[104:107], v[60:63], a[164:167], v206, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[56:59], a[168:171], v206, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[108:111], v[60:63], a[172:175], v206, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[64:67], a[160:163], v206, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[112:115], v[68:71], a[164:167], v206, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[64:67], a[168:171], v206, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[68:71], a[172:175], v206, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[40:43], a[48:51], v207, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v230, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[44:47], a[52:55], v207, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[40:43], a[56:59], v207, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v231, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[44:47], a[60:63], v207, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[48:51], a[48:51], v207, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[52:55], a[52:55], v207, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[48:51], a[56:59], v207, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[52:55], a[60:63], v207, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[56:59], a[176:179], v207, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v240, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[60:63], a[180:183], v207, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[56:59], a[184:187], v207, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[60:63], a[188:191], v207, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[64:67], a[176:179], v207, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[68:71], a[180:183], v207, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[132:135], v[64:67], a[184:187], v207, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[132:135], v[68:71], a[188:191], v207, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[40:43], a[64:67], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x100, s60                                  
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[44:47], a[68:71], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[40:43], a[72:75], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[44:47], a[76:79], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[48:51], a[64:67], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v232, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[52:55], a[68:71], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[48:51], a[72:75], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v233, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[52:55], a[76:79], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[56:59], a[192:195], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v241, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[60:63], a[196:199], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[140:143], v[56:59], a[200:203], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[60:63], a[204:207], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[64:67], a[192:195], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v216, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[68:71], a[196:199], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[64:67], a[200:203], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[68:71], a[204:207], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(11)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[40:43], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[44:47], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[40:43], a[88:91], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[44:47], a[92:95], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[48:51], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	buffer_load_dwordx4 v[72:75], v218, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[52:55], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	s_add_u32 s12, s12, s66                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[48:51], a[88:91], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[76:79], v219, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[52:55], a[92:95], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[56:59], a[208:211], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	buffer_load_dwordx4 v[80:83], v218, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[60:63], a[212:215], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[56:59], a[216:219], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v219, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[60:63], a[220:223], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[160:163], v[64:67], a[208:211], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v204, v234, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[160:163], v[68:71], a[212:215], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[164:167], v[64:67], a[216:219], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[164:167], v[68:71], a[220:223], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[40:43], a[96:99], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v220, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[44:47], a[100:103], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[40:43], a[104:107], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v221, s[16:19], 0 offen nt      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[44:47], a[108:111], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[48:51], a[96:99], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v220, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[52:55], a[100:103], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[48:51], a[104:107], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v221, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[52:55], a[108:111], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[168:171], v[56:59], a[224:227], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v235, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[168:171], v[60:63], a[228:231], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[172:175], v[56:59], a[232:235], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[172:175], v[60:63], a[236:239], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[64:67], a[224:227], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[68:71], a[228:231], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[180:183], v[64:67], a[232:235], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[180:183], v[68:71], a[236:239], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(10)                                        
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[40:43], a[112:115], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	buffer_load_dwordx4 v[104:107], v222, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[44:47], a[116:119], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b128 v[8:11], v214                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[40:43], a[120:123], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v223, s[16:19], 0 offen nt    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[44:47], a[124:127], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v214 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[48:51], a[112:115], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v222, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[52:55], a[116:119], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v214 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[48:51], a[120:123], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v223, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[52:55], a[124:127], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v214 offset:576                     
	ds_read_b32 v200, v217                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[184:187], v[56:59], a[240:243], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v236, s[24:27], 0 offen nt            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[184:187], v[60:63], a[244:247], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v214 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[188:191], v[56:59], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[188:191], v[60:63], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v214 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[192:195], v[64:67], a[240:243], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[192:195], v[68:71], a[244:247], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v214 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[64:67], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[68:71], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v214 offset:4800                    
	ds_read_b32 v201, v217 offset:256                          
	s_cbranch_scc0 label_0DC3                                  
	s_branch label_089E                                        
	
label_0DC3:
	s_waitcnt lgkmcnt(0)                                       
	s_mul_i32 s62, s47, 0x400                                  
	s_mul_i32 s63, s46, 0x100                                  
	s_add_u32 s60, s62, s63                                    
	s_add_u32 s62, s60, 0x100                                  
	s_cmp_lt_i32 s44, s62                                      
	s_cbranch_scc1 label_11DD                                  
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v246, 0, v242                                
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 64, v242                               
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0x80, v242                             
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a160                                
	v_accvgpr_read_b32 v9, a161                                
	v_accvgpr_read_b32 v10, a162                               
	v_accvgpr_read_b32 v11, a163                               
	v_accvgpr_read_b32 v12, a168                               
	v_accvgpr_read_b32 v13, a169                               
	v_accvgpr_read_b32 v14, a170                               
	v_accvgpr_read_b32 v15, a171                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a164                                
	v_accvgpr_read_b32 v9, a165                                
	v_accvgpr_read_b32 v10, a166                               
	v_accvgpr_read_b32 v11, a167                               
	v_accvgpr_read_b32 v12, a172                               
	v_accvgpr_read_b32 v13, a173                               
	v_accvgpr_read_b32 v14, a174                               
	v_accvgpr_read_b32 v15, a175                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0xc0, v242                             
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a176                                
	v_accvgpr_read_b32 v9, a177                                
	v_accvgpr_read_b32 v10, a178                               
	v_accvgpr_read_b32 v11, a179                               
	v_accvgpr_read_b32 v12, a184                               
	v_accvgpr_read_b32 v13, a185                               
	v_accvgpr_read_b32 v14, a186                               
	v_accvgpr_read_b32 v15, a187                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a180                                
	v_accvgpr_read_b32 v9, a181                                
	v_accvgpr_read_b32 v10, a182                               
	v_accvgpr_read_b32 v11, a183                               
	v_accvgpr_read_b32 v12, a188                               
	v_accvgpr_read_b32 v13, a189                               
	v_accvgpr_read_b32 v14, a190                               
	v_accvgpr_read_b32 v15, a191                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0x100, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a192                                
	v_accvgpr_read_b32 v9, a193                                
	v_accvgpr_read_b32 v10, a194                               
	v_accvgpr_read_b32 v11, a195                               
	v_accvgpr_read_b32 v12, a200                               
	v_accvgpr_read_b32 v13, a201                               
	v_accvgpr_read_b32 v14, a202                               
	v_accvgpr_read_b32 v15, a203                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a196                                
	v_accvgpr_read_b32 v9, a197                                
	v_accvgpr_read_b32 v10, a198                               
	v_accvgpr_read_b32 v11, a199                               
	v_accvgpr_read_b32 v12, a204                               
	v_accvgpr_read_b32 v13, a205                               
	v_accvgpr_read_b32 v14, a206                               
	v_accvgpr_read_b32 v15, a207                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0x140, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a208                                
	v_accvgpr_read_b32 v9, a209                                
	v_accvgpr_read_b32 v10, a210                               
	v_accvgpr_read_b32 v11, a211                               
	v_accvgpr_read_b32 v12, a216                               
	v_accvgpr_read_b32 v13, a217                               
	v_accvgpr_read_b32 v14, a218                               
	v_accvgpr_read_b32 v15, a219                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a212                                
	v_accvgpr_read_b32 v9, a213                                
	v_accvgpr_read_b32 v10, a214                               
	v_accvgpr_read_b32 v11, a215                               
	v_accvgpr_read_b32 v12, a220                               
	v_accvgpr_read_b32 v13, a221                               
	v_accvgpr_read_b32 v14, a222                               
	v_accvgpr_read_b32 v15, a223                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0x180, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a224                                
	v_accvgpr_read_b32 v9, a225                                
	v_accvgpr_read_b32 v10, a226                               
	v_accvgpr_read_b32 v11, a227                               
	v_accvgpr_read_b32 v12, a232                               
	v_accvgpr_read_b32 v13, a233                               
	v_accvgpr_read_b32 v14, a234                               
	v_accvgpr_read_b32 v15, a235                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a228                                
	v_accvgpr_read_b32 v9, a229                                
	v_accvgpr_read_b32 v10, a230                               
	v_accvgpr_read_b32 v11, a231                               
	v_accvgpr_read_b32 v12, a236                               
	v_accvgpr_read_b32 v13, a237                               
	v_accvgpr_read_b32 v14, a238                               
	v_accvgpr_read_b32 v15, a239                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_add_u32_e32 v246, 0x1c0, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a240                                
	v_accvgpr_read_b32 v9, a241                                
	v_accvgpr_read_b32 v10, a242                               
	v_accvgpr_read_b32 v11, a243                               
	v_accvgpr_read_b32 v12, a248                               
	v_accvgpr_read_b32 v13, a249                               
	v_accvgpr_read_b32 v14, a250                               
	v_accvgpr_read_b32 v15, a251                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a244                                
	v_accvgpr_read_b32 v9, a245                                
	v_accvgpr_read_b32 v10, a246                               
	v_accvgpr_read_b32 v11, a247                               
	v_accvgpr_read_b32 v12, a252                               
	v_accvgpr_read_b32 v13, a253                               
	v_accvgpr_read_b32 v14, a254                               
	v_accvgpr_read_b32 v15, a255                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_branch label_1604                                        
	
label_11DD:
	s_mul_i32 s62, s36, 16                                     
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0, v242                                
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 64, v242                               
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0x80, v242                             
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a160                                
	v_accvgpr_read_b32 v9, a161                                
	v_accvgpr_read_b32 v10, a162                               
	v_accvgpr_read_b32 v11, a163                               
	v_accvgpr_read_b32 v12, a168                               
	v_accvgpr_read_b32 v13, a169                               
	v_accvgpr_read_b32 v14, a170                               
	v_accvgpr_read_b32 v15, a171                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a164                                
	v_accvgpr_read_b32 v9, a165                                
	v_accvgpr_read_b32 v10, a166                               
	v_accvgpr_read_b32 v11, a167                               
	v_accvgpr_read_b32 v12, a172                               
	v_accvgpr_read_b32 v13, a173                               
	v_accvgpr_read_b32 v14, a174                               
	v_accvgpr_read_b32 v15, a175                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0xc0, v242                             
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a176                                
	v_accvgpr_read_b32 v9, a177                                
	v_accvgpr_read_b32 v10, a178                               
	v_accvgpr_read_b32 v11, a179                               
	v_accvgpr_read_b32 v12, a184                               
	v_accvgpr_read_b32 v13, a185                               
	v_accvgpr_read_b32 v14, a186                               
	v_accvgpr_read_b32 v15, a187                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a180                                
	v_accvgpr_read_b32 v9, a181                                
	v_accvgpr_read_b32 v10, a182                               
	v_accvgpr_read_b32 v11, a183                               
	v_accvgpr_read_b32 v12, a188                               
	v_accvgpr_read_b32 v13, a189                               
	v_accvgpr_read_b32 v14, a190                               
	v_accvgpr_read_b32 v15, a191                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0x100, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a192                                
	v_accvgpr_read_b32 v9, a193                                
	v_accvgpr_read_b32 v10, a194                               
	v_accvgpr_read_b32 v11, a195                               
	v_accvgpr_read_b32 v12, a200                               
	v_accvgpr_read_b32 v13, a201                               
	v_accvgpr_read_b32 v14, a202                               
	v_accvgpr_read_b32 v15, a203                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a196                                
	v_accvgpr_read_b32 v9, a197                                
	v_accvgpr_read_b32 v10, a198                               
	v_accvgpr_read_b32 v11, a199                               
	v_accvgpr_read_b32 v12, a204                               
	v_accvgpr_read_b32 v13, a205                               
	v_accvgpr_read_b32 v14, a206                               
	v_accvgpr_read_b32 v15, a207                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0x140, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a208                                
	v_accvgpr_read_b32 v9, a209                                
	v_accvgpr_read_b32 v10, a210                               
	v_accvgpr_read_b32 v11, a211                               
	v_accvgpr_read_b32 v12, a216                               
	v_accvgpr_read_b32 v13, a217                               
	v_accvgpr_read_b32 v14, a218                               
	v_accvgpr_read_b32 v15, a219                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a212                                
	v_accvgpr_read_b32 v9, a213                                
	v_accvgpr_read_b32 v10, a214                               
	v_accvgpr_read_b32 v11, a215                               
	v_accvgpr_read_b32 v12, a220                               
	v_accvgpr_read_b32 v13, a221                               
	v_accvgpr_read_b32 v14, a222                               
	v_accvgpr_read_b32 v15, a223                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0x180, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a224                                
	v_accvgpr_read_b32 v9, a225                                
	v_accvgpr_read_b32 v10, a226                               
	v_accvgpr_read_b32 v11, a227                               
	v_accvgpr_read_b32 v12, a232                               
	v_accvgpr_read_b32 v13, a233                               
	v_accvgpr_read_b32 v14, a234                               
	v_accvgpr_read_b32 v15, a235                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a228                                
	v_accvgpr_read_b32 v9, a229                                
	v_accvgpr_read_b32 v10, a230                               
	v_accvgpr_read_b32 v11, a231                               
	v_accvgpr_read_b32 v12, a236                               
	v_accvgpr_read_b32 v13, a237                               
	v_accvgpr_read_b32 v14, a238                               
	v_accvgpr_read_b32 v15, a239                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1604                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v246, 0x1c0, v242                            
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
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
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a240                                
	v_accvgpr_read_b32 v9, a241                                
	v_accvgpr_read_b32 v10, a242                               
	v_accvgpr_read_b32 v11, a243                               
	v_accvgpr_read_b32 v12, a248                               
	v_accvgpr_read_b32 v13, a249                               
	v_accvgpr_read_b32 v14, a250                               
	v_accvgpr_read_b32 v15, a251                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	v_accvgpr_read_b32 v8, a244                                
	v_accvgpr_read_b32 v9, a245                                
	v_accvgpr_read_b32 v10, a246                               
	v_accvgpr_read_b32 v11, a247                               
	v_accvgpr_read_b32 v12, a252                               
	v_accvgpr_read_b32 v13, a253                               
	v_accvgpr_read_b32 v14, a254                               
	v_accvgpr_read_b32 v15, a255                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v246, s[4:7], 0 offen       
	v_add_u32_e32 v246, s62, v246                              
	
label_1604:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024_ntB.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

