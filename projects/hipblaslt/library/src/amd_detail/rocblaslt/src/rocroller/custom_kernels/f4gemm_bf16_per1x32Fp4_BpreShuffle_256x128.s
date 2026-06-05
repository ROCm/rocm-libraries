// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128:
	
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
	s_add_u32 s51, s43, 0xff                                   
	s_lshr_b32 s62, s51, 8                                     
	s_lshl_b32 s62, s62, 5                                     
	s_mov_b32 s47, 0                                           
	
label_003A:
	s_cmp_lt_i32 s49, s62                                      
	s_cbranch_scc1 label_003F                                  
	s_sub_i32 s49, s49, s62                                    
	s_add_i32 s47, s47, 32                                     
	s_branch label_003A                                        
	
label_003F:
	s_sub_i32 s50, s50, s47                                    
	s_cmp_lt_i32 s50, 32                                       
	s_cbranch_scc1 label_0045                                  
	s_lshr_b32 s48, s49, 5                                     
	s_and_b32 s62, s49, 31                                     
	s_branch label_0065                                        
	
label_0045:
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
	
label_0065:
	s_add_i32 s47, s62, s47                                    
	s_lshr_b32 s37, s37, 1                                     
	s_mul_i32 s62, s48, 0x100                                  
	s_mul_hi_u32 s63, s37, s62                                 
	s_add_u32 s13, s13, s63                                    
	s_mul_i32 s63, s37, s62                                    
	s_add_u32 s12, s12, s63                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_i32 s63, s43, s62                                    
	s_cmp_lt_u32 s63, 0x100                                    
	s_cselect_b32 s62, s63, 0x100                              
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
	v_add_u32_e32 v214, s62, v213                              
	v_add_u32_e32 v215, s62, v214                              
	v_add_u32_e32 v216, s62, v215                              
	v_add_u32_e32 v217, s62, v216                              
	v_add_u32_e32 v218, s62, v217                              
	v_add_u32_e32 v219, s62, v218                              
	s_mul_i32 s64, 0x420, s46                                  
	s_add_u32 s64, 0x2000, s64                                 
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
	v_add_u32_e32 v220, v4, v220                               
	v_add_u32_e32 v220, 0x2000, v220                           
	v_add_u32_e32 v221, 0x8400, v220                           
	v_add_u32_e32 v222, 0x8400, v221                           
	v_add_u32_e32 v223, 0x8400, v222                           
	s_mul_i32 s62, s48, 0x100                                  
	s_mul_hi_u32 s63, s39, s62                                 
	s_add_u32 s21, s21, s63                                    
	s_mul_i32 s63, s39, s62                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_add_u32 s63, s43, 31                                     
	s_lshr_b32 s63, s63, 5                                     
	s_lshl_b32 s63, s63, 5                                     
	s_sub_i32 s63, s63, s62                                    
	s_cmp_lt_u32 s63, 0x100                                    
	s_cselect_b32 s62, s63, 0x100                              
	s_mul_i32 s22, s39, s62                                    
	s_mov_b32 s23, 0x20000                                     
	v_lshlrev_b32_e32 v224, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v224, s63, v224                              
	s_mul_i32 s63, 0x80, s39                                   
	v_add_u32_e32 v225, s63, v224                              
	s_mul_i32 s65, s46, 0x100                                  
	s_add_i32 s65, s65, 0                                      
	v_lshlrev_b32_e32 v226, 2, v0                              
	v_add_u32_e32 v226, 0, v226                                
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
	v_lshlrev_b32_e32 v227, 4, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v227, s62, v227                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v228, s62, v227                              
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
	v_lshlrev_b32_e32 v229, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v229, s63, v229                              
	s_mov_b32 s66, 0x80                                        
	s_mov_b32 s67, 0x800                                       
	s_mov_b32 s68, 0x100                                       
	s_mov_b32 s69, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	buffer_load_dwordx4 v[136:139], v227, s[16:19], 0 offen    
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	buffer_load_dwordx4 v[140:143], v228, s[16:19], 0 offen    
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen offset:1024
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	buffer_load_dword v208, v229, s[24:27], 0 offen            
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
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
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a102, 0                                
	v_accvgpr_write_b32 a103, 0                                
	v_accvgpr_write_b32 a104, 0                                
	v_accvgpr_write_b32 a105, 0                                
	v_accvgpr_write_b32 a106, 0                                
	v_accvgpr_write_b32 a107, 0                                
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a108, 0                                
	v_accvgpr_write_b32 a109, 0                                
	v_accvgpr_write_b32 a110, 0                                
	v_accvgpr_write_b32 a111, 0                                
	v_accvgpr_write_b32 a112, 0                                
	v_accvgpr_write_b32 a113, 0                                
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a114, 0                                
	v_accvgpr_write_b32 a115, 0                                
	v_accvgpr_write_b32 a116, 0                                
	v_accvgpr_write_b32 a117, 0                                
	v_accvgpr_write_b32 a118, 0                                
	v_accvgpr_write_b32 a119, 0                                
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a120, 0                                
	v_accvgpr_write_b32 a121, 0                                
	v_accvgpr_write_b32 a122, 0                                
	v_accvgpr_write_b32 a123, 0                                
	v_accvgpr_write_b32 a124, 0                                
	v_accvgpr_write_b32 a125, 0                                
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a126, 0                                
	v_accvgpr_write_b32 a127, 0                                
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xf780, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	buffer_load_dwordx4 v[152:155], v227, s[16:19], 0 offen    
	buffer_load_dwordx4 v[156:159], v228, s[16:19], 0 offen    
	buffer_load_dwordx4 v[160:163], v227, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[164:167], v228, s[16:19], 0 offen offset:1024
	buffer_load_dword v209, v229, s[24:27], 0 offen            
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
	buffer_load_dword v224, s[20:23], 0 offen lds              
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v225, s[20:23], 0 offen lds              
	s_add_u32 m0, 0x10800, s64                                 
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x11880, s64                                 
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x12900, s64                                 
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x13980, s64                                 
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x14a00, s64                                 
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x15a80, s64                                 
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x16b00, s64                                 
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	s_add_u32 m0, 0x17b80, s64                                 
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	buffer_load_dwordx4 v[168:171], v227, s[16:19], 0 offen    
	buffer_load_dwordx4 v[172:175], v228, s[16:19], 0 offen    
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen offset:1024
	buffer_load_dword v210, v229, s[24:27], 0 offen            
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
	s_waitcnt vmcnt(41)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v220                                 
	ds_read_b128 v[16:19], v220 offset:64                      
	ds_read_b128 v[12:15], v220 offset:512                     
	ds_read_b128 v[20:23], v220 offset:576                     
	ds_read_b32 v200, v226                                     
	ds_read_b128 v[24:27], v220 offset:4224                    
	ds_read_b128 v[32:35], v220 offset:4288                    
	ds_read_b128 v[28:31], v220 offset:4736                    
	ds_read_b128 v[36:39], v220 offset:4800                    
	ds_read_b32 v201, v226 offset:256                          
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_lshl_b32 s36, s36, 1                                     
	s_mul_i32 s62, s48, 0x100                                  
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
	s_cmp_lt_u32 s62, 0x100                                    
	s_cselect_b32 s62, s62, 0x100                              
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
	v_mul_lo_u32 v230, s36, v5                                 
	v_add_u32_e32 v230, s62, v230                              
	v_add_u32_e32 v230, v4, v230                               
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_09AD                                  
	
label_031A:
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v220 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1800, s65                                  
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v220 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v220 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1c00, s65                                  
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v220 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v226 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v220 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x18c00, s64                                 
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[24:27], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v220 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[28:31], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[32:35], a[16:19], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v220 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[36:39], a[20:23], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x19c80, s64                                 
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[32:35], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v220 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[36:39], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v226 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[40:43], a[32:35], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v220 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[44:47], a[36:39], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ad00, s64                                 
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[40:43], a[40:43], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v220 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[44:47], a[44:47], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[48:51], a[32:35], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v220 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[52:55], a[36:39], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1bd80, s64                                 
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[48:51], a[40:43], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v220 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[52:55], a[44:47], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v226 offset:1024                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[56:59], a[48:51], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v220 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[60:63], a[52:55], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ce00, s64                                 
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[56:59], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v220 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[60:63], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[64:67], a[48:51], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v220 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[68:71], a[52:55], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1de80, s64                                 
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v220 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v226 offset:1280                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[72:75], a[64:67], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v220 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[76:79], a[68:71], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ef00, s64                                 
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[72:75], a[72:75], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v220 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[76:79], a[76:79], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[80:83], a[64:67], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v220 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[84:87], a[68:71], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ff80, s64                                 
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[80:83], a[72:75], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[116:119], v220 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[84:87], a[76:79], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v206, v226 offset:1536                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[88:91], a[80:83], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[120:123], v220 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[92:95], a[84:87], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[184:187], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[88:91], a[88:91], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[128:131], v220 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[92:95], a[92:95], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[96:99], a[80:83], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[124:127], v220 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[100:103], a[84:87], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[188:191], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[96:99], a[88:91], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[132:135], v220 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[100:103], a[92:95], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v207, v226 offset:1792                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[104:107], a[96:99], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v221                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[108:111], a[100:103], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[104:107], a[104:107], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v221 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[108:111], a[108:111], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[112:115], a[96:99], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v221 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[116:119], a[100:103], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[112:115], a[104:107], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v221 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[116:119], a[108:111], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v226 offset:2048                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[120:123], a[112:115], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v221 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[124:127], a[116:119], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[120:123], a[120:123], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v221 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[124:127], a[124:127], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[128:131], a[112:115], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v221 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[132:135], a[116:119], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[128:131], a[120:123], v208, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v221 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[132:135], a[124:127], v208, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v201, v226 offset:2304                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[8:11], a[0:3], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v221 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[12:15], a[4:7], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[8:11], a[8:11], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v221 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[12:15], a[12:15], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[16:19], a[0:3], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v221 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[20:23], a[4:7], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[16:19], a[8:11], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v221 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[164:167], v[20:23], a[12:15], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v226 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[24:27], a[16:19], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v221 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[28:31], a[20:23], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[24:27], a[24:27], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v221 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[28:31], a[28:31], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:163], v[32:35], a[16:19], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v221 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[160:163], v[36:39], a[20:23], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[164:167], v[32:35], a[24:27], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v221 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[36:39], a[28:31], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v226 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[152:155], v[40:43], a[32:35], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v221 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[44:47], a[36:39], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[40:43], a[40:43], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v221 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[44:47], a[44:47], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[160:163], v[48:51], a[32:35], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v221 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[52:55], a[36:39], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[48:51], a[40:43], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v221 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[52:55], a[44:47], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v226 offset:3072                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[56:59], a[48:51], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v221 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[60:63], a[52:55], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[56:59], a[56:59], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v221 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[60:63], a[60:63], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[64:67], a[48:51], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v221 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[68:71], a[52:55], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[64:67], a[56:59], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v221 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[68:71], a[60:63], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v226 offset:3328                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[72:75], a[64:67], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v221 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[152:155], v[76:79], a[68:71], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:159], v[72:75], a[72:75], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v221 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[76:79], a[76:79], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[80:83], a[64:67], v209, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v221 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[84:87], a[68:71], v209, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[164:167], v[80:83], a[72:75], v209, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[116:119], v221 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[84:87], a[76:79], v209, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v206, v226 offset:3584                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[88:91], a[80:83], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[120:123], v221 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[92:95], a[84:87], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[136:139], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[88:91], a[88:91], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[128:131], v221 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[92:95], a[92:95], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[96:99], a[80:83], v209, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[124:127], v221 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[100:103], a[84:87], v209, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[140:143], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[96:99], a[88:91], v209, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[132:135], v221 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[100:103], a[92:95], v209, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v207, v226 offset:3840                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[152:155], v[104:107], a[96:99], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v222                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[152:155], v[108:111], a[100:103], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[104:107], a[104:107], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v222 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:159], v[108:111], a[108:111], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:163], v[112:115], a[96:99], v209, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v222 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[160:163], v[116:119], a[100:103], v209, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[112:115], a[104:107], v209, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v222 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[116:119], a[108:111], v209, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v226 offset:4096                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:155], v[120:123], a[112:115], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v222 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:155], v[124:127], a[116:119], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:159], v[120:123], a[120:123], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v222 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[124:127], a[124:127], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:163], v[128:131], a[112:115], v209, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v222 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[160:163], v[132:135], a[116:119], v209, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[128:131], a[120:123], v209, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v222 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[132:135], a[124:127], v209, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v201, v226 offset:4352                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[8:11], a[0:3], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v222 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[12:15], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:175], v[8:11], a[8:11], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v222 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[172:175], v[12:15], a[12:15], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[16:19], a[0:3], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v222 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[20:23], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[180:183], v[16:19], a[8:11], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v222 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[180:183], v[20:23], a[12:15], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v226 offset:4608                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[168:171], v[24:27], a[16:19], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v222 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:171], v[28:31], a[20:23], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[172:175], v[24:27], a[24:27], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v222 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[172:175], v[28:31], a[28:31], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[176:179], v[32:35], a[16:19], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v222 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[176:179], v[36:39], a[20:23], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[180:183], v[32:35], a[24:27], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v222 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[180:183], v[36:39], a[28:31], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v226 offset:4864                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[168:171], v[40:43], a[32:35], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v222 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[44:47], a[36:39], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[40:43], a[40:43], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v222 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[44:47], a[44:47], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[176:179], v[48:51], a[32:35], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v222 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[176:179], v[52:55], a[36:39], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[180:183], v[48:51], a[40:43], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v222 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[180:183], v[52:55], a[44:47], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v226 offset:5120                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[168:171], v[56:59], a[48:51], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v222 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[60:63], a[52:55], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:175], v[56:59], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v222 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:175], v[60:63], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[176:179], v[64:67], a[48:51], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v222 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[176:179], v[68:71], a[52:55], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[180:183], v[64:67], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v222 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[180:183], v[68:71], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v226 offset:5376                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[168:171], v[72:75], a[64:67], v210, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v222 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[168:171], v[76:79], a[68:71], v210, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[172:175], v[72:75], a[72:75], v210, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v222 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[172:175], v[76:79], a[76:79], v210, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[80:83], a[64:67], v210, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v222 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[176:179], v[84:87], a[68:71], v210, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[180:183], v[80:83], a[72:75], v210, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[116:119], v222 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[180:183], v[84:87], a[76:79], v210, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v206, v226 offset:5632                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[168:171], v[88:91], a[80:83], v210, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[120:123], v222 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[168:171], v[92:95], a[84:87], v210, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[152:155], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[172:175], v[88:91], a[88:91], v210, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[128:131], v222 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[172:175], v[92:95], a[92:95], v210, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[96:99], a[80:83], v210, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[124:127], v222 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[176:179], v[100:103], a[84:87], v210, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[156:159], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[180:183], v[96:99], a[88:91], v210, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[132:135], v222 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[180:183], v[100:103], a[92:95], v210, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v207, v226 offset:5888                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[104:107], a[96:99], v210, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v223                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[108:111], a[100:103], v210, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[104:107], a[104:107], v210, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v223 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[108:111], a[108:111], v210, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[112:115], a[96:99], v210, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v223 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[116:119], a[100:103], v210, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[112:115], a[104:107], v210, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v223 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[116:119], a[108:111], v210, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v226 offset:6144                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[168:171], v[120:123], a[112:115], v210, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v223 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:171], v[124:127], a[116:119], v210, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[172:175], v[120:123], a[120:123], v210, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v223 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[172:175], v[124:127], a[124:127], v210, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[176:179], v[128:131], a[112:115], v210, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v223 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[176:179], v[132:135], a[116:119], v210, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[128:131], a[120:123], v210, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v223 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[132:135], a[124:127], v210, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v201, v226 offset:6400                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[184:187], v[8:11], a[0:3], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v223 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[12:15], a[4:7], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1000, s65                                  
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[188:191], v[8:11], a[8:11], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v223 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[188:191], v[12:15], a[12:15], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:195], v[16:19], a[0:3], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v223 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[192:195], v[20:23], a[4:7], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[196:199], v[16:19], a[8:11], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v223 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[196:199], v[20:23], a[12:15], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v226 offset:6656                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[184:187], v[24:27], a[16:19], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v223 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[28:31], a[20:23], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x10800, s64                                 
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[188:191], v[24:27], a[24:27], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v223 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[188:191], v[28:31], a[28:31], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[192:195], v[32:35], a[16:19], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v223 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[192:195], v[36:39], a[20:23], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x11880, s64                                 
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[196:199], v[32:35], a[24:27], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v223 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[196:199], v[36:39], a[28:31], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v226 offset:6912                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[184:187], v[40:43], a[32:35], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v223 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[44:47], a[36:39], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x12900, s64                                 
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[188:191], v[40:43], a[40:43], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v223 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[44:47], a[44:47], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:195], v[48:51], a[32:35], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v223 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[192:195], v[52:55], a[36:39], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x13980, s64                                 
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[196:199], v[48:51], a[40:43], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v223 offset:17472                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[196:199], v[52:55], a[44:47], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v204, v226 offset:7168                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[184:187], v[56:59], a[48:51], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v223 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[184:187], v[60:63], a[52:55], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x14a00, s64                                 
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[56:59], a[56:59], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v223 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[188:191], v[60:63], a[60:63], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[192:195], v[64:67], a[48:51], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v223 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[192:195], v[68:71], a[52:55], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x15a80, s64                                 
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[196:199], v[64:67], a[56:59], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v223 offset:21696                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[196:199], v[68:71], a[60:63], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v205, v226 offset:7424                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[184:187], v[72:75], a[64:67], v211, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v223 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[76:79], a[68:71], v211, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x16b00, s64                                 
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[188:191], v[72:75], a[72:75], v211, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v223 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[188:191], v[76:79], a[76:79], v211, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:195], v[80:83], a[64:67], v211, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v223 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[84:87], a[68:71], v211, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x17b80, s64                                 
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[196:199], v[80:83], a[72:75], v211, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[116:119], v223 offset:25920                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[196:199], v[84:87], a[76:79], v211, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b32 v206, v226 offset:7680                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[184:187], v[88:91], a[80:83], v211, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[120:123], v223 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[184:187], v[92:95], a[84:87], v211, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[168:171], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[88:91], a[88:91], v211, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[128:131], v223 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[92:95], a[92:95], v211, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[192:195], v[96:99], a[80:83], v211, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[124:127], v223 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[192:195], v[100:103], a[84:87], v211, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[172:175], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[196:199], v[96:99], a[88:91], v211, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[132:135], v223 offset:30144                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[196:199], v[100:103], a[92:95], v211, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	ds_read_b32 v207, v226 offset:7936                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[184:187], v[104:107], a[96:99], v211, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[8:11], v220                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[184:187], v[108:111], a[100:103], v211, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[188:191], v[104:107], a[104:107], v211, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v220 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[188:191], v[108:111], a[108:111], v211, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:195], v[112:115], a[96:99], v211, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v220 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[192:195], v[116:119], a[100:103], v211, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[196:199], v[112:115], a[104:107], v211, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v220 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[196:199], v[116:119], a[108:111], v211, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v226                                     
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[120:123], a[112:115], v211, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v220 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[124:127], a[116:119], v211, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[120:123], a[120:123], v211, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[32:35], v220 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[124:127], a[124:127], v211, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[128:131], a[112:115], v211, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[28:31], v220 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[132:135], a[116:119], v211, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[128:131], a[120:123], v211, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[36:39], v220 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[132:135], a[124:127], v211, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	ds_read_b32 v201, v226 offset:256                          
	s_cbranch_scc0 label_1040                                  
	s_branch label_031A                                        
	
label_09AD:
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1800, s65                                  
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v220 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[8:11], a[8:11], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[12:15], a[12:15], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v220 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[16:19], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1c00, s65                                  
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[20:23], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v220 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[16:19], a[8:11], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[20:23], a[12:15], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v220 offset:9024                    
	ds_read_b32 v202, v226 offset:512                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x18c00, s64                                 
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v220 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[24:27], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[28:31], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v220 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[32:35], a[16:19], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x19c80, s64                                 
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[36:39], a[20:23], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v220 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[32:35], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[36:39], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v220 offset:13248                   
	ds_read_b32 v203, v226 offset:768                          
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[40:43], a[32:35], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ad00, s64                                 
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[44:47], a[36:39], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v220 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[40:43], a[40:43], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[44:47], a[44:47], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v220 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[48:51], a[32:35], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1bd80, s64                                 
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[52:55], a[36:39], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v220 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[48:51], a[40:43], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[52:55], a[44:47], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v220 offset:17472                   
	ds_read_b32 v204, v226 offset:1024                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[56:59], a[48:51], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ce00, s64                                 
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[60:63], a[52:55], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v220 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[56:59], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[60:63], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v220 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[64:67], a[48:51], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1de80, s64                                 
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[68:71], a[52:55], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v220 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v220 offset:21696                 
	ds_read_b32 v205, v226 offset:1280                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[72:75], a[64:67], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ef00, s64                                 
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[76:79], a[68:71], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v220 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[72:75], a[72:75], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[76:79], a[76:79], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v220 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[80:83], a[64:67], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1ff80, s64                                 
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[84:87], a[68:71], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[108:111], v220 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[148:151], v[80:83], a[72:75], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[84:87], a[76:79], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[116:119], v220 offset:25920                 
	ds_read_b32 v206, v226 offset:1536                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[88:91], a[80:83], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[184:187], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[92:95], a[84:87], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[120:123], v220 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[88:91], a[88:91], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[92:95], a[92:95], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[128:131], v220 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[96:99], a[80:83], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[188:191], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[100:103], a[84:87], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[124:127], v220 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[96:99], a[88:91], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[100:103], a[92:95], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[132:135], v220 offset:30144                 
	ds_read_b32 v207, v226 offset:1792                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[104:107], a[96:99], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[108:111], a[100:103], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v221                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[104:107], a[104:107], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[108:111], a[108:111], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v221 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[112:115], a[96:99], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[116:119], a[100:103], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v221 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[112:115], a[104:107], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[116:119], a[108:111], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v221 offset:576                     
	ds_read_b32 v200, v226 offset:2048                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[120:123], a[112:115], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[124:127], a[116:119], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v221 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[120:123], a[120:123], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[124:127], a[124:127], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v221 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[128:131], a[112:115], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[132:135], a[116:119], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v221 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[128:131], a[120:123], v208, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[132:135], a[124:127], v208, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v221 offset:4800                    
	ds_read_b32 v201, v226 offset:2304                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[8:11], a[0:3], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[12:15], a[4:7], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v221 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[8:11], a[8:11], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[12:15], a[12:15], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v221 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[16:19], a[0:3], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[20:23], a[4:7], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v221 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[16:19], a[8:11], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[164:167], v[20:23], a[12:15], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v221 offset:9024                    
	ds_read_b32 v202, v226 offset:2560                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[24:27], a[16:19], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[28:31], a[20:23], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v221 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[24:27], a[24:27], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[28:31], a[28:31], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v221 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[160:163], v[32:35], a[16:19], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[160:163], v[36:39], a[20:23], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v221 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[164:167], v[32:35], a[24:27], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[36:39], a[28:31], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v221 offset:13248                   
	ds_read_b32 v203, v226 offset:2816                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[152:155], v[40:43], a[32:35], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[44:47], a[36:39], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v221 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[40:43], a[40:43], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[44:47], a[44:47], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v221 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[160:163], v[48:51], a[32:35], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[52:55], a[36:39], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v221 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[48:51], a[40:43], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[52:55], a[44:47], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v221 offset:17472                   
	ds_read_b32 v204, v226 offset:3072                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[56:59], a[48:51], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[60:63], a[52:55], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v221 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[56:59], a[56:59], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[60:63], a[60:63], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v221 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[64:67], a[48:51], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[68:71], a[52:55], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v221 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[64:67], a[56:59], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[68:71], a[60:63], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v221 offset:21696                 
	ds_read_b32 v205, v226 offset:3328                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[72:75], a[64:67], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[152:155], v[76:79], a[68:71], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v221 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[156:159], v[72:75], a[72:75], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[76:79], a[76:79], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v221 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[80:83], a[64:67], v209, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[84:87], a[68:71], v209, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[108:111], v221 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[164:167], v[80:83], a[72:75], v209, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[84:87], a[76:79], v209, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[116:119], v221 offset:25920                 
	ds_read_b32 v206, v226 offset:3584                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[152:155], v[88:91], a[80:83], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[136:139], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[92:95], a[84:87], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[120:123], v221 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[88:91], a[88:91], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[92:95], a[92:95], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[128:131], v221 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[96:99], a[80:83], v209, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[140:143], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[100:103], a[84:87], v209, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[124:127], v221 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[164:167], v[96:99], a[88:91], v209, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[100:103], a[92:95], v209, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[132:135], v221 offset:30144                 
	ds_read_b32 v207, v226 offset:3840                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[152:155], v[104:107], a[96:99], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[152:155], v[108:111], a[100:103], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v222                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[104:107], a[104:107], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[156:159], v[108:111], a[108:111], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v222 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:163], v[112:115], a[96:99], v209, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[160:163], v[116:119], a[100:103], v209, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v222 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[112:115], a[104:107], v209, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[116:119], a[108:111], v209, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v222 offset:576                     
	ds_read_b32 v200, v226 offset:4096                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:155], v[120:123], a[112:115], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:155], v[124:127], a[116:119], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v222 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[156:159], v[120:123], a[120:123], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[124:127], a[124:127], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v222 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:163], v[128:131], a[112:115], v209, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[160:163], v[132:135], a[116:119], v209, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v222 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[128:131], a[120:123], v209, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[132:135], a[124:127], v209, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v222 offset:4800                    
	ds_read_b32 v201, v226 offset:4352                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[8:11], a[0:3], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[12:15], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v222 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:175], v[8:11], a[8:11], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[172:175], v[12:15], a[12:15], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v222 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[16:19], a[0:3], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[20:23], a[4:7], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v222 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[180:183], v[16:19], a[8:11], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[180:183], v[20:23], a[12:15], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v222 offset:9024                    
	ds_read_b32 v202, v226 offset:4608                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[168:171], v[24:27], a[16:19], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x8400, s64                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[168:171], v[28:31], a[20:23], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v222 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[172:175], v[24:27], a[24:27], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[172:175], v[28:31], a[28:31], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v222 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[176:179], v[32:35], a[16:19], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x9480, s64                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[176:179], v[36:39], a[20:23], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v222 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[180:183], v[32:35], a[24:27], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[180:183], v[36:39], a[28:31], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v222 offset:13248                   
	ds_read_b32 v203, v226 offset:4864                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[168:171], v[40:43], a[32:35], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xa500, s64                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[44:47], a[36:39], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v222 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[40:43], a[40:43], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[44:47], a[44:47], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v222 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[176:179], v[48:51], a[32:35], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xb580, s64                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[176:179], v[52:55], a[36:39], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v222 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[180:183], v[48:51], a[40:43], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[180:183], v[52:55], a[44:47], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v222 offset:17472                   
	ds_read_b32 v204, v226 offset:5120                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[168:171], v[56:59], a[48:51], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc600, s64                                  
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[60:63], a[52:55], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v222 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:175], v[56:59], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:175], v[60:63], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v222 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[176:179], v[64:67], a[48:51], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xd680, s64                                  
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[176:179], v[68:71], a[52:55], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v222 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[180:183], v[64:67], a[56:59], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[180:183], v[68:71], a[60:63], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v222 offset:21696                 
	ds_read_b32 v205, v226 offset:5376                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[168:171], v[72:75], a[64:67], v210, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xe700, s64                                  
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[168:171], v[76:79], a[68:71], v210, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v222 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[172:175], v[72:75], a[72:75], v210, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[172:175], v[76:79], a[76:79], v210, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v222 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[80:83], a[64:67], v210, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xf780, s64                                  
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[176:179], v[84:87], a[68:71], v210, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[108:111], v222 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[180:183], v[80:83], a[72:75], v210, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[180:183], v[84:87], a[76:79], v210, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[116:119], v222 offset:25920                 
	ds_read_b32 v206, v226 offset:5632                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[168:171], v[88:91], a[80:83], v210, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[152:155], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[168:171], v[92:95], a[84:87], v210, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[120:123], v222 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[172:175], v[88:91], a[88:91], v210, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[172:175], v[92:95], a[92:95], v210, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[128:131], v222 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[96:99], a[80:83], v210, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[156:159], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[176:179], v[100:103], a[84:87], v210, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[124:127], v222 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[180:183], v[96:99], a[88:91], v210, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[180:183], v[100:103], a[92:95], v210, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[132:135], v222 offset:30144                 
	ds_read_b32 v207, v226 offset:5888                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[168:171], v[104:107], a[96:99], v210, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[168:171], v[108:111], a[100:103], v210, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v223                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[104:107], a[104:107], v210, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[172:175], v[108:111], a[108:111], v210, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v223 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[112:115], a[96:99], v210, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[116:119], a[100:103], v210, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v223 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:183], v[112:115], a[104:107], v210, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:183], v[116:119], a[108:111], v210, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v223 offset:576                     
	ds_read_b32 v200, v226 offset:6144                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[168:171], v[120:123], a[112:115], v210, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:171], v[124:127], a[116:119], v210, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v223 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[172:175], v[120:123], a[120:123], v210, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[172:175], v[124:127], a[124:127], v210, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v223 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[176:179], v[128:131], a[112:115], v210, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[176:179], v[132:135], a[116:119], v210, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v223 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[128:131], a[120:123], v210, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[132:135], a[124:127], v210, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v223 offset:4800                    
	ds_read_b32 v201, v226 offset:6400                         
	s_cbranch_scc0 label_1040                                  
	s_waitcnt vmcnt(30) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[184:187], v[8:11], a[0:3], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1000, s65                                  
	buffer_load_dword v224, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[12:15], a[4:7], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v223 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[188:191], v[8:11], a[8:11], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[188:191], v[12:15], a[12:15], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v223 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[192:195], v[16:19], a[0:3], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1400, s65                                  
	buffer_load_dword v225, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[192:195], v[20:23], a[4:7], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v223 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[196:199], v[16:19], a[8:11], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[196:199], v[20:23], a[12:15], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v223 offset:9024                    
	ds_read_b32 v202, v226 offset:6656                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[184:187], v[24:27], a[16:19], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x10800, s64                                 
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[28:31], a[20:23], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v223 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[188:191], v[24:27], a[24:27], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[188:191], v[28:31], a[28:31], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v223 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[192:195], v[32:35], a[16:19], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x11880, s64                                 
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[192:195], v[36:39], a[20:23], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v223 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[196:199], v[32:35], a[24:27], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[196:199], v[36:39], a[28:31], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v223 offset:13248                   
	ds_read_b32 v203, v226 offset:6912                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[184:187], v[40:43], a[32:35], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x12900, s64                                 
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[44:47], a[36:39], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[72:75], v223 offset:16896                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[188:191], v[40:43], a[40:43], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[44:47], a[44:47], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[80:83], v223 offset:16960                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[192:195], v[48:51], a[32:35], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x13980, s64                                 
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[192:195], v[52:55], a[36:39], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[76:79], v223 offset:17408                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[196:199], v[48:51], a[40:43], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[196:199], v[52:55], a[44:47], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[84:87], v223 offset:17472                   
	ds_read_b32 v204, v226 offset:7168                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[184:187], v[56:59], a[48:51], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x14a00, s64                                 
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[184:187], v[60:63], a[52:55], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[88:91], v223 offset:21120                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[56:59], a[56:59], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[188:191], v[60:63], a[60:63], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[96:99], v223 offset:21184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[192:195], v[64:67], a[48:51], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x15a80, s64                                 
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[192:195], v[68:71], a[52:55], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[92:95], v223 offset:21632                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[196:199], v[64:67], a[56:59], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[196:199], v[68:71], a[60:63], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[100:103], v223 offset:21696                 
	ds_read_b32 v205, v226 offset:7424                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[184:187], v[72:75], a[64:67], v211, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x16b00, s64                                 
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[76:79], a[68:71], v211, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v223 offset:25344                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[188:191], v[72:75], a[72:75], v211, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[188:191], v[76:79], a[76:79], v211, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v223 offset:25408                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[192:195], v[80:83], a[64:67], v211, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x17b80, s64                                 
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[84:87], a[68:71], v211, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x400, s60                                  
	ds_read_b128 v[108:111], v223 offset:25856                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[196:199], v[80:83], a[72:75], v211, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s66, s66, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[196:199], v[84:87], a[76:79], v211, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	ds_read_b128 v[116:119], v223 offset:25920                 
	ds_read_b32 v206, v226 offset:7680                         
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[184:187], v[88:91], a[80:83], v211, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	buffer_load_dwordx4 v[168:171], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[184:187], v[92:95], a[84:87], v211, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	ds_read_b128 v[120:123], v223 offset:29568                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[88:91], a[88:91], v211, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[92:95], a[92:95], v211, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	ds_read_b128 v[128:131], v223 offset:29632                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[192:195], v[96:99], a[80:83], v211, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dwordx4 v[172:175], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[192:195], v[100:103], a[84:87], v211, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x400, s60                                  
	ds_read_b128 v[124:127], v223 offset:30080                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[196:199], v[96:99], a[88:91], v211, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[196:199], v[100:103], a[92:95], v211, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	ds_read_b128 v[132:135], v223 offset:30144                 
	ds_read_b32 v207, v226 offset:7936                         
	s_waitcnt vmcnt(38) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[184:187], v[104:107], a[96:99], v211, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[184:187], v[108:111], a[100:103], v211, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v220                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[188:191], v[104:107], a[104:107], v211, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[188:191], v[108:111], a[108:111], v211, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v220 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:195], v[112:115], a[96:99], v211, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[192:195], v[116:119], a[100:103], v211, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v220 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[196:199], v[112:115], a[104:107], v211, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[196:199], v[116:119], a[108:111], v211, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v220 offset:576                     
	ds_read_b32 v200, v226                                     
	s_waitcnt lgkmcnt(5)                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[184:187], v[120:123], a[112:115], v211, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v229, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[124:127], a[116:119], v211, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s67                                    
	ds_read_b128 v[24:27], v220 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[120:123], a[120:123], v211, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[188:191], v[124:127], a[124:127], v211, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s69                                    
	ds_read_b128 v[32:35], v220 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[192:195], v[128:131], a[112:115], v211, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[192:195], v[132:135], a[116:119], v211, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	ds_read_b128 v[28:31], v220 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[128:131], a[120:123], v211, v207 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[132:135], a[124:127], v211, v207 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v220 offset:4800                    
	ds_read_b32 v201, v226 offset:256                          
	s_cbranch_scc0 label_1040                                  
	s_branch label_09AD                                        
	
label_1040:
	s_waitcnt lgkmcnt(0)                                       
	s_mul_i32 s62, s47, 0x80                                   
	s_mul_i32 s63, s46, 32                                     
	s_add_u32 s60, s62, s63                                    
	s_add_u32 s62, s60, 32                                     
	s_cmp_lt_i32 s44, s62                                      
	s_cbranch_scc1 label_124B                                  
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v234, 0, v230                                
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
	s_branch label_1450                                        
	
label_124B:
	s_mul_i32 s62, s36, 16                                     
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_1450                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v234, 0, v230                                
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_u32_e32 v234, s62, v234                              
	
label_1450:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

