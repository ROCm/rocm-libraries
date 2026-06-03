// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA:
	
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
	s_add_u32 s51, s43, 31                                     
	s_lshr_b32 s62, s51, 5                                     
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
	s_mul_i32 s62, s48, 32                                     
	s_mul_hi_u32 s63, s37, s62                                 
	s_add_u32 s13, s13, s63                                    
	s_mul_i32 s63, s37, s62                                    
	s_add_u32 s12, s12, s63                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_i32 s63, s43, s62                                    
	s_cmp_lt_u32 s63, 32                                       
	s_cselect_b32 s62, s63, 32                                 
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
	v_mul_lo_u32 v144, s37, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v144, v4, v144                               
	s_lshr_b32 s62, s46, 1                                     
	s_mul_i32 s62, s62, 8                                      
	s_and_b32 s63, s46, 1                                      
	s_mul_i32 s63, s63, 2                                      
	s_add_u32 s62, s62, s63                                    
	s_mul_i32 s62, s37, s62                                    
	v_add_u32_e32 v144, s62, v144                              
	s_mul_i32 s64, 0x420, s46                                  
	s_add_u32 s64, 0x1000, s64                                 
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v145, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v145, v5, v145                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v145, v6, v145                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v145, v4, v145                               
	v_add_u32_e32 v145, 0x1000, v145                           
	v_add_u32_e32 v146, 0x1080, v145                           
	v_add_u32_e32 v147, 0x1080, v146                           
	v_add_u32_e32 v148, 0x1080, v147                           
	s_mul_i32 s62, s48, 32                                     
	s_mul_hi_u32 s63, s39, s62                                 
	s_add_u32 s21, s21, s63                                    
	s_mul_i32 s63, s39, s62                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_add_u32 s63, s43, 31                                     
	s_lshr_b32 s63, s63, 5                                     
	s_lshl_b32 s63, s63, 5                                     
	s_sub_i32 s63, s63, s62                                    
	s_cmp_lt_u32 s63, 32                                       
	s_cselect_b32 s62, s63, 32                                 
	s_mul_i32 s22, s39, s62                                    
	s_mov_b32 s23, 0x20000                                     
	v_lshlrev_b32_e32 v149, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v149, s63, v149                              
	s_mul_i32 s65, s46, 0x100                                  
	s_add_i32 s65, s65, 0                                      
	v_lshlrev_b32_e32 v150, 2, v0                              
	v_add_u32_e32 v150, 0, v150                                
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
	v_lshlrev_b32_e32 v151, 4, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v151, s62, v151                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v152, s62, v151                              
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
	v_lshlrev_b32_e32 v153, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v153, s63, v153                              
	s_mov_b32 s66, 0x80                                        
	s_mov_b32 s67, 0x800                                       
	s_mov_b32 s68, 0x100                                       
	s_mov_b32 s69, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
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
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	buffer_load_dwordx4 v[72:75], v151, s[16:19], 0 offen      
	buffer_load_dwordx4 v[76:79], v152, s[16:19], 0 offen      
	buffer_load_dwordx4 v[80:83], v151, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[84:87], v152, s[16:19], 0 offen offset:1024
	buffer_load_dword v140, v153, s[24:27], 0 offen            
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
	buffer_load_dword v149, s[20:23], 0 offen lds              
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	buffer_load_dwordx4 v[88:91], v151, s[16:19], 0 offen      
	buffer_load_dwordx4 v[92:95], v152, s[16:19], 0 offen      
	buffer_load_dwordx4 v[96:99], v151, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[100:103], v152, s[16:19], 0 offen offset:1024
	buffer_load_dword v141, v153, s[24:27], 0 offen            
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
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	buffer_load_dwordx4 v[104:107], v151, s[16:19], 0 offen    
	buffer_load_dwordx4 v[108:111], v152, s[16:19], 0 offen    
	buffer_load_dwordx4 v[112:115], v151, s[16:19], 0 offen offset:1024
	buffer_load_dwordx4 v[116:119], v152, s[16:19], 0 offen offset:1024
	buffer_load_dword v142, v153, s[24:27], 0 offen            
	s_add_u32 s62, 0x400, s60                                  
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
	s_waitcnt vmcnt(19)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v145                                 
	ds_read_b128 v[16:19], v145 offset:64                      
	ds_read_b128 v[12:15], v145 offset:512                     
	ds_read_b128 v[20:23], v145 offset:576                     
	ds_read_b32 v136, v150                                     
	ds_read_b128 v[24:27], v146                                
	ds_read_b128 v[32:35], v146 offset:64                      
	ds_read_b128 v[28:31], v146 offset:512                     
	ds_read_b128 v[36:39], v146 offset:576                     
	ds_read_b32 v137, v150 offset:1024                         
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_lshl_b32 s36, s36, 1                                     
	s_mul_i32 s62, s48, 32                                     
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
	s_cmp_lt_u32 s62, 32                                       
	s_cselect_b32 s62, s62, 32                                 
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
	v_mul_lo_u32 v154, s36, v5                                 
	v_add_u32_e32 v154, s62, v154                              
	v_add_u32_e32 v154, v4, v154                               
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_0333                                  
	
label_01D8:
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v140, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v147                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v140, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[8:11], a[8:11], v140, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v147 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[12:15], a[12:15], v140, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[16:19], a[0:3], v140, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[44:47], v147 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[20:23], a[4:7], v140, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[120:123], v151, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[16:19], a[8:11], v140, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[52:55], v147 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[20:23], a[12:15], v140, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[124:127], v152, s[16:19], 0 offen    
	ds_read_b32 v138, v150 offset:2048                         
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[128:131], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[132:135], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v143, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[24:27], a[0:3], v141, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v148                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[28:31], a[4:7], v141, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[24:27], a[8:11], v141, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v148 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[28:31], a[12:15], v141, v137 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[32:35], a[0:3], v141, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[60:63], v148 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[36:39], a[4:7], v141, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[72:75], v151, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[32:35], a[8:11], v141, v137 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[68:71], v148 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[36:39], a[12:15], v141, v137 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[76:79], v152, s[16:19], 0 offen      
	ds_read_b32 v139, v150 offset:3072                         
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[80:83], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[84:87], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v140, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[40:43], a[0:3], v142, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v145                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[44:47], a[4:7], v142, v138 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[40:43], a[8:11], v142, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v145 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[44:47], a[12:15], v142, v138 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[48:51], a[0:3], v142, v138 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[12:15], v145 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[52:55], a[4:7], v142, v138 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[88:91], v151, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[48:51], a[8:11], v142, v138 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[20:23], v145 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[52:55], a[12:15], v142, v138 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[92:95], v152, s[16:19], 0 offen      
	ds_read_b32 v136, v150                                     
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[96:99], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[100:103], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v141, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[56:59], a[0:3], v143, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v146                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[60:63], a[4:7], v143, v139 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[56:59], a[8:11], v143, v139 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v146 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[60:63], a[12:15], v143, v139 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[64:67], a[0:3], v143, v139 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[28:31], v146 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[68:71], a[4:7], v143, v139 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[104:107], v151, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[64:67], a[8:11], v143, v139 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[36:39], v146 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[68:71], a[12:15], v143, v139 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[108:111], v152, s[16:19], 0 offen    
	ds_read_b32 v137, v150 offset:1024                         
	s_add_u32 s12, s12, s66                                    
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[112:115], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[116:119], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v142, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_branch label_01D8                                        
	
label_0333:
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v140, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s65                                       
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v140, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v147                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[8:11], a[8:11], v140, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s64                                       
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[12:15], a[12:15], v140, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[48:51], v147 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[16:19], a[0:3], v140, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[120:123], v151, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:83], v[20:23], a[4:7], v140, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[44:47], v147 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[16:19], a[8:11], v140, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[124:127], v152, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[20:23], a[12:15], v140, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	ds_read_b128 v[52:55], v147 offset:576                     
	ds_read_b32 v138, v150 offset:2048                         
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[128:131], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[132:135], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v143, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[24:27], a[0:3], v141, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[28:31], a[4:7], v141, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v148                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[92:95], v[24:27], a[8:11], v141, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[92:95], v[28:31], a[12:15], v141, v137 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[64:67], v148 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[32:35], a[0:3], v141, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[72:75], v151, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[96:99], v[36:39], a[4:7], v141, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[60:63], v148 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[32:35], a[8:11], v141, v137 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[76:79], v152, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[36:39], a[12:15], v141, v137 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	ds_read_b128 v[68:71], v148 offset:576                     
	ds_read_b32 v139, v150 offset:3072                         
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[80:83], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[84:87], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v140, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[40:43], a[0:3], v142, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x800, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[44:47], a[4:7], v142, v138 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v145                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[40:43], a[8:11], v142, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[44:47], a[12:15], v142, v138 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[16:19], v145 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[48:51], a[0:3], v142, v138 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[88:91], v151, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[52:55], a[4:7], v142, v138 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[12:15], v145 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[48:51], a[8:11], v142, v138 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[92:95], v152, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[52:55], a[12:15], v142, v138 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	ds_read_b128 v[20:23], v145 offset:576                     
	ds_read_b32 v136, v150                                     
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[96:99], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[100:103], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v141, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_waitcnt vmcnt(12) lgkmcnt(5)                             
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[56:59], a[0:3], v143, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0xc00, s65                                   
	buffer_load_dword v149, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[60:63], a[4:7], v143, v139 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v146                                
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[56:59], a[8:11], v143, v139 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s64                                  
	buffer_load_dwordx4 v144, s[12:15], 0 offen nt lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[60:63], a[12:15], v143, v139 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x500, s60                                  
	ds_read_b128 v[32:35], v146 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[64:67], a[0:3], v143, v139 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	buffer_load_dwordx4 v[104:107], v151, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[68:71], a[4:7], v143, v139 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s66, s66, 0                                  
	ds_read_b128 v[28:31], v146 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[64:67], a[8:11], v143, v139 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cselect_b32 s68, s68, 0                                  
	buffer_load_dwordx4 v[108:111], v152, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[132:135], v[68:71], a[12:15], v143, v139 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s66                                    
	ds_read_b128 v[36:39], v146 offset:576                     
	ds_read_b32 v137, v150 offset:1024                         
	s_addc_u32 s13, 0, s13                                     
	buffer_load_dwordx4 v[112:115], v151, s[16:19], 0 offen offset:1024
	s_sub_u32 s14, s14, s66                                    
	s_add_u32 s20, s20, s68                                    
	buffer_load_dwordx4 v[116:119], v152, s[16:19], 0 offen offset:1024
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s68                                    
	buffer_load_dword v142, v153, s[24:27], 0 offen            
	s_add_u32 s63, 0x400, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s67, s67, 0                                  
	s_cselect_b32 s69, s69, 0                                  
	s_add_u32 s16, s16, s67                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s67                                    
	s_add_u32 s24, s24, s69                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s69                                    
	s_addk_i32 s60, 0x100                                      
	s_cmp_lt_i32 s60, s61                                      
	s_cbranch_scc0 label_048E                                  
	s_branch label_0333                                        
	
label_048E:
	s_waitcnt lgkmcnt(0)                                       
	s_mul_i32 s62, s47, 0x80                                   
	s_mul_i32 s63, s46, 32                                     
	s_add_u32 s60, s62, s63                                    
	s_add_u32 s62, s60, 32                                     
	s_cmp_lt_i32 s44, s62                                      
	s_cbranch_scc1 label_04D9                                  
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v158, 0, v154                                
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
	buffer_store_dwordx4 v[16:19], v158, s[4:7], 0 offen       
	v_add_u32_e32 v158, s62, v158                              
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
	buffer_store_dwordx4 v[16:19], v158, s[4:7], 0 offen       
	v_add_u32_e32 v158, s62, v158                              
	s_branch label_051E                                        
	
label_04D9:
	s_mul_i32 s62, s36, 16                                     
	s_cmp_lt_i32 s60, s44                                      
	s_cbranch_scc0 label_051E                                  
	s_addk_i32 s60, 0x20                                       
	v_add_u32_e32 v158, 0, v154                                
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
	buffer_store_dwordx4 v[16:19], v158, s[4:7], 0 offen       
	v_add_u32_e32 v158, s62, v158                              
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
	buffer_store_dwordx4 v[16:19], v158, s[4:7], 0 offen       
	v_add_u32_e32 v158, s62, v158                              
	
label_051E:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128_ntA.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

