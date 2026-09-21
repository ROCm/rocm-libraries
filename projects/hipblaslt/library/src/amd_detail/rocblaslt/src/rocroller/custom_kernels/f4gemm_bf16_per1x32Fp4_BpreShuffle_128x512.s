// Auto-generated from f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512
.p2align 8
.type f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512,@function

f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512:
	
	s_and_b32 s1, s1, 0xffff                                   
	s_mov_b32 s47, s2                                          
	s_mov_b32 s48, s3                                          
	s_mov_b32 s64, s4                                          
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
	s_load_dword s65, s[0:1], 0x170                            
	v_lshrrev_b32_e32 v1, 10, v0                               
	v_lshrrev_b32_e32 v2, 10, v1                               
	v_and_b32_e32 v2, 0x3ff, v2                                
	v_and_b32_e32 v1, 0x3ff, v1                                
	v_and_b32_e32 v0, 0x3ff, v0                                
	v_lshrrev_b32_e32 v3, 6, v0                                
	v_and_b32_e32 v0, 63, v0                                   
	v_readfirstlane_b32 s46, v3                                
	s_waitcnt lgkmcnt(0)                                       
	s_mov_b32 s6, -16                                          
	s_mov_b32 s10, -16                                         
	s_mov_b32 s18, -16                                         
	s_mov_b32 s14, -16                                         
	s_mov_b32 s22, -16                                         
	s_mov_b32 s26, -16                                         
	s_mov_b32 s7, 0x20000                                      
	s_mov_b32 s11, 0x20000                                     
	s_mov_b32 s19, 0x20000                                     
	s_mov_b32 s15, 0x20000                                     
	s_mov_b32 s23, 0x20000                                     
	s_mov_b32 s27, 0x20000                                     
	s_and_b32 s5, s5, 0xffff                                   
	s_and_b32 s9, s9, 0xffff                                   
	s_and_b32 s17, s17, 0xffff                                 
	s_and_b32 s13, s13, 0xffff                                 
	s_and_b32 s21, s21, 0xffff                                 
	s_and_b32 s25, s25, 0xffff                                 
	s_or_b32 s5, s5, 0x40000                                   
	s_or_b32 s9, s9, 0x40000                                   
	s_or_b32 s17, s17, 0x40000                                 
	s_or_b32 s13, s13, 0x40000                                 
	s_or_b32 s21, s21, 0x40000                                 
	s_or_b32 s25, s25, 0x40000                                 
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_0068                                  
	s_lshr_b32 s66, s45, s65                                   
	s_add_u32 s66, s66, 0xff                                   
	s_lshr_b32 s66, s66, 8                                     
	s_lshl_b32 s66, s66, 8                                     
	s_mul_i32 s63, s66, s64                                    
	s_sub_i32 s62, s45, s63                                    
	s_cmp_lt_i32 s62, s66                                      
	s_cselect_b32 s45, s62, s66                                
	
label_0068:
	s_lshr_b32 s37, s37, 1                                     
	s_mul_i32 s62, s48, 0x80                                   
	s_mul_hi_u32 s63, s37, s62                                 
	s_add_u32 s13, s13, s63                                    
	s_mul_i32 s63, s37, s62                                    
	s_add_u32 s12, s12, s63                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_i32 s63, s43, s62                                    
	s_cmp_lt_u32 s63, 0x80                                     
	s_cselect_b32 s62, s63, 0x80                               
	s_mul_i32 s63, s37, s62                                    
	s_mov_b32 s14, s63                                         
	s_mov_b32 s15, 0x20000                                     
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_0080                                  
	s_mul_i32 s63, s66, s64                                    
	s_lshr_b32 s62, s63, 1                                     
	s_add_u32 s12, s12, s62                                    
	s_addc_u32 s13, s13, 0                                     
	s_sub_u32 s14, s14, s62                                    
	
label_0080:
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
	s_mul_i32 s67, 0x420, s46                                  
	s_add_u32 s67, 0x800, s67                                  
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v216, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v216, v5, v216                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v216, v6, v216                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v216, v4, v216                               
	s_mov_b32 s62, 0x800                                       
	v_add_u32_e64 v216, v216, s62                              
	v_add_u32_e32 v217, 0x4200, v216                           
	s_mul_i32 s62, s48, 0x80                                   
	s_mul_hi_u32 s63, s39, s62                                 
	s_add_u32 s21, s21, s63                                    
	s_mul_i32 s63, s39, s62                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_add_u32 s63, s43, 31                                     
	s_lshr_b32 s63, s63, 5                                     
	s_lshl_b32 s63, s63, 5                                     
	s_sub_i32 s63, s63, s62                                    
	s_cmp_lt_u32 s63, 0x80                                     
	s_cselect_b32 s62, s63, 0x80                               
	s_mul_i32 s63, s39, s62                                    
	s_mov_b32 s22, s63                                         
	s_mov_b32 s23, 0x20000                                     
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_00D0                                  
	s_mul_i32 s63, s66, s64                                    
	s_add_u32 s20, s20, s63                                    
	s_addc_u32 s21, s21, 0                                     
	s_sub_u32 s22, s22, s63                                    
	
label_00D0:
	v_lshlrev_b32_e32 v218, 2, v0                              
	s_mul_i32 s63, s46, 32                                     
	s_mul_i32 s63, s63, s39                                    
	v_add_u32_e32 v218, s63, v218                              
	s_mul_i32 s68, s46, 0x100                                  
	s_add_i32 s68, s68, 0                                      
	v_lshlrev_b32_e32 v219, 2, v0                              
	v_add_u32_e32 v219, 0, v219                                
	s_lshr_b32 s38, s38, 1                                     
	s_mul_i32 s62, s47, 0x200                                  
	s_mul_hi_u32 s63, s38, s62                                 
	s_add_u32 s17, s17, s63                                    
	s_mul_i32 s63, s38, s62                                    
	s_add_u32 s16, s16, s63                                    
	s_addc_u32 s17, s17, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x200                                    
	s_cselect_b32 s62, s63, 0x200                              
	s_mul_i32 s63, s38, s62                                    
	s_mov_b32 s18, s63                                         
	s_mov_b32 s19, 0x20000                                     
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_00F2                                  
	s_mul_i32 s63, s66, s64                                    
	s_lshr_b32 s62, s63, 1                                     
	s_mul_i32 s62, s62, 16                                     
	s_add_u32 s16, s16, s62                                    
	s_addc_u32 s17, s17, 0                                     
	s_sub_u32 s18, s18, s62                                    
	
label_00F2:
	v_lshlrev_b32_e32 v220, 4, v0                              
	s_mul_i32 s63, s46, 0x80                                   
	s_mul_i32 s62, s63, s38                                    
	v_add_u32_e32 v220, s62, v220                              
	s_mul_i32 s62, 16, s38                                     
	v_add_u32_e32 v221, s62, v220                              
	v_add_u32_e32 v222, s62, v221                              
	v_add_u32_e32 v223, s62, v222                              
	v_add_u32_e32 v224, 0x400, v220                            
	v_add_u32_e32 v225, 0x400, v221                            
	v_add_u32_e32 v226, 0x400, v222                            
	v_add_u32_e32 v227, 0x400, v223                            
	s_mul_i32 s62, 64, s38                                     
	v_add_u32_e32 v228, s62, v220                              
	v_add_u32_e32 v229, s62, v221                              
	v_add_u32_e32 v230, s62, v222                              
	v_add_u32_e32 v231, s62, v223                              
	v_add_u32_e32 v232, s62, v224                              
	v_add_u32_e32 v233, s62, v225                              
	v_add_u32_e32 v234, s62, v226                              
	v_add_u32_e32 v235, s62, v227                              
	s_mul_i32 s62, s47, 0x200                                  
	s_mul_hi_u32 s63, s40, s62                                 
	s_add_u32 s25, s25, s63                                    
	s_mul_i32 s63, s40, s62                                    
	s_add_u32 s24, s24, s63                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_i32 s63, s44, s62                                    
	s_cmp_lt_u32 s63, 0x200                                    
	s_cselect_b32 s62, s63, 0x200                              
	s_mul_i32 s63, s40, s62                                    
	s_mov_b32 s26, s63                                         
	s_mov_b32 s27, 0x20000                                     
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_0122                                  
	s_mul_i32 s63, s66, s64                                    
	s_add_u32 s24, s24, s63                                    
	s_addc_u32 s25, s25, 0                                     
	s_sub_u32 s26, s26, s63                                    
	
label_0122:
	v_lshlrev_b32_e32 v236, 2, v0                              
	s_mul_i32 s63, s46, 0x80                                   
	s_mul_i32 s63, s63, s40                                    
	v_add_u32_e32 v236, s63, v236                              
	s_mul_i32 s62, 32, s40                                     
	v_add_u32_e32 v237, s62, v236                              
	v_add_u32_e32 v238, s62, v237                              
	v_add_u32_e32 v239, s62, v238                              
	s_mov_b32 s69, 0x80                                        
	s_mov_b32 s70, 0x800                                       
	s_mov_b32 s71, 0x100                                       
	s_mov_b32 s72, 0x100                                       
	s_mov_b32 s60, 0                                           
	s_mov_b32 s61, s45                                         
	s_add_u32 m0, 0, s67                                       
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	s_add_u32 m0, 0x1080, s67                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	s_add_u32 m0, 0, s68                                       
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	s_add_u32 m0, 0x2100, s67                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	s_add_u32 m0, 0x3180, s67                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	s_add_u32 s62, 0x100, s60                                  
	s_cmp_lt_u32 s62, s61                                      
	s_cselect_b32 s69, s69, 0                                  
	s_cselect_b32 s71, s71, 0                                  
	s_add_u32 s12, s12, s69                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s69                                    
	s_add_u32 s20, s20, s71                                    
	s_addc_u32 s21, 0, s21                                     
	s_sub_u32 s22, s22, s71                                    
	buffer_load_dwordx4 v[72:75], v220, s[16:19], 0 offen      
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	buffer_load_dwordx4 v[76:79], v221, s[16:19], 0 offen      
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	buffer_load_dwordx4 v[80:83], v222, s[16:19], 0 offen      
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	buffer_load_dwordx4 v[84:87], v223, s[16:19], 0 offen      
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	buffer_load_dwordx4 v[88:91], v224, s[16:19], 0 offen      
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	buffer_load_dwordx4 v[92:95], v225, s[16:19], 0 offen      
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	buffer_load_dwordx4 v[96:99], v226, s[16:19], 0 offen      
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	buffer_load_dwordx4 v[100:103], v227, s[16:19], 0 offen    
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
	v_accvgpr_write_b32 a102, 0                                
	v_accvgpr_write_b32 a103, 0                                
	buffer_load_dword v204, v236, s[24:27], 0 offen            
	v_accvgpr_write_b32 a104, 0                                
	v_accvgpr_write_b32 a105, 0                                
	v_accvgpr_write_b32 a106, 0                                
	v_accvgpr_write_b32 a107, 0                                
	v_accvgpr_write_b32 a108, 0                                
	v_accvgpr_write_b32 a109, 0                                
	v_accvgpr_write_b32 a110, 0                                
	v_accvgpr_write_b32 a111, 0                                
	buffer_load_dword v205, v237, s[24:27], 0 offen            
	v_accvgpr_write_b32 a112, 0                                
	v_accvgpr_write_b32 a113, 0                                
	v_accvgpr_write_b32 a114, 0                                
	v_accvgpr_write_b32 a115, 0                                
	v_accvgpr_write_b32 a116, 0                                
	v_accvgpr_write_b32 a117, 0                                
	v_accvgpr_write_b32 a118, 0                                
	v_accvgpr_write_b32 a119, 0                                
	buffer_load_dwordx4 v[104:107], v228, s[16:19], 0 offen    
	v_accvgpr_write_b32 a120, 0                                
	v_accvgpr_write_b32 a121, 0                                
	v_accvgpr_write_b32 a122, 0                                
	v_accvgpr_write_b32 a123, 0                                
	v_accvgpr_write_b32 a124, 0                                
	v_accvgpr_write_b32 a125, 0                                
	v_accvgpr_write_b32 a126, 0                                
	v_accvgpr_write_b32 a127, 0                                
	buffer_load_dwordx4 v[108:111], v229, s[16:19], 0 offen    
	v_accvgpr_write_b32 a128, 0                                
	v_accvgpr_write_b32 a129, 0                                
	v_accvgpr_write_b32 a130, 0                                
	v_accvgpr_write_b32 a131, 0                                
	v_accvgpr_write_b32 a132, 0                                
	v_accvgpr_write_b32 a133, 0                                
	v_accvgpr_write_b32 a134, 0                                
	v_accvgpr_write_b32 a135, 0                                
	buffer_load_dwordx4 v[112:115], v230, s[16:19], 0 offen    
	v_accvgpr_write_b32 a136, 0                                
	v_accvgpr_write_b32 a137, 0                                
	v_accvgpr_write_b32 a138, 0                                
	v_accvgpr_write_b32 a139, 0                                
	v_accvgpr_write_b32 a140, 0                                
	v_accvgpr_write_b32 a141, 0                                
	v_accvgpr_write_b32 a142, 0                                
	v_accvgpr_write_b32 a143, 0                                
	buffer_load_dwordx4 v[116:119], v231, s[16:19], 0 offen    
	v_accvgpr_write_b32 a144, 0                                
	v_accvgpr_write_b32 a145, 0                                
	v_accvgpr_write_b32 a146, 0                                
	v_accvgpr_write_b32 a147, 0                                
	v_accvgpr_write_b32 a148, 0                                
	v_accvgpr_write_b32 a149, 0                                
	v_accvgpr_write_b32 a150, 0                                
	v_accvgpr_write_b32 a151, 0                                
	buffer_load_dwordx4 v[120:123], v232, s[16:19], 0 offen    
	v_accvgpr_write_b32 a152, 0                                
	v_accvgpr_write_b32 a153, 0                                
	v_accvgpr_write_b32 a154, 0                                
	v_accvgpr_write_b32 a155, 0                                
	v_accvgpr_write_b32 a156, 0                                
	v_accvgpr_write_b32 a157, 0                                
	v_accvgpr_write_b32 a158, 0                                
	v_accvgpr_write_b32 a159, 0                                
	buffer_load_dwordx4 v[124:127], v233, s[16:19], 0 offen    
	v_accvgpr_write_b32 a160, 0                                
	v_accvgpr_write_b32 a161, 0                                
	v_accvgpr_write_b32 a162, 0                                
	v_accvgpr_write_b32 a163, 0                                
	v_accvgpr_write_b32 a164, 0                                
	v_accvgpr_write_b32 a165, 0                                
	v_accvgpr_write_b32 a166, 0                                
	v_accvgpr_write_b32 a167, 0                                
	buffer_load_dwordx4 v[128:131], v234, s[16:19], 0 offen    
	v_accvgpr_write_b32 a168, 0                                
	v_accvgpr_write_b32 a169, 0                                
	v_accvgpr_write_b32 a170, 0                                
	v_accvgpr_write_b32 a171, 0                                
	v_accvgpr_write_b32 a172, 0                                
	v_accvgpr_write_b32 a173, 0                                
	v_accvgpr_write_b32 a174, 0                                
	v_accvgpr_write_b32 a175, 0                                
	buffer_load_dwordx4 v[132:135], v235, s[16:19], 0 offen    
	v_accvgpr_write_b32 a176, 0                                
	v_accvgpr_write_b32 a177, 0                                
	v_accvgpr_write_b32 a178, 0                                
	v_accvgpr_write_b32 a179, 0                                
	v_accvgpr_write_b32 a180, 0                                
	v_accvgpr_write_b32 a181, 0                                
	v_accvgpr_write_b32 a182, 0                                
	v_accvgpr_write_b32 a183, 0                                
	buffer_load_dword v206, v238, s[24:27], 0 offen            
	v_accvgpr_write_b32 a184, 0                                
	v_accvgpr_write_b32 a185, 0                                
	v_accvgpr_write_b32 a186, 0                                
	v_accvgpr_write_b32 a187, 0                                
	v_accvgpr_write_b32 a188, 0                                
	v_accvgpr_write_b32 a189, 0                                
	v_accvgpr_write_b32 a190, 0                                
	v_accvgpr_write_b32 a191, 0                                
	buffer_load_dword v207, v239, s[24:27], 0 offen            
	v_accvgpr_write_b32 a192, 0                                
	v_accvgpr_write_b32 a193, 0                                
	v_accvgpr_write_b32 a194, 0                                
	v_accvgpr_write_b32 a195, 0                                
	v_accvgpr_write_b32 a196, 0                                
	v_accvgpr_write_b32 a197, 0                                
	v_accvgpr_write_b32 a198, 0                                
	v_accvgpr_write_b32 a199, 0                                
	s_add_u32 s63, 0x100, s60                                  
	s_cmp_lt_u32 s63, s61                                      
	s_cselect_b32 s70, s70, 0                                  
	s_cselect_b32 s72, s72, 0                                  
	s_add_u32 s16, s16, s70                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s70                                    
	s_add_u32 s24, s24, s72                                    
	s_addc_u32 s25, 0, s25                                     
	s_sub_u32 s26, s26, s72                                    
	s_add_u32 m0, 0x4200, s67                                  
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a200, 0                                
	v_accvgpr_write_b32 a201, 0                                
	v_accvgpr_write_b32 a202, 0                                
	v_accvgpr_write_b32 a203, 0                                
	v_accvgpr_write_b32 a204, 0                                
	v_accvgpr_write_b32 a205, 0                                
	v_accvgpr_write_b32 a206, 0                                
	v_accvgpr_write_b32 a207, 0                                
	s_add_u32 m0, 0x5280, s67                                  
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a208, 0                                
	v_accvgpr_write_b32 a209, 0                                
	v_accvgpr_write_b32 a210, 0                                
	v_accvgpr_write_b32 a211, 0                                
	v_accvgpr_write_b32 a212, 0                                
	v_accvgpr_write_b32 a213, 0                                
	v_accvgpr_write_b32 a214, 0                                
	v_accvgpr_write_b32 a215, 0                                
	s_add_u32 m0, 0x400, s68                                   
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_accvgpr_write_b32 a216, 0                                
	v_accvgpr_write_b32 a217, 0                                
	v_accvgpr_write_b32 a218, 0                                
	v_accvgpr_write_b32 a219, 0                                
	v_accvgpr_write_b32 a220, 0                                
	v_accvgpr_write_b32 a221, 0                                
	v_accvgpr_write_b32 a222, 0                                
	v_accvgpr_write_b32 a223, 0                                
	s_add_u32 m0, 0x6300, s67                                  
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a224, 0                                
	v_accvgpr_write_b32 a225, 0                                
	v_accvgpr_write_b32 a226, 0                                
	v_accvgpr_write_b32 a227, 0                                
	v_accvgpr_write_b32 a228, 0                                
	v_accvgpr_write_b32 a229, 0                                
	v_accvgpr_write_b32 a230, 0                                
	v_accvgpr_write_b32 a231, 0                                
	s_add_u32 m0, 0x7380, s67                                  
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
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
	s_waitcnt vmcnt(27)                                        
	s_barrier                                                  
	ds_read_b128 v[8:11], v216                                 
	ds_read_b128 v[24:27], v216 offset:64                      
	ds_read_b128 v[12:15], v216 offset:512                     
	ds_read_b128 v[28:31], v216 offset:576                     
	ds_read_b128 v[16:19], v216 offset:4224                    
	ds_read_b128 v[32:35], v216 offset:4288                    
	ds_read_b128 v[20:23], v216 offset:4736                    
	ds_read_b128 v[36:39], v216 offset:4800                    
	ds_read_b32 v200, v219                                     
	ds_read_b32 v201, v219 offset:256                          
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_lshl_b32 s36, s36, 1                                     
	s_mul_i32 s62, s48, 0x80                                   
	s_mul_hi_u32 s63, s36, s62                                 
	s_add_u32 s5, s5, s63                                      
	s_mul_i32 s63, s36, s62                                    
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_mul_i32 s63, s47, 0x200                                  
	s_lshl_b32 s63, s63, 1                                     
	s_add_u32 s4, s4, s63                                      
	s_addc_u32 s5, s5, 0                                       
	s_sub_i32 s62, s43, s62                                    
	s_cmp_lt_u32 s62, 0x80                                     
	s_cselect_b32 s62, s62, 0x80                               
	s_mul_i32 s62, s36, s62                                    
	s_sub_i32 s62, s62, s63                                    
	s_mov_b32 s6, s62                                          
	s_mov_b32 s7, 0x20000                                      
	s_cmp_gt_i32 s65, 0                                        
	s_cbranch_scc0 label_03D8                                  
	v_mul_i32_i24_e64 v4, v0, 4                                
	s_mul_i32 s62, s46, 0x100                                  
	v_add_u32_e32 v240, s62, v4                                
	v_add_u32_e32 v241, 0x80, v240                             
	s_mul_i32 s62, s36, 64                                     
	v_add_u32_e32 v242, s62, v240                              
	v_add_u32_e32 v243, s62, v241                              
	s_branch label_03EE                                        
	
label_03D8:
	v_and_b32_e64 v4, v0, 15                                   
	v_mul_lo_u32 v240, s36, v4                                 
	v_lshrrev_b32_e32 v4, 5, v0                                
	v_mul_i32_i24_e64 v4, v4, 16                               
	v_lshrrev_b32_e32 v5, 4, v0                                
	v_and_b32_e64 v5, v5, 1                                    
	v_mul_i32_i24_e64 v5, v5, 32                               
	v_add_u32_e32 v4, v4, v5                                   
	v_add_u32_e32 v240, v4, v240                               
	s_mul_i32 s62, s46, 0x100                                  
	v_add_u32_e32 v240, s62, v240                              
	v_add_u32_e32 v241, 0x80, v240                             
	s_mul_i32 s62, s36, 64                                     
	v_add_u32_e32 v242, s62, v240                              
	v_add_u32_e32 v243, s62, v241                              
	
label_03EE:
	s_cmp_lt_i32 s46, 2                                        
	s_cbranch_scc0 label_08F4                                  
	
label_03F0:
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v204, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v220, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[8:11], a[16:19], v204, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v216 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[12:15], a[20:23], v204, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[16:19], a[8:11], v204, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v216 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[20:23], a[12:15], v204, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v221, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[16:19], a[24:27], v204, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v216 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[20:23], a[28:31], v204, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[8:11], a[32:35], v205, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v216 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[12:15], a[36:39], v205, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v222, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[8:11], a[48:51], v205, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v216 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[12:15], a[52:55], v205, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[16:19], a[40:43], v205, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v216 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[20:23], a[44:47], v205, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v223, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[16:19], a[56:59], v205, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v216 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[20:23], a[60:63], v205, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[24:27], a[0:3], v204, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v216 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[28:31], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v224, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[24:27], a[16:19], v204, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v219 offset:512                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[28:31], a[20:23], v204, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[32:35], a[8:11], v204, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v219 offset:768                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[36:39], a[12:15], v204, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v225, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[32:35], a[24:27], v204, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[36:39], a[28:31], v204, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[24:27], a[32:35], v205, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[28:31], a[36:39], v205, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v226, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:103], v[24:27], a[48:51], v205, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:103], v[28:31], a[52:55], v205, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[32:35], a[40:43], v205, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[36:39], a[44:47], v205, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[32:35], a[56:59], v205, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[36:39], a[60:63], v205, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[8:11], a[64:67], v206, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[12:15], a[68:71], v206, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v236, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[8:11], a[80:83], v206, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:111], v[12:15], a[84:87], v206, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[16:19], a[72:75], v206, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[20:23], a[76:79], v206, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v237, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[16:19], a[88:91], v206, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[20:23], a[92:95], v206, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[8:11], a[96:99], v207, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[12:15], a[100:103], v207, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[8:11], a[112:115], v207, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[12:15], a[116:119], v207, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[16:19], a[104:107], v207, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[20:23], a[108:111], v207, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v229, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:119], v[16:19], a[120:123], v207, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:119], v[20:23], a[124:127], v207, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[24:27], a[64:67], v206, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[28:31], a[68:71], v206, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[24:27], a[80:83], v206, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[28:31], a[84:87], v206, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[32:35], a[72:75], v206, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[36:39], a[76:79], v206, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[32:35], a[88:91], v206, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[36:39], a[92:95], v206, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[24:27], a[96:99], v207, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[28:31], a[100:103], v207, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[24:27], a[112:115], v207, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[28:31], a[116:119], v207, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[128:131], v[32:35], a[104:107], v207, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[128:131], v[36:39], a[108:111], v207, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[32:35], a[120:123], v207, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[36:39], a[124:127], v207, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(18) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[40:43], a[128:131], v204, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[44:47], a[132:135], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v234, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[76:79], v[40:43], a[144:147], v204, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v217                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[76:79], v[44:47], a[148:151], v204, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:75], v[48:51], a[136:139], v204, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v217 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:75], v[52:55], a[140:143], v204, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v235, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[76:79], v[48:51], a[152:155], v204, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v217 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[76:79], v[52:55], a[156:159], v204, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[80:83], v[40:43], a[160:163], v205, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v217 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[80:83], v[44:47], a[164:167], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v238, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[84:87], v[40:43], a[176:179], v205, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v217 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[84:87], v[44:47], a[180:183], v205, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[80:83], v[48:51], a[168:171], v205, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v217 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[80:83], v[52:55], a[172:175], v205, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v239, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[84:87], v[48:51], a[184:187], v205, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v217 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[84:87], v[52:55], a[188:191], v205, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s67                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[56:59], a[128:131], v204, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v217 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[60:63], a[132:135], v204, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[56:59], a[144:147], v204, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v219 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[60:63], a[148:151], v204, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:91], v[64:67], a[136:139], v204, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v219 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:91], v[68:71], a[140:143], v204, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[64:67], a[152:155], v204, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[68:71], a[156:159], v204, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s68                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[56:59], a[160:163], v205, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[60:63], a[164:167], v205, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[56:59], a[176:179], v205, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[60:63], a[180:183], v205, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[64:67], a[168:171], v205, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[68:71], a[172:175], v205, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[64:67], a[184:187], v205, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[68:71], a[188:191], v205, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[40:43], a[192:195], v206, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[104:107], v[44:47], a[196:199], v206, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[108:111], v[40:43], a[208:211], v206, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:111], v[44:47], a[212:215], v206, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[48:51], a[200:203], v206, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[104:107], v[52:55], a[204:207], v206, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:111], v[48:51], a[216:219], v206, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:111], v[52:55], a[220:223], v206, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[40:43], a[224:227], v207, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[112:115], v[44:47], a[228:231], v207, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[116:119], v[40:43], a[240:243], v207, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[116:119], v[44:47], a[244:247], v207, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[112:115], v[48:51], a[232:235], v207, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[112:115], v[52:55], a[236:239], v207, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[116:119], v[48:51], a[248:251], v207, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s72, s72, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[116:119], v[52:55], a[252:255], v207, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[56:59], a[192:195], v206, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[120:123], v[60:63], a[196:199], v206, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[56:59], a[208:211], v206, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[124:127], v[60:63], a[212:215], v206, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[64:67], a[200:203], v206, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[120:123], v[68:71], a[204:207], v206, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[124:127], v[64:67], a[216:219], v206, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[124:127], v[68:71], a[220:223], v206, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[128:131], v[56:59], a[224:227], v207, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[128:131], v[60:63], a[228:231], v207, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[132:135], v[56:59], a[240:243], v207, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s26, s26, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[132:135], v[60:63], a[244:247], v207, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[128:131], v[64:67], a[232:235], v207, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[128:131], v[68:71], a[236:239], v207, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[64:67], a[248:251], v207, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[68:71], a[252:255], v207, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0DF7                                  
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[72:75], v220, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[8:11], a[16:19], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v217 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[12:15], a[20:23], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v217 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[76:79], v221, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[16:19], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v217 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[20:23], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[8:11], a[32:35], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v217 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[12:15], a[36:39], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[80:83], v222, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[8:11], a[48:51], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v217 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:151], v[12:15], a[52:55], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[16:19], a[40:43], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v217 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:147], v[20:23], a[44:47], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v223, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[16:19], a[56:59], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v217 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[20:23], a[60:63], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[24:27], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v217 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[28:31], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v224, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[156:159], v[24:27], a[16:19], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v219 offset:1536                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[156:159], v[28:31], a[20:23], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[32:35], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v219 offset:1792                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[36:39], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v225, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[32:35], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[36:39], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[160:163], v[24:27], a[32:35], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[28:31], a[36:39], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v226, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[164:167], v[24:27], a[48:51], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[28:31], a[52:55], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[32:35], a[40:43], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[36:39], a[44:47], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[32:35], a[56:59], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[36:39], a[60:63], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[168:171], v[8:11], a[64:67], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[168:171], v[12:15], a[68:71], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v204, v236, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:175], v[8:11], a[80:83], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[172:175], v[12:15], a[84:87], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[168:171], v[16:19], a[72:75], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[20:23], a[76:79], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v237, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[172:175], v[16:19], a[88:91], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[172:175], v[20:23], a[92:95], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[8:11], a[96:99], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[12:15], a[100:103], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[104:107], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:183], v[8:11], a[112:115], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[12:15], a[116:119], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[176:179], v[16:19], a[104:107], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[176:179], v[20:23], a[108:111], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v229, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[16:19], a[120:123], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[20:23], a[124:127], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[184:187], v[24:27], a[64:67], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[28:31], a[68:71], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v230, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[188:191], v[24:27], a[80:83], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[188:191], v[28:31], a[84:87], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[184:187], v[32:35], a[72:75], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[184:187], v[36:39], a[76:79], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v231, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[32:35], a[88:91], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[36:39], a[92:95], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:195], v[24:27], a[96:99], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[192:195], v[28:31], a[100:103], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v232, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:199], v[24:27], a[112:115], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[196:199], v[28:31], a[116:119], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[192:195], v[32:35], a[104:107], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[192:195], v[36:39], a[108:111], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v233, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[32:35], a[120:123], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[36:39], a[124:127], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(18) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[40:43], a[128:131], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[44:47], a[132:135], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v234, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[40:43], a[144:147], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v216                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[44:47], a[148:151], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[48:51], a[136:139], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v216 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[52:55], a[140:143], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v235, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[48:51], a[152:155], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v216 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[52:55], a[156:159], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[40:43], a[160:163], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v216 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[44:47], a[164:167], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v238, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[148:151], v[40:43], a[176:179], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v216 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[148:151], v[44:47], a[180:183], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[48:51], a[168:171], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v216 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[52:55], a[172:175], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v239, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[148:151], v[48:51], a[184:187], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v216 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[52:55], a[188:191], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[56:59], a[128:131], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v216 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[60:63], a[132:135], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[156:159], v[56:59], a[144:147], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v219                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[156:159], v[60:63], a[148:151], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[64:67], a[136:139], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v219 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[68:71], a[140:143], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[156:159], v[64:67], a[152:155], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[156:159], v[68:71], a[156:159], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s68                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[160:163], v[56:59], a[160:163], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[160:163], v[60:63], a[164:167], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:167], v[56:59], a[176:179], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:167], v[60:63], a[180:183], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:163], v[64:67], a[168:171], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[160:163], v[68:71], a[172:175], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[64:67], a[184:187], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:167], v[68:71], a[188:191], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:171], v[40:43], a[192:195], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:171], v[44:47], a[196:199], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[172:175], v[40:43], a[208:211], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[172:175], v[44:47], a[212:215], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:171], v[48:51], a[200:203], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:171], v[52:55], a[204:207], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[172:175], v[48:51], a[216:219], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[172:175], v[52:55], a[220:223], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[40:43], a[224:227], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[44:47], a[228:231], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[180:183], v[40:43], a[240:243], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[180:183], v[44:47], a[244:247], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:179], v[48:51], a[232:235], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:179], v[52:55], a[236:239], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[180:183], v[48:51], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s72, s72, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:183], v[52:55], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[184:187], v[56:59], a[192:195], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[184:187], v[60:63], a[196:199], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[188:191], v[56:59], a[208:211], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[188:191], v[60:63], a[212:215], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[184:187], v[64:67], a[200:203], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[184:187], v[68:71], a[204:207], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[188:191], v[64:67], a[216:219], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[188:191], v[68:71], a[220:223], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:195], v[56:59], a[224:227], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[192:195], v[60:63], a[228:231], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[196:199], v[56:59], a[240:243], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s26, s26, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[196:199], v[60:63], a[244:247], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[192:195], v[64:67], a[232:235], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[192:195], v[68:71], a[236:239], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[64:67], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[68:71], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0DF7                                  
	s_branch label_03F0                                        
	
label_08F4:
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[8:11], a[0:3], v204, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[12:15], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v216 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[8:11], a[16:19], v204, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[136:139], v220, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[12:15], a[20:23], v204, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v216 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[16:19], a[8:11], v204, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[20:23], a[12:15], v204, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v216 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[16:19], a[24:27], v204, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[140:143], v221, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[20:23], a[28:31], v204, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v216 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[8:11], a[32:35], v205, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[12:15], a[36:39], v205, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v216 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[8:11], a[48:51], v205, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[144:147], v222, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[12:15], a[52:55], v205, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v216 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[16:19], a[40:43], v205, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[20:23], a[44:47], v205, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v216 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[16:19], a[56:59], v205, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[148:151], v223, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[20:23], a[60:63], v205, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v216 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[24:27], a[0:3], v204, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[28:31], a[4:7], v204, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v219 offset:512                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:95], v[24:27], a[16:19], v204, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[152:155], v224, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:95], v[28:31], a[20:23], v204, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v219 offset:768                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[32:35], a[8:11], v204, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[36:39], a[12:15], v204, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:95], v[32:35], a[24:27], v204, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[156:159], v225, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:95], v[36:39], a[28:31], v204, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[96:99], v[24:27], a[32:35], v205, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[96:99], v[28:31], a[36:39], v205, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:103], v[24:27], a[48:51], v205, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[160:163], v226, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:103], v[28:31], a[52:55], v205, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[96:99], v[32:35], a[40:43], v205, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[96:99], v[36:39], a[44:47], v205, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[32:35], a[56:59], v205, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[164:167], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[36:39], a[60:63], v205, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[8:11], a[64:67], v206, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[12:15], a[68:71], v206, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[8:11], a[80:83], v206, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[108:111], v[12:15], a[84:87], v206, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v208, v236, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[16:19], a[72:75], v206, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[20:23], a[76:79], v206, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[16:19], a[88:91], v206, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[20:23], a[92:95], v206, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v209, v237, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[8:11], a[96:99], v207, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[112:115], v[12:15], a[100:103], v207, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[8:11], a[112:115], v207, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[12:15], a[116:119], v207, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[168:171], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[16:19], a[104:107], v207, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[20:23], a[108:111], v207, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[116:119], v[16:19], a[120:123], v207, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:119], v[20:23], a[124:127], v207, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[172:175], v229, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[24:27], a[64:67], v206, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[28:31], a[68:71], v206, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[24:27], a[80:83], v206, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[28:31], a[84:87], v206, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[176:179], v230, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[32:35], a[72:75], v206, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[36:39], a[76:79], v206, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[32:35], a[88:91], v206, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[36:39], a[92:95], v206, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[180:183], v231, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[24:27], a[96:99], v207, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[28:31], a[100:103], v207, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[24:27], a[112:115], v207, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[28:31], a[116:119], v207, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[184:187], v232, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[128:131], v[32:35], a[104:107], v207, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[128:131], v[36:39], a[108:111], v207, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[32:35], a[120:123], v207, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[36:39], a[124:127], v207, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[188:191], v233, s[16:19], 0 offen    
	s_waitcnt vmcnt(18) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[40:43], a[128:131], v204, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[44:47], a[132:135], v204, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v217                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[76:79], v[40:43], a[144:147], v204, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[192:195], v234, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[76:79], v[44:47], a[148:151], v204, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v217 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:75], v[48:51], a[136:139], v204, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:75], v[52:55], a[140:143], v204, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v217 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[76:79], v[48:51], a[152:155], v204, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[196:199], v235, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[76:79], v[52:55], a[156:159], v204, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v217 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[80:83], v[40:43], a[160:163], v205, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[80:83], v[44:47], a[164:167], v205, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v217 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[84:87], v[40:43], a[176:179], v205, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v210, v238, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[84:87], v[44:47], a[180:183], v205, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v217 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[80:83], v[48:51], a[168:171], v205, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[80:83], v[52:55], a[172:175], v205, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v217 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[84:87], v[48:51], a[184:187], v205, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v211, v239, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[84:87], v[52:55], a[188:191], v205, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v217 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[56:59], a[128:131], v204, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s67                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[60:63], a[132:135], v204, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v219 offset:1024                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[56:59], a[144:147], v204, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[60:63], a[148:151], v204, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v219 offset:1280                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:91], v[64:67], a[136:139], v204, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x1080, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:91], v[68:71], a[140:143], v204, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[64:67], a[152:155], v204, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[68:71], a[156:159], v204, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[56:59], a[160:163], v205, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0, s68                                       
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[60:63], a[164:167], v205, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[56:59], a[176:179], v205, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[60:63], a[180:183], v205, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[64:67], a[168:171], v205, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[68:71], a[172:175], v205, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[64:67], a[184:187], v205, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[68:71], a[188:191], v205, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[40:43], a[192:195], v206, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x2100, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[104:107], v[44:47], a[196:199], v206, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[108:111], v[40:43], a[208:211], v206, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:111], v[44:47], a[212:215], v206, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[48:51], a[200:203], v206, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x3180, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[104:107], v[52:55], a[204:207], v206, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:111], v[48:51], a[216:219], v206, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:111], v[52:55], a[220:223], v206, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[40:43], a[224:227], v207, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[112:115], v[44:47], a[228:231], v207, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[116:119], v[40:43], a[240:243], v207, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[116:119], v[44:47], a[244:247], v207, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[112:115], v[48:51], a[232:235], v207, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[112:115], v[52:55], a[236:239], v207, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[116:119], v[48:51], a[248:251], v207, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s72, s72, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[116:119], v[52:55], a[252:255], v207, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[56:59], a[192:195], v206, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[120:123], v[60:63], a[196:199], v206, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[56:59], a[208:211], v206, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[124:127], v[60:63], a[212:215], v206, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[64:67], a[200:203], v206, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[120:123], v[68:71], a[204:207], v206, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[124:127], v[64:67], a[216:219], v206, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[124:127], v[68:71], a[220:223], v206, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[128:131], v[56:59], a[224:227], v207, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[128:131], v[60:63], a[228:231], v207, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[132:135], v[56:59], a[240:243], v207, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s26, s26, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[132:135], v[60:63], a[244:247], v207, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[128:131], v[64:67], a[232:235], v207, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[128:131], v[68:71], a[236:239], v207, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[64:67], a[248:251], v207, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[68:71], a[252:255], v207, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0DF7                                  
	s_waitcnt vmcnt(15) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v217 offset:8448                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[8:11], a[16:19], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[72:75], v220, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[12:15], a[20:23], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[56:59], v217 offset:8512                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v217 offset:8960                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[16:19], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[76:79], v221, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[20:23], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[60:63], v217 offset:9024                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[8:11], a[32:35], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[12:15], a[36:39], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v217 offset:12672                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[8:11], a[48:51], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[80:83], v222, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:151], v[12:15], a[52:55], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[64:67], v217 offset:12736                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[16:19], a[40:43], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:147], v[20:23], a[44:47], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[52:55], v217 offset:13184                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[16:19], a[56:59], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[84:87], v223, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[20:23], a[60:63], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v217 offset:13248                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[24:27], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[28:31], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v202, v219 offset:1536                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[156:159], v[24:27], a[16:19], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[88:91], v224, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[156:159], v[28:31], a[20:23], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v203, v219 offset:1792                         
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[32:35], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[36:39], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[156:159], v[32:35], a[24:27], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[92:95], v225, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[36:39], a[28:31], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[160:163], v[24:27], a[32:35], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[28:31], a[36:39], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[164:167], v[24:27], a[48:51], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[96:99], v226, s[16:19], 0 offen      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[28:31], a[52:55], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[32:35], a[40:43], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[36:39], a[44:47], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[164:167], v[32:35], a[56:59], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[100:103], v227, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[36:39], a[60:63], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)                                        
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[168:171], v[8:11], a[64:67], v210, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s62, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[168:171], v[12:15], a[68:71], v210, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:175], v[8:11], a[80:83], v210, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s62, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[172:175], v[12:15], a[84:87], v210, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v204, v236, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[168:171], v[16:19], a[72:75], v210, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s69, s69, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[20:23], a[76:79], v210, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[172:175], v[16:19], a[88:91], v210, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s71, s71, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[172:175], v[20:23], a[92:95], v210, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v205, v237, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[8:11], a[96:99], v211, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s12, s12, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[176:179], v[12:15], a[100:103], v211, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:183], v[8:11], a[112:115], v211, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s13, 0, s13                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[12:15], a[116:119], v211, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[104:107], v228, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[176:179], v[16:19], a[104:107], v211, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_sub_u32 s14, s14, s69                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[176:179], v[20:23], a[108:111], v211, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:183], v[16:19], a[120:123], v211, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s20, s20, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:183], v[20:23], a[124:127], v211, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[108:111], v229, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[184:187], v[24:27], a[64:67], v210, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s21, 0, s21                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[28:31], a[68:71], v210, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[188:191], v[24:27], a[80:83], v210, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s22, s22, s71                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[188:191], v[28:31], a[84:87], v210, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[112:115], v230, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[184:187], v[32:35], a[72:75], v210, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[184:187], v[36:39], a[76:79], v210, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[32:35], a[88:91], v210, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[36:39], a[92:95], v210, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[116:119], v231, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:195], v[24:27], a[96:99], v211, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[192:195], v[28:31], a[100:103], v211, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:199], v[24:27], a[112:115], v211, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[196:199], v[28:31], a[116:119], v211, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[120:123], v232, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[192:195], v[32:35], a[104:107], v211, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[192:195], v[36:39], a[108:111], v211, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[32:35], a[120:123], v211, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[36:39], a[124:127], v211, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[124:127], v233, s[16:19], 0 offen    
	s_waitcnt vmcnt(18) lgkmcnt(0)                             
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[40:43], a[128:131], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier                                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[44:47], a[132:135], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[8:11], v216                                 
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[40:43], a[144:147], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[128:131], v234, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[44:47], a[148:151], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v216 offset:64                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[48:51], a[136:139], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[52:55], a[140:143], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[12:15], v216 offset:512                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[48:51], a[152:155], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v[132:135], v235, s[16:19], 0 offen    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[52:55], a[156:159], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[28:31], v216 offset:576                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[40:43], a[160:163], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[44:47], a[164:167], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v216 offset:4224                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[148:151], v[40:43], a[176:179], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v206, v238, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[148:151], v[44:47], a[180:183], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[32:35], v216 offset:4288                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[48:51], a[168:171], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[52:55], a[172:175], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v216 offset:4736                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[148:151], v[48:51], a[184:187], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v207, v239, s[24:27], 0 offen            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[52:55], a[188:191], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v216 offset:4800                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[56:59], a[128:131], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x4200, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[60:63], a[132:135], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v200, v219                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[156:159], v[56:59], a[144:147], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[156:159], v[60:63], a[148:151], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v201, v219 offset:256                          
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[64:67], a[136:139], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x5280, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[68:71], a[140:143], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[156:159], v[64:67], a[152:155], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[156:159], v[68:71], a[156:159], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[160:163], v[56:59], a[160:163], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x400, s68                                   
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[160:163], v[60:63], a[164:167], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:167], v[56:59], a[176:179], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v218, s[20:23], 0 offen lds              
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:167], v[60:63], a[180:183], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:163], v[64:67], a[168:171], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[160:163], v[68:71], a[172:175], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[64:67], a[184:187], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:167], v[68:71], a[188:191], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[168:171], v[40:43], a[192:195], v210, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x6300, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:171], v[44:47], a[196:199], v210, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[172:175], v[40:43], a[208:211], v210, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[172:175], v[44:47], a[212:215], v210, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[168:171], v[48:51], a[200:203], v210, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 m0, 0x7380, s67                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:171], v[52:55], a[204:207], v210, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[172:175], v[48:51], a[216:219], v210, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[172:175], v[52:55], a[220:223], v210, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[176:179], v[40:43], a[224:227], v211, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s63, 0x200, s60                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[176:179], v[44:47], a[228:231], v211, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[180:183], v[40:43], a[240:243], v211, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lt_u32 s63, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[180:183], v[44:47], a[244:247], v211, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[176:179], v[48:51], a[232:235], v211, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s70, s70, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[176:179], v[52:55], a[236:239], v211, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[180:183], v[48:51], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cselect_b32 s72, s72, 0                                  
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[180:183], v[52:55], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[184:187], v[56:59], a[192:195], v210, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s16, s16, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[184:187], v[60:63], a[196:199], v210, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[188:191], v[56:59], a[208:211], v210, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s17, 0, s17                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[188:191], v[60:63], a[212:215], v210, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[184:187], v[64:67], a[200:203], v210, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s18, s18, s70                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[184:187], v[68:71], a[204:207], v210, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[188:191], v[64:67], a[216:219], v210, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_add_u32 s24, s24, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[188:191], v[68:71], a[220:223], v210, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[192:195], v[56:59], a[224:227], v211, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addc_u32 s25, 0, s25                                     
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[192:195], v[60:63], a[228:231], v211, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[196:199], v[56:59], a[240:243], v211, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_sub_u32 s26, s26, s72                                    
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[196:199], v[60:63], a[244:247], v211, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[192:195], v[64:67], a[232:235], v211, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_addk_i32 s60, 0x100                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[192:195], v[68:71], a[236:239], v211, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[196:199], v[64:67], a[248:251], v211, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cmp_lt_i32 s60, s61                                      
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[196:199], v[68:71], a[252:255], v211, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_cbranch_scc0 label_0DF7                                  
	s_branch label_08F4                                        
	
label_0DF7:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_barrier                                                  
	s_cmp_eq_u32 s65, 0                                        
	s_cbranch_scc1 label_1660                                  
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e64 v4, v4, 8                                
	v_and_b32_e64 v5, v0, 15                                   
	v_lshlrev_b32_e32 v5, 8, v5                                
	v_add_i32 v4, v4, v5                                       
	s_mul_i32 s62, s46, 0x4000                                 
	s_add_i32 s62, s62, 0                                      
	v_add_i32 v4, v4, s62                                      
	v_accvgpr_read_b32 v8, a0                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a1                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a2                                 
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a3                                 
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a16                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a17                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a18                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a19                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17]                                  
	ds_write_b64 v4, v[18:19] offset:32                        
	v_accvgpr_read_b32 v8, a32                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a33                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a34                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a35                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a48                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a49                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a50                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a51                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:64                        
	ds_write_b64 v4, v[18:19] offset:96                        
	v_accvgpr_read_b32 v8, a4                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a5                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a6                                 
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a7                                 
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a20                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a21                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a22                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a23                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4096                      
	ds_write_b64 v4, v[18:19] offset:4128                      
	v_accvgpr_read_b32 v8, a36                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a37                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a38                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a39                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a52                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a53                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a54                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a55                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4160                      
	ds_write_b64 v4, v[18:19] offset:4192                      
	v_accvgpr_read_b32 v8, a8                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a9                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a10                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a11                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a24                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a25                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a26                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a27                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8192                      
	ds_write_b64 v4, v[18:19] offset:8224                      
	v_accvgpr_read_b32 v8, a40                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a41                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a42                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a43                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a56                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a57                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a58                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a59                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8256                      
	ds_write_b64 v4, v[18:19] offset:8288                      
	v_accvgpr_read_b32 v8, a12                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a13                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a14                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a15                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a28                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a29                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a30                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a31                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12288                     
	ds_write_b64 v4, v[18:19] offset:12320                     
	v_accvgpr_read_b32 v8, a44                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a45                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a46                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a47                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a60                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a61                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a62                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a63                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12352                     
	ds_write_b64 v4, v[18:19] offset:12384                     
	v_accvgpr_read_b32 v8, a64                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a65                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a66                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a67                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a80                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a81                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a82                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a83                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:128                       
	ds_write_b64 v4, v[18:19] offset:160                       
	v_accvgpr_read_b32 v8, a96                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a97                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a98                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a99                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a112                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a113                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a114                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a115                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:192                       
	ds_write_b64 v4, v[18:19] offset:224                       
	v_accvgpr_read_b32 v8, a68                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a69                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a70                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a71                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a84                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a85                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a86                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a87                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4224                      
	ds_write_b64 v4, v[18:19] offset:4256                      
	v_accvgpr_read_b32 v8, a100                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a101                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a102                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a103                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a116                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a117                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a118                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a119                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4288                      
	ds_write_b64 v4, v[18:19] offset:4320                      
	v_accvgpr_read_b32 v8, a72                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a73                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a74                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a75                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a88                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a89                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a90                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a91                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8320                      
	ds_write_b64 v4, v[18:19] offset:8352                      
	v_accvgpr_read_b32 v8, a104                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a105                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a106                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a107                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a120                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a121                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a122                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a123                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8384                      
	ds_write_b64 v4, v[18:19] offset:8416                      
	v_accvgpr_read_b32 v8, a76                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a77                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a78                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a79                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a92                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a93                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a94                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a95                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12416                     
	ds_write_b64 v4, v[18:19] offset:12448                     
	v_accvgpr_read_b32 v8, a108                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a109                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a110                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a111                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a124                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a125                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a126                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a127                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12480                     
	ds_write_b64 v4, v[18:19] offset:12512                     
	s_waitcnt lgkmcnt(0)                                       
	v_mul_i32_i24_e64 v4, v0, 4                                
	v_add_i32 v4, v4, s62                                      
	s_mul_i32 s63, s36, 0                                      
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4                                        
	ds_read_b32 v17, v4 offset:256                             
	ds_read_b32 v18, v4 offset:512                             
	ds_read_b32 v19, v4 offset:768                             
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 4                                      
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:1024                            
	ds_read_b32 v17, v4 offset:1280                            
	ds_read_b32 v18, v4 offset:1536                            
	ds_read_b32 v19, v4 offset:1792                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 8                                      
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:2048                            
	ds_read_b32 v17, v4 offset:2304                            
	ds_read_b32 v18, v4 offset:2560                            
	ds_read_b32 v19, v4 offset:2816                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 12                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:3072                            
	ds_read_b32 v17, v4 offset:3328                            
	ds_read_b32 v18, v4 offset:3584                            
	ds_read_b32 v19, v4 offset:3840                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 16                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:4096                            
	ds_read_b32 v17, v4 offset:4352                            
	ds_read_b32 v18, v4 offset:4608                            
	ds_read_b32 v19, v4 offset:4864                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 20                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:5120                            
	ds_read_b32 v17, v4 offset:5376                            
	ds_read_b32 v18, v4 offset:5632                            
	ds_read_b32 v19, v4 offset:5888                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 24                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:6144                            
	ds_read_b32 v17, v4 offset:6400                            
	ds_read_b32 v18, v4 offset:6656                            
	ds_read_b32 v19, v4 offset:6912                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 28                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:7168                            
	ds_read_b32 v17, v4 offset:7424                            
	ds_read_b32 v18, v4 offset:7680                            
	ds_read_b32 v19, v4 offset:7936                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 32                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:8192                            
	ds_read_b32 v17, v4 offset:8448                            
	ds_read_b32 v18, v4 offset:8704                            
	ds_read_b32 v19, v4 offset:8960                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 36                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:9216                            
	ds_read_b32 v17, v4 offset:9472                            
	ds_read_b32 v18, v4 offset:9728                            
	ds_read_b32 v19, v4 offset:9984                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 40                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:10240                           
	ds_read_b32 v17, v4 offset:10496                           
	ds_read_b32 v18, v4 offset:10752                           
	ds_read_b32 v19, v4 offset:11008                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 44                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:11264                           
	ds_read_b32 v17, v4 offset:11520                           
	ds_read_b32 v18, v4 offset:11776                           
	ds_read_b32 v19, v4 offset:12032                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 48                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:12288                           
	ds_read_b32 v17, v4 offset:12544                           
	ds_read_b32 v18, v4 offset:12800                           
	ds_read_b32 v19, v4 offset:13056                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 52                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:13312                           
	ds_read_b32 v17, v4 offset:13568                           
	ds_read_b32 v18, v4 offset:13824                           
	ds_read_b32 v19, v4 offset:14080                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 56                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:14336                           
	ds_read_b32 v17, v4 offset:14592                           
	ds_read_b32 v18, v4 offset:14848                           
	ds_read_b32 v19, v4 offset:15104                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 60                                     
	v_add_u32_e32 v244, s63, v240                              
	ds_read_b32 v16, v4 offset:15360                           
	ds_read_b32 v17, v4 offset:15616                           
	ds_read_b32 v18, v4 offset:15872                           
	ds_read_b32 v19, v4 offset:16128                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e64 v4, v4, 8                                
	v_and_b32_e64 v5, v0, 15                                   
	v_lshlrev_b32_e32 v5, 8, v5                                
	v_add_i32 v4, v4, v5                                       
	s_mul_i32 s62, s46, 0x4000                                 
	s_add_i32 s62, s62, 0                                      
	v_add_i32 v4, v4, s62                                      
	v_accvgpr_read_b32 v8, a128                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a129                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a130                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a131                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a144                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a145                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a146                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a147                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17]                                  
	ds_write_b64 v4, v[18:19] offset:32                        
	v_accvgpr_read_b32 v8, a160                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a161                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a162                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a163                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a176                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a177                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a178                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a179                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:64                        
	ds_write_b64 v4, v[18:19] offset:96                        
	v_accvgpr_read_b32 v8, a132                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a133                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a134                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a135                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a148                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a149                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a150                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a151                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4096                      
	ds_write_b64 v4, v[18:19] offset:4128                      
	v_accvgpr_read_b32 v8, a164                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a165                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a166                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a167                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a180                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a181                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a182                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a183                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4160                      
	ds_write_b64 v4, v[18:19] offset:4192                      
	v_accvgpr_read_b32 v8, a136                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a137                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a138                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a139                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a152                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a153                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a154                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a155                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8192                      
	ds_write_b64 v4, v[18:19] offset:8224                      
	v_accvgpr_read_b32 v8, a168                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a169                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a170                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a171                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a184                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a185                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a186                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a187                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8256                      
	ds_write_b64 v4, v[18:19] offset:8288                      
	v_accvgpr_read_b32 v8, a140                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a141                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a142                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a143                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a156                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a157                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a158                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a159                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12288                     
	ds_write_b64 v4, v[18:19] offset:12320                     
	v_accvgpr_read_b32 v8, a172                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a173                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a174                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a175                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a188                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a189                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a190                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a191                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12352                     
	ds_write_b64 v4, v[18:19] offset:12384                     
	v_accvgpr_read_b32 v8, a192                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a193                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a194                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a195                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a208                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a209                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a210                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a211                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:128                       
	ds_write_b64 v4, v[18:19] offset:160                       
	v_accvgpr_read_b32 v8, a224                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a225                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a226                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a227                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a240                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a241                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a242                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a243                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:192                       
	ds_write_b64 v4, v[18:19] offset:224                       
	v_accvgpr_read_b32 v8, a196                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a197                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a198                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a199                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a212                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a213                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a214                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a215                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4224                      
	ds_write_b64 v4, v[18:19] offset:4256                      
	v_accvgpr_read_b32 v8, a228                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a229                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a230                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a231                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a244                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a245                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a246                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a247                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:4288                      
	ds_write_b64 v4, v[18:19] offset:4320                      
	v_accvgpr_read_b32 v8, a200                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a201                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a202                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a203                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a216                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a217                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a218                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a219                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8320                      
	ds_write_b64 v4, v[18:19] offset:8352                      
	v_accvgpr_read_b32 v8, a232                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a233                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a234                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a235                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a248                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a249                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a250                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a251                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:8384                      
	ds_write_b64 v4, v[18:19] offset:8416                      
	v_accvgpr_read_b32 v8, a204                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a205                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a206                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a207                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a220                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a221                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a222                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a223                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12416                     
	ds_write_b64 v4, v[18:19] offset:12448                     
	v_accvgpr_read_b32 v8, a236                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a237                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a238                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a239                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a252                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a253                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a254                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a255                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	ds_write_b64 v4, v[16:17] offset:12480                     
	ds_write_b64 v4, v[18:19] offset:12512                     
	s_waitcnt lgkmcnt(0)                                       
	v_mul_i32_i24_e64 v4, v0, 4                                
	v_add_i32 v4, v4, s62                                      
	s_mul_i32 s63, s36, 0                                      
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4                                        
	ds_read_b32 v17, v4 offset:256                             
	ds_read_b32 v18, v4 offset:512                             
	ds_read_b32 v19, v4 offset:768                             
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 4                                      
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:1024                            
	ds_read_b32 v17, v4 offset:1280                            
	ds_read_b32 v18, v4 offset:1536                            
	ds_read_b32 v19, v4 offset:1792                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 8                                      
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:2048                            
	ds_read_b32 v17, v4 offset:2304                            
	ds_read_b32 v18, v4 offset:2560                            
	ds_read_b32 v19, v4 offset:2816                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 12                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:3072                            
	ds_read_b32 v17, v4 offset:3328                            
	ds_read_b32 v18, v4 offset:3584                            
	ds_read_b32 v19, v4 offset:3840                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 16                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:4096                            
	ds_read_b32 v17, v4 offset:4352                            
	ds_read_b32 v18, v4 offset:4608                            
	ds_read_b32 v19, v4 offset:4864                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 20                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:5120                            
	ds_read_b32 v17, v4 offset:5376                            
	ds_read_b32 v18, v4 offset:5632                            
	ds_read_b32 v19, v4 offset:5888                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 24                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:6144                            
	ds_read_b32 v17, v4 offset:6400                            
	ds_read_b32 v18, v4 offset:6656                            
	ds_read_b32 v19, v4 offset:6912                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 28                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:7168                            
	ds_read_b32 v17, v4 offset:7424                            
	ds_read_b32 v18, v4 offset:7680                            
	ds_read_b32 v19, v4 offset:7936                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 32                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:8192                            
	ds_read_b32 v17, v4 offset:8448                            
	ds_read_b32 v18, v4 offset:8704                            
	ds_read_b32 v19, v4 offset:8960                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 36                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:9216                            
	ds_read_b32 v17, v4 offset:9472                            
	ds_read_b32 v18, v4 offset:9728                            
	ds_read_b32 v19, v4 offset:9984                            
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 40                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:10240                           
	ds_read_b32 v17, v4 offset:10496                           
	ds_read_b32 v18, v4 offset:10752                           
	ds_read_b32 v19, v4 offset:11008                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 44                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:11264                           
	ds_read_b32 v17, v4 offset:11520                           
	ds_read_b32 v18, v4 offset:11776                           
	ds_read_b32 v19, v4 offset:12032                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 48                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:12288                           
	ds_read_b32 v17, v4 offset:12544                           
	ds_read_b32 v18, v4 offset:12800                           
	ds_read_b32 v19, v4 offset:13056                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 52                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:13312                           
	ds_read_b32 v17, v4 offset:13568                           
	ds_read_b32 v18, v4 offset:13824                           
	ds_read_b32 v19, v4 offset:14080                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 56                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:14336                           
	ds_read_b32 v17, v4 offset:14592                           
	ds_read_b32 v18, v4 offset:14848                           
	ds_read_b32 v19, v4 offset:15104                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_mul_i32 s63, s36, 60                                     
	v_add_u32_e32 v244, s63, v242                              
	ds_read_b32 v16, v4 offset:15360                           
	ds_read_b32 v17, v4 offset:15616                           
	ds_read_b32 v18, v4 offset:15872                           
	ds_read_b32 v19, v4 offset:16128                           
	s_waitcnt lgkmcnt(3)                                       
	buffer_atomic_pk_add_bf16 v16, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(2)                                       
	buffer_atomic_pk_add_bf16 v17, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(1)                                       
	buffer_atomic_pk_add_bf16 v18, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_waitcnt lgkmcnt(0)                                       
	buffer_atomic_pk_add_bf16 v19, v244, s[4:7], 0 offen       
	v_add_u32_e64 v244, v244, s36                              
	s_branch label_1BA0                                        
	
label_1660:
	s_mul_i32 s62, s36, 0                                      
	v_add_u32_e32 v244, s62, v240                              
	v_accvgpr_read_b32 v8, a0                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a1                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a2                                 
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a3                                 
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a16                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a17                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a18                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a19                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a32                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a33                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a34                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a35                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a48                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a49                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a50                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a51                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v244, s62, v240                              
	v_accvgpr_read_b32 v8, a4                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a5                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a6                                 
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a7                                 
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a20                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a21                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a22                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a23                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a36                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a37                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a38                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a39                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a52                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a53                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a54                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a55                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 32                                     
	v_add_u32_e32 v244, s62, v240                              
	v_accvgpr_read_b32 v8, a8                                  
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a9                                  
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a10                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a11                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a24                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a25                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a26                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a27                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a40                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a41                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a42                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a43                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a56                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a57                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a58                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a59                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 48                                     
	v_add_u32_e32 v244, s62, v240                              
	v_accvgpr_read_b32 v8, a12                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a13                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a14                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a15                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a28                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a29                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a30                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a31                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a44                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a45                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a46                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a47                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a60                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a61                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a62                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a63                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 0                                      
	v_add_u32_e32 v244, s62, v241                              
	v_accvgpr_read_b32 v8, a64                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a65                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a66                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a67                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a80                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a81                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a82                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a83                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a96                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a97                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a98                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a99                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a112                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a113                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a114                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a115                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v244, s62, v241                              
	v_accvgpr_read_b32 v8, a68                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a69                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a70                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a71                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a84                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a85                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a86                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a87                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a100                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a101                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a102                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a103                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a116                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a117                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a118                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a119                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 32                                     
	v_add_u32_e32 v244, s62, v241                              
	v_accvgpr_read_b32 v8, a72                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a73                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a74                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a75                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a88                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a89                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a90                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a91                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a104                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a105                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a106                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a107                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a120                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a121                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a122                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a123                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 48                                     
	v_add_u32_e32 v244, s62, v241                              
	v_accvgpr_read_b32 v8, a76                                 
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a77                                 
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a78                                
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a79                                
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a92                                
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a93                                
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a94                                
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a95                                
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a108                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a109                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a110                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a111                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a124                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a125                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a126                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a127                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 0                                      
	v_add_u32_e32 v244, s62, v242                              
	v_accvgpr_read_b32 v8, a128                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a129                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a130                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a131                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a144                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a145                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a146                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a147                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a160                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a161                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a162                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a163                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a176                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a177                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a178                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a179                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v244, s62, v242                              
	v_accvgpr_read_b32 v8, a132                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a133                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a134                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a135                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a148                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a149                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a150                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a151                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a164                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a165                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a166                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a167                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a180                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a181                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a182                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a183                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 32                                     
	v_add_u32_e32 v244, s62, v242                              
	v_accvgpr_read_b32 v8, a136                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a137                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a138                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a139                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a152                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a153                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a154                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a155                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a168                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a169                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a170                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a171                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a184                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a185                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a186                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a187                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 48                                     
	v_add_u32_e32 v244, s62, v242                              
	v_accvgpr_read_b32 v8, a140                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a141                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a142                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a143                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a156                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a157                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a158                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a159                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a172                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a173                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a174                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a175                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a188                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a189                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a190                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a191                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 0                                      
	v_add_u32_e32 v244, s62, v243                              
	v_accvgpr_read_b32 v8, a192                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a193                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a194                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a195                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a208                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a209                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a210                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a211                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a224                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a225                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a226                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a227                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a240                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a241                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a242                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a243                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 16                                     
	v_add_u32_e32 v244, s62, v243                              
	v_accvgpr_read_b32 v8, a196                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a197                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a198                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a199                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a212                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a213                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a214                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a215                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a228                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a229                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a230                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a231                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a244                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a245                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a246                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a247                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 32                                     
	v_add_u32_e32 v244, s62, v243                              
	v_accvgpr_read_b32 v8, a200                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a201                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a202                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a203                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a216                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a217                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a218                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a219                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a232                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a233                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a234                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a235                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a248                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a249                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a250                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a251                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	s_mul_i32 s62, s36, 48                                     
	v_add_u32_e32 v244, s62, v243                              
	v_accvgpr_read_b32 v8, a204                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a205                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a206                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a207                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a220                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a221                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a222                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a223                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	v_accvgpr_read_b32 v8, a236                                
	v_mul_f32_e32 v8, s41, v8                                  
	v_accvgpr_read_b32 v9, a237                                
	v_mul_f32_e32 v9, s41, v9                                  
	v_accvgpr_read_b32 v10, a238                               
	v_mul_f32_e32 v10, s41, v10                                
	v_accvgpr_read_b32 v11, a239                               
	v_mul_f32_e32 v11, s41, v11                                
	v_accvgpr_read_b32 v12, a252                               
	v_mul_f32_e32 v12, s41, v12                                
	v_accvgpr_read_b32 v13, a253                               
	v_mul_f32_e32 v13, s41, v13                                
	v_accvgpr_read_b32 v14, a254                               
	v_mul_f32_e32 v14, s41, v14                                
	v_accvgpr_read_b32 v15, a255                               
	v_mul_f32_e32 v15, s41, v15                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v244, s[4:7], 0 offen       
	v_add_i32 v244, v244, 64                                   
	
label_1BA0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512
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
    .name:           f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

