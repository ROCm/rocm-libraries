// Auto-generated from bf16gemm_bf16_tn_256x256.co
// This file can be reassembled with:
//   clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c file.s -o file.o
//   ld.lld -shared -o file.co file.o

// Note: Target is specified via -mcpu= command line flag

.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0

// ===== Kernel Code =====
.text
.globl _ZN5aiter24bf16gemm_bf16_tn_256x256E
.p2align 8
.type _ZN5aiter24bf16gemm_bf16_tn_256x256E,@function

_ZN5aiter24bf16gemm_bf16_tn_256x256E:
	
	s_mov_b32 s46, s2                                          
	s_mov_b32 s47, s3                                          
	s_mov_b32 s51, s4                                          
	s_and_b32 s1, s1, 0xffff                                   
	s_load_dwordx2 s[4:5], s[0:1], 0x0                         
	s_load_dwordx2 s[8:9], s[0:1], 0x10                        
	s_load_dwordx2 s[12:13], s[0:1], 0x20                      
	s_load_dwordx2 s[16:17], s[0:1], 0x30                      
	s_load_dword s38, s[0:1], 0x40                             
	s_load_dword s39, s[0:1], 0x50                             
	s_load_dword s40, s[0:1], 0x80                             
	s_load_dword s41, s[0:1], 0xa0                             
	s_load_dword s42, s[0:1], 0xc0                             
	s_load_dword s43, s[0:1], 0xe0                             
	s_load_dword s44, s[0:1], 0xf0                             
	s_load_dword s45, s[0:1], 0x100                            
	s_load_dword s58, s[0:1], 0x110                            
	s_load_dword s59, s[0:1], 0x120                            
	v_lshrrev_b32_e32 v1, 10, v0                               
	v_lshrrev_b32_e32 v2, 10, v1                               
	v_and_b32_e32 v2, 0x3ff, v2                                
	v_and_b32_e32 v1, 0x3ff, v1                                
	v_and_b32_e32 v0, 0x3ff, v0                                
	v_lshrrev_b32_e32 v3, 6, v0                                
	v_and_b32_e32 v0, 63, v0                                   
	v_readfirstlane_b32 s48, v3                                
	s_waitcnt lgkmcnt(0)                                       
	s_add_u32 s51, s44, 0xff                                   
	s_lshr_b32 s51, s51, 8                                     
	s_mul_i32 s54, s51, s47                                    
	s_add_i32 s54, s54, s46                                    
	s_lshr_b32 s55, s51, 5                                     
	s_lshl_b32 s55, s55, 5                                     
	s_sub_i32 s56, s51, s55                                    
	s_add_u32 s53, s43, 0xff                                   
	s_lshr_b32 s53, s53, 8                                     
	s_mul_i32 s57, s55, s53                                    
	s_cmp_lt_i32 s54, s57                                      
	s_cbranch_scc0 label_0060                                  
	s_lshr_b32 s51, s54, 5                                     
	v_cvt_f32_u32_e32 v4, s53                                  
	s_sub_i32 s52, 0, s53                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s52, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s51, v4                                   
	v_mul_lo_u32 v5, v4, s53                                   
	v_sub_u32_e32 v7, s51, v5                                  
	v_add_u32_e32 v6, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s53, v7                              
	v_subrev_u32_e32 v5, s53, v7                               
	s_nop 0                                                    
	v_cndmask_b32_e32 v4, v4, v6, vcc                          
	v_cndmask_b32_e32 v7, v7, v5, vcc                          
	v_add_u32_e32 v5, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s53, v7                              
	s_nop 1                                                    
	v_cndmask_b32_e32 v7, v4, v5, vcc                          
	s_nop 3                                                    
	v_readfirstlane_b32 s52, v7                                
	s_nop 3                                                    
	s_mul_i32 s47, s53, s52                                    
	s_sub_i32 s47, s51, s47                                    
	s_and_b32 s51, s54, 31                                     
	s_lshl_b32 s46, s52, 5                                     
	s_add_i32 s46, s46, s51                                    
	s_branch label_0082                                        
	
label_0060:
	s_sub_i32 s51, s54, s57                                    
	v_cvt_f32_u32_e32 v4, s56                                  
	s_sub_i32 s47, 0, s56                                      
	v_rcp_iflag_f32_e32 v4, v4                                 
	s_nop 0                                                    
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           
	v_cvt_u32_f32_e32 v4, v4                                   
	v_mul_lo_u32 v5, s47, v4                                   
	v_mul_hi_u32 v5, v4, v5                                    
	v_add_u32_e32 v4, v4, v5                                   
	v_mul_hi_u32 v4, s51, v4                                   
	v_mul_lo_u32 v5, v4, s56                                   
	v_sub_u32_e32 v7, s51, v5                                  
	v_add_u32_e32 v6, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s56, v7                              
	v_subrev_u32_e32 v5, s56, v7                               
	s_nop 0                                                    
	v_cndmask_b32_e32 v4, v4, v6, vcc                          
	v_cndmask_b32_e32 v7, v7, v5, vcc                          
	v_add_u32_e32 v5, 1, v4                                    
	v_cmp_le_u32_e32 vcc, s56, v7                              
	s_nop 1                                                    
	v_cndmask_b32_e32 v7, v4, v5, vcc                          
	s_nop 3                                                    
	v_readfirstlane_b32 s47, v7                                
	s_nop 3                                                    
	s_mul_i32 s52, s56, s47                                    
	s_sub_i32 s46, s51, s52                                    
	s_add_i32 s46, s55, s46                                    
	
label_0082:
	s_mov_b32 s6, -16                                          
	s_mov_b32 s10, -16                                         
	s_mov_b32 s18, -16                                         
	s_mov_b32 s14, -16                                         
	s_mov_b32 s7, 0x20000                                      
	s_mov_b32 s11, 0x20000                                     
	s_mov_b32 s19, 0x20000                                     
	s_mov_b32 s15, 0x20000                                     
	s_and_b32 s5, s5, 0xffff                                   
	s_and_b32 s9, s9, 0xffff                                   
	s_and_b32 s17, s17, 0xffff                                 
	s_and_b32 s13, s13, 0xffff                                 
	s_or_b32 s5, s5, 0x40000                                   
	s_or_b32 s9, s9, 0x40000                                   
	s_or_b32 s17, s17, 0x40000                                 
	s_or_b32 s13, s13, 0x40000                                 
	s_mul_i32 s51, s41, s43                                    
	s_mov_b32 s14, s51                                         
	s_mul_i32 s51, s42, s44                                    
	s_mov_b32 s18, s51                                         
	v_lshrrev_b32_e32 v4, 3, v0                                
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_lshlrev_b32_e32 v5, 4, v5                                
	v_and_b32_e32 v4, 3, v4                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_lshlrev_b32_e32 v6, 2, v6                                
	v_add_u32_e32 v5, v5, v6                                   
	v_and_b32_e32 v4, 1, v4                                    
	v_add_u32_e32 v5, v5, v4                                   
	v_mul_lo_u32 v200, s41, v5                                 
	v_mul_lo_u32 v210, s42, v5                                 
	v_and_b32_e32 v4, 7, v0                                    
	v_lshlrev_b32_e32 v4, 4, v4                                
	v_add_u32_e32 v200, v200, v4                               
	v_add_u32_e32 v210, v210, v4                               
	s_lshr_b32 s51, s48, 1                                     
	s_mul_i32 s51, s51, 8                                      
	s_and_b32 s52, s48, 1                                      
	s_mul_i32 s52, s52, 2                                      
	s_add_u32 s51, s51, s52                                    
	s_mul_i32 s52, s47, 0x100                                  
	s_add_u32 s52, s51, s52                                    
	s_mul_i32 s52, s41, s52                                    
	v_add_u32_e32 v200, s52, v200                              
	s_mul_i32 s52, s46, 0x100                                  
	s_add_u32 s52, s51, s52                                    
	s_mul_i32 s52, s42, s52                                    
	v_add_u32_e32 v210, s52, v210                              
	s_mul_i32 s51, s41, 32                                     
	v_add_u32_e32 v201, s51, v200                              
	v_add_u32_e32 v202, s51, v201                              
	v_add_u32_e32 v203, s51, v202                              
	v_add_u32_e32 v204, s51, v203                              
	v_add_u32_e32 v205, s51, v204                              
	v_add_u32_e32 v206, s51, v205                              
	v_add_u32_e32 v207, s51, v206                              
	s_mul_i32 s51, s42, 32                                     
	v_add_u32_e32 v211, s51, v210                              
	v_add_u32_e32 v212, s51, v211                              
	v_add_u32_e32 v213, s51, v212                              
	v_add_u32_e32 v214, s51, v213                              
	v_add_u32_e32 v215, s51, v214                              
	v_add_u32_e32 v216, s51, v215                              
	v_add_u32_e32 v217, s51, v216                              
	s_mul_i32 s60, 0x420, s48                                  
	s_add_u32 s60, 0, s60                                      
	s_add_u32 s61, 0x10800, s60                                
	v_and_b32_e32 v4, 15, v0                                   
	v_lshrrev_b32_e32 v5, 3, v4                                
	v_mul_i32_i24_e32 v5, 2, v5                                
	v_and_b32_e32 v4, 3, v0                                    
	v_lshrrev_b32_e32 v6, 1, v4                                
	v_add_u32_e32 v4, v5, v6                                   
	v_mul_i32_i24_e32 v208, 0x420, v4                          
	v_mul_i32_i24_e32 v218, 0x420, v4                          
	v_and_b32_e32 v4, 7, v0                                    
	v_lshrrev_b32_e32 v5, 2, v4                                
	v_mul_i32_i24_e32 v5, 0x100, v5                            
	v_add_u32_e32 v208, v5, v208                               
	v_add_u32_e32 v218, v5, v218                               
	v_and_b32_e32 v4, 1, v0                                    
	v_mul_i32_i24_e32 v6, 0x80, v4                             
	v_add_u32_e32 v208, v6, v208                               
	v_add_u32_e32 v218, v6, v218                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v208, v4, v208                               
	v_add_u32_e32 v218, v4, v218                               
	s_mov_b32 s51, 0                                           
	v_add_u32_e64 v208, v208, s51                              
	s_mov_b32 s51, 0x10800                                     
	v_add_u32_e64 v218, v218, s51                              
	s_mul_i32 s51, 0x2100, s48                                 
	v_add_u32_e64 v218, v218, s51                              
	v_add_u32_e32 v209, 0x8400, v208                           
	v_add_u32_e32 v219, 0x8400, v218                           
	s_mov_b32 s62, 0x80                                        
	s_mov_b32 s63, 0x80                                        
	s_add_u32 m0, 0, s61                                       
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a0, 0                                  
	v_accvgpr_write_b32 a1, 0                                  
	v_accvgpr_write_b32 a2, 0                                  
	v_accvgpr_write_b32 a3, 0                                  
	v_accvgpr_write_b32 a4, 0                                  
	v_accvgpr_write_b32 a5, 0                                  
	v_accvgpr_write_b32 a6, 0                                  
	v_accvgpr_write_b32 a7, 0                                  
	s_add_u32 m0, 0x1080, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a8, 0                                  
	v_accvgpr_write_b32 a9, 0                                  
	v_accvgpr_write_b32 a10, 0                                 
	v_accvgpr_write_b32 a11, 0                                 
	v_accvgpr_write_b32 a12, 0                                 
	v_accvgpr_write_b32 a13, 0                                 
	v_accvgpr_write_b32 a14, 0                                 
	v_accvgpr_write_b32 a15, 0                                 
	s_add_u32 m0, 0x2100, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a16, 0                                 
	v_accvgpr_write_b32 a17, 0                                 
	v_accvgpr_write_b32 a18, 0                                 
	v_accvgpr_write_b32 a19, 0                                 
	v_accvgpr_write_b32 a20, 0                                 
	v_accvgpr_write_b32 a21, 0                                 
	v_accvgpr_write_b32 a22, 0                                 
	v_accvgpr_write_b32 a23, 0                                 
	s_add_u32 m0, 0x3180, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a24, 0                                 
	v_accvgpr_write_b32 a25, 0                                 
	v_accvgpr_write_b32 a26, 0                                 
	v_accvgpr_write_b32 a27, 0                                 
	v_accvgpr_write_b32 a28, 0                                 
	v_accvgpr_write_b32 a29, 0                                 
	v_accvgpr_write_b32 a30, 0                                 
	v_accvgpr_write_b32 a31, 0                                 
	s_add_u32 m0, 0x4200, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a32, 0                                 
	v_accvgpr_write_b32 a33, 0                                 
	v_accvgpr_write_b32 a34, 0                                 
	v_accvgpr_write_b32 a35, 0                                 
	v_accvgpr_write_b32 a36, 0                                 
	v_accvgpr_write_b32 a37, 0                                 
	v_accvgpr_write_b32 a38, 0                                 
	v_accvgpr_write_b32 a39, 0                                 
	s_add_u32 m0, 0x5280, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a40, 0                                 
	v_accvgpr_write_b32 a41, 0                                 
	v_accvgpr_write_b32 a42, 0                                 
	v_accvgpr_write_b32 a43, 0                                 
	v_accvgpr_write_b32 a44, 0                                 
	v_accvgpr_write_b32 a45, 0                                 
	v_accvgpr_write_b32 a46, 0                                 
	v_accvgpr_write_b32 a47, 0                                 
	s_add_u32 m0, 0x6300, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a48, 0                                 
	v_accvgpr_write_b32 a49, 0                                 
	v_accvgpr_write_b32 a50, 0                                 
	v_accvgpr_write_b32 a51, 0                                 
	v_accvgpr_write_b32 a52, 0                                 
	v_accvgpr_write_b32 a53, 0                                 
	v_accvgpr_write_b32 a54, 0                                 
	v_accvgpr_write_b32 a55, 0                                 
	s_add_u32 m0, 0x7380, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a56, 0                                 
	v_accvgpr_write_b32 a57, 0                                 
	v_accvgpr_write_b32 a58, 0                                 
	v_accvgpr_write_b32 a59, 0                                 
	v_accvgpr_write_b32 a60, 0                                 
	v_accvgpr_write_b32 a61, 0                                 
	v_accvgpr_write_b32 a62, 0                                 
	v_accvgpr_write_b32 a63, 0                                 
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_accvgpr_write_b32 a64, 0                                 
	v_accvgpr_write_b32 a65, 0                                 
	v_accvgpr_write_b32 a66, 0                                 
	v_accvgpr_write_b32 a67, 0                                 
	v_accvgpr_write_b32 a68, 0                                 
	v_accvgpr_write_b32 a69, 0                                 
	v_accvgpr_write_b32 a70, 0                                 
	v_accvgpr_write_b32 a71, 0                                 
	s_add_u32 m0, 0, s60                                       
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a72, 0                                 
	v_accvgpr_write_b32 a73, 0                                 
	v_accvgpr_write_b32 a74, 0                                 
	v_accvgpr_write_b32 a75, 0                                 
	v_accvgpr_write_b32 a76, 0                                 
	v_accvgpr_write_b32 a77, 0                                 
	v_accvgpr_write_b32 a78, 0                                 
	v_accvgpr_write_b32 a79, 0                                 
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a80, 0                                 
	v_accvgpr_write_b32 a81, 0                                 
	v_accvgpr_write_b32 a82, 0                                 
	v_accvgpr_write_b32 a83, 0                                 
	v_accvgpr_write_b32 a84, 0                                 
	v_accvgpr_write_b32 a85, 0                                 
	v_accvgpr_write_b32 a86, 0                                 
	v_accvgpr_write_b32 a87, 0                                 
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a88, 0                                 
	v_accvgpr_write_b32 a89, 0                                 
	v_accvgpr_write_b32 a90, 0                                 
	v_accvgpr_write_b32 a91, 0                                 
	v_accvgpr_write_b32 a92, 0                                 
	v_accvgpr_write_b32 a93, 0                                 
	v_accvgpr_write_b32 a94, 0                                 
	v_accvgpr_write_b32 a95, 0                                 
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a96, 0                                 
	v_accvgpr_write_b32 a97, 0                                 
	v_accvgpr_write_b32 a98, 0                                 
	v_accvgpr_write_b32 a99, 0                                 
	v_accvgpr_write_b32 a100, 0                                
	v_accvgpr_write_b32 a101, 0                                
	v_accvgpr_write_b32 a102, 0                                
	v_accvgpr_write_b32 a103, 0                                
	s_add_u32 m0, 0x4200, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a104, 0                                
	v_accvgpr_write_b32 a105, 0                                
	v_accvgpr_write_b32 a106, 0                                
	v_accvgpr_write_b32 a107, 0                                
	v_accvgpr_write_b32 a108, 0                                
	v_accvgpr_write_b32 a109, 0                                
	v_accvgpr_write_b32 a110, 0                                
	v_accvgpr_write_b32 a111, 0                                
	s_add_u32 m0, 0x5280, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a112, 0                                
	v_accvgpr_write_b32 a113, 0                                
	v_accvgpr_write_b32 a114, 0                                
	v_accvgpr_write_b32 a115, 0                                
	v_accvgpr_write_b32 a116, 0                                
	v_accvgpr_write_b32 a117, 0                                
	v_accvgpr_write_b32 a118, 0                                
	v_accvgpr_write_b32 a119, 0                                
	s_add_u32 m0, 0x6300, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a120, 0                                
	v_accvgpr_write_b32 a121, 0                                
	v_accvgpr_write_b32 a122, 0                                
	v_accvgpr_write_b32 a123, 0                                
	v_accvgpr_write_b32 a124, 0                                
	v_accvgpr_write_b32 a125, 0                                
	v_accvgpr_write_b32 a126, 0                                
	v_accvgpr_write_b32 a127, 0                                
	s_add_u32 m0, 0x7380, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a128, 0                                
	v_accvgpr_write_b32 a129, 0                                
	v_accvgpr_write_b32 a130, 0                                
	v_accvgpr_write_b32 a131, 0                                
	v_accvgpr_write_b32 a132, 0                                
	v_accvgpr_write_b32 a133, 0                                
	v_accvgpr_write_b32 a134, 0                                
	v_accvgpr_write_b32 a135, 0                                
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	s_add_u32 m0, 0x8400, s61                                  
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a136, 0                                
	v_accvgpr_write_b32 a137, 0                                
	v_accvgpr_write_b32 a138, 0                                
	v_accvgpr_write_b32 a139, 0                                
	v_accvgpr_write_b32 a140, 0                                
	v_accvgpr_write_b32 a141, 0                                
	v_accvgpr_write_b32 a142, 0                                
	v_accvgpr_write_b32 a143, 0                                
	s_add_u32 m0, 0x9480, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a144, 0                                
	v_accvgpr_write_b32 a145, 0                                
	v_accvgpr_write_b32 a146, 0                                
	v_accvgpr_write_b32 a147, 0                                
	v_accvgpr_write_b32 a148, 0                                
	v_accvgpr_write_b32 a149, 0                                
	v_accvgpr_write_b32 a150, 0                                
	v_accvgpr_write_b32 a151, 0                                
	s_add_u32 m0, 0xa500, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a152, 0                                
	v_accvgpr_write_b32 a153, 0                                
	v_accvgpr_write_b32 a154, 0                                
	v_accvgpr_write_b32 a155, 0                                
	v_accvgpr_write_b32 a156, 0                                
	v_accvgpr_write_b32 a157, 0                                
	v_accvgpr_write_b32 a158, 0                                
	v_accvgpr_write_b32 a159, 0                                
	s_add_u32 m0, 0xb580, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a160, 0                                
	v_accvgpr_write_b32 a161, 0                                
	v_accvgpr_write_b32 a162, 0                                
	v_accvgpr_write_b32 a163, 0                                
	v_accvgpr_write_b32 a164, 0                                
	v_accvgpr_write_b32 a165, 0                                
	v_accvgpr_write_b32 a166, 0                                
	v_accvgpr_write_b32 a167, 0                                
	s_add_u32 m0, 0xc600, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a168, 0                                
	v_accvgpr_write_b32 a169, 0                                
	v_accvgpr_write_b32 a170, 0                                
	v_accvgpr_write_b32 a171, 0                                
	v_accvgpr_write_b32 a172, 0                                
	v_accvgpr_write_b32 a173, 0                                
	v_accvgpr_write_b32 a174, 0                                
	v_accvgpr_write_b32 a175, 0                                
	s_add_u32 m0, 0xd680, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a176, 0                                
	v_accvgpr_write_b32 a177, 0                                
	v_accvgpr_write_b32 a178, 0                                
	v_accvgpr_write_b32 a179, 0                                
	v_accvgpr_write_b32 a180, 0                                
	v_accvgpr_write_b32 a181, 0                                
	v_accvgpr_write_b32 a182, 0                                
	v_accvgpr_write_b32 a183, 0                                
	s_add_u32 m0, 0xe700, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a184, 0                                
	v_accvgpr_write_b32 a185, 0                                
	v_accvgpr_write_b32 a186, 0                                
	v_accvgpr_write_b32 a187, 0                                
	v_accvgpr_write_b32 a188, 0                                
	v_accvgpr_write_b32 a189, 0                                
	v_accvgpr_write_b32 a190, 0                                
	v_accvgpr_write_b32 a191, 0                                
	s_add_u32 m0, 0xf780, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_accvgpr_write_b32 a192, 0                                
	v_accvgpr_write_b32 a193, 0                                
	v_accvgpr_write_b32 a194, 0                                
	v_accvgpr_write_b32 a195, 0                                
	v_accvgpr_write_b32 a196, 0                                
	v_accvgpr_write_b32 a197, 0                                
	v_accvgpr_write_b32 a198, 0                                
	v_accvgpr_write_b32 a199, 0                                
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_accvgpr_write_b32 a200, 0                                
	v_accvgpr_write_b32 a201, 0                                
	v_accvgpr_write_b32 a202, 0                                
	v_accvgpr_write_b32 a203, 0                                
	v_accvgpr_write_b32 a204, 0                                
	v_accvgpr_write_b32 a205, 0                                
	v_accvgpr_write_b32 a206, 0                                
	v_accvgpr_write_b32 a207, 0                                
	s_add_u32 m0, 0x8400, s60                                  
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a208, 0                                
	v_accvgpr_write_b32 a209, 0                                
	v_accvgpr_write_b32 a210, 0                                
	v_accvgpr_write_b32 a211, 0                                
	v_accvgpr_write_b32 a212, 0                                
	v_accvgpr_write_b32 a213, 0                                
	v_accvgpr_write_b32 a214, 0                                
	v_accvgpr_write_b32 a215, 0                                
	s_add_u32 m0, 0x9480, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a216, 0                                
	v_accvgpr_write_b32 a217, 0                                
	v_accvgpr_write_b32 a218, 0                                
	v_accvgpr_write_b32 a219, 0                                
	v_accvgpr_write_b32 a220, 0                                
	v_accvgpr_write_b32 a221, 0                                
	v_accvgpr_write_b32 a222, 0                                
	v_accvgpr_write_b32 a223, 0                                
	s_add_u32 m0, 0xa500, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a224, 0                                
	v_accvgpr_write_b32 a225, 0                                
	v_accvgpr_write_b32 a226, 0                                
	v_accvgpr_write_b32 a227, 0                                
	v_accvgpr_write_b32 a228, 0                                
	v_accvgpr_write_b32 a229, 0                                
	v_accvgpr_write_b32 a230, 0                                
	v_accvgpr_write_b32 a231, 0                                
	s_add_u32 m0, 0xb580, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a232, 0                                
	v_accvgpr_write_b32 a233, 0                                
	v_accvgpr_write_b32 a234, 0                                
	v_accvgpr_write_b32 a235, 0                                
	v_accvgpr_write_b32 a236, 0                                
	v_accvgpr_write_b32 a237, 0                                
	v_accvgpr_write_b32 a238, 0                                
	v_accvgpr_write_b32 a239, 0                                
	s_add_u32 m0, 0xc600, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a240, 0                                
	v_accvgpr_write_b32 a241, 0                                
	v_accvgpr_write_b32 a242, 0                                
	v_accvgpr_write_b32 a243, 0                                
	v_accvgpr_write_b32 a244, 0                                
	v_accvgpr_write_b32 a245, 0                                
	v_accvgpr_write_b32 a246, 0                                
	v_accvgpr_write_b32 a247, 0                                
	s_add_u32 m0, 0xd680, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_accvgpr_write_b32 a248, 0                                
	v_accvgpr_write_b32 a249, 0                                
	v_accvgpr_write_b32 a250, 0                                
	v_accvgpr_write_b32 a251, 0                                
	v_accvgpr_write_b32 a252, 0                                
	v_accvgpr_write_b32 a253, 0                                
	v_accvgpr_write_b32 a254, 0                                
	v_accvgpr_write_b32 a255, 0                                
	s_add_u32 m0, 0xe700, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	s_add_u32 m0, 0xf780, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	s_waitcnt vmcnt(20)                                        
	s_barrier                                                  
	ds_read_b128 v[136:139], v218                              
	ds_read_b128 v[152:155], v218 offset:64                    
	ds_read_b128 v[140:143], v218 offset:512                   
	ds_read_b128 v[156:159], v218 offset:576                   
	ds_read_b128 v[144:147], v218 offset:4224                  
	ds_read_b128 v[160:163], v218 offset:4288                  
	ds_read_b128 v[148:151], v218 offset:4736                  
	ds_read_b128 v[164:167], v218 offset:4800                  
	ds_read_b128 v[8:11], v208                                 
	ds_read_b128 v[40:43], v208 offset:64                      
	ds_read_b128 v[12:15], v208 offset:512                     
	ds_read_b128 v[44:47], v208 offset:576                     
	ds_read_b128 v[16:19], v208 offset:4224                    
	ds_read_b128 v[48:51], v208 offset:4288                    
	ds_read_b128 v[20:23], v208 offset:4736                    
	ds_read_b128 v[52:55], v208 offset:4800                    
	ds_read_b128 v[24:27], v208 offset:8448                    
	ds_read_b128 v[56:59], v208 offset:8512                    
	ds_read_b128 v[28:31], v208 offset:8960                    
	ds_read_b128 v[60:63], v208 offset:9024                    
	ds_read_b128 v[32:35], v208 offset:12672                   
	ds_read_b128 v[64:67], v208 offset:12736                   
	ds_read_b128 v[36:39], v208 offset:13184                   
	ds_read_b128 v[68:71], v208 offset:13248                   
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_nop 0                                                    
	s_mul_i32 s51, s47, 0x100                                  
	s_mul_hi_u32 s52, s51, s40                                 
	s_add_u32 s5, s5, s52                                      
	s_mul_i32 s52, s51, s40                                    
	s_add_u32 s4, s4, s52                                      
	s_addc_u32 s5, 0, s5                                       
	s_sub_i32 s51, s43, s51                                    
	s_mul_i32 s51, s51, s40                                    
	s_mov_b32 s6, s51                                          
	v_and_b32_e64 v220, v0, 15                                 
	v_mul_lo_u32 v220, v220, s40                               
	v_lshrrev_b32_e32 v4, 5, v0                                
	v_mul_i32_i24_e32 v4, 16, v4                               
	v_add_u32_e32 v220, v4, v220                               
	v_lshrrev_b32_e32 v4, 4, v0                                
	v_and_b32_e32 v4, 1, v4                                    
	v_mul_i32_i24_e32 v4, 32, v4                               
	v_add_u32_e32 v220, v4, v220                               
	s_mul_i32 s51, s46, 0x100                                  
	s_mul_i32 s52, s48, 64                                     
	s_add_i32 s51, s51, s52                                    
	s_lshl_b32 s51, s51, 1                                     
	v_add_u32_e32 v220, s51, v220                              
	s_mul_i32 s52, s40, 16                                     
	v_add_u32_e64 v221, v220, s52                              
	v_add_u32_e64 v222, v221, s52                              
	v_add_u32_e64 v223, v222, s52                              
	v_add_u32_e64 v224, v223, s52                              
	v_add_u32_e64 v225, v224, s52                              
	v_add_u32_e64 v226, v225, s52                              
	v_add_u32_e64 v227, v226, s52                              
	v_add_u32_e64 v228, v227, s52                              
	v_add_u32_e64 v229, v228, s52                              
	v_add_u32_e64 v230, v229, s52                              
	v_add_u32_e64 v231, v230, s52                              
	v_add_u32_e64 v232, v231, s52                              
	v_add_u32_e64 v233, v232, s52                              
	v_add_u32_e64 v234, v233, s52                              
	v_add_u32_e64 v235, v234, s52                              
	s_mov_b32 s49, 0                                           
	s_mov_b32 s50, s45                                         
	s_cmp_lt_u32 0x80, s50                                     
	s_cselect_b32 s62, s62, 0                                  
	s_cmp_lt_u32 64, s50                                       
	s_cselect_b32 s63, s63, 0                                  
	s_cmp_lt_i32 s48, 2                                        
	s_cbranch_scc0 label_075D                                  
	
label_040D:
	s_waitcnt vmcnt(8) lgkmcnt(0)                              
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[0:3], v[136:139], v[8:11], a[0:3]
	s_add_u32 m0, 0, s61                                       
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[4:7], v[136:139], v[12:15], a[4:7]
	ds_read_b128 v[72:75], v208 offset:16896                   
	v_mfma_f32_16x16x32_bf16 a[8:11], v[136:139], v[16:19], a[8:11]
	v_mfma_f32_16x16x32_bf16 a[12:15], v[136:139], v[20:23], a[12:15]
	ds_read_b128 v[104:107], v208 offset:16960                 
	v_mfma_f32_16x16x32_bf16 a[16:19], v[136:139], v[24:27], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[136:139], v[28:31], a[20:23]
	ds_read_b128 v[76:79], v208 offset:17408                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[136:139], v[32:35], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[136:139], v[36:39], a[28:31]
	ds_read_b128 v[108:111], v208 offset:17472                 
	v_mfma_f32_16x16x32_bf16 a[32:35], v[140:143], v[8:11], a[32:35]
	s_add_u32 m0, 0x1080, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[36:39], v[140:143], v[12:15], a[36:39]
	ds_read_b128 v[80:83], v208 offset:21120                   
	v_mfma_f32_16x16x32_bf16 a[40:43], v[140:143], v[16:19], a[40:43]
	v_mfma_f32_16x16x32_bf16 a[44:47], v[140:143], v[20:23], a[44:47]
	ds_read_b128 v[112:115], v208 offset:21184                 
	v_mfma_f32_16x16x32_bf16 a[48:51], v[140:143], v[24:27], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[140:143], v[28:31], a[52:55]
	ds_read_b128 v[84:87], v208 offset:21632                   
	v_mfma_f32_16x16x32_bf16 a[56:59], v[140:143], v[32:35], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[140:143], v[36:39], a[60:63]
	ds_read_b128 v[116:119], v208 offset:21696                 
	v_mfma_f32_16x16x32_bf16 a[64:67], v[144:147], v[8:11], a[64:67]
	s_add_u32 m0, 0x2100, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[68:71], v[144:147], v[12:15], a[68:71]
	ds_read_b128 v[88:91], v208 offset:25344                   
	v_mfma_f32_16x16x32_bf16 a[72:75], v[144:147], v[16:19], a[72:75]
	v_mfma_f32_16x16x32_bf16 a[76:79], v[144:147], v[20:23], a[76:79]
	ds_read_b128 v[120:123], v208 offset:25408                 
	v_mfma_f32_16x16x32_bf16 a[80:83], v[144:147], v[24:27], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[144:147], v[28:31], a[84:87]
	ds_read_b128 v[92:95], v208 offset:25856                   
	v_mfma_f32_16x16x32_bf16 a[88:91], v[144:147], v[32:35], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[144:147], v[36:39], a[92:95]
	ds_read_b128 v[124:127], v208 offset:25920                 
	v_mfma_f32_16x16x32_bf16 a[96:99], v[148:151], v[8:11], a[96:99]
	s_add_u32 m0, 0x3180, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[100:103], v[148:151], v[12:15], a[100:103]
	ds_read_b128 v[96:99], v208 offset:29568                   
	v_mfma_f32_16x16x32_bf16 a[104:107], v[148:151], v[16:19], a[104:107]
	v_mfma_f32_16x16x32_bf16 a[108:111], v[148:151], v[20:23], a[108:111]
	ds_read_b128 v[128:131], v208 offset:29632                 
	v_mfma_f32_16x16x32_bf16 a[112:115], v[148:151], v[24:27], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[148:151], v[28:31], a[116:119]
	ds_read_b128 v[100:103], v208 offset:30080                 
	v_mfma_f32_16x16x32_bf16 a[120:123], v[148:151], v[32:35], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[148:151], v[36:39], a[124:127]
	ds_read_b128 v[132:135], v208 offset:30144                 
	v_mfma_f32_16x16x32_bf16 a[0:3], v[152:155], v[40:43], a[0:3]
	s_add_u32 m0, 0x4200, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[4:7], v[152:155], v[44:47], a[4:7]
	ds_read_b128 v[168:171], v219                              
	v_mfma_f32_16x16x32_bf16 a[8:11], v[152:155], v[48:51], a[8:11]
	v_mfma_f32_16x16x32_bf16 a[12:15], v[152:155], v[52:55], a[12:15]
	ds_read_b128 v[184:187], v219 offset:64                    
	v_mfma_f32_16x16x32_bf16 a[16:19], v[152:155], v[56:59], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[152:155], v[60:63], a[20:23]
	ds_read_b128 v[172:175], v219 offset:512                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[152:155], v[64:67], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[152:155], v[68:71], a[28:31]
	ds_read_b128 v[188:191], v219 offset:576                   
	v_mfma_f32_16x16x32_bf16 a[32:35], v[156:159], v[40:43], a[32:35]
	s_add_u32 m0, 0x5280, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[36:39], v[156:159], v[44:47], a[36:39]
	ds_read_b128 v[176:179], v219 offset:4224                  
	v_mfma_f32_16x16x32_bf16 a[40:43], v[156:159], v[48:51], a[40:43]
	v_mfma_f32_16x16x32_bf16 a[44:47], v[156:159], v[52:55], a[44:47]
	ds_read_b128 v[192:195], v219 offset:4288                  
	v_mfma_f32_16x16x32_bf16 a[48:51], v[156:159], v[56:59], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[156:159], v[60:63], a[52:55]
	ds_read_b128 v[180:183], v219 offset:4736                  
	v_mfma_f32_16x16x32_bf16 a[56:59], v[156:159], v[64:67], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[156:159], v[68:71], a[60:63]
	ds_read_b128 v[196:199], v219 offset:4800                  
	v_mfma_f32_16x16x32_bf16 a[64:67], v[160:163], v[40:43], a[64:67]
	s_add_u32 m0, 0x6300, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[68:71], v[160:163], v[44:47], a[68:71]
	v_mfma_f32_16x16x32_bf16 a[72:75], v[160:163], v[48:51], a[72:75]
	v_mfma_f32_16x16x32_bf16 a[76:79], v[160:163], v[52:55], a[76:79]
	v_mfma_f32_16x16x32_bf16 a[80:83], v[160:163], v[56:59], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[160:163], v[60:63], a[84:87]
	v_mfma_f32_16x16x32_bf16 a[88:91], v[160:163], v[64:67], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[160:163], v[68:71], a[92:95]
	v_mfma_f32_16x16x32_bf16 a[96:99], v[164:167], v[40:43], a[96:99]
	s_add_u32 m0, 0x7380, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[100:103], v[164:167], v[44:47], a[100:103]
	v_mfma_f32_16x16x32_bf16 a[104:107], v[164:167], v[48:51], a[104:107]
	s_add_u32 s52, 0x80, s49                                   
	s_cmp_lt_u32 s52, s50                                      
	s_cselect_b32 s63, s63, 0                                  
	v_mfma_f32_16x16x32_bf16 a[108:111], v[164:167], v[52:55], a[108:111]
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_mfma_f32_16x16x32_bf16 a[112:115], v[164:167], v[56:59], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[164:167], v[60:63], a[116:119]
	v_mfma_f32_16x16x32_bf16 a[120:123], v[164:167], v[64:67], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[164:167], v[68:71], a[124:127]
	s_waitcnt vmcnt(12) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[128:131], v[136:139], v[72:75], a[128:131]
	s_add_u32 m0, 0, s60                                       
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[132:135], v[136:139], v[76:79], a[132:135]
	ds_read_b128 v[8:11], v209                                 
	v_mfma_f32_16x16x32_bf16 a[136:139], v[136:139], v[80:83], a[136:139]
	v_mfma_f32_16x16x32_bf16 a[140:143], v[136:139], v[84:87], a[140:143]
	ds_read_b128 v[40:43], v209 offset:64                      
	v_mfma_f32_16x16x32_bf16 a[144:147], v[136:139], v[88:91], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[136:139], v[92:95], a[148:151]
	ds_read_b128 v[12:15], v209 offset:512                     
	v_mfma_f32_16x16x32_bf16 a[152:155], v[136:139], v[96:99], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[136:139], v[100:103], a[156:159]
	ds_read_b128 v[44:47], v209 offset:576                     
	v_mfma_f32_16x16x32_bf16 a[160:163], v[140:143], v[72:75], a[160:163]
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[164:167], v[140:143], v[76:79], a[164:167]
	ds_read_b128 v[16:19], v209 offset:4224                    
	v_mfma_f32_16x16x32_bf16 a[168:171], v[140:143], v[80:83], a[168:171]
	v_mfma_f32_16x16x32_bf16 a[172:175], v[140:143], v[84:87], a[172:175]
	ds_read_b128 v[48:51], v209 offset:4288                    
	v_mfma_f32_16x16x32_bf16 a[176:179], v[140:143], v[88:91], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[140:143], v[92:95], a[180:183]
	ds_read_b128 v[20:23], v209 offset:4736                    
	v_mfma_f32_16x16x32_bf16 a[184:187], v[140:143], v[96:99], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[140:143], v[100:103], a[188:191]
	ds_read_b128 v[52:55], v209 offset:4800                    
	v_mfma_f32_16x16x32_bf16 a[192:195], v[144:147], v[72:75], a[192:195]
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[196:199], v[144:147], v[76:79], a[196:199]
	ds_read_b128 v[24:27], v209 offset:8448                    
	v_mfma_f32_16x16x32_bf16 a[200:203], v[144:147], v[80:83], a[200:203]
	v_mfma_f32_16x16x32_bf16 a[204:207], v[144:147], v[84:87], a[204:207]
	ds_read_b128 v[56:59], v209 offset:8512                    
	v_mfma_f32_16x16x32_bf16 a[208:211], v[144:147], v[88:91], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[144:147], v[92:95], a[212:215]
	ds_read_b128 v[28:31], v209 offset:8960                    
	v_mfma_f32_16x16x32_bf16 a[216:219], v[144:147], v[96:99], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[144:147], v[100:103], a[220:223]
	ds_read_b128 v[60:63], v209 offset:9024                    
	v_mfma_f32_16x16x32_bf16 a[224:227], v[148:151], v[72:75], a[224:227]
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[228:231], v[148:151], v[76:79], a[228:231]
	ds_read_b128 v[32:35], v209 offset:12672                   
	v_mfma_f32_16x16x32_bf16 a[232:235], v[148:151], v[80:83], a[232:235]
	v_mfma_f32_16x16x32_bf16 a[236:239], v[148:151], v[84:87], a[236:239]
	ds_read_b128 v[64:67], v209 offset:12736                   
	v_mfma_f32_16x16x32_bf16 a[240:243], v[148:151], v[88:91], a[240:243]
	v_mfma_f32_16x16x32_bf16 a[244:247], v[148:151], v[92:95], a[244:247]
	ds_read_b128 v[36:39], v209 offset:13184                   
	v_mfma_f32_16x16x32_bf16 a[248:251], v[148:151], v[96:99], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[148:151], v[100:103], a[252:255]
	ds_read_b128 v[68:71], v209 offset:13248                   
	v_mfma_f32_16x16x32_bf16 a[128:131], v[152:155], v[104:107], a[128:131]
	s_add_u32 m0, 0x4200, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[132:135], v[152:155], v[108:111], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[136:139], v[152:155], v[112:115], a[136:139]
	v_mfma_f32_16x16x32_bf16 a[140:143], v[152:155], v[116:119], a[140:143]
	v_mfma_f32_16x16x32_bf16 a[144:147], v[152:155], v[120:123], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[152:155], v[124:127], a[148:151]
	v_mfma_f32_16x16x32_bf16 a[152:155], v[152:155], v[128:131], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[152:155], v[132:135], a[156:159]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[156:159], v[104:107], a[160:163]
	s_add_u32 m0, 0x5280, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[164:167], v[156:159], v[108:111], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[168:171], v[156:159], v[112:115], a[168:171]
	v_mfma_f32_16x16x32_bf16 a[172:175], v[156:159], v[116:119], a[172:175]
	v_mfma_f32_16x16x32_bf16 a[176:179], v[156:159], v[120:123], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[156:159], v[124:127], a[180:183]
	v_mfma_f32_16x16x32_bf16 a[184:187], v[156:159], v[128:131], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[156:159], v[132:135], a[188:191]
	v_mfma_f32_16x16x32_bf16 a[192:195], v[160:163], v[104:107], a[192:195]
	s_add_u32 m0, 0x6300, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[196:199], v[160:163], v[108:111], a[196:199]
	v_mfma_f32_16x16x32_bf16 a[200:203], v[160:163], v[112:115], a[200:203]
	v_mfma_f32_16x16x32_bf16 a[204:207], v[160:163], v[116:119], a[204:207]
	v_mfma_f32_16x16x32_bf16 a[208:211], v[160:163], v[120:123], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[160:163], v[124:127], a[212:215]
	v_mfma_f32_16x16x32_bf16 a[216:219], v[160:163], v[128:131], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[160:163], v[132:135], a[220:223]
	v_mfma_f32_16x16x32_bf16 a[224:227], v[164:167], v[104:107], a[224:227]
	s_add_u32 m0, 0x7380, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[228:231], v[164:167], v[108:111], a[228:231]
	s_add_u32 s51, 0xc0, s49                                   
	s_cmp_lt_u32 s51, s50                                      
	s_cselect_b32 s62, s62, 0                                  
	v_mfma_f32_16x16x32_bf16 a[232:235], v[164:167], v[112:115], a[232:235]
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	v_mfma_f32_16x16x32_bf16 a[236:239], v[164:167], v[116:119], a[236:239]
	v_mfma_f32_16x16x32_bf16 a[240:243], v[164:167], v[120:123], a[240:243]
	s_addk_i32 s49, 0x40                                       
	s_cmp_lt_i32 s49, s50                                      
	v_mfma_f32_16x16x32_bf16 a[244:247], v[164:167], v[124:127], a[244:247]
	v_mfma_f32_16x16x32_bf16 a[248:251], v[164:167], v[128:131], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[164:167], v[132:135], a[252:255]
	s_cbranch_scc0 label_0AA6                                  
	s_waitcnt vmcnt(8) lgkmcnt(0)                              
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[0:3], v[168:171], v[8:11], a[0:3]
	s_add_u32 m0, 0x8400, s61                                  
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[4:7], v[168:171], v[12:15], a[4:7]
	ds_read_b128 v[72:75], v209 offset:16896                   
	v_mfma_f32_16x16x32_bf16 a[8:11], v[168:171], v[16:19], a[8:11]
	v_mfma_f32_16x16x32_bf16 a[12:15], v[168:171], v[20:23], a[12:15]
	ds_read_b128 v[104:107], v209 offset:16960                 
	v_mfma_f32_16x16x32_bf16 a[16:19], v[168:171], v[24:27], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[168:171], v[28:31], a[20:23]
	ds_read_b128 v[76:79], v209 offset:17408                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[168:171], v[32:35], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[168:171], v[36:39], a[28:31]
	ds_read_b128 v[108:111], v209 offset:17472                 
	v_mfma_f32_16x16x32_bf16 a[32:35], v[172:175], v[8:11], a[32:35]
	s_add_u32 m0, 0x9480, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[36:39], v[172:175], v[12:15], a[36:39]
	ds_read_b128 v[80:83], v209 offset:21120                   
	v_mfma_f32_16x16x32_bf16 a[40:43], v[172:175], v[16:19], a[40:43]
	v_mfma_f32_16x16x32_bf16 a[44:47], v[172:175], v[20:23], a[44:47]
	ds_read_b128 v[112:115], v209 offset:21184                 
	v_mfma_f32_16x16x32_bf16 a[48:51], v[172:175], v[24:27], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[172:175], v[28:31], a[52:55]
	ds_read_b128 v[84:87], v209 offset:21632                   
	v_mfma_f32_16x16x32_bf16 a[56:59], v[172:175], v[32:35], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[172:175], v[36:39], a[60:63]
	ds_read_b128 v[116:119], v209 offset:21696                 
	v_mfma_f32_16x16x32_bf16 a[64:67], v[176:179], v[8:11], a[64:67]
	s_add_u32 m0, 0xa500, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[68:71], v[176:179], v[12:15], a[68:71]
	ds_read_b128 v[88:91], v209 offset:25344                   
	v_mfma_f32_16x16x32_bf16 a[72:75], v[176:179], v[16:19], a[72:75]
	v_mfma_f32_16x16x32_bf16 a[76:79], v[176:179], v[20:23], a[76:79]
	ds_read_b128 v[120:123], v209 offset:25408                 
	v_mfma_f32_16x16x32_bf16 a[80:83], v[176:179], v[24:27], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[176:179], v[28:31], a[84:87]
	ds_read_b128 v[92:95], v209 offset:25856                   
	v_mfma_f32_16x16x32_bf16 a[88:91], v[176:179], v[32:35], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[176:179], v[36:39], a[92:95]
	ds_read_b128 v[124:127], v209 offset:25920                 
	v_mfma_f32_16x16x32_bf16 a[96:99], v[180:183], v[8:11], a[96:99]
	s_add_u32 m0, 0xb580, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[100:103], v[180:183], v[12:15], a[100:103]
	ds_read_b128 v[96:99], v209 offset:29568                   
	v_mfma_f32_16x16x32_bf16 a[104:107], v[180:183], v[16:19], a[104:107]
	v_mfma_f32_16x16x32_bf16 a[108:111], v[180:183], v[20:23], a[108:111]
	ds_read_b128 v[128:131], v209 offset:29632                 
	v_mfma_f32_16x16x32_bf16 a[112:115], v[180:183], v[24:27], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[180:183], v[28:31], a[116:119]
	ds_read_b128 v[100:103], v209 offset:30080                 
	v_mfma_f32_16x16x32_bf16 a[120:123], v[180:183], v[32:35], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[180:183], v[36:39], a[124:127]
	ds_read_b128 v[132:135], v209 offset:30144                 
	v_mfma_f32_16x16x32_bf16 a[0:3], v[184:187], v[40:43], a[0:3]
	s_add_u32 m0, 0xc600, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[4:7], v[184:187], v[44:47], a[4:7]
	ds_read_b128 v[136:139], v218                              
	v_mfma_f32_16x16x32_bf16 a[8:11], v[184:187], v[48:51], a[8:11]
	v_mfma_f32_16x16x32_bf16 a[12:15], v[184:187], v[52:55], a[12:15]
	ds_read_b128 v[152:155], v218 offset:64                    
	v_mfma_f32_16x16x32_bf16 a[16:19], v[184:187], v[56:59], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[184:187], v[60:63], a[20:23]
	ds_read_b128 v[140:143], v218 offset:512                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[184:187], v[64:67], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[184:187], v[68:71], a[28:31]
	ds_read_b128 v[156:159], v218 offset:576                   
	v_mfma_f32_16x16x32_bf16 a[32:35], v[188:191], v[40:43], a[32:35]
	s_add_u32 m0, 0xd680, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[36:39], v[188:191], v[44:47], a[36:39]
	ds_read_b128 v[144:147], v218 offset:4224                  
	v_mfma_f32_16x16x32_bf16 a[40:43], v[188:191], v[48:51], a[40:43]
	v_mfma_f32_16x16x32_bf16 a[44:47], v[188:191], v[52:55], a[44:47]
	ds_read_b128 v[160:163], v218 offset:4288                  
	v_mfma_f32_16x16x32_bf16 a[48:51], v[188:191], v[56:59], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[188:191], v[60:63], a[52:55]
	ds_read_b128 v[148:151], v218 offset:4736                  
	v_mfma_f32_16x16x32_bf16 a[56:59], v[188:191], v[64:67], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[188:191], v[68:71], a[60:63]
	ds_read_b128 v[164:167], v218 offset:4800                  
	v_mfma_f32_16x16x32_bf16 a[64:67], v[192:195], v[40:43], a[64:67]
	s_add_u32 m0, 0xe700, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[68:71], v[192:195], v[44:47], a[68:71]
	v_mfma_f32_16x16x32_bf16 a[72:75], v[192:195], v[48:51], a[72:75]
	v_mfma_f32_16x16x32_bf16 a[76:79], v[192:195], v[52:55], a[76:79]
	v_mfma_f32_16x16x32_bf16 a[80:83], v[192:195], v[56:59], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[192:195], v[60:63], a[84:87]
	v_mfma_f32_16x16x32_bf16 a[88:91], v[192:195], v[64:67], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[192:195], v[68:71], a[92:95]
	v_mfma_f32_16x16x32_bf16 a[96:99], v[196:199], v[40:43], a[96:99]
	s_add_u32 m0, 0xf780, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[100:103], v[196:199], v[44:47], a[100:103]
	v_mfma_f32_16x16x32_bf16 a[104:107], v[196:199], v[48:51], a[104:107]
	s_add_u32 s52, 0x80, s49                                   
	s_cmp_lt_u32 s52, s50                                      
	s_cselect_b32 s63, s63, 0                                  
	v_mfma_f32_16x16x32_bf16 a[108:111], v[196:199], v[52:55], a[108:111]
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_mfma_f32_16x16x32_bf16 a[112:115], v[196:199], v[56:59], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[196:199], v[60:63], a[116:119]
	v_mfma_f32_16x16x32_bf16 a[120:123], v[196:199], v[64:67], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[196:199], v[68:71], a[124:127]
	s_waitcnt vmcnt(12) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[128:131], v[168:171], v[72:75], a[128:131]
	s_add_u32 m0, 0x8400, s60                                  
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[132:135], v[168:171], v[76:79], a[132:135]
	ds_read_b128 v[8:11], v208                                 
	v_mfma_f32_16x16x32_bf16 a[136:139], v[168:171], v[80:83], a[136:139]
	v_mfma_f32_16x16x32_bf16 a[140:143], v[168:171], v[84:87], a[140:143]
	ds_read_b128 v[40:43], v208 offset:64                      
	v_mfma_f32_16x16x32_bf16 a[144:147], v[168:171], v[88:91], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[168:171], v[92:95], a[148:151]
	ds_read_b128 v[12:15], v208 offset:512                     
	v_mfma_f32_16x16x32_bf16 a[152:155], v[168:171], v[96:99], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[168:171], v[100:103], a[156:159]
	ds_read_b128 v[44:47], v208 offset:576                     
	v_mfma_f32_16x16x32_bf16 a[160:163], v[172:175], v[72:75], a[160:163]
	s_add_u32 m0, 0x9480, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[164:167], v[172:175], v[76:79], a[164:167]
	ds_read_b128 v[16:19], v208 offset:4224                    
	v_mfma_f32_16x16x32_bf16 a[168:171], v[172:175], v[80:83], a[168:171]
	v_mfma_f32_16x16x32_bf16 a[172:175], v[172:175], v[84:87], a[172:175]
	ds_read_b128 v[48:51], v208 offset:4288                    
	v_mfma_f32_16x16x32_bf16 a[176:179], v[172:175], v[88:91], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[172:175], v[92:95], a[180:183]
	ds_read_b128 v[20:23], v208 offset:4736                    
	v_mfma_f32_16x16x32_bf16 a[184:187], v[172:175], v[96:99], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[172:175], v[100:103], a[188:191]
	ds_read_b128 v[52:55], v208 offset:4800                    
	v_mfma_f32_16x16x32_bf16 a[192:195], v[176:179], v[72:75], a[192:195]
	s_add_u32 m0, 0xa500, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[196:199], v[176:179], v[76:79], a[196:199]
	ds_read_b128 v[24:27], v208 offset:8448                    
	v_mfma_f32_16x16x32_bf16 a[200:203], v[176:179], v[80:83], a[200:203]
	v_mfma_f32_16x16x32_bf16 a[204:207], v[176:179], v[84:87], a[204:207]
	ds_read_b128 v[56:59], v208 offset:8512                    
	v_mfma_f32_16x16x32_bf16 a[208:211], v[176:179], v[88:91], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[176:179], v[92:95], a[212:215]
	ds_read_b128 v[28:31], v208 offset:8960                    
	v_mfma_f32_16x16x32_bf16 a[216:219], v[176:179], v[96:99], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[176:179], v[100:103], a[220:223]
	ds_read_b128 v[60:63], v208 offset:9024                    
	v_mfma_f32_16x16x32_bf16 a[224:227], v[180:183], v[72:75], a[224:227]
	s_add_u32 m0, 0xb580, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[228:231], v[180:183], v[76:79], a[228:231]
	ds_read_b128 v[32:35], v208 offset:12672                   
	v_mfma_f32_16x16x32_bf16 a[232:235], v[180:183], v[80:83], a[232:235]
	v_mfma_f32_16x16x32_bf16 a[236:239], v[180:183], v[84:87], a[236:239]
	ds_read_b128 v[64:67], v208 offset:12736                   
	v_mfma_f32_16x16x32_bf16 a[240:243], v[180:183], v[88:91], a[240:243]
	v_mfma_f32_16x16x32_bf16 a[244:247], v[180:183], v[92:95], a[244:247]
	ds_read_b128 v[36:39], v208 offset:13184                   
	v_mfma_f32_16x16x32_bf16 a[248:251], v[180:183], v[96:99], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[180:183], v[100:103], a[252:255]
	ds_read_b128 v[68:71], v208 offset:13248                   
	v_mfma_f32_16x16x32_bf16 a[128:131], v[184:187], v[104:107], a[128:131]
	s_add_u32 m0, 0xc600, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[132:135], v[184:187], v[108:111], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[136:139], v[184:187], v[112:115], a[136:139]
	v_mfma_f32_16x16x32_bf16 a[140:143], v[184:187], v[116:119], a[140:143]
	v_mfma_f32_16x16x32_bf16 a[144:147], v[184:187], v[120:123], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[184:187], v[124:127], a[148:151]
	v_mfma_f32_16x16x32_bf16 a[152:155], v[184:187], v[128:131], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[184:187], v[132:135], a[156:159]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[188:191], v[104:107], a[160:163]
	s_add_u32 m0, 0xd680, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[164:167], v[188:191], v[108:111], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[168:171], v[188:191], v[112:115], a[168:171]
	v_mfma_f32_16x16x32_bf16 a[172:175], v[188:191], v[116:119], a[172:175]
	v_mfma_f32_16x16x32_bf16 a[176:179], v[188:191], v[120:123], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[188:191], v[124:127], a[180:183]
	v_mfma_f32_16x16x32_bf16 a[184:187], v[188:191], v[128:131], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[188:191], v[132:135], a[188:191]
	v_mfma_f32_16x16x32_bf16 a[192:195], v[192:195], v[104:107], a[192:195]
	s_add_u32 m0, 0xe700, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[196:199], v[192:195], v[108:111], a[196:199]
	v_mfma_f32_16x16x32_bf16 a[200:203], v[192:195], v[112:115], a[200:203]
	v_mfma_f32_16x16x32_bf16 a[204:207], v[192:195], v[116:119], a[204:207]
	v_mfma_f32_16x16x32_bf16 a[208:211], v[192:195], v[120:123], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[192:195], v[124:127], a[212:215]
	v_mfma_f32_16x16x32_bf16 a[216:219], v[192:195], v[128:131], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[192:195], v[132:135], a[220:223]
	v_mfma_f32_16x16x32_bf16 a[224:227], v[196:199], v[104:107], a[224:227]
	s_add_u32 m0, 0xf780, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[228:231], v[196:199], v[108:111], a[228:231]
	s_add_u32 s51, 0xc0, s49                                   
	s_cmp_lt_u32 s51, s50                                      
	s_cselect_b32 s62, s62, 0                                  
	v_mfma_f32_16x16x32_bf16 a[232:235], v[196:199], v[112:115], a[232:235]
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	v_mfma_f32_16x16x32_bf16 a[236:239], v[196:199], v[116:119], a[236:239]
	v_mfma_f32_16x16x32_bf16 a[240:243], v[196:199], v[120:123], a[240:243]
	s_addk_i32 s49, 0x40                                       
	s_cmp_lt_i32 s49, s50                                      
	v_mfma_f32_16x16x32_bf16 a[244:247], v[196:199], v[124:127], a[244:247]
	v_mfma_f32_16x16x32_bf16 a[248:251], v[196:199], v[128:131], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[196:199], v[132:135], a[252:255]
	s_cbranch_scc0 label_0AA6                                  
	s_branch label_040D                                        
	
label_075D:
	s_waitcnt vmcnt(8) lgkmcnt(0)                              
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[0:3], v[136:139], v[8:11], a[0:3]
	v_mfma_f32_16x16x32_bf16 a[4:7], v[136:139], v[12:15], a[4:7]
	ds_read_b128 v[72:75], v208 offset:16896                   
	v_mfma_f32_16x16x32_bf16 a[8:11], v[136:139], v[16:19], a[8:11]
	s_add_u32 m0, 0, s61                                       
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[12:15], v[136:139], v[20:23], a[12:15]
	ds_read_b128 v[104:107], v208 offset:16960                 
	v_mfma_f32_16x16x32_bf16 a[16:19], v[136:139], v[24:27], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[136:139], v[28:31], a[20:23]
	ds_read_b128 v[76:79], v208 offset:17408                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[136:139], v[32:35], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[136:139], v[36:39], a[28:31]
	ds_read_b128 v[108:111], v208 offset:17472                 
	v_mfma_f32_16x16x32_bf16 a[32:35], v[140:143], v[8:11], a[32:35]
	v_mfma_f32_16x16x32_bf16 a[36:39], v[140:143], v[12:15], a[36:39]
	ds_read_b128 v[80:83], v208 offset:21120                   
	v_mfma_f32_16x16x32_bf16 a[40:43], v[140:143], v[16:19], a[40:43]
	s_add_u32 m0, 0x1080, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[44:47], v[140:143], v[20:23], a[44:47]
	ds_read_b128 v[112:115], v208 offset:21184                 
	v_mfma_f32_16x16x32_bf16 a[48:51], v[140:143], v[24:27], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[140:143], v[28:31], a[52:55]
	ds_read_b128 v[84:87], v208 offset:21632                   
	v_mfma_f32_16x16x32_bf16 a[56:59], v[140:143], v[32:35], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[140:143], v[36:39], a[60:63]
	ds_read_b128 v[116:119], v208 offset:21696                 
	v_mfma_f32_16x16x32_bf16 a[64:67], v[144:147], v[8:11], a[64:67]
	v_mfma_f32_16x16x32_bf16 a[68:71], v[144:147], v[12:15], a[68:71]
	ds_read_b128 v[88:91], v208 offset:25344                   
	v_mfma_f32_16x16x32_bf16 a[72:75], v[144:147], v[16:19], a[72:75]
	s_add_u32 m0, 0x2100, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[76:79], v[144:147], v[20:23], a[76:79]
	ds_read_b128 v[120:123], v208 offset:25408                 
	v_mfma_f32_16x16x32_bf16 a[80:83], v[144:147], v[24:27], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[144:147], v[28:31], a[84:87]
	ds_read_b128 v[92:95], v208 offset:25856                   
	v_mfma_f32_16x16x32_bf16 a[88:91], v[144:147], v[32:35], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[144:147], v[36:39], a[92:95]
	ds_read_b128 v[124:127], v208 offset:25920                 
	v_mfma_f32_16x16x32_bf16 a[96:99], v[148:151], v[8:11], a[96:99]
	v_mfma_f32_16x16x32_bf16 a[100:103], v[148:151], v[12:15], a[100:103]
	ds_read_b128 v[96:99], v208 offset:29568                   
	v_mfma_f32_16x16x32_bf16 a[104:107], v[148:151], v[16:19], a[104:107]
	s_add_u32 m0, 0x3180, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[108:111], v[148:151], v[20:23], a[108:111]
	ds_read_b128 v[128:131], v208 offset:29632                 
	v_mfma_f32_16x16x32_bf16 a[112:115], v[148:151], v[24:27], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[148:151], v[28:31], a[116:119]
	ds_read_b128 v[100:103], v208 offset:30080                 
	v_mfma_f32_16x16x32_bf16 a[120:123], v[148:151], v[32:35], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[148:151], v[36:39], a[124:127]
	ds_read_b128 v[132:135], v208 offset:30144                 
	v_mfma_f32_16x16x32_bf16 a[0:3], v[152:155], v[40:43], a[0:3]
	v_mfma_f32_16x16x32_bf16 a[4:7], v[152:155], v[44:47], a[4:7]
	ds_read_b128 v[168:171], v219                              
	v_mfma_f32_16x16x32_bf16 a[8:11], v[152:155], v[48:51], a[8:11]
	s_add_u32 m0, 0x4200, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[12:15], v[152:155], v[52:55], a[12:15]
	ds_read_b128 v[184:187], v219 offset:64                    
	v_mfma_f32_16x16x32_bf16 a[16:19], v[152:155], v[56:59], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[152:155], v[60:63], a[20:23]
	ds_read_b128 v[172:175], v219 offset:512                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[152:155], v[64:67], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[152:155], v[68:71], a[28:31]
	ds_read_b128 v[188:191], v219 offset:576                   
	v_mfma_f32_16x16x32_bf16 a[32:35], v[156:159], v[40:43], a[32:35]
	v_mfma_f32_16x16x32_bf16 a[36:39], v[156:159], v[44:47], a[36:39]
	ds_read_b128 v[176:179], v219 offset:4224                  
	v_mfma_f32_16x16x32_bf16 a[40:43], v[156:159], v[48:51], a[40:43]
	s_add_u32 m0, 0x5280, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[44:47], v[156:159], v[52:55], a[44:47]
	ds_read_b128 v[192:195], v219 offset:4288                  
	v_mfma_f32_16x16x32_bf16 a[48:51], v[156:159], v[56:59], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[156:159], v[60:63], a[52:55]
	ds_read_b128 v[180:183], v219 offset:4736                  
	v_mfma_f32_16x16x32_bf16 a[56:59], v[156:159], v[64:67], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[156:159], v[68:71], a[60:63]
	ds_read_b128 v[196:199], v219 offset:4800                  
	v_mfma_f32_16x16x32_bf16 a[64:67], v[160:163], v[40:43], a[64:67]
	v_mfma_f32_16x16x32_bf16 a[68:71], v[160:163], v[44:47], a[68:71]
	v_mfma_f32_16x16x32_bf16 a[72:75], v[160:163], v[48:51], a[72:75]
	s_add_u32 m0, 0x6300, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[76:79], v[160:163], v[52:55], a[76:79]
	v_mfma_f32_16x16x32_bf16 a[80:83], v[160:163], v[56:59], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[160:163], v[60:63], a[84:87]
	v_mfma_f32_16x16x32_bf16 a[88:91], v[160:163], v[64:67], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[160:163], v[68:71], a[92:95]
	v_mfma_f32_16x16x32_bf16 a[96:99], v[164:167], v[40:43], a[96:99]
	v_mfma_f32_16x16x32_bf16 a[100:103], v[164:167], v[44:47], a[100:103]
	v_mfma_f32_16x16x32_bf16 a[104:107], v[164:167], v[48:51], a[104:107]
	s_add_u32 m0, 0x7380, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[108:111], v[164:167], v[52:55], a[108:111]
	s_add_u32 s52, 0x80, s49                                   
	s_cmp_lt_u32 s52, s50                                      
	s_cselect_b32 s63, s63, 0                                  
	v_mfma_f32_16x16x32_bf16 a[112:115], v[164:167], v[56:59], a[112:115]
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_mfma_f32_16x16x32_bf16 a[116:119], v[164:167], v[60:63], a[116:119]
	v_mfma_f32_16x16x32_bf16 a[120:123], v[164:167], v[64:67], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[164:167], v[68:71], a[124:127]
	s_waitcnt vmcnt(12) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[128:131], v[136:139], v[72:75], a[128:131]
	v_mfma_f32_16x16x32_bf16 a[132:135], v[136:139], v[76:79], a[132:135]
	ds_read_b128 v[8:11], v209                                 
	v_mfma_f32_16x16x32_bf16 a[136:139], v[136:139], v[80:83], a[136:139]
	s_add_u32 m0, 0, s60                                       
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[140:143], v[136:139], v[84:87], a[140:143]
	ds_read_b128 v[40:43], v209 offset:64                      
	v_mfma_f32_16x16x32_bf16 a[144:147], v[136:139], v[88:91], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[136:139], v[92:95], a[148:151]
	ds_read_b128 v[12:15], v209 offset:512                     
	v_mfma_f32_16x16x32_bf16 a[152:155], v[136:139], v[96:99], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[136:139], v[100:103], a[156:159]
	ds_read_b128 v[44:47], v209 offset:576                     
	v_mfma_f32_16x16x32_bf16 a[160:163], v[140:143], v[72:75], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[140:143], v[76:79], a[164:167]
	ds_read_b128 v[16:19], v209 offset:4224                    
	v_mfma_f32_16x16x32_bf16 a[168:171], v[140:143], v[80:83], a[168:171]
	s_add_u32 m0, 0x1080, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[172:175], v[140:143], v[84:87], a[172:175]
	ds_read_b128 v[48:51], v209 offset:4288                    
	v_mfma_f32_16x16x32_bf16 a[176:179], v[140:143], v[88:91], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[140:143], v[92:95], a[180:183]
	ds_read_b128 v[20:23], v209 offset:4736                    
	v_mfma_f32_16x16x32_bf16 a[184:187], v[140:143], v[96:99], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[140:143], v[100:103], a[188:191]
	ds_read_b128 v[52:55], v209 offset:4800                    
	v_mfma_f32_16x16x32_bf16 a[192:195], v[144:147], v[72:75], a[192:195]
	v_mfma_f32_16x16x32_bf16 a[196:199], v[144:147], v[76:79], a[196:199]
	ds_read_b128 v[24:27], v209 offset:8448                    
	v_mfma_f32_16x16x32_bf16 a[200:203], v[144:147], v[80:83], a[200:203]
	s_add_u32 m0, 0x2100, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[204:207], v[144:147], v[84:87], a[204:207]
	ds_read_b128 v[56:59], v209 offset:8512                    
	v_mfma_f32_16x16x32_bf16 a[208:211], v[144:147], v[88:91], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[144:147], v[92:95], a[212:215]
	ds_read_b128 v[28:31], v209 offset:8960                    
	v_mfma_f32_16x16x32_bf16 a[216:219], v[144:147], v[96:99], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[144:147], v[100:103], a[220:223]
	ds_read_b128 v[60:63], v209 offset:9024                    
	v_mfma_f32_16x16x32_bf16 a[224:227], v[148:151], v[72:75], a[224:227]
	v_mfma_f32_16x16x32_bf16 a[228:231], v[148:151], v[76:79], a[228:231]
	ds_read_b128 v[32:35], v209 offset:12672                   
	v_mfma_f32_16x16x32_bf16 a[232:235], v[148:151], v[80:83], a[232:235]
	s_add_u32 m0, 0x3180, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[236:239], v[148:151], v[84:87], a[236:239]
	ds_read_b128 v[64:67], v209 offset:12736                   
	v_mfma_f32_16x16x32_bf16 a[240:243], v[148:151], v[88:91], a[240:243]
	v_mfma_f32_16x16x32_bf16 a[244:247], v[148:151], v[92:95], a[244:247]
	ds_read_b128 v[36:39], v209 offset:13184                   
	v_mfma_f32_16x16x32_bf16 a[248:251], v[148:151], v[96:99], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[148:151], v[100:103], a[252:255]
	ds_read_b128 v[68:71], v209 offset:13248                   
	v_mfma_f32_16x16x32_bf16 a[128:131], v[152:155], v[104:107], a[128:131]
	v_mfma_f32_16x16x32_bf16 a[132:135], v[152:155], v[108:111], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[136:139], v[152:155], v[112:115], a[136:139]
	s_add_u32 m0, 0x4200, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[140:143], v[152:155], v[116:119], a[140:143]
	v_mfma_f32_16x16x32_bf16 a[144:147], v[152:155], v[120:123], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[152:155], v[124:127], a[148:151]
	v_mfma_f32_16x16x32_bf16 a[152:155], v[152:155], v[128:131], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[152:155], v[132:135], a[156:159]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[156:159], v[104:107], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[156:159], v[108:111], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[168:171], v[156:159], v[112:115], a[168:171]
	s_add_u32 m0, 0x5280, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[172:175], v[156:159], v[116:119], a[172:175]
	v_mfma_f32_16x16x32_bf16 a[176:179], v[156:159], v[120:123], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[156:159], v[124:127], a[180:183]
	v_mfma_f32_16x16x32_bf16 a[184:187], v[156:159], v[128:131], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[156:159], v[132:135], a[188:191]
	v_mfma_f32_16x16x32_bf16 a[192:195], v[160:163], v[104:107], a[192:195]
	v_mfma_f32_16x16x32_bf16 a[196:199], v[160:163], v[108:111], a[196:199]
	v_mfma_f32_16x16x32_bf16 a[200:203], v[160:163], v[112:115], a[200:203]
	s_add_u32 m0, 0x6300, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[204:207], v[160:163], v[116:119], a[204:207]
	v_mfma_f32_16x16x32_bf16 a[208:211], v[160:163], v[120:123], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[160:163], v[124:127], a[212:215]
	v_mfma_f32_16x16x32_bf16 a[216:219], v[160:163], v[128:131], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[160:163], v[132:135], a[220:223]
	v_mfma_f32_16x16x32_bf16 a[224:227], v[164:167], v[104:107], a[224:227]
	v_mfma_f32_16x16x32_bf16 a[228:231], v[164:167], v[108:111], a[228:231]
	v_mfma_f32_16x16x32_bf16 a[232:235], v[164:167], v[112:115], a[232:235]
	s_add_u32 m0, 0x7380, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[236:239], v[164:167], v[116:119], a[236:239]
	s_add_u32 s51, 0xc0, s49                                   
	s_cmp_lt_u32 s51, s50                                      
	s_cselect_b32 s62, s62, 0                                  
	v_mfma_f32_16x16x32_bf16 a[240:243], v[164:167], v[120:123], a[240:243]
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	v_mfma_f32_16x16x32_bf16 a[244:247], v[164:167], v[124:127], a[244:247]
	s_addk_i32 s49, 0x40                                       
	s_cmp_lt_i32 s49, s50                                      
	v_mfma_f32_16x16x32_bf16 a[248:251], v[164:167], v[128:131], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[164:167], v[132:135], a[252:255]
	s_cbranch_scc0 label_0AA6                                  
	s_waitcnt vmcnt(8) lgkmcnt(0)                              
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[0:3], v[168:171], v[8:11], a[0:3]
	v_mfma_f32_16x16x32_bf16 a[4:7], v[168:171], v[12:15], a[4:7]
	ds_read_b128 v[72:75], v209 offset:16896                   
	v_mfma_f32_16x16x32_bf16 a[8:11], v[168:171], v[16:19], a[8:11]
	s_add_u32 m0, 0x8400, s61                                  
	buffer_load_dwordx4 v210, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[12:15], v[168:171], v[20:23], a[12:15]
	ds_read_b128 v[104:107], v209 offset:16960                 
	v_mfma_f32_16x16x32_bf16 a[16:19], v[168:171], v[24:27], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[168:171], v[28:31], a[20:23]
	ds_read_b128 v[76:79], v209 offset:17408                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[168:171], v[32:35], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[168:171], v[36:39], a[28:31]
	ds_read_b128 v[108:111], v209 offset:17472                 
	v_mfma_f32_16x16x32_bf16 a[32:35], v[172:175], v[8:11], a[32:35]
	v_mfma_f32_16x16x32_bf16 a[36:39], v[172:175], v[12:15], a[36:39]
	ds_read_b128 v[80:83], v209 offset:21120                   
	v_mfma_f32_16x16x32_bf16 a[40:43], v[172:175], v[16:19], a[40:43]
	s_add_u32 m0, 0x9480, s61                                  
	buffer_load_dwordx4 v211, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[44:47], v[172:175], v[20:23], a[44:47]
	ds_read_b128 v[112:115], v209 offset:21184                 
	v_mfma_f32_16x16x32_bf16 a[48:51], v[172:175], v[24:27], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[172:175], v[28:31], a[52:55]
	ds_read_b128 v[84:87], v209 offset:21632                   
	v_mfma_f32_16x16x32_bf16 a[56:59], v[172:175], v[32:35], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[172:175], v[36:39], a[60:63]
	ds_read_b128 v[116:119], v209 offset:21696                 
	v_mfma_f32_16x16x32_bf16 a[64:67], v[176:179], v[8:11], a[64:67]
	v_mfma_f32_16x16x32_bf16 a[68:71], v[176:179], v[12:15], a[68:71]
	ds_read_b128 v[88:91], v209 offset:25344                   
	v_mfma_f32_16x16x32_bf16 a[72:75], v[176:179], v[16:19], a[72:75]
	s_add_u32 m0, 0xa500, s61                                  
	buffer_load_dwordx4 v212, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[76:79], v[176:179], v[20:23], a[76:79]
	ds_read_b128 v[120:123], v209 offset:25408                 
	v_mfma_f32_16x16x32_bf16 a[80:83], v[176:179], v[24:27], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[176:179], v[28:31], a[84:87]
	ds_read_b128 v[92:95], v209 offset:25856                   
	v_mfma_f32_16x16x32_bf16 a[88:91], v[176:179], v[32:35], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[176:179], v[36:39], a[92:95]
	ds_read_b128 v[124:127], v209 offset:25920                 
	v_mfma_f32_16x16x32_bf16 a[96:99], v[180:183], v[8:11], a[96:99]
	v_mfma_f32_16x16x32_bf16 a[100:103], v[180:183], v[12:15], a[100:103]
	ds_read_b128 v[96:99], v209 offset:29568                   
	v_mfma_f32_16x16x32_bf16 a[104:107], v[180:183], v[16:19], a[104:107]
	s_add_u32 m0, 0xb580, s61                                  
	buffer_load_dwordx4 v213, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[108:111], v[180:183], v[20:23], a[108:111]
	ds_read_b128 v[128:131], v209 offset:29632                 
	v_mfma_f32_16x16x32_bf16 a[112:115], v[180:183], v[24:27], a[112:115]
	v_mfma_f32_16x16x32_bf16 a[116:119], v[180:183], v[28:31], a[116:119]
	ds_read_b128 v[100:103], v209 offset:30080                 
	v_mfma_f32_16x16x32_bf16 a[120:123], v[180:183], v[32:35], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[180:183], v[36:39], a[124:127]
	ds_read_b128 v[132:135], v209 offset:30144                 
	v_mfma_f32_16x16x32_bf16 a[0:3], v[184:187], v[40:43], a[0:3]
	v_mfma_f32_16x16x32_bf16 a[4:7], v[184:187], v[44:47], a[4:7]
	ds_read_b128 v[136:139], v218                              
	v_mfma_f32_16x16x32_bf16 a[8:11], v[184:187], v[48:51], a[8:11]
	s_add_u32 m0, 0xc600, s61                                  
	buffer_load_dwordx4 v214, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[12:15], v[184:187], v[52:55], a[12:15]
	ds_read_b128 v[152:155], v218 offset:64                    
	v_mfma_f32_16x16x32_bf16 a[16:19], v[184:187], v[56:59], a[16:19]
	v_mfma_f32_16x16x32_bf16 a[20:23], v[184:187], v[60:63], a[20:23]
	ds_read_b128 v[140:143], v218 offset:512                   
	v_mfma_f32_16x16x32_bf16 a[24:27], v[184:187], v[64:67], a[24:27]
	v_mfma_f32_16x16x32_bf16 a[28:31], v[184:187], v[68:71], a[28:31]
	ds_read_b128 v[156:159], v218 offset:576                   
	v_mfma_f32_16x16x32_bf16 a[32:35], v[188:191], v[40:43], a[32:35]
	v_mfma_f32_16x16x32_bf16 a[36:39], v[188:191], v[44:47], a[36:39]
	ds_read_b128 v[144:147], v218 offset:4224                  
	v_mfma_f32_16x16x32_bf16 a[40:43], v[188:191], v[48:51], a[40:43]
	s_add_u32 m0, 0xd680, s61                                  
	buffer_load_dwordx4 v215, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[44:47], v[188:191], v[52:55], a[44:47]
	ds_read_b128 v[160:163], v218 offset:4288                  
	v_mfma_f32_16x16x32_bf16 a[48:51], v[188:191], v[56:59], a[48:51]
	v_mfma_f32_16x16x32_bf16 a[52:55], v[188:191], v[60:63], a[52:55]
	ds_read_b128 v[148:151], v218 offset:4736                  
	v_mfma_f32_16x16x32_bf16 a[56:59], v[188:191], v[64:67], a[56:59]
	v_mfma_f32_16x16x32_bf16 a[60:63], v[188:191], v[68:71], a[60:63]
	ds_read_b128 v[164:167], v218 offset:4800                  
	v_mfma_f32_16x16x32_bf16 a[64:67], v[192:195], v[40:43], a[64:67]
	v_mfma_f32_16x16x32_bf16 a[68:71], v[192:195], v[44:47], a[68:71]
	v_mfma_f32_16x16x32_bf16 a[72:75], v[192:195], v[48:51], a[72:75]
	s_add_u32 m0, 0xe700, s61                                  
	buffer_load_dwordx4 v216, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[76:79], v[192:195], v[52:55], a[76:79]
	v_mfma_f32_16x16x32_bf16 a[80:83], v[192:195], v[56:59], a[80:83]
	v_mfma_f32_16x16x32_bf16 a[84:87], v[192:195], v[60:63], a[84:87]
	v_mfma_f32_16x16x32_bf16 a[88:91], v[192:195], v[64:67], a[88:91]
	v_mfma_f32_16x16x32_bf16 a[92:95], v[192:195], v[68:71], a[92:95]
	v_mfma_f32_16x16x32_bf16 a[96:99], v[196:199], v[40:43], a[96:99]
	v_mfma_f32_16x16x32_bf16 a[100:103], v[196:199], v[44:47], a[100:103]
	v_mfma_f32_16x16x32_bf16 a[104:107], v[196:199], v[48:51], a[104:107]
	s_add_u32 m0, 0xf780, s61                                  
	buffer_load_dwordx4 v217, s[16:19], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[108:111], v[196:199], v[52:55], a[108:111]
	s_add_u32 s52, 0x80, s49                                   
	s_cmp_lt_u32 s52, s50                                      
	s_cselect_b32 s63, s63, 0                                  
	v_mfma_f32_16x16x32_bf16 a[112:115], v[196:199], v[56:59], a[112:115]
	s_add_u32 s16, s63, s16                                    
	s_addc_u32 s17, 0, s17                                     
	s_sub_u32 s18, s18, s63                                    
	v_mfma_f32_16x16x32_bf16 a[116:119], v[196:199], v[60:63], a[116:119]
	v_mfma_f32_16x16x32_bf16 a[120:123], v[196:199], v[64:67], a[120:123]
	v_mfma_f32_16x16x32_bf16 a[124:127], v[196:199], v[68:71], a[124:127]
	s_waitcnt vmcnt(12) lgkmcnt(0)                             
	s_barrier                                                  
	v_mfma_f32_16x16x32_bf16 a[128:131], v[168:171], v[72:75], a[128:131]
	v_mfma_f32_16x16x32_bf16 a[132:135], v[168:171], v[76:79], a[132:135]
	ds_read_b128 v[8:11], v208                                 
	v_mfma_f32_16x16x32_bf16 a[136:139], v[168:171], v[80:83], a[136:139]
	s_add_u32 m0, 0x8400, s60                                  
	buffer_load_dwordx4 v200, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[140:143], v[168:171], v[84:87], a[140:143]
	ds_read_b128 v[40:43], v208 offset:64                      
	v_mfma_f32_16x16x32_bf16 a[144:147], v[168:171], v[88:91], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[168:171], v[92:95], a[148:151]
	ds_read_b128 v[12:15], v208 offset:512                     
	v_mfma_f32_16x16x32_bf16 a[152:155], v[168:171], v[96:99], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[168:171], v[100:103], a[156:159]
	ds_read_b128 v[44:47], v208 offset:576                     
	v_mfma_f32_16x16x32_bf16 a[160:163], v[172:175], v[72:75], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[172:175], v[76:79], a[164:167]
	ds_read_b128 v[16:19], v208 offset:4224                    
	v_mfma_f32_16x16x32_bf16 a[168:171], v[172:175], v[80:83], a[168:171]
	s_add_u32 m0, 0x9480, s60                                  
	buffer_load_dwordx4 v201, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[172:175], v[172:175], v[84:87], a[172:175]
	ds_read_b128 v[48:51], v208 offset:4288                    
	v_mfma_f32_16x16x32_bf16 a[176:179], v[172:175], v[88:91], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[172:175], v[92:95], a[180:183]
	ds_read_b128 v[20:23], v208 offset:4736                    
	v_mfma_f32_16x16x32_bf16 a[184:187], v[172:175], v[96:99], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[172:175], v[100:103], a[188:191]
	ds_read_b128 v[52:55], v208 offset:4800                    
	v_mfma_f32_16x16x32_bf16 a[192:195], v[176:179], v[72:75], a[192:195]
	v_mfma_f32_16x16x32_bf16 a[196:199], v[176:179], v[76:79], a[196:199]
	ds_read_b128 v[24:27], v208 offset:8448                    
	v_mfma_f32_16x16x32_bf16 a[200:203], v[176:179], v[80:83], a[200:203]
	s_add_u32 m0, 0xa500, s60                                  
	buffer_load_dwordx4 v202, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[204:207], v[176:179], v[84:87], a[204:207]
	ds_read_b128 v[56:59], v208 offset:8512                    
	v_mfma_f32_16x16x32_bf16 a[208:211], v[176:179], v[88:91], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[176:179], v[92:95], a[212:215]
	ds_read_b128 v[28:31], v208 offset:8960                    
	v_mfma_f32_16x16x32_bf16 a[216:219], v[176:179], v[96:99], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[176:179], v[100:103], a[220:223]
	ds_read_b128 v[60:63], v208 offset:9024                    
	v_mfma_f32_16x16x32_bf16 a[224:227], v[180:183], v[72:75], a[224:227]
	v_mfma_f32_16x16x32_bf16 a[228:231], v[180:183], v[76:79], a[228:231]
	ds_read_b128 v[32:35], v208 offset:12672                   
	v_mfma_f32_16x16x32_bf16 a[232:235], v[180:183], v[80:83], a[232:235]
	s_add_u32 m0, 0xb580, s60                                  
	buffer_load_dwordx4 v203, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[236:239], v[180:183], v[84:87], a[236:239]
	ds_read_b128 v[64:67], v208 offset:12736                   
	v_mfma_f32_16x16x32_bf16 a[240:243], v[180:183], v[88:91], a[240:243]
	v_mfma_f32_16x16x32_bf16 a[244:247], v[180:183], v[92:95], a[244:247]
	ds_read_b128 v[36:39], v208 offset:13184                   
	v_mfma_f32_16x16x32_bf16 a[248:251], v[180:183], v[96:99], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[180:183], v[100:103], a[252:255]
	ds_read_b128 v[68:71], v208 offset:13248                   
	v_mfma_f32_16x16x32_bf16 a[128:131], v[184:187], v[104:107], a[128:131]
	v_mfma_f32_16x16x32_bf16 a[132:135], v[184:187], v[108:111], a[132:135]
	v_mfma_f32_16x16x32_bf16 a[136:139], v[184:187], v[112:115], a[136:139]
	s_add_u32 m0, 0xc600, s60                                  
	buffer_load_dwordx4 v204, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[140:143], v[184:187], v[116:119], a[140:143]
	v_mfma_f32_16x16x32_bf16 a[144:147], v[184:187], v[120:123], a[144:147]
	v_mfma_f32_16x16x32_bf16 a[148:151], v[184:187], v[124:127], a[148:151]
	v_mfma_f32_16x16x32_bf16 a[152:155], v[184:187], v[128:131], a[152:155]
	v_mfma_f32_16x16x32_bf16 a[156:159], v[184:187], v[132:135], a[156:159]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[188:191], v[104:107], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[188:191], v[108:111], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[168:171], v[188:191], v[112:115], a[168:171]
	s_add_u32 m0, 0xd680, s60                                  
	buffer_load_dwordx4 v205, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[172:175], v[188:191], v[116:119], a[172:175]
	v_mfma_f32_16x16x32_bf16 a[176:179], v[188:191], v[120:123], a[176:179]
	v_mfma_f32_16x16x32_bf16 a[180:183], v[188:191], v[124:127], a[180:183]
	v_mfma_f32_16x16x32_bf16 a[184:187], v[188:191], v[128:131], a[184:187]
	v_mfma_f32_16x16x32_bf16 a[188:191], v[188:191], v[132:135], a[188:191]
	v_mfma_f32_16x16x32_bf16 a[192:195], v[192:195], v[104:107], a[192:195]
	v_mfma_f32_16x16x32_bf16 a[196:199], v[192:195], v[108:111], a[196:199]
	v_mfma_f32_16x16x32_bf16 a[200:203], v[192:195], v[112:115], a[200:203]
	s_add_u32 m0, 0xe700, s60                                  
	buffer_load_dwordx4 v206, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[204:207], v[192:195], v[116:119], a[204:207]
	v_mfma_f32_16x16x32_bf16 a[208:211], v[192:195], v[120:123], a[208:211]
	v_mfma_f32_16x16x32_bf16 a[212:215], v[192:195], v[124:127], a[212:215]
	v_mfma_f32_16x16x32_bf16 a[216:219], v[192:195], v[128:131], a[216:219]
	v_mfma_f32_16x16x32_bf16 a[220:223], v[192:195], v[132:135], a[220:223]
	v_mfma_f32_16x16x32_bf16 a[224:227], v[196:199], v[104:107], a[224:227]
	v_mfma_f32_16x16x32_bf16 a[228:231], v[196:199], v[108:111], a[228:231]
	v_mfma_f32_16x16x32_bf16 a[232:235], v[196:199], v[112:115], a[232:235]
	s_add_u32 m0, 0xf780, s60                                  
	buffer_load_dwordx4 v207, s[12:15], 0 offen lds            
	v_mfma_f32_16x16x32_bf16 a[236:239], v[196:199], v[116:119], a[236:239]
	s_add_u32 s51, 0xc0, s49                                   
	s_cmp_lt_u32 s51, s50                                      
	s_cselect_b32 s62, s62, 0                                  
	v_mfma_f32_16x16x32_bf16 a[240:243], v[196:199], v[120:123], a[240:243]
	s_add_u32 s12, s62, s12                                    
	s_addc_u32 s13, 0, s13                                     
	s_sub_u32 s14, s14, s62                                    
	v_mfma_f32_16x16x32_bf16 a[244:247], v[196:199], v[124:127], a[244:247]
	s_addk_i32 s49, 0x40                                       
	s_cmp_lt_i32 s49, s50                                      
	v_mfma_f32_16x16x32_bf16 a[248:251], v[196:199], v[128:131], a[248:251]
	v_mfma_f32_16x16x32_bf16 a[252:255], v[196:199], v[132:135], a[252:255]
	s_cbranch_scc0 label_0AA6                                  
	s_branch label_075D                                        
	
label_0AA6:
	s_mov_b32 s49, 0                                           
	s_mul_i32 s50, 16, s47                                     
	s_lshr_b32 s52, s43, 4                                     
	s_sub_u32 s50, s52, s50                                    
	s_waitcnt vmcnt(0) lgkmcnt(0)                              
	s_barrier                                                  
	v_accvgpr_read_b32 v8, a0                                  
	v_accvgpr_read_b32 v9, a1                                  
	v_accvgpr_read_b32 v10, a2                                 
	v_accvgpr_read_b32 v11, a3                                 
	v_accvgpr_read_b32 v12, a32                                
	v_accvgpr_read_b32 v13, a33                                
	v_accvgpr_read_b32 v14, a34                                
	v_accvgpr_read_b32 v15, a35                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v220, s[4:7], 0 offen       
	v_add_i32 v220, v220, 64                                   
	v_accvgpr_read_b32 v8, a64                                 
	v_accvgpr_read_b32 v9, a65                                 
	v_accvgpr_read_b32 v10, a66                                
	v_accvgpr_read_b32 v11, a67                                
	v_accvgpr_read_b32 v12, a96                                
	v_accvgpr_read_b32 v13, a97                                
	v_accvgpr_read_b32 v14, a98                                
	v_accvgpr_read_b32 v15, a99                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v220, s[4:7], 0 offen       
	v_add_i32 v220, v220, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a4                                  
	v_accvgpr_read_b32 v9, a5                                  
	v_accvgpr_read_b32 v10, a6                                 
	v_accvgpr_read_b32 v11, a7                                 
	v_accvgpr_read_b32 v12, a36                                
	v_accvgpr_read_b32 v13, a37                                
	v_accvgpr_read_b32 v14, a38                                
	v_accvgpr_read_b32 v15, a39                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v221, s[4:7], 0 offen       
	v_add_i32 v221, v221, 64                                   
	v_accvgpr_read_b32 v8, a68                                 
	v_accvgpr_read_b32 v9, a69                                 
	v_accvgpr_read_b32 v10, a70                                
	v_accvgpr_read_b32 v11, a71                                
	v_accvgpr_read_b32 v12, a100                               
	v_accvgpr_read_b32 v13, a101                               
	v_accvgpr_read_b32 v14, a102                               
	v_accvgpr_read_b32 v15, a103                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v221, s[4:7], 0 offen       
	v_add_i32 v221, v221, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a8                                  
	v_accvgpr_read_b32 v9, a9                                  
	v_accvgpr_read_b32 v10, a10                                
	v_accvgpr_read_b32 v11, a11                                
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
	buffer_store_dwordx4 v[16:19], v222, s[4:7], 0 offen       
	v_add_i32 v222, v222, 64                                   
	v_accvgpr_read_b32 v8, a72                                 
	v_accvgpr_read_b32 v9, a73                                 
	v_accvgpr_read_b32 v10, a74                                
	v_accvgpr_read_b32 v11, a75                                
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
	buffer_store_dwordx4 v[16:19], v222, s[4:7], 0 offen       
	v_add_i32 v222, v222, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a12                                 
	v_accvgpr_read_b32 v9, a13                                 
	v_accvgpr_read_b32 v10, a14                                
	v_accvgpr_read_b32 v11, a15                                
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
	buffer_store_dwordx4 v[16:19], v223, s[4:7], 0 offen       
	v_add_i32 v223, v223, 64                                   
	v_accvgpr_read_b32 v8, a76                                 
	v_accvgpr_read_b32 v9, a77                                 
	v_accvgpr_read_b32 v10, a78                                
	v_accvgpr_read_b32 v11, a79                                
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
	buffer_store_dwordx4 v[16:19], v223, s[4:7], 0 offen       
	v_add_i32 v223, v223, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a16                                 
	v_accvgpr_read_b32 v9, a17                                 
	v_accvgpr_read_b32 v10, a18                                
	v_accvgpr_read_b32 v11, a19                                
	v_accvgpr_read_b32 v12, a48                                
	v_accvgpr_read_b32 v13, a49                                
	v_accvgpr_read_b32 v14, a50                                
	v_accvgpr_read_b32 v15, a51                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v224, s[4:7], 0 offen       
	v_add_i32 v224, v224, 64                                   
	v_accvgpr_read_b32 v8, a80                                 
	v_accvgpr_read_b32 v9, a81                                 
	v_accvgpr_read_b32 v10, a82                                
	v_accvgpr_read_b32 v11, a83                                
	v_accvgpr_read_b32 v12, a112                               
	v_accvgpr_read_b32 v13, a113                               
	v_accvgpr_read_b32 v14, a114                               
	v_accvgpr_read_b32 v15, a115                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v224, s[4:7], 0 offen       
	v_add_i32 v224, v224, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a20                                 
	v_accvgpr_read_b32 v9, a21                                 
	v_accvgpr_read_b32 v10, a22                                
	v_accvgpr_read_b32 v11, a23                                
	v_accvgpr_read_b32 v12, a52                                
	v_accvgpr_read_b32 v13, a53                                
	v_accvgpr_read_b32 v14, a54                                
	v_accvgpr_read_b32 v15, a55                                
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v225, s[4:7], 0 offen       
	v_add_i32 v225, v225, 64                                   
	v_accvgpr_read_b32 v8, a84                                 
	v_accvgpr_read_b32 v9, a85                                 
	v_accvgpr_read_b32 v10, a86                                
	v_accvgpr_read_b32 v11, a87                                
	v_accvgpr_read_b32 v12, a116                               
	v_accvgpr_read_b32 v13, a117                               
	v_accvgpr_read_b32 v14, a118                               
	v_accvgpr_read_b32 v15, a119                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v225, s[4:7], 0 offen       
	v_add_i32 v225, v225, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a24                                 
	v_accvgpr_read_b32 v9, a25                                 
	v_accvgpr_read_b32 v10, a26                                
	v_accvgpr_read_b32 v11, a27                                
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
	buffer_store_dwordx4 v[16:19], v226, s[4:7], 0 offen       
	v_add_i32 v226, v226, 64                                   
	v_accvgpr_read_b32 v8, a88                                 
	v_accvgpr_read_b32 v9, a89                                 
	v_accvgpr_read_b32 v10, a90                                
	v_accvgpr_read_b32 v11, a91                                
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
	buffer_store_dwordx4 v[16:19], v226, s[4:7], 0 offen       
	v_add_i32 v226, v226, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a28                                 
	v_accvgpr_read_b32 v9, a29                                 
	v_accvgpr_read_b32 v10, a30                                
	v_accvgpr_read_b32 v11, a31                                
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
	buffer_store_dwordx4 v[16:19], v227, s[4:7], 0 offen       
	v_add_i32 v227, v227, 64                                   
	v_accvgpr_read_b32 v8, a92                                 
	v_accvgpr_read_b32 v9, a93                                 
	v_accvgpr_read_b32 v10, a94                                
	v_accvgpr_read_b32 v11, a95                                
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
	buffer_store_dwordx4 v[16:19], v227, s[4:7], 0 offen       
	v_add_i32 v227, v227, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a128                                
	v_accvgpr_read_b32 v9, a129                                
	v_accvgpr_read_b32 v10, a130                               
	v_accvgpr_read_b32 v11, a131                               
	v_accvgpr_read_b32 v12, a160                               
	v_accvgpr_read_b32 v13, a161                               
	v_accvgpr_read_b32 v14, a162                               
	v_accvgpr_read_b32 v15, a163                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v228, s[4:7], 0 offen       
	v_add_i32 v228, v228, 64                                   
	v_accvgpr_read_b32 v8, a192                                
	v_accvgpr_read_b32 v9, a193                                
	v_accvgpr_read_b32 v10, a194                               
	v_accvgpr_read_b32 v11, a195                               
	v_accvgpr_read_b32 v12, a224                               
	v_accvgpr_read_b32 v13, a225                               
	v_accvgpr_read_b32 v14, a226                               
	v_accvgpr_read_b32 v15, a227                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v228, s[4:7], 0 offen       
	v_add_i32 v228, v228, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a132                                
	v_accvgpr_read_b32 v9, a133                                
	v_accvgpr_read_b32 v10, a134                               
	v_accvgpr_read_b32 v11, a135                               
	v_accvgpr_read_b32 v12, a164                               
	v_accvgpr_read_b32 v13, a165                               
	v_accvgpr_read_b32 v14, a166                               
	v_accvgpr_read_b32 v15, a167                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v229, s[4:7], 0 offen       
	v_add_i32 v229, v229, 64                                   
	v_accvgpr_read_b32 v8, a196                                
	v_accvgpr_read_b32 v9, a197                                
	v_accvgpr_read_b32 v10, a198                               
	v_accvgpr_read_b32 v11, a199                               
	v_accvgpr_read_b32 v12, a228                               
	v_accvgpr_read_b32 v13, a229                               
	v_accvgpr_read_b32 v14, a230                               
	v_accvgpr_read_b32 v15, a231                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v229, s[4:7], 0 offen       
	v_add_i32 v229, v229, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a136                                
	v_accvgpr_read_b32 v9, a137                                
	v_accvgpr_read_b32 v10, a138                               
	v_accvgpr_read_b32 v11, a139                               
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
	buffer_store_dwordx4 v[16:19], v230, s[4:7], 0 offen       
	v_add_i32 v230, v230, 64                                   
	v_accvgpr_read_b32 v8, a200                                
	v_accvgpr_read_b32 v9, a201                                
	v_accvgpr_read_b32 v10, a202                               
	v_accvgpr_read_b32 v11, a203                               
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
	buffer_store_dwordx4 v[16:19], v230, s[4:7], 0 offen       
	v_add_i32 v230, v230, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a140                                
	v_accvgpr_read_b32 v9, a141                                
	v_accvgpr_read_b32 v10, a142                               
	v_accvgpr_read_b32 v11, a143                               
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
	buffer_store_dwordx4 v[16:19], v231, s[4:7], 0 offen       
	v_add_i32 v231, v231, 64                                   
	v_accvgpr_read_b32 v8, a204                                
	v_accvgpr_read_b32 v9, a205                                
	v_accvgpr_read_b32 v10, a206                               
	v_accvgpr_read_b32 v11, a207                               
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
	buffer_store_dwordx4 v[16:19], v231, s[4:7], 0 offen       
	v_add_i32 v231, v231, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a144                                
	v_accvgpr_read_b32 v9, a145                                
	v_accvgpr_read_b32 v10, a146                               
	v_accvgpr_read_b32 v11, a147                               
	v_accvgpr_read_b32 v12, a176                               
	v_accvgpr_read_b32 v13, a177                               
	v_accvgpr_read_b32 v14, a178                               
	v_accvgpr_read_b32 v15, a179                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v232, s[4:7], 0 offen       
	v_add_i32 v232, v232, 64                                   
	v_accvgpr_read_b32 v8, a208                                
	v_accvgpr_read_b32 v9, a209                                
	v_accvgpr_read_b32 v10, a210                               
	v_accvgpr_read_b32 v11, a211                               
	v_accvgpr_read_b32 v12, a240                               
	v_accvgpr_read_b32 v13, a241                               
	v_accvgpr_read_b32 v14, a242                               
	v_accvgpr_read_b32 v15, a243                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v232, s[4:7], 0 offen       
	v_add_i32 v232, v232, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a148                                
	v_accvgpr_read_b32 v9, a149                                
	v_accvgpr_read_b32 v10, a150                               
	v_accvgpr_read_b32 v11, a151                               
	v_accvgpr_read_b32 v12, a180                               
	v_accvgpr_read_b32 v13, a181                               
	v_accvgpr_read_b32 v14, a182                               
	v_accvgpr_read_b32 v15, a183                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v233, s[4:7], 0 offen       
	v_add_i32 v233, v233, 64                                   
	v_accvgpr_read_b32 v8, a212                                
	v_accvgpr_read_b32 v9, a213                                
	v_accvgpr_read_b32 v10, a214                               
	v_accvgpr_read_b32 v11, a215                               
	v_accvgpr_read_b32 v12, a244                               
	v_accvgpr_read_b32 v13, a245                               
	v_accvgpr_read_b32 v14, a246                               
	v_accvgpr_read_b32 v15, a247                               
	v_cvt_pk_bf16_f32 v16, v8, v9                              
	v_cvt_pk_bf16_f32 v17, v10, v11                            
	v_cvt_pk_bf16_f32 v18, v12, v13                            
	v_cvt_pk_bf16_f32 v19, v14, v15                            
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v16, v18                         
	s_nop 1                                                    
	v_permlane16_swap_b32_e32 v17, v19                         
	s_nop 1                                                    
	buffer_store_dwordx4 v[16:19], v233, s[4:7], 0 offen       
	v_add_i32 v233, v233, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a152                                
	v_accvgpr_read_b32 v9, a153                                
	v_accvgpr_read_b32 v10, a154                               
	v_accvgpr_read_b32 v11, a155                               
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_i32 v234, v234, 64                                   
	v_accvgpr_read_b32 v8, a216                                
	v_accvgpr_read_b32 v9, a217                                
	v_accvgpr_read_b32 v10, a218                               
	v_accvgpr_read_b32 v11, a219                               
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
	buffer_store_dwordx4 v[16:19], v234, s[4:7], 0 offen       
	v_add_i32 v234, v234, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	v_accvgpr_read_b32 v8, a156                                
	v_accvgpr_read_b32 v9, a157                                
	v_accvgpr_read_b32 v10, a158                               
	v_accvgpr_read_b32 v11, a159                               
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
	buffer_store_dwordx4 v[16:19], v235, s[4:7], 0 offen       
	v_add_i32 v235, v235, 64                                   
	v_accvgpr_read_b32 v8, a220                                
	v_accvgpr_read_b32 v9, a221                                
	v_accvgpr_read_b32 v10, a222                               
	v_accvgpr_read_b32 v11, a223                               
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
	buffer_store_dwordx4 v[16:19], v235, s[4:7], 0 offen       
	v_add_i32 v235, v235, 64                                   
	s_addk_i32 s49, 0x1                                        
	s_cmp_lt_i32 s49, s50                                      
	s_cbranch_scc0 label_0EFC                                  
	
label_0EFC:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    
	s_endpgm                                                   

// ===== Kernel Descriptor (generates .rodata) =====
.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter24bf16gemm_bf16_tn_256x256E
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
custom.config:
  Source:
    Origin: aiter
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: b
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: 1
    TransposeB: 0
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: address, semantic: AddressD, padding: 8 },
            { type: address, semantic: AddressC, padding: 8 },
            { type: address, semantic: AddressB, padding: 8 },
            { type: address, semantic: AddressA, padding: 8 },
            { type: float32, semantic: Alpha, padding: 12 },
            { type: float32, semantic: Beta, padding: 12 },
            { type: uint32, semantic: StrideD0Bytes, padding: 12 },
            { type: uint32, semantic: StrideD1, padding: 12 },
            { type: uint32, semantic: StrideC0Bytes, padding: 12 },
            { type: uint32, semantic: StrideC1, padding: 12 },
            { type: uint32, semantic: StrideB0Bytes, padding: 12 },
            { type: uint32, semantic: StrideB1, padding: 12 },
            { type: uint32, semantic: StrideA0Bytes, padding: 12 },
            { type: uint32, semantic: StrideA1, padding: 12 },
            { type: uint32, semantic: SizeFree1, padding: 12 },
            { type: uint32, semantic: SizeFree0, padding: 12 },
            { type: uint32, semantic: SizeSum, padding: 12 },
            { type: uint32, semantic: SplitK, padding: 12 },
            { type: uint32, semantic: OutputBF16, padding: 12 } ]
    macrotile: [256, 256, 64]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, One]
  MatrixInstruction: [16, 16, 32, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  AssertFree0ElementMultiple: 256
  AssertFree1ElementMultiple: 256
  WavefrontSize: 64
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
      - .name:           splitk
        .offset:         272
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         276
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         280
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         284
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           is_out_b16
        .offset:         288
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         292
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         296
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         300
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 4
    .kernarg_segment_size: 304
    .max_flat_workgroup_size: 256
    .name:           _ZN5aiter24bf16gemm_bf16_tn_256x256E
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     112
    .symbol:         _ZN5aiter24bf16gemm_bf16_tn_256x256E.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata

