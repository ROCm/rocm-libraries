.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+"
.amdhsa_kernel load_lds_kernel
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_next_free_vgpr 2
    .amdhsa_accum_offset 4
.end_amdhsa_kernel

.p2align 8
.global load_lds_kernel
.type load_lds_kernel,@function
load_lds_kernel:
        s_load_dword s3, s[0:1], 0x1c
        v_cmp_gt_u32_e32 vcc, 64, v0
        s_waitcnt lgkmcnt(0)
        s_and_b32 s3, s3, 0xffff
        s_mul_i32 s2, s2, s3
        v_add_u32_e32 v2, s2, v0
        v_ashrrev_i32_e32 v3, 31, v2
        v_lshlrev_b32_e32 v0, 2, v0
        s_and_saveexec_b64 s[2:3], vcc
        s_cbranch_execz .LBB0_2
        s_load_dwordx2 s[4:5], s[0:1], 0x0
        s_waitcnt lgkmcnt(0)
        v_lshl_add_u64 v[4:5], v[2:3], 2, s[4:5]
        global_load_dword v1, v[4:5], off
        s_waitcnt vmcnt(0)
        ds_write_b32 v0, v1
.LBB0_2:                                ; %if.end
        s_or_b64 exec, exec, s[2:3]
        s_waitcnt lgkmcnt(0)
        s_barrier
        s_and_saveexec_b64 s[2:3], vcc
        s_cbranch_execz .LBB0_4
        s_load_dwordx2 s[0:1], s[0:1], 0x8
        ds_read_b32 v4, v0
        s_waitcnt lgkmcnt(0)
        v_lshl_add_u64 v[0:1], v[2:3], 2, s[0:1]
        global_store_dword v[0:1], v4, off
.LBB0_4:                                ; %if.end13
        s_endpgm
