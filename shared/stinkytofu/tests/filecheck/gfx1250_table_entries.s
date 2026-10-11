# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>&1 >/dev/null | grep -iE "warning|error" | sed "s/^/diag: /"; echo END
#
# Opcodes our gfx1250 kernels use that were missing from the table, so they
# became opaque text (with an "Unknown mnemonic" warning) that no pass could
# see: FP8/BF8 16x16x128 WMMA (f32 and f16 result), plain global loads/stores,
# tensor_store_from_lds, named barriers, permlane16/x16, the 64-bit SALU
# subtract. Every line must now parse as an instruction: no warnings.

# CHECK-NOT: diag:
# CHECK:     END

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[8:23], v[24:39], v[0:7]
v_wmma_f32_16x16x128_bf8_fp8 v[0:7], v[8:23], v[24:39], v[0:7]
v_wmma_f16_16x16x128_fp8_bf8 v[0:3], v[8:23], v[24:39], v[0:3]
global_load_u8 v0, v4, s[0:1]
global_load_b128 v[0:3], v4, s[0:1] offset:16
global_store_b64 v4, v[0:1], s[0:1] offset:8
global_store_b128 v4, v[0:3], s[0:1]
global_load_async_to_lds_b8 v4, v5, s[2:3]
global_load_async_to_lds_b32 v6, v[8:9], off
global_load_async_to_lds_b64 v10, v[12:13], off th:TH_LOAD_NT
global_load_async_to_lds_b128 v14, v15, s[4:5] offset:16 scale_offset
cluster_load_async_to_lds_b128 v16, v17, s[6:7] scale_offset
global_store_async_from_lds_b128 v18, v19, s[8:9] offset:32 scale_offset
global_load_monitor_b32 v20, v21, s[10:11] scope:SCOPE_DEV
global_atomic_add_u32 v64, v65, s[72:73] scope:SCOPE_DEV
s_monitor_sleep 127
s_trap 2
s_cmp_lt_u32 ttmp9, s87
scratch_load_b32 v22, off, s12 offset:16 th:TH_LOAD_NT scope:SCOPE_SYS nv
tensor_store_from_lds s[0:3], s[4:11]
s_barrier_init m0
s_barrier_join m0
v_permlane16_b32 v0, v1, s0, s1
v_permlanex16_b32 v0, v1, s0, s1
s_sub_nc_u64 s[0:1], s[2:3], s[4:5]
region_end:
s_endpgm
