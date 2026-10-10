# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>/dev/null
#
# List modifiers are structured operands; unsupported flags still keep the
# whole line verbatim rather than vanish on emission (idxen).
# FLAT and GLOBAL cache controls are carried, with the TH_STORE_* spelling on
# stores (the assembler rejects TH_LOAD_* there).

# CHECK-LABEL: region_begin:
# CHECK-NEXT:  v_pk_add_f32 v[0:1], v[2:3], v[4:5] op_sel:[1,0] op_sel_hi:[0,1]
# CHECK-NEXT:  v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7] neg_lo:[1,0,0] neg_hi:[0,1,0]
# CHECK-NEXT:  v_wmma_f32_16x16x32_f16 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,0,0]
# CHECK-NEXT:  v_wmma_i32_16x16x64_iu8 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,1,0]
# CHECK-NEXT:  buffer_load_b32 v1, v2, s[4:7], s3 idxen
# CHECK-NEXT:  buffer_load_b32 v1, v2, s[4:7], s3 offen offset:16 nv
# CHECK-NEXT:  flat_load_b32 v1, v[2:3] offset:8 scope:SCOPE_SYS th:TH_LOAD_NT
# CHECK-NEXT:  flat_store_b32 v[2:3], v1 scope:SCOPE_SYS th:TH_STORE_NT
# CHECK-NEXT:  flat_load_b32 v1, v[2:3] scope:SCOPE_DEV th:TH_LOAD_RT nv
# CHECK-NEXT:  scratch_load_b32 v8, off, s4 offset:16 scope:SCOPE_SYS th:TH_LOAD_NT nv
# CHECK-NEXT:  global_store_b64 v4, v[0:1], s[0:1] offset:8 th:TH_STORE_NT
# CHECK-NEXT:  global_load_b32 v1, v2, s[0:1] th:TH_LOAD_NT scope:SCOPE_DEV nv
# CHECK-NEXT:  global_load_async_to_lds_b128 v4, v5, s[2:3] offset:16 scale_offset
# CHECK-NEXT:  cluster_load_async_to_lds_b128 v6, v7, s[4:5] scale_offset
# CHECK-NEXT:  global_store_async_from_lds_b128 v8, v9, s[6:7] offset:32 scale_offset
# CHECK-NEXT:  s_load_b32 s0, s[2:3] scope:SCOPE_SYS nv
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_pk_add_f32 v[0:1], v[2:3], v[4:5] op_sel:[1,0] op_sel_hi:[0,1]
v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7] neg_lo:[1,0,0] neg_hi:[0,1,0]
v_wmma_f32_16x16x32_f16 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,0,0]
v_wmma_i32_16x16x64_iu8 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,1,0]
buffer_load_b32 v1, v2, s[4:7], s3 idxen
buffer_load_b32 v1, v2, s[4:7], s3 offen offset:16 nv
flat_load_b32 v1, v[2:3] offset:8 th:TH_LOAD_NT scope:SCOPE_SYS
flat_store_b32 v[2:3], v1 th:TH_STORE_NT scope:SCOPE_SYS
flat_load_b32 v1, v[2:3] th:TH_LOAD_RT scope:SCOPE_DEV nv
scratch_load_b32 v8, off, s4 offset:16 th:TH_LOAD_NT scope:SCOPE_SYS nv
global_store_b64 v4, v[0:1], s[0:1] offset:8 th:TH_STORE_NT
global_load_b32 v1, v2, s[0:1] th:TH_LOAD_NT scope:SCOPE_DEV nv
global_load_async_to_lds_b128 v4, v5, s[2:3] offset:16 scale_offset
cluster_load_async_to_lds_b128 v6, v7, s[4:5] scale_offset
global_store_async_from_lds_b128 v8, v9, s[6:7] offset:32 scale_offset
s_load_b32 s0, s[2:3] scope:SCOPE_SYS nv
region_end:
s_endpgm
