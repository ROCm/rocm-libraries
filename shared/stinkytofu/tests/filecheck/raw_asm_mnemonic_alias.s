# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>&1
#
# Mnemonic lookup is case-insensitive and accepts the assembler's aliases
# (v_add_u32 = v_add_nc_u32, v_mad_u64_u32 = v_mad_co_u64_u32) and explicit
# `_e64`/`_e32` encoding suffixes, so these parse as real instructions (no
# "Unknown mnemonic ... storing as text" warning). The source spelling is kept:
# `_e64` selects the 8-byte encoding, so dropping it would change the code.

# CHECK-NOT:   Warning
# CHECK-NOT:   Error
# CHECK-LABEL: region_begin:
# CHECK-NEXT:  v_add_u32 v0, v1, v2
# CHECK-NEXT:  V_MOV_B32 v3, v4
# CHECK-NEXT:  v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]
# CHECK-NEXT:  v_mad_i64_i32 v[0:1], s0, v2, v3, v[4:5]
# CHECK-NEXT:  v_cmp_gt_u32_e64 vcc_lo, v0, v9
# CHECK-NEXT:  v_add_f32_e64 v1, v2, v3
# CHECK-NEXT:  v_mov_b32_e32 v1, v2
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_add_u32 v0, v1, v2
V_MOV_B32 v3, v4
v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]
v_mad_i64_i32 v[0:1], s0, v2, v3, v[4:5]
v_cmp_gt_u32_e64 vcc_lo, v0, v9
v_add_f32_e64 v1, v2, v3
v_mov_b32_e32 v1, v2
region_end:
s_endpgm
