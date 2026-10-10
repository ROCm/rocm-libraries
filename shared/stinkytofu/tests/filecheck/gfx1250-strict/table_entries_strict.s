# RUN: %stinkytofu-opt --arch gfx1250-strict %s --from-label region_begin --to-label region_end 2>&1 >/dev/null | grep -iE "warning|error" | sed "s/^/diag: /"; echo END
#
# The gfx1250-strict (v0) table carries the same new entries as v1: no
# "Unknown mnemonic" / "storing as text" warnings.

# CHECK-NOT: diag:
# CHECK:     END

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[8:23], v[24:39], v[0:7]
global_load_b32 v0, v4, s[0:1]
global_store_b32 v4, v0, s[0:1]
tensor_store_from_lds s[0:3], s[4:11]
v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]
region_end:
s_endpgm
