# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>&1 | sed "s/^\(.*[Ww]arning.*\)$/diag: \1/"; %stinkytofu-opt --arch gfx1250-strict %s --from-label region_begin --to-label region_end 2>&1 | sed "s/^\(.*[Ww]arning.*\)$/diag: \1/"
#
# LDS and flat atomics were missing from both gfx1250 tables, so they were kept
# as text that the wait passes could not see. Each must parse (no warning) and
# print back with its operands, for v1 and for the strict (v0) table.

# CHECK-NOT: diag:
# CHECK:      region_begin:
# CHECK-NEXT: ds_add_rtn_u32 v0, v1, v2
# CHECK-NEXT: ds_add_u32 v1, v2
# CHECK-NEXT: flat_atomic_add_u32 v0, v[2:3], v4
# CHECK-NOT: diag:
# CHECK:      region_begin:
# CHECK-NEXT: ds_add_rtn_u32 v0, v1, v2
# CHECK-NEXT: ds_add_u32 v1, v2
# CHECK-NEXT: flat_atomic_add_u32 v0, v[2:3], v4
# CHECK-NOT: diag:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
ds_add_rtn_u32 v0, v1, v2
ds_add_u32 v1, v2
flat_atomic_add_u32 v0, v[2:3], v4 th:TH_ATOMIC_RETURN
region_end:
s_endpgm
