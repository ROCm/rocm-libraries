# RUN: %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb16 --CFGBuilderPass --RaiseVgprMsbPass --InsertVgprMsbPass --from-label label_loop --to-after-branch-to label_loop
#
# The input sets MSB back to 0 before the loop's back-branch, and the code after
# the region (ds_load_b32 v5, ...) was written for MSB 0. Raise absorbs that
# setter and Insert has no VGPR use after it to re-emit it, so without a restore
# the region exits with dst MSB 1 and the ds_load writes v261 instead of v5.

# CHECK-LABEL: label_loop:
# CHECK:       v_add_f32 v[257-256], v[258-256], v[259-256]
# Insert re-encodes only the slots v_add_f32 uses (0x45); msb16 carries it in
# the high byte of the restoring setter: 0x4500.
# CHECK:       s_set_vgpr_msb 17664
# CHECK-NEXT:  s_cbranch_scc0 label_loop
# CHECK-NEXT:  ds_load_b32 v5, v6

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
label_loop:
s_set_vgpr_msb 0x55
v_add_f32 v1, v2, v3
s_set_vgpr_msb 0
s_cbranch_scc0 label_loop
ds_load_b32 v5, v6
s_endpgm
