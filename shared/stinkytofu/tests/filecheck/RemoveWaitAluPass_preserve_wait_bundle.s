# RUN: t=$(mktemp -d); %stinkytofu-opt --arch gfx1250 %s --from-label bundle_begin --to-label bundle_end > $t/original.s 2>/dev/null; %stinkytofu-opt --arch gfx1250 %s --from-label bundle_begin --to-label bundle_end --RaiseVgprMsbPass --RemoveWaitAluPass --InsertVgprMsbPass --InsertWaitAluPass > $t/transformed.s 2>/dev/null; cmp -s $t/original.s $t/transformed.s && echo "bundled region identical"; echo "=== ordinary"; %stinkytofu-opt --arch gfx1250 %s --from-label ordinary_begin --to-label ordinary_end --RaiseVgprMsbPass --RemoveWaitAluPass --InsertVgprMsbPass --InsertWaitAluPass 2>/dev/null; rm -rf $t
#
# An existing wait-alu immediately before a waitcnt is an intentional wait
# bundle: both scoreboards drain at one placement point.  Moving the VA_VDST
# drain to the eventual reader splits that bundle and can put a strict wait in
# a WMMA/DS issue run (the L2n4ws regression).  The per-pipe pass cannot prove
# that rescheduling around this pre-existing drain is profitable, so both wait
# passes preserve the function byte-for-byte.  A function without such a
# bundle remains eligible for normal transformation.

# CHECK: bundled region identical
# CHECK: === ordinary
# CHECK-LABEL: ordinary_begin:
# CHECK:      v_add_u32 v10, v1, v2
# CHECK-NEXT: v_add_u32 v11, v3, v4
# CHECK-NEXT: s_wait_alu depctr_va_vdst(1)
# CHECK-NEXT: buffer_store_b32 v10, v20, s[4:7], s3 offen offset:16

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
bundle_begin:
s_set_vgpr_msb 0x1
ds_load_b32 v0, v1
s_set_vgpr_msb 0x0
v_add_u32 v10, v1, v2
s_wait_alu depctr_va_vdst(0)
s_wait_dscnt 6
buffer_store_b32 v10, v20, s[4:7], s3 offen offset:16
bundle_end:

ordinary_begin:
v_add_u32 v10, v1, v2
s_wait_alu depctr_va_vdst(0)
v_add_u32 v11, v3, v4
buffer_store_b32 v10, v20, s[4:7], s3 offen offset:16
ordinary_end:
s_endpgm
