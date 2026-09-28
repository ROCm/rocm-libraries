# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyRemoveNopPass --from-label .Lregion_begin --to-label .Lregion_end --emit-asm
#
# --from-label/--to-label must recognize dot-prefixed local labels
# (e.g. ".Lregion_begin:"), the style compilers emit for loop bodies,
# not just plain identifier labels. The region between the labels gets
# passes applied (StinkyRemoveNopPass removes the nop); everything
# outside is emitted verbatim.
#
# CHECK: .Lregion_begin:
# CHECK-NOT: s_nop
# CHECK: .Lregion_end:
# CHECK: s_nop 0
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.Lregion_begin:
s_nop 0
.Lregion_end:
s_nop 0
s_endpgm
