# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>/dev/null
#
# Numeric s_wait_alu / s_delay_alu operands are decoded field by field (the
# DEPCTR and simm16 layouts llvm-mc uses), so they no longer print as a bare
# `s_wait_alu` that does not assemble; an unchanged s_wait_alu word prints as
# written (decoder: WaitAluDataTest). Words this model cannot spell (reserved
# DEPCTR bits set, an out-of-range VALU_DEP, an unterminated hwreg) are kept as
# verbatim text instead of being degraded or aborting the tool.

# CHECK-LABEL: region_begin:
# CHECK-NEXT:  s_wait_alu 0{{$}}
# CHECK-NEXT:  s_wait_alu 0xff9e{{$}}
# CHECK-NEXT:  s_wait_alu 0x3f87{{$}}
# CHECK-NEXT:  s_wait_alu 0xfffe
# CHECK-NEXT:  s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
# CHECK-NEXT:  s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
# CHECK-NEXT:  s_delay_alu instid0(VALU_DEP_9)
# CHECK-NEXT:  s_getreg_b32 s3, hwreg(26, 0, 2
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_wait_alu 0
s_wait_alu 0xff9e
s_wait_alu 0x3f87
s_wait_alu 0xfffe
s_delay_alu 0x91
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
s_delay_alu instid0(VALU_DEP_9)
s_getreg_b32 s3, hwreg(26, 0, 2
region_end:
s_endpgm
