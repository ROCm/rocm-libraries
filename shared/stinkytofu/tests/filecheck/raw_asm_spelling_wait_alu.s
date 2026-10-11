# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --emit-asm 2>/dev/null
#
# A numeric s_wait_alu word is printed as written while its decoded fields are
# unchanged (byte identity on a round trip with no passes). Before: `0` and
# `0x0` became all seven depctr_* fields, `0xff9e` became depctr_sa_sdst(0).
# The field spelling is kept as it was.

# CHECK:      s_wait_alu 0{{$}}
# CHECK-NEXT: s_wait_alu 0x0{{$}}
# CHECK-NEXT: s_wait_alu 0xff9e{{$}}
# CHECK-NEXT: s_wait_alu depctr_va_vdst(0){{$}}

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
s_wait_alu 0
s_wait_alu 0x0
s_wait_alu 0xff9e
s_wait_alu depctr_va_vdst(0)
s_endpgm
