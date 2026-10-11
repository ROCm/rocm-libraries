# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --emit-asm 2>/dev/null
#
# An unchanged hwreg(...) operand prints as written: symbolic HW_REG_* names,
# spacing and explicit default fields (byte identity on a round trip with no
# passes). Before: hwreg(26,0,2), hwreg(1) and hwreg(1,0,4).

# CHECK:      s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2{{$}}
# CHECK-NEXT: s_setreg_imm32_b32 hwreg(26, 0, 2), 2{{$}}
# CHECK-NEXT: s_getreg_b32 s0, hwreg(HW_REG_WAVE_MODE){{$}}
# CHECK-NEXT: s_getreg_b32 s0, hwreg(HW_REG_WAVE_MODE, 0, 32){{$}}
# CHECK-NEXT: s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), s1{{$}}

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
s_setreg_imm32_b32 hwreg(26, 0, 2), 2
s_getreg_b32 s0, hwreg(HW_REG_WAVE_MODE)
s_getreg_b32 s0, hwreg(HW_REG_WAVE_MODE, 0, 32)
s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), s1
s_endpgm
