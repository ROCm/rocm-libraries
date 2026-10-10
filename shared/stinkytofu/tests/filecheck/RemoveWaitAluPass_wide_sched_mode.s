# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --RemoveWaitAluPass; echo "=== exact"; T=$(mktemp --suffix=.s); sed 's/hwreg(26,0,3), 6/hwreg(26,0,2), 2/' %s > $T; %stinkytofu-opt --arch gfx1250 $T --whole-function --RemoveWaitAluPass; rm -f $T
#
# RemoveWaitAlu at kernel scope deletes the DEP_MODE writes and clears the
# va_vdst/vm_vsrc waits, which leaves mode 0 even if nothing re-inserts waits.
# A SCHED_MODE write that also sets other bits (here DEP_MODE = 2 plus
# bit 2) cannot be deleted; clearing the waits would leave mode 2 on without
# them, so the function is left alone. The exact DEP_MODE write still goes.

# CHECK:      s_setreg_imm32_b32 hwreg(26,0,3), 6
# CHECK-NEXT: v_add_f32 v1, v2, v3
# CHECK-NEXT: s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT: v_mov_b32 v4, v1
# CHECK:      === exact
# CHECK-NOT:  s_setreg
# CHECK:      v_add_f32 v1, v2, v3
# CHECK-NEXT: v_mov_b32 v4, v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
kernel_start:
s_setreg_imm32_b32 hwreg(26,0,3), 6
v_add_f32 v1, v2, v3
s_wait_alu depctr_va_vdst(0)
v_mov_b32 v4, v1
s_endpgm
