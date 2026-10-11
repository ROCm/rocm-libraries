# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# RawAsmParser → StinkyAsmEmitter round-trip for hwreg() operand syntax.
# Verifies that:
# - All three forms (id only / id+offset / id+offset+size) parse cleanly.
# - Each unchanged operand prints as written (symbolic id, explicit defaults);
#   id resolution and the canonical form are unit-tested (HwRegOperandTest).
# - s_setreg* (dest=hwreg) and s_getreg_b32 (src=hwreg) slot directions preserved.

test_start:
    # id only — default offset=0/size=32, both suppressed on emit.
    s_getreg_b32 s0, hwreg(1)

    # id + offset only — default size=32 suppressed on emit.
    s_getreg_b32 s1, hwreg(2,4)

    # full triple, all explicit.
    s_setreg_b32 hwreg(19,4,8), s2

    # SCHED_MODE numeric id.
    s_setreg_IMM32_b32 hwreg(26,0,2), 2

    # Symbolic id resolves through nameToId (HW_REG_WAVE_MODE -> 1 on gfx1250).
    s_setreg_IMM32_b32 hwreg(HW_REG_WAVE_MODE,0,2), 3

    # Explicit size=32 round-trips through default-suppression.
    s_getreg_b32 s3, hwreg(7,0,32)
test_end:

# CHECK-LABEL: test_start:

# id only, defaults suppressed.
# CHECK: s_getreg_b32 s0, hwreg(1)

# id + offset, size default suppressed.
# CHECK: s_getreg_b32 s1, hwreg(2,4)

# Full triple preserved.
# CHECK: s_setreg_b32 hwreg(19,4,8), s2

# Numeric id preserved.
# CHECK: s_setreg_IMM32_b32 hwreg(26,0,2), 2

# Symbolic id kept as written.
# CHECK: s_setreg_IMM32_b32 hwreg(HW_REG_WAVE_MODE,0,2), 3

# Explicit size=32 kept as written.
# CHECK: s_getreg_b32 s3, hwreg(7,0,32)

# CHECK-LABEL: test_end:
