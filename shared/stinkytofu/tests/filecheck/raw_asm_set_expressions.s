# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>&1; %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb16 --RaiseVgprMsbPass --InsertVgprMsbPass --allow-opaque-text --from-label region_begin --to-label region_end 2>&1; echo "exit=$?"; %stinkytofu-opt --arch gfx1250 %s --InsertVgprMsbPass --from-label region_begin --to-label region_end 2>&1
#
# .set values and register brackets are evaluated the way the assembler reads
# them: hex, binary and leading-zero octal literals, parentheses, checked
# arithmetic. vgprA = 0x100 = 256, so v[vgprA] is a bank-1 register that
# needs an MSB setter; it used to become an opaque string operand that no pass
# saw. A value too large for an int leaves the symbol undefined instead of
# aborting the tool, and a register that uses it is kept as text, which
# passes refuse to run over. A literal above INT_MAX keeps its spelling instead
# of being deleted from the operand list.

# No passes: values resolved, nothing dropped.
# CHECK:      Warning: 'v_mov_b32': register expression not resolvable; storing as text
# CHECK-NOT:  Warning
# CHECK:      region_begin:
# CHECK-NEXT: v_add_f32 v12, v256, v257
# CHECK-NEXT: v_add_f32 v13, v264, v264
# CHECK-NEXT: v_mov_b32 v40, v8
# CHECK-NEXT: v_mov_b32 v3, v7
# CHECK-NEXT: v_mov_b32 v[big], v2
# CHECK-NEXT: s_mov_b32 s0, 4294967295
# CHECK-NEXT: s_and_b32 s0, s1, 3000000000
# CHECK-NEXT: region_end:

# MSB passes see the bank-1 registers.
# CHECK:      region_begin:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 5
# CHECK-NEXT: v_add_f32 v12, v[256-256], v[257-256]
# CHECK-NEXT: v_add_f32 v13, v[264-256], v[264-256]
# CHECK:      v_mov_b32 v[big], v2
# CHECK:      s_mov_b32 s0, 4294967295
# CHECK-NEXT: s_and_b32 s0, s1, 3000000000
# CHECK:      exit=0

# Without --allow-opaque-text the unresolved line stops the passes.
# CHECK:      Error: {{.*}}contains an instruction kept as text
# CHECK-NEXT: v_mov_b32 v[big], v2

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.set vgprA, 0x100
.set vgprB, 12
.set vgprC, (vgprA + 0b100) * 2 - 0x100
.set vgprD, (vgprB + 4) * 2 + 010
.set vgprE, 3
.set big, 99999999999
region_begin:
v_add_f32 v[vgprB], v[vgprA], v[vgprA+1]
v_add_f32 v[vgprB+1], v[vgprC], v[vgprC]
v_mov_b32 v[vgprD], v[010]
v_mov_b32 v[vgprE], v[(vgprE + 1) + vgprE]
v_mov_b32 v[big], v2
s_mov_b32 s0, 4294967295
s_and_b32 s0, s1, 3000000000
region_end:
s_endpgm
