# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm --preserve-symbolic-regs
#
# Verify that --preserve-symbolic-regs makes the parser/emitter pipeline
# round-trip register operands using their original symbolic form
# (e.g. "v[vgprSerial-512]") instead of the resolved numeric form (v248).
#
# Two cases are exercised:
#
# (1) Single-register form with an inline offset: regression test for a
#     double-wrap bug where the parser stored the fully-bracketed name
#     ("v[vgprSerial-512]") in setSymbolicName(); the emitter, which adds
#     the type prefix and brackets itself, then produced the malformed
#     "v[v[vgprSerial-512]]". The parser must store ONLY the inner content
#     (e.g. "vgprSerial-512") so the emitter rebuilds the operand
#     correctly as "v[vgprSerial-512]".
#
# (2) Range form: "v[vgprDest+0:vgprDest+3]" — the symbolic name itself
#     contains a ':' separator, so the emitter must detect this self-
#     contained range and emit it verbatim rather than constructing a
#     synthetic "v[<sym>:<sym>+(num-1)]" range (which would also produce
#     a malformed double-colon line for this input).
#
# CHECK: v_mov_b32 v[vgprSerial-512], v[vgprSerialPersist-768]
# CHECK: ds_load_b128 v[vgprDest+0:vgprDest+3], v[vgprAddr]
#
# Regression guards:
# CHECK-NOT: v[v[
# CHECK-NOT: v248
# CHECK-NOT: v255

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.set vgprSerial, 760
.set vgprSerialPersist, 1023
.set vgprDest, 100
.set vgprAddr, 50

v_mov_b32 v[vgprSerial-512], v[vgprSerialPersist-768]
ds_load_b128 v[vgprDest+0:vgprDest+3], v[vgprAddr]
s_endpgm
