# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# IRConverter directive round-trip:
#
# (a) TEXTBLOCK pass-through: arbitrary text that the parser cannot decode
#     as an instruction (e.g. /* ... */ comment blocks, banner-style
#     comments) is wrapped as an asm_directive with a "TEXTBLOCK" sentinel
#     by RawAsmParser::makeTextBlock. IRConverter must recognize the
#     sentinel and set AsmDirectiveKind::TEXTBLOCK so the emitter prints
#     the text verbatim, instead of letting the kind default to SET (whose
#     emitter concatenates name + symbol and would print
#     "TEXTBLOCK <raw text>").
#
# (b) .set value round-trip: RawAsmParser packs `.set` as
#     srcRegs = {".set", symbol, value}. IRConverter must wire srcRegs[2]
#     into AsmDirective::value; otherwise the SET emitter branch (which
#     prints "name symbol[, value]") drops the assignment and a line like
#     ".set vgprValuMXSA_X0_I0_BASE, vgprMXSBase+0" round-trips as the
#     incomplete ".set vgprValuMXSA_X0_I0_BASE".
#
# Banner-style and inline /* ... */ comments must appear verbatim with no
# "TEXTBLOCK" sentinel leaking anywhere into the output.
#
# CHECK-NOT: TEXTBLOCK
# CHECK: /* Mapping of Acc register -> C Vgpr register */
# CHECK: /******************************************/
# CHECK: /* Begin Kernel                           */
# CHECK: /******************************************/
# CHECK: .set vgprBase, 0
# CHECK: .set vgprValuMXSA_X0_I0_BASE, vgprBase+0
# CHECK: v_mov_b32 v0, v1
# CHECK-NOT: TEXTBLOCK

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

/* Mapping of Acc register -> C Vgpr register */
/******************************************/
/* Begin Kernel                           */
/******************************************/

.set vgprBase, 0
.set vgprValuMXSA_X0_I0_BASE, vgprBase+0

v_mov_b32 v0, v1
s_endpgm
