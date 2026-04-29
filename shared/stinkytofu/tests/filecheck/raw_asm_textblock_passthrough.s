# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm
#
# TEXTBLOCK pass-through round-trip: arbitrary text that the parser cannot
# decode as an instruction (e.g. /* ... */ comment blocks, banner-style
# comments) is wrapped as an asm_directive with a "TEXTBLOCK" sentinel by
# RawAsmParser::makeTextBlock. IRConverter must recognize the sentinel and
# set AsmDirectiveKind::TEXTBLOCK so the emitter prints the text verbatim
# instead of producing "TEXTBLOCK <raw text>" (kind would default to SET
# and the SET branch concatenates name + symbol).
#
# Each comment must round-trip exactly as authored, with no "TEXTBLOCK"
# prefix injected anywhere in the output.
#
# CHECK-NOT: TEXTBLOCK
# CHECK: /* Mapping of Acc register -> C Vgpr register */
# CHECK: /******************************************/
# CHECK: /* Begin Kernel                           */
# CHECK: /******************************************/
# CHECK: v_mov_b32 v0, v1
# CHECK-NOT: TEXTBLOCK

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

/* Mapping of Acc register -> C Vgpr register */
/******************************************/
/* Begin Kernel                           */
/******************************************/

v_mov_b32 v0, v1
s_endpgm
