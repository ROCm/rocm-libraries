# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm --preserve-symbolic-regs
#
# Regression: a `.set` symbol used as an instruction source operand must
# survive the parse → IR → emit round-trip.
#
# Background:
#   parseOneRegister() recognises operands like "v2", "s5", "acc12", and
#   bracketed forms like "v[vgprFoo+0]". A bare identifier such as `MT0`
#   (defined elsewhere via `.set MT0, 64`) was previously rejected as an
#   "unknown register type" and the function returned std::nullopt. For a
#   non-first operand, parseInstLine then BROKE out of the operand loop and
#   the source was silently dropped:
#
#       Source:  v_mov_b32 v2, MT0
#       Before:  v_mov_b32 v2          <- source operand truncated
#       After :  v_mov_b32 v2, MT0     <- preserved as a LiteralString
#
# The fix preserves any unrecognised non-bracketed identifier in a
# non-first operand position as a LiteralString register so the operand
# round-trips verbatim. (Bracketed forms like `foo[12]` still bail out and
# trigger TEXTBLOCK fallback so a malformed register expression doesn't
# get silently mangled.)
#
# The recovery is gated on `opIdx > 0` so that custom-syntax mnemonics
# whose first operand is an identifier (e.g. `s_delay_alu instid0(...)`,
# `s_wait_alu depctr_va_vdst(0)`) still bail to TEXTBLOCK pass-through
# instead of being half-parsed and then tripping
# `SDelayAluData/SWaitAluData expected` asserts in the emitter.
#
# Bare arithmetic expressions used as immediate sources (e.g. `34816/2`,
# `MT0/2`, `vgprBase+0`, `NumPersistIters-1`) are emitted by the lexer as
# a chain of separate tokens (IntegerLiteral, Unknown('/'),
# IntegerLiteral). Without folding the chain back into a single
# LiteralString the operand loop would only see the leading token and
# silently drop the rest:
#
#       Source:  s_mul_i32 s86, s[sgprWaveId], 34816/2
#       Before:  s_mul_i32 s86, s[sgprWaveId], 34816
#       After :  s_mul_i32 s86, s[sgprWaveId], 34816/2
#
# Subtraction is asymmetric: the lexer special-cases `-<digit>` and emits
# the whole signed number as one IntegerLiteral (no separate '-' token —
# see IRLexer.cpp `case '-':`). gatherArithExprSuffix therefore also
# accepts a signed-literal continuation as an implicit `-<num>` so e.g.
# `NumPersistIters-1` and `34816-512` survive the round-trip.
#
# CHECK: .set MT0, 64
# CHECK: .set DepthU, 32
# CHECK: v_mov_b32 v2, MT0
# CHECK: v_mov_b32 v3, DepthU
# CHECK: s_mov_b32 s5, MT0
# CHECK: s_mul_i32 s86, s[sgprWaveId], 34816/2
# CHECK: s_mul_i32 s87, s[sgprWaveId], 0xff/2
# CHECK: s_mul_i32 s88, s[sgprWaveId], 34816-512
# CHECK: s_cmp_lt_u32 s[sgprWaveId], NumPersistIters-1
# CHECK: s_add_u32 s5, s[sgprWaveId], MT0+DepthU-1
# CHECK: s_delay_alu instid0(VALU_DEP_2)
# CHECK: s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
#
# tensor_load_to_lds carries gfx12+ memory-hint modifiers like `th:TH_LOAD_NT`
# (temporal hint, identifier-valued). parseModifiers stores `key:Identifier`
# values in its generic fields collection, but TENSOR-format instructions
# have no modKey mapping at all — so the existing `modKey.empty() &&
# sawAnyModifier` check still routes them to TEXTBLOCK pass-through, keeping
# the modifier text intact instead of silently dropping it.
#
# Arithmetic-expression modifier values like `offset:2*8704*2+32` are
# evaluated to a single integer via the symbol table; the offset
# round-trips as `offset:34848`. The evaluator supports literal-only
# chains and symbol-bearing chains using `+ - * /` with standard precedence.
#
# `v_wmma_scale_f32_*` matrix_*_fmt / matrix_*_reuse parse into typed
# MatrixFmtModifiers + MFMAModifiers. The round-trip preserves the
# modifier emission order from the input text: matrix_a_fmt /
# matrix_b_fmt come first, then matrix_b_reuse — exactly as the source
# asm wrote them.
#
# CHECK: tensor_load_to_lds s[12:15], s[16:23] th:TH_LOAD_NT
# CHECK: ds_load_b128 v[0:3], v[4] offset:34848
# CHECK: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:31], 0, v[32], v[33] matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP4 matrix_b_reuse
# CHECK: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.set MT0, 64
.set DepthU, 32
.set sgprWaveId, 4

v_mov_b32 v2, MT0
v_mov_b32 v3, DepthU
s_mov_b32 s5, MT0
s_mul_i32 s86, s[sgprWaveId], 34816/2
s_mul_i32 s87, s[sgprWaveId], 0xff/2
s_mul_i32 s88, s[sgprWaveId], 34816-512
s_cmp_lt_u32 s[sgprWaveId], NumPersistIters-1
s_add_u32 s5, s[sgprWaveId], MT0+DepthU-1
s_delay_alu instid0(VALU_DEP_2)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
tensor_load_to_lds s[12:15], s[16:23] th:TH_LOAD_NT
ds_load_b128 v[0:3], v[4] offset:2*8704*2+32
v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:31], 0, v[32], v[33] matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP4 matrix_b_reuse
s_endpgm
