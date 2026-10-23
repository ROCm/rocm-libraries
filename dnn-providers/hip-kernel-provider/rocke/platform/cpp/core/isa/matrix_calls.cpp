// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Matrix intrinsic calls described as data: the table, the schema checks and
 * the declaration derivation. Mirrors core/isa/matrix_calls.py and the type
 * half of core/isa/matrix_operand_forms.py; the emission half lives in
 * cpp/core/lower_llvm/matrix_call.cpp. */
#include "rocke/matrix_calls_internal.h"

#include <mutex>
#include <set>
#include <string>
#include <utility>

#include <string.h>

#include "rocke/error.hpp"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/mfma_scale_internal.h"

namespace ckc
{

/* A dense CDNA MFMA (Python _cdna_mfma): (A, B, C, cbsz, abid, blgp) with
 * cbsz, abid and blgp all 0; A and B share one register type and the result
 * takes the accumulator's. */
static const rocke_matrix_arg_t k_cdna_mfma_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i32", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i32", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i32", 0, true},
};

static constexpr rocke_matrix_form_t as_is(rocke_matrix_register_t reg)
{
    return rocke_matrix_form_t{ROCKE_MATRIX_FORM_AS_IS, reg};
}

static constexpr rocke_matrix_register_t reg(const char* elem, int count)
{
    return rocke_matrix_register_t{{elem, count}, NULL, 0};
}

static constexpr rocke_matrix_call_t
    cdna_mfma(const char* intrinsic, rocke_matrix_register_t operand, rocke_matrix_register_t acc)
{
    return rocke_matrix_call_t{intrinsic,
                               k_cdna_mfma_args,
                               (int)(sizeof(k_cdna_mfma_args) / sizeof(k_cdna_mfma_args[0])),
                               as_is(operand),
                               as_is(operand),
                               as_is(acc),
                               as_is(acc),
                               NULL};
}

/* A WMMA (Python _wmma) whose A and B share one register type and whose
 * accumulator and result share another, the arguments in `args` order. */
template <int N>
static constexpr rocke_matrix_call_t wmma(const char* intrinsic,
                                          const rocke_matrix_arg_t (&args)[N],
                                          rocke_matrix_register_t operand,
                                          rocke_matrix_register_t acc,
                                          const char* decl_key = NULL)
{
    return rocke_matrix_call_t{
        intrinsic, args, N, as_is(operand), as_is(operand), as_is(acc), as_is(acc), decl_key};
}

/* gfx12 fp8 / bf8 WMMA: (A, B, C). */
static const rocke_matrix_arg_t k_gfx12_wmma_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
};

/* Integer WMMA (Python _INTEGER_WMMA_ARGS): (signed A, A, signed B, B, C,
 * clamp). Each flag comes from its operand's dtype in the catalog row; the
 * result is not clamped. */
static const rocke_matrix_arg_t k_integer_wmma_args[] = {
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, true, ROCKE_MATRIX_ARG_A},
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, true, ROCKE_MATRIX_ARG_B},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};

/* The same arguments for the gfx11 integer WMMAs (Python
 * _GFX11_INTEGER_WMMA_ARGS), whose hand-written declarations spell the three
 * i1 parameters without immarg. */
static const rocke_matrix_arg_t k_gfx11_integer_wmma_args[] = {
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, false, ROCKE_MATRIX_ARG_A},
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, false, ROCKE_MATRIX_ARG_B},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, false},
};

/* One integer WMMA's call under each of its op_ids (Python _integer_wmma):
 * `op_id` itself for signed A and B, and `op_id` with _u<bits>_i<bits> (A
 * unsigned), _i<bits>_u<bits> (B unsigned) or _u<bits>_u<bits> appended. */
#define ROCKE_INTEGER_WMMA(op_id, bits, call)                                                \
    {op_id, call}, {op_id "_u" #bits "_i" #bits, call}, {op_id "_i" #bits "_u" #bits, call}, \
    {                                                                                        \
        op_id "_u" #bits "_u" #bits, call                                                    \
    }

/* gfx12 16-bit-output WMMA (Python _GFX12_16BIT_OUT_ARGS): (A, B, C, op_sel),
 * op_sel 0. */
static const rocke_matrix_arg_t k_gfx12_16bit_out_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};

/* gfx11 16-bit-output WMMA (Python _gfx11_16bit_out_wmma): (A, B, C, op_sel)
 * with A, B, C and the result all in the <16 x 16-bit> `reg`. The instruction
 * keeps its eight results per lane in the even 16-bit halves of that register
 * when op_sel is 0, so the accumulator and the result are the eight logical
 * values (EvenHalves) and op_sel is always 0. */
static const rocke_matrix_arg_t k_gfx11_16bit_out_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};

static constexpr rocke_matrix_call_t gfx11_16bit_out_wmma(const char* intrinsic,
                                                          rocke_matrix_register_t reg)
{
    return rocke_matrix_call_t{
        intrinsic,
        k_gfx11_16bit_out_args,
        (int)(sizeof(k_gfx11_16bit_out_args) / sizeof(k_gfx11_16bit_out_args[0])),
        as_is(reg),
        as_is(reg),
        rocke_matrix_form_t{ROCKE_MATRIX_FORM_EVEN_HALVES, reg},
        rocke_matrix_form_t{ROCKE_MATRIX_FORM_EVEN_HALVES, reg},
        NULL};
}

/* The gfx1250 16-bit, fp32 and integer WMMA modifiers (Python
 * _GFX1250_WMMA_ARGS, _GFX1250_FP8_WMMA_ARGS, _GFX1250_INTEGER_WMMA_ARGS). A
 * 16-bit or fp32 WMMA takes (neg A, A, neg B, B, C modifier, C, reuse A,
 * reuse B); an fp8 / bf8 one (A, B, C modifier, C, reuse A, reuse B); an
 * integer one (signed A, A, signed B, B, C, clamp, reuse A, reuse B), each
 * flag from its operand's dtype in the catalog row. Every other modifier is
 * 0. */
static const rocke_matrix_arg_t k_gfx1250_wmma_args[] = {
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i16", 0, true},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};
static const rocke_matrix_arg_t k_gfx1250_fp8_wmma_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i16", 0, true},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};
static const rocke_matrix_arg_t k_gfx1250_integer_wmma_args[] = {
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, true, ROCKE_MATRIX_ARG_A},
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_SIGNEDNESS, NULL, 0, true, ROCKE_MATRIX_ARG_B},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
    {ROCKE_MATRIX_ARG_IMM, "i1", 0, true},
};

/* gfx1250 bf16 x bf16 with an fp32 accumulator and a bf16 result (Python
 * _GFX1250_BF16F32), every modifier 0. The result type differs from the
 * accumulator's, so the call returns <8 x bfloat> while C passes as
 * <8 x float>. */
static constexpr rocke_matrix_call_t k_gfx1250_bf16f32
    = {"llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16.v8f32",
       k_gfx1250_wmma_args,
       (int)(sizeof(k_gfx1250_wmma_args) / sizeof(k_gfx1250_wmma_args[0])),
       as_is(reg("bf16", 16)),
       as_is(reg("bf16", 16)),
       as_is(reg("f32", 8)),
       as_is(reg("bf16", 8)),
       NULL};

/* Eight fp8 / bf8 values: two words at LLVM 20, one i64 from LLVM 22 on (the
 * same register types the hand-written fp8 x fp8 and bf8 x bf8 MFMAs use). */
static const rocke_matrix_flavor_override_t k_i64_from_llvm22[] = {{"llvm22", {"i64", 0}}};
static constexpr rocke_matrix_register_t k_fp8_x8 = {{"i32", 2}, k_i64_from_llvm22, 1};

/* The form an f8f6f4 operand of the matrix format with code `format` takes in
 * `reg` (Python f8f6f4_operand_form): an 8-bit format passes as is or with one
 * bitcast; a 6-bit or 4-bit format arrives as packed words, zero-padded. */
static constexpr rocke_matrix_form_t f8f6f4_operand_form(int format, rocke_matrix_register_t reg)
{
    return rocke_matrix_form_t{rocke_matrix_format_bits(format) == 8
                                   ? ROCKE_MATRIX_FORM_AS_IS
                                   : ROCKE_MATRIX_FORM_ZERO_PADDED,
                               reg};
}

/* gfx1250 f4 32x16x128 (Python _GFX1250_F4_32X16X128): (A, B, C modifier, C),
 * the C modifier 0. 128 fp4 values of A per lane in sixteen words and 64 of B
 * in eight, each zero-padded from fewer packed words; sixteen fp32 results. */
static const rocke_matrix_arg_t k_gfx1250_f4_args[] = {
    {ROCKE_MATRIX_ARG_A, NULL, 0, false},
    {ROCKE_MATRIX_ARG_B, NULL, 0, false},
    {ROCKE_MATRIX_ARG_IMM, "i16", 0, true},
    {ROCKE_MATRIX_ARG_C, NULL, 0, false},
};
static constexpr int k_fp4_format = 4;

/* op_id -> call, for every MMA instruction lowered from data rather than by a
 * hand-written handler and not generated below (Python MATRIX_CALLS, same
 * order). */
static constexpr rocke_matrix_call_entry_t k_listed_calls[] = {
    /* CDNA i8 -> i32: four i8 per lane in one word (CDNA1-2), eight in an i64
     * (CDNA3), sixteen in four words (CDNA4). */
    {"mfma_i32_16x16x16_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.16x16x16i8", reg("i32", 0), reg("i32", 4))},
    {"mfma_i32_32x32x8_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.32x32x8i8", reg("i32", 0), reg("i32", 16))},
    {"mfma_i32_16x16x32_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.16x16x32.i8", reg("i64", 0), reg("i32", 4))},
    {"mfma_i32_32x32x16_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.32x32x16.i8", reg("i64", 0), reg("i32", 16))},
    {"mfma_i32_16x16x64_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.16x16x64.i8", reg("i32", 4), reg("i32", 4))},
    {"mfma_i32_32x32x32_i8",
     cdna_mfma("llvm.amdgcn.mfma.i32.32x32x32.i8", reg("i32", 4), reg("i32", 16))},
    /* CDNA bf16 without the _1k suffix: two bf16 per lane as <2 x i16>. */
    {"mfma_f32_16x16x8_bf16",
     cdna_mfma("llvm.amdgcn.mfma.f32.16x16x8bf16", reg("i16", 2), reg("f32", 4))},
    {"mfma_f32_32x32x4_bf16",
     cdna_mfma("llvm.amdgcn.mfma.f32.32x32x4bf16", reg("i16", 2), reg("f32", 16))},
    /* CDNA3+ mixed fp8 x bf8 and bf8 x fp8. */
    {"mfma_f32_16x16x32_fp8_bf8",
     cdna_mfma("llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8", k_fp8_x8, reg("f32", 4))},
    {"mfma_f32_16x16x32_bf8_fp8",
     cdna_mfma("llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8", k_fp8_x8, reg("f32", 4))},
    {"mfma_f32_32x32x16_fp8_bf8",
     cdna_mfma("llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8", k_fp8_x8, reg("f32", 16))},
    {"mfma_f32_32x32x16_bf8_fp8",
     cdna_mfma("llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8", k_fp8_x8, reg("f32", 16))},
    /* gfx11 i8 -> i32: sixteen i8 per lane in four words; i4 -> i32: sixteen
     * i4 per lane in two words. */
    ROCKE_INTEGER_WMMA("wmma_i32_16x16x16_iu8",
                       8,
                       wmma("llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32",
                            k_gfx11_integer_wmma_args,
                            reg("i32", 4),
                            reg("i32", 8),
                            "wmma.i32.16x16x16.iu8")),
    ROCKE_INTEGER_WMMA("wmma_i32_16x16x16_iu4",
                       4,
                       wmma("llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32",
                            k_gfx11_integer_wmma_args,
                            reg("i32", 2),
                            reg("i32", 8),
                            "wmma.i32.16x16x16.iu4")),
    /* gfx11 f16 -> f16 and bf16 -> bf16: sixteen 16-bit values of A and of B
     * per lane, eight logical results in the even halves of the 16-slot
     * register; the bf16 form takes and returns them as i16. */
    {"wmma_f16_16x16x16_f16",
     gfx11_16bit_out_wmma("llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16", reg("f16", 16))},
    {"wmma_bf16_16x16x16_bf16",
     gfx11_16bit_out_wmma("llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16", reg("i16", 16))},
    /* gfx12 fp8 / bf8 -> f32: eight 8-bit values per lane in two words. */
    {"wmma_gfx12_f32_16x16x16_fp8_fp8",
     wmma("llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32",
          k_gfx12_wmma_args,
          reg("i32", 2),
          reg("f32", 8))},
    {"wmma_gfx12_f32_16x16x16_fp8_bf8",
     wmma("llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8.v8f32.v2i32",
          k_gfx12_wmma_args,
          reg("i32", 2),
          reg("f32", 8))},
    {"wmma_gfx12_f32_16x16x16_bf8_fp8",
     wmma("llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8.v8f32.v2i32",
          k_gfx12_wmma_args,
          reg("i32", 2),
          reg("f32", 8))},
    {"wmma_gfx12_f32_16x16x16_bf8_bf8",
     wmma("llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8.v8f32.v2i32",
          k_gfx12_wmma_args,
          reg("i32", 2),
          reg("f32", 8))},
    /* gfx12 i8 -> i32: eight i8 per lane in two words. */
    ROCKE_INTEGER_WMMA("wmma_gfx12_i32_16x16x16_iu8",
                       8,
                       wmma("llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32",
                            k_integer_wmma_args,
                            reg("i32", 2),
                            reg("i32", 8))),
    /* gfx12 i4 -> i32: eight i4 per lane in one word (K 16), sixteen in two
     * words (K 32). */
    ROCKE_INTEGER_WMMA("wmma_gfx12_i32_16x16x16_iu4",
                       4,
                       wmma("llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.i32",
                            k_integer_wmma_args,
                            reg("i32", 0),
                            reg("i32", 8))),
    ROCKE_INTEGER_WMMA("wmma_gfx12_i32_16x16x32_iu4",
                       4,
                       wmma("llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32",
                            k_integer_wmma_args,
                            reg("i32", 2),
                            reg("i32", 8))),
    /* gfx12 f16 -> f16 and bf16 -> bf16: eight 16-bit values per lane; the
     * bf16 form takes and returns them as <8 x i16>. */
    {"wmma_gfx12_f16_16x16x16_f16",
     wmma("llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16",
          k_gfx12_16bit_out_args,
          reg("f16", 8),
          reg("f16", 8))},
    {"wmma_gfx12_bf16_16x16x16_bf16",
     wmma("llvm.amdgcn.wmma.bf16.16x16x16.bf16.v8i16.v8i16",
          k_gfx12_16bit_out_args,
          reg("i16", 8),
          reg("i16", 8))},
    /* gfx1250 bf16 -> bf16 through an fp32 accumulator: sixteen bf16 of A and
     * of B per lane, eight fp32 accumulator values, eight bf16 results. */
    {"wmma_gfx1250_bf16f32_16x16x32_bf16", k_gfx1250_bf16f32},
    /* gfx1250 f32 -> f32: two fp32 of A and of B per lane. */
    {"wmma_gfx1250_f32_16x16x4_f32",
     wmma("llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32",
          k_gfx1250_wmma_args,
          reg("f32", 2),
          reg("f32", 8))},
    /* gfx1250 f16 -> f16 and bf16 -> bf16: sixteen 16-bit values of A and of
     * B per lane, eight results; bf16 passes as bfloat. */
    {"wmma_gfx1250_f16_16x16x32_f16",
     wmma("llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16",
          k_gfx1250_wmma_args,
          reg("f16", 16),
          reg("f16", 8))},
    {"wmma_gfx1250_bf16_16x16x32_bf16",
     wmma("llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16",
          k_gfx1250_wmma_args,
          reg("bf16", 16),
          reg("bf16", 8))},
    /* gfx1250 fp8 / bf8 -> f16: 32 8-bit values of A and of B per lane in
     * eight words (K 64), 64 in sixteen words (K 128); eight f16 results. */
    {"wmma_gfx1250_f16_16x16x64_fp8_fp8",
     wmma("llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8.v8f16.v8i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 8),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x64_fp8_bf8",
     wmma("llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8.v8f16.v8i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 8),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x64_bf8_fp8",
     wmma("llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8.v8f16.v8i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 8),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x64_bf8_bf8",
     wmma("llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8.v8f16.v8i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 8),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x128_fp8_fp8",
     wmma("llvm.amdgcn.wmma.f16.16x16x128.fp8.fp8.v8f16.v16i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 16),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x128_fp8_bf8",
     wmma("llvm.amdgcn.wmma.f16.16x16x128.fp8.bf8.v8f16.v16i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 16),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x128_bf8_fp8",
     wmma("llvm.amdgcn.wmma.f16.16x16x128.bf8.fp8.v8f16.v16i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 16),
          reg("f16", 8))},
    {"wmma_gfx1250_f16_16x16x128_bf8_bf8",
     wmma("llvm.amdgcn.wmma.f16.16x16x128.bf8.bf8.v8f16.v16i32",
          k_gfx1250_fp8_wmma_args,
          reg("i32", 16),
          reg("f16", 8))},
    /* gfx1250 i8 -> i32: 32 i8 per lane in eight words. This is the LLVM 23
     * call; LLVM 22 declares the instruction without the clamp argument (its
     * arch-domain column records this declaration as unknown). */
    ROCKE_INTEGER_WMMA("wmma_gfx1250_i32_16x16x64_iu8",
                       8,
                       wmma("llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32",
                            k_gfx1250_integer_wmma_args,
                            reg("i32", 8),
                            reg("i32", 8))),
};

#undef ROCKE_INTEGER_WMMA

namespace
{
/* gfx1250 f8f6f4 -> f32 at 16x16x128 for every format pair with a 6-bit or
 * 4-bit operand (Python _gfx1250_f8f6f4_wmma): (format A, A, format B, B,
 * C modifier, C), the format codes the pair's and the C modifier 0. A and B
 * each take the operand form of their own format in <16 x i32>; the
 * accumulator and the result are <8 x float>. The op_ids and argument lists
 * live in constexpr variables of their own so the entries can point at them. */
constexpr int k_f8f6f4_count = 21;
constexpr auto k_f8f6f4_ids = rocke_make_narrow_f8f6f4_op_ids("wmma_gfx1250_f32_16x16x128_f8f6f4");
struct f8f6f4_args
{
    rocke_matrix_arg_t args[k_f8f6f4_count][6];
};
consteval f8f6f4_args make_f8f6f4_args()
{
    f8f6f4_args result{};
    for(int i = 0; i < k_f8f6f4_count; ++i)
    {
        const rocke_matrix_format_pair& pair = k_rocke_narrow_matrix_format_pairs.pairs[i];
        result.args[i][0] = {ROCKE_MATRIX_ARG_IMM, "i32", pair.a, true};
        result.args[i][1] = {ROCKE_MATRIX_ARG_A, NULL, 0, false};
        result.args[i][2] = {ROCKE_MATRIX_ARG_IMM, "i32", pair.b, true};
        result.args[i][3] = {ROCKE_MATRIX_ARG_B, NULL, 0, false};
        result.args[i][4] = {ROCKE_MATRIX_ARG_IMM, "i16", 0, true};
        result.args[i][5] = {ROCKE_MATRIX_ARG_C, NULL, 0, false};
    }
    return result;
}
constexpr auto k_f8f6f4_args = make_f8f6f4_args();

/* gfx950 f8f6f4 MFMA -> f32 at 16x16x128 and at 32x32x64 for every format
 * pair with a 6-bit or 4-bit operand (Python _gfx950_f8f6f4_mfma): (A, B, C,
 * format A, format B, scale selector A, scale A, scale selector B, scale B),
 * the format codes the pair's and every scale argument 0, which makes the
 * instruction the unscaled f8f6f4 MFMA. A and B each take the operand form of
 * their own format in <8 x i32>. Both declaration keys are the hand-written
 * nine-argument declarations the fp8 16x16x128 MFMA and the scaled 32x32x64
 * MFMA already use. */
constexpr auto k_gfx950_f8f6f4_16x16_ids
    = rocke_make_narrow_f8f6f4_op_ids("mfma_f32_16x16x128_f8f6f4");
constexpr auto k_gfx950_f8f6f4_32x32_ids
    = rocke_make_narrow_f8f6f4_op_ids("mfma_f32_32x32x64_f8f6f4");
struct gfx950_f8f6f4_args
{
    rocke_matrix_arg_t args[k_f8f6f4_count][9];
};
consteval gfx950_f8f6f4_args make_gfx950_f8f6f4_args()
{
    gfx950_f8f6f4_args result{};
    for(int i = 0; i < k_f8f6f4_count; ++i)
    {
        const rocke_matrix_format_pair& pair = k_rocke_narrow_matrix_format_pairs.pairs[i];
        result.args[i][0] = {ROCKE_MATRIX_ARG_A, NULL, 0, false};
        result.args[i][1] = {ROCKE_MATRIX_ARG_B, NULL, 0, false};
        result.args[i][2] = {ROCKE_MATRIX_ARG_C, NULL, 0, false};
        result.args[i][3] = {ROCKE_MATRIX_ARG_IMM, "i32", pair.a, true};
        result.args[i][4] = {ROCKE_MATRIX_ARG_IMM, "i32", pair.b, true};
        result.args[i][5] = {ROCKE_MATRIX_ARG_IMM, "i32", 0, true};
        result.args[i][6] = {ROCKE_MATRIX_ARG_IMM, "i32", 0, false};
        result.args[i][7] = {ROCKE_MATRIX_ARG_IMM, "i32", 0, true};
        result.args[i][8] = {ROCKE_MATRIX_ARG_IMM, "i32", 0, false};
    }
    return result;
}
constexpr auto k_gfx950_f8f6f4_args = make_gfx950_f8f6f4_args();

constexpr int k_listed_count = (int)(sizeof(k_listed_calls) / sizeof(k_listed_calls[0]));
constexpr int k_call_count = k_listed_count + k_f8f6f4_count + 1 + 2 * k_f8f6f4_count;
struct matrix_call_table
{
    rocke_matrix_call_entry_t entries[k_call_count];
};
consteval matrix_call_table make_matrix_call_table()
{
    matrix_call_table result{};
    for(int i = 0; i < k_listed_count; ++i)
        result.entries[i] = k_listed_calls[i];
    for(int i = 0; i < k_f8f6f4_count; ++i)
    {
        const rocke_matrix_format_pair& pair = k_rocke_narrow_matrix_format_pairs.pairs[i];
        result.entries[k_listed_count + i]
            = {k_f8f6f4_ids.ids[i],
               {"llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32",
                k_f8f6f4_args.args[i],
                6,
                f8f6f4_operand_form(pair.a, reg("i32", 16)),
                f8f6f4_operand_form(pair.b, reg("i32", 16)),
                as_is(reg("f32", 8)),
                as_is(reg("f32", 8)),
                NULL}};
    }
    result.entries[k_listed_count + k_f8f6f4_count]
        = {"wmma_gfx1250_f32_32x16x128_f4",
           {"llvm.amdgcn.wmma.f32.32x16x128.f4.v16f32.v16i32.v8i32",
            k_gfx1250_f4_args,
            (int)(sizeof(k_gfx1250_f4_args) / sizeof(k_gfx1250_f4_args[0])),
            f8f6f4_operand_form(k_fp4_format, reg("i32", 16)),
            f8f6f4_operand_form(k_fp4_format, reg("i32", 8)),
            as_is(reg("f32", 16)),
            as_is(reg("f32", 16)),
            NULL}};
    for(int shape = 0; shape < 2; ++shape)
    {
        const bool m16 = shape == 0;
        const rocke_matrix_register_t acc = reg("f32", m16 ? 4 : 16);
        for(int i = 0; i < k_f8f6f4_count; ++i)
        {
            const rocke_matrix_format_pair& pair = k_rocke_narrow_matrix_format_pairs.pairs[i];
            result.entries[k_listed_count + k_f8f6f4_count + 1 + shape * k_f8f6f4_count + i]
                = {m16 ? k_gfx950_f8f6f4_16x16_ids.ids[i] : k_gfx950_f8f6f4_32x32_ids.ids[i],
                   {m16 ? "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4"
                        : "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32",
                    k_gfx950_f8f6f4_args.args[i],
                    9,
                    f8f6f4_operand_form(pair.a, reg("i32", 8)),
                    f8f6f4_operand_form(pair.b, reg("i32", 8)),
                    as_is(acc),
                    as_is(acc),
                    m16 ? "mfma.f32.16x16x128.fp8.hero"
                        : "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32"}};
        }
    }
    return result;
}
/* op_id -> call, for every MMA instruction lowered from data (Python
 * MATRIX_CALLS, same order): the listed entries, then the generated ones. */
constexpr auto k_matrix_calls = make_matrix_call_table();
} // namespace

const rocke_matrix_call_entry_t* rocke_matrix_calls(int* out_count)
{
    if(out_count)
        *out_count = k_call_count;
    return k_matrix_calls.entries;
}

const rocke_matrix_call_t* rocke_matrix_call_for(const char* op_id)
{
    if(!op_id)
        return NULL;
    for(const rocke_matrix_call_entry_t& entry : k_matrix_calls.entries)
    {
        if(strcmp(entry.op_id, op_id) == 0)
            return &entry.call;
    }
    return NULL;
}

const char* rocke_matrix_call_key(const rocke_matrix_call_t* call)
{
    return call->decl_key ? call->decl_key : call->intrinsic;
}

/* ------------------------------------------------------------ register types */

static const rocke_type_t* scalar_type(const char* name)
{
    const rocke_type_t* t = rocke_scalar_by_name(name);
    if(!t)
        throw ValueError(format_error("unknown IR scalar type '%s'", name ? name : "(null)"));
    return t;
}

/* The flavors on the ladder as Python prints the tuple LLVM_FLAVORS, for the
 * errors that list the valid choices: ('llvm20', 'llvm22', 'llvm23'). */
static std::string flavor_choices()
{
    std::string text = "(";
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        text += i ? ", '" : "'";
        text += rocke_llvm_flavor_at(i);
        text += "'";
    }
    return text + ")";
}

/* The type a register takes at `flavor`: its newest override at or below the
 * flavor on the ladder, else its base (Python FlavorType.for_flavor). */
static rocke_matrix_type_t register_at(const rocke_matrix_register_t* reg,
                                       rocke_llvm_flavor_t flavor)
{
    int rung = -1;
    for(int i = 0; i < rocke_llvm_flavor_count(); ++i)
    {
        if(rocke_llvm_flavor_from_name(rocke_llvm_flavor_at(i)) == flavor)
            rung = i;
    }
    if(rung < 0)
        throw ValueError(format_error("unknown LLVM flavor '%s'; expected one of %s",
                                      rocke_llvm_flavor_name(flavor),
                                      flavor_choices().c_str()));
    rocke_matrix_type_t resolved = reg->base;
    for(int i = 0; i <= rung; ++i)
    {
        for(int j = 0; j < reg->num_overrides; ++j)
        {
            if(strcmp(reg->overrides[j].flavor, rocke_llvm_flavor_at(i)) == 0)
                resolved = reg->overrides[j].type;
        }
    }
    return resolved;
}

const rocke_type_t* rocke_matrix_form_type_at(rocke_arena_t* arena,
                                              const rocke_matrix_form_t* form,
                                              rocke_llvm_flavor_t flavor)
{
    const rocke_matrix_type_t t = register_at(&form->register_type, flavor);
    const rocke_type_t* elem = scalar_type(t.elem);
    if(t.count == 0)
        return elem;
    const rocke_type_t* vec = rocke_i_vector_type_in_arena(arena, elem, t.count);
    if(!vec)
        throw std::bad_alloc();
    return vec;
}

/* The IR name of a register type ("vec<f16x8>", "i32"). */
static std::string type_name(const rocke_matrix_type_t& t)
{
    if(t.count == 0)
        return t.elem;
    return format_error("vec<%sx%d>", t.elem, t.count);
}

/* The LLVM text of a register type (Python _llvm_type). */
static std::string llvm_text(const rocke_matrix_type_t& t)
{
    const char* elem = rocke_ll_llvm_type(NULL, scalar_type(t.elem));
    if(t.count == 0)
        return elem;
    return format_error("<%d x %s>", t.count, elem);
}

/* ---------------------------------------------------------------- the schema */

static const char* const k_immediate_types[] = {"i1", "i16", "i32", "i64"};

static void check_immediate(const rocke_matrix_arg_t& arg)
{
    int bits = 0;
    for(const char* name : k_immediate_types)
    {
        if(arg.imm_type && strcmp(arg.imm_type, name) == 0)
            bits = rocke_i_type_bits(scalar_type(name));
    }
    if(bits == 0)
        throw TypeError(format_error("an immediate is one of i1, i16, i32, i64, got %s",
                                     arg.imm_type ? arg.imm_type : "(null)"));
    const long long low = -(1LL << (bits - 1));
    const bool fits
        = arg.imm_value >= low && (bits == 64 || arg.imm_value < (long long)(1ULL << bits));
    if(!fits)
        throw ValueError(
            format_error("%lld does not fit an %s immediate", arg.imm_value, arg.imm_type));
}

/* Every type a register takes at some flavor (Python FlavorType.every_flavor). */
template <class Fn>
static void for_every_type(const rocke_matrix_register_t& reg, Fn fn)
{
    fn(reg.base);
    for(int i = 0; i < reg.num_overrides; ++i)
        fn(reg.overrides[i].type);
}

static void check_form(const rocke_matrix_form_t& form)
{
    for(int i = 0; i < form.register_type.num_overrides; ++i)
    {
        const char* flavor = form.register_type.overrides[i].flavor;
        if(!rocke_llvm_flavor_is_known(rocke_llvm_flavor_from_name(flavor)))
            throw ValueError(format_error("FlavorType names unknown LLVM flavor '%s'; expected "
                                          "one of %s",
                                          flavor ? flavor : "(null)",
                                          flavor_choices().c_str()));
    }
    if(form.kind == ROCKE_MATRIX_FORM_ZERO_PADDED)
    {
        for_every_type(form.register_type, [](const rocke_matrix_type_t& t) {
            if(t.count == 0 || strcmp(t.elem, "i32") != 0)
                throw TypeError(
                    format_error("ZeroPadded register type must be a vector of i32 words, got %s",
                                 type_name(t).c_str()));
        });
    }
    if(form.kind == ROCKE_MATRIX_FORM_EVEN_HALVES)
    {
        for_every_type(form.register_type, [](const rocke_matrix_type_t& t) {
            if(t.count == 0 || t.count % 2 != 0 || rocke_i_type_bits(scalar_type(t.elem)) != 16)
                throw TypeError(format_error("EvenHalves register type must be a vector of an even "
                                             "number of 16-bit elements, got %s",
                                             type_name(t).c_str()));
        });
    }
}

void rocke_matrix_call_check(const rocke_matrix_call_t* call)
{
    static const char* const k_operand_names[] = {"A", "B", "C"};
    int seen[3] = {0, 0, 0};
    std::string got;
    for(int i = 0; i < call->num_args; ++i)
    {
        const rocke_matrix_arg_t& arg = call->args[i];
        if(arg.kind == ROCKE_MATRIX_ARG_IMM)
        {
            check_immediate(arg);
            continue;
        }
        if(arg.kind == ROCKE_MATRIX_ARG_SIGNEDNESS)
        {
            if(arg.operand != ROCKE_MATRIX_ARG_A && arg.operand != ROCKE_MATRIX_ARG_B)
                throw ValueError(
                    format_error("a signedness flag precedes the A or B operand, got %s",
                                 arg.operand == ROCKE_MATRIX_ARG_C ? "C" : "(not an operand)"));
            continue;
        }
        seen[arg.kind] += 1;
        got += got.empty() ? "'" : ", '";
        got += k_operand_names[arg.kind];
        got += "'";
    }
    if(seen[0] != 1 || seen[1] != 1 || seen[2] != 1)
        throw ValueError(
            format_error("MatrixCall '%s' must pass each of A, B and C exactly once, got [%s]",
                         call->intrinsic,
                         got.c_str()));
    if(call->result.kind == ROCKE_MATRIX_FORM_ZERO_PADDED)
        throw TypeError(
            format_error("MatrixCall '%s' result form must be AsIs or EvenHalves, got ZeroPadded",
                         call->intrinsic));
    for(const rocke_matrix_form_t* form : {&call->a, &call->b, &call->c, &call->result})
        check_form(*form);
}

/* ------------------------------------------------------------ declarations */

static const rocke_matrix_form_t& operand_form(const rocke_matrix_call_t* call,
                                               rocke_matrix_arg_kind_t kind)
{
    return kind == ROCKE_MATRIX_ARG_A ? call->a : kind == ROCKE_MATRIX_ARG_B ? call->b : call->c;
}

const char* rocke_matrix_call_declaration(const rocke_matrix_call_t* call,
                                          rocke_llvm_flavor_t flavor)
{
    /* Interned by content: a call's text never changes, and a set node never
     * moves, so the returned pointer stays valid for the process. */
    static std::mutex lock;
    static std::set<std::string> interned;

    std::string text = "declare " + llvm_text(register_at(&call->result.register_type, flavor))
                       + " @" + call->intrinsic + "(";
    for(int i = 0; i < call->num_args; ++i)
    {
        const rocke_matrix_arg_t& arg = call->args[i];
        if(i)
            text += ", ";
        if(arg.kind == ROCKE_MATRIX_ARG_IMM || arg.kind == ROCKE_MATRIX_ARG_SIGNEDNESS)
        {
            text += arg.kind == ROCKE_MATRIX_ARG_IMM ? arg.imm_type : "i1";
            if(arg.immarg)
                text += " immarg";
            continue;
        }
        text += llvm_text(register_at(&operand_form(call, arg.kind).register_type, flavor));
    }
    text += ")";

    std::lock_guard<std::mutex> guard(lock);
    return interned.insert(std::move(text)).first->c_str();
}

} /* namespace ckc */
