/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gemm_prologue_unary_emit.c -- C-side emitter for the operand
 * operators of the fused GEMM operand prologue. Builds the specs of
 * gemm_prologue_unary_emit.py (same order, same values) through the C API
 * (rocke_gemm_cf32_body, rocke_gemm_prologue_t) so run_diff.py can
 * byte-compare the two engines. The cases of make_spec() are generated:
 * `python gemm_prologue_unary_emit.py --c-cases`.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_cf32.h"
#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define ARCH "gfx950"

/* One config's base: cf32_body(body) and the data types. */
static void
    base(rocke_gemm_universal_spec_t* spec, const char* name, const char* body, const char* dtype)
{
    *spec = rocke_gemm_universal_spec_default();
    spec->name = name;
    if(!rocke_gemm_cf32_body(body, ARCH, spec))
    {
        fprintf(stderr, "unknown body %s\n", body);
        exit(2);
    }
    spec->data.dtype_a = dtype;
    spec->data.dtype_b = dtype;
    spec->data.dtype_c = dtype;
}

/* 1: config idx built into spec + prologue; 0: past the end. */
static int
    make_spec(int idx, rocke_gemm_universal_spec_t* spec, rocke_gemm_prologue_t* pro, bool* has_pro)
{
    *pro = rocke_gemm_prologue_default();
    *has_pro = true;
    switch(idx)
    {
    case 0:
        base(spec, "x6a_relu", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "relu";
        return 1;
    case 1:
        base(spec, "x6b_relu", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "relu";
        return 1;
    case 2:
        base(spec, "cvt_relu", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "relu";
        pro->unary_b = "relu";
        return 1;
    case 3:
        base(spec, "x6a_abs", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "abs";
        return 1;
    case 4:
        base(spec, "x6b_abs", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "abs";
        return 1;
    case 5:
        base(spec, "cvt_abs", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "abs";
        pro->unary_b = "abs";
        return 1;
    case 6:
        base(spec, "x6a_neg", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "neg";
        return 1;
    case 7:
        base(spec, "x6b_neg", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "neg";
        return 1;
    case 8:
        base(spec, "cvt_neg", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "neg";
        pro->unary_b = "neg";
        return 1;
    case 9:
        base(spec, "x6a_exp", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "exp";
        return 1;
    case 10:
        base(spec, "x6b_exp", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "exp";
        return 1;
    case 11:
        base(spec, "cvt_exp", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "exp";
        pro->unary_b = "exp";
        return 1;
    case 12:
        base(spec, "x6a_sq", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sq";
        return 1;
    case 13:
        base(spec, "x6b_sq", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "sq";
        return 1;
    case 14:
        base(spec, "cvt_sq", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "sq";
        pro->unary_b = "sq";
        return 1;
    case 15:
        base(spec, "x6a_sigmoid", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sigmoid";
        return 1;
    case 16:
        base(spec, "x6b_sigmoid", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "sigmoid";
        return 1;
    case 17:
        base(spec, "cvt_sigmoid", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "sigmoid";
        pro->unary_b = "sigmoid";
        return 1;
    case 18:
        base(spec, "x6a_sqrt", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sqrt";
        return 1;
    case 19:
        base(spec, "x6b_sqrt", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "sqrt";
        return 1;
    case 20:
        base(spec, "cvt_sqrt", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "sqrt";
        pro->unary_b = "sqrt";
        return 1;
    case 21:
        base(spec, "x6a_rcp", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "rcp";
        return 1;
    case 22:
        base(spec, "x6b_rcp", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "rcp";
        return 1;
    case 23:
        base(spec, "cvt_rcp", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "rcp";
        pro->unary_b = "rcp";
        return 1;
    case 24:
        base(spec, "x6a_tanh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "tanh";
        return 1;
    case 25:
        base(spec, "x6b_tanh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "tanh";
        return 1;
    case 26:
        base(spec, "cvt_tanh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "tanh";
        pro->unary_b = "tanh";
        return 1;
    case 27:
        base(spec, "x6a_log", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "log";
        return 1;
    case 28:
        base(spec, "x6b_log", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "log";
        return 1;
    case 29:
        base(spec, "cvt_log", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "log";
        pro->unary_b = "log";
        return 1;
    case 30:
        base(spec, "x6a_sin", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sin";
        return 1;
    case 31:
        base(spec, "x6b_sin", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "sin";
        return 1;
    case 32:
        base(spec, "cvt_sin", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "sin";
        pro->unary_b = "sin";
        return 1;
    case 33:
        base(spec, "x6a_cos", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "cos";
        return 1;
    case 34:
        base(spec, "x6b_cos", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "cos";
        return 1;
    case 35:
        base(spec, "cvt_cos", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "cos";
        pro->unary_b = "cos";
        return 1;
    case 36:
        base(spec, "x6a_tan", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "tan";
        return 1;
    case 37:
        base(spec, "x6b_tan", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "tan";
        return 1;
    case 38:
        base(spec, "cvt_tan", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "tan";
        pro->unary_b = "tan";
        return 1;
    case 39:
        base(spec, "x6a_sinh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sinh";
        return 1;
    case 40:
        base(spec, "x6b_sinh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "sinh";
        return 1;
    case 41:
        base(spec, "cvt_sinh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "sinh";
        pro->unary_b = "sinh";
        return 1;
    case 42:
        base(spec, "x6a_cosh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "cosh";
        return 1;
    case 43:
        base(spec, "x6b_cosh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "cosh";
        return 1;
    case 44:
        base(spec, "cvt_cosh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "cosh";
        pro->unary_b = "cosh";
        return 1;
    case 45:
        base(spec, "x6a_asin", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "asin";
        return 1;
    case 46:
        base(spec, "x6b_asin", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "asin";
        return 1;
    case 47:
        base(spec, "cvt_asin", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "asin";
        pro->unary_b = "asin";
        return 1;
    case 48:
        base(spec, "x6a_acos", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "acos";
        return 1;
    case 49:
        base(spec, "x6b_acos", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "acos";
        return 1;
    case 50:
        base(spec, "cvt_acos", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "acos";
        pro->unary_b = "acos";
        return 1;
    case 51:
        base(spec, "x6a_atan", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "atan";
        return 1;
    case 52:
        base(spec, "x6b_atan", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "atan";
        return 1;
    case 53:
        base(spec, "cvt_atan", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "atan";
        pro->unary_b = "atan";
        return 1;
    case 54:
        base(spec, "x6a_asinh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "asinh";
        return 1;
    case 55:
        base(spec, "x6b_asinh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "asinh";
        return 1;
    case 56:
        base(spec, "cvt_asinh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "asinh";
        pro->unary_b = "asinh";
        return 1;
    case 57:
        base(spec, "x6a_acosh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "acosh";
        return 1;
    case 58:
        base(spec, "x6b_acosh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "acosh";
        return 1;
    case 59:
        base(spec, "cvt_acosh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "acosh";
        pro->unary_b = "acosh";
        return 1;
    case 60:
        base(spec, "x6a_atanh", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "atanh";
        return 1;
    case 61:
        base(spec, "x6b_atanh", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "atanh";
        return 1;
    case 62:
        base(spec, "cvt_atanh", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "atanh";
        pro->unary_b = "atanh";
        return 1;
    case 63:
        base(spec, "x6a_ceil", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "ceil";
        return 1;
    case 64:
        base(spec, "x6b_ceil", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "ceil";
        return 1;
    case 65:
        base(spec, "cvt_ceil", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "ceil";
        pro->unary_b = "ceil";
        return 1;
    case 66:
        base(spec, "x6a_floor", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "floor";
        return 1;
    case 67:
        base(spec, "x6b_floor", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "floor";
        return 1;
    case 68:
        base(spec, "cvt_floor", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "floor";
        pro->unary_b = "floor";
        return 1;
    case 69:
        base(spec, "x6a_exp_acc", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "exp";
        pro->fast_exp = false;
        return 1;
    case 70:
        base(spec, "cvt_exp_acc", "C3", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "exp";
        pro->fast_exp = false;
        return 1;
    case 71:
        base(spec, "x6a_sigmoid_acc", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_a = "sigmoid";
        pro->fast_exp = false;
        return 1;
    case 72:
        base(spec, "cvt_sigmoid_acc", "C3", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "sigmoid";
        pro->fast_exp = false;
        return 1;
    case 73:
        base(spec, "bits_abs_fp16", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "abs";
        pro->unary_impl = "bits";
        return 1;
    case 74:
        base(spec, "bits_use_abs_fp16", "RIW", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "abs";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 75:
        base(spec, "bits_abs_bf16", "C3", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "abs";
        pro->unary_impl = "bits";
        return 1;
    case 76:
        base(spec, "bits_use_abs_bf16", "RIW", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "abs";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 77:
        base(spec, "bits_neg_fp16", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "neg";
        pro->unary_impl = "bits";
        return 1;
    case 78:
        base(spec, "bits_use_neg_fp16", "RIW", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "neg";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 79:
        base(spec, "bits_neg_bf16", "C3", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "neg";
        pro->unary_impl = "bits";
        return 1;
    case 80:
        base(spec, "bits_use_neg_bf16", "RIW", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "neg";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 81:
        base(spec, "bits_relu_fp16", "C3", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "relu";
        pro->unary_impl = "bits";
        return 1;
    case 82:
        base(spec, "bits_use_relu_fp16", "RIW", "fp16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "relu";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 83:
        base(spec, "bits_relu_bf16", "C3", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_a = "relu";
        pro->unary_impl = "bits";
        return 1;
    case 84:
        base(spec, "bits_use_relu_bf16", "RIW", "bf16");
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        pro->unary_b = "relu";
        pro->unary_impl = "bits";
        pro->split_at = "use";
        return 1;
    case 85:
        base(spec, "cf_neg_a", "RIW", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->unary_a = "neg";
        return 1;
    case 86:
        base(spec, "cf_sqrt_b_q3sk", "Q3_SK", "bf16");
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->unary_b = "sqrt";
        return 1;
    case 87:
        base(spec, "x9_tanh_b", "C3", "bf16");
        pro->src = "f32";
        pro->recipe = "x9";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->unary_b = "tanh";
        return 1;
    default:
        return 0;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: gemm_prologue_unary_emit <config_index>\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    rocke_gemm_universal_spec_t spec;
    rocke_gemm_prologue_t pro;
    bool has_pro = false;
    if(!make_spec(idx, &spec, &pro, &has_pro))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    char reason[512];
    if(!rocke_gemm_universal_is_valid_spec(&spec, ARCH, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }
    if(has_pro)
    {
        if(!rocke_gemm_prologue_check(&pro, reason, sizeof reason))
        {
            fprintf(stderr, "prologue: %s\n", reason);
            return 1;
        }
        spec._prologue = &pro;
    }
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, ARCH);
    if(kernel == NULL)
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, ARCH, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rc = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
