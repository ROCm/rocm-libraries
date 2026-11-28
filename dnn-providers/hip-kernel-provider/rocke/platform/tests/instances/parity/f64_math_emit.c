/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/f64_math_emit.c -- C-side emitter for the f64 math ops. Builds
 * each kernel exactly as f64_math_emit.py does (same statement order, so the
 * same SSA ids) so run_diff.py can byte-compare the two engines' output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct
{
    const char* op;
    const char* dtype;
    const char* arch;
} f64_math_config_t;

static const f64_math_config_t CONFIGS[] = {
    {"exp", "f64", "gfx950"},
    {"exp2", "f64", "gfx950"},
    {"log", "f64", "gfx950"},
    {"log2", "f64", "gfx950"},
    {"tanh", "f64", "gfx950"},
    {"sqrt", "f64", "gfx950"},
    {"rsqrt", "f64", "gfx950"},
    {"fma", "f64", "gfx950"},
    {"fmax3", "f64", "gfx950"},
    {"fmin3", "f64", "gfx950"},
    {"vector_fma", "f64", "gfx950"},
    {"exp", "f32", "gfx950"},
    {"log", "f32", "gfx950"},
    {"exp", "f64", "gfx942"},
    {"tanh", "f64", "gfx90a"},
};
static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

static rocke_value_t*
    global_ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, int align)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.align = align;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

static rocke_value_t* apply_unary(rocke_ir_builder_t* b, const char* op, rocke_value_t* x)
{
    if(strcmp(op, "exp") == 0)
        return rocke_b_exp(b, x);
    if(strcmp(op, "exp2") == 0)
        return rocke_b_exp2(b, x);
    if(strcmp(op, "log") == 0)
        return rocke_b_log(b, x);
    if(strcmp(op, "log2") == 0)
        return rocke_b_log2(b, x);
    if(strcmp(op, "tanh") == 0)
        return rocke_b_tanh(b, x);
    if(strcmp(op, "sqrt") == 0)
        return rocke_b_sqrt(b, x);
    return rocke_b_rsqrt(b, x);
}

/* One builder call per statement: argument evaluation order is unspecified in
 * C, and every call consumes an SSA id. */
static void build(rocke_ir_builder_t* b, const f64_math_config_t* cfg)
{
    const int is_f64 = strcmp(cfg->dtype, "f64") == 0;
    const rocke_type_t* ty = is_f64 ? rocke_f64() : rocke_f32();
    const int align = is_f64 ? 8 : 4;
    rocke_value_t* x_ptr = global_ptr_param(b, "X", ty, 16);
    rocke_value_t* y_ptr = global_ptr_param(b, "Y", ty, 16);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    if(strcmp(cfg->op, "vector_fma") == 0)
    {
        rocke_value_t* two = rocke_b_const_i32(b, 2);
        rocke_value_t* idx = rocke_b_mul(b, tid, two);
        rocke_value_t* v = rocke_b_global_load_vN(b, x_ptr, idx, ty, 2, 0);
        rocke_value_t* y = rocke_b_vector_fma(b, v, v, v);
        rocke_b_global_store_vN(b, y_ptr, idx, y, 2, 0);
        rocke_b_ret(b);
        return;
    }
    rocke_value_t* x = rocke_b_global_load(b, x_ptr, tid, ty, align);
    rocke_value_t* y;
    if(strcmp(cfg->op, "fma") == 0 || strcmp(cfg->op, "fmax3") == 0
       || strcmp(cfg->op, "fmin3") == 0)
    {
        rocke_value_t* x2 = rocke_b_fmul(b, x, x);
        if(strcmp(cfg->op, "fma") == 0)
            y = rocke_b_fma(b, x, x2, x);
        else if(strcmp(cfg->op, "fmax3") == 0)
            y = rocke_b_fmax3(b, x, x2, x);
        else
            y = rocke_b_fmin3(b, x, x2, x);
    }
    else
    {
        y = apply_unary(b, cfg->op, x);
    }
    rocke_b_global_store(b, y_ptr, tid, y, align);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: f64_math_emit <config_index>\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const f64_math_config_t* cfg = &CONFIGS[idx];
    char kname[64];
    snprintf(kname, sizeof kname, "math_%s_%s", cfg->op, cfg->dtype);

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, kname) != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    build(&b, cfg);
    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, cfg->arch, &llvm_text, err, sizeof err);
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
