/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/xf32_bf16x3_mma_emit.c -- C-side emitter for the emulated TF32
 * MMA atoms. Builds each kernel identically to xf32_bf16x3_mma_emit.py so
 * run_diff.py can byte-compare the two engines' BF16x3 expansion with nothing
 * but the archive; see that file for why the family exists.
 *
 * arch = gfx950, flavor = AUTO (matches the Python side).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Mirrors CONFIGS in the Python emitter: the logical op_id and the physical
 * bf16 atom's A/B and accumulator fragment lengths, which it borrows. */
typedef struct xf32_config
{
    const char* op_id;
    int ab_len;
    int c_len;
} xf32_config_t;

static const xf32_config_t CONFIGS[] = {
    {"mfma_f32_16x16x32_xf32", 8, 4},
    {"mfma_f32_32x32x16_xf32", 8, 16},
};

static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

/* Every builder call is its own statement, in the Python emitter's order: each
 * call consumes a value id, so leaving the order to argument evaluation would
 * emit the same IR under different SSA names. */
static void build(rocke_ir_builder_t* b, const xf32_config_t* cfg)
{
    const rocke_type_t* f32_global = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* a_ptr = rocke_b_param(b, "a", f32_global, NULL);
    rocke_value_t* b_ptr = rocke_b_param(b, "b", f32_global, NULL);
    rocke_value_t* out_ptr = rocke_b_param(b, "out", f32_global, NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* ab_stride = rocke_b_const_i32(b, cfg->ab_len);
    rocke_value_t* ab_base = rocke_b_mul(b, tid, ab_stride);
    rocke_value_t* a = rocke_b_global_load_vN(b, a_ptr, ab_base, rocke_f32(), cfg->ab_len, 0);
    rocke_value_t* bv = rocke_b_global_load_vN(b, b_ptr, ab_base, rocke_f32(), cfg->ab_len, 0);
    const rocke_type_t* tf32_vec = rocke_vector_type(b, rocke_tf32(), cfg->ab_len);
    rocke_value_t* a_tf32 = rocke_b_bitcast(b, a, tf32_vec);
    rocke_value_t* bv_tf32 = rocke_b_bitcast(b, bv, tf32_vec);
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* c = rocke_b_vector_splat(b, zero, cfg->c_len);
    rocke_value_t* d = rocke_b_mma(b, cfg->op_id, a_tf32, bv_tf32, c, NULL, 0);
    rocke_value_t* c_stride = rocke_b_const_i32(b, cfg->c_len);
    rocke_value_t* c_base = rocke_b_mul(b, tid, c_stride);
    for(int slot = 0; slot < cfg->c_len; ++slot)
    {
        rocke_value_t* offset = rocke_b_const_i32(b, slot);
        rocke_value_t* idx = rocke_b_add(b, c_base, offset);
        rocke_value_t* lane = rocke_b_vec_extract(b, d, slot);
        rocke_b_global_store(b, out_ptr, idx, lane, 1);
    }
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_CONFIGS - 1);
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

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "xf32_bf16x3_mma") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    build(&b, &CONFIGS[idx]);

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
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
    return 0;
}
