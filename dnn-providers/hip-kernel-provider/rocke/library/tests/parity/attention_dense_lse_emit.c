/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/attention_dense_lse_emit.c -- C-side emitter for the dense-
 * attention LSE helpers (rocke/instance_attention_dense_lse_internal.h). Builds
 * each kernel identically to attention_dense_lse_emit.py so run_diff.py can
 * byte-compare the two engines' .ll; see that file for the per-config rationale.
 *
 * Every builder call is its own statement, in the Python emitter's order: C
 * argument evaluation order is unspecified and each call consumes an SSA id.
 *
 * arch = gfx950, flavor = AUTO (matches the Python side).
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_attention_dense_lse_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct lse_cfg
{
    bool causal;
    int diagonal;
    int window;
    int right_bound;
    const char* dtype;
    int width;
} lse_cfg_t;

/* Index for index with _CONFIGS in attention_dense_lse_emit.py. */
static const lse_cfg_t CONFIGS[] = {
    {true, 0, 0, 0, "bf16", 4},
    {false, 0, 0, 0, "fp16", 4},
    {true, 64, 0, 0, "bf16", 2},
    {true, 0, 128, 0, "bf16", 1},
    {true, 0, 128, 64, "fp16", 4},
    {false, 0, 128, 0, "bf16", 2},
    {true, 192, 0, 0, "fp16", 4},
    {true, 0, 0, 32, "bf16", 1},
};

static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

static void build(rocke_ir_builder_t* b, const lse_cfg_t* c)
{
    const rocke_type_t* dtype = strcmp(c->dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
    rocke_value_t* lse = rocke_b_param(b, "lse", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* o = rocke_b_param(b, "o", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* kv_len = rocke_b_param(b, "kv_len", rocke_i32(), NULL);
    rocke_value_t* q_base = rocke_b_param(b, "q_base", rocke_i32(), NULL);
    rocke_value_t* m = rocke_b_param(b, "m", rocke_f32(), NULL);
    rocke_value_t* l = rocke_b_param(b, "l", rocke_f32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* query_row = rocke_b_add(b, q_base, tid);
    rocke_value_t* has_key = rocke_attention_dense_row_has_key(
        b, query_row, kv_len, c->causal, c->diagonal, c->window, c->right_bound);
    rocke_attention_dense_lse_store(b, lse, tid, m, l, has_key);
    rocke_value_t* val = rocke_b_cast_f32_to(b, m, dtype);
    rocke_value_t* vals[4] = {val, val, val, val};
    rocke_value_t* packed = rocke_b_vec_pack(b, vals, c->width, dtype);
    packed = rocke_attention_dense_zero_keyless_row(b, packed, has_key, dtype, c->width);
    rocke_value_t* stride = rocke_b_const_i32(b, c->width);
    rocke_value_t* off = rocke_b_mul(b, tid, stride);
    rocke_b_global_store_vN(b, o, off, packed, c->width, 2 * c->width);
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
    if(rocke_ir_builder_init(&b, "attention_dense_lse") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
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
    return 0;
}
