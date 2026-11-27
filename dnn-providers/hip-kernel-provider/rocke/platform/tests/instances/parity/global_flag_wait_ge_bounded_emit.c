// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of global_flag_wait_ge_bounded_emit.py (config table
 * must match). */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    const char* atomic_scope; /* NULL = unscoped */
    const char* wait_scope;
    const char* wait_ordering;
} flag_case_t;

static const flag_case_t CASES[] = {
    {NULL, "agent", "acquire"},
    {"workgroup", "workgroup", "monotonic"},
    {"agent", "system", "acquire"},
    {"system", "agent", "monotonic"},
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

static void build_flag_wait(rocke_ir_builder_t* b, const flag_case_t* c)
{
    rocke_value_t* flag = rocke_b_param(b, "Flag", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* acc = rocke_b_param(b, "Acc", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* idx = rocke_b_param(b, "Idx", rocke_i32(), NULL);
    rocke_value_t* expect = rocke_b_param(b, "Expect", rocke_i32(), NULL);
    rocke_value_t* polls = rocke_b_param(b, "Polls", rocke_i32(), NULL);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    (void)rocke_b_global_atomic_add_scoped(b, flag, idx, one, "release", c->atomic_scope);
    rocke_value_t* f1 = rocke_b_const_f32(b, 1.0);
    (void)rocke_b_global_atomic_add_scoped(b, acc, idx, f1, NULL, c->atomic_scope);
    rocke_value_t* seen = rocke_b_global_flag_wait_ge_bounded(
        b, flag, idx, expect, polls, c->wait_scope, c->wait_ordering);
    rocke_value_t* at = rocke_b_add(b, idx, one);
    rocke_b_global_store(b, flag, at, seen, 1);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_CASES - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_CASES)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "flag_wait_ge_bounded") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    build_flag_wait(&b, &CASES[idx]);
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
