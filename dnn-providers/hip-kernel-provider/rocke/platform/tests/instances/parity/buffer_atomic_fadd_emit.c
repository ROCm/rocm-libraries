// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of buffer_atomic_fadd_emit.py (config table must match). */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx950", "gfx1250"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);

static const int CASES[] = {
    0,
    1,
    ROCKE_CPOL_ATOMIC_CASCADE_DEV,
    31,
    /* rejected */
    32,
    -1,
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

static void build_buffer_atomic_fadd(rocke_ir_builder_t* b, int cachepolicy)
{
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, y, n_bytes);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* value = rocke_b_global_load(b, x, tid, rocke_f32(), 1);
    rocke_value_t* voffset = rocke_b_mul(b, tid, rocke_b_const_i32(b, 4));
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_b_buffer_atomic_fadd_f32(b, rsrc, voffset, soffset, value, cachepolicy);
    if(!rocke_ir_builder_ok(b))
        return; /* rejected cache policy: the builder error is reported by main */
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    int num_configs = NUM_CASES * NUM_ARCHES;
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], num_configs - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= num_configs)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    int cachepolicy = CASES[idx / NUM_ARCHES];
    const char* arch = ARCHES[idx % NUM_ARCHES];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_atomic_fadd") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    try
    {
        build_buffer_atomic_fadd(&b, cachepolicy);
    }
    catch(const std::exception& e)
    {
        /* The builder may report a rejected cache policy by throwing. */
        fprintf(stderr, "builder error: %s\n", e.what());
        rocke_ir_builder_free(&b);
        return 1;
    }

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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
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
