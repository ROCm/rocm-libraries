// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of buffer_load_vN_emit.py (config table must match). */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx950", "gfx1151", "gfx1201"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);

typedef struct
{
    const char* dtype;
    int n;
} case_t;

static const case_t CASES[] = {
    {"f16", 2},
    {"f16", 4},
    {"f16", 8},
    {"bf16", 2},
    {"bf16", 4},
    {"bf16", 8},
    {"f32", 1},
    {"f32", 2},
    {"f32", 4},
    {"i32", 1},
    {"i32", 4},
    /* rejected */
    {"f16", 1},
    {"f16", 3},
    {"f16", 5},
    {"f16", 6},
    {"f16", 16},
    {"bf16", 3},
    {"f32", 3},
    {"f32", 8},
    {"f32", 0},
    {"i8", 4},
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

static const rocke_type_t* type_by_name(const char* name)
{
    if(strcmp(name, "f16") == 0)
        return rocke_f16();
    if(strcmp(name, "bf16") == 0)
        return rocke_bf16();
    if(strcmp(name, "f32") == 0)
        return rocke_f32();
    if(strcmp(name, "i32") == 0)
        return rocke_i32();
    return rocke_i8();
}

static void build_buffer_load_vN(rocke_ir_builder_t* b, const case_t* c)
{
    const rocke_type_t* dtype = type_by_name(c->dtype);
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, x, n_bytes);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* voffset = rocke_b_mul(b, tid, rocke_b_const_i32(b, 16));
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_value_t* v = rocke_b_buffer_load_vN(b, rsrc, voffset, soffset, dtype, c->n);
    if(!v || !rocke_ir_builder_ok(b))
        return; /* rejected width/dtype: the builder error is reported by main */
    for(int i = 0; i < c->n; ++i)
    {
        /* C argument evaluation order is unspecified; spell out the Python
         * left-to-right order so the SSA numbering matches. */
        rocke_value_t* stride = rocke_b_const_i32(b, c->n);
        rocke_value_t* base = rocke_b_mul(b, tid, stride);
        rocke_value_t* lane = rocke_b_const_i32(b, i);
        rocke_value_t* idx = rocke_b_add(b, base, lane);
        rocke_value_t* elem = rocke_b_vec_extract(b, v, i);
        rocke_b_global_store(b, y, idx, elem, 1);
    }
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
    const case_t* c = &CASES[idx / NUM_ARCHES];
    const char* arch = ARCHES[idx % NUM_ARCHES];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "buffer_load_vN") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    try
    {
        build_buffer_load_vN(&b, c);
    }
    catch(const std::exception& e)
    {
        /* The builder may report a rejected width/dtype by throwing. */
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
