// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of lds_barrier_keep_vmem_emit.py (config table must
 * match). */
#include "rocke/helper_helpers.asm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[]
    = {"gfx1201", "gfx1200", "gfx12-generic", "gfx950", "gfx11-generic", "gfx0000"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);
static const int BLOCK = 256;

static void build(rocke_ir_builder_t* b, const char* target)
{
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, rocke_f16(), "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, rocke_f16(), "global"), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    int shape[1] = {BLOCK};
    rocke_value_t* tile = rocke_b_smem_alloc(b, rocke_f16(), shape, 1, "tile");
    rocke_value_t* v = rocke_b_global_load(b, x, tid, rocke_f16(), 1);
    rocke_value_t* st_idx[1] = {tid};
    rocke_b_smem_store_vN(b, tile, st_idx, 1, v, 1);
    rocke_lds_barrier_keep_vmem(b, target);
    rocke_value_t* c_last = rocke_b_const_i32(b, BLOCK - 1);
    rocke_value_t* flip = rocke_b_sub(b, c_last, tid);
    rocke_value_t* ld_idx[1] = {flip};
    rocke_value_t* u = rocke_b_smem_load_vN_f16(b, tile, ld_idx, 1, 1);
    rocke_value_t* h = rocke_b_vec_extract(b, u, 0);
    rocke_b_global_store(b, y, tid, h, 1);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_ARCHES - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_ARCHES)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const char* arch = ARCHES[idx];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "lds_barrier_keep_vmem") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = BLOCK */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", BLOCK);
    try
    {
        build(&b, arch);
    }
    catch(const std::exception& e)
    {
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
