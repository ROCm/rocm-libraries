// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of optimization_barrier_emit.py. */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void build_optimization_barriers(rocke_ir_builder_t* b)
{
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    const rocke_type_t* types[] = {rocke_i1(),
                                   rocke_i8(),
                                   rocke_i16(),
                                   rocke_i32(),
                                   rocke_i64(),
                                   rocke_bf16(),
                                   rocke_f16(),
                                   rocke_f32(),
                                   rocke_fp8e4m3(),
                                   rocke_bf8e5m2()};
    for(const rocke_type_t* type : types)
    {
        char name[32];
        snprintf(name, sizeof(name), "p_%s", type->name);
        rocke_value_t* ptr = rocke_b_param(b, name, rocke_ptr_type(b, type, "global"), NULL);
        rocke_value_t* value = rocke_b_global_load(b, ptr, tid, type, 1);
        value = rocke_b_optimization_barrier(b, value);
        rocke_b_global_store(b, ptr, tid, value, 1);
    }
    rocke_b_ret(b);
}

static const char* ARCHES[]
    = {"gfx1250", "gfx950", "gfx90a", "gfx942", "gfx1151", "gfx1201", "gfx11-generic"};
static const int NUM_CONFIGS = sizeof(ARCHES) / sizeof(ARCHES[0]);

static void build_hip_epilogue(rocke_ir_builder_t* b, int idx)
{
    const rocke_type_t* dtype = (idx / 2) % 2 ? rocke_bf16() : rocke_f16();
    auto* output = rocke_b_param(b, "output", rocke_ptr_type(b, dtype, "global"), NULL);
    auto* acc = rocke_b_param(b, "acc", rocke_f32(), NULL);
    auto* factor_a = rocke_b_param(b, "factor_a", rocke_f32(), NULL);
    auto* factor_b = rocke_b_param(b, "factor_b", rocke_f32(), NULL);
    auto* factors = rocke_b_fmul(b, factor_a, factor_b);
    auto* scaled = rocke_b_fmul(b, acc, factors);
    auto* opaque = rocke_b_optimization_barrier(b, scaled);
    if(idx % 2)
        rocke_attr_set_str(b, &opaque->op->attrs, "clobber", "");
    auto* tid = rocke_b_thread_id_x(b);
    auto* converted = rocke_b_cast_f32_to(b, opaque, dtype);
    rocke_b_global_store(b, output, tid, converted, 2);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    /* Installed CTest invokes native tests without arguments. */
    int idx = argc > 1 ? atoi(argv[1]) : 0;
    const char* mode = argc > 2 ? argv[2] : (argc > 1 ? "ll" : "hip");

    const bool hip = strcmp(mode, "hip") == 0;
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0 && !hip)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= (hip ? 8 : NUM_CONFIGS))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, hip ? "optimization_barrier_hip" : "optimization_barrier")
       != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    if(hip)
        build_hip_epilogue(&b, idx);
    else
        build_optimization_barriers(&b);

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(hip)
    {
        rocke_lower_hip_opts_t opts = {};
        opts.arch = idx / 4 ? "gfx1250" : "gfx950";
        rocke_strbuf_t text = {};
        rocke_status_t st = rocke_lower_kernel_to_hip(&b, kernel, &opts, &text);
        if(st != ROCKE_OK || !text.cstr())
        {
            fprintf(stderr, "HIP lower failed: status=%d\n", (int)st);
            text.free_buffer();
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text.cstr(), stdout);
        text.free_buffer();
    }
    else if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, ARCHES[idx], &llvm_text, err, sizeof err);
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
