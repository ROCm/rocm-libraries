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

/* HIP configs: gfx950/gfx1250 x bf16/fp16 x absent/empty clobber. */
static void build_hip_epilogue(rocke_ir_builder_t* b, int idx)
{
    const auto* type = idx % 4 < 2 ? rocke_bf16() : rocke_f16();
    auto* output = rocke_b_param(b, "output", rocke_ptr_type(b, type, "global"), NULL);
    auto* acc = rocke_b_param(b, "acc", rocke_f32(), NULL);
    auto* factor_a = rocke_b_param(b, "factor_a", rocke_f32(), NULL);
    auto* factor_b = rocke_b_param(b, "factor_b", rocke_f32(), NULL);
    auto* value = rocke_b_fmul(b, acc, rocke_b_fmul(b, factor_a, factor_b));
    auto* opaque = rocke_b_optimization_barrier(b, value);
    if(idx % 2)
        rocke_attr_set_str(b, &opaque->op->attrs, "clobber", "");
    /* Sequence side-effecting builder calls in Python's evaluation order. */
    auto* tid = rocke_b_thread_id_x(b);
    auto* converted = rocke_b_cast_f32_to(b, opaque, type);
    rocke_b_global_store(b, output, tid, converted, 2);
    rocke_b_ret(b);
}

/* Appended HIP configs: gfx950/gfx1250 x void/single-output multiline asm. */
static void build_hip_assembly(rocke_ir_builder_t* b, int idx)
{
    const bool output = idx % 2 != 0;
    const char* tmpl = output ? "v_mov_b32 $0, $1\n\ts_nop 0 ; \"quoted\" \\path"
                              : "s_nop 0\n\ts_nop 0 ; \"quoted\" \\path";
    rocke_value_t* pointer
        = output ? rocke_b_param(b, "output", rocke_ptr_type(b, rocke_f32(), "global"), NULL)
                 : NULL;
    rocke_value_t* value = output ? rocke_b_param(b, "value", rocke_f32(), NULL) : NULL;
    rocke_value_t* operands[] = {value};
    const rocke_type_t* types[] = {rocke_f32()};
    auto* op = rocke_b_inline_asm(
        b, tmpl, output ? "=v,0" : "", operands, output ? 1 : 0, types, output ? 1 : 0, NULL);
    if(output && op)
        rocke_b_global_store(b, pointer, rocke_b_thread_id_x(b), op->results[0], 4);
    rocke_b_ret(b);
}

/* Appended HIP configs: void/output m0 clobbers on both targets, then gfx950 LDS DMA. */
static void build_hip_clobbers(rocke_ir_builder_t* b, int idx)
{
    if(idx == 4)
    {
        auto* voff = rocke_b_param(b, "voff", rocke_i32(), NULL);
        auto* lds = rocke_b_param(b, "lds", rocke_i32(), NULL);
        auto* base = rocke_b_param(b, "base", rocke_i64(), NULL);
        rocke_value_t* operands[] = {voff, lds, base};
        rocke_b_inline_asm(b,
                           "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dword $0, $2",
                           "v,s,s,~{m0},~{memory}",
                           operands,
                           3,
                           NULL,
                           0,
                           NULL);
    }
    else
    {
        const bool output = idx % 2 != 0;
        auto* pointer
            = output ? rocke_b_param(b, "output", rocke_ptr_type(b, rocke_f32(), "global"), NULL)
                     : NULL;
        auto* value = rocke_b_param(b, "value", output ? rocke_f32() : rocke_i32(), NULL);
        auto* lds = output ? rocke_b_param(b, "lds", rocke_i32(), NULL) : NULL;
        rocke_value_t* operands[] = {value, lds};
        const rocke_type_t* types[] = {rocke_f32()};
        auto* op = rocke_b_inline_asm(b,
                                      output ? "s_mov_b32 m0, $2\n\tv_mov_b32 $0, $1"
                                             : "s_mov_b32 m0, $0",
                                      output ? "=v,0,s,~{m0},~{memory}" : "s,~{m0},~{memory}",
                                      operands,
                                      output ? 2 : 1,
                                      types,
                                      output ? 1 : 0,
                                      NULL);
        if(output && op)
        {
            auto* tid = rocke_b_thread_id_x(b);
            rocke_b_global_store(b, pointer, tid, op->results[0], 4);
        }
    }
    rocke_b_ret(b);
}

static const char* ARCHES[]
    = {"gfx1250", "gfx950", "gfx90a", "gfx942", "gfx1151", "gfx1201", "gfx11-generic"};
static const int NUM_CONFIGS = sizeof(ARCHES) / sizeof(ARCHES[0]);

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr,
                "usage: %s <config_index 0..%d> [ll|ir|verify|hip]\n",
                argv[0],
                NUM_CONFIGS - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    const bool hip = strcmp(mode, "hip") == 0;
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0 && !hip)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= (hip ? 17 : NUM_CONFIGS))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(
           &b,
           hip ? (idx < 8 ? "tensor_epilogue" : (idx < 12 ? "asm_string" : "asm_clobbers"))
               : "optimization_barrier")
       != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    if(hip && idx < 8)
        build_hip_epilogue(&b, idx);
    else if(hip && idx < 12)
        build_hip_assembly(&b, idx - 8);
    else if(hip)
        build_hip_clobbers(&b, idx - 12);
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
        const bool gfx950 = idx < 8 ? idx < 4 : (idx < 12 ? idx < 10 : (idx < 14 || idx == 16));
        opts.arch = gfx950 ? "gfx950" : "gfx1250";
        rocke_strbuf_t text = {};
        auto status = rocke_lower_kernel_to_hip(&b, kernel, &opts, &text);
        if(status != ROCKE_OK)
        {
            fprintf(stderr, "HIP lower failed: %s\n", rocke_ir_builder_error(&b));
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
