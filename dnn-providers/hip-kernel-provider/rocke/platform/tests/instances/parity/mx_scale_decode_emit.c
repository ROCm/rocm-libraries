/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/mx_scale_decode_emit.c -- C-side emitter for the E8M0 scale
 * decoder in isolation. Builds the kernel identically to
 * mx_scale_decode_emit.py so run_diff.py can byte-compare the two engines' .ll:
 * out[0] = decode_mx_scale_e8m0(codes[0]).
 *
 * Modes: "ll" (default), "ir", "verify" are the gate modes. "hip" prints the HIP
 * lowering of the kernel; the numeric tests build it, and the gate does not
 * compare it.
 *
 * arch = gfx950, flavor = AUTO (matches the Python side).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.mx_scale.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"
#include "rocke/verify.h"

/* Every builder call below is its own statement, and the statement order
 * matches the Python emitter exactly: each call consumes a value id, so a
 * different order emits the same IR under different SSA names. */
static void build_mx_scale_decode(rocke_ir_builder_t* b)
{
    rocke_param_opts_t opts;
    rocke_value_t* codes;
    rocke_value_t* out;
    rocke_value_t* idx;
    rocke_value_t* code;
    rocke_value_t* scale;

    /* codes = b.param("codes", PtrType(I8, "global"), readonly=True, align=1) */
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 1;
    opts.align_set = true;
    codes = rocke_b_param(b, "codes", rocke_ptr_type(b, rocke_i8(), "global"), &opts);

    /* out = b.param("out", PtrType(F32, "global"), writeonly=True, align=4) */
    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_f32(), "global"), &opts);

    /* idx = b.const_i32(0) */
    idx = rocke_b_const_i32(b, 0);
    /* code = b.global_load_i8(codes, idx) */
    code = rocke_b_global_load_i8(b, codes, idx, 0);
    /* scale = decode_mx_scale_e8m0(b, code) */
    scale = rocke_decode_mx_scale_e8m0(b, code);
    /* b.global_store(out, idx, scale, align=4) */
    rocke_b_global_store(b, out, idx, scale, 4);
    /* b.ret() */
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0> [ll|ir|verify|hip]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0
       && strcmp(mode, "hip") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "rocke_mx_scale_decode") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 1 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 1);
    build_mx_scale_decode(&b);

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
    else if(strcmp(mode, "hip") == 0)
    {
        rocke_strbuf_t out;
        rocke_strbuf_init(&out, 0);
        rocke_status_t st = rocke_lower_kernel_to_hip(&b, kernel, NULL, &out);
        if(st != ROCKE_OK)
        {
            fprintf(stderr, "hip lowering failed: status=%d\n", (int)st);
            rocke_strbuf_free(&out);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(rocke_strbuf_cstr(&out), stdout);
        rocke_strbuf_free(&out);
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
