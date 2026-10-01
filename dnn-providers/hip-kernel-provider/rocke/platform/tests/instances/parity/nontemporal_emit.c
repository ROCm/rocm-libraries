/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/nontemporal_emit.c -- C-side emitter for the `nontemporal`
 * flag of global_load_vN / global_store_vN. Builds each copy kernel
 * identically to nontemporal_emit.py so run_diff.py can byte-compare the two
 * engines' .ll; see that file for the config rationale.
 *
 * arch is per-config (see CONFIGS), flavor = AUTO (matches the Python side).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct nt_config
{
    int f32; /* 0 -> bf16, 1 -> f32 */
    int n;
    int load_nt;
    int store_nt;
    const char* arch;
} nt_config_t;

static const nt_config_t CONFIGS[] = {
    {0, 8, 1, 1, "gfx950"},
    {0, 8, 0, 0, "gfx950"},
    {0, 8, 1, 0, "gfx942"},
    {1, 4, 0, 1, "gfx942"},
};

static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

static rocke_value_t*
    copy_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    o.align = 16;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* One builder call per statement, in the Python emitter's order, so both
 * engines assign the same SSA ids. */
static void build(rocke_ir_builder_t* b, const nt_config_t* c)
{
    const rocke_type_t* elem = c->f32 ? rocke_f32() : rocke_bf16();
    rocke_value_t* src = copy_param(b, "S", elem, true);
    rocke_value_t* dst = copy_param(b, "D", elem, false);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* width = rocke_b_const_i32(b, c->n);
    rocke_value_t* off = rocke_b_mul(b, tid, width);
    rocke_value_t* v = rocke_b_global_load_vN_ex(b, src, off, elem, c->n, 0, c->load_nt);
    rocke_b_global_store_vN_ex(b, dst, off, v, c->n, 0, c->store_nt);
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
    if(rocke_ir_builder_init(&b, "nontemporal") != ROCKE_OK)
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, CONFIGS[idx].arch, &llvm_text, err, sizeof err);
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
