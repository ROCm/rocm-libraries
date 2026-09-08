/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/target_intrinsics_emit.c -- C-side emitter for the LLVM 23 /
 * future-operator intrinsic surface. Builds each kernel identically to
 * target_intrinsics_emit.py so run_diff.py can byte-compare the two engines'
 * .ll; see that file for why substring smoke tests were not enough.
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

/* CACHE_STREAM (gfx12 cachepolicy: SLC / one-shot streaming). */
#define ROCKE_PARITY_CACHE_STREAM 2

static rocke_value_t*
    global_ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, int align)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.align = align;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* Every builder call below is its own statement, and the statement order matches
 * the Python emitter exactly. Nesting a builder call inside an argument list
 * would leave the order to the C compiler (argument evaluation order is
 * unspecified), and since each call consumes a value id, a different order emits
 * the same IR under different SSA names -- a byte mismatch with no real defect
 * behind it. */

/* Async global->LDS copy into each of two LDS allocations, so the second
 * destination sits at a non-zero offset in the unified smem pool. Both are
 * copied into because a dead allocation is dropped from the pool, which would
 * put stageB back at offset 0 and skip the base-pointer hop entirely. */
static void build_async_lds_two_allocs(rocke_ir_builder_t* b)
{
    rocke_value_t* src = global_ptr_param(b, "src", rocke_i32(), 16);
    const int shape_a[] = {64};
    const int shape_b[] = {64, 4};
    rocke_value_t* stage_a;
    rocke_value_t* stage_b;
    rocke_value_t* tid;
    rocke_value_t* zero;
    rocke_value_t* idx_a[1];
    rocke_value_t* idx_b[2];
    stage_a = rocke_b_smem_alloc(b, rocke_i32(), shape_a, 1, "stageA");
    stage_b = rocke_b_smem_alloc(b, rocke_i32(), shape_b, 2, "stageB");
    tid = rocke_b_thread_id_x(b);
    zero = rocke_b_const_i32(b, 0);
    idx_a[0] = tid;
    idx_b[0] = tid;
    idx_b[1] = zero;
    rocke_b_global_load_async_to_lds(b,
                                     src,
                                     tid,
                                     stage_a,
                                     idx_a,
                                     /*num_lds_indices=*/1,
                                     /*width_bytes=*/16,
                                     /*coherency=*/ROCKE_PARITY_CACHE_STREAM,
                                     /*offset_bytes=*/0);
    rocke_b_global_load_async_to_lds(b,
                                     src,
                                     tid,
                                     stage_b,
                                     idx_b,
                                     /*num_lds_indices=*/2,
                                     /*width_bytes=*/16,
                                     /*coherency=*/ROCKE_PARITY_CACHE_STREAM,
                                     /*offset_bytes=*/0);
    rocke_b_s_wait_asynccnt(b, 0);
    rocke_b_ret(b);
}

static void build_async_lds_b8(rocke_ir_builder_t* b)
{
    rocke_value_t* src = global_ptr_param(b, "src", rocke_i32(), 4);
    const int shape[] = {64};
    rocke_value_t* lds = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "stage");
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* idx[1];
    idx[0] = zero;
    rocke_b_global_load_async_to_lds(b,
                                     src,
                                     zero,
                                     lds,
                                     idx,
                                     /*num_lds_indices=*/1,
                                     /*width_bytes=*/1,
                                     /*coherency=*/0,
                                     /*offset_bytes=*/0);
    rocke_b_ret(b);
}

/* s_prefetch_inst on a non-flat pointer: the operand is llvm_anyptr_ty, so the
 * address space is part of the overload. */
static void build_prefetch_inst(rocke_ir_builder_t* b)
{
    rocke_value_t* code = global_ptr_param(b, "code", rocke_i32(), 4);
    rocke_value_t* length = rocke_b_const_i32(b, 64);
    rocke_b_s_prefetch_inst(b, code, length);
    rocke_b_ret(b);
}

static void build_buffer_load_lds_async(rocke_ir_builder_t* b)
{
    rocke_value_t* X = rocke_b_param(b, "X", rocke_ptr_type(b, rocke_f16(), "global"), NULL);
    rocke_value_t* N = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, X, N);
    const int shape[] = {64, 8};
    rocke_value_t* lds = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "stage");
    rocke_value_t* lds_addr = rocke_b_smem_addr_of(b, lds);
    rocke_value_t* voffset = rocke_b_const_i32(b, 0);
    rocke_value_t* soffset = rocke_b_const_i32(b, 0);
    rocke_b_buffer_load_lds_async(b,
                                  rsrc,
                                  lds_addr,
                                  voffset,
                                  soffset,
                                  /*dwords=*/4,
                                  /*coherency=*/ROCKE_PARITY_CACHE_STREAM);
    rocke_b_ret(b);
}

static void build_permlane(rocke_ir_builder_t* b)
{
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* old = rocke_b_const_i32(b, 0);
    rocke_value_t* src1 = rocke_b_const_i32(b, 2);
    rocke_value_t* src2 = rocke_b_const_i32(b, 3);
    rocke_b_permlane16(b, old, tid, src1, src2, false, false);
    rocke_b_permlane16(b, old, tid, src1, src2, true, true);
    rocke_b_permlane64(b, tid);
    rocke_b_ret(b);
}

static void build_av_b128(rocke_ir_builder_t* b)
{
    rocke_value_t* p = global_ptr_param(b, "p", rocke_i32(), 16);
    rocke_value_t* data = rocke_b_av_load_b128(b, p);
    rocke_b_av_store_b128(b, p, data);
    rocke_b_ret(b);
}

static void build_scheduler_hints(rocke_ir_builder_t* b)
{
    rocke_b_s_alloc_vgpr(b, 8);
    rocke_b_asyncmark(b);
    rocke_b_wait_asyncmark(b, 3);
    rocke_b_s_wait_event(b, 1);
    rocke_b_ret(b);
}

typedef void (*build_fn_t)(rocke_ir_builder_t*);

static const build_fn_t BUILDERS[] = {
    build_async_lds_two_allocs,
    build_async_lds_b8,
    build_prefetch_inst,
    build_buffer_load_lds_async,
    build_permlane,
    build_av_b128,
    build_scheduler_hints,
};

static const int NUM_BUILDERS = (int)(sizeof(BUILDERS) / sizeof(BUILDERS[0]));

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_BUILDERS - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_BUILDERS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "target_intrinsics") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    BUILDERS[idx](&b);

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
