/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/emit.c -- C-side emitter for the rocke parity harness.
 *
 * Builds one of four kernels (selected by argv[1]) identically to the Python
 * emitter in emit.py and prints the lowered AMDGPU LLVM .ll to stdout, so the
 * two outputs can be byte-compared.
 *
 *   scalar  : c = const(1); r = c + c
 *   memory  : params -> tid -> 2x global_load_f32 -> fadd -> global_store
 *   forloop : scf.for accumulating loop (exercises scf.for port)
 *   vector  : vector splat + vector binop + vector fptrunc/extract
 *
 * arch = gfx950, flavor = AUTO (matches Python defaults in this harness).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

static int build_scalar(rocke_ir_builder_t* b)
{
    rocke_value_t* c = rocke_b_const_i32(b, 1);
    rocke_value_t* r = rocke_b_add(b, c, c);
    (void)r;
    rocke_b_ret(b);
    return 0;
}

static int build_memory(rocke_ir_builder_t* b)
{
    const rocke_type_t* pf32 = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* A = rocke_b_param(b, "A", pf32, NULL);
    rocke_value_t* B = rocke_b_param(b, "B", pf32, NULL);
    rocke_value_t* C = rocke_b_param(b, "C", pf32, NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* a = rocke_b_global_load_f32(b, A, tid, 4);
    rocke_value_t* bb = rocke_b_global_load_f32(b, B, tid, 4);
    rocke_value_t* s = rocke_b_fadd(b, a, bb);
    rocke_b_global_store(b, C, tid, s, 4);
    rocke_b_ret(b);
    return 0;
}

static int build_forloop(rocke_ir_builder_t* b)
{
    const rocke_type_t* pf32 = rocke_ptr_type(b, rocke_f32(), "global");
    rocke_value_t* C = rocke_b_param(b, "C", pf32, NULL);
    rocke_value_t* lo = rocke_b_const_i32(b, 0);
    rocke_value_t* hi = rocke_b_const_i32(b, 16);
    rocke_value_t* step = rocke_b_const_i32(b, 1);
    rocke_value_t* acc0 = rocke_b_const_f32(b, 0.0);
    rocke_iter_arg_t iters[1];
    iters[0].name = "acc";
    iters[0].init = acc0;
    rocke_for_t f = rocke_b_scf_for_iter(b,
                                         lo,
                                         hi,
                                         step,
                                         iters,
                                         1,
                                         "k0",
                                         /*unroll=*/false,
                                         /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, f.body);
    {
        rocke_value_t* acc = f.iter_vars[0];
        rocke_value_t* one = rocke_b_const_f32(b, 1.0);
        rocke_value_t* nacc = rocke_b_fadd(b, acc, one);
        rocke_value_t* yld[1];
        yld[0] = nacc;
        rocke_b_scf_yield(b, yld, 1);
    }
    rocke_b_region_leave(b);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_b_global_store(b, C, tid, f.op->results[0], 4);
    rocke_b_ret(b);
    return 0;
}

static int build_vector(rocke_ir_builder_t* b)
{
    const rocke_type_t* pf16 = rocke_ptr_type(b, rocke_f16(), "global");
    rocke_value_t* C = rocke_b_param(b, "C", pf16, NULL);
    rocke_value_t* s = rocke_b_const_f32(b, 2.0);
    rocke_value_t* v = rocke_b_vector_splat(b, s, 4); /* <4 x f32> */
    rocke_value_t* w = rocke_b_vector_add(b, v, v); /* <4 x f32> */
    rocke_value_t* h = rocke_b_vec_trunc_f32_to_f16(b, w); /* <4 x f16> */
    rocke_value_t* e = rocke_b_vec_extract(b, h, 0); /* f16 */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_b_store_f16(b, C, tid, e);
    rocke_b_ret(b);
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <scalar|memory|forloop|vector>\n", argv[0]);
        return 2;
    }
    const char* which = argv[1];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "parity_kernel") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }

    int rc;
    if(strcmp(which, "scalar") == 0)
        rc = build_scalar(&b);
    else if(strcmp(which, "memory") == 0)
        rc = build_memory(&b);
    else if(strcmp(which, "forloop") == 0)
        rc = build_forloop(&b);
    else if(strcmp(which, "vector") == 0)
        rc = build_vector(&b);
    else
    {
        fprintf(stderr, "unknown kernel %s\n", which);
        rocke_ir_builder_free(&b);
        return 2;
    }
    (void)rc;

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
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
    rocke_ir_builder_free(&b);
    return 0;
}
