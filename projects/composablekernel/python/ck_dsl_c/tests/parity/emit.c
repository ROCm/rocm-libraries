/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/emit.c -- C-side emitter for the ck_dsl_c parity harness.
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

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

static int build_scalar(ckc_ir_builder_t *b) {
    ckc_value_t *c = ckc_b_const_i32(b, 1);
    ckc_value_t *r = ckc_b_add(b, c, c);
    (void)r;
    ckc_b_ret(b);
    return 0;
}

static int build_memory(ckc_ir_builder_t *b) {
    const ckc_type_t *pf32 = ckc_ptr_type(b, ckc_f32(), "global");
    ckc_value_t *A = ckc_b_param(b, "A", pf32, NULL);
    ckc_value_t *B = ckc_b_param(b, "B", pf32, NULL);
    ckc_value_t *C = ckc_b_param(b, "C", pf32, NULL);
    ckc_value_t *tid = ckc_b_thread_id_x(b);
    ckc_value_t *a = ckc_b_global_load_f32(b, A, tid, 4);
    ckc_value_t *bb = ckc_b_global_load_f32(b, B, tid, 4);
    ckc_value_t *s = ckc_b_fadd(b, a, bb);
    ckc_b_global_store(b, C, tid, s, 4);
    ckc_b_ret(b);
    return 0;
}

static int build_forloop(ckc_ir_builder_t *b) {
    const ckc_type_t *pf32 = ckc_ptr_type(b, ckc_f32(), "global");
    ckc_value_t *C = ckc_b_param(b, "C", pf32, NULL);
    ckc_value_t *lo = ckc_b_const_i32(b, 0);
    ckc_value_t *hi = ckc_b_const_i32(b, 16);
    ckc_value_t *step = ckc_b_const_i32(b, 1);
    ckc_value_t *acc0 = ckc_b_const_f32(b, 0.0);
    ckc_iter_arg_t iters[1];
    iters[0].name = "acc";
    iters[0].init = acc0;
    ckc_for_t f = ckc_b_scf_for_iter(b, lo, hi, step, iters, 1, "k0",
                                     /*unroll=*/false,
                                     /*elide_trailing_barrier=*/true);
    ckc_b_region_enter(b, f.body);
    {
        ckc_value_t *acc = f.iter_vars[0];
        ckc_value_t *one = ckc_b_const_f32(b, 1.0);
        ckc_value_t *nacc = ckc_b_fadd(b, acc, one);
        ckc_value_t *yld[1];
        yld[0] = nacc;
        ckc_b_scf_yield(b, yld, 1);
    }
    ckc_b_region_leave(b);
    ckc_value_t *tid = ckc_b_thread_id_x(b);
    ckc_b_global_store(b, C, tid, f.op->results[0], 4);
    ckc_b_ret(b);
    return 0;
}

static int build_vector(ckc_ir_builder_t *b) {
    const ckc_type_t *pf16 = ckc_ptr_type(b, ckc_f16(), "global");
    ckc_value_t *C = ckc_b_param(b, "C", pf16, NULL);
    ckc_value_t *s = ckc_b_const_f32(b, 2.0);
    ckc_value_t *v = ckc_b_vector_splat(b, s, 4);   /* <4 x f32> */
    ckc_value_t *w = ckc_b_vector_add(b, v, v);     /* <4 x f32> */
    ckc_value_t *h = ckc_b_vec_trunc_f32_to_f16(b, w); /* <4 x f16> */
    ckc_value_t *e = ckc_b_vec_extract(b, h, 0);    /* f16 */
    ckc_value_t *tid = ckc_b_thread_id_x(b);
    ckc_b_store_f16(b, C, tid, e);
    ckc_b_ret(b);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <scalar|memory|forloop|vector>\n", argv[0]);
        return 2;
    }
    const char *which = argv[1];

    ckc_ir_builder_t b;
    if (ckc_ir_builder_init(&b, "parity_kernel") != CKC_OK) {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }

    int rc;
    if (strcmp(which, "scalar") == 0)        rc = build_scalar(&b);
    else if (strcmp(which, "memory") == 0)   rc = build_memory(&b);
    else if (strcmp(which, "forloop") == 0)  rc = build_forloop(&b);
    else if (strcmp(which, "vector") == 0)   rc = build_vector(&b);
    else { fprintf(stderr, "unknown kernel %s\n", which); ckc_ir_builder_free(&b); return 2; }
    (void)rc;

    if (!ckc_ir_builder_ok(&b)) {
        fprintf(stderr, "builder error: %s\n", ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return 1;
    }

    ckc_kernel_def_t *kernel = ckc_ir_builder_kernel(&b);
    char *llvm_text = NULL;
    char err[CKC_ERR_MSG_CAP];
    err[0] = 0;
    ckc_status_t st = ckc_lower_kernel_to_llvm_ex(kernel, CKC_LLVM_FLAVOR_AUTO,
                                                  "gfx950", &llvm_text,
                                                  err, sizeof err);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}
