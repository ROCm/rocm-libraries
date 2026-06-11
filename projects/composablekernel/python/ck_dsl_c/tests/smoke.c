/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/smoke.c -- link-only smoke test for the ck_dsl_c (ckc) engine.
 *
 * Purpose: prove the symbol graph of libckc_core resolves into a runnable
 * executable. It initializes an IR builder, builds a trivial kernel body,
 * runs the LLVM lowerer, and tears the builder down. A clean LINK of this
 * binary is the pass criterion; runtime status codes are informational.
 */
#include <stdio.h>
#include <stdlib.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

int main(void) {
    ckc_ir_builder_t b;
    ckc_status_t st = ckc_ir_builder_init(&b, "ckc_smoke_kernel");
    if (st != CKC_OK) {
        fprintf(stderr, "ckc_ir_builder_init failed: status=%d\n", (int)st);
        return 1;
    }

    /* Build a trivial kernel body: c = const(1) ; r = c + c. This touches the
     * arith/const/add builder paths so the op graph is non-empty. */
    ckc_value_t *c = ckc_b_const_i32(&b, 1);
    ckc_value_t *r = ckc_b_add(&b, c, c);
    (void)r;

    if (!ckc_ir_builder_ok(&b)) {
        fprintf(stderr, "builder error after body: %s\n", ckc_ir_builder_error(&b));
        /* still proceed -- link is what we are validating */
    }

    ckc_kernel_def_t *kernel = ckc_ir_builder_kernel(&b);

    char *llvm_text = NULL;
    ckc_status_t lst = ckc_lower_kernel_to_llvm(kernel,
                                                CKC_LLVM_FLAVOR_AUTO,
                                                "gfx950",
                                                &llvm_text);
    printf("ckc_lower_kernel_to_llvm: status=%d, text=%s\n",
           (int)lst, llvm_text ? "non-null" : "null");
    if (llvm_text) free(llvm_text);

    ckc_ir_builder_free(&b);
    printf("ckc_smoke: builder lifecycle + LLVM lower symbols resolved.\n");
    return 0;
}
