// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/smoke.c -- link-only smoke test for the rocke (ckc) engine.
 *
 * Purpose: prove the symbol graph of librocke_core resolves into a runnable
 * executable. It initializes an IR builder, builds a trivial kernel body,
 * runs the LLVM lowerer, and tears the builder down. A clean LINK of this
 * binary is the pass criterion; runtime status codes are informational.
 */
#include <stdio.h>
#include <stdlib.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

int main(void)
{
    rocke_ir_builder_t b;
    rocke_status_t st = rocke_ir_builder_init(&b, "rocke_smoke_kernel");
    if(st != ROCKE_OK)
    {
        fprintf(stderr, "rocke_ir_builder_init failed: status=%d\n", (int)st);
        return 1;
    }

    /* Build a trivial kernel body: c = const(1) ; r = c + c. This touches the
     * arith/const/add builder paths so the op graph is non-empty. */
    rocke_value_t* c = rocke_b_const_i32(&b, 1);
    rocke_value_t* r = rocke_b_add(&b, c, c);
    (void)r;

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error after body: %s\n", rocke_ir_builder_error(&b));
        /* still proceed -- link is what we are validating */
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);

    char* llvm_text = NULL;
    rocke_status_t lst
        = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
    printf("rocke_lower_kernel_to_llvm: status=%d, text=%s\n",
           (int)lst,
           llvm_text ? "non-null" : "null");
    if(llvm_text)
        free(llvm_text);

    rocke_ir_builder_free(&b);
    printf("rocke_smoke: builder lifecycle + LLVM lower symbols resolved.\n");
    return 0;
}
