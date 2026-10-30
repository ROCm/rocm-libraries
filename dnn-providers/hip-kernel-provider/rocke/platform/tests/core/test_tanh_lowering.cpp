// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_tanh_lowering.cpp -- focused C++ engine coverage for math.tanh.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.activations.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int expect_narrow_rejection(const rocke_type_t* type)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "tanh_reject") != ROCKE_OK)
    {
        return 1;
    }

    const char* expected = type == rocke_f16() ? "f16" : "bf16";
    char message[128];
    snprintf(message, sizeof(message), "math.tanh requires f32 operand, got %s", expected);

    bool rejected = false;
    try
    {
        rocke_value_t* x = rocke_b_param(&b, "x", type, NULL);
        rocke_value_t* result = rocke_b_tanh(&b, x);
        rejected = result == NULL && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                   && strcmp(rocke_ir_builder_error(&b), message) == 0;
    }
    catch(const ckc::Error& error)
    {
        rejected = error.code() == ROCKE_ERR_VALUE && strcmp(error.what(), message) == 0;
    }

    rocke_ir_builder_free(&b);
    return rejected ? 0 : 1;
}

static int expect_lowerable_f32(void)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "tanh_f32") != ROCKE_OK)
    {
        return 1;
    }

    rocke_value_t* x = rocke_b_param(&b, "x", rocke_f32(), NULL);
    if(rocke_tanh_via_exp2(&b, x) == NULL || !rocke_ir_builder_ok(&b))
    {
        rocke_ir_builder_free(&b);
        return 1;
    }

    char* llvm_text = NULL;
    rocke_status_t status = rocke_lower_kernel_to_llvm(
        rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM20, "gfx950", &llvm_text);
    int failed = status != ROCKE_OK || llvm_text == NULL || strstr(llvm_text, "llvm.tanh") != NULL
                 || strstr(llvm_text, "@llvm.exp2.f32") == NULL
                 || strstr(llvm_text, "@llvm.amdgcn.rcp.f32") == NULL
                 || strstr(llvm_text, "@llvm.fmuladd.f32") == NULL
                 || strstr(llvm_text, "fcmp olt float") == NULL
                 || strstr(llvm_text, "select i1") == NULL || strstr(llvm_text, "and i32") == NULL
                 || strstr(llvm_text, "or i32") == NULL;

    free(llvm_text);
    rocke_ir_builder_free(&b);
    return failed;
}

static int expect_verifier_rejection(const rocke_type_t* type)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "tanh_verify") != ROCKE_OK)
    {
        return 1;
    }

    rocke_value_t* x = rocke_b_param(&b, "x", type, NULL);
    rocke_value_t* operands[1] = {x};
    rocke_i_op1(&b, ROCKE_OP_MATH_TANH, operands, 1, type, NULL, "tanh");

    rocke_diag_t* diags = NULL;
    size_t count = 0;
    bool found = false;
    if(rocke_verify(rocke_ir_builder_kernel(&b), &diags, &count) == ROCKE_OK)
    {
        const char* expected = type == rocke_f16() ? "f16" : "bf16";
        char message[128];
        snprintf(message, sizeof(message), "math.tanh requires f32 operand, got %s", expected);
        for(size_t i = 0; i < count; ++i)
        {
            if(diags[i].severity == ROCKE_DIAG_ERROR && strcmp(diags[i].message, message) == 0)
            {
                found = true;
                break;
            }
        }
    }

    rocke_diags_free(diags, count);
    rocke_ir_builder_free(&b);
    return found ? 0 : 1;
}

int main(void)
{
    if(expect_narrow_rejection(rocke_f16()) || expect_narrow_rejection(rocke_bf16())
       || expect_verifier_rejection(rocke_f16()) || expect_verifier_rejection(rocke_bf16())
       || expect_lowerable_f32())
    {
        fprintf(stderr, "math.tanh lowering contract failed\n");
        return 1;
    }
    return 0;
}
