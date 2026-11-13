// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_global_load_index_width.cpp -- the C++ engine's scalar typed
 * global_load GEP takes its index operand's own width, so an i64 element index
 * (a paged KV cache past 2^31 bytes) lowers to an i64 GEP index and an i32 index
 * keeps the i32 GEP. Twin of test_global_load_index_width.py.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

static int expect_gep(const rocke_type_t* idx_type, const char* expected)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "gl_index_width") != ROCKE_OK)
    {
        return 1;
    }

    rocke_value_t* src = rocke_b_param(&b, "src", rocke_ptr_type(&b, rocke_bf16(), "global"), NULL);
    rocke_value_t* dst = rocke_b_param(&b, "dst", rocke_ptr_type(&b, rocke_bf16(), "global"), NULL);
    rocke_value_t* idx = rocke_b_param(&b, "idx", idx_type, NULL);
    rocke_value_t* v = rocke_b_global_load(&b, src, idx, rocke_bf16(), 2);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    rocke_b_global_store(&b, dst, zero, v, 2);
    rocke_b_ret(&b);

    char* llvm_text = NULL;
    rocke_status_t status = rocke_lower_kernel_to_llvm(
        rocke_ir_builder_kernel(&b), ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text);
    int failed = status != ROCKE_OK || llvm_text == NULL || strstr(llvm_text, expected) == NULL;
    if(failed)
    {
        fprintf(stderr, "missing \"%s\" in:\n%s\n", expected, llvm_text ? llvm_text : "(null)");
    }

    free(llvm_text);
    rocke_ir_builder_free(&b);
    return failed;
}

int main(void)
{
    if(expect_gep(rocke_i32(), "ptr addrspace(1) %src, i32 %idx")
       || expect_gep(rocke_i64(), "ptr addrspace(1) %src, i64 %idx"))
    {
        fprintf(stderr, "global_load index width contract failed\n");
        return 1;
    }
    return 0;
}
