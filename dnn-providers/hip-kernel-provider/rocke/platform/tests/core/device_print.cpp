// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int fail(const char* message)
{
    fprintf(stderr, "%s\n", message);
    return 1;
}

static int expect_invalid_text(const char* text)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "invalid_text") != ROCKE_OK)
        return fail("invalid text builder init failed");
    rocke_print_item_t item = {ROCKE_PRINT_TEXT, text, NULL, NULL};
    rocke_b_device_print(&b, &item, 1, NULL, "compact", "none");
    int accepted = rocke_ir_builder_ok(&b);
    rocke_ir_builder_free(&b);
    return accepted ? fail("device_print accepted non-ASCII text") : 0;
}

int main(void)
{
    const char non_ascii[] = {(char)0x80, '\0'};
    if(expect_invalid_text(non_ascii))
        return 1;

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "print_proto") != ROCKE_OK)
        return fail("builder init failed");

    rocke_value_t* pointer
        = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
    rocke_value_t* integer = rocke_b_const_i32(&b, -5);
    rocke_value_t* floating = rocke_b_const_f32(&b, 6.5);
    rocke_value_t* predicate = rocke_b_cmp_eq(&b, integer, integer);
    rocke_print_item_t items[] = {
        {ROCKE_PRINT_TEXT, "state=", NULL, NULL},
        {ROCKE_PRINT_VALUE, NULL, integer, NULL},
        {ROCKE_PRINT_TEXT, " unsigned=", NULL, NULL},
        {ROCKE_PRINT_VALUE, NULL, integer, "u32"},
        {ROCKE_PRINT_TEXT, " f=", NULL, NULL},
        {ROCKE_PRINT_VALUE, NULL, floating, NULL},
        {ROCKE_PRINT_TEXT, " ok=", NULL, NULL},
        {ROCKE_PRINT_VALUE, NULL, predicate, NULL},
        {ROCKE_PRINT_TEXT, " p=", NULL, NULL},
        {ROCKE_PRINT_VALUE, NULL, pointer, NULL},
    };
    rocke_b_device_print(
        &b, items, (int)(sizeof(items) / sizeof(items[0])), predicate, "compact", "ensure_newline");
    if(!rocke_ir_builder_ok(&b))
        return fail(rocke_ir_builder_error(&b));

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    rocke_diag_t* diagnostics = NULL;
    size_t diagnostic_count = 0;
    if(rocke_verify(kernel, &diagnostics, &diagnostic_count) != ROCKE_OK)
        return fail("verify failed");
    for(size_t i = 0; i < diagnostic_count; ++i)
    {
        if(diagnostics[i].severity == ROCKE_DIAG_ERROR)
            return fail(diagnostics[i].message);
    }
    rocke_diags_free(diagnostics, diagnostic_count);

    char* serialized = NULL;
    char* llvm = NULL;
    if(rocke_ir_serialize(kernel, &serialized) != ROCKE_OK || !serialized)
        return fail("serialize failed");
    if(!strstr(serialized, "format = s:\"u32\", kind = s:\"value\""))
        return fail("serialized u32 descriptor missing");
    if(!strstr(serialized, "predicate_operand = i:5"))
        return fail("serialized predicate missing");

    if(rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm) != ROCKE_OK
       || !llvm)
        return fail("lowering failed");
    if(!strstr(llvm, "state=%lld unsigned=%llu f=%.9g ok=%s p=%p"))
        return fail("canonical format lowering missing");
    if(!strstr(llvm, "br i1 %eq3, label %device.print."))
        return fail("predicate branch missing");
    if(!strstr(llvm, "ptrtoint ptr addrspace(1) %p to i64"))
        return fail("pointer lowering missing");

    rocke_ir_builder_t parsed_builder;
    rocke_kernel_def_t* parsed = NULL;
    char* serialized_again = NULL;
    char* llvm_again = NULL;
    if(rocke_ir_builder_init(&parsed_builder, "parsed") != ROCKE_OK
       || rocke_ir_parse(serialized, &parsed_builder, &parsed) != ROCKE_OK || !parsed)
        return fail("parse failed");
    if(rocke_ir_serialize(parsed, &serialized_again) != ROCKE_OK
       || strcmp(serialized, serialized_again) != 0)
        return fail("serialization roundtrip changed record");
    if(rocke_lower_kernel_to_llvm(parsed, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_again) != ROCKE_OK
       || strcmp(llvm, llvm_again) != 0)
        return fail("lowering roundtrip changed record");

    free(serialized);
    free(serialized_again);
    free(llvm);
    free(llvm_again);
    rocke_ir_builder_free(&parsed_builder);
    rocke_ir_builder_free(&b);
    return 0;
}
