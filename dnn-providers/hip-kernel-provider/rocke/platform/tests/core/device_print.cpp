// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#ifdef ROCKE_DEVICE_PRINT_MAX_LITERAL_BYTES
#error "device-print limits must not be part of the public C API"
#endif
#ifdef ROCKE_DEVICE_PRINT_MAX_VALUES
#error "device-print limits must not be part of the public C API"
#endif

static_assert(ROCKE_OP_MEMREF_GLOBAL_LOAD == 65, "public opcode ABI changed");
static_assert(ROCKE_OP_CF_RETURN == 179, "public opcode ABI changed");
static_assert(ROCKE_OP_GPU_DEVICE_PRINT == 180, "new opcodes must be appended");

static int fail(const char* message)
{
    fprintf(stderr, "%s\n", message);
    return 1;
}

static int set_test_environment(const char* name, const char* value)
{
#ifdef _WIN32
    return _putenv_s(name, value);
#else
    return setenv(name, value, 1);
#endif
}

static int clear_test_environment(const char* name)
{
#ifdef _WIN32
    return _putenv_s(name, "");
#else
    return unsetenv(name);
#endif
}

static int expect_text_limit(const char* text, int accepted, const char* error_fragment)
{
    rocke_ir_builder_t b;
    rocke_print_item_t item = {ROCKE_PRINT_TEXT, text, NULL, NULL};
    if(rocke_ir_builder_init(&b, "text_limit") != ROCKE_OK)
        return fail("text-limit builder init failed");
    rocke_b_device_print(&b, &item, 1, NULL, "compact", "none");
    int actual = rocke_ir_builder_ok(&b);
    if(actual != accepted)
    {
        const char* error = rocke_ir_builder_error(&b);
        fprintf(stderr, "unexpected text-limit result: %s\n", error ? error : "no error");
        rocke_ir_builder_free(&b);
        return 1;
    }
    if(!accepted && error_fragment)
    {
        const char* error = rocke_ir_builder_error(&b);
        if(!error || !strstr(error, error_fragment))
        {
            fprintf(stderr, "missing text-limit error fragment: %s\n", error_fragment);
            rocke_ir_builder_free(&b);
            return 1;
        }
    }
    rocke_ir_builder_free(&b);
    return 0;
}

static int expect_value_limit(void)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "value_limit") != ROCKE_OK)
        return fail("value-limit builder init failed");
    rocke_value_t* first = rocke_b_const_i32(&b, 1);
    rocke_value_t* second = rocke_b_const_i32(&b, 2);
    rocke_print_item_t items[] = {
        {ROCKE_PRINT_VALUE, NULL, first, NULL},
        {ROCKE_PRINT_VALUE, NULL, second, NULL},
    };
    rocke_b_device_print(&b, items, 2, NULL, "compact", "none");
    const char* error = rocke_ir_builder_error(&b);
    int failed = rocke_ir_builder_ok(&b) || !error || !strstr(error, "more than 1 values");
    rocke_ir_builder_free(&b);
    return failed ? fail("device_print did not enforce configured value limit") : 0;
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
    const char* literal_limit = "ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES";
    const char* value_limit = "ROCKE_ENGINE_DEVICE_PRINT_MAX_VALUE_COUNT";
    const char non_ascii[] = {(char)0x80, '\0'};
    if(expect_invalid_text(non_ascii))
        return 1;

    if(set_test_environment(literal_limit, " 4 ") || expect_text_limit("1234", 1, NULL)
       || expect_text_limit("12345", 0, "exceeds 4 bytes"))
        return 1;
    if(set_test_environment(literal_limit, "0") || expect_text_limit("x", 0, literal_limit)
       || clear_test_environment(literal_limit))
        return 1;
    if(set_test_environment(value_limit, "1") || expect_value_limit()
       || clear_test_environment(value_limit))
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
    if(!strstr(llvm, "state=%lld unsigned=%llu f=%.9g ok=%c p=%p"))
        return fail("canonical format lowering missing");
    if(!strstr(llvm, "select i1 %eq3, i64 116, i64 102"))
        return fail("bool character lowering missing");
    if(strstr(llvm, "c\"true\\00\"") || strstr(llvm, "c\"false\\00\""))
        return fail("bool string globals remain");
    if(!strstr(llvm, "br i1 %eq3, label %device.print."))
        return fail("predicate branch missing");
    if(!strstr(llvm, "sext i32 -5 to i64"))
        return fail("signed i32 promotion missing");
    if(!strstr(llvm, "zext i32 -5 to i64"))
        return fail("unsigned i32 promotion missing");
    if(!strstr(llvm, "fpext float 0x401A000000000000 to double"))
        return fail("f32 promotion missing");
    if(!strstr(llvm, "bitcast double %printf_f64.3 to i64"))
        return fail("f64 payload bitcast missing");
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
