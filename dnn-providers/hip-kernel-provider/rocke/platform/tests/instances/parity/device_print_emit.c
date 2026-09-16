/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C-side emitter for device_print_emit.py. The differential harness compares
 * canonical serialization, verifier diagnostics, and LLVM IR across engines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static void build_mixed(rocke_ir_builder_t* b)
{
    rocke_value_t* pointer = rocke_b_param(b, "p", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* integer = rocke_b_const_i32(b, -5);
    rocke_value_t* floating = rocke_b_const_f32(b, 6.5);
    rocke_value_t* predicate = rocke_b_cmp_eq(b, integer, integer);
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
        b, items, (int)(sizeof(items) / sizeof(items[0])), predicate, "compact", "ensure_newline");
}

static void build_packet_boundary(rocke_ir_builder_t* b, int count)
{
    rocke_value_t* value = rocke_b_const_i32(b, 1);
    rocke_value_t* true_value = rocke_b_cmp_eq(b, value, value);
    rocke_value_t* false_value = rocke_b_cmp_ne(b, value, value);
    rocke_print_item_t items[8];
    items[0] = (rocke_print_item_t){ROCKE_PRINT_VALUE, NULL, true_value, NULL};
    for(int i = 1; i < count - 1; ++i)
        items[i] = (rocke_print_item_t){ROCKE_PRINT_VALUE, NULL, value, "i32"};
    items[count - 1] = (rocke_print_item_t){ROCKE_PRINT_VALUE, NULL, false_value, NULL};
    rocke_b_device_print(b, items, count, NULL, "compact", "ensure_newline");
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..2> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = argc > 2 ? argv[2] : "ll";
    if(idx < 0 || idx > 2)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "device_print") != ROCKE_OK)
        return 1;
    if(idx == 0)
        build_mixed(&b);
    else
        build_packet_boundary(&b, idx == 1 ? 7 : 8);
    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* text = NULL;
        rocke_status_t status
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &text);
        if(status != ROCKE_OK || !text)
        {
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t status = rocke_ir_serialize(kernel, &text);
        if(status != ROCKE_OK || !text)
        {
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    {
        rocke_diag_t* diagnostics = NULL;
        size_t count = 0;
        if(rocke_verify(kernel, &diagnostics, &count) != ROCKE_OK)
        {
            rocke_ir_builder_free(&b);
            return 1;
        }
        for(size_t i = 0; i < count; ++i)
        {
            char* text = rocke_diag_to_string(&diagnostics[i]);
            if(text)
            {
                puts(text);
                free(text);
            }
        }
        rocke_diags_free(diagnostics, count);
    }
    rocke_ir_builder_free(&b);
    return 0;
}
