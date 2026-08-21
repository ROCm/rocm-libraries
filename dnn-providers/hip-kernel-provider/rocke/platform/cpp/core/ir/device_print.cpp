// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "device_print_internal.h"
#include "rocke/error_boundary.hpp"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

#define ROCKE_I_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES 4096
#define ROCKE_I_DEVICE_PRINT_DEFAULT_MAX_VALUE_COUNT 64
#define ROCKE_I_DEVICE_PRINT_MAX_CONFIGURED_LIMIT INT_MAX

static int parse_device_print_limit(
    const char* name, int default_value, int* value, char* error, size_t error_capacity)
{
    const char* raw = getenv(name);
    const char* begin;
    const char* limit;
    const char* cursor;
    char* end;
    long parsed;

    if(!raw)
    {
        *value = default_value;
        return 1;
    }
    begin = raw;
    while(*begin && isspace((unsigned char)*begin))
        ++begin;
    if(!*begin)
    {
        *value = default_value;
        return 1;
    }
    limit = begin + strlen(begin);
    while(limit > begin && isspace((unsigned char)limit[-1]))
        --limit;
    for(cursor = begin; cursor < limit; ++cursor)
    {
        if(!isdigit((unsigned char)*cursor))
            goto invalid;
    }
    errno = 0;
    parsed = strtol(begin, &end, 10);
    if(errno == 0 && end == limit && parsed >= 1
       && parsed <= ROCKE_I_DEVICE_PRINT_MAX_CONFIGURED_LIMIT)
    {
        *value = (int)parsed;
        return 1;
    }
invalid:
    if(error && error_capacity)
        snprintf(error,
                 error_capacity,
                 "%s='%s' is invalid; expected a base-10 integer in [1, %d]",
                 name,
                 raw,
                 ROCKE_I_DEVICE_PRINT_MAX_CONFIGURED_LIMIT);
    return 0;
}

int rocke_i_device_print_limits(rocke_device_print_limits_t* limits,
                                char* error,
                                size_t error_capacity)
{
    int literal_bytes;
    if(!limits
       || !parse_device_print_limit("ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES",
                                    ROCKE_I_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES,
                                    &literal_bytes,
                                    error,
                                    error_capacity)
       || !parse_device_print_limit("ROCKE_ENGINE_DEVICE_PRINT_MAX_VALUE_COUNT",
                                    ROCKE_I_DEVICE_PRINT_DEFAULT_MAX_VALUE_COUNT,
                                    &limits->max_value_count,
                                    error,
                                    error_capacity))
        return 0;
    limits->max_literal_bytes = (size_t)literal_bytes;
    return 1;
}

int rocke_i_valid_print_text(const unsigned char* text, size_t* bytes)
{
    size_t count = 0;
    while(*text)
    {
        if(*text > 0x7f)
            return 0;
        ++text;
        ++count;
    }
    if(bytes)
        *bytes = count;
    return 1;
}

static const char* print_default_format(const rocke_type_t* t)
{
    if(rocke_type_eq(t, rocke_i1()))
        return "bool";
    if(rocke_type_eq(t, rocke_i32()))
        return "i32";
    if(rocke_type_eq(t, rocke_f32()))
        return "f32";
    if(t && t->kind == ROCKE_TYPE_PTR && t->space && strcmp(t->space, "global") == 0)
        return "ptr";
    return NULL;
}

static int print_format_compatible(const rocke_type_t* t, const char* format)
{
    if(!t || !format)
        return 0;
    if(strcmp(format, "bool") == 0)
        return rocke_type_eq(t, rocke_i1());
    if(strcmp(format, "i32") == 0 || strcmp(format, "u32") == 0)
        return rocke_type_eq(t, rocke_i32());
    if(strcmp(format, "f32") == 0)
        return rocke_type_eq(t, rocke_f32());
    if(strcmp(format, "ptr") == 0)
        return t->kind == ROCKE_TYPE_PTR && t->space && strcmp(t->space, "global") == 0;
    return 0;
}

static void device_print_impl(rocke_ir_builder_t* b,
                              const rocke_print_item_t* items,
                              int num_items,
                              rocke_value_t* predicate,
                              const char* style,
                              const char* termination)
{
    rocke_attr_map_t attrs;
    rocke_value_t** operands = NULL;
    struct rocke_attr_map** metas = NULL;
    int input_items = num_items;
    int append_newline = 0;
    int operand_count = 0;
    int value_count = 0;
    size_t text_bytes = 0;
    rocke_device_print_limits_t limits;
    char limit_error[256];
    int i;
    if(!rocke_i_live(b))
        return;
    if(num_items < 0 || (num_items > 0 && !items))
        return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: invalid items");
    if(!rocke_i_device_print_limits(&limits, limit_error, sizeof(limit_error)))
        return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", limit_error);
    style = style ? style : "compact";
    termination = termination ? termination : "ensure_newline";
    if(strcmp(style, "compact") != 0)
        return (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "device_print: prototype style must be compact");
    if(strcmp(termination, "ensure_newline") != 0 && strcmp(termination, "none") != 0)
        return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: invalid termination");
    if(predicate && !rocke_type_eq(predicate->type, rocke_i1()))
        return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: predicate must be i1");

    if(strcmp(termination, "ensure_newline") == 0)
    {
        int ends = 0;
        if(num_items > 0 && items[num_items - 1].kind == ROCKE_PRINT_TEXT
           && items[num_items - 1].text)
        {
            size_t n = strlen(items[num_items - 1].text);
            ends = n > 0 && items[num_items - 1].text[n - 1] == '\n';
        }
        append_newline = !ends;
    }
    num_items += append_newline;
    if(num_items <= 0)
        return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: empty record");

    operands = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, sizeof(*operands) * (size_t)(num_items + (predicate ? 1 : 0)));
    metas
        = (struct rocke_attr_map**)rocke_arena_alloc(&b->arena, sizeof(*metas) * (size_t)num_items);
    if(!operands || !metas)
        return (void)rocke_i_set_err(b, ROCKE_ERR_OOM, "device_print: OOM");

    for(i = 0; i < num_items; ++i)
    {
        struct rocke_attr_map* meta
            = (struct rocke_attr_map*)rocke_arena_calloc(&b->arena, sizeof(*meta));
        if(!meta)
            return (void)rocke_i_set_err(b, ROCKE_ERR_OOM, "device_print: OOM");
        rocke_attr_map_init(meta);
        metas[i] = meta;
        if(i >= input_items || items[i].kind == ROCKE_PRINT_TEXT)
        {
            const char* text = i >= input_items ? "\n" : items[i].text;
            size_t n = 0;
            if(!text || !rocke_i_valid_print_text((const unsigned char*)text, &n))
                return (void)rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "device_print: Text must contain only ASCII");
            text_bytes += n;
            rocke_attr_set_str(b, meta, "kind", "text");
            rocke_attr_set_str(b, meta, "text", text);
        }
        else if(items[i].kind == ROCKE_PRINT_VALUE)
        {
            rocke_value_t* value = items[i].value;
            const char* format = value ? items[i].format : NULL;
            if(!value)
                return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: null Value");
            if(!format)
                format = print_default_format(value->type);
            if(!print_format_compatible(value->type, format))
                return (void)rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "device_print: incompatible format");
            operands[operand_count] = value;
            rocke_attr_set_str(b, meta, "format", format);
            rocke_attr_set_str(b, meta, "kind", "value");
            rocke_attr_set_str(b, meta, "layout", "scalar");
            rocke_attr_set_int(b, meta, "operand", operand_count);
            ++operand_count;
            ++value_count;
        }
        else
            return (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "device_print: unknown item kind");
    }
    if(text_bytes > limits.max_literal_bytes)
        return (void)rocke_i_set_err(b,
                                     ROCKE_ERR_VALUE,
                                     "device_print: literal text exceeds %zu bytes",
                                     limits.max_literal_bytes);
    if(value_count > limits.max_value_count)
        return (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "device_print: more than %d values", limits.max_value_count);

    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "items", 0);
    attrs.entries[attrs.count - 1].value.kind = ROCKE_ATTR_LIST;
    attrs.entries[attrs.count - 1].value.u.list.items = metas;
    attrs.entries[attrs.count - 1].value.u.list.count = num_items;
    rocke_attr_set_str(b, &attrs, "style", style);
    if(predicate)
    {
        rocke_attr_set_int(b, &attrs, "predicate_operand", operand_count);
        operands[operand_count++] = predicate;
    }
    (void)rocke_i_op0(b, ROCKE_OP_GPU_DEVICE_PRINT, operands, operand_count, &attrs);
}

void rocke_b_device_print(rocke_ir_builder_t* b,
                          const rocke_print_item_t* items,
                          int num_items,
                          rocke_value_t* predicate,
                          const char* style,
                          const char* termination)
{
    (void)ckc::guard_builder(b, [&]() -> rocke_ir_builder_t* {
        device_print_impl(b, items, num_items, predicate, style, termination);
        return b;
    });
}
