// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/lower_llvm_internal.h"

#include <string.h>

namespace ckc
{
/* __ockl_printf_append_args carries exactly seven i64 argument slots. */
enum
{
    ROCKE_OCKL_PRINTF_ARGUMENT_SLOTS = 7
};

/* Nine significant digits preserve every f32 value through decimal rendering. */
static const char* const device_print_f32_format = "%.9g";

static rocke_ll_printf_global_t add_printf_global(rocke_lower_t* L, const char* data, size_t len)
{
    rocke_ll_printf_global_t global;
    rocke_strbuf_t escaped;
    int rc = 0;
    rocke_strbuf_init(&escaped, 64);
    for(size_t i = 0; i < len; ++i)
    {
        unsigned char ch = (unsigned char)data[i];
        if(ch >= 0x20 && ch <= 0x7e && ch != '"' && ch != '\\')
            rocke_strbuf_append_char(&escaped, (char)ch);
        else
            rocke_strbuf_appendf(&escaped, "\\%02X", (unsigned)ch);
    }
    memset(&global, 0, sizeof(global));
    global.name = rocke_arena_printf(&L->arena, "@.rocke.printf.%zu", L->printf_globals.len);
    global.escaped = rocke_arena_strdup(&L->arena, rocke_strbuf_cstr(&escaped));
    global.byte_count = (int)len;
    rocke_vec_push(&L->arena, &L->printf_globals, global, rc);
    rocke_strbuf_free(&escaped);
    if(rc != 0 || !global.name || !global.escaped)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "device_print global");
    return global;
}

static void _op_gpu_device_print(rocke_lower_t* L, const rocke_op_t* op)
{
    const rocke_attr_value_t* items = rocke_attr_get(&op->attrs, "items");
    const rocke_attr_value_t* pred_attr = rocke_attr_get(&op->attrs, "predicate_operand");
    int predicate_index = pred_attr && pred_attr->kind == ROCKE_ATTR_INT ? (int)pred_attr->u.i : -1;
    rocke_ll_block_t* source_block = NULL;
    rocke_ll_block_t* print_block = NULL;
    const char** arguments;
    int argument_count = 0;
    rocke_strbuf_t static_text;

    if(!items || items->kind != ROCKE_ATTR_LIST)
        return (void)rocke_ll_fail(L, ROCKE_ERR_VALUE, "gpu.device_print missing items");
    if(predicate_index >= 0)
    {
        source_block = rocke_ll_current(L);
        print_block = rocke_ll_new_block(L, "device.print");
    }
    arguments = (const char**)rocke_arena_calloc(
        &L->arena, sizeof(*arguments) * (size_t)(items->u.list.count + 1));
    if(!arguments)
        return (void)rocke_ll_fail(L, ROCKE_ERR_OOM, "device_print arguments");
    rocke_strbuf_init(&static_text, 64);

    for(int i = 0; i < items->u.list.count; ++i)
    {
        const rocke_attr_map_t* item = items->u.list.items[i];
        const char* kind = rocke_attr_get_str(item, "kind");
        if(kind && strcmp(kind, "text") == 0)
        {
            const char* text = rocke_attr_get_str(item, "text");
            for(const char* p = text ? text : ""; *p; ++p)
            {
                rocke_strbuf_append_char(&static_text, *p);
                if(*p == '%')
                    rocke_strbuf_append_char(&static_text, '%');
            }
            continue;
        }

        int64_t operand_index = -1;
        const char* logical = rocke_attr_get_str(item, "format");
        if(!rocke_attr_get_int(item, "operand", &operand_index) || operand_index < 0
           || operand_index >= op->num_operands || !logical)
        {
            rocke_strbuf_free(&static_text);
            return (void)rocke_ll_fail(L, ROCKE_ERR_VALUE, "gpu.device_print malformed Value");
        }
        const rocke_value_t* value = op->operands[operand_index];
        const char* operand = rocke_ll_operand(L, value);
        if(strcmp(logical, "bool") == 0)
        {
            const char* tmp = rocke_ll_fresh(L, "printf_arg");
            rocke_strbuf_append(&static_text, "%c");
            /* Encode bool as an i64 character payload so it follows normal append_args
             * packetization and does not require a bool-only append branch. */
            rocke_ll_emitf(L, "  %s = select i1 %s, i64 116, i64 102", tmp, operand);
            arguments[argument_count++] = tmp;
        }
        else if(strcmp(logical, "i32") == 0 || strcmp(logical, "u32") == 0)
        {
            const char* tmp = rocke_ll_fresh(L, "printf_arg");
            rocke_strbuf_append(&static_text, strcmp(logical, "i32") == 0 ? "%lld" : "%llu");
            rocke_ll_emitf(L,
                           "  %s = %s i32 %s to i64",
                           tmp,
                           strcmp(logical, "i32") == 0 ? "sext" : "zext",
                           operand);
            arguments[argument_count++] = tmp;
        }
        else if(strcmp(logical, "f32") == 0)
        {
            const char* as_f64 = rocke_ll_fresh(L, "printf_f64");
            const char* bits = rocke_ll_fresh(L, "printf_arg");
            rocke_strbuf_append(&static_text, device_print_f32_format);
            rocke_ll_emitf(L, "  %s = fpext float %s to double", as_f64, operand);
            rocke_ll_emitf(L, "  %s = bitcast double %s to i64", bits, as_f64);
            arguments[argument_count++] = bits;
        }
        else if(strcmp(logical, "ptr") == 0)
        {
            const char* bits = rocke_ll_fresh(L, "printf_arg");
            rocke_strbuf_append(&static_text, "%p");
            rocke_ll_emitf(L,
                           "  %s = ptrtoint %s %s to i64",
                           bits,
                           rocke_ll_llvm_type(L, value->type),
                           operand);
            arguments[argument_count++] = bits;
        }
        else
        {
            rocke_strbuf_free(&static_text);
            return (void)rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "gpu.device_print unsupported logical format '%s'", logical);
        }
    }
    rocke_ll_need(L, "ockl.printf.begin");
    rocke_ll_need(L, "ockl.printf.append.string");
    if(argument_count > 0)
        rocke_ll_need(L, "ockl.printf.append.args");
    rocke_strbuf_append_char(&static_text, '\0');
    size_t format_len = static_text.len;
    rocke_ll_printf_global_t format_global = add_printf_global(L, static_text.data, format_len);
    rocke_strbuf_free(&static_text);
    const char* message = rocke_ll_fresh(L, "printf_msg");
    rocke_ll_emitf(L, "  %s = call i64 @__ockl_printf_begin(i64 0)", message);
    const char* next = rocke_ll_fresh(L, "printf_msg");
    rocke_ll_emitf(L,
                   "  %s = call i64 @__ockl_printf_append_string_n(i64 %s, "
                   "ptr addrspacecast (ptr addrspace(4) %s to ptr), i64 %zu, i32 %d)",
                   next,
                   message,
                   format_global.name,
                   format_len,
                   argument_count == 0 ? 1 : 0);
    message = next;

    for(int i = 0; i < argument_count;)
    {
        const char* args[ROCKE_OCKL_PRINTF_ARGUMENT_SLOTS];
        for(int slot = 0; slot < ROCKE_OCKL_PRINTF_ARGUMENT_SLOTS; ++slot)
            args[slot] = "0";
        int count = 0;
        while(i < argument_count && count < ROCKE_OCKL_PRINTF_ARGUMENT_SLOTS)
            args[count++] = arguments[i++];
        next = rocke_ll_fresh(L, "printf_msg");
        rocke_ll_emitf(L,
                       "  %s = call i64 @__ockl_printf_append_args(i64 %s, i32 %d, "
                       "i64 %s, i64 %s, i64 %s, i64 %s, i64 %s, i64 %s, i64 %s, i32 %d)",
                       next,
                       message,
                       count,
                       args[0],
                       args[1],
                       args[2],
                       args[3],
                       args[4],
                       args[5],
                       args[6],
                       i == argument_count ? 1 : 0);
        message = next;
    }

    if(source_block && print_block)
    {
        rocke_ll_block_t* join = rocke_ll_new_block(L, "device.print.end");
        const char* predicate = rocke_ll_operand(L, op->operands[predicate_index]);
        rocke_ll_block_emitf(L,
                             source_block,
                             "  br i1 %s, label %%%s, label %%%s",
                             predicate,
                             print_block->label,
                             join->label);
        source_block->terminated = 1;
        rocke_ll_block_emitf(L, print_block, "  br label %%%s", join->label);
        print_block->terminated = 1;
    }
}

void rocke_ll_register_device_print(void)
{
    rocke_ll_set_handler(ROCKE_OP_GPU_DEVICE_PRINT, _op_gpu_device_print);
}

} /* namespace ckc */
