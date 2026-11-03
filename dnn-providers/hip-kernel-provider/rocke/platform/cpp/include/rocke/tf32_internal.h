// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/tf32.py. Validation sees logical types, never carrier equality. */
#ifndef ROCKE_TF32_INTERNAL_H
#define ROCKE_TF32_INTERNAL_H
#include "rocke/arch_target.h"
#include "rocke/ir.h"
#include <stdio.h>
#include <string.h>
static inline int rocke_tf32_mma_count(const char* id)
{
    if(!id)
        return 0;
    if(strcmp(id, "mfma_f32_16x16x8_xf32") == 0)
        return 4;
    if(strcmp(id, "mfma_f32_32x32x4_xf32") == 0)
        return 16;
    return 0;
}
/* Mirrors tf32_mma_target_error: for a TF32 MMA `op_id` that the MMA catalog of
 * target `gfx` does not list, writes the error into `out` and returns true. Any
 * other op_id, and a TF32 op_id the catalog lists, returns false. Defined in
 * cpp/core/tf32.cpp, the one TU that needs the arch catalog for it. */
bool rocke_tf32_mma_target_error(const char* op_id, const char* gfx, char* out, size_t cap);
/* The operand contract of a TF32 atom the hardware does not encode (mirrors
 * core/tf32.py::_emulated_tf32_mma_error). Same ABI as the native atoms -- logical
 * TF32 A/B carried as I32, FP32 accumulator -- with fragment lengths from the
 * catalog. The message names the atom's own lengths, so it is formatted into a
 * per-thread buffer rather than returned as a literal. */
static inline const char* rocke_tf32_emulated_mma_error(
    const char* id, int a_len, int b_len, int c_len, rocke_value_t* const* args, int n)
{
    static thread_local char msg[192];
    char a[24], b[24], c[24];
    snprintf(a, sizeof a, "vec<tf32x%d>", a_len);
    snprintf(b, sizeof b, "vec<tf32x%d>", b_len);
    snprintf(c, sizeof c, "vec<f32x%d>", c_len);
    if(n != 3 || !args[0] || !args[1] || !args[2] || strcmp(args[0]->type->name, a) != 0
       || strcmp(args[1]->type->name, b) != 0 || strcmp(args[2]->type->name, c) != 0)
    {
        snprintf(msg, sizeof msg, "%s requires %s, %s and %s", id, a, b, c);
        return msg;
    }
    return NULL;
}
static inline const char*
    rocke_tf32_emulated_result_error(const char* id, int c_len, const rocke_op_t* op)
{
    static thread_local char msg[192];
    char c[24];
    snprintf(c, sizeof c, "vec<f32x%d>", c_len);
    if(op->num_results != 1 || strcmp(op->results[0]->type->name, c) != 0)
    {
        snprintf(msg, sizeof msg, "%s result must match its FP32 accumulator", id);
        return msg;
    }
    return NULL;
}
static inline const char* rocke_tf32_mma_error(const char* id, rocke_value_t* const* args, int n)
{
    int count = rocke_tf32_mma_count(id);
    if(!count)
    {
        int a_len, b_len, c_len;
        if(rocke_arch_mma_op_id_emulated_operands(id, &a_len, &b_len, &c_len))
            return rocke_tf32_emulated_mma_error(id, a_len, b_len, c_len, args, n);
        for(int i = 0; i < n; ++i)
            if(args[i]
               && (strcmp(args[i]->type->name, "tf32") == 0
                   || strncmp(args[i]->type->name, "vec<tf32x", 9) == 0))
                return "TF32 operands require an XF32 MMA";
        return NULL;
    }
    const char* acc = count == 4 ? "vec<f32x4>" : "vec<f32x16>";
    if(n != 3 || !args[0] || !args[1] || !args[2] || strcmp(args[0]->type->name, "vec<tf32x2>") != 0
       || strcmp(args[1]->type->name, "vec<tf32x2>") != 0 || strcmp(args[2]->type->name, acc) != 0)
        return "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator";
    return NULL;
}
static inline const char* rocke_tf32_op_error(const rocke_op_t* op)
{
    const char* id = rocke_attr_get_str(&op->attrs, "op_id");
    if(!id && strncmp(op->name, "tile.", 5) == 0)
        id = op->name + 5;
    const int is_mma = strcmp(op->name, "tile.mma") == 0 || strncmp(op->name, "tile.mfma", 9) == 0
                       || strncmp(op->name, "tile.wmma", 9) == 0;
    const char* error = is_mma ? rocke_tf32_mma_error(id, op->operands, op->num_operands) : NULL;
    if(error)
        return error;
    int count = rocke_tf32_mma_count(id);
    if(is_mma && !count)
    {
        int a_len, b_len, c_len;
        if(rocke_arch_mma_op_id_emulated_operands(id, &a_len, &b_len, &c_len))
        {
            error = rocke_tf32_emulated_result_error(id, c_len, op);
            if(error)
                return error;
        }
    }
    if(is_mma && !count)
        for(int i = 0; i < op->num_results; ++i)
            if(strcmp(op->results[i]->type->name, "tf32") == 0
               || strncmp(op->results[i]->type->name, "vec<tf32x", 9) == 0)
                return "MMA results must not use TF32";
    if(count
       && (op->num_results != 1
           || strcmp(op->results[0]->type->name, count == 4 ? "vec<f32x4>" : "vec<f32x16>") != 0))
        return "XF32 MMA result must match its FP32 accumulator";
    /* Only payload transport, reinterpretation, and selection preserve TF32. */
    if((strncmp(op->name, "arith.", 6) == 0 || strncmp(op->name, "math.", 5) == 0
        || strncmp(op->name, "vector.", 7) == 0)
       && strcmp(op->name, "arith.bitcast") != 0 && strcmp(op->name, "arith.select") != 0
       && strcmp(op->name, "vector.bitcast") != 0 && strcmp(op->name, "vector.extract") != 0
       && strcmp(op->name, "vector.pack") != 0 && strcmp(op->name, "vector.concat") != 0
       && strcmp(op->name, "vector.insert") != 0 && strcmp(op->name, "vector.splat") != 0
       && strcmp(op->name, "vector.select") != 0)
    {
        for(int i = 0; i < op->num_operands + op->num_results; ++i)
        {
            const rocke_value_t* v
                = i < op->num_operands ? op->operands[i] : op->results[i - op->num_operands];
            if(strcmp(v->type->name, "tf32") == 0 || strncmp(v->type->name, "vec<tf32x", 9) == 0)
                return "TF32 arithmetic requires an explicit conversion to f32";
        }
    }
    return NULL;
}
#endif
