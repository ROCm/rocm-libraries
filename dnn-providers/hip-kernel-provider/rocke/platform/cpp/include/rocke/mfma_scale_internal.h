// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_MFMA_SCALE_INTERNAL_H
#define ROCKE_MFMA_SCALE_INTERNAL_H
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/storage.h"
#include <stdio.h>
#include <string.h>
static inline const rocke_mma_op_t* rocke_gfx950_scaled_mfma(const char* id)
{
    const auto* target = rocke_arch_target_from_gfx("gfx950");
    const auto* atom = rocke_mma_catalog_by_op_id(&target->mma, id);
    return atom && strcmp(atom->family, "mfma_scaled") == 0 ? atom : NULL;
}
static inline int rocke_mfma_matrix_format(const char* dtype)
{
    const char* formats[] = {"fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"};
    for(int i = 0; i < 5; ++i)
        if(strcmp(dtype, formats[i]) == 0)
            return i;
    return -1;
}
static inline rocke_matrix_fragment_layout_t rocke_mfma_matrix_layout(const char* dtype)
{
    int fmt = rocke_mfma_matrix_format(dtype);
    int bits = fmt < 2 ? 8 : fmt < 4 ? 6 : 4;
    rocke_bit_packing_t packing;
    rocke_fragment_packing_t fragment;
    rocke_matrix_fragment_layout_t layout;
    if(fmt < 0 || !rocke_bit_packing_init(&packing, bits, 0)
       || !rocke_fragment_packing_init(&fragment, &packing, 32, 32, 8)
       || !rocke_matrix_fragment_layout_init(&layout, &fragment, fmt < 2 ? 16 : 32, 2, 32))
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled MFMA matrix dtype");
    return layout;
}
static inline const char* rocke_scaled_mfma_error(const char* id, rocke_value_t* const* args, int n)
{
    if(!id || !rocke_arch_mma_op_id_family(id)
       || strcmp(rocke_arch_mma_op_id_family(id), "mfma_scaled") != 0)
        return NULL;
    /* One refusal for a wrong count or a wrong type, as mfma_scale.py compares
     * the whole operand type list. */
    const char* error = "scaled MFMA requires two vec<i32x8> payloads, vec<f32x16> "
                        "accumulator and two i32 E8M0 scales";
    const char* types[] = {"vec<i32x8>", "vec<i32x8>", "vec<f32x16>", "i32", "i32"};
    if(n != 5)
        return error;
    for(int i = 0; i < 5; ++i)
        if(!args[i] || strcmp(args[i]->type->name, types[i]) != 0)
            return error;
    return NULL;
}

/* The fixed-operation 16x16x128 MX atom; mfma_scale.py MX16_OP_ID. */
#define ROCKE_MX16_OP_ID "mfma_scale_f32_16x16x128_f8f6f4"

/* Whether tile.mma with this op_id takes a scale_sel: a mfma_scaled catalog
 * atom or the 16x16x128 MX atom (mfma_scale.py accepts_scale_sel). */
static inline bool rocke_mma_accepts_scale_sel(const char* id)
{
    if(!id)
        return false;
    if(strncmp(id, "tile.", 5) == 0)
        id += 5;
    return strcmp(id, ROCKE_MX16_OP_ID) == 0 || rocke_gfx950_scaled_mfma(id) != NULL;
}

/* scale_sel: two byte indices in 0..3, the byte of the A and of the B scale
 * VGPR each lane reads (opsel_a / opsel_b); byte 0 of both when absent.
 * Returns false on any other value (mfma_scale.py scale_sel_error). An int
 * list attr is the only accepted kind, so a bool or float never coerces. */
static inline bool
    rocke_mma_scale_sel(const rocke_attr_map_t* attrs, int64_t* sel_a, int64_t* sel_b)
{
    const rocke_attr_value_t* v = rocke_attr_get(attrs, "scale_sel");
    *sel_a = 0;
    *sel_b = 0;
    if(!v)
        return true;
    if(v->kind != ROCKE_ATTR_INT_LIST || v->u.ilist.count != 2)
        return false;
    for(int i = 0; i < 2; ++i)
        if(v->u.ilist.ints[i] < 0 || v->u.ilist.ints[i] > 3)
            return false;
    *sel_a = v->u.ilist.ints[0];
    *sel_b = v->u.ilist.ints[1];
    return true;
}

/* Operand / result check for the 16x16x128 MX atom (mfma_scale.py
 * mx16_operand_error): 32-byte A and B, vec<f32x4> accumulator, two i32
 * scales, vec<f32x4> result. Formats the message into buf. */
static inline bool rocke_mx16_operand_error(rocke_value_t* const* args,
                                            int n,
                                            rocke_value_t* const* results,
                                            int nres,
                                            char* buf,
                                            size_t cap)
{
    const char* payloads[] = {"vec<i32x8>", "vec<i8x32>", "vec<fp8e4m3x32>", "vec<bf8e5m2x32>"};
    const char* tail[] = {"vec<f32x4>", "i32", "i32"};
    bool ok = n == 5;
    for(int i = 0; ok && i < 2; ++i)
    {
        bool match = false;
        for(int p = 0; args[i] && p < 4; ++p)
            match = match || strcmp(args[i]->type->name, payloads[p]) == 0;
        ok = match;
    }
    for(int i = 0; ok && i < 3; ++i)
        ok = args[i + 2] && strcmp(args[i + 2]->type->name, tail[i]) == 0;
    if(!ok)
    {
        size_t len = (size_t)snprintf(buf,
                                      cap,
                                      ROCKE_MX16_OP_ID
                                      " requires A and B of 32 bytes (vec<i32x8>, vec<i8x32>, "
                                      "vec<fp8e4m3x32>, vec<bf8e5m2x32>), a vec<f32x4> accumulator "
                                      "and two i32 E8M0 scales, got (");
        for(int i = 0; i < n && len < cap; ++i)
            len += (size_t)snprintf(
                buf + len, cap - len, "%s%s", i ? ", " : "", args[i] ? args[i]->type->name : "?");
        if(len < cap)
            snprintf(buf + len, cap - len, ")");
        return true;
    }
    if(results && (nres != 1 || strcmp(results[0]->type->name, "vec<f32x4>") != 0))
    {
        snprintf(buf, cap, ROCKE_MX16_OP_ID " result must be vec<f32x4>");
        return true;
    }
    return false;
}

/* The Python and C++ engines lower the 16x16x128 MX atom with fp8e4m3 A and B
 * only (cbsz = blgp = 0); any other a_format / b_format is refused rather than
 * silently computed as fp8 (mfma_scale.py mx16_format_error). The direct
 * compiler reads all five formats. */
static inline bool rocke_mx16_format_error(const rocke_attr_map_t* attrs, char* buf, size_t cap)
{
    const char* keys[] = {"a_format", "b_format"};
    for(int i = 0; i < 2; ++i)
    {
        const rocke_attr_value_t* v = rocke_attr_get(attrs, keys[i]);
        if(!v || (v->kind == ROCKE_ATTR_STR && strcmp(v->u.s, "fp8e4m3") == 0))
            continue;
        snprintf(buf,
                 cap,
                 ROCKE_MX16_OP_ID ": %s %s%s%s is not lowered by the Python or C++ engine "
                                  "(fp8e4m3 only); use a mfma_scaled catalog atom or "
                                  "IRBuilder.mfma_scale_mx for other formats",
                 keys[i],
                 v->kind == ROCKE_ATTR_STR ? "'" : "",
                 v->kind == ROCKE_ATTR_STR ? v->u.s : "<non-string>",
                 v->kind == ROCKE_ATTR_STR ? "'" : "");
        return true;
    }
    return false;
}
#endif
