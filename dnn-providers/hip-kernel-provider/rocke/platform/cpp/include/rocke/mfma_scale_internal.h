// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* The f8f6f4 matrix formats and operand layouts, and the gfx950 scaled MFMA.
 * Mirrors core/arch/mfma_scale.py:
 * MATRIX_FORMATS             -> k_rocke_matrix_formats, rocke_mfma_matrix_format
 * narrow_matrix_format_pairs -> k_rocke_narrow_matrix_format_pairs
 * require_matrix_format      -> rocke_require_matrix_format
 * matrix_operand_layout      -> rocke_matrix_operand_layout
 * matrix_layout              -> rocke_mfma_matrix_layout
 * and target.narrow_f8f6f4_op_ids -> rocke_make_narrow_f8f6f4_op_ids.
 */
#ifndef ROCKE_MFMA_SCALE_INTERNAL_H
#define ROCKE_MFMA_SCALE_INTERNAL_H
#include "rocke/arch_target.h"
#include "rocke/dtypes.h"
#include "rocke/dtypes_internal.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/storage.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* The f8f6f4 matrix formats in format-code order: the code an f8f6f4
 * instruction takes for A and for B is the index. */
inline constexpr const char* k_rocke_matrix_formats[]
    = {"fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"};
inline constexpr int k_rocke_matrix_format_count = 5;

/* The encoded bits of the matrix format with code `format`. */
constexpr int rocke_matrix_format_bits(int format)
{
    return format < 2 ? 8 : format < 4 ? 6 : 4;
}

/* One (A, B) pair of format codes. */
struct rocke_matrix_format_pair
{
    int a, b;
};

/* Every (A, B) pair with a 6-bit or 4-bit operand, A-major in format-code
 * order: 21 pairs; the four 8-bit x 8-bit pairs are not in the list. */
struct rocke_matrix_format_pairs
{
    rocke_matrix_format_pair pairs[21];
    int count;
};
consteval rocke_matrix_format_pairs rocke_make_narrow_matrix_format_pairs()
{
    rocke_matrix_format_pairs result{};
    for(int a = 0; a < k_rocke_matrix_format_count; ++a)
        for(int b = 0; b < k_rocke_matrix_format_count; ++b)
            if(rocke_matrix_format_bits(a) < 8 || rocke_matrix_format_bits(b) < 8)
                result.pairs[result.count++] = {a, b};
    return result;
}
inline constexpr rocke_matrix_format_pairs k_rocke_narrow_matrix_format_pairs
    = rocke_make_narrow_matrix_format_pairs();
static_assert(k_rocke_narrow_matrix_format_pairs.count == 21);

/* The op_ids of one f8f6f4 instruction, one per narrow format pair: `op_id`
 * with _<a>_<b> appended in canonical dtype names, in pair order. Keep the
 * result in a constexpr variable of its own so the strings have static
 * storage. */
struct rocke_f8f6f4_op_ids
{
    char ids[21][80];
};
consteval rocke_f8f6f4_op_ids rocke_make_narrow_f8f6f4_op_ids(const char* op_id)
{
    rocke_f8f6f4_op_ids result{};
    for(int i = 0; i < k_rocke_narrow_matrix_format_pairs.count; ++i)
    {
        const rocke_matrix_format_pair& pair = k_rocke_narrow_matrix_format_pairs.pairs[i];
        int length = 0;
        auto append = [&](const char* text) {
            for(int j = 0; text[j]; ++j)
            {
                if(length >= 79)
                    throw "f8f6f4 op_id too long";
                result.ids[i][length++] = text[j];
            }
        };
        append(op_id);
        append("_");
        append(k_rocke_matrix_formats[pair.a]);
        append("_");
        append(k_rocke_matrix_formats[pair.b]);
    }
    return result;
}

static inline const rocke_mma_op_t* rocke_gfx950_scaled_mfma(const char* id)
{
    const auto* target = rocke_arch_target_from_gfx("gfx950");
    const auto* atom = rocke_mma_catalog_by_op_id(&target->mma, id);
    return atom && strcmp(atom->family, "mfma_scaled") == 0 ? atom : NULL;
}

/* The format code of canonical matrix dtype `dtype`, or -1. */
static inline int rocke_mfma_matrix_format(const char* dtype)
{
    for(int i = 0; i < k_rocke_matrix_format_count; ++i)
        if(strcmp(dtype, k_rocke_matrix_formats[i]) == 0)
            return i;
    return -1;
}
/* --- K=128 f8f6f4 MFMA (16x16x128): format selectors and operand width ---
 * Mirrors core/arch/mfma_scale.py: DEFAULT_MATRIX_FORMAT, _type_bits,
 * f8f6f4_16x16x128_selectors and f8f6f4_16x16x128_operand_error. The messages
 * match the Python text. Every operand is 256 bits per lane (<8 x i32>); fp6
 * data fills the low 6 dwords and fp4 data the low 4. */
#define ROCKE_DEFAULT_MATRIX_FORMAT "fp8e4m3"
#define ROCKE_F8F6F4_16X16X128_OPERAND_BITS 256

/* Encoded bit width of an IR type, or -1 when it is not known. */
static inline int rocke_type_encoded_bits(const rocke_type_t* t)
{
    if(!t)
        return -1;
    if(t->kind == ROCKE_TYPE_VECTOR)
    {
        int e = rocke_type_encoded_bits(t->elem);
        return e < 0 ? -1 : t->count * e;
    }
    if(t->kind != ROCKE_TYPE_SCALAR)
        return -1;
    const rocke_dtype_info_t* info = rocke_dtype_info(t->name);
    return info ? info->encoded_bits : -1;
}

/* Returns NULL when the call is valid and sets *cbsz and *blgp. Otherwise
 * writes the message into msg and returns msg. a_format and b_format may be NULL
 * (default format). */
static inline const char* rocke_f8f6f4_16x16x128_check(const char* a_format,
                                                       const char* b_format,
                                                       const rocke_type_t* a_type,
                                                       const rocke_type_t* b_type,
                                                       char* msg,
                                                       size_t cap,
                                                       int* cbsz,
                                                       int* blgp)
{
    const char* fmts[2] = {a_format ? a_format : ROCKE_DEFAULT_MATRIX_FORMAT,
                           b_format ? b_format : ROCKE_DEFAULT_MATRIX_FORMAT};
    const char* roles[2] = {"a_format", "b_format"};
    const rocke_type_t* types[2] = {a_type, b_type};
    const char* type_roles[2] = {"A", "B"};
    int ids[2];
    for(int i = 0; i < 2; ++i)
    {
        ids[i] = rocke_mfma_matrix_format(fmts[i]);
        if(ids[i] < 0)
        {
            snprintf(msg,
                     cap,
                     "unknown %s '%s'; use one of fp8e4m3, bf8e5m2, fp6e2m3, fp6e3m2, fp4e2m1",
                     roles[i],
                     fmts[i]);
            return msg;
        }
    }
    if(strcmp(fmts[0], fmts[1]) != 0)
    {
        snprintf(msg,
                 cap,
                 "a_format and b_format must name the same format in this version; "
                 "got '%s' and '%s'",
                 fmts[0],
                 fmts[1]);
        return msg;
    }
    for(int i = 0; i < 2; ++i)
    {
        int bits = rocke_type_encoded_bits(types[i]);
        if(bits != ROCKE_F8F6F4_16X16X128_OPERAND_BITS)
        {
            char got[48];
            if(bits < 0)
                snprintf(got, sizeof(got), "an unknown width");
            else
                snprintf(got, sizeof(got), "%d bits", bits);
            snprintf(msg,
                     cap,
                     "the %s operand of the 16x16x128 f8f6f4 MFMA must have %d bits per lane "
                     "(pad fp6 and fp4 data to 8 dwords); got %s",
                     type_roles[i],
                     ROCKE_F8F6F4_16X16X128_OPERAND_BITS,
                     got);
            return msg;
        }
    }
    *cbsz = ids[0];
    *blgp = ids[1];
    return NULL;
}

/* The format code of matrix format `dtype` (any accepted spelling). Throws
 * ckc::ValueError naming the five formats when it is not one. */
static inline int rocke_require_matrix_format(const char* dtype)
{
    const int format = rocke_mfma_matrix_format(rocke_require_dtype_info(dtype)->name);
    if(format < 0)
        throw ckc::ValueError(
            ckc::format_error("'%s' is not an f8f6f4 matrix format; expected one of fp8e4m3, "
                              "bf8e5m2, fp6e2m3, fp6e3m2, fp4e2m1",
                              dtype));
    return format;
}

/* Where each value of one f8f6f4 operand of format `dtype` sits: `words`
 * 32-bit words holding 4 * words values per lane, packed from the low bit;
 * lane l holds row l % lanes_per_group of A (that column of B), and its values
 * are runs of consecutive K, 16 long for an 8-bit format and 32 for a 6-bit or
 * 4-bit one, which the `lane_groups` lane groups take in turn. */
static inline rocke_matrix_fragment_layout_t
    rocke_matrix_operand_layout(const char* dtype, int words, int lane_groups, int lanes_per_group)
{
    const int bits = rocke_matrix_format_bits(rocke_require_matrix_format(dtype));
    rocke_bit_packing_t packing;
    rocke_fragment_packing_t fragment;
    rocke_matrix_fragment_layout_t layout;
    if(words < 0 || !rocke_bit_packing_init(&packing, bits, 0)
       || !rocke_fragment_packing_init(&fragment, &packing, 4 * (uint64_t)words, 32, words)
       || !rocke_matrix_fragment_layout_init(
           &layout, &fragment, bits == 8 ? 16 : 32, lane_groups, lanes_per_group))
        throw ckc::ValueError("invalid matrix fragment chunk layout");
    return layout;
}

static inline rocke_matrix_fragment_layout_t rocke_mfma_matrix_layout(const char* dtype)
{
    if(rocke_mfma_matrix_format(dtype) < 0)
        ckc::raise_status(ROCKE_ERR_VALUE, "unsupported scaled MFMA matrix dtype");
    return rocke_matrix_operand_layout(dtype, 8, 2, 32);
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
#endif
