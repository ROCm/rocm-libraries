// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_MFMA_SCALE_INTERNAL_H
#define ROCKE_MFMA_SCALE_INTERNAL_H
#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/storage.h"
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
#endif
