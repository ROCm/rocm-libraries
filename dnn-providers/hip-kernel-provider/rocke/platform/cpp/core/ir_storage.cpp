// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/ir_storage.py: addressable storage units, not ABI carriers. */
#include "rocke/ir_storage.h"

#include <string.h>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"
#include "rocke/storage.h"

const rocke_type_t* rocke_storage_ir_type(const char* dtype)
{
    const rocke_dtype_info_t* info = rocke_dtype_info(dtype);
    if(!info)
        return NULL;
    if(strcmp(info->name, "tf32") == 0)
        return rocke_i32();
    if(info->encoded_bits % 8 || strcmp(info->name, "e8m0") == 0 || strcmp(info->name, "e5m3") == 0)
        return rocke_i8();
    return rocke_dtype_to_ir_type(info->name);
}

const rocke_type_t* rocke_storage_element_type(const rocke_type_t* dtype)
{
    const auto* info = dtype ? rocke_dtype_info(dtype->name) : NULL;
    if(!info || strcmp(info->name, "i1") == 0)
        return NULL;
    return info->encoded_bits < 8 ? rocke_i32() : rocke_storage_ir_type(info->name);
}

const rocke_type_t*
    rocke_storage_vector_type(rocke_ir_builder_t* b, const rocke_type_t* dtype, int count)
{
    return ckc::guard_builder(b, [&]() -> const rocke_type_t* {
        if(!rocke_i_live(b))
            return NULL;
        bool packed;
        rocke_fragment_packing_t packing;
        char err[ROCKE_ERR_MSG_CAP];
        if(!dtype
           || !rocke_register_packing(dtype->name, count, &packed, &packing, err, sizeof(err)))
            ckc::raise_status(ROCKE_ERR_VALUE, dtype ? err : "storage dtype must name an IR type");
        return rocke_vector_type(
            b, rocke_storage_element_type(dtype), packed ? int(packing.carrier_count) : count);
    });
}
