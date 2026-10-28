// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * The lane split of the transpose_bc twin (transpose_bc.cpp). Kept in its own
 * translation unit because it needs helper_rocke.helpers.transforms.h, whose
 * struct rocke_tensor_descriptor has the same tag as (but another definition
 * from) the one in helper_rocke.helpers.tensor_view.h that transpose_bc.cpp
 * uses for its views; both in one translation unit is a C++ redefinition.
 */

#include <string.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir_internal.h"

/* RichTensorDescriptor.naive("lane", lengths=[LANES_M, LANES_N],
 *     coord_names=["lr", "lc"]).transform(unmerge_magic("tid", ["lr", "lc"],
 *     dims=[LANES_M, LANES_N])).unmerge_lower(b, tid=tid) */
bool rocke_transpose_bc_lane_coords(rocke_ir_builder_t* b,
                                    rocke_value_t* tid,
                                    int lanes_m,
                                    int lanes_n,
                                    rocke_value_t** lr,
                                    rocke_value_t** lc)
{
    const int dims[2] = {lanes_m, lanes_n};
    const char* names[2] = {"lr", "lc"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "lane", dims, 2, nullptr, names, 2);
    if(desc == nullptr)
        return false;
    rocke_transform_t* xf = rocke_unmerge_magic(b, "tid", names, 2, dims);
    if(xf == nullptr)
        return false;
    const rocke_transform_t* chain[1] = {xf};
    rocke_tensor_descriptor_t* lane = rocke_tensor_descriptor_transform(b, desc, chain, 1);
    if(lane == nullptr)
        return false;
    const char* in_names[1] = {"tid"};
    rocke_value_t* in_values[1] = {tid};
    const char* out_names[8];
    rocke_value_t* out_values[8];
    const int n_out = rocke_tensor_descriptor_unmerge_lower(
        b, lane, in_names, in_values, 1, out_names, out_values, 8);
    *lr = nullptr;
    *lc = nullptr;
    for(int i = 0; i < n_out; ++i)
    {
        if(strcmp(out_names[i], "lr") == 0)
            *lr = out_values[i];
        else if(strcmp(out_names[i], "lc") == 0)
            *lc = out_values[i];
    }
    return *lr != nullptr && *lc != nullptr;
}
