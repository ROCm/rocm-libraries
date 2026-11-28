// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of Img2ColSpec from rocke/instances/common/img2col.py.
 * See rocke/helper_rocke.instances.common.img2col.h for the symbol map.
 */
#include "rocke/helper_rocke.instances.common.img2col.h"

#include <stdio.h> /* snprintf */

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */

/* ===================================================================== *
 *  Img2ColSpec
 * ===================================================================== */

rocke_img2col_spec_t rocke_img2col_spec_default(void)
{
    rocke_img2col_spec_t spec;
    spec.problem = rocke_conv_problem_default(0, 0, 0, 0, 0, 0, 0);
    spec.dtype = "f16";
    spec.block_tile_m = 8;
    spec.block_tile_k = 128;
    spec.vec_k = 8;
    spec.name = "rocke_img2col";
    return spec;
}

/* block_size = (block_tile_m * block_tile_k) // vec_k */
int rocke_img2col_block_size(const rocke_img2col_spec_t* spec)
{
    return (spec->block_tile_m * spec->block_tile_k) / spec->vec_k;
}

/* can_vector_load = vec_k > 1 and (problem.C % vec_k) == 0 */
bool rocke_img2col_can_vector_load(const rocke_img2col_spec_t* spec)
{
    return spec->vec_k > 1 && (spec->problem.C % spec->vec_k) == 0;
}

/* kernel_name():
 *   kernel_name_join(self.name, self.problem.short(), self.dtype,
 *                    f"t{block_tile_m}x{block_tile_k}", f"v{vec_k}") */
rocke_status_t
    rocke_img2col_kernel_name(const rocke_img2col_spec_t* spec, char* out, size_t out_cap)
{
    char short_part[128];
    char tile_part[64];
    char vec_part[32];
    const char* parts[4];
    rocke_status_t st;
    int wrote;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_conv_problem_short(&spec->problem, short_part, sizeof(short_part), NULL);
    if(st != ROCKE_OK)
    {
        return st;
    }

    wrote
        = snprintf(tile_part, sizeof(tile_part), "t%dx%d", spec->block_tile_m, spec->block_tile_k);
    if(wrote < 0 || (size_t)wrote >= sizeof(tile_part))
    {
        return ROCKE_ERR_VALUE;
    }

    wrote = snprintf(vec_part, sizeof(vec_part), "v%d", spec->vec_k);
    if(wrote < 0 || (size_t)wrote >= sizeof(vec_part))
    {
        return ROCKE_ERR_VALUE;
    }

    parts[0] = short_part;
    parts[1] = spec->dtype;
    parts[2] = tile_part;
    parts[3] = vec_part;

    /* kernel_name_join(prefix=self.name, *parts) -- no flags. */
    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}
