// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Exercise public catalog queries against stored keys, without normalizing
 * away malformed table entries. Alias queries must find the same atom. */
#include "rocke/arch_target.h"

#include <initializer_list>
#include <stdio.h>
#include <string.h>

static const char* short_dtype(const char* dtype)
{
    const char* aliases[][2] = {{"fp8e4m3", "fp8"},
                                {"bf8e5m2", "bf8"},
                                {"fp6e2m3", "fp6"},
                                {"fp6e3m2", "bf6"},
                                {"fp4e2m1", "fp4"}};
    for(const auto& pair : aliases)
    {
        if(strcmp(dtype, pair[0]) == 0)
            return pair[1];
    }
    return dtype;
}

int main()
{
    int checked = 0;
    for(const char* gfx : {"gfx950", "gfx1250"})
    {
        const rocke_arch_target_t* arch = rocke_arch_target_from_gfx(gfx);
        if(!arch)
            return 1;
        for(int i = 0; i < arch->mma.num_ops; ++i)
        {
            const rocke_mma_op_t* op = &arch->mma.ops[i];
            for(const char* dtype : {op->a_dtype, op->b_dtype, op->c_dtype})
            {
                char scratch[64];
                if(strcmp(dtype, rocke_normalize_dtype(dtype, scratch, sizeof(scratch))) != 0)
                {
                    fprintf(stderr, "%s: noncanonical stored dtype %s\n", op->op_id, dtype);
                    return 1;
                }
            }
            for(const char* a : {op->a_dtype, short_dtype(op->a_dtype)})
            {
                for(const char* b : {op->b_dtype, short_dtype(op->b_dtype)})
                {
                    if(!rocke_mma_catalog_has_shape(
                           &arch->mma, op->family, a, b, op->c_dtype, op->m, op->n, op->k)
                       || rocke_mma_catalog_op_for_shape(
                              &arch->mma, op->family, a, b, op->c_dtype, op->m, op->n, op->k)
                              != op
                       || rocke_mma_catalog_select_largest_k(
                              &arch->mma, op->family, a, b, op->c_dtype, op->m, op->n, op->k)
                              != op
                       || rocke_mma_catalog_enumerate(
                              &arch->mma, op->family, a, b, op->c_dtype, op->m, op->n, NULL, 0)
                              < 1)
                    {
                        fprintf(stderr, "%s: catalog query missed %s/%s\n", op->op_id, a, b);
                        return 1;
                    }
                }
            }
            ++checked;
        }
    }
    printf("catalog lookup: %d rows passed canonical and alias queries\n", checked);
    return 0;
}
