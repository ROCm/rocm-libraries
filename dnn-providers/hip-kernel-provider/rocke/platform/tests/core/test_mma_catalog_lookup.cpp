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
            // K=64 low-bit WMMA packs 32 bytes into eight i32 registers per source.
            if(strcmp(gfx, "gfx1250") == 0 && op->k == 64 && strcmp(op->family, "wmma") == 0
               && (op->srcs[0].frag_len != 8 || op->srcs[1].frag_len != 8))
            {
                fprintf(stderr, "%s: fragment length must count packed i32 elements\n", op->op_id);
                return 1;
            }
            for(const char* dtype :
                {op->srcs[0].dtype, op->srcs[1].dtype, op->srcs[2].dtype, op->dst.dtype})
            {
                char scratch[64];
                if(strcmp(dtype, rocke_normalize_dtype(dtype, scratch, sizeof(scratch))) != 0)
                {
                    fprintf(stderr, "%s: noncanonical stored dtype %s\n", op->op_id, dtype);
                    return 1;
                }
            }
            for(const char* a : {op->srcs[0].dtype, short_dtype(op->srcs[0].dtype)})
            {
                for(const char* b : {op->srcs[1].dtype, short_dtype(op->srcs[1].dtype)})
                {
                    const char* src_dtypes[3] = {a, b, op->srcs[2].dtype};
                    rocke_mma_scale_operand_t src_scales[3];
                    for(int j = 0; j < 3; ++j)
                        src_scales[j] = {op->srcs[j].scale_dtype, op->srcs[j].scale_block_size};
                    if(!rocke_mma_catalog_has_shape_indexed(&arch->mma,
                                                            op->family,
                                                            src_dtypes,
                                                            op->dst.dtype,
                                                            src_scales,
                                                            op->m,
                                                            op->n,
                                                            op->k)
                       || rocke_mma_catalog_op_for_shape_indexed(&arch->mma,
                                                                 op->family,
                                                                 src_dtypes,
                                                                 op->dst.dtype,
                                                                 src_scales,
                                                                 op->m,
                                                                 op->n,
                                                                 op->k)
                              != op
                       || rocke_mma_catalog_select_largest_k_indexed(&arch->mma,
                                                                     op->family,
                                                                     src_dtypes,
                                                                     op->dst.dtype,
                                                                     src_scales,
                                                                     op->m,
                                                                     op->n,
                                                                     op->k)
                              != op
                       || rocke_mma_catalog_enumerate_indexed(&arch->mma,
                                                              op->family,
                                                              src_dtypes,
                                                              op->dst.dtype,
                                                              src_scales,
                                                              op->m,
                                                              op->n,
                                                              NULL,
                                                              0)
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
