// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_fused_moe_fp8.c -- C99 port of the HOST-SIDE helpers from
 * rocke/instances/gfx1250/fused_moe_fp8.py.
 *
 * This file ports ONLY the spec + host-side geometry helpers (_round_up,
 * _bs_for, _bs_common, _sq_bsvec, _block_k_for). The Gfx1250Fp8Moe class
 * is a HOST DRIVER that composes already-ported component kernels; it does
 * NOT have a single-kernel build function that emits IR.
 */

#include <string.h>

#include "rocke/instance_gfx1250_fused_moe_fp8.h"

rocke_gfx1250_fp8_moe_spec_t rocke_gfx1250_fp8_moe_spec_default(void)
{
    rocke_gfx1250_fp8_moe_spec_t s;
    memset(&s, 0, sizeof(s));
    s.tokens = 0;
    s.experts = 0;
    s.topk = 0;
    s.hidden = 0;
    s.intermediate = 0;
    s.lowbit = "fp8e4m3";
    s.dtype = "bf16";
    s.name = "rocke_gfx1250_fp8_moe";
    return s;
}

int rocke_fp8_moe_round_up(int x, int m)
{
    if(m <= 0)
    {
        return x;
    }
    return ((x + m - 1) / m) * m;
}

int rocke_gfx1250_fp8_moe_slot_size(const rocke_gfx1250_fp8_moe_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return rocke_fp8_moe_round_up(spec->tokens * spec->topk, 16);
}

int rocke_gfx1250_fp8_moe_rows(const rocke_gfx1250_fp8_moe_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->experts * rocke_gfx1250_fp8_moe_slot_size(spec);
}

int rocke_fp8_moe_bs_for(int dim)
{
    int bs;
    int candidates[] = {256, 128, 64};
    int i;
    for(i = 0; i < 3; ++i)
    {
        bs = candidates[i];
        if(dim % bs == 0)
        {
            return bs;
        }
    }
    return (dim % 16 == 0) ? 16 : 1;
}

int rocke_fp8_moe_bs_common(int dim1, int dim2)
{
    int candidates[] = {256, 128, 64};
    int i;
    for(i = 0; i < 3; ++i)
    {
        int bs = candidates[i];
        if(dim1 % bs == 0 && dim2 % bs == 0)
        {
            return bs;
        }
    }
    return (dim1 % 16 == 0 && dim2 % 16 == 0) ? 16 : 1;
}

int rocke_fp8_moe_sq_bsvec(int dim, int* out_bs, int* out_vec)
{
    int bs_candidates[] = {256, 128, 64};
    int vec_candidates[] = {8, 4, 2};
    int bi, vi;
    for(bi = 0; bi < 3; ++bi)
    {
        int bs = bs_candidates[bi];
        if(dim % bs != 0)
        {
            continue;
        }
        int ept = dim / bs;
        for(vi = 0; vi < 3; ++vi)
        {
            int v = vec_candidates[vi];
            if(ept % v == 0 && dim % (bs * v) == 0)
            {
                if(out_bs)
                {
                    *out_bs = bs;
                }
                if(out_vec)
                {
                    *out_vec = v;
                }
                return 0;
            }
        }
    }
    return -1; /* no valid (block_size, vec) found */
}

int rocke_fp8_moe_block_k_for(int k)
{
    return (k % 128 == 0) ? 128 : 64;
}
