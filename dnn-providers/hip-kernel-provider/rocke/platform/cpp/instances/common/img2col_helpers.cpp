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
 *  ConvProblem (minimal subset)
 * ===================================================================== */

rocke_conv_problem_t rocke_img2col_conv_problem_default(void)
{
    rocke_conv_problem_t p;
    p.N = 0;
    p.Hi = 0;
    p.Wi = 0;
    p.C = 0;
    p.K = 0;
    p.Y = 0;
    p.X = 0;
    p.sH = 1;
    p.sW = 1;
    p.pH = 0;
    p.pW = 0;
    p.dH = 1;
    p.dW = 1;
    p.is_3d = false;
    p.Di = 0;
    p.Z = 0;
    p.sD = 0;
    p.pD = 0;
    p.dD = 0;
    return p;
}

/* Ho = (Hi + 2*pH - dH*(Y-1) - 1)//sH + 1 */
int rocke_img2col_conv_problem_ho(const rocke_conv_problem_t* p)
{
    return (p->Hi + 2 * p->pH - p->dH * (p->Y - 1) - 1) / p->sH + 1;
}

/* Wo = (Wi + 2*pW - dW*(X-1) - 1)//sW + 1 */
int rocke_img2col_conv_problem_wo(const rocke_conv_problem_t* p)
{
    return (p->Wi + 2 * p->pW - p->dW * (p->X - 1) - 1) / p->sW + 1;
}

/* M = N * Ho * Wo */
int rocke_img2col_conv_problem_m(const rocke_conv_problem_t* p)
{
    return p->N * rocke_img2col_conv_problem_ho(p) * rocke_img2col_conv_problem_wo(p);
}

/* K_gemm = Y * X * C */
int rocke_img2col_conv_problem_k_gemm(const rocke_conv_problem_t* p)
{
    return p->Y * p->X * p->C;
}

/* short(): "N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}" */
rocke_status_t
    rocke_img2col_conv_problem_short(const rocke_conv_problem_t* p, char* out, size_t out_cap)
{
    int wrote;

    if(p == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    wrote = snprintf(
        out, out_cap, "N%dH%dW%dC%d_K%dY%dX%d", p->N, p->Hi, p->Wi, p->C, p->K, p->Y, p->X);
    if(wrote < 0 || (size_t)wrote >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  Img2ColSpec
 * ===================================================================== */

rocke_img2col_spec_t rocke_img2col_spec_default(void)
{
    rocke_img2col_spec_t spec;
    spec.problem = rocke_img2col_conv_problem_default();
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

    st = rocke_img2col_conv_problem_short(&spec->problem, short_part, sizeof(short_part));
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
