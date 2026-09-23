// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of two symbols from
 * rocke/instances/common/_matmul_nbits_large_n.py: _WmmaParams and
 * _wmma_params. See the header for the original Python and the contract.
 *
 * These are builder-free value producers. The goal is byte-identical field
 * values (the wmma_op method name, frag_k, split_k_by_half) so the downstream
 * builder-call sequence in build_large_n_matmul_nbits is byte-identical to the
 * Python.
 */
#include "rocke/helper_rocke.instances.common._matmul_nbits_large_n.h"

#include <string.h>

rocke_status_t rocke_wmma_params(const char* arch, rocke_wmma_params_t* out)
{
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Python:
     *   if arch == "gfx1201":
     *       return _WmmaParams(
     *           wmma_op="wmma_gfx12_f32_16x16x16_f16",
     *           frag_k=8,
     *           split_k_by_half=True,
     *       )
     *   return _WmmaParams(
     *       wmma_op="wmma_f32_16x16x16_f16",
     *       frag_k=16,
     *       split_k_by_half=False,
     *   )
     *
     * A NULL arch can never equal the literal, so it takes the default branch,
     * matching Python's `==` against a non-string. */
    if(arch != NULL && strcmp(arch, "gfx1201") == 0)
    {
        out->wmma_op = "wmma_gfx12_f32_16x16x16_f16";
        out->frag_k = 8;
        out->split_k_by_half = 1;
        return ROCKE_OK;
    }

    out->wmma_op = "wmma_f32_16x16x16_f16";
    out->frag_k = 16;
    out->split_k_by_half = 0;
    return ROCKE_OK;
}
