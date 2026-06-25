/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common._matmul_nbits_large_n.h -- C99 port of
 * two symbols from ck_dsl/instances/common/_matmul_nbits_large_n.py:
 *
 *   Python                  C99 (this header)
 *   ---------------------   ----------------------------------------
 *   _WmmaParams             ckc_wmma_params_t
 *   _wmma_params(arch)      ckc_wmma_params(arch, ...)
 *
 * Both are pure, builder-free value producers: _WmmaParams is a frozen
 * dataclass (a 3-field record) and _wmma_params is a per-arch lookup that
 * returns one of two constant records. Neither touches the IR builder; their
 * fields are later consumed to drive a byte-identical builder-call sequence
 * (the wmma_op method name, the fragment K width, and the lane-half K split),
 * so byte-identical field values give a byte-identical downstream IR.
 *
 * _WmmaParams fields (see the Python docstring for the full ABI rationale):
 *
 *   wmma_op          IRBuilder method name for the WMMA atom.
 *                    gfx1201 -> "wmma_gfx12_f32_16x16x16_f16"
 *                    else     -> "wmma_f32_16x16x16_f16"
 *   frag_k           fp16 elements per lane in one A/B fragment (16 | 8).
 *   split_k_by_half  gfx12: K offset within a step = (lane//16)*frag_k.
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_H

#include "ckc/ir.h" /* ckc_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* _WmmaParams: per-arch WMMA 16x16x16 f16 fragment ABI.
 *
 * wmma_op is a pointer to a static string literal (never freed, NUL-terminated)
 * matching the IRBuilder method name to invoke via getattr(b, wp.wmma_op).
 * split_k_by_half is the Python bool, modelled as 0/1. */
typedef struct ckc_wmma_params_t
{
    const char* wmma_op; /* IRBuilder method name for the WMMA atom         */
    int frag_k; /* fp16 elements per lane in one A/B fragment (16|8)*/
    int split_k_by_half; /* gfx12: K step offset = (lane//16)*frag_k (0|1)  */
} ckc_wmma_params_t;

/* _wmma_params(arch):
 *
 *   if arch == "gfx1201":
 *       return _WmmaParams("wmma_gfx12_f32_16x16x16_f16", 8, True)
 *   return     _WmmaParams("wmma_f32_16x16x16_f16",       16, False)
 *
 * `arch` is the NUL-terminated arch string. The selected record is written to
 * *out (which must be non-NULL). A NULL arch is treated as a non-"gfx1201"
 * arch (i.e. takes the default gfx11 branch), matching the Python `==` compare
 * never being true for a non-string. Returns CKC_OK on success, or
 * CKC_ERR_VALUE when out == NULL. */
ckc_status_t ckc_wmma_params(const char* arch, ckc_wmma_params_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON_MATMUL_NBITS_LARGE_N_H */
