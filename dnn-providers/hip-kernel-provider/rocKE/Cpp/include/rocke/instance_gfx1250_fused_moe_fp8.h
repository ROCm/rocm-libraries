/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_fused_moe_fp8.h -- C99 port of the gfx1250 FP8/BF8
 * fused-MoE forward driver spec rocke/instances/gfx1250/fused_moe_fp8.py.
 *
 * NOTE: fused_moe_fp8 is a HOST-SIDE DRIVER that composes already-ported
 * component kernels (smoothquant, block_scaled_gemm, fused_moe gather/silu/
 * topk_weighted_reduce). It does NOT have its own single-kernel build function
 * that emits IR. Therefore there is NO build entry point and NO parity emitter.
 *
 * This header exposes the Gfx1250Fp8MoeSpec C struct mirror and the host-side
 * helper functions (_round_up, _bs_for, _sq_bsvec, etc.) used to compute the
 * sub-kernel specs, so the host orchestrator can compose the existing C-ported
 * component kernels identically to the Python driver.
 *
 *   Python (fused_moe_fp8.py)               C99 (this header)
 *   ------------------------------------    ------------------------------------------
 *   class Gfx1250Fp8MoeSpec                 rocke_gfx1250_fp8_moe_spec_t
 *   Gfx1250Fp8MoeSpec.slot_size             rocke_gfx1250_fp8_moe_slot_size()
 *   Gfx1250Fp8MoeSpec.rows                  rocke_gfx1250_fp8_moe_rows()
 *   _round_up / _bs_for / _bs_common        rocke_fp8_moe_round_up / _bs_for / _bs_common
 *   _sq_bsvec                               rocke_fp8_moe_sq_bsvec
 *   _block_k_for                            rocke_fp8_moe_block_k_for
 */
#ifndef ROCKE_INSTANCE_GFX1250_FUSED_MOE_FP8_H
#define ROCKE_INSTANCE_GFX1250_FUSED_MOE_FP8_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python Gfx1250Fp8MoeSpec (frozen dataclass). */
typedef struct rocke_gfx1250_fp8_moe_spec
{
    int tokens;
    int experts;
    int topk;
    int hidden;
    int intermediate;
    const char* lowbit; /* "fp8e4m3" | "bf8e5m2" (default "fp8e4m3") */
    const char* dtype; /* "bf16" (default) */
    const char* name; /* default "rocke_gfx1250_fp8_moe" */
} rocke_gfx1250_fp8_moe_spec_t;

/* Default-constructed spec. */
rocke_gfx1250_fp8_moe_spec_t rocke_gfx1250_fp8_moe_spec_default(void);

/* slot_size = round_up(tokens * topk, 16). */
int rocke_gfx1250_fp8_moe_slot_size(const rocke_gfx1250_fp8_moe_spec_t* spec);

/* rows = experts * slot_size. */
int rocke_gfx1250_fp8_moe_rows(const rocke_gfx1250_fp8_moe_spec_t* spec);

/* Host-side helper: round_up(x, m). */
int rocke_fp8_moe_round_up(int x, int m);

/* Host-side helper: best block_size from {256,128,64} that divides dim; else 16 or 1. */
int rocke_fp8_moe_bs_for(int dim);

/* Host-side helper: best common block_size for gate/up/silu (divides both H and I). */
int rocke_fp8_moe_bs_common(int dim1, int dim2);

/* Host-side helper: pick (block_size, vec) valid for smoothquant. Returns 0 on
 * success, -1 if no valid pair exists. */
int rocke_fp8_moe_sq_bsvec(int dim, int* out_bs, int* out_vec);

/* Host-side helper: block_k for FP8 GEMM (128 if K%128==0 else 64). */
int rocke_fp8_moe_block_k_for(int k);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_FUSED_MOE_FP8_H */
