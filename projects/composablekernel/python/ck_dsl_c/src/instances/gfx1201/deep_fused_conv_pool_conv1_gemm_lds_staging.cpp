// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_conv1_gemm_lds_staging.c
 *
 * INTEGRATION NOTE (no symbols of its own).
 *   The conv1-1x1-GEMM + two-disjoint-LDS-producer slice of the gfx1201 (RDNA4,
 *   wave32, WMMA 16x16x16) deep-fusion epilogue is NOT a standalone closure --
 *   in the Python builder it is the front half of the single epilogue_override
 *   closure (ck_dsl/instances/common/deep_fused_conv_pool.py:1334-1375). The C
 *   port keeps that closure as one translation unit:
 *     instance_gfx1201_deep_fused_conv_pool_epilogue_override_driver.c
 *       -> ckc_gfx1201_dfcp_epilogue_override
 *   which emits, in Python order, the conv0 cshuffle-LDS stage producer
 *   (ckc_dfcp_stage_accumulators_to_cshuffle_lds, sync=False), the W1 weight LDS
 *   load producer (ckc_dfcp_load_conv1_weights_to_lds, sync=False), the single
 *   merged barrier (ckc_b_sync), the deferred-epilogue decision, and the conv1
 *   1x1 GEMM (ckc_dfcp_emit_conv1_1x1) before the maxpool routing tail.
 *
 *   The three numeric producers are the family-agnostic op-driven common helpers
 *   (ckc_dfcp_*) declared in
 *   ckc/helper_ck_dsl.instances.common.deep_fused_conv_pool.h and authored once
 *   in the common TUs; the gfx1201 shim reuses them verbatim driven by the
 *   WMMA-resolved op. There are therefore no gfx1201-specific symbols for this
 *   slice to define. Defining ckc_gfx1201_dfcp_epilogue_override here as well
 *   would produce a duplicate-definition link error against the driver TU, so
 *   this part-file intentionally contributes NO symbols and is kept as a
 *   documentation placeholder.
 */
#include "ckc/instance_gfx1201_deep_fused_conv_pool_internal.h"

/* This translation unit intentionally defines no symbols -- see the header
 * comment. The typedef keeps it a strictly-conforming, non-empty C99 TU without
 * emitting any external symbol. */
typedef int ckc_gfx1201_dfcp_conv1_gemm_lds_staging_translation_unit_marker;
