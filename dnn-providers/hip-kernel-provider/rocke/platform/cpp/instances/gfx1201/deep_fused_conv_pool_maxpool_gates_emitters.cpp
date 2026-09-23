// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1201_deep_fused_conv_pool_maxpool_gates_emitters.c
 *
 * INTEGRATION NOTE (no symbols of its own).
 *   The maxpool register-residency gate (WMMA) and the two output-writeback
 *   emitter paths the gfx1201 (RDNA4, wave32, WMMA 16x16x16) epilogue selects
 *   are FAMILY-AGNOSTIC: their numeric core is driven entirely through the
 *   resolved MMA `op` (and the spec/grid), so the gfx1201 shim reuses the
 *   common ports verbatim rather than re-implementing them. The common
 *   definitions live in the family-agnostic translation units
 *     instance_deep_fused_conv_pool_maxpool_gates_and_emitters.c
 *       -> rocke_dfcp_maxpool_is_intra_lane_wmma
 *          rocke_dfcp_emit_inline_maxpool_from_cshuffle
 *          rocke_dfcp_emit_wmma_maxpool_from_registers
 *     instance_deep_fused_conv_pool_epilogue_predicates_and_loaders.c
 *       -> rocke_dfcp_apply_epilogue_scalar
 *   and are declared in
 *     rocke/helper_rocke.instances.common.deep_fused_conv_pool.h.
 *
 *   The gfx1201 epilogue driver (rocke_gfx1201_dfcp_epilogue_override) calls those
 *   common entries directly; pre-deriving ctx->use_wmma_register_maxpool via the
 *   common gate keeps the gfx1201 path byte-identical to the common one with
 *   only the WMMA geometry pinning added. Re-defining the same symbols here
 *   would produce duplicate-definition link errors against the common TUs, so
 *   this part-file intentionally contributes NO symbols. It is kept as a
 *   placeholder translation unit (the include below documents the reuse
 *   contract and keeps the file a valid, non-empty C99 unit).
 */
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"

/* This translation unit intentionally defines no symbols -- see the header
 * comment. The typedef below keeps it a strictly-conforming, non-empty C99 TU
 * without emitting any external symbol. */
typedef int rocke_gfx1201_dfcp_maxpool_gates_emitters_translation_unit_marker;
