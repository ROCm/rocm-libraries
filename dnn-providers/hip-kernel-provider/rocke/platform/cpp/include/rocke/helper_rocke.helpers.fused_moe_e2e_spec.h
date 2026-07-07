/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.fused_moe_e2e_spec.h -- C99 port of the spec /
 * tile-policy surface of
 * rocke/instances/common/fused_moe_e2e.py (the module the task references as
 * rocke.helpers.fused_moe_e2e_spec; the real symbols live in
 * instances/common/fused_moe_e2e.py).
 *
 * SCOPE. This header ports ONLY the value-type / pure-function surface the task
 * enumerates -- the GEMM tile factories, the launch-arch / dtype helpers, and
 * the FusedMoeForwardSpec dataclass with the spec-derivation methods that have a
 * C counterpart. The FusedMoeForward driver class (HIP launch orchestration,
 * torch workspaces, graph capture) is NOT ported here; it has no C analogue in
 * this codegen-only library.
 *
 *   Python (fused_moe_e2e.py)               C99 (this header)
 *   -------------------------------------   ------------------------------------
 *   _default_gemm_tile()                    rocke_fmoe_default_gemm_tile()
 *   _default_bf16_gemm_tile()               rocke_fmoe_default_bf16_gemm_tile()
 *   _default_gemm_tile_gfx942()             rocke_fmoe_default_gemm_tile_gfx942()
 *   _default_bf16_gemm_tile_gfx942()        rocke_fmoe_default_bf16_gemm_tile_gfx942()
 *   _large_batch_gemm_tile()                rocke_fmoe_large_batch_gemm_tile()
 *   _sparse_batch_gemm_tile()               rocke_fmoe_sparse_batch_gemm_tile()
 *   _sparse_batch_gemm_tile_gfx942()        rocke_fmoe_sparse_batch_gemm_tile_gfx942()
 *   _resolve_launch_arch(arch)              rocke_fmoe_resolve_launch_arch(arch)
 *   _gemm_dtype_to_universal(dtype)         rocke_fmoe_gemm_dtype_to_universal(dtype)
 *   _ensure_2byte_dtype(dtype)              rocke_fmoe_ensure_2byte_dtype(dtype)
 *   class FusedMoeForwardSpec               rocke_fmoe_forward_spec_t
 *     .total_pairs                          rocke_fmoe_forward_spec_total_pairs()
 *     .to_topk_softmax_spec()               rocke_fmoe_forward_spec_to_topk_softmax_spec()
 *     .to_gemm_spec(name_suffix=)           rocke_fmoe_forward_spec_to_gemm_spec()
 *     .to_batched_gemm_spec()               rocke_fmoe_forward_spec_to_batched_gemm_spec()
 *     .to_batched_gemm_spec_preshuffle_b()
 * rocke_fmoe_forward_spec_to_batched_gemm_spec_preshuffle_b()
 *
 * Tiles are returned by value as rocke_gemm_tile_spec_t (the gemm_universal C
 * type). The factories are byte-faithful: every TileSpec(...) keyword argument
 * is copied through, and the gemm_universal defaults (warp_k=1) fill the fields
 * the Python factory omits.
 *
 * Error model mirrors the rest of the C port: pure helpers return a sentinel
 * (NULL / -1) on the Python ValueError path; the rocke_b_* builder variants route
 * the same failure through the sticky-error IRBuilder with the Python message.
 */
#ifndef ROCKE_HELPER_FUSED_MOE_E2E_SPEC_H
#define ROCKE_HELPER_FUSED_MOE_E2E_SPEC_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_batched_gemm.h" /* rocke_batched_gemm_spec_t          */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_tile_spec_t / trait     */
#include "rocke/instance_grouped_gemm.h" /* rocke_grouped_gemm_spec_t          */
#include "rocke/instance_topk_softmax.h" /* rocke_topk_softmax_spec_t          */
#include "rocke/ir.h" /* rocke_status_t, rocke_ir_builder_t   */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------- GEMM tile policy *
 *
 * Each returns a fully-populated rocke_gemm_tile_spec_t (warp_k defaults to 1 as
 * in the gemm_universal TileSpec). The values are the measured-winner tiles
 * documented in the Python docstrings and must stay byte-identical so the
 * downstream MFMA atom selection is unchanged. */

/* _default_gemm_tile(): f16 decode/small default (gfx950 wide 32x32x16 atom).
 * tile=(32,128,64) warp=(1,2) warp_tile=(32,32,16). */
rocke_gemm_tile_spec_t rocke_fmoe_default_gemm_tile(void);

/* _default_bf16_gemm_tile(): bf16 default (gfx950 wide 16x16x32 atom).
 * tile=(32,32,32) warp=(2,2) warp_tile=(16,16,32). */
rocke_gemm_tile_spec_t rocke_fmoe_default_bf16_gemm_tile(void);

/* _default_gemm_tile_gfx942(): f16 default for gfx942 (narrow 32x32x8 atom).
 * tile=(32,128,64) warp=(1,2) warp_tile=(32,32,8). */
rocke_gemm_tile_spec_t rocke_fmoe_default_gemm_tile_gfx942(void);

/* _default_bf16_gemm_tile_gfx942(): bf16 default for gfx942 (16x16x16 atom).
 * tile=(32,32,32) warp=(2,2) warp_tile=(16,16,16). */
rocke_gemm_tile_spec_t rocke_fmoe_default_bf16_gemm_tile_gfx942(void);

/* _large_batch_gemm_tile(): large-hidden + dense-routing winner.
 * tile=(64,128,64) warp=(2,2) warp_tile=(32,32,16). */
rocke_gemm_tile_spec_t rocke_fmoe_large_batch_gemm_tile(void);

/* _sparse_batch_gemm_tile(): sparse-routing large-hidden winner (gfx950).
 * tile=(32,128,128) warp=(1,2) warp_tile=(32,32,16). */
rocke_gemm_tile_spec_t rocke_fmoe_sparse_batch_gemm_tile(void);

/* _sparse_batch_gemm_tile_gfx942(): gfx942 variant (narrow 32x32x8 atom).
 * tile=(32,128,128) warp=(1,2) warp_tile=(32,32,8). */
rocke_gemm_tile_spec_t rocke_fmoe_sparse_batch_gemm_tile_gfx942(void);

/* tile equality (TileSpec.__eq__): true iff every field matches. Used by the
 * arch tile-swap policy in FusedMoeForward.__init__ (Python `spec.gemm_tile ==
 * _default_gemm_tile()`); exposed so a C driver can reproduce that policy. */
bool rocke_fmoe_tile_eq(const rocke_gemm_tile_spec_t* a, const rocke_gemm_tile_spec_t* b);

/* ------------------------------------------------------------ arch / dtype *
 *
 * _resolve_launch_arch(arch): explicit `arch` wins; otherwise the Python probes
 * the running HIP device and falls back to "gfx950" when none is visible. This
 * codegen-only library has no HIP runtime, so the probe arm is a TODO(port)
 * stub: a NULL `arch` resolves directly to "gfx950" (the Python no-device
 * fallback). Returns a static string (never freed). */
const char* rocke_fmoe_resolve_launch_arch(const char* arch);

/* _gemm_dtype_to_universal(dtype): "f16"/"fp16" -> "fp16"; "bf16" -> "bf16";
 * anything else -> NULL (Python ValueError). Returns a static string. */
const char* rocke_fmoe_gemm_dtype_to_universal(const char* dtype);

/* Builder variant: NULL on the unsupported-dtype path, with the Python
 * ValueError message recorded on `b` (ROCKE_ERR_VALUE). A dead builder is a
 * NULL no-op. */
const char* rocke_b_fmoe_gemm_dtype_to_universal(rocke_ir_builder_t* b, const char* dtype);

/* _ensure_2byte_dtype(dtype): element size in bytes for the activation dtype.
 * Returns 2 for "f16"/"fp16"/"bf16"; -1 on any other dtype (Python
 * ValueError). */
int rocke_fmoe_ensure_2byte_dtype(const char* dtype);

/* Builder variant: -1 on the unsupported-dtype path, with the Python
 * ValueError message recorded on `b` (ROCKE_ERR_VALUE). */
int rocke_b_fmoe_ensure_2byte_dtype(rocke_ir_builder_t* b, const char* dtype);

/* ---------------------------------------------------------- FusedMoeForwardSpec
 *
 * Mirror of the Python dataclass. `gemm_tile` defaults to
 * rocke_fmoe_default_gemm_tile() (Python field(default_factory=_default_gemm_tile)).
 * `arch` is Optional[str]: NULL == Python None. Every bool/int knob carries its
 * Python default; rocke_fmoe_forward_spec_default() materialises them. */
typedef struct rocke_fmoe_forward_spec
{
    /* required shape */
    int tokens;
    int experts;
    int topk;
    int hidden;
    int intermediate;
    const char* dtype; /* default "f16" */

    /* streaming / sort / router block sizes */
    int streaming_block_size; /* default 256 */
    int streaming_vec; /* default 8   */
    int sort_block_size; /* default 64  */
    int router_block_size; /* default 64  */

    rocke_gemm_tile_spec_t gemm_tile; /* default _default_gemm_tile() */
    const char* arch; /* Optional[str]: NULL == None  */
    const char* name; /* default "rocke_fused_moe_forward" */

    bool use_experimental_fused_gate_up_silu; /* default false */
    bool use_experimental_interleaved_gate_up_silu; /* default true  */
    bool use_experimental_fused_down_reduce; /* default true  */
    bool use_experimental_static_scatter_gather; /* default false */
    bool preshuffle_w_down; /* default false */
    bool preshuffle_w_gate_up_packed; /* default false */
    bool preshuffle_w_gate_up_interleaved; /* default false */
    bool use_grouped_gemm; /* default true  */
    bool active_tile_skip_gemms; /* default true  */
} rocke_fmoe_forward_spec_t;

/* Default-constructed spec: every field at the Python dataclass default
 * (gemm_tile == _default_gemm_tile()). The caller must still set the required
 * shape fields (tokens / experts / topk / hidden / intermediate). */
rocke_fmoe_forward_spec_t rocke_fmoe_forward_spec_default(void);

/* FusedMoeForwardSpec.total_pairs == tokens * topk. */
int rocke_fmoe_forward_spec_total_pairs(const rocke_fmoe_forward_spec_t* spec);

/* FusedMoeForwardSpec.to_topk_softmax_spec(): TopkSoftmaxSpec(
 *   n_per_row=experts, k=topk, dtype="f32", out_dtype="f32",
 *   block_size=router_block_size). Returned by value. */
rocke_topk_softmax_spec_t
    rocke_fmoe_forward_spec_to_topk_softmax_spec(const rocke_fmoe_forward_spec_t* spec);

/* FusedMoeForwardSpec.to_gemm_spec(name_suffix): GroupedGemmSpec(
 *   name=f"{name}_{name_suffix}", tile=gemm_tile, trait=TraitSpec(pad_m=True),
 *   dtype=_gemm_dtype_to_universal(dtype)).
 *
 * The composed name string is written into `name_out` (capacity name_cap); the
 * returned spec's `name` points at `name_out`, so `name_out` must outlive the
 * returned spec. Returns ROCKE_OK, ROCKE_ERR_VALUE (name buffer too small / NULL
 * args / unsupported dtype). On error *out_spec is untouched. */
rocke_status_t rocke_fmoe_forward_spec_to_gemm_spec(const rocke_fmoe_forward_spec_t* spec,
                                                    const char* name_suffix,
                                                    char* name_out,
                                                    size_t name_cap,
                                                    rocke_grouped_gemm_spec_t* out_spec);

/* FusedMoeForwardSpec.to_batched_gemm_spec(): BatchedGemmSpec(
 *   name=f"{name}_batched_gemm", tile=gemm_tile,
 *   trait=TraitSpec(pad_m=True, pad_n=True),
 *   dtype=_gemm_dtype_to_universal(dtype)).
 *
 * The "<name>_batched_gemm" string is written into `name_out` (capacity
 * name_cap); the returned spec's `name` points at it. Returns ROCKE_OK or
 * ROCKE_ERR_VALUE. */
rocke_status_t rocke_fmoe_forward_spec_to_batched_gemm_spec(const rocke_fmoe_forward_spec_t* spec,
                                                            char* name_out,
                                                            size_t name_cap,
                                                            rocke_batched_gemm_spec_t* out_spec);

/* FusedMoeForwardSpec.to_batched_gemm_spec_preshuffle_b(): identical to
 * to_batched_gemm_spec() but with trait.preshuffle_b=True. */
rocke_status_t
    rocke_fmoe_forward_spec_to_batched_gemm_spec_preshuffle_b(const rocke_fmoe_forward_spec_t* spec,
                                                              char* name_out,
                                                              size_t name_cap,
                                                              rocke_batched_gemm_spec_t* out_spec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_FUSED_MOE_E2E_SPEC_H */
