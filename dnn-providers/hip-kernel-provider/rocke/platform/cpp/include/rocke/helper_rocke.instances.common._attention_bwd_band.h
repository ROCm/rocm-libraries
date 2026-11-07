/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_band.h -- C port of
 * library/kernels/common/_attention_bwd_band.py: the inverse band, the tile
 * classifier and the mask policies of the attention backward.
 *
 *   * q_tile_range / tile_inside_band / is_edge_tile / tile_class /
 *     interior_q_range / dkv_store_rule  -> rocke_attn_bwd_band_* free functions
 *   * MaskPolicy, RuntimeBand, NoMaskKTail -> rocke_attn_bwd_mask_policy_t, one
 *     struct with a kind tag (the Python class hierarchy has two leaves and no
 *     other subclass in the C port yet)
 *   * TILE_INTERIOR / TILE_EDGE             -> ROCKE_ATTN_BWD_TILE_*
 *
 * HostInt (the duck-typed integer builder that evaluates these formulas on
 * the host) has no C object: q_tile_range / is_edge_tile evaluated on HostInt
 * over a BandEmitter of int bounds are rocke_attn_bwd_host_band_* below (plain
 * int64 arithmetic, sdiv truncating toward zero as HostInt.div), and
 * StaticKvTilePlan / static_kv_tile_plan are rocke_attn_bwd_static_kv_tile_plan_t
 * / rocke_attn_bwd_static_kv_tile_plan. The plan's edge_tiles tuple is kept
 * as its count (n_edge; interior_tiles is n_qt - n_edge): the dense shell
 * only tests whether either is empty.
 *
 * Not ported: MaskPolicy.q_tile_range's optional k_n0 / k_m0 mismatch check
 * (no caller passes them).
 *
 * Errors: a Python ValueError / NotImplementedError records the status and the
 * Python message on the builder; Value-returning entries return NULL and
 * status-returning entries return the code. Builder-call order matches Python
 * exactly (see helper_rocke.helpers.attention_band.h).
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BAND_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BAND_H

#include "rocke/helper_rocke.helpers.attention_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_ATTN_BWD_TILE_INTERIOR 0
#define ROCKE_ATTN_BWD_TILE_EDGE 1

/* seq_mode of dkv_store_rule: rocke_attn_bwd_seq_mode_t ("batched" / "thd"),
 * from helper_rocke.instances.common._attention_bwd_addr.h. */

/* q_tile_range(b, band, k0, *, k_n0, k_m0, uniform) -> (qt_start, n_qt). */
rocke_status_t rocke_attn_bwd_band_q_tile_range(rocke_ir_builder_t* b,
                                                rocke_band_emitter_t* band,
                                                rocke_value_t* k0,
                                                int k_n0,
                                                int k_m0,
                                                bool uniform,
                                                rocke_value_t** out_qt_start,
                                                rocke_value_t** out_n_qt);

/* tile_inside_band(b, band, q_first, q_last, k_first, k_last) -> i1. */
rocke_value_t* rocke_attn_bwd_band_tile_inside_band(rocke_ir_builder_t* b,
                                                    rocke_band_emitter_t* band,
                                                    rocke_value_t* q_first,
                                                    rocke_value_t* q_last,
                                                    rocke_value_t* k_first,
                                                    rocke_value_t* k_last);

/* is_edge_tile(b, band, qt, k0, *, k_m0, k_n0, masked) -> i1. */
rocke_value_t* rocke_attn_bwd_band_is_edge_tile(rocke_ir_builder_t* b,
                                                rocke_band_emitter_t* band,
                                                rocke_value_t* qt,
                                                rocke_value_t* k0,
                                                int k_m0,
                                                int k_n0,
                                                bool masked);

/* tile_class(b, band, qt, k0, *, k_m0, k_n0, masked, uniform) -> i32. */
rocke_value_t* rocke_attn_bwd_band_tile_class(rocke_ir_builder_t* b,
                                              rocke_band_emitter_t* band,
                                              rocke_value_t* qt,
                                              rocke_value_t* k0,
                                              int k_m0,
                                              int k_n0,
                                              bool masked,
                                              bool uniform);

/* interior_q_range(b, band, k0, qt_start, n_qt, *, k_m0, k_n0, masked,
 * uniform) -> (i0, i1). */
rocke_status_t rocke_attn_bwd_band_interior_q_range(rocke_ir_builder_t* b,
                                                    rocke_band_emitter_t* band,
                                                    rocke_value_t* k0,
                                                    rocke_value_t* qt_start,
                                                    rocke_value_t* n_qt,
                                                    int k_m0,
                                                    int k_n0,
                                                    bool masked,
                                                    bool uniform,
                                                    rocke_value_t** out_i0,
                                                    rocke_value_t** out_i1);

/* dkv_store_rule(b, k, len_kv, *, seq_mode, s_kv_max) -> (store, live).
 * s_kv_max may be NULL (Python None); the batched mode then fails. */
rocke_status_t rocke_attn_bwd_band_dkv_store_rule(rocke_ir_builder_t* b,
                                                  rocke_value_t* k,
                                                  rocke_value_t* len_kv,
                                                  rocke_attn_bwd_seq_mode_t seq_mode,
                                                  const rocke_int_or_value_t* s_kv_max,
                                                  rocke_value_t** out_store,
                                                  rocke_value_t** out_live);

/* Mask policies. kind selects RuntimeBand or NoMaskKTail. */
typedef enum rocke_attn_bwd_mask_kind
{
    ROCKE_ATTN_BWD_MASK_RUNTIME_BAND = 0,
    ROCKE_ATTN_BWD_MASK_NO_MASK_KTAIL = 1
} rocke_attn_bwd_mask_kind_t;

/* Do NOT copy a policy once built: Python shares one BandEmitter by reference,
 * here it is embedded by value and caches the I32_FAR constant, so a copy would
 * emit a second constant. Pass it by pointer. */
typedef struct rocke_attn_bwd_mask_policy
{
    rocke_attn_bwd_mask_kind_t kind;
    rocke_band_emitter_t band;
    int k_m0;
    int k_n0;
    bool masked; /* MaskPolicy.masked: true, false for NoMaskKTail */
    rocke_value_t* hi_off; /* RuntimeBand only; NULL: no right limit */
    rocke_value_t* lo_off; /* RuntimeBand only; NULL: no left limit */
} rocke_attn_bwd_mask_policy_t;

/* RuntimeBand(b, bounds, *, k_m0, k_n0). */
rocke_status_t rocke_attn_bwd_runtime_band_init(rocke_attn_bwd_mask_policy_t* self,
                                                rocke_ir_builder_t* b,
                                                const rocke_attn_runtime_bounds_t* bounds,
                                                int k_m0,
                                                int k_n0);

/* NoMaskKTail(b, len_q, len_kv, *, k_m0, k_n0); the lengths are i32 values or
 * (dense kernels) build-time ints. */
rocke_status_t rocke_attn_bwd_no_mask_ktail_init(rocke_attn_bwd_mask_policy_t* self,
                                                 rocke_ir_builder_t* b,
                                                 rocke_int_or_value_t len_q,
                                                 rocke_int_or_value_t len_kv,
                                                 int k_m0,
                                                 int k_n0);

/* MaskPolicy.q_tile_range(b, k0) -> (qt_start, n_qt). */
rocke_status_t rocke_attn_bwd_mask_q_tile_range(rocke_attn_bwd_mask_policy_t* self,
                                                rocke_ir_builder_t* b,
                                                rocke_value_t* k0,
                                                rocke_value_t** out_qt_start,
                                                rocke_value_t** out_n_qt);

/* MaskPolicy.tile_class(b, qt, k0) -> i32. */
rocke_value_t* rocke_attn_bwd_mask_tile_class(rocke_attn_bwd_mask_policy_t* self,
                                              rocke_ir_builder_t* b,
                                              rocke_value_t* qt,
                                              rocke_value_t* k0);

/* MaskPolicy.interior_q_range(b, k0, qt_start, n_qt) -> (i0, i1). */
rocke_status_t rocke_attn_bwd_mask_interior_q_range(rocke_attn_bwd_mask_policy_t* self,
                                                    rocke_ir_builder_t* b,
                                                    rocke_value_t* k0,
                                                    rocke_value_t* qt_start,
                                                    rocke_value_t* n_qt,
                                                    rocke_value_t** out_i0,
                                                    rocke_value_t** out_i1);

/* MaskPolicy.row(b, q) -> RowBand. */
rocke_status_t rocke_attn_bwd_mask_row(rocke_attn_bwd_mask_policy_t* self,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* q,
                                       rocke_row_band_t* out);

/* MaskPolicy.keep(b, row, k): cell keep with the backward row term. */
rocke_value_t* rocke_attn_bwd_mask_keep(rocke_attn_bwd_mask_policy_t* self,
                                        rocke_ir_builder_t* b,
                                        const rocke_row_band_t* row,
                                        rocke_value_t* k);

/* MaskPolicy.cell_keep(b, q, k) (RuntimeBand overrides it with the
 * two-offset form). */
rocke_value_t* rocke_attn_bwd_mask_cell_keep(rocke_attn_bwd_mask_policy_t* self,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* q,
                                             rocke_value_t* k);

/* MaskPolicy.row_limits(b, q, ok) -> (lo, hi). `ok` may be NULL (Python None). */
rocke_status_t rocke_attn_bwd_mask_row_limits(rocke_attn_bwd_mask_policy_t* self,
                                              rocke_ir_builder_t* b,
                                              rocke_value_t* q,
                                              rocke_value_t* ok,
                                              rocke_value_t** out_lo,
                                              rocke_value_t** out_hi);

/* MaskPolicy.keep_in(b, (lo, hi), k): i1 min(max(k, lo), hi) == k. */
rocke_value_t* rocke_attn_bwd_mask_keep_in(rocke_ir_builder_t* b,
                                           rocke_value_t* lo,
                                           rocke_value_t* hi,
                                           rocke_value_t* k);

/* ------------------------------------------------- host-side evaluation */

/* BandEmitter(HostInt(), AttnRuntimeBounds(s_q, s_kv, off, left, right)) on
 * build-time ints: has_off false is a zero offset (None), has_left /
 * has_right false a static unbounded side (None). */
typedef struct rocke_attn_bwd_host_band
{
    int64_t sq;
    int64_t sk;
    int64_t sk_minus1;
    bool has_off;
    int64_t off;
    bool has_left;
    int64_t left;
    bool has_right;
    int64_t right;
} rocke_attn_bwd_host_band_t;

/* AttnRuntimeBounds(...).__post_init__ then BandEmitter(HostInt(), bounds):
 * ROCKE_ERR_VALUE (and err filled when not NULL) for a bound below -1. */
rocke_status_t rocke_attn_bwd_host_band_init(rocke_attn_bwd_host_band_t* out,
                                             int64_t s_q,
                                             int64_t s_kv,
                                             int64_t off,
                                             int64_t left,
                                             int64_t right,
                                             char* err,
                                             size_t err_cap);

/* q_tile_range(HostInt(), band, k0, k_n0=, k_m0=, uniform=False):
 * ROCKE_ERR_VALUE for a tile size below 1. */
rocke_status_t rocke_attn_bwd_host_band_q_tile_range(const rocke_attn_bwd_host_band_t* band,
                                                     int64_t k0,
                                                     int k_n0,
                                                     int k_m0,
                                                     int64_t* out_qt_start,
                                                     int64_t* out_n_qt);

/* is_edge_tile(HostInt(), band, qt, k0, k_m0=, k_n0=, masked=). */
bool rocke_attn_bwd_host_band_is_edge_tile(const rocke_attn_bwd_host_band_t* band,
                                           int64_t qt,
                                           int64_t k0,
                                           int k_m0,
                                           int k_n0,
                                           bool masked);

/* StaticKvTilePlan: the q tiles [qt_start, qt_start + n_qt) key tile k0
 * visits; n_edge of them are edge tiles (edge_tiles, counted), the other
 * n_qt - n_edge interior (interior_tiles). */
typedef struct rocke_attn_bwd_static_kv_tile_plan
{
    int64_t k0;
    int64_t qt_start;
    int64_t n_qt;
    int64_t n_edge;
} rocke_attn_bwd_static_kv_tile_plan_t;

/* len(StaticKvTilePlan.interior_tiles). */
int64_t
    rocke_attn_bwd_static_kv_tile_plan_n_interior(const rocke_attn_bwd_static_kv_tile_plan_t* p);

/* static_kv_tile_plan(s_q=, s_kv=, k0=, k_m0=, k_n0=, left=, right=,
 * bottom_right=, masked=): ROCKE_ERR_VALUE (err filled when not NULL) for
 * a bound below -1 or a tile size below 1. */
rocke_status_t rocke_attn_bwd_static_kv_tile_plan(int64_t s_q,
                                                  int64_t s_kv,
                                                  int64_t k0,
                                                  int k_m0,
                                                  int k_n0,
                                                  int64_t left,
                                                  int64_t right,
                                                  bool bottom_right,
                                                  bool masked,
                                                  rocke_attn_bwd_static_kv_tile_plan_t* out,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BAND_H */
