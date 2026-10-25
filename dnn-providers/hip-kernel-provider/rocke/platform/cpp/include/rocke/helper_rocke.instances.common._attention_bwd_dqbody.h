/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_dqbody.h -- C port of
 * library/kernels/common/_attention_bwd_dqbody.py: the body of the split dQ
 * kernel of the attention backward (Q-parallel, no atomics).
 *
 *   Python (_attention_bwd_dqbody.py)   C (this header)
 *   ----------------------------------  ----------------------------------------
 *   DQ_TRANSPOSE_READERS                ROCKE_ATTN_BWD_DQ_TRANSPOSE_SOURCES /
 *                                       rocke_attn_bwd_dq_reader
 *   kv_tile_range                       rocke_attn_bwd_kv_tile_range
 *   interior_kv_range                   rocke_attn_bwd_interior_kv_range
 *   BwdDqGeometry (+ from_spec, check,  rocke_attn_bwd_dq_geometry_t (+ _from_spec,
 *     stage_table, g0t, g4t,              _stage_table, _wave_size, _threads,
 *     relabel_plan, b4_coords,            _reader; g0t / g4t / relabel_plan /
 *     wave_size, threads, reader)         b4_coords are fields filled by check)
 *   DqLdsViews                          rocke_attn_bwd_dq_lds_views_t
 *   BwdDqStep (+ stage_q, zero_dq,      rocke_attn_bwd_dq_step_t (+ _init,
 *     prologue, run_kv_step, store_dq)    _stage_q, _zero_dq, _prologue,
 *                                         _run_kv_step, _store_dq)
 *
 * The int-or-value arguments (n_rows) are rocke_attn_bwd_iv_t: v NULL is a
 * build-time int, as Python's isinstance(x, int).
 *
 * BwdDqGeometry: Python's cached properties g0t / g4t / relabel_plan are
 * built by check() (from_spec runs it), in Python's order; b4_coords (lazy
 * in Python, read by check only for the tr16 reader) is built after the
 * relabel plan. None of them emits IR (the builder carries errors only).
 *
 * BwdDqStep: the arrays it returns (accumulators, resident fragments,
 * per-column stats) live in the builder's arena. Python's ds_mult None (a
 * multiplier the compiled placement does not apply) is NULL; mult / head_step
 * / rows NULL in store_dq are None as well. A step keeps pointers to the
 * geometry, mask, stats and LDS views it is given: they must outlive it.
 *
 * Errors: a Python ValueError is recorded on the builder's sticky error
 * (ROCKE_ERR_VALUE) and the entry returns false; run under
 * ckc::guard_builder, as the kernel entry point does. Builder calls are made
 * in Python's evaluation order (left to right, inner call first), every
 * barrier carries ROCKE_ATTN_BWD_BARRIER_FENCE.
 *
 * Not ported yet: nothing of _attention_bwd_dqbody.py's public surface.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_DQBODY_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_DQBODY_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.attention_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* tuple(DQ_TRANSPOSE_READERS): ("tr_read", "lds_plain"). */
#define ROCKE_ATTN_BWD_N_DQ_TRANSPOSE_SOURCES 2
extern const char* const ROCKE_ATTN_BWD_DQ_TRANSPOSE_SOURCES[ROCKE_ATTN_BWD_N_DQ_TRANSPOSE_SOURCES];

/* DQ_TRANSPOSE_READERS[source]: "tr16" / "plain", NULL for another source. */
const char* rocke_attn_bwd_dq_reader(const char* transpose_source);

/* kv_tile_range(b, band, q_first, n_rows, *, k_n0, uniform). */
bool rocke_attn_bwd_kv_tile_range(rocke_ir_builder_t* b,
                                  rocke_band_emitter_t* band,
                                  rocke_value_t* q_first,
                                  rocke_attn_bwd_iv_t n_rows,
                                  int k_n0,
                                  bool uniform,
                                  rocke_value_t** out_kt_start,
                                  rocke_value_t** out_n_kt);

/* interior_kv_range(b, band, q_first, n_rows, kt_start, n_kt, *, k_n0,
 * masked, uniform). */
bool rocke_attn_bwd_interior_kv_range(rocke_ir_builder_t* b,
                                      rocke_band_emitter_t* band,
                                      rocke_value_t* q_first,
                                      rocke_attn_bwd_iv_t n_rows,
                                      rocke_value_t* kt_start,
                                      rocke_value_t* n_kt,
                                      int k_n0,
                                      bool masked,
                                      bool uniform,
                                      rocke_value_t** out_i0,
                                      rocke_value_t** out_i1);

/* BwdDqGeometry. The fields up to stage_vec are the dataclass's; the rest is
 * what check() builds. */
typedef struct rocke_attn_bwd_dq_geometry
{
    const char* arch;
    const char* dtype;
    int waves;
    int k_m0;
    int k_n0;
    int head_size;
    int atom_g02;
    int atom_g4;
    const char* transpose_source;
    const char* sched;
    int kv_pad;
    bool kv_prefetch;
    int stage_vec;
    /* check() */
    rocke_attn_bwd_block_gemm_t g0t; /* S^T = K Q^T (and dP^T): kN0 x kM0 over D */
    rocke_attn_bwd_block_gemm_t g4t; /* dQ += dS K: kM0 x D over kN0 */
    rocke_attn_bwd_relabel_plan_t relabel_plan;
    rocke_attn_bwd_frag_coords_t b4_coords;
} rocke_attn_bwd_dq_geometry_t;

struct rocke_attn_bwd_dq_spec;

/* BwdDqGeometry.from_spec(spec, arch): the geometry of a resolved dQ spec,
 * checked (check() raises before any IR is emitted). */
bool rocke_attn_bwd_dq_geometry_from_spec(rocke_ir_builder_t* b,
                                          const struct rocke_attn_bwd_dq_spec* spec,
                                          const char* arch,
                                          rocke_attn_bwd_dq_geometry_t* out);
/* wave_size / threads (0 on an unknown arch) and reader ("tr16" / "plain"). */
int rocke_attn_bwd_dq_geometry_wave_size(const rocke_attn_bwd_dq_geometry_t* g);
int rocke_attn_bwd_dq_geometry_threads(const rocke_attn_bwd_dq_geometry_t* g);
const char* rocke_attn_bwd_dq_geometry_reader(const rocke_attn_bwd_dq_geometry_t* g);
/* stage_table(): DqStageTable.from_geometry(self). */
bool rocke_attn_bwd_dq_geometry_stage_table(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_dq_geometry_t* g,
                                            rocke_attn_bwd_dq_stage_table_t* out);

/* DqLdsViews. */
typedef struct rocke_attn_bwd_dq_lds_views
{
    rocke_attn_bwd_lds_view_t q;
    rocke_attn_bwd_lds_view_t do_;
    rocke_attn_bwd_lds_view_t k;
    rocke_attn_bwd_lds_view_t v;
    rocke_attn_bwd_lds_view_t out;
} rocke_attn_bwd_dq_lds_views_t;

/* BwdDqStep. cols_* hold self.cols[fn] (lse2, dsum, live, limits) for fn in
 * [0, n_cols); live NULL is None. qt_regs / dot_regs are g0t B fragments
 * [frags_n][k_steps]. */
typedef struct rocke_attn_bwd_dq_step
{
    const rocke_attn_bwd_dq_geometry_t* geom;
    rocke_attn_bwd_mask_policy_t* mask;
    const rocke_attn_bwd_stats_view_t* stats;
    const rocke_attn_bwd_dq_lds_views_t* lds;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wave;
    rocke_value_t* scale_log2;
    rocke_value_t* ds_mult;
    bool prescale;
    bool sentinel;
    const rocke_type_t* elem;
    rocke_value_t* o0[2];
    rocke_value_t* o4[2];
    rocke_value_t** qt_regs;
    rocke_value_t** dot_regs;
    int n_cols;
    rocke_value_t** col_lse2;
    rocke_value_t** col_dsum;
    rocke_value_t** col_live;
    rocke_value_t** col_lo;
    rocke_value_t** col_hi;
    rocke_attn_bwd_dq_stage_table_t sched;
} rocke_attn_bwd_dq_step_t;

/* BwdDqStep.__init__ (emits the two wave origins). */
bool rocke_attn_bwd_dq_step_init(rocke_attn_bwd_dq_step_t* t,
                                 rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_dq_geometry_t* geom,
                                 rocke_attn_bwd_mask_policy_t* mask,
                                 const rocke_attn_bwd_stats_view_t* stats,
                                 const rocke_attn_bwd_dq_lds_views_t* lds,
                                 rocke_value_t* tid,
                                 rocke_value_t* lane,
                                 rocke_value_t* wave,
                                 rocke_value_t* scale_log2,
                                 rocke_value_t* ds_mult,
                                 bool prescale,
                                 bool sentinel);

/* stage_q(b, live=, h=, q0=, q_slab=, do_slab=, q_addr=, do_addr=, rows=,
 * vec=). The slabs are read when rows is NULL (None), the addresses
 * otherwise. */
bool rocke_attn_bwd_dq_step_stage_q(rocke_attn_bwd_dq_step_t* t,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* live,
                                    rocke_value_t* h,
                                    rocke_value_t* q0,
                                    const rocke_attn_bwd_row_slab_t* q_slab,
                                    const rocke_attn_bwd_row_slab_t* do_slab,
                                    const rocke_attn_bwd_strided_tensor_addr_t* q_addr,
                                    const rocke_attn_bwd_strided_tensor_addr_t* do_addr,
                                    const rocke_attn_bwd_packed_rows_t* rows,
                                    int vec);

/* zero_dq: out[g4t frags_m * frags_n] (row-major). */
bool rocke_attn_bwd_dq_step_zero_dq(rocke_attn_bwd_dq_step_t* t,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t** out);

/* prologue(b, k_slab, v_slab, k0, live). */
bool rocke_attn_bwd_dq_step_prologue(rocke_attn_bwd_dq_step_t* t,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_row_slab_t* k_slab,
                                     const rocke_attn_bwd_row_slab_t* v_slab,
                                     rocke_value_t* k0,
                                     rocke_value_t* live);

/* run_kv_step(b, dq, k_slab, v_slab, k0, *, edge, prefetch, vec): dq and out
 * are g4t [frags_m][frags_n] row-major (out may alias dq). */
bool rocke_attn_bwd_dq_step_run_kv_step(rocke_attn_bwd_dq_step_t* t,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* const* dq,
                                        const rocke_attn_bwd_row_slab_t* k_slab,
                                        const rocke_attn_bwd_row_slab_t* v_slab,
                                        rocke_value_t* k0,
                                        bool edge,
                                        bool prefetch,
                                        int vec,
                                        rocke_value_t** out);

/* store_dq(b, dq, *, mult, dq_slab, head_step, h, q0, rows, len_q, s_q_max,
 * seq_mode). dq_slab must be unclamped (last NULL) and rows NULL exactly when
 * head_step is NULL (Python asserts both). */
bool rocke_attn_bwd_dq_step_store_dq(rocke_attn_bwd_dq_step_t* t,
                                     rocke_ir_builder_t* b,
                                     rocke_value_t* const* dq,
                                     rocke_value_t* mult,
                                     const rocke_attn_bwd_row_slab_t* dq_slab,
                                     rocke_value_t* head_step,
                                     rocke_value_t* h,
                                     rocke_value_t* q0,
                                     const rocke_attn_bwd_packed_rows_t* rows,
                                     rocke_value_t* len_q,
                                     rocke_value_t* s_q_max,
                                     rocke_attn_bwd_seq_mode_t seq_mode);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_DQBODY_H */
