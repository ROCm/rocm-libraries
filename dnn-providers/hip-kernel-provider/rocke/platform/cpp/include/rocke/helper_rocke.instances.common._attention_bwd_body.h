/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_body.h -- C port of the
 * shared inner body of the attention backward main kernel
 * (library/kernels/common/_attention_bwd_body.py): the tile geometry, the
 * CTA's flat LDS, the stats view, the copy helpers and the per-q-step G0..G4
 * program (BwdTileStep.run_q_step) with its staging, dQ sink hand-off and
 * dK / dV epilogue readers.
 *
 *   Python (_attention_bwd_body.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   STAGE_ORDERS, DEFAULT_STAGE_ORDER,  ROCKE_ATTN_BWD_STAGE_ORDERS, ..._DEFAULT_STAGE_ORDER,
 *   G4_SLICINGS, DEFAULT_G4_SLICING,    ROCKE_ATTN_BWD_G4_SLICINGS, ..._DEFAULT_G4_SLICING,
 *   G4_AFTER_G3_ORDERS, COPY_ORDERS     ROCKE_ATTN_BWD_G4_AFTER_G3_ORDERS, ..._COPY_ORDERS
 *   stage_units                         rocke_attn_bwd_stage_units
 *   BwdAtomTable                        rocke_attn_bwd_atom_table_t
 *   BwdTileGeometry                     rocke_attn_bwd_tile_geometry_t
 *     from_spec / check                   rocke_attn_bwd_tile_geometry_from_spec / _check_in_place
 *     g0 / g1 / g4, relabel_plan,         the derived members (filled by _check_in_place)
 *     b13_coords
 *     wave_size / threads / reader /      rocke_attn_bwd_tile_geometry_<name>
 *     g4_slice_buffers /
 *     g4_prefetch_before_g3 /
 *     counted_atomics / stage_table
 *   quad_transpose4                     rocke_attn_bwd_quad_transpose4
 *   BwdLds (+ has, view)                rocke_attn_bwd_bwd_lds_t (+ _init, _has, _view)
 *   copy_rows_to_lds, slab_row_loader,  rocke_attn_bwd_copy_rows_to_lds, ..._slab_row_loader,
 *   packed_row_loader,                  ..._packed_row_loader,
 *   copy_packed_rows_to_lds,            ..._copy_packed_rows_to_lds,
 *   copy_row_sets_to_lds, write_xt      ..._copy_row_sets_to_lds, ..._write_xt
 *   StatsView (+ methods)               rocke_attn_bwd_stats_view_t (+ _init, _rows, ...)
 *   PackedStep, QStepInputs, RingStep   rocke_attn_bwd_packed_step_t, ..._q_step_inputs_t,
 *                                       ..._ring_step_t
 *   DkDvAcc (+ flat, from_flat)         rocke_attn_bwd_dkdv_acc_t (+ _flat_len, _flat,
 *                                       _from_flat)
 *   BwdTileStep (+ methods)             rocke_attn_bwd_tile_step_t (+ _init, _<method>)
 *
 * Builder errors: every entry point records a Python ValueError (and the
 * RuntimeError / NotImplementedError of the helpers it calls) on the
 * builder's sticky error through rocke_i_set_err, which raises ckc::Error:
 * call these under ckc::guard_builder, as the kernel entry points do. The
 * bool entries return false and the Value entries NULL after an error.
 * stage_units has no builder and reports through a rocke_attn_bwd_err_t.
 *
 * Byte-identity rules: ops are created in the Python evaluation order (left
 * to right, inner call first), every builder call (constants included) bound
 * to a local first. Every barrier carries ROCKE_ATTN_BWD_BARRIER_FENCE (the
 * Python builders set it as IRBuilder.fence_barriers).
 *
 * Memory: arrays an entry point returns (accumulators, register fragments,
 * stats rows, staged loads, pending dQ values) are allocated from the
 * builder's arena and live as long as the builder. A tile step owns its
 * counted-wait record: release it with rocke_attn_bwd_tile_step_free. The
 * geometry, plan, mask, stats view, sink and LDS a step points to are
 * borrowed and must outlive it (Python shares them by reference).
 *
 * Python duck types a few inputs the C port spells as plain data: an atom is
 * its K extent (the spec's "16x16x16" / "wmma16x16x16" strings are parsed by
 * from_spec), the loaders of the copy helpers are rocke_attn_bwd_tile_loader_t
 * (the counted header's callable), and a K-outer LdsOperand reader is its
 * Python string (the "xt" reader is refused where Python's LdsOperand
 * refuses it).
 *
 * run_q_step's `prefetch` (a PrefetchStep) and the LdsStatsView it hands back
 * are the ring port's (helper_rocke.instances.common._attention_bwd_ring.h).
 *
 * Everything in _attention_bwd_body.py is ported.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BODY_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BODY_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_counted.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sched.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rocke_attn_bwd_spec; /* instance_attention_bwd.h */
struct rocke_attn_bwd_prefetch_step; /* _attention_bwd_ring.h */

/* ------------------------------------------------------------- constants */

#define ROCKE_ATTN_BWD_N_STAGE_ORDERS 6
extern const char* const ROCKE_ATTN_BWD_STAGE_ORDERS[ROCKE_ATTN_BWD_N_STAGE_ORDERS];
extern const char* const ROCKE_ATTN_BWD_DEFAULT_STAGE_ORDER;
#define ROCKE_ATTN_BWD_N_G4_SLICINGS 4
extern const char* const ROCKE_ATTN_BWD_G4_SLICINGS[ROCKE_ATTN_BWD_N_G4_SLICINGS];
extern const char* const ROCKE_ATTN_BWD_DEFAULT_G4_SLICING;
/* The stage orders with G3 right before G4 ("slice_db_g3"). */
#define ROCKE_ATTN_BWD_N_G4_AFTER_G3_ORDERS 2
extern const char* const ROCKE_ATTN_BWD_G4_AFTER_G3_ORDERS[ROCKE_ATTN_BWD_N_G4_AFTER_G3_ORDERS];
/* Issue orders of copy_row_sets_to_lds (the qdo_stage knob values). */
#define ROCKE_ATTN_BWD_N_COPY_ORDERS 3
extern const char* const ROCKE_ATTN_BWD_COPY_ORDERS[ROCKE_ATTN_BWD_N_COPY_ORDERS];

/* stage_units(order): the three GEMM names ("g1", "g3", "g4") in issue order
 * (static strings). An order outside STAGE_ORDERS is a ValueError. */
rocke_status_t
    rocke_attn_bwd_stage_units(const char* order, const char* out[3], rocke_attn_bwd_err_t* err);

/* -------------------------------------------------------------- geometry */

/* BwdAtomTable: atom K extent per GEMM pair. */
typedef struct rocke_attn_bwd_atom_table
{
    int g02;
    int g13;
    int g4;
} rocke_attn_bwd_atom_table_t;

/* BwdTileGeometry. Strings are borrowed (the spec's spellings); NULL is None
 * for lds_swizzle (the canonical "name=value,..." text) and agpr_route;
 * asm_tail_nop ROCKE_ATTN_BWD_NONE is None. The members after the Python
 * fields are its cached properties, filled by rocke_attn_bwd_tile_geometry_check_in_place (the name
 * rocke_attn_bwd_tile_geometry_check is the spec-level check of
 * instance_attention_bwd.h). */
typedef struct rocke_attn_bwd_tile_geometry
{
    const char* arch;
    const char* dtype;
    int waves;
    int k_m0;
    int k_n0;
    int k_k4;
    int head_size;
    int grid_g0[2];
    int grid_g1[2];
    int grid_g4[2];
    rocke_attn_bwd_atom_table_t atoms;
    const char* pt_route;
    const char* kv_residency;
    const char* kt_source;
    const char* transpose_source;
    int ring_depth; /* 1 */
    const char* global_path; /* "vgpr" */
    const char* lds_swizzle; /* NULL */
    const char* sched; /* "none" */
    bool setprio;
    bool edge_tiles;
    bool head_pack;
    rocke_attn_bwd_agpr_alloc_t agpr_alloc; /* kind NONE */
    bool s_in_agpr;
    bool acc_in_lds;
    const char* dq_mode; /* "atomic" */
    const char* agpr_route; /* NULL */
    int stage_vec; /* 8 */
    int asm_tail_nop; /* ROCKE_ATTN_BWD_NONE */
    const char* asm_cluster; /* "k_step" */
    bool asm_load_hoist;
    const char* stage_order; /* DEFAULT_STAGE_ORDER */
    const char* loop_sync; /* "full" */
    const char* dkv_epilogue; /* "lds_staged" */
    int dkv_store_vec; /* 8 */
    const char* qdo_stage; /* "split" */
    const char* stats_load; /* "at_use" */
    const char* dq_issue; /* "block" */
    bool interior_clamp;
    const char* barrier_merge; /* "off" */
    const char* ring_issue; /* "early" */
    const char* counted_issue; /* "early" */
    const char* bf16_cast; /* "generic" */
    const char* bwd_exp2; /* "exact" */
    bool stats_fma;
    int stats_vec; /* 1 */
    const char* g4_slicing; /* DEFAULT_G4_SLICING */

    /* cached properties (rocke_attn_bwd_tile_geometry_check_in_place) */
    bool derived;
    rocke_attn_bwd_block_gemm_t g0; /* S = Q K^T (and dP = dO V^T) */
    rocke_attn_bwd_block_gemm_t g1; /* dV += P^T dO (and dK += dS^T Q) */
    rocke_attn_bwd_block_gemm_t g4; /* dQ = dS K */
    bool has_relabel_plan; /* false: None (the LDS route) */
    rocke_attn_bwd_relabel_plan_t relabel_plan;
    rocke_attn_bwd_frag_coords_t b13_coords;
} rocke_attn_bwd_tile_geometry_t;

/* The dataclass defaults (the fields without one are 0 / NULL). */
rocke_attn_bwd_tile_geometry_t rocke_attn_bwd_tile_geometry_default(void);
/* BwdTileGeometry.from_spec(spec, arch) (then check). */
bool rocke_attn_bwd_tile_geometry_from_spec(rocke_ir_builder_t* b,
                                            const struct rocke_attn_bwd_spec* spec,
                                            const char* arch,
                                            rocke_attn_bwd_tile_geometry_t* out);
/* check(): the block GEMMs, the relabel proof, the transpose-read proof and
 * every coupling; fills the cached properties. No IR is emitted. */
bool rocke_attn_bwd_tile_geometry_check_in_place(rocke_ir_builder_t* b,
                                                 rocke_attn_bwd_tile_geometry_t* g);

/* wave_size / threads: 0 for an arch bwd_arch_caps does not know (Python
 * raises there; the emitting functions raise the same ValueError). */
int rocke_attn_bwd_tile_geometry_wave_size(const rocke_attn_bwd_tile_geometry_t* g);
int rocke_attn_bwd_tile_geometry_threads(const rocke_attn_bwd_tile_geometry_t* g);
/* reader: "plain", "tr16" or "xt" (static strings). */
const char* rocke_attn_bwd_tile_geometry_reader(const rocke_attn_bwd_tile_geometry_t* g);
int rocke_attn_bwd_tile_geometry_g4_slice_buffers(const rocke_attn_bwd_tile_geometry_t* g);
bool rocke_attn_bwd_tile_geometry_g4_prefetch_before_g3(const rocke_attn_bwd_tile_geometry_t* g);
int rocke_attn_bwd_tile_geometry_counted_atomics(const rocke_attn_bwd_tile_geometry_t* g);
/* stage_table(): StageTable.from_geometry(self, arch, dtype=, stage_vec=,
 * dq_mode=, acc_in_lds=). The table borrows the geometry's strings. */
bool rocke_attn_bwd_tile_geometry_stage_table(rocke_ir_builder_t* b,
                                              const rocke_attn_bwd_tile_geometry_t* g,
                                              rocke_attn_bwd_stage_table_t* out);

/* -------------------------------------------------------- free helpers */

/* quad_transpose4: out[4] from vals[4]. */
bool rocke_attn_bwd_quad_transpose4(rocke_ir_builder_t* b,
                                    rocke_value_t* const* vals,
                                    int n_vals,
                                    rocke_value_t* lane,
                                    rocke_value_t** out);

/* BwdLds: one flat allocation of plan->total_bytes (the plan is borrowed). */
typedef struct rocke_attn_bwd_bwd_lds
{
    const rocke_attn_bwd_lds_plan_t* plan;
    const rocke_type_t* elem;
    rocke_value_t* smem;
} rocke_attn_bwd_bwd_lds_t;

bool rocke_attn_bwd_bwd_lds_init(rocke_attn_bwd_bwd_lds_t* lds,
                                 rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_lds_plan_t* plan,
                                 const char* dtype);
bool rocke_attn_bwd_bwd_lds_has(const rocke_attn_bwd_bwd_lds_t* lds, const char* name);
/* view(name, cols, slot): no IR. */
bool rocke_attn_bwd_bwd_lds_view(rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_bwd_lds_t* lds,
                                 const char* name,
                                 int cols,
                                 int slot,
                                 rocke_attn_bwd_lds_view_t* out);

/* slab_row_loader(b, slab, row0, vec): load(r, c) = slab.load_vec(row0 + r,
 * c, vec). The context is borrowed by the loader; slab is copied. */
typedef struct rocke_attn_bwd_slab_row_loader
{
    rocke_ir_builder_t* b;
    rocke_attn_bwd_row_slab_t slab;
    rocke_value_t* row0;
    int vec;
} rocke_attn_bwd_slab_row_loader_t;

rocke_attn_bwd_tile_loader_t rocke_attn_bwd_slab_row_loader(rocke_attn_bwd_slab_row_loader_t* ctx);

/* packed_row_loader(b, addr, packed, vec, clamp=): load(r, c) of M row r of
 * a head-packed step (its own head). addr is borrowed. */
typedef struct rocke_attn_bwd_packed_row_loader
{
    rocke_ir_builder_t* b;
    const rocke_attn_bwd_strided_tensor_addr_t* addr;
    rocke_attn_bwd_packed_rows_t packed;
    int vec;
    bool clamp;
} rocke_attn_bwd_packed_row_loader_t;

rocke_attn_bwd_tile_loader_t
    rocke_attn_bwd_packed_row_loader(rocke_attn_bwd_packed_row_loader_t* ctx);

/* copy_rows_to_lds(b, slab, view, row0=, rows=, cols=, tid=, threads=, vec=). */
bool rocke_attn_bwd_copy_rows_to_lds(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_row_slab_t* slab,
                                     const rocke_attn_bwd_lds_view_t* view,
                                     rocke_value_t* row0,
                                     int rows,
                                     int cols,
                                     rocke_value_t* tid,
                                     int threads,
                                     int vec);
/* copy_packed_rows_to_lds(b, addr, view, packed, rows=, cols=, tid=,
 * threads=, vec=, clamp=). */
bool rocke_attn_bwd_copy_packed_rows_to_lds(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_strided_tensor_addr_t* addr,
                                            const rocke_attn_bwd_lds_view_t* view,
                                            const rocke_attn_bwd_packed_rows_t* packed,
                                            int rows,
                                            int cols,
                                            rocke_value_t* tid,
                                            int threads,
                                            int vec,
                                            bool clamp);

/* One (load, view) pair of copy_row_sets_to_lds. */
typedef struct rocke_attn_bwd_row_set
{
    rocke_attn_bwd_tile_loader_t load;
    const rocke_attn_bwd_lds_view_t* view;
} rocke_attn_bwd_row_set_t;

/* copy_row_sets_to_lds(b, sets, order=, rows=, cols=, tid=, threads=, wave=,
 * wave_size=, vec=). wave NULL is None, wave_size 0 is None. */
bool rocke_attn_bwd_copy_row_sets_to_lds(rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_row_set_t* sets,
                                         int n_sets,
                                         const char* order,
                                         int rows,
                                         int cols,
                                         rocke_value_t* tid,
                                         int threads,
                                         rocke_value_t* wave,
                                         int wave_size,
                                         int vec);

/* write_xt(b, src, dst, seq=, depth=, tid=, threads=, elem=). */
bool rocke_attn_bwd_write_xt(rocke_ir_builder_t* b,
                             const rocke_attn_bwd_lds_view_t* src,
                             const rocke_attn_bwd_lds_view_t* dst,
                             int seq,
                             int depth,
                             rocke_value_t* tid,
                             int threads,
                             const rocke_type_t* elem);

/* ------------------------------------------------------------ StatsView */

typedef enum rocke_attn_bwd_stats_bounds
{
    ROCKE_ATTN_BWD_STATS_CLAMP = 0,
    ROCKE_ATTN_BWD_STATS_BUFFER = 1
} rocke_attn_bwd_stats_bounds_t;

/* StatsView. The workspaces are copied; n_heads NULL is None. */
typedef struct rocke_attn_bwd_stats_view
{
    rocke_attn_bwd_workspace_addr_t lse2;
    rocke_attn_bwd_workspace_addr_t dsum;
    rocke_value_t* n_heads;
    rocke_attn_bwd_stats_bounds_t bounds;
    int span;
} rocke_attn_bwd_stats_view_t;

/* StatsView(lse2, dsum, n_heads, bounds=, span=); bounds is "clamp" or
 * "buffer". */
bool rocke_attn_bwd_stats_view_init(rocke_attn_bwd_stats_view_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_workspace_addr_t* lse2,
                                    const rocke_attn_bwd_workspace_addr_t* dsum,
                                    rocke_value_t* n_heads,
                                    const char* bounds,
                                    int span);
/* rows / batched_rows / packed_rows / raw / finish: out_lse2[n_rows] and
 * out_dsum[n_rows] (Python's list of (lse2, Dsum) pairs). packed NULL is
 * None (raw / finish). */
bool rocke_attn_bwd_stats_view_rows(const rocke_attn_bwd_stats_view_t* s,
                                    rocke_ir_builder_t* b,
                                    rocke_value_t* h,
                                    rocke_value_t* q0,
                                    rocke_value_t* const* rows,
                                    int n_rows,
                                    rocke_value_t** out_lse2,
                                    rocke_value_t** out_dsum);
bool rocke_attn_bwd_stats_view_batched_rows(const rocke_attn_bwd_stats_view_t* s,
                                            rocke_ir_builder_t* b,
                                            rocke_value_t* h,
                                            rocke_value_t* q0,
                                            rocke_value_t* const* rows,
                                            int n_rows,
                                            int width,
                                            rocke_value_t** out_lse2,
                                            rocke_value_t** out_dsum);
bool rocke_attn_bwd_stats_view_packed_rows(const rocke_attn_bwd_stats_view_t* s,
                                           rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_packed_rows_t* packed,
                                           rocke_value_t* const* rows,
                                           int n_rows,
                                           rocke_value_t** out_lse2,
                                           rocke_value_t** out_dsum);
bool rocke_attn_bwd_stats_view_raw(const rocke_attn_bwd_stats_view_t* s,
                                   rocke_ir_builder_t* b,
                                   rocke_value_t* h,
                                   rocke_value_t* q0,
                                   rocke_value_t* const* rows,
                                   int n_rows,
                                   const rocke_attn_bwd_packed_rows_t* packed,
                                   rocke_value_t** out_lse2,
                                   rocke_value_t** out_dsum);
bool rocke_attn_bwd_stats_view_finish(const rocke_attn_bwd_stats_view_t* s,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* h,
                                      rocke_value_t* q0,
                                      rocke_value_t* const* rows,
                                      int n_rows,
                                      rocke_value_t* const* raw_lse2,
                                      rocke_value_t* const* raw_dsum,
                                      const rocke_attn_bwd_packed_rows_t* packed,
                                      rocke_value_t** out_lse2,
                                      rocke_value_t** out_dsum);

/* ---------------------------------------------------------- step inputs */

/* PackedStep: the row map and the Q / dO tensors (borrowed). */
typedef struct rocke_attn_bwd_packed_step
{
    rocke_attn_bwd_packed_rows_t rows;
    const rocke_attn_bwd_strided_tensor_addr_t* q_addr;
    const rocke_attn_bwd_strided_tensor_addr_t* do_addr;
} rocke_attn_bwd_packed_step_t;

/* QStepInputs: has_slabs false is q_slab = do_slab = None; packed NULL is
 * None (borrowed). */
typedef struct rocke_attn_bwd_q_step_inputs
{
    rocke_value_t* q0;
    rocke_value_t* h;
    bool has_slabs;
    rocke_attn_bwd_row_slab_t q_slab;
    rocke_attn_bwd_row_slab_t do_slab;
    const rocke_attn_bwd_packed_step_t* packed;
} rocke_attn_bwd_q_step_inputs_t;

/* RingStep: the ring slot (uniform i32 0 / 1) and the next step. */
typedef struct rocke_attn_bwd_ring_step
{
    rocke_value_t* slot;
    rocke_attn_bwd_q_step_inputs_t nxt;
} rocke_attn_bwd_ring_step_t;

/* DkDvAcc: dk / dv are [frags_m][frags_n] row-major (empty, 0 x 0, with
 * acc_in_lds). Non-owning. */
typedef struct rocke_attn_bwd_dkdv_acc
{
    rocke_value_t** dk;
    rocke_value_t** dv;
    int frags_m;
    int frags_n;
} rocke_attn_bwd_dkdv_acc_t;

int rocke_attn_bwd_dkdv_acc_flat_len(const rocke_attn_bwd_dkdv_acc_t* acc);
void rocke_attn_bwd_dkdv_acc_flat(const rocke_attn_bwd_dkdv_acc_t* acc, rocke_value_t** out);
/* from_flat(values, frags_m, frags_n): the result points into values. */
bool rocke_attn_bwd_dkdv_acc_from_flat(rocke_ir_builder_t* b,
                                       rocke_value_t* const* values,
                                       int n_values,
                                       int frags_m,
                                       int frags_n,
                                       rocke_attn_bwd_dkdv_acc_t* out);

/* ------------------------------------------------------------ BwdTileStep */

typedef struct rocke_attn_bwd_tile_step
{
    const rocke_attn_bwd_tile_geometry_t* geom;
    rocke_attn_bwd_mask_policy_t* mask;
    const rocke_attn_bwd_stats_view_t* stats;
    rocke_attn_bwd_dq_sink_t* sink;
    const rocke_attn_bwd_bwd_lds_t* lds;
    rocke_value_t* tid;
    rocke_value_t* lane;
    rocke_value_t* wave;
    rocke_value_t* scale_log2;
    rocke_value_t* ds_mult; /* NULL: None (1.0, not applied) */
    bool sentinel;
    rocke_attn_bwd_stage_table_t sched;
    const rocke_type_t* elem;
    bool g0_replicated;
    rocke_value_t* o0[2];
    rocke_value_t* o1[2];
    rocke_value_t* o4[2]; /* NULL: None (no G4) */
    rocke_attn_bwd_frags_t k_regs; /* v NULL: None */
    rocke_attn_bwd_frags_t v_regs;
    rocke_attn_bwd_frags_t kt_regs;
    rocke_attn_bwd_lds_view_t v_q, v_do;
    bool has_v_ds, has_v_pt, has_v_ring_stats;
    rocke_attn_bwd_lds_view_t v_ds, v_pt, v_lse2, v_dsum;
    rocke_attn_bwd_lds_view_t v_k, v_v;
    bool has_v_qt, has_v_dot, has_v_kt;
    rocke_attn_bwd_lds_view_t v_qt, v_dot, v_kt;
    bool has_v_acc;
    rocke_attn_bwd_lds_view_t v_dk_acc, v_dv_acc;
    bool has_waits; /* loop_sync = "counted" */
    rocke_attn_bwd_counted_waits_t waits;
} rocke_attn_bwd_tile_step_t;

/* BwdTileStep(b, geom, mask=, stats=, dq_sink=, lds=, tid=, lane=, wave=,
 * scale_log2=, ds_mult=, sentinel=). geom must be checked. */
bool rocke_attn_bwd_tile_step_init(rocke_attn_bwd_tile_step_t* t,
                                   rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_tile_geometry_t* geom,
                                   rocke_attn_bwd_mask_policy_t* mask,
                                   const rocke_attn_bwd_stats_view_t* stats,
                                   rocke_attn_bwd_dq_sink_t* dq_sink,
                                   const rocke_attn_bwd_bwd_lds_t* lds,
                                   rocke_value_t* tid,
                                   rocke_value_t* lane,
                                   rocke_value_t* wave,
                                   rocke_value_t* scale_log2,
                                   rocke_value_t* ds_mult,
                                   bool sentinel);
void rocke_attn_bwd_tile_step_free(rocke_attn_bwd_tile_step_t* t);

/* stage_kv(b, k_slab, v_slab, k0, live). */
bool rocke_attn_bwd_tile_step_stage_kv(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_row_slab_t* k_slab,
                                       const rocke_attn_bwd_row_slab_t* v_slab,
                                       rocke_value_t* k0,
                                       rocke_value_t* live);
/* zero_acc(b). */
bool rocke_attn_bwd_tile_step_zero_acc(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       rocke_attn_bwd_dkdv_acc_t* out);

/* counted_n_stats / counted_zero (out of counted_zero_len values) /
 * counted_loads (loop_sync = "counted"). */
int rocke_attn_bwd_tile_step_counted_n_stats(const rocke_attn_bwd_tile_step_t* t);
int rocke_attn_bwd_tile_step_counted_zero_len(const rocke_attn_bwd_tile_step_t* t);
bool rocke_attn_bwd_tile_step_counted_zero(rocke_attn_bwd_tile_step_t* t,
                                           rocke_ir_builder_t* b,
                                           rocke_value_t** out);
/* counted_loads(b, h=, q0=, q_slab=, do_slab=, packed=, edge=); slabs NULL
 * are None. */
bool rocke_attn_bwd_tile_step_counted_loads(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            rocke_value_t* h,
                                            rocke_value_t* q0,
                                            const rocke_attn_bwd_row_slab_t* q_slab,
                                            const rocke_attn_bwd_row_slab_t* do_slab,
                                            const rocke_attn_bwd_packed_step_t* packed,
                                            bool edge,
                                            rocke_attn_bwd_staged_step_t* out);

/* run_q_step's next_loads callable: the next step's StagedStep. */
typedef bool (*rocke_attn_bwd_next_loads_fn)(void* ctx,
                                             rocke_ir_builder_t* b,
                                             rocke_attn_bwd_staged_step_t* out);

/* Keyword arguments of run_q_step (NULL is None). */
typedef struct rocke_attn_bwd_q_step_opts
{
    bool edge; /* true */
    const rocke_attn_bwd_packed_step_t* packed;
    const rocke_attn_bwd_ring_step_t* ring;
    const rocke_attn_bwd_staged_step_t* staged;
    rocke_attn_bwd_next_loads_fn next_loads;
    void* next_loads_ctx;
    const rocke_attn_bwd_pending_dq_t* pending;
    struct rocke_attn_bwd_prefetch_step* prefetch; /* the ring's part of the step */
} rocke_attn_bwd_q_step_opts_t;

rocke_attn_bwd_q_step_opts_t rocke_attn_bwd_q_step_opts_default(void);

/* run_q_step's result: the accumulators, then (loop_sync = "counted") the
 * next step's staged loads and (stage_table_atomics) the pending dQ. */
typedef struct rocke_attn_bwd_q_step_result
{
    rocke_attn_bwd_dkdv_acc_t acc;
    bool has_staged;
    rocke_attn_bwd_staged_step_t staged;
    bool has_pending;
    rocke_attn_bwd_pending_dq_t pending;
} rocke_attn_bwd_q_step_result_t;

/* run_q_step(b, acc, q_slab, do_slab, qt, h, k0, **opts); slabs NULL are
 * None. */
bool rocke_attn_bwd_tile_step_run_q_step(rocke_attn_bwd_tile_step_t* t,
                                         rocke_ir_builder_t* b,
                                         const rocke_attn_bwd_dkdv_acc_t* acc,
                                         const rocke_attn_bwd_row_slab_t* q_slab,
                                         const rocke_attn_bwd_row_slab_t* do_slab,
                                         rocke_value_t* qt,
                                         rocke_value_t* h,
                                         rocke_value_t* k0,
                                         const rocke_attn_bwd_q_step_opts_t* opts,
                                         rocke_attn_bwd_q_step_result_t* out);

/* zero_pending_dq(b, h, n_heads=, rows_per_head=) (NULL: None). */
bool rocke_attn_bwd_tile_step_zero_pending_dq(rocke_attn_bwd_tile_step_t* t,
                                              rocke_ir_builder_t* b,
                                              rocke_value_t* h,
                                              rocke_value_t* n_heads,
                                              rocke_value_t* rows_per_head,
                                              rocke_attn_bwd_pending_dq_t* out);
/* flush_dq(b, pending). */
bool rocke_attn_bwd_tile_step_flush_dq(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_pending_dq_t* pending);

/* ring_prologue / ring_publish (barrier_merge = "on"). */
bool rocke_attn_bwd_tile_step_ring_prologue(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_q_step_inputs_t* step);
bool rocke_attn_bwd_tile_step_ring_publish(rocke_attn_bwd_tile_step_t* t, rocke_ir_builder_t* b);

/* finish(b, acc). */
bool rocke_attn_bwd_tile_step_finish(rocke_attn_bwd_tile_step_t* t,
                                     rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_dkdv_acc_t* acc,
                                     rocke_attn_bwd_dkdv_acc_t* out);

/* ----- epilogue helpers (which: "dk" or "dv"; mult NULL is None) ----- */

/* One (row, [(col, value), ...]) entry of dkv_row_slots / dkv_cells. */
typedef struct rocke_attn_bwd_dkv_row
{
    rocke_value_t* row;
    int n_items;
    rocke_value_t** cols;
    rocke_value_t** values;
} rocke_attn_bwd_dkv_row_t;

/* dkv_row_slots / dkv_cells: *out points to *n_out rows (arena). */
bool rocke_attn_bwd_tile_step_dkv_row_slots(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            const rocke_attn_bwd_dkdv_acc_t* acc,
                                            rocke_attn_bwd_dkv_row_t** out,
                                            int* n_out);
bool rocke_attn_bwd_tile_step_dkv_cells(rocke_attn_bwd_tile_step_t* t,
                                        rocke_ir_builder_t* b,
                                        const char* which,
                                        const rocke_attn_bwd_dkdv_acc_t* acc,
                                        rocke_value_t* mult,
                                        rocke_attn_bwd_dkv_row_t** out,
                                        int* n_out);

/* dkv_rows(b, which, acc, mult, vec=, slot=, sync=) returns a reader. */
typedef struct rocke_attn_bwd_dkv_reader
{
    bool in_lds_acc; /* the fp32 accumulator image (acc_in_lds) */
    rocke_attn_bwd_lds_view_t view;
    rocke_value_t* mult;
    int vec;
    const rocke_type_t* elem;
} rocke_attn_bwd_dkv_reader_t;

bool rocke_attn_bwd_tile_step_dkv_rows(rocke_attn_bwd_tile_step_t* t,
                                       rocke_ir_builder_t* b,
                                       const char* which,
                                       const rocke_attn_bwd_dkdv_acc_t* acc,
                                       rocke_value_t* mult,
                                       int vec,
                                       int slot,
                                       bool sync,
                                       rocke_attn_bwd_dkv_reader_t* out);
/* read(r, c): <vec x dtype>. */
rocke_value_t* rocke_attn_bwd_dkv_reader_read(rocke_ir_builder_t* b,
                                              const rocke_attn_bwd_dkv_reader_t* rd,
                                              rocke_value_t* r,
                                              rocke_value_t* c);

/* dkv_quad_rows: one (row, col0, [4 values]) per accumulator fragment. */
typedef struct rocke_attn_bwd_dkv_quad
{
    rocke_value_t* row;
    rocke_value_t* col0;
    rocke_value_t* values[4];
} rocke_attn_bwd_dkv_quad_t;

bool rocke_attn_bwd_tile_step_dkv_quad_rows(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            const rocke_attn_bwd_dkdv_acc_t* acc,
                                            rocke_value_t* mult,
                                            rocke_attn_bwd_dkv_quad_t** out,
                                            int* n_out);
/* dkv_f32_image(which) (no IR). */
bool rocke_attn_bwd_tile_step_dkv_f32_image(rocke_attn_bwd_tile_step_t* t,
                                            rocke_ir_builder_t* b,
                                            const char* which,
                                            rocke_attn_bwd_lds_view_t* out);
/* dkv_f32_part(b, which, acc, mult, half=, slot=). */
bool rocke_attn_bwd_tile_step_dkv_f32_part(rocke_attn_bwd_tile_step_t* t,
                                           rocke_ir_builder_t* b,
                                           const char* which,
                                           const rocke_attn_bwd_dkdv_acc_t* acc,
                                           rocke_value_t* mult,
                                           int half,
                                           int slot,
                                           rocke_attn_bwd_lds_view_t* out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_BODY_H */
