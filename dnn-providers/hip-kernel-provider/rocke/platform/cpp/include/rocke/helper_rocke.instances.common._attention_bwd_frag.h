/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_frag.h -- C port of the
 * fragment helpers of the attention backward kernels
 * (library/kernels/common/_attention_bwd_frag.py): lane-map facts, the checked
 * C-to-A register relabel, the K-permuted operand load, the LDS fragment
 * loaders and the LDS views.
 *
 *   Python (_attention_bwd_frag.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   IntLaneBuilder / eval_layout        rocke_attn_bwd_eval_layout (the lane map
 *                                       is emitted into a scratch builder and
 *                                       the integer ops are folded)
 *   AtomFacts / atom_facts              rocke_attn_bwd_atom_facts_t / _atom_facts
 *   c_row_runs / c_quad_transposable    rocke_attn_bwd_c_row_runs / _c_quad_transposable
 *   RelabelPlan (+ needs_k_perm)        rocke_attn_bwd_relabel_plan_t (+ _needs_k_perm)
 *   plan_c_to_a / simulate_relabel      rocke_attn_bwd_plan_c_to_a / _simulate_relabel
 *   ir_type                             rocke_attn_bwd_ir_type
 *   emit_c_to_a                         rocke_attn_bwd_emit_c_to_a
 *   FragCoords (emit_base, int_coords)  rocke_attn_bwd_frag_coords_t (+ _emit_base,
 *                                       _int_coords)
 *   operand_coords / kperm_b_coords     rocke_attn_bwd_operand_coords / _kperm_b_coords
 *   load_frag_lds                       rocke_attn_bwd_load_frag_lds
 *   LdsView / DmaSlabView / XorLdsView  rocke_attn_bwd_lds_view_t (kind) + _init each
 *     .align_elems/.index/.load/        rocke_attn_bwd_lds_view_align_elems / _index /
 *     .store/.tr16_load                 _load / _store / _tr16_load
 *     XorLdsView.row_mask/.emit_index   rocke_attn_bwd_xor_view_row_mask / _emit_index
 *   slab_offset                         rocke_attn_bwd_slab_offset
 *   tr16_lane_address / _model_coords   rocke_attn_bwd_tr16_lane_address / _model_coords
 *   check_tr16_coords                   rocke_attn_bwd_check_tr16_coords
 *   Tr16ReadPlan / tr16_read_plan       rocke_attn_bwd_tr16_read_plan_t / _tr16_read_plan
 *   tr16_plan_address / _plan_coords    rocke_attn_bwd_tr16_plan_address / _plan_coords
 *   load_frag_lds_tr16                  rocke_attn_bwd_load_frag_lds_tr16
 *
 * Caps inputs: Python's load_frag_lds_tr16 asks _attention_bwd_caps
 * (check_transpose_read(arch, 64)); this port takes that answer as the
 * explicit `has_tr_read_b64` argument (the caller asks the caps port) and
 * does not ask the caps port itself.
 *
 * emit_c_to_a rounds through _attention_bwd_valu.cast_c_frag
 * (rocke_attn_bwd_cast_c_frag), and slab_offset uses
 * _attention_bwd_lds.DmaSlabLayout.offset (rocke_attn_bwd_dma_slab_layout_*).
 *
 * Byte-identity rules: ops are created in the Python evaluation order (left
 * to right, inner call first), every intermediate bound to a local first. An
 * int argument (rocke_attn_bwd_iv_t with v == NULL) emits exactly what Python
 * emits for it. Errors (the Python ValueErrors) go through rocke_i_set_err,
 * which raises ckc::Error (ROCKE_ERR_VALUE): call these helpers under
 * ckc::guard_builder, as the kernel entry points do. The NULL / false returns
 * after an error are the non-throwing fallbacks.
 *
 * Size limits: every per-lane table is bounded by the backward atoms (wave64,
 * at most 16 slots per fragment, K <= 64). A larger atom is refused with a
 * builder error instead of overflowing.
 *
 * Not ported yet: IntLaneBuilder has no C twin of its own (eval_layout runs
 * the C layout map on a scratch builder and folds the integer ops instead);
 * Tr16ReadPlan has no public constructor (Python builds it only inside
 * tr16_read_plan). Everything else of _attention_bwd_frag.py is ported.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_FRAG_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_FRAG_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h" /* rocke_attn_bwd_iv_t */
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h" /* DmaSlabLayout */
#include "rocke/helper_rocke.instances.common._attention_bwd_valu.h" /* cast_c_frag */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Table bounds (refused with a builder error above them). */
#define ROCKE_ATTN_BWD_FRAG_MAX_LANES 64
#define ROCKE_ATTN_BWD_FRAG_MAX_SLOTS 16
#define ROCKE_ATTN_BWD_FRAG_MAX_K 64

/* ------------------------------------------------- integer lane-map eval */

/* eval_layout: out[(lane * frag_len + slot) * 2 + i] = coordinate i of
 * (lane, slot), for every lane and slot of `layout`. `cap` is the number of
 * ints `out` holds (at least wave_size * frag_len * 2). */
bool rocke_attn_bwd_eval_layout(rocke_ir_builder_t* b,
                                const rocke_layout_map_t* layout,
                                int* out,
                                int cap);

/* -------------------------------------------------------------- atom facts */

typedef struct rocke_attn_bwd_atom_facts
{
    const char* op_id;
    bool a_contig;
    bool b_contig;
    bool affine;
    bool c_is_aT;
    bool ab_same_k;
} rocke_attn_bwd_atom_facts_t;

bool rocke_attn_bwd_atom_facts(rocke_ir_builder_t* b,
                               const rocke_mma_op_t* op,
                               rocke_attn_bwd_atom_facts_t* out);
/* c_row_runs / c_quad_transposable: the answer in *out; false on an error. */
bool rocke_attn_bwd_c_row_runs(rocke_ir_builder_t* b, const rocke_mma_op_t* op, int run, bool* out);
bool rocke_attn_bwd_c_quad_transposable(rocke_ir_builder_t* b, const rocke_mma_op_t* op, bool* out);

/* ------------------------------------------------------- C-to-A relabel */

typedef enum rocke_attn_bwd_relabel_kind
{
    ROCKE_ATTN_BWD_RELABEL_DIRECT = 0,
    ROCKE_ATTN_BWD_RELABEL_XLANE16 = 1
} rocke_attn_bwd_relabel_kind_t;

/* RelabelPlan. slot_src[s] = (tile, cslot) (direct); xlane[m] = (j, sel of the
 * low lane half, sel of the high half) (xlane16); k_perm[k_catalog] =
 * k_logical. */
typedef struct rocke_attn_bwd_relabel_plan
{
    const char* c_op_id;
    const char* a_op_id;
    rocke_attn_bwd_relabel_kind_t kind;
    int c_tiles;
    int c_tile_rows;
    int n_slot_src;
    int slot_src[ROCKE_ATTN_BWD_FRAG_MAX_SLOTS][2];
    int n_xlane;
    int64_t xlane[ROCKE_ATTN_BWD_FRAG_MAX_SLOTS][3];
    int n_k_perm;
    int k_perm[ROCKE_ATTN_BWD_FRAG_MAX_K];
} rocke_attn_bwd_relabel_plan_t;

bool rocke_attn_bwd_relabel_plan_needs_k_perm(const rocke_attn_bwd_relabel_plan_t* plan);

/* plan_c_to_a: prove and return the relabel from c_op's C to a_op's A. */
bool rocke_attn_bwd_plan_c_to_a(rocke_ir_builder_t* b,
                                const rocke_mma_op_t* c_op,
                                const rocke_mma_op_t* a_op,
                                rocke_attn_bwd_relabel_plan_t* out);

/* simulate_relabel: out[(lane * a_frag_len + slot) * 2 + i] = (consumer row,
 * logical K). `cap` ints as in eval_layout. */
bool rocke_attn_bwd_simulate_relabel(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_relabel_plan_t* plan,
                                     const rocke_mma_op_t* c_op,
                                     const rocke_mma_op_t* a_op,
                                     int* out,
                                     int cap);

/* ir_type: "fp16" / "f16" -> f16, "bf16" -> bf16. */
const rocke_type_t* rocke_attn_bwd_ir_type(rocke_ir_builder_t* b, const char* dtype);

/* emit_c_to_a. lane may be NULL (None); bf16_cast is "generic" or
 * "rne_noguard". */
rocke_value_t* rocke_attn_bwd_emit_c_to_a(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_relabel_plan_t* plan,
                                          rocke_value_t* const* c_tiles,
                                          int n_tiles,
                                          const char* dtype,
                                          rocke_value_t* lane,
                                          const char* bf16_cast);

/* ------------------------------------------------- operand coordinates */

typedef enum rocke_attn_bwd_coords_base
{
    ROCKE_ATTN_BWD_COORDS_OPERAND = 0, /* slot 0 of the role's catalog map */
    ROCKE_ATTN_BWD_COORDS_KPERM = 1 /* kperm_b_coords' base program  */
} rocke_attn_bwd_coords_base_t;

/* FragCoords. role is 'a' or 'b'; coordinates are (mn, k). The base callable
 * is the tagged program `base` (operand: layout slot 0; kperm: the C row of
 * slot_src[0] plus t0 * rows, and the B column of slot 0); int_base holds its
 * integer value for every lane. */
typedef struct rocke_attn_bwd_frag_coords
{
    char role;
    int frag_len;
    int wave_size;
    int deltas[ROCKE_ATTN_BWD_FRAG_MAX_SLOTS][2];
    bool kperm;
    rocke_attn_bwd_coords_base_t base;
    const rocke_layout_map_t* layout; /* operand: role map; kperm: B map */
    const rocke_layout_map_t* c_layout; /* kperm only */
    int kperm_t0;
    int kperm_cs0;
    int kperm_rows;
    int int_base[ROCKE_ATTN_BWD_FRAG_MAX_LANES][2];
} rocke_attn_bwd_frag_coords_t;

/* emit_base: the lane-dependent (mn, k) of slot 0. */
bool rocke_attn_bwd_frag_coords_emit_base(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_frag_coords_t* c,
                                          rocke_value_t* lane,
                                          rocke_value_t** mn,
                                          rocke_value_t** k);
/* int_coords: out[slot * 2 + i] for lane `lane` (frag_len * 2 ints). */
void rocke_attn_bwd_frag_coords_int_coords(const rocke_attn_bwd_frag_coords_t* c,
                                           int lane,
                                           int* out);

bool rocke_attn_bwd_operand_coords(rocke_ir_builder_t* b,
                                   const rocke_mma_op_t* op,
                                   char role,
                                   rocke_attn_bwd_frag_coords_t* out);
bool rocke_attn_bwd_kperm_b_coords(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_relabel_plan_t* plan,
                                   const rocke_mma_op_t* c_op,
                                   const rocke_mma_op_t* b_op,
                                   rocke_attn_bwd_frag_coords_t* out);

/* ------------------------------------------------------------- LDS views */

typedef enum rocke_attn_bwd_view_kind
{
    ROCKE_ATTN_BWD_VIEW_PLAIN = 0, /* LdsView     */
    ROCKE_ATTN_BWD_VIEW_DMA_SLAB = 1, /* DmaSlabView */
    ROCKE_ATTN_BWD_VIEW_XOR = 2 /* XorLdsView  */
} rocke_attn_bwd_view_kind_t;

/* LdsView and its subclasses. shift NULL is None; chunk is used by the XOR
 * view only. */
typedef struct rocke_attn_bwd_lds_view
{
    rocke_attn_bwd_view_kind_t kind;
    rocke_value_t* smem;
    int64_t base;
    int64_t pitch;
    int64_t rows;
    int scale;
    rocke_value_t* shift;
    int chunk;
} rocke_attn_bwd_lds_view_t;

/* The dataclass constructors (fields in dataclass order; __post_init__
 * checks). A Python ValueError raises ckc::Error through rocke_i_set_err;
 * call under ckc::guard_builder. */
bool rocke_attn_bwd_lds_view_init(rocke_attn_bwd_lds_view_t* v,
                                  rocke_ir_builder_t* b,
                                  rocke_value_t* smem,
                                  int64_t base,
                                  int64_t pitch,
                                  int64_t rows,
                                  int scale,
                                  rocke_value_t* shift);
bool rocke_attn_bwd_dma_slab_view_init(rocke_attn_bwd_lds_view_t* v,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* smem,
                                       int64_t base,
                                       int64_t pitch,
                                       int64_t rows,
                                       int scale,
                                       rocke_value_t* shift);
bool rocke_attn_bwd_xor_lds_view_init(rocke_attn_bwd_lds_view_t* v,
                                      rocke_ir_builder_t* b,
                                      rocke_value_t* smem,
                                      int64_t base,
                                      int64_t pitch,
                                      int64_t rows,
                                      int scale,
                                      rocke_value_t* shift,
                                      int chunk);

int rocke_attn_bwd_lds_view_align_elems(const rocke_attn_bwd_lds_view_t* v);
rocke_value_t* rocke_attn_bwd_lds_view_index(rocke_ir_builder_t* b,
                                             const rocke_attn_bwd_lds_view_t* v,
                                             rocke_attn_bwd_iv_t row,
                                             rocke_attn_bwd_iv_t col);
rocke_value_t* rocke_attn_bwd_lds_view_load(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_lds_view_t* v,
                                            rocke_attn_bwd_iv_t row,
                                            rocke_attn_bwd_iv_t col,
                                            const rocke_type_t* dtype,
                                            int n);
void rocke_attn_bwd_lds_view_store(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_lds_view_t* v,
                                   rocke_attn_bwd_iv_t row,
                                   rocke_attn_bwd_iv_t col,
                                   rocke_value_t* value,
                                   int n);
rocke_value_t* rocke_attn_bwd_lds_view_tr16_load(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_lds_view_t* v,
                                                 rocke_attn_bwd_iv_t row,
                                                 rocke_attn_bwd_iv_t col,
                                                 const rocke_type_t* dtype);

/* XorLdsView.row_mask (host) and .emit_index (a runtime row). */
int64_t rocke_attn_bwd_xor_view_row_mask(const rocke_attn_bwd_lds_view_t* v, int64_t row);
rocke_value_t* rocke_attn_bwd_xor_view_emit_index(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_lds_view_t* v,
                                                  rocke_value_t* row,
                                                  rocke_attn_bwd_iv_t col);

/* slab_offset (host): element offset of (row, col) in a rows x cols 16-bit DMA
 * slab tile. Returns false (error recorded) outside the tile. */
bool rocke_attn_bwd_slab_offset(
    rocke_ir_builder_t* b, int64_t rows, int64_t cols, int64_t row, int64_t col, int64_t* out);

/* ---------------------------------------------------------- LDS loaders */

typedef enum rocke_attn_bwd_storage
{
    ROCKE_ATTN_BWD_K_INNER = 0, /* smem[mn][k] */
    ROCKE_ATTN_BWD_K_OUTER = 1 /* smem[k][mn] */
} rocke_attn_bwd_storage_t;

/* load_frag_lds. Exactly one of `smem` (a 2-D allocation) and `view` (an
 * LdsView) is non-NULL. */
rocke_value_t* rocke_attn_bwd_load_frag_lds(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_frag_coords_t* coords,
                                            rocke_value_t* smem,
                                            const rocke_attn_bwd_lds_view_t* view,
                                            rocke_value_t* lane,
                                            const char* dtype,
                                            rocke_attn_bwd_iv_t mn0,
                                            rocke_attn_bwd_iv_t k0,
                                            rocke_attn_bwd_storage_t storage,
                                            int align);

/* --------------------------------------------------------- transpose reads */

/* tr16_lane_address: (k, mn) offset lane `lane` supplies for read t. */
void rocke_attn_bwd_tr16_lane_address(int lane, int t, int kpl, int* k, int* mn);
/* tr16_model_coords: out[slot * 2 + i] ((kpl / 4) * 4 slots). */
void rocke_attn_bwd_tr16_model_coords(int lane, int kpl, int* out);
/* check_tr16_coords: elements per lane, or 0 on an error. */
int rocke_attn_bwd_check_tr16_coords(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_frag_coords_t* coords);

typedef enum rocke_attn_bwd_tr16_kind
{
    ROCKE_ATTN_BWD_TR16_CATALOG = 0,
    ROCKE_ATTN_BWD_TR16_STACKED = 1
} rocke_attn_bwd_tr16_kind_t;

/* Tr16ReadPlan. Only the catalog plans (1 or 2 reads) and the stacked plan
 * exist, so k_offsets holds at most 2 entries. */
typedef struct rocke_attn_bwd_tr16_read_plan
{
    rocke_attn_bwd_tr16_kind_t kind;
    int g_stride;
    int n_offsets;
    int k_offsets[2];
} rocke_attn_bwd_tr16_read_plan_t;

void rocke_attn_bwd_tr16_plan_address(
    int lane, int t, const rocke_attn_bwd_tr16_read_plan_t* plan, int* k, int* mn);
/* out[slot * 2 + i] (n_offsets * 4 slots). */
void rocke_attn_bwd_tr16_plan_coords(int lane,
                                     const rocke_attn_bwd_tr16_read_plan_t* plan,
                                     int* out);
bool rocke_attn_bwd_tr16_read_plan(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_frag_coords_t* coords,
                                   rocke_attn_bwd_tr16_read_plan_t* out);

/* load_frag_lds_tr16. has_tr_read_b64 is check_transpose_read(arch, 64)'s
 * answer (true: legal). view NULL is the non-LdsView Python error. */
rocke_value_t* rocke_attn_bwd_load_frag_lds_tr16(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_frag_coords_t* coords,
                                                 const rocke_attn_bwd_lds_view_t* view,
                                                 rocke_value_t* lane,
                                                 const char* dtype,
                                                 bool has_tr_read_b64,
                                                 rocke_attn_bwd_iv_t mn0,
                                                 rocke_attn_bwd_iv_t k0);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_FRAG_H */
