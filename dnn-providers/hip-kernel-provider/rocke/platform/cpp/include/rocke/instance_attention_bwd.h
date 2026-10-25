/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd.h
 *
 * C port of the spec, policy and static-legality half of
 * library/kernels/common/attention_bwd.py (the layout-general SDPA backward
 * main kernel): everything the main builder needs to make Python's choices.
 * Host-side only; nothing here emits IR.
 *
 *   Python (attention_bwd.py)            C (this header)
 *   ----------------------------------   --------------------------------------
 *   AttnBwdSpec (+ __post_init__)        rocke_attn_bwd_spec_t
 *                                        (+ _spec_default, _spec_init)
 *   AttnBwdSpec.resolved(arch)           rocke_attn_bwd_spec_resolved
 *   AttnBwdSpec.non_default_fields /     rocke_attn_bwd_spec_kernel_name
 *     kernel_name(stage)
 *   AttnBwdSpec.aux_spec(stage, ...)     rocke_attn_bwd_spec_aux_spec
 *   BwdArchFacts / attn_bwd_arch_facts   rocke_attn_bwd_arch_facts_t /
 *     (+ legal_* properties)             rocke_attn_bwd_arch_facts
 *   attn_bwd_default_geometry            rocke_attn_bwd_default_geometry
 *   _policy_agpr_alloc                   rocke_attn_bwd_policy_agpr_alloc
 *   lds_request / lds_plan /             rocke_attn_bwd_lds_request /
 *     lds_footprint_bytes                _lds_plan / _lds_footprint_bytes
 *   register_estimates                   rocke_attn_bwd_register_estimates
 *   agpr_copy_budget                     rocke_attn_bwd_agpr_copy_budget
 *   check_counted_sync                   rocke_attn_bwd_check_counted_sync
 *   validate_attn_bwd_spec               rocke_attn_bwd_validate_spec
 *   static_unit_mults                    rocke_attn_bwd_static_unit_mults
 *   attn_bwd_params / KERNARG_FORMATS    rocke_attn_bwd_params
 *
 * Spec encoding. Strings carry the Python spellings (NULL is None for an
 * Optional[str]). An Optional[int] or Optional[bool] is ROCKE_ATTN_BWD_NONE
 * for None, so an explicit 0 (refused by Python) stays distinct; a bool is 0
 * or 1. A warp grid is {has, rows, cols}. agpr_alloc is the
 * rocke_attn_bwd_agpr_alloc_t of the caps header. ptr_noalias is NULL for
 * Python's False, else the string. lds_swizzle is stored canonical in the
 * struct (has_lds_swizzle false is None); _spec_init canonicalises it in
 * place, as __post_init__ does, and refuses a canonical text of
 * ROCKE_ATTN_BWD_SWIZZLE_CAP characters or more. Python's type errors (a
 * non-bool flag, a non-int count) have no C spelling.
 *
 * Validation trusts the specs Python already validated for the parts that
 * stay in Python: the recorded resource-gate refusals (_attention_bwd_refusals,
 * Python-only) are not consulted, i.e. this mirrors
 * validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False).
 *
 * BwdTileGeometry.from_spec(...).check() (body unit) is mirrored inside
 * validate, where Python calls it (loop_sync = "counted", dkv_cell_vec != 1,
 * stats_vec != 1), except its lane-map proofs (atom_facts, the C-to-A relabel
 * plan and the tr16 read plans of _attention_bwd_frag / _gemm): those hold for
 * every 16x16 backward atom and grid the spec admits (the parity pair
 * attention_bwd_layout_emit checks the Python side of that claim; it is a
 * probe, not a proof). The stats_vec / quad-transpose checks evaluate the
 * op's catalog C lane map on ints, as _attention_bwd_frag.eval_layout does.
 *
 * Departures from the phase-2 CONVENTIONS, on purpose:
 * - Errors. Nothing here builds IR, so the entry points take a
 *   rocke_attn_bwd_err_t out-parameter and return rocke_status_t (the
 *   internals throw ckc::Error, caught at each extern "C" boundary) rather
 *   than setting a builder's sticky error. The one IR emitter, emit_lds_dma
 *   (caps header), uses ckc::guard_status on its builder.
 * - Optional int. ROCKE_ATTN_BWD_NONE (INT32_MIN) is None, not 0, so None and
 *   an explicit 0 stay distinct (asm_tail_nop, waves_per_eu, ...).
 *
 * Mirrored constants. STAGE_ORDERS, G4_SLICINGS, COPY_ORDERS (body), SCHEDS
 * and the deferred-knob texts (sched), COUNTED_ISSUES / COUNTED_VMCNT_MAX
 * (counted) and BF16_CASTS / BWD_EXP2S / STATS_VECS (valu) are file-static
 * copies in attention_bwd.cpp until those modules are ported; the unit that
 * ports each module must replace the copy with its export.
 *
 * Not ported (host-only or other units): INERT_KNOBS, DECLARATION_ONLY,
 * NOT_YET_EFFECTIVE_KNOBS (sweep bookkeeping), AttnBwdSpec.dq_spec (the split
 * dQ kernel's spec, P8), _not_built / _main_noalias and the builder itself
 * (_declare_main through build_attn_bwd_main, P7), the lane-map proofs of
 * BwdTileGeometry.check (P4 / P7), BwdLdsRequest.from_geometry.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_H
#define ROCKE_INSTANCE_ATTENTION_BWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_lds.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_tiles.h"
#include "rocke/instance_attention_bwd_aux.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_ATTN_BWD_SWIZZLE_CAP 512

/* An Optional[tuple[int, int]] warp grid. */
typedef struct rocke_attn_bwd_opt_grid
{
    bool has;
    int rows;
    int cols;
} rocke_attn_bwd_opt_grid_t;

/* AttnBwdSpec, in Python's field order. */
typedef struct rocke_attn_bwd_spec
{
    int head_size;
    const char* dtype;
    const char* seq_mode;
    const char* dkv_mode;
    int stage_vec;
    const char* mask_class;
    const char* dq_mode;
    const char* static_scale;
    int waves; /* Optional */
    int block_m; /* Optional */
    int block_n; /* Optional */
    int block_k4; /* Optional */
    rocke_attn_bwd_opt_grid_t warp_grid_g4;
    const char* atom_g02; /* Optional */
    const char* atom_g13; /* Optional */
    const char* atom_g4; /* Optional */
    const char* pt_route; /* Optional */
    const char* kv_residency; /* Optional */
    const char* kt_source; /* Optional */
    const char* transpose_source; /* Optional */
    int ring_depth; /* Optional */
    const char* global_path; /* Optional */
    bool has_lds_swizzle;
    char lds_swizzle[ROCKE_ATTN_BWD_SWIZZLE_CAP];
    const char* sched; /* Optional */
    int setprio; /* Optional bool */
    int waves_per_eu; /* Optional */
    int edge_tiles; /* Optional bool */
    int head_pack; /* Optional bool */
    int acc_in_lds; /* Optional bool */
    const char* ws_layout;
    bool xcd_remap;
    rocke_attn_bwd_agpr_alloc_t agpr_alloc;
    int s_in_agpr; /* Optional bool */
    const char* agpr_route; /* Optional ("attr" is spelled None) */
    int asm_tail_nop; /* Optional */
    const char* asm_cluster;
    bool asm_load_hoist;
    const char* scheduler_strategy; /* Optional */
    rocke_attn_bwd_opt_grid_t warp_grid_g02;
    rocke_attn_bwd_opt_grid_t warp_grid_g13;
    const char* stage_order;
    const char* loop_sync;
    const char* step_order;
    const char* dkv_epilogue;
    int dkv_store_vec;
    const char* dkv_atomic_guard;
    const char* dkv_sink;
    const char* lds_order;
    const char* qdo_stage;
    const char* stats_load;
    const char* dq_issue;
    const char* g4_slicing;
    const char* ptr_noalias; /* NULL: False */
    bool q_rotate;
    bool lb_interleave;
    bool dkv_slab_hoist;
    bool interior_clamp;
    const char* barrier_merge;
    const char* ring_issue;
    const char* counted_issue;
    bool lb_mirror;
    const char* bf16_cast;
    const char* bwd_exp2;
    bool stats_fma;
    int stats_vec;
    bool worklist;
    const char* ws_bounds;
    int dkv_cell_vec;
    int narrow_vec;
    bool slab_carry;
    const char* name;
} rocke_attn_bwd_spec_t;

/* Python's defaults; the caller sets head_size. */
rocke_attn_bwd_spec_t rocke_attn_bwd_spec_default(void);

/* AttnBwdSpec.__post_init__, in place (canonical lds_swizzle, agpr_route
 * "attr" -> None). */
rocke_status_t rocke_attn_bwd_spec_init(rocke_attn_bwd_spec_t* spec, rocke_attn_bwd_err_t* err);

/* Set spec->lds_swizzle from text (NULL: None) before _spec_init. */
rocke_status_t rocke_attn_bwd_spec_set_lds_swizzle(rocke_attn_bwd_spec_t* spec,
                                                   const char* text,
                                                   rocke_attn_bwd_err_t* err);

/* AttnBwdSpec.resolved(arch) into *out (out may alias spec). */
rocke_status_t rocke_attn_bwd_spec_resolved(const rocke_attn_bwd_spec_t* spec,
                                            const char* arch,
                                            rocke_attn_bwd_spec_t* out,
                                            rocke_attn_bwd_err_t* err);

/* AttnBwdSpec.kernel_name(stage), stage "main" or "dq". */
rocke_status_t rocke_attn_bwd_spec_kernel_name(const rocke_attn_bwd_spec_t* spec,
                                               const char* stage,
                                               char* out,
                                               size_t out_cap,
                                               rocke_attn_bwd_err_t* err);

/* AttnBwdSpec.aux_spec(stage, rows_per_cta=, convert_vec=), stage "prep" or
 * "convert"; rows_per_cta ROCKE_ATTN_BWD_NONE is None. The result is checked
 * with rocke_attn_bwd_aux_spec_validate (AttnBwdAuxSpec.__post_init__). */
rocke_status_t rocke_attn_bwd_spec_aux_spec(const rocke_attn_bwd_spec_t* spec,
                                            const char* stage,
                                            int rows_per_cta,
                                            int convert_vec,
                                            rocke_attn_bwd_aux_spec_t* out,
                                            rocke_attn_bwd_err_t* err);

/* BwdArchFacts. matrix_path "mfma" | "wmma". */
typedef struct rocke_attn_bwd_arch_facts
{
    const char* arch;
    int wave_size;
    const char* matrix_path;
    int lds_capacity_bytes;
    int arch_vgprs;
    int agprs;
    bool has_tr_read;
    bool has_lds_dma;
    bool wide_k_atom;
} rocke_attn_bwd_arch_facts_t;

rocke_status_t rocke_attn_bwd_arch_facts(const char* arch,
                                         rocke_attn_bwd_arch_facts_t* out,
                                         rocke_attn_bwd_err_t* err);

/* The legal_* properties of BwdArchFacts: count of static entries in out
 * (at most 4). legal_waves / legal_ring_depths are ints; legal_atoms /
 * legal_transpose_sources / legal_global_paths strings. */
int rocke_attn_bwd_facts_legal_waves(const rocke_attn_bwd_arch_facts_t* f, int out[4]);
int rocke_attn_bwd_facts_legal_atoms(const rocke_attn_bwd_arch_facts_t* f, const char* out[4]);
int rocke_attn_bwd_facts_legal_transpose_sources(const rocke_attn_bwd_arch_facts_t* f,
                                                 const char* out[4]);
int rocke_attn_bwd_facts_legal_global_paths(const rocke_attn_bwd_arch_facts_t* f,
                                            const char* out[4]);
int rocke_attn_bwd_facts_legal_ring_depths(const rocke_attn_bwd_arch_facts_t* f, int out[4]);

/* attn_bwd_default_geometry(arch, head_size, dtype, stage_vec, seq_mode): the
 * start point as a spec whose geometry fields carry the dict's values (every
 * other field is the default; scheduler_strategy / waves_per_eu NULL /
 * ROCKE_ATTN_BWD_NONE where the dict holds None). */
rocke_status_t rocke_attn_bwd_default_geometry(const char* arch,
                                               int head_size,
                                               const char* dtype,
                                               int stage_vec,
                                               const char* seq_mode,
                                               rocke_attn_bwd_spec_t* out,
                                               rocke_attn_bwd_err_t* err);

/* _policy_agpr_alloc(spec, arch): (0, 0) or None. */
rocke_status_t rocke_attn_bwd_policy_agpr_alloc(const rocke_attn_bwd_spec_t* spec,
                                                const char* arch,
                                                rocke_attn_bwd_agpr_alloc_t* out,
                                                rocke_attn_bwd_err_t* err);

/* lds_request(spec, arch): the request borrows spec->lds_swizzle, so the
 * spec must outlive it (and any plan made from it). */
rocke_status_t rocke_attn_bwd_lds_request(const rocke_attn_bwd_spec_t* spec,
                                          const char* arch,
                                          rocke_attn_bwd_lds_request_t* out,
                                          rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_lds_plan(const rocke_attn_bwd_spec_t* spec,
                                       const char* arch,
                                       rocke_attn_bwd_lds_plan_t* out,
                                       rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_lds_footprint_bytes(const rocke_attn_bwd_spec_t* spec,
                                                  const char* arch,
                                                  int* out,
                                                  rocke_attn_bwd_err_t* err);

/* register_estimates: {"arch_vgpr", "agpr"}. */
rocke_status_t rocke_attn_bwd_register_estimates(const rocke_attn_bwd_spec_t* spec,
                                                 const char* arch,
                                                 int* arch_vgpr,
                                                 int* agpr,
                                                 rocke_attn_bwd_err_t* err);

/* agpr_copy_budget: {"reads", "writes", "moves"}. */
rocke_status_t rocke_attn_bwd_agpr_copy_budget(const rocke_attn_bwd_spec_t* spec,
                                               const char* arch,
                                               int* reads,
                                               int* writes,
                                               int* moves,
                                               rocke_attn_bwd_err_t* err);

rocke_status_t rocke_attn_bwd_check_counted_sync(const rocke_attn_bwd_spec_t* spec,
                                                 const char* arch,
                                                 rocke_attn_bwd_err_t* err);

/* validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False): the
 * resolved spec into *out (out may alias spec). */
rocke_status_t rocke_attn_bwd_validate_spec(const rocke_attn_bwd_spec_t* spec,
                                            const char* arch,
                                            rocke_attn_bwd_spec_t* out,
                                            rocke_attn_bwd_err_t* err);

/* BwdTileGeometry.from_spec(spec, arch).check() of a resolved spec (body
 * unit), without its lane-map proofs (atom_facts, the C-to-A relabel plan
 * and K permutation, the tr16 read plans): the checks validate runs where
 * Python builds the geometry. */
rocke_status_t rocke_attn_bwd_tile_geometry_check(const rocke_attn_bwd_spec_t* spec,
                                                  const char* arch,
                                                  rocke_attn_bwd_err_t* err);

/* static_unit_mults(static_scale, dkv_mode): bits of {ds_mult = 1,
 * dk_mult = 2, dq_mult = 4} into *mask. */
#define ROCKE_ATTN_BWD_UNIT_DS_MULT 1
#define ROCKE_ATTN_BWD_UNIT_DK_MULT 2
#define ROCKE_ATTN_BWD_UNIT_DQ_MULT 4
rocke_status_t rocke_attn_bwd_static_unit_mults(const char* static_scale,
                                                const char* dkv_mode,
                                                unsigned* mask,
                                                rocke_attn_bwd_err_t* err);

/* attn_bwd_params(stage): ordered (name, struct format) kernargs, stage in
 * ("main", "prep", "convert", "reduce", "dq"); formats are "Q", "q", "f",
 * "i" (KERNARG_FORMATS). Up to cap static entries; *n_out the count. */
#define ROCKE_ATTN_BWD_PARAMS_MAX 80
rocke_status_t rocke_attn_bwd_params(const char* stage,
                                     const char* names[ROCKE_ATTN_BWD_PARAMS_MAX],
                                     const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX],
                                     int* n_out,
                                     rocke_attn_bwd_err_t* err);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>

namespace rocke_attn_bwd
{
const rocke_attn_bwd_arch_facts_t& arch_facts(const std::string& arch);
struct RegEstimate
{
    int arch_vgpr;
    int agpr;
};
RegEstimate register_estimates(const rocke_attn_bwd_spec_t& spec, const std::string& arch);
int lds_footprint_bytes(const rocke_attn_bwd_spec_t& spec, const std::string& arch);
void spec_init(rocke_attn_bwd_spec_t& spec);
rocke_attn_bwd_spec_t resolved(const rocke_attn_bwd_spec_t& spec, const std::string& arch);
rocke_attn_bwd_spec_t validate_spec(const rocke_attn_bwd_spec_t& spec, const std::string& arch);
std::string kernel_name(const rocke_attn_bwd_spec_t& spec, const std::string& stage);
rocke_attn_bwd_lds_request_t lds_request(const rocke_attn_bwd_spec_t& spec,
                                         const std::string& arch);
rocke_attn_bwd_lds_plan_t lds_plan(const rocke_attn_bwd_spec_t& spec, const std::string& arch);
/* check_stats_vec_layout(op, width) (throws ckc::ValueError like Python),
 * whether it passes, and c_quad_transposable(op). All three evaluate the
 * op's catalog C lane map on ints (_attention_bwd_frag.eval_layout). */
void check_stats_vec_layout(const rocke_mma_op_t* op, int width);
bool stats_vec_layout_ok(const rocke_mma_op_t* op, int width);
bool c_quad_transposable(const rocke_mma_op_t* op);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_BWD_H */
