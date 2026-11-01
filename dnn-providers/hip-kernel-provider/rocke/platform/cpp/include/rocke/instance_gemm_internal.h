/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_internal.h -- PRIVATE shared state + phase-function
 * contract for the C99 port of build_universal_gemm
 * (rocke/instances/common/gemm_universal.py).
 *
 * WHY THIS HEADER EXISTS.
 *   build_universal_gemm() in Python is a ~1100-line function whose body is a
 *   stack of NESTED CLOSURES (emit_load_phase, _emit_frag_smem_load,
 *   _emit_wmma_phase, emit_mfma_phase, emit_compute_and_epilogue,
 *   _emit_kloop_db / _simple / _prefetch, _emit_epilogue and their inner
 *   helpers). Every closure captures the SAME enclosing-function locals: the
 *   builder, the resolved MMA op, the param Values (A/B/C/M/N/K/strides),
 *   every derived geometry constant, the SSA constants (c0, c_block_m, ...),
 *   the LDS-plan flags + smem handles, the accumulator iter-args, the swizzle
 *   parameters, the DirectToLDS plumbing, and the K-loop result vector.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function that takes a POINTER to one shared context
 *   struct, rocke_gemm_build_ctx_t, which holds EXACTLY the set of variables the
 *   closures shared. The driver (rocke_build_universal_gemm in
 *   rocke/instance_gemm_universal.h) populates the ctx in the same order the
 *   Python prologue computes its locals, then calls the phase functions.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds
 *   to. It is DESIGNED TO BE COMPLETE: every local/closured variable the
 *   Python body shares across phases is a field here. A body agent implementing
 *   a phase .c file MUST be able to read/write only ctx fields and call the
 *   prototypes below WITHOUT editing this header -- adding a field here forces
 *   every other body TU to recompile and risks a merge conflict on the shared
 *   surface. If a phase genuinely needs a value not present, that is a design
 *   bug in this header to be fixed once, deliberately, not patched per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `block_m_off`
 *   -> `ctx->block_m_off`; Python `_db` -> `ctx->db`; the leading-underscore
 *   Python "private local" prefix is dropped since C has no such convention).
 *   Phase functions mirror the Python closure / module-helper names with a
 *   `rocke_gemm_` prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by
 * the instance_gemm_*.c translation units. Public callers use
 * rocke/instance_gemm_universal.h.
 */
#ifndef ROCKE_INSTANCE_GEMM_INTERNAL_H
#define ROCKE_INSTANCE_GEMM_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_mmaop_t, rocke_archtarget_t */
#include "rocke/helper_rocke.helpers.schedule.h" /* rocke_hotloop_inst_list_t, policy */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* rocke_tensor_view_t, descriptors */
#include "rocke/instance_gemm_mn_operand.h"
#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The per-warp accumulator iter-arg count is bounded: mfmas_per_warp_m *
 * mfmas_per_warp_n. For the covered tile space (<=256/(warp*warp_tile)) this
 * never exceeds a small constant; 64 is generous (e.g. 256x256 / (1*16) =
 * 16x16 = 256 is rejected by block-size cap, real configs are <=16). We size
 * the inline arrays to cover every buildable config plus headroom. */
#define ROCKE_GEMM_MAX_ACCS 256

/* DTLA pass arrays: chunks_total / block_size is small (a handful of passes);
 * 64 covers every legal tile. */
#define ROCKE_GEMM_MAX_DTL_PASSES 64

/* The fused operand prologue's per-kernel emitter (gemm_prologue.cpp; Python
 * PrologueEmitter). Owned by rocke_build_universal_gemm. */
typedef struct rocke_gemm_pro_emitter rocke_gemm_pro_emitter_t;

/* ===================================================================== *
 *  rocke_gemm_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local that
 *  the Python closures capture. Grouped by the Python prologue's phases; field
 *  order follows the order the Python computes them so the populate routine
 *  reads top-to-bottom against the source.
 * ===================================================================== */
typedef struct rocke_gemm_build_ctx
{
    /* ---- inputs / resolved environment (build_universal_gemm args + setup) -- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`                */
    const rocke_gemm_universal_spec_t* spec; /* the UniversalGemmSpec `spec`     */
    const char* arch; /* `arch` (NULL-normalised "gfx950")*/
    const rocke_archtarget_t* target; /* ArchTarget.from_gfx(arch)        */
    const rocke_mmaop_t* op; /* _resolve_mma_op(spec, arch)      */
    const rocke_type_t* storage_dtype; /* _storage_dtype(spec)             */
    bool is_wmma; /* op->family == "wmma"             */

    /* ---- kernel params (Values) -- */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* C;
    rocke_value_t* M;
    rocke_value_t* N;
    rocke_value_t* K;
    /* batched=True only (NULL otherwise) */
    rocke_value_t* stride_a;
    rocke_value_t* stride_b;
    rocke_value_t* stride_c;
    /* batched && active_tile_skip only (NULL otherwise) */
    rocke_value_t* sorted_token_ids;
    rocke_value_t* slot_size_p;
    /* trait.streamk only (NULL otherwise): ready flags (i32) + partials (f32) */
    rocke_value_t* sk_flags;
    rocke_value_t* sk_part;

    /* ---- per-lane MMA fragment widths (_atom_frag_lengths(op)) -- */
    int a_per_lane;
    int b_per_lane;
    int c_per_lane;

    /* ---- block tile dims (aliases of spec->tile) -- */
    int block_m; /* t.tile_m */
    int block_n; /* t.tile_n */
    int block_k; /* t.tile_k */

    /* ---- common geometry constants (SSA) -- */
    rocke_value_t* c0; /* const_i32(0)             */
    /* ---- split-K K-slice bounds (_split_k / _is_split_k / k_lo / k_hi) -- *
     * Python computes these right after c0 and before c_wave; the C populate
     * follows the same order so the SSA value ids line up. */
    int split_k; /* spec->trait.split_k                              */
    bool is_split_k; /* split_k > 1                                      */
    rocke_value_t* k_lo; /* slice base: c0 (non-split) or sgpr(z*ks)         */
    rocke_value_t* k_upper; /* slice end: K (non-split, Python k_hi==None) or   */
    /* sgpr(k_lo+ks). Python `_k_upper`.                */
    rocke_value_t* c_wave; /* const_i32(wave_size)     */
    rocke_value_t* c_warps_n; /* const_i32(t.warp_n)      */
    rocke_value_t* c_block_m; /* const_i32(block_m)       */
    rocke_value_t* c_block_n; /* const_i32(block_n)       */
    rocke_value_t* c_block_k; /* const_i32(block_k)       */

    /* ---- thread / warp / lane decode (SSA) -- */
    rocke_value_t* tid; /* thread_id_x()            */
    rocke_value_t* warp_id; /* tid / wave               */
    rocke_value_t* warp_m_idx; /* warp_id / warp_n         */
    rocke_value_t* warp_n_idx; /* warp_id % warp_n         */
    rocke_value_t* lane; /* tid % wave               */

    /* ---- compv4/compv3 schedule-hint policy (_sched_hints) -- */
    /* spec.trait.emit_sched_hints if set, else (arch != "gfx950").
     * Gates the per-cluster s_setprio/sched_barrier fences AND the
     * two-stage sched_group_barrier HotLoop interleave. */
    bool sched_hints;

    /* ---- LDS XOR swizzle (lds_swizzle) -- */
    bool swz; /* _SWZ = spec.trait.lds_swizzle             */
    int swz_r; /* CK_SWZ_R   (resolved)                     */
    int swz_w; /* CK_SWZ_W                                  */
    int swz_l; /* CK_SWZ_L                                  */
    rocke_value_t* c_swr; /* const_i32(swz_r)         (_c_swr)         */
    rocke_value_t* c_swmod; /* const_i32(1 << swz_w)    (_c_swmod)       */
    rocke_value_t* c_swl; /* const_i32(swz_l)         (_c_swl)         */

    /* ---- batch-axis pointer offsets (SSA; c0 in non-batched mode) -- */
    rocke_value_t* batch_idx; /* to_sgpr_u32(block_id_z())  (batched only) */
    rocke_value_t* batch_off_a;
    rocke_value_t* batch_off_b;
    rocke_value_t* batch_off_c;

    /* ---- per-CTA tile origins (SGPR-pinned SSA) -- */
    rocke_value_t* block_m_off; /* row tile base (chiplet-swizzle aware)     */
    rocke_value_t* block_n_off; /* col tile base                             */

    /* ---- interwave_setprio MFMA-cluster priority bookends -- */
    bool interwave; /* _interwave = trait.interwave_setprio */

    /* ---- AB LDS double-buffer plan (_ab_lds_plan) -- */
    bool prefetch; /* _prefetch = trait.dtl_prefetch            */
    bool db; /* _db   (compv4 software-pipelined DB)       */
    bool two_buf; /* _two_buf = prefetch || db                 */
    int A_LDS_M; /* _A_LDS_M = (two_buf?2:1)*block_m          */
    int B_LDS_N; /* _B_LDS_N = (two_buf?2:1)*block_n          */
    int lds_pad; /* _lds_pad (non-DTL lds_k_pad else 0)       */
    int lds_k; /* _lds_k = block_k + lds_pad                */
    rocke_value_t* A_smem; /* smem_alloc A_smem                         */
    rocke_value_t* B_smem; /* smem_alloc B_smem                         */
    /* ---- alias-scoped DTL ring (trait.dtl_stages >= 2) -- *
     * one A and one B allocation per stage ("A_smem_s{j}" / "B_smem_s{j}");
     * A_smem / B_smem alias stage 0. */
    int stages; /* _stages (0 = historical body)            */
    rocke_value_t* A_stg[4];
    rocke_value_t* B_stg[4];
    bool swap_ab; /* _swap_ab = trait.mfma_swap_ab             */
    /* the persistent staged body ran the epilogue per tile */
    bool staged_persistent_done;

    /* ---- per-warp MFMA tile counts -- */
    int mfmas_m; /* t.mfmas_per_warp_m                        */
    int mfmas_n; /* t.mfmas_per_warp_n                        */
    int k_atoms; /* t.k_atoms_per_tile_k                      */
    /* _mo_perm: the MFMA issue order of a K atom (mfma_serpentine = order 1,
     * else mfma_order), slot -> accumulator index, for even / odd K atoms;
     * the identity for the default order (_mo_ilv = mo_perm[0]) */
    int mo_perm[2][ROCKE_GEMM_MAX_ACCS];

    /* ---- accumulators -- */
    rocke_value_t* acc_init; /* _emit_zero_acc_op(b, op)          */
    /* The `accs` list: one zero-acc iter-arg per (mi, ni) warp MFMA tile.
     * Names "acc_m{mi}_n{ni}" + the shared acc_init init Value. */
    const char* acc_names[ROCKE_GEMM_MAX_ACCS];
    rocke_value_t* acc_inits[ROCKE_GEMM_MAX_ACCS]; /* all == acc_init           */
    int num_accs; /* mfmas_m * mfmas_n                 */

    /* ---- global -> LDS coalesced copy plan -- */
    int threads; /* spec.block_size                           */
    int load_vec; /* _choose_load_vec(spec)                    */
    int a_total; /* block_m * block_k                         */
    int b_total; /* block_n * block_k                         */
    int a_vec_total; /* a_total / load_vec                        */
    int b_vec_total; /* b_total / load_vec                        */
    int a_vecs_per_thread; /* a_vec_total / threads                     */
    int b_vecs_per_thread; /* b_vec_total / threads                     */
    rocke_value_t* c_threads; /* const_i32(threads)                */
    rocke_value_t* c_load_vec; /* const_i32(load_vec)               */
    rocke_value_t* c_block_k_div_vec; /* const_i32(block_k / load_vec)     */

    /* ---- CK-Tile data views (host-side; hold SSA bases) -- */
    rocke_tensor_view_t a_view; /* make_global_view(A, (1,K,1))      */
    rocke_tensor_view_t b_view; /* make_global_view(Bp,(1,K,1))      */
    rocke_tensor_descriptor_t a_lds_desc; /* packed or with_strides         */
    rocke_tensor_descriptor_t b_lds_desc;
    rocke_tensor_view_t a_lds_view; /* LDS view over A_smem              */
    rocke_tensor_view_t b_lds_view; /* LDS view over B_smem              */

    /* ---- DirectToLDS (DTLA/DTLB) plumbing (direct_to_lds only) -- */
    bool dtl; /* spec.trait.direct_to_lds                  */
    int dtl_dwords; /* _DTL_DWORDS = 4                           */
    int dtl_halves; /* _DTL_HALVES (0 when a chunk is under one element) */
    int dtl_bytes_per_lane; /* _DTL_BYTES_PER_LANE = dtl_dwords*4        */
    int dtl_elem_bytes; /* _dtl_elem_bytes                           */
    int dtl_elem_log2; /* log2(dtl_elem_bytes)                      */
    bool dtl_sub_elem; /* _dtl_sub_elem: chunk smaller than an element */
    int dtl_a_chunks; /* (block_m*block_k*elem_bytes)/bytes_per_lane */
    int dtl_b_chunks; /* (block_n*block_k*elem_bytes)/bytes_per_lane */
    int dtl_a_passes; /* ceil(dtl_a_chunks / block_size)          */
    int dtl_b_passes; /* ceil(dtl_b_chunks / block_size)          */
    int dtl_pass_bytes; /* block_size * dtl_bytes_per_lane           */
    int dtl_chunks_per_row; /* block_k*elem_bytes / bytes_per_lane       */
    rocke_value_t* dtl_big_bytes; /* const_i32(0x7FFF0000)        */
    rocke_value_t* dtl_a_rsrc; /* buffer_rsrc(A, big_bytes)    */
    rocke_value_t* dtl_b_rsrc; /* buffer_rsrc(Bp, big_bytes)   */
    rocke_value_t* dtl_a_lds_base; /* smem_addr_of(A_smem)         */
    rocke_value_t* dtl_b_lds_base; /* smem_addr_of(B_smem)         */
    rocke_value_t* dtl_zero_soff; /* const_i32(0)                 */
    rocke_value_t* dtl_c_chunks_per_row; /* const_i32(dtl_chunks_per_row)*/
    rocke_value_t* dtl_c_cols_per_chunk; /* const_i32(elems or bytes)    */
    rocke_value_t* dtl_c_block_size; /* const_i32(block_size)        */

    /* ---- active-tile gate (batched && active_tile_skip) -- */
    rocke_value_t* do_work_cond; /* NULL when the gate is off        */

    /* ---- K-loop result accumulators (nonlocal _for_results) -- */
    /* The compv4/prefetch K-loops compute their final-tile MFMA accs after the
     * scf.for; the simple loop yields straight from the loop. Either way the
     * epilogue reads _for_results. */
    rocke_value_t* for_results[ROCKE_GEMM_MAX_ACCS];
    int num_for_results;
    /* ---- fused operand prologue (spec->_prologue; NULL = none) -- */
    const rocke_gemm_prologue_t* pro; /* _pro                          */
    rocke_gemm_pro_emitter_t* pe; /* _pe (NULL without a prologue)      */
    int kmul; /* _kmul: 2 for the f32 pair view, else 1          */
    bool cplx; /* _cplx: complex B through its real embedding     */
    /* ---- row-interleave LDS layout (trait.lds_row_interleave) -- */
    bool rilv; /* _rilv                                           */
    int rilv_cols; /* _rilv_cols: 8 * block_k + rilv_pad / elem bytes */
    /* ---- MN-contiguous operands (_mn_lay_a / _mn_lay_b; has_* false for a
     * K-contiguous operand) -- */
    bool has_mn_a;
    bool has_mn_b;
    rocke_gemm_mn_layout_t mn_lay_a;
    rocke_gemm_mn_layout_t mn_lay_b;
    /* ---- native tensor GETT side channel (_gett; NULL = none) -- */
    struct rocke_gett_hooks* gett; /* _gett                              */
    bool gett_addr; /* _gett_addr: the hook owns the operand addressing */
    bool gett_own_lds; /* _gett_own_lds: ... and the LDS stages too       */
    /* ---- register-staged global prefetch body (trait.vgpr_prefetch,
     * gemm_build_vp.cpp; vpf 0 = off) -- */
    int vpf; /* _vpf = trait.vgpr_prefetch                     */
    int vpd; /* _vpd = trait.vgpr_prefetch_depth               */
    struct rocke_gemm_vp_state* vp; /* the _vp_* loader state (NULL: off) */
    /* ---- in-launch split-K reduction (trait.split_k_serial, split_k > 1) -- */
    bool sk_serial; /* _sk_serial                                  */
    rocke_value_t* sks_w; /* param W (f32 partial slots)               */
    rocke_value_t* sks_cnt; /* param Cnt (i32 arrival counters)          */
} rocke_gemm_build_ctx_t;

/* ===================================================================== *
 *  PHASE FUNCTIONS -- one per Python closure / module-level helper.
 *
 *  Each phase reads/writes only ctx (and the builder it carries). They emit IR
 *  in the byte-identical Python order. The names track the Python source so a
 *  reviewer can diff each .c against its closure.
 * ===================================================================== */

/* ----- module-level pure helpers (no closure capture; spec/op-driven). These
 * mirror the file-scope Python functions of the same name. They take a ctx for
 * uniformity even when they only read spec/op, so a body TU never re-derives
 * state already on the ctx. ----- */

/* ab_double_buffer_decision(...): whether the compv4 VGPR-staged AB LDS double
 * buffer (_emit_kloop_db) is emitted. Pure; single source for the validity gate,
 * the build-ctx setup and the load phase. Mirrors the Python function of the
 * same name (gemm_universal.py). */
bool rocke_gemm_ab_db_decision(const rocke_gemm_universal_spec_t* spec,
                               const rocke_archtarget_t* target);

/* lds_swizzle_params(a_per_lane, b_per_lane, block_k, swz_vec) -> (R, W, L) of
 * the lds_swizzle permutation col ^= ((row >> R) % 2^W) << L, the granule the
 * fragment widened to the LDS store vector swz_vec. Pure and env-free; single
 * source for the validity gate and the build ctx (which applies the
 * CK_SWZ_R/W/L debug overrides on top). Mirrors the Python function of the
 * same name (gemm_universal.py). */
void rocke_gemm_lds_swizzle_params(
    int a_per_lane, int b_per_lane, int block_k, int swz_vec, int* out_r, int* out_w, int* out_l);

/* _lds_swizzle_vec(spec, load_vec): the swz_vec of spec's lds_swizzle (the load
 * vector, except on the register-staged prefetch body; 0 without lds_swizzle). */
int rocke_gemm_lds_swizzle_vec(const rocke_gemm_universal_spec_t* spec, int load_vec);

/* _spec_swizzle_params(spec, a, b): lds_swizzle_params with the trait overrides
 * lds_swizzle_r and lds_swizzle_l (which also sets W = log2(tile_k >> L)) */
void rocke_gemm_spec_swizzle_params(const rocke_gemm_universal_spec_t* spec,
                                    int a_per_lane,
                                    int b_per_lane,
                                    int* out_r,
                                    int* out_w,
                                    int* out_l);

/* DTL_BUFFER_NUM_RECORDS (gemm_universal.py): num_records of the A/B buffer
 * resources of the direct-to-LDS load path. Loads past it return 0, so a DTL
 * kernel is only correct while each operand is at most this many bytes (the
 * dispatcher enforces it per request). */
#define ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS 0x7FFF0000

/* hb_sched helpers (gemm_universal.py _hb_k_atoms / hb_last_read_slot /
 * hb_slots) and mfma_order_perm (`even` / `odd`: mfmas_m * mfmas_n entries) */
int rocke_gemm_hb_k_atoms(const rocke_gemm_universal_spec_t* spec);
int rocke_gemm_hb_last_read_slot(const rocke_gemm_universal_spec_t* spec);
void rocke_gemm_hb_slots(const rocke_gemm_universal_spec_t* spec, int* rel, int* acq);
void rocke_gemm_mfma_order_perm(int mo, int mfmas_m, int mfmas_n, int* even, int* odd);

/* _atom_frag_lengths(op) -> (a,b,c)_frag_len. Pure; fills out-params. */
void rocke_gemm_atom_frag_lengths(const rocke_mmaop_t* op, int* a_frag, int* b_frag, int* c_frag);

/* _emit_zero_acc_op(b, op): zero_vec(op->c_dtype, op->c_frag_len). Used to build
 * acc_init. (The MFMA-only _emit_zero_acc / _mfma_atom_widths variants are not
 * reached by the contract-driven body; kept here for parity with the Python
 * module surface.) */
rocke_value_t* rocke_gemm_emit_zero_acc_op(rocke_ir_builder_t* b, const rocke_mmaop_t* op);

/* _emit_mma(b, op, a, bb, c): target-neutral D = a*bb + c via rocke_b_mma. */
/* _mma_ab tail: pin an MFMA result to the AGPR file when trait.acc_agpr. */
rocke_value_t* rocke_gemm_pin_acc_agpr(rocke_ir_builder_t* b,
                                       const rocke_gemm_universal_spec_t* spec,
                                       rocke_value_t* d);
rocke_value_t* rocke_gemm_emit_mma(rocke_ir_builder_t* b,
                                   const rocke_mmaop_t* op,
                                   rocke_value_t* a,
                                   rocke_value_t* bb,
                                   rocke_value_t* c);

/* _emit_mfma(b, spec, a, bb, c): ISA-named MFMA dispatch (MFMA-only callers).
 * Present for module-surface parity; the unified body routes through
 * rocke_gemm_emit_mma. */
rocke_value_t* rocke_gemm_emit_mfma(rocke_ir_builder_t* b,
                                    const rocke_gemm_universal_spec_t* spec,
                                    rocke_value_t* a,
                                    rocke_value_t* bb,
                                    rocke_value_t* c);

/* _emit_zero_acc(b, spec): zero_vec_f32 sized from spec geometry (MFMA-only). */
rocke_value_t* rocke_gemm_emit_zero_acc(rocke_ir_builder_t* b,
                                        const rocke_gemm_universal_spec_t* spec);

/* _elem_bytes(spec): storage element size in bytes of the homogeneous A/B/C
 * dtype. */
int rocke_gemm_elem_bytes(const rocke_gemm_universal_spec_t* spec);

/* _choose_load_vec(spec) -> width. Thin adapter over rocke_choose_load_vec. */
int rocke_gemm_choose_load_vec(const rocke_gemm_universal_spec_t* spec);

/* _emit_smem_load(b, smem, row, col, n, dtype): f16/n==4 fast path else vN;
 * a one-element fragment is returned as a scalar. */
rocke_value_t* rocke_gemm_emit_smem_load(rocke_ir_builder_t* b,
                                         rocke_value_t* smem,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         int n,
                                         const rocke_type_t* dtype);

/* _swz_col(ctx, col, row): the LDS XOR-swizzle column rewrite. Returns `col`
 * unchanged when ctx->swz is false. (Python inner closure `_swz_col`.) */
rocke_value_t*
    rocke_gemm_swz_col(rocke_gemm_build_ctx_t* ctx, rocke_value_t* col, rocke_value_t* row);

/* hotloop_* family (module-level). The well-formed guard + per-tile schedule
 * emission used by emit_mfma_phase for compv3/compv4. */
/* _hotloop_inst_list(spec, load_vec). */
rocke_hotloop_inst_list_t rocke_gemm_hotloop_inst_list(rocke_ir_builder_t* b,
                                                       const rocke_gemm_universal_spec_t* spec,
                                                       int load_vec);
/* _hotloop_well_formed(il, pipeline). */
bool rocke_gemm_hotloop_well_formed(const rocke_hotloop_inst_list_t* il, const char* pipeline);
/* _emit_hotloop_schedule(b, spec, load_vec): emit compv3/compv4 schedule once,
 * with the degenerate-tile flat-hint fallback. */
void rocke_gemm_emit_hotloop_schedule(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      int load_vec);

/* ----- the load / compute closures (ctx-driven) ----- */

/* emit_load_phase(ctx, A_dst, B_dst, k_off, lds_parity). One K-tile's coalesced
 * global->LDS copy. Covers the DTLA path, the canonical strided B path, and the
 * preshuffle_b path. `lds_parity` is split into a compile-time int OR a runtime
 * Value (Python `int | Value`): pass parity_v != NULL for the runtime form,
 * else parity_v == NULL and parity_imm carries the 0/1 compile-time parity. */
void rocke_gemm_emit_load_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst,
                                rocke_value_t* k_off,
                                int parity_imm,
                                rocke_value_t* parity_v);

/* _emit_frag_smem_load(ctx, src, mn_in_atom, k_in_atom, atom_mn_base,
 * k_tile_base, frag_len): one frag_len-wide operand fragment from a row-major
 * LDS tile (8-wide chunked + vec_concat for wide WMMA frags). */
rocke_value_t* rocke_gemm_emit_frag_smem_load(rocke_gemm_build_ctx_t* ctx,
                                              rocke_value_t* src,
                                              rocke_value_t* mn_in_atom,
                                              rocke_value_t* k_in_atom,
                                              rocke_value_t* atom_mn_base,
                                              rocke_value_t* k_tile_base,
                                              int frag_len);

/* _emit_wmma_phase(ctx, A_src, B_src, iter_vars[n], out_accs[n]): one K-tile of
 * WMMA atoms, fully contract-driven. Reads iter_vars (length ctx->num_accs),
 * writes the new accs into out_accs. */
void rocke_gemm_emit_wmma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs);

/* emit_mfma_phase(ctx, A_src, B_src, iter_vars[n], parity_imm, parity_v,
 * out_accs[n]): one K-tile of MMAs across every per-warp atom position + K
 * step. Delegates to rocke_gemm_emit_wmma_phase on RDNA. Emits the compv3/compv4
 * hotloop schedule at the tail. lds_parity is int|Value as in emit_load_phase. */
void rocke_gemm_emit_mfma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                int parity_imm,
                                rocke_value_t* parity_v,
                                rocke_value_t** out_accs);

/* emit_compute_and_epilogue(ctx): the prefetch/db/simple K-loop selector +
 * _emit_epilogue. */
void rocke_gemm_emit_compute_and_epilogue(rocke_gemm_build_ctx_t* ctx);

/* _emit_kloop_db(ctx): compv4 double-buffered (VGPR-staged) K-loop. Writes
 * ctx->for_results / ctx->num_for_results. */
void rocke_gemm_emit_kloop_db(rocke_gemm_build_ctx_t* ctx);

/* _emit_kloop_simple(ctx): single-buffer load-then-compute K-loop. */
void rocke_gemm_emit_kloop_simple(rocke_gemm_build_ctx_t* ctx);

/* _emit_kloop_prefetch(ctx): DTLA ping-pong software-pipelined K-loop (falls
 * back to _simple when loads_per_tile > 63). */
void rocke_gemm_emit_kloop_prefetch(rocke_gemm_build_ctx_t* ctx);

/* _emit_kloop_staged(ctx): the alias-scoped DTL ring (dtl_stages >= 2) and its
 * DTL ring levers, incl. the persistent overlapped-epilogue body
 * (gemm_build_staged.cpp). */
void rocke_gemm_emit_kloop_staged(rocke_gemm_build_ctx_t* ctx);

/* ---- the register-staged global prefetch body (gemm_build_vp.cpp) ---- */
/* the _vp_* loader state (resources at the tile origin, per-lane voffsets):
 * emitted after the DTL plumbing, sets ctx->vp (freed by
 * rocke_gemm_vp_state_free) */
void rocke_gemm_vp_setup(rocke_gemm_build_ctx_t* ctx);
void rocke_gemm_vp_state_free(struct rocke_gemm_vp_state* st);
/* _emit_kloop_vpf (vgpr_prefetch 1 / 2) */
void rocke_gemm_emit_kloop_vpf(rocke_gemm_build_ctx_t* ctx);
/* _emit_kloop_vpf_ilv (vgpr_prefetch 3 / 4) */
void rocke_gemm_emit_kloop_vpf_ilv(rocke_gemm_build_ctx_t* ctx);
/* _vp_batch_off64(nm): the batch element offset of operand 'A' / 'B' / 'C' as an
 * i64 (the wide_addr form); NULL when not batched (no op) */
rocke_value_t* rocke_gemm_vp_batch_off64(rocke_gemm_build_ctx_t* ctx, char nm);
/* _ep_split_k_serial (trait.split_k_serial 2 / 3; gemm_build_sks.cpp) */
void rocke_gemm_ep_split_k_serial(rocke_gemm_build_ctx_t* ctx,
                                  rocke_value_t* const* vals,
                                  int num_vals);

/* _emit_store_swapped_acc (mfma_swap_ab): direct epilogue writing the C^T
 * accumulator as kCM1PerLane-wide vector stores (gemm_epilogue.cpp). */
void rocke_gemm_emit_store_swapped_acc(rocke_ir_builder_t* b,
                                       const rocke_gemm_universal_spec_t* spec,
                                       rocke_value_t* lane,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       rocke_value_t* m_base_off,
                                       rocke_value_t* n_base_off,
                                       int c_per_lane,
                                       const rocke_type_t* storage_dtype,
                                       rocke_value_t* M,
                                       rocke_value_t* N,
                                       rocke_value_t* C,
                                       rocke_value_t* batch_off_c);

/* _emit_epilogue(ctx): dispatch to the cshuffle or default epilogue using
 * ctx->for_results and the captured warp/lane/offset Values. */
void rocke_gemm_emit_epilogue(rocke_gemm_build_ctx_t* ctx);

/* ===================================================================== *
 *  EPILOGUE helpers (module-level free functions in Python; they take an
 *  explicit argument list rather than the ctx because the Python signatures
 *  are already flat parameter lists. A body TU passes the captured ctx fields).
 * ===================================================================== */

/* per-cell store callback used by the accumulator scatter. The default
 * epilogue's _store_cell and the cshuffle epilogue's _smem_cell are passed as
 * (fn, user) so the scatter stays a single shared walker. */
typedef void (*rocke_gemm_per_cell_fn)(rocke_ir_builder_t* b,
                                       rocke_value_t* c_m,
                                       rocke_value_t* c_n,
                                       rocke_value_t* acc_h,
                                       int i,
                                       void* user);

/* _emit_mfma_acc_scatter(...): the shared MFMA accumulator -> (row,col) walk
 * (CWarpDstrEncoding-driven). n_base_first selects the column grouping. */
void rocke_gemm_emit_mfma_acc_scatter(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* lane,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* m_base_off,
                                      rocke_value_t* n_base_off,
                                      int c_per_lane,
                                      const rocke_type_t* storage_dtype,
                                      rocke_gemm_per_cell_fn per_cell,
                                      void* user,
                                      bool n_base_first);

/* _emit_epilogue_default(...): direct vector-store epilogue (incl. the WMMA
 * c_layout scatter branch). fused_epilogue is opaque (NULL = matmul-only). */
/* _acc_agpr_out(b, spec, acc): with trait.acc_agpr, hand one accumulator out of
 * the AGPRs (ordered asm "", "=a,0"); identity otherwise. */
rocke_value_t* rocke_gemm_acc_agpr_out(rocke_ir_builder_t* b,
                                       const rocke_gemm_universal_spec_t* spec,
                                       rocke_value_t* acc);

/* build_universal_gemm's _ep_default(vals, bm, bn, batch_off_c, fused): the
 * default epilogue of one output tile; with trait.pad_fast_tile a uniform
 * branch sends tiles fully inside M x N to the unguarded (pad-free) body. */
void rocke_gemm_ep_default(rocke_gemm_build_ctx_t* ctx,
                           rocke_value_t* const* vals,
                           int num_vals,
                           rocke_value_t* bm,
                           rocke_value_t* bn,
                           rocke_value_t* batch_off_c,
                           void* fused_epilogue,
                           bool fused_is_mde);

/* _ep_default with the vgpr_prefetch wide_addr arguments m_ext / c_ptr (the M
 * extent and C base the stores use; NULL: M / C) */
void rocke_gemm_ep_default_ext(rocke_gemm_build_ctx_t* ctx,
                               rocke_value_t* const* vals,
                               int num_vals,
                               rocke_value_t* bm,
                               rocke_value_t* bn,
                               rocke_value_t* batch_off_c,
                               void* fused_epilogue,
                               bool fused_is_mde,
                               rocke_value_t* m_ext,
                               rocke_value_t* c_ptr);

void rocke_gemm_emit_epilogue_default(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      const rocke_mmaop_t* op,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* warp_m_idx,
                                      rocke_value_t* warp_n_idx,
                                      rocke_value_t* lane,
                                      rocke_value_t* block_m_off,
                                      rocke_value_t* block_n_off,
                                      rocke_value_t* M,
                                      rocke_value_t* N,
                                      rocke_value_t* C,
                                      int c_per_lane,
                                      rocke_value_t* batch_off_c,
                                      void* fused_epilogue,
                                      bool fused_is_mde);

/* _emit_epilogue_split_k(...): split-K atomic-add epilogue. Scatters each
 * warp's per-lane f32 accumulator to its output (c_m, c_n) (the canonical MFMA
 * CWarpDstrEncoding layout, identical to the default epilogue) and atomic-adds
 * the raw f32 value into Cf32[c_m, c_n] (atomicrmw fadd), guarded by c_m<M /
 * c_n<N when pad_m / pad_n is set. */
void rocke_gemm_emit_epilogue_split_k(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* const* accs,
                                      int num_accs,
                                      rocke_value_t* warp_m_idx,
                                      rocke_value_t* warp_n_idx,
                                      rocke_value_t* lane,
                                      rocke_value_t* block_m_off,
                                      rocke_value_t* block_n_off,
                                      rocke_value_t* M,
                                      rocke_value_t* N,
                                      rocke_value_t* Cf32,
                                      int c_per_lane);

/* _emit_epilogue_cshuffle(...): LDS-staged cshuffle epilogue. */
void rocke_gemm_emit_epilogue_cshuffle(rocke_ir_builder_t* b,
                                       const rocke_gemm_universal_spec_t* spec,
                                       rocke_value_t* smem_unused,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       rocke_value_t* warp_m_idx,
                                       rocke_value_t* warp_n_idx,
                                       rocke_value_t* lane,
                                       rocke_value_t* block_m_off,
                                       rocke_value_t* block_n_off,
                                       rocke_value_t* M,
                                       rocke_value_t* N,
                                       rocke_value_t* C,
                                       int a_per_lane,
                                       int b_per_lane,
                                       int c_per_lane,
                                       rocke_value_t* batch_off_c,
                                       void* fused_epilogue,
                                       bool fused_is_mde);

/* _load_smem_scalar(b, smem, row, col, dtype): 2-half load + extract[0]. */
rocke_value_t* rocke_gemm_load_smem_scalar(rocke_ir_builder_t* b,
                                           rocke_value_t* smem,
                                           rocke_value_t* row,
                                           rocke_value_t* col,
                                           const rocke_type_t* dtype);

/* _load_smem_vec(b, smem, row, col, n, dtype): f16/n==4 fast path else vN. */
/* ---- fused operand prologue (gemm_prologue.cpp; Python PrologueEmitter) -- */
rocke_gemm_pro_emitter_t* rocke_gemm_pro_emitter_new(rocke_ir_builder_t* b,
                                                     const rocke_gemm_prologue_t* pro,
                                                     const rocke_type_t* compute,
                                                     int frag_len);
void rocke_gemm_pro_emitter_free(rocke_gemm_pro_emitter_t* pe);
/* raw_len(): 16-bit elements one lane reads per fragment */
int rocke_gemm_pro_raw_len(const rocke_gemm_pro_emitter_t* pe);
/* cplx_mode "ni" (row-interleaved bodies) / "lane" with the lane-parity i1 */
void rocke_gemm_pro_set_cplx_ni(rocke_gemm_pro_emitter_t* pe);
void rocke_gemm_pro_set_cplx_lane(rocke_gemm_pro_emitter_t* pe, rocke_value_t* lane_odd);
/* after_load(raw, which): which is 'A' or 'B' */
rocke_value_t*
    rocke_gemm_pro_after_load(rocke_gemm_pro_emitter_t* pe, rocke_value_t* raw, char which);
/* mma(...): every product of the recipe into `acc` (pinned with acc_agpr_pin);
 * ni < 0 means None */
rocke_value_t* rocke_gemm_pro_mma(rocke_gemm_pro_emitter_t* pe,
                                  const rocke_gemm_universal_spec_t* spec,
                                  const rocke_mmaop_t* op,
                                  rocke_value_t* a_frag,
                                  rocke_value_t* b_frag,
                                  rocke_value_t* acc,
                                  bool swap,
                                  int ni);

/* _frag_ld(src, row, col_base, key, n, which, swz): one operand fragment
 * (_emit_smem_load + the prologue hook); key NULL / swz false = no swizzle */
rocke_value_t* rocke_gemm_frag_ld(rocke_gemm_build_ctx_t* ctx,
                                  rocke_value_t* src,
                                  rocke_value_t* row,
                                  rocke_value_t* col_base,
                                  rocke_value_t* key,
                                  int n,
                                  char which,
                                  bool swz);

/* _brow(r): the global B row of B' row r (complex B: r / 2) */
rocke_value_t* rocke_gemm_brow(rocke_gemm_build_ctx_t* ctx, rocke_value_t* r);

/* _mma_ab(a, b, acc, ni): one MFMA (or the prologue's products); ni < 0 = None */
rocke_value_t* rocke_gemm_mma_ab(rocke_gemm_build_ctx_t* ctx,
                                 rocke_value_t* a_frag,
                                 rocke_value_t* b_frag,
                                 rocke_value_t* acc,
                                 int ni);

/* _acc_f32_out(acc): the f32 store's view of one accumulator (prologue
 * out_f32 + acc_agpr without agpr_out: asm "", "=v,0"; else _acc_agpr_out) */
rocke_value_t* rocke_gemm_acc_f32_out(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* acc);

/* The storage type of the epilogue stores: f32 with the prologue's out_f32,
 * else the GEMM storage dtype. */
bool rocke_gemm_out_f32(const rocke_gemm_universal_spec_t* spec);

rocke_value_t* rocke_gemm_load_smem_vec(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* row,
                                        rocke_value_t* col,
                                        int n,
                                        const rocke_type_t* dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_INTERNAL_H */
