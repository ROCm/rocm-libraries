/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_implicit_gemm_internal.h -- PRIVATE shared state + phase-
 * function contract for the C99 port of build_implicit_gemm_conv
 * (rocke/instances/common/conv_implicit_gemm.py).
 *
 * WHY THIS HEADER EXISTS.
 *   build_implicit_gemm_conv() in Python is a ~750-line function whose body is a
 *   stack of NESTED CLOSURES (a_descriptor, b_descriptor, emit_load_phase,
 *   emit_wmma_phase, emit_mfma_phase, plus the three inline K-loop drivers and
 *   the issue_load/compute lambdas for the async pipeline). Every closure
 *   captures the SAME enclosing-function locals: the builder, the resolved MMA
 *   op + legacy MfmaAtom, the param Values (A/B/D + *_bytes), every derived
 *   geometry constant, the SSA constants (c0, c_block_k, c_K_gemm), the
 *   block/warp/lane decomposition (the bound WarpGrid), the per-CTA tile
 *   origins, the LDS plan + smem handles, the buffer resources + the three
 *   coordinate-transform descriptors, the loader objects (sync OR async), the
 *   schedule policy, the accumulator iter-args, and the mutable `k_off_capture`
 *   cell the descriptor closures read.
 *
 *   In C there is no closure capture. The faithful port turns each Python
 *   closure into a free function that takes a POINTER to one shared context
 *   struct, rocke_conv_build_ctx_t, which holds EXACTLY the set of variables the
 *   closures shared. The driver (rocke_build_implicit_gemm_conv in
 *   rocke/instance_conv_implicit_gemm.h) populates the ctx in the same order the
 *   Python prologue computes its locals, then calls the phase functions in the
 *   Python K-loop / epilogue order.
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every body-implementing agent binds
 *   to. It is DESIGNED TO BE COMPLETE: every local/closured variable the Python
 *   body shares across phases is a field here. A body agent implementing a phase
 *   .c file MUST be able to read/write only ctx fields and call the prototypes
 *   below WITHOUT editing this header -- adding a field forces every other body
 *   TU to recompile and risks a merge conflict on the shared surface. If a phase
 *   genuinely needs a value not present, that is a design bug in this header to
 *   be fixed once, deliberately, not patched per-phase.
 *
 *   Naming: ctx fields mirror the Python local names 1:1 (Python `block_m_off_v`
 *   -> `ctx->block_m_off_v`; Python `A_smem2` -> `ctx->A_smem2`). Phase
 *   functions mirror the Python closure / module-helper names with a `rocke_conv_`
 *   prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by
 * the instance_conv_implicit_gemm_*.c translation units. Public callers use
 * rocke/instance_conv_implicit_gemm.h.
 */
#ifndef ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_INTERNAL_H
#define ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_mmaop_t, rocke_archtarget_t */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/helper_rocke.helpers.epilogues.h" /* rocke_warp_grid_t */
#include "rocke/helper_rocke.helpers.loads.h" /* loaders + descriptor fn */
#include "rocke/helper_rocke.helpers.schedule.h" /* rocke_schedule_policy_t */
#include "rocke/helper_rocke.helpers.transforms.h" /* rocke_tensor_descriptor_t */
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The per-warp accumulator iter-arg count is mfmas_per_warp_m *
 * mfmas_per_warp_n. For the buildable conv tile space this is small; 64 is
 * generous headroom (e.g. 128x128 / (2*32) = 2x2 = 4 accs). */
#define ROCKE_CONV_MAX_ACCS 64

/* K-loop unroll / async ping-pong iteration count = ceil(K_gemm / block_k). For
 * the bake-off shape this is 9 (576/64); 4096 covers any buildable shape. */
#define ROCKE_CONV_MAX_K_ITERS 4096

/* ============================================================ *
 * LdsLayout (value-type slice)
 * ============================================================ *
 *
 * The LdsLayout C port (rocke/helpers/layouts.py) is a peer that is not yet a
 * standalone header. The conv body needs exactly this slice, mirroring
 * effective_lds_layout()'s result + storage_shape()/validate_for_async():
 *   logical_cols   : tile_k
 *   k_pad          : padded_k's +8 (sync, block_k>=16) / +0 (async/packed)
 *   row_stride     : logical_cols + k_pad
 *   swizzle        : NULL (None) | "xor" | "cyclic"
 * storage_shape(rows) -> (rows, row_stride). When the LdsLayout peer port lands
 * this struct is replaced by an include; the field set is the part the conv body
 * reads, so the populate routine fills it from spec.effective_lds_layout(). */
typedef struct rocke_conv_lds_layout
{
    int logical_cols; /* tile_k                          */
    int k_pad; /* +8 / +0 per effective policy    */
    int row_stride; /* logical_cols + k_pad            */
    const char* swizzle; /* NULL | "xor" | "cyclic"        */
    bool requires_packed_async; /* LdsLayout.requires_packed_async (async path) */
} rocke_conv_lds_layout_t;

/* LdsLayout port (rocke/helpers/layouts.py) -- the slice the conv body / spec
 * validation needs. All three mirror the Python methods 1:1 and write the
 * structured ValueError message (verbatim) into reason on rejection.
 *
 *   rocke_conv_lds_layout_validate            <- LdsLayout.validate()
 *   rocke_conv_lds_layout_validate_for_async  <- LdsLayout.validate_for_async()
 *   ..._effective_lds_layout                <- ImplicitGemmConvSpec.effective_lds_layout()
 *
 * Each returns true on success, false on failure (reason filled if non-NULL).
 * effective_lds_layout fills *out with the derived layout (and runs validate());
 * the lds_layout override branch (spec.lds_layout != NULL) is a peer port -- it
 * is unreachable through the current C spec (lds_layout is an opaque void* with
 * no C constructor) and is reported as such rather than guessed. */
bool rocke_conv_lds_layout_validate(const rocke_conv_lds_layout_t* l,
                                    char* reason,
                                    size_t reason_cap);
bool rocke_conv_lds_layout_validate_for_async(const rocke_conv_lds_layout_t* l,
                                              char* reason,
                                              size_t reason_cap);
bool rocke_implicit_gemm_conv_spec_effective_lds_layout(const rocke_implicit_gemm_conv_spec_t* s,
                                                        rocke_conv_lds_layout_t* out,
                                                        char* reason,
                                                        size_t reason_cap);

/* ============================================================ *
 * BufferResource (value-type slice)
 * ============================================================ *
 *
 * make_buffer_resource(b, ptr, num_bytes) (helpers/tensor_view.py) wraps
 * b.buffer_rsrc(ptr, num_bytes) + a pre-bound zero soffset. The conv body only
 * reads `.rsrc` off the result; this slice holds that plus the inputs so the
 * populate routine can reproduce make_buffer_resource exactly. (Full
 * BufferResource is a peer port.) */
typedef struct rocke_conv_buffer_resource
{
    rocke_value_t* ptr; /* the global PtrType param Value     */
    rocke_value_t* num_bytes; /* the I32 *_bytes param Value        */
    rocke_value_t* rsrc; /* b.buffer_rsrc(ptr, num_bytes)      */
    rocke_value_t* soffset; /* pre-bound const_i32(0)             */
} rocke_conv_buffer_resource_t;

/* ===================================================================== *
 *  rocke_conv_build_ctx_t
 *
 *  The single shared state object. Holds every enclosing-function local that
 *  the Python closures capture, grouped by the Python prologue's phases; field
 *  order follows the order the Python computes them so the populate routine
 *  reads top-to-bottom against the source (lines 787-1378).
 * ===================================================================== */
typedef struct rocke_conv_build_ctx
{
    /* ---- inputs / resolved environment (build args + setup) ---------------- */
    rocke_ir_builder_t* b; /* the IRBuilder `b`             */
    const rocke_implicit_gemm_conv_spec_t* spec; /* the ImplicitGemmConvSpec      */
    const char* arch; /* `arch` (NULL-normalised)      */
    const rocke_conv_build_overrides_t* ov; /* override callbacks (may NULL) */
    const rocke_conv_problem_t* p; /* &spec->problem (alias `p`)    */
    const rocke_archtarget_t* target; /* ArchTarget.from_gfx(arch)     */
    const rocke_mmaop_t* op; /* _resolve_conv_op(spec, arch)  */
    const rocke_mfma_atom_t* atom; /* spec.atom (NULL on WMMA path) */
    bool is_wmma; /* op->family == "wmma"          */

    /* ---- kernel params (Values) ---- */
    rocke_value_t* A; /* param A (PtrType f16 global)              */
    rocke_value_t* Bp; /* param B (PtrType f16 global)              */
    rocke_value_t* D; /* param D (PtrType f16 global)              */
    void* extra_context; /* extra_params(b) return (opaque object)    */
    rocke_value_t* A_bytes; /* param A_bytes (I32)                       */
    rocke_value_t* B_bytes; /* param B_bytes (I32)                       */
    rocke_value_t* D_bytes; /* param D_bytes (I32)                       */

    /* ---- per-lane MMA fragment widths (off op) ---- */
    int a_per_lane; /* op.a_frag_len */
    int b_per_lane; /* op.b_frag_len */
    int c_per_lane; /* op.c_frag_len */

    /* ---- block tile dims (aliases of spec geometry) ---- */
    int block_m; /* spec.tile_m */
    int block_n; /* spec.tile_n */
    int block_k; /* spec.tile_k */

    /* ---- block/warp/lane decomposition (the bound WarpGrid) ---- *
     * WarpGrid.from_atom(op, ...).bind(b, block_m_axis="y", block_n_axis="x").
     * The whole bound grid is held by value (its compile-time geometry + the SSA
     * tid/lane/warp_*_idx/block_*_off the epilogue helpers consume). The grid is
     * re-stamped (block_m_off/block_n_off) when chiplet_swizzle is on. */
    rocke_warp_grid_t grid;
    rocke_value_t* tid; /* grid.tid                                  */
    rocke_value_t* lane; /* grid.lane                                 */
    rocke_value_t* warp_id; /* grid.warp_id                              */
    rocke_value_t* warp_m_idx; /* grid.warp_m_idx                           */
    rocke_value_t* warp_n_idx; /* grid.warp_n_idx                           */

    /* ---- common geometry constants (SSA) ---- */
    rocke_value_t* c0; /* const_i32(0)         */
    rocke_value_t* c_block_k; /* const_i32(block_k)  */
    rocke_value_t* c_K_gemm; /* const_i32(p.K_gemm)  */

    /* ---- per-CTA tile origins (chiplet-swizzle aware) ---- */
    rocke_value_t* block_m_off_v; /* grid.block_m_off OR swizzled row*tile_m */
    rocke_value_t* block_n_off_v; /* grid.block_n_off OR swizzled col*tile_n */

    /* ---- LDS plan + smem handles ---- */
    rocke_conv_lds_layout_t lds_layout; /* spec.effective_lds_layout()         */
    bool double_buffer; /* compv4 || async_dma || unroll_k     */
    rocke_value_t* A_smem; /* smem_alloc A_smem                   */
    rocke_value_t* B_smem; /* smem_alloc B_smem                   */
    rocke_value_t* A_smem2; /* A_smem when single-buffer            */
    rocke_value_t* B_smem2; /* B_smem when single-buffer            */

    /* ---- per-warp MFMA tile counts ---- */
    int mfmas_m; /* spec.mfmas_per_warp_m   */
    int mfmas_n; /* spec.mfmas_per_warp_n   */
    int k_atoms; /* spec.k_atoms_per_tile_k */

    /* ---- accumulators ---- */
    rocke_value_t* acc_init; /* zero_vec_f32(c_per_lane)    */
    /* The `accs` list: one zero-acc iter-arg per (mi, ni) warp MFMA tile, named
     * "acc_m{mi}_n{ni}" + the shared acc_init. */
    const char* acc_names[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* acc_inits[ROCKE_CONV_MAX_ACCS]; /* all == acc_init             */
    int num_accs; /* mfmas_m * mfmas_n           */

    /* ---- global -> LDS coalesced copy plan ---- */
    int threads; /* spec.block_size           */
    int load_vec; /* _choose_load_vec(spec)    */

    /* ---- coordinate-transform descriptors ---- */
    rocke_tensor_descriptor_t* A_desc; /* make_a_descriptor(p, decompose_m)     */
    rocke_tensor_descriptor_t* B_desc; /* make_b_descriptor(p)                  */
    /* D_desc is built lazily inside the epilogue phase (Python builds it per
     * epilogue fn); held here so an epilogue_override and the two stock
     * epilogues share one instance. NULL until the epilogue phase populates. */
    rocke_tensor_descriptor_t* D_desc;

    /* ---- buffer resources (CK-Tile views over A/B/D) ---- */
    rocke_conv_buffer_resource_t a_buf_rsrc; /* make_buffer_resource(b, A, A_bytes) */
    rocke_conv_buffer_resource_t b_buf_rsrc; /* make_buffer_resource(b, Bp, B_bytes)*/
    rocke_conv_buffer_resource_t d_buf_rsrc; /* make_buffer_resource(b, D, D_bytes) */
    rocke_value_t* a_rsrc; /* a_buf_rsrc.rsrc                     */
    rocke_value_t* b_rsrc; /* b_buf_rsrc.rsrc                     */
    rocke_value_t* d_rsrc; /* d_buf_rsrc.rsrc                     */
    void* input_cache_context; /* input_cache_setup(...) return        */

    /* ---- loaders (exactly one family populated; the other side NULL) ---- *
     * async_dma=True  : a_loader/b_loader are AsyncTileLoader.from_tile(...).
     * async_dma=False : a_sync_loader/b_sync_loader are CoalescedTileLoader(...). */
    bool async_dma; /* spec.async_dma (cached)         */
    rocke_async_tile_loader_t a_loader; /* async A loader (valid iff async)*/
    rocke_async_tile_loader_t b_loader; /* async B loader                  */
    bool have_async_loaders; /* true => a_loader/b_loader valid */
    rocke_coalesced_tile_loader_t a_sync_loader; /* sync A loader (valid iff sync)*/
    rocke_coalesced_tile_loader_t b_sync_loader; /* sync B loader                 */
    bool have_sync_loaders; /* true => *_sync_loader valid     */

    /* ---- schedule policy ---- */
    rocke_schedule_policy_t schedule; /* SchedulePolicy.for_pipeline(...)        */

    /* ---- mutable cell read by a_descriptor / b_descriptor ---- *
     * Python `k_off_capture = [None]`: emit_load_phase writes the current k0 so
     * the descriptor closures pick it up without recompiling the loaders. The
     * K-loop drivers also restore it (unroll_k path) to the consumed tile's
     * offset before the MFMA phase. */
    rocke_value_t* k_off_capture;

    /* ---- K-loop result accumulators ---- *
     * All three K-loop drivers write their final-tile accs here; the epilogue
     * phase reads final_accs. */
    rocke_value_t* final_accs[ROCKE_CONV_MAX_ACCS];
    int num_final_accs;
} rocke_conv_build_ctx_t;

/* ===================================================================== *
 *  PHASE FUNCTIONS -- one per Python closure / module-level helper.
 *
 *  Each phase reads/writes only ctx (and the builder it carries). They emit IR
 *  in the byte-identical Python order. Names track the Python source so a
 *  reviewer can diff each .c against its closure.
 * ===================================================================== */

/* ----- module-level pure helpers (mirror the file-scope Python functions; they
 * take a ctx for uniformity even when they only read spec/op). ----- */

/* _conv_mma_family(arch) -> "wmma" (wave32) | "mma" (wave64). */
const char* rocke_conv_mma_family(const char* arch);

/* _resolve_conv_op(spec, arch) -> MmaOp. Sets the builder error on the
 * no-atom ValueError path and returns NULL. */
const rocke_mmaop_t* rocke_conv_resolve_op(rocke_ir_builder_t* b,
                                           const rocke_implicit_gemm_conv_spec_t* spec,
                                           const char* arch);

/* _choose_load_vec(spec) -> width. Thin adapter over rocke_choose_load_vec. */
int rocke_conv_choose_load_vec(const rocke_implicit_gemm_conv_spec_t* spec);

/* _emit_mfma(b, atom, a, bv, c): atom.emit MFMA dispatch. */
rocke_value_t* rocke_conv_emit_mfma(rocke_ir_builder_t* b,
                                    const rocke_mfma_atom_t* atom,
                                    rocke_value_t* a,
                                    rocke_value_t* bv,
                                    rocke_value_t* c);

/* _emit_smem_load(b, smem, row, col, n): f16/n==4 fast path else vN. */
rocke_value_t* rocke_conv_emit_smem_load(
    rocke_ir_builder_t* b, rocke_value_t* smem, rocke_value_t* row, rocke_value_t* col, int n);

/* _emit_frag_smem_load(b, src, mn_in_atom, k_in_atom, atom_mn_base, k_tile_base,
 * frag_len): one frag_len-wide operand fragment from a row-major LDS tile
 * (8-wide chunked + vec_concat for wide WMMA frags). */
rocke_value_t* rocke_conv_emit_frag_smem_load(rocke_ir_builder_t* b,
                                              rocke_value_t* src,
                                              rocke_value_t* mn_in_atom,
                                              rocke_value_t* k_in_atom,
                                              rocke_value_t* atom_mn_base,
                                              rocke_value_t* k_tile_base,
                                              int frag_len);

/* _apply_accumulator_epilogue(b, epilogue, accs[n], out_accs[n]): apply the
 * static fp32 ConvAccumulatorEpilogue per-lane then re-pack. Identity copies
 * through. Writes out_accs (length num_accs). */
void rocke_conv_apply_accumulator_epilogue(rocke_ir_builder_t* b,
                                           const rocke_conv_acc_epilogue_t* epilogue,
                                           rocke_value_t* const* accs,
                                           int num_accs,
                                           rocke_value_t** out_accs);

/* ----- descriptor address closures (ctx-driven; read ctx->k_off_capture) ----- */

/* a_descriptor(ctx, row, col) -> (off, valid). The conv-coord-transform DAG:
 * k = k_off_capture + col; m (or n/ho/wo via a_mhw_index_fn) from row. Writes
 * the valid predicate through *out_valid (NULL == always in-bounds). This is the
 * rocke_loads_descriptor_fn the loaders call; ctx is passed as `user`. */
rocke_value_t* rocke_conv_a_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* ctx_user);

/* b_descriptor(ctx, row, col) -> (off, valid). KYXC: k_out = block_n_off + row;
 * k_gemm = k_off_capture + col. Same rocke_loads_descriptor_fn contract. */
rocke_value_t* rocke_conv_b_descriptor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* ctx_user);

/* ----- load / compute closures (ctx-driven) ----- */

/* emit_load_phase(ctx, k_off, A_dst, B_dst): one K-tile global->LDS copy. Sets
 * ctx->k_off_capture = k_off, then dispatches the async (AsyncTileLoader +
 * raw_ptr_buffer_load_lds) or sync (CoalescedTileLoader) path, honouring the
 * a_load_override hook on the sync path. */
void rocke_conv_emit_load_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* k_off,
                                rocke_value_t* A_dst,
                                rocke_value_t* B_dst);

/* emit_wmma_phase(ctx, A_src, B_src, iter_vars[n], out_accs[n]): one K-tile of
 * WMMA atoms, fully MMA-contract driven (gfx1151). Reads iter_vars (length
 * ctx->num_accs), writes the new accs into out_accs. */
void rocke_conv_emit_wmma_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs);

/* emit_mfma_phase(ctx, A_src, B_src, iter_vars[n], out_accs[n]): one K-tile of
 * MFMAs across every per-warp atom position + K step. Delegates to
 * rocke_conv_emit_wmma_phase on RDNA. Honours the a_operand_override hook and
 * emits the compv3/compv4 sched_group_barrier hints at each kk step's tail. */
void rocke_conv_emit_mfma_phase(rocke_conv_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                rocke_value_t** out_accs);

/* ----- K-loop drivers (ctx-driven; write ctx->final_accs) ----- *
 * Exactly one is called per build, chosen as Python does:
 *   unroll_k                -> rocke_conv_emit_kloop_unroll
 *   else not async_dma      -> rocke_conv_emit_kloop_simple
 *   else (async_dma)        -> rocke_conv_emit_kloop_async */

/* spec.unroll_k branch (lines 1276-1310): double-buffered Python-unrolled
 * software pipeline (ping-pong A_smem/A_smem2). */
void rocke_conv_emit_kloop_unroll(rocke_conv_build_ctx_t* ctx);

/* not-async branch (lines 1311-1319): single scf.for_iter load+sync+mfma+sync. */
void rocke_conv_emit_kloop_simple(rocke_conv_build_ctx_t* ctx);

/* async_dma branch (lines 1320-1347): SoftwarePipeline.run_ping_pong over the
 * AsyncTileLoader path. (SoftwarePipeline is a peer port; the driver reproduces
 * its run_ping_pong sequencing inline against ctx until that port lands.) */
void rocke_conv_emit_kloop_async(rocke_conv_build_ctx_t* ctx);

/* ----- epilogue (ctx-driven; reads ctx->final_accs) ----- */

/* The epilogue phase (lines 1349-1377): apply the accumulator epilogue, then
 * dispatch epilogue_override / cshuffle / wmma-direct / mfma-direct exactly as
 * the Python if/elif chain. Builds ctx->D_desc as needed. */
void rocke_conv_emit_epilogue(rocke_conv_build_ctx_t* ctx);

/* _emit_direct_epilogue(b, spec, accs[n], grid, d_rsrc): MFMA per-lane scalar
 * fp16 store via DirectEpilogue + the D descriptor addr_fn. */
void rocke_conv_emit_direct_epilogue(rocke_ir_builder_t* b,
                                     const rocke_implicit_gemm_conv_spec_t* spec,
                                     rocke_value_t* const* accs,
                                     int num_accs,
                                     const rocke_warp_grid_t* grid,
                                     rocke_value_t* d_rsrc);

/* _emit_direct_epilogue_wmma(b, spec, op, accs[n], warp_m_idx, warp_n_idx, lane,
 * block_m_off, block_n_off, d_rsrc, c0): WMMA per-lane fp16 store via the op's
 * c_layout map + the D descriptor. */
void rocke_conv_emit_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                          const rocke_implicit_gemm_conv_spec_t* spec,
                                          const rocke_mmaop_t* op,
                                          rocke_value_t* const* accs,
                                          int num_accs,
                                          rocke_value_t* warp_m_idx,
                                          rocke_value_t* warp_n_idx,
                                          rocke_value_t* lane,
                                          rocke_value_t* block_m_off,
                                          rocke_value_t* block_n_off,
                                          rocke_value_t* d_rsrc,
                                          rocke_value_t* c0);

/* _emit_cshuffle_epilogue(b, spec, accs[n], grid, d_rsrc): LDS-staged cshuffle
 * store via CShuffleEpilogue.from_grid + the D descriptor addr_fn. */
void rocke_conv_emit_cshuffle_epilogue(rocke_ir_builder_t* b,
                                       const rocke_implicit_gemm_conv_spec_t* spec,
                                       rocke_value_t* const* accs,
                                       int num_accs,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* d_rsrc);

/* ----- driver-internal ctx population (the build prologue, lines 787-1032) ----
 * Splitting the long prologue out of the public entry keeps the glue TU small;
 * the driver calls this then the K-loop + epilogue phases. On any ValueError /
 * is_valid_spec reject it sets the builder error and returns false. */
bool rocke_conv_build_ctx_init(rocke_conv_build_ctx_t* ctx,
                               rocke_ir_builder_t* b,
                               const rocke_implicit_gemm_conv_spec_t* spec,
                               const char* arch,
                               const rocke_conv_build_overrides_t* overrides);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_INTERNAL_H */
