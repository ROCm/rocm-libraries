/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_deep_fused_conv_pool_internal.h -- PRIVATE shared state +
 * phase-function contract for the C99 port of the gfx1151 (RDNA3.5 / Strix Halo,
 * wave32, WMMA 16x16x16) genuine-int8/int4 deep-fused conv0 -> conv1 -> maxpool
 * builder (rocke/instances/gfx1151/deep_fused_conv_pool.py, 3238 LOC).
 *
 * WHY THIS HEADER EXISTS.
 *   The gfx1151 module is NOT a thin shim over the family-agnostic common builder
 *   (cf. gfx950 / gfx1201, which wrap build_implicit_gemm_conv with 9 closures).
 *   build_deep_fused_conv_pool() here is a self-contained driver that authors the
 *   entire numeric body inline over its OWN IRBuilder: it resolves the WMMA / iu8
 *   / iu4 ops, builds the wave32 WarpGrid, allocates the LDS tiles, and threads a
 *   spec + builder + grid + the resolved ops + the staged LDS bases + the conv0 /
 *   conv1 accumulator vectors through ~30 module-level helper functions
 *   (staging, WMMA-GEMM, fragment-load, scatter, register-handoff, maxpool).
 *
 *   In C there is no closure capture and no Python free-function-with-(b, spec)-
 *   args ergonomics worth replicating per call. The faithful, chunkable port
 *   gathers EVERY variable that build_deep_fused_conv_pool shares across its
 *   helper calls into ONE context struct, rocke_gfx1151_dfcp_build_ctx_t, and turns
 *   each module-level Python function into a free function taking a pointer to it.
 *   The driver (rocke_build_gfx1151_deep_fused_conv_pool) populates the ctx in the
 *   exact order the Python prologue computes its locals
 *   (is_valid -> op/op0/op1 -> params -> grid -> attrs/policy -> LDS allocs),
 *   then walks the phases in Python order.
 *
 * MAPPING TO THE MAP'S "NINE CLOSURE PHASES".
 *   The port-map frames the build as nine closure phases (the gfx950/gfx1201
 *   common-driver vocabulary). For gfx1151 those nine roles are realised by the
 *   self-contained phase functions below, grouped by role:
 *     _extra_params               -> param declaration in the driver prologue
 *                                    (X / W0 / Y / W1) -- no separate fn.
 *     _m_index_fn / _a_mhw_index_fn-> the per-thread (oh, ow) / (ih, iw) decode
 *                                    inlined in the staging + maxpool phases.
 *     _setup_input_cache          -> rocke_gfx1151_dfcp_stage_input_footprint[_int]
 *     _setup_specialized_a_loader -> rocke_gfx1151_dfcp_stage_conv0_a[_int]
 *     _load_a_tile_from_cache     -> rocke_gfx1151_dfcp_wmma_gemm_from_lds[_int]
 *     _load_a_tile_specialized    -> rocke_gfx1151_dfcp_wmma_gemm_conv0_direct[_int]
 *     _load_a_operand_from_cache  -> rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint*
 *     _epilogue_override          -> the conv0->conv1 requant scatter / register
 *                                    handoff + conv1 GEMM + maxpool tail in the
 *                                    driver (using the scatter / fuse / maxpool
 *                                    phase fns below).
 *
 * CONTRACT STABILITY (bucket note).
 *   This header is the ONE shared surface every gfx1151 body-implementing agent
 *   binds to. It is DESIGNED TO BE COMPLETE: every local the Python body shares
 *   across helpers is a ctx field, and every module-level Python function has a
 *   prototype. A body agent MUST be able to read/write only ctx fields and call
 *   these prototypes WITHOUT editing this header; a missing value is a design bug
 *   to fix here once. ctx fields mirror Python local names 1:1; phase functions
 *   mirror the Python module-level names with a rocke_gfx1151_dfcp_ prefix.
 *
 * THIS HEADER EMITS NO IR AND DECLARES NO PUBLIC API. It is included only by the
 * gfx1151 deep-fused-conv-pool phase translation units.
 */
#ifndef ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_INTERNAL_H
#define ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_gfx1151_deep_fused_conv_pool.h"
/* The public spec type + geometry/atom macros (_WMMA / _WAVE / _OP_ID_* /
 * _K_PER_I32 / _I4_PER_I32) + the reused problem types. */
#include "rocke/ir.h"
/* rocke_ir_builder_t / rocke_value_t / rocke_kernel_def_t / rocke_status_t. */
#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
/* opaque rocke_warp_grid_t / rocke_mma_op_t. */

#ifdef __cplusplus
extern "C" {
#endif

/* Bounded per-warp accumulator count. The deep-fusion tile space is small (one
 * CTA owns all channels; tile_n<=32, warp_tile 16x16). mfmas_per_warp_m *
 * mfmas_per_warp_n is at most (tile_m/(warp_m*16)) * (tile_n/(warp_n*16)); for
 * the encoder_0 shapes this is a single-digit count. 64 is generous headroom and
 * matches the peer ports' convention. */
#define ROCKE_GFX1151_DFCP_MAX_ACCS 64

/* Bound on conv1 in-register A-fragments (fused_c0a1 handoff): mfmas_m x mfmas_n
 * grid, same bound as the accs. */
#define ROCKE_GFX1151_DFCP_MAX_AFRAGS ROCKE_GFX1151_DFCP_MAX_ACCS

/* ===================================================================== *
 *  rocke_gfx1151_dfcp_build_ctx_t
 *
 *  The single shared state object for the gfx1151 builder. Holds EVERY local of
 *  build_deep_fused_conv_pool that the helper functions read, grouped by
 *  lifetime: (A) build-time constants set once by the driver prologue, (B) the
 *  per-tile / per-callback scratch threaded through the staged phases.
 * ===================================================================== */
typedef struct rocke_gfx1151_dfcp_build_ctx
{
    /* =============================================================== *
     * (A) BUILD-TIME CONSTANTS -- set once by the driver prologue, read by every
     *     phase. Computed in Python prologue order: spec -> op/op0/op1 -> params
     *     -> grid -> policy -> kpad/derived.
     * =============================================================== */

    /* ---- inputs / resolved environment ---- */
    rocke_ir_builder_t* b; /* the IRBuilder `b` */
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec; /* the gfx1151 spec  */
    const char* arch; /* NULL-normalised to "gfx1151" */

    /* Convenience cached spec/problem views (== &spec->problem and its .conv),
     * staged so phases read fixed pointers instead of re-deriving. */
    const rocke_fused_conv_pool_problem_t* p; /* spec->problem */
    const rocke_conv_problem_t* c; /* spec->problem.conv */

    /* ---- resolved MMA ops (Python `op` / `op0` / `op1`) ----
     * op  = WMMA 16x16x16 f16 atom (the shape source for the WarpGrid + the
     *       non-native numeric path).
     * op0 = native_int ? iu8 atom : op   (conv0 contraction).
     * op1 = native_int ? iu4 atom : op   (conv1 contraction). */
    const rocke_mma_op_t* op;
    const rocke_mma_op_t* op0;
    const rocke_mma_op_t* op1;

    /* ---- kernel params (Values, Python `X`/`W0`/`Y`/`W1`) ----
     * X  : PtrType(I8 ,"global") noalias readonly  align16  -- int8 activations.
     * W0 : PtrType(I8 ,"global") noalias readonly  align16  -- int8 conv0 weights.
     * Y  : PtrType(I32,"global") noalias writeonly align16  -- packed-int4 output.
     * W1 : PtrType(I8 ,"global") noalias readonly  align16  -- packed-int4 conv1 W. */
    rocke_value_t* X;
    rocke_value_t* W0;
    rocke_value_t* Y;
    rocke_value_t* W1;

    /* ---- the per-CTA WarpGrid (Python `grid`) ----
     * WarpGrid.from_atom(op, tile_m, tile_n, tile_k=_WMMA, warp_m, warp_n,
     *   wave_size=_WAVE).bind(b, block_m_axis="y", block_n_axis="x").
     * Holds the SSA tid/lane/warp decode + mfmas_per_warp_m/n + warp_m_off/
     * warp_n_off the GEMM / scatter helpers read. */
    const rocke_warp_grid_t* grid;

    /* ---- scheduler policy (Python `policy`) ----
     * SchedulePolicy.for_pipeline(spec.sched_policy); drives the per-k-atom
     * DS_READ/MFMA sched_group_barrier hints in _emit_wmma_k_sched. NULL / no
     * hints for "mem". Opaque to phases (passed to rocke_gfx1151_dfcp_emit_wmma_k_sched). */
    const void* policy;

    /* ---- derived geometry cached from the spec (Python prologue locals) ----
     * kpad = rocke_gfx1151_dfcp_kpad(spec). block_size, conv_tile_h/w, foot_h/w are
     * available via the public accessors; kpad is hot enough to cache. */
    int kpad;

    /* ---- LDS dtypes resolved per lever (Python `a0_dtype` etc) ----
     * a0_dtype = native_int ? I8 : F16 (footprint / im2col A tile + W0 tile).
     * c0_dtype = w1_dtype = c1_dtype = native_int ? I8 : F16. Stored as the
     * ckc IR dtype tag so the alloc phase need not re-branch. */
    int a0_dtype;
    int c0_dtype;
    int w1_dtype;
    int c1_dtype;

    /* =============================================================== *
     * (B) STAGED LDS BASES + ACCUMULATORS -- allocated / produced as the driver
     *     walks the phases. A phase reads the bases it consumes from here.
     * =============================================================== */

    /* ---- LDS tile bases (Python `a0_smem`/`w0_smem`/`c0_smem`/... Values) ----
     * a0_smem    : direct_conv0 ? INP footprint [foot_h*foot_w, C]
     *                            : im2col A tile [tile_m, kpad].
     * w0_smem    : [tile_n, kpad] conv0 weight tile.
     * c0_smem    : [tile_m, c0_cols] conv0 requant codes; NULL when fused_c0a1
     *              (register handoff, no C0 LDS). c0_cols = packed_c0 ? K/2 : K.
     * c0_packed_smem : [tile_m, K/2] extra packed-byte buffer; non-NULL only when
     *              repack_c0.
     * w1_smem    : [tile_n, w1_cols] conv1 weight tile. w1_cols = native_int ?
     *              (conv1_int8 ? K : K/2) : K.
     * c1_smem    : [tile_m, tile_n] conv1 requant codes (maxpool input). */
    rocke_value_t* a0_smem;
    rocke_value_t* w0_smem;
    rocke_value_t* c0_smem;
    rocke_value_t* c0_packed_smem;
    rocke_value_t* w1_smem;
    rocke_value_t* c1_smem;

    /* ---- conv0 / conv1 accumulator vectors (Python `accs0` / `accs1`) ----
     * accs0: mfmas_m*mfmas_n vectors (<8 x i32> native / <... x f32> fp16) from
     *        the conv0 WMMA GEMM. accs1: same shape from the conv1 GEMM. */
    rocke_value_t* accs0[ROCKE_GFX1151_DFCP_MAX_ACCS];
    size_t num_accs0;
    rocke_value_t* accs1[ROCKE_GFX1151_DFCP_MAX_ACCS];
    size_t num_accs1;

    /* ---- conv1 in-register A-fragments (Python `a_frags`, fused_c0a1 only) ----
     * a_frags[mi][kk] from _fuse_c0_to_conv1_a_regs; flattened row-major
     * (index = mi*mfmas_n + kk). NULL slots when not fused_c0a1. */
    rocke_value_t* a_frags[ROCKE_GFX1151_DFCP_MAX_AFRAGS];
    size_t num_a_frags;
    int a_frags_rows; /* mfmas_m */
    int a_frags_cols; /* mfmas_n (== conv1 K atoms) */

    /* ---- persistent grid-stride loop scratch (Python persistent branch) ----
     * Per-tile loop-carried coords threaded into the _int staging / maxpool
     * phases via their h_blk/w_blk args; staged here for the loop body. NULL in
     * the single-tile path (the phases then fall back to block ids). */
    rocke_value_t* h_blk; /* readfirstlane(tile_idx) / w_tiles */
    rocke_value_t* w_blk; /* readfirstlane(tile_idx) % w_tiles */

    /* ---- requant closure constants (Python body locals 2915-2917 / 3049) ----
     * The fp16 conv0/conv1 code closures capture spec.m0/m0b/m1 + 0.0 as f32
     * consts MATERIALIZED ONCE at the body level: c_m0/c_m0b/zero_f right after
     * the conv0 GEMM (before conv1 staging), c_m1 right after the conv1 GEMM.
     * The code-fn emitters read these cached SSA values instead of re-creating
     * the consts per-slot, so the value numbering matches the Python closures.
     * NULL until the body driver creates them. */
    rocke_value_t* c_m0; /* const_f32(spec.m0)  */
    rocke_value_t* c_m0b; /* const_f32(spec.m0b) */
    rocke_value_t* c_m1; /* const_f32(spec.m1)  */
    rocke_value_t* zero_f; /* const_f32(0.0)      */
} rocke_gfx1151_dfcp_build_ctx_t;

/* Zero-initialise the ctx and populate the (A) build-time constants from the
 * gfx1151 spec + arch (Python prologue lines 2688-2778): runs the is_valid_spec
 * gate, normalises arch to "gfx1151", resolves op / op0 / op1 (WMMA f16 + iu8 /
 * iu4 via ArchTarget.from_gfx), declares the X/W0/Y/W1 params, builds + binds the
 * WarpGrid, stamps the waves_per_eu launch bound + SchedulePolicy, resolves the
 * a0/c0/w1/c1 LDS dtypes and kpad, and allocates the LDS tiles (a0/inp, w0, c0,
 * c0_packed, w1, c1) sized per the levers. The accs / a_frags / h_blk / w_blk are
 * left zeroed for the phase walk. Returns ROCKE_OK or a status mirroring the Python
 * ValueError paths (message in b->err). */
rocke_status_t
    rocke_gfx1151_dfcp_build_ctx_init(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                      rocke_ir_builder_t* b,
                                      const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
                                      const char* arch);

/* ===================================================================== *
 *  QUANTIZATION PRIMITIVES (Python lines 648-741) -- pure IR emitters, no ctx.
 *  Each takes the builder + operand Values and emits the byte-faithful op
 *  sequence. The scalar forms feed the per-slot scatter code_fns; the _vec_i32
 *  forms feed the specialized_rne whole-accumulator path.
 * ===================================================================== */

/* clamp(round(v*inv_scale), -127,127) -> i8 (RNE). */
rocke_value_t* rocke_gfx1151_dfcp_quant_i8(rocke_ir_builder_t* b,
                                           rocke_value_t* vf32,
                                           rocke_value_t* inv_scale);
/* clamp(round(v*inv_scale), -8,7) -> i8 holding an int4 code (RNE). */
rocke_value_t* rocke_gfx1151_dfcp_quant_i4(rocke_ir_builder_t* b,
                                           rocke_value_t* vf32,
                                           rocke_value_t* inv_scale);
/* sitofp_f32(sext(qi8, I32)). */
rocke_value_t* rocke_gfx1151_dfcp_i8_to_f32(rocke_ir_builder_t* b, rocke_value_t* qi8);
/* 0 - x. */
rocke_value_t* rocke_gfx1151_dfcp_neg_i32(rocke_ir_builder_t* b, rocke_value_t* x);
/* smin(smax(x, lo), hi). */
rocke_value_t*
    rocke_gfx1151_dfcp_clamp_i32(rocke_ir_builder_t* b, rocke_value_t* x, int lo, int hi);
/* smax(x, 0). */
rocke_value_t* rocke_gfx1151_dfcp_relu_i32(rocke_ir_builder_t* b, rocke_value_t* x);
/* RNE round x / 2**shift using integer ops only (sign-aware, ties-to-even). */
rocke_value_t*
    rocke_gfx1151_dfcp_round_shift_rne_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift);
/* trunc(clamp_i32(round_shift_rne_i32(x, shift), -127,127), I8). */
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i8_shift(rocke_ir_builder_t* b, rocke_value_t* x, int shift);
/* trunc(clamp_i32(round_shift_rne_i32(x, shift), -8,7), I8). */
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i4_shift(rocke_ir_builder_t* b, rocke_value_t* x, int shift);

/* Vector (<N x i32>) twins of the scalar primitives (specialized_rne path). */
rocke_value_t* rocke_gfx1151_dfcp_splat_i32(rocke_ir_builder_t* b, int value, int n);
rocke_value_t* rocke_gfx1151_dfcp_neg_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x);
rocke_value_t*
    rocke_gfx1151_dfcp_clamp_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x, int lo, int hi);
rocke_value_t* rocke_gfx1151_dfcp_relu_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x);
rocke_value_t*
    rocke_gfx1151_dfcp_round_shift_rne_i32_vec(rocke_ir_builder_t* b, rocke_value_t* x, int shift);
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i8_shift_vec_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift);
rocke_value_t*
    rocke_gfx1151_dfcp_quant_i4_shift_vec_i32(rocke_ir_builder_t* b, rocke_value_t* x, int shift);

/* ===================================================================== *
 *  STAGING PHASES (global -> LDS) -- Python lines 744-1379.
 *  Each reads ctx (spec/grid/builder + the source param + the dest LDS base) and
 *  emits the per-thread im2col / footprint load loop. The fp16 forms convert int
 *  codes to f16; the _int forms stage raw i8. The _int footprint / maxpool forms
 *  accept optional persistent loop-carried tile coords (h_blk/w_blk == NULL =>
 *  fall back to the per-CTA block ids, byte-identical).
 * ===================================================================== */

/* _stage_conv0_a: im2col int8 activations -> a_smem as f16 codes [tile_m, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_a(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                      rocke_value_t* x_ptr,
                                      rocke_value_t* a_smem);
/* _stage_input_footprint: direct-conv f16 footprint cache [foot_h*foot_w, C]. */
void rocke_gfx1151_dfcp_stage_input_footprint(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                              rocke_value_t* x_ptr,
                                              rocke_value_t* inp_smem);
/* _stage_input_footprint_int: direct native-int raw-i8 footprint cache; uses the
 * batch_loads / interior_fastpath levers; persistent coords via h_blk/w_blk. */
void rocke_gfx1151_dfcp_stage_input_footprint_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                  rocke_value_t* x_ptr,
                                                  rocke_value_t* inp_smem,
                                                  rocke_value_t* h_blk,
                                                  rocke_value_t* w_blk);
/* _stage_conv0_w0: int8 conv0 weights -> w0_smem as f16 codes [tile_n, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_w0(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* w0_ptr,
                                       rocke_value_t* w0_smem);
/* _stage_conv0_a_int: native-int im2col raw-i8 activations -> a_smem [tile_m, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_a_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                          rocke_value_t* x_ptr,
                                          rocke_value_t* a_smem);
/* _stage_conv0_w0_int: native-int raw-i8 conv0 weights -> w0_smem [tile_n, kpad]. */
void rocke_gfx1151_dfcp_stage_conv0_w0_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                           rocke_value_t* w0_ptr,
                                           rocke_value_t* w0_smem);
/* _stage_conv1_w1: unpack packed-int4 W1 -> w1_smem as f16 codes [tile_n, K0].
 * USES the NEW helper rocke_unpack_i4_byte_to_pair_i32 (rocke.helpers.i4_dequant). */
void rocke_gfx1151_dfcp_stage_conv1_w1(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                       rocke_value_t* w1_ptr,
                                       rocke_value_t* w1_smem);
/* _stage_conv1_w1_packed: stage packed-int4 W1 bytes into LDS without unpacking. */
void rocke_gfx1151_dfcp_stage_conv1_w1_packed(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                              rocke_value_t* w1_ptr,
                                              rocke_value_t* w1_smem);
/* _stage_conv1_w1_i8: unpack packed-int4 W1 -> byte-per-code i8 LDS [tile_n, K0]
 * for the iu8 conv1 atom. USES rocke_unpack_i4_byte_to_pair_i32. */
void rocke_gfx1151_dfcp_stage_conv1_w1_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                          rocke_value_t* w1_ptr,
                                          rocke_value_t* w1_smem);

/* ===================================================================== *
 *  FRAGMENT LOADERS -- Python lines 1382-1459, 2011-2149.
 *  Per-WMMA-fragment LDS / footprint gather helpers; pure IR (builder + Values).
 * ===================================================================== */

/* _load_frag_iu8_from_lds: <16 x i8> row -> <4 x i32> iu8 fragment (bitcast). */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu8_from_lds(rocke_ir_builder_t* b,
                                                         rocke_value_t* smem,
                                                         rocke_value_t* frag_rc,
                                                         rocke_value_t* atom_off,
                                                         rocke_value_t* k_tile_base);
/* _pack_i4_codes_to_i32: <8 x i8> int4 codes -> one i32, low nibble first. */
rocke_value_t* rocke_gfx1151_dfcp_pack_i4_codes_to_i32(rocke_ir_builder_t* b, rocke_value_t* codes);
/* _load_frag_iu4_codes_from_lds: byte-per-code C0 -> <2 x i32> iu4 A-fragment. */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu4_codes_from_lds(rocke_ir_builder_t* b,
                                                               rocke_value_t* smem,
                                                               rocke_value_t* frag_rc,
                                                               rocke_value_t* atom_off,
                                                               rocke_value_t* k_tile_base);
/* _load_frag_iu4_packed_from_lds: packed-byte W1 -> <2 x i32> iu4 B-fragment. */
rocke_value_t* rocke_gfx1151_dfcp_load_frag_iu4_packed_from_lds(rocke_ir_builder_t* b,
                                                                rocke_value_t* smem,
                                                                rocke_value_t* frag_rc,
                                                                rocke_value_t* atom_off,
                                                                rocke_value_t* k_tile_base);
/* _load_conv0_a_frag_from_footprint: gather f16 conv0 A-fragment from footprint. */
rocke_value_t*
    rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                        rocke_value_t* inp_smem,
                                                        rocke_value_t* m_row,
                                                        rocke_value_t* k_base,
                                                        int frag_len);
/* _load_conv0_a_frag_from_footprint_iu8: gather+pack iu8 conv0 A-fragment. */
rocke_value_t*
    rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                            rocke_value_t* inp_smem,
                                                            rocke_value_t* m_row,
                                                            rocke_value_t* k_base);
/* _load_conv0_a_frag_from_footprint_iu8_static: static-K-map (C=8,R=S=3) twin. */
rocke_value_t* rocke_gfx1151_dfcp_load_conv0_a_frag_from_footprint_iu8_static(
    rocke_gfx1151_dfcp_build_ctx_t* ctx, rocke_value_t* inp_smem, rocke_value_t* m_row, int kk);

/* ===================================================================== *
 *  WMMA GEMM PHASES -- Python lines 1462-2276.
 *  Each accumulates A @ B.T over the K atoms and returns mfmas_m*mfmas_n accs
 *  into the provided out-array (out[ROCKE_GFX1151_DFCP_MAX_ACCS], *out_count set).
 *  `op`, `policy` and the lever flags are threaded explicitly (the driver passes
 *  ctx->op0 / ctx->op1 and the relevant spec levers).
 * ===================================================================== */

/* _emit_wmma_k_sched: per-k-atom DS_READ->MFMA sched_group_barrier hint
 * (no-op when policy is NULL or hints off). */
void rocke_gfx1151_dfcp_emit_wmma_k_sched(rocke_ir_builder_t* b,
                                          const void* policy,
                                          int n_ds,
                                          int n_mma);

/* _wmma_gemm_from_lds: generic f16 WMMA GEMM from two LDS tiles. */
void rocke_gfx1151_dfcp_wmma_gemm_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                           const rocke_mma_op_t* op,
                                           rocke_value_t* a_smem,
                                           rocke_value_t* b_smem,
                                           int k_total,
                                           const void* policy,
                                           rocke_value_t** out_accs,
                                           size_t* out_count);
/* _wmma_gemm_from_lds_int: native-int iu8 WMMA GEMM from two i8 LDS tiles. */
void rocke_gfx1151_dfcp_wmma_gemm_from_lds_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                               const rocke_mma_op_t* op,
                                               rocke_value_t* a_smem,
                                               rocke_value_t* b_smem,
                                               int k_total,
                                               const void* policy,
                                               rocke_value_t** out_accs,
                                               size_t* out_count);
/* _wmma_gemm_conv0_direct: direct-conv f16 conv0 GEMM (A from footprint cache). */
void rocke_gfx1151_dfcp_wmma_gemm_conv0_direct(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                               const rocke_mma_op_t* op,
                                               rocke_value_t* inp_smem,
                                               rocke_value_t* w0_smem,
                                               const void* policy,
                                               rocke_value_t** out_accs,
                                               size_t* out_count);
/* _wmma_gemm_conv0_direct_int: direct native-int iu8 conv0 GEMM; `reorient`
 * swaps the WMMA operands for the fused_c0a1 in-register handoff (acc[k0,m]). */
void rocke_gfx1151_dfcp_wmma_gemm_conv0_direct_int(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                   const rocke_mma_op_t* op,
                                                   rocke_value_t* inp_smem,
                                                   rocke_value_t* w0_smem,
                                                   const void* policy,
                                                   bool reorient,
                                                   rocke_value_t** out_accs,
                                                   size_t* out_count);
/* _wmma_gemm_conv1_i4_from_lds: native iu4 conv1 GEMM (C0 byte codes + packed W1);
 * prefetch_k / sched_fuse levers. */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    const rocke_mma_op_t* op,
                                                    rocke_value_t* c0_smem,
                                                    rocke_value_t* w1_smem,
                                                    int k_total,
                                                    const void* policy,
                                                    bool prefetch_k,
                                                    bool sched_fuse,
                                                    rocke_value_t** out_accs,
                                                    size_t* out_count);
/* _wmma_gemm_conv1_i4_packed_from_lds: iu4 conv1 GEMM with both A(C0) and B(W1)
 * packed in LDS (packed_c0_handoff / repack_c0 paths). */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_packed_from_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                           const rocke_mma_op_t* op,
                                                           rocke_value_t* c0_smem,
                                                           rocke_value_t* w1_smem,
                                                           int k_total,
                                                           const void* policy,
                                                           rocke_value_t** out_accs,
                                                           size_t* out_count);
/* _wmma_gemm_conv1_i4_from_regs: iu4 conv1 GEMM, A from the fused register handoff
 * (ctx->a_frags), B(W1) from packed LDS. */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i4_from_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                     const rocke_mma_op_t* op,
                                                     rocke_value_t* const* a_frags,
                                                     int a_rows,
                                                     int a_cols,
                                                     rocke_value_t* w1_smem,
                                                     int k_total,
                                                     const void* policy,
                                                     bool prefetch_k,
                                                     bool sched_fuse,
                                                     rocke_value_t** out_accs,
                                                     size_t* out_count);
/* _wmma_gemm_conv1_i8_from_regs: iu8 conv1 GEMM (conv1_int8), A from the fused
 * register handoff, B(W1) byte-per-code i8 LDS. */
void rocke_gfx1151_dfcp_wmma_gemm_conv1_i8_from_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                     const rocke_mma_op_t* op,
                                                     rocke_value_t* const* a_frags,
                                                     int a_rows,
                                                     int a_cols,
                                                     rocke_value_t* w1_smem,
                                                     int k_total,
                                                     const void* policy,
                                                     bool prefetch_k,
                                                     bool sched_fuse,
                                                     rocke_value_t** out_accs,
                                                     size_t* out_count);

/* ===================================================================== *
 *  CONV0 -> CONV1 HANDOFF (register transpose) -- Python lines 1644-1707.
 * ===================================================================== */

/* _fuse_c0_to_conv1_a_regs: in-register conv0->conv1 handoff (permlanex16
 * transpose) from the REORIENTED conv0 accs -> conv1 A-fragments. `code_fn`
 * is one of the conv0 requant code functions (here passed as an enum/selector
 * the body switches on; see rocke_gfx1151_dfcp_code_fn_t). `int8` => unsqueezed
 * <4 x i32> iu8 A-fragments (conv1_int8). Writes a_frags row-major into
 * out_a_frags (mfmas_m x mfmas_n), sets *out_count / *out_rows / *out_cols. */
void rocke_gfx1151_dfcp_fuse_c0_to_conv1_a_regs(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                const rocke_mma_op_t* op0,
                                                rocke_value_t* const* accs0,
                                                size_t num_accs0,
                                                int code_fn,
                                                bool int8,
                                                rocke_value_t** out_a_frags,
                                                size_t* out_count,
                                                int* out_rows,
                                                int* out_cols);

/* ===================================================================== *
 *  REQUANT CODE FUNCTIONS (conv0 / conv1 epilogue closures) -- Python
 *  lines 2786-2793, 2919-2940, 3051-3066. The Python passes these as closures to
 *  the scatter / fuse phases; in C the scatter phases take a selector enum and
 *  apply the matching emitter to each accumulator slot, so the driver does not
 *  marshal function pointers. The enum values name the Python closures.
 * ===================================================================== */
typedef enum rocke_gfx1151_dfcp_code_fn
{
    ROCKE_GFX1151_DFCP_CODE_CONV0_I8 = 0, /* conv0_code_i8  (native: i8-shift->relu->i4-shift) */
    ROCKE_GFX1151_DFCP_CODE_CONV0_F16, /* conv0_code_f16 (fp16 path, truncs to f16)         */
    ROCKE_GFX1151_DFCP_CODE_CONV0_VEC_I8, /* conv0_code_vec_i8 (specialized_rne whole-acc)     */
    ROCKE_GFX1151_DFCP_CODE_CONV1_I8, /* conv1_code_i8                                       */
    ROCKE_GFX1151_DFCP_CODE_CONV1_F16, /* conv1_code_f16                                      */
    ROCKE_GFX1151_DFCP_CODE_CONV1_VEC_I8 /* conv1_code_vec_i8                                   */
} rocke_gfx1151_dfcp_code_fn_t;

/* Apply a scalar conv0/conv1 code function to one acc-slot f32/i32 Value. */
rocke_value_t* rocke_gfx1151_dfcp_apply_code_fn(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                int code_fn,
                                                rocke_value_t* slot);
/* Apply a vector (whole-accumulator) code function (specialized_rne). */
rocke_value_t* rocke_gfx1151_dfcp_apply_vec_code_fn(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    int code_fn,
                                                    rocke_value_t* acc);

/* ===================================================================== *
 *  SCATTER PHASES (accumulators -> LDS) -- Python lines 2279-2452.
 *  Quantize each WMMA acc slot to its requant code and store at (row, col) in the
 *  destination LDS tile, using the code-fn selector above.
 * ===================================================================== */

/* _scatter_codes_to_lds: f16 codes (fp16 path). */
void rocke_gfx1151_dfcp_scatter_codes_to_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                             const rocke_mma_op_t* op,
                                             rocke_value_t* const* accs,
                                             size_t num_accs,
                                             rocke_value_t* dst_smem,
                                             int code_fn);
/* _scatter_codes_to_i8_lds: byte-per-code i8 (native path). */
void rocke_gfx1151_dfcp_scatter_codes_to_i8_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                const rocke_mma_op_t* op,
                                                rocke_value_t* const* accs,
                                                size_t num_accs,
                                                rocke_value_t* dst_smem,
                                                int code_fn);
/* _scatter_vec_codes_to_i8_lds: vector-quantize one acc then scatter i8 lanes
 * (specialized_rne). */
void rocke_gfx1151_dfcp_scatter_vec_codes_to_i8_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                    const rocke_mma_op_t* op,
                                                    rocke_value_t* const* accs,
                                                    size_t num_accs,
                                                    rocke_value_t* dst_smem,
                                                    int vec_code_fn);
/* _scatter_packed_i4_codes_to_lds: pack adjacent C0 columns one byte/even-lane
 * (packed_c0_handoff; uses ds_bpermute). */
void rocke_gfx1151_dfcp_scatter_packed_i4_codes_to_lds(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                       const rocke_mma_op_t* op,
                                                       rocke_value_t* const* accs,
                                                       size_t num_accs,
                                                       rocke_value_t* dst_smem,
                                                       int code_fn);
/* _repack_c0_lds_to_packed: lane-local LDS->LDS repack of byte-per-code C0 into
 * packed bytes (repack_c0). */
void rocke_gfx1151_dfcp_repack_c0_lds_to_packed(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                rocke_value_t* c0_smem,
                                                rocke_value_t* c0_packed_smem);

/* ===================================================================== *
 *  MAXPOOL FINALIZATION PHASES -- Python lines 2455-2682.
 *  One thread per pooled pixel: 2x2 max over conv1 codes, final int4 quant,
 *  pack channels into i32 words, store to Y. Persistent coords via h_blk/w_blk
 *  (NULL => block ids). mask_maxpool / vectorize_maxpool / pk_maxpool levers.
 * ===================================================================== */

/* _emit_maxpool_finalquant: fp16 path (f16 codes -> f32 max -> i4 quant). */
void rocke_gfx1151_dfcp_emit_maxpool_finalquant(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                rocke_value_t* c1_smem,
                                                rocke_value_t* y_ptr);
/* _emit_maxpool_finalpack_i8: native path (i8 int4 codes; mf=1 pack is no-op);
 * persistent coords via h_blk/w_blk. */
void rocke_gfx1151_dfcp_emit_maxpool_finalpack_i8(rocke_gfx1151_dfcp_build_ctx_t* ctx,
                                                  rocke_value_t* c1_smem,
                                                  rocke_value_t* y_ptr,
                                                  rocke_value_t* h_blk,
                                                  rocke_value_t* w_blk);

/* ===================================================================== *
 *  DRIVER SUB-PHASES -- the two top-level body branches of
 *  build_deep_fused_conv_pool (Python lines 2780-3085). Split out so the body
 *  can be chunked: the persistent grid-stride loop and the single-tile pipeline.
 *  Both consume the fully-initialised ctx (ops, grid, LDS bases) and emit the
 *  whole numeric body for their branch.
 * ===================================================================== */

/* Persistent grid-stride variant (native_int + direct_conv0 + fused_c0a1): stage
 * W0/W1 once, scf_for over the flattened tile strip streaming X/Y per tile. */
void rocke_gfx1151_dfcp_emit_persistent_body(rocke_gfx1151_dfcp_build_ctx_t* ctx);
/* Single-tile variant: conv0 (direct/im2col x native/fp16) -> requant scatter /
 * register handoff -> conv1 (iu4/iu8/fp16) -> requant scatter -> maxpool. */
void rocke_gfx1151_dfcp_emit_single_tile_body(rocke_gfx1151_dfcp_build_ctx_t* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_INTERNAL_H */
