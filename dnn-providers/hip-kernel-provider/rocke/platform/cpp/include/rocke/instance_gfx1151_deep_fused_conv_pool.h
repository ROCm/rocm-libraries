/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_deep_fused_conv_pool.h -- PUBLIC C99 API for the gfx1151
 * (RDNA3.5 / Strix Halo, wave32, WMMA 16x16x16) GENUINE-int8/int4 deep-fused
 * conv0 -> conv1 -> 2x2/s2 maxpool kernel, ported from
 *   rocke/instances/gfx1151/deep_fused_conv_pool.py  (3238 LOC).
 *
 * WHAT THIS PORT IS (NOT a thin re-export).
 *   Unlike the gfx950 / gfx1201 deep-fused shims -- which are ~70-LOC wrappers
 *   that pin geometry and re-export the family-agnostic common builder
 *   (rocke/instance_deep_fused_conv_pool.h) driven by a resolved MmaOp -- the
 *   gfx1151 module is a FULL architecture-specific spec + builder. It does NOT
 *   call build_implicit_gemm_conv; it authors the entire numeric body inline
 *   over its own IRBuilder with ~30 module-level staging / WMMA-GEMM / scatter /
 *   maxpool helpers and a genuine low-bit int8/int4 quantization path.
 *
 *   It computes the encoder_0 block in ONE kernel, no conv0/conv1 intermediate
 *   in HBM:
 *     conv0 3x3 (int8) -> Quant(i32->i8) -> ReLU -> Quant(i8->i4)
 *     -> conv1 1x1 (int4) -> Quant(i32->i4) -> ReLU
 *     -> 2x2/s2 MaxPool -> Quant(i4->i4) -> packed-int4 output.
 *   Each CTA owns a rectangular tile of pooled outputs (backward-planned: pooled
 *   tile -> conv1 patch -> conv0 region -> input halo).
 *
 *   gfx1151 has no int8/int4 matrix cores in the f16 path: the integer operands
 *   are dequantized to f16 and fed to wmma_f32_16x16x16_f16 with f32 accumulation
 *   (bit-exact to a native integer MMA over these ranges). The native_int lever
 *   additionally exposes the wmma_i32_16x16x16_iu8 (conv0) / _iu4 (conv1) atoms.
 *
 *   Python (gfx1151/deep_fused_conv_pool.py)        C99 (this header)
 *   ---------------------------------------------   --------------------------------------
 *   @dataclass Gfx1151DeepFusedConvPoolSpec         rocke_gfx1151_deep_fused_conv_pool_spec_t
 *   make_deep_fused_conv_pool_spec(...)             rocke_gfx1151_deep_fused_conv_pool_make_spec(...)
 *   is_valid_spec(spec, arch) rocke_gfx1151_deep_fused_conv_pool_is_valid_spec(...)
 *   deep_fused_conv_pool_grid(spec)                 rocke_gfx1151_deep_fused_conv_pool_grid(...)
 *   spec.kernel_name() rocke_gfx1151_deep_fused_conv_pool_kernel_name(...)
 *   build_deep_fused_conv_pool(spec, arch)          rocke_build_gfx1151_deep_fused_conv_pool(...)
 *   (+ convenience: init+build)                     rocke_build_gfx1151_deep_fused_conv_pool_new(...)
 *   (+ convenience: build -> lower .ll) rocke_gfx1151_deep_fused_conv_pool_lower_to_llvm(...)
 *
 * SYMBOL PREFIX.
 *   ALL gfx1151 symbols carry the gfx1151_deep_fused_conv_pool / gfx1151_dfcp
 *   prefix so they never clash with the same-named common/ port (bare
 *   rocke_deep_fused_conv_pool_* / rocke_dfcp_*) or the gfx950 / gfx1201 shims.
 *
 * REUSED PORTED SURFACE (header-opaque forward decls).
 *   - rocke_fused_conv_pool_problem_t / rocke_conv_problem_t (the problem geometry),
 *   - rocke_unpack_i4_byte_to_pair_i32 (rocke.helpers.i4_dequant -- the one NEW
 *     helper symbol this kernel needs, used by _stage_conv1_w1 / _stage_conv1_w1_i8),
 *   - the opaque rocke_warp_grid_t / rocke_mma_op_t MMA + WarpGrid handles
 *     (resolved internally to the WMMA 16x16x16 f16 op + the iu8 / iu4 atoms).
 *
 * Error model mirrors the rest of the C port: build / lower route errors through
 * the sticky-error IRBuilder; the validity gate returns a bool + reason; the
 * convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_H
#define ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.deep_fused_conv_pool.h"
/* rocke_fused_conv_pool_problem_t, rocke_conv_problem_t, opaque rocke_warp_grid_t /
 * rocke_mma_op_t. */
#include "rocke/arena.h" /* rocke_arena_t */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_kernel_def_t, rocke_status_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * Pinned gfx1151 geometry / atom identifiers (Python module constants).
 * ------------------------------------------------------------------ */

/* The default kernel name root (Python `Gfx1151DeepFusedConvPoolSpec.name`); the
 * full name is composed by kernel_name() with problem + tile + lever tags. */
#define ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_NAME "rocke_gfx1151_deep_fused_conv_pool"

/* Python module constants: WMMA atom is 16x16x16, wave32. */
#define ROCKE_GFX1151_DFCP_WMMA 16 /* Python `_WMMA` */
#define ROCKE_GFX1151_DFCP_WAVE 32 /* Python `_WAVE` */

/* Native integer WMMA atom op-ids (Python `_OP_ID_IU8` / `_OP_ID_IU4`). */
#define ROCKE_GFX1151_DFCP_OP_ID_IU8 "wmma_i32_16x16x16_iu8"
#define ROCKE_GFX1151_DFCP_OP_ID_IU4 "wmma_i32_16x16x16_iu4"

/* int8 K-values packed per i32 fragment slot / int4 per i32 (Python
 * `_K_PER_I32` / `_I4_PER_I32`). */
#define ROCKE_GFX1151_DFCP_K_PER_I32 4
#define ROCKE_GFX1151_DFCP_I4_PER_I32 8

/* The arch strings this builder accepts (Python is_valid_spec gate). */
#define ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_ARCH "gfx1151"
#define ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_ARCH_GENERIC "gfx11-generic"

/* ------------------------------------------------------------------ *
 * Gfx1151DeepFusedConvPoolSpec
 * ------------------------------------------------------------------ *
 *
 * Faithful field-by-field mirror of the Python @dataclass(frozen=True)
 * Gfx1151DeepFusedConvPoolSpec (lines 74-296). This is its OWN spec type (NOT a
 * wrapper over the common spec): it carries the genuine-int8/int4 levers
 * (native_int / fused_c0a1 / persistent / conv1_int8 / ...) and the four
 * per-node inverse requant multipliers (m0 / m0b / m1 / mf) that the common
 * family-agnostic spec does not have. Booleans are correctness-neutral perf
 * levers unless noted; see the Python field comments for the per-lever rationale
 * and measured A/B verdicts. Field order matches the Python declaration order.
 *
 * Derived quantities (Python @property) are NOT stored; compute them with the
 * accessor helpers below (warp_tile_*, block_size, kpad, conv_tile_h/w,
 * foot_h/w) so the single source of truth stays the spec + problem. */
typedef struct rocke_gfx1151_deep_fused_conv_pool_spec
{
    /* Problem geometry (Python `problem`). */
    rocke_fused_conv_pool_problem_t problem;

    /* Kernel-name root (Python `name`, default ROCKE_GFX1151_DEEP_FUSED_CONV_POOL_NAME). */
    const char* name;

    /* Tile geometry. tile_m is auto-derived from the pool tile by the factory. */
    int tile_m; /* default 256 (= conv_tile_h*conv_tile_w) */
    int tile_n; /* default 32  */
    int pool_tile_h; /* default 8   */
    int pool_tile_w; /* default 8   */
    int warp_m; /* default 4   */
    int warp_n; /* default 2   */

    /* Optimization toggles (correctness-neutral; in-process A/B benching). */
    bool vectorize_conv0_a; /* default true  */
    bool vectorize_maxpool; /* default true  */
    bool early_w1; /* default true  */
    bool direct_conv0; /* default true  -- direct conv (footprint cache) vs im2col */
    bool w_fast; /* default false -- grid dispatch order */

    /* Multi-lever latency-hiding campaign toggles (correctness-neutral). */
    int waves_per_eu; /* default 0     -- 0 => unset (compiler default) */
    const char* sched_policy; /* default "mem" -- {mem,compv3,compv4,intrawave} */
    bool mask_maxpool; /* default false -- branch-free maxpool tail */

    /* Native integer cleanup levers. */
    bool specialized_rne; /* default false */
    bool interior_fastpath; /* default false */
    bool static_direct_kmap; /* default false -- assumes C=8, R=S=3 */
    bool packed_c0_handoff; /* default false */
    bool repack_c0; /* default false (documented negative result) */
    bool fused_c0a1; /* default false -- in-register conv0->conv1 handoff */
    bool butterfly_conv01; /* default false (analyzed non-lever; rejected) */
    bool native_int; /* default false -- iu8/iu4 native integer WMMA path */
    bool batch_loads; /* default true  -- VGPR-staged footprint load batching */
    bool pk_maxpool; /* default false -- packed-int16 maxpool reduction */
    bool conv1_prefetch_k; /* default false -- conv1 cross-k-step frag prefetch */
    bool conv1_sched_fuse; /* default false -- conv1 fused k-step schedule group */
    bool conv1_int8; /* default false -- conv1 contraction in iu8 not iu4 */
    bool persistent; /* default false -- persistent grid-stride kernel */
    int persistent_ctas; /* default 16    -- resident CTA count when persistent */

    /* Per-node inverse requant multipliers (fold act/weight/out scales). */
    float m0; /* default 0.0625 -- conv0 i32 -> i8  */
    float m0b; /* default 0.5    -- conv0 i8  -> i4  */
    float m1; /* default 0.25   -- conv1 i32 -> i4  */
    float mf; /* default 1.0    -- maxpool i4 -> i4 */
} rocke_gfx1151_deep_fused_conv_pool_spec_t;

/* make_deep_fused_conv_pool_spec(**kwargs) -> Gfx1151DeepFusedConvPoolSpec.
 *
 * Mirrors the Python factory: builds the ConvProblem (sH=sW=1, pH=pW=1, dH=dW=1)
 * + FusedConvPoolProblem(conv, conv1_k=k1), auto-derives
 *   tile_m = (pool_tile_h*pool_stride_h) * (pool_tile_w*pool_stride_w),
 * and stamps every lever / multiplier default. Pass the Python defaults for any
 * argument the caller does not override (kept explicit to mirror the keyword-only
 * Python surface 1:1). Sample encoder_0 configs (from the port map):
 *   (n=1,h=64, w=128,c=8,k0=16,k1=16,r=3,s=3,pool_tile_h=4,pool_tile_w=8)
 *   (n=1,h=80, w=80, c=8,k0=16,k1=24,r=3,s=3,pool_tile_h=4,pool_tile_w=8)
 *   (n=1,h=56, w=112,c=8,k0=16,k1=16,r=3,s=3,pool_tile_h=2,pool_tile_w=4)
 *   (n=1,h=112,w=112,c=8,k0=16,k1=16,r=3,s=3,pool_tile_h=4,pool_tile_w=8)
 *   (n=1,h=56, w=56, c=8,k0=24,k1=16,r=3,s=3,pool_tile_h=4,pool_tile_w=8)
 *   (n=1,h=112,w=224,c=8,k0=16,k1=32,r=3,s=3,pool_tile_h=4,pool_tile_w=8). */
rocke_gfx1151_deep_fused_conv_pool_spec_t
    rocke_gfx1151_deep_fused_conv_pool_make_spec(int n,
                                                 int h,
                                                 int w,
                                                 int c,
                                                 int k0,
                                                 int k1,
                                                 int r,
                                                 int s,
                                                 int pool_tile_h,
                                                 int pool_tile_w,
                                                 int tile_n,
                                                 int warp_m,
                                                 int warp_n,
                                                 bool vectorize_conv0_a,
                                                 bool vectorize_maxpool,
                                                 bool early_w1,
                                                 bool direct_conv0,
                                                 bool w_fast,
                                                 int waves_per_eu,
                                                 const char* sched_policy,
                                                 bool mask_maxpool,
                                                 bool specialized_rne,
                                                 bool interior_fastpath,
                                                 bool static_direct_kmap,
                                                 bool packed_c0_handoff,
                                                 bool repack_c0,
                                                 bool fused_c0a1,
                                                 bool butterfly_conv01,
                                                 bool native_int,
                                                 bool batch_loads,
                                                 bool pk_maxpool,
                                                 bool conv1_prefetch_k,
                                                 bool conv1_sched_fuse,
                                                 bool conv1_int8,
                                                 bool persistent,
                                                 int persistent_ctas,
                                                 float m0,
                                                 float m0b,
                                                 float m1,
                                                 float mf);

/* Default-constructed spec carrying every Python default (the caller fills
 * `problem` + derives tile_m, or uses make_spec). */
rocke_gfx1151_deep_fused_conv_pool_spec_t rocke_gfx1151_deep_fused_conv_pool_spec_default(void);

/* ------------------------------------------------------------------ *
 * Derived-quantity accessors (Python @property mirrors).
 * ------------------------------------------------------------------ */

int rocke_gfx1151_dfcp_warp_tile_m(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec); /* _WMMA */
int rocke_gfx1151_dfcp_warp_tile_n(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec); /* _WMMA */
int rocke_gfx1151_dfcp_warp_tile_k(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec); /* _WMMA */
/* block_size = warp_m * warp_n * _WAVE. */
int rocke_gfx1151_dfcp_block_size(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);
/* kpad = ceil(conv.K_gemm / _WMMA) * _WMMA. */
int rocke_gfx1151_dfcp_kpad(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);
/* conv_tile_h = pool_tile_h * pool_stride_h; conv_tile_w analogous. */
int rocke_gfx1151_dfcp_conv_tile_h(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);
int rocke_gfx1151_dfcp_conv_tile_w(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);
/* foot_h = (conv_tile_h-1)*sH + (R-1)*dH + 1; foot_w analogous. */
int rocke_gfx1151_dfcp_foot_h(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);
int rocke_gfx1151_dfcp_foot_w(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec);

/* kernel_name(): composes name + problem.short() + tile/pool/warp tags +
 * "wmma16x16x16" + (directa|im2col) + every non-default lever tag +
 * "i8i4_realquant", joined via kernel_name_join. Writes a NUL-terminated string
 * to out[out_cap]; truncation is reported via ROCKE_ERR_*. */
rocke_status_t rocke_gfx1151_deep_fused_conv_pool_kernel_name(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec(spec, arch) -> (bool, reason)
 * ------------------------------------------------------------------ *
 *
 * Faithful port of the Python gate (lines 488-629): arch in {gfx1151,
 * gfx11-generic}; WMMA 16x16x16 f16 atom present; wave32; 2x2/s2 maxpool only;
 * N==1; positive pool tiles; tile_m == conv tile area; pool dims divisible by
 * pool tiles; K0/K1 <= tile_n; tile_m % (warp_m*16) == 0; tile_n % (warp_n*16)
 * == 0; K0 % 16 == 0; (im2col only) tile_m*kpad % block_size == 0; known
 * sched_policy; butterfly_conv01 rejected; the native_int / static_direct_kmap /
 * packed_c0_handoff / repack_c0 / fused_c0a1 / conv1_int8 / persistent lever
 * preconditions. `arch` NULL => "gfx1151". On failure writes the Python-matching
 * reason into reason[reason_cap] (when non-NULL). */
bool rocke_gfx1151_deep_fused_conv_pool_is_valid_spec(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* ------------------------------------------------------------------ *
 * deep_fused_conv_pool_grid(spec) -> (gx, gy, gz)
 * ------------------------------------------------------------------ *
 *
 * Python lines 632-645: persistent => (persistent_ctas, 1, 1);
 * w_fast => (1, w_tiles, h_tiles); else (1, h_tiles, w_tiles), where
 * h_tiles = pool_ho // pool_tile_h, w_tiles = pool_wo // pool_tile_w. */
rocke_status_t
    rocke_gfx1151_deep_fused_conv_pool_grid(const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
                                            int out[3]);

/* ------------------------------------------------------------------ *
 * Public build entry -- build_deep_fused_conv_pool(spec, arch="gfx1151")
 * ------------------------------------------------------------------ *
 *
 * Builds the gfx1151 genuine-int8/int4 fused conv0 -> conv1 -> maxpool kernel.
 * Mirrors Python build_deep_fused_conv_pool (lines 2684-3085):
 *   1. is_valid_spec gate (raises ValueError -> sticky ROCKE_ERR_VALUE here);
 *   2. resolve op = WMMA 16x16x16 f16 atom; op0 = iu8 / op1 = iu4 when native_int
 *      else both == op;
 *   3. declare the four params X(I8*), W0(I8*), Y(I32*), W1(I8*);
 *   4. build the wave32 WMMA WarpGrid (from_atom + bind, block_m_axis="y",
 *      block_n_axis="x");
 *   5. stamp the waves_per_eu launch bound + the SchedulePolicy;
 *   6. allocate the LDS tiles (a0/inp, w0, c0, c0_packed, w1, c1) sized per the
 *      native_int / direct_conv0 / packed_c0 / repack / conv1_int8 / fused_c0a1
 *      levers;
 *   7. drive the staged phases -- either the PERSISTENT grid-stride loop
 *      (native_int + direct_conv0 + fused_c0a1: weights staged once, X/Y stream
 *      per tile) or the SINGLE-TILE path through the conv0 (direct/im2col x
 *      native/fp16) -> requant scatter / register handoff -> conv1
 *      (iu4 / iu8 / fp16) -> requant scatter -> maxpool (native finalpack /
 *      fp16 finalquant) pipeline.
 *
 * `arch` NULL => "gfx1151". The KernelDef is returned (== b->kernel) on success,
 * or NULL with b's sticky error set. The caller owns `b` and frees it with
 * rocke_ir_builder_free(); the builder is initialised with the gfx1151 kernel name
 * by the convenience entry below. */
rocke_kernel_def_t* rocke_build_gfx1151_deep_fused_conv_pool(
    rocke_ir_builder_t* b, const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, const char* arch);

/* Convenience: init `b` with the gfx1151 kernel name (kernel_name(spec)), then
 * build. The caller owns `b` and frees it with rocke_ir_builder_free(). Returns the
 * kernel or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_gfx1151_deep_fused_conv_pool_new(
    rocke_ir_builder_t* b, const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec, const char* arch);

/* ------------------------------------------------------------------ *
 * Convenience: build -> lower to LLVM .ll text
 * ------------------------------------------------------------------ *
 *
 * Init a builder with the gfx1151 kernel name, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx1151". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if err !=
 * NULL, capacity err_cap) a diagnostic is written. Internally owns and frees its
 * IRBuilder. */
rocke_status_t rocke_gfx1151_deep_fused_conv_pool_lower_to_llvm(
    const rocke_gfx1151_deep_fused_conv_pool_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_DEEP_FUSED_CONV_POOL_H */
