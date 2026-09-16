/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.deep_fused_conv_pool.h -- C99 port of the
 * family-agnostic deep-fused conv0 -> conv1 -> maxpool prototype from
 * rocke/instances/common/deep_fused_conv_pool.py.
 *
 * Dataflow (one CTA owns one pooled-output tile):
 *
 *   implicit-GEMM conv0 -> accumulator epilogue -> LDS C-shuffle
 *   -> 1x1 conv1 -> LDS C-shuffle -> maxpool -> Y
 *
 *   Python (deep_fused_conv_pool.py)            C99 (this header)
 *   -----------------------------------------   ------------------------------
 *   @dataclass FusedConvPoolProblem             rocke_fused_conv_pool_problem_t
 *     .conv1_channels / .pool_ho / .pool_wo /     rocke_fused_conv_pool_problem_*
 *     .total_out / .short()
 *   @dataclass DeepFusedConvPoolSpec            rocke_deep_fused_conv_pool_spec_t
 *     .block_size / .effective_conv1_tile_k       rocke_deep_fused_conv_pool_spec_*
 *     .kernel_name() / .conv_spec()
 *   make_deep_fused_conv_pool_spec(...)         rocke_make_deep_fused_conv_pool_spec(...)
 *   is_valid_spec(spec, arch)                   rocke_deep_fused_conv_pool_is_valid_spec(...)
 *   deep_fused_conv_pool_signature(spec)        rocke_deep_fused_conv_pool_signature(...)
 *   deep_fused_conv_pool_grid(spec)             rocke_deep_fused_conv_pool_grid(...)
 *   _stage_accumulators_to_cshuffle_lds(...)    rocke_dfcp_stage_accumulators_to_cshuffle_lds(...)
 *   _load_conv1_weights_to_lds(...)             rocke_dfcp_load_conv1_weights_to_lds(...)
 *   _can_use_specialized_conv0_a_loader(spec)   rocke_dfcp_can_use_specialized_conv0_a_loader(...)
 *   _load_conv0_a_tile_specialized(...)         rocke_dfcp_load_conv0_a_tile_specialized(...)
 *   _setup_input_footprint_cache(...)           rocke_dfcp_setup_input_footprint_cache(...)
 *   _load_conv0_a_tile_from_input_cache(...)    rocke_dfcp_load_conv0_a_tile_from_input_cache(...)
 *   _load_conv0_a_operand_from_input_cache(...) rocke_dfcp_load_conv0_a_operand_from_input_cache(...)
 *   _epilogue_is_pool_deferrable(epi)           rocke_dfcp_epilogue_is_pool_deferrable(...)
 *   _epilogue_is_relu_only(epi)                 rocke_dfcp_epilogue_is_relu_only(...)
 *   _apply_epilogue_scalar(b, epi, v)           rocke_dfcp_apply_epilogue_scalar(...)
 *   _emit_conv1_1x1(...)                        rocke_dfcp_emit_conv1_1x1(...)
 *   _emit_inline_maxpool_from_cshuffle(...)     rocke_dfcp_emit_inline_maxpool_from_cshuffle(...)
 *   _maxpool_is_intra_lane(spec, grid)          rocke_dfcp_maxpool_is_intra_lane(...)
 *   _maxpool_is_intra_lane_wmma(spec,grid,op)   rocke_dfcp_maxpool_is_intra_lane_wmma(...)
 *   _emit_wmma_maxpool_from_registers(...)      rocke_dfcp_emit_wmma_maxpool_from_registers(...)
 *   _emit_inline_maxpool_from_registers(...)    rocke_dfcp_emit_inline_maxpool_from_registers(...)
 *
 * PEER DEPENDENCIES NOT YET PORTED. The builder-emit helpers reach into peer
 * modules whose C ports do not exist yet:
 *   - WarpGrid                  (rocke/helpers/geometry.py)
 *   - MmaOp / op.{a,b,c}_layout (rocke/core/arch.py)
 *   - CoalescedTileLoader       (rocke/helpers/loads.py)
 *   - LoadStoreTraits / make_static_distributed_tensor / store_tile_cshuffle
 *                               (rocke/helpers/distribution.py)
 *   - LdsLayout.cshuffle        (rocke/helpers/layouts.py)
 *   - ConvAccumulatorEpilogue / ImplicitGemmConvSpec / _apply_accumulator_epilogue
 *                               (conv_implicit_gemm.py -- only ConvProblem ported)
 * To keep this header self-contained and compilable today, the small slices of
 * those interfaces this port needs are forward-declared below as opaque handles
 * (rocke_warp_grid_t, rocke_mma_op_t) plus a value-type ConvAccumulatorEpilogue
 * (rocke_conv_acc_epilogue_t) that the pure predicate/scalar helpers fully
 * implement. The emit helpers that need the (not-yet-ported) peer DAG carry a
 * bounded body + a TODO(port) marker routed through the sticky-error builder;
 * verify+fix resolves them when the peer ports land.
 *
 * Error model mirrors the rest of the C port: pure value producers return
 * by-value / status codes; builder helpers route errors through the
 * sticky-error IRBuilder (rocke_b_*).
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_DEEP_FUSED_CONV_POOL_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_DEEP_FUSED_CONV_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (signature storage) */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t (signature) */
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h" /* rocke_status_t, rocke_value_t, rocke_ir_builder_t */
/* INTEGRATION: ConvAccumulatorEpilogue + ImplicitGemmConvSpec are owned by the
 * conv_implicit_gemm peer. This header previously redefined a divergent
 * ConvAccumulatorEpilogue slice (different field order) under the SAME tag/
 * typedef, which is an ABI-incompatible duplicate once both headers land in one
 * TU. Pull the canonical public conv header so the type (and ImplicitGemmConvSpec
 * used by spec.conv_spec()) has a single definition shared with the conv peer. */
#include "rocke/instance_conv_implicit_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * Opaque peer handles (forward-declared; real ports are peers)
 * ------------------------------------------------------------------ *
 *
 * WarpGrid (geometry.py) and MmaOp (core/arch.py) are passed through the
 * builder-emit helpers verbatim. The Python bodies read these members:
 *   grid.{tid,lane,warp_m_idx}, grid.{warp_m_off,warp_n_off}(b),
 *   grid.{tile_m,tile_n,warp_tile_m,warp_tile_n,wave_size,warp_m,warp_n},
 *   grid.{mfmas_per_warp_m,mfmas_per_warp_n}
 *   op.{m,n}, op.{a,b,c}_frag_len, op.{a,b,c}_layout().coord(b, lane, i)
 * The C accessor surface for these lands with the geometry/arch ports; here
 * they are opaque pointers so this header type-checks today. */
typedef struct rocke_warp_grid rocke_warp_grid_t;
typedef struct rocke_mma_op rocke_mma_op_t;

/* Forward to the (peer) ImplicitGemmConvSpec port. Only ConvProblem is ported
 * in conv_implicit_gemm.h; the full spec is a peer. */
typedef struct rocke_implicit_gemm_conv_spec rocke_implicit_gemm_conv_spec_t;

/* ------------------------------------------------------------------ *
 * ConvAccumulatorEpilogue (value type)
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True) ConvAccumulatorEpilogue from conv_implicit_gemm.py:
 *   relu: bool = False
 *   bias: float = 0.0
 *   scale: float = 1.0
 *   clamp_min: Optional[float] = None
 *   clamp_max: Optional[float] = None
 *
 * Optionals are encoded with has_clamp_min / has_clamp_max flags (the Python
 * None). This is the slice the deferred-epilogue predicate + scalar-apply
 * helpers need.
 *
 * INTEGRATION: rocke_conv_acc_epilogue_t and rocke_conv_acc_epilogue_is_identity are
 * the SHARED canonical type/predicate owned by conv_implicit_gemm.h (included
 * above). They are no longer redeclared here -- the prior local copy used a
 * divergent field order under the same tag, producing an ABI-incompatible
 * duplicate. Only the deep-fusion-specific relu() ctor stays. */

/* ConvAccumulatorEpilogue(relu=True) -- the deep-fusion default for both the
 * conv0 acc epilogue and the conv1 epilogue. */
rocke_conv_acc_epilogue_t rocke_conv_acc_epilogue_relu(void);

/* ------------------------------------------------------------------ *
 * FusedConvPoolProblem
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True):
 *   conv: ConvProblem
 *   conv1_k: int = 0
 *   pool_y: int = 2; pool_x: int = 2
 *   pool_stride_h: int = 2; pool_stride_w: int = 2
 */
typedef struct rocke_fused_conv_pool_problem
{
    rocke_conv_problem_t conv;
    int conv1_k; /* default 0 */
    int pool_y; /* default 2 */
    int pool_x; /* default 2 */
    int pool_stride_h; /* default 2 */
    int pool_stride_w; /* default 2 */
} rocke_fused_conv_pool_problem_t;

/* FusedConvPoolProblem(conv, conv1_k=0, pool_y=2, pool_x=2,
 *                      pool_stride_h=2, pool_stride_w=2). */
rocke_fused_conv_pool_problem_t rocke_fused_conv_pool_problem_make(rocke_conv_problem_t conv,
                                                                   int conv1_k,
                                                                   int pool_y,
                                                                   int pool_x,
                                                                   int pool_stride_h,
                                                                   int pool_stride_w);

/* conv1_channels property:  conv.K if conv1_k <= 0 else conv1_k */
int rocke_fused_conv_pool_problem_conv1_channels(const rocke_fused_conv_pool_problem_t* p);

/* pool_ho property:  (conv.Ho - pool_y) // pool_stride_h + 1 */
int rocke_fused_conv_pool_problem_pool_ho(const rocke_fused_conv_pool_problem_t* p);

/* pool_wo property:  (conv.Wo - pool_x) // pool_stride_w + 1 */
int rocke_fused_conv_pool_problem_pool_wo(const rocke_fused_conv_pool_problem_t* p);

/* total_out property:  conv.N * pool_ho * pool_wo * conv1_channels.
 * 64-bit return to avoid the int32 overflow Python's big-int never hits. */
long long rocke_fused_conv_pool_problem_total_out(const rocke_fused_conv_pool_problem_t* p);

/* short() ->
 *   f"{conv.short()}_K1{conv1_channels}_pool{pool_y}x{pool_x}_s{psh}x{psw}"
 * NUL-terminated into out (capacity out_cap); ROCKE_OK / ROCKE_ERR_VALUE. */
rocke_status_t rocke_fused_conv_pool_problem_short(const rocke_fused_conv_pool_problem_t* p,
                                                   char* out,
                                                   size_t out_cap,
                                                   size_t* out_len);

/* ------------------------------------------------------------------ *
 * DeepFusedConvPoolSpec
 * ------------------------------------------------------------------ */
typedef struct rocke_deep_fused_conv_pool_spec
{
    rocke_fused_conv_pool_problem_t problem;
    const char* name; /* default "rocke_deep_fused_conv_pool" */
    int tile_m; /* default 128 */
    int tile_n; /* default 32  */
    int tile_k; /* default 16  */
    int conv1_tile_k; /* default 0   */
    int pool_tile_h; /* default 4   */
    int pool_tile_w; /* default 8   */
    int warp_m; /* default 2   */
    int warp_n; /* default 1   */
    int warp_tile_m; /* default 32  */
    int warp_tile_n; /* default 32  */
    int warp_tile_k; /* default 16  */
    int wave_size; /* default 64  */
    const char* pipeline; /* default "mem" */
    bool async_dma; /* default false */
    bool unroll_k; /* default false */
    rocke_conv_acc_epilogue_t acc_epilogue; /* default relu=True */
    rocke_conv_acc_epilogue_t conv1_epilogue; /* default relu=True */
    bool cache_input_footprint; /* default false */
    bool direct_conv0_from_input_cache; /* default false */
} rocke_deep_fused_conv_pool_spec_t;

/* Default-constructed spec (the dataclass field defaults). The caller fills
 * `problem`. */
rocke_deep_fused_conv_pool_spec_t rocke_deep_fused_conv_pool_spec_default(void);

/* block_size property:  warp_m * warp_n * wave_size */
int rocke_deep_fused_conv_pool_spec_block_size(const rocke_deep_fused_conv_pool_spec_t* spec);

/* effective_conv1_tile_k property:  tile_k if conv1_tile_k <= 0 else conv1_tile_k */
int rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(
    const rocke_deep_fused_conv_pool_spec_t* spec);

/* kernel_name() -> NUL-terminated into out (capacity out_cap).
 * ROCKE_OK / ROCKE_ERR_VALUE. */
rocke_status_t rocke_deep_fused_conv_pool_spec_kernel_name(
    const rocke_deep_fused_conv_pool_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * make_deep_fused_conv_pool_spec
 * ------------------------------------------------------------------ *
 *
 * Keyword-only Python factory. The C signature lists every field; `tile_m` is
 * auto-derived as (pool_tile_h*pool_stride_h)*(pool_tile_w*pool_stride_w) just
 * as the Python does (pool_stride_h/w are FusedConvPoolProblem defaults = 2).
 * The conv is built with N=n, stride=1, pad=1, dilation=1 (the Python literal
 * ConvProblem). */
rocke_deep_fused_conv_pool_spec_t
    rocke_make_deep_fused_conv_pool_spec(int n,
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
                                         int tile_k,
                                         int conv1_tile_k,
                                         int warp_m,
                                         int warp_n,
                                         int warp_tile_m,
                                         int warp_tile_n,
                                         int warp_tile_k,
                                         int wave_size,
                                         const char* name,
                                         const char* pipeline,
                                         bool unroll_k,
                                         bool async_dma,
                                         bool cache_input_footprint,
                                         bool direct_conv0_from_input_cache);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On reject
 * `reason` (if non-NULL, capacity reason_cap) gets the message and returns
 * false; on accept returns true and writes "ok".
 *
 * NOTE: the leading underlying-conv gate (is_valid_conv_spec) lives in the
 * not-yet-ported conv spec peer; until it lands this routine validates the
 * deep-fusion-specific constraints (pool geometry, tiling, channel fit) and
 * documents the conv-gate dependency. */
bool rocke_deep_fused_conv_pool_is_valid_spec(const rocke_deep_fused_conv_pool_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap);

/* deep_fused_conv_pool_signature(spec): manifest signature
 *   A:f16, B:f16, Y:f16, W1:f16,
 *   W1_bytes:i32, A_bytes:i32, B_bytes:i32, Y_bytes:i32
 * Entries are arena-owned. On ROCKE_OK *out_items / *out_count hold the array. */
rocke_status_t rocke_deep_fused_conv_pool_signature(rocke_arena_t* arena,
                                                    const rocke_deep_fused_conv_pool_spec_t* spec,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count);

/* deep_fused_conv_pool_grid(spec) ->
 *   (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w)
 * out[0..2] receive (x, y, z). ROCKE_ERR_VALUE on NULL args. */
rocke_status_t rocke_deep_fused_conv_pool_grid(const rocke_deep_fused_conv_pool_spec_t* spec,
                                               int out[3]);

/* ------------------------------------------------------------------ *
 * Pure epilogue predicates / scalar apply
 * ------------------------------------------------------------------ */

/* _epilogue_is_pool_deferrable(epi):  epi.scale >= 0.0 */
bool rocke_dfcp_epilogue_is_pool_deferrable(const rocke_conv_acc_epilogue_t* epi);

/* _epilogue_is_relu_only(epi):  bias==0 and scale==1 and relu and
 *                               clamp_min is None and clamp_max is None */
bool rocke_dfcp_epilogue_is_relu_only(const rocke_conv_acc_epilogue_t* epi);

/* _apply_epilogue_scalar(b, epi, v): apply the static fp32 epilogue to one
 * scalar SSA value, mirroring _apply_accumulator_epilogue's per-lane transform.
 * Returns the (possibly unchanged) value; routes errors through b. */
rocke_value_t* rocke_dfcp_apply_epilogue_scalar(rocke_ir_builder_t* b,
                                                const rocke_conv_acc_epilogue_t* epi,
                                                rocke_value_t* v);

/* ------------------------------------------------------------------ *
 * conv0 A-loader specialization gate (pure)
 * ------------------------------------------------------------------ */

/* _can_use_specialized_conv0_a_loader(spec): whether the target-shape conv0 A
 * load can bypass TensorDescriptor math (wave32, N=1, C=8, R=S=3, unit
 * stride/pad/dilation, no input cache). */
bool rocke_dfcp_can_use_specialized_conv0_a_loader(const rocke_deep_fused_conv_pool_spec_t* spec);

/* ------------------------------------------------------------------ *
 * maxpool register-residency gates (pure)
 * ------------------------------------------------------------------ *
 *
 * These read WarpGrid members; until the geometry port lands they accept the
 * opaque grid handle and a bounded TODO(port) body. */

/* _maxpool_is_intra_lane(spec, grid): MFMA-32x32 register-resident fast path. */
bool rocke_dfcp_maxpool_is_intra_lane(const rocke_deep_fused_conv_pool_spec_t* spec,
                                      const rocke_warp_grid_t* grid);

/* _maxpool_is_intra_lane_wmma(spec, grid, op): WMMA analogue (wave32 16x16). */
bool rocke_dfcp_maxpool_is_intra_lane_wmma(const rocke_deep_fused_conv_pool_spec_t* spec,
                                           const rocke_warp_grid_t* grid,
                                           const rocke_mma_op_t* op);

/* ------------------------------------------------------------------ *
 * Builder-emit helpers
 * ------------------------------------------------------------------ *
 *
 * These mirror the Python bodies that drive the IR builder. Where the body only
 * needs ported primitives (rocke_b_*) it is byte-faithful; where it needs a
 * not-yet-ported peer (WarpGrid/MmaOp/CoalescedTileLoader/distribution) the
 * body is bounded with a TODO(port) marker routed through the sticky-error
 * builder. */

/* _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=True):
 * publish MMA accumulators to a row-major [tile_m, tile_n] LDS tile; returns
 * the LDS base (c_smem). */
rocke_value_t* rocke_dfcp_stage_accumulators_to_cshuffle_lds(rocke_ir_builder_t* b,
                                                             const rocke_mma_op_t* op,
                                                             rocke_value_t* const* accs,
                                                             size_t num_accs,
                                                             const rocke_warp_grid_t* grid,
                                                             bool sync);

/* _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=True): load
 * W1[K1, K0] into a padded row-major LDS tile; returns the LDS base. */
rocke_value_t* rocke_dfcp_load_conv1_weights_to_lds(rocke_ir_builder_t* b,
                                                    const rocke_deep_fused_conv_pool_spec_t* spec,
                                                    rocke_value_t* w1_rsrc,
                                                    const rocke_warp_grid_t* grid,
                                                    bool sync);

/* _load_conv0_a_tile_specialized(b, spec, conv_spec, k_off, a_dst, grid, a_rsrc). */
void rocke_dfcp_load_conv0_a_tile_specialized(rocke_ir_builder_t* b,
                                              const rocke_deep_fused_conv_pool_spec_t* spec,
                                              const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                              rocke_value_t* k_off,
                                              rocke_value_t* a_dst,
                                              const rocke_warp_grid_t* grid,
                                              rocke_value_t* a_rsrc);

/* _setup_input_footprint_cache(b, spec, a_rsrc, grid): load the unique conv0
 * input footprint for this pooled-output tile; returns the LDS base. */
rocke_value_t* rocke_dfcp_setup_input_footprint_cache(rocke_ir_builder_t* b,
                                                      const rocke_deep_fused_conv_pool_spec_t* spec,
                                                      rocke_value_t* a_rsrc,
                                                      const rocke_warp_grid_t* grid);

/* _load_conv0_a_tile_from_input_cache(b, spec, conv_spec, k_off, a_dst, grid,
 *                                     input_smem). */
void rocke_dfcp_load_conv0_a_tile_from_input_cache(rocke_ir_builder_t* b,
                                                   const rocke_deep_fused_conv_pool_spec_t* spec,
                                                   const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                                   rocke_value_t* k_off,
                                                   rocke_value_t* a_dst,
                                                   const rocke_warp_grid_t* grid,
                                                   rocke_value_t* input_smem);

/* _load_conv0_a_operand_from_input_cache(b, spec, row, k_off, col_base,
 *                                        frag_len, input_smem): read one MFMA A
 * operand fragment from the cached footprint; returns the packed fragment. */
rocke_value_t*
    rocke_dfcp_load_conv0_a_operand_from_input_cache(rocke_ir_builder_t* b,
                                                     const rocke_deep_fused_conv_pool_spec_t* spec,
                                                     rocke_value_t* row,
                                                     rocke_value_t* k_off,
                                                     rocke_value_t* col_base,
                                                     int frag_len,
                                                     rocke_value_t* input_smem);

/* _emit_conv1_1x1(b, spec, conv_spec, op, c0_smem, w1_smem, grid,
 *                 defer_epilogue=False): compute conv1 as a 1x1 GEMM over the
 * staged conv0 activations. Returns the accumulators into out_accs[0..*out_n).
 * out_accs capacity is out_cap; ROCKE_ERR_VALUE if too small. */
rocke_status_t rocke_dfcp_emit_conv1_1x1(rocke_ir_builder_t* b,
                                         const rocke_deep_fused_conv_pool_spec_t* spec,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                         const rocke_mma_op_t* op,
                                         rocke_value_t* c0_smem,
                                         rocke_value_t* w1_smem,
                                         const rocke_warp_grid_t* grid,
                                         bool defer_epilogue,
                                         rocke_value_t** out_accs,
                                         size_t out_cap,
                                         size_t* out_n);

/* _emit_inline_maxpool_from_cshuffle(b, spec, c_smem, y_rsrc, grid, epilogue):
 * reduce the staged conv tile into final pooled NHWK output. `epilogue` NULL =>
 * no deferred epilogue. */
void rocke_dfcp_emit_inline_maxpool_from_cshuffle(rocke_ir_builder_t* b,
                                                  const rocke_deep_fused_conv_pool_spec_t* spec,
                                                  rocke_value_t* c_smem,
                                                  rocke_value_t* y_rsrc,
                                                  const rocke_warp_grid_t* grid,
                                                  const rocke_conv_acc_epilogue_t* epilogue);

/* _emit_wmma_maxpool_from_registers(b, spec, conv1_accs, y_rsrc, grid, op,
 *                                   epilogue): RDNA4 register-resident maxpool. */
void rocke_dfcp_emit_wmma_maxpool_from_registers(rocke_ir_builder_t* b,
                                                 const rocke_deep_fused_conv_pool_spec_t* spec,
                                                 rocke_value_t* const* conv1_accs,
                                                 size_t num_accs,
                                                 rocke_value_t* y_rsrc,
                                                 const rocke_warp_grid_t* grid,
                                                 const rocke_mma_op_t* op,
                                                 const rocke_conv_acc_epilogue_t* epilogue);

/* _emit_inline_maxpool_from_registers(b, spec, conv1_accs, y_rsrc, grid,
 *                                     epilogue): MFMA-only register-resident
 * maxpool reducing each lane's vec<16> accumulator. */
void rocke_dfcp_emit_inline_maxpool_from_registers(rocke_ir_builder_t* b,
                                                   const rocke_deep_fused_conv_pool_spec_t* spec,
                                                   rocke_value_t* const* conv1_accs,
                                                   size_t num_accs,
                                                   rocke_value_t* y_rsrc,
                                                   const rocke_warp_grid_t* grid,
                                                   const rocke_conv_acc_epilogue_t* epilogue);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_DEEP_FUSED_CONV_POOL_H */
