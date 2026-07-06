/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_implicit_gemm.h -- C99 port of the implicit-GEMM
 * convolution kernel instance builder
 * rocke/instances/common/conv_implicit_gemm.py (NHWC x KYXC -> NHWK).
 *
 *   Python (conv_implicit_gemm.py)        C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   @dataclass ConvProblem                rocke_conv_problem_t
 *                                           (already ported: helper_rocke.
 *                                            instances.common.conv_implicit_gemm.h)
 *   @dataclass ConvAccumulatorEpilogue    rocke_conv_acc_epilogue_t
 *                                           (value type defined here; the deep-fused
 *                                            peer carries a narrower slice)
 *   @dataclass ImplicitGemmConvSpec       rocke_implicit_gemm_conv_spec_t
 *   spec.* @property / methods            rocke_implicit_gemm_conv_spec_*(...)
 *   is_valid_spec(spec, arch)             rocke_implicit_gemm_conv_is_valid_spec(...)
 *   make_a_descriptor / _b_ / _d_         rocke_conv_make_{a,b,d}_descriptor(...)
 *   build_implicit_gemm_conv(spec, arch)  rocke_build_implicit_gemm_conv(...)
 *   (+ convenience: build -> lower .ll)   rocke_conv_implicit_gemm_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python ImplicitGemmConvSpec is a frozen
 * dataclass with defaults plus a stack of derived @property values. In C the
 * caller fills a rocke_implicit_gemm_conv_spec_t (problem + tile geometry are the
 * only required fields); rocke_implicit_gemm_conv_spec_default() returns a struct
 * with every field set to the Python dataclass default. The derived @property
 * values (block_size, k_atoms_per_tile_k, mfmas_per_warp_m/_n, atom) are NOT
 * stored; they are recomputed by the accessor helpers below (matching the Python
 * properties, including the divisibility ValueError -> -1).
 *
 * ConvProblem is reused verbatim from the already-ported value-type helper
 * (helper_rocke.instances.common.conv_implicit_gemm.h); this header includes it
 * and embeds it by value in the spec.
 *
 * The hand-authored conv body in build_implicit_gemm_conv is a ~750-line nested
 * closure stack. Its private shared state + per-phase function contract lives in
 * the sibling PRIVATE header rocke/instance_conv_implicit_gemm_internal.h; public
 * callers only ever touch this header.
 *
 * The Python build_implicit_gemm_conv carries six optional override callbacks
 * (extra_params, m_index_fn, a_mhw_index_fn, input_cache_setup, a_load_override,
 * a_operand_override, epilogue_override) used by the deep-fusion prototype. They
 * are exposed via the rocke_conv_build_overrides_t struct: pass NULL to get the
 * stock conv body (Python's all-None default).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_H
#define ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * ConvAccumulatorEpilogue   (Python lines 142-181)
 * ============================================================ *
 *
 * @dataclass(frozen=True) ConvAccumulatorEpilogue:
 *   bias: float = 0.0
 *   scale: float = 1.0
 *   relu: bool = False
 *   clamp_min: Optional[float] = None
 *   clamp_max: Optional[float] = None
 *
 * Static fp32 accumulator transform applied before the conv store. Optionals are
 * encoded with has_clamp_min / has_clamp_max (the Python None). This is the FULL
 * port (with is_identity + tag); the deep-fused peer carries only the predicate
 * slice. */
typedef struct rocke_conv_acc_epilogue
{
    double bias; /* default 0.0   */
    double scale; /* default 1.0   */
    bool relu; /* default false */
    bool has_clamp_min; /* false => None */
    double clamp_min;
    bool has_clamp_max; /* false => None */
    double clamp_max;
} rocke_conv_acc_epilogue_t;

/* Default-constructed ConvAccumulatorEpilogue (identity). */
rocke_conv_acc_epilogue_t rocke_conv_acc_epilogue_default(void);

/* ConvAccumulatorEpilogue.is_identity(): bias==0 && scale==1 && !relu &&
 * clamp_min is None && clamp_max is None. */
bool rocke_conv_acc_epilogue_is_identity(const rocke_conv_acc_epilogue_t* epi);

/* ConvAccumulatorEpilogue.tag() -> "" when identity, else "epi_<pieces>".
 * Writes the NUL-terminated string into out (capacity out_cap). Returns ROCKE_OK,
 * or ROCKE_ERR_VALUE on NULL args / too-small buffer. */
rocke_status_t
    rocke_conv_acc_epilogue_tag(const rocke_conv_acc_epilogue_t* epi, char* out, size_t out_cap);

/* ============================================================ *
 * ImplicitGemmConvSpec   (Python lines 183-376)
 * ============================================================ *
 *
 * One concrete implicit-GEMM convolution kernel configuration. Field order
 * follows the Python dataclass declaration order. pipeline/epilogue are the
 * Python strings compared by strcmp:
 *   pipeline : "mem" | "compv3" | "compv4"
 *   epilogue : "default" | "cshuffle"
 *
 * Optionals (Python Optional[int] / Optional[LdsLayout]) are encoded with
 * companion has_* flags. lds_layout is an opaque override pointer (the LdsLayout
 * C port is a peer); NULL == Python None, which selects effective_lds_layout's
 * derivation. */
typedef struct rocke_implicit_gemm_conv_spec
{
    rocke_conv_problem_t problem;
    const char* name; /* default "conv_igemm" */

    int tile_m; /* default 64 */
    int tile_n; /* default 64 */
    int tile_k; /* default 64 */

    int warp_m; /* default 2 */
    int warp_n; /* default 2 */

    int warp_tile_m; /* default 32 */
    int warp_tile_n; /* default 32 */
    int warp_tile_k; /* default 16 */

    int wave_size; /* default 64 */

    const char* pipeline; /* default "mem"     */
    const char* epilogue; /* default "default" */
    bool async_dma; /* default false */
    bool unroll_k; /* default false */

    bool has_lds_k_pad; /* false => Python None */
    int lds_k_pad;
    /* Optional LdsLayout override (LdsLayout C port is a peer). NULL => Python
     * None (effective_lds_layout derives the policy). */
    void* lds_layout;

    bool chiplet_swizzle; /* default false */
    int chiplet_wgm; /* default 8  */
    int chiplet_num_xcds; /* default 8  */
    int chiplet_chunk_size; /* default 64 */

    bool has_waves_per_eu; /* false => Python None */
    int waves_per_eu;

    bool k0_k1_split; /* default false */
    int groups; /* default 1     */

    /* #8624 vector-sizes-as-args: per-operand load/store vector widths
     * (elements). has_* false => Python None => auto-select (choose_load_vec /
     * CShuffleEpilogue.from_grid default). When set they override the per-A/B
     * load width and the cshuffle C store width. */
    bool has_vector_size_a;
    int vector_size_a;
    bool has_vector_size_b;
    int vector_size_b;
    bool has_vector_size_c;
    int vector_size_c;

    /* #8624 ConvDataSpec: element / accumulator dtypes. Defaults all "fp16"
     * (acc "fp32"). dtype_a drives choose_load_vec's elem_bytes; A/B/D drive the
     * IR param types and the descriptor naive dtype (descriptor naive pins f16 in
     * the C port, so only the load-vec elem_bytes is observable for non-fp16). */
    const char* dtype_a; /* default "fp16" */
    const char* dtype_b; /* default "fp16" */
    const char* dtype_d; /* default "fp16" */
    const char* dtype_acc; /* default "fp32" */

    rocke_conv_acc_epilogue_t acc_epilogue; /* default identity */
} rocke_implicit_gemm_conv_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set `problem` and may override the tile geometry. */
rocke_implicit_gemm_conv_spec_t rocke_implicit_gemm_conv_spec_default(void);

/* ---- ImplicitGemmConvSpec @property analogues (pure int arithmetic) ---- */

/* spec.block_size: warp_m * warp_n * wave_size. */
int rocke_implicit_gemm_conv_spec_block_size(const rocke_implicit_gemm_conv_spec_t* s);

/* spec.k_atoms_per_tile_k: tile_k / warp_tile_k. */
int rocke_implicit_gemm_conv_spec_k_atoms_per_tile_k(const rocke_implicit_gemm_conv_spec_t* s);

/* spec.mfmas_per_warp_m: tile_m / (warp_m * warp_tile_m). */
int rocke_implicit_gemm_conv_spec_mfmas_per_warp_m(const rocke_implicit_gemm_conv_spec_t* s);

/* spec.mfmas_per_warp_n: tile_n / (warp_n * warp_tile_n). */
int rocke_implicit_gemm_conv_spec_mfmas_per_warp_n(const rocke_implicit_gemm_conv_spec_t* s);

/* spec.kernel_name() -> NUL-terminated into out (capacity out_cap). Returns
 * ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_implicit_gemm_conv_spec_kernel_name(const rocke_implicit_gemm_conv_spec_t* s,
                                                         char* out,
                                                         size_t out_cap);

/* spec.validate(): the geometry / block-size / async / clamp ValueErrors. On a
 * failure returns false and (if reason!=NULL, capacity reason_cap) writes the
 * structured message. On success returns true. (Does not touch the IR.) */
bool rocke_implicit_gemm_conv_spec_validate(const rocke_implicit_gemm_conv_spec_t* s,
                                            char* reason,
                                            size_t reason_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". The
 * arch-aware predicate (MMA-atom catalog + LDS cap + WMMA narrow-subset gates).
 * On reject returns false and writes the structured reason; on accept returns
 * true and writes "ok". */
bool rocke_implicit_gemm_conv_is_valid_spec(const rocke_implicit_gemm_conv_spec_t* s,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);

/* ============================================================ *
 * Descriptor builders   (Python lines 509-629)
 * ============================================================ *
 *
 * The user-visible coordinate-transform-DAG surface. Each builds a
 * rocke_tensor_descriptor_t into the supplied builder's arena (transforms are
 * builder-allocated, so they take `b`). Returns NULL with b's sticky error set
 * on failure.
 *
 *   make_a_descriptor(p, decompose_m=True): (m, k)->NHWC offset DAG. When
 *     decompose_m is false the leading unmerge('m'->n,ho,wo) is dropped and the
 *     user-facing upper coords become (n, ho, wo, k).
 *   make_b_descriptor(p):                   (n_gemm, k_gemm)->KYXC offset DAG.
 *   make_d_descriptor(p):                   (m, k_out)->NHWK offset DAG.
 *
 * The opaque return type avoids a transforms.h include in the public surface;
 * the concrete rocke_tensor_descriptor_t is declared in transforms.h, which the
 * internal/body TUs include. */
struct rocke_tensor_descriptor; /* fwd (full decl in helper transforms header) */

struct rocke_tensor_descriptor* rocke_conv_make_a_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p,
                                                             bool decompose_m);
struct rocke_tensor_descriptor* rocke_conv_make_b_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p);
struct rocke_tensor_descriptor* rocke_conv_make_d_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_conv_problem_t* p);

/* ============================================================ *
 * Build-time override callbacks   (Python build_implicit_gemm_conv args)
 * ============================================================ *
 *
 * The Python builder accepts six optional Callable overrides used by the
 * deep-fusion prototype. Bundled here: a NULL rocke_conv_build_overrides_t*
 * argument (or a zeroed struct) reproduces Python's all-None default = the stock
 * conv body. `extra_context` / `input_cache_context` are the opaque `object`
 * return values threaded back to the later callbacks (Python returns from
 * extra_params / input_cache_setup). Each fn pointer NULL => Python None for
 * that hook.
 *
 * NOTE: the override signatures intentionally take a rocke_warp_grid_t* (the bound
 * WarpGrid the closures read) and the descriptor-derived offset Values, matching
 * the Python callable signatures. rocke_warp_grid_t is declared in the epilogues
 * helper header, included by the body TUs. */
struct rocke_warp_grid; /* fwd (full decl in epilogues helper header) */

typedef struct rocke_conv_build_overrides
{
    /* extra_params(b) -> object. Materialise extra kernel params; the returned
     * opaque pointer is threaded to epilogue_override as extra_context. */
    void* (*extra_params)(rocke_ir_builder_t* b, void* user);

    /* m_index_fn(b, row, grid) -> Value. Custom m index for the A descriptor. */
    rocke_value_t* (*m_index_fn)(rocke_ir_builder_t* b,
                                 rocke_value_t* row,
                                 struct rocke_warp_grid* grid,
                                 void* user);

    /* a_mhw_index_fn(b, row, grid) -> (n, ho, wo). Feeds the decomposed A
     * descriptor (decompose_m=False) directly. Writes the three Values through
     * the out-params; presence (!= NULL) also flips make_a_descriptor's
     * decompose_m to false in the driver. */
    void (*a_mhw_index_fn)(rocke_ir_builder_t* b,
                           rocke_value_t* row,
                           struct rocke_warp_grid* grid,
                           rocke_value_t** out_n,
                           rocke_value_t** out_ho,
                           rocke_value_t** out_wo,
                           void* user);

    /* input_cache_setup(b, spec, grid, a_rsrc) -> object. Returns the opaque
     * input-cache context threaded to a_load_override / a_operand_override. */
    void* (*input_cache_setup)(rocke_ir_builder_t* b,
                               const rocke_implicit_gemm_conv_spec_t* spec,
                               struct rocke_warp_grid* grid,
                               rocke_value_t* a_rsrc,
                               void* user);

    /* a_load_override(b, spec, k_off, A_dst, grid, input_cache_context). */
    void (*a_load_override)(rocke_ir_builder_t* b,
                            const rocke_implicit_gemm_conv_spec_t* spec,
                            rocke_value_t* k_off,
                            rocke_value_t* A_dst,
                            struct rocke_warp_grid* grid,
                            void* input_cache_context,
                            void* user);

    /* a_operand_override(b, spec, a_row, k_off, col_base, a_per_lane, grid,
     * input_cache_context) -> Value. */
    rocke_value_t* (*a_operand_override)(rocke_ir_builder_t* b,
                                         const rocke_implicit_gemm_conv_spec_t* spec,
                                         rocke_value_t* a_row,
                                         rocke_value_t* k_off,
                                         rocke_value_t* col_base,
                                         int a_per_lane,
                                         struct rocke_warp_grid* grid,
                                         void* input_cache_context,
                                         void* user);

    /* epilogue_override(b, spec, accs, grid, d_rsrc, extra_context). */
    void (*epilogue_override)(rocke_ir_builder_t* b,
                              const rocke_implicit_gemm_conv_spec_t* spec,
                              rocke_value_t* const* accs,
                              int num_accs,
                              struct rocke_warp_grid* grid,
                              rocke_value_t* d_rsrc,
                              void* extra_context,
                              void* user);

    /* Opaque user pointer threaded to every callback above. */
    void* user;
} rocke_conv_build_overrides_t;

/* ============================================================ *
 * build_implicit_gemm_conv   (Python lines 730-1378)
 * ============================================================ *
 *
 * Builds the IR into the supplied (already rocke_ir_builder_init'd) builder `b`,
 * exactly as the Python build does, and returns the kernel (b->kernel) on
 * success or NULL with b's sticky error set. `arch` NULL => "gfx950".
 * `overrides` NULL => the stock conv body (Python all-None default).
 *
 * Like the Python, this expects the builder to have been created with the spec's
 * kernel_name(). Use rocke_build_implicit_gemm_conv_new() for the init-from-spec
 * convenience. */
rocke_kernel_def_t* rocke_build_implicit_gemm_conv(rocke_ir_builder_t* b,
                                                   const rocke_implicit_gemm_conv_spec_t* spec,
                                                   const char* arch,
                                                   const rocke_conv_build_overrides_t* overrides);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns `b`
 * and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t*
    rocke_build_implicit_gemm_conv_new(rocke_ir_builder_t* b,
                                       const rocke_implicit_gemm_conv_spec_t* spec,
                                       const char* arch,
                                       const rocke_conv_build_overrides_t* overrides);

/* Convenience: given a spec, init a builder, build (stock body), and lower to
 * LLVM .ll text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err!=NULL, capacity err_cap) a diagnostic is written. Internally owns
 * and frees its IRBuilder. */
rocke_status_t rocke_conv_implicit_gemm_lower_to_llvm(const rocke_implicit_gemm_conv_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_IMPLICIT_GEMM_H */
