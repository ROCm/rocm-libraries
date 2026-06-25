/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_pooling.h -- C99 port of the 2D pooling kernel instance builder
 * ck_dsl/instances/common/pooling.py (CK Tile ``36_pooling`` 2D counterpart,
 * NHWC max/avg/sum).
 *
 *   Python (pooling.py)                   C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   @dataclass(frozen=True) PoolingProblem  ckc_pooling_problem_t
 *     .Ho / .Wo / .total_out (@property)    ckc_pooling_problem_ho/_wo/_total_out
 *     .short()                              ckc_pooling_problem_short
 *   @dataclass(frozen=True) Pooling2DSpec    ckc_pooling2d_spec_t
 *     .kernel_name()                        ckc_pooling2d_kernel_name
 *   is_valid_spec(spec, arch)              ckc_pooling2d_is_valid_spec
 *   build_pooling2d(spec, arch)            ckc_build_pooling2d
 *   pooling2d_grid(spec)                   ckc_pooling2d_grid
 *   pooling2d_signature(spec)              ckc_pooling2d_signature
 *   (+ convenience: build -> lower .ll)    ckc_pooling2d_lower_to_llvm
 *
 * SPEC AS EXPLICIT C STRUCTS. The frozen Python dataclasses become value
 * structs; ckc_pooling_problem_default() / ckc_pooling2d_spec_default() return
 * the Python dataclass defaults so the caller overrides only the fields it
 * cares about.
 *
 * The window reduction accumulates in per-thread f32 registers (no LDS, no
 * MFMA, no cross-lane butterfly); the emitted IR is wave-size agnostic and
 * arch-polymorphic, exactly like the Python. `arch` is threaded only to
 * validate block_size against the target's max_threads_per_block.
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 *
 * DEPENDENCY NOTE: the store epilogue uses make_buffer_resource /
 * make_buffer_view / make_static_tile_distribution / make_static_distributed_
 * tensor / store_tile from ck_dsl.helpers.{tensor_view,distribution}. Of these
 * only make_static_tile_distribution + the encoding constructor are ported to C
 * so far; the buffer-view / distributed-tensor / store_tile family is declared
 * here (and forward-declared in the .c) as a TODO(port) surface so the build
 * entry can emit the byte-identical call sequence. The verify+fix loop resolves
 * any diff once those helpers land.
 */
#ifndef CKC_INSTANCE_POOLING_H
#define CKC_INSTANCE_POOLING_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ PoolingProblem *
 *
 * @dataclass(frozen=True)
 * class PoolingProblem:
 *     N, H, W, C, Y, X            # required
 *     sH=1, sW=1, pH=0, pW=0, dH=1, dW=1
 *
 *   Ho = (H + 2*pH - ((Y-1)*dH + 1)) // sH + 1
 *   Wo = (W + 2*pW - ((X-1)*dW + 1)) // sW + 1
 *   total_out = N * Ho * Wo * C
 */
typedef struct ckc_pooling_problem
{
    int N;
    int H;
    int W;
    int C;

    int Y; /* window height */
    int X; /* window width  */

    int sH; /* default 1 */
    int sW; /* default 1 */
    int pH; /* default 0 (left pad, also used as right pad) */
    int pW; /* default 0 */
    int dH; /* default 1 (dilation) */
    int dW; /* default 1 */
} ckc_pooling_problem_t;

/* PoolingProblem with dataclass defaults installed (sH=sW=dH=dW=1, pH=pW=0) and
 * the six required dims zeroed. The caller fills N, H, W, C, Y, X. */
ckc_pooling_problem_t ckc_pooling_problem_default(void);

/* PoolingProblem.Ho property: (H + 2*pH - ((Y-1)*dH + 1)) // sH + 1.
 * Floor division matches Python // for the non-negative numerator. */
int ckc_pooling_problem_ho(const ckc_pooling_problem_t* p);

/* PoolingProblem.Wo property: (W + 2*pW - ((X-1)*dW + 1)) // sW + 1. */
int ckc_pooling_problem_wo(const ckc_pooling_problem_t* p);

/* PoolingProblem.total_out property: N * Ho * Wo * C. */
int ckc_pooling_problem_total_out(const ckc_pooling_problem_t* p);

/* PoolingProblem.short() ->
 *   f"N{N}H{H}W{W}C{C}_Y{Y}X{X}_s{sH}x{sW}_p{pH}x{pW}"
 * Writes the NUL-terminated string into out (capacity out_cap). Returns CKC_OK,
 * or CKC_ERR_VALUE on NULL args / a too-small buffer. */
ckc_status_t ckc_pooling_problem_short(const ckc_pooling_problem_t* p, char* out, size_t out_cap);

/* ------------------------------------------------------------- Pooling2DSpec *
 *
 * @dataclass(frozen=True)
 * class Pooling2DSpec:
 *     problem: PoolingProblem
 *     dtype: DType = "f16"            # "f16" | "bf16"
 *     op: PoolOp = "max"             # "max" | "avg" | "sum"
 *     block_size: int = 256
 *     vec: int = 1
 *     name: str = "ck_dsl_pooling2d"
 *     tile_n: int = 1                # P81 (unused by v1 build path)
 *     use_warp_xor_reduce: bool = False  # P82 (unused by v1 build path)
 */
typedef struct ckc_pooling2d_spec
{
    ckc_pooling_problem_t problem;
    const char* dtype; /* "f16" | "bf16"; default "f16"        */
    const char* op; /* "max" | "avg" | "sum"; default "max" */
    int block_size; /* default 256                          */
    int vec; /* default 1                            */
    const char* name; /* default "ck_dsl_pooling2d"           */
    int tile_n; /* default 1                            */
    bool use_warp_xor_reduce; /* default false                 */
} ckc_pooling2d_spec_t;

/* Default-constructed spec (dtype "f16", op "max", block_size 256, vec 1, name
 * "ck_dsl_pooling2d", tile_n 1, use_warp_xor_reduce false, problem ==
 * ckc_pooling_problem_default()). The caller fills problem's six required
 * dims and overrides any field. */
ckc_pooling2d_spec_t ckc_pooling2d_spec_default(void);

/* Pooling2DSpec.kernel_name():
 *   kernel_name_join(name, problem.short(), dtype, op, f"b{block_size}",
 *                    f"v{vec}")
 * Result written NUL-terminated into out (capacity out_cap). Returns CKC_OK or
 * CKC_ERR_VALUE (NULL args / buffer too small). */
ckc_status_t ckc_pooling2d_kernel_name(const ckc_pooling2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the message and
 * false is returned. On accept returns true and writes "ok". */
bool ckc_pooling2d_is_valid_spec(const ckc_pooling2d_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap);

/* build_pooling2d(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` NULL => "gfx950". The kernel name is set by the builder init; this
 * routine does NOT re-init the builder. */
ckc_kernel_def_t*
    ckc_build_pooling2d(ckc_ir_builder_t* b, const ckc_pooling2d_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_pooling2d_new(ckc_ir_builder_t* b,
                                          const ckc_pooling2d_spec_t* spec,
                                          const char* arch);

/* pooling2d_grid(spec) -> (x, y, z): one thread per vec-element output slab.
 *   total_v = total_out // max(vec, 1); grid = ceil_div_grid(total_v, block_size)
 * out[0..2] receive the grid. Returns CKC_OK or CKC_ERR_VALUE. */
ckc_status_t ckc_pooling2d_grid(const ckc_pooling2d_spec_t* spec, int out[3]);

/* pooling2d_signature(spec): the four-entry manifest
 *   ptr X, ptr Y, scalar X_bytes:i32, scalar Y_bytes:i32.
 * Writes up to out_cap entries into out[] and sets *out_count. `arena` owns the
 * copied name/type strings. Returns CKC_OK or an error status. */
struct ckc_sig_entry; /* fwd decl (ckc/helper_ck_dsl.helpers.spec.h) */
struct ckc_arena; /* fwd decl (ckc/arena.h) */
ckc_status_t ckc_pooling2d_signature(struct ckc_arena* arena,
                                     const ckc_pooling2d_spec_t* spec,
                                     struct ckc_sig_entry* out,
                                     size_t out_cap,
                                     size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns its IRBuilder. */
ckc_status_t ckc_pooling2d_lower_to_llvm(const ckc_pooling2d_spec_t* spec,
                                         const char* arch,
                                         ckc_llvm_flavor_t flavor,
                                         char** out_ll,
                                         char* err,
                                         size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_POOLING_H */
