/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_pooling.h -- C99 port of the 2D pooling kernel instance builder
 * rocke/instances/common/pooling.py (CK Tile ``36_pooling`` 2D counterpart,
 * NHWC max/avg/sum).
 *
 *   Python (pooling.py)                   C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   @dataclass(frozen=True) PoolingProblem  rocke_pooling_problem_t
 *     .Ho / .Wo / .total_out (@property)    rocke_pooling_problem_ho/_wo/_total_out
 *     .short()                              rocke_pooling_problem_short
 *   @dataclass(frozen=True) Pooling2DSpec    rocke_pooling2d_spec_t
 *     .kernel_name()                        rocke_pooling2d_kernel_name
 *   is_valid_spec(spec, arch)              rocke_pooling2d_is_valid_spec
 *   build_pooling2d(spec, arch)            rocke_build_pooling2d
 *   pooling2d_grid(spec)                   rocke_pooling2d_grid
 *   pooling2d_signature(spec)              rocke_pooling2d_signature
 *   (+ convenience: build -> lower .ll)    rocke_pooling2d_lower_to_llvm
 *
 * SPEC AS EXPLICIT C STRUCTS. The frozen Python dataclasses become value
 * structs; rocke_pooling_problem_default() / rocke_pooling2d_spec_default() return
 * the Python dataclass defaults so the caller overrides only the fields it
 * cares about.
 *
 * The window reduction accumulates in per-thread f32 registers (no LDS, no
 * MFMA, no cross-lane butterfly); the emitted IR is wave-size agnostic and
 * arch-polymorphic, exactly like the Python. `arch` is threaded only to
 * validate block_size against the target's max_threads_per_block.
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 *
 * DEPENDENCY NOTE: the store epilogue uses make_buffer_resource /
 * make_buffer_view / make_static_tile_distribution / make_static_distributed_
 * tensor / store_tile from rocke.helpers.{tensor_view,distribution}. Of these
 * only make_static_tile_distribution + the encoding constructor are ported to C
 * so far; the buffer-view / distributed-tensor / store_tile family is declared
 * here (and forward-declared in the .c) as a TODO(port) surface so the build
 * entry can emit the byte-identical call sequence. The verify+fix loop resolves
 * any diff once those helpers land.
 */
#ifndef ROCKE_INSTANCE_POOLING_H
#define ROCKE_INSTANCE_POOLING_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

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
typedef struct rocke_pooling_problem
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
} rocke_pooling_problem_t;

/* PoolingProblem with dataclass defaults installed (sH=sW=dH=dW=1, pH=pW=0) and
 * the six required dims zeroed. The caller fills N, H, W, C, Y, X. */
rocke_pooling_problem_t rocke_pooling_problem_default(void);

/* PoolingProblem.Ho property: (H + 2*pH - ((Y-1)*dH + 1)) // sH + 1.
 * Floor division matches Python // for the non-negative numerator. */
int rocke_pooling_problem_ho(const rocke_pooling_problem_t* p);

/* PoolingProblem.Wo property: (W + 2*pW - ((X-1)*dW + 1)) // sW + 1. */
int rocke_pooling_problem_wo(const rocke_pooling_problem_t* p);

/* PoolingProblem.total_out property: N * Ho * Wo * C. */
int rocke_pooling_problem_total_out(const rocke_pooling_problem_t* p);

/* PoolingProblem.short() ->
 *   f"N{N}H{H}W{W}C{C}_Y{Y}X{X}_s{sH}x{sW}_p{pH}x{pW}"
 * Writes the NUL-terminated string into out (capacity out_cap). Returns ROCKE_OK,
 * or ROCKE_ERR_VALUE on NULL args / a too-small buffer. */
rocke_status_t
    rocke_pooling_problem_short(const rocke_pooling_problem_t* p, char* out, size_t out_cap);

/* ------------------------------------------------------------- Pooling2DSpec *
 *
 * @dataclass(frozen=True)
 * class Pooling2DSpec:
 *     problem: PoolingProblem
 *     dtype: DType = "f16"            # "f16" | "bf16"
 *     op: PoolOp = "max"             # "max" | "avg" | "sum"
 *     block_size: int = 256
 *     vec: int = 1
 *     name: str = "rocke_pooling2d"
 *     tile_n: int = 1                # P81 (unused by v1 build path)
 *     use_warp_xor_reduce: bool = False  # P82 (unused by v1 build path)
 */
typedef struct rocke_pooling2d_spec
{
    rocke_pooling_problem_t problem;
    const char* dtype; /* "f16" | "bf16"; default "f16"        */
    const char* op; /* "max" | "avg" | "sum"; default "max" */
    int block_size; /* default 256                          */
    int vec; /* default 1                            */
    const char* name; /* default "rocke_pooling2d"           */
    int tile_n; /* default 1                            */
    bool use_warp_xor_reduce; /* default false                 */
} rocke_pooling2d_spec_t;

/* Default-constructed spec (dtype "f16", op "max", block_size 256, vec 1, name
 * "rocke_pooling2d", tile_n 1, use_warp_xor_reduce false, problem ==
 * rocke_pooling_problem_default()). The caller fills problem's six required
 * dims and overrides any field. */
rocke_pooling2d_spec_t rocke_pooling2d_spec_default(void);

/* Pooling2DSpec.kernel_name():
 *   kernel_name_join(name, problem.short(), dtype, op, f"b{block_size}",
 *                    f"v{vec}")
 * Result written NUL-terminated into out (capacity out_cap). Returns ROCKE_OK or
 * ROCKE_ERR_VALUE (NULL args / buffer too small). */
rocke_status_t
    rocke_pooling2d_kernel_name(const rocke_pooling2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the message and
 * false is returned. On accept returns true and writes "ok". */
bool rocke_pooling2d_is_valid_spec(const rocke_pooling2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* build_pooling2d(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` NULL => "gfx950". The kernel name is set by the builder init; this
 * routine does NOT re-init the builder. */
rocke_kernel_def_t* rocke_build_pooling2d(rocke_ir_builder_t* b,
                                          const rocke_pooling2d_spec_t* spec,
                                          const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_pooling2d_new(rocke_ir_builder_t* b,
                                              const rocke_pooling2d_spec_t* spec,
                                              const char* arch);

/* pooling2d_grid(spec) -> (x, y, z): one thread per vec-element output slab.
 *   total_v = total_out // max(vec, 1); grid = ceil_div_grid(total_v, block_size)
 * out[0..2] receive the grid. Returns ROCKE_OK or ROCKE_ERR_VALUE. */
rocke_status_t rocke_pooling2d_grid(const rocke_pooling2d_spec_t* spec, int out[3]);

/* pooling2d_signature(spec): the four-entry manifest
 *   ptr X, ptr Y, scalar X_bytes:i32, scalar Y_bytes:i32.
 * Writes up to out_cap entries into out[] and sets *out_count. `arena` owns the
 * copied name/type strings. Returns ROCKE_OK or an error status. */
struct rocke_sig_entry; /* fwd decl (rocke/helper_rocke.helpers.spec.h) */
struct rocke_arena; /* fwd decl (rocke/arena.h) */
rocke_status_t rocke_pooling2d_signature(struct rocke_arena* arena,
                                         const rocke_pooling2d_spec_t* spec,
                                         struct rocke_sig_entry* out,
                                         size_t out_cap,
                                         size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns its IRBuilder. */
rocke_status_t rocke_pooling2d_lower_to_llvm(const rocke_pooling2d_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_POOLING_H */
