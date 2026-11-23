/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d.h -- C99 port of the BatchNorm2D forward kernel
 * instance builder rocke/instances/common/batchnorm2d.py.
 *
 * For an (N, C, H, W) activation tensor the kernel computes, per channel c:
 *
 *     mean[c]    = sum_{n,h,w}(X[n,c,h,w]) / (N*H*W)
 *     var[c]     = sum_{n,h,w}((X[n,c,h,w] - mean[c])^2) / (N*H*W)  (Welford)
 *     inv_std[c] = 1 / sqrt(var[c] + eps)
 *     Y[n,c,h,w] = (X[n,c,h,w] - mean[c]) * inv_std[c] * gamma[c] + beta[c]
 *
 * training=true additionally updates RunningMean/RunningVar with the usual
 * EMA (running = (1-momentum)*running + momentum*batch_stat); training=false
 * skips the reduction and normalises with the caller-supplied running stats.
 *
 * See the Python module's docstring for why this is its own family (not a
 * third `kind` in dispatch/families/norm.py), why the layout is NCHW, and
 * why the reduction is an N-loop of per-row sweeps rather than a new
 * cross-CTA primitive.
 *
 * The build entry mirrors build_batchnorm2d(spec) -> KernelDef:
 *
 *   Python (batchnorm2d.py)                C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   class BatchNorm2DSpec                  rocke_batchnorm2d_spec_t
 *   is_valid_spec(spec, arch)              rocke_batchnorm2d_is_valid_spec(...)
 *   build_batchnorm2d(spec)                rocke_build_batchnorm2d(b, spec)
 *   batchnorm2d_grid(c, spec)              rocke_batchnorm2d_grid(...)
 *   batchnorm2d_signature(spec)            rocke_batchnorm2d_signature(...)
 *   (+ convenience: build -> lower .ll)    rocke_batchnorm2d_lower_to_llvm(...)
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_H
#define ROCKE_INSTANCE_BATCHNORM2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ BatchNorm2DSpec
 *
 * Mirror of Python BatchNorm2DSpec. `dtype` is the Literal["f16","bf16"]
 * string (compared by strcmp like Python). The computed @property fields
 * (hw, n_per_channel, elems_per_thread) are NOT stored; use the accessor
 * functions below. */
typedef struct rocke_batchnorm2d_spec
{
    int n;
    int c;
    int h;
    int w;
    int block_size; /* default 256                       */
    int vec; /* default 4                         */
    const char* dtype; /* default "f16" ("f16"|"bf16")      */
    bool training; /* default true                      */
    bool save_mean_invstd; /* default false                     */
    int channels_per_block; /* default 1 (byte-identical when 1) */
    int n_groups; /* default 1 (byte-identical when 1) */
    bool fast_reduce; /* default false (byte-identical when false) */
    int wave_size; /* default 64                        */
    const char* name; /* default "rocke_batchnorm2d_fwd"  */
} rocke_batchnorm2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set n/c/h/w. */
rocke_batchnorm2d_spec_t rocke_batchnorm2d_spec_default(void);

/* BatchNorm2DSpec.hw property: h * w. */
int rocke_batchnorm2d_hw(const rocke_batchnorm2d_spec_t* spec);

/* BatchNorm2DSpec.n_per_channel property: n * h * w. */
int rocke_batchnorm2d_n_per_channel(const rocke_batchnorm2d_spec_t* spec);

/* BatchNorm2DSpec.elems_per_thread property: n_per_channel // block_size. */
int rocke_batchnorm2d_elems_per_thread(const rocke_batchnorm2d_spec_t* spec);

/* BatchNorm2DSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_batchnorm2d_kernel_name(const rocke_batchnorm2d_spec_t* spec,
                                             char* out,
                                             size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "". */
bool rocke_batchnorm2d_is_valid_spec(const rocke_batchnorm2d_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* build_batchnorm2d(spec). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. The caller owns `b`'s lifetime. */
rocke_kernel_def_t* rocke_build_batchnorm2d(rocke_ir_builder_t* b,
                                            const rocke_batchnorm2d_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_batchnorm2d_new(rocke_ir_builder_t* b,
                                                const rocke_batchnorm2d_spec_t* spec);

/* batchnorm2d_grid(c, spec) -> (x, y, z). Returns (c, 1, 1). */
rocke_status_t
    rocke_batchnorm2d_grid(int c, const rocke_batchnorm2d_spec_t* spec, int out[3]);

/* batchnorm2d_signature(spec): build the {name,type} manifest into the
 * caller-provided `sb` (already rocke_signature_builder_init'd). Returns the
 * accumulated entries via *out_items / *out_count (arena-owned by sb). */
rocke_status_t rocke_batchnorm2d_signature(rocke_signature_builder_t* sb,
                                           const rocke_batchnorm2d_spec_t* spec,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_batchnorm2d_lower_to_llvm(const rocke_batchnorm2d_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

/* ------------------------------------------------------ cross-CTA split reduction
 *
 * C99 port of rocke/instances/common/batchnorm2d.py's
 * BatchNorm2DSplitSpec / build_batchnorm2d_reduce_partial /
 * build_batchnorm2d_combine_apply: a two-kernel, workspace-based design for
 * large-H*W/small-C shapes where one CTA per channel leaves most CUs idle.
 * See the Python module's "Cross-CTA spatial-split reduction" section for
 * the full design rationale (why this needs two kernels, not an atomic
 * epilogue). Deliberately standalone (no dispatch/KernelCandidate wiring,
 * no save_mean_invstd) -- see that section's docstring for scope notes. */
typedef struct rocke_batchnorm2d_split_spec
{
    int n;
    int c;
    int h;
    int w;
    int split_hw;
    int block_size; /* default 256 */
    int vec; /* default 4   */
    const char* dtype; /* default "f16" ("f16"|"bf16") */
    int channels_per_block; /* default 1 */
    bool fast_reduce; /* default false */
    int wave_size; /* default 64 */
    const char* name; /* default "rocke_batchnorm2d_split" */
} rocke_batchnorm2d_split_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set n/c/h/w/split_hw. */
rocke_batchnorm2d_split_spec_t rocke_batchnorm2d_split_spec_default(void);

/* BatchNorm2DSplitSpec.hw property: h * w. */
int rocke_batchnorm2d_split_hw(const rocke_batchnorm2d_split_spec_t* spec);

/* BatchNorm2DSplitSpec.hw_chunk property: ceil(h*w / split_hw). */
int rocke_batchnorm2d_split_hw_chunk(const rocke_batchnorm2d_split_spec_t* spec);

/* batchnorm2d_split_workspace_floats(spec): f32 element count the caller
 * must allocate for Workspace (c * split_hw * 3). */
int rocke_batchnorm2d_split_workspace_floats(const rocke_batchnorm2d_split_spec_t* spec);

/* is_valid_split_spec(spec, arch) -> (ok, reason). Same reason/arch
 * contract as rocke_batchnorm2d_is_valid_spec. */
bool rocke_batchnorm2d_is_valid_split_spec(const rocke_batchnorm2d_split_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* batchnorm2d_reduce_partial_grid(c, spec) -> (c/cpb, split_hw, 1). */
rocke_status_t rocke_batchnorm2d_reduce_partial_grid(
    int c, const rocke_batchnorm2d_split_spec_t* spec, int out[3]);

/* batchnorm2d_combine_apply_grid(c, spec) -> (c/cpb, 1, 1). */
rocke_status_t rocke_batchnorm2d_combine_apply_grid(
    int c, const rocke_batchnorm2d_split_spec_t* spec, int out[3]);

/* build_batchnorm2d_reduce_partial(spec): kernel A of the split-reduction
 * pair. Builds into `b` (already rocke_ir_builder_init'd); returns b->kernel
 * or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_batchnorm2d_reduce_partial(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name("reduce"), then build. */
rocke_kernel_def_t* rocke_build_batchnorm2d_reduce_partial_new(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec);

/* build_batchnorm2d_combine_apply(spec): kernel B of the split-reduction
 * pair. Builds into `b` (already rocke_ir_builder_init'd); returns b->kernel
 * or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_batchnorm2d_combine_apply(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name("combine"), then build. */
rocke_kernel_def_t* rocke_build_batchnorm2d_combine_apply_new(
    rocke_ir_builder_t* b, const rocke_batchnorm2d_split_spec_t* spec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_H */
