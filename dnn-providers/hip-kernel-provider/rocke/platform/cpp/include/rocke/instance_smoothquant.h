/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_smoothquant.h -- C99 port of the public surface of the
 * SmoothQuant instance (rocke/instances/common/smoothquant.py).
 *
 *   Python (smoothquant.py)                C99 (this header)
 *   -------------------------------------  -------------------------------------
 *   class SmoothQuantSpec (frozen)         rocke_smoothquant_spec_t
 *   SmoothQuantSpec.elems_per_thread       rocke_smoothquant_elems_per_thread()
 *   SmoothQuantSpec.kernel_name()          rocke_smoothquant_kernel_name()
 *   is_valid_spec(spec, arch)              rocke_smoothquant_is_valid_spec()
 *   build_smoothquant(spec, arch)          rocke_build_smoothquant()
 *   smoothquant_grid(m, spec)              rocke_smoothquant_grid()
 *
 * SmoothQuant is a row-wise dynamic-quantisation kernel: for an (M, N)
 * activation tensor X and a per-channel smooth scale SmScale (N,), it emits
 * QY (M, N) quantised (i8 / fp8e4m3 / bf8e5m2) plus YScale (M,) per-row scales,
 * via a two-pass LDS-tree amax fold (pass 1 amax, pass 2 quantise + store).
 *
 * The build reproduces the Python IRBuilder call sequence op-for-op so the
 * produced rocke_kernel_def_t is byte-faithful to the Python output.
 *
 * Error model mirrors the rest of the C port: the validity gate is a bool +
 * reason buffer; the build routes errors through the sticky-error IRBuilder and
 * returns NULL; the lower convenience returns a rocke_status_t.
 *
 * PORTING NOTE: a handful of upstream helpers this instance leans on
 * (distribution.load_tile / store_tile / make_static_distributed_tensor,
 * tensor_view.make_naive_tensor_view_packed / make_lds_view, and the F32-view
 * load_vec_as_f32) are not yet present in the C helper set. The build entry is
 * laid out op-for-op against the Python; the pass-1 X-tile load and the pass-2
 * QY distributed store are wired through local STUB shims marked TODO(port)
 * that the verify+fix loop resolves once those helpers land.
 */
#ifndef ROCKE_INSTANCE_SMOOTHQUANT_H
#define ROCKE_INSTANCE_SMOOTHQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default architecture (Python default arg arch="gfx950"). */
#define ROCKE_SMOOTHQUANT_DEFAULT_ARCH "gfx950"

/* ------------------------------------------------------------------ *
 * SmoothQuantSpec
 * ------------------------------------------------------------------ *
 *
 * Mirrors the frozen dataclass field-for-field. Defaults (Python):
 *   dtype="f16", out_dtype="i8", block_size=256, vec=4, save_yscale=True,
 *   wave_size=64, name="rocke_smoothquant".
 *
 * `dtype` is one of "f16"/"bf16"; `out_dtype` is one of
 * "i8"/"fp8e4m3"/"bf8e5m2". Both are referenced as-is (not copied).
 */
typedef struct rocke_smoothquant_spec
{
    int n_per_block;
    const char* dtype; /* "f16" / "bf16"                    */
    const char* out_dtype; /* "i8" / "fp8e4m3" / "bf8e5m2"      */
    int block_size;
    int vec;
    bool save_yscale;
    int wave_size;
    const char* name; /* kernel-name prefix                */
} rocke_smoothquant_spec_t;

/* Initialise `spec` with the Python dataclass defaults and the one required
 * positional field (n_per_block). dtype/out_dtype/name point at static literals;
 * callers may overwrite any field afterwards. */
void rocke_smoothquant_spec_init(rocke_smoothquant_spec_t* spec, int n_per_block);

/* SmoothQuantSpec.elems_per_thread property: n_per_block // block_size. */
int rocke_smoothquant_elems_per_thread(const rocke_smoothquant_spec_t* spec);

/* SmoothQuantSpec.kernel_name(): writes the joined name into `out`
 * (capacity out_cap, NUL-terminated). Returns ROCKE_OK or ROCKE_ERR_VALUE when the
 * buffer is too small. */
rocke_status_t
    rocke_smoothquant_kernel_name(const rocke_smoothquant_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * Returns true (and writes "" to `reason` when non-NULL) on accept, or false
 * with the structured Python reason string on reject. `arch` NULL =>
 * ROCKE_SMOOTHQUANT_DEFAULT_ARCH ("gfx950"). `reason`/`reason_cap` may be NULL/0
 * to skip the message. Mirrors is_valid_spec(spec, arch). */
bool rocke_smoothquant_is_valid_spec(const rocke_smoothquant_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_smoothquant
 * ------------------------------------------------------------------ *
 *
 * Validates `spec` against `arch` via rocke_smoothquant_is_valid_spec(), then
 * builds the SmoothQuant forward IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, op-for-op against build_smoothquant().
 * Returns b->kernel on success or NULL with b's sticky error set. `arch` NULL
 * => "gfx950".
 *
 * Like the Python (IRBuilder(spec.kernel_name())), this does NOT re-init the
 * builder; the caller owns its lifetime and should have created it with the
 * spec's kernel name. Use rocke_build_smoothquant_new() for the convenience. */
rocke_kernel_def_t* rocke_build_smoothquant(rocke_ir_builder_t* b,
                                            const rocke_smoothquant_spec_t* spec,
                                            const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_smoothquant_new(rocke_ir_builder_t* b,
                                                const rocke_smoothquant_spec_t* spec,
                                                const char* arch);

/* ------------------------------------------------------------------ *
 * smoothquant_grid
 * ------------------------------------------------------------------ *
 *
 * Launch grid: one CTA per row -> ceil_div_grid((m, 1)). Writes (x, y, z) into
 * out[0..2]. Returns ROCKE_OK or the ceil_div_grid error. Mirrors
 * smoothquant_grid(m, spec). */
rocke_status_t rocke_smoothquant_grid(int m, const rocke_smoothquant_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build, and lower to LLVM .ll text. `arch` NULL
 * => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns/frees its IRBuilder. */
rocke_status_t rocke_smoothquant_lower_to_llvm(const rocke_smoothquant_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_SMOOTHQUANT_H */
