/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_smoothquant.h -- C99 port of the public surface of the
 * SmoothQuant instance (ck_dsl/instances/common/smoothquant.py).
 *
 *   Python (smoothquant.py)                C99 (this header)
 *   -------------------------------------  -------------------------------------
 *   class SmoothQuantSpec (frozen)         ckc_smoothquant_spec_t
 *   SmoothQuantSpec.elems_per_thread       ckc_smoothquant_elems_per_thread()
 *   SmoothQuantSpec.kernel_name()          ckc_smoothquant_kernel_name()
 *   is_valid_spec(spec, arch)              ckc_smoothquant_is_valid_spec()
 *   build_smoothquant(spec, arch)          ckc_build_smoothquant()
 *   smoothquant_grid(m, spec)              ckc_smoothquant_grid()
 *
 * SmoothQuant is a row-wise dynamic-quantisation kernel: for an (M, N)
 * activation tensor X and a per-channel smooth scale SmScale (N,), it emits
 * QY (M, N) quantised (i8 / fp8e4m3 / bf8e5m2) plus YScale (M,) per-row scales,
 * via a two-pass LDS-tree amax fold (pass 1 amax, pass 2 quantise + store).
 *
 * The build reproduces the Python IRBuilder call sequence op-for-op so the
 * produced ckc_kernel_def_t is byte-faithful to the Python output.
 *
 * Error model mirrors the rest of the C port: the validity gate is a bool +
 * reason buffer; the build routes errors through the sticky-error IRBuilder and
 * returns NULL; the lower convenience returns a ckc_status_t.
 *
 * PORTING NOTE: a handful of upstream helpers this instance leans on
 * (distribution.load_tile / store_tile / make_static_distributed_tensor,
 * tensor_view.make_naive_tensor_view_packed / make_lds_view, and the F32-view
 * load_vec_as_f32) are not yet present in the C helper set. The build entry is
 * laid out op-for-op against the Python; the pass-1 X-tile load and the pass-2
 * QY distributed store are wired through local STUB shims marked TODO(port)
 * that the verify+fix loop resolves once those helpers land.
 */
#ifndef CKC_INSTANCE_SMOOTHQUANT_H
#define CKC_INSTANCE_SMOOTHQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default architecture (Python default arg arch="gfx950"). */
#define CKC_SMOOTHQUANT_DEFAULT_ARCH "gfx950"

/* ------------------------------------------------------------------ *
 * SmoothQuantSpec
 * ------------------------------------------------------------------ *
 *
 * Mirrors the frozen dataclass field-for-field. Defaults (Python):
 *   dtype="f16", out_dtype="i8", block_size=256, vec=4, save_yscale=True,
 *   wave_size=64, name="ck_dsl_smoothquant".
 *
 * `dtype` is one of "f16"/"bf16"; `out_dtype` is one of
 * "i8"/"fp8e4m3"/"bf8e5m2". Both are referenced as-is (not copied).
 */
typedef struct ckc_smoothquant_spec
{
    int n_per_block;
    const char* dtype; /* "f16" / "bf16"                    */
    const char* out_dtype; /* "i8" / "fp8e4m3" / "bf8e5m2"      */
    int block_size;
    int vec;
    bool save_yscale;
    int wave_size;
    const char* name; /* kernel-name prefix                */
} ckc_smoothquant_spec_t;

/* Initialise `spec` with the Python dataclass defaults and the one required
 * positional field (n_per_block). dtype/out_dtype/name point at static literals;
 * callers may overwrite any field afterwards. */
void ckc_smoothquant_spec_init(ckc_smoothquant_spec_t* spec, int n_per_block);

/* SmoothQuantSpec.elems_per_thread property: n_per_block // block_size. */
int ckc_smoothquant_elems_per_thread(const ckc_smoothquant_spec_t* spec);

/* SmoothQuantSpec.kernel_name(): writes the joined name into `out`
 * (capacity out_cap, NUL-terminated). Returns CKC_OK or CKC_ERR_VALUE when the
 * buffer is too small. */
ckc_status_t
    ckc_smoothquant_kernel_name(const ckc_smoothquant_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * Returns true (and writes "" to `reason` when non-NULL) on accept, or false
 * with the structured Python reason string on reject. `arch` NULL =>
 * CKC_SMOOTHQUANT_DEFAULT_ARCH ("gfx950"). `reason`/`reason_cap` may be NULL/0
 * to skip the message. Mirrors is_valid_spec(spec, arch). */
bool ckc_smoothquant_is_valid_spec(const ckc_smoothquant_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_smoothquant
 * ------------------------------------------------------------------ *
 *
 * Validates `spec` against `arch` via ckc_smoothquant_is_valid_spec(), then
 * builds the SmoothQuant forward IR into the supplied (already
 * ckc_ir_builder_init'd) builder `b`, op-for-op against build_smoothquant().
 * Returns b->kernel on success or NULL with b's sticky error set. `arch` NULL
 * => "gfx950".
 *
 * Like the Python (IRBuilder(spec.kernel_name())), this does NOT re-init the
 * builder; the caller owns its lifetime and should have created it with the
 * spec's kernel name. Use ckc_build_smoothquant_new() for the convenience. */
ckc_kernel_def_t* ckc_build_smoothquant(ckc_ir_builder_t* b,
                                        const ckc_smoothquant_spec_t* spec,
                                        const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_smoothquant_new(ckc_ir_builder_t* b,
                                            const ckc_smoothquant_spec_t* spec,
                                            const char* arch);

/* ------------------------------------------------------------------ *
 * smoothquant_grid
 * ------------------------------------------------------------------ *
 *
 * Launch grid: one CTA per row -> ceil_div_grid((m, 1)). Writes (x, y, z) into
 * out[0..2]. Returns CKC_OK or the ceil_div_grid error. Mirrors
 * smoothquant_grid(m, spec). */
ckc_status_t ckc_smoothquant_grid(int m, const ckc_smoothquant_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build, and lower to LLVM .ll text. `arch` NULL
 * => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns/frees its IRBuilder. */
ckc_status_t ckc_smoothquant_lower_to_llvm(const ckc_smoothquant_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_SMOOTHQUANT_H */
