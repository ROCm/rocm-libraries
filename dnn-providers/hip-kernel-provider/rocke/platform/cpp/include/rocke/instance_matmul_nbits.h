/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_matmul_nbits.h -- C99 port of the public surface + family
 * dispatcher of the MatMulNBits gfx1151 instance
 * (rocke/instances/common/matmul_nbits.py).
 *
 *   Python (matmul_nbits.py)               C99 (this header)
 *   -------------------------------------  -------------------------------------
 *   is_valid_spec(spec, arch)              rocke_matmul_nbits_is_valid_spec()
 *   build_matmul_nbits(spec, arch)         rocke_build_matmul_nbits()
 *   (re-exported FAMILIES / spec / grid    see
 *    helpers)                               helper_rocke.instances.common._matmul_nbits_common.h
 *
 * This is the top-level dispatcher only. It composes the family-agnostic
 * validator (rocke_matmul_nbits_validate_common_spec) with the per-family geometry
 * extras (skinny_n narrow-N gate), then dispatches on spec->family:
 *
 *   * "large_n" / "skinny_n": share the WMMA tiled body; if spec->optimized the
 *     combined-optimization opt body is used, else the baseline large-N body
 *     (rocke_build_large_n_matmul_nbits, ported here -- the non-opt body has no
 *     dedicated peer module, only its _wmma_params helper does).
 *   * "decode_gemv": the dedicated scalar one-thread-per-column body
 *     (rocke_build_decode_gemv_matmul_nbits, peer port).
 *
 * The build reproduces the Python IRBuilder call sequence op-for-op so the
 * produced rocke_kernel_def_t is byte-faithful to the Python output.
 *
 * Error model mirrors the rest of the C port: the validity gate is a bool +
 * reason buffer; the build routes errors through the sticky-error IRBuilder and
 * returns NULL; the lower convenience returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MATMUL_NBITS_H
#define ROCKE_INSTANCE_MATMUL_NBITS_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._matmul_nbits_common.h" /* spec */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * Python:
 *   ok, reason = validate_common_spec(spec, arch)
 *   if not ok: return False, reason
 *   if spec.family == "skinny_n" and spec.N > 64:
 *       return False, "skinny_n is for narrow N (<= 64); N=... should use ..."
 *   return True, "ok"
 *
 * Returns true (and writes "ok" to `reason` when non-NULL) on accept, or false
 * with the structured Python reason string on reject. `arch` NULL => V1_ARCH
 * ("gfx1151"). `reason`/`reason_cap` may be NULL/0 to skip the message. */
bool rocke_matmul_nbits_is_valid_spec(const rocke_matmul_nbits_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_matmul_nbits
 * ------------------------------------------------------------------ *
 *
 * Validates `spec` against `arch` via rocke_matmul_nbits_is_valid_spec(), then
 * dispatches on spec->family:
 *
 *   * "large_n" / "skinny_n": spec->optimized ?
 *         rocke_build_large_n_opt_matmul_nbits() :
 *         rocke_build_large_n_matmul_nbits()
 *   * "decode_gemv": rocke_build_decode_gemv_matmul_nbits()
 *
 * Builds the IR into the supplied (already rocke_ir_builder_init'd) builder `b`,
 * exactly as the Python build does, and returns the kernel (b->kernel) on
 * success or NULL with b's sticky error set. `arch` NULL => "gfx1151".
 *
 * Like the Python (which constructs IRBuilder(spec.kernel_name())), this does
 * NOT re-init the builder; the caller owns its lifetime and should have created
 * it with the spec's kernel name. Use rocke_build_matmul_nbits_new() for the
 * init-from-spec convenience.
 *
 * A validation failure raises the Python ValueError; here it sets the builder's
 * sticky error (ROCKE_ERR_VALUE) with a Python-matching message and returns NULL. */
rocke_kernel_def_t* rocke_build_matmul_nbits(rocke_ir_builder_t* b,
                                             const rocke_matmul_nbits_spec_t* spec,
                                             const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_matmul_nbits_new(rocke_ir_builder_t* b,
                                                 const rocke_matmul_nbits_spec_t* spec,
                                                 const char* arch);

/* ------------------------------------------------------------------ *
 * rocke_build_large_n_matmul_nbits
 * ------------------------------------------------------------------ *
 *
 * The baseline (non-optimized) large-N WMMA tiled body, C99 port of
 * rocke/instances/common/_matmul_nbits_large_n.build_large_n_matmul_nbits.
 * Unlike the opt body this has no dedicated peer module (only its _wmma_params
 * helper is split out), so the body lives in this translation unit.
 *
 * Builds into the already-init'd builder `b` and returns b->kernel on success,
 * NULL with the sticky error set on failure. `arch` NULL => "gfx1151". */
rocke_kernel_def_t* rocke_build_large_n_matmul_nbits(rocke_ir_builder_t* b,
                                                     const rocke_matmul_nbits_spec_t* spec,
                                                     const char* arch);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, dispatch+build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx1151". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns/frees its
 * IRBuilder internally. */
rocke_status_t rocke_matmul_nbits_lower_to_llvm(const rocke_matmul_nbits_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MATMUL_NBITS_H */
