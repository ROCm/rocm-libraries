/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_img2col.h -- C99 port of the image-to-column (im2col) kernel
 * instance builder rocke/instances/common/img2col.py.
 *
 * im2col materialises the implicit-GEMM A operand of a convolution as a real
 * [M, K] tensor (M = N*Ho*Wo, K = R*S*C). It is a pure index-transform + copy
 * kernel: no MFMA, no LDS staging. Per thread it computes one vec_k-wide K chunk
 * at (m, k_base), loads the NHWC source via the conv address-transform
 * descriptor (OOB-safe buffer loads zero-fill the padding zone) and writes the
 * chunk row-major into Y.
 *
 *   Python (img2col.py)                   C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   Img2ColSpec (frozen dataclass)        rocke_img2col_spec_t  (helper header)
 *   ConvProblem (frozen dataclass)        rocke_conv_problem_t  (helper header)
 *   is_valid_spec(spec, arch)             rocke_img2col_is_valid_spec(...)
 *   build_img2col(spec, arch)             rocke_build_img2col(...)
 *   img2col_grid(spec)                    rocke_img2col_grid(...)
 *   img2col_block_tile_m_for_M(M)         rocke_img2col_block_tile_m_for_M(...)
 *   img2col_signature(spec)               rocke_img2col_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_img2col_lower_to_llvm(...)
 *
 * The rocke_img2col_spec_t / rocke_conv_problem_t value types and their property
 * accessors (block_size, can_vector_load, kernel_name, M, K_gemm, short) are
 * defined in rocke/helper_rocke.instances.common.img2col.h and reused here.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_IMG2COL_H
#define ROCKE_INSTANCE_IMG2COL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common.img2col.h" /* spec + problem types */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* is_valid_spec(spec, arch) -> (ok, reason).
 *
 * Python (img2col.py:is_valid_spec):
 *   - ArchTarget.from_gfx(arch); KeyError -> (False, str(e))
 *   - dtype must be "f16"
 *   - vec_k in {1, 2, 4, 8}
 *   - block_tile_k % vec_k == 0
 *   - block_size > 0
 *   - block_size <= 1024
 *   - block_size % wave_size == 0
 *
 * `arch` NULL => "gfx950". On reject, `reason` (if non-NULL, capacity
 * reason_cap) receives the structured message and false is returned; on accept
 * returns true and writes "ok". */
bool rocke_img2col_is_valid_spec(const rocke_img2col_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap);

/* build_img2col(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". This routine does NOT re-init
 * the builder (the caller controls its lifetime). */
rocke_kernel_def_t*
    rocke_build_img2col(rocke_ir_builder_t* b, const rocke_img2col_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_img2col_new(rocke_ir_builder_t* b,
                                            const rocke_img2col_spec_t* spec,
                                            const char* arch);

/* img2col_grid(spec) -> (grid_x, grid_y, grid_z).
 *
 * Python:
 *   ceil_div_grid((problem.K_gemm, block_tile_k),
 *                 (problem.M,      block_tile_m))
 *
 * On success writes out[0..2] = (x, y, z) and returns ROCKE_OK; on the Python
 * ValueError (non-positive tile) returns ROCKE_ERR_VALUE. */
rocke_status_t rocke_img2col_grid(const rocke_img2col_spec_t* spec, int out[3]);

/* img2col_block_tile_m_for_M(M, default=8): smallest value in the ladder
 * (default, 16, 32, 64, 128, 256) for which ceil(M/cand) <= 65535; falls back
 * to 256. Pass default<=0 to take the Python default (8). */
int rocke_img2col_block_tile_m_for_M(int M, int dflt);

/* img2col_signature(spec):
 *   SignatureBuilder().ptr("X", dtype).ptr("Y", dtype)
 *       .scalar("X_bytes","i32").scalar("Y_bytes","i32").build()
 *
 * Writes the four manifest entries into out_items[] (capacity out_cap) and sets
 * *out_count to 4. The entry strings live in `arena` (pass a live arena).
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE on NULL args / out_cap < 4. */
rocke_status_t rocke_img2col_signature(rocke_arena_t* arena,
                                       const rocke_img2col_spec_t* spec,
                                       rocke_sig_entry_t* out_items,
                                       size_t out_cap,
                                       size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_img2col_lower_to_llvm(const rocke_img2col_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_IMG2COL_H */
