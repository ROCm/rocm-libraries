/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_rmsnorm2d.h -- C99 port of the RMSNorm2D forward kernel
 * instance builder rocke/instances/common/rmsnorm2d.py (CK Tile
 * ``10_rmsnorm2d`` parity).
 *
 * For each row of an (M, N) activation tensor:
 *
 *     rms[m]     = sqrt(sum_n(X[m,n]^2) / N + eps)
 *     inv_rms[m] = 1 / rms[m]
 *     Y[m,n]     = X[m,n] * inv_rms[m] * gamma[n]
 *
 * The builder-call sequence is byte-identical to the Python so the emitted IR
 * op stream matches exactly. The kernel is a pure elementwise + LDS-tree
 * reduction norm (no MFMA atoms).
 *
 *   Python (rmsnorm2d.py)                 C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class RMSNorm2DSpec                   rocke_rmsnorm2d_spec_t
 *   RMSNorm2DSpec.elems_per_thread        rocke_rmsnorm2d_elems_per_thread(...)
 *   RMSNorm2DSpec.kernel_name()           rocke_rmsnorm2d_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_rmsnorm2d_is_valid_spec(...)
 *   build_rmsnorm2d(spec)                 rocke_build_rmsnorm2d(...)
 *   rmsnorm2d_grid(m, spec)               rocke_rmsnorm2d_grid(...)
 *   rmsnorm2d_signature(spec)             rocke_rmsnorm2d_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_rmsnorm2d_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type
 * with defaults. rocke_rmsnorm2d_spec_default() returns a struct with every
 * field at the Python dataclass default; the caller overrides what it needs
 * (n_per_block is required) before calling the build/validate/name entries.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_RMSNORM2D_H
#define ROCKE_INSTANCE_RMSNORM2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------- RMSNorm2DSpec *
 *
 * Mirror of the Python frozen dataclass:
 *
 *   @dataclass(frozen=True)
 *   class RMSNorm2DSpec:
 *       n_per_block: int
 *       block_size: int = 256
 *       vec: int = 4
 *       dtype: DType = "f16"          # "f16" | "bf16"
 *       save_inv_rms: bool = False
 *       wave_size: int = 64
 *       name: str = "rocke_rmsnorm2d_fwd"
 *
 * `dtype` / `name` are const char* (string literals); compared by strcmp like
 * the Python. n_per_block has no default (it is the one required field), but
 * rocke_rmsnorm2d_spec_default() seeds it to 0 so an unset value is visible. */
typedef struct rocke_rmsnorm2d_spec
{
    int n_per_block;
    int block_size; /* default 256 */
    int vec; /* default 4   */
    const char* dtype; /* default "f16" ; one of "f16" | "bf16" */
    bool save_inv_rms; /* default false */
    int wave_size; /* default 64  */
    const char* name; /* default "rocke_rmsnorm2d_fwd" */
} rocke_rmsnorm2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default;
 * n_per_block seeded to 0). The caller must set n_per_block. */
rocke_rmsnorm2d_spec_t rocke_rmsnorm2d_spec_default(void);

/* RMSNorm2DSpec.elems_per_thread @property: n_per_block // block_size. */
int rocke_rmsnorm2d_elems_per_thread(const rocke_rmsnorm2d_spec_t* spec);

/* RMSNorm2DSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *
 *   kernel_name_join(name, dtype, f"N{n_per_block}", f"b{block_size}",
 *                    f"v{vec}", flags={"sr": save_inv_rms})
 *
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE (buffer too small / format error). */
rocke_status_t
    rocke_rmsnorm2d_kernel_name(const rocke_rmsnorm2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "". */
bool rocke_rmsnorm2d_is_valid_spec(const rocke_rmsnorm2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* build_rmsnorm2d(spec). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd, with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set.
 *
 * The Python build_rmsnorm2d takes no arch (validation defaults to "gfx950");
 * `arch` is threaded here so rocke_rmsnorm2d_is_valid_spec can use it. NULL =>
 * "gfx950". This routine does NOT re-init the builder. */
rocke_kernel_def_t* rocke_build_rmsnorm2d(rocke_ir_builder_t* b,
                                          const rocke_rmsnorm2d_spec_t* spec,
                                          const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_rmsnorm2d_new(rocke_ir_builder_t* b,
                                              const rocke_rmsnorm2d_spec_t* spec,
                                              const char* arch);

/* rmsnorm2d_grid(m, spec) -> (m, 1, 1) via ceil_div_grid((m, 1)). out[0..2]
 * receive (x, y, z). Returns ROCKE_OK or the ceil_div_grid status. */
rocke_status_t rocke_rmsnorm2d_grid(int m, const rocke_rmsnorm2d_spec_t* spec, int out[3]);

/* rmsnorm2d_signature(spec): the manifest signature list. Entries (name/type
 * strings) are arena-owned; pass a live arena. On success *out_items /
 * *out_count receive the (arena-owned) array. Returns ROCKE_OK or an error.
 *
 *   X, Gamma, Y : ptr<dtype, global>
 *   [InvRms     : ptr<dtype, global>]   (only when save_inv_rms)
 *   M : i32 ; N : i32 ; eps : f32 */
rocke_status_t rocke_rmsnorm2d_signature(rocke_arena_t* arena,
                                         const rocke_rmsnorm2d_spec_t* spec,
                                         const rocke_sig_entry_t** out_items,
                                         size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_rmsnorm2d_lower_to_llvm(const rocke_rmsnorm2d_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_RMSNORM2D_H */
