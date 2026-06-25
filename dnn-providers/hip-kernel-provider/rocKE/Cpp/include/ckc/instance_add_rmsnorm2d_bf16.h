/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_add_rmsnorm2d_bf16.h -- C99 port of the fused add + RMSNorm
 * (bf16/f16 output, no quantization) kernel instance builder
 * ck_dsl/instances/common/add_rmsnorm2d_bf16.py.
 *
 * For two (M, N) activation tensors A and B and an (N,) Gamma per-channel
 * scale, the kernel produces both the residual sum X = A + B and the
 * RMSNorm-normalized output Y in one pass over global memory:
 *
 *     x[m, n]    = a[m, n] + b[m, n]
 *     sum_sq[m]  = sum_n(x[m, n] ^ 2)
 *     inv_rms[m] = 1 / sqrt(sum_sq[m] / N + eps)
 *     y[m, n]    = x[m, n] * inv_rms[m] * gamma[n]
 *
 *   Python (add_rmsnorm2d_bf16.py)        C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class AddRMSNorm2DBF16Spec            ckc_add_rmsnorm2d_bf16_spec_t
 *   is_valid_spec(spec, arch)             ckc_is_valid_spec_add_rmsnorm2d_bf16(...)
 *   build_add_rmsnorm2d_bf16(spec, arch)  ckc_build_add_rmsnorm2d_bf16(...)
 *   add_rmsnorm2d_bf16_grid(m, spec)      ckc_add_rmsnorm2d_bf16_grid(...)
 *   add_rmsnorm2d_bf16_signature(spec)    ckc_add_rmsnorm2d_bf16_signature(...)
 *   (+ convenience: build -> lower .ll)   ckc_add_rmsnorm2d_bf16_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python @dataclass(frozen=True) is a value
 * type with defaults. In C the caller fills a ckc_add_rmsnorm2d_bf16_spec_t;
 * ckc_add_rmsnorm2d_bf16_spec_default() returns a struct with every field set
 * to the Python dataclass default (only n_per_block is mandatory).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_ADD_RMSNORM2D_BF16_H
#define CKC_INSTANCE_ADD_RMSNORM2D_BF16_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t */
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------- AddRMSNorm2DBF16Spec *
 *
 * Mirror of the Python @dataclass(frozen=True). `dtype` is the Python Literal
 * "f16" | "bf16" string (compared by strcmp like Python). */
typedef struct ckc_add_rmsnorm2d_bf16_spec
{
    int n_per_block; /* mandatory; no default                       */
    int block_size; /* default 256                                 */
    int vec; /* default 4                                   */
    const char* dtype; /* default "bf16"                              */
    bool save_residual; /* default true (write x = a + b to X)         */
    int wave_size; /* default 64                                  */
    const char* name; /* default "ck_dsl_add_rmsnorm2d_bf16"         */
} ckc_add_rmsnorm2d_bf16_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `n_per_block` (the one field with no default). */
ckc_add_rmsnorm2d_bf16_spec_t ckc_add_rmsnorm2d_bf16_spec_default(void);

/* AddRMSNorm2DBF16Spec.elems_per_thread property: n_per_block // block_size. */
int ckc_add_rmsnorm2d_bf16_elems_per_thread(const ckc_add_rmsnorm2d_bf16_spec_t* spec);

/* AddRMSNorm2DBF16Spec.kernel_name() -> NUL-terminated into out (capacity
 * out_cap). Returns CKC_OK or CKC_ERR_VALUE (buffer too small). */
ckc_status_t ckc_add_rmsnorm2d_bf16_kernel_name(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "". */
bool ckc_is_valid_spec_add_rmsnorm2d_bf16(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* build_add_rmsnorm2d_bf16(spec, arch). Builds the IR into the supplied
 * (already ckc_ir_builder_init'd) builder `b`, exactly as the Python build
 * does, and returns the kernel (b->kernel) on success or NULL with b's sticky
 * error set. `arch` NULL => "gfx950".
 *
 * NOTE: like the Python, this expects the builder to have been created with the
 * spec's kernel_name(). Use ckc_build_add_rmsnorm2d_bf16_new() for the
 * init-from-spec convenience. */
ckc_kernel_def_t* ckc_build_add_rmsnorm2d_bf16(ckc_ir_builder_t* b,
                                               const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                               const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_add_rmsnorm2d_bf16_new(ckc_ir_builder_t* b,
                                                   const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                                   const char* arch);

/* add_rmsnorm2d_bf16_grid(m, spec): one CTA per row -> ceil_div_grid((m, 1)).
 * Writes (x, y, z) into out[3]. Returns CKC_OK or CKC_ERR_VALUE. */
ckc_status_t
    ckc_add_rmsnorm2d_bf16_grid(int m, const ckc_add_rmsnorm2d_bf16_spec_t* spec, int out[3]);

/* add_rmsnorm2d_bf16_signature(spec): the {name,type} ABI manifest. `arena`
 * owns the entry array + copied strings. On CKC_OK *out_items / *out_count hold
 * the array. */
ckc_status_t ckc_add_rmsnorm2d_bf16_signature(ckc_arena_t* arena,
                                              const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                              const ckc_sig_entry_t** out_items,
                                              size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_add_rmsnorm2d_bf16_lower_to_llvm(const ckc_add_rmsnorm2d_bf16_spec_t* spec,
                                                  const char* arch,
                                                  ckc_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_ADD_RMSNORM2D_BF16_H */
