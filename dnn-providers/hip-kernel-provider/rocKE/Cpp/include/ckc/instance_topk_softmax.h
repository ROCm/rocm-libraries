/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_topk_softmax.h -- C99 port of the topk-softmax kernel instance
 * builder ck_dsl/instances/common/topk_softmax.py.
 *
 * DSL counterpart of CK Tile's example/ck_tile/09_topk_softmax. Given X of
 * shape (M, N) (MoE router logits), the kernel produces:
 *   Y   : (M, K) softmax probabilities over the K selected entries.
 *   Idx : (M, K) i32 indices of the K selected entries.
 *
 *   Python (topk_softmax.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class TopkSoftmaxSpec                 ckc_topk_softmax_spec_t
 *   is_valid_spec(spec, arch)             ckc_topk_softmax_is_valid_spec(...)
 *   build_topk_softmax(spec, arch)        ckc_build_topk_softmax(...)
 *   topk_softmax_grid(m, spec)            ckc_topk_softmax_grid(...)
 *   (+ convenience: build -> lower .ll)   ckc_topk_softmax_lower_to_llvm(...)
 *
 * Byte-identical builder-call sequence vs the Python build: every ckc_b_* call
 * mirrors an IRBuilder method call in build_topk_softmax, in the same order.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a reason
 * string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_TOPK_SOFTMAX_H
#define CKC_INSTANCE_TOPK_SOFTMAX_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------- TopkSoftmaxSpec *
 *
 * Mirror of the Python frozen dataclass. dtype / out_dtype are the Python
 * Literal["f16","bf16","f32"] strings, compared by strcmp. The computed
 * @property elems_per_thread is NOT stored; recompute it with
 * ckc_topk_softmax_elems_per_thread(). */
typedef struct ckc_topk_softmax_spec
{
    int n_per_row; /* N -- entries per row (experts for MoE)        */
    int k; /* K -- top-k count                              */
    const char* dtype; /* input X dtype, default "f32"                  */
    const char* out_dtype; /* output Y dtype, default "f32"                 */
    int block_size; /* default 64                                    */
    const char* name; /* default "ck_dsl_topk_softmax"                 */
    bool cross_wave_argmax; /* default false (reserved; BS>64 uses LDS path) */
} ckc_topk_softmax_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set n_per_row + k (and any dtype/block_size overrides). */
ckc_topk_softmax_spec_t ckc_topk_softmax_spec_default(void);

/* TopkSoftmaxSpec.elems_per_thread = ceil(n_per_row / block_size). */
int ckc_topk_softmax_elems_per_thread(const ckc_topk_softmax_spec_t* spec);

/* TopkSoftmaxSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns CKC_OK or CKC_ERR_VALUE (buffer too small). */
ckc_status_t
    ckc_topk_softmax_kernel_name(const ckc_topk_softmax_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool ckc_topk_softmax_is_valid_spec(const ckc_topk_softmax_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* build_topk_softmax(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd, with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". Does NOT re-init the builder. */
ckc_kernel_def_t* ckc_build_topk_softmax(ckc_ir_builder_t* b,
                                         const ckc_topk_softmax_spec_t* spec,
                                         const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_topk_softmax_new(ckc_ir_builder_t* b,
                                             const ckc_topk_softmax_spec_t* spec,
                                             const char* arch);

/* topk_softmax_grid(m, spec): launch grid (one CTA per row). Writes (x,y,z)
 * into out[3]. Returns CKC_OK, or the ceil_div_grid status on failure. */
ckc_status_t ckc_topk_softmax_grid(int m, const ckc_topk_softmax_spec_t* spec, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_topk_softmax_lower_to_llvm(const ckc_topk_softmax_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_TOPK_SOFTMAX_H */
