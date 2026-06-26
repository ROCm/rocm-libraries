/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_qwen3_qk_norm_rope.h -- C99 port of the gfx1250 fused
 * QK-norm + RoPE kernel rocke/instances/gfx1250/qwen3_qk_norm_rope.py.
 *
 *   Python (gfx1250/qwen3_qk_norm_rope.py)  C99 (this header)
 *   --------------------------------------  ---------------------------------------
 *   class Qwen3QkNormRopeSpec               rocke_qwen3_qk_norm_rope_gfx1250_spec_t
 *   Qwen3QkNormRopeSpec.kernel_name()       rocke_qwen3_qk_norm_rope_gfx1250_kernel_name()
 *   build_qwen3_qk_norm_rope(spec, arch)    rocke_build_qwen3_qk_norm_rope_gfx1250()
 *   qwen3_qk_norm_rope_grid(n, spec)        rocke_qwen3_qk_norm_rope_gfx1250_grid()
 *
 * Per (token, head): RMSNorm over head_dim (x * rsqrt(mean(x^2)+eps) * weight[d])
 * then RoPE (half / interleaved). One thread per (token, head) row, head_dim
 * unrolled at compile time (thread-local reduction, no cross-lane reduce).
 * Arch-neutral (elementwise + reduction math, no WMMA).
 */
#ifndef ROCKE_INSTANCE_GFX1250_QWEN3_QK_NORM_ROPE_H
#define ROCKE_INSTANCE_GFX1250_QWEN3_QK_NORM_ROPE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python Qwen3QkNormRopeSpec (frozen dataclass):
 *     num_heads: int           (required, no default)
 *     head_dim: int = 64
 *     dtype: str = "bf16"      # fp16/bf16
 *     eps: float = 1e-6
 *     rope_layout: str = "half"   # "half" or "interleaved"
 *     block_size: int = 64
 *     name: str = "rocke_gfx1250_qwen3_qk_norm_rope"
 * __post_init__ validates dtype/head_dim/num_heads/rope_layout; in C those checks
 * move into the validity gate / build. */
typedef struct rocke_qwen3_qk_norm_rope_gfx1250_spec
{
    int num_heads;
    int head_dim;
    const char* dtype;
    double eps;
    const char* rope_layout;
    int block_size;
    const char* name;
} rocke_qwen3_qk_norm_rope_gfx1250_spec_t;

/* Default-constructed spec. num_heads has NO Python default; the caller MUST set
 * it (default-constructed value is 0, which the validity gate rejects). */
rocke_qwen3_qk_norm_rope_gfx1250_spec_t rocke_qwen3_qk_norm_rope_gfx1250_spec_default(void);

/* Qwen3QkNormRopeSpec.kernel_name():
 *   kernel_name_join(self.name, f"h{num_heads}", f"d{head_dim}", dtype, rope_layout). */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_kernel_name(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, char* out, size_t out_cap);

/* Validity gate (mirrors __post_init__ + _require_supported): dtype fp16/bf16,
 * head_dim positive & even, num_heads positive, rope_layout half/interleaved,
 * ArchTarget.from_gfx(arch) resolves. On reject writes `reason` & returns false. */
bool rocke_qwen3_qk_norm_rope_gfx1250_is_valid_spec(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* build_qwen3_qk_norm_rope(spec, arch). Builds into the supplied builder `b`
 * (already rocke_ir_builder_init'd with spec.kernel_name()); returns b->kernel
 * or NULL with b's sticky error set. `arch` NULL => "gfx1250".
 *
 * Signature: (x_in: ptr<dt>, weight: ptr<f32>, cos: ptr<f32>, sin: ptr<f32>,
 *             positions: ptr<i32>, x_out: ptr<dt>, num_tokens: i32). */
rocke_kernel_def_t* rocke_build_qwen3_qk_norm_rope_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_qwen3_qk_norm_rope_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, const char* arch);

/* qwen3_qk_norm_rope_grid(num_tokens, spec) ->
 *   (ceil(num_tokens * num_heads / block_size), 1, 1). */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_grid(
    int num_tokens, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, int out[3]);

/* Convenience: build + lower to .ll at arch (NULL => "gfx1250"). On ROCKE_OK
 * *out_ll is a malloc'd NUL-terminated string the caller frees with free(). */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_lower_to_llvm(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_QWEN3_QK_NORM_ROPE_H */
