/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_qwen3_sampler.h -- C99 port of the gfx1250 greedy
 * (argmax) token sampler rocke/instances/gfx1250/qwen3_sampler.py.
 *
 *   Python (gfx1250/qwen3_sampler.py)       C99 (this header)
 *   -------------------------------------   ----------------------------------------
 *   class Qwen3GreedySamplerSpec            rocke_qwen3_sampler_gfx1250_spec_t
 *   Qwen3GreedySamplerSpec.kernel_name()    rocke_qwen3_sampler_gfx1250_kernel_name()
 *   build_qwen3_greedy_sampler(spec, arch)  rocke_build_qwen3_sampler_gfx1250()
 *   qwen3_greedy_sampler_grid(n, spec)      rocke_qwen3_sampler_gfx1250_grid()
 *
 * out[t] = argmax_v logits[t, v], deterministic lowest-index tie-break (numpy
 * argmax / temperature==0 greedy path). One workgroup per token row: each thread
 * scans a strided vocab slice tracking (max, idx); an LDS index-reduction
 * (block_lds_reduce_with_index, combine=argmax) collapses to the row argmax; lane
 * 0 writes the id. Arch-neutral (reduction + scalar ops, no WMMA).
 */
#ifndef ROCKE_INSTANCE_GFX1250_QWEN3_SAMPLER_H
#define ROCKE_INSTANCE_GFX1250_QWEN3_SAMPLER_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python Qwen3GreedySamplerSpec (frozen dataclass):
 *     logits_dtype: str = "f32"     # f32/fp32/bf16/fp16/f16
 *     block_size: int = 256         # power of two
 *     name: str = "rocke_gfx1250_qwen3_greedy_sampler"
 * __post_init__ validates logits_dtype and power-of-two block_size; in C those
 * checks move into the validity gate / build. */
typedef struct rocke_qwen3_sampler_gfx1250_spec
{
    const char* logits_dtype;
    int block_size;
    const char* name;
} rocke_qwen3_sampler_gfx1250_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_qwen3_sampler_gfx1250_spec_t rocke_qwen3_sampler_gfx1250_spec_default(void);

/* Qwen3GreedySamplerSpec.kernel_name():
 *   kernel_name_join(self.name, logits_dtype, f"bs{block_size}"). */
rocke_status_t rocke_qwen3_sampler_gfx1250_kernel_name(
    const rocke_qwen3_sampler_gfx1250_spec_t* spec, char* out, size_t out_cap);

/* Validity gate (mirrors __post_init__ + _require_supported): logits_dtype in
 * {f32,fp32,bf16,fp16,f16}, block_size a power of two, ArchTarget.from_gfx(arch)
 * resolves. On reject writes `reason` (capacity reason_cap) and returns false. */
bool rocke_qwen3_sampler_gfx1250_is_valid_spec(const rocke_qwen3_sampler_gfx1250_spec_t* spec,
                                               const char* arch,
                                               char* reason,
                                               size_t reason_cap);

/* build_qwen3_greedy_sampler(spec, arch). Builds into the supplied builder `b`
 * (already rocke_ir_builder_init'd with spec.kernel_name()); returns b->kernel
 * or NULL with b's sticky error set. `arch` NULL => "gfx1250".
 *
 * Signature: (logits: ptr<dt>, out_ids: ptr<i32>, vocab: i32). */
rocke_kernel_def_t* rocke_build_qwen3_sampler_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_sampler_gfx1250_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_qwen3_sampler_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_sampler_gfx1250_spec_t* spec, const char* arch);

/* qwen3_greedy_sampler_grid(num_tokens, spec) -> (num_tokens, 1, 1). */
rocke_status_t rocke_qwen3_sampler_gfx1250_grid(int num_tokens,
                                                const rocke_qwen3_sampler_gfx1250_spec_t* spec,
                                                int out[3]);

/* Convenience: build + lower to .ll at arch (NULL => "gfx1250"). On ROCKE_OK
 * *out_ll is a malloc'd NUL-terminated string the caller frees with free(). */
rocke_status_t
    rocke_qwen3_sampler_gfx1250_lower_to_llvm(const rocke_qwen3_sampler_gfx1250_spec_t* spec,
                                              const char* arch,
                                              rocke_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_QWEN3_SAMPLER_H */
