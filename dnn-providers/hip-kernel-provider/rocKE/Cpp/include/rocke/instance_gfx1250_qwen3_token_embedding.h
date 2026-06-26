/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_qwen3_token_embedding.h -- C99 port of the gfx1250
 * token-embedding gather kernel rocke/instances/gfx1250/qwen3_token_embedding.py.
 *
 *   Python (gfx1250/qwen3_token_embedding.py)  C99 (this header)
 *   ----------------------------------------   ------------------------------------
 *   class Qwen3TokenEmbeddingSpec              rocke_qwen3_token_embedding_gfx1250_spec_t
 *   Qwen3TokenEmbeddingSpec.kernel_name()      rocke_qwen3_token_embedding_gfx1250_kernel_name()
 *   build_qwen3_token_embedding(spec, arch)    rocke_build_qwen3_token_embedding_gfx1250()
 *   qwen3_token_embedding_grid(n, spec)        rocke_qwen3_token_embedding_gfx1250_grid()
 *
 * out[t, :] = table[input_ids[t], :] for table [vocab, hidden]. Pure vectorised
 * copy (no compute), arch-neutral (builds on gfx1250 and CDNA wave64). Each
 * thread copies one vec-wide chunk of one token's hidden row.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass has defaults; in C
 * the caller fills the struct. rocke_qwen3_token_embedding_gfx1250_spec_default()
 * returns the Python dataclass defaults.
 */
#ifndef ROCKE_INSTANCE_GFX1250_QWEN3_TOKEN_EMBEDDING_H
#define ROCKE_INSTANCE_GFX1250_QWEN3_TOKEN_EMBEDDING_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python Qwen3TokenEmbeddingSpec (frozen dataclass):
 *     hidden: int = 2048
 *     dtype: str = "bf16"     # fp16/bf16
 *     vec: int = 8            # 1/2/4/8
 *     block_size: int = 256
 *     name: str = "rocke_gfx1250_qwen3_token_embedding"
 * __post_init__ validates dtype/vec/hidden; in C those checks move into the
 * validity gate / build. */
typedef struct rocke_qwen3_token_embedding_gfx1250_spec
{
    int hidden;
    const char* dtype;
    int vec;
    int block_size;
    const char* name;
} rocke_qwen3_token_embedding_gfx1250_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_qwen3_token_embedding_gfx1250_spec_t rocke_qwen3_token_embedding_gfx1250_spec_default(void);

/* Qwen3TokenEmbeddingSpec.kernel_name():
 *   kernel_name_join(self.name, f"h{hidden}", dtype, f"v{vec}"). */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_kernel_name(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, char* out, size_t out_cap);

/* Validity gate (mirrors __post_init__ + _require_supported): dtype fp16/bf16,
 * vec in {1,2,4,8}, hidden > 0 and a multiple of vec, ArchTarget.from_gfx(arch)
 * resolves. On reject writes `reason` (capacity reason_cap) and returns false. */
bool rocke_qwen3_token_embedding_gfx1250_is_valid_spec(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* build_qwen3_token_embedding(spec, arch). Builds into the supplied builder `b`
 * (already rocke_ir_builder_init'd with spec.kernel_name()); returns b->kernel
 * or NULL with b's sticky error set. `arch` NULL => "gfx1250".
 *
 * Signature: (input_ids: ptr<i32>, table: ptr<dt>, out: ptr<dt>,
 *             num_tokens: i32). */
rocke_kernel_def_t* rocke_build_qwen3_token_embedding_gfx1250(
    rocke_ir_builder_t* b,
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
    const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_qwen3_token_embedding_gfx1250_new(
    rocke_ir_builder_t* b,
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
    const char* arch);

/* qwen3_token_embedding_grid(num_tokens, spec) ->
 *   (ceil(num_tokens * (hidden/vec) / block_size), 1, 1). */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_grid(
    int num_tokens, const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, int out[3]);

/* Convenience: build + lower to .ll at arch (NULL => "gfx1250"). On ROCKE_OK
 * *out_ll is a malloc'd NUL-terminated string the caller frees with free(). */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_lower_to_llvm(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_QWEN3_TOKEN_EMBEDDING_H */
