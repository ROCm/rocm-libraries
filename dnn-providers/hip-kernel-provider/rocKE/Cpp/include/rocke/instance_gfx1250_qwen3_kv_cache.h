/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_qwen3_kv_cache.h -- C99 port of the gfx1250 Qwen3 KV-cache
 * kernels rocke/instances/gfx1250/qwen3_kv_cache.py.
 *
 *   Python (gfx1250/qwen3_kv_cache.py)        C99 (this header)
 *   ---------------------------------------   -------------------------------------------
 *   class Qwen3KvDequantSpec                  rocke_qwen3_kv_dequant_gfx1250_spec_t
 *   build_qwen3_kv_dequant_smoke(spec, arch)  rocke_build_qwen3_kv_dequant_smoke_gfx1250()
 *   class Qwen3KvAppendRopeSpec               rocke_qwen3_kv_append_rope_gfx1250_spec_t
 *   build_qwen3_kv_append_rope(spec, arch)    rocke_build_qwen3_kv_append_rope_gfx1250()
 *
 * Two KV-cache-side kernels (not part of the attention dispatcher):
 *  - kv_dequant_smoke: fp8e4m3/bf8e5m2 KV read + in-register dequant smoke.
 *  - kv_append_rope: KV append/update with optional RoPE + bf16/fp8e4m3/bf8e5m2
 *    quantized store into the paged cache.
 *
 * gfx1250-ONLY (the validity gate requires arch=="gfx1250", CDNA wave32).
 */
#ifndef ROCKE_INSTANCE_GFX1250_QWEN3_KV_CACHE_H
#define ROCKE_INSTANCE_GFX1250_QWEN3_KV_CACHE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Qwen3KvDequantSpec ----
 *     kv_storage_dtype: str        (required: "fp8e4m3" or "bf8e5m2")
 *     output_dtype: str = "bf16"   # fp16/bf16
 *     head_dim: int = 64           (must be 64)
 *     name: str = "rocke_gfx1250_qwen3_kv_dequant" */
typedef struct rocke_qwen3_kv_dequant_gfx1250_spec
{
    const char* kv_storage_dtype;
    const char* output_dtype;
    int head_dim;
    const char* name;
} rocke_qwen3_kv_dequant_gfx1250_spec_t;

/* Default-constructed spec. kv_storage_dtype has NO Python default; caller MUST
 * set it (default is NULL, which the validity gate rejects). */
rocke_qwen3_kv_dequant_gfx1250_spec_t rocke_qwen3_kv_dequant_gfx1250_spec_default(void);

/* Qwen3KvDequantSpec.kernel_name():
 *   kernel_name_join(self.name, f"d{head_dim}", kv_storage_dtype, output_dtype). */
rocke_status_t rocke_qwen3_kv_dequant_gfx1250_kernel_name(
    const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, char* out, size_t out_cap);

/* Validity gate (mirrors __post_init__ + _require_gfx1250). */
bool rocke_qwen3_kv_dequant_gfx1250_is_valid_spec(const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap);

/* build_qwen3_kv_dequant_smoke(spec, arch). Signature:
 *   (src: ptr<storage>, dst: ptr<out>, scale: f32). */
rocke_kernel_def_t* rocke_build_qwen3_kv_dequant_smoke_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_qwen3_kv_dequant_smoke_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, const char* arch);
rocke_status_t
    rocke_qwen3_kv_dequant_gfx1250_lower_to_llvm(const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

/* ---- Qwen3KvAppendRopeSpec ----
 *     input_dtype: str = "bf16"        # fp16/bf16
 *     kv_storage_dtype: str = "bf16"   # bf16/fp8e4m3/bf8e5m2
 *     head_dim: int = 64               (must be 64)
 *     block_size: int = 16             (must be 16)
 *     num_kv_heads: int = 4
 *     use_rope: bool = True
 *     name: str = "rocke_gfx1250_qwen3_kv_append_rope" */
typedef struct rocke_qwen3_kv_append_rope_gfx1250_spec
{
    const char* input_dtype;
    const char* kv_storage_dtype;
    int head_dim;
    int block_size;
    int num_kv_heads;
    bool use_rope;
    const char* name;
} rocke_qwen3_kv_append_rope_gfx1250_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_qwen3_kv_append_rope_gfx1250_spec_t rocke_qwen3_kv_append_rope_gfx1250_spec_default(void);

/* Qwen3KvAppendRopeSpec.kernel_name():
 *   kernel_name_join(self.name, f"d{head_dim}", f"b{block_size}", f"kvh{num_kv_heads}",
 *                    input_dtype, f"kv{kv_storage_dtype}", "rope" if use_rope else ""). */
rocke_status_t rocke_qwen3_kv_append_rope_gfx1250_kernel_name(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, char* out, size_t out_cap);

bool rocke_qwen3_kv_append_rope_gfx1250_is_valid_spec(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap);

/* build_qwen3_kv_append_rope(spec, arch). Signature:
 *   (key_in: ptr<in>, value_in: ptr<in>, k_cache: ptr<storage>, v_cache: ptr<storage>,
 *    block_tables: ptr<i32>, slot_ids: ptr<i32>, cos: ptr<f32>, sin: ptr<f32>,
 *    k_scale: f32, v_scale: f32). */
rocke_kernel_def_t* rocke_build_qwen3_kv_append_rope_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_qwen3_kv_append_rope_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, const char* arch);
rocke_status_t rocke_qwen3_kv_append_rope_gfx1250_lower_to_llvm(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_QWEN3_KV_CACHE_H */
