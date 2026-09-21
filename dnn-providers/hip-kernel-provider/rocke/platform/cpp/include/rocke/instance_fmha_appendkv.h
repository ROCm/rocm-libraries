/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_appendkv.h -- C99 port of the append-KV kernel instance
 * builder rocke/instances/common/fmha_appendkv.py.
 *
 * appendkv writes a new K / V token (or block of tokens) into a pre-allocated
 * KV cache at the position implied by the per-sequence seqlen_kv array, with an
 * optional rotary embedding fused on K. It is a pure vectorised KV-cache
 * scatter: it issues NO MFMA atoms and allocates NO LDS, so the f16 / bf16
 * vector load / store path is byte-identical on gfx942 and gfx950 (`arch` is
 * threaded only through is_valid_spec's per-WG thread-cap check).
 *
 *   Python (fmha_appendkv.py)            C99 (this header)
 *   -----------------------------------  --------------------------------------
 *   FmhaAppendKvSpec (frozen dataclass)  rocke_fmha_appendkv_spec_t
 *     .kernel_name()                     rocke_fmha_appendkv_kernel_name(...)
 *   is_valid_spec(spec, arch)            rocke_fmha_appendkv_is_valid_spec(...)
 *   build_fmha_fwd_appendkv(spec, arch)  rocke_build_fmha_fwd_appendkv(...)
 *   fmha_appendkv_grid(spec, total_new_q)rocke_fmha_appendkv_grid(...)
 *   fmha_appendkv_signature(spec)        rocke_fmha_appendkv_signature(...)
 *   (+ convenience: build -> lower .ll)  rocke_fmha_appendkv_lower_to_llvm(...)
 *
 * The spec reuses the shared FMHA value types:
 *   - rocke_fmha_common_spec_t  (helper_rocke.instances.common._fmha_common.h)
 *   - rocke_rotary_spec_t       (helper_rocke.helpers.rotary.h)
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_APPENDKV_H
#define ROCKE_INSTANCE_FMHA_APPENDKV_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.rotary.h" /* rocke_rotary_spec_t  */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t    */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* common spec        */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaAppendKvSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaAppendKvSpec:
 *     common: FmhaCommonSpec
 *     batch: int
 *     rotary: RotarySpec | None = None
 *     block_size: int = 256
 *     name: str = "rocke_fmha_appendkv"
 *
 * The Optional[RotarySpec] is modelled as a has_rotary flag + an embedded
 * value (Python `rotary is None` <=> has_rotary == false). `name` is referenced
 * as-is (not copied); keep it alive for the spec's use (NULL => the dataclass
 * default "rocke_fmha_appendkv"). */
typedef struct rocke_fmha_appendkv_spec
{
    rocke_fmha_common_spec_t common;
    int batch;

    bool has_rotary; /* false == Python `rotary is None`            */
    rocke_rotary_spec_t rotary; /* valid only when has_rotary is true          */

    int block_size; /* default 256                                 */
    const char* name; /* NULL => "rocke_fmha_appendkv"              */
} rocke_fmha_appendkv_spec_t;

/* FmhaAppendKvSpec(common, batch, rotary=None, block_size=256,
 * name="rocke_fmha_appendkv"): construct with the dataclass defaults
 * (has_rotary=false, block_size=256, name=NULL). */
rocke_fmha_appendkv_spec_t rocke_fmha_appendkv_spec_default(rocke_fmha_common_spec_t common,
                                                            int batch);

/* FmhaAppendKvSpec.name accessor with the dataclass default folded in. */
const char* rocke_fmha_appendkv_spec_name(const rocke_fmha_appendkv_spec_t* spec);

/* FmhaAppendKvSpec.kernel_name():
 *   kernel_name_join(name, f"H{H}", f"HK{num_kv_heads}", dtype, f"B{batch}",
 *                    "rope" if rotary else "norope", f"b{block_size}")
 *
 * Writes the NUL-terminated kernel name into `out` (capacity out_cap). Returns
 * ROCKE_OK, or ROCKE_ERR_VALUE on NULL args / too-small buffer. */
rocke_status_t rocke_fmha_appendkv_kernel_name(const rocke_fmha_appendkv_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec(spec, arch) -> (ok, reason)
 * ------------------------------------------------------------------ *
 *
 * Python (fmha_appendkv.py:is_valid_spec):
 *   - ArchTarget.from_gfx(arch); KeyError -> (False, str(e))
 *   - validate_common_spec(spec.common)
 *   - batch > 0
 *   - rotary is None or rotary.head_size == common.shape.head_size
 *   - block_size <= target.max_threads_per_block
 *
 * `arch` NULL => "gfx950". On reject, `reason` (if non-NULL, capacity
 * reason_cap) receives the structured message and false is returned; on accept
 * returns true and writes "ok". */
bool rocke_fmha_appendkv_is_valid_spec(const rocke_fmha_appendkv_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_appendkv(spec, arch)
 * ------------------------------------------------------------------ *
 *
 * Builds the IR into the supplied builder `b` (already rocke_ir_builder_init'd
 * with spec.kernel_name()), reproducing the Python build's op order + attrs
 * byte-faithfully, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". This routine does NOT re-init
 * the builder (the caller controls its lifetime). */
rocke_kernel_def_t* rocke_build_fmha_fwd_appendkv(rocke_ir_builder_t* b,
                                                  const rocke_fmha_appendkv_spec_t* spec,
                                                  const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_fmha_fwd_appendkv_new(rocke_ir_builder_t* b,
                                                      const rocke_fmha_appendkv_spec_t* spec,
                                                      const char* arch);

/* ------------------------------------------------------------------ *
 * fmha_appendkv_grid(spec, total_new_q)
 * ------------------------------------------------------------------ *
 *
 * Python:
 *   gx, _, _ = ceil_div_grid((total_new_q, spec.block_size))
 *   return (gx, spec.common.shape.num_kv_heads, 1)
 *
 * On success writes out[0..2] = (gx, num_kv_heads, 1) and returns ROCKE_OK; on
 * the Python ValueError (non-positive block_size tile) returns ROCKE_ERR_VALUE. */
rocke_status_t
    rocke_fmha_appendkv_grid(const rocke_fmha_appendkv_spec_t* spec, int total_new_q, int out[3]);

/* ------------------------------------------------------------------ *
 * fmha_appendkv_signature(spec)
 * ------------------------------------------------------------------ *
 *
 * Python:
 *   SignatureBuilder()
 *     .ptr("K_new", dtype).ptr("V_new", dtype)
 *     .ptr("K_cache", dtype).ptr("V_cache", dtype)
 *     .ptr("seqlen_kv","i32").ptr("cu_seqlens_new","i32")
 *     [+ .ptr("cos_table","f32").ptr("sin_table","f32") if rotary]
 *     .scalar("total_new_q","i32").scalar("batch","i32")
 *     .scalar("stride_in_token","i32").scalar("stride_in_head","i32")
 *     .scalar("stride_cache_token","i32").scalar("stride_cache_head","i32")
 *     .build()
 *
 * Writes the manifest entries into out_items[] (capacity out_cap; need 12, or
 * 14 with rotary) and sets *out_count. The entry strings live in `arena` (pass
 * a live arena). Returns ROCKE_OK, or ROCKE_ERR_VALUE on NULL args / too-small
 * out_cap. */
rocke_status_t rocke_fmha_appendkv_signature(rocke_arena_t* arena,
                                             const rocke_fmha_appendkv_spec_t* spec,
                                             rocke_sig_entry_t* out_items,
                                             size_t out_cap,
                                             size_t* out_count);

/* ------------------------------------------------------------------ *
 * Convenience: build + lower to LLVM .ll text.
 * ------------------------------------------------------------------ *
 *
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_fmha_appendkv_lower_to_llvm(const rocke_fmha_appendkv_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_APPENDKV_H */
