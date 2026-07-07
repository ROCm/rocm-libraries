/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx942_attention_tiled_2d.h -- PUBLIC C99 surface for the gfx942
 * (CDNA3) narrow-atom tiled-2D unified-attention kernel INSTANCE BUILDER
 * (rocke/instances/gfx942/attention_tiled_2d.py, 5287 LOC).
 *
 *   Python (attention_tiled_2d.py)                C99 (this header)
 *   -------------------------------------------   ------------------------------------------------
 *   class UnifiedAttention2DTiledSpec             rocke_attention_tiled_2d_spec_t   (already ported:
 *     (frozen dataclass + __post_init__)            helper_helper_rocke.instances.gfx942.
 *                                                   attention_tiled_2d.h -- reused verbatim here)
 *   spec.@property accessors                       rocke_attention_tiled_2d_spec_* (same helper hdr)
 *   supports_tiled_2d(...)                         rocke_gfx942_attention_tiled_2d_supports
 *   build_unified_attention_2d_tiled(spec, arch)  rocke_build_unified_attention_2d_tiled_scalar
 *   (+ convenience: build -> lower .ll)           rocke_gfx942_attention_tiled_2d_lower_to_llvm
 *
 * SCOPE NOTE / NAMING.
 *   The spec struct, the spec @property accessors, the spec validator
 *   (__post_init__), the config-from-spec derivation, and the two 32x32 C
 *   helpers (_mfma_32x32_c_row/_col) were already ported as VALUE-TYPE / pure
 *   helpers in rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h.
 *   This header REUSES those by include and does NOT redeclare them. The entry
 *   point rocke_build_unified_attention_2d_tiled_scalar keeps the exact name the
 *   porting task fixed (it is the kernel emitter, declared here as the real
 *   build whose ~4000-line body lives in the instance_gfx942_attention_tiled_2d_*
 *   .c translation units).
 *
 *   Every NEW symbol this builder family adds (beyond the already-ported spec
 *   surface) carries the ``rocke_gfx942_attention_tiled_2d_`` prefix so it never
 *   clashes with the common/ instances that share the same simple file/symbol
 *   stems (instances/common/attention_unified*, the fmha_* family, the gfx950
 *   sibling attention_tiled_2d). The internal build-context + phase-function
 *   contract lives in the sibling PRIVATE header
 *   rocke/instance_gfx942_attention_tiled_2d_internal.h; public callers only ever
 *   touch THIS header.
 *
 * THE KERNEL (per CTA = NUM_WARPS wave64s, each warp owns a 16/32-row M slice).
 *   1. AITER binary-search on cu_q to resolve seq_idx; compute the Q-block-local
 *      index; early-return for a padding block.
 *   2. Cooperatively stage Q[BLOCK_M, HD] global -> LDS (zero-fill padding rows /
 *      out-of-range heads), then gather the per-lane Q MFMA A-operand to VGPRs.
 *   3. Loop over KV tiles: resolve physical_block via block_tables; async-DMA K
 *      (double-buffered) and V (single-buffered) cache -> LDS (gfx942 caps the
 *      load-to-LDS DMA at 1 DWORD/lane); compute S = Q @ K^T with the narrow
 *      16x16x16 atom (HD/16 K-iters x T/16 N-tiles); apply qk_scale / optional
 *      softcap / causal+SW+padding mask; run the online softmax (per-row max/sum
 *      via ds_bpermute butterfly over 16-lane groups); publish P to P_lds (or
 *      keep P^T in registers on the transposed-x8 path).
 *   4. PV: acc *= alpha; acc += P @ V via 16x16x16, with the V B-operand built
 *      from strided LDS reads reproducing the MFMA distribution (gfx942 lacks
 *      ds_read_tr16_b64). Optional conflict-free transposed V_lds feeds (cfv /
 *      cfvst) and the gfx942-legal 32x32x8 wide-fragment path are selectable.
 *   5. Normalise acc /= L and store output[BLOCK_M, HD] via the Acc_lds
 *      stripe-cooperative vec8 bridge (or direct per-lane scalar stores on the
 *      transposed-x8 path).
 *
 * ENTRY RETURN. rocke_build_unified_attention_2d_tiled_scalar returns the built
 * kernel (b->kernel == Python ``b.kernel``) on success, or NULL with b's
 * sticky error set. It mirrors the Python build_unified_attention_2d_tiled:
 * require_tiled_attention_arch(arch) -> dtype gate (fp16/bf16) -> select the
 * narrow 16x16x16 atom -> derive the config -> emit. ``arch`` NULL == the Python
 * default "gfx942".
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder; the supports gate returns a bool + reason string;
 * the convenience lower returns a rocke_status_t. Every IR node is arena-owned.
 */
#ifndef ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_H
#define ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_kernel_def_t, rocke_status_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t                                 */
/* The already-ported spec struct + @property accessors + __post_init__ validator
 * + config-from-spec + 32x32 C helpers. REUSED, not redeclared. */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * supports_tiled_2d(...)   (Python lines 887-1096)
 * ============================================================ *
 *
 * The arch / dtype / geometry / LDS-budget admission gate. Faithful port of the
 * keyword-only Python ``supports_tiled_2d``: the arch gate
 * (validate_tiled_attention_arch), the fp8-K/V-cache reject, the
 * dtype/head_size/block_size/num_queries_per_kv checks, the tile_size /
 * per-wave-token DMA constraints, and the per-arch LDS-budget gate (capacity
 * from ArchTarget.from_gfx(arch).lds_capacity_bytes, gfx942 -> 65536). Returns
 * true + writes "supported" into reason on accept; on reject returns false and
 * writes the structured Python reason string. ``reason`` may be NULL (then only
 * the bool is produced). Optional[int]/Optional[str] inputs use a has_<x> flag /
 * a NULL pointer == Python None (q_dtype NULL == None). ``arch`` NULL ==
 * "gfx942". Pure: emits no IR. */
typedef struct rocke_gfx942_attention_tiled_2d_supports_args
{
    int head_size;
    int block_size;
    const char* dtype; /* "fp16" / "bf16" / other (rejected)        */
    int num_queries_per_kv;
    bool use_alibi;
    bool use_qq_bias;
    bool use_fp8;
    const char* q_dtype; /* NULL == Python None                        */
    int num_warps; /* default 1                                  */
    int block_m_per_warp; /* default 16                                 */
    const char* kv_storage_dtype; /* NULL == Python None ("fp8e4m3")            */
    bool has_tile_size; /* Optional[int] None                         */
    int tile_size;
    const char* arch; /* NULL == "gfx942"                           */
    bool use_mfma_32x32x8; /* default False                              */
    bool use_transposed_qk_32x32; /* default False                              */
    bool use_k_single_buffer; /* default False                              */
    bool use_conflict_free_v_store; /* default False                              */
    bool use_k_sliced_ring; /* default False                              */
} rocke_gfx942_attention_tiled_2d_supports_args_t;

/* Defaulted args (num_warps=1, block_m_per_warp=16, everything else 0/NULL/
 * false). The caller sets head_size/block_size/dtype/num_queries_per_kv before
 * use. */
rocke_gfx942_attention_tiled_2d_supports_args_t
    rocke_gfx942_attention_tiled_2d_supports_args_default(void);

bool rocke_gfx942_attention_tiled_2d_supports(
    const rocke_gfx942_attention_tiled_2d_supports_args_t* args, char* reason, size_t reason_cap);

/* ============================================================ *
 * build_unified_attention_2d_tiled(spec, arch="gfx942")   (lines 1104-5287)
 * ============================================================ *
 *
 * Emit the gfx942 narrow-atom tiled MFMA 2D unified-attention kernel into the
 * supplied (already rocke_ir_builder_init'd, named spec.kernel_name()) builder
 * ``b`` and return the kernel (b->kernel) on success, or NULL with b's sticky
 * error set. ``arch`` NULL == the Python default "gfx942".
 *
 * This is the REAL emitter (the porting task pins this name): the ~4000-line
 * body is split across the instance_gfx942_attention_tiled_2d_*.c translation
 * units, driven through the build-context + phase-function contract in
 * rocke/instance_gfx942_attention_tiled_2d_internal.h. It reproduces the Python
 * prologue -> Q-load -> KV-loop (QK / mask / softmax / PV) -> epilogue order
 * byte-faithfully. */
rocke_kernel_def_t* rocke_build_unified_attention_2d_tiled_scalar(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch);

/* Convenience: init ``b`` with spec.kernel_name(), then build. The caller owns
 * ``b`` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL with
 * b's sticky error set. ``arch`` NULL == "gfx942". */
rocke_kernel_def_t* rocke_build_unified_attention_2d_tiled_scalar_new(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch);

/* UnifiedAttention2DTiledSpec.kernel_name() (Python lines 827-885). Writes the
 * NUL-terminated kernel name into ``out`` (capacity ``out_cap``). Returns
 * ROCKE_ERR_VALUE on NULL args or insufficient capacity. */
rocke_status_t rocke_attention_tiled_2d_spec_kernel_name(
    const rocke_attention_tiled_2d_spec_t* spec, char* out, size_t out_cap);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * ``arch`` NULL == "gfx942". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err!=NULL, capacity err_cap) a diagnostic is written. Internally owns
 * and frees its IRBuilder. */
rocke_status_t
    rocke_gfx942_attention_tiled_2d_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX942_ATTENTION_TILED_2D_H */
