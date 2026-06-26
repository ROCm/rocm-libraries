/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx942_attention_tiled_3d.h -- PUBLIC C99 port surface for the
 * gfx942 (CDNA3) narrow-atom tiled split-KV 3D attention kernels
 * (rocke/instances/gfx942/attention_tiled_3d.py, 1133 LOC).
 *
 * SCOPE. This header is the PUBLIC entry/glue surface for the TWO kernel
 * builders the Python module exposes:
 *
 *   Python (attention_tiled_3d.py)                C99 (this header)
 *   ------------------------------------------    -------------------------------------
 *   @dataclass UnifiedAttention3DTiledSpec        rocke_unified_attention_3d_tiled_spec_t
 *     (the dataclass + __post_init__ + @property)   + rocke_..._spec_default()
 *                                                   + rocke_..._spec_validate()
 *                                                   + the @property accessors
 *   @dataclass UnifiedAttentionReduceTiledSpec    rocke_unified_attention_reduce_tiled_spec_t
 *   supports_tiled_3d(...)                        rocke_gfx942_attention_tiled_3d_supports
 *   build_unified_attention_3d_tiled(spec, arch)  rocke_build_unified_attention_3d_tiled_gfx942
 *   build_unified_attention_reduce_tiled(...)     rocke_build_unified_attention_reduce_tiled_gfx942
 *   (build -> lower .ll convenience, NEW)         rocke_..._lower_to_llvm
 *
 * The gfx942_attention_tiled_3d prefix is carried on EVERY symbol/macro to avoid
 * clashing with the same-named common/ and gfx950 ports (the gfx950 sibling
 * exports the unprefixed rocke_build_unified_attention_3d_tiled name).
 *
 * SEGMENT KERNEL (build_unified_attention_3d_tiled, lines 234-969). 64 threads /
 * single wave; per-segment tiles over [tile_start, tile_end) with double-
 * buffered K/V LDS. gfx942-specific narrow MFMA 16x16x16 atoms (no wide-K atom):
 * QK with K-step 16 + <4 x dtype> operands; PV with strided-V B-operand
 * reconstruction (no ds_read_*_tr_*). Async global->LDS DMA caps at 1 DWORD/lane
 * on gfx942 (vs gfx950's 4), so 4x more async_buffer_load_lds calls per tile.
 * fp8 K/V dequant via cvt_pk_f32_fp8x4 (dequant_fp8x8_to_dtype). Online-softmax
 * with a 4-reg-per-lane accumulator decoded via _mfma_16x16_c_row.
 *
 * REDUCE KERNEL (build_unified_attention_reduce_tiled, lines 1002-1133). Arch-
 * neutral, pure f32 load / exp2 / store (byte-for-byte port of the gfx950
 * reduce): 64 threads, single active per (q_token, q_head), three passes (max,
 * combine, normalize).
 *
 * REUSED HELPER PORTS (no IR re-emitted here):
 *   rocke/helper_helper_rocke.helpers.attention.h   -- apply_softcap_log2,
 *       binary_search_seq_idx, mfma_16x16x16_for_dtype, wave64_reduce_max,
 *       wave64_reduce_sum (the five "new helpers" named by the port map; ALREADY
 *       present)
 *   rocke/helper_rocke.helpers.attention.h          -- warp_xor_reduce_sum,
 *       (warp_xor_reduce_max via the same module), masks, safe_inv_l
 *   rocke/helper_rocke.helpers.atoms.h              -- MfmaAtom.f16_16x16x16,
 *       make_c_warp_dstr_encoding
 *   rocke/helper_rocke.helpers.distribution.h       -- make_static_tile_distribution,
 *       calculate_x (the _C16_DIST C-warp accumulator decode)
 *   rocke/helper_rocke.helpers.transforms.h         -- TensorDescriptor naive/
 *       transform, embed/indirect/unmerge (paged-KV + segm + Q descriptors)
 *   rocke/helper_rocke.instances.common.fmha_arch.h -- validate/require tiled
 *       attention arch gate (validate_tiled_attention_arch)
 *
 * ERROR MODEL. Mirrors the rest of the C port: pure accessors return a sentinel;
 * the build / validate variants route the first Python ValueError /
 * NotImplementedError through the sticky-error IRBuilder (a dead builder is a
 * NULL no-op). The lower convenience returns a rocke_status_t and writes a
 * diagnostic into the caller's err buffer.
 *
 * LIFETIME. Every emitted IR node is arena-owned (rocke_ir_builder_t.arena);
 * nothing is freed individually. The spec structs are plain by-value PODs the
 * caller owns; strings (dtype / kv_storage_dtype) are caller-owned long-lived
 * literals, not copied. The lower convenience owns/frees its IRBuilder
 * internally and mallocs *out_ll for the caller to free().
 */
#ifndef ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H
#define ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h"
#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_value_t, rocke_kernel_def_t, rocke_type_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t, rocke_status_t                              */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------- module consts *
 * MFMA_M / MFMA_N (module-level in the Python). */
#define ROCKE_GFX942_ATTN_TILED_3D_MFMA_M 16
#define ROCKE_GFX942_ATTN_TILED_3D_MFMA_N 16
/* gfx942 narrow atom: K-step is 16 for both QK and PV (no wide-K atom). */
#define ROCKE_GFX942_ATTN_TILED_3D_QK_K_STEP 16
#define ROCKE_GFX942_ATTN_TILED_3D_PV_K_STEP 16
/* CDNA3 async global->LDS DMA caps at 1 DWORD (4 bytes = 2 halves) per lane. */
#define ROCKE_GFX942_ATTN_TILED_3D_ASYNC_LDS_DWORDS 1
/* THREADS: single wave. */
#define ROCKE_GFX942_ATTN_TILED_3D_THREADS 64
/* rcp(ln2) folded into the score (exp2 domain). */
#define ROCKE_GFX942_ATTN_TILED_3D_RCP_LN2 1.4426950408889634

/* ============================================================ *
 * Coverage gate -- supports_tiled_3d(...)
 * ============================================================ *
 *
 * Faithful port of the module-level supports_tiled_3d host predicate. Returns
 * (ok). On false, *out_reason (if non-NULL) points to a static string mirroring
 * the Python reason; on true it points to "supported". Pure, emits no IR.
 *
 * q_dtype is an Optional[str]: pass NULL for Python None. kv_storage_dtype is
 * Optional[str] ("fp8e4m3" or NULL). arch NULL == the Python default "gfx942". */
bool rocke_gfx942_attention_tiled_3d_supports(int head_size,
                                              int block_size,
                                              const char* dtype,
                                              int num_queries_per_kv,
                                              bool use_alibi,
                                              bool use_qq_bias,
                                              bool use_fp8,
                                              const char* q_dtype,
                                              const char* kv_storage_dtype,
                                              const char* arch,
                                              const char** out_reason);

/* ============================================================ *
 * Segment-kernel spec -- UnifiedAttention3DTiledSpec
 * ============================================================ *
 *
 * Faithful by-value mirror of the frozen dataclass. The first block holds the
 * required (no-default) fields; the rest carry the Python field defaults
 * (materialised by rocke_unified_attention_3d_tiled_spec_default).
 *
 * Optional[int] fields are (has_<x>, <x>); has_<x>==false encodes Python None.
 * Optional[str] kv_storage_dtype is a const char* (NULL == Python None,
 * "fp8e4m3" otherwise). dtype is "fp16" or "bf16". Strings are caller-owned. */
typedef struct rocke_unified_attention_3d_tiled_spec
{
    /* ---- required (no Python default) ---- */
    int head_size; /* 64 / 128 / 256                                   */
    int block_size; /* 16 / 32 / 64                                     */
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16"                                 */
    bool use_sinks;
    int sliding_window; /* 0 == disabled                                    */
    bool has_softcap;
    int num_segments;

    /* ---- defaulted ---- */
    bool use_alibi; /* False */
    bool use_qq_bias; /* False */
    int num_seqs; /* 0 */

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;

    const char* kv_storage_dtype; /* Optional[str] None ("fp8e4m3") */

    bool has_tile_size_override; /* Optional[int] None */
    int tile_size_override;

    bool use_invariant_hoist; /* False */
    bool use_wide_kv_load; /* False */
    /* 64-bit paged-KV addressing for caches > 2 GiB (gfx950 segment kernel folds
     * the per-block byte base into a 64-bit buffer base). Kept byte-identical
     * with the gfx950 header (the two share this struct). */
    bool use_i64_kv_addr; /* False */
} rocke_unified_attention_3d_tiled_spec_t;

/* Materialise every defaulted field. Required fields are zero/NULL-init; the
 * caller must set them before use. */
rocke_unified_attention_3d_tiled_spec_t rocke_unified_attention_3d_tiled_spec_default(void);

/* __post_init__ validator (builder variant). Reproduces the Python check
 * (kv_storage_dtype must be None or "fp8e4m3"); on failure records the Python
 * ValueError text on b (ROCKE_ERR_VALUE) and returns false. Dead/NULL b is a
 * false no-op. */
bool rocke_unified_attention_3d_tiled_spec_validate(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* s);

/* ------------------------------------------------ derived @property accessors *
 * Pure, IR-free; faithful ports of the dataclass @property bodies. */
/* num_queries_per_kv = num_query_heads // num_kv_heads */
int rocke_unified_attention_3d_tiled_spec_num_queries_per_kv(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* block_m = 16 (constant) */
int rocke_unified_attention_3d_tiled_spec_block_m(const rocke_unified_attention_3d_tiled_spec_t* s);
/* block_q = block_m // num_queries_per_kv */
int rocke_unified_attention_3d_tiled_spec_block_q(const rocke_unified_attention_3d_tiled_spec_t* s);
/* tile_size = tile_size_override if has_tile_size_override else block_size */
int rocke_unified_attention_3d_tiled_spec_tile_size(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* dtype_ir: F16 for "fp16", BF16 otherwise (returns a rocke_type_t*). */
const rocke_type_t* rocke_unified_attention_3d_tiled_spec_dtype_ir(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* binary_search_iters: 32 if num_seqs<=0, else max(1, ceil(log2(num_seqs+1))) */
int rocke_unified_attention_3d_tiled_spec_binary_search_iters(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* kernel_name(): kernel_name_join(...) into caller's buf (NUL-terminated, up to
 * cap bytes incl. NUL). Returns the length that would be written (snprintf
 * convention), or -1 on a dead builder/encode error. */
int rocke_unified_attention_3d_tiled_spec_kernel_name(
    const rocke_unified_attention_3d_tiled_spec_t* s, char* buf, size_t cap);

/* ============================================================ *
 * Reduce-kernel spec -- UnifiedAttentionReduceTiledSpec
 * ============================================================ */
typedef struct rocke_unified_attention_reduce_tiled_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    const char* dtype; /* "fp16" or "bf16" */
    int num_segments;

    bool has_waves_per_eu; /* Optional[int] None */
    int waves_per_eu;
} rocke_unified_attention_reduce_tiled_spec_t;

rocke_unified_attention_reduce_tiled_spec_t rocke_unified_attention_reduce_tiled_spec_default(void);

/* dtype_ir: F16 for "fp16", BF16 otherwise. */
const rocke_type_t* rocke_unified_attention_reduce_tiled_spec_dtype_ir(
    const rocke_unified_attention_reduce_tiled_spec_t* s);
/* kernel_name(): see segment-kernel variant for the snprintf convention. */
int rocke_unified_attention_reduce_tiled_spec_kernel_name(
    const rocke_unified_attention_reduce_tiled_spec_t* s, char* buf, size_t cap);

/* ============================================================ *
 * Build entries (mirror the Python build_* functions)
 * ============================================================ *
 *
 * Each owns/inits the supplied builder b via the resolved kernel_name and emits
 * the kernel. arch NULL == the Python default "gfx942"; a non-gfx942 arch hits
 * the Python require_tiled_attention_arch gate -> sticky error + NULL. The
 * caller owns b and frees it with rocke_ir_builder_free(). Returns the kernel
 * (== b->kernel) or NULL on a sticky error. */

/* Segment kernel (narrow 16x16x16 MFMA). Call with a spec validated by
 * rocke_unified_attention_3d_tiled_spec_validate. */
rocke_kernel_def_t* rocke_build_unified_attention_3d_tiled_gfx942(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* spec, const char* arch);

/* Reduce kernel (arch-neutral pure-f32 combine). */
rocke_kernel_def_t* rocke_build_unified_attention_reduce_tiled_gfx942(
    rocke_ir_builder_t* b,
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    const char* arch);

/* ============================================================ *
 * Build -> lower-to-.ll convenience (NEW; not in the Python)
 * ============================================================ *
 *
 * Build into an internally-owned builder, then lower to LLVM .ll text. On ROCKE_OK
 * *out_ll is a malloc'd NUL-terminated string the caller frees with free(); on
 * failure *out_ll is left NULL and (if err != NULL, capacity err_cap) a
 * diagnostic is written. */
rocke_status_t rocke_build_unified_attention_3d_tiled_gfx942_lower_to_llvm(
    const rocke_unified_attention_3d_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

rocke_status_t rocke_build_unified_attention_reduce_tiled_gfx942_lower_to_llvm(
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H */
