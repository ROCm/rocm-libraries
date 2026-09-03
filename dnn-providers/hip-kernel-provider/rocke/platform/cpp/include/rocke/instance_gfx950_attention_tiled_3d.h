/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx950_attention_tiled_3d.h -- PUBLIC C99 port surface for the
 * gfx950 (CDNA4) WIDE-K tiled split-KV 3D attention kernels
 * (rocke/instances/gfx950/attention_tiled_3d.py, 1164 LOC).
 *
 * SCOPE. This header is the PUBLIC entry/glue surface for the TWO kernel
 * builders the Python module exposes:
 *
 *   Python (gfx950/attention_tiled_3d.py)         C99 (this header)
 *   ------------------------------------------    -------------------------------------
 *   @dataclass UnifiedAttention3DTiledSpec        rocke_unified_attention_3d_tiled_spec_t
 *     (the dataclass + __post_init__ + @property)   + rocke_..._spec_default()
 *                                                   + rocke_..._spec_validate()
 *                                                   + the @property accessors
 *   @dataclass UnifiedAttentionReduceTiledSpec    rocke_unified_attention_reduce_tiled_spec_t
 *   supports_tiled_3d(...)                        rocke_gfx950_attention_tiled_3d_supports
 *   build_unified_attention_3d_tiled(spec, arch)  rocke_build_unified_attention_3d_tiled_gfx950
 *   build_unified_attention_reduce_tiled(...)     rocke_build_unified_attention_reduce_tiled_gfx950
 *   (build -> lower .ll convenience, NEW)         rocke_..._lower_to_llvm
 *
 * The gfx950_attention_tiled_3d prefix is carried on EVERY symbol/macro to avoid
 * clashing with the same-named common/ and gfx942 ports (the gfx942 sibling
 * exports rocke_build_unified_attention_3d_tiled_gfx942; the shared common/
 * attention_unified files carry their own scalars). The spec STRUCT TAGS
 * (rocke_unified_attention_3d_tiled_spec_t / rocke_unified_attention_reduce_tiled_spec_t)
 * are the SAME PODs the gfx942 3D header declares: the two arch ports share the
 * identical frozen dataclass shape, so this header guards its struct/enum/default
 * definitions behind the gfx942 header's include-guard tokens (see the
 * "SHARED SPEC POD" note below) and only ever DECLARES the gfx950 build/lower
 * entries. Including BOTH the gfx942 and gfx950 3D headers in one TU is legal.
 *
 * GFX950 VS GFX942 DIFFERENCE (the only structural divergence; everything else
 * is a byte-faithful copy of the gfx942 3D shape):
 *   - WIDE-K MFMA: QK uses the 16x16x32 atom (K-step 32); PV uses 16x16x32 when
 *     T % 32 == 0, else the 16x16x16 atom (K-step 16). gfx942 is narrow-only
 *     (16x16x16, K-step 16 everywhere).
 *   - LDS TRANSPOSE READS: PV reads the V B-operand via ds_read_tr16_b64 driven
 *     by a TransposeLdsReader(K=PV_K_STEP, M=16) (helpers/layouts.py). gfx942
 *     reconstructs the B-operand from strided V_lds loads (_strided_v_b_operand).
 *   - ASYNC DMA: gfx950 async global->LDS DMA delivers 4 DWORDS (8 halves) per
 *     lane per call; there is NO wide-b128 sync feed path and NO invariant-hoist
 *     opt (both gfx942-narrow-only levers; the gfx950 spec accepts the knobs for
 *     signature parity but ignores them).
 *   - fp8 K/V dequant: the SAME UNFUSED cvt_pk_f32_fp8x4 + explicit fmul scale
 *     (dequant_fp8x8_to_dtype, the one NEW helper this kernel threads). The
 *     unfused fmul is a CRITICAL INVARIANT: the fused v_cvt_scalef32_pk_f32_fp8
 *     would truncate non-power-of-two per-tensor scales to E8M0.
 *
 * SEGMENT KERNEL (build_unified_attention_3d_tiled, lines 253-950). 64 threads /
 * single wave; per-segment tiles over [tile_start, tile_end) with double-buffered
 * K/V LDS. Flow (AITER order): segm_* workspace ptrs declared FIRST, then
 * K/V/cu/seq_lens; block/thread indexing + cu_q binary search; segment-bounds
 * computation; an early-out neutral-fill block (m=-inf, acc=0, l=0) when the
 * segment is past seq_len; LDS alloc + Q load; per-segment online-softmax scf.for
 * (KV async loads / QK wide MFMA / softmax / PV wide-or-narrow MFMA via ds_read_tr);
 * then a guarded segment-workspace epilogue (per-lane m/l + per-thread acc).
 *
 * REDUCE KERNEL (build_unified_attention_reduce_tiled, lines 985-1164). Arch-
 * neutral, pure f32 load / exp2 / store (byte-for-byte the same as the gfx942
 * reduce): 64 threads, single active per (q_token, q_head), three passes
 * (SEG_PER_LANE-strided partial max; per-lane expsum + factor cache in LDS;
 * per-dim acc reduce + normalize + fp16/bf16 cast + write).
 *
 * REUSED HELPER PORTS (no IR re-emitted here):
 *   rocke/helper_helper_rocke.helpers.attention.h   -- apply_softcap_log2,
 *       binary_search_seq_idx, wave64_reduce_max, wave64_reduce_sum
 *   rocke/helper_rocke.helpers.attention.h          -- warp_xor_reduce_sum,
 *       dequant_fp8x8_to_dtype (the ONE new helper this kernel threads), masks,
 *       safe_inv_l
 *   rocke/helper_rocke.helpers.layouts.h            -- TransposeLdsReader (the
 *       gfx950-only PV transpose-read lane formulas)
 *   rocke/helper_rocke.helpers.atoms.h              -- mfma_atom catalog
 *       (f16/bf16 16x16x32 wide-K + 16x16x16), make_c_warp_dstr_encoding
 *   rocke/helper_rocke.helpers.distribution.h       -- make_static_tile_distribution,
 *       calculate_x (the _C16_DIST C-warp accumulator decode; the C layout is
 *       dtype-independent so the f16 16x16x16 atom drives it for fp16 and bf16)
 *   rocke/helper_rocke.helpers.transforms.h         -- TensorDescriptor naive/
 *       transform, indirect/unmerge (paged-KV + segm + Q descriptors)
 *   ir.h builder ops                               -- ds_read_tr16_b64,
 *       cvt_pk_f32_fp8x4, async_buffer_load_lds_addr, warp_shuffle_xor,
 *       buffer_rsrc, smem_* (the gfx950 wide-K / transpose-read intrinsics)
 *
 * NEW-HELPER NOTE (port map). The map names a single new helper for this
 * kernel, rocke.helpers.attention.dequant_fp8x8_to_dtype, ALREADY PORTED as
 * rocke_dequant_fp8x8_to_dtype in rocke/helper_rocke.helpers.attention.h. The
 * gfx950-only wide-K MFMA / ds_read_tr ops are NOT helpers; they are direct
 * IRBuilder methods (ir.h). The wide-K mfma_16x16x32_for_dtype / narrow
 * mfma_16x16x16_for_dtype / warp_xor_reduce_max dispatchers are the SAME
 * rocke.helpers.attention symbols the gfx942 sibling consumes; they are wired
 * through the internal header (see instance_gfx950_attention_tiled_3d_internal.h
 * "wide-K MFMA dispatch" note).
 *
 * ARCH GATE. supports_tiled_3d / build both route through
 * validate_tiled_attention_arch / require_tiled_attention_arch
 * (rocke/instances/common/attention_arch.py): the tiled 3D kernel needs the
 * gfx950 wide-K 16x16x32 atom + ds_read_tr LDS transpose reads. A non-gfx950
 * target is rejected with the mirrored reason string (sticky error + NULL on
 * the build path; (false, reason) on the supports path).
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
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_TILED_3D_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_TILED_3D_H

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
#define ROCKE_GFX950_ATTN_TILED_3D_MFMA_M 16
#define ROCKE_GFX950_ATTN_TILED_3D_MFMA_N 16
/* gfx950 wide atom: QK K-step is 32. PV K-step is 32 when T%32==0 else 16; this
 * is a per-spec runtime choice (see ..._config.PV_K_STEP), so only the QK value
 * is a compile-time macro. */
#define ROCKE_GFX950_ATTN_TILED_3D_QK_K_STEP 32
/* CDNA4 async global->LDS DMA delivers 4 DWORDS (8 halves = 16 bytes) per lane. */
#define ROCKE_GFX950_ATTN_TILED_3D_ASYNC_LDS_DWORDS 4
/* THREADS: single wave. */
#define ROCKE_GFX950_ATTN_TILED_3D_THREADS 64
/* rcp(ln2) folded into the score (exp2 domain). */
#define ROCKE_GFX950_ATTN_TILED_3D_RCP_LN2 1.4426950408889634

/* ====================================================================== *
 * SHARED SPEC POD.
 *
 * UnifiedAttention3DTiledSpec / UnifiedAttentionReduceTiledSpec are the
 * IDENTICAL frozen dataclasses the gfx942 3D port already mirrors as
 * rocke_unified_attention_3d_tiled_spec_t / rocke_unified_attention_reduce_tiled_spec_t.
 * To allow including BOTH 3D headers in one TU, the struct/enum/default
 * declarations below are skipped when the gfx942 3D header has already provided
 * them (it defines ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H). When this header is
 * used standalone (gfx950 only), it provides the canonical declarations itself.
 * The semantics, field order, and defaults are byte-identical across the two.
 * ====================================================================== */
#ifndef ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H

/* ------------------------------------------------------- segment-kernel spec */
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

    bool has_tile_size_override; /* Optional[int] None (gfx942-only knob; ignored) */
    int tile_size_override;

    bool use_invariant_hoist; /* False (gfx942-only knob; ignored on gfx950) */
    bool use_wide_kv_load; /* False (gfx942-only knob; ignored on gfx950) */
    /* 64-bit paged-KV addressing for caches > 2 GiB. When set, the per-block
     * byte base (physical_block * block_stride) is folded into a 64-bit buffer
     * base (a wave-uniform per-block make_buffer_rsrc) and only the within-block
     * byte offset stays in the i32 buffer voffset -- mirrors the 2D tiled
     * kernel's use_i64_kv_addr. Default false => byte-identical small-cache
     * build. */
    bool use_i64_kv_addr; /* False */
} rocke_unified_attention_3d_tiled_spec_t;

rocke_unified_attention_3d_tiled_spec_t rocke_unified_attention_3d_tiled_spec_default(void);

/* ------------------------------------------------------- reduce-kernel spec */
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

#endif /* !ROCKE_INSTANCE_GFX942_ATTENTION_TILED_3D_H (shared spec POD) */

/* ============================================================ *
 * Coverage gate -- supports_tiled_3d(...)
 * ============================================================ *
 *
 * Faithful port of the module-level supports_tiled_3d host predicate. Returns
 * (ok). On false, *out_reason (if non-NULL) points to a static string mirroring
 * the Python reason; on true it points to "supported". Pure, emits no IR.
 *
 * Routes through validate_tiled_attention_arch first (arch gate), then the
 * dtype/head_size/block_size/num_queries_per_kv/kv_storage_dtype checks.
 *
 * q_dtype is an Optional[str]: pass NULL for Python None. kv_storage_dtype is
 * Optional[str] ("fp8e4m3" or NULL). arch NULL == the Python default "gfx950". */
bool rocke_gfx950_attention_tiled_3d_supports(int head_size,
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
 * Segment-kernel spec helpers (UnifiedAttention3DTiledSpec)
 * ============================================================ */

/* __post_init__ validator (builder variant). Reproduces the Python check
 * (kv_storage_dtype must be None or "fp8e4m3"); on failure records the Python
 * ValueError text on b (ROCKE_ERR_VALUE) and returns false. Dead/NULL b is a
 * false no-op. */
bool rocke_gfx950_unified_attention_3d_tiled_spec_validate(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* s);

/* ------------------------------------------------ derived @property accessors *
 * Pure, IR-free; faithful ports of the dataclass @property bodies. The
 * gfx950_ prefix avoids clashing with the gfx942 accessors of identical body. */
/* num_queries_per_kv = num_query_heads // num_kv_heads */
int rocke_gfx950_unified_attention_3d_tiled_spec_num_queries_per_kv(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* block_m = 16 (constant) */
int rocke_gfx950_unified_attention_3d_tiled_spec_block_m(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* block_q = block_m // num_queries_per_kv */
int rocke_gfx950_unified_attention_3d_tiled_spec_block_q(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* tile_size = block_size (the gfx950 @property ignores tile_size_override). */
int rocke_gfx950_unified_attention_3d_tiled_spec_tile_size(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* dtype_ir: F16 for "fp16", BF16 otherwise (returns a rocke_type_t*). */
const rocke_type_t* rocke_gfx950_unified_attention_3d_tiled_spec_dtype_ir(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* binary_search_iters: 32 if num_seqs<=0, else max(1, ceil(log2(num_seqs+1))) */
int rocke_gfx950_unified_attention_3d_tiled_spec_binary_search_iters(
    const rocke_unified_attention_3d_tiled_spec_t* s);
/* kernel_name(): kernel_name_join(...) into caller's buf (NUL-terminated, up to
 * cap bytes incl. NUL). Returns the length that would be written (snprintf
 * convention), or -1 on a dead builder/encode error. */
int rocke_gfx950_unified_attention_3d_tiled_spec_kernel_name(
    const rocke_unified_attention_3d_tiled_spec_t* s, char* buf, size_t cap);

/* ============================================================ *
 * Reduce-kernel spec helpers (UnifiedAttentionReduceTiledSpec)
 * ============================================================ */

/* dtype_ir: F16 for "fp16", BF16 otherwise. */
const rocke_type_t* rocke_gfx950_unified_attention_reduce_tiled_spec_dtype_ir(
    const rocke_unified_attention_reduce_tiled_spec_t* s);
/* kernel_name(): see segment-kernel variant for the snprintf convention. */
int rocke_gfx950_unified_attention_reduce_tiled_spec_kernel_name(
    const rocke_unified_attention_reduce_tiled_spec_t* s, char* buf, size_t cap);

/* ============================================================ *
 * Build entries (mirror the Python build_* functions)
 * ============================================================ *
 *
 * Each owns/inits the supplied builder b via the resolved kernel_name and emits
 * the kernel. arch NULL == the Python default "gfx950"; a non-gfx950 arch hits
 * the Python require_tiled_attention_arch gate -> sticky error + NULL. The
 * caller owns b and frees it with rocke_ir_builder_free(). Returns the kernel
 * (== b->kernel) or NULL on a sticky error.
 *
 * GRIDS (mirroring the AITER selector):
 *   segment: grid = (num_q_blocks, num_kv_heads, num_segments)
 *   reduce:  grid = (total_q, num_query_heads, 1)
 */

/* Segment kernel (wide 16x16x32 MFMA + ds_read_tr PV). Call with a spec
 * validated by rocke_gfx950_unified_attention_3d_tiled_spec_validate. */
rocke_kernel_def_t* rocke_build_unified_attention_3d_tiled_gfx950(
    rocke_ir_builder_t* b, const rocke_unified_attention_3d_tiled_spec_t* spec, const char* arch);

/* Reduce kernel (arch-neutral pure-f32 combine; same math as the gfx942 reduce). */
rocke_kernel_def_t* rocke_build_unified_attention_reduce_tiled_gfx950(
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
rocke_status_t rocke_build_unified_attention_3d_tiled_gfx950_lower_to_llvm(
    const rocke_unified_attention_3d_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

rocke_status_t rocke_build_unified_attention_reduce_tiled_gfx950_lower_to_llvm(
    const rocke_unified_attention_reduce_tiled_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX950_ATTENTION_TILED_3D_H */
