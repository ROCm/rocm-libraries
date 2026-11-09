/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_fmha_decode.h -- C99 port of the RDNA (gfx11 /
 * gfx12) split-KV, GQA-packed WMMA attention decode and its merge kernel
 * kernels/gfx1151/wmma_fmha_decode.py, with C host helpers that mirror the
 * Python launch plan (split count, pack groups, row tiles, grids, workspace).
 *
 *   Python (wmma_fmha_decode.py)              C99 (this header)
 *   ---------------------------------------   -----------------------------------------
 *   @dataclass WmmaFmhaDecodeSpec             rocke_wmma_fmha_decode_spec_t
 *     .kernel_name()                          rocke_wmma_fmha_decode_kernel_name(...)
 *     .block_size                             rocke_wmma_fmha_decode_block_size(...)
 *   resolve_wmma_fmha_decode_spec(s, arch=)   rocke_wmma_fmha_decode_resolve(...)
 *   is_valid_spec(spec, arch=)                rocke_wmma_fmha_decode_is_valid_spec(...)
 *   wmma_fmha_decode_lds_bytes(spec, arch=)   rocke_wmma_fmha_decode_lds_bytes(...)
 *   build_wmma_fmha_decode(spec, arch=)       rocke_build_wmma_fmha_decode(...)
 *   build_wmma_fmha_decode_merge(spec, arch=) rocke_build_wmma_fmha_decode_merge(...)
 *   (merge kernel name)                       rocke_wmma_fmha_decode_merge_kernel_name(...)
 *   wmma_fmha_decode_signature(spec)          rocke_wmma_fmha_decode_signature(...)
 *   wmma_fmha_decode_merge_signature(spec)    rocke_wmma_fmha_decode_merge_signature(...)
 *   wmma_fmha_decode_bucket(...)              rocke_wmma_fmha_decode_bucket(...)
 *   wmma_fmha_decode_kv_range(...)            rocke_wmma_fmha_decode_kv_range(...)
 *   wmma_fmha_decode_pack_groups(...)         rocke_wmma_fmha_decode_pack_groups(...)
 *   wmma_fmha_decode_row_tiles(...)           rocke_wmma_fmha_decode_row_tiles(...)
 *   wmma_fmha_decode_num_splits(spec, ...)    rocke_wmma_fmha_decode_num_splits(...)
 *   wmma_fmha_decode_grid(spec, ...)          rocke_wmma_fmha_decode_grid(...)
 *   wmma_fmha_decode_merge_grid(...)          rocke_wmma_fmha_decode_merge_grid(...)
 *   wmma_fmha_decode_workspace(spec, ...)     rocke_wmma_fmha_decode_workspace(...)
 *   wmma_fmha_decode_fused_merge(num_splits)  rocke_wmma_fmha_decode_fused_merge(...)
 *   (build -> lower .ll convenience)          rocke_wmma_fmha_decode_lower_to_llvm(...)
 *
 * The spec carries only compile-time facts (dtype, head-width bucket, bias
 * kind, feature flags, layout / page size, FP8 K/V) and the codegen knobs.
 * Batch, lengths, head counts, head widths, strides, window bounds, the LSE
 * gate, the split count and every feature operand are kernel arguments.
 *
 * Error model: build routines route errors through the sticky-error
 * IRBuilder; validity returns a bool + reason; host helpers return a
 * rocke_status_t (ROCKE_ERR_VALUE where Python raises ValueError).
 */
#ifndef ROCKE_INSTANCE_GFX1151_WMMA_FMHA_DECODE_H
#define ROCKE_INSTANCE_GFX1151_WMMA_FMHA_DECODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_WMMA_FMHA_DECODE_ROWS 16 /* packed query rows per WMMA tile */
#define ROCKE_WMMA_FMHA_DECODE_TILE 16 /* keys per wave per key-loop iteration */
#define ROCKE_WMMA_FMHA_DECODE_MERGE_THREADS 64 /* merge workgroup */

/* Flat mirror of @dataclass(frozen=True) WmmaFmhaDecodeSpec. String fields
 * are referenced as-is; keep them alive with the spec. Optional[bool] knobs
 * are tri-state ints: -1 = None (auto), 0 = False, 1 = True. Start from
 * rocke_wmma_fmha_decode_spec_default() so new fields take their defaults. */
typedef struct rocke_wmma_fmha_decode_spec
{
    int head_size; /* bucket 64 | 128 | 256 (required; default 128) */
    const char* dtype; /* "fp16" (default) | "bf16" */
    const char* bias; /* "" (default) | "q" | "f32" */
    bool softcap; /* default false */
    bool sinks; /* default false */
    bool alibi; /* default false */
    bool qq_bias; /* default false */
    const char* kv_dtype; /* "" (default) | "fp8e4m3" */
    const char* layout; /* "dense" (default) | "ragged" | "paged" */
    int page_block_size; /* paged only: keys per page (power of two); default 0 */
    int num_waves; /* 1 | 2 | 4 (default) | 8 */
    int dv_split; /* 0 (auto, default) | 1 | 2 | 4 */
    const char* q_mode; /* "" (auto, default) | "reg" | "lds" */
    int v_store_early; /* tri-state; default -1 */
    int prefetch; /* tri-state; default -1 */
    int lockstep; /* tri-state; default -1 */
    const char* scheduler_strategy; /* "auto" (default) | "default" | NULL (LLVM default) | name */
    int waves_per_eu; /* default 0 (none) */
    int fp8_cvt; /* tri-state; default -1 */
    int tanh_fast; /* tri-state; default -1 */
    const char* wmma_family; /* "" (unresolved, default) | "gfx11" | "gfx12" */
    const char* name; /* NULL => "rocke_wmma_fmha_decode" */
} rocke_wmma_fmha_decode_spec_t;

/* Default-constructed spec (Python dataclass defaults; head_size 128). */
rocke_wmma_fmha_decode_spec_t rocke_wmma_fmha_decode_spec_default(void);

/* WmmaFmhaDecodeSpec.block_size: threads per decode workgroup. */
int rocke_wmma_fmha_decode_block_size(const rocke_wmma_fmha_decode_spec_t* spec);

/* resolve_wmma_fmha_decode_spec(spec, arch=arch) into *out (string fields
 * point at static literals or the input's strings). ROCKE_ERR_VALUE on an
 * invalid spec, an arch without a wave32 WMMA atom, or a spec resolved for
 * the other family; `reason` receives the message. */
rocke_status_t rocke_wmma_fmha_decode_resolve(const rocke_wmma_fmha_decode_spec_t* spec,
                                              const char* arch,
                                              rocke_wmma_fmha_decode_spec_t* out,
                                              char* reason,
                                              size_t reason_cap);

/* WmmaFmhaDecodeSpec.kernel_name() of `spec` as given (resolve first for the
 * symbol the builder emits). ROCKE_ERR_VALUE on an invalid spec or a short
 * buffer. */
rocke_status_t rocke_wmma_fmha_decode_kernel_name(const rocke_wmma_fmha_decode_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap);

/* The merge kernel's symbol: rocke_wmma_fmha_decode_merge_<dtype>[_packed]. */
rocke_status_t rocke_wmma_fmha_decode_merge_kernel_name(const rocke_wmma_fmha_decode_spec_t* spec,
                                                        char* out,
                                                        size_t out_cap);

/* is_valid_spec(spec, arch=arch) -> (ok, reason). */
bool rocke_wmma_fmha_decode_is_valid_spec(const rocke_wmma_fmha_decode_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* wmma_fmha_decode_lds_bytes(spec, arch=arch); -1 when the spec does not
 * resolve for `arch`. */
int rocke_wmma_fmha_decode_lds_bytes(const rocke_wmma_fmha_decode_spec_t* spec, const char* arch);

/* build_wmma_fmha_decode(spec, arch=arch). `b` must be initialised with the
 * RESOLVED spec's kernel_name(). Returns b->kernel, or NULL on an invalid
 * spec / emission error (builder error set). */
rocke_kernel_def_t* rocke_build_wmma_fmha_decode(rocke_ir_builder_t* b,
                                                 const rocke_wmma_fmha_decode_spec_t* spec,
                                                 const char* arch);

/* build_wmma_fmha_decode_merge(spec, arch=arch). `b` must be initialised
 * with rocke_wmma_fmha_decode_merge_kernel_name(). */
rocke_kernel_def_t* rocke_build_wmma_fmha_decode_merge(rocke_ir_builder_t* b,
                                                       const rocke_wmma_fmha_decode_spec_t* spec,
                                                       const char* arch);

/* Kernel ABIs (names and type strings owned by `arena`). */
rocke_status_t rocke_wmma_fmha_decode_signature(const rocke_wmma_fmha_decode_spec_t* spec,
                                                rocke_arena_t* arena,
                                                const rocke_sig_entry_t** out_items,
                                                size_t* out_count);
rocke_status_t rocke_wmma_fmha_decode_merge_signature(const rocke_wmma_fmha_decode_spec_t* spec,
                                                      rocke_arena_t* arena,
                                                      const rocke_sig_entry_t** out_items,
                                                      size_t* out_count);

/* ---- host helpers ---- */

/* The head_size bucket serving runtime widths (head_dim_q, head_dim_v), or 0:
 * the smallest of 64 / 128 / 256 holding both (positive multiples of 8) whose
 * smallest served Q/K width (16 / 72 / 136) is at most head_dim_q. */
int rocke_wmma_fmha_decode_bucket(int head_dim_q, int head_dim_v);

/* The keys any query row can see, [lo, hi) (mirrors the kernel). */
void rocke_wmma_fmha_decode_kv_range(int seqlen_q,
                                     int seqlen_k,
                                     bool bottom_right,
                                     int window_left,
                                     int window_right,
                                     int* lo,
                                     int* hi);

/* Pack groups P = max(Hkv, Hv). ROCKE_ERR_VALUE unless Hq is a positive
 * multiple of both and one of Hkv / Hv divides the other. */
rocke_status_t rocke_wmma_fmha_decode_pack_groups(int num_query_heads,
                                                  int num_kv_heads,
                                                  int num_v_heads,
                                                  int* out);

/* 16-row tiles per pack group: ceil((Hq / P) * seqlen_q / 16). */
rocke_status_t rocke_wmma_fmha_decode_row_tiles(
    int num_query_heads, int num_kv_heads, int num_v_heads, int seqlen_q, int* out);

/* wmma_fmha_decode_num_splits: `spec` must be resolved (or carry an explicit
 * dv_split). wg_per_cu / min_tiles_per_wave take the Python defaults 2 / 4
 * when passed <= 0. kv_len is the visible key range (hi - lo of kv_range);
 * num_cus the device's multiprocessor count. */
rocke_status_t rocke_wmma_fmha_decode_num_splits(const rocke_wmma_fmha_decode_spec_t* spec,
                                                 int batch,
                                                 int pack_groups,
                                                 int row_tiles,
                                                 int kv_len,
                                                 int num_cus,
                                                 int wg_per_cu,
                                                 int min_tiles_per_wave,
                                                 int* out);

/* Decode grid {batch * splits * pack_groups * dv_split * row_tiles, 1, 1} and
 * block {32 * num_waves, 1, 1}; ROCKE_ERR_VALUE if empty or beyond I32. */
rocke_status_t rocke_wmma_fmha_decode_grid(const rocke_wmma_fmha_decode_spec_t* spec,
                                           int batch,
                                           int pack_groups,
                                           int row_tiles,
                                           int num_splits,
                                           int grid[3],
                                           int block[3]);

/* Merge grid {batch * Hq * seqlen_q, 1, 1}, block {64, 1, 1}. */
rocke_status_t rocke_wmma_fmha_decode_merge_grid(
    int batch, int num_query_heads, int seqlen_q, int grid[3], int block[3]);

/* Split workspace: byte offsets into one allocation of `bytes` (all zero with
 * one split). WS_O = base + ws_o, WS_LSE = base + ws_lse, WS_CNT = base +
 * ws_cnt (fused merge only; zero-initialised once, re-armed by the kernel). */
typedef struct rocke_wmma_fmha_decode_workspace
{
    int64_t ws_o;
    int64_t ws_lse;
    int64_t ws_cnt;
    int64_t bytes;
} rocke_wmma_fmha_decode_workspace_t;

rocke_status_t rocke_wmma_fmha_decode_workspace(const rocke_wmma_fmha_decode_spec_t* spec,
                                                int batch,
                                                int num_query_heads,
                                                int num_kv_heads,
                                                int num_v_heads,
                                                int seqlen_q,
                                                int head_dim_v,
                                                int num_splits,
                                                rocke_wmma_fmha_decode_workspace_t* out);

/* Merge in the last split (pass WS_CNT) rather than launch the merge kernel. */
bool rocke_wmma_fmha_decode_fused_merge(int num_splits);

/* Convenience: resolve, build and lower to LLVM .ll text (merge kernel when
 * `merge`). On ROCKE_OK *out_ll is a malloc'd string the caller frees; on
 * failure it is NULL and `err` receives a diagnostic. */
rocke_status_t rocke_wmma_fmha_decode_lower_to_llvm(const rocke_wmma_fmha_decode_spec_t* spec,
                                                    bool merge,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_WMMA_FMHA_DECODE_H */
