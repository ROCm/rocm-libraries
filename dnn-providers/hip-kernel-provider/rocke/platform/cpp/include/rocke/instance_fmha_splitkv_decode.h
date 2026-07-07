/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_splitkv_decode.h -- C99 port of the split-KV decode FMHA
 * forward instance builder rocke/instances/common/fmha_splitkv_decode.py.
 *
 * Decoding from a long KV cache with a small batch is bandwidth-limited;
 * splitting the K dimension across many CTAs (each handling one K-segment) and
 * then reducing the per-segment (m, l, acc) triples lifts occupancy to fully
 * saturate the SMs. Two-launch pipeline:
 *
 *   1. build_fmha_fwd_splitkv_decode_segment -- one CTA per
 *      (seq_idx, head_idx, segment_idx). Each CTA walks its slice of the K cache
 *      and emits the per-segment (m, l, acc) to the workspace.
 *   2. build_fmha_fwd_splitkv_decode_reduce -- combine the per-segment triples
 *      back into the final O = acc/l using the numerically-stable two-pass merge.
 *
 * Both kernels use the warp-distributed body (one warp per CTA, lane distributes
 * the head_dim) so no LDS state and no thread-redundant work.
 *
 *   Python (fmha_splitkv_decode.py)               C99 (this header)
 *   ------------------------------------------     -------------------------------
 *   FmhaFwdSplitKvDecodeSpec (frozen dataclass)    rocke_fmha_splitkv_decode_spec_t
 *     .kernel_name(phase)                          rocke_fmha_splitkv_decode_kernel_name
 *   is_valid_spec(spec, arch)                      rocke_fmha_splitkv_decode_is_valid_spec
 *   build_fmha_fwd_splitkv_decode_segment(...)     rocke_build_fmha_fwd_splitkv_decode_segment
 *   build_fmha_fwd_splitkv_decode_reduce(...)      rocke_build_fmha_fwd_splitkv_decode_reduce
 *   fmha_fwd_splitkv_decode_segment_grid(...)      rocke_fmha_splitkv_decode_segment_grid
 *   fmha_fwd_splitkv_decode_reduce_grid(...)       rocke_fmha_splitkv_decode_reduce_grid
 *   (+ convenience: build -> lower .ll)            rocke_..._segment_lower_to_llvm /
 *                                                  rocke_..._reduce_lower_to_llvm
 *
 * The build entries drive a rocke_fmha_kernel_builder_t and reproduce the Python
 * builder-call sequence byte-faithfully (param order, const_i32/f32 sequence,
 * magic-division GQA decode, scf.for online-softmax, scf.if lead-lane store).
 *
 * Error model mirrors the rest of the C port: build/lower route through the
 * sticky-error IRBuilder; the validity gate returns a bool + reason string; the
 * convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_SPLITKV_DECODE_H
#define ROCKE_INSTANCE_FMHA_SPLITKV_DECODE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h" /* spec types */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaFwdSplitKvDecodeSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaFwdSplitKvDecodeSpec:
 *     common: FmhaCommonSpec
 *     batch: int
 *     num_segments: int
 *     name: str = "rocke_fmha_fwd_splitkv_decode"
 *     use_mfma_body: bool = False
 *     prune_sliding_window: bool = False
 *
 * `name` is referenced as-is (not copied); keep it alive. */
typedef struct rocke_fmha_splitkv_decode_spec
{
    rocke_fmha_common_spec_t common;
    int batch;
    int num_segments;
    const char* name; /* default "rocke_fmha_fwd_splitkv_decode" */
    bool use_mfma_body; /* default false */
    bool prune_sliding_window; /* default false */
} rocke_fmha_splitkv_decode_spec_t;

/* Construct from `common`, `batch`, `num_segments` with the dataclass defaults
 * for name / use_mfma_body / prune_sliding_window. */
rocke_fmha_splitkv_decode_spec_t rocke_fmha_splitkv_decode_spec_default(
    rocke_fmha_common_spec_t common, int batch, int num_segments);

/* FmhaFwdSplitKvDecodeSpec.kernel_name(phase):
 *   kernel_name_join(name, phase, f"H{head_size}", f"HQ{num_query_heads}",
 *                    f"HK{num_kv_heads}", dtype, f"B{batch}", f"S{num_segments}")
 *
 * Writes the NUL-terminated name into `out` (capacity out_cap). Returns ROCKE_OK
 * or ROCKE_ERR_VALUE (NULL args / too-small buffer). */
rocke_status_t rocke_fmha_splitkv_decode_kernel_name(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                     const char* phase,
                                                     char* out,
                                                     size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec(spec, arch) -> (ok, reason)
 * ------------------------------------------------------------------ *
 *
 *   - validate_common_spec(spec.common)
 *   - validate_fmha_mfma_atom(spec.common.dtype, arch)
 *   - batch > 0
 *   - num_segments in {1, 2, 4, 8, 16, 32, 64, 128}
 *
 * `arch` NULL => "gfx950". On reject `reason` (if non-NULL, capacity reason_cap)
 * receives the structured message and false is returned; on accept returns true
 * and writes "ok". */
bool rocke_fmha_splitkv_decode_is_valid_spec(const rocke_fmha_splitkv_decode_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_splitkv_decode_segment(spec, arch)
 * ------------------------------------------------------------------ *
 *
 * Segment kernel: one CTA per (seq_idx, head_idx, segment_idx). `kb` must be a
 * fresh (uninitialised) builder; this routine initialises it via
 * rocke_fmha_kernel_builder_init with spec.kernel_name("seg"), builds the IR, and
 * returns the kernel (or NULL with kb's IR builder error set). The caller owns
 * `kb` and frees it with rocke_fmha_kernel_builder_free. `arch` NULL => "gfx950". */
rocke_kernel_def_t*
    rocke_build_fmha_fwd_splitkv_decode_segment(rocke_fmha_kernel_builder_t* kb,
                                                const rocke_fmha_splitkv_decode_spec_t* spec,
                                                const char* arch);

/* Reduce kernel: combine per-segment (m, l, acc) into O. Same `kb` lifetime
 * contract as the segment builder; initialises kb with spec.kernel_name("reduce").
 */
rocke_kernel_def_t*
    rocke_build_fmha_fwd_splitkv_decode_reduce(rocke_fmha_kernel_builder_t* kb,
                                               const rocke_fmha_splitkv_decode_spec_t* spec,
                                               const char* arch);

/* ------------------------------------------------------------------ *
 * grids
 * ------------------------------------------------------------------ *
 *
 * segment_grid(spec) -> (batch, num_query_heads, num_segments)
 * reduce_grid(spec)  -> (batch, num_query_heads, 1)
 *
 * On success writes out[0..2] and returns ROCKE_OK; ROCKE_ERR_VALUE on NULL args. */
rocke_status_t rocke_fmha_splitkv_decode_segment_grid(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                      int out[3]);
rocke_status_t rocke_fmha_splitkv_decode_reduce_grid(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                     int out[3]);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Build the segment / reduce kernel and lower it to LLVM .ll text. `arch` NULL
 * => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Each owns and frees its builder. */
rocke_status_t
    rocke_fmha_splitkv_decode_segment_lower_to_llvm(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);
rocke_status_t
    rocke_fmha_splitkv_decode_reduce_lower_to_llvm(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_SPLITKV_DECODE_H */
