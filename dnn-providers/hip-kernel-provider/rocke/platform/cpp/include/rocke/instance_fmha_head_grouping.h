/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_head_grouping.h -- C99 port of the GQA / MQA head-grouped
 * FMHA forward kernel instance builder
 * rocke/instances/common/fmha_head_grouping.py.
 *
 *   Python (fmha_head_grouping.py)             C99 (this header)
 *   ----------------------------------------   ---------------------------------
 *   @dataclass(frozen=True)
 *   class FmhaFwdHeadGroupingSpec              rocke_fmha_head_grouping_spec_t
 *     .kernel_name()                           rocke_fmha_head_grouping_kernel_name(...)
 *     .is_mqa / .is_gqa / .grouping_label      rocke_fmha_head_grouping_is_mqa(...) etc.
 *   is_valid_spec(spec, arch)                  rocke_fmha_head_grouping_is_valid_spec(...)
 *   build_fmha_fwd_head_grouping(spec, arch)   rocke_build_fmha_fwd_head_grouping(...)
 *   fmha_fwd_head_grouping_grid(spec, batch)   rocke_fmha_head_grouping_grid(...)
 *   (+ convenience: build -> lower .ll)        rocke_fmha_head_grouping_lower_to_llvm(...)
 *
 * The pure multi-head case is num_query_heads == num_kv_heads; the head-grouping
 * variants share one K/V head across multiple Q heads:
 *   - MQA (multi-query)  -- num_kv_heads == 1.
 *   - GQA (grouped-query)-- num_kv_heads in (2, 4, 8, ...).
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type with
 * a defaulted `name`. The caller fills a rocke_fmha_head_grouping_spec_t; the
 * `name` field carries the dataclass default
 * ("rocke_fmha_fwd_head_grouping") when set via rocke_fmha_head_grouping_spec_make
 * / _default.
 *
 * FIDELITY CONTRACT. rocke_build_fmha_fwd_head_grouping reproduces the Python
 * build() builder-call sequence op-for-op, in the same order, with the same
 * operands and compile-time constants, so the emitted IR is byte-identical to
 * the Python instance's emission (on the default arch="gfx950").
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder embedded in the FmhaKernelBuilder; the validity
 * gate returns a bool + reason string; the convenience lower returns a
 * rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_HEAD_GROUPING_H
#define ROCKE_INSTANCE_FMHA_HEAD_GROUPING_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_common_spec_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaFwdHeadGroupingSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaFwdHeadGroupingSpec:
 *     common: FmhaCommonSpec
 *     seqlen_q: int
 *     seqlen_k: int
 *     name: str = "rocke_fmha_fwd_head_grouping"
 *
 * `name` is referenced as-is (not copied); keep it alive for the spec's use.
 * Use rocke_fmha_head_grouping_spec_make() to take the dataclass default name. */
typedef struct rocke_fmha_head_grouping_spec
{
    rocke_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    const char* name; /* default "rocke_fmha_fwd_head_grouping" */
} rocke_fmha_head_grouping_spec_t;

/* FmhaFwdHeadGroupingSpec(common, seqlen_q, seqlen_k): construct with the
 * dataclass default name. */
rocke_fmha_head_grouping_spec_t
    rocke_fmha_head_grouping_spec_make(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* FmhaFwdHeadGroupingSpec.kernel_name(): writes the joined kernel name
 * NUL-terminated into `out` (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE
 * (buffer too small). */
rocke_status_t rocke_fmha_head_grouping_kernel_name(const rocke_fmha_head_grouping_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);

/* FmhaFwdHeadGroupingSpec.is_mqa property: num_kv_heads == 1. */
bool rocke_fmha_head_grouping_is_mqa(const rocke_fmha_head_grouping_spec_t* spec);

/* FmhaFwdHeadGroupingSpec.is_gqa property:
 * num_query_heads > num_kv_heads > 1. */
bool rocke_fmha_head_grouping_is_gqa(const rocke_fmha_head_grouping_spec_t* spec);

/* FmhaFwdHeadGroupingSpec.grouping_label property:
 * "mqa" if is_mqa else ("gqa" if is_gqa else "mha"). A static string. */
const char* rocke_fmha_head_grouping_grouping_label(const rocke_fmha_head_grouping_spec_t* spec);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * is_valid_spec(spec, arch="gfx950") -> (ok, reason). Runs
 * validate_common_spec, then validate_fmha_mfma_atom, then the seqlen / head /
 * grouping constraints. `arch` NULL => "gfx950". On a reject, `reason` (if
 * non-NULL, capacity reason_cap) receives the structured message; returns
 * false. On accept returns true and writes "ok". `arena` backs the reason text
 * produced by validate_common_spec; pass a live arena. */
bool rocke_fmha_head_grouping_is_valid_spec(rocke_arena_t* arena,
                                            const rocke_fmha_head_grouping_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_head_grouping
 * ------------------------------------------------------------------ *
 *
 * build_fmha_fwd_head_grouping(spec, arch="gfx950"). Initialises `kb`'s embedded
 * IR builder with spec.kernel_name(), declares the head_grouping ABI, decodes
 * the grid, emits the MFMA-tiled inner body, and returns the kernel
 * (kb->b.kernel) on success or NULL with kb's sticky error set. `arch` NULL =>
 * "gfx950". The caller owns `kb` and frees it with
 * rocke_fmha_kernel_builder_free(). On the is_valid_spec ValueError path this
 * returns NULL and sets a ROCKE_ERR_VALUE sticky error on kb's builder. */
rocke_kernel_def_t* rocke_build_fmha_fwd_head_grouping(rocke_fmha_kernel_builder_t* kb,
                                                       const rocke_fmha_head_grouping_spec_t* spec,
                                                       const char* arch);

/* ------------------------------------------------------------------ *
 * fmha_fwd_head_grouping_grid
 * ------------------------------------------------------------------ *
 *
 * fmha_fwd_head_grouping_grid(spec, batch=batch) ->
 *   (seqlen_q // BLOCK_M, num_query_heads, batch). Writes the three grid dims
 * to *gx / *gy / *gz. */
void rocke_fmha_head_grouping_grid(
    const rocke_fmha_head_grouping_spec_t* spec, int batch, int* gx, int* gy, int* gz);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Convenience: given a spec, init a kernel builder, build, and lower to LLVM .ll
 * text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err != NULL, capacity err_cap) a diagnostic is written. Internally
 * owns and frees its FmhaKernelBuilder. */
rocke_status_t rocke_fmha_head_grouping_lower_to_llvm(const rocke_fmha_head_grouping_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_HEAD_GROUPING_H */
