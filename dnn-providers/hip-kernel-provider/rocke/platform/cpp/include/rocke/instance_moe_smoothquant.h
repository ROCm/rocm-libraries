/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_smoothquant.h -- C99 port of the public surface of the
 * MoE-SmoothQuant instance (rocke/instances/common/moe_smoothquant.py).
 *
 *   Python (moe_smoothquant.py)            C99 (this header)
 *   -------------------------------------  -------------------------------------
 *   class MoeSmoothQuantSpec (frozen)      rocke_moe_smoothquant_spec_t
 *   MoeSmoothQuantSpec.elems_per_thread    rocke_moe_smoothquant_elems_per_thread()
 *   MoeSmoothQuantSpec.kernel_name()       rocke_moe_smoothquant_kernel_name()
 *   is_valid_spec(spec, arch)              rocke_moe_smoothquant_is_valid_spec()
 *   build_moe_smoothquant(spec, arch)      rocke_build_moe_smoothquant()
 *   moe_smoothquant_grid(tokens, spec)     rocke_moe_smoothquant_grid()
 *
 * MoE-SmoothQuant (CK Tile ``14_moe_smoothquant`` parity) extends SmoothQuant
 * with per-expert smooth scales and MoE router output layout:
 *
 *   * SmScale is a flat ``(experts * N,)`` per-expert smooth-scale table,
 *     gathered per CTA by the per-token expert id (an additional TopkIds i32
 *     param), not shared across all rows.
 *   * The kernel produces ``topk * tokens`` output rows; one CTA per output
 *     row, with ``(i_topk, i_token)`` decoded from the linear block_id_x.
 *     When ``spec.tokens`` is set (compile-time) the div/mod fold to a
 *     reciprocal-mul pair.
 *   * The expert id is read once per CTA (wave-uniform global_load_i32 pinned
 *     to SGPR via to_sgpr_u32) and ``sm_row_base = i_expert * N`` is
 *     pre-computed so each chunk's SmScale gather is a single s_add + load.
 *
 * The build reproduces the Python IRBuilder call sequence op-for-op so the
 * produced rocke_kernel_def_t is byte-faithful to the Python output.
 *
 * Error model mirrors the rest of the C port: the validity gate is a bool +
 * reason buffer; the build routes errors through the sticky-error IRBuilder and
 * returns NULL; the lower convenience returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MOE_SMOOTHQUANT_H
#define ROCKE_INSTANCE_MOE_SMOOTHQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default architecture (Python default arg arch="gfx950"). */
#define ROCKE_MOE_SMOOTHQUANT_DEFAULT_ARCH "gfx950"

/* ------------------------------------------------------------------ *
 * MoeSmoothQuantSpec
 * ------------------------------------------------------------------ *
 *
 * Mirrors the frozen dataclass field-for-field. Defaults (Python):
 *   dtype="f16", out_dtype="i8", block_size=256, vec=4, save_yscale=True,
 *   wave_size=64, name="rocke_moe_smoothquant", tokens=None.
 *
 * `dtype` is one of "f16"/"bf16"; `out_dtype` is one of
 * "i8"/"fp8e4m3"/"bf8e5m2". Both are referenced as-is (not copied).
 *
 * `tokens` is Optional[int] in Python: `tokens_set==false` means the runtime
 * div/mod decode path; `tokens_set==true` pins the compile-time const-fold
 * path against `tokens` (one specialised kernel per tokens value).
 */
typedef struct rocke_moe_smoothquant_spec
{
    int n_per_block; /* the hidden dim N (compile-time)   */
    int topk; /* router top-k (compile-time)       */
    int experts; /* total experts (compile-time)      */
    const char* dtype; /* "f16" / "bf16"              */
    const char* out_dtype; /* "i8" / "fp8e4m3" / "bf8e5m2" */
    int block_size;
    int vec;
    bool save_yscale;
    int wave_size;
    const char* name; /* kernel-name prefix             */
    bool tokens_set; /* Optional[int]: false => runtime div/mod path */
    int tokens; /* compile-time tokens (valid iff tokens_set)   */
} rocke_moe_smoothquant_spec_t;

/* Initialise `spec` with the Python dataclass defaults and the required
 * positional fields (n_per_block, topk, experts). dtype/out_dtype/name point at
 * static literals; tokens is unset (tokens_set=false). Callers may overwrite
 * any field afterwards. */
void rocke_moe_smoothquant_spec_init(rocke_moe_smoothquant_spec_t* spec,
                                     int n_per_block,
                                     int topk,
                                     int experts);

/* MoeSmoothQuantSpec.elems_per_thread property: n_per_block // block_size. */
int rocke_moe_smoothquant_elems_per_thread(const rocke_moe_smoothquant_spec_t* spec);

/* MoeSmoothQuantSpec.kernel_name(): writes the joined name into `out`
 * (capacity out_cap, NUL-terminated). Parts:
 *   name, dtype, out_dtype, "N{n}", "E{experts}", "K{topk}", "b{bs}", "v{vec}",
 *   flags={"ys": save_yscale}. Returns ROCKE_OK or ROCKE_ERR_VALUE when the buffer
 * is too small. */
rocke_status_t rocke_moe_smoothquant_kernel_name(const rocke_moe_smoothquant_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * Returns true (and writes "" to `reason` when non-NULL) on accept, or false
 * with the structured Python reason string on reject. `arch` NULL =>
 * ROCKE_MOE_SMOOTHQUANT_DEFAULT_ARCH ("gfx950"). `reason`/`reason_cap` may be
 * NULL/0 to skip the message. Mirrors is_valid_spec(spec, arch). */
bool rocke_moe_smoothquant_is_valid_spec(const rocke_moe_smoothquant_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_moe_smoothquant
 * ------------------------------------------------------------------ *
 *
 * Validates `spec` against `arch` via rocke_moe_smoothquant_is_valid_spec(), then
 * builds the MoE-SmoothQuant forward IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b`, op-for-op against build_moe_smoothquant().
 * Returns b->kernel on success or NULL with b's sticky error set. `arch` NULL
 * => "gfx950".
 *
 * Like the Python (IRBuilder(spec.kernel_name())), this does NOT re-init the
 * builder; the caller owns its lifetime and should have created it with the
 * spec's kernel name. Use rocke_build_moe_smoothquant_new() for the convenience. */
rocke_kernel_def_t* rocke_build_moe_smoothquant(rocke_ir_builder_t* b,
                                                const rocke_moe_smoothquant_spec_t* spec,
                                                const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_moe_smoothquant_new(rocke_ir_builder_t* b,
                                                    const rocke_moe_smoothquant_spec_t* spec,
                                                    const char* arch);

/* ------------------------------------------------------------------ *
 * moe_smoothquant_grid
 * ------------------------------------------------------------------ *
 *
 * Launch grid: one CTA per (i_topk, i_token) pair ->
 * ceil_div_grid((tokens * topk, 1)). Writes (x, y, z) into out[0..2]. Returns
 * ROCKE_OK or the ceil_div_grid error. Mirrors moe_smoothquant_grid(tokens, spec). */
rocke_status_t
    rocke_moe_smoothquant_grid(int tokens, const rocke_moe_smoothquant_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build, and lower to LLVM .ll text. `arch` NULL
 * => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns/frees its IRBuilder. */
rocke_status_t rocke_moe_smoothquant_lower_to_llvm(const rocke_moe_smoothquant_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_SMOOTHQUANT_H */
