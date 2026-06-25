/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_moe_smoothquant.h -- C99 port of the public surface of the
 * MoE-SmoothQuant instance (ck_dsl/instances/common/moe_smoothquant.py).
 *
 *   Python (moe_smoothquant.py)            C99 (this header)
 *   -------------------------------------  -------------------------------------
 *   class MoeSmoothQuantSpec (frozen)      ckc_moe_smoothquant_spec_t
 *   MoeSmoothQuantSpec.elems_per_thread    ckc_moe_smoothquant_elems_per_thread()
 *   MoeSmoothQuantSpec.kernel_name()       ckc_moe_smoothquant_kernel_name()
 *   is_valid_spec(spec, arch)              ckc_moe_smoothquant_is_valid_spec()
 *   build_moe_smoothquant(spec, arch)      ckc_build_moe_smoothquant()
 *   moe_smoothquant_grid(tokens, spec)     ckc_moe_smoothquant_grid()
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
 * produced ckc_kernel_def_t is byte-faithful to the Python output.
 *
 * Error model mirrors the rest of the C port: the validity gate is a bool +
 * reason buffer; the build routes errors through the sticky-error IRBuilder and
 * returns NULL; the lower convenience returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_MOE_SMOOTHQUANT_H
#define CKC_INSTANCE_MOE_SMOOTHQUANT_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default architecture (Python default arg arch="gfx950"). */
#define CKC_MOE_SMOOTHQUANT_DEFAULT_ARCH "gfx950"

/* ------------------------------------------------------------------ *
 * MoeSmoothQuantSpec
 * ------------------------------------------------------------------ *
 *
 * Mirrors the frozen dataclass field-for-field. Defaults (Python):
 *   dtype="f16", out_dtype="i8", block_size=256, vec=4, save_yscale=True,
 *   wave_size=64, name="ck_dsl_moe_smoothquant", tokens=None.
 *
 * `dtype` is one of "f16"/"bf16"; `out_dtype` is one of
 * "i8"/"fp8e4m3"/"bf8e5m2". Both are referenced as-is (not copied).
 *
 * `tokens` is Optional[int] in Python: `tokens_set==false` means the runtime
 * div/mod decode path; `tokens_set==true` pins the compile-time const-fold
 * path against `tokens` (one specialised kernel per tokens value).
 */
typedef struct ckc_moe_smoothquant_spec
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
} ckc_moe_smoothquant_spec_t;

/* Initialise `spec` with the Python dataclass defaults and the required
 * positional fields (n_per_block, topk, experts). dtype/out_dtype/name point at
 * static literals; tokens is unset (tokens_set=false). Callers may overwrite
 * any field afterwards. */
void ckc_moe_smoothquant_spec_init(ckc_moe_smoothquant_spec_t* spec,
                                   int n_per_block,
                                   int topk,
                                   int experts);

/* MoeSmoothQuantSpec.elems_per_thread property: n_per_block // block_size. */
int ckc_moe_smoothquant_elems_per_thread(const ckc_moe_smoothquant_spec_t* spec);

/* MoeSmoothQuantSpec.kernel_name(): writes the joined name into `out`
 * (capacity out_cap, NUL-terminated). Parts:
 *   name, dtype, out_dtype, "N{n}", "E{experts}", "K{topk}", "b{bs}", "v{vec}",
 *   flags={"ys": save_yscale}. Returns CKC_OK or CKC_ERR_VALUE when the buffer
 * is too small. */
ckc_status_t ckc_moe_smoothquant_kernel_name(const ckc_moe_smoothquant_spec_t* spec,
                                             char* out,
                                             size_t out_cap);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * Returns true (and writes "" to `reason` when non-NULL) on accept, or false
 * with the structured Python reason string on reject. `arch` NULL =>
 * CKC_MOE_SMOOTHQUANT_DEFAULT_ARCH ("gfx950"). `reason`/`reason_cap` may be
 * NULL/0 to skip the message. Mirrors is_valid_spec(spec, arch). */
bool ckc_moe_smoothquant_is_valid_spec(const ckc_moe_smoothquant_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_moe_smoothquant
 * ------------------------------------------------------------------ *
 *
 * Validates `spec` against `arch` via ckc_moe_smoothquant_is_valid_spec(), then
 * builds the MoE-SmoothQuant forward IR into the supplied (already
 * ckc_ir_builder_init'd) builder `b`, op-for-op against build_moe_smoothquant().
 * Returns b->kernel on success or NULL with b's sticky error set. `arch` NULL
 * => "gfx950".
 *
 * Like the Python (IRBuilder(spec.kernel_name())), this does NOT re-init the
 * builder; the caller owns its lifetime and should have created it with the
 * spec's kernel name. Use ckc_build_moe_smoothquant_new() for the convenience. */
ckc_kernel_def_t* ckc_build_moe_smoothquant(ckc_ir_builder_t* b,
                                            const ckc_moe_smoothquant_spec_t* spec,
                                            const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_moe_smoothquant_new(ckc_ir_builder_t* b,
                                                const ckc_moe_smoothquant_spec_t* spec,
                                                const char* arch);

/* ------------------------------------------------------------------ *
 * moe_smoothquant_grid
 * ------------------------------------------------------------------ *
 *
 * Launch grid: one CTA per (i_topk, i_token) pair ->
 * ceil_div_grid((tokens * topk, 1)). Writes (x, y, z) into out[0..2]. Returns
 * CKC_OK or the ceil_div_grid error. Mirrors moe_smoothquant_grid(tokens, spec). */
ckc_status_t
    ckc_moe_smoothquant_grid(int tokens, const ckc_moe_smoothquant_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * lower-to-.ll convenience
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build, and lower to LLVM .ll text. `arch` NULL
 * => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns/frees its IRBuilder. */
ckc_status_t ckc_moe_smoothquant_lower_to_llvm(const ckc_moe_smoothquant_spec_t* spec,
                                               const char* arch,
                                               ckc_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_MOE_SMOOTHQUANT_H */
