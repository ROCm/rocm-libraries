/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_sage_attention.h -- C99 port of the Sage attention forward
 * kernel instance builder rocke/instances/common/sage_attention.py (CK Tile
 * 49_sageattention parity).
 *
 * Sage extends FMHA-fwd-fp8 with per-block / per-head Q and K scales that
 * compensate for the dynamic range loss when Q and K live in fp8 / int8 / int4
 * storage. The pipeline (per K-block):
 *
 *     K_dequant   = dequant_codebook[K_quant[k_block, :]] * k_block_scale
 *     score_log2  = (Q[q_token, :] . K_dequant) * Q_scale * K_scale
 *     ... (online softmax as in fmha_fwd) ...
 *     V_dequant   = dequant_codebook[V_quant[k_block, :]] * v_block_scale
 *     acc        += p * V_dequant
 *
 * The four CK Tile Sage variants share one entry-point, parameterised by
 * quant_mode (SageQuantMode):
 *   * "fp16_bf16"    -- baseline (no QK quant; pipeline validation).
 *   * "fp8_bf16"     -- Q in activation dtype; K/V in fp8e4m3 + per-block scales.
 *   * "i8_fp8_bf16"  -- K/V stored as i8; codebook re-materialises fp32 values.
 *   * "i4_fp8_bf16"  -- K/V as packed i4; 16-entry codebook + per-block scale.
 *
 * Two physical kernels back the four modes:
 *   * MFMA-tiled body (rocke_mfma_attention_fwd_inner_body) -- fast path for
 *     fp16_bf16 / fp8_bf16 when seqlen_q / seqlen_k / scale-blocks align with the
 *     MFMA geometry (BLOCK_M = BLOCK_K = 16).
 *   * Warp-distributed body (rocke_fmha_warp_fwd_inner_body) -- universal fallback
 *     (one warp per (q_token, head) row), the only path for i8_fp8_bf16 /
 *     i4_fp8_bf16 and for small / unaligned fp16/fp8 specs.
 *
 *   Python (sage_attention.py)             C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   SageQuantMode (Literal)                rocke_sage_quant_mode_t (enum)
 *   @dataclass(frozen=True)                rocke_sage_attention_spec_t
 *     SageAttentionSpec
 *     .kernel_name()                       rocke_sage_attention_kernel_name(...)
 *   is_valid_spec(spec, arch)              rocke_sage_attention_is_valid_spec(...)
 *   _mfma_dimensions_ok(spec)              rocke_sage_attention_mfma_dimensions_ok(...)
 *   _uses_mfma_path(spec)                  rocke_sage_attention_uses_mfma_path(...)
 *   build_sage_attention(spec, arch)       rocke_build_sage_attention(...)
 *   _build_sage_mfma(spec, arch)           (internal; see *_internal.h)
 *   _build_sage_warp(spec)                 (internal; see *_internal.h)
 *   sage_attention_grid(spec)              rocke_sage_attention_grid(...)
 *   sage_attention_signature(spec)         rocke_sage_attention_signature(...)
 *   (+ convenience: build -> lower .ll)    rocke_sage_attention_lower_to_llvm(...)
 *
 * The build entry mirrors the Python build op-for-op (same dispatch, same
 * builder-call order, same attrs) so the emitted IR is byte-identical to
 * build_sage_attention on the default arch="gfx950". The two physical-kernel
 * bodies and every shared/closured local live in
 * rocke/instance_sage_attention_internal.h (the build-context contract); this
 * public header is the spec + dispatch glue only.
 *
 * NEW DEPENDENCY (this op's only newly-ported reusable):
 *   rocke.helpers.qk_scale -> rocke/helper_rocke.helpers.qk_scale.h
 *     (QkScaleSpec / QkScaleLayout / apply_qk_scales / load_q_scale_for_block /
 *      load_k_scale_for_block). The Sage spec embeds two rocke_qk_scale_spec_t.
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass carries defaults; in
 * C the caller fills a rocke_sage_attention_spec_t. rocke_sage_attention_spec_default()
 * returns a struct with every defaulted field set to its Python dataclass default
 * (the `common`, `quant_mode`, `q_scale`, `k_scale`, `seqlen_q`, `seqlen_k`
 * fields must still be filled by the caller).
 *
 * ERROR MODEL mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder; the validity gate returns bool + a reason string;
 * the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_SAGE_ATTENTION_H
#define ROCKE_INSTANCE_SAGE_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.qk_scale.h" /* rocke_qk_scale_spec_t */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* rocke_fmha_common_spec_t,
                                                              * rocke_fmha_kernel_builder_t,
                                                              * rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * SageQuantMode
 * ------------------------------------------------------------------ *
 *
 * SageQuantMode = Literal["fp16_bf16","fp8_bf16","i8_fp8_bf16","i4_fp8_bf16"].
 * Stored as an enum; rocke_sage_quant_mode_name() recovers the canonical lowercase
 * spelling (fed into the kernel name and ValueError-matching messages). */
typedef enum rocke_sage_quant_mode
{
    ROCKE_SAGE_QUANT_FP16_BF16 = 0, /* "fp16_bf16"   -- baseline, no QK quant   */
    ROCKE_SAGE_QUANT_FP8_BF16, /* "fp8_bf16"    -- K/V fp8e4m3 + scales    */
    ROCKE_SAGE_QUANT_I8_FP8_BF16, /* "i8_fp8_bf16" -- K/V i8 + f32 codebook   */
    ROCKE_SAGE_QUANT_I4_FP8_BF16 /* "i4_fp8_bf16" -- K/V packed i4 + codebook */
} rocke_sage_quant_mode_t;

/* Canonical lowercase spelling for a quant mode ("fp16_bf16", ...); NULL for an
 * out-of-range value. */
const char* rocke_sage_quant_mode_name(rocke_sage_quant_mode_t m);

/* True iff the quant mode is one of the MFMA-capable modes
 * (_MFMA_QUANT_MODES = ("fp16_bf16", "fp8_bf16")). */
bool rocke_sage_quant_mode_is_mfma(rocke_sage_quant_mode_t m);

/* True iff the quant mode is one of the codebook modes
 * (_CODEBOOK_QUANT_MODES = ("i8_fp8_bf16", "i4_fp8_bf16")). */
bool rocke_sage_quant_mode_is_codebook(rocke_sage_quant_mode_t m);

/* ------------------------------------------------------------------ *
 * SageAttentionSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class SageAttentionSpec:
 *     common: FmhaCommonSpec
 *     quant_mode: SageQuantMode
 *     q_scale: QkScaleSpec
 *     k_scale: QkScaleSpec
 *     seqlen_q: int
 *     seqlen_k: int
 *     name: str = "rocke_sage_attention"
 *     use_outer_scale_loop: bool = False
 *
 * `name` is referenced as-is (not copied); keep it alive for the spec's use.
 * use_outer_scale_loop is the P90 per-block-scale outer-loop opt (defaults
 * False). */
typedef struct rocke_sage_attention_spec
{
    rocke_fmha_common_spec_t common;
    rocke_sage_quant_mode_t quant_mode;
    rocke_qk_scale_spec_t q_scale;
    rocke_qk_scale_spec_t k_scale;
    int seqlen_q;
    int seqlen_k;
    const char* name; /* default "rocke_sage_attention" */
    bool use_outer_scale_loop; /* default false                   */
} rocke_sage_attention_spec_t;

/* Default-constructed spec (the two defaulted fields == Python dataclass
 * defaults: name="rocke_sage_attention", use_outer_scale_loop=false). The
 * caller must still fill common / quant_mode / q_scale / k_scale / seqlen_q /
 * seqlen_k. */
rocke_sage_attention_spec_t rocke_sage_attention_spec_default(void);

/* SageAttentionSpec.kernel_name() -> kernel_name_join(name, quant_mode,
 * "H{head_size}", "HQ{num_query_heads}", "HK{num_kv_heads}", common.dtype,
 * "Q{seqlen_q}", "K{seqlen_k}"). NUL-terminated into `out` (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small / NULL args). */
rocke_status_t rocke_sage_attention_kernel_name(const rocke_sage_attention_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* ------------------------------------------------------------------ *
 * MFMA-path predicates
 * ------------------------------------------------------------------ *
 *
 * _mfma_dimensions_ok(spec): true iff seqlen_q % BLOCK_M == 0 AND
 * seqlen_k % BLOCK_K == 0 AND head_size % BLOCK_M == 0 AND (per_block q_scale's
 * scale_block % BLOCK_M == 0) AND (per_block k_scale's scale_block % BLOCK_K ==
 * 0). Pure integer test; no IR, no builder. */
bool rocke_sage_attention_mfma_dimensions_ok(const rocke_sage_attention_spec_t* spec);

/* _uses_mfma_path(spec): quant_mode in _MFMA_QUANT_MODES AND
 * _mfma_dimensions_ok(spec). Decides which physical kernel build_sage_attention
 * dispatches to, and the grid shape. Pure integer test. */
bool rocke_sage_attention_uses_mfma_path(const rocke_sage_attention_spec_t* spec);

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ *
 *
 * is_valid_spec(spec, arch="gfx950") -> (ok, reason). `arch` NULL => "gfx950".
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured Python-matching message and the function returns false; on accept
 * returns true and writes "ok". Mirrors the Python gates exactly:
 *   - validate_common_spec failure
 *   - validate_fmha_mfma_atom(common.dtype, arch) failure (arch lacks the atom)
 *   - quant_mode not one of the four literals
 *   - seqlen_q <= 0 or seqlen_k <= 0
 *   - i4_fp8_bf16 with head_size odd
 *   - head_size % WARP_SIZE != 0 (the warp body's universal constraint)         */
bool rocke_sage_attention_is_valid_spec(const rocke_sage_attention_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

/* ------------------------------------------------------------------ *
 * build_sage_attention
 * ------------------------------------------------------------------ *
 *
 * build_sage_attention(spec, arch="gfx950") -> KernelDef. The public entry +
 * dispatch glue:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError
 *   if _uses_mfma_path(spec): return _build_sage_mfma(spec, arch)
 *   return _build_sage_warp(spec)
 *
 * Drives the supplied (already initialised) FmhaKernelBuilder. The caller passes
 * a rocke_fmha_kernel_builder_t whose rocke_fmha_kernel_builder_init was already
 * called with spec.kernel_name() + spec.common; the caller owns its lifetime
 * (rocke_fmha_kernel_builder_free). `arch` NULL => "gfx950". Returns the kernel on
 * success or NULL with the embedded builder's sticky error set (an invalid spec
 * sets ROCKE_ERR_VALUE with the Python-matching message). */
rocke_kernel_def_t* rocke_build_sage_attention(rocke_fmha_kernel_builder_t* kb,
                                               const rocke_sage_attention_spec_t* spec,
                                               const char* arch);

/* Convenience: init `kb` with spec.kernel_name() + spec.common, then build. The
 * caller owns `kb` and frees it with rocke_fmha_kernel_builder_free(). Returns the
 * kernel or NULL. `arch` NULL => "gfx950". */
rocke_kernel_def_t* rocke_build_sage_attention_new(rocke_fmha_kernel_builder_t* kb,
                                                   const rocke_sage_attention_spec_t* spec,
                                                   const char* arch);

/* ------------------------------------------------------------------ *
 * sage_attention_grid
 * ------------------------------------------------------------------ *
 *
 * sage_attention_grid(spec) -> launch grid triple:
 *   MFMA path: (seqlen_q // BLOCK_M, num_query_heads, 1)
 *   warp path: (seqlen_q,            num_query_heads, 1)
 * Writes the triple into out[3]; returns ROCKE_OK (ROCKE_ERR_VALUE on NULL
 * spec/out). */
rocke_status_t rocke_sage_attention_grid(const rocke_sage_attention_spec_t* spec, int out[3]);

/* ------------------------------------------------------------------ *
 * sage_attention_signature
 * ------------------------------------------------------------------ *
 *
 * sage_attention_signature(spec): declare the kernel ABI into a sig-probe
 * FmhaKernelBuilder ("rocke_sage_attention_sig_probe") via _declare_params and
 * emit its SignatureBuilder shape. On ROCKE_OK *out_items / *out_count hold the
 * arena-owned manifest; `arena` backs the SignatureBuilder storage. The probe
 * builder is created + freed internally. */
rocke_status_t rocke_sage_attention_signature(const rocke_sage_attention_spec_t* spec,
                                              rocke_arena_t* arena,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count);

/* ------------------------------------------------------------------ *
 * convenience lower-to-.ll
 * ------------------------------------------------------------------ *
 *
 * Given a spec, init a builder, build (via the same dispatch as
 * rocke_build_sage_attention), and lower to LLVM .ll text. `arch` NULL =>
 * "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err != NULL,
 * capacity err_cap) a diagnostic is written. Internally owns + frees its
 * builder. */
rocke_status_t rocke_sage_attention_lower_to_llvm(const rocke_sage_attention_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_SAGE_ATTENTION_H */
