/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx950_attention_tiled_2d.h -- PUBLIC C99 surface for the gfx950
 * (CDNA4 / MI355X) WIDE-ATOM tiled-2D unified-attention kernel INSTANCE BUILDER
 * (rocke/instances/gfx950/attention_tiled_2d.py, 3818 LOC).
 *
 *   Python (gfx950/attention_tiled_2d.py)            C99 (this header)
 *   ------------------------------------------- --------------------------------------------------
 *   class UnifiedAttention2DTiledSpec                rocke_attention_tiled_2d_spec_t  (ALREADY
 * PORTED: (frozen dataclass + __post_init__)               helper_helper_rocke.instances.gfx942.
 *                                                       attention_tiled_2d.h -- arch-parameterised,
 *                                                       reused verbatim here)
 *   spec.@property accessors                         rocke_attention_tiled_2d_spec_*  (same helper
 * hdr) _mfma_32x32_c_row / _mfma_32x32_c_col rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row/_col
 *   supports_tiled_2d(..., arch="gfx950")            rocke_gfx950_attention_tiled_2d_supports
 *   build_unified_attention_2d_tiled(spec, arch)     rocke_gfx950_build_unified_attention_2d_tiled
 *   (+ convenience: build -> lower .ll)              rocke_gfx950_attention_tiled_2d_lower_to_llvm
 *
 * WHY A SEPARATE gfx950 BUILDER (vs the gfx942 sibling).
 *   The two arches share the SPEC dataclass, the @property accessors, the
 *   __post_init__ validator, the config-from-spec derivation, and the
 *   supports_tiled_2d *signature* -- those are arch-parameterised and were
 *   ported once (helper_helper_rocke.instances.gfx942.attention_tiled_2d.h).
 *   The KERNEL EMITTER, however, is NOT interchangeable:
 *
 *     - gfx950 uses the WIDE-K MFMA atoms mfma_f32_16x16x32 / mfma_f32_32x32x16
 *       (rocke_mfma_32x32x16_for_dtype, Python lines 813-814); gfx942 uses the
 *       narrow 16x16x16 / 32x32x8 geometry. The two are NOT selectable from one
 *       another -- comgr aborts with "Cannot select intrinsic" on the wrong arch.
 *     - gfx950 reads the PV V B-operand through CK-Tile LDS transpose reads
 *       ds_read_b64_tr_b16 / ds_read_b64_tr_b8 (TransposeLdsReader, line 1128 /
 *       930); gfx942 has no ds_read_tr16_b64 and reconstructs the operand from
 *       strided scalar LDS reads (its _strided_v_b_operand / cfv / cfvst paths).
 *     - The gfx950 closure set is a strict SUBSET of gfx942's: it has NO
 *       transposed-V-store family (_issue_v_transposed / _cfvst_* / _v_t_slot /
 *       _v_t_store / _v_t_load / _v_load1), NO K-sliced-ring (_issue_k_slice /
 *       _kslot), and NO _strided_v_b_operand, because the wide transpose reads
 *       replace all of them.
 *     - The P->A permute on gfx950 maps the wide 32x32 C distribution directly to
 *       the PV MFMA A operand via xor shuffles (lines 1292-1351).
 *
 *   Every NEW symbol this builder family adds carries the
 *   ``rocke_gfx950_attention_tiled_2d_`` prefix so it never clashes with the gfx942
 *   sibling (same simple file/symbol stems), nor with the common/ instances
 *   (instances/common/attention_unified*, the fmha_* family). The internal
 *   build-context + phase-function contract lives in the sibling PRIVATE header
 *   rocke/instance_gfx950_attention_tiled_2d_internal.h; public callers only ever
 *   touch THIS header.
 *
 * REUSED, ALREADY-PORTED INFRASTRUCTURE (do NOT re-port; #include only):
 *   - the spec struct + @property + __post_init__ + config-from-spec
 *       (rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h)
 *   - rocke_mfma_32x32x16_for_dtype, rocke_mfma_16x16x32_for_dtype, the warp xor
 *       reduces, rocke_dequant_fp8x8_to_dtype, pv32_v_load_paired
 *       (rocke/helper_rocke.helpers.attention.h, helper_rocke.helpers.mfma_attention.h)
 *   - rocke_transpose_lds_reader_* (TransposeLdsReader)
 *       (rocke/helper_rocke.helpers.layouts.h)
 *   - the arch gate require_tiled_attention_arch / validate_tiled_attention_arch
 *       (instances/common/attention_arch.py -> rocke/helper_rocke.core.arch.h)
 *   - make_static_tile_distribution / make_c_warp_dstr_encoding
 *       (rocke/helper_rocke.helpers.distribution.h, helper_rocke.helpers.atoms.h)
 *
 * THE KERNEL (per CTA = NUM_WARPS wave64s, each warp owns a 16/32-row M slice).
 *   1. AITER binary-search on cu_q to resolve seq_idx; compute the Q-block-local
 *      index; early-return for a padding block.
 *   2. Cooperatively stage Q[BLOCK_M, HD] global -> LDS (zero-fill padding rows /
 *      out-of-range heads), then gather the per-lane Q MFMA A-operand to VGPRs.
 *   3. Loop over KV tiles: resolve physical_block via block_tables; async-DMA K
 *      (double-buffered) and V (single-buffered) cache -> LDS; compute S = Q @ K^T
 *      with the WIDE atom (16x16x32, or 32x32x16 on use_mfma_32x32, with the
 *      transposed S^T = K @ Q^T orientation on use_transposed_qk_32x32); apply
 *      qk_scale / optional softcap / causal+SW+padding mask; run the online
 *      softmax (per-row max/sum via ds_bpermute butterfly); publish P to P_lds
 *      (or keep P^T in registers on the register-P / transposed paths).
 *   4. PV: acc *= alpha; acc += P @ V via the wide atom, with the V B-operand read
 *      through TransposeLdsReader ds_read_b64_tr_b16/b8. Native-fp8 PV keeps V raw
 *      in LDS and dequants in register.
 *   5. Normalise acc /= L and store output[BLOCK_M, HD] via the Acc_lds stripe
 *      cooperative store (or the 32x32 direct/coalesced scalar store).
 *
 * ENTRY RETURN. rocke_gfx950_build_unified_attention_2d_tiled returns the built
 * kernel (b->kernel == Python ``b.kernel``) on success, or NULL with b's sticky
 * error set. It mirrors build_unified_attention_2d_tiled: require_tiled_attention_
 * arch(arch) (line 748) -> dtype gate (fp16/bf16, line 750) -> wide-atom select ->
 * derive config -> emit. ``arch`` NULL == the Python default "gfx950".
 *
 * Error model mirrors the rest of the C port: build/lower route errors through the
 * sticky-error IRBuilder; the supports gate returns a bool + reason string; the
 * convenience lower returns a rocke_status_t. Every IR node is arena-owned.
 */
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_kernel_def_t, rocke_status_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t                                 */
/* The already-ported spec struct + @property accessors + __post_init__ validator
 * + config-from-spec. Arch-parameterised; REUSED, not redeclared. */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * Module-level 32x32 C-distribution helpers (Python lines 79-133)
 * ============================================================ *
 *
 * _mfma_32x32_c_row(b, lane, elem_idx) / _mfma_32x32_c_col(b, lane, n_tile32):
 * the per-lane output row/col of CK Tile's CWarpDstrEncoding for the f16
 * 32x32x16 atom, driven by the module-static _C32_DIST (Python line 97). Pure IR
 * emitters used by the 32x32 epilogue and P-permute. The C distribution is
 * dtype-independent (the f16 atom drives both fp16 and bf16). ``elem_idx`` must be
 * 0..15; ``n_tile32`` >= 0. These do not exist on gfx942. */
rocke_value_t* rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(rocke_ir_builder_t* b,
                                                                rocke_value_t* lane,
                                                                int elem_idx);
rocke_value_t* rocke_gfx950_attention_tiled_2d_mfma_32x32_c_col(rocke_ir_builder_t* b,
                                                                rocke_value_t* lane,
                                                                int n_tile32);

/* ============================================================ *
 * supports_tiled_2d(..., arch="gfx950")   (Python lines 588-703)
 * ============================================================ *
 *
 * The arch / dtype / geometry / per-wave-DMA admission gate. Faithful port of the
 * keyword-only Python supports_tiled_2d: the arch gate
 * (validate_tiled_attention_arch), the dtype (fp16/bf16) check, the
 * head_size in {64,128,256} & %32 check, block_size in {16,32,64}, the
 * num_queries_per_kv range + BLOCK_M divisibility, the fp8 K/V-cache pairing
 * (kv_storage_dtype="fp8e4m3" + use_fp8), the q_dtype check, and the tile_size /
 * THREADS*8 payload / per-wave-token constraints. The trailing
 * use_mfma_32x32x8 / use_transposed_qk_32x32 / use_k_single_buffer /
 * use_conflict_free_v_store / use_k_sliced_ring args are accepted for signature
 * parity with the shared dispatch caller and the gfx942 gate but do NOT key
 * gfx950 admission (Python lines 609-616). Returns true + writes "supported" into
 * reason on accept; on reject returns false and writes the structured Python
 * reason string. ``reason`` may be NULL. Optional[int]/Optional[str] use a
 * has_<x> flag / a NULL pointer == Python None (q_dtype NULL == None,
 * kv_storage_dtype NULL == None). ``arch`` NULL == "gfx950". Pure: emits no IR. */
typedef struct rocke_gfx950_attention_tiled_2d_supports_args
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
    const char* arch; /* NULL == "gfx950"                           */
    bool use_mfma_32x32x8; /* accepted for parity; ignored on gfx950     */
    bool use_transposed_qk_32x32; /* accepted for parity; ignored on gfx950     */
    bool use_k_single_buffer; /* K single-buffer (T=64 2-WG/CU d128)        */
    bool use_conflict_free_v_store; /* accepted for parity; ignored on gfx950     */
    bool use_k_sliced_ring; /* accepted for parity; ignored on gfx950     */
} rocke_gfx950_attention_tiled_2d_supports_args_t;

/* Defaulted args (num_warps=1, block_m_per_warp=16, everything else 0/NULL/
 * false). The caller sets head_size/block_size/dtype/num_queries_per_kv. */
rocke_gfx950_attention_tiled_2d_supports_args_t
    rocke_gfx950_attention_tiled_2d_supports_args_default(void);

bool rocke_gfx950_attention_tiled_2d_supports(
    const rocke_gfx950_attention_tiled_2d_supports_args_t* args, char* reason, size_t reason_cap);

/* ============================================================ *
 * build_unified_attention_2d_tiled(spec, arch="gfx950")   (lines 711-3818)
 * ============================================================ *
 *
 * Emit the gfx950 wide-atom tiled MFMA 2D unified-attention kernel into the
 * supplied (already rocke_ir_builder_init'd, named spec.kernel_name()) builder
 * ``b`` and return the kernel (b->kernel) on success, or NULL with b's sticky
 * error set. ``arch`` NULL == the Python default "gfx950".
 *
 * This is the REAL emitter: the ~3100-line body is split across the
 * instance_gfx950_attention_tiled_2d_*.c translation units, driven through the
 * build-context + phase-function contract in
 * rocke/instance_gfx950_attention_tiled_2d_internal.h. It reproduces the Python
 * prologue -> Q-load/gather -> KV-loop (QK / mask / softmax / PV) -> epilogue
 * order byte-faithfully. */
rocke_kernel_def_t* rocke_gfx950_build_unified_attention_2d_tiled(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch);

/* Convenience: init ``b`` with spec.kernel_name(), then build. The caller owns
 * ``b`` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL with
 * b's sticky error set. ``arch`` NULL == "gfx950". */
rocke_kernel_def_t* rocke_gfx950_build_unified_attention_2d_tiled_new(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * ``arch`` NULL == "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left NULL
 * and (if err!=NULL, capacity err_cap) a diagnostic is written. Internally owns
 * and frees its IRBuilder. */
rocke_status_t
    rocke_gfx950_attention_tiled_2d_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX950_ATTENTION_TILED_2D_H */
