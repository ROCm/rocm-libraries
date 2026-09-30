/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common.fmha_arch.h -- C99 port of the two
 * arch-awareness symbols from rocke/instances/common/fmha_arch.py:
 *
 *   Python (rocke.instances.common.fmha_arch)   C99 (this header)
 *   ------------------------------------------    ----------------------------
 *   FMHA_MFMA_ATTN_BLOCK = 16                     ROCKE_FMHA_MFMA_ATTN_BLOCK
 *   validate_fmha_mfma_atom(dtype, arch)          rocke_validate_fmha_mfma_atom()
 *
 * The Python helper sources its legal atom set from rocke.core.arch.ArchTarget
 * (the same catalog gemm_universal.is_valid_spec consults) and returns a
 * structured (ok, reason). This port binds to the canonical arch port via
 * rocke/helper_rocke.core.arch.h: ArchTarget.from_gfx -> rocke_archtarget_from_gfx,
 * target.mma.has_shape(...) is reproduced via rocke_archtarget_op_for_shape (the
 * shape predicate has_shape(m,n,k) is true iff op_for_shape(...,k) is non-NULL,
 * since enumerate() already filters on m,n and has_shape compares full shape).
 *
 * The Python (ok, reason) tuple is mapped to a `bool` return plus a
 * caller-provided reason buffer (out / out_cap), following the sibling-helper
 * convention. The reason text is reproduced byte-faithfully, including the
 * KeyError str() double-quote wrapping for an unknown arch.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_FMHA_ARCH_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_FMHA_ARCH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The narrow attention MFMA tile both attention helpers pin (the K-dim of the
 * QK / PV atom is 16; head_size is sliced into 16-wide atoms).
 *
 *   Python: FMHA_MFMA_ATTN_BLOCK = 16
 */
#define ROCKE_FMHA_MFMA_ATTN_BLOCK 16

/* validate_fmha_mfma_atom(dtype, arch="gfx950") -> (ok, reason).
 *
 * Returns the boolean `ok`; writes the matching `reason` string into `out`
 * (NUL-terminated, truncated to out_cap if needed -- on truncation the textual
 * reason is clipped but the boolean result is unaffected). `out` may be NULL
 * (with out_cap 0) when the caller only wants the boolean.
 *
 *   - dtype in {"f16","fp16"} -> catalog dtype "f16"
 *   - dtype == "bf16"          -> catalog dtype "bf16"
 *   - otherwise: returns false, reason
 *       "unsupported FMHA dtype '<dtype>' for MFMA atom selection"
 *
 * `arch` NULL is treated as the Python default "gfx950".
 *
 * On unknown arch: returns false, reason is str(KeyError(...)) i.e. the
 * _build_target message wrapped in double quotes:
 *   "unknown gfx target '<arch>'; known: ['..','..']. Add a row to
 *    arch_specs.json."
 *
 * On a dtype with no narrow 16x16x16 (a,a,fp32) atom in the target catalog:
 *   returns false, reason
 *     "FMHA MFMA atom <a> 16x16x16 not in <arch> catalog"
 *
 * Otherwise: returns true, reason "ok".
 */
bool rocke_validate_fmha_mfma_atom(const char* dtype, const char* arch, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_FMHA_ARCH_H */
