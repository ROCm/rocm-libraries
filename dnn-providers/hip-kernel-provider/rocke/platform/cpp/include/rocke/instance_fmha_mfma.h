/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_mfma.h -- C99 port of the tiled FMHA forward kernel instance
 * builder rocke/instances/common/fmha_mfma.py.
 *
 * One CTA = one wave handles a BLOCK_M=16-row Q tile across the full K span via
 * the QK + softmax + PV matmul chain. The single inner body is arch-polymorphic:
 * the 16x16x16 f16 atom resolves to mfma_f32_16x16x16_f16 on CDNA wave64
 * (gfx942/gfx950) and wmma_f32_16x16x16_f16 on RDNA wave32 (gfx1151). The heavy
 * lifting lives in the already-ported helper
 * rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body; this module is the
 * thin spec->kernel wrapper.
 *
 *   Python (fmha_mfma.py)               C99 (this header)
 *   ----------------------------------  --------------------------------------
 *   @dataclass FmhaMfmaSpec             rocke_fmha_mfma_spec_t
 *     .kernel_name()                    rocke_fmha_mfma_kernel_name(...)
 *   _mma_family(arch)                   rocke_fmha_mfma_mma_family(...)
 *   is_valid_spec(spec, arch)           rocke_fmha_mfma_is_valid_spec(...)
 *   build_fmha_fwd_mfma(spec, arch)     rocke_build_fmha_fwd_mfma(...)
 *   fmha_fwd_mfma_grid(spec, batch)     rocke_fmha_fwd_mfma_grid(...)
 *   fmha_fwd_mfma_signature(spec)       rocke_fmha_fwd_mfma_signature(...)
 *   (+ convenience: build -> lower .ll) rocke_fmha_fwd_mfma_lower_to_llvm(...)
 *
 * SPEC AS A FLAT C STRUCT. The Python spec composes an FmhaCommonSpec (which in
 * turn composes an FmhaShape). The C entry takes a flat rocke_fmha_mfma_spec_t with
 * the head-shape + mask + dtype + seqlen fields the caller cares about; the build
 * routine assembles the equivalent rocke_fmha_common_spec_t / rocke_fmha_shape_t
 * internally so the helper-driven IR emission is byte-identical to the Python
 * path. dtype is hardcoded to "f16" in v1 (bf16 lands once the bf16 atom factory
 * is exposed -- see is_valid_spec).
 *
 * Error model mirrors the rest of the C port: the build routine routes errors
 * through the sticky-error IRBuilder; the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_MFMA_H
#define ROCKE_INSTANCE_FMHA_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* mask mode enum */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ FmhaMfmaSpec
 *
 * Flat mirror of FmhaMfmaSpec(common: FmhaCommonSpec, seqlen_q, seqlen_k, name).
 * The head shape + mask + dtype fields are the FmhaCommonSpec / FmhaShape fields
 * the build routine reconstitutes. `dtype` is "f16" only in v1. `mask_mode` is the
 * shared FMHA mask enum ("none"/"causal"/"sliding_window" map; alibi/custom are
 * rejected by validate). `name` defaults to "rocke_fmha_fwd_mfma" when NULL. */
typedef struct rocke_fmha_mfma_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    int seqlen_q;
    int seqlen_k;
    const char* dtype; /* NULL => "f16" (v1 supports f16 only)  */
    rocke_fmha_mask_mode_t mask_mode; /* ROCKE_FMHA_MASK_NONE default            */
    int sliding_window; /* default 0                             */
    double scale_log2; /* default 0.0                           */
    const char* name; /* NULL => "rocke_fmha_fwd_mfma"        */
} rocke_fmha_mfma_spec_t;

/* Default-constructed spec (Python dataclass defaults). The caller must still set
 * the required head-shape + seqlen fields. */
rocke_fmha_mfma_spec_t rocke_fmha_mfma_spec_default(void);

/* FmhaMfmaSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", mask_mode). Writes NUL-terminated into out
 * (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t
    rocke_fmha_mfma_kernel_name(const rocke_fmha_mfma_spec_t* spec, char* out, size_t out_cap);

/* _mma_family(arch): "wmma" on RDNA wave32 (gfx11xx), "mma" (MFMA) on CDNA.
 * Returns a static string; "mma" for an unknown/NULL arch (its wave_size is not
 * 32). `arch` NULL => "gfx950". */
const char* rocke_fmha_mfma_mma_family(const char* arch);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On reject
 * `reason` (if non-NULL, capacity reason_cap) receives the structured message and
 * the function returns false; on accept it returns true and writes "ok". */
bool rocke_fmha_mfma_is_valid_spec(const rocke_fmha_mfma_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* build_fmha_fwd_mfma(spec, arch). Validates, then builds the tiled FMHA forward
 * IR (one wave per CTA; MFMA on CDNA / WMMA on RDNA) and returns the kernel def.
 * `arch` NULL => "gfx950". On an invalid spec or any IR-emission error returns
 * NULL; if `b` is non-NULL its sticky error carries the diagnostic. Internally
 * owns an FmhaKernelBuilder; the returned kernel is owned by `b` when supplied --
 * see rocke_build_fmha_fwd_mfma_new for the self-contained variant.
 *
 * `b` is the destination IR builder to emit into. If `b` is NULL, the function
 * allocates a transient builder, builds, and returns a kernel that is COPIED into
 * a freshly malloc'd rocke_kernel_def_t the caller frees with
 * rocke_kernel_def_free()... (see note). For parity with the other instance entry
 * points and the documented CALL PATTERN, prefer passing a live `b`. */
rocke_kernel_def_t* rocke_build_fmha_fwd_mfma(rocke_ir_builder_t* b,
                                              const rocke_fmha_mfma_spec_t* spec,
                                              const char* arch);

/* fmha_fwd_mfma_grid(spec, batch) -> (seqlen_q/BLOCK_M, num_query_heads, batch).
 * Writes the three axes to out[0..2]; `out` must hold 3 ints. */
void rocke_fmha_fwd_mfma_grid(const rocke_fmha_mfma_spec_t* spec, int batch, int out[3]);

/* fmha_fwd_mfma_signature(spec): the kernel ABI signature (Q/K/V/O ptrs,
 * scale_log2/seqlen_q/seqlen_k scalars, q/k/v/o stride pairs). On ROCKE_OK
 * *out_items / *out_count hold the arena-owned array; `arena` backs the storage.
 * On failure the out-params are untouched and the status is returned. */
rocke_status_t rocke_fmha_fwd_mfma_signature(const rocke_fmha_mfma_spec_t* spec,
                                             rocke_arena_t* arena,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count);

/* Convenience: build the kernel and lower it to LLVM .ll text. `arch` NULL =>
 * "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns and frees its IRBuilder. */
rocke_status_t rocke_fmha_fwd_mfma_lower_to_llvm(const rocke_fmha_mfma_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_MFMA_H */
