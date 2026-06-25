/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_flatmm.h -- C99 port of the FlatMM kernel instance builder
 * ck_dsl/instances/common/flatmm.py (CK Tile 18_flatmm parity).
 *
 * FlatMM is a thin wrapper around batched_gemm (which itself delegates to
 * build_universal_gemm with batched=True). The v1 kernel body is shared
 * verbatim; FlatMM only carries the dispatch knobs plus two FlatMM-specific
 * spec fields (preshuffle_b, name) and the host-side preshuffled-B layout
 * helpers.
 *
 *   Python (flatmm.py)                     C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   class FlatMMSpec                       ckc_flatmm_spec_t
 *   FlatMMSpec.to_batched_spec()           ckc_flatmm_to_batched_spec()  (-> universal)
 *   FlatMMSpec.kernel_name()               ckc_flatmm_kernel_name()
 *   flatmm_config32(dtype)                 ckc_flatmm_config32()
 *   flatmm_config16(dtype)                 ckc_flatmm_config16()
 *   is_valid_spec(spec, arch)              ckc_flatmm_is_valid_spec()
 *   build_flatmm(spec, arch)               ckc_build_flatmm()
 *   flatmm_grid(spec, batch, m, n)         ckc_flatmm_grid()
 *   flatmm_signature(spec)                 ckc_flatmm_signature()
 *   flatmm_atom_shape(spec)                ckc_flatmm_atom_shape()
 *   flatmm_atom(spec)                      ckc_flatmm_atom()
 *   flatmm_preshuffle_b_spec(spec)         ckc_flatmm_preshuffle_b_spec()
 *   flatmm_preshuffle_b_layout(spec,n,k)   ckc_flatmm_preshuffle_b_layout()
 *   (+ convenience: build -> lower .ll)    ckc_flatmm_lower_to_llvm()
 *
 * The re-exported types (DataSpec, TileSpec, TraitSpec, UniversalGemmSpec) come
 * from ckc/instance_gemm_universal.h; this header includes it so callers get
 * them for free, mirroring flatmm.py's __all__ re-exports.
 *
 * Since batched_gemm has no standalone C port, this TU inlines its (very small)
 * conversion logic: a BatchedGemmSpec becomes a UniversalGemmSpec with
 * batched=True and the f16/fp16 dtype canonicalised. FlatMM's v1 body is then
 * ckc_build_universal_gemm.
 *
 * Error model mirrors the rest of the C port.
 */
#ifndef CKC_INSTANCE_FLATMM_H
#define CKC_INSTANCE_FLATMM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_gemm_universal.h"          /* re-exported TileSpec/TraitSpec/... */
#include "ckc/helper_ck_dsl.helpers.atoms.h"      /* ckc_mfma_atom_t */
#include "ckc/helper_ck_dsl.helpers.preshuffle.h" /* ckc_preshuffleb_spec_t */
#include "ckc/helper_ck_dsl.helpers.spec.h"       /* ckc_sig_entry_t, arena */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ FlatMMSpec *
 *
 * Mirror of Python FlatMMSpec (frozen dataclass + WarpTileBlockSizeMixin).
 * Mirrors BatchedGemmSpec (since the v1 body is shared) plus two FlatMM extras:
 *   - name defaults to "ck_dsl_flatmm"
 *   - preshuffle_b (default false), rejected at build time until the v2 body.
 *
 * Field declaration order is 1:1 with the Python dataclass:
 *   tile, trait, wave_size, block_size, batch_size, preshuffle_b, name.
 *
 * block_size==0 => derived at finalize() (warp_m*warp_n*warp_k*wave_size). */
typedef struct ckc_flatmm_spec
{
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;  /* default 64 */
    int block_size; /* default 0 => derived at finalize() */
    int batch_size; /* default 0 */
    bool preshuffle_b;
    const char* name; /* default "ck_dsl_flatmm" */
} ckc_flatmm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set the required `tile` geometry (and may override trait/name). */
ckc_flatmm_spec_t ckc_flatmm_spec_default(void);

/* WarpTileBlockSizeMixin._init_block_size(): when block_size==0, derive it as
 * warp_m*warp_n*warp_k*wave_size. Idempotent. Call after filling the spec. */
void ckc_flatmm_spec_finalize(ckc_flatmm_spec_t* spec);

/* FlatMMSpec.to_batched_spec() composed with BatchedGemmSpec.to_universal_spec():
 * builds the equivalent UniversalGemmSpec (batched=True, dtype fp16) used for
 * the v1 body. The kernel-name prefix is `name` + ("_psb" if preshuffle_b).
 * The returned spec is finalized. Returns CKC_OK, or CKC_ERR_VALUE on a NULL
 * argument. */
ckc_status_t ckc_flatmm_to_universal_spec(const ckc_flatmm_spec_t* spec,
                                          ckc_gemm_universal_spec_t* out);

/* FlatMMSpec.kernel_name() -> NUL-terminated into out (capacity out_cap). */
ckc_status_t ckc_flatmm_kernel_name(const ckc_flatmm_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------ spec convenience constructors *
 *
 * flatmm_config32 / flatmm_config16 mirrors of CK Tile FlatmmConfig32/16.
 * `dtype` must be one of "f16"/"fp16"/"bf16" (the Python ValueError otherwise).
 * On the reject path returns CKC_ERR_VALUE leaving *out untouched; otherwise
 * writes the TileSpec preset and returns CKC_OK. `dtype` NULL => "f16". */
ckc_status_t ckc_flatmm_config32(const char* dtype, ckc_gemm_tile_spec_t* out);
ckc_status_t ckc_flatmm_config16(const char* dtype, ckc_gemm_tile_spec_t* out);

/* is_valid_spec(spec, arch). `arch` NULL => "gfx950". Returns false (and writes
 * the structured reason into `reason`, capacity reason_cap, if non-NULL) when
 * preshuffle_b is set or the underlying batched/universal spec is invalid;
 * returns true and writes "ok" on accept. */
bool ckc_flatmm_is_valid_spec(const ckc_flatmm_spec_t* spec,
                              const char* arch,
                              char* reason,
                              size_t reason_cap);

/* build_flatmm(spec, arch): validate then build the v1 body via
 * ckc_build_universal_gemm. Builds into the supplied (already
 * ckc_ir_builder_init'd) builder `b` and returns the kernel or NULL with the
 * sticky error set. `arch` NULL => "gfx950". Does NOT re-init the builder.
 *
 * NOTE: like the Python, this expects the builder to have been created with the
 * spec's kernel_name(). Use ckc_build_flatmm_new() for the init-from-spec
 * convenience. */
ckc_kernel_def_t*
ckc_build_flatmm(ckc_ir_builder_t* b, const ckc_flatmm_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
ckc_kernel_def_t*
ckc_build_flatmm_new(ckc_ir_builder_t* b, const ckc_flatmm_spec_t* spec, const char* arch);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * On CKC_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 * with free(); on failure it is left NULL and a diagnostic is written into
 * `err` (capacity err_cap). Internally owns and frees its IRBuilder. */
ckc_status_t ckc_flatmm_lower_to_llvm(const ckc_flatmm_spec_t* spec,
                                      const char* arch,
                                      ckc_llvm_flavor_t flavor,
                                      char** out_ll,
                                      char* err,
                                      size_t err_cap);

/* flatmm_grid(spec, batch, m, n): same launch grid as build_batched_gemm.
 * On success out[0..2] hold (x, y, z) = ceil_div over (n,tile_n),(m,tile_m),
 * (batch,1). Returns CKC_ERR_VALUE on the Python ValueError path. */
ckc_status_t ckc_flatmm_grid(const ckc_flatmm_spec_t* spec, int batch, int m, int n, int out[3]);

/* flatmm_signature(spec): manifest-style signature mirroring
 * batched_gemm_signature. Builds the entry array into the caller-provided
 * `arena` (which must outlive the returned array) and sets *out_items /
 * *out_count to the (arena-owned) read-only array:
 *   A,B,C : ptr fp16 ; M,N,K,stride_a,stride_b,stride_c : i32
 *   (+ SortedTokenIds: ptr i32, slot_size: i32 when trait.active_tile_skip).
 * Returns CKC_OK, or an error status on a NULL argument / arena OOM. */
ckc_status_t ckc_flatmm_signature(const ckc_flatmm_spec_t* spec,
                                  ckc_arena_t* arena,
                                  const ckc_sig_entry_t** out_items,
                                  size_t* out_count);

/* ----------------------------------------------- tile-level introspection *
 *
 * flatmm_atom_shape(spec) -> (warp_tile_m, warp_tile_n, warp_tile_k). */
void ckc_flatmm_atom_shape(const ckc_flatmm_spec_t* spec, int out_mnk[3]);

/* flatmm_atom(spec) -> the MfmaAtom mfma_atom("f16", m, n, k) resolves to.
 * Returns a pointer into the static catalog (do NOT free), or NULL on a miss
 * (the Python ValueError path) / NULL spec. */
const ckc_mfma_atom_t* ckc_flatmm_atom(const ckc_flatmm_spec_t* spec);

/* flatmm_preshuffle_b_spec(spec) -> PreshuffleBSpec{block_n=tile_n,
 * block_k=tile_k, elem_bytes=2}. Returns CKC_ERR_VALUE on a NULL argument. */
ckc_status_t ckc_flatmm_preshuffle_b_spec(const ckc_flatmm_spec_t* spec,
                                          ckc_preshuffleb_spec_t* out);

/* flatmm_preshuffle_b_layout(spec, n=n, k=k): host-side preshuffled-B layout
 * (shape, strides), each a 4-element array (any may be NULL to skip). Wraps
 * ckc_host_preshuffle_layout with the spec-derived PreshuffleBSpec. Returns
 * CKC_ERR_VALUE on the divisibility ValueError path (out arrays untouched). */
ckc_status_t ckc_flatmm_preshuffle_b_layout(
    const ckc_flatmm_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_FLATMM_H */
