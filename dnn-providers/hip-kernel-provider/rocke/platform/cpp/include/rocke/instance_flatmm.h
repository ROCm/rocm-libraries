/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_flatmm.h -- C99 port of the FlatMM kernel instance builder
 * rocke/instances/common/flatmm.py (CK Tile 18_flatmm parity).
 *
 * FlatMM is a thin wrapper around batched_gemm (which itself delegates to
 * build_universal_gemm with batched=True). The v1 kernel body is shared
 * verbatim; FlatMM only carries the dispatch knobs plus two FlatMM-specific
 * spec fields (preshuffle_b, name) and the host-side preshuffled-B layout
 * helpers.
 *
 *   Python (flatmm.py)                     C99 (this header)
 *   -----------------------------------    --------------------------------------
 *   class FlatMMSpec                       rocke_flatmm_spec_t
 *   FlatMMSpec.to_batched_spec()           rocke_flatmm_to_batched_spec()  (-> universal)
 *   FlatMMSpec.kernel_name()               rocke_flatmm_kernel_name()
 *   flatmm_config32(dtype)                 rocke_flatmm_config32()
 *   flatmm_config16(dtype)                 rocke_flatmm_config16()
 *   is_valid_spec(spec, arch)              rocke_flatmm_is_valid_spec()
 *   build_flatmm(spec, arch)               rocke_build_flatmm()
 *   flatmm_grid(spec, batch, m, n)         rocke_flatmm_grid()
 *   flatmm_signature(spec)                 rocke_flatmm_signature()
 *   flatmm_atom_shape(spec)                rocke_flatmm_atom_shape()
 *   flatmm_atom(spec)                      rocke_flatmm_atom()
 *   flatmm_preshuffle_b_spec(spec)         rocke_flatmm_preshuffle_b_spec()
 *   flatmm_preshuffle_b_layout(spec,n,k)   rocke_flatmm_preshuffle_b_layout()
 *   (+ convenience: build -> lower .ll)    rocke_flatmm_lower_to_llvm()
 *
 * The re-exported types (DataSpec, TileSpec, TraitSpec, UniversalGemmSpec) come
 * from rocke/instance_gemm_universal.h; this header includes it so callers get
 * them for free, mirroring flatmm.py's __all__ re-exports.
 *
 * Since batched_gemm has no standalone C port, this TU inlines its (very small)
 * conversion logic: a BatchedGemmSpec becomes a UniversalGemmSpec with
 * batched=True and the f16/fp16 dtype canonicalised. FlatMM's v1 body is then
 * rocke_build_universal_gemm.
 *
 * Error model mirrors the rest of the C port.
 */
#ifndef ROCKE_INSTANCE_FLATMM_H
#define ROCKE_INSTANCE_FLATMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom_t */
#include "rocke/helper_rocke.helpers.preshuffle.h" /* rocke_preshuffleb_spec_t */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t, arena */
#include "rocke/instance_gemm_universal.h" /* re-exported TileSpec/TraitSpec/... */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ FlatMMSpec *
 *
 * Mirror of Python FlatMMSpec (frozen dataclass + WarpTileBlockSizeMixin).
 * Mirrors BatchedGemmSpec (since the v1 body is shared) plus two FlatMM extras:
 *   - name defaults to "rocke_flatmm"
 *   - preshuffle_b (default false), rejected at build time until the v2 body.
 *
 * Field declaration order is 1:1 with the Python dataclass:
 *   tile, trait, wave_size, block_size, batch_size, preshuffle_b, name.
 *
 * block_size==0 => derived at finalize() (warp_m*warp_n*warp_k*wave_size). */
typedef struct rocke_flatmm_spec
{
    rocke_gemm_tile_spec_t tile;
    rocke_gemm_trait_spec_t trait;
    int wave_size; /* default 64 */
    int block_size; /* default 0 => derived at finalize() */
    int batch_size; /* default 0 */
    bool preshuffle_b;
    const char* name; /* default "rocke_flatmm" */
} rocke_flatmm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set the required `tile` geometry (and may override trait/name). */
rocke_flatmm_spec_t rocke_flatmm_spec_default(void);

/* WarpTileBlockSizeMixin._init_block_size(): when block_size==0, derive it as
 * warp_m*warp_n*warp_k*wave_size. Idempotent. Call after filling the spec. */
void rocke_flatmm_spec_finalize(rocke_flatmm_spec_t* spec);

/* FlatMMSpec.to_batched_spec() composed with BatchedGemmSpec.to_universal_spec():
 * builds the equivalent UniversalGemmSpec (batched=True, dtype fp16) used for
 * the v1 body. The kernel-name prefix is `name` + ("_psb" if preshuffle_b).
 * The returned spec is finalized. Returns ROCKE_OK, or ROCKE_ERR_VALUE on a NULL
 * argument. */
rocke_status_t rocke_flatmm_to_universal_spec(const rocke_flatmm_spec_t* spec,
                                              rocke_gemm_universal_spec_t* out);

/* FlatMMSpec.kernel_name() -> NUL-terminated into out (capacity out_cap). */
rocke_status_t rocke_flatmm_kernel_name(const rocke_flatmm_spec_t* spec, char* out, size_t out_cap);

/* ------------------------------------------------ spec convenience constructors *
 *
 * flatmm_config32 / flatmm_config16 mirrors of CK Tile FlatmmConfig32/16.
 * `dtype` must be one of "f16"/"fp16"/"bf16" (the Python ValueError otherwise).
 * On the reject path returns ROCKE_ERR_VALUE leaving *out untouched; otherwise
 * writes the TileSpec preset and returns ROCKE_OK. `dtype` NULL => "f16". */
rocke_status_t rocke_flatmm_config32(const char* dtype, rocke_gemm_tile_spec_t* out);
rocke_status_t rocke_flatmm_config16(const char* dtype, rocke_gemm_tile_spec_t* out);

/* is_valid_spec(spec, arch). `arch` NULL => "gfx950". Returns false (and writes
 * the structured reason into `reason`, capacity reason_cap, if non-NULL) when
 * preshuffle_b is set or the underlying batched/universal spec is invalid;
 * returns true and writes "ok" on accept. */
bool rocke_flatmm_is_valid_spec(const rocke_flatmm_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t reason_cap);

/* build_flatmm(spec, arch): validate then build the v1 body via
 * rocke_build_universal_gemm. Builds into the supplied (already
 * rocke_ir_builder_init'd) builder `b` and returns the kernel or NULL with the
 * sticky error set. `arch` NULL => "gfx950". Does NOT re-init the builder.
 *
 * NOTE: like the Python, this expects the builder to have been created with the
 * spec's kernel_name(). Use rocke_build_flatmm_new() for the init-from-spec
 * convenience. */
rocke_kernel_def_t*
    rocke_build_flatmm(rocke_ir_builder_t* b, const rocke_flatmm_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_flatmm_new(rocke_ir_builder_t* b,
                                           const rocke_flatmm_spec_t* spec,
                                           const char* arch);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 * with free(); on failure it is left NULL and a diagnostic is written into
 * `err` (capacity err_cap). Internally owns and frees its IRBuilder. */
rocke_status_t rocke_flatmm_lower_to_llvm(const rocke_flatmm_spec_t* spec,
                                          const char* arch,
                                          rocke_llvm_flavor_t flavor,
                                          char** out_ll,
                                          char* err,
                                          size_t err_cap);

/* flatmm_grid(spec, batch, m, n): same launch grid as build_batched_gemm.
 * On success out[0..2] hold (x, y, z) = ceil_div over (n,tile_n),(m,tile_m),
 * (batch,1). Returns ROCKE_ERR_VALUE on the Python ValueError path. */
rocke_status_t
    rocke_flatmm_grid(const rocke_flatmm_spec_t* spec, int batch, int m, int n, int out[3]);

/* flatmm_signature(spec): manifest-style signature mirroring
 * batched_gemm_signature. Builds the entry array into the caller-provided
 * `arena` (which must outlive the returned array) and sets *out_items /
 * *out_count to the (arena-owned) read-only array:
 *   A,B,C : ptr fp16 ; M,N,K,stride_a,stride_b,stride_c : i32
 *   (+ SortedTokenIds: ptr i32, slot_size: i32 when trait.active_tile_skip).
 * Returns ROCKE_OK, or an error status on a NULL argument / arena OOM. */
rocke_status_t rocke_flatmm_signature(const rocke_flatmm_spec_t* spec,
                                      rocke_arena_t* arena,
                                      const rocke_sig_entry_t** out_items,
                                      size_t* out_count);

/* ----------------------------------------------- tile-level introspection *
 *
 * flatmm_atom_shape(spec) -> (warp_tile_m, warp_tile_n, warp_tile_k). */
void rocke_flatmm_atom_shape(const rocke_flatmm_spec_t* spec, int out_mnk[3]);

/* flatmm_atom(spec) -> the MfmaAtom mfma_atom("f16", m, n, k) resolves to.
 * Returns a pointer into the static catalog (do NOT free), or NULL on a miss
 * (the Python ValueError path) / NULL spec. */
const rocke_mfma_atom_t* rocke_flatmm_atom(const rocke_flatmm_spec_t* spec);

/* flatmm_preshuffle_b_spec(spec) -> PreshuffleBSpec{block_n=tile_n,
 * block_k=tile_k, elem_bytes=2}. Returns ROCKE_ERR_VALUE on a NULL argument. */
rocke_status_t rocke_flatmm_preshuffle_b_spec(const rocke_flatmm_spec_t* spec,
                                              rocke_preshuffleb_spec_t* out);

/* flatmm_preshuffle_b_layout(spec, n=n, k=k): host-side preshuffled-B layout
 * (shape, strides), each a 4-element array (any may be NULL to skip). Wraps
 * rocke_host_preshuffle_layout with the spec-derived PreshuffleBSpec. Returns
 * ROCKE_ERR_VALUE on the divisibility ValueError path (out arrays untouched). */
rocke_status_t rocke_flatmm_preshuffle_b_layout(
    const rocke_flatmm_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FLATMM_H */
