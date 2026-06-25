/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_batched_gemm.h -- C99 port of the batched GEMM kernel instance
 * builder ck_dsl/instances/common/batched_gemm.py (CK Tile
 * ``16_batched_gemm`` parity).
 *
 * batched_gemm.py is a thin wrapper around gemm_universal.py: it defines a
 * BatchedGemmSpec value type, converts it to a UniversalGemmSpec with the
 * batched=True flag set, and delegates the IR build verbatim to
 * build_universal_gemm (the universal kernel body reads block_id_z as the
 * batch index and adds the per-batch stride offsets). All perf knobs
 * (pipelines, epilogues, warp configs, MFMA atoms) are inherited from
 * universal_gemm.
 *
 *   Python (batched_gemm.py)                C99 (this header)
 *   -------------------------------------   ------------------------------------
 *   class BatchedGemmSpec                   ckc_batched_gemm_spec_t
 *     (WarpTileBlockSizeMixin)                (+ _default / _finalize)
 *   ._data_spec()                           ckc_batched_gemm_data_spec()
 *   .to_universal_spec()                    ckc_batched_gemm_to_universal_spec()
 *   .kernel_name()                          ckc_batched_gemm_kernel_name()
 *   is_valid_spec(spec, arch)               ckc_batched_gemm_is_valid_spec()
 *   build_batched_gemm(spec, arch)          ckc_build_batched_gemm()
 *   build_persistent_batched_gemm(...)      ckc_build_persistent_batched_gemm()
 *   batched_gemm_signature(spec)            ckc_batched_gemm_signature()
 *   batched_gemm_grid(batch, m, n, spec)    ckc_batched_gemm_grid()
 *   (+ convenience: build -> lower .ll)     ckc_batched_gemm_lower_to_llvm()
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type
 * with defaults + a __post_init__ that derives block_size. In C the caller
 * fills a ckc_batched_gemm_spec_t. ckc_batched_gemm_spec_default() returns a
 * struct with every field at the Python dataclass default; the caller then
 * overrides name + tile geometry and calls ckc_batched_gemm_spec_finalize()
 * which runs WarpTileBlockSizeMixin._init_block_size() (block_size==0 =>
 * warp_m*warp_n*warp_k*wave_size).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors
 * through the sticky-error IRBuilder (ckc_b_*); the validity gate returns a
 * bool + a reason string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_BATCHED_GEMM_H
#define CKC_INSTANCE_BATCHED_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h"                      /* ckc_arena_t (signature storage) */
#include "ckc/ir.h"                         /* ckc_status_t, ckc_kernel_def_t */
#include "ckc/lower_llvm.h"                 /* ckc_llvm_flavor_t */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t (signature) */
#include "ckc/instance_gemm_universal.h"    /* ckc_gemm_*_spec_t, build */

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------- BatchedGemmSpec *
 *
 * Mirror of the Python frozen dataclass:
 *
 *   @dataclass(frozen=True)
 *   class BatchedGemmSpec(WarpTileBlockSizeMixin):
 *       name: str
 *       tile: TileSpec
 *       trait: TraitSpec = field(default_factory=TraitSpec)
 *       wave_size: int = 64
 *       block_size: int = 0
 *       batch_size: int = 0
 *       dtype: str = "fp16"
 *
 * `batch_size` is informational (the launch grid passes the real batch count);
 * it is used by the dispatcher to skip configs and is NOT baked into the IR.
 */
typedef struct ckc_batched_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;     /* default 64 */
    int block_size;    /* default 0 => derived at finalize() */
    int batch_size;    /* default 0 (informational only)     */
    const char* dtype; /* default "fp16" */
} ckc_batched_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `name` and the required `tile` geometry. The default
 * trait is the gemm_universal TraitSpec() default. */
ckc_batched_gemm_spec_t ckc_batched_gemm_spec_default(void);

/* WarpTileBlockSizeMixin._init_block_size() (run from __post_init__): when
 * block_size==0, derive it as warp_m*warp_n*warp_k*wave_size. Idempotent.
 * Call after filling the spec. */
void ckc_batched_gemm_spec_finalize(ckc_batched_gemm_spec_t* spec);

/* BatchedGemmSpec._data_spec(): build the homogeneous DataSpec the universal
 * conversion uses. The Python canonicalises "f16"/"fp16" -> "fp16" and passes
 * every other dtype through verbatim, applying it to A/B/C; the acc dtype and
 * layout fall back to the DataSpec defaults ("fp32", "RCR"). */
ckc_gemm_data_spec_t ckc_batched_gemm_data_spec(const ckc_batched_gemm_spec_t* spec);

/* BatchedGemmSpec.to_universal_spec(): wrap the spec into a UniversalGemmSpec
 * with batched=True. tile / trait / wave_size / block_size are copied through;
 * data comes from _data_spec(). The returned spec is by value (the Python
 * UniversalGemmSpec is itself a frozen dataclass). */
ckc_gemm_universal_spec_t ckc_batched_gemm_to_universal_spec(const ckc_batched_gemm_spec_t* spec);

/* BatchedGemmSpec.kernel_name() == to_universal_spec().kernel_name(). Writes a
 * NUL-terminated name into out (capacity out_cap). Returns CKC_OK or
 * CKC_ERR_VALUE (buffer too small / NULL args). */
ckc_status_t
ckc_batched_gemm_kernel_name(const ckc_batched_gemm_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason): delegates to
 * is_valid_gemm_spec(spec.to_universal_spec(), arch). `arch` NULL => "gfx950".
 * On a reject, `reason` (if non-NULL, capacity reason_cap) receives the
 * structured message and returns false; on accept returns true. */
bool ckc_batched_gemm_is_valid_spec(const ckc_batched_gemm_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* build_batched_gemm(spec, arch): build the IR for one batched GEMM instance.
 * Validates (raising the Python ValueError path on reject), then delegates to
 * ckc_build_universal_gemm(spec.to_universal_spec(), arch). `b` must be an
 * initialised IRBuilder (created with spec.kernel_name()); use
 * ckc_build_batched_gemm_new() for the init-from-spec convenience. `arch` NULL
 * => "gfx950". Returns the kernel (b->kernel) on success or NULL with b's
 * sticky error set.
 *
 * Kernel signature (generated by the universal body):
 *   (A: ptr, B: ptr, C: ptr, M: i32, N: i32, K: i32,
 *    stride_a: i32, stride_b: i32, stride_c: i32)
 * Grid: (ceil_div(N, tile_n), ceil_div(M, tile_m), batch).
 * Block: (block_size, 1, 1). */
ckc_kernel_def_t*
ckc_build_batched_gemm(ckc_ir_builder_t* b, const ckc_batched_gemm_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_batched_gemm_new(ckc_ir_builder_t* b,
                                             const ckc_batched_gemm_spec_t* spec,
                                             const char* arch);

/* build_persistent_batched_gemm(spec, arch): persistent grouped-GEMM dispatch
 * (P64). Builds the persistent universal-GEMM kernel by copying the converted
 * universal spec, forcing trait.persistent=True (and renaming the kernel
 * "<name>_persistent"), validating, and delegating to
 * ckc_build_universal_gemm. `b` must be an initialised IRBuilder created with
 * the persistent kernel name. `arch` NULL => "gfx950". Returns the kernel or
 * NULL with b's sticky error set. */
ckc_kernel_def_t* ckc_build_persistent_batched_gemm(ckc_ir_builder_t* b,
                                                    const ckc_batched_gemm_spec_t* spec,
                                                    const char* arch);

/* batched_gemm_signature(spec): manifest-style kernarg signature in the exact
 * order the AMDGPU kernarg ABI expects:
 *   A, B, C, M, N, K, stride_a, stride_b, stride_c,
 *   [SortedTokenIds (i32 ptr), slot_size (i32)]   if trait.active_tile_skip
 * The A/B/C pointer dtype is spec.dtype when it is one of
 * "f16"/"fp16"/"bf16", else "f16". Entries (name + type strings) are
 * arena-owned. On CKC_OK *out_items / *out_count hold the array; on failure
 * they are untouched and the status is returned. */
ckc_status_t ckc_batched_gemm_signature(ckc_arena_t* arena,
                                        const ckc_batched_gemm_spec_t* spec,
                                        const ckc_sig_entry_t** out_items,
                                        size_t* out_count);

/* batched_gemm_grid(batch, m, n, spec): the 3D launch grid
 *   ceil_div_grid((n, tile_n), (m, tile_m), (batch, 1))
 * On success out[0..2] hold (x, y, z) = (N_tiles, M_tiles, batch); returns
 * CKC_ERR_VALUE on a non-positive tile (the Python ValueError) or NULL args. */
ckc_status_t
ckc_batched_gemm_grid(int batch, int m, int n, const ckc_batched_gemm_spec_t* spec, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_batched_gemm_lower_to_llvm(const ckc_batched_gemm_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_BATCHED_GEMM_H */
