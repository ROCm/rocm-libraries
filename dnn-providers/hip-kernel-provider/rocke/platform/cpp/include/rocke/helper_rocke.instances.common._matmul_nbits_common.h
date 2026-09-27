/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._matmul_nbits_common.h -- C99 port of the
 * shared surface of the MatMulNBits gfx1151 instance family
 * (rocke/instances/common/_matmul_nbits_common.py).
 *
 *   Python (_matmul_nbits_common.py)        C99 (this header)
 *   --------------------------------------  -------------------------------------
 *   MatMulNBitsFamily (Literal)             rocke_matmul_nbits_family_t (string set)
 *   FAMILIES                                rocke_matmul_nbits_families() / *_COUNT
 *   SUPPORTED_ARCHES                        rocke_matmul_nbits_arch_supported()
 *   V1_ARCH                                 ROCKE_MATMUL_NBITS_V1_ARCH
 *   V1_GROUP_SIZE                           ROCKE_MATMUL_NBITS_V1_GROUP_SIZE
 *   _scale_wire_dtype(scale_dtype)          rocke_matmul_nbits_scale_wire_dtype()
 *   class MatMulNBitsSpec                   rocke_matmul_nbits_spec_t
 *     .__post_init__ / _init_block_size      rocke_matmul_nbits_spec_finalize()
 *     .kernel_name()                         rocke_matmul_nbits_kernel_name()
 *   validate_common_spec(spec, arch)        rocke_matmul_nbits_validate_common_spec()
 *   matmul_nbits_signature(spec)            rocke_matmul_nbits_signature()
 *   matmul_nbits_grid(M, spec)              rocke_matmul_nbits_grid()
 *   matmul_nbits_outer_tiles(seq_len, spec) rocke_matmul_nbits_outer_tiles()
 *
 * SCOPE. This is the Milestone-1 surface only: spec, validator, signature,
 * grid, and the 64-row outer-loop driver. The host-side numpy packer /
 * reference (pack_i4_weights_for_matmul_nbits / dequant_i4_weights /
 * matmul_nbits_reference) is a host-test path that depends on numpy and is
 * deliberately NOT ported here.
 *
 * NONE of these symbols call the IR builder (rocke_b_*): they are pure value /
 * string producers. They reuse the canonical sibling helpers
 * (helper_rocke.helpers.spec.h for derive_block_size / kernel_name_join, and
 * helper_rocke.core.arch.h for ArchTarget) so the produced block_size, kernel
 * name, validity verdict, signature, and grid are byte-faithful to Python.
 *
 * Error model mirrors the rest of the C port: an out-param + rocke_status_t (or a
 * bool + reason buffer for the validity gate) stands in for the Python
 * `raise ValueError` / `(ok, reason)` tuple.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_COMMON_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (signature storage)                  */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t, builder    */
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_tile_spec_t (TileSpec) */
#include "rocke/ir.h" /* rocke_status_t                 */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * Module constants
 * ------------------------------------------------------------------ */

/* V1_ARCH = "gfx1151". */
#define ROCKE_MATMUL_NBITS_V1_ARCH "gfx1151"

/* V1_GROUP_SIZE = 32. */
#define ROCKE_MATMUL_NBITS_V1_GROUP_SIZE 32

/* FAMILIES = ("large_n", "skinny_n", "decode_gemv"). */
#define ROCKE_MATMUL_NBITS_FAMILIES_COUNT 3

/* SUPPORTED_ARCHES = frozenset({"gfx1151", "gfx1201"}). */
#define ROCKE_MATMUL_NBITS_SUPPORTED_ARCHES_COUNT 2

/* MatMulNBitsFamily is a Python Literal; in C it is just one of the FAMILIES
 * strings, compared by strcmp. This typedef documents that intent. */
typedef const char* rocke_matmul_nbits_family_t;

/* FAMILIES: ordered, NULL-terminated static array ("large_n", "skinny_n",
 * "decode_gemv"). The terminating NULL is NOT counted by
 * ROCKE_MATMUL_NBITS_FAMILIES_COUNT. */
const char* const* rocke_matmul_nbits_families(void);

/* True iff `family` is one of FAMILIES (`family` NULL => false). */
bool rocke_matmul_nbits_family_known(const char* family);

/* True iff `arch` is in SUPPORTED_ARCHES (`arch` NULL => false). */
bool rocke_matmul_nbits_arch_supported(const char* arch);

/* ------------------------------------------------------------------ *
 * _scale_wire_dtype
 * ------------------------------------------------------------------ *
 *
 * Python:
 *   if scale_dtype in ("fp16", "f16"): return "f16"
 *   if scale_dtype in ("fp32", "f32"): return "f32"
 *   raise ValueError(...)
 *
 * Returns a borrowed static string ("f16" / "f32") on success and sets *out to
 * it; returns ROCKE_ERR_VALUE on an unsupported dtype and leaves *out untouched.
 * (`out` may be NULL to use the function as a pure validity probe.) */
rocke_status_t rocke_matmul_nbits_scale_wire_dtype(const char* scale_dtype, const char** out);

/* ------------------------------------------------------------------ *
 * MatMulNBitsSpec
 * ------------------------------------------------------------------ *
 *
 * Mirror of the frozen dataclass. `tile` reuses the GEMM-family TileSpec
 * (rocke_gemm_tile_spec_t). String fields hold borrowed pointers compared by
 * strcmp, matching the Python str fields. block_size defaults to 0 and is
 * derived by rocke_matmul_nbits_spec_finalize() (the __post_init__ /
 * _init_block_size step). */
typedef struct rocke_matmul_nbits_spec
{
    const char* name;
    int N;
    int K;
    rocke_gemm_tile_spec_t tile;
    int group_size; /* default V1_GROUP_SIZE (32)        */
    int seq_len_tile; /* default 64                        */
    int wave_size; /* default 32                        */
    int block_size; /* default 0 => derived at finalize  */
    const char* scale_dtype; /* default "fp16"                    */
    bool zero_points; /* default false                     */
    const char* packing; /* default "row_k_contiguous"        */
    const char* family; /* default "large_n"                 */
    bool optimized; /* default false                     */
} rocke_matmul_nbits_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `name`, N, K, and the required `tile` geometry. */
rocke_matmul_nbits_spec_t rocke_matmul_nbits_spec_default(void);

/* __post_init__ / WarpTileBlockSizeMixin._init_block_size(): when block_size==0,
 * derive it as warp_m*warp_n*warp_k*wave_size. Idempotent. Call after filling
 * the spec. */
void rocke_matmul_nbits_spec_finalize(rocke_matmul_nbits_spec_t* spec);

/* MatMulNBitsSpec.kernel_name() -> NUL-terminated into `out` (capacity out_cap).
 * Reproduces the kernel_name_join part/flag order exactly:
 *   name, family, "f16", "N{N}K{K}", "g{group_size}",
 *   "t{m}x{n}x{k}", "w{m}x{n}x{k}", "wt{m}x{n}x{k}", "s{scale_wire}",
 *   flags={"zp": zero_points}
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE on a bad scale_dtype or too-small buffer. */
rocke_status_t rocke_matmul_nbits_kernel_name(const rocke_matmul_nbits_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* ------------------------------------------------------------------ *
 * validate_common_spec
 * ------------------------------------------------------------------ *
 *
 * Family-agnostic validity gate. Returns true (and writes "ok" to `reason` when
 * non-NULL) on accept, or false with the structured Python reason string on
 * reject. `arch` NULL => V1_ARCH ("gfx1151"). `reason`/`reason_cap` may be NULL/0
 * to skip the message. */
bool rocke_matmul_nbits_validate_common_spec(const rocke_matmul_nbits_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);

/* ------------------------------------------------------------------ *
 * matmul_nbits_signature
 * ------------------------------------------------------------------ *
 *
 * Manifest signature, reproducing the Python builder chain exactly:
 *
 *   SignatureBuilder()
 *       .ptr("A", "f16")
 *       .ptr("B", "i8")          # two signed int4 packed per byte
 *       .ptr("Scales", _scale_wire_dtype(spec.scale_dtype))
 *       .ptr("C", "f16")
 *       .scalar("M", "i32")
 *       .build()
 *
 * The entries and their {name, type} strings are allocated in `arena` via the
 * canonical rocke_signature_builder_* port (so the produced array is byte-faithful
 * to Python's List[dict]). On ROCKE_OK *out_items / *out_count expose the
 * arena-owned array (read-only for the arena's lifetime); on a bad scale_dtype
 * the underlying _scale_wire_dtype ValueError surfaces as ROCKE_ERR_VALUE. */
rocke_status_t rocke_matmul_nbits_signature(const rocke_matmul_nbits_spec_t* spec,
                                            rocke_arena_t* arena,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);

/* ------------------------------------------------------------------ *
 * matmul_nbits_grid
 * ------------------------------------------------------------------ *
 *
 * Launch grid (ceil(N / tile_n), ceil(M / tile_m), 1) for `M` rows. Returns
 * ROCKE_OK and writes gx/gy/gz, or ROCKE_ERR_VALUE if a tile dim is non-positive
 * (matching ceil_div_grid's ValueError). gx/gy/gz may individually be NULL. */
rocke_status_t rocke_matmul_nbits_grid(
    int M, const rocke_matmul_nbits_spec_t* spec, int* gx, int* gy, int* gz);

/* ------------------------------------------------------------------ *
 * matmul_nbits_outer_tiles
 * ------------------------------------------------------------------ *
 *
 * Split a dynamic `seq_len` into seq_len_tile-row outer tiles. Writes up to
 * `out_cap` (m_outer, m_tile) pairs into out_m_outer[] / out_m_tile[] and sets
 * *out_count to the number produced (ceil(seq_len / seq_len_tile)). The arrays
 * may be NULL to query *out_count only (pass out_cap=0). Returns ROCKE_OK, or
 * ROCKE_ERR_VALUE on seq_len < 0 (matching the Python ValueError) or when the
 * (non-NULL) arrays are too small for the produced count. */
rocke_status_t rocke_matmul_nbits_outer_tiles(int seq_len,
                                              const rocke_matmul_nbits_spec_t* spec,
                                              int* out_m_outer,
                                              int* out_m_tile,
                                              size_t out_cap,
                                              size_t* out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON_MATMUL_NBITS_COMMON_H */
