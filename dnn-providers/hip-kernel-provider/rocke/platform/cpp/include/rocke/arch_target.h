/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/arch_target.h -- PUBLIC API for the C99 port of
 * rocke.core.arch.target (the polymorphic-core arch SSOT).
 *
 * This is the single ROCKE-owned description of *what a gfx target supports*:
 * wave size, LDS capacity, the MMA atom catalog, memory capability bits, and
 * resource limits. Hardware facts only -- no pipeline/scheduler vocabulary, no
 * LLVM intrinsic text.
 *
 * Python -> C99 correspondence:
 *
 *   Python (target.py)                C99 (this header)
 *   ------------------------------    --------------------------------------
 *   class LayoutMap (frozen)          rocke_layout_map_t
 *     .fn closure                       rocke_lane_coord_fn (fn ptr + builder)
 *     .coord(builder, lane, slot)       rocke_layout_map_coord()
 *   class MmaOp (frozen)              rocke_mma_op_t
 *     .shape / .a_layout()/...          rocke_mma_op_shape() / *_layout() getters
 *   class MemoryCapabilities          rocke_memory_caps_t
 *   class ResourceLimits              rocke_resource_limits_t
 *   class MmaCatalog                  rocke_mma_catalog_t + rocke_mma_catalog_* ()
 *   class ArchTarget (frozen)         rocke_arch_target_t + rocke_arch_* () getters
 *   ArchTarget.from_gfx(gfx)          rocke_arch_target_from_gfx()
 *   known_arches()                    rocke_known_arches()
 *   arch_from_isa(isa)                rocke_arch_from_isa()
 *   target_id_from_isa(isa)           rocke_target_id_from_isa()
 *   base_arch_from_target_id(id)      rocke_base_arch_from_target_id()
 *   compiler_target_from_target_id(id) rocke_compiler_target_from_target_id()
 *
 * The Python loader reads core/arch/data/arch_specs.json at import time. The C99
 * port embeds that frozen SSOT as static tables (libc-only: no JSON parser), so
 * lookups never touch the filesystem. The catalog/target descriptors and their
 * layout maps are all stored in static read-only tables; an ArchTarget pointer
 * returned here is a borrow of that static storage and is valid for the program
 * lifetime (mirroring the Python @lru_cache singletons). The *only* arena use is
 * by the layout-map coordinate emitters, which create IR Values in the builder
 * passed at call time.
 *
 * Error model: the lookup entry points that can fail (unknown gfx) return NULL
 * and, when a builder is supplied to the layout-coord emitter, set the builder's
 * sticky error -- matching the rest of the C99 port. The data getters never
 * fail.
 */
#ifndef ROCKE_ARCH_TARGET_H
#define ROCKE_ARCH_TARGET_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/dtypes.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================== layout map ============================== */

/* MMA fragment role. Matrix roles mirror LayoutMap.role ("acc"/"a"/"b");
 * scale roles are "a_scale"/"b_scale" and use logical K-group coordinates. */
typedef enum rocke_mma_role
{
    ROCKE_MMA_ROLE_ACC = 0, /* accumulator C/D: coords (row, col)  */
    ROCKE_MMA_ROLE_A, /* A operand:       coords (row, k)    */
    ROCKE_MMA_ROLE_B, /* B operand:       coords (k, col)    */
    ROCKE_MMA_ROLE_A_SCALE, /* A scale: coords (row, K-group) */
    ROCKE_MMA_ROLE_B_SCALE /* B scale: coords (K-group, col) */
} rocke_mma_role_t;

/* The lane/slot -> tile-coordinate emitter. Given the builder, a runtime i32
 * lane Value and a compile-time slot index, it emits the index arithmetic for
 * the two coordinates and writes them to *out0/*out1. This is the C analog of
 * the Python `_LaneCoordFn` closure (`(builder, lane, slot) -> (c0, c1)`); the
 * coordinate meaning depends on the role (see rocke_mma_role_t). On a failed
 * builder it is a no-op leaving *out0/*out1 = NULL. */
typedef void (*rocke_lane_coord_fn)(rocke_ir_builder_t* b,
                                    rocke_value_t* lane,
                                    int slot,
                                    rocke_value_t** out0,
                                    rocke_value_t** out1);

/* Lane/slot -> tile-coordinate map for one MMA fragment role. Frozen / value
 * type; instances live in static storage. Mirrors LayoutMap. */
typedef struct rocke_layout_map
{
    rocke_mma_role_t role;
    int frag_len; /* fragment slots per lane for this role  */
    int wave_size; /* 64 (MFMA) / 32 (WMMA)                  */
    rocke_lane_coord_fn fn; /* NULL => "no verified map" (see below)  */
} rocke_layout_map_t;

/* Emit the index math for (lane, slot) and write the coordinate pair to
 * *out0/*out1. Validates slot in [0, frag_len) (Python LayoutMap.coord's
 * ValueError); on out-of-range it sets the builder error (ROCKE_ERR_VALUE) and
 * writes NULLs. Returns true on success. Mirrors LayoutMap.coord. */
bool rocke_layout_map_coord(const rocke_layout_map_t* m,
                            rocke_ir_builder_t* b,
                            rocke_value_t* lane,
                            int slot,
                            rocke_value_t** out0,
                            rocke_value_t** out1);

/* ============================== MMA atom =============================== */

/* Shared K-group size. NONE is only the unscaled sentinel. */
typedef enum rocke_mma_scale_block_k
{
    ROCKE_MMA_SCALE_NONE = 0,
    ROCKE_MMA_SCALE_K16 = 16,
    ROCKE_MMA_SCALE_K32 = 32
} rocke_mma_scale_block_k_t;

/* Complete scale query: independent A/B types, shared K-group size.
 * {NULL, NULL, 0} selects unscaled atoms. */
typedef struct rocke_mma_scale_filter
{
    const char* a_scale_dtype;
    const char* b_scale_dtype;
    rocke_mma_scale_block_k_t scale_block_k;
} rocke_mma_scale_filter_t;

/* A supported matrix-multiply-accumulate atom. Frozen, in static storage.
 * Mirrors MmaOp: family is "mma", "wmma", or "wmma_scaled"; matrix dtypes
 * are canonical catalog keys. C describes both accumulator input and result.
 * op_id is an opaque backend handle; lowering reads metadata, not its spelling.
 * Layout pointers are NULL when no verified lane map is registered. */
typedef struct rocke_mma_op
{
    const char* family;
    const char* a_dtype;
    const char* b_dtype;
    const char* c_dtype;
    int m;
    int n;
    int k;
    const char* op_id;
    int a_frag_len;
    int b_frag_len;
    int c_frag_len;
    int wave_size;
    const rocke_layout_map_t* a_layout; /* may be NULL */
    const rocke_layout_map_t* b_layout; /* may be NULL */
    const rocke_layout_map_t* c_layout; /* may be NULL */
    const char* a_scale_dtype; /* NULL when unscaled */
    const char* b_scale_dtype; /* NULL when unscaled */
    rocke_mma_scale_block_k_t scale_block_k; /* shared by A/B; NONE when unscaled */
    /* Logical scale elements per lane, independent of the packed carrier.
     * Appended ABI fields: native consumers must rebuild. */
    int a_scale_frag_len;
    int b_scale_frag_len;
    const rocke_layout_map_t* a_scale_layout; /* may be NULL */
    const rocke_layout_map_t* b_scale_layout; /* may be NULL */
} rocke_mma_op_t;

/* MmaOp.shape -> (m, n, k) via out params. */
void rocke_mma_op_shape(const rocke_mma_op_t* op, int* m, int* n, int* k);

/* Physical-layout accessors. Missing maps raise ckc::Error with
 * ROCKE_ERR_NOTIMPL, like Python's NotImplementedError. The enclosing public
 * builder/lowering boundary translates that exception into status + message;
 * direct C++ callers must catch it, including when b is NULL. A NULL op returns
 * NULL. Mirrors MmaOp's matrix and scale layout accessors. */
const rocke_layout_map_t* rocke_mma_op_a_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b);
const rocke_layout_map_t* rocke_mma_op_b_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b);
const rocke_layout_map_t* rocke_mma_op_c_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b);
const rocke_layout_map_t* rocke_mma_op_acc_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b);
const rocke_layout_map_t* rocke_mma_op_a_scale_layout(const rocke_mma_op_t* op,
                                                      rocke_ir_builder_t* b);
const rocke_layout_map_t* rocke_mma_op_b_scale_layout(const rocke_mma_op_t* op,
                                                      rocke_ir_builder_t* b);

/* ====================== memory caps / resource limits ================== */

typedef struct rocke_memory_caps
{
    bool has_async_lds;
    bool has_ds_read_tr;
    int buffer_load_max_dwords;
} rocke_memory_caps_t;

typedef struct rocke_resource_limits
{
    int max_threads_per_block;
    int vgprs;
    int agprs;
    int sgprs;
} rocke_resource_limits_t;

/* ============================== MMA catalog ============================ */

/* The arch-selected set of MMA atoms. Mirrors MmaCatalog. The ops array is a
 * borrow of static storage owned by the ArchTarget; do not free. */
typedef struct rocke_mma_catalog
{
    const rocke_mma_op_t* ops; /* static array */
    int num_ops;
} rocke_mma_catalog_t;

/* MmaCatalog.ops accessor (count + pointer). */
const rocke_mma_op_t* rocke_mma_catalog_ops(const rocke_mma_catalog_t* cat, int* num_out);

/* Optional scale filter: NULL leaves scales unconstrained; {NULL, NULL, 0}
 * selects unscaled atoms. C callers supply the filter pointer explicitly.
 * Exact selection and largest-K ties raise a query error on ambiguity.
 * As with layout queries, C++ callers catch ckc::Error at their entry boundary. */
#ifdef __cplusplus
#define ROCKE_MMA_DEFAULT_SCALES = nullptr
#else
#define ROCKE_MMA_DEFAULT_SCALES
#endif

/* MmaCatalog.enumerate: filter atoms by family + (normalised) dtype combo and
 * optional m/n. Writes up to `cap` matching op pointers into `out` and returns
 * the total number of matches (which may exceed `cap` -- the caller can size a
 * buffer with cap=0 to count first). Pass m<0 / n<0 to mean "any" (Python None).
 * `family` may be NULL => "mma". */
int rocke_mma_catalog_enumerate(const rocke_mma_catalog_t* cat,
                                const char* family,
                                const char* a_dtype,
                                const char* b_dtype,
                                const char* c_dtype,
                                int m,
                                int n,
                                const rocke_mma_op_t** out,
                                int cap,
                                const rocke_mma_scale_filter_t* scales ROCKE_MMA_DEFAULT_SCALES);

/* MmaCatalog.has_shape. family NULL => "mma". */
bool rocke_mma_catalog_has_shape(const rocke_mma_catalog_t* cat,
                                 const char* family,
                                 const char* a_dtype,
                                 const char* b_dtype,
                                 const char* c_dtype,
                                 int m,
                                 int n,
                                 int k,
                                 const rocke_mma_scale_filter_t* scales ROCKE_MMA_DEFAULT_SCALES);

/* MmaCatalog.select_largest_k: the matching atom with the largest k (k <= k_max
 * if k_max >= 0; pass k_max<0 for Python None). Returns NULL if none match. */
const rocke_mma_op_t* rocke_mma_catalog_select_largest_k(const rocke_mma_catalog_t* cat,
                                                         const char* family,
                                                         const char* a_dtype,
                                                         const char* b_dtype,
                                                         const char* c_dtype,
                                                         int m,
                                                         int n,
                                                         int k_max,
                                                         const rocke_mma_scale_filter_t* scales
                                                             ROCKE_MMA_DEFAULT_SCALES);

/* MmaCatalog.by_op_id. Returns NULL if absent. */
const rocke_mma_op_t* rocke_mma_catalog_by_op_id(const rocke_mma_catalog_t* cat, const char* op_id);

/* MmaCatalog.op_for_shape: the atom with exactly (m, n, k). NULL if absent.
 * family NULL => "mma". */
const rocke_mma_op_t*
    rocke_mma_catalog_op_for_shape(const rocke_mma_catalog_t* cat,
                                   const char* family,
                                   const char* a_dtype,
                                   const char* b_dtype,
                                   const char* c_dtype,
                                   int m,
                                   int n,
                                   int k,
                                   const rocke_mma_scale_filter_t* scales ROCKE_MMA_DEFAULT_SCALES);

/* ===================== bare-op_id SSOT lookups ======================== */
/* These resolve a bare op_id string (no MmaOp / ArchTarget in hand) against the
 * arch SSOT, so the ir bucket can size a tile.mma result without duplicating
 * any atom metadata. They mirror the target.py module-level helpers that
 * rocke.core.ir.IRBuilder.mma consults. */

#undef ROCKE_MMA_DEFAULT_SCALES

/* Accumulator fragment length for op_id -- the c_frag_len projection of
 * target.py::_MMA_FRAGMENT_INFO (rocke_arch_mma_c_frag_len(op_id) ==
 * target._frag_info(op_id).c_frag_len). Returns 0 for an unknown atom (the
 * zero-length _FragInfo fallback). */
int rocke_arch_mma_c_frag_len(const char* op_id);

/* Normalised accumulator dtype for op_id, aggregated across every arch's
 * catalog (mirrors target._op_id_c_dtype()[op_id]). The dtype is invariant
 * across the arches that list an op_id, so the first catalog hit wins. Returns
 * NULL for an op_id absent from every catalog. The returned string is an
 * interned canonical key (e.g. ROCKE_DTYPE_I32 / "fp32"); do not free. */
const char* rocke_arch_mma_op_id_c_dtype(const char* op_id);

/* Target-independent family projection. Unknown/NULL IDs return NULL;
 * conflicting families for one ID raise ROCKE_ERR_VALUE. */
const char* rocke_arch_mma_op_id_family(const char* op_id);

/* ============================== arch target =========================== */

/* Hardware-facts surface for one gfx target. Frozen; cheap to pass around.
 * Returned pointers are borrows of static storage (program-lifetime). Mirrors
 * ArchTarget. */
typedef struct rocke_arch_target
{
    const char* gfx;
    const char* family;
    const char* target_family;
    int wave_size;
    int lds_capacity_bytes;
    int vmcnt_bits;
    rocke_mma_catalog_t mma;
    rocke_memory_caps_t memory;
    rocke_resource_limits_t limits;
} rocke_arch_target_t;

/* ArchTarget.from_gfx: the canonical singleton descriptor for `gfx`. Returns
 * NULL for an unknown target (Python raises KeyError); the failure is pure data
 * lookup so no builder is involved. Mirrors ArchTarget.from_gfx / _build_target.
 */
const rocke_arch_target_t* rocke_arch_target_from_gfx(const char* gfx);

/* ArchTarget.isa_triple -> "amdgcn-amd-amdhsa--<gfx>". Writes into `out`
 * (>= out_cap bytes) and returns `out` (or NULL if out too small / NULL). */
const char* rocke_arch_isa_triple(const rocke_arch_target_t* t, char* out, size_t out_cap);

/* ArchTarget.fits_lds. */
bool rocke_arch_fits_lds(const rocke_arch_target_t* t, long bytes_in_use);

/* ArchTarget.supports_dtype_combo. family NULL => "mma". */
bool rocke_arch_supports_dtype_combo(
    const rocke_arch_target_t* t, const char* a, const char* b, const char* c, const char* family);

/* ArchTarget.max_vector_load_dwords (gated by the buffer-load path today, so the
 * dtype argument is accepted for parity but ignored). */
int rocke_arch_max_vector_load_dwords(const rocke_arch_target_t* t, const char* dtype);

/* ArchTarget.max_threads_per_block property. */
int rocke_arch_max_threads_per_block(const rocke_arch_target_t* t);

/* ============================== module fns ============================ */

/* known_arches(): a sorted, NULL-terminated array of gfx tokens. Static
 * storage; do not free. If `count` is non-NULL it receives the element count
 * (excluding the terminating NULL). */
const char* const* rocke_known_arches(int* count);

/* Target identity helpers mirror rocke.core.arch.target. They do not validate
 * compiler support or query a GPU. Inputs and output buffers must not overlap.
 * Each writes at most out_cap - 1 bytes plus a NUL and returns out; a small
 * buffer receives a truncated result (the existing arch_from_isa convention).
 * NULL input/output or zero capacity returns NULL without writing. */

/* Extract from the last "gfx", preserving profile and feature suffixes.
 * If there is no "gfx", copy the input unchanged. */
const char* rocke_target_id_from_isa(const char* isa, char* out, size_t out_cap);

/* Remove features and profiles for catalog lookup, preserving catalog names
 * such as gfx11-generic. gfx1250-strict and gfx942:xnack- become gfx1250/gfx942. */
const char* rocke_base_arch_from_target_id(const char* target_id, char* out, size_t out_cap);

/* Remove profiles but preserve compiler features: gfx1250-strict:xnack-
 * becomes gfx1250:xnack-. */
const char* rocke_compiler_target_from_target_id(const char* target_id, char* out, size_t out_cap);

/* Extract the target ID from an ISA name, then derive its base architecture. */
const char* rocke_arch_from_isa(const char* isa, char* out, size_t out_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ARCH_TARGET_H */
