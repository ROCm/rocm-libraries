/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/arch_target_internal.h -- PRIVATE shared declarations for the C99 port of
 * ck_dsl.core.arch.target. NOT a public API: only the arch_target_*.c
 * translation units include this. The public contract is ckc/arch_target.h.
 *
 * Everything here is a cross-bucket helper shared by the two parallel body
 * files. The DEFINITIONS of all functions/tables declared here live in bucket 0
 * (arch_target_data.c); bucket 1 (arch_target_query.c) only references them.
 *
 * Naming: internal helpers/tables are prefixed ckc_ati_ (ati = arch-target
 * internal) to keep them out of the public ckc_ / ckc_arch_ namespace.
 *
 * What lives where:
 *   bucket 0 (arch_target_data.c)  -- the frozen SSOT: every lane-coord emitter
 *     fn, the op_id -> _FragInfo table, the per-arch static descriptor tables
 *     (the embedded arch_specs.json), and the shared lookup helpers below
 *     (DEFINITIONS).
 *   bucket 1 (arch_target_query.c) -- MmaCatalog query methods, ArchTarget
 *     predicates/getters, and the from_gfx / known_arches / arch_from_isa
 *     module fns. It calls the bucket-0 helpers/tables via this header.
 */
#ifndef CKC_ARCH_TARGET_INTERNAL_H
#define CKC_ARCH_TARGET_INTERNAL_H

#include "ckc/arch_target.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------- registered arch table row */

/* One row of the embedded arch SSOT (the C analog of one arches[gfx] entry in
 * arch_specs.json). All fields point at static storage. The `mma` array is the
 * already-built catalog for that arch (op rows enriched with frag lengths and
 * layout-map pointers at table-build time, mirroring _build_mma_op). */
typedef struct ckc_ati_arch_row
{
    const char* gfx;
    const ckc_arch_target_t* target; /* fully-built singleton descriptor */
} ckc_ati_arch_row_t;

/* The registry of all known arches, sorted by gfx token (so ckc_known_arches
 * can return it directly). Terminated by a row whose .gfx == NULL. Defined in
 * bucket 0. Bucket 1 walks it for from_gfx / known_arches. */
extern const ckc_ati_arch_row_t ckc_ati_arch_registry[];

/* Count of real rows in ckc_ati_arch_registry (excluding the NULL terminator).
 * Defined in bucket 0. */
extern const int ckc_ati_arch_registry_len;

/* Sorted NULL-terminated array of just the gfx tokens (the cached return value
 * of ckc_known_arches). Defined in bucket 0 alongside the registry so both stay
 * in sync. */
extern const char* const ckc_ati_known_arches[];

/* ------------------------------------------------------- dtype normalisation */

/* The canonical-key core of ckc_normalize_dtype, shared so both buckets resolve
 * spellings identically. `lowered` is a caller buffer that receives the
 * strip+lower of `name` (used as the pass-through result for unknown spellings).
 * Returns a static interned canonical string for a known alias, else `lowered`.
 * Defined in bucket 0. */
const char* ckc_ati_normalize_dtype(const char* name, char* lowered, size_t cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_ARCH_TARGET_INTERNAL_H */
