// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* The C++ engine's read side of the intrinsic arch-domain table. Mirrors
 * rocke.core.arch.domain.
 *
 * tools/gen_arch_domain.py measures, per (declaration key, gfx target), whether a
 * declaration links, and commits one JSON column per LLVM flavor. The Python
 * engine reads those columns when it lowers. This engine reads no files, so the
 * build generates its copy from the same columns through the same Python reader
 * (tools/gen_arch_domain_cpp.py); there is no hand-maintained mirror to drift.
 *
 * Only the cells the Python policy calls negative (domain.is_negative) are
 * compiled in. A lookup that finds nothing therefore means "no measured
 * negative" -- the silence the Python lane keeps for an `ok` cell, every no-data
 * status, an unswept target, an unknown key and a flavor with no column.
 */
#ifndef ROCKE_ARCH_DOMAIN_H
#define ROCKE_ARCH_DOMAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One measured negative answer. Every field points at static storage. */
typedef struct rocke_arch_domain_cell
{
    const char* key; /* declaration-table key, e.g. "ds.read.tr16.b64" */
    const char* arch; /* gfx target, e.g. "gfx942" */
    const char* status; /* the measured status, e.g. "arch_absent" */
    const char* evidence; /* first line of the probe's toolchain diagnostic */
} rocke_arch_domain_cell_t;

/* The negative cell for (flavor, key, arch), or NULL when the committed table
 * has no negative answer for it. Any NULL argument returns NULL. */
const rocke_arch_domain_cell_t*
    rocke_arch_domain_negative(const char* flavor, const char* key, const char* arch);

/* Every negative cell of `flavor`'s column, sorted by (key, arch), with the count
 * in *count. A flavor with no committed column, or one with no negative cell,
 * returns NULL and sets *count to 0. Exists so a test can hold this table to the
 * Python reader cell for cell. */
const rocke_arch_domain_cell_t* rocke_arch_domain_column(const char* flavor, size_t* count);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_ARCH_DOMAIN_H */
