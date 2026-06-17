/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/ir_import.h -- import a portable CK-DSL IR artifact (schema
 * "ck.dsl.ir/v1", produced by ck_dsl.core.ir_export) into a live
 * ckc_kernel_def_t, so the pure-C backend can lower a Python-authored kernel
 * without embedding CPython.
 *
 * This is the online half of the "shape-polymorphic portable IR" path
 * (dsl_docs/architecture/portable_ir_schema.md): the offline Python exporter
 * serializes a built KernelDef graph to JSON; this importer re-drives it
 * through the C IRBuilder (ckc_b_*), reconstructing SSA values by id, region
 * bodies (scf.for / scf.if) via the real control-flow builders, and typed
 * attributes. The resulting kernel lowers byte-identically to the same kernel
 * built natively in C (verified by the portable_ir parity harness).
 *
 * Error model: returns a ckc_status_t; on failure *out_kernel is NULL and (if
 * err != NULL, capacity err_cap) a human-readable diagnostic is written. On
 * success the caller owns out_builder and frees it with ckc_ir_builder_free();
 * *out_kernel is arena-owned by out_builder (do not free separately).
 */
#ifndef CKC_IR_IMPORT_H
#define CKC_IR_IMPORT_H

#include <stddef.h>

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ckc_import_options
{
    const char* expected_kernel_name; /* optional: reject on mismatch (NULL=skip) */
    bool strict;                      /* reserved; importer is always strict today */
} ckc_import_options_t;

/* Parse `text` (NUL-terminated portable-IR JSON) and build the kernel into
 * `out_builder` (initialized by this call). `opts` may be NULL. */
ckc_status_t ckc_import_kernel_from_json(const char* text,
                                         const ckc_import_options_t* opts,
                                         ckc_ir_builder_t* out_builder,
                                         ckc_kernel_def_t** out_kernel,
                                         char* err,
                                         size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_IR_IMPORT_H */
