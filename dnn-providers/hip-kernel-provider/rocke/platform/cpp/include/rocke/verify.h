/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/verify.h -- IR verifier (LLVM-verify-style well-formedness pass).
 *
 * Faithful C99 port of rocke/core/verify.py. rocke_verify walks
 * a rocke_kernel_def_t and produces a list of rocke_diag_t. An empty list (n==0)
 * means well-formed. The checks mirror the Python verifier exactly:
 *
 *   - SSA dominance / scoping (operands defined + visible before use; no
 *     redefinition; no dangling refs).
 *   - Type consistency (binary/unary/cmp/select arith, vector.extract typing,
 *     tile.mma arity, scf.for iter-arg/result/yield typing).
 *   - Arity / result counts per opcode.
 *   - Region well-formedness (scf.for/scf.if required regions; empty body).
 *   - Required attr keys per opcode (arith.constant, *.cmp, tile.mma, scf.yield,
 *     tile.inline_asm).
 *   - Vector width / smem shape / pointer address-space sanity.
 *
 * Diagnostics are structurally comparable to the Python Diagnostic
 * (severity + message + optional op name + optional loc): the `message` text is
 * built to match the Python f-strings so a parity harness can diff them.
 */
#ifndef ROCKE_VERIFY_H
#define ROCKE_VERIFY_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rocke_diag_severity
{
    ROCKE_DIAG_ERROR = 0,
    ROCKE_DIAG_WARNING
} rocke_diag_severity_t;

/* One diagnostic. `message` / `op` / `loc` are malloc'd (freed by
 * rocke_diags_free). `op` and `loc` may be NULL (no associated op / no loc). */
typedef struct rocke_diag
{
    rocke_diag_severity_t severity;
    char* message; /* malloc'd                                   */
    char* op; /* op name (ref) or NULL                      */
    char* loc; /* op.loc if present, else NULL                */
} rocke_diag_t;

/* Verify `k`. On return *out points to a malloc'd array of `*n` diagnostics
 * (NULL / 0 when well-formed). The caller frees with rocke_diags_free. Returns
 * ROCKE_OK unless an allocation failed (then ROCKE_ERR_OOM and *out/*n are 0). */
rocke_status_t rocke_verify(const rocke_kernel_def_t* k, rocke_diag_t** out, size_t* n);

/* Free a diagnostics array returned by rocke_verify. */
void rocke_diags_free(rocke_diag_t* diags, size_t n);

/* Render one diagnostic like Python Diagnostic.__str__:
 *   "<severity>: <message>[ [<op>]][ @<loc>]"
 * into a freshly malloc'd string (caller frees). NULL on OOM. */
char* rocke_diag_to_string(const rocke_diag_t* d);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_VERIFY_H */
