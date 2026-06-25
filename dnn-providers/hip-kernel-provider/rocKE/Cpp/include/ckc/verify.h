/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/verify.h -- IR verifier (LLVM-verify-style well-formedness pass).
 *
 * Faithful C99 port of ck_dsl/core/verify.py. ckc_verify walks
 * a ckc_kernel_def_t and produces a list of ckc_diag_t. An empty list (n==0)
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
#ifndef CKC_VERIFY_H
#define CKC_VERIFY_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ckc_diag_severity
{
    CKC_DIAG_ERROR = 0,
    CKC_DIAG_WARNING
} ckc_diag_severity_t;

/* One diagnostic. `message` / `op` / `loc` are malloc'd (freed by
 * ckc_diags_free). `op` and `loc` may be NULL (no associated op / no loc). */
typedef struct ckc_diag
{
    ckc_diag_severity_t severity;
    char* message; /* malloc'd                                   */
    char* op; /* op name (ref) or NULL                      */
    char* loc; /* op.loc if present, else NULL                */
} ckc_diag_t;

/* Verify `k`. On return *out points to a malloc'd array of `*n` diagnostics
 * (NULL / 0 when well-formed). The caller frees with ckc_diags_free. Returns
 * CKC_OK unless an allocation failed (then CKC_ERR_OOM and *out/*n are 0). */
ckc_status_t ckc_verify(const ckc_kernel_def_t* k, ckc_diag_t** out, size_t* n);

/* Free a diagnostics array returned by ckc_verify. */
void ckc_diags_free(ckc_diag_t* diags, size_t n);

/* Render one diagnostic like Python Diagnostic.__str__:
 *   "<severity>: <message>[ [<op>]][ @<loc>]"
 * into a freshly malloc'd string (caller frees). NULL on OOM. */
char* ckc_diag_to_string(const ckc_diag_t* d);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_VERIFY_H */
