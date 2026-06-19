/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/ir_serialize.h -- `ck.dsl.ir/v1` round-trippable IR serialization.
 *
 * Faithful C99 port of ck_dsl/core/ir_serialize.py. This is the MACHINE
 * interchange format specified in
 * dsl_docs/architecture/ir_serialization_format.md (RFC WS1.T1.2/T1.3/T1.5).
 * Unlike ckc/ir_print.h (human-only, lossy, unparseable), it captures
 * everything needed to reconstruct a ckc_kernel_def_t exactly -- most
 * importantly the explicit SSA value ids -- so the C and Python engines emit
 * byte-identical text for the same IR (killing the SSA-numbering-drift defect
 * class).
 *
 * Public surface (mirrors the Python serialize / parse / canonicalize):
 *
 *   ckc_ir_serialize(k, &text)         KernelDef -> str  (malloc'd, free())
 *   ckc_ir_parse(text, builder, &k)    str -> KernelDef  (built in builder arena)
 *   ckc_ir_canonicalize(text, &out)    str -> normalized str (stable SSA ids,
 *                                                              loc stripped)
 *
 * Float attrs format identically to Python repr(float) (shortest round-trip
 * decimal); the C side reuses the same probe-%.*e-then-shorten algorithm the
 * IR printer already validated against ~200k CPython repr() samples.
 */
#ifndef CKC_IR_SERIALIZE_H
#define CKC_IR_SERIALIZE_H

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Render `k` as `ck.dsl.ir/v1` text. On CKC_OK *out_text receives a freshly
 * malloc'd, NUL-terminated string the caller frees with free(); on failure it
 * is left NULL. Deterministic: sorted attr keys, fixed grammar, repr floats. */
ckc_status_t ckc_ir_serialize(const ckc_kernel_def_t* k, char** out_text);

/* Parse `ck.dsl.ir/v1` text back into a ckc_kernel_def_t. The whole graph is
 * built in `b`'s arena (caller owns `b` and frees it with
 * ckc_ir_builder_free); *out receives the kernel (also b->kernel). On a parse
 * error returns a non-CKC_OK status and sets b's sticky error (the message is
 * available via ckc_ir_builder_error). `b` must be freshly initialised with
 * ckc_ir_builder_init (any kernel name; it is overwritten by the parsed one). */
ckc_status_t ckc_ir_parse(const char* text, ckc_ir_builder_t* b, ckc_kernel_def_t** out);

/* Canonicalize: parse + renumber every SSA id to %0,%1,... in first-definition
 * order (pre-order: params in ABI order, then results/iv/iter-args in textual
 * order, descending into regions) and strip @loc, then re-serialize. Two
 * kernels that differ only in incidental id gaps / authoring locations produce
 * the same canonical string. On CKC_OK *out_text is a malloc'd string (free()).
 * Internally owns and frees its own IRBuilder. */
ckc_status_t ckc_ir_canonicalize(const char* text, char** out_text);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_IR_SERIALIZE_H */
