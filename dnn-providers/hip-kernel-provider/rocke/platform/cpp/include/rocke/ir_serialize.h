/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/ir_serialize.h -- `ck.dsl.ir/v1` round-trippable IR serialization.
 *
 * Faithful C99 port of rocke/core/ir_serialize.py. This is the MACHINE
 * interchange format specified in
 * dsl_docs/architecture/ir_serialization_format.md.
 * Unlike rocke/ir_print.h (human-only, lossy, unparseable), it captures
 * everything needed to reconstruct a rocke_kernel_def_t exactly -- most
 * importantly the explicit SSA value ids -- so the C and Python engines emit
 * byte-identical text for the same IR (killing the SSA-numbering-drift defect
 * class).
 *
 * Public surface (mirrors the Python serialize / parse / canonicalize):
 *
 *   rocke_ir_serialize(k, &text)         KernelDef -> str  (malloc'd, free())
 *   rocke_ir_parse(text, builder, &k)    str -> KernelDef  (built in builder arena)
 *   rocke_ir_canonicalize(text, &out)    str -> normalized str (stable SSA ids,
 *                                                              loc stripped)
 *
 * Float attrs format identically to Python repr(float) (shortest round-trip
 * decimal); the C side reuses the same probe-%.*e-then-shorten algorithm the
 * IR printer already validated against ~200k CPython repr() samples.
 */
#ifndef ROCKE_IR_SERIALIZE_H
#define ROCKE_IR_SERIALIZE_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Render `k` as `ck.dsl.ir/v1` text. On ROCKE_OK *out_text receives a freshly
 * malloc'd, NUL-terminated string the caller frees with free(); on failure it
 * is left NULL. Deterministic: sorted attr keys, fixed grammar, repr floats. */
rocke_status_t rocke_ir_serialize(const rocke_kernel_def_t* k, char** out_text);

/* Parse `ck.dsl.ir/v1` text back into a rocke_kernel_def_t. The whole graph is
 * built in `b`'s arena (caller owns `b` and frees it with
 * rocke_ir_builder_free); *out receives the kernel (also b->kernel). On a parse
 * error returns a non-ROCKE_OK status and sets b's sticky error (the message is
 * available via rocke_ir_builder_error). `b` must be freshly initialised with
 * rocke_ir_builder_init (any kernel name; it is overwritten by the parsed one). */
rocke_status_t rocke_ir_parse(const char* text, rocke_ir_builder_t* b, rocke_kernel_def_t** out);

/* Canonicalize: parse + renumber every SSA id to %0,%1,... in first-definition
 * order (pre-order: params in ABI order, then results/iv/iter-args in textual
 * order, descending into regions) and strip @loc, then re-serialize. Two
 * kernels that differ only in incidental id gaps / authoring locations produce
 * the same canonical string. On ROCKE_OK *out_text is a malloc'd string (free()).
 * Internally owns and frees its own IRBuilder. */
rocke_status_t rocke_ir_canonicalize(const char* text, char** out_text);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_IR_SERIALIZE_H */
