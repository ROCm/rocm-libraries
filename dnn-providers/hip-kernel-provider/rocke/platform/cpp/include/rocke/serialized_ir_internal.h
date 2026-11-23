/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/serialized_ir_internal.h -- PRIVATE helpers shared by the entry points
 * that take a kernel as serialized ck.dsl.ir/v1 text and lower it:
 * rocke_lower_serialized_ir_to_llvm (rocke/lower_llvm.h) and
 * rocke_lower_serialized_ir_to_hip (rocke/lower_hip.h). NOT a public API.
 *
 * Both entries parse the same way and report a refusal the same way: a reason
 * that is never empty, appended to the caller's optional message buffer. Keeping
 * the parse step here gives the two one behaviour for an empty text, a builder
 * that cannot allocate, and text that is not a kernel.
 */
#ifndef ROCKE_SERIALIZED_IR_INTERNAL_H
#define ROCKE_SERIALIZED_IR_INTERNAL_H

#include "rocke/ir.h"
#include "rocke/strbuf.h"

namespace ckc
{

/* The builder's message for the step that just failed, or `fallback` when the
 * step set none. */
const char* serialized_ir_step_message(const rocke_ir_builder_t* b, const char* fallback);

/* Append `reason` to `message`, when the caller passed a buffer. */
void serialized_ir_report(rocke_strbuf_t* message, const char* reason);

/* Initialise `b` and parse `ir_text` into it.
 *
 * On ROCKE_OK, *out_kernel is the parsed kernel; it lives in `b`, which the
 * caller frees with rocke_ir_builder_free. On any other status there is nothing
 * to free, *out_kernel is NULL, and the reason is appended to `message` (when
 * non-NULL): "empty IR input" for NULL or empty text (ROCKE_ERR_VALUE),
 * "builder init failed (OOM)" (ROCKE_ERR_OOM), or the parser's own message. */
rocke_status_t serialized_ir_parse(const char* ir_text,
                                   rocke_ir_builder_t* b,
                                   rocke_kernel_def_t** out_kernel,
                                   rocke_strbuf_t* message);

/* Hand the reason in `message` to the caller as malloc'd text: *out_message
 * receives it (NULL when allocating it failed), and `message` is left empty.
 * A NULL out_message discards it. */
void serialized_ir_take_message(rocke_strbuf_t* message, char** out_message);

} // namespace ckc

#endif /* ROCKE_SERIALIZED_IR_INTERNAL_H */
