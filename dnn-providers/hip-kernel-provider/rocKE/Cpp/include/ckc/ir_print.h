/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/ir_print.h -- MLIR-style textual printer for the C99 CK DSL IR.
 *
 * Faithful port of ck_dsl.core.ir_print.print_ir. The output is human-readable
 * and stable: it is consumed by tests (string fixtures) and dropped into kernel
 * manifests as a `kernel.ir` field for debugging. Byte-identical to the Python
 * printer for any kernel built through the frozen IR contract (ckc/ir.h).
 */
#ifndef CKC_IR_PRINT_H
#define CKC_IR_PRINT_H

#include "ckc/ir.h"
#include "ckc/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Render `kernel` to its MLIR-style textual form, appending into `out` (which
 * must already be initialised with ckc_strbuf_init). No trailing newline is
 * added, matching the Python "\n".join(...). On allocation failure the strbuf's
 * sticky `oom` flag is set; callers should check it.
 *
 * Mirrors Python: print_ir(kernel: KernelDef) -> str. */
void ckc_print_ir(const ckc_kernel_def_t* kernel, ckc_strbuf_t* out);

/* Convenience wrapper: render `kernel` into a freshly malloc'd, NUL-terminated
 * string the caller must free(). Returns NULL on allocation failure. */
char* ckc_print_ir_alloc(const ckc_kernel_def_t* kernel);

#ifdef __cplusplus
}
#endif

#endif /* CKC_IR_PRINT_H */
