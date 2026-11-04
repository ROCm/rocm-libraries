/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/ir_print.h -- MLIR-style textual printer for the C99 CK DSL IR.
 *
 * Faithful port of rocke.core.ir_print.print_ir. The output is human-readable
 * and stable: it is consumed by tests (string fixtures) and dropped into kernel
 * manifests as a `kernel.ir` field for debugging. Byte-identical to the Python
 * printer for any kernel built through the frozen IR contract (rocke/ir.h).
 */
#ifndef ROCKE_IR_PRINT_H
#define ROCKE_IR_PRINT_H

#include "rocke/ir.h"
#include "rocke/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Render `kernel` to its MLIR-style textual form, appending into `out` (which
 * must already be initialised with rocke_strbuf_init). No trailing newline is
 * added, matching the Python "\n".join(...). On allocation failure the strbuf's
 * sticky `oom` flag is set; callers should check it.
 *
 * Mirrors Python: print_ir(kernel: KernelDef) -> str. */
void rocke_print_ir(const rocke_kernel_def_t* kernel, rocke_strbuf_t* out);

/* Convenience wrapper: render `kernel` into a freshly malloc'd, NUL-terminated
 * string the caller must free(). Returns NULL on allocation failure. */
char* rocke_print_ir_alloc(const rocke_kernel_def_t* kernel);

/* Append Python's repr() of the str `s` (UTF-8) to `out`, as repr(s) and an
 * f-string's {s!r} write it: quoted with ' -- or with " when `s` contains '
 * and no " -- with the backslash and the quote character escaped, \t \n \r
 * named and every other ASCII control character (0x00-0x1f, 0x7f) as \xNN
 * (lowercase hex). NULL appends "None", Python's repr(None). The printer
 * quotes str attrs inside lists and region labels with it, and
 * rocke_scheduler_strategy_reason (rocke/ir.h) the refused name.
 *
 * Supported subset: every ASCII string exactly. Bytes >= 0x80 are copied
 * unchanged, which is Python's repr for printable non-ASCII characters
 * (letters, CJK, ...) but not for the non-printable ones Python escapes as
 * \xNN / \uNNNN / \UNNNNNNNN (e.g. U+00A0, U+0085) or for invalid UTF-8. */
void rocke_print_repr_str(rocke_strbuf_t* out, const char* s);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_IR_PRINT_H */
