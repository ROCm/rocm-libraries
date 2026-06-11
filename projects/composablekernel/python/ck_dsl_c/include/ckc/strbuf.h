/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/strbuf.h -- growable string builder.
 *
 * Every lowerer in the Python engine accumulates output by appending to a
 * Python list and joining at the end (self.lines / parts.append). This is the
 * C99 stand-in: an owned, realloc-backed byte buffer with printf-style append.
 *
 * Unlike the arena, a strbuf owns a single heap buffer it grows in place, and
 * MUST be freed with ckc_strbuf_free (or have its buffer detached). It is the
 * natural type for the final emitted IR/HIP/LLVM text.
 */
#ifndef CKC_STRBUF_H
#define CKC_STRBUF_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ckc_strbuf
{
    char* data; /* always NUL-terminated when len < cap; may be NULL if cap==0 */
    size_t len; /* number of bytes before the NUL                              */
    size_t cap; /* allocated capacity in bytes                                 */
    int oom;    /* sticky: set to 1 once an allocation has failed              */
} ckc_strbuf_t;

/* Initialise an empty builder. `initial_cap` of 0 defers allocation until the
 * first append. Returns 0 on success, -1 on allocation failure. */
int ckc_strbuf_init(ckc_strbuf_t* sb, size_t initial_cap);

/* Append a NUL-terminated string. Returns 0 on success, -1 on OOM (sticky). */
int ckc_strbuf_append(ckc_strbuf_t* sb, const char* s);

/* Append `n` bytes (may contain embedded NULs). */
int ckc_strbuf_append_n(ckc_strbuf_t* sb, const char* s, size_t n);

/* Append a single character. */
int ckc_strbuf_append_char(ckc_strbuf_t* sb, char c);

/* printf-style append. Returns 0 on success, -1 on OOM (sticky). */
int ckc_strbuf_appendf(ckc_strbuf_t* sb, const char* fmt, ...);
int ckc_strbuf_vappendf(ckc_strbuf_t* sb, const char* fmt, va_list ap);

/* Reset length to 0 (keeps the buffer for reuse). */
void ckc_strbuf_clear(ckc_strbuf_t* sb);

/* Borrow the current contents (NUL-terminated). Valid until the next mutation
 * or free. Returns "" for an empty builder. */
const char* ckc_strbuf_cstr(const ckc_strbuf_t* sb);

/* Hand ownership of the underlying buffer to the caller (who must free() it).
 * The builder is reset to empty. Returns NULL on a builder that never
 * allocated (caller may treat as ""). */
char* ckc_strbuf_detach(ckc_strbuf_t* sb);

/* Free the underlying buffer and zero the builder. */
void ckc_strbuf_free(ckc_strbuf_t* sb);

#ifdef __cplusplus
}
#endif

#endif /* CKC_STRBUF_H */
