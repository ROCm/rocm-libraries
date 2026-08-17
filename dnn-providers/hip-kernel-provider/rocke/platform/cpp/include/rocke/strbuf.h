/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/strbuf.h -- growable string builder.
 *
 * Every lowerer in the Python engine accumulates output by appending to a
 * Python list and joining at the end (self.lines / parts.append). This is the
 * stand-in: an owned, realloc-backed byte buffer with printf-style append.
 *
 * Unlike the arena, a strbuf owns a single heap buffer it grows in place, and
 * MUST be freed with rocke_strbuf_free (or have its buffer detached). It is the
 * natural type for the final emitted IR/HIP/LLVM text.
 *
 * The buffer is a standard-layout aggregate so it can live by value on the
 * stack and be shared across the extern "C" ABI unchanged; its members are
 * public for the few call sites that read them directly (the sticky `oom`
 * flag, the `data`/`len` contents). When compiled as C++ it also exposes
 * idiomatic member functions; the formatting core is the same vsnprintf-based
 * code in both worlds, so the emitted bytes are identical. The detach contract
 * is preserved: the returned buffer is malloc/realloc-backed and the caller
 * frees it with free().
 */
#ifndef ROCKE_STRBUF_H
#define ROCKE_STRBUF_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_strbuf
{
    char* data; /* always NUL-terminated when len < cap; may be NULL if cap==0 */
    size_t len; /* number of bytes before the NUL                              */
    size_t cap; /* allocated capacity in bytes                                 */
    int oom; /* sticky: set to 1 once an allocation has failed              */

#ifdef __cplusplus
    /* Idiomatic member API (C++ only). These wrap exactly the same realloc +
     * vsnprintf core as the extern "C" functions below, so the emitted text is
     * byte-identical regardless of which spelling a call site uses. The
     * implementations live in strbuf.c (shared with the C ABI shims). */
    int init(size_t initial_cap = 0);
    int append(const char* s);
    int append_n(const char* s, size_t n);
    int append_char(char c);
    int vappendf(const char* fmt, va_list ap);
    void clear();
    const char* cstr() const;
    char* detach();
    void free_buffer();

    /* printf-style append. Defined inline so the variadic forwarding stays in
     * the header; it forwards to vappendf, keeping the formatting core single. */
    int appendf(const char* fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        int r = this->vappendf(fmt, ap);
        va_end(ap);
        return r;
    }
#endif /* __cplusplus */
} rocke_strbuf_t;

/* Initialise an empty builder. `initial_cap` of 0 defers allocation until the
 * first append. Returns 0 on success, -1 on allocation failure. */
int rocke_strbuf_init(rocke_strbuf_t* sb, size_t initial_cap);

/* Append a NUL-terminated string. Returns 0 on success, -1 on OOM (sticky). */
int rocke_strbuf_append(rocke_strbuf_t* sb, const char* s);

/* Append `n` bytes (may contain embedded NULs). */
int rocke_strbuf_append_n(rocke_strbuf_t* sb, const char* s, size_t n);

/* Append a single character. */
int rocke_strbuf_append_char(rocke_strbuf_t* sb, char c);

/* printf-style append. Returns 0 on success, -1 on OOM (sticky). */
int rocke_strbuf_appendf(rocke_strbuf_t* sb, const char* fmt, ...);
int rocke_strbuf_vappendf(rocke_strbuf_t* sb, const char* fmt, va_list ap);

/* Reset length to 0 (keeps the buffer for reuse). */
void rocke_strbuf_clear(rocke_strbuf_t* sb);

/* Borrow the current contents (NUL-terminated). Valid until the next mutation
 * or free. Returns "" for an empty builder. */
const char* rocke_strbuf_cstr(const rocke_strbuf_t* sb);

/* Hand ownership of the underlying buffer to the caller (who must free() it).
 * The builder is reset to empty. Returns NULL on a builder that never
 * allocated (caller may treat as ""). */
char* rocke_strbuf_detach(rocke_strbuf_t* sb);

/* Free the underlying buffer and zero the builder. */
void rocke_strbuf_free(rocke_strbuf_t* sb);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_STRBUF_H */
