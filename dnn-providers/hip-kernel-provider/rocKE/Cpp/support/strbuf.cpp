// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Growable string builder. The realloc-based growth and the vsnprintf-based
 * formatting are the single source of truth for both the idiomatic member API
 * and the extern "C" ABI shims (which delegate to it), so the emitted text is
 * byte-identical regardless of which spelling a call site uses.
 */
#include "ckc/strbuf.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Grow the buffer so at least `extra` more bytes (plus the NUL) fit. Returns 0
 * on success, -1 on OOM (and latches the sticky `oom` flag). Capacity doubling
 * from a 64-byte floor matches the original growth schedule exactly. */
static int ckc_strbuf_reserve(ckc_strbuf_t* sb, size_t extra)
{
    if(sb->oom)
    {
        return -1;
    }
    size_t need = sb->len + extra + 1; /* +1 for NUL */
    if(need <= sb->cap)
    {
        return 0;
    }
    size_t newcap = sb->cap ? sb->cap : 64;
    while(newcap < need)
    {
        newcap *= 2;
    }
    char* p = static_cast<char*>(realloc(sb->data, newcap));
    if(!p)
    {
        sb->oom = 1;
        return -1;
    }
    sb->data = p;
    sb->cap  = newcap;
    return 0;
}

/* --------------------------------------------------------------- methods */

int ckc_strbuf::init(size_t initial_cap)
{
    data = nullptr;
    len  = 0;
    cap  = 0;
    oom  = 0;
    if(initial_cap)
    {
        data = static_cast<char*>(malloc(initial_cap));
        if(!data)
        {
            oom = 1;
            return -1;
        }
        cap     = initial_cap;
        data[0] = '\0';
    }
    return 0;
}

int ckc_strbuf::append_n(const char* s, size_t n)
{
    if(ckc_strbuf_reserve(this, n) != 0)
    {
        return -1;
    }
    memcpy(data + len, s, n);
    len += n;
    data[len] = '\0';
    return 0;
}

int ckc_strbuf::append(const char* s)
{
    if(!s)
    {
        return 0;
    }
    return append_n(s, strlen(s));
}

int ckc_strbuf::append_char(char c) { return append_n(&c, 1); }

int ckc_strbuf::vappendf(const char* fmt, va_list ap)
{
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(nullptr, 0, fmt, aq);
    va_end(aq);
    if(n < 0)
    {
        return -1;
    }
    if(ckc_strbuf_reserve(this, static_cast<size_t>(n)) != 0)
    {
        return -1;
    }
    vsnprintf(data + len, static_cast<size_t>(n) + 1, fmt, ap);
    len += static_cast<size_t>(n);
    return 0;
}

void ckc_strbuf::clear()
{
    len = 0;
    if(data && cap)
    {
        data[0] = '\0';
    }
}

const char* ckc_strbuf::cstr() const { return data ? data : ""; }

char* ckc_strbuf::detach()
{
    char* p = data;
    data    = nullptr;
    len     = 0;
    cap     = 0;
    oom     = 0;
    return p;
}

void ckc_strbuf::free_buffer()
{
    free(data);
    data = nullptr;
    len  = 0;
    cap  = 0;
    oom  = 0;
}

/* --------------------------------------------------- extern "C" ABI shims */

extern "C" {

int ckc_strbuf_init(ckc_strbuf_t* sb, size_t initial_cap) { return sb->init(initial_cap); }

int ckc_strbuf_append_n(ckc_strbuf_t* sb, const char* s, size_t n) { return sb->append_n(s, n); }

int ckc_strbuf_append(ckc_strbuf_t* sb, const char* s) { return sb->append(s); }

int ckc_strbuf_append_char(ckc_strbuf_t* sb, char c) { return sb->append_char(c); }

int ckc_strbuf_vappendf(ckc_strbuf_t* sb, const char* fmt, va_list ap)
{
    return sb->vappendf(fmt, ap);
}

int ckc_strbuf_appendf(ckc_strbuf_t* sb, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = sb->vappendf(fmt, ap);
    va_end(ap);
    return r;
}

void ckc_strbuf_clear(ckc_strbuf_t* sb) { sb->clear(); }

const char* ckc_strbuf_cstr(const ckc_strbuf_t* sb) { return sb->cstr(); }

char* ckc_strbuf_detach(ckc_strbuf_t* sb) { return sb->detach(); }

void ckc_strbuf_free(ckc_strbuf_t* sb) { sb->free_buffer(); }

} /* extern "C" */
