/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
#include "ckc/strbuf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ckc_strbuf_reserve(ckc_strbuf_t *sb, size_t extra) {
    if (sb->oom) {
        return -1;
    }
    size_t need = sb->len + extra + 1; /* +1 for NUL */
    if (need <= sb->cap) {
        return 0;
    }
    size_t newcap = sb->cap ? sb->cap : 64;
    while (newcap < need) {
        newcap *= 2;
    }
    char *p = (char *)realloc(sb->data, newcap);
    if (!p) {
        sb->oom = 1;
        return -1;
    }
    sb->data = p;
    sb->cap = newcap;
    return 0;
}

int ckc_strbuf_init(ckc_strbuf_t *sb, size_t initial_cap) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->oom = 0;
    if (initial_cap) {
        sb->data = (char *)malloc(initial_cap);
        if (!sb->data) {
            sb->oom = 1;
            return -1;
        }
        sb->cap = initial_cap;
        sb->data[0] = '\0';
    }
    return 0;
}

int ckc_strbuf_append_n(ckc_strbuf_t *sb, const char *s, size_t n) {
    if (ckc_strbuf_reserve(sb, n) != 0) {
        return -1;
    }
    memcpy(sb->data + sb->len, s, n);
    sb->len += n;
    sb->data[sb->len] = '\0';
    return 0;
}

int ckc_strbuf_append(ckc_strbuf_t *sb, const char *s) {
    if (!s) {
        return 0;
    }
    return ckc_strbuf_append_n(sb, s, strlen(s));
}

int ckc_strbuf_append_char(ckc_strbuf_t *sb, char c) {
    return ckc_strbuf_append_n(sb, &c, 1);
}

int ckc_strbuf_vappendf(ckc_strbuf_t *sb, const char *fmt, va_list ap) {
    va_list aq;
    va_copy(aq, ap);
    int n = vsnprintf(NULL, 0, fmt, aq);
    va_end(aq);
    if (n < 0) {
        return -1;
    }
    if (ckc_strbuf_reserve(sb, (size_t)n) != 0) {
        return -1;
    }
    vsnprintf(sb->data + sb->len, (size_t)n + 1, fmt, ap);
    sb->len += (size_t)n;
    return 0;
}

int ckc_strbuf_appendf(ckc_strbuf_t *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = ckc_strbuf_vappendf(sb, fmt, ap);
    va_end(ap);
    return r;
}

void ckc_strbuf_clear(ckc_strbuf_t *sb) {
    sb->len = 0;
    if (sb->data && sb->cap) {
        sb->data[0] = '\0';
    }
}

const char *ckc_strbuf_cstr(const ckc_strbuf_t *sb) {
    return sb->data ? sb->data : "";
}

char *ckc_strbuf_detach(ckc_strbuf_t *sb) {
    char *p = sb->data;
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->oom = 0;
    return p;
}

void ckc_strbuf_free(ckc_strbuf_t *sb) {
    free(sb->data);
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    sb->oom = 0;
}
