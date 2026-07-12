// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/arena.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROCKE_ARENA_DEFAULT_BLOCK (64u * 1024u)

struct rocke_arena_block
{
    rocke_arena_block_t* next;
    size_t used;
    size_t cap;
    /* payload follows immediately after this header */
    unsigned char data[1];
};

/* Round up to the platform's max alignment. C99 has no max_align_t (C11),
 * so we derive it from the widest fundamental types via a union. */
typedef union rocke_max_align
{
    long double ld;
    void* p;
    intmax_t i;
} rocke_max_align_u;

static size_t rocke_align_up(size_t n)
{
    const size_t a = sizeof(rocke_max_align_u);
    return (n + (a - 1)) & ~(a - 1);
}

static rocke_arena_block_t* rocke_arena_new_block(size_t payload)
{
    size_t hdr = offsetof(rocke_arena_block_t, data);
    rocke_arena_block_t* b = (rocke_arena_block_t*)malloc(hdr + payload);
    if(!b)
    {
        return NULL;
    }
    b->next = NULL;
    b->used = 0;
    b->cap = payload;
    return b;
}

int rocke_arena_init(rocke_arena_t* a, size_t block_size)
{
    if(!a)
    {
        return -1;
    }
    a->block_size = block_size ? block_size : ROCKE_ARENA_DEFAULT_BLOCK;
    a->total_bytes = 0;
    a->total_alloc = 0;
    a->head = rocke_arena_new_block(a->block_size);
    if(!a->head)
    {
        return -1;
    }
    a->total_alloc += a->head->cap;
    return 0;
}

void* rocke_arena_alloc(rocke_arena_t* a, size_t size)
{
    if(!a || !a->head)
    {
        return NULL;
    }
    size_t need = rocke_align_up(size ? size : 1);
    rocke_arena_block_t* b = a->head;
    if(b->used + need > b->cap)
    {
        /* Allocate a fresh block large enough for this request. */
        size_t payload = need > a->block_size ? need : a->block_size;
        rocke_arena_block_t* nb = rocke_arena_new_block(payload);
        if(!nb)
        {
            return NULL;
        }
        nb->next = a->head;
        a->head = nb;
        a->total_alloc += nb->cap;
        b = nb;
    }
    void* p = b->data + b->used;
    b->used += need;
    a->total_bytes += need;
    return p;
}

void* rocke_arena_calloc(rocke_arena_t* a, size_t size)
{
    void* p = rocke_arena_alloc(a, size);
    if(p)
    {
        memset(p, 0, size);
    }
    return p;
}

char* rocke_arena_strdup(rocke_arena_t* a, const char* s)
{
    if(!s)
    {
        return NULL;
    }
    size_t n = strlen(s) + 1;
    char* p = (char*)rocke_arena_alloc(a, n);
    if(p)
    {
        memcpy(p, s, n);
    }
    return p;
}

char* rocke_arena_printf(rocke_arena_t* a, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if(n < 0)
    {
        return NULL;
    }
    char* p = (char*)rocke_arena_alloc(a, (size_t)n + 1);
    if(!p)
    {
        return NULL;
    }
    va_start(ap, fmt);
    vsnprintf(p, (size_t)n + 1, fmt, ap);
    va_end(ap);
    return p;
}

void rocke_arena_destroy(rocke_arena_t* a)
{
    if(!a)
    {
        return;
    }
    rocke_arena_block_t* b = a->head;
    while(b)
    {
        rocke_arena_block_t* next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
    a->block_size = 0;
    a->total_bytes = 0;
    a->total_alloc = 0;
}
