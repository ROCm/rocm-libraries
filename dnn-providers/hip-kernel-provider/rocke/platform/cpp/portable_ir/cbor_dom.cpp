/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/cbor_dom.c -- minimal CBOR (RFC 8949) decoder into the json_dom jd_val_t.
 * See cbor_dom.h. Bounds-checked; recursion-depth limited.
 */
#include "rocke/cbor_dom.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CBOR_MAX_DEPTH 256

typedef struct
{
    const unsigned char* p;
    const unsigned char* end;
    rocke_arena_t* arena;
    char* err;
    size_t err_cap;
    int failed;
} cbor_rd_t;

static void cbor_fail(cbor_rd_t* r, const char* msg)
{
    if(!r->failed && r->err && r->err_cap)
    {
        snprintf(r->err, r->err_cap, "cbor: %s", msg);
    }
    r->failed = 1;
}

/* Read `n` raw bytes big-endian into a u64. */
static uint64_t cbor_rd_be(cbor_rd_t* r, int n)
{
    uint64_t v = 0;
    if(r->p + n > r->end)
    {
        cbor_fail(r, "truncated argument");
        return 0;
    }
    for(int i = 0; i < n; i++)
        v = (v << 8) | (uint64_t)(*r->p++);
    return v;
}

/* Read major type + argument. Returns major; *arg holds the argument. */
static int cbor_rd_head(cbor_rd_t* r, uint64_t* arg)
{
    if(r->p >= r->end)
    {
        cbor_fail(r, "unexpected end");
        return -1;
    }
    unsigned char ib = *r->p++;
    int major = ib >> 5;
    int info = ib & 0x1F;
    if(info < 24)
        *arg = (uint64_t)info;
    else if(info == 24)
        *arg = cbor_rd_be(r, 1);
    else if(info == 25)
        *arg = cbor_rd_be(r, 2);
    else if(info == 26)
        *arg = cbor_rd_be(r, 4);
    else if(info == 27)
        *arg = cbor_rd_be(r, 8);
    else
    {
        cbor_fail(r, "reserved/indefinite length not supported");
        return -1;
    }
    return major;
}

static jd_val_t* cbor_node(cbor_rd_t* r)
{
    jd_val_t* v = (jd_val_t*)rocke_arena_calloc(r->arena, sizeof(jd_val_t));
    if(!v)
        cbor_fail(r, "out of memory");
    return v;
}

/* Copy `n` bytes from the stream into a fresh NUL-terminated arena string. */
static char* cbor_rd_text(cbor_rd_t* r, uint64_t n)
{
    if(r->p + n > r->end)
    {
        cbor_fail(r, "truncated string");
        return NULL;
    }
    char* s = (char*)rocke_arena_alloc(r->arena, (size_t)n + 1);
    if(!s)
    {
        cbor_fail(r, "out of memory");
        return NULL;
    }
    memcpy(s, r->p, (size_t)n);
    s[n] = '\0';
    r->p += n;
    return s;
}

static jd_val_t* cbor_value(cbor_rd_t* r, int depth)
{
    if(r->failed)
        return NULL;
    if(depth > CBOR_MAX_DEPTH)
    {
        cbor_fail(r, "max depth exceeded");
        return NULL;
    }
    if(r->p >= r->end)
    {
        cbor_fail(r, "unexpected end");
        return NULL;
    }

    /* Major 7 simple values / float are dispatched on the initial byte. */
    unsigned char ib = *r->p;
    if(ib == 0xF6 || ib == 0xF7) /* null / undefined -> null */
    {
        r->p++;
        jd_val_t* v = cbor_node(r);
        if(v)
            v->kind = JD_NULL;
        return v;
    }
    if(ib == 0xF4 || ib == 0xF5) /* false / true */
    {
        r->p++;
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_BOOL;
            v->b = (ib == 0xF5);
        }
        return v;
    }
    if(ib == 0xFB) /* float64 */
    {
        r->p++;
        uint64_t bits = cbor_rd_be(r, 8);
        double d;
        memcpy(&d, &bits, sizeof(d));
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_NUM;
            v->num = d;
        }
        return v;
    }
    if(ib == 0xFA) /* float32 */
    {
        r->p++;
        uint32_t bits = (uint32_t)cbor_rd_be(r, 4);
        float f;
        memcpy(&f, &bits, sizeof(f));
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_NUM;
            v->num = (double)f;
        }
        return v;
    }

    uint64_t arg = 0;
    int major = cbor_rd_head(r, &arg);
    if(r->failed)
        return NULL;

    switch(major)
    {
    case 0: /* unsigned int */
    {
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_NUM;
            v->num = (double)arg;
        }
        return v;
    }
    case 1: /* negative int: -1 - arg */
    {
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_NUM;
            v->num = -1.0 - (double)arg;
        }
        return v;
    }
    case 3: /* text string */
    {
        char* s = cbor_rd_text(r, arg);
        if(!s)
            return NULL;
        jd_val_t* v = cbor_node(r);
        if(v)
        {
            v->kind = JD_STR;
            v->str = s;
        }
        return v;
    }
    case 4: /* array */
    {
        jd_val_t* v = cbor_node(r);
        if(!v)
            return NULL;
        v->kind = JD_ARR;
        v->arr_len = (int)arg;
        if(arg > 0)
        {
            v->arr = (jd_val_t**)rocke_arena_calloc(r->arena, sizeof(jd_val_t*) * (size_t)arg);
            if(!v->arr)
            {
                cbor_fail(r, "out of memory");
                return NULL;
            }
            for(uint64_t i = 0; i < arg; i++)
            {
                v->arr[i] = cbor_value(r, depth + 1);
                if(r->failed)
                    return NULL;
            }
        }
        return v;
    }
    case 5: /* map */
    {
        jd_val_t* v = cbor_node(r);
        if(!v)
            return NULL;
        v->kind = JD_OBJ;
        v->obj_len = (int)arg;
        if(arg > 0)
        {
            v->obj = (jd_member_t*)rocke_arena_calloc(r->arena, sizeof(jd_member_t) * (size_t)arg);
            if(!v->obj)
            {
                cbor_fail(r, "out of memory");
                return NULL;
            }
            for(uint64_t i = 0; i < arg; i++)
            {
                /* keys must be text strings */
                if(r->p >= r->end || (*r->p >> 5) != 3)
                {
                    cbor_fail(r, "map key must be a text string");
                    return NULL;
                }
                uint64_t klen = 0;
                cbor_rd_head(r, &klen);
                if(r->failed)
                    return NULL;
                v->obj[i].key = cbor_rd_text(r, klen);
                if(r->failed)
                    return NULL;
                v->obj[i].val = cbor_value(r, depth + 1);
                if(r->failed)
                    return NULL;
            }
        }
        return v;
    }
    default:
        cbor_fail(r, "unsupported major type");
        return NULL;
    }
}

jd_val_t* rocke_cbor_parse(
    const unsigned char* data, size_t len, rocke_arena_t* arena, char* err, size_t err_cap)
{
    if(err && err_cap)
        err[0] = '\0';
    cbor_rd_t r;
    r.p = data;
    r.end = data + len;
    r.arena = arena;
    r.err = err;
    r.err_cap = err_cap;
    r.failed = 0;
    jd_val_t* root = cbor_value(&r, 0);
    if(r.failed)
        return NULL;
    return root;
}
