/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * json_dom.c -- arena-backed recursive-descent JSON parser (see json_dom.h).
 */
#include "rocke/json_dom.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    const char* p;
    const char* end;
    rocke_arena_t* arena;
    char err[256];
    bool failed;
} jdp_t;

static void jdp_fail(jdp_t* jp, const char* msg)
{
    if(!jp->failed)
    {
        snprintf(jp->err, sizeof jp->err, "json: %s", msg);
        jp->failed = true;
    }
}

static void jdp_ws(jdp_t* jp)
{
    while(jp->p < jp->end)
    {
        char c = *jp->p;
        if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
            jp->p++;
        else
            break;
    }
}

static jd_val_t* jd_new(jdp_t* jp, jd_kind_t k)
{
    jd_val_t* v = (jd_val_t*)rocke_arena_calloc(jp->arena, sizeof(jd_val_t));
    if(v)
        v->kind = k;
    return v;
}

static jd_val_t* jdp_value(jdp_t* jp);

static char* jdp_string_raw(jdp_t* jp)
{
    if(jp->p >= jp->end || *jp->p != '"')
    {
        jdp_fail(jp, "expected string");
        return NULL;
    }
    jp->p++;
    const char* start = jp->p;
    const char* q = start;
    bool closed = false;
    while(q < jp->end)
    {
        if(*q == '"')
        {
            closed = true;
            break;
        }
        if(*q == '\\')
        {
            q++;
            if(q >= jp->end)
                break;
        }
        q++;
    }
    if(!closed)
    {
        jdp_fail(jp, "unterminated string");
        return NULL;
    }
    size_t cap = (size_t)(q - start) + 1;
    char* out = (char*)rocke_arena_alloc(jp->arena, cap);
    if(!out)
    {
        jdp_fail(jp, "oom string");
        return NULL;
    }
    size_t n = 0;
    while(jp->p < jp->end)
    {
        char c = *jp->p++;
        if(c == '"')
        {
            out[n] = '\0';
            return out;
        }
        if(c == '\\')
        {
            if(jp->p >= jp->end)
                break;
            char e = *jp->p++;
            switch(e)
            {
            case '"':
                out[n++] = '"';
                break;
            case '\\':
                out[n++] = '\\';
                break;
            case '/':
                out[n++] = '/';
                break;
            case 'n':
                out[n++] = '\n';
                break;
            case 't':
                out[n++] = '\t';
                break;
            case 'r':
                out[n++] = '\r';
                break;
            case 'b':
                out[n++] = '\b';
                break;
            case 'f':
                out[n++] = '\f';
                break;
            case 'u':
            {
                if(jp->end - jp->p < 4)
                {
                    jdp_fail(jp, "bad \\u escape");
                    return NULL;
                }
                unsigned cp = 0;
                for(int i = 0; i < 4; i++)
                {
                    char h = *jp->p++;
                    cp <<= 4;
                    if(h >= '0' && h <= '9')
                        cp |= (unsigned)(h - '0');
                    else if(h >= 'a' && h <= 'f')
                        cp |= (unsigned)(h - 'a' + 10);
                    else if(h >= 'A' && h <= 'F')
                        cp |= (unsigned)(h - 'A' + 10);
                    else
                    {
                        jdp_fail(jp, "bad hex in \\u");
                        return NULL;
                    }
                }
                if(cp < 0x80)
                {
                    out[n++] = (char)cp;
                }
                else if(cp < 0x800)
                {
                    out[n++] = (char)(0xC0 | (cp >> 6));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                else
                {
                    out[n++] = (char)(0xE0 | (cp >> 12));
                    out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[n++] = (char)(0x80 | (cp & 0x3F));
                }
                break;
            }
            default:
                jdp_fail(jp, "bad escape");
                return NULL;
            }
        }
        else
        {
            out[n++] = c;
        }
    }
    jdp_fail(jp, "unterminated string");
    return NULL;
}

static jd_val_t* jdp_number(jdp_t* jp)
{
    char* endp = NULL;
    double d = strtod(jp->p, &endp);
    if(endp == jp->p)
    {
        jdp_fail(jp, "bad number");
        return NULL;
    }
    jp->p = endp;
    jd_val_t* v = jd_new(jp, JD_NUM);
    if(v)
        v->num = d;
    return v;
}

static bool jdp_lit(jdp_t* jp, const char* lit)
{
    size_t n = strlen(lit);
    if((size_t)(jp->end - jp->p) < n || strncmp(jp->p, lit, n) != 0)
        return false;
    jp->p += n;
    return true;
}

static jd_val_t* jdp_array(jdp_t* jp)
{
    jp->p++;
    jd_val_t* v = jd_new(jp, JD_ARR);
    if(!v)
    {
        jdp_fail(jp, "oom arr");
        return NULL;
    }
    int cap = 0;
    jdp_ws(jp);
    if(jp->p < jp->end && *jp->p == ']')
    {
        jp->p++;
        return v;
    }
    while(jp->p < jp->end)
    {
        jd_val_t* item = jdp_value(jp);
        if(jp->failed)
            return NULL;
        if(v->arr_len == cap)
        {
            int nc = cap ? cap * 2 : 8;
            jd_val_t** na
                = (jd_val_t**)rocke_arena_alloc(jp->arena, (size_t)nc * sizeof(jd_val_t*));
            if(!na)
            {
                jdp_fail(jp, "oom arr grow");
                return NULL;
            }
            if(v->arr && v->arr_len > 0)
                memcpy(na, v->arr, (size_t)v->arr_len * sizeof(jd_val_t*));
            v->arr = na;
            cap = nc;
        }
        v->arr[v->arr_len++] = item;
        jdp_ws(jp);
        if(jp->p < jp->end && *jp->p == ',')
        {
            jp->p++;
            jdp_ws(jp);
            continue;
        }
        if(jp->p < jp->end && *jp->p == ']')
        {
            jp->p++;
            return v;
        }
        jdp_fail(jp, "expected ',' or ']'");
        return NULL;
    }
    jdp_fail(jp, "unterminated array");
    return NULL;
}

static jd_val_t* jdp_object(jdp_t* jp)
{
    jp->p++;
    jd_val_t* v = jd_new(jp, JD_OBJ);
    if(!v)
    {
        jdp_fail(jp, "oom obj");
        return NULL;
    }
    int cap = 0;
    jdp_ws(jp);
    if(jp->p < jp->end && *jp->p == '}')
    {
        jp->p++;
        return v;
    }
    while(jp->p < jp->end)
    {
        jdp_ws(jp);
        char* key = jdp_string_raw(jp);
        if(jp->failed)
            return NULL;
        jdp_ws(jp);
        if(jp->p >= jp->end || *jp->p != ':')
        {
            jdp_fail(jp, "expected ':'");
            return NULL;
        }
        jp->p++;
        jdp_ws(jp);
        jd_val_t* val = jdp_value(jp);
        if(jp->failed)
            return NULL;
        if(v->obj_len == cap)
        {
            int nc = cap ? cap * 2 : 8;
            jd_member_t* nm
                = (jd_member_t*)rocke_arena_alloc(jp->arena, (size_t)nc * sizeof(jd_member_t));
            if(!nm)
            {
                jdp_fail(jp, "oom obj grow");
                return NULL;
            }
            if(v->obj && v->obj_len > 0)
                memcpy(nm, v->obj, (size_t)v->obj_len * sizeof(jd_member_t));
            v->obj = nm;
            cap = nc;
        }
        v->obj[v->obj_len].key = key;
        v->obj[v->obj_len].val = val;
        v->obj_len++;
        jdp_ws(jp);
        if(jp->p < jp->end && *jp->p == ',')
        {
            jp->p++;
            continue;
        }
        if(jp->p < jp->end && *jp->p == '}')
        {
            jp->p++;
            return v;
        }
        jdp_fail(jp, "expected ',' or '}'");
        return NULL;
    }
    jdp_fail(jp, "unterminated object");
    return NULL;
}

static jd_val_t* jdp_value(jdp_t* jp)
{
    jdp_ws(jp);
    if(jp->p >= jp->end)
    {
        jdp_fail(jp, "unexpected end");
        return NULL;
    }
    char c = *jp->p;
    if(c == '"')
    {
        char* s = jdp_string_raw(jp);
        if(!s)
            return NULL;
        jd_val_t* v = jd_new(jp, JD_STR);
        if(v)
            v->str = s;
        return v;
    }
    if(c == '{')
        return jdp_object(jp);
    if(c == '[')
        return jdp_array(jp);
    if(c == '-' || (c >= '0' && c <= '9'))
        return jdp_number(jp);
    if(jdp_lit(jp, "true"))
    {
        jd_val_t* v = jd_new(jp, JD_BOOL);
        if(v)
            v->b = true;
        return v;
    }
    if(jdp_lit(jp, "false"))
    {
        jd_val_t* v = jd_new(jp, JD_BOOL);
        if(v)
            v->b = false;
        return v;
    }
    if(jdp_lit(jp, "null"))
        return jd_new(jp, JD_NULL);
    jdp_fail(jp, "unexpected token");
    return NULL;
}

jd_val_t* rocke_json_parse(const char* text, rocke_arena_t* arena, char* err, size_t err_cap)
{
    jdp_t jp;
    jp.p = text;
    jp.end = text + strlen(text);
    jp.arena = arena;
    jp.err[0] = '\0';
    jp.failed = false;
    jd_val_t* root = jdp_value(&jp);
    if(jp.failed)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", jp.err);
        return NULL;
    }
    return root;
}

jd_val_t* rocke_jget(const jd_val_t* o, const char* key)
{
    if(!o || o->kind != JD_OBJ)
        return NULL;
    for(int i = 0; i < o->obj_len; i++)
        if(strcmp(o->obj[i].key, key) == 0)
            return o->obj[i].val;
    return NULL;
}

const char* rocke_jstr(const jd_val_t* v)
{
    return (v && v->kind == JD_STR) ? v->str : NULL;
}

bool rocke_jnum(const jd_val_t* v, double* out)
{
    if(!v || v->kind != JD_NUM)
        return false;
    if(out)
        *out = v->num;
    return true;
}
