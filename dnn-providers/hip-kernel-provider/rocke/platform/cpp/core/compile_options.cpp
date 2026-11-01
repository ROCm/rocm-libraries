// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * compile_options.cpp -- C++ twin of helpers/compile.py
 * _comgr_options_for_kernel (see rocke/compile_options.h).
 */
#include "rocke/compile_options.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace
{

/* Python truthiness of an attr value (absent -> false). */
bool truthy(const rocke_attr_value_t* v)
{
    if(v == nullptr)
        return false;
    switch(v->kind)
    {
    case ROCKE_ATTR_BOOL:
        return v->u.b;
    case ROCKE_ATTR_INT:
        return v->u.i != 0;
    case ROCKE_ATTR_FLOAT:
        return v->u.f != 0.0;
    case ROCKE_ATTR_STR:
        return v->u.s != nullptr && v->u.s[0] != '\0';
    case ROCKE_ATTR_LIST:
        return v->u.list.count > 0;
    case ROCKE_ATTR_INT_LIST:
        return v->u.ilist.count > 0;
    }
    return false;
}

/* Python int() of one "min,max" part: optional surrounding blanks, sign, digits. */
bool parse_int(const char* s, size_t n, long* out)
{
    char buf[64];
    if(n >= sizeof buf)
        return false;
    memcpy(buf, s, n);
    buf[n] = '\0';
    char* p = buf;
    while(*p == ' ' || *p == '\t')
        ++p;
    if(*p == '\0')
        return false;
    char* end = nullptr;
    long v = strtol(p, &end, 10);
    if(end == p)
        return false;
    while(*end == ' ' || *end == '\t')
        ++end;
    if(*end != '\0')
        return false;
    *out = v;
    return true;
}

/* Python _is_zero_agpr_alloc. */
bool zero_agpr_alloc(const rocke_attr_value_t* v)
{
    if(v == nullptr)
        return false;
    if(v->kind == ROCKE_ATTR_STR)
    {
        const char* s = v->u.s != nullptr ? v->u.s : "";
        while(*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')
            ++s;
        size_t len = strlen(s);
        while(len > 0
              && (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\n'
                  || s[len - 1] == '\r'))
            --len;
        const char* comma = static_cast<const char*>(memchr(s, ',', len));
        if(comma == nullptr || memchr(comma + 1, ',', len - (comma + 1 - s)) != nullptr)
            return false;
        long lo = 0, hi = 0;
        if(!parse_int(s, comma - s, &lo) || !parse_int(comma + 1, len - (comma + 1 - s), &hi))
            return false;
        return lo == 0 && hi == 0;
    }
    if(v->kind == ROCKE_ATTR_INT_LIST)
        return v->u.ilist.count == 2 && v->u.ilist.ints[0] == 0 && v->u.ilist.ints[1] == 0;
    if(v->kind == ROCKE_ATTR_LIST && v->u.list.count == 2)
    {
        /* a list of int entries (the parser's general list form) */
        int64_t x[2];
        for(int i = 0; i < 2; ++i)
            if(!rocke_attr_get_int(v->u.list.items[i], "value", &x[i]))
                return false;
        return x[0] == 0 && x[1] == 0;
    }
    return false;
}

} // namespace

extern "C" int rocke_comgr_options_for_kernel(const rocke_kernel_def_t* kernel,
                                              rocke_llvm_flavor_t flavor,
                                              const char** out,
                                              int cap,
                                              char* err,
                                              size_t err_cap)
{
    auto fail = [&](const char* msg) {
        if(err != nullptr && err_cap > 0)
            snprintf(err, err_cap, "%s", msg);
        return -1;
    };
    if(kernel == nullptr || out == nullptr)
        return fail("null kernel or output");
    if(!rocke_llvm_flavor_is_known(flavor))
        return fail("a concrete LLVM flavor is required");
    int n = 0;
    auto push = [&](const char* s) {
        if(n < cap)
            out[n] = s;
        ++n;
    };
    push("-O3");
    const rocke_attr_map_t* attrs = &kernel->attrs;
    if(truthy(rocke_attr_get(attrs, "mfma_vgpr_form"))
       || zero_agpr_alloc(rocke_attr_get(attrs, "agpr_alloc")))
    {
        push("-mllvm");
        push("-amdgpu-mfma-vgpr-form");
    }
    else if(truthy(rocke_attr_get(attrs, "mfma_agpr_form")))
    {
        /* -amdgpu-mfma-vgpr-form exists from LLVM 21 on: every flavor but llvm20. */
        if(flavor != ROCKE_LLVM_FLAVOR_LLVM20)
        {
            push("-mllvm");
            push("-amdgpu-mfma-vgpr-form=0");
        }
    }
    const rocke_attr_value_t* opts = rocke_attr_get(attrs, "llvm_options");
    if(opts != nullptr && truthy(opts))
    {
        if(opts->kind == ROCKE_ATTR_LIST)
        {
            for(int i = 0; i < opts->u.list.count; ++i)
            {
                const char* s = rocke_attr_get_str(opts->u.list.items[i], "value");
                if(s == nullptr)
                    return fail("llvm_options: every entry must be a string");
                push("-mllvm");
                push(s);
            }
        }
        else
        {
            return fail("llvm_options must be a list of strings");
        }
    }
    if(n > cap)
        return fail("option buffer too small");
    return n;
}
