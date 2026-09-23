// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Target-independent dtype names. Mirrors rocke.core.dtypes:
 * normalize_dtype -> rocke_normalize_dtype.
 * No architecture, IR builder, or lowering dependencies.
 */
#include "rocke/dtypes.h"

#include <string.h>

/* =========================================================================
 * target-independent dtype normalization
 * =========================================================================
 *
 * Mirrors core/dtypes.py::normalize_dtype: strip + lower(name), then
 * _DTYPE_ALIASES.get(key, key). The canonical RHS values are interned static
 * strings; unknown spellings pass through as the lowercased text in `lowered`.
 */

typedef struct rocke_dtype_alias
{
    const char* alias; /* lowercased key */
    const char* canonical; /* interned canonical value */
} rocke_dtype_alias_t;

/* Byte-for-byte the _DTYPE_ALIASES map (insertion order is irrelevant: lookup is
 * by exact lowercased key). */
static const rocke_dtype_alias_t k_dtype_aliases[] = {
    {"f16", "fp16"},
    {"half", "fp16"},
    {"fp16", "fp16"},
    {"bf16", "bf16"},
    {"bfloat16", "bf16"},
    {"f32", "fp32"},
    {"float", "fp32"},
    {"fp32", "fp32"},
    {"iu8", "iu8"},
    {"iu4", "iu4"},
    {"i8", "i8"},
    {"int8", "i8"},
    {"i4", "i4"},
    {"int4", "i4"},
    {"i32", ROCKE_DTYPE_I32},
    {"int32", ROCKE_DTYPE_I32},
    {"fp8", "fp8e4m3"},
    {"fp8e4m3", "fp8e4m3"},
    {"bf8", "bf8e5m2"},
    {"bf8e5m2", "bf8e5m2"},
    {"fp6", "fp6e2m3"},
    {"fp6e2m3", "fp6e2m3"},
    {"bf6", "fp6e3m2"},
    {"fp6e3m2", "fp6e3m2"},
    {"fp4", "fp4e2m1"},
    {"fp4e2m1", "fp4e2m1"},
};
#define K_NUM_DTYPE_ALIASES ((int)(sizeof(k_dtype_aliases) / sizeof(k_dtype_aliases[0])))

/* str.strip(): Python strips ASCII whitespace from both ends. */
static int rocke_dtype_is_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

const char* rocke_normalize_dtype(const char* name, char* lowered, size_t cap)
{
    int i;
    size_t n;
    const char* start;
    const char* end;

    if(name == NULL)
    {
        /* Python would raise on None.strip(); the C contract requires a result.
         * Treat as empty string (no alias match, passes through as ""). */
        if(lowered != NULL && cap > 0)
        {
            lowered[0] = '\0';
        }
        return (lowered != NULL && cap > 0) ? lowered : "";
    }

    /* strip(): advance over leading/trailing whitespace */
    start = name;
    while(*start != '\0' && rocke_dtype_is_ws(*start))
    {
        start++;
    }
    end = start + strlen(start);
    while(end > start && rocke_dtype_is_ws(end[-1]))
    {
        end--;
    }

    /* lower() into the caller buffer (the pass-through result) */
    n = (size_t)(end - start);
    if(lowered == NULL || cap == 0)
    {
        /* No scratch: caller guarantees a known spelling. Build a small inline
         * copy on a fixed buffer to look up; if unknown we have nowhere to
         * return it, so fall back to "". */
        static char tmp[64];
        size_t m = n < sizeof(tmp) - 1 ? n : sizeof(tmp) - 1;
        for(i = 0; (size_t)i < m; i++)
        {
            char c = start[i];
            tmp[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
        }
        tmp[m] = '\0';
        for(i = 0; i < K_NUM_DTYPE_ALIASES; i++)
        {
            if(strcmp(tmp, k_dtype_aliases[i].alias) == 0)
            {
                return k_dtype_aliases[i].canonical;
            }
        }
        return "";
    }

    if(n > cap - 1)
    {
        n = cap - 1;
    }
    for(i = 0; (size_t)i < n; i++)
    {
        char c = start[i];
        lowered[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    }
    lowered[n] = '\0';

    /* _DTYPE_ALIASES.get(key, key) */
    for(i = 0; i < K_NUM_DTYPE_ALIASES; i++)
    {
        if(strcmp(lowered, k_dtype_aliases[i].alias) == 0)
        {
            return k_dtype_aliases[i].canonical;
        }
    }
    return lowered;
}

static const rocke_dtype_info_t k_dtype_info[] = {
    {"fp32", ROCKE_DTYPE_FLOAT, 32},   {"fp16", ROCKE_DTYPE_FLOAT, 16},
    {"bf16", ROCKE_DTYPE_FLOAT, 16},   {"fp8e4m3", ROCKE_DTYPE_FLOAT, 8},
    {"bf8e5m2", ROCKE_DTYPE_FLOAT, 8}, {"fp6e2m3", ROCKE_DTYPE_FLOAT, 6},
    {"fp6e3m2", ROCKE_DTYPE_FLOAT, 6}, {"fp4e2m1", ROCKE_DTYPE_FLOAT, 4},
    {"e8m0", ROCKE_DTYPE_FLOAT, 8},    {"e4m3", ROCKE_DTYPE_FLOAT, 8},
    {"e5m3", ROCKE_DTYPE_FLOAT, 8},    {"i4", ROCKE_DTYPE_INTEGER, 4},
    {"iu4", ROCKE_DTYPE_INTEGER, 4},   {"i8", ROCKE_DTYPE_INTEGER, 8},
    {"iu8", ROCKE_DTYPE_INTEGER, 8},   {"i16", ROCKE_DTYPE_INTEGER, 16},
    {"i32", ROCKE_DTYPE_INTEGER, 32},  {"i64", ROCKE_DTYPE_INTEGER, 64},
    {"i1", ROCKE_DTYPE_PREDICATE, 1},
};

const rocke_dtype_info_t* rocke_dtype_info(const char* name)
{
    if(!name)
        return NULL;
    char scratch[64];
    const char* key = rocke_normalize_dtype(name, scratch, sizeof(scratch));
    for(const auto& info : k_dtype_info)
        if(strcmp(key, info.name) == 0)
            return &info;
    return NULL;
}
