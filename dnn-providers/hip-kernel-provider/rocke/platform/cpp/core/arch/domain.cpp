// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * domain.cpp -- lookup over the generated arch-domain table. See
 * rocke/arch_domain.h.
 *
 * The data is not in this file. CMake runs tools/gen_arch_domain_cpp.py over the
 * committed columns and the result is included below; this file only searches
 * it. Python: rocke.core.arch.domain.load(flavor).lookup(key, arch), filtered by
 * is_negative().
 */
#include "rocke/arch_domain.h"

#include <stdio.h>
#include <string.h>

#include <string>

namespace
{

/* One flavor's column: its negative cells, sorted by (key, arch). */
struct column_t
{
    const char* flavor;
    const rocke_arch_domain_cell_t* cells;
    size_t count;
};

/* Defines ROCKE_ARCH_DOMAIN_DEFAULT_MODE, the per-flavor cell arrays and
 * ROCKE_ARCH_DOMAIN_COLUMNS. Generated into the build tree; never edit or
 * commit it. */
#include "rocke/arch_domain_table.inc"

const column_t* find_column(const char* flavor)
{
    for(const column_t& c : ROCKE_ARCH_DOMAIN_COLUMNS)
    {
        if(strcmp(c.flavor, flavor) == 0)
        {
            return &c;
        }
    }
    return nullptr;
}

/* Orders cells as the generator sorts them: by key, then arch, both compared
 * byte-wise -- which is Python's str ordering because both are ASCII. */
int compare(const rocke_arch_domain_cell_t& cell, const char* key, const char* arch)
{
    const int by_key = strcmp(cell.key, key);
    return by_key != 0 ? by_key : strcmp(cell.arch, arch);
}

bool is_space(char c)
{
    return c != '\0' && strchr(" \t\n\r\f\v", c) != nullptr;
}

} // namespace

extern "C" const rocke_arch_domain_cell_t*
    rocke_arch_domain_negative(const char* flavor, const char* key, const char* arch)
{
    if(flavor == nullptr || key == nullptr || arch == nullptr)
    {
        return nullptr;
    }
    const column_t* col = find_column(flavor);
    if(col == nullptr)
    {
        return nullptr;
    }
    size_t lo = 0;
    size_t hi = col->count;
    while(lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        const int c = compare(col->cells[mid], key, arch);
        if(c == 0)
        {
            return &col->cells[mid];
        }
        if(c < 0)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return nullptr;
}

extern "C" const rocke_arch_domain_cell_t* rocke_arch_domain_column(const char* flavor,
                                                                    size_t* count)
{
    const column_t* col = flavor != nullptr ? find_column(flavor) : nullptr;
    if(count != nullptr)
    {
        *count = col != nullptr ? col->count : 0;
    }
    return col != nullptr && col->count != 0 ? col->cells : nullptr;
}

extern "C" const char* rocke_arch_domain_default_mode(void)
{
    return ROCKE_ARCH_DOMAIN_DEFAULT_MODE;
}

extern "C" rocke_status_t rocke_arch_domain_mode(const char* raw,
                                                 rocke_arch_domain_mode_t* out,
                                                 char* err,
                                                 size_t err_cap)
{
    /* Python: domain.mode(). */
    const char* begin = raw != nullptr ? raw : "";
    while(is_space(*begin))
    {
        ++begin;
    }
    size_t len = strlen(begin);
    while(len > 0 && is_space(begin[len - 1]))
    {
        --len;
    }
    const std::string value = len != 0 ? std::string(begin, len) : ROCKE_ARCH_DOMAIN_DEFAULT_MODE;
    std::string lower = value;
    for(char& ch : lower)
    {
        if(ch >= 'A' && ch <= 'Z')
        {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    rocke_arch_domain_mode_t mode;
    if(lower == "error")
    {
        mode = ROCKE_ARCH_DOMAIN_ERROR;
    }
    else if(lower == "warn")
    {
        mode = ROCKE_ARCH_DOMAIN_WARN;
    }
    else if(lower == "off")
    {
        mode = ROCKE_ARCH_DOMAIN_OFF;
    }
    else
    {
        if(err != nullptr && err_cap != 0)
        {
            snprintf(err,
                     err_cap,
                     "%s='%s' is not a mode; expected one of error, warn, off "
                     "(unset means %s).",
                     ROCKE_ARCH_DOMAIN_ENV,
                     value.c_str(),
                     ROCKE_ARCH_DOMAIN_DEFAULT_MODE);
        }
        return ROCKE_ERR_VALUE;
    }
    if(out != nullptr)
    {
        *out = mode;
    }
    return ROCKE_OK;
}

extern "C" void rocke_arch_domain_message(const rocke_arch_domain_cell_t* cell,
                                          const char* flavor,
                                          char* out,
                                          size_t cap)
{
    /* Python: domain.message(); snprintf's cut matches MESSAGE_CAP. */
    if(cell == nullptr || flavor == nullptr || out == nullptr || cap == 0)
    {
        return;
    }
    const bool has_evidence = cell->evidence[0] != '\0';
    snprintf(out,
             cap,
             "arch-domain: intrinsic '%s' is not available on %s at %s (%s)%s%s",
             cell->key,
             cell->arch,
             flavor,
             cell->status,
             has_evidence ? ": " : "",
             cell->evidence);
}
