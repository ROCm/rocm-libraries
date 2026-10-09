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

#include <string.h>

namespace
{

/* One flavor's column: its negative cells, sorted by (key, arch). */
struct column_t
{
    const char* flavor;
    const rocke_arch_domain_cell_t* cells;
    size_t count;
};

/* Defines the per-flavor cell arrays and ROCKE_ARCH_DOMAIN_COLUMNS. Generated
 * into the build tree; never edit or commit it. */
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
