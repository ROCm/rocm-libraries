// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Private immutable counterpart of Python target._op_id_family(). Owns the
 * sorted index; borrows IDs and families from the static architecture catalogs. */
#ifndef ROCKE_MMA_FAMILY_INDEX_H
#define ROCKE_MMA_FAMILY_INDEX_H

#include <stdlib.h>
#include <string.h>

#include "rocke/arch_target_internal.h"
#include "rocke/error.hpp"

struct rocke_ati_mma_family_index
{
    rocke_ati_mma_family_index(const rocke_ati_arch_row_t* registry, int count)
    {
        size_t capacity = 0;
        for(int i = 0; i < count; ++i)
            if(registry[i].target)
                capacity += registry[i].target->mma.num_ops;
        if(!capacity)
            return;
        rows = static_cast<row*>(malloc(capacity * sizeof(row)));
        if(!rows)
            ckc::raise_status(ROCKE_ERR_OOM, "MMA family index allocation failed");
        for(int i = 0; i < count; ++i)
        {
            const auto* target = registry[i].target;
            if(!target)
                continue;
            for(int j = 0; j < target->mma.num_ops; ++j)
            {
                const auto& op = target->mma.ops[j];
                rows[size++] = {op.op_id, op.family, false};
            }
        }
        qsort(rows, size, sizeof(row), [](const void* a, const void* b) {
            return strcmp(static_cast<const row*>(a)->op_id, static_cast<const row*>(b)->op_id);
        });
        size_t unique = 0;
        for(size_t i = 0; i < size; ++i)
        {
            if(unique && strcmp(rows[unique - 1].op_id, rows[i].op_id) == 0)
                rows[unique - 1].conflict |= strcmp(rows[unique - 1].family, rows[i].family) != 0;
            else
                rows[unique++] = rows[i];
        }
        size = unique;
    }

    ~rocke_ati_mma_family_index()
    {
        free(rows);
    }
    rocke_ati_mma_family_index(const rocke_ati_mma_family_index&) = delete;
    rocke_ati_mma_family_index& operator=(const rocke_ati_mma_family_index&) = delete;

    const char* lookup(const char* op_id) const
    {
        if(!op_id)
            return NULL;
        size_t first = 0, last = size;
        while(first < last)
        {
            const size_t mid = first + (last - first) / 2;
            const int cmp = strcmp(op_id, rows[mid].op_id);
            if(cmp < 0)
                last = mid;
            else if(cmp > 0)
                first = mid + 1;
            else
            {
                if(rows[mid].conflict)
                    ckc::raise_status(
                        ROCKE_ERR_VALUE,
                        "arch SSOT drift: op_id has inconsistent family across arches");
                return rows[mid].family;
            }
        }
        return NULL;
    }

private:
    struct row
    {
        const char* op_id;
        const char* family;
        bool conflict;
    };
    row* rows = NULL;
    size_t size = 0;
};

#endif /* ROCKE_MMA_FAMILY_INDEX_H */
