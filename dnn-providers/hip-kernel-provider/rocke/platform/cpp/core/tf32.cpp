// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors the target check of core/tf32.py: whether a target's MMA catalog lists
 * a TF32 MMA. The operand and op checks need no catalog and stay inline in
 * tf32_internal.h. */
#include "rocke/tf32_internal.h"

#include <stdio.h>

#include "rocke/arch_target.h"

/* Whether the MMA catalog of target `gfx` lists `op_id` (an unknown target lists
 * nothing). */
static bool target_lists(const char* gfx, const char* op_id)
{
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(gfx);
    return target && rocke_mma_catalog_by_op_id(&target->mma, op_id);
}

/* The targets whose catalogs list the op are named in known_arches() order. */
bool rocke_tf32_mma_target_error(const char* op_id, const char* gfx, char* out, size_t cap)
{
    if(!rocke_tf32_mma_count(op_id) || target_lists(gfx, op_id))
        return false;
    int used = snprintf(out,
                        cap,
                        "no MMA instruction found for '%s' on '%s' -- the target's MMA catalog "
                        "does not list it; ",
                        op_id,
                        gfx);
    int listed = 0;
    int num_arches = 0;
    const char* const* arches = rocke_known_arches(&num_arches);
    for(int i = 0; i < num_arches; ++i)
    {
        if(!target_lists(arches[i], op_id))
            continue;
        const char* lead = listed++ ? ", " : "it is listed for ";
        if(used >= 0 && (size_t)used < cap)
            used += snprintf(out + used, cap - (size_t)used, "%s%s", lead, arches[i]);
    }
    if(!listed && used >= 0 && (size_t)used < cap)
        snprintf(out + used, cap - (size_t)used, "no target lists it");
    return true;
}
