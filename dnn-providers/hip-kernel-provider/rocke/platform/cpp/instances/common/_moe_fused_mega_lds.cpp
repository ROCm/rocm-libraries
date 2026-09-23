// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * _moe_fused_mega_lds.cpp -- C99 port of
 * rocke/instances/common/_moe_fused_mega_lds.py: the whole-kernel LDS
 * accounting shared by the fused-MoE mega-kernel family.
 *
 * See rocke/helper_rocke.instances.common._moe_fused_mega_lds.h for the symbol
 * map and for why summing the aligned segments is EXACT (not conservative) for
 * the megas. None of these routines emit IR; each mega's build entry calls
 * rocke_validate_mega_lds_budget with the alloc sequence its own _lds_allocs
 * produces, in emitter declaration order.
 */
#include "rocke/helper_rocke.instances.common._moe_fused_mega_lds.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h" /* rocke_arena_printf (reason strings)                */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_* (ArchTarget)   */
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* validate_arch_and_block_size */

/* Longest reject breakdown the shipped megas produce is ~110 chars (five/six
 * buffers); the assembled message is then bounded again by ROCKE_ERR_MSG_CAP at
 * the caller's rocke_i_set_err. Truncation here would only shorten a
 * human-readable reject reason -- it never reaches IR. */
#define ROCKE_MEGA_LDS_BREAKDOWN_CAP 512

long rocke_mega_lds_alloc_nbytes(const rocke_mega_lds_alloc_t* alloc)
{
    if(alloc == NULL)
    {
        return 0;
    }
    return (long)alloc->elem_bytes * (long)alloc->elem_count;
}

int rocke_mega_lds_elem_bytes(const rocke_type_t* dtype)
{
    /* Mirror of lower_llvm's _elem_bytes map (and its width-2 fallback) so the
     * accounting cannot disagree with the packer. */
    const char* n = (dtype != NULL) ? dtype->name : NULL;
    if(n == NULL)
    {
        return 2;
    }
    if(strcmp(n, "i8") == 0 || strcmp(n, "fp8e4m3") == 0 || strcmp(n, "bf8e5m2") == 0)
    {
        return 1;
    }
    if(strcmp(n, "f16") == 0 || strcmp(n, "bf16") == 0)
    {
        return 2;
    }
    if(strcmp(n, "i32") == 0 || strcmp(n, "f32") == 0)
    {
        return 4;
    }
    if(strcmp(n, "i64") == 0)
    {
        return 8;
    }
    return 2;
}

/* _seg_align(elem_bytes): the segment alignment the packer applies -- 16 B for
 * byte-element types, else 4. */
static long rocke_mega_lds_seg_align(int elem_bytes)
{
    return (elem_bytes == 1) ? 16 : 4;
}

long rocke_mega_lds_pool_bytes(const rocke_mega_lds_alloc_t* allocs, size_t num_allocs)
{
    long end = 0;
    size_t i;

    if(allocs == NULL)
    {
        return 0;
    }
    for(i = 0; i < num_allocs; ++i)
    {
        long aln = rocke_mega_lds_seg_align(allocs[i].elem_bytes);
        end = ((end + aln - 1) & ~(aln - 1)) + rocke_mega_lds_alloc_nbytes(&allocs[i]);
    }
    return (end + 15) & ~15L;
}

bool rocke_validate_mega_lds_budget(rocke_ir_builder_t* b,
                                    const rocke_mega_lds_alloc_t* allocs,
                                    size_t num_allocs,
                                    const char* arch,
                                    const char** out_reason)
{
    const rocke_archtarget_t* target = NULL;
    const char* arch_reason = NULL;
    char breakdown[ROCKE_MEGA_LDS_BREAKDOWN_CAP];
    size_t pos = 0;
    size_t i;
    long total;

    /* try: target = ArchTarget.from_gfx(arch) / except KeyError as e: return
     * False, str(e).
     *
     * block_size 0 can never exceed a target's max_threads_per_block, so this
     * call reduces to exactly that from_gfx lookup plus the one reconstruction
     * of the Python KeyError repr the engine owns. (Unreachable from either
     * mega in practice: both run an arch gate before this one.) */
    if(!rocke_validate_arch_and_block_size(b, arch, 0, &arch_reason, &target) || target == NULL)
    {
        if(out_reason != NULL)
        {
            *out_reason = (arch_reason != NULL) ? arch_reason : "unknown gfx target";
        }
        return false;
    }

    total = rocke_mega_lds_pool_bytes(allocs, num_allocs);
    if(rocke_archtarget_fits_lds(target, total))
    {
        if(out_reason != NULL)
        {
            *out_reason = "ok";
        }
        return true;
    }

    /* breakdown = ", ".join(f"{a.name}={a.nbytes}" for a in allocs) */
    breakdown[0] = '\0';
    for(i = 0; i < num_allocs && allocs != NULL && pos + 1 < sizeof(breakdown); ++i)
    {
        int wrote = snprintf(breakdown + pos,
                             sizeof(breakdown) - pos,
                             "%s%s=%ld",
                             (i == 0) ? "" : ", ",
                             (allocs[i].name != NULL) ? allocs[i].name : "None",
                             rocke_mega_lds_alloc_nbytes(&allocs[i]));
        if(wrote < 0)
        {
            break;
        }
        pos += (size_t)wrote;
        if(pos >= sizeof(breakdown))
        {
            pos = sizeof(breakdown) - 1; /* snprintf truncated; stop appending */
            break;
        }
    }

    if(out_reason != NULL)
    {
        *out_reason = (b != NULL) ? rocke_arena_printf(&b->arena,
                                                       "LDS budget %ld > %d cap (%s) on %s",
                                                       total,
                                                       target->lds_capacity_bytes,
                                                       breakdown,
                                                       (arch != NULL) ? arch : "None")
                                  : "LDS budget over cap";
    }
    return false;
}
