// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the tiled attention arch gate in
 * library/kernels/common/attention_arch.py. See
 * rocke/helper_rocke.instances.common.attention_arch.h for the symbol map.
 */
#include "rocke/helper_rocke.instances.common.attention_arch.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"

static void rocke_tattn_set_reason(char* out, size_t out_cap, const char* reason)
{
    if(out == NULL || out_cap == 0)
        return;
    snprintf(out, out_cap, "%s", reason);
}

/* The catalog lists the 32x32x16 bf16 atom as absent even on gfx950, so a target
 * advertising the transpose-read family is taken to have the wide-K atoms. */
static bool rocke_tattn_wide_k_mfma_available(const rocke_arch_target_t* t)
{
    if(rocke_mma_catalog_has_shape(&t->mma, "mma", "f16", "f16", "fp32", 16, 16, 32))
        return true;
    return t->memory.has_ds_read_tr;
}

static bool rocke_tattn_narrow_k_mfma_available(const rocke_arch_target_t* t)
{
    return rocke_mma_catalog_has_shape(&t->mma, "mma", "f16", "f16", "fp32", 16, 16, 16)
           && rocke_mma_catalog_has_shape(&t->mma, "mma", "bf16", "bf16", "fp32", 16, 16, 16);
}

/* Must stay in lockstep with the narrow-variant routing in the unified-attention
 * dispatcher: an admitted narrow arch without a routed builder would fall through
 * to the gfx950 builder and emit ISA it cannot run. */
static const char* const rocke_tattn_narrow_tiled_2d_arches[] = {"gfx942"};

static bool rocke_tattn_in_narrow_set(const char* arch)
{
    size_t i;
    for(i = 0; i < sizeof(rocke_tattn_narrow_tiled_2d_arches)
                       / sizeof(rocke_tattn_narrow_tiled_2d_arches[0]);
        ++i)
    {
        if(strcmp(arch, rocke_tattn_narrow_tiled_2d_arches[i]) == 0)
            return true;
    }
    return false;
}

bool rocke_validate_tiled_attention_arch(const char* arch, char* out, size_t out_cap)
{
    const rocke_arch_target_t* target = arch ? rocke_arch_target_from_gfx(arch) : NULL;
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(out, out_cap, arch ? arch : "None");
        return false;
    }
    if(rocke_tattn_wide_k_mfma_available(target) && target->memory.has_ds_read_tr)
    {
        rocke_tattn_set_reason(out, out_cap, "ok");
        return true;
    }
    if(rocke_tattn_in_narrow_set(arch) && rocke_tattn_narrow_k_mfma_available(target))
    {
        rocke_tattn_set_reason(out, out_cap, "ok");
        return true;
    }
    if(out == NULL || out_cap == 0)
        return false;
    if(!rocke_tattn_wide_k_mfma_available(target))
        snprintf(out,
                 out_cap,
                 "tiled attention requires either the wide-K MFMA atoms "
                 "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a narrow "
                 "variant arch (gfx942), the 16x16x16 f16/bf16 atom; neither path is "
                 "available on %s",
                 arch);
    else
        snprintf(out,
                 out_cap,
                 "tiled attention requires LDS transpose reads (ds_read_b64_tr_b16) "
                 "for the wide-K path, absent on %s",
                 arch);
    return false;
}

bool rocke_require_tiled_attention_arch(rocke_ir_builder_t* b, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(rocke_validate_tiled_attention_arch(arch, reason, sizeof(reason)))
        return true;
    if(b != NULL && b->status == ROCKE_OK)
    {
        b->status = ROCKE_ERR_NOTIMPL;
        snprintf(b->err, ROCKE_ERR_MSG_CAP, "%s", reason);
    }
    return false;
}
