// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/fmha_arch.py:
 *   FMHA_MFMA_ATTN_BLOCK, validate_fmha_mfma_atom.
 * See rocke/helper_rocke.instances.common.fmha_arch.h for the symbol map.
 */
#include "rocke/helper_rocke.instances.common.fmha_arch.h"

#include <stdio.h> /* snprintf */
#include <string.h> /* strcmp   */

#include "rocke/arch_target.h" /* rocke_known_arches */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */

/* Copy a static reason into the caller buffer (NUL-terminated, truncating). */
static void rocke__set_reason(char* out, size_t out_cap, const char* reason)
{
    if(out == NULL || out_cap == 0)
    {
        return;
    }
    snprintf(out, out_cap, "%s", reason);
}

/* Reproduce str(KeyError(_build_target message)) for an unknown gfx target:
 *
 *   Python _build_target: raise KeyError(
 *     f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
 *     f"Add a row to {_DATA_FILE.name}.")
 *   validate_fmha_mfma_atom: return False, str(e)
 *
 * str(KeyError(msg)) == repr(msg); the message contains single quotes, so
 * Python's repr wraps it in DOUBLE quotes. sorted(specs) renders as a Python
 * list literal: ['gfx...', 'gfx...'] (single-quoted tokens, ", " separated).
 * known_arches() is already tuple(sorted(_load_specs())). */
static void rocke__set_unknown_arch_reason(char* out, size_t out_cap, const char* gfx)
{
    int count = 0;
    const char* const* arches;
    int i;
    size_t pos = 0;
    int wrote;

    if(out == NULL || out_cap == 0)
    {
        return;
    }

    arches = rocke_known_arches(&count);

    /* Leading double quote + the f-string up to "known: [" */
    wrote = snprintf(out + pos, out_cap - pos, "\"unknown gfx target '%s'; known: [", gfx);
    if(wrote < 0)
    {
        out[0] = '\0';
        return;
    }
    pos += (size_t)wrote;
    if(pos >= out_cap)
    {
        out[out_cap - 1] = '\0';
        return;
    }

    /* sorted(specs) list body: 'tok', 'tok', ... */
    for(i = 0; i < count; ++i)
    {
        wrote = snprintf(out + pos, out_cap - pos, "%s'%s'", (i == 0) ? "" : ", ", arches[i]);
        if(wrote < 0)
        {
            out[out_cap - 1] = '\0';
            return;
        }
        pos += (size_t)wrote;
        if(pos >= out_cap)
        {
            out[out_cap - 1] = '\0';
            return;
        }
    }

    /* Closing "]. Add a row to arch_specs.json." + trailing double quote. */
    snprintf(out + pos, out_cap - pos, "]. Add a row to arch_specs.json.\"");
}

bool rocke_validate_fmha_mfma_atom(const char* dtype, const char* arch, char* out, size_t out_cap)
{
    const char* a_name;
    const rocke_archtarget_t* target;
    int blk = ROCKE_FMHA_MFMA_ATTN_BLOCK;

    /* Python default: arch="gfx950". */
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: return
     * False, str(e).  rocke_archtarget_from_gfx returns NULL for an unknown
     * target (the Python KeyError path). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke__set_unknown_arch_reason(out, out_cap, arch);
        return false;
    }

    /* Map the FMHA activation dtype string to the catalog dtype name. */
    if(dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
    {
        a_name = "f16";
    }
    else if(dtype != NULL && strcmp(dtype, "bf16") == 0)
    {
        a_name = "bf16";
    }
    else
    {
        /* return False, f"unsupported FMHA dtype {dtype!r} for MFMA atom
         * selection".  {dtype!r} renders single-quoted. */
        if(out != NULL && out_cap != 0)
        {
            snprintf(out,
                     out_cap,
                     "unsupported FMHA dtype '%s' for MFMA atom selection",
                     dtype != NULL ? dtype : "");
        }
        return false;
    }

    /* if not target.mma.has_shape(a_dtype=a, b_dtype=a, c_dtype="fp32",
     *                             m=blk, n=blk, k=blk): ...
     *
     * has_shape(m,n,k) is true iff a matching op exists; since enumerate()
     * filters on m,n and has_shape compares the full (m,n,k) shape, this is
     * exactly op_for_shape(...,k) != None. family defaults to "mma". */
    if(rocke_archtarget_op_for_shape(target, "mma", a_name, a_name, "fp32", blk, blk, blk) == NULL)
    {
        if(out != NULL && out_cap != 0)
        {
            snprintf(out,
                     out_cap,
                     "FMHA MFMA atom %s %dx%dx%d not in %s catalog",
                     a_name,
                     blk,
                     blk,
                     blk,
                     arch);
        }
        return false;
    }

    rocke__set_reason(out, out_cap, "ok");
    return true;
}
