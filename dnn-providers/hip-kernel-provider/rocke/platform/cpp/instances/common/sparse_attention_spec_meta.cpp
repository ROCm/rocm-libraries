// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sparse_attention_spec_meta.c -- the spec value-type surface of the
 * chunked C99 port of rocke/instances/common/sparse_attention.py (lines 93-253).
 *
 * SCOPE OF THIS TU (no IR):
 *   - rocke_jenga_sparse_spec_default / rocke_vsa_sparse_spec_default
 *       (the @dataclass(frozen=True) field defaults).
 *   - rocke_jenga_sparse_spec_num_{q,k}_blocks /
 *     rocke_vsa_sparse_spec_num_{q,k}_blocks
 *       (the ceil-div @property accessors).
 *   - rocke_jenga_sparse_kernel_name / rocke_vsa_sparse_kernel_name
 *       (the .kernel_name() builders calling rocke_kernel_name_join with the
 *        H/HQ/HK/dtype/Q/K/BQ/BK[/MB] parts in Python positional order).
 *   - rocke_is_valid_jenga_spec / rocke_is_valid_vsa_spec
 *       (validate_common_spec -> validate_fmha_mfma_atom -> the
 *        seqlen/block/head_size divisibility checks).
 *
 * Pure host-int + string work: NONE of these touch the IR builder, so a
 * byte-identical builder-call sequence in the build phases follows from
 * byte-identical names + accept/reject decisions here.
 *
 * The build / grid / signature / lower phases live in sibling TUs and bind to
 * the shared rocke/instance_sparse_attention_internal.h contract.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h" /* transient reason-string arena */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* ROCKE_MFMA_ATTN_BLOCK_M/K        */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join          */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* validate_common_spec     */
#include "rocke/helper_rocke.instances.common.fmha_arch.h" /* validate_fmha_mfma_atom  */
#include "rocke/instance_sparse_attention.h"
#include "rocke/instance_sparse_attention_internal.h"

/* Default dataclass names. */
#define ROCKE_JENGA_SPARSE_DEFAULT_NAME "rocke_jenga_sparse_attn"
#define ROCKE_VSA_SPARSE_DEFAULT_NAME "rocke_vsa_sparse_attn"

/* Copy a (possibly truncated) reason string into the caller's buffer. Mirrors
 * the (ok, reason) Python tuple's string half; reason text never enters the IR. */
static void sparse_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

/* ===================================================================== *
 *  JengaSparseSpec  --  value-type surface.
 * ===================================================================== */

/* JengaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * name="rocke_jenga_sparse_attn"). */
rocke_jenga_sparse_spec_t
    rocke_jenga_sparse_spec_default(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k)
{
    rocke_jenga_sparse_spec_t spec;
    spec.common = common;
    spec.seqlen_q = seqlen_q;
    spec.seqlen_k = seqlen_k;
    spec.block_q = 1;
    spec.block_k = 64;
    spec.name = ROCKE_JENGA_SPARSE_DEFAULT_NAME;
    return spec;
}

/* JengaSparseSpec.num_q_blocks: (seqlen_q + block_q - 1) // block_q. */
int rocke_jenga_sparse_spec_num_q_blocks(const rocke_jenga_sparse_spec_t* spec)
{
    return (spec->seqlen_q + spec->block_q - 1) / spec->block_q;
}

/* JengaSparseSpec.num_k_blocks: (seqlen_k + block_k - 1) // block_k. */
int rocke_jenga_sparse_spec_num_k_blocks(const rocke_jenga_sparse_spec_t* spec)
{
    return (spec->seqlen_k + spec->block_k - 1) / spec->block_k;
}

/* JengaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}"). */
rocke_status_t
    rocke_jenga_sparse_kernel_name(const rocke_jenga_sparse_spec_t* spec, char* out, size_t out_cap)
{
    const char* name;
    const char* dtype;
    const rocke_fmha_shape_t* s;
    char h[32], hq[32], hk[32], q[32], k[32], bq[32], bk[32];
    const char* parts[8];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : ROCKE_JENGA_SPARSE_DEFAULT_NAME;
    dtype = (spec->common.dtype != NULL) ? spec->common.dtype : "f16";
    s = &spec->common.shape;

    snprintf(h, sizeof(h), "H%d", s->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", s->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", s->num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->seqlen_k);
    snprintf(bq, sizeof(bq), "BQ%d", spec->block_q);
    snprintf(bk, sizeof(bk), "BK%d", spec->block_k);

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = bq;
    parts[7] = bk;

    return rocke_kernel_name_join(name, parts, 8, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  VsaSparseSpec  --  value-type surface.
 * ===================================================================== */

/* VsaSparseSpec(common, seqlen_q, seqlen_k, block_q=1, block_k=64,
 * max_blocks_per_q=32, name="rocke_vsa_sparse_attn",
 * use_wave_ballot_scatter=True). */
rocke_vsa_sparse_spec_t
    rocke_vsa_sparse_spec_default(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k)
{
    rocke_vsa_sparse_spec_t spec;
    spec.common = common;
    spec.seqlen_q = seqlen_q;
    spec.seqlen_k = seqlen_k;
    spec.block_q = 1;
    spec.block_k = 64;
    spec.max_blocks_per_q = 32;
    spec.name = ROCKE_VSA_SPARSE_DEFAULT_NAME;
    spec.use_wave_ballot_scatter = true;
    return spec;
}

/* VsaSparseSpec.num_q_blocks: (seqlen_q + block_q - 1) // block_q. */
int rocke_vsa_sparse_spec_num_q_blocks(const rocke_vsa_sparse_spec_t* spec)
{
    return (spec->seqlen_q + spec->block_q - 1) / spec->block_q;
}

/* VsaSparseSpec.num_k_blocks: (seqlen_k + block_k - 1) // block_k. */
int rocke_vsa_sparse_spec_num_k_blocks(const rocke_vsa_sparse_spec_t* spec)
{
    return (spec->seqlen_k + spec->block_k - 1) / spec->block_k;
}

/* VsaSparseSpec.kernel_name(): kernel_name_join(name, "H{hd}", "HQ{hq}",
 * "HK{hk}", dtype, "Q{sq}", "K{sk}", "BQ{bq}", "BK{bk}", "MB{mb}"). */
rocke_status_t
    rocke_vsa_sparse_kernel_name(const rocke_vsa_sparse_spec_t* spec, char* out, size_t out_cap)
{
    const char* name;
    const char* dtype;
    const rocke_fmha_shape_t* s;
    char h[32], hq[32], hk[32], q[32], k[32], bq[32], bk[32], mb[32];
    const char* parts[9];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : ROCKE_VSA_SPARSE_DEFAULT_NAME;
    dtype = (spec->common.dtype != NULL) ? spec->common.dtype : "f16";
    s = &spec->common.shape;

    snprintf(h, sizeof(h), "H%d", s->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", s->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", s->num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->seqlen_k);
    snprintf(bq, sizeof(bq), "BQ%d", spec->block_q);
    snprintf(bk, sizeof(bk), "BK%d", spec->block_k);
    snprintf(mb, sizeof(mb), "MB%d", spec->max_blocks_per_q);

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = bq;
    parts[7] = bk;
    parts[8] = mb;

    return rocke_kernel_name_join(name, parts, 9, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  Validity gates.
 * ===================================================================== */

/* Run validate_common_spec(spec.common) -> validate_fmha_mfma_atom(dtype, arch).
 * On any reject writes the reason and returns false; on accept returns true with
 * `reason` untouched (the caller writes "ok" after its own checks pass). The
 * `common` reason string lives in a transient arena (reason text never enters the
 * IR -- emission is byte-identical regardless). */
static bool sparse_validate_common_and_atom(const rocke_fmha_common_spec_t* common,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap)
{
    const char* common_reason = NULL;
    rocke_arena_t arena;
    bool ok;
    char atom_buf[256];

    /* ok, why = validate_common_spec(spec.common) */
    rocke_arena_init(&arena, 0);
    ok = rocke_fmha_validate_common_spec(&arena, common, &common_reason);
    if(!ok)
    {
        sparse_set_reason(
            reason, reason_cap, common_reason != NULL ? common_reason : "invalid common spec");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch) */
    if(!rocke_validate_fmha_mfma_atom(common->dtype, arch, atom_buf, sizeof(atom_buf)))
    {
        sparse_set_reason(reason, reason_cap, atom_buf);
        return false;
    }
    return true;
}

/* is_valid_jenga_spec(spec, arch="gfx950") -> (ok, reason). Mirrors Python
 * lines 169-213. */
bool rocke_is_valid_jenga_spec(const rocke_jenga_sparse_spec_t* spec,
                               const char* arch,
                               char* reason,
                               size_t reason_cap)
{
    char buf[256];

    if(spec == NULL)
    {
        sparse_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    if(!sparse_validate_common_and_atom(&spec->common, arch, reason, reason_cap))
    {
        return false;
    }

    /* seqlen_q <= 0 or seqlen_k <= 0 */
    if(spec->seqlen_q <= 0 || spec->seqlen_k <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_q / seqlen_k must be > 0 (got %d, %d)",
                 spec->seqlen_q,
                 spec->seqlen_k);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* block_q <= 0 or block_k <= 0 */
    if(spec->block_q <= 0 || spec->block_k <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_q / block_k must be > 0 (got %d, %d)",
                 spec->block_q,
                 spec->block_k);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_k % block_k != 0 */
    if(spec->seqlen_k % spec->block_k != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_k (%d) must be divisible by block_k (%d)",
                 spec->seqlen_k,
                 spec->block_k);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_q % MFMA_ATTN_BLOCK_M != 0 */
    if(spec->seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA jenga sparse needs seqlen_q (%d) divisible by BLOCK_M (%d)",
                 spec->seqlen_q,
                 ROCKE_MFMA_ATTN_BLOCK_M);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_k % MFMA_ATTN_BLOCK_K != 0 */
    if(spec->seqlen_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA jenga sparse needs seqlen_k (%d) divisible by BLOCK_K (%d)",
                 spec->seqlen_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* block_k % MFMA_ATTN_BLOCK_K != 0 */
    if(spec->block_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "sparsity block_k (%d) must be a multiple of MFMA BLOCK_K (%d) so "
                 "each sparsity block covers a whole number of MFMA K-tiles",
                 spec->block_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* spec.common.shape.head_size % 16 != 0 */
    if(spec->common.shape.head_size % 16 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA jenga sparse needs head_size %% 16 == 0 (got %d)",
                 spec->common.shape.head_size);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }

    sparse_set_reason(reason, reason_cap, "ok");
    return true;
}

/* is_valid_vsa_spec(spec, arch="gfx950") -> (ok, reason). Mirrors Python
 * lines 216-253. */
bool rocke_is_valid_vsa_spec(const rocke_vsa_sparse_spec_t* spec,
                             const char* arch,
                             char* reason,
                             size_t reason_cap)
{
    char buf[256];

    if(spec == NULL)
    {
        sparse_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    if(!sparse_validate_common_and_atom(&spec->common, arch, reason, reason_cap))
    {
        return false;
    }

    /* seqlen_q <= 0 or seqlen_k <= 0 */
    if(spec->seqlen_q <= 0 || spec->seqlen_k <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_q / seqlen_k must be > 0 (got %d, %d)",
                 spec->seqlen_q,
                 spec->seqlen_k);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* max_blocks_per_q <= 0 */
    if(spec->max_blocks_per_q <= 0)
    {
        snprintf(buf, sizeof(buf), "max_blocks_per_q must be > 0 (got %d)", spec->max_blocks_per_q);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_k % block_k != 0 */
    if(spec->seqlen_k % spec->block_k != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "seqlen_k (%d) must be divisible by block_k (%d)",
                 spec->seqlen_k,
                 spec->block_k);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_q % MFMA_ATTN_BLOCK_M != 0 */
    if(spec->seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA VSA needs seqlen_q (%d) divisible by BLOCK_M (%d)",
                 spec->seqlen_q,
                 ROCKE_MFMA_ATTN_BLOCK_M);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* seqlen_k % MFMA_ATTN_BLOCK_K != 0 */
    if(spec->seqlen_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA VSA needs seqlen_k (%d) divisible by BLOCK_K (%d)",
                 spec->seqlen_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* block_k % MFMA_ATTN_BLOCK_K != 0 */
    if(spec->block_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "VSA block_k (%d) must be a multiple of MFMA BLOCK_K (%d)",
                 spec->block_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* spec.common.shape.head_size % 16 != 0 */
    if(spec->common.shape.head_size % 16 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA VSA needs head_size %% 16 == 0 (got %d)",
                 spec->common.shape.head_size);
        sparse_set_reason(reason, reason_cap, buf);
        return false;
    }

    sparse_set_reason(reason, reason_cap, "ok");
    return true;
}
