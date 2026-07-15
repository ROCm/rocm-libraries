// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * arch_target_arch_target_query.c -- bucket 1 of the C99 port of
 * rocke.core.arch.target.
 *
 * Implements the query surface: MmaOp getters, MmaCatalog enumeration/lookup,
 * ArchTarget predicates/getters, and the from_gfx / known_arches / arch_from_isa
 * module functions. The frozen SSOT tables + shared helpers live in bucket 0
 * (arch_target_data.c) and are referenced via rocke/arch_target_internal.h.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/arch_target_internal.h"
#include "rocke/error.hpp"
#include "rocke/ir.h"

/* ---------------------------------------------------------- local helpers */

/* Raise a genuine query failure as a ckc::Error (mirroring the Python `raise`);
 * the public entry boundary catches it and records status + message on the
 * builder, so the extern "C" ABI is unchanged. This is used ONLY for true
 * errors (the "no verified layout map" NotImplementedError-equivalent). The
 * legitimate "not found" query results in this file return NULL with no error
 * and never route through here. [[noreturn]] keeps the existing
 * `rocke_ati_q_set_err(...); return NULL;` call site valid -- the return is simply
 * never reached. */
[[noreturn]] static void
    rocke_ati_q_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

/* family argument default ("mma"), matching the Python keyword default. */
static const char* rocke_ati_family_or_default(const char* family)
{
    return family ? family : "mma";
}

/* ============================== MMA atom getters ====================== */

void rocke_mma_op_shape(const rocke_mma_op_t* op, int* m, int* n, int* k)
{
    if(!op)
    {
        if(m)
            *m = 0;
        if(n)
            *n = 0;
        if(k)
            *k = 0;
        return;
    }
    if(m)
        *m = op->m;
    if(n)
        *n = op->n;
    if(k)
        *k = op->k;
}

/* MmaOp._require_layout: returns the map, or NULL + (on a non-NULL builder)
 * the NotImplementedError-equivalent sticky error with byte-identical text. */
static const rocke_layout_map_t* rocke_ati_require_layout(const rocke_mma_op_t* op,
                                                          const rocke_layout_map_t* layout,
                                                          const char* role,
                                                          rocke_ir_builder_t* b)
{
    if(!op)
        return NULL;
    if(layout != NULL)
        return layout;
    rocke_ati_q_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "no verified '%s' layout map for MMA op_id '%s' "
                        "(%dx%dx%d); add one to _MMA_FRAGMENT_INFO before "
                        "consuming it",
                        role,
                        op->op_id ? op->op_id : "",
                        op->m,
                        op->n,
                        op->k);
    return NULL;
}

const rocke_layout_map_t* rocke_mma_op_a_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_ati_require_layout(op, op ? op->a_layout : NULL, "a", b);
}

const rocke_layout_map_t* rocke_mma_op_b_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_ati_require_layout(op, op ? op->b_layout : NULL, "b", b);
}

const rocke_layout_map_t* rocke_mma_op_c_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_ati_require_layout(op, op ? op->c_layout : NULL, "c", b);
}

const rocke_layout_map_t* rocke_mma_op_acc_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    /* Python alias: acc_layout() == c_layout(). */
    return rocke_mma_op_c_layout(op, b);
}

/* ============================== MMA catalog =========================== */

const rocke_mma_op_t* rocke_mma_catalog_ops(const rocke_mma_catalog_t* cat, int* num_out)
{
    if(!cat)
    {
        if(num_out)
            *num_out = 0;
        return NULL;
    }
    if(num_out)
        *num_out = cat->num_ops;
    return cat->ops;
}

/* Shared predicate matching MmaCatalog.enumerate's per-op filter. dtypes here
 * are already-normalised canonical keys (a/b/c). family is non-NULL. m/n < 0
 * mean "any" (Python None). */
static bool rocke_ati_op_matches(const rocke_mma_op_t* op,
                                 const char* family,
                                 const char* a,
                                 const char* b,
                                 const char* c,
                                 int m,
                                 int n)
{
    if(strcmp(op->family, family) != 0)
        return false;
    if(strcmp(op->a_dtype, a) != 0)
        return false;
    if(strcmp(op->b_dtype, b) != 0)
        return false;
    if(strcmp(op->c_dtype, c) != 0)
        return false;
    if(m >= 0 && op->m != m)
        return false;
    if(n >= 0 && op->n != n)
        return false;
    return true;
}

int rocke_mma_catalog_enumerate(const rocke_mma_catalog_t* cat,
                                const char* family,
                                const char* a_dtype,
                                const char* b_dtype,
                                const char* c_dtype,
                                int m,
                                int n,
                                const rocke_mma_op_t** out,
                                int cap)
{
    char abuf[64], bbuf[64], cbuf[64];
    const char *a, *bd, *c, *fam;
    int i, total = 0;

    if(!cat)
        return 0;
    fam = rocke_ati_family_or_default(family);
    a = rocke_ati_normalize_dtype(a_dtype, abuf, sizeof abuf);
    bd = rocke_ati_normalize_dtype(b_dtype, bbuf, sizeof bbuf);
    c = rocke_ati_normalize_dtype(c_dtype, cbuf, sizeof cbuf);

    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, m, n))
            continue;
        if(out && total < cap)
            out[total] = op;
        ++total;
    }
    return total;
}

bool rocke_mma_catalog_has_shape(const rocke_mma_catalog_t* cat,
                                 const char* family,
                                 const char* a_dtype,
                                 const char* b_dtype,
                                 const char* c_dtype,
                                 int m,
                                 int n,
                                 int k)
{
    char abuf[64], bbuf[64], cbuf[64];
    const char *a, *bd, *c, *fam;
    int i;

    if(!cat)
        return false;
    fam = rocke_ati_family_or_default(family);
    a = rocke_ati_normalize_dtype(a_dtype, abuf, sizeof abuf);
    bd = rocke_ati_normalize_dtype(b_dtype, bbuf, sizeof bbuf);
    c = rocke_ati_normalize_dtype(c_dtype, cbuf, sizeof cbuf);

    /* Python enumerates with m=m, n=n then checks op.shape == (m, n, k). */
    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, m, n))
            continue;
        if(op->m == m && op->n == n && op->k == k)
            return true;
    }
    return false;
}

const rocke_mma_op_t* rocke_mma_catalog_select_largest_k(const rocke_mma_catalog_t* cat,
                                                         const char* family,
                                                         const char* a_dtype,
                                                         const char* b_dtype,
                                                         const char* c_dtype,
                                                         int m,
                                                         int n,
                                                         int k_max)
{
    char abuf[64], bbuf[64], cbuf[64];
    const char *a, *bd, *c, *fam;
    const rocke_mma_op_t* best = NULL;
    int i;

    if(!cat)
        return NULL;
    fam = rocke_ati_family_or_default(family);
    a = rocke_ati_normalize_dtype(a_dtype, abuf, sizeof abuf);
    bd = rocke_ati_normalize_dtype(b_dtype, bbuf, sizeof bbuf);
    c = rocke_ati_normalize_dtype(c_dtype, cbuf, sizeof cbuf);

    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, m, n))
            continue;
        if(k_max >= 0 && op->k > k_max)
            continue; /* Python: k_max is None || op.k <= k_max */
        /* max(cands, key=op.k): first op wins ties (Python max keeps the first
         * maximal element when iterating in catalog order). */
        if(best == NULL || op->k > best->k)
            best = op;
    }
    return best;
}

const rocke_mma_op_t* rocke_mma_catalog_by_op_id(const rocke_mma_catalog_t* cat, const char* op_id)
{
    int i;
    if(!cat || !op_id)
        return NULL;
    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(op->op_id && strcmp(op->op_id, op_id) == 0)
            return op;
    }
    return NULL;
}

const rocke_mma_op_t* rocke_mma_catalog_op_for_shape(const rocke_mma_catalog_t* cat,
                                                     const char* family,
                                                     const char* a_dtype,
                                                     const char* b_dtype,
                                                     const char* c_dtype,
                                                     int m,
                                                     int n,
                                                     int k)
{
    char abuf[64], bbuf[64], cbuf[64];
    const char *a, *bd, *c, *fam;
    int i;

    if(!cat)
        return NULL;
    fam = rocke_ati_family_or_default(family);
    a = rocke_ati_normalize_dtype(a_dtype, abuf, sizeof abuf);
    bd = rocke_ati_normalize_dtype(b_dtype, bbuf, sizeof bbuf);
    c = rocke_ati_normalize_dtype(c_dtype, cbuf, sizeof cbuf);

    /* Python enumerates with m=m, n=n, then returns the first op whose k == k. */
    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, m, n))
            continue;
        if(op->k == k)
            return op;
    }
    return NULL;
}

/* ============================== arch target =========================== */

const rocke_arch_target_t* rocke_arch_target_from_gfx(const char* gfx)
{
    int i;
    if(!gfx)
        return NULL;
    /* Mirrors ArchTarget.from_gfx -> _build_target (lru_cache singletons): the
     * registry rows already hold fully-built descriptors. */
    for(i = 0; i < rocke_ati_arch_registry_len; ++i)
    {
        const rocke_ati_arch_row_t* row = &rocke_ati_arch_registry[i];
        if(row->gfx && strcmp(row->gfx, gfx) == 0)
            return row->target;
    }
    return NULL; /* Python raises KeyError; here NULL is the failure sentinel. */
}

const char* rocke_arch_isa_triple(const rocke_arch_target_t* t, char* out, size_t out_cap)
{
    int n;
    if(!t || !out || out_cap == 0)
        return NULL;
    n = snprintf(out, out_cap, "amdgcn-amd-amdhsa--%s", t->gfx ? t->gfx : "");
    if(n < 0 || (size_t)n >= out_cap)
        return NULL; /* truncated / too small */
    return out;
}

bool rocke_arch_fits_lds(const rocke_arch_target_t* t, long bytes_in_use)
{
    if(!t)
        return false;
    return bytes_in_use <= (long)t->lds_capacity_bytes;
}

bool rocke_arch_supports_dtype_combo(
    const rocke_arch_target_t* t, const char* a, const char* b, const char* c, const char* family)
{
    if(!t)
        return false;
    /* Python: len(enumerate(...)) > 0 (m/n omitted => None => "any"). */
    return rocke_mma_catalog_enumerate(&t->mma, family, a, b, c, -1, -1, NULL, 0) > 0;
}

int rocke_arch_max_vector_load_dwords(const rocke_arch_target_t* t, const char* dtype)
{
    (void)dtype; /* gated by the buffer-load path, not the element type today. */
    if(!t)
        return 0;
    return t->memory.buffer_load_max_dwords;
}

int rocke_arch_max_threads_per_block(const rocke_arch_target_t* t)
{
    if(!t)
        return 0;
    return t->limits.max_threads_per_block;
}

/* ============================== module fns ============================ */

const char* const* rocke_known_arches(int* count)
{
    /* Python: tuple(sorted(_load_specs())). Bucket 0 keeps rocke_ati_known_arches
     * sorted + NULL-terminated alongside the registry. */
    if(count)
        *count = rocke_ati_arch_registry_len;
    return rocke_ati_known_arches;
}

const char* rocke_arch_from_isa(const char* isa, char* out, size_t out_cap)
{
    const char* dash;
    const char* tok;
    size_t len;

    if(!isa || !out || out_cap == 0)
        return NULL;
    /* Python: isa.rsplit("-", 1)[-1] if "-" in isa else isa.
     * rsplit on the last '-' yields everything after the final '-'. */
    dash = strrchr(isa, '-');
    tok = dash ? dash + 1 : isa;
    len = strlen(tok);
    if(len + 1 > out_cap)
    {
        /* Truncate to fit (NUL-terminated); mirrors a best-effort copy. */
        len = out_cap - 1;
    }
    memcpy(out, tok, len);
    out[len] = '\0';
    return out;
}
