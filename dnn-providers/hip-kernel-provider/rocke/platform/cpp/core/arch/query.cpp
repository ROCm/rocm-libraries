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
#include <string>

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

const char* rocke_mma_op_semantic_id(const rocke_mma_op_t* op, char* out, size_t cap)
{
    if(!op || !out || !cap)
        return NULL;
    std::string id = op->family;
    for(char& c : id)
        if(c == '_')
            c = '.';
    id += "." + std::to_string(op->m) + "x" + std::to_string(op->n) + "x" + std::to_string(op->k);
    char dtype_buf[64];
    for(int i = 0; i < 3; ++i)
    {
        const rocke_mma_src_t& src = op->srcs[i];
        id += ".src" + std::to_string(i) + "_"
              + rocke_normalize_dtype(src.dtype, dtype_buf, sizeof(dtype_buf));
        if(src.scale_dtype)
        {
            const char* scale = strcmp(src.scale_dtype, "fp8e4m3") == 0 ? "e4m3" : src.scale_dtype;
            id += std::string("_") + scale + "_b" + std::to_string(src.scale_block_size);
        }
    }
    id += std::string(".dst_") + rocke_normalize_dtype(op->dst.dtype, dtype_buf, sizeof(dtype_buf));
    if(id.size() >= cap)
        return NULL;
    memcpy(out, id.c_str(), id.size() + 1);
    return out;
}

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
    return rocke_mma_op_src_layout(op, 0, b);
}

const rocke_layout_map_t* rocke_mma_op_b_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_mma_op_src_layout(op, 1, b);
}

const rocke_layout_map_t* rocke_mma_op_c_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_mma_op_dst_layout(op, b);
}

const rocke_layout_map_t*
    rocke_mma_op_src_layout(const rocke_mma_op_t* op, int index, rocke_ir_builder_t* b)
{
    if(index < 0 || index >= 3)
    {
        rocke_ati_q_set_err(b, ROCKE_ERR_VALUE, "MMA src index %d is outside [0, 3)", index);
    }
    char role[8];
    (void)snprintf(role, sizeof(role), "src%d", index);
    return rocke_ati_require_layout(op, op ? op->srcs[index].layout : NULL, role, b);
}

const rocke_layout_map_t* rocke_mma_op_dst_layout(const rocke_mma_op_t* op, rocke_ir_builder_t* b)
{
    return rocke_ati_require_layout(op, op ? op->dst.layout : NULL, "dst", b);
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
static void rocke_ati_validate_scales(const rocke_mma_scale_operand_t* scales)
{
    if(!scales)
        return;
    for(int i = 0; i < 3; ++i)
    {
        const char* dtype = scales[i].dtype;
        if(dtype && strcmp(dtype, "fp8e4m3") == 0)
            dtype = "e4m3";
        if(dtype && strcmp(dtype, "e8m0") != 0 && strcmp(dtype, "e4m3") != 0
           && strcmp(dtype, "e5m3") != 0)
            ckc::raise_status(ROCKE_ERR_VALUE, "MMA scale dtype must be e8m0, e4m3, or e5m3");
        if((dtype && scales[i].block_size != 16 && scales[i].block_size != 32)
           || (!dtype && scales[i].block_size != 0))
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid MMA scale block size");
    }
}

static bool rocke_ati_scales_match(const rocke_mma_op_t* op,
                                   const rocke_mma_scale_operand_t* scales)
{
    if(!scales)
        return true;
    for(int i = 0; i < 3; ++i)
    {
        const char* dtype = scales[i].dtype;
        if(dtype && strcmp(dtype, "fp8e4m3") == 0)
            dtype = "e4m3";
        const char* actual = op->srcs[i].scale_dtype;
        if(bool(dtype) != bool(actual) || (dtype && strcmp(dtype, actual) != 0)
           || scales[i].block_size != op->srcs[i].scale_block_size)
            return false;
    }
    return true;
}

static bool rocke_ati_op_matches(const rocke_mma_op_t* op,
                                 const char* family,
                                 const char* a,
                                 const char* b,
                                 const char* c,
                                 const char* d,
                                 int m,
                                 int n)
{
    if(strcmp(op->family, family) != 0)
        return false;
    if(strcmp(op->srcs[0].dtype, a) != 0)
        return false;
    if(strcmp(op->srcs[1].dtype, b) != 0)
        return false;
    if(strcmp(op->srcs[2].dtype, c) != 0)
        return false;
    if(strcmp(op->dst.dtype, d) != 0)
        return false;
    if(m >= 0 && op->m != m)
        return false;
    if(n >= 0 && op->n != n)
        return false;
    return true;
}

int rocke_mma_catalog_enumerate_indexed(const rocke_mma_catalog_t* cat,
                                        const char* family,
                                        const char* const src_dtypes[3],
                                        const char* dst_dtype,
                                        const rocke_mma_scale_operand_t src_scales[3],
                                        int m,
                                        int n,
                                        const rocke_mma_op_t** out,
                                        int cap)
{
    char abuf[64], bbuf[64], cbuf[64], dbuf[64];
    const char *a, *bd, *c, *d, *fam;
    int i, total = 0;

    if(!cat || !src_dtypes)
        return 0;
    rocke_ati_validate_scales(src_scales);
    fam = rocke_ati_family_or_default(family);
    a = rocke_normalize_dtype(src_dtypes[0], abuf, sizeof abuf);
    bd = rocke_normalize_dtype(src_dtypes[1], bbuf, sizeof bbuf);
    c = rocke_normalize_dtype(src_dtypes[2], cbuf, sizeof cbuf);
    d = rocke_normalize_dtype(dst_dtype, dbuf, sizeof dbuf);

    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, d, m, n)
           || !rocke_ati_scales_match(op, src_scales))
            continue;
        if(out && total < cap)
            out[total] = op;
        ++total;
    }
    return total;
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
    const char* src_dtypes[3] = {a_dtype, b_dtype, c_dtype};
    return rocke_mma_catalog_enumerate_indexed(
        cat, family, src_dtypes, c_dtype, NULL, m, n, out, cap);
}

bool rocke_mma_catalog_has_shape_indexed(const rocke_mma_catalog_t* cat,
                                         const char* family,
                                         const char* const src_dtypes[3],
                                         const char* dst_dtype,
                                         const rocke_mma_scale_operand_t src_scales[3],
                                         int m,
                                         int n,
                                         int k)
{
    char abuf[64], bbuf[64], cbuf[64], dbuf[64];
    const char *a, *bd, *c, *d, *fam;
    int i;

    if(!cat || !src_dtypes)
        return false;
    rocke_ati_validate_scales(src_scales);
    fam = rocke_ati_family_or_default(family);
    a = rocke_normalize_dtype(src_dtypes[0], abuf, sizeof abuf);
    bd = rocke_normalize_dtype(src_dtypes[1], bbuf, sizeof bbuf);
    c = rocke_normalize_dtype(src_dtypes[2], cbuf, sizeof cbuf);
    d = rocke_normalize_dtype(dst_dtype, dbuf, sizeof dbuf);

    /* Python enumerates with m=m, n=n then checks op.shape == (m, n, k). */
    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, d, m, n)
           || !rocke_ati_scales_match(op, src_scales))
            continue;
        if(op->m == m && op->n == n && op->k == k)
            return true;
    }
    return false;
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
    const char* src_dtypes[3] = {a_dtype, b_dtype, c_dtype};
    return rocke_mma_catalog_has_shape_indexed(cat, family, src_dtypes, c_dtype, NULL, m, n, k);
}

const rocke_mma_op_t*
    rocke_mma_catalog_select_largest_k_indexed(const rocke_mma_catalog_t* cat,
                                               const char* family,
                                               const char* const src_dtypes[3],
                                               const char* dst_dtype,
                                               const rocke_mma_scale_operand_t src_scales[3],
                                               int m,
                                               int n,
                                               int k_max)
{
    char abuf[64], bbuf[64], cbuf[64], dbuf[64];
    const char *a, *bd, *c, *d, *fam;
    const rocke_mma_op_t* best = NULL;
    bool ambiguous = false;
    int i;

    if(!cat || !src_dtypes)
        return NULL;
    rocke_ati_validate_scales(src_scales);
    fam = rocke_ati_family_or_default(family);
    a = rocke_normalize_dtype(src_dtypes[0], abuf, sizeof abuf);
    bd = rocke_normalize_dtype(src_dtypes[1], bbuf, sizeof bbuf);
    c = rocke_normalize_dtype(src_dtypes[2], cbuf, sizeof cbuf);
    d = rocke_normalize_dtype(dst_dtype, dbuf, sizeof dbuf);

    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, d, m, n)
           || !rocke_ati_scales_match(op, src_scales))
            continue;
        if(k_max >= 0 && op->k > k_max)
            continue; /* Python: k_max is None || op.k <= k_max */
        if(best == NULL || op->k > best->k)
        {
            best = op;
            ambiguous = false;
        }
        else if(op->k == best->k)
            ambiguous = true;
    }
    if(ambiguous)
        ckc::raise_status(ROCKE_ERR_VALUE,
                          "ambiguous MMA query; specify the full operand contract");
    return best;
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
    const char* src_dtypes[3] = {a_dtype, b_dtype, c_dtype};
    return rocke_mma_catalog_select_largest_k_indexed(
        cat, family, src_dtypes, c_dtype, NULL, m, n, k_max);
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

const rocke_mma_op_t*
    rocke_mma_catalog_op_for_shape_indexed(const rocke_mma_catalog_t* cat,
                                           const char* family,
                                           const char* const src_dtypes[3],
                                           const char* dst_dtype,
                                           const rocke_mma_scale_operand_t src_scales[3],
                                           int m,
                                           int n,
                                           int k)
{
    char abuf[64], bbuf[64], cbuf[64], dbuf[64];
    const char *a, *bd, *c, *d, *fam;
    int i;
    const rocke_mma_op_t* match = NULL;

    if(!cat || !src_dtypes)
        return NULL;
    rocke_ati_validate_scales(src_scales);
    fam = rocke_ati_family_or_default(family);
    a = rocke_normalize_dtype(src_dtypes[0], abuf, sizeof abuf);
    bd = rocke_normalize_dtype(src_dtypes[1], bbuf, sizeof bbuf);
    c = rocke_normalize_dtype(src_dtypes[2], cbuf, sizeof cbuf);
    d = rocke_normalize_dtype(dst_dtype, dbuf, sizeof dbuf);

    /* Exact selection rejects an underspecified contract. */
    for(i = 0; i < cat->num_ops; ++i)
    {
        const rocke_mma_op_t* op = &cat->ops[i];
        if(!rocke_ati_op_matches(op, fam, a, bd, c, d, m, n)
           || !rocke_ati_scales_match(op, src_scales))
            continue;
        if(op->k == k)
        {
            if(match)
                ckc::raise_status(ROCKE_ERR_VALUE,
                                  "ambiguous MMA query; specify the full operand contract");
            match = op;
        }
    }
    return match;
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
    const char* src_dtypes[3] = {a_dtype, b_dtype, c_dtype};
    return rocke_mma_catalog_op_for_shape_indexed(cat, family, src_dtypes, c_dtype, NULL, m, n, k);
}

/* ===================== bare-op_id SSOT lookups ======================== */

const char* rocke_arch_mma_op_id_dst_dtype(const char* op_id)
{
    int i;
    if(!op_id)
    {
        return NULL;
    }
    /* Mirrors target._op_id_dst_dtype()[op_id]: the dst dtype names a
     * specific atom, so it is invariant across the arches that list op_id --
     * the first catalog hit wins. The catalog dst.dtype is already the normalised
     * (canonical) key, matching normalize_dtype(o["dst"]["dtype"]). Op_ids absent from
     * every catalog return NULL. */
    for(i = 0; i < rocke_ati_arch_registry_len; ++i)
    {
        const rocke_arch_target_t* t = rocke_ati_arch_registry[i].target;
        const rocke_mma_op_t* op = t ? rocke_mma_catalog_by_op_id(&t->mma, op_id) : NULL;
        if(op)
        {
            return op->dst.dtype;
        }
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

bool rocke_arch_supports_dtype_combo_indexed(const rocke_arch_target_t* t,
                                             const char* const src_dtypes[3],
                                             const char* dst_dtype,
                                             const rocke_mma_scale_operand_t src_scales[3],
                                             const char* family)
{
    if(!t)
        return false;
    /* Python: len(enumerate(...)) > 0 (m/n omitted => None => "any"). */
    return rocke_mma_catalog_enumerate_indexed(
               &t->mma, family, src_dtypes, dst_dtype, src_scales, -1, -1, NULL, 0)
           > 0;
}

bool rocke_arch_supports_dtype_combo(
    const rocke_arch_target_t* t, const char* a, const char* b, const char* c, const char* family)
{
    const char* src_dtypes[3] = {a, b, c};
    return rocke_arch_supports_dtype_combo_indexed(t, src_dtypes, c, NULL, family);
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

/* Return the final gfx component without discarding profile or feature suffixes. */
static const char* rocke_target_id_start(const char* isa)
{
    const char* target = isa;
    const char* next = isa;
    while((next = strstr(next, "gfx")) != NULL)
    {
        target = next;
        next += 3;
    }
    return target;
}

/* Length of the base architecture in target_id; consult the shared catalog so
 * names such as gfx11-generic are preserved before removing profile suffixes. */
static size_t rocke_base_arch_length(const char* target_id)
{
    size_t name_len = strcspn(target_id, ":");
    size_t best = 0;
    int i;
    for(i = 0; i < rocke_ati_arch_registry_len; ++i)
    {
        const char* arch = rocke_ati_known_arches[i];
        size_t len = strlen(arch);
        if(len <= name_len && strncmp(target_id, arch, len) == 0)
        {
            if(len == name_len)
                return len;
            if(target_id[len] == '-' && len > best)
                best = len;
        }
    }
    if(best != 0)
        return best;
    /* Python's fallback is ^(gfx[0-9a-z]+), with ASCII character semantics. */
    if(name_len > 3 && strncmp(target_id, "gfx", 3) == 0)
    {
        size_t len = 3;
        while(len < name_len
              && ((target_id[len] >= '0' && target_id[len] <= '9')
                  || (target_id[len] >= 'a' && target_id[len] <= 'z')))
            ++len;
        if(len > 3)
            return len;
    }
    return name_len;
}

static const char* rocke_copy_target(const char* target, size_t len, char* out, size_t out_cap)
{
    if(len >= out_cap)
        len = out_cap - 1;
    memcpy(out, target, len);
    out[len] = '\0';
    return out;
}

const char* rocke_target_id_from_isa(const char* isa, char* out, size_t out_cap)
{
    if(!isa || !out || out_cap == 0)
        return NULL;
    const char* target = rocke_target_id_start(isa);
    return rocke_copy_target(target, strlen(target), out, out_cap);
}

const char* rocke_base_arch_from_target_id(const char* target_id, char* out, size_t out_cap)
{
    if(!target_id || !out || out_cap == 0)
        return NULL;
    return rocke_copy_target(target_id, rocke_base_arch_length(target_id), out, out_cap);
}

const char* rocke_compiler_target_from_target_id(const char* target_id, char* out, size_t out_cap)
{
    if(!target_id || !out || out_cap == 0)
        return NULL;
    size_t name_len = strcspn(target_id, ":");
    const char* features = target_id + name_len;
    size_t base_len = rocke_base_arch_length(target_id);
    if(base_len < name_len && target_id[base_len] == '-')
        name_len = base_len;
    if(name_len >= out_cap)
        return rocke_copy_target(target_id, name_len, out, out_cap);
    memcpy(out, target_id, name_len);
    rocke_copy_target(features, strlen(features), out + name_len, out_cap - name_len);
    return out;
}

const char* rocke_arch_from_isa(const char* isa, char* out, size_t out_cap)
{
    if(!isa || !out || out_cap == 0)
        return NULL;
    return rocke_base_arch_from_target_id(rocke_target_id_start(isa), out, out_cap);
}
