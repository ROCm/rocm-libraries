// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.distribution.c -- C99 port of make_static_tile_distribution
 * (and the supporting TileDistributionEncoding constructor + validation) from
 * rocke/helpers/distribution.py.
 *
 * make_static_tile_distribution emits NO IR (it is a pure host-side analysis
 * pass), so there are zero rocke_b_* calls and the byte-identical-op-sequence
 * requirement is the empty sequence. The fidelity that matters is the produced
 * contributor table, which is reproduced bucket-for-bucket from the Python:
 *
 *   contributors: List[List[Optional[_HBucketRef]]] = [
 *       [None for _ in hs] for hs in encoding.Hs
 *   ]
 *   # Ps cover their entries first; Ys fill any remaining holes.
 *   for pi, (maj_seq, min_seq) in enumerate(zip(Ps2RHs_major, Ps2RHs_minor)):
 *       for inner_idx, (maj, minor) in enumerate(zip(maj_seq, min_seq)):
 *           if maj == 0: continue            # R contributors don't enter X
 *           contributors[maj-1][minor] = _HBucketRef("P", pi, inner_idx)
 *   for yi, (maj, minor) in enumerate(zip(Ys2RHs_major, Ys2RHs_minor)):
 *       if maj == 0: continue                # R contributor; not in X
 *       contributors[maj-1][minor] = _HBucketRef("Y", yi)
 *   # defensive: every H bucket must be mapped.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Set the builder's sticky error (first failure wins) and return NULL. Mirrors
 * the private rocke_i_set_err but is reproduced here so this public helper module
 * binds only to rocke/ir.h's public struct fields (status + err), not the private
 * ir_internal.h. */
/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing `return (T*)rocke_dist_set_err(...)`
 * call sites valid -- the cast/return is simply never reached. */
[[noreturn]] static void*
    rocke_dist_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
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

static bool rocke_dist_live(const rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

/* Python frozen-dataclass _prod over a small int sequence. (Currently used only
 * by the validation reproduction below; kept for parity with the Python
 * module's _prod helper that the encoding properties share.) */
static int rocke_dist_prod(const int* xs, int n)
{
    int p = 1;
    int i;
    for(i = 0; i < n; ++i)
    {
        p *= xs[i];
    }
    return p;
}

/* Length of the (major, minor) bucket. Python _bucket_length. */
static int
    rocke_dist_bucket_length(const rocke_tile_distribution_encoding_t* e, int major, int minor)
{
    if(major == 0)
    {
        return e->Rs[minor];
    }
    return e->Hs[major - 1].levels[minor];
}

/* Python TileDistributionEncoding._validate_target. Returns false + sets sticky
 * error on an out-of-range (major, minor). kind is "P"/"Y" for the message. */
static bool rocke_dist_validate_target(rocke_ir_builder_t* b,
                                       const rocke_tile_distribution_encoding_t* e,
                                       const char* kind,
                                       int idx,
                                       int major,
                                       int minor)
{
    if(major == 0)
    {
        if(minor < 0 || minor >= e->num_R)
        {
            rocke_dist_set_err(b,
                               ROCKE_ERR_VALUE,
                               "%s%d (R-major=0, minor=%d) out of range; Rs has %d levels",
                               kind,
                               idx,
                               minor,
                               e->num_R);
            return false;
        }
        return true;
    }
    if(major < 1 || major > e->num_X)
    {
        rocke_dist_set_err(b,
                           ROCKE_ERR_VALUE,
                           "%s%d major=%d out of range (0 for R, 1..%d for X)",
                           kind,
                           idx,
                           major,
                           e->num_X);
        return false;
    }
    if(minor < 0 || minor >= e->Hs[major - 1].count)
    {
        rocke_dist_set_err(b,
                           ROCKE_ERR_VALUE,
                           "%s%d (major=%d, minor=%d) out of range; H[%d] has %d levels",
                           kind,
                           idx,
                           major,
                           minor,
                           major - 1,
                           e->Hs[major - 1].count);
        return false;
    }
    return true;
}

/* --------------------------------------------------- encoding construction */

/* Copy an int array into the arena (NULL/0 -> NULL). */
static const int* rocke_dist_dup_ints(rocke_ir_builder_t* b, const int* src, int n)
{
    int* out;
    if(n <= 0 || src == NULL)
    {
        return NULL;
    }
    out = (int*)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(int));
    if(out == NULL)
    {
        rocke_dist_set_err(b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        return NULL;
    }
    memcpy(out, src, (size_t)n * sizeof(int));
    return out;
}

rocke_tile_distribution_encoding_t* rocke_make_tile_distribution_encoding(rocke_ir_builder_t* b,
                                                                          const int* Rs,
                                                                          int num_R,
                                                                          const rocke_h_row_t* Hs,
                                                                          int num_X,
                                                                          const rocke_p_seq_t* Ps,
                                                                          int num_P,
                                                                          const int* Ys_major,
                                                                          const int* Ys_minor,
                                                                          int num_Y)
{
    rocke_tile_distribution_encoding_t* e;
    rocke_h_row_t* hs_copy = NULL;
    rocke_p_seq_t* ps_copy = NULL;
    int i;
    int pi;

    if(!rocke_dist_live(b))
    {
        return NULL;
    }

    /* __post_init__ rank checks that are structural to our representation.
     * (Python keeps Ps2RHs_major/minor as two parallel tuples; here a P dim's
     * major/minor share one rocke_p_seq, so the per-P "major/minor sub-sequence
     * length mismatch" is impossible by construction. We still honour the
     * remaining __post_init__ semantics below.) */

    e = (rocke_tile_distribution_encoding_t*)rocke_arena_calloc(
        &b->arena, sizeof(rocke_tile_distribution_encoding_t));
    if(e == NULL)
    {
        return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }

    e->Rs = rocke_dist_dup_ints(b, Rs, num_R);
    e->num_R = num_R;
    e->num_X = num_X;
    e->num_P = num_P;
    e->Ys_major = rocke_dist_dup_ints(b, Ys_major, num_Y);
    e->Ys_minor = rocke_dist_dup_ints(b, Ys_minor, num_Y);
    e->num_Y = num_Y;

    if(num_X > 0)
    {
        hs_copy
            = (rocke_h_row_t*)rocke_arena_alloc(&b->arena, (size_t)num_X * sizeof(rocke_h_row_t));
        if(hs_copy == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
        for(i = 0; i < num_X; ++i)
        {
            hs_copy[i].count = Hs[i].count;
            hs_copy[i].levels = rocke_dist_dup_ints(b, Hs[i].levels, Hs[i].count);
        }
    }
    e->Hs = hs_copy;

    if(num_P > 0)
    {
        ps_copy
            = (rocke_p_seq_t*)rocke_arena_alloc(&b->arena, (size_t)num_P * sizeof(rocke_p_seq_t));
        if(ps_copy == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
        for(i = 0; i < num_P; ++i)
        {
            ps_copy[i].count = Ps[i].count;
            ps_copy[i].major = rocke_dist_dup_ints(b, Ps[i].major, Ps[i].count);
            ps_copy[i].minor = rocke_dist_dup_ints(b, Ps[i].minor, Ps[i].count);
        }
    }
    e->Ps = ps_copy;

    if(!rocke_dist_live(b))
    {
        return NULL; /* an alloc above failed */
    }

    /* -- __post_init__ validation (faithful port) ------------------------- */

    /* P targets in range. */
    for(pi = 0; pi < num_P; ++pi)
    {
        int j;
        for(j = 0; j < e->Ps[pi].count; ++j)
        {
            if(!rocke_dist_validate_target(b, e, "P", pi, e->Ps[pi].major[j], e->Ps[pi].minor[j]))
            {
                return NULL;
            }
        }
    }
    /* Y targets in range. */
    for(i = 0; i < num_Y; ++i)
    {
        if(!rocke_dist_validate_target(b, e, "Y", i, e->Ys_major[i], e->Ys_minor[i]))
        {
            return NULL;
        }
    }

    /* Every (major, minor) bucket must be referenced exactly once. The Python
     * uses a set; here the bucket space is small (R levels + sum of H levels),
     * so a flat "seen" array over a compact bucket id suffices. Bucket id:
     *   major == 0 -> minor                              (R levels)
     *   major  > 0 -> num_R + (prefix H levels) + minor  (H levels)
     * which is a dense bijection onto [0, total_buckets). */
    {
        int total_buckets = num_R;
        int x;
        int* h_base; /* base bucket id for X dim x's H levels */
        char* seen;
        int total;

        h_base = NULL;
        if(num_X > 0)
        {
            h_base = (int*)rocke_arena_alloc(&b->arena, (size_t)num_X * sizeof(int));
            if(h_base == NULL)
            {
                return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                    b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
            }
        }
        for(x = 0; x < num_X; ++x)
        {
            h_base[x] = total_buckets;
            total_buckets += e->Hs[x].count;
        }
        total = total_buckets;
        seen = (char*)rocke_arena_calloc(&b->arena, (size_t)(total > 0 ? total : 1));
        if(seen == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }

#define ROCKE_DIST_BUCKET_ID(maj, mnr) ((maj) == 0 ? (mnr) : (h_base[(maj) - 1] + (mnr)))

        for(pi = 0; pi < num_P; ++pi)
        {
            int j;
            for(j = 0; j < e->Ps[pi].count; ++j)
            {
                int id = ROCKE_DIST_BUCKET_ID(e->Ps[pi].major[j], e->Ps[pi].minor[j]);
                if(seen[id])
                {
                    return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                        b,
                        ROCKE_ERR_VALUE,
                        "bucket (%d,%d) referenced by multiple P/Y entries",
                        e->Ps[pi].major[j],
                        e->Ps[pi].minor[j]);
                }
                seen[id] = 1;
            }
        }
        for(i = 0; i < num_Y; ++i)
        {
            int id = ROCKE_DIST_BUCKET_ID(e->Ys_major[i], e->Ys_minor[i]);
            if(seen[id])
            {
                return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                    b,
                    ROCKE_ERR_VALUE,
                    "bucket (%d,%d) referenced by multiple P/Y entries",
                    e->Ys_major[i],
                    e->Ys_minor[i]);
            }
            seen[id] = 1;
        }

        /* Coverage: every H bucket must be referenced. */
        for(x = 0; x < num_X; ++x)
        {
            int level;
            for(level = 0; level < e->Hs[x].count; ++level)
            {
                if(!seen[h_base[x] + level])
                {
                    return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                        b,
                        ROCKE_ERR_VALUE,
                        "H bucket X%d level %d has no P or Y contributor",
                        x,
                        level);
                }
            }
        }
        /* Coverage: every R bucket should be referenced too. */
        for(i = 0; i < num_R; ++i)
        {
            if(!seen[i])
            {
                return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                    b, ROCKE_ERR_VALUE, "R bucket level %d has no P or Y contributor", i);
            }
        }

#undef ROCKE_DIST_BUCKET_ID
    }

    /* Silence unused-helper warnings in builds where the bucket-length/_prod
     * parity helpers are not otherwise referenced. They mirror Python
     * properties and are kept for fidelity. */
    (void)rocke_dist_bucket_length;
    (void)rocke_dist_prod;

    return e;
}

/* ------------------------------------------- make_static_tile_distribution */

rocke_tile_distribution_t*
    rocke_make_static_tile_distribution(rocke_ir_builder_t* b,
                                        const rocke_tile_distribution_encoding_t* encoding)
{
    rocke_tile_distribution_t* dist;
    rocke_h_bucket_ref_t** rows; /* [num_X] -> rocke_h_bucket_ref_t[Hs[x].count] */
    int num_X;
    int x;
    int pi;
    int yi;

    if(!rocke_dist_live(b))
    {
        return NULL;
    }
    if(encoding == NULL)
    {
        return (rocke_tile_distribution_t*)rocke_dist_set_err(
            b, ROCKE_ERR_VALUE, "make_static_tile_distribution: encoding is NULL");
    }

    num_X = encoding->num_X;

    /* contributors = [[None for _ in hs] for hs in encoding.Hs] */
    rows = NULL;
    if(num_X > 0)
    {
        rows = (rocke_h_bucket_ref_t**)rocke_arena_alloc(
            &b->arena, (size_t)num_X * sizeof(rocke_h_bucket_ref_t*));
        if(rows == NULL)
        {
            return (rocke_tile_distribution_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
    }
    for(x = 0; x < num_X; ++x)
    {
        int n = encoding->Hs[x].count;
        /* calloc => every ref starts kind == '\0' (== Python None). */
        rows[x] = (rocke_h_bucket_ref_t*)rocke_arena_calloc(
            &b->arena, (size_t)(n > 0 ? n : 1) * sizeof(rocke_h_bucket_ref_t));
        if(rows[x] == NULL)
        {
            return (rocke_tile_distribution_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
    }

    /* Ps cover their entries first; Ys fill any remaining holes. */
    for(pi = 0; pi < encoding->num_P; ++pi)
    {
        int inner;
        const rocke_p_seq_t* p = &encoding->Ps[pi];
        for(inner = 0; inner < p->count; ++inner)
        {
            int maj = p->major[inner];
            int minor = p->minor[inner];
            if(maj == 0)
            {
                continue; /* R contributors don't enter X */
            }
            x = maj - 1;
            rows[x][minor].kind = 'P';
            rows[x][minor].outer_idx = pi;
            rows[x][minor].inner_idx = inner;
        }
    }
    for(yi = 0; yi < encoding->num_Y; ++yi)
    {
        int maj = encoding->Ys_major[yi];
        int minor = encoding->Ys_minor[yi];
        if(maj == 0)
        {
            continue; /* R contributor; not in X */
        }
        x = maj - 1;
        rows[x][minor].kind = 'Y';
        rows[x][minor].outer_idx = yi;
        rows[x][minor].inner_idx = 0;
    }

    /* Defensive: every H bucket must have a contributor (Python re-checks even
     * though the encoding validation already covers it). */
    for(x = 0; x < num_X; ++x)
    {
        int level;
        for(level = 0; level < encoding->Hs[x].count; ++level)
        {
            if(rows[x][level].kind == '\0')
            {
                return (rocke_tile_distribution_t*)rocke_dist_set_err(
                    b,
                    ROCKE_ERR_VALUE,
                    "H bucket X%d level %d unmapped after assignment",
                    x,
                    level);
            }
        }
    }

    dist = (rocke_tile_distribution_t*)rocke_arena_calloc(&b->arena,
                                                          sizeof(rocke_tile_distribution_t));
    if(dist == NULL)
    {
        return (rocke_tile_distribution_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }
    dist->encoding = encoding;
    dist->contributors = (const rocke_h_bucket_ref_t* const*)rows;
    return dist;
}

/* TileDistribution._lookup_contributor: resolve one H-bucket ref to an SSA
 * value supplied by either the ys (Y kind) or the ps sub-arrays (P kind). */
static rocke_value_t* rocke_dist_lookup_contributor(const rocke_h_bucket_ref_t* ref,
                                                    rocke_value_t* const* ys,
                                                    rocke_value_t* const* const* ps)
{
    if(ref->kind == 'Y')
    {
        return ys[ref->outer_idx];
    }
    /* 'P' */
    return ps[ref->outer_idx][ref->inner_idx];
}

int rocke_tile_distribution_calculate_x(rocke_ir_builder_t* b,
                                        const rocke_tile_distribution_t* dist,
                                        rocke_value_t* const* ys,
                                        int num_ys,
                                        rocke_value_t* const* const* ps,
                                        const int* ps_counts,
                                        int num_ps,
                                        rocke_value_t** out_x,
                                        int out_x_cap)
{
    const rocke_tile_distribution_encoding_t* enc;
    int x_dim;

    (void)ps_counts;
    if(!rocke_dist_live(b) || dist == NULL || dist->encoding == NULL)
    {
        return 0;
    }
    enc = dist->encoding;
    if(num_ys != enc->num_Y || num_ps != enc->num_P || out_x_cap < enc->num_X)
    {
        /* Python raises ValueError on rank mismatch. */
        rocke_dist_set_err(b,
                           ROCKE_ERR_VALUE,
                           "calculate_x: rank mismatch (ys=%d/%d ps=%d/%d x_cap=%d/%d)",
                           num_ys,
                           enc->num_Y,
                           num_ps,
                           enc->num_P,
                           out_x_cap,
                           enc->num_X);
        return 0;
    }

    for(x_dim = 0; x_dim < enc->num_X; ++x_dim)
    {
        const rocke_h_row_t* hs = &enc->Hs[x_dim];
        /* x = b.const_i32(0); stride = 1; walk innermost (small stride) outward. */
        rocke_value_t* x = rocke_b_const_i32(b, 0);
        int stride = 1;
        int level;
        for(level = hs->count - 1; level >= 0; --level)
        {
            const rocke_h_bucket_ref_t* ref = &dist->contributors[x_dim][level];
            rocke_value_t* contributor = rocke_dist_lookup_contributor(ref, ys, ps);
            if(stride == 1)
            {
                x = rocke_b_add(b, x, contributor);
            }
            else
            {
                x = rocke_b_add(b, x, rocke_b_mul(b, contributor, rocke_b_const_i32(b, stride)));
            }
            stride *= hs->levels[level];
        }
        out_x[x_dim] = x;
    }
    return 1;
}

/* ============================================================================
 *  Reduce distribution + static distributed tensor + block_tile_reduce_sync.
 *  Ports of distribution.py: make_reduce_tile_distribution_encoding,
 *  make_static_distributed_tensor, _lane_p_index, _r_butterfly_plan,
 *  block_tile_reduce_sync.
 * ========================================================================== */

/* TileDistributionEncoding.num_elements_per_thread == prod(Y_lengths), where a Y
 * mapped to R (major==0) draws its length from Rs, otherwise from Hs. Mirrors
 * the Python Y_lengths / num_elements_per_thread properties. */
static int rocke_dist_num_elements_per_thread(const rocke_tile_distribution_encoding_t* e)
{
    int prod = 1;
    int i;
    for(i = 0; i < e->num_Y; ++i)
    {
        prod *= rocke_dist_bucket_length(e, e->Ys_major[i], e->Ys_minor[i]);
    }
    return prod;
}

/* bit_length()-1 of a positive int (== floor(log2(n)) for a power of two). */
static int rocke_dist_bit_length_m1(int n)
{
    int bits = 0;
    while(n > 0)
    {
        ++bits;
        n >>= 1;
    }
    return bits - 1;
}

rocke_tile_distribution_encoding_t*
    rocke_make_reduce_tile_distribution_encoding(rocke_ir_builder_t* b,
                                                 const rocke_tile_distribution_encoding_t* encoding,
                                                 const int* reduce_dim_xs,
                                                 int num_reduce)
{
    const rocke_tile_distribution_encoding_t* e = encoding;
    int num_x_in;
    int num_rh_major; /* num_x_in + 1 */
    char* is_rh_major_for_reduce = NULL; /* [num_rh_major] */
    char* is_y_for_reduce = NULL; /* [num_Y] */
    int* in2out_rh_major = NULL; /* [num_rh_major] */
    /* in2out_rh_minor stored as a flat table keyed by (major, minor). We use a
     * per-major base offset into a dense array. */
    int* minor_base = NULL; /* [num_rh_major]: base index for (major,*) */
    int* in2out_rh_minor = NULL; /* dense */
    char* consumed_by_reduced_y = NULL; /* dense, same indexing as minor */
    int* rs_lengths_out = NULL;
    int rs_count = 0;
    rocke_h_row_t* hs_out = NULL;
    int hs_count = 0;
    rocke_p_seq_t* ps_out = NULL;
    int* ys_major_out = NULL;
    int* ys_minor_out = NULL;
    int ys_count = 0;
    int total_minor = 0;
    int i, r;
    int cnt_major_out;
    int cnt_r_out;
    rocke_tile_distribution_encoding_t* out = NULL;

    if(!rocke_dist_live(b))
    {
        return NULL;
    }
    if(e == NULL)
    {
        return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
            b, ROCKE_ERR_VALUE, "make_reduce_tile_distribution_encoding: NULL encoding");
    }

    num_x_in = e->num_X;
    num_rh_major = num_x_in + 1;

    for(i = 0; i < num_reduce; ++i)
    {
        int d = reduce_dim_xs[i];
        if(d < 0 || d >= num_x_in)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_VALUE, "reduce_dim %d out of range (num_X=%d)", d, num_x_in);
        }
    }

    is_rh_major_for_reduce = (char*)rocke_arena_calloc(&b->arena, (size_t)num_rh_major);
    in2out_rh_major = (int*)rocke_arena_alloc(&b->arena, (size_t)num_rh_major * sizeof(int));
    minor_base = (int*)rocke_arena_alloc(&b->arena, (size_t)num_rh_major * sizeof(int));
    if(is_rh_major_for_reduce == NULL || in2out_rh_major == NULL || minor_base == NULL)
    {
        return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }
    for(i = 0; i < num_reduce; ++i)
    {
        is_rh_major_for_reduce[reduce_dim_xs[i] + 1] = 1;
    }

    /* Dense (major,minor) index space: major 0 (R) has num_R minors; major k>0
     * has Hs[k-1].count minors. minor_base[major] = prefix sum. */
    minor_base[0] = 0;
    total_minor = e->num_R;
    for(i = 1; i < num_rh_major; ++i)
    {
        minor_base[i] = total_minor;
        total_minor += e->Hs[i - 1].count;
    }
    if(total_minor <= 0)
    {
        total_minor = 1;
    }
    in2out_rh_minor = (int*)rocke_arena_alloc(&b->arena, (size_t)total_minor * sizeof(int));
    consumed_by_reduced_y = (char*)rocke_arena_calloc(&b->arena, (size_t)total_minor);
    if(in2out_rh_minor == NULL || consumed_by_reduced_y == NULL)
    {
        return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }

    /* is_y_for_reduce[i] = is_rh_major_for_reduce[Ys_major[i]] */
    if(e->num_Y > 0)
    {
        is_y_for_reduce = (char*)rocke_arena_calloc(&b->arena, (size_t)e->num_Y);
        if(is_y_for_reduce == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
    }
    for(i = 0; i < e->num_Y; ++i)
    {
        is_y_for_reduce[i] = is_rh_major_for_reduce[e->Ys_major[i]];
        if(is_y_for_reduce[i])
        {
            int mb = minor_base[e->Ys_major[i]] + e->Ys_minor[i];
            consumed_by_reduced_y[mb] = 1;
        }
    }

    /* in2out_rh_major: reduced X -> 0 (R); surviving X -> dense 1..K. */
    cnt_major_out = 0;
    for(i = 0; i < num_rh_major; ++i)
    {
        if(is_rh_major_for_reduce[i])
        {
            in2out_rh_major[i] = 0;
        }
        else
        {
            in2out_rh_major[i] = cnt_major_out;
            ++cnt_major_out;
        }
    }

    /* rs_lengths_out + in2out_rh_minor. Upper bound on rs_count = num_R + all H
     * levels of reduced X dims. */
    rs_lengths_out = (int*)rocke_arena_alloc(
        &b->arena, (size_t)(total_minor > 0 ? total_minor : 1) * sizeof(int));
    if(rs_lengths_out == NULL)
    {
        return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }
    rs_count = 0;
    /* input R levels carry over unchanged. */
    for(i = 0; i < e->num_R; ++i)
    {
        rs_lengths_out[rs_count] = e->Rs[i];
        in2out_rh_minor[minor_base[0] + i] = i;
        ++rs_count;
    }
    cnt_r_out = e->num_R;
    for(i = 1; i < num_rh_major; ++i)
    {
        const rocke_h_row_t* h = &e->Hs[i - 1];
        int level;
        if(is_rh_major_for_reduce[i])
        {
            for(level = 0; level < h->count; ++level)
            {
                int mb = minor_base[i] + level;
                if(!consumed_by_reduced_y[mb])
                {
                    rs_lengths_out[rs_count] = h->levels[level];
                    ++rs_count;
                    in2out_rh_minor[mb] = cnt_r_out;
                    ++cnt_r_out;
                }
                /* else: consumed by a reduced Y -> folded away entirely. */
            }
        }
        else
        {
            for(level = 0; level < h->count; ++level)
            {
                in2out_rh_minor[minor_base[i] + level] = level;
            }
        }
    }

    /* Surviving X dims keep their H decomposition. */
    if(cnt_major_out > 1)
    {
        hs_out = (rocke_h_row_t*)rocke_arena_alloc(
            &b->arena, (size_t)(cnt_major_out - 1) * sizeof(rocke_h_row_t));
        if(hs_out == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
    }
    hs_count = 0;
    for(i = 0; i < num_x_in; ++i)
    {
        if(!is_rh_major_for_reduce[i + 1])
        {
            hs_out[hs_count] = e->Hs[i]; /* arena-owned levels reused */
            ++hs_count;
        }
    }

    /* P dims: re-point every contribution. */
    if(e->num_P > 0)
    {
        ps_out = (rocke_p_seq_t*)rocke_arena_alloc(&b->arena,
                                                   (size_t)e->num_P * sizeof(rocke_p_seq_t));
        if(ps_out == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
        for(i = 0; i < e->num_P; ++i)
        {
            const rocke_p_seq_t* p = &e->Ps[i];
            int* nm = NULL;
            int* nn = NULL;
            int j;
            if(p->count > 0)
            {
                nm = (int*)rocke_arena_alloc(&b->arena, (size_t)p->count * sizeof(int));
                nn = (int*)rocke_arena_alloc(&b->arena, (size_t)p->count * sizeof(int));
                if(nm == NULL || nn == NULL)
                {
                    return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                        b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
                }
            }
            for(j = 0; j < p->count; ++j)
            {
                int maj = p->major[j];
                int minor = p->minor[j];
                nm[j] = in2out_rh_major[maj];
                nn[j] = in2out_rh_minor[minor_base[maj] + minor];
            }
            ps_out[i].count = p->count;
            ps_out[i].major = nm;
            ps_out[i].minor = nn;
        }
    }

    /* Surviving Y dims. */
    if(e->num_Y > 0)
    {
        ys_major_out = (int*)rocke_arena_alloc(&b->arena, (size_t)e->num_Y * sizeof(int));
        ys_minor_out = (int*)rocke_arena_alloc(&b->arena, (size_t)e->num_Y * sizeof(int));
        if(ys_major_out == NULL || ys_minor_out == NULL)
        {
            return (rocke_tile_distribution_encoding_t*)rocke_dist_set_err(
                b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
        }
    }
    ys_count = 0;
    for(i = 0; i < e->num_Y; ++i)
    {
        if(!is_y_for_reduce[i])
        {
            int maj = e->Ys_major[i];
            int minor = e->Ys_minor[i];
            ys_major_out[ys_count] = in2out_rh_major[maj];
            ys_minor_out[ys_count] = in2out_rh_minor[minor_base[maj] + minor];
            ++ys_count;
        }
    }

    (void)r;
    out = rocke_make_tile_distribution_encoding(b,
                                                rs_count > 0 ? rs_lengths_out : NULL,
                                                rs_count,
                                                hs_count > 0 ? hs_out : NULL,
                                                hs_count,
                                                e->num_P > 0 ? ps_out : NULL,
                                                e->num_P,
                                                ys_count > 0 ? ys_major_out : NULL,
                                                ys_count > 0 ? ys_minor_out : NULL,
                                                ys_count);
    return out;
}

rocke_static_distributed_tensor_t* rocke_make_static_distributed_tensor(
    rocke_ir_builder_t* b, const rocke_tile_distribution_t* distribution, const rocke_type_t* dtype)
{
    rocke_static_distributed_tensor_t* t;
    int n;

    if(!rocke_dist_live(b))
    {
        return NULL;
    }
    if(distribution == NULL || distribution->encoding == NULL)
    {
        return (rocke_static_distributed_tensor_t*)rocke_dist_set_err(
            b, ROCKE_ERR_VALUE, "make_static_distributed_tensor: NULL distribution");
    }
    n = rocke_dist_num_elements_per_thread(distribution->encoding);
    if(n < 1)
    {
        n = 1;
    }
    t = (rocke_static_distributed_tensor_t*)rocke_arena_calloc(
        &b->arena, sizeof(rocke_static_distributed_tensor_t));
    if(t == NULL)
    {
        return (rocke_static_distributed_tensor_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }
    t->distribution = distribution;
    t->dtype = dtype;
    t->num_storage = n;
    t->storage = (rocke_value_t**)rocke_arena_calloc(&b->arena, (size_t)n * sizeof(rocke_value_t*));
    if(t->storage == NULL)
    {
        return (rocke_static_distributed_tensor_t*)rocke_dist_set_err(
            b, ROCKE_ERR_OOM, "distribution: arena alloc failed");
    }
    return t;
}

/* _r_butterfly_plan(encoding, p_idx): for the R levels P dim p_idx owns, emit
 * (r_length, derivative) ascending by R-level. Mirrors distribution.py
 * _r_butterfly_plan / p_feeds_r. Writes up to ``cap`` entries; returns count. */
typedef struct rocke_r_plan_entry
{
    int r_length;
    int derivative;
} rocke_r_plan_entry_t;

static int rocke_dist_r_butterfly_plan(const rocke_tile_distribution_encoding_t* enc,
                                       int p_idx,
                                       rocke_r_plan_entry_t* out,
                                       int cap)
{
    /* owned = [(inner_idx, r_minor)] for contributions with major==0. */
    int r_minors[64];
    int lengths[64];
    int owned = 0;
    int k, j;
    const rocke_p_seq_t* p;

    if(p_idx < 0 || p_idx >= enc->num_P)
    {
        return 0;
    }
    p = &enc->Ps[p_idx];
    for(k = 0; k < p->count; ++k)
    {
        if(p->major[k] == 0)
        {
            int rm = p->minor[k];
            if(owned < 64)
            {
                r_minors[owned] = rm;
                lengths[owned] = enc->Rs[rm];
                ++owned;
            }
        }
    }
    if(owned == 0)
    {
        return 0;
    }
    /* derivative[k] = product of lengths of owned R levels AFTER k (in
     * contribution order). Then sort entries by r-level ascending. */
    {
        /* Build (r_minor, deriv, length) then selection-sort by r_minor. */
        int rm[64];
        int dv[64];
        int ln[64];
        int cnt = owned;
        for(k = 0; k < cnt; ++k)
        {
            int deriv = 1;
            for(j = k + 1; j < cnt; ++j)
            {
                deriv *= lengths[j];
            }
            rm[k] = r_minors[k];
            dv[k] = deriv;
            ln[k] = lengths[k];
        }
        /* stable selection sort by rm */
        for(k = 0; k < cnt; ++k)
        {
            int best = k;
            for(j = k + 1; j < cnt; ++j)
            {
                if(rm[j] < rm[best])
                {
                    best = j;
                }
            }
            if(best != k)
            {
                int tr = rm[k];
                rm[k] = rm[best];
                rm[best] = tr;
                int td = dv[k];
                dv[k] = dv[best];
                dv[best] = td;
                int tl = ln[k];
                ln[k] = ln[best];
                ln[best] = tl;
            }
        }
        {
            int n = cnt < cap ? cnt : cap;
            for(k = 0; k < n; ++k)
            {
                out[k].r_length = ln[k];
                out[k].derivative = dv[k];
            }
            return n;
        }
    }
}

static rocke_value_t* rocke_dist_fold(rocke_ir_builder_t* b,
                                      rocke_reduce_combine_t combine,
                                      rocke_value_t* x,
                                      rocke_value_t* y)
{
    return combine == ROCKE_REDUCE_SUM ? rocke_b_fadd(b, x, y) : rocke_b_fmax(b, x, y);
}

void rocke_block_tile_reduce_sync(rocke_ir_builder_t* b,
                                  rocke_static_distributed_tensor_t* reduced,
                                  rocke_reduce_combine_t combine,
                                  rocke_value_t* lds_buf,
                                  rocke_value_t* tid,
                                  int wave_size)
{
    const rocke_tile_distribution_encoding_t* enc;
    rocke_r_plan_entry_t lane_plan[64];
    rocke_r_plan_entry_t warp_plan[64];
    int lane_p;
    int n_lane;
    int n_warp;
    int off;
    int num_reduce_warps;
    int thread_buf_size;
    int i;

    if(!rocke_dist_live(b))
    {
        return;
    }
    if(combine != ROCKE_REDUCE_SUM && combine != ROCKE_REDUCE_MAX)
    {
        rocke_dist_set_err(b, ROCKE_ERR_VALUE, "combine must be 'sum' or 'max'");
        return;
    }
    if(reduced == NULL || reduced->distribution == NULL || reduced->distribution->encoding == NULL)
    {
        rocke_dist_set_err(b, ROCKE_ERR_VALUE, "block_tile_reduce_sync: NULL reduced");
        return;
    }
    enc = reduced->distribution->encoding;
    if(enc->num_P == 0)
    {
        rocke_dist_set_err(
            b, ROCKE_ERR_VALUE, "block_tile_reduce_sync requires at least one P dim");
        return;
    }

    /* -- Stage 1: warp-local XOR butterfly over lane-owned R levels -------- */
    lane_p = enc->num_P - 1; /* _lane_p_index */
    n_lane = rocke_dist_r_butterfly_plan(enc, lane_p, lane_plan, 64);
    for(off = 0; off < reduced->num_storage; ++off)
    {
        rocke_value_t* v_local = reduced->storage[off];
        int pi;
        if(v_local == NULL)
        {
            rocke_dist_set_err(b, ROCKE_ERR_VALUE, "reduce slot %d not initialised", off);
            return;
        }
        for(pi = 0; pi < n_lane; ++pi)
        {
            int r_length = lane_plan[pi].r_length;
            int derivative = lane_plan[pi].derivative;
            int stages = rocke_dist_bit_length_m1(r_length);
            int istage;
            for(istage = 0; istage < stages; ++istage)
            {
                rocke_value_t* remote = rocke_b_warp_shuffle_xor(b, v_local, derivative << istage);
                v_local = rocke_dist_fold(b, combine, v_local, remote);
            }
        }
        reduced->storage[off] = v_local;
    }

    /* -- Stage 2: cross-warp LDS over warp-owned R levels ------------------ */
    n_warp = (enc->num_P >= 2) ? rocke_dist_r_butterfly_plan(enc, 0, warp_plan, 64) : 0;
    num_reduce_warps = 1;
    for(i = 0; i < n_warp; ++i)
    {
        num_reduce_warps *= warp_plan[i].r_length;
    }
    if(num_reduce_warps <= 1)
    {
        return;
    }
    if(lds_buf == NULL || tid == NULL)
    {
        rocke_dist_set_err(b,
                           ROCKE_ERR_VALUE,
                           "block_tile_reduce_sync needs lds_buf + tid for cross-warp "
                           "reduce (num_reduce_warps=%d)",
                           num_reduce_warps);
        return;
    }

    {
        rocke_value_t* c_wave = rocke_b_const_i32(b, wave_size);
        rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);
        rocke_value_t* warp = rocke_b_div(b, tid, c_wave);
        rocke_value_t* local_warp_id;
        rocke_value_t* local_smem_os;

        thread_buf_size = reduced->num_storage;

        /* if lane == 0: publish thread buffer interleaved by warp. */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, lane, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, iff.then_region);
            for(i = 0; i < thread_buf_size; ++i)
            {
                rocke_value_t* v = reduced->storage[i];
                rocke_value_t* slot
                    = rocke_b_add(b, warp, rocke_b_const_i32(b, i * num_reduce_warps));
                rocke_value_t* idx[1];
                idx[0] = slot;
                rocke_b_smem_store_vN_f32(b, lds_buf, idx, 1, v, 1);
            }
            rocke_b_region_leave(b);
        }
        rocke_b_sync(b);

        local_warp_id = rocke_b_div(b, warp, rocke_b_const_i32(b, num_reduce_warps));
        local_smem_os = rocke_b_mul(b, local_warp_id, rocke_b_const_i32(b, num_reduce_warps));
        for(i = 0; i < thread_buf_size; ++i)
        {
            rocke_value_t* parts[256];
            int np = num_reduce_warps;
            int idx_i;
            for(idx_i = 0; idx_i < num_reduce_warps; ++idx_i)
            {
                rocke_value_t* slot = rocke_b_add(
                    b, local_smem_os, rocke_b_const_i32(b, i * num_reduce_warps + idx_i));
                rocke_value_t* sidx[1];
                rocke_value_t* vec;
                sidx[0] = slot;
                vec = rocke_b_smem_load_vN_f32(b, lds_buf, sidx, 1, 1);
                parts[idx_i] = rocke_b_vec_extract(b, vec, 0);
            }
            /* Pairwise power-of-two tree fold. */
            while(np > 1)
            {
                int j;
                int m = 0;
                for(j = 0; j < np; j += 2)
                {
                    parts[m++] = rocke_dist_fold(b, combine, parts[j], parts[j + 1]);
                }
                np = m;
            }
            reduced->storage[i] = parts[0];
        }
    }
}
