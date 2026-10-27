// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of kernels/common/_split_kv_reduce.py::reduce_weighted_segments,
 * the weighted segment sum (pass 3) of the gfx942 / gfx950 tiled-3D split-KV
 * reduce. Every lane loads 4 consecutive f32 head dims per segment, so
 * head_size / 4 lanes cover one segment row and the wave splits into
 * 64 / (head_size / 4) lane groups that sum contiguous segment runs; groups
 * 1..G-1 hand their partial sums to group 0 through LDS. The builder-call
 * sequence is a byte-identical translation of the Python helper: each Python
 * argument expression that creates a value is bound to a local first, in
 * Python evaluation order.
 */

#include <stdio.h> /* snprintf */

#include "rocke/instance_attention_split_kv_reduce.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

enum
{
    ROCKE_SKR_VEC = 4, /* f32 elements per lane per segment, long list (16-byte loads) */
    /* Segment loads one lane issues per loop trip: 16 when the wave splits
     * into several lane groups, 32 when one group (head size 192 / 256) walks
     * every segment alone. */
    ROCKE_SKR_UNROLL = 16,
    ROCKE_SKR_UNROLL_ONE_GROUP = 32
};

/* From this many segments on, the reduce asks for one wave per SIMD so the
 * backend keeps more segment loads in flight. Such a split only comes with a
 * small grid (few rows), so the lost occupancy costs nothing there. */
#define ROCKE_SKR_ONE_WAVE_MIN_SEGMENTS 32

int rocke_split_kv_reduce_waves_per_eu(bool has_requested, int requested, int num_segments)
{
    if(has_requested)
    {
        return requested;
    }
    return num_segments >= ROCKE_SKR_ONE_WAVE_MIN_SEGMENTS ? 1 : 0;
}

typedef struct rocke_skr_state
{
    rocke_ir_builder_t* b;
    const rocke_split_kv_reduce_args_t* a;
    rocke_value_t* col;
    int vec; /* f32 elements per lane per segment */
} rocke_skr_state_t;

/* accs[k] += vals[k] * factor[seg] (Python _accumulate). in_rng == NULL is
 * Python None. */
static void rocke__skr_accumulate(rocke_skr_state_t* st,
                                  rocke_value_t** accs,
                                  rocke_value_t* seg,
                                  rocke_value_t* in_rng)
{
    rocke_ir_builder_t* b = st->b;
    const rocke_split_kv_reduce_args_t* a = st->a;
    rocke_value_t* seg_safe = seg;
    rocke_value_t* factor;
    rocke_value_t* vals;
    rocke_value_t* idx = NULL;
    rocke_value_t* valid = NULL;
    int k;

    if(in_rng != NULL)
    {
        rocke_value_t* zero_i = rocke_b_const_i32(b, 0);
        seg_safe = rocke_b_select(b, in_rng, seg, zero_i);
    }
    {
        rocke_value_t* indices[1];
        indices[0] = seg_safe;
        factor
            = rocke_b_vec_extract(b, rocke_b_smem_load_vN_f32(b, a->factor_lds, indices, 1, 1), 0);
    }
    if(in_rng != NULL)
    {
        factor = rocke_b_select(b, in_rng, factor, a->zero_f);
    }
    {
        const char* names[4] = {"token", "head", "seg", "dim"};
        rocke_value_t* values[4];
        values[0] = a->q_token;
        values[1] = a->q_head;
        values[2] = seg_safe;
        values[3] = st->col;
        if(!rocke_transforms_descriptor_offset(b, a->seg_acc_desc, names, values, 4, &idx, &valid))
        {
            return;
        }
    }
    vals = rocke_b_global_load_vN(b, a->seg_out, idx, rocke_f32(), st->vec, 4 * st->vec);
    for(k = 0; k < st->vec; ++k)
    {
        rocke_value_t* e = rocke_b_vec_extract(b, vals, k);
        rocke_value_t* p = rocke_b_fmul(b, e, factor);
        accs[k] = rocke_b_fadd(b, accs[k], p);
    }
}

/* Group 0 (or every live lane when there is one group): fold the other groups'
 * partials, normalize, cast, store (Python _finish). */
static void rocke__skr_finish(rocke_skr_state_t* st,
                              rocke_value_t* const* accs,
                              int groups,
                              rocke_value_t* part_lds)
{
    rocke_ir_builder_t* b = st->b;
    const rocke_split_kv_reduce_args_t* a = st->a;
    rocke_value_t* final_v[ROCKE_SKR_VEC];
    rocke_value_t* outs[ROCKE_SKR_VEC];
    rocke_value_t* out_idx = NULL;
    rocke_value_t* valid = NULL;
    int g;
    int k;

    for(k = 0; k < st->vec; ++k)
    {
        final_v[k] = accs[k];
    }
    for(g = 1; g < groups; ++g)
    {
        rocke_value_t* indices[2];
        rocke_value_t* part;
        indices[0] = rocke_b_const_i32(b, g - 1);
        indices[1] = st->col;
        part = rocke_b_smem_load_vN_f32(b, part_lds, indices, 2, st->vec);
        for(k = 0; k < st->vec; ++k)
        {
            rocke_value_t* e = rocke_b_vec_extract(b, part, k);
            final_v[k] = rocke_b_fadd(b, final_v[k], e);
        }
    }
    for(k = 0; k < st->vec; ++k)
    {
        rocke_value_t* scaled = rocke_b_fmul(b, final_v[k], a->inv_l);
        outs[k] = rocke_b_cast_f32_to(b, scaled, a->dtype);
    }
    {
        const char* names[3] = {"token", "head", "dim"};
        rocke_value_t* values[3];
        values[0] = a->q_token;
        values[1] = a->q_head;
        values[2] = st->col;
        if(!rocke_transforms_descriptor_offset(b, a->out_desc, names, values, 3, &out_idx, &valid))
        {
            return;
        }
    }
    {
        rocke_value_t* packed = rocke_b_vec_pack(b, outs, st->vec, a->dtype);
        rocke_b_global_store_vN(b, a->out, out_idx, packed, st->vec, 2 * st->vec);
    }
}

void rocke_split_kv_reduce_weighted_segments(rocke_ir_builder_t* b,
                                             const rocke_split_kv_reduce_args_t* a)
{
    rocke_skr_state_t st;
    int vec;
    int row_vec;
    int lanes_per_row;
    int groups;
    int seg_per_group;
    int always_valid;
    int unroll;
    int trips;
    int tail;
    int u;
    int k;
    rocke_value_t* lpr_v;
    rocke_value_t* grp;
    rocke_value_t* seg0;
    rocke_value_t* lane_ok = NULL;
    rocke_value_t* writer;
    rocke_value_t* part_lds = NULL;
    rocke_value_t* accs[ROCKE_SKR_VEC];

    if(b == NULL || a == NULL)
    {
        return;
    }
    /* 16-byte loads for a long segment list; a short list (a large grid of
     * rows) covers the row with one lane group when head_size / 64 is 2 or 4. */
    vec = ROCKE_SKR_VEC;
    row_vec = a->head_size / a->threads;
    if(a->num_segments < ROCKE_SKR_ONE_WAVE_MIN_SEGMENTS && (row_vec == 2 || row_vec == 4)
       && row_vec * a->threads == a->head_size)
    {
        vec = row_vec;
    }
    lanes_per_row = a->head_size / vec;
    if(lanes_per_row * vec != a->head_size || lanes_per_row > a->threads)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "split-KV reduce: head_size %d needs head_size / %d <= %d lanes",
                        a->head_size,
                        vec,
                        a->threads);
        return;
    }
    groups = a->threads / lanes_per_row;
    seg_per_group = (a->num_segments + groups - 1) / groups;
    /* Group g owns the contiguous segments [g * seg_per_group, (g + 1) *
     * seg_per_group). Only the last group can run past num_segments, and only
     * in its slots >= always_valid; those slots go to the range-checked tail. */
    always_valid = a->num_segments - (groups - 1) * seg_per_group;
    if(always_valid > seg_per_group)
    {
        always_valid = seg_per_group;
    }
    if(always_valid < 0)
    {
        always_valid = 0;
    }
    unroll = groups == 1 ? ROCKE_SKR_UNROLL_ONE_GROUP : ROCKE_SKR_UNROLL;
    trips = always_valid / unroll;
    tail = seg_per_group - trips * unroll;

    st.b = b;
    st.a = a;
    st.vec = vec;

    lpr_v = rocke_b_const_i32(b, lanes_per_row);
    grp = rocke_b_div(b, a->tid, lpr_v);
    {
        rocke_value_t* m = rocke_b_mul(b, grp, lpr_v);
        rocke_value_t* s = rocke_b_sub(b, a->tid, m);
        rocke_value_t* vec_v = rocke_b_const_i32(b, vec);
        st.col = rocke_b_mul(b, s, vec_v);
    }
    if(groups * lanes_per_row != a->threads)
    {
        rocke_value_t* g_v = rocke_b_const_i32(b, groups);
        rocke_value_t* zero_i;
        lane_ok = rocke_b_cmp_lt(b, grp, g_v);
        zero_i = rocke_b_const_i32(b, 0);
        grp = rocke_b_select(b, lane_ok, grp, zero_i);
    }
    {
        rocke_value_t* spg = rocke_b_const_i32(b, seg_per_group);
        seg0 = rocke_b_mul(b, grp, spg);
    }

    for(k = 0; k < vec; ++k)
    {
        accs[k] = a->zero_f;
    }
    if(trips > 0)
    {
        rocke_iter_arg_t iter_args[ROCKE_SKR_VEC];
        char names[ROCKE_SKR_VEC][16];
        rocke_for_t loop;
        rocke_value_t* lo = rocke_b_const_i32(b, 0);
        rocke_value_t* hi = rocke_b_const_i32(b, trips);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        for(k = 0; k < vec; ++k)
        {
            snprintf(names[k], sizeof(names[k]), "racc%d", k);
            iter_args[k].name = names[k];
            iter_args[k].init = a->zero_f;
        }
        loop = rocke_b_scf_for_iter(b, lo, hi, step, iter_args, vec, "rtrip", false, true);
        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* body_accs[ROCKE_SKR_VEC];
            rocke_value_t* base;
            {
                rocke_value_t* stride = rocke_b_const_i32(b, unroll);
                rocke_value_t* off = rocke_b_mul(b, loop.iv, stride);
                base = rocke_b_add(b, seg0, off);
            }
            for(k = 0; k < vec; ++k)
            {
                body_accs[k] = loop.iter_vars ? loop.iter_vars[k] : NULL;
            }
            for(u = 0; u < unroll; ++u)
            {
                rocke_value_t* seg = base;
                if(u != 0)
                {
                    rocke_value_t* du = rocke_b_const_i32(b, u);
                    seg = rocke_b_add(b, base, du);
                }
                rocke__skr_accumulate(&st, body_accs, seg, NULL);
            }
            rocke_b_scf_yield(b, body_accs, vec);
        }
        rocke_b_region_leave(b);
        for(k = 0; k < vec; ++k)
        {
            accs[k] = (loop.op != NULL && loop.op->num_results > k) ? loop.op->results[k] : NULL;
        }
    }
    for(u = 0; u < tail; ++u)
    {
        int slot = trips * unroll + u;
        rocke_value_t* in_rng = NULL;
        rocke_value_t* seg;
        {
            rocke_value_t* ds = rocke_b_const_i32(b, slot);
            seg = rocke_b_add(b, seg0, ds);
        }
        if(slot >= always_valid)
        {
            rocke_value_t* ns = rocke_b_const_i32(b, a->num_segments);
            in_rng = rocke_b_cmp_lt(b, seg, ns);
        }
        rocke__skr_accumulate(&st, accs, seg, in_rng);
    }

    writer = lane_ok;
    if(groups > 1)
    {
        int shape[2];
        rocke_if_t guard;
        shape[0] = groups - 1;
        shape[1] = a->head_size;
        part_lds = rocke_b_smem_alloc_f32(b, shape, 2, "seg_part");
        {
            rocke_value_t* zero_i = rocke_b_const_i32(b, 0);
            rocke_value_t* is_part = rocke_b_cmp_gt(b, grp, zero_i);
            guard = rocke_b_scf_if(b, is_part);
        }
        rocke_b_region_enter(b, guard.then_region);
        {
            rocke_value_t* indices[2];
            rocke_value_t* one = rocke_b_const_i32(b, 1);
            rocke_value_t* packed;
            indices[0] = rocke_b_sub(b, grp, one);
            indices[1] = st.col;
            packed = rocke_b_vec_pack(b, accs, vec, rocke_f32());
            rocke_b_smem_store_vN_f32(b, part_lds, indices, 2, packed, vec);
        }
        rocke_b_region_leave(b);
        rocke_b_sync(b);
        {
            rocke_value_t* zero_i = rocke_b_const_i32(b, 0);
            writer = rocke_b_cmp_eq(b, grp, zero_i);
        }
    }

    if(writer == NULL)
    {
        rocke__skr_finish(&st, accs, groups, part_lds);
    }
    else
    {
        rocke_if_t guard = rocke_b_scf_if(b, writer);
        rocke_b_region_enter(b, guard.then_region);
        rocke__skr_finish(&st, accs, groups, part_lds);
        rocke_b_region_leave(b);
    }
}
