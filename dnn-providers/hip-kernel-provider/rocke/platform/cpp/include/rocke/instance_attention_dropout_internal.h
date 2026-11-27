// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Attention dropout internals. Mirrors library/kernels/common/_attention_dropout.py
 * (dropout on P) op for op and in the same order, so the
 * two engines emit byte-identical IR. Every builder call is its own statement:
 * C leaves the evaluation order of call arguments unspecified, and the order
 * the ops are created in decides their SSA names. Not a public ABI.
 *
 * Symbol map:
 *   DropoutState            -> rocke_dropout_state_t
 *   _value_or_load          -> rocke_dropout_value_or_load
 *   declare_dropout_params  -> rocke_declare_dropout_params
 *   dropout_query_counter   -> rocke_dropout_query_counter
 *   dropout_tile_counter    -> rocke_dropout_tile_counter
 *   _run_counter            -> rocke_dropout_run_counter
 *   dropout_keep_run        -> rocke_dropout_keep_run
 *   apply_dropout_tile      -> rocke_apply_dropout_tile (in place)
 * A Python slot_key callable is a rocke_dropout_slot_key_t (function + user
 * pointer); it emits IR on every call, as the Python closures do.
 * Not ported: dropout_signature, _seed_or_offset and dropout_launch_args
 * (host-side launch packing, no C caller). */
#ifndef ROCKE_INSTANCE_ATTENTION_DROPOUT_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_DROPOUT_INTERNAL_H

#include <stdbool.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.attention_dropout.h"
#include "rocke/helper_rocke.helpers.rng.h"
#include "rocke/ir.h"

/* KEYS_PER_CALL: the four words of one Philox call serve four consecutive keys. */
#define ROCKE_DROPOUT_KEYS_PER_CALL 4

/* DropoutState: what a body reads once per CTA. */
typedef struct rocke_dropout_state
{
    rocke_value_t* seed_lo;
    rocke_value_t* seed_hi;
    rocke_value_t* offset_lo;
    rocke_value_t* offset_hi;
    rocke_value_t* threshold;
    rocke_value_t* scale;
    bool xor3;
    bool mad64;
} rocke_dropout_state_t;

/* slot_key(slot): the key a slot holds within its sub-tile. */
typedef struct rocke_dropout_slot_key
{
    rocke_value_t* (*fn)(void* user, int slot);
    void* user;
} rocke_dropout_slot_key_t;

/* _value_or_load: the load sits behind the null check. */
static inline rocke_value_t*
    rocke_dropout_value_or_load(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* value)
{
    rocke_value_t* addr = rocke_b_global_ptr_to_i64(b, ptr);
    rocke_value_t* null64 = rocke_b_const_i64(b, 0);
    rocke_value_t* from_memory = rocke_b_cmp_ne(b, addr, null64);
    const rocke_type_t* result_types[1] = {rocke_i64()};
    rocke_if_else_t branch = rocke_b_scf_if_else_results(b, from_memory, result_types, 1);
    if(branch.op == NULL || branch.then_region == NULL || branch.else_region == NULL)
        return NULL;
    rocke_b_region_enter(b, branch.then_region);
    rocke_value_t* idx0 = rocke_b_const_i32(b, 0);
    rocke_value_t* loaded = rocke_b_global_load(b, ptr, idx0, rocke_i64(), 8);
    rocke_b_scf_yield(b, &loaded, 1);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, branch.else_region);
    rocke_b_scf_yield(b, &value, 1);
    rocke_b_region_leave(b);
    return branch.op->results[0];
}

/* declare_dropout_params: the params of DROPOUT_PARAMS in ABI order, then seed
 * and offset resolved and split into SGPR words. */
static inline void rocke_declare_dropout_params(rocke_ir_builder_t* b,
                                                bool xor3,
                                                bool mad64,
                                                rocke_dropout_state_t* st)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 8;
    o.align_set = true;
    const rocke_type_t* ptr_i64 = rocke_ptr_type(b, rocke_i64(), "global");
    rocke_value_t* seed_ptr = rocke_b_param(b, "dropout_seed_ptr", ptr_i64, &o);
    rocke_value_t* offset_ptr = rocke_b_param(b, "dropout_offset_ptr", ptr_i64, &o);
    rocke_value_t* seed_val = rocke_b_param(b, "dropout_seed", rocke_i64(), NULL);
    rocke_value_t* offset_val = rocke_b_param(b, "dropout_offset", rocke_i64(), NULL);
    rocke_value_t* threshold = rocke_b_param(b, "dropout_threshold", rocke_i32(), NULL);
    rocke_value_t* scale = rocke_b_param(b, "dropout_scale", rocke_f32(), NULL);
    rocke_value_t* seed = rocke_dropout_value_or_load(b, seed_ptr, seed_val);
    rocke_value_t* offset = rocke_dropout_value_or_load(b, offset_ptr, offset_val);
    rocke_value_t* shift = rocke_b_const_i64(b, 32);
    rocke_value_t* t;
    t = rocke_b_trunc(b, seed, rocke_i32());
    st->seed_lo = rocke_b_to_sgpr_u32(b, t);
    t = rocke_b_lshr(b, seed, shift);
    t = rocke_b_trunc(b, t, rocke_i32());
    st->seed_hi = rocke_b_to_sgpr_u32(b, t);
    t = rocke_b_trunc(b, offset, rocke_i32());
    st->offset_lo = rocke_b_to_sgpr_u32(b, t);
    t = rocke_b_lshr(b, offset, shift);
    t = rocke_b_trunc(b, t, rocke_i32());
    st->offset_hi = rocke_b_to_sgpr_u32(b, t);
    st->threshold = threshold;
    st->scale = scale;
    st->xor3 = xor3;
    st->mad64 = mad64;
}

/* dropout_query_counter: counter words of key 0 of row (batch, head, query),
 * row = (batch * num_heads + head) * seqlen_q + query, widened to i64 for the
 * multiply by seqlen_q (varlen batches outgrow i32 there). */
static inline void rocke_dropout_query_counter(rocke_ir_builder_t* b,
                                               const rocke_dropout_state_t* st,
                                               rocke_value_t* batch,
                                               rocke_value_t* head,
                                               rocke_value_t* query,
                                               rocke_value_t* num_heads,
                                               rocke_value_t* seqlen_q,
                                               rocke_value_t* seqlen_kv,
                                               rocke_value_t* ctr[2])
{
    rocke_value_t* bh = rocke_b_mul(b, batch, num_heads);
    bh = rocke_b_add(b, bh, head);
    rocke_value_t* bh64 = rocke_b_zext(b, bh, rocke_i64());
    rocke_value_t* sq64 = rocke_b_zext(b, seqlen_q, rocke_i64());
    rocke_value_t* r = rocke_b_mul(b, bh64, sq64);
    rocke_value_t* query64 = rocke_b_zext(b, query, rocke_i64());
    r = rocke_b_add(b, r, query64);
    rocke_b_dropout_row_counter(b, st->offset_lo, st->offset_hi, r, seqlen_kv, &ctr[0], &ctr[1]);
}

/* dropout_tile_counter: counter of this lane's slot 0 in the KV tile starting
 * at key tile_key0. */
static inline void rocke_dropout_tile_counter(rocke_ir_builder_t* b,
                                              rocke_value_t* const row_ctr[2],
                                              rocke_value_t* tile_key0,
                                              rocke_dropout_slot_key_t slot_key,
                                              rocke_value_t* out[2])
{
    rocke_value_t* key0 = slot_key.fn(slot_key.user, 0);
    rocke_value_t* key = rocke_b_add(b, tile_key0, key0);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* k4 = rocke_b_lshr(b, key, two);
    rocke_b_philox_counter_add(b, row_ctr[0], row_ctr[1], k4, &out[0], &out[1]);
}

/* _run_counter: counter of the call of run (nsub, g). */
static inline void rocke_dropout_run_counter(rocke_ir_builder_t* b,
                                             rocke_value_t* const tile_ctr[2],
                                             int nsub,
                                             int g,
                                             rocke_dropout_slot_key_t slot_key,
                                             int sub_keys,
                                             rocke_value_t* out[2])
{
    if(nsub == 0 && g == 0)
    {
        out[0] = tile_ctr[0];
        out[1] = tile_ctr[1];
        return;
    }
    rocke_value_t* sub0 = rocke_b_const_i32(b, (int64_t)nsub * sub_keys);
    rocke_value_t* key_g = slot_key.fn(slot_key.user, ROCKE_DROPOUT_KEYS_PER_CALL * g);
    rocke_value_t* key_0 = slot_key.fn(slot_key.user, 0);
    rocke_value_t* in_sub = rocke_b_sub(b, key_g, key_0);
    rocke_value_t* dist = rocke_b_add(b, sub0, in_sub);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    rocke_value_t* k4 = rocke_b_lshr(b, dist, two);
    rocke_b_philox_counter_add(b, tile_ctr[0], tile_ctr[1], k4, &out[0], &out[1]);
}

/* dropout_keep_run: keep predicates of slots 4g .. 4g + 3 of sub-tile nsub. */
static inline void rocke_dropout_keep_run(rocke_ir_builder_t* b,
                                          const rocke_dropout_state_t* st,
                                          rocke_value_t* const tile_ctr[2],
                                          int nsub,
                                          int g,
                                          rocke_dropout_slot_key_t slot_key,
                                          int sub_keys,
                                          rocke_value_t* keep[4])
{
    rocke_value_t* ctr[2];
    rocke_value_t* words[4];
    rocke_dropout_run_counter(b, tile_ctr, nsub, g, slot_key, sub_keys, ctr);
    rocke_b_dropout_words(b, st->seed_lo, st->seed_hi, ctr[0], ctr[1], st->xor3, st->mad64, words);
    for(int w = 0; w < 4; ++w)
        keep[w] = rocke_b_dropout_keep(b, words[w], st->threshold);
}

/* apply_dropout_tile: zero the dropped entries of one KV tile's P, in place.
 * p is flat [nsub * slots_per_sub + slot]; the scale stays out (folded into
 * 1 / l). */
static inline void rocke_apply_dropout_tile(rocke_ir_builder_t* b,
                                            const rocke_dropout_state_t* st,
                                            rocke_value_t* const row_ctr[2],
                                            rocke_value_t** p,
                                            int nsubs,
                                            int slots_per_sub,
                                            rocke_value_t* tile_key0,
                                            rocke_dropout_slot_key_t slot_key,
                                            int sub_keys)
{
    rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
    rocke_value_t* tile_ctr[2];
    rocke_dropout_tile_counter(b, row_ctr, tile_key0, slot_key, tile_ctr);
    for(int nsub = 0; nsub < nsubs; ++nsub)
    {
        rocke_value_t** sub = p + nsub * slots_per_sub;
        for(int g = 0; g < slots_per_sub / ROCKE_DROPOUT_KEYS_PER_CALL; ++g)
        {
            rocke_value_t* keep[4];
            rocke_dropout_keep_run(b, st, tile_ctr, nsub, g, slot_key, sub_keys, keep);
            for(int w = 0; w < 4; ++w)
            {
                int slot = g * ROCKE_DROPOUT_KEYS_PER_CALL + w;
                sub[slot] = rocke_b_select(b, keep[w], sub[slot], zero);
            }
        }
    }
}

#endif /* ROCKE_INSTANCE_ATTENTION_DROPOUT_INTERNAL_H */
