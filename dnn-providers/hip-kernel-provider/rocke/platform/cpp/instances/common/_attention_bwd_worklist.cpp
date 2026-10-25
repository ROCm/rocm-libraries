// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of the work-list emitters of
 * library/kernels/common/_attention_bwd_worklist.py (the parts the prep kernel
 * uses). See rocke/helper_rocke.instances.common._attention_bwd_worklist.h.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"

rocke_value_t* rocke_attn_bwd_worklist_tiles(rocke_ir_builder_t* b,
                                             rocke_value_t* len_kv,
                                             rocke_value_t* block_n)
{
    /* py: b.div(b.add(len_kv, b.sub(block_n, b.const_i32(1))), block_n) */
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* bn1 = rocke_b_sub(b, block_n, c1);
    rocke_value_t* sum = rocke_b_add(b, len_kv, bn1);
    return rocke_b_div(b, sum, block_n);
}

rocke_value_t* rocke_attn_bwd_worklist_sorts_before(rocke_ir_builder_t* b,
                                                    rocke_value_t* key_o,
                                                    rocke_value_t* o,
                                                    rocke_value_t* key_s,
                                                    rocke_value_t* s)
{
    rocke_value_t* longer = rocke_b_cmp_gt(b, key_o, key_s);
    rocke_value_t* eq = rocke_b_cmp_eq(b, key_o, key_s);
    rocke_value_t* lt = rocke_b_cmp_lt(b, o, s);
    rocke_value_t* tie = rocke_b_land(b, eq, lt);
    return rocke_b_lor(b, longer, tie);
}

void rocke_attn_bwd_worklist_store_item(rocke_ir_builder_t* b,
                                        rocke_value_t* wl,
                                        rocke_value_t* pos,
                                        rocke_value_t* seq,
                                        rocke_value_t* tile)
{
    /* py: w = b.add(b.mul(pos, b.const_i32(2)), b.const_i32(WL_HEADER_WORDS)) */
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* p2 = rocke_b_mul(b, pos, c2);
    rocke_value_t* hdr = rocke_b_const_i32(b, ROCKE_ATTN_BWD_WL_HEADER_WORDS);
    rocke_value_t* w = rocke_b_add(b, p2, hdr);
    rocke_b_global_store(b, wl, w, seq, 4);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* w1 = rocke_b_add(b, w, c1);
    rocke_b_global_store(b, wl, w1, tile, 4);
}

void rocke_attn_bwd_worklist_store_count(rocke_ir_builder_t* b,
                                         rocke_value_t* wl,
                                         rocke_value_t* count)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, wl, c0, count, 4);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* z = rocke_b_const_i32(b, 0);
    rocke_b_global_store(b, wl, c1, z, 4);
}
