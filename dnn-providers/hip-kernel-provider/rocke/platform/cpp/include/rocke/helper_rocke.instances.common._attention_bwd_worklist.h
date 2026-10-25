/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_worklist.h -- C port of
 * the CTA work decode and work list of the attention backward kernels
 * (library/kernels/common/_attention_bwd_worklist.py).
 *
 *   Python (_attention_bwd_worklist.py)  C (this header)
 *   ----------------------------------   --------------------------------------
 *   WL_HEADER_WORDS                      ROCKE_ATTN_BWD_WL_HEADER_WORDS
 *   worklist_tiles                       rocke_attn_bwd_worklist_tiles
 *   worklist_sorts_before                rocke_attn_bwd_worklist_sorts_before
 *   worklist_store_item                  rocke_attn_bwd_worklist_store_item
 *   worklist_store_count                 rocke_attn_bwd_worklist_store_count
 *
 * Ops are created in the Python evaluation order (see the addr header).
 *
 * Not ported yet (later units): LB_* constants, worklist_words,
 * worklist_reference (host model, stays Python), WorkList, worklist_item,
 * main_grid_x (host), CtaWork, XcdRemap, xcd_remap_id, decode_cta, HeadRange
 * and head_range.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* i32 words before the first work-list item (the count, then 0). */
#define ROCKE_ATTN_BWD_WL_HEADER_WORDS 2

/* Live kv tiles of a sequence: ceil(len_kv / block_n) (len_kv >= 0). */
rocke_value_t* rocke_attn_bwd_worklist_tiles(rocke_ir_builder_t* b,
                                             rocke_value_t* len_kv,
                                             rocke_value_t* block_n);

/* i1: sequence o (query length key_o) precedes sequence s. */
rocke_value_t* rocke_attn_bwd_worklist_sorts_before(rocke_ir_builder_t* b,
                                                    rocke_value_t* key_o,
                                                    rocke_value_t* o,
                                                    rocke_value_t* key_s,
                                                    rocke_value_t* s);

/* Item pos of the list: (seq, tile) at words 2 + 2 pos / + 1. */
void rocke_attn_bwd_worklist_store_item(rocke_ir_builder_t* b,
                                        rocke_value_t* wl,
                                        rocke_value_t* pos,
                                        rocke_value_t* seq,
                                        rocke_value_t* tile);

/* The header: the item count, then 0. */
void rocke_attn_bwd_worklist_store_count(rocke_ir_builder_t* b,
                                         rocke_value_t* wl,
                                         rocke_value_t* count);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H */
