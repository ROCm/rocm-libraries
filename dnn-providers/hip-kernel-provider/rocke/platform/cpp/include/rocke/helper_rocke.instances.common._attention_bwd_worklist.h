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
 *   LB_NATURAL / _REVERSE / _MIRROR /    ROCKE_ATTN_BWD_LB_NATURAL / ...
 *     _INTERLEAVE
 *   WorkList, worklist_item              rocke_attn_bwd_work_list_t, ..._worklist_item
 *   CtaWork (+ kv_tile)                  rocke_attn_bwd_cta_work_t (+ _cta_work_kv_tile)
 *   XcdRemap, xcd_remap_id               rocke_attn_bwd_xcd_remap_t, ..._xcd_remap_id
 *   decode_cta                           rocke_attn_bwd_decode_cta
 *   HeadRange, head_range                rocke_attn_bwd_head_range_t, ..._head_range
 *
 * Ops are created in the Python evaluation order (see the addr header). A
 * Python ValueError is recorded on the builder (ROCKE_ERR_VALUE).
 *
 * Not ported (host-only, stay Python): worklist_words, worklist_reference,
 * main_grid_x.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H

#include <stdbool.h>

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

/* Load-balance orders of the runtime lb_order kernarg. */
#define ROCKE_ATTN_BWD_LB_NATURAL 0
#define ROCKE_ATTN_BWD_LB_REVERSE 1
#define ROCKE_ATTN_BWD_LB_MIRROR 2
#define ROCKE_ATTN_BWD_LB_INTERLEAVE 3

/* WorkList: the WORKLIST pointer and the runtime use_worklist flag. */
typedef struct rocke_attn_bwd_work_list
{
    rocke_value_t* ptr;
    rocke_value_t* use;
} rocke_attn_bwd_work_list_t;

/* worklist_item(b, wl, idx, n_kv_tiles=, grid_batch=, grid_kt=): the
 * (sequence, kv tile) of work-list index idx. */
void rocke_attn_bwd_worklist_item(rocke_ir_builder_t* b,
                                  const rocke_attn_bwd_work_list_t* wl,
                                  rocke_value_t* idx,
                                  rocke_value_t* n_kv_tiles,
                                  rocke_value_t* grid_batch,
                                  rocke_value_t* grid_kt,
                                  rocke_value_t** out_seq,
                                  rocke_value_t** out_kt);

/* CtaWork: the decoded work of one CTA (every value CTA-uniform, i32). */
typedef struct rocke_attn_bwd_cta_work
{
    rocke_value_t* t;
    rocke_value_t* unit;
    rocke_value_t* split;
    rocke_value_t* batch;
    rocke_value_t* first_kt;
    rocke_value_t* second_kt;
    rocke_value_t* n_pass;
} rocke_attn_bwd_cta_work_t;

/* CtaWork.kv_tile(b, p): the kv tile of pass p (0 or 1). */
rocke_value_t* rocke_attn_bwd_cta_work_kv_tile(const rocke_attn_bwd_cta_work_t* w,
                                               rocke_ir_builder_t* b,
                                               rocke_value_t* p);

/* XcdRemap: n XCDs (0 = identity), chunk plane ids per XCD, gy the grid y. */
typedef struct rocke_attn_bwd_xcd_remap
{
    rocke_value_t* n;
    rocke_value_t* chunk;
    rocke_value_t* gy;
} rocke_attn_bwd_xcd_remap_t;

/* xcd_remap_id(b, wgid, num_wgs=, n=, chunk=). */
rocke_value_t* rocke_attn_bwd_xcd_remap_id(rocke_ir_builder_t* b,
                                           rocke_value_t* wgid,
                                           rocke_value_t* num_wgs,
                                           rocke_value_t* n,
                                           rocke_value_t* chunk);

/* decode_cta(b, x=, y=, z=, n_kv_tiles=, lb_order=, pair=, g_split=,
 * interleave=, xcd=, worklist=): pair / g_split / xcd / worklist NULL are
 * None. A work list with a pair is a ValueError (false). */
bool rocke_attn_bwd_decode_cta(rocke_ir_builder_t* b,
                               rocke_value_t* x,
                               rocke_value_t* y,
                               rocke_value_t* z,
                               rocke_value_t* n_kv_tiles,
                               rocke_value_t* lb_order,
                               rocke_value_t* pair,
                               rocke_value_t* g_split,
                               bool interleave,
                               const rocke_attn_bwd_xcd_remap_t* xcd,
                               const rocke_attn_bwd_work_list_t* worklist,
                               rocke_attn_bwd_cta_work_t* out);

/* HeadRange: query heads [h0, h0 + n_heads) and the K / V heads. */
typedef struct rocke_attn_bwd_head_range
{
    rocke_value_t* h0;
    rocke_value_t* n_heads;
    rocke_value_t* hk;
    rocke_value_t* hv;
} rocke_attn_bwd_head_range_t;

/* head_range(b, unit=, split=, group=, gk=, gv=, g_split=); g_split NULL is
 * None (direct mode). */
rocke_attn_bwd_head_range_t rocke_attn_bwd_head_range(rocke_ir_builder_t* b,
                                                      rocke_value_t* unit,
                                                      rocke_value_t* split,
                                                      rocke_value_t* group,
                                                      rocke_value_t* gk,
                                                      rocke_value_t* gv,
                                                      rocke_value_t* g_split);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_WORKLIST_H */
