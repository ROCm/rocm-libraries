// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* runtime_layout kernel arguments of the gfx942 and gfx950 tiled 2D unified
 * attention twins. Mirrors RUNTIME_LAYOUT_EXTRA_ARGS
 * (kernels/gfx942/attention_tiled_2d.py) and _RTL_PARAMS
 * (kernels/gfx950/attention_tiled_2d.py): the same 15 arguments, in the same
 * order, appended after the fixed ones and the runtime shape arguments
 * (num_query_heads, num_kv_heads, sliding_window). Not a public ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_TILED_2D_RTL_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_TILED_2D_RTL_INTERNAL_H

#include <string.h>

#include "rocke/ir.h"

/* Kernel argument index, in ABI order. The first two are ptr<i32>, the rest
 * i32. Strides are in elements; kv_block_span_bytes is the byte range of one
 * page's buffer resource. block_tables_v_ptr is V's block table; it shares
 * block_table_stride with block_tables_ptr (K's) and may be the same pointer.
 * q_uniform_len is L > 0 for a dense Q and 0 for a ragged one (offsets
 * non-decreasing, off[b+1] - off[b] >= seq_lens_q[b], off[B] the end of the
 * last block). */
typedef enum rocke_attn2d_rtl_param
{
    ROCKE_ATTN2D_RTL_BLOCK_TABLES_V = 0,
    ROCKE_ATTN2D_RTL_SEQ_LENS_Q,
    ROCKE_ATTN2D_RTL_HAS_SEQ_LENS_Q,
    ROCKE_ATTN2D_RTL_Q_UNIFORM_LEN,
    ROCKE_ATTN2D_RTL_Q_SEQ_STRIDE,
    ROCKE_ATTN2D_RTL_Q_TOK_STRIDE,
    ROCKE_ATTN2D_RTL_Q_HEAD_STRIDE,
    ROCKE_ATTN2D_RTL_O_SEQ_STRIDE,
    ROCKE_ATTN2D_RTL_O_TOK_STRIDE,
    ROCKE_ATTN2D_RTL_O_HEAD_STRIDE,
    ROCKE_ATTN2D_RTL_KV_BLOCK_STRIDE,
    ROCKE_ATTN2D_RTL_KV_TOK_STRIDE,
    ROCKE_ATTN2D_RTL_KV_HEAD_STRIDE,
    ROCKE_ATTN2D_RTL_KV_BLOCK_SPAN_BYTES,
    ROCKE_ATTN2D_RTL_Q_BLOCK_GRID,
    ROCKE_ATTN2D_RTL_NUM_PARAMS
} rocke_attn2d_rtl_param_t;

/* Declare the appended arguments, in order, into out[0..NUM_PARAMS). The
 * pointers are readonly with align 4, the scalars plain i32. */
static inline void rocke_attn2d_rtl_declare_params(rocke_ir_builder_t* b, rocke_value_t** out)
{
    static const char* const names[ROCKE_ATTN2D_RTL_NUM_PARAMS] = {
        "block_tables_v_ptr",
        "seq_lens_q_ptr",
        "has_seq_lens_q",
        "q_uniform_len",
        "q_seq_stride",
        "q_tok_stride",
        "q_head_stride",
        "o_seq_stride",
        "o_tok_stride",
        "o_head_stride",
        "kv_block_stride",
        "kv_tok_stride",
        "kv_head_stride",
        "kv_block_span_bytes",
        "q_block_grid",
    };
    const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
    for(int i = 0; i < ROCKE_ATTN2D_RTL_NUM_PARAMS; ++i)
    {
        if(i <= ROCKE_ATTN2D_RTL_SEQ_LENS_Q)
        {
            rocke_param_opts_t o;
            memset(&o, 0, sizeof(o));
            o.readonly = true;
            o.readonly_set = true;
            o.align = 4;
            o.align_set = true;
            out[i] = rocke_b_param(b, names[i], ptr_i32, &o);
        }
        else
        {
            out[i] = rocke_b_param(b, names[i], rocke_i32(), NULL);
        }
    }
}

#endif /* ROCKE_INSTANCE_ATTENTION_TILED_2D_RTL_INTERNAL_H */
