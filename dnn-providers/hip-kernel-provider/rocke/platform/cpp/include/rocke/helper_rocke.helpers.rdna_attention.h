/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helpers/rdna_attention.py: the multi-wave WMMA flash-attention
 * forward body for RDNA (gfx11 and gfx12), FP16/BF16.
 *
 * Python                              C99 (this header)
 * ----------------------------------  ---------------------------------------
 * RdnaAttentionConfig                 rocke_rdna_attn_config_t
 *   .validate()                       rocke_rdna_attn_config_validate(...)
 *   .pair / .waves / .threads         rocke_rdna_attn_pair/_waves/_threads(...)
 *   .log_domain                       rocke_rdna_attn_log_domain(...)
 * rdna_attention_lds_bytes(cfg)       rocke_rdna_attention_lds_bytes(...)
 * rdna_attention_fwd_body(b, cfg, ..) rocke_rdna_attention_fwd_body(...)
 *
 * Scores are computed transposed (S^T = K Q^T) with one query row per lane,
 * PV transposed (O^T += V^T P^T) with P fed from registers, and K / V^T
 * staged by cooperative 16-byte buffer loads into padded LDS shared by every
 * wave of the CTA. The caller declares the kernel parameters, decodes the CTA
 * and builds the per-(batch, head) buffer resources; the body emits the K
 * loop and the epilogue (O and optional LSE stores), not ret().
 *
 * FIDELITY. Every rocke_b_* call reproduces the Python builder-call order,
 * operands and compile-time constants op for op, so the emitted IR is
 * byte-identical to the Python helper's emission.
 *
 * ERROR MODEL. A Python `raise ValueError` maps to a sticky
 * rocke_i_set_err(ROCKE_ERR_VALUE) on the builder plus an early return.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_RDNA_ATTENTION_H
#define ROCKE_HELPER_ROCKE_HELPERS_RDNA_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flat mirror of @dataclass(frozen=True) RdnaAttentionConfig. */
typedef struct rocke_rdna_attn_config
{
    const char* dtype; /* "fp16" | "bf16" */
    int head_dim; /* Q/K head-width bucket (multiple of 16) */
    int head_dim_v; /* V head-width bucket (multiple of 16) */
    int block_m; /* query rows per CTA */
    int block_n; /* keys per K-loop tile (multiple of 16) */
    bool causal; /* mask class: runtime causal / window bounds */
    bool gfx12; /* gfx12 WMMA operand layout (else gfx11) */
    bool lazy_rescale; /* Python default true */
    int k_pad; /* LDS row padding of the K tile; Python default 8 */
    int v_pad; /* LDS row padding of the V^T tile; Python default 8 */
    const char* bias; /* NULL | "f32" | "q" */
    int wave_pair; /* -1: automatic (either width above 128); 0: no; 1: yes */
    bool softcap; /* runtime tanh logit cap */
    bool sinks; /* per-head sink logit seeds the softmax */
    bool alibi; /* s += slope * (key - context) */
    bool qq_bias; /* s += qq[query, key - context] inside its bounds */
    bool kv_fp8; /* K / V are OCP E4M3FN bytes (runtime scales) */
    int page_block_size; /* paged K / V page (power of two); 0: contiguous */
} rocke_rdna_attn_config_t;

/* RdnaAttentionConfig.log_domain: scores move to the log2 domain up front. */
bool rocke_rdna_attn_log_domain(const rocke_rdna_attn_config_t* cfg);

/* RdnaAttentionConfig.pair / .waves / .threads. */
bool rocke_rdna_attn_pair(const rocke_rdna_attn_config_t* cfg);
int rocke_rdna_attn_waves(const rocke_rdna_attn_config_t* cfg);
int rocke_rdna_attn_threads(const rocke_rdna_attn_config_t* cfg);

/* RdnaAttentionConfig.validate(): true if valid; otherwise false with the
 * Python ValueError message in `reason` (capacity reason_cap). */
bool rocke_rdna_attn_config_validate(const rocke_rdna_attn_config_t* cfg,
                                     char* reason,
                                     size_t reason_cap);

/* rdna_attention_lds_bytes(cfg): static LDS bytes of one CTA. */
int rocke_rdna_attention_lds_bytes(const rocke_rdna_attn_config_t* cfg);

/* Keyword operands of rdna_attention_fwd_body. Optional ones are NULL when
 * absent (lse_rsrc / stride_lse together; the causal class needs
 * diag_offset, window_left and window_right; a bias needs bias_rsrc and
 * stride_bias_q, and a Q-dtype bias also bias_lead and bias_bytes; each
 * feature needs its operands, see the Python docstring; k_bytes / v_bytes are
 * NULL for paged K / V). */
typedef struct rocke_rdna_attn_params
{
    rocke_value_t* q_rsrc;
    rocke_value_t* k_rsrc;
    rocke_value_t* v_rsrc;
    rocke_value_t* o_rsrc;
    rocke_value_t* lse_rsrc;
    rocke_value_t* stride_q;
    rocke_value_t* stride_k;
    rocke_value_t* stride_v;
    rocke_value_t* stride_o;
    rocke_value_t* stride_lse;
    rocke_value_t* k_bytes;
    rocke_value_t* v_bytes;
    rocke_value_t* head_dim_q;
    rocke_value_t* head_dim_v;
    rocke_value_t* seqlen_q;
    rocke_value_t* seqlen_k;
    rocke_value_t* q_start;
    rocke_value_t* scale_log2;
    rocke_value_t* diag_offset;
    rocke_value_t* window_left;
    rocke_value_t* window_right;
    rocke_value_t* bias_rsrc;
    rocke_value_t* stride_bias_q;
    rocke_value_t* bias_lead;
    rocke_value_t* bias_bytes;
    rocke_value_t* softcap;
    rocke_value_t* sink_log2;
    rocke_value_t* alibi_slope_log2;
    rocke_value_t* context;
    rocke_value_t* qq_rsrc;
    rocke_value_t* qq_rows;
    rocke_value_t* qq_cols;
    rocke_value_t* qq_stride;
    rocke_value_t* k_scale;
    rocke_value_t* v_scale;
    rocke_value_t* block_table;
    rocke_value_t* table_row;
    rocke_value_t* stride_k_block;
    rocke_value_t* stride_v_block;
    rocke_value_t* num_blocks;
} rocke_rdna_attn_params_t;

/* rdna_attention_fwd_body(b, cfg, **params): emit the K loop and epilogue of
 * one CTA into the caller's open kernel region. Returns the builder status. */
rocke_status_t rocke_rdna_attention_fwd_body(rocke_ir_builder_t* b,
                                             const rocke_rdna_attn_config_t* cfg,
                                             const rocke_rdna_attn_params_t* p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_RDNA_ATTENTION_H */
