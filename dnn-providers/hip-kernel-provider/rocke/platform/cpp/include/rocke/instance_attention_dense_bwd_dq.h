/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_dense_bwd_dq.h
 *
 * C port of library/kernels/common/attention_dense_bwd_dq.py, the split dQ
 * kernel of the dense SDPA backward (rocke.attn_dense_bwd_dq.v1): the
 * general split dQ body (helper_rocke.instances.common._attention_bwd_dqbody.h)
 * behind the dense addressing and band classification
 * (helper_rocke.instances.common._attention_dense_bwd_shell.h).
 *
 *   Python (attention_dense_bwd_dq.py)  C (this header)
 *   ----------------------------------  --------------------------------------
 *   AttentionDenseBwdDqSpec             rocke_attn_bwd_dense_dq_spec_t
 *     (+ __post_init__, kernel_name)      (+ _default, _init, _kernel_name)
 *   validate_attn_dense_bwd_dq_spec     rocke_attn_bwd_dense_dq_validate
 *   dense_dq_static_classes             rocke_attn_bwd_dense_dq_static_classes
 *   attn_dense_bwd_dq_params            rocke_attn_bwd_dense_dq_params
 *   build_attention_dense_bwd_dq        rocke_build_attn_dense_bwd_dq_new
 *     (emit_dense_bwd_dq)
 *
 * Spec encoding: as rocke/instance_attention_dense_bwd_spec.h (an
 * Optional[int] is ROCKE_ATTN_BWD_NONE for None); has_dq false is dq=None,
 * which _init replaces with the problem's default AttnBwdDqSpec (as
 * __post_init__ does).
 *
 * Refusals stay Python: the dQ validator here does not consult the recorded
 * resource-gate tables of attention_bwd_dq.py (see
 * rocke/instance_attention_bwd_dq.h).
 *
 * Not ported (host-only): AttentionDenseBwdDqSpec.of (plan glue) and
 * attn_dense_bwd_dq_grid (launch geometry); emit_dense_bwd_dq's
 * sentinel=False (negative-control tests only).
 */
#ifndef ROCKE_INSTANCE_ATTENTION_DENSE_BWD_DQ_H
#define ROCKE_INSTANCE_ATTENTION_DENSE_BWD_DQ_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/instance_attention_bwd_dq.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AttentionDenseBwdDqSpec, in Python's field order. */
typedef struct rocke_attn_bwd_dense_dq_spec
{
    int seqlen_q;
    int seqlen_kv;
    int num_query_heads;
    int num_kv_heads;
    int head_size;
    int batch; /* Optional */
    bool causal;
    bool causal_bottom_right;
    const char* dtype;
    const char* layout;
    int sliding_window;
    int window_left; /* Optional */
    int window_right; /* Optional */
    const char* oob;
    const char* ws_layout;
    bool has_dq; /* false: dq=None */
    rocke_attn_bwd_dq_spec_t dq;
    const char* name;
} rocke_attn_bwd_dense_dq_spec_t;

/* Python's defaults; the caller sets the problem fields. */
rocke_attn_bwd_dense_dq_spec_t rocke_attn_bwd_dense_dq_spec_default(void);

/* __post_init__ in place (dq=None becomes the problem's AttnBwdDqSpec). */
rocke_status_t rocke_attn_bwd_dense_dq_spec_init(rocke_attn_bwd_dense_dq_spec_t* spec,
                                                 rocke_attn_bwd_err_t* err);

/* .kernel_name() */
rocke_status_t rocke_attn_bwd_dense_dq_spec_kernel_name(const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                        char* out,
                                                        size_t out_cap,
                                                        rocke_attn_bwd_err_t* err);

/* validate_attn_dense_bwd_dq_spec(spec, arch): the resolved, validated dQ
 * spec into *out (without the Python-only refusal tables). */
rocke_status_t rocke_attn_bwd_dense_dq_validate(const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                const char* arch,
                                                rocke_attn_bwd_dq_spec_t* out,
                                                rocke_attn_bwd_err_t* err);

/* dense_dq_static_classes(s_q=, s_kv=, rows=, k_n0=, left=, right=,
 * bottom_right=, masked=) -> (any_edge, any_interior), evaluated with the
 * kernel's range formulas on host ints (HostInt). */
rocke_status_t rocke_attn_bwd_dense_dq_static_classes(long long s_q,
                                                      long long s_kv,
                                                      int rows,
                                                      int k_n0,
                                                      int left,
                                                      int right,
                                                      bool bottom_right,
                                                      bool masked,
                                                      bool* any_edge,
                                                      bool* any_interior,
                                                      rocke_attn_bwd_err_t* err);

/* attn_dense_bwd_dq_params(): ordered (name, struct format) kernargs. */
#define ROCKE_ATTN_BWD_DENSE_DQ_PARAMS 13
int rocke_attn_bwd_dense_dq_params(const char* names[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS],
                                   const char* formats[ROCKE_ATTN_BWD_DENSE_DQ_PARAMS]);

/* build_attention_dense_bwd_dq(spec, arch=arch) (emit_dense_bwd_dq with
 * sentinel=True) into b, initialised with the spec's kernel_name(). spec
 * must have passed rocke_attn_bwd_dense_dq_spec_init. A spec the validator
 * refuses is ROCKE_ERR_VALUE (b initialised with a placeholder name). NULL
 * and the builder's sticky error on any failure. */
rocke_kernel_def_t* rocke_build_attn_dense_bwd_dq_new(rocke_ir_builder_t* b,
                                                      const rocke_attn_bwd_dense_dq_spec_t* spec,
                                                      const char* arch);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_DENSE_BWD_DQ_H */
