/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd_unified.h
 *
 * C port of library/kernels/common/attention_bwd_unified.py: the unified
 * (paged) SDPA backward main kernel, rocke.attn_bwd_unified.v1 (THD queries
 * over a paged KV cache), with its spec, validation and kernarg ABI.
 *
 *   Python (attention_bwd_unified.py)    C (this header)
 *   -----------------------------------  -------------------------------------
 *   AttnBwdUnifiedSpec (+ __post_init__) rocke_attn_bwd_unified_spec_t
 *                                        (+ _spec_default, _spec_init)
 *   AttnBwdUnifiedSpec.general_spec      rocke_attn_bwd_unified_general_spec
 *   AttnBwdUnifiedSpec.page_class        rocke_attn_bwd_unified_page_class
 *   AttnBwdUnifiedSpec.kernel_name       rocke_attn_bwd_unified_kernel_name
 *   validate_attn_bwd_unified_spec       rocke_attn_bwd_unified_validate_spec
 *   attn_bwd_unified_params("main")      rocke_attn_bwd_unified_params
 *   build_attn_bwd_unified_main          rocke_build_attn_bwd_unified_main_new
 *     (_emit_main, _declare_main,        (attention_bwd_unified.cpp)
 *      _kernel_attrs, the page atomic /
 *      direct epilogues)
 *
 * Spec encoding. The unified fields are spelled out; every knob of the
 * general family (UNIFIED_KNOBS: the AttnBwdSpec fields other than
 * head_size, dtype, seq_mode, dkv_mode, mask_class, dq_mode, stage_vec and
 * name) lives in `knobs`, a rocke_attn_bwd_spec_t whose other fields are
 * ignored (general_spec sets them). Strings carry the Python spellings.
 * sub_page is ROCKE_ATTN_BWD_NONE for None (the "multi" page class), so an
 * explicit 0 stays distinct (and is refused, as Python refuses it).
 * Python's type errors (a bool sub_page) have no C spelling.
 *
 * The main kernel reuses the general builder's _kv_tile
 * (rocke/instance_attention_bwd_main_internal.h) with the paged K / V slabs
 * of _attention_bwd_paged and its own epilogue, as Python does.
 *
 * Not ported (host-only): attn_bwd_unified_default_geometry (the general
 * family's, rocke_attn_bwd_default_geometry with seq_mode "thd"),
 * AttnBwdUnifiedSpec.resolved / aux_spec / non_default_fields (the aux
 * kernels are the general family's, rocke/instance_attention_bwd_aux.h), the
 * page_* stages of attn_bwd_unified_params (the aux params), the sentinel=
 * negative-control switch of _emit_main (always True here), and the plan /
 * run / cache modules (attention_bwd_unified_plan.py, _run.py).
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_UNIFIED_H
#define ROCKE_INSTANCE_ATTENTION_BWD_UNIFIED_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_attention_bwd.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every legal page size is a multiple of this (PAGE_GRANULE). */
#define ROCKE_ATTN_BWD_UNIFIED_PAGE_GRANULE 16

/* AttnBwdUnifiedSpec, in Python's field order. */
typedef struct rocke_attn_bwd_unified_spec
{
    int head_size;
    const char* dtype; /* default "bf16" */
    const char* page_mode; /* "atomic" (default) | "direct" */
    const char* mask_class; /* "band" (default) | "none" */
    int sub_page; /* Optional: ROCKE_ATTN_BWD_NONE is None */
    int stage_vec; /* 8 (default) | 1 */
    rocke_attn_bwd_spec_t knobs; /* the UNIFIED_KNOBS fields only */
    const char* name; /* default "rocke_attn_bwd_unified" */
} rocke_attn_bwd_unified_spec_t;

/* Python's defaults; the caller sets head_size. */
rocke_attn_bwd_unified_spec_t rocke_attn_bwd_unified_spec_default(void);

/* AttnBwdUnifiedSpec.__post_init__, in place (the knobs canonicalised by the
 * general spec's __post_init__). */
rocke_status_t rocke_attn_bwd_unified_spec_init(rocke_attn_bwd_unified_spec_t* spec,
                                                rocke_attn_bwd_err_t* err);

/* general_spec(): the layout-general spec of the same tile step (THD, atomic
 * dQ), checked by AttnBwdSpec.__post_init__. */
rocke_status_t rocke_attn_bwd_unified_general_spec(const rocke_attn_bwd_unified_spec_t* spec,
                                                   rocke_attn_bwd_spec_t* out,
                                                   rocke_attn_bwd_err_t* err);

/* page_class: "multi" (sub_page None) or "sub". */
const char* rocke_attn_bwd_unified_page_class(const rocke_attn_bwd_unified_spec_t* spec);

/* kernel_name("main") into out (out_cap bytes). */
rocke_status_t rocke_attn_bwd_unified_kernel_name(const rocke_attn_bwd_unified_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap,
                                                  rocke_attn_bwd_err_t* err);

/* validate_attn_bwd_unified_spec(spec, arch): the resolved spec into *out
 * (out may alias spec). */
rocke_status_t rocke_attn_bwd_unified_validate_spec(const rocke_attn_bwd_unified_spec_t* spec,
                                                    const char* arch,
                                                    rocke_attn_bwd_unified_spec_t* out,
                                                    rocke_attn_bwd_err_t* err);

/* attn_bwd_unified_params("main"): ordered (name, struct format) kernargs;
 * formats "Q", "f", "i". Another stage is a ValueError (the page_* stages
 * are the aux params). */
rocke_status_t rocke_attn_bwd_unified_params(const char* stage,
                                             const char* names[ROCKE_ATTN_BWD_PARAMS_MAX],
                                             const char* formats[ROCKE_ATTN_BWD_PARAMS_MAX],
                                             int* n_out,
                                             rocke_attn_bwd_err_t* err);

/* build_attn_bwd_unified_main(spec, arch=) (= _emit_main(spec, arch)): the
 * unified backward main kernel into b, which this initialises with the
 * validated spec's kernel_name(). spec must have passed
 * rocke_attn_bwd_unified_spec_init (Python's builder takes a constructed
 * AttnBwdUnifiedSpec). A spec the validator refuses is ROCKE_ERR_VALUE (b
 * initialised with a placeholder name). NULL and the builder's sticky error
 * on any failure. */
rocke_kernel_def_t* rocke_build_attn_bwd_unified_main_new(rocke_ir_builder_t* b,
                                                          const rocke_attn_bwd_unified_spec_t* spec,
                                                          const char* arch);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_BWD_UNIFIED_H */
