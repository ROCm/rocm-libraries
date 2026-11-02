/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_dense_bwd_spec.h
 *
 * C port of library/kernels/common/attention_dense_bwd_spec.py (the dense
 * SDPA backward's spec, validation and kernarg ABI rocke.attn_dense_bwd.v1)
 * and of the per-arch builders library/kernels/gfx942/attention_dense_bwd.py
 * and library/kernels/gfx950/attention_dense_bwd.py. The kernel emitter is
 * helper_rocke.instances.common._attention_dense_bwd_shell.h.
 *
 *   Python                                  C (this header)
 *   --------------------------------------  ----------------------------------
 *   AttentionDenseBwdSpec and the per-arch  rocke_attn_bwd_dense_spec_t
 *     Gfx942/Gfx950AttentionDenseBwdSpec      (arch_class: the class's ARCH)
 *     (+ __post_init__)                       (+ _default, _init)
 *   .kernel_name() / .band() / .masked /    rocke_attn_bwd_dense_spec_kernel_name /
 *     .strides(t) / .body_spec()              _band / _masked / _strides /
 *                                             _body_spec
 *   dense_layout_strides                    rocke_attn_bwd_dense_layout_strides
 *   validate_attn_dense_bwd_spec            rocke_attn_bwd_dense_validate
 *   dense_bwd_not_built_reasons             rocke_attn_bwd_dense_not_built
 *     (and _not_built_all)
 *   attn_dense_bwd_params                   rocke_attn_bwd_dense_params
 *   build_attention_dense_bwd (per arch)    rocke_build_attn_dense_bwd_main_new
 *
 * Spec encoding: strings carry the Python spellings; an Optional[int]
 * (batch, window_left, window_right) is ROCKE_ATTN_BWD_NONE for None (an
 * explicit 0 stays distinct). body_knobs is an AttnBwdSpec
 * (rocke_attn_bwd_spec_default() is "no knobs"): every tuning field that
 * differs from AttnBwdSpec's default is a knob, exactly the canonical
 * body_knobs Python keeps (a knob equal to its default is dropped there). A
 * dense-owned field (_BODY_OWNED) set away from its default is the "not
 * tuning knobs" ValueError. Python also checks the names as passed: a name
 * given twice, or a dense-owned name passed at its AttnBwdSpec default value
 * (body_knobs=(("dtype", "fp16"),)), is refused there although the value
 * changes nothing. A caller that has those names passes them in
 * body_knob_names (n_body_knob_names of them, in the caller's order); _init
 * then refuses the same specs Python refuses. NULL (the default) means the
 * names are the non-default fields of body_knobs. Python's _is_body_default
 * also tells a bool from an int (("q_rotate", 0) stays a knob); C's typed
 * fields cannot carry that difference, nor a malformed (name, value) pair.
 * arch_class is NULL for the architecture-neutral AttentionDenseBwdSpec,
 * else the per-arch class's target; body_variant is the per-arch field
 * (ignored, and NULL, for the base class).
 *
 * Errors: the spec entry points take a rocke_attn_bwd_err_t out-parameter
 * (the Python exception's code and message); the builder records it on b
 * and returns NULL.
 *
 * Not ported (host-only): check_dense_bwd_spec_class beyond the arch_class
 * test in the builder, shape_key / SHAPE_FIELDS / non_default_fields as
 * public API, and DENSE_BODY_KNOBS as a table.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_DENSE_BWD_SPEC_H
#define ROCKE_INSTANCE_ATTENTION_DENSE_BWD_SPEC_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/instance_attention_bwd.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AttentionDenseBwdSpec / ArchAttentionDenseBwdSpec, in Python's field order
 * (arch_class first: the class, not a field). */
typedef struct rocke_attn_bwd_dense_spec
{
    const char* arch_class; /* NULL: AttentionDenseBwdSpec; "gfx942" / "gfx950" */
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
    bool ragged;
    const char* dkv_mode;
    const char* dq_mode;
    const char* epilogue;
    const char* oob;
    bool edge_tiles;
    const char* ws_layout;
    bool persistent;
    const char* persist_decode;
    bool xcd_remap;
    rocke_attn_bwd_spec_t body_knobs;
    const char* name;
    const char* body_variant; /* per-arch classes only */
    /* The body_knobs names as passed (see above); NULL: the non-default
     * fields of body_knobs. Not a Python field. */
    const char* const* body_knob_names;
    int n_body_knob_names;
} rocke_attn_bwd_dense_spec_t;

/* Python's defaults with the given class (NULL: the base); the caller sets
 * the problem fields (seqlen_q .. head_size). */
rocke_attn_bwd_dense_spec_t rocke_attn_bwd_dense_spec_default(const char* arch_class);

/* __post_init__ (__init_subclass__'s arch check included). */
rocke_status_t rocke_attn_bwd_dense_spec_init(const rocke_attn_bwd_dense_spec_t* spec,
                                              rocke_attn_bwd_err_t* err);

/* .kernel_name(): base key plus the salt over every non-default field. */
rocke_status_t rocke_attn_bwd_dense_spec_kernel_name(const rocke_attn_bwd_dense_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap,
                                                     rocke_attn_bwd_err_t* err);

/* dense_bwd_band(spec): (left, right, bottom_right), -1 = unbounded. */
void rocke_attn_bwd_dense_spec_band(const rocke_attn_bwd_dense_spec_t* spec,
                                    int* left,
                                    int* right,
                                    bool* bottom_right);

/* .masked */
bool rocke_attn_bwd_dense_spec_masked(const rocke_attn_bwd_dense_spec_t* spec);

/* dense_layout_strides(layout, heads, seq, d): (batch, head, token)
 * element strides; ROCKE_ERR_VALUE for an unknown layout. */
rocke_status_t rocke_attn_bwd_dense_layout_strides(const char* layout,
                                                   long long heads,
                                                   long long seq,
                                                   long long d,
                                                   long long out[3],
                                                   rocke_attn_bwd_err_t* err);

/* .strides(tensor), tensor one of "q", "o", "do", "dq" (query side) or any
 * other name (key side, as Python). */
rocke_status_t rocke_attn_bwd_dense_spec_strides(const rocke_attn_bwd_dense_spec_t* spec,
                                                 const char* tensor,
                                                 long long out[3],
                                                 rocke_attn_bwd_err_t* err);

/* .body_spec(): the general family's spec of the shared body, constructed
 * (rocke_attn_bwd_spec_init), not resolved. */
rocke_status_t rocke_attn_bwd_dense_spec_body_spec(const rocke_attn_bwd_dense_spec_t* spec,
                                                   rocke_attn_bwd_spec_t* out,
                                                   rocke_attn_bwd_err_t* err);

/* validate_attn_dense_bwd_spec(spec, arch): the resolved, validated body
 * spec into *out (the general validator without its main-kernel refusals,
 * as Python passes main_kernel_refusals=False). */
rocke_status_t rocke_attn_bwd_dense_validate(const rocke_attn_bwd_dense_spec_t* spec,
                                             const char* arch,
                                             rocke_attn_bwd_spec_t* out,
                                             rocke_attn_bwd_err_t* err);

/* dense_bwd_not_built_reasons(spec) (a per-arch spec): the not-built values
 * joined with ", " into out ("" when everything is built). */
rocke_status_t rocke_attn_bwd_dense_not_built(const rocke_attn_bwd_dense_spec_t* spec,
                                              char* out,
                                              size_t out_cap,
                                              rocke_attn_bwd_err_t* err);

/* attn_dense_bwd_params(): ordered (name, struct format) kernargs. */
#define ROCKE_ATTN_BWD_DENSE_PARAMS 24
int rocke_attn_bwd_dense_params(const char* names[ROCKE_ATTN_BWD_DENSE_PARAMS],
                                const char* formats[ROCKE_ATTN_BWD_DENSE_PARAMS]);

/* kernels.<arch>.attention_dense_bwd.build_attention_dense_bwd(spec,
 * arch=arch): the dense backward main kernel (rocke.attn_dense_bwd.v1, the
 * shell's emit_dense_bwd_main with sentinel=True) into b, initialised with
 * the spec's kernel_name(). An arch without a dense module is
 * ROCKE_ERR_VALUE, a spec of another class ROCKE_ERR_TYPE, an illegal spec
 * ROCKE_ERR_VALUE, a not-built knob ROCKE_ERR_NOTIMPL (b initialised with a
 * placeholder name). spec must have passed rocke_attn_bwd_dense_spec_init.
 * NULL and the builder's sticky error on any failure. */
rocke_kernel_def_t* rocke_build_attn_dense_bwd_main_new(rocke_ir_builder_t* b,
                                                        const rocke_attn_bwd_dense_spec_t* spec,
                                                        const char* arch);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <utility>
#include <vector>

namespace rocke_attn_bwd
{
/* Throwing C++ twins of the entry points above (ckc::Error like the Python
 * raises), for the dense dQ kernel. */
void dense_spec_init(const rocke_attn_bwd_dense_spec_t& spec);
std::string dense_mask_token(int left, int right, bool bottom_right);
void dense_layout_strides(
    const char* layout, long long heads, long long seq, long long d, long long out[3]);
rocke_attn_bwd_spec_t dense_body_spec(const rocke_attn_bwd_dense_spec_t& spec);
/* _check_in(name, value, allowed) */
void dense_check_in(const std::string& name,
                    const char* value,
                    const std::vector<std::string>& allowed);
extern const std::vector<std::string> DENSE_BWD_ARCHS;
/* validate_attn_dense_bwd_spec(spec, arch): the resolved body spec. */
rocke_attn_bwd_spec_t dense_validate_spec(const rocke_attn_bwd_dense_spec_t& spec,
                                          const std::string& arch);
/* _not_built_all(spec, body) */
std::vector<std::string> dense_not_built_all(const rocke_attn_bwd_dense_spec_t& spec,
                                             const rocke_attn_bwd_spec_t& body);
std::string dense_spec_kernel_name(const rocke_attn_bwd_dense_spec_t& spec);
/* salted_kernel_name(base, salted): salted are (name, repr(value)). */
std::string salted_kernel_name(const std::string& base,
                               std::vector<std::pair<std::string, std::string>> salted);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_DENSE_BWD_SPEC_H */
