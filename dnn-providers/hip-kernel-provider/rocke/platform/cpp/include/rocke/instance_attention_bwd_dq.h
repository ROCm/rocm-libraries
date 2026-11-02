/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd_dq.h
 *
 * C port of library/kernels/common/attention_bwd_dq.py, the split dQ kernel
 * of the layout-general SDPA backward (dq_mode = "split"), and of
 * AttnBwdSpec.dq_spec (attention_bwd.py). The body is
 * helper_rocke.instances.common._attention_bwd_dqbody.h.
 *
 *   Python (attention_bwd_dq.py)        C (this header)
 *   ----------------------------------  --------------------------------------
 *   AttnBwdDqSpec (+ __post_init__)     rocke_attn_bwd_dq_spec_t
 *                                       (+ _spec_default, _spec_init)
 *   AttnBwdDqSpec.resolved(arch)        rocke_attn_bwd_dq_spec_resolved
 *   AttnBwdDqSpec.kernel_name()         rocke_attn_bwd_dq_spec_kernel_name
 *   AttnBwdSpec.dq_spec(**knobs)        rocke_attn_bwd_spec_dq_spec
 *   dq_static_scale                     rocke_attn_bwd_dq_static_scale
 *   dq_static_unit_mults                rocke_attn_bwd_dq_static_unit_mults
 *   attn_bwd_dq_default_geometry        rocke_attn_bwd_dq_default_geometry
 *   DqLdsPlan / dq_lds_plan             rocke_attn_bwd_dq_lds_plan_t / _dq_lds_plan
 *   dq_register_estimates               rocke_attn_bwd_dq_register_estimates
 *   validate_attn_bwd_dq_spec           rocke_attn_bwd_validate_dq_spec
 *   build_attn_bwd_dq (_emit_dq)        rocke_build_attn_bwd_dq_new
 *
 * Spec encoding (as rocke/instance_attention_bwd.h): strings carry the
 * Python spellings, NULL is None for an Optional[str]; an Optional[int] is
 * ROCKE_ATTN_BWD_NONE for None (an explicit 0 stays distinct); agpr_alloc is
 * the caps header's rocke_attn_bwd_agpr_alloc_t.
 *
 * Refusals stay Python. validate (and so the builder) does not consult the
 * recorded resource-gate tables DQ_SPILLING_COMBOS, DQ_RESOURCE_REFUSED and
 * DQ_SCHED_SPILLING (attention_bwd_dq.py, keyed through
 * _attention_bwd_refusals.spec_digest): the C builder emits the kernel of a
 * spec those tables refuse, so a caller that launches it must consult
 * Python's validate_attn_bwd_dq_spec first.
 *
 * Errors: the spec entry points take a rocke_attn_bwd_err_t out-parameter
 * and return rocke_status_t (the Python ValueError message); the builder
 * records it on b (ROCKE_ERR_VALUE) and returns NULL.
 *
 * Not ported (host-only): dq_rows_per_head and attn_bwd_dq_grid (launch
 * geometry), DQ_TUNING_FIELDS / DQ_PROBLEM_FIELDS (plan bookkeeping) and the
 * refusal tables above. _emit_dq's sentinel=False (negative-control tests
 * only) is not exposed: the builder always emits the dead-row term.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_DQ_H
#define ROCKE_INSTANCE_ATTENTION_BWD_DQ_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/instance_attention_bwd.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AttnBwdDqSpec, in Python's field order. */
typedef struct rocke_attn_bwd_dq_spec
{
    int head_size;
    const char* dtype;
    const char* seq_mode;
    int stage_vec;
    const char* mask_class;
    const char* ws_layout;
    const char* static_scale;
    int waves; /* Optional */
    int block_m; /* Optional */
    int block_n; /* Optional */
    const char* atom_g02; /* Optional */
    const char* atom_g4; /* Optional */
    const char* transpose_source; /* Optional */
    const char* head_loop;
    bool stats_prescale;
    bool edge_tiles;
    int kv_pad;
    bool kv_prefetch;
    int waves_per_eu; /* Optional */
    rocke_attn_bwd_agpr_alloc_t agpr_alloc;
    const char* scheduler_strategy; /* Optional */
    const char* sched;
    bool xcd_remap;
    int narrow_vec;
    const char* name;
} rocke_attn_bwd_dq_spec_t;

/* Python's defaults; the caller sets head_size. */
rocke_attn_bwd_dq_spec_t rocke_attn_bwd_dq_spec_default(void);

/* AttnBwdDqSpec.__post_init__ (in place: agpr_alloc canonical). */
rocke_status_t rocke_attn_bwd_dq_spec_init(rocke_attn_bwd_dq_spec_t* spec,
                                           rocke_attn_bwd_err_t* err);

/* AttnBwdDqSpec.resolved(arch) into *out (out may alias spec). */
rocke_status_t rocke_attn_bwd_dq_spec_resolved(const rocke_attn_bwd_dq_spec_t* spec,
                                               const char* arch,
                                               rocke_attn_bwd_dq_spec_t* out,
                                               rocke_attn_bwd_err_t* err);

/* AttnBwdDqSpec.kernel_name(). */
rocke_status_t rocke_attn_bwd_dq_spec_kernel_name(const rocke_attn_bwd_dq_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap,
                                                  rocke_attn_bwd_err_t* err);

/* AttnBwdSpec.dq_spec(**knobs): the problem fields (head_size, dtype,
 * seq_mode, stage_vec, mask_class, ws_layout, static_scale through
 * dq_static_scale, narrow_vec, name) from *spec, every other field (the
 * knobs) from *knobs (NULL: the dataclass defaults); then __post_init__. */
rocke_status_t rocke_attn_bwd_spec_dq_spec(const rocke_attn_bwd_spec_t* spec,
                                           const rocke_attn_bwd_dq_spec_t* knobs,
                                           rocke_attn_bwd_dq_spec_t* out,
                                           rocke_attn_bwd_err_t* err);

/* dq_static_scale(main_static_scale): a static string. */
rocke_status_t rocke_attn_bwd_dq_static_scale(const char* main_static_scale,
                                              const char** out,
                                              rocke_attn_bwd_err_t* err);

/* dq_static_unit_mults(static_scale): bits of ROCKE_ATTN_BWD_UNIT_DS_MULT /
 * ROCKE_ATTN_BWD_UNIT_DQ_MULT (instance_attention_bwd.h). */
rocke_status_t rocke_attn_bwd_dq_static_unit_mults(const char* static_scale,
                                                   unsigned* mask,
                                                   rocke_attn_bwd_err_t* err);

/* attn_bwd_dq_default_geometry(arch, head_size, dtype, stage_vec): the dict
 * as a spec whose geometry fields (waves, block_m, block_n, atom_g02,
 * atom_g4, transpose_source) carry its values; every other field is the
 * default. */
rocke_status_t rocke_attn_bwd_dq_default_geometry(const char* arch,
                                                  int head_size,
                                                  const char* dtype,
                                                  int stage_vec,
                                                  rocke_attn_bwd_dq_spec_t* out,
                                                  rocke_attn_bwd_err_t* err);

/* DqLdsPlan: buffers in Python's order (q, do, k, v, out). */
#define ROCKE_ATTN_BWD_DQ_LDS_BUFFERS 5
typedef struct rocke_attn_bwd_dq_lds_buffer
{
    const char* name;
    int offset;
    int nbytes;
    int pitch;
    int rows;
} rocke_attn_bwd_dq_lds_buffer_t;

typedef struct rocke_attn_bwd_dq_lds_plan
{
    const char* arch;
    rocke_attn_bwd_dq_lds_buffer_t buffers[ROCKE_ATTN_BWD_DQ_LDS_BUFFERS];
    int total_bytes;
    int capacity_bytes;
} rocke_attn_bwd_dq_lds_plan_t;

/* dq_lds_plan(spec, arch) of a resolved spec (not checked against capacity). */
rocke_status_t rocke_attn_bwd_dq_lds_plan(const rocke_attn_bwd_dq_spec_t* spec,
                                          const char* arch,
                                          rocke_attn_bwd_dq_lds_plan_t* out,
                                          rocke_attn_bwd_err_t* err);

/* dq_register_estimates(spec, arch): {"arch_vgpr", "agpr"}. */
rocke_status_t rocke_attn_bwd_dq_register_estimates(const rocke_attn_bwd_dq_spec_t* spec,
                                                    const char* arch,
                                                    int* arch_vgpr,
                                                    int* agpr,
                                                    rocke_attn_bwd_err_t* err);

/* validate_attn_bwd_dq_spec(spec, arch) without the Python-only refusal
 * tables (see the header comment): the resolved spec into *out (out may
 * alias spec). */
rocke_status_t rocke_attn_bwd_validate_dq_spec(const rocke_attn_bwd_dq_spec_t* spec,
                                               const char* arch,
                                               rocke_attn_bwd_dq_spec_t* out,
                                               rocke_attn_bwd_err_t* err);

/* build_attn_bwd_dq(spec, arch=arch) (_emit_dq with sentinel=True) into b,
 * which this initialises with the validated spec's kernel_name(). spec must
 * have passed rocke_attn_bwd_dq_spec_init. A spec the validator refuses is
 * ROCKE_ERR_VALUE (b initialised with a placeholder name). NULL and the
 * builder's sticky error on any failure. */
rocke_kernel_def_t* rocke_build_attn_bwd_dq_new(rocke_ir_builder_t* b,
                                                const rocke_attn_bwd_dq_spec_t* spec,
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
/* AttnBwdDqSpec.non_default_fields(): (name, repr(value)) of every field
 * (head_size and name skipped) that differs from its default, in field
 * order. */
std::vector<std::pair<std::string, std::string>>
    dq_non_default_fields(const rocke_attn_bwd_dq_spec_t& spec);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_BWD_DQ_H */
