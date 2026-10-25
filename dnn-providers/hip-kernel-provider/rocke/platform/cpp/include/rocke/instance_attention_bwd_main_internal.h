/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd_main_internal.h -- C++-only internals of the
 * backward main-kernel builder (attention_bwd_main.cpp) shared with the dense
 * backward shell (_attention_dense_bwd_shell.cpp): the kernarg map, the
 * kernel and per-CTA state, and _kv_tile (attention_bwd.py) with its two
 * dense-shell hooks.
 *
 *   Python (attention_bwd.py)           C++ (this header)
 *   ---------------------------------   ---------------------------------------
 *   the kernarg dict `a`                main_kernel::Kargs
 *   _emit_main's locals                 main_kernel::Main
 *   _Cta / _cta_setup's objects         main_kernel::Cta
 *   _kv_tile(..., epilogue=, ring=,     main_kernel::kv_tile
 *     tag=)
 *
 * Dense-shell hooks of _kv_tile:
 *   * mask.static_runs / the dense policies' interior_q_range shortcuts
 *     (_attention_dense_bwd_shell._StaticClassMixin): Cta::has_static with
 *     Cta::any_edge / Cta::any_interior;
 *   * epilogue=: the kv_tile argument `epilogue` (NULL is None);
 *   * the dense addressing (DenseFixedAddr, whose slab may be a buffer slab):
 *     Cta::slab_fn (empty: the StridedTensorAddr fields q_addr .. v_addr).
 * The dense shell also passes compile-time strides in its `shared` dict:
 * Kargs::ints holds them (Kargs::iv reads either).
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H

#ifdef __cplusplus

#include <stdint.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_body.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_ring.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_sinks.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_worklist.h"
#include "rocke/instance_attention_bwd.h"

namespace rocke_attn_bwd
{
namespace main_kernel
{

/* The kernargs in declaration order; a static unit multiplier is NULL.
 * `ints` holds the build-time ints of the dense shell's `shared` dict (its
 * compile-time strides); the main kernel leaves it empty. */
struct Kargs
{
    std::vector<std::string> order;
    std::map<std::string, rocke_value_t*> m;
    std::map<std::string, int64_t> ints;
    rocke_value_t* operator[](const std::string& k) const
    {
        auto it = m.find(k);
        if(it == m.end())
            throw ckc::KeyError("'" + k + "'");
        return it->second;
    }
    /* a[k] as an IntOrValue (an int of `ints`, else the value). */
    rocke_attn_bwd_iv_t iv(const std::string& k) const
    {
        auto it = ints.find(k);
        if(it != ints.end())
            return rocke_attn_bwd_iv_int(it->second);
        return rocke_attn_bwd_iv_val((*this)[k]);
    }
};

/* The kernel-wide state of _emit_main (and of the dense shell). */
struct Main
{
    rocke_ir_builder_t* b;
    rocke_attn_bwd_spec_t s;
    std::string arch;
    rocke_attn_bwd_arch_facts_t facts;
    rocke_attn_bwd_tile_geometry_t geom;
    rocke_attn_bwd_lds_plan_t plan;
    const rocke_type_t* io;
    int seq_guards;
    bool sentinel;
};

/* Tensors of Cta::slab_fn. */
enum
{
    SLAB_Q = 0,
    SLAB_DO = 1,
    SLAB_K = 2,
    SLAB_V = 3
};

/* The decoded work and per-CTA objects of one kv-tile pass (_Cta and the
 * objects _cta_setup builds, owned here so their addresses are stable). */
struct Cta
{
    rocke_ir_builder_t* b = NULL;
    rocke_attn_bwd_cta_work_t work;
    rocke_value_t* ids[3];
    rocke_attn_bwd_seq_info_t seq_q, seq_kv;
    rocke_value_t *hk, *hv, *h0, *n_h;
    rocke_attn_bwd_strided_tensor_addr_t q_addr, do_addr, k_addr, v_addr;
    rocke_attn_bwd_workspace_addr_t ws_lse2, ws_dsum, ws_dq;
    rocke_attn_bwd_stats_view_t stats;
    rocke_attn_bwd_dq_sink_t sink;
    bool has_sink = false;
    rocke_attn_bwd_mask_policy_t mask;
    rocke_attn_bwd_bwd_lds_t* lds = NULL;
    rocke_attn_bwd_tile_step_t step;
    bool has_step = false;
    rocke_attn_bwd_q_ring_t ring;
    bool has_ring = false;
    rocke_value_t* tid;
    /* x_addr.slab(h, row0, clamp=) of tensor `which` (SLAB_*); empty: the
     * StridedTensorAddr fields above. */
    std::function<rocke_attn_bwd_row_slab_t(
        int which, rocke_attn_bwd_iv_t h, rocke_attn_bwd_iv_t row0, bool clamp)>
        slab_fn;
    /* A mask policy classified at build time (the dense shell's
     * _StaticClassMixin): static_runs(edge) and the interior_q_range
     * shortcuts. */
    bool has_static = false;
    bool any_edge = true;
    bool any_interior = true;
    ~Cta()
    {
        if(has_step)
            rocke_attn_bwd_tile_step_free(&step);
        if(has_sink)
            rocke_attn_bwd_dq_sink_free(&sink);
    }
};

/* _kv_tile's epilogue(b, step, acc, k0=, live=). */
typedef std::function<void(
    const rocke_attn_bwd_dkdv_acc_t& acc, rocke_value_t* k0, rocke_value_t* live)>
    KvEpilogue;

/* _kv_tile(b, s, a, step, mask, kt=, ..., epilogue=, ring=, tag=) over the
 * CTA objects of c (c.has_ring: ring); epilogue NULL is None. Throws
 * ckc::Error like the rest of the builder (callers run it under
 * ckc::guard_builder). */
void kv_tile(Main& m,
             const Kargs& a,
             Cta& c,
             rocke_value_t* kt,
             const std::string& tag,
             const KvEpilogue* epilogue);

} // namespace main_kernel
} // namespace rocke_attn_bwd

#endif /* __cplusplus */

#endif /* ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H */
