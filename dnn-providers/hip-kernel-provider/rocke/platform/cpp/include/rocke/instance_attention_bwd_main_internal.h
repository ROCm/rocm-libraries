/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd_main_internal.h -- C++-only internals of the
 * attention backward main-kernel builder (attention_bwd_main.cpp) shared with
 * the two shells that drive its _kv_tile, as their Python modules import
 * _kv_tile from attention_bwd.py: the dense backward shell
 * (_attention_dense_bwd_shell.cpp) and the unified (paged) main kernel
 * (attention_bwd_unified.cpp). Holds the builder plumbing
 * (main_kernel::plumbing; the dense shell keeps its own), the kernarg map,
 * the per-kernel state, the per-CTA objects and _kv_tile itself.
 *
 *   Python (attention_bwd.py)           C++ (this header)
 *   ---------------------------------   ---------------------------------------
 *   the kernarg dict `a`                main_kernel::Kargs
 *   _emit_main's locals                 main_kernel::Main
 *   _Cta / _cta_setup's objects         main_kernel::Cta
 *   _kv_tile(..., epilogue=, ring=,     main_kernel::kv_tile
 *     tag=)
 *
 * _kv_tile's keyword arguments other than kt=, epilogue= and tag= (step,
 * mask, hk .. n_h, seq_q, seq_kv, q_addr .. v_addr, tid, ring) are the Cta
 * fields. Its duck-typed parameters:
 *   * mask.static_runs / the interior_q_range shortcuts of a policy
 *     classified at build time (the dense shell's _StaticClassMixin):
 *     Cta::has_static with Cta::any_edge / Cta::any_interior;
 *   * q_addr .. v_addr, whose slab(h, row0, clamp=) need not be a
 *     StridedTensorAddr's: Cta::slab_fn when the slab is still a RowSlab (the
 *     dense shell's DenseFixedAddr, maybe a buffer slab), Cta::kv_rows_fn when
 *     the K / V slab is only a row source with load_vec (the unified family's
 *     PagedCacheRows, whose PagedSlab step.stage_kv reads); both empty: the
 *     StridedTensorAddr fields q_addr .. v_addr;
 *   * epilogue=: the kv_tile argument `epilogue` (NULL is None).
 * The dense shell also passes compile-time strides in its `shared` dict:
 * Kargs::ints holds them (Kargs::iv reads either).
 *
 * Not a public API: every name lives in rocke_attn_bwd::main_kernel and may
 * change with the builders.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H

#ifndef __cplusplus
#error "rocke/instance_attention_bwd_main_internal.h is C++ only"
#endif

#include <stdint.h>
#include <string.h>

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
#include "rocke/ir_internal.h"

namespace rocke_attn_bwd
{
namespace main_kernel
{

typedef rocke_ir_builder_t B;
typedef rocke_value_t* V;
typedef rocke_attn_bwd_iv_t IV;

/* ------------------------------------------------------------- plumbing */

/* The builders' helper-status plumbing, in a namespace of its own so a shell
 * with its own (the dense shell's) can still use main_kernel. */
namespace plumbing
{

inline bool live(const B* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

[[noreturn]] inline void raise_builder(B* b)
{
    if(b != NULL && b->status != ROCKE_OK)
        ckc::raise_status(b->status, b->err);
    throw ckc::ValueError("attention backward main kernel: a helper failed");
}

inline void need(B* b, bool ok)
{
    if(!ok || !live(b))
        raise_builder(b);
}

inline V needv(B* b, V v)
{
    if(v == NULL || !live(b))
        raise_builder(b);
    return v;
}

inline void need_st(B* b, rocke_status_t st)
{
    if(st != ROCKE_OK || !live(b))
        raise_builder(b);
}

inline void need_err(rocke_status_t st, const rocke_attn_bwd_err_t& e)
{
    if(st != ROCKE_OK)
        ckc::raise_status(e.code, e.msg);
}

inline bool eq(const char* a, const char* b)
{
    if(a == NULL || b == NULL)
        return a == b;
    return strcmp(a, b) == 0;
}

inline IV ivv(V v)
{
    return rocke_attn_bwd_iv_val(v);
}

inline V ci(B* b, int64_t x)
{
    return rocke_b_const_i32(b, x);
}

/* b.sync() under the family's barrier fence (BWD_BARRIER_FENCE). */
inline void sync(B* b)
{
    rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE);
    need(b, true);
}

/* An scf_if region: enter on construction, leave on destruction. */
struct IfScope
{
    B* b;
    IfScope(B* bb, V cond)
        : b(bb)
    {
        rocke_if_t f = rocke_b_scf_if(b, cond);
        if(f.op == NULL)
            raise_builder(b);
        rocke_b_region_enter(b, f.then_region);
    }
    ~IfScope()
    {
        rocke_b_region_leave(b);
    }
    IfScope(const IfScope&) = delete;
    IfScope& operator=(const IfScope&) = delete;
};

} // namespace plumbing

/* ---------------------------------------------------------------- kernargs */

/* The kernargs in declaration order; a static unit multiplier is NULL.
 * `ints` holds the build-time ints of the dense shell's `shared` dict (its
 * compile-time strides); the other kernels leave it empty. */
struct Kargs
{
    std::vector<std::string> order;
    std::map<std::string, V> m;
    std::map<std::string, int64_t> ints;
    V operator[](const std::string& k) const
    {
        auto it = m.find(k);
        if(it == m.end())
            throw ckc::KeyError("'" + k + "'");
        return it->second;
    }
    /* a[k] as an IntOrValue (an int of `ints`, else the value). */
    IV iv(const std::string& k) const
    {
        auto it = ints.find(k);
        if(it != ints.end())
            return rocke_attn_bwd_iv_int(it->second);
        return rocke_attn_bwd_iv_val((*this)[k]);
    }
};

/* ------------------------------------------------------------- the kernel */

/* The kernel-wide state of _emit_main (and of the dense and unified shells;
 * s is the unified kernel's general_spec()). */
struct Main
{
    B* b;
    rocke_attn_bwd_spec_t s;
    std::string arch;
    rocke_attn_bwd_arch_facts_t facts;
    rocke_attn_bwd_tile_geometry_t geom;
    rocke_attn_bwd_lds_plan_t plan;
    const rocke_type_t* io;
    int seq_guards;
    bool sentinel;
};

inline int threads_of(const Main& m)
{
    return m.geom.waves * m.facts.wave_size;
}

/* Tensors of Cta::slab_fn / Cta::kv_rows_fn. */
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
    B* b = NULL;
    rocke_attn_bwd_cta_work_t work;
    V ids[3];
    rocke_attn_bwd_seq_info_t seq_q, seq_kv;
    V hk, hv, h0, n_h;
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
    V tid;
    /* x_addr.slab(h, row0, clamp=) of tensor `which` (SLAB_*) when it is a
     * RowSlab; empty: the StridedTensorAddr fields above. */
    std::function<rocke_attn_bwd_row_slab_t(int which, IV h, IV row0, bool clamp)> slab_fn;
    /* k_addr.slab / v_addr.slab (which: SLAB_K, SLAB_V) when the slab is only
     * a row source (load_vec); step.stage_kv then reads it as one. The source's
     * ctx must stay valid until the next call for the same tensor. Takes
     * precedence over slab_fn for K / V. */
    std::function<rocke_attn_bwd_row_source_t(int which, IV h, IV row0, bool clamp)> kv_rows_fn;
    /* A mask policy classified at build time (the dense shell's
     * _StaticClassMixin): static_runs(edge) and the interior_q_range
     * shortcuts. */
    bool has_static = false;
    bool any_edge = true;
    bool any_interior = true;
    Cta() = default;
    Cta(const Cta&) = delete;
    Cta& operator=(const Cta&) = delete;
    ~Cta()
    {
        if(has_step)
            rocke_attn_bwd_tile_step_free(&step);
        if(has_sink)
            rocke_attn_bwd_dq_sink_free(&sink);
    }
};

/* _kv_tile's epilogue(b, step, acc, k0=, live=): replaces the strided dK / dV
 * epilogue of dkv_mode. */
typedef std::function<void(const rocke_attn_bwd_dkdv_acc_t& acc, V k0, V live)> KvEpilogue;

/* _kv_tile(b, m.s, a, c.step, c.mask, kt=, hk=c.hk, hv=c.hv, h0=c.h0,
 * n_h=c.n_h, seq_q=, seq_kv=, q_addr=, do_addr=, k_addr=, v_addr=, io=,
 * tid=c.tid, epilogue=, ring=, tag=) over the CTA objects of c (c.has_ring:
 * ring); epilogue NULL is None. Throws ckc::Error like the rest of the
 * builder (callers run it under ckc::guard_builder). */
void kv_tile(
    Main& m, const Kargs& a, Cta& c, V kt, const std::string& tag, const KvEpilogue* epilogue);

} // namespace main_kernel
} // namespace rocke_attn_bwd

#endif /* ROCKE_INSTANCE_ATTENTION_BWD_MAIN_INTERNAL_H */
