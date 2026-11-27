// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * batchnorm2d_ro.cpp -- C99 port of rocke/instances/common/batchnorm2d_ro.py
 * (read-once NCHW BatchNorm training forward / backward).
 *
 * Byte-faithful reproduction of the Python builder-call sequence: every
 * nested Python call is unrolled into its own statement in Python's
 * left-to-right evaluation order (C++ leaves argument evaluation order
 * unspecified, and the order fixes the SSA numbering).
 */

#include "rocke/instance_batchnorm2d_ro.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define RO_MAX_REGS 64
#define RO_MAX_VEC 8

rocke_batchnorm2d_ro_spec_t rocke_batchnorm2d_ro_spec_default(void)
{
    rocke_batchnorm2d_ro_spec_t s;
    s.block_size = 256;
    s.vec = 8;
    s.regs = 16;
    s.dtype = "f16";
    s.fast_reduce = false;
    s.save_mean_invstd = true;
    s.wave_size = 64;
    s.name = "rocke_batchnorm2d_ro";
    return s;
}

int rocke_batchnorm2d_ro_cap_elems(const rocke_batchnorm2d_ro_spec_t* spec)
{
    return spec ? spec->block_size * spec->vec * spec->regs : 0;
}

/* kernel_name_join(name, kind, dtype, f"b{block_size}", f"v{vec}",
 *                  f"r{regs}", flags={"fr": fast_reduce}) */
rocke_status_t rocke_batchnorm2d_ro_kernel_name(const rocke_batchnorm2d_ro_spec_t* spec,
                                                const char* kind,
                                                char* out,
                                                size_t out_cap)
{
    char pb[32], pv[32], pr[32];
    const char* parts[5];
    const char* flag_names[1] = {"fr"};
    int flag_on[1];
    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    snprintf(pb, sizeof pb, "b%d", spec->block_size);
    snprintf(pv, sizeof pv, "v%d", spec->vec);
    snprintf(pr, sizeof pr, "r%d", spec->regs);
    parts[0] = kind ? kind : "train";
    parts[1] = spec->dtype;
    parts[2] = pb;
    parts[3] = pv;
    parts[4] = pr;
    flag_on[0] = spec->fast_reduce ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap, NULL);
}

static bool ro_reject(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
    return false;
}

bool rocke_batchnorm2d_is_valid_ro_spec(const rocke_batchnorm2d_ro_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    char msg[64];
    bool f32;
    (void)arch;
    if(reason != NULL && reason_cap > 0)
        reason[0] = '\0';
    if(spec == NULL)
        return ro_reject(reason, reason_cap, "null spec");
    if(spec->dtype == NULL
       || (strcmp(spec->dtype, "f16") && strcmp(spec->dtype, "bf16") && strcmp(spec->dtype, "f32")))
        return ro_reject(reason, reason_cap, "unsupported dtype");
    f32 = strcmp(spec->dtype, "f32") == 0;
    if(f32 ? (spec->vec != 2 && spec->vec != 4) : (spec->vec != 4 && spec->vec != 8))
    {
        snprintf(msg, sizeof msg, "vec %d not in %s", spec->vec, f32 ? "(2, 4)" : "(4, 8)");
        return ro_reject(reason, reason_cap, msg);
    }
    if(spec->wave_size <= 0 || spec->block_size % spec->wave_size || spec->block_size > 1024
       || spec->block_size < 64)
        return ro_reject(
            reason, reason_cap, "block_size must be a multiple of the wave size, <= 1024");
    if(spec->regs < 1 || spec->regs > RO_MAX_REGS)
        return ro_reject(reason, reason_cap, "regs out of range");
    return true;
}

static long long ro_cdiv(long long a, long long b)
{
    return (a + b - 1) / b;
}

bool rocke_batchnorm2d_ro_plan(const rocke_batchnorm2d_ro_spec_t* spec,
                               int n,
                               int c,
                               int h,
                               int w,
                               int sn_req,
                               int sh_req,
                               int max_resident,
                               rocke_batchnorm2d_ro_plan_t* out)
{
    long long hw = (long long)h * w;
    long long hw_chunk, sh, n_chunk, sn, splits, groups;
    if(spec == NULL || out == NULL)
        return false;
    if(n <= 0 || c <= 0 || hw <= 0 || sn_req < 1 || sh_req < 1)
        return false;
    if(hw % spec->vec || (long long)n * c * hw >= (1LL << 31))
        return false;
    hw_chunk = ro_cdiv(hw, sh_req);
    hw_chunk = ro_cdiv(hw_chunk, spec->vec) * spec->vec;
    sh = ro_cdiv(hw, hw_chunk);
    n_chunk = ro_cdiv(n, sn_req);
    sn = ro_cdiv(n, n_chunk);
    splits = sn * sh;
    if(splits > ROCKE_BATCHNORM2D_RO_MAX_SPLITS || splits > spec->block_size)
        return false;
    if(n_chunk * hw_chunk > rocke_batchnorm2d_ro_cap_elems(spec))
        return false;
    /* Python floor division; max_resident is a non-negative count. */
    groups = (long long)max_resident / splits;
    if(groups > c)
        groups = c;
    if(groups < 1)
        return false;
    out->n_chunk = (int)n_chunk;
    out->hw_chunk = (int)hw_chunk;
    out->sn = (int)sn;
    out->sh = (int)sh;
    out->grid = (int)(groups * splits);
    return true;
}

long long rocke_batchnorm2d_ro_sync_ints(int c)
{
    return (long long)c * ROCKE_BATCHNORM2D_RO_STRIDE;
}

/* ------------------------------------------------------------------ *
 * shared pieces
 * ------------------------------------------------------------------ */

static rocke_value_t*
    ro_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

static rocke_value_t* ro_param(rocke_ir_builder_t* b,
                               const char* name,
                               const rocke_type_t* elem,
                               bool noalias,
                               bool readonly,
                               bool writeonly,
                               int align)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    if(noalias)
    {
        o.noalias = true;
        o.noalias_set = true;
    }
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    if(writeonly)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    if(align > 0)
    {
        o.align = align;
        o.align_set = true;
    }
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

static rocke_value_t* ro_lds(rocke_ir_builder_t* b, int n, const char* hint)
{
    rocke_tensor_view_t v;
    int shape[1] = {n};
    if(rocke_make_lds_view(b, &v, rocke_f32(), shape, 1, hint, NULL) != ROCKE_OK)
        return NULL;
    return v.base;
}

/* Python _ro_item: decode a work item. */
typedef struct ro_item
{
    rocke_value_t *ch, *sub, *S, *hw_lo, *hw_len, *n_lo, *n_cnt, *total;
} ro_item_t;

static ro_item_t ro_item(rocke_ir_builder_t* b,
                         rocke_value_t* p_c,
                         rocke_value_t* p_n,
                         rocke_value_t* HW,
                         rocke_value_t* p_nch,
                         rocke_value_t* p_hwch,
                         rocke_value_t* p_sh,
                         rocke_value_t* p_sn,
                         rocke_value_t* w_id)
{
    ro_item_t it;
    rocke_value_t *t, *c1;
    (void)p_c;
    (void)rocke_b_const_i32(b, 0); /* Python: zero = b.const_i32(0) (unused) */
    it.S = rocke_b_mul(b, p_sh, p_sn);
    it.ch = rocke_b_div(b, w_id, it.S);
    t = rocke_b_mul(b, it.ch, it.S);
    it.sub = rocke_b_sub(b, w_id, t);
    rocke_value_t* sn_id = rocke_b_div(b, it.sub, p_sh);
    t = rocke_b_mul(b, sn_id, p_sh);
    rocke_value_t* sh_id = rocke_b_sub(b, it.sub, t);
    it.hw_lo = rocke_b_mul(b, sh_id, p_hwch);
    t = rocke_b_sub(b, HW, it.hw_lo);
    t = rocke_b_smin(b, t, p_hwch);
    c1 = rocke_b_const_i32(b, 1);
    it.hw_len = rocke_b_smax(b, t, c1);
    it.n_lo = rocke_b_mul(b, sn_id, p_nch);
    t = rocke_b_sub(b, p_n, it.n_lo);
    t = rocke_b_smin(b, t, p_nch);
    c1 = rocke_b_const_i32(b, 1);
    it.n_cnt = rocke_b_smax(b, t, c1);
    it.total = rocke_b_mul(b, it.n_cnt, it.hw_len);
    return it;
}

/* Python _ro_tile_index: (valid, flat) of vector j of this thread. */
static void ro_tile_index(rocke_ir_builder_t* b,
                          const rocke_batchnorm2d_ro_spec_t* spec,
                          rocke_value_t* tid,
                          int j,
                          rocke_value_t* p_c,
                          rocke_value_t* HW,
                          const ro_item_t* it,
                          rocke_value_t** ok_out,
                          rocke_value_t** flat_out)
{
    rocke_value_t* c = rocke_b_const_i32(b, j * spec->block_size);
    rocke_value_t* t = rocke_b_add(b, c, tid);
    c = rocke_b_const_i32(b, spec->vec);
    rocke_value_t* k = rocke_b_mul(b, t, c);
    rocke_value_t* ok = rocke_b_cmp_lt(b, k, it->total);
    c = rocke_b_const_i32(b, 0);
    rocke_value_t* kk = rocke_b_select(b, ok, k, c);
    rocke_value_t* n_rel = rocke_b_div(b, kk, it->hw_len);
    t = rocke_b_mul(b, n_rel, it->hw_len);
    rocke_value_t* hw_rel = rocke_b_sub(b, kk, t);
    t = rocke_b_add(b, it->n_lo, n_rel);
    t = rocke_b_mul(b, t, p_c);
    rocke_value_t* row = rocke_b_add(b, t, it->ch);
    t = rocke_b_mul(b, row, HW);
    t = rocke_b_add(b, t, it->hw_lo);
    *flat_out = rocke_b_add(b, t, hw_rel);
    *ok_out = ok;
}

/* Python _ro_block_pair. */
static void ro_block_pair(rocke_ir_builder_t* b,
                          const rocke_batchnorm2d_ro_spec_t* spec,
                          rocke_value_t* v1,
                          rocke_value_t* v2,
                          rocke_value_t* lds_a,
                          rocke_value_t* lds_c,
                          rocke_value_t* tid,
                          rocke_value_t** t1,
                          rocke_value_t** t2)
{
    if(spec->fast_reduce)
    {
        *t1 = rocke_block_lds_reduce_with_wave_prologue(
            b, v1, lds_a, tid, spec->block_size, ROCKE_REDUCE_SUM, spec->wave_size);
        *t2 = rocke_block_lds_reduce_with_wave_prologue(
            b, v2, lds_c, tid, spec->block_size, ROCKE_REDUCE_SUM, spec->wave_size);
        return;
    }
    (void)rocke_block_lds_reduce_pair(
        b, v1, v2, lds_a, lds_c, tid, spec->block_size, ROCKE_REDUCE_SUM, ROCKE_REDUCE_SUM, t1, t2);
}

/* Python _ro_exchange: add the block totals to the channel accumulators, wait
 * for the other S - 1 items, return the channel totals on every thread. */
static void ro_exchange(rocke_ir_builder_t* b,
                        rocke_value_t* SyncI,
                        rocke_value_t* SyncF,
                        rocke_value_t* Err,
                        rocke_value_t* tid,
                        rocke_value_t* lds_bc,
                        rocke_value_t* ch,
                        rocke_value_t* sub,
                        rocke_value_t* S,
                        rocke_value_t* p_epoch,
                        rocke_value_t* p_spin,
                        rocke_value_t* t1,
                        rocke_value_t* t2,
                        rocke_value_t** s1,
                        rocke_value_t** s2)
{
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* c = rocke_b_const_i32(b, ROCKE_BATCHNORM2D_RO_STRIDE);
    rocke_value_t* rec = rocke_b_mul(b, ch, c);
    rocke_value_t* parity = rocke_b_land(b, p_epoch, one);
    rocke_value_t* t = rocke_b_add(b, rec, one);
    c = rocke_b_const_i32(b, 2);
    rocke_value_t* u = rocke_b_mul(b, parity, c);
    rocke_value_t* acc = rocke_b_add(b, t, u);
    t = rocke_b_add(b, rec, one);
    u = rocke_b_sub(b, one, parity);
    c = rocke_b_const_i32(b, 2);
    u = rocke_b_mul(b, u, c);
    rocke_value_t* other = rocke_b_add(b, t, u);
    t = rocke_b_add(b, p_epoch, one);
    rocke_value_t* target = rocke_b_mul(b, t, S);
    {
        rocke_value_t* cond = rocke_b_cmp_eq(b, tid, zero);
        rocke_if_t iff = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_value_t* cond2 = rocke_b_cmp_eq(b, sub, zero);
            rocke_if_t iff2 = rocke_b_scf_if(b, cond2);
            rocke_b_region_enter(b, iff2.then_region);
            rocke_value_t* f = rocke_b_const_f32(b, 0.0);
            rocke_b_global_store(b, SyncF, other, f, 1);
            rocke_value_t* o1 = rocke_b_add(b, other, one);
            f = rocke_b_const_f32(b, 0.0);
            rocke_b_global_store(b, SyncF, o1, f, 1);
            rocke_b_region_leave(b);
        }
        (void)rocke_b_global_atomic_add_scoped(b, SyncF, acc, t1, "monotonic", "agent");
        rocke_value_t* acc1 = rocke_b_add(b, acc, one);
        (void)rocke_b_global_atomic_add_scoped(b, SyncF, acc1, t2, "monotonic", "agent");
        rocke_b_s_waitcnt(b, 0, -1, -1);
        (void)rocke_b_global_atomic_add_scoped(b, SyncI, rec, one, "monotonic", "agent");
        rocke_value_t* seen = rocke_b_global_flag_wait_ge_bounded(
            b, SyncI, rec, target, p_spin, "agent", "monotonic");
        {
            rocke_value_t* late = rocke_b_cmp_lt(b, seen, target);
            rocke_if_t iff3 = rocke_b_scf_if(b, late);
            rocke_b_region_enter(b, iff3.then_region);
            (void)rocke_b_global_atomic_add_scoped(b, Err, zero, one, "monotonic", "agent");
            rocke_b_region_leave(b);
        }
        rocke_value_t* f0 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* a1
            = rocke_b_global_atomic_add_scoped(b, SyncF, acc, f0, "monotonic", "agent");
        acc1 = rocke_b_add(b, acc, one);
        rocke_value_t* a2
            = rocke_b_global_atomic_add_scoped(b, SyncF, acc1, f0, "monotonic", "agent");
        rocke_b_smem_store_vN_f32(b, lds_bc, &zero, 1, a1, 1);
        rocke_b_smem_store_vN_f32(b, lds_bc, &one, 1, a2, 1);
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    rocke_value_t* v = rocke_b_smem_load_vN_f32(b, lds_bc, &zero, 1, 1);
    *s1 = rocke_b_vec_extract(b, v, 0);
    v = rocke_b_smem_load_vN_f32(b, lds_bc, &one, 1, 1);
    *s2 = rocke_b_vec_extract(b, v, 0);
}

/* Python _bwd_coefs (batchnorm2d.py): dx = a * dy - k2 * (x - mean) - k3. */
static void ro_bwd_coefs(rocke_ir_builder_t* b,
                         rocke_value_t* s1,
                         rocke_value_t* s2,
                         rocke_value_t* scale,
                         rocke_value_t* invstd,
                         rocke_value_t* m_f,
                         rocke_value_t** a,
                         rocke_value_t** k2,
                         rocke_value_t** k3)
{
    rocke_value_t* one = rocke_b_const_f32(b, 1.0);
    rocke_value_t* inv_m = rocke_b_fdiv(b, one, m_f);
    *a = rocke_b_fmul(b, scale, invstd);
    rocke_value_t* t1 = rocke_b_fmul(b, *a, invstd);
    rocke_value_t* t2 = rocke_b_fmul(b, s2, inv_m);
    *k2 = rocke_b_fmul(b, t1, t2);
    rocke_value_t* t3 = rocke_b_fmul(b, s1, inv_m);
    *k3 = rocke_b_fmul(b, *a, t3);
}

/* Shared geometry params / prologue of both kernels. */
typedef struct ro_common
{
    rocke_value_t *p_n, *p_c, *p_h, *p_w;
    rocke_value_t *p_nch, *p_hwch, *p_sh, *p_sn, *p_grid, *p_epoch, *p_spin;
} ro_common_t;

static void ro_common_params(rocke_ir_builder_t* b, ro_common_t* P)
{
    P->p_n = rocke_b_param(b, "N", rocke_i32(), NULL);
    P->p_c = rocke_b_param(b, "C", rocke_i32(), NULL);
    P->p_h = rocke_b_param(b, "H", rocke_i32(), NULL);
    P->p_w = rocke_b_param(b, "W", rocke_i32(), NULL);
}

static void ro_geom_params(rocke_ir_builder_t* b, ro_common_t* P)
{
    P->p_nch = rocke_b_param(b, "NChunk", rocke_i32(), NULL);
    P->p_hwch = rocke_b_param(b, "HwChunk", rocke_i32(), NULL);
    P->p_sh = rocke_b_param(b, "SplitHw", rocke_i32(), NULL);
    P->p_sn = rocke_b_param(b, "SplitN", rocke_i32(), NULL);
    P->p_grid = rocke_b_param(b, "Grid", rocke_i32(), NULL);
    P->p_epoch = rocke_b_param(b, "Epoch", rocke_i32(), NULL);
    P->p_spin = rocke_b_param(b, "MaxSpin", rocke_i32(), NULL);
}

static bool ro_check(rocke_ir_builder_t* b, const rocke_batchnorm2d_ro_spec_t* spec)
{
    char why[128];
    if(!rocke_batchnorm2d_is_valid_ro_spec(spec, NULL, why, sizeof why))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid read-once batchnorm spec: %s", why);
        return false;
    }
    return true;
}

/* The persistent-loop prologue shared by both kernels (after the params). */
typedef struct ro_loop
{
    rocke_value_t *HW, *tid, *cta, *lds_a, *lds_c, *lds_bc, *S, *total_items, *rounds, *m_f;
} ro_loop_t;

static void ro_prologue(rocke_ir_builder_t* b,
                        const rocke_batchnorm2d_ro_spec_t* spec,
                        const ro_common_t* P,
                        ro_loop_t* L)
{
    L->HW = rocke_b_mul(b, P->p_h, P->p_w);
    L->tid = rocke_b_thread_id_x(b);
    L->cta = rocke_b_block_id_x(b);
    L->lds_a = ro_lds(b, spec->block_size, "lds_s1");
    L->lds_c = ro_lds(b, spec->block_size, "lds_s2");
    L->lds_bc = ro_lds(b, 2, "lds_bc");
    L->S = rocke_b_mul(b, P->p_sh, P->p_sn);
    L->total_items = rocke_b_mul(b, P->p_c, L->S);
    rocke_value_t* t = rocke_b_add(b, L->total_items, P->p_grid);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    t = rocke_b_sub(b, t, c1);
    L->rounds = rocke_b_div(b, t, P->p_grid);
    t = rocke_b_mul(b, P->p_n, L->HW);
    L->m_f = rocke_b_sitofp_f32(b, t);
}

/* Opens the persistent loop and the live-item branch; returns w_id. The
 * caller emits the item body, then calls ro_loop_close. */
static rocke_value_t* ro_loop_open(rocke_ir_builder_t* b,
                                   const ro_common_t* P,
                                   const ro_loop_t* L,
                                   rocke_if_else_t* branch)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_for_t loop = rocke_b_scf_for(b, c0, L->rounds, c1, "ro_round");
    rocke_b_region_enter(b, loop.body);
    rocke_value_t* t = rocke_b_mul(b, loop.iv, P->p_grid);
    rocke_value_t* w_id = rocke_b_add(b, t, L->cta);
    rocke_value_t* live = rocke_b_cmp_lt(b, w_id, L->total_items);
    *branch = rocke_b_scf_if_else(b, live);
    rocke_b_region_enter(b, branch->then_region);
    return w_id;
}

static void ro_loop_close(rocke_ir_builder_t* b, const rocke_if_else_t* branch)
{
    rocke_b_region_leave(b); /* then */
    rocke_b_region_enter(b, branch->else_region);
    rocke_b_region_leave(b); /* else: pass */
    rocke_b_region_leave(b); /* loop body */
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_ro_train
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_ro_train(rocke_ir_builder_t* b,
                                                     const rocke_batchnorm2d_ro_spec_t* spec)
{
    if(b == NULL || spec == NULL || !ro_check(b, spec))
        return NULL;
    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    const int VEC = spec->vec, R = spec->regs;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    rocke_value_t* X = ro_param(b, "X", io_ty, true, true, false, 16);
    rocke_value_t* Gamma = ro_param(b, "Gamma", io_ty, true, true, false, 16);
    rocke_value_t* Beta = ro_param(b, "Beta", io_ty, true, true, false, 16);
    rocke_value_t* Y = ro_param(b, "Y", io_ty, true, false, true, 16);
    rocke_value_t* RunningMean = ro_param(b, "RunningMean", rocke_f32(), true, false, false, 0);
    rocke_value_t* RunningVar = ro_param(b, "RunningVar", rocke_f32(), true, false, false, 0);
    rocke_value_t* Mean = NULL;
    rocke_value_t* InvStd = NULL;
    if(spec->save_mean_invstd)
    {
        Mean = ro_param(b, "Mean", io_ty, true, false, true, 0);
        InvStd = ro_param(b, "InvStd", io_ty, true, false, true, 0);
    }
    rocke_value_t* SyncI
        = rocke_b_param(b, "SyncI", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* SyncF
        = rocke_b_param(b, "SyncF", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* Err = ro_param(b, "Err", rocke_i32(), true, false, false, 0);
    ro_common_t P;
    ro_common_params(b, &P);
    rocke_value_t* eps = rocke_b_param(b, "eps", rocke_f32(), NULL);
    rocke_value_t* momentum = rocke_b_param(b, "momentum", rocke_f32(), NULL);
    ro_geom_params(b, &P);

    ro_loop_t L;
    ro_prologue(b, spec, &P, &L);
    rocke_if_else_t branch;
    rocke_value_t* w_id = ro_loop_open(b, &P, &L, &branch);
    {
        ro_item_t it = ro_item(b, P.p_c, P.p_n, L.HW, P.p_nch, P.p_hwch, P.p_sh, P.p_sn, w_id);
        rocke_value_t* t = rocke_b_mul(b, it.ch, L.HW);
        rocke_value_t* shift = rocke_b_load_scalar_as_f32(b, X, t, spec->dtype);
        rocke_value_t* raws[RO_MAX_REGS];
        rocke_value_t* oks[RO_MAX_REGS];
        rocke_value_t* flats[RO_MAX_REGS];
        for(int j = 0; j < R; ++j)
        {
            ro_tile_index(b, spec, L.tid, j, P.p_c, L.HW, &it, &oks[j], &flats[j]);
            raws[j] = rocke_b_global_load_vN(b, X, flats[j], io_ty, VEC, 0);
        }
        rocke_value_t* s1 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* s2 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
        for(int j = 0; j < R; ++j)
        {
            rocke_value_t* xs[RO_MAX_VEC];
            rocke_value_t* ds[RO_MAX_VEC];
            rocke_value_t* sq[RO_MAX_VEC];
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, raws[j], i);
                xs[i] = rocke_b_cast_to_f32(b, e);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* d = rocke_b_fsub(b, xs[i], shift);
                ds[i] = rocke_b_select(b, oks[j], d, zero_f);
            }
            rocke_value_t* r = rocke_tree_reduce(b, ro_fadd_combine, NULL, ds, VEC);
            s1 = rocke_b_fadd(b, s1, r);
            for(int i = 0; i < VEC; ++i)
                sq[i] = rocke_b_fmul(b, ds[i], ds[i]);
            r = rocke_tree_reduce(b, ro_fadd_combine, NULL, sq, VEC);
            s2 = rocke_b_fadd(b, s2, r);
        }
        rocke_value_t *t1 = NULL, *t2 = NULL, *a1 = NULL, *a2 = NULL;
        ro_block_pair(b, spec, s1, s2, L.lds_a, L.lds_c, L.tid, &t1, &t2);
        ro_exchange(b,
                    SyncI,
                    SyncF,
                    Err,
                    L.tid,
                    L.lds_bc,
                    it.ch,
                    it.sub,
                    L.S,
                    P.p_epoch,
                    P.p_spin,
                    t1,
                    t2,
                    &a1,
                    &a2);
        rocke_value_t* mean_d = rocke_b_fdiv(b, a1, L.m_f);
        rocke_value_t* mean = rocke_b_fadd(b, shift, mean_d);
        rocke_value_t* q = rocke_b_fdiv(b, a2, L.m_f);
        rocke_value_t* m2 = rocke_b_fmul(b, mean_d, mean_d);
        q = rocke_b_fsub(b, q, m2);
        rocke_value_t* z = rocke_b_const_f32(b, 0.0);
        rocke_value_t* var = rocke_b_fmax(b, q, z);
        q = rocke_b_fadd(b, var, eps);
        rocke_value_t* inv_std = rocke_b_rsqrt(b, q);
        rocke_value_t* c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* e1 = rocke_b_cmp_eq(b, it.sub, c0);
        c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* e2 = rocke_b_cmp_eq(b, L.tid, c0);
        rocke_value_t* owner = rocke_b_land(b, e1, e2);
        {
            rocke_if_t iff = rocke_b_scf_if(b, owner);
            rocke_b_region_enter(b, iff.then_region);
            rocke_value_t* old_mean = rocke_b_global_load_f32(b, RunningMean, it.ch, 4);
            rocke_value_t* old_var = rocke_b_global_load_f32(b, RunningVar, it.ch, 4);
            rocke_value_t* c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* omm = rocke_b_fsub(b, c1, momentum);
            rocke_value_t* u1 = rocke_b_fmul(b, omm, old_mean);
            rocke_value_t* u2 = rocke_b_fmul(b, momentum, mean);
            rocke_b_global_store(b, RunningMean, it.ch, rocke_b_fadd(b, u1, u2), 1);
            c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* mm1 = rocke_b_fsub(b, L.m_f, c1);
            c1 = rocke_b_const_f32(b, 1.0);
            rocke_value_t* den = rocke_b_fmax(b, mm1, c1);
            rocke_value_t* unbias = rocke_b_fdiv(b, L.m_f, den);
            rocke_value_t* v1 = rocke_b_fmul(b, omm, old_var);
            rocke_value_t* v2 = rocke_b_fmul(b, var, unbias);
            rocke_value_t* v3 = rocke_b_fmul(b, momentum, v2);
            rocke_b_global_store(b, RunningVar, it.ch, rocke_b_fadd(b, v1, v3), 1);
            if(spec->save_mean_invstd)
            {
                rocke_b_store_scalar_from_f32(b, Mean, it.ch, mean, spec->dtype);
                rocke_b_store_scalar_from_f32(b, InvStd, it.ch, inv_std, spec->dtype);
            }
            rocke_b_region_leave(b);
        }
        rocke_value_t* gamma_v = rocke_b_load_scalar_as_f32(b, Gamma, it.ch, spec->dtype);
        rocke_value_t* beta_v = rocke_b_load_scalar_as_f32(b, Beta, it.ch, spec->dtype);
        rocke_value_t* scale = rocke_b_fmul(b, inv_std, gamma_v);
        for(int j = 0; j < R; ++j)
        {
            rocke_value_t* xs[RO_MAX_VEC];
            rocke_value_t* ys[RO_MAX_VEC];
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, raws[j], i);
                xs[i] = rocke_b_cast_to_f32(b, e);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* d = rocke_b_fsub(b, xs[i], mean);
                d = rocke_b_fmul(b, d, scale);
                ys[i] = rocke_b_fadd(b, d, beta_v);
            }
            rocke_if_t iff = rocke_b_scf_if(b, oks[j]);
            rocke_b_region_enter(b, iff.then_region);
            rocke_value_t* packed = rocke_b_pack_f32_to(b, ys, VEC, spec->dtype);
            rocke_b_store_vec(b, Y, flats[j], packed, VEC);
            rocke_b_region_leave(b);
        }
        rocke_b_sync(b);
    }
    ro_loop_close(b, &branch);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ------------------------------------------------------------------ *
 * build_batchnorm2d_ro_bwd
 * ------------------------------------------------------------------ */

rocke_kernel_def_t* rocke_build_batchnorm2d_ro_bwd(rocke_ir_builder_t* b,
                                                   const rocke_batchnorm2d_ro_spec_t* spec)
{
    if(b == NULL || spec == NULL || !ro_check(b, spec))
        return NULL;
    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
        return NULL;
    const int VEC = spec->vec, R = spec->regs;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    rocke_value_t* X = ro_param(b, "X", io_ty, true, true, false, 16);
    rocke_value_t* dY = ro_param(b, "dY", io_ty, true, true, false, 16);
    rocke_value_t* Scale = ro_param(b, "Scale", rocke_f32(), true, true, false, 0);
    rocke_value_t* Mean = ro_param(b, "Mean", rocke_f32(), true, true, false, 0);
    rocke_value_t* InvStd = ro_param(b, "InvStd", rocke_f32(), true, true, false, 0);
    rocke_value_t* dX = ro_param(b, "dX", io_ty, true, false, true, 16);
    rocke_value_t* dScale = ro_param(b, "dScale", rocke_f32(), true, false, true, 0);
    rocke_value_t* dBias = ro_param(b, "dBias", rocke_f32(), true, false, true, 0);
    rocke_value_t* SyncI
        = rocke_b_param(b, "SyncI", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* SyncF
        = rocke_b_param(b, "SyncF", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* Err = ro_param(b, "Err", rocke_i32(), true, false, false, 0);
    ro_common_t P;
    ro_common_params(b, &P);
    ro_geom_params(b, &P);

    ro_loop_t L;
    ro_prologue(b, spec, &P, &L);
    rocke_if_else_t branch;
    rocke_value_t* w_id = ro_loop_open(b, &P, &L, &branch);
    {
        ro_item_t it = ro_item(b, P.p_c, P.p_n, L.HW, P.p_nch, P.p_hwch, P.p_sh, P.p_sn, w_id);
        rocke_value_t* mean_c = rocke_b_global_load_f32(b, Mean, it.ch, 4);
        rocke_value_t* invstd_c = rocke_b_global_load_f32(b, InvStd, it.ch, 4);
        rocke_value_t* scale_c = rocke_b_global_load_f32(b, Scale, it.ch, 4);
        rocke_value_t* xr[RO_MAX_REGS];
        rocke_value_t* dr[RO_MAX_REGS];
        rocke_value_t* oks[RO_MAX_REGS];
        rocke_value_t* flats[RO_MAX_REGS];
        for(int j = 0; j < R; ++j)
        {
            ro_tile_index(b, spec, L.tid, j, P.p_c, L.HW, &it, &oks[j], &flats[j]);
            xr[j] = rocke_b_global_load_vN(b, X, flats[j], io_ty, VEC, 0);
            dr[j] = rocke_b_global_load_vN(b, dY, flats[j], io_ty, VEC, 0);
        }
        rocke_value_t* s1 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* s2 = rocke_b_const_f32(b, 0.0);
        rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
        for(int j = 0; j < R; ++j)
        {
            rocke_value_t* xs[RO_MAX_VEC];
            rocke_value_t* dys[RO_MAX_VEC];
            rocke_value_t* prods[RO_MAX_VEC];
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, xr[j], i);
                xs[i] = rocke_b_cast_to_f32(b, e);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, dr[j], i);
                e = rocke_b_cast_to_f32(b, e);
                dys[i] = rocke_b_select(b, oks[j], e, zero_f);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* d = rocke_b_fsub(b, xs[i], mean_c);
                d = rocke_b_fmul(b, d, invstd_c);
                prods[i] = rocke_b_fmul(b, dys[i], d);
            }
            rocke_value_t* r = rocke_tree_reduce(b, ro_fadd_combine, NULL, dys, VEC);
            s1 = rocke_b_fadd(b, s1, r);
            r = rocke_tree_reduce(b, ro_fadd_combine, NULL, prods, VEC);
            s2 = rocke_b_fadd(b, s2, r);
        }
        rocke_value_t *t1 = NULL, *t2 = NULL, *a1 = NULL, *a2 = NULL;
        ro_block_pair(b, spec, s1, s2, L.lds_a, L.lds_c, L.tid, &t1, &t2);
        ro_exchange(b,
                    SyncI,
                    SyncF,
                    Err,
                    L.tid,
                    L.lds_bc,
                    it.ch,
                    it.sub,
                    L.S,
                    P.p_epoch,
                    P.p_spin,
                    t1,
                    t2,
                    &a1,
                    &a2);
        rocke_value_t* c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* e1 = rocke_b_cmp_eq(b, it.sub, c0);
        c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* e2 = rocke_b_cmp_eq(b, L.tid, c0);
        rocke_value_t* owner = rocke_b_land(b, e1, e2);
        {
            rocke_if_t iff = rocke_b_scf_if(b, owner);
            rocke_b_region_enter(b, iff.then_region);
            rocke_b_global_store(b, dScale, it.ch, a2, 1);
            rocke_b_global_store(b, dBias, it.ch, a1, 1);
            rocke_b_region_leave(b);
        }
        rocke_value_t *a, *k2, *k3;
        ro_bwd_coefs(b, a1, a2, scale_c, invstd_c, L.m_f, &a, &k2, &k3);
        for(int j = 0; j < R; ++j)
        {
            rocke_value_t* xs[RO_MAX_VEC];
            rocke_value_t* dys[RO_MAX_VEC];
            rocke_value_t* dxs[RO_MAX_VEC];
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, xr[j], i);
                xs[i] = rocke_b_cast_to_f32(b, e);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, dr[j], i);
                dys[i] = rocke_b_cast_to_f32(b, e);
            }
            for(int i = 0; i < VEC; ++i)
            {
                rocke_value_t* ady = rocke_b_fmul(b, a, dys[i]);
                rocke_value_t* xc = rocke_b_fsub(b, xs[i], mean_c);
                rocke_value_t* kx = rocke_b_fmul(b, k2, xc);
                rocke_value_t* u = rocke_b_fsub(b, ady, kx);
                dxs[i] = rocke_b_fsub(b, u, k3);
            }
            rocke_if_t iff = rocke_b_scf_if(b, oks[j]);
            rocke_b_region_enter(b, iff.then_region);
            rocke_value_t* packed = rocke_b_pack_f32_to(b, dxs, VEC, spec->dtype);
            rocke_b_store_vec(b, dX, flats[j], packed, VEC);
            rocke_b_region_leave(b);
        }
        rocke_b_sync(b);
    }
    ro_loop_close(b, &branch);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batchnorm2d_ro_new(rocke_ir_builder_t* b,
                                                   const rocke_batchnorm2d_ro_spec_t* spec,
                                                   const char* kind)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        bool bwd;
        if(b == NULL || spec == NULL || kind == NULL)
            return NULL;
        bwd = strcmp(kind, "bwd") == 0;
        if(rocke_batchnorm2d_ro_kernel_name(spec, bwd ? "bwd" : "train", name, sizeof name)
           != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return bwd ? rocke_build_batchnorm2d_ro_bwd(b, spec)
                   : rocke_build_batchnorm2d_ro_train(b, spec);
    });
}
