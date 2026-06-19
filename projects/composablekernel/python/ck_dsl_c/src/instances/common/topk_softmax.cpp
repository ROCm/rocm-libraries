// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_topk_softmax.c -- C99 port of ck_dsl/instances/common/topk_softmax.py.
 *
 * Byte-identical builder-call sequence vs the Python build_topk_softmax: each
 * ckc_b_* call below mirrors an IRBuilder method call, in the same order.
 *
 * Algorithm (per row, K iterations of find + mask + softmax):
 *   1. each lane scans its slice of the row -> (local_max_val, local_max_idx);
 *   2a. single-wave (BS <= wave_size): packed wave-XOR argmax butterfly (no LDS);
 *   2b. multi-wave  (BS >  wave_size): LDS tree max reduce + LDS race-write idx;
 *   3. winner masked to -inf in the per-lane cache for the next iteration;
 *   4. softmax over the K picks computed in registers (exp2 form, log2(e));
 *   5. K stores distributed across K lanes (lane k writes (Y[m,k], Idx[m,k])).
 */

#include "ckc/instance_topk_softmax.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arch_target.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.reduction.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary (ckc::guard_builder in ckc_build_topk_softmax_new) catches it
 * and records status + message on the builder, so the extern "C" ABI is
 * unchanged. [[noreturn]] keeps the existing `...; return NULL;` call sites
 * valid -- the trailing return is simply never reached. */
[[noreturn]] static void ckc_topk_fail(ckc_ir_builder_t* b, ckc_status_t st, const char* msg)
{
    (void)b;
    ckc::raise_status(st, msg ? msg : "");
}

/* topk_softmax.py: _NEG_INF_F32 = -3.4028234663852886e38. */
static const double CKC_TOPK_NEG_INF_F32 = -3.4028234663852886e38;

/* topk_softmax.py: LN2_E = log2(e). */
static const double CKC_TOPK_LN2_E = 1.4426950408889634;

/* Max per-thread cache (elems_per_thread <= 64 by is_valid_spec). */
#define CKC_TOPK_MAX_EPOT 64
/* Max K (<= 32 by is_valid_spec). */
#define CKC_TOPK_MAX_K 32

/* ===================================================================== *
 *  small host helpers (no IR)
 * ===================================================================== */

/* topk_softmax.py: _ilog2(n) -- floor log2 for power-of-two n. Returns -1 on a
 * non-power-of-two (the Python ValueError; callers gate BS to {32,64,128,256}). */
static int ckc_topk_ilog2(int n)
{
    int out = 0;
    if(n <= 0 || (n & (n - 1)) != 0)
    {
        return -1;
    }
    while((1 << out) < n)
    {
        out += 1;
    }
    return out;
}

ckc_topk_softmax_spec_t ckc_topk_softmax_spec_default(void)
{
    ckc_topk_softmax_spec_t s;
    s.n_per_row         = 0;
    s.k                 = 0;
    s.dtype             = "f32";
    s.out_dtype         = "f32";
    s.block_size        = 64;
    s.name              = "ck_dsl_topk_softmax";
    s.cross_wave_argmax = false;
    return s;
}

int ckc_topk_softmax_elems_per_thread(const ckc_topk_softmax_spec_t* spec)
{
    /* (n_per_row + block_size - 1) // block_size. */
    if(spec == NULL || spec->block_size <= 0)
    {
        return 0;
    }
    return (spec->n_per_row + spec->block_size - 1) / spec->block_size;
}

ckc_status_t
ckc_topk_softmax_kernel_name(const ckc_topk_softmax_spec_t* spec, char* out, size_t out_cap)
{
    /* kernel_name_join(name, dtype, out_dtype, "N{N}", "K{K}", "b{BS}"). */
    char nbuf[24];
    char kbuf[24];
    char bbuf[24];
    const char* parts[5];

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    snprintf(nbuf, sizeof(nbuf), "N%d", spec->n_per_row);
    snprintf(kbuf, sizeof(kbuf), "K%d", spec->k);
    snprintf(bbuf, sizeof(bbuf), "b%d", spec->block_size);
    parts[0] = spec->dtype;
    parts[1] = spec->out_dtype;
    parts[2] = nbuf;
    parts[3] = kbuf;
    parts[4] = bbuf;
    return ckc_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

static void ckc_topk_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason != NULL && reason_cap > 0)
    {
        size_t n = strlen(msg);
        if(n >= reason_cap)
        {
            n = reason_cap - 1;
        }
        memcpy(reason, msg, n);
        reason[n] = '\0';
    }
}

bool ckc_topk_softmax_is_valid_spec(const ckc_topk_softmax_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    char buf[128];
    int epot;

    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(spec == NULL)
    {
        ckc_topk_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    /* ArchTarget.from_gfx(arch) -- KeyError -> reject. */
    if(ckc_arch_target_from_gfx(arch) == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s", arch);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!(strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "bf16") == 0 ||
         strcmp(spec->dtype, "f32") == 0))
    {
        snprintf(buf, sizeof(buf), "unsupported dtype '%s'", spec->dtype);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!(strcmp(spec->out_dtype, "f16") == 0 || strcmp(spec->out_dtype, "bf16") == 0 ||
         strcmp(spec->out_dtype, "f32") == 0))
    {
        snprintf(buf, sizeof(buf), "unsupported out_dtype '%s'", spec->out_dtype);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->k <= 0)
    {
        snprintf(buf, sizeof(buf), "k must be > 0 (got %d)", spec->k);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->k > 32)
    {
        snprintf(buf, sizeof(buf), "k must be <= 32 (got %d)", spec->k);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->k > spec->n_per_row)
    {
        snprintf(buf, sizeof(buf), "k (%d) > n_per_row (%d)", spec->k, spec->n_per_row);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!(spec->block_size == 32 || spec->block_size == 64 || spec->block_size == 128 ||
         spec->block_size == 256))
    {
        snprintf(buf, sizeof(buf), "block_size %d not in {32, 64, 128, 256}", spec->block_size);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    epot = ckc_topk_softmax_elems_per_thread(spec);
    if(epot > 64)
    {
        snprintf(buf, sizeof(buf), "elems_per_thread %d > 64; pick a larger block_size", epot);
        ckc_topk_set_reason(reason, reason_cap, buf);
        return false;
    }
    ckc_topk_set_reason(reason, reason_cap, "ok");
    return true;
}

ckc_status_t ckc_topk_softmax_grid(int m, const ckc_topk_softmax_spec_t* spec, int out[3])
{
    /* topk_softmax_grid: ceil_div_grid((m, 1)). */
    int totals[1];
    int tiles[1];
    (void)spec;
    totals[0] = m;
    tiles[0]  = 1;
    return ckc_ceil_div_grid(totals, tiles, 1, out);
}

/* ===================================================================== *
 *  per-build small helpers (mirror the Python module functions)
 * ===================================================================== */

/* topk_softmax.py: _scalar_store_from_f32(b, ptr, idx, value_f32, dtype). */
static void ckc_topk_scalar_store_from_f32(ckc_ir_builder_t* b,
                                           ckc_value_t* ptr,
                                           ckc_value_t* idx,
                                           ckc_value_t* value_f32,
                                           const char* dtype)
{
    if(strcmp(dtype, "f32") == 0)
    {
        ckc_b_global_store(b, ptr, idx, value_f32, 4);
        return;
    }
    /* f16 / bf16: b.global_store(ptr, idx, b.cast_f32_to(value_f32, io_ir_type(dtype))). */
    {
        const ckc_type_t* ty = ckc_b_io_ir_type(b, dtype);
        ckc_b_global_store(b, ptr, idx, ckc_b_cast_f32_to(b, value_f32, ty), 0);
    }
}

/* topk_softmax.py: _wave_argmax_butterfly(b, val, idx, stages).
 * Wave-XOR butterfly argmax over `stages` halving steps. After all stages every
 * participating lane holds the same (max_val, argmax_idx). Tie-break: smaller
 * idx wins. Writes the converged (val, idx) to *out_val / *out_idx. */
static void ckc_topk_wave_argmax_butterfly(ckc_ir_builder_t* b,
                                           ckc_value_t* val,
                                           ckc_value_t* idx,
                                           int stages,
                                           ckc_value_t** out_val,
                                           ckc_value_t** out_idx)
{
    ckc_value_t* cur_val = val;
    ckc_value_t* cur_idx = idx;
    int k;
    for(k = 0; k < stages; ++k)
    {
        ckc_value_t* remote_val = ckc_b_warp_shuffle_xor(b, cur_val, 1 << k);
        ckc_value_t* remote_idx = ckc_b_warp_shuffle_xor(b, cur_idx, 1 << k);
        /* Python evaluates b.lor(a, b.land(c, d)) left-to-right: the lor's first
         * arg (fcmp ogt) is built first, then the land's args (fcmp oeq, then
         * cmp_lt), then land, then lor. C leaves argument-eval order
         * unspecified, so hoist each builder call to its own sequenced statement
         * to pin the emission order: ogt, oeq, slt, and, or. */
        ckc_value_t* ogt              = ckc_b_fcmp(b, "ogt", remote_val, cur_val);
        ckc_value_t* oeq              = ckc_b_fcmp(b, "oeq", remote_val, cur_val);
        ckc_value_t* lt               = ckc_b_cmp_lt(b, remote_idx, cur_idx);
        ckc_value_t* tie              = ckc_b_land(b, oeq, lt);
        ckc_value_t* is_remote_better = ckc_b_lor(b, ogt, tie);
        cur_val                       = ckc_b_select(b, is_remote_better, remote_val, cur_val);
        cur_idx                       = ckc_b_select(b, is_remote_better, remote_idx, cur_idx);
    }
    *out_val = cur_val;
    *out_idx = cur_idx;
}

/* topk_softmax.py: make_input_distribution(...).calculate_x(b, ys=[e], ps=[[tid]]).
 *
 * The warp-per-row input distribution decomposes N as (IssuesPerThread,
 * block_size) over a single X dim; calculate_x emits local_idx = e*BS + tid.
 * We inline that exact op sequence here, matching calculate_x byte for byte
 * (it seeds x with const_i32(0) and walks levels innermost-first):
 *
 *   x = b.const_i32(0)
 *   x = b.add(x, tid)                             # level 1 (block_size), stride 1
 *   x = b.add(x, b.mul(e_value, b.const_i32(BS))) # level 0 (issue), stride block_size
 *
 * The stride-1 level emits a literal `add nsw i32 0, %tid` (calculate_x does
 * NOT fold the seed-zero add away); the outer level multiplies the issue index
 * by the inner extent (block_size). */
static ckc_value_t* ckc_topk_input_local_idx(ckc_ir_builder_t* b,
                                             ckc_value_t* e_value,
                                             ckc_value_t* tid,
                                             int block_size)
{
    /* x = b.const_i32(0). */
    ckc_value_t* x = ckc_b_const_i32(b, 0);
    /* level 1 (block_size, stride 1): contributor is P0 = tid -> add(x, tid). */
    x = ckc_b_add(b, x, tid);
    /* level 0 (issue, stride block_size): contributor is Y0 = e. */
    x = ckc_b_add(b, x, ckc_b_mul(b, e_value, ckc_b_const_i32(b, block_size)));
    return x;
}

/* ===================================================================== *
 *  build_topk_softmax
 * ===================================================================== */

ckc_kernel_def_t*
ckc_build_topk_softmax(ckc_ir_builder_t* b, const ckc_topk_softmax_spec_t* spec, const char* arch)
{
    const ckc_arch_target_t* target;
    int wave_size;
    int N, K, BS, epot;
    int use_wave_argmax;
    int intra_stages;
    int e, pick_k, kk;
    char reason[128];

    ckc_param_opts_t opts;
    const ckc_type_t* x_elem;
    const ckc_type_t* y_elem;
    ckc_value_t* X;
    ckc_value_t* Y;
    ckc_value_t* Idx;
    ckc_value_t* tid;
    ckc_value_t* row;
    ckc_value_t* c_N;
    ckc_value_t* row_base;
    ckc_value_t* c_neg_inf;

    ckc_value_t* cache[CKC_TOPK_MAX_EPOT];
    ckc_value_t* cache_idx[CKC_TOPK_MAX_EPOT];
    ckc_value_t* picks_val[CKC_TOPK_MAX_K];
    ckc_value_t* picks_idx[CKC_TOPK_MAX_K];
    ckc_value_t* exps[CKC_TOPK_MAX_K] = {0};

    ckc_value_t* lds_red    = NULL;
    ckc_value_t* lds_winner = NULL;

    ckc_value_t* vmax;
    ckc_value_t* LN2_E;
    ckc_value_t* s_sum;
    ckc_value_t* inv_sum;
    ckc_value_t* c_K;
    ckc_value_t* row_out_base;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...). */
    if(!ckc_topk_softmax_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        /* raise ValueError(...) -- caught at the public entry boundary. */
        ckc_topk_fail(
            b, CKC_ERR_VALUE, ckc::format_error("invalid topk_softmax spec: %s", reason).c_str());
    }

    target    = ckc_arch_target_from_gfx(arch);
    wave_size = target->wave_size;

    N    = spec->n_per_row;
    K    = spec->k;
    BS   = spec->block_size;
    epot = ckc_topk_softmax_elems_per_thread(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS. */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* X = b.param("X", PtrType(io_ir_type(dtype) if dtype!="f32" else F32, "global"),
     *             noalias=True, readonly=True, align=16). */
    x_elem = (strcmp(spec->dtype, "f32") == 0) ? ckc_f32() : ckc_b_io_ir_type(b, spec->dtype);
    memset(&opts, 0, sizeof(opts));
    opts.noalias      = true;
    opts.noalias_set  = true;
    opts.readonly     = true;
    opts.readonly_set = true;
    opts.align        = 16;
    opts.align_set    = true;
    X                 = ckc_b_param(b, "X", ckc_ptr_type(b, x_elem, "global"), &opts);

    /* Y = b.param("Y", PtrType(io_ir_type(out_dtype) if out_dtype!="f32" else F32,"global"),
     *             noalias=True, writeonly=True, align=16). */
    y_elem =
        (strcmp(spec->out_dtype, "f32") == 0) ? ckc_f32() : ckc_b_io_ir_type(b, spec->out_dtype);
    memset(&opts, 0, sizeof(opts));
    opts.noalias       = true;
    opts.noalias_set   = true;
    opts.writeonly     = true;
    opts.writeonly_set = true;
    opts.align         = 16;
    opts.align_set     = true;
    Y                  = ckc_b_param(b, "Y", ckc_ptr_type(b, y_elem, "global"), &opts);

    /* Idx = b.param("Idx", PtrType(I32,"global"), noalias=True, writeonly=True, align=4). */
    memset(&opts, 0, sizeof(opts));
    opts.noalias       = true;
    opts.noalias_set   = true;
    opts.writeonly     = true;
    opts.writeonly_set = true;
    opts.align         = 4;
    opts.align_set     = true;
    Idx                = ckc_b_param(b, "Idx", ckc_ptr_type(b, ckc_i32(), "global"), &opts);

    /* M = b.param("M", I32); _ = b.param("N", I32). */
    (void)ckc_b_param(b, "M", ckc_i32(), NULL);
    (void)ckc_b_param(b, "N", ckc_i32(), NULL);

    tid = ckc_b_thread_id_x(b);
    row = ckc_b_block_id_x(b);

    /* c_N = const_i32(N); row_base = mul(row, c_N); c_neg_inf = const_f32(_NEG_INF_F32). */
    c_N       = ckc_b_const_i32(b, N);
    row_base  = ckc_b_mul(b, row, c_N);
    c_neg_inf = ckc_b_const_f32(b, CKC_TOPK_NEG_INF_F32);

    /* Per-thread cache: load this thread's slice of the row into f32 registers. */
    for(e = 0; e < epot; ++e)
    {
        ckc_value_t* local_idx;
        ckc_value_t* in_bounds;
        ckc_value_t* addr;
        ckc_value_t* other;
        const ckc_type_t* load_ty;
        ckc_value_t* loaded;
        ckc_value_t* v_f32;

        /* (local_idx,) = in_dist.calculate_x(b, ys=[const_i32(e)], ps=[[tid]]). */
        local_idx = ckc_topk_input_local_idx(b, ckc_b_const_i32(b, e), tid, BS);

        /* in_bounds = cmp_lt(local_idx, c_N). */
        in_bounds = ckc_b_cmp_lt(b, local_idx, c_N);

        /* Python: b.masked_global_load(X, b.add(row_base, local_idx), in_bounds,
         * <other>, ...) -- args eval left-to-right, so the address add() is
         * emitted BEFORE the `other` cast. Hoist the add to pin that order. */
        addr = ckc_b_add(b, row_base, local_idx);

        /* other = c_neg_inf if dtype=="f32" else cast_f32_to(c_neg_inf, io_ir_type(dtype)). */
        if(strcmp(spec->dtype, "f32") == 0)
        {
            other   = c_neg_inf;
            load_ty = ckc_f32();
        }
        else
        {
            const ckc_type_t* io_ty = ckc_b_io_ir_type(b, spec->dtype);
            other                   = ckc_b_cast_f32_to(b, c_neg_inf, io_ty);
            load_ty                 = io_ty;
        }

        /* loaded = masked_global_load(X, addr, in_bounds, other, load_ty). */
        loaded = ckc_b_masked_global_load(b, X, addr, in_bounds, other, load_ty, 0);

        /* v_f32 = loaded if dtype=="f32" else cast_to_f32(loaded). */
        if(strcmp(spec->dtype, "f32") == 0)
        {
            v_f32 = loaded;
        }
        else
        {
            v_f32 = ckc_b_cast_to_f32(b, loaded);
        }
        cache[e]     = v_f32;
        cache_idx[e] = local_idx;
    }

    /* use_wave_argmax = BS <= wave_size. */
    use_wave_argmax = (BS <= wave_size);
    if(use_wave_argmax)
    {
        intra_stages = ckc_topk_ilog2(BS);
        lds_red      = NULL;
        lds_winner   = NULL;
    }
    else
    {
        int shape_red[1];
        int shape_one[1];
        intra_stages = 0;
        /* lds_red    = make_lds_view(b, dtype=F32, shape=(BS,), ...).base. */
        shape_red[0] = BS;
        lds_red      = ckc_b_smem_alloc(b, ckc_f32(), shape_red, 1, "lds_red");
        /* lds_winner = make_lds_view(b, dtype=F32, shape=(1,), ...).base. */
        shape_one[0] = 1;
        lds_winner   = ckc_b_smem_alloc(b, ckc_f32(), shape_one, 1, "lds_winner");
    }

    /* K iterations of pick-max + mask. */
    for(pick_k = 0; pick_k < K; ++pick_k)
    {
        ckc_value_t* local_max;
        ckc_value_t* local_arg;
        ckc_value_t* global_max;
        ckc_value_t* winner_idx;

        /* 1) per-thread local argmax over this lane's cache slice. */
        local_max = c_neg_inf;
        local_arg = ckc_b_const_i32(b, -1);
        for(e = 0; e < epot; ++e)
        {
            ckc_value_t* is_greater = ckc_b_fcmp(b, "ogt", cache[e], local_max);
            local_max               = ckc_b_select(b, is_greater, cache[e], local_max);
            local_arg               = ckc_b_select(b, is_greater, cache_idx[e], local_arg);
        }

        if(use_wave_argmax)
        {
            /* 2a) wave-XOR packed argmax butterfly. */
            ckc_topk_wave_argmax_butterfly(
                b, local_max, local_arg, intra_stages, &global_max, &winner_idx);
        }
        else
        {
            /* 2b) LDS-tree max reduce + LDS race-write argmax. */
            ckc_value_t* matches;
            ckc_if_t gate;
            ckc_value_t* winner_vec_f32;
            ckc_value_t* idx0;
            global_max = ckc_block_lds_reduce(b, local_max, lds_red, tid, BS, CKC_REDUCE_MAX);
            matches    = ckc_b_fcmp(b, "oeq", local_max, global_max);
            gate       = ckc_b_scf_if(b, matches);
            ckc_b_region_enter(b, gate.then_region);
            {
                ckc_value_t* arg_as_f32 = ckc_b_bitcast(b, local_arg, ckc_f32());
                ckc_value_t* z          = ckc_b_const_i32(b, 0);
                ckc_b_smem_store_vN_f32(b, lds_winner, &z, 1, arg_as_f32, 1);
            }
            ckc_b_region_leave(b);
            ckc_b_sync(b);
            idx0           = ckc_b_const_i32(b, 0);
            winner_vec_f32 = ckc_b_smem_load_vN_f32(b, lds_winner, &idx0, 1, 1);
            winner_idx     = ckc_b_bitcast(b, ckc_b_vec_extract(b, winner_vec_f32, 0), ckc_i32());
        }

        picks_val[pick_k] = global_max;
        picks_idx[pick_k] = winner_idx;

        /* 3) mask out the winning element for the next iteration. */
        for(e = 0; e < epot; ++e)
        {
            ckc_value_t* owns = ckc_b_cmp_eq(b, cache_idx[e], winner_idx);
            cache[e]          = ckc_b_select(b, owns, c_neg_inf, cache[e]);
        }
    }

    /* 5) softmax over the K picked values. vmax = picks_val[0]. */
    vmax  = picks_val[0];
    LN2_E = ckc_b_const_f32(b, CKC_TOPK_LN2_E);
    for(kk = 0; kk < K; ++kk)
    {
        ckc_value_t* s = ckc_b_fmul(b, ckc_b_fsub(b, picks_val[kk], vmax), LN2_E);
        exps[kk]       = ckc_b_exp2(b, s);
    }
    s_sum = exps[0];
    for(kk = 1; kk < K; ++kk)
    {
        s_sum = ckc_b_fadd(b, s_sum, exps[kk]);
    }
    inv_sum = ckc_b_rcp(b, s_sum);

    /* 6) per-row write distributed across K lanes (lane k writes (Y[m,k], Idx[m,k])). */
    c_K          = ckc_b_const_i32(b, K);
    row_out_base = ckc_b_mul(b, row, c_K);
    for(kk = 0; kk < K; ++kk)
    {
        ckc_value_t* cond = ckc_b_cmp_eq(b, tid, ckc_b_const_i32(b, kk));
        ckc_if_t gate     = ckc_b_scf_if(b, cond);
        ckc_b_region_enter(b, gate.then_region);
        {
            ckc_value_t* y_f32   = ckc_b_fmul(b, exps[kk], inv_sum);
            ckc_value_t* out_off = ckc_b_add(b, row_out_base, ckc_b_const_i32(b, kk));
            ckc_topk_scalar_store_from_f32(b, Y, out_off, y_f32, spec->out_dtype);
            ckc_b_global_store(b, Idx, out_off, picks_idx[kk], 4);
        }
        ckc_b_region_leave(b);
    }

    return ckc_ir_builder_kernel(b);
}

ckc_kernel_def_t* ckc_build_topk_softmax_new(ckc_ir_builder_t* b,
                                             const ckc_topk_softmax_spec_t* spec,
                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[160];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_topk_softmax_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_topk_softmax(b, spec, arch);
    });
}

/* ===================================================================== *
 *  ckc_topk_softmax_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */

static void ckc_topk_set_err(char* err, size_t err_cap, const char* m)
{
    if(err != NULL && err_cap > 0)
    {
        size_t n = strlen(m);
        if(n >= err_cap)
        {
            n = err_cap - 1;
        }
        memcpy(err, m, n);
        err[n] = '\0';
    }
}

ckc_status_t ckc_topk_softmax_lower_to_llvm(const ckc_topk_softmax_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        ckc_topk_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_topk_softmax_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m;
        st = ckc_ir_builder_status(&b);
        m  = ckc_ir_builder_error(&b);
        if(m == NULL)
        {
            m = "build_topk_softmax failed";
        }
        ckc_topk_set_err(err, err_cap, m);
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
