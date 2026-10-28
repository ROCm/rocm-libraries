// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_universal.c -- PUBLIC entry chunk of the C99 port of
 * rocke/instances/common/gemm_universal.py :: build_universal_gemm.
 *
 * SCOPE (this translation unit):
 *   - rocke_build_universal_gemm        (the driver: alloc+init ctx, then invoke
 *                                      the phase functions in Python order)
 *   - rocke_build_universal_gemm_new    (init-from-spec convenience wrapper)
 *   - rocke_gemm_universal_lower_to_llvm (build + lower-to-.ll convenience)
 *   - all_dispatcher_configs          (host-side config enumerator; see note)
 *
 * The phase functions (emit_load_phase, emit_mfma_phase, the K-loop selectors,
 * the epilogues, ...) live in PEER translation units and are declared in
 * rocke/instance_gemm_internal.h. This file only sets up the shared
 * rocke_gemm_build_ctx_t exactly as the Python prologue computes its locals, then
 * dispatches into those peers. The populate order below tracks
 * build_universal_gemm() top-to-bottom so a reviewer can diff line by line.
 */

#include <stdio.h> /* snprintf (C++ build: not pulled in transitively) */
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_internal.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_tensor_gett.h" /* the _gett side channel */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.grid.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/lower_llvm.h"

/* File-local integer formatter (used for arena-stable acc names). Defined at
 * the bottom of this TU; forward-declared here so the driver can call it. */
static int rocke_gemm_itoa(int v, char* out);

/* ===================================================================== *
 *  module-level pure helpers used by the driver prologue
 *
 *  These mirror the file-scope Python helpers _resolve_mma_op / _storage_dtype
 *  / _mma_family. They are local to this TU (not declared in the internal
 *  header), so they are static. They are the prologue's "resolve environment"
 *  step (build_universal_gemm lines 308-364 in the Python).
 * ===================================================================== */

/* _mma_family(arch): "wmma" on RDNA (gfx11xx/gfx12xx), else "mma". The
 * canonical decision lives in the arch port; here we only need the family the
 * catalog query expects. We mirror the Python: RDNA targets => "wmma". */
static const char* rocke_gemm_mma_family(const char* arch)
{
    if(arch != NULL && (strncmp(arch, "gfx11", 5) == 0 || strncmp(arch, "gfx12", 5) == 0))
    {
        return "wmma";
    }
    return "mma";
}

/* _resolve_mma_op(spec, arch): resolve the warp-tile atom from the target
 * catalog. Returns NULL when the target has no atom for the shape+dtype. */
static const rocke_mmaop_t* rocke_gemm_resolve_mma_op(const rocke_gemm_universal_spec_t* spec,
                                                      const char* arch,
                                                      const rocke_archtarget_t* target)
{
    const char* a_name; /* homogeneous A/B/C (validated in _storage_dtype) */
    const rocke_gemm_tile_spec_t* t;
    if(spec == NULL || target == NULL)
    {
        return NULL;
    }
    t = &spec->tile;
    a_name = spec->data.dtype_a;
    return rocke_archtarget_op_for_shape(target,
                                         rocke_gemm_mma_family(arch),
                                         a_name,
                                         a_name,
                                         "fp32",
                                         t->warp_tile_m,
                                         t->warp_tile_n,
                                         t->warp_tile_k,
                                         nullptr);
}

/* _storage_dtype(spec): validate homogeneous A/B/C + fp32 acc + RCR layout,
 * then map the storage dtype string to its IR type. On a Python ValueError
 * path this returns NULL and sets the builder's sticky error. */
static const rocke_type_t* rocke_gemm_storage_dtype(rocke_ir_builder_t* b,
                                                    const rocke_gemm_universal_spec_t* spec)
{
    const rocke_gemm_data_spec_t* d = &spec->data;
    if(strcmp(d->dtype_a, d->dtype_b) != 0 || strcmp(d->dtype_a, d->dtype_c) != 0)
    {
        /* "requires homogeneous A/B/C dtypes" */
        return rocke_b_io_ir_type(b, "");
    }
    if(strcmp(d->dtype_acc, "fp32") != 0 && strcmp(d->dtype_acc, "f32") != 0)
    {
        return rocke_b_io_ir_type(b, "");
    }
    /* GEMM_LAYOUTS: RCR and the MN-contiguous RRR / CRR / CCR */
    if(strcmp(d->layout, "RCR") != 0 && strcmp(d->layout, "RRR") != 0
       && strcmp(d->layout, "CRR") != 0 && strcmp(d->layout, "CCR") != 0)
    {
        return rocke_b_io_ir_type(b, "");
    }
    return rocke_b_io_ir_type(b, d->dtype_a); /* _dtype_ir == io_ir_type */
}

/* getenv override that mirrors int(os.environ.get(name, str(dflt))). */
static int rocke_gemm_env_int(const char* name, int dflt)
{
    const char* v = getenv(name);
    if(v == NULL || v[0] == '\0')
    {
        return dflt;
    }
    return (int)strtol(v, NULL, 10);
}

/* getenv string-equals helper, mirroring os.environ.get(name, dflt) == "1". */
static int rocke_gemm_env_eq(const char* name, const char* dflt, const char* want)
{
    const char* v = getenv(name);
    if(v == NULL || v[0] == '\0')
    {
        v = dflt;
    }
    return strcmp(v, want) == 0;
}

/* ===================================================================== *
 *  rocke_build_wsp3_gemm -- C99 port of gemm_wsp3.py :: build_wsp3_gemm.
 *
 *  Warp-specialized producer/consumer 3-stage GEMM (gfx950-only, fp16-only).
 *  Mirrors the Python emitter 1:1 so the LLVM .ll is byte-identical. The
 *  routing point in rocke_build_universal_gemm forwards every pipeline=="wsp3"
 *  spec here (the universal entry, the grouped/batched delegations).
 * ===================================================================== */

/* NUM_PRODUCER_WARPS default (gemm_wsp3.py module constant). */
#define ROCKE_WSP3_NUM_PRODUCER_WARPS 2
#define ROCKE_WSP3_MAX_ACCS ROCKE_GEMM_MAX_ACCS

/* _pick_load_vec(elems_a, elems_b, threads). */
static int rocke_wsp3_pick_load_vec(int elems_a, int elems_b, int threads)
{
    static const int cand[4] = {8, 4, 2, 1};
    int i;
    for(i = 0; i < 4; ++i)
    {
        int v = cand[i];
        if(elems_a % (v * threads) == 0 && elems_b % (v * threads) == 0)
        {
            return v;
        }
    }
    return 1;
}

/* Shared state captured by the three nested closures in gemm_wsp3.py
 * (_producer_load, _producer_load_async, _consumer_mfma). Mirrors exactly the
 * enclosing-function locals those closures reference. */
typedef struct rocke_wsp3_ctx
{
    rocke_ir_builder_t* b;
    const rocke_gemm_universal_spec_t* spec;
    const rocke_mmaop_t* op;
    const rocke_type_t* storage_dtype;
    int block_m, block_n, block_k;
    int load_vec, a_vecs, b_vecs;
    int wtm, wtn, wtk;
    int mfmas_m, mfmas_n, k_atoms;
    int a_per_lane, b_per_lane;
    int prod_threads;
    int dtl_dwords, dtl_halves, dtl_bpl;
    int a_passes, b_passes;
    int sched;
    /* SSA values */
    rocke_value_t *c0, *c_lv, *c_kdv, *c_prod, *c_block_k;
    rocke_value_t *tid, *lane;
    rocke_value_t *block_m_off, *block_n_off, *K;
    rocke_value_t *A_smem, *B_smem;
    rocke_tensor_view_t *a_view, *b_view, *a_lds_view, *b_lds_view;
    rocke_value_t *warp_m_off, *warp_n_off, *k_blk_kbase;
    rocke_value_t *m_in_atom, *n_in_atom;
    /* async-only */
    rocke_value_t *a_rsrc, *b_rsrc, *a_lds0, *b_lds0, *zsoff, *warp_off, *c2;
    rocke_value_t *c_chunks_per_row, *c_halves_per_chunk;
} rocke_wsp3_ctx_t;

/* _rc(vec_idx) -> (row, col). */
static void rocke_wsp3_rc(rocke_wsp3_ctx_t* w,
                          rocke_value_t* vec_idx,
                          rocke_value_t** row,
                          rocke_value_t** col)
{
    rocke_ir_builder_t* b = w->b;
    *row = rocke_b_div(b, vec_idx, w->c_kdv);
    *col = rocke_b_mul(b, rocke_b_mod(b, vec_idx, w->c_kdv), w->c_lv);
}

/* _producer_load(k_off, a_buf_row, b_buf_row): coalesced global->LDS for one
 * K-tile (global->VGPR->LDS). */
static void rocke_wsp3_producer_load(rocke_wsp3_ctx_t* w,
                                     rocke_value_t* k_off,
                                     rocke_value_t* a_buf_row,
                                     rocke_value_t* b_buf_row)
{
    rocke_ir_builder_t* b = w->b;
    rocke_tile_window_t a_gt, b_gt, a_lt, b_lt;
    int e;
    {
        int a_glen[3] = {1, w->block_m, w->block_k};
        int b_glen[3] = {1, w->block_n, w->block_k};
        rocke_value_t* a_gorigin[3] = {w->c0, w->block_m_off, k_off};
        rocke_value_t* b_gorigin[3] = {w->c0, w->block_n_off, k_off};
        rocke_make_tile_window(&a_gt, w->a_view, a_glen, a_gorigin, 3);
        rocke_make_tile_window(&b_gt, w->b_view, b_glen, b_gorigin, 3);
        int a_llen[2] = {0, w->block_k}; /* depth*block_m filled below */
        int b_llen[2] = {0, w->block_k};
        a_llen[0] = w->a_lds_view->desc.shape[0];
        b_llen[0] = w->b_lds_view->desc.shape[0];
        rocke_value_t* a_lorigin[2] = {w->c0, w->c0};
        rocke_value_t* b_lorigin[2] = {w->c0, w->c0};
        rocke_make_tile_window(&a_lt, w->a_lds_view, a_llen, a_lorigin, 2);
        rocke_make_tile_window(&b_lt, w->b_lds_view, b_llen, b_lorigin, 2);
    }
    for(e = 0; e < w->a_vecs; ++e)
    {
        rocke_value_t *row, *col, *val;
        rocke_value_t* vi
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), w->c_prod), w->tid);
        rocke_wsp3_rc(w, vi, &row, &col);
        rocke_value_t* gidx[3] = {w->c0, row, col};
        val = rocke_tile_window_load_vec(b, &a_gt, gidx, 3, w->load_vec);
        rocke_value_t* lidx[2] = {rocke_b_add(b, a_buf_row, row), col};
        rocke_tile_window_store_vec(b, &a_lt, lidx, 2, val, w->load_vec);
    }
    for(e = 0; e < w->b_vecs; ++e)
    {
        rocke_value_t *row, *col, *val;
        rocke_value_t* vi
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), w->c_prod), w->tid);
        rocke_wsp3_rc(w, vi, &row, &col);
        rocke_value_t* gidx[3] = {w->c0, row, col};
        val = rocke_tile_window_load_vec(b, &b_gt, gidx, 3, w->load_vec);
        rocke_value_t* lidx[2] = {rocke_b_add(b, b_buf_row, row), col};
        rocke_tile_window_store_vec(b, &b_lt, lidx, 2, val, w->load_vec);
    }
}

/* _producer_load_async(k_off, a_buf, b_buf): direct-to-LDS async loads. */
static void rocke_wsp3_producer_load_async(rocke_wsp3_ctx_t* w,
                                           rocke_value_t* k_off,
                                           rocke_value_t* a_buf,
                                           rocke_value_t* b_buf)
{
    rocke_ir_builder_t* b = w->b;
    int p;
    rocke_value_t* a_base = rocke_b_smem_ptr_add(
        b, rocke_b_smem_ptr_add(b, w->a_lds0, rocke_b_zext(b, a_buf, rocke_i64())), w->warp_off);
    rocke_value_t* b_base = rocke_b_smem_ptr_add(
        b, rocke_b_smem_ptr_add(b, w->b_lds0, rocke_b_zext(b, b_buf, rocke_i64())), w->warp_off);
    for(p = 0; p < w->a_passes; ++p)
    {
        rocke_value_t* lds
            = (p == 0) ? a_base
                       : rocke_b_smem_ptr_add(
                             b,
                             a_base,
                             rocke_b_zext(b,
                                          rocke_b_const_i32(b, p * w->prod_threads * w->dtl_bpl),
                                          rocke_i64()));
        rocke_value_t* cidx = rocke_b_add(b, w->tid, rocke_b_const_i32(b, p * w->prod_threads));
        rocke_value_t* row = rocke_b_div(b, cidx, w->c_chunks_per_row);
        rocke_value_t* col
            = rocke_b_mul(b, rocke_b_mod(b, cidx, w->c_chunks_per_row), w->c_halves_per_chunk);
        rocke_value_t* off = rocke_b_add(b,
                                         rocke_b_mul(b, rocke_b_add(b, w->block_m_off, row), w->K),
                                         rocke_b_add(b, k_off, col));
        rocke_b_async_buffer_load_lds_addr(
            b, w->a_rsrc, lds, rocke_b_mul(b, off, w->c2), w->zsoff, w->dtl_dwords, 2);
    }
    for(p = 0; p < w->b_passes; ++p)
    {
        rocke_value_t* lds
            = (p == 0) ? b_base
                       : rocke_b_smem_ptr_add(
                             b,
                             b_base,
                             rocke_b_zext(b,
                                          rocke_b_const_i32(b, p * w->prod_threads * w->dtl_bpl),
                                          rocke_i64()));
        rocke_value_t* cidx = rocke_b_add(b, w->tid, rocke_b_const_i32(b, p * w->prod_threads));
        rocke_value_t* row = rocke_b_div(b, cidx, w->c_chunks_per_row);
        rocke_value_t* col
            = rocke_b_mul(b, rocke_b_mod(b, cidx, w->c_chunks_per_row), w->c_halves_per_chunk);
        rocke_value_t* off = rocke_b_add(b,
                                         rocke_b_mul(b, rocke_b_add(b, w->block_n_off, row), w->K),
                                         rocke_b_add(b, k_off, col));
        rocke_b_async_buffer_load_lds_addr(
            b, w->b_rsrc, lds, rocke_b_mul(b, off, w->c2), w->zsoff, w->dtl_dwords, 2);
    }
}

/* _consumer_mfma(accs, a_buf_row, b_buf_row, out_new[num_accs]): one K-tile of
 * MFMAs for this consumer warp. */
static void rocke_wsp3_consumer_mfma(rocke_wsp3_ctx_t* w,
                                     rocke_value_t* const* accs,
                                     rocke_value_t* a_buf_row,
                                     rocke_value_t* b_buf_row,
                                     rocke_value_t** out_new)
{
    rocke_ir_builder_t* b = w->b;
    int num_accs = w->mfmas_m * w->mfmas_n;
    int kk, mi, ni, i;
    rocke_value_t* a_base = rocke_b_add(b, a_buf_row, w->warp_m_off);
    rocke_value_t* b_base = rocke_b_add(b, b_buf_row, w->warp_n_off);
    for(i = 0; i < num_accs; ++i)
    {
        out_new[i] = accs[i];
    }
    for(kk = 0; kk < w->k_atoms; ++kk)
    {
        rocke_value_t* col_base = rocke_b_add(b, w->k_blk_kbase, rocke_b_const_i32(b, kk * w->wtk));
        rocke_value_t* a_rows[ROCKE_WSP3_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_WSP3_MAX_ACCS];
        if(w->sched)
        {
            for(mi = 0; mi < w->mfmas_m; ++mi)
                a_rows[mi] = NULL;
            for(ni = 0; ni < w->mfmas_n; ++ni)
                b_cols[ni] = NULL;
            a_rows[0] = rocke_gemm_emit_smem_load(
                b,
                w->A_smem,
                rocke_b_add(
                    b, a_base, rocke_b_add(b, rocke_b_const_i32(b, 0 * w->wtm), w->m_in_atom)),
                col_base,
                w->a_per_lane,
                w->storage_dtype);
            rocke_b_s_setprio(b, 1);
            for(mi = 0; mi < w->mfmas_m; ++mi)
            {
                if(mi + 1 < w->mfmas_m)
                {
                    a_rows[mi + 1] = rocke_gemm_emit_smem_load(
                        b,
                        w->A_smem,
                        rocke_b_add(
                            b,
                            a_base,
                            rocke_b_add(b, rocke_b_const_i32(b, (mi + 1) * w->wtm), w->m_in_atom)),
                        col_base,
                        w->a_per_lane,
                        w->storage_dtype);
                }
                for(ni = 0; ni < w->mfmas_n; ++ni)
                {
                    if(mi == 0)
                    {
                        b_cols[ni] = rocke_gemm_emit_smem_load(
                            b,
                            w->B_smem,
                            rocke_b_add(
                                b,
                                b_base,
                                rocke_b_add(b, rocke_b_const_i32(b, ni * w->wtn), w->n_in_atom)),
                            col_base,
                            w->b_per_lane,
                            w->storage_dtype);
                    }
                    int flat = mi * w->mfmas_n + ni;
                    out_new[flat]
                        = rocke_gemm_emit_mma(b, w->op, a_rows[mi], b_cols[ni], out_new[flat]);
                    rocke_b_sched_barrier(b, 0);
                }
            }
            rocke_b_s_setprio(b, 0);
        }
        else
        {
            for(mi = 0; mi < w->mfmas_m; ++mi)
            {
                a_rows[mi] = rocke_gemm_emit_smem_load(
                    b,
                    w->A_smem,
                    rocke_b_add(
                        b, a_base, rocke_b_add(b, rocke_b_const_i32(b, mi * w->wtm), w->m_in_atom)),
                    col_base,
                    w->a_per_lane,
                    w->storage_dtype);
            }
            for(ni = 0; ni < w->mfmas_n; ++ni)
            {
                b_cols[ni] = rocke_gemm_emit_smem_load(
                    b,
                    w->B_smem,
                    rocke_b_add(
                        b, b_base, rocke_b_add(b, rocke_b_const_i32(b, ni * w->wtn), w->n_in_atom)),
                    col_base,
                    w->b_per_lane,
                    w->storage_dtype);
            }
            int flat = 0;
            for(mi = 0; mi < w->mfmas_m; ++mi)
            {
                for(ni = 0; ni < w->mfmas_n; ++ni)
                {
                    out_new[flat]
                        = rocke_gemm_emit_mma(b, w->op, a_rows[mi], b_cols[ni], out_new[flat]);
                    flat += 1;
                }
            }
        }
    }
}

rocke_kernel_def_t* rocke_build_wsp3_gemm(rocke_ir_builder_t* b,
                                          const rocke_gemm_universal_spec_t* spec,
                                          const char* arch)
{
    const rocke_gemm_tile_spec_t* t;
    const rocke_mmaop_t* op;
    const rocke_archtarget_t* target;
    const rocke_type_t* storage_dtype;
    int a_per_lane, b_per_lane, c_per_lane;
    int block_m, block_n, block_k;
    int warp_m, warp_n, wtm, wtn, wtk;
    int mfmas_m, mfmas_n, k_atoms;
    int n_prod_warps, n_cons_warps, total_warps, wave, block_size, prod_threads;
    int a_total, b_total, load_vec, a_vecs, b_vecs, kdv;
    int depth, e;
    int sched, async_mode;
    int DTL_DWORDS = 4, DTL_HALVES = 8, DTL_BPL = 16;
    int async_ok, a_chunks, b_chunks, a_passes, b_passes, loads_per_tile;
    int a_buf_bytes, b_buf_bytes, prologue_tiles, keep_vmcnt;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    /* Python: gfx950-only; fp16 A/B only. */
    if(strcmp(arch, "gfx950") != 0)
    {
        return NULL;
    }
    if(strcmp(spec->data.dtype_a, "fp16") != 0 || strcmp(spec->data.dtype_b, "fp16") != 0)
    {
        return NULL;
    }

    target = rocke_archtarget_from_gfx(arch);
    op = rocke_gemm_resolve_mma_op(spec, arch, target);
    if(op == NULL)
    {
        return NULL;
    }
    rocke_gemm_atom_frag_lengths(op, &a_per_lane, &b_per_lane, &c_per_lane);
    storage_dtype = rocke_gemm_storage_dtype(b, spec);
    if(storage_dtype == NULL)
    {
        return NULL;
    }

    t = &spec->tile;
    block_m = t->tile_m;
    block_n = t->tile_n;
    block_k = t->tile_k;
    warp_m = t->warp_m;
    warp_n = t->warp_n;
    wtm = t->warp_tile_m;
    wtn = t->warp_tile_n;
    wtk = t->warp_tile_k;
    mfmas_m = (block_m / warp_m) / wtm;
    mfmas_n = (block_n / warp_n) / wtn;
    k_atoms = block_k / wtk;

    n_prod_warps = rocke_gemm_env_int("CK_WSP3_PROD", ROCKE_WSP3_NUM_PRODUCER_WARPS);
    n_cons_warps = warp_m * warp_n;
    total_warps = n_prod_warps + n_cons_warps;
    wave = spec->wave_size;
    block_size = total_warps * wave;
    prod_threads = n_prod_warps * wave;

    a_total = block_m * block_k;
    b_total = block_n * block_k;
    load_vec = rocke_wsp3_pick_load_vec(a_total, b_total, prod_threads);
    a_vecs = a_total / load_vec / prod_threads;
    b_vecs = b_total / load_vec / prod_threads;
    kdv = block_k / load_vec;

    /* IRBuilder(spec.kernel_name()): set the kernel symbol. */
    {
        char uni_name[1024];
        if(rocke_gemm_universal_kernel_name(spec, uni_name, sizeof(uni_name)) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_wsp3_gemm: kernel_name() overflow");
            return NULL;
        }
        b->kernel->name = rocke_arena_strdup(&b->arena, uni_name);
        if(b->kernel->name == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "build_wsp3_gemm: kernel name OOM");
            return NULL;
        }
    }
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", block_size);
    if(spec->trait.waves_per_eu_set)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }

    /* ---- params ---- */
    rocke_value_t *A, *Bp, *C, *M, *N, *K;
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_storage = rocke_ptr_type(b, storage_dtype, "global");
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", ptr_storage, &opts);
        Bp = rocke_b_param(b, "B", ptr_storage, &opts);
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = rocke_b_param(b, "C", ptr_storage, &opts);
        M = rocke_b_param(b, "M", rocke_i32(), NULL);
        N = rocke_b_param(b, "N", rocke_i32(), NULL);
        K = rocke_b_param(b, "K", rocke_i32(), NULL);
    }
    (void)M;
    (void)N;

    /* ---- constants ---- */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_wave = rocke_b_const_i32(b, wave);
    rocke_value_t* c_block_k = rocke_b_const_i32(b, block_k);
    rocke_value_t* c_block_m = rocke_b_const_i32(b, block_m);
    rocke_value_t* c_block_n = rocke_b_const_i32(b, block_n);
    rocke_value_t* c_prod = rocke_b_const_i32(b, prod_threads);
    rocke_value_t* c_lv = rocke_b_const_i32(b, load_vec);
    rocke_value_t* c_kdv = rocke_b_const_i32(b, kdv);
    rocke_value_t* c_nprodw = rocke_b_const_i32(b, n_prod_warps);
    rocke_value_t* c_warpn = rocke_b_const_i32(b, warp_n);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* warp_id = rocke_b_div(b, tid, c_wave);
    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);
    rocke_value_t* is_producer = rocke_b_cmp_lt(b, warp_id, c_nprodw);
    rocke_value_t* is_consumer = rocke_b_cmp_ge(b, warp_id, c_nprodw);
    rocke_value_t* cwarp = rocke_b_sub(b, warp_id, c_nprodw);
    rocke_value_t* cwarp_m = rocke_b_div(b, cwarp, c_warpn);
    rocke_value_t* cwarp_n = rocke_b_mod(b, cwarp, c_warpn);

    /* Python evals b.block_id_*() BEFORE b.const_i32() (left-to-right call args);
     * bind each to a temp so the C right-to-left arg eval matches SSA ids. */
    rocke_value_t* bid_y = rocke_b_block_id_y(b);
    rocke_value_t* block_m_off = rocke_b_mul(b, bid_y, rocke_b_const_i32(b, block_m));
    rocke_value_t* bid_x = rocke_b_block_id_x(b);
    rocke_value_t* block_n_off = rocke_b_mul(b, bid_x, rocke_b_const_i32(b, block_n));

    depth = rocke_gemm_env_int("CK_WSP3_DEPTH", 2);
    rocke_value_t* cD = rocke_b_const_i32(b, depth);

    /* ---- LDS ring ---- */
    rocke_value_t* A_smem;
    rocke_value_t* B_smem;
    {
        int a_shape[2] = {depth * block_m, block_k};
        int b_shape[2] = {depth * block_n, block_k};
        A_smem = rocke_b_smem_alloc(b, storage_dtype, a_shape, 2, "A_smem");
        B_smem = rocke_b_smem_alloc(b, storage_dtype, b_shape, 2, "B_smem");
    }
    rocke_tensor_view_t a_lds_view, b_lds_view, a_view, b_view;
    {
        int al_shape[2] = {depth * block_m, block_k};
        int bl_shape[2] = {depth * block_n, block_k};
        rocke_make_global_view(&a_lds_view, A_smem, al_shape, 2, storage_dtype, NULL);
        a_lds_view.addr_space = ROCKE_ADDR_LDS;
        rocke_make_global_view(&b_lds_view, B_smem, bl_shape, 2, storage_dtype, NULL);
        b_lds_view.addr_space = ROCKE_ADDR_LDS;
        int g_shape[3] = {1, 1, 1};
        rocke_stride_t g_strides[3]
            = {rocke_stride_imm(1), rocke_stride_value(K), rocke_stride_imm(1)};
        rocke_make_global_view(&a_view, A, g_shape, 3, storage_dtype, g_strides);
        rocke_make_global_view(&b_view, Bp, g_shape, 3, storage_dtype, g_strides);
    }

    /* ---- async DTL plumbing (Phase 3) ---- */
    async_ok = (block_k % DTL_HALVES) == 0;
    a_chunks = (block_m * block_k) / DTL_HALVES;
    b_chunks = (block_n * block_k) / DTL_HALVES;
    a_passes = (a_chunks + prod_threads - 1) / prod_threads;
    b_passes = (b_chunks + prod_threads - 1) / prod_threads;
    loads_per_tile = a_passes + b_passes;
    a_buf_bytes = block_m * block_k * 2;
    b_buf_bytes = block_n * block_k * 2;

    rocke_value_t *c_chunks_per_row = NULL, *c_halves_per_chunk = NULL, *c2 = NULL;
    rocke_value_t *a_rsrc = NULL, *b_rsrc = NULL, *a_lds0 = NULL, *b_lds0 = NULL;
    rocke_value_t *zsoff = NULL, *warp_off = NULL;
    if(async_ok)
    {
        rocke_value_t* big;
        c_chunks_per_row = rocke_b_const_i32(b, block_k / DTL_HALVES);
        c_halves_per_chunk = rocke_b_const_i32(b, DTL_HALVES);
        c2 = rocke_b_const_i32(b, 2);
        big = rocke_b_const_i32(b, 0x7FFF0000);
        a_rsrc = rocke_b_buffer_rsrc(b, A, big);
        b_rsrc = rocke_b_buffer_rsrc(b, Bp, big);
        a_lds0 = rocke_b_smem_addr_of(b, A_smem);
        b_lds0 = rocke_b_smem_addr_of(b, B_smem);
        zsoff = rocke_b_const_i32(b, 0);
        warp_off = rocke_b_zext(
            b, rocke_b_mul(b, warp_id, rocke_b_const_i32(b, wave * DTL_BPL)), rocke_i64());
    }

    rocke_value_t* m_in_atom = rocke_b_mod(b, lane, rocke_b_const_i32(b, wtm));
    rocke_value_t* k_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, wtm));
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, rocke_b_const_i32(b, wtn));
    rocke_value_t* warp_m_off = rocke_b_mul(b, cwarp_m, rocke_b_const_i32(b, mfmas_m * wtm));
    rocke_value_t* warp_n_off = rocke_b_mul(b, cwarp_n, rocke_b_const_i32(b, mfmas_n * wtn));
    rocke_value_t* k_blk_kbase = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, a_per_lane));

    sched = rocke_gemm_env_eq("CK_WSP3_SCHED", "1", "1");

    int num_accs = mfmas_m * mfmas_n;

    /* Populate the closure-capture context once. */
    rocke_wsp3_ctx_t w;
    memset(&w, 0, sizeof(w));
    w.b = b;
    w.spec = spec;
    w.op = op;
    w.storage_dtype = storage_dtype;
    w.block_m = block_m;
    w.block_n = block_n;
    w.block_k = block_k;
    w.load_vec = load_vec;
    w.a_vecs = a_vecs;
    w.b_vecs = b_vecs;
    w.wtm = wtm;
    w.wtn = wtn;
    w.wtk = wtk;
    w.mfmas_m = mfmas_m;
    w.mfmas_n = mfmas_n;
    w.k_atoms = k_atoms;
    w.a_per_lane = a_per_lane;
    w.b_per_lane = b_per_lane;
    w.prod_threads = prod_threads;
    w.dtl_dwords = DTL_DWORDS;
    w.dtl_halves = DTL_HALVES;
    w.dtl_bpl = DTL_BPL;
    w.a_passes = a_passes;
    w.b_passes = b_passes;
    w.sched = sched;
    w.c0 = c0;
    w.c_lv = c_lv;
    w.c_kdv = c_kdv;
    w.c_prod = c_prod;
    w.c_block_k = c_block_k;
    w.tid = tid;
    w.lane = lane;
    w.block_m_off = block_m_off;
    w.block_n_off = block_n_off;
    w.K = K;
    w.A_smem = A_smem;
    w.B_smem = B_smem;
    w.a_view = &a_view;
    w.b_view = &b_view;
    w.a_lds_view = &a_lds_view;
    w.b_lds_view = &b_lds_view;
    w.warp_m_off = warp_m_off;
    w.warp_n_off = warp_n_off;
    w.k_blk_kbase = k_blk_kbase;
    w.m_in_atom = m_in_atom;
    w.n_in_atom = n_in_atom;
    w.a_rsrc = a_rsrc;
    w.b_rsrc = b_rsrc;
    w.a_lds0 = a_lds0;
    w.b_lds0 = b_lds0;
    w.zsoff = zsoff;
    w.warp_off = warp_off;
    w.c2 = c2;
    w.c_chunks_per_row = c_chunks_per_row;
    w.c_halves_per_chunk = c_halves_per_chunk;

    /* depth==1 serialized path. */
    if(depth == 1)
    {
        /* producer scf.if */
        rocke_if_t pif = rocke_b_scf_if(b, is_producer);
        rocke_b_region_enter(b, pif.then_region);
        {
            rocke_for_t pfor = rocke_b_scf_for(b, c0, K, c_block_k, "kp");
            rocke_b_region_enter(b, pfor.body);
            rocke_wsp3_producer_load(&w, pfor.iv, c0, c0);
            rocke_b_sync(b);
            rocke_b_sync(b);
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);

        /* consumer scf.if */
        rocke_if_t cif = rocke_b_scf_if(b, is_consumer);
        rocke_b_region_enter(b, cif.then_region);
        {
            rocke_iter_arg_t loop_args[ROCKE_WSP3_MAX_ACCS];
            char namebuf[ROCKE_WSP3_MAX_ACCS][24];
            int i;
            /* Python: accs0 = [_emit_zero_acc_op(b, op) for _ in range(N)] --
             * one DISTINCT zero-acc Value per accumulator. */
            for(i = 0; i < num_accs; ++i)
            {
                snprintf(namebuf[i], sizeof(namebuf[i]), "acc%d", i);
                loop_args[i].name = namebuf[i];
                loop_args[i].init = rocke_gemm_emit_zero_acc_op(b, op);
            }
            rocke_for_t cfor
                = rocke_b_scf_for_iter(b, c0, K, c_block_k, loop_args, num_accs, "kc", false, true);
            rocke_b_region_enter(b, cfor.body);
            {
                rocke_value_t* new_accs[ROCKE_WSP3_MAX_ACCS];
                rocke_b_sync(b);
                rocke_wsp3_consumer_mfma(&w, cfor.iter_vars, c0, c0, new_accs);
                rocke_b_sync(b);
                rocke_b_scf_yield(b, new_accs, num_accs);
            }
            rocke_b_region_leave(b);

            rocke_gemm_emit_epilogue_default(b,
                                             spec,
                                             op,
                                             cfor.op->results,
                                             num_accs,
                                             cwarp_m,
                                             cwarp_n,
                                             lane,
                                             block_m_off,
                                             block_n_off,
                                             M,
                                             N,
                                             C,
                                             c_per_lane,
                                             NULL,
                                             NULL,
                                             false);
        }
        rocke_b_region_leave(b);

        rocke_b_ret(b);
        return b->kernel;
    }

    /* depth>=2 ping-pong ring. */
    async_mode = rocke_gemm_env_eq("CK_WSP3_ASYNC", "0", "1") && async_ok;
    prologue_tiles = depth - 1;
    keep_vmcnt = (depth - 2) * loads_per_tile;
    if(rocke_gemm_env_eq("CK_WSP3_DRAIN", "", "1"))
    {
        keep_vmcnt = 0;
    }

    /* producer scf.if */
    {
        rocke_if_t pif = rocke_b_scf_if(b, is_producer);
        rocke_b_region_enter(b, pif.then_region);
        if(async_mode)
        {
            for(e = 0; e < prologue_tiles; ++e)
            {
                /* Python builds the 3 const args left-to-right. */
                rocke_value_t* k_arg = rocke_b_const_i32(b, e * block_k);
                rocke_value_t* a_arg = rocke_b_const_i32(b, e * a_buf_bytes);
                rocke_value_t* b_arg = rocke_b_const_i32(b, e * b_buf_bytes);
                rocke_wsp3_producer_load_async(&w, k_arg, a_arg, b_arg);
            }
        }
        else
        {
            for(e = 0; e < prologue_tiles; ++e)
            {
                /* Python builds the 3 const args left-to-right. */
                rocke_value_t* k_arg = rocke_b_const_i32(b, e * block_k);
                rocke_value_t* a_arg = rocke_b_const_i32(b, e * block_m);
                rocke_value_t* b_arg = rocke_b_const_i32(b, e * block_n);
                rocke_wsp3_producer_load(&w, k_arg, a_arg, b_arg);
            }
        }
        rocke_for_t pfor = rocke_b_scf_for(b, c0, K, c_block_k, "kp");
        rocke_b_region_enter(b, pfor.body);
        {
            rocke_value_t* knext
                = rocke_b_add(b, pfor.iv, rocke_b_const_i32(b, prologue_tiles * block_k));
            if(async_mode)
            {
                rocke_b_s_waitcnt(b, keep_vmcnt, -1, -1);
                rocke_b_s_barrier_bare(b);
                rocke_if_t nif = rocke_b_scf_if(b, rocke_b_cmp_lt(b, knext, K));
                rocke_b_region_enter(b, nif.then_region);
                {
                    rocke_value_t* tnext = rocke_b_div(b, knext, c_block_k);
                    rocke_value_t* wbuf = rocke_b_mod(b, tnext, cD);
                    /* Python evals the A buf-offset arg before the B one. */
                    rocke_value_t* a_arg = rocke_b_mul(b, wbuf, rocke_b_const_i32(b, a_buf_bytes));
                    rocke_value_t* b_arg = rocke_b_mul(b, wbuf, rocke_b_const_i32(b, b_buf_bytes));
                    rocke_wsp3_producer_load_async(&w, knext, a_arg, b_arg);
                }
                rocke_b_region_leave(b);
            }
            else
            {
                rocke_b_sync(b);
                rocke_if_t nif = rocke_b_scf_if(b, rocke_b_cmp_lt(b, knext, K));
                rocke_b_region_enter(b, nif.then_region);
                {
                    rocke_value_t* tnext = rocke_b_div(b, knext, c_block_k);
                    rocke_value_t* wbuf = rocke_b_mod(b, tnext, cD);
                    /* Python evals the A buf-row arg before the B one. */
                    rocke_value_t* a_arg = rocke_b_mul(b, wbuf, c_block_m);
                    rocke_value_t* b_arg = rocke_b_mul(b, wbuf, c_block_n);
                    rocke_wsp3_producer_load(&w, knext, a_arg, b_arg);
                }
                rocke_b_region_leave(b);
            }
        }
        rocke_b_region_leave(b);
        rocke_b_region_leave(b);
    }

    /* consumer scf.if */
    {
        rocke_if_t cif = rocke_b_scf_if(b, is_consumer);
        rocke_b_region_enter(b, cif.then_region);
        {
            rocke_iter_arg_t loop_args[ROCKE_WSP3_MAX_ACCS];
            char namebuf[ROCKE_WSP3_MAX_ACCS][24];
            int i;
            /* Python: one DISTINCT zero-acc Value per accumulator. */
            for(i = 0; i < num_accs; ++i)
            {
                snprintf(namebuf[i], sizeof(namebuf[i]), "acc%d", i);
                loop_args[i].name = namebuf[i];
                loop_args[i].init = rocke_gemm_emit_zero_acc_op(b, op);
            }
            rocke_for_t cfor
                = rocke_b_scf_for_iter(b, c0, K, c_block_k, loop_args, num_accs, "kc", false, true);
            rocke_b_region_enter(b, cfor.body);
            {
                rocke_value_t* new_accs[ROCKE_WSP3_MAX_ACCS];
                if(async_mode)
                {
                    rocke_b_s_barrier_bare(b);
                }
                else
                {
                    rocke_b_sync(b);
                }
                rocke_value_t* tcur = rocke_b_div(b, cfor.iv, c_block_k);
                rocke_value_t* rbuf = rocke_b_mod(b, tcur, cD);
                /* Python evals the A buf-row arg before the B one. */
                rocke_value_t* a_row = rocke_b_mul(b, rbuf, c_block_m);
                rocke_value_t* b_row = rocke_b_mul(b, rbuf, c_block_n);
                rocke_wsp3_consumer_mfma(&w, cfor.iter_vars, a_row, b_row, new_accs);
                rocke_b_scf_yield(b, new_accs, num_accs);
            }
            rocke_b_region_leave(b);

            rocke_gemm_emit_epilogue_default(b,
                                             spec,
                                             op,
                                             cfor.op->results,
                                             num_accs,
                                             cwarp_m,
                                             cwarp_n,
                                             lane,
                                             block_m_off,
                                             block_n_off,
                                             M,
                                             N,
                                             C,
                                             c_per_lane,
                                             NULL,
                                             NULL,
                                             false);
        }
        rocke_b_region_leave(b);
    }

    rocke_b_ret(b);
    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_universal_gemm -- THE DRIVER
 *
 *  Populates ctx in the exact order of the Python prologue (lines 737-1110),
 *  sets up the active-tile gate (1609-1618), then calls
 *  rocke_gemm_emit_compute_and_epilogue either directly or under an scf.if gate
 *  (1825-1829), and returns b->kernel.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_universal_gemm(rocke_ir_builder_t* b,
                                               const rocke_gemm_universal_spec_t* spec,
                                               const char* arch)
{
    rocke_gemm_build_ctx_t ctx;
    const rocke_gemm_tile_spec_t* t;
    char reason[ROCKE_ERR_MSG_CAP];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* wsp3 lives in a separate emitter (gemm_wsp3.py). The C port of that
     * emitter is a peer module; route to it here for parity. */
    if(spec->trait.pipeline != NULL && strcmp(spec->trait.pipeline, "wsp3") == 0)
    {
        return rocke_build_wsp3_gemm(b, spec, arch);
    }

    /* is_valid_spec(spec, arch). Python raises ValueError on reject. */
    if(!rocke_gemm_universal_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return NULL;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;
    ctx.target = rocke_archtarget_from_gfx(arch);
    /* the prologue emitter is owned by this build */
    struct pe_owner
    {
        rocke_gemm_build_ctx_t* c;
        ~pe_owner()
        {
            if(c->pe != NULL)
                rocke_gemm_pro_emitter_free(c->pe);
            c->pe = NULL;
        }
    } pe_guard{&ctx};

    ctx.op = rocke_gemm_resolve_mma_op(spec, arch, ctx.target);
    if(ctx.op == NULL)
    {
        /* is_valid_spec already rejects this; mirror the Python guard. */
        return NULL;
    }
    ctx.is_wmma = (ctx.op->family != NULL && strcmp(ctx.op->family, "wmma") == 0);

    /* fused operand prologue (side channel spec->_prologue, see gemm_prologue):
     * attach_prologue's check(), then the Python builder's guards */
    ctx.pro = spec->_prologue;
    if(ctx.pro != NULL)
    {
        if(!rocke_gemm_prologue_check(ctx.pro, reason, sizeof(reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
            return NULL;
        }
        if(ctx.is_wmma)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "prologue: MFMA bodies only");
            return NULL;
        }
        if(ctx.pro->out_f32 && strcmp(spec->trait.epilogue, "default") != 0)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "prologue out_f32 needs the default epilogue");
            return NULL;
        }
        if(spec->trait.split_k > 1)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "prologue: split_k unsupported");
            return NULL;
        }
        if(strcmp(spec->data.layout, "RCR") != 0 || spec->_gett != NULL)
        {
            /* MN-contiguous operand / GETT fragments bypass the prologue's
             * fragment transform (they would be used untransformed) */
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "prologue: K-contiguous (RCR) operands only");
            return NULL;
        }
    }
    if(spec->_gett != NULL && spec->_fused_epilogue != NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "the GETT epilogue does not support a fused epilogue");
        return NULL;
    }

    /* Python build_universal_gemm opens with `b = IRBuilder(spec.kernel_name())`,
     * so the emitted kernel symbol is ALWAYS spec.kernel_name() -- derived from
     * spec.name here, not whatever name the caller pre-initialised the builder
     * with. For the plain-GEMM entry the caller already used this exact name (a
     * no-op rename); for the multi-D / multi-ABD delegation the base spec was
     * renamed to the multi-D/ABD kernel_name(), so re-deriving here appends the
     * universal suffix on top -- reproducing the doubled symbol Python emits. */
    {
        char uni_name[1024];
        if(rocke_gemm_universal_kernel_name(spec, uni_name, sizeof(uni_name)) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_universal_gemm: kernel_name() overflow");
            return NULL;
        }
        if(ctx.pro != NULL)
        {
            /* Python: IRBuilder(spec.kernel_name() + _pro.suffix()) */
            size_t used = strlen(uni_name);
            if(rocke_gemm_prologue_suffix(ctx.pro, uni_name + used, sizeof(uni_name) - used) != 0)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_universal_gemm: kernel_name() overflow");
                return NULL;
            }
        }
        b->kernel->name = rocke_arena_strdup(&b->arena, uni_name);
        if(b->kernel->name == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "build_universal_gemm: kernel name OOM");
            return NULL;
        }
    }

    /* Set the kernel descriptor attrs the Python prologue bakes in. */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    if(spec->trait.waves_per_eu_set)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }
    /* (Python: elif spec.trait.acc_agpr after the
     * ROCKE_EXP_AGPR_ALLOC experiment, which the C++ engine does not read):
     * reserve exactly the accumulator registers as AGPRs. */
    if(spec->trait.acc_agpr)
    {
        int64_t n_acc = (int64_t)rocke_gemm_acc_agpr_regs(spec);
        int64_t agpr_pair[2];
        agpr_pair[0] = n_acc;
        agpr_pair[1] = n_acc;
        rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", agpr_pair, 2);
        /* Python helpers.compile turns this into -amdgpu-mfma-vgpr-form=0. */
        rocke_attr_set_bool(b, &b->kernel->attrs, "mfma_agpr_form", true);
    }

    ctx.storage_dtype = rocke_gemm_storage_dtype(b, spec);
    if(ctx.storage_dtype == NULL)
    {
        return NULL;
    }
    /* native NN / TN / TT (gemm_mn_operand): the LDS block layout of each
     * MN-contiguous operand (has_mn_* false: K-contiguous, the RCR path) */
    if(!rocke_gemm_spec_mn_layouts(
           spec, &ctx.mn_lay_a, &ctx.has_mn_a, &ctx.mn_lay_b, &ctx.has_mn_b, reason, sizeof reason))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
        return NULL;
    }
    /* _gett (native GETT, rocke/instance_tensor_gett.h): a side channel like
     * _fused_epilogue; the MN-layout hook keeps the MN operand path's LDS
     * stages and replaces only the operand addressing and the epilogue, the
     * ring-body hook owns all four */
    ctx.gett = const_cast<struct rocke_gett_hooks*>(spec->_gett);
    ctx.gett_addr = ctx.gett != NULL && ctx.gett->addr;
    ctx.gett_own_lds = ctx.gett_addr && !ctx.gett->mn;

    /* ---- kernel params (Values) -- */
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_storage = rocke_ptr_type(b, ctx.storage_dtype, "global");

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx.A = rocke_b_param(b, "A", ptr_storage, &opts);
        ctx.Bp = rocke_b_param(b, "B", ptr_storage, &opts);

        /* In split-K mode the output is the f32 accumulation workspace the
         * epilogue atomic-adds into (the caller casts it afterwards). It is
         * read+write (atomicrmw), so drop writeonly. Outside split-K the C
         * param is byte-identical to before. */
        if(spec->trait.split_k > 1)
        {
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.align = 4;
            opts.align_set = true;
            ctx.C = rocke_b_param(b, "Cf32", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
        }
        else
        {
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            /* prologue out_f32: the f32 accumulators are the output */
            ctx.C = rocke_b_param(b,
                                  "C",
                                  (ctx.pro != NULL && ctx.pro->out_f32)
                                      ? rocke_ptr_type(b, rocke_f32(), "global")
                                      : ptr_storage,
                                  &opts);
        }

        ctx.M = rocke_b_param(b, "M", rocke_i32(), NULL);
        ctx.N = rocke_b_param(b, "N", rocke_i32(), NULL);
        ctx.K = rocke_b_param(b, "K", rocke_i32(), NULL);
        if(ctx.gett != NULL)
        {
            ctx.gett->add_params(b);
        }
        if(spec->trait.streamk)
        {
            /* stream-K fixup workspace (Python streamk_workspace_bytes) */
            rocke_param_opts_t fo;
            memset(&fo, 0, sizeof(fo));
            fo.align = 4;
            fo.align_set = true;
            ctx.sk_flags
                = rocke_b_param(b, "SkFlags", rocke_ptr_type(b, rocke_i32(), "global"), &fo);
            memset(&fo, 0, sizeof(fo));
            fo.noalias = true;
            fo.noalias_set = true;
            fo.align = 16;
            fo.align_set = true;
            ctx.sk_part = rocke_b_param(b, "SkPart", rocke_ptr_type(b, rocke_f32(), "global"), &fo);
        }

        if(spec->batched)
        {
            ctx.stride_a = rocke_b_param(b, "stride_a", rocke_i32(), NULL);
            ctx.stride_b = rocke_b_param(b, "stride_b", rocke_i32(), NULL);
            ctx.stride_c = rocke_b_param(b, "stride_c", rocke_i32(), NULL);
        }
        if(spec->batched && spec->trait.active_tile_skip)
        {
            rocke_param_opts_t sopts;
            memset(&sopts, 0, sizeof(sopts));
            sopts.noalias = true;
            sopts.noalias_set = true;
            sopts.readonly = true;
            sopts.readonly_set = true;
            sopts.align = 4;
            sopts.align_set = true;
            ctx.sorted_token_ids = rocke_b_param(
                b, "SortedTokenIds", rocke_ptr_type(b, rocke_i32(), "global"), &sopts);
            ctx.slot_size_p = rocke_b_param(b, "slot_size", rocke_i32(), NULL);
        }
    }

    t = &spec->tile;

    /* ---- per-lane MMA fragment widths -- */
    rocke_gemm_atom_frag_lengths(ctx.op, &ctx.a_per_lane, &ctx.b_per_lane, &ctx.c_per_lane);

    /* ---- block tile dims (aliases) -- */
    ctx.block_m = t->tile_m;
    ctx.block_n = t->tile_n;
    ctx.block_k = t->tile_k;

    /* ---- common geometry constants (SSA) -- */
    ctx.c0 = rocke_b_const_i32(b, 0);

    /* ---- split-K K-slice bounds -- *
     * Each CTA along block_id_z owns the slice [z*ks, (z+1)*ks) where
     * ks = K // split_k; the K-loop runs over that slice instead of [0, K).
     * For split_k == 1 these collapse to k_lo = c0 / k_upper = K so the
     * non-split body stays byte-identical (Python: k_hi None -> _k_upper = K). */
    ctx.split_k = spec->trait.split_k;
    ctx.is_split_k = ctx.split_k > 1;
    if(ctx.is_split_k)
    {
        rocke_value_t* ks = rocke_b_div(b, ctx.K, rocke_b_const_i32(b, ctx.split_k));
        ctx.k_lo = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_z(b), ks));
        ctx.k_upper = rocke_b_to_sgpr_u32(b, rocke_b_add(b, ctx.k_lo, ks));
    }
    else
    {
        ctx.k_lo = ctx.c0;
        ctx.k_upper = ctx.K;
    }

    ctx.c_wave = rocke_b_const_i32(b, spec->wave_size);
    ctx.c_warps_n = rocke_b_const_i32(b, t->warp_n);
    ctx.c_block_m = rocke_b_const_i32(b, ctx.block_m);
    ctx.c_block_n = rocke_b_const_i32(b, ctx.block_n);
    ctx.c_block_k = rocke_b_const_i32(b, ctx.block_k);

    /* ---- thread / warp / lane decode (SSA) -- */
    ctx.tid = rocke_b_thread_id_x(b);
    ctx.warp_id = rocke_b_div(b, ctx.tid, ctx.c_wave);
    ctx.warp_m_idx = rocke_b_div(b, ctx.warp_id, ctx.c_warps_n);
    ctx.warp_n_idx = rocke_b_mod(b, ctx.warp_id, ctx.c_warps_n);
    ctx.lane = rocke_b_mod(b, ctx.tid, ctx.c_wave);

    /* ---- compv4/compv3 schedule-hint policy (_sched_hints) -- *
     * explicit trait override wins; the default (None / not-set) takes the
     * measured gfx950 uplift (hints OFF) and preserves the historical
     * emission on every other arch. */
    ctx.sched_hints = spec->trait.emit_sched_hints_set
                          ? spec->trait.emit_sched_hints
                          : (ctx.arch == NULL || strcmp(ctx.arch, "gfx950") != 0);
    /* interwave_setprio: per-K-atom MFMA cluster priority bookends (opt-in). */
    ctx.interwave = spec->trait.interwave_setprio;

    /* ---- LDS XOR swizzle parameters -- */
    ctx.swz = spec->trait.lds_swizzle;
    {
        /* lds_swizzle_params (env-free single source shared with the gate);
         * the CK_SWZ_* env vars are debug overrides that bypass the gate. */
        int def_r, def_w, def_l;
        rocke_gemm_lds_swizzle_params(
            ctx.a_per_lane, ctx.b_per_lane, ctx.block_k, &def_r, &def_w, &def_l);
        if(spec->trait.lds_swizzle_r >= 0)
            def_r = spec->trait.lds_swizzle_r; /* AGPR / interleave knob */
        ctx.swz_r = rocke_gemm_env_int("CK_SWZ_R", def_r);
        ctx.swz_w = rocke_gemm_env_int("CK_SWZ_W", def_w);
        ctx.swz_l = rocke_gemm_env_int("CK_SWZ_L", def_l);
        ctx.c_swr = rocke_b_const_i32(b, ctx.swz_r);
        ctx.c_swmod = rocke_b_const_i32(b, (int64_t)1 << ctx.swz_w);
        ctx.c_swl = rocke_b_const_i32(b, ctx.swz_l);
    }

    /* ---- batch-axis pointer offsets (SSA) -- */
    if(spec->batched)
    {
        ctx.batch_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_z(b));
        ctx.batch_off_a = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx.batch_idx, ctx.stride_a));
        ctx.batch_off_b = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx.batch_idx, ctx.stride_b));
        ctx.batch_off_c = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, ctx.batch_idx, ctx.stride_c));
    }
    else
    {
        ctx.batch_off_a = ctx.c0;
        ctx.batch_off_b = ctx.c0;
        ctx.batch_off_c = ctx.c0;
    }

    /* ---- per-CTA tile origins (SGPR-pinned) -- */
    if(spec->trait.chiplet_swizzle)
    {
        rocke_value_t* n_pid_m = rocke_b_div(
            b, rocke_b_add(b, ctx.M, rocke_b_const_i32(b, ctx.block_m - 1)), ctx.c_block_m);
        rocke_value_t* n_pid_n = rocke_b_div(
            b, rocke_b_add(b, ctx.N, rocke_b_const_i32(b, ctx.block_n - 1)), ctx.c_block_n);
        /* Python: b.add(b.mul(b.block_id_y(), n_pid_n), b.block_id_x()).
         * Python evaluates the add's first arg (block_id_y -> mul) fully before
         * its second arg (block_id_x), so the SSA order is y, mul, x, add. C's
         * argument evaluation order is unspecified, so pin the order with
         * explicit temporaries to match the Python emission. */
        rocke_value_t* wgid_mul = rocke_b_mul(b, rocke_b_block_id_y(b), n_pid_n);
        rocke_value_t* wgid_bx = rocke_b_block_id_x(b);
        rocke_value_t* wgid_flat = rocke_b_add(b, wgid_mul, wgid_bx);
        rocke_super_tile_swizzle_result_t swz
            = rocke_chiplet_aware_super_tile_dynamic(b,
                                                     wgid_flat,
                                                     n_pid_m,
                                                     n_pid_n,
                                                     spec->trait.chiplet_wgm,
                                                     spec->trait.chiplet_num_xcds,
                                                     spec->trait.chiplet_chunk_size);
        ctx.block_m_off = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, swz.row, ctx.c_block_m));
        ctx.block_n_off = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, swz.col, ctx.c_block_n));
    }
    else
    {
        ctx.block_m_off
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_y(b), ctx.c_block_m));
        ctx.block_n_off
            = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_x(b), ctx.c_block_n));
    }
    /* mn_ld_any: the last tile along an MN-contiguous operand's dimension ends
     * at the extent (origin ld - tile) */
    if(spec->trait.mn_ld_any && ctx.has_mn_a)
    {
        rocke_value_t* ct = rocke_b_const_i32(b, ctx.block_m);
        rocke_value_t* lim = rocke_b_sub(b, ctx.M, ct);
        rocke_value_t* mn = rocke_b_smin(b, ctx.block_m_off, lim);
        ctx.block_m_off = rocke_b_to_sgpr_u32(b, mn);
    }
    if(spec->trait.mn_ld_any && ctx.has_mn_b)
    {
        rocke_value_t* ct = rocke_b_const_i32(b, ctx.block_n);
        rocke_value_t* lim = rocke_b_sub(b, ctx.N, ct);
        rocke_value_t* mn = rocke_b_smin(b, ctx.block_n_off, lim);
        ctx.block_n_off = rocke_b_to_sgpr_u32(b, mn);
    }

    /* ---- AB LDS double-buffer plan (_ab_lds_plan) -- */
    ctx.prefetch = spec->trait.dtl_prefetch;
    if(ctx.prefetch && !spec->trait.direct_to_lds)
    {
        /* "dtl_prefetch requires direct_to_lds=True" -- ValueError path. */
        return NULL;
    }
    {
        /* _ab_lds_plan(spec, arch) -> (db, two_buf), mirroring
         * gemm_universal.py:_ab_lds_plan:
         *
         *   db      = ab_double_buffer_decision(...)  (rocke_gemm_ab_db_decision:
         *             explicit ab_double_buffer, else the historical rule
         *             compv4 && epilogue!=cshuffle && !direct_to_lds
         *             && (2*ab_single)*2 <= lds_capacity_bytes)
         *   two_buf = dtl_prefetch || db
         *
         * One decision function shared with the validity gate and the load-path
         * helper (gemm_build_load.cpp). */
        bool db = false;
        bool two_buf = false;
        db = rocke_gemm_ab_db_decision(spec, ctx.target);
        two_buf = (spec->trait.dtl_prefetch ? true : false) || db;
        ctx.db = db;
        ctx.two_buf = two_buf;
    }
    ctx.A_LDS_M = (ctx.two_buf ? 2 : 1) * ctx.block_m;
    ctx.B_LDS_N = (ctx.two_buf ? 2 : 1) * ctx.block_n;
    ctx.lds_pad = spec->trait.direct_to_lds ? 0 : spec->trait.lds_k_pad;
    ctx.lds_k = ctx.block_k + ctx.lds_pad;
    /* dtl_stages >= 2 : one A and one B allocation per stage +
     * LDS alias scopes. */
    ctx.stages = (ctx.prefetch && spec->trait.direct_to_lds) ? spec->trait.dtl_stages : 0;
    ctx.swap_ab = spec->trait.mfma_swap_ab;
    ctx.staged_persistent_done = false;
    ctx.rilv = spec->trait.lds_row_interleave;
    if(ctx.stages >= 2)
    {
        static const char* const a_names[4] = {"A_smem_s0", "A_smem_s1", "A_smem_s2", "A_smem_s3"};
        static const char* const b_names[4] = {"B_smem_s0", "B_smem_s1", "B_smem_s2", "B_smem_s3"};
        int a_shape[2] = {ctx.block_m, ctx.block_k};
        int b_shape[2] = {ctx.block_n, ctx.block_k};
        int j;
        if(ctx.gett_own_lds)
        {
            /* GETT: K-outer stages in the hook's padded block layout */
            ctx.gett->lds_shape('A', a_shape);
            ctx.gett->lds_shape('B', b_shape);
        }
        else if(ctx.rilv)
        {
            /* lds_row_interleave: rows of 8 x block_k elements + rilv_pad bytes of
             * pad (one DTL instruction per wave fills exactly one such row; an
             * MN-contiguous operand keeps its gemm_mn_operand block layout) */
            int rpad = spec->trait.rilv_pad ? spec->trait.rilv_pad : 16;
            ctx.rilv_cols = 8 * ctx.block_k + rpad / (rocke_i_type_bits(ctx.storage_dtype) / 8);
            a_shape[0] = ctx.block_m / 8;
            a_shape[1] = ctx.rilv_cols;
            b_shape[0] = ctx.block_n / 8;
            b_shape[1] = ctx.rilv_cols;
        }
        if(!ctx.gett_own_lds)
        {
            /* MN-contiguous operand (layout RRR / CRR / CCR): its stage is the
             * gemm_mn_operand block layout (1 KiB blocks + mn_pad bytes) */
            if(ctx.has_mn_a)
            {
                a_shape[0] = rocke_gemm_mn_n_blocks(&ctx.mn_lay_a);
                a_shape[1] = rocke_gemm_mn_block_bytes(&ctx.mn_lay_a) / 2;
            }
            if(ctx.has_mn_b)
            {
                b_shape[0] = rocke_gemm_mn_n_blocks(&ctx.mn_lay_b);
                b_shape[1] = rocke_gemm_mn_block_bytes(&ctx.mn_lay_b) / 2;
            }
        }
        if(ctx.stages > 4)
        {
            return NULL;
        }
        for(j = 0; j < ctx.stages; ++j)
            ctx.A_stg[j] = rocke_b_smem_alloc(b, ctx.storage_dtype, a_shape, 2, a_names[j]);
        for(j = 0; j < ctx.stages; ++j)
            ctx.B_stg[j] = rocke_b_smem_alloc(b, ctx.storage_dtype, b_shape, 2, b_names[j]);
        ctx.A_smem = ctx.A_stg[0];
        ctx.B_smem = ctx.B_stg[0];
        rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    }
    else
    {
        int a_shape[2] = {ctx.A_LDS_M, ctx.lds_k};
        int b_shape[2] = {ctx.B_LDS_N, ctx.lds_k};
        ctx.A_smem = rocke_b_smem_alloc(b, ctx.storage_dtype, a_shape, 2, "A_smem");
        ctx.B_smem = rocke_b_smem_alloc(b, ctx.storage_dtype, b_shape, 2, "B_smem");
    }

    /* ---- per-warp MFMA tile counts -- */
    ctx.mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    ctx.mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    ctx.k_atoms = rocke_gemm_tile_k_atoms_per_tile_k(t);
    ctx.kmul = 1;
    ctx.pe = NULL;
    if(ctx.pro != NULL)
    {
        int a_pl = 0, b_pl = 0, c_pl = 0;
        ctx.kmul = rocke_gemm_prologue_kmul(ctx.pro);
        if(ctx.k_atoms % ctx.kmul)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "prologue src=f32: tile_k (%d halves) must hold whole f32 atoms",
                            ctx.block_k);
            return NULL;
        }
        ctx.k_atoms /= ctx.kmul;
        rocke_gemm_atom_frag_lengths(ctx.op, &a_pl, &b_pl, &c_pl);
        if(a_pl != b_pl)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "prologue: a_per_lane != b_per_lane");
            return NULL;
        }
        ctx.pe = rocke_gemm_pro_emitter_new(b, ctx.pro, ctx.storage_dtype, a_pl);
    }
    /* complex B (real embedding, see Prologue.cplx_b): tile row r of B' is
     * complex row r >> 1 */
    ctx.cplx = ctx.pe != NULL && ctx.pro->cplx_b;
    if(ctx.cplx)
    {
        if(!spec->trait.direct_to_lds)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "cplx_b needs the direct-to-LDS ring");
            return NULL;
        }
        if(ctx.rilv && !spec->trait.rilv_vec_epilogue)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "cplx_b + lds_row_interleave needs rilv_vec_epilogue (linear rows)");
            return NULL;
        }
        if(ctx.rilv && (ctx.mfmas_n % 2))
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "cplx_b + lds_row_interleave needs an even mfmas_per_warp_n");
            return NULL;
        }
        if(ctx.rilv)
        {
            rocke_gemm_pro_set_cplx_ni(ctx.pe);
        }
        else
        {
            rocke_value_t* c1 = rocke_b_const_i32(b, 1);
            rocke_value_t* la = rocke_b_land(b, ctx.lane, c1);
            rocke_value_t* cz = rocke_b_const_i32(b, 0);
            rocke_gemm_pro_set_cplx_lane(ctx.pe, rocke_b_cmp_ne(b, la, cz));
        }
    }

    /* ---- accumulators -- */
    ctx.acc_init = rocke_gemm_emit_zero_acc_op(b, ctx.op);
    if(spec->trait.acc_agpr)
    {
        /* opaque AGPR zero seed (Python TraitSpec.acc_agpr): keeps the
         * loop-carried accumulator phis whole vectors. */
        rocke_inline_asm_opts_t opts;
        const rocke_type_t* acc_ty = ctx.acc_init->type;
        rocke_op_t* asm_op;
        memset(&opts, 0, sizeof opts);
        opts.sideeffect = false;
        opts.sideeffect_set = true;
        asm_op = rocke_b_inline_asm(b, "", "=a,0", &ctx.acc_init, 1, &acc_ty, 1, &opts);
        if(asm_op == NULL)
        {
            return NULL;
        }
        ctx.acc_init = asm_op->results[0];
    }
    ctx.num_accs = ctx.mfmas_m * ctx.mfmas_n;
    {
        int idx = 0;
        int mi, ni;
        for(mi = 0; mi < ctx.mfmas_m; ++mi)
        {
            for(ni = 0; ni < ctx.mfmas_n; ++ni)
            {
                char* nm;
                if(idx >= ROCKE_GEMM_MAX_ACCS)
                {
                    return NULL; /* config exceeds the bounded acc array */
                }
                /* "acc_m{mi}_n{ni}" -- arena-stable name. */
                nm = (char*)rocke_arena_alloc(&b->arena, 24);
                if(nm == NULL)
                {
                    return NULL;
                }
                /* snprintf-free formatting to keep the dependency surface tiny. */
                {
                    int p = 0;
                    const char* pre = "acc_m";
                    while(*pre)
                    {
                        nm[p++] = *pre++;
                    }
                    p += rocke_gemm_itoa(mi, nm + p);
                    nm[p++] = '_';
                    nm[p++] = 'n';
                    p += rocke_gemm_itoa(ni, nm + p);
                    nm[p] = '\0';
                }
                ctx.acc_names[idx] = nm;
                ctx.acc_inits[idx] = ctx.acc_init;
                ++idx;
            }
        }
    }

    /* ---- global -> LDS coalesced copy plan -- */
    ctx.threads = spec->block_size;
    ctx.load_vec = rocke_gemm_choose_load_vec(spec);
    if(ctx.load_vec == 0)
    {
        /* choose_load_vec found no usable width: Python raises
         * ValueError("no usable load_vec for tile_m=.. tile_n=.. tile_k=..
         * block_size=.."). Reproduce the same rejection (no IR, sticky error)
         * instead of dividing by zero below. */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no usable load_vec for tile_m=%d tile_n=%d tile_k=%d block_size=%d",
                        spec->tile.tile_m,
                        spec->tile.tile_n,
                        spec->tile.tile_k,
                        spec->block_size);
        return NULL;
    }
    ctx.a_total = ctx.block_m * ctx.block_k;
    ctx.b_total = ctx.block_n * ctx.block_k;
    ctx.a_vec_total = ctx.a_total / ctx.load_vec;
    ctx.b_vec_total = ctx.b_total / ctx.load_vec;
    ctx.a_vecs_per_thread = ctx.a_vec_total / ctx.threads;
    ctx.b_vecs_per_thread = ctx.b_vec_total / ctx.threads;
    ctx.c_threads = rocke_b_const_i32(b, ctx.threads);
    ctx.c_load_vec = rocke_b_const_i32(b, ctx.load_vec);
    ctx.c_block_k_div_vec = rocke_b_const_i32(b, ctx.block_k / ctx.load_vec);

    /* ---- CK-Tile data views (host-side; hold SSA bases) -- */
    {
        int gshape[3] = {1, 1, 1};
        rocke_stride_t gstride[3];
        gstride[0] = rocke_stride_imm(1);
        gstride[1] = rocke_stride_value(ctx.K);
        gstride[2] = rocke_stride_imm(1);
        rocke_make_global_view(&ctx.a_view, ctx.A, gshape, 3, ctx.storage_dtype, gstride);
        rocke_make_global_view(&ctx.b_view, ctx.Bp, gshape, 3, ctx.storage_dtype, gstride);
    }
    {
        int a2[2] = {ctx.block_m, ctx.block_k};
        int b2[2] = {ctx.block_n, ctx.block_k};
        if(ctx.lds_pad)
        {
            rocke_stride_t as[2];
            rocke_stride_t bs[2];
            as[0] = rocke_stride_imm(ctx.lds_k);
            as[1] = rocke_stride_imm(1);
            bs[0] = rocke_stride_imm(ctx.lds_k);
            bs[1] = rocke_stride_imm(1);
            rocke_tensor_descriptor_with_strides(&ctx.a_lds_desc, a2, as, 2, ctx.storage_dtype);
            rocke_tensor_descriptor_with_strides(&ctx.b_lds_desc, b2, bs, 2, ctx.storage_dtype);
        }
        else
        {
            rocke_tensor_descriptor_packed(&ctx.a_lds_desc, a2, 2, ctx.storage_dtype);
            rocke_tensor_descriptor_packed(&ctx.b_lds_desc, b2, 2, ctx.storage_dtype);
        }
        ctx.a_lds_view.base = ctx.A_smem;
        ctx.a_lds_view.desc = ctx.a_lds_desc;
        ctx.a_lds_view.addr_space = ROCKE_ADDR_LDS;
        ctx.b_lds_view.base = ctx.B_smem;
        ctx.b_lds_view.desc = ctx.b_lds_desc;
        ctx.b_lds_view.addr_space = ROCKE_ADDR_LDS;
    }

    /* ---- DirectToLDS plumbing -- */
    ctx.dtl = spec->trait.direct_to_lds;
    if(ctx.dtl)
    {
        /* Capped at the arch's own buffer_load_lds width (is_valid_spec has
         * already checked tile_k divides it): CDNA4 (gfx950) has the full
         * dword=4 form, CDNA3 (gfx942) only dword=1. */
        ctx.dtl_dwords = rocke_archtarget_async_lds_max_dwords(ctx.target);
        if(ctx.dtl_dwords > 4)
        {
            ctx.dtl_dwords = 4;
        }
        ctx.dtl_halves = ctx.dtl_dwords * 2;
        ctx.dtl_bytes_per_lane = ctx.dtl_dwords * 4;
        if((ctx.block_k % ctx.dtl_halves) != 0)
        {
            /* "direct_to_lds requires block_k % 8 == 0" -- ValueError path. */
            return NULL;
        }
        ctx.dtl_a_chunks = (ctx.block_m * ctx.block_k) / ctx.dtl_halves;
        ctx.dtl_b_chunks = (ctx.block_n * ctx.block_k) / ctx.dtl_halves;
        ctx.dtl_a_passes = (ctx.dtl_a_chunks + spec->block_size - 1) / spec->block_size;
        ctx.dtl_b_passes = (ctx.dtl_b_chunks + spec->block_size - 1) / spec->block_size;
        ctx.dtl_pass_bytes = spec->block_size * ctx.dtl_bytes_per_lane;
        ctx.dtl_chunks_per_row = ctx.block_k / ctx.dtl_halves;

        ctx.dtl_big_bytes = rocke_b_const_i32(b, ROCKE_GEMM_DTL_BUFFER_NUM_RECORDS);
        ctx.dtl_a_rsrc = rocke_b_buffer_rsrc(b, ctx.A, ctx.dtl_big_bytes);
        ctx.dtl_b_rsrc = rocke_b_buffer_rsrc(b, ctx.Bp, ctx.dtl_big_bytes);
        ctx.dtl_a_lds_base = rocke_b_smem_addr_of(b, ctx.A_smem);
        ctx.dtl_b_lds_base = rocke_b_smem_addr_of(b, ctx.B_smem);
        ctx.dtl_zero_soff = rocke_b_const_i32(b, 0);
        ctx.dtl_c_chunks_per_row = rocke_b_const_i32(b, ctx.dtl_chunks_per_row);
        ctx.dtl_c_halves_per_chunk = rocke_b_const_i32(b, ctx.dtl_halves);
        ctx.dtl_c_block_size = rocke_b_const_i32(b, spec->block_size);
    }

    /* ---- active-tile gate (batched && active_tile_skip) -- */
    ctx.do_work_cond = NULL;
    if(spec->batched && spec->trait.active_tile_skip)
    {
        rocke_value_t* bucket_head = rocke_b_add(
            b, rocke_b_mul(b, rocke_b_block_id_z(b), ctx.slot_size_p), ctx.block_m_off);
        rocke_value_t* first_token
            = rocke_b_global_load_i32(b, ctx.sorted_token_ids, bucket_head, 0);
        ctx.do_work_cond = rocke_b_cmp_ge(b, first_token, ctx.c0);
    }

    /* ---- K-loop result accumulators init -- */
    ctx.num_for_results = 0;

    /* ---- dispatch: emit_compute_and_epilogue, optionally under the gate -- */
    if(ctx.do_work_cond == NULL)
    {
        rocke_gemm_emit_compute_and_epilogue(&ctx);
    }
    else
    {
        rocke_if_t gate = rocke_b_scf_if(b, ctx.do_work_cond);
        rocke_b_region_enter(b, gate.then_region);
        rocke_gemm_emit_compute_and_epilogue(&ctx);
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_universal_gemm_new -- init the builder with spec.kernel_name()
 *  then build. (public convenience.)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_universal_gemm_new(rocke_ir_builder_t* b,
                                                   const rocke_gemm_universal_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_gemm_universal_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_universal_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_gemm_universal_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_gemm_universal_lower_to_llvm(const rocke_gemm_universal_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_universal_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_universal_gemm failed";
            }
            n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  all_dispatcher_configs
 *
 *  The Python ``all_dispatcher_configs(...)`` is a *generator* that yields every
 *  valid UniversalGemmSpec over the dispatcher's config-space product. It is a
 *  host-side enumerator -- it emits NO IR and is not part of the kernel-build
 *  API. The public C header (rocke/instance_gemm_universal.h) does NOT declare a C
 *  signature for it (a streaming generator does not translate to a single C
 *  call), so there is no contract to bind to in this chunk.
 *
 *  NAMED GAP (blocked on a public-header symbol): when the dispatcher front-end
 *  needs this in C, the natural shape is a callback-driven walker:
 *
 *      typedef void (*rocke_gemm_config_fn)(const rocke_gemm_universal_spec_t*, void*);
 *      void rocke_gemm_all_dispatcher_configs(<axis arrays...>,
 *                                           const char* arch,
 *                                           rocke_gemm_config_fn cb, void* user);
 *
 *  enumerating the same nested product (tile_m x tile_n x tile_k x warp_m x
 *  warp_n x warp_k x warp_tile x pipeline x scheduler x epilogue x pad x
 *  persistent), finalising each spec and invoking the callback only for specs
 *  that pass rocke_gemm_universal_is_valid_spec. That entry point must be added to
 *  the PUBLIC header first (rocke/instance_gemm_universal.h), so it is left as a
 *  documented gap here rather than introducing an unheadered symbol. This is a
 *  host-side enumerator that emits NO IR, so it is off the byte-identity path.
 */

/* Self-contained file-local helper used by the acc-name formatting above.
 * Returns the number of chars written (excluding the NUL, which it does not
 * write). */
static int rocke_gemm_itoa(int v, char* out)
{
    char tmp[16];
    int n = 0;
    int p = 0;
    if(v == 0)
    {
        out[0] = '0';
        return 1;
    }
    if(v < 0)
    {
        out[p++] = '-';
        v = -v;
    }
    while(v > 0)
    {
        tmp[n++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while(n > 0)
    {
        out[p++] = tmp[--n];
    }
    return p;
}
