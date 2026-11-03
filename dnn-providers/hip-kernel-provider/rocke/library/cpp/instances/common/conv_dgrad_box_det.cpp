/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ port of library/kernels/common/conv_dgrad_box_det.py
 * (build_dgrad_box_det_reduce): Stage 2 of the dgrad pp_box_det
 * deterministic split-K. Every operand is bound to a temporary in Python's
 * left-to-right evaluation order (the emitted IR is byte-compared).
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_dgrad_box_det.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom */
#include "rocke/helper_rocke.helpers.epilogues.h" /* CShuffleEpilogue, WarpGrid */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

rocke_dgrad_box_det_reduce_spec_t rocke_dgrad_box_det_reduce_spec_default(void)
{
    rocke_dgrad_box_det_reduce_spec_t s;
    memset(&s, 0, sizeof(s));
    s.dtype_d = "bf16";
    s.store_vec = 8;
    s.m_chunks = 1;
    s.wave_size = 64;
    s.msplit = 1;
    s.unroll = 1;
    s.wave_wg = false;
    s.slot_bytes = 4;
    s.load_aux = 0;
    s.pix_pre = false;
    s.name = "conv_dgrad_box_det_reduce";
    return s;
}

int rocke_dgrad_box_det_reduce_block_size(const rocke_dgrad_box_det_reduce_spec_t* spec)
{
    return spec->wave_wg ? spec->wave_size : spec->warp_m * spec->warp_n * spec->wave_size;
}

rocke_status_t rocke_dgrad_box_det_reduce_kernel_name(const rocke_dgrad_box_det_reduce_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap)
{
    char t[32], w[32], v[16], mch[16], ms[16], u[16], la[16];
    snprintf(t, sizeof(t), "t%dx%d", spec->tile_m, spec->tile_n);
    snprintf(w, sizeof(w), "w%dx%d", spec->warp_m, spec->warp_n);
    snprintf(v, sizeof(v), "v%d", spec->store_vec);
    snprintf(mch, sizeof(mch), "mch%d", spec->m_chunks);
    snprintf(ms, sizeof(ms), "ms%d", spec->msplit);
    snprintf(u, sizeof(u), "u%d", spec->unroll);
    snprintf(la, sizeof(la), "la%d", spec->load_aux);
    const char* parts[12] = {t,
                             w,
                             v,
                             mch,
                             ms,
                             u,
                             spec->wave_wg ? "wwg" : "",
                             spec->slot_bytes == 2 ? "sb2" : "",
                             spec->load_aux ? la : "",
                             spec->pix_pre ? "pp" : "",
                             spec->dtype_d ? spec->dtype_d : "bf16"};
    return rocke_kernel_name_join(spec->name, parts, 11, NULL, NULL, 0, out, out_cap, NULL);
}

namespace
{

/* WarpGrid.from_atom(atom 16x16x16, ...).bind(b, block_m_axis="y",
 * block_n_axis="x") with warp_k 1. */
rocke_warp_grid_t pbd_bind_grid(
    rocke_ir_builder_t* b, int tile_m, int tile_n, int warp_m, int warp_n, int wave_size)
{
    rocke_warp_grid_t g;
    memset(&g, 0, sizeof(g));
    g.tile_m = tile_m;
    g.tile_n = tile_n;
    g.tile_k = 16;
    g.warp_m = warp_m;
    g.warp_n = warp_n;
    g.warp_k = 1;
    g.warp_tile_m = 16;
    g.warp_tile_n = 16;
    g.warp_tile_k = 16;
    g.wave_size = wave_size;
    if(b->kernel)
        rocke_attr_set_int(
            b, &b->kernel->attrs, "max_workgroup_size", rocke_warp_grid_block_size(&g));
    rocke_value_t* wave = rocke_b_const_i32(b, wave_size);
    rocke_value_t* c_warps_n = rocke_b_const_i32(b, warp_n);
    (void)rocke_b_const_i32(b, warp_n * warp_m);
    rocke_value_t* c_tile_m = rocke_b_const_i32(b, tile_m);
    rocke_value_t* c_tile_n = rocke_b_const_i32(b, tile_n);
    (void)rocke_b_const_i32(b, 16);
    g.tid = rocke_b_thread_id_x(b);
    g.lane = rocke_b_mod(b, g.tid, wave);
    g.warp_id = rocke_b_div(b, g.tid, wave);
    g.warp_m_idx = rocke_b_div(b, g.warp_id, c_warps_n);
    g.warp_n_idx = rocke_b_mod(b, g.warp_id, c_warps_n);
    g.warp_k_idx = rocke_b_const_i32(b, 0);
    g.block_m_off = rocke_b_mul(b, rocke_b_block_id_y(b), c_tile_m);
    g.block_n_off = rocke_b_mul(b, rocke_b_block_id_x(b), c_tile_n);
    g.block_k_off = rocke_b_const_i32(b, 0);
    return g;
}

struct pbd_ctx
{
    const rocke_dgrad_box_det_reduce_spec_t* spec;
    rocke_value_t* ws_ptr;
    rocke_value_t* rows_word;
    rocke_value_t* m_valid;
    rocke_value_t* c_C;
    rocke_value_t* row0;
    rocke_value_t* c_g16;
    rocke_value_t* q_off;
    int mfm;
    /* pix_pre: (row, pixel) per store vector, consumed in order. */
    rocke_value_t* pre_r[256];
    rocke_value_t* pre_pix[256];
    int n_pre;
    int pre_next;
};

/* _row(m): sub-tile row -> the split tile's row. */
rocke_value_t* pbd_row(rocke_ir_builder_t* b, const pbd_ctx* c, rocke_value_t* m_val)
{
    const rocke_dgrad_box_det_reduce_spec_t* spec = c->spec;
    if(spec->wave_wg)
        return rocke_b_add(b, c->row0, m_val);
    if(spec->msplit == 1)
        return m_val;
    rocke_value_t* wm = rocke_b_div(b, m_val, c->c_g16);
    rocke_value_t* m1 = rocke_b_mul(b, wm, rocke_b_const_i32(b, c->mfm * 16));
    rocke_value_t* a1 = rocke_b_add(b, m1, c->q_off);
    rocke_value_t* m2 = rocke_b_mul(b, wm, c->c_g16);
    rocke_value_t* s2 = rocke_b_sub(b, m_val, m2);
    return rocke_b_add(b, a1, s2);
}

rocke_value_t* pbd_dx_addr(rocke_ir_builder_t* b,
                           rocke_value_t* m_val,
                           rocke_value_t* n_val,
                           rocke_value_t** out_valid,
                           void* user)
{
    pbd_ctx* c = (pbd_ctx*)user;
    rocke_value_t *r, *pix;
    if(c->pre_next < c->n_pre)
    {
        r = c->pre_r[c->pre_next];
        pix = c->pre_pix[c->pre_next];
        c->pre_next++;
    }
    else
    {
        r = pbd_row(b, c, m_val);
        pix = rocke_b_global_load_i32(b, c->ws_ptr, rocke_b_add(b, c->rows_word, r), 4);
    }
    *out_valid
        = (c->spec->msplit > 1 || c->spec->wave_wg) ? rocke_b_cmp_lt(b, r, c->m_valid) : NULL;
    return rocke_b_add(b, rocke_b_mul(b, pix, c->c_C), n_val);
}

} // namespace

rocke_kernel_def_t* rocke_build_dgrad_box_det_reduce(rocke_ir_builder_t* b,
                                                     const rocke_dgrad_box_det_reduce_spec_t* spec,
                                                     const char* arch)
{
    (void)arch;
    const int BS = rocke_dgrad_box_det_reduce_block_size(spec);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = 16;
    wo.align_set = true;
    rocke_value_t* ws_ptr = rocke_b_param(b, "ws", rocke_ptr_type(b, rocke_i32(), "global"), &ro);
    rocke_value_t* dx_ptr = rocke_b_param(
        b, "dx", rocke_ptr_type(b, rocke_b_io_ir_type(b, spec->dtype_d), "global"), &wo);
    rocke_value_t* tab_word = rocke_b_param(b, "tab_word", rocke_i32(), NULL);
    rocke_value_t* ws_word = rocke_b_param(b, "ws_word", rocke_i32(), NULL);
    rocke_value_t* c_C = rocke_b_param(b, "C", rocke_i32(), NULL);
    rocke_value_t* dx_bytes = rocke_b_param(b, "dx_bytes", rocke_i32(), NULL);

    const rocke_mfma_atom_t* atom = rocke_mfma_atom("bf16", 16, 16, 16);
    if(spec->tile_m % (spec->warp_m * 16) || spec->tile_n % (spec->warp_n * 16))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "box det reduce: tile not divisible by the warp grid");
        return NULL;
    }
    const int mfm = spec->tile_m / (spec->warp_m * 16);
    const int mfn = spec->tile_n / (spec->warp_n * 16);
    if(mfm % spec->msplit)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "msplit %d does not divide %d row blocks", spec->msplit, mfm);
        return NULL;
    }
    const int g = mfm / spec->msplit;
    rocke_warp_grid_t grid = spec->wave_wg
                                 ? pbd_bind_grid(b, g * 16, mfn * 16, 1, 1, spec->wave_size)
                                 : pbd_bind_grid(b,
                                                 spec->tile_m / spec->msplit,
                                                 spec->tile_n,
                                                 spec->warp_m,
                                                 spec->warp_n,
                                                 spec->wave_size);
    rocke_value_t* tid = grid.tid;
    rocke_value_t* lane = grid.lane;
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    (void)rocke_b_const_i32(b, 1);
    rocke_value_t* q = spec->msplit > 1 ? rocke_b_block_id_y(b) : c0;

    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* e
        = rocke_b_add(b, tab_word, rocke_b_mul(b, bx, rocke_b_const_i32(b, ROCKE_PBD_ENTRY_WORDS)));
    rocke_value_t* ent[6];
    for(int i = 0; i < 6; ++i)
    {
        rocke_value_t* ix = rocke_b_add(b, e, rocke_b_const_i32(b, i));
        ent[i] = rocke_b_readfirstlane(b, rocke_b_global_load_i32(b, ws_ptr, ix, 4));
    }
    rocke_value_t *slot0 = ent[0], *S = ent[1], *step = ent[2], *rows_word = ent[3],
                  *n_off = ent[4], *m_valid = ent[5];

    const int n_w = spec->warp_m * spec->warp_n;
    const int tile_elems = spec->tile_m * spec->tile_n;
    rocke_value_t* ws_nb = rocke_b_const_i32(b, 0x7FFFFFF0);
    rocke_value_t* ws_rsrc = rocke_b_buffer_rsrc(b, ws_ptr, ws_nb);
    (void)rocke_b_const_i32(b, 0); /* make_buffer_resource's soffset */
    rocke_value_t* wave;
    if(spec->wave_wg)
        wave = rocke_b_block_id_z(b);
    else
        wave = rocke_b_readfirstlane(b, rocke_b_div(b, tid, rocke_b_const_i32(b, spec->wave_size)));
    const int eb = spec->slot_bytes;
    rocke_value_t* w64 = rocke_b_mul(b, wave, rocke_b_const_i32(b, 64));
    rocke_value_t* wl = rocke_b_add(b, w64, lane);
    rocke_value_t* lane_off = rocke_b_mul(b, wl, rocke_b_const_i32(b, 4 * eb));
    const int acc_stride = n_w * 64 * 4 * eb;
    rocke_value_t* wsw4 = rocke_b_mul(b, ws_word, rocke_b_const_i32(b, 4));
    rocke_value_t* qs = rocke_b_mul(b, q, rocke_b_const_i32(b, g * mfn * acc_stride));
    rocke_value_t* ws_bytes0 = rocke_b_add(b, wsw4, qs);

    pbd_ctx* c = new pbd_ctx();
    memset(c, 0, sizeof(*c));
    c->spec = spec;
    c->ws_ptr = ws_ptr;
    c->rows_word = rows_word;
    c->m_valid = m_valid;
    c->c_C = c_C;
    c->mfm = mfm;

    rocke_value_t* pre_rows[256];
    int n_pre = 0;
    if(spec->pix_pre && spec->m_chunks == 1)
    {
        rocke_cshuffle_epilogue_t e0
            = rocke_cshuffle_epilogue_from_grid(atom, &grid, spec->store_vec);
        const int sv = e0.store_vec;
        const int gbs = rocke_warp_grid_block_size(&grid);
        const int nvec = grid.tile_m * grid.tile_n / sv / gbs;
        for(int e_ = 0; e_ < nvec && n_pre < 256; ++e_)
        {
            rocke_value_t* a = rocke_b_add(b, rocke_b_const_i32(b, e_ * gbs), tid);
            pre_rows[n_pre++] = rocke_b_div(b, a, rocke_b_const_i32(b, grid.tile_n / sv));
        }
    }
    if(spec->wave_wg)
    {
        rocke_value_t* wm = rocke_b_div(b, wave, rocke_b_const_i32(b, spec->warp_n));
        rocke_value_t* wmn = rocke_b_mul(b, wm, rocke_b_const_i32(b, spec->warp_n));
        rocke_value_t* wn = rocke_b_sub(b, wave, wmn);
        grid.block_m_off = c0;
        rocke_value_t* wo_ = rocke_b_mul(b, wn, rocke_b_const_i32(b, mfn * 16));
        grid.block_n_off = rocke_b_add(b, n_off, wo_);
        rocke_value_t* r1 = rocke_b_mul(b, wm, rocke_b_const_i32(b, mfm * 16));
        rocke_value_t* r2 = rocke_b_mul(b, q, rocke_b_const_i32(b, g * 16));
        c->row0 = rocke_b_add(b, r1, r2);
    }
    else
    {
        grid.block_m_off = c0;
        grid.block_n_off = n_off;
    }
    rocke_value_t* dx_rsrc = rocke_b_buffer_rsrc(b, dx_ptr, dx_bytes);
    (void)rocke_b_const_i32(b, 0); /* make_buffer_resource's soffset */
    c->c_g16 = rocke_b_const_i32(b, g * 16);
    c->q_off = rocke_b_mul(b, q, c->c_g16);

    if(n_pre)
    {
        for(int i = 0; i < n_pre; ++i)
        {
            rocke_value_t* r = pbd_row(b, c, pre_rows[i]);
            rocke_value_t* pix
                = rocke_b_global_load_i32(b, ws_ptr, rocke_b_add(b, rows_word, r), 4);
            c->pre_r[i] = r;
            c->pre_pix[i] = pix;
        }
        /* Python reverses and pops from the end: first in, first used. */
        c->n_pre = n_pre;
        c->pre_next = 0;
    }

    const int n_acc = g * mfn;
    if(n_acc > 256)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "box det reduce: too many accumulators");
        delete c;
        return NULL;
    }
    rocke_iter_arg_t it[256];
    char nb[32];
    for(int j = 0; j < n_acc; ++j)
    {
        snprintf(nb, sizeof(nb), "pbd_a%d", j);
        it[j].name = rocke_arena_strdup(&b->arena, nb);
        it[j].init = rocke_b_zero_vec_f32(b, 4);
    }
    const int U = spec->unroll;
    rocke_value_t* c_oob = rocke_b_const_i32(b, 0x7FFFFFF0);
    rocke_for_t red
        = rocke_b_scf_for_iter(b, c0, S, rocke_b_const_i32(b, U), it, n_acc, "s", false, true);
    rocke_b_region_enter(b, red.body);
    {
        rocke_value_t* s = red.iv;
        rocke_value_t* parts[8][256];
        for(int u = 0; u < U && u < 8; ++u)
        {
            rocke_value_t* su = u ? rocke_b_add(b, s, rocke_b_const_i32(b, u)) : s;
            rocke_value_t* slot = rocke_b_add(b, slot0, rocke_b_mul(b, su, step));
            rocke_value_t* sb = rocke_b_mul(b, slot, rocke_b_const_i32(b, tile_elems * eb));
            rocke_value_t* base = rocke_b_add(b, rocke_b_add(b, ws_bytes0, sb), lane_off);
            if(u)
                base = rocke_b_select(b, rocke_b_cmp_lt(b, su, S), base, c_oob);
            if(eb == 2)
            {
                rocke_value_t* raw[256];
                for(int j = 0; j < n_acc; ++j)
                {
                    rocke_value_t* off = rocke_b_add(b, base, rocke_b_const_i32(b, j * acc_stride));
                    raw[j] = rocke_b_buffer_load_vN_aux(
                        b, ws_rsrc, off, c0, rocke_bf16(), 4, spec->load_aux);
                }
                for(int j = 0; j < n_acc; ++j)
                {
                    rocke_value_t* el[4];
                    for(int i = 0; i < 4; ++i)
                        el[i] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, raw[j], i));
                    parts[u][j] = rocke_b_vec_pack(b, el, 4, rocke_f32());
                }
            }
            else
            {
                for(int j = 0; j < n_acc; ++j)
                {
                    rocke_value_t* off = rocke_b_add(b, base, rocke_b_const_i32(b, j * acc_stride));
                    parts[u][j] = rocke_b_buffer_load_vN_aux(
                        b, ws_rsrc, off, c0, rocke_f32(), 4, spec->load_aux);
                }
            }
        }
        rocke_value_t* nw[256];
        for(int j = 0; j < n_acc; ++j)
            nw[j] = red.iter_vars[j];
        for(int u = 0; u < U && u < 8; ++u)
            for(int j = 0; j < n_acc; ++j)
                nw[j] = rocke_b_fadd(b, nw[j], parts[u][j]);
        rocke_b_scf_yield(b, nw, n_acc);
    }
    rocke_b_region_leave(b);
    rocke_value_t* accs[256];
    for(int j = 0; j < n_acc; ++j)
        accs[j] = red.op->results[j];

    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, &grid, spec->store_vec);
    epi.out_dtype = spec->dtype_d;
    epi.war_barriers = 1;
    epi.m_chunks = spec->m_chunks;
    rocke_value_t* bound_m
        = (spec->msplit == 1 && !spec->wave_wg) ? m_valid : rocke_b_const_i32(b, spec->tile_m);
    rocke_cshuffle_epilogue_store(b, &epi, accs, n_acc, pbd_dx_addr, c, dx_rsrc, bound_m, c_C);
    delete c;
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_dgrad_box_det_reduce_new(
    rocke_ir_builder_t* b, const rocke_dgrad_box_det_reduce_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(b == NULL || spec == NULL)
            return NULL;
        char name[256];
        rocke_dgrad_box_det_reduce_kernel_name(spec, name, sizeof(name));
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_dgrad_box_det_reduce(b, spec, arch);
    });
}
