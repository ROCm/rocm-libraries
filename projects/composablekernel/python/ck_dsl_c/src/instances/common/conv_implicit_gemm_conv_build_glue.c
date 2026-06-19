/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_conv_implicit_gemm_conv_build_glue.c -- PUBLIC build entry + the
 * build prologue glue for the C99 chunked port of build_implicit_gemm_conv
 * (ck_dsl/instances/common/conv_implicit_gemm.py).
 *
 * SCOPE (this TU only):
 *   - ckc_conv_build_ctx_init               (Python prologue, lines 787-1032)
 *   - ckc_build_implicit_gemm_conv          (driver: ctx_init -> selected
 *                                            K-loop driver -> epilogue, return
 *                                            b->kernel)  (lines 730-1378)
 *   - ckc_build_implicit_gemm_conv_new      (init b from spec.kernel_name then
 *                                            build)
 *   - ckc_conv_implicit_gemm_lower_to_llvm  (build stock body -> lower .ll)
 *
 * The phase functions (a_descriptor / b_descriptor / emit_load_phase /
 * emit_mfma_phase / emit_wmma_phase, the three K-loop drivers, the epilogue, and
 * the module-level pure helpers) are peers implemented in sibling TUs and
 * declared in ckc/instance_conv_implicit_gemm_internal.h. This TU calls them but
 * does not implement them.
 *
 * Byte-identical builder-call sequence: ckc_conv_build_ctx_init walks the Python
 * prologue (787-1032) top-to-bottom, computing each local in source order and
 * stashing it into the shared ckc_conv_build_ctx_t. The driver then selects the
 * one K-loop driver Python would (unroll_k / sync / async) and runs the epilogue
 * phase, returning b->kernel.
 *
 * STUB NOTES (peers not yet landed as headers):
 *   - WarpGrid.from_atom(...).bind(...) has no C helper yet; the compile-time
 *     grid geometry is filled here and the bound SSA values are left for the
 *     grid peer port (TODO(port)). The descriptor/epilogue phases read
 *     ctx->grid / ctx->tid / ... so they pick up the bound values once the peer
 *     populates them.
 *   - spec.atom (the legacy MfmaAtom) has no public accessor yet; ctx->atom is
 *     left NULL pending that accessor port (TODO(port)). It is NULL on the WMMA
 *     path regardless.
 *   - spec.effective_lds_layout() is ported (LdsLayout peer in the
 *     spec_descriptors TU): ckc_implicit_gemm_conv_spec_effective_lds_layout
 *     fills ctx->lds_layout and ckc_conv_lds_layout_validate_for_async runs the
 *     async-only assert. The lds_layout override branch is still a peer port
 *     (the spec field is an opaque void* with no C constructor).
 *   - make_buffer_resource() has no C helper yet; the buffer-resource slice is
 *     filled directly via ckc_b_buffer_rsrc + a pre-bound const_i32(0) soffset,
 *     exactly as make_buffer_resource does.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"                       /* ckc_arena_strdup */
#include "ckc/helper_ck_dsl.helpers.grid.h"  /* chiplet_aware_super_tile_dynamic */
#include "ckc/instance_conv_implicit_gemm.h"
#include "ckc/instance_conv_implicit_gemm_internal.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/lower_llvm.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  make_buffer_resource(b, ptr, num_bytes) (helpers/tensor_view.py)
 *
 *  Wraps b.buffer_rsrc(ptr, num_bytes) + a pre-bound zero soffset. Fills the
 *  value-type slice the conv body reads. Mirrors the Python constructor exactly
 *  (the BufferResource peer carries the full type; the conv body only reads
 *  .rsrc / .soffset). No-op-safe: if b is in error the rsrc come back NULL.
 * ===================================================================== */
static void ckc_conv_make_buffer_resource(ckc_ir_builder_t* b,
                                          ckc_value_t* ptr,
                                          ckc_value_t* num_bytes,
                                          ckc_conv_buffer_resource_t* out)
{
    out->ptr       = ptr;
    out->num_bytes = num_bytes;
    out->rsrc      = ckc_b_buffer_rsrc(b, ptr, num_bytes);
    out->soffset   = ckc_b_const_i32(b, 0);
}

/* ===================================================================== *
 *  ckc_conv_build_ctx_init -- the build prologue (Python lines 787-1032).
 *
 *  Walks the Python prologue top-to-bottom, computing each local in source
 *  order and stashing it into ctx. Returns false (builder error set) on the
 *  spec.validate() / is_valid_spec reject paths or any builder ValueError.
 * ===================================================================== */
bool ckc_conv_build_ctx_init(ckc_conv_build_ctx_t* ctx,
                             ckc_ir_builder_t* b,
                             const ckc_implicit_gemm_conv_spec_t* spec,
                             const char* arch,
                             const ckc_conv_build_overrides_t* overrides)
{
    char reason[CKC_ERR_MSG_CAP];
    int mi, ni;
    int i;

    if (ctx == NULL || b == NULL || spec == NULL)
    {
        if (b != NULL)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "conv_build_ctx_init: null ctx/spec");
        }
        return false;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->b    = b;
    ctx->spec = spec;
    ctx->arch = (arch != NULL) ? arch : "gfx950";
    ctx->ov   = overrides;
    ctx->p    = &spec->problem; /* p = spec.problem */

    /* ---- spec.validate() ---- (line 787) */
    if (!ckc_implicit_gemm_conv_spec_validate(spec, reason, sizeof(reason)))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "%s", reason);
        return false;
    }

    /* ---- ok, why = is_valid_spec(spec, arch=arch); raise on reject ----
     * (lines 788-790) */
    if (!ckc_implicit_gemm_conv_is_valid_spec(spec, ctx->arch, reason, sizeof(reason)))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "invalid conv_igemm spec for %s: %s", ctx->arch, reason);
        return false;
    }

    /* ---- b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu ---- (792-795)
     * The builder `b` is already constructed with spec.kernel_name() by the
     * caller (the Python `b = IRBuilder(spec.kernel_name())`). */
    if (spec->has_waves_per_eu && b->kernel != NULL)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    /* ---- params (A/B/D + extra_params + *_bytes) ---- (797-803) */
    {
        ckc_param_opts_t a_opts;
        ckc_param_opts_t bp_opts;
        ckc_param_opts_t d_opts;
        const ckc_type_t* f16_global = ckc_ptr_type(b, ckc_f16(), "global");

        memset(&a_opts, 0, sizeof(a_opts));
        a_opts.noalias = true;  a_opts.noalias_set = true;
        a_opts.readonly = true; a_opts.readonly_set = true;
        a_opts.align = 16;      a_opts.align_set = true;
        ctx->A = ckc_b_param(b, "A", f16_global, &a_opts);

        memset(&bp_opts, 0, sizeof(bp_opts));
        bp_opts.noalias = true;  bp_opts.noalias_set = true;
        bp_opts.readonly = true; bp_opts.readonly_set = true;
        bp_opts.align = 16;      bp_opts.align_set = true;
        ctx->Bp = ckc_b_param(b, "B", f16_global, &bp_opts);

        memset(&d_opts, 0, sizeof(d_opts));
        d_opts.noalias = true;   d_opts.noalias_set = true;
        d_opts.writeonly = true; d_opts.writeonly_set = true;
        d_opts.align = 16;       d_opts.align_set = true;
        ctx->D = ckc_b_param(b, "D", f16_global, &d_opts);

        /* extra_context = extra_params(b) if extra_params is not None else None */
        if (overrides != NULL && overrides->extra_params != NULL)
        {
            ctx->extra_context = overrides->extra_params(b, overrides->user);
        }
        else
        {
            ctx->extra_context = NULL;
        }

        ctx->A_bytes = ckc_b_param(b, "A_bytes", ckc_i32(), NULL);
        ctx->B_bytes = ckc_b_param(b, "B_bytes", ckc_i32(), NULL);
        ctx->D_bytes = ckc_b_param(b, "D_bytes", ckc_i32(), NULL);
    }

    /* ---- resolve op + atom + frag widths ---- (811-815) */
    ctx->op = ckc_conv_resolve_op(b, spec, ctx->arch);
    if (ctx->op == NULL)
    {
        return false; /* ckc_conv_resolve_op set the builder error */
    }
    ctx->is_wmma = (ctx->op->family != NULL && strcmp(ctx->op->family, "wmma") == 0);
    /* atom = spec.atom if op.family == "mma" else None.
     *
     * Python: ``ImplicitGemmConvSpec.atom`` is
     *     mfma_atom("f16", warp_tile_m, warp_tile_n, warp_tile_k)
     * and ``build_implicit_gemm_conv`` keeps it only on the MFMA path
     * (``atom = spec.atom if op.family == "mma" else None``). ``ckc_mfma_atom``
     * returns a pointer into the same immutable static catalog the epilogues
     * use, so the lookup is byte-identical to ``spec.atom``. NULL stays correct
     * on the WMMA family. */
    ctx->atom = ctx->is_wmma
                    ? NULL
                    : ckc_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n,
                                    spec->warp_tile_k);

    ctx->a_per_lane = ctx->op->a_frag_len;
    ctx->b_per_lane = ctx->op->b_frag_len;
    ctx->c_per_lane = ctx->op->c_frag_len;

    /* ---- block tile dims ---- (817) */
    ctx->block_m = spec->tile_m;
    ctx->block_n = spec->tile_n;
    ctx->block_k = spec->tile_k;

    /* ---- block/warp/lane decomposition (WarpGrid.from_atom().bind()) ----
     * (826-841). Fill the compile-time geometry the epilogues read; the bound
     * SSA (tid/lane/warp_*_idx/block_*_off) is produced by WarpGrid.bind.
     * TODO(port): WarpGrid.from_atom/bind has no C helper yet -- the bound SSA
     * values are left NULL for the grid peer to populate, including the
     * max_workgroup_size kernel attr the bind emits. */
    ctx->grid.tile_m      = ctx->block_m;
    ctx->grid.tile_n      = ctx->block_n;
    ctx->grid.tile_k      = ctx->block_k;
    ctx->grid.warp_m      = spec->warp_m;
    ctx->grid.warp_n      = spec->warp_n;
    ctx->grid.warp_k      = 1;
    ctx->grid.warp_tile_m = spec->warp_tile_m;
    ctx->grid.warp_tile_n = spec->warp_tile_n;
    ctx->grid.warp_tile_k = spec->warp_tile_k;
    ctx->grid.wave_size   = spec->wave_size;
    /* WarpGrid.from_atom(op, tile_*, warp_*, wave_size).bind(b,
     *     block_m_axis="y", block_n_axis="x")  (geometry.py WarpGrid.bind).
     * warp_k == 1 and block_k_axis is None for the conv body, so this is the
     * simple (no split-K) lane decomposition. Emitted in the byte-identical
     * builder-call order of WarpGrid.bind (geometry.py lines 222-278):
     *
     *   b.kernel.attrs["max_workgroup_size"] = block_size
     *   wave = const(wave_size); c_warps_n = const(warp_n)
     *   c_warps_n_warp_m = const(warp_n*warp_m)
     *   c_tile_m/n/k = const(tile_m/n/k)
     *   tid = thread_id_x(); lane = mod(tid, wave); warp_id = div(tid, wave)
     *   warp_m_idx = div(warp_id, c_warps_n); warp_n_idx = mod(...)
     *   warp_k_idx = const(0)
     *   block_m_off = mul(block_id_y(), c_tile_m)
     *   block_n_off = mul(block_id_x(), c_tile_n); block_k_off = const(0)
     */
    if (b->kernel != NULL)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size",
                         ckc_warp_grid_block_size(&ctx->grid));
    }

    ckc_value_t* wave             = ckc_b_const_i32(b, spec->wave_size);
    ckc_value_t* c_warps_n        = ckc_b_const_i32(b, spec->warp_n);
    /* c_warps_n_warp_m is emitted by bind even though warp_k == 1 leaves it
     * unused on the no-split-K path; keep the const for byte-identity. */
    ckc_value_t* c_warps_n_warp_m = ckc_b_const_i32(b, spec->warp_n * spec->warp_m);
    ckc_value_t* c_tile_m         = ckc_b_const_i32(b, ctx->grid.tile_m);
    ckc_value_t* c_tile_n         = ckc_b_const_i32(b, ctx->grid.tile_n);
    /* c_tile_k is emitted by bind; block_k_axis is None so it is unused. */
    ckc_value_t* c_tile_k         = ckc_b_const_i32(b, ctx->grid.tile_k);
    (void) c_warps_n_warp_m;
    (void) c_tile_k;

    ckc_value_t* tid_v     = ckc_b_thread_id_x(b);
    ckc_value_t* lane_v    = ckc_b_mod(b, tid_v, wave);
    ckc_value_t* warp_id_v = ckc_b_div(b, tid_v, wave);

    /* warp_k == 1 path */
    ckc_value_t* warp_m_idx_v = ckc_b_div(b, warp_id_v, c_warps_n);
    ckc_value_t* warp_n_idx_v = ckc_b_mod(b, warp_id_v, c_warps_n);
    ckc_value_t* warp_k_idx_v = ckc_b_const_i32(b, 0);

    ckc_value_t* block_m_off_b = ckc_b_mul(b, ckc_b_block_id_y(b), c_tile_m);
    ckc_value_t* block_n_off_b = ckc_b_mul(b, ckc_b_block_id_x(b), c_tile_n);
    ckc_value_t* block_k_off_b = ckc_b_const_i32(b, 0);

    ctx->grid.tid         = tid_v;
    ctx->grid.lane        = lane_v;
    ctx->grid.warp_id     = warp_id_v;
    ctx->grid.warp_m_idx  = warp_m_idx_v;
    ctx->grid.warp_n_idx  = warp_n_idx_v;
    ctx->grid.warp_k_idx  = warp_k_idx_v;
    ctx->grid.block_m_off = block_m_off_b;
    ctx->grid.block_n_off = block_n_off_b;
    ctx->grid.block_k_off = block_k_off_b;

    ctx->tid        = ctx->grid.tid;        /* grid.tid        */
    ctx->lane       = ctx->grid.lane;       /* grid.lane       */
    ctx->warp_id    = ctx->grid.warp_id;    /* grid.warp_id    */
    ctx->warp_m_idx = ctx->grid.warp_m_idx; /* grid.warp_m_idx */
    ctx->warp_n_idx = ctx->grid.warp_n_idx; /* grid.warp_n_idx */

    /* ---- common geometry constants ---- (843-845) */
    ctx->c0       = ckc_b_const_i32(b, 0);
    ctx->c_block_k = ckc_b_const_i32(b, ctx->block_k);
    ctx->c_K_gemm = ckc_b_const_i32(b, ckc_conv_problem_k_gemm(ctx->p));

    /* ---- per-CTA tile origins (chiplet-swizzle aware) ---- (858-879) */
    if (spec->chiplet_swizzle)
    {
        int num_pid_m = (ckc_conv_problem_m(ctx->p) + ctx->block_m - 1) / ctx->block_m;
        int num_pid_n = (ckc_conv_problem_n_gemm(ctx->p) + ctx->block_n - 1) / ctx->block_n;
        ckc_value_t* c_num_pid_n = ckc_b_const_i32(b, num_pid_n);
        /* wgid_flat = b.add(b.mul(b.block_id_y(), c_num_pid_n), b.block_id_x()).
         * Python evaluates the add's first arg (b.mul(b.block_id_y(), ...)) fully
         * before the second (b.block_id_x()): block_id_y -> mul -> block_id_x ->
         * add. C arg eval order is unspecified, so pin it with temporaries. */
        ckc_value_t* bid_y = ckc_b_block_id_y(b);
        ckc_value_t* mul_y = ckc_b_mul(b, bid_y, c_num_pid_n);
        ckc_value_t* bid_x = ckc_b_block_id_x(b);
        ckc_value_t* wgid_flat = ckc_b_add(b, mul_y, bid_x);
        /* Python calls the COMPILE-TIME chiplet_aware_super_tile (conv tile
         * counts are derived from the static problem shape): limit and
         * num_wgid_in_group are folded consts, not div/mul IR. */
        ckc_super_tile_swizzle_result_t swz =
            ckc_chiplet_aware_super_tile(b,
                                         wgid_flat,
                                         num_pid_m,
                                         num_pid_n,
                                         spec->chiplet_wgm,
                                         spec->chiplet_num_xcds,
                                         spec->chiplet_chunk_size);
        ctx->block_m_off_v = ckc_b_mul(b, swz.row, ckc_b_const_i32(b, ctx->block_m));
        ctx->block_n_off_v = ckc_b_mul(b, swz.col, ckc_b_const_i32(b, ctx->block_n));
        /* grid = dc_replace(grid, block_m_off=..., block_n_off=...) so the
         * downstream loaders / epilogues pick up the remapped origins. */
        ctx->grid.block_m_off = ctx->block_m_off_v;
        ctx->grid.block_n_off = ctx->block_n_off_v;
    }
    else
    {
        ctx->block_m_off_v = ctx->grid.block_m_off;
        ctx->block_n_off_v = ctx->grid.block_n_off;
    }

    /* ---- LDS plan ---- (894-896). lds_layout = spec.effective_lds_layout():
     * sync path pads each K-row by +8 halves (when tile_k >= 16) to dodge LDS
     * bank conflicts; the async / packed path uses +0 (lane-contiguous LDS).
     * Ported via the LdsLayout accessor (spec_descriptors TU) so the derivation
     * + layout.validate() + (on the async path) validate_for_async() match
     * Python. On rejection set the builder's sticky error and bail. */
    {
        char lds_reason[CKC_ERR_MSG_CAP];
        /* spec.validate() (above) already ran effective_lds_layout +
         * validate_for_async, so these cannot fail here for a validated spec;
         * route the population through the accessor anyway (single source of
         * truth) and surface any error defensively. */
        if (!ckc_implicit_gemm_conv_spec_effective_lds_layout(
                spec, &ctx->lds_layout, lds_reason, sizeof(lds_reason)))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", lds_reason);
            return false;
        }
        if (spec->async_dma &&
            !ckc_conv_lds_layout_validate_for_async(&ctx->lds_layout, lds_reason,
                                                    sizeof(lds_reason)))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", lds_reason);
            return false;
        }
    }

    /* smem_alloc A_smem / B_smem with storage_shape(rows) = (rows, row_stride) */
    {
        int a_shape[2];
        int b_shape[2];
        a_shape[0] = ctx->block_m;
        a_shape[1] = ctx->lds_layout.row_stride;
        b_shape[0] = ctx->block_n;
        b_shape[1] = ctx->lds_layout.row_stride;
        ctx->A_smem = ckc_b_smem_alloc(b, ckc_f16(), a_shape, 2, "A_smem");
        ctx->B_smem = ckc_b_smem_alloc(b, ckc_f16(), b_shape, 2, "B_smem");

        /* double_buffer = compv4 || async_dma || unroll_k */
        ctx->double_buffer =
            (spec->pipeline != NULL && strcmp(spec->pipeline, "compv4") == 0) ||
            spec->async_dma || spec->unroll_k;
        if (ctx->double_buffer)
        {
            ctx->A_smem2 = ckc_b_smem_alloc(b, ckc_f16(), a_shape, 2, "A_smem2");
            ctx->B_smem2 = ckc_b_smem_alloc(b, ckc_f16(), b_shape, 2, "B_smem2");
        }
        else
        {
            ctx->A_smem2 = ctx->A_smem;
            ctx->B_smem2 = ctx->B_smem;
        }
    }

    /* ---- per-warp MFMA tile counts ---- (915-917) */
    ctx->mfmas_m = ckc_implicit_gemm_conv_spec_mfmas_per_warp_m(spec);
    ctx->mfmas_n = ckc_implicit_gemm_conv_spec_mfmas_per_warp_n(spec);
    ctx->k_atoms = ckc_implicit_gemm_conv_spec_k_atoms_per_tile_k(spec);

    /* ---- accumulators ---- (919-922).
     * acc_init = zero_vec_f32(c_per_lane); accs = [(acc_m{mi}_n{ni}, acc_init)
     * for mi in range(mfmas_m) for ni in range(mfmas_n)]. */
    ctx->acc_init = ckc_b_zero_vec_f32(b, ctx->c_per_lane);
    ctx->num_accs = ctx->mfmas_m * ctx->mfmas_n;
    if (ctx->num_accs < 0 || ctx->num_accs > CKC_CONV_MAX_ACCS)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "conv: too many accumulators (%d)", ctx->num_accs);
        return false;
    }
    i = 0;
    for (mi = 0; mi < ctx->mfmas_m; ++mi)
    {
        for (ni = 0; ni < ctx->mfmas_n; ++ni)
        {
            /* "acc_m{mi}_n{ni}" -- arena-stable name matching the Python
             * f-string exactly (no fresh-counter suffix). */
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "acc_m%d_n%d", mi, ni);
            ctx->acc_names[i] = ckc_arena_strdup(&b->arena, tmp);
            ctx->acc_inits[i] = ctx->acc_init;
            ++i;
        }
    }

    /* ---- global -> LDS coalesced copy plan ---- (924-925) */
    ctx->threads  = ckc_implicit_gemm_conv_spec_block_size(spec);
    ctx->load_vec = ckc_conv_choose_load_vec(spec);

    /* ---- coordinate-transform descriptors ---- (935-936).
     * A_desc decompose_m = (a_mhw_index_fn is None). */
    {
        bool decompose_m = !(overrides != NULL && overrides->a_mhw_index_fn != NULL);
        ctx->A_desc = ckc_conv_make_a_descriptor(b, ctx->p, decompose_m);
        ctx->B_desc = ckc_conv_make_b_descriptor(b, ctx->p);
        ctx->D_desc = NULL; /* built lazily in the epilogue phase */
    }

    /* ---- buffer resources (CK-Tile views over A/B/D) ---- (946-951) */
    ckc_conv_make_buffer_resource(b, ctx->A, ctx->A_bytes, &ctx->a_buf_rsrc);
    ckc_conv_make_buffer_resource(b, ctx->Bp, ctx->B_bytes, &ctx->b_buf_rsrc);
    ckc_conv_make_buffer_resource(b, ctx->D, ctx->D_bytes, &ctx->d_buf_rsrc);
    ctx->a_rsrc = ctx->a_buf_rsrc.rsrc;
    ctx->b_rsrc = ctx->b_buf_rsrc.rsrc;
    ctx->d_rsrc = ctx->d_buf_rsrc.rsrc;

    /* ---- input_cache_setup(b, spec, grid, a_rsrc) ---- (952-956) */
    if (overrides != NULL && overrides->input_cache_setup != NULL)
    {
        ctx->input_cache_context =
            overrides->input_cache_setup(b, spec, &ctx->grid, ctx->a_rsrc, overrides->user);
    }
    else
    {
        ctx->input_cache_context = NULL;
    }

    /* ---- mutable cell read by a_descriptor / b_descriptor ---- (984) */
    ctx->k_off_capture = NULL; /* Python: k_off_capture = [None] */

    /* ---- loaders (exactly one family populated) ---- (986-1027) */
    ctx->async_dma = spec->async_dma;
    if (ctx->async_dma)
    {
        ckc_status_t sa = ckc_async_tile_loader_from_tile(
            ctx->block_m, ctx->block_k, ctx->threads, spec->wave_size, 4, &ctx->a_loader);
        ckc_status_t sb = ckc_async_tile_loader_from_tile(
            ctx->block_n, ctx->block_k, ctx->threads, spec->wave_size, 4, &ctx->b_loader);
        if (sa != CKC_OK || sb != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "conv: async tile loader from_tile failed");
            return false;
        }
        ctx->have_async_loaders = true;
        ctx->have_sync_loaders  = false;
    }
    else
    {
        /* #8624 vector-sizes-as-args: per-operand load width override. Python:
         *   load_vec_a = spec.vector_size_a if not None else _auto_load_vec
         *   load_vec_b = spec.vector_size_b if not None else _auto_load_vec */
        int load_vec_a = spec->has_vector_size_a ? spec->vector_size_a : ctx->load_vec;
        int load_vec_b = spec->has_vector_size_b ? spec->vector_size_b : ctx->load_vec;
        ckc_status_t sa = ckc_coalesced_tile_loader_from_tile(
            ctx->block_m, ctx->block_k, ctx->threads, load_vec_a, true, &ctx->a_sync_loader);
        ckc_status_t sb = ckc_coalesced_tile_loader_from_tile(
            ctx->block_n, ctx->block_k, ctx->threads, load_vec_b, true, &ctx->b_sync_loader);
        if (sa != CKC_OK || sb != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "conv: coalesced tile loader from_tile failed");
            return false;
        }
        ctx->have_sync_loaders  = true;
        ctx->have_async_loaders = false;
    }

    /* ---- schedule policy + prologue ---- (1029-1032) */
    ctx->schedule = ckc_schedule_policy_for_pipeline(
        b, ctx->async_dma ? "async_dma" : spec->pipeline);
    ckc_schedule_policy_emit_prologue(&ctx->schedule, b);

    if (!ckc_ir_builder_ok(b))
    {
        return false;
    }
    return true;
}

/* ===================================================================== *
 *  ckc_build_implicit_gemm_conv -- the public build driver (730-1378).
 *
 *  Populates the shared ctx (the prologue), selects the one K-loop driver
 *  Python would, runs the epilogue phase, and returns b->kernel.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_implicit_gemm_conv(ckc_ir_builder_t* b,
                                               const ckc_implicit_gemm_conv_spec_t* spec,
                                               const char* arch,
                                               const ckc_conv_build_overrides_t* overrides)
{
    ckc_conv_build_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* The Python build_implicit_gemm_conv ALWAYS constructs its own
     * IRBuilder(spec.kernel_name()) (line 793), so the emitted kernel symbol is
     * the CONV spec's kernel_name regardless of any name the caller gave the
     * builder. The C entry reuses a caller-supplied builder, so name its kernel
     * here to stay byte-faithful: this is the lever that fixes wrapped callers
     * (e.g. deep_fused_conv_pool, which inits the builder with the deep-fusion
     * kernel_name() but must emit under the conv1 wrapped name). */
    if (b->kernel != NULL)
    {
        char name[256];
        if (ckc_implicit_gemm_conv_spec_kernel_name(spec, name, sizeof(name)) !=
            CKC_OK)
        {
            return NULL;
        }
        b->kernel->name = ckc_arena_strdup(&b->arena, name);
        if (b->kernel->name == NULL)
        {
            ckc_i_set_err(b, CKC_ERR_OOM, "implicit_gemm_conv: OOM kernel name");
            return NULL;
        }
    }

    /* ---- build prologue (787-1032) ---- */
    if (!ckc_conv_build_ctx_init(&ctx, b, spec, arch, overrides))
    {
        return NULL;
    }

    /* ---- K-loop driver selection (1276-1347) ----
     *   unroll_k            -> unroll
     *   else not async_dma  -> simple (scf.for_iter)
     *   else (async_dma)    -> async (SoftwarePipeline.run_ping_pong) */
    if (spec->unroll_k)
    {
        ckc_conv_emit_kloop_unroll(&ctx);
    }
    else if (!ctx.async_dma)
    {
        ckc_conv_emit_kloop_simple(&ctx);
    }
    else
    {
        ckc_conv_emit_kloop_async(&ctx);
    }

    /* ---- epilogue (1349-1377): apply acc epilogue + dispatch the override /
     * cshuffle / wmma-direct / mfma-direct chain. Reads ctx.final_accs. ---- */
    ckc_conv_emit_epilogue(&ctx);

    if (!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    /* return b.kernel (line 1378) */
    return b->kernel;
}

/* Convenience: init `b` with spec.kernel_name(), then build (Python builds its
 * own IRBuilder(spec.kernel_name()) inside; the C public entry takes an
 * already-init'd builder, so the _new wrapper performs that init). */
ckc_kernel_def_t* ckc_build_implicit_gemm_conv_new(ckc_ir_builder_t* b,
                                                   const ckc_implicit_gemm_conv_spec_t* spec,
                                                   const char* arch,
                                                   const ckc_conv_build_overrides_t* overrides)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if (b == NULL || spec == NULL)
        {
            return NULL;
        }
        if (ckc_implicit_gemm_conv_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if (ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_implicit_gemm_conv(b, spec, arch, overrides);

    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build (stock body) -> lower to LLVM .ll text. Owns and frees
 *  its own IRBuilder. On CKC_OK *out_ll receives a malloc'd NUL-terminated
 *  string the caller frees with free(); on failure it is left NULL and (if
 *  err != NULL, cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void ckc_conv_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if (err == NULL || err_cap == 0)
    {
        return;
    }
    if (msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if (n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

ckc_status_t ckc_conv_implicit_gemm_lower_to_llvm(const ckc_implicit_gemm_conv_spec_t* spec,
                                                  const char* arch,
                                                  ckc_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_conv_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* build -> the _new entry owns the builder init via spec.kernel_name().
     * Stock body: overrides == NULL (Python all-None default). */
    kernel = ckc_build_implicit_gemm_conv_new(&b, spec, arch, NULL);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_conv_set_err(err, err_cap,
                         (m != NULL && m[0] != '\0') ? m : "build_implicit_gemm_conv failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
