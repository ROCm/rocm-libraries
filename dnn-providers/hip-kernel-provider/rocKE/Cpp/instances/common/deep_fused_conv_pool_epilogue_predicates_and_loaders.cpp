// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_deep_fused_conv_pool_epilogue_predicates_and_loaders.c -- chunked
 * C99 port of the conv0-side emit helpers + deferred-epilogue scalar transform
 * from ck_dsl/instances/common/deep_fused_conv_pool.py.
 *
 * SCOPE (this part-file):
 *   _epilogue_is_pool_deferrable      (py 768-778) -> ckc_dfcp_epilogue_is_pool_deferrable
 *   _epilogue_is_relu_only            (py 781-790) -> ckc_dfcp_epilogue_is_relu_only
 *   _apply_epilogue_scalar            (py 793-813) -> ckc_dfcp_apply_epilogue_scalar
 *   _can_use_specialized_conv0_a_loader(py 501-519)-> ckc_dfcp_can_use_specialized_conv0_a_loader
 *   _load_conv0_a_tile_specialized    (py 522-591) -> ckc_dfcp_load_conv0_a_tile_specialized
 *   _setup_input_footprint_cache      (py 594-652) -> ckc_dfcp_setup_input_footprint_cache
 *   _load_conv0_a_tile_from_input_cache(py 655-707)-> ckc_dfcp_load_conv0_a_tile_from_input_cache
 *   _load_conv0_a_operand_from_input_cache(py 709-765)
 *                                                  ->
 * ckc_dfcp_load_conv0_a_operand_from_input_cache
 *
 * These are the conv0 A-load gate/loaders (CoalescedTileLoader + smem
 * load/store) plus the deferred-epilogue scalar transform reused by every
 * maxpool phase. Builder-call sequences are byte-identical to the Python.
 *
 * Python integer semantics: every // / % here operates on non-negative operands
 * (valid conv/pool shapes), so C truncating division matches Python floor
 * division exactly.
 */
#include "ckc/instance_deep_fused_conv_pool_internal.h"
/* pulls in instance_deep_fused_conv_pool.h + helper header + ir.h, and the
 * peer phase prototypes this file calls (none here -- all scope-local). */
#include "ckc/helper_ck_dsl.helpers.epilogues.h"
/* full ckc_warp_grid_t definition (grid->tid). The helper header only
 * forward-declares it as opaque; the loaders below read grid->tid. */
#include "ckc/helper_ck_dsl.helpers.loads.h"
/* CoalescedTileLoader value type + choose_vec/load. */

/* ------------------------------------------------------------------ *
 * Pure epilogue predicates / scalar apply
 * ------------------------------------------------------------------ */

/* _epilogue_is_pool_deferrable(epi):  return epi.scale >= 0.0 */
bool ckc_dfcp_epilogue_is_pool_deferrable(const ckc_conv_acc_epilogue_t* epi)
{
    return epi != NULL && epi->scale >= 0.0;
}

/* _epilogue_is_relu_only(epi):
 *   return (epi.bias == 0.0 and epi.scale == 1.0 and epi.relu
 *           and epi.clamp_min is None and epi.clamp_max is None) */
bool ckc_dfcp_epilogue_is_relu_only(const ckc_conv_acc_epilogue_t* epi)
{
    return epi != NULL && epi->bias == 0.0 && epi->scale == 1.0 && epi->relu
           && (!epi->has_clamp_min) && (!epi->has_clamp_max);
}

/* _apply_epilogue_scalar(b, epi, v): apply the static fp32 epilogue to one
 * scalar value, mirroring the per-lane transform in _apply_accumulator_epilogue
 * so the deferred-past-pool path is numerically identical to applying it on the
 * accs. */
ckc_value_t* ckc_dfcp_apply_epilogue_scalar(ckc_ir_builder_t* b,
                                            const ckc_conv_acc_epilogue_t* epi,
                                            ckc_value_t* v)
{
    if(b == NULL || epi == NULL || v == NULL)
    {
        return v;
    }
    if(ckc_conv_acc_epilogue_is_identity(epi))
    {
        return v;
    }
    if(epi->bias != 0.0)
    {
        v = ckc_b_fadd(b, v, ckc_b_const_f32(b, epi->bias));
    }
    if(epi->scale != 1.0)
    {
        v = ckc_b_fmul(b, v, ckc_b_const_f32(b, epi->scale));
    }
    if(epi->relu)
    {
        v = ckc_b_fmax(b, v, ckc_b_const_f32(b, 0.0));
    }
    if(epi->has_clamp_min)
    {
        v = ckc_b_fmax(b, v, ckc_b_const_f32(b, epi->clamp_min));
    }
    if(epi->has_clamp_max)
    {
        v = ckc_b_fmin(b, v, ckc_b_const_f32(b, epi->clamp_max));
    }
    return v;
}

/* ------------------------------------------------------------------ *
 * conv0 A-loader specialization gate (pure)
 * ------------------------------------------------------------------ */

/* _can_use_specialized_conv0_a_loader(spec): whether the target-shape conv0 A
 * load can bypass TensorDescriptor math. */
bool ckc_dfcp_can_use_specialized_conv0_a_loader(const ckc_deep_fused_conv_pool_spec_t* spec)
{
    const ckc_conv_problem_t* c;

    if(spec == NULL)
    {
        return false;
    }
    c = &spec->problem.conv;
    return spec->wave_size == 32 && !spec->cache_input_footprint
           && !spec->direct_conv0_from_input_cache && c->N == 1 && c->C == 8 && c->Y == 3
           && c->X == 3 && c->sH == 1 && c->sW == 1 && c->pH == 1 && c->pW == 1 && c->dH == 1
           && c->dW == 1;
}

/* Host-side (x).bit_length() for x >= 0, matching Python int.bit_length(). */
static int dfcp_bit_length(int x)
{
    int n = 0;
    while(x > 0)
    {
        n++;
        x >>= 1;
    }
    return n;
}

/* ------------------------------------------------------------------ *
 * _load_conv0_a_tile_specialized -- descriptor closure state
 * ------------------------------------------------------------------ *
 *
 * The Python `descriptor(b_, row, col)` closure captures the enclosing locals.
 * In C there is no closure capture, so the captured locals are bundled into this
 * struct threaded through the loader's `user` pointer and unpacked in the
 * descriptor callback. */
typedef struct dfcp_spec_desc_state
{
    ckc_value_t* k_off;
    int conv_tile_w;
    ckc_value_t* c_conv_tile_w;
    ckc_value_t* c_wi;
    ckc_value_t* c_c;
    ckc_value_t* c_sc;
    ckc_value_t* c_k_gemm;
    ckc_value_t* h_base;
    ckc_value_t* w_base;
    int pH;
    int pW;
    int Hi;
} dfcp_spec_desc_state;

/* descriptor(b_, row, col) -> (off_elems, valid) for the specialized loader. */
static ckc_value_t* dfcp_spec_descriptor(
    ckc_ir_builder_t* b_, ckc_value_t* row, ckc_value_t* col, ckc_value_t** out_valid, void* user)
{
    dfcp_spec_desc_state* st = (dfcp_spec_desc_state*)user;
    ckc_value_t* kg;
    ckc_value_t* local_oh;
    ckc_value_t* local_ow;
    ckc_value_t* r;
    ckc_value_t* rem;
    ckc_value_t* s_col;
    ckc_value_t* ci;
    ckc_value_t* hi;
    ckc_value_t* wi;
    ckc_value_t* h_ok;
    ckc_value_t* w_ok;
    ckc_value_t* kg_ok;
    ckc_value_t* valid;
    ckc_value_t* off_elems;

    kg = ckc_b_add(b_, st->k_off, col);
    if(st->conv_tile_w > 0 && (st->conv_tile_w & (st->conv_tile_w - 1)) == 0)
    {
        int shift = dfcp_bit_length(st->conv_tile_w - 1);
        local_oh = ckc_b_lshr(b_, row, ckc_b_const_i32(b_, shift));
        local_ow = ckc_b_land(b_, row, ckc_b_const_i32(b_, st->conv_tile_w - 1));
    }
    else
    {
        local_oh = ckc_b_div(b_, row, st->c_conv_tile_w);
        local_ow = ckc_b_mod(b_, row, st->c_conv_tile_w);
    }

    r = ckc_b_div(b_, kg, st->c_sc);
    rem = ckc_b_mod(b_, kg, st->c_sc);
    s_col = ckc_b_lshr(b_, rem, ckc_b_const_i32(b_, 3));
    ci = ckc_b_land(b_, rem, ckc_b_const_i32(b_, 7));

    /* hi = b.sub(b.add(b.add(h_base, local_oh), r), b.const_i32(pH))
     * Python evaluates the sub's args left-to-right: the add chain emits BEFORE
     * the const_i32(pH). C call-arg order is unspecified (GCC: right-to-left),
     * which would emit the const first and shift later SSA names. Sequence the
     * add chain into a temp to pin Python source-order. */
    {
        ckc_value_t* h_acc = ckc_b_add(b_, ckc_b_add(b_, st->h_base, local_oh), r);
        hi = ckc_b_sub(b_, h_acc, ckc_b_const_i32(b_, st->pH));
    }
    {
        ckc_value_t* w_acc = ckc_b_add(b_, ckc_b_add(b_, st->w_base, local_ow), s_col);
        wi = ckc_b_sub(b_, w_acc, ckc_b_const_i32(b_, st->pW));
    }
    /* Python evaluates land(cmp_ge(...), cmp_lt(...)) arguments left-to-right,
     * so the `icmp sge` is emitted BEFORE the `icmp slt`. C leaves argument
     * evaluation order unspecified (this compiler evaluates right-to-left, which
     * emitted the slt first), so sequence the two compares into temporaries to
     * pin Python's emit order and stay byte-identical. */
    {
        ckc_value_t* h_ge = ckc_b_cmp_ge(b_, hi, ckc_b_const_i32(b_, 0));
        ckc_value_t* h_lt = ckc_b_cmp_lt(b_, hi, ckc_b_const_i32(b_, st->Hi));
        h_ok = ckc_b_land(b_, h_ge, h_lt);
    }
    {
        ckc_value_t* w_ge = ckc_b_cmp_ge(b_, wi, ckc_b_const_i32(b_, 0));
        ckc_value_t* w_lt = ckc_b_cmp_lt(b_, wi, st->c_wi);
        w_ok = ckc_b_land(b_, w_ge, w_lt);
    }
    kg_ok = ckc_b_cmp_lt(b_, kg, st->c_k_gemm);
    valid = ckc_b_land(b_, kg_ok, ckc_b_land(b_, h_ok, w_ok));
    off_elems
        = ckc_b_add(b_, ckc_b_mul(b_, ckc_b_add(b_, ckc_b_mul(b_, hi, st->c_wi), wi), st->c_c), ci);

    *out_valid = valid;
    return off_elems;
}

/* _load_conv0_a_tile_specialized(b, spec, conv_spec, k_off, a_dst, grid, a_rsrc).
 * Specialized NHWC conv0 A loader for the fixed deep-fusion target
 * (N=1, C=8, R=S=3, stride=1, pad=1). */
void ckc_dfcp_load_conv0_a_tile_specialized(ckc_ir_builder_t* b,
                                            const ckc_deep_fused_conv_pool_spec_t* spec,
                                            const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                            ckc_value_t* k_off,
                                            ckc_value_t* a_dst,
                                            const ckc_warp_grid_t* grid,
                                            ckc_value_t* a_rsrc)
{
    const ckc_fused_conv_pool_problem_t* p;
    const ckc_conv_problem_t* c;
    int conv_tile_w;
    int load_vec;
    ckc_coalesced_tile_loader_t loader;
    dfcp_spec_desc_state st;

    (void)conv_spec; /* Python carries it for parity but the body never reads it. */

    if(b == NULL || spec == NULL || grid == NULL)
    {
        return;
    }
    p = &spec->problem;
    c = &p->conv;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;

    st.k_off = k_off;
    st.conv_tile_w = conv_tile_w;
    st.c_conv_tile_w = ckc_b_const_i32(b, conv_tile_w);
    st.c_wi = ckc_b_const_i32(b, c->Wi);
    st.c_c = ckc_b_const_i32(b, c->C);
    st.c_sc = ckc_b_const_i32(b, c->X * c->C); /* 24 for the target shape. */
    st.c_k_gemm = ckc_b_const_i32(b, ckc_conv_problem_k_gemm(c));
    /* h_base = b.mul(b.block_id_y(), b.const_i32(...)); w_base likewise.
     * Python evaluates the mul's args left-to-right (block_id_* before the
     * const), so the block_id takes its SSA counter slot first. C call-arg order
     * is unspecified (GCC: right-to-left) and would swap the slots, shifting
     * every later numbered SSA name. Hoist the block_id calls into temps to pin
     * Python source-order. */
    {
        ckc_value_t* bid_y = ckc_b_block_id_y(b);
        st.h_base = ckc_b_mul(b, bid_y, ckc_b_const_i32(b, spec->pool_tile_h * p->pool_stride_h));
    }
    {
        ckc_value_t* bid_z = ckc_b_block_id_z(b);
        st.w_base = ckc_b_mul(b, bid_z, ckc_b_const_i32(b, spec->pool_tile_w * p->pool_stride_w));
    }
    st.pH = c->pH;
    st.pW = c->pW;
    st.Hi = c->Hi;

    /* CoalescedTileLoader(tile_rows=tile_m, tile_cols=tile_k,
     *                     block_size=block_size,
     *                     load_vec=choose_vec(...,max_vec=8)). */
    if(ckc_coalesced_tile_loader_choose_vec(
           spec->tile_m, spec->tile_k, ckc_deep_fused_conv_pool_spec_block_size(spec), 8, &load_vec)
       != CKC_OK)
    {
        return; /* choose_vec records its ValueError; nothing more to emit. */
    }
    loader.tile_rows = spec->tile_m;
    loader.tile_cols = spec->tile_k;
    loader.block_size = ckc_deep_fused_conv_pool_spec_block_size(spec);
    loader.load_vec = load_vec;
    loader.use_buffer_rsrc = true;
    loader.oob_sentinel = 0x7fffffff; /* Python (1 << 31) - 1 */
    loader.has_inner_dim = false;
    loader.inner_dim = 0;

    /* loader.load(b, tid=grid.tid, smem_dst=a_dst, descriptor=descriptor,
     *             rsrc=a_rsrc). */
    ckc_coalesced_tile_loader_load(b,
                                   &loader,
                                   grid->tid,
                                   a_dst,
                                   dfcp_spec_descriptor,
                                   &st,
                                   a_rsrc,
                                   /*ptr=*/NULL);
}

/* ------------------------------------------------------------------ *
 * _setup_input_footprint_cache
 * ------------------------------------------------------------------ *
 *
 * Load the unique conv0 input footprint for this pooled-output tile. */
ckc_value_t* ckc_dfcp_setup_input_footprint_cache(ckc_ir_builder_t* b,
                                                  const ckc_deep_fused_conv_pool_spec_t* spec,
                                                  ckc_value_t* a_rsrc,
                                                  const ckc_warp_grid_t* grid)
{
    const ckc_fused_conv_pool_problem_t* p;
    const ckc_conv_problem_t* c;
    int conv_tile_h;
    int conv_tile_w;
    int foot_h;
    int foot_w;
    int total;
    int block_size;
    int elems_per_thread;
    int shape[2];
    ckc_value_t* input_smem;
    ckc_value_t* c_total;
    ckc_value_t* c_c;
    ckc_value_t* c_foot_w;
    ckc_value_t* c_half_bytes;
    ckc_value_t* oob;
    ckc_value_t* h_base;
    ckc_value_t* w_base;
    int e;

    if(b == NULL || spec == NULL || grid == NULL)
    {
        return NULL;
    }
    p = &spec->problem;
    c = &p->conv;
    conv_tile_h = spec->pool_tile_h * p->pool_stride_h;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    foot_h = conv_tile_h + (c->Y - 1) * c->dH;
    foot_w = conv_tile_w + (c->X - 1) * c->dW;

    shape[0] = foot_h * foot_w;
    shape[1] = c->C;
    input_smem = ckc_b_smem_alloc(b, ckc_f16(), shape, 2, "InputFoot_smem");

    total = foot_h * foot_w * c->C;
    block_size = ckc_deep_fused_conv_pool_spec_block_size(spec);
    elems_per_thread = (total + block_size - 1) / block_size;
    c_total = ckc_b_const_i32(b, total);
    c_c = ckc_b_const_i32(b, c->C);
    c_foot_w = ckc_b_const_i32(b, foot_w);
    c_half_bytes = ckc_b_const_i32(b, 2);
    oob = ckc_b_const_i32(b, 0x7fffffff); /* Python (1 << 31) - 1 */
    /* h_base = b.sub(b.mul(b.block_id_y(), b.const_i32(...)), b.const_i32(pH))
     * Python evaluates left-to-right: block_id_y -> its const -> the mul -> the
     * pH const -> the sub. C call-arg order is unspecified (GCC: right-to-left),
     * which both swaps the mul's two args AND evaluates the sub's pH const before
     * the mul. Hoist block_id AND the mul into temps to pin Python source-order
     * so later SSA names line up (the pH const must take its slot AFTER the mul). */
    {
        ckc_value_t* bid_y = ckc_b_block_id_y(b);
        ckc_value_t* mul_y
            = ckc_b_mul(b, bid_y, ckc_b_const_i32(b, spec->pool_tile_h * p->pool_stride_h));
        h_base = ckc_b_sub(b, mul_y, ckc_b_const_i32(b, c->pH));
    }
    {
        ckc_value_t* bid_z = ckc_b_block_id_z(b);
        ckc_value_t* mul_z
            = ckc_b_mul(b, bid_z, ckc_b_const_i32(b, spec->pool_tile_w * p->pool_stride_w));
        w_base = ckc_b_sub(b, mul_z, ckc_b_const_i32(b, c->pW));
    }

    for(e = 0; e < elems_per_thread; e++)
    {
        ckc_value_t* idx;
        ckc_value_t* idx_ok;
        ckc_value_t* safe_idx;
        ckc_value_t* ci;
        ckc_value_t* t0;
        ckc_value_t* local_w;
        ckc_value_t* local_h;
        ckc_value_t* global_h;
        ckc_value_t* global_w;
        ckc_value_t* h_ok;
        ckc_value_t* w_ok;
        ckc_value_t* valid;
        ckc_value_t* off_elems;
        ckc_value_t* off_bytes;
        ckc_value_t* safe_off;
        ckc_value_t* v;
        ckc_value_t* idxv[2];

        {
            /* hoist mul operands in Python's left-to-right order */
            ckc_value_t* e_c = ckc_b_const_i32(b, e);
            ckc_value_t* bs_c = ckc_b_const_i32(b, block_size);
            idx = ckc_b_add(b, ckc_b_mul(b, e_c, bs_c), grid->tid);
        }
        idx_ok = ckc_b_cmp_lt(b, idx, c_total);
        safe_idx = ckc_b_select(b, idx_ok, idx, ckc_b_const_i32(b, 0));
        ci = ckc_b_mod(b, safe_idx, c_c);
        t0 = ckc_b_div(b, safe_idx, c_c);
        local_w = ckc_b_mod(b, t0, c_foot_w);
        local_h = ckc_b_div(b, t0, c_foot_w);
        global_h = ckc_b_add(b, h_base, local_h);
        global_w = ckc_b_add(b, w_base, local_w);
        /* Pin Python's left-to-right land() arg order (sge before slt); C
         * argument evaluation order is unspecified. */
        {
            ckc_value_t* h_ge = ckc_b_cmp_ge(b, global_h, ckc_b_const_i32(b, 0));
            ckc_value_t* h_lt = ckc_b_cmp_lt(b, global_h, ckc_b_const_i32(b, c->Hi));
            h_ok = ckc_b_land(b, h_ge, h_lt);
        }
        {
            ckc_value_t* w_ge = ckc_b_cmp_ge(b, global_w, ckc_b_const_i32(b, 0));
            ckc_value_t* w_lt = ckc_b_cmp_lt(b, global_w, ckc_b_const_i32(b, c->Wi));
            w_ok = ckc_b_land(b, w_ge, w_lt);
        }
        valid = ckc_b_land(b, idx_ok, ckc_b_land(b, h_ok, w_ok));
        off_elems = ckc_b_add(
            b,
            ckc_b_mul(
                b, ckc_b_add(b, ckc_b_mul(b, global_h, ckc_b_const_i32(b, c->Wi)), global_w), c_c),
            ci);
        off_bytes = ckc_b_mul(b, off_elems, c_half_bytes);
        safe_off = ckc_b_select(b, valid, off_bytes, oob);
        v = ckc_b_buffer_load_f16(b, a_rsrc, safe_off, ckc_b_const_i32(b, 0));
        idxv[0] = t0;
        idxv[1] = ci;
        ckc_b_smem_store_f16(b, input_smem, idxv, 2, v);
    }

    ckc_b_sync(b);
    return input_smem;
}

/* ------------------------------------------------------------------ *
 * _load_conv0_a_tile_from_input_cache
 * ------------------------------------------------------------------ *
 *
 * Materialize the conv0 implicit-GEMM A tile from cached input footprint. */
void ckc_dfcp_load_conv0_a_tile_from_input_cache(ckc_ir_builder_t* b,
                                                 const ckc_deep_fused_conv_pool_spec_t* spec,
                                                 const ckc_implicit_gemm_conv_spec_t* conv_spec,
                                                 ckc_value_t* k_off,
                                                 ckc_value_t* a_dst,
                                                 const ckc_warp_grid_t* grid,
                                                 ckc_value_t* input_smem)
{
    const ckc_fused_conv_pool_problem_t* p;
    const ckc_conv_problem_t* c;
    int conv_tile_w;
    int foot_w;
    int total;
    int block_size;
    int elems_per_thread;
    ckc_value_t* c_total;
    ckc_value_t* c_tile_k;
    ckc_value_t* c_conv_tile_w;
    ckc_value_t* c_sc;
    ckc_value_t* c_c;
    ckc_value_t* c_foot_w;
    ckc_value_t* c_k_gemm;
    ckc_value_t* zero_h;
    int e;

    (void)conv_spec; /* parity arg; body never reads it. */

    if(b == NULL || spec == NULL || grid == NULL)
    {
        return;
    }
    p = &spec->problem;
    c = &p->conv;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    foot_w = conv_tile_w + (c->X - 1) * c->dW;
    total = spec->tile_m * spec->tile_k;
    block_size = ckc_deep_fused_conv_pool_spec_block_size(spec);
    elems_per_thread = (total + block_size - 1) / block_size;
    c_total = ckc_b_const_i32(b, total);
    c_tile_k = ckc_b_const_i32(b, spec->tile_k);
    c_conv_tile_w = ckc_b_const_i32(b, conv_tile_w);
    c_sc = ckc_b_const_i32(b, c->X * c->C);
    c_c = ckc_b_const_i32(b, c->C);
    c_foot_w = ckc_b_const_i32(b, foot_w);
    c_k_gemm = ckc_b_const_i32(b, ckc_conv_problem_k_gemm(c));
    zero_h = ckc_b_trunc_f32_to_f16(b, ckc_b_const_f32(b, 0.0));

    for(e = 0; e < elems_per_thread; e++)
    {
        ckc_value_t* idx;
        ckc_value_t* idx_ok;
        ckc_value_t* safe_idx;
        ckc_value_t* row;
        ckc_value_t* col;
        ckc_value_t* kg;
        ckc_value_t* kg_ok;
        ckc_value_t* local_oh;
        ckc_value_t* local_ow;
        ckc_value_t* r;
        ckc_value_t* rem;
        ckc_value_t* s_col;
        ckc_value_t* ci;
        ckc_value_t* ih;
        ckc_value_t* iw;
        ckc_value_t* foot_row;
        ckc_value_t* v;
        ckc_value_t* idxv[2];

        {
            /* hoist mul operands in Python's left-to-right order */
            ckc_value_t* e_c = ckc_b_const_i32(b, e);
            ckc_value_t* bs_c = ckc_b_const_i32(b, block_size);
            idx = ckc_b_add(b, ckc_b_mul(b, e_c, bs_c), grid->tid);
        }
        idx_ok = ckc_b_cmp_lt(b, idx, c_total);
        safe_idx = ckc_b_select(b, idx_ok, idx, ckc_b_const_i32(b, 0));
        row = ckc_b_div(b, safe_idx, c_tile_k);
        col = ckc_b_mod(b, safe_idx, c_tile_k);
        kg = ckc_b_add(b, k_off, col);
        kg_ok = ckc_b_cmp_lt(b, kg, c_k_gemm);

        local_oh = ckc_b_div(b, row, c_conv_tile_w);
        local_ow = ckc_b_mod(b, row, c_conv_tile_w);
        r = ckc_b_div(b, kg, c_sc);
        rem = ckc_b_mod(b, kg, c_sc);
        /* VALU opt: strength-reduce div/mod by C=8 to shift/mask. */
        if(c->C == 8)
        {
            s_col = ckc_b_lshr(b, rem, ckc_b_const_i32(b, 3));
            ci = ckc_b_land(b, rem, ckc_b_const_i32(b, 7));
        }
        else
        {
            s_col = ckc_b_div(b, rem, c_c);
            ci = ckc_b_mod(b, rem, c_c);
        }
        /* ih = b.add(b.mul(local_oh, sH), b.mul(r, dH)); iw similarly. Python
         * evaluates the left mul before the right; C call-arg order is
         * unspecified (GCC: right-to-left), which swaps the two mul SSA slots.
         * Hoist the left mul into a temp to pin Python source-order. */
        {
            ckc_value_t* mul_h = ckc_b_mul(b, local_oh, ckc_b_const_i32(b, c->sH));
            ih = ckc_b_add(b, mul_h, ckc_b_mul(b, r, ckc_b_const_i32(b, c->dH)));
        }
        {
            ckc_value_t* mul_w = ckc_b_mul(b, local_ow, ckc_b_const_i32(b, c->sW));
            iw = ckc_b_add(b, mul_w, ckc_b_mul(b, s_col, ckc_b_const_i32(b, c->dW)));
        }
        foot_row = ckc_b_add(b, ckc_b_mul(b, ih, c_foot_w), iw);
        {
            ckc_value_t* lidx[2];
            lidx[0] = foot_row;
            lidx[1] = ci;
            v = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f16(b, input_smem, lidx, 2, /*n=*/1), 0);
        }
        v = ckc_b_select(b, ckc_b_land(b, idx_ok, kg_ok), v, zero_h);
        idxv[0] = row;
        idxv[1] = col;
        ckc_b_smem_store_f16(b, a_dst, idxv, 2, v);
    }
}

/* ------------------------------------------------------------------ *
 * _load_conv0_a_operand_from_input_cache
 * ------------------------------------------------------------------ *
 *
 * Read one MFMA A operand fragment directly from the cached input footprint. */
ckc_value_t*
    ckc_dfcp_load_conv0_a_operand_from_input_cache(ckc_ir_builder_t* b,
                                                   const ckc_deep_fused_conv_pool_spec_t* spec,
                                                   ckc_value_t* row,
                                                   ckc_value_t* k_off,
                                                   ckc_value_t* col_base,
                                                   int frag_len,
                                                   ckc_value_t* input_smem)
{
    const ckc_fused_conv_pool_problem_t* p;
    const ckc_conv_problem_t* c;
    int conv_tile_w;
    int foot_w;
    ckc_value_t* c_conv_tile_w;
    ckc_value_t* c_sc;
    ckc_value_t* c_c;
    ckc_value_t* c_foot_w;
    ckc_value_t* c_k_gemm;
    ckc_value_t* zero_h;
    ckc_value_t* local_oh;
    ckc_value_t* local_ow;
    ckc_value_t* oh_base;
    ckc_value_t* ow_base;
    /* frag_len is the MFMA A-operand width (op.a_frag_len), bounded small; the
     * per-warp accumulator headroom is a safe upper bound for the scratch. */
    ckc_value_t* elems[CKC_DFCP_MAX_ACCS];
    ckc_value_t* packed;
    int i;

    if(b == NULL || spec == NULL || frag_len <= 0 || frag_len > CKC_DFCP_MAX_ACCS)
    {
        return NULL;
    }
    p = &spec->problem;
    c = &p->conv;
    conv_tile_w = spec->pool_tile_w * p->pool_stride_w;
    foot_w = conv_tile_w + (c->X - 1) * c->dW;
    c_conv_tile_w = ckc_b_const_i32(b, conv_tile_w);
    c_sc = ckc_b_const_i32(b, c->X * c->C);
    c_c = ckc_b_const_i32(b, c->C);
    c_foot_w = ckc_b_const_i32(b, foot_w);
    c_k_gemm = ckc_b_const_i32(b, ckc_conv_problem_k_gemm(c));
    zero_h = ckc_b_trunc_f32_to_f16(b, ckc_b_const_f32(b, 0.0));

    /* VALU opt 1: hoist row-dependent coordinates out of the per-element loop. */
    local_oh = ckc_b_div(b, row, c_conv_tile_w);
    local_ow = ckc_b_mod(b, row, c_conv_tile_w);

    oh_base = ckc_b_mul(b, local_oh, ckc_b_const_i32(b, c->sH));
    ow_base = ckc_b_mul(b, local_ow, ckc_b_const_i32(b, c->sW));

    for(i = 0; i < frag_len; i++)
    {
        ckc_value_t* kg;
        ckc_value_t* kg_ok;
        ckc_value_t* r;
        ckc_value_t* rem;
        ckc_value_t* s_col;
        ckc_value_t* ci;
        ckc_value_t* ih;
        ckc_value_t* iw;
        ckc_value_t* foot_row;
        ckc_value_t* raw;

        kg = ckc_b_add(b, k_off, ckc_b_add(b, col_base, ckc_b_const_i32(b, i)));
        kg_ok = ckc_b_cmp_lt(b, kg, c_k_gemm);
        r = ckc_b_div(b, kg, c_sc);
        rem = ckc_b_mod(b, kg, c_sc);

        /* VALU opt 2: strength-reduce div/mod by C=8 (power-of-2) to shift/mask. */
        if(c->C == 8)
        {
            s_col = ckc_b_lshr(b, rem, ckc_b_const_i32(b, 3));
            ci = ckc_b_land(b, rem, ckc_b_const_i32(b, 7));
        }
        else
        {
            s_col = ckc_b_div(b, rem, c_c);
            ci = ckc_b_mod(b, rem, c_c);
        }

        ih = ckc_b_add(b, oh_base, ckc_b_mul(b, r, ckc_b_const_i32(b, c->dH)));
        iw = ckc_b_add(b, ow_base, ckc_b_mul(b, s_col, ckc_b_const_i32(b, c->dW)));
        foot_row = ckc_b_add(b, ckc_b_mul(b, ih, c_foot_w), iw);
        {
            ckc_value_t* lidx[2];
            lidx[0] = foot_row;
            lidx[1] = ci;
            raw = ckc_b_vec_extract(b, ckc_b_smem_load_vN_f16(b, input_smem, lidx, 2, /*n=*/1), 0);
        }
        elems[i] = ckc_b_select(b, kg_ok, raw, zero_h);
    }
    /* b.vec_pack(elems, elems[0].type). */
    packed = ckc_b_vec_pack(b, elems, frag_len, elems[0]->type);
    return packed;
}
