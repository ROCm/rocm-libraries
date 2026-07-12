// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_forward_dynamic.c.c -- C99 port
 * of the DYNAMIC (host-roundtrip) FORWARD PATH of
 * rocke/instances/common/fused_moe_e2e.py.
 *
 * SCOPE (this TU):
 *   rocke_fmoe_forward_dynamic     <- FusedMoeForward._forward_dynamic
 *                                   (lines 1629-2058): prologue (workspace
 *       prepare + accumulator/scratch pre-zero), stage 1+2 router+sort, stage
 *       2.5 gather, the D->H copy of Counts/Offsets, the grouped-vs-uniform
 *       dispatch, the uniform-padded layout sizing (max_padded_m / total_padded
 *       + per-expert padded copies), stages 3-6 gate/up GEMM + silu_mul + down
 *       GEMM + topk_reduce with the active_tile_skip / preshuffle launcher +
 *       B-tensor selection, and stage 7 cast.
 *   rocke_fmoe_dispatch_grouped_gemm <- FusedMoeForward._dispatch_grouped_gemm
 *                                   (lines 1426-1582): the de-padded grouped
 *       path -- blocks_per_expert / num_m_blocks / total_packed, the dense
 *       tile-aligned packed layout + BlockExpertIds host build, and the grouped
 *       gate-up + down-reduce dispatch.
 *
 * The host-side SIZING + PACKING arithmetic is ported byte-faithfully: every
 * scalar (blocks_per_expert, num_m_blocks, total_packed, padded_counts_per_expert,
 * max_padded_m, total_padded, the gate/down grid dims, the per-call launcher /
 * B-tensor selections) is reproduced 1:1 against the source, and the cross-phase
 * results are stashed on the ctx working set so the (future) launch bodies and a
 * test harness can read exactly what the Python path computed. The actual HIP
 * launch chain, torch buffer zero_/fill_/copy_ and the D->H Counts/Offsets copy
 * are a runtime concern with no IR-builder analogue; they are marked TODO(port)
 * and the entries return ROCKE_ERR_NOTIMPL once the host-side bookkeeping is done.
 *
 * Peers (launcher ensures, weight-packing ensures, _ensure_compiled, the spec
 * getters, ctx lifecycle) live in sibling TUs and are reached only through
 * rocke/instance_fused_moe_e2e_internal.h. This TU does not emit IR.
 */

#include "rocke/instance_fused_moe_e2e_internal.h"

/* ===================================================================== *
 *  _dispatch_grouped_gemm (lines 1426-1582)
 *
 *  De-padded grouped GEMM for the dynamic path. Packs the routed tokens into a
 *  dense, tile_m-aligned per-expert layout and dispatches the grouped=True
 *  gate+up+silu and down-reduce kernels over a flat M-block grid.
 *
 *  *handled receives the Python bool return: true == it handled the forward
 *  (the caller returns), false == fall back to the batched padded path.
 * ===================================================================== */
rocke_status_t rocke_fmoe_dispatch_grouped_gemm(rocke_fmoe_build_ctx_t* ctx, bool* handled)
{
    if(ctx == NULL || handled == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    *handled = false;

    const rocke_fmoe_forward_spec_t* s = &ctx->spec;
    const int tile_m = s->gemm_tile.tile_m;
    /* Only the M (token) axis is packed here; the N axis (hidden / intermediate
     * columns) is tiled inside the GEMM kernel's grid, so tile_n is not needed
     * in this host-side de-padding packer. */

    if(s->experts < 0 || s->experts > ROCKE_FMOE_MAX_EXPERTS)
    {
        return ROCKE_ERR_VALUE;
    }
    if(tile_m <= 0)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Per-expert block counts + dense packed block layout (lines 1467-1475). */
    int num_m_blocks = 0;
    for(int e = 0; e < s->experts; ++e)
    {
        const int ce = (int)ctx->counts_cpu[e];
        const int be = (ce + tile_m - 1) / tile_m;
        ctx->blocks_per_expert[e] = be;
        num_m_blocks += be;
    }
    ctx->num_m_blocks = num_m_blocks;

    if(num_m_blocks == 0)
    {
        /* No tokens routed anywhere; output stays zero (line 1473). */
        /* TODO(port): Y.copy_(y_f32.to(Y.dtype)) */
        ctx->total_packed = 0;
        *handled = true;
        return ROCKE_ERR_NOTIMPL;
    }

    const int total_packed = num_m_blocks * tile_m;
    ctx->total_packed = total_packed;

    /* Per-call de-padded packed scratch buffers (lines 1477-1492). In this
     * codegen-only port the pool / tensor peers are opaque; allocation is a
     * runtime concern. The handle slots are recorded so a future HIP backend +
     * a host-side test can see exactly which buffers this path requests with
     * which sizes.
     *
     * TODO(port):
     *   ctx->ws.GroupedInputPacked   = pool.get_spec("GroupedInputPacked",
     *                                     (total_packed, s->hidden), act, dev)
     *   ctx->ws.HiddenPacked         = pool.get_spec("HiddenPacked",
     *                                     (total_packed, s->intermediate), act, dev)
     *   ctx->ws.SortedTokenIdsPacked = pool.get_spec("SortedTokenIdsPacked",
     *                                     (total_packed,), i32, dev)
     *   ctx->ws.SortedWeightsPacked  = pool.get_spec("SortedWeightsPacked",
     *                                     (total_packed,), f32, dev)
     *   ctx->ws.BlockExpertIds       = pool.get_spec("BlockExpertIds",
     *                                     (num_m_blocks,), i32, dev) */

    /* Padded tail rows of partial tiles must read zero A, carry token id -1 (so
     * down-reduce skips their atomic write), and zero weights (lines 1498-1500).
     * hidden_packed is fully overwritten by gate-up, so it needs no pre-zero.
     *
     * TODO(port):
     *   grouped_input_packed.zero_()
     *   sorted_token_ids_packed.fill_(-1)
     *   sorted_weights_packed.zero_() */

    /* Host-build BlockExpertIds + the dense per-expert copies (lines 1505-1525).
     * The packing arithmetic is reproduced exactly so a test can verify the
     * dense layout offsets; the copies themselves are device-buffer slice copies
     * (TODO(port)). blk walks packed M-blocks; poff = blk * tile_m. */
    {
        int blk = 0;
        for(int e = 0; e < s->experts; ++e)
        {
            const int be = ctx->blocks_per_expert[e];
            if(be == 0)
            {
                continue;
            }
            const int ce = (int)ctx->counts_cpu[e];
            const int uoff = (int)ctx->offsets_cpu[e];
            const int poff = blk * tile_m;
            (void)ce;
            (void)uoff;
            (void)poff;
            /* TODO(port):
             *   block_expert_cpu[blk : blk + be] = e
             *   grouped_input_packed[poff : poff + ce] =
             *       grouped_input[uoff : uoff + ce]
             *   sorted_token_ids_packed[poff : poff + ce] =
             *       sorted_token_ids[uoff : uoff + ce]
             *   sorted_weights_packed[poff : poff + ce] =
             *       sorted_weights[uoff : uoff + ce] */
            blk += be;
        }
        /* TODO(port): block_expert_ids.copy_(block_expert_cpu) */
    }

    /* Compile / fetch the grouped launchers (lines 1527-1528). */
    rocke_kernel_launcher_t* gate_up_launcher = rocke_fmoe_grouped_gate_up_silu_launcher(ctx);
    rocke_kernel_launcher_t* down_launcher = rocke_fmoe_grouped_down_reduce_launcher(ctx);
    (void)gate_up_launcher;
    (void)down_launcher;

    /* Resolve the grouped gate-up / down specs the grid sizers consume (lines
     * 1531/1534). Reproduced for parity; the grids + the block dim
     * (to_batched_gemm_spec().block_size, 1, 1) are computed in the launch body
     * (TODO(port)). */
    rocke_moe_gate_up_silu_gemm_spec_t gate_up_spec = rocke_fmoe_grouped_gate_up_spec(ctx);
    rocke_moe_down_reduce_gemm_spec_t down_spec = rocke_fmoe_grouped_down_spec(ctx);
    (void)gate_up_spec;
    (void)down_spec;

    /* TODO(port): the grouped launch chain (lines 1530-1581):
     *   gate_up_grid = moe_gate_up_silu_gemm_grouped_grid(num_m_blocks,
     *                      s->intermediate, gate_up_spec)
     *   down_grid    = moe_down_reduce_gemm_grouped_grid(num_m_blocks,
     *                      s->hidden, down_spec)
     *   gemm_block   = (to_batched_gemm_spec().block_size, 1, 1)
     *   gate_up_callable args:
     *     A=grouped_input_packed, WGate=W_gate, WUp=W_up, Hidden=hidden_packed,
     *     M=total_packed, N=s->intermediate, K=s->hidden,
     *     stride_a=0, stride_b=s->intermediate*s->hidden, stride_c=0,
     *     BlockExpertIds=block_expert_ids
     *   down_callable args:
     *     A=hidden_packed, WDown=W_down, SortedTokenIds=sorted_token_ids_packed,
     *     SortedWeights=sorted_weights_packed, Y=y_f32,
     *     M=total_packed, N=s->hidden, K=s->intermediate,
     *     stride_a=0, stride_b=s->hidden*s->intermediate, slot_size=0,
     *     tokens=s->tokens, BlockExpertIds=block_expert_ids
     *   launch_kernel(StreamConfig(stream), gate_up_callable, down_callable)
     *   Y.copy_(y_f32.to(Y.dtype)) */

    *handled = true; /* line 1582: return True */
    return ROCKE_ERR_NOTIMPL;
}

/* ===================================================================== *
 *  _forward_dynamic (lines 1629-2058)
 *
 *  The host-roundtrip path. Prologue (workspaces + pre-zero), stage 1+2 router +
 *  sort, stage 2.5 gather, D->H copy of Counts/Offsets, grouped-vs-uniform
 *  dispatch, stages 3-6 GEMM chain, stage 7 cast.
 * ===================================================================== */
rocke_status_t rocke_fmoe_forward_dynamic(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* _ensure_compiled() (line 1671): compile every component up front. */
    {
        const rocke_status_t st = rocke_fmoe_ensure_compiled(ctx);
        if(st != ROCKE_OK)
        {
            return st;
        }
    }

    const rocke_fmoe_forward_spec_t* s = &ctx->spec;
    const int tile_m = s->gemm_tile.tile_m;
    const int tile_n = s->gemm_tile.tile_n; /* (line 1926, used in stages 3-6) */

    if(s->experts < 0 || s->experts > ROCKE_FMOE_MAX_EXPERTS)
    {
        return ROCKE_ERR_VALUE;
    }
    if(tile_m <= 0)
    {
        return ROCKE_ERR_VALUE;
    }

    ctx->tile_m = tile_m;
    ctx->tile_n = tile_n;

    /* ---- prologue: resolve workspaces (lines 1675-1691) ---- *
     * device = X.device; ws = pool.prepare(workspace_specs(device)); unpack the
     * 15 named handles. The pool / tensor peers are opaque in this codegen-only
     * port, so the handle resolution is TODO(port); the spec table itself is
     * ported in rocke_fmoe_workspace_specs and consumed by a future HIP backend.
     *
     * TODO(port):
     *   ctx->device = X.device
     *   rocke_fmoe_workspace_specs(ctx, specs, ROCKE_FMOE_NUM_WORKSPACE_SPECS, &n)
     *   ws = pool.prepare(specs); unpack TopkIds, TopkWeights, Hist, Counter,
     *     Offsets, Counts, SortedTokenIds, SortedTopkIds, SortedWeights,
     *     GroupedInput, GateOut, UpOut, Hidden, DownOut, Y_f32 into ctx->ws.* */

    /* ---- pre-zero accumulator + per-expert GEMM A/C buffers (lines 1696-1708) ----
     * TODO(port):
     *   hist.zero_(); counter.zero_(); y_f32.zero_();
     *   grouped_input.zero_(); gate_out.zero_(); up_out.zero_();
     *   hidden_buf.zero_(); down_out.zero_(); */

    /* ---- warm the topk + grouped GEMM launchers (lines 1710-1718) ---- *
     * topk_launcher is referenced by the router callable below; the grouped
     * (single-launch-per-expert) launcher is warmed so a future toggle pays no
     * mid-loop compile cost (its result is intentionally discarded here). */
    rocke_kernel_launcher_t* topk_launcher = rocke_fmoe_ensure_topk_launcher(ctx);
    (void)topk_launcher;
    (void)rocke_fmoe_ensure_gemm_launcher(ctx);

    /* ---- Stage 1+2: router + sort (lines 1720-1766) ----
     * router_grid = topk_softmax_grid(s->tokens, to_topk_softmax_spec())
     * router_block = (s->router_block_size, 1, 1)
     * sort_callables = sort_launcher.make_callables({histogram, scan, scatter})
     * router_callable = make_kernel(topk_launcher,
     *     {X=routing_logits, Y=topk_weights, Idx=topk_ids, M=s->tokens,
     *      N=s->experts}, router_grid, router_block)
     *
     * ---- Stage 2.5: gather (unpadded layout) (lines 1767-1790) ----
     * gather_callable = make_kernel(fused_moe_launcher.gather,
     *     {X, SortedTokenIds=sorted_token_ids, GroupedInput=grouped_input,
     *      tokens=s->tokens, hidden=s->hidden},
     *     moe_gather_grid(to_fused_moe_spec()), (s->streaming_block_size,1,1))
     * launch_kernel(StreamConfig(stream), router_callable, *sort_callables,
     *               gather_callable)
     * TODO(port): the above launch chain. */

    /* ---- D->H copy of Counts / Offsets + stream sync (lines 1796-1798) ----
     * The per-expert GEMM dispatch needs Counts and Offsets on the host. After
     * the launch chain completes, copy the two (experts,) i32 buffers into
     * ctx->counts_cpu / ctx->offsets_cpu and synchronize so they are valid.
     *
     * TODO(port):
     *   counts_cpu  = counts.cpu()  -> ctx->counts_cpu[0..experts)
     *   offsets_cpu = offsets.cpu() -> ctx->offsets_cpu[0..experts)
     *   torch.cuda.synchronize() */

    /* ---- grouped-vs-uniform dispatch (lines 1800-1816) ---- */
    if(s->use_grouped_gemm)
    {
        bool done = false;
        const rocke_status_t st = rocke_fmoe_dispatch_grouped_gemm(ctx, &done);
        if(done)
        {
            /* The grouped path handled the forward (line 1816: return). Its
             * launch body is itself a TODO(port) returning ROCKE_ERR_NOTIMPL; the
             * host-side packing has been reproduced. */
            return st;
        }
        /* st without done is the fall-through error channel; ignore and continue
         * to the uniform-padded path (Python: done==False -> fall back). */
        (void)st;
    }

    /* ---- uniform per-expert padded layout sizing (lines 1818-1837) ----
     * padded_counts_per_expert[e] = ceil(count[e]/tile_m)*tile_m
     * max_padded_m = max_e padded_counts_per_expert[e] (floored at tile_m) */
    int max_padded_m = 0;
    for(int e = 0; e < s->experts; ++e)
    {
        const int ce = (int)ctx->counts_cpu[e];
        const int pc = ((ce + tile_m - 1) / tile_m) * tile_m;
        ctx->padded_counts_per_expert[e] = pc;
        if(pc > max_padded_m)
        {
            max_padded_m = pc;
        }
    }

    if(max_padded_m == 0)
    {
        /* Degenerate: every expert count=0 (no tokens routed). Output stays at
         * zero, return early (lines 1830-1834). */
        ctx->max_padded_m = 0;
        ctx->total_padded = 0;
        /* TODO(port): Y.copy_(y_f32.to(Y.dtype)) */
        return ROCKE_ERR_NOTIMPL;
    }
    if(max_padded_m < tile_m)
    {
        max_padded_m = tile_m; /* lines 1835-1836 */
    }
    const int total_padded = s->experts * max_padded_m; /* line 1837 */
    ctx->max_padded_m = max_padded_m;
    ctx->total_padded = total_padded;

    /* ---- uniform-padded scratch (lines 1841-1897) ----
     * act_dtype = grouped_input.dtype. Pool get_spec for:
     *   GroupedInputPaddedUniform (total_padded, s->hidden)       act
     *   GateOutPaddedUniform      (total_padded, s->intermediate) act
     *   UpOutPaddedUniform        (total_padded, s->intermediate) act
     *   HiddenPaddedUniform       (total_padded, s->intermediate) act
     *   DownOutPaddedUniform      (total_padded, s->hidden)       act
     *   SortedTokenIdsPaddedUniform (total_padded,) i32
     *   SortedWeightsPaddedUniform  (total_padded,) f32
     * TODO(port): allocate the above into ctx->ws.*PaddedUniform.
     *
     * ---- pre-zero / fill the uniform-padded scratch (lines 1899-1905) ----
     * TODO(port):
     *   grouped_input_padded.zero_(); gate_out_padded.zero_();
     *   up_out_padded.zero_(); hidden_padded.zero_(); down_out_padded.zero_();
     *   sorted_token_ids_padded.fill_(-1); sorted_weights_padded.zero_(); */

    /* ---- per-expert copy unpadded -> uniform padded layout (lines 1915-1923) ----
     * Each expert e copies count[e] rows from the contiguous gather output at
     * unpadded offset offsets[e] into the uniform slot at e*max_padded_m. The
     * offset arithmetic is reproduced for parity; the slice copies are device
     * copies (TODO(port)). */
    for(int e = 0; e < s->experts; ++e)
    {
        const int ce = (int)ctx->counts_cpu[e];
        if(ce == 0)
        {
            continue;
        }
        const int u = (int)ctx->offsets_cpu[e];
        const int p = e * max_padded_m;
        (void)u;
        (void)p;
        /* TODO(port):
         *   grouped_input_padded[p : p + ce] = grouped_input[u : u + ce]
         *   sorted_token_ids_padded[p : p + ce] = sorted_token_ids[u : u + ce]
         *   sorted_weights_padded[p : p + ce] = sorted_weights[u : u + ce] */
    }

    /* ---- batched GEMM launcher + block dim (lines 1925-1927) ---- */
    rocke_kernel_launcher_t* batched_gemm_launcher = rocke_fmoe_ensure_batched_gemm_launcher(ctx);
    /* gemm_block = (to_batched_gemm_spec().block_size, 1, 1). Resolve the block
     * size onto the ctx so the launch body + a test can read it. */
    {
        rocke_batched_gemm_spec_t bg;
        char bg_name[256];
        const rocke_status_t st
            = rocke_fmoe_forward_spec_to_batched_gemm_spec(s, bg_name, sizeof(bg_name), &bg);
        if(st != ROCKE_OK)
        {
            return st;
        }
        rocke_batched_gemm_spec_finalize(&bg);
        ctx->gemm_block_size = bg.block_size;
    }

    /* ---- Stages 3-6: gate/up GEMM + silu_mul + down GEMM + topk_reduce ---- *
     * Resolve the per-call launcher / B-tensor selections exactly as the Python
     * if/elif chains do, stashing them on the ctx working set so the (future)
     * launch body reads the precise selection the prologue made.
     *
     * gate_grid (lines 1936-1940):
     *   ((s->intermediate + tile_n - 1)/tile_n,
     *    (max_padded_m + tile_m - 1)/tile_m, s->experts) */

    /* gate_up_dyn_launcher (lines 1941-1947): active_tile_skip -> the
     * parameterized (preshuffle_b=False, active_tile_skip=True) launcher, else
     * the plain batched launcher. */
    if(s->active_tile_skip_gemms)
    {
        ctx->sel_gate_up_dyn_launcher
            = rocke_fmoe_moe_batched_gemm_launcher(ctx,
                                                   /*preshuffle_b=*/false,
                                                   /*active_tile_skip=*/true);
    }
    else
    {
        ctx->sel_gate_up_dyn_launcher = batched_gemm_launcher;
    }
    /* The gate / up callables both use gate_up_dyn_launcher with B=W_gate /
     * B=W_up and C=gate_out_padded / up_out_padded; _gate_up_args adds
     * SortedTokenIds=sorted_token_ids_padded + slot_size=max_padded_m when
     * active_tile_skip_gemms (lines 1949-1977). */

    /* down GEMM launcher + B-tensor selection (lines 1995-2010). */
    if(s->active_tile_skip_gemms)
    {
        ctx->sel_down_b_launcher
            = rocke_fmoe_moe_batched_gemm_launcher(ctx,
                                                   /*preshuffle_b=*/s->preshuffle_w_down,
                                                   /*active_tile_skip=*/true);
        ctx->sel_down_b_tensor = s->preshuffle_w_down
                                     ? rocke_fmoe_ensure_w_down_preshuffled(ctx, /*W_down=*/NULL)
                                     : NULL /* == W_down device ptr (TODO port) */;
    }
    else if(s->preshuffle_w_down)
    {
        ctx->sel_down_b_launcher = rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher(ctx);
        ctx->sel_down_b_tensor = rocke_fmoe_ensure_w_down_preshuffled(ctx, /*W_down=*/NULL);
    }
    else
    {
        ctx->sel_down_b_launcher = batched_gemm_launcher;
        ctx->sel_down_b_tensor = NULL /* == W_down device ptr (TODO port) */;
    }

    /* down_grid (lines 1990-1994):
     *   ((s->hidden + tile_n - 1)/tile_n,
     *    (max_padded_m + tile_m - 1)/tile_m, s->experts) */

    /* TODO(port): build the 5 callables + the single launch_kernel chain
     * (lines 1929-2052):
     *   gate_callable = make_kernel(gate_up_dyn_launcher,
     *       _gate_up_args(W_gate, gate_out_padded), gate_grid, gemm_block)
     *   up_callable   = make_kernel(gate_up_dyn_launcher,
     *       _gate_up_args(W_up, up_out_padded), gate_grid, gemm_block)
     *   silu_mul_callable = make_kernel(fused_moe_launcher.silu_mul,
     *       {GateOut=gate_out_padded, UpOut=up_out_padded, Hidden=hidden_padded,
     *        total_pairs=total_padded, intermediate=s->intermediate},
     *       (total_padded,1,1), (s->streaming_block_size,1,1))
     *   down_args = {A=hidden_padded, B=down_b_tensor, C=down_out_padded,
     *       M=max_padded_m, N=s->hidden, K=s->intermediate,
     *       stride_a=max_padded_m*s->intermediate,
     *       stride_b=s->hidden*s->intermediate,
     *       stride_c=max_padded_m*s->hidden}
     *       (+ SortedTokenIds=sorted_token_ids_padded, slot_size=max_padded_m
     *          when active_tile_skip_gemms)
     *   down_callable = make_kernel(down_b_launcher, down_args, down_grid,
     *       gemm_block)
     *   reduce_callable = make_kernel(fused_moe_launcher.topk_reduce,
     *       {DownOut=down_out_padded, SortedTokenIds=sorted_token_ids_padded,
     *        SortedWeights=sorted_weights_padded, Y=y_f32,
     *        total_pairs=total_padded, hidden=s->hidden, tokens=s->tokens},
     *       (total_padded,1,1), (s->streaming_block_size,1,1))
     *   launch_kernel(StreamConfig(stream), gate_callable, up_callable,
     *       silu_mul_callable, down_callable, reduce_callable) */

    /* ---- Stage 7: dtype cast (lines 2054-2058) ----
     * TODO(port): Y.copy_(y_f32.to(Y.dtype)) */

    return ROCKE_ERR_NOTIMPL;
}
