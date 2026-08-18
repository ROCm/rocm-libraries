// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_forward_static.c.c -- C99 port of
 * the STATIC (no-roundtrip) FORWARD PATH + HIP-graph capture/replay surface of
 * rocke/instances/common/fused_moe_e2e.py.
 *
 * SCOPE (this TU):
 *   rocke_fmoe_forward_static  <- FusedMoeForward._forward_static (lines 2175-2678):
 *       prologue (workspaces + one-shot static_offsets arange*slot_size + the
 *       per-call resets), resolve the use_experimental_fused / interleaved /
 *       down_reduce / static_sg path-selection booleans and the sel_* launcher /
 *       weight selections, build the router + route-stage (static_scatter_gather
 *       OR scatter+gather) + gate-stage (fused / interleaved / packed-gemm+silu)
 *       + down-stage (down_reduce OR down-gemm+topk_reduce) callables, the single
 *       launch_kernel chain, and the stage-7 cast.
 *   rocke_fmoe_capture_graph   <- FusedMoeForward.capture_graph  (lines 2064-2156)
 *   rocke_fmoe_replay_graph    <- FusedMoeForward.replay_graph   (lines 2158-2169)
 *
 * The path-selection decisions (the if/elif chains that choose booleans, the
 * sel_* launchers, and the cached weight tensors) are reproduced byte-faithfully
 * by driving the peer _ensure_* / _moe_* launcher caches through the internal
 * header and recording the result on ctx. The actual HIP launch chain, the torch
 * buffer resets (counter.zero_ / fill_(-1) / etc.), the workspace allocation, the
 * make_kernel/launch_kernel dispatch, and the graph capture have no IR-builder
 * analogue in this codegen-only library; they are marked TODO(port) and the
 * forward / capture / replay entries return ROCKE_ERR_NOTIMPL once the faithful
 * decisions have been made.
 *
 * Peers (ctx lifecycle, launcher ensures, host helpers, dynamic path) live in
 * sibling TUs and are reached only through
 * rocke/instance_fused_moe_e2e_internal.h. This TU does not emit IR.
 */

#include <string.h>

#include "rocke/instance_fused_moe_e2e_internal.h"

/* ------------------------------------------------------------------ *
 * _forward_static (lines 2175-2678)
 * ------------------------------------------------------------------ */

rocke_status_t rocke_fmoe_forward_static(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* self._ensure_compiled() (line 2206): compile every component up front. */
    {
        rocke_status_t st = rocke_fmoe_ensure_compiled(ctx);
        if(st != ROCKE_OK)
        {
            return st;
        }
    }

    const rocke_fmoe_forward_spec_t* s = &ctx->spec;

    /* slot_size = self._static_slot_size; total_padded = E * slot_size
     * (lines 2208-2209). */
    const int slot_size = ctx->static_slot_size;
    const int total_padded = s->experts * slot_size;
    ctx->slot_size = slot_size;
    ctx->total_padded = total_padded;

    /* device = X.device (line 2210). The opaque device handle is carried on ctx;
     * in this codegen-only library there is no torch device object, so the
     * workspace resolution / pool.prepare is a TODO(port). */

    /* ---- Workspace (lines 2213-2277) ----
     * ws = self._pool.prepare(self._workspace_specs(device)) then unpack the 15
     * named entries, plus the seven Static* padded get_spec buffers. The pool /
     * tensor peers are opaque forward-decls with no allocator wired here; the
     * resolved handles live on ctx->ws and stay NULL until a HIP backend wires
     * the pool. The _workspace_specs table itself IS faithfully ported (peer
     * rocke_fmoe_workspace_specs); we materialise it so the table-walk side effect
     * (and any spec ValueError) matches the Python prepare() call. */
    {
        rocke_fmoe_ws_spec_t ws_specs[ROCKE_FMOE_NUM_WORKSPACE_SPECS];
        size_t n_ws = 0;
        rocke_status_t st
            = rocke_fmoe_workspace_specs(ctx, ws_specs, ROCKE_FMOE_NUM_WORKSPACE_SPECS, &n_ws);
        if(st != ROCKE_OK)
        {
            return st;
        }
        /* TODO(port): self._pool.prepare(ws_specs) -> ctx->ws.{TopkIds,
         * TopkWeights, Counter, Y_f32, ...}; self._pool.get_spec(_WS("Static*",
         * ...)) -> ctx->ws.{StaticSortedTokenIdsPadded, StaticSortedTopkIdsPadded,
         * StaticSortedWeightsPadded, StaticGroupedInputPadded, StaticGateUpPacked,
         * StaticHiddenPadded, StaticDownOutPadded}. The Static* shapes are
         * (total_padded,) i32/f32 and (total_padded, hidden|intermediate|2*I) act
         * per lines 2222-2277. */
        (void)ws_specs;
        (void)n_ws;
    }

    /* ---- One-shot init (lines 2280-2283) ----
     * if self._static_offsets is None:
     *     self._static_offsets = arange(E, i32, device) * slot_size
     * i.e. fixed offsets [0, S, 2S, ..., (E-1)*S]. The arange*slot_size is a
     * device tensor with no C analogue; record that it has been materialised so
     * the route stage selects it (the value is [e*slot_size for e in range(E)]).
     * TODO(port): allocate + fill ctx->static_offsets on `device`. */
    if(ctx->static_offsets == NULL)
    {
        /* TODO(port): ctx->static_offsets =
         *     torch.arange(s->experts, i32, device) * slot_size; */
    }

    /* ---- Reset per-call state (lines 2292-2299) ----
     * counter.zero_(); sorted_token_ids_padded.fill_(-1);
     * sorted_weights_padded.zero_(); grouped_input_padded.zero_(); y_f32.zero_().
     * These are torch ops on the current stream with no IR analogue.
     * TODO(port): zero/fill ctx->ws.{Counter, StaticSortedTokenIdsPadded(-1),
     * StaticSortedWeightsPadded, StaticGroupedInputPadded, Y_f32}. */

    /* ---- Launcher ensures + path-selection booleans (lines 2301-2357) ---- */

    rocke_kernel_launcher_t* topk_launcher = rocke_fmoe_ensure_topk_launcher(ctx);
    rocke_kernel_launcher_t* batched_gemm_launcher = rocke_fmoe_ensure_batched_gemm_launcher(ctx);
    (void)topk_launcher; /* recorded via ctx->topk_launcher by the ensure peer */

    /* use_experimental_fused = bool(s.use_experimental_fused_gate_up_silu)
     * use_experimental_interleaved = interleaved && not fused
     * use_experimental_down_reduce = bool(...fused_down_reduce)
     * use_experimental_static_sg   = bool(...static_scatter_gather)
     * (lines 2303-2309). */
    const bool use_experimental_fused = s->use_experimental_fused_gate_up_silu;
    const bool use_experimental_interleaved
        = s->use_experimental_interleaved_gate_up_silu && !use_experimental_fused;
    const bool use_experimental_down_reduce = s->use_experimental_fused_down_reduce;
    const bool use_experimental_static_sg = s->use_experimental_static_scatter_gather;

    ctx->use_experimental_fused = use_experimental_fused;
    ctx->use_experimental_interleaved = use_experimental_interleaved;
    ctx->use_experimental_down_reduce = use_experimental_down_reduce;
    ctx->use_experimental_static_sg = use_experimental_static_sg;

    /* gate_up_silu_launcher = _ensure_gate_up_silu_launcher() if fused else None
     * (lines 2310-2312). */
    rocke_kernel_launcher_t* gate_up_silu_launcher
        = use_experimental_fused ? rocke_fmoe_ensure_gate_up_silu_launcher(ctx) : NULL;

    /* interleaved branch selection (lines 2313-2330):
     *   if active_tile_skip_gemms:
     *       _moe_interleaved_gate_up_silu_launcher(preshuffle_b=interleaved,
     *                                              active_tile_skip=True)
     *   elif preshuffle_w_gate_up_interleaved:
     *       _ensure_interleaved_gate_up_silu_preshuffle_launcher()
     *   else:
     *       _ensure_interleaved_gate_up_silu_launcher()
     *   else None. */
    rocke_kernel_launcher_t* interleaved_gate_up_silu_launcher = NULL;
    if(use_experimental_interleaved)
    {
        if(s->active_tile_skip_gemms)
        {
            interleaved_gate_up_silu_launcher = rocke_fmoe_moe_interleaved_gate_up_silu_launcher(
                ctx,
                /*preshuffle_b=*/s->preshuffle_w_gate_up_interleaved,
                /*active_tile_skip=*/true);
        }
        else if(s->preshuffle_w_gate_up_interleaved)
        {
            interleaved_gate_up_silu_launcher
                = rocke_fmoe_ensure_interleaved_gate_up_silu_preshuffle_launcher(ctx);
        }
        else
        {
            interleaved_gate_up_silu_launcher
                = rocke_fmoe_ensure_interleaved_gate_up_silu_launcher(ctx);
        }
    }

    /* down_reduce_launcher = _ensure_down_reduce_launcher() if down_reduce else
     * None (lines 2331-2335). */
    rocke_kernel_launcher_t* down_reduce_launcher
        = use_experimental_down_reduce ? rocke_fmoe_ensure_down_reduce_launcher(ctx) : NULL;

    /* silu_mul_packed_launcher = None if fused else _ensure_silu_mul_packed (2336-2338). */
    rocke_kernel_launcher_t* silu_mul_packed_launcher
        = use_experimental_fused ? NULL : rocke_fmoe_ensure_silu_mul_packed_launcher(ctx);

    /* static_scatter_gather_launcher = _ensure_static_scatter_gather_launcher()
     * (line 2339): ALWAYS built (used only when use_experimental_static_sg). */
    rocke_kernel_launcher_t* static_scatter_gather_launcher
        = rocke_fmoe_ensure_static_scatter_gather_launcher(ctx);

    /* sort_launchers = None if static_sg else self._sort_launcher._ensure_launchers()
     * fmoe_launchers = self._fused_moe_launcher._ensure_launchers()
     * (lines 2340-2345). The sub-launcher ensure on the cached sort/fused_moe
     * launcher objects is a HIP-runtime concern; we record the SELECTION
     * (None vs the cached launcher) on ctx. */
    rocke_moe_sorting_launcher_t* sort_launchers
        = use_experimental_static_sg ? NULL : ctx->sort_launcher;
    rocke_fused_moe_launcher_t* fmoe_launchers = ctx->fused_moe_launcher;
    /* TODO(port): drive sort_launcher->_ensure_launchers() /
     * fused_moe_launcher->_ensure_launchers() so their HSACOs are compiled. */

    /* gu_concat = None if (fused or interleaved) else _ensure_gu_concat(Wg, Wu)
     * (lines 2346-2350). */
    rocke_tensor_t* gu_concat = NULL;
    if(!(use_experimental_fused || use_experimental_interleaved))
    {
        gu_concat = rocke_fmoe_ensure_gu_concat(
            ctx, (rocke_tensor_t*)(uintptr_t)ctx->W_gate, (rocke_tensor_t*)(uintptr_t)ctx->W_up);
    }

    /* gu_interleaved selection (lines 2351-2357):
     *   if interleaved:
     *       if preshuffle_w_gate_up_interleaved:
     *           _ensure_gu_interleaved_preshuffled(Wg, Wu)
     *       else: _ensure_gu_interleaved(Wg, Wu)
     *   else None. */
    rocke_tensor_t* gu_interleaved = NULL;
    if(use_experimental_interleaved)
    {
        if(s->preshuffle_w_gate_up_interleaved)
        {
            gu_interleaved = rocke_fmoe_ensure_gu_interleaved_preshuffled(
                ctx,
                (rocke_tensor_t*)(uintptr_t)ctx->W_gate,
                (rocke_tensor_t*)(uintptr_t)ctx->W_up);
        }
        else
        {
            gu_interleaved
                = rocke_fmoe_ensure_gu_interleaved(ctx,
                                                   (rocke_tensor_t*)(uintptr_t)ctx->W_gate,
                                                   (rocke_tensor_t*)(uintptr_t)ctx->W_up);
        }
    }

    /* Record the resolved selections on ctx so the (TODO) launch chain reads the
     * exact selection this prologue made. */
    ctx->sel_gate_up_silu_launcher = gate_up_silu_launcher;
    ctx->sel_interleaved_gate_up_silu_launcher = interleaved_gate_up_silu_launcher;
    ctx->sel_down_reduce_launcher = down_reduce_launcher;
    ctx->sel_silu_mul_packed_launcher = silu_mul_packed_launcher;
    ctx->sel_static_scatter_gather_launcher = static_scatter_gather_launcher;
    ctx->sel_sort_launchers = sort_launchers;
    ctx->sel_fmoe_launchers = fmoe_launchers;
    ctx->sel_gu_concat = gu_concat;
    ctx->sel_gu_interleaved = gu_interleaved;

    /* ---- Build callables (lines 2359-2656) ----
     * Each make_kernel(...) binds a launcher + a kwarg dict + grid + block. None
     * of make_kernel / launch_kernel has an IR-builder analogue here, so the
     * callable construction is TODO(port). The decisions that select WHICH
     * callables get built -- and their tile-derived grids -- are reproduced
     * below as the value-producing surface. */

    /* router (lines 2360-2373): grid = topk_softmax_grid(tokens, to_topk_softmax_spec)
     * block = (router_block_size, 1, 1); make_kernel(topk_launcher, {X, Y, Idx,
     * M=tokens, N=experts}, grid, block). */
    /* TODO(port): router_callable = make_kernel(ctx->topk_launcher, {...},
     *   topk_softmax_grid(s->tokens, to_topk_softmax_spec), (router_block,1,1)); */

    /* route stage (lines 2375-2434):
     *   if use_experimental_static_sg:
     *       [make_kernel(static_scatter_gather_launcher, {TopkIds, TopkWeights,
     *           Counter, X, SortedTokenIds, SortedWeights, GroupedInput,
     *           tokens, topk, num_experts, hidden, slot_size},
     *           moe_static_scatter_gather_grid(fmoe_spec),
     *           (streaming_block_size,1,1))]
     *   else:
     *       scatter_callable + gather_callable (2 callables) keyed off
     *       sort_launchers["scatter"] + fmoe_launchers["gather"], using
     *       self._static_offsets as the fixed Offsets. */
    /* TODO(port): build route_stage_callables per the branch above. The branch
     * key is ctx->use_experimental_static_sg (recorded). */

    /* gate stage (lines 2436-2569):
     *   tile_m = gemm_tile.tile_m; tile_n = gemm_tile.tile_n
     *   gemm_block = (to_batched_gemm_spec().block_size, 1, 1)
     *   if use_experimental_fused:   one fused gate+up+silu MFMA callable
     *   elif use_experimental_interleaved: one interleaved callable
     *   else: packed gate+up GEMM (N=2*I) + silu_mul (2 callables). */
    const int tile_m = s->gemm_tile.tile_m;
    const int tile_n = s->gemm_tile.tile_n;
    ctx->tile_m = tile_m;
    ctx->tile_n = tile_n;
    /* gate_up_n = 2 * intermediate (line 2516) is the packed gate-up N used by
     * the fast-default branch and the StaticGateUpPacked buffer (2*I columns). */
    ctx->gate_up_n = 2 * s->intermediate;

    if(use_experimental_fused)
    {
        /* FusedGateUpSiluGemmSpec(name=<name>_gate_up_silu, tile=gemm_tile,
         *   trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
         *   dtype=_gemm_dtype_to_universal(dtype)) (lines 2440-2445).
         * grid = moe_gate_up_silu_gemm_grid(E, slot_size, I, spec). */
        /* TODO(port): gate_stage_callables = [make_kernel(gate_up_silu_launcher,
         *   {A=GroupedInputPadded, WGate, WUp, Hidden, M=slot_size, N=I, K=hidden,
         *    stride_a=slot_size*hidden, stride_b=I*hidden, stride_c=slot_size*I},
         *   moe_gate_up_silu_gemm_grid(E, slot_size, I, spec),
         *   (spec.block_size,1,1))]. */
    }
    else if(use_experimental_interleaved)
    {
        /* FusedInterleavedGateUpSiluGemmSpec(..., trait active_tile_skip=
         *   s.active_tile_skip_gemms) (lines 2475-2485). gate_up_args adds
         * SortedTokenIds + slot_size when active_tile_skip (lines 2497-2499). */
        /* TODO(port): gate_stage_callables = [make_kernel(
         *   interleaved_gate_up_silu_launcher, {A=GroupedInputPadded,
         *   WGateUp=gu_interleaved, Hidden, M=slot_size, N=I, K=hidden,
         *   stride_a=slot_size*hidden, stride_b=(2*I)*hidden,
         *   stride_c=slot_size*I [, SortedTokenIds, slot_size]},
         *   moe_interleaved_gate_up_silu_gemm_grid(E, slot_size, I, spec),
         *   (spec.block_size,1,1))]. */
    }
    else
    {
        /* Fast default: one batched GEMM (N=2*I) -> StaticGateUpPacked, then a
         * streaming silu_mul (lines 2510-2569). The gate-up B launcher + tensor
         * are selected by the active_tile_skip / preshuffle chain (2522-2537);
         * record the selection on ctx. */
        const int gate_up_n = ctx->gate_up_n;
        (void)gate_up_n; /* gate_up_grid = (ceil(2I/tile_n), ceil(slot/tile_m), E) */

        rocke_kernel_launcher_t* gate_up_b_launcher = NULL;
        rocke_tensor_t* gate_up_b_tensor = NULL;
        if(s->active_tile_skip_gemms)
        {
            gate_up_b_launcher = rocke_fmoe_moe_batched_gemm_launcher(
                ctx,
                /*preshuffle_b=*/s->preshuffle_w_gate_up_packed,
                /*active_tile_skip=*/true);
            gate_up_b_tensor = s->preshuffle_w_gate_up_packed
                                   ? rocke_fmoe_ensure_gu_concat_preshuffled(
                                         ctx,
                                         (rocke_tensor_t*)(uintptr_t)ctx->W_gate,
                                         (rocke_tensor_t*)(uintptr_t)ctx->W_up)
                                   : gu_concat;
        }
        else if(s->preshuffle_w_gate_up_packed)
        {
            gate_up_b_launcher = rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher(ctx);
            gate_up_b_tensor
                = rocke_fmoe_ensure_gu_concat_preshuffled(ctx,
                                                          (rocke_tensor_t*)(uintptr_t)ctx->W_gate,
                                                          (rocke_tensor_t*)(uintptr_t)ctx->W_up);
        }
        else
        {
            gate_up_b_launcher = batched_gemm_launcher;
            gate_up_b_tensor = gu_concat;
        }
        ctx->sel_gate_up_b_launcher = gate_up_b_launcher;
        ctx->sel_gate_up_b_tensor = gate_up_b_tensor;
        /* TODO(port): gate_up_callable = make_kernel(gate_up_b_launcher,
         *   {A=GroupedInputPadded, B=gate_up_b_tensor, C=GateUpPacked, M=slot_size,
         *    N=gate_up_n, K=hidden, stride_a=slot_size*hidden,
         *    stride_b=gate_up_n*hidden, stride_c=slot_size*gate_up_n
         *    [, SortedTokenIds, slot_size]}, gate_up_grid, gemm_block);
         *   silu_mul_callable = make_kernel(silu_mul_packed_launcher,
         *   {GateUp=GateUpPacked, Hidden, total_pairs=total_padded,
         *    intermediate=I}, (total_padded,1,1), (streaming_block,1,1)). */
    }

    /* down stage (lines 2570-2656):
     *   down_grid = (ceil(hidden/tile_n), ceil(slot/tile_m), E)
     *   if use_experimental_down_reduce: one fused down+reduce callable
     *   else: down GEMM + topk_reduce (2 callables), B launcher/tensor selected
     *         by the active_tile_skip / preshuffle chain (2606-2621). */
    if(use_experimental_down_reduce)
    {
        /* FusedDownReduceGemmSpec(...) (lines 2576-2581). */
        /* TODO(port): down_stage_callables = [make_kernel(down_reduce_launcher,
         *   {A=Hidden, WDown=W_down, SortedTokenIds, SortedWeights, Y=Y_f32,
         *    M=slot_size, N=hidden, K=I, stride_a=slot_size*I, stride_b=hidden*I,
         *    slot_size, tokens}, moe_down_reduce_gemm_grid(E, slot_size, hidden,
         *    spec), (spec.block_size,1,1))]. */
    }
    else
    {
        rocke_kernel_launcher_t* down_b_launcher = NULL;
        rocke_tensor_t* down_b_tensor = NULL;
        if(s->active_tile_skip_gemms)
        {
            down_b_launcher
                = rocke_fmoe_moe_batched_gemm_launcher(ctx,
                                                       /*preshuffle_b=*/s->preshuffle_w_down,
                                                       /*active_tile_skip=*/true);
            down_b_tensor = s->preshuffle_w_down ? rocke_fmoe_ensure_w_down_preshuffled(
                                                       ctx, (rocke_tensor_t*)(uintptr_t)ctx->W_down)
                                                 : (rocke_tensor_t*)(uintptr_t)ctx->W_down;
        }
        else if(s->preshuffle_w_down)
        {
            down_b_launcher = rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher(ctx);
            down_b_tensor = rocke_fmoe_ensure_w_down_preshuffled(
                ctx, (rocke_tensor_t*)(uintptr_t)ctx->W_down);
        }
        else
        {
            down_b_launcher = batched_gemm_launcher;
            down_b_tensor = (rocke_tensor_t*)(uintptr_t)ctx->W_down;
        }
        ctx->sel_down_b_launcher = down_b_launcher;
        ctx->sel_down_b_tensor = down_b_tensor;
        /* TODO(port): down_callable = make_kernel(down_b_launcher,
         *   {A=Hidden, B=down_b_tensor, C=DownOut, M=slot_size, N=hidden, K=I,
         *    stride_a=slot_size*I, stride_b=hidden*I, stride_c=slot_size*hidden
         *    [, SortedTokenIds, slot_size]}, down_grid, gemm_block);
         *   reduce_callable = make_kernel(fmoe_launchers["topk_reduce"],
         *   {DownOut, SortedTokenIds, SortedWeights, Y=Y_f32,
         *    total_pairs=total_padded, hidden, tokens}, (total_padded,1,1),
         *    (streaming_block,1,1)). */
    }

    /* ---- Single launch_kernel for the whole forward (lines 2671-2677) ----
     *   launch_kernel(StreamConfig(stream_id=int(stream)), router_callable,
     *       *route_stage_callables, *gate_stage_callables, *down_stage_callables)
     * The same-stream FIFO + static offsets carry the data flow:
     *   router -> route(scatter+gather) -> gate/up -> down GEMM -> topk_reduce.
     * No host roundtrip. */
    /* TODO(port): launch the chain on StreamConfig(stream_id=ctx->stream). */

    /* ---- stage 7 cast (line 2678): Y.copy_(y_f32.to(Y.dtype)). ---- */
    /* TODO(port): cast Y_f32 -> Y (ctx->Y) in the activation dtype. */

    /* The faithful path-selection decisions above are complete; the HIP launch
     * chain + buffer resets + stage-7 cast are runtime concerns with no IR
     * analogue in this codegen-only library. */
    return ROCKE_ERR_NOTIMPL;
}

/* ------------------------------------------------------------------ *
 * capture_graph (lines 2064-2156)
 * ------------------------------------------------------------------ */

rocke_status_t rocke_fmoe_capture_graph(rocke_fmoe_build_ctx_t* ctx,
                                        uint64_t routing_logits,
                                        uint64_t X,
                                        uint64_t W_gate,
                                        uint64_t W_up,
                                        uint64_t W_down,
                                        uint64_t Y,
                                        int warmup_iters)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* if not self._use_static_offsets: raise RuntimeError (lines 2098-2102).
     * capture_graph is valid only in static-offset mode. Map the Python
     * RuntimeError to ROCKE_ERR_VALUE. */
    if(!ctx->use_static_offsets)
    {
        return ROCKE_ERR_VALUE;
    }

    /* self._ensure_compiled() (line 2105). */
    {
        rocke_status_t st = rocke_fmoe_ensure_compiled(ctx);
        if(st != ROCKE_OK)
        {
            return st;
        }
    }

    /* Record the inputs the capture is bound to (the caller contract: the same
     * pointers must be used on every replay). These feed the warmup +
     * captured _forward_static calls. */
    ctx->routing_logits = routing_logits;
    ctx->X = X;
    ctx->W_gate = W_gate;
    ctx->W_up = W_up;
    ctx->W_down = W_down;
    ctx->Y = Y;

    /* Eagerly build the packed gate/up weights for paths that need a cached
     * packed B tensor so the packing does not happen inside the captured region
     * (lines 2106-2122):
     *   if use_experimental_fused_gate_up_silu: pass
     *   elif use_experimental_interleaved_gate_up_silu:
     *       if preshuffle_w_gate_up_interleaved: _ensure_gu_interleaved_preshuffled
     *       else: _ensure_gu_interleaved
     *   else:
     *       if preshuffle_w_gate_up_packed: _ensure_gu_concat_preshuffled
     *       else: _ensure_gu_concat
     *   if preshuffle_w_down: _ensure_w_down_preshuffled. */
    const rocke_fmoe_forward_spec_t* s = &ctx->spec;
    rocke_tensor_t* Wg = (rocke_tensor_t*)(uintptr_t)W_gate;
    rocke_tensor_t* Wu = (rocke_tensor_t*)(uintptr_t)W_up;
    rocke_tensor_t* Wd = (rocke_tensor_t*)(uintptr_t)W_down;

    if(s->use_experimental_fused_gate_up_silu)
    {
        /* pass */
    }
    else if(s->use_experimental_interleaved_gate_up_silu)
    {
        if(s->preshuffle_w_gate_up_interleaved)
        {
            (void)rocke_fmoe_ensure_gu_interleaved_preshuffled(ctx, Wg, Wu);
        }
        else
        {
            (void)rocke_fmoe_ensure_gu_interleaved(ctx, Wg, Wu);
        }
    }
    else
    {
        if(s->preshuffle_w_gate_up_packed)
        {
            (void)rocke_fmoe_ensure_gu_concat_preshuffled(ctx, Wg, Wu);
        }
        else
        {
            (void)rocke_fmoe_ensure_gu_concat(ctx, Wg, Wu);
        }
    }
    if(s->preshuffle_w_down)
    {
        (void)rocke_fmoe_ensure_w_down_preshuffled(ctx, Wd);
    }

    /* Warmup on a side stream + capture into a torch.cuda.CUDAGraph
     * (lines 2124-2156). Both the side-stream warmup (max(1, warmup_iters)
     * _forward_static calls) and the graph capture require a HIP runtime; there
     * is no IR-builder analogue in this codegen-only library.
     * TODO(port):
     *   capture_stream = torch.cuda.Stream();
     *   capture_stream.wait_stream(current_stream());
     *   for _ in range(max(1, warmup_iters)):
     *       _forward_static(..., stream=capture_stream);
     *   current_stream().wait_stream(capture_stream); synchronize();
     *   graph = torch.cuda.CUDAGraph();
     *   with torch.cuda.graph(graph, stream=capture_stream):
     *       _forward_static(..., stream=capture_stream);
     *   ctx->captured_graph = graph; ctx->captured_graph_stream = capture_stream. */
    (void)warmup_iters;

    return ROCKE_ERR_NOTIMPL;
}

/* ------------------------------------------------------------------ *
 * replay_graph (lines 2158-2169)
 * ------------------------------------------------------------------ */

rocke_status_t rocke_fmoe_replay_graph(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* if self._captured_graph is None: raise RuntimeError (lines 2163-2168).
     * replay before capture is a usage error -> ROCKE_ERR_VALUE. */
    if(ctx->captured_graph == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* self._captured_graph.replay() (line 2169). Requires a HIP runtime.
     * TODO(port): ((CUDAGraph*)ctx->captured_graph)->replay(). */
    return ROCKE_ERR_NOTIMPL;
}
