// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_launchers.c.c -- C99 port of the
 * LAZY LAUNCHER ENSURE + CACHE surface of
 * rocke/instances/common/fused_moe_e2e.py (FusedMoeForward._ensure_* /
 * _grouped_* / _moe_* methods, Python lines 836-1353 + 1584-1591).
 *
 * SCOPE (this TU):
 *   FusedMoeForward._ensure_topk_launcher            -> rocke_fmoe_ensure_topk_launcher
 *   FusedMoeForward._ensure_gemm_launcher            -> rocke_fmoe_ensure_gemm_launcher
 *   FusedMoeForward._ensure_batched_gemm_launcher    -> rocke_fmoe_ensure_batched_gemm_launcher
 *   _ensure_batched_gemm_preshuffle_b_launcher       ->
 * rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher _ensure_silu_mul_packed_launcher ->
 * rocke_fmoe_ensure_silu_mul_packed_launcher _ensure_gate_up_silu_launcher                    ->
 * rocke_fmoe_ensure_gate_up_silu_launcher _ensure_interleaved_gate_up_silu_launcher        ->
 * rocke_fmoe_ensure_interleaved_gate_up_silu_launcher
 *   _ensure_interleaved_gate_up_silu_preshuffle_launcher
 *                                                    ->
 * rocke_fmoe_ensure_interleaved_gate_up_silu_preshuffle_launcher _ensure_down_reduce_launcher ->
 * rocke_fmoe_ensure_down_reduce_launcher _ensure_static_scatter_gather_launcher           ->
 * rocke_fmoe_ensure_static_scatter_gather_launcher _moe_batched_gemm_launcher ->
 * rocke_fmoe_moe_batched_gemm_launcher _moe_interleaved_gate_up_silu_launcher            ->
 * rocke_fmoe_moe_interleaved_gate_up_silu_launcher _grouped_gate_up_silu_launcher ->
 * rocke_fmoe_grouped_gate_up_silu_launcher _grouped_down_reduce_launcher                     ->
 * rocke_fmoe_grouped_down_reduce_launcher _grouped_gate_up_spec                             ->
 * rocke_fmoe_grouped_gate_up_spec _grouped_down_spec                                ->
 * rocke_fmoe_grouped_down_spec _ensure_compiled                                  ->
 * rocke_fmoe_ensure_compiled
 *
 * BYTE-IDENTICAL BUILDER-CALL SEQUENCE. Each ensure/cache method below
 * reproduces, in Python order, the spec construction + the call to the
 * already-ported sub-kernel builder (build_topk_softmax / build_batched_gemm /
 * build_grouped_gemm / build_moe_*). The Python `compile_kernel(...)` step lowers
 * that emitted IR to an HSACO and wraps it (+ its signature + cache_key) in a
 * KernelLauncher. In this codegen-only library there is no HIP runtime and the
 * opaque rocke_kernel_launcher_t carries no allocator, so the HSACO compile +
 * KernelLauncher construction are a TODO(port) runtime concern: this TU drives
 * the IR-emitting builder exactly as Python does (so the emitted IR / golden
 * digest is unchanged) and records the resulting "compiled" state in the ctx's
 * launcher slot via a stable non-NULL sentinel so the lazy-once + cache
 * semantics are faithful.
 *
 * Peers (ctx init/destroy, forward phases, host helpers) live in sibling TUs and
 * are reached only through rocke/instance_fused_moe_e2e_internal.h.
 */

#include <stdio.h>
#include <string.h>

#include "rocke/instance_fused_moe_e2e_internal.h"

#include "rocke/instance_batched_gemm.h" /* rocke_build_batched_gemm           */
#include "rocke/instance_fused_moe.h" /* rocke_build_moe_silu_mul_packed... */
#include "rocke/instance_grouped_gemm.h" /* rocke_build_grouped_gemm           */
#include "rocke/instance_moe_gemm_fused.h" /* rocke_build_moe_gate_up_silu_gemm.. */
#include "rocke/instance_topk_softmax.h" /* rocke_build_topk_softmax            */

/* ------------------------------------------------------------------ *
 * launcher "compiled" sentinel
 * ------------------------------------------------------------------ *
 *
 * rocke_kernel_launcher_t is an opaque runtime peer with no allocator in this
 * codegen-only library (see the internal header). Until a HIP backend wires the
 * compile_kernel -> KernelLauncher construction, an ensure that successfully
 * drove its builder records "this slot is compiled" by storing a stable non-NULL
 * address. Reads only ever compare the slot to NULL (the Python `is None`
 * lazy-once gate), so any unique non-NULL pointer is faithful.
 *
 * TODO(port): replace every &g_*_compiled sentinel store with the real
 * KernelLauncher built from the artifact returned by compile_kernel(...). */
static char g_topk_compiled;
static char g_gemm_compiled;
static char g_batched_compiled;
static char g_batched_preb_compiled;
static char g_silu_mul_packed_compiled;
static char g_gate_up_silu_compiled;
static char g_interleaved_compiled;
static char g_interleaved_preb_compiled;
static char g_down_reduce_compiled;
static char g_static_sg_compiled;
static char g_cache_compiled;

#define ROCKE_FMOE_SENTINEL(p) ((rocke_kernel_launcher_t*)(void*)&(p))
#define ROCKE_FMOE_GROUPED_SENTINEL(p) ((rocke_grouped_gemm_launcher_t*)(void*)&(p))

/* Drive a sub-kernel builder over a throwaway IRBuilder exactly as Python's
 * compile_kernel(build_*(spec), ...) does, then discard the builder. Returns
 * true iff the build emitted a kernel without a sticky error. The IR itself is
 * the golden-verified artifact; the HSACO compile + launcher wrap is TODO(port).
 *
 * `kernel_name` seeds rocke_ir_builder_init exactly as the Python builders expect
 * (each *_new() convenience inits `b` with spec.kernel_name()). The builder is
 * the bare rocke_build_* entry: we init `b` ourselves so a future port can also
 * capture b->kernel for the launcher. */
typedef rocke_kernel_def_t* (*rocke_fmoe_build_thunk_fn)(rocke_ir_builder_t* b, void* user);

static bool
    rocke_fmoe_drive_build(const char* kernel_name, rocke_fmoe_build_thunk_fn thunk, void* user)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, kernel_name) != ROCKE_OK)
    {
        return false;
    }
    rocke_kernel_def_t* kernel = thunk(&b, user);
    bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
    /* TODO(port): instead of freeing, hand b->kernel to compile_kernel(...) and
     * build the KernelLauncher (hsaco + kernel_name + signature + cache_key). */
    rocke_ir_builder_free(&b);
    return ok;
}

/* ================================================================== *
 * single-slot lazy launcher ensures (FusedMoeForward._ensure_*)
 * ================================================================== */

/* ----------------------------- thunks ----------------------------- */

static rocke_kernel_def_t* rocke_fmoe_thunk_topk(rocke_ir_builder_t* b, void* user)
{
    const rocke_topk_softmax_spec_t* spec = (const rocke_topk_softmax_spec_t*)user;
    return rocke_build_topk_softmax(b, spec, NULL);
}

rocke_kernel_launcher_t* rocke_fmoe_ensure_topk_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._topk_launcher is None: */
    if(ctx->topk_launcher == NULL)
    {
        /* spec = self.spec.to_topk_softmax_spec() */
        rocke_topk_softmax_spec_t spec = rocke_fmoe_forward_spec_to_topk_softmax_spec(&ctx->spec);
        /* artifact = compile_kernel(build_topk_softmax(spec), arch=self.arch) */
        if(!rocke_fmoe_drive_build("topk_softmax", rocke_fmoe_thunk_topk, &spec))
        {
            return NULL;
        }
        /* self._topk_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->topk_launcher = ROCKE_FMOE_SENTINEL(g_topk_compiled);
    }
    return ctx->topk_launcher;
}

static rocke_kernel_def_t* rocke_fmoe_thunk_grouped_gemm(rocke_ir_builder_t* b, void* user)
{
    const rocke_grouped_gemm_spec_t* spec = (const rocke_grouped_gemm_spec_t*)user;
    return rocke_build_grouped_gemm(b, spec, NULL);
}

rocke_grouped_gemm_launcher_t* rocke_fmoe_ensure_gemm_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._gemm_launcher is None: */
    if(ctx->gemm_launcher == NULL)
    {
        /* gemm_spec = self.spec.to_gemm_spec(name_suffix="gemm") */
        char name_buf[256];
        rocke_grouped_gemm_spec_t gemm_spec;
        if(rocke_fmoe_forward_spec_to_gemm_spec(
               &ctx->spec, "gemm", name_buf, sizeof(name_buf), &gemm_spec)
           != ROCKE_OK)
        {
            return NULL;
        }
        /* artifact = compile_kernel(build_grouped_gemm(gemm_spec), arch=self.arch) */
        if(!rocke_fmoe_drive_build("gemm", rocke_fmoe_thunk_grouped_gemm, &gemm_spec))
        {
            return NULL;
        }
        /* self._gemm_launcher = GroupedGemmLauncher(...) -- TODO(port) */
        ctx->gemm_launcher = ROCKE_FMOE_GROUPED_SENTINEL(g_gemm_compiled);
    }
    return ctx->gemm_launcher;
}

static rocke_kernel_def_t* rocke_fmoe_thunk_batched_gemm(rocke_ir_builder_t* b, void* user)
{
    const rocke_batched_gemm_spec_t* spec = (const rocke_batched_gemm_spec_t*)user;
    return rocke_build_batched_gemm(b, spec, NULL);
}

rocke_kernel_launcher_t* rocke_fmoe_ensure_batched_gemm_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._batched_gemm_launcher is None: */
    if(ctx->batched_gemm_launcher == NULL)
    {
        /* spec = self.spec.to_batched_gemm_spec() */
        char name_buf[256];
        rocke_batched_gemm_spec_t spec;
        if(rocke_fmoe_forward_spec_to_batched_gemm_spec(
               &ctx->spec, name_buf, sizeof(name_buf), &spec)
           != ROCKE_OK)
        {
            return NULL;
        }
        /* artifact = compile_kernel(build_batched_gemm(spec), arch=self.arch) */
        if(!rocke_fmoe_drive_build("batched_gemm", rocke_fmoe_thunk_batched_gemm, &spec))
        {
            return NULL;
        }
        /* self._batched_gemm_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->batched_gemm_launcher = ROCKE_FMOE_SENTINEL(g_batched_compiled);
    }
    return ctx->batched_gemm_launcher;
}

rocke_kernel_launcher_t*
    rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._batched_gemm_preshuffle_b_launcher is None: */
    if(ctx->batched_gemm_preshuffle_b_launcher == NULL)
    {
        /* spec = self.spec.to_batched_gemm_spec_preshuffle_b() */
        char name_buf[256];
        rocke_batched_gemm_spec_t spec;
        if(rocke_fmoe_forward_spec_to_batched_gemm_spec_preshuffle_b(
               &ctx->spec, name_buf, sizeof(name_buf), &spec)
           != ROCKE_OK)
        {
            return NULL;
        }
        /* artifact = compile_kernel(build_batched_gemm(spec), arch=self.arch) */
        if(!rocke_fmoe_drive_build("batched_gemm", rocke_fmoe_thunk_batched_gemm, &spec))
        {
            return NULL;
        }
        /* self._batched_gemm_preshuffle_b_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->batched_gemm_preshuffle_b_launcher = ROCKE_FMOE_SENTINEL(g_batched_preb_compiled);
    }
    return ctx->batched_gemm_preshuffle_b_launcher;
}

static rocke_kernel_def_t* rocke_fmoe_thunk_silu_mul_packed(rocke_ir_builder_t* b, void* user)
{
    const rocke_fused_moe_spec_t* spec = (const rocke_fused_moe_spec_t*)user;
    return rocke_build_moe_silu_mul_packed(b, spec, NULL);
}

/* to_fused_moe_spec(): FusedMoeSpec(tokens, experts, topk, hidden, intermediate,
 * dtype, block_size=streaming_block_size, vec=streaming_vec). Python lines
 * 590-600. The name/bf16_accumulator fields keep the FusedMoeSpec defaults. */
static rocke_fused_moe_spec_t rocke_fmoe_to_fused_moe_spec(const rocke_fmoe_build_ctx_t* ctx)
{
    rocke_fused_moe_spec_t fmoe = rocke_fused_moe_spec_default();
    fmoe.tokens = ctx->spec.tokens;
    fmoe.experts = ctx->spec.experts;
    fmoe.topk = ctx->spec.topk;
    fmoe.hidden = ctx->spec.hidden;
    fmoe.intermediate = ctx->spec.intermediate;
    fmoe.dtype = ctx->spec.dtype;
    fmoe.block_size = ctx->spec.streaming_block_size;
    fmoe.vec = ctx->spec.streaming_vec;
    return fmoe;
}

rocke_kernel_launcher_t* rocke_fmoe_ensure_silu_mul_packed_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._silu_mul_packed_launcher is None: */
    if(ctx->silu_mul_packed_launcher == NULL)
    {
        /* fmoe_spec = self.spec.to_fused_moe_spec() */
        rocke_fused_moe_spec_t fmoe_spec = rocke_fmoe_to_fused_moe_spec(ctx);
        /* artifact = compile_kernel(build_moe_silu_mul_packed(fmoe_spec), arch=self.arch) */
        if(!rocke_fmoe_drive_build("silu_mul_packed", rocke_fmoe_thunk_silu_mul_packed, &fmoe_spec))
        {
            return NULL;
        }
        /* self._silu_mul_packed_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->silu_mul_packed_launcher = ROCKE_FMOE_SENTINEL(g_silu_mul_packed_compiled);
    }
    return ctx->silu_mul_packed_launcher;
}

/* --------- gate_up / interleaved / down spec builders (shared) --------- *
 *
 * The non-grouped _ensure_* gate/down specs are constructed inline here; the
 * grouped variants live in the dedicated rocke_fmoe_grouped_*_spec helpers below.
 * Every one uses TraitSpec(pad_m=True, pad_n=True, epilogue="default") with
 * dtype=_gemm_dtype_to_universal(self.spec.dtype) and name f"{name}_<suffix>". */

/* TraitSpec(pad_m=True, pad_n=True, epilogue="default", preshuffle_b=?,
 * active_tile_skip=?): start from the matching spec default's trait (carries the
 * Python TraitSpec field defaults) and override the named keyword args. */
static rocke_gemm_trait_spec_t rocke_fmoe_gemm_trait(bool preshuffle_b, bool active_tile_skip)
{
    /* The gate_up_silu spec default already carries epilogue="default". */
    rocke_moe_gate_up_silu_gemm_spec_t tmpl = rocke_moe_gate_up_silu_gemm_spec_default();
    rocke_gemm_trait_spec_t trait = tmpl.trait;
    trait.epilogue = "default";
    trait.pad_m = true;
    trait.pad_n = true;
    trait.preshuffle_b = preshuffle_b;
    trait.active_tile_skip = active_tile_skip;
    return trait;
}

rocke_kernel_launcher_t* rocke_fmoe_ensure_gate_up_silu_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._gate_up_silu_launcher is None: */
    if(ctx->gate_up_silu_launcher == NULL)
    {
        const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
        if(universal_dtype == NULL)
        {
            return NULL;
        }
        /* spec = FusedGateUpSiluGemmSpec(name=f"{name}_gate_up_silu",
         *   tile=gemm_tile, trait=TraitSpec(pad_m, pad_n, epilogue="default"),
         *   dtype=_gemm_dtype_to_universal(dtype)) */
        char name_buf[256];
        (void)snprintf(name_buf,
                       sizeof(name_buf),
                       "%s_gate_up_silu",
                       ctx->spec.name != NULL ? ctx->spec.name : "");
        rocke_moe_gate_up_silu_gemm_spec_t spec = rocke_moe_gate_up_silu_gemm_spec_default();
        spec.name = name_buf;
        spec.tile = ctx->spec.gemm_tile;
        spec.trait = rocke_fmoe_gemm_trait(/*preshuffle_b=*/false,
                                           /*active_tile_skip=*/false);
        spec.dtype = universal_dtype;
        rocke_moe_gate_up_silu_gemm_spec_finalize(&spec);
        /* artifact = compile_kernel(build_moe_gate_up_silu_gemm(spec, arch=self.arch),
         *   arch=self.arch) */
        rocke_ir_builder_t b;
        if(rocke_ir_builder_init(&b, "gate_up_silu") != ROCKE_OK)
        {
            return NULL;
        }
        rocke_kernel_def_t* kernel = rocke_build_moe_gate_up_silu_gemm(&b, &spec, ctx->arch);
        bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
        rocke_ir_builder_free(&b);
        if(!ok)
        {
            return NULL;
        }
        /* self._gate_up_silu_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->gate_up_silu_launcher = ROCKE_FMOE_SENTINEL(g_gate_up_silu_compiled);
    }
    return ctx->gate_up_silu_launcher;
}

/* interleaved gate/up spec template (FusedInterleavedGateUpSiluGemmSpec). */
static rocke_moe_interleaved_gate_up_silu_gemm_spec_t
    rocke_fmoe_make_interleaved_spec(const rocke_fmoe_build_ctx_t* ctx,
                                     char* name_buf,
                                     size_t name_cap,
                                     bool preshuffle_b,
                                     const char* universal_dtype)
{
    (void)snprintf(name_buf,
                   name_cap,
                   "%s_interleaved_gate_up_silu",
                   ctx->spec.name != NULL ? ctx->spec.name : "");
    rocke_moe_interleaved_gate_up_silu_gemm_spec_t spec
        = rocke_moe_interleaved_gate_up_silu_gemm_spec_default();
    spec.name = name_buf;
    spec.tile = ctx->spec.gemm_tile;
    spec.trait = rocke_fmoe_gemm_trait(preshuffle_b, /*active_tile_skip=*/false);
    spec.dtype = universal_dtype;
    rocke_moe_interleaved_gate_up_silu_gemm_spec_finalize(&spec);
    return spec;
}

rocke_kernel_launcher_t*
    rocke_fmoe_ensure_interleaved_gate_up_silu_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._interleaved_gate_up_silu_launcher is None: */
    if(ctx->interleaved_gate_up_silu_launcher == NULL)
    {
        const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
        if(universal_dtype == NULL)
        {
            return NULL;
        }
        /* spec = FusedInterleavedGateUpSiluGemmSpec(name=..., tile=gemm_tile,
         *   trait=TraitSpec(pad_m, pad_n, epilogue="default"),
         *   dtype=_gemm_dtype_to_universal(dtype)) */
        char name_buf[256];
        rocke_moe_interleaved_gate_up_silu_gemm_spec_t spec
            = rocke_fmoe_make_interleaved_spec(ctx,
                                               name_buf,
                                               sizeof(name_buf),
                                               /*preshuffle_b=*/false,
                                               universal_dtype);
        /* artifact = compile_kernel(
         *   build_moe_interleaved_gate_up_silu_gemm(spec, arch=self.arch),
         *   arch=self.arch) */
        rocke_ir_builder_t b;
        if(rocke_ir_builder_init(&b, "interleaved_gate_up_silu") != ROCKE_OK)
        {
            return NULL;
        }
        rocke_kernel_def_t* kernel
            = rocke_build_moe_interleaved_gate_up_silu_gemm(&b, &spec, ctx->arch);
        bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
        rocke_ir_builder_free(&b);
        if(!ok)
        {
            return NULL;
        }
        /* self._interleaved_gate_up_silu_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->interleaved_gate_up_silu_launcher = ROCKE_FMOE_SENTINEL(g_interleaved_compiled);
    }
    return ctx->interleaved_gate_up_silu_launcher;
}

rocke_kernel_launcher_t*
    rocke_fmoe_ensure_interleaved_gate_up_silu_preshuffle_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._interleaved_gate_up_silu_preshuffle_launcher is None: */
    if(ctx->interleaved_gate_up_silu_preshuffle_launcher == NULL)
    {
        const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
        if(universal_dtype == NULL)
        {
            return NULL;
        }
        /* Same kernel body but TraitSpec(..., preshuffle_b=True). */
        char name_buf[256];
        rocke_moe_interleaved_gate_up_silu_gemm_spec_t spec
            = rocke_fmoe_make_interleaved_spec(ctx,
                                               name_buf,
                                               sizeof(name_buf),
                                               /*preshuffle_b=*/true,
                                               universal_dtype);
        rocke_ir_builder_t b;
        if(rocke_ir_builder_init(&b, "interleaved_gate_up_silu") != ROCKE_OK)
        {
            return NULL;
        }
        rocke_kernel_def_t* kernel
            = rocke_build_moe_interleaved_gate_up_silu_gemm(&b, &spec, ctx->arch);
        bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
        rocke_ir_builder_free(&b);
        if(!ok)
        {
            return NULL;
        }
        /* self._interleaved_gate_up_silu_preshuffle_launcher = KernelLauncher(...)
         *   -- TODO(port) */
        ctx->interleaved_gate_up_silu_preshuffle_launcher
            = ROCKE_FMOE_SENTINEL(g_interleaved_preb_compiled);
    }
    return ctx->interleaved_gate_up_silu_preshuffle_launcher;
}

/* down-reduce spec template (FusedDownReduceGemmSpec). */
static rocke_moe_down_reduce_gemm_spec_t
    rocke_fmoe_make_down_reduce_spec(const rocke_fmoe_build_ctx_t* ctx,
                                     char* name_buf,
                                     size_t name_cap,
                                     bool grouped,
                                     const char* universal_dtype)
{
    (void)snprintf(
        name_buf, name_cap, "%s_down_reduce", ctx->spec.name != NULL ? ctx->spec.name : "");
    rocke_moe_down_reduce_gemm_spec_t spec = rocke_moe_down_reduce_gemm_spec_default();
    spec.name = name_buf;
    spec.tile = ctx->spec.gemm_tile;
    spec.trait = rocke_fmoe_gemm_trait(/*preshuffle_b=*/false,
                                       /*active_tile_skip=*/false);
    spec.dtype = universal_dtype;
    spec.grouped = grouped;
    rocke_moe_down_reduce_gemm_spec_finalize(&spec);
    return spec;
}

rocke_kernel_launcher_t* rocke_fmoe_ensure_down_reduce_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._down_reduce_launcher is None: */
    if(ctx->down_reduce_launcher == NULL)
    {
        const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
        if(universal_dtype == NULL)
        {
            return NULL;
        }
        /* spec = FusedDownReduceGemmSpec(name=f"{name}_down_reduce",
         *   tile=gemm_tile, trait=TraitSpec(pad_m, pad_n, epilogue="default"),
         *   dtype=_gemm_dtype_to_universal(dtype)) */
        char name_buf[256];
        rocke_moe_down_reduce_gemm_spec_t spec = rocke_fmoe_make_down_reduce_spec(ctx,
                                                                                  name_buf,
                                                                                  sizeof(name_buf),
                                                                                  /*grouped=*/false,
                                                                                  universal_dtype);
        /* artifact = compile_kernel(build_moe_down_reduce_gemm(spec, arch=self.arch),
         *   arch=self.arch) */
        rocke_ir_builder_t b;
        if(rocke_ir_builder_init(&b, "down_reduce") != ROCKE_OK)
        {
            return NULL;
        }
        rocke_kernel_def_t* kernel = rocke_build_moe_down_reduce_gemm(&b, &spec, ctx->arch);
        bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
        rocke_ir_builder_free(&b);
        if(!ok)
        {
            return NULL;
        }
        /* self._down_reduce_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->down_reduce_launcher = ROCKE_FMOE_SENTINEL(g_down_reduce_compiled);
    }
    return ctx->down_reduce_launcher;
}

static rocke_kernel_def_t* rocke_fmoe_thunk_static_sg(rocke_ir_builder_t* b, void* user)
{
    const rocke_fused_moe_spec_t* spec = (const rocke_fused_moe_spec_t*)user;
    return rocke_build_moe_static_scatter_gather(b, spec, NULL);
}

rocke_kernel_launcher_t*
    rocke_fmoe_ensure_static_scatter_gather_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* if self._static_scatter_gather_launcher is None: */
    if(ctx->static_scatter_gather_launcher == NULL)
    {
        /* spec = self.spec.to_fused_moe_spec() */
        rocke_fused_moe_spec_t spec = rocke_fmoe_to_fused_moe_spec(ctx);
        /* artifact = compile_kernel(build_moe_static_scatter_gather(spec),
         *   arch=self.arch) */
        if(!rocke_fmoe_drive_build("static_scatter_gather", rocke_fmoe_thunk_static_sg, &spec))
        {
            return NULL;
        }
        /* self._static_scatter_gather_launcher = KernelLauncher(...) -- TODO(port) */
        ctx->static_scatter_gather_launcher = ROCKE_FMOE_SENTINEL(g_static_sg_compiled);
    }
    return ctx->static_scatter_gather_launcher;
}

/* ================================================================== *
 * _grouped_gate_up_spec / _grouped_down_spec  (Python 912-919, 1584-1591)
 * ================================================================== */

rocke_moe_gate_up_silu_gemm_spec_t
    rocke_fmoe_grouped_gate_up_spec(const rocke_fmoe_build_ctx_t* ctx)
{
    /* FusedGateUpSiluGemmSpec(name=f"{name}_gate_up_silu", tile=gemm_tile,
     *   trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
     *   dtype=_gemm_dtype_to_universal(dtype), grouped=True) */
    rocke_moe_gate_up_silu_gemm_spec_t spec = rocke_moe_gate_up_silu_gemm_spec_default();
    if(ctx == NULL)
    {
        return spec;
    }
    /* NOTE: the spec.name points at ctx->spec.name with the "_gate_up_silu"
     * suffix. The Python builds an f-string each call; the C caller that needs
     * the composed name builds it with the spec's kernel_name() / its own buffer.
     * Here we carry the base name through; the trait + tile + dtype + grouped
     * flag (the build-affecting fields) are byte-identical. */
    spec.name = ctx->spec.name;
    spec.tile = ctx->spec.gemm_tile;
    spec.trait = rocke_fmoe_gemm_trait(/*preshuffle_b=*/false,
                                       /*active_tile_skip=*/false);
    spec.dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
    spec.grouped = true;
    rocke_moe_gate_up_silu_gemm_spec_finalize(&spec);
    return spec;
}

rocke_moe_down_reduce_gemm_spec_t rocke_fmoe_grouped_down_spec(const rocke_fmoe_build_ctx_t* ctx)
{
    /* FusedDownReduceGemmSpec(name=f"{name}_down_reduce", tile=gemm_tile,
     *   trait=TraitSpec(pad_m=True, pad_n=True, epilogue="default"),
     *   dtype=_gemm_dtype_to_universal(dtype), grouped=True) */
    rocke_moe_down_reduce_gemm_spec_t spec = rocke_moe_down_reduce_gemm_spec_default();
    if(ctx == NULL)
    {
        return spec;
    }
    spec.name = ctx->spec.name;
    spec.tile = ctx->spec.gemm_tile;
    spec.trait = rocke_fmoe_gemm_trait(/*preshuffle_b=*/false,
                                       /*active_tile_skip=*/false);
    spec.dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
    spec.grouped = true;
    rocke_moe_down_reduce_gemm_spec_finalize(&spec);
    return spec;
}

/* ================================================================== *
 * parameterized cache lookups (self._moe_gemm_launcher_cache dict)
 * ================================================================== *
 *
 * The Python dict is a small fixed table on the ctx. The cache key is the
 * (kind, preshuffle_b, active_tile_skip) tuple; the grouped kinds ignore the two
 * trait bools (Python keys them on a 1-tuple, so the bools are don't-care). */

static rocke_kernel_launcher_t* rocke_fmoe_cache_get(rocke_fmoe_build_ctx_t* ctx,
                                                     rocke_fmoe_gemm_kind_t kind,
                                                     bool preshuffle_b,
                                                     bool active_tile_skip)
{
    const bool keyed_on_traits
        = (kind == ROCKE_FMOE_GEMM_BATCHED || kind == ROCKE_FMOE_GEMM_INTERLEAVED_GATE_UP);
    for(int i = 0; i < ctx->moe_gemm_launcher_cache_len; ++i)
    {
        const rocke_fmoe_gemm_cache_entry_t* e = &ctx->moe_gemm_launcher_cache[i];
        if(!e->used || e->kind != kind)
        {
            continue;
        }
        if(keyed_on_traits
           && (e->preshuffle_b != preshuffle_b || e->active_tile_skip != active_tile_skip))
        {
            continue;
        }
        return e->launcher;
    }
    return NULL;
}

static rocke_kernel_launcher_t* rocke_fmoe_cache_put(rocke_fmoe_build_ctx_t* ctx,
                                                     rocke_fmoe_gemm_kind_t kind,
                                                     bool preshuffle_b,
                                                     bool active_tile_skip,
                                                     rocke_kernel_launcher_t* launcher)
{
    if(ctx->moe_gemm_launcher_cache_len >= ROCKE_FMOE_GEMM_CACHE_CAP)
    {
        /* Cache full: return the launcher uncached (Python's dict is unbounded;
         * the fixed table headroom (16) exceeds the live key set, so this is a
         * defensive no-op path). */
        return launcher;
    }
    rocke_fmoe_gemm_cache_entry_t* e
        = &ctx->moe_gemm_launcher_cache[ctx->moe_gemm_launcher_cache_len++];
    e->used = true;
    e->kind = kind;
    e->preshuffle_b = preshuffle_b;
    e->active_tile_skip = active_tile_skip;
    e->launcher = launcher;
    return launcher;
}

rocke_kernel_launcher_t* rocke_fmoe_moe_batched_gemm_launcher(rocke_fmoe_build_ctx_t* ctx,
                                                              bool preshuffle_b,
                                                              bool active_tile_skip)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* key = ("batched", bool(preshuffle_b), bool(active_tile_skip))
     * cached = self._moe_gemm_launcher_cache.get(key); if cached: return cached */
    rocke_kernel_launcher_t* cached
        = rocke_fmoe_cache_get(ctx, ROCKE_FMOE_GEMM_BATCHED, preshuffle_b, active_tile_skip);
    if(cached != NULL)
    {
        return cached;
    }
    /* trait = TraitSpec(pad_m=True, pad_n=True, preshuffle_b=?, active_tile_skip=?)
     * spec = BatchedGemmSpec(name=f"{name}_batched_gemm", tile=gemm_tile,
     *   trait=trait, dtype=_gemm_dtype_to_universal(dtype)) */
    const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
    if(universal_dtype == NULL)
    {
        return NULL;
    }
    char name_buf[256];
    (void)snprintf(name_buf,
                   sizeof(name_buf),
                   "%s_batched_gemm",
                   ctx->spec.name != NULL ? ctx->spec.name : "");
    rocke_batched_gemm_spec_t spec = rocke_batched_gemm_spec_default();
    spec.name = name_buf;
    spec.tile = ctx->spec.gemm_tile;
    {
        rocke_gemm_trait_spec_t trait = spec.trait; /* carry BatchedGemm defaults */
        trait.pad_m = true;
        trait.pad_n = true;
        trait.preshuffle_b = preshuffle_b;
        trait.active_tile_skip = active_tile_skip;
        spec.trait = trait;
    }
    spec.dtype = universal_dtype;
    /* artifact = compile_kernel(build_batched_gemm(spec), arch=self.arch) */
    if(!rocke_fmoe_drive_build("batched_gemm", rocke_fmoe_thunk_batched_gemm, &spec))
    {
        return NULL;
    }
    /* launcher = KernelLauncher(...); self._moe_gemm_launcher_cache[key] = launcher
     *   -- TODO(port) for the launcher value; cache entry is faithful. */
    rocke_kernel_launcher_t* launcher = ROCKE_FMOE_SENTINEL(g_cache_compiled);
    return rocke_fmoe_cache_put(
        ctx, ROCKE_FMOE_GEMM_BATCHED, preshuffle_b, active_tile_skip, launcher);
}

rocke_kernel_launcher_t* rocke_fmoe_moe_interleaved_gate_up_silu_launcher(
    rocke_fmoe_build_ctx_t* ctx, bool preshuffle_b, bool active_tile_skip)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* key = ("interleaved_gate_up_silu", bool(preshuffle_b), bool(active_tile_skip))
     * cached = self._moe_gemm_launcher_cache.get(key); if cached: return cached */
    rocke_kernel_launcher_t* cached = rocke_fmoe_cache_get(
        ctx, ROCKE_FMOE_GEMM_INTERLEAVED_GATE_UP, preshuffle_b, active_tile_skip);
    if(cached != NULL)
    {
        return cached;
    }
    const char* universal_dtype = rocke_fmoe_gemm_dtype_to_universal(ctx->spec.dtype);
    if(universal_dtype == NULL)
    {
        return NULL;
    }
    /* trait = TraitSpec(pad_m, pad_n, epilogue="default", preshuffle_b=?,
     *   active_tile_skip=?)
     * spec = FusedInterleavedGateUpSiluGemmSpec(name=..., tile=gemm_tile,
     *   trait=trait, dtype=_gemm_dtype_to_universal(dtype)) */
    char name_buf[256];
    (void)snprintf(name_buf,
                   sizeof(name_buf),
                   "%s_interleaved_gate_up_silu",
                   ctx->spec.name != NULL ? ctx->spec.name : "");
    rocke_moe_interleaved_gate_up_silu_gemm_spec_t spec
        = rocke_moe_interleaved_gate_up_silu_gemm_spec_default();
    spec.name = name_buf;
    spec.tile = ctx->spec.gemm_tile;
    spec.trait = rocke_fmoe_gemm_trait(preshuffle_b, active_tile_skip);
    spec.dtype = universal_dtype;
    rocke_moe_interleaved_gate_up_silu_gemm_spec_finalize(&spec);
    /* artifact = compile_kernel(
     *   build_moe_interleaved_gate_up_silu_gemm(spec, arch=self.arch),
     *   arch=self.arch) */
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "interleaved_gate_up_silu") != ROCKE_OK)
    {
        return NULL;
    }
    rocke_kernel_def_t* kernel
        = rocke_build_moe_interleaved_gate_up_silu_gemm(&b, &spec, ctx->arch);
    bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
    rocke_ir_builder_free(&b);
    if(!ok)
    {
        return NULL;
    }
    /* launcher = KernelLauncher(...); self._moe_gemm_launcher_cache[key] = launcher */
    rocke_kernel_launcher_t* launcher = ROCKE_FMOE_SENTINEL(g_cache_compiled);
    return rocke_fmoe_cache_put(
        ctx, ROCKE_FMOE_GEMM_INTERLEAVED_GATE_UP, preshuffle_b, active_tile_skip, launcher);
}

rocke_kernel_launcher_t* rocke_fmoe_grouped_gate_up_silu_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* key = ("grouped_gate_up_silu",); cached = ...get(key); if cached: return */
    rocke_kernel_launcher_t* cached
        = rocke_fmoe_cache_get(ctx, ROCKE_FMOE_GEMM_GROUPED_GATE_UP, false, false);
    if(cached != NULL)
    {
        return cached;
    }
    /* spec = self._grouped_gate_up_spec() */
    rocke_moe_gate_up_silu_gemm_spec_t spec = rocke_fmoe_grouped_gate_up_spec(ctx);
    /* artifact = compile_kernel(build_moe_gate_up_silu_gemm(spec, arch=self.arch),
     *   arch=self.arch) */
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "gate_up_silu") != ROCKE_OK)
    {
        return NULL;
    }
    rocke_kernel_def_t* kernel = rocke_build_moe_gate_up_silu_gemm(&b, &spec, ctx->arch);
    bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
    rocke_ir_builder_free(&b);
    if(!ok)
    {
        return NULL;
    }
    /* launcher = KernelLauncher(...); self._moe_gemm_launcher_cache[key] = launcher */
    rocke_kernel_launcher_t* launcher = ROCKE_FMOE_SENTINEL(g_cache_compiled);
    return rocke_fmoe_cache_put(ctx, ROCKE_FMOE_GEMM_GROUPED_GATE_UP, false, false, launcher);
}

rocke_kernel_launcher_t* rocke_fmoe_grouped_down_reduce_launcher(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return NULL;
    }
    /* key = ("grouped_down_reduce",); cached = ...get(key); if cached: return */
    rocke_kernel_launcher_t* cached
        = rocke_fmoe_cache_get(ctx, ROCKE_FMOE_GEMM_GROUPED_DOWN_REDUCE, false, false);
    if(cached != NULL)
    {
        return cached;
    }
    /* spec = FusedDownReduceGemmSpec(name=f"{name}_down_reduce", tile=gemm_tile,
     *   trait=TraitSpec(pad_m, pad_n, epilogue="default"),
     *   dtype=_gemm_dtype_to_universal(dtype), grouped=True)
     * (the same value _grouped_down_spec() returns). */
    rocke_moe_down_reduce_gemm_spec_t spec = rocke_fmoe_grouped_down_spec(ctx);
    /* artifact = compile_kernel(build_moe_down_reduce_gemm(spec, arch=self.arch),
     *   arch=self.arch) */
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "down_reduce") != ROCKE_OK)
    {
        return NULL;
    }
    rocke_kernel_def_t* kernel = rocke_build_moe_down_reduce_gemm(&b, &spec, ctx->arch);
    bool ok = (kernel != NULL) && rocke_ir_builder_ok(&b);
    rocke_ir_builder_free(&b);
    if(!ok)
    {
        return NULL;
    }
    /* launcher = KernelLauncher(...); self._moe_gemm_launcher_cache[key] = launcher */
    rocke_kernel_launcher_t* launcher = ROCKE_FMOE_SENTINEL(g_cache_compiled);
    return rocke_fmoe_cache_put(ctx, ROCKE_FMOE_GEMM_GROUPED_DOWN_REDUCE, false, false, launcher);
}

/* ================================================================== *
 * _ensure_compiled  (Python lines 1334-1353)
 * ================================================================== *
 *
 * Compile every component up front (idempotent): topk, sort, fused-moe gather/
 * silu/reduce, batched GEMM (+ preshuffle/interleaved-preshuffle when the
 * preshuffle flags request them), packed silu_mul, gate+up silu, interleaved
 * gate+up silu, down-reduce, static scatter-gather. The sort / fused-moe
 * launchers' own _ensure_launchers() are runtime peers (TODO(port)); the IR-
 * emitting ensures below are driven in Python order. */
rocke_status_t rocke_fmoe_ensure_compiled(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* self._ensure_topk_launcher() */
    if(rocke_fmoe_ensure_topk_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* self._sort_launcher._ensure_launchers()
     * self._fused_moe_launcher._ensure_launchers()
     * TODO(port): the MoeSortingLauncher / FusedMoeLauncher own their three
     * sub-kernel compiles; in this codegen-only port those opaque runtime peers
     * have no allocator. The sort/gather/silu/reduce sub-kernel IR builders are
     * driven by the forward phases when wired. */

    /* self._ensure_batched_gemm_launcher() */
    if(rocke_fmoe_ensure_batched_gemm_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* if self.spec.preshuffle_w_down or self.spec.preshuffle_w_gate_up_packed: */
    if(ctx->spec.preshuffle_w_down || ctx->spec.preshuffle_w_gate_up_packed)
    {
        /* self._ensure_batched_gemm_preshuffle_b_launcher() */
        if(rocke_fmoe_ensure_batched_gemm_preshuffle_b_launcher(ctx) == NULL)
        {
            return ROCKE_ERR_VALUE;
        }
    }

    /* if self.spec.preshuffle_w_gate_up_interleaved: */
    if(ctx->spec.preshuffle_w_gate_up_interleaved)
    {
        /* self._ensure_interleaved_gate_up_silu_preshuffle_launcher() */
        if(rocke_fmoe_ensure_interleaved_gate_up_silu_preshuffle_launcher(ctx) == NULL)
        {
            return ROCKE_ERR_VALUE;
        }
    }

    /* self._ensure_silu_mul_packed_launcher() */
    if(rocke_fmoe_ensure_silu_mul_packed_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* self._ensure_gate_up_silu_launcher() */
    if(rocke_fmoe_ensure_gate_up_silu_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* self._ensure_interleaved_gate_up_silu_launcher() */
    if(rocke_fmoe_ensure_interleaved_gate_up_silu_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* self._ensure_down_reduce_launcher() */
    if(rocke_fmoe_ensure_down_reduce_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* self._ensure_static_scatter_gather_launcher() */
    if(rocke_fmoe_ensure_static_scatter_gather_launcher(ctx) == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    return ROCKE_OK;
}
