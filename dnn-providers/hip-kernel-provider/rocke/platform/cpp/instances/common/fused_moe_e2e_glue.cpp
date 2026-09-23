// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_glue.c.c -- PUBLIC entry + GLUE
 * for the C99 chunked port of the end-to-end fused-MoE forward orchestrator
 * (rocke/instances/common/fused_moe_e2e.py).
 *
 * SCOPE (this TU only -- the "bucket that calls phases in pipeline order"):
 *   - rocke_fmoe_forward_init / rocke_fmoe_forward_destroy
 *       (FusedMoeForward.__init__ public surface; delegates the heavy lifting
 *        to the peer rocke_fmoe_build_ctx_init / rocke_fmoe_build_ctx_destroy).
 *   - rocke_build_fused_moe_forward / rocke_build_fused_moe_forward_simple
 *       (allocate the orchestrator + its build ctx, run __init__'s tile policy +
 *        static gate, build the representative KernelDef, bind
 *        forward_fn = rocke_fmoe_forward_dispatch).
 *   - rocke_fused_moe_forward_free
 *       (free the KernelDef + bound orchestrator + internal ctx).
 *   - rocke_fused_moe_forward_lower_to_llvm
 *       (the per-stage .ll convenience switching on rocke_fmoe_stage_t).
 *   - rocke_fmoe_forward_dispatch
 *       (Python forward(): if use_static_offsets -> _forward_static
 *        else _forward_dynamic -- the single function that drives the
 *        router -> sort -> gather -> gate+up -> silu_mul -> down -> topk_reduce
 *        chain in declaration order; both phase bodies are peers).
 *
 * The phase bodies (ctx init, the two forward paths, the launcher-ensures, the
 * grouped-gemm dispatch) live in sibling TUs declared in
 * rocke/instance_fused_moe_e2e_internal.h. This TU owns ONLY the public surface,
 * the orchestrator/ctx lifetime, the forward_fn binding, and the call ordering.
 *
 * Byte-identical builder-call sequence -- the public forward() (Python 1597-1627):
 *   def forward(...):
 *       if self._use_static_offsets:
 *           self._forward_static(...); return
 *       self._forward_dynamic(...)
 * which this TU reproduces in rocke_fmoe_forward_dispatch.
 *
 * RUNTIME NOTE.  fused_moe_e2e.py defines NO new kernel: it is a host runtime
 * driver. The forward launch chain (HIP launches, torch workspaces, D->H copies)
 * has no IR-builder analogue in this codegen-only library, so the two forward
 * paths are TODO(port) stubs returning ROCKE_ERR_NOTIMPL (see the peer TUs). The
 * IR that DOES get emitted -- and is .ll-lowerable for golden-digest checks --
 * is the spec-selected sub-kernel builders, reached via the per-stage lower
 * convenience below.
 */
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.fused_moe_e2e_spec.h"
#include "rocke/instance_batched_gemm.h"
#include "rocke/instance_fused_moe.h" /* gather/silu_mul/topk_reduce spec+lower */
#include "rocke/instance_fused_moe_e2e.h"
#include "rocke/instance_fused_moe_e2e_internal.h"
#include "rocke/instance_moe_sorting.h" /* sort hist/scan/scatter spec + lower    */
#include "rocke/instance_topk_softmax.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  small local helpers
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void rocke_fmoe_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

/* to_sort_spec() (Python lines 616-622): MoeSortingSpec(tokens, topk, experts,
 * block_size=sort_block_size). Other fields keep the dataclass defaults. */
static rocke_moe_sorting_spec_t rocke_fmoe_to_sort_spec(const rocke_fmoe_forward_spec_t* spec)
{
    rocke_moe_sorting_spec_t s = rocke_moe_sorting_spec_default();
    s.tokens = spec->tokens;
    s.topk = spec->topk;
    s.experts = spec->experts;
    s.block_size = spec->sort_block_size;
    return s;
}

/* to_fused_moe_spec() (Python lines 624-634): FusedMoeSpec(tokens, experts, topk,
 * hidden, intermediate, dtype, block_size=streaming_block_size,
 * vec=streaming_vec). Other fields keep the dataclass defaults. */
static rocke_fused_moe_spec_t
    rocke_fmoe_to_fused_moe_spec_local(const rocke_fmoe_forward_spec_t* spec)
{
    rocke_fused_moe_spec_t s = rocke_fused_moe_spec_default();
    s.tokens = spec->tokens;
    s.experts = spec->experts;
    s.topk = spec->topk;
    s.hidden = spec->hidden;
    s.intermediate = spec->intermediate;
    s.dtype = spec->dtype;
    s.block_size = spec->streaming_block_size;
    s.vec = spec->streaming_vec;
    return s;
}

/* ===================================================================== *
 *  forward_fn trampoline  (KernelDef.forward_fn ABI)
 *
 *  The public forward_fn signature takes a `void* self` that is the
 *  rocke_fmoe_forward_t* the KernelDef threads through. The orchestrator's per-call
 *  state lives in the internal rocke_fmoe_build_ctx_t hung off self->ctx. This
 *  trampoline binds the call's device pointers onto the ctx working set and
 *  delegates to the pipeline dispatcher (Python FusedMoeForward.forward).
 * ===================================================================== */
static rocke_status_t rocke_fmoe_forward_fn_impl(void* self_v,
                                                 uint64_t routing_logits,
                                                 uint64_t X,
                                                 uint64_t W_gate,
                                                 uint64_t W_up,
                                                 uint64_t W_down,
                                                 uint64_t Y,
                                                 uint64_t stream)
{
    rocke_fmoe_forward_t* self = (rocke_fmoe_forward_t*)self_v;
    rocke_fmoe_build_ctx_t* ctx;

    if(self == NULL || self->ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    ctx = (rocke_fmoe_build_ctx_t*)self->ctx;

    /* Pipeline order is driven inside rocke_fmoe_forward_dispatch (Python
     * forward()): it binds the kwargs onto the ctx working set and selects
     * _forward_static / _forward_dynamic. */
    return rocke_fmoe_forward_dispatch(ctx, routing_logits, X, W_gate, W_up, W_down, Y, stream);
}

/* ===================================================================== *
 *  rocke_fmoe_forward_dispatch  (Python FusedMoeForward.forward, lines 1597-1627)
 *
 *  The single function that drives the
 *    router -> sort -> gather -> gate+up -> silu_mul -> down -> topk_reduce
 *  chain in declaration order, by selecting the static or dynamic path. Binds
 *  the per-call device pointers onto the shared ctx working set first (the
 *  Python kwargs), then dispatches.
 *
 *  Python:
 *      def forward(self, *, routing_logits, X, W_gate, W_up, W_down, Y, stream=0):
 *          if self._use_static_offsets:
 *              self._forward_static(...); return
 *          self._forward_dynamic(...)
 * ===================================================================== */
rocke_status_t rocke_fmoe_forward_dispatch(rocke_fmoe_build_ctx_t* ctx,
                                           uint64_t routing_logits,
                                           uint64_t X,
                                           uint64_t W_gate,
                                           uint64_t W_up,
                                           uint64_t W_down,
                                           uint64_t Y,
                                           uint64_t stream)
{
    if(ctx == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Bind the forward() kwargs onto the shared per-call working set so the
     * selected phase body reads the exact inputs (mirrors the Python kwargs
     * captured by _forward_static / _forward_dynamic). */
    ctx->routing_logits = routing_logits;
    ctx->X = X;
    ctx->W_gate = W_gate;
    ctx->W_up = W_up;
    ctx->W_down = W_down;
    ctx->Y = Y;
    ctx->stream = stream;

    /* Python (1608-1627): static-offset gate selects the path. */
    if(ctx->use_static_offsets)
    {
        return rocke_fmoe_forward_static(ctx);
    }
    return rocke_fmoe_forward_dynamic(ctx);
}

/* ===================================================================== *
 *  rocke_fmoe_forward_init  (FusedMoeForward.__init__ public surface)
 *
 *  Allocate + run the internal build ctx (which performs the arch resolve, the
 *  shape-aware tile-swap policy, the static-offset gate, and zeroes the launcher
 *  caches), then mirror the eagerly-computed scalars onto the public handle and
 *  bind the dispatch forward_fn.
 * ===================================================================== */
rocke_status_t rocke_fmoe_forward_init(rocke_fmoe_forward_t* self,
                                       const rocke_fmoe_forward_spec_t* spec,
                                       const char* arch)
{
    rocke_fmoe_build_ctx_t* ctx;
    rocke_status_t st;

    if(self == NULL || spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    memset(self, 0, sizeof(*self));

    ctx = (rocke_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(ctx == NULL)
    {
        return ROCKE_ERR_OOM;
    }

    /* The internal ctx_init reproduces FusedMoeForward.__init__ (lines 692-831):
     * _resolve_launch_arch, the tile-swap policy, the static gate + slot size,
     * and zeroing the launcher / cache / weight slots. */
    st = rocke_fmoe_build_ctx_init(ctx, spec, arch);
    if(st != ROCKE_OK)
    {
        rocke_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return st;
    }

    /* Surface the eagerly-computed scalars on the public handle (the spec here
     * is the tile-policy-adjusted spec the ctx now holds). */
    self->spec = ctx->spec;
    self->arch = ctx->arch;
    self->use_static_offsets = ctx->use_static_offsets;
    self->static_slot_size = ctx->static_slot_size;
    self->ctx = ctx;
    self->forward_fn = rocke_fmoe_forward_fn_impl;

    return ROCKE_OK;
}

void rocke_fmoe_forward_destroy(rocke_fmoe_forward_t* self)
{
    rocke_fmoe_build_ctx_t* ctx;
    if(self == NULL)
    {
        return;
    }
    ctx = (rocke_fmoe_build_ctx_t*)self->ctx;
    if(ctx != NULL)
    {
        rocke_fmoe_build_ctx_destroy(ctx);
        free(ctx);
    }
    self->ctx = NULL;
    self->forward_fn = NULL;
}

/* ===================================================================== *
 *  representative KernelDef
 *
 *  The orchestrator emits no single monolithic kernel; the build entry returns a
 *  rocke_kernel_def_t* that ENCAPSULATES the pipeline. For the representative
 *  artifact we build the router stage (topk_softmax), which is the head of the
 *  declaration-order chain and is unconditional on every path. The orchestrator
 *  instance owns the builder backing this KernelDef; rocke_fused_moe_forward_free
 *  releases it.
 * ===================================================================== */

/* Owner record kept beside a built KernelDef so rocke_fused_moe_forward_free can
 * release the orchestrator + the builder that owns the KernelDef arena, and so
 * rocke_build_fused_moe_forward_simple can recover the bound `self` from the
 * returned KernelDef (the "KernelDef-adjacent registry"). */
typedef struct rocke_fmoe_kernel_owner
{
    rocke_kernel_def_t* kernel; /* the returned KernelDef (registry key) */
    rocke_ir_builder_t builder; /* owns the KernelDef arena              */
    rocke_fmoe_forward_t self; /* the bound orchestrator instance       */
    struct rocke_fmoe_kernel_owner* next;
} rocke_fmoe_kernel_owner_t;

/* Tiny intrusive registry mapping KernelDef* -> owner record. Single-threaded
 * codegen use; linear scan. */
static rocke_fmoe_kernel_owner_t* g_fmoe_owners = NULL;

static void rocke_fmoe_registry_add(rocke_fmoe_kernel_owner_t* o)
{
    o->next = g_fmoe_owners;
    g_fmoe_owners = o;
}

static rocke_fmoe_kernel_owner_t* rocke_fmoe_registry_take(rocke_kernel_def_t* kernel)
{
    rocke_fmoe_kernel_owner_t** pp = &g_fmoe_owners;
    while(*pp != NULL)
    {
        if((*pp)->kernel == kernel)
        {
            rocke_fmoe_kernel_owner_t* found = *pp;
            *pp = found->next;
            found->next = NULL;
            return found;
        }
        pp = &(*pp)->next;
    }
    return NULL;
}

/* Build the representative KernelDef into `b` from the (tile-policy-adjusted)
 * spec carried on the orchestrator instance. Returns the KernelDef or NULL. */
static rocke_kernel_def_t* rocke_fmoe_build_representative(rocke_ir_builder_t* b,
                                                           const rocke_fmoe_forward_t* self)
{
    rocke_topk_softmax_spec_t topk_spec;

    /* Router head of the pipeline: spec.to_topk_softmax_spec(). */
    topk_spec = rocke_fmoe_forward_spec_to_topk_softmax_spec(&self->spec);
    return rocke_build_topk_softmax_new(b, &topk_spec, self->arch);
}

rocke_kernel_def_t* rocke_build_fused_moe_forward(const rocke_fmoe_forward_spec_t* spec,
                                                  const char* arch,
                                                  rocke_fmoe_forward_t** out_self,
                                                  rocke_fmoe_forward_fn_t* out_forward_fn)
{
    rocke_fmoe_kernel_owner_t* owner;
    rocke_kernel_def_t* kernel;

    if(out_self != NULL)
    {
        *out_self = NULL;
    }
    if(out_forward_fn != NULL)
    {
        *out_forward_fn = NULL;
    }
    if(spec == NULL)
    {
        return NULL;
    }

    owner = (rocke_fmoe_kernel_owner_t*)calloc(1, sizeof(*owner));
    if(owner == NULL)
    {
        return NULL;
    }

    /* __init__ tile policy + static gate + ctx alloc, and forward_fn bind. */
    if(rocke_fmoe_forward_init(&owner->self, spec, arch) != ROCKE_OK)
    {
        free(owner);
        return NULL;
    }

    /* Build the representative KernelDef (router head). The builder is owned by
     * the owner record so the KernelDef arena outlives this call. */
    kernel = rocke_fmoe_build_representative(&owner->builder, &owner->self);
    if(kernel == NULL)
    {
        rocke_ir_builder_free(&owner->builder);
        rocke_fmoe_forward_destroy(&owner->self);
        free(owner);
        return NULL;
    }

    owner->kernel = kernel;
    rocke_fmoe_registry_add(owner);

    if(out_self != NULL)
    {
        *out_self = &owner->self;
    }
    if(out_forward_fn != NULL)
    {
        *out_forward_fn = owner->self.forward_fn;
    }
    return kernel;
}

rocke_kernel_def_t* rocke_build_fused_moe_forward_simple(const rocke_fmoe_forward_spec_t* spec,
                                                         const char* arch)
{
    /* The bound `self` is recoverable from the returned KernelDef via the
     * registry; callers that need it use rocke_build_fused_moe_forward directly. */
    return rocke_build_fused_moe_forward(spec, arch, NULL, NULL);
}

void rocke_fused_moe_forward_free(rocke_kernel_def_t* kernel)
{
    rocke_fmoe_kernel_owner_t* owner;
    if(kernel == NULL)
    {
        return;
    }
    owner = rocke_fmoe_registry_take(kernel);
    if(owner == NULL)
    {
        /* Not one of ours (or already freed). Nothing safe to do. */
        return;
    }
    rocke_fmoe_forward_destroy(&owner->self);
    rocke_ir_builder_free(&owner->builder);
    free(owner);
}

/* ===================================================================== *
 *  rocke_fused_moe_forward_lower_to_llvm  (per-stage .ll convenience)
 *
 *  Lowers ONE named pipeline stage to AMDGPU LLVM IR text. The spec's tile +
 *  trait policy is applied exactly as the live orchestrator would: we run the
 *  __init__ tile-swap policy via the build ctx, then convert the (adjusted) spec
 *  to the selected sub-kernel spec and delegate to that sub-kernel's own
 *  build->lower convenience.
 *
 *  Every pipeline stage lowers via its own sub-kernel build->lower convenience:
 *  router (topk_softmax), gate-up / down (batched_gemm), sort hist/scan/scatter
 *  (MoeSortingSpec via to_sort_spec), and gather / silu_mul / topk_reduce
 *  (FusedMoeSpec via to_fused_moe_spec).
 * ===================================================================== */
rocke_status_t rocke_fused_moe_forward_lower_to_llvm(const rocke_fmoe_forward_spec_t* spec,
                                                     const char* arch,
                                                     rocke_fmoe_stage_t stage,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap)
{
    rocke_fmoe_build_ctx_t* ctx;
    rocke_status_t st;
    const char* resolved_arch;
    rocke_fmoe_forward_spec_t adj_spec;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_fmoe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    /* Apply __init__'s arch resolve + tile-swap policy via the build ctx so the
     * stage builders see the exact tile the live orchestrator selects. */
    ctx = (rocke_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(ctx == NULL)
    {
        rocke_fmoe_set_err(err, err_cap, "lower_to_llvm: out of memory");
        return ROCKE_ERR_OOM;
    }
    st = rocke_fmoe_build_ctx_init(ctx, spec, arch);
    if(st != ROCKE_OK)
    {
        const char* m = "lower_to_llvm: ctx init failed (tile policy / dtype)";
        rocke_fmoe_set_err(err, err_cap, m);
        rocke_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return st;
    }
    adj_spec = ctx->spec; /* tile-policy-adjusted spec */
    resolved_arch = ctx->arch; /* resolved launch arch      */

    switch(stage)
    {
    case ROCKE_FMOE_STAGE_ROUTER:
    {
        /* build_topk_softmax (spec.to_topk_softmax_spec()). */
        rocke_topk_softmax_spec_t s = rocke_fmoe_forward_spec_to_topk_softmax_spec(&adj_spec);
        st = rocke_topk_softmax_lower_to_llvm(&s, resolved_arch, flavor, out_ll, err, err_cap);
        break;
    }
    case ROCKE_FMOE_STAGE_GATE_UP_GEMM:
    case ROCKE_FMOE_STAGE_DOWN_GEMM:
    {
        /* The batched gate+up / down GEMM stage: spec.to_batched_gemm_spec().
         * The down stage reuses the same batched-GEMM builder shape (the
         * orchestrator parameterises it per-call; the representative .ll for
         * golden diffing is the batched_gemm builder). */
        char name_buf[256];
        rocke_batched_gemm_spec_t s;
        st = rocke_fmoe_forward_spec_to_batched_gemm_spec(
            &adj_spec, name_buf, sizeof(name_buf), &s);
        if(st != ROCKE_OK)
        {
            rocke_fmoe_set_err(err, err_cap, "lower_to_llvm: to_batched_gemm_spec failed");
            break;
        }
        st = rocke_batched_gemm_lower_to_llvm(&s, resolved_arch, flavor, out_ll, err, err_cap);
        break;
    }
    case ROCKE_FMOE_STAGE_SORT_HISTOGRAM:
    case ROCKE_FMOE_STAGE_SORT_SCAN:
    case ROCKE_FMOE_STAGE_SORT_SCATTER:
    {
        /* The 3-phase sort stages: spec.to_sort_spec() -> MoeSortingSpec, then
         * the matching sort sub-kernel's own build->lower convenience. */
        rocke_moe_sorting_spec_t ss = rocke_fmoe_to_sort_spec(&adj_spec);
        if(stage == ROCKE_FMOE_STAGE_SORT_HISTOGRAM)
        {
            st = rocke_build_moe_sort_histogram_lower_to_llvm(
                &ss, resolved_arch, flavor, out_ll, err, err_cap);
        }
        else if(stage == ROCKE_FMOE_STAGE_SORT_SCAN)
        {
            st = rocke_build_moe_sort_scan_lower_to_llvm(
                &ss, resolved_arch, flavor, out_ll, err, err_cap);
        }
        else
        {
            st = rocke_build_moe_sort_scatter_lower_to_llvm(
                &ss, resolved_arch, flavor, out_ll, err, err_cap);
        }
        break;
    }
    case ROCKE_FMOE_STAGE_GATHER:
    case ROCKE_FMOE_STAGE_SILU_MUL:
    case ROCKE_FMOE_STAGE_TOPK_REDUCE:
    {
        /* The streaming fused-MoE stages: spec.to_fused_moe_spec() ->
         * FusedMoeSpec, then the matching sub-kernel build->lower convenience.
         * The orchestrator's dynamic path uses the unpacked silu_mul (the packed
         * variant is a separate static-path stage), so SILU_MUL maps to
         * build_moe_silu_mul. */
        rocke_fused_moe_spec_t fs = rocke_fmoe_to_fused_moe_spec_local(&adj_spec);
        if(stage == ROCKE_FMOE_STAGE_GATHER)
        {
            st = rocke_moe_gather_lower_to_llvm(&fs, resolved_arch, flavor, out_ll, err, err_cap);
        }
        else if(stage == ROCKE_FMOE_STAGE_SILU_MUL)
        {
            st = rocke_moe_silu_mul_lower_to_llvm(&fs, resolved_arch, flavor, out_ll, err, err_cap);
        }
        else
        {
            st = rocke_moe_topk_weighted_reduce_lower_to_llvm(
                &fs, resolved_arch, flavor, out_ll, err, err_cap);
        }
        break;
    }
    default:
        rocke_fmoe_set_err(err, err_cap, "lower_to_llvm: unknown stage");
        st = ROCKE_ERR_VALUE;
        break;
    }

    rocke_fmoe_build_ctx_destroy(ctx);
    free(ctx);
    return st;
}
