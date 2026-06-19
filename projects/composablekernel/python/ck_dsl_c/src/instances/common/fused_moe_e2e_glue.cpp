// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_glue.c.c -- PUBLIC entry + GLUE
 * for the C99 chunked port of the end-to-end fused-MoE forward orchestrator
 * (ck_dsl/instances/common/fused_moe_e2e.py).
 *
 * SCOPE (this TU only -- the "bucket that calls phases in pipeline order"):
 *   - ckc_fmoe_forward_init / ckc_fmoe_forward_destroy
 *       (FusedMoeForward.__init__ public surface; delegates the heavy lifting
 *        to the peer ckc_fmoe_build_ctx_init / ckc_fmoe_build_ctx_destroy).
 *   - ckc_build_fused_moe_forward / ckc_build_fused_moe_forward_simple
 *       (allocate the orchestrator + its build ctx, run __init__'s tile policy +
 *        static gate, build the representative KernelDef, bind
 *        forward_fn = ckc_fmoe_forward_dispatch).
 *   - ckc_fused_moe_forward_free
 *       (free the KernelDef + bound orchestrator + internal ctx).
 *   - ckc_fused_moe_forward_lower_to_llvm
 *       (the per-stage .ll convenience switching on ckc_fmoe_stage_t).
 *   - ckc_fmoe_forward_dispatch
 *       (Python forward(): if use_static_offsets -> _forward_static
 *        else _forward_dynamic -- the single function that drives the
 *        router -> sort -> gather -> gate+up -> silu_mul -> down -> topk_reduce
 *        chain in declaration order; both phase bodies are peers).
 *
 * The phase bodies (ctx init, the two forward paths, the launcher-ensures, the
 * grouped-gemm dispatch) live in sibling TUs declared in
 * ckc/instance_fused_moe_e2e_internal.h. This TU owns ONLY the public surface,
 * the orchestrator/ctx lifetime, the forward_fn binding, and the call ordering.
 *
 * Byte-identical builder-call sequence -- the public forward() (Python 1597-1627):
 *   def forward(...):
 *       if self._use_static_offsets:
 *           self._forward_static(...); return
 *       self._forward_dynamic(...)
 * which this TU reproduces in ckc_fmoe_forward_dispatch.
 *
 * RUNTIME NOTE.  fused_moe_e2e.py defines NO new kernel: it is a host runtime
 * driver. The forward launch chain (HIP launches, torch workspaces, D->H copies)
 * has no IR-builder analogue in this codegen-only library, so the two forward
 * paths are TODO(port) stubs returning CKC_ERR_NOTIMPL (see the peer TUs). The
 * IR that DOES get emitted -- and is .ll-lowerable for golden-digest checks --
 * is the spec-selected sub-kernel builders, reached via the per-stage lower
 * convenience below.
 */
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_fused_moe_e2e.h"
#include "ckc/instance_fused_moe_e2e_internal.h"
#include "ckc/helper_ck_dsl.helpers.fused_moe_e2e_spec.h"
#include "ckc/instance_topk_softmax.h"
#include "ckc/instance_batched_gemm.h"
#include "ckc/lower_llvm.h"
#include "ckc/ir.h"

/* ===================================================================== *
 *  small local helpers
 * ===================================================================== */

/* Copy `msg` into the (err, err_cap) buffer, NUL-terminated and truncated to
 * fit. No-op if err is NULL or err_cap is 0. */
static void ckc_fmoe_set_err(char* err, size_t err_cap, const char* msg)
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

/* ===================================================================== *
 *  forward_fn trampoline  (KernelDef.forward_fn ABI)
 *
 *  The public forward_fn signature takes a `void* self` that is the
 *  ckc_fmoe_forward_t* the KernelDef threads through. The orchestrator's per-call
 *  state lives in the internal ckc_fmoe_build_ctx_t hung off self->ctx. This
 *  trampoline binds the call's device pointers onto the ctx working set and
 *  delegates to the pipeline dispatcher (Python FusedMoeForward.forward).
 * ===================================================================== */
static ckc_status_t ckc_fmoe_forward_fn_impl(void* self_v,
                                             uint64_t routing_logits,
                                             uint64_t X,
                                             uint64_t W_gate,
                                             uint64_t W_up,
                                             uint64_t W_down,
                                             uint64_t Y,
                                             uint64_t stream)
{
    ckc_fmoe_forward_t* self = (ckc_fmoe_forward_t*)self_v;
    ckc_fmoe_build_ctx_t* ctx;

    if(self == NULL || self->ctx == NULL)
    {
        return CKC_ERR_VALUE;
    }
    ctx = (ckc_fmoe_build_ctx_t*)self->ctx;

    /* Pipeline order is driven inside ckc_fmoe_forward_dispatch (Python
     * forward()): it binds the kwargs onto the ctx working set and selects
     * _forward_static / _forward_dynamic. */
    return ckc_fmoe_forward_dispatch(ctx, routing_logits, X, W_gate, W_up, W_down, Y, stream);
}

/* ===================================================================== *
 *  ckc_fmoe_forward_dispatch  (Python FusedMoeForward.forward, lines 1597-1627)
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
ckc_status_t ckc_fmoe_forward_dispatch(ckc_fmoe_build_ctx_t* ctx,
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
        return CKC_ERR_VALUE;
    }

    /* Bind the forward() kwargs onto the shared per-call working set so the
     * selected phase body reads the exact inputs (mirrors the Python kwargs
     * captured by _forward_static / _forward_dynamic). */
    ctx->routing_logits = routing_logits;
    ctx->X              = X;
    ctx->W_gate         = W_gate;
    ctx->W_up           = W_up;
    ctx->W_down         = W_down;
    ctx->Y              = Y;
    ctx->stream         = stream;

    /* Python (1608-1627): static-offset gate selects the path. */
    if(ctx->use_static_offsets)
    {
        return ckc_fmoe_forward_static(ctx);
    }
    return ckc_fmoe_forward_dynamic(ctx);
}

/* ===================================================================== *
 *  ckc_fmoe_forward_init  (FusedMoeForward.__init__ public surface)
 *
 *  Allocate + run the internal build ctx (which performs the arch resolve, the
 *  shape-aware tile-swap policy, the static-offset gate, and zeroes the launcher
 *  caches), then mirror the eagerly-computed scalars onto the public handle and
 *  bind the dispatch forward_fn.
 * ===================================================================== */
ckc_status_t ckc_fmoe_forward_init(ckc_fmoe_forward_t* self,
                                   const ckc_fmoe_forward_spec_t* spec,
                                   const char* arch)
{
    ckc_fmoe_build_ctx_t* ctx;
    ckc_status_t st;

    if(self == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }

    memset(self, 0, sizeof(*self));

    ctx = (ckc_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(ctx == NULL)
    {
        return CKC_ERR_OOM;
    }

    /* The internal ctx_init reproduces FusedMoeForward.__init__ (lines 692-831):
     * _resolve_launch_arch, the tile-swap policy, the static gate + slot size,
     * and zeroing the launcher / cache / weight slots. */
    st = ckc_fmoe_build_ctx_init(ctx, spec, arch);
    if(st != CKC_OK)
    {
        ckc_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return st;
    }

    /* Surface the eagerly-computed scalars on the public handle (the spec here
     * is the tile-policy-adjusted spec the ctx now holds). */
    self->spec               = ctx->spec;
    self->arch               = ctx->arch;
    self->use_static_offsets = ctx->use_static_offsets;
    self->static_slot_size   = ctx->static_slot_size;
    self->ctx                = ctx;
    self->forward_fn         = ckc_fmoe_forward_fn_impl;

    return CKC_OK;
}

void ckc_fmoe_forward_destroy(ckc_fmoe_forward_t* self)
{
    ckc_fmoe_build_ctx_t* ctx;
    if(self == NULL)
    {
        return;
    }
    ctx = (ckc_fmoe_build_ctx_t*)self->ctx;
    if(ctx != NULL)
    {
        ckc_fmoe_build_ctx_destroy(ctx);
        free(ctx);
    }
    self->ctx        = NULL;
    self->forward_fn = NULL;
}

/* ===================================================================== *
 *  representative KernelDef
 *
 *  The orchestrator emits no single monolithic kernel; the build entry returns a
 *  ckc_kernel_def_t* that ENCAPSULATES the pipeline. For the representative
 *  artifact we build the router stage (topk_softmax), which is the head of the
 *  declaration-order chain and is unconditional on every path. The orchestrator
 *  instance owns the builder backing this KernelDef; ckc_fused_moe_forward_free
 *  releases it.
 * ===================================================================== */

/* Owner record kept beside a built KernelDef so ckc_fused_moe_forward_free can
 * release the orchestrator + the builder that owns the KernelDef arena, and so
 * ckc_build_fused_moe_forward_simple can recover the bound `self` from the
 * returned KernelDef (the "KernelDef-adjacent registry"). */
typedef struct ckc_fmoe_kernel_owner
{
    ckc_kernel_def_t* kernel; /* the returned KernelDef (registry key) */
    ckc_ir_builder_t builder; /* owns the KernelDef arena              */
    ckc_fmoe_forward_t self;  /* the bound orchestrator instance       */
    struct ckc_fmoe_kernel_owner* next;
} ckc_fmoe_kernel_owner_t;

/* Tiny intrusive registry mapping KernelDef* -> owner record. Single-threaded
 * codegen use; linear scan. */
static ckc_fmoe_kernel_owner_t* g_fmoe_owners = NULL;

static void ckc_fmoe_registry_add(ckc_fmoe_kernel_owner_t* o)
{
    o->next       = g_fmoe_owners;
    g_fmoe_owners = o;
}

static ckc_fmoe_kernel_owner_t* ckc_fmoe_registry_take(ckc_kernel_def_t* kernel)
{
    ckc_fmoe_kernel_owner_t** pp = &g_fmoe_owners;
    while(*pp != NULL)
    {
        if((*pp)->kernel == kernel)
        {
            ckc_fmoe_kernel_owner_t* found = *pp;
            *pp                            = found->next;
            found->next                    = NULL;
            return found;
        }
        pp = &(*pp)->next;
    }
    return NULL;
}

/* Build the representative KernelDef into `b` from the (tile-policy-adjusted)
 * spec carried on the orchestrator instance. Returns the KernelDef or NULL. */
static ckc_kernel_def_t* ckc_fmoe_build_representative(ckc_ir_builder_t* b,
                                                       const ckc_fmoe_forward_t* self)
{
    ckc_topk_softmax_spec_t topk_spec;

    /* Router head of the pipeline: spec.to_topk_softmax_spec(). */
    topk_spec = ckc_fmoe_forward_spec_to_topk_softmax_spec(&self->spec);
    return ckc_build_topk_softmax_new(b, &topk_spec, self->arch);
}

ckc_kernel_def_t* ckc_build_fused_moe_forward(const ckc_fmoe_forward_spec_t* spec,
                                              const char* arch,
                                              ckc_fmoe_forward_t** out_self,
                                              ckc_fmoe_forward_fn_t* out_forward_fn)
{
    ckc_fmoe_kernel_owner_t* owner;
    ckc_kernel_def_t* kernel;

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

    owner = (ckc_fmoe_kernel_owner_t*)calloc(1, sizeof(*owner));
    if(owner == NULL)
    {
        return NULL;
    }

    /* __init__ tile policy + static gate + ctx alloc, and forward_fn bind. */
    if(ckc_fmoe_forward_init(&owner->self, spec, arch) != CKC_OK)
    {
        free(owner);
        return NULL;
    }

    /* Build the representative KernelDef (router head). The builder is owned by
     * the owner record so the KernelDef arena outlives this call. */
    kernel = ckc_fmoe_build_representative(&owner->builder, &owner->self);
    if(kernel == NULL)
    {
        ckc_ir_builder_free(&owner->builder);
        ckc_fmoe_forward_destroy(&owner->self);
        free(owner);
        return NULL;
    }

    owner->kernel = kernel;
    ckc_fmoe_registry_add(owner);

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

ckc_kernel_def_t* ckc_build_fused_moe_forward_simple(const ckc_fmoe_forward_spec_t* spec,
                                                     const char* arch)
{
    /* The bound `self` is recoverable from the returned KernelDef via the
     * registry; callers that need it use ckc_build_fused_moe_forward directly. */
    return ckc_build_fused_moe_forward(spec, arch, NULL, NULL);
}

void ckc_fused_moe_forward_free(ckc_kernel_def_t* kernel)
{
    ckc_fmoe_kernel_owner_t* owner;
    if(kernel == NULL)
    {
        return;
    }
    owner = ckc_fmoe_registry_take(kernel);
    if(owner == NULL)
    {
        /* Not one of ours (or already freed). Nothing safe to do. */
        return;
    }
    ckc_fmoe_forward_destroy(&owner->self);
    ckc_ir_builder_free(&owner->builder);
    free(owner);
}

/* ===================================================================== *
 *  ckc_fused_moe_forward_lower_to_llvm  (per-stage .ll convenience)
 *
 *  Lowers ONE named pipeline stage to AMDGPU LLVM IR text. The spec's tile +
 *  trait policy is applied exactly as the live orchestrator would: we run the
 *  __init__ tile-swap policy via the build ctx, then convert the (adjusted) spec
 *  to the selected sub-kernel spec and delegate to that sub-kernel's own
 *  build->lower convenience.
 *
 *  Stages whose sub-kernel spec converter is not part of this port's header
 *  surface (sort hist/scan/scatter, gather, silu_mul, topk_reduce -- the
 *  fmoe -> sort/fused_moe spec converters are not exposed) are TODO(port) and
 *  return CKC_ERR_NOTIMPL with a diagnostic.
 * ===================================================================== */
ckc_status_t ckc_fused_moe_forward_lower_to_llvm(const ckc_fmoe_forward_spec_t* spec,
                                                 const char* arch,
                                                 ckc_fmoe_stage_t stage,
                                                 ckc_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    ckc_fmoe_build_ctx_t* ctx;
    ckc_status_t st;
    const char* resolved_arch;
    ckc_fmoe_forward_spec_t adj_spec;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        ckc_fmoe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }

    /* Apply __init__'s arch resolve + tile-swap policy via the build ctx so the
     * stage builders see the exact tile the live orchestrator selects. */
    ctx = (ckc_fmoe_build_ctx_t*)calloc(1, sizeof(*ctx));
    if(ctx == NULL)
    {
        ckc_fmoe_set_err(err, err_cap, "lower_to_llvm: out of memory");
        return CKC_ERR_OOM;
    }
    st = ckc_fmoe_build_ctx_init(ctx, spec, arch);
    if(st != CKC_OK)
    {
        const char* m = "lower_to_llvm: ctx init failed (tile policy / dtype)";
        ckc_fmoe_set_err(err, err_cap, m);
        ckc_fmoe_build_ctx_destroy(ctx);
        free(ctx);
        return st;
    }
    adj_spec      = ctx->spec; /* tile-policy-adjusted spec */
    resolved_arch = ctx->arch; /* resolved launch arch      */

    switch(stage)
    {
    case CKC_FMOE_STAGE_ROUTER: {
        /* build_topk_softmax (spec.to_topk_softmax_spec()). */
        ckc_topk_softmax_spec_t s = ckc_fmoe_forward_spec_to_topk_softmax_spec(&adj_spec);
        st = ckc_topk_softmax_lower_to_llvm(&s, resolved_arch, flavor, out_ll, err, err_cap);
        break;
    }
    case CKC_FMOE_STAGE_GATE_UP_GEMM:
    case CKC_FMOE_STAGE_DOWN_GEMM: {
        /* The batched gate+up / down GEMM stage: spec.to_batched_gemm_spec().
         * The down stage reuses the same batched-GEMM builder shape (the
         * orchestrator parameterises it per-call; the representative .ll for
         * golden diffing is the batched_gemm builder). */
        char name_buf[256];
        ckc_batched_gemm_spec_t s;
        st = ckc_fmoe_forward_spec_to_batched_gemm_spec(&adj_spec, name_buf, sizeof(name_buf), &s);
        if(st != CKC_OK)
        {
            ckc_fmoe_set_err(err, err_cap, "lower_to_llvm: to_batched_gemm_spec failed");
            break;
        }
        st = ckc_batched_gemm_lower_to_llvm(&s, resolved_arch, flavor, out_ll, err, err_cap);
        break;
    }
    case CKC_FMOE_STAGE_SORT_HISTOGRAM:
    case CKC_FMOE_STAGE_SORT_SCAN:
    case CKC_FMOE_STAGE_SORT_SCATTER:
    case CKC_FMOE_STAGE_GATHER:
    case CKC_FMOE_STAGE_SILU_MUL:
    case CKC_FMOE_STAGE_TOPK_REDUCE:
        /* TODO(port): the fmoe -> sort / fused_moe spec converters are not
         * part of this port's header surface; these stages lower via their
         * own sub-kernel convenience once those converters are exposed. */
        ckc_fmoe_set_err(err, err_cap, "lower_to_llvm: stage not yet wired (TODO port)");
        st = CKC_ERR_NOTIMPL;
        break;
    default:
        ckc_fmoe_set_err(err, err_cap, "lower_to_llvm: unknown stage");
        st = CKC_ERR_VALUE;
        break;
    }

    ckc_fmoe_build_ctx_destroy(ctx);
    free(ctx);
    return st;
}
