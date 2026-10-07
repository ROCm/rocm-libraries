// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ port of the implicit-GEMM backward-weight convolution builder
 * (rocke/instances/common/conv_implicit_gemm_wgrad.py).
 *
 * GEMM orientation (wgrad):
 *   M     = K          (output channels -- weight rows)
 *   N_wg  = Y*X*C      (filter spatial x input channel -- weight cols)
 *   K_wg  = N*Ho*Wo    (output spatial positions -- reduction)
 *
 * Operand roles:
 *   A  = dY  (NHWK, output gradient)  -- the M-row / K_wg-reduction operand
 *   B  = X   (NHWC, input activations) -- reuses the forward make_a_descriptor
 *   D  = dW  (KYXC, weight gradient)  -- written at the end
 *
 * Implementation strategy: reuse the forward-conv phase infrastructure
 * (rocke_conv_build_ctx_t, all K-loop drivers, MFMA/WMMA phases, epilogue) by
 * building an ImplicitGemmConvSpec that matches the wgrad GEMM geometry
 * (tile_m/n/k come from wgrad spec; problem is adapted so the existing code
 * sees M/N/K_gemm correctly) and substituting wgrad-specific descriptors into
 * the ctx after rocke_conv_build_ctx_init populates it.
 *
 * load_vec_a = load_vec_b = 1 always (Python comment: "Force vec=1 for both
 * operands regardless of what the auto-picker or the caller requests.").
 */
#include "rocke/instance_conv_implicit_gemm_wgrad.h"

/* Mirrors _DEFAULT_WS_REPLICAS in the Python instance. */
#ifndef ROCKE_WGRAD_DEFAULT_WS_REPLICAS
#define ROCKE_WGRAD_DEFAULT_WS_REPLICAS 8
#endif

#include <cstdint> /* int64_t */
#include <cstdio> /* snprintf */
#include <cstring> /* strcmp, memset, memcpy */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_abi.h"
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/lower_llvm.h"

// ---------------------------------------------------------------------------
// Pure-arithmetic spec properties  (Python WgradConvSpec @property)
// ---------------------------------------------------------------------------

rocke_implicit_gemm_conv_wgrad_spec_t rocke_implicit_gemm_conv_wgrad_spec_default(void)
{
    rocke_implicit_gemm_conv_wgrad_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "conv_igemm_wgrad";
    s.dtype_a = "fp16";
    s.dtype_b = "fp16";
    s.dtype_d = "fp16";
    s.dtype_acc = "fp32";
    s.tile_m = 64;
    s.tile_n = 64;
    s.tile_k = 64;
    s.warp_m = 2;
    s.warp_n = 2;
    s.warp_tile_m = 32;
    s.warp_tile_n = 32;
    s.warp_tile_k = 16;
    s.wave_size = 64;
    s.pipeline = "mem";
    s.epilogue = "default";
    s.chiplet_wgm = 8;
    s.chiplet_num_xcds = 8;
    s.chiplet_chunk_size = 64;
    s.split_k = 1;
    s.ws_replicas = ROCKE_WGRAD_DEFAULT_WS_REPLICAS;
    return s;
}

int rocke_wgrad_conv_spec_block_size(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    return s->warp_m * s->warp_n * s->wave_size;
}

int rocke_wgrad_conv_spec_k_atoms_per_tile_k(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    return s->tile_k / s->warp_tile_k;
}

int rocke_wgrad_conv_spec_mfmas_per_warp_m(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    return s->tile_m / (s->warp_m * s->warp_tile_m);
}

int rocke_wgrad_conv_spec_mfmas_per_warp_n(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    return s->tile_n / (s->warp_n * s->warp_tile_n);
}

/* wg_M = K  (output channels, groups=1 always for wgrad) */
int rocke_wgrad_conv_spec_wg_M(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    return s->problem.K;
}

/* wg_N = Z*Y*X*C  (filter spatial x input channel) */
int rocke_wgrad_conv_spec_wg_N(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    const rocke_conv_problem_t* p = &s->problem;
    int z = p->is_3d ? p->Z : 1;
    return z * p->Y * p->X * p->C;
}

/* wg_K = N*Ho*Wo  (output spatial positions) */
int rocke_wgrad_conv_spec_wg_K(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    const rocke_conv_problem_t* p = &s->problem;
    int ho = rocke_conv_problem_ho(p);
    int wo = rocke_conv_problem_wo(p);
    int base = p->N * ho * wo;
    if(p->is_3d)
    {
        base *= rocke_conv_problem_do(p);
    }
    return base;
}

bool rocke_wgrad_conv_spec_is_deterministic(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    /* split_k == 1 (and the -1 auto sentinel, which only ever resolves to >= 1):
     *   plain store, no atomics, deterministic.
     * split_k > 1: non-deterministic, two_stage or not. The two-stage path is
     *   NOT an exception. It used to be -- Stage 1 wrote one private slab per
     *   K-slice and Stage 2 folded them in a fixed order -- but Stage 1 now
     *   f32-atomic-adds into ws_replicas shared slabs, so the order in which a
     *   group's slices land in a slab is scheduler-dependent and f32 addition
     *   is not associative. Stage 2's fold over the replicas is ordered, which
     *   does nothing for partial sums that were already reordered. */
    return s->split_k <= 1;
}

size_t rocke_wgrad_conv_workspace_bytes(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    if(!s->two_stage)
        return 0;
    /* Two-stage needs a split (split_k > 1); the scratch does not depend on
     * the degree. */
    if(s->split_k <= 1)
        return 0;
    int wg_M = rocke_wgrad_conv_spec_wg_M(s);
    int wg_N = rocke_wgrad_conv_spec_wg_N(s);
    int groups = s->problem.groups > 0 ? s->problem.groups : 1;
    /* R replica slabs per group, no split_k factor: a group's K-slices
     * atomic-add on top of each other inside those slabs rather than each
     * getting its own. Mirrors Python wgrad_two_stage_workspace_nbytes. */
    int reps = s->ws_replicas > 0 ? s->ws_replicas : 1;
    return (size_t)groups * (size_t)reps * (size_t)wg_M * (size_t)wg_N * sizeof(float);
}

/* wg_K_padded = ceil(wg_K / (tile_k * split_k)) * (tile_k * split_k) */
int rocke_wgrad_conv_spec_wg_K_padded(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    int sk = (s->split_k > 1) ? s->split_k : 1;
    int stride = s->tile_k * sk;
    int k = rocke_wgrad_conv_spec_wg_K(s);
    return ((k + stride - 1) / stride) * stride;
}

// ---------------------------------------------------------------------------
// Kernel name
// ---------------------------------------------------------------------------

rocke_status_t rocke_wgrad_conv_spec_kernel_name(const rocke_implicit_gemm_conv_wgrad_spec_t* s,
                                                 char* out,
                                                 size_t out_cap)
{
    /*
     * Python:
     *   kernel_name_join(
     *     self.name,
     *     p.short(),
     *     f"t{tile_m}x{tile_n}x{tile_k}",
     *     f"w{warp_m}x{warp_n}",
     *     f"a{warp_tile_m}x{warp_tile_n}x{warp_tile_k}",
     *     f"{pipeline}_{epilogue}",
     *     self.acc_epilogue.tag(),   -- always "" (omitted) in this port
     *     flags={"async": async_dma, "kouter": lds_k_outer, "pad{N}": ...,
     *            "spk": split_k>1, "spkauto": split_k==-1, "twostage": ...,
     *            "wsr{N}": ..., "unroll": unroll_k},
     *   )
     */
    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;

    char short_buf[128];
    char t_buf[48];
    char w_buf[32];
    char a_buf[48];
    char pe_buf[64];

    rocke_status_t st = rocke_conv_problem_short(&s->problem, short_buf, sizeof(short_buf), NULL);
    if(st != ROCKE_OK)
        return st;

    snprintf(t_buf, sizeof(t_buf), "t%dx%dx%d", s->tile_m, s->tile_n, s->tile_k);
    snprintf(w_buf, sizeof(w_buf), "w%dx%d", s->warp_m, s->warp_n);
    snprintf(a_buf, sizeof(a_buf), "a%dx%dx%d", s->warp_tile_m, s->warp_tile_n, s->warp_tile_k);
    snprintf(pe_buf,
             sizeof(pe_buf),
             "%s_%s",
             s->pipeline ? s->pipeline : "",
             s->epilogue ? s->epilogue : "");

    /* acc_epilogue.tag() is always "" in this port (field omitted from struct). */
    const char* parts[5] = {short_buf, t_buf, w_buf, a_buf, pe_buf};

    /* flags: async, kouter, pad{N}, spk/spkauto, twostage, wsr{N}, unroll */
    char pad_flag[32] = {0};
    const char* flag_names[7];
    int flag_on[7];
    int n_flags = 0;

    flag_names[n_flags] = "async";
    flag_on[n_flags] = s->async_dma ? 1 : 0;
    n_flags++;

    flag_names[n_flags] = "kouter";
    flag_on[n_flags] = s->lds_k_outer ? 1 : 0;
    n_flags++;

    /* An explicit lds_k_pad changes the LDS row stride and so the emitted code;
     * without it in the name two pads collide on one symbol and a cache keyed on
     * the kernel name hands them the same binary.  Only tagged when set, so a
     * spec that leaves it unset keeps its historical name.  Mirrors Python. */
    if(s->has_lds_k_pad)
    {
        snprintf(pad_flag, sizeof(pad_flag), "pad%d", s->lds_k_pad);
        flag_names[n_flags] = pad_flag;
        flag_on[n_flags] = 1;
        n_flags++;
    }

    /* The split degree is a launch parameter, so the name records only that
     * the kernel splits: every degree > 1 is one binary. Mirrors Python. */
    if(s->split_k > 1)
    {
        flag_names[n_flags] = "spk";
        flag_on[n_flags] = 1;
        n_flags++;
    }
    else if(s->split_k == -1)
    {
        flag_names[n_flags] = "spkauto";
        flag_on[n_flags] = 1;
        n_flags++;
    }

    /* Tag the EFFECTIVE two-stage flag, not the raw field.  The builder promotes
     * two_stage into a local (effective_two_stage) without writing back to
     * spec->two_stage, so naming
     * off s->two_stage would emit a two-stage body -- which carries an extra
     * `ws` workspace pointer parameter and needs a Stage-2 reduce launch --
     * under a symbol identical to the split-K atomic kernel built from the same
     * spec with two_stage=false.  That is both an ABI collision for a
     * cache keyed on the kernel name (same hazard the lds_k_pad comment above
     * describes) and a byte-identity break against Python, which promotes by
     * rewriting the spec before the IRBuilder is named. */
    flag_names[n_flags] = "twostage";
    flag_on[n_flags] = s->two_stage ? 1 : 0;
    n_flags++;

    /* wsr<R>: same reasoning as gm/pad -- the replica count changes the scratch
     * addressing and so the emitted body. Gated on two_stage as well as on the
     * default, because there is no scratch at all on the atomic path; tagging
     * it there renames every single-stage wgrad kernel over a knob its body
     * never reads. Tracks the `twostage` flag above. */
    char wsr_buf[32];
    if(s->two_stage && s->ws_replicas > 1)
    {
        snprintf(wsr_buf, sizeof(wsr_buf), "wsr%d", s->ws_replicas);
        flag_names[n_flags] = wsr_buf;
        flag_on[n_flags] = 1;
        n_flags++;
    }

    /* unroll_k hand-rolls a double-buffered K-loop -- a different body under
     * the same name otherwise. Only tagged when set, so every other kernel
     * keeps its name. Mirrors Python. */
    flag_names[n_flags] = "unroll";
    flag_on[n_flags] = s->unroll_k ? 1 : 0;
    n_flags++;

    return rocke_kernel_name_join(
        s->name, parts, 5, flag_names, flag_on, n_flags, out, out_cap, NULL);
}

// ---------------------------------------------------------------------------
// is_valid_wgrad_spec
// ---------------------------------------------------------------------------

bool rocke_implicit_gemm_conv_wgrad_is_valid_spec(const rocke_implicit_gemm_conv_wgrad_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    /*
     * Mirror Python is_valid_wgrad_spec -- geometry + block size + MMA atom +
     * LDS + WMMA narrow subset + split_k + vec_c gates.
     * Build an equivalent ImplicitGemmConvSpec and delegate to the forward-conv
     * validator (rocke_implicit_gemm_conv_is_valid_spec) which implements the
     * same gates.  The spec adapter: M=tile_m, N=tile_n, K=tile_k, same atoms.
     */
    if(s == NULL)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";

    /* groups > 1 is not supported for wgrad (Python validate() raises on it). */
    if(s->problem.groups != 1)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "grouped convolution (groups=%d > 1) is not supported for wgrad",
                     s->problem.groups);
        return false;
    }

    /* geometry */
    if(s->tile_m % (s->warp_m * s->warp_tile_m))
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "tile_m not divisible by warp_m * warp_tile_m");
        return false;
    }
    if(s->tile_n % (s->warp_n * s->warp_tile_n))
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "tile_n not divisible by warp_n * warp_tile_n");
        return false;
    }
    if(s->tile_k % s->warp_tile_k)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "tile_k not divisible by warp_tile_k");
        return false;
    }
    int block_size = rocke_wgrad_conv_spec_block_size(s);
    if(block_size > 1024)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "block_size %d > 1024", block_size);
        return false;
    }

    int sk = s->split_k;
    if(sk < -1 || sk == 0)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "split_k must be -1 (auto), 1, or >1 (got %d)", sk);
        return false;
    }
    /* two_stage=true requires split_k > 1 (or -1 for auto); mirrors Python validate(). */
    if(s->two_stage && sk == 1)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "two_stage=true requires split_k > 1 (got split_k=1); "
                     "with split_k=1 there is nothing to reduce and two_stage is a no-op");
        return false;
    }
    /* Mirrors Python is_valid_wgrad_spec / WgradConvSpec.validate(). Without
     * this a C++-built spec with a zero or negative count is accepted, the
     * emitter's `reps > 1` guards elide the slab term, and Stage 1 silently
     * writes a single slab -- a layout Stage 2 (which validates the same field)
     * will not fold. */
    if(s->ws_replicas < 1)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "ws_replicas must be >= 1 (got %d); it is the number of scratch "
                     "slabs a group's K-slices spread over",
                     s->ws_replicas);
        return false;
    }

    /* split_k > 1 (atomic) requires a MFMA arch
     * (ctx->atom != NULL at build time).
     *
     * TODO: gate on resolved wave_size == 64 / op->family == "mma" (matching
     * Python which uses family == "wmma") instead of the arch string, so
     * gfx10* and any future or unknown arch prefix cannot fall through.  This
     * is not reachable on today's supported targets but would be more robust. */
    if(sk > 1)
    {
        /* Quick arch check: gfx11xx / gfx12xx are RDNA.
         * Note: gfx10* and any unknown prefix are not rejected here — they would
         * reach the split-K epilogue where ctx->atom is NULL (null deref). */
        if(arch && (strncmp(arch, "gfx11", 5) == 0 || strncmp(arch, "gfx12", 5) == 0))
        {
            if(reason && reason_cap)
                snprintf(reason, reason_cap, "split_k atomic is only supported on CDNA targets");
            return false;
        }

        /* For fp16/bf16 output the packed atomic writes pairs of elements via
         * global_atomic_add_pk_f16/bf16.  The pair is addressed as a flat
         * `m * wg_N + n` element index with n rounded down to even, so it is
         * dword-aligned iff the dW row length wg_N = Z*Y*X*(C/groups) is even.
         *
         * Two corrections to the previous form of this gate, both mirrored from
         * Python is_valid_wgrad_spec / WgradConvSpec.validate():
         *   - it tested the dense problem.C, but the dW row is per-group, so on
         *     any grouped conv it disagreed with Python (which tests cpg);
         *   - it did not exempt the two-stage path, which stores f32 to a
         *     workspace and emits no atomic at all.
         * The local per-group computation is deliberate: the shared
         * rocke_wgrad_conv_spec_wg_N() helper still returns the dense Z*Y*X*C
         * and is used for workspace sizing, so it is not interchangeable here. */
        const bool effective_two_stage_gate = s->two_stage && sk > 1;
        const char* dt = s->dtype_d ? s->dtype_d : "fp16";
        if(!effective_two_stage_gate && (strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0))
        {
            const int groups_v = s->problem.groups > 0 ? s->problem.groups : 1;
            const int cpg_v = s->problem.C / groups_v;
            const int z_v = s->problem.is_3d ? s->problem.Z : 1;
            /* 64-bit: every factor is an int from the problem description, so a
             * 32-bit product is UB on a pathological shape even though no real
             * conv reaches it. The comparison below only needs the parity. */
            const int64_t wg_N_v = (int64_t)z_v * s->problem.Y * s->problem.X * cpg_v;

            /* The packed atomic needs BOTH halves of "can this problem form
             * pairs at all", mirroring Python wgrad_atomic_epilogue_available():
             *   - an even dW row length wg_N = Z*Y*X*(C/groups), so the flat
             *     `m * wg_N + n` pair index stays dword-aligned; and
             *   - an even store-vector width, because the epilogue emits sv/2
             *     pairs per thread and sv == 1 (what cpg == 1 yields) leaves no
             *     partner.
             * Checking only wg_N admits a spec that CShuffleEpilogue::atomic_store
             * then rejects -- the same admits/build split this gate exists to
             * close. store_vec mirrors default_vector_sizes(..., split_k=1):
             * widest of 8/4/2/1 dividing the channel run (per-group when grouped). */
            int store_vec;
            if(s->has_vector_size_c)
            {
                store_vec = s->vector_size_c;
            }
            else
            {
                /* vec_c is sized by the C run only (dW's last dim is the C axis). */
                const int vc_c = (s->problem.groups > 1) ? cpg_v : s->problem.C;
                store_vec = (vc_c % 8 == 0) ? 8 : (vc_c % 4 == 0) ? 4 : (vc_c % 2 == 0) ? 2 : 1;
            }

            if(wg_N_v % 2 != 0 || store_vec % 2 != 0)
            {
                if(reason && reason_cap)
                    snprintf(reason,
                             reason_cap,
                             "split_k atomic with dtype_d=%s requires an even dW row length "
                             "wg_N=Z*Y*X*(C/groups) and an even store-vector width (packed "
                             "<2 x dtype> atomic pairs are dword-aligned only on an even row, "
                             "and sv=1 leaves no partner); got wg_N=%lld, store_vec=%d "
                             "(Z=%d, Y=%d, X=%d, cpg=%d). Use two_stage=true "
                             "to reach split-K via the f32 "
                             "workspace path, which emits no atomics.",
                             dt,
                             (long long)wg_N_v,
                             store_vec,
                             z_v,
                             s->problem.Y,
                             s->problem.X,
                             cpg_v);
                return false;
            }
        }
    }

    /* For bf16/fp16 output the *packed atomic* store emits zero-fill pairs at
     * the scattered MFMA layout, so it needs cshuffle's contiguous pairs.  This
     * is an atomic-epilogue constraint only, and there are two ways to not be
     * on it: at split_k == 1 the epilogue is a direct store, and under
     * two_stage (with split_k > 1) it is a
     * workspace store.  Neither emits packed atomics, so the default epilogue
     * is fine for both.  Guarding on sk rather than on dtype alone keeps the
     * non-atomic 16-bit output path reachable -- it is the only one WMMA wgrad
     * can use, since WMMA rejects cshuffle.
     * Matches Python is_valid_wgrad_spec / validate(): _needs_atomic guard. */
    if(sk > 1)
    {
        /* Mirrors the builder's is_two_stage = is_split_k && two_stage: a
         * two-stage spec takes the f32 scratch epilogue, not the packed atomic. */
        bool effective_two_stage_v = s->two_stage;
        if(!effective_two_stage_v)
        {
            const char* dt = s->dtype_d ? s->dtype_d : "fp16";
            bool is_default_epi = (s->epilogue == NULL || strcmp(s->epilogue, "default") == 0);
            if(is_default_epi && (strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0))
            {
                if(reason && reason_cap)
                    snprintf(reason,
                             reason_cap,
                             "split_k atomic with dtype_d=%s requires epilogue='cshuffle' "
                             "(default emits zero-fill packed atomics with scattered MFMA "
                             "layout; cshuffle produces contiguous pairs)",
                             dt);
                return false;
            }
        }
    }

    /* wgrad can only use the direct global->LDS load on the K-outer tile.
     * raw_ptr_buffer_load_lds
     * maps contiguous-global bytes to contiguous-LDS bytes, so the contiguous
     * LDS axis must be the global stride-1 axis; wgrad reduces over
     * K_wg = N*Ho*Wo, which is stride-K in dY (NHWK) and stride-C in X (NHWC).
     * Emitting it produces numerically wrong dW.  Mirrors Python
     * is_valid_wgrad_spec / WgradConvSpec.validate(). */
    /* The K-outer row stride comes from ROCKE_WGRAD_KOUTER_PAD in the builder,
     * so an explicit pad changes the kernel name and the LDS budget charged
     * here without changing a single emitted op. Reject rather than ignore.
     * Mirrors Python is_valid_wgrad_spec / WgradConvSpec.validate. */
    if(s->lds_k_outer && s->has_lds_k_pad)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer does not honour an explicit lds_k_pad: the "
                     "K-outer row stride is fixed by the transpose-read bank "
                     "analysis, not by the layout; got lds_k_pad=%d",
                     s->lds_k_pad);
        return false;
    }

    /* validate() covers the dtype/atom/wave_size half of the lds_k_outer gate,
     * but it has no arch to check against. Without this an older target builds
     * cleanly and emits ds_read_tr16_b64, which only exists on CDNA4.
     * Mirrors Python _LDS_K_OUTER_ARCH in conv_implicit_gemm_wgrad.py. */
    if(s->lds_k_outer)
    {
        /* Two regimes: gfx950 wave64 (ds_read_b64_tr_b16) and gfx1250 wave32
         * (ds_load_tr16_b128). The arch and the wave must agree, since the lane
         * mapping is derived per wave size. Mirrors _LDS_K_OUTER_ARCH_WAVE. */
        const bool ko_950 = (arch && strcmp(arch, "gfx950") == 0);
        const bool ko_1250 = (arch && strcmp(arch, "gfx1250") == 0);
        if(!ko_950 && !ko_1250)
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer requires gfx950 or gfx1250 (the LDS transpose "
                         "read); got %s",
                         arch ? arch : "(null)");
            return false;
        }
        const int ko_wave = ko_950 ? 64 : 32;
        if(s->wave_size != ko_wave)
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer on %s requires wave_size=%d; got %d",
                         arch,
                         ko_wave,
                         s->wave_size);
            return false;
        }
    }

    if(s->async_dma && !s->lds_k_outer)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "wgrad async_dma requires lds_k_outer=True: the direct "
                     "global->LDS load needs a stride-1 reduction axis, which wgrad "
                     "only has once the tile is stored K-outer");
        return false;
    }

    if(s->pipeline && strcmp(s->pipeline, "basic") == 0 && s->async_dma)
    {
        if(reason && reason_cap)
            snprintf(reason, reason_cap, "pipeline='basic' is incompatible with async_dma=True");
        return false;
    }

    /* The wgrad builder has no load/math wave split, so "wavelet" would build
     * the "mem" kernel under another name. Mirrors Python is_valid_wgrad_spec. */
    if(s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' is not implemented for wgrad (it would build "
                     "the 'mem' kernel); use pipeline='mem'");
        return false;
    }

    /* lds_k_outer: ds_read_b64_tr_b16 is a gfx950 wave64 16-bit transpose read.
     * The fragment formula is derived per 16-lane group over a 16- or 32-wide
     * atom edge; it carries the per-lane fragment length (4 for 16x16x16, 8 for
     * 16x16x32 and 32x32x16) rather than assuming 8, so every 16-bit atom on
     * those edges is covered.  Mirrors WgradConvSpec.validate(). */
    if(s->lds_k_outer)
    {
        const char* da = s->dtype_a ? s->dtype_a : "fp16";
        const char* db = s->dtype_b ? s->dtype_b : "fp16";
        bool a16 = (strcmp(da, "fp16") == 0 || strcmp(da, "bf16") == 0);
        bool b16 = (strcmp(db, "fp16") == 0 || strcmp(db, "bf16") == 0);
        if(!a16 || !b16)
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer requires 16-bit A/B dtypes (ds_read_b64_tr_b16 "
                         "is a 16-bit transpose read); got dtype_a=%s dtype_b=%s",
                         da,
                         db);
            return false;
        }
        if((s->warp_tile_m != 16 && s->warp_tile_m != 32)
           || (s->warp_tile_n != 16 && s->warp_tile_n != 32))
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer requires warp_tile_m/n in (16, 32) -- the "
                         "transpose-read lane mapping is derived per 16-lane group "
                         "over the atom edge; got %dx%d",
                         s->warp_tile_m,
                         s->warp_tile_n);
            return false;
        }
        if(s->wave_size != 64 && s->wave_size != 32)
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer requires wave_size 64 (ds_read_b64_tr_b16) or "
                         "32 (ds_load_tr16_b128); got %d",
                         s->wave_size);
            return false;
        }
        /* One atom in the wave32 regime: gfx1250 WMMA 16x16x32, whose A/B
         * fragment is 16 per lane and whose B lane map is col = lane % 16,
         * k = (lane / 16) * 16 + i. Nothing else is derived. */
        if(s->wave_size == 32
           && (s->warp_tile_m != 16 || s->warp_tile_n != 16 || s->warp_tile_k != 32))
        {
            if(reason && reason_cap)
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer on wave32 supports only the 16x16x32 atom "
                         "(got %dx%dx%d)",
                         s->warp_tile_m,
                         s->warp_tile_n,
                         s->warp_tile_k);
            return false;
        }
    }

    /* Delegate the MMA-atom + LDS + WMMA gates to the forward validator via an
     * adapter spec.  The forward validator only reads: tile_m/n/k, warp_m/n,
     * warp_tile_m/n/k, wave_size, pipeline, epilogue, async_dma, unroll_k,
     * dtype_a/b/d, block_size (derived), and lds_k_pad/lds_layout for LDS.
     * All other fields (groups, chiplet_swizzle, etc.) default to non-blocking
     * values. */
    rocke_implicit_gemm_conv_spec_t fwd = rocke_implicit_gemm_conv_spec_default();
    fwd.tile_m = s->tile_m;
    fwd.tile_n = s->tile_n;
    fwd.tile_k = s->tile_k;
    fwd.warp_m = s->warp_m;
    fwd.warp_n = s->warp_n;
    fwd.warp_tile_m = s->warp_tile_m;
    fwd.warp_tile_n = s->warp_tile_n;
    fwd.warp_tile_k = s->warp_tile_k;
    fwd.wave_size = s->wave_size;
    fwd.pipeline = s->pipeline;
    fwd.epilogue = s->epilogue;
    fwd.async_dma = s->async_dma;
    fwd.unroll_k = s->unroll_k;
    fwd.dtype_a = s->dtype_a;
    fwd.dtype_b = s->dtype_b;
    fwd.dtype_d = s->dtype_d;
    /* Use a dummy problem that keeps K_gemm/M positive for the LDS calc. */
    fwd.problem = rocke_conv_problem_default(1, 8, 8, 16, 16, 1, 1);

    if(!rocke_implicit_gemm_conv_is_valid_spec(&fwd, arch, reason, reason_cap))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Wgrad-specific tensor descriptors
// ---------------------------------------------------------------------------

/*
 * wgrad_make_dy_descriptor:
 *   dY stored NHWK layout.  In the wgrad GEMM:
 *     - M dimension (wg_M = K) indexes output channels -> called "m" to match
 *       the forward rocke_conv_a_descriptor which queries ctx->A_desc with ("m","k")
 *     - K_wg reduction (output positions) -> called "k"
 *
 * So the user-facing coords must be ("m"=k_out, "k"=k_wg_red).
 *
 * Python original uses ("k_wg", "k_out") but we rename to ("k"=k_wg_red, "m"=k_out)
 * so rocke_conv_a_descriptor (which calls A_desc.offset(m=m_val, k=k_val)) works
 * correctly: m_val = block_m_off + row (indexes the M tile = output channels)
 *             k_val = k_off + col (indexes the K reduction = output positions).
 */
static rocke_tensor_descriptor_t* wgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                           const rocke_conv_problem_t* p)
{
    int ho = rocke_conv_problem_ho(p);
    int wo = rocke_conv_problem_wo(p);

    const char* into[4];
    int dims[4];
    int n_into;

    if(p->is_3d)
    {
        int do_ = rocke_conv_problem_do(p);
        /* naive(NDHWK): last coord is "m" (= k_out, the M dimension of wgrad) */
        int lengths[5] = {p->N, do_, ho, wo, p->K};
        const char* coords[5] = {"n", "do_", "ho", "wo", "m"};
        rocke_tensor_descriptor_t* desc
            = rocke_tensor_descriptor_naive(b, "dY_ndhwk", lengths, 5, NULL, coords, 5);
        if(desc == NULL)
            return NULL;
        /* unmerge "k" (k_wg_red) -> (n, do_, ho, wo) so the user sees (k, m) */
        into[0] = "n";
        into[1] = "do_";
        into[2] = "ho";
        into[3] = "wo";
        dims[0] = p->N;
        dims[1] = do_;
        dims[2] = ho;
        dims[3] = wo;
        n_into = 4;
        const rocke_transform_t* xf = rocke_unmerge_magic(b, "k", into, n_into, dims);
        if(xf == NULL)
            return NULL;
        return rocke_tensor_descriptor_transform(b, desc, &xf, 1);
    }

    /* 2-D: naive(NHWK), last coord "m" (= k_out, the M/output-channel dimension) */
    int lengths[4] = {p->N, ho, wo, p->K};
    const char* coords[4] = {"n", "ho", "wo", "m"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dY_nhwk", lengths, 4, NULL, coords, 4);
    if(desc == NULL)
        return NULL;
    /* unmerge "k" (= k_wg_red, the K-reduction dimension) -> (n, ho, wo) */
    into[0] = "n";
    into[1] = "ho";
    into[2] = "wo";
    dims[0] = p->N;
    dims[1] = ho;
    dims[2] = wo;
    n_into = 3;
    const rocke_transform_t* xf = rocke_unmerge_magic(b, "k", into, n_into, dims);
    if(xf == NULL)
        return NULL;
    return rocke_tensor_descriptor_transform(b, desc, &xf, 1);
}

/*
 * wgrad_make_x_descriptor:
 *   X (input activations) is the B operand in the wgrad GEMM.
 *   rocke_conv_b_descriptor queries ctx->B_desc with coord names ("k_out", "k_gemm")
 *   where: k_out = block_n_off + row  (= n_wg, the filter+channel N dimension)
 *          k_gemm = k_off + col        (= k_wg_red, the output-position K reduction)
 *
 *   This is the same NHWC transform DAG as make_a_descriptor(decompose_m=True)
 *   but with coord name aliases:
 *     "k_gemm" (outer, = output position)   <-> "m" in the A descriptor
 *     "k_out"  (inner, = filter+channel)    <-> "k" in the A descriptor
 *
 *   We build it by calling make_a_descriptor(decompose_m=True) which produces
 *   ("m", "k") as top-level coords.  Then we alias "m"->"k_gemm" and "k"->"k_out"
 *   via rename transforms.  If the transforms API lacks rename, we build the
 *   full DAG manually with the correct names.
 *
 *   Simple approach: build the DAG manually mirroring make_a_descriptor but
 *   substituting "m"->"k_gemm" and "k"->"k_out" throughout.
 */
static rocke_tensor_descriptor_t* wgrad_make_x_descriptor(rocke_ir_builder_t* b,
                                                          const rocke_conv_problem_t* p)
{
    /*
     * 2-D DAG (same as make_a_descriptor(decompose_m=True) with name aliases):
     *   unmerge_magic("k_gemm" -> [n, ho, wo], [N, Ho, Wo])
     *   embed(["ho","y"] -> "hi", strides=[sH,dH], offset=-pH, lo=0, hi=Hi)
     *   embed(["wo","x"] -> "wi", strides=[sW,dW], offset=-pW, lo=0, hi=Wi)
     *   unmerge_magic("k_out" -> [y, x, c], [Y, X, C])
     *   pad("y"), pad("x")
     *   naive("X_nhwc", [N, Hi, Wi, C], coords=["n","hi","wi","c"])
     */
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);

    const rocke_transform_t* xforms[10];
    int n_x = 0;

    if(p->is_3d)
    {
        int Do = rocke_conv_problem_do(p);
        /* naive(X_ndhwc) */
        int lengths[5] = {p->N, p->Di, p->Hi, p->Wi, p->C};
        const char* coords[5] = {"n", "di", "hi", "wi", "c"};
        rocke_tensor_descriptor_t* desc
            = rocke_tensor_descriptor_naive(b, "X_ndhwc", lengths, 5, NULL, coords, 5);
        if(desc == NULL)
            return NULL;

        /* unmerge_magic("k_gemm" -> [n,do,ho,wo]) */
        const char* into_m[4] = {"n", "do", "ho", "wo"};
        int dims_m[4] = {p->N, Do, Ho, Wo};
        xforms[n_x] = rocke_unmerge_magic(b, "k_gemm", into_m, 4, dims_m);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        /* embed(["do","z"] -> "di") */
        const char* up_do[2] = {"do", "z"};
        int strides_do[2] = {p->sD, p->dD};
        xforms[n_x] = rocke_embed_bounded(b, up_do, 2, "di", strides_do, -p->pD, 0, p->Di);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        /* embed(["ho","y"] -> "hi") */
        const char* up_ho[2] = {"ho", "y"};
        int strides_ho[2] = {p->sH, p->dH};
        xforms[n_x] = rocke_embed_bounded(b, up_ho, 2, "hi", strides_ho, -p->pH, 0, p->Hi);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        /* embed(["wo","x"] -> "wi") */
        const char* up_wo[2] = {"wo", "x"};
        int strides_wo[2] = {p->sW, p->dW};
        xforms[n_x] = rocke_embed_bounded(b, up_wo, 2, "wi", strides_wo, -p->pW, 0, p->Wi);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        /* unmerge_magic("k_out" -> [z,y,x,c]) */
        const char* into_k[4] = {"z", "y", "x", "c"};
        int dims_k[4] = {p->Z, p->Y, p->X, p->C};
        xforms[n_x] = rocke_unmerge_magic(b, "k_out", into_k, 4, dims_k);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        xforms[n_x] = rocke_pad(b, "z", 0, p->Z);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        xforms[n_x] = rocke_pad(b, "x", 0, p->X);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;

        return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
    }

    /* 2-D */
    int lengths[4] = {p->N, p->Hi, p->Wi, p->C};
    const char* coords[4] = {"n", "hi", "wi", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "X_nhwc", lengths, 4, NULL, coords, 4);
    if(desc == NULL)
        return NULL;

    /* unmerge_magic("k_gemm" -> [n, ho, wo]) */
    const char* into_m[3] = {"n", "ho", "wo"};
    int dims_m[3] = {p->N, Ho, Wo};
    xforms[n_x] = rocke_unmerge_magic(b, "k_gemm", into_m, 3, dims_m);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;

    /* embed(["ho","y"] -> "hi") */
    const char* up_ho[2] = {"ho", "y"};
    int strides_ho[2] = {p->sH, p->dH};
    xforms[n_x] = rocke_embed_bounded(b, up_ho, 2, "hi", strides_ho, -p->pH, 0, p->Hi);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;

    /* embed(["wo","x"] -> "wi") */
    const char* up_wo[2] = {"wo", "x"};
    int strides_wo[2] = {p->sW, p->dW};
    xforms[n_x] = rocke_embed_bounded(b, up_wo, 2, "wi", strides_wo, -p->pW, 0, p->Wi);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;

    /* unmerge_magic("k_out" -> [y, x, c]) */
    const char* into_k[3] = {"y", "x", "c"};
    int dims_k[3] = {p->Y, p->X, p->C};
    xforms[n_x] = rocke_unmerge_magic(b, "k_out", into_k, 3, dims_k);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;

    xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;
    xforms[n_x] = rocke_pad(b, "x", 0, p->X);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;

    return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
}

/*
 * wgrad_make_dw_descriptor:
 *   dW stored KYXC (2-D) / KZYXC (3-D).
 *   The epilogue queries D_desc with coord names ("k_out", "n_wg") where:
 *     "k_out" = output channel index (= K dimension of dW, wg_M = K)
 *     "n_wg"  = filter+channel index (= Y*X*C dimension, wg_N)
 *
 * Matches Python original: dW_desc.offset(b_, k_out=m_val, n_wg=n_val).
 *
 * Layout: naive("dW_kyxc", [K,Y,X,C], coords=["k_out","y","x","c"]).transform(
 *           unmerge_magic("n_wg" -> [y,x,c], [Y,X,C]), pad('y'), pad('x'))
 */
static rocke_tensor_descriptor_t* wgrad_make_dw_descriptor(rocke_ir_builder_t* b,
                                                           const rocke_conv_problem_t* p)
{
    if(p->is_3d)
    {
        /* naive coords: first dim "m" = K (output channels), rest are spatial */
        int lengths[5] = {p->K, p->Z, p->Y, p->X, p->C};
        const char* coords[5] = {"k_out", "z", "y", "x", "c"};
        rocke_tensor_descriptor_t* desc
            = rocke_tensor_descriptor_naive(b, "dW_kzyxc", lengths, 5, NULL, coords, 5);
        if(desc == NULL)
            return NULL;
        const char* into[4] = {"z", "y", "x", "c"};
        int dims[4] = {p->Z, p->Y, p->X, p->C};
        const rocke_transform_t* xforms[4];
        int n_x = 0;
        xforms[n_x] = rocke_unmerge_magic(b, "n_wg", into, 4, dims);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        xforms[n_x] = rocke_pad(b, "z", 0, p->Z);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        xforms[n_x] = rocke_pad(b, "x", 0, p->X);
        if(xforms[n_x] == NULL)
            return NULL;
        n_x++;
        return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
    }

    /* 2-D: naive("dW_kyxc", [K,Y,X,C], coords=["k_out","y","x","c"]) */
    int lengths[4] = {p->K, p->Y, p->X, p->C};
    const char* coords[4] = {"k_out", "y", "x", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dW_kyxc", lengths, 4, NULL, coords, 4);
    if(desc == NULL)
        return NULL;
    const char* into[3] = {"y", "x", "c"};
    int dims[3] = {p->Y, p->X, p->C};
    const rocke_transform_t* xforms[4];
    int n_x = 0;
    xforms[n_x] = rocke_unmerge_magic(b, "n_wg", into, 3, dims);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;
    xforms[n_x] = rocke_pad(b, "y", 0, p->Y);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;
    xforms[n_x] = rocke_pad(b, "x", 0, p->X);
    if(xforms[n_x] == NULL)
        return NULL;
    n_x++;
    return rocke_tensor_descriptor_transform(b, desc, xforms, n_x);
}

// Public descriptor wrappers declared in the header
struct rocke_tensor_descriptor* rocke_wgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* /*dtype*/)
{
    return wgrad_make_dy_descriptor(b, p);
}

struct rocke_tensor_descriptor* rocke_wgrad_make_x_descriptor(rocke_ir_builder_t* b,
                                                              const rocke_conv_problem_t* p,
                                                              const char* /*dtype*/)
{
    return wgrad_make_x_descriptor(b, p);
}

struct rocke_tensor_descriptor* rocke_wgrad_make_dw_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* /*dtype*/)
{
    return wgrad_make_dw_descriptor(b, p);
}

// Wgrad A-descriptor (dY): mirrors Python dy_descriptor closure order.
//
// Python wgrad dy_descriptor (build_implicit_gemm_conv_wgrad.py):
//     k_out   = b_.add(block_m_off_v, row)   <- m_val computed FIRST
//     k_wg_red = b_.add(k_off_capture[0], col) <- k_val computed SECOND
//     return dY_desc.offset(b_, k_wg=k_wg_red, k_out=k_out)
//
// The forward rocke_conv_a_descriptor computes k_val first then m_val, which
// matches the forward Python a_descriptor.  Wgrad is opposite -- m_val first --
// so we need a wgrad-specific closure rather than reusing the forward one.
static rocke_value_t* wgrad_dy_descriptor(rocke_ir_builder_t* b,
                                          rocke_value_t* row,
                                          rocke_value_t* col,
                                          rocke_value_t** out_valid,
                                          void* ctx_user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)ctx_user;
    /* k_out = block_m_off + row (= output channel, m_val) -- computed FIRST */
    rocke_value_t* m_val = rocke_b_add(b, ctx->block_m_off_v, row);
    /* k_wg_red = k_off + col (= output position, k_val) -- computed SECOND */
    rocke_value_t* k_val = rocke_b_add(b, ctx->k_off_capture, col);

    /* Pointwise fast path: flat offset = k_wg_red * kpg + k_out
     * Use pre-emitted constants to match Python's SSA ordering. */
    if(ctx->is_pointwise)
    {
        rocke_value_t* c_K = ctx->ir_c_C_pw; /* kpg (pre-emitted as 1st const) */
        rocke_value_t* c_wgM = ctx->ir_c_M_pw; /* wg_M (3rd) */
        rocke_value_t* c_wgK = ctx->ir_c_wgN_pw; /* wg_K (5th, stored in ir_c_wgN_pw) */
        rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, k_val, c_K), m_val);
        rocke_value_t* kred_ok = rocke_b_cmp_lt(b, k_val, c_wgK);
        rocke_value_t* kout_ok = rocke_b_cmp_lt(b, m_val, c_wgM);
        if(out_valid)
            *out_valid = rocke_b_land(b, kred_ok, kout_ok);
        return off;
    }

    const char* names[2] = {"m", "k"};
    rocke_value_t* vals[2] = {m_val, k_val};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, ctx->A_desc, names, vals, 2, &off, &valid);
    if(out_valid)
        *out_valid = valid;
    return off;
}

// Wgrad x_descriptor: B-operand address closure for the X (input) tensor.
// Python wgrad x_descriptor (build_implicit_gemm_conv_wgrad.py):
//   k_val = b_.add(block_n_off_v, row)    # N_wg: filter+channel position
//   m_val = b_.add(k_off_capture[0], col) # K_wg: output spatial position
//   if p.is_pointwise:
//     off = b_.add(b_.mul(m_val, _c_C_ir), k_val)  # k_wg * C + n_wg
//     return off, land(cmp_lt(m_val, wg_K), cmp_lt(k_val, wg_N))
//   return X_desc.offset(b_, m=m_val, k=k_val)
//
// When is_pointwise, the flat formula is: offset = k_wg * C + n_wg.
// Note the operand order is reversed vs the forward b_descriptor (which computes
// k_out * C + c). wg_N (filter+channel count) is stored in ctx->c_N_pw.
static rocke_value_t* wgrad_x_descriptor(rocke_ir_builder_t* b,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         rocke_value_t** out_valid,
                                         void* ctx_user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)ctx_user;
    /* k_val = block_n_off + row (N_wg: filter+channel position) */
    rocke_value_t* k_val = rocke_b_add(b, ctx->block_n_off_v, row);
    /* m_val = k_off + col (K_wg: output spatial position) */
    rocke_value_t* m_val = rocke_b_add(b, ctx->k_off_capture, col);

    if(ctx->is_pointwise)
    {
        /* Flat: offset = k_wg * cpg + n_wg. Use pre-emitted constants. */
        rocke_value_t* c_C = ctx->ir_c_K_pw; /* cpg (pre-emitted as 2nd const) */
        rocke_value_t* c_wgK = ctx->ir_c_wgN_pw; /* wg_K (5th, stored in ir_c_wgN_pw) */
        rocke_value_t* c_wgN = ctx->ir_always_valid; /* wg_N (4th, stored in ir_always_valid) */
        rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, m_val, c_C), k_val);
        rocke_value_t* kwg_ok = rocke_b_cmp_lt(b, m_val, c_wgK);
        rocke_value_t* nwg_ok = rocke_b_cmp_lt(b, k_val, c_wgN);
        if(out_valid)
            *out_valid = rocke_b_land(b, kwg_ok, nwg_ok);
        return off;
    }

    /* Full descriptor path: X_desc.offset(k_gemm=m_val, k_out=k_val)
     * X descriptor top-level coords: "k_gemm" (K_wg reduction) and "k_out" (N_wg). */
    const char* names[2] = {"k_gemm", "k_out"};
    rocke_value_t* vals[2] = {m_val, k_val};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, ctx->B_desc, names, vals, 2, &off, &valid);
    if(out_valid)
        *out_valid = valid;
    return off;
}

// K-outer descriptor adapters: the K-outer tile is indexed (k, free) while the
// descriptors take (free, k). Swapping the two coordinates keeps the descriptors
// -- and therefore the global addressing -- byte-identical.
static rocke_value_t* wgrad_dy_descriptor_kouter(rocke_ir_builder_t* b,
                                                 rocke_value_t* row,
                                                 rocke_value_t* col,
                                                 rocke_value_t** out_valid,
                                                 void* ctx_user)
{
    return wgrad_dy_descriptor(b, col, row, out_valid, ctx_user);
}

static rocke_value_t* wgrad_x_descriptor_kouter(rocke_ir_builder_t* b,
                                                rocke_value_t* row,
                                                rocke_value_t* col,
                                                rocke_value_t** out_valid,
                                                void* ctx_user)
{
    return wgrad_x_descriptor(b, col, row, out_valid, ctx_user);
}

// Wgrad a_load_override: calls the sync coalesced loader with wgrad_dy_descriptor
// instead of the shared rocke_conv_a_descriptor, preserving Python's
// m_val-first / k_val-second SSA emission order inside the dy_descriptor closure.
static void wgrad_a_load_override(rocke_ir_builder_t* b,
                                  const rocke_implicit_gemm_conv_spec_t* /*spec*/,
                                  rocke_value_t* /*k_off*/, /* already in ctx->k_off_capture */
                                  rocke_value_t* A_dst,
                                  struct rocke_warp_grid* /*grid*/,
                                  void* /*input_cache_context*/,
                                  void* user)
{
    rocke_conv_build_ctx_t* ctx = (rocke_conv_build_ctx_t*)user;
    rocke_coalesced_tile_loader_load(b,
                                     &ctx->a_sync_loader,
                                     ctx->tid,
                                     A_dst,
                                     ctx->lds_k_outer ? wgrad_dy_descriptor_kouter
                                                      : wgrad_dy_descriptor,
                                     ctx,
                                     ctx->a_rsrc,
                                     NULL);
}

// ---------------------------------------------------------------------------
// Additional includes for split-K epilogue and ctx init helpers
// ---------------------------------------------------------------------------
#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.epilogues.h"
#include "rocke/helper_rocke.helpers.grid.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.schedule.h"

// ---------------------------------------------------------------------------
// Two-stage deterministic workspace store epilogue
// ---------------------------------------------------------------------------

/*
 * Emit per-lane f32 atomic-adds into the scratch accumulator for two_stage=True.
 * Mirrors Python _emit_wgrad_workspace_store_epilogue.
 *
 * Every CTA adds its partial into one of its group's R replica slabs:
 *   ws_ptr[(group * R + blockIdx.z % R) * wg_M * wg_N + c_m * wg_N + c_n]
 * so the reduction over split_k is done by the hardware atomics and Stage 2
 * (conv_wgrad_workspace_reduce) only folds the R slabs and casts.  R > 1 exists
 * because a dW-sized scratch is a few dozen cache lines and pointing every CTA
 * at one slab serialises the atomics in L2.
 *
 * The scratch is always f32 regardless of dtype_d; the dtype conversion happens
 * in Stage 2.  The caller must zero the scratch first -- these are adds.
 * Out-of-bounds elements are guarded by scf_if rather than a sentinel offset
 * (an atomic to a sentinel would compute a real address and fault).
 */
static void wgrad_emit_workspace_store_epilogue(rocke_ir_builder_t* b,
                                                const rocke_conv_build_ctx_t* ctx,
                                                const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                                rocke_value_t* ws_ptr,
                                                int wg_M /*unused: bounds are kernargs*/,
                                                int wg_N /*unused: bounds are kernargs*/)
{
    (void)wg_M; /* bounds come from the ctx kernargs now */
    (void)wg_N;
    /* Mirrors Python _emit_wgrad_workspace_store_epilogue exactly.
     * Ordering of IR operations must match Python's line-by-line. */
    const rocke_mfma_atom_t* atom = ctx->atom;
    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;
    int c_per_lane = ctx->c_per_lane;

    /* 1. wg_M_v, wg_N_v -- Python: wg_M_v = b.const_i32(wg_M) */
    /* AOT: the GEMM extents are kernargs, not folded constants. */
    rocke_value_t* wg_M_v = ctx->p_wg_M;
    rocke_value_t* wg_N_v = ctx->p_wg_N;
    rocke_value_t* slab_v = rocke_b_mul(b, wg_M_v, wg_N_v);

    /* 2. slab_off -- Python: slab index = group * R + (block_id_z % R), elided
     *    entirely for the ungrouped R == 1 case so that path keeps its SSA
     *    numbering. */
    /* Every sub-expression is sequenced through a local: C++ leaves the
     * evaluation order of call arguments unspecified, and the builder hands
     * out SSA ids in call order, so an inline nest renumbers the module
     * against Python (which evaluates left-to-right). The order below is
     * Python's, statement for statement. */
    /* R is a build-time knob; the slab size is the runtime wg_M * wg_N. */
    const int reps = spec->ws_replicas;
    rocke_value_t* group_v = ctx->group_idx;
    rocke_value_t* slab_idx = NULL;
    if(group_v != NULL && reps > 1)
    {
        rocke_value_t* c_reps_a = rocke_b_const_i32(b, reps);
        rocke_value_t* grp_term = rocke_b_mul(b, group_v, c_reps_a);
        rocke_value_t* z = rocke_b_block_id_z(b);
        rocke_value_t* c_reps_b = rocke_b_const_i32(b, reps);
        rocke_value_t* rep_term = rocke_b_mod(b, z, c_reps_b);
        slab_idx = rocke_b_add(b, grp_term, rep_term);
    }
    else if(group_v != NULL)
    {
        slab_idx = group_v;
    }
    else if(reps > 1)
    {
        rocke_value_t* z = rocke_b_block_id_z(b);
        rocke_value_t* c_reps = rocke_b_const_i32(b, reps);
        slab_idx = rocke_b_mod(b, z, c_reps);
    }
    rocke_value_t* slab_off = NULL;
    if(slab_idx != NULL)
        slab_off = rocke_b_mul(b, slab_idx, slab_v);

    /* 3. Per-warp M/N offsets -- Python: warp_m_off = b.mul(warp_m_idx, ...) */
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    rocke_value_t* block_warp_m_off = rocke_b_add(b, ctx->block_m_off_v, warp_m_off);
    rocke_value_t* block_warp_n_off = rocke_b_add(b, ctx->block_n_off_v, warp_n_off);

    /* 4. c_warp_params: compile-time lookup, no IR ops -- Python: c_warp_params(atom) */
    int m0, m_lane, m1, n_lane;
    if(rocke_b_c_warp_params(b, atom, &m0, &m_lane, &m1, &n_lane) != ROCKE_OK)
        return;

    /* 5. c_dist -- Python: c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding) */
    rocke_tile_distribution_encoding_t* enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
        return;
    rocke_tile_distribution_t* c_dist = rocke_make_static_tile_distribution(b, enc);
    if(c_dist == NULL)
        return;

    /* 6. c_nlane, n_in_atom, m_blk -- Python ordering */
    rocke_value_t* c_nlane_v = rocke_b_const_i32(b, n_lane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, ctx->lane, c_nlane_v);
    rocke_value_t* m_blk = rocke_b_div(b, ctx->lane, c_nlane_v);
    rocke_value_t* p_lane_subs[2] = {m_blk, n_in_atom};
    rocke_value_t* const* p_arr[1] = {p_lane_subs};
    int p_counts[1] = {2};

    /* 7. Pre-compute per-slot (row, col) within the atom */
    rocke_value_t* slot_rows[ROCKE_CONV_MAX_ACCS * 4];
    rocke_value_t* slot_cols[ROCKE_CONV_MAX_ACCS * 4];
    for(int i = 0; i < c_per_lane; ++i)
    {
        rocke_value_t* y0 = rocke_b_const_i32(b, i / m1);
        rocke_value_t* y1 = rocke_b_const_i32(b, i % m1);
        rocke_value_t* ys[2] = {y0, y1};
        rocke_value_t* out_x[2] = {NULL, NULL};
        rocke_tile_distribution_calculate_x(b, c_dist, ys, 2, p_arr, p_counts, 1, out_x, 2);
        slot_rows[i] = out_x[0];
        slot_cols[i] = out_x[1];
    }

    /* 8. Main mi/ni/i loop */
    int flat = 0;
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        rocke_value_t* atom_m_base
            = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));

        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = ctx->final_accs[flat++];
            rocke_value_t* atom_n_base
                = rocke_b_add(b, block_warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < c_per_lane; ++i)
            {
                rocke_value_t* c_m = rocke_b_add(b, atom_m_base, slot_rows[i]);
                rocke_value_t* c_n = rocke_b_add(b, atom_n_base, slot_cols[i]);
                rocke_value_t* val_f32 = rocke_b_vec_extract(b, acc, i);

                /* OOB guard: scf_if instead of sentinel -- an atomic to a
                 * sentinel offset would compute a real address and fault. */
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, wg_M_v);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, wg_N_v);
                rocke_value_t* in_bounds = rocke_b_land(b, m_ok, n_ok);

                rocke_if_t if_op = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, if_op.then_region);
                {
                    rocke_value_t* ws_off = rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_v), c_n);
                    if(slab_off != NULL)
                        ws_off = rocke_b_add(b, slab_off, ws_off);
                    rocke_b_global_atomic_add(b, ws_ptr, ws_off, val_f32, NULL);
                }
                rocke_b_region_leave(b);
            }
        }
    }
}

// Split-K atomic epilogue for wgrad
// ---------------------------------------------------------------------------

/*
 * Emit per-lane atomic-adds into dW for split_k > 1.
 * Mirrors Python _emit_wgrad_split_k_epilogue for fp32, bf16, and fp16 outputs.
 *
 * fp32: scalar global_atomic_add per slot (exact, monotonic).
 * fp16/bf16: _emit_single_packed_atomic per slot -- packs into <2 x dtype> with
 *   the correct even/odd column placement, then global_atomic_add_pk_f16/bf16.
 *   Note: each CTA rounds its partial to output precision before the atomic add,
 *   so accumulation error grows with split_k.  For large split_k consider using
 *   an fp32 workspace and downcasting after the kernel completes.
 */
static void wgrad_emit_split_k_epilogue_f32(rocke_ir_builder_t* b,
                                            const rocke_conv_build_ctx_t* ctx,
                                            const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                            rocke_value_t* dw_ptr,
                                            int wg_M /*unused: bounds are kernargs*/,
                                            int wg_N /*unused: bounds are kernargs*/)
{
    (void)wg_M; /* bounds come from the ctx kernargs now */
    (void)wg_N;
    const rocke_mfma_atom_t* atom = ctx->atom;
    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;
    int c_per_lane = ctx->c_per_lane;
    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
    bool is_fp32 = (strcmp(dtype_d, "fp32") == 0);
    bool is_bf16 = (strcmp(dtype_d, "bf16") == 0);
    (void)is_bf16; /* used conditionally below */

    /* Python emission order: warp offsets first, then c_warp_params / c_dist. */
    rocke_value_t* warp_m_off_v
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off_v
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    rocke_value_t* block_warp_m_off = rocke_b_add(b, ctx->block_m_off_v, warp_m_off_v);
    rocke_value_t* block_warp_n_off = rocke_b_add(b, ctx->block_n_off_v, warp_n_off_v);

    int m0, m_lane, m1, n_lane;
    if(rocke_b_c_warp_params(b, atom, &m0, &m_lane, &m1, &n_lane) != ROCKE_OK)
        return;

    rocke_tile_distribution_encoding_t* enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
        return;
    rocke_tile_distribution_t* c_dist = rocke_make_static_tile_distribution(b, enc);
    if(c_dist == NULL)
        return;

    rocke_value_t* c_nlane_v = rocke_b_const_i32(b, n_lane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, ctx->lane, c_nlane_v);
    rocke_value_t* m_blk = rocke_b_div(b, ctx->lane, c_nlane_v);
    rocke_value_t* p_lane_subs[2] = {m_blk, n_in_atom};
    rocke_value_t* const* p_arr[1] = {p_lane_subs};
    int p_counts[1] = {2};

    /* Pre-compute per-slot (row, col) within the atom.
     * Python creates wg_M_v / wg_N_v AFTER this decode loop, so defer them.
     * Python: kc_m1 = c_warp_params(atom)[2] = m1 (index 2, not m0=index 0).
     * ys = [i // kc_m1, i % kc_m1] -> [i // m1, i % m1]. */
    rocke_value_t* slot_rows[ROCKE_CONV_MAX_ACCS * 4]; /* generous */
    rocke_value_t* slot_cols[ROCKE_CONV_MAX_ACCS * 4];
    for(int i = 0; i < c_per_lane; ++i)
    {
        rocke_value_t* y0 = rocke_b_const_i32(b, i / m1); /* i // kc_m1 */
        rocke_value_t* y1 = rocke_b_const_i32(b, i % m1); /* i % kc_m1 */
        rocke_value_t* ys[2] = {y0, y1};
        rocke_value_t* out_x[2] = {NULL, NULL};
        rocke_tile_distribution_calculate_x(b, c_dist, ys, 2, p_arr, p_counts, 1, out_x, 2);
        slot_rows[i] = out_x[0];
        slot_cols[i] = out_x[1];
    }

    /* Python creates wg_M_v / wg_N_v after the slot decode loop. */
    /* AOT: the GEMM extents are kernargs, not folded constants. */
    rocke_value_t* wg_M_v = ctx->p_wg_M;
    rocke_value_t* wg_N_v = ctx->p_wg_N;

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        /* atom_m_base = add(block_warp_m_off, mi * warp_tile_m) */
        rocke_value_t* atom_m_base
            = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = ctx->final_accs[flat++];
            rocke_value_t* atom_n_base
                = rocke_b_add(b, block_warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < c_per_lane; ++i)
            {
                rocke_value_t* c_m = rocke_b_add(b, atom_m_base, slot_rows[i]);
                rocke_value_t* c_n = rocke_b_add(b, atom_n_base, slot_cols[i]);
                /* Python: val = vec_extract evaluated before _emit_scalar_atomic body */
                rocke_value_t* val_f32 = rocke_b_vec_extract(b, acc, i);

                if(is_fp32)
                {
                    /* _emit_scalar_atomic fp32 path:
                     *   c_off = add(mul(c_m, wg_N_v), c_n)
                     *   ok    = land(cmp_lt(c_m, wg_M_v), cmp_lt(c_n, wg_N_v))
                     *   scf_if(ok): global_atomic_add(ptr, c_off, val) */
                    rocke_value_t* c_off = rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_v), c_n);
                    rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, wg_M_v);
                    rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, wg_N_v);
                    rocke_value_t* ok = rocke_b_land(b, m_ok, n_ok);
                    rocke_if_t if_op = rocke_b_scf_if(b, ok);
                    rocke_b_region_enter(b, if_op.then_region);
                    rocke_b_global_atomic_add(b, dw_ptr, c_off, val_f32, NULL);
                    rocke_b_region_leave(b);
                }
                else
                {
                    /* _emit_single_packed_atomic for fp16/bf16:
                     *   zero = trunc_f32_to_dtype(const_f32(0.0))
                     *   val  = trunc_f32_to_dtype(val_f32)
                     *   m_ok = cmp_lt(c_m, wg_M_v); n_ok = cmp_lt(c_n, wg_N_v)
                     *   scf_if(land(m_ok, n_ok)):
                     *     c_n_is_odd = mod(c_n, 2)
                     *     is_odd = cmp_ne(c_n_is_odd, 0)
                     *     c_n_even  = sub(c_n, c_n_is_odd)
                     *     c_off_even = add(mul(c_m, wg_N_v), c_n_even)
                     *     v_even = select(is_odd, zero, val)
                     *     v_odd  = select(is_odd, val, zero)
                     *     vec    = vec_pack([v_even, v_odd])
                     *     global_atomic_add_pk_f16/bf16(ptr, c_off_even, vec) */
                    rocke_value_t* zero
                        = (is_bf16) ? rocke_b_trunc_f32_to_bf16(b, rocke_b_const_f32(b, 0.0))
                                    : rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));
                    rocke_value_t* val = (is_bf16) ? rocke_b_trunc_f32_to_bf16(b, val_f32)
                                                   : rocke_b_trunc_f32_to_f16(b, val_f32);
                    rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, wg_M_v);
                    rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, wg_N_v);
                    rocke_value_t* ok = rocke_b_land(b, m_ok, n_ok);
                    rocke_if_t if_op = rocke_b_scf_if(b, ok);
                    rocke_b_region_enter(b, if_op.then_region);
                    {
                        rocke_value_t* c2 = rocke_b_const_i32(b, 2);
                        rocke_value_t* c_n_is_odd = rocke_b_mod(b, c_n, c2);
                        rocke_value_t* is_odd
                            = rocke_b_cmp_ne(b, c_n_is_odd, rocke_b_const_i32(b, 0));
                        rocke_value_t* c_n_even = rocke_b_sub(b, c_n, c_n_is_odd);
                        rocke_value_t* c_off_even
                            = rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_v), c_n_even);
                        rocke_value_t* v_even = rocke_b_select(b, is_odd, zero, val);
                        rocke_value_t* v_odd = rocke_b_select(b, is_odd, val, zero);
                        rocke_value_t* elems[2] = {v_even, v_odd};
                        rocke_value_t* vec = rocke_b_vec_pack(b, elems, 2, val->type);
                        if(is_bf16)
                            rocke_b_global_atomic_add_pk_bf16(b, dw_ptr, c_off_even, vec, NULL);
                        else
                            rocke_b_global_atomic_add_pk_f16(b, dw_ptr, c_off_even, vec, NULL);
                    }
                    rocke_b_region_leave(b);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wgrad split-K cshuffle epilogue
// Mirrors Python _emit_wgrad_split_k_cshuffle_epilogue:
//   CShuffleEpilogue.from_grid(atom, grid, max_store_vec=vec_c)
//       .atomic_store(b, accs, dw_ptr=dW, wg_N=wg_N_v, bounds=(wg_M_v, wg_N_v))
// vec_c uses split_k=1 semantics (same as the non-atomic cshuffle path) because
// the cshuffle atomic path is not contraindicated by wide store_vec.
// groups > 1 is not supported for C++ wgrad (rejected by the validator).
// ---------------------------------------------------------------------------
static void wgrad_emit_split_k_cshuffle_epilogue(rocke_ir_builder_t* b,
                                                 const rocke_conv_build_ctx_t* ctx,
                                                 const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                                 rocke_value_t* dw_ptr,
                                                 int wg_M,
                                                 int wg_N)
{
    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
    bool is_fp32_vec = (strcmp(dtype_d, "fp32") == 0);
    (void)wg_M; /* bounds come from the ctx kernargs now */
    (void)wg_N;
    int C = spec->problem.C;
    int vec_c;

    /* default_vector_sizes(..., split_k=1): widest vec that divides C.
     * fp32: cap at 4; fp16/bf16: cap at 8. */
    if(is_fp32_vec)
    {
        if(C % 4 == 0)
            vec_c = 4;
        else if(C % 2 == 0)
            vec_c = 2;
        else
            vec_c = 1;
    }
    else
    {
        if(C % 8 == 0)
            vec_c = 8;
        else if(C % 4 == 0)
            vec_c = 4;
        else if(C % 2 == 0)
            vec_c = 2;
        else
            vec_c = 1;
    }

    rocke_cshuffle_epilogue_t cepi
        = rocke_cshuffle_epilogue_from_grid(ctx->atom, &ctx->grid, vec_c);
    cepi.out_dtype = dtype_d;

    rocke_cshuffle_epilogue_atomic_store(b,
                                         &cepi,
                                         ctx->final_accs,
                                         ctx->num_final_accs,
                                         dw_ptr,
                                         ctx->p_wg_N,
                                         ctx->p_wg_M,
                                         ctx->p_wg_N);
}

// ---------------------------------------------------------------------------
// Wgrad direct epilogue (split_k == 1, default path)
// Mirrors Python _emit_wgrad_direct_epilogue:
//   DirectEpilogue(atom, grid, out_dtype).store(b, accs, addr_fn=dw_addr,
//       d_rsrc=dw_rsrc, bounds=(wg_M, wg_N))
// where dw_addr queries dW_desc with ("k_out"=m_val, "n_wg"=n_val).
// ---------------------------------------------------------------------------

struct WgradDwAddrCtx
{
    rocke_tensor_descriptor_t* dW_desc;
};

static rocke_value_t* wgrad_dw_addr(rocke_ir_builder_t* b,
                                    rocke_value_t* m_global,
                                    rocke_value_t* n_global,
                                    rocke_value_t** out_valid,
                                    void* user)
{
    WgradDwAddrCtx* wc = static_cast<WgradDwAddrCtx*>(user);
    /* Python: dW_desc.offset(b_, k_out=m_val, n_wg=n_val)
     * dW_desc top-level coords: ("k_out" = output channel, "n_wg" = filter+chan).
     * The epilogue calls addr_fn(m_global=output_channel, n_global=filter+channel). */
    const char* names[2] = {"k_out", "n_wg"};
    rocke_value_t* vals[2] = {m_global, n_global};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, wc->dW_desc, names, vals, 2, &off, &valid);
    if(out_valid)
        *out_valid = valid;
    return off;
}

/* Pointwise dW addr: flat offset = k_out * wg_N + n_wg, always valid.
 * Python: def dw_addr(b_, m_val, n_val): return b_.add(b_.mul(m_val, _c_N), n_val), 1 */
static rocke_value_t* wgrad_dw_addr_pointwise(rocke_ir_builder_t* b,
                                              rocke_value_t* m_global,
                                              rocke_value_t* n_global,
                                              rocke_value_t** out_valid,
                                              void* user)
{
    rocke_value_t* c_N = (rocke_value_t*)user;
    rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, m_global, c_N), n_global);
    if(out_valid)
        *out_valid = rocke_b_const_i32(b, 1);
    return off;
}

/* dW descriptor, built at the point of use.
 *
 * Python builds it inside the epilogue rather than up front, so its stride
 * constant lands after the K-loop; building it earlier would shift every SSA
 * id in between and break byte-identity. */
static rocke_tensor_descriptor_t* wgrad_build_dw_descriptor(rocke_ir_builder_t* b,
                                                            const rocke_conv_build_ctx_t* ctx)
{
    rocke_conv_dyn_desc_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.name = ctx->p->is_3d ? "dW_kzyxc" : "dW_kyxc";
    opts.stride_n = ctx->p_dW_stride_k;
    opts.stride_di = ctx->p_dW_stride_z;
    opts.stride_hi = ctx->p_dW_stride_y;
    opts.stride_wi = ctx->p_dW_stride_x;
    opts.channel_y.mult = ctx->p_magic_n_Y_mult;
    opts.channel_y.shift = ctx->p_magic_n_Y_shift;
    opts.channel_x.mult = ctx->p_magic_n_X_mult;
    opts.channel_x.shift = ctx->p_magic_n_X_shift;
    opts.channel_c.mult = ctx->p_magic_n_cpg_mult;
    opts.channel_c.shift = ctx->p_magic_n_cpg_shift;
    return (rocke_tensor_descriptor_t*)rocke_conv_make_b_descriptor_dynamic_opts(
        b, ctx, "n_wg", &opts);
}

static void wgrad_emit_direct_epilogue(rocke_ir_builder_t* b,
                                       const rocke_conv_build_ctx_t* ctx,
                                       const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                       rocke_value_t* dw_rsrc,
                                       int wg_M,
                                       int wg_N)
{
    /* AOT: the store bounds are the p_wg_M / p_wg_N kernargs. */
    (void)wg_M;
    (void)wg_N;
    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";

    if(!ctx->is_wmma)
    {
        /* MFMA path: use DirectEpilogue.store with wgrad addr fn */
        rocke_direct_epilogue_t epi;
        epi.atom = ctx->atom;
        epi.grid = ctx->grid;
        epi.out_dtype = dtype_d;

        if(ctx->is_pointwise)
        {
            /* Pointwise: Python _emit_wgrad_direct_epilogue emits _c_N FIRST:
             *   _c_N = b.const_i32(wg_N)                      <- first
             *   bounds = (b.const_i32(wg_M), b.const_i32(wg_N)) <- second/third */
            rocke_value_t* c_N = ctx->p_wg_N;
            rocke_value_t* bound_m = ctx->p_wg_M;
            rocke_value_t* bound_n = ctx->p_wg_N;
            rocke_direct_epilogue_store(b,
                                        &epi,
                                        ctx->final_accs,
                                        ctx->num_final_accs,
                                        wgrad_dw_addr_pointwise,
                                        (void*)c_N,
                                        dw_rsrc,
                                        bound_m,
                                        bound_n,
                                        false);
        }
        else
        {
            WgradDwAddrCtx addr_ctx;
            /* Built here, where Python builds it: its stride constant must
             * land at this point in the SSA sequence. */
            addr_ctx.dW_desc = wgrad_build_dw_descriptor(b, ctx);
            rocke_value_t* bound_m = ctx->p_wg_M;
            rocke_value_t* bound_n = ctx->p_wg_N;
            rocke_direct_epilogue_store(b,
                                        &epi,
                                        ctx->final_accs,
                                        ctx->num_final_accs,
                                        wgrad_dw_addr,
                                        &addr_ctx,
                                        dw_rsrc,
                                        bound_m,
                                        bound_n,
                                        false);
        }
        return;
    }

    /* WMMA path: mirrors Python _emit_wgrad_direct_epilogue_wmma.
     * op.c_layout().coord(b, lane, i) gives per-slot (row_off, col_off);
     * dW_desc.offset(k_out=m_val, n_wg=n_val) gives the byte offset. */
    const rocke_mmaop_t* op = ctx->op;
    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    /* Python builds the dW descriptor after the warp offsets on the WMMA
     * path, so its stride constant lands here. */
    rocke_tensor_descriptor_t* dW_desc
        = ctx->is_pointwise ? NULL : wgrad_build_dw_descriptor(b, ctx);

    /* AOT: bounds are the p_wg_M / p_wg_N kernargs. */
    rocke_value_t* c_M = ctx->p_wg_M;
    rocke_value_t* c_N = ctx->p_wg_N;

    bool _fp32_out = (strcmp(dtype_d, "fp32") == 0);
    bool _bf16_out = (strcmp(dtype_d, "bf16") == 0);
    int elem_bytes = _fp32_out ? 4 : 2;

    bool _is_pw = ctx->is_pointwise;
    rocke_value_t* c_wgN_wmma = _is_pw ? rocke_b_const_i32(b, wg_N) : NULL;

    const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);
    rocke_value_t* c0 = ctx->c0;

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = ctx->final_accs[flat++];

            rocke_value_t* m_inner = rocke_b_add(b, ctx->block_m_off_v, warp_m_off);
            rocke_value_t* m_const = rocke_b_const_i32(b, mi * spec->warp_tile_m);
            rocke_value_t* atom_m_off = rocke_b_add(b, m_inner, m_const);

            rocke_value_t* n_inner = rocke_b_add(b, ctx->block_n_off_v, warp_n_off);
            rocke_value_t* n_const = rocke_b_const_i32(b, ni * spec->warp_tile_n);
            rocke_value_t* atom_n_off = rocke_b_add(b, n_inner, n_const);

            for(int i = 0; i < op->c_frag_len; ++i)
            {
                rocke_value_t* row_off = NULL;
                rocke_value_t* col_off = NULL;
                rocke_arch_layout_map_coord(c_map, b, ctx->lane, i, &row_off, &col_off);

                rocke_value_t* m_val = rocke_b_add(b, atom_m_off, row_off);
                rocke_value_t* n_val = rocke_b_add(b, atom_n_off, col_off);
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, c_M);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_val, c_N);
                rocke_value_t* ok = rocke_b_land(b, m_ok, n_ok);
                rocke_value_t* v_f32 = rocke_b_vec_extract(b, acc, i);

                /* dW_desc.offset(k_out=m_val, n_wg=n_val)
                 * Pointwise: flat offset = m_val * wg_N + n_val */
                rocke_value_t* d_off_elems = NULL;
                if(_is_pw)
                {
                    d_off_elems = rocke_b_add(b, rocke_b_mul(b, m_val, c_wgN_wmma), n_val);
                }
                else
                {
                    const char* names[2] = {"k_out", "n_wg"};
                    rocke_value_t* vals[2] = {m_val, n_val};
                    rocke_value_t* valid = NULL;
                    rocke_transforms_descriptor_offset(
                        b, dW_desc, names, vals, 2, &d_off_elems, &valid);
                }
                rocke_value_t* d_off_bytes
                    = rocke_b_mul(b, d_off_elems, rocke_b_const_i32(b, elem_bytes));
                rocke_value_t* safe_off = rocke_b_select(
                    b, ok, d_off_bytes, rocke_b_const_i32(b, (int64_t)((1u << 31) - 1u)));
                if(_fp32_out)
                    rocke_b_buffer_store_f32(b, dw_rsrc, safe_off, c0, v_f32);
                else if(_bf16_out)
                    rocke_b_buffer_store_bf16(
                        b, dw_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                else
                    rocke_b_buffer_store_f16(
                        b, dw_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Wgrad ctx init -- mirrors rocke_conv_build_ctx_init but uses wgrad param names
// and sets the wg_K K-loop bound (not K_gemm from an adapter problem).
// ---------------------------------------------------------------------------

static bool wgrad_build_ctx_init(rocke_conv_build_ctx_t* ctx,
                                 rocke_ir_builder_t* b,
                                 const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                 const char* arch,
                                 int wg_K, /* rocke_wgrad_conv_spec_wg_K(spec) */
                                 rocke_value_t* ks_param, /* slice width  (always) */
                                 rocke_value_t* ks_count_param) /* slice count (always) */
{
    if(ctx == NULL || b == NULL || spec == NULL)
        return false;

    /* The caller has already emitted the AOT kernarg block into ctx, so this
     * must not clear it -- the params are ctx state, not scratch. Only the
     * fields this function owns are (re)initialised below. */
    ctx->b = b;
    ctx->arch = arch;

    /* Set up wgrad-specific overrides.  The only override needed is
     * a_load_override, which calls the sync loader with wgrad_dy_descriptor
     * (m_val-first SSA order) instead of the shared rocke_conv_a_descriptor
     * (k_val-first).  The async path goes through rocke_conv_emit_load_phase
     * which calls rocke_conv_a_descriptor directly for the async slot; wgrad
     * supplies the wgrad dY descriptor through ctx->a_descriptor_fn. */
    {
        rocke_conv_build_overrides_t* ov_ptr = (rocke_conv_build_overrides_t*)rocke_arena_alloc(
            &b->arena, sizeof(rocke_conv_build_overrides_t));
        if(!ov_ptr)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: arena alloc for overrides failed");
            return false;
        }
        memset(ov_ptr, 0, sizeof(*ov_ptr));
        ov_ptr->a_load_override = wgrad_a_load_override;
        ov_ptr->user = ctx; /* ctx pointer re-used as user; populated after this block */
        ctx->ov = ov_ptr;
    }

    /* Arena-allocate the stub problem so it lives for the builder's lifetime.
     *
     * The stub is queried by the epilogue phase for bounds and D descriptor:
     *   p.M        = N*Ho*Wo  -> must equal wg_M
     *   p.N_gemm   = K        -> must equal wg_N
     *   p.K_gemm   = Y*X*C    -> must equal wg_K  (used by kloop_unroll only)
     *
     * Use N=wg_M, Hi=Wi=1 (so Ho=Wo=1), C=wg_K, K=wg_N, Y=X=1:
     *   K_gemm = 1*1*wg_K = wg_K  (ok)   M = wg_M*1*1 = wg_M  (ok)   N_gemm = wg_N  (ok)
     *   Ho = (1 + 0 - 1*(1-1) - 1)/1 + 1 = 1  (ok)  (not negative!) */
    rocke_conv_problem_t* stub_p
        = (rocke_conv_problem_t*)rocke_arena_alloc(&b->arena, sizeof(rocke_conv_problem_t));
    if(!stub_p)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: arena alloc failed");
        return false;
    }
    {
        int wg_M_val = spec->problem.K; /* groups==1 */
        int wg_N_val = rocke_wgrad_conv_spec_wg_N(spec);
        /* N=wg_M, Hi=1, Wi=1, C=wg_K, K=wg_N, Y=1, X=1 -> Ho=Wo=1, all positive */
        *stub_p = rocke_conv_problem_default(wg_M_val, 1, 1, wg_K, wg_N_val, 1, 1);
    }
    ctx->p = stub_p;

    /* Pointwise fast path: mirrors Python `if p.is_pointwise:` where p is the
     * real conv problem (not the stub).  The stub always has Y=X=1 so we must
     * set is_pointwise from the real problem here and store wgrad-specific
     * constants used by wgrad_dy_descriptor, wgrad_x_descriptor, and epilogues. */
    {
        bool pw = rocke_conv_problem_is_pointwise(&spec->problem);
        ctx->is_pointwise = pw;
        if(pw)
        {
            /* Python:
             *   _c_K_ir   = b.const_i32(p.kpg)  -> kpg   (output channels per group)
             *   _c_C_ir   = b.const_i32(p.cpg)  -> cpg   (input channels per group)
             *   _c_wgM_ir = b.const_i32(wg_M)   -> wg_M  (= kpg for groups==1)
             *   _c_wgN_ir = b.const_i32(wg_N)   -> wg_N  (= Y*X*C)
             *   _c_wgK_ir = b.const_i32(wg_K)   -> wg_K  (= N*Ho*Wo reduction) */
            ctx->c_K_pw = rocke_conv_problem_kpg(&spec->problem); /* kpg */
            ctx->c_C_pw = rocke_conv_problem_cpg(&spec->problem); /* cpg */
            ctx->c_M_pw = spec->problem.K; /* wg_M (groups==1) */
            ctx->c_wgK_pw = wg_K; /* wg_K (reduction) */
            ctx->c_wgN_pw = rocke_wgrad_conv_spec_wg_N(spec); /* wg_N */
            /* Use wgrad_x_descriptor for B (X) tile: different pointwise formula. */
            ctx->b_descriptor_fn
                = spec->lds_k_outer ? wgrad_x_descriptor_kouter : wgrad_x_descriptor;
        }
        else
        {
            ctx->c_K_pw = 0;
            ctx->c_C_pw = 0;
            ctx->c_M_pw = 0;
            ctx->c_wgK_pw = 0;
            ctx->c_wgN_pw = 0;
            /* always use the wgrad X descriptor (swapped coords when K-outer) */
            ctx->b_descriptor_fn
                = spec->lds_k_outer ? wgrad_x_descriptor_kouter : wgrad_x_descriptor;
        }
    }

    /* waves_per_eu */
    if(spec->has_waves_per_eu && b->kernel != NULL)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);

    /* Resolve op + atom */
    {
        /* Build a dummy forward conv spec to route through rocke_conv_resolve_op. */
        rocke_implicit_gemm_conv_spec_t fwd = rocke_implicit_gemm_conv_spec_default();
        fwd.problem = *ctx->p;
        fwd.tile_m = spec->tile_m;
        fwd.tile_n = spec->tile_n;
        fwd.tile_k = spec->tile_k;
        fwd.warp_m = spec->warp_m;
        fwd.warp_n = spec->warp_n;
        fwd.warp_tile_m = spec->warp_tile_m;
        fwd.warp_tile_n = spec->warp_tile_n;
        fwd.warp_tile_k = spec->warp_tile_k;
        fwd.wave_size = spec->wave_size;
        fwd.pipeline = spec->pipeline;
        fwd.epilogue = spec->epilogue;
        fwd.dtype_a = spec->dtype_a;
        fwd.dtype_b = spec->dtype_b;
        fwd.dtype_d = spec->dtype_d;
        /* The shared load phase and k-loop drivers branch on the forward spec,
         * so these must be carried across or the async / unrolled paths are
         * silently unreachable from wgrad. */
        fwd.async_dma = spec->async_dma;
        fwd.unroll_k = spec->unroll_k;
        /* A temporary spec we keep alive for the duration; store pointer */
        rocke_implicit_gemm_conv_spec_t* tmp_fwd_spec
            = (rocke_implicit_gemm_conv_spec_t*)rocke_arena_alloc(
                &b->arena, sizeof(rocke_implicit_gemm_conv_spec_t));
        if(!tmp_fwd_spec)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: arena alloc");
            return false;
        }
        *tmp_fwd_spec = fwd;
        ctx->spec = tmp_fwd_spec;
        ctx->op = rocke_conv_resolve_op(b, tmp_fwd_spec, arch);
        if(ctx->op == NULL)
            return false;
        ctx->is_wmma = (ctx->op->family != NULL && strcmp(ctx->op->family, "wmma") == 0);
        ctx->atom = ctx->is_wmma ? NULL
                                 : rocke_mfma_atom(spec->dtype_a ? spec->dtype_a : "fp16",
                                                   spec->warp_tile_m,
                                                   spec->warp_tile_n,
                                                   spec->warp_tile_k);
    }
    ctx->a_per_lane = ctx->op->a_frag_len;
    ctx->b_per_lane = ctx->op->b_frag_len;
    ctx->c_per_lane = ctx->op->c_frag_len;

    /* Block tile dims */
    ctx->block_m = spec->tile_m;
    ctx->block_n = spec->tile_n;
    ctx->block_k = spec->tile_k;

    /* WarpGrid.bind -- emit the same SSA in the same order as the forward conv */
    ctx->grid.tile_m = ctx->block_m;
    ctx->grid.tile_n = ctx->block_n;
    ctx->grid.tile_k = ctx->block_k;
    ctx->grid.warp_m = spec->warp_m;
    ctx->grid.warp_n = spec->warp_n;
    ctx->grid.warp_k = 1;
    ctx->grid.warp_tile_m = spec->warp_tile_m;
    ctx->grid.warp_tile_n = spec->warp_tile_n;
    ctx->grid.warp_tile_k = spec->warp_tile_k;
    ctx->grid.wave_size = spec->wave_size;

    if(b->kernel != NULL)
        rocke_attr_set_int(
            b, &b->kernel->attrs, "max_workgroup_size", rocke_warp_grid_block_size(&ctx->grid));

    rocke_value_t* wave = rocke_b_const_i32(b, spec->wave_size);
    rocke_value_t* c_warps_n = rocke_b_const_i32(b, spec->warp_n);
    rocke_value_t* c_warps_nm = rocke_b_const_i32(b, spec->warp_n * spec->warp_m);
    rocke_value_t* c_tile_m = rocke_b_const_i32(b, ctx->block_m);
    rocke_value_t* c_tile_n = rocke_b_const_i32(b, ctx->block_n);
    rocke_value_t* c_tile_k = rocke_b_const_i32(b, ctx->block_k);
    (void)c_warps_nm;
    (void)c_tile_k;

    rocke_value_t* tid_v = rocke_b_thread_id_x(b);
    rocke_value_t* lane_v = rocke_b_mod(b, tid_v, wave);
    rocke_value_t* warp_id_v = rocke_b_div(b, tid_v, wave);
    rocke_value_t* warp_m_v = rocke_b_div(b, warp_id_v, c_warps_n);
    rocke_value_t* warp_n_v = rocke_b_mod(b, warp_id_v, c_warps_n);
    rocke_value_t* warp_k_v = rocke_b_const_i32(b, 0);
    rocke_value_t* bm_off = rocke_b_mul(b, rocke_b_block_id_y(b), c_tile_m);
    rocke_value_t* bn_off = rocke_b_mul(b, rocke_b_block_id_x(b), c_tile_n);
    rocke_value_t* bk_off = rocke_b_const_i32(b, 0);

    ctx->grid.tid = tid_v;
    ctx->grid.lane = lane_v;
    ctx->grid.warp_id = warp_id_v;
    ctx->grid.warp_m_idx = warp_m_v;
    ctx->grid.warp_n_idx = warp_n_v;
    ctx->grid.warp_k_idx = warp_k_v;
    ctx->grid.block_m_off = bm_off;
    ctx->grid.block_n_off = bn_off;
    ctx->grid.block_k_off = bk_off;
    ctx->tid = tid_v;
    ctx->lane = lane_v;
    ctx->warp_id = warp_id_v;
    ctx->warp_m_idx = warp_m_v;
    ctx->warp_n_idx = warp_n_v;

    /* Geometry constants -- K-loop bound is wg_K (or split-K slice size).
     *
     * Creation order must mirror Python (build_implicit_gemm_conv_wgrad, after bind):
     *   c0        = b.const_i32(0)         -- always
     *   c_block_k = b.const_i32(block_k)   -- always
     *   c_wg_K    = p_wg_K kernarg         -- always
     *   c_ks = ks_param, slice = z % ks_count, k_lo = to_sgpr(slice * c_ks),
     *   k_hi = to_sgpr(k_lo + c_ks) -- always; the degree is a kernarg
     */
    /* c0: always const(0). For split_k=1 this is also k_lo. */
    rocke_value_t* c0_node = rocke_b_const_i32(b, 0);
    /* c_block_k: always created here (matches Python ordering). */
    ctx->c_block_k = rocke_b_const_i32(b, ctx->block_k);
    /* AOT: the reduction extent is the p_wg_K kernarg, not a folded constant. */
    rocke_value_t* c_wg_K = ctx->p_wg_K;

    /* Split-K K-slice bounds. The degree is a launch parameter (ks_count),
     * never a compile-time constant, so there is exactly one shape of decode:
     *
     *     slice = z % ks_count
     *     k_lo  = slice * ks ;  k_hi = k_lo + ks
     *
     * An unsplit launch passes ks_count = 1 and ks = wg_K rounded up to a
     * whole number of K tiles, which collapses that to k_lo = 0, k_hi = padded
     * wg_K. The tail past wg_K reads zero through the descriptor bounds, so
     * the padding contributes nothing -- which is what lets the split and
     * unsplit cases share one computation instead of two branches that have to
     * be kept in step. Grouped wgrad is rejected by the validator here, so the
     * group half of the decode is not emitted.
     *
     * C leaves argument evaluation order unspecified, so each subexpression is
     * bound to a temp in Python's left-to-right order. */
    rocke_value_t* z_id = rocke_b_block_id_z(b);
    rocke_value_t* slice_id = rocke_b_mod(b, z_id, ks_count_param);
    rocke_value_t* mul_lo = rocke_b_mul(b, slice_id, ks_param);
    rocke_value_t* k_lo = rocke_b_to_sgpr_u32(b, mul_lo);
    rocke_value_t* add_hi = rocke_b_add(b, k_lo, ks_param);
    rocke_value_t* k_hi_v = rocke_b_to_sgpr_u32(b, add_hi);
    /* ctx->c0 is the literal zero the epilogue passes as a buffer soffset.
     * It used to be aliased to k_lo, which only worked while the unsplit path
     * made k_lo the constant 0; now that every launch computes k_lo from the
     * slice index, the two have to stay separate or the stores pick up the
     * slice base as their soffset. */
    ctx->c0 = c0_node;
    ctx->c_K_gemm = k_hi_v;
    /* wgrad's async k-loop offsets are b.add(k_lo, const_i32(...)) -- the slice
     * base is part of the expression, unlike the forward conv which uses a bare
     * const. Handing the driver k_lo keeps the emitted SSA identical. */
    ctx->kloop_k_lo = k_lo;
    /* AOT: the drivers walk a runtime [k_lo, k_hi) range, so there is no
     * build-time iteration count left to derive. */
    ctx->kloop_k_hi = ctx->c_K_gemm;
    /* k_hi is a slice end inside the tensor under split-K; past wg_K the
     * descriptor zero-fills, so that is where a stray prefetch is sent. */
    ctx->kloop_k_zero_fill = c_wg_K;
    ctx->kloop_num_iters = 0;

    /* Chiplet swizzle */
    if(spec->chiplet_swizzle)
    {
        /* AOT: the tile counts are kernargs -- the host computes them from
         * the launch shape and the tile size it dispatched. */
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        rocke_value_t* mul_y = rocke_b_mul(b, bid_y, ctx->p_num_pid_n);
        rocke_value_t* bid_x = rocke_b_block_id_x(b);
        rocke_value_t* wgflat = rocke_b_add(b, mul_y, bid_x);
        rocke_super_tile_swizzle_result_t swz
            = rocke_chiplet_aware_super_tile_dynamic(b,
                                                     wgflat,
                                                     ctx->p_num_pid_m,
                                                     ctx->p_num_pid_n,
                                                     spec->chiplet_wgm,
                                                     spec->chiplet_num_xcds,
                                                     spec->chiplet_chunk_size);
        ctx->block_m_off_v = rocke_b_mul(b, swz.row, rocke_b_const_i32(b, ctx->block_m));
        ctx->block_n_off_v = rocke_b_mul(b, swz.col, rocke_b_const_i32(b, ctx->block_n));
        ctx->grid.block_m_off = ctx->block_m_off_v;
        ctx->grid.block_n_off = ctx->block_n_off_v;
    }
    else
    {
        ctx->block_m_off_v = ctx->grid.block_m_off;
        ctx->block_n_off_v = ctx->grid.block_n_off;
    }

    /* LDS layout -- reuse the forward accessor (same logic) */
    {
        rocke_implicit_gemm_conv_spec_t lds_fwd = *ctx->spec;
        char lds_reason[256];
        if(!rocke_implicit_gemm_conv_spec_effective_lds_layout(
               &lds_fwd, &ctx->lds_layout, lds_reason, sizeof(lds_reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", lds_reason);
            return false;
        }
    }

    /* smem_alloc A_smem / B_smem */
    {
        /* K-outer: rows are K_wg, columns are the free axis (M for dY, N_wg for
         * X), with an 8-element pad so the row stride is not a multiple of the
         * LDS bank period (see the Python comment for the derivation). */
        int a_sh[2] = {ctx->block_m, ctx->lds_layout.row_stride};
        int b_sh[2] = {ctx->block_n, ctx->lds_layout.row_stride};
        if(spec->lds_k_outer)
        {
            a_sh[0] = ctx->block_k;
            /* Direct load writes packed lane-contiguous bytes and cannot skip a
             * row pad, so the async path uses a pad of 0. */
            const int kpad = spec->async_dma ? 0 : ROCKE_WGRAD_KOUTER_PAD;
            a_sh[1] = ctx->block_m + kpad;
            b_sh[0] = ctx->block_k;
            b_sh[1] = ctx->block_n + kpad;
        }
        /* Element type must follow the operand dtype, exactly as Python does with
         * ir_dtype_a / ir_dtype_b. Hardcoding f16 typed the LDS pool and every
         * tile store `half` for a bf16 kernel, diverging from Python -- invisible
         * until now because no wgrad parity config used bf16 A/B operands. */
        const rocke_type_t* ir_dtype_a = rocke_conv_tr_elem_dtype(spec->dtype_a);
        const rocke_type_t* ir_dtype_b = rocke_conv_tr_elem_dtype(spec->dtype_b);
        ctx->A_smem = rocke_b_smem_alloc(b, ir_dtype_a, a_sh, 2, "A_smem");
        ctx->B_smem = rocke_b_smem_alloc(b, ir_dtype_b, b_sh, 2, "B_smem");
        /* Only async_dma and unroll_k reach a K-loop that alternates buffers:
         * async_dma takes the SoftwarePipeline branch and unroll_k hand-rolls a
         * ping-pong.  "compv4" alone shares the plain single-buffer loop with
         * "mem"/"compv3" and differs only in scheduling hints, so allocating a
         * second A/B tile for it was dead and charged LDS the kernel never used.
         * Mirrors the Python double_buffer condition. */
        ctx->double_buffer = spec->async_dma || spec->unroll_k;
        if(ctx->double_buffer)
        {
            ctx->A_smem2 = rocke_b_smem_alloc(b, ir_dtype_a, a_sh, 2, "A_smem2");
            ctx->B_smem2 = rocke_b_smem_alloc(b, ir_dtype_b, b_sh, 2, "B_smem2");
        }
        else
        {
            ctx->A_smem2 = ctx->A_smem;
            ctx->B_smem2 = ctx->B_smem;
        }
    }

    /* Per-warp MFMA tile counts */
    ctx->mfmas_m = spec->tile_m / (spec->warp_m * spec->warp_tile_m);
    ctx->mfmas_n = spec->tile_n / (spec->warp_n * spec->warp_tile_n);
    ctx->k_atoms = spec->tile_k / spec->warp_tile_k;

    /* Accumulators */
    ctx->acc_init = rocke_b_zero_vec_f32(b, ctx->c_per_lane);
    ctx->num_accs = ctx->mfmas_m * ctx->mfmas_n;
    if(ctx->num_accs <= 0 || ctx->num_accs > ROCKE_CONV_MAX_ACCS)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: too many accumulators (%d)", ctx->num_accs);
        return false;
    }
    for(int i = 0, mi = 0; mi < ctx->mfmas_m; ++mi)
        for(int ni = 0; ni < ctx->mfmas_n; ++ni, ++i)
        {
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "acc_m%d_n%d", mi, ni);
            ctx->acc_names[i] = rocke_arena_strdup(&b->arena, tmp);
            ctx->acc_inits[i] = ctx->acc_init;
        }

    /* Load plan.  See the Python comment in conv_implicit_gemm_wgrad.py: A (dY,
     * NHWK) and B (X, NHWC) have their GEMM reduction axis K_wg = N*Ho*Wo, which
     * is NOT the stride-1 tensor axis.  The stride-1 axis is the free axis
     * (k_out = M for dY, inner C of N_wg for X), so vectorise the global load
     * along that axis and transpose-on-store into the (M/N, K) LDS tile
     * (vector_axis="row").  The transpose-on-store fills the SAME row-major LDS
     * the scalar path produced, so the MMA consumer (MFMA or WMMA) reads it
     * unchanged.  Enabled for every sync path (MFMA and WMMA); only the async
     * path is excluded (it writes lane-contiguous LDS and cannot host a
     * transpose-on-store).  vec_a | K and vec_b | C keep the free-axis vector
     * within one stride-1 run; choose_vec_axis enforces even tile distribution;
     * width 1 falls back to the scalar vector_axis="col" path (byte-identical). */
    ctx->threads = spec->warp_m * spec->warp_n * spec->wave_size;
    ctx->load_vec = 1;

    /* Loaders */
    ctx->async_dma = spec->async_dma;
    ctx->lds_k_outer = spec->lds_k_outer;
    /* The async A slot has no a_load_override; give it the wgrad dY descriptor
     * (swapped coordinates on the K-outer tile) so it does not fall back to the
     * forward descriptor. */
    ctx->a_descriptor_fn = spec->lds_k_outer ? wgrad_dy_descriptor_kouter : wgrad_dy_descriptor;
    if(ctx->async_dma)
    {
        /* K-outer: rows are K_wg, columns are the free axis, so a chunk is a run
         * along the free axis at one k_wg -- contiguous in global for both
         * operands, which is exactly what the intrinsic requires. contig_cols
         * keeps a chunk inside one such run: kpg for dY (NHWK, dense over the
         * output-channel slab) and cpg for X (NHWC, dense only within one
         * filter position). Mirrors the Python call. */
        const int a_rows = ctx->lds_k_outer ? ctx->block_k : ctx->block_m;
        const int a_cols = ctx->lds_k_outer ? ctx->block_m : ctx->block_k;
        const int b_rows = ctx->lds_k_outer ? ctx->block_k : ctx->block_n;
        const int b_cols = ctx->lds_k_outer ? ctx->block_n : ctx->block_k;
        const int a_contig = ctx->lds_k_outer ? rocke_conv_problem_kpg(&spec->problem) : 0;
        const int b_contig = ctx->lds_k_outer ? rocke_conv_problem_cpg(&spec->problem) : 0;
        rocke_status_t sa = rocke_async_tile_loader_from_tile(
            a_rows, a_cols, ctx->threads, spec->wave_size, 4, a_contig, &ctx->a_loader);
        rocke_status_t sb = rocke_async_tile_loader_from_tile(
            b_rows, b_cols, ctx->threads, spec->wave_size, 4, b_contig, &ctx->b_loader);
        if(sa != ROCKE_OK || sb != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: async tile loader init failed");
            return false;
        }
        ctx->have_async_loaders = true;
        ctx->have_sync_loaders = false;
    }
    else
    {
        int va = 1;
        int vb = 1;
        bool axis_a = false;
        bool axis_b = false;
        /* Every sync path (MFMA and WMMA); the async branch is handled above. */
        {
            /* free-axis width: widest of {8,4,2,1} ({4,2,1} for fp32) dividing
             * the channel count -- Python _free_axis_vec(kpg/cpg, dtype). */
            int kpg = rocke_conv_problem_kpg(&spec->problem);
            int cpg = rocke_conv_problem_cpg(&spec->problem);
            bool a_fp32 = (spec->dtype_a != NULL && strcmp(spec->dtype_a, "fp32") == 0);
            bool b_fp32 = (spec->dtype_b != NULL && strcmp(spec->dtype_b, "fp32") == 0);
            int cap_a = 1;
            int cap_b = 1;
            int wa, wb;
            for(wa = a_fp32 ? 4 : 8; wa >= 1; wa /= 2)
                if(kpg % wa == 0)
                {
                    cap_a = wa;
                    break;
                }
            for(wb = b_fp32 ? 4 : 8; wb >= 1; wb /= 2)
                if(cpg % wb == 0)
                {
                    cap_b = wb;
                    break;
                }
            /* Mirror the Python choose_vec, which RAISES ValueError when the tile
             * geometry admits no usable width (not even 1): fail fast here too so
             * a bad geometry surfaces as a clear error instead of silently
             * degrading to the scalar path and diverging from the Python engine. */
            rocke_status_t sva = rocke_coalesced_tile_loader_choose_vec_axis(
                ctx->block_m, ctx->block_k, ctx->threads, cap_a, /*row=*/true, &va);
            rocke_status_t svb = rocke_coalesced_tile_loader_choose_vec_axis(
                ctx->block_n, ctx->block_k, ctx->threads, cap_b, /*row=*/true, &vb);
            if(sva != ROCKE_OK || svb != ROCKE_OK)
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "wgrad: no usable free-axis load_vec for tile geometry");
                return false;
            }
            if(va > 1)
                axis_a = true;
            else
                va = 1;
            if(vb > 1)
                axis_b = true;
            else
                vb = 1;

            if(spec->lds_k_outer)
            {
                /* K-outer tile: the free axis is stride-1 in global AND contiguous in
                 * LDS, so the classic vector_axis="col" loader applies directly and
                 * its store collapses to one wide smem_store_vN. */
                rocke_status_t ka = rocke_coalesced_tile_loader_choose_vec_axis(
                    ctx->block_k, ctx->block_m, ctx->threads, cap_a, /*row=*/false, &va);
                rocke_status_t kb = rocke_coalesced_tile_loader_choose_vec_axis(
                    ctx->block_k, ctx->block_n, ctx->threads, cap_b, /*row=*/false, &vb);
                if(ka != ROCKE_OK || kb != ROCKE_OK)
                {
                    rocke_i_set_err(
                        b, ROCKE_ERR_VALUE, "wgrad: no usable load_vec for K-outer tile geometry");
                    return false;
                }
                axis_a = false;
                axis_b = false;
            }
        }

        /* Direct struct construction mirrors the Python CoalescedTileLoader(...)
         * call (not from_tile): explicit load_vec + vector_axis, use_buffer_rsrc
         * default True, oob_sentinel default (1 << 31) - 1. */
        ctx->a_sync_loader.tile_rows = spec->lds_k_outer ? ctx->block_k : ctx->block_m;
        ctx->a_sync_loader.tile_cols = spec->lds_k_outer ? ctx->block_m : ctx->block_k;
        ctx->a_sync_loader.block_size = ctx->threads;
        ctx->a_sync_loader.load_vec = va;
        ctx->a_sync_loader.use_buffer_rsrc = true;
        ctx->a_sync_loader.oob_sentinel = 2147483647;
        ctx->a_sync_loader.vector_axis_row = axis_a;
        ctx->a_sync_loader.has_inner_dim = false;
        ctx->a_sync_loader.inner_dim = 0;

        ctx->b_sync_loader.tile_rows = spec->lds_k_outer ? ctx->block_k : ctx->block_n;
        ctx->b_sync_loader.tile_cols = spec->lds_k_outer ? ctx->block_n : ctx->block_k;
        ctx->b_sync_loader.block_size = ctx->threads;
        ctx->b_sync_loader.load_vec = vb;
        ctx->b_sync_loader.use_buffer_rsrc = true;
        ctx->b_sync_loader.oob_sentinel = 2147483647;
        ctx->b_sync_loader.vector_axis_row = axis_b;
        ctx->b_sync_loader.has_inner_dim = false;
        ctx->b_sync_loader.inner_dim = 0;

        ctx->have_sync_loaders = true;
        ctx->have_async_loaders = false;
    }

    /* Schedule -- only compute the policy here; the caller emits the prologue
     * AFTER the buffer resources so the SSA order matches Python:
     *   buffer_rsrc(dY/X/dW) -> schedule.emit_prologue(b) -> k-loop. */
    ctx->schedule
        = rocke_schedule_policy_for_pipeline(b, ctx->async_dma ? "async_dma" : spec->pipeline);

    return rocke_ir_builder_ok(b);
}

// ---------------------------------------------------------------------------
// rocke_build_implicit_gemm_conv_wgrad
// ---------------------------------------------------------------------------

/* Pointer declarations for the wgrad kernarg list. The attributes are
 * builder-specific (dW is read-modify-write under split-K atomics), so the
 * ABI list names the slot and this callback owns the type and the opts. */
struct WgradPtrDecl
{
    const rocke_type_t* a_type;
    const rocke_type_t* b_type;
    const rocke_type_t* d_type;
    const rocke_param_opts_t* ro_opts;
    const rocke_param_opts_t* d_opts;
    const rocke_param_opts_t* ws_opts;
};

static rocke_value_t* wgrad_declare_ptr(rocke_ir_builder_t* b,
                                        const char* name,
                                        rocke_conv_arg_kind_t kind,
                                        void* user)
{
    const WgradPtrDecl* d = static_cast<const WgradPtrDecl*>(user);
    switch(kind)
    {
    case ROCKE_CONV_ARG_A:
        return rocke_b_param(b, name, d->a_type, d->ro_opts);
    case ROCKE_CONV_ARG_B:
        return rocke_b_param(b, name, d->b_type, d->ro_opts);
    case ROCKE_CONV_ARG_D:
        return rocke_b_param(b, name, d->d_type, d->d_opts);
    case ROCKE_CONV_ARG_F32_PTR:
        /* Two-stage: f32 workspace that receives the partial sums. */
        return rocke_b_param(b, name, rocke_ptr_type(b, rocke_f32(), "global"), d->ws_opts);
    default:
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "wgrad: unexpected pointer kind for %s", name);
    }
}

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_wgrad(
    rocke_ir_builder_t* b, const rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";

    /* --- validation --- */
    char reason[256];
    if(!rocke_implicit_gemm_conv_wgrad_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: %s", reason);
        return NULL;
    }

    /* split_k=-1 (auto) is not supported in this port: Python resolves it via
     * select_split_k_wgrad which picks a degree > 1 for most shapes.  Silently
     * collapsing to 1 would mismatch the Python source of truth (wrong kernel
     * name, wrong grid) and could cause races if the host launches with the
     * degree Python's formula implies.  Callers must resolve -1 before calling
     * this function or pass an explicit degree. */
    int split_k = spec->split_k;
    if(split_k == -1)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "wgrad: split_k=-1 (auto) is not supported in the C port; "
                        "resolve via select_split_k_wgrad and pass the explicit degree");
        return NULL;
    }

    bool effective_two_stage = spec->two_stage;

    bool is_split_k = split_k > 1;

    /* split_k atomic (>1) supported for fp32, fp16, bf16 output dtypes */
    if(is_split_k)
    {
        const char* dt = spec->dtype_d ? spec->dtype_d : "fp16";
        if(strcmp(dt, "fp32") != 0 && strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "wgrad: split_k atomic requires dtype_d in fp32/fp16/bf16");
            return NULL;
        }
    }

    int wg_K = rocke_wgrad_conv_spec_wg_K(spec);
    int wg_M = rocke_wgrad_conv_spec_wg_M(spec);
    int wg_N = rocke_wgrad_conv_spec_wg_N(spec);

    const rocke_conv_problem_t* p = &spec->problem;
    /* --- wgrad ctx: host state only, filled as the params are declared --- */
    rocke_conv_build_ctx_t ctx;
    bool is_two_stage = is_split_k && effective_two_stage;
    rocke_value_t* ks_param = NULL;
    rocke_value_t* ks_count_param = NULL;
    rocke_value_t* ws_ptr = NULL;
    memset(&ctx, 0, sizeof(ctx));

    /* --- kernel params with wgrad names (Python: dY, X, dW, *_bytes) --- */
    rocke_param_opts_t ro_opts;
    memset(&ro_opts, 0, sizeof(ro_opts));
    ro_opts.noalias = true;
    ro_opts.noalias_set = true;
    ro_opts.readonly = true;
    ro_opts.readonly_set = true;
    ro_opts.align = 16;
    ro_opts.align_set = true;

    rocke_param_opts_t d_opts;
    memset(&d_opts, 0, sizeof(d_opts));
    d_opts.noalias = true;
    d_opts.noalias_set = true;
    /* split_k>1: dW is read+write (atomic); split_k=1: writeonly.
     * Caller MUST zero-init dW before launch for atomic paths -- the kernel only
     * issues atomic-adds.  See the header contract note for details. */
    d_opts.writeonly = !is_split_k;
    d_opts.writeonly_set = true;
    d_opts.align = 16;
    d_opts.align_set = true;

    /* dtype for dW/dY/X: rocke_b_io_ir_type handles f16/bf16 only.
     * fp32 inputs/outputs use rocke_f32() directly. */
#define _WGRAD_ELEM_TYPE(dt_field, fallback)                                             \
    (((dt_field) && (strcmp((dt_field), "fp32") == 0 || strcmp((dt_field), "f32") == 0)) \
         ? rocke_f32()                                                                   \
         : rocke_b_io_ir_type(b, (dt_field) ? (dt_field) : (fallback)))

    const rocke_type_t* dw_elem = _WGRAD_ELEM_TYPE(spec->dtype_d, "fp16");
    const rocke_type_t* dw_glob = rocke_ptr_type(b, dw_elem, "global");

    const rocke_type_t* dy_glob
        = rocke_ptr_type(b, _WGRAD_ELEM_TYPE(spec->dtype_a, "fp16"), "global");
    const rocke_type_t* x_glob
        = rocke_ptr_type(b, _WGRAD_ELEM_TYPE(spec->dtype_b, "fp16"), "global");
#undef _WGRAD_ELEM_TYPE
    rocke_value_t* dY = NULL;
    rocke_value_t* X = NULL;
    rocke_value_t* dW = NULL;
    rocke_value_t* dY_bytes = NULL;
    rocke_value_t* X_bytes = NULL;
    rocke_value_t* dW_bytes = NULL;

    /* ---- the whole kernarg list ----
     * Pointers, byte sizes, the shared extent block, wgrad's own GEMM dims,
     * strides and magic pairs, then the variant-specific extras -- emitted
     * from the ordered ABI list, as the Python builder emits from
     * conv_arg_names(direction="wgrad"), so the order cannot drift from the
     * launch signature. */
    {
        const bool is_3d = spec->problem.is_3d;
        rocke_param_opts_t ws_opts;
        memset(&ws_opts, 0, sizeof(ws_opts));
        ws_opts.noalias = true;
        ws_opts.noalias_set = true;
        /* Not writeonly: an atomicrmw reads its target. */
        ws_opts.align = 16;
        ws_opts.align_set = true;
        WgradPtrDecl ptrs;
        ptrs.a_type = dy_glob;
        ptrs.b_type = x_glob;
        ptrs.d_type = dw_glob;
        ptrs.ro_opts = &ro_opts;
        ptrs.d_opts = &d_opts;
        ptrs.ws_opts = &ws_opts;
        const rocke_conv_param_slot_t slots[] = {
            {"dY", &dY},
            {"X", &X},
            {"dW", &dW},
            {"dY_bytes", &dY_bytes},
            {"X_bytes", &X_bytes},
            {"dW_bytes", &dW_bytes},
            {"p_N", &ctx.p_N},
            {"p_Hi", &ctx.p_Hi},
            {"p_Wi", &ctx.p_Wi},
            {"p_C", &ctx.p_C},
            {"p_K", &ctx.p_K},
            {"p_Y", &ctx.p_Y},
            {"p_X", &ctx.p_X},
            {"p_Z", &ctx.p_Z},
            {"p_Di", &ctx.p_Di},
            {"p_sH", &ctx.p_sH},
            {"p_sW", &ctx.p_sW},
            {"p_pH", &ctx.p_pH},
            {"p_pW", &ctx.p_pW},
            {"p_dH", &ctx.p_dH},
            {"p_dW", &ctx.p_dW},
            {"p_sD", &ctx.p_sD},
            {"p_pD", &ctx.p_pD},
            {"p_dD", &ctx.p_dD},
            {"p_groups", &ctx.p_groups},
            {"p_Ho", &ctx.p_Ho},
            {"p_Wo", &ctx.p_Wo},
            {"p_Do", &ctx.p_Do},
            {"p_cpg", &ctx.p_cpg},
            {"p_kpg", &ctx.p_kpg},
            {"p_wg_M", &ctx.p_wg_M},
            {"p_wg_N", &ctx.p_wg_N},
            {"p_wg_K", &ctx.p_wg_K},
            {"p_dY_stride_n", &ctx.p_dY_stride_n},
            {"p_dY_stride_do", &ctx.p_dY_stride_do},
            {"p_dY_stride_ho", &ctx.p_dY_stride_ho},
            {"p_dY_stride_wo", &ctx.p_dY_stride_wo},
            {"p_X_stride_n", &ctx.p_X_stride_n},
            {"p_X_stride_di", &ctx.p_X_stride_di},
            {"p_X_stride_hi", &ctx.p_X_stride_hi},
            {"p_X_stride_wi", &ctx.p_X_stride_wi},
            {"p_dW_stride_k", &ctx.p_dW_stride_k},
            {"p_dW_stride_z", &ctx.p_dW_stride_z},
            {"p_dW_stride_y", &ctx.p_dW_stride_y},
            {"p_dW_stride_x", &ctx.p_dW_stride_x},
            {"p_magic_k_Do_mult", &ctx.p_magic_k_Do_mult},
            {"p_magic_k_Do_shift", &ctx.p_magic_k_Do_shift},
            {"p_magic_k_Ho_mult", &ctx.p_magic_k_Ho_mult},
            {"p_magic_k_Ho_shift", &ctx.p_magic_k_Ho_shift},
            {"p_magic_k_Wo_mult", &ctx.p_magic_k_Wo_mult},
            {"p_magic_k_Wo_shift", &ctx.p_magic_k_Wo_shift},
            {"p_magic_n_Y_mult", &ctx.p_magic_n_Y_mult},
            {"p_magic_n_Y_shift", &ctx.p_magic_n_Y_shift},
            {"p_magic_n_X_mult", &ctx.p_magic_n_X_mult},
            {"p_magic_n_X_shift", &ctx.p_magic_n_X_shift},
            {"p_magic_n_cpg_mult", &ctx.p_magic_n_cpg_mult},
            {"p_magic_n_cpg_shift", &ctx.p_magic_n_cpg_shift},
            {"p_num_pid_m", &ctx.p_num_pid_m},
            {"p_num_pid_n", &ctx.p_num_pid_n},
            /* ws_bytes is consumed by the host only; it has no slot. */
            {"ws_ptr", &ws_ptr},
            {"ks", &ks_param},
            {"ks_count", &ks_count_param},
        };
        rocke_conv_arg_list_t abi;
        ctx.params_is_3d = is_3d;
        if(!rocke_conv_arg_names("wgrad", is_3d, is_two_stage, &abi))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: no kernarg ABI for this variant");
            return NULL;
        }
        if(!rocke_conv_emit_param_block(
               b, &abi, wgrad_declare_ptr, &ptrs, slots, (int)(sizeof(slots) / sizeof(slots[0]))))
        {
            return NULL;
        }
    }

    if(!wgrad_build_ctx_init(&ctx, b, spec, arch, wg_K, ks_param, ks_count_param))
        return NULL;

    /* Wire the params we declared into the ctx slots the phases read */
    ctx.A = dY;
    ctx.Bp = X;
    ctx.D = dW;
    ctx.A_bytes = dY_bytes;
    ctx.B_bytes = X_bytes;
    ctx.D_bytes = dW_bytes;

    /* Pointwise prologue IR constants (mirrors Python build_implicit_gemm_conv_wgrad,
     * before dy_buf_rsrc = make_buffer_resource):
     *   if p.is_pointwise:
     *     _c_K_ir   = b.const_i32(p.kpg)  <- first
     *     _c_C_ir   = b.const_i32(p.cpg)  <- second
     *     _c_wgM_ir = b.const_i32(wg_M)   <- third
     *     _c_wgN_ir = b.const_i32(wg_N)   <- fourth
     *     _c_wgK_ir = b.const_i32(wg_K)   <- fifth
     * Emitted before buffer_rsrc so the SSA sequence matches Python. */
    if(ctx.is_pointwise)
    {
        /* AOT: every pointwise bound is a kernarg, so nothing is emitted here.
         * The slot names below are historical; the comment gives the Python
         * name each one carries. */
        ctx.ir_c_C_pw = ctx.p_kpg; /* _c_K_ir   */
        ctx.ir_c_K_pw = ctx.p_cpg; /* _c_C_ir   */
        ctx.ir_c_M_pw = ctx.p_wg_M; /* _c_wgM_ir */
        ctx.ir_always_valid = ctx.p_wg_N; /* _c_wgN_ir */
        ctx.ir_c_wgN_pw = ctx.p_wg_K; /* _c_wgK_ir */
    }
    else
    {
        ctx.ir_c_C_pw = ctx.ir_c_K_pw = ctx.ir_c_M_pw = ctx.ir_always_valid = NULL;
        ctx.ir_c_wgN_pw = NULL;
    }

    /* --- wgrad-specific descriptors ---
     * Pointwise fast path (Y=X=1, stride=1, pad=0): descriptors are NULL; the
     * wgrad_dy_descriptor / wgrad_x_descriptor closures use flat arithmetic.
     * Non-pointwise: build the full coordinate-transform descriptor DAGs. */
    rocke_tensor_descriptor_t* dY_desc = NULL;
    rocke_tensor_descriptor_t* X_desc = NULL;
    if(!ctx.is_pointwise)
    {
        /* AOT: every extent, stride and magic constant is a kernarg. The dY
         * and X descriptors are the forward D/A DAGs under wgrad's coord
         * names, so they reuse the shared builders with wgrad's stride slots
         * and magic families (k_ for the reduction, n_ for the filter). */
        rocke_conv_dyn_desc_opts_t dy_opts;
        rocke_conv_dyn_desc_opts_t x_opts;
        rocke_conv_dyn_desc_opts_t dw_opts;
        const bool is_3d = spec->problem.is_3d;

        memset(&dy_opts, 0, sizeof(dy_opts));
        dy_opts.name = is_3d ? "dY_ndhwk" : "dY_nhwk";
        dy_opts.stride_n = ctx.p_dY_stride_n;
        dy_opts.stride_di = ctx.p_dY_stride_do;
        dy_opts.stride_hi = ctx.p_dY_stride_ho;
        dy_opts.stride_wi = ctx.p_dY_stride_wo;
        dy_opts.spatial_di.mult = ctx.p_magic_k_Do_mult;
        dy_opts.spatial_di.shift = ctx.p_magic_k_Do_shift;
        dy_opts.spatial_hi.mult = ctx.p_magic_k_Ho_mult;
        dy_opts.spatial_hi.shift = ctx.p_magic_k_Ho_shift;
        dy_opts.spatial_wi.mult = ctx.p_magic_k_Wo_mult;
        dy_opts.spatial_wi.shift = ctx.p_magic_k_Wo_shift;
        dY_desc
            = (rocke_tensor_descriptor_t*)rocke_conv_make_dy_descriptor_dynamic(b, &ctx, &dy_opts);

        memset(&x_opts, 0, sizeof(x_opts));
        x_opts.name = is_3d ? "X_ndhwc" : "X_nhwc";
        /* The shared load phase queries B_desc with ("k_out", "k_gemm"), so
         * the wgrad X descriptor exposes those names for the same axes the
         * forward A descriptor calls ("m", "k"). */
        x_opts.spatial_upper = "k_gemm";
        x_opts.channel_upper = "k_out";
        x_opts.stride_n = ctx.p_X_stride_n;
        x_opts.stride_di = ctx.p_X_stride_di;
        x_opts.stride_hi = ctx.p_X_stride_hi;
        x_opts.stride_wi = ctx.p_X_stride_wi;
        x_opts.spatial_di.mult = ctx.p_magic_k_Do_mult;
        x_opts.spatial_di.shift = ctx.p_magic_k_Do_shift;
        x_opts.spatial_hi.mult = ctx.p_magic_k_Ho_mult;
        x_opts.spatial_hi.shift = ctx.p_magic_k_Ho_shift;
        x_opts.spatial_wi.mult = ctx.p_magic_k_Wo_mult;
        x_opts.spatial_wi.shift = ctx.p_magic_k_Wo_shift;
        x_opts.channel_y.mult = ctx.p_magic_n_Y_mult;
        x_opts.channel_y.shift = ctx.p_magic_n_Y_shift;
        x_opts.channel_x.mult = ctx.p_magic_n_X_mult;
        x_opts.channel_x.shift = ctx.p_magic_n_X_shift;
        x_opts.channel_c.mult = ctx.p_magic_n_cpg_mult;
        x_opts.channel_c.shift = ctx.p_magic_n_cpg_shift;
        X_desc = (rocke_tensor_descriptor_t*)rocke_conv_make_a_descriptor_dynamic_opts(
            b, &ctx, /*decompose_m=*/true, &x_opts);

        /* dW is built lazily at the epilogue, where Python builds it -- doing
         * it here would emit its stride constant before the K-loop and shift
         * every SSA id in between. */
        (void)dw_opts;

        if(dY_desc == NULL || X_desc == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: descriptor build failed");
            return NULL;
        }
    }
    /* The forward phase functions query A_desc with ("m","k") and B_desc with
     * ("k_out","k_gemm") -- our descriptors are built with exactly those names. */
    ctx.A_desc = dY_desc;
    ctx.B_desc = X_desc;
    /* dW is built at the epilogue; see wgrad_build_dw_descriptor. */
    ctx.D_desc = NULL;

    /* Buffer resources */
    rocke_conv_buffer_resource_t a_rsrc, b_rsrc, d_rsrc;
    {
        a_rsrc.ptr = dY;
        a_rsrc.num_bytes = dY_bytes;
        a_rsrc.rsrc = rocke_b_buffer_rsrc(b, dY, dY_bytes);
        a_rsrc.soffset = rocke_b_const_i32(b, 0);
        b_rsrc.ptr = X;
        b_rsrc.num_bytes = X_bytes;
        b_rsrc.rsrc = rocke_b_buffer_rsrc(b, X, X_bytes);
        b_rsrc.soffset = rocke_b_const_i32(b, 0);
        d_rsrc.ptr = dW;
        d_rsrc.num_bytes = dW_bytes;
        d_rsrc.rsrc = rocke_b_buffer_rsrc(b, dW, dW_bytes);
        d_rsrc.soffset = rocke_b_const_i32(b, 0);
    }
    ctx.a_buf_rsrc = a_rsrc;
    ctx.b_buf_rsrc = b_rsrc;
    ctx.d_buf_rsrc = d_rsrc;
    ctx.a_rsrc = a_rsrc.rsrc;
    ctx.b_rsrc = b_rsrc.rsrc;
    ctx.d_rsrc = d_rsrc.rsrc;

    /* Emit schedule prologue AFTER buffer resources -- mirrors Python ordering:
     *   make_buffer_resource(dY/X/dW) then schedule.emit_prologue(b). */
    rocke_schedule_policy_emit_prologue(&ctx.schedule, b);

    /* Element type for the transpose read. Mirrors Python's
     *   _smem_dtype = BF16 if a_dtype == bf16 else F32 if fp32 else None
     *   tr_dtype    = _smem_dtype if not None else F16
     * Type selection only -- emits no IR, so it is unconditional. Leaving this
     * NULL lets rocke_b_ds_read_tr16_b128 default to f16, which on gfx1250
     * (where the opcode is element-typed) would emit .v8f16 for a bf16 kernel. */
    ctx.tr_dtype = rocke_conv_tr_elem_dtype(spec->dtype_a);

    /* K-outer transpose-read lane constants. Python emits these immediately
     * after schedule.emit_prologue() and before the k-loop (the intervening
     * closures emit no IR), so they must be materialised here to keep the SSA
     * numbering byte-identical. Emitted only on the K-outer path: an
     * unconditional emission would add ops to every existing config. */
    if(ctx.lds_k_outer && spec->wave_size == 64)
    {
        /* Python: b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)) -- evaluated
         * strictly left-to-right. C argument order is unspecified, so sequence
         * every operand into a temporary or the SSA numbering drifts. */
        rocke_value_t* c4a = rocke_b_const_i32(b, 4);
        rocke_value_t* m4 = rocke_b_mod(b, ctx.lane, c4a);
        rocke_value_t* c4b = rocke_b_const_i32(b, 4);
        ctx.tr_lane_mod4 = rocke_b_mul(b, m4, c4b);

        /* Python: b.div(b.mod(lane, b.const_i32(16)), b.const_i32(4)) */
        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
        rocke_value_t* m16 = rocke_b_mod(b, ctx.lane, c16);
        rocke_value_t* c4c = rocke_b_const_i32(b, 4);
        ctx.tr_grp16 = rocke_b_div(b, m16, c4c);
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;

    /* --- K-loop --- */
    if(spec->unroll_k)
        rocke_conv_emit_kloop_unroll(&ctx);
    else if(!spec->async_dma)
        rocke_conv_emit_kloop_simple(&ctx);
    else
        rocke_conv_emit_kloop_async(&ctx);

    if(!rocke_ir_builder_ok(b))
        return NULL;

    /* --- epilogue --- */
    if(is_two_stage)
    {
        /* Two-stage deterministic: plain f32 store to per-k workspace slice. */
        rocke_conv_acc_epilogue_t identity = rocke_conv_acc_epilogue_default();
        rocke_value_t* post_accs[ROCKE_CONV_MAX_ACCS];
        rocke_conv_apply_accumulator_epilogue(
            b, &identity, ctx.final_accs, ctx.num_final_accs, post_accs);
        for(int i = 0; i < ctx.num_final_accs; ++i)
            ctx.final_accs[i] = post_accs[i];

        wgrad_emit_workspace_store_epilogue(b, &ctx, spec, ws_ptr, wg_M, wg_N);
    }
    else if(is_split_k)
    {
        /* Apply identity acc epilogue (wgrad spec has no acc_epilogue field). */
        rocke_conv_acc_epilogue_t identity = rocke_conv_acc_epilogue_default();
        rocke_value_t* post_accs[ROCKE_CONV_MAX_ACCS];
        rocke_conv_apply_accumulator_epilogue(
            b, &identity, ctx.final_accs, ctx.num_final_accs, post_accs);
        /* Overwrite final_accs with the post-epilogue values */
        for(int i = 0; i < ctx.num_final_accs; ++i)
            ctx.final_accs[i] = post_accs[i];

        if(spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0)
            wgrad_emit_split_k_cshuffle_epilogue(b, &ctx, spec, dW, wg_M, wg_N);
        else
            wgrad_emit_split_k_epilogue_f32(b, &ctx, spec, dW, wg_M, wg_N);
    }
    else
    {
        /* Apply identity acc epilogue (wgrad spec has no acc_epilogue field). */
        rocke_conv_acc_epilogue_t identity = rocke_conv_acc_epilogue_default();
        rocke_value_t* post_accs[ROCKE_CONV_MAX_ACCS];
        rocke_conv_apply_accumulator_epilogue(
            b, &identity, ctx.final_accs, ctx.num_final_accs, post_accs);
        for(int i = 0; i < ctx.num_final_accs; ++i)
            ctx.final_accs[i] = post_accs[i];

        /* The dW descriptor is built inside whichever epilogue runs, at the
         * point Python builds it -- the cshuffle path builds it first thing,
         * the direct path after the warp offsets. Building it out here would
         * emit its stride constant for both and in the wrong place. */
        WgradDwAddrCtx dw_addr_ctx;
        dw_addr_ctx.dW_desc = NULL;

        if(spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0)
        {
            dw_addr_ctx.dW_desc = ctx.is_pointwise ? NULL : wgrad_build_dw_descriptor(b, &ctx);
            /* _emit_wgrad_cshuffle_epilogue: CShuffleEpilogue.from_grid(...).store(...)
             * vec_c = WgradConvSpec.default_vector_sizes(C, K, dtype_d, split_k=1)[2]
             * For split_k=1: vec_c = _vec(C) where _vec picks largest of [8,4,2,1]
             * that divides C for fp16/bf16, or [4,2,1] for fp32. */
            const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
            bool is_fp32_vec = (strcmp(dtype_d, "fp32") == 0);
            int C = p->C;
            int vec_c;
            if(is_fp32_vec)
            {
                if(C % 4 == 0)
                    vec_c = 4;
                else if(C % 2 == 0)
                    vec_c = 2;
                else
                    vec_c = 1;
            }
            else
            {
                if(C % 8 == 0)
                    vec_c = 8;
                else if(C % 4 == 0)
                    vec_c = 4;
                else if(C % 2 == 0)
                    vec_c = 2;
                else
                    vec_c = 1;
            }
            rocke_cshuffle_epilogue_t cepi
                = rocke_cshuffle_epilogue_from_grid(ctx.atom, &ctx.grid, vec_c);
            cepi.out_dtype = dtype_d;
            if(ctx.is_pointwise)
            {
                rocke_value_t* c_wgN = rocke_b_const_i32(b, wg_N);
                rocke_cshuffle_epilogue_store(b,
                                              &cepi,
                                              post_accs,
                                              ctx.num_final_accs,
                                              wgrad_dw_addr_pointwise,
                                              (void*)c_wgN,
                                              ctx.d_rsrc,
                                              ctx.p_wg_M,
                                              ctx.p_wg_N);
            }
            else
            {
                rocke_cshuffle_epilogue_store(b,
                                              &cepi,
                                              post_accs,
                                              ctx.num_final_accs,
                                              wgrad_dw_addr,
                                              &dw_addr_ctx,
                                              ctx.d_rsrc,
                                              ctx.p_wg_M,
                                              ctx.p_wg_N);
            }
        }
        else
        {
            /* Use wgrad-specific direct epilogue. Mirrors Python
             * _emit_wgrad_direct_epilogue for MFMA,
             * _emit_wgrad_direct_epilogue_wmma for WMMA. */
            /* Pointwise uses flat arithmetic and never touches dW_desc, so
             * building one would emit a stride constant Python does not. */
            wgrad_emit_direct_epilogue(b, &ctx, spec, ctx.d_rsrc, wg_M, wg_N);
        }
    }

    if(!rocke_ir_builder_ok(b))
        return NULL;

    return b->kernel;
}

// ---------------------------------------------------------------------------
// rocke_build_implicit_gemm_conv_wgrad_new
// ---------------------------------------------------------------------------

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_wgrad_new(
    rocke_ir_builder_t* b, const rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(b == NULL || spec == NULL)
            return NULL;
        char name[256];
        if(rocke_wgrad_conv_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_implicit_gemm_conv_wgrad(b, spec, arch);
    });
}

// ---------------------------------------------------------------------------
// rocke_conv_implicit_gemm_wgrad_lower_to_llvm
// ---------------------------------------------------------------------------

rocke_status_t
    rocke_conv_implicit_gemm_wgrad_lower_to_llvm(const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    auto set_err = [&](const char* msg) {
        if(err && err_cap && msg)
        {
            size_t n = strlen(msg);
            if(n >= err_cap)
                n = err_cap - 1;
            memcpy(err, msg, n);
            err[n] = '\0';
        }
    };

    if(out_ll)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        set_err("lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx950";

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_implicit_gemm_conv_wgrad_new(&b, spec, arch);
    if(kernel == NULL)
    {
        /* Capture status BEFORE free — free memsets the struct to zero. */
        rocke_status_t st = rocke_ir_builder_status(&b);
        const char* m = rocke_ir_builder_error(&b);
        set_err((m && m[0]) ? m : "build_implicit_gemm_conv_wgrad failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
