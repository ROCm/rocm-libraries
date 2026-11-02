// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ port of the implicit-GEMM backward-data convolution spec, validation,
 * descriptors, and tilde decomposition helpers
 * (rocke/instances/common/conv_implicit_gemm_dgrad.py).
 *
 * GEMM orientation (dgrad):
 *   M     = N*Hi*Wi       (input spatial positions)
 *   N_dg  = cpg           (input channels per group; C when ungrouped)
 *   K_dg  = Y*X*kpg       (filter spatial x output channels per group, reduction)
 *
 * Operand roles:
 *   A = dY (NHWK, output gradient)
 *   B = W  (KYXC, weights)
 *   D = dX (NHWC, input gradient)
 */
#include "rocke/instance_conv_implicit_gemm_dgrad.h"

#include <cmath> /* ceil, log2 */
#include <cstdio> /* snprintf */
#include <cstdlib> /* malloc, free */
#include <cstring> /* strcmp, memset */
#include <functional> /* std::function (the dY halo row map) */

#include "rocke/arena.h" /* rocke_arena_strdup */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t, rocke_mmaop_t */
#include "rocke/helper_rocke.helpers.atoms.h" /* rocke_mfma_atom, rocke_c_warp_params, decode */
#include "rocke/helper_rocke.helpers.distribution.h" /* tile distribution */
#include "rocke/helper_rocke.helpers.epilogues.h" /* DirectEpilogue, CShuffleEpilogue, WarpGrid */
#include "rocke/helper_rocke.helpers.grid.h" /* chiplet_aware_super_tile */
#include "rocke/helper_rocke.helpers.loads.h" /* CoalescedTileLoader */
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* decode_mfma_lanes */
#include "rocke/helper_rocke.helpers.schedule.h" /* SchedulePolicy */
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_abi.h" /* rocke_conv_arg_names, emit_param_block */
#include "rocke/instance_conv_implicit_gemm.h" /* rocke_conv_acc_epilogue_default */
#include "rocke/instance_conv_implicit_gemm_internal.h" /* rocke_conv_build_ctx_t, phase fns */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

// ---------------------------------------------------------------------------
// Pure-arithmetic helpers
// ---------------------------------------------------------------------------

static int _ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}
static int _floor_div(int a, int b)
{
    return a / b;
}
static int _gcd(int a, int b)
{
    while(b)
    {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
static int _max(int a, int b)
{
    return a > b ? a : b;
}
static int _min(int a, int b)
{
    return a < b ? a : b;
}

/* Map a dtype string ("fp16", "bf16", "fp32") to the corresponding IR scalar type. */
static const rocke_type_t* _dtype_to_ir(const char* dtype)
{
    if(dtype && strcmp(dtype, "bf16") == 0)
        return rocke_bf16();
    if(dtype && strcmp(dtype, "fp32") == 0)
        return rocke_f32();
    return rocke_f16(); /* fp16 and default */
}

// ---------------------------------------------------------------------------
// Spec default + property accessors  (Python DgradConvSpec @property)
// ---------------------------------------------------------------------------

rocke_dgrad_conv_spec_t rocke_dgrad_conv_spec_default(void)
{
    rocke_dgrad_conv_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "conv_igemm_dgrad";
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
    s.acc_epilogue = rocke_conv_acc_epilogue_default();
    s.split_k = 1;
    s.num_load_waves = 4;
    s.max_sub_gemms = 1024;
    s.xcd_tile_order = true;
    return s;
}

int rocke_dgrad_conv_spec_block_size(const rocke_dgrad_conv_spec_t* s)
{
    return s->warp_m * s->warp_n * s->wave_size;
}

int rocke_dgrad_conv_spec_launch_block_size(const rocke_dgrad_conv_spec_t* s)
{
    int bs = rocke_dgrad_conv_spec_block_size(s);
    if(s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
        return bs + s->num_load_waves * s->wave_size;
    return bs;
}

int rocke_dgrad_conv_spec_k_atoms_per_tile_k(const rocke_dgrad_conv_spec_t* s)
{
    return s->warp_tile_k > 0 ? s->tile_k / s->warp_tile_k : 0;
}

int rocke_dgrad_conv_spec_mfmas_per_warp_m(const rocke_dgrad_conv_spec_t* s)
{
    int d = s->warp_m * s->warp_tile_m;
    return d > 0 ? s->tile_m / d : 0;
}

int rocke_dgrad_conv_spec_mfmas_per_warp_n(const rocke_dgrad_conv_spec_t* s)
{
    int d = s->warp_n * s->warp_tile_n;
    return d > 0 ? s->tile_n / d : 0;
}

int rocke_dgrad_conv_spec_dg_M(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.N * s->problem.Hi * s->problem.Wi;
}

/* Per group, as Python _dg_N / _dg_K: a grouped CTA reduces over its own
 * group's kpg output channels and writes its group's cpg input channels. */
int rocke_dgrad_conv_spec_dg_N(const rocke_dgrad_conv_spec_t* s)
{
    return rocke_conv_problem_cpg(&s->problem);
}

int rocke_dgrad_conv_spec_dg_K(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.Y * s->problem.X * rocke_conv_problem_kpg(&s->problem);
}

int rocke_dgrad_conv_spec_dg_K_padded(const rocke_dgrad_conv_spec_t* s)
{
    int sk = s->split_k > 1 ? s->split_k : 1;
    int stride = s->tile_k * sk;
    int k = rocke_dgrad_conv_spec_dg_K(s);
    return _ceil_div(k, stride) * stride;
}

bool rocke_dgrad_conv_spec_is_strided(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.sH != 1 || s->problem.sW != 1 || s->problem.dH != 1 || s->problem.dW != 1;
}

/* Python _acc_regs_per_lane: fp32 accumulators per lane of one warp's tile. */
static int _dgrad_acc_regs_per_lane(const rocke_dgrad_conv_spec_t* s)
{
    return rocke_dgrad_conv_spec_mfmas_per_warp_m(s) * rocke_dgrad_conv_spec_mfmas_per_warp_n(s)
           * (s->warp_tile_m * s->warp_tile_n / _max(s->wave_size, 1));
}

/* Python _default_lds_k_pad: row pad of the M-outer LDS tiles when lds_k_pad
 * is None. */
static int _dgrad_default_lds_k_pad(int tile_k)
{
    return tile_k >= 16 ? 8 : 0;
}

/* Defined further down (shared with the builder). */
static rocke_conv_lds_layout_t _dgrad_effective_lds_layout(const rocke_dgrad_conv_spec_t* spec);
static const char* _dgrad_mma_family(const char* arch);

/* Python _dgrad_lds_charge: (A/B bytes, C bytes, total bytes) of LDS the
 * builder allocates. One A and one B tile whatever the pipeline (compv4 is a
 * schedule policy here; async_dma / unroll_k are rejected). The spec has no
 * cshuffle_no_alias field (Python's default False), so only the wavelet
 * pipeline keeps C unaliased. */
/* Python DgradConvSpec.dy_halo_lds_rows: LDS rows of the staged dY halo tile,
 * tile_m + (Y-1)*halo_max_wo + (X-1) pixels padded to whole loader passes
 * (block_size threads x 8 elements), plus the trailing all-zero row. */
static int _dgrad_dy_halo_lds_rows(const rocke_dgrad_conv_spec_t* s)
{
    const rocke_conv_problem_t* p = &s->problem;
    int rows = s->tile_m + (p->Y - 1) * s->halo_max_wo + (p->X - 1);
    int pass_elems = rocke_dgrad_conv_spec_block_size(s) * 8;
    int quantum = pass_elems / _gcd(pass_elems, s->tile_k);
    return (rows + quantum - 1) / quantum * quantum + 1;
}

/* Python DgradConvSpec.kouter_b_pad: the K-outer B row pad. */
static int _dgrad_kouter_b_pad(const rocke_dgrad_conv_spec_t* s)
{
    if(s->dy_halo > 0 && s->dy_halo_kouter_pad > 0)
        return s->dy_halo_kouter_pad;
    return ROCKE_DGRAD_KOUTER_PAD;
}

static void _dgrad_lds_charge(const rocke_dgrad_conv_spec_t* s, long* ab, long* c, long* total)
{
    int ab_dtype_bytes = (s->dtype_a && strcmp(s->dtype_a, "fp32") == 0) ? 4 : 2;
    rocke_conv_lds_layout_t layout = _dgrad_effective_lds_layout(s);
    long a_rows = s->dy_halo > 0 ? _dgrad_dy_halo_lds_rows(s) : s->tile_m;
    long a_elems = a_rows * layout.row_stride;
    long b_elems = s->lds_k_outer ? (long)s->tile_k * (s->tile_n + _dgrad_kouter_b_pad(s))
                                  : (long)s->tile_n * layout.row_stride;
    /* dy_halo=2 double-buffers the B tile. */
    long b_copies = s->dy_halo == 2 ? 2 : 1;
    long ab_bytes = (a_elems + b_copies * b_elems) * ab_dtype_bytes;
    int c_dtype_bytes = (s->dtype_d && strcmp(s->dtype_d, "fp32") == 0) ? 4 : 2;
    bool cshuffle = s->epilogue && strcmp(s->epilogue, "cshuffle") == 0;
    long c_bytes = cshuffle ? (long)s->tile_m * s->tile_n * c_dtype_bytes : 0;
    bool no_alias = s->pipeline && strcmp(s->pipeline, "wavelet") == 0;
    *ab = ab_bytes;
    *c = c_bytes;
    *total = no_alias ? ab_bytes + c_bytes : (ab_bytes > c_bytes ? ab_bytes : c_bytes);
}

/* Python DgradConvSpec.folds_sub_gemm_record: at stride 1 the only tilde
 * sub-GEMM's record is a function of the kernargs, so the kernel builds it
 * from them instead of searching and loading the record buffer. Keys on the
 * stride alone (an AOT binary's stride capability). */
static bool _dgrad_folds_sub_gemm_record(const rocke_dgrad_conv_spec_t* s)
{
    return s->problem.sH == 1 && s->problem.sW == 1;
}

/* Python _XCD_TILE_ORDER_ARCHES: the target whose grouped stride-1 kernels
 * lay their tiles out XCD-contiguous, and its XCD count (8 on gfx950). */
#define ROCKE_DGRAD_XCD_TILE_ORDER_ARCH "gfx950"
#define ROCKE_DGRAD_XCD_TILE_ORDER_XCDS 8

/* Python xcd_tile_order_xcds: the XCD count the kernel lays its tile order
 * out for, or 0 (launch order). The build-time half of the rule; the kernel
 * decides the rest from its kernargs. */
static int _dgrad_xcd_tile_order_xcds(const rocke_dgrad_conv_spec_t* s, const char* arch)
{
    if(!arch || strcmp(arch, ROCKE_DGRAD_XCD_TILE_ORDER_ARCH) != 0 || !s->xcd_tile_order)
        return 0;
    if(s->problem.groups <= 1 || s->split_k != 1)
        return 0;
    if(!_dgrad_folds_sub_gemm_record(s))
        return 0;
    return ROCKE_DGRAD_XCD_TILE_ORDER_XCDS;
}

bool rocke_dgrad_conv_spec_needs_atomic(const rocke_dgrad_conv_spec_t* s)
{
    /* split_k>1: multiple CTAs accumulate into the same dX elements via atomic_add.
     * Tilde sub-GEMMs with split_k=1 write to disjoint (n,hi,wi) positions and
     * use direct buffer_store — no atomics needed. */
    return s->split_k > 1;
}

rocke_status_t
    rocke_dgrad_conv_spec_kernel_name(const rocke_dgrad_conv_spec_t* s, char* out, size_t out_cap)
{
    if(!out || out_cap < 16)
        return ROCKE_ERR_VALUE;
    char prob[128];
    rocke_conv_problem_short(&s->problem, prob, sizeof(prob), NULL);
    int n = snprintf(out,
                     out_cap,
                     "%s_%s_t%dx%dx%d_w%dx%d_a%dx%dx%d_%s_%s",
                     s->name ? s->name : "conv_igemm_dgrad",
                     prob,
                     s->tile_m,
                     s->tile_n,
                     s->tile_k,
                     s->warp_m,
                     s->warp_n,
                     s->warp_tile_m,
                     s->warp_tile_n,
                     s->warp_tile_k,
                     s->pipeline ? s->pipeline : "mem",
                     s->epilogue ? s->epilogue : "default");
    /* n is the length snprintf wanted, not what it wrote: check it before
     * using it to offset into out, and check every append the same way. */
    if(n < 0 || (size_t)n >= out_cap)
        return ROCKE_ERR_VALUE;
#define ROCKE_DGRAD_NAME_APPEND(...)                                  \
    do                                                                \
    {                                                                 \
        int w_ = snprintf(out + n, out_cap - (size_t)n, __VA_ARGS__); \
        if(w_ < 0 || (size_t)w_ >= out_cap - (size_t)n)               \
            return ROCKE_ERR_VALUE;                                   \
        n += w_;                                                      \
    } while(0)
    /* Flag order must match Python's kernel_name_join flags dict:
     * async, kouter, spk, ukd, kp, tapk, noxcd, halo, hprio, hkp, hwo. C previously emitted spk before async, which only
     * diverged for a spec setting both -- none exists -- but the K-outer flag
     * sits between them, so the order is now load-bearing. */
    if(s->async_dma)
        ROCKE_DGRAD_NAME_APPEND("_async");
    if(s->lds_k_outer)
        ROCKE_DGRAD_NAME_APPEND("_kouter");
    if(s->split_k > 1)
        ROCKE_DGRAD_NAME_APPEND("_spk%d", s->split_k);
    /* A different K-loop body; only tagged when set (Python flag "ukd"). */
    if(s->uniform_k_decode)
        ROCKE_DGRAD_NAME_APPEND("_ukd");
    if(s->patch)
        ROCKE_DGRAD_NAME_APPEND("_patch");
    /* A non-default A row pad changes the LDS layout (Python flag "kp<n>"). */
    if(s->has_lds_k_pad && s->lds_k_pad != _dgrad_default_lds_k_pad(s->tile_k))
        ROCKE_DGRAD_NAME_APPEND("_kp%d", s->lds_k_pad);
    /* A different K-loop body; only tagged when set (Python flag "tapk"). */
    if(s->tap_outer_k)
        ROCKE_DGRAD_NAME_APPEND("_tapk");
    /* The launch-order tile walk (Python flag "noxcd"). */
    if(!s->xcd_tile_order)
        ROCKE_DGRAD_NAME_APPEND("_noxcd");
    /* The dY halo loop and its knobs (Python flags halo/hprio/hkp/hwo). */
    if(s->dy_halo > 0)
        ROCKE_DGRAD_NAME_APPEND("_halo%d", s->dy_halo);
    if(s->dy_halo_setprio > 0)
        ROCKE_DGRAD_NAME_APPEND("_hprio%d", s->dy_halo_setprio);
    if(s->dy_halo_kouter_pad > 0)
        ROCKE_DGRAD_NAME_APPEND("_hkp%d", s->dy_halo_kouter_pad);
    if(s->halo_max_wo > 0)
        ROCKE_DGRAD_NAME_APPEND("_hwo%d", s->halo_max_wo);
#undef ROCKE_DGRAD_NAME_APPEND
    return ROCKE_OK;
}

/* Python _dy_halo_conflict: false + reason when the dY halo knobs cannot be
 * honoured (reasons verbatim). */
static bool _dgrad_dy_halo_ok(const rocke_dgrad_conv_spec_t* s, char* reason, size_t reason_cap)
{
    const rocke_conv_problem_t* p = &s->problem;
    if(s->dy_halo < 0 || s->dy_halo > 2)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo must be 0 (off), 1 (staged halo) or 2 (staged halo with "
                 "double-buffered B); got %d",
                 s->dy_halo);
        return false;
    }
    if(s->dy_halo_setprio < 0 || s->dy_halo_setprio > 3)
    {
        snprintf(reason, reason_cap, "dy_halo_setprio must be in 0..3; got %d", s->dy_halo_setprio);
        return false;
    }
    if(s->dy_halo_kouter_pad < 0 || s->dy_halo_kouter_pad % 8 != 0)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo_kouter_pad must be a non-negative multiple of 8 (16-byte "
                 "rows for the wide LDS store and the transpose read); got %d",
                 s->dy_halo_kouter_pad);
        return false;
    }
    if(s->halo_max_wo < 0)
    {
        snprintf(reason, reason_cap, "halo_max_wo must be non-negative; got %d", s->halo_max_wo);
        return false;
    }
    if(s->dy_halo == 0)
    {
        if(s->dy_halo_setprio > 0)
        {
            snprintf(reason, reason_cap, "dy_halo_setprio needs dy_halo > 0");
            return false;
        }
        if(s->dy_halo_kouter_pad > 0)
        {
            snprintf(reason, reason_cap, "dy_halo_kouter_pad needs dy_halo > 0 and lds_k_outer");
            return false;
        }
        if(s->halo_max_wo > 0)
        {
            snprintf(reason, reason_cap, "halo_max_wo needs dy_halo > 0");
            return false;
        }
        return true;
    }
    if(!s->tap_outer_k)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo needs the tap-outer K loop (tap_outer_k: stride 1, "
                 "dilation 1, kpg %% tile_k == 0, no split-K, wave64)");
        return false;
    }
    const int Ho = rocke_conv_problem_ho(p);
    const int Wo = rocke_conv_problem_wo(p);
    if(Ho != p->Hi || Wo != p->Wi)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo needs an output the size of the input (Ho == Hi and Wo == Wi)");
        return false;
    }
    if(p->Y * p->X == 1)
    {
        snprintf(reason, reason_cap, "dy_halo needs a filter with more than one tap");
        return false;
    }
    if(s->halo_max_wo <= 0)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo needs halo_max_wo > 0 (the widest Wo served; the staged "
                 "halo is sized for it)");
        return false;
    }
    if(Wo > s->halo_max_wo)
    {
        snprintf(reason,
                 reason_cap,
                 "dy_halo serves Wo <= halo_max_wo; got Wo=%d, halo_max_wo=%d",
                 Wo,
                 s->halo_max_wo);
        return false;
    }
    if(s->dy_halo_kouter_pad > 0 && !s->lds_k_outer)
    {
        snprintf(reason, reason_cap, "dy_halo_kouter_pad needs dy_halo > 0 and lds_k_outer");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Arch-aware spec validation (Python is_valid_dgrad_spec)
// ---------------------------------------------------------------------------

bool rocke_dgrad_conv_is_valid_spec(const rocke_dgrad_conv_spec_t* s,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    if(!arch)
        arch = "gfx950";
    if(!reason || reason_cap < 4)
        return false;

    const rocke_conv_problem_t* p = &s->problem;

    /* Python's ConvProblem.__post_init__ rejects these at construction, so a
     * Python spec can never carry them; the C struct has no constructor, so
     * the validator is the first place to stop a group count that does not
     * divide the channels (cpg / kpg would silently truncate). */
    if(p->groups < 1)
    {
        snprintf(reason, reason_cap, "groups must be >= 1, got %d", p->groups);
        return false;
    }
    if(p->C % p->groups != 0 || p->K % p->groups != 0)
    {
        snprintf(reason,
                 reason_cap,
                 "C=%d and K=%d must both be divisible by groups=%d",
                 p->C,
                 p->K,
                 p->groups);
        return false;
    }

    if(p->is_3d)
    {
        snprintf(reason, reason_cap, "dgrad only supports 2-D convolution currently");
        return false;
    }

    if(s->max_sub_gemms < 1)
    {
        snprintf(reason, reason_cap, "max_sub_gemms must be at least 1");
        return false;
    }
    /* Python: _patch_conflict, before the record count (a patch record exists
     * for patch problems only). */
    if(s->patch)
    {
        const char* why = NULL;
        if(s->split_k != 1)
            why = "patch requires split_k == 1 (the atomic epilogue has no patch store)";
        else if(s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
            why = "patch is not supported with pipeline='wavelet'";
        else if(p->groups != 1)
            why = "patch serves ungrouped problems only";
        else if(!rocke_dgrad_conv_spec_is_strided(s))
            why = "patch runs on the tilde (strided) path";
        else if(p->is_3d
                || !(p->pH == 0 && p->pW == 0 && p->dH == 1 && p->dW == 1 && p->sH >= p->Y
                     && p->sW >= p->X))
            why = "patch serves patch embeddings only (no padding or dilation, "
                  "stride >= filter)";
        if(why)
        {
            snprintf(reason, reason_cap, "%s", why);
            return false;
        }
    }
    {
        rocke_tilde_decomposition_t tl = rocke_compute_tilde(p);
        rocke_sub_gemm_params_t probe[256];
        /* A patch kernel runs one record (Python patch_sub_gemms). */
        int n_sub = s->patch ? 1
                             : rocke_enumerate_sub_gemms(
                                   p, &tl, s->tile_m, s->tile_n, s->tile_k, s->split_k, probe, 256);
        if(n_sub > s->max_sub_gemms)
        {
            snprintf(reason,
                     reason_cap,
                     "this problem decomposes into %d tilde sub-GEMMs but the kernel's "
                     "record table supports at most %d (max_sub_gemms); raise "
                     "max_sub_gemms to build a kernel that can serve it",
                     n_sub,
                     s->max_sub_gemms);
            return false;
        }
    }
    if(s->tile_m % (s->warp_m * s->warp_tile_m))
    {
        snprintf(reason, reason_cap, "tile_m not divisible by warp_m * warp_tile_m");
        return false;
    }
    if(s->tile_n % (s->warp_n * s->warp_tile_n))
    {
        snprintf(reason, reason_cap, "tile_n not divisible by warp_n * warp_tile_n");
        return false;
    }
    if(s->tile_k % s->warp_tile_k)
    {
        snprintf(reason, reason_cap, "tile_k not divisible by warp_tile_k");
        return false;
    }

    /* Arch target lookup (Python: ArchTarget.from_gfx). */
    const rocke_arch_target_t* tgt = rocke_arch_target_from_gfx(arch);
    if(!tgt)
    {
        snprintf(reason, reason_cap, "unknown arch %s", arch);
        return false;
    }

    /* block_size vs hardware cap (Python: block_size > target.max_threads_per_block). */
    int bs = rocke_dgrad_conv_spec_block_size(s);
    int max_tpb = rocke_arch_max_threads_per_block(tgt);
    if(bs > max_tpb)
    {
        snprintf(reason, reason_cap, "block_size %d > %d (hardware cap) on %s", bs, max_tpb, arch);
        return false;
    }

    /* vector_size_c > 1 incompatible with the default (scalar) epilogue,
     * including the paths that would ignore it (split-K atomic, strided
     * tilde). Mirrors Python is_valid_dgrad_spec. */
    {
        if(s->vector_size_c > 1 && strcmp(s->epilogue, "default") == 0)
        {
            snprintf(reason,
                     reason_cap,
                     "default epilogue is not supported with vector size c: %d",
                     s->vector_size_c);
            return false;
        }
    }
    /* Explicit vector widths fit one 16-byte per-lane access (Python:
     * vector_width_reason in is_valid_dgrad_spec). */
    if(!rocke_conv_vector_width_ok(
           "a", s->has_vector_size_a, s->vector_size_a, s->dtype_a, reason, reason_cap)
       || !rocke_conv_vector_width_ok(
           "b", s->has_vector_size_b, s->vector_size_b, s->dtype_b, reason, reason_cap)
       || !rocke_conv_vector_width_ok(
           "c", s->has_vector_size_c, s->vector_size_c, s->dtype_d, reason, reason_cap))
    {
        return false;
    }

    /* wave_size must match arch (Python: spec.wave_size != target.wave_size). */
    if(s->wave_size != tgt->wave_size)
    {
        snprintf(reason,
                 reason_cap,
                 "spec wave_size %d != %s wave_size %d",
                 s->wave_size,
                 arch,
                 tgt->wave_size);
        return false;
    }

    const char* family = (tgt->wave_size == 32) ? "wmma" : "mma";

    /* split_k range and arch/dtype gating (Python: sk checks). */
    int sk = s->split_k;
    if(sk < -1 || sk == 0)
    {
        snprintf(reason, reason_cap, "split_k must be -1 (auto), 1, or >1 (got %d)", sk);
        return false;
    }
    if(sk > 1 && strcmp(family, "mma") != 0)
    {
        snprintf(
            reason, reason_cap, "split_k > 1 is CDNA-only (got family %s on %s)", family, arch);
        return false;
    }
    if(s->patch && strcmp(family, "mma") != 0)
    {
        snprintf(reason, reason_cap, "patch is CDNA-only (got family %s on %s)", family, arch);
        return false;
    }
    if(sk > 1)
    {
        /* Accept fp32, bf16, fp16 for atomic accumulation. */
        const char* dd = s->dtype_d;
        int ok_dtype
            = (strcmp(dd, "fp32") == 0 || strcmp(dd, "bf16") == 0 || strcmp(dd, "fp16") == 0);
        if(!ok_dtype)
        {
            snprintf(
                reason, reason_cap, "split_k > 1 requires dtype_d in fp32/bf16/fp16 (got %s)", dd);
            return false;
        }
        /* Packed bf16/fp16 atomics write <2 x dtype> pairs on the innermost
         * NHWC dimension, and a pair must stay within one group's slab: the
         * channel run that has to be even is cpg (C when ungrouped). */
        int packed = (strcmp(dd, "bf16") == 0 || strcmp(dd, "fp16") == 0);
        int cpg = rocke_conv_problem_cpg(p);
        if(packed && cpg % 2 != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "split_k > 1 with dtype_d=%s requires even channels-per-group "
                     "(packed <2 x dtype> atomic pairs on the innermost NHWC dimension "
                     "must stay within one group's slab); got cpg=%d",
                     dd,
                     cpg);
            return false;
        }
    }

    /* MMA shape availability (Python: target.mma.has_shape). */
    if(!rocke_mma_catalog_has_shape(&tgt->mma,
                                    family,
                                    s->dtype_a,
                                    s->dtype_b,
                                    "fp32",
                                    s->warp_tile_m,
                                    s->warp_tile_n,
                                    s->warp_tile_k))
    {
        snprintf(reason,
                 reason_cap,
                 "unsupported %s warp_tile %dx%dx%d on %s",
                 s->dtype_a,
                 s->warp_tile_m,
                 s->warp_tile_n,
                 s->warp_tile_k,
                 arch);
        return false;
    }

    /* Knobs the dgrad builder does not implement: rejected rather than
     * silently ignored. Mirrors is_valid_dgrad_spec. */
    {
        const char* label = s->async_dma         ? "async_dma"
                            : s->unroll_k        ? "unroll_k"
                            : s->chiplet_swizzle ? "chiplet_swizzle"
                                                 : NULL;
        if(label)
        {
            snprintf(reason,
                     reason_cap,
                     "dgrad does not implement %s (the builder has no such path; "
                     "the flag would be silently ignored)",
                     label);
            return false;
        }
    }

    /* Accumulator footprint per lane (Python: _MAX_ACC_REGS_PER_LANE). A
     * correctness guard, not a no-spill guarantee. */
    {
        int acc_regs = _dgrad_acc_regs_per_lane(s);
        if(acc_regs > ROCKE_DGRAD_MAX_ACC_REGS_PER_LANE)
        {
            snprintf(reason,
                     reason_cap,
                     "accumulator tile needs %d fp32 registers per lane "
                     "(> %d); it fills the whole register file",
                     acc_regs,
                     ROCKE_DGRAD_MAX_ACC_REGS_PER_LANE);
            return false;
        }
    }

    /* LDS budget (Python: _dgrad_lds_charge + target.fits_lds). */
    {
        long ab_lds = 0;
        long c_lds = 0;
        long total_lds = 0;
        _dgrad_lds_charge(s, &ab_lds, &c_lds, &total_lds);
        if(!rocke_arch_fits_lds(tgt, total_lds))
        {
            snprintf(reason,
                     reason_cap,
                     "LDS budget %ld bytes (A/B=%ld, C=%ld) > %d cap on %s",
                     total_lds,
                     ab_lds,
                     c_lds,
                     tgt->lds_capacity_bytes,
                     arch);
            return false;
        }
    }

    /* Python: _uniform_k_decode_conflict. The wavelet load/math split has its
     * own K loop, so the flag would only rename a kernel it does not change. */
    if(s->uniform_k_decode && s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
    {
        snprintf(reason,
                 reason_cap,
                 "uniform_k_decode is not supported with pipeline='wavelet': the "
                 "wavelet load/math split has its own K loop");
        return false;
    }

    /* Python _tap_outer_k_conflict (reasons verbatim): the tap-outer loop
     * has no fallback, so a problem it cannot compute is rejected. */
    if(s->tap_outer_k)
    {
        const int kpg = rocke_conv_problem_kpg(p);
        if(p->sH != 1 || p->sW != 1 || p->dH != 1 || p->dW != 1)
        {
            snprintf(reason,
                     reason_cap,
                     "tap_outer_k needs stride 1 and dilation 1 (the filter tap is the "
                     "plain (y, x) only there)");
            return false;
        }
        if(s->tile_k <= 0 || kpg % s->tile_k != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "tap_outer_k needs kpg %% tile_k == 0 (a K tile must not straddle "
                     "two filter taps); got kpg=%d, tile_k=%d",
                     kpg,
                     s->tile_k);
            return false;
        }
        if(s->split_k != 1)
        {
            snprintf(reason,
                     reason_cap,
                     "tap_outer_k needs split_k == 1 (a K slice could start mid-tap)");
            return false;
        }
        if(s->wave_size != 64 || (s->pipeline && strcmp(s->pipeline, "wavelet") == 0))
        {
            snprintf(reason,
                     reason_cap,
                     "tap_outer_k is a wave64 MFMA K loop (the wavelet pipeline has its "
                     "own K loop)");
            return false;
        }
        if(s->uniform_k_decode)
        {
            snprintf(reason,
                     reason_cap,
                     "tap_outer_k replaces the K loop uniform_k_decode builds; set one "
                     "of them");
            return false;
        }
    }

    /* Python _dy_halo_conflict (reasons verbatim). */
    if(!_dgrad_dy_halo_ok(s, reason, reason_cap))
        return false;

    /* wavelet-specific checks (Python: spec.pipeline == "wavelet" block). */
    bool is_wavelet = (s->pipeline && strcmp(s->pipeline, "wavelet") == 0);
    if(is_wavelet)
    {
        if(s->num_load_waves < 1)
        {
            snprintf(reason, reason_cap, "pipeline='wavelet' requires num_load_waves >= 1");
            return false;
        }
        if(strcmp(family, "wmma") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' is WMMA/gfx1250 only: on MFMA targets "
                     "the single-buffer LDS is overwritten each K iteration and load/math "
                     "waves execute sequentially rather than truly concurrently.");
            return false;
        }
        if(s->async_dma)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' is incompatible with async_dma=True: "
                     "the wavelet loaders are only constructed in the non-async branch "
                     "and a_wavelet_loader/b_wavelet_loader would be None at fetch time.");
            return false;
        }
        int mfmas_m = s->tile_m / (_max(s->warp_m * s->warp_tile_m, 1));
        int mfmas_n = s->tile_n / (_max(s->warp_n * s->warp_tile_n, 1));
        int dg_K = rocke_dgrad_conv_spec_dg_K(s);
        int k_iters = _ceil_div(dg_K, _max(s->tile_k, 1));
        int wmma_cost = k_iters * mfmas_m * mfmas_n;
        const int WMMA_COST_LIMIT = 4096;
        if(wmma_cost > WMMA_COST_LIMIT)
        {
            snprintf(reason,
                     reason_cap,
                     "pipeline='wavelet' unrolled WMMA count %d "
                     "(K_iters=%d x mfmas=%dx%d) exceeds compile-time limit %d; "
                     "reduce tile_k, tile_m, or tile_n",
                     wmma_cost,
                     k_iters,
                     mfmas_m,
                     mfmas_n,
                     WMMA_COST_LIMIT);
            return false;
        }
        int launch_bs = rocke_dgrad_conv_spec_launch_block_size(s);
        if(launch_bs > max_tpb)
        {
            snprintf(reason,
                     reason_cap,
                     "launch_block_size %d > %d (hardware cap) on %s",
                     launch_bs,
                     max_tpb,
                     arch);
            return false;
        }
    }

    /* WMMA-specific restrictions (Python: family == "wmma" block). */
    if(strcmp(family, "wmma") == 0)
    {
        /* gfx1250 supports 16x16x16 and 16x16x32; other WMMA supports only 16x16x16.
         * Both atoms are valid for both "mem" and "wavelet" pipelines. */
        bool atom_ok = (s->warp_tile_m == 16 && s->warp_tile_n == 16
                        && (s->warp_tile_k == 16 || s->warp_tile_k == 32));
        if(!atom_ok)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports 16x16x16 or 16x16x32 (got %dx%dx%d) on %s",
                     s->warp_tile_m,
                     s->warp_tile_n,
                     s->warp_tile_k,
                     arch);
            return false;
        }
        if(strcmp(s->pipeline, "mem") != 0 && strcmp(s->pipeline, "wavelet") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports only 'mem' or 'wavelet' pipeline (got %s) on %s",
                     s->pipeline,
                     arch);
            return false;
        }
        if(strcmp(s->epilogue, "default") != 0 && strcmp(s->epilogue, "cshuffle") != 0)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad supports 'default' and 'cshuffle' epilogues (got %s) on %s",
                     s->epilogue,
                     arch);
            return false;
        }
        bool split_k_bad = (s->split_k > 1 && !is_wavelet);
        if(s->async_dma || s->unroll_k || s->chiplet_swizzle || split_k_bad)
        {
            snprintf(reason,
                     reason_cap,
                     "WMMA dgrad does not support async_dma/unroll_k/chiplet_swizzle"
                     "/split_k>1 (non-wavelet) on %s",
                     arch);
            return false;
        }
    }

    /* K-outer transpose-read gates. Mirrors the DgradConvSpec.validate() and
     * is_valid_dgrad_spec blocks; asymmetric on purpose (dtype_b/warp_tile_n
     * only) because just the B tile flips. */
    if(s->lds_k_outer)
    {
        /* Two regimes: gfx950 wave64 MFMA (ds_read_b64_tr_b16) and gfx1250
         * wave32 WMMA (ds_load_tr16_b128). The arch and the wave must agree or
         * the lane formula addresses a layout the hardware does not implement. */
        const bool k_outer_950 = (strcmp(arch, "gfx950") == 0);
        const bool k_outer_1250 = (strcmp(arch, "gfx1250") == 0);
        if(!k_outer_950 && !k_outer_1250)
        {
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer requires gfx950 or gfx1250 (the LDS transpose "
                     "read); got %s",
                     arch);
            return false;
        }
        const int want_wave = k_outer_950 ? 64 : 32;
        if(s->wave_size != want_wave)
        {
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer on %s requires wave_size=%d; got %d",
                     arch,
                     want_wave,
                     s->wave_size);
            return false;
        }
        if(!(s->dtype_b && (strcmp(s->dtype_b, "bf16") == 0 || strcmp(s->dtype_b, "fp16") == 0)))
        {
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer requires a 16-bit B dtype (ds_read_tr16_b64 is a "
                     "16-bit transpose read); got dtype_b=%s",
                     s->dtype_b ? s->dtype_b : "(null)");
            return false;
        }
        if(s->wave_size == 32)
        {
            /* One atom in the wave32 regime: gfx1250 WMMA 16x16x32. */
            if(s->warp_tile_n != 16 || s->warp_tile_k != 32)
            {
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
        else if(s->warp_tile_n != 16 && s->warp_tile_n != 32)
        {
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer requires warp_tile_n in (16, 32); got %d",
                     s->warp_tile_n);
            return false;
        }
        if(s->lds_layout != NULL)
        {
            snprintf(reason, reason_cap, "lds_k_outer does not honour an explicit lds_layout");
            return false;
        }
        if(s->async_dma)
        {
            snprintf(reason, reason_cap, "lds_k_outer is not supported with async_dma on dgrad");
            return false;
        }
        /* Same shape of problem as async_dma: the alternate load path does not
         * implement the K-outer tile. build_wavelet_loaders pins the B tile to
         * (block_n, block_k) and takes the unswapped descriptor, so it writes
         * M-outer into a K-outer allocation -- wrong row stride for every
         * element, and out of bounds past B_smem when tile_n > tile_k.
         * Matches Python is_valid_dgrad_spec and validate(). */
        if(is_wavelet)
        {
            snprintf(reason,
                     reason_cap,
                     "lds_k_outer is not supported with pipeline='wavelet' on dgrad "
                     "(the wavelet loader writes the B tile M-outer into a K-outer "
                     "allocation)");
            return false;
        }
        /* Python _kouter_frag_reason: the transpose read returns 4 elements
         * per lane on wave64 (ds_read_tr16_b64) and 8 on wave32
         * (ds_load_tr16_b128), so the atom's B fragment length must be a
         * multiple of it. The builder refuses it too; the validator says so
         * first so a sweep filters the configuration out. */
        {
            const rocke_archtarget_t* kt = rocke_archtarget_from_gfx(arch);
            const rocke_mmaop_t* kop = kt ? rocke_archtarget_op_for_shape(kt,
                                                                          _dgrad_mma_family(arch),
                                                                          s->dtype_a,
                                                                          s->dtype_b,
                                                                          "fp32",
                                                                          s->warp_tile_m,
                                                                          s->warp_tile_n,
                                                                          s->warp_tile_k,
                                                                          nullptr)
                                          : NULL;
            const int tr_lanes = s->wave_size == 32 ? 8 : 4;
            if(!kop || kop->b_frag_len % tr_lanes != 0)
            {
                snprintf(reason,
                         reason_cap,
                         "lds_k_outer needs a B fragment length that is a multiple of "
                         "%d (%s returns %d elements per lane on wave%d); got "
                         "b_per_lane=%d for atom %dx%dx%d",
                         tr_lanes,
                         s->wave_size == 32 ? "ds_load_tr16_b128" : "ds_read_tr16_b64",
                         tr_lanes,
                         s->wave_size,
                         kop ? kop->b_frag_len : 0,
                         s->warp_tile_m,
                         s->warp_tile_n,
                         s->warp_tile_k);
                return false;
            }
        }
    }

    /* The dY tile loader (Python: the coalesced_load_reason check at the end
     * of is_valid_dgrad_spec). An explicit vector_size_a is used verbatim at
     * split_k <= 1 and has to split the tile evenly over the block's threads;
     * B's width and the wavelet loaders' are chosen by the loader itself. */
    if(!(s->pipeline && strcmp(s->pipeline, "wavelet") == 0) && s->split_k <= 1
       && s->has_vector_size_a)
    {
        if(!rocke_conv_coalesced_load_ok("A",
                                         s->tile_m,
                                         s->tile_k,
                                         rocke_dgrad_conv_spec_block_size(s),
                                         s->vector_size_a,
                                         reason,
                                         reason_cap))
            return false;
    }

    snprintf(reason, reason_cap, "ok");
    return true;
}

// ---------------------------------------------------------------------------
// Tilde decomposition (Python lines 128-330)
// ---------------------------------------------------------------------------

rocke_tilde_decomposition_t rocke_compute_tilde(const rocke_conv_problem_t* p)
{
    rocke_tilde_decomposition_t t;
    memset(&t, 0, sizeof(t));
    t.gcd_h = _gcd(p->sH > 0 ? p->sH : 1, p->dH > 0 ? p->dH : 1);
    t.gcd_w = _gcd(p->sW > 0 ? p->sW : 1, p->dW > 0 ? p->dW : 1);
    int sH = p->sH > 0 ? p->sH : 1;
    int sW = p->sW > 0 ? p->sW : 1;
    int dH = p->dH > 0 ? p->dH : 1;
    int dW = p->dW > 0 ? p->dW : 1;
    t.y_tilde = sH / t.gcd_h;
    t.x_tilde = sW / t.gcd_w;
    t.y_dot = _ceil_div(p->Y, t.y_tilde);
    t.x_dot = _ceil_div(p->X, t.x_tilde);
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);
    t.h_tilde = p->Y > 1 ? Ho + _ceil_div(dH * (p->Y - 1), sH) : Ho;
    t.w_tilde = p->X > 1 ? Wo + _ceil_div(dW * (p->X - 1), sW) : Wo;
    return t;
}

int rocke_enumerate_sub_gemms(const rocke_conv_problem_t* p,
                              const rocke_tilde_decomposition_t* tilde,
                              int tile_m,
                              int tile_n,
                              int tile_k,
                              int split_k,
                              rocke_sub_gemm_params_t* out,
                              int out_cap)
{
    int count = 0;
    int cumulative_tiles = 0;
    int sH = p->sH > 0 ? p->sH : 1;
    int sW = p->sW > 0 ? p->sW : 1;
    int dH = p->dH > 0 ? p->dH : 1;
    int dW = p->dW > 0 ? p->dW : 1;
    /* Per group (Python compute_sub_gemms): every group shares one record
     * table, its own group rides blockIdx.y. */
    int cpg = rocke_conv_problem_cpg(p);
    int kpg = rocke_conv_problem_kpg(p);

    for(int i_yt = 0; i_yt < tilde->y_tilde; i_yt++)
    {
        for(int i_xt = 0; i_xt < tilde->x_tilde; i_xt++)
        {
            int y_dot_slice = _ceil_div(p->Y - i_yt, tilde->y_tilde);
            int x_dot_slice = _ceil_div(p->X - i_xt, tilde->x_tilde);
            if(y_dot_slice <= 0 || x_dot_slice <= 0)
                continue;

            int h_tsb = _floor_div(_max(0, p->pH - dH * (tilde->y_tilde - 1)), sH);
            int h_tse = _min(tilde->h_tilde, _ceil_div(p->pH + p->Hi - 1, sH) + 1);
            int w_tsb = _floor_div(_max(0, p->pW - dW * (tilde->x_tilde - 1)), sW);
            int w_tse = _min(tilde->w_tilde, _ceil_div(p->pW + p->Wi - 1, sW) + 1);

            int h_ts = h_tse - h_tsb;
            int w_ts = w_tse - w_tsb;
            if(h_ts <= 0 || w_ts <= 0)
                continue;

            int gemm_m = p->N * h_ts * w_ts;
            int gemm_n = cpg;
            int gemm_k = y_dot_slice * x_dot_slice * kpg;
            int num_m_tiles = _ceil_div(gemm_m, tile_m);
            int num_n_tiles = _ceil_div(gemm_n, tile_n);
            int num_tiles = num_m_tiles * num_n_tiles;
            int block_start = cumulative_tiles;
            int block_end = cumulative_tiles + num_tiles;
            cumulative_tiles = block_end;

            int sk = split_k > 1 ? split_k : 1;
            int stride_k = tile_k * sk;
            int gemm_k_padded = _ceil_div(gemm_k, stride_k) * stride_k;

            if(count < out_cap)
            {
                rocke_sub_gemm_params_t* sg = &out[count];
                memset(sg, 0, sizeof(*sg));
                sg->i_ytilde = i_yt;
                sg->i_xtilde = i_xt;
                sg->y_dot_slice = y_dot_slice;
                sg->x_dot_slice = x_dot_slice;
                sg->h_tilde_slice_begin = h_tsb;
                sg->h_tilde_slice = h_ts;
                sg->w_tilde_slice_begin = w_tsb;
                sg->w_tilde_slice = w_ts;
                sg->gemm_m = gemm_m;
                sg->gemm_n = gemm_n;
                sg->gemm_k = gemm_k;
                sg->block_start = block_start;
                sg->block_end = block_end;
                sg->a_embed_h_coeff = -(dH / tilde->gcd_h);
                sg->a_embed_w_coeff = -(dW / tilde->gcd_w);
                sg->b_y_stride = tilde->y_tilde;
                sg->b_y_offset = i_yt;
                sg->b_x_stride = tilde->x_tilde;
                sg->b_x_offset = i_xt;
                sg->d_h_stride = sH;
                sg->d_h_offset = dH * i_yt + sH * h_tsb - p->pH;
                sg->d_w_stride = sW;
                sg->d_w_offset = dW * i_xt + sW * w_tsb - p->pW;
                sg->gemm_k_padded = gemm_k_padded;
            }
            count++;
        }
    }
    return count;
}

int rocke_pack_sub_gemm_buffer(const rocke_sub_gemm_params_t* sgs,
                               int count,
                               int tile_m,
                               int tile_n,
                               int* out_buf,
                               int out_cap)
{
    int total = count * (ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS + ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS);
    if(total > out_cap)
        return 0;
    for(int i = 0; i < count; i++)
    {
        const rocke_sub_gemm_params_t* sg = &sgs[i];
        int* rec = &out_buf[i * ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS];
        rec[0] = sg->block_start;
        rec[1] = _ceil_div(sg->gemm_m, tile_m);
        rec[2] = _ceil_div(sg->gemm_n, tile_n);
        rec[3] = sg->gemm_m;
        rec[4] = sg->gemm_k;
        rec[5] = sg->h_tilde_slice;
        rec[6] = sg->w_tilde_slice;
        rec[7] = sg->h_tilde_slice_begin;
        rec[8] = sg->w_tilde_slice_begin;
        rec[9] = sg->y_dot_slice;
        rec[10] = sg->x_dot_slice;
        rec[11] = sg->a_embed_h_coeff;
        rec[12] = sg->a_embed_w_coeff;
        rec[13] = sg->b_y_stride;
        rec[14] = sg->b_y_offset;
        rec[15] = sg->b_x_stride;
        rec[16] = sg->b_x_offset;
        rec[17] = sg->d_h_stride;
        rec[18] = sg->d_h_offset;
        rec[19] = sg->d_w_stride;
        rec[20] = sg->d_w_offset;
        rec[21] = sg->gemm_k_padded;
    }
    /* Magic-division table after the records (Python sub_gemm_magic_pairs):
     * one (mult, shift) pair per divisor -- kpg, x_dot_slice,
     * h_tilde_slice*w_tilde_slice, w_tilde_slice. A non-positive divisor here
     * means a corrupt record; reject the whole buffer (Python twin:
     * _magic_i32_pair raising via calculate_magic_numbers) instead of
     * clamping it into a plausible-looking table. */
    for(int i = 0; i < count; i++)
    {
        const rocke_sub_gemm_params_t* sg = &sgs[i];
        if(sg->y_dot_slice <= 0 || sg->x_dot_slice <= 0)
            return 0;
        int* mag = &out_buf[count * ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS
                            + i * ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS];
        int divisors[4] = {sg->gemm_k / (sg->y_dot_slice * sg->x_dot_slice),
                           sg->x_dot_slice,
                           sg->h_tilde_slice * sg->w_tilde_slice,
                           sg->w_tilde_slice};
        for(int j = 0; j < 4; j++)
        {
            uint64_t mult = 0;
            int shift = 0;
            if(!rocke_calculate_magic_numbers(NULL, divisors[j], &mult, &shift))
                return 0;
            mag[2 * j] = (int)(uint32_t)mult;
            mag[2 * j + 1] = shift;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// Descriptor builders (Python lines 332-450)
// ---------------------------------------------------------------------------

struct rocke_tensor_descriptor* rocke_dgrad_make_dy_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype)
{
    (void)dtype; /* descriptor pins f16 internally */
    int Ho = rocke_conv_problem_ho(p);
    int Wo = rocke_conv_problem_wo(p);

    const int lengths[] = {p->N, Ho, Wo, p->K};
    const char* coord_names[] = {"n", "ho", "wo", "k_out"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dY_nhwk", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    /* unmerge('m' -> [n, hi, wi], dims=[N, Hi, Wi]) */
    const char* m_into[] = {"n", "hi", "wi"};
    const int m_dims[] = {p->N, p->Hi, p->Wi};
    rocke_transform_t* t_um = rocke_unmerge_magic(b, "m", m_into, 3, m_dims);

    /* embed([hi, y] -> ho, strides=[1, -1], offset=pH, lo=0, hi=Ho) */
    const char* hi_y[] = {"hi", "y"};
    const int hi_strides[] = {1, -1};
    rocke_transform_t* t_eh = rocke_embed_bounded(b, hi_y, 2, "ho", hi_strides, p->pH, 0, Ho);

    /* embed([wi, x] -> wo, strides=[1, -1], offset=pW, lo=0, hi=Wo) */
    const char* wi_x[] = {"wi", "x"};
    const int wi_strides[] = {1, -1};
    rocke_transform_t* t_ew = rocke_embed_bounded(b, wi_x, 2, "wo", wi_strides, p->pW, 0, Wo);

    /* unmerge('k_dg' -> [k_out, y, x], dims=[K, Y, X])  -- k_out outermost for split-K */
    const char* k_into[] = {"k_out", "y", "x"};
    const int k_dims[] = {p->K, p->Y, p->X};
    rocke_transform_t* t_uk = rocke_unmerge_magic(b, "k_dg", k_into, 3, k_dims);

    /* pad k_out, y, x */
    rocke_transform_t* t_pk = rocke_pad(b, "k_out", 0, p->K);
    rocke_transform_t* t_py = rocke_pad(b, "y", 0, p->Y);
    rocke_transform_t* t_px = rocke_pad(b, "x", 0, p->X);

    const rocke_transform_t* chain[] = {t_um, t_eh, t_ew, t_uk, t_pk, t_py, t_px};
    return rocke_tensor_descriptor_transform(b, desc, chain, 7);
}

struct rocke_tensor_descriptor* rocke_dgrad_make_w_descriptor(rocke_ir_builder_t* b,
                                                              const rocke_conv_problem_t* p,
                                                              const char* dtype)
{
    (void)dtype;
    const int lengths[] = {p->K, p->Y, p->X, p->C};
    const char* coord_names[] = {"k_out", "y", "x", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "W_kyxc", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    const char* k_into[] = {"k_out", "y", "x"};
    const int k_dims[] = {p->K, p->Y, p->X};
    rocke_transform_t* t_uk = rocke_unmerge_magic(b, "k_dg", k_into, 3, k_dims);
    rocke_transform_t* t_pk = rocke_pad(b, "k_out", 0, p->K);
    rocke_transform_t* t_py = rocke_pad(b, "y", 0, p->Y);
    rocke_transform_t* t_px = rocke_pad(b, "x", 0, p->X);

    const rocke_transform_t* chain[] = {t_uk, t_pk, t_py, t_px};
    return rocke_tensor_descriptor_transform(b, desc, chain, 4);
}

/* AOT counterpart of rocke_dgrad_make_dx_descriptor: (m, c) -> NHWC.
 *
 * m -> (n, hi, wi) peels one extent at a time, so it divides by Wi and then
 * by Hi; the host supplies a magic pair per extent. Mirrors Python
 * make_dgrad_dx_descriptor_dynamic. */
static struct rocke_dynamic_tensor_descriptor*
    _dgrad_make_dx_descriptor_dynamic(rocke_ir_builder_t* b, const rocke_conv_build_ctx_t* ctx)
{
    const char* coord_names[4] = {"n", "hi", "wi", "c"};
    rocke_value_t* strides[4];
    rocke_dynamic_tensor_descriptor_t* desc;

    strides[0] = ctx->p_dX_stride_n;
    strides[1] = ctx->p_dX_stride_hi;
    strides[2] = ctx->p_dX_stride_wi;
    strides[3] = rocke_b_const_i32(b, 1);

    desc = rocke_tensor_descriptor_naive_dynamic(b, "dX_nhwc", coord_names, 4, strides);
    if(!desc)
        return NULL;

    {
        const char* into_m[3] = {"n", "hi", "wi"};
        rocke_magic_triple_t trips[2];
        const rocke_transform_t* xforms[1];
        rocke_tensor_descriptor_t* chained;

        trips[0].mult = ctx->p_magic_m_Hi_mult;
        trips[0].shift = ctx->p_magic_m_Hi_shift;
        trips[0].dim = ctx->p_Hi;
        trips[1].mult = ctx->p_magic_m_Wi_mult;
        trips[1].shift = ctx->p_magic_m_Wi_shift;
        trips[1].dim = ctx->p_Wi;

        xforms[0] = rocke_unmerge_magic_dynamic(b, "m", into_m, 3, trips);
        if(!xforms[0])
            return NULL;
        chained = rocke_tensor_descriptor_transform(b, &desc->base, xforms, 1);
        if(!chained)
            return NULL;
        desc->base = *chained;
    }
    return desc;
}

struct rocke_tensor_descriptor* rocke_dgrad_make_dx_descriptor(rocke_ir_builder_t* b,
                                                               const rocke_conv_problem_t* p,
                                                               const char* dtype)
{
    (void)dtype;
    const int lengths[] = {p->N, p->Hi, p->Wi, p->C};
    const char* coord_names[] = {"n", "hi", "wi", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dX_nhwc", lengths, 4, NULL, coord_names, 4);
    if(!desc)
        return NULL;

    const char* m_into[] = {"n", "hi", "wi"};
    const int m_dims[] = {p->N, p->Hi, p->Wi};
    rocke_transform_t* t_um = rocke_unmerge_magic(b, "m", m_into, 3, m_dims);

    const rocke_transform_t* chain[] = {t_um};
    return rocke_tensor_descriptor_transform(b, desc, chain, 1);
}

// ===========================================================================
// MMA resolution (mirrors _dgrad_mma_family / _resolve_dgrad_op)
// ===========================================================================

static const char* _dgrad_mma_family(const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target && target->wave_size == 32)
        return "wmma";
    return "mma";
}

static const rocke_mmaop_t*
    _resolve_dgrad_op(rocke_ir_builder_t* b, const rocke_dgrad_conv_spec_t* spec, const char* arch)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(!target)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad: unknown arch '%s'", arch);
        return NULL;
    }
    const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(target,
                                                            _dgrad_mma_family(arch),
                                                            spec->dtype_a,
                                                            spec->dtype_b,
                                                            "fp32",
                                                            spec->warp_tile_m,
                                                            spec->warp_tile_n,
                                                            spec->warp_tile_k,
                                                            nullptr);
    if(!op)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no MMA atom for dgrad warp_tile (%d,%d,%d) on %s",
                        spec->warp_tile_m,
                        spec->warp_tile_n,
                        spec->warp_tile_k,
                        arch);
        return NULL;
    }
    return op;
}

// ===========================================================================
// Direct epilogue helpers
// ===========================================================================

struct dgrad_dx_addr_ctx
{
    rocke_tensor_descriptor_t* desc;
    /* g*cpg, the conv group's first dX channel; NULL when ungrouped. */
    rocke_value_t* c_group_base;
};

/* Max dX store width -- mirrors Python _dgrad_store_vec. An explicit
 * vector_size_c is used as given; otherwise the widest of 8/4/2/1 (4/2/1 for
 * an fp32 dX) dividing the per-group channel run cpg (C when ungrouped), the
 * stride-1 run a store may cover without crossing into the next group. */
static int _dgrad_store_vec(const rocke_dgrad_conv_spec_t* spec)
{
    if(spec->has_vector_size_c)
        return spec->vector_size_c;
    const int cpg = rocke_conv_problem_cpg(&spec->problem);
    const bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    const int cand[] = {is_fp32_d ? 4 : 8, 4, 2, 1};
    for(int v : cand)
        if(cpg % v == 0)
            return v;
    return 1;
}

static rocke_value_t* _dgrad_dx_addr_fn(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    struct dgrad_dx_addr_ctx* ctx = (struct dgrad_dx_addr_ctx*)user;
    /* n_global is the group-local input channel (< cpg); the absolute NHWC
     * channel is g*cpg + n_global. Ungrouped: no add, c = n_global. */
    rocke_value_t* c_abs
        = ctx->c_group_base ? rocke_b_add(b, n_global, ctx->c_group_base) : n_global;
    const char* names[] = {"m", "c"};
    rocke_value_t* vals[] = {m_global, c_abs};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, ctx->desc, names, vals, 2, &off, &valid);
    if(out_valid)
        *out_valid = valid;
    return off;
}

static void _emit_dgrad_direct_epilogue(rocke_ir_builder_t* b,
                                        const rocke_dgrad_conv_spec_t* spec,
                                        const rocke_conv_build_ctx_t* params,
                                        rocke_value_t* const* accs,
                                        int num_accs,
                                        const rocke_warp_grid_t* grid,
                                        rocke_value_t* dx_rsrc,
                                        rocke_value_t* c_group_base)
{
    rocke_dynamic_tensor_descriptor_t* dX_desc = _dgrad_make_dx_descriptor_dynamic(b, params);
    if(!dX_desc)
        return;

    struct dgrad_dx_addr_ctx addr_ctx;
    addr_ctx.desc = &dX_desc->base;
    addr_ctx.c_group_base = c_group_base;

    rocke_direct_epilogue_t epi;
    epi.atom
        = rocke_mfma_atom(spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;

    /* AOT: the GEMM extents are kernargs. */
    rocke_value_t* bounds_m = params->p_dg_M;
    rocke_value_t* bounds_n = params->p_dg_N;

    rocke_direct_epilogue_store(
        b, &epi, accs, num_accs, _dgrad_dx_addr_fn, &addr_ctx, dx_rsrc, bounds_m, bounds_n, false);
}

// ===========================================================================
// MFMA cshuffle epilogue for stride=1 (Python _emit_dgrad_cshuffle_epilogue)
// ===========================================================================

static void _emit_dgrad_cshuffle_epilogue(rocke_ir_builder_t* b,
                                          const rocke_dgrad_conv_spec_t* spec,
                                          const rocke_conv_build_ctx_t* params,
                                          rocke_value_t* const* accs,
                                          int num_accs,
                                          const rocke_warp_grid_t* grid,
                                          rocke_value_t* dx_rsrc,
                                          rocke_value_t* c_group_base)
{
    rocke_dynamic_tensor_descriptor_t* dX_desc = _dgrad_make_dx_descriptor_dynamic(b, params);
    if(!dX_desc)
        return;

    struct dgrad_dx_addr_ctx addr_ctx;
    addr_ctx.desc = &dX_desc->base;
    addr_ctx.c_group_base = c_group_base;

    int max_store_vec = _dgrad_store_vec(spec);

    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom(spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    epi.out_dtype = spec->dtype_d;

    /* AOT: the GEMM extents are kernargs. */
    rocke_value_t* bounds_m = params->p_dg_M;
    rocke_value_t* bounds_n = params->p_dg_N;

    rocke_cshuffle_epilogue_store(
        b, &epi, accs, num_accs, _dgrad_dx_addr_fn, &addr_ctx, dx_rsrc, bounds_m, bounds_n);
}

// ===========================================================================
// WMMA direct epilogue (Python _emit_dgrad_direct_epilogue_wmma)
// ===========================================================================

static void _emit_dgrad_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                             const rocke_dgrad_conv_spec_t* spec,
                                             const rocke_conv_build_ctx_t* params,
                                             const rocke_mmaop_t* op,
                                             rocke_value_t* const* accs,
                                             int num_accs,
                                             rocke_value_t* warp_m_idx,
                                             rocke_value_t* warp_n_idx,
                                             rocke_value_t* lane,
                                             rocke_value_t* block_m_off,
                                             rocke_value_t* block_n_off,
                                             rocke_value_t* dx_rsrc,
                                             rocke_value_t* c0,
                                             rocke_value_t* c_group_base)
{
    (void)num_accs;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);

    bool is_fp32_out = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16_out = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);
    int elem_bytes = is_fp32_out ? 4 : 2;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    /* AOT: the GEMM extents are kernargs. */
    rocke_value_t* c_M = params->p_dg_M;
    rocke_value_t* c_N = params->p_dg_N;

    rocke_dynamic_tensor_descriptor_t* dX_dyn = _dgrad_make_dx_descriptor_dynamic(b, params);
    if(!dX_dyn)
        return;
    rocke_tensor_descriptor_t* dX_desc = &dX_dyn->base;

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            rocke_value_t* _bwm = rocke_b_add(b, block_m_off, warp_m_off);
            rocke_value_t* atom_m_off
                = rocke_b_add(b, _bwm, rocke_b_const_i32(b, mi * spec->warp_tile_m));
            rocke_value_t* _bwn = rocke_b_add(b, block_n_off, warp_n_off);
            rocke_value_t* atom_n_off
                = rocke_b_add(b, _bwn, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);
            for(int i = 0; i < op->c_frag_len; i++)
            {
                rocke_value_t* row_off;
                rocke_value_t* col_off;
                rocke_arch_layout_map_coord(c_map, b, lane, i, &row_off, &col_off);

                rocke_value_t* m_val = rocke_b_add(b, atom_m_off, row_off);
                rocke_value_t* n_val = rocke_b_add(b, atom_n_off, col_off);
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, c_M);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_val, c_N);
                rocke_value_t* ok = rocke_b_land(b, m_ok, n_ok);

                rocke_value_t* v_f32 = rocke_b_vec_extract(b, acc, i);
                /* Absolute NHWC channel g*cpg + n_val; the bound stays local. */
                rocke_value_t* c_abs = c_group_base ? rocke_b_add(b, n_val, c_group_base) : n_val;
                const char* dx_names[] = {"m", "c"};
                rocke_value_t* dx_vals[] = {m_val, c_abs};
                rocke_value_t* dx_off_elems = NULL;
                rocke_value_t* dx_valid = NULL;
                rocke_transforms_descriptor_offset(
                    b, dX_desc, dx_names, dx_vals, 2, &dx_off_elems, &dx_valid);
                rocke_value_t* dx_off_bytes
                    = rocke_b_mul(b, dx_off_elems, rocke_b_const_i32(b, elem_bytes));
                rocke_value_t* safe_off
                    = rocke_b_select(b, ok, dx_off_bytes, rocke_b_const_i32(b, 0x7FFFFFFF));

                if(is_fp32_out)
                    rocke_b_buffer_store_f32(b, dx_rsrc, safe_off, c0, v_f32);
                else if(is_bf16_out)
                    rocke_b_buffer_store_bf16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                else
                    rocke_b_buffer_store_f16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
            }
        }
    }
}

// ===========================================================================
// LDS layout for dgrad (mirrors spec.effective_lds_layout())
// ===========================================================================

static rocke_conv_lds_layout_t _dgrad_effective_lds_layout(const rocke_dgrad_conv_spec_t* spec)
{
    rocke_conv_lds_layout_t l;
    memset(&l, 0, sizeof(l));
    l.logical_cols = spec->tile_k;
    if(spec->has_lds_k_pad)
        l.k_pad = spec->lds_k_pad;
    else
        l.k_pad = (spec->tile_k >= 16) ? 8 : 0;
    if(spec->async_dma)
        l.k_pad = 0;
    l.row_stride = l.logical_cols + l.k_pad;
    l.swizzle = NULL;
    l.requires_packed_async = false;
    return l;
}

// ===========================================================================
// Tilde binary search (Python _emit_binary_search, lines 1324-1352)
// ===========================================================================

/* AOT: the record count is the runtime num_sub_gemms kernarg -- the tilde
 * decomposition produces a different count per stride/dilation, so it cannot
 * be folded in. So is the trip count: each halving leaves at most
 * ceil(size / 2) records, so ceil(log2(n)) of them bracket n records, and a
 * uniform loop runs exactly that many. A stride-1 problem (one record) skips
 * the search outright instead of waiting out a chain of dependent scalar
 * loads sized for the largest decomposition. max_sub_gemms only bounds the
 * trip-count computation. */
static rocke_value_t* _emit_binary_search(rocke_ir_builder_t* b,
                                          rocke_value_t* flat_block_id,
                                          rocke_value_t* sub_gemm_buf,
                                          rocke_value_t* num_sub_gemms_val,
                                          int max_sub_gemms)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_record_stride = rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS);

    /* ceil(log2(n)) = the number of powers of two 2**t (t >= 0) below n. */
    rocke_value_t* n_iters = c0;
    int max_steps = (int)ceil(log2(_max(max_sub_gemms, 2)));
    for(int t = 0; t < max_steps; t++)
    {
        rocke_value_t* below = rocke_b_cmp_lt(b, rocke_b_const_i32(b, 1 << t), num_sub_gemms_val);
        n_iters = rocke_b_add(b, n_iters, rocke_b_select(b, below, c1, c0));
    }

    rocke_iter_arg_t search_args[2];
    search_args[0].name = "bs_lo";
    search_args[0].init = c0;
    search_args[1].name = "bs_hi";
    search_args[1].init = num_sub_gemms_val;
    rocke_for_t search
        = rocke_b_scf_for_iter(b, c0, n_iters, c1, search_args, 2, "bs_it", false, true);
    rocke_b_region_enter(b, search.body);
    {
        rocke_value_t* lo = search.iter_vars[0];
        rocke_value_t* hi = search.iter_vars[1];
        rocke_value_t* _mid_sum = rocke_b_add(b, lo, hi);
        rocke_value_t* mid = rocke_b_div(b, _mid_sum, rocke_b_const_i32(b, 2));
        rocke_value_t* mid_off = rocke_b_mul(b, mid, c_record_stride);
        rocke_value_t* mid_block_start = rocke_b_global_load_i32(b, sub_gemm_buf, mid_off, 4);
        rocke_value_t* take_lo = rocke_b_cmp_le(b, mid_block_start, flat_block_id);
        rocke_value_t* next[2];
        next[0] = rocke_b_select(b, take_lo, mid, lo);
        next[1] = rocke_b_select(b, take_lo, hi, mid);
        rocke_b_scf_yield(b, next, 2);
    }
    rocke_b_region_leave(b);
    return search.op->results[0];
}

static rocke_value_t* _emit_load_record_field(rocke_ir_builder_t* b,
                                              rocke_value_t* sub_gemm_buf,
                                              rocke_value_t* sg_idx,
                                              int field_idx)
{
    rocke_value_t* base
        = rocke_b_mul(b, sg_idx, rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS));
    rocke_value_t* offset = rocke_b_add(b, base, rocke_b_const_i32(b, field_idx));
    return rocke_b_global_load_i32(b, sub_gemm_buf, offset, 4);
}

/* A runtime divisor with its host-computed magic-division pair (Python
 * _MagicDiv). */
struct dgrad_magic_div_t
{
    rocke_value_t* value;
    rocke_value_t* mult;
    rocke_value_t* shift;
};

/* Python _magic_divmod: (n // d, n % d) through d's magic pair. n must be
 * non-negative and below 2**31 (the magic-division range). */
static void _magic_divmod(rocke_ir_builder_t* b,
                          rocke_value_t* n,
                          const dgrad_magic_div_t* d,
                          rocke_value_t** q_out,
                          rocke_value_t** r_out)
{
    rocke_value_t* tmp = rocke_b_umul_hi_i32(b, n, d->mult);
    rocke_value_t* summed = rocke_b_add_nuw(b, tmp, n);
    rocke_value_t* q = rocke_b_lshr(b, summed, d->shift);
    rocke_value_t* qd = rocke_b_mul(b, q, d->value);
    *q_out = q;
    *r_out = rocke_b_sub(b, n, qd);
}

// ===========================================================================
// Tilde atomic epilogue (Python _emit_dgrad_tilde_atomic_epilogue)
// ===========================================================================

static void _emit_dgrad_tilde_atomic_epilogue(rocke_ir_builder_t* b,
                                              const rocke_dgrad_conv_spec_t* spec,
                                              const rocke_mfma_atom_t* atom,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* warp_m_idx,
                                              rocke_value_t* warp_n_idx,
                                              rocke_value_t* lane,
                                              rocke_value_t* block_m_off,
                                              rocke_value_t* block_n_off,
                                              rocke_value_t* dx_ptr,
                                              int c_per_lane,
                                              rocke_value_t* gemm_m,
                                              rocke_value_t* gemm_n,
                                              const dgrad_magic_div_t* hw_tilde,
                                              const dgrad_magic_div_t* w_tilde_slice,
                                              rocke_value_t* d_h_stride,
                                              rocke_value_t* d_h_offset,
                                              rocke_value_t* d_w_stride,
                                              rocke_value_t* d_w_offset,
                                              rocke_value_t* c_Hi,
                                              rocke_value_t* c_Wi,
                                              rocke_value_t* c_C,
                                              rocke_value_t* c_group_base)
{
    (void)num_accs;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    bool is_fp32 = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16 = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));
    rocke_value_t* block_warp_m_off = rocke_b_add(b, block_m_off, warp_m_off);
    rocke_value_t* block_warp_n_off = rocke_b_add(b, block_n_off, warp_n_off);

    int kc_m0, kc_mlane, kc_m1, kc_nlane;
    rocke_c_warp_params(atom, &kc_m0, &kc_mlane, &kc_m1, &kc_nlane);

    rocke_tile_distribution_encoding_t* enc = rocke_make_c_warp_dstr_encoding(b, atom);
    const rocke_tile_distribution_t* dist = rocke_make_static_tile_distribution(b, enc);

    rocke_value_t* c_nlane = rocke_b_const_i32(b, kc_nlane);
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, c_nlane);
    rocke_value_t* m_blk = rocke_b_div(b, lane, c_nlane);

    rocke_value_t* rows[ROCKE_CONV_MAX_ACCS];
    rocke_value_t* cols[ROCKE_CONV_MAX_ACCS];
    for(int i = 0; i < c_per_lane; i++)
    {
        rocke_value_t* ys[2] = {rocke_b_const_i32(b, i / kc_m1), rocke_b_const_i32(b, i % kc_m1)};
        rocke_value_t* p_lane_arr[2] = {m_blk, n_in_atom};
        rocke_value_t* const* ps_arr[1] = {p_lane_arr};
        int ps_counts[1] = {2};
        rocke_value_t* x_out[2];
        rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps_arr, ps_counts, 1, x_out, 2);
        rows[i] = x_out[0];
        cols[i] = x_out[1];
    }

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        rocke_value_t* atom_m_base
            = rocke_b_add(b, block_warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            rocke_value_t* atom_n_base
                = rocke_b_add(b, block_warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < c_per_lane; i++)
            {
                rocke_value_t* c_m = rocke_b_add(b, atom_m_base, rows[i]);
                rocke_value_t* c_n = rocke_b_add(b, atom_n_base, cols[i]);
                /* Absolute dX channel = g*cpg + c_n (the group base is even, so
                 * the packed <2 x dtype> pair parity below is unchanged); the
                 * bound uses the group-local c_n (< gemm_n = cpg). */
                rocke_value_t* c_n_off = c_group_base ? rocke_b_add(b, c_n, c_group_base) : c_n;

                // Decompose c_m -> (n, htl, wtl)
                rocke_value_t *n_val, *m_rem, *htl, *wtl;
                _magic_divmod(b, c_m, hw_tilde, &n_val, &m_rem);
                _magic_divmod(b, m_rem, w_tilde_slice, &htl, &wtl);

                // hi = htl * d_h_stride + d_h_offset
                rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, d_h_stride), d_h_offset);
                rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, d_w_stride), d_w_offset);

                // Bounds: c_m < gemm_m, c_n < gemm_n, 0<=hi<Hi, 0<=wi<Wi
                rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, gemm_m);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, gemm_n);
                rocke_value_t* hi_ge = rocke_b_cmp_ge(b, hi, c0);
                rocke_value_t* hi_lt = rocke_b_cmp_lt(b, hi, c_Hi);
                rocke_value_t* hi_ok = rocke_b_land(b, hi_ge, hi_lt);
                rocke_value_t* wi_ge = rocke_b_cmp_ge(b, wi, c0);
                rocke_value_t* wi_lt = rocke_b_cmp_lt(b, wi, c_Wi);
                rocke_value_t* wi_ok = rocke_b_land(b, wi_ge, wi_lt);
                rocke_value_t* _mn_ok = rocke_b_land(b, m_ok, n_ok);
                rocke_value_t* _hw_ok = rocke_b_land(b, hi_ok, wi_ok);
                rocke_value_t* ok = rocke_b_land(b, _mn_ok, _hw_ok);

                // NHWC offset: ((n * Hi + hi) * Wi + wi) * C + c
                rocke_value_t* _nhwc0 = rocke_b_mul(b, n_val, c_Hi);
                rocke_value_t* _nhwc1 = rocke_b_add(b, _nhwc0, hi);
                rocke_value_t* _nhwc2 = rocke_b_mul(b, _nhwc1, c_Wi);
                rocke_value_t* _nhwc3 = rocke_b_add(b, _nhwc2, wi);
                rocke_value_t* _nhwc4 = rocke_b_mul(b, _nhwc3, c_C);
                rocke_value_t* dx_offset = rocke_b_add(b, _nhwc4, c_n_off);

                rocke_value_t* val_f32 = rocke_b_vec_extract(b, acc, i);
                rocke_if_t if_ok = rocke_b_scf_if(b, ok);
                rocke_b_region_enter(b, if_ok.then_region);
                {
                    if(is_fp32)
                    {
                        rocke_b_global_atomic_add(b, dx_ptr, dx_offset, val_f32, NULL);
                    }
                    else
                    {
                        rocke_value_t* val_cvt = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, val_f32)
                                                         : rocke_b_trunc_f32_to_f16(b, val_f32);
                        rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);
                        rocke_value_t* zero_cvt = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32)
                                                          : rocke_b_trunc_f32_to_f16(b, zero_f32);
                        rocke_value_t* c_n_is_odd
                            = rocke_b_mod(b, c_n_off, rocke_b_const_i32(b, 2));
                        rocke_value_t* is_odd
                            = rocke_b_cmp_ne(b, c_n_is_odd, rocke_b_const_i32(b, 0));
                        rocke_value_t* c_n_even = rocke_b_sub(b, c_n_off, c_n_is_odd);
                        rocke_value_t* _oe0 = rocke_b_mul(b, n_val, c_Hi);
                        rocke_value_t* _oe1 = rocke_b_add(b, _oe0, hi);
                        rocke_value_t* _oe2 = rocke_b_mul(b, _oe1, c_Wi);
                        rocke_value_t* _oe3 = rocke_b_add(b, _oe2, wi);
                        rocke_value_t* _oe4 = rocke_b_mul(b, _oe3, c_C);
                        rocke_value_t* off_even = rocke_b_add(b, _oe4, c_n_even);
                        rocke_value_t* v_even = rocke_b_select(b, is_odd, zero_cvt, val_cvt);
                        rocke_value_t* v_odd = rocke_b_select(b, is_odd, val_cvt, zero_cvt);
                        rocke_value_t* comps[2] = {v_even, v_odd};
                        rocke_value_t* vec = rocke_b_vec_pack(b, comps, 2, val_cvt->type);
                        if(is_bf16)
                            rocke_b_global_atomic_add_pk_bf16(b, dx_ptr, off_even, vec, NULL);
                        else
                            rocke_b_global_atomic_add_pk_f16(b, dx_ptr, off_even, vec, NULL);
                    }
                }
                rocke_b_region_leave(b);
            }
        }
    }
}

// ===========================================================================
// Tilde dgrad kernel builder (Python _build_tilde_dgrad, lines 1366-1757)
// ===========================================================================

// ===========================================================================
// Tilde descriptor closure context types + callbacks
// ===========================================================================

struct tilde_dy_ctx_t
{
    rocke_value_t* block_m_off;
    rocke_value_t* k_off;
    dgrad_magic_div_t kpg_div; // k_sub -> (yx_rem, k_out)
    dgrad_magic_div_t xdot_div; // yx_rem -> (ydot, xdot)
    dgrad_magic_div_t hw_div; // m_sub -> (n, m_rem)
    dgrad_magic_div_t w_div; // m_rem -> (htl, wtl)
    rocke_value_t* rec_h_tilde_slice_begin;
    rocke_value_t* rec_w_tilde_slice_begin;
    rocke_value_t* rec_a_embed_h_coeff;
    rocke_value_t* rec_a_embed_w_coeff;
    rocke_value_t* c_Ho;
    rocke_value_t* c_Wo;
    rocke_value_t* c_K; // total K -- the NHWK row stride
    /* Innermost divisor of the k_dg decode: kpg when grouped (the reduction is
     * per group), c_K otherwise. */
    rocke_value_t* c_kdiv;
    /* g*kpg, the conv group's first output channel; NULL when ungrouped. */
    rocke_value_t* k_out_group_base;
    rocke_value_t* c0;
    /* Pointwise (Y=X=1, stride 1, pad 0, ungrouped) fast path -- mirrors the
     * Python dy_descriptor. dg_M is N*Ho*Wo, materialised inside the descriptor
     * at the same point Python creates it so the IR order matches. */
    bool is_pointwise;
    /* AOT: N*Ho*Wo as an SSA value, materialised once in the builder
     * prologue (Python emits it there too, so the SSA order matches). */
    rocke_value_t* dg_M;
};

static rocke_value_t* _tilde_dy_descriptor(rocke_ir_builder_t* b_,
                                           rocke_value_t* row,
                                           rocke_value_t* col,
                                           rocke_value_t** out_valid,
                                           void* user)
{
    tilde_dy_ctx_t* ctx = (tilde_dy_ctx_t*)user;
    rocke_value_t* m_sub = rocke_b_add(b_, ctx->block_m_off, row);
    rocke_value_t* k_sub = rocke_b_add(b_, ctx->k_off, col);

    // Pointwise (Y=X=1, stride 1, pad 0, ungrouped) fast path. The tilde
    // decomposition is the identity here, so the offset reduces exactly to
    // m_sub*K + k_sub. Mirrors Python dy_descriptor.
    if(ctx->is_pointwise)
    {
        rocke_value_t* pw_off = rocke_b_add(b_, rocke_b_mul(b_, m_sub, ctx->c_K), k_sub);
        if(out_valid)
        {
            rocke_value_t* m_ok = rocke_b_cmp_lt(b_, m_sub, ctx->dg_M);
            rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_sub, ctx->c_K);
            *out_valid = rocke_b_land(b_, m_ok, k_ok);
        }
        return pw_off;
    }

    // k_out innermost (CK-compatible): k_sub = ydot*xdot_slice*kpg + xdot*kpg + k_out.
    // The reduction is per group, so the decode divisor is kpg (K when ungrouped)
    // and k_out is group-local in [0, kpg); the absolute NHWK channel is g*kpg + k_out.
    // Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, last dim K).
    rocke_value_t *yx_rem, *k_out, *ydot, *xdot;
    _magic_divmod(b_, k_sub, &ctx->kpg_div, &yx_rem, &k_out);
    _magic_divmod(b_, yx_rem, &ctx->xdot_div, &ydot, &xdot);
    rocke_value_t* k_out_abs
        = ctx->k_out_group_base ? rocke_b_add(b_, k_out, ctx->k_out_group_base) : k_out;

    rocke_value_t *n_val, *m_rem, *htl, *wtl;
    _magic_divmod(b_, m_sub, &ctx->hw_div, &n_val, &m_rem);
    _magic_divmod(b_, m_rem, &ctx->w_div, &htl, &wtl);

    rocke_value_t* ho_base = rocke_b_add(b_, htl, ctx->rec_h_tilde_slice_begin);
    rocke_value_t* ho_mul = rocke_b_mul(b_, ydot, ctx->rec_a_embed_h_coeff);
    rocke_value_t* ho = rocke_b_add(b_, ho_base, ho_mul);
    rocke_value_t* wo_base = rocke_b_add(b_, wtl, ctx->rec_w_tilde_slice_begin);
    rocke_value_t* wo_mul = rocke_b_mul(b_, xdot, ctx->rec_a_embed_w_coeff);
    rocke_value_t* wo = rocke_b_add(b_, wo_base, wo_mul);

    rocke_value_t* ho_ge = rocke_b_cmp_ge(b_, ho, ctx->c0);
    rocke_value_t* ho_lt = rocke_b_cmp_lt(b_, ho, ctx->c_Ho);
    rocke_value_t* ho_ok = rocke_b_land(b_, ho_ge, ho_lt);
    rocke_value_t* wo_ge = rocke_b_cmp_ge(b_, wo, ctx->c0);
    rocke_value_t* wo_lt = rocke_b_cmp_lt(b_, wo, ctx->c_Wo);
    rocke_value_t* wo_ok = rocke_b_land(b_, wo_ge, wo_lt);
    rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_out, ctx->c_kdiv);
    rocke_value_t* valid = rocke_b_land(b_, rocke_b_land(b_, ho_ok, wo_ok), k_ok);

    /* The K stride stays total K; only the channel index carries g*kpg. */
    rocke_value_t* n_ho = rocke_b_add(b_, rocke_b_mul(b_, n_val, ctx->c_Ho), ho);
    rocke_value_t* wo_k = rocke_b_mul(b_, ctx->c_Wo, ctx->c_K);
    rocke_value_t* hi_part = rocke_b_mul(b_, n_ho, wo_k);
    rocke_value_t* wo_part = rocke_b_add(b_, rocke_b_mul(b_, wo, ctx->c_K), k_out_abs);
    rocke_value_t* offset = rocke_b_add(b_, hi_part, wo_part);
    rocke_value_t* safe_offset = rocke_b_select(b_, valid, offset, rocke_b_const_i32(b_, 0));

    if(out_valid)
        *out_valid = valid;
    return safe_offset;
}

struct tilde_w_ctx_t
{
    rocke_value_t* block_n_off;
    rocke_value_t* k_off;
    dgrad_magic_div_t kpg_div; // k_sub -> (yx_rem, k_out)
    dgrad_magic_div_t xdot_div; // yx_rem -> (ydot, xdot)
    rocke_value_t* rec_b_y_stride;
    rocke_value_t* rec_b_y_offset;
    rocke_value_t* rec_b_x_stride;
    rocke_value_t* rec_b_x_offset;
    rocke_value_t* c_Y;
    rocke_value_t* c_X;
    rocke_value_t* c_K;
    rocke_value_t* c_C;
    /* Decode divisor (kpg when grouped, c_K otherwise), as tilde_dy_ctx_t. */
    rocke_value_t* c_kdiv;
    /* g*kpg; NULL when ungrouped. */
    rocke_value_t* k_out_group_base;
    /* Innermost KYXC stride: cpg when grouped (W is stored per-group packed,
     * [K, Y, X, cpg]), c_C otherwise. */
    rocke_value_t* c_cstride;
    rocke_value_t* c0;
    /* Pointwise fast path -- mirrors the Python w_descriptor. */
    bool is_pointwise;
};

static rocke_value_t* _tilde_w_descriptor(rocke_ir_builder_t* b_,
                                          rocke_value_t* row,
                                          rocke_value_t* col,
                                          rocke_value_t** out_valid,
                                          void* user)
{
    tilde_w_ctx_t* ctx = (tilde_w_ctx_t*)user;
    rocke_value_t* c_val = rocke_b_add(b_, ctx->block_n_off, row);
    rocke_value_t* k_sub = rocke_b_add(b_, ctx->k_off, col);

    // Pointwise fast path: Y == X == 1 means KYXC is just [K, cpg], so the
    // offset is k_sub*C + c_val. Must stay in lockstep with the dy fast path.
    if(ctx->is_pointwise)
    {
        rocke_value_t* pw_off = rocke_b_add(b_, rocke_b_mul(b_, k_sub, ctx->c_C), c_val);
        if(out_valid)
        {
            rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_sub, ctx->c_K);
            rocke_value_t* c_ok = rocke_b_cmp_lt(b_, c_val, ctx->c_C);
            *out_valid = rocke_b_land(b_, k_ok, c_ok);
        }
        return pw_off;
    }

    // Same k_out-innermost decomposition as _tilde_dy_descriptor (must match).
    // c (row axis) is stride-1 in KYXC; vectorised loads along c use vector_axis_row=true.
    rocke_value_t *yx_rem, *k_out, *ydot, *xdot;
    _magic_divmod(b_, k_sub, &ctx->kpg_div, &yx_rem, &k_out);
    _magic_divmod(b_, yx_rem, &ctx->xdot_div, &ydot, &xdot);
    rocke_value_t* k_out_abs
        = ctx->k_out_group_base ? rocke_b_add(b_, k_out, ctx->k_out_group_base) : k_out;

    rocke_value_t* y_mul = rocke_b_mul(b_, ydot, ctx->rec_b_y_stride);
    rocke_value_t* y = rocke_b_add(b_, y_mul, ctx->rec_b_y_offset);
    rocke_value_t* x_mul = rocke_b_mul(b_, xdot, ctx->rec_b_x_stride);
    rocke_value_t* x = rocke_b_add(b_, x_mul, ctx->rec_b_x_offset);

    rocke_value_t* y_lt = rocke_b_cmp_lt(b_, y, ctx->c_Y);
    rocke_value_t* x_lt = rocke_b_cmp_lt(b_, x, ctx->c_X);
    rocke_value_t* yx_ok = rocke_b_land(b_, y_lt, x_lt);
    rocke_value_t* k_ok = rocke_b_cmp_lt(b_, k_out, ctx->c_kdiv);
    rocke_value_t* valid = rocke_b_land(b_, yx_ok, k_ok);

    /* W is per-group packed [K, Y, X, cpg]: the group is carried entirely by
     * the absolute output channel k_out_abs, the input channel c_val is
     * group-local and the innermost stride is cpg (C when ungrouped). */
    rocke_value_t* k_y = rocke_b_add(b_, rocke_b_mul(b_, k_out_abs, ctx->c_Y), y);
    rocke_value_t* k_y_x = rocke_b_add(b_, rocke_b_mul(b_, k_y, ctx->c_X), x);
    rocke_value_t* offset = rocke_b_add(b_, rocke_b_mul(b_, k_y_x, ctx->c_cstride), c_val);
    rocke_value_t* safe_offset = rocke_b_select(b_, valid, offset, rocke_b_const_i32(b_, 0));

    if(out_valid)
        *out_valid = valid;
    return safe_offset;
}

/* K-outer coordinate swap: the K-outer tile is indexed (k, free) while
 * _tilde_w_descriptor takes (free, k). The descriptor itself is untouched, so
 * the global addressing and its OOB select stay byte-identical -- this is a
 * swap, not a redesign. Mirrors _b_desc_fn in conv_implicit_gemm_dgrad.py. */
static rocke_value_t* _tilde_w_descriptor_kouter(rocke_ir_builder_t* b_,
                                                 rocke_value_t* row,
                                                 rocke_value_t* col,
                                                 rocke_value_t** out_valid,
                                                 void* user)
{
    return _tilde_w_descriptor(b_, col, row, out_valid, user);
}

/* ---- uniform_k_decode descriptors (Python _emit_uniform_k_decode_loops) ----
 *
 * Used by the K loop that carries the tile's decoded (k_out, xdot, ydot). Each
 * offset is a loop-invariant per-element part plus the uniform per-tile part
 * the loop body computes into the context before every load; the A bounds
 * compare the invariant (ho, wo) base against shifted uniform limits. */
struct uk_dy_ctx_t
{
    rocke_value_t* block_m_off;
    dgrad_magic_div_t hw_div; // m_sub -> (n, m_rem)
    dgrad_magic_div_t w_div; // m_rem -> (htl, wtl)
    rocke_value_t* rec_h_tilde_slice_begin;
    rocke_value_t* rec_w_tilde_slice_begin;
    rocke_value_t* c_Ho;
    rocke_value_t* c_K;
    rocke_value_t* wo_k; /* Wo*K */
    rocke_value_t* c0;
    /* per-tile uniforms */
    rocke_value_t* ho_lo;
    rocke_value_t* ho_hi;
    rocke_value_t* wo_lo;
    rocke_value_t* wo_hi;
    rocke_value_t* a_off;
};

static rocke_value_t* _uk_dy_descriptor(rocke_ir_builder_t* b_,
                                        rocke_value_t* row,
                                        rocke_value_t* col,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    uk_dy_ctx_t* ctx = (uk_dy_ctx_t*)user;
    rocke_value_t* m_sub = rocke_b_add(b_, ctx->block_m_off, row);
    rocke_value_t *n_val, *m_rem, *htl, *wtl;
    _magic_divmod(b_, m_sub, &ctx->hw_div, &n_val, &m_rem);
    _magic_divmod(b_, m_rem, &ctx->w_div, &htl, &wtl);
    rocke_value_t* ho_b = rocke_b_add(b_, htl, ctx->rec_h_tilde_slice_begin);
    rocke_value_t* wo_b = rocke_b_add(b_, wtl, ctx->rec_w_tilde_slice_begin);

    rocke_value_t* ho_ge = rocke_b_cmp_ge(b_, ho_b, ctx->ho_lo);
    rocke_value_t* ho_lt = rocke_b_cmp_lt(b_, ho_b, ctx->ho_hi);
    rocke_value_t* ho_ok = rocke_b_land(b_, ho_ge, ho_lt);
    rocke_value_t* wo_ge = rocke_b_cmp_ge(b_, wo_b, ctx->wo_lo);
    rocke_value_t* wo_lt = rocke_b_cmp_lt(b_, wo_b, ctx->wo_hi);
    rocke_value_t* wo_ok = rocke_b_land(b_, wo_ge, wo_lt);
    rocke_value_t* valid = rocke_b_land(b_, ho_ok, wo_ok);

    rocke_value_t* n_ho = rocke_b_add(b_, rocke_b_mul(b_, n_val, ctx->c_Ho), ho_b);
    rocke_value_t* hi_part = rocke_b_mul(b_, n_ho, ctx->wo_k);
    rocke_value_t* wo_part = rocke_b_mul(b_, wo_b, ctx->c_K);
    rocke_value_t* row_off = rocke_b_add(b_, hi_part, wo_part);
    rocke_value_t* offset = rocke_b_add(b_, rocke_b_add(b_, row_off, col), ctx->a_off);
    rocke_value_t* safe_offset = rocke_b_select(b_, valid, offset, ctx->c0);

    if(out_valid)
        *out_valid = valid;
    return safe_offset;
}

struct uk_w_ctx_t
{
    rocke_value_t* block_n_off;
    rocke_value_t* b_step_k; /* Y*X*Cs (Cs = cpg when grouped, C otherwise) */
    rocke_value_t* c0;
    /* per-tile uniforms */
    rocke_value_t* b_off;
    rocke_value_t* b_ok;
};

static rocke_value_t* _uk_w_descriptor(rocke_ir_builder_t* b_,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       rocke_value_t** out_valid,
                                       void* user)
{
    uk_w_ctx_t* ctx = (uk_w_ctx_t*)user;
    rocke_value_t* c_val = rocke_b_add(b_, ctx->block_n_off, row);
    rocke_value_t* k_part = rocke_b_mul(b_, col, ctx->b_step_k);
    rocke_value_t* offset = rocke_b_add(b_, rocke_b_add(b_, k_part, c_val), ctx->b_off);
    rocke_value_t* safe_offset = rocke_b_select(b_, ctx->b_ok, offset, ctx->c0);
    if(out_valid)
        *out_valid = ctx->b_ok;
    return safe_offset;
}

/* K-outer coordinate swap, as _tilde_w_descriptor_kouter. */
static rocke_value_t* _uk_w_descriptor_kouter(rocke_ir_builder_t* b_,
                                              rocke_value_t* row,
                                              rocke_value_t* col,
                                              rocke_value_t** out_valid,
                                              void* user)
{
    return _uk_w_descriptor(b_, col, row, out_valid, user);
}

/* ---- tap_outer_k descriptors (Python _emit_tap_outer_loops) ----
 *
 * The K loop is (filter tap outer) x (output-channel chunk inner); the loop
 * bodies write the per-tap (dh, dw, dy_tap) and per-chunk (kb, w_k) scalars
 * into the contexts before every load. Operands are sequenced into
 * temporaries in Python's left-to-right evaluation order. */
struct tap_dy_ctx_t
{
    rocke_value_t* block_m_off;
    dgrad_magic_div_t hw_div; // m_sub -> (n, m_rem)
    dgrad_magic_div_t w_div; // m_rem -> (hi, wi)
    rocke_value_t* c_Ho;
    rocke_value_t* c_Wo;
    rocke_value_t* c_K;
    rocke_value_t* c0;
    rocke_value_t* dh; /* pH - y (per tap) */
    rocke_value_t* dw; /* pW - x (per tap) */
    rocke_value_t* dy_tap; /* (dh*Wo + dw)*K (per tap) */
    rocke_value_t* kb; /* absolute output-channel chunk base (per chunk) */
};

static rocke_value_t* _tap_dy_descriptor(rocke_ir_builder_t* b_,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         rocke_value_t** out_valid,
                                         void* user)
{
    tap_dy_ctx_t* ctx = (tap_dy_ctx_t*)user;
    rocke_value_t* m_sub = rocke_b_add(b_, ctx->block_m_off, row);
    rocke_value_t *n_val, *m_rem, *hi, *wi;
    _magic_divmod(b_, m_sub, &ctx->hw_div, &n_val, &m_rem);
    _magic_divmod(b_, m_rem, &ctx->w_div, &hi, &wi);
    rocke_value_t* ho = rocke_b_add(b_, hi, ctx->dh);
    rocke_value_t* wo = rocke_b_add(b_, wi, ctx->dw);
    rocke_value_t* ho_ge = rocke_b_cmp_ge(b_, ho, ctx->c0);
    rocke_value_t* ho_lt = rocke_b_cmp_lt(b_, ho, ctx->c_Ho);
    rocke_value_t* ho_ok = rocke_b_land(b_, ho_ge, ho_lt);
    rocke_value_t* wo_ge = rocke_b_cmp_ge(b_, wo, ctx->c0);
    rocke_value_t* wo_lt = rocke_b_cmp_lt(b_, wo, ctx->c_Wo);
    rocke_value_t* wo_ok = rocke_b_land(b_, wo_ge, wo_lt);
    rocke_value_t* valid = rocke_b_land(b_, ho_ok, wo_ok);
    /* ((n*Ho + hi)*Wo + wi)*K + col */
    rocke_value_t* n_ho = rocke_b_mul(b_, n_val, ctx->c_Ho);
    rocke_value_t* n_ho_hi = rocke_b_add(b_, n_ho, hi);
    rocke_value_t* row_off = rocke_b_mul(b_, n_ho_hi, ctx->c_Wo);
    rocke_value_t* pix_w = rocke_b_add(b_, row_off, wi);
    rocke_value_t* pix_k = rocke_b_mul(b_, pix_w, ctx->c_K);
    rocke_value_t* pix = rocke_b_add(b_, pix_k, col);
    rocke_value_t* tap_off = rocke_b_add(b_, pix, ctx->dy_tap);
    rocke_value_t* offset = rocke_b_add(b_, tap_off, ctx->kb);
    if(out_valid)
        *out_valid = valid;
    return offset;
}

struct tap_w_ctx_t
{
    rocke_value_t* block_n_off;
    rocke_value_t* c_yxc; /* Y*X*Cs */
    rocke_value_t* w_k; /* kb*Y*X*Cs + tap*Cs (per chunk) */
};

static rocke_value_t* _tap_w_descriptor(rocke_ir_builder_t* b_,
                                        rocke_value_t* row,
                                        rocke_value_t* col,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    tap_w_ctx_t* ctx = (tap_w_ctx_t*)user;
    rocke_value_t* c_val = rocke_b_add(b_, ctx->block_n_off, row);
    rocke_value_t* col_off = rocke_b_mul(b_, col, ctx->c_yxc);
    rocke_value_t* inv = rocke_b_add(b_, col_off, c_val);
    rocke_value_t* offset = rocke_b_add(b_, inv, ctx->w_k);
    if(out_valid)
        *out_valid = NULL; /* k_abs and the tap are in range by construction */
    return offset;
}

/* K-outer coordinate swap, as _tilde_w_descriptor_kouter. */
static rocke_value_t* _tap_w_descriptor_kouter(rocke_ir_builder_t* b_,
                                               rocke_value_t* row,
                                               rocke_value_t* col,
                                               rocke_value_t** out_valid,
                                               void* user)
{
    return _tap_w_descriptor(b_, col, row, out_valid, user);
}

/* ---- dy_halo loader descriptor (Python _emit_dy_halo_loop halo_dy) ----
 * Linear dY pixel q = halo_base + row, valid inside [0, N*Hi*Wi) and below
 * the tile's real row count (Wo < halo_max_wo leaves rows unloaded). */
struct halo_dy_ctx_t
{
    rocke_value_t* halo_base;
    rocke_value_t* c0;
    rocke_value_t* npix; /* p_dg_M = N*Hi*Wi */
    rocke_value_t* real_rows;
    rocke_value_t* c_K;
    rocke_value_t* kb; /* channel-chunk base (per chunk) */
};

static rocke_value_t* _halo_dy_descriptor(rocke_ir_builder_t* b_,
                                          rocke_value_t* row,
                                          rocke_value_t* col,
                                          rocke_value_t** out_valid,
                                          void* user)
{
    halo_dy_ctx_t* ctx = (halo_dy_ctx_t*)user;
    rocke_value_t* q = rocke_b_add(b_, ctx->halo_base, row);
    rocke_value_t* q_ge = rocke_b_cmp_ge(b_, q, ctx->c0);
    rocke_value_t* q_lt = rocke_b_cmp_lt(b_, q, ctx->npix);
    rocke_value_t* q_ok = rocke_b_land(b_, q_ge, q_lt);
    rocke_value_t* in_tile = rocke_b_cmp_lt(b_, row, ctx->real_rows);
    rocke_value_t* valid = rocke_b_land(b_, q_ok, in_tile);
    rocke_value_t* q_k = rocke_b_mul(b_, q, ctx->c_K);
    rocke_value_t* q_col = rocke_b_add(b_, q_k, col);
    rocke_value_t* offset = rocke_b_add(b_, q_col, ctx->kb);
    if(out_valid)
        *out_valid = valid;
    return offset;
}

// ===========================================================================
// WMMA tilde direct epilogue (Python _emit_dgrad_tilde_direct_epilogue_wmma)
// ===========================================================================

static void _emit_dgrad_tilde_direct_epilogue_wmma(rocke_ir_builder_t* b,
                                                   const rocke_dgrad_conv_spec_t* spec,
                                                   const rocke_mmaop_t* op,
                                                   rocke_value_t* const* accs,
                                                   int num_accs,
                                                   rocke_value_t* warp_m_idx,
                                                   rocke_value_t* warp_n_idx,
                                                   rocke_value_t* lane,
                                                   rocke_value_t* block_m_off,
                                                   rocke_value_t* block_n_off,
                                                   rocke_value_t* dx_rsrc,
                                                   rocke_value_t* c0,
                                                   rocke_value_t* bounds_m,
                                                   rocke_value_t* bounds_n,
                                                   const dgrad_magic_div_t* hw_tilde,
                                                   const dgrad_magic_div_t* w_tilde_slice,
                                                   rocke_value_t* d_h_stride,
                                                   rocke_value_t* d_h_offset,
                                                   rocke_value_t* d_w_stride,
                                                   rocke_value_t* d_w_offset,
                                                   rocke_value_t* c_Hi,
                                                   rocke_value_t* c_Wi,
                                                   rocke_value_t* c_C,
                                                   rocke_value_t* c_group_base)
{
    (void)num_accs;
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    bool is_fp32 = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    bool is_bf16 = (spec->dtype_d && strcmp(spec->dtype_d, "bf16") == 0);
    int elem_bytes = is_fp32 ? 4 : 2;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * spec->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * spec->warp_tile_n));

    const rocke_arch_layout_map_t* c_map = rocke_mmaop_c_layout(op, b);

    /* Wherever two operands of one builder call would both emit ops, they are
     * sequenced into temporaries: C leaves argument evaluation order
     * unspecified while Python evaluates left to right, so an inline nest can
     * emit the same ops in a different order. */
    int flat = 0;
    for(int mi = 0; mi < mfmas_m; mi++)
    {
        for(int ni = 0; ni < mfmas_n; ni++)
        {
            rocke_value_t* acc = accs[flat++];
            /* Both atom offsets per (mi, ni), as Python computes them. */
            rocke_value_t* _bwm = rocke_b_add(b, block_m_off, warp_m_off);
            rocke_value_t* atom_m_off
                = rocke_b_add(b, _bwm, rocke_b_const_i32(b, mi * spec->warp_tile_m));
            rocke_value_t* _bwn = rocke_b_add(b, block_n_off, warp_n_off);
            rocke_value_t* atom_n_off
                = rocke_b_add(b, _bwn, rocke_b_const_i32(b, ni * spec->warp_tile_n));

            for(int i = 0; i < op->c_frag_len; i++)
            {
                rocke_value_t* row_off;
                rocke_value_t* col_off;
                rocke_arch_layout_map_coord(c_map, b, lane, i, &row_off, &col_off);

                rocke_value_t* m_val = rocke_b_add(b, atom_m_off, row_off);
                rocke_value_t* n_val = rocke_b_add(b, atom_n_off, col_off);

                // Tilde M decomposition → (n_batch, htl, wtl) → (hi, wi)
                rocke_value_t *n_batch, *m_rem, *htl, *wtl;
                _magic_divmod(b, m_val, hw_tilde, &n_batch, &m_rem);
                _magic_divmod(b, m_rem, w_tilde_slice, &htl, &wtl);
                rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, d_h_stride), d_h_offset);
                rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, d_w_stride), d_w_offset);

                rocke_value_t* m_ok = rocke_b_cmp_lt(b, m_val, bounds_m);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_val, bounds_n);
                rocke_value_t* hi_ge = rocke_b_cmp_ge(b, hi, c0);
                rocke_value_t* hi_lt = rocke_b_cmp_lt(b, hi, c_Hi);
                rocke_value_t* hi_ok = rocke_b_land(b, hi_ge, hi_lt);
                rocke_value_t* wi_ge = rocke_b_cmp_ge(b, wi, c0);
                rocke_value_t* wi_lt = rocke_b_cmp_lt(b, wi, c_Wi);
                rocke_value_t* wi_ok = rocke_b_land(b, wi_ge, wi_lt);
                rocke_value_t* mn_ok = rocke_b_land(b, m_ok, n_ok);
                rocke_value_t* hw_ok = rocke_b_land(b, hi_ok, wi_ok);
                rocke_value_t* ok = rocke_b_land(b, mn_ok, hw_ok);

                // NHWC offset: ((n_batch*Hi + hi)*Wi + wi)*C + c_abs, where the
                // absolute channel c_abs = g*cpg + n_val is formed first, as in
                // Python.
                rocke_value_t* c_abs = c_group_base ? rocke_b_add(b, n_val, c_group_base) : n_val;
                rocke_value_t* _o0 = rocke_b_mul(b, n_batch, c_Hi);
                rocke_value_t* _o1 = rocke_b_add(b, _o0, hi);
                rocke_value_t* _o2 = rocke_b_mul(b, _o1, c_Wi);
                rocke_value_t* _o3 = rocke_b_add(b, _o2, wi);
                rocke_value_t* _o4 = rocke_b_mul(b, _o3, c_C);
                rocke_value_t* off_elems = rocke_b_add(b, _o4, c_abs);
                rocke_value_t* off_bytes
                    = rocke_b_mul(b, off_elems, rocke_b_const_i32(b, elem_bytes));
                rocke_value_t* safe_off
                    = rocke_b_select(b, ok, off_bytes, rocke_b_const_i32(b, 0x7FFFFFFF));

                rocke_value_t* v_f32 = rocke_b_vec_extract(b, acc, i);
                if(is_fp32)
                    rocke_b_buffer_store_f32(b, dx_rsrc, safe_off, c0, v_f32);
                else if(is_bf16)
                    rocke_b_buffer_store_bf16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_bf16(b, v_f32));
                else
                    rocke_b_buffer_store_f16(
                        b, dx_rsrc, safe_off, c0, rocke_b_trunc_f32_to_f16(b, v_f32));
            }
        }
    }
}

// ===========================================================================
// ===========================================================================
// Tilde non-atomic epilogues (split_k==1 strided convolutions)
//
// For split_k==1 the tilde decomposition guarantees each sub-GEMM writes to
// a disjoint subset of dX elements, so plain buffer_store is safe.
//
// Shared addr_fn (tilde_dx_ctx_t / _tilde_dx_addr_fn):
//   m_global  -> (n_val, htl, wtl) via hw_tilde / w_tilde_slice
//   hi        = htl * d_h_stride + d_h_offset
//   wi        = wtl * d_w_stride + d_w_offset
//   NHWC off  = ((n_val * Hi + hi) * Wi + wi) * C + n_global
//
// Two functions mirror the non-tilde (stride=1) pattern:
//   _emit_dgrad_tilde_direct_epilogue   -- scalar store, no LDS staging
//   _emit_dgrad_tilde_cshuffle_epilogue -- LDS-staged, vector_size_c-wide stores
// ===========================================================================

struct tilde_dx_ctx_t
{
    dgrad_magic_div_t hw_tilde;
    dgrad_magic_div_t w_tilde_slice;
    rocke_value_t* d_h_stride;
    rocke_value_t* d_h_offset;
    rocke_value_t* d_w_stride;
    rocke_value_t* d_w_offset;
    rocke_value_t* c_Hi;
    rocke_value_t* c_Wi;
    rocke_value_t* c_C;
    rocke_value_t* c_group_base; /* g*cpg; NULL when ungrouped */
    /* patch store (DgradConvSpec.patch): X*C and Wi*C; NULL otherwise. */
    rocke_value_t* patch_xc;
    rocke_value_t* patch_row;
    // c0 is NOT stored here: it is created locally inside _tilde_dx_addr_fn,
    // matching Python's closure which does c0 = b.const_i32(0) inside the fn.
};

static tilde_dx_ctx_t _make_tilde_dx_ctx(const dgrad_magic_div_t* hw_tilde,
                                         const dgrad_magic_div_t* w_tilde_slice,
                                         rocke_value_t* d_h_stride,
                                         rocke_value_t* d_h_offset,
                                         rocke_value_t* d_w_stride,
                                         rocke_value_t* d_w_offset,
                                         rocke_value_t* c_Hi,
                                         rocke_value_t* c_Wi,
                                         rocke_value_t* c_C,
                                         rocke_value_t* c_group_base)
{
    tilde_dx_ctx_t ctx;
    ctx.hw_tilde = *hw_tilde;
    ctx.w_tilde_slice = *w_tilde_slice;
    ctx.d_h_stride = d_h_stride;
    ctx.d_h_offset = d_h_offset;
    ctx.d_w_stride = d_w_stride;
    ctx.d_w_offset = d_w_offset;
    ctx.c_Hi = c_Hi;
    ctx.c_Wi = c_Wi;
    ctx.c_C = c_C;
    ctx.c_group_base = c_group_base;
    ctx.patch_xc = NULL;
    ctx.patch_row = NULL;
    /* c0 not stored — created locally inside _tilde_dx_addr_fn */
    return ctx;
}

// Epilogue addr_fn: (m_global, n_global) -> (element offset, hw_valid).
// bounds_m / bounds_n (m < gemm_m, n < C) are handled by the epilogue caller.
static rocke_value_t* _tilde_dx_addr_fn(rocke_ir_builder_t* b,
                                        rocke_value_t* m_global,
                                        rocke_value_t* n_global,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    tilde_dx_ctx_t* ctx = (tilde_dx_ctx_t*)user;

    // c0 created locally to match Python closure order (b.const_i32(0) first in fn).
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    // Decompose m_global -> (n_val, htl, wtl)
    rocke_value_t *n_val, *m_rem, *htl, *wtl;
    _magic_divmod(b, m_global, &ctx->hw_tilde, &n_val, &m_rem);
    _magic_divmod(b, m_rem, &ctx->w_tilde_slice, &htl, &wtl);

    rocke_value_t* hi = rocke_b_add(b, rocke_b_mul(b, htl, ctx->d_h_stride), ctx->d_h_offset);
    rocke_value_t* wi = rocke_b_add(b, rocke_b_mul(b, wtl, ctx->d_w_stride), ctx->d_w_offset);

    // hi/wi validity — explicit sequencing (sge before slt) to match Python left-to-right.
    rocke_value_t* hi_ge = rocke_b_cmp_ge(b, hi, c0);
    rocke_value_t* hi_lt = rocke_b_cmp_lt(b, hi, ctx->c_Hi);
    rocke_value_t* hi_ok = rocke_b_land(b, hi_ge, hi_lt);
    rocke_value_t* wi_ge = rocke_b_cmp_ge(b, wi, c0);
    rocke_value_t* wi_lt = rocke_b_cmp_lt(b, wi, ctx->c_Wi);
    rocke_value_t* wi_ok = rocke_b_land(b, wi_ge, wi_lt);
    rocke_value_t* hw_ok = rocke_b_land(b, hi_ok, wi_ok);

    if(out_valid)
        *out_valid = hw_ok;

    // NHWC element offset: ((n_val * Hi + hi) * Wi + wi) * C + c_abs, with the
    // absolute channel c_abs = g*cpg + n_global (n_global when ungrouped).
    rocke_value_t* _o0 = rocke_b_mul(b, n_val, ctx->c_Hi);
    rocke_value_t* _o1 = rocke_b_add(b, _o0, hi);
    rocke_value_t* _o2 = rocke_b_mul(b, _o1, ctx->c_Wi);
    rocke_value_t* _o3 = rocke_b_add(b, _o2, wi);
    rocke_value_t* _o4 = rocke_b_mul(b, _o3, ctx->c_C);
    rocke_value_t* c_abs
        = ctx->c_group_base ? rocke_b_add(b, n_global, ctx->c_group_base) : n_global;
    if(ctx->patch_xc)
    {
        /* patch: n_global = (y, x*C + c) of the patch at (hi, wi), which lands
         * y*Wi*C + x*C + c past it. */
        rocke_value_t* patch_y = rocke_b_div(b, n_global, ctx->patch_xc);
        rocke_value_t* patch_yxc = rocke_b_mul(b, patch_y, ctx->patch_xc);
        rocke_value_t* patch_rem = rocke_b_sub(b, n_global, patch_yxc);
        rocke_value_t* patch_yrow = rocke_b_mul(b, patch_y, ctx->patch_row);
        c_abs = rocke_b_add(b, patch_yrow, patch_rem);
    }
    rocke_value_t* offset = rocke_b_add(b, _o4, c_abs);

    // Return raw element offset (no sentinel select here).
    // DirectEpilogue/CShuffleEpilogue apply the sentinel via the out_valid flag,
    // matching stride-1 (_dgrad_dx_addr_fn) which also returns raw offset + valid.
    return offset;
}

// Scalar (per-element) store — mirrors _emit_dgrad_direct_epilogue for the tilde case.
static void _emit_dgrad_tilde_direct_epilogue(rocke_ir_builder_t* b,
                                              const rocke_dgrad_conv_spec_t* spec,
                                              const rocke_mfma_atom_t* atom,
                                              const rocke_warp_grid_t* grid,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* dx_rsrc,
                                              rocke_value_t* bounds_m,
                                              rocke_value_t* bounds_n,
                                              const dgrad_magic_div_t* hw_tilde,
                                              const dgrad_magic_div_t* w_tilde_slice,
                                              rocke_value_t* d_h_stride,
                                              rocke_value_t* d_h_offset,
                                              rocke_value_t* d_w_stride,
                                              rocke_value_t* d_w_offset,
                                              rocke_value_t* c_Hi,
                                              rocke_value_t* c_Wi,
                                              rocke_value_t* c_C,
                                              rocke_value_t* c_group_base,
                                              rocke_value_t* patch_xc,
                                              rocke_value_t* patch_row)
{
    tilde_dx_ctx_t ctx = _make_tilde_dx_ctx(hw_tilde,
                                            w_tilde_slice,
                                            d_h_stride,
                                            d_h_offset,
                                            d_w_stride,
                                            d_w_offset,
                                            c_Hi,
                                            c_Wi,
                                            c_C,
                                            c_group_base);
    ctx.patch_xc = patch_xc;
    ctx.patch_row = patch_row;
    rocke_direct_epilogue_t epi;
    epi.atom = atom;
    epi.grid = *grid;
    epi.out_dtype = spec->dtype_d;
    rocke_direct_epilogue_store(
        b, &epi, accs, num_accs, _tilde_dx_addr_fn, &ctx, dx_rsrc, bounds_m, bounds_n, false);
}

// LDS-staged wide store — mirrors _emit_dgrad_cshuffle_epilogue for the tilde case.
// The N dimension (channel index) is contiguous in dX for fixed m, so the
// cshuffle epilogue can issue vector_size_c-wide buffer stores along N.
static void _emit_dgrad_tilde_cshuffle_epilogue(rocke_ir_builder_t* b,
                                                const rocke_dgrad_conv_spec_t* spec,
                                                const rocke_mfma_atom_t* atom,
                                                const rocke_warp_grid_t* grid,
                                                rocke_value_t* const* accs,
                                                int num_accs,
                                                rocke_value_t* dx_rsrc,
                                                rocke_value_t* bounds_m,
                                                rocke_value_t* bounds_n,
                                                const dgrad_magic_div_t* hw_tilde,
                                                const dgrad_magic_div_t* w_tilde_slice,
                                                rocke_value_t* d_h_stride,
                                                rocke_value_t* d_h_offset,
                                                rocke_value_t* d_w_stride,
                                                rocke_value_t* d_w_offset,
                                                rocke_value_t* c_Hi,
                                                rocke_value_t* c_Wi,
                                                rocke_value_t* c_C,
                                                rocke_value_t* c_group_base,
                                                rocke_value_t* patch_xc,
                                                rocke_value_t* patch_row)
{
    tilde_dx_ctx_t ctx = _make_tilde_dx_ctx(hw_tilde,
                                            w_tilde_slice,
                                            d_h_stride,
                                            d_h_offset,
                                            d_w_stride,
                                            d_w_offset,
                                            c_Hi,
                                            c_Wi,
                                            c_C,
                                            c_group_base);
    ctx.patch_xc = patch_xc;
    ctx.patch_row = patch_row;
    /* Same width rule as the stride-1 cshuffle (Python _dgrad_store_vec): the
     * per-group channel run, not a fixed 8 -- a store wider than cpg would
     * straddle two groups' dX slabs. */
    int max_store_vec = _dgrad_store_vec(spec);
    rocke_cshuffle_epilogue_t epi = rocke_cshuffle_epilogue_from_grid(atom, grid, max_store_vec);
    epi.out_dtype = spec->dtype_d;
    rocke_cshuffle_epilogue_store(
        b, &epi, accs, num_accs, _tilde_dx_addr_fn, &ctx, dx_rsrc, bounds_m, bounds_n);
}

// ===========================================================================
// Tilde dgrad kernel builder (Python _build_tilde_dgrad, lines 1366-1757)
// ===========================================================================

/* Pointer declarations for the dgrad kernarg list. dY and W share the input
 * dtype; dX is read-modify-write only under split-K atomics; the tilde record
 * buffer is a 4-byte-aligned i32 table. */
struct DgradPtrDecl
{
    const rocke_type_t* ab_type;
    const rocke_type_t* d_type;
    const rocke_param_opts_t* ro_opts;
    const rocke_param_opts_t* d_opts;
    const rocke_param_opts_t* buf_opts;
};

static rocke_value_t* dgrad_declare_ptr(rocke_ir_builder_t* b,
                                        const char* name,
                                        rocke_conv_arg_kind_t kind,
                                        void* user)
{
    const DgradPtrDecl* d = static_cast<const DgradPtrDecl*>(user);
    switch(kind)
    {
    case ROCKE_CONV_ARG_A:
    case ROCKE_CONV_ARG_B:
        return rocke_b_param(b, name, d->ab_type, d->ro_opts);
    case ROCKE_CONV_ARG_D:
        return rocke_b_param(b, name, d->d_type, d->d_opts);
    case ROCKE_CONV_ARG_I32_PTR:
        return rocke_b_param(b, name, rocke_ptr_type(b, rocke_i32(), "global"), d->buf_opts);
    default:
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "dgrad: unexpected pointer kind for %s", name);
    }
}

static rocke_kernel_def_t*
    _build_tilde_dgrad(rocke_ir_builder_t* b, const rocke_dgrad_conv_spec_t* spec, const char* arch)
{
    const rocke_conv_problem_t* p = &spec->problem;
    rocke_tilde_decomposition_t tilde = rocke_compute_tilde(p);

    rocke_sub_gemm_params_t sub_gemms[128];
    int num_sub_gemms = rocke_enumerate_sub_gemms(
        p, &tilde, spec->tile_m, spec->tile_n, spec->tile_k, spec->split_k, sub_gemms, 128);
    if(num_sub_gemms <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad tilde: no non-empty sub-GEMMs");
        return NULL;
    }

    int block_m = spec->tile_m;
    int block_n = spec->tile_n;
    int block_k = spec->tile_k;

    // ---- waves_per_eu ----
    if(spec->has_waves_per_eu && b->kernel)
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);

    // ---- params ----
    const rocke_type_t* ab_ir = _dtype_to_ir(spec->dtype_a);
    const rocke_type_t* d_ir = _dtype_to_ir(spec->dtype_d);
    const rocke_type_t* ab_global = rocke_ptr_type(b, ab_ir, "global");
    const rocke_type_t* d_global = rocke_ptr_type(b, d_ir, "global");
    rocke_param_opts_t ro_opts;
    memset(&ro_opts, 0, sizeof(ro_opts));
    ro_opts.noalias = true;
    ro_opts.noalias_set = true;
    ro_opts.readonly = true;
    ro_opts.readonly_set = true;
    ro_opts.align = 16;
    ro_opts.align_set = true;

    // split_k>1 uses atomic_add (multiple blocks accumulate into same dX elements).
    // Tilde sub-GEMMs with split_k=1 use direct buffer_store (disjoint writes), so writeonly.
    bool uses_atomic_store = (spec->split_k > 1);
    rocke_param_opts_t d_opts;
    memset(&d_opts, 0, sizeof(d_opts));
    d_opts.noalias = true;
    d_opts.noalias_set = true;
    if(!uses_atomic_store)
    {
        d_opts.writeonly = true;
        d_opts.writeonly_set = true;
    }
    d_opts.align = 16;
    d_opts.align_set = true;

    // sub_gemm_buf: the tilde decomposition record table
    rocke_param_opts_t buf_opts;
    memset(&buf_opts, 0, sizeof(buf_opts));
    buf_opts.noalias = true;
    buf_opts.noalias_set = true;
    buf_opts.readonly = true;
    buf_opts.readonly_set = true;
    buf_opts.align = 4;
    buf_opts.align_set = true;

    /* ---- the whole kernarg list ----
     * Emitted from the ordered ABI list, as the Python builder emits from
     * conv_arg_names(direction="dgrad"): pointers, byte sizes, the shared
     * extent block, dgrad's GEMM dims, the three tensors' strides, the
     * m-decode magic pair, the grid counts and the tilde record buffer.
     * dgrad rejects 3-D in validate(), so the list is always the 2-D form. */
    rocke_value_t* dY = NULL;
    rocke_value_t* W = NULL;
    rocke_value_t* dX = NULL;
    rocke_value_t* dY_bytes = NULL;
    rocke_value_t* W_bytes = NULL;
    rocke_value_t* dX_bytes = NULL;
    rocke_value_t* sub_gemm_buf = NULL;
    rocke_value_t* num_sub_gemms_param = NULL;
    rocke_conv_build_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.params_is_3d = false;
    {
        DgradPtrDecl ptrs;
        ptrs.ab_type = ab_global;
        ptrs.d_type = d_global;
        ptrs.ro_opts = &ro_opts;
        ptrs.d_opts = &d_opts;
        ptrs.buf_opts = &buf_opts;
        const rocke_conv_param_slot_t slots[] = {
            {"dY", &dY},
            {"W", &W},
            {"dX", &dX},
            {"dY_bytes", &dY_bytes},
            {"W_bytes", &W_bytes},
            {"dX_bytes", &dX_bytes},
            {"p_N", &ctx.p_N},
            {"p_Hi", &ctx.p_Hi},
            {"p_Wi", &ctx.p_Wi},
            {"p_C", &ctx.p_C},
            {"p_K", &ctx.p_K},
            {"p_Y", &ctx.p_Y},
            {"p_X", &ctx.p_X},
            {"p_sH", &ctx.p_sH},
            {"p_sW", &ctx.p_sW},
            {"p_pH", &ctx.p_pH},
            {"p_pW", &ctx.p_pW},
            {"p_dH", &ctx.p_dH},
            {"p_dW", &ctx.p_dW},
            {"p_groups", &ctx.p_groups},
            {"p_Ho", &ctx.p_Ho},
            {"p_Wo", &ctx.p_Wo},
            {"p_cpg", &ctx.p_cpg},
            {"p_kpg", &ctx.p_kpg},
            {"p_dg_M", &ctx.p_dg_M}, /* N*Hi*Wi */
            {"p_dg_N", &ctx.p_dg_N}, /* cpg     */
            {"p_dg_K", &ctx.p_dg_K}, /* Y*X*kpg */
            {"p_dY_stride_n", &ctx.p_dY_stride_n},
            {"p_dY_stride_ho", &ctx.p_dY_stride_ho},
            {"p_dY_stride_wo", &ctx.p_dY_stride_wo},
            {"p_W_stride_k", &ctx.p_W_stride_k},
            {"p_W_stride_y", &ctx.p_W_stride_y},
            {"p_W_stride_x", &ctx.p_W_stride_x},
            {"p_dX_stride_n", &ctx.p_dX_stride_n},
            {"p_dX_stride_hi", &ctx.p_dX_stride_hi},
            {"p_dX_stride_wi", &ctx.p_dX_stride_wi},
            {"p_magic_m_Hi_mult", &ctx.p_magic_m_Hi_mult},
            {"p_magic_m_Hi_shift", &ctx.p_magic_m_Hi_shift},
            {"p_magic_m_Wi_mult", &ctx.p_magic_m_Wi_mult},
            {"p_magic_m_Wi_shift", &ctx.p_magic_m_Wi_shift},
            {"p_num_pid_m", &ctx.p_num_pid_m},
            {"p_num_pid_n", &ctx.p_num_pid_n},
            {"sub_gemm_buf", &sub_gemm_buf},
            {"num_sub_gemms", &num_sub_gemms_param},
        };
        rocke_conv_arg_list_t abi;
        rocke_conv_arg_names("dgrad", /*is_3d=*/false, /*two_stage=*/false, &abi);
        if(!rocke_conv_emit_param_block(
               b, &abi, dgrad_declare_ptr, &ptrs, slots, (int)(sizeof(slots) / sizeof(slots[0]))))
        {
            return NULL;
        }
    }

    // ---- resolve op + atom ----
    const rocke_mmaop_t* op = _resolve_dgrad_op(b, spec, arch);
    if(!op)
        return NULL;
    bool is_wmma = (op->family && strcmp(op->family, "wmma") == 0);
    const rocke_mfma_atom_t* atom
        = is_wmma ? NULL
                  : rocke_mfma_atom(
                        spec->dtype_a, spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    int a_per_lane = op->a_frag_len;
    int b_per_lane = op->b_frag_len;
    int c_per_lane = op->c_frag_len;

    // ---- 1D grid: block_id_x covers all sub-GEMMs' tiles ----
    rocke_value_t* flat_block_id = rocke_b_block_id_x(b);

    // ---- XCD-contiguous tile order (Python xcd_tile_order_xcds) ----
    // Remap the launch-order linear id over (group, tile) when the kernargs
    // ask for it (one uniform select) and split it back into tile and group.
    // Operands are sequenced in Python's left-to-right evaluation order.
    rocke_value_t* xcd_group = NULL;
    const int num_xcds = _dgrad_xcd_tile_order_xcds(spec, arch);
    if(num_xcds)
    {
        rocke_value_t* c1_xcd = rocke_b_const_i32(b, 1);
        rocke_value_t* flat_tiles = rocke_b_mul(b, ctx.p_num_pid_m, ctx.p_num_pid_n);
        rocke_value_t* num_wgs = rocke_b_mul(b, flat_tiles, ctx.p_groups);
        rocke_value_t* by = rocke_b_block_id_y(b);
        rocke_value_t* by_tiles = rocke_b_mul(b, by, flat_tiles);
        rocke_value_t* linear_id = rocke_b_add(b, flat_block_id, by_tiles);
        rocke_value_t* remapped
            = rocke_chiplet_transform_chunked_runtime(b, linear_id, num_wgs, num_xcds, NULL);
        if(!remapped)
            return NULL;
        rocke_value_t* taps = rocke_b_mul(b, ctx.p_Y, ctx.p_X);
        rocke_value_t* multi_tap = rocke_b_cmp_gt(b, taps, c1_xcd);
        rocke_value_t* multi_n = rocke_b_cmp_gt(b, ctx.p_num_pid_n, c1_xcd);
        rocke_value_t* shares_rows = rocke_b_lor(b, multi_tap, multi_n);
        rocke_value_t* is_grouped = rocke_b_cmp_gt(b, ctx.p_groups, c1_xcd);
        rocke_value_t* grp_rows = rocke_b_land(b, is_grouped, shares_rows);
        rocke_value_t* c_xcds = rocke_b_const_i32(b, num_xcds);
        rocke_value_t* enough = rocke_b_cmp_ge(b, num_wgs, c_xcds);
        rocke_value_t* use_xcd = rocke_b_land(b, grp_rows, enough);
        rocke_value_t* logical_id = rocke_b_select(b, use_xcd, remapped, linear_id);
        flat_block_id = rocke_b_mod(b, logical_id, flat_tiles);
        xcd_group = rocke_b_div(b, logical_id, flat_tiles);
    }

    // ---- sub-GEMM record ----
    // Stride 1 (Python DgradConvSpec.folds_sub_gemm_record): the only
    // sub-GEMM's record is rebuilt from the kernargs -- no search, no record
    // loads. Otherwise binary-search the record buffer and load the fields.
    const bool fold_record = _dgrad_folds_sub_gemm_record(spec);
    rocke_value_t* sg_idx = NULL;
    rocke_value_t* c0_rec = NULL;
    rocke_value_t* c1_rec = NULL;
    if(fold_record)
    {
        c0_rec = rocke_b_const_i32(b, 0);
        c1_rec = rocke_b_const_i32(b, 1);
    }
    else
    {
        sg_idx = _emit_binary_search(
            b, flat_block_id, sub_gemm_buf, num_sub_gemms_param, spec->max_sub_gemms);
    }
    auto _ld = [&](int field_idx) -> rocke_value_t* {
        if(!fold_record)
            return _emit_load_record_field(b, sub_gemm_buf, sg_idx, field_idx);
        switch(field_idx)
        {
        case 0:
            return c0_rec; /* block_start */
        case 1:
            return ctx.p_num_pid_m;
        case 2:
            return ctx.p_num_pid_n;
        case 3:
            return ctx.p_dg_M;
        case 4:
            return ctx.p_dg_K;
        case 5:
            return ctx.p_Hi; /* h_tilde_slice */
        case 6:
            return ctx.p_Wi; /* w_tilde_slice */
        case 7:
            return ctx.p_pH; /* h_tilde_slice_begin */
        case 8:
            return ctx.p_pW; /* w_tilde_slice_begin */
        case 9:
            return ctx.p_Y; /* y_dot_slice */
        case 10:
            return ctx.p_X; /* x_dot_slice */
        case 11:
            return rocke_b_sub(b, c0_rec, ctx.p_dH); /* a_embed_h_coeff */
        case 12:
            return rocke_b_sub(b, c0_rec, ctx.p_dW); /* a_embed_w_coeff */
        case 21:
        {
            /* gemm_k_padded: dg_K rounded up to tile_k * split_k. */
            const int kq = spec->tile_k * spec->split_k;
            rocke_value_t* c_kq = rocke_b_const_i32(b, kq);
            rocke_value_t* c_kq1 = rocke_b_const_i32(b, kq - 1);
            rocke_value_t* k_up = rocke_b_add(b, ctx.p_dg_K, c_kq1);
            rocke_value_t* k_q = rocke_b_div(b, k_up, c_kq);
            return rocke_b_mul(b, k_q, c_kq);
        }
        default: /* 13..20: unit strides, zero offsets */
            return (field_idx % 2) ? c1_rec : c0_rec;
        }
    };

    // ---- record fields ----
    rocke_value_t* rec_block_start = _ld(0);
    (void)_ld(1); /* rec_num_m_tiles: unused */
    rocke_value_t* rec_num_n_tiles = _ld(2);
    rocke_value_t* rec_gemm_m = _ld(3);
    rocke_value_t* rec_gemm_k = _ld(4);
    rocke_value_t* rec_h_tilde_slice = _ld(5);
    rocke_value_t* rec_w_tilde_slice = _ld(6);
    rocke_value_t* rec_h_tilde_slice_begin = _ld(7);
    rocke_value_t* rec_w_tilde_slice_begin = _ld(8);
    (void)_ld(9); /* rec_y_dot_slice: unused (k_out innermost) */
    rocke_value_t* rec_x_dot_slice = _ld(10);
    rocke_value_t* rec_a_embed_h_coeff = _ld(11);
    rocke_value_t* rec_a_embed_w_coeff = _ld(12);
    rocke_value_t* rec_b_y_stride = _ld(13);
    rocke_value_t* rec_b_y_offset = _ld(14);
    rocke_value_t* rec_b_x_stride = _ld(15);
    rocke_value_t* rec_b_x_offset = _ld(16);
    rocke_value_t* rec_d_h_stride = _ld(17);
    rocke_value_t* rec_d_h_offset = _ld(18);
    rocke_value_t* rec_d_w_stride = _ld(19);
    rocke_value_t* rec_d_w_offset = _ld(20);

    // ---- magic-division pairs of this record's divisors ----
    // (Python sub_gemm_magic_pairs); the table starts after the last record.
    // The folded record's table is one entry right after the one record, at
    // constant offsets.
    rocke_value_t* rec_magic[ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS];
    if(fold_record)
    {
        for(int j = 0; j < ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS; j++)
        {
            rocke_value_t* _off = rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS + j);
            rec_magic[j] = rocke_b_global_load_i32(b, sub_gemm_buf, _off, 4);
        }
    }
    else
    {
        rocke_value_t* _c_rec_fields = rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS);
        rocke_value_t* _magic_rec_end = rocke_b_mul(b, num_sub_gemms_param, _c_rec_fields);
        rocke_value_t* _c_magic_fields = rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS);
        rocke_value_t* _magic_rec_off = rocke_b_mul(b, sg_idx, _c_magic_fields);
        rocke_value_t* _magic_base = rocke_b_add(b, _magic_rec_end, _magic_rec_off);
        for(int j = 0; j < ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS; j++)
        {
            rocke_value_t* _c_j = rocke_b_const_i32(b, j);
            rocke_value_t* _off = rocke_b_add(b, _magic_base, _c_j);
            rec_magic[j] = rocke_b_global_load_i32(b, sub_gemm_buf, _off, 4);
        }
    }

    // ---- compute local tile indices ----
    // The folded record's block_start is 0, so the subtraction is skipped.
    rocke_value_t* local_flat
        = fold_record ? flat_block_id : rocke_b_sub(b, flat_block_id, rec_block_start);
    rocke_value_t* local_m_tile = rocke_b_div(b, local_flat, rec_num_n_tiles);
    rocke_value_t* local_n_tile = rocke_b_mod(b, local_flat, rec_num_n_tiles);

    // ---- WarpGrid ----
    rocke_warp_grid_t grid;
    memset(&grid, 0, sizeof(grid));
    grid.tile_m = block_m;
    grid.tile_n = block_n;
    grid.tile_k = block_k;
    grid.warp_m = spec->warp_m;
    grid.warp_n = spec->warp_n;
    grid.warp_k = 1;
    grid.warp_tile_m = spec->warp_tile_m;
    grid.warp_tile_n = spec->warp_tile_n;
    grid.warp_tile_k = spec->warp_tile_k;
    grid.wave_size = spec->wave_size;

    int block_size = rocke_dgrad_conv_spec_block_size(spec);
    bool is_wavelet = (spec->pipeline && strcmp(spec->pipeline, "wavelet") == 0);
    int launch_block_size = rocke_dgrad_conv_spec_launch_block_size(spec);
    if(b->kernel)
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", launch_block_size);

    rocke_value_t* wave = rocke_b_const_i32(b, spec->wave_size);
    rocke_value_t* c_warps_n = rocke_b_const_i32(b, spec->warp_n);
    rocke_value_t* c_warps_n_warp_m = rocke_b_const_i32(b, spec->warp_n * spec->warp_m);
    rocke_value_t* c_tile_m = rocke_b_const_i32(b, block_m);
    rocke_value_t* c_tile_n = rocke_b_const_i32(b, block_n);
    rocke_value_t* c_tile_k = rocke_b_const_i32(b, block_k);
    (void)c_warps_n_warp_m;
    (void)c_tile_k;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* lane = rocke_b_mod(b, tid, wave);
    rocke_value_t* warp_id = rocke_b_div(b, tid, wave);
    rocke_value_t* warp_m_idx = rocke_b_div(b, warp_id, c_warps_n);
    rocke_value_t* warp_n_idx = rocke_b_mod(b, warp_id, c_warps_n);

    grid.tid = tid;
    grid.lane = lane;
    grid.warp_id = warp_id;
    grid.warp_m_idx = warp_m_idx;
    grid.warp_n_idx = warp_n_idx;
    grid.warp_k_idx = rocke_b_const_i32(b, 0);
    grid.block_m_off = rocke_b_mul(b, rocke_b_block_id_y(b), c_tile_m);
    grid.block_n_off = rocke_b_mul(b, rocke_b_block_id_x(b), c_tile_n);
    grid.block_k_off = rocke_b_const_i32(b, 0);

    // Override block_m/n_off with local tile offsets
    rocke_value_t* block_m_off_v = rocke_b_mul(b, local_m_tile, rocke_b_const_i32(b, block_m));
    rocke_value_t* block_n_off_v = rocke_b_mul(b, local_n_tile, rocke_b_const_i32(b, block_n));
    grid.block_m_off = block_m_off_v;
    grid.block_n_off = block_n_off_v;

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_block_k = rocke_b_const_i32(b, block_k);

    // ---- split-K bounds ----
    bool is_split_k = spec->split_k > 1;
    rocke_value_t* k_lo;
    rocke_value_t* k_hi;
    if(is_split_k)
    {
        rocke_value_t* rec_gemm_k_padded = _ld(21);
        rocke_value_t* c_split_k = rocke_b_const_i32(b, spec->split_k);
        rocke_value_t* k_slice = rocke_b_div(b, rec_gemm_k_padded, c_split_k);
        k_lo = rocke_b_mul(b, rocke_b_block_id_z(b), k_slice);
        k_hi = rocke_b_add(b, k_lo, k_slice);
    }
    else
    {
        k_lo = c0;
        k_hi = rec_gemm_k;
    }

    /* ---- runtime problem values (AOT) ----
     * Every former const_i32(p.<field>) reads the matching kernarg. The names
     * keep the c_ prefix so the rest of the body -- descriptors, bounds and
     * epilogues -- reads the same as the JIT version did. */
    rocke_value_t* c_Ho = ctx.p_Ho;
    rocke_value_t* c_Wo = ctx.p_Wo;
    rocke_value_t* c_K = ctx.p_K;
    rocke_value_t* c_Hi = ctx.p_Hi;
    rocke_value_t* c_Wi = ctx.p_Wi;
    rocke_value_t* c_C = ctx.p_C;
    rocke_value_t* c_Y = ctx.p_Y;
    rocke_value_t* c_X = ctx.p_X;
    rocke_value_t* c_dg_N = ctx.p_cpg;
    /* patch: the GEMM's N is the filter's (y, x, c) -- Y*X*C columns, a run of
     * X*C contiguous in dX per filter row y, the rows Wi*C apart. Only the
     * store reads these (Python DgradConvSpec.patch). */
    rocke_value_t* patch_xc = NULL;
    rocke_value_t* patch_row = NULL;
    if(spec->patch)
    {
        if(is_wmma)
            return NULL; /* Python: "patch dgrad is implemented for MFMA only" */
        patch_xc = rocke_b_mul(b, ctx.p_X, ctx.p_C);
        c_dg_N = rocke_b_mul(b, ctx.p_Y, patch_xc);
        patch_row = rocke_b_mul(b, ctx.p_Wi, ctx.p_C);
    }

    /* dY row count N*Ho*Wo, used by the pointwise fast path's bounds check. */
    rocke_value_t* c_dY_rows = rocke_b_mul(b, rocke_b_mul(b, ctx.p_N, ctx.p_Ho), ctx.p_Wo);

    /* Grouped conv (groups > 1): the conv group rides blockIdx.y. blockIdx.z
     * is split_k and blockIdx.x the flat tilde-tile index, so y is free (the
     * WarpGrid block_m offset read from it above is overridden). Each CTA
     * handles one group: its reduction is confined to that group's kpg output
     * channels and it writes that group's cpg input-channel slab of dX. The
     * group offsets the absolute output channel (g*kpg + local) on dY and W
     * and the absolute input channel (g*cpg + local) on dX, and the k_sub
     * decode divides by kpg rather than the total K. For groups == 1 nothing
     * is emitted, so the ungrouped IR is unchanged. Mirrors Python. */
    const bool grouped = p->groups > 1;
    rocke_value_t* c_kpg = NULL;
    rocke_value_t* k_out_group_base = NULL;
    rocke_value_t* c_group_base = NULL;
    if(grouped)
    {
        /* Under the XCD-contiguous order the group comes from the remapped id. */
        rocke_value_t* group_idx = xcd_group ? xcd_group : rocke_b_block_id_y(b);
        c_kpg = ctx.p_kpg;
        k_out_group_base = rocke_b_mul(b, group_idx, c_kpg);
        c_group_base = rocke_b_mul(b, group_idx, ctx.p_cpg);
    }
    /* Decode divisor of the reduction index (kpg when grouped, K otherwise)
     * and the innermost KYXC stride (cpg when grouped, C otherwise). */
    rocke_value_t* c_kdiv = grouped ? c_kpg : c_K;
    rocke_value_t* c_cstride = grouped ? c_dg_N : c_C;

    // k_out-innermost: k_sub = ydot*xdot_slice*kpg + xdot*kpg + k_out
    // (c_kdiv is the innermost divisor; ydot_times_xdot no longer needed in descriptors)
    rocke_value_t* hw_tilde = rocke_b_mul(b, rec_h_tilde_slice, rec_w_tilde_slice);
    // The record's divisors, each with its magic pair. The K divisor is
    // c_kdiv (Python: kpg if grouped else K).
    const dgrad_magic_div_t kpg_div = {c_kdiv, rec_magic[0], rec_magic[1]};
    const dgrad_magic_div_t xdot_div = {rec_x_dot_slice, rec_magic[2], rec_magic[3]};
    const dgrad_magic_div_t hw_div = {hw_tilde, rec_magic[4], rec_magic[5]};
    const dgrad_magic_div_t w_div = {rec_w_tilde_slice, rec_magic[6], rec_magic[7]};

    // ---- LDS ----
    rocke_conv_lds_layout_t lds_layout = _dgrad_effective_lds_layout(spec);
    /* The staged dY halo tile (dy_halo) replaces the A tile. */
    int a_shape[2]
        = {spec->dy_halo ? _dgrad_dy_halo_lds_rows(spec) : block_m, lds_layout.row_stride};
    /* A stays M-outer; only B flips. Mirrors the Python branch. */
    int b_shape_arr[2] = {block_n, lds_layout.row_stride};
    if(spec->lds_k_outer)
    {
        b_shape_arr[0] = block_k;
        b_shape_arr[1] = block_n + _dgrad_kouter_b_pad(spec);
    }
    rocke_value_t* A_smem = rocke_b_smem_alloc(b, ab_ir, a_shape, 2, "A_smem");
    rocke_value_t* B_smem = rocke_b_smem_alloc(b, ab_ir, b_shape_arr, 2, "B_smem");
    /* dy_halo=2: the second B buffer of the per-tap double buffer. */
    rocke_value_t* B_smem2
        = spec->dy_halo == 2 ? rocke_b_smem_alloc(b, ab_ir, b_shape_arr, 2, "B_smem2") : NULL;

    // ---- MFMA tile counts ----
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    int k_atoms = rocke_dgrad_conv_spec_k_atoms_per_tile_k(spec);
    int num_accs = mfmas_m * mfmas_n;

    // ---- accumulators ----
    rocke_value_t* acc_init = rocke_b_zero_vec_f32(b, c_per_lane);
    rocke_iter_arg_t iter_args[ROCKE_CONV_MAX_ACCS];
    char acc_name_bufs[ROCKE_CONV_MAX_ACCS][32];
    for(int i = 0, idx = 0; i < mfmas_m; i++)
    {
        for(int j = 0; j < mfmas_n; j++, idx++)
        {
            snprintf(acc_name_bufs[idx], sizeof(acc_name_bufs[0]), "acc_m%d_n%d", i, j);
            iter_args[idx].name = acc_name_bufs[idx];
            iter_args[idx].init = acc_init;
        }
    }

    // ---- buffer resources ----
    rocke_value_t* dy_rsrc = rocke_b_buffer_rsrc(b, dY, dY_bytes);
    (void)rocke_b_const_i32(b, 0); /* dy soffset */
    rocke_value_t* w_rsrc = rocke_b_buffer_rsrc(b, W, W_bytes);
    (void)rocke_b_const_i32(b, 0); /* w soffset */
    rocke_value_t* dx_rsrc = rocke_b_buffer_rsrc(b, dX, dX_bytes);
    (void)rocke_b_const_i32(b, 0); /* dx soffset */

    // k_off_capture is managed via the tilde descriptor context structs

    int threads = block_size;
    /* Per-group vector widths: dY (NHWK) vectorises along the per-group output
     * channel run kpg and W (KYXC) along the per-group input channel run cpg
     * (K and C when ungrouped), so a grouped load never straddles two groups.
     * Mirrors Python default_vector_sizes(p.cpg, p.kpg, ...). */
    const int cpg = rocke_conv_problem_cpg(p);
    const int kpg = rocke_conv_problem_kpg(p);
    // load_vec_a: k_out innermost -> consecutive k_sub -> consecutive k_out
    // -> contiguous in dY (NHWK, last dim K).  Condition: kpg % load_vec_a == 0.
    // For split_k > 1 the slice boundary may not be K-aligned; use 1 there.
    int load_vec_a = 1;
    if(spec->split_k <= 1)
    {
        bool is_fp32_a = (spec->dtype_a && strcmp(spec->dtype_a, "fp32") == 0);
        int max_from_K = 1;
        int kand[] = {is_fp32_a ? 4 : 8, 4, 2, 1};
        for(int v : kand)
            if(kpg % v == 0)
            {
                max_from_K = v;
                break;
            }
        int safe_vec = max_from_K;
        rocke_coalesced_tile_loader_choose_vec(block_m, block_k, block_size, max_from_K, &safe_vec);
        load_vec_a = spec->has_vector_size_a ? spec->vector_size_a : safe_vec;
    }
    // load_vec_b: B (W, KYXC) — the GEMM row axis is N_dg = c (input channels), which
    // is the stride-1 axis of KYXC.  Vectorise along the free (row) axis and transpose
    // into the row-major (N, K) LDS tile on store (vector_axis_row=true),
    // exactly as wgrad does for its B (X, NHWC) operand.  Condition: cpg % load_vec_b
    // == 0 (cpg == C when ungrouped); the group base g*cpg is a multiple of cpg.
    int load_vec_b = 1;
    bool axis_b_row = false;
    {
        bool is_fp32_b = (spec->dtype_b && strcmp(spec->dtype_b, "fp32") == 0);
        int cand[] = {is_fp32_b ? 4 : 8, 4, 2, 1};
        int max_from_C = 1;
        for(int v : cand)
            if(cpg % v == 0)
            {
                max_from_C = v;
                break;
            }
        int chosen = 1;
        rocke_status_t st = rocke_coalesced_tile_loader_choose_vec_axis(
            block_n, block_k, threads, max_from_C, true, &chosen);
        if(st != ROCKE_OK)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "dgrad tilde: no usable free-axis load_vec for B tile geometry");
            return NULL;
        }
        if(spec->lds_k_outer)
        {
            /* Recompute in col mode over the swapped tile. choose_vec tests
             * tile_rows in row mode and tile_cols in col mode, so both calls
             * test the same extent against the same product: load_vec_b is
             * invariant under the swap and only the LDS store changes. */
            int cap = spec->has_vector_size_b
                          ? (spec->vector_size_b < max_from_C ? spec->vector_size_b : max_from_C)
                          : max_from_C;
            int chosen_ko = 1;
            rocke_status_t st_ko = rocke_coalesced_tile_loader_choose_vec_axis(
                block_k, block_n, threads, cap, false, &chosen_ko);
            if(st_ko != ROCKE_OK)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "dgrad tilde: no usable K-outer load_vec for B tile geometry");
                return NULL;
            }
            load_vec_b = chosen_ko;
            axis_b_row = false;
        }
        else if(spec->has_vector_size_b)
        {
            /* Clamp, exactly as the K-outer branch above does. vector_size_* is
             * a CAP, not a demand, so an explicit width wider than the tile
             * geometry supports must be narrowed rather than obeyed. Taking it
             * verbatim let a spec pass validation and then fail inside the
             * coalesced tile loader. Emission-neutral: choose_vec's accepted
             * set is a strict subset of vecs_per_thread's, so this yields
             * exactly spec->vector_size_b wherever the verbatim path built. */
            int cap_mo = spec->vector_size_b < max_from_C ? spec->vector_size_b : max_from_C;
            int chosen_mo = 1;
            rocke_status_t st_mo = rocke_coalesced_tile_loader_choose_vec_axis(
                block_n, block_k, threads, cap_mo, true, &chosen_mo);
            if(st_mo != ROCKE_OK)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "dgrad tilde: no usable free-axis load_vec for B tile geometry");
                return NULL;
            }
            load_vec_b = chosen_mo;
            axis_b_row = (load_vec_b > 1);
        }
        else if(chosen > 1)
        {
            load_vec_b = chosen;
            axis_b_row = true;
        }
    }

    rocke_coalesced_tile_loader_t a_sync_loader;
    a_sync_loader.tile_rows = block_m;
    a_sync_loader.tile_cols = block_k;
    a_sync_loader.block_size = threads;
    a_sync_loader.load_vec = load_vec_a;
    a_sync_loader.use_buffer_rsrc = true;
    a_sync_loader.oob_sentinel = 2147483647;
    a_sync_loader.vector_axis_row = false;
    a_sync_loader.has_inner_dim = false;
    a_sync_loader.inner_dim = 0;

    rocke_coalesced_tile_loader_t b_sync_loader;
    b_sync_loader.tile_rows = spec->lds_k_outer ? block_k : block_n;
    b_sync_loader.tile_cols = spec->lds_k_outer ? block_n : block_k;
    b_sync_loader.block_size = threads;
    b_sync_loader.load_vec = load_vec_b;
    b_sync_loader.use_buffer_rsrc = true;
    b_sync_loader.oob_sentinel = 2147483647;
    b_sync_loader.vector_axis_row = axis_b_row;
    b_sync_loader.has_inner_dim = false;
    b_sync_loader.inner_dim = 0;

    // ---- wavelet loaders (pipeline="wavelet" only) ----
    rocke_coalesced_tile_loader_t a_wavelet_loader;
    rocke_coalesced_tile_loader_t b_wavelet_loader;
    rocke_value_t* wavelet_is_math = NULL;
    rocke_value_t* wavelet_load_tid = NULL;
    int wavelet_epi_barriers = 0;
    int wavelet_K_iters = 0;
    if(is_wavelet)
    {
        int load_threads = spec->num_load_waves * spec->wave_size;
        rocke_status_t sa = rocke_coalesced_tile_loader_from_tile(
            block_m, block_k, load_threads, load_vec_a, true, &a_wavelet_loader);
        rocke_status_t sb = rocke_coalesced_tile_loader_from_tile(
            block_n, block_k, load_threads, load_vec_b, true, &b_wavelet_loader);
        if(sa != ROCKE_OK || sb != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "dgrad: wavelet tile loader from_tile failed");
            return NULL;
        }
        int n_math_warps = spec->warp_m * spec->warp_n;
        rocke_value_t* c_nmath = rocke_b_const_i32(b, n_math_warps);
        rocke_value_t* warp_id_s = rocke_b_readfirstlane(b, warp_id);
        wavelet_is_math = rocke_b_cmp_lt(b, warp_id_s, c_nmath);
        wavelet_load_tid = rocke_b_sub(b, tid, rocke_b_const_i32(b, block_size));

        /* epi_barriers mirrors compute_wavelet_epi_barriers(spec.epilogue, no_alias).
         * no_alias=true for wavelet (A/B live across both branches). */
        bool no_alias = true;
        const int war_barriers = 2;
        bool use_cshuffle = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
        wavelet_epi_barriers = use_cshuffle ? (no_alias ? 0 : war_barriers) + 1 : 0;

        /* K_iters uses dg_K_padded (worst-case across sub-GEMMs) so it is a
         * compile-time constant that wavelet can unroll. */
        int dg_K_padded = rocke_dgrad_conv_spec_dg_K_padded(spec);
        int slice_k = (spec->split_k <= 1) ? dg_K_padded : (dg_K_padded / spec->split_k);
        wavelet_K_iters = _ceil_div(slice_k, block_k);
    }

    tilde_dy_ctx_t dy_tctx;
    dy_tctx.block_m_off = block_m_off_v;
    dy_tctx.k_off = NULL;
    dy_tctx.kpg_div = kpg_div;
    dy_tctx.xdot_div = xdot_div;
    dy_tctx.hw_div = hw_div;
    dy_tctx.w_div = w_div;
    dy_tctx.rec_h_tilde_slice_begin = rec_h_tilde_slice_begin;
    dy_tctx.rec_w_tilde_slice_begin = rec_w_tilde_slice_begin;
    dy_tctx.rec_a_embed_h_coeff = rec_a_embed_h_coeff;
    dy_tctx.rec_a_embed_w_coeff = rec_a_embed_w_coeff;
    dy_tctx.c_Ho = c_Ho;
    dy_tctx.c_Wo = c_Wo;
    dy_tctx.c_K = c_K;
    dy_tctx.c_kdiv = c_kdiv;
    dy_tctx.k_out_group_base = k_out_group_base;
    dy_tctx.c0 = c0;
    dy_tctx.is_pointwise = rocke_conv_problem_is_pointwise(p) && p->groups <= 1;
    dy_tctx.dg_M = c_dY_rows;

    tilde_w_ctx_t w_tctx;
    w_tctx.block_n_off = block_n_off_v;
    w_tctx.k_off = NULL;
    w_tctx.kpg_div = kpg_div;
    w_tctx.xdot_div = xdot_div;
    w_tctx.rec_b_y_stride = rec_b_y_stride;
    w_tctx.rec_b_y_offset = rec_b_y_offset;
    w_tctx.rec_b_x_stride = rec_b_x_stride;
    w_tctx.rec_b_x_offset = rec_b_x_offset;
    w_tctx.c_Y = c_Y;
    w_tctx.c_X = c_X;
    w_tctx.c_K = c_K;
    w_tctx.c_C = c_C;
    w_tctx.c_kdiv = c_kdiv;
    w_tctx.k_out_group_base = k_out_group_base;
    w_tctx.c_cstride = c_cstride;
    w_tctx.c0 = c0;
    w_tctx.is_pointwise = rocke_conv_problem_is_pointwise(p) && p->groups <= 1;

    // ---- schedule ----
    rocke_schedule_policy_t schedule = rocke_schedule_policy_for_pipeline(b, spec->pipeline);
    rocke_schedule_policy_emit_prologue(&schedule, b);

    /* Hoisted lane constants for the K-outer transpose read. Guarded because
     * unconditional emission would add ops to every existing dgrad config and
     * move every dgrad golden. Emitted here to match Python's position in the
     * instruction stream (immediately after the schedule prologue). */
    rocke_value_t* tr_lane_mod4 = NULL;
    rocke_value_t* tr_grp16 = NULL;
    /* Element type for the transpose read -- see rocke_conv_tr_elem_dtype.
     * Type selection only, emits no IR, so it is computed unconditionally. */
    const rocke_type_t* tr_dtype = rocke_conv_tr_elem_dtype(spec->dtype_a);
    if(spec->lds_k_outer && spec->wave_size == 64)
    {
        /* Python: b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)) -- evaluated
         * strictly left-to-right. C argument order is unspecified, so sequence
         * every operand into a temporary or the SSA numbering drifts. */
        rocke_value_t* c4a = rocke_b_const_i32(b, 4);
        rocke_value_t* m4 = rocke_b_mod(b, lane, c4a);
        rocke_value_t* c4b = rocke_b_const_i32(b, 4);
        tr_lane_mod4 = rocke_b_mul(b, m4, c4b);

        /* Python: b.div(b.mod(lane, b.const_i32(16)), b.const_i32(4)) */
        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
        rocke_value_t* m16 = rocke_b_mod(b, lane, c16);
        rocke_value_t* c4c = rocke_b_const_i32(b, 4);
        tr_grp16 = rocke_b_div(b, m16, c4c);
    }

    // ---- helper lambda-equivalent: emit WMMA phase from LDS into accs ----
    // Used by both the wavelet and standard K-loop paths.
    auto emit_wmma_phase = [&](rocke_value_t* A_src,
                               rocke_value_t* B_src,
                               rocke_value_t* const* in_accs,
                               rocke_value_t** out_accs) {
        const rocke_arch_layout_map_t* a_map = rocke_mmaop_a_layout(op, b);
        const rocke_arch_layout_map_t* b_map = rocke_mmaop_b_layout(op, b);
        rocke_value_t* a_row_in_atom = NULL;
        rocke_value_t* a_k_in_atom = NULL;
        rocke_value_t* b_k_in_atom = NULL;
        rocke_value_t* b_col_in_atom = NULL;
        rocke_arch_layout_map_coord(a_map, b, lane, 0, &a_row_in_atom, &a_k_in_atom);
        rocke_arch_layout_map_coord(b_map, b, lane, 0, &b_k_in_atom, &b_col_in_atom);
        rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, &grid);
        rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, &grid);
        rocke_value_t* a_rows[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* b_wma_cols[ROCKE_CONV_MAX_ACCS];
        for(int i = 0; i < num_accs; i++)
            out_accs[i] = in_accs[i];
        for(int kk = 0; kk < k_atoms; kk++)
        {
            rocke_value_t* k_tile_base = rocke_b_const_i32(b, kk * spec->warp_tile_k);
            for(int mi = 0; mi < mfmas_m; mi++)
            {
                rocke_value_t* atom_row
                    = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * spec->warp_tile_m));
                a_rows[mi] = rocke_conv_emit_frag_smem_load(
                    b, A_src, a_row_in_atom, a_k_in_atom, atom_row, k_tile_base, a_per_lane);
            }
            for(int ni = 0; ni < mfmas_n; ni++)
            {
                rocke_value_t* atom_row
                    = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * spec->warp_tile_n));
                if(spec->lds_k_outer)
                {
                    /* B only: dgrad's A tile is genuinely still M-outer. This
                     * branch existed in the MFMA phase but not here, so a
                     * wave32 K-outer dgrad silently fell back to ordinary
                     * M-outer smem loads in the C engine while Python emitted
                     * the transpose read -- there was no gfx1250 dgrad parity
                     * config to catch the divergence. */
                    b_wma_cols[ni] = rocke_conv_tr_frag(b,
                                                        lane,
                                                        tr_lane_mod4,
                                                        tr_grp16,
                                                        B_src,
                                                        atom_row,
                                                        k_tile_base,
                                                        spec->warp_tile_n,
                                                        b_per_lane,
                                                        spec->wave_size,
                                                        tr_dtype);
                    continue;
                }
                b_wma_cols[ni] = rocke_conv_emit_frag_smem_load(
                    b, B_src, b_col_in_atom, b_k_in_atom, atom_row, k_tile_base, b_per_lane);
            }
            int flat2 = 0;
            for(int mi = 0; mi < mfmas_m; mi++)
                for(int ni = 0; ni < mfmas_n; ni++)
                {
                    out_accs[flat2] = rocke_b_mma(
                        b, op->op_id, a_rows[mi], b_wma_cols[ni], out_accs[flat2], NULL, 0);
                    flat2++;
                }
        }
    };

    // ---- helper: dispatch dgrad epilogue ----
    auto dispatch_dgrad_epilogue = [&](rocke_value_t* const* epi_accs_, int n_epi) {
        bool is_split_k_atomic_ = (spec->split_k > 1);
        bool is_strided_ = rocke_dgrad_conv_spec_is_strided(spec);
        if(!is_split_k_atomic_ && !is_strided_)
        {
            if(is_wmma)
            {
                bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
                if(use_cshuffle_)
                    _emit_dgrad_direct_epilogue(
                        b, spec, &ctx, epi_accs_, n_epi, &grid, dx_rsrc, c_group_base);
                else
                    _emit_dgrad_direct_epilogue_wmma(b,
                                                     spec,
                                                     &ctx,
                                                     op,
                                                     epi_accs_,
                                                     n_epi,
                                                     warp_m_idx,
                                                     warp_n_idx,
                                                     lane,
                                                     block_m_off_v,
                                                     block_n_off_v,
                                                     dx_rsrc,
                                                     c0,
                                                     c_group_base);
            }
            else
            {
                bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
                if(use_cshuffle_)
                    _emit_dgrad_cshuffle_epilogue(
                        b, spec, &ctx, epi_accs_, n_epi, &grid, dx_rsrc, c_group_base);
                else
                    _emit_dgrad_direct_epilogue(
                        b, spec, &ctx, epi_accs_, n_epi, &grid, dx_rsrc, c_group_base);
            }
        }
        else if(!is_split_k_atomic_ && is_wmma)
        {
            _emit_dgrad_tilde_direct_epilogue_wmma(b,
                                                   spec,
                                                   op,
                                                   epi_accs_,
                                                   n_epi,
                                                   warp_m_idx,
                                                   warp_n_idx,
                                                   lane,
                                                   block_m_off_v,
                                                   block_n_off_v,
                                                   dx_rsrc,
                                                   c0,
                                                   rec_gemm_m,
                                                   c_dg_N,
                                                   &hw_div,
                                                   &w_div,
                                                   rec_d_h_stride,
                                                   rec_d_h_offset,
                                                   rec_d_w_stride,
                                                   rec_d_w_offset,
                                                   c_Hi,
                                                   c_Wi,
                                                   c_C,
                                                   c_group_base);
        }
        else if(!is_split_k_atomic_ && !is_wmma && atom)
        {
            bool use_cshuffle_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
            if(use_cshuffle_)
                _emit_dgrad_tilde_cshuffle_epilogue(b,
                                                    spec,
                                                    atom,
                                                    &grid,
                                                    epi_accs_,
                                                    n_epi,
                                                    dx_rsrc,
                                                    rec_gemm_m,
                                                    c_dg_N,
                                                    &hw_div,
                                                    &w_div,
                                                    rec_d_h_stride,
                                                    rec_d_h_offset,
                                                    rec_d_w_stride,
                                                    rec_d_w_offset,
                                                    c_Hi,
                                                    c_Wi,
                                                    c_C,
                                                    c_group_base,
                                                    patch_xc,
                                                    patch_row);
            else
                _emit_dgrad_tilde_direct_epilogue(b,
                                                  spec,
                                                  atom,
                                                  &grid,
                                                  epi_accs_,
                                                  n_epi,
                                                  dx_rsrc,
                                                  rec_gemm_m,
                                                  c_dg_N,
                                                  &hw_div,
                                                  &w_div,
                                                  rec_d_h_stride,
                                                  rec_d_h_offset,
                                                  rec_d_w_stride,
                                                  rec_d_w_offset,
                                                  c_Hi,
                                                  c_Wi,
                                                  c_C,
                                                  c_group_base,
                                                  patch_xc,
                                                  patch_row);
        }
        else
        {
            _emit_dgrad_tilde_atomic_epilogue(b,
                                              spec,
                                              atom,
                                              epi_accs_,
                                              n_epi,
                                              warp_m_idx,
                                              warp_n_idx,
                                              lane,
                                              block_m_off_v,
                                              block_n_off_v,
                                              dX,
                                              c_per_lane,
                                              rec_gemm_m,
                                              c_dg_N,
                                              &hw_div,
                                              &w_div,
                                              rec_d_h_stride,
                                              rec_d_h_offset,
                                              rec_d_w_stride,
                                              rec_d_w_offset,
                                              c_Hi,
                                              c_Wi,
                                              c_C,
                                              c_group_base);
        }
    };

    // ---- wavelet K-loop (pipeline="wavelet", WMMA/gfx1250 only) ----
    if(is_wavelet)
    {
        /* WMMA path: scf_if_else with a shared join block (gfx1250).
         * Barrier protocol mirrors rocke_conv_emit_kloop_wavelet WMMA branch. */
        rocke_ctl_staged_t a_staged;
        rocke_ctl_staged_t b_staged;
        rocke_if_else_t ife = rocke_b_scf_if_else(b, wavelet_is_math);

        // ---- MATH WAVE branch ----
        rocke_value_t* current_accs[ROCKE_CONV_MAX_ACCS];
        rocke_value_t* new_accs_wv[ROCKE_CONV_MAX_ACCS];
        for(int i = 0; i < num_accs; i++)
            current_accs[i] = iter_args[i].init;

        rocke_b_region_enter(b, ife.then_region);
        {
            rocke_b_sync(b); /* barrier_0 */
            for(int it = 0; it < wavelet_K_iters - 1; it++)
            {
                dy_tctx.k_off = rocke_b_const_i32(b, it * block_k);
                w_tctx.k_off = rocke_b_const_i32(b, it * block_k);
                emit_wmma_phase(A_smem, B_smem, current_accs, new_accs_wv);
                for(int i = 0; i < num_accs; i++)
                    current_accs[i] = new_accs_wv[i];
                rocke_b_sync(b); /* barrier_A */
                rocke_b_sync(b); /* barrier_B */
            }
            /* tail MFMA -- no barriers */
            dy_tctx.k_off = rocke_b_const_i32(b, (wavelet_K_iters - 1) * block_k);
            w_tctx.k_off = rocke_b_const_i32(b, (wavelet_K_iters - 1) * block_k);
            emit_wmma_phase(A_smem, B_smem, current_accs, new_accs_wv);
            for(int i = 0; i < num_accs; i++)
                current_accs[i] = new_accs_wv[i];

            /* epilogue (inside math branch, no iter-var yield) */
            rocke_value_t* epi_accs[ROCKE_CONV_MAX_ACCS];
            rocke_conv_apply_accumulator_epilogue(
                b, &spec->acc_epilogue, current_accs, num_accs, epi_accs);
            dispatch_dgrad_epilogue(epi_accs, num_accs);
        }
        rocke_b_region_leave(b);

        // ---- LOAD WAVE branch ----
        rocke_b_region_enter(b, ife.else_region);
        {
            /* fetch tile 0 -> regs, store -> LDS, barrier_0 */
            dy_tctx.k_off = c0;
            w_tctx.k_off = c0;
            rocke_coalesced_tile_loader_load_global(b,
                                                    &a_wavelet_loader,
                                                    wavelet_load_tid,
                                                    _tilde_dy_descriptor,
                                                    &dy_tctx,
                                                    dy_rsrc,
                                                    NULL,
                                                    &a_staged);
            rocke_coalesced_tile_loader_load_global(b,
                                                    &b_wavelet_loader,
                                                    wavelet_load_tid,
                                                    _tilde_w_descriptor,
                                                    &w_tctx,
                                                    w_rsrc,
                                                    NULL,
                                                    &b_staged);
            rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
            rocke_coalesced_tile_loader_store_lds(b, &a_wavelet_loader, A_smem, &a_staged);
            rocke_coalesced_tile_loader_store_lds(b, &b_wavelet_loader, B_smem, &b_staged);
            rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
            rocke_b_sync(b); /* barrier_0 */

            for(int it = 0; it < wavelet_K_iters - 1; it++)
            {
                dy_tctx.k_off = rocke_b_const_i32(b, (it + 1) * block_k);
                w_tctx.k_off = rocke_b_const_i32(b, (it + 1) * block_k);
                rocke_coalesced_tile_loader_load_global(b,
                                                        &a_wavelet_loader,
                                                        wavelet_load_tid,
                                                        _tilde_dy_descriptor,
                                                        &dy_tctx,
                                                        dy_rsrc,
                                                        NULL,
                                                        &a_staged);
                rocke_coalesced_tile_loader_load_global(b,
                                                        &b_wavelet_loader,
                                                        wavelet_load_tid,
                                                        _tilde_w_descriptor,
                                                        &w_tctx,
                                                        w_rsrc,
                                                        NULL,
                                                        &b_staged);
                rocke_b_sync(b); /* barrier_A */
                rocke_b_s_waitcnt(b, 0, -1, -1); /* vmcnt=0 */
                rocke_coalesced_tile_loader_store_lds(b, &a_wavelet_loader, A_smem, &a_staged);
                rocke_coalesced_tile_loader_store_lds(b, &b_wavelet_loader, B_smem, &b_staged);
                rocke_b_s_waitcnt(b, -1, 0, -1); /* lgkmcnt=0 */
                rocke_b_sync(b); /* barrier_B */
            }
            /* epilogue stub: epi_barriers bare barriers matching math branch */
            for(int i = 0; i < wavelet_epi_barriers; i++)
                rocke_b_sync(b);
        }
        rocke_b_region_leave(b);

        return b->kernel;
    }

    // ---- MFMA phase (shared by every synchronous K loop below) ----
    // Emits exactly the ops of the former inline loop body, so a spec with
    // a single K loop lowers byte-identically.
    typedef std::function<rocke_value_t*(rocke_value_t*)> a_row_map_fn;
    auto emit_mfma_phase_src = [&](rocke_value_t* const* in_vars,
                                   rocke_value_t** new_accs,
                                   rocke_value_t* B_src,
                                   const a_row_map_fn* a_row_map) {
        if(!is_wmma && atom)
        {
            rocke_lane_decode_t decoded = rocke_decode_mfma_lanes(b, atom, lane);
            rocke_value_t* m_in_atom = decoded.m_in_atom;
            rocke_value_t* n_in_atom = decoded.n_in_atom;
            rocke_value_t* k_blk = decoded.k_blk;
            rocke_value_t* warp_m_off = rocke_warp_grid_warp_m_off(b, &grid);
            rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, &grid);

            for(int i = 0; i < num_accs; i++)
                new_accs[i] = in_vars[i];

            for(int kk = 0; kk < k_atoms; kk++)
            {
                rocke_value_t* _cb_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, a_per_lane));
                rocke_value_t* col_base
                    = rocke_b_add(b, _cb_mul, rocke_b_const_i32(b, kk * spec->warp_tile_k));

                rocke_value_t* a_rows[ROCKE_CONV_MAX_ACCS];
                for(int mi = 0; mi < mfmas_m; mi++)
                {
                    rocke_value_t* a_row = rocke_b_add(
                        b,
                        warp_m_off,
                        rocke_b_add(b, rocke_b_const_i32(b, mi * spec->warp_tile_m), m_in_atom));
                    /* Python a_row_map (the dY halo loop). */
                    if(a_row_map)
                        a_row = (*a_row_map)(a_row);
                    a_rows[mi] = rocke_conv_emit_smem_load(b, A_smem, a_row, col_base, a_per_lane);
                }

                rocke_value_t* b_cols[ROCKE_CONV_MAX_ACCS];
                for(int ni = 0; ni < mfmas_n; ni++)
                {
                    if(spec->lds_k_outer)
                    {
                        /* Python skips the b_row computation entirely on this
                             * path, so emitting it here would add ops the Python
                             * engine never emits. Operands sequenced into
                             * temporaries to preserve left-to-right evaluation. */
                        rocke_value_t* mn_c = rocke_b_const_i32(b, ni * spec->warp_tile_n);
                        rocke_value_t* mn_base = rocke_b_add(b, warp_n_off, mn_c);
                        rocke_value_t* k_c = rocke_b_const_i32(b, kk * spec->warp_tile_k);
                        b_cols[ni] = rocke_conv_tr_frag(b,
                                                        lane,
                                                        tr_lane_mod4,
                                                        tr_grp16,
                                                        B_src,
                                                        mn_base,
                                                        k_c,
                                                        spec->warp_tile_n,
                                                        b_per_lane,
                                                        spec->wave_size,
                                                        tr_dtype);
                        continue;
                    }
                    rocke_value_t* b_row = rocke_b_add(
                        b,
                        warp_n_off,
                        rocke_b_add(b, rocke_b_const_i32(b, ni * spec->warp_tile_n), n_in_atom));
                    b_cols[ni] = rocke_conv_emit_smem_load(b, B_src, b_row, col_base, b_per_lane);
                }

                int flat = 0;
                for(int mi = 0; mi < mfmas_m; mi++)
                {
                    for(int ni = 0; ni < mfmas_n; ni++)
                    {
                        new_accs[flat]
                            = rocke_conv_emit_mfma(b, atom, a_rows[mi], b_cols[ni], new_accs[flat]);
                        flat++;
                    }
                }

                rocke_schedule_policy_emit_after_mfma_step(
                    &schedule, b, mfmas_m + mfmas_n, mfmas_m * mfmas_n);
            }
        }
        else if(is_wmma)
        {
            emit_wmma_phase(A_smem, B_src, in_vars, new_accs);
        }
        else
        {
            for(int i = 0; i < num_accs; i++)
                new_accs[i] = in_vars[i];
        }
    };
    auto emit_mfma_phase = [&](rocke_value_t* const* in_vars, rocke_value_t** new_accs) {
        emit_mfma_phase_src(in_vars, new_accs, B_smem, NULL);
    };

    // ---- plain K-loop body: per-element descriptors, read+write per chunk ----
    auto emit_plain_body = [&](rocke_value_t* kv, rocke_value_t* const* in_vars) {
        // Set k_off_capture for descriptor closures
        dy_tctx.k_off = kv;
        w_tctx.k_off = kv;

        rocke_coalesced_tile_loader_load(
            b, &a_sync_loader, tid, A_smem, _tilde_dy_descriptor, &dy_tctx, dy_rsrc, NULL);
        rocke_coalesced_tile_loader_load(b,
                                         &b_sync_loader,
                                         tid,
                                         B_smem,
                                         spec->lds_k_outer ? _tilde_w_descriptor_kouter
                                                           : _tilde_w_descriptor,
                                         &w_tctx,
                                         w_rsrc,
                                         NULL);
        rocke_b_sync(b);

        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
        emit_mfma_phase(in_vars, new_accs);

        rocke_b_sync(b);
        rocke_b_scf_yield(b, new_accs, num_accs);
    };

    rocke_value_t* final_accs[ROCKE_CONV_MAX_ACCS];
    int num_final = 0;
    if(spec->dy_halo)
    {
        /* Python _emit_dy_halo_loop: output-channel chunk outer, filter tap
         * inner (unrolled over the baked Y x X), one staged dY halo per
         * chunk. The pad is baked with the filter ((Y-1)/2, (X-1)/2 at stride
         * 1 with Ho == Hi), Wo and everything derived from it is runtime. */
        const int Y = p->Y;
        const int X = p->X;
        const int pH = p->pH;
        const int pW = p->pW;
        const int lds_rows = _dgrad_dy_halo_lds_rows(spec);
        const int load_rows = lds_rows - 1;
        rocke_value_t* c_kd = c_kdiv;
        rocke_value_t* c_cs = c_cstride;
        rocke_value_t* c_yx = rocke_b_const_i32(b, Y * X);
        rocke_value_t* c_yxc = rocke_b_mul(b, c_yx, c_cs);

        rocke_coalesced_tile_loader_t halo_loader = a_sync_loader;
        halo_loader.tile_rows = load_rows;

        /* min_shift = (pH - Y + 1)*Wo + (pW - X + 1); real rows. */
        rocke_value_t* c_ms_h = rocke_b_const_i32(b, pH - Y + 1);
        rocke_value_t* ms_h = rocke_b_mul(b, c_ms_h, c_Wo);
        rocke_value_t* c_ms_w = rocke_b_const_i32(b, pW - X + 1);
        rocke_value_t* min_shift = rocke_b_add(b, ms_h, c_ms_w);
        rocke_value_t* c_rr_h = rocke_b_const_i32(b, Y - 1);
        rocke_value_t* rr_h = rocke_b_mul(b, c_rr_h, c_Wo);
        rocke_value_t* c_rr_w = rocke_b_const_i32(b, block_m + X - 1);
        rocke_value_t* real_rows = rocke_b_add(b, rr_h, c_rr_w);
        rocke_value_t* halo_base = rocke_b_add(b, block_m_off_v, min_shift);
        rocke_value_t* c_zero_row = rocke_b_const_i32(b, lds_rows - 1);

        halo_dy_ctx_t hctx;
        hctx.halo_base = halo_base;
        hctx.c0 = c0;
        hctx.npix = ctx.p_dg_M;
        hctx.real_rows = real_rows;
        hctx.c_K = c_K;
        hctx.kb = NULL;
        tap_w_ctx_t tw;
        tw.block_n_off = block_n_off_v;
        tw.c_yxc = c_yxc;
        tw.w_k = NULL;
        rocke_loads_descriptor_fn w_desc
            = spec->lds_k_outer ? _tap_w_descriptor_kouter : _tap_w_descriptor;

        /* Zero row: every thread writes the same zeros; the first chunk's
         * barrier orders it before any read and the loader never writes it. */
        const int zvec = block_k < 8 ? block_k : 8;
        rocke_value_t* c_zc = rocke_b_const_i32(b, block_k / zvec);
        rocke_value_t* z_mod = rocke_b_mod(b, tid, c_zc);
        rocke_value_t* c_zv = rocke_b_const_i32(b, zvec);
        rocke_value_t* zcol = rocke_b_mul(b, z_mod, c_zv);
        rocke_value_t* zval = rocke_b_zero_vec(b, ab_ir, zvec);
        rocke_value_t* zidx[2] = {c_zero_row, zcol};
        rocke_b_smem_store_vN(b, A_smem, zidx, 2, zval, zvec);

        /* Python row_map(y, x): the tap's scalars once, then per fragment
         * row the (hi, wi) decode, its bounds and the select. */
        auto row_map = [&](int y, int x) -> a_row_map_fn {
            const int dh = pH - y;
            const int dw = pW - x;
            rocke_value_t* c_sh = rocke_b_const_i32(b, Y - 1 - y);
            rocke_value_t* sh_h = rocke_b_mul(b, c_sh, c_Wo);
            rocke_value_t* c_sw = rocke_b_const_i32(b, X - 1 - x);
            rocke_value_t* shift = rocke_b_add(b, sh_h, c_sw);
            rocke_value_t* h_hi = NULL;
            if(dh > 0)
            {
                rocke_value_t* c_dh = rocke_b_const_i32(b, dh);
                h_hi = rocke_b_sub(b, c_Ho, c_dh);
            }
            rocke_value_t* w_hi = NULL;
            if(dw > 0)
            {
                rocke_value_t* c_dw = rocke_b_const_i32(b, dw);
                w_hi = rocke_b_sub(b, c_Wo, c_dw);
            }
            return [=, &hw_div, &w_div](rocke_value_t* a_row) -> rocke_value_t* {
                rocke_value_t* m = rocke_b_add(b, block_m_off_v, a_row);
                rocke_value_t *n_q, *rem, *hi, *wi;
                _magic_divmod(b, m, &hw_div, &n_q, &rem);
                _magic_divmod(b, rem, &w_div, &hi, &wi);
                rocke_value_t* conds[4];
                int nc = 0;
                if(dh < 0)
                {
                    rocke_value_t* c = rocke_b_const_i32(b, -dh);
                    conds[nc++] = rocke_b_cmp_ge(b, hi, c);
                }
                if(h_hi)
                    conds[nc++] = rocke_b_cmp_lt(b, hi, h_hi);
                if(dw < 0)
                {
                    rocke_value_t* c = rocke_b_const_i32(b, -dw);
                    conds[nc++] = rocke_b_cmp_ge(b, wi, c);
                }
                if(w_hi)
                    conds[nc++] = rocke_b_cmp_lt(b, wi, w_hi);
                rocke_value_t* row = rocke_b_add(b, a_row, shift);
                if(nc == 0)
                    return row;
                rocke_value_t* ok = conds[0];
                for(int i = 1; i < nc; i++)
                    ok = rocke_b_land(b, ok, conds[i]);
                return rocke_b_select(b, ok, row, c_zero_row);
            };
        };

        const int n_taps = Y * X;
        auto mfma_tap
            = [&](int t, rocke_value_t* b_src, rocke_value_t* const* cur_in, rocke_value_t** out) {
                  const int y = t / X;
                  const int x = t % X;
                  if(spec->dy_halo_setprio > 0)
                      rocke_b_s_setprio(b, spec->dy_halo_setprio);
                  a_row_map_fn map = row_map(y, x);
                  emit_mfma_phase_src(cur_in, out, b_src, &map);
                  if(spec->dy_halo_setprio > 0)
                      rocke_b_s_setprio(b, 0);
              };

        rocke_value_t* bufs[2] = {B_smem, B_smem2};
        rocke_for_t chunk_for
            = rocke_b_scf_for_iter(b, c0, c_kd, c_block_k, iter_args, num_accs, "kb", false, true);
        rocke_b_region_enter(b, chunk_for.body);
        {
            rocke_value_t* kb0 = chunk_for.iv;
            rocke_value_t* kb = grouped ? rocke_b_add(b, kb0, k_out_group_base) : kb0;
            hctx.kb = kb;
            rocke_value_t* w_kb = rocke_b_mul(b, kb, c_yxc);
            auto set_tap = [&](int t) {
                rocke_value_t* c_t = rocke_b_const_i32(b, t);
                rocke_value_t* t_cs = rocke_b_mul(b, c_t, c_cs);
                tw.w_k = rocke_b_add(b, w_kb, t_cs);
            };
            rocke_value_t* cur[ROCKE_CONV_MAX_ACCS];
            rocke_value_t* nxt_accs[ROCKE_CONV_MAX_ACCS];
            for(int i = 0; i < num_accs; i++)
                cur[i] = chunk_for.iter_vars[i];
            if(spec->dy_halo == 2)
            {
                /* Chunk prologue: the halo and tap 0's B tile. */
                set_tap(0);
                rocke_ctl_staged_t a_st;
                rocke_ctl_staged_t b_st;
                rocke_coalesced_tile_loader_load_global(
                    b, &halo_loader, tid, _halo_dy_descriptor, &hctx, dy_rsrc, NULL, &a_st);
                rocke_coalesced_tile_loader_load_global(
                    b, &b_sync_loader, tid, w_desc, &tw, w_rsrc, NULL, &b_st);
                rocke_coalesced_tile_loader_store_lds(b, &halo_loader, A_smem, &a_st);
                rocke_coalesced_tile_loader_store_lds(b, &b_sync_loader, bufs[0], &b_st);
                rocke_b_sync(b);
                for(int t = 0; t < n_taps; t++)
                {
                    rocke_ctl_staged_t nxt;
                    const bool has_nxt = t + 1 < n_taps;
                    if(has_nxt)
                    {
                        /* Tap t+1's B tile into registers, pinned ahead of
                         * the MFMAs. */
                        set_tap(t + 1);
                        rocke_coalesced_tile_loader_load_global(
                            b, &b_sync_loader, tid, w_desc, &tw, w_rsrc, NULL, &nxt);
                        rocke_b_sched_barrier(b, 0);
                    }
                    mfma_tap(t, bufs[t % 2], cur, nxt_accs);
                    for(int i = 0; i < num_accs; i++)
                        cur[i] = nxt_accs[i];
                    if(has_nxt)
                    {
                        rocke_b_sched_barrier(b, 0);
                        rocke_coalesced_tile_loader_store_lds(
                            b, &b_sync_loader, bufs[(t + 1) % 2], &nxt);
                    }
                    rocke_b_sync(b);
                }
            }
            else
            {
                for(int t = 0; t < n_taps; t++)
                {
                    set_tap(t);
                    rocke_ctl_staged_t a_st;
                    rocke_ctl_staged_t b_st;
                    if(t == 0)
                        rocke_coalesced_tile_loader_load_global(
                            b, &halo_loader, tid, _halo_dy_descriptor, &hctx, dy_rsrc, NULL, &a_st);
                    rocke_coalesced_tile_loader_load_global(
                        b, &b_sync_loader, tid, w_desc, &tw, w_rsrc, NULL, &b_st);
                    if(t == 0)
                        rocke_coalesced_tile_loader_store_lds(b, &halo_loader, A_smem, &a_st);
                    rocke_coalesced_tile_loader_store_lds(b, &b_sync_loader, bufs[0], &b_st);
                    rocke_b_sync(b);
                    mfma_tap(t, bufs[0], cur, nxt_accs);
                    for(int i = 0; i < num_accs; i++)
                        cur[i] = nxt_accs[i];
                    rocke_b_sync(b);
                }
            }
            rocke_b_scf_yield(b, cur, num_accs);
        }
        rocke_b_region_leave(b);

        for(int i = 0; i < chunk_for.op->num_results; i++)
            final_accs[i] = chunk_for.op->results[i];
        num_final = chunk_for.op->num_results;
    }
    else if(spec->tap_outer_k)
    {
        /* Python _emit_tap_outer_loops: filter tap outer, output-channel
         * chunk inner. kpg % tile_k == 0 (validator), so a chunk never
         * straddles a tap. */
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* c_kd = c_kdiv;
        rocke_value_t* c_cs = c_cstride;
        rocke_value_t* c_taps = rocke_b_mul(b, c_Y, c_X);
        rocke_value_t* c_yxc = rocke_b_mul(b, c_taps, c_cs);

        tap_dy_ctx_t tdy;
        tdy.block_m_off = block_m_off_v;
        tdy.hw_div = hw_div;
        tdy.w_div = w_div;
        tdy.c_Ho = c_Ho;
        tdy.c_Wo = c_Wo;
        tdy.c_K = c_K;
        tdy.c0 = c0;
        tdy.dh = tdy.dw = tdy.dy_tap = tdy.kb = NULL;
        tap_w_ctx_t tw;
        tw.block_n_off = block_n_off_v;
        tw.c_yxc = c_yxc;
        tw.w_k = NULL;

        rocke_for_t tap_for
            = rocke_b_scf_for_iter(b, c0, c_taps, c1, iter_args, num_accs, "tap", false, true);
        rocke_b_region_enter(b, tap_for.body);
        {
            rocke_value_t* tap = tap_for.iv;
            rocke_value_t* ydot = rocke_b_div(b, tap, c_X);
            rocke_value_t* ydot_x = rocke_b_mul(b, ydot, c_X);
            rocke_value_t* xdot = rocke_b_sub(b, tap, ydot_x);
            tdy.dh = rocke_b_sub(b, ctx.p_pH, ydot);
            tdy.dw = rocke_b_sub(b, ctx.p_pW, xdot);
            rocke_value_t* dh_wo = rocke_b_mul(b, tdy.dh, c_Wo);
            rocke_value_t* dh_wo_dw = rocke_b_add(b, dh_wo, tdy.dw);
            tdy.dy_tap = rocke_b_mul(b, dh_wo_dw, c_K);
            rocke_value_t* tap_cs = rocke_b_mul(b, tap, c_cs);

            rocke_iter_arg_t chunk_args[ROCKE_CONV_MAX_ACCS];
            char chunk_names[ROCKE_CONV_MAX_ACCS][40];
            for(int i = 0; i < num_accs; i++)
            {
                snprintf(chunk_names[i], sizeof(chunk_names[0]), "%s_t", iter_args[i].name);
                chunk_args[i].name = chunk_names[i];
                chunk_args[i].init = tap_for.iter_vars[i];
            }
            rocke_for_t chunk_for = rocke_b_scf_for_iter(
                b, c0, c_kd, c_block_k, chunk_args, num_accs, "kb", false, true);
            rocke_b_region_enter(b, chunk_for.body);
            {
                rocke_value_t* kb0 = chunk_for.iv;
                tdy.kb = grouped ? rocke_b_add(b, kb0, k_out_group_base) : kb0;
                rocke_value_t* kb_yxc = rocke_b_mul(b, tdy.kb, c_yxc);
                tw.w_k = rocke_b_add(b, kb_yxc, tap_cs);
                /* Every global read of the tile before the first LDS write. */
                rocke_ctl_staged_t a_staged;
                rocke_ctl_staged_t b_staged;
                rocke_coalesced_tile_loader_load_global(
                    b, &a_sync_loader, tid, _tap_dy_descriptor, &tdy, dy_rsrc, NULL, &a_staged);
                rocke_coalesced_tile_loader_load_global(b,
                                                        &b_sync_loader,
                                                        tid,
                                                        spec->lds_k_outer ? _tap_w_descriptor_kouter
                                                                          : _tap_w_descriptor,
                                                        &tw,
                                                        w_rsrc,
                                                        NULL,
                                                        &b_staged);
                rocke_coalesced_tile_loader_store_lds(b, &a_sync_loader, A_smem, &a_staged);
                rocke_coalesced_tile_loader_store_lds(b, &b_sync_loader, B_smem, &b_staged);
                rocke_b_sync(b);
                rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
                emit_mfma_phase(chunk_for.iter_vars, new_accs);
                rocke_b_sync(b);
                rocke_b_scf_yield(b, new_accs, num_accs);
            }
            rocke_b_region_leave(b);
            rocke_b_scf_yield(b, chunk_for.op->results, num_accs);
        }
        rocke_b_region_leave(b);

        for(int i = 0; i < tap_for.op->num_results; i++)
            final_accs[i] = tap_for.op->results[i];
        num_final = tap_for.op->num_results;
    }
    else if(spec->uniform_k_decode)
    {
        /* Python _emit_uniform_k_decode_loops. Loop 1 carries the K tile's
         * decoded (k_out, xdot, ydot) and runs when kpg (K when ungrouped) is
         * a multiple of tile_k; loop 2 is the plain loop and runs otherwise.
         * Exactly one has a non-empty range. k_out is group-local, exactly as
         * in the plain descriptors; the group's output-channel base is added
         * into both per-tile offsets. */
        rocke_value_t* c_kd = c_kdiv;
        rocke_value_t* c_xs = rec_x_dot_slice;
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* kd_rem = rocke_b_mod(b, c_kd, c_block_k);
        rocke_value_t* uk_ok = rocke_b_cmp_eq(b, kd_rem, c0);
        rocke_value_t* q_lo = rocke_b_div(b, k_lo, c_kd);
        rocke_value_t* uk_k0 = rocke_b_mod(b, k_lo, c_kd);
        rocke_value_t* uk_x0 = rocke_b_mod(b, q_lo, c_xs);
        rocke_value_t* uk_y0 = rocke_b_div(b, q_lo, c_xs);
        /* dY (NHWK): ho moves by ydot*a_embed_h_coeff rows of Wo*K, wo by
         * xdot*a_embed_w_coeff rows of K. */
        rocke_value_t* wo_k = rocke_b_mul(b, c_Wo, c_K);
        rocke_value_t* a_step_y = rocke_b_mul(b, rec_a_embed_h_coeff, wo_k);
        rocke_value_t* a_step_x = rocke_b_mul(b, rec_a_embed_w_coeff, c_K);
        /* W (KYXC, per-group packed): offset = k*Y*X*Cs + y*X*Cs + x*Cs + c,
         * Cs = cpg when grouped, C otherwise. */
        rocke_value_t* c_cs = c_cstride;
        rocke_value_t* b_step_y = rocke_b_mul(b, c_X, c_cs);
        rocke_value_t* b_step_k = rocke_b_mul(b, c_Y, b_step_y);
        rocke_value_t* uk_hi = rocke_b_select(b, uk_ok, k_hi, k_lo);

        uk_dy_ctx_t uk_dy;
        uk_dy.block_m_off = block_m_off_v;
        uk_dy.hw_div = hw_div;
        uk_dy.w_div = w_div;
        uk_dy.rec_h_tilde_slice_begin = rec_h_tilde_slice_begin;
        uk_dy.rec_w_tilde_slice_begin = rec_w_tilde_slice_begin;
        uk_dy.c_Ho = c_Ho;
        uk_dy.c_K = c_K;
        uk_dy.wo_k = wo_k;
        uk_dy.c0 = c0;
        uk_dy.ho_lo = uk_dy.ho_hi = uk_dy.wo_lo = uk_dy.wo_hi = uk_dy.a_off = NULL;
        uk_w_ctx_t uk_w;
        uk_w.block_n_off = block_n_off_v;
        uk_w.b_step_k = b_step_k;
        uk_w.c0 = c0;
        uk_w.b_off = uk_w.b_ok = NULL;

        rocke_iter_arg_t uk_args[ROCKE_CONV_MAX_ACCS + 3];
        for(int i = 0; i < num_accs; i++)
            uk_args[i] = iter_args[i];
        uk_args[num_accs + 0].name = "uk_k";
        uk_args[num_accs + 0].init = uk_k0;
        uk_args[num_accs + 1].name = "uk_x";
        uk_args[num_accs + 1].init = uk_x0;
        uk_args[num_accs + 2].name = "uk_y";
        uk_args[num_accs + 2].init = uk_y0;
        rocke_for_t for_uk = rocke_b_scf_for_iter(
            b, k_lo, uk_hi, c_block_k, uk_args, num_accs + 3, "k0", false, true);
        rocke_b_region_enter(b, for_uk.body);
        {
            rocke_value_t* ko = for_uk.iter_vars[num_accs + 0];
            rocke_value_t* xd = for_uk.iter_vars[num_accs + 1];
            rocke_value_t* yd = for_uk.iter_vars[num_accs + 2];

            /* Per-tile uniforms (Python tile_uniforms). */
            rocke_value_t* a_h = rocke_b_mul(b, yd, rec_a_embed_h_coeff);
            rocke_value_t* a_w = rocke_b_mul(b, xd, rec_a_embed_w_coeff);
            uk_dy.ho_lo = rocke_b_sub(b, c0, a_h);
            uk_dy.ho_hi = rocke_b_sub(b, c_Ho, a_h);
            uk_dy.wo_lo = rocke_b_sub(b, c0, a_w);
            uk_dy.wo_hi = rocke_b_sub(b, c_Wo, a_w);
            rocke_value_t* ay = rocke_b_mul(b, yd, a_step_y);
            rocke_value_t* a_k = rocke_b_add(b, ko, ay);
            rocke_value_t* ax = rocke_b_mul(b, xd, a_step_x);
            uk_dy.a_off = rocke_b_add(b, a_k, ax);
            if(grouped)
                uk_dy.a_off = rocke_b_add(b, uk_dy.a_off, k_out_group_base);
            rocke_value_t* y = rocke_b_add(b, rocke_b_mul(b, yd, rec_b_y_stride), rec_b_y_offset);
            rocke_value_t* x = rocke_b_add(b, rocke_b_mul(b, xd, rec_b_x_stride), rec_b_x_offset);
            rocke_value_t* k_abs = grouped ? rocke_b_add(b, ko, k_out_group_base) : ko;
            rocke_value_t* bk = rocke_b_mul(b, k_abs, b_step_k);
            rocke_value_t* by = rocke_b_mul(b, y, b_step_y);
            rocke_value_t* bx = rocke_b_mul(b, x, c_cs);
            rocke_value_t* byx = rocke_b_add(b, by, bx);
            uk_w.b_off = rocke_b_add(b, bk, byx);
            rocke_value_t* y_lt = rocke_b_cmp_lt(b, y, c_Y);
            rocke_value_t* x_lt = rocke_b_cmp_lt(b, x, c_X);
            uk_w.b_ok = rocke_b_land(b, y_lt, x_lt);

            /* Every global read of the tile before the first LDS write. */
            rocke_ctl_staged_t a_staged;
            rocke_ctl_staged_t b_staged;
            rocke_coalesced_tile_loader_load_global(
                b, &a_sync_loader, tid, _uk_dy_descriptor, &uk_dy, dy_rsrc, NULL, &a_staged);
            rocke_coalesced_tile_loader_load_global(b,
                                                    &b_sync_loader,
                                                    tid,
                                                    spec->lds_k_outer ? _uk_w_descriptor_kouter
                                                                      : _uk_w_descriptor,
                                                    &uk_w,
                                                    w_rsrc,
                                                    NULL,
                                                    &b_staged);
            rocke_coalesced_tile_loader_store_lds(b, &a_sync_loader, A_smem, &a_staged);
            rocke_coalesced_tile_loader_store_lds(b, &b_sync_loader, B_smem, &b_staged);
            rocke_b_sync(b);

            rocke_value_t* uk_new[ROCKE_CONV_MAX_ACCS + 3];
            emit_mfma_phase(for_uk.iter_vars, uk_new);
            rocke_b_sync(b);

            /* Next tile: k_out wraps into the next filter tap. */
            rocke_value_t* nk = rocke_b_add(b, ko, c_block_k);
            rocke_value_t* wrap_k = rocke_b_cmp_ge(b, nk, c_kd);
            nk = rocke_b_select(b, wrap_k, c0, nk);
            rocke_value_t* cx = rocke_b_select(b, wrap_k, c1, c0);
            rocke_value_t* nx = rocke_b_add(b, xd, cx);
            rocke_value_t* wrap_x = rocke_b_cmp_ge(b, nx, c_xs);
            nx = rocke_b_select(b, wrap_x, c0, nx);
            rocke_value_t* cy = rocke_b_select(b, wrap_x, c1, c0);
            rocke_value_t* ny = rocke_b_add(b, yd, cy);
            uk_new[num_accs + 0] = nk;
            uk_new[num_accs + 1] = nx;
            uk_new[num_accs + 2] = ny;
            rocke_b_scf_yield(b, uk_new, num_accs + 3);
        }
        rocke_b_region_leave(b);

        rocke_value_t* fb_lo = rocke_b_select(b, uk_ok, k_hi, k_lo);
        rocke_iter_arg_t fb_args[ROCKE_CONV_MAX_ACCS];
        char fb_name_bufs[ROCKE_CONV_MAX_ACCS][40];
        for(int i = 0; i < num_accs; i++)
        {
            snprintf(fb_name_bufs[i], sizeof(fb_name_bufs[0]), "fb_%s", iter_args[i].name);
            fb_args[i].name = fb_name_bufs[i];
            fb_args[i].init = for_uk.op->results[i];
        }
        rocke_for_t for_fb
            = rocke_b_scf_for_iter(b, fb_lo, k_hi, c_block_k, fb_args, num_accs, "k1", false, true);
        rocke_b_region_enter(b, for_fb.body);
        emit_plain_body(for_fb.iv, for_fb.iter_vars);
        rocke_b_region_leave(b);

        for(int i = 0; i < for_fb.op->num_results; i++)
            final_accs[i] = for_fb.op->results[i];
        num_final = for_fb.op->num_results;
    }
    else
    {
        // ---- K loop (simple scf.for_iter) ----
        rocke_for_t for_op = rocke_b_scf_for_iter(
            b, k_lo, k_hi, c_block_k, iter_args, num_accs, "k0", false, true);
        rocke_b_region_enter(b, for_op.body);
        emit_plain_body(for_op.iv, for_op.iter_vars);
        rocke_b_region_leave(b);

        for(int i = 0; i < for_op.op->num_results; i++)
            final_accs[i] = for_op.op->results[i];
        num_final = for_op.op->num_results;
    }

    // ---- accumulator epilogue ----
    rocke_value_t* epi_accs[ROCKE_CONV_MAX_ACCS];
    rocke_conv_apply_accumulator_epilogue(b, &spec->acc_epilogue, final_accs, num_final, epi_accs);

    // ---- epilogue dispatch ----
    dispatch_dgrad_epilogue(epi_accs, num_final);

    return b->kernel;
}

// ===========================================================================
// Public build entry: rocke_build_implicit_gemm_conv_dgrad
// (Python build_implicit_gemm_conv_dgrad, lines 833-1314)
// ===========================================================================

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad(rocke_ir_builder_t* b,
                                                         const rocke_dgrad_conv_spec_t* spec,
                                                         const char* arch)
{
    char reason[256];

    if(!b || !spec)
        return NULL;
    if(!arch)
        arch = "gfx950";

    // Name the kernel
    if(b->kernel)
    {
        char name[256];
        if(rocke_dgrad_conv_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        b->kernel->name = rocke_arena_strdup(&b->arena, name);
    }

    // Validate
    if(!rocke_dgrad_conv_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid dgrad spec for %s: %s", arch, reason);
        return NULL;
    }

    return _build_tilde_dgrad(b, spec, arch);
}

// ===========================================================================
// Convenience: init builder then build
// ===========================================================================

rocke_kernel_def_t* rocke_build_implicit_gemm_conv_dgrad_new(rocke_ir_builder_t* b,
                                                             const rocke_dgrad_conv_spec_t* spec,
                                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(!b || !spec)
            return NULL;
        if(rocke_dgrad_conv_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_implicit_gemm_conv_dgrad(b, spec, arch);
    });
}

// ===========================================================================
// Convenience: build + lower to LLVM .ll
// ===========================================================================

static void _dgrad_set_err(char* err, size_t err_cap, const char* msg)
{
    if(!err || err_cap == 0)
        return;
    if(!msg)
        msg = "";
    size_t n = strlen(msg);
    if(n >= err_cap)
        n = err_cap - 1;
    memcpy(err, msg, n);
    err[n] = '\0';
}

rocke_status_t rocke_dgrad_conv_implicit_gemm_lower_to_llvm(const rocke_dgrad_conv_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll)
        *out_ll = NULL;
    if(!spec || !out_ll)
    {
        _dgrad_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(!arch)
        arch = "gfx950";

    kernel = rocke_build_implicit_gemm_conv_dgrad_new(&b, spec, arch);
    if(!kernel)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        _dgrad_set_err(err, err_cap, (m && m[0]) ? m : "build dgrad failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
