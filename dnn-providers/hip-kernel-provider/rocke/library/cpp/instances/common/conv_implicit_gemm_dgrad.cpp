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
#include "rocke/instance_conv_implicit_gemm_dgrad.h"
#include "rocke/instance_conv_implicit_gemm_internal.h" /* rocke_conv_build_ctx_t, phase fns */
#include "rocke/instance_conv_m_band.h"
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
    s.mma_dtype = NULL;
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
    s.cshuffle_m_chunks = 1;
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
    const int mch = s->cshuffle_m_chunks > 1 ? s->cshuffle_m_chunks : 1;
    long c_bytes = cshuffle ? (long)(s->tile_m / mch) * s->tile_n * c_dtype_bytes : 0;
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
    /* xcd_remap overrides the arch's XCD count and also serves groups == 1. */
    const int arch_xcds = (arch && strcmp(arch, ROCKE_DGRAD_XCD_TILE_ORDER_ARCH) == 0)
                              ? ROCKE_DGRAD_XCD_TILE_ORDER_XCDS
                              : 0;
    const int num_xcds = s->xcd_remap ? s->xcd_remap : arch_xcds;
    if(!num_xcds || !s->xcd_tile_order)
        return 0;
    if((s->problem.groups <= 1 && !s->xcd_remap) || s->split_k != 1)
        return 0;
    if(s->pp_streamk || s->pp_boxes)
        return 0;
    if(!_dgrad_folds_sub_gemm_record(s))
        return 0;
    return num_xcds;
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
     * async, kouter, spk, ukd, kp, tapk, noxcd, halo, hprio, hkp, hwo, tf32. C previously emitted spk before async, which only
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
    /* Branch knob tags, in the Python dict order. */
    if(s->tapk_prefetch > 0)
        ROCKE_DGRAD_NAME_APPEND("_tpf%d", s->tapk_prefetch);
    if(s->tapk_xhalo)
        ROCKE_DGRAD_NAME_APPEND("_xh");
    if(s->b_kpack > 0)
        ROCKE_DGRAD_NAME_APPEND("_bkp%d", s->b_kpack);
    if(s->tapk_ilv > 0)
        ROCKE_DGRAD_NAME_APPEND("_ilv%d", s->tapk_ilv);
    if(s->xhalo_exact)
        ROCKE_DGRAD_NAME_APPEND("_xhx");
    if(s->xcd_remap > 0)
        ROCKE_DGRAD_NAME_APPEND("_xcd%d", s->xcd_remap);
    if(s->m_band > 0)
        ROCKE_DGRAD_NAME_APPEND("_mb%d", s->m_band);
    if(s->cshuffle_m_chunks > 1)
        ROCKE_DGRAD_NAME_APPEND("_mch%d", s->cshuffle_m_chunks);
    if(s->tapk_lds_ahead > 0)
        ROCKE_DGRAD_NAME_APPEND("_lda%d", s->tapk_lds_ahead);
    if(s->tapk_lds_barrier)
        ROCKE_DGRAD_NAME_APPEND("_lsb");
    if(s->tapk_setprio > 0)
        ROCKE_DGRAD_NAME_APPEND("_sp%d", s->tapk_setprio);
    if(s->lds_xor)
        ROCKE_DGRAD_NAME_APPEND("_lxor");
    if(s->xh_b_wide)
        ROCKE_DGRAD_NAME_APPEND("_bw");
    if(s->xh_b_kpad > 0)
        ROCKE_DGRAD_NAME_APPEND("_bpad%d", s->xh_b_kpad);
    if(s->tapk_pingpong > 0)
        ROCKE_DGRAD_NAME_APPEND("_bpp%d", s->tapk_pingpong);
    if(s->tapk_pp_sched > 0)
        ROCKE_DGRAD_NAME_APPEND("_pps%d", s->tapk_pp_sched);
    if(s->tapk_pp_epi > 0)
        ROCKE_DGRAD_NAME_APPEND("_ppe%d", s->tapk_pp_epi);
    if(s->split_atomic_pairs)
        ROCKE_DGRAD_NAME_APPEND("_sapl");
    if(s->tapk_pp_persist > 0)
        ROCKE_DGRAD_NAME_APPEND("_ppp%d", s->tapk_pp_persist);
    if(s->tapk_pp_xhalo > 0)
        ROCKE_DGRAD_NAME_APPEND("_ppx%d", s->tapk_pp_xhalo);
    if(s->tapk_ztrim > 0)
        ROCKE_DGRAD_NAME_APPEND("_zt%d", s->tapk_ztrim);
    if(s->tapk_pp_bslab > 0)
        ROCKE_DGRAD_NAME_APPEND("_pbs%d", s->tapk_pp_bslab);
    if(s->xh_a_k4)
        ROCKE_DGRAD_NAME_APPEND("_ak4");
    if(s->xh_order > 0)
        ROCKE_DGRAD_NAME_APPEND("_xo%d", s->xh_order);
    if(s->xh_defer > 0)
        ROCKE_DGRAD_NAME_APPEND("_xdf%d", s->xh_defer);
    if(s->xh_fast_a > 0)
        ROCKE_DGRAD_NAME_APPEND("_xfa%d", s->xh_fast_a);
    if(s->xh_epi > 0)
        ROCKE_DGRAD_NAME_APPEND("_xep%d", s->xh_epi);
    if(s->xh_persist > 0)
        ROCKE_DGRAD_NAME_APPEND("_xhp%d", s->xh_persist);
    if(s->xh_stride_m)
        ROCKE_DGRAD_NAME_APPEND("_xsm");
    if(s->xh_dlds > 0)
        ROCKE_DGRAD_NAME_APPEND("_xdl%d", s->xh_dlds);
    if(s->xh_epi_lin)
        ROCKE_DGRAD_NAME_APPEND("_xel");
    if(s->epi_lds_barriers && !s->pp_boxes)
        ROCKE_DGRAD_NAME_APPEND("_elb");
    if(s->xh_tap_rsrc)
        ROCKE_DGRAD_NAME_APPEND("_xtr");
    if(s->xh_yring > 0)
        ROCKE_DGRAD_NAME_APPEND("_xyr%d", s->xh_yring);
    if(s->no_agpr)
        ROCKE_DGRAD_NAME_APPEND("_nag");
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
    if(s->pp_streamk > 0)
        ROCKE_DGRAD_NAME_APPEND("_psk%d", s->pp_streamk);
    if(s->pp_sk_mode > 0)
        ROCKE_DGRAD_NAME_APPEND("_pskm%d", s->pp_sk_mode);
    if(s->pp_zskip)
        ROCKE_DGRAD_NAME_APPEND("_pzs");
    if(s->pp_boxes > 0)
        ROCKE_DGRAD_NAME_APPEND("_pbx");
    if(s->pp_boxes == 7 || s->pp_boxes == 77)
        ROCKE_DGRAD_NAME_APPEND("_pbl");
    if(s->pp_box_order > 0)
        ROCKE_DGRAD_NAME_APPEND("_pbo%d", s->pp_box_order);
    if(s->pp_box_det > 0)
        ROCKE_DGRAD_NAME_APPEND("_pbd%d", s->pp_box_det);
    /* The name carries no dtype, and on gfx950 a TF32 atom shares its shape
     * with the bf16 one, so the cache would otherwise collide an fp32-storage
     * TF32 kernel with a 16-bit one (Python flag "tf32", the last flag). */
    if(rocke_conv_data_is_tf32(s->mma_dtype))
        ROCKE_DGRAD_NAME_APPEND("_tf32");
    /* Python appends "_elb" after the joined name when pp_boxes is set. */
    if(s->epi_lds_barriers && s->pp_boxes)
        ROCKE_DGRAD_NAME_APPEND("_elb");
#undef ROCKE_DGRAD_NAME_APPEND
    return ROCKE_OK;
}

/* Python _mma_dtype_conflict(spec, arch=None): the shared TF32 gate
 * (rocke_conv_tf32_conv_spec_reason) for dgrad. validate() calls it arch-free
 * (arch NULL); is_valid_dgrad_spec with the arch. Returns true and writes the
 * reason when the spec cannot build. */
static bool _dgrad_mma_dtype_conflict(const rocke_dgrad_conv_spec_t* s,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    rocke_conv_tf32_spec_view_t v;
    memset(&v, 0, sizeof(v));
    v.dtype_a = s->dtype_a;
    v.dtype_b = s->dtype_b;
    v.dtype_d = s->dtype_d;
    v.mma_dtype = s->mma_dtype;
    v.wave_size = s->wave_size;
    v.tile_m = s->tile_m;
    v.tile_n = s->tile_n;
    v.warp_m = s->warp_m;
    v.warp_n = s->warp_n;
    v.warp_tile_m = s->warp_tile_m;
    v.warp_tile_n = s->warp_tile_n;
    v.warp_tile_k = s->warp_tile_k;
    v.lds_k_outer = s->lds_k_outer;
    v.async_dma = s->async_dma;
    /* effective_lds_layout() (with its layout.validate()): the row stride, or
     * the ValueError text _tf32_mfma_reason returns. */
    rocke_conv_lds_layout_t layout = _dgrad_effective_lds_layout(s);
    char layout_err[256];
    if(rocke_conv_lds_layout_validate(&layout, layout_err, sizeof(layout_err)))
        v.lds_row_stride = layout.row_stride;
    else
        v.lds_layout_error = layout_err;
    return rocke_conv_tf32_conv_spec_reason(&v, arch, false, reason, reason_cap);
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

/* Defined with the epilogue helpers below. */
static int _dgrad_split_k_cshuffle_store_vec(const rocke_dgrad_conv_spec_t* spec);

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
        /* Python _3d_conflict: the folded stride-1 record only. */
        if(p->sD != 1 || p->sH != 1 || p->sW != 1)
        {
            snprintf(reason,
                     reason_cap,
                     "3-D dgrad needs stride 1 along depth, height and width (the folded "
                     "stride-1 record); got stride (%d, %d, %d)",
                     p->sD,
                     p->sH,
                     p->sW);
            return false;
        }
        if(s->pipeline && strcmp(s->pipeline, "wavelet") == 0)
        {
            snprintf(reason, reason_cap, "3-D dgrad is not supported with pipeline='wavelet'");
            return false;
        }
        if(s->uniform_k_decode)
        {
            snprintf(
                reason, reason_cap, "3-D dgrad does not implement the uniform_k_decode K loop");
            return false;
        }
        if(s->dy_halo)
        {
            snprintf(reason, reason_cap, "3-D dgrad does not implement the dY halo K loop");
            return false;
        }
        if(s->patch)
        {
            snprintf(reason, reason_cap, "3-D dgrad does not implement patch");
            return false;
        }
    }
    /* Branch knobs the C++ engine does not implement yet. */
    const bool xh_ok = s->tapk_xhalo && s->tap_outer_k;
    if(s->xh_yring
       && (!xh_ok || s->tapk_prefetch != 3 || !s->xhalo_exact || !s->xh_a_k4 || !s->xh_b_wide
           || s->tapk_ztrim != 1 || s->split_k > 1 || s->xh_persist))
    {
        snprintf(
            reason,
            reason_cap,
            "xh_yring (dgrad) needs the x-halo prefetch loop (tpf 3) with xhalo_exact, xh_a_k4, "
            "xh_b_wide, tapk_ztrim 1 (zt0 failed verify on s2), no split-K / xh_persist, and a "
            "ring of P = k * Wi >= tile_m + X - 1 rows within the A slab");
        return false;
    }
    if((s->tapk_ilv && !xh_ok) || (s->xhalo_exact && !xh_ok) || (s->m_band && !xh_ok)
       || (s->xh_stride_m && !xh_ok) || s->xh_dlds || (s->xh_epi_lin && !xh_ok)
       || (s->xh_tap_rsrc && !xh_ok) || (s->xh_yring && !xh_ok) || (s->tapk_lds_ahead && !xh_ok)
       || (s->tapk_lds_barrier && !xh_ok) || (s->tapk_setprio && !xh_ok)
       || (s->lds_xor && !s->tapk_pingpong && !xh_ok) || (s->xh_b_wide && !xh_ok)
       || (s->xh_b_kpad && !xh_ok)
       || (s->tapk_pp_epi && !(s->tapk_pp_epi == 2 && s->pp_boxes && s->split_k <= 1))
       || s->split_atomic_pairs || s->tapk_pp_persist || s->tapk_pp_xhalo
       || (s->tapk_ztrim && !xh_ok) || (s->tapk_pp_bslab && !s->tapk_pingpong)
       || (s->xh_a_k4 && !xh_ok) || (s->xh_order && !xh_ok) || (s->xh_defer && !xh_ok)
       || (s->xh_fast_a && !xh_ok) || (s->xh_epi && !xh_ok) || s->xh_persist || s->pp_streamk
       || s->pp_sk_mode || s->pp_zskip
       || ((s->pp_boxes || s->pp_box_cus || s->pp_box_order) && !s->tapk_pingpong)
       || (s->pp_box_det && !(s->pp_box_det == 1 && s->pp_boxes && s->split_k > 1))
       || (s->epi_lds_barriers && !s->pp_boxes && !xh_ok) || (s->no_agpr && !xh_ok)
       || (s->cshuffle_m_chunks > 1 && !s->pp_boxes && !xh_ok))
    {
        snprintf(reason,
                 reason_cap,
                 "dgrad x-halo / ping-pong / box knobs are Python-only (no C++ twin yet)");
        return false;
    }
    if(_dgrad_mma_dtype_conflict(s, arch, reason, reason_cap))
    {
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
        /* The split-K cshuffle epilogue atomic-adds 16-bit dX in packed
         * pairs, so its store width must be even (Python
         * _dgrad_split_k_cshuffle_store_vec). */
        if(packed && s->epilogue && strcmp(s->epilogue, "cshuffle") == 0)
        {
            const int sv = _dgrad_split_k_cshuffle_store_vec(s);
            if(sv % 2 != 0)
            {
                snprintf(reason,
                         reason_cap,
                         "split_k > 1 with the cshuffle epilogue and dtype_d='%s' needs an "
                         "even dX store width (packed <2 x dtype> atomic pairs); got %d",
                         dd,
                         sv);
                return false;
            }
        }
    }

    /* MMA shape availability (Python: target.mma.has_shape), asked for the
     * MMA dtypes: a TF32 spec stores fp32 and issues its (tf32, tf32, fp32)
     * atom. */
    const char* mma_a = rocke_conv_data_mma_operand(s->dtype_a, s->mma_dtype);
    const char* mma_b = rocke_conv_data_mma_operand(s->dtype_b, s->mma_dtype);
    if(!rocke_mma_catalog_has_shape(
           &tgt->mma, family, mma_a, mma_b, "fp32", s->warp_tile_m, s->warp_tile_n, s->warp_tile_k))
    {
        /* f"unsupported {spec.data.mma_a} warp_tile {atom} on {arch}", atom a
         * (m, n, k) tuple. */
        snprintf(reason,
                 reason_cap,
                 "unsupported %s warp_tile (%d, %d, %d) on %s",
                 mma_a,
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
        /* the x-halo prefetch / ping-pong loops slice whole tiles */
        if(s->split_k != 1
           && !((s->tapk_xhalo || s->tapk_pingpong)
                && (s->tapk_prefetch == 1 || s->tapk_prefetch == 3) && s->split_k > 1))
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
                                                                          mma_a,
                                                                          mma_b,
                                                                          "fp32",
                                                                          s->warp_tile_m,
                                                                          s->warp_tile_n,
                                                                          s->warp_tile_k,
                                                                          nullptr)
                                          : NULL;
            const int tr_lanes = s->wave_size == 32 ? 8 : 4;
            if(!kop || kop->srcs[1].frag_len % tr_lanes != 0)
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
                         kop ? kop->srcs[1].frag_len : 0,
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
    if(ctx->params_is_3d)
    {
        /* N[D]HWC: m -> (n, di, hi, wi) divides by Wi, then Hi, then Di. */
        const char* cn3[5] = {"n", "di", "hi", "wi", "c"};
        rocke_value_t* st3[5] = {ctx->p_dX_stride_n,
                                 ctx->p_dX_stride_di,
                                 ctx->p_dX_stride_hi,
                                 ctx->p_dX_stride_wi,
                                 NULL};
        st3[4] = rocke_b_const_i32(b, 1);
        rocke_dynamic_tensor_descriptor_t* d3
            = rocke_tensor_descriptor_naive_dynamic(b, "dX_ndhwc", cn3, 5, st3);
        if(!d3)
            return NULL;
        const char* into3[4] = {"n", "di", "hi", "wi"};
        rocke_magic_triple_t t3[3];
        memset(t3, 0, sizeof(t3));
        t3[0].mult = ctx->p_magic_m_Di_mult;
        t3[0].shift = ctx->p_magic_m_Di_shift;
        t3[0].dim = ctx->p_Di;
        t3[1].mult = ctx->p_magic_m_Hi_mult;
        t3[1].shift = ctx->p_magic_m_Hi_shift;
        t3[1].dim = ctx->p_Hi;
        t3[2].mult = ctx->p_magic_m_Wi_mult;
        t3[2].shift = ctx->p_magic_m_Wi_shift;
        t3[2].dim = ctx->p_Wi;
        const rocke_transform_t* xf[1];
        xf[0] = rocke_unmerge_magic_dynamic(b, "m", into3, 4, t3);
        if(!xf[0])
            return NULL;
        rocke_tensor_descriptor_t* ch = rocke_tensor_descriptor_transform(b, &d3->base, xf, 1);
        if(!ch)
            return NULL;
        d3->base = *ch;
        return d3;
    }
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
    const rocke_mmaop_t* op
        = rocke_archtarget_op_for_shape(target,
                                        _dgrad_mma_family(arch),
                                        rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                                        rocke_conv_data_mma_operand(spec->dtype_b, spec->mma_dtype),
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

/* The dX width the split-K cshuffle epilogue atomic-adds with -- mirrors Python
 * _dgrad_split_k_cshuffle_store_vec (rocke_cshuffle_epilogue_from_grid's pick
 * for _dgrad_store_vec). */
static int _dgrad_split_k_cshuffle_store_vec(const rocke_dgrad_conv_spec_t* spec)
{
    const bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    return rocke_cshuffle_pick_store_vec(spec->tile_m,
                                         spec->tile_n,
                                         spec->warp_m * spec->warp_n * spec->wave_size,
                                         _dgrad_store_vec(spec),
                                         is_fp32_d);
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
    epi.atom = rocke_mfma_atom(rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                               spec->warp_tile_m,
                               spec->warp_tile_n,
                               spec->warp_tile_k);
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
        = rocke_mfma_atom(rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                          spec->warp_tile_m,
                          spec->warp_tile_n,
                          spec->warp_tile_k);
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

            const rocke_arch_layout_map_t* c_map = rocke_mma_op_dst_layout(op, b);
            for(int i = 0; i < op->dst.frag_len; i++)
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
    /* 3-D: n -> (n, di) by the kernarg magic pair; dd = pD - z (per tap). */
    bool is_3d;
    dgrad_magic_div_t di_div;
    rocke_value_t* p_Do;
    rocke_value_t* dd;
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
    if(ctx->is_3d)
    {
        /* NDHWK: n -> (n, di); the dY row is n*Do + di, in range when
         * 0 <= di + dd < Do. */
        rocke_value_t *nq, *di;
        _magic_divmod(b_, n_val, &ctx->di_div, &nq, &di);
        rocke_value_t* dov = rocke_b_add(b_, di, ctx->dd);
        rocke_value_t* d_ge = rocke_b_cmp_ge(b_, dov, ctx->c0);
        rocke_value_t* d_lt = rocke_b_cmp_lt(b_, dov, ctx->p_Do);
        rocke_value_t* d_ok = rocke_b_land(b_, d_ge, d_lt);
        valid = rocke_b_land(b_, valid, d_ok);
        rocke_value_t* nd = rocke_b_mul(b_, nq, ctx->p_Do);
        n_val = rocke_b_add(b_, nd, di);
    }
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

    const rocke_arch_layout_map_t* c_map = rocke_mma_op_dst_layout(op, b);

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

            for(int i = 0; i < op->dst.frag_len; i++)
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
    /* pp_boxes: m_map (the pixel-list row table, pp_boxes 7/77) and n_map
     * (_bx_nval: the box-local n*ds + (d - db) to the dX row n*Di + d). */
    rocke_value_t* rm_base;
    rocke_value_t* rm_buf;
    rocke_value_t* bx_ds;
    rocke_value_t* bx_db;
    rocke_value_t* p_Di;
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
    ctx.rm_base = NULL;
    ctx.rm_buf = NULL;
    ctx.bx_ds = NULL;
    ctx.bx_db = NULL;
    ctx.p_Di = NULL;
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
    if(ctx->rm_base != NULL)
    {
        /* m_map (pp_boxes 7/77): box row -> the row the map decodes. */
        rocke_value_t* ra = rocke_b_add(b, ctx->rm_base, m_global);
        m_global = rocke_b_global_load_i32(b, ctx->rm_buf, ra, 4);
    }

    // c0 created locally to match Python closure order (b.const_i32(0) first in fn).
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    // Decompose m_global -> (n_val, htl, wtl)
    rocke_value_t *n_val, *m_rem, *htl, *wtl;
    _magic_divmod(b, m_global, &ctx->hw_tilde, &n_val, &m_rem);
    _magic_divmod(b, m_rem, &ctx->w_tilde_slice, &htl, &wtl);
    if(ctx->bx_ds != NULL)
    {
        /* n_map = _bx_nval (3-D boxes) */
        rocke_value_t* img = rocke_b_div(b, n_val, ctx->bx_ds);
        rocke_value_t* im = rocke_b_mul(b, img, ctx->bx_ds);
        rocke_value_t* r_ = rocke_b_sub(b, n_val, im);
        rocke_value_t* d = rocke_b_add(b, r_, ctx->bx_db);
        rocke_value_t* id = rocke_b_mul(b, img, ctx->p_Di);
        n_val = rocke_b_add(b, id, d);
    }

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
/* pp_boxes: the cshuffle store of a box of dX rows (Python
 * _emit_dgrad_box_cshuffle_epilogue): the tilde address map with the record's
 * (h, w) offsets, n_map / m_map, war_barriers 1 and cshuffle_m_chunks. */
static void _emit_dgrad_box_cshuffle_epilogue(rocke_ir_builder_t* b,
                                              const rocke_dgrad_conv_spec_t* spec,
                                              const rocke_mfma_atom_t* atom,
                                              const rocke_warp_grid_t* grid,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* dx_rsrc,
                                              rocke_value_t* bounds_m,
                                              rocke_value_t* bounds_n,
                                              const tilde_dx_ctx_t* tctx)
{
    tilde_dx_ctx_t ctx = *tctx;
    rocke_cshuffle_epilogue_t epi
        = rocke_cshuffle_epilogue_from_grid(atom, grid, _dgrad_store_vec(spec));
    epi.out_dtype = spec->dtype_d;
    epi.war_barriers = 1;
    epi.m_chunks = spec->cshuffle_m_chunks;
    epi.lds_barriers = spec->epi_lds_barriers; /* the epilogue's barriers LDS-only */
    rocke_cshuffle_epilogue_store(
        b, &epi, accs, num_accs, _tilde_dx_addr_fn, &ctx, dx_rsrc, bounds_m, bounds_n);
}

/* _xh_f32x4_to_bf16 / _xh_f32x4_to_bf16_perm: four f32 -> <4 x bf16> (RNE,
 * no NaN fix-up); the perm form packs each pair with one v_perm_b32. */
static rocke_value_t*
    _xh_f32x4_to_bf16(rocke_ir_builder_t* b, rocke_value_t* const* vals, bool perm)
{
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* bias = rocke_b_const_i32(b, 0x7FFF);
    rocke_value_t* sel = perm ? rocke_b_const_i32(b, 0x07060302) : NULL;
    rocke_value_t* rs[4];
    for(int i = 0; i < 4; ++i)
    {
        rocke_value_t* xi = rocke_b_bitcast(b, vals[i], rocke_i32());
        rocke_value_t* xs = rocke_b_lshr(b, xi, c16);
        rocke_value_t* lsb = rocke_b_land(b, xs, c1);
        rocke_value_t* xb = rocke_b_add(b, xi, bias);
        rocke_value_t* xl = rocke_b_add(b, xb, lsb);
        rs[i] = perm ? xl : rocke_b_lshr(b, xl, c16);
    }
    rocke_value_t* words[2];
    for(int j = 0; j < 2; ++j)
    {
        if(perm)
            words[j] = rocke_b_perm_b32(b, rs[2 * j + 1], rs[2 * j], sel);
        else
        {
            rocke_value_t* sh = rocke_b_shl(b, rs[2 * j + 1], c16);
            words[j] = rocke_b_lor(b, rs[2 * j], sh);
        }
    }
    rocke_value_t* pk = rocke_b_vec_pack(b, words, 2, rocke_i32());
    return rocke_b_vec_bitcast(b, pk, rocke_vector_type(b, rocke_bf16(), 4));
}

/* _emit_xh_cshuffle_epilogue, xh_epi 1: the cshuffle store of transposed
 * accumulators in cshuffle_m_chunks row slabs (xh_stride_m rows, the
 * xh_epi_lin row-major fast address). */
static void _emit_xh_cshuffle_epilogue(rocke_ir_builder_t* b,
                                       const rocke_dgrad_conv_spec_t* spec,
                                       const rocke_conv_build_ctx_t* params,
                                       const rocke_mfma_atom_t* atom,
                                       rocke_value_t* const* accs,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* dx_rsrc,
                                       rocke_value_t* c_group_base,
                                       bool lds_sync)
{
    rocke_dynamic_tensor_descriptor_t* dX_desc = _dgrad_make_dx_descriptor_dynamic(b, params);
    if(!dX_desc)
        return;
    struct dgrad_dx_addr_ctx addr_ctx;
    addr_ctx.desc = &dX_desc->base;
    addr_ctx.c_group_base = c_group_base;
    rocke_value_t* bound_m = params->p_dg_M;
    rocke_value_t* bound_n = params->p_dg_N;
    const int mc = spec->cshuffle_m_chunks > 1 ? spec->cshuffle_m_chunks : 1;
    const int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    const int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    const int cm = mfmas_m / mc;
    const int am = spec->warp_tile_m, an = spec->warp_tile_n;
    const int rows_c = spec->tile_m / mc;
    const int tile_n = spec->tile_n;
    const int threads = rocke_dgrad_conv_spec_block_size(spec);
    const int sv = 8;
    int shp[2] = {rows_c, tile_n};
    rocke_value_t* c_smem = rocke_b_smem_alloc(b, rocke_bf16(), shp, 2, "C_smem");
    rocke_value_t* c3 = rocke_b_const_i32(b, 3);
    auto swz = [&](rocke_value_t* row, rocke_value_t* col) -> rocke_value_t* {
        rocke_value_t* r7 = rocke_b_land(b, row, rocke_b_const_i32(b, 7));
        rocke_value_t* sh = rocke_b_shl(b, r7, c3);
        return rocke_b_xor(b, col, sh);
    };
    auto sync = [&]() {
        if(lds_sync)
            rocke_b_sync_lds_only(b);
        else
            rocke_b_sync(b);
    };
    rocke_value_t* warp_m_off_c = rocke_b_mul(b, grid->warp_m_idx, rocke_b_const_i32(b, cm * am));
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    rocke_value_t* c_threads = rocke_b_const_i32(b, threads);
    rocke_value_t* c_tnv = rocke_b_const_i32(b, tile_n / sv);
    rocke_value_t* c_slab = rocke_b_const_i32(b, cm * am);
    rocke_value_t* c_wrows = rocke_b_const_i32(b, mfmas_m * am);
    rocke_value_t* c_eb = rocke_b_const_i32(b, 2);
    rocke_value_t* oob = rocke_b_const_i32(b, 2147483647);
    const int vecs_per_thread = (rows_c * tile_n / sv) / threads;
    rocke_value_t* r_lane = rocke_b_land(b, grid->lane, rocke_b_const_i32(b, am - 1));
    rocke_value_t* l4 = rocke_b_lshr(b, grid->lane, rocke_b_const_i32(b, am == 16 ? 4 : 5));
    rocke_value_t* n_blk = rocke_b_shl(b, l4, rocke_b_const_i32(b, 2));
    const int n_groups = atom->c_per_lane / 4;
    rocke_value_t* lin_ok = NULL;
    rocke_value_t* sw = params->p_dX_stride_wi;
    if(spec->xh_epi_lin)
    {
        rocke_value_t* sh = params->p_dX_stride_hi;
        rocke_value_t* swi = rocke_b_mul(b, sw, params->p_Wi);
        lin_ok = rocke_b_cmp_eq(b, sh, swi);
        if(params->params_is_3d)
        {
            rocke_value_t* sd = params->p_dX_stride_di;
            rocke_value_t* shh = rocke_b_mul(b, sh, params->p_Hi);
            rocke_value_t* e1 = rocke_b_cmp_eq(b, sd, shh);
            lin_ok = rocke_b_land(b, lin_ok, e1);
            rocke_value_t* sdd = rocke_b_mul(b, sd, params->p_Di);
            rocke_value_t* e2 = rocke_b_cmp_eq(b, params->p_dX_stride_n, sdd);
            lin_ok = rocke_b_land(b, lin_ok, e2);
        }
        else
        {
            rocke_value_t* shh = rocke_b_mul(b, sh, params->p_Hi);
            rocke_value_t* e2 = rocke_b_cmp_eq(b, params->p_dX_stride_n, shh);
            lin_ok = rocke_b_land(b, lin_ok, e2);
        }
    }
    for(int c = 0; c < mc; ++c)
    {
        sync();
        for(int mi_l = 0; mi_l < cm; ++mi_l)
        {
            const int mi = c * cm + mi_l;
            rocke_value_t* rm = rocke_b_add(b, rocke_b_const_i32(b, mi_l * am), r_lane);
            rocke_value_t* row = rocke_b_add(b, warp_m_off_c, rm);
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* acc = accs[mi * mfmas_n + ni];
                for(int g = 0; g < n_groups; ++g)
                {
                    rocke_value_t* xs[4];
                    for(int j = 0; j < 4; ++j)
                        xs[j] = rocke_b_vec_extract(b, acc, 4 * g + j);
                    rocke_value_t* v = _xh_f32x4_to_bf16(b, xs, spec->xh_epi_lin);
                    rocke_value_t* cn
                        = rocke_b_add(b, rocke_b_const_i32(b, ni * an + 8 * g), n_blk);
                    rocke_value_t* col = rocke_b_add(b, warp_n_off, cn);
                    rocke_value_t* sc = swz(row, col);
                    rocke_value_t* idx[2] = {row, sc};
                    rocke_b_smem_store_vN(b, c_smem, idx, 2, v, 4);
                }
            }
        }
        sync();
        auto store_phase = [&](bool lin) {
            for(int e = 0; e < vecs_per_thread; ++e)
            {
                rocke_value_t* em = rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads);
                rocke_value_t* vec_idx = rocke_b_add(b, em, grid->tid);
                rocke_value_t* row = rocke_b_div(b, vec_idx, c_tnv);
                rocke_value_t* rt = rocke_b_mul(b, row, c_tnv);
                rocke_value_t* rr = rocke_b_sub(b, vec_idx, rt);
                rocke_value_t* col = rocke_b_mul(b, rr, rocke_b_const_i32(b, sv));
                rocke_value_t* w = rocke_b_div(b, row, c_slab);
                rocke_value_t* ws = rocke_b_mul(b, w, c_slab);
                rocke_value_t* r_in = rocke_b_sub(b, row, ws);
                rocke_value_t* g_row;
                if(spec->xh_stride_m)
                {
                    int bl = 0;
                    for(int t = am; t; t >>= 1)
                        bl++;
                    rocke_value_t* mi_l_v = rocke_b_lshr(b, r_in, rocke_b_const_i32(b, bl - 1));
                    rocke_value_t* r_v = rocke_b_land(b, r_in, rocke_b_const_i32(b, am - 1));
                    rocke_value_t* mi_v = (c * cm != 0)
                                              ? rocke_b_add(b, mi_l_v, rocke_b_const_i32(b, c * cm))
                                              : mi_l_v;
                    rocke_value_t* wr = rocke_b_mul(b, w, c_wrows);
                    rocke_value_t* rv = rocke_b_mul(b, r_v, rocke_b_const_i32(b, mfmas_m));
                    rocke_value_t* rvm = rocke_b_add(b, rv, mi_v);
                    g_row = rocke_b_add(b, wr, rvm);
                }
                else
                {
                    rocke_value_t* wr = rocke_b_mul(b, w, c_wrows);
                    rocke_value_t* wrc = rocke_b_add(b, wr, rocke_b_const_i32(b, c * cm * am));
                    g_row = rocke_b_add(b, wrc, r_in);
                }
                rocke_value_t* m_val = rocke_b_add(b, grid->block_m_off, g_row);
                rocke_value_t* n_val = rocke_b_add(b, grid->block_n_off, col);
                rocke_value_t* ok
                    = rocke_direct_epilogue_bounds_check(b, m_val, n_val, bound_m, bound_n, sv);
                rocke_value_t* valid = NULL;
                rocke_value_t* off_elems;
                if(lin)
                {
                    rocke_value_t* c_ = c_group_base ? rocke_b_add(b, n_val, c_group_base) : n_val;
                    rocke_value_t* mw = rocke_b_mul(b, m_val, sw);
                    off_elems = rocke_b_add(b, mw, c_);
                }
                else
                    off_elems = _dgrad_dx_addr_fn(b, m_val, n_val, &valid, &addr_ctx);
                if(valid != NULL)
                    ok = ok == NULL ? valid : rocke_b_land(b, ok, valid);
                rocke_value_t* off_b = rocke_b_mul(b, off_elems, c_eb);
                rocke_value_t* safe = ok != NULL ? rocke_b_select(b, ok, off_b, oob) : off_b;
                rocke_value_t* sc = swz(row, col);
                rocke_value_t* li[2] = {row, sc};
                rocke_value_t* vbf = rocke_b_smem_load_vN(b, c_smem, li, 2, rocke_bf16(), sv);
                rocke_b_buffer_store_vN_bf16(
                    b, dx_rsrc, safe, rocke_b_const_i32(b, 0), vbf, sv / 2);
            }
        };
        if(lin_ok == NULL)
            store_phase(false);
        else
        {
            rocke_if_else_t ie = rocke_b_scf_if_else(b, lin_ok);
            rocke_b_region_enter(b, ie.then_region);
            store_phase(true);
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ie.else_region);
            store_phase(false);
            rocke_b_region_leave(b);
        }
    }
}

/* _pp_f32_to_bf16 (tapk_pp_epi 2): RNE without the NaN fix-up. */
static rocke_value_t*
    _pp_f32_to_bf16(rocke_ir_builder_t* b, const rocke_dgrad_conv_spec_t* spec, rocke_value_t* acc)
{
    if(spec->tapk_pp_epi != 2)
        return rocke_b_vec_trunc_f32_to_bf16(b, acc);
    const int n = acc->type->count;
    rocke_value_t* bits = rocke_b_vec_bitcast(b, acc, rocke_vector_type(b, rocke_i32(), n));
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* bias = rocke_b_const_i32(b, 0x7FFF);
    rocke_value_t* his[16];
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* x = rocke_b_vec_extract(b, bits, i);
        rocke_value_t* xs = rocke_b_lshr(b, x, c16);
        rocke_value_t* lsb = rocke_b_land(b, xs, c1);
        rocke_value_t* xb = rocke_b_add(b, x, bias);
        rocke_value_t* xl = rocke_b_add(b, xb, lsb);
        his[i] = rocke_b_lshr(b, xl, c16);
    }
    rocke_value_t* words[8];
    for(int j = 0; j < n / 2; ++j)
    {
        rocke_value_t* sh = rocke_b_shl(b, his[2 * j + 1], c16);
        words[j] = rocke_b_lor(b, his[2 * j], sh);
    }
    rocke_value_t* pk = rocke_b_vec_pack(b, words, n / 2, rocke_i32());
    return rocke_b_vec_bitcast(b, pk, rocke_vector_type(b, rocke_bf16(), n));
}

/* _emit_pp_cshuffle_epilogue, tapk_pp_epi 2 (transposed C), with an explicit
 * (addr_fn, bounds) (pp_boxes) and the pre-converted bf16 accumulators. */
static void _emit_pp_cshuffle_epilogue(rocke_ir_builder_t* b,
                                       const rocke_dgrad_conv_spec_t* spec,
                                       const rocke_mfma_atom_t* atom,
                                       const rocke_warp_grid_t* grid,
                                       rocke_value_t* const* accs,
                                       rocke_value_t* const* pre_bf16,
                                       rocke_value_t* dx_rsrc,
                                       rocke_loads_descriptor_fn addr_fn,
                                       void* addr_user,
                                       rocke_value_t* bound_m,
                                       rocke_value_t* bound_n,
                                       bool lds_sync)
{
    const int mc = spec->cshuffle_m_chunks > 1 ? spec->cshuffle_m_chunks : 1;
    const int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    const int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    const int cm = mfmas_m / mc;
    const int am = spec->warp_tile_m, an = spec->warp_tile_n;
    const int rows_c = spec->tile_m / mc;
    const int tile_n = spec->tile_n;
    const int threads = rocke_dgrad_conv_spec_block_size(spec);
    const int sv = 8;
    int shp[2] = {rows_c, tile_n};
    rocke_value_t* c_smem
        = rocke_b_smem_alloc(b, rocke_bf16(), shp, 2, "C_smem"); /* make_lds_view */
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    (void)c1;
    rocke_value_t* c3 = rocke_b_const_i32(b, 3);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    auto swz = [&](rocke_value_t* row, rocke_value_t* col) -> rocke_value_t* {
        rocke_value_t* r7 = rocke_b_land(b, row, rocke_b_const_i32(b, 7));
        rocke_value_t* sh = rocke_b_shl(b, r7, c3);
        return rocke_b_xor(b, col, sh);
    };
    auto sync = [&]() {
        if(lds_sync)
            rocke_b_sync_lds_only(b);
        else
            rocke_b_sync(b);
    };
    rocke_value_t* warp_m_off_c = rocke_b_mul(b, grid->warp_m_idx, rocke_b_const_i32(b, cm * am));
    rocke_value_t* warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    rocke_value_t* c_threads = rocke_b_const_i32(b, threads);
    rocke_value_t* c_tnv = rocke_b_const_i32(b, tile_n / sv);
    rocke_value_t* c_slab = rocke_b_const_i32(b, cm * am);
    rocke_value_t* c_wrows = rocke_b_const_i32(b, mfmas_m * am);
    rocke_value_t* c_eb = rocke_b_const_i32(b, 2);
    rocke_value_t* oob = rocke_b_const_i32(b, 2147483647);
    const int vecs_per_thread = (rows_c * tile_n / sv) / threads;
    for(int c = 0; c < mc; ++c)
    {
        sync();
        for(int mi_l = 0; mi_l < cm; ++mi_l)
        {
            const int mi = c * cm + mi_l;
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* v = pre_bf16 != NULL
                                       ? pre_bf16[mi * mfmas_n + ni]
                                       : _pp_f32_to_bf16(b, spec, accs[mi * mfmas_n + ni]);
                rocke_value_t* r = rocke_b_land(b, grid->lane, rocke_b_const_i32(b, am - 1));
                rocke_value_t* l4 = rocke_b_lshr(b, grid->lane, c4);
                rocke_value_t* cc = rocke_b_shl(b, l4, rocke_b_const_i32(b, 2));
                rocke_value_t* rm = rocke_b_add(b, rocke_b_const_i32(b, mi_l * am), r);
                rocke_value_t* row = rocke_b_add(b, warp_m_off_c, rm);
                rocke_value_t* cn = rocke_b_add(b, rocke_b_const_i32(b, ni * an), cc);
                rocke_value_t* col = rocke_b_add(b, warp_n_off, cn);
                rocke_value_t* sc = swz(row, col);
                rocke_value_t* idx[2] = {row, sc};
                rocke_b_smem_store_vN(b, c_smem, idx, 2, v, atom->c_per_lane);
            }
        }
        sync();
        for(int e = 0; e < vecs_per_thread; ++e)
        {
            rocke_value_t* em = rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads);
            rocke_value_t* vec_idx = rocke_b_add(b, em, grid->tid);
            rocke_value_t* row = rocke_b_div(b, vec_idx, c_tnv);
            rocke_value_t* rt = rocke_b_mul(b, row, c_tnv);
            rocke_value_t* rr = rocke_b_sub(b, vec_idx, rt);
            rocke_value_t* col = rocke_b_mul(b, rr, rocke_b_const_i32(b, sv));
            rocke_value_t* w = rocke_b_div(b, row, c_slab);
            rocke_value_t* ws = rocke_b_mul(b, w, c_slab);
            rocke_value_t* r_in = rocke_b_sub(b, row, ws);
            rocke_value_t* wr = rocke_b_mul(b, w, c_wrows);
            rocke_value_t* wrc = rocke_b_add(b, wr, rocke_b_const_i32(b, c * cm * am));
            rocke_value_t* g_row = rocke_b_add(b, wrc, r_in);
            rocke_value_t* m_val = rocke_b_add(b, grid->block_m_off, g_row);
            rocke_value_t* n_val = rocke_b_add(b, grid->block_n_off, col);
            rocke_value_t* ok
                = rocke_direct_epilogue_bounds_check(b, m_val, n_val, bound_m, bound_n, sv);
            rocke_value_t* valid = NULL;
            rocke_value_t* off_elems = addr_fn(b, m_val, n_val, &valid, addr_user);
            if(valid != NULL)
                ok = ok == NULL ? valid : rocke_b_land(b, ok, valid);
            rocke_value_t* off_b = rocke_b_mul(b, off_elems, c_eb);
            rocke_value_t* safe = ok != NULL ? rocke_b_select(b, ok, off_b, oob) : off_b;
            rocke_value_t* sc = swz(row, col);
            rocke_value_t* li[2] = {row, sc};
            rocke_value_t* vbf = rocke_b_smem_load_vN(b, c_smem, li, 2, rocke_bf16(), sv);
            rocke_b_buffer_store_vN_bf16(b, dx_rsrc, safe, rocke_b_const_i32(b, 0), vbf, sv / 2);
        }
    }
}

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
                                                rocke_value_t* patch_row,
                                                rocke_value_t* atomic_dx_ptr)
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
    /* Split-K (MFMA): atomic-add the staged tile into dX through the same
     * address function (Python atomic_dx_ptr). */
    if(atomic_dx_ptr != NULL)
    {
        rocke_cshuffle_epilogue_atomic_store_addr(b,
                                                  &epi,
                                                  accs,
                                                  num_accs,
                                                  atomic_dx_ptr,
                                                  NULL,
                                                  _tilde_dx_addr_fn,
                                                  &ctx,
                                                  bounds_m,
                                                  bounds_n);
        return;
    }
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

// ---------------------------------------------------------------------------
// KPackedRowLoader (Python conv_implicit_gemm_dgrad.KPackedRowLoader): the
// B (W) tile stored b_kpack k values per LDS write. A staged unit is b_kpack
// consecutive rocke_ctl_staged_t entries (same row / col), so the packed
// store is CoalescedTileLoader.store_lds with row_kpack = b_kpack.
// ---------------------------------------------------------------------------

static bool _kpr_fits(int tile_rows, int tile_cols, int block_size, int load_vec, int kpack)
{
    if(load_vec < 2 || tile_rows % load_vec || tile_cols % kpack)
        return false;
    const int units = (tile_rows / load_vec) * (tile_cols / kpack);
    return units >= block_size && units % block_size == 0;
}

static int _kpr_choose_vec(int tile_rows, int tile_cols, int block_size, int max_vec, int kpack)
{
    for(int v = max_vec; v >= 2; v /= 2)
        if(_kpr_fits(tile_rows, tile_cols, block_size, v, kpack))
            return v;
    return 0;
}

static void _kpr_load_global(rocke_ir_builder_t* b,
                             const rocke_coalesced_tile_loader_t* L,
                             rocke_value_t* tid,
                             rocke_loads_descriptor_fn desc,
                             void* user,
                             rocke_value_t* rsrc,
                             rocke_ctl_staged_t* staged)
{
    const int kp = L->row_kpack;
    const rocke_type_t* et = L->elem_dtype ? L->elem_dtype : rocke_f16();
    const int eb = et == rocke_f32() ? 4 : 2;
    rocke_value_t* c_eb = rocke_b_const_i32(b, eb);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_oob = rocke_b_const_i32(b, 2147483647);
    const int k_groups = L->tile_cols / kp;
    const int units = (L->tile_rows / L->load_vec) * k_groups;
    rocke_value_t* c_kg = rocke_b_const_i32(b, k_groups);
    const int n_u = units / L->block_size;
    rocke_value_t* rows[ROCKE_CTL_MAX_VECS_PER_THREAD];
    rocke_value_t* cols[ROCKE_CTL_MAX_VECS_PER_THREAD];
    const int n_chunks = L->tile_rows / L->load_vec;
    for(int e = 0; e < n_u; ++e)
    {
        rocke_value_t* u = rocke_b_add(b, rocke_b_const_i32(b, e * L->block_size), tid);
        rocke_value_t *nc, *kg;
        if(L->kp_line /* n_fast */)
        {
            kg = rocke_b_div(b, u, rocke_b_const_i32(b, n_chunks));
            rocke_value_t* km = rocke_b_mul(b, kg, rocke_b_const_i32(b, n_chunks));
            nc = rocke_b_sub(b, u, km);
        }
        else
        {
            nc = rocke_b_div(b, u, c_kg);
            rocke_value_t* nk = rocke_b_mul(b, nc, c_kg);
            kg = rocke_b_sub(b, u, nk);
        }
        rows[e] = rocke_b_mul(b, nc, rocke_b_const_i32(b, L->load_vec));
        cols[e] = rocke_b_mul(b, kg, rocke_b_const_i32(b, kp));
    }
    staged->count = 0;
    for(int e = 0; e < n_u; ++e)
        for(int j = 0; j < kp; ++j)
        {
            rocke_value_t* col = j ? rocke_b_add(b, cols[e], rocke_b_const_i32(b, j)) : cols[e];
            rocke_value_t* valid = NULL;
            rocke_value_t* off = desc(b, rows[e], col, &valid, user);
            rocke_value_t* off_b = rocke_b_mul(b, off, c_eb);
            if(valid != NULL)
                off_b = rocke_b_select(b, valid, off_b, c_oob);
            rocke_value_t* v = rocke_b_buffer_load_vN(b, rsrc, off_b, c0, et, L->load_vec);
            staged->vecs[staged->count].row = rows[e];
            staged->vecs[staged->count].col = cols[e];
            staged->vecs[staged->count].v = v;
            staged->count++;
        }
}

/* The B loader of the tap loops: the k-packed one when b_kpack is set. */
static void _dg_b_load_global(rocke_ir_builder_t* b,
                              const rocke_coalesced_tile_loader_t* L,
                              rocke_value_t* tid,
                              rocke_loads_descriptor_fn desc,
                              void* user,
                              rocke_value_t* rsrc,
                              rocke_ctl_staged_t* staged)
{
    if(L->row_kpack > 1)
        _kpr_load_global(b, L, tid, desc, user, rsrc, staged);
    else
        rocke_coalesced_tile_loader_load_global(b, L, tid, desc, user, rsrc, NULL, staged);
}

/* tapk_prefetch: the per-tile uniforms set_tile() computes (Python cap). */
struct tpf_ctx_t
{
    rocke_value_t* block_m_off;
    rocke_value_t* block_n_off;
    dgrad_magic_div_t hw_div;
    dgrad_magic_div_t w_div;
    dgrad_magic_div_t di_div;
    bool is_3d;
    rocke_value_t *c0, *c_Ho, *c_Wo, *c_K, *p_Do, *c_yxc;
    rocke_value_t *dh, *dw, *dd, *a_off, *w_k, *live, *tap;
    bool pp; /* tapk_pingpong: the unsigned-compare dY form */
    /* pp_boxes: the box's (h, w) origin, its depth split (ds = field 11,
     * db = field 12) and the pixel-list row table (pp_boxes 7/77). */
    bool boxes;
    rocke_value_t *bx_hb, *bx_wb, *bx_ds, *bx_db, *rm_base, *rm_buf;
    bool pp_epi; /* tapk_pp_epi: W loads past the last tile go out of range */
};

/* dy_pf (the non-pingpong form) */
static rocke_value_t* _tpf_dy_descriptor(rocke_ir_builder_t* b_,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         rocke_value_t** out_valid,
                                         void* user)
{
    tpf_ctx_t* c = (tpf_ctx_t*)user;
    rocke_value_t* m_sub = rocke_b_add(b_, c->block_m_off, row);
    if(c->rm_base != NULL)
    {
        rocke_value_t* ra = rocke_b_add(b_, c->rm_base, m_sub);
        m_sub = rocke_b_global_load_i32(b_, c->rm_buf, ra, 4);
    }
    rocke_value_t *n_val, *m_rem, *hi, *wi;
    _magic_divmod(b_, m_sub, &c->hw_div, &n_val, &m_rem);
    _magic_divmod(b_, m_rem, &c->w_div, &hi, &wi);
    if(c->boxes)
    {
        hi = rocke_b_add(b_, hi, c->bx_hb);
        wi = rocke_b_add(b_, wi, c->bx_wb);
    }
    rocke_value_t* ho = rocke_b_add(b_, hi, c->dh);
    rocke_value_t* wo = rocke_b_add(b_, wi, c->dw);
    if(c->pp)
    {
        /* One unsigned compare per axis (a negative coordinate wraps). */
        rocke_value_t* hu = rocke_b_cmp_ult(b_, ho, c->c_Ho);
        rocke_value_t* wu = rocke_b_cmp_ult(b_, wo, c->c_Wo);
        rocke_value_t* valid = rocke_b_land(b_, hu, wu);
        if(c->is_3d)
        {
            rocke_value_t *nq, *di;
            if(c->boxes)
            {
                /* _bx_split_d: (image, dX depth) of n*ds + (d - db). */
                nq = rocke_b_div(b_, n_val, c->bx_ds);
                rocke_value_t* nd_ = rocke_b_mul(b_, nq, c->bx_ds);
                rocke_value_t* r_ = rocke_b_sub(b_, n_val, nd_);
                di = rocke_b_add(b_, r_, c->bx_db);
            }
            else
                _magic_divmod(b_, n_val, &c->di_div, &nq, &di);
            rocke_value_t* dov = rocke_b_add(b_, di, c->dd);
            rocke_value_t* du = rocke_b_cmp_ult(b_, dov, c->p_Do);
            valid = rocke_b_land(b_, valid, du);
            rocke_value_t* nd = rocke_b_mul(b_, nq, c->p_Do);
            n_val = rocke_b_add(b_, nd, di);
        }
        rocke_value_t* n_ho = rocke_b_mul(b_, n_val, c->c_Ho);
        rocke_value_t* n_ho_hi = rocke_b_add(b_, n_ho, hi);
        rocke_value_t* row_off = rocke_b_mul(b_, n_ho_hi, c->c_Wo);
        rocke_value_t* pix_w = rocke_b_add(b_, row_off, wi);
        rocke_value_t* pix_k = rocke_b_mul(b_, pix_w, c->c_K);
        rocke_value_t* pix = rocke_b_add(b_, pix_k, col);
        *out_valid = valid;
        return rocke_b_add(b_, pix, c->a_off);
    }
    rocke_value_t* ho_ge = rocke_b_cmp_ge(b_, ho, c->c0);
    rocke_value_t* ho_lt = rocke_b_cmp_lt(b_, ho, c->c_Ho);
    rocke_value_t* ho_ok = rocke_b_land(b_, ho_ge, ho_lt);
    rocke_value_t* wo_ge = rocke_b_cmp_ge(b_, wo, c->c0);
    rocke_value_t* wo_lt = rocke_b_cmp_lt(b_, wo, c->c_Wo);
    rocke_value_t* wo_ok = rocke_b_land(b_, wo_ge, wo_lt);
    rocke_value_t* valid = rocke_b_land(b_, ho_ok, wo_ok);
    if(c->is_3d)
    {
        rocke_value_t *nq, *di;
        _magic_divmod(b_, n_val, &c->di_div, &nq, &di);
        rocke_value_t* dov = rocke_b_add(b_, di, c->dd);
        rocke_value_t* d_ge = rocke_b_cmp_ge(b_, dov, c->c0);
        rocke_value_t* d_lt = rocke_b_cmp_lt(b_, dov, c->p_Do);
        rocke_value_t* d_ok = rocke_b_land(b_, d_ge, d_lt);
        valid = rocke_b_land(b_, valid, d_ok);
        rocke_value_t* nd = rocke_b_mul(b_, nq, c->p_Do);
        n_val = rocke_b_add(b_, nd, di);
    }
    if(c->live != NULL)
        valid = rocke_b_land(b_, valid, c->live);
    rocke_value_t* n_ho = rocke_b_mul(b_, n_val, c->c_Ho);
    rocke_value_t* n_ho_hi = rocke_b_add(b_, n_ho, hi);
    rocke_value_t* row_off = rocke_b_mul(b_, n_ho_hi, c->c_Wo);
    rocke_value_t* pix_w = rocke_b_add(b_, row_off, wi);
    rocke_value_t* pix_k = rocke_b_mul(b_, pix_w, c->c_K);
    rocke_value_t* pix = rocke_b_add(b_, pix_k, col);
    *out_valid = valid;
    return rocke_b_add(b_, pix, c->a_off);
}

/* w_pf */
static rocke_value_t* _tpf_w_descriptor(rocke_ir_builder_t* b_,
                                        rocke_value_t* row,
                                        rocke_value_t* col,
                                        rocke_value_t** out_valid,
                                        void* user)
{
    tpf_ctx_t* c = (tpf_ctx_t*)user;
    rocke_value_t* c_val = rocke_b_add(b_, c->block_n_off, row);
    rocke_value_t* cm = rocke_b_mul(b_, col, c->c_yxc);
    rocke_value_t* cv = rocke_b_add(b_, cm, c_val);
    *out_valid = (c->pp_epi && c->live != NULL) ? c->live : NULL;
    return rocke_b_add(b_, cv, c->w_k);
}

static rocke_value_t* _tpf_w_descriptor_kouter(rocke_ir_builder_t* b_,
                                               rocke_value_t* row,
                                               rocke_value_t* col,
                                               rocke_value_t** out_valid,
                                               void* user)
{
    return _tpf_w_descriptor(b_, col, row, out_valid, user);
}

/* DgradConvSpec.xhalo_lds_rows */
static int _dgrad_xhalo_lds_rows(const rocke_dgrad_conv_spec_t* s)
{
    if(s->xhalo_exact)
        return s->tile_m + s->problem.X + 1;
    const int rows = s->tile_m + s->problem.X - 1;
    const int pass_elems = rocke_dgrad_conv_spec_block_size(s) * 8;
    const int quantum = pass_elems / _gcd(pass_elems, s->tile_k);
    return (rows + quantum - 1) / quantum * quantum + 1;
}

/* DgradConvSpec.xh_a_k4_rows (no xh_dlds) */
static int _dgrad_xh_a_k4_rows(const rocke_dgrad_conv_spec_t* s)
{
    const int rows = _dgrad_xhalo_lds_rows(s);
    const int r = 16 * 8 / s->tile_k > 1 ? 16 * 8 / s->tile_k : 1;
    const int want = r >= 2 ? (r / 2) % 8 : 1;
    return rows + (((want - rows) % 8) + 8) % 8;
}

/* Python _dgrad_b_load_vec: the M-outer row-mode B load width. */
static int _dgrad_b_load_vec(const rocke_dgrad_conv_spec_t* s)
{
    const int cpg = rocke_conv_problem_cpg(&s->problem);
    const bool f32 = s->dtype_a && strcmp(s->dtype_a, "fp32") == 0;
    int def_vec_b = 1;
    const int cand[] = {f32 ? 4 : 8, 4, 2, 1};
    for(int v : cand)
        if(cpg % v == 0)
        {
            def_vec_b = v;
            break;
        }
    int max_vb = s->has_vector_size_b
                     ? (s->vector_size_b < def_vec_b ? s->vector_size_b : def_vec_b)
                     : def_vec_b;
    int out = 1;
    if(rocke_coalesced_tile_loader_choose_vec_axis(
           s->tile_n, s->tile_k, rocke_dgrad_conv_spec_block_size(s), max_vb, true, &out)
       != ROCKE_OK)
        return 1;
    return out;
}

#ifndef ROCKE_DGRAD_PP_BOX_SPLIT_CUS
#define ROCKE_DGRAD_PP_BOX_SPLIT_CUS 228 /* Python PP_BOX_SPLIT_CUS (ROCKE_PP_BOX_CUS) */
#endif

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
    if(spec->no_agpr && b->kernel)
        rocke_attr_set_str(b, &b->kernel->attrs, "agpr_alloc", "0,0");

    // ---- params ----
    const rocke_type_t* ab_ir = _dtype_to_ir(spec->dtype_a);
    /* ir_dtype_b: the W tensor, its LDS tile and loader (ab_ir is ir_dtype_a). */
    const rocke_type_t* b_ir = _dtype_to_ir(spec->dtype_b);
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
    /* pp_box_det writes its workspace behind the record table. */
    buf_opts.readonly = !spec->pp_box_det;
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
    const bool is_3d = p->is_3d;
    ctx.params_is_3d = is_3d;
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
            {"p_Z", &ctx.p_Z},
            {"p_Di", &ctx.p_Di},
            {"p_sD", &ctx.p_sD},
            {"p_pD", &ctx.p_pD},
            {"p_dD", &ctx.p_dD},
            {"p_groups", &ctx.p_groups},
            {"p_Ho", &ctx.p_Ho},
            {"p_Wo", &ctx.p_Wo},
            {"p_Do", &ctx.p_Do},
            {"p_cpg", &ctx.p_cpg},
            {"p_kpg", &ctx.p_kpg},
            {"p_dg_M", &ctx.p_dg_M}, /* N*Hi*Wi */
            {"p_dg_N", &ctx.p_dg_N}, /* cpg     */
            {"p_dg_K", &ctx.p_dg_K}, /* Y*X*kpg */
            {"p_dY_stride_n", &ctx.p_dY_stride_n},
            {"p_dY_stride_do", &ctx.p_dY_stride_do},
            {"p_dY_stride_ho", &ctx.p_dY_stride_ho},
            {"p_dY_stride_wo", &ctx.p_dY_stride_wo},
            {"p_W_stride_k", &ctx.p_W_stride_k},
            {"p_W_stride_z", &ctx.p_W_stride_z},
            {"p_W_stride_y", &ctx.p_W_stride_y},
            {"p_W_stride_x", &ctx.p_W_stride_x},
            {"p_dX_stride_n", &ctx.p_dX_stride_n},
            {"p_dX_stride_di", &ctx.p_dX_stride_di},
            {"p_dX_stride_hi", &ctx.p_dX_stride_hi},
            {"p_dX_stride_wi", &ctx.p_dX_stride_wi},
            {"p_magic_m_Di_mult", &ctx.p_magic_m_Di_mult},
            {"p_magic_m_Di_shift", &ctx.p_magic_m_Di_shift},
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
        rocke_conv_arg_names("dgrad", is_3d, /*two_stage=*/false, &abi);
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
                  : rocke_mfma_atom(rocke_conv_data_mma_operand(spec->dtype_a, spec->mma_dtype),
                                    spec->warp_tile_m,
                                    spec->warp_tile_n,
                                    spec->warp_tile_k);
    int a_per_lane = op->srcs[0].frag_len;
    int b_per_lane = op->srcs[1].frag_len;
    /* LDS element type follows STORAGE: a TF32 kernel holds fp32 tiles and
     * reinterprets each fragment after the read (_to_mma_operand). */
    const rocke_type_t* smem_dtype = rocke_conv_smem_dtype_for(spec->dtype_a);
    const bool is_tf32 = rocke_conv_data_is_tf32(spec->mma_dtype);
    int c_per_lane = op->dst.frag_len;
    /* Scheduling-hint group sizes. Only TF32 departs from one ds_read and one
     * MFMA per fragment / atom: _load_mma_operand reads its fp32 fragment in
     * loads of at most 4 elements (two ds_read_b128 for the 8-wide gfx950
     * fragment) and the gfx950 BF16x3 emulation issues three physical MFMAs
     * per logical atom. */
    int ds_reads_per_frag = 1;
    int mfma_issues_per_atom = 1;
    if(is_tf32)
    {
        ds_reads_per_frag
            = (a_per_lane + ROCKE_CONV_TF32_LDS_LOAD_MAX - 1) / ROCKE_CONV_TF32_LDS_LOAD_MAX;
        mfma_issues_per_atom = rocke_conv_mfma_issues_per_mma(op);
    }

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
        /* ungrouped (xcd_remap): the order also serves groups == 1. */
        rocke_value_t* is_grouped = spec->xcd_remap ? rocke_b_cmp_ge(b, ctx.p_groups, c1_xcd)
                                                    : rocke_b_cmp_gt(b, ctx.p_groups, c1_xcd);
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
    const bool fold_record = _dgrad_folds_sub_gemm_record(spec) && !spec->pp_boxes;
    if(spec->pp_boxes && spec->xcd_remap)
    {
        /* pp_boxes: XCD-contiguous order over the boxes' tiles (the last
         * record's end is the launch's tile count). */
        rocke_value_t* last = rocke_b_sub(b, num_sub_gemms_param, rocke_b_const_i32(b, 1));
        rocke_value_t* rb
            = rocke_b_mul(b, last, rocke_b_const_i32(b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS));
        rocke_value_t* l0 = rocke_b_global_load_i32(b, sub_gemm_buf, rb, 4);
        rocke_value_t* r1 = rocke_b_add(b, rb, rocke_b_const_i32(b, 1));
        rocke_value_t* l1 = rocke_b_global_load_i32(b, sub_gemm_buf, r1, 4);
        rocke_value_t* r2 = rocke_b_add(b, rb, rocke_b_const_i32(b, 2));
        rocke_value_t* l2 = rocke_b_global_load_i32(b, sub_gemm_buf, r2, 4);
        rocke_value_t* l12 = rocke_b_mul(b, l1, l2);
        rocke_value_t* sk;
        if(spec->split_k > 1)
        {
            rocke_value_t* r15 = rocke_b_add(b, rb, rocke_b_const_i32(b, 15));
            rocke_value_t* l15 = rocke_b_global_load_i32(b, sub_gemm_buf, r15, 4);
            sk = rocke_b_smax(b, l15, rocke_b_const_i32(b, 1));
        }
        else
            sk = rocke_b_const_i32(b, 1);
        rocke_value_t* l12s = rocke_b_mul(b, l12, sk);
        rocke_value_t* tot = rocke_b_add(b, l0, l12s);
        rocke_value_t* nx = rocke_b_const_i32(b, spec->xcd_remap);
        if(spec->pp_box_order & 2)
        {
            rocke_value_t* win = rocke_b_const_i32(b, ROCKE_DGRAD_PP_BOX_SPLIT_CUS);
            rocke_value_t* fw = rocke_b_div(b, flat_block_id, win);
            rocke_value_t* w0 = rocke_b_mul(b, fw, win);
            rocke_value_t* tw0 = rocke_b_sub(b, tot, w0);
            rocke_value_t* wt = rocke_b_smin(b, tw0, win);
            rocke_value_t* loc = rocke_b_sub(b, flat_block_id, w0);
            rocke_value_t* xw = rocke_b_mod(b, loc, nx);
            rocke_value_t* q = rocke_b_div(b, wt, nx);
            rocke_value_t* qn = rocke_b_mul(b, q, nx);
            rocke_value_t* remv = rocke_b_sub(b, wt, qn);
            rocke_value_t* xq = rocke_b_mul(b, xw, q);
            rocke_value_t* mn = rocke_b_smin(b, xw, remv);
            rocke_value_t* a0 = rocke_b_add(b, xq, mn);
            rocke_value_t* ld_ = rocke_b_div(b, loc, nx);
            rocke_value_t* v = rocke_b_add(b, a0, ld_);
            rocke_value_t* ge = rocke_b_cmp_ge(b, wt, nx);
            rocke_value_t* sel = rocke_b_select(b, ge, v, loc);
            flat_block_id = rocke_b_add(b, w0, sel);
        }
        else
        {
            rocke_value_t* xw = rocke_b_mod(b, flat_block_id, nx);
            rocke_value_t* q = rocke_b_div(b, tot, nx);
            rocke_value_t* qn = rocke_b_mul(b, q, nx);
            rocke_value_t* remv = rocke_b_sub(b, tot, qn);
            rocke_value_t* xq = rocke_b_mul(b, xw, q);
            rocke_value_t* mn = rocke_b_smin(b, xw, remv);
            rocke_value_t* a0 = rocke_b_add(b, xq, mn);
            rocke_value_t* ld_ = rocke_b_div(b, flat_block_id, nx);
            rocke_value_t* v = rocke_b_add(b, a0, ld_);
            rocke_value_t* ge = rocke_b_cmp_ge(b, tot, nx);
            flat_block_id = rocke_b_select(b, ge, v, flat_block_id);
        }
    }
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
    rocke_value_t* rec_num_m_tiles = _ld(1);
    rocke_value_t* rec_num_n_tiles = _ld(2);
    rocke_value_t* rec_gemm_m = _ld(3);
    rocke_value_t* rec_gemm_k = _ld(4);
    rocke_value_t* rec_h_tilde_slice = _ld(5);
    rocke_value_t* rec_w_tilde_slice = _ld(6);
    rocke_value_t* rec_h_tilde_slice_begin = _ld(7);
    rocke_value_t* rec_w_tilde_slice_begin = _ld(8);
    rocke_value_t* rec_y_dot_slice = _ld(9);
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
    rocke_value_t* rm_base = NULL; /* pp_boxes 7/77 row table base (_rm_map) */
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
        /* pp_boxes 7/77: a record's rows are its pixel list (after the magic
         * table: row bases, then the rows); emitted before the magic loads. */
        if(spec->pp_boxes == 7 || spec->pp_boxes == 77)
        {
            rocke_value_t* cm = rocke_b_const_i32(
                b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS + ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS);
            rocke_value_t* nm = rocke_b_mul(b, num_sub_gemms_param, cm);
            rocke_value_t* tab = rocke_b_add(b, nm, num_sub_gemms_param);
            if(spec->pp_box_det)
                tab = rocke_b_add(b, tab, rocke_b_const_i32(b, 2));
            rocke_value_t* tsub = rocke_b_sub(b, tab, num_sub_gemms_param);
            rocke_value_t* toff = rocke_b_add(b, tsub, sg_idx);
            rocke_value_t* tl = rocke_b_global_load_i32(b, sub_gemm_buf, toff, 4);
            rm_base = rocke_b_add(b, tab, tl);
        }
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
    rocke_value_t* local_m_tile = NULL;
    rocke_value_t* local_n_tile = NULL;
    /* pp_boxes split-K: a record's blocks are its tiles x its own split
     * degree (record field 15), slice fastest (order bit 1: (n, slice, m)). */
    rocke_value_t* bx_slice = NULL;
    rocke_value_t* bx_sk = NULL;
    const bool bx_split = spec->pp_boxes && spec->split_k > 1;
    if(bx_split && (spec->pp_box_order & 1))
    {
        bx_sk = rocke_b_smax(b, rec_b_x_stride, rocke_b_const_i32(b, 1));
        local_m_tile = rocke_b_mod(b, local_flat, rec_num_m_tiles);
        rocke_value_t* rest = rocke_b_div(b, local_flat, rec_num_m_tiles);
        local_n_tile = rocke_b_div(b, rest, bx_sk);
        rocke_value_t* nk = rocke_b_mul(b, local_n_tile, bx_sk);
        bx_slice = rocke_b_sub(b, rest, nk);
    }
    if(bx_split && !(spec->pp_box_order & 1))
    {
        bx_sk = rocke_b_smax(b, rec_b_x_stride, rocke_b_const_i32(b, 1));
        rocke_value_t* tl = rocke_b_div(b, local_flat, bx_sk);
        rocke_value_t* tk = rocke_b_mul(b, tl, bx_sk);
        bx_slice = rocke_b_sub(b, local_flat, tk);
        local_flat = tl;
    }
    if(!(bx_split && (spec->pp_box_order & 1)))
    {
        local_m_tile = rocke_b_div(b, local_flat, rec_num_n_tiles);
        local_n_tile = rocke_b_mod(b, local_flat, rec_num_n_tiles);
    }
    if(spec->m_band && fold_record)
    {
        /* Output-plane band order (DgradConvSpec.m_band). */
        rocke_value_t* plane = rocke_b_mul(b, ctx.p_Hi, ctx.p_Wi);
        local_m_tile = rocke_conv_emit_m_band_order(
            b, local_m_tile, ctx.p_num_pid_m, plane, block_m, NULL, spec->m_band);
    }

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
    if(is_3d)
        c_dY_rows = rocke_b_mul(b, c_dY_rows, ctx.p_Do);

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
    const bool xh = spec->tap_outer_k && spec->tapk_xhalo;
    const bool xh_ak4 = xh && spec->xh_a_k4;
    const int ak4_mp = xh_ak4 ? _dgrad_xh_a_k4_rows(spec) : 0;
    int a_shape[2] = {spec->dy_halo ? _dgrad_dy_halo_lds_rows(spec)
                      : xh          ? _dgrad_xhalo_lds_rows(spec)
                                    : block_m,
                      lds_layout.row_stride};
    if(xh_ak4)
    {
        /* xh_a_k4: K0 slabs of Mp rows x 4 elements. */
        a_shape[0] = block_k / 4 * ak4_mp;
        a_shape[1] = 4;
    }
    /* A stays M-outer; only B flips. Mirrors the Python branch. */
    int b_shape_arr[2] = {block_n, lds_layout.row_stride};
    if(spec->lds_k_outer)
    {
        b_shape_arr[0] = block_k;
        b_shape_arr[1] = block_n + _dgrad_kouter_b_pad(spec);
    }
    if(xh && spec->xh_b_kpad && !spec->lds_k_outer)
    {
        /* xh_b_kpad: the B tiles padded (not swizzled) under lds_xor. */
        b_shape_arr[0] = block_n;
        b_shape_arr[1] = block_k + spec->xh_b_kpad;
    }
    rocke_value_t* A_smem = rocke_b_smem_alloc(b, ab_ir, a_shape, 2, "A_smem");
    rocke_value_t* B_smem;
    if(xh && spec->xh_b_wide)
    {
        /* xh_b_wide: the X taps' B tiles stacked (+ load_vec junk rows). */
        int bw_shape[2] = {spec->problem.X * block_n + _dgrad_b_load_vec(spec), b_shape_arr[1]};
        B_smem = rocke_b_smem_alloc(b, b_ir, bw_shape, 2, "B_smem");
    }
    else if(spec->tapk_pp_bslab)
    {
        /* tapk_pp_bslab: K0 slabs of tile_n rows x 4 elements. */
        int bs_shape[2] = {block_k / 4 * block_n, 4};
        B_smem = rocke_b_smem_alloc(b, b_ir, bs_shape, 2, "B_smem");
    }
    else
        B_smem = rocke_b_smem_alloc(b, b_ir, b_shape_arr, 2, "B_smem");
    /* dy_halo=2: the second B buffer of the per-tap double buffer. */
    rocke_value_t* B_smem2 = (spec->dy_halo == 2 || spec->tapk_prefetch == 2)
                                 ? rocke_b_smem_alloc(b, b_ir, b_shape_arr, 2, "B_smem2")
                                 : NULL;
    /* tapk_xhalo: one B tile per x tap (stacked in B_smem with xh_b_wide). */
    rocke_value_t* B_smem_x[8] = {};
    int xh_b_off[8] = {};
    if(xh)
        for(int x = 0; x < spec->problem.X && x < 8; ++x)
        {
            if(spec->xh_b_wide || x == 0)
                B_smem_x[x] = B_smem;
            else
            {
                char nm[24];
                snprintf(nm, sizeof(nm), "B_smem_x%d", x);
                B_smem_x[x] = rocke_b_smem_alloc(b, b_ir, b_shape_arr, 2, nm);
            }
            xh_b_off[x] = spec->xh_b_wide ? x * block_n : 0;
        }
    bool xh_yr_done = false; /* xh_yring emitted both branches' epilogues */
    rocke_value_t* pp_bf16[ROCKE_CONV_MAX_ACCS]; /* _pp_state["bf16"] */
    int pp_bf16_n = 0;
    /* tapk_prefetch=2: the second A buffer of the ping-pong. */
    rocke_value_t* A_smem2
        = spec->tapk_prefetch == 2 ? rocke_b_smem_alloc(b, ab_ir, a_shape, 2, "A_smem2") : NULL;

    // ---- MFMA tile counts ----
    int mfmas_m = rocke_dgrad_conv_spec_mfmas_per_warp_m(spec);
    int mfmas_n = rocke_dgrad_conv_spec_mfmas_per_warp_n(spec);
    int k_atoms = rocke_dgrad_conv_spec_k_atoms_per_tile_k(spec);
    int num_accs = mfmas_m * mfmas_n;

    // ---- accumulators ----
    rocke_value_t* acc_init = rocke_b_zero_vec_f32(b, op->srcs[2].frag_len);
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
    if(spec->split_k <= 1 || ((spec->tapk_xhalo || spec->tapk_pingpong) && spec->tap_outer_k))
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

    rocke_coalesced_tile_loader_t a_sync_loader{};
    a_sync_loader.tile_rows = block_m;
    a_sync_loader.tile_cols = block_k;
    a_sync_loader.block_size = threads;
    a_sync_loader.load_vec = load_vec_a;
    a_sync_loader.elem_dtype = NULL; /* F16 */
    a_sync_loader.use_buffer_rsrc = true;
    a_sync_loader.oob_sentinel = 2147483647;
    a_sync_loader.vector_axis_row = false;
    a_sync_loader.has_inner_dim = false;
    a_sync_loader.inner_dim = 0;
    a_sync_loader.has_col_xor = false;
    a_sync_loader.elem_dtype = ab_ir; /* elem_dtype=ir_dtype_a */

    rocke_coalesced_tile_loader_t b_sync_loader{};
    b_sync_loader.tile_rows = spec->lds_k_outer ? block_k : block_n;
    b_sync_loader.tile_cols = spec->lds_k_outer ? block_n : block_k;
    b_sync_loader.block_size = threads;
    b_sync_loader.load_vec = load_vec_b;
    b_sync_loader.elem_dtype = NULL; /* F16 */
    b_sync_loader.use_buffer_rsrc = true;
    b_sync_loader.oob_sentinel = 2147483647;
    b_sync_loader.vector_axis_row = axis_b_row;
    b_sync_loader.has_inner_dim = false;
    b_sync_loader.inner_dim = 0;
    b_sync_loader.has_col_xor = false;
    b_sync_loader.elem_dtype = b_ir; /* elem_dtype=ir_dtype_b */
    if(spec->b_kpack)
    {
        /* KPackedRowLoader: the widest width giving every thread whole units. */
        if(!axis_b_row || load_vec_b < 2)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "b_kpack needs a row-mode B load of width >= 2");
            return NULL;
        }
        b_sync_loader.tile_rows = block_n;
        b_sync_loader.tile_cols = block_k;
        b_sync_loader.load_vec = _kpr_choose_vec(
            block_n,
            block_k,
            threads,
            spec->tapk_pp_bslab == 2 ? (load_vec_b < 2 ? load_vec_b : 2) : load_vec_b,
            spec->b_kpack);
        b_sync_loader.row_kpack = spec->b_kpack;
        /* KPackedRowLoader.n_fast (tapk_pingpong >= 3), carried in kp_line. */
        b_sync_loader.kp_line = spec->tapk_pingpong >= 3 ? 1 : 0;
    }

    // ---- wavelet loaders (pipeline="wavelet" only) ----
    rocke_coalesced_tile_loader_t a_wavelet_loader{};
    rocke_coalesced_tile_loader_t b_wavelet_loader{};
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
        a_wavelet_loader.elem_dtype = ab_ir;
        b_wavelet_loader.elem_dtype = b_ir;
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
                a_rows[mi] = rocke_conv_emit_frag_smem_load(b,
                                                            A_src,
                                                            a_row_in_atom,
                                                            a_k_in_atom,
                                                            atom_row,
                                                            k_tile_base,
                                                            a_per_lane,
                                                            smem_dtype);
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
                b_wma_cols[ni] = rocke_conv_emit_frag_smem_load(b,
                                                                B_src,
                                                                b_col_in_atom,
                                                                b_k_in_atom,
                                                                atom_row,
                                                                k_tile_base,
                                                                b_per_lane,
                                                                smem_dtype);
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
        if(spec->split_k > 1 && spec->tapk_pingpong && !is_wmma && spec->epilogue
           && strcmp(spec->epilogue, "cshuffle") == 0)
        {
            /* x-halo / ping-pong split-K (pp_boxes): slices of a split record
             * (bx_sk > 1) leave their partials in the pp_box_det workspace
             * slot; a record the host did not split stores like the unsplit
             * box. */
            rocke_value_t* gt = rocke_b_cmp_gt(b, bx_sk, rocke_b_const_i32(b, 1));
            rocke_if_else_t ie = rocke_b_scf_if_else(b, gt);
            rocke_b_region_enter(b, ie.then_region);
            {
                /* _pbd_store_slot */
                const int n_w = spec->warp_m * spec->warp_n;
                const int tile_elems = block_m * block_n;
                const int eb = spec->pp_box_det == 2 ? 2 : 4;
                rocke_value_t* nb = rocke_b_const_i32(b, 0x7FFFFFF0);
                rocke_value_t* ws = rocke_b_buffer_rsrc(b, sub_gemm_buf, nb);
                (void)rocke_b_const_i32(b, 0); /* make_buffer_resource's soffset */
                rocke_value_t* cf = rocke_b_const_i32(
                    b, ROCKE_DGRAD_SUB_GEMM_RECORD_FIELDS + ROCKE_DGRAD_SUB_GEMM_MAGIC_FIELDS);
                rocke_value_t* wo = rocke_b_mul(b, num_sub_gemms_param, cf);
                rocke_value_t* wl = rocke_b_global_load_i32(b, sub_gemm_buf, wo, 4);
                rocke_value_t* ws_word = rocke_b_readfirstlane(b, wl);
                rocke_value_t* wd = rocke_b_div(b, tid, rocke_b_const_i32(b, spec->wave_size));
                rocke_value_t* wave = rocke_b_readfirstlane(b, wd);
                rocke_value_t* w64 = rocke_b_mul(b, wave, rocke_b_const_i32(b, 64));
                rocke_value_t* wl_ = rocke_b_add(b, w64, lane);
                rocke_value_t* lane_off = rocke_b_mul(b, wl_, rocke_b_const_i32(b, 4 * eb));
                const int acc_stride = n_w * 64 * 4 * eb;
                rocke_value_t* w4 = rocke_b_mul(b, ws_word, rocke_b_const_i32(b, 4));
                rocke_value_t* fb
                    = rocke_b_mul(b, flat_block_id, rocke_b_const_i32(b, tile_elems * eb));
                rocke_value_t* slot_base = rocke_b_add(b, w4, fb);
                rocke_value_t* base = rocke_b_add(b, slot_base, lane_off);
                for(int j = 0; j < n_epi; ++j)
                {
                    rocke_value_t* off = rocke_b_add(b, base, rocke_b_const_i32(b, j * acc_stride));
                    if(eb == 2)
                        rocke_b_buffer_store_vN_bf16(
                            b, ws, off, c0, rocke_b_vec_trunc_f32_to_bf16(b, epi_accs_[j]), 2);
                    else
                        rocke_b_buffer_store_vN_f32(b, ws, off, c0, epi_accs_[j], 4);
                }
            }
            rocke_b_region_leave(b);
            rocke_b_region_enter(b, ie.else_region);
            {
                tilde_dx_ctx_t bctx = _make_tilde_dx_ctx(&hw_div,
                                                         &w_div,
                                                         rec_d_h_stride,
                                                         rec_d_h_offset,
                                                         rec_d_w_stride,
                                                         rec_d_w_offset,
                                                         c_Hi,
                                                         c_Wi,
                                                         c_C,
                                                         c_group_base);
                bctx.rm_base = rm_base;
                bctx.rm_buf = sub_gemm_buf;
                if(is_3d)
                {
                    bctx.bx_ds = rec_a_embed_h_coeff;
                    bctx.bx_db = rec_a_embed_w_coeff;
                    bctx.p_Di = ctx.p_Di;
                }
                _emit_dgrad_box_cshuffle_epilogue(
                    b, spec, atom, &grid, epi_accs_, n_epi, dx_rsrc, rec_gemm_m, c_dg_N, &bctx);
            }
            rocke_b_region_leave(b);
            return;
        }
        if(spec->tapk_pp_epi && spec->pp_boxes && spec->split_k <= 1 && !is_wmma && spec->epilogue
           && strcmp(spec->epilogue, "cshuffle") == 0)
        {
            /* pp_boxes + tapk_pp_epi: the pp cshuffle store through the box
             * row map. */
            tilde_dx_ctx_t bctx = _make_tilde_dx_ctx(&hw_div,
                                                     &w_div,
                                                     rec_d_h_stride,
                                                     rec_d_h_offset,
                                                     rec_d_w_stride,
                                                     rec_d_w_offset,
                                                     c_Hi,
                                                     c_Wi,
                                                     c_C,
                                                     c_group_base);
            bctx.rm_base = rm_base;
            bctx.rm_buf = sub_gemm_buf;
            if(is_3d)
            {
                bctx.bx_ds = rec_a_embed_h_coeff;
                bctx.bx_db = rec_a_embed_w_coeff;
                bctx.p_Di = ctx.p_Di;
            }
            _emit_pp_cshuffle_epilogue(b,
                                       spec,
                                       atom,
                                       &grid,
                                       epi_accs_,
                                       pp_bf16_n ? pp_bf16 : NULL,
                                       dx_rsrc,
                                       _tilde_dx_addr_fn,
                                       &bctx,
                                       rec_gemm_m,
                                       c_dg_N,
                                       spec->epi_lds_barriers);
            (void)n_epi;
            return;
        }
        bool is_split_k_atomic_ = (spec->split_k > 1);
        bool is_strided_ = rocke_dgrad_conv_spec_is_strided(spec);
        bool use_cshuffle_sk_ = (spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0);
        if(is_split_k_atomic_ && use_cshuffle_sk_ && !is_wmma)
        {
            /* Split-K MFMA cshuffle: the staged tile is atomic-added into dX
             * (Python: _emit_dgrad_tilde_cshuffle_epilogue(atomic_dx_ptr=dX)). */
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
                                                patch_row,
                                                dX);
        }
        else if(!is_split_k_atomic_ && !is_strided_)
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
                if(use_cshuffle_ && spec->xh_epi == 1)
                    _emit_xh_cshuffle_epilogue(b,
                                               spec,
                                               &ctx,
                                               atom,
                                               epi_accs_,
                                               &grid,
                                               dx_rsrc,
                                               c_group_base,
                                               spec->epi_lds_barriers);
                else if(use_cshuffle_ && spec->pp_boxes)
                {
                    tilde_dx_ctx_t bctx = _make_tilde_dx_ctx(&hw_div,
                                                             &w_div,
                                                             rec_d_h_stride,
                                                             rec_d_h_offset,
                                                             rec_d_w_stride,
                                                             rec_d_w_offset,
                                                             c_Hi,
                                                             c_Wi,
                                                             c_C,
                                                             c_group_base);
                    bctx.rm_base = rm_base;
                    bctx.rm_buf = sub_gemm_buf;
                    if(is_3d)
                    {
                        bctx.bx_ds = rec_a_embed_h_coeff;
                        bctx.bx_db = rec_a_embed_w_coeff;
                        bctx.p_Di = ctx.p_Di;
                    }
                    _emit_dgrad_box_cshuffle_epilogue(
                        b, spec, atom, &grid, epi_accs_, n_epi, dx_rsrc, rec_gemm_m, c_dg_N, &bctx);
                }
                else if(use_cshuffle_)
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
                                                    patch_row,
                                                    NULL);
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
    rocke_value_t* A_src_cur = A_smem; /* tapk_prefetch=2 reads A_smem2 too */

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
                    a_rows[mi] = rocke_conv_load_mma_operand(
                        b, A_src_cur, a_row, col_base, a_per_lane, smem_dtype, is_tf32);
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
                    b_cols[ni] = rocke_conv_load_mma_operand(
                        b, B_src, b_row, col_base, b_per_lane, smem_dtype, is_tf32);
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

                rocke_schedule_policy_emit_after_mfma_step(&schedule,
                                                           b,
                                                           (mfmas_m + mfmas_n) * ds_reads_per_frag,
                                                           mfmas_m * mfmas_n
                                                               * mfma_issues_per_atom);
            }
        }
        else if(is_wmma)
        {
            emit_wmma_phase(A_src_cur, B_src, in_vars, new_accs);
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
    else if(spec->tap_outer_k && spec->tapk_xhalo)
    {
        /* xh_yring: ring tiles (no (n, d) plane starts inside the reused rows)
       * run the ring loop, the others the full-load loop, each with its own
       * A / B LDS buffers and epilogue (Python: an scf_if_else over them). */
        const bool yr = spec->xh_yring > 0;
        rocke_if_else_t yr_ie;
        memset(&yr_ie, 0, sizeof(yr_ie));
        if(yr)
        {
            rocke_value_t* bm2
                = rocke_b_add(b, block_m_off_v, rocke_b_const_i32(b, 2 - spec->problem.X));
            rocke_value_t* yr_first = rocke_b_smax(b, bm2, c0);
            rocke_value_t *yr_q, *yr_r;
            _magic_divmod(b, yr_first, &hw_div, &yr_q, &yr_r);
            rocke_value_t* req = rocke_b_cmp_eq(b, yr_r, c0);
            rocke_value_t* q1 = rocke_b_add(b, yr_q, rocke_b_const_i32(b, 1));
            rocke_value_t* qm = rocke_b_mul(b, q1, hw_div.value);
            rocke_value_t* yr_next = rocke_b_select(b, req, yr_first, qm);
            rocke_value_t* bme = rocke_b_add(b, block_m_off_v, rocke_b_const_i32(b, block_m));
            rocke_value_t* ge = rocke_b_cmp_ge(b, yr_next, bme);
            rocke_value_t* s1v = rocke_b_select(b, ge, rocke_b_const_i32(b, 1), c0);
            rocke_value_t* yr_ok = rocke_b_readfirstlane(b, s1v);
            rocke_value_t* ne = rocke_b_cmp_ne(b, yr_ok, c0);
            yr_ie = rocke_b_scf_if_else(b, ne);
        }
        rocke_value_t* A_smem_keep = A_smem;
        rocke_value_t* B_smem_keep = B_smem;
        rocke_value_t* B_smem_x_keep[8];
        for(int x = 0; x < 8; ++x)
            B_smem_x_keep[x] = B_smem_x[x];
        for(int yr_pass = 0; yr_pass < (yr ? 2 : 1); ++yr_pass)
        {
            const bool ring = yr && yr_pass == 0;
            if(yr)
            {
                rocke_b_region_enter(b, yr_pass == 0 ? yr_ie.then_region : yr_ie.else_region);
                if(yr_pass == 1)
                {
                    A_smem = rocke_b_smem_alloc(b, ab_ir, a_shape, 2, "A_smem_yf");
                    int bw_shape[2]
                        = {spec->problem.X * block_n + _dgrad_b_load_vec(spec), b_shape_arr[1]};
                    B_smem = rocke_b_smem_alloc(b, b_ir, bw_shape, 2, "B_smem_yf");
                    for(int x = 0; x < spec->problem.X && x < 8; ++x)
                        B_smem_x[x] = B_smem;
                }
            }
            /* Python _emit_tap_outer_xhalo_loop + _xhalo_prefetch (_xh_plain):
         * (z, y) taps outer, chunks inner, the X x taps unrolled over one
         * staged dY halo tile. Ported: tapk_prefetch 1/3, xhalo_exact,
         * xh_fast_a 1 (+ xh_tap_rsrc), xh_b_wide (+ xh_b_kpad), xh_a_k4,
         * xh_stride_m, tapk_lds_ahead / tapk_ilv / xh_defer schedules,
         * tapk_setprio 1, tapk_lds_barrier, xh_order 1/2/4, tapk_ztrim 1/2. */
            const int XB = spec->problem.X;
            const int back = XB - 1;
            const rocke_type_t* adt = smem_dtype ? smem_dtype : rocke_f16();
            const bool xsm = spec->xh_stride_m;
            /* _emit_tap_outer_xhalo_loop preamble */
            {
                (void)rocke_b_const_i32(b, 1);
                rocke_value_t* yx_ = rocke_b_mul(b, c_Y, c_X);
                rocke_value_t* tp_ = is_3d ? rocke_b_mul(b, ctx.p_Z, yx_) : yx_;
                if(is_3d)
                    (void)rocke_b_mul(b, ctx.p_Z, c_Y);
                (void)rocke_b_mul(b, tp_, c_cstride);
            }
            /* NB: the preamble products above feed only the plain loop (c_yxc of
         * the outer function); _xhalo_prefetch recomputes its own. */
            const int lds_rows = _dgrad_xhalo_lds_rows(spec);
            rocke_value_t* c_zero_row = rocke_b_const_i32(b, lds_rows - 1);
            /* cap */
            rocke_value_t *cap_dh = NULL, *cap_dd = NULL, *cap_kb = NULL, *cap_w_k0 = NULL;
            /* xh_yring state (the ring loads and row maps replace the exact ones) */
            bool ring_on_ = false;
            bool use_ring_load = false, use_ring_rmap = false, ring_full_cur = true;
            int ring_shift_cur = 0, ring_shift_ld = 0;
            std::function<void(bool, int, rocke_value_t*, rocke_ctl_staged_t*)> ring_load_fn;
            std::function<rocke_value_t*(int, int, rocke_value_t*)> ring_rmap_fn;
            std::function<void(int, int)> ring_rmap_make;
            (void)ring_on_;
            /* _xhalo_exact_parts */
            if(!spec->xhalo_exact)
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "tapk_xhalo without xhalo_exact is Python-only");
                return NULL;
            }
            rocke_value_t* c_junk_row = rocke_b_const_i32(b, lds_rows - 2);
            rocke_coalesced_tile_loader_t main_ld{};
            main_ld.tile_rows = block_m;
            main_ld.tile_cols = block_k;
            main_ld.block_size = threads;
            main_ld.load_vec = load_vec_a;
            main_ld.use_buffer_rsrc = true;
            main_ld.oob_sentinel = 2147483647;
            main_ld.vector_axis_row = false;
            main_ld.elem_dtype = smem_dtype;
            const int back_span = block_k / load_vec_a;
            const int n_back_vec = back * back_span;
            auto xh_dy_tail = [&](rocke_value_t* m,
                                  rocke_value_t* m_ok,
                                  rocke_value_t* col,
                                  rocke_value_t** out_valid) -> rocke_value_t* {
                m = rocke_b_select(b, m_ok, m, c0);
                rocke_value_t *n_val, *m_rem, *hi, *wi;
                _magic_divmod(b, m, &hw_div, &n_val, &m_rem);
                _magic_divmod(b, m_rem, &w_div, &hi, &wi);
                rocke_value_t* ho = rocke_b_add(b, hi, cap_dh);
                rocke_value_t* h0 = rocke_b_cmp_ge(b, ho, c0);
                rocke_value_t* h1 = rocke_b_cmp_lt(b, ho, c_Ho);
                rocke_value_t* h01 = rocke_b_land(b, h0, h1);
                rocke_value_t* w1 = rocke_b_cmp_lt(b, wi, c_Wo);
                rocke_value_t* hw = rocke_b_land(b, h01, w1);
                rocke_value_t* valid = rocke_b_land(b, m_ok, hw);
                if(is_3d)
                {
                    rocke_value_t *nq, *di;
                    dgrad_magic_div_t dd_
                        = {ctx.p_Di, ctx.p_magic_m_Di_mult, ctx.p_magic_m_Di_shift};
                    _magic_divmod(b, n_val, &dd_, &nq, &di);
                    rocke_value_t* dov = rocke_b_add(b, di, cap_dd);
                    rocke_value_t* d0 = rocke_b_cmp_ge(b, dov, c0);
                    rocke_value_t* d1 = rocke_b_cmp_lt(b, dov, ctx.p_Do);
                    rocke_value_t* d01 = rocke_b_land(b, d0, d1);
                    valid = rocke_b_land(b, valid, d01);
                    rocke_value_t* nd = rocke_b_mul(b, nq, ctx.p_Do);
                    n_val = rocke_b_add(b, nd, dov);
                }
                rocke_value_t* nh = rocke_b_mul(b, n_val, c_Ho);
                rocke_value_t* nhh = rocke_b_add(b, nh, ho);
                rocke_value_t* nw = rocke_b_mul(b, nhh, c_Wo);
                rocke_value_t* pix = rocke_b_add(b, nw, wi);
                rocke_value_t* pk = rocke_b_mul(b, pix, c_K);
                rocke_value_t* pc = rocke_b_add(b, pk, col);
                *out_valid = valid;
                return rocke_b_add(b, pc, cap_kb);
            };
            /* the exact-parts dy(row, col) */
            auto slow_dy = [&](rocke_value_t* row, rocke_value_t* col, rocke_value_t** out_valid) {
                rocke_value_t* m = rocke_b_add(b, block_m_off_v, row);
                rocke_value_t* m0 = rocke_b_cmp_ge(b, m, c0);
                rocke_value_t* m1 = rocke_b_cmp_lt(b, m, ctx.p_dg_M);
                rocke_value_t* m_ok = rocke_b_land(b, m0, m1);
                return xh_dy_tail(m, m_ok, col, out_valid);
            };
            /* _xh_fast_a_descriptor state */
            struct fa_rec
            {
                rocke_value_t *pix, *mask;
            };
            static fa_rec rec[64];
            int n_rec = 0;
            int fa_ctr = 0;
            bool fa_armed = false;
            bool fa_recording = false;
            rocke_value_t* fa_c1 = NULL;
            /* the descriptor callback adaptor */
            struct xh_desc_env
            {
                std::function<rocke_value_t*(rocke_value_t*, rocke_value_t*, rocke_value_t**)> fn;
            };
            xh_desc_env xh_env;
            auto xh_desc_tramp = [](rocke_ir_builder_t* bb,
                                    rocke_value_t* row,
                                    rocke_value_t* col,
                                    rocke_value_t** out_valid,
                                    void* user) -> rocke_value_t* {
                (void)bb;
                return ((xh_desc_env*)user)->fn(row, col, out_valid);
            };
            auto axis_bits = [&](rocke_value_t* coord,
                                 rocke_value_t* pad,
                                 rocke_value_t* out,
                                 rocke_value_t* ext) -> rocke_value_t* {
                rocke_value_t* hi_ = rocke_b_add(b, coord, pad);
                rocke_value_t* d0 = rocke_b_sub(b, hi_, out);
                rocke_value_t* lo_ = rocke_b_add(b, d0, fa_c1);
                rocke_value_t* l0 = rocke_b_smax(b, lo_, c0);
                rocke_value_t* lo_c = rocke_b_smin(b, l0, ext);
                rocke_value_t* e1 = rocke_b_sub(b, ext, fa_c1);
                rocke_value_t* hi_c = rocke_b_smin(b, hi_, e1);
                rocke_value_t* w0 = rocke_b_sub(b, hi_c, lo_c);
                rocke_value_t* w1 = rocke_b_add(b, w0, fa_c1);
                rocke_value_t* width = rocke_b_smax(b, w1, c0);
                rocke_value_t* s1 = rocke_b_shl(b, fa_c1, width);
                rocke_value_t* s2 = rocke_b_sub(b, s1, fa_c1);
                return rocke_b_shl(b, s2, lo_c);
            };
            auto fa_record
                = [&](rocke_value_t* row, rocke_value_t* col, rocke_value_t** out_valid) {
                      rocke_value_t* m = rocke_b_add(b, block_m_off_v, row);
                      rocke_value_t* m0 = rocke_b_cmp_ge(b, m, c0);
                      rocke_value_t* m1 = rocke_b_cmp_lt(b, m, ctx.p_dg_M);
                      rocke_value_t* m_ok = rocke_b_land(b, m0, m1);
                      m = rocke_b_select(b, m_ok, m, c0);
                      rocke_value_t *n_val, *m_rem, *hi, *wi;
                      _magic_divmod(b, m, &hw_div, &n_val, &m_rem);
                      _magic_divmod(b, m_rem, &w_div, &hi, &wi);
                      rocke_value_t* di = NULL;
                      if(is_3d)
                      {
                          rocke_value_t* nq;
                          dgrad_magic_div_t dd_
                              = {ctx.p_Di, ctx.p_magic_m_Di_mult, ctx.p_magic_m_Di_shift};
                          _magic_divmod(b, n_val, &dd_, &nq, &di);
                          rocke_value_t* nd = rocke_b_mul(b, nq, ctx.p_Do);
                          n_val = rocke_b_add(b, nd, di);
                      }
                      rocke_value_t* nh = rocke_b_mul(b, n_val, c_Ho);
                      rocke_value_t* nhh = rocke_b_add(b, nh, hi);
                      rocke_value_t* nw = rocke_b_mul(b, nhh, c_Wo);
                      rocke_value_t* pw = rocke_b_add(b, nw, wi);
                      rocke_value_t* pk = rocke_b_mul(b, pw, c_K);
                      rocke_value_t* pix = rocke_b_add(b, pk, col);
                      rocke_value_t* vy = axis_bits(hi, ctx.p_pH, c_Ho, c_Y);
                      rocke_value_t* mask;
                      if(is_3d)
                      {
                          rocke_value_t* vz = axis_bits(di, ctx.p_pD, ctx.p_Do, ctx.p_Z);
                          mask = c0;
                          for(int z = 0; z < 3; ++z)
                          {
                              rocke_value_t* vzs = rocke_b_lshr(b, vz, rocke_b_const_i32(b, z));
                              rocke_value_t* vb = rocke_b_land(b, vzs, fa_c1);
                              rocke_value_t* zb = rocke_b_sub(b, c0, vb);
                              rocke_value_t* zc = rocke_b_mul(b, rocke_b_const_i32(b, z), c_Y);
                              rocke_value_t* vsh = rocke_b_shl(b, vy, zc);
                              rocke_value_t* an = rocke_b_land(b, zb, vsh);
                              mask = rocke_b_lor(b, mask, an);
                          }
                      }
                      else
                          mask = vy;
                      rocke_value_t* wl = rocke_b_cmp_lt(b, wi, c_Wo);
                      rocke_value_t* row_ok = rocke_b_land(b, m_ok, wl);
                      mask = rocke_b_select(b, row_ok, mask, c0);
                      rec[n_rec].pix = pix;
                      rec[n_rec].mask = mask;
                      n_rec++;
                      *out_valid = rocke_b_cmp_lt(b, fa_c1, c0);
                      return c0;
                  };
            auto fa_fast = [&](rocke_value_t* row, rocke_value_t* col, rocke_value_t** out_valid) {
                const int i = fa_ctr % n_rec;
                fa_ctr++;
                rocke_value_t* pix = rec[i].pix;
                rocke_value_t* mask = rec[i].mask;
                rocke_value_t* zy;
                if(is_3d)
                {
                    rocke_value_t* z = rocke_b_sub(b, ctx.p_pD, cap_dd);
                    rocke_value_t* zc = rocke_b_mul(b, z, c_Y);
                    rocke_value_t* yy = rocke_b_sub(b, ctx.p_pH, cap_dh);
                    zy = rocke_b_add(b, zc, yy);
                    rocke_value_t* t0 = rocke_b_mul(b, cap_dd, c_Ho);
                    rocke_value_t* t1 = rocke_b_add(b, t0, cap_dh);
                    rocke_value_t* wk = rocke_b_mul(b, c_Wo, c_K);
                    rocke_value_t* t2 = rocke_b_mul(b, t1, wk);
                    (void)rocke_b_add(b, t2, cap_kb); /* tap (unused under xh_tap_rsrc) */
                }
                else
                {
                    zy = rocke_b_sub(b, ctx.p_pH, cap_dh);
                    rocke_value_t* wk = rocke_b_mul(b, c_Wo, c_K);
                    rocke_value_t* t2 = rocke_b_mul(b, cap_dh, wk);
                    (void)rocke_b_add(b, t2, cap_kb);
                }
                (void)row;
                (void)col;
                if(spec->xh_tap_rsrc)
                {
                    rocke_value_t* nm = rocke_b_xor(b, mask, rocke_b_const_i32(b, -1));
                    rocke_value_t* sh = rocke_b_lshr(b, nm, zy);
                    rocke_value_t* bad = rocke_b_land(b, sh, fa_c1);
                    rocke_value_t* b30 = rocke_b_shl(b, bad, rocke_b_const_i32(b, 30));
                    rocke_value_t* off = rocke_b_lor(b, pix, b30);
                    *out_valid = i == n_rec - 1 ? rocke_b_cmp_lt(b, c0, fa_c1) : NULL;
                    return off;
                }
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "xh_fast_a without xh_tap_rsrc is Python-only");
                *out_valid = NULL;
                return pix;
            };
            auto xh_dy = [&](rocke_value_t* row,
                             rocke_value_t* col,
                             rocke_value_t** out_valid) -> rocke_value_t* {
                if(fa_recording)
                    return fa_record(row, col, out_valid);
                if(fa_armed)
                    return fa_fast(row, col, out_valid);
                return slow_dy(row, col, out_valid);
            };
            xh_env.fn = xh_dy;
            /* _HaloLoader.load_global */
            auto halo_load = [&](rocke_value_t* rsrc, rocke_ctl_staged_t* st) {
                main_ld.no_loads = fa_recording;
                rocke_coalesced_tile_loader_load_global(
                    b, &main_ld, tid, xh_desc_tramp, &xh_env, rsrc, NULL, st);
                main_ld.no_loads = false;
                rocke_value_t* c_span = rocke_b_const_i32(b, back_span);
                rocke_value_t* j = rocke_b_div(b, tid, c_span);
                rocke_value_t* tm = rocke_b_mod(b, tid, c_span);
                rocke_value_t* col = rocke_b_mul(b, tm, rocke_b_const_i32(b, load_vec_a));
                rocke_value_t* lane_ok = rocke_b_cmp_lt(b, tid, rocke_b_const_i32(b, n_back_vec));
                rocke_value_t* jb = rocke_b_sub(b, j, rocke_b_const_i32(b, back));
                rocke_value_t* valid = NULL;
                rocke_value_t* off = xh_dy(jb, col, &valid);
                valid = rocke_b_land(b, valid, lane_ok);
                rocke_value_t* ob
                    = rocke_b_mul(b, off, rocke_b_const_i32(b, adt == rocke_f32() ? 4 : 2));
                rocke_value_t* oobv = rocke_b_const_i32(b, 2147483647);
                rocke_value_t* off_b = rocke_b_select(b, valid, ob, oobv);
                rocke_value_t* v
                    = fa_recording ? NULL
                                   : rocke_b_buffer_load_vN(b, rsrc, off_b, c0, adt, load_vec_a);
                rocke_value_t* jr = rocke_b_add(b, j, rocke_b_const_i32(b, block_m));
                rocke_value_t* row = rocke_b_select(b, lane_ok, jr, c_junk_row);
                st->vecs[st->count].row = row;
                st->vecs[st->count].col = col;
                st->vecs[st->count].v = v;
                st->count++;
            };
            if(spec->xh_fast_a)
            {
                if(spec->xh_fast_a != 1)
                {
                    rocke_i_set_err(b, ROCKE_ERR_VALUE, "xh_fast_a 2 is Python-only");
                    return NULL;
                }
                fa_c1 = rocke_b_const_i32(b, 1);
                fa_recording = true;
                rocke_ctl_staged_t tmp;
                halo_load(dy_rsrc, &tmp);
                fa_recording = false;
                fa_armed = true;
            }
            /* _ak4_row (xh_stride_m) */
            auto ak4_row = [&](rocke_value_t* row) -> rocke_value_t* {
                if(!xsm)
                    return row;
                rocke_value_t* s4 = rocke_b_lshr(b, row, rocke_b_const_i32(b, 4));
                rocke_value_t* m = rocke_b_land(b, s4, rocke_b_const_i32(b, mfmas_m - 1));
                return rocke_b_xor(b, row, m);
            };
            if(!xh_ak4 || !spec->xh_b_wide || !spec->xh_b_kpad || spec->tapk_prefetch == 2
               || !spec->tapk_prefetch)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "this tapk_xhalo layout is Python-only (needs xh_a_k4, xh_b_wide, "
                                "xh_b_kpad, tapk_prefetch 1/3)");
                return NULL;
            }
            /* zero row: one 4-element zero piece per k-group slab */
            {
                rocke_value_t* zk0 = rocke_b_mod(b, tid, rocke_b_const_i32(b, block_k / 4));
                rocke_value_t* zm = rocke_b_mul(b, zk0, rocke_b_const_i32(b, ak4_mp));
                rocke_value_t* zr = ak4_row(c_zero_row);
                rocke_value_t* za = rocke_b_add(b, zm, zr);
                rocke_value_t* zv = rocke_b_zero_vec(b, adt, 4);
                rocke_value_t* idx[2] = {za, c0};
                rocke_b_smem_store_vN(b, A_smem, idx, 2, zv, 4);
            }
            /* ---- _xhalo_prefetch ---- */
            rocke_value_t* c1 = rocke_b_const_i32(b, 1);
            rocke_value_t* c_kd = c_kdiv;
            rocke_value_t* c_cs = c_cstride;
            rocke_value_t* c_yx = rocke_b_mul(b, c_Y, c_X);
            rocke_value_t* c_taps = is_3d ? rocke_b_mul(b, ctx.p_Z, c_yx) : c_yx;
            rocke_value_t* c_zy = is_3d ? rocke_b_mul(b, ctx.p_Z, c_Y) : c_Y;
            rocke_value_t* c_yxc = rocke_b_mul(b, c_taps, c_cs);
            rocke_value_t* kdb = rocke_b_div(b, c_kd, c_block_k);
            rocke_value_t* n_tiles = rocke_b_mul(b, c_zy, kdb);
            rocke_value_t *y_lo = NULL, *y_cnt = NULL, *y_end = NULL;
            auto ztrim_range = [&](rocke_value_t** lo_o, rocke_value_t** cnt_o) {
                rocke_value_t* m0 = block_m_off_v;
                rocke_value_t* ma = rocke_b_add(b, m0, rocke_b_const_i32(b, block_m - 1));
                rocke_value_t* mm = rocke_b_sub(b, ctx.p_dg_M, rocke_b_const_i32(b, 1));
                rocke_value_t* m1 = rocke_b_smin(b, ma, mm);
                rocke_value_t *q0, *r0, *q1, *r1;
                _magic_divmod(b, m0, &hw_div, &q0, &r0);
                _magic_divmod(b, m1, &hw_div, &q1, &r1);
                dgrad_magic_div_t dd_ = {ctx.p_Di, ctx.p_magic_m_Di_mult, ctx.p_magic_m_Di_shift};
                rocke_value_t *n0, *di0, *n1, *di1;
                _magic_divmod(b, q0, &dd_, &n0, &di0);
                _magic_divmod(b, q1, &dd_, &n1, &di1);
                rocke_value_t* c1_ = rocke_b_const_i32(b, 1);
                rocke_value_t* zmax = rocke_b_sub(b, ctx.p_Z, c1_);
                rocke_value_t* a0 = rocke_b_add(b, di0, ctx.p_pD);
                rocke_value_t* a1 = rocke_b_sub(b, a0, ctx.p_Do);
                rocke_value_t* a2 = rocke_b_add(b, a1, c1_);
                rocke_value_t* lo = rocke_b_smax(b, a2, c0);
                rocke_value_t* h0 = rocke_b_add(b, di1, ctx.p_pD);
                rocke_value_t* hi = rocke_b_smin(b, h0, zmax);
                rocke_value_t* same = rocke_b_cmp_eq(b, n0, n1);
                lo = rocke_b_select(b, same, lo, c0);
                hi = rocke_b_select(b, same, hi, zmax);
                rocke_value_t* d = rocke_b_sub(b, hi, lo);
                rocke_value_t* d1 = rocke_b_add(b, d, c1_);
                rocke_value_t* cnt = rocke_b_smax(b, d1, c0);
                *lo_o = rocke_b_readfirstlane(b, lo);
                *cnt_o = rocke_b_readfirstlane(b, cnt);
            };
            if(spec->tapk_ztrim == 2)
            {
                /* _ytrim_range */
                rocke_value_t* m0 = block_m_off_v;
                rocke_value_t* ma = rocke_b_add(b, m0, rocke_b_const_i32(b, block_m - 1));
                rocke_value_t* mm = rocke_b_sub(b, ctx.p_dg_M, rocke_b_const_i32(b, 1));
                rocke_value_t* m1 = rocke_b_smin(b, ma, mm);
                rocke_value_t *q0, *r0, *q1, *r1, *h0, *h1, *t;
                _magic_divmod(b, m0, &hw_div, &q0, &r0);
                _magic_divmod(b, m1, &hw_div, &q1, &r1);
                _magic_divmod(b, r0, &w_div, &h0, &t);
                _magic_divmod(b, r1, &w_div, &h1, &t);
                rocke_value_t* c1_ = rocke_b_const_i32(b, 1);
                rocke_value_t* ymax = rocke_b_sub(b, c_Y, c1_);
                rocke_value_t* a0 = rocke_b_add(b, h0, ctx.p_pH);
                rocke_value_t* a1 = rocke_b_sub(b, a0, c_Ho);
                rocke_value_t* a2 = rocke_b_add(b, a1, c1_);
                rocke_value_t* lo = rocke_b_smax(b, a2, c0);
                rocke_value_t* hh = rocke_b_add(b, h1, ctx.p_pH);
                rocke_value_t* hi = rocke_b_smin(b, hh, ymax);
                rocke_value_t* same = rocke_b_cmp_eq(b, q0, q1);
                lo = rocke_b_select(b, same, lo, c0);
                hi = rocke_b_select(b, same, hi, ymax);
                rocke_value_t* d = rocke_b_sub(b, hi, lo);
                rocke_value_t* d1 = rocke_b_add(b, d, c1_);
                rocke_value_t* cnt = rocke_b_smax(b, d1, c0);
                y_lo = rocke_b_readfirstlane(b, lo);
                y_cnt = rocke_b_readfirstlane(b, cnt);
                y_end = rocke_b_add(b, y_lo, y_cnt);
            }
            /* XTapKPackLoader (xh_b_wide) */
            const int bw_lv = _dgrad_b_load_vec(spec);
            const int bw_kp = spec->b_kpack;
            auto set_tile = [&](rocke_value_t* kb0, rocke_value_t* yd, rocke_value_t* zd) {
                cap_dh = rocke_b_sub(b, ctx.p_pH, yd);
                rocke_value_t* zy = yd;
                if(is_3d)
                {
                    cap_dd = rocke_b_sub(b, ctx.p_pD, zd);
                    rocke_value_t* zc = rocke_b_mul(b, zd, c_Y);
                    zy = rocke_b_add(b, zc, yd);
                }
                rocke_value_t* kb = grouped ? rocke_b_add(b, kb0, k_out_group_base) : kb0;
                cap_kb = kb;
                rocke_value_t* kw = rocke_b_mul(b, kb, c_yxc);
                rocke_value_t* zx = rocke_b_mul(b, zy, c_X);
                rocke_value_t* zc2 = rocke_b_mul(b, zx, c_cs);
                cap_w_k0 = rocke_b_add(b, kw, zc2);
            };
            auto advance = [&](rocke_value_t* const* st, rocke_value_t** out) {
                rocke_value_t* nk = rocke_b_add(b, st[0], c_block_k);
                rocke_value_t* wrap_k = rocke_b_cmp_ge(b, nk, c_kd);
                nk = rocke_b_select(b, wrap_k, c0, nk);
                rocke_value_t* sy = rocke_b_select(b, wrap_k, c1, c0);
                rocke_value_t* ny = rocke_b_add(b, st[1], sy);
                out[0] = nk;
                if(!is_3d)
                {
                    out[1] = ny;
                    out[2] = NULL;
                    return;
                }
                rocke_value_t* wrap_y;
                if(y_lo != NULL)
                {
                    wrap_y = rocke_b_cmp_ge(b, ny, y_end);
                    ny = rocke_b_select(b, wrap_y, y_lo, ny);
                }
                else
                {
                    wrap_y = rocke_b_cmp_ge(b, ny, c_Y);
                    ny = rocke_b_select(b, wrap_y, c0, ny);
                }
                rocke_value_t* sz = rocke_b_select(b, wrap_y, c1, c0);
                out[1] = ny;
                out[2] = rocke_b_add(b, st[2], sz);
            };
            auto w_wide_load = [&](rocke_value_t* rsrc, rocke_ctl_staged_t* st) {
                /* XTapKPackLoader.load_global with w_wide_fn */
                const rocke_type_t* et = adt;
                rocke_value_t* c_eb = rocke_b_const_i32(b, et == rocke_f32() ? 4 : 2);
                rocke_value_t* cz = rocke_b_const_i32(b, 0);
                rocke_value_t* c_oob = rocke_b_const_i32(b, 2147483647);
                const int k_groups = block_k / bw_kp;
                const int upt = (block_n / bw_lv) * k_groups;
                rocke_value_t* c_upt = rocke_b_const_i32(b, upt);
                rocke_value_t* c_kg = rocke_b_const_i32(b, k_groups);
                const int units = XB * upt;
                st->count = 0;
                for(int e = 0; e < (units + threads - 1) / threads; ++e)
                {
                    rocke_value_t* u = rocke_b_add(b, rocke_b_const_i32(b, e * threads), tid);
                    const bool partial = (e + 1) * threads > units;
                    rocke_value_t* active
                        = partial ? rocke_b_cmp_lt(b, u, rocke_b_const_i32(b, units)) : NULL;
                    if(active != NULL)
                        u = rocke_b_select(b, active, u, cz);
                    rocke_value_t* x = rocke_b_div(b, u, c_upt);
                    rocke_value_t* xu = rocke_b_mul(b, x, c_upt);
                    rocke_value_t* lu = rocke_b_sub(b, u, xu);
                    rocke_value_t* nc = rocke_b_div(b, lu, c_kg);
                    rocke_value_t* nk = rocke_b_mul(b, nc, c_kg);
                    rocke_value_t* kg = rocke_b_sub(b, lu, nk);
                    rocke_value_t* row = rocke_b_mul(b, nc, rocke_b_const_i32(b, bw_lv));
                    rocke_value_t* col = rocke_b_mul(b, kg, rocke_b_const_i32(b, bw_kp));
                    rocke_value_t* vs[8];
                    for(int jj = 0; jj < bw_kp; ++jj)
                    {
                        rocke_value_t* cj
                            = jj ? rocke_b_add(b, col, rocke_b_const_i32(b, jj)) : col;
                        /* w_wide_fn(x, row, cj) */
                        rocke_value_t* c_val = rocke_b_add(b, block_n_off_v, row);
                        rocke_value_t* cm = rocke_b_mul(b, cj, c_yxc);
                        rocke_value_t* cv = rocke_b_add(b, cm, c_val);
                        rocke_value_t* off;
                        if(spec->xh_tap_rsrc)
                        {
                            rocke_value_t* xc = rocke_b_mul(b, x, c_cs);
                            off = rocke_b_add(b, cv, xc);
                        }
                        else
                        {
                            rocke_value_t* xc = rocke_b_mul(b, x, c_cs);
                            rocke_value_t* wx = rocke_b_add(b, cap_w_k0, xc);
                            off = rocke_b_add(b, cv, wx);
                        }
                        rocke_value_t* off_b = rocke_b_mul(b, off, c_eb);
                        rocke_value_t* valid = active;
                        if(valid != NULL)
                            off_b = rocke_b_select(b, valid, off_b, c_oob);
                        vs[jj] = rocke_b_buffer_load_vN(b, rsrc, off_b, cz, et, bw_lv);
                    }
                    rocke_value_t* xr = rocke_b_mul(b, x, rocke_b_const_i32(b, block_n));
                    rocke_value_t* srow = rocke_b_add(b, xr, row);
                    if(active != NULL)
                        srow = rocke_b_select(b, active, srow, rocke_b_const_i32(b, XB * block_n));
                    for(int jj = 0; jj < bw_kp; ++jj)
                    {
                        st->vecs[st->count].row = srow;
                        st->vecs[st->count].col = col;
                        st->vecs[st->count].v = vs[jj];
                        st->count++;
                    }
                }
            };
            /* the stacked B tile store (unswizzled: xh_b_kpad) */
            rocke_coalesced_tile_loader_t bw_store{};
            bw_store.load_vec = bw_lv;
            bw_store.row_kpack = bw_kp;
            bw_store.elem_dtype = smem_dtype;
            auto load_tile = [&](rocke_ctl_staged_t* a_st, rocke_ctl_staged_t* w_st) {
                rocke_value_t* dy_t = dy_rsrc;
                rocke_value_t* w_t = w_rsrc;
                if(spec->xh_tap_rsrc)
                {
                    rocke_value_t* tap_;
                    rocke_value_t* wk = NULL;
                    if(is_3d)
                    {
                        rocke_value_t* t0 = rocke_b_mul(b, cap_dd, c_Ho);
                        rocke_value_t* t1 = rocke_b_add(b, t0, cap_dh);
                        wk = rocke_b_mul(b, c_Wo, c_K);
                        rocke_value_t* t2 = rocke_b_mul(b, t1, wk);
                        tap_ = rocke_b_add(b, t2, cap_kb);
                    }
                    else
                    {
                        wk = rocke_b_mul(b, c_Wo, c_K);
                        rocke_value_t* t2 = rocke_b_mul(b, cap_dh, wk);
                        tap_ = rocke_b_add(b, t2, cap_kb);
                    }
                    rocke_value_t* ta2 = rocke_b_mul(b, tap_, rocke_b_const_i32(b, 2));
                    rocke_value_t* sx = rocke_b_sext(b, ta2, rocke_i64());
                    rocke_value_t* pa = rocke_b_global_ptr_add(b, dY, sx);
                    rocke_value_t* nb = rocke_b_sub(b, dY_bytes, ta2);
                    dy_t = rocke_b_buffer_rsrc(b, pa, nb);
                    rocke_value_t* tw2 = rocke_b_mul(b, cap_w_k0, rocke_b_const_i32(b, 2));
                    rocke_value_t* sxw = rocke_b_sext(b, tw2, rocke_i64());
                    rocke_value_t* pw = rocke_b_global_ptr_add(b, W, sxw);
                    rocke_value_t* nbw = rocke_b_sub(b, W_bytes, tw2);
                    w_t = rocke_b_buffer_rsrc(b, pw, nbw);
                }
                const bool b_first = spec->xh_order & 1;
                a_st->count = 0;
                auto a_ld = [&]() {
                    if(use_ring_load)
                        ring_load_fn(ring_full_cur, ring_shift_ld, dy_t, a_st);
                    else
                        halo_load(dy_t, a_st);
                };
                if(!b_first)
                    a_ld();
                w_wide_load(w_t, w_st);
                if(b_first)
                    a_ld();
            };
            auto store_tile = [&](const rocke_ctl_staged_t* a_st, const rocke_ctl_staged_t* w_st) {
                /* xh_store(halo, A_smem, a) with xh_a_k4 slabs */
                rocke_value_t* c_mp = rocke_b_const_i32(b, ak4_mp);
                for(int e = 0; e < a_st->count; ++e)
                {
                    rocke_value_t* row = ak4_row(a_st->vecs[e].row);
                    rocke_value_t* k0 = rocke_b_lshr(b, a_st->vecs[e].col, rocke_b_const_i32(b, 2));
                    for(int jj = 0; jj < load_vec_a / 4; ++jj)
                    {
                        rocke_value_t* kj = jj ? rocke_b_add(b, k0, rocke_b_const_i32(b, jj)) : k0;
                        rocke_value_t* ps[4];
                        for(int i = 0; i < 4; ++i)
                            ps[i] = rocke_b_vec_extract(b, a_st->vecs[e].v, 4 * jj + i);
                        rocke_value_t* piece = rocke_b_vec_pack(b, ps, 4, adt);
                        rocke_value_t* km = rocke_b_mul(b, kj, c_mp);
                        rocke_value_t* ra = rocke_b_add(b, km, row);
                        rocke_value_t* idx[2] = {ra, rocke_b_const_i32(b, 0)};
                        rocke_b_smem_store_vN(b, A_smem, idx, 2, piece, 4);
                    }
                }
                rocke_coalesced_tile_loader_store_lds(b, &bw_store, B_smem, w_st);
            };
            /* row_map(x) of the exact halo */
            struct xh_rmap
            {
                int x;
                rocke_value_t *c_x, *c_wrap;
            };
            auto mk_rmap = [&](int x) {
                xh_rmap r;
                r.x = x;
                r.c_x = rocke_b_const_i32(b, x);
                r.c_wrap = rocke_b_const_i32(b, block_m + back - x);
                return r;
            };
            auto apply_rmap = [&](const xh_rmap& r, rocke_value_t* a_row) -> rocke_value_t* {
                if(r.x == 0)
                    return a_row;
                rocke_value_t* ge = rocke_b_cmp_ge(b, a_row, r.c_x);
                rocke_value_t* sb = rocke_b_sub(b, a_row, r.c_x);
                rocke_value_t* ad = rocke_b_add(b, a_row, r.c_wrap);
                rocke_value_t* row = rocke_b_select(b, ge, sb, ad);
                rocke_value_t* m = rocke_b_add(b, block_m_off_v, a_row);
                rocke_value_t *q, *rem, *q2, *wi;
                _magic_divmod(b, m, &hw_div, &q, &rem);
                _magic_divmod(b, rem, &w_div, &q2, &wi);
                rocke_value_t* wg = rocke_b_cmp_ge(b, wi, r.c_x);
                return rocke_b_select(b, wg, row, c_zero_row);
            };
            /* xh_a_read (xh_a_k4) */
            auto xh_a_read = [&](rocke_value_t* row, rocke_value_t* col, int n) -> rocke_value_t* {
                rocke_value_t* k0 = rocke_b_lshr(b, col, rocke_b_const_i32(b, 2));
                rocke_value_t* km = rocke_b_mul(b, k0, rocke_b_const_i32(b, ak4_mp));
                rocke_value_t* ar = ak4_row(row);
                rocke_value_t* r = rocke_b_add(b, km, ar);
                return rocke_conv_emit_smem_load(
                    b, A_smem, r, rocke_b_const_i32(b, 0), n, smem_dtype);
            };
            auto xh_mfma = [&](rocke_value_t* a, rocke_value_t* bb, rocke_value_t* acc) {
                return spec->xh_epi ? rocke_conv_emit_mfma(b, atom, bb, a, acc)
                                    : rocke_conv_emit_mfma(b, atom, a, bb, acc);
            };
            /* deferred steps (xh_defer): the fragments of each deferred step */
            struct xh_frag
            {
                rocke_value_t* a[ROCKE_CONV_MAX_ACCS];
                rocke_value_t* bc[ROCKE_CONV_MAX_ACCS];
            };
            static xh_frag frags[64];
            int n_deferred = 0;
            int deferred_idx[64];
            /* emit_mfma_phases_ahead over the X taps */
            auto mfma = [&](rocke_value_t* const* acc_in, rocke_value_t** out) {
                if(!spec->tapk_lds_ahead && !spec->xh_defer)
                {
                    /* emit_mfma_phase per x tap */
                    n_deferred = 0;
                    for(int i = 0; i < num_accs; ++i)
                        out[i] = acc_in[i];
                    for(int x = 0; x < XB; ++x)
                    {
                        xh_rmap rm = mk_rmap(x);
                        rocke_lane_decode_t dec = rocke_decode_mfma_lanes(b, atom, lane);
                        rocke_value_t* wmo = rocke_warp_grid_warp_m_off(b, &grid);
                        rocke_value_t* wno = rocke_warp_grid_warp_n_off(b, &grid);
                        for(int kk = 0; kk < k_atoms; ++kk)
                        {
                            rocke_value_t* cm
                                = rocke_b_mul(b, dec.k_blk, rocke_b_const_i32(b, a_per_lane));
                            rocke_value_t* col_base
                                = rocke_b_add(b, cm, rocke_b_const_i32(b, kk * spec->warp_tile_k));
                            rocke_value_t* ar[ROCKE_CONV_MAX_ACCS];
                            rocke_value_t* bc[ROCKE_CONV_MAX_ACCS];
                            for(int mi = 0; mi < mfmas_m; ++mi)
                            {
                                rocke_value_t* c_ = rocke_b_const_i32(b, mi * spec->warp_tile_m);
                                rocke_value_t* t = rocke_b_add(b, c_, dec.m_in_atom);
                                rocke_value_t* a_row = rocke_b_add(b, wmo, t);
                                a_row = apply_rmap(rm, a_row);
                                ar[mi] = xh_a_read(a_row, col_base, a_per_lane);
                            }
                            for(int ni = 0; ni < mfmas_n; ++ni)
                            {
                                rocke_value_t* c_
                                    = rocke_b_const_i32(b, ni * spec->warp_tile_n + xh_b_off[x]);
                                rocke_value_t* t = rocke_b_add(b, c_, dec.n_in_atom);
                                rocke_value_t* b_row = rocke_b_add(b, wno, t);
                                bc[ni] = rocke_conv_emit_smem_load(
                                    b, B_smem_x[x], b_row, col_base, b_per_lane, smem_dtype);
                            }
                            int flat = 0;
                            for(int mi = 0; mi < mfmas_m; ++mi)
                                for(int ni = 0; ni < mfmas_n; ++ni, ++flat)
                                    out[flat] = xh_mfma(ar[mi], bc[ni], out[flat]);
                            rocke_schedule_policy_emit_after_mfma_step(
                                &schedule, b, mfmas_m + mfmas_n, mfmas_m * mfmas_n);
                        }
                    }
                    return;
                }
                xh_rmap rms[8];
                for(int x = 0; x < XB; ++x)
                {
                    if(use_ring_rmap)
                        ring_rmap_make(x, ring_shift_cur);
                    else
                        rms[x] = mk_rmap(x);
                }
                rocke_lane_decode_t dec = rocke_decode_mfma_lanes(b, atom, lane);
                rocke_value_t* wmo = rocke_warp_grid_warp_m_off(b, &grid);
                rocke_value_t* wno = rocke_warp_grid_warp_n_off(b, &grid);
                rocke_value_t* xsm_base = NULL;
                if(xsm)
                {
                    rocke_value_t* mm
                        = rocke_b_mul(b, dec.m_in_atom, rocke_b_const_i32(b, mfmas_m));
                    xsm_base = rocke_b_add(b, wmo, mm);
                }
                const int n_steps = XB * k_atoms;
                rocke_value_t* row_cache[8][ROCKE_CONV_MAX_ACCS] = {};
                static rocke_value_t* afc[8][16][ROCKE_CONV_MAX_ACCS];
                auto reads = [&](int s, xh_frag* f) {
                    const int xi = s / k_atoms, kk = s % k_atoms;
                    rocke_value_t* cm = rocke_b_mul(b, dec.k_blk, rocke_b_const_i32(b, a_per_lane));
                    rocke_value_t* col_base
                        = rocke_b_add(b, cm, rocke_b_const_i32(b, kk * spec->warp_tile_k));
                    for(int mi = 0; mi < mfmas_m; ++mi)
                    {
                        if(xsm && xi > 0 && mi > 0)
                        {
                            f->a[mi] = afc[xi - 1][kk][mi - 1];
                            afc[xi][kk][mi] = f->a[mi];
                            continue;
                        }
                        if(row_cache[xi][mi] == NULL)
                        {
                            rocke_value_t* a_row;
                            if(xsm)
                                a_row = mi ? rocke_b_add(b, xsm_base, rocke_b_const_i32(b, mi))
                                           : xsm_base;
                            else
                            {
                                rocke_value_t* c_ = rocke_b_const_i32(b, mi * spec->warp_tile_m);
                                rocke_value_t* t = rocke_b_add(b, c_, dec.m_in_atom);
                                a_row = rocke_b_add(b, wmo, t);
                            }
                            row_cache[xi][mi] = use_ring_rmap
                                                    ? ring_rmap_fn(xi, ring_shift_cur, a_row)
                                                    : apply_rmap(rms[xi], a_row);
                        }
                        f->a[mi] = xh_a_read(row_cache[xi][mi], col_base, a_per_lane);
                        if(xsm)
                            afc[xi][kk][mi] = f->a[mi];
                    }
                    for(int ni = 0; ni < mfmas_n; ++ni)
                    {
                        rocke_value_t* c_
                            = rocke_b_const_i32(b, ni * spec->warp_tile_n + xh_b_off[xi]);
                        rocke_value_t* t = rocke_b_add(b, c_, dec.n_in_atom);
                        rocke_value_t* b_row = rocke_b_add(b, wno, t);
                        f->bc[ni] = rocke_conv_emit_smem_load(
                            b, B_smem_x[xi], b_row, col_base, b_per_lane, smem_dtype);
                    }
                };
                for(int i = 0; i < num_accs; ++i)
                    out[i] = acc_in[i];
                const int depth = spec->tapk_lds_ahead;
                const int defer = spec->xh_defer;
                n_deferred = 0;
                static xh_frag ring[64];
                for(int s = 0; s < depth && s < n_steps; ++s)
                    reads(s, &ring[s]);
                for(int i = 0; i < n_steps; ++i)
                {
                    if(i + depth < n_steps)
                        reads(i + depth, &ring[i + depth]);
                    if(defer && i >= n_steps - defer)
                    {
                        frags[n_deferred] = ring[i];
                        deferred_idx[n_deferred] = i;
                        n_deferred++;
                        continue;
                    }
                    int flat = 0;
                    for(int mi = 0; mi < mfmas_m; ++mi)
                        for(int ni = 0; ni < mfmas_n; ++ni, ++flat)
                            out[flat] = xh_mfma(ring[i].a[mi], ring[i].bc[ni], out[flat]);
                }
            };
            auto lds_bar = [&]() {
                if(spec->tapk_lds_barrier)
                    rocke_b_sync_lds_only(b);
                else
                    rocke_b_sync(b);
            };
            auto xh_schedule = [&](int n_vmem_all) {
                const int ilv = spec->tapk_ilv;
                const int lda = spec->tapk_lds_ahead;
                if(lda)
                {
                    int n_vmem = n_vmem_all;
                    const int per_step = mfmas_m * mfmas_n;
                    const int n_steps = XB * k_atoms;
                    const int reads_n = mfmas_m + mfmas_n;
                    auto rd = [&](int s_) {
                        if(xsm && s_ >= k_atoms)
                            return 1 + mfmas_n;
                        return reads_n;
                    };
                    if(!ilv)
                    {
                        rocke_b_sched_group_barrier(b, 0x020, n_vmem, 0);
                        n_vmem = 0;
                    }
                    rocke_b_sched_group_barrier(b, 0x100, rd(0), 0);
                    int left_vmem = n_vmem;
                    const int n_pre = n_steps - spec->xh_defer;
                    int left_ds = 0;
                    for(int s_i = 0; s_i < n_pre; ++s_i)
                    {
                        left_ds = s_i + 1 < n_steps ? rd(s_i + 1) : 0;
                        if(spec->xh_defer && s_i == n_pre - 1)
                        {
                            left_ds = 0;
                            for(int t = s_i + 1; t < n_steps; ++t)
                                left_ds += rd(t);
                        }
                        for(int jj = 0; jj < per_step; ++jj)
                        {
                            rocke_b_sched_group_barrier(b, 0x008, 1, 0);
                            if(left_ds > 0)
                            {
                                const int k = lda < left_ds ? lda : left_ds;
                                rocke_b_sched_group_barrier(b, 0x100, k, 0);
                                left_ds -= k;
                            }
                            const int g = s_i * per_step + jj;
                            if(ilv && left_vmem > 0 && g % ilv == ilv - 1)
                            {
                                rocke_b_sched_group_barrier(b, 0x020, 1, 0);
                                left_vmem--;
                            }
                        }
                    }
                    if(left_vmem > 0)
                        rocke_b_sched_group_barrier(b, 0x020, left_vmem, 0);
                    if(spec->xh_defer && left_ds > 0)
                        rocke_b_sched_group_barrier(b, 0x100, left_ds, 0);
                    rocke_b_sched_barrier(b, 0);
                }
                else if(ilv)
                {
                    const int n_mfma = (XB * k_atoms - spec->xh_defer) * mfmas_m * mfmas_n;
                    int used = 0;
                    for(int v = 0; v < n_vmem_all; ++v)
                    {
                        rocke_b_sched_group_barrier(b, 0x020, 1, 0);
                        const int k = ilv < n_mfma - used ? ilv : n_mfma - used;
                        if(k > 0)
                        {
                            rocke_b_sched_group_barrier(b, 0x008, k, 0);
                            used += k;
                        }
                    }
                    rocke_b_sched_barrier(b, 0);
                }
            };
            auto xh_store_deferred = [&](const rocke_ctl_staged_t* ra_,
                                         const rocke_ctl_staged_t* rb_,
                                         rocke_value_t** acc_out) {
                const rocke_ctl_staged_t& ra = *ra_;
                const rocke_ctl_staged_t& rb = *rb_;
                if(n_deferred)
                {
                    store_tile(ra_, rb_);
                    for(int d = 0; d < n_deferred; ++d)
                    {
                        int flat = 0;
                        for(int mi = 0; mi < mfmas_m; ++mi)
                            for(int ni = 0; ni < mfmas_n; ++ni, ++flat)
                                acc_out[flat]
                                    = xh_mfma(frags[d].a[mi], frags[d].bc[ni], acc_out[flat]);
                    }
                    const int n_m = n_deferred * mfmas_m * mfmas_n;
                    const int a_pieces = load_vec_a / 4 > 1 ? load_vec_a / 4 : 1;
                    const int n_w = ra.count * a_pieces + (rb.count / bw_kp) * bw_lv;
                    int per_w = n_m / (n_w > 1 ? n_w : 1);
                    per_w = per_w < 1 ? 1 : per_w;
                    per_w = per_w > 2 ? 2 : per_w;
                    const int lim = n_w < n_m ? n_w : n_m;
                    for(int i = 0; i < lim; ++i)
                    {
                        rocke_b_sched_group_barrier(b, 0x200, 1, 0);
                        rocke_b_sched_group_barrier(b, 0x008, per_w, 0);
                    }
                    rocke_b_sched_barrier(b, 0);
                }
                else
                    store_tile(ra_, rb_);
            };
            /* ---- _xh_yring_dispatch (xh_yring) ---- */
            if(yr)
            {
                const int wi_ = spec->xh_yring % 1000;
                const int need = block_m + back;
                const int P = (need + wi_ - 1) / wi_ * wi_;
                const int n_main = wi_ - back;
                const int Yn = spec->problem.Y;
                const int d_ = spec->xh_yring / 1000;
                if(!xh_ak4 || load_vec_a != 8 || XB < 2 || d_ == 1 || P + 2 > ak4_mp)
                {
                    rocke_i_set_err(
                        b, ROCKE_ERR_VALUE, "xh_yring: the ring geometry does not fit this spec");
                    return NULL;
                }
                bool fulls[16];
                int shifts[16];
                for(int y = 0; y < Yn; ++y)
                    fulls[y] = y == 0 || (d_ && y % d_ == 0);
                for(int y = 0; y < Yn; ++y)
                {
                    int y_f = 0;
                    for(int t = 0; t <= y; ++t)
                        if(fulls[t])
                            y_f = t;
                    shifts[y] = ((y - y_f) * wi_) % P;
                }
                rocke_value_t* c_junk = rocke_b_const_i32(b, P);
                rocke_value_t* c_zrow = rocke_b_const_i32(b, P + 1);
                rocke_value_t* c_threads_r = rocke_b_const_i32(b, threads);
                const int span = block_k / load_vec_a;
                rocke_value_t* c_span_r = rocke_b_const_i32(b, span);
                rocke_value_t* c_lv_r = rocke_b_const_i32(b, load_vec_a);
                int n_full = 0;
                rocke_coalesced_tile_loader_vecs_per_thread(&main_ld, &n_full);
                const int rows_per_pass = threads / span;
                const int n_part = n_main / rows_per_pass;
                auto main_slot = [&](int e, rocke_value_t* row, int shift) -> rocke_value_t* {
                    const int lo = e * rows_per_pass - shift;
                    const int hi = lo + rows_per_pass - 1;
                    if(shift == 0)
                        return row;
                    if(hi < 0)
                        return rocke_b_add(b, row, rocke_b_const_i32(b, P - shift));
                    if(lo >= 0)
                        return rocke_b_sub(b, row, rocke_b_const_i32(b, shift));
                    rocke_value_t* v_ = rocke_b_sub(b, row, rocke_b_const_i32(b, shift));
                    rocke_value_t* lt = rocke_b_cmp_lt(b, v_, c0);
                    rocke_value_t* ad = rocke_b_add(b, v_, rocke_b_const_i32(b, P));
                    return rocke_b_select(b, lt, ad, v_);
                };
                auto back_cb = [&](int shift) { return (((-back - shift) % P) + P) % P; };
                /* ring_layout(full, shift): the (row, col) the ring stores use */
                auto ring_layout = [&](bool full, int shift, rocke_ctl_staged_t* out) {
                    out->count = 0;
                    for(int e = 0; e < (full ? n_full : n_part); ++e)
                    {
                        rocke_value_t* em = rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads_r);
                        rocke_value_t* vec_idx = rocke_b_add(b, em, tid);
                        /* col-mode _decode_row_col */
                        rocke_value_t* row = rocke_b_div(b, vec_idx, c_span_r);
                        rocke_value_t* cm = rocke_b_mod(b, vec_idx, c_span_r);
                        rocke_value_t* col = rocke_b_mul(b, cm, c_lv_r);
                        out->vecs[out->count].row = main_slot(e, row, shift);
                        out->vecs[out->count].col = col;
                        out->vecs[out->count].v = NULL;
                        out->count++;
                    }
                    rocke_value_t* j = rocke_b_div(b, tid, c_span_r);
                    rocke_value_t* tm = rocke_b_mod(b, tid, c_span_r);
                    rocke_value_t* col = rocke_b_mul(b, tm, c_lv_r);
                    rocke_value_t* lane_ok
                        = rocke_b_cmp_lt(b, tid, rocke_b_const_i32(b, n_back_vec));
                    rocke_value_t* jc = rocke_b_add(b, j, rocke_b_const_i32(b, back_cb(shift)));
                    out->vecs[out->count].row = rocke_b_select(b, lane_ok, jc, c_junk);
                    out->vecs[out->count].col = col;
                    out->vecs[out->count].v = NULL;
                    out->count++;
                };
                /* ring_load(full, shift, rsrc) */
                auto ring_load =
                    [&](bool full, int shift, rocke_value_t* rsrc, rocke_ctl_staged_t* out) {
                        const int ne = full ? n_full : n_part;
                        main_ld.vec_lo = 0;
                        main_ld.vec_hi = ne;
                        int k_ = 0;
                        xh_desc_env renv;
                        renv.fn = [&](rocke_value_t* row,
                                      rocke_value_t* col,
                                      rocke_value_t** ov) -> rocke_value_t* {
                            if(fa_armed)
                                fa_ctr = k_;
                            k_++;
                            return xh_dy(row, col, ov);
                        };
                        rocke_ctl_staged_t st;
                        rocke_coalesced_tile_loader_load_global(
                            b, &main_ld, tid, xh_desc_tramp, &renv, rsrc, NULL, &st);
                        main_ld.vec_hi = 0;
                        out->count = 0;
                        for(int e = 0; e < st.count; ++e)
                        {
                            out->vecs[out->count].row = main_slot(e, st.vecs[e].row, shift);
                            out->vecs[out->count].col = st.vecs[e].col;
                            out->vecs[out->count].v = st.vecs[e].v;
                            out->count++;
                        }
                        rocke_value_t* j = rocke_b_div(b, tid, c_span_r);
                        rocke_value_t* tm = rocke_b_mod(b, tid, c_span_r);
                        rocke_value_t* col = rocke_b_mul(b, tm, c_lv_r);
                        rocke_value_t* lane_ok
                            = rocke_b_cmp_lt(b, tid, rocke_b_const_i32(b, n_back_vec));
                        if(fa_armed)
                            fa_ctr = n_rec - 1;
                        rocke_value_t* jb = rocke_b_sub(b, j, rocke_b_const_i32(b, back));
                        rocke_value_t* valid = NULL;
                        rocke_value_t* off = xh_dy(jb, col, &valid);
                        valid = rocke_b_land(b, valid, lane_ok);
                        rocke_value_t* ob
                            = rocke_b_mul(b, off, rocke_b_const_i32(b, adt == rocke_f32() ? 4 : 2));
                        rocke_value_t* oobv = rocke_b_const_i32(b, 2147483647);
                        rocke_value_t* off_b = rocke_b_select(b, valid, ob, oobv);
                        rocke_value_t* v
                            = rocke_b_buffer_load_vN(b, rsrc, off_b, c0, adt, load_vec_a);
                        rocke_value_t* jc = rocke_b_add(b, j, rocke_b_const_i32(b, back_cb(shift)));
                        out->vecs[out->count].row = rocke_b_select(b, lane_ok, jc, c_junk);
                        out->vecs[out->count].col = col;
                        out->vecs[out->count].v = v;
                        out->count++;
                    };
                ring_on_ = ring;
                ring_load_fn = ring_load;
                /* ring_rmap(shift)(x): created once per (x, shift) */
                static bool rm_made[8][256];
                memset(rm_made, 0, sizeof(rm_made));
                ring_shift_cur = 0;
                ring_rmap_fn = [&](int x, int shift, rocke_value_t* a_row) -> rocke_value_t* {
                    const int tot = x + shift;
                    if(tot == 0)
                        return a_row;
                    rocke_value_t* v_ = rocke_b_sub(b, a_row, rocke_b_const_i32(b, tot));
                    rocke_value_t* lt = rocke_b_cmp_lt(b, v_, c0);
                    rocke_value_t* ad = rocke_b_add(b, v_, rocke_b_const_i32(b, P));
                    return rocke_b_select(b, lt, ad, v_);
                };
                ring_rmap_make = [&](int x, int shift) {
                    if(!rm_made[x][shift])
                    {
                        rm_made[x][shift] = true;
                        (void)rocke_b_const_i32(b, x); /* rm()'s c_x */
                    }
                };
                auto sh = [&](int y) { return ring ? shifts[y] : 0; };
                int y_load = 0;
                auto yr_set = [&](rocke_value_t* kb, int y, rocke_value_t* z) {
                    set_tile(kb, rocke_b_const_i32(b, y), is_3d ? z : NULL);
                };
                auto yr_load = [&](int y, rocke_ctl_staged_t* a_st, rocke_ctl_staged_t* w_st) {
                    const bool full = fulls[y] || !ring;
                    ring_full_cur = full;
                    ring_shift_ld = sh(y);
                    use_ring_load = true;
                    load_tile(a_st, w_st);
                    use_ring_load = false;
                };
                /* the zero row of the ring */
                {
                    rocke_value_t* zk0 = rocke_b_mod(b, tid, rocke_b_const_i32(b, block_k / 4));
                    rocke_value_t* zm = rocke_b_mul(b, zk0, rocke_b_const_i32(b, ak4_mp));
                    rocke_value_t* zr = ak4_row(c_zrow);
                    rocke_value_t* za = rocke_b_add(b, zm, zr);
                    rocke_value_t* zv = rocke_b_zero_vec(b, adt, 4);
                    rocke_value_t* idx[2] = {za, c0};
                    rocke_b_smem_store_vN(b, A_smem, idx, 2, zv, 4);
                }
                rocke_value_t* nkc = rocke_b_div(b, c_kd, c_block_k);
                rocke_value_t* z0 = is_3d ? c0 : NULL;
                rocke_value_t* z_cnt_ = is_3d ? ctx.p_Z : c1;
                if(spec->tapk_ztrim && is_3d)
                    ztrim_range(&z0, &z_cnt_);
                rocke_value_t* n_iter_r = rocke_b_mul(b, z_cnt_, nkc);
                yr_set(c0, 0, z0);
                rocke_ctl_staged_t st0a, st0b;
                yr_load(0, &st0a, &st0b);
                store_tile(&st0a, &st0b);
                rocke_b_sync(b);
                yr_set(c0, 1, z0);
                rocke_ctl_staged_t st1a, st1b;
                yr_load(1, &st1a, &st1b);
                const int n_regs_r = st1a.count + st1b.count;
                const char* pf = ring ? "yr" : "yf";
                rocke_iter_arg_t rit[256];
                static char rnames[256][32];
                int n_rit = 0;
                for(int i = 0; i < num_accs; ++i, ++n_rit)
                {
                    snprintf(rnames[n_rit], 32, "%s_%s", iter_args[i].name, pf);
                    rit[n_rit].name = rnames[n_rit];
                    rit[n_rit].init = iter_args[i].init;
                }
                for(int i = 0; i < n_regs_r; ++i, ++n_rit)
                {
                    snprintf(rnames[n_rit], 32, "%s_r%d", pf, i);
                    rit[n_rit].name = rnames[n_rit];
                    rit[n_rit].init = i < st1a.count ? st1a.vecs[i].v : st1b.vecs[i - st1a.count].v;
                }
                snprintf(rnames[n_rit], 32, "%s_kb", pf);
                rit[n_rit].name = rnames[n_rit];
                rit[n_rit].init = c0;
                ++n_rit;
                if(is_3d)
                {
                    snprintf(rnames[n_rit], 32, "%s_z", pf);
                    rit[n_rit].name = rnames[n_rit];
                    rit[n_rit].init = z0;
                    ++n_rit;
                }
                char tname[16];
                snprintf(tname, sizeof(tname), "%s_t", pf);
                rocke_for_t rl
                    = rocke_b_scf_for_iter(b, c0, n_iter_r, c1, rit, n_rit, tname, false, true);
                rocke_b_region_enter(b, rl.body);
                {
                    rocke_value_t** iv = rl.iter_vars;
                    rocke_value_t* cur[ROCKE_CONV_MAX_ACCS];
                    for(int i = 0; i < num_accs; ++i)
                        cur[i] = iv[i];
                    rocke_value_t* const* regs = iv + num_accs;
                    rocke_value_t* kb = iv[num_accs + n_regs_r];
                    rocke_value_t* z = is_3d ? iv[num_accs + n_regs_r + 1] : NULL;
                    rocke_value_t* nk = rocke_b_add(b, kb, c_block_k);
                    rocke_value_t* wrap = rocke_b_cmp_ge(b, nk, c_kd);
                    rocke_value_t* kb_n = rocke_b_select(b, wrap, c0, nk);
                    rocke_value_t* z_n = NULL;
                    if(is_3d)
                    {
                        rocke_value_t* sz = rocke_b_select(b, wrap, c1, c0);
                        z_n = rocke_b_add(b, z, sz);
                    }
                    rocke_ctl_staged_t prev_a, prev_b; /* st_next */
                    prev_a.count = prev_b.count = 0;
                    for(int y = 0; y < Yn; ++y)
                    {
                        if(y + 2 < Yn)
                        {
                            yr_set(kb, y + 2, z);
                            y_load = y + 2;
                        }
                        else
                        {
                            yr_set(kb_n, y + 2 - Yn, z_n);
                            y_load = y + 2 - Yn;
                        }
                        rocke_ctl_staged_t tmpl_a;
                        tmpl_a.count = 0;
                        if(y == 0)
                            ring_layout(fulls[1 % Yn] || !ring, sh(1 % Yn), &tmpl_a);
                        /* yr_step(cur, y, store_st) */
                        if((spec->xh_order & 2) && y == 0)
                            rocke_b_s_waitcnt(b, -1, 0, -1);
                        const int sprio = spec->tapk_setprio;
                        if(sprio)
                        {
                            rocke_b_sched_barrier(b, 0);
                            rocke_b_s_setprio(b, 1);
                        }
                        rocke_ctl_staged_t sa, sb;
                        yr_load(y_load, &sa, &sb);
                        const int ilv = spec->tapk_ilv;
                        const int lda = spec->tapk_lds_ahead;
                        if(!ilv && !lda)
                            rocke_b_sched_barrier(b, 0);
                        rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
                        ring_shift_cur = sh(y);
                        use_ring_rmap = true;
                        mfma(cur, new_accs);
                        use_ring_rmap = false;
                        xh_schedule(sa.count + sb.count);
                        if(sprio)
                        {
                            rocke_b_sched_barrier(b, 0);
                            rocke_b_s_setprio(b, 0);
                        }
                        lds_bar();
                        rocke_ctl_staged_t ssa, ssb;
                        if(y == 0)
                        {
                            /* rebuild([tmpl_a] + staged[1:], regs) */
                            ssa = tmpl_a;
                            ssb = sb;
                            for(int i = 0; i < ssa.count; ++i)
                                ssa.vecs[i].v = regs[i];
                            for(int i = 0; i < ssb.count; ++i)
                                ssb.vecs[i].v = regs[ssa.count + i];
                        }
                        else
                        {
                            ssa = prev_a;
                            ssb = prev_b;
                        }
                        xh_store_deferred(&ssa, &ssb, new_accs);
                        lds_bar();
                        for(int i = 0; i < num_accs; ++i)
                            cur[i] = new_accs[i];
                        prev_a = sa;
                        prev_b = sb;
                    }
                    rocke_value_t* yv[256];
                    int nyv = 0;
                    for(int i = 0; i < num_accs; ++i)
                        yv[nyv++] = cur[i];
                    for(int i = 0; i < prev_a.count; ++i)
                        yv[nyv++] = prev_a.vecs[i].v;
                    for(int i = 0; i < prev_b.count; ++i)
                        yv[nyv++] = prev_b.vecs[i].v;
                    yv[nyv++] = kb_n;
                    if(is_3d)
                        yv[nyv++] = z_n;
                    rocke_b_scf_yield(b, yv, nyv);
                }
                rocke_b_region_leave(b);
                for(int i = 0; i < num_accs; i++)
                    final_accs[i] = rl.op->results[i];
                num_final = num_accs;
            }
            else
            {
                /* ---- _xh_plain ---- */
                const int n_st = is_3d ? 3 : 2;
                const char* st_names[3] = {"xp_kb", "xp_y", "xp_z"};
                rocke_value_t* s0[3] = {c0, c0, is_3d ? c0 : NULL};
                if(spec->tapk_ztrim)
                {
                    rocke_value_t *z_lo, *z_cnt;
                    ztrim_range(&z_lo, &z_cnt);
                    s0[2] = z_lo;
                    rocke_value_t* zc = rocke_b_mul(b, z_cnt, c_Y);
                    rocke_value_t* kd2 = rocke_b_div(b, c_kd, c_block_k);
                    n_tiles = rocke_b_mul(b, zc, kd2);
                    if(y_lo != NULL)
                    {
                        s0[1] = y_lo;
                        rocke_value_t* zyc = rocke_b_mul(b, z_cnt, y_cnt);
                        rocke_value_t* kd3 = rocke_b_div(b, c_kd, c_block_k);
                        n_tiles = rocke_b_mul(b, zyc, kd3);
                    }
                }
                if(spec->split_k > 1)
                {
                    rocke_i_set_err(
                        b, ROCKE_ERR_VALUE, "x-halo split-K is Python-only (no C++ twin yet)");
                    return NULL;
                }
                set_tile(s0[0], s0[1], s0[2]);
                rocke_ctl_staged_t st0a, st0b;
                load_tile(&st0a, &st0b);
                store_tile(&st0a, &st0b);
                rocke_b_sync(b);
                rocke_value_t* s1[3] = {};
                advance(s0, s1);
                rocke_ctl_staged_t r0a, r0b;
                r0a.count = r0b.count = 0;
                rocke_value_t* s_next[3] = {};
                if(spec->tapk_prefetch == 3)
                {
                    set_tile(s1[0], s1[1], s1[2]);
                    load_tile(&r0a, &r0b);
                    advance(s1, s_next);
                }
                else
                    for(int i = 0; i < 3; ++i)
                        s_next[i] = s1[i];
                const int n_regs = r0a.count + r0b.count;
                rocke_iter_arg_t xit[256];
                static char xnames[256][24];
                int n_it = 0;
                for(int i = 0; i < num_accs; ++i, ++n_it)
                {
                    xit[n_it].name = iter_args[i].name;
                    xit[n_it].init = iter_args[i].init;
                }
                for(int i = 0; i < n_regs; ++i, ++n_it)
                {
                    snprintf(xnames[i], 24, "xp_r%d", i);
                    xit[n_it].name = xnames[i];
                    xit[n_it].init = i < r0a.count ? r0a.vecs[i].v : r0b.vecs[i - r0a.count].v;
                }
                for(int i = 0; i < n_st; ++i, ++n_it)
                {
                    xit[n_it].name = st_names[i];
                    xit[n_it].init = s_next[i];
                }
                /* one tile of the loop: next loads, this tile's MFMAs, store */
                auto xh_body = [&](rocke_value_t* const* acc_in,
                                   rocke_value_t* const* regs,
                                   rocke_value_t* const* st,
                                   rocke_value_t** acc_out,
                                   rocke_value_t** carry_out,
                                   rocke_value_t** st_out) {
                    set_tile(st[0], st[1], is_3d ? st[2] : NULL);
                    if(spec->xh_order & 2)
                        rocke_b_s_waitcnt(b, -1, 0, -1);
                    const int sprio = spec->tapk_setprio;
                    if(sprio)
                    {
                        rocke_b_sched_barrier(b, 0);
                        rocke_b_s_setprio(b, 1);
                    }
                    rocke_ctl_staged_t sa, sb;
                    load_tile(&sa, &sb);
                    const int ilv = spec->tapk_ilv;
                    const int lda = spec->tapk_lds_ahead;
                    if(!ilv && !lda && sprio != 2)
                        rocke_b_sched_barrier(b, 0);
                    mfma(acc_in, acc_out);
                    if(sprio == 2)
                    {
                        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tapk_setprio 2 is Python-only");
                        return;
                    }
                    xh_schedule(sa.count + sb.count);
                    if(sprio)
                    {
                        rocke_b_sched_barrier(b, 0);
                        rocke_b_s_setprio(b, 0);
                    }
                    lds_bar();
                    /* rebuild(staged, regs) */
                    rocke_ctl_staged_t ra = sa, rb = sb;
                    if(n_regs)
                    {
                        for(int i = 0; i < ra.count; ++i)
                            ra.vecs[i].v = regs[i];
                        for(int i = 0; i < rb.count; ++i)
                            rb.vecs[i].v = regs[ra.count + i];
                    }
                    xh_store_deferred(&ra, &rb, acc_out);
                    lds_bar();
                    rocke_value_t* nx[3] = {};
                    advance(st, nx);
                    for(int i = 0; i < 3; ++i)
                        st_out[i] = nx[i];
                    if(n_regs)
                    {
                        for(int i = 0; i < sa.count; ++i)
                            carry_out[i] = sa.vecs[i].v;
                        for(int i = 0; i < sb.count; ++i)
                            carry_out[sa.count + i] = sb.vecs[i].v;
                    }
                };
                const bool unroll2 = spec->xh_order & 4;
                rocke_value_t* n_iter
                    = unroll2 ? rocke_b_div(b, n_tiles, rocke_b_const_i32(b, 2)) : n_tiles;
                rocke_for_t loop
                    = rocke_b_scf_for_iter(b, c0, n_iter, c1, xit, n_it, "t", false, true);
                auto run_body = [&](rocke_for_t& lp) {
                    rocke_value_t** iv = lp.iter_vars;
                    rocke_value_t* a_[ROCKE_CONV_MAX_ACCS];
                    rocke_value_t* r_[256];
                    rocke_value_t* s_[3];
                    rocke_value_t* st_in[3] = {iv[num_accs + n_regs],
                                               iv[num_accs + n_regs + 1],
                                               is_3d ? iv[num_accs + n_regs + 2] : NULL};
                    xh_body(iv, iv + num_accs, st_in, a_, r_, s_);
                    if(unroll2 && &lp == &loop)
                    {
                        rocke_value_t* a2[ROCKE_CONV_MAX_ACCS];
                        rocke_value_t* r2[256];
                        rocke_value_t* s2[3];
                        xh_body(a_, r_, s_, a2, r2, s2);
                        for(int i = 0; i < num_accs; ++i)
                            a_[i] = a2[i];
                        for(int i = 0; i < n_regs; ++i)
                            r_[i] = r2[i];
                        for(int i = 0; i < 3; ++i)
                            s_[i] = s2[i];
                    }
                    rocke_value_t* y[256];
                    int ny = 0;
                    for(int i = 0; i < num_accs; ++i)
                        y[ny++] = a_[i];
                    for(int i = 0; i < n_regs; ++i)
                        y[ny++] = r_[i];
                    for(int i = 0; i < n_st; ++i)
                        y[ny++] = s_[i];
                    rocke_b_scf_yield(b, y, ny);
                };
                rocke_b_region_enter(b, loop.body);
                run_body(loop);
                rocke_b_region_leave(b);
                if(!unroll2)
                {
                    for(int i = 0; i < num_accs; i++)
                        final_accs[i] = loop.op->results[i];
                    num_final = num_accs;
                }
                else
                {
                    /* the odd last tile */
                    rocke_iter_arg_t tit[256];
                    static char tnames[256][32];
                    for(int i = 0; i < n_it; ++i)
                    {
                        snprintf(tnames[i], 32, "%s_t1", xit[i].name);
                        tit[i].name = tnames[i];
                        tit[i].init = loop.op->results[i];
                    }
                    rocke_value_t* n2 = rocke_b_mul(b, n_iter, rocke_b_const_i32(b, 2));
                    rocke_value_t* rem = rocke_b_sub(b, n_tiles, n2);
                    rocke_for_t tail
                        = rocke_b_scf_for_iter(b, c0, rem, c1, tit, n_it, "t1", false, true);
                    rocke_b_region_enter(b, tail.body);
                    run_body(tail);
                    rocke_b_region_leave(b);
                    for(int i = 0; i < num_accs; i++)
                        final_accs[i] = tail.op->results[i];
                    num_final = num_accs;
                }
            } /* _xh_plain */
            if(yr)
            {
                /* this branch's epilogue */
                rocke_value_t* yr_epi[ROCKE_CONV_MAX_ACCS];
                rocke_conv_apply_accumulator_epilogue(
                    b, &spec->acc_epilogue, final_accs, num_final, yr_epi);
                dispatch_dgrad_epilogue(yr_epi, num_final);
                rocke_b_region_leave(b);
            }
        } /* yr passes */
        if(yr)
        {
            A_smem = A_smem_keep;
            B_smem = B_smem_keep;
            for(int x = 0; x < 8; ++x)
                B_smem_x[x] = B_smem_x_keep[x];
            xh_yr_done = true;
        }
    }
    else if(spec->tap_outer_k && spec->tapk_prefetch)
    {
        /* Python _emit_tap_outer_prefetch_loop (no ping-pong / boxes / x-halo):
         * the (tap, chunk) nest as one flat loop over its tiles, carrying the
         * next tile's (kb, x, y[, z]). */
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* c_kd = c_kdiv;
        rocke_value_t* c_cs = c_cstride;
        rocke_value_t* c_yx = rocke_b_mul(b, c_Y, c_X);
        rocke_value_t* c_taps = is_3d ? rocke_b_mul(b, ctx.p_Z, c_yx) : c_yx;
        rocke_value_t* c_yxc = rocke_b_mul(b, c_taps, c_cs);
        rocke_value_t* c_wo_k = rocke_b_mul(b, c_Wo, c_K);
        rocke_value_t* c_ho_wo_k = rocke_b_mul(b, c_Ho, c_wo_k);
        rocke_value_t* kdb = rocke_b_div(b, c_kd, c_block_k);
        rocke_value_t* n_tiles = rocke_b_mul(b, c_taps, kdb);
        /* pp_boxes: the record's box runs filter rows [yb, yb + ys) and
         * columns [xb, xb + xs) only (and depths [zb, zb + zs) in 3-D). */
        rocke_value_t* c_Xl = c_X;
        rocke_value_t* c_Yl = c_Y;
        rocke_value_t *bx_zb = NULL, *bx_zs = NULL;
        if(spec->pp_boxes)
        {
            rocke_value_t* bt = rocke_b_mul(b, rec_y_dot_slice, rec_x_dot_slice);
            if(is_3d)
            {
                bx_zb = rec_b_y_stride;
                bx_zs = _ld(21);
                bt = rocke_b_mul(b, bx_zs, bt);
            }
            rocke_value_t* kdb2 = rocke_b_div(b, c_kd, c_block_k);
            n_tiles = rocke_b_mul(b, bt, kdb2);
            c_Xl = rec_x_dot_slice;
            c_Yl = rec_y_dot_slice;
        }
        tpf_ctx_t tc;
        memset(&tc, 0, sizeof(tc));
        tc.boxes = spec->pp_boxes != 0;
        tc.bx_hb = rec_h_tilde_slice_begin;
        tc.bx_wb = rec_w_tilde_slice_begin;
        tc.bx_ds = rec_a_embed_h_coeff;
        tc.bx_db = rec_a_embed_w_coeff;
        tc.rm_base = rm_base;
        tc.rm_buf = sub_gemm_buf;
        tc.block_m_off = block_m_off_v;
        tc.block_n_off = block_n_off_v;
        tc.hw_div = hw_div;
        tc.w_div = w_div;
        tc.di_div = {ctx.p_Di, ctx.p_magic_m_Di_mult, ctx.p_magic_m_Di_shift};
        tc.is_3d = is_3d;
        tc.c0 = c0;
        tc.c_Ho = c_Ho;
        tc.c_Wo = c_Wo;
        tc.c_K = c_K;
        tc.p_Do = ctx.p_Do;
        tc.c_yxc = c_yxc;
        auto set_tile = [&](rocke_value_t* kb0,
                            rocke_value_t* xd,
                            rocke_value_t* yd,
                            rocke_value_t* zd,
                            rocke_value_t* live) {
            if(spec->pp_boxes)
            {
                xd = rocke_b_add(b, rec_b_x_offset, xd);
                yd = rocke_b_add(b, rec_b_y_offset, yd);
                if(is_3d)
                    zd = rocke_b_add(b, bx_zb, zd);
            }
            tc.dh = rocke_b_sub(b, ctx.p_pH, yd);
            tc.dw = rocke_b_sub(b, ctx.p_pW, xd);
            rocke_value_t* yx_ = rocke_b_mul(b, yd, c_X);
            rocke_value_t* tap = rocke_b_add(b, yx_, xd);
            rocke_value_t* t0 = rocke_b_mul(b, tc.dh, c_wo_k);
            rocke_value_t* t1 = rocke_b_mul(b, tc.dw, c_K);
            rocke_value_t* dy_tap = rocke_b_add(b, t0, t1);
            if(is_3d)
            {
                tc.dd = rocke_b_sub(b, ctx.p_pD, zd);
                rocke_value_t* zt = rocke_b_mul(b, zd, c_yx);
                tap = rocke_b_add(b, zt, tap);
                rocke_value_t* dt = rocke_b_mul(b, tc.dd, c_ho_wo_k);
                dy_tap = rocke_b_add(b, dt, dy_tap);
            }
            rocke_value_t* kb = grouped ? rocke_b_add(b, kb0, k_out_group_base) : kb0;
            tc.a_off = rocke_b_add(b, dy_tap, kb);
            rocke_value_t* kw = rocke_b_mul(b, kb, c_yxc);
            rocke_value_t* tw_ = rocke_b_mul(b, tap, c_cs);
            tc.w_k = rocke_b_add(b, kw, tw_);
            tc.live = live;
            tc.tap = tap;
        };
        tc.pp = spec->tapk_pingpong > 0;
        tc.pp_epi = spec->tapk_pp_epi > 0;
        /* _pp_swz: the 16-byte chunk XOR the row (tapk_pingpong + lds_xor). */
        auto pp_swz = [&](rocke_value_t* row, rocke_value_t* col) -> rocke_value_t* {
            if(!spec->lds_xor)
                return col;
            rocke_value_t* r = block_k == 32 ? rocke_b_lshr(b, row, rocke_b_const_i32(b, 1)) : row;
            if(spec->tapk_pingpong >= 3)
            {
                rocke_value_t* r8 = rocke_b_lshr(b, row, rocke_b_const_i32(b, 3));
                r = rocke_b_xor(b, r, r8);
            }
            rocke_value_t* key = rocke_b_land(b, r, rocke_b_const_i32(b, block_k / 8 - 1));
            rocke_value_t* ks = rocke_b_shl(b, key, rocke_b_const_i32(b, 3));
            return rocke_b_xor(b, col, ks);
        };
        /* _pbs_perm: tapk_pp_bslab 3, the slab row of B row n (bit 1 ^= bit 4). */
        auto pbs_perm = [&](rocke_value_t* n) -> rocke_value_t* {
            if(spec->tapk_pp_bslab != 3)
                return n;
            rocke_value_t* t = rocke_b_lshr(b, n, rocke_b_const_i32(b, 4));
            rocke_value_t* t1 = rocke_b_land(b, t, rocke_b_const_i32(b, 1));
            rocke_value_t* t2 = rocke_b_shl(b, t1, rocke_b_const_i32(b, 1));
            return rocke_b_xor(b, n, t2);
        };
        auto pp_store = [&](const rocke_coalesced_tile_loader_t* L,
                            rocke_value_t* dst,
                            const rocke_ctl_staged_t* st) {
            if(!spec->lds_xor)
            {
                rocke_coalesced_tile_loader_store_lds(b, L, dst, st);
                return;
            }
            const rocke_type_t* et = L->elem_dtype ? L->elem_dtype : rocke_f16();
            if(L->row_kpack > 1 && spec->tapk_pp_bslab)
            {
                const int kp = L->row_kpack;
                rocke_value_t* c_n = rocke_b_const_i32(b, block_n);
                for(int g = 0; g < st->count; g += kp)
                {
                    rocke_value_t* row = st->vecs[g].row;
                    rocke_value_t* col = st->vecs[g].col;
                    rocke_value_t* cs = rocke_b_lshr(b, col, rocke_b_const_i32(b, 2));
                    rocke_value_t* kgn = rocke_b_mul(b, cs, c_n);
                    rocke_value_t* base = rocke_b_add(b, kgn, row);
                    for(int j = 0; j < L->load_vec / 2; ++j)
                    {
                        rocke_value_t* r;
                        if(spec->tapk_pp_bslab == 3)
                        {
                            r = j ? rocke_b_add(b, row, rocke_b_const_i32(b, 2 * j)) : row;
                            rocke_value_t* pr = pbs_perm(r);
                            r = rocke_b_add(b, kgn, pr);
                        }
                        else
                            r = j ? rocke_b_add(b, base, rocke_b_const_i32(b, 2 * j)) : base;
                        rocke_value_t* parts[32];
                        int np = 0;
                        for(int q = 0; q < kp; ++q)
                            parts[np++] = rocke_b_vec_extract(b, st->vecs[g + q].v, 2 * j);
                        for(int q = 0; q < kp; ++q)
                            parts[np++] = rocke_b_vec_extract(b, st->vecs[g + q].v, 2 * j + 1);
                        /* [r, const 0] is evaluated before the pack (Python order). */
                        rocke_value_t* cz0 = rocke_b_const_i32(b, 0);
                        rocke_value_t* pk = rocke_b_vec_pack(b, parts, np, et);
                        rocke_value_t* idx[2] = {r, cz0};
                        rocke_b_smem_store_vN(b, dst, idx, 2, pk, 8);
                    }
                }
                return;
            }
            if(L->row_kpack > 1)
            {
                const int kp = L->row_kpack;
                for(int g = 0; g < st->count; g += kp)
                {
                    rocke_value_t* row = st->vecs[g].row;
                    rocke_value_t* col = st->vecs[g].col;
                    for(int i = 0; i < L->load_vec; ++i)
                    {
                        rocke_value_t* r = i ? rocke_b_add(b, row, rocke_b_const_i32(b, i)) : row;
                        rocke_value_t* parts[16];
                        for(int q = 0; q < kp; ++q)
                            parts[q] = rocke_b_vec_extract(b, st->vecs[g + q].v, i);
                        rocke_value_t* pk = rocke_b_vec_pack(b, parts, kp, et);
                        rocke_value_t* sc = pp_swz(r, col);
                        rocke_value_t* idx[2] = {r, sc};
                        rocke_b_smem_store_vN(b, dst, idx, 2, pk, kp);
                    }
                }
                return;
            }
            for(int e = 0; e < st->count; ++e)
            {
                rocke_value_t* sc = pp_swz(st->vecs[e].row, st->vecs[e].col);
                rocke_value_t* idx[2] = {st->vecs[e].row, sc};
                rocke_b_smem_store_vN(b, dst, idx, 2, st->vecs[e].v, L->load_vec);
            }
        };
        auto advance = [&](rocke_value_t* const* st, rocke_value_t** out) {
            rocke_value_t* nk = rocke_b_add(b, st[0], c_block_k);
            rocke_value_t* wrap_k = rocke_b_cmp_ge(b, nk, c_kd);
            nk = rocke_b_select(b, wrap_k, c0, nk);
            rocke_value_t* sx = rocke_b_select(b, wrap_k, c1, c0);
            rocke_value_t* nx = rocke_b_add(b, st[1], sx);
            rocke_value_t* wrap_x = rocke_b_cmp_ge(b, nx, c_Xl);
            nx = rocke_b_select(b, wrap_x, c0, nx);
            rocke_value_t* sy = rocke_b_select(b, wrap_x, c1, c0);
            rocke_value_t* ny = rocke_b_add(b, st[2], sy);
            out[0] = nk;
            out[1] = nx;
            if(!is_3d)
            {
                out[2] = ny;
                return;
            }
            rocke_value_t* wrap_y = rocke_b_cmp_ge(b, ny, c_Yl);
            ny = rocke_b_select(b, wrap_y, c0, ny);
            rocke_value_t* sz = rocke_b_select(b, wrap_y, c1, c0);
            out[2] = ny;
            out[3] = rocke_b_add(b, st[3], sz);
        };
        auto load_tile = [&](rocke_ctl_staged_t* a_st, rocke_ctl_staged_t* b_st) {
            rocke_coalesced_tile_loader_load_global(
                b, &a_sync_loader, tid, _tpf_dy_descriptor, &tc, dy_rsrc, NULL, a_st);
            _dg_b_load_global(b,
                              &b_sync_loader,
                              tid,
                              spec->lds_k_outer ? _tpf_w_descriptor_kouter : _tpf_w_descriptor,
                              &tc,
                              w_rsrc,
                              b_st);
        };
        auto store_tile = [&](const rocke_ctl_staged_t* a_st,
                              const rocke_ctl_staged_t* b_st,
                              rocke_value_t* a_dst,
                              rocke_value_t* b_dst) {
            if(spec->tapk_pingpong)
            {
                pp_store(&a_sync_loader, a_dst, a_st);
                pp_store(&b_sync_loader, b_dst, b_st);
                return;
            }
            rocke_coalesced_tile_loader_store_lds(b, &a_sync_loader, a_dst, a_st);
            rocke_coalesced_tile_loader_store_lds(b, &b_sync_loader, b_dst, b_st);
        };
        /* _pp_fast_a (tapk_pingpong >= 4): per-row pixel offsets and tap masks
         * built once; a tile's dY load is pixel + the tile's uniform offset,
         * bit 31 set from the tap's mask bit. */
        const int pf_lv = a_sync_loader.load_vec;
        rocke_value_t* fa_rows[ROCKE_CTL_MAX_VECS_PER_THREAD];
        rocke_value_t* fa_cols[ROCKE_CTL_MAX_VECS_PER_THREAD];
        rocke_value_t* fa_pix[ROCKE_CTL_MAX_VECS_PER_THREAD];
        rocke_value_t* fa_idle[ROCKE_CTL_MAX_VECS_PER_THREAD];
        rocke_value_t* fa_masks[ROCKE_CTL_MAX_VECS_PER_THREAD];
        int fa_n = 0;
        rocke_value_t *fa_c1 = NULL, *fa_csh = NULL, *fa_c31 = NULL;
        int fa_eb_sh = 0;
        bool have_fast_a = false;
        auto pp_fast_a = [&]() {
            const int lv = pf_lv;
            const int cpv = block_k / lv;
            rocke_value_t* c_cpv = rocke_b_const_i32(b, cpv);
            const rocke_type_t* et
                = a_sync_loader.elem_dtype ? a_sync_loader.elem_dtype : rocke_f16();
            fa_eb_sh = et == rocke_f32() ? 2 : 1;
            const int n_chunks = block_m * block_k / lv;
            rocke_value_t* crd_h[ROCKE_CTL_MAX_VECS_PER_THREAD];
            rocke_value_t* crd_w[ROCKE_CTL_MAX_VECS_PER_THREAD];
            rocke_value_t* crd_d[ROCKE_CTL_MAX_VECS_PER_THREAD];
            fa_n = 0;
            for(int e = 0; e < (n_chunks + threads - 1) / threads; ++e)
            {
                rocke_value_t* idx = rocke_b_add(b, rocke_b_const_i32(b, e * threads), tid);
                rocke_value_t* row = rocke_b_div(b, idx, c_cpv);
                rocke_value_t* rc = rocke_b_mul(b, row, c_cpv);
                rocke_value_t* rem = rocke_b_sub(b, idx, rc);
                rocke_value_t* col = rocke_b_mul(b, rem, rocke_b_const_i32(b, lv));
                fa_idle[e] = NULL;
                if((e + 1) * threads > n_chunks)
                {
                    rocke_value_t* off_ = rocke_b_cmp_ge(b, idx, rocke_b_const_i32(b, n_chunks));
                    rocke_value_t* bm = rocke_b_const_i32(b, block_m);
                    rocke_value_t* t7 = rocke_b_land(b, tid, rocke_b_const_i32(b, 7));
                    rocke_value_t* jr = rocke_b_add(b, bm, t7);
                    row = rocke_b_select(b, off_, jr, row);
                    rocke_value_t* t3 = rocke_b_lshr(b, tid, rocke_b_const_i32(b, 3));
                    rocke_value_t* tm = rocke_b_land(b, t3, rocke_b_const_i32(b, cpv - 1));
                    rocke_value_t* jc = rocke_b_mul(b, tm, rocke_b_const_i32(b, lv));
                    col = rocke_b_select(b, off_, jc, col);
                    fa_idle[e] = off_;
                }
                rocke_value_t* m_sub = rocke_b_add(b, block_m_off_v, row);
                if(rm_base != NULL)
                {
                    rocke_value_t* ra = rocke_b_add(b, rm_base, m_sub);
                    m_sub = rocke_b_global_load_i32(b, sub_gemm_buf, ra, 4);
                }
                rocke_value_t *n_val, *m_rem, *hi, *wi;
                _magic_divmod(b, m_sub, &hw_div, &n_val, &m_rem);
                _magic_divmod(b, m_rem, &w_div, &hi, &wi);
                if(spec->pp_boxes)
                {
                    hi = rocke_b_add(b, hi, rec_h_tilde_slice_begin);
                    wi = rocke_b_add(b, wi, rec_w_tilde_slice_begin);
                }
                rocke_value_t* di = NULL;
                if(is_3d)
                {
                    rocke_value_t* nq;
                    if(spec->pp_boxes)
                    {
                        nq = rocke_b_div(b, n_val, rec_a_embed_h_coeff);
                        rocke_value_t* nd_ = rocke_b_mul(b, nq, rec_a_embed_h_coeff);
                        rocke_value_t* r_ = rocke_b_sub(b, n_val, nd_);
                        di = rocke_b_add(b, r_, rec_a_embed_w_coeff);
                    }
                    else
                        _magic_divmod(b, n_val, &tc.di_div, &nq, &di);
                    rocke_value_t* nd = rocke_b_mul(b, nq, ctx.p_Do);
                    n_val = rocke_b_add(b, nd, di);
                }
                rocke_value_t* n_ho = rocke_b_mul(b, n_val, c_Ho);
                rocke_value_t* n_ho_hi = rocke_b_add(b, n_ho, hi);
                rocke_value_t* row_off = rocke_b_mul(b, n_ho_hi, c_Wo);
                rocke_value_t* pix_w = rocke_b_add(b, row_off, wi);
                rocke_value_t* pix_k = rocke_b_mul(b, pix_w, c_K);
                fa_pix[e] = rocke_b_add(b, pix_k, col);
                fa_rows[e] = row;
                fa_cols[e] = col;
                crd_h[e] = hi;
                crd_w[e] = wi;
                crd_d[e] = di;
                fa_n++;
            }
            fa_c1 = rocke_b_const_i32(b, 1);
            const int F = 3; /* _PP_MAX_FILTER */
            auto axis_bits = [&](rocke_value_t* coord,
                                 rocke_value_t* pad,
                                 rocke_value_t* out,
                                 rocke_value_t* ext) -> rocke_value_t* {
                rocke_value_t* hi_ = rocke_b_add(b, coord, pad);
                rocke_value_t* d0 = rocke_b_sub(b, hi_, out);
                rocke_value_t* lo_ = rocke_b_add(b, d0, fa_c1);
                rocke_value_t* l0 = rocke_b_smax(b, lo_, c0);
                rocke_value_t* lo_c = rocke_b_smin(b, l0, ext);
                rocke_value_t* e1 = rocke_b_sub(b, ext, fa_c1);
                rocke_value_t* hi_c = rocke_b_smin(b, hi_, e1);
                rocke_value_t* w0 = rocke_b_sub(b, hi_c, lo_c);
                rocke_value_t* w1 = rocke_b_add(b, w0, fa_c1);
                rocke_value_t* width = rocke_b_smax(b, w1, c0);
                rocke_value_t* s1 = rocke_b_shl(b, fa_c1, width);
                rocke_value_t* s2 = rocke_b_sub(b, s1, fa_c1);
                return rocke_b_shl(b, s2, lo_c);
            };
            const int nz = is_3d ? F : 1;
            for(int e = 0; e < fa_n; ++e)
            {
                rocke_value_t* vx = axis_bits(crd_w[e], ctx.p_pW, c_Wo, c_X);
                rocke_value_t* vy = axis_bits(crd_h[e], ctx.p_pH, c_Ho, c_Y);
                rocke_value_t* vz
                    = is_3d ? axis_bits(crd_d[e], ctx.p_pD, ctx.p_Do, ctx.p_Z) : fa_c1;
                rocke_value_t* valid = c0;
                for(int z = 0; z < nz; ++z)
                    for(int y = 0; y < F; ++y)
                    {
                        rocke_value_t* ok_u = rocke_b_cmp_lt(b, rocke_b_const_i32(b, y), c_Y);
                        if(is_3d)
                        {
                            rocke_value_t* zc = rocke_b_cmp_lt(b, rocke_b_const_i32(b, z), ctx.p_Z);
                            ok_u = rocke_b_land(b, ok_u, zc);
                        }
                        rocke_value_t* zcY = rocke_b_mul(b, rocke_b_const_i32(b, z), c_Y);
                        rocke_value_t* zy = rocke_b_add(b, zcY, rocke_b_const_i32(b, y));
                        rocke_value_t* sh = rocke_b_mul(b, zy, c_X);
                        sh = rocke_b_select(b, ok_u, sh, c0);
                        rocke_value_t* vys = rocke_b_lshr(b, vy, rocke_b_const_i32(b, y));
                        rocke_value_t* bit = rocke_b_land(b, vys, fa_c1);
                        if(is_3d)
                        {
                            rocke_value_t* vzs = rocke_b_lshr(b, vz, rocke_b_const_i32(b, z));
                            bit = rocke_b_land(b, bit, vzs);
                        }
                        rocke_value_t* bv = rocke_b_mul(b, bit, vx);
                        rocke_value_t* part = rocke_b_select(b, ok_u, bv, c0);
                        rocke_value_t* ps = rocke_b_shl(b, part, sh);
                        valid = rocke_b_lor(b, valid, ps);
                    }
                fa_masks[e] = rocke_b_xor(b, valid, rocke_b_const_i32(b, -1));
            }
            fa_csh = rocke_b_const_i32(b, fa_eb_sh);
            fa_c31 = rocke_b_const_i32(b, 31);
            have_fast_a = true;
        };
        auto load_a_fast = [&](rocke_ctl_staged_t* a_st) {
            a_st->count = 0;
            for(int e = 0; e < fa_n; ++e)
            {
                rocke_value_t* off = rocke_b_add(b, fa_pix[e], tc.a_off);
                off = fa_eb_sh ? rocke_b_shl(b, off, fa_csh) : off;
                rocke_value_t* ms = rocke_b_lshr(b, fa_masks[e], tc.tap);
                rocke_value_t* bad = rocke_b_land(b, ms, fa_c1);
                if(fa_idle[e] != NULL)
                {
                    rocke_value_t* sel = rocke_b_select(b, fa_idle[e], fa_c1, c0);
                    bad = rocke_b_lor(b, bad, sel);
                }
                if(tc.live != NULL)
                {
                    rocke_value_t* sel = rocke_b_select(b, tc.live, c0, fa_c1);
                    bad = rocke_b_lor(b, bad, sel);
                }
                rocke_value_t* bs = rocke_b_shl(b, bad, fa_c31);
                off = rocke_b_lor(b, off, bs);
                const rocke_type_t* et
                    = a_sync_loader.elem_dtype ? a_sync_loader.elem_dtype : rocke_f16();
                rocke_value_t* v = rocke_b_buffer_load_vN(b, dy_rsrc, off, c0, et, pf_lv);
                a_st->vecs[a_st->count].row = fa_rows[e];
                a_st->vecs[a_st->count].col = fa_cols[e];
                a_st->vecs[a_st->count].v = v;
                a_st->count++;
            }
        };
        const int n_st = is_3d ? 4 : 3;
        const char* st_names[4] = {"pf_kb", "pf_x", "pf_y", "pf_z"};
        rocke_value_t* s0[4] = {c0, c0, c0, is_3d ? c0 : NULL};
        if(spec->tapk_pingpong && spec->split_k > 1 && spec->pp_boxes)
        {
            /* pp_boxes split-K: the record's slice of the box's K tiles. */
            rocke_value_t* sk1 = rocke_b_sub(b, bx_sk, c1);
            rocke_value_t* nt1 = rocke_b_add(b, n_tiles, sk1);
            rocke_value_t* per = rocke_b_div(b, nt1, bx_sk);
            rocke_value_t* t0 = rocke_b_mul(b, bx_slice, per);
            rocke_value_t* tp = rocke_b_add(b, t0, per);
            rocke_value_t* tm = rocke_b_smin(b, tp, n_tiles);
            rocke_value_t* cnt = rocke_b_sub(b, tm, t0);
            n_tiles = rocke_b_smax(b, cnt, c0);
            /* _decode_t0 */
            rocke_value_t* nkc = rocke_b_div(b, c_kd, c_block_k);
            rocke_value_t* tap0 = rocke_b_div(b, t0, nkc);
            rocke_value_t* tn = rocke_b_mul(b, tap0, nkc);
            rocke_value_t* tr = rocke_b_sub(b, t0, tn);
            s0[0] = rocke_b_mul(b, tr, c_block_k);
            rocke_value_t* yz0 = rocke_b_div(b, tap0, c_Xl);
            rocke_value_t* yx0 = rocke_b_mul(b, yz0, c_Xl);
            s0[1] = rocke_b_sub(b, tap0, yx0);
            if(is_3d)
            {
                rocke_value_t* z0_ = rocke_b_div(b, yz0, c_Yl);
                rocke_value_t* zy0 = rocke_b_mul(b, z0_, c_Yl);
                s0[2] = rocke_b_sub(b, yz0, zy0);
                s0[3] = z0_;
            }
            else
                s0[2] = yz0;
        }
        set_tile(s0[0], s0[1], s0[2], s0[3], NULL);
        {
            rocke_ctl_staged_t a_st, b_st;
            load_tile(&a_st, &b_st);
            /* tapk_pingpong >= 4: the tap masks are built while tile 0's
             * loads are in flight. */
            if(spec->tapk_pingpong >= 4)
                pp_fast_a();
            store_tile(&a_st, &b_st, A_smem, B_smem);
        }
        rocke_b_sync(b);
        rocke_value_t* nxt0[4] = {};
        advance(s0, nxt0);
        static char pf_names[256][24];
        rocke_iter_arg_t pit[256];
        int n_it = 0;
        for(int i = 0; i < num_accs; ++i, ++n_it)
        {
            pit[n_it].name = iter_args[i].name;
            pit[n_it].init = iter_args[i].init;
        }
        if(spec->tapk_prefetch == 3)
        {
            set_tile(nxt0[0], nxt0[1], nxt0[2], is_3d ? nxt0[3] : NULL, NULL);
            rocke_ctl_staged_t a1, b1;
            load_tile(&a1, &b1);
            rocke_value_t* nxt1[4] = {};
            advance(nxt0, nxt1);
            const int n_regs = a1.count + b1.count;
            for(int i = 0; i < n_regs; ++i, ++n_it)
            {
                snprintf(pf_names[i], 24, "pf_r%d", i);
                pit[n_it].name = pf_names[i];
                pit[n_it].init = i < a1.count ? a1.vecs[i].v : b1.vecs[i - a1.count].v;
            }
            for(int i = 0; i < n_st; ++i, ++n_it)
            {
                pit[n_it].name = st_names[i];
                pit[n_it].init = nxt1[i];
            }
            rocke_for_t loop
                = rocke_b_scf_for_iter(b, c0, n_tiles, c1, pit, n_it, "t", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                rocke_value_t** iv = loop.iter_vars;
                rocke_value_t* const* regs = iv + num_accs;
                rocke_value_t* const* st = iv + num_accs + n_regs;
                set_tile(st[0], st[1], st[2], is_3d ? st[3] : NULL, NULL);
                rocke_ctl_staged_t a2, b2;
                load_tile(&a2, &b2);
                rocke_b_sched_barrier(b, 0);
                rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
                emit_mfma_phase(iv, new_accs);
                rocke_b_sync(b);
                rocke_ctl_staged_t ra = a2, rb = b2;
                for(int i = 0; i < ra.count; ++i)
                    ra.vecs[i].v = regs[i];
                for(int i = 0; i < rb.count; ++i)
                    rb.vecs[i].v = regs[ra.count + i];
                store_tile(&ra, &rb, A_smem, B_smem);
                rocke_b_sync(b);
                rocke_value_t* nxt[4] = {};
                advance(st, nxt);
                rocke_value_t* y[256];
                int ny = 0;
                for(int i = 0; i < num_accs; ++i)
                    y[ny++] = new_accs[i];
                for(int i = 0; i < a2.count; ++i)
                    y[ny++] = a2.vecs[i].v;
                for(int i = 0; i < b2.count; ++i)
                    y[ny++] = b2.vecs[i].v;
                for(int i = 0; i < n_st; ++i)
                    y[ny++] = nxt[i];
                rocke_b_scf_yield(b, y, ny);
            }
            rocke_b_region_leave(b);
            for(int i = 0; i < num_accs; i++)
                final_accs[i] = loop.op->results[i];
            num_final = num_accs;
        }
        else if(spec->tapk_pingpong)
        {
            /* Python _tapk_pingpong_loop -> _pp_two_ahead (tapk_pp_sched > 0). */
            rocke_value_t* pc1 = rocke_b_const_i32(b, 1);
            const int n_w = spec->warp_m * spec->warp_n;
            rocke_lane_decode_t dec = rocke_decode_mfma_lanes(b, atom, lane);
            rocke_value_t* wmo = rocke_warp_grid_warp_m_off(b, &grid);
            rocke_value_t* wno = rocke_warp_grid_warp_n_off(b, &grid);
            rocke_value_t* a_idx[ROCKE_CONV_MAX_ACCS];
            rocke_value_t* b_idx[ROCKE_CONV_MAX_ACCS];
            rocke_value_t* b_idx_slab[ROCKE_CONV_MAX_ACCS];
            for(int mi = 0; mi < mfmas_m; ++mi)
            {
                rocke_value_t* cm = rocke_b_const_i32(b, mi * spec->warp_tile_m);
                rocke_value_t* t = rocke_b_add(b, cm, dec.m_in_atom);
                a_idx[mi] = rocke_b_add(b, wmo, t);
            }
            for(int ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* cn = rocke_b_const_i32(b, ni * spec->warp_tile_n);
                rocke_value_t* t = rocke_b_add(b, cn, dec.n_in_atom);
                b_idx[ni] = rocke_b_add(b, wno, t);
            }
            for(int ni = 0; ni < mfmas_n; ++ni)
                b_idx_slab[ni] = spec->tapk_pp_bslab ? pbs_perm(b_idx[ni]) : b_idx[ni];
            const int kp = spec->tapk_pingpong >= 2 ? 2 : 1;
            rocke_value_t* k_lane
                = rocke_b_mul(b, dec.k_blk, rocke_b_const_i32(b, kp * a_per_lane));
            const rocke_type_t* sdt = smem_dtype ? smem_dtype : rocke_f16();
            auto part = [&](rocke_value_t* v, int h, int n) -> rocke_value_t* {
                if(kp == 1)
                    return v;
                rocke_value_t* ps[16];
                for(int i = 0; i < n; ++i)
                    ps[i] = rocke_b_vec_extract(b, v, h * n + i);
                return rocke_b_vec_pack(b, ps, n, sdt);
            };
            /* frags[kk] = (a_rows, b_cols) of k atom kk */
            static rocke_value_t* fa_[16][ROCKE_CONV_MAX_ACCS];
            static rocke_value_t* fb_[16][ROCKE_CONV_MAX_ACCS];
            auto frag_reads = [&]() {
                int nf = 0;
                for(int pk = 0; pk < k_atoms / kp; ++pk)
                {
                    rocke_value_t* col
                        = rocke_b_add(b, k_lane, rocke_b_const_i32(b, pk * kp * spec->warp_tile_k));
                    rocke_value_t* a_w[ROCKE_CONV_MAX_ACCS];
                    rocke_value_t* b_w[ROCKE_CONV_MAX_ACCS];
                    for(int mi = 0; mi < mfmas_m; ++mi)
                    {
                        rocke_value_t* sc = pp_swz(a_idx[mi], col);
                        a_w[mi] = rocke_conv_emit_smem_load(
                            b, A_smem, a_idx[mi], sc, kp * a_per_lane, smem_dtype);
                    }
                    if(spec->tapk_pp_bslab)
                    {
                        rocke_value_t* c2s = rocke_b_lshr(b, col, rocke_b_const_i32(b, 2));
                        rocke_value_t* kg0 = rocke_b_mul(b, c2s, rocke_b_const_i32(b, block_n));
                        const int nrun = kp * b_per_lane / 4;
                        for(int ni = 0; ni < mfmas_n; ++ni)
                        {
                            rocke_value_t* r = b_idx_slab[ni];
                            rocke_value_t* runs[8];
                            for(int q = 0; q < nrun; ++q)
                            {
                                rocke_value_t* rr;
                                if(q)
                                {
                                    rocke_value_t* kq
                                        = rocke_b_add(b, kg0, rocke_b_const_i32(b, q * block_n));
                                    rr = rocke_b_add(b, kq, r);
                                }
                                else
                                    rr = rocke_b_add(b, kg0, r);
                                runs[q] = rocke_conv_emit_smem_load(
                                    b, B_smem, rr, rocke_b_const_i32(b, 0), 4, smem_dtype);
                            }
                            rocke_value_t* ps[32];
                            int np = 0;
                            for(int q = 0; q < nrun; ++q)
                                for(int i = 0; i < 4; ++i)
                                    ps[np++] = rocke_b_vec_extract(b, runs[q], i);
                            b_w[ni] = rocke_b_vec_pack(b, ps, np, sdt);
                        }
                    }
                    else
                        for(int ni = 0; ni < mfmas_n; ++ni)
                        {
                            rocke_value_t* sc = pp_swz(b_idx[ni], col);
                            b_w[ni] = rocke_conv_emit_smem_load(
                                b, B_smem, b_idx[ni], sc, kp * b_per_lane, smem_dtype);
                        }
                    for(int h = 0; h < kp; ++h)
                    {
                        for(int mi = 0; mi < mfmas_m; ++mi)
                            fa_[nf][mi] = part(a_w[mi], h, a_per_lane);
                        for(int ni = 0; ni < mfmas_n; ++ni)
                            fb_[nf][ni] = part(b_w[ni], h, b_per_lane);
                        nf++;
                    }
                }
            };
            auto mfmas = [&](int kk0, int kk1, rocke_value_t* const* acc_in, rocke_value_t** out) {
                for(int i = 0; i < num_accs; ++i)
                    out[i] = acc_in[i];
                for(int kk = kk0; kk < kk1; ++kk)
                {
                    int flat = 0;
                    for(int mi = 0; mi < mfmas_m; ++mi)
                        for(int ni = 0; ni < mfmas_n; ++ni, ++flat)
                            out[flat] = spec->tapk_pp_epi == 2
                                            ? rocke_conv_emit_mfma(
                                                  b, atom, fb_[kk][ni], fa_[kk][mi], out[flat])
                                            : rocke_conv_emit_mfma(
                                                  b, atom, fa_[kk][mi], fb_[kk][ni], out[flat]);
                }
            };
            rocke_value_t* wd = rocke_b_div(b, tid, rocke_b_const_i32(b, spec->wave_size));
            rocke_value_t* wave = rocke_b_readfirstlane(b, wd);
            rocke_value_t* upper = rocke_b_cmp_ge(b, wave, rocke_b_const_i32(b, n_w / 2));
            {
                rocke_if_t ifo = rocke_b_scf_if(b, upper);
                rocke_b_region_enter(b, ifo.then_region);
                rocke_b_sync_lds_only(b);
                rocke_b_region_leave(b);
            }
            const int half = k_atoms / 2;
            const int pps = spec->tapk_pp_sched;
            if(pps == 0 || pps == 3 || pps == 4 || pps == 6 || pps == 7 || pps == 11)
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "tapk_pp_sched %d is Python-only (no C++ twin yet)", pps);
                return NULL;
            }
            /* _pp_two_ahead: tile 1 rides in registers across the back edge. */
            set_tile(nxt0[0], nxt0[1], nxt0[2], is_3d ? nxt0[3] : NULL, NULL);
            rocke_ctl_staged_t s1a, s1b;
            load_tile(&s1a, &s1b);
            rocke_value_t* nxt1[4] = {};
            advance(nxt0, nxt1);
            const int n_regs = s1a.count + s1b.count;
            for(int i = 0; i < n_regs; ++i, ++n_it)
            {
                snprintf(pf_names[i], 24, "pp_r%d", i);
                pit[n_it].name = pf_names[i];
                pit[n_it].init = i < s1a.count ? s1a.vecs[i].v : s1b.vecs[i - s1a.count].v;
            }
            for(int i = 0; i < n_st; ++i, ++n_it)
            {
                pit[n_it].name = st_names[i];
                pit[n_it].init = nxt1[i];
            }
            rocke_for_t loop
                = rocke_b_scf_for_iter(b, c0, n_tiles, pc1, pit, n_it, "t", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                rocke_value_t** iv = loop.iter_vars;
                rocke_value_t* const* regs = iv + num_accs;
                rocke_value_t* st[4] = {iv[num_accs + n_regs],
                                        iv[num_accs + n_regs + 1],
                                        iv[num_accs + n_regs + 2],
                                        is_3d ? iv[num_accs + n_regs + 3] : NULL};
                auto ld_a = [&](rocke_ctl_staged_t* a_st) {
                    if(have_fast_a)
                        load_a_fast(a_st);
                    else
                        rocke_coalesced_tile_loader_load_global(
                            b, &a_sync_loader, tid, _tpf_dy_descriptor, &tc, dy_rsrc, NULL, a_st);
                };
                auto ld_b = [&](rocke_ctl_staged_t* b_st) {
                    _dg_b_load_global(b,
                                      &b_sync_loader,
                                      tid,
                                      spec->lds_k_outer ? _tpf_w_descriptor_kouter
                                                        : _tpf_w_descriptor,
                                      &tc,
                                      w_rsrc,
                                      b_st);
                };
                auto interleave = [&](int n_mf, int n_vm) {
                    const int first_vm = n_mf / 2;
                    int left = n_vm;
                    for(int j = 0; j < n_mf; ++j)
                    {
                        rocke_b_sched_group_barrier(b, 0x008, 1, 0);
                        rocke_b_sched_group_barrier(b, 0x002, 4, 0);
                        rocke_b_sched_group_barrier(b, 0x004, 2, 0);
                        if(j >= first_vm && left > 0 && (j - first_vm) % 2 == 1)
                        {
                            rocke_b_sched_group_barrier(b, 0x020, 1, 0);
                            left--;
                        }
                    }
                    if(left)
                        rocke_b_sched_group_barrier(b, 0x020, left, 0);
                };
                const int n_mf_half = half * mfmas_m * mfmas_n;
                /* [mem A] */
                frag_reads();
                rocke_b_sched_barrier(b, 0);
                if(spec->tapk_pingpong == 5)
                {
                    rocke_if_t ifo = rocke_b_scf_if(b, upper);
                    rocke_b_region_enter(b, ifo.then_region);
                    rocke_b_s_waitcnt(b, -1, 0, -1);
                    rocke_b_region_leave(b);
                    rocke_b_s_barrier_bare(b);
                }
                else
                    rocke_b_sync_lds_only(b);
                /* [compute A] */
                rocke_b_s_setprio(b, 1);
                if(spec->tapk_pp_epi)
                {
                    rocke_value_t* t2 = rocke_b_add(b, loop.iv, rocke_b_const_i32(b, 2));
                    rocke_value_t* lv2 = rocke_b_cmp_lt(b, t2, n_tiles);
                    set_tile(st[0], st[1], st[2], st[3], lv2);
                }
                else
                    set_tile(st[0], st[1], st[2], st[3], NULL);
                rocke_value_t* nxt[4] = {};
                advance(st, nxt);
                rocke_ctl_staged_t a2, b2;
                a2.count = b2.count = 0;
                if(pps == 12)
                    ld_b(&b2);
                rocke_value_t* acc_a[ROCKE_CONV_MAX_ACCS];
                mfmas(0, half, iv, acc_a);
                if(pps == 12)
                    interleave(n_mf_half, b2.count);
                else
                    for(int j = 0; j < n_mf_half; ++j)
                    {
                        rocke_b_sched_group_barrier(b, 0x008, 1, 0);
                        rocke_b_sched_group_barrier(b, 0x004, 3, 0);
                    }
                rocke_b_sched_barrier(b, 0);
                rocke_b_s_setprio(b, 0);
                rocke_b_sync_lds_only(b);
                /* [mem B]: store tile t+1 */
                {
                    rocke_ctl_staged_t ra = s1a, rb = s1b;
                    for(int i = 0; i < ra.count; ++i)
                        ra.vecs[i].v = regs[i];
                    for(int i = 0; i < rb.count; ++i)
                        rb.vecs[i].v = regs[ra.count + i];
                    store_tile(&ra, &rb, A_smem, B_smem);
                }
                if(pps == 1)
                {
                    ld_a(&a2);
                    ld_b(&b2);
                }
                if(pps == 5)
                    ld_a(&a2);
                rocke_b_sched_barrier(b, 0);
                rocke_b_sync_lds_only(b);
                /* [compute B] */
                rocke_b_s_setprio(b, 1);
                if(pps == 2)
                {
                    ld_a(&a2);
                    ld_b(&b2);
                }
                if(pps == 5)
                    ld_b(&b2);
                if(pps == 12)
                    ld_a(&a2);
                rocke_value_t* acc_b[ROCKE_CONV_MAX_ACCS];
                mfmas(half, k_atoms, acc_a, acc_b);
                if(pps == 2)
                    interleave(n_mf_half, a2.count + b2.count);
                if(pps == 5)
                    interleave(n_mf_half, b2.count);
                if(pps == 12)
                    interleave(n_mf_half, a2.count);
                rocke_b_sched_barrier(b, 0);
                rocke_b_s_setprio(b, 0);
                rocke_b_sync_lds_only(b);
                rocke_value_t* y[256];
                int ny = 0;
                for(int i = 0; i < num_accs; ++i)
                    y[ny++] = acc_b[i];
                for(int i = 0; i < a2.count; ++i)
                    y[ny++] = a2.vecs[i].v;
                for(int i = 0; i < b2.count; ++i)
                    y[ny++] = b2.vecs[i].v;
                for(int i = 0; i < n_st; ++i)
                    y[ny++] = nxt[i];
                rocke_b_scf_yield(b, y, ny);
            }
            rocke_b_region_leave(b);
            for(int i = 0; i < num_accs; i++)
                final_accs[i] = loop.op->results[i];
            num_final = num_accs;
            /* _pp_finish: with tapk_pp_epi the accumulators are converted to
             * bf16 before the lower half's matching barrier. */
            if(spec->tapk_pp_epi && rocke_conv_acc_epilogue_is_identity(&spec->acc_epilogue))
            {
                pp_bf16_n = num_accs;
                for(int i = 0; i < num_accs; ++i)
                    pp_bf16[i] = _pp_f32_to_bf16(b, spec, final_accs[i]);
            }
            {
                rocke_value_t* lw = rocke_b_cmp_lt(b, wave, rocke_b_const_i32(b, n_w / 2));
                rocke_if_t ifo = rocke_b_scf_if(b, lw);
                rocke_b_region_enter(b, ifo.then_region);
                rocke_b_sync_lds_only(b);
                rocke_b_region_leave(b);
            }
        }
        else if(spec->tapk_prefetch == 1)
        {
            for(int i = 0; i < n_st; ++i, ++n_it)
            {
                pit[n_it].name = st_names[i];
                pit[n_it].init = nxt0[i];
            }
            rocke_for_t loop
                = rocke_b_scf_for_iter(b, c0, n_tiles, c1, pit, n_it, "t", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                rocke_value_t** iv = loop.iter_vars;
                rocke_value_t* const* st = iv + num_accs;
                set_tile(st[0], st[1], st[2], is_3d ? st[3] : NULL, NULL);
                rocke_ctl_staged_t a2, b2;
                load_tile(&a2, &b2);
                rocke_b_sched_barrier(b, 0);
                rocke_value_t* new_accs[ROCKE_CONV_MAX_ACCS];
                emit_mfma_phase(iv, new_accs);
                rocke_b_sync(b);
                store_tile(&a2, &b2, A_smem, B_smem);
                rocke_b_sync(b);
                rocke_value_t* nxt[4] = {};
                advance(st, nxt);
                rocke_value_t* y[256];
                int ny = 0;
                for(int i = 0; i < num_accs; ++i)
                    y[ny++] = new_accs[i];
                for(int i = 0; i < n_st; ++i)
                    y[ny++] = nxt[i];
                rocke_b_scf_yield(b, y, ny);
            }
            rocke_b_region_leave(b);
            for(int i = 0; i < num_accs; i++)
                final_accs[i] = loop.op->results[i];
            num_final = num_accs;
        }
        else
        {
            /* tapk_prefetch == 2: ping-pong over two buffers, two tiles a body. */
            rocke_value_t* c2 = rocke_b_const_i32(b, 2);
            for(int i = 0; i < n_st; ++i, ++n_it)
            {
                pit[n_it].name = st_names[i];
                pit[n_it].init = nxt0[i];
            }
            rocke_for_t loop
                = rocke_b_scf_for_iter(b, c0, n_tiles, c2, pit, n_it, "t2", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                rocke_value_t** iv = loop.iter_vars;
                rocke_value_t* st[4] = {iv[num_accs],
                                        iv[num_accs + 1],
                                        iv[num_accs + 2],
                                        is_3d ? iv[num_accs + 3] : NULL};
                rocke_value_t* t1 = rocke_b_add(b, loop.iv, c1);
                rocke_value_t* live = rocke_b_cmp_lt(b, t1, n_tiles);
                set_tile(st[0], st[1], st[2], st[3], live);
                rocke_ctl_staged_t a2, b2;
                load_tile(&a2, &b2);
                rocke_b_sched_barrier(b, 0);
                rocke_value_t* accs_a[ROCKE_CONV_MAX_ACCS];
                emit_mfma_phase(iv, accs_a);
                rocke_b_sched_barrier(b, 0);
                store_tile(&a2, &b2, A_smem2, B_smem2);
                rocke_b_sync(b);
                rocke_value_t* st2[4] = {};
                advance(st, st2);
                set_tile(st2[0], st2[1], st2[2], is_3d ? st2[3] : NULL, NULL);
                load_tile(&a2, &b2);
                rocke_b_sched_barrier(b, 0);
                rocke_value_t* accs_b[ROCKE_CONV_MAX_ACCS];
                A_src_cur = A_smem2;
                emit_mfma_phase_src(accs_a, accs_b, B_smem2, NULL);
                A_src_cur = A_smem;
                rocke_b_sched_barrier(b, 0);
                store_tile(&a2, &b2, A_smem, B_smem);
                rocke_b_sync(b);
                rocke_value_t* nxt[4] = {};
                advance(st2, nxt);
                rocke_value_t* y[256];
                int ny = 0;
                for(int i = 0; i < num_accs; ++i)
                    y[ny++] = accs_b[i];
                for(int i = 0; i < n_st; ++i)
                    y[ny++] = nxt[i];
                rocke_b_scf_yield(b, y, ny);
            }
            rocke_b_region_leave(b);
            for(int i = 0; i < num_accs; i++)
                final_accs[i] = loop.op->results[i];
            num_final = num_accs;
        }
    }
    else if(spec->tap_outer_k)
    {
        /* Python _emit_tap_outer_loops: filter tap outer, output-channel
         * chunk inner. kpg % tile_k == 0 (validator), so a chunk never
         * straddles a tap. */
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* c_kd = c_kdiv;
        rocke_value_t* c_cs = c_cstride;
        rocke_value_t* c_yx = rocke_b_mul(b, c_Y, c_X);
        /* 3-D: taps (z, y, x), KZYXC rows of Z*Y*X*Cs. */
        rocke_value_t* c_taps = is_3d ? rocke_b_mul(b, ctx.p_Z, c_yx) : c_yx;
        rocke_value_t* c_yxc = rocke_b_mul(b, c_taps, c_cs);

        tap_dy_ctx_t tdy;
        tdy.is_3d = is_3d;
        tdy.di_div = {ctx.p_Di, ctx.p_magic_m_Di_mult, ctx.p_magic_m_Di_shift};
        tdy.p_Do = ctx.p_Do;
        tdy.dd = NULL;
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
            rocke_value_t* yx = tap;
            rocke_value_t* zdot = NULL;
            if(is_3d)
            {
                zdot = rocke_b_div(b, tap, c_yx);
                rocke_value_t* zyx = rocke_b_mul(b, zdot, c_yx);
                yx = rocke_b_sub(b, tap, zyx);
            }
            rocke_value_t* ydot = rocke_b_div(b, yx, c_X);
            rocke_value_t* ydot_x = rocke_b_mul(b, ydot, c_X);
            rocke_value_t* xdot = rocke_b_sub(b, yx, ydot_x);
            tdy.dh = rocke_b_sub(b, ctx.p_pH, ydot);
            tdy.dw = rocke_b_sub(b, ctx.p_pW, xdot);
            rocke_value_t* row_shift = tdy.dh;
            if(is_3d)
            {
                tdy.dd = rocke_b_sub(b, ctx.p_pD, zdot);
                rocke_value_t* dho = rocke_b_mul(b, tdy.dd, c_Ho);
                row_shift = rocke_b_add(b, dho, row_shift);
            }
            rocke_value_t* dh_wo = rocke_b_mul(b, row_shift, c_Wo);
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
                _dg_b_load_global(b,
                                  &b_sync_loader,
                                  tid,
                                  spec->lds_k_outer ? _tap_w_descriptor_kouter : _tap_w_descriptor,
                                  &tw,
                                  w_rsrc,
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

    if(xh_yr_done)
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
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
    char reason[512];

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

    // Validate. spec.validate() opens with the arch-free TF32 gate (the rest of
    // validate() is folded into is_valid_dgrad_spec), raised as its bare
    // ValueError before is_valid_dgrad_spec runs.
    if(_dgrad_mma_dtype_conflict(spec, NULL, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
        return NULL;
    }
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
