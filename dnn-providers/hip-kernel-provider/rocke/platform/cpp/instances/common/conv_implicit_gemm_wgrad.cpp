// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ port of the implicit-GEMM backward-weight convolution builder
 * (rocke/instances/common/conv_implicit_gemm_wgrad.py).
 *
 * GEMM orientation (wgrad):
 *   M     = K          (output channels — weight rows)
 *   N_wg  = Y*X*C      (filter spatial × input channel — weight cols)
 *   K_wg  = N*Ho*Wo    (output spatial positions — reduction)
 *
 * Operand roles:
 *   A  = dY  (NHWK, output gradient)  — the M-row / K_wg-reduction operand
 *   B  = X   (NHWC, input activations) — reuses the forward make_a_descriptor
 *   D  = dW  (KYXC, weight gradient)  — written at the end
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

#include <cstdio> /* snprintf */
#include <cstring> /* strcmp, memset, memcpy */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
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

/* wg_N = Z*Y*X*C  (filter spatial × input channel) */
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
     *     flags={"async": async_dma, "spk{N}": split_k>1, "spkauto": split_k==-1},
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

    /* flags: async, spk{N}, spkauto  — Python boolean flags */
    char spk_flag[32] = {0};
    const char* flag_names[3];
    int flag_on[3];
    int n_flags = 0;

    flag_names[n_flags] = "async";
    flag_on[n_flags] = s->async_dma ? 1 : 0;
    n_flags++;

    if(s->split_k > 1)
    {
        snprintf(spk_flag, sizeof(spk_flag), "spk%d", s->split_k);
        flag_names[n_flags] = spk_flag;
        flag_on[n_flags] = 1;
        n_flags++;
    }
    else if(s->split_k == -1)
    {
        flag_names[n_flags] = "spkauto";
        flag_on[n_flags] = 1;
        n_flags++;
    }

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
     * Mirror Python is_valid_wgrad_spec — geometry + block size + MMA atom +
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
 *   The epilogue queries D_desc with coord names ("m", "k_out") where:
 *     "m"     = output channel index (= K dimension of dW, wg_M = K)
 *     "k_out" = filter+channel index (= Y*X*C dimension, wg_N)
 *
 * Python original uses ("k_out", "n_wg") but we must use ("m", "k_out") to
 * match the forward epilogue's D_desc.offset(m=m_val, k_out=n_val) call.
 *
 * Layout: naive("dW_kyxc", [K,Y,X,C], coords=["m","y","x","c"]).transform(
 *           unmerge_magic("k_out" -> [y,x,c], [Y,X,C]), pad('y'), pad('x'))
 */
static rocke_tensor_descriptor_t* wgrad_make_dw_descriptor(rocke_ir_builder_t* b,
                                                           const rocke_conv_problem_t* p)
{
    if(p->is_3d)
    {
        /* naive coords: first dim "m" = K (output channels), rest are spatial */
        int lengths[5] = {p->K, p->Z, p->Y, p->X, p->C};
        const char* coords[5] = {"m", "z", "y", "x", "c"};
        rocke_tensor_descriptor_t* desc
            = rocke_tensor_descriptor_naive(b, "dW_kzyxc", lengths, 5, NULL, coords, 5);
        if(desc == NULL)
            return NULL;
        const char* into[4] = {"z", "y", "x", "c"};
        int dims[4] = {p->Z, p->Y, p->X, p->C};
        const rocke_transform_t* xforms[4];
        int n_x = 0;
        /* unmerge "k_out" (= n_wg, filter+channel N dimension) into spatial dims */
        xforms[n_x] = rocke_unmerge_magic(b, "k_out", into, 4, dims);
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

    /* 2-D: naive("dW_kyxc", [K,Y,X,C], coords=["m","y","x","c"]) */
    int lengths[4] = {p->K, p->Y, p->X, p->C};
    const char* coords[4] = {"m", "y", "x", "c"};
    rocke_tensor_descriptor_t* desc
        = rocke_tensor_descriptor_naive(b, "dW_kyxc", lengths, 4, NULL, coords, 4);
    if(desc == NULL)
        return NULL;
    const char* into[3] = {"y", "x", "c"};
    int dims[3] = {p->Y, p->X, p->C};
    const rocke_transform_t* xforms[4];
    int n_x = 0;
    xforms[n_x] = rocke_unmerge_magic(b, "k_out", into, 3, dims);
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

// (No custom closures needed: rocke_conv_a_descriptor and rocke_conv_b_descriptor
//  read ctx->A_desc / ctx->B_desc with coords ("m","k") and ("k_out","k_gemm")
//  respectively, which our wgrad descriptors are built to satisfy.)

// ---------------------------------------------------------------------------
// Build helper: build a forward-conv compatible spec from wgrad spec
// ---------------------------------------------------------------------------

/*
 * Map wgrad spec fields onto a forward ImplicitGemmConvSpec so we can call
 * rocke_conv_build_ctx_init.  The problem is adapted:
 *   - M_fwd  = wg_M (= K, output channels)  -> set as N*Ho*Wo by using N=wg_M, Ho=Wo=1
 *   - N_fwd  = wg_N (= Y*X*C)               -> set via K and a 1-kernel problem
 *   - K_gemm = wg_K (= N*Ho*Wo)             -> set via Y=wg_K/C, X=C, C=1... tricky
 *
 * Actually: the ctx_init uses tile_m/tile_n/tile_k, block_size, mfmas_m/n for
 * geometry, and the GEMM loop bounds come from ctx->c_K_gemm which is
 * p->K_gemm.  We need to supply a problem where K_gemm == wg_K.
 *
 * Simplest adapter: set the problem so that Y*X*C == wg_K and K == wg_N and
 * N*Ho*Wo == wg_M.  We override the descriptors afterwards so the actual
 * address logic doesn't matter — only the loop bound c_K_gemm is used from p.
 *
 * Because we override A_desc / B_desc after ctx_init, the only field the
 * k-loop drivers use from p is K_gemm (via c_K_gemm = const_i32(p.K_gemm)).
 * We also need wg_M and wg_N for bounds in the epilogue; we stash those in
 * the adapted problem as K (= N_gemm in the forward sense, used for epilogue
 * bounds) and M (used for epilogue bounds).
 *
 * Forward conv: epilogue bounds = (p.M, p.N_gemm) = (N*Ho*Wo, K).
 * Wgrad:        epilogue bounds = (wg_M, wg_N) = (K, Y*X*C).
 *
 * So we set: adapter_problem.K = wg_N, and adapter_problem.N*Ho*Wo = wg_M,
 * and Y*X*C (K_gemm) = wg_K.  Easy way: Y=wg_K, X=1, C=1, K=wg_N, N=wg_M, Hi=Wi=1.
 *
 * Then M = N*1*1 = wg_M, N_gemm = K = wg_N, K_gemm = Y*1*1 = wg_K.  Perfect.
 */
static rocke_conv_problem_t make_adapter_problem(const rocke_implicit_gemm_conv_wgrad_spec_t* s)
{
    int wg_M = rocke_wgrad_conv_spec_wg_M(s);
    int wg_N = rocke_wgrad_conv_spec_wg_N(s);
    int wg_K = rocke_wgrad_conv_spec_wg_K(s);
    /* N=wg_M, Hi=1, Wi=1, C=1, K=wg_N, Y=wg_K, X=1 */
    return rocke_conv_problem_default(wg_M, 1, 1, 1, wg_N, wg_K, 1);
}

static rocke_implicit_gemm_conv_spec_t
    make_adapter_fwd_spec(const rocke_implicit_gemm_conv_wgrad_spec_t* s,
                          const rocke_conv_problem_t* adapter_p)
{
    rocke_implicit_gemm_conv_spec_t fwd = rocke_implicit_gemm_conv_spec_default();
    fwd.problem = *adapter_p;
    fwd.name = s->name ? s->name : "conv_igemm_wgrad";
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
    fwd.dtype_acc = s->dtype_acc;
    /* lds_k_pad */
    fwd.has_lds_k_pad = s->has_lds_k_pad;
    fwd.lds_k_pad = s->lds_k_pad;
    /* waves_per_eu */
    fwd.has_waves_per_eu = s->has_waves_per_eu;
    fwd.waves_per_eu = s->waves_per_eu;
    /* chiplet swizzle */
    fwd.chiplet_swizzle = s->chiplet_swizzle;
    fwd.chiplet_wgm = s->chiplet_wgm;
    fwd.chiplet_num_xcds = s->chiplet_num_xcds;
    fwd.chiplet_chunk_size = s->chiplet_chunk_size;
    /* vector sizes: wgrad always uses vec=1 for A and B */
    fwd.has_vector_size_a = true;
    fwd.vector_size_a = 1;
    fwd.has_vector_size_b = true;
    fwd.vector_size_b = 1;
    /* vector_size_c (store width) from wgrad spec if set */
    fwd.has_vector_size_c = s->has_vector_size_c;
    fwd.vector_size_c = s->vector_size_c;
    /* acc_epilogue: wgrad struct omits it, forward defaults to identity */
    return fwd;
}

// ---------------------------------------------------------------------------
// rocke_build_implicit_gemm_conv_wgrad
// ---------------------------------------------------------------------------

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

    /* Resolve split_k=-1 (auto): treat as split_k=1 for now (simple path). */
    int split_k = spec->split_k;
    if(split_k == -1)
        split_k = 1;
    if(split_k > 1)
    {
        /* split_k > 1 (atomic-add) is not yet implemented in this C port.
         * Return NULL so the parity gate sees both C and Python reject, OR
         * accept whichever configs have split_k=1. */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "wgrad: split_k > 1 not implemented in C port (split_k=%d)",
                        split_k);
        return NULL;
    }

    /* --- build an adapter forward-conv spec + problem --- */
    rocke_conv_problem_t adapter_p = make_adapter_problem(spec);
    rocke_implicit_gemm_conv_spec_t fwd_spec = make_adapter_fwd_spec(spec, &adapter_p);

    /* --- init the shared build context --- */
    rocke_conv_build_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    if(!rocke_conv_build_ctx_init(&ctx, b, &fwd_spec, arch, /*overrides=*/NULL))
        return NULL; /* builder error already set */

    /* --- substitute wgrad-specific descriptors --- */
    const rocke_conv_problem_t* p = &spec->problem;

    rocke_tensor_descriptor_t* dY_desc = wgrad_make_dy_descriptor(b, p);
    rocke_tensor_descriptor_t* X_desc = wgrad_make_x_descriptor(b, p);
    rocke_tensor_descriptor_t* dW_desc = wgrad_make_dw_descriptor(b, p);

    if(dY_desc == NULL || X_desc == NULL || dW_desc == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: descriptor build failed");
        return NULL;
    }

    /* Override the descriptors in the ctx.  The forward K-loop drivers call
     * rocke_conv_a_descriptor (reads ctx->A_desc with coords "m","k") and
     * rocke_conv_b_descriptor (reads ctx->B_desc with coords "k_out","k_gemm").
     * Our wgrad descriptors are built with exactly those coord names so no
     * additional closure patching is needed. */
    ctx.A_desc = dY_desc; /* A: dY NHWK, top-level coords ("m"=k_out, "k"=k_wg_red) */
    ctx.B_desc = X_desc; /* B: X  NHWC, top-level coords ("k_out"=n_wg, "k_gemm"=k_wg_red) */
    ctx.D_desc = dW_desc; /* D: dW KYXC, top-level coords ("m"=k_out, "k_out"=n_wg) */

    /* --- K-loop (same drivers as forward conv) --- */
    if(fwd_spec.unroll_k)
        rocke_conv_emit_kloop_unroll(&ctx);
    else if(!fwd_spec.async_dma)
        rocke_conv_emit_kloop_simple(&ctx);
    else
        rocke_conv_emit_kloop_async(&ctx);

    if(!rocke_ir_builder_ok(b))
        return NULL;

    /* --- epilogue ---
     * The forward epilogue writes to D using ctx.D_desc which we set to dW_desc.
     * The epilogue bounds use (p.M, p.N_gemm) from the adapter problem, which
     * equal (wg_M, wg_N) by construction.  So rocke_conv_emit_epilogue works
     * without modification. */
    rocke_conv_emit_epilogue(&ctx);

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
        const char* m = rocke_ir_builder_error(&b);
        set_err((m && m[0]) ? m : "build_implicit_gemm_conv_wgrad failed");
        rocke_ir_builder_free(&b);
        return rocke_ir_builder_status(&b);
    }

    rocke_status_t st = rocke_lower_kernel_to_llvm(kernel, flavor, arch, out_ll);
    rocke_ir_builder_free(&b);
    return st;
}
