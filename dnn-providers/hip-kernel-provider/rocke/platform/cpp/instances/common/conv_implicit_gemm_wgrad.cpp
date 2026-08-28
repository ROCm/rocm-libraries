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

    /* flags: async, spk{N}, spkauto  -- Python boolean flags */
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
    else if(s->split_k == 0)
    {
        flag_names[n_flags] = "spkrt";
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
    if(sk < -1)
    {
        if(reason && reason_cap)
            snprintf(reason,
                     reason_cap,
                     "split_k must be -1 (auto), 0 (runtime), 1, or >1 (got %d)",
                     sk);
        return false;
    }

    /* split_k > 1 or split_k == 0 (runtime atomic) requires a MFMA arch
     * (ctx->atom != NULL at build time).
     *
     * TODO: gate on resolved wave_size == 64 / op->family == "mma" (matching
     * Python which uses family == "wmma") instead of the arch string, so
     * gfx10* and any future or unknown arch prefix cannot fall through.  This
     * is not reachable on today's supported targets but would be more robust. */
    if(sk > 1 || sk == 0)
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
         * global_atomic_add_pk_f16/bf16.  Each pair spans two adjacent C
         * positions within one (y,x) filter position.  An odd C means the last
         * element of a row has no partner and the pair straddles a filter-position
         * boundary, producing a wrong-geometry atomic.
         * Matches Python is_valid_wgrad_spec: "requires even C". */
        const char* dt = s->dtype_d ? s->dtype_d : "fp16";
        if(strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0)
        {
            if(s->problem.C % 2 != 0)
            {
                if(reason && reason_cap)
                    snprintf(reason,
                             reason_cap,
                             "split_k atomic with dtype_d=%s requires even C "
                             "(packed <2 x dtype> atomic pairs must stay within one filter "
                             "position); got C=%d",
                             dt,
                             s->problem.C);
                return false;
            }
        }
    }

    /* For bf16/fp16 output the default epilogue emits zero-fill packed atomics
     * at the scattered MFMA layout.  Matches Python is_valid_wgrad_spec and
     * validate(): epilogue='cshuffle' is required for these dtypes. */
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
    rocke_coalesced_tile_loader_load(
        b, &ctx->a_sync_loader, ctx->tid, A_dst, wgrad_dy_descriptor, ctx, ctx->a_rsrc, NULL);
}

// ---------------------------------------------------------------------------
// Additional includes for split-K epilogue and ctx init helpers
// ---------------------------------------------------------------------------
#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.epilogues.h"
#include "rocke/helper_rocke.helpers.grid.h"
#include "rocke/helper_rocke.helpers.schedule.h"

// ---------------------------------------------------------------------------
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
                                            int wg_M,
                                            int wg_N)
{
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
    rocke_value_t* wg_M_v = rocke_b_const_i32(b, wg_M);
    rocke_value_t* wg_N_v = rocke_b_const_i32(b, wg_N);

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
                                         rocke_b_const_i32(b, wg_N),
                                         rocke_b_const_i32(b, wg_M),
                                         rocke_b_const_i32(b, wg_N));
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

static void wgrad_emit_direct_epilogue(rocke_ir_builder_t* b,
                                       const rocke_conv_build_ctx_t* ctx,
                                       const rocke_implicit_gemm_conv_wgrad_spec_t* spec,
                                       rocke_tensor_descriptor_t* dW_desc,
                                       rocke_value_t* dw_rsrc,
                                       int wg_M,
                                       int wg_N)
{
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
            rocke_value_t* c_N = rocke_b_const_i32(b, wg_N);
            rocke_value_t* bound_m = rocke_b_const_i32(b, wg_M);
            rocke_value_t* bound_n = rocke_b_const_i32(b, wg_N);
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
            addr_ctx.dW_desc = dW_desc;
            rocke_value_t* bound_m = rocke_b_const_i32(b, wg_M);
            rocke_value_t* bound_n = rocke_b_const_i32(b, wg_N);
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

    rocke_value_t* c_M = rocke_b_const_i32(b, wg_M);
    rocke_value_t* c_N = rocke_b_const_i32(b, wg_N);

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
                                 int split_k, /* resolved (>= 1, or 0 for runtime) */
                                 rocke_value_t* ks_param) /* non-NULL iff split_k == 0 */
{
    if(ctx == NULL || b == NULL || spec == NULL)
        return false;

    memset(ctx, 0, sizeof(*ctx));
    ctx->b = b;
    ctx->arch = arch;

    /* Set up wgrad-specific overrides.  The only override needed is
     * a_load_override, which calls the sync loader with wgrad_dy_descriptor
     * (m_val-first SSA order) instead of the shared rocke_conv_a_descriptor
     * (k_val-first).  The async path goes through rocke_conv_emit_load_phase
     * which calls rocke_conv_a_descriptor directly for the async slot; wgrad
     * does not yet support async_dma, so that path is unreachable. */
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
            ctx->b_descriptor_fn = wgrad_x_descriptor;
        }
        else
        {
            ctx->c_K_pw = 0;
            ctx->c_C_pw = 0;
            ctx->c_M_pw = 0;
            ctx->c_wgK_pw = 0;
            ctx->c_wgN_pw = 0;
            ctx->b_descriptor_fn = wgrad_x_descriptor; /* always use wgrad X descriptor */
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
        ctx->atom
            = ctx->is_wmma
                  ? NULL
                  : rocke_mfma_atom("f16", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
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
     *   c0        = b.const_i32(0)         -- always (split_k=1: k_lo; split_k>1/0: unused 0)
     *   c_block_k = b.const_i32(block_k)   -- always
     *   c_wg_K    = b.const_i32(wg_K)      -- always (used as loop bound when split_k=1)
     *   [split_k>1 only] c_ks, k_lo=to_sgpr(mul(block_id_z,c_ks)), k_hi=to_sgpr(add(k_lo,c_ks))
     *   [split_k==0 only] c_ks=ks_param,   k_lo=to_sgpr(mul(block_id_z,c_ks)), k_hi=to_sgpr(add(k_lo,c_ks))
     */
    int wg_K_padded_val = rocke_wgrad_conv_spec_wg_K_padded(spec);

    /* c0: always const(0). For split_k=1 this is also k_lo. */
    rocke_value_t* c0_node = rocke_b_const_i32(b, 0);
    /* c_block_k: always created here (matches Python ordering). */
    ctx->c_block_k = rocke_b_const_i32(b, ctx->block_k);
    /* c_wg_K: always created (occupies SSA slot even for split_k>1). */
    rocke_value_t* c_wg_K = rocke_b_const_i32(b, wg_K);

    rocke_value_t* k_lo;
    rocke_value_t* k_hi_v; /* NULL => loop runs to c_wg_K */
    if(split_k > 1)
    {
        int ks = wg_K_padded_val / split_k;
        rocke_value_t* c_ks = rocke_b_const_i32(b, ks);
        k_lo = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_z(b), c_ks));
        k_hi_v = rocke_b_to_sgpr_u32(b, rocke_b_add(b, k_lo, c_ks));
    }
    else if(split_k == 0)
    {
        /* Runtime atomic: ks is a kernel argument passed at launch time.
         * Mirrors Python: c_ks = _ks_param; k_lo = to_sgpr(mul(block_id_z, c_ks))
         * groups==1 is the only supported path for C++ (grouped wgrad is rejected by
         * the validator), so no ks_count / group decode is needed. */
        rocke_value_t* c_ks = ks_param; /* i32 kernel arg */
        k_lo = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, rocke_b_block_id_z(b), c_ks));
        k_hi_v = rocke_b_to_sgpr_u32(b, rocke_b_add(b, k_lo, c_ks));
    }
    else
    {
        k_lo = c0_node;
        k_hi_v = NULL;
    }
    ctx->c0 = k_lo;
    ctx->c_K_gemm = (k_hi_v != NULL) ? k_hi_v : c_wg_K;

    /* Chiplet swizzle */
    if(spec->chiplet_swizzle)
    {
        int wg_M_val = spec->problem.K;
        int wg_N_val = rocke_wgrad_conv_spec_wg_N(spec);
        int npm = (wg_M_val + ctx->block_m - 1) / ctx->block_m;
        int npn = (wg_N_val + ctx->block_n - 1) / ctx->block_n;
        rocke_value_t* c_npn = rocke_b_const_i32(b, npn);
        rocke_value_t* bid_y = rocke_b_block_id_y(b);
        rocke_value_t* mul_y = rocke_b_mul(b, bid_y, c_npn);
        rocke_value_t* bid_x = rocke_b_block_id_x(b);
        rocke_value_t* wgflat = rocke_b_add(b, mul_y, bid_x);
        rocke_super_tile_swizzle_result_t swz
            = rocke_chiplet_aware_super_tile(b,
                                             wgflat,
                                             npm,
                                             npn,
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
        int a_sh[2] = {ctx->block_m, ctx->lds_layout.row_stride};
        int b_sh[2] = {ctx->block_n, ctx->lds_layout.row_stride};
        ctx->A_smem = rocke_b_smem_alloc(b, rocke_f16(), a_sh, 2, "A_smem");
        ctx->B_smem = rocke_b_smem_alloc(b, rocke_f16(), b_sh, 2, "B_smem");
        ctx->double_buffer = (spec->pipeline && strcmp(spec->pipeline, "compv4") == 0)
                             || spec->async_dma || spec->unroll_k;
        if(ctx->double_buffer)
        {
            ctx->A_smem2 = rocke_b_smem_alloc(b, rocke_f16(), a_sh, 2, "A_smem2");
            ctx->B_smem2 = rocke_b_smem_alloc(b, rocke_f16(), b_sh, 2, "B_smem2");
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
    if(ctx->async_dma)
    {
        rocke_status_t sa = rocke_async_tile_loader_from_tile(
            ctx->block_m, ctx->block_k, ctx->threads, spec->wave_size, 4, &ctx->a_loader);
        rocke_status_t sb = rocke_async_tile_loader_from_tile(
            ctx->block_n, ctx->block_k, ctx->threads, spec->wave_size, 4, &ctx->b_loader);
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
        }

        /* Direct struct construction mirrors the Python CoalescedTileLoader(...)
         * call (not from_tile): explicit load_vec + vector_axis, use_buffer_rsrc
         * default True, oob_sentinel default (1 << 31) - 1. */
        ctx->a_sync_loader.tile_rows = ctx->block_m;
        ctx->a_sync_loader.tile_cols = ctx->block_k;
        ctx->a_sync_loader.block_size = ctx->threads;
        ctx->a_sync_loader.load_vec = va;
        ctx->a_sync_loader.use_buffer_rsrc = true;
        ctx->a_sync_loader.oob_sentinel = 2147483647;
        ctx->a_sync_loader.vector_axis_row = axis_a;
        ctx->a_sync_loader.has_inner_dim = false;
        ctx->a_sync_loader.inner_dim = 0;

        ctx->b_sync_loader.tile_rows = ctx->block_n;
        ctx->b_sync_loader.tile_cols = ctx->block_k;
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
    bool is_split_k = (split_k > 1 || split_k == 0);
    bool split_k_runtime = (split_k == 0);

    /* split_k atomic (>1 or ==0) supported for fp32, fp16, bf16 output dtypes */
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
    const rocke_type_t* f16_glob = rocke_ptr_type(b, rocke_f16(), "global");

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
    /* split_k>1 or split_k==0: dW is read+write (atomic); split_k=1: writeonly.
     * Caller MUST zero-init dW before launch for atomic paths -- the kernel only
     * issues atomic-adds.  See the header contract note for details. */
    d_opts.writeonly = !is_split_k;
    d_opts.writeonly_set = true;
    d_opts.align = 16;
    d_opts.align_set = true;

    /* dtype for dW: use dtype_d field */
    bool is_fp32_d = (spec->dtype_d && strcmp(spec->dtype_d, "fp32") == 0);
    const rocke_type_t* dw_glob = is_fp32_d ? rocke_ptr_type(b, rocke_f32(), "global") : f16_glob;

    rocke_value_t* dY = rocke_b_param(b, "dY", f16_glob, &ro_opts);
    rocke_value_t* X = rocke_b_param(b, "X", f16_glob, &ro_opts);
    rocke_value_t* dW = rocke_b_param(b, "dW", dw_glob, &d_opts);
    rocke_value_t* dY_bytes = rocke_b_param(b, "dY_bytes", rocke_i32(), NULL);
    rocke_value_t* X_bytes = rocke_b_param(b, "X_bytes", rocke_i32(), NULL);
    rocke_value_t* dW_bytes = rocke_b_param(b, "dW_bytes", rocke_i32(), NULL);
    /* Runtime split-K: ks = slice width, supplied by the launcher at dispatch.
     * Only emitted when split_k == 0; fixed-degree kernels bake ks as a const.
     * Mirrors Python: _ks_param = b.param("ks", I32) if _split_k_runtime else None */
    rocke_value_t* ks_param = split_k_runtime ? rocke_b_param(b, "ks", rocke_i32(), NULL) : NULL;

    /* --- build wgrad ctx (with correct param names) --- */
    rocke_conv_build_ctx_t ctx;
    if(!wgrad_build_ctx_init(&ctx, b, spec, arch, wg_K, split_k, ks_param))
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
        ctx.ir_c_C_pw = rocke_b_const_i32(b, ctx.c_K_pw); /* kpg = _c_K_ir */
        ctx.ir_c_K_pw = rocke_b_const_i32(b, ctx.c_C_pw); /* cpg = _c_C_ir */
        ctx.ir_c_M_pw = rocke_b_const_i32(b, ctx.c_M_pw); /* wg_M = _c_wgM_ir */
        ctx.ir_always_valid = rocke_b_const_i32(b, ctx.c_wgN_pw); /* wg_N = _c_wgN_ir */
        ctx.ir_c_wgN_pw = rocke_b_const_i32(b, ctx.c_wgK_pw); /* wg_K = _c_wgK_ir */
    }
    else
    {
        ctx.ir_c_C_pw = ctx.ir_c_K_pw = ctx.ir_c_M_pw = ctx.ir_always_valid = NULL;
        ctx.ir_c_wgN_pw = NULL;
    }

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

    /* --- wgrad-specific descriptors ---
     * Pointwise fast path (Y=X=1, stride=1, pad=0): descriptors are NULL; the
     * wgrad_dy_descriptor / wgrad_x_descriptor closures use flat arithmetic.
     * Non-pointwise: build the full coordinate-transform descriptor DAGs. */
    rocke_tensor_descriptor_t* dY_desc = NULL;
    rocke_tensor_descriptor_t* X_desc = NULL;
    rocke_tensor_descriptor_t* dW_desc = NULL;
    if(!ctx.is_pointwise)
    {
        dY_desc = wgrad_make_dy_descriptor(b, p);
        X_desc = wgrad_make_x_descriptor(b, p);
        dW_desc = wgrad_make_dw_descriptor(b, p);
        if(dY_desc == NULL || X_desc == NULL || dW_desc == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad: descriptor build failed");
            return NULL;
        }
    }
    /* The forward phase functions query A_desc with ("m","k") and B_desc with
     * ("k_out","k_gemm") -- our descriptors are built with exactly those names. */
    ctx.A_desc = dY_desc;
    ctx.B_desc = X_desc;
    ctx.D_desc = dW_desc;

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
    if(is_split_k)
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

        WgradDwAddrCtx dw_addr_ctx;
        dw_addr_ctx.dW_desc = dW_desc;

        if(spec->epilogue && strcmp(spec->epilogue, "cshuffle") == 0)
        {
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
                                              rocke_b_const_i32(b, wg_M),
                                              rocke_b_const_i32(b, wg_N));
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
                                              rocke_b_const_i32(b, wg_M),
                                              rocke_b_const_i32(b, wg_N));
            }
        }
        else
        {
            /* Use wgrad-specific direct epilogue. Mirrors Python _emit_wgrad_direct_epilogue
             * for MFMA, _emit_wgrad_direct_epilogue_wmma for WMMA. */
            wgrad_emit_direct_epilogue(b, &ctx, spec, dW_desc, ctx.d_rsrc, wg_M, wg_N);
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
        const char* m = rocke_ir_builder_error(&b);
        set_err((m && m[0]) ? m : "build_implicit_gemm_conv_wgrad failed");
        rocke_ir_builder_free(&b);
        return rocke_ir_builder_status(&b);
    }

    rocke_status_t st = rocke_lower_kernel_to_llvm(kernel, flavor, arch, out_ll);
    rocke_ir_builder_free(&b);
    return st;
}
