// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the ConvProblem dataclass from
 * rocke/instances/common/conv_implicit_gemm.py.
 *
 * Python integer semantics note: every // in ConvProblem operates on
 * non-negative operands (valid convolution shapes), so C integer division
 * (truncation toward zero) matches Python floor division exactly. The flops
 * product is computed in 64-bit because the int32 result Python's
 * arbitrary-precision int never overflows can exceed 2^31 for large shapes.
 *
 * #8355: ConvProblem fields R->Y, S->X and optional 3-D depth dims
 * (Di/Z/sD/pD/dD). is_3d gates the depth-aware Do/M/K_gemm/short().
 */
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h"

#include <stdio.h> /* snprintf */
#include <stdlib.h> /* qsort -- the sorted TF32 atom list */
#include <string.h> /* strcmp */

#include "rocke/dtypes.h" /* rocke_normalize_dtype -- _is_tf32_spelling */
#include "rocke/helper_rocke.core.arch.h" /* ArchTarget -- the TF32 atom gate */
#include "rocke/helper_rocke.helpers.atoms.h" /* mfma_atom -- the arch-free TF32 gate */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_spec_set_reasonf */

rocke_conv_problem_t rocke_conv_problem_make(int N,
                                             int Hi,
                                             int Wi,
                                             int C,
                                             int K,
                                             int Y,
                                             int X,
                                             int sH,
                                             int sW,
                                             int pH,
                                             int pW,
                                             int dH,
                                             int dW)
{
    rocke_conv_problem_t p;
    p.N = N;
    p.Hi = Hi;
    p.Wi = Wi;
    p.C = C;
    p.K = K;
    p.Y = Y;
    p.X = X;
    p.sH = sH;
    p.sW = sW;
    p.pH = pH;
    p.pW = pW;
    p.dH = dH;
    p.dW = dW;
    p.is_3d = false;
    p.Di = 0;
    p.Z = 0;
    p.sD = 0;
    p.pD = 0;
    p.dD = 0;
    p.groups = 1;
    return p;
}

rocke_conv_problem_t rocke_conv_problem_make_3d(int N,
                                                int Di,
                                                int Hi,
                                                int Wi,
                                                int C,
                                                int K,
                                                int Z,
                                                int Y,
                                                int X,
                                                int sD,
                                                int sH,
                                                int sW,
                                                int pD,
                                                int pH,
                                                int pW,
                                                int dD,
                                                int dH,
                                                int dW)
{
    rocke_conv_problem_t p = rocke_conv_problem_make(N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW);
    p.is_3d = true;
    p.Di = Di;
    p.Z = Z;
    p.sD = sD;
    p.pD = pD;
    p.dD = dD;
    return p;
}

rocke_conv_problem_t rocke_conv_problem_default(int N, int Hi, int Wi, int C, int K, int Y, int X)
{
    /* sH=1, sW=1, pH=0, pW=0, dH=1, dW=1 (Python dataclass defaults). */
    return rocke_conv_problem_make(N, Hi, Wi, C, K, Y, X, 1, 1, 0, 0, 1, 1);
}

bool rocke_conv_problem_is_3d(const rocke_conv_problem_t* p)
{
    return p->is_3d;
}

/* (Di + 2*pD - dD*(Z - 1) - 1) // sD + 1 ; 1 for 2-D. */
int rocke_conv_problem_do(const rocke_conv_problem_t* p)
{
    if(!p->is_3d)
    {
        return 1;
    }
    return (p->Di + 2 * p->pD - p->dD * (p->Z - 1) - 1) / p->sD + 1;
}

/* (Hi + 2*pH - dH*(Y - 1) - 1) // sH + 1 */
int rocke_conv_problem_ho(const rocke_conv_problem_t* p)
{
    return (p->Hi + 2 * p->pH - p->dH * (p->Y - 1) - 1) / p->sH + 1;
}

/* (Wi + 2*pW - dW*(X - 1) - 1) // sW + 1 */
int rocke_conv_problem_wo(const rocke_conv_problem_t* p)
{
    return (p->Wi + 2 * p->pW - p->dW * (p->X - 1) - 1) / p->sW + 1;
}

/* N * Ho * Wo  (* Do for 3-D) */
int rocke_conv_problem_m(const rocke_conv_problem_t* p)
{
    int base = p->N * rocke_conv_problem_ho(p) * rocke_conv_problem_wo(p);
    return p->is_3d ? base * rocke_conv_problem_do(p) : base;
}

/* K / groups  (= kpg; per-group output channels) */
int rocke_conv_problem_n_gemm(const rocke_conv_problem_t* p)
{
    return rocke_conv_problem_kpg(p);
}

/* Y * X * cpg  (Z * Y * X * cpg for 3-D) */
int rocke_conv_problem_k_gemm(const rocke_conv_problem_t* p)
{
    int z = p->is_3d ? p->Z : 1;
    return z * p->Y * p->X * rocke_conv_problem_cpg(p);
}

/* 2 * M * N_gemm * K_gemm * groups  (Python property: groups factor restores total FLOPs) */
long long rocke_conv_problem_flops(const rocke_conv_problem_t* p)
{
    long long m = (long long)rocke_conv_problem_m(p);
    long long n = (long long)rocke_conv_problem_n_gemm(p);
    long long k = (long long)rocke_conv_problem_k_gemm(p);
    long long g = (long long)(p->groups > 1 ? p->groups : 1);
    return 2LL * m * n * k * g;
}

/* 2-D: f"N{N}H{Hi}W{Wi}C{C}_K{K}Y{Y}X{X}{g}"   where g="G{groups}" if groups>1 else ""
 * 3-D: f"N{N}D{Di}H{Hi}W{Wi}C{C}_K{K}Z{Z}Y{Y}X{X}{g}" */
rocke_status_t rocke_conv_problem_short(const rocke_conv_problem_t* p,
                                        char* out,
                                        size_t out_cap,
                                        size_t* out_len)
{
    int written;

    if(p == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Python: g = f"G{groups}" if groups > 1 else "" */
    const char* g_suffix = "";
    char g_buf[16];
    if(p->groups > 1)
    {
        snprintf(g_buf, sizeof(g_buf), "G%d", p->groups);
        g_suffix = g_buf;
    }

    if(p->is_3d)
    {
        written = snprintf(out,
                           out_cap,
                           "N%dD%dH%dW%dC%d_K%dZ%dY%dX%d%s",
                           p->N,
                           p->Di,
                           p->Hi,
                           p->Wi,
                           p->C,
                           p->K,
                           p->Z,
                           p->Y,
                           p->X,
                           g_suffix);
    }
    else
    {
        written = snprintf(out,
                           out_cap,
                           "N%dH%dW%dC%d_K%dY%dX%d%s",
                           p->N,
                           p->Hi,
                           p->Wi,
                           p->C,
                           p->K,
                           p->Y,
                           p->X,
                           g_suffix);
    }
    if(written < 0 || (size_t)written >= out_cap)
    {
        /* Encoding error or truncation: the buffer is too small. */
        return ROCKE_ERR_VALUE;
    }
    if(out_len != NULL)
    {
        *out_len = (size_t)written;
    }
    return ROCKE_OK;
}

bool rocke_conv_coalesced_load_ok(const char* operand,
                                  int tile_rows,
                                  int tile_cols,
                                  int block_size,
                                  int load_vec,
                                  char* reason,
                                  size_t reason_cap)
{
    /* Python: chunks = (tile_rows * tile_cols) // load_vec
     *         if chunks % block_size: return f"{operand} load: ..." */
    const int chunks = load_vec > 0 ? (tile_rows * tile_cols) / load_vec : 0;
    if(block_size > 0 && load_vec > 0 && chunks % block_size == 0)
        return true;
    if(reason != NULL && reason_cap > 0)
        snprintf(reason,
                 reason_cap,
                 "%s load: tile %dx%d / %d = %d not divisible by block_size %d",
                 operand,
                 tile_rows,
                 tile_cols,
                 load_vec,
                 chunks,
                 block_size);
    return false;
}

/* Python twin: _VECTOR_WIDTH_DTYPE_BYTES in _conv_implicit_gemm_common.py.
 * Keyed explicitly so a future 1- or 8-byte dtype is a one-line table edit
 * here and there, instead of a silent fallthrough to the 2-byte default. */
static int rocke_conv_vector_width_elem_bytes(const char* dtype)
{
    if(strcmp(dtype, "fp32") == 0)
        return 4;
    return 2;
}

bool rocke_conv_vector_width_ok(
    const char* operand, bool has_vec, int vec, const char* dtype, char* reason, size_t reason_cap)
{
    /* Python: nbytes = vec * (4 if dtype == "fp32" else 2)
     *         if nbytes > MAX_VECTOR_BYTES: return f"vector_size_{operand}=..." */
    const int max_vector_bytes = 16;
    const char* dt = dtype ? dtype : "fp16";
    if(!has_vec)
        return true;
    const long long nbytes = (long long)vec * rocke_conv_vector_width_elem_bytes(dt);
    if(nbytes <= max_vector_bytes)
        return true;
    if(reason != NULL && reason_cap > 0)
        snprintf(reason,
                 reason_cap,
                 "vector_size_%s=%d x %s is %lld bytes > %d-byte max per-lane access",
                 operand,
                 vec,
                 dt,
                 nbytes,
                 max_vector_bytes);
    return false;
}

/* ===================================================================== *
 *  ConvDataSpec.mma_dtype -- TF32 compute
 *
 *  Python twin: the ConvDataSpec properties and the "TF32 compute helpers"
 *  block of _conv_implicit_gemm_common.py. Each *_reason returns true and
 *  writes the Python reason text when the spec cannot build.
 * ===================================================================== */

/* Write the printf-style reason (rocke_spec_set_reasonf) and evaluate to true,
 * "cannot build". */
#define ROCKE_CONV_TF32_FAIL(reason, reason_cap, ...) \
    (rocke_spec_set_reasonf((reason), (reason_cap), __VA_ARGS__), true)

bool rocke_conv_is_tf32_spelling(const char* dtype)
{
    /* isinstance(dtype, str) and normalize_dtype(dtype) == "tf32" */
    if(dtype == NULL)
        return false;
    char scratch[64];
    const char* canon = rocke_normalize_dtype(dtype, scratch, sizeof(scratch));
    return canon != NULL && strcmp(canon, "tf32") == 0;
}

bool rocke_conv_data_is_tf32(const char* mma_dtype)
{
    return mma_dtype != NULL && strcmp(mma_dtype, "tf32") == 0;
}

const char* rocke_conv_data_mma_operand(const char* storage, const char* mma_dtype)
{
    /* self.mma_dtype or self.dtype_a ("" is falsy, as in Python) */
    return (mma_dtype != NULL && mma_dtype[0] != '\0') ? mma_dtype : storage;
}

bool rocke_conv_mma_dtype_reason(const char* dtype_a,
                                 const char* dtype_b,
                                 const char* dtype_d,
                                 const char* mma_dtype,
                                 char* reason,
                                 size_t reason_cap)
{
    if(rocke_conv_is_tf32_spelling(dtype_a) || rocke_conv_is_tf32_spelling(dtype_b))
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "tf32 is a compute mode, not a storage dtype: use "
                                    "dtype_a='fp32', dtype_b='fp32' with mma_dtype='tf32'");
    if(mma_dtype == NULL)
        return false;
    if(strcmp(mma_dtype, "tf32") != 0)
        return ROCKE_CONV_TF32_FAIL(
            reason, reason_cap, "mma_dtype must be None or 'tf32', got '%s'", mma_dtype);
    const char* da = dtype_a ? dtype_a : "fp16";
    const char* db = dtype_b ? dtype_b : "fp16";
    if(strcmp(da, "fp32") != 0 || strcmp(db, "fp32") != 0)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32' needs fp32 A and B storage, got "
                                    "dtype_a='%s', dtype_b='%s'",
                                    da,
                                    db);
    /* TF32 is a compute mode for fp32 convolutions (as in cuDNN and MIOpen):
     * one rule for fwd, dgrad and wgrad, so they cannot drift. */
    const char* dd = dtype_d ? dtype_d : "fp16";
    if(strcmp(dd, "fp32") != 0)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32' is an fp32-convolution compute mode and "
                                    "needs an fp32 output (dtype_d='fp32'), got dtype_d='%s'",
                                    dd);
    return false;
}

/* _TF32_EMULATED_MAX_ACC_PER_LANE: emulated (BF16x3) TF32 warp tiles at or
 * above this many fp32 accumulators per lane sit at the 512-VGPR ceiling once
 * the hi/lo split temporaries are live, where comgr's VGPR->AGPR spill path
 * miscompiles (see the Python comment). */
static const int kTf32EmulatedMaxAccPerLane = 384;

/* _tf32_any_arch_frag_len(spec): the per-lane fragment of the TF32 atom the
 * warp tile names on any arch, from the arch-agnostic MfmaAtom table; -1 for
 * Python None (no such atom). */
static int rocke_conv_tf32_any_arch_frag_len(const rocke_conv_tf32_spec_view_t* spec)
{
    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom("tf32", spec->warp_tile_m, spec->warp_tile_n, spec->warp_tile_k);
    return atom != NULL ? atom->a_per_lane : -1;
}

/* _tf32_register_reason(spec, op): reject an emulated-TF32 warp tile that
 * reaches the VGPR ceiling. A NULL or native op is not checked. */
static bool rocke_conv_tf32_register_reason(const rocke_conv_tf32_spec_view_t* spec,
                                            const rocke_mma_op_t* op,
                                            char* reason,
                                            size_t reason_cap)
{
    if(op == NULL || rocke_arch_mma_op_is_native(op))
        return false;
    const int acc = (spec->tile_m / spec->warp_m) * (spec->tile_n / spec->warp_n) / 64;
    if(acc >= kTf32EmulatedMaxAccPerLane)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32': %d fp32 accumulators per lane with the "
                                    "BF16x3-emulated atom reach the 512-VGPR ceiling, where the "
                                    "VGPR->AGPR spill path miscompiles; use a smaller warp tile "
                                    "(< %d accumulators per lane)",
                                    acc,
                                    kTf32EmulatedMaxAccPerLane);
    return false;
}

/* _tf32_lds_stride_reason(row_stride, frag): the LDS row stride must keep a
 * TF32 fragment read aligned -- a multiple of min(frag, 4) fp32 elements. */
static bool
    rocke_conv_tf32_lds_stride_reason(int row_stride, int frag, char* reason, size_t reason_cap)
{
    const int need = frag < ROCKE_CONV_TF32_LDS_LOAD_MAX ? frag : ROCKE_CONV_TF32_LDS_LOAD_MAX;
    if(need != 0 && row_stride % need != 0)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32': LDS row stride %d fp32 elements is not "
                                    "a multiple of %d, so the %d-wide fragment read would be "
                                    "misaligned; use an lds_k_pad that keeps tile_k + pad a "
                                    "multiple of %d",
                                    row_stride,
                                    need,
                                    frag,
                                    need);
    return false;
}

/* A TF32 atom's (m, n, k) as it sorts in Python's sorted() list of tuples. */
typedef struct
{
    int m, n, k;
} rocke_conv_tf32_shape_t;

static int rocke_conv_tf32_shape_cmp(const void* lhs, const void* rhs)
{
    const rocke_conv_tf32_shape_t* a = (const rocke_conv_tf32_shape_t*)lhs;
    const rocke_conv_tf32_shape_t* b = (const rocke_conv_tf32_shape_t*)rhs;
    if(a->m != b->m)
        return a->m < b->m ? -1 : 1;
    if(a->n != b->n)
        return a->n < b->n ? -1 : 1;
    if(a->k != b->k)
        return a->k < b->k ? -1 : 1;
    return 0;
}

static bool rocke_conv_tf32_is_atom_row(const rocke_mma_op_t* op)
{
    return op->family != NULL && strcmp(op->family, "mma") == 0 && op->srcs[0].dtype != NULL
           && strcmp(op->srcs[0].dtype, "tf32") == 0 && op->srcs[1].dtype != NULL
           && strcmp(op->srcs[1].dtype, "tf32") == 0 && op->srcs[2].dtype != NULL
           && strcmp(op->srcs[2].dtype, "fp32") == 0;
}

/* _tf32_reason(spec, target, family, arch): why a TF32 spec cannot build on
 * `arch` -- a WMMA target, an MFMA arch with no TF32 row, a warp tile that is
 * not one of its TF32 atoms, or the emulated atom's register ceiling. */
static bool rocke_conv_tf32_reason(const rocke_conv_tf32_spec_view_t* spec,
                                   const rocke_archtarget_t* target,
                                   const char* family,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    if(strcmp(family, "mma") != 0)
        return ROCKE_CONV_TF32_FAIL(
            reason, reason_cap, "mma_dtype='tf32' is MFMA-only; %s is a WMMA target", arch);

    int num_ops = 0;
    const rocke_mma_op_t* ops = rocke_mma_catalog_ops(rocke_archtarget_mma(target), &num_ops);
    rocke_conv_tf32_shape_t shapes[64];
    int num_shapes = 0;
    for(int i = 0; i < num_ops && num_shapes < (int)(sizeof(shapes) / sizeof(shapes[0])); ++i)
    {
        if(!rocke_conv_tf32_is_atom_row(&ops[i]))
            continue;
        shapes[num_shapes].m = ops[i].m;
        shapes[num_shapes].n = ops[i].n;
        shapes[num_shapes].k = ops[i].k;
        ++num_shapes;
    }
    if(num_shapes == 0)
        return ROCKE_CONV_TF32_FAIL(
            reason, reason_cap, "mma_dtype='tf32': %s has no TF32 MFMA atom", arch);
    qsort(shapes, (size_t)num_shapes, sizeof(shapes[0]), rocke_conv_tf32_shape_cmp);

    bool listed = false;
    for(int i = 0; i < num_shapes; ++i)
        listed = listed
                 || (shapes[i].m == spec->warp_tile_m && shapes[i].n == spec->warp_tile_n
                     && shapes[i].k == spec->warp_tile_k);
    if(!listed)
    {
        /* f"(have {_shapes})": Python's repr of a sorted list of tuples. */
        char have[512];
        size_t pos = 0;
        have[0] = '\0';
        for(int i = 0; i < num_shapes && pos < sizeof(have); ++i)
        {
            int w = snprintf(have + pos,
                             sizeof(have) - pos,
                             "%s(%d, %d, %d)",
                             i == 0 ? "" : ", ",
                             shapes[i].m,
                             shapes[i].n,
                             shapes[i].k);
            if(w < 0)
                break;
            pos += (size_t)w;
        }
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32': warp_tile (%d, %d, %d) is not a TF32 atom "
                                    "on %s (have [%s])",
                                    spec->warp_tile_m,
                                    spec->warp_tile_n,
                                    spec->warp_tile_k,
                                    arch,
                                    have);
    }
    const rocke_mma_op_t* op = rocke_archtarget_op_for_shape(target,
                                                             "mma",
                                                             "tf32",
                                                             "tf32",
                                                             "fp32",
                                                             spec->warp_tile_m,
                                                             spec->warp_tile_n,
                                                             spec->warp_tile_k,
                                                             nullptr);
    return rocke_conv_tf32_register_reason(spec, op, reason, reason_cap);
}

bool rocke_conv_tf32_conv_spec_reason(const rocke_conv_tf32_spec_view_t* spec,
                                      const char* arch,
                                      bool async_needs_k_outer,
                                      char* reason,
                                      size_t reason_cap)
{
    if(rocke_conv_mma_dtype_reason(
           spec->dtype_a, spec->dtype_b, spec->dtype_d, spec->mma_dtype, reason, reason_cap))
        return true;
    if(!rocke_conv_data_is_tf32(spec->mma_dtype))
        return false;
    if(spec->lds_k_outer)
        return ROCKE_CONV_TF32_FAIL(
            reason,
            reason_cap,
            "mma_dtype='tf32': lds_k_outer is not supported -- the K-outer operand fetch is a "
            "16-bit transpose read (ds_read_b64_tr_b16) with no 32-bit form, and TF32 keeps "
            "fp32 tiles in LDS");
    if(async_needs_k_outer && spec->async_dma)
        return ROCKE_CONV_TF32_FAIL(
            reason,
            reason_cap,
            "mma_dtype='tf32': async_dma is not supported -- this kernel's direct "
            "global->LDS load needs the K-outer tile, which TF32 cannot use");
    if(spec->lds_layout_swizzled)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32': a swizzled lds_layout is not supported -- "
                                    "the fragment is read as contiguous fp32 vectors");
    return rocke_conv_tf32_mfma_reason(spec, arch, reason, reason_cap);
}

bool rocke_conv_tf32_mfma_reason(const rocke_conv_tf32_spec_view_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    if(spec->wave_size != 64)
        return ROCKE_CONV_TF32_FAIL(reason,
                                    reason_cap,
                                    "mma_dtype='tf32' is MFMA-only; wave_size=%d is a WMMA target",
                                    spec->wave_size);
    int frag;
    if(arch == NULL)
    {
        frag = rocke_conv_tf32_any_arch_frag_len(spec);
        if(frag < 0)
            return ROCKE_CONV_TF32_FAIL(reason,
                                        reason_cap,
                                        "mma_dtype='tf32': warp_tile (%d, %d, %d) is not a TF32 "
                                        "atom on any arch",
                                        spec->warp_tile_m,
                                        spec->warp_tile_n,
                                        spec->warp_tile_k);
    }
    else
    {
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        if(target == NULL)
            /* ArchTarget.from_gfx raises KeyError; every caller resolves the
             * arch first, so this is the defensive form of that error. */
            return ROCKE_CONV_TF32_FAIL(reason, reason_cap, "unknown gfx target '%s'", arch);
        const char* family = target->wave_size == 32 ? "wmma" : "mma";
        if(rocke_conv_tf32_reason(spec, target, family, arch, reason, reason_cap))
            return true;
        const rocke_mma_op_t* op = rocke_archtarget_op_for_shape(target,
                                                                 "mma",
                                                                 "tf32",
                                                                 "tf32",
                                                                 "fp32",
                                                                 spec->warp_tile_m,
                                                                 spec->warp_tile_n,
                                                                 spec->warp_tile_k,
                                                                 nullptr);
        frag = op->srcs[0].frag_len;
    }
    if(spec->lds_layout_error != NULL)
        return ROCKE_CONV_TF32_FAIL(reason, reason_cap, "%s", spec->lds_layout_error);
    /* A and B share the layout, so one stride covers both tiles. */
    return rocke_conv_tf32_lds_stride_reason(spec->lds_row_stride, frag, reason, reason_cap);
}
