// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_spec_and_validate.c -- C99 port of the SPEC +
 * VALIDITY + SIGNATURE surface of
 * rocke/instances/common/conv_direct_grouped.py.
 *
 * This translation unit owns the "host-side, IR-free" value/property layer that
 * both kernels (16c / 4c) share. NONE of it calls the IR builder (rocke_b_*):
 *
 *   Python (conv_direct_grouped.py)            C99 (this file)
 *   --------------------------------------     ----------------------------------
 *   DirectConvProblem defaults                 rocke_direct_conv_problem_default()
 *     .total_c / .total_k / .flops             rocke_direct_conv_problem_total_c/...
 *     .short()                                 rocke_direct_conv_problem_short()
 *   DirectConv16cSpec defaults                 rocke_direct_conv_16c_spec_default()
 *     .threads_per_block / .n_acc_slots        rocke_direct_conv_16c_*
 *     .kernel_name() / .validate()             rocke_direct_conv_16c_kernel_name / _validate
 *   DirectConv4cSpec defaults                  rocke_direct_conv_4c_spec_default()
 *     .threads_per_block                       rocke_direct_conv_4c_threads_per_block
 *     .kernel_name() / .validate()             rocke_direct_conv_4c_kernel_name / _validate
 *   is_valid_spec_16c(spec, arch)              rocke_direct_conv_16c_is_valid_spec()
 *   is_valid_spec_4c(spec, arch)               rocke_direct_conv_4c_is_valid_spec()
 *
 * The reason strings + the kernel name are formatted byte-identically to Python
 * (kernel_name_join, the ValueError messages) so a sweep driver sees the same
 * accept/reject and the same kernel identifier. The IR-emitting builders + their
 * phase closures live in the sibling TUs that bind to
 * rocke/instance_conv_direct_grouped_internal.h.
 */

#include "rocke/instance_conv_direct_grouped.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, has_shape */
#include "rocke/helper_rocke.helpers.fuse.h" /* rocke_fuse_dtype_to_ir_str          */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, sig entry   */
#include "rocke/instance_conv_direct_grouped_internal.h" /* ROCKE_DCONV4C_* bounds */

/* ===================================================================== *
 *  DirectConvProblem
 * ===================================================================== */

rocke_direct_conv_problem_t rocke_direct_conv_problem_default(void)
{
    rocke_direct_conv_problem_t p;
    memset(&p, 0, sizeof(p));
    /* Required dims (N,H,W,groups,cpg,kpg) have no Python default -> 0.    */
    p.N = 0;
    p.H = 0;
    p.W = 0;
    p.groups = 0;
    p.cpg = 0;
    p.kpg = 0;
    /* Dataclass defaults. */
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.stride = 1;
    p.dtype = "fp16";
    p.dil_h = 1;
    p.dil_w = 1;
    return p;
}

int rocke_direct_conv_problem_total_c(const rocke_direct_conv_problem_t* p)
{
    return p->groups * p->cpg;
}

int rocke_direct_conv_problem_total_k(const rocke_direct_conv_problem_t* p)
{
    return p->groups * p->kpg;
}

/* Dilated filter extent (KH - 1) * dil + 1. A dilation below 1 is not
 * normalised: every validator rejects it (dilation_reason, and the non-grouped
 * kernel's positivity check). */
static int dconv_dilated_extent(int k, int dil)
{
    return (k - 1) * dil + 1;
}

int rocke_direct_conv_problem_Ho(const rocke_direct_conv_problem_t* p)
{
    return (p->H + 2 * p->PAD - dconv_dilated_extent(p->KH, p->dil_h)) / p->stride + 1;
}

int rocke_direct_conv_problem_Wo(const rocke_direct_conv_problem_t* p)
{
    return (p->W + 2 * p->PAD - dconv_dilated_extent(p->KW, p->dil_w)) / p->stride + 1;
}

long long rocke_direct_conv_problem_flops(const rocke_direct_conv_problem_t* p)
{
    /* 2 * N * H * W * groups * kpg * KH * KW * cpg, accumulated in int64. */
    long long f = 2;
    f *= (long long)p->N;
    f *= (long long)p->H;
    f *= (long long)p->W;
    f *= (long long)p->groups;
    f *= (long long)p->kpg;
    f *= (long long)p->KH;
    f *= (long long)p->KW;
    f *= (long long)p->cpg;
    return f;
}

rocke_status_t
    rocke_direct_conv_problem_short(const rocke_direct_conv_problem_t* p, char* out, size_t out_cap)
{
    int n;
    if(p == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* f"N{N}H{H}W{W}_g{groups}_c{cpg}k{kpg}" */
    n = snprintf(out, out_cap, "N%dH%dW%d_g%d_c%dk%d", p->N, p->H, p->W, p->groups, p->cpg, p->kpg);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  DirectConv16cSpec
 * ===================================================================== */

rocke_direct_conv_16c_spec_t rocke_direct_conv_16c_spec_default(void)
{
    rocke_direct_conv_16c_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_conv_16c";
    s.block_q = 16;
    s.block_groups = 8;
    s.wave_size = 64;
    s.double_buffer = true;
    s.fold_k32 = true;
    return s;
}

int rocke_direct_conv_16c_threads_per_block(const rocke_direct_conv_16c_spec_t* spec)
{
    /* block_groups * wave_size */
    return spec->block_groups * spec->wave_size;
}

int rocke_direct_conv_16c_n_acc_slots(const rocke_direct_conv_16c_spec_t* spec)
{
    /* problem.KH */
    return spec->problem.KH;
}

rocke_status_t rocke_direct_conv_16c_kernel_name(const rocke_direct_conv_16c_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char short_buf[128];
    char bq_buf[32];
    char bg_buf[32];
    const char* db_part;
    const char* parts[4];
    const char* flag_names[2];
    int flag_on[2];
    rocke_status_t st;
    const char* dtype;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* p.short() */
    st = rocke_direct_conv_problem_short(&spec->problem, short_buf, sizeof(short_buf));
    if(st != ROCKE_OK)
    {
        return st;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    db_part = spec->double_buffer ? "db" : "sb";

    /* kernel_name_join(name, short, "bq..", "bg..", "db"/"sb",
     *                  flags={"k32": fold_k32, "bf16": dtype=="bf16"}) */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    parts[3] = db_part;

    dtype = spec->problem.dtype ? spec->problem.dtype : "fp16";
    flag_names[0] = "k32";
    flag_on[0] = spec->fold_k32 ? 1 : 0;
    flag_names[1] = "bf16";
    flag_on[1] = (strcmp(dtype, "bf16") == 0) ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 2, out, out_cap, NULL);
}

/* Why a forward direct-conv kernel cannot compute `p`; false when it can.
 *
 * Mirrors conv_direct_grouped.forward_padding_reason. Every forward
 * direct-conv kernel streams input rows and flushes output row y - (KH-1)
 * while it is below H -- it indexes output rows by input rows. That is only
 * right under "same" padding PAD == (KH-1)/2 with an odd filter; any other
 * padding makes the kernel write wrong rows (including the next image's)
 * rather than fail, so it is refused before a binary is built. */
/* dilation_reason(p): why a direct-conv kernel without dilation support cannot
 * compute p. Only the non-grouped forward kernel computes a dilated conv. Any
 * dilation other than 1 -- zero and negatives included -- is refused, as in
 * Python's is_dilated. */
static bool
    rocke_direct_dilation_reason(const rocke_direct_conv_problem_t* p, char* why, size_t why_cap)
{
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        snprintf(why,
                 why_cap,
                 "this direct conv kernel has no dilation (got %dx%d); only the non-grouped "
                 "forward kernel computes a dilated convolution",
                 p->dil_h,
                 p->dil_w);
        return true;
    }
    return false;
}

static bool rocke_direct_forward_padding_reason(const rocke_direct_conv_problem_t* p,
                                                char* why,
                                                size_t why_cap)
{
    if(rocke_direct_dilation_reason(p, why, why_cap))
        return true;
    if(p->KH % 2 == 0 || p->KW % 2 == 0)
    {
        snprintf(
            why, why_cap, "forward direct conv needs odd filter extents (got %dx%d)", p->KH, p->KW);
        return true;
    }
    if(p->PAD != (p->KH - 1) / 2)
    {
        snprintf(why,
                 why_cap,
                 "forward direct conv needs 'same' padding PAD == (KH-1)/2 = %d (got PAD=%d); "
                 "the row stream indexes output rows by input rows",
                 (p->KH - 1) / 2,
                 p->PAD);
        return true;
    }
    return false;
}

rocke_status_t rocke_direct_conv_16c_validate(const rocke_direct_conv_16c_spec_t* spec,
                                              char* reason,
                                              size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConv16cSpec: unsupported dtype '%s'", dt);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    /* if p.cpg != 16 or p.kpg != 16: raise ValueError(...) */
    if(p->cpg != 16 || p->kpg != 16)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv16cSpec expects cpg=kpg=16 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    /* if p.groups % self.block_groups != 0: raise ValueError(...) */
    if(spec->block_groups == 0 || (p->groups % spec->block_groups) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

bool rocke_direct_conv_16c_is_valid_spec(const rocke_direct_conv_16c_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const rocke_direct_conv_problem_t* p;

#define CK_DCONV16C_REJECT(...)                        \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(spec == NULL)
    {
        CK_DCONV16C_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Full Python str(KeyError) text, reproduced verbatim. */
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }

    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            CK_DCONV16C_REJECT("unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
        }
    }
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    /* the fold_k32 16c kernel folds taps s=0/1 into one MFMA atom and handles s=2 as
     * a separate residual, so the filter has to be exactly three wide. */
    if(spec->fold_k32 && p->KW != 3)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "the fold_k32 16c kernel is built for a 3-wide filter (got KW=%d)",
                     p->KW);
        return false;
    }
    /* if p.cpg != 16 or p.kpg != 16: return False, ... */
    if(p->cpg != 16 || p->kpg != 16)
    {
        CK_DCONV16C_REJECT("DirectConv16cSpec expects cpg=kpg=16 (got %d, %d)", p->cpg, p->kpg);
    }
    /* if p.groups % spec.block_groups != 0: return False, ... */
    if(spec->block_groups == 0 || (p->groups % spec->block_groups) != 0)
    {
        CK_DCONV16C_REJECT(
            "groups %d not divisible by block_groups %d", p->groups, spec->block_groups);
    }

    mma = rocke_archtarget_mma(target);
    {
        /* ab_dtype: "f16" or "bf16" based on problem.dtype */
        const char* ab = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? "bf16" : "f16";
        /* if not target.mma.has_shape(ab,ab,fp32, 16,16,16): return False, ... */
        if(!rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 16))
        {
            CK_DCONV16C_REJECT("missing 16x16x16 %s MFMA atom on %s", ab, arch);
        }
        /* if spec.fold_k32 and not target.mma.has_shape(ab,ab,fp32, 16,16,32): ... */
        if(spec->fold_k32 && !rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 32))
        {
            CK_DCONV16C_REJECT("fold_k32=True needs the 16x16x32 %s MFMA atom, absent on %s; use "
                               "fold_k32=False for a %s-capable kernel",
                               ab,
                               arch,
                               arch);
        }
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef CK_DCONV16C_REJECT
}

/* ===================================================================== *
 *  DirectConv4cSpec
 * ===================================================================== */

rocke_direct_conv_4c_spec_t rocke_direct_conv_4c_spec_default(void)
{
    rocke_direct_conv_4c_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_conv_4c";
    s.block_q = 4;
    s.block_groups = 16;
    s.wave_size = 64;
    s.dgrad_fused_weights = false;
    s.dgrad_weights_lds = false;
    s.stage_rows = false;
    s.waves_q = 1;
    return s;
}

int rocke_direct_conv_4c_threads_per_block(const rocke_direct_conv_4c_spec_t* spec)
{
    /* (block_groups // 16) * waves_q * wave_size */
    return (spec->block_groups / 16) * spec->waves_q * spec->wave_size;
}

/* Python _staged_4c_geometry. */
void rocke_dconv4c_staged_geometry(const rocke_direct_conv_4c_spec_t* spec,
                                   rocke_dconv4c_staged_geo_t* geo)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int threads = rocke_direct_conv_4c_threads_per_block(spec);
    const int wl_vecs = spec->block_groups * p->cpg * p->KH * p->KW * p->kpg / 8;
    const int wl_elems = ((wl_vecs + threads - 1) / threads) * threads * 8;

    geo->bc = spec->block_groups * p->cpg;
    geo->lds_w = spec->block_q + p->KW - 1;
    geo->vpc = geo->bc / 8;
    geo->row_stride = geo->bc + 32;
    geo->nchunk = geo->lds_w * geo->vpc;
    geo->passes = (geo->nchunk + threads - 1) / threads;
    geo->row_elems = ((geo->passes * threads + geo->vpc - 1) / geo->vpc) * geo->row_stride;
    geo->lds_bytes = (2 * geo->row_elems + wl_elems) * 2;
}

/* Python _stage_rows_reject_reason: true + reason when illegal. */
bool rocke_dconv4c_stage_rows_reject(const rocke_direct_conv_4c_spec_t* spec,
                                     char* reason,
                                     size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char buf[ROCKE_ERR_MSG_CAP];
    bool bad = true;

    buf[0] = '\0';
    if(spec->waves_q < 1)
    {
        snprintf(buf, sizeof buf, "waves_q must be >= 1 (got %d)", spec->waves_q);
    }
    else if(!spec->stage_rows)
    {
        if(spec->waves_q != 1)
            snprintf(
                buf, sizeof buf, "waves_q > 1 needs stage_rows (got waves_q=%d)", spec->waves_q);
        else
            bad = false;
    }
    else if(!(spec->dgrad_fused_weights && spec->dgrad_weights_lds))
    {
        snprintf(buf, sizeof buf, "stage_rows needs dgrad_fused_weights and dgrad_weights_lds");
    }
    else if(spec->block_q != 4 * spec->waves_q)
    {
        snprintf(buf,
                 sizeof buf,
                 "stage_rows needs block_q == 4*waves_q (got block_q=%d, waves_q=%d)",
                 spec->block_q,
                 spec->waves_q);
    }
    else if(p->stride != 1 || p->KH != 2 * p->PAD + 1 || p->KW != 2 * p->PAD + 1)
    {
        snprintf(buf,
                 sizeof buf,
                 "stage_rows needs stride 1 and 'same' padding (Ho == H, Wo == W: "
                 "KH == KW == 2*PAD+1; got stride=%d, KH=%d, KW=%d, PAD=%d)",
                 p->stride,
                 p->KH,
                 p->KW,
                 p->PAD);
    }
    else if(rocke_direct_conv_4c_threads_per_block(spec) > ROCKE_DCONV4C_MAX_THREADS)
    {
        snprintf(buf,
                 sizeof buf,
                 "stage_rows needs threads_per_block <= %d (got %d)",
                 ROCKE_DCONV4C_MAX_THREADS,
                 rocke_direct_conv_4c_threads_per_block(spec));
    }
    else
    {
        bad = false;
    }
    if(bad && reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", buf);
    }
    return bad;
}

rocke_status_t rocke_direct_conv_4c_kernel_name(const rocke_direct_conv_4c_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char short_buf[128];
    char bq_buf[32];
    char bg_buf[32];
    const char* parts[3];
    rocke_status_t st;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* p.short() */
    st = rocke_direct_conv_problem_short(&spec->problem, short_buf, sizeof(short_buf));
    if(st != ROCKE_OK)
    {
        return st;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);

    /* kernel_name_join(name, short, "bq..", "bg..",
     *                  flags={"bf16", "fw", "fwl", f"sr{waves_q}"}) */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    {
        const char* dt4c = spec->problem.dtype ? spec->problem.dtype : "fp16";
        const char* flag_names4c[4];
        int flag_on4c[4];
        char sr_buf[32];
        snprintf(sr_buf, sizeof(sr_buf), "sr%d", spec->waves_q);
        flag_names4c[0] = "bf16";
        flag_names4c[1] = "fw";
        flag_names4c[2] = "fwl";
        flag_names4c[3] = sr_buf;
        flag_on4c[0] = (strcmp(dt4c, "bf16") == 0) ? 1 : 0;
        flag_on4c[1] = (spec->dgrad_fused_weights && !spec->dgrad_weights_lds) ? 1 : 0;
        flag_on4c[2] = (spec->dgrad_fused_weights && spec->dgrad_weights_lds) ? 1 : 0;
        flag_on4c[3] = spec->stage_rows ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names4c, flag_on4c, 4, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_conv_4c_validate(const rocke_direct_conv_4c_spec_t* spec,
                                             char* reason,
                                             size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConv4cSpec: unsupported dtype '%s'", dt);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    /* if p.cpg != 4 or p.kpg != 4: raise ValueError(...) */
    if(p->cpg != 4 || p->kpg != 4)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv4cSpec expects cpg=kpg=4 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    /* if self.block_groups % 16 != 0: raise ValueError("...") */
    if((spec->block_groups % 16) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv4cSpec block_groups must be a multiple of 16");
        }
        return ROCKE_ERR_VALUE;
    }
    /* if self.block_q % 4 != 0: raise ValueError("...") */
    if((spec->block_q % 4) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv4cSpec block_q must be a multiple of 4");
        }
        return ROCKE_ERR_VALUE;
    }
    /* if p.groups % self.block_groups != 0: raise ValueError(...) */
    if(spec->block_groups == 0 || (p->groups % spec->block_groups) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return ROCKE_ERR_VALUE;
    }
    /* if self.dgrad_weights_lds and not self.dgrad_fused_weights: raise */
    if(spec->dgrad_weights_lds && !spec->dgrad_fused_weights)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv4cSpec dgrad_weights_lds requires dgrad_fused_weights");
        }
        return ROCKE_ERR_VALUE;
    }
    /* why = _stage_rows_reject_reason(self); raise ValueError(f"DirectConv4cSpec {why}") */
    {
        char why[ROCKE_ERR_MSG_CAP];
        if(rocke_dconv4c_stage_rows_reject(spec, why, sizeof why))
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConv4cSpec %s", why);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    return ROCKE_OK;
}

bool rocke_direct_conv_4c_is_valid_spec(const rocke_direct_conv_4c_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;

#define CK_DCONV4C_REJECT(...)                         \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(spec == NULL)
    {
        CK_DCONV4C_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* try: ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Full Python str(KeyError) text, reproduced verbatim. */
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }

    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): return False, ... */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            CK_DCONV4C_REJECT("unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
        }
    }
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    /* if p.cpg != 4 or p.kpg != 4: return False, ... */
    if(p->cpg != 4 || p->kpg != 4)
    {
        CK_DCONV4C_REJECT("DirectConv4cSpec expects cpg=kpg=4 (got %d, %d)", p->cpg, p->kpg);
    }
    /* if spec.block_groups % 16 != 0: return False, ... */
    if((spec->block_groups % 16) != 0)
    {
        CK_DCONV4C_REJECT("DirectConv4cSpec block_groups must be a multiple of 16");
    }
    /* if spec.block_q % 4 != 0: return False, ... */
    if((spec->block_q % 4) != 0)
    {
        CK_DCONV4C_REJECT("DirectConv4cSpec block_q must be a multiple of 4");
    }
    /* if p.groups % spec.block_groups != 0: return False, ... */
    if(spec->block_groups == 0 || (p->groups % spec->block_groups) != 0)
    {
        CK_DCONV4C_REJECT(
            "groups %d not divisible by block_groups %d", p->groups, spec->block_groups);
    }
    /* if p.KH * p.KW > DCONV4C_MAX_TAPS: return False, ... (the builder's
     * per-tap arrays are sized ROCKE_DCONV4C_MAX_TAPS). */
    if(p->KH * p->KW > ROCKE_DCONV4C_MAX_TAPS)
    {
        CK_DCONV4C_REJECT("DirectConv4cSpec supports KH*KW <= %d (got %d)",
                          ROCKE_DCONV4C_MAX_TAPS,
                          p->KH * p->KW);
    }
    /* why = _stage_rows_reject_reason(spec); if why: return False, why */
    if(rocke_dconv4c_stage_rows_reject(spec, reason, reason_cap))
    {
        return false;
    }
    /* if spec.dgrad_weights_lds: needs fused weights + transpose LDS reads. */
    if(spec->dgrad_weights_lds)
    {
        if(!spec->dgrad_fused_weights)
        {
            CK_DCONV4C_REJECT("dgrad_weights_lds requires dgrad_fused_weights");
        }
        if(!target->memory.has_ds_read_tr)
        {
            CK_DCONV4C_REJECT("dgrad_weights_lds needs ds_read_b64_tr_b16 (absent on %s)", arch);
        }
        /* wl_passes = ceil(block_groups*cpg*KH*KW*kpg/8 / threads) <= bound */
        {
            const int wl_vecs = spec->block_groups * p->cpg * p->KH * p->KW * p->kpg / 8;
            const int wl_threads = rocke_direct_conv_4c_threads_per_block(spec);
            const int wl_passes = (wl_vecs + wl_threads - 1) / wl_threads;
            if(wl_passes > ROCKE_DCONV4C_MAX_WL_PASSES)
            {
                CK_DCONV4C_REJECT("dgrad_weights_lds needs %d staging passes (max %d)",
                                  wl_passes,
                                  ROCKE_DCONV4C_MAX_WL_PASSES);
            }
        }
    }
    /* stage_rows: row-staging pass bound and the target's LDS capacity. */
    if(spec->stage_rows)
    {
        rocke_dconv4c_staged_geo_t geo;
        rocke_dconv4c_staged_geometry(spec, &geo);
        if(geo.passes > ROCKE_DCONV4C_MAX_ROW_PASSES)
        {
            CK_DCONV4C_REJECT("stage_rows needs %d row staging passes (max %d)",
                              geo.passes,
                              ROCKE_DCONV4C_MAX_ROW_PASSES);
        }
        if(geo.lds_bytes > target->lds_capacity_bytes)
        {
            CK_DCONV4C_REJECT("stage_rows needs %d bytes of LDS (more than %d on %s)",
                              geo.lds_bytes,
                              target->lds_capacity_bytes,
                              arch);
        }
    }

    /* The 4x4x4 atom is deliberately NOT gated through has_shape (catalog lists
     * only the warp-tile shapes; comgr selects the 4x4x4 intrinsic on both
     * targets). `target` is validated for resolution only. */
    (void)target;

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef CK_DCONV4C_REJECT
}

/* The launch signature is not built here: every direct kernel takes the AOT
 * argument list of rocke_conv_direct_arg_names() (conv_abi.cpp), which hosts
 * pack from directly. */

/* ===================================================================== *
 *  DirectConv8cSpec  (cpg = kpg = 8)
 * ===================================================================== */

rocke_direct_conv_8c_spec_t rocke_direct_conv_8c_spec_default(void)
{
    rocke_direct_conv_8c_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_conv_8c";
    spec.block_q = 16;
    spec.block_groups = 8;
    spec.wave_size = 64;
    spec.double_buffer = true;
    return spec;
}

int rocke_direct_conv_8c_threads_per_block(const rocke_direct_conv_8c_spec_t* spec)
{
    return spec->block_groups * spec->wave_size;
}

rocke_status_t rocke_direct_conv_8c_kernel_name(const rocke_direct_conv_8c_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char prob_short[128];
    const char* parts[4];
    char bq_buf[24];
    char bg_buf[24];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    parts[0] = prob_short;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    parts[3] = spec->double_buffer ? "db" : "sb";
    {
        /* flags={"bf16": dtype=="bf16"} */
        const char* flag_names8c[1] = {"bf16"};
        const char* dt8c = spec->problem.dtype ? spec->problem.dtype : "fp16";
        int flag_on8c[1];
        flag_on8c[0] = (strcmp(dt8c, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 4, flag_names8c, flag_on8c, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_conv_8c_validate(const rocke_direct_conv_8c_spec_t* spec,
                                             char* reason,
                                             size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConv8cSpec: unsupported dtype '%s'", dt);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 8 || p->kpg != 8)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv8cSpec expects cpg=kpg=8 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    if(p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return ROCKE_ERR_VALUE;
    }
    if(spec->block_q % 16 != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv8cSpec block_q must be a multiple of 16");
        }
        return ROCKE_ERR_VALUE;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

bool rocke_direct_conv_8c_is_valid_spec(const rocke_direct_conv_8c_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const char* ab;

    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(
                    reason, reason_cap, "unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
            }
            return false;
        }
    }
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    /* the 8c kernel folds taps s=0/1 into one MFMA atom and handles s=2 as
     * a separate residual, so the filter has to be exactly three wide. */
    if(p->KW != 3)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "the 8c kernel is built for a 3-wide filter (got KW=%d)",
                     p->KW);
        return false;
    }
    if(p->cpg != 8 || p->kpg != 8)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv8cSpec expects cpg=kpg=8 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return false;
    }
    if(p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return false;
    }
    if(spec->block_q % 16 != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv8cSpec block_q must be a multiple of 16");
        }
        return false;
    }
    mma = rocke_archtarget_mma(target);
    ab = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 16))
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "missing 16x16x16 %s MFMA atom on %s", ab, arch);
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectConv32cSpec  (cpg = kpg = 32)
 * ===================================================================== */

rocke_direct_conv_32c_spec_t rocke_direct_conv_32c_spec_default(void)
{
    rocke_direct_conv_32c_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_conv_32c";
    spec.block_q = 32;
    spec.block_groups = 4;
    spec.wave_size = 64;
    spec.double_buffer = true;
    return spec;
}

int rocke_direct_conv_32c_threads_per_block(const rocke_direct_conv_32c_spec_t* spec)
{
    return spec->block_groups * spec->wave_size;
}

rocke_status_t rocke_direct_conv_32c_kernel_name(const rocke_direct_conv_32c_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char prob_short[128];
    const char* parts[4];
    char bq_buf[24];
    char bg_buf[24];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    parts[0] = prob_short;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    parts[3] = spec->double_buffer ? "db" : "sb";
    {
        /* flags={"bf16": dtype=="bf16"} */
        const char* flag_names32c[1] = {"bf16"};
        const char* dt32c = spec->problem.dtype ? spec->problem.dtype : "fp16";
        int flag_on32c[1];
        flag_on32c[0] = (strcmp(dt32c, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 4, flag_names32c, flag_on32c, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_conv_32c_validate(const rocke_direct_conv_32c_spec_t* spec,
                                              char* reason,
                                              size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConv32cSpec: unsupported dtype '%s'", dt);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 32 || p->kpg != 32)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv32cSpec expects cpg=kpg=32 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    if(p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return ROCKE_ERR_VALUE;
    }
    if(spec->block_q % 32 != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv32cSpec block_q must be a multiple of 32");
        }
        return ROCKE_ERR_VALUE;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

bool rocke_direct_conv_32c_is_valid_spec(const rocke_direct_conv_32c_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const char* ab;

    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(
                    reason, reason_cap, "unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
            }
            return false;
        }
    }
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    if(p->cpg != 32 || p->kpg != 32)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectConv32cSpec expects cpg=kpg=32 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return false;
    }
    if(p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return false;
    }
    if(spec->block_q % 32 != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv32cSpec block_q must be a multiple of 32");
        }
        return false;
    }
    mma = rocke_archtarget_mma(target);
    ab = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 32, 32, 8))
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "missing 32x32x8 %s MFMA atom on %s", ab, arch);
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectDepthwiseSpec  (cpg = kpg = 1)
 * ===================================================================== */

rocke_direct_depthwise_spec_t rocke_direct_depthwise_spec_default(void)
{
    rocke_direct_depthwise_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise";
    spec.block_w = 8;
    spec.block_waves = 1;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_threads_per_block(const rocke_direct_depthwise_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_block_ch(const rocke_direct_depthwise_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

rocke_status_t rocke_direct_depthwise_kernel_name(const rocke_direct_depthwise_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char prob_short[128];
    const char* parts[3];
    char bw_buf[24];
    char bwv_buf[24];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"bw{block_w}", f"bw{block_waves}wv",
     *                  flags={"bf16": dtype=="bf16"}) */
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = bw_buf;
    parts[2] = bwv_buf;
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_depthwise_validate(const rocke_direct_depthwise_spec_t* spec,
                                               char* reason,
                                               size_t reason_cap)
{
    int block_ch;
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseSpec: unsupported dtype '%s'; expected fp16 or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseSpec requires cpg=kpg=1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    block_ch = rocke_direct_depthwise_block_ch(spec);
    (void)block_ch;
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

/* Mirrors Python _DW_MAX_PRELOAD_TAPS in
 * library/kernels/common/conv_direct_grouped.py. */
#define ROCKE_DCONV_DW_MAX_PRELOAD_TAPS 200

bool rocke_direct_depthwise_is_valid_spec(const rocke_direct_depthwise_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    int block_ch;
    int taps;
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "cpg and kpg must both be 1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        }
        return false;
    }
    if(p->KH < 1 || p->KH > ROCKE_DCONV_DW_MAX_KH)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "KH must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KH, p->KH);
        }
        return false;
    }
    if(p->KW < 1 || p->KW > ROCKE_DCONV_DW_MAX_KW)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "KW must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KW, p->KW);
        }
        return false;
    }
    /* Python _DW_MAX_PRELOAD_TAPS: this variant hoists all KH*KW weights into
     * registers before the H sweep, so the filter area is the register-pressure
     * bound. Past the cap the caller must use DirectDepthwiseColSpec. */
    taps = p->KH * p->KW;
    if(taps > ROCKE_DCONV_DW_MAX_PRELOAD_TAPS)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "preloaded filter of %d taps (= KH*KW = %d*%d) exceeds max %d; "
                     "use DirectDepthwiseColSpec, whose live-register cost is linear "
                     "in KH and independent of KW",
                     taps,
                     p->KH,
                     p->KW,
                     ROCKE_DCONV_DW_MAX_PRELOAD_TAPS);
        }
        return false;
    }
    block_ch = rocke_direct_depthwise_block_ch(spec);
    (void)block_ch;
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectDepthwiseSpatialSpec implementations
 * ===================================================================== */

rocke_direct_depthwise_spatial_spec_t rocke_direct_depthwise_spatial_spec_default(void)
{
    rocke_direct_depthwise_spatial_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_spatial";
    spec.block_waves = 1;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_spatial_n_w_per_wave(const rocke_direct_depthwise_spatial_spec_t* spec)
{
    int groups = spec->problem.groups;
    if(groups <= 0)
    {
        return 0;
    }
    return spec->wave_size / groups;
}

int rocke_direct_depthwise_spatial_block_w(const rocke_direct_depthwise_spatial_spec_t* spec)
{
    return spec->block_waves * rocke_direct_depthwise_spatial_n_w_per_wave(spec);
}

int rocke_direct_depthwise_spatial_threads_per_block(
    const rocke_direct_depthwise_spatial_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

rocke_status_t rocke_direct_depthwise_spatial_kernel_name(
    const rocke_direct_depthwise_spatial_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    const char* parts[2];
    char bwv_buf[32];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"bwv{block_waves}", flags={"bf16": dtype=="bf16"}) */
    snprintf(bwv_buf, sizeof(bwv_buf), "bwv%d", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = bwv_buf;
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 2, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

bool rocke_direct_depthwise_spatial_is_valid_spec(const rocke_direct_depthwise_spatial_spec_t* spec,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    int n_w;

    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char why_[256];
        if(rocke_direct_forward_padding_reason(p, why_, sizeof why_))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", why_);
            return false;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "cpg and kpg must both be 1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        }
        return false;
    }
    if(p->KH < 1 || p->KH > ROCKE_DCONV_DW_MAX_KH)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "KH must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KH, p->KH);
        }
        return false;
    }
    if(p->KW < 1 || p->KW > ROCKE_DCONV_DW_MAX_KW)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "KW must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KW, p->KW);
        }
        return false;
    }
    if(p->groups > spec->wave_size)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "groups %d > wave_size %d", p->groups, spec->wave_size);
        }
        return false;
    }
    n_w = rocke_direct_depthwise_spatial_n_w_per_wave(spec);
    if(n_w <= 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups == wave_size: no W positions per wave (n_w_per_wave=0)");
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

rocke_status_t rocke_direct_depthwise_spatial_validate(
    const rocke_direct_depthwise_spatial_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseSpatialSpec: unsupported dtype '%s'; "
                         "expected fp16 or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseSpatialSpec requires cpg=kpg=1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        return ROCKE_ERR_VALUE;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  DirectDepthwiseColSpec  (column-streamed depthwise)
 * ===================================================================== */

/* Python's `//` floors; C's `/` truncates toward zero. DirectConvProblem.Ho/.Wo
 * use `//`, and the validator reports those values for degenerate geometries
 * where the numerator IS negative, so the difference is observable. */
static int rocke_dconv_col__floor_div(int a, int b)
{
    int q;
    if(b == 0)
    {
        return 0;
    }
    q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
    {
        --q;
    }
    return q;
}

/* DirectConvProblem.Ho / .Wo. */
static int rocke_dconv_col__Ho(const rocke_direct_conv_problem_t* p)
{
    return rocke_dconv_col__floor_div(p->H + 2 * p->PAD - p->KH, p->stride) + 1;
}

static int rocke_dconv_col__Wo(const rocke_direct_conv_problem_t* p)
{
    return rocke_dconv_col__floor_div(p->W + 2 * p->PAD - p->KW, p->stride) + 1;
}

/* Python `f"dtype {spec.dtype!r}"`: repr of a str is quoted, repr of None is
 * the bare word None. A NULL const char* stands in for Python's None. */
static void rocke_dconv_col__dtype_repr(const char* dtype, char* out, size_t out_cap)
{
    if(out == NULL || out_cap == 0)
    {
        return;
    }
    if(dtype == NULL)
    {
        snprintf(out, out_cap, "None");
    }
    else
    {
        snprintf(out, out_cap, "'%s'", dtype);
    }
}

rocke_direct_depthwise_col_spec_t rocke_direct_depthwise_col_spec_default(void)
{
    rocke_direct_depthwise_col_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_col";
    spec.block_w = 1;
    spec.block_waves = 1;
    spec.wave_size = 64;
    spec.dtype = "fp16";
    spec.max_live_f32 = 0; /* Python None */
    spec.block_h = 16;
    return spec;
}

int rocke_direct_depthwise_col_threads_per_block(const rocke_direct_depthwise_col_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_col_block_ch(const rocke_direct_depthwise_col_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_col_n_iters(const rocke_direct_depthwise_col_spec_t* spec)
{
    return (spec->block_h - 1) * spec->problem.stride + spec->problem.KH;
}

int rocke_direct_depthwise_col_live_f32(const rocke_direct_depthwise_col_spec_t* spec)
{
    return spec->block_h * spec->block_w + spec->problem.KH;
}

int rocke_direct_depthwise_col_resolve_max_live_f32(const rocke_direct_depthwise_col_spec_t* spec,
                                                    const char* arch)
{
    const rocke_archtarget_t* target;
    int budget;
    if(spec == NULL)
    {
        return 0;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        return 0;
    }
    budget = target->limits.vgprs * 3 / 8; /* assumes vgprs % 8 == 0 (holds for all CDNAx) */
    if(spec->max_live_f32 == 0)
    {
        return budget; /* 0 is the sentinel for Python None: take the arch budget as-is */
    }
    return (spec->max_live_f32 < budget) ? spec->max_live_f32 : budget;
}

rocke_status_t rocke_direct_depthwise_col_dtype_tag(const rocke_direct_depthwise_col_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap)
{
    const rocke_type_t* dt;
    const char* s;
    size_t i;
    size_t n;
    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    dt = (spec->dtype != NULL) ? rocke_fuse_dtype_to_ir_str(spec->dtype) : NULL;
    if(dt != NULL)
    {
        if(strlen(dt->name) + 1 > out_cap)
        {
            return ROCKE_ERR_VALUE;
        }
        strcpy(out, dt->name);
        return ROCKE_OK;
    }
    /* Python fallback: "".join(c if c.isalnum() else "_" for c in str(self.dtype)) */
    s = (spec->dtype != NULL) ? spec->dtype : "None";
    n = strlen(s);
    if(n + 1 > out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    for(i = 0; i < n; ++i)
    {
        char c = s[i];
        bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        out[i] = alnum ? c : '_';
    }
    out[n] = '\0';
    return ROCKE_OK;
}

rocke_status_t rocke_direct_depthwise_col_kernel_name(const rocke_direct_depthwise_col_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap)
{
    char prob_short[128];
    char r_buf[32];
    char p_buf[24];
    char s_buf[24];
    char dt_buf[64];
    char bh_buf[24];
    char bw_buf[24];
    char bwv_buf[24];
    const char* parts[8];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_depthwise_col_dtype_tag(spec, dt_buf, sizeof(dt_buf)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"r{KH}x{KW}", f"p{PAD}", f"s{stride}",
     *                  dtype_tag(), f"bh{block_h}", f"bw{block_w}",
     *                  f"bw{block_waves}wv") */
    snprintf(r_buf, sizeof(r_buf), "r%dx%d", spec->problem.KH, spec->problem.KW);
    snprintf(p_buf, sizeof(p_buf), "p%d", spec->problem.PAD);
    snprintf(s_buf, sizeof(s_buf), "s%d", spec->problem.stride);
    snprintf(bh_buf, sizeof(bh_buf), "bh%d", spec->block_h);
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = r_buf;
    parts[2] = p_buf;
    parts[3] = s_buf;
    parts[4] = dt_buf;
    parts[5] = bh_buf;
    parts[6] = bw_buf;
    parts[7] = bwv_buf;
    return rocke_kernel_name_join(spec->name, parts, 8, NULL, NULL, 0, out, out_cap, NULL);
}

/* Prerequisite check only (cpg=kpg=1, block_h > 0).
 * Call rocke_direct_depthwise_col_is_valid_spec() for the full constraint set
 * (dtype, geometry, VGPR budget) before dispatch. */
rocke_status_t rocke_direct_depthwise_col_validate(const rocke_direct_depthwise_col_spec_t* spec,
                                                   char* reason,
                                                   size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseColSpec requires cpg=kpg=1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    /* The tap indexing has no dilation term. */
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return ROCKE_ERR_VALUE;
        }
    }
    if(spec->block_h <= 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseColSpec block_h must be > 0 (got %d): the input-row "
                     "loop is unrolled at build time, so an untiled kernel would have to "
                     "bake the output height into its trip count and could not serve "
                     "other shapes",
                     spec->block_h);
        }
        return ROCKE_ERR_VALUE;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

bool rocke_direct_depthwise_col_is_valid_spec(const rocke_direct_depthwise_col_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    const rocke_type_t* dt;
    char dtype_repr[96];
    int Ho;
    int Wo;
    int max_threads;
    int threads;
    int live;
    int allowed;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
            reason[reason_cap - 1] = '\0';
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    /* dtype allow-list: fp16 / bf16 (aliases resolved by dtype_to_ir). */
    dt = (spec->dtype != NULL) ? rocke_fuse_dtype_to_ir_str(spec->dtype) : NULL;
    if(dt == NULL || !(strcmp(dt->name, "f16") == 0 || strcmp(dt->name, "bf16") == 0))
    {
        if(reason && reason_cap > 0)
        {
            rocke_dconv_col__dtype_repr(spec->dtype, dtype_repr, sizeof(dtype_repr));
            snprintf(
                reason, reason_cap, "dtype %s is not supported; expected fp16 or bf16", dtype_repr);
        }
        return false;
    }
    p = &spec->problem;
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "cpg and kpg must both be 1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        }
        return false;
    }
    /* The tap indexing has no dilation term. */
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    if(p->stride < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "stride must be >= 1 (got %d)", p->stride);
        }
        return false;
    }
    if(p->KH < 1 || p->KW < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(
                reason, reason_cap, "filter extents must be >= 1 (got KH=%d, KW=%d)", p->KH, p->KW);
        }
        return false;
    }
    if(p->PAD < 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "PAD must be >= 0 (got %d)", p->PAD);
        }
        return false;
    }
    Ho = rocke_dconv_col__Ho(p);
    Wo = rocke_dconv_col__Wo(p);
    if(Ho < 1 || Wo < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "filter does not fit the padded input: Ho=%d, Wo=%d "
                     "(H=%d, W=%d, KH=%d, KW=%d, PAD=%d, stride=%d)",
                     Ho,
                     Wo,
                     p->H,
                     p->W,
                     p->KH,
                     p->KW,
                     p->PAD,
                     p->stride);
        }
        return false;
    }
    /* Over-padded configs (PAD > (KH-1)/2) at stride=1 push Ho > H.
     * They are geometrically valid at stride>1 but are rejected here because
     * the current builder does not need them. Remove if a use-case arises. */
    if(Ho > p->H)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "requires Ho <= H (got Ho=%d, H=%d)", Ho, p->H);
        }
        return false;
    }
    /* PAD >= KH is degenerate -- the first output row's receptive field is then
     * entirely padding, so it is identically zero -- and at stride > 1 the
     * Ho <= H check above no longer bounds it. */
    if(p->PAD >= p->KH || p->PAD >= p->KW)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "PAD %d must be < min(KH, KW) = %d; at or beyond the filter extent "
                     "the first output row reads only padding",
                     p->PAD,
                     (p->KH < p->KW) ? p->KH : p->KW);
        }
        return false;
    }
    if(spec->block_h < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "block_h must be >= 1 (got %d)", spec->block_h);
        }
        return false;
    }
    if(spec->max_live_f32 < 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "max_live_f32 must be 0 (Python None, use arch budget) or positive "
                     "(got %d)",
                     spec->max_live_f32);
        }
        return false;
    }
    if(spec->block_w < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "block_w must be >= 1 (got %d)", spec->block_w);
        }
        return false;
    }
    if(spec->block_w > Wo)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "block_w %d > Wo %d; reduce block_w to avoid wasted masked loads",
                     spec->block_w,
                     Wo);
        }
        return false;
    }
    if(spec->block_waves < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "block_waves must be >= 1 (got %d)", spec->block_waves);
        }
        return false;
    }
    /* The builder derives the per-lane channel index from wave_size, so a spec
     * whose wave_size disagrees with the target's would emit a kernel that
     * silently reads the wrong channel; wave_size=0 divides by zero outright. */
    if(spec->wave_size != target->wave_size)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "wave_size %d does not match the %s wave_size %d",
                     spec->wave_size,
                     arch,
                     target->wave_size);
        }
        return false;
    }
    max_threads = target->limits.max_threads_per_block;
    threads = rocke_direct_depthwise_col_threads_per_block(spec);
    if(threads > max_threads)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "threads_per_block %d (= block_waves*wave_size = %d*%d) exceeds the "
                     "%s limit of %d",
                     threads,
                     spec->block_waves,
                     spec->wave_size,
                     arch,
                     max_threads);
        }
        return false;
    }
    live = rocke_direct_depthwise_col_live_f32(spec);
    allowed = rocke_direct_depthwise_col_resolve_max_live_f32(spec, arch);
    if(live > allowed)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "live f32 per lane %d (= block_h*block_w + KH = %d*%d + %d) exceeds "
                     "max_live_f32=%d on %s; lower block_h or block_w",
                     live,
                     spec->block_h,
                     spec->block_w,
                     p->KH,
                     allowed,
                     arch);
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectConvDgradSpec  (grouped dgrad — scalar FMA)
 * ===================================================================== */

rocke_direct_conv_dgrad_spec_t rocke_direct_conv_dgrad_spec_default(void)
{
    rocke_direct_conv_dgrad_spec_t spec;
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_conv_dgrad";
    spec.block_q = 16;
    spec.block_groups = 8;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_conv_dgrad_threads_per_block(const rocke_direct_conv_dgrad_spec_t* spec)
{
    return spec->block_groups * spec->wave_size;
}

rocke_status_t rocke_direct_conv_dgrad_kernel_name(const rocke_direct_conv_dgrad_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char prob_short[128];
    const char* parts[3];
    char bq_buf[32];
    char bg_buf[32];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"bq{block_q}", f"bg{block_groups}",
     *                  flags={"bf16": dtype=="bf16"}) */
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    parts[0] = prob_short;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_conv_dgrad_validate(const rocke_direct_conv_dgrad_spec_t* spec,
                                                char* reason,
                                                size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "DirectConvDgradSpec: unsupported dtype '%s'", dt);
            }
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConvDgradSpec requires cpg >= 1 (got %d)", p->cpg);
        }
        return ROCKE_ERR_VALUE;
    }
    if(p->kpg < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConvDgradSpec requires kpg >= 1 (got %d)", p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    if(spec->block_groups > 0 && p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

bool rocke_direct_conv_dgrad_is_valid_spec(const rocke_direct_conv_dgrad_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    /* if p.dtype not in ("fp16", "bf16"): return False, ... */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
            {
                snprintf(
                    reason, reason_cap, "unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
            }
            return false;
        }
    }
    if(p->cpg < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "cpg must be >= 1 (got %d)", p->cpg);
        }
        return false;
    }
    if(p->kpg < 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "kpg must be >= 1 (got %d)", p->kpg);
        }
        return false;
    }
    if(spec->block_groups > 0 && p->groups % spec->block_groups != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "groups %d not divisible by block_groups %d",
                     p->groups,
                     spec->block_groups);
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectDepthwiseDgradSpec  (cpg=kpg=1 dgrad — scalar FMA)
 * ===================================================================== */

rocke_direct_depthwise_dgrad_spec_t rocke_direct_depthwise_dgrad_spec_default(void)
{
    rocke_direct_depthwise_dgrad_spec_t spec;
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_dgrad";
    spec.block_w = 8;
    spec.block_waves = 1;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_dgrad_threads_per_block(const rocke_direct_depthwise_dgrad_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_block_ch(const rocke_direct_depthwise_dgrad_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

rocke_status_t rocke_direct_depthwise_dgrad_kernel_name(
    const rocke_direct_depthwise_dgrad_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    const char* parts[3];
    char bw_buf[32];
    char bwv_buf[32];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"bw{block_w}", f"bw{block_waves}wv",
     *                  flags={"bf16": dtype=="bf16"}) */
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = bw_buf;
    parts[2] = bwv_buf;
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_depthwise_dgrad_validate(
    const rocke_direct_depthwise_dgrad_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseDgradSpec: unsupported dtype '%s'; "
                         "expected fp16 or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseDgradSpec requires cpg=kpg=1 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        }
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

bool rocke_direct_depthwise_dgrad_is_valid_spec(const rocke_direct_depthwise_dgrad_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
        {
            strncpy(reason, "null spec", reason_cap);
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
        }
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectConvWgradSpec  (backward weights)
 * ===================================================================== */

/* The row loop walks INPUT rows and pairs row hi with output row hi + PAD - r,
 * and one LDS strip row serves all KW s-taps by being read at a one-column
 * shift. Both identities hold only at stride 1. Python: _WGRAD_STRIDE_WHY. */
#define ROCKE_WGRAD_STRIDE_WHY                                                   \
    "direct wgrad is a stride-1 algorithm (input-row iteration + shifted S-row " \
    "strip); got stride=%d"

/* The lane->fragment mapping is the wave64 MFMA one (c4 = lane / 16 picks the
 * accumulator row group, lane % 16 the column) and ds_read_tr16_b64 hands back
 * a 64-lane fragment. There is no wave32 variant of either.
 * Python: _WGRAD_WAVE64_WHY. */
#define ROCKE_WGRAD_WAVE64_WHY                                                  \
    "direct wgrad needs wave_size 64 (wave64 MFMA fragment + ds_read_tr16_b64 " \
    "lane mapping); got wave_size=%d"

rocke_direct_conv_wgrad_spec_t rocke_direct_conv_wgrad_spec_default(void)
{
    rocke_direct_conv_wgrad_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_conv_wgrad";
    spec.wave_tile_k = 16;
    spec.wave_tile_c = 16;
    spec.waves_k = 1;
    spec.waves_c = 1;
    spec.waves_q = 1;
    spec.wave_size = 64;
    spec.ho_per_block = 4;
    spec.mfma_k = 32;
    return spec;
}

int rocke_direct_conv_wgrad_block_k(const rocke_direct_conv_wgrad_spec_t* spec)
{
    return spec->waves_k * spec->wave_tile_k;
}

int rocke_direct_conv_wgrad_block_c(const rocke_direct_conv_wgrad_spec_t* spec)
{
    return spec->waves_c * spec->wave_tile_c;
}

int rocke_direct_conv_wgrad_threads_per_block(const rocke_direct_conv_wgrad_spec_t* spec)
{
    return spec->waves_k * spec->waves_c * spec->waves_q * spec->wave_size;
}

int rocke_direct_conv_wgrad_wo_block(const rocke_direct_conv_wgrad_spec_t* spec)
{
    return spec->mfma_k;
}

int rocke_direct_conv_wgrad_n_ho_blocks(const rocke_direct_conv_wgrad_spec_t* spec)
{
    /* INPUT height: the builder decodes `by` as an input-row block
     * (hi_block_start = by * HPB) and the row loop walks hi. Mirrors
     * DirectConvWgradSpec.n_ho_blocks. */
    return (spec->problem.H + spec->ho_per_block - 1) / spec->ho_per_block;
}

int rocke_direct_conv_wgrad_n_wo_tiles(const rocke_direct_conv_wgrad_spec_t* spec)
{
    int Wo = rocke_direct_conv_problem_Wo(&spec->problem);
    int wo_block = rocke_direct_conv_wgrad_wo_block(spec);
    return (Wo + wo_block - 1) / wo_block;
}

int rocke_direct_conv_wgrad_n_q_blocks(const rocke_direct_conv_wgrad_spec_t* spec)
{
    int n_wo_tiles = rocke_direct_conv_wgrad_n_wo_tiles(spec);
    return (n_wo_tiles + spec->waves_q - 1) / spec->waves_q;
}

rocke_status_t rocke_direct_conv_wgrad_kernel_name(const rocke_direct_conv_wgrad_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char prob_short[128];
    const char* parts[5];
    char bk_buf[24];
    char bc_buf[24];
    char hpb_buf[24];
    char mk_buf[24];
    const char* flag_names[3];
    int flag_on[3];
    const char* dtype;

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(bk_buf, sizeof(bk_buf), "bk%d", rocke_direct_conv_wgrad_block_k(spec));
    snprintf(bc_buf, sizeof(bc_buf), "bc%d", rocke_direct_conv_wgrad_block_c(spec));
    snprintf(hpb_buf, sizeof(hpb_buf), "hpb%d", spec->ho_per_block);
    snprintf(mk_buf, sizeof(mk_buf), "mk%d", spec->mfma_k);
    parts[0] = prob_short;
    parts[1] = bk_buf;
    parts[2] = bc_buf;
    parts[3] = hpb_buf;
    parts[4] = mk_buf;
    /* Python: flags={"wq": waves_q} if waves_q > 1 else {}, then flags["bf16"]
     * when p.dtype == "bf16". kernel_name_join appends the NAME (not the value)
     * for a truthy entry, and waves_q > 1 is exactly when the entry exists and
     * is truthy -- so a false entry and an absent one name the same kernel.
     * Order matches the Python dict's insertion order: wq, bf16, then rtc
     * (runtime_channels). */
    dtype = spec->problem.dtype ? spec->problem.dtype : "fp16";
    flag_names[0] = "wq";
    flag_on[0] = (spec->waves_q > 1) ? 1 : 0;
    flag_names[1] = "bf16";
    flag_on[1] = (strcmp(dtype, "bf16") == 0) ? 1 : 0;
    flag_names[2] = "rtc";
    flag_on[2] = spec->runtime_channels ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 3, out, out_cap, NULL);
}

rocke_status_t rocke_direct_conv_wgrad_validate(const rocke_direct_conv_wgrad_spec_t* spec,
                                                char* reason,
                                                size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;

#define ROCKE_DCONV_WGRAD_RAISE(...)                   \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return ROCKE_ERR_VALUE;                        \
    } while(0)

    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;
    /* if p.dtype not in ("fp16", "bf16"): raise ValueError(...) */
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            ROCKE_DCONV_WGRAD_RAISE("DirectConvWgradSpec: unsupported dtype '%s'", dt);
        }
    }
    if(p->kpg < spec->wave_tile_k)
    {
        ROCKE_DCONV_WGRAD_RAISE("kpg %d must be >= wave_tile_k %d", p->kpg, spec->wave_tile_k);
    }
    if(p->cpg < spec->wave_tile_c)
    {
        ROCKE_DCONV_WGRAD_RAISE("cpg %d must be >= wave_tile_c %d", p->cpg, spec->wave_tile_c);
    }
    if(spec->wave_tile_k != 16)
    {
        ROCKE_DCONV_WGRAD_RAISE("wave_tile_k must be 16");
    }
    if(spec->wave_tile_c != 16)
    {
        ROCKE_DCONV_WGRAD_RAISE("wave_tile_c must be 16");
    }
    if(p->KH < 1 || p->KH > ROCKE_DCONV_WGRAD_MAX_KH)
    {
        ROCKE_DCONV_WGRAD_RAISE("KH must be in 1..%d (got %d)", ROCKE_DCONV_WGRAD_MAX_KH, p->KH);
    }
    if(p->KW < 1 || p->KW > ROCKE_DCONV_WGRAD_MAX_KW)
    {
        ROCKE_DCONV_WGRAD_RAISE("KW must be in 1..%d (got %d)", ROCKE_DCONV_WGRAD_MAX_KW, p->KW);
    }
    /* Checked before the product: waves_k=0 would sail through
     * `waves_k * waves_c <= 16` and then divide by a zero block_k. */
    if(spec->waves_k < 1)
    {
        ROCKE_DCONV_WGRAD_RAISE("waves_k must be >= 1");
    }
    if(spec->waves_c < 1)
    {
        ROCKE_DCONV_WGRAD_RAISE("waves_c must be >= 1");
    }
    if(spec->waves_k * spec->waves_c > 16)
    {
        ROCKE_DCONV_WGRAD_RAISE("waves_k * waves_c must be <= 16");
    }
    if(spec->waves_q < 1)
    {
        ROCKE_DCONV_WGRAD_RAISE("waves_q must be >= 1");
    }
    if(spec->wave_size != 64)
    {
        ROCKE_DCONV_WGRAD_RAISE(ROCKE_WGRAD_WAVE64_WHY, spec->wave_size);
    }
    if(spec->ho_per_block <= 0)
    {
        ROCKE_DCONV_WGRAD_RAISE("ho_per_block must be > 0");
    }
    if(spec->mfma_k != 16 && spec->mfma_k != 32)
    {
        ROCKE_DCONV_WGRAD_RAISE("mfma_k must be 16 or 32 (got %d)", spec->mfma_k);
    }
    if(p->stride != 1)
    {
        ROCKE_DCONV_WGRAD_RAISE(ROCKE_WGRAD_STRIDE_WHY, p->stride);
    }
    if(reason != NULL && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;

#undef ROCKE_DCONV_WGRAD_RAISE
}

bool rocke_direct_conv_wgrad_is_valid_spec(const rocke_direct_conv_wgrad_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const rocke_direct_conv_problem_t* p;

#define ROCKE_DCONV_WGRAD_REJECT(...)                  \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
        return false;                                  \
    } while(0)

    if(spec == NULL)
    {
        ROCKE_DCONV_WGRAD_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            ROCKE_DCONV_WGRAD_REJECT("unsupported dtype '%s'; expected 'fp16' or 'bf16'", dt);
        }
    }
    if(p->kpg < spec->wave_tile_k)
    {
        ROCKE_DCONV_WGRAD_REJECT("kpg %d must be >= wave_tile_k %d", p->kpg, spec->wave_tile_k);
    }
    if(p->cpg < spec->wave_tile_c)
    {
        ROCKE_DCONV_WGRAD_REJECT("cpg %d must be >= wave_tile_c %d", p->cpg, spec->wave_tile_c);
    }
    if(spec->wave_tile_k != 16 || spec->wave_tile_c != 16)
    {
        ROCKE_DCONV_WGRAD_REJECT("wave_tile_k and wave_tile_c must be 16");
    }
    if(p->KH < 1 || p->KH > ROCKE_DCONV_WGRAD_MAX_KH)
    {
        ROCKE_DCONV_WGRAD_REJECT("KH must be in 1..%d (got %d)", ROCKE_DCONV_WGRAD_MAX_KH, p->KH);
    }
    if(p->KW < 1 || p->KW > ROCKE_DCONV_WGRAD_MAX_KW)
    {
        ROCKE_DCONV_WGRAD_REJECT("KW must be in 1..%d (got %d)", ROCKE_DCONV_WGRAD_MAX_KW, p->KW);
    }
    if(spec->waves_k < 1)
    {
        ROCKE_DCONV_WGRAD_REJECT("waves_k must be >= 1");
    }
    if(spec->waves_c < 1)
    {
        ROCKE_DCONV_WGRAD_REJECT("waves_c must be >= 1");
    }
    if(spec->waves_k * spec->waves_c > 16)
    {
        ROCKE_DCONV_WGRAD_REJECT("waves_k * waves_c must be <= 16");
    }
    if(spec->waves_q < 1)
    {
        ROCKE_DCONV_WGRAD_REJECT("waves_q must be >= 1");
    }
    if(spec->wave_size != target->wave_size)
    {
        ROCKE_DCONV_WGRAD_REJECT("wave_size %d does not match the %s wave size %d",
                                 spec->wave_size,
                                 arch,
                                 target->wave_size);
    }
    if(spec->wave_size != 64)
    {
        ROCKE_DCONV_WGRAD_REJECT(ROCKE_WGRAD_WAVE64_WHY, spec->wave_size);
    }
    if(rocke_direct_conv_wgrad_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        ROCKE_DCONV_WGRAD_REJECT("threads_per_block %d > %d (hardware cap) on %s",
                                 rocke_direct_conv_wgrad_threads_per_block(spec),
                                 rocke_archtarget_max_threads_per_block(target),
                                 arch);
    }
    if(spec->ho_per_block <= 0)
    {
        ROCKE_DCONV_WGRAD_REJECT("ho_per_block must be > 0");
    }
    if(spec->mfma_k != 16 && spec->mfma_k != 32)
    {
        ROCKE_DCONV_WGRAD_REJECT("mfma_k must be 16 or 32 (got %d)", spec->mfma_k);
    }
    if(p->stride != 1)
    {
        ROCKE_DCONV_WGRAD_REJECT(ROCKE_WGRAD_STRIDE_WHY, p->stride);
    }
    mma = rocke_archtarget_mma(target);
    {
        /* ab_dtype: "f16" or "bf16" based on problem.dtype */
        const char* ab = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? "bf16" : "f16";
        if(!rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 16))
        {
            ROCKE_DCONV_WGRAD_REJECT("missing mfma_f32_16x16x16_%s on %s", ab, arch);
        }
        if(spec->mfma_k == 32
           && !rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 32))
        {
            ROCKE_DCONV_WGRAD_REJECT(
                "mfma_k=32 needs mfma_f32_16x16x32_%s, absent on %s", ab, arch);
        }
    }
    if(!target->memory.has_ds_read_tr)
    {
        ROCKE_DCONV_WGRAD_REJECT(
            "wgrad LDS staging requires ds_read_tr16_b64 (gfx950+), absent on %s", arch);
    }
    if(reason != NULL && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;

#undef ROCKE_DCONV_WGRAD_REJECT
}

/* ===================================================================== *
 *  DirectDepthwiseTiledSpec implementations
 * ===================================================================== */

rocke_direct_depthwise_tiled_spec_t rocke_direct_depthwise_tiled_spec_default(void)
{
    rocke_direct_depthwise_tiled_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_tiled";
    spec.block_w = 8;
    spec.block_h = 4;
    spec.block_waves = 1;
    spec.unroll_rows = false;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_tiled_threads_per_block(const rocke_direct_depthwise_tiled_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_tiled_block_ch(const rocke_direct_depthwise_tiled_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_tiled_n_cols(const rocke_direct_depthwise_tiled_spec_t* spec)
{
    return (spec->block_w - 1) * spec->problem.stride + spec->problem.KW;
}

int rocke_direct_depthwise_tiled_live_regs(const rocke_direct_depthwise_tiled_spec_t* spec)
{
    return spec->block_h * spec->block_w + spec->block_h * rocke_direct_depthwise_tiled_n_cols(spec)
           + 2 * spec->problem.KW;
}

int rocke_direct_depthwise_tiled_unrolled_fmas(const rocke_direct_depthwise_tiled_spec_t* spec)
{
    return spec->problem.KH * spec->block_h * spec->block_w * spec->problem.KW;
}

rocke_status_t rocke_direct_depthwise_tiled_kernel_name(
    const rocke_direct_depthwise_tiled_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    const char* parts[5];
    char f_buf[32];
    char bw_buf[24];
    char bh_buf[24];
    char bwv_buf[24];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"f{KH}x{KW}", f"bw{block_w}",
     *                  f"bh{block_h}", f"bw{block_waves}wv",
     *                  flags={"ur": unroll_rows, "bf16": dtype=="bf16"}) */
    snprintf(f_buf, sizeof(f_buf), "f%dx%d", spec->problem.KH, spec->problem.KW);
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bh_buf, sizeof(bh_buf), "bh%d", spec->block_h);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = f_buf;
    parts[2] = bw_buf;
    parts[3] = bh_buf;
    parts[4] = bwv_buf;
    {
        const char* flag_names[2] = {"ur", "bf16"};
        int flag_on[2];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = spec->unroll_rows ? 1 : 0;
        flag_on[1] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 5, flag_names, flag_on, 2, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_depthwise_tiled_validate(
    const rocke_direct_depthwise_tiled_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseTiledSpec: unsupported dtype '%s'; "
                         "expected fp16 or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseTiledSpec requires cpg=kpg=1 (got cpg=%d, kpg=%d)",
                     p->cpg,
                     p->kpg);
        return ROCKE_ERR_VALUE;
    }
    if(spec->block_w < 1 || spec->block_h < 1 || spec->block_waves < 1 || spec->wave_size < 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseTiledSpec: block_w, block_h, block_waves and "
                     "wave_size must be positive");
        return ROCKE_ERR_VALUE;
    }
    if(p->stride < 1 || p->PAD < 0)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseTiledSpec: stride must be >= 1 and PAD >= 0 "
                     "(got stride=%d, PAD=%d)",
                     p->stride,
                     p->PAD);
        return ROCKE_ERR_VALUE;
    }
    {
        const int Ho = rocke_dconv_col__Ho(p);
        const int Wo = rocke_dconv_col__Wo(p);
        if(Ho < 1 || Wo < 1)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseTiledSpec: filter does not fit the padded input: "
                         "Ho=%d, Wo=%d (H=%d, W=%d, KH=%d, KW=%d, PAD=%d, stride=%d)",
                         Ho,
                         Wo,
                         p->H,
                         p->W,
                         p->KH,
                         p->KW,
                         p->PAD,
                         p->stride);
            return ROCKE_ERR_VALUE;
        }
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return ROCKE_OK;
}

bool rocke_direct_depthwise_tiled_is_valid_spec(const rocke_direct_depthwise_tiled_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    int live, fmas;

    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
            strncpy(reason, "null spec", reason_cap);
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(rocke_direct_depthwise_tiled_validate(spec, reason, reason_cap) != ROCKE_OK)
        return false;
    p = &spec->problem;
    {
        char dil_why[256];
        if(rocke_direct_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    if(p->KH < 1 || p->KH > ROCKE_DCONV_DW_MAX_KH)
    {
        if(reason && reason_cap > 0)
            snprintf(
                reason, reason_cap, "KH must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KH, p->KH);
        return false;
    }
    if(p->KW < 1 || p->KW > ROCKE_DCONV_DW_MAX_KW)
    {
        if(reason && reason_cap > 0)
            snprintf(
                reason, reason_cap, "KW must be in 1..%d (got %d)", ROCKE_DCONV_DW_MAX_KW, p->KW);
        return false;
    }
    if(p->stride > p->KH)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "stride %d > KH %d: some phases see no filter row",
                     p->stride,
                     p->KH);
        return false;
    }
    /* A filter larger than the padded input leaves nothing to compute and a
     * non-positive launch grid. Floored like Python's Ho / Wo (no dilation
     * term: a dilated problem is refused above). */
    {
        const int Ho = rocke_dconv_col__Ho(p);
        const int Wo = rocke_dconv_col__Wo(p);
        if(Ho < 1 || Wo < 1)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "filter does not fit the padded input: Ho=%d, Wo=%d (H=%d, W=%d, KH=%d, "
                         "KW=%d, PAD=%d, stride=%d)",
                         Ho,
                         Wo,
                         p->H,
                         p->W,
                         p->KH,
                         p->KW,
                         p->PAD,
                         p->stride);
            return false;
        }
    }
    if(spec->wave_size != target->wave_size)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "wave_size %d != %s wave %d",
                     spec->wave_size,
                     arch,
                     target->wave_size);
        return false;
    }
    if(rocke_direct_depthwise_tiled_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "threads_per_block %d exceeds arch limit",
                     rocke_direct_depthwise_tiled_threads_per_block(spec));
        return false;
    }
    live = rocke_direct_depthwise_tiled_live_regs(spec);
    if(live > ROCKE_DCONV_DW_TILED_REG_BUDGET)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "~%d live registers per lane exceed the %d-register budget: "
                     "shrink block_h or block_w",
                     live,
                     ROCKE_DCONV_DW_TILED_REG_BUDGET);
        return false;
    }
    fmas = rocke_direct_depthwise_tiled_unrolled_fmas(spec);
    if(spec->unroll_rows && fmas > ROCKE_DCONV_DW_TILED_UNROLL_MAX_FMAS)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "unrolling the filter-row loop emits %d FMAs (max %d)",
                     fmas,
                     ROCKE_DCONV_DW_TILED_UNROLL_MAX_FMAS);
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  DirectConvSpec (parametric kernel; also the MFMA dgrad compute pass)
 * ===================================================================== */

/* PRELOAD_WEIGHT_VGPR_BUDGET / DGRAD_WEIGHTS_LDS_BUDGET /
 * XCD_TILES_MIN_ROW_BANDS. */
#define ROCKE_DCONV_GEN_PRELOAD_VGPR_BUDGET 128
#define ROCKE_DCONV_GEN_WEIGHTS_LDS_BUDGET (64 * 1024)
#define ROCKE_DCONV_GEN_XCD_MIN_ROW_BANDS 8
/* Python's ZeroDivisionError text for an int // or % by zero. */
#define ROCKE_DCONV_GEN_ZERO_DIV "integer division or modulo by zero"
#define ROCKE_DCONV_GEN_ZERO_MOD "integer modulo by zero"

/* Python floor division (a // b) for b != 0. */
static long dconv_gen_floordiv(long a, long b)
{
    long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

/* -(-a // b): Python ceil division for b != 0. */
static long dconv_gen_ceildiv(long a, long b)
{
    return -dconv_gen_floordiv(-a, b);
}

rocke_direct_conv_spec_t rocke_direct_conv_spec_default(void)
{
    rocke_direct_conv_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_conv";
    s.block_q = 16;
    s.block_groups = 8;
    s.wave_size = 64;
    s.double_buffer = true;
    s.block_h = 16;
    s.waves_q = 1;
    s.waves_k = 1;
    s.runtime_k_loop = false;
    s.persistent_grid = false;
    s.persistent_rounds = 64;
    s.fold_k32 = false;
    s.runtime_channels = false;
    s.k_atoms = 0;
    s.m_tiles = 0;
    s.preload_weights = false;
    s.dgrad_fused_weights = false;
    s.dgrad_weights_lds = false;
    s.waves_per_eu = 0;
    s.row_guard = false;
    s.prefetch_rows = 0;
    s.lds_only_sync = false;
    s.waves_m = 1;
    s.xcd_tiles = false;
    s.lds_pad = 0;
    s.stage_out = false;
    s.row_exit = false;
    return s;
}

int rocke_direct_conv_threads_per_block(const rocke_direct_conv_spec_t* spec)
{
    return (spec->block_groups * spec->waves_q * spec->waves_k * spec->waves_m) * spec->wave_size;
}

/* The preloads_weights property: every weight fragment is loaded once in the
 * prologue. */
static bool dconv_gen_preloads_weights(const rocke_direct_conv_spec_t* spec)
{
    return spec->waves_k > 1 || spec->preload_weights || spec->dgrad_fused_weights;
}

int rocke_direct_conv_preload_weight_vgprs(const rocke_direct_conv_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const long k_atom = spec->fold_k32 ? 32 : 16;
    const long n_k_atoms
        = spec->fold_k32 ? dconv_gen_floordiv(p->cpg, k_atom) : dconv_gen_ceildiv(p->cpg, k_atom);
    /* Each of the waves_m waves of a group keeps only its own M-tiles. */
    const long n_m_tiles = dconv_gen_floordiv(dconv_gen_ceildiv(p->kpg, 16), spec->waves_m);
    return (int)dconv_gen_floordiv(
        (long)p->KH * p->KW * n_k_atoms * n_m_tiles * dconv_gen_floordiv(k_atom, 4), 2);
}

/* dgrad_weights_lds_bytes(spec). */
static long dconv_gen_weights_lds_bytes(const rocke_direct_conv_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    return (long)spec->block_groups * p->cpg * p->KH * p->KW * p->kpg * 2;
}

/* direct_conv_stage_out_stride(spec). */
static long dconv_gen_stage_out_stride(const rocke_direct_conv_spec_t* spec)
{
    return (long)spec->block_groups * spec->problem.kpg + 8;
}

long rocke_direct_conv_lds_bytes(const rocke_direct_conv_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const long load_vec = spec->fold_k32 ? 8 : 4;
    const long threads = rocke_direct_conv_threads_per_block(spec);
    const long lds_w = (long)(spec->block_q - 1) * p->stride + p->KW;
    const long num_chunks = lds_w * spec->block_groups * dconv_gen_floordiv(p->cpg, load_vec);
    const long passes = dconv_gen_floordiv(num_chunks + threads - 1, threads);
    long stage = passes * threads * load_vec * 2;
    long total;
    if(spec->lds_pad)
    {
        const long cols
            = dconv_gen_ceildiv(passes * threads,
                                spec->block_groups * dconv_gen_floordiv(p->cpg, load_vec))
              + 1;
        stage += cols * spec->lds_pad * 2;
    }
    if(spec->dgrad_weights_lds)
    {
        const long wl_vecs = dconv_gen_floordiv(dconv_gen_weights_lds_bytes(spec), 16);
        const long wl_passes = dconv_gen_floordiv(wl_vecs + threads - 1, threads);
        const long wl = wl_passes * threads * 16;
        total = stage > wl ? stage : wl;
        if(spec->double_buffer)
            total += stage;
    }
    else
        total = stage * (spec->double_buffer ? 2 : 1);
    if(spec->waves_k > 1)
        total += (long)spec->waves_q * spec->waves_k * spec->wave_size * 4 * 4;
    if(spec->stage_out)
        total += 2L * spec->block_q * dconv_gen_stage_out_stride(spec) * 2;
    return total;
}

rocke_status_t
    rocke_direct_conv_kernel_name(const rocke_direct_conv_spec_t* spec, char* out, size_t out_cap)
{
    char short_buf[128];
    char bq_buf[32];
    char bg_buf[32];
    char bh_buf[32];
    char wq_buf[32];
    char wk_buf[32];
    char rtc_buf[64];
    char we_buf[32];
    char pf_buf[32];
    char wm_buf[32];
    char lp_buf[32];
    const char* parts[22];
    const char* dtype;
    rocke_status_t st;

    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    st = rocke_direct_conv_problem_short(&spec->problem, short_buf, sizeof(short_buf));
    if(st != ROCKE_OK)
        return st;
    if(spec->problem.dtype == NULL)
        return ROCKE_ERR_VALUE;
    dtype = spec->problem.dtype;
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    bh_buf[0] = '\0';
    if(spec->block_h > 0)
        snprintf(bh_buf, sizeof(bh_buf), "bh%d", spec->block_h);
    wq_buf[0] = '\0';
    if(spec->waves_q > 1)
        snprintf(wq_buf, sizeof(wq_buf), "wq%d", spec->waves_q);
    wk_buf[0] = '\0';
    if(spec->waves_k > 1)
        snprintf(wk_buf, sizeof(wk_buf), "wk%d", spec->waves_k);
    rtc_buf[0] = '\0';
    if(spec->runtime_channels)
        snprintf(rtc_buf, sizeof(rtc_buf), "rtc_ka%d_mt%d", spec->k_atoms, spec->m_tiles);
    we_buf[0] = '\0';
    if(spec->waves_per_eu > 0)
        snprintf(we_buf, sizeof(we_buf), "we%d", spec->waves_per_eu);
    pf_buf[0] = '\0';
    if(spec->prefetch_rows > 1)
        snprintf(pf_buf, sizeof(pf_buf), "pf%d", spec->prefetch_rows);
    wm_buf[0] = '\0';
    if(spec->waves_m > 1)
        snprintf(wm_buf, sizeof(wm_buf), "wm%d", spec->waves_m);
    lp_buf[0] = '\0';
    if(spec->lds_pad > 0)
        snprintf(lp_buf, sizeof(lp_buf), "lp%d", spec->lds_pad);

    /* kernel_name_join(name, p.short(), bq, bg, db|sb, bh, wq, wk, rk, k32,
     *                  bf16, rtc, pw, fw|fwl, we, pf, lso, wm, xc, lp, so, rg,
     *                  rx) -- empty parts are dropped. */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    parts[3] = spec->double_buffer ? "db" : "sb";
    parts[4] = bh_buf;
    parts[5] = wq_buf;
    parts[6] = wk_buf;
    parts[7] = spec->runtime_k_loop ? "rk" : "";
    parts[8] = spec->fold_k32 ? "k32" : "";
    parts[9] = (strcmp(dtype, "bf16") == 0) ? "bf16" : "";
    parts[10] = rtc_buf;
    parts[11] = (spec->preload_weights && !spec->dgrad_fused_weights) ? "pw" : "";
    parts[12] = spec->dgrad_fused_weights ? (spec->dgrad_weights_lds ? "fwl" : "fw") : "";
    parts[13] = we_buf;
    parts[14] = pf_buf;
    parts[15] = spec->lds_only_sync ? "lso" : "";
    parts[16] = wm_buf;
    parts[17] = spec->xcd_tiles ? "xc" : "";
    parts[18] = lp_buf;
    parts[19] = spec->stage_out ? "so" : "";
    parts[20] = spec->row_guard ? "rg" : "";
    parts[21] = spec->row_exit ? "rx" : "";
    return rocke_kernel_name_join(spec->name, parts, 22, NULL, NULL, 0, out, out_cap, NULL);
}

/* _direct_conv_weight_knobs_reason(spec): 0 when the weight knobs combine, 1
 * with the reason in `out`, 2 where Python raises ZeroDivisionError (waves_m
 * == 0 under a weight preload; `out` holds its text, which is not prefixed). */
static int
    dconv_gen_weight_knobs_reason(const rocke_direct_conv_spec_t* spec, char* out, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(spec->preload_weights || spec->dgrad_fused_weights)
    {
        if(spec->waves_k != 1 || spec->runtime_k_loop || spec->persistent_grid
           || spec->runtime_channels)
        {
            snprintf(out,
                     cap,
                     "preload_weights / dgrad_fused_weights require waves_k=1, "
                     "runtime_k_loop=False, persistent_grid=False and baked channels");
            return 1;
        }
        if(spec->waves_m == 0)
        {
            snprintf(out, cap, ROCKE_DCONV_GEN_ZERO_DIV);
            return 2;
        }
        if(rocke_direct_conv_preload_weight_vgprs(spec) > ROCKE_DCONV_GEN_PRELOAD_VGPR_BUDGET)
        {
            snprintf(out,
                     cap,
                     "weight preload needs %d VGPRs per lane (budget %d)",
                     rocke_direct_conv_preload_weight_vgprs(spec),
                     ROCKE_DCONV_GEN_PRELOAD_VGPR_BUDGET);
            return 1;
        }
    }
    if(spec->dgrad_weights_lds)
    {
        if(!spec->dgrad_fused_weights)
        {
            snprintf(out, cap, "dgrad_weights_lds requires dgrad_fused_weights");
            return 1;
        }
        if(p->kpg % 4 != 0)
        {
            snprintf(out, cap, "dgrad_weights_lds requires kpg %% 4 == 0 (got %d)", p->kpg);
            return 1;
        }
        if(dconv_gen_weights_lds_bytes(spec) > ROCKE_DCONV_GEN_WEIGHTS_LDS_BUDGET)
        {
            snprintf(out,
                     cap,
                     "dgrad_weights_lds stages %ld bytes per workgroup (budget %d)",
                     dconv_gen_weights_lds_bytes(spec),
                     ROCKE_DCONV_GEN_WEIGHTS_LDS_BUDGET);
            return 1;
        }
    }
    if(spec->waves_per_eu < 0)
    {
        snprintf(out, cap, "waves_per_eu must be >= 0 (got %d)", spec->waves_per_eu);
        return 1;
    }
    if(spec->row_guard && !dconv_gen_preloads_weights(spec))
    {
        snprintf(
            out, cap, "row_guard needs preloaded weights (preload_weights, fused or waves_k > 1)");
        return 1;
    }
    if(spec->row_guard && spec->persistent_grid)
    {
        snprintf(out, cap, "row_guard needs persistent_grid=False");
        return 1;
    }
    return 0;
}

/* _direct_conv_row_stream_reason(spec): true with the reason in `out`. */
static bool dconv_gen_row_stream_reason(const rocke_direct_conv_spec_t* spec, char* out, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(spec->waves_m < 1)
    {
        snprintf(out, cap, "waves_m must be >= 1 (got %d)", spec->waves_m);
        return true;
    }
    if(!(0 <= spec->prefetch_rows && spec->prefetch_rows <= 3))
    {
        snprintf(out, cap, "prefetch_rows must be in 0..3 (got %d)", spec->prefetch_rows);
        return true;
    }
    if(spec->lds_pad < 0 || spec->lds_pad % 8 != 0)
    {
        snprintf(out, cap, "lds_pad must be a multiple of 8 >= 0 (got %d)", spec->lds_pad);
        return true;
    }
    if(spec->lds_pad && spec->runtime_channels)
    {
        snprintf(out, cap, "lds_pad needs baked channels");
        return true;
    }
    if(spec->stage_out
       && (p->kpg % 16 != 0 || spec->waves_q != 1 || spec->waves_k != 1 || spec->persistent_grid
           || spec->runtime_channels))
    {
        snprintf(out,
                 cap,
                 "stage_out needs kpg %% 16 == 0, waves_q == waves_k == 1, baked channels and "
                 "no persistent grid");
        return true;
    }
    if(spec->waves_m > 1
       && (p->kpg % (16 * spec->waves_m) != 0 || spec->waves_q != 1 || spec->waves_k != 1
           || spec->runtime_k_loop || spec->persistent_grid || spec->runtime_channels
           || !(spec->preload_weights || spec->dgrad_fused_weights)))
    {
        snprintf(out,
                 cap,
                 "waves_m > 1 needs kpg %% (16*waves_m) == 0, waves_q == waves_k == 1, "
                 "preloaded or fused weights, baked channels and no runtime K loop / "
                 "persistent grid");
        return true;
    }
    if((spec->prefetch_rows > 1 || spec->lds_only_sync)
       && (!spec->double_buffer || spec->persistent_grid))
    {
        snprintf(out,
                 cap,
                 "prefetch_rows > 1 / lds_only_sync need double_buffer=True and "
                 "persistent_grid=False");
        return true;
    }
    if(spec->xcd_tiles && spec->persistent_grid)
    {
        snprintf(out, cap, "xcd_tiles needs persistent_grid=False");
        return true;
    }
    if(spec->row_exit && spec->persistent_grid)
    {
        snprintf(out, cap, "row_exit needs persistent_grid=False");
        return true;
    }
    return false;
}

/* The XCD count the xcd_tiles remap is built for (_XCD_TILES_ARCHES); 0 when
 * the target has none. */
static int dconv_gen_xcd_tiles_count(const char* arch)
{
    return (arch != NULL && strcmp(arch, "gfx950") == 0) ? 8 : 0;
}

rocke_status_t rocke_direct_conv_validate(const rocke_direct_conv_spec_t* spec,
                                          char* reason,
                                          size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    char why[ROCKE_ERR_MSG_CAP];
#define ROCKE_DCONV_GEN_RAISE(...)                     \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        return ROCKE_ERR_VALUE;                        \
    } while(0)

    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    if(p->dtype == NULL)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec: dtype is not set");
    if(strcmp(p->dtype, "fp16") != 0 && strcmp(p->dtype, "bf16") != 0)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec: unsupported dtype '%s'", p->dtype);
    if(p->cpg % 4 != 0 || p->cpg < 4)
        ROCKE_DCONV_GEN_RAISE(
            "DirectConvSpec requires cpg to be a positive multiple of 4 (got cpg=%d)", p->cpg);
    if(p->kpg < 1)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec requires kpg >= 1 (got %d)", p->kpg);
    if(spec->block_groups == 0)
        ROCKE_DCONV_GEN_RAISE(ROCKE_DCONV_GEN_ZERO_MOD);
    if(spec->block_groups < 0 || p->groups % spec->block_groups != 0)
        ROCKE_DCONV_GEN_RAISE(
            "groups %d not divisible by block_groups %d", p->groups, spec->block_groups);
    if(spec->block_q % 16 != 0)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec block_q must be a multiple of 16");
    if(spec->persistent_grid && spec->persistent_rounds < 1)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec persistent_rounds must be >= 1");
    if(spec->block_h <= 0)
        ROCKE_DCONV_GEN_RAISE(
            "DirectConvSpec block_h must be > 0: the H-row loop is unrolled at build time, so an "
            "untiled kernel would have to bake the image height into its trip count and could "
            "not serve other shapes. Pick a tile height (16 is a reasonable default).");
    if(spec->fold_k32 && p->cpg % 32 != 0)
        ROCKE_DCONV_GEN_RAISE(
            "DirectConvSpec fold_k32 requires cpg to be a multiple of 32 (got %d)", p->cpg);
    if(spec->block_groups > 1 && spec->waves_k > 1)
        ROCKE_DCONV_GEN_RAISE(
            "block_groups=%d > 1 combined with waves_k=%d > 1 is not supported: the LDS reduction "
            "row index does not account for wave_group_idx, causing cross-group partial-sum "
            "corruption",
            spec->block_groups,
            spec->waves_k);
    /* Python divides by these below; a zero is a ZeroDivisionError there. */
    if(spec->waves_k == 0)
        ROCKE_DCONV_GEN_RAISE(ROCKE_DCONV_GEN_ZERO_MOD);
    {
        const int n_k_atoms = (p->cpg + 15) / 16;
        if(n_k_atoms % spec->waves_k != 0)
            ROCKE_DCONV_GEN_RAISE("N_K_ATOMS=%d (ceil(cpg/16)) must be divisible by waves_k=%d",
                                  n_k_atoms,
                                  spec->waves_k);
    }
    if(spec->waves_q == 0)
        ROCKE_DCONV_GEN_RAISE(ROCKE_DCONV_GEN_ZERO_DIV);
    if(dconv_gen_floordiv(spec->block_q, spec->waves_q) < 16)
        ROCKE_DCONV_GEN_RAISE("block_q//waves_q must be >= 16 (got %d//%d=%ld)",
                              spec->block_q,
                              spec->waves_q,
                              dconv_gen_floordiv(spec->block_q, spec->waves_q));
    /* Not a Python rule: a negative wave count passes the checks above there
     * and fails later; the C builder sizes arrays from it, so stop here. */
    if(spec->waves_k < 0 || spec->waves_q < 0)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec waves_q and waves_k must be >= 1");
    switch(dconv_gen_weight_knobs_reason(spec, why, sizeof why))
    {
    case 1:
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec %s", why);
    case 2:
        ROCKE_DCONV_GEN_RAISE("%s", why);
    default:
        break;
    }
    if(dconv_gen_row_stream_reason(spec, why, sizeof why))
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec %s", why);
    if(spec->xcd_tiles)
    {
        /* The one extent rule of the row-stream knobs (direct_conv_grid()[2]):
         * it does not change the IR. */
        const long gz = (long)p->N * dconv_gen_ceildiv(p->H, spec->block_h);
        if(gz < ROCKE_DCONV_GEN_XCD_MIN_ROW_BANDS)
            ROCKE_DCONV_GEN_RAISE("DirectConvSpec xcd_tiles needs at least %d image row bands "
                                  "(got %ld); the remap would be the identity",
                                  ROCKE_DCONV_GEN_XCD_MIN_ROW_BANDS,
                                  gz);
    }
    if(!spec->runtime_channels)
    {
        if(spec->k_atoms != 0 || spec->m_tiles != 0)
            ROCKE_DCONV_GEN_RAISE("DirectConvSpec k_atoms / m_tiles only apply with "
                                  "runtime_channels (a baked kernel takes them from the problem)");
        return ROCKE_OK;
    }
    if(spec->k_atoms < 1 || spec->m_tiles < 1)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec runtime_channels needs k_atoms >= 1 and m_tiles >= 1 "
                              "(got %d, %d)",
                              spec->k_atoms,
                              spec->m_tiles);
    if(spec->k_atoms % spec->waves_k != 0)
        ROCKE_DCONV_GEN_RAISE(
            "k_atoms=%d must be divisible by waves_k=%d", spec->k_atoms, spec->waves_k);
    {
        const int atom = spec->fold_k32 ? 32 : 16;
        const int need = spec->fold_k32 ? p->cpg / 32 : (p->cpg + 15) / 16;
        if(need != spec->k_atoms)
            ROCKE_DCONV_GEN_RAISE(
                "cpg=%d needs %d K-atoms of %d; this kernel is built for k_atoms=%d",
                p->cpg,
                need,
                atom,
                spec->k_atoms);
    }
    if((p->kpg + 15) / 16 != spec->m_tiles)
        ROCKE_DCONV_GEN_RAISE("kpg=%d needs %d M-tiles; this kernel is built for m_tiles=%d",
                              p->kpg,
                              (p->kpg + 15) / 16,
                              spec->m_tiles);
    if(p->kpg % 4 != 0)
        ROCKE_DCONV_GEN_RAISE("DirectConvSpec runtime_channels requires kpg %% 4 == 0 (got %d)",
                              p->kpg);
    return ROCKE_OK;
#undef ROCKE_DCONV_GEN_RAISE
}

bool rocke_direct_conv_is_valid_spec(const rocke_direct_conv_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_arch_mma_catalog_t* mma;
    const rocke_direct_conv_problem_t* p;
    const char* ab;
    char why[ROCKE_ERR_MSG_CAP];
#define ROCKE_DCONV_GEN_REJECT(...)                    \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        return false;                                  \
    } while(0)

    if(spec == NULL)
        ROCKE_DCONV_GEN_REJECT("spec is NULL");
    if(arch == NULL)
        arch = "gfx950";
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    if(p->dtype == NULL)
        ROCKE_DCONV_GEN_REJECT("dtype is not set; expected 'fp16' or 'bf16'");
    if(strcmp(p->dtype, "fp16") != 0 && strcmp(p->dtype, "bf16") != 0)
        ROCKE_DCONV_GEN_REJECT("unsupported dtype '%s'; expected 'fp16' or 'bf16'", p->dtype);
    if(rocke_direct_forward_padding_reason(p, why, sizeof why))
        ROCKE_DCONV_GEN_REJECT("%s", why);
    if(2 * p->PAD != p->KW - 1)
        ROCKE_DCONV_GEN_REJECT("needs same padding 2*PAD == KW-1 on the column axis too "
                               "(got KH=%d, KW=%d, PAD=%d)",
                               p->KH,
                               p->KW,
                               p->PAD);
    if(p->stride != 1)
        ROCKE_DCONV_GEN_REJECT("stride > 1 is not supported (got %d)", p->stride);
    if(p->cpg % 4 != 0 || p->cpg < 4)
        ROCKE_DCONV_GEN_REJECT("cpg must be a positive multiple of 4 (got %d)", p->cpg);
    if(p->kpg < 1)
        ROCKE_DCONV_GEN_REJECT("kpg must be >= 1 (got %d)", p->kpg);
    if(p->kpg % 4 != 0)
        ROCKE_DCONV_GEN_REJECT("kpg must be a multiple of 4 (got kpg=%d)", p->kpg);
    if(dconv_gen_weight_knobs_reason(spec, why, sizeof why) != 0)
        ROCKE_DCONV_GEN_REJECT("%s", why);
    if(spec->dgrad_weights_lds && !target->memory.has_ds_read_tr)
        ROCKE_DCONV_GEN_REJECT("dgrad_weights_lds needs ds_read_b64_tr_b16 (absent on %s)", arch);
    if(dconv_gen_row_stream_reason(spec, why, sizeof why))
        ROCKE_DCONV_GEN_REJECT("%s", why);
    if(spec->xcd_tiles && dconv_gen_xcd_tiles_count(arch) == 0)
        ROCKE_DCONV_GEN_REJECT("xcd_tiles is built for ['gfx950'] only (got %s)", arch);
    if(spec->block_groups <= 0 || p->groups % spec->block_groups != 0)
        ROCKE_DCONV_GEN_REJECT(
            "groups %d not divisible by block_groups %d", p->groups, spec->block_groups);
    if(spec->block_q % 16 != 0)
        ROCKE_DCONV_GEN_REJECT("block_q must be a multiple of 16");
    ab = (strcmp(p->dtype, "bf16") == 0) ? "bf16" : "f16";
    mma = rocke_archtarget_mma(target);
    if(!rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 16))
        ROCKE_DCONV_GEN_REJECT("missing mfma_f32_16x16x16_%s on %s", ab, arch);
    if(spec->fold_k32 && !rocke_mma_catalog_has_shape(mma, "mma", ab, ab, "fp32", 16, 16, 32))
        ROCKE_DCONV_GEN_REJECT("fold_k32 requires mfma_f32_16x16x32_%s on %s", ab, arch);
    {
        /* LDS budget: the A/B staging ring plus the waves_k > 1 cross-wave
         * reduction scratch (build_direct_conv's own sizing). */
        const long load_vec = spec->fold_k32 ? 8 : 4;
        const long n_vecs = dconv_gen_floordiv(p->cpg, load_vec);
        const long lds_w = (long)(spec->block_q - 1) * p->stride + p->KW;
        const long threads = rocke_direct_conv_threads_per_block(spec);
        const long num_chunks = lds_w * spec->block_groups * n_vecs;
        const long passes = dconv_gen_floordiv(num_chunks + threads - 1, threads);
        const long lds_total_fp16 = passes * threads * load_vec;
        const long ab_bytes = lds_total_fp16 * 2 * (spec->double_buffer ? 2 : 1);
        const long red_bytes = spec->waves_k > 1
                                   ? (long)(spec->waves_q * spec->waves_k) * spec->wave_size * 4 * 4
                                   : 0;
        const long total = ab_bytes + red_bytes;
        if(!rocke_archtarget_fits_lds(target, total))
            ROCKE_DCONV_GEN_REJECT("LDS budget %ld bytes (A/B=%s%ld, red=%ld) > %d cap on %s",
                                   total,
                                   spec->double_buffer ? "x2 " : "",
                                   lds_total_fp16 * 2,
                                   red_bytes,
                                   target->lds_capacity_bytes,
                                   arch);
    }
    if(spec->dgrad_weights_lds || spec->lds_pad || spec->stage_out)
    {
        /* The staged weight slice shares a pool slot with the first row
         * buffer; the column pad and the output tiles add to the footprint. */
        const long lds = rocke_direct_conv_lds_bytes(spec);
        if(!rocke_archtarget_fits_lds(target, lds))
            ROCKE_DCONV_GEN_REJECT("LDS footprint %ld B (row buffers, staged weights, output "
                                   "tiles) exceeds %d cap on %s",
                                   lds,
                                   target->lds_capacity_bytes,
                                   arch);
    }
    if(reason != NULL && reason_cap > 0)
        snprintf(reason, reason_cap, "ok");
    return true;
#undef ROCKE_DCONV_GEN_REJECT
}
