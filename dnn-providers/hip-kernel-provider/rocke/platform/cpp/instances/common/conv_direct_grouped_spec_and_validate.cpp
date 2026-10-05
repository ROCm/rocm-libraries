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
 *   (C-port-only 6-entry manifest ABI)         rocke_direct_conv_signature()
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
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, sig entry   */
#include "rocke/instance_conv_direct_grouped_internal.h" /* ROCKE_DCONV4C_MAX_WL_PASSES */

/* Reproduce str(KeyError(_build_target message)) for an unknown gfx target:
 *
 *   Python _build_target: raise KeyError(
 *     f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
 *     f"Add a row to {_DATA_FILE.name}.")
 *   is_valid_spec: except KeyError as e: return False, str(e)
 *
 * str(KeyError(msg)) == repr(msg); the single quotes make Python DOUBLE-quote
 * the whole message. sorted(specs) renders as ['gfx...', 'gfx...'].
 * rocke_known_arches() == tuple(sorted(_load_specs())). Mirrors fmha_arch.cpp. */
static void rocke_dconv__set_unknown_arch_reason(char* out, size_t out_cap, const char* gfx)
{
    int count = 0;
    const char* const* arches;
    int i;
    size_t pos = 0;
    int wrote;

    if(out == NULL || out_cap == 0)
    {
        return;
    }

    arches = rocke_known_arches(&count);

    wrote = snprintf(out + pos, out_cap - pos, "\"unknown gfx target '%s'; known: [", gfx);
    if(wrote < 0)
    {
        out[0] = '\0';
        return;
    }
    pos += (size_t)wrote;
    if(pos >= out_cap)
    {
        out[out_cap - 1] = '\0';
        return;
    }

    for(i = 0; i < count; ++i)
    {
        wrote = snprintf(out + pos, out_cap - pos, "%s'%s'", (i == 0) ? "" : ", ", arches[i]);
        if(wrote < 0)
        {
            out[out_cap - 1] = '\0';
            return;
        }
        pos += (size_t)wrote;
        if(pos >= out_cap)
        {
            out[out_cap - 1] = '\0';
            return;
        }
    }

    snprintf(out + pos, out_cap - pos, "]. Add a row to arch_specs.json.\"");
}

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

int rocke_direct_conv_problem_Ho(const rocke_direct_conv_problem_t* p)
{
    return (p->H + 2 * p->PAD - p->KH) / p->stride + 1;
}

int rocke_direct_conv_problem_Wo(const rocke_direct_conv_problem_t* p)
{
    return (p->W + 2 * p->PAD - p->KW) / p->stride + 1;
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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
    return s;
}

int rocke_direct_conv_4c_threads_per_block(const rocke_direct_conv_4c_spec_t* spec)
{
    /* (block_groups // 16) * wave_size */
    return (spec->block_groups / 16) * spec->wave_size;
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

    /* kernel_name_join(name, short, "bq..", "bg..", flags={"bf16": dtype=="bf16"}) */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    {
        const char* flag_names4c[3] = {"bf16", "fw", "fwl"};
        const char* dt4c = spec->problem.dtype ? spec->problem.dtype : "fp16";
        int flag_on4c[3];
        flag_on4c[0] = (strcmp(dt4c, "bf16") == 0) ? 1 : 0;
        flag_on4c[1] = (spec->dgrad_fused_weights && !spec->dgrad_weights_lds) ? 1 : 0;
        flag_on4c[2] = (spec->dgrad_fused_weights && spec->dgrad_weights_lds) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names4c, flag_on4c, 3, out, out_cap, NULL);
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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
    /* if p.stride != 1: return False, ... */
    if(p->stride != 1)
    {
        CK_DCONV4C_REJECT("stride > 1 is not supported (got %d)", p->stride);
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

/* ===================================================================== *
 *  SIGNATURE (manifest) -- the 6-entry ABI shared by both kernels:
 *    ptr A:f16, ptr B:f16, ptr D:f16, scalar A_bytes:i32, B_bytes:i32,
 *    D_bytes:i32.
 * ===================================================================== */

rocke_status_t rocke_direct_conv_signature(struct rocke_arena* arena,
                                           struct rocke_sig_entry* out,
                                           size_t out_cap,
                                           size_t* out_count)
{
    rocke_status_t st;

    if(arena == NULL || out == NULL || out_cap < 6)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_sig_param(arena, "A", "f16", NULL, &out[0]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "B", "f16", NULL, &out[1]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "D", "f16", NULL, &out[2]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "A_bytes", "i32", &out[3]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "B_bytes", "i32", &out[4]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "D_bytes", "i32", &out[5]);
    if(st != ROCKE_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 6;
    }
    return ROCKE_OK;
}

rocke_status_t rocke_direct_conv_signature_for_dtype(struct rocke_arena* arena,
                                                     const char* dtype,
                                                     struct rocke_sig_entry* out,
                                                     size_t out_cap,
                                                     size_t* out_count)
{
    rocke_status_t st;
    /* Resolve dtype: "fp16" -> "f16", "bf16" -> "bf16", NULL -> "f16". */
    const char* dt;

    if(arena == NULL || out == NULL || out_cap < 6)
    {
        return ROCKE_ERR_VALUE;
    }
    if(dtype == NULL || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
    {
        dt = "f16";
    }
    else if(strcmp(dtype, "bf16") == 0)
    {
        dt = "bf16";
    }
    else
    {
        return ROCKE_ERR_VALUE; /* unsupported dtype */
    }

    st = rocke_sig_param(arena, "A", dt, NULL, &out[0]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "B", dt, NULL, &out[1]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "D", dt, NULL, &out[2]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "A_bytes", "i32", &out[3]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "B_bytes", "i32", &out[4]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "D_bytes", "i32", &out[5]);
    if(st != ROCKE_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 6;
    }
    return ROCKE_OK;
}

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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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

bool rocke_direct_depthwise_is_valid_spec(const rocke_direct_depthwise_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    int block_ch;
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
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
    const char* flag_names[2];
    int flag_on[2];
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
     * Order matches the Python dict's insertion order: wq, then bf16. */
    dtype = spec->problem.dtype ? spec->problem.dtype : "fp16";
    flag_names[0] = "wq";
    flag_on[0] = (spec->waves_q > 1) ? 1 : 0;
    flag_names[1] = "bf16";
    flag_on[1] = (strcmp(dtype, "bf16") == 0) ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 2, out, out_cap, NULL);
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
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
 *  DirectDepthwiseDgradWindowedSpec  (cpg=kpg=1 dgrad, stride 1, windowed)
 * ===================================================================== */

rocke_direct_depthwise_dgrad_win_spec_t rocke_direct_depthwise_dgrad_win_spec_default(void)
{
    rocke_direct_depthwise_dgrad_win_spec_t spec;
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_dgrad_win";
    spec.block_w = 8;
    spec.block_waves = 1;
    spec.ch_per_lane = 1;
    spec.block_h = 0;
    spec.dot2 = false;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_dgrad_win_threads_per_block(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_win_block_ch(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return rocke_direct_depthwise_dgrad_win_threads_per_block(spec) * spec->ch_per_lane;
}

int rocke_direct_depthwise_dgrad_win_rows_per_block(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const int H = spec->problem.H;
    return (spec->block_h <= 0 || spec->block_h >= H) ? H : spec->block_h;
}

int rocke_direct_depthwise_dgrad_win_h_tiles(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const int rows = rocke_direct_depthwise_dgrad_win_rows_per_block(spec);
    return (spec->problem.H + rows - 1) / rows;
}

void rocke_direct_depthwise_dgrad_win_grid(const rocke_direct_depthwise_dgrad_win_spec_t* spec,
                                           int out_grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int block_ch = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    out_grid[0] = (p->W + spec->block_w - 1) / spec->block_w;
    out_grid[1] = (p->groups + block_ch - 1) / block_ch;
    out_grid[2] = p->N * rocke_direct_depthwise_dgrad_win_h_tiles(spec);
}

long long
    rocke_direct_depthwise_dgrad_win_unrolled_fmas(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    return (long long)rocke_direct_depthwise_dgrad_win_rows_per_block(spec) * p->KH * p->KW
           * spec->block_w * spec->ch_per_lane;
}

rocke_status_t rocke_direct_depthwise_dgrad_win_kernel_name(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    char rsp_buf[48];
    char bw_buf[32];
    char wv_buf[32];
    char cpl_buf[32];
    char bh_buf[32];
    const char* parts[6];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    /* kernel_name_join(name, p.short(), f"r{KH}s{KW}p{PAD}", f"bw{block_w}",
     *                  f"wv{block_waves}", f"cpl{ch_per_lane}",
     *                  f"bh{rows_per_block}" if h_tiles > 1 else "",
     *                  flags={"dot2": dot2, "bf16": dtype == "bf16"}) */
    snprintf(rsp_buf,
             sizeof(rsp_buf),
             "r%ds%dp%d",
             spec->problem.KH,
             spec->problem.KW,
             spec->problem.PAD);
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(wv_buf, sizeof(wv_buf), "wv%d", spec->block_waves);
    snprintf(cpl_buf, sizeof(cpl_buf), "cpl%d", spec->ch_per_lane);
    bh_buf[0] = '\0';
    if(rocke_direct_depthwise_dgrad_win_h_tiles(spec) > 1)
    {
        snprintf(
            bh_buf, sizeof(bh_buf), "bh%d", rocke_direct_depthwise_dgrad_win_rows_per_block(spec));
    }
    parts[0] = prob_short;
    parts[1] = rsp_buf;
    parts[2] = bw_buf;
    parts[3] = wv_buf;
    parts[4] = cpl_buf;
    parts[5] = bh_buf;
    {
        const char* flag_names[2] = {"dot2", "bf16"};
        int flag_on[2];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = spec->dot2 ? 1 : 0;
        flag_on[1] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 6, flag_names, flag_on, 2, out, out_cap, NULL);
    }
}

/* Python _depthwise_dgrad_win_check: (ok, reason) without the class prefix. */
static bool rocke_dw_dgrad_win__check(const rocke_direct_depthwise_dgrad_win_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dt = p->dtype ? p->dtype : "fp16";
    char buf[256];
    bool ok = true;

    buf[0] = '\0';
    if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported dtype '%s'; expected fp16 or bf16", dt);
        ok = false;
    }
    else if(p->cpg != 1 || p->kpg != 1)
    {
        snprintf(buf, sizeof(buf), "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
        ok = false;
    }
    else if(p->stride != 1)
    {
        snprintf(buf, sizeof(buf), "requires stride=1 (got %d)", p->stride);
        ok = false;
    }
    else
    {
        /* stride == 1 here, so C truncation equals Python floor division. */
        const int Ho = p->H + 2 * p->PAD - p->KH + 1;
        const int Wo = p->W + 2 * p->PAD - p->KW + 1;
        const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
        const int cpl = spec->ch_per_lane;
        if(p->PAD < 0 || Ho <= 0 || Wo <= 0)
        {
            snprintf(buf,
                     sizeof(buf),
                     "degenerate geometry (PAD=%d, Ho=%d, Wo=%d)",
                     p->PAD,
                     Ho,
                     Wo);
            ok = false;
        }
        else if(spec->block_w < 1)
        {
            snprintf(buf, sizeof(buf), "block_w must be >= 1 (got %d)", spec->block_w);
            ok = false;
        }
        else if(spec->block_waves < 1 || threads > 1024)
        {
            snprintf(buf,
                     sizeof(buf),
                     "block_waves must give 1..1024 threads (got %d)",
                     spec->block_waves);
            ok = false;
        }
        else if(cpl != 1 && cpl != 2 && cpl != 4 && cpl != 8)
        {
            snprintf(buf, sizeof(buf), "ch_per_lane must be 1, 2, 4 or 8 (got %d)", cpl);
            ok = false;
        }
        else if(p->groups % cpl != 0)
        {
            snprintf(buf,
                     sizeof(buf),
                     "groups=%d is not a multiple of ch_per_lane=%d",
                     p->groups,
                     cpl);
            ok = false;
        }
        else if(spec->dot2 && cpl != 1)
        {
            snprintf(buf, sizeof(buf), "dot2 requires ch_per_lane=1 (got %d)", cpl);
            ok = false;
        }
        else if(spec->block_h < 0)
        {
            snprintf(buf, sizeof(buf), "block_h must be >= 0 (got %d)", spec->block_h);
            ok = false;
        }
        else
        {
            const long long dy_bytes = (long long)p->N * Ho * Wo * p->groups * 2;
            const long long dx_bytes = (long long)p->N * p->H * p->W * p->groups * 2;
            const long long big = dy_bytes > dx_bytes ? dy_bytes : dx_bytes;
            const long long fmas = rocke_direct_depthwise_dgrad_win_unrolled_fmas(spec);
            if(big > ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES)
            {
                snprintf(buf,
                         sizeof(buf),
                         "dY/dX of %lld bytes exceed the %lld-byte sentinel addressing range",
                         big,
                         (long long)ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES);
                ok = false;
            }
            else if(fmas > ROCKE_DW_DGRAD_WIN_MAX_UNROLL)
            {
                snprintf(buf,
                         sizeof(buf),
                         "unrolled body too large (%lld FMAs > %d); lower block_w or set block_h",
                         fmas,
                         ROCKE_DW_DGRAD_WIN_MAX_UNROLL);
                ok = false;
            }
        }
    }
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", ok ? "ok" : buf);
    }
    return ok;
}

rocke_status_t rocke_direct_depthwise_dgrad_win_validate(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* reason, size_t reason_cap)
{
    char why[256];
    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(!rocke_dw_dgrad_win__check(spec, why, sizeof(why)))
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectDepthwiseDgradWindowedSpec: %s", why);
        }
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

bool rocke_direct_depthwise_dgrad_win_is_valid_spec(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
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
        rocke_dconv__set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    /* DW_DGRAD_WIN_DOT2_ARCHES = ("gfx950",) */
    if(spec->dot2 && strcmp(arch, "gfx950") != 0)
    {
        if(reason && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "dot2 needs one of ('gfx950',) (got %s)", arch);
        }
        return false;
    }
    return rocke_dw_dgrad_win__check(spec, reason, reason_cap);
}
