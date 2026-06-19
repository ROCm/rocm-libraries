// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_direct_grouped_spec_and_validate.c -- C99 port of the SPEC +
 * VALIDITY + SIGNATURE surface of
 * ck_dsl/instances/common/conv_direct_grouped.py.
 *
 * This translation unit owns the "host-side, IR-free" value/property layer that
 * both kernels (16c / 4c) share. NONE of it calls the IR builder (ckc_b_*):
 *
 *   Python (conv_direct_grouped.py)            C99 (this file)
 *   --------------------------------------     ----------------------------------
 *   DirectConvProblem defaults                 ckc_direct_conv_problem_default()
 *     .total_c / .total_k / .flops             ckc_direct_conv_problem_total_c/...
 *     .short()                                 ckc_direct_conv_problem_short()
 *   DirectConv16cSpec defaults                 ckc_direct_conv_16c_spec_default()
 *     .threads_per_block / .n_acc_slots        ckc_direct_conv_16c_*
 *     .kernel_name() / .validate()             ckc_direct_conv_16c_kernel_name / _validate
 *   DirectConv4cSpec defaults                  ckc_direct_conv_4c_spec_default()
 *     .threads_per_block                       ckc_direct_conv_4c_threads_per_block
 *     .kernel_name() / .validate()             ckc_direct_conv_4c_kernel_name / _validate
 *   is_valid_spec_16c(spec, arch)              ckc_direct_conv_16c_is_valid_spec()
 *   is_valid_spec_4c(spec, arch)               ckc_direct_conv_4c_is_valid_spec()
 *   (C-port-only 6-entry manifest ABI)         ckc_direct_conv_signature()
 *
 * The reason strings + the kernel name are formatted byte-identically to Python
 * (kernel_name_join, the ValueError messages) so a sweep driver sees the same
 * accept/reject and the same kernel identifier. The IR-emitting builders + their
 * phase closures live in the sibling TUs that bind to
 * ckc/instance_conv_direct_grouped_internal.h.
 */

#include "ckc/instance_conv_direct_grouped.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.core.arch.h"    /* ckc_archtarget_from_gfx, has_shape */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_kernel_name_join, sig entry   */

/* ===================================================================== *
 *  DirectConvProblem
 * ===================================================================== */

ckc_direct_conv_problem_t ckc_direct_conv_problem_default(void)
{
    ckc_direct_conv_problem_t p;
    memset(&p, 0, sizeof(p));
    /* Required dims (N,H,W,groups,cpg,kpg) have no Python default -> 0.    */
    p.N      = 0;
    p.H      = 0;
    p.W      = 0;
    p.groups = 0;
    p.cpg    = 0;
    p.kpg    = 0;
    /* Dataclass defaults. */
    p.KH     = 3;
    p.KW     = 3;
    p.PAD    = 1;
    p.stride = 1;
    return p;
}

int ckc_direct_conv_problem_total_c(const ckc_direct_conv_problem_t* p)
{
    return p->groups * p->cpg;
}

int ckc_direct_conv_problem_total_k(const ckc_direct_conv_problem_t* p)
{
    return p->groups * p->kpg;
}

long long ckc_direct_conv_problem_flops(const ckc_direct_conv_problem_t* p)
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

ckc_status_t
ckc_direct_conv_problem_short(const ckc_direct_conv_problem_t* p, char* out, size_t out_cap)
{
    int n;
    if(p == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }
    /* f"N{N}H{H}W{W}_g{groups}_c{cpg}k{kpg}" */
    n = snprintf(out, out_cap, "N%dH%dW%d_g%d_c%dk%d", p->N, p->H, p->W, p->groups, p->cpg, p->kpg);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  DirectConv16cSpec
 * ===================================================================== */

ckc_direct_conv_16c_spec_t ckc_direct_conv_16c_spec_default(void)
{
    ckc_direct_conv_16c_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem       = ckc_direct_conv_problem_default();
    s.name          = "direct_conv_16c";
    s.block_q       = 16;
    s.block_groups  = 8;
    s.wave_size     = 64;
    s.double_buffer = true;
    s.fold_k32      = true;
    return s;
}

int ckc_direct_conv_16c_threads_per_block(const ckc_direct_conv_16c_spec_t* spec)
{
    /* block_groups * wave_size */
    return spec->block_groups * spec->wave_size;
}

int ckc_direct_conv_16c_n_acc_slots(const ckc_direct_conv_16c_spec_t* spec)
{
    /* problem.KH */
    return spec->problem.KH;
}

ckc_status_t
ckc_direct_conv_16c_kernel_name(const ckc_direct_conv_16c_spec_t* spec, char* out, size_t out_cap)
{
    char short_buf[128];
    char bq_buf[32];
    char bg_buf[32];
    const char* db_part;
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];
    ckc_status_t st;

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    /* p.short() */
    st = ckc_direct_conv_problem_short(&spec->problem, short_buf, sizeof(short_buf));
    if(st != CKC_OK)
    {
        return st;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);
    db_part = spec->double_buffer ? "db" : "sb";

    /* kernel_name_join(name, short, "bq..", "bg..", "db"/"sb",
     *                  flags={"k32": fold_k32}) */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;
    parts[3] = db_part;

    flag_names[0] = "k32";
    flag_on[0]    = spec->fold_k32 ? 1 : 0;

    return ckc_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

ckc_status_t ckc_direct_conv_16c_validate(const ckc_direct_conv_16c_spec_t* spec,
                                          char* reason,
                                          size_t reason_cap)
{
    const ckc_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    p = &spec->problem;
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
        return CKC_ERR_VALUE;
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
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

bool ckc_direct_conv_16c_is_valid_spec(const ckc_direct_conv_16c_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap)
{
    const ckc_archtarget_t* target;
    const ckc_arch_mma_catalog_t* mma;
    const ckc_direct_conv_problem_t* p;

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
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Python str(KeyError) carries the full "...; known: [...]" message; the
         * C arch API does not expose the known-list, so we emit a faithful-shape
         * prefix.
         * TODO(port): reproduce the exact "; known: [...]" known-list suffix. */
        CK_DCONV16C_REJECT("unknown gfx target '%s'", arch);
    }

    p = &spec->problem;
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

    mma = ckc_archtarget_mma(target);
    /* if not target.mma.has_shape(f16,f16,fp32, 16,16,16): return False, ... */
    if(!ckc_mma_catalog_has_shape(mma, "mma", "f16", "f16", "fp32", 16, 16, 16))
    {
        CK_DCONV16C_REJECT("missing 16x16x16 f16 MFMA atom on %s", arch);
    }
    /* if spec.fold_k32 and not target.mma.has_shape(f16,f16,fp32, 16,16,32): ... */
    if(spec->fold_k32 && !ckc_mma_catalog_has_shape(mma, "mma", "f16", "f16", "fp32", 16, 16, 32))
    {
        CK_DCONV16C_REJECT("fold_k32=True needs the 16x16x32 f16 MFMA atom, absent on %s; use "
                           "fold_k32=False for a %s-capable kernel",
                           arch,
                           arch);
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

ckc_direct_conv_4c_spec_t ckc_direct_conv_4c_spec_default(void)
{
    ckc_direct_conv_4c_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem      = ckc_direct_conv_problem_default();
    s.name         = "direct_conv_4c";
    s.block_q      = 4;
    s.block_groups = 16;
    s.wave_size    = 64;
    return s;
}

int ckc_direct_conv_4c_threads_per_block(const ckc_direct_conv_4c_spec_t* spec)
{
    /* (block_groups // 16) * wave_size */
    return (spec->block_groups / 16) * spec->wave_size;
}

ckc_status_t
ckc_direct_conv_4c_kernel_name(const ckc_direct_conv_4c_spec_t* spec, char* out, size_t out_cap)
{
    char short_buf[128];
    char bq_buf[32];
    char bg_buf[32];
    const char* parts[3];
    ckc_status_t st;

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    /* p.short() */
    st = ckc_direct_conv_problem_short(&spec->problem, short_buf, sizeof(short_buf));
    if(st != CKC_OK)
    {
        return st;
    }
    snprintf(bq_buf, sizeof(bq_buf), "bq%d", spec->block_q);
    snprintf(bg_buf, sizeof(bg_buf), "bg%d", spec->block_groups);

    /* kernel_name_join(name, short, "bq..", "bg..")  -- no flags */
    parts[0] = short_buf;
    parts[1] = bq_buf;
    parts[2] = bg_buf;

    return ckc_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

ckc_status_t
ckc_direct_conv_4c_validate(const ckc_direct_conv_4c_spec_t* spec, char* reason, size_t reason_cap)
{
    const ckc_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    p = &spec->problem;
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
        return CKC_ERR_VALUE;
    }
    /* if self.block_groups % 16 != 0: raise ValueError("...") */
    if((spec->block_groups % 16) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv4cSpec block_groups must be a multiple of 16");
        }
        return CKC_ERR_VALUE;
    }
    /* if self.block_q % 4 != 0: raise ValueError("...") */
    if((spec->block_q % 4) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "DirectConv4cSpec block_q must be a multiple of 4");
        }
        return CKC_ERR_VALUE;
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
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

bool ckc_direct_conv_4c_is_valid_spec(const ckc_direct_conv_4c_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    const ckc_archtarget_t* target;
    const ckc_direct_conv_problem_t* p;

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
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* TODO(port): reproduce the exact "; known: [...]" known-list suffix. */
        CK_DCONV4C_REJECT("unknown gfx target '%s'", arch);
    }

    p = &spec->problem;
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

ckc_status_t ckc_direct_conv_signature(struct ckc_arena* arena,
                                       struct ckc_sig_entry* out,
                                       size_t out_cap,
                                       size_t* out_count)
{
    ckc_status_t st;

    if(arena == NULL || out == NULL || out_cap < 6)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_sig_param(arena, "A", "f16", NULL, &out[0]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "B", "f16", NULL, &out[1]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "D", "f16", NULL, &out[2]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "A_bytes", "i32", &out[3]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "B_bytes", "i32", &out[4]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "D_bytes", "i32", &out[5]);
    if(st != CKC_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 6;
    }
    return CKC_OK;
}
