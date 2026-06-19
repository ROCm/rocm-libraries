// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ckc/instance_grouped_gemm.c -- C99 port of
 * ck_dsl/instances/common/grouped_gemm.py.
 *
 * GroupedGemmSpec is a thin wrapper around UniversalGemmSpec: every entry point
 * converts the grouped spec to a UniversalGemmSpec and delegates to the
 * already-ported gemm_universal machinery. No new IR is emitted, so the lowered
 * IR is byte-identical to the universal-GEMM body the Python delegates to.
 */
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_grouped_gemm.h"

#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/ir_internal.h"      /* ckc_i_set_err (sticky-error helper) */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  ckc_grouped_gemm_spec_default
 *
 *  Python GroupedGemmSpec field defaults: wave_size=64, block_size=0,
 *  dtype="fp16". tile/trait carry their own dataclass defaults; trait reuses
 *  the universal-GEMM trait defaults (TraitSpec). The tile geometry is required
 *  and left zeroed for the caller to fill (matching the universal default).
 * ===================================================================== */
ckc_grouped_gemm_spec_t ckc_grouped_gemm_spec_default(void)
{
    ckc_grouped_gemm_spec_t s;
    ckc_gemm_universal_spec_t u = ckc_gemm_universal_spec_default();

    memset(&s, 0, sizeof(s));
    s.name       = NULL;
    s.tile       = u.tile;  /* TileSpec defaults (warp_k=1, warp_tile 32/32/16) */
    s.trait      = u.trait; /* TraitSpec defaults                               */
    s.wave_size  = 64;
    s.block_size = 0;
    s.dtype      = "fp16";
    return s;
}

/* ===================================================================== *
 *  ckc_grouped_gemm_spec_finalize
 *
 *  __post_init__ -> WarpTileBlockSizeMixin._init_block_size(): if block_size==0
 *  derive warp_m*warp_n*warp_k*wave_size. Idempotent.
 * ===================================================================== */
void ckc_grouped_gemm_spec_finalize(ckc_grouped_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    spec->block_size = ckc_warp_tile_init_block_size(
        spec->block_size, spec->tile.warp_m, spec->tile.warp_n, spec->tile.warp_k, spec->wave_size);
}

/* ===================================================================== *
 *  ckc_grouped_gemm_data_spec
 *
 *  GroupedGemmSpec._data_spec():
 *      dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
 *      return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)
 *  The remaining DataSpec fields (dtype_acc, layout) take their universal-GEMM
 *  defaults ("fp32" / "RCR").
 * ===================================================================== */
ckc_gemm_data_spec_t ckc_grouped_gemm_data_spec(const ckc_grouped_gemm_spec_t* spec)
{
    ckc_gemm_universal_spec_t u = ckc_gemm_universal_spec_default();
    ckc_gemm_data_spec_t d      = u.data; /* dtype_acc="fp32", layout="RCR" defaults */
    const char* dt;

    if(spec == NULL || spec->dtype == NULL)
    {
        dt = "fp16";
    }
    else if(strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "fp16") == 0)
    {
        dt = "fp16";
    }
    else
    {
        dt = spec->dtype;
    }
    d.dtype_a = dt;
    d.dtype_b = dt;
    d.dtype_c = dt;
    return d;
}

/* ===================================================================== *
 *  ckc_grouped_gemm_to_universal_spec
 *
 *  GroupedGemmSpec.to_universal_spec():
 *      UniversalGemmSpec(name, tile, trait, data=_data_spec(),
 *                        wave_size, block_size, batched=False)
 *  The returned spec is finalized (block_size derived) so it is build-ready.
 * ===================================================================== */
ckc_gemm_universal_spec_t ckc_grouped_gemm_to_universal_spec(const ckc_grouped_gemm_spec_t* spec)
{
    ckc_gemm_universal_spec_t u = ckc_gemm_universal_spec_default();

    if(spec == NULL)
    {
        return u;
    }
    u.name       = spec->name;
    u.tile       = spec->tile;
    u.trait      = spec->trait;
    u.data       = ckc_grouped_gemm_data_spec(spec);
    u.wave_size  = spec->wave_size;
    u.block_size = spec->block_size;
    u.batched    = false;

    /* Python passes self.block_size (already derived by __post_init__). Mirror
     * that: ensure the universal spec's block_size matches the grouped spec's
     * finalized value. */
    ckc_gemm_universal_spec_finalize(&u);
    return u;
}

/* ===================================================================== *
 *  ckc_grouped_gemm_kernel_name
 *
 *  GroupedGemmSpec.kernel_name() == to_universal_spec().kernel_name().
 * ===================================================================== */
ckc_status_t
ckc_grouped_gemm_kernel_name(const ckc_grouped_gemm_spec_t* spec, char* out, size_t out_cap)
{
    ckc_gemm_universal_spec_t u;

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }
    u = ckc_grouped_gemm_to_universal_spec(spec);
    return ckc_gemm_universal_kernel_name(&u, out, out_cap);
}

/* ===================================================================== *
 *  ckc_grouped_gemm_is_valid_spec
 *
 *  is_valid_spec(spec, arch): delegates to
 *  ck_dsl.instances.common.gemm_universal.is_valid_spec(to_universal_spec()).
 * ===================================================================== */
bool ckc_grouped_gemm_is_valid_spec(const ckc_grouped_gemm_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    ckc_gemm_universal_spec_t u;

    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            reason[0] = '\0';
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    u = ckc_grouped_gemm_to_universal_spec(spec);
    return ckc_gemm_universal_is_valid_spec(&u, arch, reason, reason_cap);
}

/* ===================================================================== *
 *  ckc_build_grouped_gemm
 *
 *  build_grouped_gemm(spec, arch):
 *      universal = spec.to_universal_spec()
 *      ok, why = is_valid_gemm_spec(universal, arch)
 *      if not ok: raise ValueError(...)
 *      return build_universal_gemm(universal, arch)
 *
 *  Python raises ValueError on reject; here we set the builder's sticky error
 *  and return NULL (so the caller's lower path surfaces the same message).
 * ===================================================================== */
ckc_kernel_def_t*
ckc_build_grouped_gemm(ckc_ir_builder_t* b, const ckc_grouped_gemm_spec_t* spec, const char* arch)
{
    ckc_gemm_universal_spec_t u;
    char reason[CKC_ERR_MSG_CAP];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    u = ckc_grouped_gemm_to_universal_spec(spec);
    if(!ckc_gemm_universal_is_valid_spec(&u, arch, reason, sizeof(reason)))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid grouped_gemm spec for %s: %s", arch, reason);
    }
    return ckc_build_universal_gemm(b, &u, arch);
}

/* ===================================================================== *
 *  ckc_build_grouped_gemm_new -- init builder with spec.kernel_name() + build.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_grouped_gemm_new(ckc_ir_builder_t* b,
                                             const ckc_grouped_gemm_spec_t* spec,
                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_grouped_gemm_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_grouped_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  ckc_build_grouped_gemm_single_launch
 *
 *  build_grouped_gemm_single_launch(spec, arch):
 *      base_spec = spec.to_universal_spec()
 *      batched_spec = UniversalGemmSpec(name=base_spec.name + "_single_launch",
 *                                       tile, trait, data, wave_size,
 *                                       block_size, batched=True)
 *      ok, why = is_valid_gemm_spec(batched_spec, arch)
 *      if not ok: raise ValueError(...)
 *      return build_universal_gemm(batched_spec, arch)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_grouped_gemm_single_launch(ckc_ir_builder_t* b,
                                                       const ckc_grouped_gemm_spec_t* spec,
                                                       const char* arch)
{
    ckc_gemm_universal_spec_t u;
    char reason[CKC_ERR_MSG_CAP];
    char name[256];
    const char* base_name;
    size_t blen, slen;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    u = ckc_grouped_gemm_to_universal_spec(spec);

    /* name = base_spec.name + "_single_launch" */
    base_name = (u.name != NULL) ? u.name : "";
    blen      = strlen(base_name);
    slen      = strlen("_single_launch");
    if(blen + slen >= sizeof(name))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "grouped_gemm_single_launch: name too long");
    }
    memcpy(name, base_name, blen);
    memcpy(name + blen, "_single_launch", slen + 1);
    u.name    = name;
    u.batched = true;

    if(!ckc_gemm_universal_is_valid_spec(&u, arch, reason, sizeof(reason)))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid grouped_gemm_single_launch spec for %s: %s", arch, reason);
    }
    return ckc_build_universal_gemm(b, &u, arch);
}

/* ===================================================================== *
 *  ckc_build_grouped_gemm_single_launch_new
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_grouped_gemm_single_launch_new(ckc_ir_builder_t* b,
                                                           const ckc_grouped_gemm_spec_t* spec,
                                                           const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        ckc_gemm_universal_spec_t u;
        char name[256];
        const char* base_name;
        size_t blen, slen;

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }

        /* Compute the single-launch kernel name = <base kernel_name>_single_launch.
         * Python: build_grouped_gemm_single_launch(spec).name on the renamed spec.
         * The kernel name is derived from the (renamed) UniversalGemmSpec. */
        u         = ckc_grouped_gemm_to_universal_spec(spec);
        base_name = (u.name != NULL) ? u.name : "";
        blen      = strlen(base_name);
        slen      = strlen("_single_launch");
        if(blen + slen >= sizeof(name))
        {
            return NULL;
        }
        memcpy(name, base_name, blen);
        memcpy(name + blen, "_single_launch", slen + 1);
        u.name    = name;
        u.batched = true;

        {
            char kname[256];
            if(ckc_gemm_universal_kernel_name(&u, kname, sizeof(kname)) != CKC_OK)
            {
                return NULL;
            }
            if(ckc_ir_builder_init(b, kname) != CKC_OK)
            {
                return NULL;
            }
        }
        return ckc_build_grouped_gemm_single_launch(b, spec, arch);
    });
}

/* ===================================================================== *
 *  signature helpers
 *
 *  ptr_dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16".
 * ===================================================================== */
static const char* grouped_gemm_ptr_dt(const ckc_grouped_gemm_spec_t* spec)
{
    const char* dt = (spec != NULL) ? spec->dtype : NULL;
    if(dt != NULL && (strcmp(dt, "f16") == 0 || strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0))
    {
        return dt;
    }
    return "f16";
}

/* grouped_gemm_signature(spec): (A,B,C ptr ; M,N,K i32). */
ckc_status_t ckc_grouped_gemm_signature(const ckc_grouped_gemm_spec_t* spec,
                                        ckc_arena_t* arena,
                                        const ckc_sig_entry_t** out_items,
                                        size_t* out_count)
{
    ckc_signature_builder_t sb;
    const char* ptr_dt;
    ckc_status_t st;

    if(spec == NULL || arena == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ptr_dt = grouped_gemm_ptr_dt(spec);

    ckc_signature_builder_ptr(&sb, "A", ptr_dt, NULL);
    ckc_signature_builder_ptr(&sb, "B", ptr_dt, NULL);
    ckc_signature_builder_ptr(&sb, "C", ptr_dt, NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* grouped_gemm_single_launch_signature(spec):
 *   (A,B,C ptr ; M,N,K i32 ; stride_a,stride_b,stride_c i32). */
ckc_status_t ckc_grouped_gemm_single_launch_signature(const ckc_grouped_gemm_spec_t* spec,
                                                      ckc_arena_t* arena,
                                                      const ckc_sig_entry_t** out_items,
                                                      size_t* out_count)
{
    ckc_signature_builder_t sb;
    const char* ptr_dt;
    ckc_status_t st;

    if(spec == NULL || arena == NULL)
    {
        return CKC_ERR_VALUE;
    }
    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
    {
        return st;
    }
    ptr_dt = grouped_gemm_ptr_dt(spec);

    ckc_signature_builder_ptr(&sb, "A", ptr_dt, NULL);
    ckc_signature_builder_ptr(&sb, "B", ptr_dt, NULL);
    ckc_signature_builder_ptr(&sb, "C", ptr_dt, NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b", "i32");
    ckc_signature_builder_scalar(&sb, "stride_c", "i32");

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  ckc_grouped_gemm_lower_to_llvm -- build the per-group base kernel + lower.
 *  Owns and frees its own IRBuilder (mirrors ckc_gemm_universal_lower_to_llvm).
 * ===================================================================== */
ckc_status_t ckc_grouped_gemm_lower_to_llvm(const ckc_grouped_gemm_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "grouped_gemm lower_to_llvm: null spec/out";
            size_t n      = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_grouped_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_grouped_gemm failed";
            }
            n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
