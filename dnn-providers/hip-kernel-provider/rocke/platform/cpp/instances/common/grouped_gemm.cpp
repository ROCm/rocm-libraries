// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_grouped_gemm.c -- C99 port of
 * rocke/instances/common/grouped_gemm.py.
 *
 * GroupedGemmSpec is a thin wrapper around UniversalGemmSpec: every entry point
 * converts the grouped spec to a UniversalGemmSpec and delegates to the
 * already-ported gemm_universal machinery. No new IR is emitted, so the lowered
 * IR is byte-identical to the universal-GEMM body the Python delegates to.
 */
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_grouped_gemm.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err (sticky-error helper) */

/* ===================================================================== *
 *  rocke_grouped_gemm_spec_default
 *
 *  Python GroupedGemmSpec field defaults: wave_size=64, block_size=0,
 *  dtype="fp16". tile/trait carry their own dataclass defaults; trait reuses
 *  the universal-GEMM trait defaults (TraitSpec). The tile geometry is required
 *  and left zeroed for the caller to fill (matching the universal default).
 * ===================================================================== */
rocke_grouped_gemm_spec_t rocke_grouped_gemm_spec_default(void)
{
    rocke_grouped_gemm_spec_t s;
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();

    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.tile = u.tile; /* TileSpec defaults (warp_k=1, warp_tile 32/32/16) */
    s.trait = u.trait; /* TraitSpec defaults                               */
    s.wave_size = 64;
    s.block_size = 0;
    s.dtype = "fp16";
    return s;
}

/* ===================================================================== *
 *  rocke_grouped_gemm_spec_finalize
 *
 *  __post_init__ -> WarpTileBlockSizeMixin._init_block_size(): if block_size==0
 *  derive warp_m*warp_n*warp_k*wave_size. Idempotent.
 * ===================================================================== */
void rocke_grouped_gemm_spec_finalize(rocke_grouped_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    spec->block_size = rocke_warp_tile_init_block_size(
        spec->block_size, spec->tile.warp_m, spec->tile.warp_n, spec->tile.warp_k, spec->wave_size);
}

/* ===================================================================== *
 *  rocke_grouped_gemm_data_spec
 *
 *  GroupedGemmSpec._data_spec():
 *      dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
 *      return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)
 *  The remaining DataSpec fields (dtype_acc, layout) take their universal-GEMM
 *  defaults ("fp32" / "RCR").
 * ===================================================================== */
rocke_gemm_data_spec_t rocke_grouped_gemm_data_spec(const rocke_grouped_gemm_spec_t* spec)
{
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
    rocke_gemm_data_spec_t d = u.data; /* dtype_acc="fp32", layout="RCR" defaults */
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
 *  rocke_grouped_gemm_to_universal_spec
 *
 *  GroupedGemmSpec.to_universal_spec():
 *      UniversalGemmSpec(name, tile, trait, data=_data_spec(),
 *                        wave_size, block_size, batched=False)
 *  The returned spec is finalized (block_size derived) so it is build-ready.
 * ===================================================================== */
rocke_gemm_universal_spec_t
    rocke_grouped_gemm_to_universal_spec(const rocke_grouped_gemm_spec_t* spec)
{
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();

    if(spec == NULL)
    {
        return u;
    }
    u.name = spec->name;
    u.tile = spec->tile;
    u.trait = spec->trait;
    u.data = rocke_grouped_gemm_data_spec(spec);
    u.wave_size = spec->wave_size;
    u.block_size = spec->block_size;
    u.batched = false;

    /* Python passes self.block_size (already derived by __post_init__). Mirror
     * that: ensure the universal spec's block_size matches the grouped spec's
     * finalized value. */
    rocke_gemm_universal_spec_finalize(&u);
    return u;
}

/* ===================================================================== *
 *  rocke_grouped_gemm_kernel_name
 *
 *  GroupedGemmSpec.kernel_name() == to_universal_spec().kernel_name().
 * ===================================================================== */
rocke_status_t
    rocke_grouped_gemm_kernel_name(const rocke_grouped_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u;

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    u = rocke_grouped_gemm_to_universal_spec(spec);
    return rocke_gemm_universal_kernel_name(&u, out, out_cap);
}

/* ===================================================================== *
 *  rocke_grouped_gemm_is_valid_spec
 *
 *  is_valid_spec(spec, arch): delegates to
 *  rocke.instances.common.gemm_universal.is_valid_spec(to_universal_spec()).
 * ===================================================================== */
bool rocke_grouped_gemm_is_valid_spec(const rocke_grouped_gemm_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    rocke_gemm_universal_spec_t u;

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
    u = rocke_grouped_gemm_to_universal_spec(spec);
    return rocke_gemm_universal_is_valid_spec(&u, arch, reason, reason_cap);
}

/* ===================================================================== *
 *  rocke_build_grouped_gemm
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
rocke_kernel_def_t* rocke_build_grouped_gemm(rocke_ir_builder_t* b,
                                             const rocke_grouped_gemm_spec_t* spec,
                                             const char* arch)
{
    rocke_gemm_universal_spec_t u;
    char reason[ROCKE_ERR_MSG_CAP];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    u = rocke_grouped_gemm_to_universal_spec(spec);
    if(!rocke_gemm_universal_is_valid_spec(&u, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid grouped_gemm spec for %s: %s", arch, reason);
    }
    return rocke_build_universal_gemm(b, &u, arch);
}

/* ===================================================================== *
 *  rocke_build_grouped_gemm_new -- init builder with spec.kernel_name() + build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_grouped_gemm_new(rocke_ir_builder_t* b,
                                                 const rocke_grouped_gemm_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_grouped_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_grouped_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_build_grouped_gemm_single_launch
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
rocke_kernel_def_t* rocke_build_grouped_gemm_single_launch(rocke_ir_builder_t* b,
                                                           const rocke_grouped_gemm_spec_t* spec,
                                                           const char* arch)
{
    rocke_gemm_universal_spec_t u;
    char reason[ROCKE_ERR_MSG_CAP];
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

    u = rocke_grouped_gemm_to_universal_spec(spec);

    /* name = base_spec.name + "_single_launch" */
    base_name = (u.name != NULL) ? u.name : "";
    blen = strlen(base_name);
    slen = strlen("_single_launch");
    if(blen + slen >= sizeof(name))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "grouped_gemm_single_launch: name too long");
    }
    memcpy(name, base_name, blen);
    memcpy(name + blen, "_single_launch", slen + 1);
    u.name = name;
    u.batched = true;

    if(!rocke_gemm_universal_is_valid_spec(&u, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid grouped_gemm_single_launch spec for %s: %s", arch, reason);
    }
    return rocke_build_universal_gemm(b, &u, arch);
}

/* ===================================================================== *
 *  rocke_build_grouped_gemm_single_launch_new
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_grouped_gemm_single_launch_new(
    rocke_ir_builder_t* b, const rocke_grouped_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gemm_universal_spec_t u;
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
        u = rocke_grouped_gemm_to_universal_spec(spec);
        base_name = (u.name != NULL) ? u.name : "";
        blen = strlen(base_name);
        slen = strlen("_single_launch");
        if(blen + slen >= sizeof(name))
        {
            return NULL;
        }
        memcpy(name, base_name, blen);
        memcpy(name + blen, "_single_launch", slen + 1);
        u.name = name;
        u.batched = true;

        {
            char kname[256];
            if(rocke_gemm_universal_kernel_name(&u, kname, sizeof(kname)) != ROCKE_OK)
            {
                return NULL;
            }
            if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
            {
                return NULL;
            }
        }
        return rocke_build_grouped_gemm_single_launch(b, spec, arch);
    });
}

/* ===================================================================== *
 *  signature helpers
 *
 *  ptr_dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16".
 * ===================================================================== */
static const char* grouped_gemm_ptr_dt(const rocke_grouped_gemm_spec_t* spec)
{
    const char* dt = (spec != NULL) ? spec->dtype : NULL;
    if(dt != NULL && (strcmp(dt, "f16") == 0 || strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0))
    {
        return dt;
    }
    return "f16";
}

/* grouped_gemm_signature(spec): (A,B,C ptr ; M,N,K i32). */
rocke_status_t rocke_grouped_gemm_signature(const rocke_grouped_gemm_spec_t* spec,
                                            rocke_arena_t* arena,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    const char* ptr_dt;
    rocke_status_t st;

    if(spec == NULL || arena == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    ptr_dt = grouped_gemm_ptr_dt(spec);

    rocke_signature_builder_ptr(&sb, "A", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "B", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "C", ptr_dt, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* grouped_gemm_single_launch_signature(spec):
 *   (A,B,C ptr ; M,N,K i32 ; stride_a,stride_b,stride_c i32). */
rocke_status_t rocke_grouped_gemm_single_launch_signature(const rocke_grouped_gemm_spec_t* spec,
                                                          rocke_arena_t* arena,
                                                          const rocke_sig_entry_t** out_items,
                                                          size_t* out_count)
{
    rocke_signature_builder_t sb;
    const char* ptr_dt;
    rocke_status_t st;

    if(spec == NULL || arena == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    ptr_dt = grouped_gemm_ptr_dt(spec);

    rocke_signature_builder_ptr(&sb, "A", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "B", ptr_dt, NULL);
    rocke_signature_builder_ptr(&sb, "C", ptr_dt, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "stride_c", "i32");

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  rocke_grouped_gemm_lower_to_llvm -- build the per-group base kernel + lower.
 *  Owns and frees its own IRBuilder (mirrors rocke_gemm_universal_lower_to_llvm).
 * ===================================================================== */
rocke_status_t rocke_grouped_gemm_lower_to_llvm(const rocke_grouped_gemm_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "grouped_gemm lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_grouped_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
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
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
