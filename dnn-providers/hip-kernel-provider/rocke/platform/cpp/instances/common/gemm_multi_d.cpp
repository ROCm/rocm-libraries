// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_multi_d.c -- task-mandated public facade for the C99 port of
 * rocke/instances/common/gemm_multi_d.py.
 *
 * Implements:
 *   rocke_gemm_multi_d_spec_new      -- construct a GemmMultiDSpec (arena-owned)
 *   rocke_build_gemm_multi_d         -- KernelDef rocke_build_gemm_multi_d(spec, arch)
 *   rocke_build_gemm_multi_d_into    -- build into a caller-supplied builder/arena
 *   rocke_gemm_multi_d_kernel_free   -- tear down a self-contained build
 *   rocke_gemm_multi_d_lower_to_llvm -- build + lower to .ll convenience
 *
 * The heavy lifting (is_valid_spec, the _MultiDEpilogue apply_vec sequence,
 * _build_fused_epilogue, kernel_name) lives in the full faithful port
 * helper_rocke.instances.common.gemm_multi_d.c; this TU is the thin facade
 * over it that matches the workflow's mandated entry shape. Its build path is
 * a byte-faithful re-statement of Python build_gemm_multi_d (same op order):
 *   is_valid_spec -> _build_fused_epilogue -> dataclasses.replace(name=...)
 *   -> object.__setattr__(_fused_epilogue) -> build_universal_gemm.
 */

/* Bring in the full port's spec/op/load-kind types + helpers. Its own 4-arg
 * rocke_build_gemm_multi_d is renamed to rocke_build_gemm_multi_d_builder by the
 * facade header so the 2-arg entry below owns the canonical symbol. */
#include "rocke/instance_gemm_multi_d.h"

#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  Lowered-symbol name
 * ===================================================================== *
 *
 * The kernel symbol the IR lowers under is NOT spec.kernel_name() but the name
 * build_universal_gemm derives from the renamed base spec:
 *
 *   base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
 *   symbol       = base_renamed.kernel_name()
 *                = <spec.kernel_name()>_<dtype_a>_t..._<pipeline_sched_epi>...
 *
 * i.e. UniversalGemmSpec.kernel_name() re-appends the tile/pipeline suffix to
 * the already-multi-D name. The builder must be initialised with THIS symbol
 * (rocke_build_universal_gemm trusts the pre-init'd name and does not re-derive
 * it), so the lowered .ll matches the Python byte-for-byte. */
static rocke_status_t
    rocke_md_lowered_symbol_name(const rocke_gemm_multi_d_spec_t* spec, char* out, size_t out_cap)
{
    char md_name[512];
    rocke_gemm_universal_spec_t base_renamed;
    rocke_status_t st;

    st = rocke_gemm_multi_d_kernel_name(spec, md_name, sizeof(md_name));
    if(st != ROCKE_OK)
    {
        return st;
    }
    base_renamed = spec->base;
    base_renamed.name = md_name;
    return rocke_gemm_universal_kernel_name(&base_renamed, out, out_cap);
}

/* ===================================================================== *
 *  rocke_gemm_multi_d_spec_new
 * ===================================================================== *
 *
 * Python analogue: constructing the GemmMultiDSpec dataclass
 *   GemmMultiDSpec(base=..., d_operands=((name, op), ...), d_dtype=...,
 *                  name=..., d_load_kind=...)
 *
 * Defaults match the dataclass: d_dtype "fp16", name "rocke_gemm_multi_d",
 * d_load_kind "vector" (any unrecognised string also resolves to "vector",
 * mirroring _build_fused_epilogue's "vector (default) or any unrecognised
 * value" branch). */
static rocke_d_load_kind_t rocke_md_parse_load_kind(const char* s)
{
    if(s != NULL)
    {
        if(strcmp(s, "stock") == 0)
        {
            return ROCKE_D_LOAD_STOCK;
        }
        if(strcmp(s, "tiled") == 0)
        {
            return ROCKE_D_LOAD_TILED;
        }
    }
    /* "vector" (default) or any unrecognised value. */
    return ROCKE_D_LOAD_VECTOR;
}

rocke_gemm_multi_d_spec_t*
    rocke_gemm_multi_d_spec_new(rocke_arena_t* arena,
                                const rocke_gemm_universal_spec_t* base,
                                const rocke_gemm_multi_d_operand_t* d_operands,
                                int num_d,
                                const char* d_dtype,
                                const char* name,
                                const char* d_load_kind)
{
    rocke_gemm_multi_d_spec_t* spec;
    int i;

    if(arena == NULL || base == NULL)
    {
        return NULL;
    }
    /* num_d must be representable as a non-empty operand list within MAX_D.
     * (The cshuffle / uniqueness / reserved-name checks are deferred to
     * is_valid_spec at build time, exactly like the Python dataclass.) */
    if(num_d <= 0 || num_d > ROCKE_GEMM_MULTI_D_MAX_D)
    {
        return NULL;
    }
    if(d_operands == NULL)
    {
        return NULL;
    }

    spec = (rocke_gemm_multi_d_spec_t*)rocke_arena_alloc(arena, sizeof(*spec));
    if(spec == NULL)
    {
        return NULL;
    }

    /* base= : copy by value (dataclass holds the base spec). */
    spec->base = *base;

    /* d_operands= : translate each (param_name, "add"|"mul") tuple. The op
     * string is validated structurally here (membership in {"add","mul"});
     * param_name strings are duplicated into the arena so the spec owns them. */
    spec->num_d_operands = (size_t)num_d;
    for(i = 0; i < num_d; ++i)
    {
        const char* pname = d_operands[i].param_name;
        const char* op = d_operands[i].op;
        char* pcopy;
        bool is_mul;

        if(pname == NULL || op == NULL)
        {
            return NULL;
        }
        if(strcmp(op, "add") == 0)
        {
            is_mul = false;
        }
        else if(strcmp(op, "mul") == 0)
        {
            is_mul = true;
        }
        else
        {
            /* op not in {"add","mul"} -- Python raises in _build_fused_epilogue
             * / is_valid_spec; reject structurally up front. */
            return NULL;
        }
        pcopy = rocke_arena_strdup(arena, pname);
        if(pcopy == NULL)
        {
            return NULL;
        }
        spec->d_operands[i].param_name = pcopy;
        spec->d_operands[i].op_is_mul = is_mul;
    }
    /* Zero the unused tail so kernel_name / iteration stay well-defined. */
    for(i = num_d; i < ROCKE_GEMM_MULTI_D_MAX_D; ++i)
    {
        spec->d_operands[i].param_name = NULL;
        spec->d_operands[i].op_is_mul = false;
    }

    /* d_dtype= : default "fp16". */
    spec->d_dtype = (d_dtype != NULL) ? rocke_arena_strdup(arena, d_dtype) : "fp16";
    if(spec->d_dtype == NULL)
    {
        return NULL;
    }
    /* name= : default "rocke_gemm_multi_d". */
    spec->name = (name != NULL) ? rocke_arena_strdup(arena, name) : "rocke_gemm_multi_d";
    if(spec->name == NULL)
    {
        return NULL;
    }
    /* d_load_kind= : default "vector". */
    spec->d_load_kind = rocke_md_parse_load_kind(d_load_kind);

    return spec;
}

/* ===================================================================== *
 *  rocke_build_gemm_multi_d_into  (byte-faithful build_gemm_multi_d body)
 * ===================================================================== *
 *
 * This is the same sequence as the full port's 4-arg builder (now linked under
 * rocke_build_gemm_multi_d_builder). Re-stated here so the facade owns the build
 * seam without taking a second copy of the implementation: it simply forwards
 * to the renamed full-port builder, which performs:
 *   1. rocke_gemm_multi_d_is_valid_spec(spec, arch)
 *   2. rocke_gemm_multi_d_build_fused_epilogue(arena, spec)
 *   3. kernel_name + dataclasses.replace(base, name=...)
 *   4. rocke_gemm_universal_spec_set_fused_epilogue(&base_renamed, fused)
 *   5. rocke_build_universal_gemm(b, &base_renamed, arch)
 */
rocke_kernel_def_t* rocke_build_gemm_multi_d_into(rocke_ir_builder_t* b,
                                                  rocke_arena_t* arena,
                                                  const rocke_gemm_multi_d_spec_t* spec,
                                                  const char* arch)
{
    if(b == NULL || arena == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return rocke_build_gemm_multi_d_builder(b, arena, spec, arch);
}

/* ===================================================================== *
 *  Self-contained build ownership registry
 * ===================================================================== *
 *
 * The KernelDef returned by rocke_build_gemm_multi_d lives in its IRBuilder's
 * arena; freeing the builder frees the kernel. Since rocke_kernel_def_t carries
 * no back-pointer to its owning builder, we keep a tiny registry mapping the
 * returned kernel to the heap-allocated builder (+ build arena) so
 * rocke_gemm_multi_d_kernel_free can tear both down.
 */
typedef struct rocke_md_owner
{
    rocke_kernel_def_t* kernel;
    rocke_ir_builder_t* builder; /* heap-allocated, owns the kernel arena */
    rocke_arena_t* arena; /* heap-allocated epilogue/op-chain arena */
} rocke_md_owner_t;

#define ROCKE_MD_OWNER_MAX 256
static rocke_md_owner_t g_md_owners[ROCKE_MD_OWNER_MAX];
static size_t g_md_owner_count = 0;

static void rocke_md_owner_register(rocke_kernel_def_t* kernel,
                                    rocke_ir_builder_t* builder,
                                    rocke_arena_t* arena)
{
    if(g_md_owner_count < ROCKE_MD_OWNER_MAX)
    {
        g_md_owners[g_md_owner_count].kernel = kernel;
        g_md_owners[g_md_owner_count].builder = builder;
        g_md_owners[g_md_owner_count].arena = arena;
        ++g_md_owner_count;
    }
    /* If the registry is full the kernel is simply not tracked; freeing it is
     * then a no-op (intentional, bounded leak under pathological churn). */
}

/* ===================================================================== *
 *  rocke_build_gemm_multi_d  (KernelDef rocke_build_gemm_multi_d(spec, arch))
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_gemm_multi_d(rocke_gemm_multi_d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder((rocke_ir_builder_t*)nullptr, [&]() -> rocke_kernel_def_t* {
        rocke_ir_builder_t* b;
        rocke_arena_t* arena;
        rocke_kernel_def_t* kernel;
        char name_buf[512];

        if(spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* The builder must be created with base_renamed.kernel_name() (the universal
         * build does not re-init it), matching the Python where build_universal_gemm
         * lowers under the renamed base spec's kernel symbol. */
        if(rocke_md_lowered_symbol_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
        {
            return NULL;
        }

        b = (rocke_ir_builder_t*)calloc(1, sizeof(*b));
        arena = (rocke_arena_t*)calloc(1, sizeof(*arena));
        if(b == NULL || arena == NULL)
        {
            free(b);
            free(arena);
            return NULL;
        }
        if(rocke_arena_init(arena, 0) != 0)
        {
            free(b);
            free(arena);
            return NULL;
        }
        if(rocke_ir_builder_init(b, name_buf) != ROCKE_OK)
        {
            rocke_arena_destroy(arena);
            free(b);
            free(arena);
            return NULL;
        }

        kernel = rocke_build_gemm_multi_d_into(b, arena, spec, arch);
        if(kernel == NULL)
        {
            rocke_ir_builder_free(b);
            rocke_arena_destroy(arena);
            free(b);
            free(arena);
            return NULL;
        }

        rocke_md_owner_register(kernel, b, arena);
        return kernel;
    });
}

void rocke_gemm_multi_d_kernel_free(rocke_kernel_def_t* kernel)
{
    size_t i;

    if(kernel == NULL)
    {
        return;
    }
    for(i = 0; i < g_md_owner_count; ++i)
    {
        if(g_md_owners[i].kernel == kernel)
        {
            rocke_ir_builder_free(g_md_owners[i].builder);
            rocke_arena_destroy(g_md_owners[i].arena);
            free(g_md_owners[i].builder);
            free(g_md_owners[i].arena);
            /* Compact: move the last entry into this slot. */
            g_md_owners[i] = g_md_owners[g_md_owner_count - 1];
            --g_md_owner_count;
            return;
        }
    }
    /* Unrecognised kernel (not produced by rocke_build_gemm_multi_d, or registry
     * was full): no-op. */
}

/* ===================================================================== *
 *  rocke_gemm_multi_d_lower_to_llvm  (build + lower to .ll convenience)
 * ===================================================================== *
 *
 * Mirrors rocke_gemm_universal_lower_to_llvm: owns an IRBuilder + arena, builds,
 * lowers, then tears everything down. The KernelDef does not escape, so the
 * registry above is bypassed.
 */
static void rocke_md_set_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0 || m == NULL)
    {
        return;
    }
    n = strlen(m);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, m, n);
    err[n] = '\0';
}

rocke_status_t rocke_gemm_multi_d_lower_to_llvm(const rocke_gemm_multi_d_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_arena_t arena;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;
    char name_buf[512];

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_md_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    if(rocke_md_lowered_symbol_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        rocke_md_set_err(err, err_cap, "lower_to_llvm: kernel_name failed");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_arena_init(&arena, 0) != 0)
    {
        rocke_md_set_err(err, err_cap, "lower_to_llvm: arena_init failed");
        return ROCKE_ERR_VALUE;
    }
    if(rocke_ir_builder_init(&b, name_buf) != ROCKE_OK)
    {
        rocke_arena_destroy(&arena);
        rocke_md_set_err(err, err_cap, "lower_to_llvm: builder_init failed");
        return ROCKE_ERR_VALUE;
    }

    kernel = rocke_build_gemm_multi_d_into(&b, &arena, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        {
            const char* m = rocke_ir_builder_error(&b);
            rocke_md_set_err(err, err_cap, (m != NULL) ? m : "build_gemm_multi_d failed");
        }
        rocke_ir_builder_free(&b);
        rocke_arena_destroy(&arena);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    rocke_arena_destroy(&arena);
    return st;
}
