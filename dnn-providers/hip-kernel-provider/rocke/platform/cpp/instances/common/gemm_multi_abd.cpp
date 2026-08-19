// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/gemm_multi_abd.py.
 * See rocke/instance_gemm_multi_abd.h for the Python -> C mapping.
 *
 * gemm_multi_abd.py is a thin wrapper over gemm_multi_d.py with an extended
 * spec (AOperand / BOperand / DOperand tuples). In v1 it requires
 * num_a == num_b == 1 and delegates directly to build_gemm_multi_d after
 * renaming the base spec and wrapping the D operands in a GemmMultiDSpec; when
 * num_d == 0 it delegates straight to build_universal_gemm. The actual IR
 * generation is 100% reused from gemm_universal via the fused-epilogue path the
 * multi-D builder constructs.
 */
#include "rocke/instance_gemm_multi_abd.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, SignatureBuilder */
#include "rocke/instance_gemm_universal.h" /* rocke_build_universal_gemm, kernel_name */
#include "rocke/lower_llvm.h" /* rocke_lower_kernel_to_llvm_ex */

/* ------------------------------------------------------------------ helpers */

/* op string for one D operand ("add" / "mul"), mirroring the Python tuple's
 * second element. */
static const char* rocke_abd_d_op_str(const rocke_gemm_multi_d_op_t* d)
{
    return (d != NULL && d->op_is_mul) ? "mul" : "add";
}

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary (ckc::guard_builder in the *_new entry) catches it and records
 * status + message on the builder, so the extern "C" ABI is unchanged.
 * [[noreturn]] keeps the existing `rocke_abd_fail(...); return NULL;` call sites
 * valid -- the trailing return is simply never reached. */
[[noreturn]] static void rocke_abd_fail(rocke_ir_builder_t* b, rocke_status_t st, const char* msg)
{
    (void)b;
    ckc::raise_status(st, msg ? msg : "");
}

/* ------------------------------------------------------------------ defaults */

rocke_gemm_multi_abd_spec_t rocke_gemm_multi_abd_spec_default(void)
{
    rocke_gemm_multi_abd_spec_t s;
    memset(&s, 0, sizeof(s));
    s.base = rocke_gemm_universal_spec_default();
    s.a_operands = NULL; /* caller points at {"A","fp16"} default storage */
    s.num_a_operands = 0;
    s.b_operands = NULL; /* caller points at {"B","fp16"} default storage */
    s.num_b_operands = 0;
    s.num_d_operands = 0;
    s.d_dtype = "fp16";
    s.name = "rocke_gemm_multi_abd";
    s.d_load_kind = ROCKE_D_LOAD_VECTOR;
    return s;
}

size_t rocke_gemm_multi_abd_num_a(const rocke_gemm_multi_abd_spec_t* spec)
{
    return spec ? spec->num_a_operands : 0;
}

size_t rocke_gemm_multi_abd_num_b(const rocke_gemm_multi_abd_spec_t* spec)
{
    return spec ? spec->num_b_operands : 0;
}

size_t rocke_gemm_multi_abd_num_d(const rocke_gemm_multi_abd_spec_t* spec)
{
    return spec ? spec->num_d_operands : 0;
}

/* ------------------------------------------------------------ kernel_name() */

/* Python:
 *   d_suffix = "_".join(f"{n}{op}" for n, op in self.d_operands) or "noD"
 *   return kernel_name_join(
 *       self.name, self.base.kernel_name(),
 *       f"ma{num_a}", f"mb{num_b}", f"md{num_d}", d_suffix, self.d_dtype)
 */
rocke_status_t rocke_gemm_multi_abd_kernel_name(const rocke_gemm_multi_abd_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char base_name[256];
    char d_suffix[512];
    char part_ma[32];
    char part_mb[32];
    char part_md[32];
    const char* parts[6];
    rocke_status_t st;
    size_t i;
    size_t off;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* base.kernel_name() */
    st = rocke_gemm_universal_kernel_name(&spec->base, base_name, sizeof(base_name));
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* d_suffix = "_".join(f"{name}{op}") or "noD" */
    if(spec->num_d_operands == 0)
    {
        if(sizeof(d_suffix) < 4)
        {
            return ROCKE_ERR_VALUE;
        }
        memcpy(d_suffix, "noD", 4);
    }
    else
    {
        off = 0;
        for(i = 0; i < spec->num_d_operands; ++i)
        {
            const rocke_gemm_multi_d_op_t* d = &spec->d_operands[i];
            int n;
            if(i != 0)
            {
                if(off + 1 >= sizeof(d_suffix))
                {
                    return ROCKE_ERR_VALUE;
                }
                d_suffix[off++] = '_';
            }
            n = snprintf(d_suffix + off,
                         sizeof(d_suffix) - off,
                         "%s%s",
                         d->param_name ? d->param_name : "",
                         rocke_abd_d_op_str(d));
            if(n < 0 || (size_t)n >= sizeof(d_suffix) - off)
            {
                return ROCKE_ERR_VALUE;
            }
            off += (size_t)n;
        }
    }

    snprintf(part_ma, sizeof(part_ma), "ma%zu", spec->num_a_operands);
    snprintf(part_mb, sizeof(part_mb), "mb%zu", spec->num_b_operands);
    snprintf(part_md, sizeof(part_md), "md%zu", spec->num_d_operands);

    parts[0] = base_name;
    parts[1] = part_ma;
    parts[2] = part_mb;
    parts[3] = part_md;
    parts[4] = d_suffix;
    parts[5] = spec->d_dtype;

    return rocke_kernel_name_join(spec->name, parts, 6, NULL, NULL, 0, out, out_cap, NULL);
}

/* --------------------------------------------------------------- is_valid_spec
 *
 * Mirrors the Python rule order exactly so the returned reason byte-matches. */
bool rocke_gemm_multi_abd_is_valid_spec(const rocke_gemm_multi_abd_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    size_t i;
    size_t j;

    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(spec == NULL)
    {
        if(reason && reason_cap)
        {
            snprintf(reason, reason_cap, "spec is NULL");
        }
        return false;
    }

    if(spec->num_a_operands == 0)
    {
        if(reason && reason_cap)
        {
            snprintf(reason, reason_cap, "a_operands must contain at least one entry");
        }
        return false;
    }
    if(spec->num_b_operands == 0)
    {
        if(reason && reason_cap)
        {
            snprintf(reason, reason_cap, "b_operands must contain at least one entry");
        }
        return false;
    }
    if(spec->num_a_operands > ROCKE_GEMM_ABD_MAX_A)
    {
        if(reason && reason_cap)
        {
            snprintf(reason,
                     reason_cap,
                     "num_a %zu > MAX_A %d",
                     spec->num_a_operands,
                     ROCKE_GEMM_ABD_MAX_A);
        }
        return false;
    }
    if(spec->num_b_operands > ROCKE_GEMM_ABD_MAX_B)
    {
        if(reason && reason_cap)
        {
            snprintf(reason,
                     reason_cap,
                     "num_b %zu > MAX_B %d",
                     spec->num_b_operands,
                     ROCKE_GEMM_ABD_MAX_B);
        }
        return false;
    }
    if(spec->num_d_operands > ROCKE_GEMM_MULTI_D_MAX_D)
    {
        if(reason && reason_cap)
        {
            snprintf(reason,
                     reason_cap,
                     "num_d %zu > MAX_D %d",
                     spec->num_d_operands,
                     ROCKE_GEMM_MULTI_D_MAX_D);
        }
        return false;
    }
    if(spec->base.trait.epilogue == NULL || strcmp(spec->base.trait.epilogue, "cshuffle") != 0)
    {
        if(reason && reason_cap)
        {
            snprintf(reason,
                     reason_cap,
                     "GemmMultiAbd requires base.trait.epilogue='cshuffle'; got %s%s%s",
                     spec->base.trait.epilogue ? "'" : "",
                     spec->base.trait.epilogue ? spec->base.trait.epilogue : "None",
                     spec->base.trait.epilogue ? "'" : "");
        }
        return false;
    }

    /* Validate D ops + names (no duplicate, no reserved). Python uses a set for
     * dedup; we scan linearly which yields the same accept/reject decisions
     * (op is encoded by op_is_mul so it is always "add"/"mul" -- the Python
     * "op not in {'add','mul'}" branch cannot fire here; the per-D op check is
     * vacuously satisfied by the typed enum field). */
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        const rocke_gemm_multi_d_op_t* d = &spec->d_operands[i];
        const char* name = d->param_name;
        /* duplicate check against earlier entries (Python: name in seen). */
        for(j = 0; j < i; ++j)
        {
            const char* prev = spec->d_operands[j].param_name;
            if(name && prev && strcmp(name, prev) == 0)
            {
                if(reason && reason_cap)
                {
                    snprintf(reason, reason_cap, "duplicate D param_name '%s'", name);
                }
                return false;
            }
        }
        if(name
           && (strcmp(name, "M") == 0 || strcmp(name, "N") == 0 || strcmp(name, "K") == 0
               || strcmp(name, "C") == 0))
        {
            if(reason && reason_cap)
            {
                snprintf(reason, reason_cap, "D param_name '%s' reserved", name);
            }
            return false;
        }
    }

    /* Validate A / B name uniqueness across the three pools. Python sorts the
     * overlapping names before reporting; we report the first overlap found in
     * A-pool order, which is the common single-element case (no overlap). */
    /* A & B */
    {
        size_t ai;
        for(ai = 0; ai < spec->num_a_operands; ++ai)
        {
            const char* an = spec->a_operands[ai].name;
            size_t bi;
            for(bi = 0; bi < spec->num_b_operands; ++bi)
            {
                const char* bn = spec->b_operands[bi].name;
                if(an && bn && strcmp(an, bn) == 0)
                {
                    if(reason && reason_cap)
                    {
                        snprintf(reason, reason_cap, "A and B share param_names ['%s']", an);
                    }
                    return false;
                }
            }
        }
    }
    /* A & D */
    {
        size_t ai;
        for(ai = 0; ai < spec->num_a_operands; ++ai)
        {
            const char* an = spec->a_operands[ai].name;
            size_t di;
            for(di = 0; di < spec->num_d_operands; ++di)
            {
                const char* dn = spec->d_operands[di].param_name;
                if(an && dn && strcmp(an, dn) == 0)
                {
                    if(reason && reason_cap)
                    {
                        snprintf(reason, reason_cap, "A and D share param_names ['%s']", an);
                    }
                    return false;
                }
            }
        }
    }
    /* B & D */
    {
        size_t bi;
        for(bi = 0; bi < spec->num_b_operands; ++bi)
        {
            const char* bn = spec->b_operands[bi].name;
            size_t di;
            for(di = 0; di < spec->num_d_operands; ++di)
            {
                const char* dn = spec->d_operands[di].param_name;
                if(bn && dn && strcmp(bn, dn) == 0)
                {
                    if(reason && reason_cap)
                    {
                        snprintf(reason, reason_cap, "B and D share param_names ['%s']", bn);
                    }
                    return false;
                }
            }
        }
    }

    /* Arch-aware base GEMM validation (delegates to gemm_universal). */
    {
        char base_reason[256];
        bool ok_base = rocke_gemm_universal_is_valid_spec(
            &spec->base, arch, base_reason, sizeof(base_reason));
        if(!ok_base)
        {
            if(reason && reason_cap)
            {
                snprintf(reason, reason_cap, "base GEMM spec invalid: %s", base_reason);
            }
            return false;
        }
    }

    if(reason && reason_cap)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

/* -------------------------------------------------------- build_gemm_multi_abd
 *
 * Python:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...)
 *   if num_a > 1 or num_b > 1: raise NotImplementedError(...)
 *   base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
 *   md_spec = GemmMultiDSpec(base=base_renamed, d_operands=..., d_dtype=...,
 *                            name=spec.kernel_name(), d_load_kind=...)
 *   if num_d == 0: return build_universal_gemm(base_renamed, arch)
 *   return build_gemm_multi_d(md_spec, arch)
 */
rocke_kernel_def_t* rocke_build_gemm_multi_abd(rocke_ir_builder_t* b,
                                               rocke_arena_t* arena,
                                               const rocke_gemm_multi_abd_spec_t* spec,
                                               const char* arch)
{
    char reason[256];
    char kname[512];
    char* kname_owned;
    rocke_status_t st;
    rocke_gemm_universal_spec_t base_renamed;
    rocke_gemm_multi_d_spec_t md_spec;
    size_t i;

    if(b == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(spec == NULL)
    {
        rocke_abd_fail(b, ROCKE_ERR_VALUE, "spec is NULL");
        return NULL;
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!rocke_gemm_multi_abd_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        char msg[384];
        snprintf(msg, sizeof(msg), "invalid GemmMultiAbd spec for %s: %s", arch, reason);
        rocke_abd_fail(b, ROCKE_ERR_VALUE, msg);
        return NULL;
    }

    /* v1 gap: num_a > 1 or num_b > 1 -> NotImplementedError. */
    if(spec->num_a_operands > 1 || spec->num_b_operands > 1)
    {
        char msg[256];
        snprintf(msg,
                 sizeof(msg),
                 "multi-A (%zu) / multi-B (%zu) load-combine is a v2 gap; "
                 "today only num_a == num_b == 1 is implemented.",
                 spec->num_a_operands,
                 spec->num_b_operands);
        rocke_abd_fail(b, ROCKE_ERR_NOTIMPL, msg);
        return NULL;
    }

    /* base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name()) */
    st = rocke_gemm_multi_abd_kernel_name(spec, kname, sizeof(kname));
    if(st != ROCKE_OK)
    {
        rocke_abd_fail(b, st, "kernel_name() overflow");
        return NULL;
    }
    /* The renamed name must outlive this stack frame: the universal/multi-D
     * builders store the const char* on the spec and use it during build. Copy
     * it into the arena (the same arena the multi-D path threads its epilogue
     * allocations through). When num_d == 0 / arena == NULL we point straight at
     * the stack buffer because build runs entirely within this frame. */
    if(arena != NULL)
    {
        kname_owned = rocke_arena_strdup(arena, kname);
        if(kname_owned == NULL)
        {
            rocke_abd_fail(b, ROCKE_ERR_OOM, "kernel_name() arena strdup failed");
            return NULL;
        }
    }
    else
    {
        kname_owned = kname;
    }

    base_renamed = spec->base;
    base_renamed.name = kname_owned;

    if(spec->num_d_operands == 0)
    {
        /* No D operands -> plain GEMM. Delegate to build_universal_gemm. */
        return rocke_build_universal_gemm(b, &base_renamed, arch);
    }

    /* md_spec = GemmMultiDSpec(base=base_renamed, d_operands=..., d_dtype=...,
     *                          name=spec.kernel_name(), d_load_kind=...) */
    md_spec = rocke_gemm_multi_d_spec_default();
    md_spec.base = base_renamed;
    md_spec.num_d_operands = spec->num_d_operands;
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        md_spec.d_operands[i] = spec->d_operands[i];
    }
    md_spec.d_dtype = spec->d_dtype;
    md_spec.name = kname_owned;
    md_spec.d_load_kind = spec->d_load_kind;

    if(arena == NULL)
    {
        rocke_abd_fail(
            b, ROCKE_ERR_VALUE, "build_gemm_multi_abd: num_d > 0 requires a non-NULL arena");
        return NULL;
    }
    /* Delegate to the multi-D 4-arg builder seam (the facade renamed it to
     * rocke_build_gemm_multi_d_builder; the 2-arg rocke_build_gemm_multi_d owns its
     * own IRBuilder/arena and would not build into ours). */
    return rocke_build_gemm_multi_d_builder(b, arena, &md_spec, arch);
}

/* ===================================================================== *
 *  rocke_build_gemm_multi_abd_new -- init the builder with spec.kernel_name()
 *  then build. (public convenience.)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_gemm_multi_abd_new(rocke_ir_builder_t* b,
                                                   rocke_arena_t* arena,
                                                   const rocke_gemm_multi_abd_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_gemm_multi_abd_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_gemm_multi_abd(b, arena, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_gemm_multi_abd_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder and arena.
 * ===================================================================== */
rocke_status_t rocke_gemm_multi_abd_lower_to_llvm(const rocke_gemm_multi_abd_spec_t* spec,
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
    int arena_inited = 0;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
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

    /* Back the fused-epilogue + op-chain allocations the multi-D path needs.
     * Block size mirrors the arena defaults used elsewhere in the port. */
    if(rocke_arena_init(&arena, 0) != 0)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: arena init failed";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_OOM;
    }
    arena_inited = 1;

    kernel = rocke_build_gemm_multi_abd_new(&b, &arena, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_gemm_multi_abd failed";
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
        if(arena_inited)
        {
            rocke_arena_destroy(&arena);
        }
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    if(arena_inited)
    {
        rocke_arena_destroy(&arena);
    }
    return st;
}

/* ----------------------------------------------------------------- signature
 *
 * Python:
 *   sb = SignatureBuilder()
 *   for name, dtype in a_operands: sb.ptr(name, dtype)
 *   for name, dtype in b_operands: sb.ptr(name, dtype)
 *   sb.ptr("C", base.data.dtype_c).scalar("M","i32").scalar("N","i32")
 *     .scalar("K","i32")
 *   if base.batched: stride_a / stride_b / stride_c (i32)
 *   for name, _op in d_operands: sb.ptr(name, d_dtype)
 *   return sb.build()
 */
rocke_status_t rocke_gemm_multi_abd_signature(const rocke_gemm_multi_abd_spec_t* spec,
                                              rocke_arena_t* arena,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    size_t i;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* for name, dtype in a_operands: sb.ptr(name, dtype) */
    for(i = 0; i < spec->num_a_operands; ++i)
    {
        rocke_signature_builder_ptr(&sb, spec->a_operands[i].name, spec->a_operands[i].dtype, NULL);
    }
    /* for name, dtype in b_operands: sb.ptr(name, dtype) */
    for(i = 0; i < spec->num_b_operands; ++i)
    {
        rocke_signature_builder_ptr(&sb, spec->b_operands[i].name, spec->b_operands[i].dtype, NULL);
    }
    /* sb.ptr("C", base.data.dtype_c).scalar("M","i32").scalar("N","i32")
     *   .scalar("K","i32") */
    rocke_signature_builder_ptr(&sb, "C", spec->base.data.dtype_c, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");

    /* if base.batched: stride_a / stride_b / stride_c (i32) */
    if(spec->base.batched)
    {
        rocke_signature_builder_scalar(&sb, "stride_a", "i32");
        rocke_signature_builder_scalar(&sb, "stride_b", "i32");
        rocke_signature_builder_scalar(&sb, "stride_c", "i32");
    }

    /* for name, _op in d_operands: sb.ptr(name, d_dtype) */
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        rocke_signature_builder_ptr(&sb, spec->d_operands[i].param_name, spec->d_dtype, NULL);
    }

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ---------------------------------------------------------------------- grid
 *
 * Python:
 *   return gemm_multi_d_grid(
 *       GemmMultiDSpec(base=spec.base,
 *                      d_operands=spec.d_operands or (("D0","add"),),
 *                      d_dtype=spec.d_dtype),
 *       m, n, batch)
 */
rocke_status_t rocke_gemm_multi_abd_grid(
    const rocke_gemm_multi_abd_spec_t* spec, int m, int n, int batch, int out[3])
{
    rocke_gemm_multi_d_spec_t md_spec;
    size_t i;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    md_spec = rocke_gemm_multi_d_spec_default();
    md_spec.base = spec->base;
    if(spec->num_d_operands != 0)
    {
        md_spec.num_d_operands = spec->num_d_operands;
        for(i = 0; i < spec->num_d_operands; ++i)
        {
            md_spec.d_operands[i] = spec->d_operands[i];
        }
    }
    else
    {
        /* spec.d_operands or (("D0","add"),): a single default D so the
         * synthesised GemmMultiDSpec is well-formed. The grid only reads
         * base.tile / base.batched, so the D list value is immaterial to the
         * result, but we mirror the substitution faithfully. */
        md_spec.num_d_operands = 1;
        md_spec.d_operands[0].param_name = "D0";
        md_spec.d_operands[0].op_is_mul = false; /* "add" */
    }
    md_spec.d_dtype = spec->d_dtype;
    /* name / d_load_kind left at GemmMultiDSpec defaults (unused by grid). */

    return rocke_gemm_multi_d_grid(&md_spec, m, n, batch, out);
}
