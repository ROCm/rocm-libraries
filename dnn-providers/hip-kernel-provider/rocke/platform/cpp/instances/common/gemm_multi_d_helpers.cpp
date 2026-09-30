// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.instances.common.gemm_multi_d.c -- C99 port of
 * rocke/instances/common/gemm_multi_d.py.
 *
 * Faithful translation of the multiple-D GEMM kernel instance: the spec, its
 * validity gate, the per-D fused-epilogue composition (incl. the _MultiDEpilogue
 * apply_vec op sequence), the IR build delegation, the kernarg signature, and
 * the launch grid. The IR builder call sequence in apply_vec is byte-faithful
 * to the Python so the lowered output matches.
 *
 * See the header for the symbol-by-symbol mapping and the peer fuse-layer
 * dependency note.
 */
/* Include the facade header (not the helper header directly): it applies the
 * `#define rocke_build_gemm_multi_d rocke_build_gemm_multi_d_builder` rename before
 * pulling in this module's own helper header, so the 4-arg builder defined below
 * emits the rocke_build_gemm_multi_d_builder symbol the facade's 2-arg
 * rocke_build_gemm_multi_d (and the multi-ABD wrapper) call. Including the helper
 * header directly would leave the un-renamed name and clash with the facade. */
#include "rocke/instance_gemm_multi_d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.fuse.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"

/* ===================================================================== *
 *  _MultiDEpilogue
 * ===================================================================== *
 *
 * The _MultiDEpilogue class (its from_ops classification + the apply_vec op
 * sequence) lives in the fuse port next to its FusedEpilogue base:
 * helper_rocke.helpers.fuse.{h,c} (rocke_multi_d_epilogue_t, rocke_mde_from_ops,
 * rocke_mde_apply_vec). This module composes one in
 * rocke_gemm_multi_d_build_fused_epilogue below and the universal cshuffle
 * epilogue dispatches apply_vec to it. Both the previous in-file copy and the
 * never-implemented rocke_fuse_* shim it called are gone. */

/* ===================================================================== *
 *  GemmMultiDSpec
 * ===================================================================== */
rocke_gemm_multi_d_spec_t rocke_gemm_multi_d_spec_default(void)
{
    rocke_gemm_multi_d_spec_t s;
    memset(&s, 0, sizeof(s));
    s.base = rocke_gemm_universal_spec_default();
    s.num_d_operands = 0;
    s.d_dtype = "fp16";
    s.name = "rocke_gemm_multi_d";
    s.d_load_kind = ROCKE_D_LOAD_VECTOR;
    return s;
}

/* object.__setattr__(spec, "_fused_epilogue", ep): drop the composed
 * FusedEpilogue into the universal spec side-channel the cshuffle epilogue reads
 * via getattr(spec, "_fused_epilogue", None). `is_mde` records whether `ep` is
 * the base sub-object of a _MultiDEpilogue (so apply_vec dispatches to the
 * optimised override) or a plain FusedEpilogue (stock). */
void rocke_gemm_universal_spec_set_fused_epilogue(rocke_gemm_universal_spec_t* spec,
                                                  rocke_fused_epilogue_t* ep,
                                                  bool is_mde)
{
    if(spec == NULL)
    {
        return;
    }
    spec->_fused_epilogue = ep;
    spec->_fused_epilogue_is_mde = (ep != NULL) ? is_mde : false;
}

/* GemmMultiDSpec.kernel_name():
 *   d_suffix = "_".join(f"{name}{op}" for name, op in self.d_operands)
 *   return kernel_name_join(self.name, self.base.kernel_name(),
 *                           f"md{self.num_d}", d_suffix, self.d_dtype) */
rocke_status_t
    rocke_gemm_multi_d_kernel_name(const rocke_gemm_multi_d_spec_t* spec, char* out, size_t out_cap)
{
    char base_name[256];
    char md_part[32];
    char d_suffix[ROCKE_GEMM_MULTI_D_MAX_D * 64];
    const char* parts[4];
    rocke_status_t st;
    size_t i;
    size_t pos;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_gemm_universal_kernel_name(&spec->base, base_name, sizeof(base_name));
    if(st != ROCKE_OK)
    {
        return st;
    }

    snprintf(md_part, sizeof(md_part), "md%zu", spec->num_d_operands);

    /* d_suffix = "_".join(f"{name}{op}" ...) -- "" when no D operands. */
    pos = 0;
    d_suffix[0] = '\0';
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        const char* opname = spec->d_operands[i].op_is_mul ? "mul" : "add";
        int wrote;
        if(i > 0)
        {
            if(pos + 1 < sizeof(d_suffix))
            {
                d_suffix[pos++] = '_';
                d_suffix[pos] = '\0';
            }
            else
            {
                return ROCKE_ERR_VALUE;
            }
        }
        wrote = snprintf(
            d_suffix + pos, sizeof(d_suffix) - pos, "%s%s", spec->d_operands[i].param_name, opname);
        if(wrote < 0 || (size_t)wrote >= sizeof(d_suffix) - pos)
        {
            return ROCKE_ERR_VALUE;
        }
        pos += (size_t)wrote;
    }

    parts[0] = base_name;
    parts[1] = md_part;
    parts[2] = d_suffix; /* empty string is skipped by kernel_name_join */
    parts[3] = spec->d_dtype;

    /* kernel_name_join(prefix=self.name, *parts) -- no flags. */
    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */
bool rocke_gemm_multi_d_is_valid_spec(const rocke_gemm_multi_d_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    size_t i;
    size_t j;
    char base_reason[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "spec is NULL");
        }
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* if spec.num_d == 0: return False, "d_operands must contain at least one entry" */
    if(spec->num_d_operands == 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "d_operands must contain at least one entry");
        }
        return false;
    }
    /* if spec.num_d > MAX_D: return False, f"num_d {n} > MAX_D {MAX_D}" */
    if(spec->num_d_operands > ROCKE_GEMM_MULTI_D_MAX_D)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "num_d %zu > MAX_D %d",
                     spec->num_d_operands,
                     ROCKE_GEMM_MULTI_D_MAX_D);
        }
        return false;
    }
    /* if spec.base.trait.epilogue != "cshuffle": reject. */
    if(spec->base.trait.epilogue == NULL || strcmp(spec->base.trait.epilogue, "cshuffle") != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "GemmMultiD requires base.trait.epilogue='cshuffle' "
                     "(the default epilogue doesn't have the fused-op hook); "
                     "got '%s'",
                     spec->base.trait.epilogue ? spec->base.trait.epilogue : "");
        }
        return false;
    }

    /* names = set(); for name, op in spec.d_operands: ... */
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        const rocke_gemm_multi_d_op_t* d = &spec->d_operands[i];
        const char* name = d->param_name;

        /* if op not in ("add", "mul"): reject -- the bool field cannot encode
         * a third value, so this branch can never trip in C; preserved for
         * structural parity only. (Validation of op-string membership happens
         * at construction.) */

        /* if not name: return False, "D param_name must be a non-empty string" */
        if(name == NULL || name[0] == '\0')
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason, reason_cap, "D param_name must be a non-empty string");
            }
            return false;
        }
        /* if name in names: return False, f"duplicate D param_name {name!r}" */
        for(j = 0; j < i; ++j)
        {
            if(spec->d_operands[j].param_name != NULL
               && strcmp(spec->d_operands[j].param_name, name) == 0)
            {
                if(reason != NULL && reason_cap > 0)
                {
                    snprintf(reason, reason_cap, "duplicate D param_name '%s'", name);
                }
                return false;
            }
        }
        /* if name in ("A","B","C","M","N","K"): reject (reserved). */
        if(strcmp(name, "A") == 0 || strcmp(name, "B") == 0 || strcmp(name, "C") == 0
           || strcmp(name, "M") == 0 || strcmp(name, "N") == 0 || strcmp(name, "K") == 0)
        {
            if(reason != NULL && reason_cap > 0)
            {
                snprintf(reason,
                         reason_cap,
                         "D param_name '%s' collides with a reserved GEMM kernel parameter",
                         name);
            }
            return false;
        }
    }

    /* ok_base, why_base = is_valid_spec(spec.base, arch=arch)
     * if not ok_base: return False, f"base GEMM spec invalid: {why_base}" */
    if(!rocke_gemm_universal_is_valid_spec(&spec->base, arch, base_reason, sizeof(base_reason)))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "base GEMM spec invalid: %s", base_reason);
        }
        return false;
    }

    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

/* ===================================================================== *
 *  _build_fused_epilogue
 * ===================================================================== */
rocke_fused_epilogue_t* rocke_gemm_multi_d_build_fused_epilogue(
    rocke_arena_t* arena, const rocke_gemm_multi_d_spec_t* spec, bool* out_is_mde)
{
    const rocke_type_t* d_ir_dtype;
    rocke_epilogue_op_t* ops; /* arena-owned op chain; the epilogue borrows it */
    size_t i;
    size_t num;

    if(out_is_mde != NULL)
    {
        *out_is_mde = false;
    }
    if(arena == NULL || spec == NULL)
    {
        return NULL;
    }
    num = spec->num_d_operands;

    /* d_ir_dtype = dtype_to_ir(spec.d_dtype) */
    d_ir_dtype = rocke_fuse_dtype_to_ir_str(spec->d_dtype);
    if(d_ir_dtype == NULL)
    {
        return NULL; /* Python ValueError (unsupported epilogue dtype) */
    }

    /* ops = []; for name, op in spec.d_operands:
     *     if op == "add": ops.append(ResidualAdd(param_name=name, dtype=d_ir_dtype))
     *     elif op == "mul": ops.append(ResidualMul(param_name=name, dtype=d_ir_dtype))
     *
     * The fuse ResidualAdd/Mul are value constructors; the chain (and the
     * epilogue object that borrows it) must outlive the build, so both live in
     * the arena the spec side-channel pointer references. */
    ops = (rocke_epilogue_op_t*)rocke_arena_alloc(arena,
                                                  sizeof(rocke_epilogue_op_t) * (num ? num : 1));
    if(ops == NULL)
    {
        return NULL;
    }
    for(i = 0; i < num; ++i)
    {
        const rocke_gemm_multi_d_op_t* d = &spec->d_operands[i];
        if(d->op_is_mul)
        {
            ops[i] = rocke_residual_mul(d->param_name, d_ir_dtype);
        }
        else
        {
            ops[i] = rocke_residual_add(d->param_name, d_ir_dtype);
        }
    }

    /* if spec.d_load_kind == "stock":
     *     return FusedEpilogue(ops=tuple(ops), dtype=d_ir_dtype) */
    if(spec->d_load_kind == ROCKE_D_LOAD_STOCK)
    {
        rocke_fused_epilogue_t* fe = (rocke_fused_epilogue_t*)rocke_arena_alloc(arena, sizeof(*fe));
        if(fe == NULL)
        {
            return NULL;
        }
        rocke_fe_init(fe, ops, num, d_ir_dtype);
        return fe; /* out_is_mde stays false */
    }

    /* if spec.d_load_kind == "tiled":
     *     return _MultiDEpilogue.from_ops(ops, dtype=d_ir_dtype, load_kind="tiled")
     * else (vector / unrecognised):
     *     return _MultiDEpilogue.from_ops(ops, dtype=d_ir_dtype, load_kind="vector") */
    {
        rocke_multi_d_epilogue_t* md
            = (rocke_multi_d_epilogue_t*)rocke_arena_alloc(arena, sizeof(*md));
        rocke_mde_load_kind_t lk
            = (spec->d_load_kind == ROCKE_D_LOAD_TILED) ? ROCKE_MDE_TILED : ROCKE_MDE_VECTOR;
        if(md == NULL)
        {
            return NULL;
        }
        if(rocke_mde_from_ops(md, ops, num, d_ir_dtype, lk) != ROCKE_OK)
        {
            return NULL;
        }
        if(out_is_mde != NULL)
        {
            *out_is_mde = true;
        }
        /* The base sub-object IS the FusedEpilogue the spec attaches; its first
         * member, so &md->base == (rocke_fused_epilogue_t*)md. */
        return &md->base;
    }
}

/* ===================================================================== *
 *  build_gemm_multi_d
 * ===================================================================== */
/* The full-port 4-arg builder, defined under its real symbol name. The facade
 * (instance_gemm_multi_d.h) #undef's the rocke_build_gemm_multi_d rename after the
 * helper-header include, so at this definition point the public 2-arg
 * rocke_build_gemm_multi_d name belongs to the facade; the helper's 4-arg seam
 * therefore spells out rocke_build_gemm_multi_d_builder directly (the symbol the
 * facade's 2-arg entry and the multi-ABD wrapper call). */
rocke_kernel_def_t* rocke_build_gemm_multi_d_builder(rocke_ir_builder_t* b,
                                                     rocke_arena_t* arena,
                                                     const rocke_gemm_multi_d_spec_t* spec,
                                                     const char* arch)
{
    rocke_fused_epilogue_t* fused;
    bool fused_is_mde;
    rocke_gemm_universal_spec_t base_renamed;
    char* renamed_name;
    char name_buf[512];

    if(b == NULL || arena == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch=arch); if not ok: raise ValueError(...) */
    if(!rocke_gemm_multi_d_is_valid_spec(spec, arch, NULL, 0))
    {
        return NULL;
    }

    /* fused = _build_fused_epilogue(spec) */
    fused = rocke_gemm_multi_d_build_fused_epilogue(arena, spec, &fused_is_mde);
    if(fused == NULL)
    {
        return NULL;
    }

    /* base_renamed = dataclasses.replace(spec.base, name=spec.kernel_name())
     * -- a fresh copy of the base spec with the multi-D kernel name. */
    if(rocke_gemm_multi_d_kernel_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        return NULL;
    }
    renamed_name = rocke_arena_strdup(arena, name_buf);
    if(renamed_name == NULL)
    {
        return NULL;
    }
    base_renamed = spec->base;
    base_renamed.name = renamed_name;

    /* object.__setattr__(base_renamed, "_fused_epilogue", fused) */
    rocke_gemm_universal_spec_set_fused_epilogue(&base_renamed, fused, fused_is_mde);

    /* return build_universal_gemm(base_renamed, arch=arch) */
    return rocke_build_universal_gemm(b, &base_renamed, arch);
}

/* ===================================================================== *
 *  gemm_multi_d_signature
 * ===================================================================== */
rocke_status_t rocke_gemm_multi_d_signature(rocke_arena_t* arena,
                                            const rocke_gemm_multi_d_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    size_t i;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    /* sb = (SignatureBuilder()
     *       .ptr("A", spec.base.data.dtype_a)
     *       .ptr("B", spec.base.data.dtype_b)
     *       .ptr("C", spec.base.data.dtype_c)
     *       .scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")) */
    rocke_signature_builder_ptr(&sb, "A", spec->base.data.dtype_a, NULL);
    rocke_signature_builder_ptr(&sb, "B", spec->base.data.dtype_b, NULL);
    rocke_signature_builder_ptr(&sb, "C", spec->base.data.dtype_c, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");

    /* if spec.base.batched:
     *     sb.scalar("stride_a","i32").scalar("stride_b","i32").scalar("stride_c","i32") */
    if(spec->base.batched)
    {
        rocke_signature_builder_scalar(&sb, "stride_a", "i32");
        rocke_signature_builder_scalar(&sb, "stride_b", "i32");
        rocke_signature_builder_scalar(&sb, "stride_c", "i32");
    }

    /* for name, _op in spec.d_operands: sb.ptr(name, spec.d_dtype) */
    for(i = 0; i < spec->num_d_operands; ++i)
    {
        rocke_signature_builder_ptr(&sb, spec->d_operands[i].param_name, spec->d_dtype, NULL);
    }

    /* return sb.build() */
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  gemm_multi_d_grid
 * ===================================================================== */
rocke_status_t rocke_gemm_multi_d_grid(
    const rocke_gemm_multi_d_spec_t* spec, int m, int n, int batch, int out[3])
{
    const rocke_gemm_tile_spec_t* t;
    int z;
    int totals[3];
    int tiles[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* t = spec.base.tile
     * z = batch if spec.base.batched else 1
     * return ceil_div_grid((n, t.tile_n), (m, t.tile_m), (z, 1)) */
    t = &spec->base.tile;
    z = spec->base.batched ? batch : 1;

    totals[0] = n;
    tiles[0] = t->tile_n;
    totals[1] = m;
    tiles[1] = t->tile_m;
    totals[2] = z;
    tiles[2] = 1;

    return rocke_ceil_div_grid(totals, tiles, 3, out);
}
