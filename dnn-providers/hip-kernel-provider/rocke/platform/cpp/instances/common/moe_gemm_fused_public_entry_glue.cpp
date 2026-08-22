// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_gemm_fused_public-entry-glue.c -- the PUBLIC ENTRY / GLUE bucket
 * of the chunked C99 port of rocke/instances/common/moe_gemm_fused.py.
 *
 * SCOPE (this TU only):
 *   - rocke_build_moe_gate_up_silu_gemm               (Python build_moe_gate_up_silu_gemm)
 *   - rocke_build_moe_interleaved_gate_up_silu_gemm   (Python build_moe_interleaved_...)
 *   - rocke_build_moe_down_reduce_gemm                (Python build_moe_down_reduce_gemm)
 *   - rocke_build_moe_down_silu_reduce_gemm           (Python build_moe_down_silu_reduce_gemm)
 *   - the four *_new init-from-spec convenience wrappers
 *   - the four *_lower_to_llvm convenience entries
 *   - the three *_signature pure functions
 *   - the six *_grid / *_grouped_grid pure functions
 *
 * These are the glue entries: each build entry constructs the shared per-family
 * context struct (rocke_moe_*_build_ctx_t), drives the prologue (ctx_init) and the
 * compute+epilogue phase function in the exact order the Python builder runs
 * them -- directly (batched) or inside the active-tile scf_if gate (grouped /
 * active_tile_skip) -- and returns b->kernel (NULL on a rejected spec / builder
 * error, with the sticky error set on `b`).
 *
 * Byte-identical builder-call sequence (mirrors the Python):
 *   gate+up   (build_moe_gate_up_silu_gemm, lines 710-909):
 *     ctx_init (validate via is_valid_gemm_spec; prologue lines 723-866)
 *     if grouped:  scf_if(expert_idx >= c0) { emit_compute }   (904-905)
 *     else:        emit_compute                                (907)
 *   interleaved (build_moe_interleaved_..., lines 1144-1391):
 *     ctx_init (validate + even-tile_n + do_work_cond; lines 1156-1349)
 *     if do_work_cond == NULL:  emit_compute                   (1386-1387)
 *     else:                     scf_if(do_work_cond){emit_compute} (1388-1390)
 *   down+reduce (build_moe_down_reduce_gemm, lines 1637-1821):
 *     ctx_init (validate; prologue lines 1650-1777)
 *     if grouped:  scf_if(expert_idx >= c0_dr){ emit_compute } (1815-1818)
 *     else:        emit_compute                                (1819-1820)
 *   down+silu+reduce (build_moe_down_silu_reduce_gemm, lines 2006-2031):
 *     convert FusedDownSiluReduceGemmSpec -> FusedDownReduceGemmSpec
 *     then delegate to build_moe_down_reduce_gemm
 *
 * The phase functions own all IR emission; this TU owns only ctx lifetime, the
 * field seeding the prologue reads, the scf_if gate scaffolding, and the call
 * ordering. The phases are peers declared in
 * rocke/instance_moe_gemm_fused_internal.h and implemented in sibling TUs.
 */
#include <stdlib.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.spec.h" /* SignatureBuilder / sig_entry */
#include "rocke/instance_moe_gemm_fused.h"
#include "rocke/instance_moe_gemm_fused_internal.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  GATE+UP+SILU  BUILD ENTRY   (Python build_moe_gate_up_silu_gemm)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_gate_up_silu_gemm(
    rocke_ir_builder_t* b, const rocke_moe_gate_up_silu_gemm_spec_t* spec, const char* arch)
{
    rocke_moe_gate_up_build_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    /* Zero the whole context so every unfilled handle/table slot starts NULL
     * (mirrors the Python locals being undefined until first assignment). */
    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 723-866): validate via is_valid_gemm_spec, decl params,
     * batched/grouped dispatch, smem + views + accumulators, _MoeKloopPlan +
     * operands. Returns false with the builder error set on a rejected spec. */
    if(!rocke_moe_gate_up_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* Compute + SiLU epilogue, optionally under the empty-tail gate.
     *   if grouped:  with b.scf_if(b.cmp_ge(expert_idx, c0)): _emit_gate_up_compute()
     *   else:        _emit_gate_up_compute()                       (lines 902-907) */
    if(ctx.grouped)
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
        rocke_b_region_enter(b, iff.then_region);
        rocke_moe_gate_up_emit_compute(&ctx);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_moe_gate_up_emit_compute(&ctx);
    }

    /* return b.kernel (NULL on a builder error) */
    return (rocke_ir_builder_status(b) == ROCKE_OK) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_moe_gate_up_silu_gemm_new(
    rocke_ir_builder_t* b, const rocke_moe_gate_up_silu_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(rocke_moe_gate_up_silu_gemm_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_gate_up_silu_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  INTERLEAVED GATE+UP+SILU BUILD ENTRY
 *  (Python build_moe_interleaved_gate_up_silu_gemm)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_interleaved_gate_up_silu_gemm(
    rocke_ir_builder_t* b,
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    rocke_moe_interleaved_build_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 1156-1349): validate, decl params (+ grouped /
     * active_tile_skip optionals), even-tile_n check, dispatch, smem + views +
     * accumulators, _MoeKloopPlan, the preshuffle-or-canonical operand, and the
     * do_work_cond gate. Returns false on reject / ValueError (incl. odd tile_n). */
    if(!rocke_moe_interleaved_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* if do_work_cond is None: emit_compute_and_epilogue()
     * else: with b.scf_if(do_work_cond): emit_compute_and_epilogue()
     *                                                       (lines 1386-1390) */
    if(ctx.do_work_cond == NULL)
    {
        rocke_moe_interleaved_emit_compute(&ctx);
    }
    else
    {
        rocke_if_t iff = rocke_b_scf_if(b, ctx.do_work_cond);
        rocke_b_region_enter(b, iff.then_region);
        rocke_moe_interleaved_emit_compute(&ctx);
        rocke_b_region_leave(b);
    }

    return (rocke_ir_builder_status(b) == ROCKE_OK) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_moe_interleaved_gate_up_silu_gemm_new(
    rocke_ir_builder_t* b,
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(spec, name, sizeof(name))
           != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_interleaved_gate_up_silu_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  DOWN+REDUCE BUILD ENTRY   (Python build_moe_down_reduce_gemm)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_down_reduce_gemm(rocke_ir_builder_t* b,
                                                     const rocke_moe_down_reduce_gemm_spec_t* spec,
                                                     const char* arch)
{
    rocke_moe_down_build_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 1650-1777): validate, decl params (+ grouped optional),
     * batched/grouped dispatch (incl. the bucket base), smem + views +
     * accumulators, _MoeKloopPlan + single operand. */
    if(!rocke_moe_down_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* if grouped: with b.scf_if(b.cmp_ge(expert_idx, c0_dr)): _emit_down_compute()
     * else:       _emit_down_compute()                       (lines 1815-1820) */
    if(ctx.grouped)
    {
        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_ge(b, ctx.expert_idx, ctx.c0_dr));
        rocke_b_region_enter(b, iff.then_region);
        rocke_moe_down_emit_compute(&ctx);
        rocke_b_region_leave(b);
    }
    else
    {
        rocke_moe_down_emit_compute(&ctx);
    }

    return (rocke_ir_builder_status(b) == ROCKE_OK) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_moe_down_reduce_gemm_new(
    rocke_ir_builder_t* b, const rocke_moe_down_reduce_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_down_reduce_gemm_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_down_reduce_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  DOWN+SILU+REDUCE BUILD ENTRY (P65 MVP wrapper)
 *  (Python build_moe_down_silu_reduce_gemm, lines 2006-2031)
 *
 *  Converts its FusedDownSiluReduceGemmSpec to a FusedDownReduceGemmSpec
 *  (name/tile/trait/wave_size/block_size carried; grouped defaults false,
 *  dtype defaults "fp16") and calls build_moe_down_reduce_gemm. The Python
 *  build_moe_down_reduce_gemm creates its OWN IRBuilder from the converted
 *  spec's kernel_name() (``..._down_reduce``), so for a byte-identical kernel
 *  we re-init `b` with the converted spec's kernel_name before delegating
 *  (mirroring the fresh ``b = IRBuilder(spec.kernel_name())`` in the callee).
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_moe_down_silu_reduce_gemm(
    rocke_ir_builder_t* b, const rocke_moe_down_silu_reduce_gemm_spec_t* spec, const char* arch)
{
    rocke_moe_down_reduce_gemm_spec_t dr;
    char name[256];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* FusedDownReduceGemmSpec(name=spec.name, tile=spec.tile, trait=spec.trait,
     *                         wave_size=spec.wave_size, block_size=spec.block_size)
     * grouped / dtype keep the FusedDownReduceGemmSpec dataclass defaults. */
    dr = rocke_moe_down_reduce_gemm_spec_default();
    dr.name = spec->name;
    dr.tile = spec->tile;
    dr.trait = spec->trait;
    dr.wave_size = spec->wave_size;
    dr.block_size = spec->block_size;
    /* dr.grouped stays false; dr.dtype stays "fp16" (dataclass defaults). */

    /* The callee builds a fresh IRBuilder from the converted spec's name; mirror
     * that here so the delegated build uses the ``..._down_reduce`` kernel name. */
    if(rocke_moe_down_reduce_gemm_spec_kernel_name(&dr, name, sizeof(name)) != ROCKE_OK)
    {
        return NULL;
    }
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
    {
        return NULL;
    }

    return rocke_build_moe_down_reduce_gemm(b, &dr, arch);
}

rocke_kernel_def_t* rocke_build_moe_down_silu_reduce_gemm_new(
    rocke_ir_builder_t* b, const rocke_moe_down_silu_reduce_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()); the build entry re-inits with the
         * converted down-reduce name, matching the Python fresh-builder delegation. */
        if(rocke_moe_down_silu_reduce_gemm_spec_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_down_silu_reduce_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  SIGNATURE DESCRIPTORS   (moe_*_gemm_signature)
 *
 *  Each builds the ordered SignatureBuilder param list into the arena, then
 *  copies the realised entries into the caller's (out, out_cap) array and the
 *  count into *out_count. Returns ROCKE_OK, or ROCKE_ERR_VALUE on a too-small
 *  buffer / NULL args (the entry strings are arena-owned).
 * ===================================================================== */

/* dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16". */
static const char* rocke_moe_sig_dt(const char* dtype)
{
    if(dtype != NULL
       && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0))
    {
        return dtype;
    }
    return "f16";
}

/* Copy `n` arena-owned entries into the caller buffer (cap out_cap) and publish
 * the count. ROCKE_ERR_VALUE if they do not fit. */
static rocke_status_t rocke_moe_sig_emit(const rocke_sig_entry_t* items,
                                         size_t n,
                                         rocke_sig_entry_t* out,
                                         size_t out_cap,
                                         size_t* out_count)
{
    size_t i;
    if(out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(n > out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    for(i = 0; i < n; ++i)
    {
        out[i] = items[i];
    }
    *out_count = n;
    return ROCKE_OK;
}

rocke_status_t rocke_moe_gate_up_silu_gemm_signature(const rocke_moe_gate_up_silu_gemm_spec_t* spec,
                                                     rocke_arena_t* arena,
                                                     rocke_sig_entry_t* out,
                                                     size_t out_cap,
                                                     size_t* out_count)
{
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    dt = rocke_moe_sig_dt(spec->dtype);

    if(rocke_signature_builder_init(&sb, arena) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    rocke_signature_builder_ptr(&sb, "A", dt, NULL);
    rocke_signature_builder_ptr(&sb, "WGate", dt, NULL);
    rocke_signature_builder_ptr(&sb, "WUp", dt, NULL);
    rocke_signature_builder_ptr(&sb, "Hidden", dt, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "stride_c", "i32");
    if(spec->grouped)
    {
        rocke_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }

    st = rocke_signature_builder_build(&sb, &items, &n);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return rocke_moe_sig_emit(items, n, out, out_cap, out_count);
}

rocke_status_t rocke_moe_interleaved_gate_up_silu_gemm_signature(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    rocke_arena_t* arena,
    rocke_sig_entry_t* out,
    size_t out_cap,
    size_t* out_count)
{
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    dt = rocke_moe_sig_dt(spec->dtype);

    if(rocke_signature_builder_init(&sb, arena) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    rocke_signature_builder_ptr(&sb, "A", dt, NULL);
    rocke_signature_builder_ptr(&sb, "WGateUp", dt, NULL);
    rocke_signature_builder_ptr(&sb, "Hidden", dt, NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "stride_c", "i32");
    /* if grouped: BlockExpertIds; elif active_tile_skip: SortedTokenIds + slot_size. */
    if(spec->grouped)
    {
        rocke_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }
    else if(spec->trait.active_tile_skip)
    {
        rocke_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
        rocke_signature_builder_scalar(&sb, "slot_size", "i32");
    }

    st = rocke_signature_builder_build(&sb, &items, &n);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return rocke_moe_sig_emit(items, n, out, out_cap, out_count);
}

rocke_status_t rocke_moe_down_reduce_gemm_signature(const rocke_moe_down_reduce_gemm_spec_t* spec,
                                                    rocke_arena_t* arena,
                                                    rocke_sig_entry_t* out,
                                                    size_t out_cap,
                                                    size_t* out_count)
{
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    dt = rocke_moe_sig_dt(spec->dtype);

    if(rocke_signature_builder_init(&sb, arena) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    rocke_signature_builder_ptr(&sb, "A", dt, NULL);
    rocke_signature_builder_ptr(&sb, "WDown", dt, NULL);
    rocke_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    rocke_signature_builder_ptr(&sb, "SortedWeights", "f32", NULL);
    rocke_signature_builder_ptr(&sb, "Y", "f32", NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");
    rocke_signature_builder_scalar(&sb, "stride_a", "i32");
    rocke_signature_builder_scalar(&sb, "stride_b", "i32");
    rocke_signature_builder_scalar(&sb, "slot_size", "i32");
    rocke_signature_builder_scalar(&sb, "tokens", "i32");
    if(spec->grouped)
    {
        rocke_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }

    st = rocke_signature_builder_build(&sb, &items, &n);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return rocke_moe_sig_emit(items, n, out, out_cap, out_count);
}

/* ===================================================================== *
 *  LAUNCH GRIDS   (moe_*_gemm_grid / *_grouped_grid)
 *
 *  Pure integer arithmetic over the spec tile geometry. The batched grids take
 *  (batch, m, n); the grouped grids take (num_m_blocks, n). The interleaved
 *  grids use GEMM N == 2*n.
 * ===================================================================== */

static int rocke_moe_ceil_div(int total, int tile)
{
    /* (total + tile - 1) // tile, matching the Python ceil division. */
    return (total + tile - 1) / tile;
}

void rocke_moe_gate_up_silu_gemm_grid(
    int batch, int m, int n, const rocke_moe_gate_up_silu_gemm_spec_t* spec, int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = rocke_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = rocke_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void rocke_moe_gate_up_silu_gemm_grouped_grid(int num_m_blocks,
                                              int n,
                                              const rocke_moe_gate_up_silu_gemm_spec_t* spec,
                                              int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), num_m_blocks, 1) */
    out_grid[0] = rocke_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

void rocke_moe_interleaved_gate_up_silu_gemm_grid(
    int batch,
    int m,
    int n,
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(2*n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = rocke_moe_ceil_div(2 * n, spec->tile.tile_n);
    out_grid[1] = rocke_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void rocke_moe_interleaved_gate_up_silu_gemm_grouped_grid(
    int num_m_blocks,
    int n,
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(2*n/tile_n), num_m_blocks, 1) */
    out_grid[0] = rocke_moe_ceil_div(2 * n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

void rocke_moe_down_reduce_gemm_grid(
    int batch, int m, int n, const rocke_moe_down_reduce_gemm_spec_t* spec, int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = rocke_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = rocke_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void rocke_moe_down_reduce_gemm_grouped_grid(int num_m_blocks,
                                             int n,
                                             const rocke_moe_down_reduce_gemm_spec_t* spec,
                                             int out_grid[3])
{
    if(spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), num_m_blocks, 1) */
    out_grid[0] = rocke_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Each owns and frees its own
 *  IRBuilder. On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into (err, err_cap), NUL-terminated and truncated to fit. */
static void rocke_moe_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

rocke_status_t rocke_moe_gate_up_silu_lower_to_llvm(const rocke_moe_gate_up_silu_gemm_spec_t* spec,
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
        rocke_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_moe_gate_up_silu_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_moe_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_moe_gate_up_silu_gemm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t rocke_moe_interleaved_gate_up_silu_lower_to_llvm(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
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
        rocke_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_moe_interleaved_gate_up_silu_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_moe_set_err(
            err,
            err_cap,
            (m != NULL && m[0] != '\0') ? m : "build_moe_interleaved_gate_up_silu_gemm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t rocke_moe_down_reduce_lower_to_llvm(const rocke_moe_down_reduce_gemm_spec_t* spec,
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
        rocke_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_moe_down_reduce_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_moe_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_moe_down_reduce_gemm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t
    rocke_moe_down_silu_reduce_lower_to_llvm(const rocke_moe_down_silu_reduce_gemm_spec_t* spec,
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
        rocke_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_moe_down_silu_reduce_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_moe_set_err(err,
                          err_cap,
                          (m != NULL && m[0] != '\0') ? m
                                                      : "build_moe_down_silu_reduce_gemm failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
