/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_moe_gemm_fused_public-entry-glue.c -- the PUBLIC ENTRY / GLUE bucket
 * of the chunked C99 port of ck_dsl/instances/common/moe_gemm_fused.py.
 *
 * SCOPE (this TU only):
 *   - ckc_build_moe_gate_up_silu_gemm               (Python build_moe_gate_up_silu_gemm)
 *   - ckc_build_moe_interleaved_gate_up_silu_gemm   (Python build_moe_interleaved_...)
 *   - ckc_build_moe_down_reduce_gemm                (Python build_moe_down_reduce_gemm)
 *   - ckc_build_moe_down_silu_reduce_gemm           (Python build_moe_down_silu_reduce_gemm)
 *   - the four *_new init-from-spec convenience wrappers
 *   - the four *_lower_to_llvm convenience entries
 *   - the three *_signature pure functions
 *   - the six *_grid / *_grouped_grid pure functions
 *
 * These are the glue entries: each build entry constructs the shared per-family
 * context struct (ckc_moe_*_build_ctx_t), drives the prologue (ctx_init) and the
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
 * ckc/instance_moe_gemm_fused_internal.h and implemented in sibling TUs.
 */
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_moe_gemm_fused.h"
#include "ckc/instance_moe_gemm_fused_internal.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* SignatureBuilder / sig_entry */
#include "ckc/lower_llvm.h"

/* ===================================================================== *
 *  GATE+UP+SILU  BUILD ENTRY   (Python build_moe_gate_up_silu_gemm)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_moe_gate_up_silu_gemm(
    ckc_ir_builder_t* b,
    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    ckc_moe_gate_up_build_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    /* Zero the whole context so every unfilled handle/table slot starts NULL
     * (mirrors the Python locals being undefined until first assignment). */
    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 723-866): validate via is_valid_gemm_spec, decl params,
     * batched/grouped dispatch, smem + views + accumulators, _MoeKloopPlan +
     * operands. Returns false with the builder error set on a rejected spec. */
    if (!ckc_moe_gate_up_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* Compute + SiLU epilogue, optionally under the empty-tail gate.
     *   if grouped:  with b.scf_if(b.cmp_ge(expert_idx, c0)): _emit_gate_up_compute()
     *   else:        _emit_gate_up_compute()                       (lines 902-907) */
    if (ctx.grouped)
    {
        ckc_if_t iff = ckc_b_scf_if(b, ckc_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
        ckc_b_region_enter(b, iff.then_region);
        ckc_moe_gate_up_emit_compute(&ctx);
        ckc_b_region_leave(b);
    }
    else
    {
        ckc_moe_gate_up_emit_compute(&ctx);
    }

    /* return b.kernel (NULL on a builder error) */
    return (ckc_ir_builder_status(b) == CKC_OK) ? b->kernel : NULL;
}

ckc_kernel_def_t* ckc_build_moe_gate_up_silu_gemm_new(
    ckc_ir_builder_t* b,
    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    /* b = IRBuilder(spec.kernel_name()) */
    if (ckc_moe_gate_up_silu_gemm_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_moe_gate_up_silu_gemm(b, spec, arch);
}

/* ===================================================================== *
 *  INTERLEAVED GATE+UP+SILU BUILD ENTRY
 *  (Python build_moe_interleaved_gate_up_silu_gemm)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_moe_interleaved_gate_up_silu_gemm(
    ckc_ir_builder_t* b,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    ckc_moe_interleaved_build_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 1156-1349): validate, decl params (+ grouped /
     * active_tile_skip optionals), even-tile_n check, dispatch, smem + views +
     * accumulators, _MoeKloopPlan, the preshuffle-or-canonical operand, and the
     * do_work_cond gate. Returns false on reject / ValueError (incl. odd tile_n). */
    if (!ckc_moe_interleaved_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* if do_work_cond is None: emit_compute_and_epilogue()
     * else: with b.scf_if(do_work_cond): emit_compute_and_epilogue()
     *                                                       (lines 1386-1390) */
    if (ctx.do_work_cond == NULL)
    {
        ckc_moe_interleaved_emit_compute(&ctx);
    }
    else
    {
        ckc_if_t iff = ckc_b_scf_if(b, ctx.do_work_cond);
        ckc_b_region_enter(b, iff.then_region);
        ckc_moe_interleaved_emit_compute(&ctx);
        ckc_b_region_leave(b);
    }

    return (ckc_ir_builder_status(b) == CKC_OK) ? b->kernel : NULL;
}

ckc_kernel_def_t* ckc_build_moe_interleaved_gate_up_silu_gemm_new(
    ckc_ir_builder_t* b,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(spec, name, sizeof(name))
        != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_moe_interleaved_gate_up_silu_gemm(b, spec, arch);
}

/* ===================================================================== *
 *  DOWN+REDUCE BUILD ENTRY   (Python build_moe_down_reduce_gemm)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_moe_down_reduce_gemm(
    ckc_ir_builder_t* b,
    const ckc_moe_down_reduce_gemm_spec_t* spec,
    const char* arch)
{
    ckc_moe_down_build_ctx_t ctx;

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));

    /* Prologue (lines 1650-1777): validate, decl params (+ grouped optional),
     * batched/grouped dispatch (incl. the bucket base), smem + views +
     * accumulators, _MoeKloopPlan + single operand. */
    if (!ckc_moe_down_build_ctx_init(&ctx, b, spec, arch))
    {
        return NULL;
    }

    /* if grouped: with b.scf_if(b.cmp_ge(expert_idx, c0_dr)): _emit_down_compute()
     * else:       _emit_down_compute()                       (lines 1815-1820) */
    if (ctx.grouped)
    {
        ckc_if_t iff = ckc_b_scf_if(b, ckc_b_cmp_ge(b, ctx.expert_idx, ctx.c0_dr));
        ckc_b_region_enter(b, iff.then_region);
        ckc_moe_down_emit_compute(&ctx);
        ckc_b_region_leave(b);
    }
    else
    {
        ckc_moe_down_emit_compute(&ctx);
    }

    return (ckc_ir_builder_status(b) == CKC_OK) ? b->kernel : NULL;
}

ckc_kernel_def_t* ckc_build_moe_down_reduce_gemm_new(
    ckc_ir_builder_t* b,
    const ckc_moe_down_reduce_gemm_spec_t* spec,
    const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_moe_down_reduce_gemm_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_moe_down_reduce_gemm(b, spec, arch);
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
ckc_kernel_def_t* ckc_build_moe_down_silu_reduce_gemm(
    ckc_ir_builder_t* b,
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec,
    const char* arch)
{
    ckc_moe_down_reduce_gemm_spec_t dr;
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* FusedDownReduceGemmSpec(name=spec.name, tile=spec.tile, trait=spec.trait,
     *                         wave_size=spec.wave_size, block_size=spec.block_size)
     * grouped / dtype keep the FusedDownReduceGemmSpec dataclass defaults. */
    dr = ckc_moe_down_reduce_gemm_spec_default();
    dr.name = spec->name;
    dr.tile = spec->tile;
    dr.trait = spec->trait;
    dr.wave_size = spec->wave_size;
    dr.block_size = spec->block_size;
    /* dr.grouped stays false; dr.dtype stays "fp16" (dataclass defaults). */

    /* The callee builds a fresh IRBuilder from the converted spec's name; mirror
     * that here so the delegated build uses the ``..._down_reduce`` kernel name. */
    if (ckc_moe_down_reduce_gemm_spec_kernel_name(&dr, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }

    return ckc_build_moe_down_reduce_gemm(b, &dr, arch);
}

ckc_kernel_def_t* ckc_build_moe_down_silu_reduce_gemm_new(
    ckc_ir_builder_t* b,
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec,
    const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    /* b = IRBuilder(spec.kernel_name()); the build entry re-inits with the
     * converted down-reduce name, matching the Python fresh-builder delegation. */
    if (ckc_moe_down_silu_reduce_gemm_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_moe_down_silu_reduce_gemm(b, spec, arch);
}

/* ===================================================================== *
 *  SIGNATURE DESCRIPTORS   (moe_*_gemm_signature)
 *
 *  Each builds the ordered SignatureBuilder param list into the arena, then
 *  copies the realised entries into the caller's (out, out_cap) array and the
 *  count into *out_count. Returns CKC_OK, or CKC_ERR_VALUE on a too-small
 *  buffer / NULL args (the entry strings are arena-owned).
 * ===================================================================== */

/* dt = spec.dtype if spec.dtype in ("f16","fp16","bf16") else "f16". */
static const char* ckc_moe_sig_dt(const char* dtype)
{
    if (dtype != NULL
        && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0
            || strcmp(dtype, "bf16") == 0))
    {
        return dtype;
    }
    return "f16";
}

/* Copy `n` arena-owned entries into the caller buffer (cap out_cap) and publish
 * the count. CKC_ERR_VALUE if they do not fit. */
static ckc_status_t ckc_moe_sig_emit(const ckc_sig_entry_t* items,
                                     size_t n,
                                     ckc_sig_entry_t* out,
                                     size_t out_cap,
                                     size_t* out_count)
{
    size_t i;
    if (out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    if (n > out_cap)
    {
        return CKC_ERR_VALUE;
    }
    for (i = 0; i < n; ++i)
    {
        out[i] = items[i];
    }
    *out_count = n;
    return CKC_OK;
}

ckc_status_t ckc_moe_gate_up_silu_gemm_signature(
    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
    ckc_arena_t* arena,
    ckc_sig_entry_t* out,
    size_t out_cap,
    size_t* out_count)
{
    ckc_signature_builder_t sb;
    const ckc_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    ckc_status_t st;

    if (spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    dt = ckc_moe_sig_dt(spec->dtype);

    if (ckc_signature_builder_init(&sb, arena) != CKC_OK)
    {
        return CKC_ERR_VALUE;
    }
    ckc_signature_builder_ptr(&sb, "A", dt, NULL);
    ckc_signature_builder_ptr(&sb, "WGate", dt, NULL);
    ckc_signature_builder_ptr(&sb, "WUp", dt, NULL);
    ckc_signature_builder_ptr(&sb, "Hidden", dt, NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b", "i32");
    ckc_signature_builder_scalar(&sb, "stride_c", "i32");
    if (spec->grouped)
    {
        ckc_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }

    st = ckc_signature_builder_build(&sb, &items, &n);
    if (st != CKC_OK)
    {
        return st;
    }
    return ckc_moe_sig_emit(items, n, out, out_cap, out_count);
}

ckc_status_t ckc_moe_interleaved_gate_up_silu_gemm_signature(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    ckc_arena_t* arena,
    ckc_sig_entry_t* out,
    size_t out_cap,
    size_t* out_count)
{
    ckc_signature_builder_t sb;
    const ckc_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    ckc_status_t st;

    if (spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    dt = ckc_moe_sig_dt(spec->dtype);

    if (ckc_signature_builder_init(&sb, arena) != CKC_OK)
    {
        return CKC_ERR_VALUE;
    }
    ckc_signature_builder_ptr(&sb, "A", dt, NULL);
    ckc_signature_builder_ptr(&sb, "WGateUp", dt, NULL);
    ckc_signature_builder_ptr(&sb, "Hidden", dt, NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b", "i32");
    ckc_signature_builder_scalar(&sb, "stride_c", "i32");
    /* if grouped: BlockExpertIds; elif active_tile_skip: SortedTokenIds + slot_size. */
    if (spec->grouped)
    {
        ckc_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }
    else if (spec->trait.active_tile_skip)
    {
        ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
        ckc_signature_builder_scalar(&sb, "slot_size", "i32");
    }

    st = ckc_signature_builder_build(&sb, &items, &n);
    if (st != CKC_OK)
    {
        return st;
    }
    return ckc_moe_sig_emit(items, n, out, out_cap, out_count);
}

ckc_status_t ckc_moe_down_reduce_gemm_signature(
    const ckc_moe_down_reduce_gemm_spec_t* spec,
    ckc_arena_t* arena,
    ckc_sig_entry_t* out,
    size_t out_cap,
    size_t* out_count)
{
    ckc_signature_builder_t sb;
    const ckc_sig_entry_t* items = NULL;
    size_t n = 0;
    const char* dt;
    ckc_status_t st;

    if (spec == NULL || arena == NULL || out == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }
    dt = ckc_moe_sig_dt(spec->dtype);

    if (ckc_signature_builder_init(&sb, arena) != CKC_OK)
    {
        return CKC_ERR_VALUE;
    }
    ckc_signature_builder_ptr(&sb, "A", dt, NULL);
    ckc_signature_builder_ptr(&sb, "WDown", dt, NULL);
    ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "SortedWeights", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "Y", "f32", NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b", "i32");
    ckc_signature_builder_scalar(&sb, "slot_size", "i32");
    ckc_signature_builder_scalar(&sb, "tokens", "i32");
    if (spec->grouped)
    {
        ckc_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    }

    st = ckc_signature_builder_build(&sb, &items, &n);
    if (st != CKC_OK)
    {
        return st;
    }
    return ckc_moe_sig_emit(items, n, out, out_cap, out_count);
}

/* ===================================================================== *
 *  LAUNCH GRIDS   (moe_*_gemm_grid / *_grouped_grid)
 *
 *  Pure integer arithmetic over the spec tile geometry. The batched grids take
 *  (batch, m, n); the grouped grids take (num_m_blocks, n). The interleaved
 *  grids use GEMM N == 2*n.
 * ===================================================================== */

static int ckc_moe_ceil_div(int total, int tile)
{
    /* (total + tile - 1) // tile, matching the Python ceil division. */
    return (total + tile - 1) / tile;
}

void ckc_moe_gate_up_silu_gemm_grid(int batch, int m, int n,
                                    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
                                    int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = ckc_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = ckc_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void ckc_moe_gate_up_silu_gemm_grouped_grid(
    int num_m_blocks, int n,
    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), num_m_blocks, 1) */
    out_grid[0] = ckc_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

void ckc_moe_interleaved_gate_up_silu_gemm_grid(
    int batch, int m, int n,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(2*n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = ckc_moe_ceil_div(2 * n, spec->tile.tile_n);
    out_grid[1] = ckc_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void ckc_moe_interleaved_gate_up_silu_gemm_grouped_grid(
    int num_m_blocks, int n,
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(2*n/tile_n), num_m_blocks, 1) */
    out_grid[0] = ckc_moe_ceil_div(2 * n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

void ckc_moe_down_reduce_gemm_grid(int batch, int m, int n,
                                   const ckc_moe_down_reduce_gemm_spec_t* spec,
                                   int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), ceil(m/tile_m), batch) */
    out_grid[0] = ckc_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = ckc_moe_ceil_div(m, spec->tile.tile_m);
    out_grid[2] = batch;
}

void ckc_moe_down_reduce_gemm_grouped_grid(
    int num_m_blocks, int n,
    const ckc_moe_down_reduce_gemm_spec_t* spec,
    int out_grid[3])
{
    if (spec == NULL || out_grid == NULL)
    {
        return;
    }
    /* (ceil(n/tile_n), num_m_blocks, 1) */
    out_grid[0] = ckc_moe_ceil_div(n, spec->tile.tile_n);
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 *
 *  Convenience: build -> lower to LLVM .ll text. Each owns and frees its own
 *  IRBuilder. On CKC_OK *out_ll receives a malloc'd NUL-terminated string the
 *  caller frees with free(); on failure it is left NULL and (if err != NULL,
 *  cap err_cap) a diagnostic is written.
 * ===================================================================== */

/* Copy `msg` into (err, err_cap), NUL-terminated and truncated to fit. */
static void ckc_moe_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if (err == NULL || err_cap == 0)
    {
        return;
    }
    if (msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if (n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

ckc_status_t ckc_moe_gate_up_silu_lower_to_llvm(
    const ckc_moe_gate_up_silu_gemm_spec_t* spec,
    const char* arch, ckc_llvm_flavor_t flavor,
    char** out_ll, char* err, size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_moe_gate_up_silu_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_set_err(err, err_cap,
                        (m != NULL && m[0] != '\0') ? m
                                                    : "build_moe_gate_up_silu_gemm failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_moe_interleaved_gate_up_silu_lower_to_llvm(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec,
    const char* arch, ckc_llvm_flavor_t flavor,
    char** out_ll, char* err, size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_moe_interleaved_gate_up_silu_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_set_err(err, err_cap,
                        (m != NULL && m[0] != '\0')
                            ? m
                            : "build_moe_interleaved_gate_up_silu_gemm failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_moe_down_reduce_lower_to_llvm(
    const ckc_moe_down_reduce_gemm_spec_t* spec,
    const char* arch, ckc_llvm_flavor_t flavor,
    char** out_ll, char* err, size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_moe_down_reduce_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_set_err(err, err_cap,
                        (m != NULL && m[0] != '\0') ? m
                                                    : "build_moe_down_reduce_gemm failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_moe_down_silu_reduce_lower_to_llvm(
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec,
    const char* arch, ckc_llvm_flavor_t flavor,
    char** out_ll, char* err, size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        ckc_moe_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_moe_down_silu_reduce_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_set_err(err, err_cap,
                        (m != NULL && m[0] != '\0')
                            ? m
                            : "build_moe_down_silu_reduce_gemm failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
