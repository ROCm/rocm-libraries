// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_spec_helpers_and_glue.c.c --
 * PUBLIC ENTRY / GLUE bucket of the chunked C99 port of
 * ck_dsl/instances/common/moe_sorting.py.
 *
 * SCOPE (this TU only):
 *   spec surface          ckc_moe_sorting_spec_default / _total_pairs / _kernel_name
 *   validity gate         ckc_moe_sorting_is_valid_spec (public wrapper) +
 *                         ckc_moe_sort_is_valid_spec_impl (shared gate + wave_size)
 *   grid helpers          ckc_moe_sort_{histogram,scan,scatter,persistent}_grid
 *   signature builders    ckc_moe_sort_{histogram,scan,scatter,persistent}_signature
 *   workspace             ckc_moe_sorting_workspace_bytes
 *   shared module helpers ckc_moe_sort_decode_pair_token_topk,
 *                         ckc_moe_sort_decode_expert_load,
 *                         ckc_moe_sort_wave_kogge_stone_scan_i32
 *   public build entries  ckc_build_moe_sort_{histogram,scan,scatter,persistent}
 *                         (+ their _new init variants)
 *   lower convenience     ckc_build_moe_sort_*_lower_to_llvm
 *
 * The four build entries orchestrate: build ctx, populate inputs, then call the
 * matching prologue + phase functions (PEERS, declared in
 * ckc/instance_moe_sorting_internal.h, implemented in sibling TUs) in the exact
 * Python builder-call order; they return ctx->b->kernel.
 *
 * IR-free value/property helpers do not touch the builder; the three shared
 * module helpers + the build entries do, via the ckc_b_* surface, byte-faithful
 * to the Python op sequence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.core.arch.h"          /* ckc_archtarget_from_gfx, .wave_size */
#include "ckc/helper_ck_dsl.helpers.spec.h"       /* ckc_kernel_name_join, sig entry, grid */
#include "ckc/helper_ck_dsl.helpers.transforms.h" /* magic-division for pair decode */
#include "ckc/instance_moe_sorting.h"
#include "ckc/instance_moe_sorting_internal.h"
#include "ckc/lower_llvm.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  MoeSortingSpec value/property surface (IR-free).
 * ===================================================================== */

ckc_moe_sorting_spec_t ckc_moe_sorting_spec_default(void)
{
    ckc_moe_sorting_spec_t s;
    memset(&s, 0, sizeof(s));
    /* Required dims (tokens / topk / experts) have no Python default -> 0. */
    s.tokens  = 0;
    s.topk    = 0;
    s.experts = 0;
    /* @dataclass defaults. */
    s.block_size = 256;
    s.name       = "ck_dsl_moe_sorting";
    return s;
}

int ckc_moe_sorting_spec_total_pairs(const ckc_moe_sorting_spec_t* spec)
{
    /* @property total_pairs -> tokens * topk. */
    if(spec == NULL)
    {
        return 0;
    }
    return spec->tokens * spec->topk;
}

ckc_status_t ckc_moe_sorting_spec_kernel_name(const ckc_moe_sorting_spec_t* spec,
                                              const char* phase,
                                              char* out,
                                              size_t out_cap)
{
    char t_buf[32];
    char k_buf[32];
    char e_buf[32];
    char b_buf[32];
    const char* parts[5];

    if(spec == NULL || phase == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    /* kernel_name_join(name, phase, f"T{tokens}", f"K{topk}", f"E{experts}",
     *                  f"b{block_size}") */
    snprintf(t_buf, sizeof(t_buf), "T%d", spec->tokens);
    snprintf(k_buf, sizeof(k_buf), "K%d", spec->topk);
    snprintf(e_buf, sizeof(e_buf), "E%d", spec->experts);
    snprintf(b_buf, sizeof(b_buf), "b%d", spec->block_size);

    parts[0] = phase;
    parts[1] = t_buf;
    parts[2] = k_buf;
    parts[3] = e_buf;
    parts[4] = b_buf;

    return ckc_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

int ckc_moe_sorting_workspace_bytes(const ckc_moe_sorting_spec_t* spec)
{
    /* moe_sorting_workspace_bytes(spec) -> 4 * experts. */
    if(spec == NULL)
    {
        return 0;
    }
    return 4 * spec->experts;
}

/* ===================================================================== *
 *  Validity gate.
 *
 *  is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 *  Checks (in Python order): ArchTarget.from_gfx(arch) resolves; tokens / topk /
 *  experts > 0; experts <= 1024 (LDS scan cap); block_size in {64,128,256,512,
 *  1024}; experts <= block_size. The shared impl also resolves the ArchTarget
 *  wave size for the scan path-select.
 * ===================================================================== */

bool ckc_moe_sort_is_valid_spec_impl(const ckc_moe_sorting_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap,
                                     int* out_wave_size)
{
    const ckc_archtarget_t* target;

#define CK_MOE_SORT_REJECT(...)                        \
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
        CK_MOE_SORT_REJECT("spec is NULL");
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    /* try: ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e) */
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* TODO(port): reproduce the exact KeyError "...; known: [...]" suffix. */
        CK_MOE_SORT_REJECT("unknown gfx target '%s'", arch);
    }

    /* if tokens <= 0 or topk <= 0 or experts <= 0: ... */
    if(spec->tokens <= 0 || spec->topk <= 0 || spec->experts <= 0)
    {
        CK_MOE_SORT_REJECT("tokens / topk / experts must be > 0 (got %d, %d, %d)",
                           spec->tokens,
                           spec->topk,
                           spec->experts);
    }
    /* if experts > 1024: ... */
    if(spec->experts > 1024)
    {
        CK_MOE_SORT_REJECT("experts %d > 1024 (LDS scan cap)", spec->experts);
    }
    /* if block_size not in (64,128,256,512,1024): ... */
    if(spec->block_size != 64 && spec->block_size != 128 && spec->block_size != 256 &&
       spec->block_size != 512 && spec->block_size != 1024)
    {
        CK_MOE_SORT_REJECT("block_size %d not in {64..1024}", spec->block_size);
    }
    /* if experts > block_size: ... */
    if(spec->experts > spec->block_size)
    {
        CK_MOE_SORT_REJECT("experts (%d) > block_size (%d); pick a larger block_size or wait for "
                           "multi-pass scan",
                           spec->experts,
                           spec->block_size);
    }

    if(out_wave_size != NULL)
    {
        /* wave_size = ArchTarget.from_gfx(arch).wave_size */
        *out_wave_size = target->wave_size;
    }
    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;

#undef CK_MOE_SORT_REJECT
}

bool ckc_moe_sorting_is_valid_spec(const ckc_moe_sorting_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    /* Public wrapper: same gate, wave_size discarded. */
    return ckc_moe_sort_is_valid_spec_impl(spec, arch, reason, reason_cap, NULL);
}

/* ===================================================================== *
 *  GRID HELPERS.
 *    histogram / scatter : ceil_div_grid((total_pairs, block_size)).
 *    scan / persistent   : (1, 1, 1).
 * ===================================================================== */

static ckc_status_t ckc_i_moe_sort_pairs_grid(const ckc_moe_sorting_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];

    if(spec == NULL || out == NULL || spec->block_size <= 0)
    {
        return CKC_ERR_VALUE;
    }
    /* ceil_div_grid((spec.total_pairs, spec.block_size)). */
    totals[0] = spec->tokens * spec->topk;
    tiles[0]  = spec->block_size;
    return ckc_ceil_div_grid(totals, tiles, 1, out);
}

ckc_status_t ckc_moe_sort_histogram_grid(const ckc_moe_sorting_spec_t* spec, int out[3])
{
    return ckc_i_moe_sort_pairs_grid(spec, out);
}

ckc_status_t ckc_moe_sort_scatter_grid(const ckc_moe_sorting_spec_t* spec, int out[3])
{
    return ckc_i_moe_sort_pairs_grid(spec, out);
}

ckc_status_t ckc_moe_sort_scan_grid(const ckc_moe_sorting_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* return (1, 1, 1) */
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    return CKC_OK;
}

ckc_status_t ckc_moe_sort_persistent_grid(const ckc_moe_sorting_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* return (1, 1, 1) */
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  SIGNATURE (manifest) builders.
 *
 *  Each mirrors the Python SignatureBuilder().ptr(...).scalar(...).build()
 *  chain. ckc_sig_param / ckc_sig_scalar are exactly what the builder appends,
 *  so the emitted {name,type} sequence is byte-identical; the out[]/out_cap form
 *  matches this TU's public prototype.
 * ===================================================================== */

ckc_status_t ckc_moe_sort_histogram_signature(struct ckc_arena* arena,
                                              const ckc_moe_sorting_spec_t* spec,
                                              struct ckc_sig_entry* out,
                                              size_t out_cap,
                                              size_t* out_count)
{
    ckc_status_t st;

    (void)spec;
    if(arena == NULL || out == NULL || out_cap < 4)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_sig_param(arena, "TopkIds", "i32", NULL, &out[0]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "Hist", "i32", NULL, &out[1]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "num_pairs", "i32", &out[2]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "num_experts", "i32", &out[3]);
    if(st != CKC_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 4;
    }
    return CKC_OK;
}

ckc_status_t ckc_moe_sort_scan_signature(struct ckc_arena* arena,
                                         const ckc_moe_sorting_spec_t* spec,
                                         struct ckc_sig_entry* out,
                                         size_t out_cap,
                                         size_t* out_count)
{
    ckc_status_t st;

    (void)spec;
    if(arena == NULL || out == NULL || out_cap < 4)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_sig_param(arena, "Hist", "i32", NULL, &out[0]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "Offsets", "i32", NULL, &out[1]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "Counts", "i32", NULL, &out[2]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "num_experts", "i32", &out[3]);
    if(st != CKC_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 4;
    }
    return CKC_OK;
}

/* The 10-entry ABI shared by scatter + persistent (persistent is a superset). */
static ckc_status_t ckc_i_moe_sort_scatter_abi(struct ckc_arena* arena,
                                               struct ckc_sig_entry* out,
                                               size_t out_cap,
                                               size_t* out_count)
{
    ckc_status_t st;

    if(arena == NULL || out == NULL || out_cap < 10)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_sig_param(arena, "TopkIds", "i32", NULL, &out[0]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "TopkWeights", "f32", NULL, &out[1]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "Offsets", "i32", NULL, &out[2]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "Counter", "i32", NULL, &out[3]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "SortedTokenIds", "i32", NULL, &out[4]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "SortedTopkIds", "i32", NULL, &out[5]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_param(arena, "SortedWeights", "f32", NULL, &out[6]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "tokens", "i32", &out[7]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "topk", "i32", &out[8]);
    if(st != CKC_OK)
    {
        return st;
    }
    st = ckc_sig_scalar(arena, "num_experts", "i32", &out[9]);
    if(st != CKC_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 10;
    }
    return CKC_OK;
}

ckc_status_t ckc_moe_sort_scatter_signature(struct ckc_arena* arena,
                                            const ckc_moe_sorting_spec_t* spec,
                                            struct ckc_sig_entry* out,
                                            size_t out_cap,
                                            size_t* out_count)
{
    (void)spec;
    return ckc_i_moe_sort_scatter_abi(arena, out, out_cap, out_count);
}

ckc_status_t ckc_moe_sort_persistent_signature(struct ckc_arena* arena,
                                               const ckc_moe_sorting_spec_t* spec,
                                               struct ckc_sig_entry* out,
                                               size_t out_cap,
                                               size_t* out_count)
{
    /* Same 10 entries as scatter (superset ABI). */
    (void)spec;
    return ckc_i_moe_sort_scatter_abi(arena, out, out_cap, out_count);
}

/* ===================================================================== *
 *  SHARED MODULE HELPERS (module-level Python functions; emit IR).
 * ===================================================================== */

/* _decode_pair_token_topk(b, pair_idx, topk) -> (t_idx, k_idx).
 *
 *   split = UnmergeMagicDiv("pair", ("t_idx","k_idx"), dims=(1, topk))
 *   lowered = split.apply(b, {"pair": CoordVar("pair", pair_idx)})
 *   return lowered["t_idx"].value, lowered["k_idx"].value
 *
 * UnmergeMagicDiv.apply with dims=(1, topk), lowers=("t_idx","k_idx") (n=2):
 *   tmp = pair_idx
 *   i = 1:  d = topk
 *       if d == 1:  rem = const_i32(0); quot = tmp
 *       else:       quot = do_magic_division(b, tmp, mult, shift)
 *                   rem  = sub(tmp, mul(quot, const_i32(d)))
 *       k_idx = rem;  tmp = quot
 *   t_idx = tmp
 */
void ckc_moe_sort_decode_pair_token_topk(ckc_ir_builder_t* b,
                                         ckc_value_t* pair_idx,
                                         int topk,
                                         ckc_value_t** out_t_idx,
                                         ckc_value_t** out_k_idx)
{
    ckc_value_t* tmp;
    ckc_value_t* rem;
    ckc_value_t* quot;

    if(out_t_idx != NULL)
    {
        *out_t_idx = NULL;
    }
    if(out_k_idx != NULL)
    {
        *out_k_idx = NULL;
    }
    if(b == NULL || pair_idx == NULL)
    {
        return;
    }

    tmp = pair_idx;

    /* i = 1, d = dims[1] = topk. */
    if(topk == 1)
    {
        /* x // 1 == x, x % 1 == 0; no magic needed. */
        rem  = ckc_b_const_i32(b, 0);
        quot = tmp;
    }
    else
    {
        uint64_t mult = 0;
        int shift     = 0;
        if(!ckc_calculate_magic_numbers(b, topk, &mult, &shift))
        {
            return; /* builder error already set */
        }
        quot = ckc_do_magic_division(b, tmp, mult, shift);
        rem  = ckc_b_sub(b, tmp, ckc_b_mul(b, quot, ckc_b_const_i32(b, topk)));
    }
    /* k_idx = rem; tmp = quot. */
    tmp = quot;

    /* t_idx = tmp (the lower[0]). */
    if(out_t_idx != NULL)
    {
        *out_t_idx = tmp;
    }
    if(out_k_idx != NULL)
    {
        *out_k_idx = rem;
    }
}

/* _decode_expert_load(b, TopkIds, pair_idx, num_experts) -> (eid, valid_e).
 *
 *   eid = b.global_load_i32(TopkIds, pair_idx)
 *   valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)), b.cmp_lt(eid, num_experts))
 *
 * Op order: load -> const(0) -> ge -> lt -> land. */
void ckc_moe_sort_decode_expert_load(ckc_ir_builder_t* b,
                                     ckc_value_t* TopkIds,
                                     ckc_value_t* pair_idx,
                                     ckc_value_t* num_experts,
                                     ckc_value_t** out_eid,
                                     ckc_value_t** out_valid_e)
{
    ckc_value_t* eid;
    ckc_value_t* ge;
    ckc_value_t* lt;

    if(out_eid != NULL)
    {
        *out_eid = NULL;
    }
    if(out_valid_e != NULL)
    {
        *out_valid_e = NULL;
    }
    if(b == NULL)
    {
        return;
    }

    /* global_load_i32 has no explicit align in the Python call -> default. */
    eid = ckc_b_global_load_i32(b, TopkIds, pair_idx, 0);
    ge  = ckc_b_cmp_ge(b, eid, ckc_b_const_i32(b, 0));
    lt  = ckc_b_cmp_lt(b, eid, num_experts);

    if(out_eid != NULL)
    {
        *out_eid = eid;
    }
    if(out_valid_e != NULL)
    {
        *out_valid_e = ckc_b_land(b, ge, lt);
    }
}

/* _wave_kogge_stone_scan_i32(b, val, length=, lane_id=) -> inclusive.
 *
 *   cur = val; stride = 1
 *   while stride < length:
 *       c_stride  = const_i32(stride)
 *       do_add    = cmp_ge(lane_id, c_stride)
 *       src_lane  = select(do_add, sub(lane_id, c_stride), const_i32(0))
 *       addr      = shl(src_lane, const_i32(2))
 *       neighbour = ds_bpermute(addr, cur)
 *       cur       = select(do_add, add(cur, neighbour), cur)
 *       stride   *= 2
 *   return cur
 */
ckc_value_t* ckc_moe_sort_wave_kogge_stone_scan_i32(ckc_ir_builder_t* b,
                                                    ckc_value_t* val,
                                                    int length,
                                                    ckc_value_t* lane_id)
{
    ckc_value_t* cur;
    int stride;

    if(b == NULL || val == NULL || lane_id == NULL)
    {
        return val;
    }

    cur = val;
    for(stride = 1; stride < length; stride *= 2)
    {
        ckc_value_t* c_stride = ckc_b_const_i32(b, stride);
        ckc_value_t* do_add   = ckc_b_cmp_ge(b, lane_id, c_stride);
        /* Python evaluates select() args left-to-right: b.sub(...) emits its
         * SSA temp BEFORE b.const_i32(0). C leaves argument evaluation order
         * unspecified, so hoist the sub into its own statement to pin the
         * sub-then-const ordering and keep SSA value ids byte-identical
         * (otherwise the sub temp drifts +1, e.g. %sub11 -> %sub12). */
        ckc_value_t* src_sub   = ckc_b_sub(b, lane_id, c_stride);
        ckc_value_t* src_lane  = ckc_b_select(b, do_add, src_sub, ckc_b_const_i32(b, 0));
        ckc_value_t* addr      = ckc_b_shl(b, src_lane, ckc_b_const_i32(b, 2));
        ckc_value_t* neighbour = ckc_b_ds_bpermute(b, addr, cur);
        /* Same left-to-right pin for the add temp inside the merge select. */
        ckc_value_t* merged = ckc_b_add(b, cur, neighbour);
        cur                 = ckc_b_select(b, do_add, merged, cur);
    }
    return cur;
}

/* ===================================================================== *
 *  BUILD ENTRIES.
 *
 *  Each seeds the shared ctx with the inputs the prologue reads, runs the
 *  matching prologue (the is_valid_spec gate + params + decode) then the phase
 *  functions in Python order, and returns the kernel the last phase built.
 *  The prologue + phase functions are PEERS (sibling TUs).
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_moe_sort_histogram(ckc_ir_builder_t* b,
                                               const ckc_moe_sorting_spec_t* spec,
                                               const char* arch)
{
    ckc_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: is_valid_spec gate, max_workgroup_size, BS/E, params, tid/bid,
     * pair_idx = bid*BS + tid. (lines 224-243) */
    if(!ckc_moe_sort_hist_prologue(&ctx))
    {
        return NULL;
    }

    /* stage 1: per-block LDS histogram. (lines 245-258) */
    ckc_moe_sort_hist_block_histogram(&ctx);

    /* stage 2 + return: merge LDS bins to global Hist. (lines 260-272) */
    return ckc_moe_sort_hist_merge_to_global(&ctx);
}

ckc_kernel_def_t*
ckc_build_moe_sort_scan(ckc_ir_builder_t* b, const ckc_moe_sorting_spec_t* spec, const char* arch)
{
    ckc_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, resolve wave_size, max_workgroup_size, BS/E, params, tid,
     * c_E + in_bounds. (lines 363-384) */
    if(!ckc_moe_sort_scan_prologue(&ctx))
    {
        return NULL;
    }

    /* Path select on E <= wave_size (Python `if E <= wave_size:` line 386). */
    if(ctx.E <= ctx.wave_size)
    {
        return ckc_moe_sort_scan_wave_path(&ctx);
    }
    return ckc_moe_sort_scan_lds_path(&ctx);
}

ckc_kernel_def_t* ckc_build_moe_sort_scatter(ckc_ir_builder_t* b,
                                             const ckc_moe_sorting_spec_t* spec,
                                             const char* arch)
{
    ckc_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, max_workgroup_size, 10-entry ABI params, tid/bid,
     * pair_idx, _decode_pair_token_topk -> t_idx/k_idx, num_pairs, in_bounds.
     * (lines 481-525) */
    if(!ckc_moe_sort_scatter_prologue(&ctx))
    {
        return NULL;
    }

    /* scatter body + return. (lines 527-540) */
    return ckc_moe_sort_scatter_body(&ctx);
}

ckc_kernel_def_t* ckc_build_moe_sort_persistent(ckc_ir_builder_t* b,
                                                const ckc_moe_sorting_spec_t* spec,
                                                const char* arch)
{
    ckc_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b    = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, max_workgroup_size, BS/E/NP/n_pairs_per_thread, 10-entry
     * ABI params, tid, c_one/c_zero/c_E/c_BS/c_NP. (lines 639-682) */
    if(!ckc_moe_sort_persistent_prologue(&ctx))
    {
        return NULL;
    }

    /* phase 1: LDS histogram + write Counts. (lines 684-704) */
    ckc_moe_sort_persistent_histogram(&ctx);

    /* phase 2: in-place exclusive scan + write Offsets. (lines 706-713) */
    ckc_moe_sort_persistent_scan(&ctx);

    /* phase 3 + return: LDS scatter. (lines 715-741) */
    return ckc_moe_sort_persistent_scatter(&ctx);
}

/* ----- _new convenience init variants: init `b` with kernel_name(phase). ----- */

ckc_kernel_def_t* ckc_build_moe_sort_histogram_new(ckc_ir_builder_t* b,
                                                   const ckc_moe_sorting_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_sorting_spec_kernel_name(spec, "hist", name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_sort_histogram(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_sort_scan_new(ckc_ir_builder_t* b,
                                              const ckc_moe_sorting_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_sorting_spec_kernel_name(spec, "scan", name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_sort_scan(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_sort_scatter_new(ckc_ir_builder_t* b,
                                                 const ckc_moe_sorting_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_sorting_spec_kernel_name(spec, "scatter", name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_sort_scatter(b, spec, arch);
    });
}

ckc_kernel_def_t* ckc_build_moe_sort_persistent_new(ckc_ir_builder_t* b,
                                                    const ckc_moe_sorting_spec_t* spec,
                                                    const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_moe_sorting_spec_kernel_name(spec, "persistent", name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_sort_persistent(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM CONVENIENCE.
 *  build -> lower to LLVM .ll text. Each owns and frees its own IRBuilder.
 * ===================================================================== */

static void ckc_i_moe_sort_set_err(char* err, size_t err_cap, const char* msg)
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

/* Build via `build_new` (own builder), lower to LLVM, free the builder. */
static ckc_status_t ckc_i_moe_sort_lower(
    ckc_kernel_def_t* (*build_new)(ckc_ir_builder_t*, const ckc_moe_sorting_spec_t*, const char*),
    const char* build_fail_msg,
    const ckc_moe_sorting_spec_t* spec,
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
        ckc_i_moe_sort_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = build_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st            = ckc_ir_builder_status(&b);
        ckc_i_moe_sort_set_err(err, err_cap, (m != NULL && m[0] != '\0') ? m : build_fail_msg);
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_build_moe_sort_histogram_lower_to_llvm(const ckc_moe_sorting_spec_t* spec,
                                                        const char* arch,
                                                        ckc_llvm_flavor_t flavor,
                                                        char** out_ll,
                                                        char* err,
                                                        size_t err_cap)
{
    return ckc_i_moe_sort_lower(ckc_build_moe_sort_histogram_new,
                                "build_moe_sort_histogram failed",
                                spec,
                                arch,
                                flavor,
                                out_ll,
                                err,
                                err_cap);
}

ckc_status_t ckc_build_moe_sort_scan_lower_to_llvm(const ckc_moe_sorting_spec_t* spec,
                                                   const char* arch,
                                                   ckc_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    return ckc_i_moe_sort_lower(ckc_build_moe_sort_scan_new,
                                "build_moe_sort_scan failed",
                                spec,
                                arch,
                                flavor,
                                out_ll,
                                err,
                                err_cap);
}

ckc_status_t ckc_build_moe_sort_scatter_lower_to_llvm(const ckc_moe_sorting_spec_t* spec,
                                                      const char* arch,
                                                      ckc_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap)
{
    return ckc_i_moe_sort_lower(ckc_build_moe_sort_scatter_new,
                                "build_moe_sort_scatter failed",
                                spec,
                                arch,
                                flavor,
                                out_ll,
                                err,
                                err_cap);
}

ckc_status_t ckc_build_moe_sort_persistent_lower_to_llvm(const ckc_moe_sorting_spec_t* spec,
                                                         const char* arch,
                                                         ckc_llvm_flavor_t flavor,
                                                         char** out_ll,
                                                         char* err,
                                                         size_t err_cap)
{
    return ckc_i_moe_sort_lower(ckc_build_moe_sort_persistent_new,
                                "build_moe_sort_persistent failed",
                                spec,
                                arch,
                                flavor,
                                out_ll,
                                err,
                                err_cap);
}
